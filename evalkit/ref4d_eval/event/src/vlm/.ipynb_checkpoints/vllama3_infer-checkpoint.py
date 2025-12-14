# /root/autodl-tmp/event_eval/src/vlm/vllama3_infer.py
# -*- coding: utf-8 -*-
"""
Segment-wise event description with VideoLLaMA3
（最小改动：严格模式、提示词解析增强、提示词日志、语言后缀只在需要时追加）
"""
from __future__ import annotations
import os, math, json, time, argparse, logging, traceback, re
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import decord
    from decord import VideoReader, cpu as decord_cpu
    _HAVE_DECORD = True
except Exception:
    _HAVE_DECORD = False

try:
    import cv2
    _HAVE_OPENCV = True
except Exception:
    _HAVE_OPENCV = False

from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from event_eval.src.common.io import read_json, write_json, read_yaml, ensure_dir, set_random_seed, expand_path

LOGGER = logging.getLogger("event_eval.vlm.vllama3")
if not LOGGER.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    LOGGER.addHandler(h)
    LOGGER.setLevel(logging.INFO)


# -------------------------------
# Helpers
# -------------------------------

def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "fp16").lower()
    if s == "bf16": return torch.bfloat16
    if s in ("fp32", "float32"): return torch.float32
    return torch.float16

def _postprocess_caption(text: str) -> str:
    t = (text or "").strip()
    t = t.replace("<think>", "").replace("</think>", "")
    t = t.replace("```", " ").replace("\n", " ").replace("\r", " ")
    t = " ".join(t.split())
    if len(t) > 160:
        for stop in [". ", "! ", "? ", "。", "！", "？"]:
            k = t.find(stop)
            if k != -1 and k < 160:
                t = t[:k+1]
                break
        if len(t) > 160:
            t = t[:160].rstrip(" ,;")
    return t

def _strip_prompt_prefix(text: str, prompt_text: str) -> str:
    t = (text or "").strip()
    if prompt_text and t.lower().startswith(prompt_text.lower()):
        t = t[len(prompt_text):].lstrip(":：- ").strip()
    return t

def _expand_placeholders(s: str, cfg: Dict[str, Any]) -> str:
    data_root = cfg.get("paths", {}).get("data_root", "/root/autodl-tmp/event_eval/data")
    return s.replace("${paths.data_root}", str(data_root))

def _contains_chinese(s: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', s or ""))


# -------------------------------
# Video decode & frame sampling
# -------------------------------

def _resize_short_keep_ar(img: Image.Image, short: int) -> Image.Image:
    if not short or short <= 0: return img
    w, h = img.size
    if min(w, h) == short: return img
    if w <= h:
        new_w = short
        new_h = int(round(h * short / w))
    else:
        new_h = short
        new_w = int(round(w * short / h))
    return img.resize((new_w, new_h), resample=Image.BICUBIC)

def _sample_frame_indices_by_fps(start_s: float, end_s: float, fps_req: float,
                                 max_frames: int, fps_video: float, n_total: int) -> List[int]:
    duration = max(0.0, end_s - start_s)
    if duration <= 0:
        center = int(min(max(0, round(start_s * fps_video)), max(0, n_total - 1)))
        return [center]
    want = int(math.ceil(duration * max(0.0, fps_req)))
    want = max(1, min(max_frames, want))
    t = np.linspace(start_s, max(start_s, end_s - 1e-6), num=want, dtype=np.float64)
    idx = np.clip((t * fps_video).round().astype(np.int64), 0, max(0, n_total - 1))
    return idx.tolist()

def _frames_by_decord(video_path: str, s_abs: float, e_abs: float,
                      fps_req: float, max_frames: int, short_edge: int) -> List[Image.Image]:
    vr = VideoReader(video_path, ctx=decord_cpu(0))
    n_total = len(vr)
    try:
        fps_video = float(vr.get_avg_fps())
    except Exception:
        fps_video = 25.0
    fps_video = fps_video if fps_video > 0 else 25.0
    idxs = _sample_frame_indices_by_fps(s_abs, e_abs, fps_req, max_frames, fps_video, n_total)
    batch = vr.get_batch(idxs).asnumpy()  # (N,H,W,3), uint8
    images: List[Image.Image] = []
    for arr in batch:
        img = Image.fromarray(arr)  # RGB
        img = _resize_short_keep_ar(img, short_edge)
        images.append(img)
    if not images and n_total > 0:
        arr = vr[0].asnumpy()
        images = [_resize_short_keep_ar(Image.fromarray(arr), short_edge)]
    return images

def _frames_by_opencv(video_path: str, s_abs: float, e_abs: float,
                      fps_req: float, max_frames: int, short_edge: int) -> List[Image.Image]:
    if not _HAVE_OPENCV:
        raise RuntimeError("OpenCV not available and decord failed.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fps_video = fps_video if fps_video > 0 else 25.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idxs = _sample_frame_indices_by_fps(s_abs, e_abs, fps_req, max_frames, fps_video, n_total)
        images: List[Image.Image] = []
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None: continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = _resize_short_keep_ar(Image.fromarray(frame), short_edge)
            images.append(img)
        if not images and n_total > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if ok and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images = [_resize_short_keep_ar(Image.fromarray(frame), short_edge)]
        return images
    finally:
        cap.release()

def sample_frames(video_path: str, s_abs: float, e_abs: float,
                  backend: str, fps_req: float, max_frames: int, short_edge: int) -> List[Image.Image]:
    backend = (backend or "decord").lower()
    if backend == "decord" and _HAVE_DECORD:
        try:
            return _frames_by_decord(video_path, s_abs, e_abs, fps_req, max_frames, short_edge)
        except Exception as e:
            LOGGER.warning(f"decord failed, fallback to opencv. err={e}")
    return _frames_by_opencv(video_path, s_abs, e_abs, fps_req, max_frames, short_edge)


# -------------------------------
# Model loading
# -------------------------------

def _health_check_local_model_dir(model_dir: Path) -> None:
    must_exist = ["config.json", "processor_config.json"]
    missing = [x for x in must_exist if not (model_dir / x).exists()]
    has_weights = any(model_dir.glob("*.safetensors")) or \
                  (model_dir / "pytorch_model.bin").exists() or \
                  (model_dir / "pytorch_model.bin.index.json").exists()
    if missing or not has_weights:
        raise FileNotFoundError(
            f"[VLM] Local model dir incomplete: {model_dir}\n"
            f"Missing: {missing}; weights_present={has_weights}"
        )

def load_vlmodel(cfg: Dict[str, Any]):
    vcfg = cfg.get("vllama3", {}) or {}
    model_path = expand_path(vcfg.get("model", "DAMO-NLP-SG/VideoLLaMA3-7B"))
    device = (vcfg.get("device", "auto") or "auto").lower()
    dtype = _dtype_from_str(vcfg.get("dtype", "bf16"))
    attn_impl = vcfg.get("attn_impl", None)
    local_files_only = bool(vcfg.get("local_files_only", False))

    is_local_dir = Path(model_path).exists() and Path(model_path).is_dir()
    if is_local_dir:
        _health_check_local_model_dir(Path(model_path))

    device_map = None
    target_device = "cuda"
    if device == "auto":
        device_map = "auto"
    elif device == "cpu":
        target_device = "cpu"
    else:
        target_device = "cuda"

    # CPU 上强制 dtype=fp32
    if (target_device == "cpu" or (device == "auto" and not torch.cuda.is_available())) and dtype != torch.float32:
        LOGGER.info(f"[VLM] Forcing dtype=float32 on CPU (was {dtype}).")
        dtype = torch.float32

    LOGGER.info(
        f"Loading VLM: {model_path} (dtype={dtype}, device={device}, "
        f"device_map={device_map}, local_files_only={local_files_only})"
    )

    load_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=dtype,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=local_files_only)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False,
                                                  local_files_only=local_files_only)
        processor = {"tokenizer": tokenizer}

    if device_map is None:
        model.to(target_device)
    model.eval()

    seed = vcfg.get("gen", {}).get("seed", None)
    if seed is not None:
        set_random_seed(int(seed), deterministic_torch=True, quiet=True)

    return model, processor


# -------------------------------
# Prompt resolving (NEW)
# -------------------------------

def _resolve_prompt(cfg: Dict[str, Any]) -> tuple[str, str, bool]:
    """
    返回: (final_prompt_text, lang, strict)
    解析优先级：
      1) vllama3.prompt.template
      2) vllama3.prompt.text
      3) vllama3.prompt.templates[lang]  (当 template/text 缺失时)
    严格模式：vllama3.prompt.strict=true 时，不做任何追加/修饰。
    """
    vcfg = cfg.get("vllama3", {}) or {}
    pcfg = vcfg.get("prompt", {}) or {}

    lang = (pcfg.get("lang", "") or "").lower()
    strict = bool(pcfg.get("strict", False))

    prompt_text = str(pcfg.get("template", "") or pcfg.get("text", ""))
    if not prompt_text:
        # 支持 templates: { zh: "...", en: "..." }
        templates = pcfg.get("templates", {}) or {}
        if lang in templates:
            prompt_text = str(templates[lang])
        elif templates:
            # 任意取一个（尽量 en/zh）
            prompt_text = str(templates.get("en") or templates.get("zh") or next(iter(templates.values())))

    if not prompt_text:
        prompt_text = "Summarize the main event in one concise sentence (verb + key nouns)."

    # 语言后缀：仅当 strict=False 且 template 本身未显式规定语言时追加
    if not strict:
        if lang in ("zh", "cn", "chinese", "zh-cn", "zh_hans"):
            if not _contains_chinese(prompt_text) and "中文" not in prompt_text:
                prompt_text = f"{prompt_text} 请用简体中文回答。"
        elif lang in ("en", "english"):
            if ("中文" in prompt_text) or _contains_chinese(prompt_text):
                pass
            elif "English" not in prompt_text and "Answer in English" not in prompt_text:
                prompt_text = f"{prompt_text} Answer in English."

    return prompt_text, lang, strict


# -------------------------------
# Inference
# -------------------------------

def _build_messages(frames: List[Image.Image], prompt_text: str) -> List[Dict[str, Any]]:
    contents: List[Dict[str, Any]] = [{"type": "image"} for _ in frames]
    contents.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": contents}]

def _move_inputs_to_device(inputs: Dict[str, Any], device: str, dtype: torch.dtype):
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            if k in ("pixel_values", "video_pixel_values"):
                inputs[k] = v.to(device, dtype=dtype, non_blocking=True)
            elif k in ("input_ids", "attention_mask", "position_ids"):
                inputs[k] = v.to(device, non_blocking=True)
            elif ("grid" in k) or ("merge" in k):
                inputs[k] = v.to(device, non_blocking=True)
            else:
                inputs[k] = v.to(device, non_blocking=True)
        elif isinstance(v, (list, tuple)):
            if ("grid" in k) or ("merge" in k):
                try:
                    inputs[k] = torch.tensor(v, device=device)
                except Exception:
                    pass
    return inputs

@torch.inference_mode()
def infer_segment_text(
    model,
    processor,
    frames: List[Image.Image],
    prompt_text: str,
    gen_cfg: Dict[str, Any],
    device_choice: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    force_generate: bool = False,
) -> str:
    # 首选 chat（除非强制 generate）
    if hasattr(model, "chat") and not force_generate:
        try:
            resp, _ = model.chat(
                processor,
                images=frames,
                question=prompt_text,
                history=None,
                do_sample=bool(gen_cfg.get("do_sample", False)),
                temperature=float(gen_cfg.get("temperature", 0.2)),
                top_p=float(gen_cfg.get("top_p", 0.9)),
                max_new_tokens=int(gen_cfg.get("max_new_tokens", 32)),
            )
            return _postprocess_caption(resp)
        except Exception as e:
            LOGGER.warning(f"chat() failed, fallback to template+generate(). err={e}")

    # --- generate() 路径 ---
    tokenizer = None
    _processor = processor
    if isinstance(processor, dict) and "tokenizer" in processor:
        tokenizer = processor["tokenizer"]
        _processor = None

    if _processor is not None and hasattr(_processor, "apply_chat_template"):
        messages = _build_messages(frames, prompt_text)
        prompt_bos = _processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = _processor(text=prompt_bos, images=frames, return_tensors="pt")
        inputs = _move_inputs_to_device(inputs, device_choice, dtype)

        allowed = (
            "max_new_tokens","do_sample","temperature","top_p","top_k",
            "repetition_penalty","no_repeat_ngram_size","length_penalty",
            "num_beams","early_stopping","min_new_tokens"
        )
        gen_kwargs = {k: gen_cfg[k] for k in allowed if k in gen_cfg}
        # 类型规范化
        if "max_new_tokens" in gen_kwargs: gen_kwargs["max_new_tokens"] = int(gen_kwargs["max_new_tokens"])
        if "min_new_tokens" in gen_kwargs: gen_kwargs["min_new_tokens"] = int(gen_kwargs["min_new_tokens"])
        if "no_repeat_ngram_size" in gen_kwargs: gen_kwargs["no_repeat_ngram_size"] = int(gen_kwargs["no_repeat_ngram_size"])
        if "num_beams" in gen_kwargs: gen_kwargs["num_beams"] = int(gen_kwargs["num_beams"])
        if "length_penalty" in gen_kwargs: gen_kwargs["length_penalty"] = float(gen_kwargs["length_penalty"])
        if "repetition_penalty" in gen_kwargs: gen_kwargs["repetition_penalty"] = float(gen_kwargs["repetition_penalty"])
        if "temperature" in gen_kwargs: gen_kwargs["temperature"] = float(gen_kwargs["temperature"])
        if "top_p" in gen_kwargs: gen_kwargs["top_p"] = float(gen_kwargs["top_p"])

        out = model.generate(**inputs, use_cache=True, **gen_kwargs)
        tok = getattr(_processor, "tokenizer", None)
        text = tok.decode(out[0], skip_special_tokens=True) if tok is not None else ""
        text = _strip_prompt_prefix(text, prompt_bos)
        return _postprocess_caption(text)

    # 兜底
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device_choice)
    out = model.generate(input_ids=input_ids, max_new_tokens=int(gen_cfg.get("max_new_tokens", 32)))
    return _postprocess_caption(tokenizer.decode(out[0], skip_special_tokens=True))


# -------------------------------
# Runner & I/O
# -------------------------------

def _infer_side(events_path: str) -> str:
    p = str(events_path)
    if "/events/ref/" in p or p.endswith("/events/ref"):
        return "ref"
    if "/events/gen/" in p or p.endswith("/events/gen"):
        return "gen"
    return "ref"

def _infer_sample_id(events_path: str) -> str:
    name = Path(events_path).name
    if name.endswith(".events.json"):
        return name[:-len(".events.json")]
    return Path(events_path).stem

def _derive_out_path(cfg: Dict[str, Any], side: str, sample_id: str) -> Path:
    vcfg = cfg.get("vllama3", {}) or {}
    ecfg = vcfg.get("export", {}) or {}
    out_dir = _expand_placeholders(str(ecfg.get("out_dir", "${paths.data_root}/vlm")), cfg)
    pat = ecfg.get("fname_ref" if side == "ref" else "fname_gen", "{sample_id}.vlm.json")
    rel = pat.format(sample_id=sample_id)
    out_path = Path(out_dir) / rel
    ensure_dir(out_path.parent)
    return out_path

def run(video_path: str, events_json_path: str, out_json_path: Optional[str], cfg_path: str) -> Dict[str, Any]:
    cfg = read_yaml(cfg_path)
    vcfg = cfg.get("vllama3", {}) or {}

    model, processor = load_vlmodel(cfg)

    # decode cfg
    dc = vcfg.get("video_decode", {}) or {}
    backend = str(dc.get("backend", "decord"))
    fps_req = float(dc.get("fps", 1.0))
    max_frames = int(dc.get("max_frames", 180))
    short_edge = int(dc.get("short_edge", 384))

    # NEW: 解析提示词（严格按你的 YAML 优先）
    prompt_text, lang, strict = _resolve_prompt(cfg)
    gen_cfg = vcfg.get("gen", {}) or {}

    device = vcfg.get("device", "auto").lower()
    dtype = _dtype_from_str(vcfg.get("dtype", "bf16"))
    device_choice = "cuda" if device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
    force_generate = bool(gen_cfg.get("force_generate", False))

    events = read_json(events_json_path)
    if not isinstance(events, list):
        raise ValueError(f"events json must be a list: {events_json_path}")

    side = _infer_side(events_json_path)
    sample_id = _infer_sample_id(events_json_path)

    if out_json_path:
        out_path = Path(out_json_path)
        ensure_dir(out_path.parent)
    else:
        out_path = _derive_out_path(cfg, side, sample_id)

    # 仅打印一次，便于核对“最终生效的提示词”
    LOGGER.info(f"[Prompt] strict={strict} lang={lang} -> {prompt_text}")

    t0 = time.time()
    out_events: List[Dict[str, Any]] = []
    for idx, seg in enumerate(events):
        s_abs = float(seg.get("s_abs", 0.0))
        e_abs = float(seg.get("e_abs", 0.0))
        seg_id = str(seg.get("id", f"e{idx:03d}"))

        try:
            frames = sample_frames(
                video_path=video_path,
                s_abs=s_abs,
                e_abs=e_abs,
                backend=backend,
                fps_req=fps_req,
                max_frames=max_frames,
                short_edge=short_edge,
            )

            caption = infer_segment_text(
                model=model,
                processor=processor,
                frames=frames,
                prompt_text=prompt_text,
                gen_cfg=gen_cfg,
                device_choice=device_choice,
                dtype=dtype,
                force_generate=force_generate,
            )
        except Exception as e:
            LOGGER.warning(f"[{idx+1}/{len(events)}] seg={seg_id} failed: {e}")
            LOGGER.debug(traceback.format_exc())
            caption = ""

        seg_out = dict(seg)
        seg_out["text"] = _postprocess_caption(caption)
        out_events.append(seg_out)
        LOGGER.info(f"[{idx+1}/{len(events)}] seg={seg_id} frames={len(frames)} -> '{seg_out['text']}'")

    write_json(out_events, out_path, indent=2)
    LOGGER.info(f"Done: {out_path} (segments={len(out_events)}) in {time.time()-t0:.2f}s")
    return {"n_segments": len(out_events), "out": str(out_path)}

def parse_args():
    ap = argparse.ArgumentParser(description="VideoLLaMA3 segment captioning")
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--events", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--out", type=str, default=None, help="optional; override config export path")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(
        video_path=args.video,
        events_json_path=args.events,
        out_json_path=args.out,
        cfg_path=args.config,
    )
