# -*- coding: utf-8 -*-
"""
批量评测脚本 evaluate_bank.py（无示例库版 / 完整可运行）

功能要点：
1) 按规范化 stem 将视频与题库（*_scored.json / *_scord.json）匹配；
2) 不再使用 few-shot 示例库，提示词为纯规则说明；
3) 其它：GPU/精度/抽帧/日志等与原版一致。
"""

import os, re, json, math, argparse, warnings, inspect, csv, glob
from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
from PIL import Image

import torch
from transformers import AutoModel, AutoTokenizer

# ================= CLI =================
p = argparse.ArgumentParser()

# —— 批量入口（必须） ——
p.add_argument('--bank-dir',   required=True, help='*_scord.json / *_scored.json 所在目录（可含子目录）')
p.add_argument('--video-dir',  required=True, help='视频根目录（可含子目录）')
p.add_argument('--out-dir',    required=True, help='输出根目录（CSV 与 detail.json）')

# 模型与缓存
p.add_argument('--local-path', default='', help='本地模型目录，如 /root/autodl-tmp/aiv/models/openbmb__MiniCPM-V-4_5')
p.add_argument('--model-id',   default='openbmb/MiniCPM-V-4_5')
p.add_argument('--revision',   default='main')
p.add_argument('--cache-dir',  default='')
p.add_argument('--local-files-only', action='store_true')

# 设备/精度/注意力
p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
p.add_argument('--dtype',  default='bf16', choices=['bf16','fp16','fp32'])
p.add_argument('--disable-flash-sdp', dest='disable_flash_sdp', action='store_true')
p.add_argument('--force-math-sdp',    dest='force_math_sdp',    action='store_true')

# 抽帧/解码（整段）
p.add_argument('--fps',           type=int,   default=3,   help='全局抽帧目标 fps')
p.add_argument('--cap-frames',    type=int,   default=300, help='最多送入模型的总帧数上限（全局）')
p.add_argument('--resize-short',  type=int,   default=448, help='短边等比缩放到该尺寸（0=不缩放）')
p.add_argument('--decode-backend', default='auto', choices=['auto','cv2','decord'])

# 生成控制
p.add_argument('--max-new-tokens', type=int, default=256)
p.add_argument('--temperature',    type=float, default=0.0)
p.add_argument('--enable-thinking', action='store_true')
p.add_argument('--verbose', action='store_true')

# 其它
p.add_argument('--dump-per-item', action='store_true', help='保存逐题原始输出与解析到 detail.json')
p.add_argument('--no-dump-per-item', dest='dump-per-item', action='store_false')
p.set_defaults(dump_per_item=True)

args = p.parse_args()

# ================= 常量/工具 =================
TIME_SCALE = 0.1
MAX_NUM_FRAMES = 450
VIDEO_EXTS = {'.mp4', '.mkv', '.mov', '.avi', '.m4v', '.webm'}

def _dtype_of(device: str, choice: str):
    if device == 'cpu':
        return torch.float32
    return {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[choice]

def _dtype_kw(d):
    sig = inspect.signature(AutoModel.from_pretrained)
    return {"dtype": d} if "dtype" in sig.parameters else {"torch_dtype": d}

def _ensure_dir(path: str):
    os.makedirs(path or ".", exist_ok=True)

def _ensure_dir_for_file(file_path: str):
    _ensure_dir(os.path.dirname(file_path))

# ====== “规范化 stem”以匹配 _prompt（视频）与 _scord/_scored（题库） ======
_SUFFIXES = [
    r"_prompt",
    r"_scord", r"_scored",
    r"_vqa", r"_assertion", r"_assertions", r"_qa", r"_bank",
]
_suffix_re = re.compile("(" + "|".join(s + r"$" for s in _SUFFIXES) + ")", re.I)

def _norm_stem(s: str) -> str:
    x = (s or "").strip().lower()
    if not x:
        return x
    while True:
        y = _suffix_re.sub("", x)
        if y == x:
            return y
        x = y

# —— 类名提取
_CLASS_RE = re.compile(r'^(?P<cls>.+)_(single|multi)_[0-9]+$', re.I)

def _class_from_norm_stem(norm_stem: str, fallback: str) -> str:
    if not norm_stem:
        return fallback
    m = _CLASS_RE.match(norm_stem)
    if m:
        return m.group('cls').strip()
    return fallback

# ================== 视频IO ==================
def read_video_meta_and_backend(video_path: str, preference: str = 'auto'):
    cap = None; vr = None
    backend = None; fps = None; total = None; duration = None
    if preference in ('auto', 'cv2'):
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            duration = total / max(fps, 1e-6) if total > 0 else 0.0
            if total > 0:
                backend = 'cv2'
                return backend, (cap, None), fps, total, duration
        except Exception:
            pass
    if preference in ('auto', 'decord'):
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            total = len(vr)
            try:
                fps = float(vr.get_avg_fps())
            except Exception:
                fps = 25.0
            duration = total / max(fps, 1e-6)
            backend = 'decord'
            return backend, (None, vr), fps, total, duration
        except Exception:
            pass
    raise RuntimeError(f"No available video backend for {video_path} (tried cv2, decord)")

def _release_video_backend(meta: Tuple[str, Tuple[Any, Any], float, int, float]):
    """释放视频 IO 资源：cap.release() 或删除 VideoReader 实例。"""
    backend, handles, _, _, _ = meta
    if backend == 'cv2':
        cap, _ = handles
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
    else:
        _, vr = handles
        try:
            if vr is not None:
                # decord 没有显式 release，删除并触发 GC
                del vr
        except Exception:
            pass

def read_frames(backend: str, handles, idx_all: np.ndarray) -> List[Image.Image]:
    frames: List[Image.Image] = []
    if backend == 'cv2':
        cap, _ = handles
        import cv2
        if len(idx_all) == 0:
            return frames
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx_all[0]))
        except Exception:
            pass
        cur = int(idx_all[0])
        want = set(int(i) for i in idx_all.tolist())
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if cur in want:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
                if len(frames) >= len(idx_all):
                    break
            cur += 1
    else:
        _, vr = handles
        if len(idx_all) == 0:
            return frames
        arr = vr.get_batch(idx_all.tolist()).asnumpy()
        frames = [Image.fromarray(x.astype('uint8')).convert('RGB') for x in arr]
    return frames

def _temporal_ids(ts: np.ndarray) -> List[List[int]]:
    grid = np.round(ts / TIME_SCALE).astype(np.int32)
    n = len(grid)
    packing = min(3, max(1, int(math.ceil(n / MAX_NUM_FRAMES))))
    chunks = []
    for k in range(packing):
        part = grid[k::packing]
        chunks.append(part.tolist())
    return chunks

def encode_whole_video(meta, choose_fps: int, cap_frames: int,
                       resize_short: int) -> Tuple[List[Image.Image], List[List[int]], Tuple[float, float], int]:
    backend, handles, fps, total, duration = meta
    s, e = 0.0, duration
    clip_len = max(1e-6, e - s)
    want = int(min(cap_frames, round(clip_len * min(choose_fps, int(round(fps))))))
    start_idx = 0
    end_idx   = max(0, total - 1)
    if end_idx <= start_idx:
        end_idx = min(total - 1, start_idx + max(1, want))
    idx_all = np.linspace(start_idx, end_idx, max(1, want)).astype(np.int64)
    ts = idx_all / max(fps, 1e-6)
    frames = read_frames(backend, handles, idx_all)
    if resize_short and resize_short > 0:
        nf = []
        for img in frames:
            w, h = img.size
            short = min(w, h)
            if short > resize_short:
                if w < h:
                    new_w = resize_short; new_h = int(h * (resize_short / w))
                else:
                    new_h = resize_short; new_w = int(w * (resize_short / h))
                img = img.resize((new_w, new_h), Image.BILINEAR)
            nf.append(img)
        frames = nf
    tids = _temporal_ids(ts)
    return frames, tids, (s, e), len(idx_all)

def load_model(where: str, device: str, dtype, cache_dir: str, local_only: bool, revision: str, is_dir: bool):
    common = dict(trust_remote_code=True, attn_implementation='sdpa', **_dtype_kw(dtype))
    if cache_dir: common['cache_dir'] = cache_dir
    if local_only: common['local_files_only'] = True
    if is_dir:
        m = AutoModel.from_pretrained(where, **common).eval()
        t = AutoTokenizer.from_pretrained(where, trust_remote_code=True, local_files_only=local_only)
    else:
        m = AutoModel.from_pretrained(where, revision=revision, **common).eval()
        t = AutoTokenizer.from_pretrained(where, revision=revision, trust_remote_code=True,
                                          **({} if not cache_dir else {'cache_dir': cache_dir}))
    if device == 'cuda' and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            if args.disable_flash_sdp: torch.backends.cuda.enable_flash_sdp(False)
            if args.force_math_sdp:
                torch.backends.cuda.enable_math_sdp(True); torch.backends.cuda.enable_mem_efficient_sdp(False)
            else:
                torch.backends.cuda.enable_math_sdp(True); torch.backends.cuda.enable_mem_efficient_sdp(True)
        except Exception: pass
        m = m.to('cuda')
    return m, t

# =============== 评测提示词（无示例库） ===============
_PROMPT_ASSERT_BASE = """
You will see a video and an “assertion (a true/false statement)”. Judge it strictly based on the video content and output JSON as specified.

[Scoring Objective]
- answer: Whether the assertion holds in the video, based on what can actually be observed (the object performs the action or the event truly occurs). Must be "YES" or "NO".
- consistency: An integer from 0–100 measuring consistency with real-world physics/common sense (higher = more consistent). Using this video, think about the objects/events/aspects involved in the assertion and score how consistent they are with world knowledge and physical laws (e.g., no unreasonable shot breaks or object disappearance; object shape/size/state not illogical; living beings without extra/distorted limbs).

[Rubric]
0–19: grossly inconsistent with real-world common sense
20–39: largely inconsistent with real-world common sense
40–59: partially consistent, partially inconsistent
60–79: mostly consistent with real-world scenarios
80–100: fine-grained details match the real world; video is not blurry; fully consistent

[Output Schema]
Only output JSON (no extra text):
{{"answer":"YES|NO","consistency":<0-100 integer>}}

Assertion:
{TEXT}
""".strip()

_PROMPT_VQA_BASE = """
You will see a video and a “question about the video (VQA)”. Answer strictly based on the video content with a short answer (≤ 20 English words) and provide a common-sense consistency score, then output JSON as specified.

[Scoring Objective]
- answer: Provide a concise short answer (≤ 20 English words) strictly based on the current video content.
- consistency: An integer from 0–100 measuring how consistent the aspects involved in the question/answer are with real-world physics/common sense. Based on this video, score whether the phenomena related to the question follow world knowledge and physical laws (e.g., no exaggerated distortions, no sudden disappearance of people/objects). Lower if it violates common sense; higher if it matches. Output an integer 0–100 (higher = more consistent).

[Rubric]
0–19: grossly inconsistent with real-world common sense
20–39: largely inconsistent with real-world common sense
40–59: partially consistent, partially inconsistent
60–79: mostly consistent with real-world scenarios
80–100: fine-grained details match the real world; video is not blurry; fully consistent

[Guideline for Yes/No]
For yes/no questions, answer in a way that makes the ground truth in this video clearly identifiable (confirm the correct behavior exists, or confirm no abnormality is observed).

[Output Schema]
Only output JSON (no extra text):
{{"answer":"<short answer>","consistency":<0-100 integer>}}

Question:
{TEXT}
""".strip()

_INT100_RE = re.compile(r'(?<!\d)(\d{1,2}|100)(?!\d)')
_JSON_GUARD = re.compile(r'^\s*\{.*\}\s*$', re.S)

def _strip_think(text: str) -> str:
    if not isinstance(text, str):
        return text
    return re.sub(r'<think>.*?</think>', '', text, flags=re.S|re.I).strip()

def _strip_code_fences_keep_inner(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    lines = text.replace('\r', '\n').split('\n')
    out_lines, fence = [], False
    for ln in lines:
        if re.match(r'^\s*```', ln):
            fence = not fence
            continue
        out_lines.append(ln)
    return '\n'.join(out_lines)

def _clean(text: str) -> str:
    return _strip_code_fences_keep_inner(_strip_think(text or ""))

def _parse_eval_json(raw: str, *, want_yesno: bool) -> Tuple[str, int]:
    raw = _clean(raw)
    if _JSON_GUARD.search(raw):
        try:
            obj = json.loads(raw)
            ans = str(obj.get("answer","")).strip()
            cns = int(obj.get("consistency", 0))
            if want_yesno:
                ans = "YES" if ans.upper().startswith("Y") else ("NO" if ans.upper().startswith("N") else "")
            else:
                ans = ans[:64]
            cns = max(0, min(100, int(cns)))
            return ans, cns
        except Exception:
            pass
    m = _INT100_RE.search(raw)
    cns = int(m.group(1)) if m else 0
    return ("", max(0, min(100, cns)))

def _build_prompt_assert(text: str) -> str:
    return _PROMPT_ASSERT_BASE.format(TEXT=text)

def _build_prompt_vqa(text: str) -> str:
    return _PROMPT_VQA_BASE.format(TEXT=text)

# ========= 新增：每视频一次绑定，避免每题重复编码 =========
def _try_bind_images(model, frames: List[Image.Image], tids: List[List[int]]) -> Optional[Union[str, int]]:
    """
    尝试把整段 frames 绑定到模型，返回 image_id（或 None）。
    - 若模型无该接口或失败，返回 None（回退到逐题传 frames）。
    - 不改变原有其它行为。
    """
    try:
        # 常见命名：bind_images / encode_video / cache_images等，尽量自适配
        if hasattr(model, "bind_images"):
            return model.bind_images(frames, temporal_ids=tids)
        if hasattr(model, "encode_video"):
            return model.encode_video(frames, temporal_ids=tids)
        if hasattr(model, "cache_images"):
            return model.cache_images(frames, temporal_ids=tids)
    except Exception:
        pass
    return None

# —— 单题评测（同视频共享文本历史） —— #
def _call_model_eval(frames: List[Image.Image],
                     tids: List[List[int]],
                     model, tok,
                     *,
                     text: str,
                     typ: str,
                     chat_history: Optional[List[Dict[str,str]]] = None,
                     image_binding: Optional[Union[str,int]] = None) -> Tuple[str, int, str]:
    if typ.lower() == 'assertion':
        prompt = _build_prompt_assert(text)
        want_yesno = True
    else:
        prompt = _build_prompt_vqa(text)
        want_yesno = False

    msgs = []
    if chat_history:
        msgs += chat_history

    if image_binding is not None:
        # 使用缓存的视觉特征：只传 image_id + 文本
        msgs.append({'role': 'user', 'content': [ {"type":"image_id", "image_id": image_binding}, prompt ]})
        out = model.chat(
            msgs=msgs,
            tokenizer=tok,
            temporal_ids=tids,
            use_image_id=True,
            max_slice_nums=1,
            do_sample=(args.temperature > 0),
            temperature=(args.temperature if args.temperature > 0 else None),
            enable_thinking=args.enable_thinking,
            max_new_tokens=args.max_new_tokens
        )
    else:
        # 回退：逐题传 frames（保持原有行为）
        msgs.append({'role': 'user', 'content': frames + [prompt]})
        out = model.chat(
            msgs=msgs,
            tokenizer=tok,
            temporal_ids=tids,
            use_image_id=False,
            max_slice_nums=1,
            do_sample=(args.temperature > 0),
            temperature=(args.temperature if args.temperature > 0 else None),
            enable_thinking=args.enable_thinking,
            max_new_tokens=args.max_new_tokens
        )

    raw = _clean(out if isinstance(out, str) else str(out))
    ans, score = _parse_eval_json(raw, want_yesno=want_yesno)

    if chat_history is not None:
        # “一个视频一个对话”的多轮对话：把问答追加进该视频的对话历史
        q_line = f"[{typ}] {text}"
        a_line = f"[answer]={ans} [consistency]={score}"
        chat_history.append({'role':'user','content':q_line})
        chat_history.append({'role':'assistant','content':a_line})

    return ans, score, raw

def _map_avg_to_band(x: float) -> int:
    if x < 20: return 1
    if x < 40: return 2
    if x < 60: return 3
    if x < 80: return 4
    return 5

# ================== 批量辅助 ==================
def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def _scan_video_files(root: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for dirpath, _, files in os.walk(root):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in VIDEO_EXTS:
                full = os.path.join(dirpath, fn)
                raw_stem = _stem(full)
                norm = _norm_stem(raw_stem)
                out[norm] = full
    return out

def _iter_banks(bank_dir: str) -> List[str]:
    paths = []
    paths += glob.glob(os.path.join(bank_dir, "**", "*_scord.json"), recursive=True)
    paths += glob.glob(os.path.join(bank_dir, "**", "*_scored.json"), recursive=True)
    return sorted(set(paths))

def _index_banks(bank_dir: str) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for bj in _iter_banks(bank_dir):
        base = os.path.basename(bj)
        name_stem = os.path.splitext(base)[0]
        key1 = _norm_stem(name_stem)
        if key1:
            idx.setdefault(key1, bj)
        try:
            with open(bj, 'r', encoding='utf-8') as f:
                obj = json.load(f)
            v = obj.get('video') or ""
            if isinstance(v, str) and v:
                vstem = _stem(v); key2 = _norm_stem(vstem)
                if key2 and key2 not in idx:
                    idx[key2] = bj
        except Exception:
            pass
    return idx

def _category_of_video(video_path: str, video_root: str) -> str:
    rel = os.path.relpath(os.path.dirname(video_path), video_root)
    parts = [p for p in rel.split(os.sep) if p and p != "."]
    return parts[0] if parts else "root"

# ================== 主流程 ==================
def main():
    _ensure_dir(args.out_dir)

    # “模型名”由 --video-dir 的最后一级目录名决定
    model_name = os.path.basename(os.path.normpath(args.video_dir)) or "model"

    # 1) 先索引视频
    stem2video = _scan_video_files(args.video_dir)
    if args.verbose:
        print(f"[Index] videos={len(stem2video)} under: {args.video_dir} | model={model_name}")

    # 2) 建立 bank 索引
    bank_index = _index_banks(args.bank_dir)
    if args.verbose:
        print(f"[Index] banks={len(bank_index)} under: {args.bank_dir}")

    # 3) 载入模型（一次加载，复用）
    device = 'cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu'
    dtype  = _dtype_of(device, args.dtype)
    where  = args.local_path if args.local_path else args.model_id
    is_dir = bool(args.local_path and os.path.isdir(args.local_path))
    model, tok = load_model(where, device, dtype, args.cache_dir, args.local_files_only, args.revision, is_dir)

    # 4) 遍历“视频”，用规范化 stem 去匹配 bank
    per_class_rows: Dict[str, List[Tuple[str, str, int]]] = {}

    for norm_stem, video_file in sorted(stem2video.items()):
        bank_json = bank_index.get(norm_stem)
        if not bank_json:
            raw_stem = _stem(video_file)
            bank_json = bank_index.get(_norm_stem(raw_stem)) or bank_json
        if not bank_json:
            if args.verbose:
                print(f"[Skip] no bank for video (norm_stem={norm_stem}) -> {os.path.basename(video_file)}")
            continue

        # 读取题库
        try:
            with open(bank_json, 'r', encoding='utf-8') as f:
                bank = json.load(f)
        except Exception as e:
            print(f"[WARN] failed to load bank: {bank_json} | {e}")
            continue

        raw_items = bank.get('items') or []
        items = [it for it in raw_items if float(it.get('total_score_100', 0)) >= 80.0]
        if args.verbose:
            print(f"[Bank] {os.path.basename(bank_json)} | select {len(items)}/{len(raw_items)} for video={os.path.basename(video_file)}")

        # 5) 准备视频帧（读取 -> 编码 -> 绑定 -> 释放 IO）
        meta = read_video_meta_and_backend(video_file, preference=args.decode_backend)
        backend, handles, fps, total, duration = meta
        if args.verbose:
            print(f"[Video] {os.path.basename(video_file)} | {duration:.2f}s | fps={fps:.2f} | frames={total}")
        frames, tids, (cs, ce), _ = encode_whole_video(meta, args.fps, min(args.cap_frames, MAX_NUM_FRAMES), args.resize_short)

        # —— 新增：每视频一次绑定（若可用），避免每题重复视觉编码
        image_binding = _try_bind_images(model, frames, tids)

        # 6) 类名（用于 CSV 聚合）
        video_base = _stem(video_file)  # e.g., architecture_multi_005_prompt
        category_guess = _category_of_video(video_file, args.video_dir)
        class_name = _class_from_norm_stem(norm_stem, fallback=category_guess)

        # 7) 一个视频一个对话：同一视频共享文本历史（多轮对话）
        chat_history: List[Dict[str, str]] = []

        per_item_results = []
        csum = 0.0
        for it in items:
            typ  = str(it.get('type','VQA') or 'VQA')
            text = str(it.get('name') or it.get('text') or '').strip()
            ans, score, raw = _call_model_eval(
                frames, tids, model, tok,
                text=text, typ=typ,
                chat_history=chat_history,
                image_binding=image_binding  # 关键：使用缓存的视觉特征
            )
            csum += float(score)
            per_item_results.append({
                "id": it.get("id",""),
                "type": typ,
                "text": text,
                "answer": ans,
                "consistency": int(score),
                "raw": raw if args.dump_per_item else ""
            })
            if args.verbose:
                show = (text[:60] + '...') if len(text) > 60 else text
                print(f"[Item] {typ:<9} | {show:<64} -> score={score}")

        avg = (csum / max(1, len(per_item_results))) if per_item_results else 0.0
        band = _map_avg_to_band(avg)

        per_class_rows.setdefault(class_name, []).append((model_name, video_base, band))

        # detail
        if args.dump_per_item:
            detail_dir = os.path.join(args.out_dir, class_name)
            _ensure_dir(detail_dir)
            sidecar = os.path.join(detail_dir, f"{video_base}.detail.json")
            with open(sidecar, 'w', encoding='utf-8') as f:
                json.dump({
                    "video": video_base,
                    "class": class_name,
                    "model": model_name,
                    "avg_consistency": round(avg, 2),
                    "mapped_score_0_5": band,
                    "selected_items": per_item_results
                }, f, ensure_ascii=False, indent=2)
            if args.verbose:
                print(f"[OK] Detail -> {sidecar}")

        # —— 新增：本视频处理完毕后释放视频 IO（再加载下一个 IO）
        _release_video_backend(meta)

    # 8) 写出“按类名聚合”的 CSV：
    _ensure_dir(args.out_dir)

    def _append_rows_to_csv(csv_path: str, rows: List[Tuple[str, str, int]]):
        need_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
        _ensure_dir_for_file(csv_path)
        with open(csv_path, 'a', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            if need_header:
                w.writerow(["模型","名称","得分"])
            for model, name, band in rows:
                w.writerow([model, name, band])

    for cls, rows in sorted(per_class_rows.items()):
        rows_sorted = sorted(rows, key=lambda x: x[1])
        cls_csv = os.path.join(args.out_dir, f"{cls}.csv")
        _append_rows_to_csv(cls_csv, rows_sorted)
        if args.verbose:
            print(f"[OK] CSV (CLASS, append) -> {cls_csv} (+{len(rows_sorted)} rows)")

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6')
        main()
