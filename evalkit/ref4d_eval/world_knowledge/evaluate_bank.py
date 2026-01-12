# -*- coding: utf-8 -*-
"""
evaluate_bank.py (终极混合版)
功能：
1. 作为 Python 库：提供 VideoEvaluator 类，供外部项目 import 调用 (get_score)。
2. 作为 命令行工具：保留原有的批量扫描、CSV输出、文件名匹配逻辑。
"""

import os, re, json, math, argparse, warnings, inspect, csv, glob
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer

# ================= 1. 核心配置与常量 (保留原有逻辑) =================
TIME_SCALE = 0.1
MAX_NUM_FRAMES = 450
VIDEO_EXTS = {'.mp4', '.mkv', '.mov', '.avi', '.m4v', '.webm'}

# Prompt 模板保持不变
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

# ================= 2. 核心工具函数 (保留原有逻辑) =================
def _dtype_kw(d):
    sig = inspect.signature(AutoModel.from_pretrained)
    return {"dtype": d} if "dtype" in sig.parameters else {"torch_dtype": d}

def _dtype_of(device: str, choice: str):
    if device == 'cpu': return torch.float32
    return {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[choice]

def _ensure_dir(path: str):
    os.makedirs(path or ".", exist_ok=True)

def _ensure_dir_for_file(file_path: str):
    _ensure_dir(os.path.dirname(file_path))

def _strip_think(text: str) -> str:
    if not isinstance(text, str): return text
    return re.sub(r'<think>.*?</think>', '', text, flags=re.S|re.I).strip()

def _strip_code_fences_keep_inner(text: str) -> str:
    if not isinstance(text, str): text = str(text)
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
        except Exception: pass
    m = _INT100_RE.search(raw)
    cns = int(m.group(1)) if m else 0
    return ("", max(0, min(100, cns)))

def _map_avg_to_band(x: float) -> int:
    if x < 20: return 1
    if x < 40: return 2
    if x < 60: return 3
    if x < 80: return 4
    return 5

# ================= 3. 视频处理逻辑 (完全保留) =================
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
        except Exception: pass
    if preference in ('auto', 'decord'):
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            total = len(vr)
            try: fps = float(vr.get_avg_fps())
            except Exception: fps = 25.0
            duration = total / max(fps, 1e-6)
            backend = 'decord'
            return backend, (None, vr), fps, total, duration
        except Exception: pass
    raise RuntimeError(f"No available video backend for {video_path}")

def _release_video_backend(meta):
    backend, handles, _, _, _ = meta
    if backend == 'cv2':
        cap, _ = handles
        if cap: cap.release()
    else:
        _, vr = handles
        if vr: del vr

def read_frames(backend: str, handles, idx_all: np.ndarray) -> List[Image.Image]:
    frames = []
    if backend == 'cv2':
        cap, _ = handles
        import cv2
        if len(idx_all) == 0: return frames
        try: cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx_all[0]))
        except: pass
        cur = int(idx_all[0]); want = set(int(i) for i in idx_all.tolist())
        while True:
            ok, frame = cap.read()
            if not ok: break
            if cur in want:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
                if len(frames) >= len(idx_all): break
            cur += 1
    else:
        _, vr = handles
        if len(idx_all) == 0: return frames
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

def encode_whole_video(meta, choose_fps, cap_frames, resize_short):
    backend, handles, fps, total, duration = meta
    s, e = 0.0, duration
    clip_len = max(1e-6, e - s)
    want = int(min(cap_frames, round(clip_len * min(choose_fps, int(round(fps))))))
    start_idx = 0; end_idx = max(0, total - 1)
    if end_idx <= start_idx: end_idx = min(total - 1, start_idx + max(1, want))
    idx_all = np.linspace(start_idx, end_idx, max(1, want)).astype(np.int64)
    ts = idx_all / max(fps, 1e-6)
    frames = read_frames(backend, handles, idx_all)
    if resize_short and resize_short > 0:
        nf = []
        for img in frames:
            w, h = img.size
            short = min(w, h)
            if short > resize_short:
                scale = resize_short / short
                img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
            nf.append(img)
        frames = nf
    tids = _temporal_ids(ts)
    return frames, tids, (s, e), len(idx_all)


# ================= 4. 核心类定义 (New: 方便外部项目调用) =================

class EvaluatorConfig:
    """集中管理所有配置参数"""
    def __init__(self, **kwargs):
        # 默认值
        self.fps = kwargs.get('fps', 3)
        self.cap_frames = kwargs.get('cap_frames', 300)
        self.resize_short = kwargs.get('resize_short', 448)
        self.decode_backend = kwargs.get('decode_backend', 'auto')
        self.max_new_tokens = kwargs.get('max_new_tokens', 256)
        self.temperature = kwargs.get('temperature', 0.0)
        self.enable_thinking = kwargs.get('enable_thinking', False)
        
        # 批量运行专用
        self.dump_per_item = kwargs.get('dump_per_item', True)
        self.verbose = kwargs.get('verbose', True)

class VideoEvaluator:
    """
    评测器类：
    - 负责加载模型 (__init__)
    - 负责评测单个视频 (evaluate_single)
    """
    def __init__(self, model_path_or_id: str, device: str = 'cuda', dtype: str = 'bf16', local_files_only: bool = False, **kwargs):
        self.cfg = EvaluatorConfig(**kwargs)
        self.device = device
        
        print(f"[Evaluator] Loading model from {model_path_or_id} ...")
        dt_map = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
        dt = dt_map[dtype] if device != 'cpu' else torch.float32
        
        common = dict(trust_remote_code=True, attn_implementation='sdpa', **_dtype_kw(dt))
        if local_files_only: common['local_files_only'] = True
        
        is_dir = os.path.isdir(model_path_or_id)
        if is_dir:
            self.model = AutoModel.from_pretrained(model_path_or_id, **common).eval()
            self.tok = AutoTokenizer.from_pretrained(model_path_or_id, trust_remote_code=True, local_files_only=local_files_only)
        else:
            self.model = AutoModel.from_pretrained(model_path_or_id, revision='main', **common).eval()
            self.tok = AutoTokenizer.from_pretrained(model_path_or_id, revision='main', trust_remote_code=True)

        if device == 'cuda' and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                # 兼容性尝试
                if not kwargs.get('disable_flash_sdp', False): torch.backends.cuda.enable_flash_sdp(True)
                if kwargs.get('force_math_sdp', False): torch.backends.cuda.enable_math_sdp(True)
            except: pass
            self.model = self.model.to('cuda')
        print("[Evaluator] Ready.")

    def _try_bind_images(self, frames, tids):
        """尝试使用 bind_images 缓存视觉特征"""
        try:
            if hasattr(self.model, "bind_images"): return self.model.bind_images(frames, temporal_ids=tids)
            if hasattr(self.model, "encode_video"): return self.model.encode_video(frames, temporal_ids=tids)
            if hasattr(self.model, "cache_images"): return self.model.cache_images(frames, temporal_ids=tids)
        except: pass
        return None

    def evaluate_single(self, video_path: str, bank_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        对单个视频进行评测的核心入口
        :param video_path: 视频路径
        :param bank_data: 题库字典 (json.load后的结果)
        :return: { "band": 1-5, "avg_score": float, "details": list, "meta": dict }
        """
        # 1. 过滤题目
        raw_items = bank_data.get('items') or []
        items = [it for it in raw_items if float(it.get('total_score_100', 0)) >= 80.0]
        
        if not items:
            return {"band": 1, "avg_score": 0.0, "details": [], "meta": {"valid_items": 0}}

        # 2. 读取视频与编码
        meta = read_video_meta_and_backend(video_path, preference=self.cfg.decode_backend)
        frames, tids, _, _ = encode_whole_video(meta, self.cfg.fps, self.cfg.cap_frames, self.cfg.resize_short)
        
        # 3. 视觉绑定 (加速)
        img_binding = self._try_bind_images(frames, tids)

        # 4. 循环打分
        chat_history = []
        results = []
        csum = 0.0

        for it in items:
            typ = str(it.get('type', 'VQA') or 'VQA')
            text = str(it.get('name') or it.get('text') or '').strip()
            
            if typ.lower() == 'assertion':
                prompt = _PROMPT_ASSERT_BASE.format(TEXT=text)
                want_yesno = True
            else:
                prompt = _PROMPT_VQA_BASE.format(TEXT=text)
                want_yesno = False

            msgs = list(chat_history)
            
            if img_binding is not None:
                msgs.append({'role': 'user', 'content': [{"type":"image_id", "image_id": img_binding}, prompt]})
                use_img_id = True
            else:
                msgs.append({'role': 'user', 'content': frames + [prompt]})
                use_img_id = False

            out = self.model.chat(
                msgs=msgs,
                tokenizer=self.tok,
                temporal_ids=tids,
                use_image_id=use_img_id,
                max_slice_nums=1,
                do_sample=(self.cfg.temperature > 0),
                temperature=(self.cfg.temperature if self.cfg.temperature > 0 else None),
                enable_thinking=self.cfg.enable_thinking,
                max_new_tokens=self.cfg.max_new_tokens
            )

            raw = _clean(out if isinstance(out, str) else str(out))
            ans, score = _parse_eval_json(raw, want_yesno=want_yesno)
            
            # 更新历史
            q_line = f"[{typ}] {text}"
            a_line = f"[answer]={ans} [consistency]={score}"
            chat_history.append({'role':'user','content':q_line})
            chat_history.append({'role':'assistant','content':a_line})

            csum += score
            results.append({
                "id": it.get("id", ""),
                "type": typ,
                "text": text,
                "answer": ans,
                "consistency": score,
                "raw": raw if self.cfg.dump_per_item else ""
            })
            
            if self.cfg.verbose:
                show = (text[:60] + '...') if len(text) > 60 else text
                print(f"[Item] {typ:<9} | {show:<64} -> score={score}")

        # 5. 释放资源
        _release_video_backend(meta)

        avg = (csum / max(1, len(results))) if results else 0.0
        band = _map_avg_to_band(avg)

        return {
            "band": band,
            "avg_score": avg,
            "details": results,
            "meta": {"valid_items": len(items)}
        }

# ================= 5. CLI 辅助函数 (保留原有逻辑用于文件名匹配) =================
_SUFFIXES = [r"_prompt", r"_scord", r"_scored", r"_vqa", r"_assertion", r"_assertions", r"_qa", r"_bank"]
_suffix_re = re.compile("(" + "|".join(s + r"$" for s in _SUFFIXES) + ")", re.I)

def _norm_stem(s: str) -> str:
    x = (s or "").strip().lower()
    if not x: return x
    while True:
        y = _suffix_re.sub("", x)
        if y == x: return y
        x = y

_CLASS_RE = re.compile(r'^(?P<cls>.+)_(single|multi)_[0-9]+$', re.I)
def _class_from_norm_stem(norm_stem: str, fallback: str) -> str:
    if not norm_stem: return fallback
    m = _CLASS_RE.match(norm_stem)
    if m: return m.group('cls').strip()
    return fallback

def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def _scan_video_files(root: str) -> Dict[str, str]:
    out = {}
    for dirpath, _, files in os.walk(root):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in VIDEO_EXTS:
                full = os.path.join(dirpath, fn)
                out[_norm_stem(_stem(full))] = full
    return out

def _index_banks(bank_dir: str) -> Dict[str, str]:
    idx = {}
    paths = glob.glob(os.path.join(bank_dir, "**", "*_scord.json"), recursive=True) + \
            glob.glob(os.path.join(bank_dir, "**", "*_scored.json"), recursive=True)
    for bj in paths:
        base = os.path.basename(bj)
        key1 = _norm_stem(os.path.splitext(base)[0])
        if key1: idx.setdefault(key1, bj)
        try:
            with open(bj, 'r', encoding='utf-8') as f:
                v = json.load(f).get('video', "")
                if v: 
                    key2 = _norm_stem(_stem(v))
                    if key2 and key2 not in idx: idx[key2] = bj
        except: pass
    return idx

def _category_of_video(video_path: str, video_root: str) -> str:
    rel = os.path.relpath(os.path.dirname(video_path), video_root)
    parts = [p for p in rel.split(os.sep) if p and p != "."]
    return parts[0] if parts else "root"

# ================= 6. CLI 主入口 (逻辑全量保留) =================
def cli_main():
    p = argparse.ArgumentParser()
    # 批量入口
    p.add_argument('--bank-dir',   help='题库目录 (批量模式)')
    p.add_argument('--video-dir',  help='视频目录 (批量模式)')
    p.add_argument('--out-dir',    help='输出目录')
    
    # API入口 (单文件)
    p.add_argument('--video-file', help='单个视频文件')
    p.add_argument('--bank-file',  help='单个题库文件')

    # 模型参数
    p.add_argument('--local-path', default='', help='模型路径')
    p.add_argument('--model-id',   default='openbmb/MiniCPM-V-4_5')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype',  default='bf16')
    p.add_argument('--fps',           type=int,   default=3)
    p.add_argument('--cap-frames',    type=int,   default=300)
    p.add_argument('--resize-short',  type=int,   default=448)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--temperature',    type=float, default=0.0)
    p.add_argument('--enable-thinking', action='store_true')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--dump-per-item', action='store_true')
    p.add_argument('--decode-backend', default='auto')
    
    # 兼容参数
    p.add_argument('--revision', default='main')
    p.add_argument('--cache-dir', default='')
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--disable-flash-sdp', action='store_true')
    p.add_argument('--force-math-sdp', action='store_true')
    p.set_defaults(dump_per_item=True)
    
    args = p.parse_args()

    # 初始化评测器
    evaluator = VideoEvaluator(
        model_path_or_id=(args.local_path or args.model_id),
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        # 传递其他配置
        fps=args.fps, cap_frames=args.cap_frames, resize_short=args.resize_short,
        decode_backend=args.decode_backend, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, enable_thinking=args.enable_thinking,
        verbose=args.verbose, dump_per_item=args.dump_per_item
    )

    # 模式 A: 单文件调用 (API 风格)
    if args.video_file and args.bank_file:
        print(f"[Mode] Single File: {args.video_file}")
        with open(args.bank_file, 'r', encoding='utf-8') as f:
            bank_data = json.load(f)
        
        res = evaluator.evaluate_single(args.video_file, bank_data)
        
        print("\n" + "="*30)
        print(f"Avg Score: {res['avg_score']:.2f} / 100")
        print(f"Band     : {res['band']} (1-5)")
        print("="*30)
        
        if args.out_dir:
            _ensure_dir(args.out_dir)
            out_path = os.path.join(args.out_dir, f"{_stem(args.video_file)}_result.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            print(f"Result saved to {out_path}")
        return

    # 模式 B: 批量扫描 (原逻辑恢复)
    if args.video_dir and args.bank_dir and args.out_dir:
        _ensure_dir(args.out_dir)
        stem2video = _scan_video_files(args.video_dir)
        bank_index = _index_banks(args.bank_dir)
        print(f"[Mode] Batch: {len(stem2video)} videos, {len(bank_index)} banks.")
        
        per_class_rows = {}
        model_name = os.path.basename(os.path.normpath(args.video_dir))

        for norm_stem, video_file in sorted(stem2video.items()):
            bank_json = bank_index.get(norm_stem)
            if not bank_json:
                raw_stem = _stem(video_file)
                bank_json = bank_index.get(_norm_stem(raw_stem))
            
            if not bank_json:
                if args.verbose: print(f"[Skip] No bank for {os.path.basename(video_file)}")
                continue

            try:
                with open(bank_json, 'r', encoding='utf-8') as f:
                    bank_data = json.load(f)
            except Exception as e:
                print(f"[Error] Load bank failed: {e}")
                continue

            # 调用核心评测
            if args.verbose: print(f"[Run] {os.path.basename(video_file)} ...")
            res = evaluator.evaluate_single(video_file, bank_data)
            
            # 聚合结果
            video_base = _stem(video_file)
            cat_guess = _category_of_video(video_file, args.video_dir)
            cls_name = _class_from_norm_stem(norm_stem, fallback=cat_guess)
            
            per_class_rows.setdefault(cls_name, []).append((model_name, video_base, res['band']))
            
            # 写入 Detail
            if args.dump_per_item:
                detail_dir = os.path.join(args.out_dir, cls_name)
                _ensure_dir(detail_dir)
                with open(os.path.join(detail_dir, f"{video_base}.detail.json"), 'w', encoding='utf-8') as f:
                    json.dump({
                        "video": video_base, "class": cls_name, "model": model_name,
                        "avg_consistency": round(res['avg_score'], 2),
                        "mapped_score_0_5": res['band'],
                        "selected_items": res['details']
                    }, f, ensure_ascii=False, indent=2)

        # 写出 CSV
        for cls, rows in sorted(per_class_rows.items()):
            rows.sort(key=lambda x: x[1])
            csv_path = os.path.join(args.out_dir, f"{cls}.csv")
            need_header = not os.path.exists(csv_path)
            with open(csv_path, 'a', encoding='utf-8', newline='') as f:
                w = csv.writer(f)
                if need_header: w.writerow(["modelname","sample_id","world_score"])
                for r in rows: w.writerow(r)
            print(f"[CSV] Appended {len(rows)} rows to {csv_path}")

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6')
        cli_main()