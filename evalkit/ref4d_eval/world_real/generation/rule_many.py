# -*- coding: utf-8 -*-
import os, re, json, argparse, warnings, inspect, gc
from typing import List, Tuple, Dict, Any, Set, Optional
import math
import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer
from rule_examples import (
    VIDEO_EXTS, TIME_SCALE, MAX_NUM_FRAMES, BLACKLIST,
    SIGNAL_LIST, SCHEMA, EXAMPLE_BANK_CORE, render_header_examples
)

# ---------------- CLI ----------------
p = argparse.ArgumentParser()
p.add_argument('--json-dir', required=True, help='基础语义 JSON 目录（与视频同名 *.json）')
p.add_argument('--video-dir', required=True, help='视频目录（支持 mp4/mkv/mov/avi/m4v/webm）')
p.add_argument('--out-dir',   required=True, help='输出目录（每个视频生成一个 *_rules.json）')

# 模型
p.add_argument('--local-path', default='', help='本地权重目录（优先）')
p.add_argument('--model-id',   default='openbmb/MiniCPM-V-4_5')
p.add_argument('--revision',   default='main')
p.add_argument('--cache-dir',  default='')
p.add_argument('--local-files-only', action='store_true')

# 设备/精度
p.add_argument('--device', default='cuda', choices=['cuda','cpu'])
p.add_argument('--dtype',  default='bf16', choices=['bf16','fp16','fp32'])

# 时间窗 / 抽帧
p.add_argument('--window-sec', type=float, default=4.0, help='时间窗时长（秒）')
p.add_argument('--hop-sec',    type=float, default=2.0, help='时间窗步长（秒）')
p.add_argument('--fps',        type=int,   default=3,   help='每窗抽帧的目标 fps')
p.add_argument('--cap-frames', type=int,   default=180, help='每窗/全片最多帧数上限（抽帧后）')
p.add_argument('--resize-short', type=int, default=448, help='短边缩放（像素）')
p.add_argument('--max-packing',  type=int, default=3,   help='temporal packing 通道数上限（1~6）')
p.add_argument('--decode-backend', default='auto', choices=['auto','cv2','decord'])

# 生成参数
p.add_argument('--max-new-tokens', type=int, default=256)
p.add_argument('--temperature', type=float, default=0.2)
p.add_argument('--enable-thinking', action='store_true')

# Prompt 示例库
p.add_argument('--with_examples', action='store_true', help='将示例库 few-shot 追加到 prompt 中')

# 时间单位/容差/调试
p.add_argument('--f-time-unit', choices=['sec','frame','auto'], default='auto',
              help='F 中时间单位：sec=秒, frame=帧号, auto=自动判断(默认)')
p.add_argument('--time-epsilon', type=float, default=0.2, help='时间裁剪容差秒数（默认 0.2s）')
p.add_argument('--dump-debug', action='store_true',
              help='调试：将每个窗口的 F 子集与模型原始输出落盘到 out-dir/debug/<视频名>/')

# 全局回退（兼容旧参；行为：统一按“全片总数 < 阈值”触发一次性回退）
p.add_argument('--global-fallback', action='store_true',
              help='兼容参数：现在始终按“全片总数 < 阈值”触发一次性回退（抽帧全片）')
p.add_argument('--global-fallback-threshold', type=int, default=3,
              help='当全片规则总数 < 阈值时触发回退（默认 3）')
p.add_argument('--global-fallback-scope', choices=['deficit','all'], default='deficit',
              help='兼容参数：已不生效，回退统一按全片一次性执行')
p.add_argument('--fallback-whole-video', action='store_true', help='回退整个视频进行推理')
p.add_argument('--fallback-whole-fps', type=int, default=4, help='回退时使用的FPS，默认为4')

p.add_argument('--verbose', action='store_true')
args = p.parse_args()

MAX_NUM_PACKING = max(1, min(args.max_packing, 6))
_header_extras = {
    "MAX_NUM_PACKING": MAX_NUM_PACKING,
    "f_time_unit": args.f_time_unit,
    "time_epsilon": args.time_epsilon,
    "decode_backend": args.decode_backend,
    "fps": args.fps,
    "window_sec": args.window_sec,
    "hop_sec": args.hop_sec,
    "resize_short": args.resize_short,
    "max_new_tokens": args.max_new_tokens,
    "temperature": args.temperature,
    "enable_thinking": bool(args.enable_thinking),
    "global_fallback": bool(args.global_fallback),
    "global_fallback_threshold": args.global_fallback_threshold,
    "fallback_whole_video": bool(args.fallback_whole_video),
    "fallback_whole_fps": args.fallback_whole_fps,
}

HEADER_EXAMPLES = render_header_examples(p, args, extra_consts=_header_extras)

if getattr(args, "with_examples", False):
    EXAMPLE_BANK = HEADER_EXAMPLES + "\n\n" + EXAMPLE_BANK_CORE
else:
    EXAMPLE_BANK = EXAMPLE_BANK_CORE


# ---------------- 实用函数 ----------------
def _ensure_dir(pth: str): os.makedirs(pth, exist_ok=True)

def _strip_think(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.S|re.I).strip() if isinstance(text, str) else text

def _guarded_json(s: str):
    if not isinstance(s, str): return None
    try:
        return json.loads(s)
    except Exception:
        lb = s.find('['); rb = s.rfind(']')
        lc = s.find('{'); rc = s.rfind('}')
        cand = None
        if lb != -1 and rb != -1 and rb > lb: cand = s[lb:rb+1]
        elif lc != -1 and rc != -1 and rc > lc: cand = s[lc:rc+1]
        if cand:
            try: return json.loads(cand)
            except: return None
    return None

def _dtype_of(device: str, choice: str):
    if device=='cpu': return torch.float32
    return {'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32}[choice]

def _dtype_kw(d):
    sig = inspect.signature(AutoModel.from_pretrained)
    return {"dtype": d} if "dtype" in sig.parameters else {"torch_dtype": d}

def _empty_cuda():
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass

# ---------- 解码后端选择 ----------
def read_video_meta_and_backend(video_path: str, preference: str = 'auto'):
    """
    返回: backend, (cap, vr), fps, total, duration
      - backend: 'cv2' 或 'decord'
      - cap/vr: 仅有其一非 None
    """
    # 优先 cv2
    if preference in ('auto','cv2'):
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps   = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps <= 1e-3:
                fps = 25.0
            duration = total / max(fps, 1e-6) if total > 0 else 0.0
            if total > 0:
                return 'cv2', (cap, None), fps, total, duration
            try:
                cap.release()
            except Exception:
                pass
        except Exception:
            pass
    if preference in ('auto','decord'):
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            total = len(vr)
            try:
                fps = float(vr.get_avg_fps())
                if fps <= 1e-3: fps = 25.0
            except Exception:
                fps = 25.0
            duration = total / max(fps, 1e-6)
            return 'decord', (None, vr), fps, total, duration
        except Exception:
            pass
    raise RuntimeError(f"No available backend for {video_path}")

# ---------- 帧读取 ----------
def read_frames(backend, handles, idx_all):
    """
    idx_all: np.ndarray[int64] 升序帧索引
    返回: List[PIL.Image]
    """
    frames = []
    if backend == 'cv2':
        cap, _ = handles
        import cv2
        if len(idx_all) == 0:
            return frames
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx_all[0]))
        except Exception:
            pass
        cur  = int(idx_all[0])
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

# ---------- 工具：时间网格映射与分组 ----------
def _map_to_nearest_scale(ts: np.ndarray, scale: np.ndarray) -> np.ndarray:
    idx = np.abs(ts[:, None] - scale[None, :]).argmin(axis=1)
    return scale[idx]

def _group_array(arr: np.ndarray, groups: int):
    groups = max(1, int(groups))
    return [arr[i::groups].astype(int).tolist() for i in range(groups)]

# ---------- 统一缩放（短边） ----------
def _resize_short_side_pil(img: Image.Image, short: int) -> Image.Image:
    if not short or short <= 0:
        return img
    w, h = img.size
    s = min(w, h)
    if s <= short:
        return img
    if w <= h:
        new_w = short
        new_h = int(round(h * short / max(w, 1)))
    else:
        new_h = short
        new_w = int(round(w * short / max(h, 1)))
    return img.resize((new_w, new_h), Image.BILINEAR)

# ---------- 抽帧（融合 packing + 对齐 + 统一缩放 + 兼容 cap 语义） ----------
def encode_clip(meta,
                start_s: float,
                end_s: float,
                choose_fps: int,
                cap_frames: int,
                resize_short: int,
                max_packing: int = MAX_NUM_PACKING):
    backend, handles, fps, total, duration = meta
    s = max(0.0, float(start_s))
    e = min(duration, float(end_s))
    if e <= s:
        e = min(duration, s + 1.0)
    clip_len = max(0.0, e - s)
    target_total = choose_fps * int(round(clip_len))  # 粗估目标数
    if target_total <= MAX_NUM_FRAMES:
        packing_nums  = 1
        choose_frames = round(min(choose_fps, int(round(fps))) * min(MAX_NUM_FRAMES, clip_len))
    else:
        packing_nums = int(math.ceil(clip_len * max(1, choose_fps) / MAX_NUM_FRAMES))
        if packing_nums <= max_packing:
            choose_frames = round(clip_len * max(1, choose_fps))
        else:
            packing_nums  = int(max_packing)
            choose_frames = round(MAX_NUM_FRAMES * packing_nums)
    if isinstance(cap_frames, int) and cap_frames > 0:
        choose_frames = int(min(choose_frames, cap_frames))
    else:
        choose_frames = int(max(1, choose_frames))
    start_idx = int(round(s * fps))
    end_idx   = max(0, int(round(e * fps)) - 1)
    end_idx   = min(total - 1, end_idx)
    if end_idx <= start_idx:
        end_idx = min(total - 1, start_idx + max(1, choose_frames))
    idx_all = np.linspace(start_idx, end_idx, max(1, choose_frames)).astype(np.int64)
    ts = idx_all / max(fps, 1e-6)
    scale = np.arange(0.0, duration + 1e-6, TIME_SCALE, dtype=float)
    tids  = _map_to_nearest_scale(ts, scale)
    tids  = (tids / TIME_SCALE).astype(np.int32)
    tids  = _group_array(tids, packing_nums)
    frames = read_frames(backend, handles, idx_all)
    if resize_short and resize_short > 0 and len(frames) > 0:
        frames = [_resize_short_side_pil(im, resize_short) for im in frames]

    return frames, tids, (s, e), len(idx_all)

# ---------------- 模型加载 ----------------
def _dtype_kw(d):
    sig = inspect.signature(AutoModel.from_pretrained)
    return {"dtype": d} if "dtype" in sig.parameters else {"torch_dtype": d}

def load_model(where: str,
               device: str,
               dtype,
               cache_dir: str = None,
               local_only: bool = False,
               revision: str = "main",
               is_dir: bool = False,
               use_fast_tokenizer: bool = False,
               attn_impl: str = "sdpa",
               disable_flash_sdp: bool = False,
               force_math_sdp: bool = False):
    common = dict(trust_remote_code=True, **_dtype_kw(dtype))
    if cache_dir: common["cache_dir"] = cache_dir
    if local_only: common["local_files_only"] = True
    if attn_impl:
        try:
            common["attn_implementation"] = attn_impl
        except Exception:
            pass
    if is_dir:
        m = AutoModel.from_pretrained(where, **common).eval()
    else:
        m = AutoModel.from_pretrained(where, revision=revision, **common).eval()
    tok_kw = dict(trust_remote_code=True, local_files_only=local_only, use_fast=use_fast_tokenizer)
    if cache_dir: tok_kw["cache_dir"] = cache_dir
    if (not is_dir) and revision: tok_kw["revision"] = revision
    t = AutoTokenizer.from_pretrained(where, **tok_kw)
    if device == 'cuda' and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            if disable_flash_sdp:
                torch.backends.cuda.enable_flash_sdp(False)
            if force_math_sdp:
                torch.backends.cuda.enable_math_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            else:
                pass
        except Exception:
            pass
        m = m.to('cuda')

    return m, t

# --------------- 时间单位识别（保留以便未来需要） ----------------
def _guess_time_unit(sem_json, fps, duration):
    fine = sem_json.get("fine", {}) or sem_json.get("basic_semantics", {}) or sem_json
    cand = []
    for seq in (fine.get("entities") or []):
        for ab in (seq.get("spans") or []):
            try: cand.append(float(ab[1]))
            except: pass
    for seq in (fine.get("relations") or []):
        for ab in (seq.get("spans") or []):
            try: cand.append(float(ab[1]))
            except: pass
    if not cand:
        return 'sec'
    m = float(np.nanmedian(cand))
    return 'frame' if m > max(3.0*duration, 90.0) else 'sec'

def _to_sec(x, fps, unit):
    if x is None: return None
    try:
        x = float(x)
        return x if unit == 'sec' else (x / max(fps, 1e-6))
    except:
        return None


# ---------- 兼容你现有的辅助 ----------
def _norm_eid(en):
    return en.get("id") or en.get("eid") or en.get("name")

def _iter_views_as_dicts(sem_json: dict, fine: dict):
    # 现在不再使用 views 做基础语义，但保留函数以兼容旧代码
    out = []
    for container in (sem_json, fine):
        v = container.get("views", None)
        if isinstance(v, dict):
            out.append(v)
        elif isinstance(v, list):
            out.extend([x for x in v if isinstance(x, dict)])
    return out

# ---------- 通用：收集 signature 的“值部分” ----------
def _collect_signatures_only(values) -> List[str]:
    buf = []

    def _add(v):
        if v is None:
            return
        if isinstance(v, (list, tuple, set)):
            for x in v:
                _add(x)
            return
        s = str(v).strip()
        if s:
            buf.append(s)

    _add(values)
    seen = set()
    out = []
    for s in buf:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

# ---------- 事件抽取 ----------
def _iter_events_as_dicts(*containers: dict) -> List[dict]:

    out: List[dict] = []
    for c in containers:
        if not isinstance(c, dict):
            continue
        for key in ("event", "events"):
            v = c.get(key)
            if isinstance(v, list):
                out.extend([x for x in v if isinstance(x, dict)])
    return out

def _format_time(t: Optional[float]) -> str:
    if t is None:
        return "?"
    try:
        return f"{float(t):.2f}"
    except Exception:
        return str(t)

def _events_in_window(events: List[dict], t0: float, t1: float, slack: float = 0.2) -> List[dict]:
    """
    返回与时间窗 [t0, t1] 有重叠的事件（给一点 slack 容忍边界）。
    """
    win0, win1 = t0 - slack, t1 + slack
    picked = []
    for ev in events:
        s = ev.get("s_abs", ev.get("start", ev.get("t0")))
        e = ev.get("e_abs", ev.get("end",   ev.get("t1")))
        try:
            s = float(s) if s is not None else None
            e = float(e) if e is not None else None
        except Exception:
            s = s or None
            e = e or None
        overlap = ((s is None or e is None) or not (e < win0 or s > win1))
        if overlap:
            picked.append(ev)
    return picked

# ---------- 构建“极简全局载荷”：objects + signatures_by_object + events ----------
def build_sem_payload_global(sem_json: dict,
                             evidence_json: Optional[dict] = None,
                             max_items_per_section: int = 256) -> dict:
    fine = sem_json.get("fine", {}) or sem_json.get("basic_semantics", {}) or sem_json

    objects: List[str] = []
    sig_by_object: Dict[str, List[str]] = {}

    # --- 路径 A：basic_semantics（你刚上传文件的结构）
    basic = sem_json.get("basic_semantics")
    if isinstance(basic, dict):
        # 对象名来自 objects_count 的 key 与 attributes 的 key 的并集
        obj_keys = set()
        if isinstance(basic.get("objects_count"), dict):
            obj_keys |= set(basic["objects_count"].keys())
        if isinstance(basic.get("attributes"), dict):
            obj_keys |= set(basic["attributes"].keys())

        # 排序以稳定输出
        for obj_name in sorted(obj_keys):
            name = str(obj_name).strip()
            if not name:
                continue
            objects.append(name)
            if len(objects) >= max_items_per_section:
                break

        # signatures：仅取 attributes 下每个对象的 'signature' 值
        attrs = basic.get("attributes") or {}
        if isinstance(attrs, dict):
            for obj_name, obj_attrs in attrs.items():
                if not isinstance(obj_attrs, dict):
                    continue
                sig_values = obj_attrs.get("signature", None)  # 只取 signature
                sigs = _collect_signatures_only(sig_values)
                if sigs:
                    sig_by_object[obj_name] = sigs

    # --- 路径 B：fine.entities（旧式）
    if not objects:
        seen_obj = set()
        for en in (fine.get("entities") or []):
            if not isinstance(en, dict):
                continue
            # 对象名称（只取“值”）
            obj_name = (en.get("name") or en.get("label") or en.get("object") or "").strip()
            if not obj_name:
                obj_name = str(_norm_eid(en) or "").strip()
            if not obj_name:
                continue
            if obj_name not in seen_obj:
                seen_obj.add(obj_name)
                objects.append(obj_name)
                if len(objects) >= max_items_per_section:
                    break

            # attributes.signature（只取“值部分”）
            attrs = en.get("attributes", {}) or {}
            sig_values = None
            if isinstance(attrs, dict) and "signature" in attrs:
                sig_values = attrs.get("signature")

            sigs = _collect_signatures_only(sig_values)
            if sigs:
                exist = sig_by_object.get(obj_name, [])
                merged = []
                seen = set(exist)
                for s in exist + sigs:
                    if s not in seen:
                        seen.add(s)
                        merged.append(s)
                sig_by_object[obj_name] = merged if exist else sigs

    events: List[dict] = []
    for ev in _iter_events_as_dicts(sem_json, fine, evidence_json or {}):
        events.append({
            "id":   (ev.get("id") or ev.get("eid") or ""),
            "t0":   ev.get("s_abs", ev.get("start", ev.get("t0"))),
            "t1":   ev.get("e_abs", ev.get("end",   ev.get("t1"))),
            "text": (ev.get("text") or ev.get("desc") or ev.get("description") or "").strip(),
        })
        if len(events) >= max_items_per_section:
            break

    return {
        "objects": objects,                    
        "signatures_by_object": sig_by_object, 
        "events": events,                    
    }


# ---------- 渲染“全局基础语义”：仅对象 + 每对象的 signatures（不含事件） ----------
def render_sem_for_prompt(payload: dict, max_len: int = 3000) -> str:
    lines = []

    # 对象（只输出值）
    objs = payload.get("objects") or []
    if objs:
        lines.append("Objects:")
        for name in objs:
            lines.append(f"  - {name}")

    # 每对象的 signatures（只输出值）
    sobj = payload.get("signatures_by_object") or {}
    if sobj:
        lines.append("Signatures per object:")
        # 按 objects 顺序优先，其次遍历剩余键
        ordered_keys = list(objs) + [k for k in sobj.keys() if k not in set(objs)]
        for name in ordered_keys:
            sigs = sobj.get(name, [])
            if not sigs:
                continue
            joined = ", ".join(str(s).strip() for s in sigs if str(s).strip())
            if joined:
                lines.append(f"  - {name}: {joined}")

    txt = "\n".join(lines) if lines else "(no base semantics)"
    if len(txt) > max_len:
        txt = txt[:max_len] + "\n  ... (truncated)"
    return txt


# ---------- Prompt：全局对象+签名（无时间） + “窗口内事件线索”（有时间） ----------
def build_rules_prompt(global_sem_text: str,
                       t0: float, t1: float,
                       payload_for_window: Optional[dict] = None) -> str:
    window_event_lines = ""
    if payload_for_window:
        evs = payload_for_window.get("events") or []
        picked = _events_in_window(evs, t0, t1, slack=0.2)
        if picked:
            window_event_lines = "\n[窗口内事件线索]\n" + "\n".join(
                f"- {ev.get('id','')} [{_format_time(ev.get('t0'))}, {_format_time(ev.get('t1'))}] {ev.get('text','')}".strip()
                for ev in picked
            )

    base = f"""
请基于给定时间窗内的视频帧，参考视频基础语义与窗口内事件，归纳该时间窗内存在的、可验证的高阶语义（Affordance/Task/Safety/Physics），用于判断 AIGC 视频是否符合世界知识与日常常识（如物体是否突然消失、两个物体突然合并、违背重力、穿模、生物的行为动作是否合理等）

[时间窗]
- [{t0:.2f}, {t1:.2f}] 秒（仅观察这段帧，避免引用窗口外的时空信息）

[全局基础语义（仅对象与 signatures）]
{global_sem_text}{window_event_lines}

[硬约束]
1) 规则聚焦高阶：功能/因果/安全/物理一致性；避免纯低阶外观描述。
2) 每条规则必须包含≥1 个 anchors （检测到规则的时间 [t0:, t1:]）和 ≥1 个 required_signals（从信号词表中选），无需在rule_text中显式打印。
3) 若事件线索与可见帧不一致，以可见帧为准。
4) 仅输出合法 JSON 数组，遵循“输出Schema”；不输出解释或多余文本，anchors格式为 ["t0": 0.0, "t1": 2.0]，信号词从信号词表中选取，请严格输出。
5) 要求rule_text必须是一段可理解的文本，为语义完整的一段话（不需要有信号词）。

[信号词表]
{json.dumps(SIGNAL_LIST, ensure_ascii=False)}

[输出Schema]
{SCHEMA}

严格要求：仅输出 JSON 数组；所有字段必须填写；JSON 可被严格解析。
""".strip()

    if getattr(args, "with_examples", False) and EXAMPLE_BANK:
        base += "\n\n[示例（few-shot，统一示例库）]\n" + EXAMPLE_BANK

    return base + "\n\n仅输出 JSON。"

# ---- 回退提示（视频直观） ----
def build_fallback_prompt() -> str:
    base = f"""
请根据给定视频的可见内容，归纳高阶语义（Affordance/Task/Safety/Physics），用于判断 AIGC 视频是否符合世界知识与日常常识（如物体是否突然消失、两个物体突然合并、违背重力、穿模、生物的行为动作是否合理等）请注意输出规则语句精简凝练。

[硬约束]
1) 每条规则必须包含≥1 个 anchors （检测到规则的时间 [t0:, t1:]）和 ≥1 个 required_signals（从信号词表中选），无需在rule_text中显式打印。
2) 仅产出高阶规则：功能/因果/安全/状态变化；禁止低阶：{BLACKLIST}（注：允许“颜色合理性/一致性/光照因果”的高阶规则，但禁止纯“描述颜色”的低阶规则）。
3) 仅输出合法 JSON 数组，遵循“输出Schema”；不输出解释或多余文本，anchors格式为 ["t0": 0.0, "t1": 2.0]，信号词从信号词表中选取，请严格输出。
4) 要求rule_text必须是一段可理解的文本，为语义完整的一段话（不需要有信号词）。
5) 若事件线索与可见帧不一致，以可见帧为准。

[信号词表]
{json.dumps(SIGNAL_LIST, ensure_ascii=False)}

[输出Schema]
{SCHEMA}
""".strip()

    # NEW: 回退同样拼接示例库
    if args.with_examples and EXAMPLE_BANK:
        base += "\n\n[示例（few-shot，统一示例库）]\n" + EXAMPLE_BANK

    return base + "\n\n仅输出 JSON。"

# =================== 全局语义版校验（不依赖时间窗/F子集；不重复定义 SIGNAL_LIST） ===================
ALLOWED_TYPES: Set[str] = {"Affordance", "Task", "Safety", "Physics"}

# -------- 动态“信号白名单（词根）” --------
ALLOWED_SIGNAL_PREFIXES: Set[str] = set()
_printed_allowed_signals = True

# =================== 统一的“词根”提取规则 ===================
import re
from typing import Any, Dict, List, Optional, Set

def _norm_prefix(s: str) -> str:
    """通用规整：小写、非字母数字转下划线、前导数字加前缀。"""
    s = (s or "").strip().lower()
    s = s.replace("/", "_").replace("-", "_").replace(" ", "_").replace(".", "_")
    s = re.sub(r"[^a-z0-9_]", "_", s)
    if not s:
        s = "attr"
    if s[0].isdigit():
        s = "attr_" + s
    return s

def _signal_token(s: str) -> str:
    s = _norm_prefix(s)
    s = s.split("(", 1)[0]
    s = s.split("_", 1)[0]
    s = s.split(":", 1)[0].split(",", 1)[0]
    return s or "attr"

def _list_tokens_from_signal_list(siglist: List[str]) -> Set[str]:
    """从 SIGNAL_LIST 中抽取“词根 token”集合。"""
    out: Set[str] = set()
    for s in siglist:
        if isinstance(s, str):
            out.add(_signal_token(s))
    return out

# ============ 仅针对当前“简化抽取”的辅助：从 basic_semantics 拿 object 与 signature ============
def _iter_basic_objects(sem: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    basic = sem.get("basic_semantics") or {}
    obj_keys: Set[str] = set()
    oc = basic.get("objects_count")
    if isinstance(oc, dict):
        obj_keys |= set(oc.keys())
    attrs = basic.get("attributes")
    if isinstance(attrs, dict):
        obj_keys |= set(attrs.keys())
    for k in sorted(obj_keys):
        name = str(k).strip()
        if name:
            out.append(name)
    return out

def _iter_signature_values(sem: Dict[str, Any]) -> List[str]:
    """
    遍历 basic_semantics.attributes.<obj>.signature 的“值部分”（可为 str 或 list[str]）。
    """
    out: List[str] = []
    basic = sem.get("basic_semantics") or {}
    attrs = basic.get("attributes") or {}
    if not isinstance(attrs, dict):
        return out
    for _obj, amap in attrs.items():
        if not isinstance(amap, dict):
            continue
        sig_val = amap.get("signature", None)
        if sig_val is None:
            continue
        if isinstance(sig_val, (list, tuple, set)):
            for x in sig_val:
                if x is not None:
                    out.append(str(x))
        else:
            out.append(str(sig_val))
    return out

# 仅允许从 object/signature 文本里“顺便补充”的极小词根白名单（避免引入噪声）
_SEM_HINT_TOKENS: Set[str] = {
    "support", "contain", "containment", "balance",
    "shadow", "reflection", "deformation", "fracture",
    "grasp", "spill", "spillage", "collision", "interpenetration",
    "liquid", "level", "monotonic", "trajectory", "accel", "acceleration",
    "rigid", "rigidbody", "illumination", "color", "hue"
}

# =================== 从“SIGNAL_LIST + (objects/signatures)”构建允许的信号前缀 ===================
def build_signal_prefixes_from_sem(sem: Dict[str, Any]) -> Set[str]:
    prefixes: Set[str] = set()

    # (1) SIGNAL_LIST 作为主白名单
    prefixes |= _list_tokens_from_signal_list(SIGNAL_LIST)

    # (2) 启发式补充：从对象名与 signature 文本中，抓取极少量“已允许”的词根
    try:
        texts: List[str] = []
        texts.extend(_iter_basic_objects(sem))
        texts.extend(_iter_signature_values(sem))
        for t in texts:
            t_low = (t or "").lower()
            # 取字母数字片段做粗分词
            for w in re.findall(r"[a-z0-9]+", t_low):
                if w in _SEM_HINT_TOKENS:
                    prefixes.add(_signal_token(w))
    except Exception:
        # 安全兜底：若解析出错，至少保留 SIGNAL_LIST 的前缀集合
        pass

    return prefixes

def build_signal_tokens_union(sem: Dict[str, Any]) -> Set[str]:
    # 直接用上面新版的构造（其内部已含 SIGNAL_LIST）
    return build_signal_prefixes_from_sem(sem)

# -------- 动态“信号白名单（词根）” --------
ALLOWED_SIGNAL_PREFIXES: Set[str] = set()
_printed_allowed_signals = False

def init_allowed_signal_prefixes_from_sem(sem: Dict[str, Any], *, verbose: bool=False, print_limit:int=64) -> None:
    """
    初始化“允许的信号词根”，来源：SIGNAL_LIST ∪ (基于 objects/signatures 的启发式少量补充)。
    """
    global ALLOWED_SIGNAL_PREFIXES, _printed_allowed_signals
    ALLOWED_SIGNAL_PREFIXES = build_signal_tokens_union(sem)
    if verbose and not _printed_allowed_signals:
        names = sorted(list(ALLOWED_SIGNAL_PREFIXES))
        head  = names[:print_limit]
        print(f"[Signals] allowed tokens (root) count = {len(names)}")
        print(f"[Signals] first {len(head)} = {head}{' ...' if len(names) > len(head) else ''}")
        _printed_allowed_signals = True

def debug_print_allowed_signals(print_limit:int=64) -> None:
    names = sorted(list(ALLOWED_SIGNAL_PREFIXES))
    head  = names[:print_limit]
    print(f"[Signals] allowed tokens (root) count = {len(names)}")
    print(f"[Signals] first {len(head)} = {head}{' ...' if len(names) > len(head) else ''}")

# =================== anchors & signals 校验（其余校验保持不变） ===================
def _anchors_ok_global(anchors: List[Dict[str, Any]],
                       *,
                       duration: Optional[float] = None) -> bool:
    """
    仅校验时间：anchors 必须是非空列表；每个元素含数值 t0/t1 且 t1>=t0；
    若给定 duration，则还需 0<=t0,t1<=duration。
    """
    if not isinstance(anchors, list) or not anchors:
        return False
    for a in anchors:
        if not isinstance(a, dict):
            return False
        t0 = a.get("t0", None)
        t1 = a.get("t1", None)
        try:
            t0 = float(t0); t1 = float(t1)
        except Exception:
            return False
        if t1 < t0:
            return False
        if duration is not None:
            if not (0.0 <= t0 <= duration and 0.0 <= t1 <= duration):
                return False
    return True

def _signals_ok(req: List[str], allowed_prefixes: Optional[Set[str]] = None) -> bool:
    if not isinstance(req, list) or len(req) == 0:
        return False
    prefixes = allowed_prefixes if allowed_prefixes is not None else ALLOWED_SIGNAL_PREFIXES
    if not prefixes:
        return False
    for s in req:
        if not isinstance(s, str):
            return False
        token = _signal_token(s)   # 只要词根命中即可
        if token not in prefixes:
            return False
    return True

# -------- 顶层：单条 / 多条规则校验（无难度校验） --------
def validate_rule_global(rule: Dict[str, Any],
                         sem: Dict[str, Any],
                         *,
                         duration: Optional[float] = None,
                         verbose_signals: bool = False,
                         print_limit: int = 64) -> bool:
    if not ALLOWED_SIGNAL_PREFIXES:
        init_allowed_signal_prefixes_from_sem(sem, verbose=verbose_signals, print_limit=print_limit)
    elif verbose_signals:
        debug_print_allowed_signals(print_limit=print_limit)

    if not isinstance(rule, dict):
        return False
    tp = (rule.get("type") or "").strip()
    if tp not in ALLOWED_TYPES:
        return False
    rt = (rule.get("rule_text") or "").strip()
    if not rt:
        return False
    req = rule.get("required_signals") or []
    if not _signals_ok(req):
        return False
    if not _anchors_ok_global(rule.get("anchors") or [], duration=duration):
        return False
    return True


def validate_ruleset_global(rules: Any,
                            sem: Dict[str, Any],
                            *,
                            duration: Optional[float] = None,
                            verbose_signals: bool = False,
                            print_limit: int = 64,
                            renumber: bool = True) -> List[Dict[str, Any]]:
    """
    - 过滤非法/重复规则；重复判定基于 (type, rule_text)
    - 可选重编号为 r1, r2, ...
    """
    init_allowed_signal_prefixes_from_sem(sem, verbose=verbose_signals, print_limit=print_limit)

    if not isinstance(rules, list):
        return []
    seen = set()
    kept: List[Dict[str, Any]] = []
    for r in rules:
        if not isinstance(r, dict):
            continue
        if not validate_rule_global(r, sem,
                                    duration=duration,
                                    verbose_signals=False):  # 避免反复打印
            continue
        key = ((r.get("type") or "").strip(), (r.get("rule_text") or "").strip())
        if key in seen:
            continue
        seen.add(key)
        kept.append(r)

    if renumber:
        for i, rr in enumerate(kept, 1):
            rr["id"] = f"r{i}"

    return kept

def validate_ruleset_fallback(rules: Any,
                              sem: Dict[str, Any],
                              *,
                              verbose_signals: bool = False,
                              print_limit: int = 64,
                              renumber: bool = True) -> List[Dict[str, Any]]:
    init_allowed_signal_prefixes_from_sem(sem, verbose=verbose_signals, print_limit=print_limit)

    if not isinstance(rules, list):
        return []
    seen_rt: Set[str] = set()
    kept: List[Dict[str, Any]] = []
    for r in rules:
        if not isinstance(r, dict):
            continue
        rt = (r.get("rule_text") or "").strip()
        if not rt:
            continue
        req = r.get("required_signals") or []
        if not _signals_ok(req):
            continue

        key = rt.lower()
        if key in seen_rt:
            continue
        seen_rt.add(key)

        kept.append(r)

    if renumber:
        for i, rr in enumerate(kept, 1):
            rr["id"] = f"r{i}"

    return kept


# ---------------- 推理：首轮 + 回退 ----------------
def run_one_window(model,
                   tok,
                   frames: List[Image.Image],
                   tids,
                   win: Tuple[float,float],
                   sem_text_global: str, 
                   max_new_tokens: int,
                   temperature: float,
                   enable_thinking: bool,
                   verbose: bool=False,
                   debug_raw_path: str=None) -> List[Dict[str,Any]]:

    t0, t1 = float(win[0]), float(win[1])
    prompt = build_rules_prompt(sem_text_global, t0, t1)

    msgs = [{'role': 'user', 'content': frames + [prompt]}]
    out = model.chat(
        msgs=msgs, tokenizer=tok,
        temporal_ids=tids, use_image_id=False, max_slice_nums=1,
        do_sample=(temperature>0), temperature=(temperature if temperature>0 else None),
        enable_thinking=enable_thinking, max_new_tokens=max_new_tokens
    )

    text = _strip_think(out if isinstance(out,str) else str(out))
    if verbose:
        print(f"[RAW OUT] {text[:2000]}{'...' if len(text)>2000 else ''}")
    if debug_raw_path:
        try:
            with open(debug_raw_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass

    obj  = _guarded_json(text)
    if not isinstance(obj, list):
        return []

    fixed=[]
    for r in obj:
        try:
            rid = (r.get("id") or "").strip()
            rtype = (r.get("type") or "Physics").strip()
            rule_text = (r.get("rule_text") or "").strip()
            anchors = r.get("anchors") or []
            req = r.get("required_signals") or []
            diff = r.get("difficulty") or {}
            steps = int(max(1, int(diff.get("steps", 1))))
            span_sec = float(max(0.0, float(diff.get("span_sec", max(0.0, t1 - t0)))))
            occ = diff.get("occlusion") or "med"
            fixed.append({
                "id": rid or "r?",
                "type": rtype,
                "rule_text": rule_text,
                "anchors": anchors,
                "required_signals": req,
                "difficulty": {"steps": steps, "span_sec": span_sec, "occlusion": occ},
                "window": {"t0": round(t0,2), "t1": round(t1,2)}  # 仅溯源；校验阶段不用
            })
        except Exception:
            continue
    return fixed


def run_fallback_window(model,
                        tok,
                        frames: List[Image.Image],
                        tids,
                        max_new_tokens: int,
                        temperature: float,
                        enable_thinking: bool,
                        verbose: bool=False,
                        debug_raw_path: str=None) -> List[Dict[str,Any]]:
    """回退推理：不带时间窗、不读取 JSON；仅基于这批帧提问。"""
    prompt = build_fallback_prompt()

    msgs = [{'role': 'user', 'content': frames + [prompt]}]
    out = model.chat(
        msgs=msgs, tokenizer=tok,
        temporal_ids=tids, use_image_id=False, max_slice_nums=1,
        do_sample=(temperature>0), temperature=(temperature if temperature>0 else None),
        enable_thinking=enable_thinking, max_new_tokens=max_new_tokens
    )

    text = _strip_think(out if isinstance(out,str) else str(out))
    if verbose:
        print(f"[RAW OUT FB] {text[:2000]}{'...' if len(text)>2000 else ''}")
    if debug_raw_path:
        try:
            with open(debug_raw_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass

    obj = _guarded_json(text)
    if not isinstance(obj, list):
        return []

    fixed=[]
    for r in obj:
        try:
            rid = (r.get("id") or "").strip()
            rtype = (r.get("type") or "Physics").strip()
            rule_text = (r.get("rule_text") or "").strip()
            anchors = r.get("anchors") or []
            req = r.get("required_signals") or []
            diff = r.get("difficulty") or {}
            steps = int(max(1, int(diff.get("steps", 1))))
            span_sec = float(max(0.0, float(diff.get("span_sec", 0.0))))
            occ = diff.get("occlusion") or "med"
            fixed.append({
                "id": rid or "r?",
                "type": rtype,
                "rule_text": rule_text,
                "anchors": anchors,
                "required_signals": req,
                "difficulty": {"steps": steps, "span_sec": span_sec, "occlusion": occ},
                "source": "fallback"  # 溯源；无 window
            })
        except Exception:
            continue
    return fixed
    
# ====== 新增小工具 ======
_VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm")

def _iter_json_files(json_dir: str) -> List[str]:
    if not json_dir or not os.path.isdir(json_dir):
        return []
    files = []
    for fn in os.listdir(json_dir):
        if fn.lower().endswith(".json"):
            files.append(os.path.join(json_dir, fn))
    return sorted(files)

def _find_video_for_json(stem: str, video_dir: str) -> str | None:
    if not video_dir or not os.path.isdir(video_dir):
        return None
    for fn in os.listdir(video_dir):
        name, ext = os.path.splitext(fn)
        if name == stem and ext.lower() in _VIDEO_EXTS:
            return os.path.join(video_dir, fn)
    return None

def _normalize_stem_for_video(stem: str) -> str:
    s = stem
    s = re.sub(r'(?i)([._-]?evidence(?:[_-]?[A-Za-z0-9]+)?)$', '', s)
    s = re.sub(r'(?i)([._-]?rules)$', '', s)
    return s
# ====== 批量版 main======
def main():
    import sys, os, gc, json, torch, traceback
    from typing import List, Tuple

    # 允许在未显式定义 _normalize_stem_for_video 时退化为原样
    norm_stem_fn = globals().get('_normalize_stem_for_video', lambda s: s)

    # 启动横幅（立刻可见）
    print("[INIT] starting rule generation pipeline...", flush=True)
    print(
        f"[INIT] device={args.device} (cuda_available={torch.cuda.is_available()}), "
        f"dtype={args.dtype}, model_src={args.local_path or args.model_id}",
        flush=True,
    )

    # 输出目录
    _ensure_dir(args.out_dir)

    # 设备/精度
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    dtype = _dtype_of(device, args.dtype)
    where = args.local_path if args.local_path else args.model_id
    is_dir = bool(args.local_path and os.path.isdir(args.local_path))

    # 模型加载
    try:
        print("[LOAD] loading model/tokenizer...", flush=True)
        model, tok = load_model(
            where,
            device,
            dtype,
            args.cache_dir,
            args.local_files_only,
            args.revision,
            is_dir,
        )
        print("[LOAD] model ready.", flush=True)
    except Exception as e:
        print(f"[ERROR] load_model failed: {e}", flush=True)
        traceback.print_exc()
        return

    # -------- 输入来源：优先批量；若未提供 json_dir 则回退到单文件参数 --------
    json_dir = getattr(args, "json_dir", None)
    video_dir = getattr(args, "video_dir", None)

    # 组装待处理清单
    tasks: List[Tuple[str, str]] = []
    if json_dir:
        if not os.path.isdir(json_dir):
            print(f"[ERROR] --json-dir not found: {json_dir}", flush=True)
            return
        if video_dir and not os.path.isdir(video_dir):
            print(f"[ERROR] --video-dir not found: {video_dir}", flush=True)
            return

        # 收集 JSON 列表并打印数量
        json_files = list(_iter_json_files(json_dir))
        if len(json_files) == 0:
            print(f"[WARN] No .json found in: {json_dir}", flush=True)

        for jpath in json_files:
            stem_json = os.path.splitext(os.path.basename(jpath))[0]
            vpath = None
            # 候选基名：原始 + 规范化（去 evidence/rules 等）
            cand_stems = [stem_json]
            norm_stem = norm_stem_fn(stem_json)
            if norm_stem and norm_stem not in cand_stems:
                cand_stems.append(norm_stem)
            # 优先从 video_dir 找同名视频（按候选尝试）
            if video_dir:
                for cand in cand_stems:
                    vpath = _find_video_for_json(cand, video_dir)
                    if vpath:
                        break
            # 若命令行还传了单文件 --video-file，且（原始/规范化后）基名相同也可兜底
            if not vpath and getattr(args, "video_file", None):
                name, _ = os.path.splitext(os.path.basename(args.video_file))
                name_norm = norm_stem_fn(name)
                if name in cand_stems or name_norm in cand_stems:
                    vpath = args.video_file

            if not vpath:
                print(
                    f"[Skip] video not found for json={jpath} (search dir: {video_dir})",
                    flush=True,
                )
                continue

            tasks.append((jpath, vpath))
    else:
        # 兼容单文件模式
        jpath = getattr(args, "json_file", None)
        vpath = getattr(args, "video_file", None)
        if not jpath or not vpath:
            print(
                "[ERROR] Please provide --json-dir and --video-dir for batch, or --json-file and --video-file for single.",
                flush=True,
            )
            return

        # 单文件模式也做一次规范化校验（仅提示，不中断）
        stem_json = os.path.splitext(os.path.basename(jpath))[0]
        name, _ = os.path.splitext(os.path.basename(vpath))
        stem_norm = norm_stem_fn(stem_json)
        name_norm = norm_stem_fn(name)
        if not (name in (stem_json, stem_norm) or name_norm in (stem_json, stem_norm)):
            print(
                f"[WARN] single-file names look mismatched after normalization: "
                f"json='{stem_json}' -> '{stem_norm}', video='{name}' -> '{name_norm}'",
                flush=True,
            )

        tasks.append((jpath, vpath))

    print(f"[INIT] discovered tasks: {len(tasks)}", flush=True)
    if not tasks:
        print("[WARN] No tasks to run.", flush=True)
        return

    # -------- 逐个任务处理 --------
    total_ok = 0
    total_final = 0

    # 新增：严格阈值（满足“总规则>5 且 物理>3”）
    min_total_rules = int(getattr(args, "min_total_rules", 5))     # >5 等价于 >=6
    min_physics_rules = int(getattr(args, "min_physics_rules", 3)) # >3 等价于 >=4
    max_rounds_default = int(getattr(args, "max_fallback_rounds", 12))
    base_fb_fps_default = max(1, int(getattr(args, "fallback_whole_fps", 4)))
    fps_schedule_default = getattr(args, "fallback_fps_schedule", None)  # 可选 list/tuple

    def _count_physics(rules: List[dict]) -> int:
        return sum(1 for r in rules if str(r.get("type", "")).lower() == "physics")

    for ti, (jpath, vpath) in enumerate(tasks, start=1):
        print(
            f"\n========== [{ti}/{len(tasks)}] {os.path.basename(jpath)} ==========",
            flush=True,
        )

        # ---------- 读取 sem ----------
        sem = {}
        try:
            with open(jpath, "r", encoding="utf-8") as f:
                sem = json.load(f)
        except Exception as e:
            print(f"[Warn] load sem failed ({jpath}) -> {e}  (继续，无全局语义)", flush=True)
            sem = {}

        # 初始化本轮白名单（每个 JSON 单独构建，避免跨视频污染）
        try:
            init_allowed_signal_prefixes_from_sem(
                sem, verbose=getattr(args, "verbose", False)
            )
        except Exception as e:
            print(f"[WARN] init signal prefixes failed -> {e}", flush=True)

        # ---------- 打开视频 ----------
        try:
            meta = read_video_meta_and_backend(vpath, args.decode_backend)
        except Exception as e:
            print(f"[Skip] {vpath}: open video failed -> {e}", flush=True)
            continue

        backend, handles, fps, total, duration = meta
        print(
            f"[Video] {vpath} | dur={duration:.2f}s fps≈{fps:.3f} frames={total} backend={backend}",
            flush=True,
        )

        # ---------- 全局语义渲染 ----------
        if sem:
            try:
                sem_payload_global = build_sem_payload_global(sem)
                sem_text_global = render_sem_for_prompt(sem_payload_global)
            except Exception as e:
                print(f"[WARN] render sem failed -> {e}", flush=True)
                sem_text_global = "（无全局语义可用，仅依据窗口帧聚焦）"
        else:
            sem_text_global = "（无全局语义可用，仅依据窗口帧聚焦）"

        # ---------- 生成时间窗 ----------
        windows = []
        t = 0.0
        while t < duration:
            s = t
            e = min(duration, t + args.window_sec)
            if e > s:
                windows.append((s, e))
            t += args.hop_sec
            if e >= duration:
                break
        print(
            f"[Plan] windows={len(windows)} (win={args.window_sec}s, hop={args.hop_sec}s)",
            flush=True,
        )

        # ---------- debug 目录 ----------
        debug_dir = os.path.join(
            args.out_dir, "debug", os.path.splitext(os.path.basename(vpath))[0]
        )
        if args.dump_debug:
            os.makedirs(debug_dir, exist_ok=True)

        # ---------- 主路径 ----------
        rules_main_raw = []
        for wi, (ws, we) in enumerate(windows, start=1):
            try:
                print(
                    f"[Run] window {wi}/{len(windows)} @ {ws:.2f}-{we:.2f}s ...",
                    flush=True,
                )
                frames, tids, _, k = encode_clip(
                    meta,
                    ws,
                    we,
                    args.fps,
                    args.cap_frames,
                    args.resize_short,
                    args.max_packing,
                )
                if k <= 0:
                    print(
                        f"[Run] window {wi}: no frames after sampling, skip.",
                        flush=True,
                    )
                    continue
                raw_path = (
                    os.path.join(
                        debug_dir, f"raw_{wi:03d}_{ws:.2f}-{we:.2f}.txt"
                    )
                    if args.dump_debug
                    else None
                )
                rules = run_one_window(
                    model,
                    tok,
                    frames,
                    tids,
                    (ws, we),
                    sem_text_global=sem_text_global,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    enable_thinking=args.enable_thinking,
                    verbose=args.verbose,
                    debug_raw_path=raw_path,
                )
                print(f"[Run] window {wi}: got {len(rules)} rules.", flush=True)
                rules_main_raw.extend(rules)
            except torch.cuda.OutOfMemoryError as e:
                print(f"[OOM] win#{wi} {ws:.2f}-{we:.2f}: {e}", flush=True)
                _empty_cuda()
            except Exception as e:
                print(f"[WARN] win#{wi} {ws:.2f}-{we:.2f}: {e}", flush=True)

        # ---------- 全局校验/去重/编号（主路径） ----------
        rules_main_ok = validate_ruleset_global(rules_main_raw, sem, duration=duration)
        first_pass_total = len(rules_main_ok)
        print(
            f"[Check] first-pass keep={first_pass_total}/{len(rules_main_raw)}",
            flush=True,
        )

        # ---------- 初次回退（若需要） ----------
        # 若显式 --fallback-whole-video，或主路径仍未达标（总>5 且 物理>3），则触发初次回退
        need_fb = bool(getattr(args, "fallback_whole_video", False)) or not (
            len(rules_main_ok) >= min_total_rules and _count_physics(rules_main_ok) >= min_physics_rules
        )
        rules_fb_ok = []
        if need_fb:
            try:
                fb_fps = base_fb_fps_default
                frames_fb, tids_fb, _, k_fb = encode_clip(
                    meta,
                    0.0,
                    duration,
                    fb_fps,
                    args.cap_frames,
                    args.resize_short,
                    args.max_packing,
                )
                if k_fb > 0:
                    fb_raw_path = os.path.join(debug_dir, "raw_fallback.txt") if args.dump_debug else None
                    rules_fb_raw = run_fallback_window(
                        model,
                        tok,
                        frames_fb,
                        tids_fb,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        enable_thinking=args.enable_thinking,
                        verbose=args.verbose,
                        debug_raw_path=fb_raw_path,
                    )
                    rules_fb_ok = validate_ruleset_fallback(rules_fb_raw, sem,verbose_signals=getattr(args, "verbose", False),print_limit=64,renumber=True)
                    print(f"[Fallback] added {len(rules_fb_ok)} rules.", flush=True)
                else:
                    print("[Fallback] no frames for fallback clip.", flush=True)
            except torch.cuda.OutOfMemoryError as e:
                print(f"[OOM] fallback init: {e}", flush=True)
                _empty_cuda()
            except Exception as e:
                print(f"[WARN] fallback init failed: {e}", flush=True)

        # ---------- 合并（基线） ----------
        rules_final = validate_ruleset_global(rules_main_ok + rules_fb_ok, sem, duration=duration)
        total_valid = len(rules_final)
        physics_ok = _count_physics(rules_final)
        print(
            f"[Final-Stage0] total_valid={total_valid} (first={first_pass_total}, "
            f"fb_added={max(0, len(rules_final)-first_pass_total)}), physics_ok={physics_ok}",
            flush=True,
        )

        # ---------- 核心：持续回退，直到“总>5 且 物理>3” ----------
        max_rounds = max_rounds_default
        base_fb_fps = base_fb_fps_default
        fps_schedule = fps_schedule_default  # list/tuple 或 None
        rounds = 0

        while not (total_valid >= min_total_rules and physics_ok >= min_physics_rules) and rounds < max_rounds:
            rounds += 1
            try:
                # FPS 调度：优先使用自定义序列；否则缓升并封顶 24fps
                fb_fps = (
                    int(fps_schedule[rounds - 1])
                    if isinstance(fps_schedule, (list, tuple)) and rounds - 1 < len(fps_schedule)
                    else min(base_fb_fps * (2 ** ((rounds - 1) // 2)), 24)
                )
                # 温度渐增，避免模式坍缩（上限 0.95）
                retry_temperature = float(min((args.temperature or 0.0) + 0.1 * rounds, 0.95))
                # 抽帧相位抖动：改变起点，避免重复帧序
                jitter = (rounds % 5) / (5.0 * max(fb_fps, 1))
                start_t = min(jitter, max(0.0, duration - (1.0 / max(fb_fps, 1))))

                if args.verbose:
                    print(
                        f"[Fallback-Retry] round={rounds}/{max_rounds} fps={fb_fps} "
                        f"temp={retry_temperature:.2f} start={start_t:.3f}s",
                        flush=True,
                    )

                frames_fb, tids_fb, _, k_fb = encode_clip(
                    meta, start_t, duration, fb_fps,
                    args.cap_frames, args.resize_short, args.max_packing,
                )
                if k_fb <= 0:
                    print(f"[Fallback-Retry] round {rounds}: no frames after sampling, skip.", flush=True)
                    continue

                fb_raw_path = (
                    os.path.join(debug_dir, f"raw_fallback_retry_{rounds:02d}.txt")
                    if args.dump_debug else None
                )
                rules_fb_raw = run_fallback_window(
                    model, tok, frames_fb, tids_fb,
                    max_new_tokens=args.max_new_tokens,
                    temperature=retry_temperature,
                    enable_thinking=args.enable_thinking,
                    verbose=args.verbose,
                    debug_raw_path=fb_raw_path,
                )
                # 校验 + 合并去重
                rules_fb_ok_round = validate_ruleset_fallback(rules_fb_raw, sem,verbose_signals=getattr(args, "verbose", False),print_limit=64,renumber=True)
                before = len(rules_final)
                rules_final = validate_ruleset_global(rules_final + rules_fb_ok_round, sem, duration=duration)
                after = len(rules_final)

                total_valid = len(rules_final)
                physics_ok = _count_physics(rules_final)

                print(
                    f"[Fallback-Retry] round {rounds}: new_valid={len(rules_fb_ok_round)}, "
                    f"merged={after - before}, total_valid={total_valid}/{min_total_rules}, "
                    f"physics_ok={physics_ok}/{min_physics_rules}",
                    flush=True,
                )
            except torch.cuda.OutOfMemoryError as e:
                print(f"[OOM] fallback-retry #{rounds}: {e}", flush=True)
                _empty_cuda()
            except Exception as e:
                print(f"[WARN] fallback-retry #{rounds} failed: {e}", flush=True)

        if not (total_valid >= min_total_rules and physics_ok >= min_physics_rules):
            print(
                f"[WARN] Still below targets after {rounds} retries: "
                f"total_valid={total_valid} (need>{min_total_rules-1}), "
                f"physics_ok={physics_ok} (need>{min_physics_rules-1}). "
                f"可尝试增大 --max-fallback-rounds / --fallback-whole-fps，或放宽校验。",
                flush=True,
            )

        # ---------- 最终统计打印 ----------
        print(
            f"[Final] total_valid={total_valid} (first={first_pass_total}, "
            f"fb_added={max(0, len(rules_final) - first_pass_total)}), "
            f"physics_ok={physics_ok}",
            flush=True,
        )

        # ---------- 写出 ----------
        out_obj = {
            "meta": {
                "video": os.path.abspath(vpath),
                "json": os.path.abspath(jpath) if sem else "",
                "model": where,
                "dtype": args.dtype,
                "window_sec": args.window_sec,
                "hop_sec": args.hop_sec,
                "fps_per_window": args.fps,
                "resize_short": args.resize_short,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "with_examples": bool(getattr(args, "with_examples", False)),
                "duration": round(float(duration), 3),
                "first_pass_total": first_pass_total,

                # 兼容信息
                "fallback_whole_video": bool(getattr(args, "fallback_whole_video", False)),
                "global_fallback_threshold": int(getattr(args, "global_fallback_threshold", 3)),

                # 关键统计
                "fallback_added": max(0, len(rules_final) - first_pass_total),
                "final_total": len(rules_final),
                "physics_final": int(physics_ok),

                # 新增：严格门槛记录（总>5，物理>3）
                "min_total_rules": int(min_total_rules),
                "min_physics_rules": int(min_physics_rules),
                "max_fallback_rounds": int(max_rounds_default),
                "base_fallback_fps": int(base_fb_fps_default),
            },
            "rules": rules_final,
        }
        out_name = f"{os.path.splitext(os.path.basename(vpath))[0]}_rules.json"
        out_path = os.path.join(args.out_dir, out_name)
        _ensure_dir(os.path.dirname(out_path) or ".")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        print(
            f"[SAVE] {out_path} | first={first_pass_total} final={len(rules_final)} physics={physics_ok}",
            flush=True,
        )

        total_ok += first_pass_total
        total_final += len(rules_final)

        # ---------- 资源清理 ----------
        try:
            if backend == "cv2" and handles[0] is not None:
                handles[0].release()
        except Exception:
            pass
        _empty_cuda()
        gc.collect()

    print(
        f"\n[SUMMARY] tasks={len(tasks)} | first_pass_total={total_ok} | final_total={total_final}",
        flush=True,
    )

if __name__ == '__main__':
    import os, warnings, traceback
    # 运行期环境与警告抑制
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        os.environ.setdefault(
            'PYTORCH_CUDA_ALLOC_CONF',
            'expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6'
        )
        try:
            print("[ENTRY] launching main() ...", flush=True)
            main()
            print("[ENTRY] main() finished.", flush=True)
        except Exception as e:
            # 关键：把异常完整打出来，方便排查“没有反应”的根因
            print(f"[FATAL] Unhandled exception: {e}", flush=True)
            traceback.print_exc()
            raise





