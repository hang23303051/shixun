# -*- coding: utf-8 -*-
"""
MiniCPM-V-4.5 视频细粒度抽取 —— 单卡稳定版（多窗稳健 | 开放词表 | 回退兜底）
- 多回合：A(对象计数) → B(细粒度属性) → C(关系) → V(一致性校正) → F(细粒度补全)
- 单轮回退：若主流程为空，触发一次“富属性+弱先验”的单轮提示，避免空输出
- 跨窗追踪：signature(颜色/花纹/标记/方位) + 属性Jaccard + 时间重叠/临近；更保守合并阈值，防止重复计数
- 输出：全局 fine + per-window fine_windows + 视图汇总
- 解码后端：cv2 / decord（默认优先 cv2）
- Attention 稳态：可禁用 Flash-SDP / 强制 math-SDP
- OOM 自适应：max_new_tokens -> fps -> cap_frames -> resize_short -> max_packing
- 严格 JSON 守卫 + 失败回退（强制 JSON + 关闭思考）
- 批量模式：--batch-from（目录 / .txt / .json）+ --out-dir
- 可选场景切分：--scene-split 开启后，每个场景作为单窗推理（不开启则用原滑窗）
"""

import os, re, json, math, argparse, warnings, inspect
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

import torch
from transformers import AutoModel, AutoTokenizer

# ---------------- CLI ----------------
p = argparse.ArgumentParser()
p.add_argument('--debug-dump', default='', help='调试：把每个窗口A/B/C/V/F/FB的原始文本落盘到该目录')

# 单文件模式
p.add_argument('--video', default='', help='单个视频路径')
p.add_argument('--out',   default='', help='单个视频的输出 JSON 路径')

# 批量模式
p.add_argument('--batch-from', default='', help='目录 / .txt(.list) 文本 / .json 列表（批量模式）')
p.add_argument('--out-dir',    default='', help='批量模式输出目录（必填）')

# 模型与缓存
p.add_argument('--local-path', default='', help='本地模型目录，如 /data/hf_home/models/openbmb__MiniCPM-V-4_5 或 -int4')
p.add_argument('--model-id',   default='openbmb/MiniCPM-V-4_5')
p.add_argument('--revision',   default='main')
p.add_argument('--cache-dir',  default='')
p.add_argument('--local-files-only', action='store_true')

# 设备/精度/注意力
p.add_argument('--device', default='cuda', choices=['cuda','cpu'])
p.add_argument('--dtype', default='bf16', choices=['bf16','fp16','fp32'])
p.add_argument('--disable-flash-sdp', action='store_true', help='禁用 Flash SDP（推荐稳定）')
p.add_argument('--force-math-sdp', action='store_true', help='强制 math SDP（最稳，但更慢）')

# 滑窗/采样/解码
p.add_argument('--window-sec', type=float, default=6.0)
p.add_argument('--hop-sec',    type=float, default=3.0)
p.add_argument('--fps', type=int, default=3)
p.add_argument('--cap-frames', type=int, default=180)
p.add_argument('--resize-short', type=int, default=448)
p.add_argument('--max-packing', type=int, default=3)
p.add_argument('--decode-backend', default='auto', choices=['auto','cv2','decord'])

# 生成与思考
p.add_argument('--max-new-tokens', type=int, default=256)
p.add_argument('--min-max-new-tokens', type=int, default=96)
p.add_argument('--temperature', type=float, default=0.0)
p.add_argument('--enable-thinking', action='store_true')
p.add_argument('--verbose', action='store_true')

# 稳定性
p.add_argument('--min-span-sec', type=float, default=0.3, help='最短有效时间段（秒），更短将被过滤')

# OOM 自适应
p.add_argument('--min-fps', type=int, default=2)

# 场景切分（可选）
p.add_argument('--scene-split', action='store_true', help='启用简易场景分割：每个场景作为单窗推理，关闭则用滑窗')
p.add_argument('--scene-thres', type=float, default=0.6, help='场景切分阈值（0.5~0.9）')
p.add_argument('--scene-sample-fps', type=float, default=2.0, help='场景检测采样FPS（仅用于切分）')
p.add_argument('--scene-min-sec', type=float, default=1.0, help='最短场景时长（秒），短于此会并入前一段')

args = p.parse_args()

# ---------- 常量 ----------
TIME_SCALE = 0.1
MAX_NUM_FRAMES = 180
MAX_NUM_PACKING = max(1, min(args.max_packing, 6))
VIDEO_EXTS = {'.mp4','.mkv','.mov','.avi','.m4v','.webm','.flv','.ts','.mpg','.mpeg','.wmv'}
PROX_GAP = 1.0  # 跨窗时间临近阈值（秒）
MIN_REL_SPAN = max(0.05, float(args.min_span_sec) * 0.5)  # 关系最小时长（独立于实体）

# [COUNT FIX] 群体类与复数映射（仅用于显示，不影响内部 canonical 名）
HERD_CLASSES = {"person","cow","sheep","goat","deer","bird","fish","duck","chicken","horse"}
PLURAL_MAP = {
    "person":"people","man":"men","woman":"women","child":"children",
    "cow":"cows","sheep":"sheep","goat":"goats","deer":"deer",
    "bird":"birds","fish":"fish","duck":"ducks","chicken":"chickens","horse":"horses"
}

def _display_plural(name:str, count:int)->str:
    base = _canon_name(name)
    if count >= 10:
        return PLURAL_MAP.get(base, base+"s")
    return base

# ---------- 工具 ----------
def _norm_token(s:str) -> str:
    return (s or "").strip().lower().replace(" ", "-")

def _canon_name(n:str) -> str:
    n=_norm_token(n)
    syn={'people':'person','men':'person','women':'person','man':'person','woman':'person',
         'boy':'person','girl':'person','bike':'bicycle','motorbike':'motorcycle','pickup':'truck',
         'cattle':'cow','bull':'cow','calf':'cow','ox':'cow'}
    return syn.get(n,n)

def _jaccard(a:set,b:set) -> float:
    if not a and not b: return 1.0
    return len(a & b) / max(1, len(a | b))

def _map_to_nearest_scale(values, scale):
    tree=cKDTree(np.asarray(scale)[:,None]); _, idx=tree.query(np.asarray(values)[:,None])
    return np.asarray(scale)[idx]

def _group_array(arr, size):
    return [arr[i:i+size] for i in range(0,len(arr),size)]

def _dtype_of(device,choice):
    if device=='cpu': return torch.float32
    return {'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32}.get(choice, torch.bfloat16)

def _guarded_json(s:str):
    if not isinstance(s, str): return None, 'not_str'
    s = s.strip()
    if not s: return None, 'empty'
    try:
        return json.loads(s), None
    except Exception as e:
        m=re.search(r'\{.*\}', s, flags=re.S)
        if m:
            try: return json.loads(m.group(0)), None
            except Exception as e2: return None, f'parse_fail_after_brace:{e2}'
        return None, f'parse_fail:{e}'

def _strip_think(text:str):
    if not isinstance(text,str): return text
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S|re.I)
    text = re.sub(r'^\s*<think>.*$', '', text, flags=re.S|re.I)
    return text.strip()

def _empty_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

# ====== 仅做“单字母/非字母”清理（无词表限制） ======
def _clean_open_attrs(attrs: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for k, vs in (attrs or {}).items():
        k2 = _norm_token(k)
        keep = []
        for v in (vs or []):
            if not isinstance(v, str):
                continue
            t = _norm_token(v)
            if not t:
                continue
            if not re.search(r'[a-z]', t):
                continue
            if len(re.sub(r'[^a-z]', '', t)) == 1 and len(t) <= 2:
                continue
            keep.append(t)
        if keep:
            out[k2] = sorted(set(keep))
    return out

# ---------- 抽签名（用于跨窗追踪） ----------
def _mk_signature_from_attrs(name:str, attrs:Dict[str,List[str]]) -> str:
    cues=[]
    for k in ['color','pattern','printed-text','number-or-id','brand-or-logo','texture','species-or-breed']:
        cues += (attrs.get(k) or [])[:2]
    for k in ['position','orientation','facing-direction']:
        if (attrs.get(k) or []):
            cues.append(attrs[k][0])
    cues = [_norm_token(x) for x in cues if isinstance(x,str) and x.strip()]
    if not cues:
        return _norm_token(name)
    cues = sorted(set(cues))[:4]
    return "-".join(cues)

# ---------- NEW: spans 形态纠正 ----------
def _coerce_spans(x):
    """将任意形态的 spans 转为 [[a,b], ...] 浮点二维列表。"""
    if isinstance(x, (int, float, str)) or x is None:
        return []
    if isinstance(x, (list, tuple)):
        if len(x) == 2 and all(isinstance(t, (int, float)) for t in x):
            return [[float(x[0]), float(x[1])]]
        out = []
        for it in x:
            if isinstance(it, (list, tuple)) and len(it) >= 2:
                a, b = it[0], it[1]
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    out.append([float(a), float(b)])
        return out
    return []

# ---------- span 工具 ----------
def _span_total(spans: List[List[float]]) -> float:
    return sum(max(0.0, float(b)-float(a)) for a,b in (spans or []))

def _span_intersection(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    inter=[]
    for a1,b1 in (A or []):
        for a2,b2 in (B or []):
            s=max(a1,a2); e=min(b1,b2)
            if e>s: inter.append([s,e])
    return _merge_spans(inter)

def _nearest_midpair(A: List[List[float]], B: List[List[float]]):
    best_gap=1e9; best=None
    for a1,b1 in (A or []):
        for a2,b2 in (B or []):
            if b1<=a2:
                gap=a2-b1; pair=(b1,a2)
            elif b2<=a1:
                gap=a1-b2; pair=(b2,a1)
            else:
                return None
            if gap<best_gap:
                best=pair; best_gap=gap
    return best

def _rel_span_fallback(sub_sp: List[List[float]], obj_sp: List[List[float]],
                       start_s: float, end_s: float, min_rel: float) -> List[List[float]]:
    inter = _span_intersection(sub_sp, obj_sp)
    if inter:
        kept = [[round(max(start_s,a),1), round(min(end_s,b),1)] for a,b in inter if (b-a)>=min_rel]
        if kept: return kept
        a,b = inter[0]
        mid=(a+b)/2.0
    else:
        pair=_nearest_midpair(sub_sp, obj_sp)
        if not pair: return []
        mid=(pair[0]+pair[1])/2.0
    half=min_rel/2.0
    a=max(start_s, mid-half); b=min(end_s, mid+half)
    if b>a:
        return [[round(a,1), round(b,1)]]
    return []

# ---------- 批量输入解析 ----------
def _discover_inputs(batch_from:str, out_dir:str):
    """返回 [(video_path, out_path), ...]"""
    items=[]
    if not batch_from:
        return items
    if os.path.isdir(batch_from):
        for root, _, fns in os.walk(batch_from):
            for fn in sorted(fns):
                fp=os.path.join(root, fn)
                if os.path.isfile(fp) and os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
                    rel=os.path.relpath(fp, batch_from)
                    base=os.path.splitext(rel)[0] + '.json'
                    items.append((fp, os.path.join(out_dir, base)))
        return items
    ext=os.path.splitext(batch_from)[1].lower()
    if ext in ('.txt','.list'):
        with open(batch_from,'r',encoding='utf-8') as f:
            for line in f:
                line=line.strip()
                if not line or line.startswith('#'): continue
                parts=re.split(r'\s+', line, maxsplit=1)
                vin=parts[0]
                if len(parts)==2:
                    vout=parts[1]
                else:
                    base=os.path.splitext(os.path.basename(vin))[0]+'.json'
                    vout=os.path.join(out_dir, base)
                items.append((vin, vout))
        return items
    if ext=='.json':
        with open(batch_from,'r',encoding='utf-8') as f:
            obj=json.load(f)
        if isinstance(obj, list):
            for it in obj:
                if isinstance(it,str):
                    base=os.path.splitext(os.path.basename(it))[0]+'.json'
                    items.append((it, os.path.join(out_dir, base)))
                elif isinstance(it,dict) and 'video' in it:
                    vout=it.get('out')
                    if not vout:
                        base=os.path.splitext(os.path.basename(it['video']))[0]+'.json'
                        vout=os.path.join(out_dir, base)
                    items.append((it['video'], vout))
        elif isinstance(obj, dict) and 'videos' in obj and isinstance(obj['videos'], list):
            for it in obj['videos']:
                if isinstance(it,str):
                    base=os.path.splitext(os.path.basename(it))[0]+'.json'  # 修正括号次序
                    items.append((it, os.path.join(out_dir, base)))
                elif isinstance(it,dict) and 'video' in it:
                    vout=it.get('out')
                    if not vout:
                        base=os.path.splitext(os.path.basename(it['video']))[0]+'.json'
                        vout=os.path.join(out_dir, base)
                    items.append((it['video'], vout))
        return items
    base=os.path.splitext(os.path.basename(batch_from))[0]+'.json'
    items.append((batch_from, os.path.join(out_dir, base)))
    return items

# ---------- 输出路径一致性矫正 ----------
def _sanitize_out(vin:str, vout:str, out_dir:str) -> str:
    """强制将输出文件名校正为 basename(vin).json，目录优先采用 vout 的目录；若为空则落到 out_dir。"""
    expect = os.path.splitext(os.path.basename(vin))[0] + '.json'
    parent = os.path.dirname(vout) or out_dir or "."
    return os.path.join(parent, expect)

# ---------- 解码 ----------
def read_video_meta_and_backend(video_path:str, preference:str='auto'):
    cap = None
    vr  = None
    backend=None; fps=None; total=None; duration=None
    if preference in ('auto','cv2'):
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            duration = total/max(fps,1e-6) if total>0 else 0.0
            if total>0:
                backend='cv2'
                return backend, (cap, None), fps, total, duration
        except Exception:
            pass
    if preference in ('auto','decord'):
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            total=len(vr)
            try: fps=float(vr.get_avg_fps())
            except Exception: fps=25.0
            duration=total/max(fps,1e-6)
            backend='decord'
            return backend, (None, vr), fps, total, duration
        except Exception:
            pass
    raise RuntimeError("No available video backend (tried cv2, decord)")

def read_frames(backend, handles, idx_all):
    frames=[]
    if backend=='cv2':
        cap,_ = handles
        import cv2
        try: cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except Exception: pass
        i=0; want=set(idx_all.tolist())
        while True:
            ok, frame = cap.read()
            if not ok: break
            if i in want:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
                if len(frames)>=len(idx_all): break
            i+=1
    else:
        _,vr = handles
        arr=vr.get_batch(idx_all.tolist()).asnumpy()
        frames=[Image.fromarray(x.astype('uint8')).convert('RGB') for x in arr]
    return frames

# ---------- 抽帧 ----------
def encode_clip(meta, start_s:float, end_s:float, choose_fps:int, cap_frames:int, resize_short:int, max_packing:int):
    backend, handles, fps, total, duration = meta
    s=max(0.0,start_s); e=min(duration,end_s)
    if e<=s: e=min(duration,s+1.0)
    clip_len=e-s
    if choose_fps * int(clip_len) <= MAX_NUM_FRAMES:
        packing_nums=1
        choose_frames=round(min(choose_fps, round(fps)) * min(MAX_NUM_FRAMES, clip_len))
    else:
        packing_nums=math.ceil(clip_len*choose_fps/MAX_NUM_FRAMES)
        if packing_nums<=max_packing:
            choose_frames=round(clip_len * choose_fps)
        else:
            choose_frames=round(MAX_NUM_FRAMES * max_packing)
            packing_nums=max_packing
    choose_frames=int(min(choose_frames, max(1, cap_frames)))
    start_idx=int(round(s*fps)); end_idx=max(0, int(round(e*fps))-1)
    end_idx=min(total-1, end_idx)
    idx_all=np.linspace(start_idx, end_idx, choose_frames).astype(np.int64)

    ts=idx_all / max(fps,1e-6)
    scale=np.arange(0, duration+1e-6, TIME_SCALE)
    tids=(_map_to_nearest_scale(ts, scale)/TIME_SCALE).astype(np.int32)
    tids=_group_array(tids, packing_nums)

    frames = read_frames(backend, handles, idx_all)
    if resize_short and resize_short>0:
        nf=[]
        for img in frames:
            w,h=img.size; short=min(w,h)
            if short>resize_short:
                if w<h:
                    new_w=resize_short; new_h=int(h*(resize_short/w))
                else:
                    new_h=resize_short; new_w=int(w*(resize_short/h))
                img=img.resize((new_w,new_h), Image.BILINEAR)
            nf.append(img)
        frames=nf
    return frames, tids, (s,e), len(idx_all)

# ---------- 提示词 ----------
def build_prompts(min_span:float=0.3):
    facets = (
        "color, pattern, texture, material, size, age, sex, state, pose, action, "
        "orientation, facing-direction, position, object-part, tool-or-instrument, equipment, "
        "species-or-breed, vehicle-type, food-type, brand-or-logo, printed-text, number-or-id, "
        "art-medium, style, weather, lighting, scene, camera-view"
    )

    # A：对象与计数
    PA = (
        "你将看到覆盖绝对时间 [START, END] 秒的一组帧。\n\n"
        "请问本片段里有哪些对象？请你按“类名 + 数量 + 稳定时间段(0.1s)”准确回答。\n"
        "要求：name 用简洁英文单数（如 person/dog/cat/bird/cow/car/horse 等）；若同类对象数量明显超过 8 个，请改用英文复数（如 ants/people），"
        "并把 count 写成字符串 \">8\"；多镜头时按整段唯一体保守计数；spans 在 [START,END] 内，相邻(≤0.2s)可合并，忽略 <0.1s 噪声。\n\n"
        "只输出一个 JSON：\n"
        "{\n"
        '  "objects":[\n'
        '    {"name":"cow","count":2,"spans":[[ABS_S,ABS_E],...]},\n'
        '    {"name":"person","count":1,"spans":[[ABS_S,ABS_E]]},\n'
        '    {"name":"ants","count":">8","spans":[[ABS_S,ABS_E]]}\n'
        "  ]\n"
        "}"
    )

    # B：实例与属性
    PB = (
        "问：逐个“具体实例”输出属性与时间段。尽可能完整枚举可见实例（不受A中的数量限制）。\n"
        "可见则填属性，值用小写连字符（如 black-white、left-facing）；不可见不猜测。"
        "为跨窗追踪，给出短 'signature'（≥2 稳定线索，如颜色/花纹/编号/文字/方位）。\n"
        f"建议槽位（开放词表）：{facets}\n\n"
        "只输出一个 JSON：\n"
        "{\n"
        '  "entities":[\n'
        '    {\n'
        '      "id":"e1","name":"cow","signature":"black-white-left-ear-tag-37",\n'
        '      "attributes":{"color":["black","white"],"action":["grazing"],"position":["left"],"number-or-id":["37"]},\n'
        '      "spans":[[ABS_S,ABS_E],...]\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    # C：关系三元组
    PC = (
        "问：基于 B 的实体，给出清晰可见的关系三元组（含时间段与置信度）。\n"
        "可用谓词（开放示例）：left-of/right-of/above/below/front-of/behind/over/under/inside/overlapping/next-to/"
        "holding/carrying/touching/looking-at/feeding/chasing/following/riding/pulling/pushing/"
        "passing/crossing/entering/exiting/throwing/catching/drinking/eating\n\n"
        "只输出一个 JSON：\n"
        '{\n  "relations":[{"subject":"e1","predicate":"left-of","object":"e2","spans":[[ABS_S,ABS_E]],"confidence":0.8}]\n}'
        "\n要求：subject/object 必须是 B.entities 的 id；spans 在 [START,END] 内，精度 0.1s；相邻(≤0.2s)可合并。"
    )

    # V：一致性校正与时间规整
    PV = (
        "一致性校正（只输出最终 JSON）：输入含 A/B/C。\n"
        "1) 删除不确定对象与关系；修复无效 id。\n"
        "2) 合并重叠或相邻(≤0.2s)时间段；删除短于 __MIN_SPAN__s 的时间段；截断到 [START,END]。\n"
        "3) 仅输出：\n"
        "{\n"
        '  "entities":[{"id":"e1","name":"cow","signature":"...","attributes":{...},"spans":[[ABS_S,ABS_E],... ]}, ...],\n'
        '  "relations":[{"subject":"e1","predicate":"next-to","object":"e2","spans":[[ABS_S,ABS_E]],"confidence":0.8}, ...]\n'
        "}"
    ).replace("__MIN_SPAN__", f"{min_span:.1f}")

    # F：细粒度补全
    PF = (
        "细粒度补全（只输出最终 JSON）：在不改变 spans 的前提下，补充可能遗漏的细粒度属性（开放词表），注重："
        "logo/招牌/文字、号码或标识、动物花纹/耳标、服饰/防护件、车辆子类与状态、场景/光照/机位等；仅去重补齐。"
    )

    # FB：单轮回退（空结果兜底；弱先验）
    FB = (
        "You are a careful video understanding expert. Frames cover [START, END] sec.\n"
        "Return ONE JSON only. No extra text.\n\n"
        "Extract visible object instances and relations. Prefer concise English singular names.\n"
        "Animals you might see (if visible): cow/cattle/ox/bull/calf, horse, sheep, goat, dog, cat, bird, chicken, pig, deer.\n"
        "Attributes (open set): color, pattern, texture, material, size, age, sex, state, pose, action, orientation, facing-direction, position,\n"
        "species-or-breed, printed-text, number-or-id, brand-or-logo, scene, weather, lighting, camera-view.\n"
        "Each entity must include a short 'signature' (>=2 stable cues like colors/pattern/mark/ear-tag/side left/right).\n\n"
        "Schema:\n"
        "{\n"
        '  "entities":[\n'
        '    {"id":"e1","name":"cow","signature":"black-white-left-ear-tag-37",\n'
        '     "attributes":{"color":["black","white"],"action":["grazing"],"position":["left"],"number-or-id":["37"]},\n'
        '     "spans":[[ABS_S,ABS_E]]}\n'
        "  ],\n"
        '  "relations":[{"subject":"e1","predicate":"left-of","object":"e2","spans":[[ABS_S,ABS_E]],"confidence":0.7}]\n'
        "}\n"
        f"ONLY output the JSON; times within [START,END], 0.1s precision; merge adjacent (<=0.2s); drop spans shorter than {min_span:.1f}s."
    )

    return PA, PB, PC, PV, PF, FB

# ---------- 时长过滤 ----------
def _filter_spans(spans, start_s, end_s, min_span:float, merge_tol:float=0.2):
    segs=[]
    for a,b in (spans or []):
        try:
            a=float(a); b=float(b)
            a=max(a, start_s); b=min(b, end_s)
            if b>a: segs.append([a,b])
        except:
            continue
    if not segs:
        return []
    segs.sort(key=lambda x: x[0])
    merged=[segs[0]]
    for a,b in segs[1:]:
        if a <= merged[-1][1] + merge_tol:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a,b])
    out = [[round(a,1), round(b,1)] for a,b in merged if (b-a) >= min_span]
    if not out:
        thr = max(0.1, min_span*0.5)
        out = [[round(a,1), round(b,1)] for a,b in merged if (b-a) >= thr]
    return out

# ---------- 模型加载 ----------
def _dtype_kw(d):
    sig=inspect.signature(AutoModel.from_pretrained)
    return {"dtype": d} if "dtype" in sig.parameters else {"torch_dtype": d}

def load_model(where:str, device:str, dtype, cache_dir:str, local_only:bool, revision:str, is_dir:bool):
    common=dict(trust_remote_code=True, attn_implementation='sdpa', **_dtype_kw(dtype))
    if cache_dir: common['cache_dir']=cache_dir
    if local_only: common['local_files_only']=True
    if is_dir:
        m=AutoModel.from_pretrained(where, **common).eval()
        t=AutoTokenizer.from_pretrained(where, trust_remote_code=True, local_files_only=local_only)
    else:
        m=AutoModel.from_pretrained(where, revision=revision, **common).eval()
        t=AutoTokenizer.from_pretrained(where, revision=revision, trust_remote_code=True, **({} if not cache_dir else {'cache_dir': cache_dir}))
    if device=='cuda' and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
        if args.disable_flash_sdp or args.force_math_sdp:
            try:
                torch.backends.cuda.enable_flash_sdp(False)
                if args.force_math_sdp:
                    torch.backends.cuda.enable_math_sdp(True)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                else:
                    torch.backends.cuda.enable_math_sdp(True)
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
            except Exception:
                pass
        m=m.to('cuda')
    return m,t

# ---------- 单窗推理（含回退+计数兜底） ----------
def run_clip(model, tok, frames, tids, start_s, end_s,
             max_new_tokens:int, temperature:float, enable_thinking:bool,
             fps:int, cap_frames:int, max_packing:int, resize_short:int,
             prompts:Tuple[str,str,str,str,str,str], min_span_sec:float):

    PA, PB, PC, PV, PF, FB = prompts

    def _chat(prompt:str, thinking:bool):
        p = prompt.replace('[START]', f'{start_s:.1f}').replace('[END]', f'{end_s:.1f}')
        msgs=[{'role':'user','content': frames + [p]}]
        out = model.chat(
            msgs=msgs, tokenizer=tok,
            temporal_ids=tids, use_image_id=False, max_slice_nums=1,
            do_sample=(temperature>0), temperature=(temperature if temperature>0 else None),
            enable_thinking=thinking, max_new_tokens=max_new_tokens
        )
        return _strip_think(out)

    STRICT_SUFFIX = "\n严格要求：仅输出**一个**JSON对象，禁止任何解释、注释或多余字符；若无法判断，输出上面示例的空结构。"

    def _list_of_dicts(x):
        if isinstance(x, list):
            return [it for it in x if isinstance(it, dict)]
        return []

    def _ask_and_parse(prompt: str, empty_schema: dict, allow_thinking: bool, tag: str = ""):
        txt1 = _chat(prompt, thinking=allow_thinking)
        if args.debug_dump:
            os.makedirs(args.debug_dump, exist_ok=True)
            with open(os.path.join(args.debug_dump, f"win_{start_s:.1f}_{end_s:.1f}_{tag}_1.txt"), "w", encoding="utf-8") as f:
                f.write(str(txt1))
        obj, _ = _guarded_json(txt1)
        if isinstance(obj, dict):
            return obj
        txt2 = _chat(prompt + STRICT_SUFFIX, thinking=False)
        if args.debug_dump:
            with open(os.path.join(args.debug_dump, f"win_{start_s:.1f}_{end_s:.1f}_{tag}_2.txt"), "w", encoding="utf-8") as f:
                f.write(str(txt2))
        obj2, _ = _guarded_json(txt2)
        if isinstance(obj2, dict):
            return obj2
        return empty_schema

    cur_tokens=max_new_tokens; cur_temp=temperature
    while True:
        try:
            A = _ask_and_parse(PA, {"objects": []}, allow_thinking=enable_thinking, tag="A")

            # 提取 A 的计数提示
            hint_from_A = {}
            try:
                for it in (A.get("objects") or []):
                    if not isinstance(it, dict):  # 守卫
                        continue
                    cname = _canon_name(it.get("name",""))
                    cnt_raw = it.get("count", 0)
                    try:
                        cnt = int(float(cnt_raw))
                    except Exception:
                        cnt = 0
                    if cname:
                        hint_from_A[cname] = max(hint_from_A.get(cname, 0), max(0, cnt))
            except Exception:
                hint_from_A = {}

            ref=json.dumps({"objects":A.get("objects",[])}, ensure_ascii=False)

            B = _ask_and_parse(PB+"\n\n"+ref, {"entities":[]}, allow_thinking=enable_thinking, tag="B")
            ents=json.dumps({"entities":B.get("entities",[])}, ensure_ascii=False)

            C = _ask_and_parse(PC+"\n\n"+ents, {"relations":[]}, allow_thinking=enable_thinking, tag="C")

            payload=json.dumps({"A":A,"B":B,"C":C}, ensure_ascii=False)
            V = _ask_and_parse(PV+"\n\n"+payload,
                               {"entities":B.get("entities",[]), "relations":C.get("relations",[])},
                               allow_thinking=enable_thinking, tag="V")

            V2 = _ask_and_parse(
                PF+"\n\n"+json.dumps({"entities":V.get("entities",[]), "relations":V.get("relations",[])}, ensure_ascii=False),
                {"entities":V.get("entities",[]), "relations":V.get("relations",[])},
                allow_thinking=enable_thinking, tag="F"
            )

            ents_out=[]
            for e in _list_of_dicts(V2.get("entities", [])):
                spans=_filter_spans(_coerce_spans(e.get("spans")), start_s, end_s, min_span_sec)
                if not spans:
                    continue

                raw_attrs_obj = e.get("attributes") if isinstance(e.get("attributes"), dict) else {}
                raw_attrs = { _norm_token(k): [ _norm_token(x) for x in (vs or []) if isinstance(x,str) ]
                              for k,vs in raw_attrs_obj.items() }
                attrs = _clean_open_attrs(raw_attrs)

                name_raw = e.get("name","")
                name = _canon_name(name_raw if isinstance(name_raw, str) else "")

                sig_raw = e.get("signature","")
                sig = _norm_token(sig_raw if isinstance(sig_raw, str) else "") or _mk_signature_from_attrs(name, attrs)
                if sig:
                    attrs.setdefault("signature", [sig])

                ents_out.append({
                    "id": str(e.get("id","")),
                    "name": name,
                    "attributes": attrs,
                    "spans": spans
                })

            rels_out=[]
            for r in _list_of_dicts(V2.get("relations", [])):
                spans=_filter_spans(_coerce_spans(r.get("spans")), start_s, end_s, min_span_sec)
                if not spans:
                    continue
                conf_raw = r.get("confidence", 0.0)
                try:
                    conf_val = float(conf_raw)
                except Exception:
                    conf_val = 0.0
                rels_out.append({
                    "subject": str(r.get("subject","")),
                    "predicate": _norm_token(r.get("predicate","")),
                    "object": str(r.get("object","")),
                    "spans": spans,
                    "confidence": conf_val
                })

            # 若主流程空，触发一次 FB；再不行则计数兜底
            if not ents_out:
                FB_obj = _ask_and_parse(prompts[-1], {"entities":[],"relations":[]}, allow_thinking=enable_thinking, tag="FB")

                for e in _list_of_dicts(FB_obj.get("entities", [])):
                    spans=_filter_spans(_coerce_spans(e.get("spans")), start_s, end_s, min_span_sec)
                    if not spans:
                        continue
                    raw_attrs_obj = e.get("attributes") if isinstance(e.get("attributes"), dict) else {}
                    raw_attrs = { _norm_token(k): [ _norm_token(x) for x in (vs or []) if isinstance(x,str) ]
                                  for k,vs in raw_attrs_obj.items() }
                    attrs = _clean_open_attrs(raw_attrs)
                    name_raw = e.get("name","")
                    name = _canon_name(name_raw if isinstance(name_raw, str) else "")
                    sig_raw = e.get("signature","")
                    sig = _norm_token(sig_raw if isinstance(sig_raw, str) else "") or _mk_signature_from_attrs(name, attrs)
                    if sig:
                        attrs.setdefault("signature", [sig])
                    ents_out.append({
                        "id": str(e.get("id","")),
                        "name": name,
                        "attributes": attrs,
                        "spans": spans
                    })

                for r in _list_of_dicts(FB_obj.get("relations", [])):
                    spans=_filter_spans(_coerce_spans(r.get("spans")), start_s, end_s, min_span_sec)
                    if not spans:
                        continue
                    conf_raw = r.get("confidence", 0.0)
                    try: conf_val = float(conf_raw)
                    except: conf_val = 0.0
                    rels_out.append({
                        "subject": str(r.get("subject","")),
                        "predicate": _norm_token(r.get("predicate","")),
                        "object": str(r.get("object","")),
                        "spans": spans,
                        "confidence": conf_val
                    })

                if not ents_out:
                    rescue=[]
                    for it in (A.get("objects") or []):
                        if not isinstance(it, dict):  # 守卫
                            continue
                        cname = _canon_name(it.get("name",""))
                        try: cnt = int(float(it.get("count", 0)))
                        except: cnt = 0
                        for j in range(max(0, cnt)):
                            rescue.append({
                                "id": f"r{j+1}",
                                "name": cname,
                                "attributes": {"signature":[f"{cname}-coarse"]},
                                "spans": [[round(start_s,1), round(end_s,1)]]
                            })
                    return rescue, [], hint_from_A

            return ents_out, rels_out, hint_from_A

        except torch.cuda.OutOfMemoryError as e:
            print(f"[OOM] {e}")
            _empty_cuda()
            if cur_tokens>args.min_max_new_tokens:
                cur_tokens=max(args.min_max_new_tokens, int(cur_tokens*0.75))
                print(f"[OOM] reduce max_new_tokens -> {cur_tokens}")
            elif cur_temp>0.0:
                cur_temp=0.0
                print(f"[OOM] set temperature -> 0")
            else:
                raise
        except RuntimeError:
            raise

# ---------- 聚合 ----------
@dataclass
class AggE:
    name:str
    attributes:Dict[str,List[str]]
    spans:List[List[float]]
    sig:str

def _merge_spans(spans:List[List[float]], tol:float=0.2):
    if not spans: return []
    segs=sorted([[float(a),float(b)] for a,b in spans])
    res=[segs[0]]
    for a,b in segs[1:]:
        if a<=res[-1][1]+tol: res[-1][1]=max(res[-1][1],b)
        else: res.append([a,b])
    return res

def _time_overlap(a_spans,b_spans):
    inter=0.0
    for s1,e1 in a_spans:
        for s2,e2 in b_spans:
            inter+=max(0.0, min(e1,e2)-max(s1,s2))
    la=sum(max(0.0,x[1]-x[0]) for x in a_spans)
    lb=sum(max(0.0,x[1]-x[0]) for x in b_spans)
    den=max(1e-6, min(la,lb))
    return inter/den

def _time_proximity(a_spans,b_spans, max_gap:float=PROX_GAP):
    gaps=[]
    for s1,e1 in a_spans:
        for s2,e2 in b_spans:
            if e1 < s2:
                gaps.append(s2 - e1)
            elif e2 < s1:
                gaps.append(s1 - e2)
            else:
                return 1.0
    if not gaps:
        return 0.0
    g=min(gaps)
    if g>=max_gap: return 0.0
    return max(0.0, 1.0 - g/max_gap)

def _sim_attrs(a:Dict[str,List[str]], b:Dict[str,List[str]]):
    def flat(d):
        u=[]
        for k,vs in (d or {}).items():
            if k=="signature": continue
            u+=vs or []
        return set(u)
    return _jaccard(flat(a), flat(b))

def _match_score(E:AggE, name:str, attrs:Dict[str,List[str]], spans:List[List[float]]):
    if _canon_name(E.name)!=_canon_name(name):
        return 0.0
    sig1=_norm_token(E.sig or "")
    sig2=_norm_token(_mk_signature_from_attrs(name, attrs))
    sig_eq = 1.0 if (sig1 and sig2 and sig1==sig2) else 0.0
    a_sim = _sim_attrs(E.attributes, attrs)
    t_sim = max(_time_overlap(E.spans, spans), _time_proximity(E.spans, spans))
    return 0.6*sig_eq + 0.25*a_sim + 0.15*t_sim

# ---------- 场景切分（轻量） ----------
def _detect_scenes_by_hist(video_path: str,
                           thres: float = 0.6,
                           sample_fps: float = 2.0,
                           min_sec: float = 1.0) -> List[Tuple[float, float]]:
    try:
        import cv2
    except Exception:
        return [(0.0, float('inf'))]
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps   = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if total <= 0 or fps <= 1e-6:
        cap.release()
        return [(0.0, float('inf'))]
    step = max(1, int(round(fps / max(0.1, sample_fps))))
    bins = 16
    prev_hist = None
    cuts = [0.0]
    idx = 0
    while True:
        ok = cap.grab()
        if not ok: break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok: break
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
            s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
            v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
            hist = np.concatenate([h, s, v], axis=0).astype(np.float32)
            hist = hist / (np.sum(hist) + 1e-6)
            if prev_hist is not None:
                diff = np.sum(np.abs(hist - prev_hist))  # L1
                score = diff / 2.0
                if score >= thres:
                    t = idx / fps
                    if t - cuts[-1] >= min_sec:
                        cuts.append(t)
            prev_hist = hist
        idx += 1
    duration = total / fps
    if duration - cuts[-1] >= 1e-3:
        cuts.append(duration)
    scenes = []
    for i in range(len(cuts) - 1):
        s, e = cuts[i], cuts[i+1]
        if e - s >= min_sec:
            scenes.append((float(s), float(e)))
        else:
            if scenes:
                scenes[-1] = (scenes[-1][0], float(e))
            else:
                scenes.append((float(s), float(e)))
    cap.release()
    return scenes if scenes else [(0.0, duration)]

# ---------- 单视频处理 ----------
def process_one_video(video_path:str, out_path:str, model, tok, prompts):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # [SKIP_EXIST] 已有结果则直接复用
    try:
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            print(f"[SKIP] exists -> {out_path}")
            return
    except Exception:
        pass

    meta = read_video_meta_and_backend(video_path, preference=args.decode_backend)
    backend, handles, fps, total, duration = meta
    print(f"[Video] {os.path.basename(video_path)} | backend={backend} | duration={duration:.2f}s | fps={fps:.2f} | frames={total}")
    print(f"[Cfg]   min_span_sec={args.min_span_sec}")

    # 计划：场景切分 or 滑窗
    if args.scene_split:
        scenes = _detect_scenes_by_hist(
            video_path,
            thres=args.scene_thres,
            sample_fps=args.scene_sample_fps,
            min_sec=args.scene_min_sec
        )
        wins = [(max(0.0, s), min(duration, e if math.isfinite(e) else duration)) for s, e in scenes]
        wins = [(s, e) for (s, e) in wins if e > s + 1e-3]
        print(f"[Plan]  scenes={len(wins)} (scene-split on)")
    else:
        wins=[]; t=0.0
        while t<duration:
            s=t; e=min(duration, t+args.window_sec)
            wins.append((s,e)); t+=args.hop_sec
            if e>=duration: break
        if not wins: wins=[(0.0, duration)]
        print(f"[Plan]  windows={len(wins)} (sliding)")

    ALL_ENTS:List[AggE]=[]
    ALL_RELS=[]
    FINE_WINS=[]

    GLOBAL_NEED = {}

    for wi,(s,e) in enumerate(wins, 1):
        fps_cur    = args.fps
        cap_cur    = args.cap_frames
        tokens_cur = args.max_new_tokens
        temp_cur   = args.temperature
        pack_cur   = MAX_NUM_PACKING
        resize_cur = args.resize_short

        while True:
            try:
                frames, tids, (cs,ce), kframes = encode_clip(meta, s, e, fps_cur, cap_cur, resize_cur, pack_cur)
                ents, rels, hint_from_A = run_clip(
                    model,tok,frames,tids,cs,ce,tokens_cur,temp_cur,args.enable_thinking,
                    fps_cur,cap_cur,pack_cur,resize_cur,prompts, args.min_span_sec
                )

                need_local={}
                for eobj in (ents or []):
                    cname=_canon_name(eobj.get("name",""))
                    need_local[cname]=need_local.get(cname,0)+1
                for k,v in (hint_from_A or {}).items():
                    GLOBAL_NEED[k]=max(GLOBAL_NEED.get(k,0), int(v))
                for k,v in need_local.items():
                    GLOBAL_NEED[k]=max(GLOBAL_NEED.get(k,0), int(v))

                local2global = {}
                win_entities=[]
                win_relations=[]

                # 本窗索引
                local_index = {"id2info":{}, "class2ids":{}, "order_ids":[]}

                for eobj in (ents or []):
                    if not isinstance(eobj, dict):  # 守卫
                        continue
                    name=_canon_name(eobj.get("name",""))
                    if not name:
                        if hint_from_A and len([k for k,v in hint_from_A.items() if v>0])==1:
                            name = next(k for k,v in hint_from_A.items() if v>0)
                        elif hint_from_A and sum(hint_from_A.values())==1:
                            name = max(hint_from_A, key=hint_from_A.get)
                        else:
                            continue

                    spans=[[float(a),float(b)] for a,b in (eobj.get("spans") or []) if isinstance(a,(int,float)) and isinstance(b,(int,float))]
                    attrs = eobj.get("attributes", {}) or {}
                    if not isinstance(attrs, dict): attrs={}
                    sig = (attrs.get("signature",[None]) or [None])[0] if isinstance(attrs.get("signature"), list) else None
                    sig = _norm_token(sig or "") or _mk_signature_from_attrs(name, attrs)

                    # 匹配已有全局实体
                    best_i, best = -1, 0.0
                    for i,E in enumerate(ALL_ENTS):
                        sc=_match_score(E, name, attrs, spans)
                        if sc>best: best_i,best=i,sc

                    need = GLOBAL_NEED.get(name, 0)
                    cur_cnt = sum(1 for E in ALL_ENTS if _canon_name(E.name)==name)
                    force_new = (cur_cnt < need)

                    merge_thr = 0.65 if name not in HERD_CLASSES else 0.72

                    if (best>=merge_thr and best_i>=0) and not force_new:
                        E=ALL_ENTS[best_i]
                        for k,vs in (attrs or {}).items():
                            cur=set(E.attributes.get(k,[]))
                            for v in (vs or []):
                                if isinstance(v,str) and v.strip():
                                    cur.add(_norm_token(v))
                            E.attributes[k]=sorted(cur)
                        E.spans=_merge_spans(E.spans+spans)
                        if not E.sig and sig:
                            E.sig=_norm_token(sig)
                        gid = best_i
                    else:
                        newE = AggE(
                            name=name,
                            attributes={
                                k:sorted(set([_norm_token(v) for v in (vs or []) if isinstance(v,str) and v.strip()]))
                                for k,vs in (attrs or {}).items()
                            },
                            spans=_merge_spans(spans),
                            sig=_norm_token(sig)
                        )
                        ALL_ENTS.append(newE)
                        gid = len(ALL_ENTS)-1

                    local_id = str(eobj.get("id","")).strip() or f"e{wi}_{len(local2global)+1}"
                    local2global[local_id] = f"o{gid+1}"

                    if local_id not in local_index["id2info"]:
                        local_index["id2info"][local_id] = {"name": name, "spans": _merge_spans(spans)}
                        local_index["order_ids"].append(local_id)
                        local_index["class2ids"].setdefault(name, []).append(local_id)

                    win_entities.append({
                        "id": f"o{gid+1}",
                        "name": ALL_ENTS[gid].name,
                        "attributes": _clean_open_attrs(ALL_ENTS[gid].attributes),
                        "spans": [[round(cs,1), round(ce,1)]] if not spans else [[round(a,1),round(b,1)] for a,b in _merge_spans(spans)]
                    })

                # 端点解析器
                def _resolve_rel_endpoint(token, rel_spans, idx):
                    t = str(token if token is not None else "").strip()
                    if not t:
                        return ""
                    if t in idx["id2info"]:
                        return t
                    m = re.match(r'^([a-zA-Z][\w-]*)[ #_\-]*([0-9]+)$', t)
                    if m:
                        cls = _canon_name(m.group(1))
                        n = int(m.group(2))
                        ids = idx["class2ids"].get(cls, [])
                        if 1 <= n <= len(ids):
                            return ids[n-1]
                    cls = _canon_name(t)
                    ids = idx["class2ids"].get(cls, [])
                    if len(ids) == 1:
                        return ids[0]
                    if len(ids) > 1:
                        if rel_spans:
                            best, best_sc = "", -1.0
                            for lid in ids:
                                inter = _span_intersection(idx["id2info"][lid]["spans"], rel_spans)
                                sc = _span_total(inter)
                                if sc > best_sc:
                                    best, best_sc = lid, sc
                            if best_sc > 0:
                                return best
                        return max(ids, key=lambda z: _span_total(idx["id2info"][z]["spans"]))
                    m2 = re.search(r'([0-9]+)$', t)
                    if m2:
                        n = int(m2.group(1))
                        if 1 <= n <= len(idx["order_ids"]):
                            return idx["order_ids"][n-1]
                    return ""

                # 关系解析 + span 兜底（元素守卫）
                rel_saved = rel_drop_id = rel_drop_span = 0
                for r in (rels or []):
                    if not isinstance(r, dict):
                        continue
                    raw_rspan = _filter_spans(_coerce_spans(r.get("spans")), cs, ce, args.min_span_sec)
                    sid_l = _resolve_rel_endpoint(r.get("subject",""), raw_rspan, local_index)
                    oid_l = _resolve_rel_endpoint(r.get("object",""),  raw_rspan, local_index)
                    if not sid_l or not oid_l:
                        rel_drop_id += 1
                        continue

                    r_spans = raw_rspan
                    if not r_spans:
                        sub_sp = local_index["id2info"][sid_l]["spans"]
                        obj_sp = local_index["id2info"][oid_l]["spans"]
                        r_spans = _rel_span_fallback(sub_sp, obj_sp, cs, ce, MIN_REL_SPAN)
                        if not r_spans:
                            rel_drop_span += 1
                            continue

                    sid = local2global.get(sid_l, "")
                    oid = local2global.get(oid_l, "")
                    if not sid or not oid:
                        rel_drop_id += 1
                        continue

                    conf_raw = r.get("confidence", 0.0)
                    try: conf_val = float(conf_raw)
                    except: conf_val = 0.0
                    pred = _norm_token(r.get("predicate",""))

                    rel_rec = {
                        "subject": sid,
                        "predicate": pred,
                        "object": oid,
                        "spans": [[round(a,1), round(b,1)] for a,b in _merge_spans(r_spans)],
                        "confidence": conf_val
                    }
                    win_relations.append(rel_rec)
                    ALL_RELS.append(rel_rec)
                    rel_saved += 1

                if args.verbose:
                    print(f"[REL] window {wi}: saved={rel_saved} drop_id={rel_drop_id} drop_span={rel_drop_span}")

                FINE_WINS.append({
                    "start": round(cs,1), "end": round(ce,1),
                    "entities": win_entities,
                    "relations": win_relations
                })

                if args.verbose:
                    print(f"[OK] window {wi}/{len(wins)} {cs:.1f}-{ce:.1f}s frames={kframes} ent={len(win_entities)} rel={len(win_relations)}")
                break

            except torch.cuda.OutOfMemoryError:
                if fps_cur>args.min_fps:
                    fps_cur=max(args.min_fps, fps_cur-1); print(f"[OOM] retry fps={fps_cur}")
                elif cap_cur>96:
                    cap_cur=max(96,int(cap_cur*0.75)); print(f"[OOM] retry cap_frames={cap_cur}")
                elif tokens_cur>args.min_max_new_tokens:
                    tokens_cur=max(args.min_max_new_tokens,int(tokens_cur*0.75)); print(f"[OOM] retry max_new_tokens={tokens_cur}")
                elif resize_cur>320:
                    resize_cur=max(320, int(resize_cur*0.85)); print(f"[OOM] retry resize_short={resize_cur}")
                elif pack_cur>1:
                    pack_cur=max(1, pack_cur-1); print(f"[OOM] retry max_packing={pack_cur}")
                else:
                    raise
                _empty_cuda()
            except Exception as ex:
                print(f"[WARN] window {wi} failed: {ex}")
                FINE_WINS.append({
                    "start": round(s,1), "end": round(e,1),
                    "entities": [], "relations":[]
                })
                break

    def _merge_spans_global(ss): return [[round(a,1),round(b,1)] for a,b in _merge_spans(ss)]

    # 全局导出
    fine_entities=[]
    for i,E in enumerate(ALL_ENTS, start=1):
        if not _canon_name(E.name):
            continue
        attrs = _clean_open_attrs(E.attributes)
        if E.sig:
            attrs.setdefault("signature", [E.sig])
        fine_entities.append({
            "id": f"o{i}",
            "name": E.name,
            "attributes": attrs,
            "spans": _merge_spans_global(E.spans)
        })

    obj_counts={}
    for ent in fine_entities:
        obj_counts[ent["name"]] = obj_counts.get(ent["name"], 0) + 1

    attr_view={}
    for ent in fine_entities:
        c=ent["name"]; A=attr_view.setdefault(c,{})
        for k,vs in (ent["attributes"] or {}).items():
            S=set(A.get(k,[])); [S.add(v) for v in vs]; A[k]=sorted(S)

    triplets=[]
    for r in ALL_RELS:
        triplets.append({
            "subject": r.get("subject",""),
            "predicate": r.get("predicate",""),
            "object": r.get("object","")
        })

    display_counts = { _display_plural(k, v): v for k,v in obj_counts.items() }

    out={
        "meta":{
            "video": os.path.abspath(video_path),
            "window_sec": args.window_sec, "hop_sec": args.hop_sec,
            "fps_per_window": args.fps, "resize_short": args.resize_short,
            "model": args.model_id or "local",
            "backend": backend, "dtype": args.dtype,
            "min_span_sec": args.min_span_sec
        },
        "fine":{ "entities": fine_entities, "relations": ALL_RELS },
        "fine_windows": FINE_WINS,
        "views":{
            "objects_count": obj_counts,
            "objects_count_display": display_counts,
            "attributes": attr_view,
            "triplets": triplets
        }
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path,"w",encoding="utf-8") as f:
        json.dump(out,f,ensure_ascii=False,indent=2)
    print(f"[OK] saved -> {out_path}")

# ---------- 主流程 ----------
def main():
    single_mode = bool(args.video)
    batch_mode  = bool(args.batch_from)

    if not single_mode and not batch_mode:
        raise SystemExit("请提供 --video/--out（单文件）或 --batch-from/--out-dir（批量）")

    if single_mode and not args.out:
        raise SystemExit("--out 不能为空（单文件模式）")

    if batch_mode:
        if not args.out_dir:
            raise SystemExit("--out-dir 不能为空（批量模式）")
        os.makedirs(args.out_dir, exist_ok=True)

    # 设备/精度
    device = 'cuda' if (args.device=='cuda' and torch.cuda.is_available()) else 'cpu'
    dtype  = _dtype_of(device, args.dtype)

    # 模型（仅加载一次，批量复用）
    where = args.local_path if args.local_path else args.model_id
    is_dir = bool(args.local_path and os.path.isdir(args.local_path))
    model, tok = load_model(where, device, dtype, args.cache_dir, args.local_files_only, args.revision, is_dir)

    # 统一构建一次提示词
    prompts = build_prompts(args.min_span_sec)

    if single_mode:
        if not os.path.exists(args.video):
            raise FileNotFoundError(args.video)
        # 单文件也做一次输出名一致性矫正
        expect = os.path.splitext(os.path.basename(args.video))[0] + '.json'
        out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
        out_final = os.path.join(out_dir, expect)
        os.makedirs(out_dir, exist_ok=True)
        process_one_video(args.video, out_final, model, tok, prompts)
        return

    # 批量模式
    tasks = _discover_inputs(args.batch_from, args.out_dir)
    if not tasks:
        raise SystemExit(f"批量模式未发现视频：{args.batch_from}")
    # 关键修复：统一矫正输出文件名为 basename(video).json（目录保留）
    tasks = [(vin, _sanitize_out(vin, vout, args.out_dir)) for vin, vout in tasks]

    print(f"[Batch] total videos: {len(tasks)}")
    for i,(vin, vout) in enumerate(tasks, start=1):
        try:
            print(f"\n=== [{i}/{len(tasks)}] {vin} ===")
            if not os.path.exists(vin):
                raise FileNotFoundError(vin)
            os.makedirs(os.path.dirname(vout), exist_ok=True)
            process_one_video(vin, vout, model, tok, prompts)
        except Exception as e:
            print(f"[ERROR] {vin} failed: {e}")

if __name__=="__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6')
        main()
