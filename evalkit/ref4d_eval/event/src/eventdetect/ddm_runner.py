# -*- coding: utf-8 -*-
"""
DDM-Net GEBD runner — Direct inference with official getModel(model_name, args)
- 严格 5D 输入 [B, T, 3, H, W]
- 自动对齐位置编码期望的时间窗口长度（从异常里解析 T_expected）
- 兼容 forward 返回 tuple/list/dict，自动抽取 logits -> 概率
- 输出契约: [{id,s_abs,e_abs,s,e}, ...]
- 稳定性：
  * 固定随机种子/确定性
  * model.eval()
  * 峰值检测增加 min_peak_prominence（默认 0.08）
  * 去掉近邻合并，防止跨镜头被拼成整段
  * batch==1 时临时复制，避免注意力 softmax 的 1D 崩溃
"""
import argparse
import json
import os
import re
import sys
import inspect
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

# ---------------------------- 基础 I/O ---------------------------- #

def _ensure_dir(p: Union[str, Path]):
    Path(p).mkdir(parents=True, exist_ok=True)

def _read_video_meta(video_path: str) -> Tuple[float, int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = n / fps if fps > 0 else 0.0
    cap.release()
    return float(fps), n, float(dur)

def _load_scenes_json(scenes_json: Optional[str]) -> Optional[List[Tuple[float, float]]]:
    if not scenes_json:
        return None
    if not os.path.isfile(scenes_json):
        raise FileNotFoundError(f"scenes json not found: {scenes_json}")
    with open(scenes_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    scenes = data.get("scenes", [])
    return [(float(s), float(e)) for s, e in scenes]

def _normalize_events(events_se: List[Tuple[float, float]], total_dur: float) -> List[Dict]:
    total_dur = max(total_dur, 1e-6)
    items = []
    for k, (s_abs, e_abs) in enumerate(events_se, start=1):
        s = max(0.0, min(1.0, s_abs / total_dur))
        e = max(0.0, min(1.0, e_abs / total_dur))
        if e < s:
            e = s
        items.append({"id": f"e{k:04d}", "s_abs": float(s_abs), "e_abs": float(e_abs), "s": float(s), "e": float(e)})
    return items

def _save_events_json(out_path: str, items: List[Dict]):
    _ensure_dir(Path(out_path).parent)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

# -------------------------- 读取/预处理 -------------------------- #

def _get_stats(cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    ddm = cfg.get("ddm", {})
    mean = ddm.get("mean", [0.485, 0.456, 0.406])
    std  = ddm.get("std",  [0.229, 0.224, 0.225])
    return np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32)

def _read_frames_rgb(video_path: str, t0: float, t1: float, fps: float, n_frames: int,
                     size: Tuple[int, int]) -> List[np.ndarray]:
    """读 [t0,t1] 的帧，BGR->RGB，Resize 到 size=(W,H)，返回 list[np.uint8(H,W,3)]"""
    W, H = size
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    start_f = max(0, int(round(t0 * fps)))
    end_f   = min(n_frames - 1, int(round(t1 * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frames = []
    for _ in range(start_f, end_f + 1):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
        frames.append(frame.astype(np.uint8))
    cap.release()
    return frames

def _to_tensor(frames: List[np.ndarray], mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    """List[H,W,3] uint8 -> [T,3,H,W] float32(归一化)"""
    if len(frames) == 0:
        return torch.empty(0, 3, 224, 224)
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # [T,H,W,3]
    arr = (arr - mean) / std
    arr = np.transpose(arr, (0, 3, 1, 2))  # [T,3,H,W]
    return torch.from_numpy(arr)  # float32

# ------------------------ 窗口构造（严格 5D） ------------------------ #

def _build_windows_bct(x: torch.Tensor, win: int, stride: int) -> Tuple[torch.Tensor, List[int]]:
    """
    x: [T,3,H,W] -> windows: [B, T, 3, H, W]；centers: 每个窗口中心帧索引（段内）
    约定 win 为奇数；若为偶数则用 win-1 以避免长度偏差。
    """
    if win % 2 == 0 and win > 1:
        win -= 1
    T = x.shape[0]
    if T < win:
        return torch.empty(0), []
    half = win // 2
    centers, chunks = [], []
    for c in range(half, T - half, stride):
        s = c - half
        e = c + half + 1
        clip = x[s:e].unsqueeze(0)  # [1,win,3,H,W]
        chunks.append(clip)
        centers.append(c)
    if not chunks:
        return torch.empty(0), []
    windows = torch.cat(chunks, dim=0)  # [B,win,3,H,W]
    return windows, centers

# ----------------------------- 模型 ----------------------------- #

def _import_ddm_repo(repo_dir: str):
    repo_dir = str(Path(repo_dir).resolve())
    ddm_net = Path(repo_dir) / "DDM-Net"
    if not ddm_net.is_dir():
        raise RuntimeError(f"Invalid repo_dir (DDM-Net not found): {repo_dir}")
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    if str(ddm_net) not in sys.path:
        sys.path.insert(0, str(ddm_net))

class _DummyArgs:
    """最小 args：提供 num_classes，其余访问返回 None。"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __getattr__(self, name):
        return None

def _build_model(cfg: Dict) -> torch.nn.Module:
    ddm = cfg.get("ddm", {})
    model_name = ddm.get("model", None)
    if not model_name or not isinstance(model_name, str):
        raise RuntimeError("Config 'ddm.model' 缺失或不是字符串，请设置为 DDM 仓库 getModel 支持的确切名称。")
    try:
        from utils.getter import getModel  # type: ignore
    except Exception as e:
        raise RuntimeError(f"无法从 DDM 仓库导入 utils.getter.getModel：{e}")

    num_classes = int(ddm.get("num_classes", cfg.get("num_classes", 2)))
    args = _DummyArgs(num_classes=num_classes)
    for k in ["backbone", "resnet_type", "drop_path_rate", "pretrained", "arch", "embed_dim",
              "depths", "num_heads", "mlp_ratio", "qkv_bias", "attn_drop_rate", "drop_rate"]:
        if not hasattr(args, k):
            setattr(args, k, None)

    sig = inspect.signature(getModel)
    if len(sig.parameters) >= 1:
        model = getModel(model_name=model_name, args=args)
    else:
        model = getModel(model_name, args)
    return model

def _load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    new_state = {}
    for k, v in state.items():
        new_state[k[7:]] = v if k.startswith("module.") else v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print("[DDM] missing keys:", missing)
    if unexpected:
        print("[DDM] unexpected keys:", unexpected)

# ----------------------------- forward 适配 ----------------------------- #

def _flatten_tensors(obj) -> List[torch.Tensor]:
    out = []
    if torch.is_tensor(obj):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            out.extend(_flatten_tensors(x))
    elif isinstance(obj, dict):
        for k in obj:
            out.extend(_flatten_tensors(obj[k]))
    return out

def _pick_logits_from_any(out_any, batch: int) -> torch.Tensor:
    """
    从 forward 返回结构中挑选 logits：
      1) 2D [B,K]；
      2) 3D [B,T,K] -> T 上 mean -> [B,K]；
      3) 1D [B] -> [B,1]
    """
    tens = _flatten_tensors(out_any)
    if not tens:
        raise RuntimeError("Model forward returned no tensor outputs.")
    c2 = [t for t in tens if t.ndim == 2 and t.shape[0] == batch]
    if c2:
        return c2[-1]
    c3 = [t for t in tens if t.ndim == 3 and t.shape[0] == batch]
    if c3:
        return c3[-1].mean(dim=1)
    c1 = [t for t in tens if t.ndim == 1 and t.shape[0] == batch]
    if c1:
        return c1[-1].unsqueeze(-1)
    shapes = [tuple(t.shape) for t in tens]
    raise RuntimeError(f"Cannot pick logits from outputs. Candidate tensor shapes: {shapes} (expected batch={batch})")

def _parse_expected_T_from_error(err: Exception) -> Optional[int]:
    """
    从错误文本中解析模型期望的时间长度 T_expected。
    优先匹配: 'size of tensor a (5) must match the size of tensor b (11) ... dimension 0'
    其次回退到提取所有括号里的数字取最大值。
    """
    s = str(err)
    m = re.search(r"size of tensor a \((\d+)\) .* size of tensor b \((\d+)\).*dimension 0", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return max(a, b)
    nums = [int(x) for x in re.findall(r"\((\d+)\)", s) if int(x) > 1]
    return max(nums) if nums else None

@torch.no_grad()
def _forward_prob(model: torch.nn.Module, inp5d: torch.Tensor) -> torch.Tensor:
    """
    inp5d: [B, T, 3, H, W]
    返回 prob: [B]
    - 若 B==1，临时复制一份 -> [2,T,3,H,W]，前向后再切回 [:1]，避免某些实现下 softmax 的 1D 崩溃。
    """
    b = inp5d.shape[0]
    if b == 1:
        inp5d_run = torch.cat([inp5d, inp5d], dim=0)
        out_any = model(inp5d_run)
        logits = _pick_logits_from_any(out_any, batch=inp5d_run.shape[0])  # [2,K] / [2]...
        if logits.ndim == 2 and logits.shape[-1] >= 2:
            prob = F.softmax(logits, dim=-1)[:, 1]
        else:
            prob = torch.sigmoid(logits.squeeze(-1))
        return prob[:1].detach().float().cpu()
    else:
        out_any = model(inp5d)
        logits = _pick_logits_from_any(out_any, batch=b)
        if logits.ndim == 2 and logits.shape[-1] >= 2:
            prob = F.softmax(logits, dim=-1)[:, 1]
        else:
            prob = torch.sigmoid(logits.squeeze(-1))
        return prob.detach().float().cpu()

@torch.no_grad()
def _infer_segment_scores(model: torch.nn.Module,
                          x: torch.Tensor,
                          device: torch.device,
                          win: int,
                          stride: int,
                          batch_size: int) -> Tuple[np.ndarray, List[int]]:
    """
    x: [T,3,H,W]; 严格 5D 前向（[B, T, 3, H, W]）。
    若出现时序长度不匹配，自动解析 T_expected 并重建窗口再跑一次。
    """
    def run_windows(windows, centers):
        scores = []
        for st in range(0, windows.shape[0], batch_size):
            ed = min(windows.shape[0], st + batch_size)
            inp = windows[st:ed].to(device)                    # [b,win,3,h,w]
            prob = _forward_prob(model, inp)                   # [b]
            scores.append(prob)
        probs = torch.cat(scores, dim=0).numpy()
        return probs, centers

    windows, centers = _build_windows_bct(x, win=win, stride=stride)  # [B,win,3,H,W]
    if windows.numel() == 0:
        return np.zeros((0,), dtype=np.float32), []
    try:
        return run_windows(windows, centers)
    except RuntimeError as e:
        T_expected = _parse_expected_T_from_error(e)
        if T_expected is None or T_expected == win:
            raise
        windows2, centers2 = _build_windows_bct(x, win=T_expected, stride=stride)
        if windows2.numel() == 0:
            raise
        return run_windows(windows2, centers2)

# ----------------------------- 后处理 ----------------------------- #

def _scores_to_boundaries(scores: np.ndarray,
                          centers: List[int],
                          threshold: float,
                          min_distance: int,
                          min_prominence: float = 0.08) -> List[int]:
    """阈值 + 峰值抑制 -> 边界中心帧（全局帧号）"""
    import scipy.signal
    if len(scores) == 0:
        return []
    s = scores.astype(np.float32).reshape(-1)
    try:
        s = cv2.GaussianBlur(s.reshape(-1, 1), (9, 1), 0).reshape(-1)
    except Exception:
        pass
    peaks, _ = scipy.signal.find_peaks(
        s,
        height=float(threshold),
        distance=int(min_distance),
        prominence=float(min_prominence) if min_prominence is not None else None
    )
    return [int(centers[p]) for p in peaks.tolist()]

def _rseed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

def _boundaries_to_events(boundary_frames: List[int],
                          fps: float,
                          seg_t0: float,
                          seg_t1: float,
                          min_event_dur: float) -> List[Tuple[float, float]]:
    """
    边界 -> 事件区间；补足首段与末段；无边界则整段作为单事件（若时长达标）。
    事件一律裁剪到 [seg_t0, seg_t1]。
    """
    if not boundary_frames:
        if seg_t1 - seg_t0 >= float(min_event_dur):
            return [(float(seg_t0), float(seg_t1))]
        return []

    bsec = sorted({max(seg_t0, min(seg_t1, bf / max(fps, 1e-6))) for bf in boundary_frames})
    events: List[Tuple[float, float]] = []

    # 首段
    first_b = bsec[0]
    if first_b - seg_t0 >= float(min_event_dur):
        events.append((float(seg_t0), float(first_b)))

    # 中间段
    for a, c in zip(bsec[:-1], bsec[1:]):
        s_abs = max(seg_t0, float(a))
        e_abs = min(seg_t1, float(c))
        if e_abs - s_abs >= float(min_event_dur):
            events.append((s_abs, e_abs))

    # 末段
    last_b = bsec[-1]
    if seg_t1 - last_b >= float(min_event_dur):
        events.append((float(last_b), float(seg_t1)))

    return events

# ----------------------------- 统筹 ----------------------------- #

def _run_ddm_direct(video_path: str,
                    cfg: Dict,
                    scenes: Optional[List[Tuple[float, float]]] = None) -> List[Tuple[float, float]]:
    ddm = cfg.get("ddm", {})
    repo_dir   = ddm.get("repo_dir", cfg.get("repo_dir"))
    ckpt_path  = ddm.get("ckpt", cfg.get("checkpoint"))
    device_str = ddm.get("device", cfg.get("device", "auto"))
    win        = int(ddm.get("window", cfg.get("window", 5)))
    stride     = int(ddm.get("stride", cfg.get("stride", 1)))
    input_size = int(ddm.get("input_size", cfg.get("input_size", 224)))

    # 稳定性相关配置
    seed             = int(cfg.get("seed", 1337))
    batch_size       = int(cfg.get("batch_size", 128))
    threshold        = float(cfg.get("threshold", 0.5))
    min_peak_dist    = int(cfg.get("min_peak_distance", 3))
    min_peak_prom    = float(cfg.get("min_peak_prominence", 0.08))
    min_event_dur    = float(cfg.get("min_event_dur", 0.10))

    if not repo_dir or not os.path.isdir(repo_dir):
        raise RuntimeError(f"Invalid repo_dir: {repo_dir}")
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")

    _rseed_all(seed)
    _import_ddm_repo(repo_dir)

    device = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available()) else device_str)
    model = _build_model(cfg).to(device)
    _load_checkpoint(model, ckpt_path, device)
    model.eval()

    mean, std = _get_stats(cfg)
    fps, n_frames, dur = _read_video_meta(video_path)

    def run_one_range(t0: float, t1: float) -> List[Tuple[float, float]]:
        frames = _read_frames_rgb(video_path, t0, t1, fps, n_frames, size=(input_size, input_size))
        if len(frames) < win:
            x = _to_tensor(frames, mean, std)
            scores, centers_local = np.zeros((0,), dtype=np.float32), []
        else:
            x = _to_tensor(frames, mean, std)  # [T,3,H,W]
            scores, centers_local = _infer_segment_scores(
                model, x, device, win=win, stride=stride, batch_size=batch_size
            )
        seg_start_f = max(0, int(round(t0 * fps)))
        centers_global = [c + seg_start_f for c in centers_local]
        bframes = _scores_to_boundaries(
            scores, centers_global, threshold=threshold,
            min_distance=min_peak_dist, min_prominence=min_peak_prom
        )
        return _boundaries_to_events(bframes, fps=fps, seg_t0=t0, seg_t1=t1, min_event_dur=min_event_dur)

    events_all: List[Tuple[float, float]] = []
    if scenes:
        for (s, e) in scenes:
            s = float(s); e = float(e)
            if e - s <= 1e-6:
                continue
            events_all.extend(run_one_range(s, e))
    else:
        events_all.extend(run_one_range(0.0, dur))

    # 不做近邻合并
    events_all.sort(key=lambda x: x[0])
    return events_all

# ------------------------------- CLI ------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="input video")
    ap.add_argument("--out", required=True, help="output events json")
    ap.add_argument("--config", required=True, help="model_gebd.yaml")
    ap.add_argument("--scenes", default=None, help="optional scenes json to run per-shot")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    scenes = _load_scenes_json(args.scenes) if args.scenes else None
    fps, n_frames, dur = _read_video_meta(args.video)
    events = _run_ddm_direct(args.video, cfg, scenes=scenes)

    items = _normalize_events(events, total_dur=dur)
    _save_events_json(args.out, items)
    print(f"[DDM-Net] events saved -> {args.out}  #events={len(items)}")

if __name__ == "__main__":
    main()
