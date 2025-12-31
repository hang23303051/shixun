# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import numpy as np

# ------------------------
# 可用性检测（最小依赖）
# ------------------------
def _jax_available() -> bool:
    """
    仅检测 jax/haiku 是否可用；绝不 import tapnet，避免触发其 __init__ 的 JAX 依赖。
    """
    try:
        import jax  # noqa: F401
        import haiku  # noqa: F401
        return True
    except Exception:
        return False


# --- 仅加载本地 Torch 版 tapir_model.py，绕过 tapnet/__init__.py ---
def _load_local_torch_tapir_model():
    """
    直接按文件路径加载 third_party/tapnet/tapnet/torch/tapir_model.py，
    不执行 tapnet/__init__.py，避免 JAX/Haiku 依赖。
    """
    import sys, types, importlib.util

    _BASE = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         "..", "..", "third_party", "tapnet", "tapnet"))
    _TORCH_DIR  = os.path.join(_BASE, "torch")
    _TORCH_FILE = os.path.join(_TORCH_DIR, "tapir_model.py")
    if not os.path.isfile(_TORCH_FILE):
        raise FileNotFoundError(f"Torch TAPIR file not found: {_TORCH_FILE}")

    # 伪包：tapnet / tapnet.torch（仅为支持相对导入，不会运行原 __init__.py）
    if "tapnet" not in sys.modules:
        tapnet_pkg = types.ModuleType("tapnet")
        tapnet_pkg.__path__ = [_BASE]
        sys.modules["tapnet"] = tapnet_pkg
    if "tapnet.torch" not in sys.modules:
        torch_pkg = types.ModuleType("tapnet.torch")
        torch_pkg.__path__ = [_TORCH_DIR]
        sys.modules["tapnet.torch"] = torch_pkg

    spec = importlib.util.spec_from_file_location("tapnet.torch.tapir_model", _TORCH_FILE)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "tapnet.torch"
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_sibling_module(basename: str):
    """
    从 torch 目录旁加载同名文件，如 inference.py / inference_utils.py。
    不存在则返回 None。
    """
    import importlib.util
    _BASE = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         "..", "..", "third_party", "tapnet", "tapnet"))
    _TORCH_DIR  = os.path.join(_BASE, "torch")
    cand = [os.path.join(_TORCH_DIR, f"{basename}.py"),
            os.path.join(_TORCH_DIR, f"{basename}_utils.py")]
    for p in cand:
        if os.path.isfile(p):
            spec = importlib.util.spec_from_file_location(f"tapnet.torch.{basename}", p)
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = "tapnet.torch"
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            return mod
    return None


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        _ = _load_local_torch_tapir_model()
        return True
    except Exception:
        return False


# ------------------------
# 后端实现
# ------------------------
class TapirJAXBackend:
    """
    JAX/Haiku TAPIR 前向（仅在 backend='jax' 或 auto 且 JAX 真可用时才会用到）。
    权重：causal_tapir_checkpoint.npy
    """
    def __init__(self, ckpt_path: str, causal: bool = True):
        import jax
        import haiku as hk
        from jax import numpy as jnp
        self.jax = jax; self.jnp = jnp; self.hk = hk

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"TAPIR JAX checkpoint not found: {ckpt_path}")
        params = np.load(ckpt_path, allow_pickle=True).item()
        self.params = params

        # 若只用 torch 后端，永远不会走到这里
        from tapnet.jax import tapir as tapir_jax  # noqa: E402
        from tapnet.jax import model_utils as mu   # noqa: E402

        self.mu = mu
        self.tapir = tapir_jax
        self.causal = causal

        def forward_fn(frames_bhwc: np.ndarray, query_xy1: np.ndarray):
            f = self.mu.preprocess_frames(frames_bhwc)
            feat = self.tapir.get_feature_grids(f, is_training=False)
            out = self.tapir.infer(
                frames=f,
                feature_grids=feat,
                query_points=query_xy1,
                is_training=False,
                causal=True
            )
            tracks = np.array(out.get('tracks'))
            if 'visibility' in out:
                vis = np.array(out['visibility']).astype(bool)
            else:
                occ = np.array(out.get('occluded'))
                vis = ~(occ.astype(bool))
            return tracks, vis

        self._fwd = forward_fn

    def __call__(self, frames_rgb: np.ndarray, seeds_xy: np.ndarray, seeds_t0: int = 0):
        assert frames_rgb.ndim == 4 and frames_rgb.shape[-1] == 3
        N = seeds_xy.shape[0]
        q = np.concatenate([seeds_xy.astype(np.float32), np.full((N, 1), seeds_t0, np.float32)], axis=1)
        tracks, vis = self._fwd(frames_rgb, q)
        return tracks.astype(np.float32), vis.astype(bool)


class TapirTorchBackend:
    """
    PyTorch 版 TAPIR/BootsTAPIR。权重如：bootstapir_checkpoint_v2.pt
    """
    def __init__(self, ckpt_path_pt: str):
        import torch
        tapir_model = _load_local_torch_tapir_model()  # 只加载 torch 子模块
        self.torch = torch
        self.model = tapir_model.TAPIR()
        if not os.path.isfile(ckpt_path_pt):
            raise FileNotFoundError(f"TAPIR Torch checkpoint not found: {ckpt_path_pt}")
        state = torch.load(ckpt_path_pt, map_location='cpu')
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        # 尝试发现同目录的 inference.py（若存在则优先用）
        self._infer_fn = None
        self._infer_kind = None  # 'method' | 'module' | 'forward'
        if hasattr(self.model, "infer") and callable(getattr(self.model, "infer")):
            self._infer_kind = "method"
        else:
            infer_mod = _load_sibling_module("inference")
            if infer_mod is not None and hasattr(infer_mod, "infer") and callable(infer_mod.infer):
                self._infer_fn = infer_mod.infer
                self._infer_kind = "module"
            else:
                # 最后尝试直接 forward
                if callable(getattr(self.model, "__call__", None)) or callable(getattr(self.model, "forward", None)):
                    self._infer_kind = "forward"
                else:
                    raise AttributeError("Neither TAPIR.infer nor inference.infer nor forward() is available in Torch TAPIR.")

    @staticmethod
    def _normalize_output(out):
        """
        统一输出为 (tracks[N,T,2], vis[N,T])；支持多种返回格式。
        """
        import torch
        def _to_np(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return x

        tracks = None
        vis = None

        if isinstance(out, dict):
            if "tracks" in out:
                tr = _to_np(out["tracks"])
                # 可能是 [B,N,T,2] / [N,T,2] / [B,T,N,2]
                if tr.ndim == 4:
                    if tr.shape[0] == 1 and tr.shape[3] == 2:
                        # 假设 [B,N,T,2]
                        tr = tr[0]
                    elif tr.shape[0] == 1 and tr.shape[2] == 2:
                        # 假设 [B,T,N,2] -> [N,T,2]
                        tr = np.transpose(tr[0], (1, 0, 2))
                    else:
                        tr = tr[0]
                tracks = tr
            if "visibility" in out:
                vs = _to_np(out["visibility"])
                if vs.ndim == 3:   # [B,N,T]
                    vs = vs[0]
                vis = vs.astype(bool)
            elif "occluded" in out:
                occ = _to_np(out["occluded"])
                if occ.ndim == 3:
                    occ = occ[0]
                vis = (~occ.astype(bool))
        elif isinstance(out, (tuple, list)) and len(out) >= 1:
            tr = _to_np(out[0])
            if tr.ndim == 4 and tr.shape[0] == 1:
                tr = tr[0]
            tracks = tr
            if len(out) >= 2:
                vs = _to_np(out[1])
                if vs.ndim == 3 and vs.shape[0] == 1:
                    vs = vs[0]
                vis = vs.astype(bool)

        if tracks is None:
            raise ValueError("Torch TAPIR output does not contain 'tracks'.")

        if vis is None:
            # 若不可见性没有提供，则全部置 True（保守处理）
            vis = np.ones(tracks.shape[:2], dtype=bool)

        # 规范到 [N,T,2]
        if tracks.ndim == 2 and tracks.shape[-1] == 2:
            tracks = tracks[None, ...]  # [1,T,2] -> [N=1,T,2]
        if tracks.shape[-1] != 2:
            # 若是 [T,N,2]，转置
            if tracks.ndim == 3 and tracks.shape[1] == 2 and tracks.shape[2] != 2:
                tracks = np.transpose(tracks, (0, 2, 1))
        return tracks.astype(np.float32), vis.astype(bool)

    def __call__(self, frames_rgb: np.ndarray, seeds_xy: np.ndarray, seeds_t0: int = 0):
        import torch, cv2 as cv
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        # 环境变量可控（不改动外部 cfg）
        max_hw_env = os.getenv("RRM_TAPIR_MAX_HW", "").strip()
        try:
            max_hw0 = int(max_hw_env) if len(max_hw_env) else 384
        except Exception:
            max_hw0 = 384
        use_fp16 = os.getenv("RRM_TAPIR_FP16", "1") not in ("0", "false", "False")

        def _resize_video(frames_f32: np.ndarray, max_hw: int, mult: int = 8):
            """把视频按最短边不超过 max_hw 且对齐到 mult 倍数缩放；返回缩放后视频与 scale_x/y。"""
            T, H, W, C = frames_f32.shape
            if max(H, W) <= max_hw and (H % mult == 0) and (W % mult == 0):
                return frames_f32, 1.0, 1.0
            s = min(1.0, max_hw / float(max(H, W)))
            new_h = max(mult, int(np.floor(H * s / mult) * mult))
            new_w = max(mult, int(np.floor(W * s / mult) * mult))
            # 避免 0 尺寸
            new_h = max(new_h, mult)
            new_w = max(new_w, mult)
            out = np.empty((T, new_h, new_w, C), dtype=np.float32)
            for i in range(T):
                out[i] = cv.resize(frames_f32[i], (new_w, new_h), interpolation=cv.INTER_AREA)
            sx = new_w / float(W)
            sy = new_h / float(H)
            return out, sx, sy

        # 统一准备查询与视频
        seeds_xy = seeds_xy.astype(np.float32)
        q_base = np.concatenate([seeds_xy, np.full((seeds_xy.shape[0], 1), seeds_t0, np.float32)], 1)

        # 自适应重试：OOM 时降低分辨率
        max_hw = max_hw0
        last_err = None
        for _attempt in range(4):  # 最多 4 次（384→256→192→192）
            try:
                # 1) 缩放到 max_hw、[0,1]；确保 8 的倍数，减少“round down”警告
                vid_f32 = frames_rgb.astype(np.float32)  # [T,H,W,3] in [0,1]
                vid_scaled, sx, sy = _resize_video(vid_f32, max_hw=max_hw, mult=8)

                # 2) 同步缩放查询点到缩放后的坐标系
                q_scaled = q_base.copy()
                q_scaled[:, 0] *= sx
                q_scaled[:, 1] *= sy

                # 3) 转张量
                v = torch.from_numpy(vid_scaled[None]).to(device)  # [1,T,H,W,3]
                q_t = torch.from_numpy(q_scaled[None]).to(device)  # [1,N,3]

                # 4) 推理（多策略）
                with torch.no_grad():
                    def _do_infer():
                        if self._infer_kind == "method":
                            return self.model.infer(v, q_t)
                        elif self._infer_kind == "module" and self._infer_fn is not None:
                            try:
                                return self._infer_fn(self.model, v, q_t)
                            except TypeError:
                                return self._infer_fn(self.model, video=v, queries=q_t)
                        elif self._infer_kind == "forward":
                            # 依次尝试三种常见签名
                            try:
                                return self.model(v, q_t)
                            except Exception:
                                try:
                                    return self.model(v, query_points=q_t)
                                except Exception:
                                    return self.model({"video": v, "query_points": q_t})
                        else:
                            raise RuntimeError("Unknown TAPIR torch infer kind")

                    if use_fp16 and device.type == 'cuda':
                        # 半精度自动混合
                        with torch.autocast('cuda', dtype=torch.float16):
                            out = _do_infer()
                    else:
                        out = _do_infer()

                # 5) 规范输出，并把坐标还原回原分辨率
                tracks_s, vis = self._normalize_output(out)  # [N,T,2], [N,T]
                tracks = tracks_s.copy()
                tracks[..., 0] /= (sx if sx != 0 else 1.0)
                tracks[..., 1] /= (sy if sy != 0 else 1.0)
                return tracks.astype(np.float32), vis.astype(bool)

            except RuntimeError as e:
                msg = str(e)
                last_err = e
                is_oom = ("out of memory" in msg.lower()) or ("cuda error: out of memory" in msg.lower())
                if is_oom and device.type == 'cuda' and max_hw > 192:
                    torch.cuda.empty_cache()
                    max_hw = max(192, int(max_hw // 1.5))  # 384→256→192
                    continue
                raise

        raise RuntimeError(f"Torch TAPIR inference failed after retries (last={repr(last_err)})")


# ------------------------
# 单例缓存（关键优化）
# ------------------------
_BACKEND_CACHE = {}  # key = ("jax", jax_ckpt) 或 ("torch", torch_ckpt) -> backend instance


def _get_or_create_backend(backend: str,
                           ckpt_dir: str,
                           jax_ckpt_name: str,
                           torch_ckpt_name: str):
    """
    同一进程内、相同 (backend, weights) 只创建一次，大幅减少重复 load_state_dict 时间。
    """
    if backend == "jax":
        key = ("jax", os.path.join(ckpt_dir, jax_ckpt_name))
        if key not in _BACKEND_CACHE:
            _BACKEND_CACHE[key] = TapirJAXBackend(key[1], causal=True)
        return _BACKEND_CACHE[key]

    if backend == "torch":
        key = ("torch", os.path.join(ckpt_dir, torch_ckpt_name))
        if key not in _BACKEND_CACHE:
            _BACKEND_CACHE[key] = TapirTorchBackend(key[1])
        return _BACKEND_CACHE[key]

    raise ValueError(f"unknown backend: {backend}")


# ------------------------
# 统一入口（接口保持不变）
# ------------------------
def run_tapir_tracks(
    frames_bgr: np.ndarray,
    seeds_xy: np.ndarray,
    seeds_t0: int = 0,
    backend: str = "auto",
    ckpt_dir: str = "/root/autodl-tmp/aigc_motion_eval/weights/tapnet_checkpoints",
    jax_ckpt_name: str = "causal_tapir_checkpoint.npy",
    torch_ckpt_name: str = "bootstapir_checkpoint_v2.pt",
):
    """
    统一入口：优先 JAX；否则 Torch；再失败则 LK 光流跟踪。
    frames_bgr: [T,H,W,3] (BGR)，内部转 RGB 并归一化到 [0,1]。
    返回：tracks_xy(N,T,2), visibility(N,T)
    """
    assert frames_bgr.ndim == 4 and frames_bgr.shape[-1] == 3
    frames_rgb = (frames_bgr[..., ::-1].astype(np.float32) / 255.0)  # BGR->RGB, [0,1]

    # 选择后端 + 单例缓存
    if backend == "jax" or (backend == "auto" and _jax_available()):
        be = _get_or_create_backend("jax", ckpt_dir, jax_ckpt_name, torch_ckpt_name)
        return be(frames_rgb, seeds_xy, seeds_t0)

    if backend == "torch" or (backend == "auto" and _torch_available()):
        be = _get_or_create_backend("torch", ckpt_dir, jax_ckpt_name, torch_ckpt_name)
        return be(frames_rgb, seeds_xy, seeds_t0)

    # 兜底：Lucas-Kanade（很快，但仅应在两种 TAPIR 都不可用时触发）
    import cv2 as cv
    N = int(seeds_xy.shape[0])
    T = frames_bgr.shape[0]
    tracks = np.zeros((N, T, 2), np.float32)
    vis = np.zeros((N, T), bool)
    grays = [cv.cvtColor(f, cv.COLOR_BGR2GRAY) for f in frames_bgr]
    p = seeds_xy.reshape(-1, 1, 2).astype(np.float32)
    H, W = grays[0].shape[:2]
    tracks[:, 0, :] = seeds_xy.astype(np.float32)
    vis[:, 0] = ((p[..., 0] >= 0) & (p[..., 0] < W) & (p[..., 1] >= 0) & (p[..., 1] < H)).reshape(-1)
    prev = grays[0]; prev_p = p.copy()
    term = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.03)
    for t in range(1, T):
        nxt = grays[t]
        nxt_p, st, err = cv.calcOpticalFlowPyrLK(prev, nxt, prev_p, None,
                                                 winSize=(21, 21), maxLevel=3, criteria=term)
        if nxt_p is None or st is None:
            st = np.zeros((N, 1), np.uint8); nxt_p = prev_p
        st = st.reshape(-1).astype(bool)
        coords = nxt_p.reshape(-1, 2)
        inb = (coords[:, 0] >= 0) & (coords[:, 0] < W) & (coords[:, 1] >= 0) & (coords[:, 1] < H)
        ok = st & inb
        tracks[:, t, :] = np.where(ok[:, None], coords, tracks[:, t - 1, :])
        vis[:, t] = ok
        prev, prev_p = nxt, nxt_p
    return tracks, vis


# 兼容别名（供 CLI 调用），接口不变
def track_points_tapir(frames, seeds_xy, cfg_tapir=None):
    """
    兼容两种配置：
      - torch-only: {'backend':'torch', 'weights':'/abs/path/to/bootstapir_checkpoint_v2.pt'}
      - 目录+文件名: {'backend':'torch', 'ckpt_dir': '/dir', 'torch_ckpt':'bootstapir_checkpoint_v2.pt'}
    """
    cfg = (cfg_tapir or {})
    backend = cfg.get("backend", "auto")

    # 优先识别绝对权重路径（常见写法）
    weights = cfg.get("weights", None)
    if isinstance(weights, str) and len(weights) > 0:
        ckpt_dir = os.path.dirname(weights)
        torch_ckpt = os.path.basename(weights)
    else:
        ckpt_dir = cfg.get("ckpt_dir", "/root/autodl-tmp/aigc_motion_eval/weights/tapnet_checkpoints")
        torch_ckpt = cfg.get("torch_ckpt", "bootstapir_checkpoint_v2.pt")

    jax_ckpt = cfg.get("jax_ckpt", "causal_tapir_checkpoint.npy")

    return run_tapir_tracks(
        frames_bgr=np.stack(frames, 0),
        seeds_xy=seeds_xy,
        seeds_t0=int(cfg.get("t0", 0)),
        backend=backend,
        ckpt_dir=ckpt_dir,
        jax_ckpt_name=jax_ckpt,
        torch_ckpt_name=torch_ckpt
    )
