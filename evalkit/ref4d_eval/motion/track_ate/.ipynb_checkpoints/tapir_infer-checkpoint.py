# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import numpy as np

def _jax_available() -> bool:
    try:
        import jax, haiku  # noqa
        import tapnet  # noqa
        return True
    except Exception:
        return False

def _torch_available() -> bool:
    try:
        import torch  # noqa
        from tapnet.torch import tapir_model as _  # noqa
        return True
    except Exception:
        return False

class TapirJAXBackend:
    """
    JAX/Haiku TAPIR 前向（首选）。
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

        try:
            from tapnet.jax import tapir as tapir_jax
            from tapnet.jax import model_utils as mu
        except Exception as e:
            raise ImportError(
                "Cannot import tapnet JAX modules. Install from DeepMind tapnet repo."
            ) from e

        self.mu = mu
        self.tapir = tapir_jax
        self.causal = causal

        def forward_fn(frames_bhwc: np.ndarray, query_xy1: np.ndarray):
            """
            frames_bhwc: [T,H,W,3], uint8 or float in [0,1] (RGB)
            query_xy1:   [N,3] -> (x,y,t_index)
            """
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
    PyTorch 版 TAPIR/BootsTAPIR 兜底。权重如：bootstapir_checkpoint_v2.pt
    """
    def __init__(self, ckpt_path_pt: str):
        import torch
        from tapnet.torch import tapir_model
        self.torch = torch
        self.model = tapir_model.TAPIR()
        if not os.path.isfile(ckpt_path_pt):
            raise FileNotFoundError(f"TAPIR Torch checkpoint not found: {ckpt_path_pt}")
        state = torch.load(ckpt_path_pt, map_location='cpu')
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    def __call__(self, frames_rgb: np.ndarray, seeds_xy: np.ndarray, seeds_t0: int = 0):
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        T, H, W, C = frames_rgb.shape
        video = frames_rgb.astype(np.float32)[None]  # [1,T,H,W,3] in [0,1]
        seeds_xy = seeds_xy.astype(np.float32)
        with torch.no_grad():
            v = torch.from_numpy(video).to(device)
            q = np.concatenate([seeds_xy, np.full((seeds_xy.shape[0], 1), seeds_t0, np.float32)], 1)
            q_t = torch.from_numpy(q[None]).to(device)  # [1,N,3]
            out = self.model.infer(v, q_t)
            tracks = out['tracks'][0].detach().float().cpu().numpy()  # [N,T,2]
            if 'visibility' in out:
                vis = out['visibility'][0].detach().cpu().numpy().astype(bool)
            else:
                occ = out.get('occluded')[0].detach().cpu().numpy().astype(bool)
                vis = ~occ
        return tracks, vis

def run_tapir_tracks(
    frames_bgr: np.ndarray,
    seeds_xy: np.ndarray,
    seeds_t0: int = 0,
    backend: str = "auto",
    ckpt_dir: str = "/root/autodl-tmp/aigc_motion_eval/third_party/tapnet_checkpoints",
    jax_ckpt_name: str = "causal_tapir_checkpoint.npy",
    torch_ckpt_name: str = "bootstapir_checkpoint_v2.pt",
):
    """
    统一入口：优先 JAX；失败则 Torch；再失败则 LK 光流跟踪。
    frames_bgr: [T,H,W,3] (BGR)，内部转 RGB 并归一化到 [0,1]。
    返回：tracks_xy(N,T,2), visibility(N,T)
    """
    assert frames_bgr.ndim == 4 and frames_bgr.shape[-1] == 3
    frames_rgb = (frames_bgr[..., ::-1].astype(np.float32) / 255.0)  # BGR->RGB, [0,1]

    if backend == "jax" or (backend == "auto" and _jax_available()):
        jax_ckpt = os.path.join(ckpt_dir, jax_ckpt_name)
        tapir = TapirJAXBackend(jax_ckpt, causal=True)
        return tapir(frames_rgb, seeds_xy, seeds_t0)

    if backend == "torch" or (backend == "auto" and _torch_available()):
        torch_ckpt = os.path.join(ckpt_dir, torch_ckpt_name)
        tapir = TapirTorchBackend(torch_ckpt)
        return tapir(frames_rgb, seeds_xy, seeds_t0)

    # 兜底：LK
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

# 兼容别名
def track_points_tapir(frames, seeds_xy, cfg_tapir=None):
    backend = (cfg_tapir or {}).get("backend", "auto")
    ckpt_dir = (cfg_tapir or {}).get("ckpt_dir",
        "/root/autodl-tmp/aigc_motion_eval/third_party/tapnet_checkpoints")
    jax_ckpt = (cfg_tapir or {}).get("jax_ckpt", "causal_tapir_checkpoint.npy")
    torch_ckpt = (cfg_tapir or {}).get("torch_ckpt", "bootstapir_checkpoint_v2.pt")
    return run_tapir_tracks(
        frames_bgr=np.stack(frames, 0), seeds_xy=seeds_xy,
        backend=backend, ckpt_dir=ckpt_dir,
        jax_ckpt_name=jax_ckpt, torch_ckpt_name=torch_ckpt
    )
