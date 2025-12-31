# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2 as cv

__all__ = ["sample_points_strict", "sample_fg_bg_points_strict"]


def _ensure_bool(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    if m.dtype != np.bool_:
        m = m.astype(bool)
    return m


def _texture_score(gray: np.ndarray) -> np.ndarray:
    """
    返回逐像素纹理分数 map ∈ [0,1]（Sobel 幅值分位数归一化）。
    """
    g = gray.astype(np.float32) / 255.0
    gx = cv.Sobel(g, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(g, cv.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    # 以 P95 做鲁棒归一化，避免极端大值把整体压扁
    p95 = float(np.quantile(mag, 0.95)) if np.isfinite(mag).all() else 0.0
    if p95 > 1e-6:
        mag = np.clip(mag / p95, 0.0, 1.0)
    else:
        mag = np.zeros_like(mag, dtype=np.float32)
    return mag


def _apply_border(mask: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return mask
    H, W = mask.shape[:2]
    out = np.zeros_like(mask, dtype=bool)
    out[border:H - border, border:W - border] = mask[border:H - border, border:W - border]
    return out


def _edge_bonus_weights(mask: np.ndarray, edge_bonus: bool, w_edge: float = 2.0) -> np.ndarray:
    if not edge_bonus:
        return mask.astype(np.float32)
    m = mask.astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8)
    eroded = cv.erode(m, k, iterations=1)
    edge = (m > 0) & (eroded == 0)
    w = mask.astype(np.float32)
    w[edge] *= float(w_edge)
    return w


def _weighted_choice(coords_xy: np.ndarray, weights: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    N = int(coords_xy.shape[0])
    if k > N:
        raise ValueError(f"not enough candidate pixels: need {k}, have {N}")
    w = np.asarray(weights, dtype=np.float64)
    s = w.sum()
    if (not np.isfinite(s)) or s <= 0:
        raise ValueError("invalid weights for sampling (sum<=0 or non-finite)")
    p = w / s
    idx = rng.choice(N, size=k, replace=False, p=p)
    return coords_xy[idx]


def _texture_mag01(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32) / 255.0
    gx = cv.Sobel(g, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(g, cv.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mmax = float(mag.max()) if mag.size and np.isfinite(mag.max()) else 0.0
    return mag / (mmax + 1e-6)


def sample_points_strict(
    mask: np.ndarray,
    N: int,
    *,
    border: int = 4,
    edge_bonus: bool = True,
    min_tex: Optional[float] = None,
    frame_bgr: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    # 新增：与调用点兼容（最小入侵）
    adaptive: bool = True,
    min_take: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在“给定掩码”内撒点（支持自适应）：
      - 先做边界裁切与可选纹理门控；
      - 若纹理门控后为空，则软回退：用原掩码，但以纹理分数作为采样权重；
      - adaptive=True: 采样 K = max(min_take, min(N, 可用像素数))；
        adaptive=False: 若可用像素数 < N，直接报错；
      - 返回 (xy[K,2], wtex[K])，wtex ∈ [0,1] 作为纹理置信。
    """
    if rng is None:
        rng = np.random.default_rng(2025)
    if N <= 0:
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.float32)

    m = _ensure_bool(mask)
    m = _apply_border(m, border=border)
    if m.sum() == 0:
        raise ValueError("empty mask after border cropping")

    # 纹理 map（即便 min_tex=None 也计算，用作软权重）
    if frame_bgr is not None:
        gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
        tex_map = _texture_score(gray)  # [H,W] ∈ [0,1]
    else:
        tex_map = np.ones_like(m, dtype=np.float32)

    # 硬门控 + 软回退
    m_tex = m
    if (min_tex is not None) and (frame_bgr is not None):
        m_hard = m & (tex_map > float(min_tex))
        if m_hard.sum() > 0:
            m_tex = m_hard     # 正常硬门控
        else:
            m_tex = m          # 软回退：保留原掩码

    ys, xs = np.nonzero(m_tex)
    avail = int(ys.size)
    if avail == 0:
        raise ValueError("no pixels inside mask after texture handling")

    coords_xy = np.stack([xs, ys], axis=1).astype(np.float32)  # (x,y)

    # 组合采样权重：边界增益 * 纹理分数（加 eps 防 0）
    w_edge_map = _edge_bonus_weights(m_tex, edge_bonus=edge_bonus)  # [H,W]
    w_pick = (w_edge_map[ys, xs].astype(np.float64) *
              (tex_map[ys, xs].astype(np.float64) + 1e-6))

    s = float(w_pick.sum())
    if (not np.isfinite(s)) or s <= 0:
        w_pick = np.ones_like(w_pick, dtype=np.float64)
        s = float(w_pick.sum())
    p = w_pick / s

    if adaptive:
        K = int(max(int(min_take), min(int(N), avail)))
    else:
        if avail < int(N):
            raise ValueError(f"not enough pixels inside mask: need {N}, have {avail}")
        K = int(N)

    if K == 0:
        # 极端情况：允许返回空
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.float32)

    idx = rng.choice(avail, size=K, replace=False, p=p)
    picked_xy = coords_xy[idx].astype(np.float32)
    picked_wtex = tex_map[ys, xs][idx].astype(np.float32)  # ∈[0,1]

    return picked_xy, picked_wtex


def sample_fg_bg_points_strict(
    masks_fg: List[Optional[np.ndarray]],
    masks_bg: List[Optional[np.ndarray]],
    *,
    num_fg: int,
    num_bg: int,
    border: int = 4,
    edge_bonus: bool = True,
    min_tex: Optional[float] = None,
    frames_bgr: Optional[List[np.ndarray]] = None,
    rng: Optional[np.random.Generator] = None,
    # —— 每 K 帧一个起点 ——
    t0_stride: int = 1,
    t0_offset: int = 0,
    # 自适应参数（与调用点保持一致）
    adaptive: bool = True,
    min_take_fg: int = 1,
    min_take_bg: int = 1,
) -> Dict[str, np.ndarray]:
    """
    仅在满足 (t - t0_offset) % t0_stride == 0 的帧撒点，并将该帧索引作为 t0。
    跳过 None 掩码帧；支持自适应像素不足。
    返回：
      fg_xy, bg_xy: (N_all,2) float32
      fg_t0, bg_t0: (N_all,)  int32
      fg_wtex, bg_wtex: (N_all,) float32  ∈[0,1]
    """
    if rng is None:
        rng = np.random.default_rng(2025)
    if frames_bgr is None and min_tex is not None:
        raise ValueError("min_tex gating requires frames_bgr")
    assert len(masks_fg) == len(masks_bg), "masks_fg/masks_bg length mismatch"

    T = len(masks_fg)
    fg_xy_list, fg_w_list, fg_t0_list = [], [], []
    bg_xy_list, bg_w_list, bg_t0_list = [], [], []

    for t in range(T):
        # 仅关键帧起点
        if ((t - int(t0_offset)) % max(1, int(t0_stride))) != 0:
            continue
        mfg = masks_fg[t]
        mbg = masks_bg[t]
        if (mfg is None) or (mbg is None):
            # 最小入侵：跳过 None 帧
            continue
        f = frames_bgr[t] if frames_bgr is not None else None

        # 前景
        fg_xy, fg_wtex = sample_points_strict(
            mfg, num_fg, border=border, edge_bonus=edge_bonus,
            min_tex=min_tex, frame_bgr=f, rng=rng,
            adaptive=adaptive, min_take=min_take_fg,
        )
        # 背景
        bg_xy, bg_wtex = sample_points_strict(
            mbg, num_bg, border=border, edge_bonus=edge_bonus,
            min_tex=min_tex, frame_bgr=f, rng=rng,
            adaptive=adaptive, min_take=min_take_bg,
        )

        if fg_xy.shape[0] == 0 and bg_xy.shape[0] == 0:
            continue

        if fg_xy.shape[0] > 0:
            fg_xy_list.append(fg_xy)
            fg_w_list.append(fg_wtex)
            fg_t0_list.append(np.full((fg_xy.shape[0],), t, np.int32))
        if bg_xy.shape[0] > 0:
            bg_xy_list.append(bg_xy)
            bg_w_list.append(bg_wtex)
            bg_t0_list.append(np.full((bg_xy.shape[0],), t, np.int32))

    if (not fg_xy_list) and (not bg_xy_list):
        raise ValueError("no seeds after applying t0_stride/t0_offset and skipping None masks")

    def _cat(lst, axis=0, dtype=None):
        if not lst:
            return np.zeros((0,), dtype=dtype if dtype is not None else np.float32)
        arr = np.concatenate(lst, axis=axis)
        return arr.astype(dtype) if dtype is not None else arr

    fg_xy = _cat(fg_xy_list, axis=0, dtype=np.float32)
    bg_xy = _cat(bg_xy_list, axis=0, dtype=np.float32)
    fg_w = _cat(fg_w_list, axis=0, dtype=np.float32)
    bg_w = _cat(bg_w_list, axis=0, dtype=np.float32)
    fg_t0 = _cat(fg_t0_list, axis=0, dtype=np.int32)
    bg_t0 = _cat(bg_t0_list, axis=0, dtype=np.int32)

    return {
        "fg_xy": fg_xy, "bg_xy": bg_xy,
        "fg_t0": fg_t0, "bg_t0": bg_t0,
        "fg_wtex": fg_w, "bg_wtex": bg_w,
    }
