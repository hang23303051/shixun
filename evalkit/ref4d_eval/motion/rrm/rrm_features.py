# -*- coding: utf-8 -*-
"""
RRM features: 从前景/背景点轨迹得到相对速度 r(t)，并计算
- s(t)=|r(t)|, theta(t)=atan2(ry, rx)
- 方向直方图 HOF
- 平滑度谱 phi（供 R-SMO）
支持 “遮挡/纹理置信给权”：权重 = visibility * wtex（wtex∈[0,1]，来自撒点阶段的纹理强度）
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np


__all__ = ["RRMFeaturePack", "compute_all"]


@dataclass
class RRMFeaturePack:
    # 基本序列
    r: np.ndarray         # [T-1, 2]
    s: np.ndarray         # [T-1]
    theta: np.ndarray     # [T-1]
    # 方向分布（单位圆上做直方图）
    hof: np.ndarray       # [dir_bins]
    # 平滑度相关（R-SMO 用）：一/二阶差分能量占比
    phi: Dict[str, float] # {"e1":..., "e2":..., "ratio":...}


def _safe_mean_2d(vecs: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    vecs: [N,2]; w: [N] 或 [N,1]
    返回加权均值，若权重和=0则返回0向量（避免传播NaN）
    """
    w = w.reshape(-1, 1).astype(np.float32)
    den = float(np.sum(w)) + eps
    if den <= eps:
        return np.zeros((2,), np.float32)
    return (vecs * w).sum(axis=0) / den


def _make_hof(theta: np.ndarray, bins: int = 16) -> np.ndarray:
    """
    在 [-pi, pi) 上做有符号方向直方图（HOF），归一化到和为1
    """
    edges = np.linspace(-np.pi, np.pi, num=bins + 1, dtype=np.float32)
    hist, _ = np.histogram(theta.astype(np.float32), bins=edges)
    h = hist.astype(np.float32)
    s = h.sum()
    return h if s <= 0 else (h / s)


def _smoothness_phi(r: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
    """
    基于 r(t) 的一/二阶差分能量，给出比例（体现 jerk）
    """
    if r.shape[0] < 3:
        return {"e1": 0.0, "e2": 0.0, "ratio": 0.0}
    dr = np.diff(r, axis=0)                 # [T-2,2]
    d2r = np.diff(dr, axis=0)               # [T-3,2]
    e1 = float(np.mean(np.sum(dr * dr, axis=1))) if dr.size else 0.0
    e2 = float(np.mean(np.sum(d2r * d2r, axis=1))) if d2r.size else 0.0
    ratio = e2 / (e1 + eps)
    return {"e1": e1, "e2": e2, "ratio": ratio}


def compute_all(
    tracks_fg: np.ndarray, vis_fg: np.ndarray,
    tracks_bg: np.ndarray, vis_bg: np.ndarray,
    *,
    dir_bins: int = 16,
    wtex_fg: Optional[np.ndarray] = None,  # NEW: 每个前景点的纹理权重 ∈[0,1]，长度=N_fg
    wtex_bg: Optional[np.ndarray] = None,  # NEW: 每个背景点的纹理权重 ∈[0,1]，长度=N_bg
    eps: float = 1e-6,
) -> Tuple[np.ndarray, RRMFeaturePack]:
    """
    输入：
      tracks_* : [N,T,2]
      vis_*    : [N,T] (bool)
      wtex_*   : [N] 或 None（None 时视为全1）
    输出：
      r(t)、以及打包的特征
    """
    assert tracks_fg.ndim == 3 and tracks_bg.ndim == 3
    assert tracks_fg.shape[2] == 2 and tracks_bg.shape[2] == 2
    assert vis_fg.shape[:2] == tracks_fg.shape[:2]
    assert vis_bg.shape[:2] == tracks_bg.shape[:2]

    Nf, T, _ = tracks_fg.shape
    Nb, T2, _ = tracks_bg.shape
    assert T == T2 and T >= 2, "tracks time length mismatch or too short"

    # 位移（需要 t 与 t-1 可见）：采用 “相邻帧都可见” 的遮挡门控
    u_fg = tracks_fg[:, 1:, :] - tracks_fg[:, :-1, :]   # [Nf,T-1,2]
    u_bg = tracks_bg[:, 1:, :] - tracks_bg[:, :-1, :]   # [Nb,T-1,2]
    m_fg = (vis_fg[:, 1:] & vis_fg[:, :-1]).astype(np.float32)  # [Nf,T-1]
    m_bg = (vis_bg[:, 1:] & vis_bg[:, :-1]).astype(np.float32)  # [Nb,T-1]

    # 纹理置信（静态、逐点）：若未提供，则默认为1
    if wtex_fg is None:
        wtex_fg = np.ones((Nf,), np.float32)
    if wtex_bg is None:
        wtex_bg = np.ones((Nb,), np.float32)
    wtex_fg = np.clip(wtex_fg.astype(np.float32), 0.0, 1.0).reshape(Nf, 1)  # [Nf,1]
    wtex_bg = np.clip(wtex_bg.astype(np.float32), 0.0, 1.0).reshape(Nb, 1)  # [Nb,1]

    # 权重 = 遮挡门控 * 纹理置信
    w_fg = m_fg * wtex_fg                         # [Nf,T-1]
    w_bg = m_bg * wtex_bg                         # [Nb,T-1]

    # 逐时刻加权平均速度
    v_fg = np.zeros((T - 1, 2), np.float32)
    v_bg = np.zeros((T - 1, 2), np.float32)
    for t in range(T - 1):
        v_fg[t] = _safe_mean_2d(u_fg[:, t, :], w_fg[:, t], eps=eps)
        v_bg[t] = _safe_mean_2d(u_bg[:, t, :], w_bg[:, t], eps=eps)

    # 相对速度 r(t) 及派生量
    r = v_fg - v_bg                                # [T-1,2]
    s = np.linalg.norm(r, axis=1)                  # [T-1]
    theta = np.arctan2(r[:, 1], r[:, 0])           # [T-1]
    hof = _make_hof(theta, bins=dir_bins)          # [bins]
    phi = _smoothness_phi(r)

    pack = RRMFeaturePack(r=r, s=s, theta=theta, hof=hof, phi=phi)
    return r, pack
