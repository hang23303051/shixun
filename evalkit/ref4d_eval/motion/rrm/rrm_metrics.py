# -*- coding: utf-8 -*-
"""
RRM metrics: 将特征差异映射为距离与分数。
- D_dir: 方向分布（HoF, 16-bin）EMD（循环最优旋转）
- D_mag: 相对强度（|r(t)|）分布差异（1D Wasserstein/EMD 近似）
- D_smo: 平滑/jerk 的能量比例差异（支持 phi 为标量或向量/字典）

S_k = 100 * exp(-lambda_k * D_k)
S_motion = sum_k alpha_k * S_k * (1 - eta * FRZ)
"""

from __future__ import annotations
from typing import Dict, Any, Iterable, Tuple
import numpy as np

__all__ = ["distances", "to_scores", "aggregate"]


# --------- helpers ---------

def _safe_hist(h: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    s = h.sum()
    if not np.isfinite(s) or s <= 0:
        # 避免全零/NaN
        h = np.ones_like(h, dtype=np.float64)
        s = float(h.sum())
    return (h / (s + eps)).astype(np.float64)

def _emd_1d_unit(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    1D 离散 EMD（相邻 bin 距离=1）= sum |CDF1 - CDF2|
    """
    diff = np.cumsum(h1 - h2)
    return float(np.sum(np.abs(diff)))

def _emd_circular_minrot(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    环形直方图的 EMD：在所有循环位移上取最小 1D EMD。
    """
    h1 = _safe_hist(h1)
    h2 = _safe_hist(h2)
    B = h1.shape[0]
    best = np.inf
    for k in range(B):
        e = _emd_1d_unit(h1, np.roll(h2, k))
        if e < best:
            best = e
    return float(best)

def _wasserstein_1d_from_samples(x: np.ndarray, y: np.ndarray) -> float:
    """
    简洁的 1D Wasserstein：|CDF_x - CDF_y| 在所有样本并集上的积分近似。
    这里做无依赖近似（避免强依赖 SciPy）：按排序后累积差 L1。
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0:
        return 0.0
    xs = np.sort(x); ys = np.sort(y)
    # 将两边重采样到相同长度（分位点匹配）
    n = max(len(xs), len(ys))
    q = np.linspace(0.0, 1.0, n, endpoint=True)
    xq = np.quantile(xs, q, method="linear")
    yq = np.quantile(ys, q, method="linear")
    return float(np.mean(np.abs(xq - yq)))

def _phi_to_vec(phi: Any) -> np.ndarray:
    """
    将 phi 标准化为 1D 向量：
      - scalar -> [scalar]
      - dict   -> 按 key 排序取值
      - array  -> ravel()
    """
    if isinstance(phi, dict):
        keys = sorted(list(phi.keys()))
        vec = np.array([phi[k] for k in keys], dtype=np.float64)
        return vec.reshape(-1)
    arr = np.asarray(phi)
    if arr.ndim == 0:
        return np.array([float(arr)], dtype=np.float64)
    return arr.astype(np.float64).ravel()

def _l1_prob_dist(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """
    将向量视为概率分布（非负、归一化），计算 L1 差。
    """
    a = np.maximum(a, 0.0); b = np.maximum(b, 0.0)
    sa, sb = a.sum(), b.sum()
    if sa <= 0: a = np.ones_like(a); sa = a.sum()
    if sb <= 0: b = np.ones_like(b); sb = b.sum()
    a = a / (sa + eps); b = b / (sb + eps)
    return float(np.mean(np.abs(a - b)))


# --------- core distances ---------

def distances(
    feats_ref: Dict[str, Any],
    feats_gen: Dict[str, Any],
    *,
    eps: float = 1e-3,
) -> Dict[str, float]:
    """
    输入：
      feats_* 包含：
        - "hof": 方向直方图（16-bin 或同维），非负
        - "s":   |r(t)| 样本（1D 向量）或已做分布特征
        - "phi": 平滑/jerk 能量比例（可为标量、向量或 dict）

    输出：
      {"dir": D_dir, "mag": D_mag, "smo": D_smo}
    """
    # --- DIR: circular HoF EMD ---
    h_ref = np.asarray(feats_ref["hof"], dtype=np.float64).reshape(-1)
    h_gen = np.asarray(feats_gen["hof"], dtype=np.float64).reshape(-1)
    D_dir = _emd_circular_minrot(h_ref, h_gen)

    # --- MAG: 1D Wasserstein from samples (或已是直方图时近似为样本重采样) ---
    s_ref = np.asarray(feats_ref["s"], dtype=np.float64).reshape(-1)
    s_gen = np.asarray(feats_gen["s"], dtype=np.float64).reshape(-1)
    # 允许存在 NaN/Inf，先清洗
    s_ref = s_ref[np.isfinite(s_ref)]
    s_gen = s_gen[np.isfinite(s_gen)]
    D_mag = _wasserstein_1d_from_samples(s_ref, s_gen)

    # --- SMO: 支持 phi 为标量/向量/dict ---
    phi_ref_vec = _phi_to_vec(feats_ref["phi"])
    phi_gen_vec = _phi_to_vec(feats_gen["phi"])
    # 对齐维度（取较小长度，保持最小改动）
    m = min(phi_ref_vec.size, phi_gen_vec.size)
    if m <= 0:
        D_smo = 0.0
    else:
        D_smo = _l1_prob_dist(phi_ref_vec[:m], phi_gen_vec[:m])

    return {"dir": float(D_dir), "mag": float(D_mag), "smo": float(D_smo)}


# --------- scoring & aggregation ---------

def to_scores(
    D: Dict[str, float],
    lambdas: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    S_k = 100 * exp(-lambda_k * D_k)
    """
    lam = {"dir": 4.0, "mag": 2.0, "smo": 3.0}
    if lambdas:
        lam.update({k: float(v) for k, v in lambdas.items() if k in lam})

    S = {
        k: float(100.0 * np.exp(-float(lam[k]) * float(D[k])))
        for k in ("dir", "mag", "smo")
    }
    return S

def aggregate(
    S: Dict[str, float],
    frz: float,
    alphas: Dict[str, float] = None,
    eta: float = 0.6,
) -> float:
    """
    S_motion = sum_k alpha_k * S_k * (1 - eta * FRZ)
    """
    a = {"dir": 0.35, "mag": 0.30, "smo": 0.35}
    if alphas:
        for k in a.keys():
            if k in alphas:
                a[k] = float(alphas[k])

    base = a["dir"] * float(S["dir"]) + a["mag"] * float(S["mag"]) + a["smo"] * float(S["smo"])
    out = base * (1.0 - float(eta) * float(frz))
    return float(out)
