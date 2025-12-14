# softalign/similarity.py
from __future__ import annotations

import numpy as np
from typing import List, Tuple

from .config import Config
from .types import EntityRepr
from .encoder import TextEncoder


__all__ = [
    "cosine_to_unit",
    "pairwise_cosine",
    "pairwise_similarity",
    "fuse_channels",
    "value_pairwise_similarity",
]


# -----------------------------
# 余弦 → [0,1] 单调映射（可校准）
# -----------------------------
def _tau0(cfg: Config) -> float:
    s = getattr(cfg, "sim", None)
    return float(getattr(s, "tau0", 0.30))


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0, out=x)


def cosine_to_unit(cosine: np.ndarray, tau0: float) -> np.ndarray:
    """
    单调线性映射：
      s = clip((cos - tau0) / (1 - tau0), 0, 1)
    tau0 ∈ [0,1)
    """
    s = (cosine - tau0) / max(1e-12, (1.0 - tau0))
    return _clip01(s)


# -----------------------------
# 通道内余弦计算
# -----------------------------
def pairwise_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    计算两组单位向量的余弦相似度矩阵：
      A: [n, d], B: [m, d]  ->  [n, m]
    要求 A/B 已 L2 归一（编码器与集合池化保证）。
    """
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    return (A @ B.T).astype(np.float32, copy=False)


# -----------------------------
# 两通道融合（name / set）
# -----------------------------
def _combine_mode(cfg: Config) -> str:
    s = getattr(cfg, "sim", None)
    return str(getattr(s, "combine", "max")).lower()  # 'max' | 'weighted'


def _combine_weights(cfg: Config) -> Tuple[float, float]:
    s = getattr(cfg, "sim", None)
    w_name = float(getattr(s, "w_name", 0.5))
    w_set = float(getattr(s, "w_set", 0.5))
    S = max(1e-12, w_name + w_set)
    return w_name / S, w_set / S


def fuse_channels(name_s: np.ndarray, set_s: np.ndarray, cfg: Config) -> np.ndarray:
    """
    通道融合：
      - 'max'：逐对取最大值（有则加分、无则不扣）
      - 'weighted'：按 (w_name, w_set) 加权
    """
    mode = _combine_mode(cfg)
    if mode == "weighted":
        wn, ws = _combine_weights(cfg)
        return (wn * name_s + ws * set_s).astype(np.float32, copy=False)
    return np.maximum(name_s, set_s).astype(np.float32, copy=False)


# -----------------------------
# 实体表示 → 实体级相似度矩阵（[0,1]）
# -----------------------------
def pairwise_similarity(R: List[EntityRepr], G: List[EntityRepr], cfg: Config) -> np.ndarray:
    tau0 = _tau0(cfg)

    A_name = np.stack([e.name_vec for e in R], axis=0) if R else np.zeros((0, 0), dtype=np.float32)
    B_name = np.stack([e.name_vec for e in G], axis=0) if G else np.zeros((0, 0), dtype=np.float32)

    A_set = np.stack([e.set_vec for e in R], axis=0) if R else np.zeros((0, 0), dtype=np.float32)
    B_set = np.stack([e.set_vec for e in G], axis=0) if G else np.zeros((0, 0), dtype=np.float32)

    cos_name = pairwise_cosine(A_name, B_name)  # [|R|, |G|]
    cos_set = pairwise_cosine(A_set, B_set)

    s_name = cosine_to_unit(cos_name, tau0)
    s_set = cosine_to_unit(cos_set, tau0)

    return fuse_channels(s_name, s_set, cfg)


# -----------------------------
# 属性值对相似（用于 AIC 的 s(v, v')）
# -----------------------------
def value_pairwise_similarity(
    vals_left: List[str],
    vals_right: List[str],
    encoder: TextEncoder,
    cfg: Config,
    *,
    purpose: str = "passage",
    max_length: int = 32,
) -> np.ndarray:
    if not vals_left or not vals_right:
        return np.zeros((len(vals_left), len(vals_right)), dtype=np.float32)

    # 与集合通道保持一致的 purpose
    emb_l = [e.vec for e in encoder.embed_texts(vals_left, purpose=purpose, max_length=max_length)]
    emb_r = [e.vec for e in encoder.embed_texts(vals_right, purpose=purpose, max_length=max_length)]

    A = np.stack(emb_l, axis=0)
    B = np.stack(emb_r, axis=0)

    cos = pairwise_cosine(A, B)
    return cosine_to_unit(cos, _tau0(cfg))
