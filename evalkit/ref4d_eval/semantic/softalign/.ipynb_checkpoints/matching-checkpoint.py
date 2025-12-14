# softalign/matching.py
from __future__ import annotations

import warnings
from typing import List, Tuple

import numpy as np

from .config import Config
from .types import MatchPair, MatchResult


__all__ = [
    "hungarian_match",
    "greedy_match",
    "compute_matching",
]


# ----------------------------
# 配置：门控阈值
# ----------------------------
def _min_match_score(cfg: Config) -> float:
    m = getattr(cfg, "matching", None)
    return float(getattr(m, "min_score", 0.30))


def _unmatch_cost_from_gate(gate: float) -> float:
    """
    构造最小化代价问题：
      - 真实配对 cost = 1 - s
      - 未匹配（连各自 dummy）cost = 1 - gate
    只有当 s > gate 时才更倾向连线。
    """
    gate = float(max(0.0, min(0.999999, gate)))
    return 1.0 - gate


# ----------------------------
# 将相似度矩阵扩展为“带 dummy 的方阵成本”
# ----------------------------
def _build_padded_cost(sim: np.ndarray, gate: float) -> Tuple[np.ndarray, int, int]:
    R, G = sim.shape
    N = R + G
    if N == 0:
        return np.zeros((0, 0), dtype=np.float32), R, G

    M = np.full((N, N), fill_value=1e6, dtype=np.float32)

    # 左上角：真实配对
    if R > 0 and G > 0:
        M[:R, :G] = (1.0 - sim).astype(np.float32, copy=False)

    c_unmatch = _unmatch_cost_from_gate(gate)

    # 右上角：参考侧 dummy 列（对角可用）
    for i in range(R):
        M[i, G + i] = c_unmatch

    # 左下角：生成侧 dummy 行（对角可用）
    for j in range(G):
        M[R + j, j] = c_unmatch

    # 右下角：dummy-dummy（可为 0）
    if R > 0 and G > 0:
        M[R:, G:] = 0.0

    return M, R, G


# ----------------------------
# Hungarian（SciPy） + 贪心回退
# ----------------------------
def _hungarian_min_cost(cost: np.ndarray):
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
    except Exception as e:
        raise ImportError("需要 scipy.optimize.linear_sum_assignment 以使用 Hungarian") from e
    row_ind, col_ind = linear_sum_assignment(cost)
    total = float(cost[row_ind, col_ind].sum())
    return row_ind, col_ind, total


def _greedy_maximum(sim: np.ndarray, gate: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    R, G = sim.shape
    used_r = np.zeros(R, dtype=bool)
    used_g = np.zeros(G, dtype=bool)
    pairs: List[Tuple[int, int]] = []
    while True:
        best = gate
        bi = -1
        bj = -1
        for i in range(R):
            if used_r[i]:
                continue
            for j in range(G):
                if used_g[j]:
                    continue
                s = sim[i, j]
                if s > best:
                    best = s
                    bi, bj = i, j
        if bi < 0 or bj < 0:
            break
        pairs.append((bi, bj))
        used_r[bi] = True
        used_g[bj] = True
    ref_un = [i for i in range(R) if not used_r[i]]
    gen_un = [j for j in range(G) if not used_g[j]]
    return pairs, ref_un, gen_un


# ----------------------------
# 对外：Hungarian 匹配（带门控）
# ----------------------------
def hungarian_match(sim: np.ndarray, gate: float) -> MatchResult:
    sim = np.asarray(sim, dtype=np.float32)
    R, G = sim.shape
    if R == 0 and G == 0:
        return MatchResult(pairs=[], method="hungarian")

    C, R0, G0 = _build_padded_cost(sim, gate)

    used_backup = False
    try:
        rows, cols, _ = _hungarian_min_cost(C)
    except Exception:
        warnings.warn("[softalign.matching] SciPy 不可用，回退贪心匹配（近似最优）", RuntimeWarning)
        used_backup = True
        pairs_greedy, _, _ = _greedy_maximum(sim, gate)
        pairs = [MatchPair(r_idx=i, g_idx=j, score=float(sim[i, j])) for (i, j) in pairs_greedy]
        return MatchResult(pairs=sorted(pairs, key=lambda x: (-x.score, x.r_idx, x.g_idx)), method="greedy")

    pairs: List[MatchPair] = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        if r < R0 and c < G0:
            s = float(sim[r, c])
            if s > gate:
                pairs.append(MatchPair(r_idx=r, g_idx=c, score=s))

    return MatchResult(
        pairs=sorted(pairs, key=lambda x: (-x.score, x.r_idx, x.g_idx)),
        method=("hungarian" if not used_backup else "greedy"),
    )


# ----------------------------
# 统一入口
# ----------------------------
def greedy_match(sim: np.ndarray, gate: float) -> MatchResult:
    pairs_greedy, _, _ = _greedy_maximum(np.asarray(sim, dtype=np.float32), gate)
    pairs = [MatchPair(r_idx=i, g_idx=j, score=float(sim[i, j])) for (i, j) in pairs_greedy]
    return MatchResult(pairs=sorted(pairs, key=lambda x: (-x.score, x.r_idx, x.g_idx)), method="greedy")


def compute_matching(sim: np.ndarray, cfg: Config) -> MatchResult:
    algo = str(getattr(getattr(cfg, "matching", None), "algorithm", "hungarian")).lower()
    gate = _min_match_score(cfg)
    if algo == "greedy":
        return greedy_match(sim, gate)
    return hungarian_match(sim, gate)
