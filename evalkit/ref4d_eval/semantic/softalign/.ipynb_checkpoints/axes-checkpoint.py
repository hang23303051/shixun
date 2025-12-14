# softalign/axes.py
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np

from .config import Config
from .types import (
    EntityRepr,
    MatchPair,
    MatchResult,
    CatCovPairDetail,
    AICCoverageByKey,
    AICMisbindItem,
    AICPairDetail,
    ParExtraCategory,
    ParExtraAttr,
)
from .similarity import pairwise_similarity, value_pairwise_similarity
from .matching import compute_matching

_EPS = 1e-8


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den + _EPS)


def _key_weight(cfg: Config, key: str) -> float:
    """结构权重（非词表），未配置时回落 1.0。"""
    rp = getattr(cfg, "repr", None)
    if rp and isinstance(getattr(rp, "key_weight", None), dict):
        try:
            return float(rp.key_weight.get(key, 1.0))
        except Exception:
            return 1.0
    return 1.0


def _extract_kv_triplets(ent: EntityRepr) -> List[Tuple[str, str]]:
    """将实体属性字典展开为 (key, value) 列表；缺省返回空列表。"""
    out: List[Tuple[str, str]] = []
    attrs = getattr(ent.entity, "attrs", {}) or {}
    for k, vs in attrs.items():
        if not vs:
            continue
        for v in vs:
            if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                out.append((k, v))
    return out


def _build_ref_bank_by_key(R: List[EntityRepr]) -> Dict[str, List[Tuple[int, str]]]:
    """AIC / Parsimony 预构建：key -> [(ref_idx, value), ...]"""
    bank: Dict[str, List[Tuple[int, str]]] = {}
    for i, e in enumerate(R):
        for k, v in _extract_kv_triplets(e):
            bank.setdefault(k, []).append((i, v))
    return bank


# ========== 1) SoftCatCov：软类别覆盖 ==========
class SoftCatCovCalculator:
    """参考每个实体在生成侧找到一个最相近匹配获得部分分；未匹配得 0。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def compute(
        self,
        R: List[EntityRepr],
        G: List[EntityRepr],
        *,
        sim_matrix: Optional[np.ndarray] = None,
        matching: Optional[MatchResult] = None,
    ) -> Tuple[float, List[CatCovPairDetail]]:
        if sim_matrix is None:
            sim_matrix = pairwise_similarity(R, G, self.cfg)  # [|R|, |G|]
        if matching is None:
            matching = compute_matching(sim_matrix, self.cfg)

        Rn = len(R)
        cov_per_r = [0.0] * Rn
        matched_g_by_r = {p.r_idx: p.g_idx for p in matching.pairs}

        details: List[CatCovPairDetail] = []
        for ri in range(Rn):
            if ri in matched_g_by_r:
                gi = matched_g_by_r[ri]
                s = float(sim_matrix[ri, gi])
                cov_per_r[ri] = s
                details.append(CatCovPairDetail(r_idx=ri, g_idx=gi, sim=s, note="matched"))
            else:
                details.append(CatCovPairDetail(r_idx=ri, g_idx=-1, sim=0.0, note="unmatched"))

        coverage = (sum(cov_per_r) / max(1, Rn)) if Rn > 0 else 0.0
        score = 100.0 * coverage
        return score, details


# ========== 2) SoftAIC：Coverage × (1 − Misbind) ==========
class SoftAICCalculator:
    """
    - 覆盖 Coverage：同 key 下，参考值 v 对生成值集 {v'} 的最大相似，按 w_k 加权归一。
    - 错绑 Misbind：对生成 (k, v')，计算 s*（对任一参考实体/同 key 的最佳相似）
                    与 s^(r)（对当前配对 r/同 key 的最佳相似），
                    Δ = max(0, s* − s^(r))；按 w_k 加权，
                    Misbind = Σ w_k·Δ / (Σ w_k·s* + ε)。
    - 样本级：对所有 (r,g)∈匹配集，按参考侧做 1/|R| 平均。
    """

    def __init__(self, cfg: Config, encoder):
        self.cfg = cfg
        self.encoder = encoder

    def _coverage_for_pair(self, r: EntityRepr, g: EntityRepr) -> Tuple[float, float, List[AICCoverageByKey]]:
        UR = _extract_kv_triplets(r)
        if not UR:
            return 0.0, 0.0, []

        # 生成侧同 key 值集合
        G_dict: Dict[str, List[str]] = {}
        for k, v in _extract_kv_triplets(g):
            G_dict.setdefault(k, []).append(v)

        numer = 0.0
        denom = 0.0
        by_key_details: List[AICCoverageByKey] = []

        # 按参考的每个 (k,v) 统计命中；同 key 聚合
        # 这里直接逐 key 汇总（避免一一条目过大）
        keys = sorted({k for (k, _) in UR})
        for k in keys:
            w = _key_weight(self.cfg, k)
            vals_r = [v for (kk, v) in UR if kk == k]
            vals_g = G_dict.get(k, [])
            if vals_g:
                S = value_pairwise_similarity(vals_r, vals_g, self.encoder, self.cfg, purpose="passage", max_length=32)
                s_hit = float(S.max(axis=1).mean()) if S.size > 0 else 0.0  # 参考该 key 下多值的平均命中强度
            else:
                s_hit = 0.0
            weighted_hit = w * s_hit
            numer += weighted_hit
            denom += w
            by_key_details.append(AICCoverageByKey(key=k, weighted_hit=weighted_hit, weighted_total=w, score=_safe_div(weighted_hit, w)))
        return numer, max(denom, _EPS), by_key_details

    def _misbind_for_pair(
        self,
        r_idx: int,
        g: EntityRepr,
        ref_bank: Dict[str, List[Tuple[int, str]]],
        R: List[EntityRepr],
    ) -> Tuple[float, float, List[AICMisbindItem]]:
        UG = _extract_kv_triplets(g)
        if not UG:
            return 0.0, _EPS, []

        # 当前 r 的值库（按 key）
        r_bank: Dict[str, List[str]] = {}
        for k, v in _extract_kv_triplets(R[r_idx]):
            r_bank.setdefault(k, []).append(v)

        items: List[AICMisbindItem] = []
        numer = 0.0
        denom = 0.0

        # 分 key 处理，减少无关比对
        by_key: Dict[str, List[str]] = {}
        for k, vprime in UG:
            by_key.setdefault(k, []).append(vprime)

        for k, vprimes in by_key.items():
            ref_all = ref_bank.get(k, [])
            if not ref_all:
                # 参考侧此 key 不存在 -> 不算错绑，分母也不增加
                continue

            vals_all = [val for (_idx, val) in ref_all]
            vals_r = r_bank.get(k, [])

            # s*（对任一参考实体同 key 最佳） & s^(r)（对当前 r 同 key 最佳）
            Sstar = value_pairwise_similarity(vprimes, vals_all, self.encoder, self.cfg, purpose="passage", max_length=32)
            s_star_best = Sstar.max(axis=1) if Sstar.size > 0 else np.zeros((len(vprimes),), dtype=np.float32)

            if vals_r:
                Sr = value_pairwise_similarity(vprimes, vals_r, self.encoder, self.cfg, purpose="passage", max_length=32)
                s_r_best = Sr.max(axis=1)
            else:
                s_r_best = np.zeros((len(vprimes),), dtype=np.float32)

            w = _key_weight(self.cfg, k)
            deltas = np.maximum(0.0, s_star_best - s_r_best)
            numer += float((w * deltas).sum())
            denom += float((w * s_star_best).sum())

            # 记录条目
            for idx, vpr in enumerate(vprimes):
                items.append(AICMisbindItem(
                    g_idx=-1,  # g_idx 在 scoring 中可按需补记；此处聚焦键值明细
                    key=k,
                    value=vpr,
                    s_star=float(s_star_best[idx]),
                    s_ref=float(s_r_best[idx]),
                    delta=float(deltas[idx]),
                    weight=float(w),
                    best_r_idx=None,  # 可在需要时通过 Sstar.argmax 定位
                ))
        return numer, max(denom, _EPS), items

    def compute(
        self,
        R: List[EntityRepr],
        G: List[EntityRepr],
        *,
        sim_matrix: Optional[np.ndarray] = None,
        matching: Optional[MatchResult] = None,
    ) -> Tuple[float, List[AICPairDetail]]:
        if sim_matrix is None:
            sim_matrix = pairwise_similarity(R, G, self.cfg)
        if matching is None:
            matching = compute_matching(sim_matrix, self.cfg)

        if not R:
            return 0.0, []

        # 参考库
        ref_bank = _build_ref_bank_by_key(R)

        cov_sum = 0.0
        cov_den = 0.0
        mis_sum = 0.0
        mis_den = 0.0

        results: List[AICPairDetail] = []

        for p in matching.pairs:
            r_idx, g_idx = p.r_idx, p.g_idx
            numer_cov, denom_cov, cov_by_key = self._coverage_for_pair(R[r_idx], G[g_idx])
            numer_mis, denom_mis, mis_items = self._misbind_for_pair(r_idx, G[g_idx], ref_bank, R)

            cov_sum += numer_cov
            cov_den += denom_cov
            mis_sum += numer_mis
            mis_den += denom_mis

            cov_val = _safe_div(numer_cov, denom_cov)
            mis_val = min(max(_safe_div(numer_mis, denom_mis), 0.0), 1.0)

            results.append(AICPairDetail(
                r_idx=r_idx,
                g_idx=g_idx,
                coverage=cov_val,
                misbind=mis_val,
                coverage_by_key=cov_by_key,
                misbind_items=mis_items,
            ))

        # 样本级：按参考侧平均（|R|）
        Rn = max(1, len(R))
        coverage = _safe_div(cov_sum, cov_den) if cov_den > 0 else 0.0
        misbind = min(max(_safe_div(mis_sum, mis_den) if mis_den > 0 else 0.0, 0.0), 1.0)
        score = float(100.0 * coverage * (1.0 - misbind))
        return score, results


# ========== 3) SoftParsimony：软简约性 ==========
class SoftParsimonyCalculator:
    """
    - 额外类别：未匹配 g，E_cat = 1 - max_r w(r,g)
    - 额外属性：配对 (r,g) 内每个 (k,v')，E_attr = 1 - s*，其中 s* 为对参考任一实体/同 key 的最佳相似
    - HallRate = (ΣE_cat + Σ w_k·E_attr) / (|G| + Σ w_k)
    - Score = 100 · (1 - HallRate)
    """

    def __init__(self, cfg: Config, encoder):
        self.cfg = cfg
        self.encoder = encoder

    def _extra_cat_penalty(
        self,
        sim_matrix: np.ndarray,
        pairs: List[MatchPair],
        R_size: int,
        G_size: int,
    ) -> Tuple[float, List[ParExtraCategory]]:
        matched_g = {p.g_idx for p in pairs}
        extras: List[ParExtraCategory] = []
        numer = 0.0

        for j in range(G_size):
            if j in matched_g:
                continue
            s_best = float(sim_matrix[:, j].max()) if R_size > 0 else 0.0
            pen = 1.0 - s_best
            numer += pen
            # 最相似参考实体索引
            best_r = int(sim_matrix[:, j].argmax()) if R_size > 0 else None
            extras.append(ParExtraCategory(g_idx=j, best_r_idx=best_r if R_size > 0 else None, w_max=s_best, penalty=pen))
        return numer, extras

    def _extra_attr_penalty(
        self,
        R: List[EntityRepr],
        G: List[EntityRepr],
        pairs: List[MatchPair],
        ref_bank: Dict[str, List[Tuple[int, str]]],
    ) -> Tuple[float, float, List[ParExtraAttr]]:
        numer = 0.0
        denom = 0.0
        items: List[ParExtraAttr] = []

        for p in pairs:
            r_idx, g_idx = p.r_idx, p.g_idx
            UG = _extract_kv_triplets(G[g_idx])
            if not UG:
                continue

            by_key: Dict[str, List[str]] = {}
            for k, vprime in UG:
                by_key.setdefault(k, []).append(vprime)

            for k, vprimes in by_key.items():
                w = _key_weight(self.cfg, k)
                denom += w * len(vprimes)

                ref_all = ref_bank.get(k, [])
                if not ref_all:
                    # 参考没有此 key：s* = 0 → 惩罚 1
                    for vpr in vprimes:
                        items.append(ParExtraAttr(r_idx=r_idx, g_idx=g_idx, key=k, value=vpr, s_star=0.0, penalty=1.0, weight=w, best_r_idx=None))
                    numer += w * len(vprimes)
                    continue

                vals_all = [val for (_ii, val) in ref_all]
                Sstar = value_pairwise_similarity(vprimes, vals_all, self.encoder, self.cfg, purpose="passage", max_length=32)
                s_star_best = Sstar.max(axis=1) if Sstar.size > 0 else np.zeros((len(vprimes),), dtype=np.float32)

                numer += float((w * (1.0 - s_star_best)).sum())
                # 明细
                for idx, vpr in enumerate(vprimes):
                    items.append(ParExtraAttr(
                        r_idx=r_idx,
                        g_idx=g_idx,
                        key=k,
                        value=vpr,
                        s_star=float(s_star_best[idx]),
                        penalty=float(1.0 - s_star_best[idx]),
                        weight=float(w),
                        best_r_idx=None,
                    ))
        return numer, max(denom, _EPS), items

    def compute(
        self,
        R: List[EntityRepr],
        G: List[EntityRepr],
        *,
        sim_matrix: Optional[np.ndarray] = None,
        matching: Optional[MatchResult] = None,
    ) -> Tuple[float, List[ParExtraCategory], List[ParExtraAttr]]:
        if sim_matrix is None:
            sim_matrix = pairwise_similarity(R, G, self.cfg)
        if matching is None:
            matching = compute_matching(sim_matrix, self.cfg)

        ref_bank = _build_ref_bank_by_key(R)

        # 额外类别（未匹配 g）
        extra_cat_numer, extra_cat_items = self._extra_cat_penalty(
            sim_matrix, matching.pairs, len(R), len(G)
        )

        # 额外属性（仅匹配对）
        extra_attr_numer, extra_attr_denom, extra_attr_items = self._extra_attr_penalty(
            R, G, matching.pairs, ref_bank
        )

        hall_numer = extra_cat_numer + extra_attr_numer
        hall_denom = float(len(G)) + extra_attr_denom
        hall_rate = min(max(_safe_div(hall_numer, hall_denom) if hall_denom > 0 else 0.0, 0.0), 1.0)
        score = float(100.0 * (1.0 - hall_rate))
        return score, extra_cat_items, extra_attr_items
