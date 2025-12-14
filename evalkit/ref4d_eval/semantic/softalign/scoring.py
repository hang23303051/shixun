# softalign/scoring.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import Config
from .types import (
    Entity,
    EntityRepr,
    AxisScores,
    AxisDetails,
    ScoreReport,
    MatchResult,
)
from .preprocessing import coerce_entities_from_raw
from .repr_builder import encode_entity_reprs
from .similarity import pairwise_similarity
from .matching import compute_matching
from .axes import (
    SoftCatCovCalculator,
    SoftAICCalculator,
    SoftParsimonyCalculator,
)

# ======== JSON 基础校验 ========

def _validate_non_empty(doc: Optional[Dict]) -> Tuple[bool, str]:
    """
    返回 (ok, reason)：
      - fine.entities 存在且非空 -> ok
      - 或 basic_semantics.attributes 为非空 dict -> ok
    """
    if not isinstance(doc, dict) or not doc:
        return False, "doc is empty or not a JSON object"
    fine = doc.get("fine")
    if isinstance(fine, dict):
        e_list = fine.get("entities", [])
        if isinstance(e_list, list) and len(e_list) > 0:
            return True, ""
    bs = doc.get("basic_semantics", {})
    if isinstance(bs, dict) and isinstance(bs.get("attributes", {}), dict) and len(bs.get("attributes", {})) > 0:
        return True, ""
    return False, "no entities found in fine.entities and no per-category attributes in basic_semantics"


# ======== 融合权重 ========

@dataclass
class FusionWeights:
    alpha: float = 0.30  # CatCov
    beta: float = 0.50   # AIC
    gamma: float = 0.20  # Parsimony

def _fusion_from_cfg(cfg: Config) -> FusionWeights:
    fw = getattr(cfg, "fusion", None)
    if fw is None:
        return FusionWeights()
    try:
        return FusionWeights(
            alpha=float(getattr(fw, "alpha", 0.30)),
            beta=float(getattr(fw, "beta", 0.50)),
            gamma=float(getattr(fw, "gamma", 0.20)),
        )
    except Exception:
        return FusionWeights()


# ======== 工具：未匹配集合 ========

def _calc_unmatched(Rn: int, Gn: int, m: MatchResult) -> Tuple[List[int], List[int]]:
    matched_r = {p.r_idx for p in m.pairs}
    matched_g = {p.g_idx for p in m.pairs}
    ref_un = [i for i in range(Rn) if i not in matched_r]
    gen_un = [j for j in range(Gn) if j not in matched_g]
    return ref_un, gen_un


# ======== 调度器 ========

class SampleScorer:
    """
    样本级调度：
      - 读取/校验 JSON
      - 解析实体（无词表轻规整）
      - 构建实体表示 -> 实体级相似度矩阵 -> 一对一匹配
      - 三轴计算：SoftCatCov / SoftAIC / SoftParsimony
      - 线性融合 & 结构化报告（0~100）
    """

    def __init__(self, cfg: Config, encoder):
        self.cfg = cfg
        self.encoder = encoder
        self._catcov = SoftCatCovCalculator(cfg)
        self._aic = SoftAICCalculator(cfg, encoder)
        self._par = SoftParsimonyCalculator(cfg, encoder)
        self._fusion = _fusion_from_cfg(cfg)

    @staticmethod
    def _load_json(path: str | Path) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def score_pair(
        self,
        ref_doc_or_path: Dict | str | Path,
        gen_doc_or_path: Dict | str | Path,
        sample_id: Optional[str] = None,
    ) -> ScoreReport:
        # 读取 JSON
        ref_doc = ref_doc_or_path if isinstance(ref_doc_or_path, dict) else self._load_json(ref_doc_or_path)
        gen_doc = gen_doc_or_path if isinstance(gen_doc_or_path, dict) else self._load_json(gen_doc_or_path)

        # 非空校验
        ok_r, why_r = _validate_non_empty(ref_doc)
        ok_g, why_g = _validate_non_empty(gen_doc)

        if not ok_r or not ok_g:
            # 不计分：返回 NaN + skip_reason
            info = {
                "skipped": True,
                "skip_reason": f"ref invalid: {why_r}" if not ok_r else (f"gen invalid: {why_g}" if not ok_g else ""),
            }
            nan = float("nan")
            return ScoreReport(
                sample_id=sample_id or "",
                axis=AxisScores(catcov=nan, aic=nan, par=nan),
                s_base=nan,
                details=AxisDetails(),
                sizes={"R": 0, "G": 0, "matched": 0},
                info=info,
            )

        # 解析实体（使用无词表的轻规整）
        preproc_cfg = getattr(self.cfg, "preproc", None)
        R_entities: List[Entity] = coerce_entities_from_raw(ref_doc, preproc_cfg)
        G_entities: List[Entity] = coerce_entities_from_raw(gen_doc, preproc_cfg)

        # 构建实体表示
        R: List[EntityRepr] = encode_entity_reprs(R_entities, self.encoder, self.cfg)
        G: List[EntityRepr] = encode_entity_reprs(G_entities, self.encoder, self.cfg)

        if len(R) == 0 or len(G) == 0:
            # 不计分：返回 NaN + skip_reason
            info = {
                "skipped": True,
                "skip_reason": "no valid entities after preprocessing",
            }
            nan = float("nan")
            return ScoreReport(
                sample_id=sample_id or "",
                axis=AxisScores(catcov=nan, aic=nan, par=nan),
                s_base=nan,
                details=AxisDetails(),
                sizes={"R": len(R), "G": len(G), "matched": 0},
                info=info,
            )

        # 公用相似度与匹配
        sim = pairwise_similarity(R, G, self.cfg)  # 已映射到 [0,1]
        match: MatchResult = compute_matching(sim, self.cfg)
        ref_un, gen_un = _calc_unmatched(len(R), len(G), match)

        # 三条轴
        s_catcov, catcov_details = self._catcov.compute(R, G, sim_matrix=sim, matching=match)
        s_aic, aic_pair_details = self._aic.compute(R, G, sim_matrix=sim, matching=match)
        s_par, par_extra_cats, par_extra_attrs = self._par.compute(R, G, sim_matrix=sim, matching=match)

        # 融合
        total = float(self._fusion.alpha * s_catcov + self._fusion.beta * s_aic + self._fusion.gamma * s_par)

        # 汇总到 AxisDetails
        details = AxisDetails(
            catcov_pairs=catcov_details,
            aic_pairs=aic_pair_details,
            par_extra_categories=par_extra_cats,
            par_extra_attrs=par_extra_attrs,
            misc={
                "matching": {
                    "algo": getattr(match, "method", "hungarian"),
                    "num_pairs": len(match.pairs),
                    "ref_unmatched": ref_un,
                    "gen_unmatched": gen_un,
                    "min_gate": float(getattr(getattr(self.cfg, "matching", None), "min_score", 0.30)),
                },
                "fusion": {"alpha": self._fusion.alpha, "beta": self._fusion.beta, "gamma": self._fusion.gamma},
            },
        )

        return ScoreReport(
            sample_id=sample_id or "",
            axis=AxisScores(catcov=s_catcov, aic=s_aic, par=s_par),
            s_base=total,
            details=details,
            sizes={"R": len(R), "G": len(G), "matched": len(match.pairs)},
            info={
                "sim": {
                    "tau0": float(getattr(getattr(self.cfg, "sim", None), "tau0", 0.30)),
                    "combine": str(getattr(getattr(self.cfg, "sim", None), "combine", "max")),
                }
            },
        )

    # 便捷文件接口
    def score_pair_from_files(
        self,
        ref_json_path: str | Path,
        gen_json_path: str | Path,
        sample_id: Optional[str] = None,
    ) -> ScoreReport:
        return self.score_pair(ref_json_path, gen_json_path, sample_id=sample_id)
