# softalign/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

__all__ = [
    # 基础输入/输出结构
    "Entity", "VideoDoc",
    # 表示/嵌入
    "EntityRepr", "Embedding",
    # 相似度矩阵
    "SimMatrix",
    # 匹配结果
    "MatchPair", "MatchResult",
    # CatCov 细节
    "CatCovPairDetail",
    # AIC 细节
    "AICCoverageByKey", "AICMisbindItem", "AICPairDetail",
    # Parsimony 细节
    "ParExtraCategory", "ParExtraAttr",
    # 汇总得分与报告
    "AxisScores", "AxisDetails", "ScoreReport",
]


# =========================
# 一) 基础输入结构（与抽取 JSON 对齐）
# =========================

@dataclass
class Entity:
    """
    单个实体的基础信息。
    - id: 源 JSON 的实体 id（可为空字符串）
    - name: 实体类名（预处理后的文本）
    - attrs: 属性键 -> 值列表（值为原始可见字符串的轻规整版本，保持开放词表）
    """
    id: str
    name: str
    attrs: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class VideoDoc:
    """
    单侧（参考或生成）文档容器。
    - entities: 实体列表（仅语义层：name/attrs）
    - meta: 任意元信息（如 video 路径、模型名、时间窗参数等）
    """
    entities: List[Entity] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


# =========================
# 二) 表示/嵌入（与表示构造 & 编码器接口对齐）
# =========================

@dataclass
class EntityRepr:
    """
    实体两通道表示 + 诊断信息（顺序无关集合池化）。
    - entity: 源实体
    - name_text: name 通道的原文本（规整后）
    - frag_texts / frag_weights: 集合通道的片段与其结构权重
    - name_vec: 仅 name 片段编码后的单位向量，shape [dim]
    - set_vec : 集合片段加权池化后的单位向量，shape [dim]
    """
    entity: Entity
    name_text: str
    frag_texts: List[str] = field(default_factory=list)
    frag_weights: List[float] = field(default_factory=list)
    name_vec: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    set_vec: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))


@dataclass
class Embedding:
    """
    文本编码后的向量（要求 L2 归一化）。
    - vec: np.ndarray, shape [dim]
    """
    vec: np.ndarray


# =========================
# 三) 相似度矩阵（name 通道 / 集合通道 / 融合）
# =========================

@dataclass
class SimMatrix:
    """
    实体级相似度矩阵（已映射到 [0,1]）。
    - name:   仅 name 通道，shape [|R|, |G|]
    - set:    集合池化通道，shape [|R|, |G|]
    - fused:  通道融合后的矩阵（如 pairwise max），shape [|R|, |G|]
    """
    name: np.ndarray
    set: np.ndarray
    fused: np.ndarray


# =========================
# 四) 匹配结果（默认一对一最大权匹配）
# =========================

@dataclass
class MatchPair:
    """
    单个配对 (r_idx, g_idx)。
    - r_idx: 参考实体在 R 中的下标
    - g_idx: 生成实体在 G 中的下标
    - score: 融合相似度矩阵 fused[r_idx, g_idx] 的取值（[0,1]）
    """
    r_idx: int
    g_idx: int
    score: float


@dataclass
class MatchResult:
    """
    匹配输出。
    - pairs: 选中的一对一配对列表（已应用最小门槛/剪枝）
    - method: "hungarian" / "greedy" / "ot"
    """
    pairs: List[MatchPair] = field(default_factory=list)
    method: str = "hungarian"


# =========================
# 五) 三条轴的可解释明细结构
# =========================

# 1) SoftCatCov
# -------------------------

@dataclass
class CatCovPairDetail:
    """
    CatCov 轴的逐实体覆盖明细。
    - r_idx: 参考实体下标
    - g_idx: 匹配到的生成实体下标（未匹配用 -1）
    - sim:   覆盖得分贡献（匹配权重 w(r,g)；未匹配为 0）
    - note:  可选说明（如门槛过滤等）
    """
    r_idx: int
    g_idx: int
    sim: float
    note: str = ""


# 2) SoftAIC
# -------------------------

@dataclass
class AICCoverageByKey:
    """
    属性覆盖在单个 (r,g) 配对下的按 key 聚合明细。
    - key: 属性键（字符串相等判定）
    - weighted_hit: 加权命中累计（∑ w_k * s_{kv}）
    - weighted_total: 加权分母（∑ w_k）
    - score: weighted_hit / max(weighted_total, eps)，位于 [0,1]
    """
    key: str
    weighted_hit: float
    weighted_total: float
    score: float


@dataclass
class AICMisbindItem:
    """
    单个生成属性值 (k, v') 的错绑明细。
    - g_idx: 生成实体下标
    - key: 属性键
    - value: 生成侧属性值 v'
    - s_star: v' 对参考任意实体/同 key 的最佳相似（应当去的地方）
    - s_ref:  v' 对“当前配对参考实体 r/同 key”的最佳相似（实际留在当前的强度）
    - delta:  错绑增量 max(0, s_star - s_ref)（[0,1]）
    - weight: 该 key 的权重 w_k
    - best_r_idx: s_star 对应的最佳参考实体下标（若存在）
    """
    g_idx: int
    key: str
    value: str
    s_star: float
    s_ref: float
    delta: float
    weight: float
    best_r_idx: Optional[int] = None


@dataclass
class AICPairDetail:
    """
    AIC 轴的单对 (r,g) 明细。
    - r_idx, g_idx: 当前配对
    - coverage: 该配对的属性覆盖值（[0,1]）
    - misbind:  该配对的错绑率（[0,1]）
    - coverage_by_key: 按 key 的覆盖列表
    - misbind_items:   该配对涉及到的错绑条目
    """
    r_idx: int
    g_idx: int
    coverage: float
    misbind: float
    coverage_by_key: List[AICCoverageByKey] = field(default_factory=list)
    misbind_items: List[AICMisbindItem] = field(default_factory=list)


# 3) SoftParsimony
# -------------------------

@dataclass
class ParExtraCategory:
    """
    额外类别惩罚条目（对未良好匹配的生成实体 g）。
    - g_idx: 生成实体下标
    - best_r_idx: 与 g 最相似的参考实体下标（若存在）
    - w_max: 与最佳参考的相似度（[0,1]）
    - penalty: 额外类别强度（1 - w_max）
    """
    g_idx: int
    best_r_idx: Optional[int]
    w_max: float
    penalty: float


@dataclass
class ParExtraAttr:
    """
    额外属性惩罚条目（针对配对内每个生成属性值）。
    - r_idx: 所属配对的参考实体下标（未配对可置 None）
    - g_idx: 所属配对的生成实体下标
    - key: 属性键
    - value: 生成属性值 v'
    - s_star: 对参考任一实体/同 key 的最佳相似
    - penalty: 额外属性强度（1 - s_star）
    - weight: key 的结构权重 w_k
    - best_r_idx: s_star 对应的参考实体下标（若存在）
    """
    r_idx: Optional[int]
    g_idx: int
    key: str
    value: str
    s_star: float
    penalty: float
    weight: float
    best_r_idx: Optional[int] = None


# =========================
# 六) 汇总得分与最终报告
# =========================

@dataclass
class AxisScores:
    """
    三条轴的顶层得分（0~100）。
    """
    catcov: float = 0.0
    aic: float = 0.0
    par: float = 0.0


@dataclass
class AxisDetails:
    """
    三条轴的可解释细节（面向审阅与误差归因）。
    - CatCov: 逐参考实体的覆盖配对与分数
    - AIC:    逐配对的覆盖/错绑细项
    - Par:    额外类别/属性条目
    """
    # CatCov
    catcov_pairs: List[CatCovPairDetail] = field(default_factory=list)
    # AIC
    aic_pairs: List[AICPairDetail] = field(default_factory=list)
    # Parsimony
    par_extra_categories: List[ParExtraCategory] = field(default_factory=list)
    par_extra_attrs: List[ParExtraAttr] = field(default_factory=list)

    # 附加信息（可选：如阈值、映射参数、统计摘要）
    misc: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreReport:
    """
    单样本的最终报告。
    - sample_id: 样本 id（可用参考 meta.video 或外部传入）
    - axis: 三轴分数
    - s_base: 线性融合后的总分
    - details: 三轴细节
    - sizes: |R| / |G| 及匹配统计
    - info: 其他元信息（编码器/阈值/匹配器等摘要，便于复现）
    """
    sample_id: str
    axis: AxisScores
    s_base: float
    details: AxisDetails = field(default_factory=AxisDetails)

    # 规模与计数
    sizes: Dict[str, int] = field(default_factory=lambda: {"R": 0, "G": 0, "matched": 0})
    # 运行信息（可选）
    info: Dict[str, Any] = field(default_factory=dict)
