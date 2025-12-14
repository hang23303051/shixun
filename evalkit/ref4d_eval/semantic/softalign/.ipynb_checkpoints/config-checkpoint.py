# softalign/config.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import copy
import yaml

# =========================
# 默认结构权重（非词表）
# =========================
DEFAULT_FACETS: Tuple[str, ...] = (
    "color", "pattern", "texture", "material", "size", "age", "sex", "state", "pose", "action",
    "orientation", "facing-direction", "position", "object-part", "tool-or-instrument", "equipment",
    "species-or-breed", "vehicle-type", "food-type", "brand-or-logo", "printed-text", "number-or-id",
    "art-medium", "style", "weather", "lighting", "scene", "camera-view"
)

DEFAULT_KEY_WEIGHT: Dict[str, float] = {
    **{k: 1.0 for k in DEFAULT_FACETS},
    "signature": 2.0,
    "number-or-id": 2.0,
    "brand-or-logo": 2.0,
    "printed-text": 2.0,
    "pattern": 1.5,
    "pose": 1.5,
    "state": 1.5,
}

# =========================
# Dataclasses（与实现对齐）
# =========================
@dataclass
class EncoderConfig:
    model_name_or_path: str = "intfloat/e5-large-v2"
    device: str = "cuda"                 # ["cuda","cpu"]
    dtype: str = "bf16"                  # ["bf16","fp16","fp32"]
    batch_size: int = 128
    max_length_query: int = 64
    max_length_passage: int = 128
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    normalize: bool = True
    cache_dir: Optional[str] = None
    local_files_only: bool = False
    revision: Optional[str] = None
    trust_remote_code: bool = False


@dataclass
class PreprocConfig:
    # 与 preprocessing.py 使用字段一致
    lowercase: bool = True
    strip: bool = True
    canonical_hyphen: bool = True
    drop_nonalpha: bool = True
    drop_short_token_len: int = 1  # 过滤极短无字母 token 的阈值


@dataclass
class ReprConfig:
    # 与 repr_builder.py 使用字段一致
    key_weight: Dict[str, float] = field(default_factory=lambda: copy.deepcopy(DEFAULT_KEY_WEIGHT))
    include_name_in_set: bool = True
    name_weight_in_set: float = 1.0
    name_channel_purpose: str = "query"
    set_channel_purpose: str = "passage"


@dataclass
class SimConfig:
    # 与 similarity.py 使用字段一致
    tau0: float = 0.30
    combine: str = "max"      # ["max","weighted"]
    w_name: float = 0.5       # combine="weighted" 时使用
    w_set: float = 0.5


@dataclass
class MatchingConfig:
    # 与 matching.py 使用字段一致
    algorithm: str = "hungarian"
    min_score: float = 0.30   # 门控阈值（映射后 [0,1]）


@dataclass
class FusionConfig:
    # 与 scoring.py 使用字段一致
    alpha: float = 0.30  # CatCov
    beta: float = 0.50   # AIC
    gamma: float = 0.20  # Parsimony


@dataclass
class Config:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    preproc: PreprocConfig = field(default_factory=PreprocConfig)
    repr: ReprConfig = field(default_factory=ReprConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# =========================
# 工具：合并/别名映射/校验
# =========================
def _merge_into_dataclass(dc_obj, overrides: Dict[str, Any]) -> None:
    if not is_dataclass(dc_obj) or not isinstance(overrides, dict):
        return
    for k, v in overrides.items():
        if not hasattr(dc_obj, k):
            continue
        cur = getattr(dc_obj, k)
        if is_dataclass(cur) and isinstance(v, dict):
            _merge_into_dataclass(cur, v)
        else:
            setattr(dc_obj, k, v)

def _clip01(x: float) -> float:
    try:
        return float(max(0.0, min(1.0, x)))
    except Exception:
        return 0.0

def _alias_yaml_to_internal(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    兼容旧 YAML 命名：
      - preprocess -> preproc
          - unify_hyphen_underscore -> canonical_hyphen
          - drop_if_no_alpha        -> drop_nonalpha
          - max_value_len           -> 忽略（由抽取层负责），不与本实现冲突
      - similarity -> sim
          - combine='mean' 映射为 combine='weighted' 且 w_name=w_set=0.5
          - gate_min -> matching.min_score
      - matching.min_gate -> matching.min_score
    """
    d = copy.deepcopy(d) if isinstance(d, dict) else {}

    # preprocess -> preproc
    if "preprocess" in d and "preproc" not in d:
        src = d.get("preprocess") or {}
        pre = {
            "lowercase": bool(src.get("lowercase", True)),
            "strip": bool(src.get("strip", True)),
            "canonical_hyphen": bool(src.get("unify_hyphen_underscore", True)),
            "drop_nonalpha": bool(src.get("drop_if_no_alpha", True)),
        }
        # 可选：若用户显式给 drop_short_token_len，则沿用
        if "drop_short_token_len" in src:
            try:
                pre["drop_short_token_len"] = int(src["drop_short_token_len"])
            except Exception:
                pass
        d["preproc"] = pre

    # similarity -> sim
    if "similarity" in d and "sim" not in d:
        src = d.get("similarity") or {}
        sim = {
            "tau0": float(src.get("tau0", 0.30)),
            "combine": str(src.get("combine", "max")).lower(),
        }
        if sim["combine"] == "mean":
            sim["combine"] = "weighted"
            sim["w_name"] = 0.5
            sim["w_set"] = 0.5
        # gate_min -> matching.min_score
        if "gate_min" in src:
            d.setdefault("matching", {})
            try:
                d["matching"]["min_score"] = float(src["gate_min"])
            except Exception:
                pass
        d["sim"] = sim

    # matching.min_gate -> matching.min_score
    if "matching" in d:
        mt = d["matching"]
        if isinstance(mt, dict) and ("min_gate" in mt) and ("min_score" not in mt):
            try:
                mt["min_score"] = float(mt["min_gate"])
            except Exception:
                pass

        # 算法命名兼容：algo -> algorithm
        if "algo" in mt and "algorithm" not in mt:
            mt["algorithm"] = mt["algo"]

    return d

def _validate_and_fix(cfg: Config) -> Config:
    # encoder
    if cfg.encoder.device not in ("cuda", "cpu"):
        cfg.encoder.device = "cuda"
    if cfg.encoder.dtype not in ("bf16", "fp16", "fp32"):
        cfg.encoder.dtype = "bf16"
    cfg.encoder.batch_size = max(1, int(cfg.encoder.batch_size))
    cfg.encoder.max_length_query = max(8, int(cfg.encoder.max_length_query))
    cfg.encoder.max_length_passage = max(8, int(cfg.encoder.max_length_passage))

    # preproc
    cfg.preproc.drop_short_token_len = max(0, int(cfg.preproc.drop_short_token_len))

    # repr
    # 规范化 key_weight：全转小写字符串键
    fixed_kw: Dict[str, float] = {}
    for k, v in (cfg.repr.key_weight or {}).items():
        try:
            fixed_kw[str(k).strip().lower()] = float(v)
        except Exception:
            continue
    # 回填默认 facets
    for k, v in DEFAULT_KEY_WEIGHT.items():
        fixed_kw.setdefault(k, float(v))
    cfg.repr.key_weight = fixed_kw

    # sim
    cfg.sim.tau0 = min(0.99, max(0.0, float(cfg.sim.tau0)))
    if cfg.sim.combine not in ("max", "weighted"):
        cfg.sim.combine = "max"
    try:
        cfg.sim.w_name = float(cfg.sim.w_name)
    except Exception:
        cfg.sim.w_name = 0.5
    try:
        cfg.sim.w_set = float(cfg.sim.w_set)
    except Exception:
        cfg.sim.w_set = 0.5
    total_w = max(1e-12, cfg.sim.w_name + cfg.sim.w_set)
    cfg.sim.w_name = float(cfg.sim.w_name / total_w)
    cfg.sim.w_set = float(cfg.sim.w_set / total_w)

    # matching
    cfg.matching.algorithm = str(cfg.matching.algorithm or "hungarian").lower()
    if cfg.matching.algorithm not in ("hungarian", "greedy"):
        cfg.matching.algorithm = "hungarian"
    cfg.matching.min_score = _clip01(float(cfg.matching.min_score))

    # fusion
    for k in ("alpha", "beta", "gamma"):
        try:
            setattr(cfg.fusion, k, float(getattr(cfg.fusion, k)))
        except Exception:
            pass

    return cfg

# =========================
# YAML 装载
# =========================
def load_config(yaml_path: Union[str, Path]) -> Config:
    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"Config YAML not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # 别名映射（兼容旧键）
    raw = _alias_yaml_to_internal(raw)

    # 构造默认并递归覆盖
    cfg = Config()
    _merge_into_dataclass(cfg, raw)

    # 校验与回填
    cfg = _validate_and_fix(cfg)
    return cfg
