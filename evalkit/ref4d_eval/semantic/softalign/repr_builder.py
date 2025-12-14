# softalign/repr_builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .config import Config
from .types import Entity, EntityRepr, Embedding
from .encoder import TextEncoder


__all__ = [
    "build_fragments_for_entity",
    "encode_entity_repr",
    "encode_entity_reprs",
]


def _get_key_weight(key: str, cfg: Config) -> float:
    """
    结构权重（不属于“词表”）：来自配置；未命中则回落 default=1.0
    建议在 config.yaml 中设置，如：
      repr:
        key_weight:
          signature: 2.0
          number-or-id: 2.0
          brand-or-logo: 2.0
          printed-text: 2.0
          pattern: 1.5
          pose: 1.5
          state: 1.5
    """
    kw = getattr(cfg, "repr", None)
    if kw and isinstance(getattr(kw, "key_weight", None), dict):
        return float(kw.key_weight.get(key, 1.0))
    return 1.0


def _include_name_in_set(cfg: Config) -> bool:
    r = getattr(cfg, "repr", None)
    return bool(getattr(r, "include_name_in_set", True))


def _name_in_set_weight(cfg: Config) -> float:
    r = getattr(cfg, "repr", None)
    return float(getattr(r, "name_weight_in_set", 1.0))


def _name_purpose(cfg: Config) -> str:
    r = getattr(cfg, "repr", None)
    return str(getattr(r, "name_channel_purpose", "query"))  # 常用：query


def _set_purpose(cfg: Config) -> str:
    r = getattr(cfg, "repr", None)
    return str(getattr(r, "set_channel_purpose", "passage"))  # 常用：passage


@dataclass
class _Fragments:
    name_text: str
    frag_texts: List[str]
    frag_weights: List[float]


def build_fragments_for_entity(ent: Entity, cfg: Config) -> _Fragments:
    """
    构造实体的“顺序无关片段集合”：
      - name 片段单独存放（作为 name 通道）
      - 集合通道由 {key: value} 片段组成（一个 value 一条）
      - 可选将 name 合入集合通道（以小权重），提升鲁棒性
    """
    name_text = ent.name or ""

    frag_texts: List[str] = []
    frag_weights: List[float] = []

    # 属性片段：每个 value 生成 "key: value"
    for k, values in (ent.attrs or {}).items():
        if not values:
            continue
        w_k = _get_key_weight(k, cfg)
        for v in values:
            # 每个 value 一个片段；权重为结构权（不做 value 个数均摊，后续会整体归一）
            frag_texts.append(f"{k}: {v}")
            frag_weights.append(float(w_k))

    # （可选）把 name 也并入集合通道，增加对缺失属性的鲁棒性
    if _include_name_in_set(cfg) and name_text:
        frag_texts.append(name_text)
        frag_weights.append(_name_in_set_weight(cfg))

    # 若集合片段为空，至少塞入 name 兜底（避免空向量）
    if not frag_texts and name_text:
        frag_texts.append(name_text)
        frag_weights.append(1.0)

    return _Fragments(name_text=name_text, frag_texts=frag_texts, frag_weights=frag_weights)


def _weighted_average(vectors: List[np.ndarray], weights: List[float]) -> np.ndarray:
    """
    对一批向量做加权均值；入参均为 np.float32，向量已 L2 归一。
    最终再做一次 L2 归一，保持为单位向量。
    """
    if not vectors:
        return np.zeros((0,), dtype=np.float32)
    W = np.asarray(weights, dtype=np.float32)
    W = np.clip(W, 0.0, np.finfo(np.float32).max)
    if float(W.sum()) <= 0.0:
        W = np.ones_like(W)
    W = W / (W.sum() + 1e-12)
    M = np.stack(vectors, axis=0)  # [n, d]
    pooled = (M * W[:, None]).sum(axis=0)  # [d]
    # L2 normalize
    norm = np.linalg.norm(pooled) + 1e-12
    pooled = (pooled / norm).astype(np.float32, copy=False)
    return pooled


def encode_entity_repr(
    ent: Entity,
    encoder: TextEncoder,
    cfg: Config,
    *,
    max_length: int = 512,
) -> EntityRepr:
    """
    将单个 Entity 转为两通道表示：
      - name_vec：仅 name 片段编码（默认 purpose='query'）
      - set_vec ：集合通道（属性片段 + 可选 name），逐片段编码后按权重加权平均
    """
    frags = build_fragments_for_entity(ent, cfg)

    # name 通道
    name_embs = encoder.embed_texts([frags.name_text], purpose=_name_purpose(cfg), max_length=max_length)
    name_vec = name_embs[0].vec if isinstance(name_embs[0], Embedding) else np.asarray(name_embs[0], dtype=np.float32)

    # 集合通道（分片编码）
    piece_embs = encoder.embed_texts(frags.frag_texts, purpose=_set_purpose(cfg), max_length=max_length)
    piece_vecs = [e.vec if isinstance(e, Embedding) else np.asarray(e, dtype=np.float32) for e in piece_embs]
    set_vec = _weighted_average(piece_vecs, frags.frag_weights)

    return EntityRepr(
        entity=ent,
        name_text=frags.name_text,
        frag_texts=frags.frag_texts,
        frag_weights=frags.frag_weights,
        name_vec=name_vec,
        set_vec=set_vec,
    )


def encode_entity_reprs(
    ents: List[Entity],
    encoder: TextEncoder,
    cfg: Config,
    *,
    max_length: int = 512,
) -> List[EntityRepr]:
    """
    批量版本（内部仍按片段池化；分两次批量调用编码器以降低显存压力）
    """
    out: List[EntityRepr] = []

    # 先构造所有片段，集中编码 name 与集合片段
    name_texts: List[str] = []
    set_texts_all: List[str] = []
    set_slices: List[Tuple[int, int]] = []  # 每个实体在 set_texts_all 的切片 [s, e)
    set_weights_all: List[float] = []
    fragments_cache: List[_Fragments] = []

    for ent in ents:
        fr = build_fragments_for_entity(ent, cfg)
        fragments_cache.append(fr)
        name_texts.append(fr.name_text)

        s = len(set_texts_all)
        set_texts_all.extend(fr.frag_texts)
        set_weights_all.extend(fr.frag_weights)
        e = len(set_texts_all)
        set_slices.append((s, e))

    # 批量编码
    name_embs = encoder.embed_texts(name_texts, purpose=_name_purpose(cfg), max_length=max_length)
    name_vecs = [e.vec if isinstance(e, Embedding) else np.asarray(e, dtype=np.float32) for e in name_embs]

    set_piece_embs = encoder.embed_texts(set_texts_all, purpose=_set_purpose(cfg), max_length=max_length)
    set_piece_vecs = [e.vec if isinstance(e, Embedding) else np.asarray(e, dtype=np.float32) for e in set_piece_embs]

    # 组装
    for ent, fr, name_v, (s, e) in zip(ents, fragments_cache, name_vecs, set_slices):
        set_vec = _weighted_average(set_piece_vecs[s:e], set_weights_all[s:e])
        out.append(
            EntityRepr(
                entity=ent,
                name_text=fr.name_text,
                frag_texts=fr.frag_texts,
                frag_weights=fr.frag_weights,
                name_vec=name_v,
                set_vec=set_vec,
            )
        )

    return out
