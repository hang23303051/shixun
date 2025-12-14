# /root/autodl-tmp/event_eval/src/metrics/ega.py
# -*- coding: utf-8 -*-
"""
EGA*: Event-Graph Alignment (local quality on matched pairs)

Inputs:
  - pairs.json: data/match/<pair_id>/pairs.json
      {
        "M": [
          ["<ref_id>", "<gen_id>", {"sim_sem": float, "r_tIoU": float, "q": float}],
          ...
        ],
        "meta": {..., "w1":..., "w2":...}
      }
  - ref_emb_json: data/embeds/ref/<sample_id>.emb[.merged].json
      (仅用来推断 <sample_id>，实际 s/e 从 events_merged 或 events 读取)

Computation:
  For each matched pair (i,j) with ref i:
    dur_i = e_i - s_i    (normalized on [0,1])
    weight_i = dur_i ** rho
    q_ij = (use cfg w1,w2) * (sim_sem, r_tIoU)
  S_EGA* = 100 * sum_i weight_i * q_ij / sum_i weight_i

Returns:
  { "score": float in [0,100], "valid": bool, "details": {...} }
"""

from __future__ import annotations
import json
import argparse
import logging
from typing import Any, Dict, List, Tuple
import os
from pathlib import Path

import numpy as np

from ..common.io import read_json, read_yaml

LOGGER = logging.getLogger("event_eval.metrics.ega")
if not LOGGER.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    LOGGER.addHandler(h)
    LOGGER.setLevel(logging.INFO)


def _lane_and_key_from_emb_path(emb_path: str) -> Tuple[str, str, str]:
    """
    emb_path 示例
      - ref: <repo>/data/metadata/event_evidence/embeds_merged_ref/<sample>.emb.merged.json
             或   <repo>/data/embeds/ref/<sample>.emb.json
      - gen: <repo>/outputs/event/cache/embeds/gen/<sample__model>.emb.merged.json
             或   <repo>/data/embeds/gen/<sample>.emb.json

    返回:
      root : "<repo>/data/" 或 "<repo>/outputs/event/"
      lane : "ref" / "gen"
      key  : "<sample>" 或 "<sample__model>"
    """
    ap = os.path.abspath(emb_path).replace("\\", "/")

    # 先判断是在 data 下面还是 outputs/event 下面
    idx_data = ap.rfind("/data/")
    idx_out  = ap.rfind("/outputs/event/")
    if idx_data >= 0:
        root = ap[: idx_data + len("/data/")]
    elif idx_out >= 0:
        root = ap[: idx_out + len("/outputs/event/")]
    else:
        # 极端兜底：往上退 3 层
        root = str(Path(ap).parents[3])

    # lane: 路径里显式有 /gen/ 就当 gen，否则 ref
    lane = "gen" if "/gen/" in ap else "ref"

    # key: 去掉 emb 后缀
    fname = os.path.basename(ap)
    key = fname
    for suf in (".emb.merged.json", ".emb.json", ".json"):
        if key.endswith(suf):
            key = key[: -len(suf)]
            break
    return root, lane, key


def _resolve_events_path_from_emb(emb_path: str) -> Tuple[str, str]:
    """
    从 embedding 路径推断对应事件 json 路径。

    Ref4D-VideoBench 新布局:
      ref: <repo>/data/metadata/event_evidence/events_merged_ref/<sample>.newevents.json
      gen: <repo>/outputs/event/cache/events/gen/<sample__model>.events.json

    兼容老的 event_eval 布局:
      ref/gen: <repo>/data/events_merged/{ref,gen}/<key>.newevents.json
               或 <repo>/data/events/{ref,gen}/<key>.events.json
    """
    root, lane, key = _lane_and_key_from_emb_path(emb_path)
    root_norm = root.replace("\\", "/").rstrip("/")

    # ---------- 参考端 ----------
    if lane == "ref":
        # 1) Ref4D-VideoBench: data/metadata/event_evidence/events_merged_ref
        p_meta = os.path.join(
            root, "metadata", "event_evidence",
            "events_merged_ref", f"{key}.newevents.json"
        )
        # 2) 老 event_eval: data/events_merged/ref, data/events/ref
        p_merged = os.path.join(root, "events_merged", "ref", f"{key}.newevents.json")
        p_orig   = os.path.join(root, "events",        "ref", f"{key}.events.json")

        if os.path.exists(p_meta):
            return p_meta, "meta_merged"
        if os.path.exists(p_merged):
            return p_merged, "merged"
        return p_orig, "original"

    # ---------- 生成端 ----------
    # A. 老 event_eval: 在 data 下面
    if "/data/" in root_norm:
        p_merged = os.path.join(root, "events_merged", "gen", f"{key}.newevents.json")
        p_orig   = os.path.join(root, "events",        "gen", f"{key}.events.json")
        if os.path.exists(p_merged):
            return p_merged, "merged"
        return p_orig, "original"

    # B. Ref4D-VideoBench: outputs/event/cache/events/gen
    p_cache     = os.path.join(root, "cache", "events", "gen", f"{key}.events.json")
    p_cache_new = os.path.join(root, "cache", "events", "gen", f"{key}.newevents.json")
    if os.path.exists(p_cache):
        return p_cache, "cache"
    return p_cache_new, "cache"



def _load_ref_span_map(ref_emb_json: str) -> Tuple[Dict[str, Tuple[float, float]], str]:
    """
    加载参考端 id -> (s,e)，支持:
      - events_merged_ref/*.newevents.json (s_abs/e_abs)
      - events_merged/ref 或 events/ref (s/e)
    """
    events_path, source = _resolve_events_path_from_emb(ref_emb_json)
    data = read_json(events_path)
    if isinstance(data, dict) and "events" in data:
        data = data["events"]
    d: Dict[str, Tuple[float, float]] = {}
    for it in data:
        rid = str(it.get("id") or it.get("eid") or it.get("event_id"))
        if not rid:
            continue
        s = float(it.get("s_abs", it.get("s", 0.0)))
        e = float(it.get("e_abs", it.get("e", 0.0)))
        d[rid] = (s, e)
    return d, source


def _weights_from_cfg(cfg: Dict[str, Any], pairs_meta: Dict[str, Any]) -> Tuple[float, float, float]:
    ega = cfg.get("ega") or {}
    # prefer cfg; if not provided, fall back to pairs meta
    w1 = float(ega.get("w1", pairs_meta.get("w1", 0.7)))
    w2 = float(ega.get("w2", pairs_meta.get("w2", 0.3)))
    rho = float(ega.get("rho", 0.5))
    return w1, w2, rho


def compute_ega(pairs_json_path: str, ref_emb_json: str, cfg_path: str) -> Dict[str, Any]:
    pairs = read_json(pairs_json_path)
    M = pairs.get("M", [])
    meta = pairs.get("meta", {})
    if not isinstance(M, list) or len(M) == 0:
        return {"score": 0.0, "valid": False, "details": {"n_pairs": 0}}

    cfg = read_yaml(cfg_path)
    w1, w2, rho = _weights_from_cfg(cfg, meta)

    span_map, src = _load_ref_span_map(ref_emb_json)
    weights: List[float] = []
    scores: List[float] = []

    missing_spans = 0
    for trip in M:
        ref_id, gen_id, d = trip
        s = float(d.get("sim_sem", 0.0))
        u = float(d.get("r_tIoU", 0.0))
        q = w1 * s + w2 * u
        if ref_id not in span_map:
            missing_spans += 1
            continue
        s0, e0 = span_map[ref_id]
        dur = max(0.0, e0 - s0)
        w = (dur ** rho)
        if w <= 0:
            continue
        weights.append(w)
        scores.append(q)

    if not weights:
        return {"score": 0.0, "valid": False, "details": {"n_pairs": len(M), "missing_spans": missing_spans, "ref_spans": src}}

    weights_np = np.asarray(weights, dtype=np.float64)
    scores_np = np.asarray(scores, dtype=np.float64)
    S = float(100.0 * (weights_np * scores_np).sum() / max(1e-12, weights_np.sum()))
    return {
        "score": S,
        "valid": True,
        "details": {
            "n_pairs": len(M),
            "used_pairs": int(len(weights)),
            "missing_spans": missing_spans,
            "w1": w1, "w2": w2, "rho": rho,
            "ref_spans": src
        }
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Compute EGA* score")
    ap.add_argument("--pairs", type=str, required=True)
    ap.add_argument("--ref_emb", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    out = compute_ega(args.pairs, args.ref_emb, args.config)
    print(json.dumps(out, ensure_ascii=False, indent=2))
