# /root/autodl-tmp/event_eval/src/metrics/aggregate.py
# -*- coding: utf-8 -*-
"""
Aggregate final score:
  S_event = theta1 * EGA* + theta2 * ESD^r + theta3 * ECR
If ESD^r is invalid (|M|<2), set theta2=0 (no renorm by default; can enable via cfg).

Inputs:
  - ref_emb_json: data/embeds/ref/<sample_id>.emb[.merged].json  ← 支持两者，优先用 .emb.merged.json
  - gen_emb_json: data/embeds/gen/<sample_id>.emb[.merged].json
  - pairs.json   : data/match/<pair_id>/pairs.json
  - configs/default.yaml

Outputs:
  - data/scores/<pair_id>/scores.json
      {
        "EGA*": {...}, "ESD^r": {...}, "ECR": {...},
        "S_event": float,
        "weights": {"theta1":..., "theta2":..., "theta3":..., "renorm": false},
        "paths": {...}
      }
"""

from __future__ import annotations
import json
import argparse
import logging
from typing import Any, Dict
from pathlib import Path
import os

from event_eval.src.common.io import read_yaml, ensure_dir, write_json
from event_eval.src.metrics.ega import compute_ega
from event_eval.src.metrics.esdr import compute_esdr
from event_eval.src.metrics.ecr import compute_ecr

LOGGER = logging.getLogger("event_eval.metrics.aggregate")
if not LOGGER.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    LOGGER.addHandler(h)
    LOGGER.setLevel(logging.INFO)


def _thetas_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    agg = cfg.get("aggregate") or {}
    return {
        "theta1": float(agg.get("theta1", 0.50)),
        "theta2": float(agg.get("theta2", 0.35)),
        "theta3": float(agg.get("theta3", 0.15)),
        "renorm": bool(agg.get("renorm", False)),
    }

def _prefer_merged_emb(path: str) -> str:
    """
    If <key>.emb.merged.json exists next to <key>.emb.json, prefer the merged one.
    Otherwise return the original path.
    """
    p = Path(path)
    if p.name.endswith(".emb.json"):
        alt = p.with_name(p.name.replace(".emb.json", ".emb.merged.json"))
        if alt.exists():
            return str(alt)
    return str(p)

def _infer_pair_id(pairs_path: str) -> str:
    p = Path(pairs_path)
    try:
        return p.parent.name
    except Exception:
        return "pair"


def aggregate_scores(ref_emb_json: str, gen_emb_json: str, pairs_json: str, cfg_path: str, out_json: str) -> Dict[str, Any]:
    cfg = read_yaml(cfg_path)
    th = _thetas_from_cfg(cfg)

    # --- 优先使用 .emb.merged.json
    ref_emb_use = _prefer_merged_emb(ref_emb_json)
    gen_emb_use = _prefer_merged_emb(gen_emb_json)

    # compute components
    ega = compute_ega(pairs_json, ref_emb_use, cfg_path)
    esdr = compute_esdr(pairs_json, ref_emb_use, gen_emb_use, cfg_path)
    ecr = compute_ecr(ref_emb_use, gen_emb_use, pairs_json)

    t1, t2, t3 = th["theta1"], th["theta2"], th["theta3"]
    if not esdr.get("valid", False):
        # 方案：ESD^r 置缺省 => 聚合时该项权重记为 0（可选 renorm）
        t2_eff = 0.0
        if th["renorm"]:
            s = t1 + t3
            if s > 0:
                t1, t3 = t1 / s, t3 / s
    else:
        t2_eff = t2

    S_event = float(
        t1 * float(ega.get("score", 0.0)) +
        t2_eff * float(esdr.get("score", 0.0)) +
        t3 * float(ecr.get("score", 0.0))
    )

    out = {
        "EGA*": ega,
        "ESD^r": esdr,
        "ECR": ecr,
        "S_event": S_event,
        "weights": {"theta1": t1, "theta2": t2_eff, "theta3": t3, "renorm": th["renorm"]},
        "paths": {"ref_emb": ref_emb_use, "gen_emb": gen_emb_use, "pairs": pairs_json, "config": cfg_path},
    }

    ensure_dir(Path(out_json).parent)
    write_json(out, out_json, indent=2)
    LOGGER.info(f"Wrote scores: {out_json}  (S_event={S_event:.2f})")
    return out


def parse_args():
    ap = argparse.ArgumentParser(description="Aggregate event-level scores")
    ap.add_argument("--ref_emb", type=str, required=True)
    ap.add_argument("--gen_emb", type=str, required=True)
    ap.add_argument("--pairs", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    aggregate_scores(args.ref_emb, args.gen_emb, args.pairs, args.config, args.out)
