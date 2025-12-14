# /root/autodl-tmp/event_eval/src/metrics/aggregate.py
# -*- coding: utf-8 -*-
"""
Aggregate final score:
  S_event = theta1 * EGA* + theta2 * ERel + theta3 * ECR

ERel 无效 (|M|<2) 时（置缺省）：
  - theta2=0，且其余两项按策略分配（不把缺省当 0）：
      equal（默认 0.5/0.5）/ proportional / keep
  - 可在 YAML 控制：
      aggregate:
        theta1: 0.50
        theta2: 0.35
        theta3: 0.15
        renorm: false          # True 等价于 ERel_missing: proportional
        ERel_missing: "equal"  # equal / proportional / keep
        # 自定义 equal 的数值：
        # ERel_missing_fallback: [0.5, 0.5]

Inputs:
  - ref_emb_json: data/embeds/ref/<sample_id>.emb[.merged].json
  - gen_emb_json: data/embeds/gen/<pair_id>.emb[.merged].json
  - pairs.json   : data/match/<pair_id>/pairs.json
  - configs/default.yaml

Outputs:
  - data/scores/<pair_id>/scores.json
      {
        "EGA*": {...}, "ERel": {...}, "ECR": {...},
        "S_event": float,
        "weights": {"theta1":..., "theta2":..., "theta3":..., "policy":"...", "renorm": false},
        "paths": {...}
      }
"""

from __future__ import annotations
import json
import argparse
import logging
from typing import Any, Dict
from pathlib import Path

from ..common.io import read_yaml, ensure_dir, write_json
from ..metrics.ega import compute_ega
from ..metrics.erel import compute_ERel
from ..metrics.ecr import compute_ecr

LOGGER = logging.getLogger("event_eval.metrics.aggregate")
if not LOGGER.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    LOGGER.addHandler(h)
    LOGGER.setLevel(logging.INFO)


def _thetas_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    agg = cfg.get("aggregate") or {}
    # 基础权重
    theta1 = float(agg.get("theta1", 0.50))
    theta2 = float(agg.get("theta2", 0.35))
    theta3 = float(agg.get("theta3", 0.15))
    # 兼容旧参数：renorm=True -> proportional
    renorm = bool(agg.get("renorm", False))
    policy = str(agg.get("ERel_missing", "equal")).lower()
    if renorm:
        policy = "proportional"
    # 可选自定义 equal 的数值
    fb = agg.get("ERel_missing_fallback", None)
    if isinstance(fb, (list, tuple)) and len(fb) == 2:
        fb1, fb3 = float(fb[0]), float(fb[1])
    else:
        fb1, fb3 = 0.5, 0.5
    return {
        "theta1": theta1,
        "theta2": theta2,
        "theta3": theta3,
        "renorm": renorm,
        "policy": policy,     # equal / proportional / keep
        "fb1": fb1,
        "fb3": fb3,
    }


def _prefer_merged_emb(path: str) -> str:
    """Prefer <key>.emb.merged.json if exists."""
    p = Path(path)
    if p.name.endswith(".emb.json"):
        alt = p.with_name(p.name.replace(".emb.json", ".emb.merged.json"))
        if alt.exists():
            return str(alt)
    return str(p)


def aggregate_scores(ref_emb_json: str, gen_emb_json: str, pairs_json: str, cfg_path: str, out_json: str) -> Dict[str, Any]:
    cfg = read_yaml(cfg_path)
    th = _thetas_from_cfg(cfg)

    # --- 优先使用 .emb.merged.json
    ref_emb_use = _prefer_merged_emb(ref_emb_json)
    gen_emb_use = _prefer_merged_emb(gen_emb_json)

    # 组件分
    ega = compute_ega(pairs_json, ref_emb_use, cfg_path)
    ERel = compute_ERel(pairs_json, ref_emb_use, gen_emb_use, cfg_path)
    ecr = compute_ecr(ref_emb_use, gen_emb_use, pairs_json)

    # 原始权重
    t1, t2, t3 = th["theta1"], th["theta2"], th["theta3"]
    t1_eff, t2_eff, t3_eff = t1, t2, t3
    policy_used = "normal"

    # 当 ERel 无效：置缺省（theta2=0），其余两项按策略分配
    if not ERel.get("valid", False):
        t2_eff = 0.0
        policy = th["policy"]
        if policy == "proportional":
            s = t1 + t3
            if s > 0:
                t1_eff, t3_eff = t1 / s, t3 / s
            policy_used = "proportional"
        elif policy == "keep":
            # 不改 t1/t3；总和 < 1
            t1_eff, t3_eff = t1, t3
            policy_used = "keep"
        else:
            # 默认 equal：改为 0.5/0.5（或 YAML 自定义）
            t1_eff, t3_eff = th["fb1"], th["fb3"]
            policy_used = "equal"

    S_event = float(
        t1_eff * float(ega.get("score", 0.0)) +
        t2_eff * float(ERel.get("score") or 0.0) +
        t3_eff * float(ecr.get("score", 0.0))
    )

    out = {
        "EGA*": ega,
        "ERel": ERel,
        "ECR": ecr,
        "S_event": S_event,
        "weights": {
            "theta1": t1_eff, "theta2": t2_eff, "theta3": t3_eff,
            "policy": policy_used, "renorm": th["renorm"],
            "base": {"theta1": t1, "theta2": t2, "theta3": t3}
        },
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
