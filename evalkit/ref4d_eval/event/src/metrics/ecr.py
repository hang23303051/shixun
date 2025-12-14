# /root/autodl-tmp/event_eval/src/metrics/ecr.py
# -*- coding: utf-8 -*-
"""
ECR: Event Coverage & Redundancy

Inputs:
  - ref_emb_json: data/embeds/ref/<sample_id>.emb[.merged].json
  - gen_emb_json: data/embeds/gen/<sample_id>.emb[.merged].json
  - pairs.json   : data/match/<pair_id>/pairs.json

Metrics:
  Cov = |M| / |E_ref|
  Red = (|E_gen| - |M|) / |E_gen| = 1 - |M|/|E_gen|
  S_ECR = 100 * F1(Cov, 1-Red) = 100 * F1(Cov, Precision)
"""
from __future__ import annotations
import json
import argparse
import logging
from typing import Any, Dict, List
from pathlib import Path

from ..common.io import read_json

LOGGER = logging.getLogger("event_eval.metrics.ecr")
if not LOGGER.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    LOGGER.addHandler(h)
    LOGGER.setLevel(logging.INFO)

def _count_emb_items(obj: Any) -> int:
    """Robustly count number of event embeddings in json (list or dict)."""
    if obj is None:
        return 0
    # Case 1: list of dict items: [{"id":..., "emb":[...]}]
    if isinstance(obj, list):
        c = 0
        for it in obj:
            if isinstance(it, dict):
                emb = it.get("emb") or it.get("embedding") or it.get("vec")
                # 允许没有 id 的条目，只要有有效向量就计数
                if isinstance(emb, list) and len(emb) > 0:
                    c += 1
        # 如果列表不是上述结构，退化为长度
        return c if c > 0 else len(obj)
    # Case 2: dict mapping: { "e0001": [..], ... } 或 { "e0001": {"emb":[..]}, ... }
    if isinstance(obj, dict):
        c = 0
        for _, v in obj.items():
            if isinstance(v, list) and len(v) > 0:
                c += 1
            elif isinstance(v, dict):
                emb = v.get("emb") or v.get("embedding") or v.get("vec")
                if isinstance(emb, list) and len(emb) > 0:
                    c += 1
        # 兜底：若为空但像 events 文件（含 "events"），则不把它当作嵌入计数
        return c
    return 0

def _safe_len_emb_json(path: str) -> int:
    data = read_json(path)
    return _count_emb_items(data)

def _f1(p: float, r: float) -> float:
    den = (p + r)
    if den <= 0:
        return 0.0
    return 2.0 * p * r / den

def compute_ecr(ref_emb_json: str, gen_emb_json: str, pairs_json_path: str) -> Dict[str, Any]:
    n_ref = _safe_len_emb_json(ref_emb_json)
    n_gen = _safe_len_emb_json(gen_emb_json)
    pairs = read_json(pairs_json_path)
    M = pairs.get("M", [])
    m = len(M) if isinstance(M, list) else 0

    cov = (m / n_ref) if n_ref > 0 else 0.0
    prec = (m / n_gen) if n_gen > 0 else 0.0  # precision = 1 - Red
    red = 1.0 - prec
    S = 100.0 * _f1(prec, cov)

    return {
        "score": S,
        "valid": True,
        "details": {
            "n_ref": n_ref, "n_gen": n_gen, "m": m,
            "coverage": cov, "redundancy": red, "precision": prec
        }
    }

def parse_args():
    ap = argparse.ArgumentParser(description="Compute ECR score")
    ap.add_argument("--ref_emb", type=str, required=True)
    ap.add_argument("--gen_emb", type=str, required=True)
    ap.add_argument("--pairs", type=str, required=True)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    out = compute_ecr(args.ref_emb, args.gen_emb, args.pairs)
    print(json.dumps(out, ensure_ascii=False, indent=2))
