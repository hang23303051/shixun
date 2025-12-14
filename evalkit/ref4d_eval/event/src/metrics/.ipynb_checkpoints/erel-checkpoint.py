# /root/autodl-tmp/event_eval/src/metrics/esdr.py
# -*- coding: utf-8 -*-
"""
ESD^r: Edit-&-Segment Duality (temporal relation consistency on matched pairs)

Procedure:
  - Use matched set M = {(i,j)} from pairs.json.
  - Sort M by the start time of the reference events (ascending).
  - Build all unordered pairs on ref side: (i,k), i!=k.
  - Allen's 13 relations with tolerance eps (eq/lt/gt) and the fixed order:
      E → S/Si → F/Fi → D/Di → M/Mi → B/Bi → O/Oi
  - Binary consistency u_ik = 1 if equal; else 0. Uniform weights.
If |M| < 2 => valid=False.
"""

from __future__ import annotations
import json
import argparse
import logging
from typing import Any, Dict, List, Tuple
import os
import numpy as np

from event_eval.src.common.io import read_json, read_yaml

LOGGER = logging.getLogger("event_eval.metrics.esdr")
if not LOGGER.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    LOGGER.addHandler(h)
    LOGGER.setLevel(logging.INFO)


# ---------- path helpers (reuse logic from EGA*) ----------

def _lane_and_key_from_emb_path(emb_path: str) -> Tuple[str, str, str]:
    ap = os.path.abspath(emb_path).replace("\\", "/")
    idx = ap.rfind("/data/")
    data_root = ap[: idx + len("/data/")] if idx >= 0 else os.path.dirname(os.path.dirname(os.path.dirname(ap)))
    lane = "ref" if "/ref/" in ap else ("gen" if "/gen/" in ap else "ref")
    fname = os.path.basename(ap)
    key = fname
    for suf in [".emb.merged.json", ".emb.json", ".json"]:
        if key.endswith(suf):
            key = key[: -len(suf)]
            break
    return data_root, lane, key


def _resolve_events_path_from_emb(emb_path: str) -> Tuple[str, str]:
    data_root, lane, key = _lane_and_key_from_emb_path(emb_path)
    p_merged = os.path.join(data_root, "events_merged", lane, f"{key}.newevents.json")
    p_orig   = os.path.join(data_root, "events",        lane, f"{key}.events.json")
    if os.path.exists(p_merged): return p_merged, "merged"
    return p_orig, "original"


def _load_span_map_from_emb(emb_path: str) -> Tuple[Dict[str, Tuple[float,float]], str]:
    events_path, src = _resolve_events_path_from_emb(emb_path)
    data = read_json(events_path)
    if isinstance(data, dict) and "events" in data:
        data = data["events"]
    spans = {}
    for it in data:
        eid = str(it.get("id") or it.get("eid") or it.get("event_id"))
        s = float(it.get("s", 0.0)); e = float(it.get("e", 0.0))
        spans[eid] = (s, e)
    return spans, src


# ---------- Allen 13 with tolerance ----------

def _eq(x: float, y: float, eps: float) -> bool: return abs(x - y) <= eps
def _lt(x: float, y: float, eps: float) -> bool: return x < (y - eps)
def _gt(x: float, y: float, eps: float) -> bool: return x > (y + eps)

def _allen_relation(a: Tuple[float,float], b: Tuple[float,float], eps: float) -> str:
    sa, ea = a; sb, eb = b
    # E
    if _eq(sa,sb,eps) and _eq(ea,eb,eps): return "E"
    # S / Si
    if _eq(sa,sb,eps) and _lt(ea,eb,eps): return "S"
    if _eq(sa,sb,eps) and _gt(ea,eb,eps): return "Si"
    # F / Fi
    if _eq(ea,eb,eps) and _gt(sa,sb,eps): return "F"
    if _eq(ea,eb,eps) and _lt(sa,sb,eps): return "Fi"
    # D / Di
    if _lt(sb,sa,eps) and _lt(ea,eb,eps): return "D"
    if _lt(sa,sb,eps) and _lt(eb,ea,eps): return "Di"
    # M / Mi
    if _eq(ea,sb,eps): return "M"
    if _eq(eb,sa,eps): return "Mi"
    # B / Bi
    if _lt(ea,sb,eps): return "B"
    if _gt(sa,eb,eps): return "Bi"
    # O / Oi  ← 使用 lt/gt 带容差，避免边界抖动误入
    if _lt(sa,sb,eps) and _lt(sb,ea,eps) and _lt(ea,eb,eps): return "O"
    if _lt(sb,sa,eps) and _lt(sa,eb,eps) and _lt(eb,ea,eps): return "Oi"
    # fallback（极端数值场景）
    return "O" if (sa + ea) <= (sb + eb) else "Oi"


# ---------- API ----------

def compute_esdr(pairs_json_path: str, ref_emb_json: str, gen_emb_json: str, cfg_path: str) -> Dict[str, Any]:
    pairs = read_json(pairs_json_path)
    M = pairs.get("M", [])
    if not isinstance(M, list) or len(M) < 2:
        return {"score": 0.0, "valid": False, "details": {"n_pairs": len(M)}}

    cfg = read_yaml(cfg_path)
    eps = float(((cfg.get("esdr") or {}).get("allen_eps") or {}).get("eq", 1e-3))

    ref_spans, ref_src = _load_span_map_from_emb(ref_emb_json)
    gen_spans, gen_src = _load_span_map_from_emb(gen_emb_json)

    # —— 关键修复：按参考端开始时间对 M 排序（若缺失 span，放末尾）
    def _start_or_inf(eid: str) -> float:
        return ref_spans[eid][0] if eid in ref_spans else float("inf")
    M_sorted = sorted(M, key=lambda x: _start_or_inf(str(x[0])))

    ref_ids = [str(x[0]) for x in M_sorted]
    gen_ids = [str(x[1]) for x in M_sorted]
    Nr = len(M_sorted)

    correct = 0
    total = 0
    missing = 0
    for a in range(Nr):
        for b in range(a + 1, Nr):
            ri, rj = ref_ids[a], ref_ids[b]
            gi, gj = gen_ids[a], gen_ids[b]
            if (ri not in ref_spans) or (rj not in ref_spans) or (gi not in gen_spans) or (gj not in gen_spans):
                missing += 1
                continue
            R_ref = _allen_relation(ref_spans[ri], ref_spans[rj], eps)
            R_gen = _allen_relation(gen_spans[gi], gen_spans[gj], eps)
            total += 1
            if R_ref == R_gen:
                correct += 1

    if total == 0:
        return {"score": 0.0, "valid": False, "details": {"n_pairs": Nr, "checked": 0, "missing_pairs": missing,
                                                           "eps": eps, "ref_spans": ref_src, "gen_spans": gen_src}}

    S = 100.0 * correct / total
    return {"score": S, "valid": True,
            "details": {"n_pairs": Nr, "checked": total, "correct": correct, "missing_pairs": missing,
                        "eps": eps, "ref_spans": ref_src, "gen_spans": gen_src}}


def parse_args():
    ap = argparse.ArgumentParser(description="Compute ESD^r score (Allen 13 relations)")
    ap.add_argument("--pairs", type=str, required=True)
    ap.add_argument("--ref_emb", type=str, required=True)
    ap.add_argument("--gen_emb", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    out = compute_esdr(args.pairs, args.ref_emb, args.gen_emb, args.config)
    print(json.dumps(out, ensure_ascii=False, indent=2))
