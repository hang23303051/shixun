# /root/autodl-tmp/event_eval/src/match/costs.py
# -*- coding: utf-8 -*-
"""
Build padded cost matrix for partial bipartite matching (Hungarian on square matrix).

Cost:
  c_ij = 1 - (w1*Sim_sem + w2*r_tIoU)
Non-candidate edges -> BIG; pad with dummies (cost=DUMMY ~ 0.999).

This file is robust in I/O keys:
- Saves both modern keys (C, Nr, Ng) and legacy keys (cost, nr, ng).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

from event_eval.src.common.io import read_yaml, ensure_dir

BIG_COST   = 1e6
DUMMY_COST = 0.999  # slightly worse than any plausible real match (in [0,1])

def _load_gate(gate_npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(gate_npz_path, allow_pickle=True)
    sim = d["sim_sem"].astype(np.float64)
    rt  = d["r_tiou"].astype(np.float64)
    gate= d["gate"].astype(bool)
    ref_ids = d["ref_ids"]
    gen_ids = d["gen_ids"]
    return sim, rt, gate, ref_ids, gen_ids

def _weights_from_cfg(cfg_path: str) -> Dict[str, float]:
    cfg = read_yaml(cfg_path)
    ega = (cfg.get("ega") or {}) if isinstance(cfg, dict) else {}
    w1 = float(ega.get("w1", 0.7))
    w2 = float(ega.get("w2", 0.3))
    # normalize just in case
    s = w1 + w2
    if s <= 0:
        w1, w2 = 0.7, 0.3
    else:
        w1, w2 = w1/s, w2/s
    return {"w1": w1, "w2": w2}

def _build_square_cost(sim: np.ndarray, rt: np.ndarray, gate: np.ndarray, w1: float, w2: float) -> Tuple[np.ndarray, int, int]:
    Nr, Ng = sim.shape
    base = 1.0 - (w1 * sim + w2 * rt)
    cost = np.where(gate, base, BIG_COST)
    N = max(Nr, Ng)
    C = np.full((N, N), BIG_COST, dtype=np.float64)
    C[:Nr, :Ng] = cost
    if Nr < N:
        C[Nr:, :Ng] = DUMMY_COST
    if Ng < N:
        C[:Nr, Ng:] = DUMMY_COST
    if Nr < N and Ng < N:
        C[Nr:, Ng:] = DUMMY_COST
    return C, Nr, Ng

def build_and_save(gate_npz_path: str, cfg_path: str, out_npz_path: str) -> Dict[str, Any]:
    sim, rt, gate, ref_ids, gen_ids = _load_gate(gate_npz_path)
    W = _weights_from_cfg(cfg_path)
    C, Nr, Ng = _build_square_cost(sim, rt, gate, W["w1"], W["w2"])
    meta = {
        "Nr": int(Nr), "Ng": int(Ng), "Npad": int(C.shape[0]),
        "w1": W["w1"], "w2": W["w2"],
        "sources": {"gate_npz": str(gate_npz_path), "cfg": str(cfg_path)}
    }
    ensure_dir(Path(out_npz_path).parent)
    # Save with modern + legacy keys for compatibility
    np.savez_compressed(
        out_npz_path,
        C=C, Nr=np.int32(Nr), Ng=np.int32(Ng),
        cost=C, nr=np.int32(Nr), ng=np.int32(Ng),
        w1=np.float32(W["w1"]), w2=np.float32(W["w2"]),
        dummy=np.float32(DUMMY_COST), big=np.float32(BIG_COST),
        ref_ids=ref_ids, gen_ids=gen_ids,
        meta=json.dumps(meta, ensure_ascii=False)
    )
    print(f"[costs] Wrote cost matrix: {out_npz_path} (shape={C.shape}, Nr={Nr}, Ng={Ng}, w1={W['w1']}, w2={W['w2']})")
    return {"nr": int(Nr), "ng": int(Ng), "out": str(out_npz_path), "w1": W["w1"], "w2": W["w2"]}

def parse_args():
    ap = argparse.ArgumentParser(description="Build padded cost matrix for Hungarian")
    ap.add_argument("--gate", type=str, required=True, help="path to gate_masks.npz")
    ap.add_argument("--config", type=str, required=True, help="path to default.yaml")
    ap.add_argument("--out", type=str, required=True, help="path to write cost_matrix.npz")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build_and_save(args.gate, args.config, args.out)
