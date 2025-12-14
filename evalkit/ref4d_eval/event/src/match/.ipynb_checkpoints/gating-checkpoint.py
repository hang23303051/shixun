# /root/autodl-tmp/event_eval/src/match/gating.py
# -*- coding: utf-8 -*-
"""
Build candidate edge masks and similarity matrices for bipartite matching.

Inputs (merged, per-side):
  - data/events_merged/{ref|gen}/<id>.newevents.json   (normalized [0,1], optional abs seconds)
  - data/embeds/{ref|gen}/<id>.emb.merged.json         (id-aligned embeddings; each item: {"id","emb":[...]} )

Outputs:
  - data/match/<pair_id>/gate_masks.npz
      - ref_ids: list[str]  (length Nr)
      - gen_ids: list[str]  (length Ng)
      - sim_sem: float32 [Nr,Ng] in [0,1]
      - r_tiou : float32 [Nr,Ng] in [0,1]
      - gate   : bool    [Nr,Ng] where (sim_sem >= s0) & (r_tiou >= u0)
      - s0, u0, delta: float32  # delta is the final normalized smoothing used in r_tIoU
      - meta: json string (paths, shapes, and delta decomposition when available)

Notes:
  - Robust to missing embeddings: rows/cols without a valid vector are treated as sim_sem=0.
  - If events don't contain normalized [0,1] (s,e), will attempt to compute from absolute seconds.
"""
from __future__ import annotations
import json
from typing import Any, Dict, List, Tuple
from pathlib import Path
import argparse
import logging
import numpy as np

from event_eval.src.common.io import read_yaml, ensure_dir

LOGGER = logging.getLogger("event_eval.match.gating")
if not LOGGER.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    LOGGER.addHandler(h)
    LOGGER.setLevel(logging.INFO)

# ---------- helpers ----------

def _load_events(path: str) -> List[Dict[str, Any]]:
    dat = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(dat, dict) and "events" in dat:
        evs = dat["events"]
    else:
        evs = dat
    out = []
    for d in evs:
        e = dict(d)
        e["id"] = e.get("id") or e.get("eid") or e.get("event_id")
        e["s"] = float(e.get("s", 0.0))
        e["e"] = float(e.get("e", 0.0))
        if "s_abs" in e and "e_abs" in e:
            e["s_abs"] = float(e["s_abs"]); e["e_abs"] = float(e["e_abs"])
        out.append(e)
    out.sort(key=lambda x: (x["s"], x["e"]))
    return out

def _load_embeds(path: str) -> Dict[str, List[float]]:
    dat = json.loads(Path(path).read_text(encoding="utf-8"))
    m: Dict[str, List[float]] = {}
    if isinstance(dat, dict) and all(isinstance(k, str) for k in dat.keys()):
        for k,v in dat.items():
            if isinstance(v, list):
                m[k] = _l2norm([float(x) for x in v])
        return m
    if isinstance(dat, list):
        for it in dat:
            if not isinstance(it, dict): continue
            eid = it.get("id") or it.get("eid") or it.get("event_id")
            emb = it.get("emb") or it.get("embedding") or it.get("vec")
            if isinstance(eid, str) and isinstance(emb, list) and len(emb) > 0:
                m[eid] = _l2norm([float(x) for x in emb])
    return m

def _l2norm(vec: List[float]) -> List[float]:
    s = float(np.linalg.norm(np.asarray(vec, dtype=np.float64)))
    if s <= 0:
        return []
    return [float(x/s) for x in vec]

def _maybe_norm_from_abs(events: List[Dict[str, Any]]) -> None:
    if not events:
        return
    ok = sum(0.0 - 1e-6 <= e.get("s",0) <= 1.0 + 1e-6 and 0.0 - 1e-6 <= e.get("e",0) <= 1.0 + 1e-6 for e in events)
    if ok >= max(1, int(0.7*len(events))):
        return
    has_abs = all(("s_abs" in e and "e_abs" in e) for e in events)
    if not has_abs:
        return
    s0 = min(e["s_abs"] for e in events)
    e1 = max(e["e_abs"] for e in events)
    dur = max(1e-9, float(e1) - float(s0))
    for e in events:
        e["s"] = (float(e["s_abs"]) - s0) / dur
        e["e"] = (float(e["e_abs"]) - s0) / dur

def _video_abs_duration(events: List[Dict[str, Any]]) -> float | None:
    if not events or not all(("s_abs" in e and "e_abs" in e) for e in events):
        return None
    s0 = min(e["s_abs"] for e in events)
    e1 = max(e["e_abs"] for e in events)
    return float(max(1e-9, e1 - s0))

def _pairwise_cos_sim(A: np.ndarray, B: np.ndarray, valid_r: np.ndarray, valid_c: np.ndarray) -> np.ndarray:
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    C = A @ B.T
    C = C.astype(np.float32)
    if valid_r is not None and valid_c is not None:
        mask = np.outer(~valid_r, np.ones(B.shape[0], dtype=bool)) | np.outer(np.ones(A.shape[0], dtype=bool), ~valid_c)
        C[mask] = -1.0
    return C

def _r_tiou_delta(a: Tuple[float,float], b: Tuple[float,float], delta: float) -> float:
    s1,e1 = a; s2,e2 = b
    inter = max(0.0, min(e1,e2) - max(s1,s2))
    union = max(0.0, max(e1,e2) - min(s1,s2))
    return float((inter + delta) / (union + delta)) if union >= 0.0 else 0.0

def _matrix_r_tiou(ref_events, gen_events, delta: float) -> np.ndarray:
    Nr, Ng = len(ref_events), len(gen_events)
    out = np.zeros((Nr, Ng), dtype=np.float32)
    for i, re in enumerate(ref_events):
        a = (float(re["s"]), float(re["e"]))
        for j, ge in enumerate(gen_events):
            b = (float(ge["s"]), float(ge["e"]))
            out[i,j] = _r_tiou_delta(a,b,delta)
    return out

# ---------- main ----------

def save_gate_npz(ref_events_path: str,
                  ref_embeds_path: str,
                  gen_events_path: str,
                  gen_embeds_path: str,
                  cfg_path: str,
                  out_npz_path: str) -> Dict[str, Any]:
    cfg = read_yaml(cfg_path)

    # thresholds & smoothing
    def _getf(*keys, default=None):
        for k in keys:
            v = cfg
            for kk in k.split("."):
                if isinstance(v, dict) and kk in v:
                    v = v[kk]
                else:
                    v = None; break
            if v is not None:
                try: return float(v)
                except Exception: pass
        return float(default) if default is not None else None

    s0 = _getf("gating.s0","ega.s0","s0", default=0.35)
    u0 = _getf("gating.u0","ega.u0","u0", default=0.25)

    # legacy single-delta (kept for backward compatibility)
    delta_legacy = _getf("gating.delta","ega.delta","delta", default=1e-6)

    # new optional dual-parameter smoothing
    delta_rel = _getf("gating.rtiou.delta_rel", default=None)         # e.g., 0.05 on normalized timeline
    delta_abs_sec = _getf("gating.rtiou.delta_abs_sec", default=None) # e.g., 0.01 seconds

    ref_events = _load_events(ref_events_path)
    gen_events = _load_events(gen_events_path)
    _maybe_norm_from_abs(ref_events)
    _maybe_norm_from_abs(gen_events)

    # map absolute-seconds smoothing to normalized timeline if abs durations are available
    dur_ref = _video_abs_duration(ref_events)
    dur_gen = _video_abs_duration(gen_events)
    delta_abs_norm = None
    if delta_abs_sec is not None:
        cand = []
        if dur_ref is not None:
            cand.append(delta_abs_sec / max(dur_ref, 1e-9))
        if dur_gen is not None:
            cand.append(delta_abs_sec / max(dur_gen, 1e-9))
        if cand:
            delta_abs_norm = max(cand)

    # choose final normalized delta: prefer max(delta_rel, delta_abs_norm); else use legacy
    if delta_rel is not None or delta_abs_norm is not None:
        delta = float(max(x for x in [delta_rel, delta_abs_norm] if x is not None))
    else:
        delta = float(delta_legacy)

    ref_ids = [e["id"] for e in ref_events]
    gen_ids = [e["id"] for e in gen_events]

    # embeddings
    ref_emb_map = _load_embeds(ref_embeds_path)
    gen_emb_map = _load_embeds(gen_embeds_path)
    D = 0
    if ref_emb_map:
        D = len(next(iter(ref_emb_map.values())))
    if gen_emb_map and D == 0:
        D = len(next(iter(gen_emb_map.values())))
    if D == 0:
        LOGGER.warning("No embeddings found on either side; sim_sem will be zeros and gate likely empty.")

    Nr, Ng = len(ref_ids), len(gen_ids)
    A = np.zeros((Nr, D), dtype=np.float32)
    B = np.zeros((Ng, D), dtype=np.float32)
    valid_r = np.zeros((Nr,), dtype=bool)
    valid_c = np.zeros((Ng,), dtype=bool)
    for i,eid in enumerate(ref_ids):
        v = ref_emb_map.get(eid, [])
        if isinstance(v, list) and len(v) == D:
            A[i,:] = np.asarray(v, dtype=np.float32)
            valid_r[i] = np.linalg.norm(A[i,:]) > 0
    for j,eid in enumerate(gen_ids):
        v = gen_emb_map.get(eid, [])
        if isinstance(v, list) and len(v) == D:
            B[j,:] = np.asarray(v, dtype=np.float32)
            valid_c[j] = np.linalg.norm(B[j,:]) > 0

    cos = _pairwise_cos_sim(A, B, valid_r, valid_c)              # [-1,1]
    sim_sem = (cos + 1.0) * 0.5                                  # [0,1]
    r_tiou = _matrix_r_tiou(ref_events, gen_events, delta)       # [0,1]
    gate = (sim_sem >= s0) & (r_tiou >= u0)

    ensure_dir(Path(out_npz_path).parent)
    meta = {
        "ref_events": str(ref_events_path),
        "gen_events": str(gen_events_path),
        "ref_embeds": str(ref_embeds_path),
        "gen_embeds": str(gen_embeds_path),
        "cfg": str(cfg_path),
        "Nr": int(Nr), "Ng": int(Ng),
        "delta": float(delta),
        "delta_rel": float(delta_rel) if delta_rel is not None else None,
        "delta_abs_sec": float(delta_abs_sec) if delta_abs_sec is not None else None,
        "dur_ref_sec": float(dur_ref) if dur_ref is not None else None,
        "dur_gen_sec": float(dur_gen) if dur_gen is not None else None
    }
    np.savez_compressed(
        out_npz_path,
        ref_ids=np.array(ref_ids, dtype=object),
        gen_ids=np.array(gen_ids, dtype=object),
        sim_sem=sim_sem.astype(np.float32),
        r_tiou=r_tiou.astype(np.float32),
        gate=gate.astype(np.bool_),
        s0=np.float32(s0),
        u0=np.float32(u0),
        delta=np.float32(delta),
        meta=json.dumps(meta)
    )
    LOGGER.info(f"Wrote gate masks: {out_npz_path} (Nr={Nr}, Ng={Ng}, s0={s0}, u0={u0}, delta={delta})")
    return {"nr": Nr, "ng": Ng, "out": str(out_npz_path), "s0": float(s0), "u0": float(u0), "delta": float(delta)}

def parse_args():
    ap = argparse.ArgumentParser(description="Build gating masks and similarity matrices for matching (EGA*/ESD^r/ECR shared).")
    ap.add_argument("--ref-events", type=str, required=True)
    ap.add_argument("--ref-embeds", type=str, required=True)
    ap.add_argument("--gen-events", type=str, required=True)
    ap.add_argument("--gen-embeds", type=str, required=True)
    ap.add_argument("--config",     type=str, required=True)
    ap.add_argument("--out",        type=str, required=True, help="data/match/<pair>/gate_masks.npz")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    save_gate_npz(
        ref_events_path=args.ref_events,
        ref_embeds_path=args.ref_embeds,
        gen_events_path=args.gen_events,
        gen_embeds_path=args.gen_embeds,
        cfg_path=args.config,
        out_npz_path=args.out
    )
