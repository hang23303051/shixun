
"""

Procedure:
  - Use matched set M = {(i,j)} from pairs.json.
  - Sort M by the start time of the reference events (ascending).
  - Build all unordered pairs on ref side: (i,k), i!=k.
  - Allen's 13 relations with tolerance eps (eq/lt/gt) and the fixed order:
      E → S/Si → F/Fi → D/Di → M/Mi → B/Bi → O/Oi
  - Consistency:
      * Strict (default): 1 if equal; else 0.  (A=I_13)
      * Soft (enable by YAML `ERel.affinity: default_v1` or custom):
          use A[R_ref, R_gen] ∈ [0,1] as partial credit.
If |M| < 2 => valid=False (treated as "missing", NOT as 0).
"""

from __future__ import annotations
import json
import argparse
import logging
from typing import Any, Dict, List, Tuple
import os
import numpy as np

from ..common.io import read_json, read_yaml
from .ega import _lane_and_key_from_emb_path, _resolve_events_path_from_emb

LOGGER = logging.getLogger("event_eval.metrics.ERel")
if not LOGGER.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    LOGGER.addHandler(h)
    LOGGER.setLevel(logging.INFO)


# ---------- path helpers (reuse logic from EGA*) ----------

def _load_span_map_from_emb(emb_path: str) -> Tuple[Dict[str, Tuple[float,float]], str]:
    """
    统一从 embedding 路径拿到事件 spans:
      - ref: data/metadata/event_evidence/events_merged_ref 或 data/events_merged/ref / data/events/ref
      - gen: outputs/event/cache/events/gen 或 data/events_merged/gen / data/events/gen
    """
    events_path, src = _resolve_events_path_from_emb(emb_path)
    data = read_json(events_path)
    if isinstance(data, dict) and "events" in data:
        data = data["events"]
    spans: Dict[str, Tuple[float, float]] = {}
    for it in data:
        eid = str(it.get("id") or it.get("eid") or it.get("event_id"))
        if not eid:
            continue
        s = float(it.get("s_abs", it.get("s", 0.0)))
        e = float(it.get("e_abs", it.get("e", 0.0)))
        spans[eid] = (s, e)
    return spans, src


# ---------- Allen 13 with tolerance ----------

def _eq(x: float, y: float, eps: float) -> bool: return abs(x - y) <= eps
def _lt(x: float, y: float, eps: float) -> bool: return x < (y - eps)
def _gt(x: float, y: float, eps: float) -> bool: return x > (y + eps)

_ALLEN_LABELS: List[str] = ["E","S","Si","F","Fi","D","Di","M","Mi","B","Bi","O","Oi"]
_L2I: Dict[str, int] = {r:i for i,r in enumerate(_ALLEN_LABELS)}

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
    # O / Oi
    if _lt(sa,sb,eps) and _lt(sb,ea,eps) and _lt(ea,eb,eps): return "O"
    if _lt(sb,sa,eps) and _lt(sa,eb,eps) and _lt(eb,ea,eps): return "Oi"
    # fallback（极端数值场景）
    return "O" if (sa + ea) <= (sb + eb) else "Oi"


# ---------- Affinity matrix helpers (S2.a) ----------

def _affinity_from_cfg(cfg: Dict[str, Any]) -> np.ndarray:
    """
    Build a 13x13 affinity matrix A ∈ [0,1].
    - Default (no cfg or False): identity (strict equality).
    - 'default_v1': a conservative near-neighbor scheme:
        E <-> {S,Si,F,Fi}: 0.70
        M <-> B, Mi <-> Bi: 0.80
        S <-> D, Si <-> Di, F <-> D, Fi <-> Di: 0.75
        O <-> D, Oi <-> Di: 0.60
      (symmetric; diagonal = 1.0; others = 0.0)
    - Custom: cfg['ERel']['affinity']['pairs'] as list of [a,b,w].
    """
    ERel_cfg = (cfg.get("ERel") or {})
    aff_cfg = ERel_cfg.get("affinity", None)

    A = np.zeros((13, 13), dtype=float)
    np.fill_diagonal(A, 1.0)  # strict equality baseline

    if aff_cfg in (None, False, 0):
        return A  # strict mode

    if isinstance(aff_cfg, str) and aff_cfg.lower() == "default_v1":
        def setw(a, b, w):
            ia, ib = _L2I.get(a), _L2I.get(b)
            if ia is None or ib is None: return
            A[ia, ib] = max(A[ia, ib], w)
            A[ib, ia] = max(A[ib, ia], w)

        for lab in ["S","Si","F","Fi"]:
            setw("E", lab, 0.70)

        setw("M", "B", 0.80)
        setw("Mi","Bi", 0.80)

        setw("S", "D", 0.75)
        setw("Si","Di", 0.75)
        setw("F", "D", 0.75)
        setw("Fi","Di", 0.75)

        setw("O", "D", 0.60)
        setw("Oi","Di", 0.60)

        return A

    # custom matrix by pairs
    if isinstance(aff_cfg, dict):
        pairs = aff_cfg.get("pairs", [])
        for it in pairs:
            if isinstance(it, (list, tuple)) and len(it) == 3:
                a, b, w = str(it[0]), str(it[1]), float(it[2])
                ia, ib = _L2I.get(a), _L2I.get(b)
                if ia is None or ib is None: continue
                w = float(np.clip(w, 0.0, 1.0))
                A[ia, ib] = max(A[ia, ib], w)
                A[ib, ia] = max(A[ib, ia], w)
        return A

    # fallback to strict
    return A


def _affinity_score(A: np.ndarray, r1: str, r2: str) -> float:
    i, j = _L2I.get(r1), _L2I.get(r2)
    if i is None or j is None:
        return 0.0
    return float(A[i, j])


# ---------- API ----------

def compute_ERel(pairs_json_path: str, ref_emb_json: str, gen_emb_json: str, cfg_path: str) -> Dict[str, Any]:
    pairs = read_json(pairs_json_path)
    M = pairs.get("M", [])
    if not isinstance(M, list) or len(M) < 2:
        # S1: 置缺省（valid=False），不把缺省当 0 计入聚合
        return {"score": 0.0, "valid": False, "details": {"n_pairs": len(M)}}

    cfg = read_yaml(cfg_path)
    eps = float(((cfg.get("ERel") or {}).get("allen_eps") or {}).get("eq", 1e-3))
    A = _affinity_from_cfg(cfg)  # S2.a affinity matrix (identity if strict)

    ref_spans, ref_src = _load_span_map_from_emb(ref_emb_json)
    gen_spans, gen_src = _load_span_map_from_emb(gen_emb_json)

    # —— 按参考端开始时间对 M 排序（若缺失 span，放末尾）
    def _start_or_inf(eid: str) -> float:
        return ref_spans[eid][0] if eid in ref_spans else float("inf")
    M_sorted = sorted(M, key=lambda x: _start_or_inf(str(x[0])))

    ref_ids = [str(x[0]) for x in M_sorted]
    gen_ids = [str(x[1]) for x in M_sorted]
    Nr = len(M_sorted)

    # 累计
    total = 0           # 参与判定的对数
    missing = 0         # 因缺 span 跳过
    exact = 0           # 严格一致计数（仅用于详情）
    aff_sum = 0.0       # 软一致性累计（含严格一致=1）

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
                exact += 1
            aff_sum += _affinity_score(A, R_ref, R_gen)

    if total == 0:
        return {
            "score":  None, "valid": False,
            "details": {
                "n_pairs": Nr, "checked": 0, "missing_pairs": missing,
                "eps": eps, "ref_spans": ref_src, "gen_spans": gen_src,
                "affinity": "enabled" if (A.sum() > 13.0) else "strict_identity"
            }
        }

    # soft score（strict 情况下等价于原二值打分）
    S = 100.0 * (aff_sum / float(total))
    return {
        "score": float(S), "valid": True,
        "details": {
            "n_pairs": Nr, "checked": total, "exact_equal": exact,
            "missing_pairs": missing, "eps": eps,
            "ref_spans": ref_src, "gen_spans": gen_src,
            "affinity": "enabled" if (A.sum() > 13.0) else "strict_identity"
        }
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Compute ERel score (Allen 13 relations)")
    ap.add_argument("--pairs", type=str, required=True)
    ap.add_argument("--ref_emb", type=str, required=True)
    ap.add_argument("--gen_emb", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    out = compute_ERel(args.pairs, args.ref_emb, args.gen_emb, args.config)
    print(json.dumps(out, ensure_ascii=False, indent=2))
