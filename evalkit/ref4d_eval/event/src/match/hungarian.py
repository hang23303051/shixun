
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Tuple
import numpy as np

# SciPy 优先；无 SciPy 则回退到纯 NumPy 实现
try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _hungarian_numpy(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """方阵的 O(n^3) Hungarian（Kuhn–Munkres）实现，SciPy 缺失时回退用。"""
    C = C.copy().astype(float)
    n = C.shape[0]
    C -= C.min(axis=1, keepdims=True)
    C -= C.min(axis=0, keepdims=True)

    STAR = -1
    PRIME = -2
    marks = np.zeros_like(C, dtype=int)
    row_covered = np.zeros(n, dtype=bool)
    col_covered = np.zeros(n, dtype=bool)

    # 初始打星：每行尽量选一个 0 且该列未被占用
    for i in range(n):
        js = np.where((C[i] == 0) & (~col_covered))[0]
        if js.size > 0:
            marks[i, js[0]] = STAR
            col_covered[js[0]] = True
    col_covered[:] = False

    def cover_star_cols():
        col_covered[:] = np.any(marks == STAR, axis=0)

    def find_zero():
        for i in range(n):
            if row_covered[i]: continue
            for j in range(n):
                if not col_covered[j] and C[i, j] == 0:
                    return i, j
        return None, None

    def star_in_row(r):
        js = np.where(marks[r] == STAR)[0]
        return js[0] if js.size > 0 else None

    def star_in_col(c):
        is_ = np.where(marks[:, c] == STAR)[0]
        return is_[0] if is_.size > 0 else None

    def prime_in_row(r):
        js = np.where(marks[r] == PRIME)[0]
        return js[0] if js.size > 0 else None

    def augment(path):
        for (r, c) in path:
            if marks[r, c] == STAR:
                marks[r, c] = 0
            else:
                marks[r, c] = STAR

    def clear_primes():
        marks[marks == PRIME] = 0

    cover_star_cols()
    while col_covered.sum() < n:
        while True:
            r, c = find_zero()
            if r is None:
                ur = ~row_covered; uc = ~col_covered
                m = np.min(C[np.ix_(ur, uc)])
                C[ur, :] -= m
                C[:, col_covered] += m
            else:
                marks[r, c] = PRIME
                s = star_in_row(r)
                if s is None:
                    path = [(r, c)]
                    cc = c
                    rr = star_in_col(cc)
                    while rr is not None:
                        path.append((rr, cc))
                        cc = prime_in_row(rr)
                        path.append((rr, cc))
                        rr = star_in_col(cc)
                    augment(path)
                    clear_primes()
                    row_covered[:] = False
                    col_covered[:] = False
                    cover_star_cols()
                    break
                else:
                    row_covered[r] = True
                    col_covered[s] = False

    row_ind = np.arange(n)
    col_ind = np.zeros(n, dtype=int)
    for i in range(n):
        j = np.where(marks[i] == STAR)[0]
        col_ind[i] = j[0]
    return row_ind, col_ind

def _read_weights_from_cost_npz(Cdat) -> Tuple[float, float]:
    """兼容多种写法：优先 w1/w2 → 尝试 meta(json) → 默认值。"""
    # 1) 直接键
    if "w1" in Cdat and "w2" in Cdat:
        try:
            return float(Cdat["w1"]), float(Cdat["w2"])
        except Exception:
            pass
    # 2) 从 meta 解析
    if "meta" in Cdat:
        try:
            meta = json.loads(str(Cdat["meta"]))
            w1 = float(meta.get("w1", 0.7))
            w2 = float(meta.get("w2", 0.3))
            return w1, w2
        except Exception:
            pass
    # 3) 兜底默认
    return 0.7, 0.3

def solve_and_save(cost_npz: str, gate_npz: str, out_json: str):
    Cdat = np.load(cost_npz, allow_pickle=True)
    Gdat = np.load(gate_npz, allow_pickle=True)

    # 兼容键名：C / cost；Nr/Ng 或 nr/ng
    C = (Cdat["C"] if "C" in Cdat else Cdat["cost"]).astype(float)
    Nr = int(Cdat["Nr"] if "Nr" in Cdat else Cdat["nr"])
    Ng = int(Cdat["Ng"] if "Ng" in Cdat else Cdat["ng"])

    w1, w2 = _read_weights_from_cost_npz(Cdat)

    ref_ids = [str(x) for x in Gdat["ref_ids"].tolist()]
    gen_ids = [str(x) for x in Gdat["gen_ids"].tolist()]
    sim     = Gdat["sim_sem"].astype(float)   # [Nr,Ng]
    rt      = Gdat["r_tiou"].astype(float)    # [Nr,Ng]
    gate    = Gdat["gate"].astype(bool)       # [Nr,Ng]

    # 求解
    if _HAS_SCIPY:
        row_ind, col_ind = linear_sum_assignment(C)
    else:
        row_ind, col_ind = _hungarian_numpy(C)

    # 过滤 dummy & 非门控
    M = []
    for r, c in zip(row_ind, col_ind):
        if r < Nr and c < Ng and gate[r, c]:
            s = float(sim[r, c]); u = float(rt[r, c]); q = float(w1 * s + w2 * u)
            M.append([ref_ids[r], gen_ids[c], {"sim_sem": s, "r_tIoU": u, "q": q}])

    meta = {
        "Nr": Nr, "Ng": Ng, "Npad": int(C.shape[0]),
        "w1": float(w1), "w2": float(w2),
        "sources": {"cost_npz": str(cost_npz), "gate_npz": str(gate_npz)}
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"M": M, "meta": meta}, f, ensure_ascii=False, indent=2)
    print(f"[hungarian] wrote: {out_json} (|M|={len(M)})")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--out",  required=True)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    solve_and_save(args.cost, args.gate, args.out)
