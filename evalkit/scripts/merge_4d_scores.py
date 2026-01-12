# -*- coding: utf-8 -*-
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

Key = Tuple[str, str]  # (modelname, sample_id)

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")

def _norm(s: str) -> str:
    return re.sub(r"[\s\-]+", "_", (s or "").strip().lower())

def _strip_video_ext(name: str) -> str:
    s = (name or "").strip()
    low = s.lower()
    for ext in VIDEO_EXTS:
        if low.endswith(ext):
            return s[: -len(ext)]
    return s

def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None

def _read_csv_any(path: Path) -> List[Dict[str, str]]:
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with path.open("r", encoding=enc, newline="") as f:
                r = csv.DictReader(f)
                return [dict(row) for row in r]
        except Exception:
            continue
    return []

def _get(row: Dict[str, str], *cands: str) -> Optional[str]:
    if not row:
        return None
    keys = list(row.keys())
    m = {_norm(k): k for k in keys}
    for c in cands:
        nk = _norm(c)
        if nk in m:
            return row.get(m[nk])
    return None

def _set_first(rec: Dict[str, Any], k: str, v: Any) -> None:
    """只写入第一次出现的非空值（避免多源重复覆盖）"""
    if v is None:
        return
    if isinstance(v, str) and v.strip() == "":
        return
    if k not in rec or rec[k] is None or (isinstance(rec[k], str) and str(rec[k]).strip() == ""):
        rec[k] = v

def _pick_first_number(obj: Any, prefer_keys: List[str]) -> Optional[float]:
    """在 scores.json 里兜底找分数"""
    if isinstance(obj, dict):
        # 先按 prefer_keys 找
        for k in prefer_keys:
            if k in obj:
                v = _to_float(obj.get(k))
                if v is not None:
                    return v
        # 再找“像 score 的字段”
        for k, v in obj.items():
            if isinstance(k, str) and ("score" in k.lower() or k.lower() in ("band", "final_band")):
                fv = _to_float(v)
                if fv is not None:
                    return fv
        # 深搜
        for v in obj.values():
            fv = _pick_first_number(v, prefer_keys)
            if fv is not None:
                return fv
    elif isinstance(obj, list):
        for it in obj:
            fv = _pick_first_number(it, prefer_keys)
            if fv is not None:
                return fv
    return None

def _scan_semantic(base: Path, recs: Dict[Key, Dict[str, Any]]) -> None:
    root = base / "outputs" / "semantic" / "scores"
    if not root.exists():
        return
    for p in sorted(root.rglob("*.csv")):
        for row in _read_csv_any(p):
            model = _get(row, "modelname", "model", "模型")
            sid = _get(row, "sample_id", "sample", "name", "名称", "video", "id")
            if not model or not sid:
                continue
            model = model.strip()
            sid = _strip_video_ext(sid.strip())
            k = (model, sid)
            rec = recs.setdefault(k, {"modelname": model, "sample_id": sid})

            v = _to_float(_get(row, "s_base", "S_base", "semantic__s_base", "semanticscore", "semantic_score"))
            _set_first(rec, "semantic_score", v)

def _scan_motion(base: Path, recs: Dict[Key, Dict[str, Any]]) -> None:
    p = base / "outputs" / "motion" / "motion_rrm.csv"
    if not p.exists():
        return
    for row in _read_csv_any(p):
        model = _get(row, "modelname", "model", "模型")
        sid = _get(row, "sample_id", "sample", "name", "名称", "video", "id")
        if not model or not sid:
            continue
        model = model.strip()
        sid = _strip_video_ext(sid.strip())
        k = (model, sid)
        rec = recs.setdefault(k, {"modelname": model, "sample_id": sid})

        v = _to_float(_get(row, "s_motion_w_frz", "S_motion_w_frz", "motion__s_motion_w_frz", "motionscore", "motion_score"))
        _set_first(rec, "motion_score", v)

def _scan_event(base: Path, recs: Dict[Key, Dict[str, Any]]) -> None:
    # 1) 优先 summary.csv
    summary = base / "outputs" / "event" / "scores" / "summary.csv"
    if summary.exists():
        for row in _read_csv_any(summary):
            model = _get(row, "modelname", "model", "模型")
            sid = _get(row, "sample_id", "sample", "name", "名称", "video", "id")
            if not model or not sid:
                continue
            model = model.strip()
            sid = _strip_video_ext(sid.strip())
            k = (model, sid)
            rec = recs.setdefault(k, {"modelname": model, "sample_id": sid})

            v = _to_float(_get(row, "s_event", "S_event", "event__s_event", "eventscore", "event_score", "score"))
            _set_first(rec, "event_score", v)

    # 2) 兜底 scores.json
    root = base / "outputs" / "event" / "scores"
    if not root.exists():
        return
    prefer_keys = [
        "s_event", "event_score", "eventscore",
        "final_score", "score", "score_0_5", "mapped_score_0_5",
        "band", "final_band"
    ]
    for jp in sorted(root.rglob("scores.json")):
        try:
            obj = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue

        # .../scores/<model>/<sample_id>__<model>/scores.json
        parts = jp.parts
        try:
            model = parts[parts.index("scores") + 1]
        except Exception:
            continue
        parent = jp.parent.name
        sid = parent.split("__")[0] if "__" in parent else parent

        model = (model or "").strip()
        sid = _strip_video_ext((sid or "").strip())
        if not model or not sid:
            continue

        v = _pick_first_number(obj, prefer_keys)
        if v is None:
            continue

        k = (model, sid)
        rec = recs.setdefault(k, {"modelname": model, "sample_id": sid})
        _set_first(rec, "event_score", v)

def _scan_world(base: Path, recs: Dict[Key, Dict[str, Any]]) -> None:
    root = base / "outputs" / "world_knowledge"
    if not root.exists():
        return

    # 只扫 csv；兼容你现在的中文表头：模型,名称,得分
    for p in sorted(root.rglob("*.csv")):
        # 可选：跳过 .ipynb_checkpoints
        if ".ipynb_checkpoints" in str(p):
            continue

        for row in _read_csv_any(p):
            model = _get(row, "modelname", "model", "模型")
            sid = _get(row, "sample_id", "sample", "name", "名称", "video", "id")
            if not model or not sid:
                continue
            model = model.strip()
            sid = _strip_video_ext(sid.strip())

            k = (model, sid)
            rec = recs.setdefault(k, {"modelname": model, "sample_id": sid})

            # 你希望用 world__worldscore，但实际 csv 可能叫 得分/score/worldscore
            v = _to_float(_get(row, "worldscore", "world_score", "得分", "score", "mapped_score_0_5", "band"))
            _set_first(rec, "world_score", v)

def _fmt(v: Optional[float]) -> str:
    if v is None:
        return ""
    # 你现有表里 event/semantic/motion 可能是 0-100；world 可能是 0-5；都原样输出
    return f"{v:.6f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Repo root path")
    ap.add_argument("--out", required=True, help="Output per-sample CSV path")
    ap.add_argument("--out-model", default="", help="Optional: output model-avg CSV path (default: <out>_model_avg.csv)")
    args = ap.parse_args()

    base = Path(args.base).resolve()
    out_samples = Path(args.out).resolve()
    out_samples.parent.mkdir(parents=True, exist_ok=True)

    out_model = Path(args.out_model).resolve() if args.out_model else out_samples.with_name(out_samples.stem + "_model_avg.csv")

    recs: Dict[Key, Dict[str, Any]] = {}

    _scan_semantic(base, recs)
    _scan_motion(base, recs)
    _scan_event(base, recs)
    _scan_world(base, recs)

    # ---- 样本级表 ----
    sample_rows = []
    for (m, sid), rec in recs.items():
        s_sem = rec.get("semantic_score")
        s_mot = rec.get("motion_score")
        s_evt = rec.get("event_score")
        s_wld = rec.get("world_score")

        # 没有任何维度分数的行就丢掉
        if s_sem is None and s_mot is None and s_evt is None and s_wld is None:
            continue

        sample_rows.append({
            "modelname": m,
            "sample_id": sid,
            "semantic_score": _fmt(_to_float(s_sem)),
            "motion_score": _fmt(_to_float(s_mot)),
            "event_score": _fmt(_to_float(s_evt)),
            "world_score": _fmt(_to_float(s_wld)),
        })

    sample_rows.sort(key=lambda r: (r["modelname"], r["sample_id"]))

    with out_samples.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["modelname", "sample_id", "semantic_score", "motion_score", "event_score", "world_score"]
        )
        w.writeheader()
        w.writerows(sample_rows)

    # ---- 模型级平均分表 ----
    by_model: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in sample_rows:
        by_model[r["modelname"]].append(r)

    model_rows = []
    for m, rows in sorted(by_model.items(), key=lambda x: x[0]):
        count = len(rows)

        def mean_of(col: str) -> Optional[float]:
            vals = []
            for rr in rows:
                v = _to_float(rr.get(col))
                if v is not None:
                    vals.append(v)
            if not vals:
                return None
            return sum(vals) / len(vals)

        model_rows.append({
            "modelname": m,
            "count_sample_id": str(count),
            "semantic_score": _fmt(mean_of("semantic_score")),
            "motion_score": _fmt(mean_of("motion_score")),
            "event_score": _fmt(mean_of("event_score")),
            "world_score": _fmt(mean_of("world_score")),
        })

    with out_model.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["modelname", "count_sample_id", "semantic_score", "motion_score", "event_score", "world_score"]
        )
        w.writeheader()
        w.writerows(model_rows)

    print(f"[merge_selected] samples : {len(sample_rows)} -> {out_samples}")
    print(f"[merge_selected] models  : {len(model_rows)} -> {out_model}")
    print("[merge_selected] sample header: modelname,sample_id,semantic_score,motion_score,event_score,world_score")
    print("[merge_selected] model  header: modelname,count_sample_id,semantic_score,motion_score,event_score,world_score")

if __name__ == "__main__":
    main()
