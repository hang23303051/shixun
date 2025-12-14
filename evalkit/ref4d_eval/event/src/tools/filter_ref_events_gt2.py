#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 summary.csv 中筛出“参考视频事件数 > 2”的样本（使用 merged 优先，否则用原始 ref）。
输出：data/scores/summary.refevt_gt2.csv
"""
from pathlib import Path
import csv, json, sys

ROOT = Path("/root/autodl-tmp/event_eval").resolve()
DATA = ROOT / "data"
SUMMARY = DATA / "scores" / "summary.csv"
OUT = DATA / "scores" / "summary.refevt_gt2.csv"

def p_events_ref(sample_id: str) -> Path:
    return DATA / "events" / "ref" / f"{sample_id}.events.json"

def p_events_ref_merged(sample_id: str) -> Path:
    return DATA / "events_merged" / "ref" / f"{sample_id}.newevents.json"

def exists_and_nonempty(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0

def count_events(sample_id: str) -> int:
    """
    读取参考视频事件数量。优先使用 merged，再回退到原始 ref。
    同时适配常见结构：list / {"events":[...]} / {"spans":[...]} / 嵌套一层等。
    """
    cand = p_events_ref_merged(sample_id)
    if not exists_and_nonempty(cand):
        cand = p_events_ref(sample_id)
        if not exists_and_nonempty(cand):
            return -1

    try:
        with open(cand, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return -1

    if isinstance(obj, list):
        return len(obj)

    if isinstance(obj, dict):
        # 直达常见键
        for key in ("events", "spans", "segments", "items"):
            v = obj.get(key)
            if isinstance(v, list):
                return len(v)
        # 再看下一层
        for v in obj.values():
            if isinstance(v, list):
                return len(v)
            if isinstance(v, dict):
                for key in ("events", "spans", "segments", "items"):
                    vv = v.get(key)
                    if isinstance(vv, list):
                        return len(vv)
    return -1

def main():
    if not exists_and_nonempty(SUMMARY):
        print(f"[ERR] summary.csv 不存在：{SUMMARY}")
        sys.exit(1)

    rows = []
    with open(SUMMARY, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sample_id = r.get("sample_id", "")
            if not sample_id:
                continue
            n = count_events(sample_id)
            r["_ref_events"] = n
            if n > 2:
                rows.append(r)

    # 输出筛选后的表
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # 只保留原有列（可按需追加 _ref_events）
    fieldnames = [c for c in ("modelname","sample_id","EGA","ESD","ECR","S_event") if c in (rows[0] if rows else {})]
    # 如果希望在结果里带上参考事件数，取消下一行注释：
    # fieldnames = ["modelname","sample_id","EGA","ESD","ECR","S_event","_ref_events"]

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"[OK] 参考事件数 > 2 的样本行：{len(rows)}  ->  {OUT}")

if __name__ == "__main__":
    main()
