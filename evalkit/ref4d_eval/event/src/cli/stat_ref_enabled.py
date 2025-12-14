# /root/autodl-tmp/event_eval/src/cli/stat_ref_enabled.py
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import csv

ROOT = Path("/root/autodl-tmp/event_eval").resolve()
DATA = ROOT / "data"

def p_events_ref(sample_id: str) -> Path:
    return DATA / "events" / "ref" / f"{sample_id}.events.json"

def p_events_ref_merged(sample_id: str) -> Path:
    return DATA / "events_merged" / "ref" / f"{sample_id}.newevents.json"

def discover_samples(topics: List[str]) -> List[Tuple[str, str]]:
    """扫描 refvideo/<topic>/*.mp4 得到 (topic, sample_id) 列表"""
    items: List[Tuple[str, str]] = []
    for topic in topics:
        for mp4 in sorted((DATA / "refvideo" / topic).glob("*.mp4")):
            items.append((topic, mp4.stem))
    return items

def _first_event_list(obj: Any) -> List[Dict]:
    """
    在 JSON 里尽量找到事件列表：
    - 首选 obj["events"] 是 list
    - 其次在所有键里递归找第一个“像事件”的 list[dict]
    """
    if isinstance(obj, dict):
        ev = obj.get("events")
        if isinstance(ev, list):
            return ev
        for v in obj.values():
            got = _first_event_list(v)
            if got:
                return got
    elif isinstance(obj, list):
        # 不递归 list 的元素过深（避免超大遍历），只看前若干个元素是否是 dict
        for v in obj[:5]:
            if isinstance(v, dict):
                # list of dicts 也可认为是事件列表
                return obj
    return []

def _count_enabled(ev: Dict) -> bool:
    """
    判断事件是否“启用”：
    - 优先看 'enabled' 字段
    - 次选 'valid' 字段
    - 两者都没有则默认启用（True）
    """
    if isinstance(ev, dict):
        if "enabled" in ev:
            return bool(ev["enabled"])
        if "valid" in ev:
            return bool(ev["valid"])
    return True

def ref_enabled_count(sample_id: str) -> Tuple[int, int]:
    """返回 (total, enabled)；优先读 merged，否则读原始 events"""
    path = p_events_ref_merged(sample_id)
    if not path.exists():
        path = p_events_ref(sample_id)
    if not path.exists() or path.stat().st_size == 0:
        return 0, 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0, 0
    evs = _first_event_list(data)
    total = len(evs)
    enabled = sum(1 for e in evs if _count_enabled(e))
    return total, enabled

def main():
    ap = argparse.ArgumentParser(description="统计参考视频启用事件数 > 阈值 的样本")
    ap.add_argument("--topics", required=True, help="逗号分隔，如：people_daily,news_v1")
    ap.add_argument("--threshold", type=int, default=2, help="阈值（严格大于）")
    ap.add_argument("--out", default="", help="导出 CSV 路径；默认写到 data/scores/ref_enabled_gt{thr}.csv")
    args = ap.parse_args()

    topics = [t for t in args.topics.split(",") if t.strip()]
    samples = discover_samples(topics)
    thr = args.threshold
    out_path = Path(args.out) if args.out else (DATA / "scores" / f"ref_enabled_gt{thr}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    for topic, sample_id in samples:
        total, enabled = ref_enabled_count(sample_id)
        if enabled > thr:
            rows.append({
                "topic": topic,
                "sample_id": sample_id,
                "ref_events_total": str(total),
                "ref_events_enabled": str(enabled),
            })

    # 输出 CSV
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["topic", "sample_id", "ref_events_total", "ref_events_enabled"])
        writer.writeheader()
        for r in sorted(rows, key=lambda x: (x["topic"], x["sample_id"])):
            writer.writerow(r)

    print(f"[stat] matched {len(rows)} samples (enabled > {thr}) -> {out_path}")

if __name__ == "__main__":
    main()
