# -*- coding: utf-8 -*-
from __future__ import annotations
"""
compose_evidence.py
把合并后的事件时间（events_merged）与合并后的描述（vlm_merged）按 id 对齐，生成仅含“绝对时间 + 文本”的证据文件。
- 不输出相对时间/时长
- 不触发任何重计算
- 支持单样本和按 topic 批量
默认路径（参考侧）：
  events_merged/ref/<sample_id>.newevents.json
  vlm_merged/ref/<sample_id>.vlm.json
输出：
  evidence/ref/<sample_id>.evidence.json
"""

import argparse, json, os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

ROOT = Path("/root/autodl-tmp/event_eval").resolve()
DATA = ROOT / "data"

def _load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _default_paths(side: str, sample_id: str) -> Tuple[Path, Path, Path]:
    # 默认只处理参考侧；如需生成侧，可用 --in-events/--in-vlm/--out 覆盖
    ev = DATA / "events_merged" / side / f"{sample_id}.newevents.json"
    vl = DATA / "vlm_merged" / side / f"{sample_id}.vlm.json"
    out = DATA / "evidence" / side / f"{sample_id}.evidence.json"
    return ev, vl, out

def _compose_one(in_events: Path, in_vlm: Path, out_path: Path):
    if not in_events.is_file():
        raise FileNotFoundError(f"events_merged not found: {in_events}")
    if not in_vlm.is_file():
        raise FileNotFoundError(f"vlm_merged not found: {in_vlm}")

    events = _load_json(in_events)  # list of {id,s_abs,e_abs,...}（合并后的）
    texts  = _load_json(in_vlm)     # list of {id,text}

    # 以 id 精确对齐；若 id 缺失，回退按顺序 zip（长度需一致）
    by_id = {str(x.get("id")): x for x in events if "id" in x}
    out: List[Dict] = []

    all_have_id = all("id" in x for x in texts)
    if all_have_id and set(by_id.keys()) & set(str(t["id"]) for t in texts):
        for t in texts:
            tid = str(t["id"])
            if tid not in by_id:
                # 该条没有对应区间，跳过（也可选择报错，这里稳妥跳过）
                continue
            e = by_id[tid]
            out.append({
                "id": tid,
                "s_abs": float(e.get("s_abs", 0.0)),
                "e_abs": float(e.get("e_abs", 0.0)),
                "text": t.get("text", "")
            })
    else:
        # 回退策略：严格长度一致时按顺序拼接
        if len(texts) != len(events):
            raise RuntimeError(
                f"ID 不可对齐且长度不一致，无法安全拼接：len(events)={len(events)}, len(texts)={len(texts)}\n"
                f"建议检查 merger 产物或提供 --in-events/--in-vlm 明确对应文件。"
            )
        for e, t in zip(events, texts):
            out.append({
                "id": str(e.get("id", "")),
                "s_abs": float(e.get("s_abs", 0.0)),
                "e_abs": float(e.get("e_abs", 0.0)),
                "text": t.get("text", "")
            })

    # 只保留绝对时间与文本，不输出相对时间/时长
    _save_json(out_path, out)
    print(f"[compose] saved -> {out_path}  #items={len(out)}")

def _discover_samples(side: str, topics_csv: Optional[str]) -> List[Tuple[str, str]]:
    # 仅参考侧的批量：扫描 data/refvideo/<topic>/*.mp4
    items: List[Tuple[str, str]] = []
    if topics_csv:
        topics = [t for t in topics_csv.split(",") if t.strip()]
    else:
        topics = sorted([d.name for d in (DATA/"refvideo").iterdir() if d.is_dir()])
    for topic in topics:
        for mp4 in sorted((DATA / "refvideo" / topic).glob("*.mp4")):
            items.append((topic, mp4.stem))
    return items

def main():
    ap = argparse.ArgumentParser(description="Compose absolute-time evidence from merged events & merged texts")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("run", help="单样本")
    p1.add_argument("--side", default="ref", choices=["ref","gen"], help="默认 ref")
    p1.add_argument("--topic", required=False, help="仅用于 gen 侧自定义推导时可忽略")
    p1.add_argument("--sample-id", required=True)
    p1.add_argument("--in-events", default=None, help="覆盖输入 events_merged 路径")
    p1.add_argument("--in-vlm",    default=None, help="覆盖输入 vlm_merged 路径")
    p1.add_argument("--out",       default=None, help="覆盖输出 evidence 路径")

    p2 = sub.add_parser("batch", help="批量（参考侧）")
    p2.add_argument("--topics", required=False, help="逗号分隔；为空则自动扫描 data/refvideo/*")
    p2.add_argument("--side", default="ref", choices=["ref"], help="批量目前仅支持 ref")
    p2.add_argument("--force", action="store_true", help="存在则覆盖")

    args = ap.parse_args()

    if args.cmd == "run":
        if args.in_events and args.in_vlm and args.out:
            in_events = Path(args.in_events)
            in_vlm    = Path(args.in_vlm)
            out_path  = Path(args.out)
        else:
            ev, vl, out = _default_paths(args.side, args.sample_id)
            in_events = Path(args.in_events) if args.in_events else ev
            in_vlm    = Path(args.in_vlm)    if args.in_vlm    else vl
            out_path  = Path(args.out)       if args.out       else out
        _compose_one(in_events, in_vlm, out_path)

    else:
        samples = _discover_samples(args.side, args.topics)
        print(f"[batch] found {len(samples)} samples (ref)")
        for topic, sid in samples:
            ev, vl, out = _default_paths("ref", sid)
            if (not args.force) and out.is_file() and out.stat().st_size > 0:
                print(f"[skip] evidence exists: {out.name}")
                continue
            try:
                _compose_one(ev, vl, out)
            except Exception as e:
                print(f"[ERROR] {topic}/{sid}: {e}")
