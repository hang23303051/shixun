# -*- coding: utf-8 -*-
"""
Merge adjacent items within a single video based on:
  - semantic gate (cosine similarity mapped to [0,1])
  - temporal adjacency (absolute-gap preferred; normalized-gap fallback)
and produce merged micro-event set for the video.

行为约定（最小改动、向下兼容）：
- 如果 embeds/*.emb.json 是“微事件级”（记录带 parent_id 或 id 含 '#'），
  则以“每个微事件”为合并单元（拥有独立的 {id, s,e, s_abs,e_abs, text, emb}）。
- 如果 embeds/*.emb.json 是“事件级”（每个事件仅一条向量，id 无 '#'，无 parent_id），
  则以“事件”为合并单元（保持原先事件级行为）。

输出：
  data/events_merged/{ref|gen}/<key>.newevents.json
  data/vlm_merged/{ref|gen}/<key>.vlm.json
  data/embeds/{ref|gen}/<key>.emb.merged.json
  data/merge_map/{ref|gen}/<key>.json
"""

from __future__ import annotations
import os, json, math, argparse, hashlib
from typing import Dict, List, Tuple, Any, Optional

# ---------------------------
# I/O helpers
# ---------------------------

def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _dump_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _infer_lane_and_key(events_path: str, out_dir_root: str) -> Tuple[str, str]:
    """
    events_path: data/events/{ref|gen}/<key>.events.json
    return lane in {"ref","gen"} and <key> (filename stem without suffix)
    """
    ap = os.path.abspath(events_path)
    rel = os.path.relpath(ap, os.path.abspath(out_dir_root))
    rel = rel.replace("\\", "/")
    if "/ref/" in rel: lane = "ref"
    elif "/gen/" in rel: lane = "gen"
    else: lane = "ref"
    stem = os.path.splitext(os.path.basename(ap))[0].replace(".events", "")
    return lane, stem

# ---------------------------
# math helpers
# ---------------------------

def _l2norm(vec: List[float]) -> List[float]:
    s = math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x/s for x in vec]

def _cosine(u: List[float], v: List[float]) -> float:
    if not u or not v:
        return 0.0
    du = math.sqrt(sum(x*x for x in u))
    dv = math.sqrt(sum(x*x for x in v))
    if du == 0.0 or dv == 0.0:
        return 0.0
    dot = sum(x*y for x,y in zip(u,v))
    return dot/(du*dv)

def _vec_add_inplace(acc: List[float], src: List[float], w: float):
    if not src:
        return
    if not acc:
        acc.extend([w*x for x in src])
    else:
        for i,x in enumerate(src):
            if i < len(acc):
                acc[i] += w*x
            else:
                acc.append(w*x)

# ---------------------------
# parsing helpers
# ---------------------------

def _to_event_list(events_json) -> List[Dict[str, Any]]:
    # 接受 {"events":[...]} 或 直接 list
    if isinstance(events_json, dict) and "events" in events_json:
        evs = events_json["events"]
    else:
        evs = events_json
    evs = sorted(evs, key=lambda d: (float(d.get("s", 0.0)), float(d.get("e", 0.0))))
    for d in evs:
        d["id"] = d.get("id") or d.get("eid") or d.get("event_id")
        d["s"] = float(d.get("s", 0.0)); d["e"] = float(d.get("e", 0.0))
        if "s_abs" in d and "e_abs" in d:
            d["s_abs"] = float(d["s_abs"]); d["e_abs"] = float(d["e_abs"])
    return evs

def _to_text_map_from_vlm(vlm_json) -> Dict[str, str]:
    # list of dict 或 dict id->text
    if isinstance(vlm_json, dict) and all(isinstance(k, str) for k in vlm_json.keys()):
        return {k: str(v) for k,v in vlm_json.items()}
    m = {}
    for item in vlm_json:
        eid = item.get("id") or item.get("eid") or item.get("event_id")
        if eid is not None:
            m[str(eid)] = str(item.get("text", "")).strip()
    return m

def _is_micro_embeds(emb_json) -> bool:
    """
    判定 embeds 是否为微事件级：记录包含 parent_id 或 id 中包含 '#'
    """
    if not isinstance(emb_json, list) or not emb_json:
        return False
    for it in emb_json:
        if not isinstance(it, dict): 
            continue
        if "parent_id" in it: 
            return True
        eid = it.get("id")
        if isinstance(eid, str) and "#" in eid:
            return True
    return False

def _build_micro_items_from_embeds(emb_json) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]], Dict[str, str]]:
    """
    从微事件级 embeds 构造：
      - items: [{id,s,e,s_abs?,e_abs?,...}]（以微事件为合并单元）
      - emb_map: id -> L2 向量
      - text_map: id -> text
    要求每条记录自带 s/e（及可选 s_abs/e_abs）；符合 e5_encoder 的产出契约。
    """
    items: List[Dict[str, Any]] = []
    emb_map: Dict[str, List[float]] = {}
    text_map: Dict[str, str] = {}

    for it in emb_json:
        if not isinstance(it, dict): 
            continue
        eid = it.get("id")
        emb = it.get("emb") or it.get("embedding") or it.get("vec")
        if not isinstance(eid, str) or not isinstance(emb, list):
            continue
        s = float(it.get("s", 0.0)); e = float(it.get("e", 0.0))
        rec = {"id": eid, "s": s, "e": e}
        if "s_abs" in it and "e_abs" in it:
            rec["s_abs"] = float(it["s_abs"]); rec["e_abs"] = float(it["e_abs"])
        items.append(rec)
        emb_map[eid] = _l2norm([float(x) for x in emb])
        text_map[eid] = str(it.get("text", "")).strip()

    # 时间排序（稳定）
    items.sort(key=lambda d: (d["s"], d["e"], d["id"]))
    return items, emb_map, text_map

def _to_embed_map_event_level(emb_json, event_ids_in_order: List[str]) -> Dict[str, List[float]]:
    """
    事件级向量（兼容旧格式）：list[dict{id,emb}] 或 list aligned-to-events。
    """
    m: Dict[str, List[float]] = {}
    if isinstance(emb_json, list) and emb_json and isinstance(emb_json[0], dict):
        for item in emb_json:
            eid = item.get("id") or item.get("eid") or item.get("event_id")
            emb = item.get("emb") or item.get("embedding") or item.get("vec")
            if isinstance(eid, str) and isinstance(emb, list):
                m[eid] = _l2norm([float(x) for x in emb])
        return m
    if isinstance(emb_json, list) and len(emb_json) == len(event_ids_in_order):
        for eid, vec in zip(event_ids_in_order, emb_json):
            if isinstance(vec, list):
                m[eid] = _l2norm([float(x) for x in vec])
    return m

# ---------------------------
# core helpers
# ---------------------------

def _event_duration_norm(ev: Dict[str, Any]) -> float:
    return max(0.0, float(ev.get("e", 0.0)) - float(ev.get("s", 0.0)))

def _event_duration_abs(ev: Dict[str, Any]) -> float:
    if "s_abs" in ev and "e_abs" in ev:
        return max(0.0, float(ev["e_abs"]) - float(ev["s_abs"]))
    return 0.0

def _cluster_gap_ok(cluster_end_ev: Dict[str, Any], next_ev: Dict[str, Any],
                    tau_gap_abs_sec: float, gap_norm_fallback: Optional[float]) -> Tuple[bool, float]:
    """
    返回 (ok, gap_used)。优先绝对时间间隙；否则回退到归一化间隙。
    """
    if cluster_end_ev.get("e_abs") is not None and next_ev.get("s_abs") is not None:
        gap = max(0.0, float(next_ev["s_abs"]) - float(cluster_end_ev["e_abs"]))
        return (gap <= tau_gap_abs_sec, gap)
    if gap_norm_fallback is not None:
        gap_n = max(0.0, float(next_ev["s"]) - float(cluster_end_ev["e"]))
        return (gap_n <= gap_norm_fallback, gap_n)
    gap = max(0.0, float(next_ev["s"]) - float(cluster_end_ev["e"]))
    return (True, gap)

def _make_merge_id(idx: int) -> str:
    return f"m{idx:04d}"

# ---------------------------
# main
# ---------------------------

def merge_events(events_path: str, vlm_path: str, embeds_path: str, out_dir_root: str, cfg: dict) -> dict:
    """
    合并（同视频内）并写产物；返回统计信息。
    cfg 需包含：
      cfg['merge']['tau_sem']          : float in [0,1]
      cfg['merge']['gap_abs_sec']      : float seconds
      cfg['merge']['embed_after_merge']: "avg_weighted" | "none"
      cfg['esdr']['allen_eps']['eq']   : float（极短段阈）
    """
    stats = {
        "num_input_items": 0,      # 输入合并单元数（微事件或事件）
        "num_kept_items": 0,
        "num_clusters": 0,
        "num_merged_ops": 0,
        "skipped_short": 0,
        "skipped_no_embed": 0,
        "skipped_low_sem": 0,
        "skipped_large_gap": 0,
        "used_gap_abs": 0,
        "used_gap_norm": 0,
        "lane": None,
        "key": None,
        "source_hash": None,
        "mode": "micro"            # "micro" | "event"
    }

    # --- load raw
    events_raw = _load_json(events_path)
    vlm_raw    = _load_json(vlm_path)
    embeds_raw = _load_json(embeds_path)

    # --- cfg
    merge_cfg = (cfg or {}).get("merge", {})
    esdr_cfg  = (cfg or {}).get("esdr", {})
    tau_sem   = float(merge_cfg.get("tau_sem", 0.85))
    tau_gap_abs_sec = float(merge_cfg.get("gap_abs_sec", 0.30))
    eps       = float(((esdr_cfg.get("allen_eps") or {}).get("eq", 1.0e-3)))

    lane, key = _infer_lane_and_key(events_path, out_dir_root)
    stats["lane"], stats["key"] = lane, key

    # --- decide mode & build items/text/emb maps
    items: List[Dict[str, Any]]
    emb_map: Dict[str, List[float]]
    text_map: Dict[str, str]

    if _is_micro_embeds(embeds_raw):
        # 微事件模式：以 embeds 中的每条微事件为合并单元
        items, emb_map, text_map = _build_micro_items_from_embeds(embeds_raw)
        stats["mode"] = "micro"
    else:
        # 事件模式：保留原行为
        events = _to_event_list(events_raw)
        text_map = _to_text_map_from_vlm(vlm_raw)
        emb_map  = _to_embed_map_event_level(embeds_raw, [e["id"] for e in events])
        items = [{"id": e["id"], "s": e["s"], "e": e["e"], **({} if "s_abs" not in e else {"s_abs": e["s_abs"], "e_abs": e["e_abs"]})} for e in events]
        stats["mode"] = "event"

    stats["num_input_items"] = len(items)
    if len(items) == 0:
        _write_empty_outputs(out_dir_root, lane, key)
        return stats

    # --- filter extremely short items
    kept: List[Dict[str, Any]] = []
    for it in items:
        if (it.get("e", 0.0) - it.get("s", 0.0)) < 2.0 * eps:
            stats["skipped_short"] += 1
            continue
        kept.append(it)
    items = kept
    stats["num_kept_items"] = len(items)

    have_abs = all(("s_abs" in it and "e_abs" in it) for it in items)
    gap_norm_fallback = None if have_abs else 2.0 * eps

    def it_text(iid: str) -> str:
        return text_map.get(iid, "").strip()

    def it_vec(iid: str) -> Optional[List[float]]:
        v = emb_map.get(iid)
        return _l2norm(v) if isinstance(v, list) else None

    # --- if no embedding anywhere -> 1:1 passthrough
    any_vec = any(it_vec(it["id"]) is not None for it in items)
    if not any_vec:
        merged_events, merged_texts, merged_vecs, merge_map = [], [], [], {}
        for j, it in enumerate(items, start=1):
            mid = _make_merge_id(j)
            evt = {"id": mid, "s": float(it["s"]), "e": float(it["e"]), "dur": max(0.0, float(it["e"])-float(it["s"])),
                   "members": [it["id"]]}
            if have_abs:
                evt.update({"s_abs": float(it["s_abs"]), "e_abs": float(it["e_abs"]),
                            "dur_abs": max(0.0, float(it["e_abs"])-float(it["s_abs"]))})
            merged_events.append(evt)
            merged_texts.append({"id": mid, "text": it_text(it["id"])})
            merged_vecs.append({"id": mid, "emb": emb_map.get(it["id"], []) or []})
            merge_map[mid] = [it["id"]]
        stats["num_clusters"] = len(merged_events)
        _write_outputs(out_dir_root, lane, key, merged_events, merged_texts, merged_vecs, merge_map, cfg)
        return stats

    # --- init cluster at first item with embedding
    idx0 = 0
    while idx0 < len(items) and it_vec(items[idx0]["id"]) is None:
        stats["skipped_no_embed"] += 1
        idx0 += 1
    if idx0 == len(items):
        # 所有项都没有向量（不太可能走到，这里兜底）
        return merge_events_passthrough(items, text_map, emb_map, out_dir_root, lane, key, have_abs, stats, cfg)

    merged_events: List[Dict[str, Any]] = []
    merged_texts : List[Dict[str, Any]] = []
    merged_vecs  : List[Dict[str, Any]] = []
    merge_map    : Dict[str, List[str]] = {}

    cluster_members: List[Dict[str, Any]] = [items[idx0]]
    cs, ce = float(items[idx0]["s"]), float(items[idx0]["e"])
    cs_abs = float(items[idx0].get("s_abs", 0.0)) if have_abs else None
    ce_abs = float(items[idx0].get("e_abs", 0.0)) if have_abs else None
    crep_text = it_text(items[idx0]["id"])
    sum_vec: List[float] = []
    sum_w = 0.0

    v0 = it_vec(items[idx0]["id"])
    if v0 is not None:
        w0 = (float(items[idx0]["e_abs"])-float(items[idx0]["s_abs"])) if have_abs else (ce-cs)
        _vec_add_inplace(sum_vec, v0, w0); sum_w += w0

    for k in range(idx0+1, len(items)):
        cur = items[k]
        v_next = it_vec(cur["id"])
        if v_next is None:
            stats["skipped_no_embed"] += 1
            _emit_cluster(cluster_members, cs, ce, cs_abs, ce_abs, crep_text,
                          sum_vec, sum_w, have_abs,
                          merged_events, merged_texts, merged_vecs, merge_map)
            cluster_members = [cur]
            cs, ce = float(cur["s"]), float(cur["e"])
            cs_abs = float(cur.get("s_abs", 0.0)) if have_abs else None
            ce_abs = float(cur.get("e_abs", 0.0)) if have_abs else None
            crep_text = it_text(cur["id"])
            sum_vec, sum_w = [], 0.0
            if v_next is not None:
                w = (float(cur["e_abs"])-float(cur["s_abs"])) if have_abs else (float(cur["e"])-float(cur["s"]))
                _vec_add_inplace(sum_vec, v_next, w); sum_w += w
            continue

        center = _l2norm(sum_vec) if sum_w > 0 else it_vec(cluster_members[-1]["id"])
        cos = _cosine(center, v_next)
        sim_hat = (cos + 1.0) * 0.5

        ok_gap, _ = _cluster_gap_ok(
            {"e_abs": ce_abs, "e": ce},
            {"s_abs": float(cur.get("s_abs", 0.0)), "s": float(cur["s"])},
            tau_gap_abs_sec, gap_norm_fallback
        )
        if have_abs: stats["used_gap_abs"] += 1
        else:        stats["used_gap_norm"] += 1

        if (sim_hat >= tau_sem) and ok_gap:
            # 归并到当前簇
            cluster_members.append(cur)
            ce = max(ce, float(cur["e"]))
            ce_abs = max(ce_abs, float(cur.get("e_abs", ce_abs))) if have_abs else None
            w = (float(cur["e_abs"])-float(cur["s_abs"])) if have_abs else (float(cur["e"])-float(cur["s"]))
            _vec_add_inplace(sum_vec, v_next, w); sum_w += w
            stats["num_merged_ops"] += 1
        else:
            if sim_hat < tau_sem: stats["skipped_low_sem"] += 1
            if not ok_gap:        stats["skipped_large_gap"] += 1
            _emit_cluster(cluster_members, cs, ce, cs_abs, ce_abs, crep_text,
                          sum_vec, sum_w, have_abs,
                          merged_events, merged_texts, merged_vecs, merge_map)
            cluster_members = [cur]
            cs, ce = float(cur["s"]), float(cur["e"])
            cs_abs = float(cur.get("s_abs", 0.0)) if have_abs else None
            ce_abs = float(cur.get("e_abs", 0.0)) if have_abs else None
            crep_text = it_text(cur["id"])
            sum_vec, sum_w = [], 0.0
            w = (float(cur["e_abs"])-float(cur["s_abs"])) if have_abs else (float(cur["e"])-float(cur["s"]))
            _vec_add_inplace(sum_vec, v_next, w); sum_w += w

    # 封尾
    _emit_cluster(cluster_members, cs, ce, cs_abs, ce_abs, crep_text,
                  sum_vec, sum_w, have_abs,
                  merged_events, merged_texts, merged_vecs, merge_map)

    # 重新编号 m0001...
    for idx, (ev, tx, ve) in enumerate(zip(merged_events, merged_texts, merged_vecs), start=1):
        mid = _make_merge_id(idx)
        old_id = ev["id"]
        ev["id"] = tx["id"] = ve["id"] = mid
        merge_map[mid] = merge_map.pop(old_id)

    stats["num_clusters"] = len(merged_events)
    _write_outputs(out_dir_root, lane, key, merged_events, merged_texts, merged_vecs, merge_map, cfg)

    # provenance（轻量）
    src_fps = [events_path, vlm_path, embeds_path]
    h = hashlib.sha256()
    for p in src_fps:
        try:
            with open(p, "rb") as f: h.update(f.read(1024))
        except Exception:
            pass
    h.update(json.dumps(cfg, sort_keys=True).encode("utf-8"))
    stats["source_hash"] = h.hexdigest()[:12]
    return stats

# ---------------------------
# emit & fallback
# ---------------------------

def _emit_cluster(
    members: List[Dict[str, Any]],
    s: float, e: float, s_abs: float|None, e_abs: float|None,
    rep_text: str,
    sum_vec: List[float], sum_w: float, use_abs: bool,
    out_events: List[Dict[str, Any]],
    out_texts:  List[Dict[str, Any]],
    out_vecs:   List[Dict[str, Any]],
    merge_map:  Dict[str, List[str]],
):
    mid = f"tmp{len(out_events)+1:04d}"
    evt = {"id": mid, "s": s, "e": e, "dur": max(0.0, e - s), "members": [m["id"] for m in members]}
    if use_abs:
        evt.update({"s_abs": s_abs, "e_abs": e_abs, "dur_abs": max(0.0, (e_abs or 0.0) - (s_abs or 0.0))})
    out_events.append(evt)
    out_texts.append({"id": mid, "text": rep_text})
    if sum_w > 0 and sum_vec:
        merged = _l2norm(sum_vec)
    else:
        merged = []
    out_vecs.append({"id": mid, "emb": merged})
    merge_map[mid] = [m["id"] for m in members]

def _write_outputs(out_root: str, lane: str, key: str,
                   events, texts, vecs, merge_map, cfg: dict):
    p_events = os.path.join(out_root, "events_merged", lane, f"{key}.newevents.json")
    p_vlm    = os.path.join(out_root, "vlm_merged", lane, f"{key}.vlm.json")
    p_emb    = os.path.join(out_root, "embeds", lane, f"{key}.emb.merged.json")
    p_map    = os.path.join(out_root, "merge_map", lane, f"{key}.json")
    _dump_json(events, p_events)
    _dump_json(texts,  p_vlm)
    _dump_json(vecs,   p_emb)
    _dump_json(merge_map, p_map)

def _write_empty_outputs(out_root: str, lane: str, key: str):
    _write_outputs(out_root, lane, key, [], [], [], {}, cfg={})

def merge_events_passthrough(items, text_map, emb_map, out_root, lane, key, have_abs, stats, cfg):
    merged_events, merged_texts, merged_vecs, merge_map = [], [], [], {}
    for j, it in enumerate(items, start=1):
        mid = _make_merge_id(j)
        evt = {"id": mid, "s": float(it["s"]), "e": float(it["e"]), "dur": max(0.0, float(it["e"])-float(it["s"])),
               "members": [it["id"]]}
        if have_abs:
            evt.update({"s_abs": float(it["s_abs"]), "e_abs": float(it["e_abs"]),
                        "dur_abs": max(0.0, float(it["e_abs"])-float(it["s_abs"]))})
        merged_events.append(evt)
        merged_texts.append({"id": mid, "text": text_map.get(it["id"], "")})
        merged_vecs.append({"id": mid, "emb": emb_map.get(it["id"], []) or []})
        merge_map[mid] = [it["id"]]
    stats["num_clusters"] = len(merged_events)
    _write_outputs(out_root, lane, key, merged_events, merged_texts, merged_vecs, merge_map, cfg)
    return stats

# ---------------------------
# CLI
# ---------------------------

def _load_cfg_from_yaml_or_none(path: str|None) -> dict:
    if not path: return {}
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def _merge_cli():
    ap = argparse.ArgumentParser("Micro-Event Merge (semantic + adjacency)")
    ap.add_argument("--events", required=True, help="data/events/{ref|gen}/<id>.events.json")
    ap.add_argument("--vlm",    required=True, help="data/vlm/{ref|gen}/<id>.vlm.json")
    ap.add_argument("--embeds", required=True, help="data/embeds/{ref|gen}/<id>.emb.json  (微事件或事件级)")
    ap.add_argument("--out-root", default="data", help="data/ root")
    ap.add_argument("--cfg", default="", help="configs/default.yaml (可选)")
    # quick overrides
    ap.add_argument("--tau-sem", type=float, default=None)
    ap.add_argument("--gap-abs-sec", type=float, default=None)
    ap.add_argument("--eps", type=float, default=None)
    args = ap.parse_args()

    cfg = _load_cfg_from_yaml_or_none(args.cfg)
    if "merge" not in cfg: cfg["merge"] = {}
    if "esdr"  not in cfg: cfg["esdr"]  = {"allen_eps": {"eq": 1.0e-3}}
    if args.tau_sem is not None:
        cfg["merge"]["tau_sem"] = float(args.tau_sem)
    if args.gap_abs_sec is not None:
        cfg["merge"]["gap_abs_sec"] = float(args.gap_abs_sec)
    if args.eps is not None:
        cfg["esdr"]["allen_eps"]["eq"] = float(args.eps)

    stats = merge_events(args.events, args.vlm, args.embeds, args.out_root, cfg)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    _merge_cli()
