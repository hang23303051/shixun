# src/cli/main.py
from __future__ import annotations
"""
event_eval 全链路批处理入口（统一单一 Conda 环境）
"""
import argparse
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import shutil
import json
import csv

# 当前统一使用“本进程 Python”作为 launcher
LAUNCHER: List[str] = [sys.executable]

# =========================
# 目录约定（去除本机绝对路径）
# =========================
# 假定本文件在：
#   <repo_root>/ref4d_eval/event/src/cli/main.py
# 这里的 ROOT 始终取“事件模块根”（ event/）
ROOT = Path(__file__).resolve().parents[2]      # .../event
PROJECT = ROOT.parents[1]                       # 仓库根，如 Ref4D-VideoBench/
DATA = PROJECT / "outputs" / "event" / "cache"  # 事件维度自己的 data（中间产物）
# 参考侧事件证据（开源随仓库发布）
DATA_META = PROJECT / "data" / "metadata" / "event_evidence"

PKG = "ref4d_eval.event"


# -------------------------
# 路径推导
# -------------------------
def p_ref_video(topic: str, sample_id: str) -> Path:
    # 本地重建用；开源不提供 refvideo 时可以不存在
    return DATA / "refvideo" / topic / f"{sample_id}.mp4"


def p_gen_video(model: str, topic: str, sample_id: str) -> Path:
    """
    兼容多种生成视频存放方式（优先使用统一大仓 data/，再回退到本模块 data/）：
      1) 推荐：<repo_root>/data/genvideo/<model>/<sample_id>.mp4
      2) 兼容：<repo_root>/data/genvideo/<model>/<topic>/<sample_id>.mp4
      3) 旧：  <event_root>/data/genvideo/<model>/<topic>/<sample_id>.mp4
      4) 旧：  <event_root>/data/genvideo/<model>/<sample_id>.mp4
    """
    # 1) 推荐新路径（无 topic）
    p_new = PROJECT / "data" / "genvideo" / model / f"{sample_id}.mp4"
    if p_new.exists():
        return p_new

    # 2) 新仓兼容带 topic
    p_new_topic = PROJECT / "data" / "genvideo" / model / topic / f"{sample_id}.mp4"
    if p_new_topic.exists():
        return p_new_topic

    # 3) 旧本地路径（带 topic）
    p_old = DATA / "genvideo" / model / topic / f"{sample_id}.mp4"
    if p_old.exists():
        return p_old

    # 4) 旧本地路径（无 topic）
    return DATA / "genvideo" / model / f"{sample_id}.mp4"


def p_events_ref(sample_id: str) -> Path:
    return DATA / "events" / "ref" / f"{sample_id}.events.json"


def p_events_gen(pair_id: str) -> Path:
    return DATA / "events" / "gen" / f"{pair_id}.events.json"


def p_events_ref_merged(sample_id: str) -> Path:
    """
    参考侧 merged 事件优先从开源发布的元数据目录读取：
      <repo_root>/data/metadata/event_evidence/events_merged_ref/<sample_id>.newevents.json
    若不存在则回退到旧路径（本模块 data/events_merged/ref/...），方便你本地调试。
    """
    p_meta = DATA_META / "events_merged_ref" / f"{sample_id}.newevents.json"
    if p_meta.exists():
        return p_meta
    return DATA / "events_merged" / "ref" / f"{sample_id}.newevents.json"


def p_events_gen_merged(pair_id: str) -> Path:
    return DATA / "events_merged" / "gen" / f"{pair_id}.newevents.json"


def p_vlm_ref(sample_id: str) -> Path:
    return DATA / "vlm" / "ref" / f"{sample_id}.vlm.json"


def p_vlm_gen(pair_id: str) -> Path:
    return DATA / "vlm" / "gen" / f"{pair_id}.vlm.json"


def p_emb_ref(sample_id: str) -> Path:
    return DATA / "embeds" / "ref" / f"{sample_id}.emb.json"


def p_emb_gen(pair_id: str) -> Path:
    return DATA / "embeds" / "gen" / f"{pair_id}.emb.json"


def p_emb_ref_merged(sample_id: str) -> Path:
    """
    参考侧 merged embedding 优先从开源发布的元数据目录读取：
      <repo_root>/data/metadata/event_evidence/embeds_merged_ref/<sample_id>.emb.merged.json
    若不存在则回退到旧路径（本模块 data/embeds/ref/...），方便你本地调试。
    """
    p_meta = DATA_META / "embeds_merged_ref" / f"{sample_id}.emb.merged.json"
    if p_meta.exists():
        return p_meta
    return DATA / "embeds" / "ref" / f"{sample_id}.emb.merged.json"


def p_emb_gen_merged(pair_id: str) -> Path:
    return DATA / "embeds" / "gen" / f"{pair_id}.emb.merged.json"


def p_scene(video_id: str) -> Path:
    return DATA / "scenes" / f"{video_id}.scenes.json"


def p_match_dir(pair_id: str) -> Path:
    return DATA / "match" / pair_id


def p_gate(pair_id: str) -> Path:
    return p_match_dir(pair_id) / "gate_masks.npz"


def p_cost(pair_id: str) -> Path:
    return p_match_dir(pair_id) / "cost_matrix.npz"


def p_pairs(pair_id: str) -> Path:
    return p_match_dir(pair_id) / "pairs.json"


def p_scores(pair_id: str) -> Path:
    return DATA / "scores" / pair_id / "scores.json"


def p_summary_csv() -> Path:
    return DATA / "scores" / "summary.csv"


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def exists_and_nonempty(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def pair_id_of(sample_id: str, model: str) -> str:
    return f"{sample_id}__{model}"


# 统计参考端 merge 后的事件数（用于 summary.csv 中的 ref_merged_events 列）
def _count_ref_merged_events(sample_id: str) -> str:
    """
    返回字符串：成功则为整数的字符串；失败/缺失则返回空串 ""（便于与历史 CSV 合并）。
    兼容两种结构：
      - 列表：[ {...}, {...}, ... ]
      - 字典：{ "events": [ {...}, ... ], ... }
    """
    p = p_events_ref_merged(sample_id)
    try:
        if p.exists() and p.stat().st_size > 0:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("events"), list):
                return str(len(data["events"]))
            if isinstance(data, list):
                return str(len(data))
    except Exception as e:
        print(f"[summary] read merged events fail: {p} -> {e}")
    return ""


# -------------------------
# Launcher & 子模块执行
# -------------------------
def run_module(launcher: List[str], module: str, args: List[str], env: Optional[Dict[str, str]] = None):
    cmd = list(launcher) + ["-m", module] + args
    print("[CMD]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(PROJECT), env=env)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


# -------------------------
# 结果读取与总表写出
# -------------------------
def _norm_key(s: str) -> str:
    return "".join(ch for ch in s if ch.isalpha()).lower()


def _dig_score(v):
    # 从值里尽可能挖出一个数值；支持直接数值以及常见包装字段/一层嵌套
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        for k in ("S", "score", "value", "val", "mean", "avg", "final"):
            x = v.get(k)
            if isinstance(x, (int, float)):
                return float(x)
        for k in ("summary", "metrics", "event"):
            y = _dig_score(v.get(k))
            if y is not None:
                return y
        for x in v.values():
            y = _dig_score(x)
            if y is not None:
                return y
    return None


def _extract_scores_from_json(scores_json: Path) -> Optional[Tuple[float, float, float, float, Optional[bool]]]:
    try:
        with open(scores_json, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:
        print(f"[summary] read fail: {scores_json} -> {e}")
        return None

    targets = ("EGA", "ERel", "ECR", "S_event")
    targets_norm = {t: _norm_key(t) for t in targets}

    layers = [d]
    for k in ("scores", "summary", "metrics", "event", "event_scores"):
        v = d.get(k)
        if isinstance(v, dict):
            layers.append(v)

    got: Dict[str, float] = {}
    ERel_valid: Optional[bool] = None

    def _maybe_read_ERel_valid(val, cur):
        if cur is not None:
            return cur
        if isinstance(val, dict):
            v = val.get("valid", None)
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
        return cur

    # 1) 直接命中键
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        for t in targets:
            if t in got:
                continue
            for key in (t, t.upper(), t.lower(), f"S_{t}", f"s_{t}"):
                if key in layer:
                    val = layer[key]
                    if t == "ERel":
                        ERel_valid = _maybe_read_ERel_valid(val, ERel_valid)
                    y = _dig_score(val)
                    if isinstance(y, float):
                        got[t] = y
                        break

    # 2) 模糊匹配
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        for k, v in layer.items():
            if not isinstance(k, str):
                continue
            nk = _norm_key(k)
            for t in targets:
                if t in got:
                    continue
                nt = targets_norm[t]
                if nk.startswith(nt) or nt.startswith(nk) or nt in nk or nk in nt:
                    if t == "ERel":
                        ERel_valid = _maybe_read_ERel_valid(v, ERel_valid)
                    y = _dig_score(v)
                    if isinstance(y, float):
                        got[t] = y

    if all(x in got for x in targets):
        return got["EGA"], got["ERel"], got["ECR"], got["S_event"], ERel_valid

    print(f"[summary] keys missing in {scores_json} (got={list(got.keys())})")
    return None


def _write_summary_csv(pairs: List[Tuple[str, str]], out_csv: Optional[Path] = None):
    if out_csv is None:
        out_csv = p_summary_csv()
    ensure_dir(out_csv)

    # 读取旧 summary（若存在），补充新列的兼容
    merged: Dict[Tuple[str, str], Dict[str, str]] = {}
    if out_csv.exists() and out_csv.stat().st_size > 0:
        try:
            with open(out_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row.get("modelname", ""), row.get("sample_id", ""))
                    merged[key] = {
                        "modelname": row.get("modelname", ""),
                        "sample_id": row.get("sample_id", ""),
                        "EGA": row.get("EGA", ""),
                        "ERel": row.get("ERel", ""),
                        "ECR": row.get("ECR", ""),
                        "S_event": row.get("S_event", ""),
                        "ERel_valid": row.get("ERel_valid", ""),
                        "ref_merged_events": row.get("ref_merged_events", ""),
                    }
        except Exception as e:
            print(f"[summary] read existing summary fail: {out_csv} -> {e}")

    # 生成本次的行并合并（新结果覆盖旧结果）
    new_count = 0
    for sample_id, model in pairs:
        sp = p_scores(pair_id_of(sample_id, model))
        if not exists_and_nonempty(sp):
            print(f"[summary] skip (scores missing): {sp}")
            continue
        res = _extract_scores_from_json(sp)
        if res is None:
            print(f"[summary] skip (scores parse fail): {sp}")
            continue

        EGA, ERel, ECR, S_event, ERel_valid = res
        ref_evt_cnt_str = _count_ref_merged_events(sample_id)

        key = (model, sample_id)
        merged[key] = {
            "modelname": model,
            "sample_id": sample_id,
            "EGA": f"{EGA:.6f}",
            "ERel": f"{ERel:.6f}",
            "ECR": f"{ECR:.6f}",
            "S_event": f"{S_event:.6f}",
            "ERel_valid": ("1" if ERel_valid is True else ("0" if ERel_valid is False else "")),
            "ref_merged_events": ref_evt_cnt_str,
        }
        new_count += 1

    # 写回
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["modelname", "sample_id", "EGA", "ERel", "ECR", "S_event", "ERel_valid", "ref_merged_events"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (_, _), row in sorted(merged.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            writer.writerow(row)

    print(f"[summary] merged {len(merged)} rows (+{new_count} new) -> {out_csv}")


# -------------------------
# 步骤：镜头切 + GEBD
# -------------------------
def step_detect(topic: str, sample_id: str, model: str,
                cfg_shot: str, cfg_gebd: str, force: bool):
    ref_vid = p_ref_video(topic, sample_id)
    gen_vid = p_gen_video(model, topic, sample_id)

    # 开源用户只需提供生成视频；refvideo 缺失时不再报错
    if not gen_vid.exists():
        raise FileNotFoundError(f"gen video missing: {gen_vid}")

    ref_scene = p_scene(sample_id)
    gen_scene = p_scene(pair_id_of(sample_id, model))
    ref_evt = p_events_ref(sample_id)
    gen_evt = p_events_gen(pair_id_of(sample_id, model))

    L_trans = LAUNCHER
    L_ddm = LAUNCHER

    # 1) 参考端（若有 refvideo 则跑镜头与 GEBD，否则跳过）
    if ref_vid.exists():
        if force or not exists_and_nonempty(ref_scene):
            ensure_dir(ref_scene)
            run_module(
                L_trans,
                f"{PKG}.src.eventdetect.transnetv2_runner",
                ["--video", str(ref_vid), "--out", str(ref_scene), "--config", str(cfg_shot)],
            )
        else:
            print(f"[detect] skip scenes (ref): {ref_scene.name}")

        if force or not exists_and_nonempty(ref_evt):
            ensure_dir(ref_evt)
            run_module(
                L_ddm,
                f"{PKG}.src.eventdetect.ddm_runner",
                ["--video", str(ref_vid), "--out", str(ref_evt),
                 "--config", str(cfg_gebd), "--scenes", str(ref_scene)],
            )
        else:
            print(f"[detect] skip events (ref): {ref_evt.name}")
    else:
        print(f"[detect] no ref video, skip ref side: {ref_vid}")

    # 2) 生成端：必须跑
    if force or not exists_and_nonempty(gen_scene):
        ensure_dir(gen_scene)
        run_module(
            L_trans,
            f"{PKG}.src.eventdetect.transnetv2_runner",
            ["--video", str(gen_vid), "--out", str(gen_scene), "--config", str(cfg_shot)],
        )
    else:
        print(f"[detect] skip scenes (gen): {gen_scene.name}")

    if force or not exists_and_nonempty(gen_evt):
        ensure_dir(gen_evt)
        run_module(
            L_ddm,
            f"{PKG}.src.eventdetect.ddm_runner",
            ["--video", str(gen_vid), "--out", str(gen_evt),
             "--config", str(cfg_gebd), "--scenes", str(gen_scene)],
        )
    else:
        print(f"[detect] skip events (gen): {gen_evt.name}")


# -------------------------
# 步骤：VLM
# -------------------------
def step_vlm(topic: str, sample_id: str, model: str, cfg_vlm: str, force: bool):
    ref_vid = p_ref_video(topic, sample_id)
    gen_vid = p_gen_video(model, topic, sample_id)
    ref_evt = p_events_ref(sample_id)
    gen_evt = p_events_gen(pair_id_of(sample_id, model))
    ref_vlm = p_vlm_ref(sample_id)
    gen_vlm = p_vlm_gen(pair_id_of(sample_id, model))

    L_vlm = LAUNCHER

    # 参考端：仅在 refvideo + ref events 同时存在时才跑（开源场景通常直接复用 merged 证据，而不会走这一步）
    if ref_vid.exists() and ref_evt.exists():
        if force or not exists_and_nonempty(ref_vlm):
            ensure_dir(ref_vlm)
            run_module(
                L_vlm,
                f"{PKG}.src.vlm.vllama3_infer",
                ["--video", str(ref_vid), "--events", str(ref_evt),
                 "--config", str(cfg_vlm), "--out", str(ref_vlm)],
            )
        else:
            print(f"[vlm] skip (ref): {ref_vlm.name}")
    else:
        print(f"[vlm] no ref video/events, skip ref side: {ref_vid}")

    # 生成端：必须跑
    if force or not exists_and_nonempty(gen_vlm):
        ensure_dir(gen_vlm)
        run_module(
            L_vlm,
            f"{PKG}.src.vlm.vllama3_infer",
            ["--video", str(gen_vid), "--events", str(gen_evt),
             "--config", str(cfg_vlm), "--out", str(gen_vlm)],
        )
    else:
        print(f"[vlm] skip (gen): {gen_vlm.name}")


# -------------------------
# 步骤：E5 嵌入
# -------------------------
def step_embed(sample_id: str, model: str, cfg_embed: str, force: bool):
    ref_vlm = p_vlm_ref(sample_id)
    gen_vlm = p_vlm_gen(pair_id_of(sample_id, model))
    ref_emb = p_emb_ref(sample_id)
    gen_emb = p_emb_gen(pair_id_of(sample_id, model))

    L_emb = LAUNCHER

    # 参考端：
    #   若已经随仓库提供 merged ref embedding（embeds_merged_ref），则直接跳过 ref 端嵌入；
    #   若未提供但本地有 ref_vlm，可以按需计算；
    merged_ref_emb = p_emb_ref_merged(sample_id)
    if not (not force and exists_and_nonempty(merged_ref_emb)):
        if ref_vlm.exists():
            if force or not exists_and_nonempty(ref_emb):
                ensure_dir(ref_emb)
                run_module(
                    L_emb,
                    f"{PKG}.src.embed.e5_encoder",
                    ["--vlm", str(ref_vlm), "--config", str(cfg_embed), "--out", str(ref_emb)],
                )
            else:
                print(f"[embed] skip (ref): {ref_emb.name}")
        else:
            print("[embed] skip (ref): no ref VLM file, assuming merged ref embedding is provided offline if needed")
    else:
        print(f"[embed] skip (ref): merged ref embedding already provided ({merged_ref_emb})")

    # 生成端：照常跑
    if force or not exists_and_nonempty(gen_emb):
        ensure_dir(gen_emb)
        run_module(
            L_emb,
            f"{PKG}.src.embed.e5_encoder",
            ["--vlm", str(gen_vlm), "--config", str(cfg_embed), "--out", str(gen_emb)],
        )
    else:
        print(f"[embed] skip (gen): {gen_emb.name}")


# -------------------------
# 步骤：MERGE
# -------------------------
def step_merge(sample_id: str, model: str, cfg_default: str, force: bool):
    pair = pair_id_of(sample_id, model)
    ref_evt = p_events_ref(sample_id)
    gen_evt = p_events_gen(pair)
    ref_vlm = p_vlm_ref(sample_id)
    gen_vlm = p_vlm_gen(pair)
    ref_emb = p_emb_ref(sample_id)
    gen_emb = p_emb_gen(pair)
    ref_emb_m = p_emb_ref_merged(sample_id)
    gen_emb_m = p_emb_gen_merged(pair)

    L_eval = LAUNCHER

    # 参考端：若已随仓库提供 merged ref embedding，则不再重算
    if not (force or not exists_and_nonempty(ref_emb_m)):
        print(f"[merge] skip (ref): {ref_emb_m.name}")
    else:
        if ref_evt.exists() and ref_vlm.exists() and ref_emb.exists():
            ensure_dir(ref_emb_m)
            run_module(
                L_eval,
                f"{PKG}.src.merge.merger",
                ["--events", str(ref_evt), "--vlm", str(ref_vlm), "--embeds", str(ref_emb),
                 "--out-root", str(DATA), "--cfg", str(cfg_default)],
            )
        else:
            print("[merge] skip (ref): missing ref events / vlm / embeds, assume merged ref evidence is precomputed")

    # 生成端：照常跑
    if not (force or not exists_and_nonempty(gen_emb_m)):
        print(f"[merge] skip (gen): {gen_emb_m.name}")
    else:
        ensure_dir(gen_emb_m)
        run_module(
            L_eval,
            f"{PKG}.src.merge.merger",
            ["--events", str(gen_evt), "--vlm", str(gen_vlm), "--embeds", str(gen_emb),
             "--out-root", str(DATA), "--cfg", str(cfg_default)],
        )


# -------------------------
# 步骤：匹配（门控→代价→匈牙利）
# -------------------------
def _prefer_merged(path_merged: Path, path_orig: Path) -> Path:
    return path_merged if path_merged.exists() else path_orig


def step_match(sample_id: str, model: str, cfg_default: str, force: bool):
    pair = pair_id_of(sample_id, model)
    gate_p = p_gate(pair)
    cost_p = p_cost(pair)
    pairs_p = p_pairs(pair)

    ref_emb = _prefer_merged(p_emb_ref_merged(sample_id), p_emb_ref(sample_id))
    gen_emb = _prefer_merged(p_emb_gen_merged(pair), p_emb_gen(pair))
    ref_evt_m = p_events_ref_merged(sample_id)
    gen_evt_m = p_events_gen_merged(pair)

    L_eval = LAUNCHER

    # 1) gate
    if force or not exists_and_nonempty(gate_p):
        ensure_dir(gate_p)
        run_module(
            L_eval,
            f"{PKG}.src.match.gating",
            ["--ref-events", str(ref_evt_m),
             "--ref-embeds", str(ref_emb),
             "--gen-events", str(gen_evt_m),
             "--gen-embeds", str(gen_emb),
             "--config", str(cfg_default),
             "--out", str(gate_p)],
        )
    else:
        print(f"[gate] skip: {gate_p.name}")

    # 2) cost
    need_cost = force or not exists_and_nonempty(cost_p)
    if need_cost:
        ensure_dir(cost_p)
        run_module(
            L_eval,
            f"{PKG}.src.match.costs",
            ["--gate", str(gate_p), "--config", str(cfg_default), "--out", str(cost_p)],
        )
        if not exists_and_nonempty(cost_p):
            print("[match] cost_matrix.npz 未生成，尝试重建 gate 与 cost ...")
            try:
                if gate_p.exists():
                    gate_p.unlink()
            except Exception:
                pass
            ensure_dir(gate_p)
            run_module(
                L_eval,
                f"{PKG}.src.match.gating",
                ["--ref-events", str(ref_evt_m),
                 "--ref-embeds", str(ref_emb),
                 "--gen-events", str(gen_evt_m),
                 "--gen-embeds", str(gen_emb),
                 "--config", str(cfg_default),
                 "--out", str(gate_p)],
            )
            ensure_dir(cost_p)
            run_module(
                L_eval,
                f"{PKG}.src.match.costs",
                ["--gate", str(gate_p), "--config", str(cfg_default), "--out", str(cost_p)],
            )
            if not exists_and_nonempty(cost_p):
                raise RuntimeError("cost_matrix.npz 仍然缺失，请检查 costs.py 的落盘路径或 gate_masks.npz 的内容。")
    else:
        print(f"[cost] skip: {cost_p.name}")

    # 3) hungarian
    if force or not exists_and_nonempty(pairs_p):
        ensure_dir(pairs_p)
        run_module(
            L_eval,
            f"{PKG}.src.match.hungarian",
            ["--cost", str(cost_p), "--gate", str(gate_p), "--out", str(pairs_p)],
        )
    else:
        print(f"[hung] skip: {pairs_p.name}")


# -------------------------
# 步骤：指标聚合
# -------------------------
def step_metrics(sample_id: str, model: str, cfg_default: str, force: bool):
    pair = pair_id_of(sample_id, model)
    pairs_p = p_pairs(pair)
    scores_p = p_scores(pair)

    ref_emb = _prefer_merged(p_emb_ref_merged(sample_id), p_emb_ref(sample_id))
    gen_emb = _prefer_merged(p_emb_gen_merged(pair), p_emb_gen(pair))

    L_eval = LAUNCHER

    if force or not exists_and_nonempty(scores_p):
        ensure_dir(scores_p)
        run_module(
            L_eval,
            f"{PKG}.src.metrics.aggregate",
            ["--ref_emb", str(ref_emb), "--gen_emb", str(gen_emb),
             "--pairs", str(pairs_p), "--config", str(cfg_default), "--out", str(scores_p)],
        )
    else:
        print(f"[metrics] skip: {scores_p.name}")


# -------------------------
# 单 pair 调度
# -------------------------
def run_single(topic: str, sample_id: str, model: str,
               steps: List[str],
               cfg_default: str, cfg_vlm: str, cfg_embed: str,
               cfg_shot: Optional[str], cfg_gebd: Optional[str],
               force: bool):
    steps = [s.strip().lower() for s in steps]
    if "detect" in steps:
        if not cfg_shot or not cfg_gebd:
            raise ValueError("detect 步需要同时提供 --cfg-shot 与 --cfg-gebd")
        step_detect(topic, sample_id, model, cfg_shot, cfg_gebd, force)
    if "vlm" in steps:
        step_vlm(topic, sample_id, model, cfg_vlm, force)
    if "embed" in steps:
        step_embed(sample_id, model, cfg_embed, force)
    if "merge" in steps:
        step_merge(sample_id, model, cfg_default, force)
    if any(s in steps for s in ("gate", "cost", "hung", "match")):
        step_match(sample_id, model, cfg_default, force)
    if "metrics" in steps or "score" in steps:
        step_metrics(sample_id, model, cfg_default, force)

    # 单 pair 运行后写总表（非 force 时会直接读取已有分数）
    _write_summary_csv([(sample_id, model)])


# -------------------------
# 批量发现
# -------------------------
def discover_samples(topics: List[str]) -> List[Tuple[str, str]]:
    """
    优先从 ref4d_meta.jsonl 发现样本：
      <repo_root>/data/metadata/ref4d_meta.jsonl
    若不存在，则退回到旧逻辑：按 DATA/refvideo/<topic>/*.mp4 搜索。
    """
    items: List[Tuple[str, str]] = []

    meta_path = PROJECT / "data" / "metadata" / "ref4d_meta.jsonl"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    topic = obj.get("topic") or obj.get("theme")
                    sid = obj.get("sample_id") or obj.get("id")
                    if not topic or not sid:
                        continue
                    if topics and topic not in topics:
                        continue
                    items.append((topic, str(sid)))
        except Exception as e:
            print(f"[discover] read meta fail: {meta_path} -> {e}")
        return items

    # fallback：旧逻辑
    for topic in topics:
        ref_dir = DATA / "refvideo" / topic
        if not ref_dir.exists():
            continue
        for mp4 in sorted(ref_dir.glob("*.mp4")):
            items.append((topic, mp4.stem))
    return items


def batch_run(topics: List[str], models: List[str],
              steps: List[str],
              cfg_default: str, cfg_vlm: str, cfg_embed: str,
              cfg_shot: Optional[str], cfg_gebd: Optional[str],
              force: bool):
    samples = discover_samples(topics)
    print(f"[batch] found {len(samples)} ref samples")
    pairs_for_summary: List[Tuple[str, str]] = []

    for topic, sample_id in samples:
        for model in models:
            if not p_gen_video(model, topic, sample_id).exists():
                print(f"[batch] skip (gen video missing): {model}/{topic}/{sample_id}")
                continue
            print(f"\n=== RUN: topic={topic} sample={sample_id} model={model} ===")
            pairs_for_summary.append((sample_id, model))
            try:
                run_single(
                    topic, sample_id, model, steps,
                    cfg_default, cfg_vlm, cfg_embed,
                    cfg_shot, cfg_gebd,
                    force,
                )
            except Exception as e:
                print(f"[ERROR] topic={topic} sample={sample_id} model={model} -> {e}")
                continue

    _write_summary_csv(pairs_for_summary)


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="event_eval end-to-end pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # 单 pair
    p1 = sub.add_parser("run", help="run a single pair")
    p1.add_argument("--topic", required=True)
    p1.add_argument("--sample-id", required=True)
    p1.add_argument("--model", required=True)
    p1.add_argument(
        "--steps",
        default="detect,vlm,embed,merge,match,metrics",
        help="逗号分隔：detect,vlm,embed,merge,gate,cost,hung,match,metrics",
    )
    p1.add_argument("--cfg-default", required=True)
    p1.add_argument("--cfg-vlm", required=True)
    p1.add_argument("--cfg-embed", required=True)
    p1.add_argument("--cfg-shot", required=False)
    p1.add_argument("--cfg-gebd", required=False)
    p1.add_argument("--force", action="store_true")

    # 批量
    p2 = sub.add_parser("batch", help="batch over topics & models")
    p2.add_argument("--topics", required=True, help="如：people_daily,news_v1")
    p2.add_argument("--models", required=True, help="如：modelA,modelB")
    p2.add_argument("--steps", default="detect,vlm,embed,merge,match,metrics")
    p2.add_argument("--cfg-default", required=True)
    p2.add_argument("--cfg-vlm", required=True)
    p2.add_argument("--cfg-embed", required=True)
    p2.add_argument("--cfg-shot", required=False)
    p2.add_argument("--cfg-gebd", required=False)
    p2.add_argument("--force", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.cmd == "run":
        steps = [s for s in args.steps.split(",") if s.strip()]
        run_single(
            args.topic, args.sample_id, args.model,
            steps, args.cfg_default, args.cfg_vlm, args.cfg_embed,
            args.cfg_shot, args.cfg_gebd,
            args.force,
        )
    else:
        topics = [t for t in args.topics.split(",") if t.strip()]
        models = [m for m in args.models.split(",") if m.strip()]
        steps = [s for s in args.steps.split(",") if s.strip()]
        batch_run(
            topics, models, steps,
            args.cfg_default, args.cfg_vlm, args.cfg_embed,
            args.cfg_shot, args.cfg_gebd,
            args.force,
        )


if __name__ == "__main__":
    main()
