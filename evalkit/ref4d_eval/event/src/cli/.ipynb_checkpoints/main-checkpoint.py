# /root/autodl-tmp/event_eval/src/cli/main.py
from __future__ import annotations
"""
event_eval 全链路批处理入口（支持跨 Conda 环境的分步执行）
- 默认环境C名称：envC_vlm_eval（VLM/E5/合并/匹配/指标均在此环境）
"""
import argparse
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import shutil

# 目录约定
ROOT = Path("/root/autodl-tmp/event_eval").resolve()
PROJECT = ROOT.parent  # /root/autodl-tmp
DATA = ROOT / "data"

# 默认环境C（VLM/E5/MERGE/MATCH/METRICS）
DEFAULT_ENV_C = "conda:envC_vlm_eval"

# -------------------------
# 路径推导
# -------------------------
def p_ref_video(topic: str, sample_id: str) -> Path:
    return DATA / "refvideo" / topic / f"{sample_id}.mp4"

def p_gen_video(model: str, topic: str, sample_id: str) -> Path:
    return DATA / "genvideo" / model / topic / f"{sample_id}.mp4"

def p_events_ref(sample_id: str) -> Path:
    return DATA / "events" / "ref" / f"{sample_id}.events.json"

def p_events_gen(pair_id: str) -> Path:
    return DATA / "events" / "gen" / f"{pair_id}.events.json"

def p_events_ref_merged(sample_id: str) -> Path:
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

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def exists_and_nonempty(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0

def pair_id_of(sample_id: str, model: str) -> str:
    return f"{sample_id}__{model}"

# -------------------------
# Launcher 解析
# -------------------------
def _guess_python_for_conda_env(env_name: str) -> Optional[str]:
    candidates = [
        f"/root/miniconda3/envs/{env_name}/bin/python",
        f"/opt/conda/envs/{env_name}/bin/python",
        f"/miniconda/envs/{env_name}/bin/python",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def build_launcher(spec: Optional[str]) -> List[str]:
    if not spec or spec.strip().lower() in ("", "current"):
        return [sys.executable]
    spec = spec.strip()
    if spec.startswith("python:"):
        return [spec.split("python:", 1)[1]]
    if spec.startswith("conda:"):
        env = spec.split("conda:", 1)[1]
        if shutil.which("conda"):
            return ["conda", "run", "-n", env, "python"]
        py = _guess_python_for_conda_env(env)
        if py:
            return [py]
        raise RuntimeError(f"找不到 conda 或目标 env 的 python：{spec}")
    if spec.startswith("mamba:"):
        env = spec.split("mamba:", 1)[1]
        if shutil.which("mamba"):
            return ["mamba", "run", "-n", env, "python"]
        py = _guess_python_for_conda_env(env)
        if py:
            return [py]
        raise RuntimeError(f"找不到 mamba 或目标 env 的 python：{spec}")
    return [spec]

def run_module(launcher: List[str], module: str, args: List[str], env: Optional[Dict[str, str]] = None):
    cmd = list(launcher) + ["-m", module] + args
    print("[CMD]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(PROJECT), env=env)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

def _pick_env_c(env_vlm: Optional[str], env_embed: Optional[str]) -> str:
    """
    选择用于 Env-C 步骤的 launcher spec：
      1) 优先 --env-embed
      2) 其次 --env-vlm
      3) 否则默认 DEFAULT_ENV_C（conda:envC_vlm_eval）
    """
    return (env_embed or env_vlm or DEFAULT_ENV_C)

# -------------------------
# 步骤：镜头切 + GEBD
# -------------------------
def step_detect(topic: str, sample_id: str, model: str,
                cfg_shot: str, cfg_gebd: str, force: bool,
                env_transnet: Optional[str], env_ddm: Optional[str]):
    ref_vid = p_ref_video(topic, sample_id)
    gen_vid = p_gen_video(model, topic, sample_id)
    if not ref_vid.exists():
        raise FileNotFoundError(f"ref video missing: {ref_vid}")
    if not gen_vid.exists():
        raise FileNotFoundError(f"gen video missing: {gen_vid}")

    ref_scene = p_scene(sample_id)
    gen_scene = p_scene(pair_id_of(sample_id, model))
    ref_evt = p_events_ref(sample_id)
    gen_evt = p_events_gen(pair_id_of(sample_id, model))

    L_trans = build_launcher(env_transnet)
    L_ddm = build_launcher(env_ddm)

    # 1) 镜头切分
    if force or not exists_and_nonempty(ref_scene):
        ensure_dir(ref_scene)
        run_module(L_trans, "event_eval.src.eventdetect.transnetv2_runner",
                   ["--video", str(ref_vid), "--out", str(ref_scene), "--config", str(cfg_shot)])
    else:
        print(f"[detect] skip scenes (ref): {ref_scene.name}")

    if force or not exists_and_nonempty(gen_scene):
        ensure_dir(gen_scene)
        run_module(L_trans, "event_eval.src.eventdetect.transnetv2_runner",
                   ["--video", str(gen_vid), "--out", str(gen_scene), "--config", str(cfg_shot)])
    else:
        print(f"[detect] skip scenes (gen): {gen_scene.name}")

    # 2) GEBD（DDM-Net，按镜头内推理）
    if force or not exists_and_nonempty(ref_evt):
        ensure_dir(ref_evt)
        run_module(L_ddm, "event_eval.src.eventdetect.ddm_runner",
                   ["--video", str(ref_vid), "--out", str(ref_evt),
                    "--config", str(cfg_gebd), "--scenes", str(ref_scene)])
    else:
        print(f"[detect] skip events (ref): {ref_evt.name}")

    if force or not exists_and_nonempty(gen_evt):
        ensure_dir(gen_evt)
        run_module(L_ddm, "event_eval.src.eventdetect.ddm_runner",
                   ["--video", str(gen_vid), "--out", str(gen_evt),
                    "--config", str(cfg_gebd), "--scenes", str(gen_scene)])
    else:
        print(f"[detect] skip events (gen): {gen_evt.name}")

# -------------------------
# 步骤：VLM
# -------------------------
def step_vlm(topic: str, sample_id: str, model: str, cfg_vlm: str, force: bool,
             env_vlm: Optional[str]):
    ref_vid = p_ref_video(topic, sample_id)
    gen_vid = p_gen_video(model, topic, sample_id)
    ref_evt = p_events_ref(sample_id)
    gen_evt = p_events_gen(pair_id_of(sample_id, model))
    ref_vlm = p_vlm_ref(sample_id)
    gen_vlm = p_vlm_gen(pair_id_of(sample_id, model))

    # 默认落到 Env-C
    env_spec = env_vlm or DEFAULT_ENV_C
    L_vlm = build_launcher(env_spec)

    if force or not exists_and_nonempty(ref_vlm):
        ensure_dir(ref_vlm)
        run_module(L_vlm, "event_eval.src.vlm.vllama3_infer",
                   ["--video", str(ref_vid), "--events", str(ref_evt),
                    "--config", str(cfg_vlm), "--out", str(ref_vlm)])
    else:
        print(f"[vlm] skip: {ref_vlm.name}")

    if force or not exists_and_nonempty(gen_vlm):
        ensure_dir(gen_vlm)
        run_module(L_vlm, "event_eval.src.vlm.vllama3_infer",
                   ["--video", str(gen_vid), "--events", str(gen_evt),
                    "--config", str(cfg_vlm), "--out", str(gen_vlm)])
    else:
        print(f"[vlm] skip: {gen_vlm.name}")

# -------------------------
# 步骤：E5 嵌入
# -------------------------
def step_embed(sample_id: str, model: str, cfg_embed: str, force: bool,
               env_embed: Optional[str], env_vlm: Optional[str]):
    ref_vlm = p_vlm_ref(sample_id)
    gen_vlm = p_vlm_gen(pair_id_of(sample_id, model))
    ref_emb = p_emb_ref(sample_id)
    gen_emb = p_emb_gen(pair_id_of(sample_id, model))

    # 默认落到 Env-C（优先 --env-embed，否则 --env-vlm，否则默认）
    env_spec = _pick_env_c(env_vlm=env_vlm, env_embed=env_embed)
    L_emb = build_launcher(env_spec)

    if force or not exists_and_nonempty(ref_emb):
        ensure_dir(ref_emb)
        run_module(L_emb, "event_eval.src.embed.e5_encoder",
                   ["--vlm", str(ref_vlm), "--config", str(cfg_embed), "--out", str(ref_emb)])
    else:
        print(f"[embed] skip: {ref_emb.name}")

    if force or not exists_and_nonempty(gen_emb):
        ensure_dir(gen_emb)
        run_module(L_emb, "event_eval.src.embed.e5_encoder",
                   ["--vlm", str(gen_vlm), "--config", str(cfg_embed), "--out", str(gen_emb)])
    else:
        print(f"[embed] skip: {gen_emb.name}")

# -------------------------
# 步骤：MERGE（Env-C）
# -------------------------
def step_merge(sample_id: str, model: str, cfg_default: str, force: bool,
               env_vlm: Optional[str], env_embed: Optional[str]):
    pair = pair_id_of(sample_id, model)
    ref_evt = p_events_ref(sample_id)
    gen_evt = p_events_gen(pair)
    ref_vlm = p_vlm_ref(sample_id)
    gen_vlm = p_vlm_gen(pair)
    ref_emb = p_emb_ref(sample_id)
    gen_emb = p_emb_gen(pair)
    ref_emb_m = p_emb_ref_merged(sample_id)
    gen_emb_m = p_emb_gen_merged(pair)

    env_spec = _pick_env_c(env_vlm=env_vlm, env_embed=env_embed)
    L_eval = build_launcher(env_spec)

    if not (force or not exists_and_nonempty(ref_emb_m)):
        print(f"[merge] skip (ref): {ref_emb_m.name}")
    else:
        ensure_dir(ref_emb_m)
        run_module(L_eval, "event_eval.src.merge.merger",
                   ["--events", str(ref_evt), "--vlm", str(ref_vlm), "--embeds", str(ref_emb),
                    "--out-root", str(DATA), "--cfg", str(cfg_default)])

    if not (force or not exists_and_nonempty(gen_emb_m)):
        print(f"[merge] skip (gen): {gen_emb_m.name}")
    else:
        ensure_dir(gen_emb_m)
        run_module(L_eval, "event_eval.src.merge.merger",
                   ["--events", str(gen_evt), "--vlm", str(gen_vlm), "--embeds", str(gen_emb),
                    "--out-root", str(DATA), "--cfg", str(cfg_default)])

# -------------------------
# 步骤：匹配（门控→代价→匈牙利）【Env-C】
# -------------------------
def _prefer_merged(path_merged: Path, path_orig: Path) -> Path:
    return path_merged if path_merged.exists() else path_orig

def step_match(sample_id: str, model: str, cfg_default: str, force: bool,
               env_vlm: Optional[str], env_embed: Optional[str]):
    pair = pair_id_of(sample_id, model)
    gate_p = p_gate(pair)
    cost_p = p_cost(pair)
    pairs_p = p_pairs(pair)

    ref_emb = _prefer_merged(p_emb_ref_merged(sample_id), p_emb_ref(sample_id))
    gen_emb = _prefer_merged(p_emb_gen_merged(pair), p_emb_gen(pair))
    ref_evt_m = p_events_ref_merged(sample_id)
    gen_evt_m = p_events_gen_merged(pair)

    env_spec = _pick_env_c(env_vlm=env_vlm, env_embed=env_embed)
    L_eval = build_launcher(env_spec)

    # 1) gate
    if force or not exists_and_nonempty(gate_p):
        ensure_dir(gate_p)
        run_module(L_eval, "event_eval.src.match.gating",
                   ["--ref-events", str(ref_evt_m),
                    "--ref-embeds", str(ref_emb),
                    "--gen-events", str(gen_evt_m),
                    "--gen-embeds", str(gen_emb),
                    "--config", str(cfg_default),
                    "--out", str(gate_p)])
    else:
        print(f"[gate] skip: {gate_p.name}")

    # 2) cost
    need_cost = force or not exists_and_nonempty(cost_p)
    if need_cost:
        ensure_dir(cost_p)
        run_module(L_eval, "event_eval.src.match.costs",
                   ["--gate", str(gate_p), "--config", str(cfg_default), "--out", str(cost_p)])
        if not exists_and_nonempty(cost_p):
            print("[match] cost_matrix.npz 未生成，尝试重建 gate 与 cost ...")
            try:
                if gate_p.exists():
                    gate_p.unlink()
            except Exception:
                pass
            ensure_dir(gate_p)
            run_module(L_eval, "event_eval.src.match.gating",
                       ["--ref-events", str(ref_evt_m),
                        "--ref-embeds", str(ref_emb),
                        "--gen-events", str(gen_evt_m),
                        "--gen-embeds", str(gen_emb),
                        "--config", str(cfg_default),
                        "--out", str(gate_p)])
            ensure_dir(cost_p)
            run_module(L_eval, "event_eval.src.match.costs",
                       ["--gate", str(gate_p), "--config", str(cfg_default), "--out", str(cost_p)])
            if not exists_and_nonempty(cost_p):
                raise RuntimeError("cost_matrix.npz 仍然缺失，请检查 costs.py 的落盘路径或 gate_masks.npz 的内容。")
    else:
        print(f"[cost] skip: {cost_p.name}")

    # 3) hungarian
    if force or not exists_and_nonempty(pairs_p):
        ensure_dir(pairs_p)
        run_module(L_eval, "event_eval.src.match.hungarian",
                   ["--cost", str(cost_p), "--gate", str(gate_p), "--out", str(pairs_p)])
    else:
        print(f"[hung] skip: {pairs_p.name}")

# -------------------------
# 步骤：指标聚合【Env-C】
# -------------------------
def step_metrics(sample_id: str, model: str, cfg_default: str, force: bool,
                 env_vlm: Optional[str], env_embed: Optional[str]):
    pair = pair_id_of(sample_id, model)
    pairs_p = p_pairs(pair)
    scores_p = p_scores(pair)

    ref_emb = _prefer_merged(p_emb_ref_merged(sample_id), p_emb_ref(sample_id))
    gen_emb = _prefer_merged(p_emb_gen_merged(pair), p_emb_gen(pair))

    env_spec = _pick_env_c(env_vlm=env_vlm, env_embed=env_embed)
    L_eval = build_launcher(env_spec)

    if force or not exists_and_nonempty(scores_p):
        ensure_dir(scores_p)
        run_module(L_eval, "event_eval.src.metrics.aggregate",
                   ["--ref_emb", str(ref_emb), "--gen_emb", str(gen_emb),
                    "--pairs", str(pairs_p), "--config", str(cfg_default), "--out", str(scores_p)])
    else:
        print(f"[metrics] skip: {scores_p.name}")

# -------------------------
# 单 pair 调度
# -------------------------
def run_single(topic: str, sample_id: str, model: str,
               steps: List[str],
               cfg_default: str, cfg_vlm: str, cfg_embed: str,
               cfg_shot: Optional[str], cfg_gebd: Optional[str],
               env_transnet: Optional[str], env_ddm: Optional[str],
               env_vlm: Optional[str], env_embed: Optional[str],
               force: bool):
    steps = [s.strip().lower() for s in steps]
    if "detect" in steps:
        if not cfg_shot or not cfg_gebd:
            raise ValueError("detect 步需要同时提供 --cfg-shot 与 --cfg-gebd")
        step_detect(topic, sample_id, model, cfg_shot, cfg_gebd, force, env_transnet, env_ddm)
    if "vlm" in steps:
        step_vlm(topic, sample_id, model, cfg_vlm, force, env_vlm)
    if "embed" in steps:
        step_embed(sample_id, model, cfg_embed, force, env_embed, env_vlm)
    if "merge" in steps:
        step_merge(sample_id, model, cfg_default, force, env_vlm, env_embed)
    if any(s in steps for s in ("gate","cost","hung","match")):
        step_match(sample_id, model, cfg_default, force, env_vlm, env_embed)
    if "metrics" in steps or "score" in steps:
        step_metrics(sample_id, model, cfg_default, force, env_vlm, env_embed)

# -------------------------
# 批量发现
# -------------------------
def discover_samples(topics: List[str]) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for topic in topics:
        for mp4 in sorted((DATA / "refvideo" / topic).glob("*.mp4")):
            items.append((topic, mp4.stem))
    return items

def batch_run(topics: List[str], models: List[str],
              steps: List[str],
              cfg_default: str, cfg_vlm: str, cfg_embed: str,
              cfg_shot: Optional[str], cfg_gebd: Optional[str],
              env_transnet: Optional[str], env_ddm: Optional[str],
              env_vlm: Optional[str], env_embed: Optional[str],
              force: bool):
    samples = discover_samples(topics)
    print(f"[batch] found {len(samples)} ref samples")
    for topic, sample_id in samples:
        for model in models:
            if not p_gen_video(model, topic, sample_id).exists():
                print(f"[batch] skip (gen video missing): {model}/{topic}/{sample_id}")
                continue
            print(f"\n=== RUN: topic={topic} sample={sample_id} model={model} ===")
            try:
                run_single(topic, sample_id, model, steps,
                           cfg_default, cfg_vlm, cfg_embed,
                           cfg_shot, cfg_gebd,
                           env_transnet, env_ddm, env_vlm, env_embed,
                           force)
            except Exception as e:
                print(f"[ERROR] topic={topic} sample={sample_id} model={model} -> {e}")
                continue

# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="event_eval end-to-end pipeline (multi-environment supported)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # 单 pair
    p1 = sub.add_parser("run", help="run a single pair")
    p1.add_argument("--topic", required=True)
    p1.add_argument("--sample-id", required=True)
    p1.add_argument("--model", required=True)
    p1.add_argument("--steps", default="detect,vlm,embed,merge,match,metrics",
                    help="逗号分隔：detect,vlm,embed,merge,gate,cost,hung,match,metrics")
    p1.add_argument("--cfg-default", required=True)
    p1.add_argument("--cfg-vlm", required=True)
    p1.add_argument("--cfg-embed", required=True)
    p1.add_argument("--cfg-shot", required=False)
    p1.add_argument("--cfg-gebd", required=False)
    # 环境选择（可选；不传则默认落到 envC_vlm_eval）
    p1.add_argument("--env-transnet", required=False, default=None)
    p1.add_argument("--env-ddm", required=False, default=None)
    p1.add_argument("--env-vlm", required=False, default=None)
    p1.add_argument("--env-embed", required=False, default=None)
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
    p2.add_argument("--env-transnet", required=False, default=None)
    p2.add_argument("--env-ddm", required=False, default=None)
    p2.add_argument("--env-vlm", required=False, default=None)
    p2.add_argument("--env-embed", required=False, default=None)
    p2.add_argument("--force", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    if args.cmd == "run":
        steps = [s for s in args.steps.split(",") if s.strip()]
        run_single(args.topic, args.sample_id, args.model,
                   steps, args.cfg_default, args.cfg_vlm, args.cfg_embed,
                   args.cfg_shot, args.cfg_gebd,
                   args.env_transnet, args.env_ddm, args.env_vlm, args.env_embed,
                   args.force)
    else:
        topics = [t for t in args.topics.split(",") if t.strip()]
        models = [m for m in args.models.split(",") if m.strip()]
        steps = [s for s in args.steps.split(",") if s.strip()]
        batch_run(topics, models, steps,
                  args.cfg_default, args.cfg_vlm, args.cfg_embed,
                  args.cfg_shot, args.cfg_gebd,
                  args.env_transnet, args.env_ddm, args.env_vlm, args.env_embed,
                  args.force)

if __name__ == "__main__":
    main()
