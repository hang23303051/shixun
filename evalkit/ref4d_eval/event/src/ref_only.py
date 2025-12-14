# -*- coding: utf-8 -*-
from __future__ import annotations

"""
ref_only.py — 仅对参考视频执行：
镜头切（TransNetV2）→ GEBD（DDM-Net）→ VLM 段级描述（VideoLLaMA3）→ E5 嵌入 → 语义合并（MERGE）

输出：
- data/scenes/<sample_id>.scenes.json
- data/events/ref/<sample_id>.events.json
- data/vlm/ref/<sample_id>.vlm.json
- data/embeds/ref/<sample_id>.emb.json
- data/events_merged/ref/<sample_id>.newevents.json           # 合并后的事件（证据）
- data/embeds/ref/<sample_id>.emb.merged.json                 # 合并后的嵌入
# merger 还会生成：
# - data/vlm_merged/ref/<sample_id>.vlm.json
# - data/merge_map/ref/<sample_id>.json

跨环境支持（可选）：
  --env-transnet  例如 "conda:envA_transnetv2" 或 "python:/root/miniconda3/envs/envA_transnetv2/bin/python"
  --env-ddm       例如 "conda:envB_ddmnet"
  --env-vlm       例如 "conda:envC_vlm_eval"
  --env-embed     例如 "conda:envC_vlm_eval"（默认回落至 env-vlm，再回落到 "conda:envC_vlm_eval"）
"""

import argparse
import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Tuple

# 目录约定
ROOT = Path("/root/autodl-tmp/event_eval").resolve()
PROJECT = ROOT.parent  # /root/autodl-tmp
DATA = ROOT / "data"

# 默认的 Env-C（嵌入/合并/评测）
DEFAULT_ENV_C = "conda:envC_vlm_eval"

# -------------------------
# 路径推导
# -------------------------

def p_ref_video(topic: str, sample_id: str) -> Path:
    return DATA / "refvideo" / topic / f"{sample_id}.mp4"

def p_scene(sample_id: str) -> Path:
    return DATA / "scenes" / f"{sample_id}.scenes.json"

def p_events_ref(sample_id: str) -> Path:
    return DATA / "events" / "ref" / f"{sample_id}.events.json"

def p_vlm_ref(sample_id: str) -> Path:
    return DATA / "vlm" / "ref" / f"{sample_id}.vlm.json"

def p_emb_ref(sample_id: str) -> Path:
    return DATA / "embeds" / "ref" / f"{sample_id}.emb.json"

def p_events_ref_merged(sample_id: str) -> Path:
    return DATA / "events_merged" / "ref" / f"{sample_id}.newevents.json"

def p_emb_ref_merged(sample_id: str) -> Path:
    return DATA / "embeds" / "ref" / f"{sample_id}.emb.merged.json"

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def exists_and_nonempty(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0

# -------------------------
# launcher 解析（与 main.py 一致）
# -------------------------

def _guess_python_for_conda_env(env_name: str) -> Optional[str]:
    for base in ("/root/miniconda3", "/opt/conda", "/miniconda"):
        py = f"{base}/envs/{env_name}/bin/python"
        if os.path.exists(py):
            return py
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
    """选择用于 E5/MERGE 的环境规范：优先 --env-embed，其次 --env-vlm，最后默认 DEFAULT_ENV_C。"""
    return env_embed or env_vlm or DEFAULT_ENV_C

# -------------------------
# 步骤实现（仅参考侧）
# -------------------------

def step_ref_detect(topic: str, sample_id: str,
                    cfg_shot: str, cfg_gebd: str,
                    env_transnet: Optional[str], env_ddm: Optional[str],
                    force: bool):
    """TransNetV2 scenes -> DDM-Net events (ref only)"""
    ref_vid = p_ref_video(topic, sample_id)
    if not ref_vid.exists():
        raise FileNotFoundError(f"ref video missing: {ref_vid}")

    ref_scene = p_scene(sample_id)
    ref_evt = p_events_ref(sample_id)

    L_trans = build_launcher(env_transnet)
    L_ddm = build_launcher(env_ddm)

    # scenes
    if force or not exists_and_nonempty(ref_scene):
        ensure_dir(ref_scene)
        run_module(L_trans, "event_eval.src.eventdetect.transnetv2_runner",
                   ["--video", str(ref_vid), "--out", str(ref_scene), "--config", str(cfg_shot)])
    else:
        print(f"[detect] skip scenes (ref): {ref_scene.name}")

    # events
    if force or not exists_and_nonempty(ref_evt):
        ensure_dir(ref_evt)
        run_module(L_ddm, "event_eval.src.eventdetect.ddm_runner",
                   ["--video", str(ref_vid), "--out", str(ref_evt),
                    "--config", str(cfg_gebd), "--scenes", str(ref_scene)])
    else:
        print(f"[detect] skip events (ref): {ref_evt.name}")

def step_ref_vlm(topic: str, sample_id: str,
                 cfg_vlm: str, env_vlm: Optional[str], force: bool):
    """VideoLLaMA3 段级描述（ref only）"""
    ref_vid = p_ref_video(topic, sample_id)
    ref_evt = p_events_ref(sample_id)
    ref_vlm = p_vlm_ref(sample_id)

    if not ref_evt.exists():
        raise FileNotFoundError(f"events (ref) not found: {ref_evt}，请先执行 detect")

    L_vlm = build_launcher(env_vlm or DEFAULT_ENV_C)
    if force or not exists_and_nonempty(ref_vlm):
        ensure_dir(ref_vlm)
        run_module(L_vlm, "event_eval.src.vlm.vllama3_infer",
                   ["--video", str(ref_vid), "--events", str(ref_evt),
                    "--config", str(cfg_vlm), "--out", str(ref_vlm)])
    else:
        print(f"[vlm] skip: {ref_vlm.name}")

def step_ref_embed(sample_id: str, cfg_embed: str,
                   env_vlm: Optional[str], env_embed: Optional[str], force: bool):
    """E5 文本嵌入（ref only）"""
    ref_vlm = p_vlm_ref(sample_id)
    ref_emb = p_emb_ref(sample_id)

    if not ref_vlm.exists():
        raise FileNotFoundError(f"vlm (ref) not found: {ref_vlm}，请先执行 vlm")

    L_emb = build_launcher(_pick_env_c(env_vlm, env_embed))
    if force or not exists_and_nonempty(ref_emb):
        ensure_dir(ref_emb)
        run_module(L_emb, "event_eval.src.embed.e5_encoder",
                   ["--vlm", str(ref_vlm), "--config", str(cfg_embed), "--out", str(ref_emb)])
    else:
        print(f"[embed] skip: {ref_emb.name}")

def step_ref_merge(sample_id: str, cfg_default: str,
                   env_vlm: Optional[str], env_embed: Optional[str], force: bool):
    """语义合并（ref only）：使用 events+vlm+embeds 生成合并后的微事件与嵌入"""
    ref_evt = p_events_ref(sample_id)
    ref_vlm = p_vlm_ref(sample_id)
    ref_emb = p_emb_ref(sample_id)
    ref_evt_m = p_events_ref_merged(sample_id)
    ref_emb_m = p_emb_ref_merged(sample_id)

    for pth, name in [(ref_evt, "events"), (ref_vlm, "vlm"), (ref_emb, "embeds")]:
        if not pth.exists():
            raise FileNotFoundError(f"{name} (ref) not found: {pth}，请先执行 detect/vlm/embed")

    L_eval = build_launcher(_pick_env_c(env_vlm, env_embed))
    # 以合并后的目标文件作为是否跳过的判断依据
    if force or not (exists_and_nonempty(ref_evt_m) and exists_and_nonempty(ref_emb_m)):
        # merger 会在 data 根目录下写出 events_merged/vlm_merged/embeds.merged/merge_map 等
        run_module(L_eval, "event_eval.src.merge.merger",
                   ["--events", str(ref_evt), "--vlm", str(ref_vlm), "--embeds", str(ref_emb),
                    "--out-root", str(DATA), "--cfg", str(cfg_default)])
    else:
        print(f"[merge] skip: {ref_evt_m.name} & {ref_emb_m.name}")

# -------------------------
# 批量发现
# -------------------------

def discover_samples(topics: List[str]) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for topic in topics:
        for mp4 in sorted((DATA / "refvideo" / topic).glob("*.mp4")):
            items.append((topic, mp4.stem))
    return items

# -------------------------
# CLI
# -------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Reference-only: detect (TransNetV2 + DDM) & VLM & EMBED & MERGE")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # 单样本
    p1 = sub.add_parser("run", help="run pipeline for a single reference video")
    p1.add_argument("--topic", required=True)
    p1.add_argument("--sample-id", required=True)
    p1.add_argument("--steps", default="detect,vlm,embed,merge", help="detect,vlm,embed,merge")
    p1.add_argument("--cfg-shot", required=True, help="TransNetV2 config yaml")
    p1.add_argument("--cfg-gebd", required=True, help="DDM-Net config yaml")
    p1.add_argument("--cfg-vlm", required=True, help="VideoLLaMA3 config yaml")
    p1.add_argument("--cfg-embed", required=True, help="E5 encoder config yaml")
    p1.add_argument("--cfg-default", required=True, help="default.yaml（含 merge 阈值等）")
    p1.add_argument("--env-transnet", required=False, default=None)
    p1.add_argument("--env-ddm", required=False, default=None)
    p1.add_argument("--env-vlm", required=False, default=None)
    p1.add_argument("--env-embed", required=False, default=None)
    p1.add_argument("--force", action="store_true")

    # 批量
    p2 = sub.add_parser("batch", help="batch over topics (reference only)")
    p2.add_argument("--topics", required=True, help="comma-separated topics")
    p2.add_argument("--steps", default="detect,vlm,embed,merge", help="detect,vlm,embed,merge")
    p2.add_argument("--cfg-shot", required=True)
    p2.add_argument("--cfg-gebd", required=True)
    p2.add_argument("--cfg-vlm", required=True)
    p2.add_argument("--cfg-embed", required=True)
    p2.add_argument("--cfg-default", required=True)
    p2.add_argument("--env-transnet", required=False, default=None)
    p2.add_argument("--env-ddm", required=False, default=None)
    p2.add_argument("--env-vlm", required=False, default=None)
    p2.add_argument("--env-embed", required=False, default=None)
    p2.add_argument("--force", action="store_true")

    return ap.parse_args()

def run_single(topic: str, sample_id: str, steps: List[str],
               cfg_shot: str, cfg_gebd: str, cfg_vlm: str, cfg_embed: str, cfg_default: str,
               env_transnet: Optional[str], env_ddm: Optional[str],
               env_vlm: Optional[str], env_embed: Optional[str],
               force: bool):
    steps = [s.strip().lower() for s in steps]
    if "detect" in steps:
        step_ref_detect(topic, sample_id, cfg_shot, cfg_gebd, env_transnet, env_ddm, force)
    if "vlm" in steps:
        step_ref_vlm(topic, sample_id, cfg_vlm, env_vlm, force)
    if "embed" in steps:
        step_ref_embed(sample_id, cfg_embed, env_vlm, env_embed, force)
    if "merge" in steps:
        step_ref_merge(sample_id, cfg_default, env_vlm, env_embed, force)

def batch_run(topics: List[str], steps: List[str],
              cfg_shot: str, cfg_gebd: str, cfg_vlm: str, cfg_embed: str, cfg_default: str,
              env_transnet: Optional[str], env_ddm: Optional[str],
              env_vlm: Optional[str], env_embed: Optional[str],
              force: bool):
    samples = discover_samples(topics)
    print(f"[batch] found {len(samples)} ref samples")
    for topic, sample_id in samples:
        print(f"\n=== REF RUN: topic={topic} sample={sample_id} ===")
        try:
            run_single(topic, sample_id, steps,
                       cfg_shot, cfg_gebd, cfg_vlm, cfg_embed, cfg_default,
                       env_transnet, env_ddm, env_vlm, env_embed,
                       force)
        except Exception as e:
            print(f"[ERROR] topic={topic} sample={sample_id} -> {e}")
            continue

def main():
    args = parse_args()
    if args.cmd == "run":
        steps = [s for s in args.steps.split(",") if s.strip()]
        run_single(args.topic, args.sample_id, steps,
                   args.cfg_shot, args.cfg_gebd, args.cfg_vlm, args.cfg_embed, args.cfg_default,
                   args.env_transnet, args.env_ddm, args.env_vlm, args.env_embed,
                   args.force)
    else:
        topics = [t for t in args.topics.split(",") if t.strip()]
        steps = [s for s in args.steps.split(",") if s.strip()]
        batch_run(topics, steps,
                  args.cfg_shot, args.cfg_gebd, args.cfg_vlm, args.cfg_embed, args.cfg_default,
                  args.env_transnet, args.env_ddm, args.env_vlm, args.env_embed,
                  args.force)

if __name__ == "__main__":
    main()
