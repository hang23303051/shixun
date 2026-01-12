# ref4d_eval/motion/rrm/precompute_ref_motion.py
# -*- coding: utf-8 -*-
"""
离线预计算参考侧 RRM 特征（hof/s/phi） + FRZ 参考侧元信息 ref_meta

使用场景：
  - 仅在有 refvideo 的内部环境跑一次；
  - 针对 dataset.format=legacy：使用 data/refvideo/<theme>/<sid>.mp4；
  - 输出 npz 到 data/metadata/motion_ref/rrm_448x12/<sid>.npz，包含：
        hof, s, phi, frz_meta

开源评测时：
  - 无需 refvideo，只需加载这些 npz + 生成视频，即可准确计算：
        D_* / S_* / S_motion / FRZ 以及 RF/LS/median_s_ref/uniq_ratio_*
"""

from __future__ import annotations
import os
import glob
import argparse
from typing import Dict, List, Tuple, Optional, Any

import multiprocessing as mp
import traceback  # 用于打印子任务异常

import numpy as np
import yaml

# ---- 限制 CPU 线程，避免和 TAPIR/GPU 抢资源（与 run_batch_rrm 保持一致）----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

from ..preprocess.io_video import load_video_cv2, resample_video
from .subject_mask_rmm import build_fg_bg_masks_single, repair_masks_temporal
from .seeds_rmm import sample_fg_bg_points_strict
from ..track_ate.tapir_infer import track_points_tapir
from .rrm_features import compute_all as rrm_compute_all
from .rrm_freeze import compute_frz_refside


def _load_and_resample(path: str, short_side: int, fps: int):
    frames, fps_src = load_video_cv2(path, bgr=True, return_fps=True)
    frames = resample_video(frames, short_side=short_side, fps=fps, src_fps=fps_src)
    return frames


def _ensure_bool_list(lst):
    return [None if m is None else (m.astype(bool)) for m in lst]


def _sanitize_masks_for_sampling(
    masks_fg: List[Optional[np.ndarray]],
    masks_bg: List[Optional[np.ndarray]],
    border: int,
):
    """
    最小入侵的预筛：若某帧掩码在“去掉 border 内缘”后内域为空，则把该帧置为 None。
    与批量 RRM 中的逻辑保持一致，避免撒点时因边界裁切报错。
    """
    def _inner_nonzero(m: Optional[np.ndarray]) -> bool:
        if m is None:
            return False
        H, W = m.shape[:2]
        if border <= 0:
            return bool(np.any(m))
        if H - 2 * border <= 0 or W - 2 * border <= 0:
            return False
        inner = m[border:H - border, border:W - border]
        return bool(np.any(inner))

    masks_fg2 = [m if _inner_nonzero(m) else None for m in masks_fg]
    masks_bg2 = [m if _inner_nonzero(m) else None for m in masks_bg]
    return masks_fg2, masks_bg2


def _run_tapir_grouped(
    frames: List[np.ndarray],
    seeds_xy: np.ndarray,
    seeds_t0: np.ndarray,
    cfg_tapir: dict,
):
    """
    与 run_batch_rrm.py 中保持一致：
      - 预分配 zeros；vis=False
      - 按 t0 分组在子序列 frames[t0:] 上跑 TAPIR
      - 右对齐回全局时间轴，并在 t0 写回初始点坐标
    """
    T = len(frames)
    if seeds_xy.shape[0] != seeds_t0.shape[0]:
        raise ValueError("seeds_xy and seeds_t0 length mismatch")

    N = int(seeds_xy.shape[0])
    tracks = np.zeros((N, T, 2), np.float32)
    vis = np.zeros((N, T), dtype=bool)

    if N == 0:
        return tracks, vis

    uniq_t0 = np.unique(seeds_t0.astype(int))
    for t in uniq_t0:
        idx = np.where(seeds_t0 == t)[0]
        if idx.size == 0:
            continue
        xy = seeds_xy[idx]
        sub_frames = frames[t:]
        tr_sub, vs_sub = track_points_tapir(sub_frames, xy, cfg_tapir)
        tracks[idx, t:t + tr_sub.shape[1], :] = tr_sub
        vis[idx, t:t + vs_sub.shape[1]] = vs_sub
        tracks[idx, t, :] = xy
    return tracks, vis


def _scan_ref_legacy(base: str) -> Dict[str, str]:
    """
    扫描参考视频，支持多种布局：
      1) data/refvideo/*.mp4
      2) data/refvideo/<theme>/*.mp4
      3) refvideo/*.mp4
      4) refvideo/<theme>/*.mp4

    返回：{sample_id: ref_path}，其中 sample_id = 去掉扩展名的文件名。
    """
    sid2ref: Dict[str, str] = {}

    # 兼容两种根目录
    cand_roots = [
        os.path.join(base, "data", "refvideo"),
        os.path.join(base, "refvideo"),
    ]

    for ref_root in cand_roots:
        if not os.path.isdir(ref_root):
            continue

        # 1) 根目录下直接是 mp4
        for mp4 in sorted(glob.glob(os.path.join(ref_root, "*.mp4"))):
            sid = os.path.splitext(os.path.basename(mp4))[0]
            # 若同名 sid 已存在，保持第一次出现的路径（避免覆盖）
            sid2ref.setdefault(sid, mp4)

        # 2) 子目录（当作 theme）
        for theme_dir in sorted(glob.glob(os.path.join(ref_root, "*"))):
            if not os.path.isdir(theme_dir):
                continue
            for mp4 in sorted(glob.glob(os.path.join(theme_dir, "*.mp4"))):
                sid = os.path.splitext(os.path.basename(mp4))[0]
                sid2ref.setdefault(sid, mp4)

    print(f"[scan_ref] total refs found: {len(sid2ref)}")
    return sid2ref


# ---------------- 单个 ref 处理函数（便于多进程复用） ----------------

def _process_one_ref(args: Tuple[Any, ...]) -> str:
    """
    处理单个参考视频 sid，方便在主进程或子进程中复用。
    args 解包顺序请与 main() 中构造保持一致。

    约定：无论成功与否，都返回 sid；所有异常在本函数内部被捕获并打印，
    避免在多进程 imap_unordered 中把异常抛回主进程。
    """
    (
        idx, total,
        sid, ref_path, out_root,
        ss, fps,
        cfg_subject,
        num_fg, num_bg, border, edge_bonus, min_tex,
        t0_stride, t0_offset,
        do_repair, rep_max_skip, rep_dilate_it, rep_min_area,
        cfg_tapir,
        dir_bins,
        frz_cfg,
    ) = args

    try:
        out_path = os.path.join(out_root, f"{sid}.npz")
        if os.path.isfile(out_path):
            print(f"[{idx}/{total}] skip {sid}, exists")
            return sid

        print(f"[{idx}/{total}] {sid} -> {out_path}")

        # 1) 读参考视频并重采样
        frames_ref = _load_and_resample(ref_path, ss, fps)

        # 2) 构造掩码（语义证据来自 subject_mask_rmm 里 _read_semantics_for）
        masks_fg_ref, masks_bg_ref = build_fg_bg_masks_single(frames_ref, ref_path, cfg_subject)
        masks_fg_ref = _ensure_bool_list(masks_fg_ref)
        masks_bg_ref = _ensure_bool_list(masks_bg_ref)
        if do_repair:
            masks_fg_ref, masks_bg_ref, _ = repair_masks_temporal(
                masks_fg_ref,
                masks_bg_ref,
                max_skip=rep_max_skip,
                dilate_iters=rep_dilate_it,
                min_area_px=rep_min_area,
            )
        masks_fg_ref, masks_bg_ref = _sanitize_masks_for_sampling(masks_fg_ref, masks_bg_ref, border)

        # 3) 撒点（注意：这里可能抛 ValueError: no seeds after ...）
        seeds_ref = sample_fg_bg_points_strict(
            masks_fg_ref,
            masks_bg_ref,
            num_fg=num_fg,
            num_bg=num_bg,
            border=border,
            edge_bonus=edge_bonus,
            min_tex=min_tex,
            frames_bgr=frames_ref,
            t0_stride=t0_stride,
            t0_offset=t0_offset,
        )

        # 4) TAPIR 轨迹
        tracks_fg_ref, vis_fg_ref = _run_tapir_grouped(
            frames_ref, seeds_ref["fg_xy"], seeds_ref["fg_t0"], cfg_tapir
        )
        tracks_bg_ref, vis_bg_ref = _run_tapir_grouped(
            frames_ref, seeds_ref["bg_xy"], seeds_ref["bg_t0"], cfg_tapir
        )

        # 5) RRM 特征（hof/s/phi）
        _, pack_ref = rrm_compute_all(
            tracks_fg_ref,
            vis_fg_ref,
            tracks_bg_ref,
            vis_bg_ref,
            dir_bins=dir_bins,
            wtex_fg=seeds_ref.get("fg_wtex", None),
            wtex_bg=seeds_ref.get("bg_wtex", None),
        )

        # 6) FRZ 参考侧元信息：只依赖 s_ref
        frz_meta = compute_frz_refside(
            pack_ref.s,
            tau_motion_abs=float(frz_cfg.get("tau_motion_abs", 0.0)),
            tau_s_quantile_ref=float(frz_cfg.get("tau_s_quantile_ref", 0.40)),
        )

        # 7) 保存到 npz：hof/s/phi + frz_meta
        np.savez(
            out_path,
            hof=pack_ref.hof,
            s=pack_ref.s,
            phi=pack_ref.phi,
            frz_meta=np.array(frz_meta, dtype=object),
        )

        return sid

    except Exception as e:
        # 与 run_batch_rrm 的 worker 行为对齐：单样本失败只记录错误，不中断整体
        print(f"[{idx}/{total}] ERROR {sid}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        # 不生成 npz，直接返回 sid，让主进程继续其它样本
        return sid


def main():
    ap = argparse.ArgumentParser("Precompute reference-side RRM features + FRZ ref_meta")
    ap.add_argument("--cfg", required=True, type=str)
    ap.add_argument("--base", required=True, type=str,
                    help="repo base (Ref4D-VideoBench)")
    ap.add_argument("--out-root", required=True, type=str,
                    help="output dir for npz, e.g. data/metadata/motion_ref/rrm_448x12")
    ap.add_argument("--limit", type=int, default=0,
                    help="optional: only process first N samples")
    ap.add_argument("--workers", type=int, default=1,
                    help="number of worker processes; 1 = 单进程，<=0 = 使用 CPU 核数")
    args = ap.parse_args()

    base_dir = os.path.abspath(args.base)
    out_root = os.path.join(base_dir, args.out_root)
    os.makedirs(out_root, exist_ok=True)

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sample_cfg = cfg.get("sample", {}) or {}
    ss = int(sample_cfg.get("short_side", 448))
    fps = int(sample_cfg.get("fps", 8))

    cfg_subject = cfg.get("subject", cfg.get("cfg_subject", {})) or {}
    tr = cfg.get("tracking", {}) or {}
    num_fg = int(tr.get("num_fg", 128))
    num_bg = int(tr.get("num_bg", 128))
    border = int(tr.get("border", 1))
    edge_bonus = bool(tr.get("edge_bonus", True))
    min_tex = tr.get("min_tex", None)
    min_tex = float(min_tex) if min_tex is not None else None
    t0_stride = int(tr.get("t0_stride", 1))
    t0_offset = int(tr.get("t0_offset", 0))

    fallback_mask_cfg = (cfg.get("fallback", {}) or {}).get("mask", {}) or {}
    do_repair = bool(fallback_mask_cfg.get("enable", False))
    rep_max_skip = int(fallback_mask_cfg.get("max_skip", 6))
    rep_dilate_it = int(fallback_mask_cfg.get("dilate_iters", 1))
    rep_min_area = int(fallback_mask_cfg.get("min_area_px", 0))

    cfg_tapir = cfg.get("tapir", {}) or {}
    dir_bins = int(cfg.get("features", {}).get("dir_bins", 16))

    frz_cfg = cfg.get("freeze", {}) or {}

    sid2ref = _scan_ref_legacy(base_dir)
    sids = sorted(sid2ref.keys())
    if args.limit > 0:
        sids = sids[:args.limit]

    total = len(sids)
    print(f"[precompute] total refs: {total}, out_root={out_root}")

    # 构造任务列表（把所有必要配置打包到 args 里，方便多进程调用）
    jobs: List[Tuple[Any, ...]] = []
    for i, sid in enumerate(sids, start=1):
        ref_path = sid2ref[sid]
        jobs.append(
            (
                i, total,
                sid, ref_path, out_root,
                ss, fps,
                cfg_subject,
                num_fg, num_bg, border, edge_bonus, min_tex,
                t0_stride, t0_offset,
                do_repair, rep_max_skip, rep_dilate_it, rep_min_area,
                cfg_tapir,
                dir_bins,
                frz_cfg,
            )
        )

    # workers 处理
    workers = int(args.workers)
    if workers <= 0:
        workers = max(1, os.cpu_count() or 1)

    if workers == 1:
        # 单进程：顺序处理，行为不变（异常在 _process_one_ref 内部就被处理了）
        for job in jobs:
            _process_one_ref(job)
    else:
        print(f"[precompute] use multiprocessing with workers={workers}")
        # 使用 spawn 更安全（尤其是和 torch/cuda 共存时）
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for _sid_done in pool.imap_unordered(_process_one_ref, jobs):
                # 这里不需要处理返回值；_process_one_ref 内部已经处理好成功/失败情况
                pass

    print("[precompute] done.")


if __name__ == "__main__":
    main()
