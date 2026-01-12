# -*- coding: utf-8 -*-
"""
Alignment-Free RRM - CLI
端到端：读取 → 掩码 → 撒点（严格，无回退） → TAPIR → r(t) → 特征 → 距离/打分 → FRZ → 导出

严格遵循：
- 不做时空对齐/单应/镜头切分
- 不做回退（掩码/采样不足等直接抛异常）
"""

from __future__ import annotations
import os, json, argparse
from typing import Dict, Any, List
import numpy as np
import yaml

from .preprocess.io_video import load_video_cv2, resample_video
from .rrm.subject_mask_rmm import build_fg_bg_masks_single
from .rrm.seeds_rmm import sample_fg_bg_points_strict
from .track_ate.tapir_infer import track_points_tapir
from .rrm.rrm_features import compute_all as rrm_compute_all
from .rrm.rrm_metrics import distances as rrm_distances, to_scores as rrm_to_scores, aggregate as rrm_aggregate
from .rrm.rrm_freeze import compute_frz as rrm_compute_frz


def _load_and_resample(path: str, short_side: int, fps: int):
    frames, fps_src = load_video_cv2(path, bgr=True, return_fps=True)
    frames = resample_video(frames, short_side=short_side, fps=fps, src_fps=fps_src)
    return frames


def _ensure_bool_list(lst: List[np.ndarray]) -> List[np.ndarray | None]:
    return [None if (m is None) else (m.astype(bool)) for m in lst]


def _run_tapir_grouped(frames_bgr: List[np.ndarray],
                       seeds_xy: np.ndarray,
                       seeds_t0: np.ndarray,
                       cfg_tapir: Dict[str, Any]):
    """
    按每个点的 t0 将查询分组：每组在 frames[t0:] 上独立跑 TAPIR，
    再把轨迹右对齐到整段时间轴（t < t0 处用起点占位；可见性 False）。
    """
    T = len(frames_bgr)
    if seeds_xy.shape[0] != seeds_t0.shape[0]:
        raise ValueError("seeds_xy and seeds_t0 length mismatch")

    # 预分配
    N = seeds_xy.shape[0]
    tracks = np.zeros((N, T, 2), np.float32)
    vis = np.zeros((N, T), bool)

    # 按 t0 分组
    uniq_t0 = np.unique(seeds_t0)
    offset = 0
    for t in uniq_t0:
        idx = np.where(seeds_t0 == t)[0]
        xy = seeds_xy[idx]                       # [n_i, 2]
        sub = frames_bgr[t:]                     # 从 t 帧开始的子序列
        # 在子序列上跟踪
        tr_sub, vs_sub = track_points_tapir(sub, xy, cfg_tapir)  # [n_i, T-t, 2], [n_i, T-t]
        # 对齐到全局时间轴
        tracks[idx, t:, :] = tr_sub
        vis[idx, t:] = vs_sub
        # 起始帧坐标填充（保持语义完整）
        tracks[idx, t, :] = xy

    return tracks, vis


def main():
    ap = argparse.ArgumentParser("Alignment-Free RRM CLI")
    ap.add_argument("--ref", required=True, type=str, help="reference video path")
    ap.add_argument("--gen", required=True, type=str, help="generated video path")
    ap.add_argument("--cfg", required=True, type=str, help="YAML config for RRM")
    ap.add_argument("--out", required=True, type=str, help="output JSON path")
    args = ap.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as h:
        cfg = yaml.safe_load(h)

    # ---------- 1) 读取/统一采样 ----------
    ss = int(cfg.get("sample", {}).get("short_side", 448))
    fps = int(cfg.get("sample", {}).get("fps", ))
    frames_ref = _load_and_resample(args.ref, ss, fps)
    frames_gen = _load_and_resample(args.gen, ss, fps)

    # ---------- 2) 掩码（单端，不做对齐） ----------
    cfg_subject = cfg.get("subject", cfg.get("cfg_subject", {})) or {}
    masks_fg_ref, masks_bg_ref = build_fg_bg_masks_single(frames_ref, args.ref, cfg_subject)  # may raise
    masks_fg_gen, masks_bg_gen = build_fg_bg_masks_single(frames_gen, args.gen, cfg_subject)  # may raise
    masks_fg_ref = _ensure_bool_list(masks_fg_ref)
    masks_bg_ref = _ensure_bool_list(masks_bg_ref)
    masks_fg_gen = _ensure_bool_list(masks_fg_gen)
    masks_bg_gen = _ensure_bool_list(masks_bg_gen)

    # ---------- 3) 撒点（严格无回退） ----------
    tr = cfg.get("tracking", {})
    num_fg = int(tr.get("num_fg", 128))
    num_bg = int(tr.get("num_bg", 256))
    border = int(tr.get("border", 4))
    edge_bonus = bool(tr.get("edge_bonus", True))
    min_tex = tr.get("min_tex", None)
    if min_tex is not None:
        min_tex = float(min_tex)
        
    # 新增：每 K 帧一个起点
    t0_stride = int(tr.get("t0_stride", 1))   # K
    t0_offset = int(tr.get("t0_offset", 0))   # 起点偏移，可为 0..K-1
    seeds_ref = sample_fg_bg_points_strict(
        masks_fg_ref, masks_bg_ref,
        num_fg=num_fg, num_bg=num_bg,
        border=border, edge_bonus=edge_bonus,
        min_tex=min_tex, frames_bgr=frames_ref,
        t0_stride=t0_stride, t0_offset=t0_offset,
    )
    seeds_gen = sample_fg_bg_points_strict(
        masks_fg_gen, masks_bg_gen,
        num_fg=num_fg, num_bg=num_bg,
        border=border, edge_bonus=edge_bonus,
        min_tex=min_tex, frames_bgr=frames_gen,
        t0_stride=t0_stride, t0_offset=t0_offset,
    )


    # ---------- 4) TAPIR 跟踪（按 t0 分组） ----------
    cfg_tapir = cfg.get("tapir", {}) or {}
    tracks_fg_ref, vis_fg_ref = _run_tapir_grouped(frames_ref, seeds_ref["fg_xy"], seeds_ref["fg_t0"], cfg_tapir)
    tracks_bg_ref, vis_bg_ref = _run_tapir_grouped(frames_ref, seeds_ref["bg_xy"], seeds_ref["bg_t0"], cfg_tapir)
    tracks_fg_gen, vis_fg_gen = _run_tapir_grouped(frames_gen, seeds_gen["fg_xy"], seeds_gen["fg_t0"], cfg_tapir)
    tracks_bg_gen, vis_bg_gen = _run_tapir_grouped(frames_gen, seeds_gen["bg_xy"], seeds_gen["bg_t0"], cfg_tapir)

    # ---------- 5) RRM 特征 ----------
    dir_bins = int(cfg.get("features", {}).get("dir_bins", 16))
    wtex_fg_ref = seeds_ref.get("fg_wtex", None)
    wtex_bg_ref = seeds_ref.get("bg_wtex", None)
    wtex_fg_gen = seeds_gen.get("fg_wtex", None)
    wtex_bg_gen = seeds_gen.get("bg_wtex", None)
    _r_ref, pack_ref = rrm_compute_all(tracks_fg_ref, vis_fg_ref, tracks_bg_ref, vis_bg_ref,
                                       dir_bins=dir_bins,wtex_fg=wtex_fg_ref,wtex_bg=wtex_bg_ref)
    _r_gen, pack_gen = rrm_compute_all(tracks_fg_gen, vis_fg_gen, tracks_bg_gen, vis_bg_gen,
                                       dir_bins=dir_bins,wtex_fg=wtex_fg_gen,wtex_bg=wtex_bg_gen)

    # ---------- 6) 指标距离 / 分数 ----------
    D = rrm_distances({"hof": pack_ref.hof, "s": pack_ref.s, "phi": pack_ref.phi},
                      {"hof": pack_gen.hof, "s": pack_gen.s, "phi": pack_gen.phi},
                      eps=float(cfg.get("features", {}).get("eps", 1e-3)))
    S = rrm_to_scores(D, cfg.get("scoring", {}).get("lambdas", {"dir": 4.0, "mag": 2.0, "smo": 3.0}))

    # ---------- 7) 冻结惩罚（仅 RF+LS） ----------
    frz_cfg = cfg.get("freeze", {}) or {}
    frz, frz_diag = rrm_compute_frz(
        frames_ref, frames_gen, pack_ref.s, pack_gen.s,
        tau_motion_abs=float(frz_cfg.get("tau_motion_abs", 0.0)),
        w_rf=float(frz_cfg.get("w_rf", 1.0)),
        w_ls=float(frz_cfg.get("w_ls", 1.0)),
        ncc_patch=int(frz_cfg.get("ncc_patch", 32)),
        ncc_stride=frz_cfg.get("ncc_stride", None),
        ncc_thr=float(frz_cfg.get("ncc_thr", 0.98)),
        unique_min_ratio=float(frz_cfg.get("unique_min_ratio", 0.20)),
        tau_s_quantile_ref=float(frz_cfg.get("tau_s_quantile_ref", 0.40)),
        roi_border=int(frz_cfg.get("roi_border", 2)),
    )

    # ---------- 8) 融合为总分 ----------
    alphas = cfg.get("scoring", {}).get("alphas", {"dir": 0.35, "mag": 0.30, "smo": 0.35})
    eta = float(cfg.get("scoring", {}).get("eta", 0.6))
    S_motion = rrm_aggregate(S, frz, alphas, eta)

    # ---------- 9) 导出 ----------
    out = {
        "D": D,
        "S": S,
        "FRZ": frz,
        "FRZ_diag": frz_diag,
        "S_motion": float(S_motion),
        "meta": {
            "ref": os.path.abspath(args.ref),
            "gen": os.path.abspath(args.gen),
            "short_side": ss,
            "fps": fps,
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as h:
        json.dump(out, h, ensure_ascii=False, indent=2)

    print(json.dumps({"S_motion": S_motion, "FRZ": frz}, ensure_ascii=False))


if __name__ == "__main__":
    main()
