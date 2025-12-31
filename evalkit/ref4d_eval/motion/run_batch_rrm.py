# tools/run_batch_rrm.py
# -*- coding: utf-8 -*-
"""
Batch RRM scorer (multi-process, cache reference-side per sample_id)
- Scans:
    ref : <base>/data/refvideo/<theme>/<sample_id>.mp4
    gen : <base>/data/genvideo/<modelname>/<sample_id>.mp4
- Groups by sample_id; same sample_id is assigned to the same worker
- Within a worker, reference side is computed ONCE and reused for all models

Outputs a single CSV with atomic metrics and diagnostics:
    modelname, sample_id, ref, gen,
    D_dir, D_mag, D_smo,
    RF, LS, median_s_ref, uniq_ratio_mean, uniq_ratio_min, uniq_ratio_max,
    FRZ, S_dir, S_mag, S_smo, S_motion, S_motion_w_frz, error
"""

from __future__ import annotations
import os, sys, csv, glob, time, argparse, traceback, hashlib
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from multiprocessing import Process

# ---- 限制CPU线程，避免和 TAPIR/GPU 抢资源 ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

# ---- 引入项目内模块 ----
from motion_eval.preprocess.io_video import load_video_cv2, resample_video
from motion_eval.rrm.subject_mask_rmm import build_fg_bg_masks_single, repair_masks_temporal
from motion_eval.rrm.seeds_rmm import sample_fg_bg_points_strict
from motion_eval.track_ate.tapir_infer import track_points_tapir
from motion_eval.rrm.rrm_features import compute_all as rrm_compute_all
from motion_eval.rrm.rrm_metrics import distances as rrm_distances, to_scores as rrm_to_scores, aggregate as rrm_aggregate
from motion_eval.rrm.rrm_freeze import compute_frz as rrm_compute_frz
import yaml


# ---------------------------- helpers ----------------------------

def _load_and_resample(path: str, short_side: int, fps: int):
    frames, fps_src = load_video_cv2(path, bgr=True, return_fps=True)
    frames = resample_video(frames, short_side=short_side, fps=fps, src_fps=fps_src)
    return frames

def _ensure_bool_list(lst):
    return [None if m is None else (m.astype(bool)) for m in lst]

def _sanitize_masks_for_sampling(masks_fg: List[Optional[np.ndarray]],
                                 masks_bg: List[Optional[np.ndarray]],
                                 border: int) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
    """
    最小入侵的预筛：若某帧掩码在“去掉 border 内缘”后内域为空，则把该帧置为 None（与单样本“跳过 None 帧”的语义一致）。
    仅批量脚本做这个预筛，避免 sample_points_strict 在 border 后直接报错。
    """
    def _inner_nonzero(m: Optional[np.ndarray]) -> bool:
        if m is None: return False
        H, W = m.shape[:2]
        if border <= 0: return bool(np.any(m))
        if H - 2*border <= 0 or W - 2*border <= 0:
            return False
        inner = m[border:H-border, border:W-border]
        return bool(np.any(inner))
    masks_fg2 = [m if _inner_nonzero(m) else None for m in masks_fg]
    masks_bg2 = [m if _inner_nonzero(m) else None for m in masks_bg]
    return masks_fg2, masks_bg2

def _run_tapir_grouped(frames: List[np.ndarray], seeds_xy: np.ndarray, seeds_t0: np.ndarray, cfg_tapir: dict):
    """
    与单样本版保持一致：
      - 预分配 zeros；vis=False
      - 按 t0 分组在子序列 frames[t0:] 跑 TAPIR
      - 右对齐回全局时间轴，并在 t0 写回初始点坐标，避免后续位移/phi 计算出现 NaN
    """
    T = len(frames)
    if seeds_xy.shape[0] != seeds_t0.shape[0]:
        raise ValueError("seeds_xy and seeds_t0 length mismatch")

    N = int(seeds_xy.shape[0])
    tracks = np.zeros((N, T, 2), np.float32)
    vis    = np.zeros((N, T), dtype=bool)

    if N == 0:
        return tracks, vis

    uniq_t0 = np.unique(seeds_t0.astype(int))
    for t in uniq_t0:
        idx = np.where(seeds_t0 == t)[0]
        if idx.size == 0:
            continue
        xy = seeds_xy[idx]            # [n_i,2]
        sub_frames = frames[t:]       # list of BGR frames
        tr_sub, vs_sub = track_points_tapir(sub_frames, xy, cfg_tapir)  # [n_i, T-t, 2], [n_i, T-t]
        tracks[idx, t:t+tr_sub.shape[1], :] = tr_sub
        vis[idx,    t:t+vs_sub.shape[1]   ] = vs_sub
        # 起始帧写回初始点坐标（与单样本一致）
        tracks[idx, t, :] = xy
    return tracks, vis

def _atomic_from_pair(ref_path: str, gen_path: str, cfg: dict, cache_ref: dict) -> Dict[str, float]:
    """
    计算一对 (ref, gen) 的原子项与诊断。
    cache_ref: 以 sample_id 为 key，缓存 {'frames', 'pack', 'masks_fg', 'masks_bg'}（单样本一致的参考端）
    """
    # --- cfg pieces ---
    ss  = int(cfg.get("sample", {}).get("short_side", 448))
    fps = int(cfg.get("sample", {}).get("fps", 12))

    cfg_subject = cfg.get("subject", cfg.get("cfg_subject", {})) or {}
    tr = cfg.get("tracking", {}) or {}
    num_fg = int(tr.get("num_fg", 128)); num_bg = int(tr.get("num_bg", 128))
    border = int(tr.get("border", 1)); edge_bonus = bool(tr.get("edge_bonus", True))
    min_tex = tr.get("min_tex", None); min_tex = float(min_tex) if min_tex is not None else None
    t0_stride = int(tr.get("t0_stride", 1))
    t0_offset = int(tr.get("t0_offset", 0))

    fallback_mask_cfg = (cfg.get("fallback", {}) or {}).get("mask", {}) or {}
    do_repair     = bool(fallback_mask_cfg.get("enable", False))
    rep_max_skip  = int(fallback_mask_cfg.get("max_skip", 6))
    rep_dilate_it = int(fallback_mask_cfg.get("dilate_iters", 1))
    rep_min_area  = int(fallback_mask_cfg.get("min_area_px", 0))

    cfg_tapir = cfg.get("tapir", {}) or {}
    dir_bins = int(cfg.get("features", {}).get("dir_bins", 16))
    eps = float(cfg.get("features", {}).get("eps", 1e-3))

    sc_lmb   = cfg.get("scoring", {}).get("lambdas", {"dir": 4.0, "mag": 2.0, "smo": 3.0})
    sc_alpha = cfg.get("scoring", {}).get("alphas",  {"dir": 0.35, "mag": 0.30, "smo": 0.35})
    sc_eta   = float(cfg.get("scoring", {}).get("eta", 0.6))

    frz_cfg = cfg.get("freeze", {}) or {}
    frz_kwargs = dict(
        tau_motion_abs=float(frz_cfg.get("tau_motion_abs", 0.0)),
        w_rf=float(frz_cfg.get("w_rf", 1.0)),
        w_ls=float(frz_cfg.get("w_ls", 1.0)),
        ncc_patch=int(frz_cfg.get("ncc_patch", 32)),
        ncc_stride=frz_cfg.get("ncc_stride", None),
        ncc_thr=float(frz_cfg.get("ncc_thr", 0.98)),
        unique_min_ratio=float(frz_cfg.get("unique_min_ratio", 0.20)),
        tau_s_quantile_ref=float(frz_cfg.get("tau_s_quantile_ref", 0.40)),
        roi_bbox_expand=int(frz_cfg.get("roi_bbox_expand", 0)),
        roi_on_empty=str(frz_cfg.get("roi_on_empty", "union")),
    )

    # ---------------- reference (cache by sample_id) ----------------
    sample_id = os.path.splitext(os.path.basename(ref_path))[0]
    if sample_id not in cache_ref:
        frames_ref = _load_and_resample(ref_path, ss, fps)
        masks_fg_ref, masks_bg_ref = build_fg_bg_masks_single(frames_ref, ref_path, cfg_subject)
        masks_fg_ref = _ensure_bool_list(masks_fg_ref); masks_bg_ref = _ensure_bool_list(masks_bg_ref)
        if do_repair:
            masks_fg_ref, masks_bg_ref, _ = repair_masks_temporal(
                masks_fg_ref, masks_bg_ref,
                max_skip=rep_max_skip, dilate_iters=rep_dilate_it, min_area_px=rep_min_area
            )
        # 批量专用：在撒点前做“内域为空”→ None 的预筛，避免边界裁切报错
        masks_fg_ref, masks_bg_ref = _sanitize_masks_for_sampling(masks_fg_ref, masks_bg_ref, border)

        # 参考端撒点（关键帧 t0_stride；跳过 None）
        seeds_ref = sample_fg_bg_points_strict(
            masks_fg_ref, masks_bg_ref,
            num_fg=num_fg, num_bg=num_bg,
            border=border, edge_bonus=edge_bonus,
            min_tex=min_tex, frames_bgr=frames_ref,
            t0_stride=t0_stride, t0_offset=t0_offset,
        )
        # 跟踪
        tracks_fg_ref, vis_fg_ref = _run_tapir_grouped(frames_ref, seeds_ref["fg_xy"], seeds_ref["fg_t0"], cfg_tapir)
        tracks_bg_ref, vis_bg_ref = _run_tapir_grouped(frames_ref, seeds_ref["bg_xy"], seeds_ref["bg_t0"], cfg_tapir)
        # 特征（带纹理权重）
        _r_ref, pack_ref = rrm_compute_all(
            tracks_fg_ref, vis_fg_ref, tracks_bg_ref, vis_bg_ref,
            dir_bins=dir_bins,
            wtex_fg=seeds_ref.get("fg_wtex", None),
            wtex_bg=seeds_ref.get("bg_wtex", None),
        )
        cache_ref[sample_id] = {
            "frames": frames_ref,
            "pack": pack_ref,
            "masks_fg": masks_fg_ref, "masks_bg": masks_bg_ref,
        }
    else:
        frames_ref = cache_ref[sample_id]["frames"]
        pack_ref   = cache_ref[sample_id]["pack"]

    # ---------------- generated ----------------
    frames_gen = _load_and_resample(gen_path, ss, fps)
    masks_fg_gen, masks_bg_gen = build_fg_bg_masks_single(frames_gen, gen_path, cfg_subject)
    masks_fg_gen = _ensure_bool_list(masks_fg_gen); masks_bg_gen = _ensure_bool_list(masks_bg_gen)
    if do_repair:
        masks_fg_gen, masks_bg_gen, _ = repair_masks_temporal(
            masks_fg_gen, masks_bg_gen,
            max_skip=rep_max_skip, dilate_iters=rep_dilate_it, min_area_px=rep_min_area
        )
    masks_fg_gen, masks_bg_gen = _sanitize_masks_for_sampling(masks_fg_gen, masks_bg_gen, border)

    seeds_gen = sample_fg_bg_points_strict(
        masks_fg_gen, masks_bg_gen,
        num_fg=num_fg, num_bg=num_bg,
        border=border, edge_bonus=edge_bonus,
        min_tex=min_tex, frames_bgr=frames_gen,
        t0_stride=t0_stride, t0_offset=t0_offset,
    )
    tracks_fg_gen, vis_fg_gen = _run_tapir_grouped(frames_gen, seeds_gen["fg_xy"], seeds_gen["fg_t0"], cfg_tapir)
    tracks_bg_gen, vis_bg_gen = _run_tapir_grouped(frames_gen, seeds_gen["bg_xy"], seeds_gen["bg_t0"], cfg_tapir)
    _r_gen, pack_gen = rrm_compute_all(
        tracks_fg_gen, vis_fg_gen, tracks_bg_gen, vis_bg_gen,
        dir_bins=dir_bins,
        wtex_fg=seeds_gen.get("fg_wtex", None),
        wtex_bg=seeds_gen.get("bg_wtex", None),
    )

    # ---------------- distances & scores ----------------
    D = rrm_distances(
        {"hof": pack_ref.hof, "s": pack_ref.s, "phi": pack_ref.phi},
        {"hof": pack_gen.hof, "s": pack_gen.s, "phi": pack_gen.phi},
        eps=eps
    )
    S = rrm_to_scores(D, sc_lmb)
    S_motion = float(rrm_aggregate(S, 0.0, sc_alpha, 0.0))  # 基础分（不衰减），便于对比

    # ---------------- freeze penalty (FRZ) ----------------
    frz, frz_diag = rrm_compute_frz(frames_ref, frames_gen, pack_ref.s, pack_gen.s, **frz_kwargs)

    # 叠上 FRZ 的衰减
    S_motion_w_frz = float(rrm_aggregate(S, frz, sc_alpha, sc_eta))

    out = {
        "D_dir": float(D["dir"]), "D_mag": float(D["mag"]), "D_smo": float(D["smo"]),
        "S_dir": float(S["dir"]), "S_mag": float(S["mag"]), "S_smo": float(S["smo"]),
        "FRZ": float(frz),
        "RF": float(frz_diag.get("RF", 0.0)),
        "LS": float(frz_diag.get("LS", 0.0)),
        "median_s_ref": float(frz_diag.get("median_s_ref", 0.0)),
        "uniq_ratio_mean": float(frz_diag.get("uniq_ratio_mean", 0.0)),
        "uniq_ratio_min": float(frz_diag.get("uniq_ratio_min", 0.0)),
        "uniq_ratio_max": float(frz_diag.get("uniq_ratio_max", 0.0)),
        "S_motion": float(S_motion),
        "S_motion_w_frz": float(S_motion_w_frz),
    }
    return out


def _scan_dataset(base: str):
    """返回
        sample_map: {sample_id: {"ref":ref_path, "theme":theme}}
        model_map:  {sample_id: [(modelname, gen_path), ...] }
    """
    ref_root = os.path.join(base, "data", "refvideo")
    gen_root = os.path.join(base, "data", "genvideo")

    # 参考
    sample_map: Dict[str, Dict[str,str]] = {}
    for theme_dir in sorted(glob.glob(os.path.join(ref_root, "*"))):
        if not os.path.isdir(theme_dir):
            continue
        theme = os.path.basename(theme_dir)
        for mp4 in sorted(glob.glob(os.path.join(theme_dir, "*.mp4"))):
            sid = os.path.splitext(os.path.basename(mp4))[0]
            sample_map[sid] = {"ref": mp4, "theme": theme}

    # 生成
    model_map: Dict[str, List[Tuple[str,str]]] = {sid: [] for sid in sample_map.keys()}
    for model_dir in sorted(glob.glob(os.path.join(gen_root, "*"))):
        if not os.path.isdir(model_dir):
            continue
        modelname = os.path.basename(model_dir)
        for sid in list(sample_map.keys()):
            gp = os.path.join(model_dir, f"{sid}.mp4")
            if os.path.isfile(gp):
                model_map[sid].append((modelname, gp))

    # 仅保留有生成视频的样本
    sample_map = {sid:info for sid,info in sample_map.items() if model_map.get(sid)}
    model_map  = {sid:lst  for sid,lst  in model_map.items()  if lst}
    return sample_map, model_map


def _worker_loop(wid: int,
                 tasks: List[str],
                 sample_map,
                 model_map,
                 cfg,
                 chunk_csv: str,
                 log_txt: str,
                 skip_pairs: Set[Tuple[str,str]]):
    cache_ref: Dict[str, dict] = {}
    os.makedirs(os.path.dirname(chunk_csv), exist_ok=True)
    with open(chunk_csv, "w", newline="", encoding="utf-8") as fw, open(log_txt, "w", encoding="utf-8") as flog:
        writer = csv.writer(fw)
        writer.writerow([
            "modelname","sample_id","ref","gen",
            "D_dir","D_mag","D_smo",
            "RF","LS","median_s_ref","uniq_ratio_mean","uniq_ratio_min","uniq_ratio_max",
            "FRZ","S_dir","S_mag","S_smo","S_motion","S_motion_w_frz","error"
        ])
        for sid in tasks:
            ref_path = sample_map[sid]["ref"]
            for modelname, gen_path in model_map[sid]:
                # --- 跳过命中缓存的 pair ---
                if (modelname, sid) in skip_pairs:
                    print(f"[worker {wid}] skip cached {modelname}/{sid}", file=flog, flush=True)
                    continue

                row = {
                    "modelname": modelname, "sample_id": sid,
                    "ref": ref_path, "gen": gen_path,
                    "error": "",
                }
                t0 = time.time()
                try:
                    out = _atomic_from_pair(ref_path, gen_path, cfg, cache_ref)
                    row.update(out)
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {e}"
                    traceback.print_exc(file=flog)
                writer.writerow([
                    row["modelname"], row["sample_id"], row["ref"], row["gen"],
                    row.get("D_dir",""), row.get("D_mag",""), row.get("D_smo",""),
                    row.get("RF",""), row.get("LS",""), row.get("median_s_ref",""),
                    row.get("uniq_ratio_mean",""), row.get("uniq_ratio_min",""), row.get("uniq_ratio_max",""),
                    row.get("FRZ",""), row.get("S_dir",""), row.get("S_mag",""), row.get("S_smo",""),
                    row.get("S_motion",""), row.get("S_motion_w_frz",""), row.get("error","")
                ])
                fw.flush()
                dt = time.time() - t0
                print(f"[worker {wid}] {modelname}/{sid} done in {dt:.1f}s", file=flog, flush=True)


def _idx(header: List[str], name: str, default: Optional[int] = None) -> int:
    """从表头里找列名的索引（大小写精确匹配）。找不到返回 default（若为 None 则抛错）。"""
    try:
        return header.index(name)
    except ValueError:
        if default is None:
            raise
        return default

def _read_existing_csv(path: str):
    """读取已有 CSV：返回 (header, rows, skip_pairs)。
       skip_pairs 基于 (modelname, sample_id) 两列，列位置从 header 中解析（若缺省，回退到 0/1）。"""
    header, rows, skip_pairs = [], [], set()
    with open(path, "r", encoding="utf-8") as fr:
        rdr = csv.reader(fr)
        header = next(rdr, None) or []
        try:
            i_model = _idx(header, "modelname", 0)
            i_sid   = _idx(header, "sample_id", 1)
        except Exception:
            # 回退：按前两列
            i_model, i_sid = 0, 1
        for row in rdr:
            if not row:
                continue
            rows.append(row)
            if len(row) > max(i_model, i_sid):
                skip_pairs.add((row[i_model], row[i_sid]))
    return header, rows, skip_pairs


def main():
    ap = argparse.ArgumentParser("Batch RRM scorer")
    ap.add_argument("--cfg", required=True, type=str)
    ap.add_argument("--base", required=True, type=str, help="repo base, e.g., /root/autodl-tmp/aigc_motion_eval")
    ap.add_argument("--out", required=True, type=str, help="final CSV path")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="覆盖重算；若不指定则复用已有CSV中的结果（含额外列）")
    args = ap.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sample_map, model_map = _scan_dataset(args.base)
    all_sids = sorted(sample_map.keys())
    if not all_sids:
        print("No samples found. Check ref/gen directories.", file=sys.stderr)
        sys.exit(1)

    # -------- 读取已有结果（用于跳过 & 合并 & 复用所有额外列）--------
    skip_pairs: Set[Tuple[str,str]] = set()
    existing_rows: List[List[str]] = []
    existing_header: List[str] = []
    if (not args.force) and os.path.isfile(args.out):
        try:
            existing_header, existing_rows, skip_pairs = _read_existing_csv(args.out)
            print(f"[info] cache found: {len(skip_pairs)} pairs in {args.out}")
        except Exception as e:
            print(f"[warn] failed to read existing out: {e}", file=sys.stderr)

    # =================== 新：先基于“跳过规则”统计总任务，再均衡分发 ===================
    # 仅保留“仍需计算”的 sample_id（至少有一个待算的 model/sample pair）
    sid_pending_counts: Dict[str, int] = {}
    pending_sids: List[str] = []
    total_pending_pairs = 0
    for sid in all_sids:
        # 统计该 sample_id 下尚未计算的模型对数量
        cnt = 0
        for modelname, _ in model_map.get(sid, []):
            if (modelname, sid) not in skip_pairs:
                cnt += 1
        if cnt > 0:
            sid_pending_counts[sid] = cnt
            pending_sids.append(sid)
            total_pending_pairs += cnt

    if not pending_sids:
        # 没有任何新任务，直接把旧结果写回并收尾
        std_header = [
            "modelname","sample_id","ref","gen",
            "D_dir","D_mag","D_smo",
            "RF","LS","median_s_ref","uniq_ratio_mean","uniq_ratio_min","uniq_ratio_max",
            "FRZ","S_dir","S_mag","S_smo","S_motion","S_motion_w_frz","error"
        ]
        header_out = existing_header[:] if (existing_header and (not args.force)) else std_header[:]
        with open(args.out, "w", newline="", encoding="utf-8") as fw:
            wr = csv.writer(fw)
            wr.writerow(header_out)
            for row in existing_rows:
                wr.writerow(row)
        print("[done] nothing to do. reused existing CSV.")
        return

    # 基于“预计工作量（待算 pair 数）”做贪心负载均衡，把 sample_id 分配到各 worker
    W = max(1, int(args.workers))
    buckets: List[List[str]] = [[] for _ in range(W)]
    loads: List[int] = [0 for _ in range(W)]
    # 大任务优先：按待算 pair 数从大到小分配，始终放到当前负载最小的 worker
    for sid in sorted(pending_sids, key=lambda s: sid_pending_counts[s], reverse=True):
        wid = min(range(W), key=lambda i: loads[i])
        buckets[wid].append(sid)
        loads[wid] += sid_pending_counts[sid]

    print(f"[plan] total pending pairs: {total_pending_pairs} over {len(pending_sids)} samples; workers={W}")
    for i in range(W):
        print(f"[plan] worker-{i}: samples={len(buckets[i])}, est_pairs={loads[i]}")

    # =================== 启动 workers（仅为非空桶启动进程） ===================
    procs: List[Process] = []
    chunk_paths = []
    for wid in range(W):
        if not buckets[wid]:
            continue  # 空桶无需启动
        chunk_csv = f"{os.path.splitext(args.out)[0]}.part{wid}.csv"
        log_txt   = f"{os.path.splitext(args.out)[0]}.part{wid}.log"
        chunk_paths.append(chunk_csv)
        p = Process(target=_worker_loop,
                    args=(wid, buckets[wid], sample_map, model_map, cfg, chunk_csv, log_txt, skip_pairs))
        p.daemon = False
        p.start()
        procs.append(p)

    # 等待
    for p in procs:
        p.join()

    # 合并 CSV：
    #   - 若存在旧表头且未 --force，则沿用旧表头（保留所有额外列：如 fli_*）
    #   - 否则使用标准表头
    std_header = [
        "modelname","sample_id","ref","gen",
        "D_dir","D_mag","D_smo",
        "RF","LS","median_s_ref","uniq_ratio_mean","uniq_ratio_min","uniq_ratio_max",
        "FRZ","S_dir","S_mag","S_smo","S_motion","S_motion_w_frz","error"
    ]
    if existing_header and (not args.force):
        header_out = existing_header[:]  # 完全复用旧表头（含附加列）
    else:
        header_out = std_header[:]

    # 记录已写入 pair，避免重复
    written_pairs: Set[Tuple[str,str]] = set()
    # 解析输出表头中的关键索引
    try:
        i_model_out = _idx(header_out, "modelname", 0)
        i_sid_out   = _idx(header_out, "sample_id", 1)
    except Exception:
        i_model_out, i_sid_out = 0, 1

    with open(args.out, "w", newline="", encoding="utf-8") as fw:
        writer = csv.writer(fw)
        writer.writerow(header_out)

        # 先写旧行（若非 force），保留所有附加列原样
        if existing_rows and (not args.force):
            try:
                i_model_old = _idx(header_out, "modelname", 0)
                i_sid_old   = _idx(header_out, "sample_id", 1)
            except Exception:
                i_model_old, i_sid_old = 0, 1
            for row in existing_rows:
                if not row: 
                    continue
                key = None
                if len(row) > max(i_model_old, i_sid_old):
                    key = (row[i_model_old], row[i_sid_old])
                if key and key not in written_pairs:
                    writer.writerow(row)
                    written_pairs.add(key)

        # 追加新分片（其表头是标准表头），需要按 header_out 映射并补齐缺失列（如 fli_*）
        out_cols_pos = {name: idx for idx, name in enumerate(header_out)}
        for part in chunk_paths:
            if not os.path.isfile(part):
                continue
            with open(part, "r", encoding="utf-8") as fr:
                rdr = csv.reader(fr)
                part_header = next(rdr, None) or []
                # 建立 part_header 名 -> idx
                part_pos = {name: idx for idx, name in enumerate(part_header)}
                # 定位关键列以去重
                try:
                    i_model_part = _idx(part_header, "modelname", 0)
                    i_sid_part   = _idx(part_header, "sample_id", 1)
                except Exception:
                    i_model_part, i_sid_part = 0, 1

                for row in rdr:
                    if not row:
                        continue
                    if len(row) <= max(i_model_part, i_sid_part):
                        continue
                    key = (row[i_model_part], row[i_sid_part])
                    if key in written_pairs:
                        continue
                    # 生成与 header_out 对齐的新行：存在的列取值，不存在的列填空
                    row_out = []
                    for col in header_out:
                        if col in part_pos and part_pos[col] < len(row):
                            row_out.append(row[part_pos[col]])
                        else:
                            row_out.append("")  # 例如 fli_* 等附加列在分片中不存在 → 留空
                    writer.writerow(row_out)
                    written_pairs.add(key)

    print(f"[done] merged CSV -> {args.out}")


if __name__ == "__main__":
    main()
