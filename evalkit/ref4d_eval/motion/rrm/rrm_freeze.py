# -*- coding: utf-8 -*-
"""
Alignment-Free RRM - Freeze penalty (FRZ)

实现两项：
- RF（重复率）：基于相邻帧局部 ZNCC 的“独特像素率不足帧比例”
- LS（低速占比）：生成端 s_gen 低于参考端阈值 tau_s 的时间占比

启用条件：
- 仅当 median(s_ref) > tau_motion_abs 时启用 FRZ；否则 FRZ=0

兼容性：
- compute_frz / compute_rf_ls 均兼容别名 roi_masks（自动映射为 roi_masks_gen）
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2 as cv

__all__ = ["rf_unique_ratio_ncc", "compute_rf_ls", "compute_frz"]


# -------------------- 小工具 --------------------

def _to_gray_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[-1] == 3:
        g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    elif img.ndim == 2:
        g = img
    else:
        raise ValueError("unexpected frame shape")
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8, copy=False)
    return g


def _ncc_patch(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    A = a.astype(np.float32); B = b.astype(np.float32)
    Am = A - A.mean(); Bm = B - B.mean()
    num = float((Am * Bm).sum())
    den = float(np.sqrt((Am * Am).sum()) * np.sqrt((Bm * Bm).sum()) + eps)
    return num / den


def _bbox_from_mask(m: Optional[np.ndarray]) -> Optional[Tuple[int, int, int, int]]:
    if m is None:
        return None
    if m.dtype != bool:
        m = m.astype(bool)
    ys, xs = np.where(m)
    if ys.size == 0:
        return None
    # y0,y1,x0,x1  (y1/x1 为开区间)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _intersect_bbox(a: Optional[Tuple[int, int, int, int]],
                    b: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
    if a is None or b is None:
        return None
    ay0, ay1, ax0, ax1 = a
    by0, by1, bx0, bx1 = b
    y0, y1 = max(ay0, by0), min(ay1, by1)
    x0, x1 = max(ax0, bx0), min(ax1, bx1)
    if y1 <= y0 or x1 <= x0:
        return None
    return y0, y1, x0, x1


def _union_bbox(a: Optional[Tuple[int, int, int, int]],
                b: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    ay0, ay1, ax0, ax1 = a
    by0, by1, bx0, bx1 = b
    return min(ay0, by0), max(ay1, by1), min(ax0, bx0), max(ax1, bx1)


def _expand_bbox(bb: Tuple[int, int, int, int], pad: int, H: int, W: int) -> Tuple[int, int, int, int]:
    if pad <= 0:
        return bb
    y0, y1, x0, x1 = bb
    return max(0, y0 - pad), min(H, y1 + pad), max(0, x0 - pad), min(W, x1 + pad)


# -------------------- RF（支持 ROI） --------------------

def rf_unique_ratio_ncc(
    frames_bgr: List[np.ndarray],
    *,
    patch: int = 32,
    stride: Optional[int] = None,
    ncc_thr: float = 0.98,
    unique_min_ratio: float = 0.20,
    roi_masks: Optional[List[Optional[np.ndarray]]] = None,
    roi_bbox_expand: int = 0,
    roi_on_empty: str = "union",  # "union" | "use_curr" | "use_prev" | "full" | "skip"
) -> Tuple[float, List[float]]:
    """
    RF：独特像素率不足帧比例。
      对每帧 t>0：unique_ratio(t) = mean( NCC < ncc_thr )；
      RF = mean( unique_ratio(t) < unique_min_ratio ).

    若提供 roi_masks（与 frames 对齐的逐帧掩码），只在选定 ROI 内统计：
      - 首选相邻帧 ROI 交集；为空时按 roi_on_empty 处理；
      - ROI 太小时，在 ROI 中心放一个 patch（裁剪到图内）保证有样本。
    """
    if len(frames_bgr) < 2:
        raise ValueError("need at least 2 frames for RF")
    if roi_masks is not None and len(roi_masks) != len(frames_bgr):
        raise ValueError("len(roi_masks) must equal len(frames_bgr)")

    patch = int(patch)
    if patch <= 0:
        raise ValueError("patch must be positive")
    stride = int(stride or patch)

    uniq_ratios: List[float] = []

    for t in range(1, len(frames_bgr)):
        g0 = _to_gray_uint8(frames_bgr[t - 1])
        g1 = _to_gray_uint8(frames_bgr[t])
        H, W = g0.shape
        if g1.shape != (H, W):
            raise ValueError("frame size mismatch")

        # 选择 ROI
        if roi_masks is not None:
            bb0 = _bbox_from_mask(roi_masks[t - 1])
            bb1 = _bbox_from_mask(roi_masks[t])
            bb = _intersect_bbox(bb0, bb1)
            if bb is None:
                mode = (roi_on_empty or "union").lower()
                if mode == "union":
                    bb = _union_bbox(bb0, bb1)
                elif mode == "use_curr":
                    bb = bb1 if bb1 is not None else bb0
                elif mode == "use_prev":
                    bb = bb0 if bb0 is not None else bb1
                elif mode == "full":
                    bb = (0, H, 0, W)
                elif mode == "skip":
                    # 跳过该帧对
                    continue
                else:
                    raise ValueError(f"invalid roi_on_empty: {roi_on_empty}")
                if bb is None:
                    # union/use_* 仍为空，则退回全帧（确保可采样）
                    bb = (0, H, 0, W)
            y0, y1, x0, x1 = _expand_bbox(bb, int(roi_bbox_expand), H, W)
        else:
            y0, y1, x0, x1 = 0, H, 0, W

        scores = []
        if (y1 - y0) >= patch and (x1 - x0) >= patch:
            for y in range(y0, y1 - patch + 1, stride):
                for x in range(x0, x1 - patch + 1, stride):
                    a = g0[y:y + patch, x:x + patch]
                    b = g1[y:y + patch, x:x + patch]
                    scores.append(_ncc_patch(a, b))
        else:
            # ROI 太小：在 ROI 中心放 1 个 patch（裁剪到图内）
            if H < patch or W < patch:
                raise ValueError(f"frame smaller than patch at t={t}: frame=({H},{W}), patch={patch}")
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            y = int(np.clip(cy - patch // 2, 0, max(0, H - patch)))
            x = int(np.clip(cx - patch // 2, 0, max(0, W - patch)))
            a = g0[y:y + patch, x:x + patch]
            b = g1[y:y + patch, x:x + patch]
            scores.append(_ncc_patch(a, b))

        if not scores:
            # 可能因为连续 skip 导致无样本
            raise ValueError(f"no patches for NCC at t={t} (check patch/stride/ROI size)")

        unique_ratio = float(np.mean(np.asarray(scores, np.float32) < float(ncc_thr)))
        uniq_ratios.append(unique_ratio)

    if not uniq_ratios:
        raise ValueError("no valid frame pairs for RF (all skipped or empty ROI)")

    rf = float(np.mean(np.asarray(uniq_ratios) < float(unique_min_ratio)))
    return rf, uniq_ratios


# -------------------- RF + LS --------------------

def compute_rf_ls(
    frames_ref: List[np.ndarray],
    frames_gen: List[np.ndarray],
    s_ref: np.ndarray,
    s_gen: np.ndarray,
    *,
    tau_s_quantile_ref: float = 0.40,
    ncc_patch: int = 32,
    ncc_stride: Optional[int] = None,
    ncc_thr: float = 0.98,
    unique_min_ratio: float = 0.20,
    roi_masks_gen: Optional[List[Optional[np.ndarray]]] = None,
    roi_bbox_expand: int = 0,
    roi_on_empty: str = "union",
    **kwargs,  # 兼容别名 roi_masks
) -> Tuple[float, float, Dict[str, float]]:
    """
    计算 RF 与 LS（不含门控/融合）：
      - RF：基于生成端 frames_gen 的局部 NCC 独特像素率（可选 ROI）
      - LS：s_gen < tau_s 的时间占比；tau_s = P_q(s_ref)
    返回：(RF, LS, 诊断信息)
    """
    if len(frames_gen) == 0 or s_gen.size == 0 or s_ref.size == 0:
        raise ValueError("empty inputs for RF/LS")

    # 兼容别名：roi_masks -> roi_masks_gen
    if roi_masks_gen is None and "roi_masks" in kwargs:
        roi_masks_gen = kwargs.pop("roi_masks")

    rf, uniq_ratios = rf_unique_ratio_ncc(
        frames_gen,
        patch=ncc_patch,
        stride=ncc_stride,
        ncc_thr=ncc_thr,
        unique_min_ratio=unique_min_ratio,
        roi_masks=roi_masks_gen,
        roi_bbox_expand=roi_bbox_expand,
        roi_on_empty=roi_on_empty,
    )

    tau_s = float(np.quantile(s_ref.astype(np.float64), float(tau_s_quantile_ref)))
    ls = float(np.mean(s_gen.astype(np.float64) < tau_s))

    diag = {
        "tau_s": tau_s,
        "uniq_ratio_mean": float(np.mean(uniq_ratios)),
        "uniq_ratio_min": float(np.min(uniq_ratios)),
        "uniq_ratio_max": float(np.max(uniq_ratios)),
    }
    return rf, ls, diag


# -------------------- FRZ 聚合 --------------------

def compute_frz(
    frames_ref: List[np.ndarray],
    frames_gen: List[np.ndarray],
    s_ref: np.ndarray,
    s_gen: np.ndarray,
    *,
    tau_motion_abs: float = 0.0,  # 仅当 median(s_ref) > tau_motion_abs 才启用 FRZ
    w_rf: float = 1.0,
    w_ls: float = 1.0,
    roi_masks_gen: Optional[List[Optional[np.ndarray]]] = None,
    roi_bbox_expand: int = 0,
    roi_on_empty: str = "union",
    **ncc_kwargs,  # 兼容别名 roi_masks
) -> Tuple[float, Dict[str, float]]:
    """
    最终 FRZ：
      若 median(s_ref) <= tau_motion_abs，则 FRZ=0；
      否则 FRZ = clip( w_rf*RF + w_ls*LS, 0, 1 )
    """
    med_ref = float(np.median(s_ref.astype(np.float64)))
    if med_ref <= float(tau_motion_abs):
        return 0.0, {"median_s_ref": med_ref, "enabled": 0.0}

    # 兼容别名：roi_masks -> roi_masks_gen
    if roi_masks_gen is None and "roi_masks" in ncc_kwargs:
        roi_masks_gen = ncc_kwargs.pop("roi_masks")

    rf, ls, diag = compute_rf_ls(
        frames_ref, frames_gen, s_ref, s_gen,
        tau_s_quantile_ref=float(ncc_kwargs.pop("tau_s_quantile_ref", 0.40)),
        ncc_patch=int(ncc_kwargs.pop("ncc_patch", 32)),
        ncc_stride=ncc_kwargs.pop("ncc_stride", None),
        ncc_thr=float(ncc_kwargs.pop("ncc_thr", 0.98)),
        unique_min_ratio=float(ncc_kwargs.pop("unique_min_ratio", 0.20)),
        roi_masks_gen=roi_masks_gen,
        roi_bbox_expand=int(ncc_kwargs.pop("roi_bbox_expand", roi_bbox_expand)),
        roi_on_empty=str(ncc_kwargs.pop("roi_on_empty", roi_on_empty)),
    )

    # 其余未知 kwargs 忽略（保持最小改动兼容）
    frz = float(np.clip(w_rf * rf + w_ls * ls, 0.0, 1.0))
    diag_out = {
        "RF": rf, "LS": ls, "FRZ": frz,
        "median_s_ref": med_ref, "enabled": 1.0, **diag
    }
    return frz, diag_out
