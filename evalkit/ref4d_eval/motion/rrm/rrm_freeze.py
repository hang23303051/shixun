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

新增：
- compute_frz_refside(s_ref, ...): 只用 s_ref 预计算参考侧元信息（median_s_ref, tau_s 等）
- compute_frz_from_cache(ref_meta, frames_gen, s_gen, ...): 在无参考视频情况下，根据 ref_meta + frames_gen + s_gen 计算 FRZ
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2 as cv

__all__ = [
    "rf_unique_ratio_ncc",
    "compute_rf_ls",
    "compute_frz_refside",
    "compute_frz_from_cache",
    "compute_frz",
]


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


# -------------------- RF + LS（原始实现，保留兼容） --------------------

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

    注意：frames_ref 在当前实现中未直接使用，仅用于接口兼容。
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


# -------------------- 新增：参考侧预计算 --------------------

def compute_frz_refside(
    s_ref: np.ndarray,
    *,
    tau_motion_abs: float = 0.0,
    tau_s_quantile_ref: float = 0.40,
    **kwargs,
) -> Dict[str, Any]:
    """
    只用参考侧的 s_ref 预计算 FRZ 需要的元信息。
    返回的 ref_meta 可被保存到缓存，用于后续无 refvideo 情况下计算 FRZ。

    ref_meta 字段:
      - median_s_ref: 参考端 motion 强度中位数
      - tau_s: 用于 LS 的阈值 P_q(s_ref)
      - tau_s_quantile_ref: 使用的分位数 q
      - tau_motion_abs: 启用 FRZ 的绝对阈值（用于门控）
    """
    if s_ref.size == 0:
        raise ValueError("empty s_ref in compute_frz_refside")

    s_ref_f = s_ref.astype(np.float64)
    med_ref = float(np.median(s_ref_f))
    tau_s = float(np.quantile(s_ref_f, float(tau_s_quantile_ref)))

    ref_meta: Dict[str, Any] = {
        "median_s_ref": med_ref,
        "tau_s": tau_s,
        "tau_s_quantile_ref": float(tau_s_quantile_ref),
        "tau_motion_abs": float(tau_motion_abs),
    }
    return ref_meta


# -------------------- 新增：基于 ref_meta 的 FRZ 计算 --------------------

def compute_frz_from_cache(
    ref_meta: Dict[str, Any],
    frames_gen: List[np.ndarray],
    s_gen: np.ndarray,
    *,
    tau_motion_abs: Optional[float] = None,
    w_rf: float = 1.0,
    w_ls: float = 1.0,
    roi_masks_gen: Optional[List[Optional[np.ndarray]]] = None,
    roi_bbox_expand: int = 0,
    roi_on_empty: str = "union",
    **ncc_kwargs,
) -> Tuple[float, Dict[str, float]]:
    """
    在没有参考视频的情况下，根据:
      - ref_meta: 来自 compute_frz_refside(s_ref, ...)
      - frames_gen, s_gen: 生成端数据
    计算 FRZ（以及 RF/LS/诊断信息）。

    要求：当 ref_meta 来自相同的 s_ref 且参数一致时，应满足数值等价：
      compute_frz(frames_ref, frames_gen, s_ref, s_gen, ...) == compute_frz_from_cache(ref_meta, frames_gen, s_gen, ...)
    """
    if frames_gen is None or len(frames_gen) == 0 or s_gen.size == 0:
        raise ValueError("empty inputs for compute_frz_from_cache")

    # 解析 ref_meta
    med_ref = float(ref_meta.get("median_s_ref", 0.0))
    tau_s = float(ref_meta.get("tau_s", 0.0))
    tau_s_quantile_ref = float(ref_meta.get("tau_s_quantile_ref", 0.40))
    tau_motion_abs_meta = float(ref_meta.get("tau_motion_abs", 0.0))

    # 若调用方未显式给 tau_motion_abs，则使用 ref_meta 中的值（保证与预计算阶段一致）
    if tau_motion_abs is None:
        tau_motion_abs = tau_motion_abs_meta
    tau_motion_abs = float(tau_motion_abs)

    # --- 门控：参考端运动太弱则直接关闭 FRZ ---
    if med_ref <= tau_motion_abs:
        diag_out = {
            "median_s_ref": med_ref,
            "enabled": 0.0,
            # 其余诊断量置零
            "RF": 0.0,
            "LS": 0.0,
            "FRZ": 0.0,
            "tau_s": tau_s,
            "uniq_ratio_mean": 0.0,
            "uniq_ratio_min": 0.0,
            "uniq_ratio_max": 0.0,
        }
        return 0.0, diag_out

    # 兼容别名：roi_masks -> roi_masks_gen
    if roi_masks_gen is None and "roi_masks" in ncc_kwargs:
        roi_masks_gen = ncc_kwargs.pop("roi_masks")

    # NCC / RF 参数（默认值与 compute_frz / compute_rf_ls 保持一致）
    ncc_patch = int(ncc_kwargs.pop("ncc_patch", 32))
    ncc_stride = ncc_kwargs.pop("ncc_stride", None)
    ncc_thr = float(ncc_kwargs.pop("ncc_thr", 0.98))
    unique_min_ratio = float(ncc_kwargs.pop("unique_min_ratio", 0.20))
    roi_bbox_expand = int(ncc_kwargs.pop("roi_bbox_expand", roi_bbox_expand))
    roi_on_empty = str(ncc_kwargs.pop("roi_on_empty", roi_on_empty))

    # 其余未知 kwargs 忽略（保持最小入侵兼容）
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

    s_gen_f = s_gen.astype(np.float64)
    ls = float(np.mean(s_gen_f < tau_s))

    diag = {
        "tau_s": tau_s,
        "uniq_ratio_mean": float(np.mean(uniq_ratios)),
        "uniq_ratio_min": float(np.min(uniq_ratios)),
        "uniq_ratio_max": float(np.max(uniq_ratios)),
    }

    frz = float(np.clip(w_rf * rf + w_ls * ls, 0.0, 1.0))
    diag_out: Dict[str, float] = {
        "RF": rf,
        "LS": ls,
        "FRZ": frz,
        "median_s_ref": med_ref,
        "enabled": 1.0,
        **diag,
    }
    return frz, diag_out


# -------------------- FRZ 聚合（对外接口，保持兼容） --------------------

def compute_frz(
    frames_ref: List[np.ndarray],
    frames_gen: List[np.ndarray],
    s_ref: np.ndarray,
    s_gen: np.ndarray,
    *,
    tau_motion_abs: float = 0.0,  # 仅当 median(s_ref) > tau_motion_abs 才启用 FRZ（写入 ref_meta）
    w_rf: float = 1.0,
    w_ls: float = 1.0,
    roi_masks_gen: Optional[List[Optional[np.ndarray]]] = None,
    roi_bbox_expand: int = 0,
    roi_on_empty: str = "union",
    refside_meta: Optional[Dict[str, Any]] = None,
    **ncc_kwargs,
) -> Tuple[float, Dict[str, float]]:
    """
    最终 FRZ（兼容原接口 + 支持 refside_meta）：

    - 若 refside_meta 为 None：
        1）先用 s_ref 调用 compute_frz_refside(s_ref, tau_motion_abs, tau_s_quantile_ref) 得到 ref_meta
        2）再调用 compute_frz_from_cache(ref_meta, frames_gen, s_gen, ...) 计算 FRZ
    - 若 refside_meta 不为 None：
        - 直接视为预计算好的 ref_meta（通常来自缓存 npz 的 frz_meta），
          不再从 s_ref 重新计算门限，保证与预计算完全一致。

    关键设计：
    - 始终让 compute_frz_from_cache 从 ref_meta 中读取 tau_motion_abs，
      因此只要 ref_meta 一样，在线 / 缓存 两种模式的 FRZ 结果就完全对齐。
    - frames_ref 仅保留接口兼容，目前 FRZ 的计算本身只依赖：
        * 参考侧的运动强度统计（已经压缩到 ref_meta）
        * 生成端的 frames_gen + s_gen
    """
    # tau_s_quantile_ref 只在“需要从 s_ref 重新推 ref_meta”时使用；
    # 不传给 compute_frz_from_cache，避免第二阶段再改阈值。
    tau_s_quantile_ref = float(ncc_kwargs.pop("tau_s_quantile_ref", 0.40))

    # 1）确定 ref_meta：优先使用预计算的 refside_meta；否则在线从 s_ref 计算。
    if refside_meta is not None:
        # 直接信任缓存的参考端元信息（通常是 dict）
        ref_meta = refside_meta
    else:
        # 在线模式：根据当前 s_ref 和 tau_motion_abs / tau_s_quantile_ref 计算 ref_meta
        ref_meta = compute_frz_refside(
            s_ref,
            tau_motion_abs=tau_motion_abs,
            tau_s_quantile_ref=tau_s_quantile_ref,
        )

    # 2）真正的 FRZ 计算统一走 compute_frz_from_cache
    #    注意：这里故意把 tau_motion_abs 设为 None，
    #    让 compute_frz_from_cache 一律从 ref_meta["tau_motion_abs"] 里取值，
    #    确保“预计算 vs 在线”只要 ref_meta 相同，最终分数就 1:1 对齐。
    frz, diag_out = compute_frz_from_cache(
        ref_meta,
        frames_gen,
        s_gen,
        tau_motion_abs=None,  # 始终使用 ref_meta 中的 tau_motion_abs
        w_rf=w_rf,
        w_ls=w_ls,
        roi_masks_gen=roi_masks_gen,
        roi_bbox_expand=roi_bbox_expand,
        roi_on_empty=roi_on_empty,
        **ncc_kwargs,
    )
    return frz, diag_out

