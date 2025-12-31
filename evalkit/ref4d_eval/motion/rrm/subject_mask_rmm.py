# -*- coding: utf-8 -*-
"""
Single-video subject & background masks for Alignment-Free RRM.

依赖：
- motion_eval/preprocess/subject_mask.py:
  - _gather_text_prompts
  - _detect_boxes_grounding_dino
  - _segment_video_sam2_with_boxes
  - _postprocess_mask

参考：
- GroundingDINO（开集检测） :contentReference[oaicite:0]{index=0}
- SAM 2（视频分割传播） :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
import os
import numpy as np
import cv2 as cv

# 相对导入：按你的目录结构，当前文件位于 motion_eval/rrm/
from ..preprocess.subject_mask import (
    _gather_text_prompts,
    _detect_boxes_grounding_dino,
    _segment_video_sam2_with_boxes,
    _postprocess_mask,
)

__all__ = [
    "build_subject_masks_single",
    "build_fg_bg_masks_single",
    "make_bg_ring",
]


def _set_env_from_cfg(cfg_subject: Dict[str, Any]) -> None:
    # GroundingDINO（不变）
    gdino_cfg  = str(cfg_subject.get("gdino_cfg", "") or "").strip()
    gdino_ckpt = str(cfg_subject.get("gdino_ckpt", "") or "").strip()
    if gdino_cfg and gdino_ckpt:
        os.environ["GROUNDING_DINO_CFG"] = gdino_cfg
        os.environ["GROUNDING_DINO_WEIGHTS"] = gdino_ckpt

    # SAM2：你的 subject_mask.py 只读 SAM2_CFG_NAME 与 SAM2_CKPT / SAM2_CHECKPOINT
    sam2_cfg_name = str(cfg_subject.get("sam2_cfg_name", "") or "").strip()
    sam2_ckpt     = str(cfg_subject.get("sam2_ckpt", "") or "").strip()
    if sam2_cfg_name:
        os.environ["SAM2_CFG_NAME"] = sam2_cfg_name
    if sam2_ckpt:
        os.environ["SAM2_CKPT"] = sam2_ckpt



def build_subject_masks_single(
    frames_bgr: List[np.ndarray],
    ref_path: str,
    cfg_subject: Dict[str, Any],
) -> Optional[List[Optional[np.ndarray]]]:
    """
    仅在单个视频上构建“主体掩码时间序列”（bool）。
    - 不做时空对齐；不引入任何回退机制。
    - 失败时返回 None；上层自行决定是否忽略该样本。

    参数
    ----
    frames_bgr : List[np.ndarray]  # [T,H,W,3] BGR
    ref_path   : str               # 用于从语义 JSON / 路径派生 prompts
    cfg_subject: Dict[str,Any]     # 超参（阈值/模型路径/后处理核尺寸等）

    返回
    ----
    masks_fg : Optional[List[Optional[np.ndarray]]]
      - 若成功：长度为 T 的列表，每项为 HxW 的 bool 掩码或 None（该帧未成功分割）
      - 若失败：None
    """
    if not bool(cfg_subject.get("enable", True)):
        return None

    _set_env_from_cfg(cfg_subject)

    # 1) 文本提示（严格单端）：用户显式 + 语义 JSON + 路径派生
    prompts = _gather_text_prompts(ref_path,cfg_subject.get("text_prompts", cfg_subject.get("prompts", None)),)
    if not prompts:
        return None  # 不做回退

    # 2) GroundingDINO：逐帧检测 bbox（只要阈值给出）
    box_conf_thr = float(cfg_subject.get("box_conf_thr", 0.35))
    text_thr     = float(cfg_subject.get("text_thr", 0.25))
    topk         = int(cfg_subject.get("topk_instances", 3))
    boxes_per_frame = _detect_boxes_grounding_dino(
        frames_bgr=frames_bgr,
        text_prompts=prompts,
        box_conf_thr=box_conf_thr,
        text_thr=text_thr,
        topk_instances=topk,
    )

    # 3) SAM2：以检测框为正例点，做全视频掩码传播
    pred_iou_thr = float(cfg_subject.get("sam2_pred_iou_thr", 0.86))
    masks_raw = _segment_video_sam2_with_boxes(
        frames_bgr=frames_bgr,
        per_frame_boxes=boxes_per_frame,
        pred_iou_thr=pred_iou_thr,
    )


    if masks_raw is None:
        return None  # 不做回退

    # 4) 后处理（腐蚀/膨胀/小孔填充），逐帧
    erode = int(cfg_subject.get("post_erode", 1))
    dilate = int(cfg_subject.get("post_dilate", 2))
    fill_ratio = float(cfg_subject.get("post_fill_ratio", 0.005))
    masks_fg: List[Optional[np.ndarray]] = []
    for m in masks_raw:
        if m is None:
            masks_fg.append(None)
        else:
            masks_fg.append(_postprocess_mask(m, erode=erode, dilate=dilate, fill_hole_ratio=fill_ratio))
    return masks_fg


def make_bg_ring(
    mask_fg: np.ndarray,
    ring_ratio: float = 0.08,
    min_ring_px: int = 6,
) -> np.ndarray:
    """
    由主体掩码生成“背景环带”：
      - 对 mask 进行外扩 r 像素（r = max(min_ring_px, short_side*ring_ratio)）
      - 环带 = 外扩后的掩码 − 原掩码
    绝不对空掩码兜底（调用方需保证 mask_fg 为非空布尔阵）。

    返回：HxW 的 bool
    """
    m = (mask_fg.astype(np.uint8) * 255)
    H, W = m.shape[:2]
    short_side = min(H, W)
    r = int(max(min_ring_px, round(short_side * float(ring_ratio))))
    r = max(1, r)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    dil = cv.dilate(m, kernel, iterations=1)
    ring = (dil > 0) & (m == 0)
    return ring.astype(bool)


def build_fg_bg_masks_single(
    frames_bgr: List[np.ndarray],
    ref_path: str,
    cfg_subject: Dict[str, Any],
    ring_ratio: float = 0.08,
    min_ring_px: int = 6,
) -> Optional[Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]]:
    """
    一次性构建主体掩码序列 + 背景环带序列（严格单端）。
    任一帧主体为 None → 对应帧的背景也置为 None（不做回退）。

    返回：masks_fg, masks_bg（均为长度 T 的 List[Optional[np.ndarray[H,W,bool]]]
    """
    masks_fg = build_subject_masks_single(frames_bgr, ref_path, cfg_subject)
    if masks_fg is None:
        return None

    masks_bg: List[Optional[np.ndarray]] = []
    for m in masks_fg:
        if m is None or m.sum() == 0:
            masks_bg.append(None)  # 不做回退/填充
        else:
            masks_bg.append(make_bg_ring(m, ring_ratio=ring_ratio, min_ring_px=min_ring_px))
    return masks_fg, masks_bg


def repair_masks_temporal(
    masks_fg: list, masks_bg: list,
    *,
    enable: bool = True,
    max_skip: int = 3,          # 仅修补间隔 ≤ max_skip 的短洞
    dilate_iters: int = 0,      # 修补后可选膨胀
    min_area_px: int = 0        # 若面积依旧很小可再膨胀一次
):
    """
    单视频内部的短洞修补（不跨视频、不过度修改）：
      - 前景/背景分别独立按最近邻修补；
      - 最近邻距离 > max_skip 时保持 None；
      - 可选少量形态学膨胀，避免掩码过瘦。
    返回：修补后的 (masks_fg, masks_bg, diag)
    """
    if not enable:
        return masks_fg, masks_bg, {"fixed_fg":0, "fixed_bg":0, "skipped_fg":sum(m is None for m in masks_fg), "skipped_bg":sum(m is None for m in masks_bg)}

    import numpy as np, cv2 as cv

    def _dilate_bool(mask_bool, it):
        if it <= 0: return mask_bool
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*it+1, 2*it+1))
        out = cv.dilate(mask_bool.astype(np.uint8)*255, k, iterations=1) > 0
        return out

    T = len(masks_fg)
    if T != len(masks_bg):
        raise ValueError("fg/bg mask length mismatch")

    out_fg = list(masks_fg)
    out_bg = list(masks_bg)

    valid_fg = [i for i, m in enumerate(out_fg) if m is not None]
    valid_bg = [i for i, m in enumerate(out_bg) if m is not None]

    def _nearest(valid_list, t):
        if not valid_list:
            return None
        return min(valid_list, key=lambda j: abs(j - t))

    fixed_fg = 0
    fixed_bg = 0

    for t in range(T):
        if out_fg[t] is None:
            j = _nearest(valid_fg, t)
            if j is not None and abs(j - t) <= max_skip and out_fg[j] is not None:
                f = out_fg[j].copy()
                if dilate_iters > 0:
                    f = _dilate_bool(f, dilate_iters)
                if (min_area_px > 0) and (int(f.sum()) < min_area_px):
                    f = _dilate_bool(f, 1)
                out_fg[t] = f
                fixed_fg += 1
        if out_bg[t] is None:
            j = _nearest(valid_bg, t)
            if j is not None and abs(j - t) <= max_skip and out_bg[j] is not None:
                b = out_bg[j].copy()
                if dilate_iters > 0:
                    b = _dilate_bool(b, dilate_iters)
                if (min_area_px > 0) and (int(b.sum()) < min_area_px):
                    b = _dilate_bool(b, 1)
                out_bg[t] = b
                fixed_bg += 1

    diag = {
        "fixed_fg": int(fixed_fg),
        "fixed_bg": int(fixed_bg),
        "skipped_fg": int(sum(m is None for m in out_fg)),
        "skipped_bg": int(sum(m is None for m in out_bg)),
    }
    return out_fg, out_bg, diag
