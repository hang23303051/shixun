# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple, Optional
import cv2 as cv
import numpy as np

__all__ = [
    "load_video_cv2",
    "save_video_cv2",
    "resize_keep_short_side",
    "resample_video",
]

def _maybe_rotate_by_meta(frame: np.ndarray, orientation: int) -> np.ndarray:
    """
    部分平台写入了 ORIENTATION_META；若能读到就做旋转矫正。
    0/未定义：不旋；90/180/270：顺时针旋转。
    """
    if not isinstance(orientation, (int, float)):  # 未知
        return frame
    o = int(orientation)
    if o == 90:
        return cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
    if o == 180:
        return cv.rotate(frame, cv.ROTATE_180)
    if o == 270:
        return cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
    return frame

def load_video_cv2(path: str, bgr: bool = True, return_fps: bool = False):
    """
    读取视频为 list[H,W,3] (uint8)。默认 BGR。
    若 return_fps=True，返回 (frames, fps_src: float)。
    - 修复：在 release 之前读取 FPS / ORIENTATION_META；
    - 统一把每帧保证为连续内存、uint8。
    """
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    # 在循环前读取元数据（release 之后不可再读）
    try:
        fps_src = float(cap.get(cv.CAP_PROP_FPS))
        if not np.isfinite(fps_src) or fps_src <= 0:
            fps_src = 0.0
    except Exception:
        fps_src = 0.0

    try:
        # 有些构建没有该属性，容错
        orientation = cap.get(cv.CAP_PROP_ORIENTATION_META)
    except Exception:
        orientation = 0

    frames: List[np.ndarray] = []
    ok, frame = cap.read()
    while ok:
        if not bgr:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # 方向矫正
        frame = _maybe_rotate_by_meta(frame, orientation)
        # 确保 dtype 与内存布局
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        frames.append(frame)
        ok, frame = cap.read()
    cap.release()

    if return_fps:
        return frames, fps_src
    return frames

def save_video_cv2(path: str, frames: List[np.ndarray], fps: int = 25) -> None:
    """
    将 list[H,W,3] 写为 mp4（mp4v 编码）；仅用于调试/可视化。
    """
    if len(frames) == 0:
        raise ValueError("No frames to save.")
    h, w = frames[0].shape[:2]
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    vw = cv.VideoWriter(path, fourcc, float(fps), (w, h))
    for img in frames:
        if img.shape[:2] != (h, w):
            img = cv.resize(img, (w, h), interpolation=cv.INTER_AREA)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        vw.write(img)
    vw.release()

def resize_keep_short_side(img: np.ndarray, short_side: int) -> np.ndarray:
    """
    保持纵横比，把短边缩放到 short_side；短边已等于目标或 short_side<=0 时原样返回。
    """
    h, w = img.shape[:2]
    if short_side <= 0 or h == 0 or w == 0:
        return img
    m = min(h, w)
    if m == short_side:
        return img
    if h < w:
        new_h = short_side
        new_w = int(round(w * (short_side / h)))
    else:
        new_w = short_side
        new_h = int(round(h * (short_side / w)))
    return cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA if (new_w < w or new_h < h) else cv.INTER_LINEAR)

def _time_resample_indices(T: int, src_fps: float, tgt_fps: float) -> np.ndarray:
    """
    给定原帧数 T、源/目标 fps，返回等时长重采样索引（长度约为 round(T* tgt/src)）。
    - 若 src_fps 无效（<=0），退化为“不改帧数”，保证与旧 CLI 兼容。
    """
    if T <= 1 or tgt_fps <= 0:
        return np.arange(T, dtype=np.int32)
    if src_fps <= 0:
        return np.arange(T, dtype=np.int32)
    duration = T / float(src_fps)
    tgt_len = max(1, int(round(duration * float(tgt_fps))))
    idx = np.linspace(0, T - 1, num=tgt_len).round().astype(np.int32)
    return np.clip(idx, 0, T - 1)

def resample_video(
    frames: List[np.ndarray],
    short_side: int = 448,
    fps: int = 12,
    src_fps: Optional[float] = None,
) -> List[np.ndarray]:
    """
    尺度与帧率统一：
      1) 逐帧缩放到短边=short_side，保持比例（AREA 下采样、LINEAR 上采样）；
      2) 按 src_fps → fps 重采样；src_fps<=0 时保持帧数不变（与旧逻辑兼容）。
    """
    if len(frames) == 0:
        return []
    # 1) 尺度
    scaled = [resize_keep_short_side(f, short_side) for f in frames]
    # 2) 时间采样
    idx = _time_resample_indices(len(scaled), float(src_fps or 0.0), float(fps))
    return [scaled[i] for i in idx.tolist()]
