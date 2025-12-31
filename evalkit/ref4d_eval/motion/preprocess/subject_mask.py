# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import os, sys, json, shutil, glob, uuid, traceback
import numpy as np
import cv2 as cv
import torch

__all__ = ["build_subject_masks", "warp_masks_gen_to_ref", "derive_prompts_from_path"]

# ---------------- small utils ----------------
def _dbg_on() -> bool:
    return os.environ.get("SUBJ_DEBUG", "0") not in ("0", "", "false", "False", "OFF")

def _log(*args):
    if _dbg_on():
        print("[SUBJ]", *args, flush=True)

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _postprocess_mask(m: np.ndarray, erode: int = 1, dilate: int = 2,
                      fill_hole_ratio: float = 0.005) -> np.ndarray:
    if m is None or m.size == 0:
        return m
    m = (m > 0).astype(np.uint8)
    if erode > 0:
        m = cv.erode(m, cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*erode+1, 2*erode+1)))
    if dilate > 0:
        m = cv.dilate(m, cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*dilate+1, 2*dilate+1)))
    H, W = m.shape[:2]
    if fill_hole_ratio > 0:
        thr = int(fill_hole_ratio * H * W)
        inv = (1 - m).astype(np.uint8)
        num, lab = cv.connectedComponents(inv, 8)
        for i in range(1, num):
            if int((lab == i).sum()) <= thr:
                inv[lab == i] = 0
        m = (1 - inv).astype(np.uint8)
    return m.astype(bool)

def derive_prompts_from_path(ref_path: str) -> List[str]:
    base = os.path.basename(os.path.dirname(ref_path))
    toks = [t for t in base.replace("-", "_").split("_") if t.isalpha()]
    toks = [t.lower() for t in toks if len(t) >= 3]
    return toks or ["object"]

def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as h:
            return json.load(h)
    except Exception:
        try:
            # 去掉BOM/注释等常见脏格式
            txt = open(path, "r", encoding="utf-8", errors="ignore").read()
            txt = "\n".join([ln for ln in txt.splitlines() if not ln.strip().startswith("//")])
            return json.loads(txt.lstrip("\ufeff"))
        except Exception as e2:
            _log("semantics read error:", e2)
            return None

def _read_semantics_for(ref_path: str) -> List[str]:
    """从 Ref4D 语义证据 JSON 中抽取主体名/物种名等，返回若干 token."""
    # 1) 找到仓库根目录：从 ref_path 中截到 "data" 之前
    norm = os.path.normpath(ref_path)
    parts = norm.split(os.sep)
    if "data" in parts:
        idx = parts.index("data")
        root = os.sep.join(parts[:idx]) or "."
    else:
        root = os.path.dirname(os.path.dirname(ref_path))

    sample_id = os.path.splitext(os.path.basename(ref_path))[0]

    # 2) 语义证据目录：优先 data/metadata/semantic_evidence，兼容旧 data/semantics
    sem_dirs = [
        os.path.join(root, "data", "metadata", "semantic_evidence"),
        os.path.join(root, "data", "semantics"),
    ]

    cand: List[str] = []
    for d in sem_dirs:
        cand.append(os.path.join(d, sample_id + ".json"))
        cand.extend(sorted(glob.glob(os.path.join(d, sample_id + "*.json"))))

    js = None
    for f in cand:
        if os.path.exists(f):
            js = _read_json(f)
            if _dbg_on():
                _log("[SEM] use file:", f)
            break

    if js is None or not isinstance(js, dict):
        if _dbg_on():
            _log("[SEM] no semantic json for", sample_id, "under", sem_dirs)
        return []

    toks: List[str] = []

    # 3) 直接字段（兼容其它 schema），现在可以不重要，但保留以防你以后扩展
    for k in ("prompts", "text", "labels", "objects", "nouns"):
        v = js.get(k, None)
        if isinstance(v, str):
            toks.append(v)
        elif isinstance(v, (list, tuple)):
            toks.extend(str(x) for x in v)

    # 4) Ref4D 的主体实体：fine.entities
    try:
        fine = js.get("fine", {}) or {}
        ents = fine.get("entities", []) or []

        def span_len(e: dict) -> float:
            L = 0.0
            for s in e.get("spans", []) or []:
                if isinstance(s, (list, tuple)) and len(s) == 2:
                    try:
                        L += float(s[1]) - float(s[0])
                    except Exception:
                        pass
            return L

        # 先按出现时长排序，优先取前 3 个实体
        ents_sorted = sorted(ents, key=span_len, reverse=True)

        for e in ents_sorted[:3]:
            name = str(e.get("name", "")).strip().lower()
            if name:
                toks.append(name)

            attrs = e.get("attributes", {}) or {}
            # 一些有辨识度的属性，可以选取部分
            for ak in ("species-or-breed", "species", "breed",
                       "signature", "color", "pattern", "size", "age",
                       "pose", "action", "state"):
                vs = attrs.get(ak, [])
                if isinstance(vs, str):
                    toks.append(vs)
                elif isinstance(vs, (list, tuple)):
                    toks.extend(str(x) for x in vs)

        # 5) 再从 views.objects_count / objects_count_display 里拿 top 类别名
        views = js.get("views", {}) or {}
        oc = views.get("objects_count", {}) or views.get("objects_count_display", {}) or {}
        if isinstance(oc, dict) and len(oc) > 0:
            # 取计数最多的前 2 个对象名
            for k, _v in sorted(oc.items(), key=lambda kv: -int(kv[1]))[:2]:
                toks.append(str(k))
    except Exception as e:
        if _dbg_on():
            _log("semantics parse warn:", e)

    # 6) 清洗 + 去重
    bag: List[str] = []
    for s in toks:
        if not s:
            continue
        s = str(s).strip().lower()
        if not s:
            continue
        # 把逗号/斜杠/连字符都当分隔
        s = s.replace(",", " ").replace("/", " ").replace("-", " ")
        bag.extend(
            t for t in s.split()
            if t.isalpha() and len(t) >= 3
        )

    uniq: List[str] = []
    for t in bag:
        if t not in uniq:
            uniq.append(t)

    if _dbg_on():
        _log("[SEM] raw toks:", toks[:10])
        _log("[SEM] cleaned:", uniq[:10])

    return uniq



STOP_TOKENS = {"refvideo", "genvideo", "object", "video", "mp4"}
STOP_TOKENS = {"refvideo", "genvideo", "object", "video", "mp4"}

def _gather_text_prompts(ref_path: str, cfg_prompts: Optional[List[str]]) -> List[str]:
    user: List[str] = []
    if isinstance(cfg_prompts, (list, tuple)):
        user = [str(x).strip().lower() for x in cfg_prompts if str(x).strip()]

    sems = _read_semantics_for(ref_path)
    auto = derive_prompts_from_path(ref_path)

    allp = user + sems + auto
    uniq: List[str] = []
    for t in allp:
        if not t:
            continue
        t = t.strip().lower()
        if len(t) < 3:
            continue
        if t in STOP_TOKENS:
            continue
        if t not in uniq:
            uniq.append(t)

    if _dbg_on():
        _log("[PROM] path:", ref_path)
        _log("[PROM]  user:", user)
        _log("[PROM]  sems:", sems[:10], " (total", len(sems), ")")
        _log("[PROM]  auto:", auto[:10], " (total", len(auto), ")")
        _log("[PROM] final:", uniq[:12], " ... total:", len(uniq))

    return uniq or ["object"]


# ---------------- warping ----------------
def warp_masks_gen_to_ref(
    masks_gen: List[Optional[np.ndarray]],
    pi: np.ndarray,
    H_list: List[np.ndarray],
    size_hw: Tuple[int, int],
) -> List[Optional[np.ndarray]]:
    H_ref, W_ref = size_hw
    out: List[Optional[np.ndarray]] = []
    for t in range(len(pi)):
        j = int(np.clip(int(pi[t]), 0, len(masks_gen) - 1)) if len(masks_gen) > 0 else 0
        m = masks_gen[j] if (0 <= j < len(masks_gen)) else None
        if m is None:
            out.append(None); continue
        Ht = H_list[t] if t < len(H_list) else np.eye(3, dtype=np.float32)
        warped = cv.warpPerspective((m.astype(np.uint8)*255), Ht.astype(np.float32),
                                    (W_ref, H_ref), flags=cv.INTER_NEAREST, borderValue=0)
        out.append(warped > 127)
    return out

# ---------------- GroundingDINO ----------------
def _detect_boxes_grounding_dino(
    frames_bgr: List[np.ndarray],
    text_prompts: List[str],
    box_conf_thr: float,
    text_thr: float,
    topk_instances: int,
) -> List[Optional[np.ndarray]]:
    try:
        from groundingdino.util.inference import load_model, predict, load_image
        from groundingdino.util import box_ops
    except Exception as e:
        raise RuntimeError(f"[subject_mask] GroundingDINO not available: {e}")

    model_path = os.environ.get("GROUNDING_DINO_WEIGHTS", "")
    model_cfg  = os.environ.get("GROUNDING_DINO_CFG", "")
    if not (model_path and model_cfg):
        raise RuntimeError("[subject_mask] GROUNDING_DINO_WEIGHTS / GROUNDING_DINO_CFG not set.")

    model = load_model(model_cfg, model_path)
    model.eval()

    caption = ", ".join([str(t).strip().lower() for t in text_prompts if str(t).strip()])

    tmp_dir = os.path.join("/tmp", "gdino_" + uuid.uuid4().hex[:8])
    os.makedirs(tmp_dir, exist_ok=True)

    out_boxes: List[Optional[np.ndarray]] = []
    try:
        for idx, bgr in enumerate(frames_bgr):
            try:
                jpg = os.path.join(tmp_dir, f"{idx:05d}.jpg")
                cv.imwrite(jpg, bgr)
                image_source, image = load_image(jpg)
                boxes, logits, phrases = predict(
                    model=model, image=image, caption=caption,
                    box_threshold=box_conf_thr, text_threshold=text_thr
                )
                if boxes is None or (hasattr(boxes, "__len__") and len(boxes) == 0):
                    out_boxes.append(None); continue
                xyxy = box_ops.box_cxcywh_to_xyxy(boxes) if getattr(boxes, "shape", None) is not None and boxes.max() <= 1.1 else boxes
                xyxy = np.asarray(xyxy)
                if logits is not None and len(logits) == len(xyxy):
                    idxs = np.argsort(-np.asarray(logits).reshape(-1))[:max(1, topk_instances)]
                    xyxy = xyxy[idxs]
                else:
                    xyxy = xyxy[:max(1, topk_instances)]
                out_boxes.append(xyxy.astype(np.float32))
            except Exception as e:
                _log(f"GDINO frame{idx} error: {e}")
                out_boxes.append(None)
    finally:
        if os.environ.get("SUBJ_DEBUG", "0") != "2":
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return out_boxes

# ---------------- SAM2 (Video Predictor) ----------------
def _build_sam2_predictor_from_env():
    try:
        sys.path.append(os.environ.get("SAM2_PYDIR", ""))  # 可选：third_party/sam2
        from sam2.build_sam import build_sam2_video_predictor
    except Exception as e:
        raise RuntimeError(f"[subject_mask] SAM2 import failed: {e}")

    cfg_env = os.environ.get("SAM2_CFG_NAME", "").strip()
    ckpt = os.environ.get("SAM2_CHECKPOINT", "") or os.environ.get("SAM2_CKPT", "")
    if not ckpt or not os.path.exists(ckpt):
        raise RuntimeError("[subject_mask] SAM2 checkpoint missing (SAM2_CHECKPOINT / SAM2_CKPT)")

    candidates: List[str] = []
    if cfg_env:
        candidates.append(cfg_env)
        if cfg_env.endswith(".yaml"):
            base = cfg_env.replace(".yaml", "")
            if "/" in base and not base.startswith("configs/"):
                candidates.append("configs/" + base)
    candidates.append("configs/sam2.1/sam2.1_hiera_l")

    last_err = None
    for name in candidates:
        try:
            _log("[TRY] build_sam2_video_predictor(", name, ",", ckpt, ")")
            pred = build_sam2_video_predictor(name, ckpt)
            _log("[OK ] predictor built with:", name)
            return pred
        except Exception as e:
            _log("[ERR]", repr(e))
            last_err = e
    raise RuntimeError(f"[subject_mask] build_sam2_video_predictor failed. Last error: {repr(last_err)}")

def _boxes_to_pos_points(boxes_xyxy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.int32)
    bb = np.asarray(boxes_xyxy, np.float32).reshape(-1, 4)
    cx = 0.5 * (bb[:, 0] + bb[:, 2])
    cy = 0.5 * (bb[:, 1] + bb[:, 3])
    pts = np.stack([cx, cy], axis=-1).astype(np.float32)
    labels = np.ones((pts.shape[0],), np.int32)
    return pts, labels

def _segment_video_sam2_with_boxes(
    frames_bgr: List[np.ndarray],
    per_frame_boxes: List[Optional[np.ndarray]],
    pred_iou_thr: float,
) -> List[Optional[np.ndarray]]:
    predictor = _build_sam2_predictor_from_env()

    tmp_dir = os.path.join("/tmp", "subj_sam2_" + uuid.uuid4().hex[:8])
    _ensure_dir(tmp_dir)
    try:
        for i, bgr in enumerate(frames_bgr):
            rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
            cv.imwrite(os.path.join(tmp_dir, f"{i:05d}.jpeg"), rgb)

        with torch.inference_mode():
            state = predictor.init_state(video_path=tmp_dir)
            predictor.reset_state(state)

            first_idx = None
            for t, b in enumerate(per_frame_boxes):
                if b is not None and len(b) > 0:
                    first_idx = t
                    break
            if first_idx is None:
                _log("SAM2 skip: no boxes at any frame")
                return [None] * len(frames_bgr)

            pts, lbs = _boxes_to_pos_points(per_frame_boxes[first_idx])
            if pts.shape[0] == 0:
                _log("SAM2 skip: boxes->points empty")
                return [None] * len(frames_bgr)

            obj_id = 1
            predictor.add_new_points(
                inference_state=state,
                frame_idx=int(first_idx),
                obj_id=obj_id,
                points=pts,
                labels=lbs,
            )

            masks_bool: List[Optional[np.ndarray]] = [None] * len(frames_bgr)
            for frame_idx, object_ids, masks_out in predictor.propagate_in_video(state):
                ml = masks_out
                if isinstance(ml, dict):
                    ml = ml.get("masks", ml.get("mask_logits", None))
                if ml is None:
                    masks_bool[frame_idx] = None; continue
                ml = torch.as_tensor(ml)
                if ml.ndim == 4 and ml.shape[1] == 1:
                    ml = ml[:, 0, ...]
                elif ml.ndim != 3:
                    masks_bool[frame_idx] = None; continue
                m_np = (ml > 0).any(dim=0).to(torch.uint8).cpu().numpy()
                masks_bool[frame_idx] = _postprocess_mask(
                    m_np, erode=1, dilate=2, fill_hole_ratio=0.005
                )

            cov = [
                float(np.count_nonzero(m)) / float(m.size)
                for m in masks_bool if m is not None and m.size > 0
            ]
            med = float(np.median(cov)) if len(cov) else 0.0
            _log(f"SAM2 coverage: med={med:.6f}, frames={len(cov)}/{len(frames_bgr)}")
            return masks_bool
    finally:
        if os.environ.get("SUBJ_DEBUG", "0") != "2":
            shutil.rmtree(tmp_dir, ignore_errors=True)

# ---------------- public entry ----------------
def build_subject_masks(
    frames_ref: List[np.ndarray],
    frames_gen: List[np.ndarray],
    ref_path: str,
    pi: np.ndarray,
    H_list: List[np.ndarray],
    cfg_subject: Dict[str, Any],
) -> Optional[List[Optional[np.ndarray]]]:
    """
    返回 REF 时间线上的主体掩码（bool）。失败/置信不足 → None（绝不 bbox 兜底）。
    """
    if not bool(cfg_subject.get("enable", False)):
        _log("subject branch disabled by config")
        return None

    # --- 将 YAML 的路径映射到环境变量（以兼容现有实现）
    gdino_cfg  = str(cfg_subject.get("gdino_cfg", "") or "").strip()
    gdino_ckpt = str(cfg_subject.get("gdino_ckpt", "") or "").strip()
    if gdino_cfg and gdino_ckpt:
        os.environ["GROUNDING_DINO_CFG"] = gdino_cfg
        os.environ["GROUNDING_DINO_WEIGHTS"] = gdino_ckpt

    # SAM2：优先 sam2_cfg_name；否则 sam2_cfg + sam2_ckpt
    sam2_cfg_name = str(cfg_subject.get("sam2_cfg_name", "") or "").strip()
    sam2_cfg = str(cfg_subject.get("sam2_cfg", "") or "").strip()
    sam2_ckpt = str(cfg_subject.get("sam2_ckpt", "") or "").strip()
    if sam2_cfg_name:
        os.environ["SAM2_CFG_NAME"] = sam2_cfg_name
    elif sam2_cfg:
        os.environ["SAM2_CFG_NAME"] = sam2_cfg  # _build_sam2_predictor_from_env 做了 .yaml 兼容
    if sam2_ckpt:
        os.environ["SAM2_CKPT"] = sam2_ckpt

    # prompts：优先 cfg，其次 semantics，再退回目录名
    prompts = _gather_text_prompts(ref_path, cfg_subject.get("text_prompts"))

    box_conf_thr = float(cfg_subject.get("box_conf_thr", 0.45))
    text_thr     = float(cfg_subject.get("text_thr", 0.30))
    pred_iou_thr = float(cfg_subject.get("pred_iou_thr", 0.80))
    topk         = int(cfg_subject.get("topk_instances", 1))
    cov_min      = float(cfg_subject.get("cov_min", 0.005))
    mask_logic   = str(cfg_subject.get("mask_logic", "ref")).lower()

    # 1) Ref 侧：GDINO→SAM2
    _log("detect on REF ...")
    boxes_ref = _detect_boxes_grounding_dino(frames_ref, prompts, box_conf_thr, text_thr, topk)
    masks_ref = _segment_video_sam2_with_boxes(frames_ref, boxes_ref, pred_iou_thr)

    # 2) Gen 侧：GDINO→SAM2，并对齐回 REF
    _log("detect on GEN ...")
    boxes_gen = _detect_boxes_grounding_dino(frames_gen, prompts, box_conf_thr, text_thr, topk)
    masks_gen = _segment_video_sam2_with_boxes(frames_gen, boxes_gen, pred_iou_thr)
    H_ref, W_ref = frames_ref[0].shape[:2]
    masks_gen2ref = warp_masks_gen_to_ref(masks_gen, pi=np.asarray(pi, np.int32), H_list=H_list, size_hw=(H_ref, W_ref))

    # 3) 组合策略
    out: List[Optional[np.ndarray]] = []
    for t in range(len(frames_ref)):
        mr = masks_ref[t] if (masks_ref is not None and t < len(masks_ref)) else None
        mg = masks_gen2ref[t] if (masks_gen2ref is not None and t < len(masks_gen2ref)) else None
        if mask_logic == "ref":
            S = mr
        elif mask_logic == "gen_warp":
            S = mg
        elif mask_logic == "intersection":
            S = (None if (mr is None or mg is None) else np.logical_and(mr, mg))
        else:  # union
            if (mr is None) and (mg is None): S = None
            elif mr is None: S = mg
            elif mg is None: S = mr
            else: S = np.logical_or(mr, mg)

        if S is not None:
            S = _postprocess_mask(
                S,
                erode=int(cfg_subject.get("erode", 1)),
                dilate=int(cfg_subject.get("dilate", 2)),
                fill_hole_ratio=float(cfg_subject.get("fill_hole_ratio", 0.005))
            )
        out.append(S)

    # 4) 覆盖率门控
    cov = [float(np.count_nonzero(m))/float(m.size) for m in out if m is not None and m.size > 0]
    med = float(np.median(cov)) if len(cov) else 0.0
    _log(f"final coverage median={med:.6f}, min_thr={cov_min}")
    if len(cov) == 0 or med < cov_min:
        return None

    # 调试可视化
    if _dbg_on():
        try:
            dbg_dir = "/tmp/subj_dbg"; _ensure_dir(dbg_dir)
            f0 = frames_ref[0].copy()
            if boxes_ref and boxes_ref[0] is not None:
                for b in boxes_ref[0]:
                    x1,y1,x2,y2 = [int(v) for v in b]
                    cv.rectangle(f0, (x1,y1), (x2,y2), (0,255,0), 2)
            cv.imwrite(os.path.join(dbg_dir, "ref_frame0_box.jpg"), f0)
            if out[0] is not None:
                m = (out[0].astype(np.uint8)*255)
                cv.imwrite(os.path.join(dbg_dir, "ref_frame0_mask.jpg"), m)
            _log("saved visualizations to", dbg_dir)
        except Exception:
            _log("viz save error:", traceback.format_exc())

    return out
