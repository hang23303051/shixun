# -*- coding: utf-8 -*-
"""
autodl-tmp/aigc_motion_eval/motion_eval/rrm/flicker_rrm.py

Flicker / Artifact Index with GMFlow (P95 Warp-Error / MAD)
- 输入：rrm_batch.csv
- 输出：新增列 ['fli_p95_warperr', 'D_fli_raw', 'fli_pairs_used', 'fli_debug']
- 仅用 GMFlow；相邻帧逐对计算；FB 一致性 + in-bounds；对输入先右下零填充到 8 的倍数，保证和 GMFlow 输出尺寸一致

新增：
- --force：默认复用已有结果（不重算）；加 --force 才覆盖重算
- 复用来源：当前 rrm_csv 里已有列，或 --out_csv / --inplace 指向的文件里已有结果

用法（支持 YAML；CLI > YAML > 默认）:
python -m motion_eval.rrm.flicker_rrm \
  --rrm_csv /root/autodl-tmp/aigc_motion_eval/runs/rrm_batch.csv \
  --weights  /root/autodl-tmp/aigc_motion_eval/weights/gmflow/gmflow_sintel-0c07dcb3.pth \
  --video_col gen \
  --gmflow_repo /root/autodl-tmp/aigc_motion_eval/third_party/gmflow \
  --cfg /root/autodl-tmp/aigc_motion_eval/configs/flicker_rrm.yaml \
  --inplace
"""
from __future__ import annotations
import os, sys, argparse
from typing import List, Tuple, Optional, Dict, Any

import cv2 as cv
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# YAML（可选）
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

# ========== 项目内：解码 / 重采样 ==========
try:
    from motion_eval.preprocess.io_video import load_video_cv2, resample_video
except Exception as e:
    raise ImportError("无法导入 motion_eval.preprocess.io_video") from e


# ========== 确保 GMFlow 可导入 ==========
def _ensure_gmflow_on_path(repo_root: Optional[str] = None) -> None:
    cand = []
    if repo_root:
        cand.append(repo_root)
    env = os.environ.get("GMFLOW_REPO") or os.environ.get("GMFLOW_HOME")
    if env:
        cand.append(env)
    cand.append("/root/autodl-tmp/aigc_motion_eval/third_party/gmflow")
    cand.append("/root/autodl-tmp/aigc_motion_eval/motion_eval/third_party/gmflow")

    tried = []
    for p in list(dict.fromkeys(cand)):
        if not os.path.isdir(p):
            tried.append(f"{p} (not dir)")
            continue
        if os.path.isdir(os.path.join(p, "gmflow")):
            add = p
        elif os.path.basename(os.path.normpath(p)) == "gmflow":
            add = os.path.dirname(p)
        else:
            add = p
        if add not in sys.path:
            sys.path.insert(0, add)
        try:
            import gmflow  # noqa: F401
            return
        except Exception as e:
            tried.append(f"{p} -> add:{add} -> {repr(e)}")
            if add in sys.path:
                try:
                    sys.path.remove(add)
                except ValueError:
                    pass
            continue
    raise ImportError("仍无法导入 gmflow。请用 --gmflow_repo 指向 GMFlow 仓库根目录或 gmflow 包目录。\n" +
                      "\n".join("  - "+t for t in tried))


# ========== 解码兜底（仅当项目自带解码失败） ==========
def _fallback_load_video_cv(path: str) -> Tuple[Optional[List[np.ndarray]], Optional[float]]:
    try:
        cap = cv.VideoCapture(path)
        if not cap.isOpened():
            return None, None
        fps = cap.get(cv.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = None
        frames = []
        ok = True
        while ok:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None:
                continue
            frames.append(frame)
        cap.release()
        if len(frames) < 2:
            return None, fps
        return frames, fps
    except Exception:
        return None, None


# ========== 工具 ==========
def _to_tensor_rgb(frames_bgr: List[np.ndarray]) -> torch.Tensor:
    ts = []
    for f in frames_bgr:
        if f is None or f.size == 0:
            continue
        if f.ndim != 3 or f.shape[2] != 3:
            f = cv.cvtColor(f, cv.COLOR_GRAY2BGR)
        rgb = cv.cvtColor(f, cv.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255.0
        ts.append(t)
    if not ts:
        return torch.empty(0, 3, 0, 0)
    return torch.stack(ts, dim=0)

def _ceil_to_multiple(x: int, m: int) -> int:
    return int((x + m - 1) // m * m)

def _pad_right_bottom(x: torch.Tensor, H_pad: int, W_pad: int) -> torch.Tensor:
    """右/下零填充到 H_pad x W_pad；x: [1,3,H,W] or [T,3,H,W]"""
    *lead, C, H, W = x.shape
    if H == H_pad and W == W_pad:
        return x
    pad = (0, W_pad - W, 0, H_pad - H)  # (left,right,top,bottom)
    return F.pad(x, pad, mode="constant", value=0.0)

def _orig_region_mask(H: int, W: int, H_pad: int, W_pad: int, device: torch.device) -> torch.Tensor:
    m = torch.zeros(1, 1, H_pad, W_pad, device=device)
    m[:, :, :H, :W] = 1.0
    return m

def _grid_from_flow(flow: torch.Tensor) -> torch.Tensor:
    B, C, H, W = flow.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=flow.device), torch.arange(W, device=flow.device), indexing="ij"
    )
    xx = xx.float().unsqueeze(0).expand(B, -1, -1)
    yy = yy.float().unsqueeze(0).expand(B, -1, -1)
    x2 = xx + flow[:, 0]
    y2 = yy + flow[:, 1]
    gx = 2.0 * (x2 / max(W - 1, 1.0)) - 1.0
    gy = 2.0 * (y2 / max(H - 1, 1.0)) - 1.0
    return torch.stack([gx, gy], dim=-1)  # [B,H,W,2]

def flow_warp_image(x: torch.Tensor, flow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回 (warp图, inbounds掩码)，二者尺寸与 flow 对齐"""
    grid = _grid_from_flow(flow)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    inb = (grid[..., 0].abs() <= 1.0) & (grid[..., 1].abs() <= 1.0)
    return out, inb.unsqueeze(1).float()

def fb_consistency_mask(fwd_flow: torch.Tensor, bwd_flow: torch.Tensor,
                        alpha1: float = 0.01, alpha2: float = 1.5) -> Tuple[torch.Tensor, torch.Tensor]:
    Bw, inb = flow_warp_image(bwd_flow, fwd_flow)
    diff = (fwd_flow + Bw).pow(2).sum(dim=1, keepdim=True)
    mag = (fwd_flow.pow(2) + Bw.pow(2)).sum(dim=1, keepdim=True)
    valid_fb = (diff <= alpha1 * mag + alpha2).float() * inb
    return valid_fb, inb


@torch.no_grad()
def gmflow_once(model, img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
    out = model(
        img0, img1,
        attn_splits_list=[2],
        corr_radius_list=[-1],
        prop_radius_list=[-1],
        pred_bidir_flow=False,
    )
    flow_preds = out.get("flow_preds", None)
    if isinstance(flow_preds, (list, tuple)) and len(flow_preds) > 0:
        return flow_preds[-1]
    return out["flow"]


@torch.no_grad()
def compute_flicker_p95(
    model,
    frames_bgr: List[np.ndarray],
    fps: int = 12,
    short_side: int = 448,
    src_fps: Optional[float] = None,
    device: torch.device = torch.device("cuda"),
    min_valid_pix: int = 1,
    min_valid_ratio: float = 0.0,
    use_median_mad: bool = False,
    alpha1: float = 0.01,
    alpha2: float = 1.5,
    allow_relax_fb: bool = True,
    pad_multiple: int = 8,          # 补齐到 8 的倍数（与 GMFlow 对齐）
) -> Tuple[float, int, Dict[str, Any]]:
    dbg = {"eff_fps": None, "T": 0, "H": None, "W": None,
           "Hp": None, "Wp": None, "pairs": 0, "skipped_pairs": 0, "reasons": []}

    if frames_bgr is None or not isinstance(frames_bgr, list) or len(frames_bgr) < 2:
        dbg["reasons"].append("frames<2")
        return float("nan"), 0, dbg

    eff_fps = fps
    if (src_fps is not None) and (src_fps > 0):
        try:
            eff_fps = int(max(1, min(int(round(src_fps)), int(fps))))
        except Exception:
            eff_fps = fps
    dbg["eff_fps"] = eff_fps

    frames_sr = resample_video(frames_bgr, short_side=short_side, fps=eff_fps, src_fps=src_fps)
    if frames_sr is None or len(frames_sr) < 2:
        dbg["reasons"].append("resample<2")
        return float("nan"), 0, dbg

    imgs = _to_tensor_rgb(frames_sr).to(device)     # [T,3,H,W]
    T, _, H, W = imgs.shape
    dbg["T"], dbg["H"], dbg["W"] = int(T), int(H), int(W)
    if T < 2:
        dbg["reasons"].append("tensor<2")
        return float("nan"), 0, dbg

    # ---- 关键：预填充到 8 的倍数 ----
    Hp = _ceil_to_multiple(H, pad_multiple)
    Wp = _ceil_to_multiple(W, pad_multiple)
    dbg["Hp"], dbg["Wp"] = int(Hp), int(Wp)
    imgs_p = _pad_right_bottom(imgs, Hp, Wp)        # [T,3,Hp,Wp]
    orig_mask = _orig_region_mask(H, W, Hp, Wp, device)  # [1,1,Hp,Wp]

    min_valid = max(int(min_valid_pix), int(min_valid_ratio * H * W))
    errs = []

    for t in range(1, T):
        img0 = imgs_p[t - 1].unsqueeze(0)  # [1,3,Hp,Wp]
        img1 = imgs_p[t].unsqueeze(0)

        fwd = gmflow_once(model, img0, img1)        # [1,2,Hp,Wp]
        bwd = gmflow_once(model, img1, img0)

        valid_fb, inb = fb_consistency_mask(fwd, bwd, alpha1=alpha1, alpha2=alpha2)  # 与 flow 对齐（Hp,Wp）
        valid = valid_fb * orig_mask               # 只在原图区域统计

        if valid.sum().item() < max(1, min_valid) and allow_relax_fb:
            valid = inb * orig_mask                # 放宽到 in-bounds + 原图区域

        if valid.sum().item() < max(1, min_valid):
            dbg["skipped_pairs"] += 1
            continue

        img0_warp, _ = flow_warp_image(img0, fwd)   # [1,3,Hp,Wp]
        diff = (img1 - img0_warp).abs() * valid     # [1,3,Hp,Wp]

        if use_median_mad:
            sel = (valid > 0).expand_as(diff)
            mad = diff[sel].median().item()
        else:
            denom = valid.sum().clamp(min=1.0) * 3.0
            mad = (diff.sum() / denom).item()

        errs.append(float(mad))
        dbg["pairs"] += 1

    if not errs:
        dbg["reasons"].append("no_valid_pairs")
        return float("nan"), 0, dbg

    D_fli_raw = float(np.nanpercentile(np.asarray(errs, dtype=np.float32), 95))
    return D_fli_raw, len(errs), dbg


def _pick_video_col(df: pd.DataFrame, user_col: Optional[str]) -> str:
    if user_col:
        if user_col not in df.columns:
            raise KeyError(f"指定的视频列 {user_col} 不在 rrm_batch 中")
        return user_col
    for c in ["gen_path", "gen", "video_path", "gen_video", "video", "path_gen", "path"]:
        if c in df.columns:
            return c
    raise KeyError("无法自动识别生成端视频路径列，请用 --video_col 指定。")


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if not _HAS_YAML:
        raise ImportError("未安装 pyyaml：请 pip install pyyaml 或删除 --cfg 参数")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("--cfg 内容不是字典")
    return cfg


def parse_args():
    ap = argparse.ArgumentParser()
    # 运行/IO
    ap.add_argument("--rrm_csv", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video_col", default=None)
    ap.add_argument("--gmflow_repo", default="/root/autodl-tmp/aigc_motion_eval/third_party/gmflow")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--max_videos", type=int, default=-1)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--force", action="store_true", help="强制覆盖重算 Flicker，即使已有结果也重跑")

    # 模型/采样/阈值
    ap.add_argument("--short_side", type=int, default=448)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--min_valid_pix", type=int, default=1)
    ap.add_argument("--min_valid_ratio", type=float, default=0.0)
    ap.add_argument("--use_median_mad", action="store_true")
    ap.add_argument("--alpha1", type=float, default=0.01)
    ap.add_argument("--alpha2", type=float, default=1.5)
    ap.add_argument("--allow_relax_fb", action="store_true", default=True)
    ap.add_argument("--pad_multiple", type=int, default=8)   # 与 GMFlow 补齐策略一致

    # YAML 配置（可选）
    ap.add_argument("--cfg", type=str, default=None, help="YAML 配置文件路径（CLI 覆盖 YAML）")
    return ap.parse_args()


def _apply_yaml(args: argparse.Namespace) -> argparse.Namespace:
    if not args.cfg:
        return args
    yml = _load_yaml(args.cfg)
    for k, v in yml.items():
        if hasattr(args, k):
            setattr(args, k, v)
    return args


def _safe_load_frames(path: str) -> Tuple[Optional[List[np.ndarray]], Optional[float], str]:
    try:
        frames, fps_src = load_video_cv2(path, bgr=True, return_fps=True)
    except Exception:
        frames, fps_src = None, None
    if (frames is None) or (not isinstance(frames, list)) or (len(frames) < 2):
        fb_frames, fb_fps = _fallback_load_video_cv(path)
        if fb_frames is not None and len(fb_frames) >= 2:
            return fb_frames, fb_fps, "fallback_ok"
        return None, None, "decode_fail"
    return frames, fps_src, "ok"


def _build_key_series(df: pd.DataFrame, vid_col: str) -> pd.Series:
    """优先用 (modelname, sample_id) 作为键；否则退化为视频路径键。"""
    if ("modelname" in df.columns) and ("sample_id" in df.columns):
        return df["modelname"].astype(str) + "||" + df["sample_id"].astype(str)
    return df[vid_col].astype(str)


def _load_prev_results_if_any(out_csv_path: Optional[str]) -> Optional[pd.DataFrame]:
    if out_csv_path and os.path.isfile(out_csv_path):
        try:
            return pd.read_csv(out_csv_path)
        except Exception:
            return None
    return None


def main():
    args = parse_args()
    args = _apply_yaml(args)

    # 1) GMFlow
    _ensure_gmflow_on_path(args.gmflow_repo)
    from gmflow.gmflow import GMFlow  # noqa

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    model = GMFlow(
        num_scales=1,
        upsample_factor=8,
        feature_channels=128,
        attention_type="swin",
        num_transformer_layers=6,
        num_head=1,
    )
    ckpt = torch.load(args.weights, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device).eval()

    # 2) 读取 rrm_batch
    df = pd.read_csv(args.rrm_csv)
    vid_col = _pick_video_col(df, args.video_col)
    key_series = _build_key_series(df, vid_col)

    # 2.1 目标输出路径
    out_path = args.rrm_csv if args.inplace else (args.out_csv or (os.path.splitext(args.rrm_csv)[0] + "_with_fli.csv"))

    # 2.2 若存在历史 out_csv（或 inplace 文件），加载并将已完成结果“映射/复用”到当前 df
    prev_df = None
    if (not args.force) and os.path.isfile(out_path):
        prev_df = _load_prev_results_if_any(out_path)
        if prev_df is not None and not prev_df.empty:
            # 建立键
            try:
                prev_vid_col = _pick_video_col(prev_df, args.video_col if args.video_col in prev_df.columns else None)
            except Exception:
                # 回退：若 prev_df 不含视频列，尝试 (modelname, sample_id)
                prev_vid_col = vid_col if vid_col in prev_df.columns else None
            prev_key = _build_key_series(prev_df, prev_vid_col) if prev_vid_col else None

            # 把 prev 中已有列合入 df（仅缺失时填充）
            out_cols = ["fli_p95_warperr", "D_fli_raw", "fli_pairs_used", "fli_debug"]
            if prev_key is not None:
                prev_map = prev_df.assign(__key=prev_key)[["__key"] + [c for c in out_cols if c in prev_df.columns]]
                df = df.assign(__key=key_series).merge(prev_map, on="__key", how="left", suffixes=("", "_prev"))
                for c in out_cols:
                    if (c not in df.columns) and (c + "_prev" in df.columns):
                        df[c] = df[c + "_prev"]
                    elif (c in df.columns) and (c + "_prev" in df.columns):
                        # 仅在 df[c] 缺失时用历史值补
                        df[c] = df[c].where(df[c].notna(), df[c + "_prev"])
                # 清理临时列
                keep = [c for c in df.columns if not c.endswith("_prev") and c != "__key"]
                df = df[keep]

    # 3) 初始化结果列（若不存在则填充 NaN/0）
    if "fli_p95_warperr" not in df.columns: df["fli_p95_warperr"] = np.nan
    if "D_fli_raw" not in df.columns:       df["D_fli_raw"]       = np.nan
    if "fli_pairs_used" not in df.columns:  df["fli_pairs_used"]  = 0
    if "fli_debug" not in df.columns:       df["fli_debug"]       = ""

    # 4) 建立“已完成”掩码：默认只要 D_fli_raw 非 NaN 即视为已有结果，跳过不重算
    done_mask = df["D_fli_raw"].notna() if (not args.force) else pd.Series(False, index=df.index)

    rows = df.shape[0]
    work_rows = range(rows) if args.max_videos <= 0 else range(min(rows, args.max_videos))

    for i in tqdm(work_rows, desc="GMFlow Flicker (P95 Warp-Error)"):
        if (not args.force) and bool(done_mask.iloc[i]):
            # 直接复用已存在的 df 值，不做任何计算
            continue

        vpath = str(df.iloc[i][vid_col]) if vid_col in df.columns else ""
        if not (isinstance(vpath, str) and os.path.isfile(vpath)):
            df.at[i, "fli_p95_warperr"] = np.nan
            df.at[i, "D_fli_raw"] = np.nan
            df.at[i, "fli_pairs_used"] = 0
            df.at[i, "fli_debug"] = "NOFILE"
            continue

        frames, fps_src, dec_state = _safe_load_frames(vpath)
        if frames is None or not isinstance(frames, list) or len(frames) < 2:
            df.at[i, "fli_p95_warperr"] = np.nan
            df.at[i, "D_fli_raw"] = np.nan
            df.at[i, "fli_pairs_used"] = 0
            df.at[i, "fli_debug"] = "DECODE_FAIL"
            continue

        try:
            val, used, dbg = compute_flicker_p95(
                model=model,
                frames_bgr=frames,
                fps=int(args.fps),
                short_side=int(args.short_side),
                src_fps=float(fps_src) if fps_src is not None else None,
                device=device,
                min_valid_pix=int(args.min_valid_pix),
                min_valid_ratio=float(args.min_valid_ratio),
                use_median_mad=bool(args.use_median_mad),
                alpha1=float(args.alpha1),
                alpha2=float(args.alpha2),
                allow_relax_fb=bool(args.allow_relax_fb),
                pad_multiple=int(args.pad_multiple),
            )
            reason = "OK" if used > 0 else "NO_VALID"
            debug_msg = (
                f"{reason}|dec={dec_state}|fps_src={fps_src}|eff_fps={dbg.get('eff_fps')}|"
                f"T={dbg.get('T')}|H×W={dbg.get('H')}x{dbg.get('W')}|"
                f"Hp×Wp={dbg.get('Hp')}x{dbg.get('Wp')}|pairs={dbg.get('pairs')}|"
                f"skip={dbg.get('skipped_pairs')}|notes={';'.join(dbg.get('reasons', []))}"
            )
            df.at[i, "fli_p95_warperr"] = val
            df.at[i, "D_fli_raw"] = val
            df.at[i, "fli_pairs_used"] = int(used)
            df.at[i, "fli_debug"] = debug_msg

        except AssertionError as e:
            df.at[i, "fli_p95_warperr"] = np.nan
            df.at[i, "D_fli_raw"] = np.nan
            df.at[i, "fli_pairs_used"] = 0
            df.at[i, "fli_debug"] = f"ASSERT:{repr(e)}"
        except Exception as e:
            df.at[i, "fli_p95_warperr"] = np.nan
            df.at[i, "D_fli_raw"] = np.nan
            df.at[i, "fli_pairs_used"] = 0
            df.at[i, "fli_debug"] = f"EXC:{repr(e)}"

    # 5) 写回
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[done] 写出：{out_path}")
    print("新增/维护列：['fli_p95_warperr', 'D_fli_raw', 'fli_pairs_used', 'fli_debug']")
    try:
        if "modelname" in df.columns:
            used_series = (df["fli_pairs_used"] > 0).astype(int)
            grp = pd.DataFrame({"used": used_series, "modelname": df["modelname"]}).groupby("modelname")["used"].sum()
            print("[per-model videos with pairs>0]")
            for m, c in grp.items():
                print(f"  {m}: {int(c)}")
    except Exception:
        pass
    print(f"[stats] 有效帧对>0的视频数: {(df['fli_pairs_used']>0).sum()}/{len(df)} | "
          f"fps={args.fps} short_side={args.short_side} pad_multiple={args.pad_multiple} "
          f"min_valid_pix={args.min_valid_pix} min_valid_ratio={args.min_valid_ratio} "
          f"allow_relax_fb={args.allow_relax_fb} gmflow_repo={args.gmflow_repo}")
    

if __name__ == "__main__":
    main()
