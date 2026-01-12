# -*- coding: utf-8 -*-
"""
批量测试：ref.mode = 'video' 与 ref.mode = 'cached' 的评测结果是否一致

思路：
- 从 data/refvideo/*/*.mp4 扫描出若干 sample_id（通过 _scan_ref_legacy）
- 对每个 sample_id：
    ref_path_true = 真正的参考视频路径
    gen_path      = ref_path_true        # 把参考视频当作“生成视频”进行自测
- 分别用：
    a) cfg_video  : ref.mode='video'  + dataset.format='legacy'
    b) cfg_cached : ref.mode='cached' + dataset.format='ref4d' + cache_root
  调用 _atomic_from_pair，比较 D_* / S_* / FRZ / S_motion / S_motion_w_frz 等是否一致。

前提：
- 对选中的 sample_id，data/metadata/motion_ref/rrm_448x8 下已有对应的 *.npz，
  且里面包含 hof / s / phi / frz_meta 等字段（由预计算脚本生成）。
"""

from __future__ import annotations
import os
import copy
from typing import Dict, Any, List

import numpy as np
import yaml

# 引入评测主逻辑
from ref4d_eval.motion.run_batch_rrm import _atomic_from_pair
# 复用你在预计算脚本里的扫描函数
from ref4d_eval.motion.rrm.precompute_ref_motion import _scan_ref_legacy


# ----------------- 小工具 -----------------

def _load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _pretty_diff(name: str, v1: float, v2: float):
    print(f"    {name:16s}: video={v1:8.4f}, cached={v2:8.4f}, diff={abs(v1 - v2):8.4f}")


# ----------------- 主逻辑 -----------------

def main():
    # ======== 手动设置 ========
    base = "/root/autodl-tmp/Ref4D-VideoBench"
    cfg_path = os.path.join(base, "ref4d_eval/motion/configs/rrm_ref4d.yaml")
    cache_root = os.path.join(base, "data/metadata/motion_ref/rrm_448x8")

    # 批量测试使用多少个样本（用 refvideo 同时当 ref & gen）
    num_samples = 10

    # ======== 1. 扫描 refvideo，拿到 sample_id -> ref_path 的映射 ========
    sid2ref = _scan_ref_legacy(base)
    if not sid2ref:
        raise RuntimeError("没有在 data/refvideo/*/*.mp4 下找到任何参考视频，请检查路径。")

    # 只保留“既有 refvideo，又有 motion_ref 缓存 npz”的 sample
    candidates: List[str] = []
    for sid, ref_path in sid2ref.items():
        npz_path = os.path.join(cache_root, f"{sid}.npz")
        if os.path.isfile(ref_path) and os.path.isfile(npz_path):
            candidates.append(sid)

    if not candidates:
        raise RuntimeError(
            f"在 data/refvideo/*/*.mp4 和 {cache_root} 下都找不到同时存在的视频+缓存的样本。"
        )

    candidates = sorted(candidates)
    sel_sids = candidates[: min(num_samples, len(candidates))]

    print(f"[INFO] 共发现 {len(candidates)} 个有缓存的样本，选取前 {len(sel_sids)} 个用于批量测试。")
    for sid in sel_sids:
        print(f"  - {sid}")

    # ======== 2. 读取基础配置，构造 video / cached 两个变体 ========
    cfg_base = _load_cfg(cfg_path)

    # a) video 模式：从真实 refvideo 路径计算参考端
    cfg_video = copy.deepcopy(cfg_base)
    cfg_video.setdefault("dataset", {})["format"] = "legacy"  # 让 _atomic_from_pair 走 ref_path 真路径逻辑
    cfg_video.setdefault("ref", {})
    cfg_video["ref"]["mode"] = "video"
    cfg_video["ref"]["cache_root"] = None

    # b) cached 模式：只从缓存读取参考端（开源时用的设定）
    cfg_cached = copy.deepcopy(cfg_base)
    cfg_cached.setdefault("dataset", {})["format"] = "ref4d"  # 与 run_batch_rrm 中 ref4d 模式一致
    cfg_cached.setdefault("ref", {})
    cfg_cached["ref"]["mode"] = "cached"
    cfg_cached["ref"]["cache_root"] = cache_root

    # ======== 3. 批量跑 _atomic_from_pair（ref 当 gen）并比较 ========
    cache_ref_video: Dict[str, dict] = {}
    cache_ref_cached: Dict[str, dict] = {}

    keys_main = [
        "D_dir", "D_mag", "D_smo",
        "S_dir", "S_mag", "S_smo",
        "FRZ", "S_motion", "S_motion_w_frz",
    ]

    max_diff = {k: 0.0 for k in keys_main}
    any_large_diff = False
    tol = 1e-6  # 理论上应该完全相等，这里留一个很小容差用于 sanity

    for sid in sel_sids:
        ref_path_true = sid2ref[sid]
        # 把 refvideo 当作“生成视频”进行自测
        gen_path = ref_path_true

        print("\n[CASE] sample_id =", sid)
        print("       ref_path_true =", ref_path_true)
        print("       gen_path      = (same as ref_path_true)")

        # 3.1 video 模式
        out_video = _atomic_from_pair(
            ref_path_true,  # 真正的 mp4 路径
            gen_path,
            cfg_video,
            cache_ref_video,
        )

        # 3.2 cached 模式
        out_cached = _atomic_from_pair(
            sid,           # 注意：这里传 sample_id，占位符
            gen_path,
            cfg_cached,
            cache_ref_cached,
        )

        # 3.3 对比关键指标
        print("  [COMPARE] key metrics (video vs cached):")
        for k in keys_main:
            v1 = float(out_video[k])
            v2 = float(out_cached[k])
            d = abs(v1 - v2)
            _pretty_diff(k, v1, v2)
            max_diff[k] = max(max_diff[k], d)
            if d > tol:
                any_large_diff = True

    # ======== 4. 汇总整体最大的差值 ========
    print("\n[SUMMARY] Max |video - cached| diff over {} samples:".format(len(sel_sids)))
    for k in keys_main:
        print(f"  {k:16s}: max_diff={max_diff[k]:.8f}")

    if any_large_diff:
        print("\n[WARN] 存在大于 tol={} 的差异，请检查上面对应样本的明细输出。".format(tol))
    else:
        print("\n[OK] 所有选定样本在上述关键指标上 video/cached 完全对齐（在 tol={} 内）。".format(tol))


if __name__ == "__main__":
    main()
