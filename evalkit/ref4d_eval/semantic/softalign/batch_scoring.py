#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量算分（缓存复用 + 过滤 NaN）
- 支持单模型目录或多模型目录（--multi-model）
- 已有且有效的报告将直接复用（可用 --force 强制重算）
- 输出：
  * 每模型：scores_<model>_<ts>.csv（仅有效分） + reports/*.json（完整报告/错误/缺失）
  * 总表：scores_all_models_<ts>.csv
          列：modelname,sample_id,catcov,aic,par,s_base,ref_obj_cnt,ref_attr_cnt  ← 新增两列
"""

import argparse
import sys
from pathlib import Path
import json
from datetime import datetime
import math

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from softalign.api import build_api


def iter_models(gen_root: Path, multi_model: bool):
    if not multi_model:
        if not list(gen_root.glob("*.json")):
            raise FileNotFoundError(
                f"[单模型模式] {gen_root} 下没有 *.json；如需多模型，请加 --multi-model 并将 --gen-dir 指向包含子目录的父目录。"
            )
        yield gen_root.name, gen_root
        return

    any_found = False
    for sub in sorted([p for p in gen_root.iterdir() if p.is_dir()]):
        if list(sub.glob("*.json")):
            any_found = True
            yield sub.name, sub
    if not any_found:
        raise FileNotFoundError(f"[多模型模式] {gen_root} 的子目录里未找到任何 *.json。")


def is_valid_score(report) -> bool:
    vals = [
        getattr(report.axis, "catcov", float("nan")),
        getattr(report.axis, "aic", float("nan")),
        getattr(report.axis, "par", float("nan")),
        getattr(report, "s_base", float("nan")),
    ]
    for v in vals:
        try:
            if not isinstance(v, (int, float)) or math.isnan(float(v)):
                return False
        except Exception:
            return False
    return True


def load_existing_report(report_path: Path):
    if not report_path.exists():
        return False, None
    try:
        data = json.load(open(report_path, "r", encoding="utf-8"))
    except Exception:
        return False, None

    axis = data.get("axis", {})
    try:
        catcov = float(axis.get("catcov", float("nan")))
        aic = float(axis.get("aic", float("nan")))
        par = float(axis.get("par", float("nan")))
        s_base = float(data.get("s_base", float("nan")))
        if any(math.isnan(x) for x in (catcov, aic, par, s_base)):
            return False, data
    except Exception:
        return False, data

    return True, data


# ----------------------- 新增：从参考 JSON 解析对象数与属性数 -----------------------
def parse_ref_counts(ref_path: Path):
    """
    返回 (ref_obj_cnt, ref_attr_cnt)
    - 对象数：优先 views.objects_count 求和；否则用 fine.entities 的长度
    - 属性数：优先 views.attributes 里所有 (属性类别, 属性值) 去重的个数；
             否则遍历 fine.entities[*].attributes 的 (类别, 值) 去重个数
    """
    obj_cnt = 0
    attr_cnt = 0
    try:
        data = json.load(open(ref_path, "r", encoding="utf-8"))
    except Exception:
        return obj_cnt, attr_cnt

    # 对象数
    try:
        views = data.get("views", {})
        oc = views.get("objects_count", None)
        if isinstance(oc, dict) and oc:
            obj_cnt = int(sum(int(v) for v in oc.values() if isinstance(v, (int, float))))
        else:
            # fallback: fine.entities
            fine = data.get("fine", {})
            ents = fine.get("entities", [])
            if isinstance(ents, list):
                obj_cnt = int(len(ents))
    except Exception:
        pass

    # 属性数（(类别, 值) 去重）
    try:
        seen = set()
        attrs = views.get("attributes", None) if isinstance(views, dict) else None
        if isinstance(attrs, dict) and attrs:
            # attrs: {obj_name: {attr_category: [values...]}}
            for _obj, adict in attrs.items():
                if isinstance(adict, dict):
                    for cat, vals in adict.items():
                        if isinstance(vals, list):
                            for val in vals:
                                if isinstance(val, str):
                                    seen.add((cat, val))
        else:
            # fallback: fine.entities[*].attributes
            fine = data.get("fine", {})
            ents = fine.get("entities", [])
            if isinstance(ents, list):
                for e in ents:
                    ed = e.get("attributes", {})
                    if isinstance(ed, dict):
                        for cat, vals in ed.items():
                            if isinstance(vals, list):
                                for val in vals:
                                    if isinstance(val, str):
                                        seen.add((cat, val))
        attr_cnt = int(len(seen))
    except Exception:
        pass

    return obj_cnt, attr_cnt
# ---------------------------------------------------------------------------


def write_csv_rows(
    pcsv,
    gcsv_path: Path,
    model_name: str,
    sample_id: str,
    catcov: float,
    aic: float,
    par: float,
    s_base: float,
    ref_obj_cnt: int,
    ref_attr_cnt: int,
):
    """
    - 逐模型 CSV 仍写旧列（保持兼容）
    - 总表 CSV 追加 ref_obj_cnt/ref_attr_cnt 两列
    """
    row_vals = [f"{catcov:.6f}", f"{aic:.6f}", f"{par:.6f}", f"{s_base:.6f}"]
    # per-model csv（不变）
    pcsv.write(",".join([sample_id] + row_vals) + "\n")
    # global csv（新增两列）
    with open(gcsv_path, "a", encoding="utf-8") as gcsv:
        gcsv.write(",".join([model_name, sample_id] + row_vals + [str(ref_obj_cnt), str(ref_attr_cnt)]) + "\n")


def main():
    parser = argparse.ArgumentParser(description="SoftAlign 批量算分（缓存复用 + 过滤 NaN）")
    parser.add_argument(
        "--yaml",
        type=str,
        default="/root/autodl-tmp/aiv/softalign/softalign.yaml",  # ← 新位置
        help="配置文件 softalign.yaml 路径",
    )
    parser.add_argument(
        "--ref-dir",
        type=str,
        default="/root/autodl-tmp/aiv/evi_and_scores/refvideo_evi",
        help="参考 JSON 目录",
    )
    parser.add_argument(
        "--gen-dir",
        type=str,
        default="/root/autodl-tmp/aiv/evi_and_scores/genvideo_evi/cogvideo_5b",
        help="生成 JSON 目录（单模型）或其父目录（多模型）",
    )
    parser.add_argument(
        "--multi-model",
        action="store_true",
        help="开启后将 --gen-dir 视为包含多个模型子目录的父目录",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/root/autodl-tmp/aiv/evi_and_scores/scores_out",
        help="输出根目录（将按模型名再建子目录）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅处理前 N 个样本（调试用）；0 表示不限制",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重算（忽略已有报告，覆盖写入）",
    )
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir)
    gen_dir = Path(args.gen_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if not ref_dir.exists():
        print(f"[ERR] 参考目录不存在：{ref_dir}", file=sys.stderr)
        sys.exit(1)
    if not gen_dir.exists():
        print(f"[ERR] 生成目录不存在：{gen_dir}", file=sys.stderr)
        sys.exit(1)

    api = build_api(args.yaml)

    ref_list = sorted(ref_dir.glob("*.json"))
    if args.limit > 0:
        ref_list = ref_list[: args.limit]
    if not ref_list:
        print(f"[ERR] 参考目录没有 *.json：{ref_dir}", file=sys.stderr)
        sys.exit(1)

    # 预读一遍参考 JSON，建立 sample_id -> (obj_cnt, attr_cnt) 映射，避免在多模型下重复解析
    ref_counts = {}
    for ref_path in ref_list:
        sample_id = ref_path.name[:-5]
        ref_counts[sample_id] = parse_ref_counts(ref_path)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    global_csv = out_root / f"scores_all_models_{ts}.csv"
    with open(global_csv, "w", encoding="utf-8") as gcsv:
        # 总表新增两列
        gcsv.write(",".join(["modelname", "sample_id", "catcov", "aic", "par", "s_base", "ref_obj_cnt", "ref_attr_cnt"]) + "\n")

    for model_name, model_path in iter_models(gen_dir, args.multi_model):
        print(f"\n=== 模型：{model_name} ===")
        out_dir = out_root / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        reports_dir = out_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        per_csv = out_dir / f"scores_{model_name}_{ts}.csv"
        with open(per_csv, "w", encoding="utf-8") as pcsv:
            # 逐模型 csv 不改列
            pcsv.write(",".join(["sample_id", "catcov", "aic", "par", "s_base"]) + "\n")

            iterable = ref_list
            if tqdm is not None:
                iterable = tqdm(ref_list, desc=f"{model_name}", ncols=100)

            stats = {"ok": 0, "reused": 0, "skipped": 0, "missing": 0, "error": 0, "written": 0}

            for ref_path in iterable:
                name = ref_path.name
                sample_id = name[:-5]
                gen_path = model_path / name
                report_path = reports_dir / name
                error_path = reports_dir / f"{name}.error.json"
                missing_path = reports_dir / f"{name}.missing.json"

                # 取参考视频统计（若缺失，默认 0,0）
                ref_obj_cnt, ref_attr_cnt = ref_counts.get(sample_id, (0, 0))

                if not gen_path.exists():
                    stats["missing"] += 1
                    if args.force or not missing_path.exists():
                        with open(missing_path, "w", encoding="utf-8") as fw:
                            json.dump({"error": "generated json missing", "ref": str(ref_path), "gen": str(gen_path)}, fw, ensure_ascii=False, indent=2)
                    continue

                if not args.force:
                    ok_valid, payload = load_existing_report(report_path)
                    if ok_valid:
                        stats["reused"] += 1
                        try:
                            axis = payload["axis"]
                            catcov = float(axis["catcov"])
                            aic = float(axis["aic"])
                            par = float(axis["par"])
                            s_base = float(payload["s_base"])
                            write_csv_rows(
                                pcsv, global_csv, model_name, sample_id,
                                catcov, aic, par, s_base,
                                ref_obj_cnt, ref_attr_cnt
                            )
                            stats["written"] += 1
                        except Exception:
                            pass
                        else:
                            continue

                try:
                    report = api.score_pair_from_files(str(ref_path), str(gen_path), sample_id=sample_id)
                    stats["ok"] += 1

                    with open(report_path, "w", encoding="utf-8") as fw:
                        json.dump({
                            "sample_id": report.sample_id,
                            "axis": {
                                "catcov": report.axis.catcov,
                                "aic": report.axis.aic,
                                "par": report.axis.par,
                            },
                            "s_base": report.s_base,
                            "sizes": report.sizes,
                            "details": {
                                "catcov_pairs": [d.__dict__ for d in report.details.catcov_pairs],
                                "aic_pairs": [{
                                    **{k: getattr(p, k) for k in ("r_idx","g_idx","coverage","misbind")},
                                    "coverage_by_key": [c.__dict__ for c in p.coverage_by_key],
                                    "misbind_items": [m.__dict__ for m in p.misbind_items],
                                } for p in report.details.aic_pairs],
                                "par_extra_categories": [x.__dict__ for x in report.details.par_extra_categories],
                                "par_extra_attrs": [x.__dict__ for x in report.details.par_extra_attrs],
                                "misc": report.details.misc,
                            },
                            "info": report.info,
                        }, fw, ensure_ascii=False, indent=2)

                    if is_valid_score(report):
                        write_csv_rows(
                            pcsv, global_csv, model_name, sample_id,
                            float(report.axis.catcov), float(report.axis.aic),
                            float(report.axis.par), float(report.s_base),
                            ref_obj_cnt, ref_attr_cnt
                        )
                        stats["written"] += 1
                    else:
                        stats["skipped"] += 1

                    if error_path.exists():
                        try: error_path.unlink()
                        except Exception: pass
                    if missing_path.exists():
                        try: missing_path.unlink()
                        except Exception: pass

                except Exception as e:
                    stats["error"] += 1
                    with open(error_path, "w", encoding="utf-8") as fw:
                        json.dump({"error": str(e), "ref": str(ref_path), "gen": str(gen_path)}, fw, ensure_ascii=False, indent=2)


            print(f"[{model_name}] CSV: {per_csv}")
            print(f"[all-models] CSV: {global_csv}")


if __name__ == "__main__":
    main()
