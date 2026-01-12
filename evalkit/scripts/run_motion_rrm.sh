#!/usr/bin/env bash
set -e

# ============================================================
# Ref4D-VideoBench / Motion Dimension
# 一键评测脚本（RRM-only, ref.mode=cached, dataset.format=ref4d）
#
# 前置条件：
#   1) 已创建并激活 conda 环境 ref4d_motion
#        conda activate ref4d_motion
#   2) 已运行：
#        - bash scripts/setup_motion_env.sh
#        - bash scripts/download_motion_models.sh
#   3) 已准备好：
#        - data/metadata/ref4d_meta.jsonl
#        - data/metadata/motion_ref/rrm_448x8/*.npz
#        - data/genvideo/<modelname>/<sample_id>.mp4
#
# 使用方式：
#   # 默认 3 个 worker，输出到 results/motion_rrm.csv（增量复用）
#   bash scripts/run_motion_rrm.sh
#
#   # 指定 worker 数，例如 4 个
#   WORKERS=4 bash scripts/run_motion_rrm.sh
#
#   # 强制重算（忽略已存在 CSV，全部重新评测）
#   FORCE=1 bash scripts/run_motion_rrm.sh
#
# 输出：
#   results/motion/motion_rrm.csv
#   同时生成分片日志：results/motion/motion_rrm.part*.log
# ============================================================

# 仓库根目录（假定脚本位于 scripts/ 下）
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 配置文件（RRM + ref4d + cached）
CFG_PATH="${BASE_DIR}/ref4d_eval/motion/configs/rrm_ref4d.yaml"

# 输出 CSV
RESULTS_DIR="${BASE_DIR}/outputs/motion"
OUT_CSV="${RESULTS_DIR}/motion_rrm.csv"

# worker 数，可通过环境变量覆盖：WORKERS=4 bash ...
WORKERS="${WORKERS:-3}"

# 是否强制重算：FORCE=1 bash ... → 加上 --force
FORCE_FLAG=""
if [ "${FORCE:-0}" -eq 1 ]; then
  FORCE_FLAG="--force"
fi

echo "[INFO] BASE_DIR   = ${BASE_DIR}"
echo "[INFO] CFG_PATH   = ${CFG_PATH}"
echo "[INFO] OUT_CSV    = ${OUT_CSV}"
echo "[INFO] WORKERS    = ${WORKERS}"
echo "[INFO] FORCE_FLAG = ${FORCE_FLAG:-<none>}"

# ------------------------------------------------------------
# 环境与目录检查（尽量在跑之前就把坑报清楚）
# ------------------------------------------------------------

# 1) 检查是否在 ref4d_motion 环境中（只做提示，不阻止）
if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
  echo "[INFO] Current conda env: ${CONDA_DEFAULT_ENV}"
  if [ "${CONDA_DEFAULT_ENV}" != "ref4d_motion" ]; then
    echo "[WARN] 当前环境不是 ref4d_motion，若你另起了名字可以忽略；"
    echo "       若还没激活 ref4d_motion，请先执行："
    echo "         conda activate ref4d_motion"
  fi
else
  echo "[WARN] 检测不到 conda 环境（CONDA_DEFAULT_ENV 未设置），"
  echo "       请确认你已在 ref4d_motion 环境中运行该脚本。"
fi

# 2) 检查配置文件
if [ ! -f "${CFG_PATH}" ]; then
  echo "[ERROR] 找不到配置文件：${CFG_PATH}"
  echo "        请确认 ref4d_eval/motion/configs/rrm_ref4d.yaml 是否存在。"
  exit 1
fi

# 3) 检查 ref4d meta
META_PATH="${BASE_DIR}/data/metadata/ref4d_meta.jsonl"
if [ ! -f "${META_PATH}" ]; then
  echo "[ERROR] 找不到 meta 文件：${META_PATH}"
  echo "        该文件描述 sample_id / theme 等信息，是 ref4d 模式的必需文件。"
  exit 1
fi

# 4) 检查 motion_ref 缓存目录
MOTION_REF_DIR="${BASE_DIR}/data/metadata/motion_ref/rrm_448x8"
if [ ! -d "${MOTION_REF_DIR}" ]; then
  echo "[ERROR] 找不到参考侧 RRM 缓存目录：${MOTION_REF_DIR}"
  echo "        请先运行你的预计算脚本，或按 README 获取预计算好的 .npz。"
  exit 1
fi

# 5) 检查是否有生成视频
GEN_ROOT="${BASE_DIR}/data/genvideo"
if [ ! -d "${GEN_ROOT}" ]; then
  echo "[ERROR] 找不到生成视频根目录：${GEN_ROOT}"
  echo "        期望目录结构：data/genvideo/<modelname>/<sample_id>.mp4"
  exit 1
fi

# 至少检查一个模型目录里有 mp4
HAS_MP4=0
if ls "${GEN_ROOT}"/*/*.mp4 >/dev/null 2>&1; then
  HAS_MP4=1
fi

if [ "${HAS_MP4}" -ne 1 ]; then
  echo "[ERROR] data/genvideo 下没有找到任何 .mp4 生成视频。"
  echo "        请先把待评测模型的结果放到："
  echo "          data/genvideo/<your_modelname>/<sample_id>.mp4"
  exit 1
fi

# 创建结果目录
mkdir -p "${RESULTS_DIR}"

# ------------------------------------------------------------
# 运行评测
# ------------------------------------------------------------
echo ""
echo "[RUN] Start RRM batch evaluation..."
echo "      python -m ref4d_eval.motion.run_batch_rrm \\"
echo "          --cfg \"${CFG_PATH}\" \\"
echo "          --base \"${BASE_DIR}\" \\"
echo "          --out \"${OUT_CSV}\" \\"
echo "          --workers ${WORKERS} ${FORCE_FLAG}"

python -m ref4d_eval.motion.run_batch_rrm \
  --cfg "${CFG_PATH}" \
  --base "${BASE_DIR}" \
  --out "${OUT_CSV}" \
  --workers "${WORKERS}" \
  ${FORCE_FLAG}

echo ""
echo "[DONE] RRM evaluation finished."
echo "       Result CSV: ${OUT_CSV}"
echo "       若你再次运行且未设置 FORCE=1，将在原 CSV 基础上增量复用已算结果。"
