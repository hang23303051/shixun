#!/usr/bin/env bash
set -euo pipefail

# ---------------- 基本路径 ----------------
# 仓库根目录（假定当前文件在 Ref4D-VideoBench/scripts/ 下）
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 参考侧语义证据（你已经预生成好的 JSON）
REF_EVI_DIR="${ROOT}/data/metadata/semantic_evidence"

# 默认待评测视频根目录（用户自己放到 data/genvideo/ 下）
GEN_VIDEO_ROOT="${ROOT}/data/genvideo"

# 输出目录统一到 outputs/semantic/ 下
GEN_EVI_ROOT="${ROOT}/outputs/semantic/evidence_gen"
SCORES_OUT="${ROOT}/outputs/semantic/scores"

# MiniCPM-V-4_5 本地权重路径（用户只要按 README 软链到 checkpoints/ 下）
MINICPM_CKPT="${ROOT}/checkpoints/minicpm-v-4_5"

# 可选：环境变量控制要评测的模型子目录（逗号分隔，留空=全评）
INCLUDE_MODELS="${INCLUDE_MODELS:-}"
EXCLUDE_MODELS="${EXCLUDE_MODELS:-}"

# 可选：GPU 配置（默认为 auto，由 Python 脚本自己检测）
GPUS="${GPUS:-auto}"

# ---------------- 解析简单命令行参数 ----------------
# 支持：
#   --gen-root <path>         指定生成视频根目录（默认 data/genvideo）
#   --use-example             改为用 data/example_models/ 做演示
#   --include-models <comma>  只评这些模型（逗号分隔）
#   --exclude-models <comma>  排除这些模型（逗号分隔）
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gen-root)
      GEN_VIDEO_ROOT="$2"
      shift 2
      ;;
    --use-example)
      GEN_VIDEO_ROOT="${ROOT}/data/example_models"
      shift 1
      ;;
    --include-models)
      INCLUDE_MODELS="$2"
      shift 2
      ;;
    --exclude-models)
      EXCLUDE_MODELS="$2"
      shift 2
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      echo "Usage: $0 [--gen-root PATH | --use-example] [--include-models name1,name2] [--exclude-models name3]" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${GEN_EVI_ROOT}" "${SCORES_OUT}"

# ---------------- 对用户展示的最小输出 ----------------
echo "=========== Ref4D Semantic Evaluation ==========="
echo "[INPUT ] Ref evidence dir : ${REF_EVI_DIR}"
echo "[INPUT ] Gen video root   : ${GEN_VIDEO_ROOT}"
echo "[OUTPUT] Gen evidence dir : ${GEN_EVI_ROOT}"
echo "[OUTPUT] Score dir        : ${SCORES_OUT}"
echo "[MODEL ] MiniCPM-V-4_5    : ${MINICPM_CKPT}"
if [[ -n "${INCLUDE_MODELS}" ]]; then
  echo "[FILTER] Include models   : ${INCLUDE_MODELS}"
elif [[ -n "${EXCLUDE_MODELS}" ]]; then
  echo "[FILTER] Exclude models   : ${EXCLUDE_MODELS}"
else
  echo "[FILTER] Eval all models under ${GEN_VIDEO_ROOT}"
fi
echo "================================================="
echo

# ---------------- 调用 Python 总控脚本 ----------------
python -m ref4d_eval.semantic.semantics_evi_score_dist \
  --evi-extract-py   "${ROOT}/ref4d_eval/semantic/evidence_extract/evi_extract.py" \
  --batch-scoring-py "${ROOT}/ref4d_eval/semantic/softalign/batch_scoring.py" \
  --softalign-yaml   "${ROOT}/ref4d_eval/semantic/softalign/softalign.yaml" \
  --ref-out-dir      "${REF_EVI_DIR}" \
  --gen-video-root   "${GEN_VIDEO_ROOT}" \
  --gen-out-root     "${GEN_EVI_ROOT}" \
  --model-local-path "${MINICPM_CKPT}" \
  --scores-out-dir   "${SCORES_OUT}" \
  --gpus             "${GPUS}" \
  --include-models   "${INCLUDE_MODELS}" \
  --exclude-models   "${EXCLUDE_MODELS}" \
  --steps            both \
  --quiet

echo
echo "[INFO] Semantic evaluation finished."
echo "[INFO] GEN evidence JSON under : ${GEN_EVI_ROOT}"
echo "[INFO] Scores CSV under        : ${SCORES_OUT}"
