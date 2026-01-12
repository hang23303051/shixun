#!/usr/bin/env bash
set -euo pipefail

# =========================
# Ref4D-VideoBench - World Knowledge (One-Click)
# - batch evaluate: data/genvideo/<model> vs data/metadata/world_QA
# - outputs: outputs/world_knowledge/<model>/{*.csv, <cls>/*.detail.json}
# - env: reuse semantic env (MiniCPM-V-4.5 etc.)
# =========================

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_world_knowledge.sh [options]

Options:
  --base <path>            Repo root. (default: auto-detect from script location)
  --gen-root <path>        Generated videos root. (default: $BASE/data/genvideo)
  --bank-dir <path>        World QA bank dir. (default: $BASE/data/metadata/world_QA)
  --out-root <path>        Output root. (default: $BASE/outputs/world_knowledge)

  --model <name>           Only evaluate one model subdir under gen-root.
  --all                    Evaluate all model subdirs (default behavior).

  --model-local <path>     Local model path (directory). If set, will use --local-path + --local-files-only.
  --model-id <id>          HF model id. (default: openbmb/MiniCPM-V-4_5)

  --device <cuda|cpu>      (default: cuda)
  --dtype <bf16|fp16|fp32> (default: bf16)

  --fps <int>              (default: 3)
  --cap-frames <int>       (default: 300)
  --resize-short <int>     (default: 448)
  --max-new-tokens <int>   (default: 256)
  --temperature <float>    (default: 0.0)
  --enable-thinking        Enable thinking mode (if model supports).
  --decode-backend <auto|cv2|decord> (default: auto)

  --cuda <id>              Set CUDA_VISIBLE_DEVICES=<id> (optional)
  --force                  Remove per-model out dir before running.
  -h, --help               Show this help.

Examples:
  # evaluate all models
  bash scripts/run_world_knowledge.sh

  # evaluate one model only
  bash scripts/run_world_knowledge.sh --model Wan2.1

  # use local model weights (offline)
  bash scripts/run_world_knowledge.sh --model Wan2.1 --model-local /root/autodl-tmp/models/MiniCPM-V-4_5
EOF
}

# ---------------- defaults ----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "${SCRIPT_DIR}/.." && pwd)"

GEN_ROOT=""
BANK_DIR=""
OUT_ROOT=""

MODEL_NAME=""
RUN_ALL=1

MODEL_LOCAL="/root/autodl-tmp/Ref4D-VideoBench/checkpoints/minicpm-v-4_5"
MODEL_ID="openbmb/MiniCPM-V-4_5"

DEVICE="cuda"
DTYPE="bf16"

FPS=3
CAP_FRAMES=300
RESIZE_SHORT=448
MAX_NEW_TOKENS=256
TEMPERATURE=0.0
ENABLE_THINKING=0
DECODE_BACKEND="auto"

CUDA_ID=""
FORCE=0

# ---------------- parse args ----------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE="$2"; shift 2;;
    --gen-root) GEN_ROOT="$2"; shift 2;;
    --bank-dir) BANK_DIR="$2"; shift 2;;
    --out-root) OUT_ROOT="$2"; shift 2;;

    --model) MODEL_NAME="$2"; RUN_ALL=0; shift 2;;
    --all) RUN_ALL=1; MODEL_NAME=""; shift 1;;

    --model-local) MODEL_LOCAL="$2"; shift 2;;
    --model-id) MODEL_ID="$2"; shift 2;;

    --device) DEVICE="$2"; shift 2;;
    --dtype) DTYPE="$2"; shift 2;;

    --fps) FPS="$2"; shift 2;;
    --cap-frames) CAP_FRAMES="$2"; shift 2;;
    --resize-short) RESIZE_SHORT="$2"; shift 2;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2;;
    --temperature) TEMPERATURE="$2"; shift 2;;
    --enable-thinking) ENABLE_THINKING=1; shift 1;;
    --decode-backend) DECODE_BACKEND="$2"; shift 2;;

    --cuda) CUDA_ID="$2"; shift 2;;
    --force) FORCE=1; shift 1;;

    -h|--help) usage; exit 0;;
    *) echo "[Error] Unknown arg: $1"; usage; exit 1;;
  esac
done

# ---------------- fill remaining defaults ----------------
GEN_ROOT="${GEN_ROOT:-${BASE}/data/genvideo}"
BANK_DIR="${BANK_DIR:-${BASE}/data/metadata/world_QA}"
OUT_ROOT="${OUT_ROOT:-${BASE}/outputs/world_knowledge}"
EVAL_PY="${BASE}/ref4d_eval/world_knowledge/evaluate_bank.py"

# ---------------- sanity checks ----------------
if [[ ! -f "${EVAL_PY}" ]]; then
  echo "[Error] evaluate_bank.py not found: ${EVAL_PY}"
  exit 1
fi
if [[ ! -d "${GEN_ROOT}" ]]; then
  echo "[Error] gen-root not found: ${GEN_ROOT}"
  exit 1
fi
if [[ ! -d "${BANK_DIR}" ]]; then
  echo "[Error] bank-dir not found: ${BANK_DIR}"
  exit 1
fi
mkdir -p "${OUT_ROOT}"

# ---------------- env ----------------
export PYTHONPATH="${BASE}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6}"

if [[ -n "${CUDA_ID}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_ID}"
fi

# ---------------- build model args ----------------
MODEL_ARGS=()
if [[ -n "${MODEL_LOCAL}" ]]; then
  if [[ ! -d "${MODEL_LOCAL}" ]]; then
    echo "[Error] --model-local is set but not a directory: ${MODEL_LOCAL}"
    exit 1
  fi
  MODEL_ARGS+=( --local-path "${MODEL_LOCAL}" --local-files-only )
else
  MODEL_ARGS+=( --model-id "${MODEL_ID}" )
fi

# 注意：这里不要放 --out-dir（会被后面覆盖/置空导致批量模式不触发）
COMMON_ARGS=(
  --bank-dir "${BANK_DIR}"
  --device "${DEVICE}"
  --dtype  "${DTYPE}"
  --fps "${FPS}"
  --cap-frames "${CAP_FRAMES}"
  --resize-short "${RESIZE_SHORT}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --decode-backend "${DECODE_BACKEND}"
)
if [[ "${ENABLE_THINKING}" -eq 1 ]]; then
  COMMON_ARGS+=( --enable-thinking )
fi

run_one_model() {
  local model_dir="$1"
  local model_name
  model_name="$(basename "${model_dir}")"

  if [[ ! -d "${model_dir}" ]]; then
    echo "[Skip] not a dir: ${model_dir}"
    return 0
  fi

  local out_dir="${OUT_ROOT}/${model_name}"
  if [[ "${FORCE}" -eq 1 ]]; then
    rm -rf "${out_dir}"
  fi
  mkdir -p "${out_dir}"

  echo "============================================================"
  echo "[WorldKnowledge] model = ${model_name}"
  echo "  video-dir : ${model_dir}"
  echo "  bank-dir  : ${BANK_DIR}"
  echo "  out-dir   : ${out_dir}"
  echo "  device/dtype: ${DEVICE}/${DTYPE}  fps=${FPS} cap_frames=${CAP_FRAMES} resize_short=${RESIZE_SHORT}"
  echo "============================================================"

  # evaluate_bank.py 会在 out-dir 下写：
  # - <cls>.csv（append）
  # - <cls>/<video>.detail.json（当 --dump-per-item 打开）
  python -u "${EVAL_PY}" \
    --video-dir "${model_dir}" \
    --out-dir  "${out_dir}" \
    "${MODEL_ARGS[@]}" \
    "${COMMON_ARGS[@]}" \
    --verbose --dump-per-item

  echo "[Done] ${model_name} -> ${out_dir}"
}

# ---------------- dispatch ----------------
if [[ "${RUN_ALL}" -eq 0 ]]; then
  md="${GEN_ROOT}/${MODEL_NAME}"
  if [[ ! -d "${md}" ]]; then
    echo "[Error] model dir not found: ${md}"
    exit 1
  fi
  run_one_model "${md}"
else
  shopt -s nullglob
  model_dirs=( "${GEN_ROOT}"/* )
  if [[ ${#model_dirs[@]} -eq 0 ]]; then
    echo "[Error] no model subdirs under: ${GEN_ROOT}"
    exit 1
  fi
  for d in "${model_dirs[@]}"; do
    run_one_model "${d}"
  done
fi

echo "All done. Outputs at: ${OUT_ROOT}"
