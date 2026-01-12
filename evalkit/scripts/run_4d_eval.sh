#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Ref4D-VideoBench - 4D One-Click Evaluation
# - run: semantic + motion + event + world_knowledge
# - merge: outputs -> outputs/overall/ref4d_4d_scores.csv
#
# Output CSV header (fixed):
#   modelname,sample_id,semanticscore,motionscore,eventscore,worldscore,total_4d
# ============================================================

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_4d_eval.sh [options]

Options:
  --base <path>             Repo root (default: auto-detect)
  --gen-root <path>         Generated videos root (default: <base>/data/genvideo)

  --include-models <comma>  Only evaluate these models (comma-separated). (default: all)
  --exclude-models <comma>  Exclude these models (comma-separated).

  --semantic-env <name>     Conda env for semantic/world (default: minicpmv)
  --motion-env <name>       Conda env for motion (default: ref4d_motion)
  --event-env <name>        Conda env for event (default: env_event_all)

  --model-local <path>      Local MiniCPM-V-4.5 dir for world script
  --world-model-id <id>     HF model id for world script (default: openbmb/MiniCPM-V-4_5)

  --force                   Remove old outputs before running (best-effort)
  --skip-semantic           Skip semantic run
  --skip-motion             Skip motion run
  --skip-event              Skip event run
  --skip-world              Skip world knowledge run
  --skip-merge              Skip final merge

  -h, --help                Show this help
EOF
}

# ---------- detect base ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "${SCRIPT_DIR}/.." && pwd)"

GEN_ROOT=""
INCLUDE_MODELS=""
EXCLUDE_MODELS=""

SEMANTIC_ENV="minicpmv"
MOTION_ENV="ref4d_motion"
EVENT_ENV="env_event_all"

WORLD_MODEL_LOCAL=""
WORLD_MODEL_ID="openbmb/MiniCPM-V-4_5"

FORCE=0
SKIP_SEM=0
SKIP_MOT=0
SKIP_EVT=0
SKIP_WLD=0
SKIP_MERGE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE="$2"; shift 2;;
    --gen-root) GEN_ROOT="$2"; shift 2;;

    --include-models) INCLUDE_MODELS="$2"; shift 2;;
    --exclude-models) EXCLUDE_MODELS="$2"; shift 2;;

    --semantic-env) SEMANTIC_ENV="$2"; shift 2;;
    --motion-env) MOTION_ENV="$2"; shift 2;;
    --event-env) EVENT_ENV="$2"; shift 2;;

    --model-local) WORLD_MODEL_LOCAL="$2"; shift 2;;
    --world-model-id) WORLD_MODEL_ID="$2"; shift 2;;

    --force) FORCE=1; shift 1;;

    --skip-semantic) SKIP_SEM=1; shift 1;;
    --skip-motion) SKIP_MOT=1; shift 1;;
    --skip-event) SKIP_EVT=1; shift 1;;
    --skip-world) SKIP_WLD=1; shift 1;;
    --skip-merge) SKIP_MERGE=1; shift 1;;

    -h|--help) usage; exit 0;;
    *) echo "[Error] Unknown arg: $1"; usage; exit 1;;
  esac
done

GEN_ROOT="${GEN_ROOT:-${BASE}/data/genvideo}"

SEM_SCRIPT="${BASE}/scripts/run_semantic_eval.sh"
MOT_SCRIPT="${BASE}/scripts/run_motion_rrm.sh"
EVT_SCRIPT="${BASE}/scripts/run_event_eval.sh"
WLD_SCRIPT="${BASE}/scripts/run_world_knowledge.sh"

MERGE_PY="${BASE}/scripts/merge_4d_scores.py"
OUT_DIR="${BASE}/outputs/overall"
OUT_CSV="${OUT_DIR}/ref4d_4d_scores.csv"

# -------- sanity --------
[[ -d "${BASE}" ]] || { echo "[Error] base not found: ${BASE}"; exit 1; }
[[ -d "${GEN_ROOT}" ]] || { echo "[Error] gen-root not found: ${GEN_ROOT}"; exit 1; }

[[ -f "${SEM_SCRIPT}" ]] || { echo "[Error] missing: ${SEM_SCRIPT}"; exit 1; }
[[ -f "${MOT_SCRIPT}" ]] || { echo "[Error] missing: ${MOT_SCRIPT}"; exit 1; }
[[ -f "${EVT_SCRIPT}" ]] || { echo "[Error] missing: ${EVT_SCRIPT}"; exit 1; }
[[ -f "${WLD_SCRIPT}" ]] || { echo "[Error] missing: ${WLD_SCRIPT}"; exit 1; }
[[ -f "${MERGE_PY}" ]] || { echo "[Error] missing: ${MERGE_PY}"; exit 1; }

mkdir -p "${OUT_DIR}"
cd "${BASE}"

# -------- force cleanup (best-effort) --------
if [[ "${FORCE}" -eq 1 ]]; then
  echo "[4D] --force enabled: cleaning old outputs (best-effort)"
  rm -rf "${BASE}/outputs/semantic/scores" "${BASE}/outputs/semantic/evidence_gen" || true
  rm -f  "${BASE}/outputs/motion/motion_rrm.csv" || true
  rm -rf "${BASE}/outputs/event/scores" || true
  rm -rf "${BASE}/outputs/world_knowledge" || true
fi

echo "============================================================"
echo "[4D] base        = ${BASE}"
echo "[4D] gen-root    = ${GEN_ROOT}"
echo "[4D] include     = ${INCLUDE_MODELS:-<all>}"
echo "[4D] exclude     = ${EXCLUDE_MODELS:-<none>}"
echo "[4D] envs        = semantic/world:${SEMANTIC_ENV} | motion:${MOTION_ENV} | event:${EVENT_ENV}"
echo "[4D] out-csv     = ${OUT_CSV}"
echo "============================================================"
echo

# ---------- helper: check excluded ----------
_has_in_csv_list() {
  local name="$1"
  local csv="$2"
  [[ -z "${csv}" ]] && return 1
  IFS=',' read -r -a arr <<< "${csv}"
  for x in "${arr[@]}"; do
    x="$(echo "${x}" | xargs)"
    [[ -z "${x}" ]] && continue
    [[ "${x}" == "${name}" ]] && return 0
  done
  return 1
}

# -------- 1) semantic --------
if [[ "${SKIP_SEM}" -eq 0 ]]; then
  echo "[4D] >>> Running SEMANTIC ..."
  SEM_ARGS=( --gen-root "${GEN_ROOT}" )
  [[ -n "${INCLUDE_MODELS}" ]] && SEM_ARGS+=( --include-models "${INCLUDE_MODELS}" )
  [[ -n "${EXCLUDE_MODELS}" ]] && SEM_ARGS+=( --exclude-models "${EXCLUDE_MODELS}" )

  # 不要传空值，否则参数解析会乱
  conda run -n "${SEMANTIC_ENV}" bash "${SEM_SCRIPT}" "${SEM_ARGS[@]}"
  echo
else
  echo "[4D] >>> Skip SEMANTIC"
  echo
fi

# -------- 2) motion --------
if [[ "${SKIP_MOT}" -eq 0 ]]; then
  echo "[4D] >>> Running MOTION ..."
  # 若 motion 脚本支持 include/exclude，你也可以在 motion 脚本里读 INCLUDE_MODELS/EXCLUDE_MODELS
  INCLUDE_MODELS="${INCLUDE_MODELS}" EXCLUDE_MODELS="${EXCLUDE_MODELS}" \
    conda run -n "${MOTION_ENV}" bash "${MOT_SCRIPT}"
  echo
else
  echo "[4D] >>> Skip MOTION"
  echo
fi

# -------- 3) event --------
if [[ "${SKIP_EVT}" -eq 0 ]]; then
  echo "[4D] >>> Running EVENT ..."
  # event 脚本如果内部自己 conda run，这里只传环境变量
  INCLUDE_MODELS="${INCLUDE_MODELS}" EXCLUDE_MODELS="${EXCLUDE_MODELS}" \
  EVENT_ENV="${EVENT_ENV}" GEN_ROOT="${GEN_ROOT}" BASE="${BASE}" bash "${EVT_SCRIPT}"
  echo
else
  echo "[4D] >>> Skip EVENT"
  echo
fi

# -------- 4) world knowledge --------
if [[ "${SKIP_WLD}" -eq 0 ]]; then
  echo "[4D] >>> Running WORLD KNOWLEDGE ..."

  W_ARGS=(
    --base "${BASE}"
    --gen-root "${GEN_ROOT}"
    --out-root "${BASE}/outputs/world_knowledge"
    --bank-dir "${BASE}/data/metadata/world_QA"
    --device cuda
    --dtype bf16
  )
  if [[ -n "${WORLD_MODEL_LOCAL}" ]]; then
    W_ARGS+=( --model-local "${WORLD_MODEL_LOCAL}" )
  else
    W_ARGS+=( --model-id "${WORLD_MODEL_ID}" )
  fi
  [[ "${FORCE}" -eq 1 ]] && W_ARGS+=( --force )

  if [[ -n "${INCLUDE_MODELS}" ]]; then
    IFS=',' read -r -a arr <<< "${INCLUDE_MODELS}"
    for m in "${arr[@]}"; do
      m="$(echo "${m}" | xargs)"
      [[ -z "${m}" ]] && continue
      _has_in_csv_list "${m}" "${EXCLUDE_MODELS}" && { echo "[World] skip excluded: ${m}"; continue; }
      conda run -n "${SEMANTIC_ENV}" bash "${WLD_SCRIPT}" "${W_ARGS[@]}" --model "${m}"
    done
  else
    # 如果设置了 exclude，则不能用 --all（world 脚本没有 exclude 参数）
    if [[ -n "${EXCLUDE_MODELS}" ]]; then
      shopt -s nullglob
      for d in "${GEN_ROOT}"/*; do
        [[ -d "${d}" ]] || continue
        m="$(basename "${d}")"
        _has_in_csv_list "${m}" "${EXCLUDE_MODELS}" && { echo "[World] skip excluded: ${m}"; continue; }
        conda run -n "${SEMANTIC_ENV}" bash "${WLD_SCRIPT}" "${W_ARGS[@]}" --model "${m}"
      done
    else
      conda run -n "${SEMANTIC_ENV}" bash "${WLD_SCRIPT}" "${W_ARGS[@]}" --all
    fi
  fi
  echo
else
  echo "[4D] >>> Skip WORLD KNOWLEDGE"
  echo
fi

# -------- 5) merge --------
if [[ "${SKIP_MERGE}" -eq 0 ]]; then
  echo "[4D] >>> Merging outputs -> ${OUT_CSV}"
  conda run -n "${SEMANTIC_ENV}" python "${MERGE_PY}" \
    --base "${BASE}" \
    --out "${OUT_CSV}"
  echo
else
  echo "[4D] >>> Skip MERGE"
  echo
fi

echo "[4D] Done."
echo "[4D] Total table: ${OUT_CSV}"
