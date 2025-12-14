#!/usr/bin/env bash
set -euo pipefail

# ------------- 基本配置（用户只需改这里） -------------
# 事件评测用的 conda 环境名
EVENT_ENV="${EVENT_ENV:-env_event_vlm}"

# 允许覆盖的路径/配置（一般不用管）
PYTHON_BIN="${PYTHON_BIN:-python}"
GEN_ROOT="${GEN_ROOT:-data/genvideo}"

CFG_DEFAULT="${CFG_DEFAULT:-ref4d_eval/event/configs/default.yaml}"
CFG_VLM="${CFG_VLM:-ref4d_eval/event/configs/model_vlm.yaml}"
CFG_EMBED="${CFG_EMBED:-ref4d_eval/event/configs/model_embed.yaml}"
CFG_SHOT="${CFG_SHOT:-ref4d_eval/event/configs/model_shot.yaml}"
CFG_GEBD="${CFG_GEBD:-ref4d_eval/event/configs/model_gebd.yaml}"

# 要跑的步骤（保持和你单样本命令一致）
STEPS="${STEPS:-detect,vlm,embed,merge,match,metrics}"

# VERBOSE=1 时显示 Python 的详细日志；否则只显示进度，内部输出丢到 /dev/null
VERBOSE="${VERBOSE:-0}"

# scores 缓存与对外暴露的“干净目录”
SCORES_CACHE_ROOT="${SCORES_CACHE_ROOT:-outputs/event/cache/scores}"
SCORES_ROOT="${SCORES_ROOT:-outputs/event/scores}"

# ------------- 主逻辑（尽量简单，不依赖额外包） -------------

# 切到仓库根目录（假定脚本位于 scripts/ 下）
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p "${SCORES_CACHE_ROOT}"
mkdir -p "${SCORES_ROOT}"

# 自动发现所有模型目录：data/genvideo/<model>/
MODELS_ENV="${MODELS-}"
if [ -z "${MODELS_ENV}" ]; then
  if [ ! -d "${GEN_ROOT}" ]; then
    echo "[run_event] ERROR: GEN_ROOT not found: ${GEN_ROOT}" >&2
    exit 1
  fi
  MODELS=""
  for d in "${GEN_ROOT}"/*; do
    [ -d "${d}" ] || continue
    MODELS+="${d##*/} "
  done
else
  MODELS="${MODELS_ENV}"
fi

echo "[run_event] repo_root = ${REPO_ROOT}"
echo "[run_event] using conda env = ${EVENT_ENV}"
echo "[run_event] GEN_ROOT = ${GEN_ROOT}"
echo "[run_event] models = ${MODELS}"
echo "[run_event] steps = ${STEPS}"
echo

# 先统计总共多少个视频，用于进度条
total=0
for model in ${MODELS}; do
  MODEL_DIR="${GEN_ROOT}/${model}"
  [ -d "${MODEL_DIR}" ] || continue
  for video in "${MODEL_DIR}"/*.mp4; do
    [ -e "${video}" ] || continue
    total=$((total+1))
  done
done

if [ "${total}" -eq 0 ]; then
  echo "[run_event] ERROR: no .mp4 videos found under ${GEN_ROOT}/*/." >&2
  exit 1
fi

idx=0

for model in ${MODELS}; do
  MODEL_DIR="${GEN_ROOT}/${model}"
  if [ ! -d "${MODEL_DIR}" ]; then
    echo "[run_event] skip model='${model}', dir not found: ${MODEL_DIR}" >&2
    continue
  fi

  echo "[run_event] ===== Model: ${model} ====="

  # 遍历 data/genvideo/<model> 下所有 mp4
  for video in "${MODEL_DIR}"/*.mp4; do
    [ -e "${video}" ] || continue

    idx=$((idx+1))

    fname="$(basename "${video}")"
    sample_id="${fname%.mp4}"

    # 约定命名：<topic>_<4位id>_single / multi
    # 例如 animals_and_ecology_001_single -> topic = animals_and_ecology
    topic="${sample_id%_*_*}"

    # cache scores.json 的位置（由 pipeline 决定）
    cache_scores_dir="${SCORES_CACHE_ROOT}/${sample_id}__${model}"
    cache_scores_json="${cache_scores_dir}/scores.json"

    # 对外暴露的“干净路径”：outputs/event/scores/<model>/<sample_id>__<model>/scores.json
    final_scores_dir="${SCORES_ROOT}/${model}/${sample_id}__${model}"
    final_scores_json="${final_scores_dir}/scores.json"

    # 进度 + 输入/输出信息（只展示最终 scores 输出位置）
    echo "[run_event] [${idx}/${total}] video=${video}"
    echo "           -> scores=${final_scores_json}"

    # 运行事件评测 pipeline
    if [ "${VERBOSE}" -eq 1 ]; then
      # 详细模式：直接把内部日志打到终端
      if ! conda run -n "${EVENT_ENV}" \
        "${PYTHON_BIN}" -m ref4d_eval.event.src.cli.main run \
          --topic "${topic}" \
          --sample-id "${sample_id}" \
          --model "${model}" \
          --steps "${STEPS}" \
          --cfg-default "${CFG_DEFAULT}" \
          --cfg-vlm     "${CFG_VLM}" \
          --cfg-embed   "${CFG_EMBED}" \
          --cfg-shot    "${CFG_SHOT}" \
          --cfg-gebd    "${CFG_GEBD}"; then
        echo "[run_event]   !! ERROR: failed for sample_id=${sample_id}, model=${model}" >&2
        continue
      fi
    else
      # 安静模式：内部输出全部丢弃，只保留我们自己的进度信息
      if ! conda run -n "${EVENT_ENV}" \
        "${PYTHON_BIN}" -m ref4d_eval.event.src.cli.main run \
          --topic "${topic}" \
          --sample-id "${sample_id}" \
          --model "${model}" \
          --steps "${STEPS}" \
          --cfg-default "${CFG_DEFAULT}" \
          --cfg-vlm     "${CFG_VLM}" \
          --cfg-embed   "${CFG_EMBED}" \
          --cfg-shot    "${CFG_SHOT}" \
          --cfg-gebd    "${CFG_GEBD}" \
        >/dev/null 2>&1; then
        echo "[run_event]   !! ERROR: failed for sample_id=${sample_id}, model=${model}" >&2
        continue
      fi
    fi

    # ---- 将 cache 中的 scores.json 同步到 outputs/event/scores/<model>/... ----
    if [ -f "${cache_scores_json}" ]; then
      mkdir -p "${final_scores_dir}"
      cp -f "${cache_scores_json}" "${final_scores_json}"
    else
      echo "[run_event]   !! WARNING: cache scores not found for sample_id=${sample_id}, model=${model}" >&2
    fi

  done
done

# ---- 同步 summary.csv 到 outputs/event/scores/summary.csv ----
if [ -f "${SCORES_CACHE_ROOT}/summary.csv" ]; then
  cp -f "${SCORES_CACHE_ROOT}/summary.csv" "${SCORES_ROOT}/summary.csv"
fi

echo
echo "[run_event] Done."
echo "[run_event] Per-sample scores are under:"
echo "  ${SCORES_ROOT}/<model>/<sample_id>__<model>/scores.json"
echo "[run_event] Global summary (if generated) at:"
echo "  ${SCORES_ROOT}/summary.csv"
