#!/usr/bin/env bash
set -e

# 获取仓库根目录
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# -----------------------------
# [A] 强制把 conda/pip 的写入都放到 autodl-tmp，避免系统盘爆
# -----------------------------
export CONDA_PKGS_DIRS="/root/autodl-tmp/conda_pkgs"
export CONDA_ENVS_PATH="/root/autodl-tmp/conda_envs"
mkdir -p "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH"

# pip 缓存/临时目录也挪走（sam2/torch 大包会写很多临时文件）
export PIP_CACHE_DIR="/root/autodl-tmp/pip_cache"
export TMPDIR="/root/autodl-tmp/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

# （可选但推荐）HF/transformers 缓存也挪走，后面下 bert 会用到
export HF_HOME="/root/autodl-tmp/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/root/autodl-tmp/.cache/huggingface"
export TRANSFORMERS_CACHE="/root/autodl-tmp/.cache/huggingface"
mkdir -p "$HF_HOME"

echo "[0/3] Repo root = $BASE_DIR"
echo "[0/3] CONDA_PKGS_DIRS = $CONDA_PKGS_DIRS"
echo "[0/3] CONDA_ENVS_PATH = $CONDA_ENVS_PATH"
echo "[0/3] PIP_CACHE_DIR   = $PIP_CACHE_DIR"
echo "[0/3] TMPDIR          = $TMPDIR"

echo "[1/3] Create conda env: ref4d_motion"
# libmamba solver 更省内存（如果你 conda 版本不支持 --solver，就删掉这一项）
conda env create --solver libmamba -f "$BASE_DIR/envs/motion_eval.yml" || {
  echo "   Env may already exist, trying: conda env update..."
  conda env update --solver libmamba -f "$BASE_DIR/envs/motion_eval.yml"
}

echo "[2/3] Please activate env manually:"
echo "   conda activate ref4d_motion"
echo "然后再运行本脚本的剩余部分："
read -p "如果已经在 ref4d_motion 里了，按回车继续；否则 Ctrl+C 退出。" _


# -----------------------------
if [[ "${CONDA_DEFAULT_ENV:-}" != "/root/autodl-tmp/conda_envs/ref4d_motion" ]]; then
  echo "[ERR] Current env = ${CONDA_DEFAULT_ENV:-<empty>}"
  echo "      You are NOT in ref4d_motion. Please run:"
  echo "      conda activate ref4d_motion"
  exit 1
fi

echo "[3/3] Install third_party repos in editable mode (RRM-only)..."

# GroundingDINO（subject mask 用到）
if [ -d "$BASE_DIR/third_party/grounding_dino" ]; then
  echo " - installing GroundingDINO (editable) from third_party/grounding_dino"
  pip install -e "$BASE_DIR/third_party/grounding_dino" --no-build-isolation --no-deps
elif [ -d "$BASE_DIR/third_party/GroundingDINO" ]; then
  echo " - installing GroundingDINO (editable) from third_party/GroundingDINO"
  pip install -e "$BASE_DIR/third_party/GroundingDINO" --no-build-isolation --no-deps
else
  echo " - GroundingDINO repo not found, skip."
fi

# SAM2（强烈建议这样装：禁用 build isolation + 禁用 deps，避免 pip 重新下 torch）
if [ -d "$BASE_DIR/third_party/sam2" ]; then
  echo " - installing SAM2 (editable, no-build-isolation, no-deps)"
  pip install -e "$BASE_DIR/third_party/sam2" --no-build-isolation --no-deps
fi

# TAPIR：按你们 repo 的方式处理（如有 setup.py/pyproject 可改成 pip -e）
if [ -d "$BASE_DIR/third_party/tapir" ]; then
  echo " - TAPIR found at third_party/tapir (follow its README for deps/weights if needed)"
fi

echo " - installing Ref4D-VideoBench (motion module) as editable"
pip install -e "$BASE_DIR"

echo "[done] RRM 基础环境安装完成。"
echo "下一步：bash scripts/download_motion_models.sh"
echo "自检： python -m ref4d_eval.motion.tests.test_rrm_cache_equiv"
