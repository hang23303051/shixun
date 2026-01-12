#!/usr/bin/env bash
set -e

# ========================================================================
# Ref4D-VideoBench / Motion Dimension
# 一键下载运动维度相关权重（RRM-only + BERT）
#
# 默认下载：
#   1) TAPIR / BootStabPIR:
#        checkpoints/tapnet_checkpoints/bootstapir_checkpoint_v2.pt
#   2) GroundingDINO (SWIN-T OGC):
#        checkpoints/groundingdino/groundingdino_swint_ogc.pth
#   3) SAM2 (sam2.1_hiera_large):
#        checkpoints/sam2/sam2.1_hiera_large.pt
#   4) BERT-base-uncased (HF Transformers):
#        checkpoints/bert-base-uncased/{config.json, pytorch_model.bin, ...}
#
# 可选：
#   - SAM v1 ViT-H: checkpoints/sam/sam_vit_h_4b8939.pth
#       通过环境变量控制：DOWNLOAD_SAM_V1=1 ./scripts/download_motion_models.sh
#
# 说明：
#   - GMFlow 权重不在此脚本中自动下载，推荐用 ptlflow 自动拉取。
# ========================================================================

# 仓库根目录（假定脚本放在 scripts/ 目录下）
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${BASE_DIR}/checkpoints"

# 可选开关：是否下载 SAM v1 ViT-H
DOWNLOAD_SAM_V1="${DOWNLOAD_SAM_V1:-0}"   # 0=不下载, 1=下载 sam_vit_h_4b8939.pth

echo "[INFO] BASE_DIR = ${BASE_DIR}"
echo "[INFO] CKPT_DIR = ${CKPT_DIR}"
mkdir -p "${CKPT_DIR}"

# ------------------------------------------------------------------------
# 小工具：带断点续传的下载函数（优先 wget，其次 curl）
# ------------------------------------------------------------------------
download_file() {
  local url="$1"
  local out_path="$2"

  if [ -f "${out_path}" ]; then
    echo "  [SKIP] ${out_path} already exists."
    return 0
  fi

  mkdir -p "$(dirname "${out_path}")"
  echo "  [DL] ${url}"
  echo "      -> ${out_path}"

  # 1) curl（更稳，支持 TLS 参数与重试）
  if command -v curl >/dev/null 2>&1; then
    if curl -L --retry 8 --retry-all-errors --connect-timeout 20 \
         --tlsv1.2 --http1.1 \
         -o "${out_path}.part" "${url}"; then
      mv "${out_path}.part" "${out_path}"
      return 0
    fi
  fi

  # 2) wget（备用）
  if command -v wget >/dev/null 2>&1; then
    if wget --tries=8 --timeout=20 --waitretry=2 \
         --secure-protocol=TLSv1_2 \
         -O "${out_path}.part" "${url}"; then
      mv "${out_path}.part" "${out_path}"
      return 0
    fi
  fi

  # 3) python requests（很多情况下比 wget 更容易过 SSL）
  python - <<PY
import os, sys
import requests
url="${url}"
out="${out_path}.part"
os.makedirs(os.path.dirname(out), exist_ok=True)
with requests.get(url, stream=True, timeout=60) as r:
    r.raise_for_status()
    with open(out, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
print("[PY] downloaded", out)
PY
  mv "${out_path}.part" "${out_path}"
}


# ------------------------------------------------------------------------
# 1. TAPIR / BootStabPIR（Google Research / dm-tapnet 官方）
#    bootstapir_checkpoint_v2.pt
# ------------------------------------------------------------------------
echo ""
echo "[1/5] Download TAPIR / BootStabPIR checkpoint..."

TAPIR_DIR="${CKPT_DIR}/tapnet_checkpoints"
TAPIR_FILE="${TAPIR_DIR}/bootstapir_checkpoint_v2.pt"
TAPIR_URL="https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt"

download_file "${TAPIR_URL}" "${TAPIR_FILE}"
echo "  [OK] TAPIR checkpoint ready at ${TAPIR_FILE}"

# ------------------------------------------------------------------------
# 2. GroundingDINO SWIN-T OGC
#    groundingdino_swint_ogc.pth
# ------------------------------------------------------------------------
echo ""
echo "[2/5] Download GroundingDINO (SWIN-T OGC)..."

GDINO_DIR="${CKPT_DIR}/groundingdino"
GDINO_FILE="${GDINO_DIR}/groundingdino_swint_ogc.pth"
GDINO_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

download_file "${GDINO_URL}" "${GDINO_FILE}"
echo "  [OK] GroundingDINO checkpoint ready at ${GDINO_FILE}"

# ------------------------------------------------------------------------
# 3. SAM2 (Meta) — sam2.1_hiera_large.pt
# ------------------------------------------------------------------------
echo ""
echo "[3/5] Download SAM2 (sam2.1_hiera_large)..."

SAM2_DIR="${CKPT_DIR}/sam2"
SAM2_FILE="${SAM2_DIR}/sam2.1_hiera_large.pt"
SAM2_URL="https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt"

download_file "${SAM2_URL}" "${SAM2_FILE}"
echo "  [OK] SAM2 checkpoint ready at ${SAM2_FILE}"

# ------------------------------------------------------------------------
# 4. BERT-base-uncased（Transformers，从 HF 拉取并 save_pretrained）
#    输出目录结构与本地一致：checkpoints/bert-base-uncased/*
# ------------------------------------------------------------------------
echo ""
echo "[4/5] Download BERT-base-uncased via Transformers..."

BERT_DIR="${CKPT_DIR}/bert-base-uncased"
if [ -f "${BERT_DIR}/pytorch_model.bin" ]; then
  echo "  [SKIP] BERT-base-uncased already exists at ${BERT_DIR}"
else
  mkdir -p "${BERT_DIR}"
  export BERT_DIR
  python - << 'PY'
import os
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

out = Path(os.environ["BERT_DIR"])
out.mkdir(parents=True, exist_ok=True)

print(f"[PY] Downloading bert-base-uncased to {out} ...")
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model.save_pretrained(out)
tokenizer.save_pretrained(out)

print("[PY] BERT-base-uncased saved.")
PY
fi
echo "  [OK] BERT-base-uncased ready at ${BERT_DIR}"

# ------------------------------------------------------------------------
# 5. SAM v1 ViT-H（可选）
# ------------------------------------------------------------------------
if [ "${DOWNLOAD_SAM_V1}" = "1" ]; then
  echo ""
  echo "[5/5] Download SAM v1 ViT-H (optional)..."

  SAM1_DIR="${CKPT_DIR}/sam"
  SAM1_FILE="${SAM1_DIR}/sam_vit_h_4b8939.pth"
  SAM1_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

  download_file "${SAM1_URL}" "${SAM1_FILE}"
  echo "  [OK] SAM ViT-H checkpoint ready at ${SAM1_FILE}"
else
  echo ""
  echo "[5/5] Skip SAM v1 ViT-H (set DOWNLOAD_SAM_V1=1 to enable)."
fi

# ------------------------------------------------------------------------
# GMFlow 说明（不自动下载）
# ------------------------------------------------------------------------
echo ""
echo "[NOTE] GMFlow checkpoints are NOT downloaded automatically in this script."
echo "       推荐：使用 ptlflow 在首次调用 GMFlow 时自动下载权重。"
echo "       如果需要手动放置，请参考 GMFlow 官方仓库："
echo "         https://github.com/haofeixu/gmflow"
echo "       并将 *.pth 移动到："
echo "         ${CKPT_DIR}/gmflow/"

echo ""
echo "[DONE] All required motion-dimension checkpoints are prepared."
