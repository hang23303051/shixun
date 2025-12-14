#!/usr/bin/env bash
set -euo pipefail

# ================= 基本路径设定 =================
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TP_DIR="${TP_DIR:-third_party}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"

mkdir -p "${TP_DIR}" "${CKPT_DIR}"

# HuggingFace CLI 命令名，可通过环境变量 HF_CMD 覆盖
HF_CMD="${HF_CMD:-huggingface-cli}"

echo "[download_event] repo_root = ${REPO_ROOT}"
echo "[download_event] third_party dir = ${TP_DIR}"
echo "[download_event] checkpoints dir = ${CKPT_DIR}"
echo "[download_event] huggingface cli = ${HF_CMD}"
echo

# 小工具：若目录不存在则 git clone，存在则跳过
clone_if_needed () {
  local url="$1"
  local target="$2"
  if [ -d "${target}" ]; then
    echo "[download_event] git repo exists, skip: ${target}"
  else
    echo "[download_event] cloning ${url} -> ${target}"
    git clone --depth 1 "${url}" "${target}"
  fi
}

# 小工具：用 HF CLI 下载到指定目录（若目录已有内容则跳过）
hf_download_if_needed () {
  local repo_id="$1"
  local target_dir="$2"
  if [ -d "${target_dir}" ] && [ -n "$(ls -A "${target_dir}" 2>/dev/null || true)" ]; then
    echo "[download_event] HF repo already downloaded, skip: ${target_dir}"
    return 0
  fi
  mkdir -p "${target_dir}"
  echo "[download_event] downloading HF repo ${repo_id} -> ${target_dir}"
  "${HF_CMD}" download "${repo_id}" --local-dir "${target_dir}" --local-dir-use-symlinks False
}

# ================= 1) TransNetV2（shot detect） =================
echo "=== [1/4] TransNetV2 (shot detection) ==="

TP_TRANS="${TP_DIR}/transnetv2"
CK_TRANS="${CKPT_DIR}/transnetv2"
mkdir -p "${CK_TRANS}"

# 代码
clone_if_needed "https://github.com/soCzech/TransNetV2.git" "${TP_TRANS}"

# 权重：从 HF Sn4kehead/TransNetV2 下载；然后统一重命名一个 .pth
hf_download_if_needed "Sn4kehead/TransNetV2" "${CK_TRANS}"

# 尝试把目录里第一个 *.pth 统一改名为 transnetv2-pytorch-weights.pth
TRANS_PTH="$(find "${CK_TRANS}" -maxdepth 2 -type f -name '*.pth' | head -n 1 || true)"
if [ -n "${TRANS_PTH}" ]; then
  if [ "${TRANS_PTH}" != "${CK_TRANS}/transnetv2-pytorch-weights.pth" ]; then
    echo "[download_event] rename ${TRANS_PTH} -> ${CK_TRANS}/transnetv2-pytorch-weights.pth"
    mv "${TRANS_PTH}" "${CK_TRANS}/transnetv2-pytorch-weights.pth"
  else
    echo "[download_event] transnetv2 weights already named correctly."
  fi
else
  echo "[download_event] WARNING: no *.pth found in ${CK_TRANS}, please check manually."
fi
echo

# ================= 2) DDM-Net（GEBD） =================
echo "=== [2/4] DDM-Net (GEBD) ==="

TP_DDM="${TP_DIR}/ddmnet"
CK_DDM="${CKPT_DIR}/ddmnet"
mkdir -p "${CK_DDM}"

# 代码
clone_if_needed "https://github.com/MCG-NJU/DDM.git" "${TP_DDM}"

# 权重：官方提供的是 Google Drive，目前不强行脚本化，给出明确提示
if [ -f "${CK_DDM}/checkpoint.pth.tar" ]; then
  echo "[download_event] DDM-Net ckpt exists: ${CK_DDM}/checkpoint.pth.tar"
else
  echo "[download_event] TODO: 请按 DDM 官方仓库 README 下载 GEBD checkpoint："
  echo "  repo: https://github.com/MCG-NJU/DDM"
  echo "  然后将对应的 GEBD 模型权重重命名为：checkpoint.pth.tar"
  echo "  并放置到：${CK_DDM}/checkpoint.pth.tar"
fi
echo

# ================= 3) VideoLLaMA3-7B（段级语义描述 VLM） =================
echo "=== [3/4] VideoLLaMA3-7B (segment-level VLM) ==="

TP_VLLM3="${TP_DIR}/videollama3"
CK_VLLM3="${CKPT_DIR}/videollama3-7b"
mkdir -p "${CK_VLLM3}"

# 代码仓库
clone_if_needed "https://github.com/DAMO-NLP-SG/VideoLLaMA3.git" "${TP_VLLM3}"

# 权重：从 HF DAMO-NLP-SG/VideoLLaMA3-7B 下载
hf_download_if_needed "DAMO-NLP-SG/VideoLLaMA3-7B" "${CK_VLLM3}"
echo

# ================= 4) E5-large-v2（事件文本嵌入） =================
echo "=== [4/4] E5-large-v2 (text embedding) ==="

CK_E5="${CKPT_DIR}/e5-large-v2"
mkdir -p "${CK_E5}"

hf_download_if_needed "intfloat/e5-large-v2" "${CK_E5}"
echo

echo "=============================================="
echo "[download_event] All steps finished."
echo
echo "路径约定："
echo "  - TransNetV2 代码:       ${TP_TRANS}"
echo "  - TransNetV2 权重:       ${CK_TRANS}/transnetv2-pytorch-weights.pth"
echo "  - DDM-Net 代码:          ${TP_DDM}"
echo "  - DDM-Net 权重:          ${CK_DDM}/checkpoint.pth.tar   (需按提示手动下载)"
echo "  - VideoLLaMA3 代码:      ${TP_VLLM3}"
echo "  - VideoLLaMA3-7B 权重:   ${CK_VLLM3}/"
echo "  - E5-large-v2 权重:      ${CK_E5}/"
echo
echo "请在 ref4d_eval/event/configs/ 中确认："
echo "  - model_shot.yaml   中的 TransNetV2 权重路径指向 transnetv2-pytorch-weights.pth"
echo "  - model_gebd.yaml   中的 DDM-Net 权重路径指向 checkpoints/ddmnet/checkpoint.pth.tar"
echo "  - model_vlm.yaml    中 VideoLLaMA3-7B 和 E5 的 local_path 与以上目录一致。"
