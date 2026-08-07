#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(dirname "${SCRIPT_DIR}")"
SWIFTVLN_ROOT="$(cd "${BASELINE_DIR}/../.." && pwd)"
# shellcheck source=../../../src/swiftvln/scripts/lib/local_env.sh
source "${SWIFTVLN_ROOT}/src/swiftvln/scripts/lib/local_env.sh"
swiftvln_load_local_env "${BASELINE_DIR}"

CONDA_ENV="${OPENFLY_CONDA_ENV:-openfly-baseline}"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
FLASH_ATTN_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

swiftvln_init_conda

conda create -n "${CONDA_ENV}" python=3.10 -y
conda activate "${CONDA_ENV}"

# Direct-download large wheels instead of inheriting local loopback proxies.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

pip install --upgrade pip
pip install torch==2.3.0 torchvision==0.18.0 --index-url "${TORCH_INDEX}"
pip install "${FLASH_ATTN_WHL}"
pip install \
    transformers==4.48.1 \
    accelerate==0.33.0 \
    deepspeed==0.14.4 \
    timm==0.9.16 \
    tokenizers==0.21.1 \
    sentencepiece \
    safetensors \
    huggingface_hub \
    numpy \
    scipy \
    pandas \
    pillow \
    opencv-python-headless \
    pyyaml \
    tqdm \
    omegaconf \
    packaging \
    ninja

SATNAV_REPO="${SWIFTVLN_SATNAV_REPO:-../SatNav}"
pip install -e "${SATNAV_REPO}"

echo "=========================================="
echo "Env setup complete: ${CONDA_ENV}"
echo "torch=$(python -c 'import torch; print(torch.__version__)')"
echo "cuda_available=$(python -c 'import torch; print(torch.cuda.is_available())')"
echo "=========================================="
