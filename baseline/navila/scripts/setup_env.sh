#!/usr/bin/env bash
# Setup navila-baseline conda environment for NaVILA training
# System: CUDA 12.1 (/usr/local/cuda-12.1), driver supports CUDA 13.0
# Skips conda cuda-toolkit install (system CUDA already available)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(dirname "${SCRIPT_DIR}")"
SWIFTVLN_ROOT="$(cd "${BASELINE_DIR}/../.." && pwd)"
# shellcheck source=../../../src/swiftvln/scripts/lib/local_env.sh
source "${SWIFTVLN_ROOT}/src/swiftvln/scripts/lib/local_env.sh"
swiftvln_load_local_env "${BASELINE_DIR}"

CONDA_ENV="${NAVILA_CONDA_ENV:-navila-baseline}"
NAVILA_REPO="${NAVILA_REPO:-../NaVILA}"
FLASH_ATTN_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"

echo "=========================================="
echo " NaVILA Baseline Env Setup"
echo " Target env: $CONDA_ENV"
echo "=========================================="

# Init conda
swiftvln_init_conda

# Step 1: Create conda env
echo "[1/6] Creating conda env: $CONDA_ENV (Python 3.10)"
conda create -n "$CONDA_ENV" python=3.10 -y
conda activate "$CONDA_ENV"

# Step 2: Upgrade pip
echo "[2/6] Upgrading pip"
pip install --upgrade pip

# Step 3: Install PyTorch 2.3.0 with CUDA 12.1
# Note: using cu121 wheels since system CUDA toolkit is 12.1
echo "[3/6] Installing PyTorch 2.3.0 (cu121)"
pip install torch==2.3.0 torchvision==0.18.0 \
    --index-url "$TORCH_INDEX"

# Step 4: Install FlashAttention2 from prebuilt whl
# cu122 whl works at runtime because flash-attn uses PyTorch's bundled CUDA runtime
# and the NVIDIA driver supports CUDA 13.0
echo "[4/6] Installing FlashAttention 2.5.8 (prebuilt whl)"
pip install "$FLASH_ATTN_WHL"

# Step 5: Install VILA (NaVILA base framework) editable
# extra-index-url ensures pip resolves torch from CUDA-aware index (prevents downgrade)
echo "[5/6] Installing VILA (editable + train extras)"
cd "$NAVILA_REPO"
pip install -e . --extra-index-url "$TORCH_INDEX"
pip install -e ".[train]" --extra-index-url "$TORCH_INDEX"
pip install -e ".[eval]" --extra-index-url "$TORCH_INDEX"

# Step 6: Install HuggingFace Transformers v4.37.2 + apply patches
echo "[6/6] Installing Transformers v4.37.2 + patches"
pip install git+https://github.com/huggingface/transformers@v4.37.2

site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
echo "    Site packages: $site_pkg_path"

if [ -d "$NAVILA_REPO/llava/train/transformers_replace" ]; then
    cp -rv "$NAVILA_REPO/llava/train/transformers_replace/"* "$site_pkg_path/transformers/"
    echo "    [OK] transformers patches applied"
else
    echo "    [WARN] transformers_replace dir not found, skipping"
fi

if [ -d "$NAVILA_REPO/llava/train/deepspeed_replace" ]; then
    cp -rv "$NAVILA_REPO/llava/train/deepspeed_replace/"* "$site_pkg_path/deepspeed/"
    echo "    [OK] deepspeed patches applied"
else
    echo "    [WARN] deepspeed_replace dir not found, skipping"
fi

echo ""
echo "=========================================="
echo " Setup complete!"
echo " Env: $CONDA_ENV"
echo " torch: $(python -c 'import torch; print(torch.__version__)')"
echo " CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "=========================================="
