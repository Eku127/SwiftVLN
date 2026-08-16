# Installation

[简体中文](../../zh-CN/getting-started/INSTALLATION.md) | English

SwiftVLN uses two Conda environments:

| Purpose | Conda Environment | Python |
| --- | --- | --- |
| Supervised fine-tuning and Satellite-to-UAV Stage-A training |`swiftvln-train`| 3.10 |
| SatNav / Habitat Online Evaluation |`swiftvln-eval`| 3.9 |

> SatNav training can use Python 3.10; Habitat online evaluation requires Python 3.9.

## 1. Get the source code

Clone SwiftVLN:

```bash
git clone https://github.com/Eku127/SwiftVLN.git
cd SwiftVLN
export SWIFTVLN_ROOT="${PWD}"
```

Pull the required submodules according to the task:

```bash
# Training
git submodule update --init third_party/ms-swift

# SatNav Evaluation
git submodule update --init third_party/ms-swift third_party/SatNav

# Habitat Evaluation
git submodule update --init \
  third_party/ms-swift \
  third_party/habitat-lab-0.2.4
```

When you need to use SatNav and Habitat at the same time, you can pull all submodules:

```bash
git submodule update --init --recursive
```

## 2. Install the training environment

Create and activate the training environment:

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
cd "${SWIFTVLN_ROOT}"

conda env create -f environments/train/conda.yml
conda activate swiftvln-train
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch CUDA 12.8:

```bash
python -m pip install \
  torch==2.8.0+cu128 \
  torchvision==0.23.0+cu128 \
  torchaudio==2.8.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
```

Install ms-swift, training dependencies, and SwiftVLN:

```bash
python -m pip install -e "${SWIFTVLN_ROOT}/third_party/ms-swift"
python -m pip install -r environments/train/requirements.txt

export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
MAX_JOBS=8 python -m pip install flash-attn==2.8.3 --no-build-isolation

python -m pip install -e "${SWIFTVLN_ROOT}"
```

## 3. Install the evaluation environment

Create and activate the evaluation environment:

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
cd "${SWIFTVLN_ROOT}"

conda env create -f environments/eval/conda.yml
conda activate swiftvln-eval
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch CUDA 12.8 and evaluation dependencies:

```bash
python -m pip install \
  torch==2.8.0+cu128 \
  torchvision==0.23.0+cu128 \
  torchaudio==2.8.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install -r environments/eval/requirements.txt

export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
MAX_JOBS=8 python -m pip install flash-attn==2.8.3 --no-build-isolation
```

Install ms-swift and SwiftVLN:

```bash
python -m pip install -e "${SWIFTVLN_ROOT}/third_party/ms-swift"
python -m pip install -e "${SWIFTVLN_ROOT}"
```

SatNav evaluation also requires SatNav to be installed:

```bash
python -m pip install -e "${SWIFTVLN_ROOT}/third_party/SatNav"
```

Habitat evaluation also requires Habitat-Lab and Habitat-Baselines to be installed:

```bash
python -m pip install -e \
  "${SWIFTVLN_ROOT}/third_party/habitat-lab-0.2.4/habitat-lab"
python -m pip install -e \
  "${SWIFTVLN_ROOT}/third_party/habitat-lab-0.2.4/habitat-baselines"
```

## 4. Configure the local path

Copy the local configuration template:

```bash
cd "${SWIFTVLN_ROOT}"
mkdir -p .local
cp local.env.example .local/env.sh
${EDITOR:-vi} .local/env.sh
```

Fill in the Conda, model, dataset, and scene paths in `.local/env.sh`. The training and evaluation scripts load this file automatically.

The model path can use the Hugging Face model ID or point to a local directory:

```bash
export SWIFTVLN_QWEN25_MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
export SWIFTVLN_QWEN3_MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"
```

## 5. Verify installation

Verify training environment:

```bash
conda activate swiftvln-train

python - <<'PY'
import flash_attn
import swift
import swiftvln
import torch

print("torch", torch.__version__)
print("cuda", torch.version.cuda, torch.cuda.is_available())
print("swiftvln", swiftvln.__file__)
PY

swiftvln --help
python -m swiftvln.experiment --help
```

Verify evaluation environment:

```bash
conda activate swiftvln-eval

python - <<'PY'
import flash_attn
import swiftvln
import torch

print("torch", torch.__version__)
print("cuda", torch.version.cuda, torch.cuda.is_available())
print("swiftvln", swiftvln.__file__)
PY

python -m swiftvln.evaluation --help
```

Verify the platform you plan to evaluate:

```bash
# SatNav
python -c "from satnav.core.env import Env; print(Env.__module__)"

# Habitat
python -c "import habitat, habitat_sim; print(habitat.__file__, habitat_sim.__file__)"
```
