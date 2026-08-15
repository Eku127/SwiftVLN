# 安装

SwiftVLN 使用两套 Conda 环境：

| 用途 | Conda 环境 | Python |
| --- | --- | --- |
| 监督微调与 S2R Stage-A 训练 | `swiftvln-train` | 3.10 |
| SatNav / Habitat 在线评测 | `swiftvln-eval` | 3.9 |

> SatNav 训练可以使用 Python 3.10；Habitat 在线评测需要使用 Python 3.9。

## 1. 获取源码

克隆 SwiftVLN：

```bash
git clone https://github.com/Eku127/SwiftVLN.git
cd SwiftVLN
export SWIFTVLN_ROOT="${PWD}"
```

根据任务拉取所需子模块：

```bash
# 训练
git submodule update --init third_party/ms-swift

# SatNav 评测
git submodule update --init third_party/ms-swift third_party/SatNav

# Habitat 评测
git submodule update --init \
  third_party/ms-swift \
  third_party/habitat-lab-0.2.4
```

同时需要使用 SatNav 和 Habitat 时，可以拉取全部子模块：

```bash
git submodule update --init --recursive
```

## 2. 安装训练环境

创建并激活训练环境：

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
cd "${SWIFTVLN_ROOT}"

conda env create -f environments/train/conda.yml
conda activate swiftvln-train
python -m pip install --upgrade pip setuptools wheel
```

安装 PyTorch CUDA 12.8：

```bash
python -m pip install \
  torch==2.8.0+cu128 \
  torchvision==0.23.0+cu128 \
  torchaudio==2.8.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
```

安装 ms-swift、训练依赖和 SwiftVLN：

```bash
python -m pip install -e "${SWIFTVLN_ROOT}/third_party/ms-swift"
python -m pip install -r environments/train/requirements.txt

export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
MAX_JOBS=8 python -m pip install flash-attn==2.8.3 --no-build-isolation

python -m pip install -e "${SWIFTVLN_ROOT}"
```

## 3. 安装评测环境

创建并激活评测环境：

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
cd "${SWIFTVLN_ROOT}"

conda env create -f environments/eval/conda.yml
conda activate swiftvln-eval
python -m pip install --upgrade pip setuptools wheel
```

安装 PyTorch CUDA 12.8 和评测依赖：

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

安装 ms-swift 和 SwiftVLN：

```bash
python -m pip install -e "${SWIFTVLN_ROOT}/third_party/ms-swift"
python -m pip install -e "${SWIFTVLN_ROOT}"
```

SatNav 评测还需安装 SatNav：

```bash
python -m pip install -e "${SWIFTVLN_ROOT}/third_party/SatNav"
```

Habitat 评测还需安装 Habitat-Lab 和 Habitat-Baselines：

```bash
python -m pip install -e \
  "${SWIFTVLN_ROOT}/third_party/habitat-lab-0.2.4/habitat-lab"
python -m pip install -e \
  "${SWIFTVLN_ROOT}/third_party/habitat-lab-0.2.4/habitat-baselines"
```

## 4. 配置本机路径

复制本机配置模板：

```bash
cd "${SWIFTVLN_ROOT}"
mkdir -p .local
cp local.env.example .local/env.sh
${EDITOR:-vi} .local/env.sh
```

在 `.local/env.sh` 中填写 Conda、模型、数据集和场景路径。训练与评测脚本会自动
加载该文件。

模型路径可以使用 Hugging Face 模型 ID，也可以指向本地目录：

```bash
export SWIFTVLN_QWEN25_MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
export SWIFTVLN_QWEN3_MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"
```

## 5. 验证安装

验证训练环境：

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

验证评测环境：

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

根据评测平台检查 SatNav 或 Habitat：

```bash
# SatNav
python -c "from satnav.core.env import Env; print(Env.__module__)"

# Habitat
python -c "import habitat, habitat_sim; print(habitat.__file__, habitat_sim.__file__)"
```

## 6. 下一步

- [快速开始](QUICKSTART.md)
- [SatNav 训练数据](../data/TRAINING_DATA_SATNAV.md)
- [Habitat 训练数据](../data/TRAINING_DATA_HABITAT.md)
- [SwiftVLN 训练](../training/README.md)
- [SwiftVLN 评测](../evaluation/README.md)
