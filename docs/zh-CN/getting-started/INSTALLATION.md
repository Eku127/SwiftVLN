# 安装

SwiftVLN 从源码运行。训练与在线评测使用两套独立环境：训练环境面向 Qwen-VL
监督微调与 S2R Stage-A 训练，评测环境用于 SatNav 或 Habitat 在线评测。

| 用途 | Conda 环境 | Python | 主要入口 |
| --- | --- | --- | --- |
| SwiftVLN / S2R 训练 | `swiftvln-train` | 3.10 | `scripts/train/` |
| SatNav / Habitat 评测 | `swiftvln-eval` | 3.9 | `scripts/eval/` |

## 1. 系统要求

当前仓库验证的软件栈如下。完整 Python 依赖分别记录在
[`environments/train/requirements.txt`](../../../environments/train/requirements.txt)
和
[`environments/eval/requirements.txt`](../../../environments/eval/requirements.txt)。

目前已经验证的安装环境如下。

| 组件 | 训练环境 | 评测环境 |
| --- | --- | --- |
| Python | 3.10 | 3.9 |
| CUDA Toolkit | 12.8 | 12.8 |
| PyTorch | 2.8.0+cu128 | 2.8.0+cu128 |
| ms-swift | 4.2.0.dev0 | 4.2.0.dev0 |
| Transformers | 4.57.3 | 4.57.3 |
| qwen-vl-utils | 0.0.14 | 0.0.14 |
| FlashAttention | 2.8.3 | 2.8.3 |
| DeepSpeed | 0.17.6 | 0.17.6 |
| NumPy | 2.2.6 | 1.26.1 |
| Habitat-Lab / Habitat-Sim | — | 0.2.4 |

Habitat 室内导航评测使用 Python 3.9 和 NumPy 1.26.1，与 Habitat 0.2.4
运行栈保持一致。

## 2. 获取源码

SwiftVLN 使用 Git Submodule 管理 `third_party/` 中的外部源码。先克隆主仓库：

```bash
git clone https://github.com/Eku127/SwiftVLN.git
cd SwiftVLN
export SWIFTVLN_ROOT="${PWD}"
```

训练只需要初始化官方 ms-swift：

```bash
git submodule update --init third_party/ms-swift
```

仅运行 SatNav 评测时，初始化 ms-swift 和 SatNav：

```bash
git submodule update --init \
  third_party/ms-swift \
  third_party/SatNav
```

运行 Habitat 评测时，初始化 ms-swift 和 Habitat-Lab：

```bash
git submodule update --init \
  third_party/ms-swift \
  third_party/habitat-lab-0.2.4
```

同时运行 SatNav 和 Habitat 评测时，初始化全部子模块：

```bash
git submodule update --init --recursive
```

Submodule 会检出当前 SwiftVLN 版本记录的 commit：

| 仓库 | 来源 | Commit |
| --- | --- | --- |
| ms-swift | `modelscope/ms-swift` 官方仓库 | `ad7d5c5157b59afa1faceb04266274392f145745` |
| SatNav | `Eku127/SatNav` 的 `master` | `4bd6652c875af00236a09730d76c4b391046e6a4` |
| Habitat-Lab | `facebookresearch/habitat-lab` 的 `v0.2.4` | `1639e1ae732ba1e84199a1a04b79c7243c3f8586` |

## 3. 安装训练环境

进入 SwiftVLN 仓库并创建训练环境：

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

安装固定版本的 ms-swift 和训练依赖：

```bash
python -m pip install -e "${SWIFTVLN_ROOT}/third_party/ms-swift"

cd "${SWIFTVLN_ROOT}"
python -m pip install -r environments/train/requirements.txt
```

FlashAttention 在 PyTorch 和 CUDA Toolkit 安装完成后编译：

```bash
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
MAX_JOBS=8 python -m pip install flash-attn==2.8.3 --no-build-isolation
```

最后以 editable 模式安装 SwiftVLN：

```bash
cd "${SWIFTVLN_ROOT}"
python -m pip install -e .
```

## 4. 安装评测环境

评测环境包含 Habitat-Sim、Habitat-Lab、SatNav、ms-swift 和 SwiftVLN。

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
```

编译 FlashAttention：

```bash
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
MAX_JOBS=8 python -m pip install flash-attn==2.8.3 --no-build-isolation
```

安装 ms-swift 和 SwiftVLN：

```bash
python -m pip install -e "${SWIFTVLN_ROOT}/third_party/ms-swift"
python -m pip install -e "${SWIFTVLN_ROOT}"
```

SatNav 评测安装 SatNav：

```bash
python -m pip install -e "${SWIFTVLN_ROOT}/third_party/SatNav"
```

Habitat 评测安装 Habitat-Lab 和 Habitat-Baselines：

```bash
python -m pip install -e \
  "${SWIFTVLN_ROOT}/third_party/habitat-lab-0.2.4/habitat-lab"

python -m pip install -e \
  "${SWIFTVLN_ROOT}/third_party/habitat-lab-0.2.4/habitat-baselines"
```

## 5. 配置本机路径

SwiftVLN 使用仓库根目录下的 `.local/env.sh` 保存模型、数据、Conda 和缓存路径。
该文件不进入 Git。

```bash
cd "${SWIFTVLN_ROOT}"
mkdir -p .local
cp local.env.example .local/env.sh
${EDITOR:-vi} .local/env.sh
```

根据研究任务填写下列变量：

| 变量 | 用途 |
| --- | --- |
| `SWIFTVLN_CONDA_SH` | Conda 的 `etc/profile.d/conda.sh` |
| `SWIFTVLN_TRAIN_CONDA_ENV` | 训练环境名，默认 `swiftvln-train` |
| `SWIFTVLN_EVAL_CONDA_ENV` | 评测环境名，默认 `swiftvln-eval` |
| `SWIFTVLN_QWEN25_MODEL_PATH` | Qwen2.5-VL 模型 ID 或本地目录 |
| `SWIFTVLN_QWEN3_MODEL_PATH` | Qwen3-VL 模型 ID 或本地目录 |
| `SWIFTVLN_MODELSCOPE_CACHE` | ModelScope 模型缓存目录 |

SatNav 训练与评测使用：

| 变量 | 目录或文件 |
| --- | --- |
| `SWIFTVLN_SATNAV_TRAIN_DATA_PATH` | 包含 `annotations.json` 与 `images/` 的离线轨迹目录 |
| `SWIFTVLN_SATNAV_EVAL_DATA_PATH` | Episode 路径模板，例如 `.../episodes/eval/{split}/all_episodes.json` |
| `SWIFTVLN_SATNAV_SCENES_DIR` | SatSim 使用的 GeoTIFF 场景目录 |

Habitat 训练与评测使用：

| 变量 | 目录或文件 |
| --- | --- |
| `SWIFTVLN_HABITAT_R2R_TRAIN_PATH` | R2R 离线轨迹目录 |
| `SWIFTVLN_HABITAT_RXR_TRAIN_PATH` | RxR 离线轨迹目录 |
| `SWIFTVLN_HABITAT_R2R_EVAL_DATA_PATH` | R2R-VLNCE Episode 路径模板 |
| `SWIFTVLN_HABITAT_SCENES_DIR` | Matterport3D 场景目录 |

Qwen 模型变量可以填写公开模型 ID：

```bash
export SWIFTVLN_QWEN25_MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
export SWIFTVLN_QWEN3_MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"
```

离线运行时，将其改为已经下载完成的模型目录。

## 6. 验证训练环境

```bash
conda activate swiftvln-train

python - <<'PY'
import importlib.metadata as metadata
import sys

import flash_attn
import swift
import swiftvln
import torch

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("cuda_runtime", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
for package in (
    "ms-swift",
    "transformers",
    "qwen-vl-utils",
    "flash-attn",
    "deepspeed",
):
    print(package, metadata.version(package))
print("swiftvln", swiftvln.__file__)
PY

swiftvln --help
python -m swiftvln.experiment --help
```

关键输出应包含：

```text
python 3.10.x
torch 2.8.0+cu128
cuda_runtime 12.8
cuda_available True
ms-swift 4.2.0.dev0
transformers 4.57.3
qwen-vl-utils 0.0.14
flash-attn 2.8.3
deepspeed 0.17.6
```

## 7. 验证评测环境

```bash
conda activate swiftvln-eval

python - <<'PY'
import importlib.metadata as metadata
import sys

import flash_attn
import swiftvln
import torch

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("cuda_runtime", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
for package in (
    "ms-swift",
    "transformers",
    "flash-attn",
    "numpy",
    "protobuf",
):
    print(package, metadata.version(package))
print("swiftvln", swiftvln.__file__)
PY

python -m swiftvln.evaluation --help
```

SatNav 评测环境继续检查：

```bash
python - <<'PY'
from satnav.core.env import Env

print("satnav", Env.__module__)
PY
```

Habitat 评测环境继续检查：

```bash
python - <<'PY'
import importlib.metadata as metadata

import habitat
import habitat_sim

print("habitat", metadata.version("habitat-lab"), habitat.__file__)
print("habitat_sim", metadata.version("habitat-sim"), habitat_sim.__file__)
PY
```

关键输出应包含：

```text
python 3.9.x
torch 2.8.0+cu128
cuda_runtime 12.8
cuda_available True
ms-swift 4.2.0.dev0
transformers 4.57.3
flash-attn 2.8.3
numpy 1.26.1
protobuf 3.20.1
```

SatNav 检查输出包含 `satnav satnav.core.env`；Habitat 检查输出中的
`habitat-lab` 与 `habitat-sim` 版本均为 `0.2.4`。

## 8. 安装问题定位

### 当前终端没有使用目标环境

```bash
which python
python -m pip show swiftvln ms-swift torch
```

`swiftvln` 的 editable 路径应指向当前 SwiftVLN checkout，训练和评测时的 Python
分别来自对应 Conda 环境。

### FlashAttention 编译时找不到 CUDA

```bash
conda activate swiftvln-train  # 或 swiftvln-eval
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

确认输出为 CUDA 12.8 后重新执行 FlashAttention 安装命令。

### Habitat 模块没有从固定 checkout 加载

```bash
conda activate swiftvln-eval
python - <<'PY'
import habitat
import habitat_sim

print(habitat.__file__)
print(habitat_sim.__file__)
PY
```

`habitat` 应来自 `habitat-lab-0.2.4/habitat-lab` 的 editable install，
`habitat_sim` 版本应为 0.2.4。

## 9. 下一步

- [快速开始](QUICKSTART.md)：检查训练配置并运行最小训练与评测；
- [训练数据](../data/TRAINING_DATA.md)：准备 SatNav 或 Habitat 离线轨迹；
- [SwiftVLN 训练](../training/README.md)：选择训练方案并启动完整训练；
- [SwiftVLN 评测](../evaluation/README.md)：运行 SatNav 或 Habitat 在线评测。
