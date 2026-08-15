# SwiftVLN Installation

本文档用于从零创建当前主线训练环境 `swift-vln-train-update` 和评测环境
`swift-vln-eval-update`，并安装 SwiftVLN 与对应的 ms-swift 4.x 代码版本。
旧 `swift-vln`、`swift-vln-base`、`swift-vln-train`、`swift-vln-eval`
环境均已删除；当前环境必须按本文档从零创建。

## 路径配置方式

SwiftVLN 支持两种等价方式：

1. 推荐方式：复制 `local.env.example` 为 `.local/env.sh`，填写本机绝对路径；
   该文件被 Git 忽略，训练/评测脚本会自动加载。
2. 直接方式：在 shell 中导出同名环境变量，或者直接修改脚本和 YAML 中的公开
   默认值。这样不依赖 `.local`，适合希望维护自定义 fork 的用户。

下面的 `${...}` 均指 `local.env.example` 中的变量。安装前可先定义仓库位置：

```bash
export WORKSPACE="${WORKSPACE:-$HOME/workspace}"
export SWIFTVLN_ROOT="${SWIFTVLN_ROOT:-$WORKSPACE/SwiftVLN}"
export MS_SWIFT_REPO="${MS_SWIFT_REPO:-$WORKSPACE/ms-swift}"
export SWIFTVLN_SATNAV_REPO="${SWIFTVLN_SATNAV_REPO:-$WORKSPACE/SatNav}"
export HABITAT_LAB_REPO="${HABITAT_LAB_REPO:-$WORKSPACE/habitat-lab-0.2.4}"
```

仓库目录职责：

- `ms-swift`：当前 SwiftVLN 主线使用的 ms-swift 4.x，固定在下述验证 commit。
- `ms-swift/dist/legacy/` 保留冻结的 3.x wheel，仅用于必要时重建历史环境；
  当前不再保留任何 ms-swift 3.x Conda 环境。
- 旧 3.x 源码目录已经删除；当前未加后缀的 `ms-swift` 即 4.x 主线源码目录。

主线环境定义按用途成对存放，避免在仓库根目录混放 Conda 与 pip 文件：

```text
environments/
├── train/
│   ├── conda.yml
│   └── requirements.txt
└── eval/
    ├── conda.yml
    └── requirements.txt
```

`conda.yml` 只负责 Python、CUDA toolkit 和 Habitat-Sim 等 Conda 依赖；
`requirements.txt` 负责对应环境的 pip 依赖。PyTorch、FlashAttention 和 editable
上游仓库仍按下文顺序单独安装。

## Train Environment (`swift-vln-train-update`)

### Train 版本固定

当前验证版本：

- SwiftVLN repo: `https://github.com/Eku127/SwiftVLN.git`
- SwiftVLN branch: `master`
- SwiftVLN commit: 当前 `master` 分支 HEAD
- ms-swift repo: `https://github.com/modelscope/ms-swift.git`
- ms-swift commit: `ad7d5c5157b59afa1faceb04266274392f145745`
- conda env: `swift-vln-train-update`
- Python: `3.10`
- PyTorch: `2.8.0+cu128`
- CUDA toolkit: `12.8`
- ms-swift: `4.2.0.dev0`
- Transformers: `4.57.3`
- qwen-vl-utils: `0.0.14`

### Train 1. 拉取代码

建议保持两个仓库并列放在同一个 workspace 下：

```bash
export WORKSPACE="${WORKSPACE:-$HOME/workspace}"
export SWIFTVLN_ROOT="${SWIFTVLN_ROOT:-$WORKSPACE/SwiftVLN}"
export MS_SWIFT_REPO="${MS_SWIFT_REPO:-$WORKSPACE/ms-swift}"
mkdir -p "${WORKSPACE}"
cd "${WORKSPACE}"

git clone https://github.com/Eku127/SwiftVLN.git
cd SwiftVLN
git fetch origin
git checkout master

cd "${WORKSPACE}"
git clone https://github.com/modelscope/ms-swift.git ms-swift
cd ms-swift
git checkout ad7d5c5157b59afa1faceb04266274392f145745
```

克隆后可创建本地覆盖文件；后续安装命令应在已加载该文件的 shell 中执行：

```bash
cd "${SWIFTVLN_ROOT}"
mkdir -p .local
cp local.env.example .local/env.sh
${EDITOR:-vi} .local/env.sh
source .local/env.sh
```

如需完全复现某次已验证安装，请在安装前记录并固定当前 `master`
的 `git rev-parse HEAD` 输出。

### Train 2. 创建 conda 环境

```bash
source "${SWIFTVLN_CONDA_SH}"
cd "${SWIFTVLN_ROOT}"

conda env create -f environments/train/conda.yml
conda activate swift-vln-train-update

export PYTHONNOUSERSITE=1
python -m pip install -U pip setuptools wheel
```

如果本机已经存在同名环境并且你确认要重建，先执行：

```bash
conda env remove -n swift-vln-train-update
```

### Train 3. 安装 PyTorch CUDA 12.8

```bash
conda activate swift-vln-train-update
export PYTHONNOUSERSITE=1

pip install \
  torch==2.8.0+cu128 \
  torchvision==0.23.0+cu128 \
  torchaudio==2.8.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
```

### Train 4. 安装 ms-swift 与训练依赖

```bash
conda activate swift-vln-train-update
export PYTHONNOUSERSITE=1

cd "${MS_SWIFT_REPO}"
pip install -e .

cd "${SWIFTVLN_ROOT}"
pip install -r environments/train/requirements.txt
```

`flash-attn` 需要在 PyTorch 和 CUDA toolkit 已就绪后单独安装：

```bash
conda activate swift-vln-train-update
export PYTHONNOUSERSITE=1
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"

MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation
```

如果在 `/tmp` 或其他独立挂载点里安装，并遇到 `Invalid cross-device link`，
把 pip cache 和临时目录放到同一个文件系统后重试：

```bash
mkdir -p "$PWD/.tmp" "$PWD/.pip-cache"
export TMPDIR="$PWD/.tmp"
export PIP_CACHE_DIR="$PWD/.pip-cache"
MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation
```

最后安装 SwiftVLN 本仓库：

```bash
cd "${SWIFTVLN_ROOT}"
pip install -e .
```

### Train 5. 验证环境

```bash
conda activate swift-vln-train-update
export PYTHONNOUSERSITE=1

python - <<'PY'
import importlib.metadata as m
import torch

print("torch", torch.__version__)
for pkg in [
    "torchvision",
    "torchaudio",
    "ms-swift",
    "transformers",
    "qwen-vl-utils",
    "decord",
    "flash-attn",
    "deepspeed",
]:
    print(pkg, m.version(pkg))

print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
print("device_count", torch.cuda.device_count())
PY
```

期望关键输出：

```text
torch 2.8.0+cu128
torchvision 0.23.0+cu128
torchaudio 2.8.0+cu128
ms-swift 4.2.0.dev0
transformers 4.57.3
qwen-vl-utils 0.0.14
decord 0.6.0
flash-attn 2.8.3
deepspeed 0.17.6
cuda_available True
cuda_version 12.8
```

`pip check` 在当前环境可能报告 `decord 0.6.0 is not supported on this platform`；
当前验证口径以 `import decord` 成功和训练脚本 dry run 成功为准。

### Train 6. 路径约定

训练脚本从 `.local/env.sh` 或当前 shell 读取以下路径变量：

```text
SwiftVLN root:
${SWIFTVLN_ROOT}

ms-swift root:
${MS_SWIFT_REPO}

Qwen2.5-VL 3B base model:
${SWIFTVLN_QWEN25_MODEL_PATH}

SatNav training data:
${SWIFTVLN_SATNAV_TRAIN_DATA_PATH}
```

模板中的值需要改为本机路径；如果模型值使用公开模型 ID，则运行时可以从模型仓库
解析。离线机器应设置为已有的本地模型目录。

### Train 7. 训练脚本 dry run

dry run 只解析配置、GPU 和环境，不启动 `torchrun`：

```bash
source "${SWIFTVLN_CONDA_SH}"
conda activate swift-vln-train-update
export PYTHONNOUSERSITE=1
cd "${SWIFTVLN_ROOT}"

MODEL_FAMILY=qwen2_5_vl \
VLN_ENV_TYPE=satnav \
TRAIN_NUM_GPUS=1 \
TRAIN_DRY_RUN=true \
USE_SWANLAB=false \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

### Train 8. 最小训练 smoke

确认 GPU、模型和数据都可用后，可以跑 2 step smoke：

```bash
source "${SWIFTVLN_CONDA_SH}"
conda activate swift-vln-train-update
export PYTHONNOUSERSITE=1
cd "${SWIFTVLN_ROOT}"

MODEL_FAMILY=qwen2_5_vl \
VLN_ENV_TYPE=satnav \
MAX_SAMPLES=16 \
MAX_STEPS=2 \
SAVE_STEPS=1 \
SAVE_TOTAL_LIMIT=1 \
USE_SWANLAB=false \
TRAIN_NUM_GPUS=2 \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

训练输出默认写入：

```text
output/swiftvln/
```

主训练脚本默认激活 `swift-vln-train-update`。如需临时使用其他环境，可以覆盖：

```bash
SWIFTVLN_TRAIN_CONDA_ENV=<env_name> bash scripts/train/train_swiftvln_qwen_vl.sh
```

## Eval Environment (`swift-vln-eval-update`)

Eval 环境从零安装，不从旧 `swift-vln-eval` clone。它与训练环境的关键差异是：
eval 使用 Python `3.9` 和 Habitat `0.2.4` 栈；训练环境使用 Python `3.10`，
不能直接复用训练环境跑 Habitat eval。

### Eval 版本固定

当前验证版本：

- SwiftVLN repo: `https://github.com/Eku127/SwiftVLN.git`
- SwiftVLN branch: `master`
- SwiftVLN commit: 当前 `master` 分支 HEAD
- ms-swift repo: `https://github.com/modelscope/ms-swift.git`
- ms-swift commit: `ad7d5c5157b59afa1faceb04266274392f145745`
- SatNav repo: `https://github.com/Eku127/SatNav.git`
- SatNav commit: `c0c0e72ea4575b36d74a5e8f777942172978938e`
- Habitat-Lab repo: `https://github.com/facebookresearch/habitat-lab.git`
- Habitat-Lab tag: `v0.2.4`
- Habitat-Lab commit: `1639e1ae732ba1e84199a1a04b79c7243c3f8586`
- conda env: `swift-vln-eval-update`
- Python: `3.9`
- PyTorch: `2.8.0+cu128`
- CUDA toolkit: `12.8`
- habitat-sim: `0.2.4` (`headless_bullet`)
- habitat-lab / habitat-baselines: `0.2.4`
- ms-swift: `4.2.0.dev0`
- Transformers: `4.57.3`
- qwen-vl-utils: `0.0.14`
- numpy: `1.26.1`
- protobuf: `3.20.1`

### Eval 1. 拉取代码

建议保持四个仓库并列放在同一个 workspace 下：

```bash
mkdir -p "${WORKSPACE}"
cd "${WORKSPACE}"

git clone https://github.com/Eku127/SwiftVLN.git
cd SwiftVLN
git fetch origin
git checkout master

cd "${WORKSPACE}"
git clone https://github.com/modelscope/ms-swift.git ms-swift
cd ms-swift
git checkout ad7d5c5157b59afa1faceb04266274392f145745

cd "${WORKSPACE}"
git clone https://github.com/Eku127/SatNav.git
cd SatNav
git checkout c0c0e72ea4575b36d74a5e8f777942172978938e

cd "${WORKSPACE}"
git clone https://github.com/facebookresearch/habitat-lab.git habitat-lab-0.2.4
cd habitat-lab-0.2.4
git checkout v0.2.4
```

### Eval 2. 创建 conda 环境

```bash
source "${SWIFTVLN_CONDA_SH}"
cd "${SWIFTVLN_ROOT}"

conda env create -f environments/eval/conda.yml
conda activate swift-vln-eval-update

export PYTHONNOUSERSITE=1
python -m pip install -U pip setuptools wheel
```

如果本机已经存在同名环境并且你确认要重建，先执行：

```bash
conda env remove -n swift-vln-eval-update
```

### Eval 3. 安装 PyTorch CUDA 12.8

```bash
conda activate swift-vln-eval-update
export PYTHONNOUSERSITE=1

pip install \
  torch==2.8.0+cu128 \
  torchvision==0.23.0+cu128 \
  torchaudio==2.8.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
```

### Eval 4. 安装 eval 依赖和本地仓库

```bash
conda activate swift-vln-eval-update
export PYTHONNOUSERSITE=1

cd "${SWIFTVLN_ROOT}"
pip install -r environments/eval/requirements.txt
```

Eval 默认使用 `flash_attn`，需要在 PyTorch 和 CUDA toolkit 已就绪后单独安装：

```bash
conda activate swift-vln-eval-update
export PYTHONNOUSERSITE=1
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"

MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation
```

如果在 `/tmp` 或其他独立挂载点里安装，并遇到 `Invalid cross-device link`，
把 pip cache 和临时目录放到同一个文件系统后重试：

```bash
mkdir -p "$PWD/.tmp" "$PWD/.pip-cache"
export TMPDIR="$PWD/.tmp"
export PIP_CACHE_DIR="$PWD/.pip-cache"
MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation
```

按以下顺序安装 editable 本地仓库：

```bash
conda activate swift-vln-eval-update
export PYTHONNOUSERSITE=1

cd "${HABITAT_LAB_REPO}/habitat-lab"
pip install -e .

cd "${HABITAT_LAB_REPO}/habitat-baselines"
pip install -e .

cd "${SWIFTVLN_SATNAV_REPO}"
pip install -e .

cd "${MS_SWIFT_REPO}"
pip install -e .

cd "${SWIFTVLN_ROOT}"
pip install -e .
```

### Eval 5. 验证环境

```bash
conda activate swift-vln-eval-update
export PYTHONNOUSERSITE=1

python - <<'PY'
import importlib.metadata as m
import torch

print("torch", torch.__version__)
for pkg in [
    "torchvision",
    "torchaudio",
    "ms-swift",
    "transformers",
    "qwen-vl-utils",
    "numpy",
    "habitat-lab",
    "habitat-sim",
    "protobuf",
    "decord",
    "flash-attn",
]:
    print(pkg, m.version(pkg))

print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
print("device_count", torch.cuda.device_count())
PY
```

期望关键输出：

```text
torch 2.8.0+cu128
torchvision 0.23.0+cu128
torchaudio 2.8.0+cu128
ms-swift 4.2.0.dev0
transformers 4.57.3
qwen-vl-utils 0.0.14
numpy 1.26.1
habitat-lab 0.2.4
habitat-sim 0.2.4
protobuf 3.20.1
decord 0.6.0
flash-attn 2.8.3
cuda_available True
cuda_version 12.8
```

继续验证关键 import：

```bash
python - <<'PY'
import swiftvln.evaluation
from satnav.core.env import Env as SatNavEnv
import habitat
import habitat_sim
import decord

print("swiftvln.evaluation OK")
print("SatNavEnv OK", SatNavEnv)
print("habitat OK", habitat.__file__)
print("habitat_sim OK", habitat_sim.__file__)
print("decord OK", decord.__version__)
PY
```

`pip check` 在当前环境可能报告 `decord 0.6.0 is not supported on this platform`；
当前验证口径以 `import decord` 成功和 eval smoke 成功为准。

### Eval 6. 路径约定

默认 eval 脚本使用以下本地路径：

```text
SwiftVLN root:
${SWIFTVLN_ROOT}

ms-swift root:
${MS_SWIFT_REPO}

SatNav root:
${SWIFTVLN_SATNAV_REPO}

Habitat-Lab root:
${HABITAT_LAB_REPO}

SatNav eval data:
${SWIFTVLN_SATNAV_EVAL_DATA_PATH}

SatNav scenes:
${SWIFTVLN_SATNAV_SCENES_DIR}

R2R VLN-CE data:
${SWIFTVLN_HABITAT_R2R_EVAL_DATA_PATH}

MP3D scenes:
${SWIFTVLN_HABITAT_SCENES_DIR}

ModelScope cache:
${SWIFTVLN_MODELSCOPE_CACHE}
```

如果这些路径不存在，需要先同步数据、场景和模型 cache；eval 脚本默认不会自动下载远端模型。

### Eval 7. eval 脚本 check-only

`CHECK_ONLY=true` 只解析模型名和 eval 参数，不检查 checkpoint，也不启动 `torchrun`：

```bash
source "${SWIFTVLN_CONDA_SH}"
conda activate swift-vln-eval-update
export PYTHONNOUSERSITE=1
cd "${SWIFTVLN_ROOT}"

CHECK_ONLY=true \
ENV_TYPE=satnav \
EVAL_SPLIT=val_seen \
bash scripts/eval/eval_by_name.sh \
  swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-b1.0-pool-s2-noembed-bs64-lr2e-5-20260419-113050
```

### Eval 8. 最小 eval smoke

确认 GPU、checkpoint、模型 cache 和数据都可用后，可以跑 SatNav `MAX_EPISODES=1`
或 `2` 的最小评测：

```bash
source "${SWIFTVLN_CONDA_SH}"
conda activate swift-vln-eval-update
export PYTHONNOUSERSITE=1
cd "${SWIFTVLN_ROOT}"

MAX_EPISODES=1 \
CUDA_DEVICES=0 \
ENV_TYPE=satnav \
EVAL_SPLIT=val_seen \
bash scripts/eval/eval_by_name.sh <swiftvln_satnav_exp_name>
```

Habitat eval smoke 示例：

```bash
source "${SWIFTVLN_CONDA_SH}"
conda activate swift-vln-eval-update
export PYTHONNOUSERSITE=1
cd "${SWIFTVLN_ROOT}"

MAX_EPISODES=1 \
CUDA_DEVICES=0 \
ENV_TYPE=habitat \
EVAL_SPLIT=val_unseen \
bash scripts/eval/eval_by_name.sh <swiftvln_habitat_exp_name>
```

底层 eval 脚本默认激活 `swift-vln-eval-update`。如需临时使用其他环境，可以覆盖：

```bash
SWIFTVLN_EVAL_CONDA_ENV=<env_name> bash scripts/eval/eval_swiftvln_qwen_vl_distributed.sh
```
