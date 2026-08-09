# Uni-NaVid 训练 + 评测环境安装教程

> 环境名：`uninavid-baseline`
> 用途：SatNav 数据 finetune Uni-NaVid，以及在 SatNav 环境下进行在线评测
> 已验证：H100/CUDA 12.x 环境，PyTorch 2.5.1 + FlashAttention 2.8.3

---

## 路径变量

以下命令默认从 SwiftVLN 仓库根目录执行：

```bash
export SWIFTVLN_ROOT="$PWD"
export WORKSPACE="${SWIFTVLN_ROOT}/.."
export CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
export UNINAVID_REPO="${UNINAVID_REPO:-${WORKSPACE}/Uni-NaVid}"
export SWIFTVLN_SATNAV_REPO="${SWIFTVLN_SATNAV_REPO:-${WORKSPACE}/SatNav}"
export FLASH_ATTN_WHL="${FLASH_ATTN_WHL:-${WORKSPACE}/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp39-cp39-linux_x86_64.whl}"
```

如果本机路径不同，只需要覆盖这些变量，不需要改文档里的命令。

---

## 前提条件

- Conda 已安装，并可通过 `${CONDA_HOME}/etc/profile.d/conda.sh` 激活。
- 本机存在 FlashAttention wheel：`${FLASH_ATTN_WHL}`。
- Uni-NaVid 上游仓库已 clone 至 `${UNINAVID_REPO}`。
- SatNav 仓库已 clone 至 `${SWIFTVLN_SATNAV_REPO}`。

---

## 安装步骤

### Step 1：创建 conda 环境（Python 3.9）

```bash
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda create -n uninavid-baseline python=3.9 -y
conda activate uninavid-baseline
pip install --upgrade pip
```

> **为什么用 Python 3.9**：当前预编译 FlashAttention wheel 是 cp39 版本，必须匹配。

---

### Step 2：安装 PyTorch 2.5.1（CUDA 12.1）

```bash
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

---

### Step 3：从 whl 安装 FlashAttention

```bash
pip install "${FLASH_ATTN_WHL}"
```

---

### Step 4：安装训练依赖

精确 pin 版本（版本选择原因见下方说明）：

```bash
pip install \
  transformers==4.34.1 \
  tokenizers==0.14.1 \
  huggingface-hub==0.23.2 \
  accelerate==0.27.2 \
  peft==0.6.0 \
  --no-deps

pip install deepspeed sentencepiece decord wandb \
  scikit-learn fairscale timm einops einops-exts \
  shortuuid opencv-python-headless
```

---

### Step 5：安装 Uni-NaVid 包

```bash
cd "${UNINAVID_REPO}"
pip install -e . --no-deps
```

---

### Step 6：安装 SatNav（评测必需，editable install）

```bash
pip install -e "${SWIFTVLN_SATNAV_REPO}"
```

> SatNav 的依赖（omegaconf、selenium、pillow 等）会一并安装，无需额外指定。

---

### Step 7：验证安装

```bash
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate uninavid-baseline

python -c "
import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
import flash_attn; print('flash_attn:', flash_attn.__version__)
import deepspeed; print('deepspeed:', deepspeed.__version__)
import transformers; print('transformers:', transformers.__version__)
import peft; print('peft:', peft.__version__)
import decord; print('decord: OK')
from uninavid.model import LlavaLlamaAttForCausalLM; print('LlavaLlamaAttForCausalLM: OK')
from uninavid.train.train import train; print('train module: OK')
import satnav; print('satnav:', satnav.__file__)
from satnav.core.env import Env as SatNavEnv; print('SatNavEnv: OK')
print('=== ALL CHECKS PASSED ===')
"
```

预期输出：

```text
torch: 2.5.1+cu121 | cuda: True
flash_attn: 2.8.3
deepspeed: 0.18.7
transformers: 4.34.1
peft: 0.6.0
decord: OK
LlavaLlamaAttForCausalLM: OK
train module: OK
satnav: <SWIFTVLN_SATNAV_REPO>/satnav/__init__.py
SatNavEnv: OK
=== ALL CHECKS PASSED ===
```

---

## 已验证的包版本清单

| 包 | 版本 |
|----|------|
| torch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| flash_attn | 2.8.3 |
| transformers | 4.34.1 |
| tokenizers | 0.14.1 |
| huggingface-hub | 0.23.2 |
| accelerate | 0.27.2 |
| peft | 0.6.0 |
| deepspeed | 0.18.7 |
| decord | 0.6.0 |
| timm | 1.0.25 |
| einops | 0.8.2 |
| einops-exts | 0.0.4 |
| fairscale | 0.4.13 |
| scikit-learn | 1.6.1 |
| sentencepiece | 0.2.1 |
| wandb | 0.25.0 |
| opencv-python-headless | 4.13.0 |
| satnav | editable install from `${SWIFTVLN_SATNAV_REPO}` |

---

## 版本选择原因（重要）

Uni-NaVid 的原始要求是 `transformers==4.31.0`，但在新版 torch（2.5）下需要调整：

| 约束 | 原因 |
|------|------|
| `transformers==4.34.1`（不能 4.35+） | `llava_trainer.py` 依赖 `ShardedDDPOption`，该 API 在 transformers 4.35.0 被移除 |
| `transformers==4.34.1`（不能 4.36+） | transformers 4.36.0 将 `llava` 注册进 AutoConfig，与 Uni-NaVid 的自定义注册冲突 |
| `huggingface-hub==0.23.2` | accelerate 0.27+ 需要 `split_torch_state_dict_into_shards`，此函数从 0.23 开始可用 |
| `peft==0.6.0` | peft 0.17+ 需要 `transformers.EncoderDecoderCache`（4.46+ 才有） |
| `torch==2.5.1` | 与当前 FlashAttention 2.8.3 wheel 和训练代码对齐 |
| Python 3.9 | 匹配当前 FlashAttention cp39 wheel |

---

## 注意事项

- 所有 `pip install` 中 `uninavid 1.0` 的版本不兼容警告均为预期行为，可安全忽略（pyproject.toml 中的原始 pin 过时）。
- FutureWarning（`torch.utils._pytree`，timm deprecation）属于无害警告，不影响训练与评测。
- 环境不包含 Habitat 依赖（`habitat-lab`, `habitat-sim`）；评测使用 SatNav 环境。
