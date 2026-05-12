# OpenFly 训练 + 评测环境安装教程

> 环境名：`openfly-baseline`
> 用途：SatNav 数据 finetune OpenFly，以及在 SatNav 环境下进行在线评测
> 已验证：H100/CUDA 12.x 环境，PyTorch 2.3.0 + FlashAttention 2.5.8

---

## 路径变量

以下命令默认从 SwiftVLN 仓库根目录执行：

```bash
export SWIFTVLN_ROOT="$PWD"
export WORKSPACE="${SWIFTVLN_ROOT}/.."
export CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
export SATNAV_REPO="${SATNAV_REPO:-${WORKSPACE}/SatNav}"
```

如果本机路径不同，只需要覆盖这些变量，不需要改文档里的命令。

---

## 前提条件

- Conda 已安装，并可通过 `${CONDA_HOME}/etc/profile.d/conda.sh` 激活。
- SatNav 仓库已 clone 至 `${SATNAV_REPO}`。
- `continue` 后端需要 HF OpenFly 模型目录：`baseline/openfly/model/openfly-agent-7b`。
- `scratch` 后端还需要本地 Prismatic/OpenVLA checkpoint：`baseline/openfly/model/openvlaopenvla-7b-prismatic`。
- OpenFly baseline 不需要 AirSim / UnrealCV / ROS2 / TFDS。

---

## 安装步骤

### Step 1：一键安装

优先使用仓库内脚本：

```bash
bash baseline/openfly/scripts/setup_env.sh
```

该脚本会创建 `openfly-baseline` 环境，并安装 PyTorch、FlashAttention、Transformers、DeepSpeed、SatNav editable install 等依赖。

---

### Step 2：手动创建 conda 环境（可选）

如果需要排查一键安装问题，可以按以下步骤手动执行：

```bash
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda create -n openfly-baseline python=3.10 -y
conda activate openfly-baseline
pip install --upgrade pip
```

---

### Step 3：安装 PyTorch 2.3.0（CUDA 12.1）

```bash
pip install torch==2.3.0 torchvision==0.18.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

---

### Step 4：安装 FlashAttention 2.5.8

```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

---

### Step 5：安装训练与评测依赖

```bash
pip install \
  transformers==4.48.1 \
  accelerate==0.33.0 \
  deepspeed==0.14.4 \
  timm==0.9.16 \
  tokenizers==0.21.1 \
  sentencepiece \
  safetensors \
  huggingface_hub \
  numpy scipy pandas pillow opencv-python-headless \
  pyyaml tqdm omegaconf packaging ninja
```

---

### Step 6：安装 SatNav（评测必需，editable install）

```bash
pip install -e "${SATNAV_REPO}"
```

---

### Step 7：验证安装

```bash
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate openfly-baseline

python -c "
import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
import flash_attn; print('flash_attn:', flash_attn.__version__)
import deepspeed; print('deepspeed:', deepspeed.__version__)
import transformers; print('transformers:', transformers.__version__)
import satnav; print('satnav:', satnav.__file__)
print('=== ALL CHECKS PASSED ===')
"
```

预期输出：

```text
torch: 2.3.0+cu121 | cuda: True
flash_attn: 2.5.8
deepspeed: 0.14.4
transformers: 4.48.1
satnav: <SATNAV_REPO>/satnav/__init__.py
=== ALL CHECKS PASSED ===
```

---

## 已验证的包版本清单

| 包 | 版本 |
|----|------|
| torch | 2.3.0+cu121 |
| torchvision | 0.18.0+cu121 |
| flash_attn | 2.5.8 |
| transformers | 4.48.1 |
| tokenizers | 0.21.1 |
| accelerate | 0.33.0 |
| deepspeed | 0.14.4 |
| timm | 0.9.16 |
| satnav | editable install from `${SATNAV_REPO}` |

---

## 版本选择原因（重要）

| 约束 | 原因 |
|------|------|
| Python 3.10 | 匹配 OpenFly baseline 当前依赖栈和 FlashAttention cp310 wheel |
| `torch==2.3.0` | 与 FlashAttention 2.5.8 wheel 对齐，训练脚本已验证 |
| `transformers==4.48.1` | OpenFly HF/Trainer backend 当前使用的 transformers 版本 |
| `deepspeed==0.14.4` | 支持当前 ZeRO 配置和 8-GPU 训练 |

---

## 注意事项

- `scratch` 后端第一次加载 native checkpoint 时会转换为 HF safetensors cache；默认 cache 根目录可通过 `OPENFLY_NATIVE_HF_CACHE_DIR` 覆盖。
- `continue` 与 `scratch` 后端都复用 `baseline/openfly/scripts/train_satnav.sh` 和 `baseline/openfly/scripts/eval_satnav.sh`。
- 若模型目录不存在，先运行 `baseline/openfly/scripts/download_model.sh` 或手动准备对应 checkpoint。
