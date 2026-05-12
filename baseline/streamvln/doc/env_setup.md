# StreamVLN 训练 + 评测环境安装教程

> 环境名：`streamvln-baseline`
> 用途：SatNav 数据 finetune StreamVLN，以及在 SatNav 环境下进行在线评测
> 已验证：H100/CUDA 12.x 环境，PyTorch 2.5.1 + FlashAttention 2.8.3

---

## 路径变量

以下命令默认从 SwiftVLN 仓库根目录执行：

```bash
export SWIFTVLN_ROOT="$PWD"
export WORKSPACE="${SWIFTVLN_ROOT}/.."
export CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
export STREAMVLN_REPO="${STREAMVLN_REPO:-${WORKSPACE}/StreamVLN}"
export SATNAV_REPO="${SATNAV_REPO:-${WORKSPACE}/SatNav}"
export FLASH_ATTN_WHL="${FLASH_ATTN_WHL:-${WORKSPACE}/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp39-cp39-linux_x86_64.whl}"
```

如果本机路径不同，只需要覆盖这些变量，不需要改文档里的命令。

---

## 前提条件

- Conda 已安装，并可通过 `${CONDA_HOME}/etc/profile.d/conda.sh` 激活。
- StreamVLN 上游仓库已 clone 至 `${STREAMVLN_REPO}`。
- SatNav 仓库已 clone 至 `${SATNAV_REPO}`。
- 本机存在 FlashAttention wheel：`${FLASH_ATTN_WHL}`。
- 目标模型目录已准备在 `baseline/streamvln/model/` 下。

---

## 安装步骤

### Step 1：创建 conda 环境（Python 3.9）

```bash
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda create -n streamvln-baseline python=3.9 -y
conda activate streamvln-baseline
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

```bash
pip install -r baseline/streamvln/requirements.txt
```

---

### Step 5：安装 SatNav（评测必需，editable install）

```bash
pip install -e "${SATNAV_REPO}"
```

---

### Step 6：验证安装

```bash
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate streamvln-baseline

python -c "
import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
import flash_attn; print('flash_attn:', flash_attn.__version__)
import deepspeed; print('deepspeed:', deepspeed.__version__)
import transformers; print('transformers:', transformers.__version__)
import peft; print('peft:', peft.__version__)
import decord; print('decord: OK')
import satnav; print('satnav:', satnav.__file__)
print('=== ALL CHECKS PASSED ===')
"
```

预期输出：

```text
torch: 2.5.1+cu121 | cuda: True
flash_attn: 2.8.3
deepspeed: 0.14.4
transformers: 4.45.1
peft: 0.5.0
decord: OK
satnav: <SATNAV_REPO>/satnav/__init__.py
=== ALL CHECKS PASSED ===
```

---

## 已验证的包版本清单

| 包 | 版本 |
|----|------|
| torch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| flash_attn | 2.8.3 |
| transformers | 4.45.1 |
| tokenizers | 0.20.3 |
| accelerate | 0.28.0 |
| deepspeed | 0.14.4 |
| peft | 0.5.0 |
| decord | 0.6.0 |
| satnav | editable install from `${SATNAV_REPO}` |

---

## 版本选择原因（重要）

| 约束 | 原因 |
|------|------|
| Python 3.9 | 匹配当前 FlashAttention cp39 wheel |
| `torch==2.5.1` | 与 StreamVLN baseline 工作环境和 FlashAttention 2.8.3 wheel 对齐 |
| `transformers==4.45.1` | 与当前 `baseline/streamvln/requirements.txt` 固定版本一致 |
| `deepspeed==0.14.4` | 与当前 ZeRO-2 配置和训练脚本一致 |

---

## 注意事项

- 训练脚本会把 `${STREAMVLN_REPO}` 加入 `PYTHONPATH`，该上游源码目录必须存在。
- 若使用 ModelScope 下载模型，需要额外安装 `modelscope`。
- H100 上如需编译 CUDA op，可设置 `CUDA_HOME` 指向支持 `sm_90` 的 CUDA Toolkit。
