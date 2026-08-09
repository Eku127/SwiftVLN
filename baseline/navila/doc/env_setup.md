# NaVILA 训练 + 评测环境安装教程

> 环境名：`navila-baseline`
> 用途：SatNav 数据 finetune NaVILA，以及在 SatNav 环境下进行在线评测
> 已验证：H100/CUDA 12.x 环境，PyTorch 2.3.0 + FlashAttention 2.5.8

---

## 路径变量

以下命令默认从 SwiftVLN 仓库根目录执行：

```bash
export SWIFTVLN_ROOT="$PWD"
export WORKSPACE="${SWIFTVLN_ROOT}/.."
export CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
export NAVILA_REPO="${NAVILA_REPO:-${WORKSPACE}/NaVILA}"
export SWIFTVLN_SATNAV_REPO="${SWIFTVLN_SATNAV_REPO:-${WORKSPACE}/SatNav}"
```

如果本机路径不同，只需要覆盖这些变量，不需要改文档里的命令。

---

## 前提条件

- Conda 已安装，并可通过 `${CONDA_HOME}/etc/profile.d/conda.sh` 激活。
- NaVILA 上游仓库已 clone 至 `${NAVILA_REPO}`。
- SatNav 仓库已 clone 至 `${SWIFTVLN_SATNAV_REPO}`。
- NVIDIA Driver 支持 CUDA 12.x；PyTorch 使用 cu121 wheel，自带 CUDA runtime。

---

## 安装步骤

### Step 1：一键安装

优先使用仓库内脚本：

```bash
bash baseline/navila/scripts/setup_env.sh
```

该脚本会创建 `navila-baseline` 环境、安装 PyTorch/FlashAttention/VILA，并应用 NaVILA 对 Transformers 和 DeepSpeed 的补丁。

---

### Step 2：手动创建 conda 环境（可选）

如果需要排查一键安装问题，可以按以下步骤手动执行：

```bash
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda create -n navila-baseline python=3.10 -y
conda activate navila-baseline
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

使用预编译 wheel，避免本地编译：

```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

> cu122 wheel 可在 cu121 PyTorch runtime 上运行；实际要求是 GPU driver 兼容。

---

### Step 5：安装 VILA / NaVILA

```bash
cd "${NAVILA_REPO}"
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu121
pip install -e ".[train]" --extra-index-url https://download.pytorch.org/whl/cu121
pip install -e ".[eval]" --extra-index-url https://download.pytorch.org/whl/cu121
```

---

### Step 6：安装 Transformers v4.37.2 并应用补丁

```bash
pip install git+https://github.com/huggingface/transformers@v4.37.2

site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')

cp -rv "${NAVILA_REPO}/llava/train/transformers_replace/"* \
  "${site_pkg_path}/transformers/"

cp -rv "${NAVILA_REPO}/llava/train/deepspeed_replace/"* \
  "${site_pkg_path}/deepspeed/"
```

---

### Step 7：验证安装

```bash
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate navila-baseline

python -c "
import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
import flash_attn; print('flash_attn:', flash_attn.__version__)
import deepspeed; print('deepspeed:', deepspeed.__version__)
import transformers; print('transformers:', transformers.__version__)
print('=== ALL CHECKS PASSED ===')
"
```

预期输出：

```text
torch: 2.3.0+cu121 | cuda: True
flash_attn: 2.5.8
deepspeed: 0.9.5
transformers: 4.37.2
=== ALL CHECKS PASSED ===
```

---

## 已验证的包版本清单

| 包 | 版本 |
|----|------|
| torch | 2.3.0+cu121 |
| torchvision | 0.18.0+cu121 |
| flash_attn | 2.5.8 |
| transformers | 4.37.2（已应用 NaVILA 补丁） |
| deepspeed | 0.9.5（已应用 NaVILA 补丁） |
| accelerate | 0.27.2 |
| timm | 0.9.12 |
| decord | 0.6.0 |
| einops | 0.6.1 |

---

## 版本选择原因（重要）

| 约束 | 原因 |
|------|------|
| Python 3.10 | 匹配当前 NaVILA/VILA 安装脚本与 FlashAttention cp310 wheel |
| `torch==2.3.0` | 与 NaVILA 当前训练栈和 FlashAttention 2.5.8 wheel 对齐 |
| `transformers==4.37.2` | NaVILA 需要在该版本基础上覆盖模型实现补丁 |
| `deepspeed==0.9.5` | 与 NaVILA 提供的 ZeRO/MiCS patch 对齐 |

---

## 注意事项

- 不需要安装 `conda cuda-toolkit`；PyTorch wheel 已包含 CUDA runtime。
- 使用 `--extra-index-url https://download.pytorch.org/whl/cu121` 是为了避免 pip 误装 CPU 版 torch。
- 如果补丁目录不存在，先确认 `${NAVILA_REPO}` 指向的是完整 NaVILA 上游仓库。
