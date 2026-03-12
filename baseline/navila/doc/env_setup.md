# NaVILA Baseline 环境安装说明

## 环境概述

| 项目 | 内容 |
|------|------|
| Conda 环境名 | `navila-baseline` |
| Python 版本 | 3.10 |
| PyTorch | 2.3.0+cu121 |
| FlashAttention | 2.5.8 (cu122 whl，运行时使用 torch 内置 CUDA) |
| Transformers | 4.37.2（HuggingFace，含 NaVILA 补丁） |
| DeepSpeed | 0.9.5（含 NaVILA mics.py 补丁） |
| Accelerate | 0.27.2 |
| 基础框架 | VILA（Editable Install，来自 `/mnt/data1/home/jiangjiajun/workspace/NaVILA`） |

## 系统前置条件

- CUDA Toolkit 12.1（`/usr/local/cuda-12.1`）
- NVIDIA Driver 支持 CUDA ≥ 12.2（本机 Driver 580.82.07，支持 CUDA 13.0）
- Conda：`/mnt/data1/home/jiangjiajun/miniconda3`

**注意**：不需要 `conda install -c nvidia cuda-toolkit`。系统已有 `/usr/local/cuda-12.1`，环境不会安装任何内容到根目录 `/`。所有包安装至 `/mnt/data1/home/jiangjiajun/miniconda3/envs/navila-baseline/`。

## 一键安装

```bash
bash /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/baseline/navila/scripts/setup_env.sh
```

脚本日志保存于：`baseline/navila/doc/setup_env_log.txt`

## 手动分步安装

### Step 1：创建 conda 环境

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda create -n navila-baseline python=3.10 -y
conda activate navila-baseline
```

### Step 2：升级 pip

```bash
pip install --upgrade pip
```

### Step 3：安装 PyTorch 2.3.0（cu121）

使用 cu121 wheel，torch 自带 CUDA 运行时（cudnn、cublas、nccl 等），不依赖系统 CUDA toolkit 版本：

```bash
pip install torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

### Step 4：安装 FlashAttention 2.5.8（预编译 whl）

使用 GitHub Releases 预编译 whl，避免本地编译（编译耗时 30 分钟以上）。
whl 版本为 `cu122+torch2.3+cp310`，由于 flash-attn 运行时使用 torch 内置 CUDA runtime，在 cu121 系统上可正常运行：

```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### Step 5：安装 VILA（NaVILA 基础框架，Editable）

使用 `--extra-index-url` 防止 pip 从 PyPI 重装 CPU 版 torch：

```bash
cd /mnt/data1/home/jiangjiajun/workspace/NaVILA
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu121
pip install -e ".[train]" --extra-index-url https://download.pytorch.org/whl/cu121
pip install -e ".[eval]" --extra-index-url https://download.pytorch.org/whl/cu121
```

主要安装内容：
- `deepspeed==0.9.5`、`ninja`、`wandb`（train extras）
- `lmms-eval`、`word2number`、`Levenshtein`（eval extras）
- `timm==0.9.12`、`decord==0.6.0`、`einops==0.6.1` 等

### Step 6：安装 Transformers v4.37.2 + 补丁

```bash
pip install git+https://github.com/huggingface/transformers@v4.37.2

site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')

# 覆盖 modeling_utils.py 及 gemma/llama/mistral/mixtral 模型实现
cp -rv /mnt/data1/home/jiangjiajun/workspace/NaVILA/llava/train/transformers_replace/* \
    $site_pkg_path/transformers/

# 覆盖 deepspeed mics.py（ZeRO 相关修复）
cp -rv /mnt/data1/home/jiangjiajun/workspace/NaVILA/llava/train/deepspeed_replace/* \
    $site_pkg_path/deepspeed/
```

补丁覆盖的文件：
- `transformers/modeling_utils.py`
- `transformers/models/gemma/`（configuration + modeling）
- `transformers/models/llama/`（configuration + modeling + tokenization）
- `transformers/models/mistral/`（configuration + modeling）
- `transformers/models/mixtral/`（configuration + modeling）
- `deepspeed/runtime/zero/mics.py`

## 安装验证

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate navila-baseline

python -c "
import torch
import flash_attn
import deepspeed
import transformers
print('torch:          ', torch.__version__)
print('flash_attn:     ', flash_attn.__version__)
print('deepspeed:      ', deepspeed.__version__)
print('transformers:   ', transformers.__version__)
print('CUDA available: ', torch.cuda.is_available())
"
```

预期输出：

```
torch:           2.3.0+cu121
flash_attn:      2.5.8
deepspeed:       0.9.5
transformers:    4.37.2
CUDA available:  True
```

## 环境激活

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate navila-baseline
```

## 关键说明

- **FlashAttention cu122 whl 兼容性**：whl 编译时使用 CUDA 12.2，但运行时使用 PyTorch 内置的 CUDA runtime（bundled cudart），实际加速计算由 GPU 驱动处理。本机驱动支持 CUDA 13.0，完全向下兼容 12.2，因此可正常运行。
- **torch.cuda.is_available() = True** 已验证。
- **不写 `/` 根目录**：所有安装路径均在 `/mnt/data1/home/jiangjiajun/miniconda3/envs/navila-baseline/` 及 pip cache `~/.cache/pip`（均在 `/mnt/data1` 7TB 磁盘上）。
