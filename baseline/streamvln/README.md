# StreamVLN Baseline（SatNav）

本目录提供 StreamVLN baseline 在 SatNav 数据上的训练与评测入口。
当前推荐通过以下脚本使用：

- 训练：`baseline/streamvln/scripts/train_satnav.sh`
- 评测：`baseline/streamvln/scripts/eval_satnav.sh`
- 一键串行（train + eval）：`baseline/streamvln/scripts/train_eval_satnav.sh`
- 模型下载：`baseline/streamvln/scripts/download_model.sh`

## 1. 目录与关键文件

- `src/train_satnav.py`：训练主程序（SatNav 数据集接入）
- `src/eval_satnav.py`：评测主程序（SatNav 环境评测）
- `configs/zero2.json`：DeepSpeed ZeRO-2 配置
- `configs/satnav_task.yaml`：SatNav 评测配置模板
- `src/dataset/satnav_action_dataset.py`：训练数据集实现
- `model/`：本地模型目录（官方 ckpt / base model / vision tower）

### 1.1 目录分层说明（重构后）

- `scripts/`：shell 启动入口（训练 / 评测 / 模型下载）
- `src/`：Python 实现（训练、评测、数据集）
- `configs/`：训练和评测配置
- `model/`：本地模型与视觉塔
- `checkpoints/`、`results/`：历史产物目录（legacy）；当前主流程为“模型在 `output/streamvln-baseline/`，评测结果在 `results/streamvln-baseline/`”

## 2. 运行环境

推荐环境：`streamvln-baseline`。

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate streamvln-baseline
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN
```

### 2.1 依赖安装（按需）

仓库内提供了 baseline 依赖清单：

- `baseline/streamvln/doc/requirements.streamvln-baseline.no-clip-depth-habitat.no-av.txt`
- `baseline/streamvln/doc/requirements.streamvln-train-only.txt`

示例：

```bash
pip install -r baseline/streamvln/doc/requirements.streamvln-baseline.no-clip-depth-habitat.no-av.txt
```

### 2.2 SatNav 包安装

评测依赖 `satnav` 包：

```bash
pip install -e /mnt/data1/home/jiangjiajun/workspace/SatNav
python -c "import satnav; print('satnav OK:', satnav.__file__)"
```

## 3. 模型准备

训练脚本依赖以下本地目录：

- 官方 StreamVLN checkpoint（continue 模式）
  - `baseline/streamvln/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3`
- LLaVA-Video-7B-Qwen2（scratch 模式 / tokenizer fallback）
  - `baseline/streamvln/model/LLaVA-Video-7B-Qwen2`
- vision tower
  - `baseline/streamvln/model/siglip-so400m-patch14-384`

### 3.1 下载模型

默认下载官方 StreamVLN checkpoint（hf-mirror）：

```bash
bash baseline/streamvln/scripts/download_model.sh
```

常用参数：

```bash
# 指定 repo
bash baseline/streamvln/scripts/download_model.sh \
  --repo lmms-lab/LLaVA-Video-7B-Qwen2

# 指定下载源（hf 或 modelscope）
bash baseline/streamvln/scripts/download_model.sh \
  --repo lmms-lab/LLaVA-Video-7B-Qwen2 --source modelscope

# 自定义目标目录与进度间隔
bash baseline/streamvln/scripts/download_model.sh \
  --repo Qwen/Qwen2.5-7B-Instruct \
  --name qwen2_5_7b \
  --model-dir /path/to/model \
  --monitor-interval 10
```

## 4. 训练

训练脚本：`baseline/streamvln/scripts/train_satnav.sh`

```bash
# continue（默认）：从官方 StreamVLN checkpoint 微调
bash baseline/streamvln/scripts/train_satnav.sh continue

# scratch：从 LLaVA-Video-7B-Qwen2 开始训练
bash baseline/streamvln/scripts/train_satnav.sh scratch
```

### 4.1 输出约定

训练模型输出统一写到 SwiftVLN 根目录 `output/streamvln-baseline/`（不是 `baseline/streamvln/checkpoints`）：

- 普通训练：`output/streamvln-baseline/<EXP_NAME>/`
- smoke test：`output/streamvln-baseline/smoketest/<EXP_NAME>/`
- 日志：`output/streamvln-baseline/.../train.log`
- checkpoint：`output/streamvln-baseline/.../checkpoint-*`

`EXP_NAME` 格式：

```text
streamvln-baseline-{mode}-{epochs}ep-f{frames}h{history}s{future}-data{ver}-bs{eff_bs}-lr{lr}-{timestamp}
```

### 4.2 常用环境变量

- `SATNAV_VERSION`：指定数据版本（如 `ver_260306`，默认自动检测最新）
- `NUM_EPOCHS`：训练 epoch（默认 `1`）
- `LEARNING_RATE`：学习率（默认 `2e-5`）
- `BATCH_SIZE`：单卡 batch（默认 `2`）
- `GRAD_ACCUM`：梯度累积（默认 `2`）
- `GPUS_PER_NODE`：GPU 数量（默认 `8`）
- `SAVE_STRATEGY`：`epoch` 或 `steps`（默认 `epoch`）
- `SAVE_STEPS`：`steps` 模式保存间隔（默认 `1000`）
- `USE_SWANLAB`：是否启用 SwanLab（默认 `false`）
- `USE_WXWORK_NOTIFICATION`：SwanLab 企业微信通知（默认 `false`）
- `SMOKE_TEST`：是否写入 `smoketest/` 子目录（默认 `false`）

示例：

```bash
SATNAV_VERSION=ver_260306 \
NUM_EPOCHS=1 \
LEARNING_RATE=2e-5 \
BATCH_SIZE=2 \
GRAD_ACCUM=2 \
GPUS_PER_NODE=8 \
USE_SWANLAB=true \
USE_WXWORK_NOTIFICATION=true \
  bash baseline/streamvln/scripts/train_satnav.sh continue
```

### 4.3 一键串行：训练后自动评测

脚本：`baseline/streamvln/scripts/train_eval_satnav.sh`

```bash
# 默认：continue + val_unseen
bash baseline/streamvln/scripts/train_eval_satnav.sh

# 指定模式和 split
bash baseline/streamvln/scripts/train_eval_satnav.sh continue val_unseen
```

常用环境变量：

- `SATNAV_VERSION`：指定数据版本（默认自动检测最新）
- `TRAIN_GPUS`：训练 GPU 数（默认 `8`）
- `EVAL_GPUS`：评测 GPU 数（默认 `8`）
- `CLEAN_EVAL_FIRST`：评测前是否清理该实验该 split 的旧结果（默认 `true`）

示例：

```bash
SATNAV_VERSION=ver_260306 \
TRAIN_GPUS=8 \
EVAL_GPUS=8 \
  bash baseline/streamvln/scripts/train_eval_satnav.sh continue val_unseen
```

说明：

- 脚本会自动解析训练产出的 `EXP_NAME` 并触发按名称评测；
- 内置 4 个 webhook：训练开始/结束、评测开始/结束；
- 若训练失败，会发送训练结束（FAILED）并停止，不会触发评测。

## 5. 评测

评测脚本：`baseline/streamvln/scripts/eval_satnav.sh`

支持两种调用模式。

### 5.1 按实验名评测（推荐）

```bash
# 默认 split=val_unseen, gpus=8
bash baseline/streamvln/scripts/eval_satnav.sh <EXP_NAME>

# 指定 split 和 GPU 数
bash baseline/streamvln/scripts/eval_satnav.sh <EXP_NAME> val_unseen 8

# 限制评测 episode 数（调试）
bash baseline/streamvln/scripts/eval_satnav.sh <EXP_NAME> val_seen 8 20
```

说明：

- 脚本会优先在 `output/streamvln-baseline/<EXP_NAME或子路径>/` 下找最新 `checkpoint-*`（兼容回退到 `results/streamvln-baseline/`）
- 会从 `EXP_NAME` 的 `dataXXXXXX` 字段自动解析 `SATNAV_VERSION`
- 若 checkpoint 中无 tokenizer，会自动 fallback 到本地 `LLaVA-Video-7B-Qwen2`

输出目录：

- `results/streamvln-baseline/<EXP_NAME或子路径>/<split>/`
- 日志：`results/streamvln-baseline/<EXP_NAME或子路径>/<split>/eval.log`

### 5.2 按 checkpoint 路径评测（兼容旧方式）

```bash
bash baseline/streamvln/scripts/eval_satnav.sh /path/to/checkpoint val_unseen 8
```

输出目录：

- `results/streamvln-baseline/by-path/<ckpt_name>/<split>/`

### 5.3 覆盖数据版本

```bash
SATNAV_VERSION=ver_260306 \
  bash baseline/streamvln/scripts/eval_satnav.sh <EXP_NAME> val_unseen 8
```

## 6. 数据路径约定

SatNav 数据根目录：

- `/mnt/data3/jiangjiajun/dataset/satnav_datasets`

脚本使用的关键路径：

- 训练数据：`<ver_xxxxxx>/trajectory_data`
- 评测 episodes：`<ver_xxxxxx>/episodes/eval/all_episodes.json`
- 场景目录：`/mnt/data3/jiangjiajun/dataset/satnav_datasets/scenes`

## 7. 常见问题

- `No SatNav data versions found`
  - 检查 `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_*` 是否存在
- `Official StreamVLN checkpoint not found`
  - 先执行 `bash baseline/streamvln/scripts/download_model.sh`
- `Base model not found`
  - 检查 `baseline/streamvln/model/LLaVA-Video-7B-Qwen2`
- `No checkpoint found in exp dir`
  - 训练可能未完成，查看 `output/streamvln-baseline/<EXP_NAME>/train.log`
- `tokenizer_config.json not found`
  - 正常回退逻辑，确认本地 `LLaVA-Video-7B-Qwen2` 可用
- `CUDA out of memory`
  - 降低 `BATCH_SIZE` 或 `GRAD_ACCUM`

## 8. 与旧文档差异（重要）

当前实现已切换到以下约定：

- 不再使用旧脚本名 `download_streamvln_official_model.sh`
- 训练主输出不再写入 `baseline/streamvln/checkpoints/*`
- 训练主输出改为 `output/streamvln-baseline/<EXP_NAME>/`
- 推荐评测入口为按 `EXP_NAME` 评测，而不是手工拼 checkpoint 路径

## 9. 历史安装流程（保留，Legacy）

以下内容为之前使用过的详细安装步骤，保留用于兼容老环境或排障参考。  
当前仍建议优先使用本文前面的“2. 运行环境”与 requirements 文件安装方式。

### Step 1: 删除旧环境并创建新环境

```bash
conda deactivate
conda remove -n streamvln-baseline --all -y  # 如果存在旧环境
conda create -n streamvln-train python=3.9 -y
conda activate streamvln-train
```

### Step 2: 复制 PyTorch 及 CUDA 依赖（加速安装）

```bash
conda activate streamvln-train

# 复制 torch 和 torchvision
cp -r /mnt/data1/home/jiangjiajun/miniconda3/envs/pipeline-vln/lib/python3.9/site-packages/torch* \
      /mnt/data1/home/jiangjiajun/miniconda3/envs/streamvln-train/lib/python3.9/site-packages/

# 复制 nvidia CUDA 依赖包
CUDA_PACKAGES=(
    "nvidia_cublas_cu12" "nvidia_cuda_cupti_cu12" "nvidia_cuda_nvrtc_cu12"
    "nvidia_cuda_runtime_cu12" "nvidia_cudnn_cu12" "nvidia_cufft_cu12"
    "nvidia_curand_cu12" "nvidia_cusolver_cu12" "nvidia_cusparse_cu12"
    "nvidia_nccl_cu12" "nvidia_nvjitlink_cu12" "nvidia_nvtx_cu12"
)
SRC=/mnt/data1/home/jiangjiajun/miniconda3/envs/pipeline-vln/lib/python3.9/site-packages
DST=/mnt/data1/home/jiangjiajun/miniconda3/envs/streamvln-train/lib/python3.9/site-packages
for pkg in "${CUDA_PACKAGES[@]}"; do
    [ -d "$SRC/${pkg}" ] && cp -r "$SRC/${pkg}" "$DST/" && echo "  ✓ ${pkg}"
done

# 复制 triton
cp -r $SRC/triton $DST/
echo "✓ 完成"
```

### Step 3: 安装基础依赖（阿里云镜像）

```bash
conda activate streamvln-train
pip install \
  typing_extensions==4.15.0 packaging==26.0 numpy==2.0.2 filelock==3.19.1 \
  sympy==1.13.1 networkx==3.2.1 jinja2==3.1.6 fsspec==2025.10.0 \
  mpmath==1.3.0 MarkupSafe==3.0.3 \
  --index-url https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

### Step 4: 安装训练核心依赖（阿里云镜像）

```bash
conda activate streamvln-train
pip install \
  transformers==4.45.1 tokenizers==0.20.3 accelerate==0.28.0 deepspeed==0.14.4 \
  peft==0.5.0 safetensors==0.5.3 bitsandbytes==0.41.0 huggingface-hub==0.36.2 \
  pyyaml==6.0.3 regex==2026.1.15 requests==2.32.5 tqdm==4.67.3 psutil==7.2.2 \
  hjson==3.1.0 ninja==1.13.0 nvidia-ml-py==13.590.48 py-cpuinfo==9.0.0 \
  pydantic==2.12.5 annotated-types==0.7.0 pydantic-core==2.41.5 \
  typing-inspection==0.4.2 charset_normalizer==3.4.5 idna==3.11 \
  urllib3==2.6.3 certifi==2026.2.25 \
  --index-url https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

### Step 5: 安装视觉/数据处理依赖

```bash
conda activate streamvln-train
pip install \
  pillow==11.2.1 decord==0.6.0 opencv-python==4.11.0.86 einops==0.6.1 \
  einops-exts==0.0.4 timm==1.0.15 sentencepiece==0.1.99 wandb==0.20.1 \
  datasets==2.16.1 omegaconf==2.3.0 shortuuid==1.0.13 scipy==1.13.1 \
  pandas==2.3.0 setproctitle==1.3.6 tyro==0.9.24 \
  --index-url https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 安装 av（需要 ffmpeg，conda-forge 自动处理依赖）
conda install -c conda-forge av==15.0.0 -y
```

### Step 6: 修复 protobuf 版本

```bash
conda activate streamvln-train
pip uninstall protobuf -y
pip install protobuf==3.20.1 \
  --index-url https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

### Step 7: 将 StreamVLN 仓库路径加入 Python 搜索路径

```bash
conda activate streamvln-train
echo "/mnt/data1/home/jiangjiajun/workspace/StreamVLN" > \
  /mnt/data1/home/jiangjiajun/miniconda3/envs/streamvln-train/lib/python3.9/site-packages/streamvln.pth
echo "✓ 已写入 streamvln.pth"
```

### Step 8: 安装 Flash Attention（预编译 wheel，推荐）

```bash
conda activate streamvln-train

# 下载预编译 wheel（244 MB）
wget "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.5cxx11abiFALSE-cp39-cp39-linux_x86_64.whl" \
  -O ~/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

# 安装（秒级，无需编译）
pip install ~/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp39-cp39-linux_x86_64.whl \
  --no-build-isolation
```

验证：

```bash
python -c "
import flash_attn, torch
print('flash_attn version:', flash_attn.__version__)
from flash_attn import flash_attn_func
q = torch.randn(2, 8, 4, 64, dtype=torch.float16, device='cuda')
k = torch.randn(2, 8, 4, 64, dtype=torch.float16, device='cuda')
v = torch.randn(2, 8, 4, 64, dtype=torch.float16, device='cuda')
out = flash_attn_func(q, k, v)
print('✅ GPU forward pass 成功, shape:', out.shape)
"
```
