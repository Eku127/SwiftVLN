# StreamVLN Baseline

## 🛠 环境安装 (训练专用)

本指南详细说明了如何为 StreamVLN 训练搭建一个优化的 Conda 环境。

### Step 1: 删除旧环境并创建新环境

```bash
conda deactivate
conda remove -n streamvln-baseline --all -y  # 如果存在旧环境
conda create -n streamvln-train python=3.9 -y
conda activate streamvln-train
```

### Step 2: 复制 PyTorch 及 CUDA 依赖 (加速安装)

由于 `download.pytorch.org` 速度较慢，从现有 `pipeline-vln` 环境中直接复制 `torch`、`torchvision`、`nvidia` CUDA 库和 `triton`，避免重复下载。

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

### Step 3: 安装基础依赖 (阿里云镜像)

```bash
conda activate streamvln-train
pip install \
  typing_extensions==4.15.0 packaging==26.0 numpy==2.0.2 filelock==3.19.1 \
  sympy==1.13.1 networkx==3.2.1 jinja2==3.1.6 fsspec==2025.10.0 \
  mpmath==1.3.0 MarkupSafe==3.0.3 \
  --index-url https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

### Step 4: 安装训练核心依赖 (阿里云镜像)

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

`av` 库需要 `ffmpeg` 依赖，通过 `conda-forge` 安装可以自动处理。

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

StreamVLN 没有 `setup.py`，通过 `.pth` 文件使其可被 Python 发现。

```bash
conda activate streamvln-train
echo "/mnt/data1/home/jiangjiajun/workspace/StreamVLN" > \
  /mnt/data1/home/jiangjiajun/miniconda3/envs/streamvln-train/lib/python3.9/site-packages/streamvln.pth
echo "✓ 已写入 streamvln.pth"
```

### Step 8: 安装 Flash Attention（预编译 wheel，推荐）

由于 flash-attn 源码编译耗时极长（H100 上 nvcc 易崩溃），推荐直接下载 GitHub Release 中对应的预编译 wheel。

**环境对应的 wheel（torch 2.5.1 + CUDA 12.x + Python 3.9）：**

```bash
conda activate streamvln-train

# 下载预编译 wheel（244 MB）
wget "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.5cxx11abiFALSE-cp39-cp39-linux_x86_64.whl" \
  -O ~/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

# 安装（秒级，无需编译）
pip install ~/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp39-cp39-linux_x86_64.whl \
  --no-build-isolation
```

**验证安装：**

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

> **备注**：若 GitHub 下载过慢，可在训练脚本中添加 `--attn_implementation sdpa` 参数，使用 PyTorch 原生 SDPA 代替，H100 上性能差距不大。

---

## Step 1: 下载官方模型（非 realworld）

下载官方 benchmark 模型到 `baseline/streamvln/model/`：

```bash
bash baseline/streamvln/scripts/download_streamvln_official_model.sh
```

可调整进度播报间隔（秒）：

```bash
bash baseline/streamvln/scripts/download_streamvln_official_model.sh --monitor-interval 10
```

自定义下载目录：

```bash
bash baseline/streamvln/scripts/download_streamvln_official_model.sh --model-dir /path/to/model
```

说明：
- 模型固定为：`mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3`
- 通过 `hf-mirror.com` 直连下载，无需代理，实测速度约 2–3 MB/s。
- 脚本会周期输出下载进度和实时速度，支持断点续传。
