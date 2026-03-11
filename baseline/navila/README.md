# NaVILA Baseline（SatNav）

本说明面向在 SwiftVLN 仓库内运行 NaVILA baseline 的训练与评测流程。

- 论文：[NaVILA: Legged Robot Vision-Language-Action Model for Navigation (RSS'25)](https://arxiv.org/abs/2412.04453)
- 上游仓库：[NaVILA GitHub](https://github.com/a8cheng/NaVILA)（已 clone 至 `/mnt/data1/home/jiangjiajun/workspace/NaVILA`）

## 1. 模型

NaVILA 提供两个官方模型（**仅 HuggingFace 可用，ModelScope 上没有**）：

| 模型 | HuggingFace Repo | 用途 |
|------|-------------------|------|
| Pretrain checkpoint | `a8cheng/navila-siglip-llama3-8b-v1.5-pretrain` | SFT 训练起点 |
| SFT-trained model | `a8cheng/navila-llama3-8b-8f` | 评测用最终模型 |

基础架构：**SigLIP**（vision encoder）+ **LLaMA-3-8B**（LLM backbone），基于 NVIDIA VILA 框架。

下载模型：

```bash
# 下载全部模型（pretrain + SFT）
bash baseline/navila/scripts/download_model.sh

# 仅下载 pretrain 模型
bash baseline/navila/scripts/download_model.sh \
  --repo a8cheng/navila-siglip-llama3-8b-v1.5-pretrain

# 仅下载 SFT 评测模型
bash baseline/navila/scripts/download_model.sh \
  --repo a8cheng/navila-llama3-8b-8f
```

## 2. 目录结构

```
baseline/navila/
├── configs/          # 训练/评测配置文件
├── model/            # 模型权重存放目录
│   ├── navila-siglip-llama3-8b-v1.5-pretrain/
│   └── navila-llama3-8b-8f/
├── scripts/          # 启动脚本
│   └── download_model.sh
├── src/              # 训练/评测 Python 代码
└── README.md
```

## 3. 环境准备

NaVILA 训练环境（无需 Habitat）：

```bash
cd /mnt/data1/home/jiangjiajun/workspace/NaVILA
./environment_setup.sh navila
conda activate navila
```

该脚本自动完成：conda 创建 Python 3.10 环境 → CUDA Toolkit → FlashAttention2 → VILA editable install → Transformers v4.37.2 + 补丁。

## 4. 训练

TODO：适配 SatNav 数据的训练脚本。

## 5. 评测

TODO：适配 SatNav 数据的评测脚本。
