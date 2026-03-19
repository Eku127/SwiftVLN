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

SatNav 训练入口：

```bash
bash baseline/navila/scripts/train_satnav.sh
```

常用覆盖项：

```bash
DATA_PATH=/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260306/trajectory_data/annotations.json \
IMAGE_FOLDER=/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260306/trajectory_data \
NUM_GPUS=8 \
TRAIN_BSZ=10 \
GRAD_ACCUM=2 \
bash baseline/navila/scripts/train_satnav.sh
```

实现方式：

- 不改上游 `NaVILA` repo 的 model / trainer 主逻辑
- 在 `baseline/navila/src/dataset/satnav_dataset.py` 中把 SatNav episode 在线展开成 NaVILA 原版需要的“历史帧 + 当前帧 + 单步动作文本”
- 在 `baseline/navila/src/train_satnav.py` 中只 monkey-patch data module，再调用原版 `train()`

## 5. 评测

SatNav 评测入口：

```bash
bash baseline/navila/scripts/eval_satnav.sh navila-llama3-8b-8f
```

支持两种模式：

- 按实验名评测：从 `output/navila-baseline/<EXP_NAME>/` 自动解析 checkpoint
- 按 checkpoint 路径评测：直接传入绝对路径

SatNav 评测 split 约定：

- 不传 `split`：默认顺序运行 `val_seen` 和 `val_unseen`
- 传 `val_seen` / `val_unseen` / `test`：只跑指定单个 split

常用覆盖项：

```bash
SATNAV_VERSION=ver_260306 \
MODEL_BASE=/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/baseline/navila/model/navila-siglip-llama3-8b-v1.5-pretrain \
bash baseline/navila/scripts/eval_satnav.sh \
  /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/baseline/navila/model/navila-llama3-8b-8f \
  val_seen 1 10
```

评测实现约定：

- prompt 与原版 `NaVILA/evaluation/vlnce_baselines/navila_trainer.py` 保持一致
- 动作解析保持原版自然语言正则逻辑：`stop / move forward / turn left / turn right`
- 距离、角度会被解析成 SatNav 离散动作队列（10m 前进、15 度转向）
- 结果输出到 `results/navila-baseline/...`
