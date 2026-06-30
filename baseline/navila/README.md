# NaVILA Baseline（SatNav）

本说明面向在 SwiftVLN 仓库内运行 NaVILA baseline 的训练与评测流程。

- 论文：[NaVILA: Legged Robot Vision-Language-Action Model for Navigation (RSS'25)](https://arxiv.org/abs/2412.04453)
- 上游仓库：[NaVILA GitHub](https://github.com/AnjieCheng/NaVILA)（按第 0 节的 `$NAVILA_REPO` 路径准备本地 clone）

## 0. 上游源码 clone 与路径

当前 NaVILA baseline 只在本仓库维护 SatNav 数据接入、启动脚本和评测 wrapper；
VILA/NaVILA 的模型、trainer、Transformers/DeepSpeed patch 仍来自本地上游
NaVILA clone。SatNav 评测环境也需要本地 SatNav editable install。建议按当前
workspace 使用的 commit 固定版本：

| 依赖 | 推荐本地路径 | 上游仓库 | 当前使用 commit |
|------|--------------|----------|-----------------|
| NaVILA | `/mnt/data1/home/jiangjiajun/workspace/NaVILA` | `git@github.com:AnjieCheng/NaVILA.git` | `76b98f233dd0fff05dfcd69435eec6740febff9d` |
| SatNav | `/mnt/data1/home/jiangjiajun/workspace/SatNav` | `git@github.com:Eku127/SatNav.git` | `c0c0e72ea4575b36d74a5e8f777942172978938e` |

从空 workspace 准备源码：

```bash
WORKSPACE=/mnt/data1/home/jiangjiajun/workspace

git clone git@github.com:AnjieCheng/NaVILA.git "$WORKSPACE/NaVILA"
git -C "$WORKSPACE/NaVILA" checkout 76b98f233dd0fff05dfcd69435eec6740febff9d

git clone git@github.com:Eku127/SatNav.git "$WORKSPACE/SatNav"
git -C "$WORKSPACE/SatNav" checkout c0c0e72ea4575b36d74a5e8f777942172978938e
```

路径约定：

```bash
export NAVILA_REPO=/mnt/data1/home/jiangjiajun/workspace/NaVILA
export SATNAV_REPO=/mnt/data1/home/jiangjiajun/workspace/SatNav
```

`baseline/navila/scripts/setup_env.sh` 当前默认使用
`/mnt/data1/home/jiangjiajun/workspace/NaVILA` 安装 VILA/NaVILA，并从该目录复制
`llava/train/transformers_replace` 与 `llava/train/deepspeed_replace` patch。
因此推荐直接 clone 到上表路径。若使用其他路径，需要同步调整
`setup_env.sh` 中的 `NAVILA_REPO`，或按 `baseline/navila/doc/env_setup.md`
手动安装并使用 `pip install -e "$SATNAV_REPO"` 安装 SatNav。

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
│   ├── download_model.sh
│   ├── setup_env.sh
│   ├── train_satnav.sh
│   └── eval_satnav.sh
├── src/              # 训练/评测 Python 代码
└── README.md
```

## 3. 环境准备

NaVILA 训练与评测统一使用 conda 环境：`navila-baseline`。

优先使用本仓库内的一键安装脚本：

```bash
bash baseline/navila/scripts/setup_env.sh
conda activate navila-baseline
```

该脚本会创建 Python 3.10 环境，安装 PyTorch 2.3.0、FlashAttention 2.5.8、
VILA/NaVILA editable package，并应用 NaVILA 对 Transformers v4.37.2 和 DeepSpeed 的补丁。

详细手动安装与验证步骤见：

```text
baseline/navila/doc/env_setup.md
```

关键说明：

- 需要提前 clone 上游 NaVILA repo，并保证 `baseline/navila/scripts/setup_env.sh`
  中的 `NAVILA_REPO` 指向该路径，默认是 `/mnt/data1/home/jiangjiajun/workspace/NaVILA`。
- 评测使用 SatNav 环境，不需要 Habitat。

## 4. 训练

SatNav 训练入口：

```bash
bash baseline/navila/scripts/train_satnav.sh
```

当前默认训练数据集：

```bash
SATNAV_DATASET=SatNav-v0.1
SATNAV_TRAIN_DATA_DIR=/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/trajectory_data
```

训练模式约定：

- `scratch`：从 `baseline/navila/model/navila-siglip-llama3-8b-v1.5-pretrain` 起训
- `continue`：从 `baseline/navila/model/navila-llama3-8b-8f` 继续训练
- 默认无参调用等价于 `scratch`
- 兼容旧调用：若只传一个非模式参数，仍视为 `EXP_NAME`

示例：

```bash
# 默认 scratch
bash baseline/navila/scripts/train_satnav.sh

# 显式 scratch
bash baseline/navila/scripts/train_satnav.sh scratch

# 显式 continue
bash baseline/navila/scripts/train_satnav.sh continue
```

常用覆盖项：

```bash
SATNAV_DATASET=SatNav-v0.1 \
NUM_GPUS=8 \
TRAIN_BSZ=10 \
GRAD_ACCUM=2 \
bash baseline/navila/scripts/train_satnav.sh
```

如需临时使用其他 trajectory export，优先显式覆盖：

```bash
SATNAV_TRAIN_DATA_DIR=/path/to/trajectory_data \
bash baseline/navila/scripts/train_satnav.sh scratch
```

训练日志约定：

- 主日志：`output/navila-baseline/<EXP_NAME>/train.log`
- GPU 指标日志：`output/navila-baseline/<EXP_NAME>/gpu_metrics.log`
- 默认每 `60s` 采样一次 `nvidia-smi`，记录 `temperature.gpu / utilization.gpu / memory.used / power.draw`
- 可通过 `ENABLE_GPU_MONITOR=false` 关闭，或用 `GPU_MONITOR_INTERVAL=<秒>` 调整采样周期

实现方式：

- 不改上游 `NaVILA` repo 的 model / trainer 主逻辑
- 在 `baseline/navila/src/dataset/satnav_dataset.py` 中把 SatNav episode 在线展开成 NaVILA 原版需要的“历史帧 + 当前帧 + 单步动作文本”
- 在 `baseline/navila/src/train_satnav.py` 中只 monkey-patch data module，再调用原版 `train()`

## 5. 评测

SatNav 评测入口：

```bash
bash baseline/navila/scripts/eval_satnav.sh \
  --model_dir /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/model_zoo/baseline \
  --model_name navila-continue0404-r2-20260427-141143-sample-hk7-fs7-stopx4 \
  --gpus 8
```

SatNav 评测 split 约定：

- split 和 eval 数据只由 `baseline/navila/configs/satnav_task.yaml` 控制。
- `SPLIT: all` 会顺序运行 `val_seen` 和 `val_unseen`。
- `DATA_PATH` 必须是 eval split 父目录，例如
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/episodes/eval`。

常用覆盖项：

```bash
MODEL_BASE=baseline/navila/model/navila-siglip-llama3-8b-v1.5-pretrain \
bash baseline/navila/scripts/eval_satnav.sh \
  --model_dir /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/model_zoo/baseline \
  --model_name navila-scratch0404-r1-20260407-195154-sample-hk7-fs7-stopx4 \
  --gpus 8 \
  --max_episodes 10
```

当前 model zoo 中可直接评测的 NaVILA 模型名：

```text
navila-continue0404-r2-20260427-141143-sample-hk7-fs7-stopx4
navila-scratch0404-r1-20260407-195154-sample-hk7-fs7-stopx4
```

当前 Hugging Face upload-ready 公开版目录名：

```text
navila-satnav-continue-1ep-8f-sample-hk7-fs7-stopx4
navila-satnav-scratch-1ep-8f-sample-hk7-fs7-stopx4
```

公开版目录放在：

```text
/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/model_zoo/baseline/HF_model
```

示例：

```bash
bash baseline/navila/scripts/eval_satnav.sh \
  --model_dir /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/model_zoo/baseline/HF_model \
  --model_name navila-satnav-continue-1ep-8f-sample-hk7-fs7-stopx4 \
  --gpus 8
```

评测实现约定：

- prompt 与原版 `NaVILA/evaluation/vlnce_baselines/navila_trainer.py` 保持一致
- 动作解析保持原版自然语言正则逻辑：`stop / move forward / turn left / turn right`
- 距离、角度会被解析成 SatNav 离散动作队列（10m 前进、15 度转向）
- eval 不再从模型名中的 `data{ver}` 自动解析数据版本；数据选择只来自
  `baseline/navila/configs/satnav_task.yaml`。
- 结果输出到 `results/navila-baseline/...`
