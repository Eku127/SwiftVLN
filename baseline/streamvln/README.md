# StreamVLN Baseline（SatNav）

本说明面向在 SwiftVLN 仓库内运行 StreamVLN baseline 的训练与评测流程。

- 上游仓库：按第 0 节的 `$STREAMVLN_REPO` 路径准备本地 StreamVLN clone
- 运行范围：当前集成面向 SatNav trajectory 数据训练和 SatNav 在线评测。
- 训练/评测 skill：`.codex/skills/run-streamvln-baseline/SKILL.md`

## 0. 代码来源、适配基准与上游路径

> **代码来源声明：** 本目录的 SatNav dataset adapter、训练入口和评测 wrapper，
> 以
> [`Eku127/StreamVLN@60476e81f4c01b29f1a51a7469f1cb4addbc1d62`](https://github.com/Eku127/StreamVLN/commit/60476e81f4c01b29f1a51a7469f1cb4addbc1d62)
> 为明确的上游适配与验证基准；StreamVLN 模型核心仍在运行时从该上游 clone 加载，
> 并未完整复制到本目录。SatNav 接口开发与验证基于
> [`Eku127/SatNav@c0c0e72ea4575b36d74a5e8f777942172978938e`](https://github.com/Eku127/SatNav/commit/c0c0e72ea4575b36d74a5e8f777942172978938e)。

当前 StreamVLN baseline 不在本仓库内复制完整上游实现；训练和评测会把本地
StreamVLN 上游源码加入 `PYTHONPATH`。SatNav 评测环境也需要本地 SatNav
editable install。建议按当前 workspace 使用的 commit 固定版本：

| 依赖 | 推荐本地路径 | 上游仓库 | 当前使用 commit |
|------|--------------|----------|-----------------|
| StreamVLN | `/mnt/data1/home/jiangjiajun/workspace/StreamVLN` | `git@github.com:Eku127/StreamVLN.git` | `60476e81f4c01b29f1a51a7469f1cb4addbc1d62` |
| SatNav | `/mnt/data1/home/jiangjiajun/workspace/SatNav` | `git@github.com:Eku127/SatNav.git` | `c0c0e72ea4575b36d74a5e8f777942172978938e` |

从空 workspace 准备源码：

```bash
WORKSPACE=/mnt/data1/home/jiangjiajun/workspace

git clone git@github.com:Eku127/StreamVLN.git "$WORKSPACE/StreamVLN"
git -C "$WORKSPACE/StreamVLN" checkout 60476e81f4c01b29f1a51a7469f1cb4addbc1d62

git clone git@github.com:Eku127/SatNav.git "$WORKSPACE/SatNav"
git -C "$WORKSPACE/SatNav" checkout c0c0e72ea4575b36d74a5e8f777942172978938e
```

路径约定：

```bash
export STREAMVLN_REPO=/mnt/data1/home/jiangjiajun/workspace/StreamVLN
export SATNAV_REPO=/mnt/data1/home/jiangjiajun/workspace/SatNav
export PYTHONPATH="${STREAMVLN_REPO}:${STREAMVLN_REPO}/streamvln:${PYTHONPATH:-}"
```

当前 `train_satnav.sh` 和 `eval_satnav.sh` 默认也会把
`/mnt/data1/home/jiangjiajun/workspace/StreamVLN` 加入 `PYTHONPATH`，因此最稳妥的
做法是 clone 到上表路径。若 clone 到其他位置，启动前显式设置上面的
`PYTHONPATH`，并在环境安装阶段使用 `pip install -e "$SATNAV_REPO"`。

## 1. 模型

StreamVLN 训练与评测依赖三个本地模型目录：

| 模型 | 默认路径 | 用途 |
|------|----------|------|
| 官方 StreamVLN checkpoint | `baseline/streamvln/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3` | `continue` 模式训练起点 |
| LLaVA-Video-7B-Qwen2 | `baseline/streamvln/model/LLaVA-Video-7B-Qwen2` | `scratch` 模式训练起点；eval tokenizer fallback |
| SigLIP vision tower | `baseline/streamvln/model/siglip-so400m-patch14-384` | 视觉塔 |

下载默认官方 StreamVLN checkpoint：

```bash
bash baseline/streamvln/scripts/download_model.sh
```

下载 LLaVA-Video-7B-Qwen2：

```bash
bash baseline/streamvln/scripts/download_model.sh \
  --repo lmms-lab/LLaVA-Video-7B-Qwen2 \
  --source modelscope
```

下载 SigLIP vision tower：

```bash
bash baseline/streamvln/scripts/download_model.sh \
  --repo google/siglip-so400m-patch14-384 \
  --name siglip-so400m-patch14-384
```

下载策略建议：

- 默认官方 checkpoint 优先从 HuggingFace hf-mirror 下载。
- LLaVA-Video-7B-Qwen2 在国内网络通常优先 ModelScope。
- 跨机器运行时需保证三个模型目录路径一致。

## 2. 目录结构

```text
baseline/streamvln/
├── configs/          # 训练/评测配置文件
│   ├── satnav_task.yaml
│   └── zero2.json
├── model/            # 模型权重存放目录
│   ├── StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3/
│   ├── LLaVA-Video-7B-Qwen2/
│   └── siglip-so400m-patch14-384/
├── scripts/          # 启动脚本
│   ├── download_model.sh
│   ├── train_satnav.sh
│   ├── train_eval_satnav.sh
│   └── eval_satnav.sh
├── src/              # 训练/评测 Python 代码
│   ├── train_satnav.py
│   └── eval_satnav.py
├── requirements.txt
└── README.md
```

## 3. 环境准备

StreamVLN 训练与评测统一使用 conda 环境：`streamvln-baseline`。

```bash
# Step 1: 创建 conda 环境
conda create -n streamvln-baseline python=3.9
conda activate streamvln-baseline

# Step 2: 安装 PyTorch（CUDA 12.1）
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Step 3: 安装 flash_attn（从预编译 whl，须匹配 torch+CUDA+Python 版本）
pip install "$FLASH_ATTN_WHL"

# Step 4: 安装其余依赖
pip install -r baseline/streamvln/requirements.txt

# Step 5: 安装 SatNav（评测必需，editable install）
pip install -e "$SATNAV_REPO"
```

关键说明：

- 若通过 ModelScope 下载模型，需额外安装 `modelscope`。
- 训练脚本会把 `$STREAMVLN_REPO` 加入 `PYTHONPATH`，因此该上游源码目录必须存在。
- 默认 SatNav 数据集为 `SatNav-v0.1`；训练需要
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/trajectory_data/annotations.json`
  以及对应 image folders，评测需要 `SatNav-v0.1/episodes/eval`。
- H100 上如需编译 CUDA op，可设置 `CUDA_HOME` 指向支持 `sm_90` 的 CUDA Toolkit。

## 4. 训练

SatNav 训练入口：

```bash
bash baseline/streamvln/scripts/train_satnav.sh [continue|scratch]
```

训练模式约定：

- `continue`：从官方 StreamVLN checkpoint 继续训练，默认模式。
- `scratch`：从 `baseline/streamvln/model/LLaVA-Video-7B-Qwen2` 起训。

示例：

```bash
# 默认 continue
bash baseline/streamvln/scripts/train_satnav.sh

# 显式 continue
bash baseline/streamvln/scripts/train_satnav.sh continue

# 显式 scratch
bash baseline/streamvln/scripts/train_satnav.sh scratch
```

常用覆盖项：

```bash
SATNAV_DATASET=SatNav-v0.1 \
GPUS_PER_NODE=8 \
BATCH_SIZE=3 \
GRAD_ACCUM=2 \
NUM_EPOCHS=1 \
LEARNING_RATE=2e-5 \
bash baseline/streamvln/scripts/train_satnav.sh continue
```

当前默认训练配置：

- `SATNAV_DATASET=SatNav-v0.1`
- `SATNAV_TRAIN_DATA_DIR=/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/trajectory_data`
- `NUM_FRAMES=32`
- `NUM_HISTORY=8`
- `NUM_FUTURE_STEPS=4`
- `BATCH_SIZE=3`
- `GRAD_ACCUM=2`
- `GPUS_PER_NODE=8`
- `LEARNING_RATE=2e-5`
- `SAVE_STRATEGY=epoch`
- `SAVE_TOTAL_LIMIT=1`
- `USE_SWANLAB=false`

训练日志与产物约定：

- 普通训练输出：`output/streamvln-baseline/<EXP_NAME>/`
- smoke 输出：`output/streamvln-baseline/smoketest/<EXP_NAME>/`
- 默认实验名格式：`streamvln-baseline-{mode}-{epochs}ep-f{frames}h{history}s{future}-data{dataset}-bs{effective_bs}-lr{lr}-{timestamp}`

实现方式：

- 训练脚本使用上游 StreamVLN/LLaVA 代码路径，不在本仓库复制完整模型实现。
- `baseline/streamvln/src/train_satnav.py` 负责 SatNav trajectory 数据接入。
- 默认使用 `baseline/streamvln/configs/zero2.json` 做 DeepSpeed 训练。
- 可通过 `USE_SWANLAB=true` 开启 SwanLab。

## 5. 评测

StreamVLN baseline 评测分两步：先确定评测数据，再指定模型目录和模型名启动 eval。

### 5.1 配置评测数据

评测数据写在 `baseline/streamvln/configs/satnav_task.yaml`：

```yaml
DATASET:
  TYPE: SatNav
  SPLIT: all
  DATA_PATH: /mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/episodes/eval
  SCENES_DIR: /mnt/data3/jiangjiajun/dataset/satnav_datasets/scenes
```

字段说明：

- `SPLIT: all`：默认依次评测 `val_seen` 和 `val_unseen`。
- `SPLIT: val_seen` / `val_unseen`：默认只评测对应 split。
- `DATA_PATH` 推荐填写 eval split 父目录。脚本会解析为 `<DATA_PATH>/<split>/all_episodes.json`。
- `SCENES_DIR` 指向 SatNav scenes 目录。

### 5.2 启动评测

推荐使用命名参数：

```bash
bash baseline/streamvln/scripts/eval_satnav.sh \
  --model_dir /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/model_zoo/baseline \
  --model_name streamvln-baseline-continue-1ep-f32h8s4-lr2e-5 \
  --gpus 8
```

常用模型名：

```text
streamvln-baseline-continue-1ep-f32h8s4-lr2e-5
streamvln-baseline-scratch-1ep-f32h8s4-lr2e-5
streamvln-satnav-continue-1ep-f32h8s4-lr2e-5
streamvln-satnav-scratch-1ep-f32h8s4-lr2e-5
```

模型名约定：

- `streamvln-baseline-*`：本地 model zoo 归档版，保留训练日志、`trainer_state.json` 和历史 `checkpoint-*` 目录。
- `streamvln-satnav-*`：对应权重的 Hugging Face upload-ready 精简版，只保留 eval/inference 所需的 safetensors、config、tokenizer 等文件。
- 两组都可以用当前 `eval_satnav.sh` 直接评测；`baseline_0418` 报告当前主引用 `streamvln-baseline-*`，对外发布/复现实验优先使用 `streamvln-satnav-*`。

### 5.3 行为说明

- 脚本会直接加载 `<model_dir>/<model_name>/` 下的 Hugging Face safetensors/bin 模型文件。
- 脚本会从模型名里的 `f32h8s4` 解析 `frames=32`、`history=8`、`future_steps=4`。
- 脚本会从模型 `config.json` 读取 `mm_vision_tower` / `vision_tower`，优先搜索本地同名视觉塔目录；若本地不存在，则保留原始 Hugging Face id 交给 Transformers 解析或下载。
- 如需手动指定视觉塔，可加 `--vision_tower /path/or/hf-id`。
- 脚本不会从模型名里的数据版本字段选择 eval 数据；eval 数据和 split 都由 `satnav_task.yaml` 控制。
- 输出目录：`results/streamvln-baseline/<model_name>/<split>/`。
- 评测日志：`results/streamvln-baseline/<model_name>/<split>/eval.log`。

## 致谢

感谢 [StreamVLN](https://github.com/OpenRobotLab/StreamVLN) 的作者和贡献者公开代码、
模型与研究成果，也感谢 LLaVA-Video、SigLIP 和 SatNav 等相关工作的作者为本适配提供
基础模型与评测环境。本目录是面向 SatNav 的非官方适配；使用相关成果时请遵循各上游
项目的许可证并引用原始工作。
