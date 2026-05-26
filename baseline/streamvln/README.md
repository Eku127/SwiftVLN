# StreamVLN Baseline（SatNav）

本说明面向在 SwiftVLN 仓库内运行 StreamVLN baseline 的训练与评测流程。

- 上游仓库：默认通过 `$STREAMVLN_REPO` 指向本地 StreamVLN clone
- 运行范围：当前集成面向 SatNav trajectory 数据训练和 SatNav 在线评测。
- 训练/评测 skill：`.codex/skills/run-streamvln-baseline/SKILL.md`

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
- 目标 SatNav 数据版本需包含 `trajectory_data` 和 `episodes/eval`。
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
SATNAV_VERSION=ver_260418 \
GPUS_PER_NODE=8 \
BATCH_SIZE=3 \
GRAD_ACCUM=2 \
NUM_EPOCHS=1 \
LEARNING_RATE=2e-5 \
bash baseline/streamvln/scripts/train_satnav.sh continue
```

当前默认训练配置：

- `SATNAV_VERSION=ver_260418`
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
- 默认实验名格式：`streamvln-baseline-{mode}-{epochs}ep-f{frames}h{history}s{future}-data{ver}-bs{effective_bs}-lr{lr}-{timestamp}`

实现方式：

- 训练脚本使用上游 StreamVLN/LLaVA 代码路径，不在本仓库复制完整模型实现。
- `baseline/streamvln/src/train_satnav.py` 负责 SatNav trajectory 数据接入。
- 默认使用 `baseline/streamvln/configs/zero2.json` 做 DeepSpeed 训练。
- 可通过 `USE_SWANLAB=true` 开启 SwanLab；`USE_WXWORK_NOTIFICATION=true` 可打开企业微信通知。

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
```

### 5.3 行为说明

- 脚本会在 `<model_dir>/<model_name>/` 下选择编号最大的 `checkpoint-*`。
- 脚本会从模型名里的 `f32h8s4` 解析 `frames=32`、`history=8`、`future_steps=4`。
- 脚本不会从模型名里的数据版本字段选择 eval 数据；eval 数据和 split 都由 `satnav_task.yaml` 控制。
- 输出目录：`results/streamvln-baseline/<model_name>/<split>/`。
- 评测日志：`results/streamvln-baseline/<model_name>/<split>/eval.log`。
