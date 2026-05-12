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

SatNav 评测入口：

```bash
bash baseline/streamvln/scripts/eval_satnav.sh <exp_name_or_checkpoint_path>
```

支持两种模式：

- 按实验名评测：从 `output/streamvln-baseline/<EXP_NAME>/` 自动解析最新 checkpoint。
- 按 checkpoint 路径评测：直接传入绝对路径，用于兼容历史目录或手工路径。

SatNav 评测 split 约定：

- 不传 `split`：默认顺序运行 `val_seen` 和 `val_unseen`
- 传 `val_seen` / `val_unseen` / `test`：只跑指定单个 split

常用覆盖项：

```bash
SATNAV_VERSION=ver_260418 \
bash baseline/streamvln/scripts/eval_satnav.sh \
  streamvln-baseline-continue-1ep-f32h8s4-data260418-bs48-lr2e-5-<timestamp> \
  val_seen 8
```

评测实现约定：

- 输出目录：`results/streamvln-baseline/<EXP_NAME_or_subpath>/<split>/`
- 评测日志：`results/streamvln-baseline/<EXP_NAME_or_subpath>/<split>/eval.log`
- eval by name 会从实验名中的 `data{ver}` 自动解析 `SATNAV_VERSION`；也可用环境变量显式覆盖。
- 若 checkpoint 缺少 tokenizer，评测脚本会回退到 `baseline/streamvln/model/LLaVA-Video-7B-Qwen2`。
- 评测固定使用 `num_frames=32`、`num_history=8`、`num_future_steps=4`、`model_max_length=32768`。
