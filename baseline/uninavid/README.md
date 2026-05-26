# Uni-NaVid Baseline（SatNav）

本说明面向在 SwiftVLN 仓库内运行 Uni-NaVid baseline 的训练与评测流程。

- 论文/项目：Uni-NaVid（RSS 2025）
- 上游仓库：[Uni-NaVid GitHub](https://github.com/jzhzhang/Uni-NaVid)（默认通过 `$UNINAVID_REPO` 指向本地 clone）
- 训练与评测约定：详见 `baseline/uninavid/doc/train_eval_conventions.md`

## 1. 模型

Uni-NaVid 当前需要以下本地模型文件或目录：

| 模型 | 默认路径 | 用途 | 默认下载 |
|------|----------|------|----------|
| EVA-CLIP | `baseline/uninavid/model/eva_vit_g.pth` | vision tower | 是 |
| Uni-NaVid weights | `baseline/uninavid/model/Uni-Navid` | `continue` 模式训练起点 | 是 |
| Vicuna-7B | `baseline/uninavid/model/vicuna-7b-v1.5` | `scratch` 模式训练起点 | 否，需显式指定 |

下载默认模型（EVA-CLIP + Uni-NaVid weights）：

```bash
bash baseline/uninavid/scripts/download_uninavid_models.sh
```

同时下载 Vicuna-7B 底座模型：

```bash
bash baseline/uninavid/scripts/download_uninavid_models.sh --vicuna
```

下载全部模型：

```bash
bash baseline/uninavid/scripts/download_uninavid_models.sh --all
```

其他常用选项：

```bash
# 自定义保存目录
bash baseline/uninavid/scripts/download_uninavid_models.sh --model-dir "$MODEL_DIR"

# 调整进度播报间隔
bash baseline/uninavid/scripts/download_uninavid_models.sh --monitor-interval 10

# 只下载 Uni-NaVid weights，跳过 EVA-CLIP
bash baseline/uninavid/scripts/download_uninavid_models.sh --skip-eva

# 查看帮助
bash baseline/uninavid/scripts/download_uninavid_models.sh --help
```

关键说明：

- EVA-CLIP 从 Google Storage 直连下载，约 3.9 GB。
- HuggingFace 模型默认通过 `hf-mirror.com` 下载，支持断点续传。
- 训练脚本默认读取 `baseline/uninavid/model/Uni-Navid`；若下载目录名称不同，需要保证最终路径与脚本默认值一致。

## 2. 目录结构

```text
baseline/uninavid/
├── configs/          # 训练/评测配置文件
│   ├── satnav_task.yaml
│   ├── zero1.json
│   ├── zero2.json
│   └── zero2_safe.json
├── doc/              # 环境、训练评测约定、排障与历史归档
├── model/            # 模型权重存放目录
│   ├── eva_vit_g.pth
│   ├── Uni-Navid/
│   └── vicuna-7b-v1.5/
├── scripts/          # 启动脚本
│   ├── download_uninavid_models.sh
│   ├── train_satnav.sh
│   └── eval_satnav.sh
├── src/              # 训练/评测 Python 代码
│   ├── dataset/satnav_dataset.py
│   ├── train_satnav.py
│   └── eval_satnav.py
└── README.md
```

## 3. 环境准备

Uni-NaVid 训练与评测统一使用 conda 环境：`uninavid-baseline`。

详细安装步骤见：

```text
baseline/uninavid/doc/env_setup.md
```

安装完成后激活环境：

```bash
conda activate uninavid-baseline
```

关键说明：

- 环境不包含 Habitat 依赖；评测使用 SatNav 环境。
- 评测需要 SatNav editable install：`pip install -e "$SATNAV_REPO"`
- 训练脚本依赖上游 Uni-NaVid 源码目录：`$UNINAVID_REPO`
- 当前默认 DeepSpeed 配置为 `baseline/uninavid/configs/zero1.json`；ZeRO-2 相关注意事项见 `baseline/uninavid/doc/troubleshooting.md`。

## 4. 训练

SatNav 训练入口：

```bash
bash baseline/uninavid/scripts/train_satnav.sh [continue|scratch] [EXP_NAME]
```

训练模式约定：

- `continue`：从 `baseline/uninavid/model/Uni-Navid` 继续训练，默认模式。
- `scratch`：从 `baseline/uninavid/model/vicuna-7b-v1.5` 起训。
- 兼容旧调用：只传一个非模式参数时，视为自定义 `EXP_NAME`，训练模式仍按 `continue`。

示例：

```bash
# 默认 continue
bash baseline/uninavid/scripts/train_satnav.sh

# 显式 continue
bash baseline/uninavid/scripts/train_satnav.sh continue

# 显式 scratch
bash baseline/uninavid/scripts/train_satnav.sh scratch

# 自定义实验名
bash baseline/uninavid/scripts/train_satnav.sh continue my_uninavid_exp
```

常用覆盖项：

```bash
DATA_PATH=$SATNAV_DATA_ROOT/ver_260418/trajectory_data/annotations.json \
VIDEO_FOLDER=$SATNAV_DATA_ROOT/ver_260418/trajectory_data \
NUM_GPUS=8 \
TRAIN_BSZ=24 \
GRAD_ACCUM=1 \
NUM_EPOCHS=1 \
LEARNING_RATE=1e-5 \
bash baseline/uninavid/scripts/train_satnav.sh continue
```

当前默认训练配置：

- `DATA_PATH=$SATNAV_DATA_ROOT/ver_260418/trajectory_data/annotations.json`
- `VIDEO_FOLDER=$SATNAV_DATA_ROOT/ver_260418/trajectory_data`
- `NUM_GPUS=8`
- `TRAIN_BSZ=24`
- `GRAD_ACCUM=1`
- `NUM_EPOCHS=1`
- `LEARNING_RATE=1e-5`
- `SAVE_STRATEGY=steps`
- `SAVE_STEPS=2000`
- `SAVE_TOTAL_LIMIT=1`
- `MODEL_MAX_LENGTH=1536`
- `GROUP_BY_MODALITY_LENGTH=False`
- `DS_CONFIG=baseline/uninavid/configs/zero1.json`

训练日志与产物约定：

- 输出目录：`output/uninavid-baseline/<EXP_NAME>/`
- 默认实验名格式：`uninavid-baseline-{mode}-{epochs}ep-data{ver}-bs{effective_bs}-lr{lr}-{timestamp}`
- `USE_SWANLAB=true` 时通过 `--report_to swanlab` 上报；显式 `REPORT_TO` 优先级高于 `USE_SWANLAB`。

实现方式：

- `baseline/uninavid/src/train_satnav.py` 接入 SatNav trajectory 数据。
- 训练目标是结构化四步动作文本，例如 `1. forward 2. left 3. right 4. stop`。
- 默认关闭 `GROUP_BY_MODALITY_LENGTH`，用于保持 SatNav 样本顺序更稳定。

## 5. 评测

SatNav 评测入口：

```bash
bash baseline/uninavid/scripts/eval_satnav.sh <exp_name_or_checkpoint_path>
```

支持两种模式：

- 按实验名评测：从 `output/uninavid-baseline/<EXP_NAME>/` 自动解析最新 checkpoint；也可通过 `--model_dir` 指定其他模型根目录，例如 `output/model_zoo/baseline`。
- 按 checkpoint 路径评测：直接传入绝对路径。

SatNav 评测 split 约定：

- 不传 `split`：默认顺序运行 `val_seen` 和 `val_unseen`
- 传 `val_seen` / `val_unseen` / `test`：只跑指定单个 split

常用覆盖项：

```bash
SATNAV_VERSION=ver_260418 \
bash baseline/uninavid/scripts/eval_satnav.sh \
  uninavid-baseline-continue-1ep-data260418-bs192-lr1e-5-<timestamp> \
  val_seen 8
```

也可以用命名参数从 model zoo 按名字评测：

```bash
bash baseline/uninavid/scripts/eval_satnav.sh \
  --model_dir /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/model_zoo/baseline \
  --model_name uninavid-baseline-continue-1ep-data260418-bs192-lr1e-5-20260418-203618 \
  --split val_seen \
  --gpus 8
```

评测实现约定：

- 输出目录：`results/uninavid-baseline/<EXP_NAME_or_subpath>/<split>/`
- 评测日志：`results/uninavid-baseline/<EXP_NAME_or_subpath>/<split>/eval.log`
- eval by name 会从实验名中的 `data{ver}` 自动解析 `SATNAV_VERSION`；也可用环境变量显式覆盖。
- UniNaVid eval 没有额外需要从模型名解析的窗口参数。
- `MODEL_BASE` 可用于 adapter-only checkpoint 的底座模型路径。
- `LOCAL_CACHE_DIR` 可用于指定本地 checkpoint cache。
- 评测默认使用确定性解码，prompt 语义是预测 next four actions，评测端按动作词正则提取并截断到最多 4 个动作。
