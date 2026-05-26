# OpenFly Baseline（SatNav）

本说明面向在 SwiftVLN 仓库内运行 OpenFly baseline 的训练与评测流程。

- 上游模型：`IPEC-COMMUNITY/openfly-agent-7b`
- 运行范围：当前集成是 SatNav-only baseline，不依赖外部 `OpenFly-Platform` repo。
- 配置历史：详见 `baseline/openfly/doc/config_history.md`

## 1. 模型

OpenFly 当前支持两种起训后端：

| 后端 | 默认模型路径 | 用途 |
|------|--------------|------|
| `continue` | `baseline/openfly/model/openfly-agent-7b` | 从 HF OpenFly checkpoint 继续训练 |
| `scratch` | `baseline/openfly/model/openvlaopenvla-7b-prismatic` | 从原始 OpenVLA Prismatic `.pt` checkpoint 初始化，再接入 HF/Trainer 流程 |

`scratch` 模式还需要一个 HF processor/tokenizer/image preprocessor 来源，默认使用：

```text
baseline/openfly/model/openfly-agent-7b
```

下载默认 OpenFly HF 模型：

```bash
bash baseline/openfly/scripts/download_model.sh
```

自定义模型来源或保存目录：

```bash
MODEL_ID=IPEC-COMMUNITY/openfly-agent-7b \
TARGET_DIR=baseline/openfly/model/openfly-agent-7b \
bash baseline/openfly/scripts/download_model.sh
```

## 2. 目录结构

```text
baseline/openfly/
├── configs/          # 训练/评测配置文件
│   ├── satnav_task.yaml
│   ├── zero1.json
│   └── zero2.json
├── doc/              # 环境与配置演进说明
├── model/            # 模型权重存放目录
│   ├── openfly-agent-7b/
│   └── openvlaopenvla-7b-prismatic/
├── scripts/          # 启动脚本
│   ├── download_model.sh
│   ├── setup_env.sh
│   ├── train_satnav.sh
│   └── eval_satnav.sh
├── src/              # 训练/评测 Python 代码
│   ├── dataset/satnav_dataset.py
│   ├── train_satnav.py
│   ├── eval_satnav.py
│   └── prompting.py
└── README.md
```

## 3. 环境准备

OpenFly 训练与评测统一使用 conda 环境：`openfly-baseline`。

```bash
bash baseline/openfly/scripts/setup_env.sh
conda activate openfly-baseline
```

该脚本会创建 Python 3.10 环境，并安装 PyTorch 2.3.0、FlashAttention 2.5.8、Transformers 4.48.1、DeepSpeed 0.14.4、SatNav editable install 等依赖。

关键说明：

- 评测只依赖 SatNav，不需要 AirSim / UnrealCV / ROS2 / TFDS。
- `continue` 后端需要 `baseline/openfly/model/openfly-agent-7b` 是正常 HF 模型目录。
- `scratch` 后端还需要本地 Prismatic/OpenVLA checkpoint 目录；native `.pt` 到 HF safetensors 的转换会使用本地 cache，可通过 `OPENFLY_NATIVE_HF_CACHE_DIR` 覆盖 cache 根目录。

## 4. 训练

SatNav 训练入口：

```bash
bash baseline/openfly/scripts/train_satnav.sh
```

训练模式约定：

- `continue`：从 `baseline/openfly/model/openfly-agent-7b` 继续训练，默认后端。
- `scratch`：从 `baseline/openfly/model/openvlaopenvla-7b-prismatic` 初始化，并使用 `openfly-agent-7b` 作为 processor/tokenizer 来源。
- 后端通过环境变量选择：`OPENFLY_BACKEND=continue|scratch`

示例：

```bash
# 默认 continue
bash baseline/openfly/scripts/train_satnav.sh

# 显式 continue
OPENFLY_BACKEND=continue \
bash baseline/openfly/scripts/train_satnav.sh

# 显式 scratch
OPENFLY_BACKEND=scratch \
bash baseline/openfly/scripts/train_satnav.sh
```

常用覆盖项：

```bash
DATA_PATH=$SATNAV_DATA_ROOT/ver_260418/trajectory_data/annotations.json \
IMAGE_FOLDER=$SATNAV_DATA_ROOT/ver_260418/trajectory_data \
NUM_GPUS=8 \
TRAIN_BSZ=12 \
GRAD_ACCUM=1 \
LEARNING_RATE=2e-5 \
bash baseline/openfly/scripts/train_satnav.sh
```

当前默认训练配置：

- `OPENFLY_ACTION_FORMAT=compact`
- `OPENFLY_ACTION_HISTORY_LIMIT=16`
- `TRAIN_BSZ=12`
- `GRAD_ACCUM=1`
- `TORCH_DTYPE=bfloat16`
- `DEEPSPEED_MODE=zero2`
- `USE_FLASH_ATTENTION_2=true`
- `LEARNING_RATE=2e-5`
- `SAVE_STEPS=10000`
- `REPORT_TO=none`

当前默认 SatNav 采样策略：

- `SATNAV_HEAD_KEEP=7`
- `SATNAV_SAMPLE_STRIDE=3`
- `SATNAV_STOP_REPEAT=2`
- `SATNAV_STOP_WINDOW=0`
- `SATNAV_TAIL_KEEP=5`
- `SATNAV_STOP_HISTORY_AUG=1`

训练日志与产物约定：

- 输出目录：`output/openfly-baseline/<EXP_NAME>/`
- checkpoint：`output/openfly-baseline/<EXP_NAME>/checkpoint-*`
- 默认实验名包含后端、动作格式、采样策略、有效 batch size 和学习率。
- `scratch` 后端会写出 `backend_meta.json`，记录 checkpoint 来源。

实现方式：

- `baseline/openfly/src/dataset/satnav_dataset.py` 将 SatNav trajectory 展开成 step-wise 训练样本。
- `baseline/openfly/src/prompting.py` 构造带 action history 的 prompt。
- `baseline/openfly/src/train_satnav.py` 使用 Transformers Trainer 执行训练。
- `compact` 动作格式只监督 `stop / forward / left / right` 四类文本动作。
- `original` 动作格式保留 OpenFly-style 8D action vector token，但当前 SatNav 只启用前 4 个合法动作维度。

## 5. 评测

SatNav 评测入口：

```bash
bash baseline/openfly/scripts/eval_satnav.sh <exp_name_or_checkpoint_path>
```

支持两种模式：

- 按实验名评测：从 `output/openfly-baseline/<EXP_NAME>/` 自动解析最新 checkpoint；也可通过 `--model_dir` 指定其他模型根目录，例如 `output/model_zoo/baseline`。
- 按 checkpoint 路径评测：直接传入绝对路径。

SatNav 评测 split 约定：

- 不传 `split`：默认顺序运行 `val_seen` 和 `val_unseen`
- 传 `val_seen` / `val_unseen` / `test`：只跑指定单个 split

常用覆盖项：

```bash
SATNAV_VERSION=ver_260418 \
OPENFLY_ACTION_HISTORY_LIMIT=16 \
bash baseline/openfly/scripts/eval_satnav.sh \
  openfly-baseline-1ep-data260418-bkcontinue-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-bs96-lr2e-5-<timestamp> \
  val_seen 8
```

也可以用命名参数从 model zoo 按名字评测：

```bash
bash baseline/openfly/scripts/eval_satnav.sh \
  --model_dir /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/model_zoo/baseline \
  --model_name openfly-baseline-1ep-data260418-bkcontinue-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-bs96-lr2e-5-20260420-095357 \
  --split val_seen \
  --gpus 8
```

评测实现约定：

- 输出目录：`results/openfly-baseline/<EXP_NAME>/<split>/`
- `result.jsonl` 会记录 `action`、`parsed_action`、`generated_text` 和 `action_trace`。
- eval 会捕获 simulator out-of-bounds 错误，并将对应 episode 记为失败，不中断整轮评测。
- train 与 eval 使用同一个 `OPENFLY_ACTION_HISTORY_LIMIT`，默认是 `16`。
- eval by name 会从实验名中的 `data{ver}` 自动解析 `SATNAV_VERSION`，从 `actcompact` / `actoriginal` 解析动作格式，并从 `hist{N}` 解析 action history 长度；环境变量或命名参数可显式覆盖。

当前已记录结果（`val_seen`, `ver_260404`）：

| Config | Ckpt | Overall SR | Boundary SR | Road SR | LandmarkSet SR | OS |
|---|---|---|---|---|---|---|
| original format, stop_window=2, no hist | 33430 | 0.0% | 0.0% | - | - | 18.8% |
| **compact + stop_window=0 + hist16** | **8000** | **12.3%** | **17.4%** | **18.1%** | **5.2%** | **23.3%** |
| compact + stop_window=0 + hist16 | 6000 | 11.9% | 15.3% | 18.2% | 5.6% | 22.8% |

关键有效改动：

1. 将动作格式从 `original` 切到 `compact`，去掉 8D vector 噪声。
2. 设置 `SATNAV_STOP_WINDOW=0`，只在真实轨迹终点监督 `stop`，降低 premature-stop bias。
3. 在 prompt 中加入历史动作：`OPENFLY_ACTION_HISTORY_LIMIT=16`。
