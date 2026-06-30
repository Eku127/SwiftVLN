# OpenFly Baseline（SatNav）

本说明面向在 SwiftVLN 仓库内运行 OpenFly baseline 的训练与评测流程。

- 上游模型：`IPEC-COMMUNITY/openfly-agent-7b`
- 运行范围：当前集成是 SatNav-only baseline，不依赖外部 `OpenFly-Platform` repo。
- 配置历史：详见 `baseline/openfly/doc/config_history.md`

## 0. 上游源码 clone 与路径

当前 OpenFly baseline 的运行代码已经在本仓库 `baseline/openfly/src` 内做了
SatNav-only 适配，训练/评测脚本不会从 `OpenFly-Platform` 动态 import 代码。
但为了明确来源、对照上游实现或重建环境，建议仍按当前 workspace 使用的
OpenFly-Platform 与 SatNav commit 固定版本。SatNav 评测环境需要本地 SatNav
editable install。

| 依赖 | 推荐本地路径 | 上游仓库 | 当前使用 commit |
|------|--------------|----------|-----------------|
| OpenFly-Platform | `/mnt/data1/home/jiangjiajun/workspace/OpenFly-Platform` | `git@github.com:SHAILAB-IPEC/OpenFly-Platform.git` | `c075075497a7122bad82f5b76b9be926ad5a81b3` |
| SatNav | `/mnt/data1/home/jiangjiajun/workspace/SatNav` | `git@github.com:Eku127/SatNav.git` | `c0c0e72ea4575b36d74a5e8f777942172978938e` |

从空 workspace 准备源码：

```bash
WORKSPACE=/mnt/data1/home/jiangjiajun/workspace

git clone git@github.com:SHAILAB-IPEC/OpenFly-Platform.git "$WORKSPACE/OpenFly-Platform"
git -C "$WORKSPACE/OpenFly-Platform" checkout c075075497a7122bad82f5b76b9be926ad5a81b3

git clone git@github.com:Eku127/SatNav.git "$WORKSPACE/SatNav"
git -C "$WORKSPACE/SatNav" checkout c0c0e72ea4575b36d74a5e8f777942172978938e
```

路径约定：

```bash
export OPENFLY_PLATFORM_REPO=/mnt/data1/home/jiangjiajun/workspace/OpenFly-Platform
export SATNAV_REPO=/mnt/data1/home/jiangjiajun/workspace/SatNav
```

`OPENFLY_PLATFORM_REPO` 当前主要用于人工对照上游代码，不是训练/评测脚本的运行时
必需变量。实际训练起点来自 `baseline/openfly/model/openfly-agent-7b` 或
`baseline/openfly/model/openvlaopenvla-7b-prismatic`，模型下载见下一节。环境安装
阶段仍需 `pip install -e "$SATNAV_REPO"`。

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

下载 scratch 后端需要的 OpenVLA/Prismatic native checkpoint：

```bash
bash baseline/openfly/scripts/download_model.sh --backend scratch
```

一次性下载 continue + scratch 两套起训资产：

```bash
bash baseline/openfly/scripts/download_model.sh --backend all
```

自定义模型来源或保存目录：

```bash
MODEL_ID=openvla/openvla-7b-prismatic \
TARGET_DIR=baseline/openfly/model/openvlaopenvla-7b-prismatic \
bash baseline/openfly/scripts/download_model.sh --backend scratch
```

scratch 目录必须是 native Prismatic run 结构，至少包含：

```text
baseline/openfly/model/openvlaopenvla-7b-prismatic/
└── checkpoints/
    └── *.pt
```

训练脚本会自动选择 `checkpoints/` 下 step 最大的 `.pt`，并在第一次运行时转换为本地
HF safetensors cache；processor/tokenizer/image preprocessor 仍默认来自
`baseline/openfly/model/openfly-agent-7b`。

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
- `scratch` 后端还需要
  `baseline/openfly/model/openvlaopenvla-7b-prismatic/checkpoints/*.pt`；
  native `.pt` 到 HF safetensors 的转换会使用本地 cache，可通过
  `OPENFLY_NATIVE_HF_CACHE_DIR` 覆盖 cache 根目录。

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
SATNAV_DATASET=SatNav-v0.1 \
NUM_GPUS=8 \
TRAIN_BSZ=12 \
GRAD_ACCUM=1 \
LEARNING_RATE=2e-5 \
bash baseline/openfly/scripts/train_satnav.sh
```

当前默认训练配置：

- `SATNAV_DATASET=SatNav-v0.1`
- `SATNAV_TRAIN_DATA_DIR=$SATNAV_DATA_ROOT/SatNav-v0.1/trajectory_data`
- `OPENFLY_ACTION_FORMAT=compact`
- `OPENFLY_ACTION_HISTORY_LIMIT=16`
- `NUM_GPUS=8`
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
bash baseline/openfly/scripts/eval_satnav.sh \
  --model_dir /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/model_zoo/baseline \
  --model_name openfly-baseline-1ep-data260418-bkcontinue-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-bs96-lr2e-5-20260420-095357 \
  --gpus 8
```

SatNav 评测 split 约定：

- split 和 eval 数据只由 `baseline/openfly/configs/satnav_task.yaml` 控制。
- `SPLIT: all` 会顺序运行 `val_seen` 和 `val_unseen`。
- `DATA_PATH` 必须是 eval split 父目录，例如
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/episodes/eval`。

常用覆盖项：

```bash
bash baseline/openfly/scripts/eval_satnav.sh \
  --model_dir /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/model_zoo/baseline/HF_model \
  --model_name openfly-satnav-continue-1ep-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-lr2e-5 \
  --gpus 8 \
  --max_episodes 10
```

当前 model zoo 中可直接评测的 OpenFly 模型名：

```text
openfly-baseline-1ep-data260418-bkcontinue-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-bs96-lr2e-5-20260420-095357
openfly-baseline-1ep-data260418-bkscratch-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-bs96-lr2e-5-20260420-233110
```

当前 Hugging Face upload-ready 公开版目录名：

```text
openfly-satnav-continue-1ep-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-lr2e-5
openfly-satnav-scratch-1ep-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-lr2e-5
```

```bash
bash baseline/openfly/scripts/eval_satnav.sh \
  --model_dir /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/model_zoo/baseline/HF_model \
  --model_name openfly-satnav-scratch-1ep-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-lr2e-5 \
  --gpus 8
```

评测实现约定：

- 输出目录：`results/openfly-baseline/<EXP_NAME>/<split>/`
- `result.jsonl` 会记录 `action`、`parsed_action`、`generated_text` 和 `action_trace`。
- eval 会捕获 simulator out-of-bounds 错误，并将对应 episode 记为失败，不中断整轮评测。
- train 与 eval 使用同一个 `OPENFLY_ACTION_HISTORY_LIMIT`，默认是 `16`。
- eval 不再从模型名中的 `data{ver}` 自动解析数据版本；数据选择只来自
  `baseline/openfly/configs/satnav_task.yaml`。
- eval 仍会从模型名中的 `actcompact` / `actoriginal` 解析动作格式，并从
  `hist{N}` 解析 action history 长度；这两个字段影响模型行为，应保留在公开模型名中。
