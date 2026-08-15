# 模型与 Checkpoint

本章介绍 SwiftVLN 基础模型与已训练 checkpoint 的下载和配置。

## 1. 安装下载工具

```bash
python -m pip install --upgrade huggingface_hub
hf --help
```

## 2. Qwen 基础模型

### 2.1 支持的模型

SwiftVLN 支持以下基础模型：

| `MODEL_FAMILY` | Hugging Face 模型 | SwiftVLN 配置变量 |
| --- | --- | --- |
| `qwen2_5_vl` | [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | `SWIFTVLN_QWEN25_MODEL_PATH` |
| `qwen3_vl` | [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | `SWIFTVLN_QWEN3_MODEL_PATH` |

### 2.2 下载 Checkpoint

当前发布的 SatNav checkpoint 均基于 Qwen2.5-VL 3B。下载该基础模型：

```bash
export SWIFTVLN_MODEL_ROOT=/path/to/models

hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir "${SWIFTVLN_MODEL_ROOT}/Qwen2.5-VL-3B-Instruct"
```

下载 Qwen3-VL：

```bash
hf download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir "${SWIFTVLN_MODEL_ROOT}/Qwen3-VL-2B-Instruct"
```

### 2.3 配置模型路径

在 `${SWIFTVLN_ROOT}/.local/env.sh` 中配置模型路径：

```bash
export SWIFTVLN_QWEN25_MODEL_PATH="/path/to/models/Qwen2.5-VL-3B-Instruct"
export SWIFTVLN_QWEN3_MODEL_PATH="/path/to/models/Qwen3-VL-2B-Instruct"
```

基础模型的训练配置与启动方式见 [SwiftVLN 训练](../training/README.md)。

## 3. SatNav 模型

### 3.1 下载默认 Checkpoint

SatNav 模型发布在
[SwiftVLN SatNav Ablation Model Zoo](https://huggingface.co/collections/Eku127/swiftvln-satnav-ablation-model-zoo)。
默认配置模型为：

[Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed)

SatNav 默认评测与后续训练只需下载该模型；其余 10 个模型用于 ablation 对比：

```bash
export MODEL_NAME=swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed
export SWIFTVLN_HF_MODEL_ROOT="${SWIFTVLN_ROOT}/output/model_zoo/swiftvln/HF_model"
export SWIFTVLN_SATNAV_MODEL_PATH="${SWIFTVLN_HF_MODEL_ROOT}/${MODEL_NAME}"

hf download "Eku127/${MODEL_NAME}" \
  --local-dir "${SWIFTVLN_SATNAV_MODEL_PATH}"
```

下载目录包含 `config.json`、processor、tokenizer、权重索引和全部 safetensors shard：

```bash
test -f "${SWIFTVLN_SATNAV_MODEL_PATH}/config.json"
test -f "${SWIFTVLN_SATNAV_MODEL_PATH}/model.safetensors.index.json"
```

下载后的模型可以用于 SatNav 评测或作为后续训练的初始化 checkpoint。具体命令见
[SwiftVLN 评测](../evaluation/README.md)和[SwiftVLN 训练](../training/README.md)。

### 3.2 Ablation 模型

11 个模型共享以下训练配置：

| 配置 | 值 |
| --- | --- |
| 基础模型 | Qwen2.5-VL 3B |
| 训练环境 | SatNav |
| Epoch | `1` |
| 轨迹窗口 | `NUM_FRAMES=32` |
| 每轮动作数 | `NUM_FUTURE_STEPS=4` |
| 有效 batch size | `64` |
| Learning rate | `2e-5` |

各模型的差异如下：

| 模型 | 配置 |
| --- | --- |
| [`swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed`](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed) | 默认配置：无重叠，均匀采样 8 张历史帧，per-frame pooling，stride 2，无 embedding enhancement |
| [`swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h0-nomem-pool-s2-noembed`](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h0-nomem-pool-s2-noembed) | No-memory：`NUM_HISTORY=0` |
| [`swiftvln-satnav-3b-1ep-f32s4-overlap4-pf-h8-pool-s2-noembed`](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap4-pf-h8-pool-s2-noembed) | 相邻窗口重叠 4 个动作 |
| [`swiftvln-satnav-3b-1ep-f32s4-overlap16-pf-h8-pool-s2-noembed`](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap16-pf-h8-pool-s2-noembed) | 相邻窗口重叠 16 个动作 |
| [`swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-random-pool-s2-noembed`](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-random-pool-s2-noembed) | 随机采样 8 张历史帧 |
| [`swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-b2.0-pool-s2-noembed`](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-b2.0-pool-s2-noembed) | 对数历史采样：`LOG_BASE=2.0` |
| [`swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-initial-noembed`](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-initial-noembed) | `SYSTEM_PROMPT_SETTING=initial`，在 system prompt 中加入初始 observation |
| [`swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-posefilm`](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-posefilm) | Per-frame history，`EMBEDDING_MODE=posefilm` |
| [`swiftvln-satnav-3b-1ep-f32s4-overlap0-map-g1000-l400-r448-d20-s2-noembed`](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap0-map-g1000-l400-r448-d20-s2-noembed) | Map memory：global 1000 m、local 400 m、448 px、`dilate20`、stride 2 |
| [`swiftvln-satnav-3b-1ep-f32s4-overlap0-gtc-k512-noembed`](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap0-gtc-k512-noembed) | GTC history processor，输出 512 个 token |
| [`swiftvln-satnav-3b-1ep-f32s4-overlap0-sgtc-k512-noembed`](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap0-sgtc-k512-noembed) | Segment-GTC history processor，输出 512 个 token |

### 3.3 模型名称

模型名称是 SwiftVLN 评测配置的一部分。`eval_by_name.sh` 从名称中解析环境、模型族、轨迹
窗口、overlap、memory、history processor、system prompt 和 embedding enhancement。

保留 Hugging Face 仓库中的完整模型名称，并使用该名称作为下载目录名和评测参数：

```bash
python -m swiftvln.experiment parse-name "${MODEL_NAME}" --format json
```

默认模型名称的字段为：

```text
swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed
         │      │   │     │        │     │     │       └─ 无 embedding enhancement
         │      │   │     │        │     │     └───────── pooling stride 2
         │      │   │     │        │     └─────────────── per-frame，8 张历史帧
         │      │   │     │        └───────────────────── 窗口无重叠
         │      │   │     └────────────────────────────── 32 帧窗口，每轮 4 个动作
         │      │   └──────────────────────────────────── 训练 1 epoch
         │      └──────────────────────────────────────── Qwen2.5-VL 3B
         └─────────────────────────────────────────────── SatNav
```

## 4. Habitat 模型

### 4.1 Checkpoint

TBD。SwiftVLN Habitat checkpoint 尚未发布。

## 5. 下一步

- [SatNav 训练数据](../data/TRAINING_DATA_SATNAV.md)
- [Habitat 训练数据](../data/TRAINING_DATA_HABITAT.md)
- [评测数据准备](../data/EVALUATION_DATA.md)
- [SwiftVLN 训练](../training/README.md)
- [SwiftVLN 评测](../evaluation/README.md)
