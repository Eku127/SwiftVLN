# Models and Checkpoints

[简体中文](../../zh-CN/getting-started/CHECKPOINTS.md) | English

This page describes how to download and configure Qwen base models and trained SwiftVLN checkpoints.

## 1. Install the download tool

```bash
python -m pip install --upgrade huggingface_hub
hf --help
```

## 2. Qwen base models

### 2.1 Supported models

SwiftVLN supports the following base models:

|`MODEL_FAMILY`| Hugging Face model | SwiftVLN configuration variables |
| --- | --- | --- |
| `qwen2_5_vl` | [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | `SWIFTVLN_QWEN25_MODEL_PATH` |
| `qwen3_vl` | [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | `SWIFTVLN_QWEN3_MODEL_PATH` |

### 2.2 Download checkpoints

SatNav default model and ablation model are based on Qwen2.5-VL 3B. Download this base model:

```bash
export SWIFTVLN_MODEL_ROOT=/path/to/models

hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir "${SWIFTVLN_MODEL_ROOT}/Qwen2.5-VL-3B-Instruct"
```

Download Qwen3-VL:

```bash
hf download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir "${SWIFTVLN_MODEL_ROOT}/Qwen3-VL-2B-Instruct"
```

### 2.3 Configure model path

Configure the model path in `${SWIFTVLN_ROOT}/.local/env.sh`:

```bash
export SWIFTVLN_QWEN25_MODEL_PATH="/path/to/models/Qwen2.5-VL-3B-Instruct"
export SWIFTVLN_QWEN3_MODEL_PATH="/path/to/models/Qwen3-VL-2B-Instruct"
```

For the training configuration and startup method of the base model, see [SwiftVLN training](../training/README.md).

## 3. SatNav model

### 3.1 Download the default checkpoint

SatNav models are published in the [SwiftVLN SatNav Ablation Model Zoo](https://huggingface.co/collections/Eku127/swiftvln-satnav-ablation-model-zoo). The default configuration model is:

[Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed)

For SatNav default evaluation and subsequent training, you only need to download this model; the remaining 10 models are used for ablation comparison:

```bash
export MODEL_NAME=swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed
export SWIFTVLN_HF_MODEL_ROOT="${SWIFTVLN_ROOT}/output/model_zoo/swiftvln/HF_model"
export SWIFTVLN_SATNAV_MODEL_PATH="${SWIFTVLN_HF_MODEL_ROOT}/${MODEL_NAME}"

hf download "Eku127/${MODEL_NAME}" \
  --local-dir "${SWIFTVLN_SATNAV_MODEL_PATH}"
```

The download directory contains `config.json`, processor, tokenizer, weight index and all safetensors shard:

```bash
test -f "${SWIFTVLN_SATNAV_MODEL_PATH}/config.json"
test -f "${SWIFTVLN_SATNAV_MODEL_PATH}/model.safetensors.index.json"
```

The downloaded model can be used for SatNav evaluation or as an initialization checkpoint for subsequent training. For specific commands, see [SwiftVLN evaluation](../evaluation/README.md)and [SwiftVLN training](../training/README.md).

### 3.2 Ablation models

The 11 models share the following training configuration:

| Configuration | Value |
| --- | --- |
| Base model | Qwen2.5-VL 3B |
| Training environment | SatNav |
| Epoch | `1` |
| Trajectory window |`NUM_FRAMES=32`|
| Number of actions per round |`NUM_FUTURE_STEPS=4`|
| Valid batch size |`64`|
| Learning rate | `2e-5` |

The differences between the models are as follows:

| Model | Configuration |
| --- | --- |
| [`swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed`](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed)| Default configuration: no overlap, uniform sampling of 8 history frames, per-frame pooling, stride 2, no embedding enhancement |
| [`swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h0-nomem-pool-s2-noembed`](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h0-nomem-pool-s2-noembed) | No-memory: `NUM_HISTORY=0` |
| [`swiftvln-satnav-3b-1ep-f32s4-overlap4-pf-h8-pool-s2-noembed`](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap4-pf-h8-pool-s2-noembed)| Adjacent windows overlap 4 actions |
| [`swiftvln-satnav-3b-1ep-f32s4-overlap16-pf-h8-pool-s2-noembed`](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap16-pf-h8-pool-s2-noembed)| Adjacent windows overlap 16 actions |
| [`swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-random-pool-s2-noembed`](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-random-pool-s2-noembed)| Randomly sample 8 history frames |
| [`swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-b2.0-pool-s2-noembed`](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-b2.0-pool-s2-noembed)| Logarithmic history sampling: `LOG_BASE=2.0`|
| [`swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-initial-noembed`](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-initial-noembed)|`SYSTEM_PROMPT_SETTING=initial`, add initial observation in system prompt |
| [`swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-posefilm`](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-posefilm) | Per-frame history, `EMBEDDING_MODE=posefilm` |
| [`swiftvln-satnav-3b-1ep-f32s4-overlap0-map-g1000-l400-r448-d20-s2-noembed`](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap0-map-g1000-l400-r448-d20-s2-noembed) | Map memory: global 1000 m, local 400 m, 448 px, `dilate20`, stride 2 |
| [`swiftvln-satnav-3b-1ep-f32s4-overlap0-gtc-k512-noembed`](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap0-gtc-k512-noembed)| GTC history processor, output 512 tokens |
| [`swiftvln-satnav-3b-1ep-f32s4-overlap0-sgtc-k512-noembed`](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap0-sgtc-k512-noembed)| Segment-GTC history processor, output 512 tokens |

### 3.3 Backbone models

The following models use the same SwiftVLN reference memory configuration to compare different Qwen-VL backbones:

| Model | Backbone | Starting Checkpoint |
| --- | --- | --- |
| [`swiftvln-satnav-7b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed`](https://huggingface.co/Eku127/swiftvln-satnav-7b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed) | Qwen2.5-VL 7B | `Qwen/Qwen2.5-VL-7B-Instruct` |
| [`swiftvln-satnav-qwen3vl-2b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed`](https://huggingface.co/Eku127/swiftvln-satnav-qwen3vl-2b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed) | Qwen3-VL 2B | `Qwen/Qwen3-VL-2B-Instruct` |
| [`swiftvln-satnav-qwen3vl-8b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed`](https://huggingface.co/Eku127/swiftvln-satnav-qwen3vl-8b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed) | Qwen3-VL 8B | `Qwen/Qwen3-VL-8B-Instruct` |


Select the desired model name and download:

```bash
export MODEL_NAME=swiftvln-satnav-qwen3vl-2b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed
export SWIFTVLN_BACKBONE_MODEL_ROOT="${SWIFTVLN_ROOT}/output/model_zoo/backbones/HF_model"
export MODEL_PATH="${SWIFTVLN_BACKBONE_MODEL_ROOT}/${MODEL_NAME}"

hf download "Eku127/${MODEL_NAME}" \
  --local-dir "${MODEL_PATH}"
```

Explicitly pass in the download path when evaluating the backbone model:

```bash
MODEL_PATH="${MODEL_PATH}" \
bash scripts/eval/eval_by_name.sh "${MODEL_NAME}"
```

### 3.4 Model name

The model name is part of the SwiftVLN evaluation configuration. `eval_by_name.sh` parses the environment, model family, trajectory window, overlap, Memory, history processor, system prompt, and embedding enhancement from the name.

Keep the full model name in the Hugging Face repository and use that name as the download directory name and evaluation parameters:

```bash
python -m swiftvln.experiment parse-name "${MODEL_NAME}" --format json
```

The fields for the default model name are:

```text
swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed
         │      │   │     │        │     │     │       └─ no embedding enhancement
         │      │   │     │        │     │     └───────── pooling stride 2
         │      │   │     │        │     └─────────────── per-frame, 8 history frames
         │      │   │     │        └───────────────────── non-overlapping windows
         │      │   │     └────────────────────────────── 32-frame window, 4 actions per turn
         │      │   └──────────────────────────────────── trained for 1 epoch
         │      └──────────────────────────────────────── Qwen2.5-VL 3B
         └─────────────────────────────────────────────── SatNav
```

## 4. Habitat models

### 4.1 Checkpoint

TBD. SwiftVLN Habitat checkpoint has not been released yet.
