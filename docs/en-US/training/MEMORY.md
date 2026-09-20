# Memory Configuration

[简体中文](../../zh-CN/training/MEMORY.md) | English

For the computations behind these settings, see [historical memory and token compression](../concepts/MEMORY.md), [map memory](../concepts/MAP_MEMORY.md), and [input augmentation](../concepts/AUGMENTATION.md). Window updates are explained in [dual memory and sliding windows](../concepts/PIPELINE.md).

SwiftVLN organizes observations outside the current trajectory window as Memory. The configuration covers history-frame sampling, input augmentation, and long-term Memory compression.

Before running the command, complete the environment, model and data settings in [SwiftVLN training](README.md).

## 1. Supported environments

All history-based Memory configurations support both SatNav and Habitat. The Satellite-to-UAV Stage-A adapter and Map memory are available only for SatNav.

## 2. Memory necessity

### 2.1 SwiftVLN reference

The reference configuration uses per-frame Memory. It uniformly samples eight observations before the current window and compresses the visual tokens of each frame independently:

```bash
MEMORY_METHOD=history \
HISTORY_PROCESSOR_TYPE=per_frame \
NUM_HISTORY=8 \
USE_RANDOM=false \
LOG_BASE=1.0 \
COMPRESS_STRIDE=2 \
USE_TOME=false \
SYSTEM_PROMPT_SETTING=vanilla \
EMBEDDING_MODE=none \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

| Parameters | Reference value | Meaning |
| --- | --- | --- |
|`MEMORY_METHOD`|`history`| Use previous RGB observations as Memory |
|`HISTORY_PROCESSOR_TYPE`|`per_frame`| Compress visual tokens independently per frame and concatenate them chronologically |
|`NUM_HISTORY`|`8`| Select up to eight history frames; use all available frames when fewer than eight exist |
|`COMPRESS_STRIDE`|`2`| Each spatial dimension is compressed by stride 2, and the number of visual tokens is approximately the original `1/4`|
|`USE_TOME`|`false`| Use average pooling; use GridToMe when set to `true`|

Average pooling and GridToMe in Reference have been verified in SatNav and Habitat.

### 2.2 Short-term only

Short-term only retains the current trajectory window and provides no observations from before that window:

```bash
MEMORY_METHOD=history \
HISTORY_PROCESSOR_TYPE=per_frame \
NUM_HISTORY=0 \
USE_RANDOM=false \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

`NUM_HISTORY=0` is the current no-memory configuration. The prompt contains no `<history_memory>` token; the training target and current trajectory window remain unchanged.

## 3. History-frame sampling

Per-frame Memory uses `NUM_HISTORY` to control the number of history frames and `USE_RANDOM` with `LOG_BASE` to select the sampling strategy.

| Parameters | Default value | Meaning |
| --- | --- | --- |
|`NUM_HISTORY`|`8`| Number of history frames selected from observations before the current window |
|`USE_RANDOM`|`false`| Whether to perform uniform random sampling without replacement |
|`LOG_BASE`|`1.0`| Temporal bias for deterministic sampling; must be at least `1.0` |

When `USE_RANDOM=false`, the `i` sampling point is taken first `u=i/(NUM_HISTORY-1)`, and then mapped to the historical timeline through the following formula:

```text
t = 1 - (1 - u)^LOG_BASE
```

`LOG_BASE=1.0` samples evenly within the complete historical interval; when the value increases, more sampling points are distributed in recent observations.

### 3.1 Random sampling

```bash
MEMORY_METHOD=history \
HISTORY_PROCESSOR_TYPE=per_frame \
NUM_HISTORY=8 \
USE_RANDOM=true \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

The configuration fragment in the model name is `pf-h8-random`.

### 3.2 Temporal-biased sampling

Temporal-biased sampling uses `LOG_BASE=2.0`:

```bash
MEMORY_METHOD=history \
HISTORY_PROCESSOR_TYPE=per_frame \
NUM_HISTORY=8 \
USE_RANDOM=false \
LOG_BASE=2.0 \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

The configuration fragment in the model name is `pf-h8-b2.0`.

## 4. Input augmentation

Input augmentation adds an initial observation, relative pose, or Satellite-to-UAV Stage-A adapter to the reference Memory configuration. Other Memory parameters retain their reference values.

`EMBEDDING_MODE` selects exactly one mode:

| Mode | Configuration |
| --- | --- |
| No embedding enhancement |`EMBEDDING_MODE=none`|
| Additive pose embedding | `EMBEDDING_MODE=pose` |
| FiLM pose embedding | `EMBEDDING_MODE=posefilm` |
| Satellite-to-UAV Stage-A adapter | `EMBEDDING_MODE=uav` |

### 4.1 Initial observation

```bash
SYSTEM_PROMPT_SETTING=initial \
EMBEDDING_MODE=none \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

`SYSTEM_PROMPT_SETTING=initial` puts the first frame of the Episode into the system prompt as an uncompressed image; `vanilla` does not add this observation. The corresponding model name contains `initial-noembed`. Both environments have verified initial Training and cross-window evaluation of observation.

### 4.2 Relative pose

```bash
SYSTEM_PROMPT_SETTING=vanilla \
EMBEDDING_MODE=posefilm \
POSE_NORM_SCALE=100 \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

The `posefilm` implementation represents each image with `[delta_forward, delta_right, sin(delta_heading), cos(delta_heading)]`. An MLP maps this relative pose and injects it into visual tokens through FiLM. `POSE_NORM_SCALE` normalizes forward and lateral displacement with `tanh(position / scale)`; heading sine and cosine remain unchanged. The corresponding model name ends in `posefilm`.

Additive pose embedding uses the same set of relative pose inputs and adds pose features directly to the visual token:

```bash
SYSTEM_PROMPT_SETTING=vanilla \
EMBEDDING_MODE=pose \
POSE_NORM_SCALE=100 \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

FiLM and Additive pose embedding have been verified in SatNav and Habitat.

### 4.3 Satellite-to-UAV Stage-A adapter

Loading [Satellite-to-UAV Stage-A](S2R_STAGE_A.md)adapter:

```bash
EMBEDDING_MODE=uav \
UAV_ADAPTER_PATH=/path/to/stage-a-checkpoint.pt \
UAV_ADAPTER_TYPE=transformer_v1 \
UAV_ADAPTER_APPLY_SCOPE=all_images \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

| Parameters | Default value | Meaning |
| --- | --- | --- |
|`UAV_ADAPTER_PATH`| Empty | Stage-A adapter checkpoint path |
|`UAV_ADAPTER_TYPE`|`transformer_v1`| Adapter structure |
|`UAV_ADAPTER_APPLY_SCOPE`|`all_images`| Apply adapter to all input images |

The Satellite-to-UAV Stage-A adapter supports SatNav only.

## 5. Long-term Memory compression

### 5.1 Map memory

Map memory replaces the historical RGB with global and local maps of the SatNav explored area:

```bash
MEMORY_METHOD=map \
HISTORY_PROCESSOR_TYPE=per_frame \
USE_RANDOM=false \
USE_TOME=false \
EMBEDDING_MODE=none \
COMPRESS_STRIDE=2 \
MAP_GLOBAL_SIDE_M=1000 \
MAP_LOCAL_SIDE_M=400 \
MAP_RENDER_PX=448 \
MAP_MASK_METHOD=dilate20 \
MAP_CACHE_DIR=auto \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

| Parameters | Default value | Meaning |
| --- | --- | --- |
|`MAP_GLOBAL_SIDE_M`|`1000`| The side length of the square covered by the global map, in meters |
|`MAP_LOCAL_SIDE_M`|`400`| The side length of the local graph centered on the current position, in meters |
|`MAP_RENDER_PX`|`448`| The output side length of the global image and local image, in pixels |
|`MAP_MASK_METHOD`|`dilate20`| Explored area mask; supports `strict` or `dilate<N>`|
|`MAP_CACHE_DIR`|`auto`| Render cache directory; `auto` uses `map_cache/` in the data set directory |
|`COMPRESS_STRIDE`|`2`| Visual token compression stride of global and local graphs |

Map memory supports SatNav only. `MAP_LOCAL_SIDE_M` must not exceed `MAP_GLOBAL_SIDE_M`. Do not combine Map memory with random history sampling, GridToMe, or embedding enhancement.

### 5.2 Global Token Clustering (GTC)

GTC treats the visual tokens of all history frames as a set and compresses them into a fixed number of tokens through Soft K-Means:

```bash
MEMORY_METHOD=history \
HISTORY_PROCESSOR_TYPE=gtc \
USE_RANDOM=false \
USE_TOME=false \
GTC_OUTPUT_TOKENS=512 \
GTC_TEMPERATURE=0.1 \
GTC_NUM_ITERATIONS=1 \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

GTC is verified on SatNav and Habitat.

### 5.3 Segment Token Clustering (STC)

STC is implemented as Segment-GTC and selected with `HISTORY_PROCESSOR_TYPE=segment_gtc`. The current implementation divides the history frames into 8 time periods, executes GTC in each time period, and then splices the results in chronological order:

```bash
MEMORY_METHOD=history \
HISTORY_PROCESSOR_TYPE=segment_gtc \
USE_RANDOM=false \
USE_TOME=false \
GTC_OUTPUT_TOKENS=512 \
GTC_TEMPERATURE=0.1 \
GTC_NUM_ITERATIONS=1 \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

GTC shares the following parameters with Segment-GTC:

| Parameters | Default value | Meaning |
| --- | --- | --- |
|`GTC_OUTPUT_TOKENS`|`512`| The total number of target tokens after clustering; tokens will not be expanded when the input is insufficient |
|`GTC_TEMPERATURE`|`0.1`| Soft K-Means assignment temperature; the smaller the value, the more concentrated the assignment |
|`GTC_NUM_ITERATIONS`|`1`| Soft K-Means update times |

These processors do not use `NUM_HISTORY` to control frame count. During training, the data pipeline reads history frames at the action-prediction interval before clustering their tokens. GTC clusters all history tokens together; Segment-GTC preserves coarse temporal order through segmentation. Their model-name fragments are `gtc-k512` and `sgtc-k512`, respectively. Segment-GTC has been validated on SatNav and Habitat.

## 6. Mapping configurations to model names

| Configuration | Key parameters | Configuration fragment in model name |
| --- | --- | --- |
| SwiftVLN reference | `per_frame`, `NUM_HISTORY=8` | `pf-h8-pool-s2` |
| Short-term only | `NUM_HISTORY=0` | `pf-h0-nomem-pool-s2` |
| Random sampling | `USE_RANDOM=true` | `pf-h8-random` |
| Temporal-biased sampling | `LOG_BASE=2.0` | `pf-h8-b2.0` |
| Initial observation | `SYSTEM_PROMPT_SETTING=initial` | `initial-noembed` |
| Relative pose | `EMBEDDING_MODE=posefilm` | `posefilm` |
| Additive pose | `EMBEDDING_MODE=pose` | `pose` |
| Satellite-to-UAV Stage-A adapter | `EMBEDDING_MODE=uav` | `uav` |
| Map memory | `MEMORY_METHOD=map` | `map-g1000-l400-r448-d20-s2` |
| GTC | `HISTORY_PROCESSOR_TYPE=gtc` | `gtc-k512` |
| STC / Segment-GTC | `HISTORY_PROCESSOR_TYPE=segment_gtc` | `sgtc-k512` |

Published models and their full names are listed in [Models and checkpoints](../getting-started/CHECKPOINTS.md). For training, output directories, and resume, see [SwiftVLN training](README.md).
