# Memory 训练配置

SwiftVLN 将当前轨迹窗口之外的观测组织为 Memory。Memory 配置包括 history-frame
sampling、input augmentation 和 long-term Memory compression。

运行命令前先完成 [SwiftVLN 训练](README.md)中的环境、模型和数据设置。

## 1. 适用环境

SatNav 与 Habitat 共用 history-based Memory 接口。Habitat 已完成 8 卡训练、checkpoint
加载和跨窗口评测 smoke，结果见
[Habitat Memory Smoke Test](../../../reports/habitat_memory_smoke_73_2026-08-15.md)。

| 配置 | SatNav | Habitat |
| --- | --- | --- |
| Per-frame reference | 已验证 | 已验证 |
| Short-term only / no-memory | 已验证 | 已验证 |
| Random sampling | 已验证 | 已验证 |
| Temporal-biased sampling | 已验证 | 已验证 |
| Initial observation | 已验证 | 已验证 |
| FiLM pose embedding | 已验证 | 已验证 |
| GTC | 已验证 | 已验证 |
| Segment-GTC / STC | 已验证 | 已验证 |
| GridToMe | 已验证 | 已验证 |
| Additive pose embedding | 已验证 | 已验证 |
| S2R Stage-A adapter | 已验证 | 不支持 |
| Map memory | 已验证 | 不支持 |

## 2. Memory necessity

### 2.1 SwiftVLN reference

Reference 配置使用 per-frame Memory，从当前窗口之前的观测中均匀采样 8 帧，并分别压缩
每帧的视觉 token：

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

| 参数 | Reference 值 | 含义 |
| --- | --- | --- |
| `MEMORY_METHOD` | `history` | 使用历史 RGB 观测作为 Memory |
| `HISTORY_PROCESSOR_TYPE` | `per_frame` | 逐帧压缩历史视觉 token，并按时间顺序拼接 |
| `NUM_HISTORY` | `8` | 最多选取 8 张历史帧；可用帧不足时使用全部历史帧 |
| `COMPRESS_STRIDE` | `2` | 每个空间维度按 stride 2 压缩，视觉 token 数约为原来的 `1/4` |
| `USE_TOME` | `false` | 使用 average pooling；设为 `true` 时使用 GridToMe |

Reference 中的 average pooling 与 GridToMe 均已在 SatNav 和 Habitat 验证。

### 2.2 Short-term only

Short-term only 保留当前轨迹窗口，但不再向模型提供窗口之前的历史观测：

```bash
MEMORY_METHOD=history \
HISTORY_PROCESSOR_TYPE=per_frame \
NUM_HISTORY=0 \
USE_RANDOM=false \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

`NUM_HISTORY=0` 是当前实现中的 no-memory 配置。此时 prompt 中不插入
`<history_memory>`，训练目标与当前轨迹窗口保持不变。

## 3. History-frame sampling

Per-frame Memory 使用 `NUM_HISTORY` 控制历史帧数量，并通过 `USE_RANDOM` 和 `LOG_BASE`
选择采样策略。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `NUM_HISTORY` | `8` | 从当前窗口之前的观测中选取的历史帧数 |
| `USE_RANDOM` | `false` | 是否执行无放回均匀随机采样 |
| `LOG_BASE` | `1.0` | 确定性采样的时间偏置，取值必须不小于 `1.0` |

当 `USE_RANDOM=false` 时，第 `i` 个采样点先取
`u=i/(NUM_HISTORY-1)`，再通过下式映射到历史时间轴：

```text
t = 1 - (1 - u)^LOG_BASE
```

`LOG_BASE=1.0` 在完整历史区间内均匀取样；数值增大时，更多采样点分布在近期观测。

### 3.1 Random sampling

```bash
MEMORY_METHOD=history \
HISTORY_PROCESSOR_TYPE=per_frame \
NUM_HISTORY=8 \
USE_RANDOM=true \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

模型名中的配置片段为 `pf-h8-random`。

### 3.2 Temporal-biased sampling

Temporal-biased sampling 使用 `LOG_BASE=2.0`：

```bash
MEMORY_METHOD=history \
HISTORY_PROCESSOR_TYPE=per_frame \
NUM_HISTORY=8 \
USE_RANDOM=false \
LOG_BASE=2.0 \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

模型名中的配置片段为 `pf-h8-b2.0`。

## 4. Input augmentation

Input augmentation 在 reference Memory 之上加入初始观测、相对位姿或 S2R Stage-A
adapter。其余 Memory 参数保持 reference 配置。

`EMBEDDING_MODE` 每次选择一种模式：

| 模式 | 配置 |
| --- | --- |
| 无 embedding enhancement | `EMBEDDING_MODE=none` |
| Additive pose embedding | `EMBEDDING_MODE=pose` |
| FiLM pose embedding | `EMBEDDING_MODE=posefilm` |
| S2R Stage-A adapter | `EMBEDDING_MODE=uav` |

### 4.1 Initial observation

```bash
SYSTEM_PROMPT_SETTING=initial \
EMBEDDING_MODE=none \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

`SYSTEM_PROMPT_SETTING=initial` 将 Episode 第一帧作为未压缩图像放入 system prompt；
`vanilla` 不添加该观测。对应模型名包含 `initial-noembed`。两种环境均已验证 initial
observation 的训练与跨窗口评测。

### 4.2 Relative pose

```bash
SYSTEM_PROMPT_SETTING=vanilla \
EMBEDDING_MODE=posefilm \
POSE_NORM_SCALE=100 \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

Relative pose 使用当前实现的 `posefilm`：每张图像使用
`[delta_forward, delta_right, sin(delta_heading), cos(delta_heading)]` 表示相对位姿，
经 MLP 后通过 FiLM 注入视觉 token。`POSE_NORM_SCALE` 用于对前向与横向位移执行
`tanh(position / scale)` 归一化；航向的正弦和余弦分量保持不变。对应模型名以
`posefilm` 结尾。

Additive pose embedding 使用同一组相对位姿输入，并将位姿特征直接加到视觉 token：

```bash
SYSTEM_PROMPT_SETTING=vanilla \
EMBEDDING_MODE=pose \
POSE_NORM_SCALE=100 \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

FiLM 与 Additive pose embedding 均已在 SatNav 和 Habitat 验证。

### 4.3 S2R Stage-A adapter

加载 [S2R Stage-A](S2R_STAGE_A.md) adapter：

```bash
EMBEDDING_MODE=uav \
UAV_ADAPTER_PATH=/path/to/stage-a-checkpoint.pt \
UAV_ADAPTER_TYPE=transformer_v1 \
UAV_ADAPTER_APPLY_SCOPE=all_images \
bash scripts/train/train_swiftvln_qwen_vl.sh
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `UAV_ADAPTER_PATH` | 空 | Stage-A adapter checkpoint 路径 |
| `UAV_ADAPTER_TYPE` | `transformer_v1` | Adapter 结构 |
| `UAV_ADAPTER_APPLY_SCOPE` | `all_images` | 将 adapter 应用于全部输入图像 |

S2R Stage-A adapter 仅支持 SatNav；Habitat 不支持，本次未测试。

## 5. Long-term Memory compression

### 5.1 Map memory

Map memory 使用 SatNav 已探索区域的全局图和局部图替代历史 RGB：

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

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `MAP_GLOBAL_SIDE_M` | `1000` | 全局图覆盖的正方形边长，单位为米 |
| `MAP_LOCAL_SIDE_M` | `400` | 以当前位置为中心的局部图边长，单位为米 |
| `MAP_RENDER_PX` | `448` | 全局图和局部图的输出边长，单位为像素 |
| `MAP_MASK_METHOD` | `dilate20` | 已探索区域 mask；支持 `strict` 或 `dilate<N>` |
| `MAP_CACHE_DIR` | `auto` | 渲染缓存目录；`auto` 使用数据集目录下的 `map_cache/` |
| `COMPRESS_STRIDE` | `2` | 全局图和局部图的视觉 token 压缩 stride |

Map memory 仅支持 SatNav。`MAP_LOCAL_SIDE_M` 不得大于 `MAP_GLOBAL_SIDE_M`，并且该配置
不与随机历史采样、GridToMe 或 embedding enhancement 组合使用。

### 5.2 Global token clustering（GTC）

GTC 将全部历史帧的视觉 token 视为一个集合，通过 Soft K-Means 压缩为固定数量的 token：

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

GTC 已在 SatNav 和 Habitat 验证。

### 5.3 Segment token clustering（STC）

STC 在仓库中实现为 Segment-GTC，即 `HISTORY_PROCESSOR_TYPE=segment_gtc`。
当前实现将历史帧固定划分为 8 个时间段，在每个时间段内执行 GTC，再按时间顺序拼接结果：

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

GTC 与 Segment-GTC 共用以下参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `GTC_OUTPUT_TOKENS` | `512` | 聚类后的目标 token 总数；输入不足时不扩充 token |
| `GTC_TEMPERATURE` | `0.1` | Soft K-Means assignment 温度；数值越小，分配越集中 |
| `GTC_NUM_ITERATIONS` | `1` | Soft K-Means 的更新次数 |

这两种 processor 不使用 `NUM_HISTORY` 控制帧数。数据管线按照训练时的动作预测间隔读取
历史帧，再执行 token clustering。GTC 直接聚类全部历史 token；Segment-GTC 通过分段
保留粗粒度时间顺序。模型名分别使用 `gtc-k512` 和 `sgtc-k512`。Segment-GTC 已在
SatNav 和 Habitat 验证。

## 6. 配置与模型对应关系

| 配置 | 关键参数 | 模型名中的配置片段 |
| --- | --- | --- |
| SwiftVLN reference | `per_frame`、`NUM_HISTORY=8` | `pf-h8-pool-s2` |
| Short-term only | `NUM_HISTORY=0` | `pf-h0-nomem-pool-s2` |
| Random sampling | `USE_RANDOM=true` | `pf-h8-random` |
| Temporal-biased sampling | `LOG_BASE=2.0` | `pf-h8-b2.0` |
| Initial observation | `SYSTEM_PROMPT_SETTING=initial` | `initial-noembed` |
| Relative pose | `EMBEDDING_MODE=posefilm` | `posefilm` |
| Additive pose | `EMBEDDING_MODE=pose` | `pose` |
| S2R Stage-A adapter | `EMBEDDING_MODE=uav` | `uav` |
| Map memory | `MEMORY_METHOD=map` | `map-g1000-l400-r448-d20-s2` |
| GTC | `HISTORY_PROCESSOR_TYPE=gtc` | `gtc-k512` |
| STC / Segment-GTC | `HISTORY_PROCESSOR_TYPE=segment_gtc` | `sgtc-k512` |

已发布模型及完整名称见[模型与 Checkpoint](../getting-started/CHECKPOINTS.md)。训练、输出目录和
恢复训练见 [SwiftVLN 训练](README.md)。
