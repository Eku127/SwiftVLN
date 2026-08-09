# SwiftVLN 架构与配置概览

本文档描述当前 SwiftVLN 主线的稳定边界。默认值和跨字段约束以
`swiftvln/experiment.py` 为唯一事实源；训练参数定义见
`training/sft/arguments.py`，
实际启动约定见 train/eval shell。历史实验指标不在本文维护，统一查阅
各次评测生成的结果 JSON。

## 支持范围

- 模型族：Qwen2.5-VL、Qwen3-VL。
- 环境：SatNav 和 Habitat；两者都有独立 wrapper 与配置。
- SatNav 默认数据：`SatNav-v0.1`。
- 记忆：历史 RGB、SatNav explored map、no-memory 特例。
- 历史处理：per-frame、GTC、Segment-GTC。
- embedding enhancement：`none|pose|posefilm|uav`，严格四选一。
- 评测：单机多卡、逐 episode 持久化、断点续跑、去重、视频可视化。

## 主线结构

```text
训练
train shell
  -> SwiftVLNTrainArguments
  -> SwiftVLNDataset（轨迹窗口、多轮消息、loss mask）
  -> SwiftVLNTemplate（训练侧视觉 token/embedding 注入）
  -> SwiftVLN model loader
  -> ms-swift SFT

评测
eval_by_name.sh -> ExperimentSpec 解析模型名
  -> evaluation/cli.py（CLI）
  -> SwiftVLNEvaluationRunner（模型、rank、resume、汇总）
  -> SwiftVLNEvaluator（环境、推理、episode loop 的组合根）
  -> SwiftVLNInferenceSession（窗口、缓存、prompt、生成）
  -> backend factory / EnvWrapper（SatNav 或 Habitat）
  -> ResultRecorder（JSONL 与 evaluation_summary.json）
```

评测不再构造训练 template。推理 session 直接构造 token 与视觉 embedding；
`modeling/template.py` 只属于训练链路。

## 当前默认配置

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `NUM_FRAMES` | `32` | 一个轨迹窗口的动作数 |
| `NUM_FUTURE_STEPS` | `4` | 每轮生成的动作数 |
| `NUM_OVERLAP` | `0` | 相邻窗口重叠动作数；默认不重叠 |
| `MEMORY_METHOD` | `history` | 使用历史 RGB 记忆 |
| `HISTORY_PROCESSOR_TYPE` | `per_frame` | 逐帧历史压缩 |
| `NUM_HISTORY` | `8` | per-frame 采样帧数 |
| `COMPRESS_STRIDE` | `2` | pooling/ToMe 空间压缩步长 |
| `LOG_BASE` | `1.0` | 均匀历史采样 |
| `USE_RANDOM` / `USE_TOME` | `false` / `false` | 不随机采样，使用 pooling |
| `SYSTEM_PROMPT_SETTING` | `vanilla` | 不额外注入首帧 |
| `EMBEDDING_MODE` | `none` | 不启用 embedding enhancement |

训练 shell 的优化默认值为 1 epoch、per-rank batch size 8、learning rate `2e-5`、
DeepSpeed ZeRO-2。模型名中编码的是有效 batch size，而不是单 rank batch size。

## 轨迹窗口与多轮训练

`training/sft/dataset.py` 把轨迹切成窗口，并按 `NUM_FUTURE_STEPS` 组织 user/assistant
多轮对话。窗口步长为：

```text
stride = NUM_FRAMES - NUM_OVERLAP
```

- `NUM_OVERLAP=0` 时保持完整尾窗覆盖。
- `NUM_OVERLAP>0` 时不回挪尾窗；非首窗中重复的 overlap action loss 被 mask。
- overlap 必须小于窗口，并按 `NUM_FUTURE_STEPS` 对齐。
- `NUM_HISTORY=0` 只在 per-frame 下表示 no-memory，不存在第二个 memory 布尔开关。

训练侧使用两个特殊 token：`<history_memory>` 表示处理后的历史块，
`<current_image>` 表示当前观察。dataset 生成消息与图像/pose 元数据，template
负责把占位符扩展并将视觉特征注入语言模型输入。

## History processor 与地图记忆

| 模式 | 行为 | 主要参数 |
| --- | --- | --- |
| `per_frame` | 采样若干历史帧，逐帧 pooling 或 GridToMe 后按时间拼接 | `NUM_HISTORY`、`LOG_BASE`、`COMPRESS_STRIDE`、`USE_RANDOM`、`USE_TOME` |
| `gtc` | 将跨帧视觉 token 用 soft K-means 聚合为固定数量 | `GTC_OUTPUT_TOKENS`、temperature、iterations |
| `segment_gtc` | 分段执行 GTC，再按时间顺序拼接 | 同 GTC |

`MEMORY_METHOD=map` 用 SatNav explored-map 图像替换历史 RGB。它只支持 SatNav +
per-frame + pooling，并要求 `EMBEDDING_MODE=none`。默认 map cache 从数据根推导为
`{dataset_root}/map_cache`，也可通过 `SWIFTVLN_MAP_CACHE_DIR` 覆盖或关闭。

## Embedding enhancement：严格四选一

`EMBEDDING_MODE` 是训练、评测、模型名、loader 与运行时的唯一开关：

| mode | 注入方式 | 是否需要 pose |
| --- | --- | --- |
| `none` | 不挂载 enhancement module | 否 |
| `pose` | PoseEmbedding additive 融合 | 是 |
| `posefilm` | PoseEmbedding FiLM 融合 | 是 |
| `uav` | 加载 Stage-A UAV adapter | 否 |

任一时刻 pipeline 最多只有一个 module。pose 与 UAV 不能叠加；旧
`USE_POSE_EMBED`、`USE_UAV_ADAPTER`、`POSE_FUSION_METHOD` 配置会被显式拒绝。
UAV mode 当前只支持 `UAV_ADAPTER_APPLY_SCOPE=all_images`。

### Pose 表示与融合

训练和推理都通过 action 积分构造同一个 ego-frame pose：

```text
[delta_forward, delta_right, sin(delta_heading), cos(delta_heading)]
```

- SatNav：forward 10m，turn 15°。
- Habitat：forward 0.25m，turn 30°。
- 位置分量使用 `tanh(value / POSE_NORM_SCALE)`；默认 scale 为 `100.0`。
- 朝向的 sin/cos 已在 `[-1, 1]`，不再缩放。

Additive mode：

```text
embed = embed + beta * MLP(pose)
```

FiLM mode：

```text
[gamma, bias] = MLP(pose)
embed = embed * (1 + beta * gamma) + beta * bias
```

Pose MLP 的最后一层零初始化，因此训练开始时与 `none` 的视觉特征一致。
模块权重随训练 checkpoint 保存，并在评测加载时恢复。

相关实现：

- `modeling/embeddings/pose_utils.py`：action 到 pose。
- `modeling/embeddings/pose_embed.py`：additive/FiLM 模块。
- `modeling/embeddings/s2r_adapter.py`：运行时 Stage-A adapter 架构。
- `modeling/embeddings/uav_adapter.py`：Stage-A checkpoint loader。
- `modeling/embeddings/pipeline.py`、`runtime.py`：互斥 pipeline 与挂载。

## 推理与评测

`SwiftVLNInferenceSession` 管理当前窗口、历史特征、overlap context、map memory、
pose history、动作 history 和视觉编码缓存。`EnvironmentEpisodeLoop` 只负责
reset/predict/step 状态机，以及按需收集可视化帧；环境差异由 wrapper 隔离。

`SwiftVLNEvaluationRunner` 负责：

- 按 scene 稳定排序后对 rank 做全局 round-robin；
- 以 `scene_id::episode_id` 为键恢复和去重；
- 每个 rank 逐 episode append `result.jsonl`；
- 等待 rank marker，并由 rank 0 离线生成 `evaluation_summary.json`；
- 保留 SatNav 与 Habitat 两条环境路径。

`SAVE_VIDEO=true` 时，Habitat 输出 observation/top-down 可视化，SatNav 输出 RGB 与
top-down 组合视频；`VIDEO_COMPRESSION=true` 控制压缩。视频能力与指标汇总互不依赖。
评测只持久化标准导航指标及真实运行异常的 `error` 字段，不再执行事后失败分类。

## S2R

Stage-A 的 manifest、teacher/adapter 训练、checkpoint、retrieval eval 与原始数据
转换均位于仓库级 `tools/s2r/`，不进入 SwiftVLN wheel。Stage-B 通过
`EMBEDDING_MODE=uav` 将 Stage-A adapter 接入 SwiftVLN；安装包只保留
`modeling/embeddings/s2r_adapter.py` 和 checkpoint loader。离线工具依赖核心 adapter
定义，核心包不反向 import `tools/`。

## 入口与验证

```bash
# 训练
bash scripts/train/train_swiftvln_qwen_vl.sh

# 按规范模型名评测；SatNav 默认跑 val_seen + val_unseen
bash scripts/eval/eval_by_name.sh <model_name>

```

核心入口：

- `swiftvln/experiment.py`：约束与模型名 codec。
- `modeling/`：模型、template、embedding、history 与 memory 能力；注册由
  `register_swiftvln_models()` 显式触发。
- `training/sft/`：SFT 参数、dataset 与 trainer。
- `evaluation/`：评测参数、分布式 runner、episode loop、inference 与结果持久化。
- `backends/`：SatNav/Habitat wrapper 与 simulator 专有扩展。
- `utils/`：小型、依赖中立的运行时工具。
- `scripts/train/`、`scripts/eval/`：单次训练与评测 shell 入口；
- `scripts/queue/`：仅在仓库 checkout 中运行的训练与评测队列。

涉及模型名、默认数据、入口路径或 train/eval 主流程的变更，应同步更新公开配置与
本文，并进行相应的本地 contract/smoke 验证。
