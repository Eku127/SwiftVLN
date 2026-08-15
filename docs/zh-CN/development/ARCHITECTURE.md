# 代码架构

SwiftVLN 将训练、在线评测、模型扩展和模拟器适配划分为独立模块。训练读取离线
trajectory，评测通过 Backend 连接 SatNav 或 Habitat；两条链路共享模型、Memory
处理、embedding enhancement 与实验配置。

## 1. 仓库结构

```text
SwiftVLN/
├── src/swiftvln/
│   ├── experiment.py          # 实验约束与模型名称编解码
│   ├── training/sft/          # 训练参数、Dataset 和 ms-swift trainer
│   ├── modeling/              # 模型、Template、Memory 与 embedding
│   ├── evaluation/            # 在线评测、推理状态和结果持久化
│   ├── backends/              # SatNav / Habitat 适配层
│   ├── data/                  # Habitat 轨迹生成与数据校验 CLI
│   ├── configs/               # 随包发布的评测任务配置
│   └── utils/                 # 分布式、图像与视频工具
├── scripts/
│   ├── train/                 # 训练启动脚本
│   ├── eval/                  # 评测启动脚本
│   ├── queue/                 # 文件队列
│   └── lib/                   # Shell 公共函数
├── tools/s2r/                 # SatDronePair 与 S2R Stage-A 离线工具
├── baseline/                  # 外部 VLN baseline 的 SatNav 适配
├── environments/             # swiftvln-train / swiftvln-eval 环境定义
├── tests/                     # 结构、配置和运行 contract
├── third_party/               # 固定版本的外部源码
└── docs/                      # 用户与开发文档
```

`src/swiftvln` 是可安装 Python package。Shell 启动器、离线 S2R 工具、baseline 和文档
属于仓库级资产，不进入核心 Python package。

## 2. 配置与模型名称

[`experiment.py`](../../../src/swiftvln/experiment.py) 定义
`SwiftVLNExperimentSpec`，集中维护以下配置：

- 环境与模型族；
- 轨迹窗口、future steps 和 overlap；
- Memory method 与 history processor；
- system prompt 与 embedding enhancement；
- Map memory 的环境和参数约束；
- 模型名称的生成、解析与 shell/JSON 输出。

训练脚本在启动前调用 `build-name` 生成模型名称；`eval_by_name.sh` 使用同一模块解析名称，
恢复评测参数：

```text
training variables ──> SwiftVLNExperimentSpec ──> model name
                                                     │
                                                     ▼
evaluation variables <── shell assignments <── parse-name
```

训练参数和评测参数分别定义在：

| 配置入口 | 位置 |
| --- | --- |
| 共享实验约束 | `src/swiftvln/experiment.py` |
| 训练参数 | `src/swiftvln/training/sft/arguments.py` |
| 评测参数 | `src/swiftvln/evaluation/arguments.py` |
| 训练启动默认值 | `scripts/train/train_swiftvln_qwen_vl.sh` |
| 评测启动默认值 | `scripts/eval/eval_swiftvln_qwen_vl_distributed.sh` |

## 3. 训练链路

```text
train_swiftvln_qwen_vl.sh
        │
        ├─ 加载本机路径、选择 GPU、生成模型名称
        ▼
SwiftVLNSft + SwiftVLNTrainArguments
        │
        ├─ SwiftVLNDataset：轨迹窗口、Memory、消息与 loss mask
        ├─ SwiftVLNTemplate：视觉编码、token 压缩与 embedding 注入
        └─ registered SwiftVLN model：Qwen-VL + embedding enhancement
        ▼
ms-swift Full SFT
        ▼
checkpoint + train metadata
```

### 3.1 Dataset

[`training/sft/dataset.py`](../../../src/swiftvln/training/sft/dataset.py) 读取一个或多个
包含 `annotations.json` 的 trajectory 目录，并完成：

- 按 `NUM_FRAMES` 和 `NUM_OVERLAP` 划分轨迹窗口；
- 按 `NUM_FUTURE_STEPS` 组织多轮动作预测；
- 为重叠窗口中的上下文 assistant turn 设置 loss mask；
- 采样历史 RGB，或生成 SatNav Map memory；
- 构造初始观测和相对位姿；
- 输出 ms-swift 接受的多模态对话样本。

Dataset 输出的核心字段为：

| 字段 | 内容 |
| --- | --- |
| `messages` | System、user image turn 和 assistant action turn |
| `images` | History、initial observation、current observation，按该顺序排列 |
| `num_history_images` | Template 需要处理的历史图像数量 |
| `num_initial_images` | System prompt 中的初始观测数量 |
| `memory_method` | `history` 或 `map` |
| `frame_poses` | 启用 pose enhancement 时与图像顺序对齐的位姿 |

### 3.2 Template

[`modeling/template.py`](../../../src/swiftvln/modeling/template.py) 为 Qwen2.5-VL 和
Qwen3-VL 注册 SwiftVLN Template。Template 使用两个特殊 token：

| Token | 用途 |
| --- | --- |
| `<history_memory>` | 压缩后的历史 Memory token block |
| `<current_image>` | Initial/current observation 的视觉 token block |

编码阶段先计算占位 token 数量，视觉塔编码完成后再调用 history processor，并将真实
embedding 注入对应位置。训练侧的图像顺序、占位 token 数量和视觉 embedding 数量必须
保持一致。

### 3.3 Trainer 与 Checkpoint

[`training/sft/trainer.py`](../../../src/swiftvln/training/sft/trainer.py) 扩展 ms-swift
`SwiftSft`：检测 VLN trajectory、创建 `SwiftVLNDataset`、配置 Template 的 history
processor，并将 embedding enhancement 挂载到模型。模型参数、optimizer、scheduler
和 enhancement 参数由 ms-swift 统一训练和保存。

## 4. 评测链路

```text
eval_by_name.sh
      │ 解析模型名、定位 checkpoint、选择 split
      ▼
evaluation.arguments
      ▼
SwiftVLNEvaluationRunner
      ├─ 加载模型并初始化 rank
      ├─ 创建 SwiftVLNEvaluator
      ├─ 按 scene 排序并 round-robin 分配 Episode
      └─ 追加结果、恢复进度、汇总 rank
             │
             ▼
EnvironmentEpisodeLoop
      ├─ SwiftVLNInferenceSession
      └─ EvaluationBackend / EnvWrapper
             │
             ▼
ResultRecorder
```

评测不会创建训练 Template。`SwiftVLNInferenceSession` 直接构造 prompt token、编码视觉
特征并注入 embedding，以保持在线窗口状态和 simulator step 一致。

### 4.1 Runner

[`evaluation/runner.py`](../../../src/swiftvln/evaluation/runner.py) 负责模型加载、分布式
初始化、Episode 分配、恢复和结果汇总。Episode 先按 scene 稳定排序，再对全局序列执行
round-robin 分片。

### 4.2 Episode loop

[`evaluation/episode_loop.py`](../../../src/swiftvln/evaluation/episode_loop.py) 只处理
`reset → predict → step → metrics` 状态机。每个 Episode 使用一个独立的
`SwiftVLNInferenceSession` 状态，并通过 Backend 读取 observation、执行动作和保存视频。

### 4.3 Inference session

`evaluation/inference/` 按职责拆分：

| 模块 | 职责 |
| --- | --- |
| `session.py` | 组合推理参数、processor、Memory builder 和生成入口 |
| `prompt.py` | 构造 system/user/assistant token 与最终 `inputs_embeds` |
| `encoding.py` | 编码 initial/current/history 图像并构建 Memory cache |
| `window.py` | 保存 turn、overlap context、pose 和滑动窗口状态 |

## 5. Modeling

### 5.1 模型注册

[`modeling/registry.py`](../../../src/swiftvln/modeling/registry.py) 通过
`register_swiftvln_models()` 显式注册：

- Transformers config 与 model class；
- ms-swift model metadata 与 loader；
- Qwen2.5-VL / Qwen3-VL Template；
- SwiftVLN 特殊视觉 token。

导入 `swiftvln` 不会触发模型注册。训练和评测入口在解析或加载模型前显式调用注册函数。

### 5.2 History processor

`modeling/history/` 提供统一接口：

```text
get_output_token_count(...)  # 编码前计算占位 token 数量
process(...)                 # 处理视觉 embedding
```

当前实现包括 per-frame pooling/GridToMe、GTC 和 Segment-GTC。训练 Template 与在线
Inference session 使用同一个 factory 和同一组 processor 参数。

### 5.3 Memory

`modeling/memory/` 负责 SatNav Map memory：解析 Episode/trajectory 元数据，根据动作重建
路径，渲染 global/local explored map，并管理磁盘缓存。History RGB 与 Map memory 最终
都转换为视觉 token，交给 Template 或 Inference session 注入模型。

### 5.4 Embedding enhancement

`modeling/embeddings/` 在视觉塔输出后、History 压缩前处理单张图像 embedding。当前
`EMBEDDING_MODE` 为互斥单选：`none`、`pose`、`posefilm` 或 `uav`。

`EmbeddingEnhancementPipeline` 使用 `nn.ModuleDict` 保存模块，因此 enhancement 参数会
进入 `model.parameters()` 和 checkpoint `state_dict`。模型加载器会在评测时从
safetensors 恢复 `embed_enhance.*` 权重。

## 6. Backend 边界

核心训练、推理和 Episode loop 不直接导入 SatNav 或 Habitat。Backend 由三层组成：

| 层 | 位置 | 职责 |
| --- | --- | --- |
| 静态语义 | `backends/specs.py` | 动作符号、前进距离、转向角、prompt 和能力标记 |
| Backend | `backends/<env>/backend.py` | 加载配置、创建 simulator、解析动作和视频 hook |
| EnvWrapper | `backends/<env>/wrapper.py` | 统一 reset、step、observation、metrics 和 Episode 访问 |

`backends/factory.py` 在选定环境后才导入对应 Backend。模拟器依赖因此只在实际使用该
环境时加载。

`EnvWrapper` 的统一接口包括：

- `reset()`、`step()` 和 `close()`；
- `get_rgb()`、`get_instruction()` 和 `get_metrics()`；
- `episodes`、`episode_over`、`max_steps` 和 `env_type`。

## 7. 结果持久化与分布式

训练由 `torchrun + DeepSpeed` 管理模型分片、optimizer 和 checkpoint，模型名称同时作为
输出目录名。启用 SwanLab 时，训练脚本额外写入 `train_metadata.json`，记录 project、
experiment name 和 run URL。

评测使用 [`evaluation/results.py`](../../../src/swiftvln/evaluation/results.py) 持久化：

| 产物 | 作用 |
| --- | --- |
| `result.jsonl` | 每完成一个 Episode 立即追加，作为恢复日志 |
| `all_results.jsonl` | 按 `scene_id::episode_id` 去重后的完整结果 |
| `evaluation_summary.json` | 指标、模型配置、split 与 GPU 信息 |
| `timing_summary.json` | 各推理阶段的耗时统计 |
| `.dist_sync/rank_<n>.done.json` | 各 rank 的完成标记 |

Rank 0 等待全部完成标记后，从 `result.jsonl` 离线生成最终结果和汇总。再次使用同一未完成
输出目录时，已经存在的 `scene_id::episode_id` 会被跳过。

## 8. S2R 依赖方向

S2R Stage-A 的数据转换、Manifest、训练和 retrieval 评测位于 `tools/s2r/`，仅从仓库
checkout 运行。SwiftVLN package 中只保留运行模型所需的 adapter 结构和 checkpoint
loader：

```text
tools/s2r/ ──> src/swiftvln/modeling/embeddings/s2r_adapter.py
                         ▲
                         │
src/swiftvln/modeling/embeddings/uav_adapter.py

src/swiftvln/  -X->  tools/s2r/
```

核心 package 不导入 `tools/s2r`，离线数据依赖不会进入 SwiftVLN 训练和评测运行时。
