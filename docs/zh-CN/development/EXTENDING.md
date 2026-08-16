# 扩展 SwiftVLN

简体中文 | [English](../../en-US/development/EXTENDING.md)

SwiftVLN 的训练与评测共享实验配置、模型名称、Memory 和 embedding 实现。新增能力时，
需要同时保持训练参数、在线推理、checkpoint 和模型名称的一致性。各模块关系见
[代码架构](ARCHITECTURE.md)。

## 1. 扩展涉及的公共接口

| 接口 | 事实源 | 需要保持的约束 |
| --- | --- | --- |
| 实验配置 | `src/swiftvln/experiment.py` | 参数组合可以验证，并能够通过模型名称往返解析 |
| 训练参数 | `training/sft/arguments.py` | Dataset、Template 和 model loader 接收相同配置 |
| 评测参数 | `evaluation/arguments.py` | 与训练配置一致，并写入评测 summary |
| Shell 入口 | `scripts/train/`、`scripts/eval/` | 环境变量正确传递给 Python CLI |
| Checkpoint | `modeling/model.py` | 新增参数和权重能够保存、恢复并继续评测 |
| 测试 | `tests/` | 配置、结构、数据流和 checkpoint contract 保持稳定 |

当新增能力会出现在模型名称中时，先确定唯一的名称片段，再修改训练和评测入口。

## 2. 新增环境 Backend

### 2.1 定义环境语义

在 `src/swiftvln/backends/specs.py` 中新增 `EnvironmentSpec`，并加入
`_ENVIRONMENT_SPECS`。Spec 负责与模拟器无关的静态语义：

| 字段 | 内容 |
| --- | --- |
| `name` | 环境标识，用于训练参数、评测参数和模型名称 |
| `forward_step_m` | Forward action 的距离 |
| `turn_angle_deg` | Left/right action 的转向角 |
| `prompt_template` | 导航任务 system prompt |
| `config_argument` | 评测参数中对应的配置文件字段名 |
| `supports_map_memory` | 是否支持 Map memory |
| `reports_trajectory_types` | 是否输出按 trajectory type 分组的指标 |
| `action_symbols` | Action ID 与输出符号的顺序 |

训练 Dataset 通过 `get_environment_spec()` 获取 prompt、动作符号和位姿积分参数，不在
Dataset 中增加环境分支。

### 2.2 实现 Backend 与 Wrapper

创建 `src/swiftvln/backends/<env>/`：

| 文件 | 职责 |
| --- | --- |
| `config.py` | 加载任务配置并应用 split、数据路径和 GPU override |
| `wrapper.py` | 将模拟器适配为 `EnvWrapper` |
| `backend.py` | 组合配置、模拟器、action parser 和视频 hook |
| `video.py` | 可选的视频帧采集与保存 |
| `__init__.py` | 保持轻量导出 |

`EnvWrapper` 必须实现：

- `reset(episode)` 与 `step(action)`；
- `get_rgb()`、`get_instruction()` 和 `get_metrics()`；
- `episodes`、`episode_over`、`max_steps` 和 `env_type`；
- `close()`。

模拟器 import 放在 `Backend.create_wrapper()` 或 `config.py` 的实际加载函数中。
`wrapper.py` 使用 `TYPE_CHECKING` 标注模拟器类型，使核心 package 在未安装该模拟器时仍可
导入。

### 2.3 注册训练与评测入口

完成以下改动：

1. 在 `backends/factory.py` 的 `_BACKEND_CLASSES` 注册 Backend；
2. 在 `evaluation/arguments.py` 增加环境 choice 和配置路径参数；
3. 在 `experiment.py` 更新 `ENV_TYPES`、模型名称正则和环境约束；
4. 在训练脚本增加默认 trajectory 路径；
5. 在评测脚本增加默认任务配置、split 和 Python 参数；
6. 在 `src/swiftvln/configs/<env>/` 添加随包发布的 YAML；
7. 在 `local.env.example` 增加本机数据与场景路径。

新环境的离线 trajectory 需要满足第 6 节的数据 contract。在线 Episode 的 simulator
字段差异由 Wrapper 消化，`EnvironmentEpisodeLoop` 和 `SwiftVLNInferenceSession` 不增加
环境判断。

## 3. 新增 History Processor

### 3.1 实现处理器

在 `src/swiftvln/modeling/history/` 中新增 `HistoryProcessor` 子类：

| 方法 | Contract |
| --- | --- |
| `get_output_token_count(num_frames, frame_infos)` | 在视觉编码前返回占位 token 数量 |
| `process(frame_embeds_list, frame_grid_thws)` | 返回 `[num_tokens, hidden_size]` |
| `name` | 用于日志的稳定名称 |

`get_output_token_count()` 的结果必须与 `process()` 的第一维完全一致。Processor 以单个
样本的历史帧为输入，不能在 batch 内混合不同样本的 token。

### 3.2 注册与传参

1. 在 `modeling/history/__init__.py` 注册类型和 factory 构造参数；
2. 在 `experiment.py` 的 `HISTORY_PROCESSORS` 中增加公共名称；
3. 为模型名称增加稳定的编码和解析规则；
4. 在训练、评测 arguments 中增加超参数；
5. 在训练和评测 shell 中设置默认值并传递参数；
6. 在 `SwiftVLNSft._prepare_template()` 中创建相同配置的 processor；
7. 在 `SwiftVLNInferenceSession` 中创建在线评测使用的 processor；
8. 在 `build_summary_extras()` 中记录影响推理结果的参数。

### 3.3 对齐训练与在线推理

训练侧的历史帧选择位于 `SwiftVLNDataset._sample_history_frames()`；在线侧的 cache 构建
位于 `evaluation/inference/encoding.py`。如果新 processor 使用不同的帧采样或 cache
策略，需要同时增加两侧路由。

Template 会先调用 `get_output_token_count()` 创建 `<history_memory>` 占位，再调用
`process()` 生成 embedding。Inference session 则直接将 `process()` 输出写入
`history_cache`。两条路径应使用相同的帧顺序、processor 参数和输出 token 数量。

## 4. 新增 Embedding Enhancement

Embedding enhancement 位于视觉塔之后、History 压缩之前。当前公开配置为互斥单选，
一个模型只启用一种 enhancement。

### 4.1 实现模块

在 `src/swiftvln/modeling/embeddings/` 中新增 `BaseEmbeddingEnhancement` 子类，实现：

| 接口 | Contract |
| --- | --- |
| `forward(embed, H, W, **kwargs)` | 输入和输出均为 `[num_tokens, hidden_size]` |
| `name` | 稳定的日志名称 |

模块通过 `EmbeddingEnhancementPipeline.enhancements` 中的 `nn.ModuleDict` 挂载。其参数会
自动进入 optimizer 和 checkpoint `state_dict`。

### 4.2 注册公共模式

需要同步修改：

1. `experiment.py`：`EMBEDDING_MODES`、名称解析、互斥规则和辅助判断函数；
2. `modeling/embeddings/__init__.py`：在 `create_embedding_pipeline()` 中创建模块；
3. `modeling/embeddings/runtime.py`：声明目标模块、重建条件和兼容 alias；
4. `training/sft/arguments.py` 与 `evaluation/arguments.py`：增加参数；
5. `modeling/model.py`：从 loader kwargs 读取参数，并写入模型 config；
6. 训练和评测 shell：校验模式、传递参数并生成名称；
7. `build_summary_extras()`：记录推理配置。

### 4.3 增加图像元数据

训练 Template 和在线 `VisualEncodingMixin` 都以
`model.embed_enhance(embed, H, W, **kwargs)` 调用模块。如果 enhancement 需要 pose 之外的
逐图像元数据，还需要更新：

- `SwiftVLNDataset` 的图像元数据生成和顺序；
- `SwiftVLNTemplateMixin._encode()` 与 multimodal collator；
- `SwiftVLNInferenceSession` 的状态记录；
- `VisualEncodingMixin._encode_frame()` 和 `_encode_batch_frames()`。

元数据顺序必须与 `images`、视觉塔输出的 per-image token boundary 一致。保存 checkpoint
后，从纯 checkpoint 目录加载模型时应恢复相同的 pipeline 类型、配置和权重。

## 5. 新增模型族

### 5.1 Transformers 与 ms-swift 注册

在 `modeling/model.py` 中增加：

- 继承上游 Transformers config 的 SwiftVLN config；
- 继承上游 conditional-generation model 的 SwiftVLN model class；
- 能够添加特殊 token、挂载 enhancement 并恢复权重的 loader；
- 新模型需要的 `inputs_embeds` 或视觉输出兼容逻辑。

随后在 `modeling/registry.py` 中注册 Transformers class、ms-swift `ModelMeta`、默认模型
ID、model architecture、Template 和 loader。`register_swiftvln_models()` 必须保持可重复
调用。

### 5.2 Template

在 `modeling/template.py` 中基于对应的 ms-swift Template 创建 SwiftVLN Template，并复用
`SwiftVLNTemplateMixin`。新模型族需要确认：

- image processor 返回 `image_grid_thw`；
- 视觉输出可以按单张图像拆分；
- `<history_memory>` 与 `<current_image>` 的 token ID 正确；
- `inputs_embeds`、attention mask、position IDs 和 generation API 兼容；
- History processor 的占位 token 数量与实际输出一致。

### 5.3 配置与名称

同步更新：

1. `experiment.py` 的 `MODEL_FAMILIES`、模型名称前缀和解析规则；
2. 训练与评测 arguments 中 `model_type → model_family` 的显式映射；
3. 训练和评测 shell 的 `MODEL_FAMILY` 分支、默认模型路径和运行参数；
4. `local.env.example` 与模型下载文档；
5. 模型名称 contract 中的合法与非法样例。

## 6. 修改数据 Contract

### 6.1 离线训练数据

每个 trajectory 目录包含 `annotations.json` 和对应的 RGB 目录。核心 annotation 字段为：

```json
{
  "video": "relative/trajectory/path",
  "instructions": ["navigation instruction"],
  "actions": [-1, 1, 2, 1, 0]
}
```

`video/rgb/` 存放按时间排序的观测帧。`actions` 中 `-1` 表示初始状态，`0/1/2/3`
对应 `EnvironmentSpec.action_symbols` 中的 STOP、forward、left 和 right。

修改 annotation schema 时，需要同步：

- trajectory 生成器与 validation CLI；
- `SwiftVLNDataset` 的读取、窗口索引和输出字段；
- Map metadata resolver 或逐图像元数据生成；
- SatNav/Habitat 训练数据文档。

### 6.2 Dataset 与 Template 边界

`SwiftVLNDataset` 输出 `messages`、`images` 和图像计数元数据。图像顺序固定为：

```text
history images → initial observation → current observations
```

新增字段必须能够通过 ms-swift `StdTemplateInputs.extra_kwargs` 到达 Template。Template
collator 需要按样本保留边界，不能把不同样本的图像计数或元数据展平后失去对应关系。

### 6.3 在线 Episode 与结果

模拟器 Episode 的字段差异由 `EnvWrapper` 适配。Runner 对外只依赖 `episode_id`、
`scene_id`、instruction 和可选的 `trajectory_type`。

修改评测结果字段时，同时更新：

- `SwiftVLNEvaluationRunner.evaluate_episode()`；
- `ResultRecorder` 与 `evaluation/reporting.py`；
- resume/dedup contract；
- [SwiftVLN 评测](../evaluation/README.md)中的输出说明。

## 7. 验证改动

在 `swiftvln-train` 环境运行与改动对应的 contract：

| 改动 | 测试入口 |
| --- | --- |
| 实验配置与模型名称 | `tests.test_experiment_contract`、`tests.test_model_name_contract` |
| Backend | `tests.test_backend_contracts`、`tests.test_evaluation_structure_contract` |
| History processor | `tests.test_history_contracts`、`tests.test_phase6_behavior_contracts` |
| Embedding enhancement | `tests.test_embedding_contract`、`tests.test_uav_adapter_strategy` |
| Dataset / Template | `tests.test_dataset_contracts`、`tests.test_training_structure_contract` |
| 结果与恢复 | `tests.test_eval_contracts` |
| CLI 与仓库边界 | `tests.test_cli_contracts`、`tests.test_repository_layout_contract` |

```bash
python -m unittest \
  tests.test_experiment_contract \
  tests.test_model_name_contract \
  tests.test_dataset_contracts \
  tests.test_training_structure_contract \
  tests.test_evaluation_structure_contract
```

修改 Shell 入口后检查语法：

```bash
bash -n scripts/train/train_swiftvln_qwen_vl.sh
bash -n scripts/eval/eval_by_name.sh
bash -n scripts/eval/eval_swiftvln_qwen_vl_distributed.sh
```

涉及模型前向或数据流的改动还需要完成一次真实 optimizer step、保存 checkpoint、从
checkpoint 重新加载，并运行单 Episode 在线评测。修改分布式代码时，再验证多 rank
Episode 分片、结果去重和中断恢复。

## 8. 同步文档

| 变更 | 同步页面 |
| --- | --- |
| 安装依赖、第三方仓库 | [安装](../getting-started/INSTALLATION.md) |
| 基础模型与 checkpoint | [模型与 Checkpoint](../getting-started/CHECKPOINTS.md) |
| 训练参数与模型名称 | [SwiftVLN 训练](../training/README.md) |
| Memory / embedding | [Memory 训练配置](../training/MEMORY.md) |
| 训练或评测数据 | `docs/zh-CN/data/` 对应页面 |
| 评测入口、输出与指标 | [SwiftVLN 评测](../evaluation/README.md) |
| 模块边界或依赖方向 | [代码架构](ARCHITECTURE.md) |
