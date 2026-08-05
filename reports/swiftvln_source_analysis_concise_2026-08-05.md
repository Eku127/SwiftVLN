# `src/swiftvln` 功能与精简审计

日期：2026-08-05
范围：当前工作树中的 `src/swiftvln/`；已处于删除状态的旧文件不计入本次快照。

## 结论

`src/swiftvln` 已经是一个围绕单一 SwiftVLN 主线组织的完整包，训练、评测、
环境、历史压缩、S2R、数据处理和队列都仍有现实入口。当前最值得做的不是删除
模型主线，而是重写失配的 Docker 初始化脚本、清理评测侧未使用的 template 链路、两个
无内部消费者的导出 facade，以及合并 S2R 数据生产中的重复工具。

- P0 已完成：17 服务器 Docker bootstrap、eval-time template 清理、两个
  `__init__.py` facade 收缩、UAV 测试迁移和模型文档收口均已落地并验证。
- P1 已完成：4 个 preview、2 个 recrop 与 SUES 专用 variant merge 均先迁移到
  共享实现并通过等价 contract，再删除旧文件。
- 已完成文档收口：`model/doc/OVERVIEW.md` 已按当前架构重写，独立
  `model/doc/pose_embed.md` 的有效内容并入后删除。
- 不应作为“死代码”删除：history/map/pose/UAV、Habitat 支持、诊断、视频、
  S2R Stage-A，以及当前 train/eval/data queue。它们都有入口、
  已保留模型、结果协议或仓库技能依赖。
- 按产品范围决定，评测失败分类已从实现、结果 schema 和测试中删除；标准导航指标、
  真实运行异常的 `error` 字段、Habitat 路径和视频可视化均保持。

在不取消现有能力的前提下，P0 清理预计可从安装包移出或删除约 200–300 行代码，
并压缩 600 行以上过时文档；完成 P1 合并后，预计还可净减约 500–700 行重复代码。

## 当前规模与核验方式

| 区域 | 文件数 | 行数 | 主要职责 |
| --- | ---: | ---: | --- |
| 包根 | 4 | 709 | CLI、实验规格与模型名 codec |
| `common/` | 28 | 3,703 | 环境、评测结果、history、embedding、工具 |
| `configs/` | 5 | 303 | SatNav/Habitat 正式与 smoke 配置 |
| `habitat_extensions/` | 2 | 59 | 自定义 Habitat measures |
| `model/` | 17 | 8,067 | 模型、训练、推理、评测、诊断、文档与入口 |
| `s2r/` | 35 | 6,949 | Stage-A/Stage-B 与四类数据转换 |
| `scripts/` | 15 | 3,383 | 数据、同步、train/eval queue、Docker |
| **合计** | **106** | **23,173** | 85 Python、12 shell、6 YAML、3 Markdown |

检查依据：逐文件阅读入口与关键实现、全仓引用检索、动态注册表核对、当前 context/
skills/tests 交叉核对。85 个 Python 文件均通过 AST 解析，12 个 shell 文件均通过
`bash -n`，Ruff `F401/F841` 无告警。

已在仓库规定的 `swift-vln-eval-update` 环境运行完整 contract suite：59 项通过。
后续清理落地后仍需追加一次 train→eval smoke。

## 变更前 SatNav train→eval smoke 基线

按 `.codex/skills/swiftvln-smoke-test/SKILL.md` 的固定入口，在任何 P0/P1 清理落地前完成
了 2-GPU 训练与评测基线。训练使用 `MAX_SAMPLES=16`、`SAVE_STEPS=1`、
`SAVE_TOTAL_LIMIT=1`、`USE_SWANLAB=false`、`TRAIN_NUM_GPUS=2`、
`EMBEDDING_MODE=none`；评测使用 `MAX_EPISODES=10`、`ENV_TYPE=satnav`、
`CUDA_DEVICES=0,1`，并覆盖 `val_seen`、`val_unseen`。

- 训练入口：`src/swiftvln/model/script/train/train_swiftvln_qwen_vl.sh`。
- 模型名：`swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-b1.0-pool-s2-noembed-bs16-lr2e-5-210207`。
- checkpoint：`output/swiftvln/swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-b1.0-pool-s2-noembed-bs16-lr2e-5-210207/v0-20260805-210217/checkpoint-1`；配置和 2 个 safetensors shard 完整，训练 loss 为 `1.29994798`。
- `val_seen`：10 episodes，SR `0.00%`、SPL `0.0000`、OS `0.00%`、NE `209.67844198863096m`、平均 `170.0` steps；摘要位于 `results/eval/swiftvln/<模型名>/val_seen/20260805_210539/evaluation_summary.json`。
- `val_unseen`：10 episodes，SR `0.00%`、SPL `0.0000`、OS `0.00%`、NE `19.665423601546554m`、平均 `16.5` steps；摘要位于 `results/eval/swiftvln/<模型名>/val_unseen/20260805_210856/evaluation_summary.json`。
- 日志：`/tmp/smoke_swiftvln_pre_p0p1_train_20260805.log`、`/tmp/smoke_swiftvln_pre_p0p1_eval_20260805.log`。smoke 产物按 skill 约定保留，供变更后对照。

## 功能总览

```text
swiftvln CLI
├─ train ─> trainer ─> dataset + template ─> registered Qwen-VL model
├─ eval  ─> eval_runner ─> evaluator ─> inference + environment ─> ResultRecorder
├─ queue ─> train_queue / eval_queue ─> shell entrypoints
└─ s2r-data ─> registry ─> dataset-specific pair builders and QA tools

S2R Stage-A: raw datasets ─> manifest ─> teacher/adapter training ─> retrieval eval
S2R Stage-B: Stage-A checkpoint ─> UAV adapter ─> SwiftVLN train/eval embedding path
```

### 1. 入口、实验约束与命名

| 文件 | 已实现功能 |
| --- | --- |
| `cli.py`, `__main__.py` | 提供 `swiftvln train/eval/queue/s2r-data`，按需导入重依赖模块。 |
| `experiment.py` | 定义 `SwiftVLNExperimentSpec`；统一校验环境、模型族、窗口、memory、embedding 和 map 约束；编码/解析当前及历史模型名；输出 shell/JSON。 |
| `model/__init__.py` | 注册 Qwen2.5-VL/Qwen3-VL 的 SwiftVLN model 与 template。 |

实验规格支持 Habitat/SatNav、Qwen2.5-VL/Qwen3-VL、history/map memory、
per-frame/GTC/Segment-GTC、vanilla/initial prompt、none/pose/pose-FiLM/UAV
embedding，并显式拒绝不兼容组合。

### 2. 训练与模型

| 文件 | 已实现功能 |
| --- | --- |
| `model/arguments.py` | ms-swift SFT 参数扩展；窗口、overlap、history、map、prompt、pose、UAV 等配置与校验。 |
| `model/dataset.py` | 读取 trajectory data；把轨迹切成多轮窗口；采样历史帧；构造 action symbol、图像、pose/map memory；为 overlap 样本屏蔽重复动作 loss。 |
| `model/template.py` | Qwen2.5/Qwen3 对话模板；把 `<history_memory>`/`<current_image>` 替换为真实视觉 embedding，并处理压缩后的 token 布局。 |
| `model/model.py` | 注册 HF/ms-swift wrapper；扩展特殊 token；适配 Qwen3 `inputs_embeds`；挂载并恢复 pose/UAV embedding 模块。 |
| `model/trainer.py` | 把自定义参数、dataset 和 template 接入 ms-swift SFT。 |
| `model/script/train/*.sh` | 规范化单次多卡训练入口、环境变量默认值与训练元数据。 |

训练主线已经实现：32-frame 窗口、未来动作分组、可选滑窗重叠与 loss mask、
历史采样/压缩、首帧提示、地图记忆、pose additive/FiLM、Stage-A UAV adapter，
以及 Qwen2.5/Qwen3 两个模型族。

### 3. 推理与评测

| 文件 | 已实现功能 |
| --- | --- |
| `model/eval.py` | 评测 CLI 参数和 `EvalRunner` 启动。 |
| `model/eval_runner.py` | 模型加载、分布式初始化、scene 稳定切分、rank 恢复、episode 编排、最终结果汇总。 |
| `model/evaluator.py` | 组合环境、诊断和 inference；执行 episode state machine、动作 step、视频与运行异常兜底。 |
| `model/inference.py` | 视觉特征缓存、history/map 构造、多轮 prompt、window/overlap 状态、action generation。 |
| `model/diagnostics.py` | `SWIFTVLN_DEBUG` 下的 map、initial-view、token 注入和 timing 诊断。 |
| `common/eval/results.py` | 追加式 `result.jsonl`、去重/恢复、rank 完成标记、最终 summary 与压缩触发。 |
| `common/eval/reporting.py` | 总体/trajectory-type/timing 指标和 SwanLab 报告。 |
| `common/eval/environment.py` | 加载配置、创建 wrapper、动作解析、视频与俯视图。 |
| `model/script/eval/*.sh`, `scripts/eval/*.sh` | 分布式评测、按模型名解析、入队、队列消费和常驻 worker。 |

评测不是简单的一次性脚本：它支持 Habitat/SatNav、单机多卡、确定性 scene 分配、
中断续跑、跨 rank 去重、持久化完成标记、视频和 trajectory type 分组。

### 4. History、地图与 embedding

| 区域 | 已实现功能 |
| --- | --- |
| `common/history_processors/` | per-frame 历史选择，pooling/ToMe 压缩，Soft K-Means GTC，分段 Segment-GTC；统一 processor 接口。 |
| `model/map_memory.py` | 根据 SatNav 轨迹位置渲染 global/local map memory，执行可见区域 mask、缓存与调试统计。 |
| `common/embedding_enhancement/` | 可组合增强 pipeline；pose 归一化、additive/FiLM 融合；加载 Stage-A checkpoint 并执行 UAV adapter。 |
| `common/constants.py` | action symbol、视觉占位 token 和 prompt 常量。 |

这些并非未使用的研究残留。当前保留模型覆盖 vanilla、无历史、随机/log 历史、GTC、
Segment-GTC、map、initial、pose-FiLM 和 overlap 变体；删除对应实现会破坏模型名语义、
checkpoint 兼容或现有评测能力。

### 5. 环境与配置

| 区域 | 已实现功能 |
| --- | --- |
| `common/env/` | `EnvWrapper` 抽象，以及 Habitat/SatNav episode、observation、action、metrics 适配。 |
| `habitat_extensions/` | `OracleNavigationError`、`OracleSuccess` 两个仍需自定义的 measure。 |
| `configs/vln_r2r*.yaml` | Habitat R2R 正式与 smoke 配置。 |
| `configs/satnav_task*.yaml` | SatNav 正式与 smoke 评测配置；当前默认数据为 `SatNav-v0.1`。 |
| `configs/satnav_trajectory_generation.yaml` | SatNav trajectory generation 配置。 |

### 6. S2R

| 区域 | 已实现功能 |
| --- | --- |
| `s2r/dataset.py`, `split.py` | SatDronePair manifest、数据加载、去重和 split。 |
| `s2r/model.py` | teacher vision tower、token adapter/projector、masked pooling。 |
| `s2r/losses.py` | 双向对比损失、global cosine loss、retrieval metrics。 |
| `s2r/trainer.py`, `eval.py`, `arguments.py` | Stage-A 训练、断点恢复、checkpoint/progress、检索评测和参数。 |
| `s2r/data_generation/` | DenseUAV、GTA-UAV、SUES-200、UAV-VisLoc 转换；统一命令注册、配置启动、preview、recrop、variant merge 和 QA。 |

Stage-A 的产物由 `common/embedding_enhancement/uav_adapter.py` 在 Stage-B 加载，
因此不能把整个 `s2r/` 当作离线脚本删除。若只制作推理镜像，可以不打包
`s2r/data_generation/`，但仓库仍应保留它以保证数据可复现。

### 7. 数据与运维脚本

| 区域 | 已实现功能 |
| --- | --- |
| `scripts/data_process/` | 检查 SatNav 版本/城市/trajectory type；从 canonical episodes 生成 train/eval split。 |
| `scripts/data_sync/` | 跨服务器同步和校验 SatNav 数据。 |
| `scripts/train/` | 训练队列、并发资源选择、任务状态与元数据写入。 |
| `scripts/eval/` | 由模型名恢复实验规格，维护 eval queue 和 worker。 |

这些脚本是 `.codex/CODEX_CONTEXT.md` 和仓库 skills 中声明的 canonical workflow，
不应仅因没有 Python import 就判定为死代码。

## 可精简项

### P0：低风险，优先处理

| 对象 | 建议 | 证据与边界 |
| --- | --- | --- |
| `scripts/docker/docker_run.sh` | **已完成**：重写为 17 服务器的非交互、幂等常驻容器 bootstrap。 | 默认复用/启动 `streamvln-container`，使用 `--restart unless-stopped` 和 `sleep infinity`；不再使用 `--rm` 或交互询问，只有 `RECREATE=true` 才显式重建，另提供无副作用的 `DRY_RUN=true`。通过 `bash -n`、默认/指定 GPU dry-run 和非法布尔值拒绝测试。 |
| eval-time template 链路 | **已完成**：删除 `eval.py --template_type`、eval shell 的 `TEMPLATE_TYPE`、`EvalRunner.load_template()`、`create_evaluator(..., template)`、`SwiftVLNEvaluator.template` 及 summary 冗余字段。 | template 在评测中只创建、传递、保存，从未读取；`inference.py` 自行构造 prompt token 与视觉 embedding。训练使用的 `model/template.py` 与模型注册保持不变；7 项结构 contract 通过，旧 CLI 参数被拒绝，现有 checkpoint 的 2-GPU `val_unseen` 单 episode 实测通过（摘要：`results/eval/swiftvln/<基线模型名>/val_unseen/20260805_211258/evaluation_summary.json`）。 |
| `s2r/__init__.py` | **已完成**：缩成最小 package marker，删除历史 lazy re-export 表和 `__getattr__`。 | 仓库内部全部显式导入 `swiftvln.s2r.dataset/model/losses`，没有顶层导出消费者；contract test 固定要求显式子模块导入。 |
| `scripts/data_process/__init__.py` | **已完成**：缩成最小 package marker。 | 删除 eager import、副作用和过期 `ver_260202` 示例；`run_all.py`、`process_episodes.py` 等 canonical 脚本及其直接入口保持不变。 |
| `model/script/test/test_uav_adapter_strategy.py` | **已完成**：迁移并改造成可由 unittest discover 执行的 `tests/test_uav_adapter_strategy.py`。 | Stage-B synthetic checkpoint loader/forward smoke 保留，测试代码已从运行时 package 删除；context 已同步新路径。 |
| `model/doc/OVERVIEW.md` | **已完成**：用当前架构、约束、入口与职责短文替换 727 行历史说明。 | 现在记录 `SatNav-v0.1`、默认 overlap 0、runner/evaluator/inference 分层、Habitat/视频保留边界及 embedding 四选一；历史实验表与失真的伪代码已删除，P0 完成后的完整 contract suite 为 63/63 通过。 |
| `model/doc/pose_embed.md` | **已完成**：有效内容并入新 overview 后删除独立文件。 | pose 表示、归一化、additive/FiLM 公式、零初始化、checkpoint 恢复和真实实现路径只维护一份，避免再次漂移。 |

### P1：先合并，验证后删除旧实现

| 重复组 | 当前规模 | 合并方案 |
| --- | ---: | --- |
| 四个 `*/sample_preview.py` | **已完成**：合并为根目录 `sample_preview.py` 的 schema-aware renderer。 | DenseUAV/GTA-UAV/SUES/UAV-VisLoc adapter 只负责分组、字段、路径和 label；registry、launcher、SUES pipeline、README/config 已迁移，四个旧文件删除。四类合成 contract 与 `/mnt/data3/.../SatDronePair/{denseuav,gta,sues,uavvisloc}` 真实 schema 抽样均成功输出 JPEG。 |
| SUES/UAV-VisLoc `center_recrop_pairs.py` | **已完成**：合并为根目录共享实现并删除两个旧 wrapper。 | `PairSchema` 自动适配 `satellite_file/drone_file` 与 `export_*_path`，统一 image recrop、CSV 复制和 metadata 写入；两个 registry 命令与 YAML/README 接口保持不变，3 项双 schema contract 通过。 |
| `sues/merge_variants_dense_style.py` 与通用 `merge_variants.py` | **已完成**：通用命令增加 one-root/variant-map 模式，专用实现删除。 | SUES 已迁到 `merge_variants --dataset-dir --variant-map`，UAV-VisLoc 原 `--variant NAME=DIR` 模式保持；registry、README/config 已移除旧命令，两种输入模式均有 materialization/schema/metadata contract。 |

这些文件都被 registry、README 或 SUES pipeline 使用，不能先删再补。它们是“重复实现”，
不是“未使用文件”。

### P2：只有明确缩减产品范围时才删

| 能力切片 | 相关文件 | 删除代价 |
| --- | --- | --- |
| 深度诊断 | `model/diagnostics.py`（326 行）及 map debug 分支 | 失去 `SWIFTVLN_DEBUG` 下的 token/map/initial/timing 定位能力。 |
| 失败分类 | **已完成**：删除 `common/utils/error_analyzer.py` 及 Habitat 轨迹采样、分析和结果透传。 | 三个派生分类字段不再写入结果；真实 episode 异常仍以 `error` 持久化，标准指标不变。结构契约同时固定 Habitat wrapper/config/extensions 和两类视频入口必须保留；定向测试 9/9、完整 contract suite 73/73、Ruff 均通过。 |
| 视频与可视化 | `video_utils.py`、部分 `image_utils.py`、evaluator/environment 分支 | 失去 `--save_video`、压缩和俯视图输出。 |
| Habitat 支持 | `common/env/habitat.py`、`habitat_extensions/`、`vln_r2r*.yaml` | 项目变成 SatNav-only；需同步 experiment codec、CLI、tests 和文档。 |
| S2R 数据生产 | `s2r/data_generation/` | 只能在数据/manifest 已冻结且接受不可从原始集复现时删除；更合理的是从推理部署包排除。 |

## 明确保留

- `experiment.py`：train/eval 模型名与约束的单一事实来源。
- `model/{arguments,dataset,template,model,trainer}.py`：训练闭环。
- `model/{eval,eval_runner,evaluator,inference}.py` 与 `common/eval/`：可恢复的分布式评测闭环。
- `common/history_processors/`、`model/map_memory.py`、pose/UAV embedding：现有模型变体和 checkpoint 所需。
- Habitat/SatNav wrapper、正式/smoke configs、自定义 measures：两个受支持环境所需。
- train/eval queue、data process/sync：当前 canonical operation 所需。
- S2R Stage-A core 和四类数据 builder：Stage-B 权重来源与数据可复现性所需。

## 推荐执行顺序

1. 把 Docker 脚本改成可重建 17 常驻容器的 bootstrap；精简两个 facade；迁移 UAV test；重写两份过时文档。
2. 删除 eval-time template 传递，并做一次模型名解析、单 episode eval smoke 和续跑测试。
3. 先为 preview/recrop/merge 写统一实现与等价性测试，再逐项切 registry 和 pipeline。
4. 在正确项目环境运行全部 contract tests、train smoke、eval smoke；路径变化同步
   `.codex/CODEX_CONTEXT.md`。
5. 失败分类已按产品决定删除；Habitat、视频和 S2R 数据重建能力仍按当前产品边界保留。

判定原则：shell 入口、动态 registry 和可复现工具不能只靠 Python 静态引用计数判死；
“无内部引用”也不代表没有仓库外 notebook/manual consumer。因此 P0 中涉及 public facade
和 Docker 的项，在真正删除前仍应做一次外部使用确认。
