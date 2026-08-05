# Codex Repo Context (SwiftVLN)

最后核验：2026-08-05

本文件只保存会影响当前开发、训练、评测、数据处理和资产管理的事实。
它不是 changelog，也不记录一次性故障、历史 smoke 结果或已经删除的路径。

- 清理前的全量快照：
  `.codex/archive/CODEX_CONTEXT_PRE_CLEANUP_2026-08-05.md`
- 归档文件只在追溯旧实验、迁移过程或已清理资产时按需读取；新 agent 启动时不要加载。
- 若本文件与当前源码、配置或文件系统冲突，先以可验证的当前状态为准，再同步更新本文件。
- 更细的当前态报告按需读取：
  - `reports/current_state.md`：代码入口、环境、数据和持久化产物
  - `reports/model_zoo.md`：22 个保留模型及最终指标
  - `reports/s2r_rebuttal.md`：S2R、rebuttal 和可复现性边界

## Repository Identity

- 唯一 active 仓库根：
  `/mnt/data1/home/jiangjiajun/workspace/SwiftVLN`
- Python 包：`src/swiftvln`
- 项目名/版本：`swiftvln 0.1.0`
- Python 要求：`>=3.9`
- 主线模型标识：`swiftvln`
- 主线训练输出：`output/swiftvln/<experiment>/`
- 主线评测输出：`results/eval/swiftvln/<model>/<split>/<timestamp>/`
- 当前长期实验资产只以 `output/model_zoo/` 为准。

不要引用已删除的 `SwiftVLN-refactor`、`output/overlapvln` 或旧 ms-swift 3.x
源码目录。结果 JSON 中的旧路径只表示 provenance，不代表路径仍可执行。

## Current Source Layout

- `src/swiftvln/model/`：SwiftVLN 主线模型、训练和评测实现；评测职责分为
  `evaluator.py`（组合环境服务并运行 episode loop）、`inference.py`
  （frame/history/prompt/window）、`eval_runner.py`（分布式编排）和
  `diagnostics.py`（可选 map/initial/timing 诊断）；
  训练参数和 dataset hook 直接位于
  `arguments.py` / `trainer.py`，不经过仓库内单实现 Base 层
- `src/swiftvln/cli.py`：单模型 CLI，直接分发 SwiftVLN 与 queue；无 model registry/runner 中间层
- `src/swiftvln/experiment.py`：train/eval 共用的 ExperimentSpec、约束和模型名 codec
- `src/swiftvln/common/eval/`：评测环境能力与报告公共组件；
  `EvaluationEnvironment` 组合 Habitat/SatNav 配置、wrapper、动作与可视化，
  `ResultRecorder` 独立负责 JSONL 恢复/去重、分布式完成标记和最终指标落盘；
  评测主线不经过仓库内单实现 Base 类
- `src/swiftvln/common/env/`：Habitat / SatNav 环境抽象
- `src/swiftvln/common/history_processors/`：history 压缩实现
- `src/swiftvln/common/embedding_enhancement/`：pose / UAV adapter 增强
- `swiftvln.common` 顶层不提供跨子包 facade；内部代码直接从 `constants`、
  `env`、`eval`、`history_processors`、`utils` 等具体模块导入
- `src/swiftvln/configs/`：主线配置
- `src/swiftvln/scripts/`：训练、评测、数据处理、同步和监控入口
- `src/swiftvln/s2r/`：S2R Stage-A / Stage-B 相关实现
- `baseline/{streamvln,navila,uninavid,openfly}/`：四个 SatNav baseline
- `runtime/`：队列、计划、状态、临时运行元数据；不是长期实验归档

## Canonical Entry Points

| 用途 | 当前入口 |
| --- | --- |
| CLI | `src/swiftvln/cli.py`（安装后命令 `swiftvln`；train/eval 默认即 SwiftVLN） |
| SwiftVLN 单次训练 | `src/swiftvln/model/script/train/train_swiftvln_qwen_vl.sh` |
| SwiftVLN 训练队列 | `src/swiftvln/scripts/train/train_queue.sh` |
| 训练 watchdog | `src/swiftvln/scripts/train/train_watchdog.sh` |
| SwiftVLN 分布式评测 | `src/swiftvln/model/script/eval/eval_swiftvln_qwen_vl_distributed.sh` |
| 按模型名评测 | `src/swiftvln/scripts/eval/eval_by_name.sh` |
| Eval 队列 | `src/swiftvln/scripts/eval/eval_queue.sh` |
| Eval 入队 | `src/swiftvln/scripts/eval/enqueue_eval.sh` |
| Eval 常驻 worker | `src/swiftvln/scripts/eval/start_eval_worker.sh` |
| SatNav 全流程处理 | `src/swiftvln/scripts/data_process/run_all.py` |
| SatNav 数据合并 | `src/swiftvln/scripts/data_process/merge_satnav_data.py` |
| 数据同步 | `src/swiftvln/scripts/data_sync/*.sh` |
| GPU 健康监控 | `src/swiftvln/scripts/monitor/gpu_health_monitor.sh` |
| 实验计划 | `runtime/plans/` |
| S2R 数据生产 | `swiftvln s2r-data ...` / `src/swiftvln/s2r/data_generation/` |
| S2R Stage-A 训练/评测 | `src/swiftvln/s2r/trainer.py` / `src/swiftvln/s2r/eval.py` |
| 安装说明 | `docs/installation.md` |

优先使用上述入口和对应 repo skill，不拼装一次性替代流程。

## Supported Model Families

SwiftVLN 的 Qwen-VL family 共用同一组 train/eval shell 入口：

| `MODEL_FAMILY` | `MODEL_TYPE` | 默认离线 base model |
| --- | --- | --- |
| `qwen2_5_vl`（默认） | `swiftvln_qwen2_5_vl` | `/mnt/data1/home/jiangjiajun/.cache/modelscope/models/Qwen/Qwen2___5-VL-3B-Instruct` |
| `qwen3_vl` | `swiftvln_qwen3_vl` | `/mnt/data1/home/jiangjiajun/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct` |

- 更大模型通过 `BASE_MODEL_PATH` 或 `MODEL_PATH` 显式覆盖。
- Qwen3-VL 训练/smoke 必须显式设置 `USE_LIGER_KERNEL=false`；当前默认 `true`
  只适用于默认 Qwen2.5 路径。
- 当前入口只接受 Qwen2.5-VL 和 Qwen3-VL；Qwen3.5 不是受支持模型族。
- 起训前检查本地模型目录存在且非空；正式任务默认离线，不依赖临时在线下载。

## Current SatNav Data

- 数据根：`/mnt/data3/jiangjiajun/dataset/satnav_datasets`
- 当前默认数据集：`SatNav-v0.1`
- 训练轨迹：
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/trajectory_data`
- 评测 episodes：
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/episodes/eval/{split}/all_episodes.json`
- Scenes：
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/scenes`
- 当前训练 trajectory cache 为 `105164` 条。
- 主线 `src/swiftvln/configs/satnav_task.yaml` 默认 `SPLIT: val_seen`。
- 四个 baseline 的 `satnav_task.yaml` 默认 `SPLIT: all`，即顺序评测
  `val_seen` 和 `val_unseen`。
- 正式横向比较必须同时报告 `val_seen` 与 `val_unseen`。

自定义数据覆盖：

- 训练：`VLN_DATA_PATH=<trajectory_data_dir>`
- 评测：`EVAL_CONFIG_PATH=<yaml>`；相对路径按 `src/swiftvln/` 解析

若默认 SatNav 版本变化，至少同步：

- `src/swiftvln/configs/satnav_task.yaml`
- `src/swiftvln/scripts/train/train_queue.sh`
- 若 baseline 默认也变化，再同步四个 baseline 的 config 和 `*satnav.sh` 入口

历史 `ver_260327` / `ver_260403` / `ver_260404` / `ver_260418` 的构建统计、
重映射结果和 split 修复均已移入归档，不属于当前默认数据说明。

## Mainline Training Conventions

当前单次训练默认值：

- `VLN_ENV_TYPE=satnav`
- `MODEL_FAMILY=qwen2_5_vl`
- `NUM_FRAMES=32`
- `NUM_HISTORY=8`
- `NUM_FUTURE_STEPS=4`
- `NUM_OVERLAP=0`
- `HISTORY_PROCESSOR_TYPE=per_frame`
- `COMPRESS_STRIDE=2`
- `USE_TOME=false`
- `MEMORY_METHOD=history`
- `MAX_SAMPLES=0`（全量）
- `NUM_EPOCHS=1`
- `LEARNING_RATE=2e-5`
- `BATCH_SIZE=8`（per rank）
- `GRAD_ACCUM_STEPS=1`
- `USE_DEEPSPEED=true` / `DEEPSPEED_CONFIG=zero2`
- `SAVE_STEPS=1000` / `SAVE_TOTAL_LIMIT=1`
- 默认 Conda 环境：`swift-vln-train-update`

GPU 选择：

- 默认使用当前环境全部可见 GPU，不硬编码 `0-7`。
- `TRAIN_NUM_GPUS=<N>`：取当前可见集合前 N 张。
- `TRAIN_CUDA_DEVICES=<csv>`：显式选择。
- `TRAIN_DRY_RUN=true`：只检查配置和 GPU 解析。

恢复训练：

- `RESUME_FROM_CHECKPOINT=<abs_path>`
- `RESUME_ONLY_MODEL=true|false`
- `OUTPUT_DIR_OVERRIDE=<path>`

不常用但必须保留的语义约束：

- no-memory 的唯一支持配置是 `HISTORY_PROCESSOR_TYPE=per_frame` 且
  `NUM_HISTORY=0`；没有独立 `USE_MEMORY=false`。
- `NUM_OVERLAP>0` 固定使用 `stride = NUM_FRAMES - NUM_OVERLAP`，尾窗不回挪；
  `NUM_OVERLAP=0` 保持完整尾窗覆盖；overlap 必须小于窗口且按
  `NUM_FUTURE_STEPS` 对齐。
- `MEMORY_METHOD=map` 会替换历史 RGB frame，只支持 SatNav + per-frame，要求
  `USE_TOME=false`、`USE_POSE_EMBED=false`、`USE_UAV_ADAPTER=false`。
- map render cache 默认推导为 `{dataset_root}/map_cache`；用
  `SWIFTVLN_MAP_CACHE_DIR=<path>` 覆盖，或用 `off|false|none|0|disable|disabled|no`
  关闭。
- `USE_UAV_ADAPTER` 当前仅支持 `UAV_ADAPTER_APPLY_SCOPE=all_images`。
- 对外 embedding mode 只有 `none|pose|posefilm|uav`；pose 与 UAV 不允许组合，
  ExperimentSpec、模型 loader、运行时配置和底层 factory 都执行同一约束。

模型命名、解析和跨字段校验以 `src/swiftvln/experiment.py` 为唯一事实源；
非默认 GTC temperature/iterations 会编码进名称。更多参数语义按需查看
`src/swiftvln/model/doc/OVERVIEW.md`，不要把所有参数复制回启动上下文。

## Mainline Evaluation Conventions

- 默认 Conda 环境：`swift-vln-eval-update`。
- SatNav 主线配置：`src/swiftvln/configs/satnav_task.yaml`。
- `eval_by_name.sh` 只接受 ExperimentSpec 可验证的 `swiftvln-*` 模型名，并从名称
  解析 family、窗口、overlap、history、memory 和 embed 配置。
- 模型解析优先级：
  1. 显式 `MODEL_PATH`
  2. `output/swiftvln/<model>/v*/checkpoint-*` 或 `checkpoint-*`
  3. `output/model_zoo/swiftvln/HF_model/<model>`
- `output/model_zoo/backbones/HF_model/*` 不在自动搜索范围，必须显式设置
  `MODEL_PATH`。
- 未显式设置 `OUTPUT_DIR` 时，`AUTO_RESUME_EVAL=true` 会复用最近的未完成目录。
- 每个 rank 逐 episode append `result.jsonl`，resume/去重键为
  `scene_id::episode_id`；rank0 等待文件 marker 后离线汇总。
- episode 先按 scene 稳定排序，再在全局序列上按 rank round-robin，保证小规模
  多 scene smoke 也能均衡使用各 rank。
- `SwiftVLNEvaluator` 只组合环境服务、`SwiftVLNInferenceSession` 和
  `EnvironmentEpisodeLoop`；新增 history/inference 能力不要重新塞回 episode loop。
- `evaluation_summary.json` 顶层指标是当前 split 全 episode 的直接平均；
  `by_trajectory_type` 只提供细分，不做重加权。

Eval 队列文件固定为：

- `runtime/eval_queue/eval_todo.txt`
- `runtime/eval_queue/eval_done.txt`
- `runtime/eval_queue/eval_failed_todo.txt`
- per-host 状态：`runtime/eval_queue/eval_queue_last_run_<hostname>.json`

`eval_queue.sh` 只从 `eval_todo.txt` 读取模型；新增任务统一使用
`enqueue_eval.sh`，不再支持位置参数模型列表或交互式向导。`DYNAMIC_TODO=true`
会在每轮后刷新 todo，`WAIT_FOR_NEW_TASKS=true` 需要动态模式并在空队列时常驻；
`AUTO_TODO=true` 为旧启动器保留，并会启用动态模式。可用 `--check-queue`
校验队列中的 ExperimentSpec 模型名而不启动评测。

## Baseline Conventions

| Baseline | 训练入口 | 评测入口 | Conda 环境 |
| --- | --- | --- | --- |
| StreamVLN | `baseline/streamvln/scripts/train_satnav.sh` | `baseline/streamvln/scripts/eval_satnav.sh` | `streamvln-baseline` |
| NaVILA | `baseline/navila/scripts/train_satnav.sh` | `baseline/navila/scripts/eval_satnav.sh` | `navila-baseline` |
| UniNaVid | `baseline/uninavid/scripts/train_satnav.sh` | `baseline/uninavid/scripts/eval_satnav.sh` | `uninavid-baseline` |
| OpenFly | `baseline/openfly/scripts/train_satnav.sh` | `baseline/openfly/scripts/eval_satnav.sh` | `openfly-baseline` |

- 四个 baseline 默认训练数据都是 `SatNav-v0.1/trajectory_data`。
- 评测公开接口统一使用 `--model_dir` + `--model_name`；split 和数据由各自
  `configs/satnav_task.yaml` 控制，不使用位置 split 参数。
- 模型占位目录 `baseline/<name>/model/` 不跟踪权重；下载或本地准备流程负责落盘。
- 新训练输出为 `output/<name>-baseline/<experiment>/`，评测输出为
  `results/<name>-baseline/...`；精选长期副本进入 `output/model_zoo/baseline/`。
- scratch/continue、动作格式、采样和 checkpoint 加载差异以各 baseline 脚本、文档
  及对应 repo skill 为准，不在启动上下文重复维护。

## Data Processing and Merge

- SatNav 数据处理任务使用 `.codex/skills/satnav-data/SKILL.md`。
- 两版本合并、overlap 分析或 episode ID remap 使用
  `.codex/skills/merge-satnav-data/SKILL.md`。
- 默认处理入口 `run_all.py` 会先标准化 trajectory type，再生成 episodes。
- 合并必须先 analyze；identical overlap 去重，conflicting overlap 对 secondary
  remap ID，并同步重写 trajectory summary、annotations 和 image directory。
- 合并后必须校验 `scene_id + episode_id` 唯一，以及 summary / annotations /
  images 三者严格对齐。

## Runtime and Servers

三台服务器共享同一工作区和输出挂载；代码、队列和结果文件只需在共享目录修改一次。

| Server | 访问方式 | GPU/运行约定 |
| --- | --- | --- |
| 98 | localhost | 本机 8 GPU |
| 73 | `ssh 10.246.152.73` | 远端 8 GPU |
| 17 | `ssh 10.246.132.17` 后进入 Docker | 远端 8 GPU；容器名先用 `docker ps` 查询 |

- SSH 只用于远端 GPU/进程检查以及 tmux 启停；共享文件操作始终本地执行。
- Conda 初始化：
  `source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh`
- 主线训练环境：`swift-vln-train-update`（Python 3.10）
- 主线评测环境：`swift-vln-eval-update`（Python 3.9 + Habitat 0.2.4）
- SatNav 工具环境：`satnav`
- baseline 环境见上一节表格。
- 当前 ms-swift 源码：`/mnt/data1/home/jiangjiajun/workspace/ms-swift`
- 固定版本：`ms-swift 4.2.0.dev0`，commit
  `ad7d5c5157b59afa1faceb04266274392f145745`
- SwiftVLN 当前直接使用 ms-swift 4.x API，不保留 3.x fallback。

所有长训练/评测任务必须在 tmux 中运行：

- `train_<short_desc>_<HHMMSS>`
- `eval_<short_desc>_<HHMMSS>`

训练队列非交互入口：

```bash
TRAIN_EXPERIMENTS_FILE=/path/to/experiments.sh \
  bash src/swiftvln/scripts/train/train_queue.sh
```

训练队列只接受上述文件协议，不再提供交互式向导或单字母 shortcode。
配置文件必须定义 `ENV_TYPE=satnav|habitat`，并提供至少一个五字段实验：
`model|KEY=VALUE 覆盖（或 default）|描述|数据集名|数据路径`；当前 `model`
只接受 `swiftvln`，覆盖键使用大写环境变量名。可用 `--check-config` 只校验配置而不启动训练。
运行状态写入：

- `runtime/train_queue/train_queue_last_run_<hostname>.json`
- `runtime/train_queue/runs/<hostname>_<session>/`

## Persistent Assets

`output/model_zoo/` 当前共有 22 个精选模型：

- `backbones`：3
- `baseline`：8
- `swiftvln`：11

约定：

- `HF_model/`：可直接加载的 config、processor/tokenizer 和独立权重。
- `Results/`：最终 `val_seen` / `val_unseen` 结果的权威副本。
- `Training_Log/`：精选训练元数据，不代表存在 optimizer state。
- 历史 DeepSpeed `global_step*`、`latest` 和 `zero_to_fp32.py` 已清理；model zoo
  可用于推理、评测或仅权重初始化，不能原样恢复 optimizer/scheduler。
- 当前工作区没有 `output/swiftvln/`、四个 baseline 原始输出、`results/`、`logs/`
  或 `swanlog/`；新运行会按脚本重新创建。

精确模型名、指标和结果格式只维护在 `reports/model_zoo.md` 及对应 JSON 中。

## S2R and Rebuttal

- SatDronePair 数据根：`/mnt/data3/jiangjiajun/dataset/SatDronePair`
- canonical manifest：`runtime/s2r/manifests/manifest_v1.jsonl`（19365 条）
- 当前唯一正式 Stage-A 权重：
  `output/s2r/s2r-swiftvln-baseline-gta-dedup-10ep-bs64-lr1e-4-20260727-160227/best.pt`
- 该权重已通过独立评测和 Stage-B loader check，可评测、可接 Stage-B。
- 对应 `train_args.json` 引用的 `manifest_v2_gta_dedup.jsonl` 当前不存在；
  `manifest_v1.jsonl` 不能视为同一训练输入。精确复训前必须重新生成并验证 v2。
- Rebuttal episodes：`output/rebuttal/`
- 相关场景资产：`output/PCD-data/`、`output/3DGS-data/`；删除前必须确认
  PCD/3DGS rebuttal 已不再需要。

指标、数据源分布和可复现性细节只维护在 `reports/s2r_rebuttal.md`。

## Smoke Tests and Tests

- 主线 train -> checkpoint -> eval smoke 使用
  `.codex/skills/swiftvln-smoke-test/SKILL.md`。
- 任一 baseline smoke 使用 `.codex/skills/baseline-smoke-test/SKILL.md`。
- smoke 必须显式限样、验证 checkpoint 可评测，并在完成后只清理本次 smoke 产物。
- 顶层 `tests/` 是重构契约 suite，覆盖 CLI、模型名、dataset/window、history
  processor、Habitat 扩展和 JSONL resume/去重；运行：
  `PYTHONPATH=src python -m unittest discover -s tests -v`。
- 仅保留 Stage-B loader smoke：
  `src/swiftvln/model/script/test/test_uav_adapter_strategy.py`。

不要在启动上下文保存某次 smoke 的输出路径、loss、时间或已删除日志；需要当前验证时
重新运行对应 smoke skill。

## Operating Rules

- 优先使用现有脚本和 repo skills。
- 未经用户明确要求，不修改外部仓库；SwiftVLN 适配优先在本仓库完成。
- 不回滚用户的无关改动。
- 除非用户明确要求，不创建 commit。
- commit 使用 conventional type：`feat/fix/refactor/docs/test/perf/chore`，中文简洁说明。
- 默认失败策略：单项失败可继续后续队列项，但必须记录失败原因。
- 训练、评测、数据和模型资产的长期事实必须落在代码、配置、原始结果或当前报告中，
  不依赖聊天记忆。

## Context Maintenance

以下变化发生时必须原位更新本文件：

- 目录/包结构或关键入口变化
- 训练/评测主流程、队列路径或状态文件变化
- 默认数据版本、模型 family 或 Conda 环境变化
- model zoo、S2R、rebuttal 的长期保留策略或当前限制变化

维护原则：

1. 只写变化后的当前事实，不追加“曾经如何修复”的过程。
2. 一次性实验、benchmark、smoke 和故障日志不要写入本文件。
3. 已失效但仍需追溯的内容移入 `.codex/archive/` 或依赖 Git 历史。
4. 详细参数优先留在源码/脚本/专项文档；本文件只保留启动时必须知道的约束。
5. 更新后检查所有路径存在性，并确认没有同时保留互相冲突的新旧说法。
