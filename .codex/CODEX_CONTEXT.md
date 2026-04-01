# Codex Repo Context (SwiftVLN)

本文件是 SwiftVLN 仓库内的默认工作上下文。
在本仓库执行任务时，优先以本文件为准。

## Repository Scope

- Repo: `SwiftVLN`
- Root: `/mnt/data1/home/jiangjiajun/workspace/SwiftVLN`
- 主要代码域：`src/swiftvln/*`
- 当前主线模型：`overlapvln`（未明确指定时默认按 overlapvln 处理）

## Current Architecture Snapshot

SwiftVLN 已从 `ms-swift/examples/vln` 迁移为独立仓库，核心结构如下：

- 包根：`src/swiftvln`
- 模型目录：`src/swiftvln/models/{streamvln,compressvln,overlapvln,monovln,uninavid}`
- 配置目录：`src/swiftvln/configs`
- 脚本目录：`src/swiftvln/scripts`

### common 子包重组状态（已完成）

`src/swiftvln/common` 已按职责拆分：

- `common/training/*`: arguments / base_sft / dataset / mixed_dataset / trainer_mixin
- `common/eval/*`: runner / evaluator / reporting
- `common/env/*`: base / habitat / satnav
- `common/history_processors/compressor.py`: 原 `common/compressor.py` 已迁入

兼容性策略：

- 旧路径 shim 仍保留（如 `common/base_sft.py` 等），用于兼容历史导入。
- 新代码优先使用新路径导入（`common.training.*`, `common.eval.*`, `common.env.*`）。

## Key Directories & Entry Scripts

- 实验计划目录：`runtime/plans/` （自然语言实验计划文件，供 orchestrate-plan skill 读取）
- 实验计划 skill：`.codex/skills/orchestrate-plan/SKILL.md`
- 训练队列：`src/swiftvln/scripts/train/train_queue.sh`
- 训练 watchdog：`src/swiftvln/scripts/train/train_watchdog.sh`
- 评测单模型：`src/swiftvln/scripts/eval/eval_by_name.sh`
- 评测队列：`src/swiftvln/scripts/eval/eval_queue.sh`
- 评测入队：`src/swiftvln/scripts/eval/enqueue_eval.sh`
- 评测 worker：`src/swiftvln/scripts/eval/start_eval_worker.sh`
- 评测 monitor：`src/swiftvln/scripts/eval/start_eval_monitor.sh`
- 评测 watchdog：`src/swiftvln/scripts/eval/eval_watchdog.sh`
- 数据处理：`src/swiftvln/scripts/data_process/*.py`
- 数据同步：`src/swiftvln/scripts/data_sync/*.sh`

### Baseline StreamVLN Layout (Updated: 2026-03-09)

`baseline/streamvln` 已按“入口脚本 / 源码实现”分层：

- 入口脚本：`baseline/streamvln/scripts/*.sh`
  - `scripts/train_satnav.sh` -> 调用 `baseline/streamvln/src/train_satnav.py`
  - `scripts/eval_satnav.sh` -> 调用 `baseline/streamvln/src/eval_satnav.py`
  - `scripts/train_eval_satnav.sh` -> 串行执行 train 后自动 eval（含 webhook）
  - `scripts/download_model.sh`
- 源码目录：`baseline/streamvln/src/*`
  - `src/train_satnav.py`
  - `src/eval_satnav.py`
  - `src/dataset/satnav_action_dataset.py`
- 历史产物目录：`baseline/streamvln/checkpoints`、`baseline/streamvln/results`（legacy）
- 当前主流程输出统一在仓库根：
  - 训练模型：`output/streamvln-baseline/<EXP_NAME>/`
  - smoke test 模型：`output/streamvln-baseline/smoketest/<EXP_NAME>/`
  - 评测结果：`results/streamvln-baseline/<EXP_NAME_or_subpath>/<split>/`
- baseline eval 默认：`8` 卡 + `val_unseen`（`baseline/streamvln/scripts/eval_satnav.sh`）
  - SatNav eval split 路由约定（0319 起）：
    - `DATA_PATH` 使用 `{split}` 占位符：`episodes/eval/{split}/all_episodes.json`
    - `--eval_split val_seen` → 展开为 `episodes/eval/val_seen/all_episodes.json`
    - `--eval_split val_unseen` → 展开为 `episodes/eval/val_unseen/all_episodes.json`（目录暂不存在，待新城市引入后生成）
    - 路径不存在时直接报错（`FileNotFoundError`），无 fallback
    - 路由逻辑：`src/swiftvln/common/eval/evaluator.py` 的 `_init_satnav_config()`
    - 配置文件：`src/swiftvln/configs/satnav_task.yaml` 的 `DATASET.DATA_PATH`
  - baseline SatNav eval 默认 split 约定（0319 更新）：
    - `baseline/streamvln/scripts/eval_satnav.sh`
    - `baseline/navila/scripts/eval_satnav.sh`
    - `baseline/uninavid/scripts/eval_satnav.sh`
    - 若**未显式传 split 参数**，默认顺序运行 `val_seen` 和 `val_unseen`
    - 若显式传 `val_seen` / `val_unseen` / `test`，则只跑该单个 split
  - 评测结果目录约定（0319 起）：
    - SwiftVLN 主线模型（overlapvln/streamvln/compressvln/monovln/uninavid）：
      `results/eval/<arch>/<model_name>/<split>/<timestamp>/`
      例：`results/eval/overlapvln/<model>/val_seen/20260319_143025/`
    - SatNav 默认同时跑 `val_seen` + `val_unseen`（不设置 `EVAL_SPLIT`），Habitat 默认 `EVAL_SPLIT=val_unseen`
  - StreamVLN SatNav eval 约定（Updated: 2026-04-01）：
    - 多卡汇总改为 rank0 从 `result.jsonl` 离线去重汇总，不再用末尾 `all_gather(...)` 汇总本地 `results`
    - resume / 去重唯一键使用 `scene_id + episode_id`，避免仅按 `episode_id` 导致跨 scene 冲突
    - `--max_episodes` 语义为“先截断总 episode，再做分布式切分”

### Baseline NaVILA Layout (Updated: 2026-03-12)

`baseline/navila` 已补齐 SatNav train + eval 分层：

- 入口脚本：`baseline/navila/scripts/*.sh`
  - `scripts/train_satnav.sh`
  - `scripts/eval_satnav.sh`
  - `scripts/download_model.sh`
  - `scripts/setup_env.sh`
- 源码目录：`baseline/navila/src/*`
  - `src/train_satnav.py`
  - `src/eval_satnav.py`
  - `src/dataset/satnav_dataset.py`
- 配置目录：
  - `baseline/navila/configs/satnav_task.yaml`
  - `baseline/navila/configs/zero{2,3}.json`
- 模型目录：
  - `baseline/navila/model/navila-siglip-llama3-8b-v1.5-pretrain/`
  - `baseline/navila/model/navila-llama3-8b-8f/`
- 当前主流程输出统一在仓库根：
  - 训练模型：`output/navila-baseline/<EXP_NAME>/`
  - 评测结果：`results/navila-baseline/<EXP_NAME_or_subpath>/<split>/`
  - 路径评测结果：`results/navila-baseline/by-path/<ckpt_name>/<split>/`

NaVILA SatNav eval 约定：

- prompt 与上游 `NaVILA/evaluation/vlnce_baselines/navila_trainer.py` 保持一致
- 动作解析沿用上游自然语言正则逻辑（`stop / move forward / turn left / turn right`）
- 评测环境依赖 `navila-baseline` conda env + `pip install -e /mnt/data1/home/jiangjiajun/workspace/SatNav`
- 输出解析只解码生成后缀（Updated: 2026-04-01）：
  - `baseline/navila/src/eval_satnav.py` 在 `model.generate(...)` 后仅对 `output_ids[:, input_token_len:]` 做 `batch_decode`
  - 避免把 prompt 一起解码后因模板中的 `stop` 文案污染动作正则匹配
- forward 距离解析与训练标签对齐（Updated: 2026-04-01）：
  - 训练标签默认是 `move forward 10 meters`
  - 评测优先解析 `meters`，同时兼容上游遗留的 `cm` 写法
- 断点续跑唯一键使用 `scene_id + episode_id`（Updated: 2026-04-01）：
  - `baseline/navila/src/eval_satnav.py` 在读取 `result.jsonl` 时按联合键去重与跳过
  - 避免仅使用 `episode_id` 导致跨 scene 冲突、误判“已完成”
- 多卡汇总改为 rank0 离线汇总（Updated: 2026-04-01）：
  - 不再依赖末尾 `dist.all_gather(...)` 做跨 rank 汇总
  - rank0 直接读取 `result.jsonl`（联合键去重）并写 `evaluation_summary.json`
  - 用于规避长尾 rank 导致的 NCCL/TCPStore 超时退出
  - 汇总前增加 `dist.barrier()`，避免 rank0 在其他 rank 尚未写完 `result.jsonl` 时提前出 summary

NaVILA SatNav train 补充约定（Updated: 2026-03-25）：

- `baseline/navila/scripts/train_satnav.sh` 现在支持两种初始化模式：
  - `scratch`：从 `baseline/navila/model/navila-siglip-llama3-8b-v1.5-pretrain` 起训
  - `continue`：从 `baseline/navila/model/navila-llama3-8b-8f` 继续训练
  - 无参默认 `scratch`
  - 兼容旧调用：若只传一个非模式参数，则视为 `EXP_NAME`
- `baseline/navila/scripts/train_satnav.sh` 默认使用
  `MASTER_ADDR=127.0.0.1` + 显式 `MASTER_PORT`，
  避免 Docker 容器内 `torchrun --standalone` 的 hostname 解析卡死
- `baseline/navila/scripts/train_satnav.sh` 现在会稳定落盘：
  - `output/navila-baseline/<EXP_NAME>/train.log`
  - `output/navila-baseline/<EXP_NAME>/gpu_metrics.log`
  默认开启 GPU 监控（`ENABLE_GPU_MONITOR=true`），每 `60s` 用 `nvidia-smi` 采样
  `temperature.gpu / utilization.gpu / memory.used / memory.total / power.draw`
- `baseline/navila/scripts/train_satnav.sh` 中：
  - 默认 `MAX_STEPS=120`
  - 若显式传空值 `MAX_STEPS=`，则**不传 `--max_steps`**，用于全量训练
  - 默认 `SAVE_STEPS=20000`
  - 若显式传空值 `SAVE_STEPS=`，则按真实总 step 数自动推导：
    `SAVE_STEPS = ceil(total_steps / SAVE_COUNT_TARGET)`，默认 `SAVE_COUNT_TARGET=4`
  - `SAVE_TOTAL_LIMIT` 默认 `1`（保留最新一个 `checkpoint-*`；训练结束仍会在 `output/navila-baseline/<EXP_NAME>/` 根目录保存最终模型）
- NaVILA SatNav train 默认采样策略（head+stop+turn-protect+fwd-stride，Updated: 2026-03-25）：
  - `SATNAV_HEAD_KEEP=7`：保留每条轨迹前 7 步（帧数 < num_video_frames=8 的独特分布区间，全部保留）
  - stop 步（每 episode 末尾）：**全部保留**（稀有关键动作）
  - 中间区间 turn 步（left/right）：**全部保留**（决策关键少数类，不做 stride）
  - `SATNAV_SAMPLE_STRIDE=5`：中间区间连续 forward run 每 5 步取 1 步（遇 turn 重置计数）
  - 效果：原始 476 万 → **约 264 万（55%）**，turn 覆盖率 100%（旧均匀 stride 仅 33.7%）
  - 若需完整全量训练（不做采样），显式传空值：`SATNAV_HEAD_KEEP= SATNAV_SAMPLE_STRIDE=`
- smoke / 调试时可额外叠加：
  - `SATNAV_MAX_EPISODES`
  - `SATNAV_MAX_SAMPLES`
  - `SATNAV_SAMPLE_RATIO`
  以上三个变量默认为空，**正式全量训练不要设置**

### Baseline UniNaVid Train Modes (Updated: 2026-03-26)

- 训练入口：`baseline/uninavid/scripts/train_satnav.sh`
- 该脚本现在支持两种初始化模式：
  - `continue`：从 `baseline/uninavid/model/Uni-Navid` 继续训练
  - `scratch`：从 `baseline/uninavid/model/vicuna-7b-v1.5` 起训
- 调用方式：
  - `bash baseline/uninavid/scripts/train_satnav.sh continue [EXP_NAME]`
  - `bash baseline/uninavid/scripts/train_satnav.sh scratch [EXP_NAME]`
  - 兼容旧调用：若只传一个非模式参数，则视为 `EXP_NAME`，默认模式仍为 `continue`
- 默认实验命名：
  - `uninavid-baseline-continue-{epochs}ep-data{ver}-bs{effective_bs}-lr{lr}-{timestamp}`
  - `uninavid-baseline-scratch-{epochs}ep-data{ver}-bs{effective_bs}-lr{lr}-{timestamp}`

## Eval Queue Path Convention

评测队列文件统一放在：

- `runtime/eval_queue/eval_todo.txt`
- `runtime/eval_queue/eval_done.txt`
- `runtime/eval_queue/eval_failed_todo.txt`

注意：不再使用旧路径 `src/swiftvln/scripts/eval/*.txt`。

### Eval Watchdog 异步回调机制（Updated: 2026-03-18）

评测默认使用 **tmux + watchdog** 异步模式，多服务器并发安全：

- 评测在 tmux session 中运行（命名：`eval_<short_desc>_<HHMMSS>`）
- `eval_watchdog.sh` 后台监控 tmux session，完成/失败时通过 `codex exec resume` 回调
- Per-host 完成状态：`runtime/eval_queue/eval_queue_last_run_<hostname>.json`
- Per-run 独立目录：`runtime/eval_queue/runs/<hostname>_<session_name>/`
  - `watchdog_result.json`、`watchdog.log`、`codex_response.txt`、`eval_queue_status.json`
- 自动清理：watchdog 启动时默认清理 7 天前的旧 run 目录（`--cleanup-days`）

Watchdog 启动方式（Codex 在启动评测后自动注册）：

```bash
nohup bash src/swiftvln/scripts/eval/eval_watchdog.sh \
  --tmux-session <session_name> \
  --codex-session <codex_uuid> \
  --eval-log <log_path> \
  --cleanup-days 7 &
```

### Train Watchdog 异步回调机制（Updated: 2026-03-18）

训练同样使用 **tmux + watchdog** 事件驱动模式，支持多服务器并行启动：

- 训练在 tmux session 中运行（命名：`train_<short_desc>_<HHMMSS>`）
- `train_watchdog.sh` 后台监控，通过 `train_events.log` 消费实验事件
- 事件驱动回调（非轮询）：
  - 实验失败 → `codex exec resume` 让 Codex 分析修复
  - 全部完成 → `codex exec`（新 session）按 eval skill 启动评测
  - 进程崩溃 → `codex exec resume` 诊断恢复
- Per-host 完成状态：`runtime/train_queue/train_queue_last_run_<hostname>.json`
- Per-run 独立目录：`runtime/train_queue/runs/<hostname>_<session>/`
- `train_queue.sh` 通过 `_emit_train_event()` 写入事件日志

```bash
nohup bash src/swiftvln/scripts/train/train_watchdog.sh \
  --tmux-session <session_name> \
  --codex-session <codex_uuid> \
  --train-log <log_path> \
  --on-all-done eval &
```

## Current SatNav Dataset Defaults

- Dataset root: `/mnt/data3/jiangjiajun/dataset/satnav_datasets`
- 当前常用版本：`ver_260327`
- Eval episodes (val_seen):
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260327/episodes/eval/val_seen/all_episodes.json`
- Eval episodes (val_unseen):
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260327/episodes/eval/val_unseen/all_episodes.json`
- **注意**：`episodes/eval/` 下只有 `val_seen/` 和 `val_unseen/` 子目录，不再有顶层扁平文件
- QA JSONL:
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260327/data/qa_swift.jsonl`
- Trajectory data:
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260327/trajectory_data`
- Scene maps:
  - active: `/mnt/data3/jiangjiajun/dataset/satnav_datasets/scenes`
  - backup(old): `/mnt/data3/jiangjiajun/dataset/satnav_datasets/old_scenes`

### SatNav Data Processing Convention (Updated: 2026-03-27)

- 数据处理默认会先执行 trajectory type 标准化：
  - 删除不完整城市目录 `Venezia`
  - 将 `highway / multiway / multway / waterway` 统一映射为 `trajectory_type = Road`
  - 同时保留细分类到顶层字段 `trajectory_subtype`，规范值为 `Highway / Multiway / Waterway`
- 标准化脚本：
  `src/swiftvln/scripts/data_process/normalize_trajectory_types.py`
- 默认入口 `src/swiftvln/scripts/data_process/run_all.py` 会先执行标准化，再生成 `episodes` 与 `qa_swift.jsonl`
- 单独执行 `src/swiftvln/scripts/data_process/process_episodes.py` 时，也会自动先做同样的标准化
- trajectory 生成默认并发（`generate_parallel.py`）为：
  `min(num_scenes, cpu_count//4, 72)`（2026-03-27 更新，原上限 24）
- `episodes/train/*.json` 与 `episodes/eval/*.json` 输出会保留 `trajectory_subtype` 字段
- 当前默认城市划分（0327 起）：
  - eval: `LosAngeles-1`, `Rome-1`, `NewYork-1`, `Auckland-1`, `Orlando-1`, `Rotterdam-1`
  - train: 其余全部城市（含 `Amsterdam-1`, `Dube-1`）
- Eval 城市按 seen/unseen 自动分类（0319 起）：
  - **val_seen**：eval 城市的基础名（如 `LosAngeles`）在 train 中有任意 TIF → 当前为 `LosAngeles-1`, `Rome-1`, `NewYork-1`
  - **val_unseen**：eval 城市的基础名完全不出现于 train → 当前为 `Auckland-1`, `Orlando-1`, `Rotterdam-1`
  - `episodes/eval/val_seen/` 和 `episodes/eval/val_unseen/` 在每次 `process_episodes.py` 时自动生成
  - `episodes/eval/all_episodes.json` 继续保留（全量 eval，向后兼容）

## Runtime/Infra Conventions

### Server Settings

三台服务器共享同一挂载工作区，所有文件操作（脚本、队列、输出）在任意服务器上本地可见。

| Server | Host | Access | GPUs | Notes |
|---|---|---|---|---|
| **98** | localhost | 直接执行 | 8× | 本机，无需 SSH |
| **73** | `10.246.152.73` | `ssh 10.246.152.73` | 8× | 远程 SSH |
| **17** | `10.246.132.17` | `ssh 10.246.132.17` 后进入 Docker 容器 | 8× | 远程 SSH + Docker |

- 共享工作区挂载点：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN`
- **SSH 仅用于**：远端 GPU/进程状态检查、远程 tmux 启动/停止。
- **文件操作**（脚本、队列写入、输出读写）：始终是本地操作，无需 SSH。
- server 17 Docker 容器名查询：`ssh 10.246.132.17 "docker ps"`

### Conda Environments

| Env | Purpose |
|---|---|
| `swift-vln-train` | SwiftVLN 主线训练（OverlapVLN / StreamVLN / CompressVLN） |
| `swift-vln-eval` | SwiftVLN 主线评测 |
| `streamvln-baseline` | baseline/streamvln 训练与评测（独立环境） |
| `uninavid-baseline` | baseline/uninavid 训练与评测（独立环境） |
| `navila-baseline` | baseline/navila 训练与评测（VILA + LLaMA-3-8B，torch 2.3.0+cu121，flash-attn 2.5.8） |

Conda 初始化命令（所有服务器统一）：
```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
```

### train_queue.sh 非交互模式（Updated: 2026-03-18）

`train_queue.sh` 支持通过环境变量 `TRAIN_EXPERIMENTS_FILE` 跳过交互式向导：

```bash
TRAIN_EXPERIMENTS_FILE='/path/to/experiments.sh' bash src/swiftvln/scripts/train/train_queue.sh
```

该文件需 source 可读，至少定义：
- `EXPERIMENTS` 数组（格式：`model|config|changes|ds_names|ds_paths|stage2_path|qa_ratio`）
- `TRAIN_STAGE`（`stage1` 或 `stage2`）
- `ENV_TYPE`（`satnav` 或 `habitat`）

`train_queue.sh` 的 SwanLab 约定（Updated: 2026-03-20）：
- 交互式与非交互式均**强制启用** SwanLab
- 默认 `SWANLAB_PROJECT=SatNav`
- 非交互配置文件里若写 `USE_SWANLAB=false` 会被忽略；如需自定义只改 `SWANLAB_PROJECT`

由 `orchestrate-plan` skill 在运行时通过 Write tool 生成，放在 `runtime/plans/generated/` 下。

### Offline Model Convention (Updated: 2026-03-17)

- 后续 VLN 训练任务默认使用**离线本地模型**，不依赖在线下载（避免 DNS/外网波动导致训练失败）。
- OverlapVLN/StreamVLN/CompressVLN 的 stage1 基座模型默认路径：
  `/mnt/data1/home/jiangjiajun/.cache/modelscope/models/Qwen/Qwen2___5-VL-3B-Instruct`
- 启动训练前必须先检查该路径存在且非空；若缺失，先修复模型路径/缓存，再启动训练。
- 若脚本默认值仍是 `Qwen/Qwen2.5-VL-3B-Instruct`（在线 ID），运行时需显式覆盖为上述本地绝对路径。

### tmux Session Naming Convention

所有长时间运行的任务（训练/评测）必须在 tmux 中启动：

```
train_<short_desc>_<HHMMSS>   # 例: train_baseline_streamvln_143025
eval_<short_desc>_<HHMMSS>    # 例: eval_streamvln_baseline_150200
```

## Smoke Test Convention

- 使用 skill：`.codex/skills/swiftvln-smoke-test/SKILL.md`
- 原则：小规模、可复现、不可破坏。
- 默认范围：SatNav-only。
- 训练 smoke 仅使用多卡多 GPU（不做单卡 smoke）。
- smoke 覆盖模型：`streamvln baseline` 与 `overlapvln baseline`，两者都要 train+eval。
- 必须保存并验证 checkpoint 可用于 eval。
- smoke 完成后需清理本次测试产物（仅清理 smoke 产物，不影响历史正式结果）。
- 必须避免：
  - 不必要的模型/数据重复下载
  - 全量长训练代替 smoke
  - 未经用户确认改动生产默认配置

### Uni-NaVid Baseline

- 训练/评测约定、smoke 参数、正式训练默认值：详见 `baseline/uninavid/doc/train_eval_conventions.md`
- `baseline/uninavid/scripts/train_satnav.sh` 已对齐 streamvln 风格：默认结构化 `EXP_NAME`，`USE_SWANLAB` 默认关闭，可按需开启
- `baseline/uninavid/scripts/eval_satnav.sh` 的 NFS checkpoint cache 逻辑（`maybe_cache_checkpoint()`）已修复：
  - 仅将最终 checkpoint 路径写到 stdout
  - cache 命中/rsync 进度等日志统一写到 stderr
  - 避免 `by_name` 模式下 `CHECKPOINT_DIR=$(...)` 被日志污染，导致 `--model_path` 变成多行字符串
- Uni-NaVid SatNav eval 约定（Updated: 2026-04-01）：
  - `baseline/uninavid/src/eval_satnav.py` 默认使用确定性解码（`do_sample=False`, `temperature=0.0`），确保基线评测可复现
  - 动作解析按 prompt 语义最多执行 4 个动作词（`forward/left/right/stop`）
  - 多卡 + resume 汇总改为在 rank0 从 `result.jsonl` 去重汇总（按 `episode_id` 最后写入覆盖），避免历史结果被各 rank 重复计入
- DeepSpeed ZeRO-2 NaN 问题根因：详见 `baseline/uninavid/doc/deepspeed_zero2_nan_analysis.md`

## Dependency Note

- SwiftVLN 已独立，但运行时仍可能依赖外部安装的 `swift` 包。
- 未经用户明确要求，不修改外部仓库；优先在 SwiftVLN 内完成适配。

## Commit Style

- 使用 conventional commit：`feat/fix/refactor/docs/test/perf/chore`
- commit message 优先简洁中文
- 保持原子提交，避免混入无关改动

## Webhook

- **Codex Webhook（企业微信机器人）**：
  `https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=48e434da-fb2d-453c-a180-c4041b4c7f1e`
- 凡需要发送 Codex webhook 通知时，使用上述地址（POST JSON，`msgtype: text`）。

## Operating Rules for Codex

- 优先使用现有脚本，不拼装临时一次性流程。
- 高风险步骤（远程/资源密集）需先说明预期和阻塞点。
- 除非用户明确要求，不创建 commit。
- 不回滚用户已有的无关改动。
- 默认失败策略：单项失败可继续后续队列项，并记录失败原因。
