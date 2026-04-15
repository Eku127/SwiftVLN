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
- 训练队列端口/重试修复（Updated: 2026-04-15）：
  - `src/swiftvln/scripts/train/train_queue.sh`
  - 串行训练在每次 attempt 启动前会先检查临时训练脚本中的 `MASTER_PORT` 是否可用；
    若端口已被占用，会在启动前直接改写为本机空闲端口，避免 `torchrun`
    在 rendezvous 阶段直接因 `EADDRINUSE` 失败
  - 训练执行现在按 `bash "$temp_script" | tee "$run_log_file"` 的真实
    `PIPESTATUS[0]` 判断成功/失败，不再被 `tee` 的返回码掩盖
  - 因此 `address already in use` 这类错误现在可以稳定进入 auto-fix 重试链路
- 训练 watchdog：`src/swiftvln/scripts/train/train_watchdog.sh`
- GPU 健康监控（Updated: 2026-04-07）：
  - 单机监控脚本：`src/swiftvln/scripts/monitor/gpu_health_monitor.sh`
  - 双机启动脚本：`src/swiftvln/scripts/monitor/start_gpu_health_monitors.sh`
  - 当前约定：在 `98` 与 `73` 各常驻一个本地监控进程，分别监控本机 GPU
  - 默认告警项：
    - `nvidia-smi` 查询失败
    - 可见 GPU 数量低于期望值
    - 基线 UUID 丢失（用于检测“下卡”）
    - `temperature.gpu >= 85C`
  - 默认 webhook：使用 Codex Webhook（企业微信机器人）
  - 默认 tmux session 名：
    - `gpu_health_98`
    - `gpu_health_73`
- OverlapVLN 单机训练 GPU 选择（Updated: 2026-04-07）：
  - `src/swiftvln/models/overlapvln/script/train/train_overlapvln_qwen2_5_vl.sh`
  - 默认不再硬编码 `0-7`，而是自动使用当前环境里**全部可见 GPU**
  - 可选覆盖：
    - `TRAIN_NUM_GPUS=<N>`：从当前可见 GPU 集合中取前 N 张
    - `TRAIN_CUDA_DEVICES=<csv>`：显式指定 GPU 列表，例如 `0,1,3,5`
    - `TRAIN_DRY_RUN=true`：仅做配置与 GPU 解析检查，不实际启动 `torchrun`
  - `src/swiftvln/scripts/train/train_queue.sh` 会把以上三个变量透传给单次训练脚本
- SwanLab 直连默认（Updated: 2026-04-08）：
  - `src/swiftvln/scripts/train/train_queue.sh`
  - `src/swiftvln/models/{overlapvln,streamvln,compressvln,monovln,uninavid}/script/train/*.sh`
  - 当 `USE_SWANLAB=true` 时，训练脚本默认会清理 `http_proxy/https_proxy/HTTP_PROXY/HTTPS_PROXY/all_proxy/ALL_PROXY`
  - 目的：避免误继承本地 `127.0.0.1:7890` 一类代理，导致 SwanLab 登录失败
  - 如需保留代理，可显式设置 `SWANLAB_DIRECT_NETWORK=false`
  - `train_queue.sh` 现在也会把全局 `QA_DATASET` 显式写入临时训练脚本，避免 `qa*` 实验回退到模型脚本内的旧默认 QA 路径
- OverlapVLN 训练恢复支持（Updated: 2026-04-08）：
  - `src/swiftvln/models/overlapvln/script/train/train_overlapvln_qwen2_5_vl.sh`
  - `src/swiftvln/scripts/train/train_queue.sh`
  - 单次训练脚本新增：
    - `RESUME_FROM_CHECKPOINT=<abs_path>`：传给 ms-swift 的 `--resume_from_checkpoint`
    - `RESUME_ONLY_MODEL=true|false`：可选，仅恢复模型权重
    - `OUTPUT_DIR_OVERRIDE=<path>`：覆盖实验根输出目录，再由 ms-swift 在其下生成新的 `v*/` 子目录
  - `train_queue.sh` 现在会把以上三个变量透传到临时训练脚本，适用于“从旧 checkpoint 真恢复后继续跑队列”的场景
  - 若要延续原实验命名并避免新建不同根目录，resume 时应同时设置 `RESUME_FROM_CHECKPOINT` 与 `OUTPUT_DIR_OVERRIDE`
- 评测单模型：`src/swiftvln/scripts/eval/eval_by_name.sh`
- 评测队列：`src/swiftvln/scripts/eval/eval_queue.sh`
- 评测入队：`src/swiftvln/scripts/eval/enqueue_eval.sh`
- 评测 worker：`src/swiftvln/scripts/eval/start_eval_worker.sh`
- 评测 monitor：`src/swiftvln/scripts/eval/start_eval_monitor.sh`
- 评测 watchdog：`src/swiftvln/scripts/eval/eval_watchdog.sh`
- 数据处理：`src/swiftvln/scripts/data_process/*.py`
- 数据集 merge：`src/swiftvln/scripts/data_process/merge_satnav_data.py`
- 数据同步：`src/swiftvln/scripts/data_sync/*.sh`
- SatNav merge skill：`.codex/skills/merge-satnav-data/SKILL.md`
- 主线评测 summary 重加权指标（Updated: 2026-04-14）：
  - 公共汇总逻辑：`src/swiftvln/common/eval/runner.py`
  - 权重/统计 helper：`src/swiftvln/common/eval/reporting.py`
  - 当 `env_type=satnav` 且存在 `by_trajectory_type` 时，`evaluation_summary.json`
    现在额外写出 `weighted_by_seen_unseen_distribution`
  - 该字段使用 `satnav_task.yaml` 中 `DATA_PATH` 对应的 `val_seen + val_unseen`
    episode 合并分布，输出统一加权后的：
    `success_rate / mean_spl / oracle_success / navigation_error / avg_steps`
  - 目的：在不修改实际评测 episode 的前提下，降低 `seen/unseen`
    任务类型配比差异对总指标解释的干扰

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
  - `evaluation_summary.json`（Updated: 2026-04-14）在存在 `by_trajectory_type` 时会额外写出
    `weighted_by_seen_unseen_distribution`
  - 该字段使用 `src/swiftvln/configs/satnav_task.yaml` 对应的
    `val_seen + val_unseen` 合并任务类型分布做统一加权，输出
    `SR / SPL / OS / NE / avg_steps`

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
- 动作输出格式现已抽象为共享组件（Updated: 2026-04-07）：
  - 共享文件：`baseline/navila/src/action_formats.py`
  - `baseline/navila/src/eval_satnav.py` 与 `baseline/navila/src/diagnose_train_vs_eval.py`
    支持 `--action_format {sentence,compact}`，默认已切到 `compact`
  - 环境变量：`SATNAV_ACTION_FORMAT`
  - `compact` 模式使用单词级目标与解析：`stop / forward / left / right`
  - `sentence` 现为 legacy/显式回退选项；若需旧句式监督，需显式传 `SATNAV_ACTION_FORMAT=sentence`
  - 适用场景：当句式监督出现“loss 很低但动作塌缩”时，默认优先走 `compact`
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
  - 评测收尾同步（Updated: 2026-04-13）改为基于结果目录下 `_rank_sync/<run_id>/rank_<n>.json`
    的文件标记等待，不再依赖末尾 `dist.barrier()`
  - `baseline/navila/scripts/eval_satnav.sh` 现会为每个 split 传唯一 `--run_id`
  - 目的：保留 8 卡断点续跑能力，同时规避收尾阶段 `Socket Timeout` / NCCL barrier 崩溃
  - `evaluation_summary.json`（Updated: 2026-04-14）在存在 `by_trajectory_type` 时会额外写出
    `weighted_by_seen_unseen_distribution`
  - 该字段使用 `src/swiftvln/configs/satnav_task.yaml` 对应的
    `val_seen + val_unseen` 合并任务类型分布做统一加权，输出
    `SR / SPL / OS / NE / avg_steps`
- 生成停止条件与 dtype 稳定性修复（Updated: 2026-04-07）：
  - `baseline/navila/src/eval_satnav.py` 不再直接使用上游 `KeywordsStoppingCriteria`
  - 当前改为本地 `SafeKeywordsStoppingCriteria`，只匹配**生成后缀**，避免 Llama 3 prompt 内自带 `<|eot_id|>` 时在 `0 token` 阶段被误判为 stop
  - 评测新增 `--debug_generation` / `--debug_generation_limit`，可打印 `stop_str`、prompt 命中、首步 token/top scores、是否立即终止
  - 评测新增 `--eval_dtype {auto,float16,bfloat16,float32}`
  - `auto` 默认在支持时优先使用 `bfloat16`，用于规避部分 NaVILA checkpoint 在 `float16` 视觉前向下首步 logits 变成 `NaN`、输出 `!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!` 的问题
- NaVILA eval checkpoint 缓存约定（Updated: 2026-04-07）：
  - `baseline/navila/scripts/eval_satnav.sh`
  - `LOCAL_CACHE_DIR=""` 现在会**真正禁用**本地缓存，而不是回退到默认缓存目录
  - 若 checkpoint 位于 NFS 且容器内缺少 `rsync`，脚本会直接回退到源 checkpoint，不再写出空缓存并伪造 `.cache_complete`

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
  - 默认 `MAX_STEPS=60000`
  - 若显式传空值 `MAX_STEPS=`，则**不传 `--max_steps`**，用于全量训练
  - 默认 `SAVE_STEPS=20000`
  - 若显式传空值 `SAVE_STEPS=`，则按真实总 step 数自动推导：
    `SAVE_STEPS = ceil(total_steps / SAVE_COUNT_TARGET)`，默认 `SAVE_COUNT_TARGET=4`
  - `SAVE_TOTAL_LIMIT` 默认 `1`（保留最新一个 `checkpoint-*`；训练结束仍会在 `output/navila-baseline/<EXP_NAME>/` 根目录保存最终模型）
- NaVILA SatNav train 默认采样策略（head+stop+turn-protect+fwd-stride+stop-oversample，Updated: 2026-04-02）：
  - `SATNAV_HEAD_KEEP=7`：保留每条轨迹前 7 步（帧数 < num_video_frames=8 的独特分布区间，全部保留）
  - stop 步（每 episode 末尾）：**全部保留**
  - 中间区间 turn 步（left/right）：**全部保留**（决策关键少数类，不做 stride）
  - `SATNAV_SAMPLE_STRIDE=7`：中间区间连续 forward run 每 7 步取 1 步（遇 turn 重置计数）
  - `SATNAV_STOP_REPEAT=4`：对 stop 样本重复 4 次，增强 stop 监督，但不做完全类均衡
  - 效果：默认采样后约 **286.9 万**条样本；其中 `forward≈128.7 万`、`left≈62.6 万`、`right≈56.6 万`、`stop≈39.0 万`
  - 若需完整全量训练（不做采样），显式传空值：`SATNAV_HEAD_KEEP= SATNAV_SAMPLE_STRIDE=`，并可选 `SATNAV_STOP_REPEAT=1`
  - 默认自动实验名会带 sample 标识，例如：`sample-hk7-fs7-stopx4`
- smoke / 调试时可额外叠加：
  - `SATNAV_MAX_EPISODES`
  - `SATNAV_MAX_SAMPLES`
  - `SATNAV_SAMPLE_RATIO`
  以上三个变量默认为空，**正式全量训练不要设置**
- NaVILA SatNav action supervision 约定（Updated: 2026-04-07）：
  - `baseline/navila/src/dataset/satnav_dataset.py` 支持 `SATNAV_ACTION_FORMAT={sentence,compact}`
  - 默认值已切到 `compact`
  - `sentence` 为历史格式：`The next action is stop / move forward 10 meters / turn left / turn right`
  - `compact` 为单词级格式：`stop / forward / left / right`
  - `compact` 训练时会同步切换 prompt 文案为“reply with exactly one word”
  - `baseline/navila/scripts/train_satnav.sh` 在非默认动作格式下会自动把实验名后缀标成 `-act<format>`；当前仅 legacy `sentence` 会自动追加 `-actsentence`

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

UniNaVid SatNav eval 约定（Updated: 2026-04-01）：

- `baseline/uninavid/src/eval_satnav.py` 的断点续跑与离线汇总唯一键使用 `scene_id + episode_id`
- 避免仅按 `episode_id` 去重时，`val_seen` 中跨 scene 重复 episode id 导致的误跳过与汇总失真
- 多卡汇总为 rank0 在 `dist.barrier()` 后从 `result.jsonl` 按联合键离线去重并写 `evaluation_summary.json`
- 分布式收尾使用带 `device_ids=[local_rank]` 的 barrier，并在 `main()` 退出时显式 `destroy_process_group()`，避免 NCCL barrier / process group 清理 warning
- 0317（`data260317`）历史 UniNaVid baseline 产物归档到 legacy（Updated: 2026-04-03）：
  - 模型目录：`output/uninavid-baseline/legacy/<EXP_NAME>/`
  - 结果目录：`results/uninavid-baseline/legacy/<EXP_NAME>/`
  - `baseline/uninavid/scripts/eval_satnav.sh` 按实验名评测时会先查当前路径，再 fallback 到 `output/uninavid-baseline/legacy/<EXP_NAME>/`
- UniNaVid 多卡 eval 收尾同步（Updated: 2026-04-03）：
  - `baseline/uninavid/src/eval_satnav.py` 不再依赖收尾 NCCL barrier 来等待所有 rank 完成写盘
  - 改为每个 rank 在 `output_path/.dist_sync/rank_<rank>.done.json` 写完成标记，rank0 轮询标记后再从 `result.jsonl` 离线汇总
  - 用于规避某些长尾 rank / TCPStore 超时导致的 barrier 失败，把几乎完成的 eval 整体打断
- UniNaVid eval 启动前置检查（Updated: 2026-04-04）：
  - `baseline/uninavid/scripts/eval_satnav.sh` 会在每个 split 启动前检查当前可见 GPU 列表
  - 若请求卡数大于当前可见卡数，会直接报错并打印 `CUDA_VISIBLE_DEVICES` 与检测到的 GPU 列表，不再等到 `torchrun` 内部以 `invalid device ordinal` 失败
  - 多 split 顺序评测时，脚本会在每次 `torchrun/python` 启动前清理 `RANK/WORLD_SIZE/LOCAL_RANK/MASTER_*` 等分布式环境变量，并显式导出当前检测到的 `CUDA_VISIBLE_DEVICES`

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
- 当前常用版本：`ver_260404`
- Eval episodes (val_seen):
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260404/episodes/eval/val_seen/all_episodes.json`
- Eval episodes (val_unseen):
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260404/episodes/eval/val_unseen/all_episodes.json`
- **注意**：`episodes/eval/` 下只有 `val_seen/` 和 `val_unseen/` 子目录，不再有顶层扁平文件
- QA JSONL:
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260404/data/qa_swift.jsonl`
- Trajectory data:
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260404/trajectory_data`
- Scene maps:
  - active: `/mnt/data3/jiangjiajun/dataset/satnav_datasets/scenes`
  - backup(old): `/mnt/data3/jiangjiajun/dataset/satnav_datasets/old_scenes`

### SatNav ver_260403 Snapshot (Updated: 2026-04-03)

- `ver_260403` 为 boundary-only 数据集：
  - 总 episodes：`8820`
  - train：`7476`
  - val_seen：`426`
  - val_unseen：`918`
- 所有 episode 的 `trajectory_type` 均为 `Boundary`
- `trajectory_subtype` 在原始 `data/*/VLN_episodes.json` 与处理后的 `episodes/*.json` 中都保留
  - 当前 observed subtype：`arc`、`overlap`
- `ver_260403/data/` 下无 `qa.json`
  - `run_all.py` 仍会生成空文件 `data/qa_swift.jsonl`（0 行），属预期行为
- `ver_260403/trajectory_data` 生成结果：
  - `annotations.json`：`7476`
  - `summary.json`：`7476` 行
  - `images/` episode 目录：`7476`

### SatNav Dataset Merge Convention (Updated: 2026-04-03)

- 合并入口脚本：
  `src/swiftvln/scripts/data_process/merge_satnav_data.py`
- 默认流程：
  - 先 `--analyze-only` 做 preflight
  - identical overlap 去重
  - conflicting overlap 不丢弃，给 secondary 分配新的 `episode_id`
  - 同步重写 secondary `trajectory_data/summary.json`、`annotations.json`、`images/<scene>_satnav_<id>`
  - 最后再跑 `process_episodes.py` 重建 merged `episodes/`
- 输出工件：
  - `<version>/merge_manifest.json`
  - `<version>/episode_id_remap.jsonl`
- 默认校验：
  - merged `scene_id + episode_id` 唯一
  - train / val_seen / val_unseen 相比 primary 有增长（除非显式允许无增长）
  - `trajectory_data/summary.json` / `annotations.json` / `images/` 三者严格对齐

### SatNav ver_260404 Snapshot (Updated: 2026-04-03)

- 来源：`ver_260327 + ver_260403`
- 生成方式：
  `python3 src/swiftvln/scripts/data_process/merge_satnav_data.py ver_260327 ver_260403 ver_260404`
- 关键结论：
  - `ver_260403` 与 `ver_260327` 在 `scene_id + episode_id` 上**完全重叠**
  - 但 payload 不同，属于 conflicting overlap，不是 identical duplicate
  - merge 时对 `ver_260403` 的 `8820` 条 episode 全部做了 secondary ID remap
- split 结果：
  - train：`105197`（相对 `ver_260327` `+7476`）
  - val_seen：`6338`（`+426`）
  - val_unseen：`8917`（`+918`）
- trajectory 结果：
  - `summary.json`：`104954`
  - `annotations.json`：`104954`
  - `images/` 目录：`104954`
  - 从 primary 源里额外清理了 `240` 个未被 merged summary 引用的旧 image 目录
- QA 结果：
  - `data/qa_swift.jsonl`：`273120`
  - 原因：`ver_260403` 无 `qa.json`，因此 merged QA 没有增长

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
  - 本地 cache 目录名现在包含 `父实验目录 + checkpoint 名 + 源路径 hash`
  - 避免不同实验都叫 `checkpoint-6000` 时复用同一个本地缓存，导致评测误读到错误权重
- Uni-NaVid SatNav eval 约定（Updated: 2026-04-01）：
  - `baseline/uninavid/src/eval_satnav.py` 默认使用确定性解码（`do_sample=False`, `temperature=0.0`），确保基线评测可复现
  - 动作解析按 prompt 语义最多执行 4 个动作词（`forward/left/right/stop`）
  - 多卡 + resume 汇总改为在 rank0 从 `result.jsonl` 去重汇总（按 `episode_id` 最后写入覆盖），避免历史结果被各 rank 重复计入
- DeepSpeed ZeRO-2 NaN 问题根因：详见 `baseline/uninavid/doc/deepspeed_zero2_nan_analysis.md`

## Dependency Note

- SwiftVLN 已独立，但运行时仍可能依赖外部安装的 `swift` 包。
- 未经用户明确要求，不修改外部仓库；优先在 SwiftVLN 内完成适配。

## S2R Stage-A (Updated: 2026-04-03)

新增 Stage-A sim-to-real 对齐包：

- 包目录：`src/swiftvln/s2r/`
- 目标：基于真实 `UAV ↔ Satellite` 配对图做视觉对齐，产出后续可接入 OverlapVLN 的 adapter

当前已实现的 Stage-A 入口：

- manifest 构建脚本：`src/swiftvln/s2r/scripts/build_manifest.py`
- 训练入口：`src/swiftvln/s2r/trainer.py`
- 评估入口：`src/swiftvln/s2r/eval.py`
- 训练包装脚本：`src/swiftvln/s2r/scripts/train_s2r.sh`

Stage-A 关键模块：

- `src/swiftvln/s2r/dataset.py`
- `src/swiftvln/s2r/model.py`
- `src/swiftvln/s2r/losses.py`
- `src/swiftvln/s2r/split.py`
- `src/swiftvln/s2r/arguments.py`

Stage-A 当前设计约定：

- 只做 `contrastive + global cosine distill`
- **不做 token distill**
- teacher 默认对齐到目标 OverlapVLN checkpoint 的 visual tower 输出空间
- `TeacherVisionTower` 必须使用 `Qwen2_5_VLForConditionalGeneration.from_pretrained(...)`
  加载 Qwen2.5-VL checkpoint；不要再走 `AutoModelForCausalLM`，否则会在
  `qwen2_5_vl` config 上报模型类型不识别
- 当前推荐 teacher：
  `output/overlapvln/overlapvln-satnav-stage1-3b-1ep-f32s4-overlap16-gtc-k512-noembed-data260317-bs64-lr2e-5-20260318-202149/v0-20260318-202212/checkpoint-3957`
- `SatDronePairDataset(max_samples=...)` 现在采用跨数据源 round-robin 限样
  （不是 manifest 头部截断），用于保证 smoke train/eval 在小样本下仍覆盖多数据源

Stage-A 数据约定：

- 数据根目录：`/mnt/data3/jiangjiajun/dataset/SatDronePair`
- 真实 manifest 默认输出：`runtime/s2r/manifests/manifest_v1.jsonl`
- 训练输出目录约定：`output/s2r/<EXP_NAME>/`

Stage-A split 约定：

- 四个数据源全部先视为候选训练池，不直接沿用原始 `train/test`
- 统一重建 `train/val`
- `denseuav`：按基础位置 ID 分组（同位置不同高度同 split）
- `gta`：按 `area_mode + satellite_img_name` 分组
- `sues`：按 `scene_id` 分组
- `uavvisloc`：按 `seq_id` 分组

Stage-A 测试文件：

- `tests/test_s2r_split_and_manifest.py`
- `tests/test_s2r_dataset.py`
- `tests/test_s2r_model_and_losses.py`

## OverlapVLN UAV Adapter Stage-B (Updated: 2026-04-07)

OverlapVLN 已接入 `uav_adapter` 的 Stage-B 最小链路：

- 在线 enhancement 模块：`src/swiftvln/common/embedding_enhancement/uav_adapter.py`
- pipeline 工厂：`src/swiftvln/common/embedding_enhancement/__init__.py`
- OverlapVLN 模型加载与本地/外部权重恢复：
  `src/swiftvln/models/overlapvln/model.py`
- OverlapVLN trainer 参数透传与 pipeline 重建：
  `src/swiftvln/models/overlapvln/trainer.py`
- OverlapVLN eval 参数透传：
  `src/swiftvln/models/overlapvln/eval.py`
- OverlapVLN 分布式评测脚本参数透传：
  `src/swiftvln/models/overlapvln/script/eval/eval_overlapvln_qwen2_5_vl_distributed.sh`
- 训练脚本参数透传：
  `src/swiftvln/models/overlapvln/script/train/train_overlapvln_qwen2_5_vl.sh`

Stage-B 当前参数约定：

- `use_uav_adapter: bool = False`
- `uav_adapter_path: str = ""`
- `uav_adapter_type: str = "transformer_v1"`
- `uav_adapter_apply_scope: str = "all_images"`

Stage-B 当前实现约定：

- `uav_adapter` 作为 `embed_enhance` pipeline 的一个插件，插入位置仍是
  `visual encoder -> embed_enhance -> history processor`
- 当前只支持 `uav_adapter_apply_scope=all_images`
- 若 `uav_adapter_path` 非空，则显式外部 Stage-A checkpoint 会覆盖本地 checkpoint 中的
  `embed_enhance.uav` 权重
- 支持直接传 Stage-A `.pt` 文件，或 Stage-A 输出目录；目录解析优先级：
  `best.pt -> latest.pt -> checkpoints/step_*.pt`
- 分布式评测脚本现已支持同名环境变量：
  `USE_UAV_ADAPTER` / `UAV_ADAPTER_PATH` / `UAV_ADAPTER_TYPE` / `UAV_ADAPTER_APPLY_SCOPE`

Stage-B smoke / regression 测试：

- Stage-B 单测：`tests/test_uav_adapter_enhancement.py`
- Stage-B loader smoke：
  `src/swiftvln/models/overlapvln/script/test/test_uav_adapter_strategy.py`
- 现有 pixel/pose loader smoke：
  `src/swiftvln/models/overlapvln/script/test/test_pixel_embed_strategy.py`
- 全模型导航 eval smoke 已通过（2026-04-07）：
  - 环境：`satnav`
  - 模式：`1 GPU / max_episodes=1 / val_seen`
  - 启用：`USE_UAV_ADAPTER=true`
  - 外部 Stage-A checkpoint：
    `output/s2r/smoke-large-multisrc-20260403-152012/best.pt`
  - 说明：验证了 `checkpoint 加载 -> 外部 UAV adapter 注入 -> SatNav 环境 rollout -> summary 写出`

Stage-A 当前验证状态（2026-04-03）：

- 单测：`python -m unittest tests.test_s2r_split_and_manifest tests.test_s2r_dataset tests.test_s2r_model_and_losses`
  已通过（`8 tests`）
- 最小 smoke train 已通过：
  - 单卡 `cuda:0`
  - `max_steps=2`
  - `max_train_samples=8`
  - `max_val_samples=4`
  - 成功产出 `best.pt`、`latest.pt`、`checkpoints/step_*.pt`、`metrics.jsonl`
- 最小独立 smoke eval 已通过：
  - 从 `best.pt` 重新加载 adapter + projection + teacher 后可完成 `val` 评估
  - 一次已验证产物目录：
    `output/s2r/smoke-20260403-145213`
- 较大 smoke train 已通过：
  - `2 GPU` 分布式（`CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2`）
  - `batch_size=4`
  - `max_train_samples=256`
  - `max_val_samples=64`
  - `max_steps=32`
  - 中间与结束评估均成功落盘
  - 最优 `u2s_r@1 = 0.265625`，对应 `step=16`
  - 一次已验证产物目录：
    `output/s2r/smoke-large-20260403-151621`
- 多数据源较大 smoke train+eval 已通过：
  - `train` 限样分布：`64 x 4`（`denseuav/gta/sues/uavvisloc`）
  - `val` 限样分布：`16 x 4`
  - `2 GPU` 分布式，`batch_size=4`，`max_steps=32`
  - 独立 `best.pt` 重载评估已通过
  - 最优 `u2s_r@1 = 0.46875`，对应 `step=32`
  - 一次已验证产物目录：
    `output/s2r/smoke-large-multisrc-20260403-152012`

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
