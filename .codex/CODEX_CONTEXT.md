# Codex Repo Context (SwiftVLN)

本文件是 SwiftVLN 仓库内的默认工作上下文。
在本仓库执行任务时，优先以本文件为准。

## Repository Scope

- Repo: `SwiftVLN`
- Root: `/mnt/data1/home/jiangjiajun/workspace/SwiftVLN-refactor`
- 主要代码域：`src/swiftvln/*`
- 当前主线模型：`swiftvln`（未明确指定时默认按 swiftvln 处理）
- Rename status（Updated: 2026-05-11）：
  - 主线 active 标识已切到 `swiftvln` / `SwiftVLN` / `SWIFTVLN`
  - 新训练输出默认写入 `output/swiftvln/<swiftvln-exp-name>`
  - 新训练实验名不再包含 `stage1`、SatNav `data...` 版本标签、`-notailadj` 默认行为标签；末尾运行标识只保留 `HHMMSS`
  - 旧本地输出已迁移：`output/overlapvln/*` 已改名并移动到 `output/swiftvln/*`
  - data4 archive legacy 模型目录已迁移：`/mnt/data4/jiangjiajun/archive/**/overlapvln*` 已改名为 `swiftvln*`
  - 主线按名评测只接受 `swiftvln-*`，默认结果路径为 `results/eval/swiftvln/<exp>/<split>/<timestamp>`
  - 算法参数中的 `overlap` / `NUM_OVERLAP` / `num_overlap` / `-overlap{N}` 仍按原语义保留

## Current Architecture Snapshot

SwiftVLN 已从 `ms-swift/examples/vln` 迁移为独立仓库，核心结构如下：

- 包根：`src/swiftvln`
- 模型目录：`src/swiftvln/model`（主线 swiftvln 唯一实现）
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
  - 2026-04-17 起脚本末尾显式 `exit $?`，避免长跑队列执行期间若脚本文件被原地改写，
    在收尾阶段继续解释被修改后的尾部内容，导致异常“重入”重跑
- 链式启动脚本进程检测修复（Updated: 2026-04-29）：
  - `runtime/train_queue/launchers/launch_overlap0418_notail_after_navila.sh`
  - `runtime/tmp/navila0418_eval_after_98_and_17.sh`
  - NaVILA train/eval 等待逻辑中的 `pgrep -f` 现使用 bracketed regex，避免匹配到
    `pgrep` 自身命令行后误判训练仍在运行，导致后续 overlap 队列或 98 eval 无法启动。
- 训练 watchdog：`src/swiftvln/scripts/train/train_watchdog.sh`
- StreamVLN baseline 训练数据与标签命名（Updated: 2026-06-29）：
  - 脚本：`baseline/streamvln/scripts/train_satnav.sh`
  - 当前默认数据集标识为 `SATNAV_DATASET=SatNav-v0.1`，训练默认读取：
    `/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/trajectory_data`
  - `SatNav-v0.1/trajectory_data` 已在本机完成生产并同步到 73/17 相同目录：
    `/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/trajectory_data`
    （Updated: 2026-06-29；annotations/summary/episode cache 均为 105164 条）
  - `SATNAV_VERSION` 仅作为旧启动脚本兼容 alias，不再作为新文档主变量
  - `EXP_NAME` 中的 `data...` 段来自 `SATNAV_DATASET`，会清理非法路径字符；例如：
    `SatNav-v0.1 -> dataSatNav-v0.1`
  - 训练脚本默认优先使用：
    `/mnt/data1/home/jiangjiajun/miniconda3/envs/streamvln-baseline/bin/python`
    并把该 env 的 `bin` prepend 到 `PATH`，保证 DeepSpeed JIT 能找到 conda 内的
    `ninja`
  - 默认 `STREAMVLN_OFFLINE=true`，shell 和 Python 入口都会设置
    `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`，避免 import 阶段访问 HF 网络
  - smoke 专用限流变量：`SATNAV_MAX_EPISODES` / `SATNAV_MAX_SAMPLES`
- SatNav 数据集存储清理（Updated: 2026-06-29）：
  - 已删除重复的 `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260418/check_trajectory_data`
  - `ver_260404` 已从 data3 迁移到：
    `/mnt/data4/jiangjiajun/dataset/satnav_datasets/ver_260404`
  - 为保持旧路径兼容，data3 保留 symlink：
    `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260404 -> /mnt/data4/jiangjiajun/dataset/satnav_datasets/ver_260404`
  - 0404 数据 dry-run 校验差异为 0，关键文件大小与 `trajectory_data/images` episode 目录数一致
- Baseline 0418 model zoo（Updated: 2026-05-26）：
  - `baseline_0418_seen_unseen_all.csv` 中本仓库内可定位的收口 baseline 模型已集中移动到：
    `output/model_zoo/baseline/`
  - 已移动模型包括：StreamVLN scratch/continue、OpenFly scratch/continue、NaVILA scratch/continue、UniNaVid scratch/continue
  - Seq2Seq、CMA、OverlapVLN/SwiftVLN 对应模型目录本次未在本仓库输出路径下定位到；仅结果目录或 raw data 仍保留在报告中
  - StreamVLN model zoo 目录名已精简为 eval 所需窗口参数加训练摘要：
    `streamvln-baseline-continue-1ep-f32h8s4-lr2e-5`、
    `streamvln-baseline-scratch-1ep-f32h8s4-lr2e-5`
  - StreamVLN 另有 Hugging Face upload-ready 精简副本：
    `streamvln-satnav-continue-1ep-f32h8s4-lr2e-5`、
    `streamvln-satnav-scratch-1ep-f32h8s4-lr2e-5`；二者与对应
    `streamvln-baseline-*` 权重一致，但只保留 eval/inference 所需文件
- Baseline eval by name with model root（Updated: 2026-06-29）：
  - 脚本：`baseline/streamvln/scripts/eval_satnav.sh`、`baseline/navila/scripts/eval_satnav.sh`、`baseline/uninavid/scripts/eval_satnav.sh`、`baseline/openfly/scripts/eval_satnav.sh`
  - StreamVLN / NaVILA / UniNaVid / OpenFly eval 均已收敛为命名参数主路径；公开入口使用
    `--model_dir` + `--model_name`，并通过 `--gpus`、`--max_episodes` 等显式参数控制运行规模
  - 四个 baseline 的 eval 均不再支持位置参数、`--checkpoint_path`、`--split` 或 `--satnav_version`
  - by-name 默认模型根目录仍是各自 `output/<baseline>-baseline`；model zoo 和公开上传目录可用
    `--model_dir output/model_zoo/baseline` 或 `--model_dir output/model_zoo/baseline/HF_model`
  - StreamVLN eval 会直接把 `<model_dir>/<model_name>` 作为 Hugging Face 模型目录传给 `from_pretrained()`；该目录必须包含 `config.json` 与 safetensors/bin 权重，不再查找或加载 `checkpoint-*` 子目录
  - StreamVLN eval 会从模型 `config.json` 读取 `mm_vision_tower` / `vision_tower`，先搜索本地同名视觉塔目录（如 `baseline/streamvln/model/siglip-so400m-patch14-384`）；若本地不存在，则保留原始 Hugging Face id 交给 Transformers 使用 cache 或下载；也可用 `--vision_tower <path-or-hf-id>` 强制覆盖
  - StreamVLN eval 的 Python 解释器默认优先使用
    `/mnt/data1/home/jiangjiajun/miniconda3/envs/streamvln-baseline/bin/python`；
    可用 `PYTHON_BIN=/path/to/python` 覆盖。这样在 73 机这类没有 `python` 或系统 `python3`
    没有 torch 的环境中，`torchrun` fallback 仍会进入正确 conda 环境。
  - StreamVLN eval 不再从模型名中的 `data{version}` 解析 `SATNAV_VERSION`；默认读取 `baseline/streamvln/configs/satnav_task.yaml` 中的 `DATASET.SPLIT` / `DATA_PATH` / `SCENES_DIR`；`SPLIT: all` 默认跑 `val_seen` + `val_unseen`；`DATA_PATH` 必须填写 eval split 父目录（当前为 `SatNav-v0.1/episodes/eval`），脚本解析为 `<DATA_PATH>/<split>/all_episodes.json`
  - `baseline/streamvln/scripts/train_eval_satnav.sh` 已同步为新 by-name eval 调用；不再接受位置 split 参数，split 只由 `satnav_task.yaml` 控制
  - StreamVLN eval 仍会从模型名中的 `f{frames}h{history}s{future}` 解析窗口参数
  - NaVILA / UniNaVid / OpenFly 当前默认也使用 `SatNav-v0.1` 数据：
    `baseline/{navila,uninavid,openfly}/configs/satnav_task.yaml` 中 `DATASET.DATA_PATH`
    指向 `/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/episodes/eval`
  - 四个 baseline 的 `satnav_task.yaml` 不再携带 `SIMULATOR.AERIAL` / API key；
    eval 只保留任务、传感器、数据路径和 scenes 路径
  - OpenFly eval 会从模型名中的 `actcompact|actoriginal`、`hist<N>` 解析动作格式和 action history 长度；
    不再从模型名中的 `data<N>` 解析 eval 数据版本
  - `output/model_zoo/baseline/HF_model/` 已集中放置后续 Hugging Face upload-ready 模型目录
  - 2026-06-29 已完成四个 baseline 在 `SatNav-v0.1` 上的 8 卡 train/eval smoke：
    - eval smoke 在 98 机完成，HF upload-ready continue 模型，`--gpus 8 --max_episodes 2`
    - eval 汇总：`logs/baseline_satnav_v01_smoke/local98_eval_smoke_20260629_142339.csv`
    - UniNaVid 修复分布式 marker 后重跑通过：
      `logs/baseline_satnav_v01_smoke/local98_uninavid_retry_20260629_143332.status`
    - train smoke：NaVILA / UniNaVid / OpenFly 在 73 机通过：
      `logs/baseline_satnav_v01_smoke/train_smoke_73_20260629_143641.csv`
    - StreamVLN train smoke 在 98 机通过：
      `logs/baseline_satnav_v01_smoke/train_streamvln_local98_20260629_150840.status`
    - smoke 输出模型目录均已清理，只保留日志、CSV/status 与 evaluation summary 副本
- OpenFly scratch native checkpoint 本地 HF cache（Updated: 2026-04-20）：
  - 核心实现：`baseline/openfly/src/native_core/checkpoint_conversion.py`
  - 当前 `scratch` backend 不再让 8 个 rank 各自重复读取 `openvlaopenvla-7b-prismatic/checkpoints/*.pt`
  - 现改为：首个进入的 rank 在本机本地盘把 native `.pt` 单次转换为 HF `safetensors` shard，其他 rank 等待完成后直接从共享本地 cache 加载
  - 默认 cache 根目录：`/mnt/data4/jiangjiajun/openfly_native_hf_cache`
  - 关闭方式：`OPENFLY_NATIVE_HF_CACHE_DIR=off|false|none|0`
  - cache key 包含：native checkpoint 路径/大小/mtime、processor source、`grid_size`、`unnorm_key`
  - cache 完成标记：`.cache_complete`；并发互斥通过同目录下 `.<cache_name>.lock` 文件锁实现
  - 2026-04-20 同步在 `baseline/openfly/src/train_satnav.py` 增加了阶段日志：
    - `Building SatNavOpenFlyDataset...`
    - `Dataset ready: samples=...`
    - `Building training backend...`
    - `Backend ready: backend=...`
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
- SwiftVLN 单机训练 GPU 选择（Updated: 2026-04-07）：
  - `src/swiftvln/model/script/train/train_swiftvln_qwen2_5_vl.sh`
  - 默认不再硬编码 `0-7`，而是自动使用当前环境里**全部可见 GPU**
  - 可选覆盖：
    - `TRAIN_NUM_GPUS=<N>`：从当前可见 GPU 集合中取前 N 张
    - `TRAIN_CUDA_DEVICES=<csv>`：显式指定 GPU 列表，例如 `0,1,3,5`
    - `TRAIN_DRY_RUN=true`：仅做配置与 GPU 解析检查，不实际启动 `torchrun`
  - `src/swiftvln/scripts/train/train_queue.sh` 会把以上三个变量透传给单次训练脚本
- SwiftVLN 训练默认值与 smoke 隔离（Updated: 2026-04-17）：
  - `src/swiftvln/model/script/train/train_swiftvln_qwen2_5_vl.sh`
  - 当前仓库默认值按**正式训练**维护，不再使用 smoke 残留默认：
    - `MAX_SAMPLES=0`（显式表示全量）
    - `NUM_OVERLAP=0`（当前 baseline 默认）
    - `SAVE_STEPS=1000`
    - `SAVE_TOTAL_LIMIT=1`
  - overlap baseline 约定：
    - 当前 baseline 默认：`overlap=0`
    - 历史 baseline 默认：`overlap=16`
  - `train_queue.sh` 复制单次训练脚本时会继承这些默认值，因此若不显式覆盖，
    串行队列将按正式训练口径启动
  - smoke 测试必须在调用时显式覆盖：
    - `MAX_SAMPLES=16`
    - `SAVE_STEPS=1`
    - `SAVE_TOTAL_LIMIT=1`
    - 见 `.codex/skills/swiftvln-smoke-test/SKILL.md`
- SwiftVLN overlap 训练尾窗行为（Updated: 2026-05-12）：
  - `OVERLAP_TAIL_WINDOW_ADJUST` 控制参数已移除。
  - `NUM_OVERLAP>0` 时固定保持严格 `stride = num_frames - num_overlap`
    的窗口起点，短尾窗不再向后回挪；该行为不再写入实验名。
  - `NUM_OVERLAP=0` 保持原有尾窗覆盖逻辑不变，避免影响当前 baseline。
- SwiftVLN 评测窗口默认与 `overlap=0` 修复（Updated: 2026-04-27）：
  - 相关文件：
    - `src/swiftvln/scripts/eval/eval_by_name.sh`
    - `src/swiftvln/model/script/eval/eval_swiftvln_qwen2_5_vl_distributed.sh`
    - `src/swiftvln/model/eval.py`
    - `src/swiftvln/model/evaluator.py`
  - `eval_by_name.sh` 会从实验名中的 `f{num_frames}s{num_future_steps}` 与
    `overlap{num_overlap}` 解析并透传：
    - `NUM_FRAMES`
    - `NUM_FUTURE_STEPS`
    - `NUM_OVERLAP`
  - 底层 eval 默认 `NUM_OVERLAP` 已与当前 baseline 对齐为 `0`；
    直接调用 distributed eval 或 Python eval 时，不再回落到历史 `16`
  - `evaluator._prepare_overlap_context()` 已修复 `NUM_OVERLAP=0` 时
    `self.window_turns[-0:]` 等价于整窗复用的问题；当前 `overlap_turns <= 0`
    会明确禁用上一窗口 context
- SwiftVLN `7B` 启动约定（Updated: 2026-04-18）：
  - 主训练脚本默认 base model **仍是 3B**：
    - `/mnt/data1/home/jiangjiajun/.cache/modelscope/models/Qwen/Qwen2___5-VL-3B-Instruct`
  - 当前脚本已支持通过环境变量覆盖关键训练参数，而不修改默认值：
    - `STAGE1_MODEL_PATH`
    - `MODEL_PATH`
    - `BATCH_SIZE`
    - `LEARNING_RATE`
    - `NUM_EPOCHS`
    - `GRAD_ACCUM_STEPS`
    - `NUM_HISTORY`
    - `HISTORY_PROCESSOR_TYPE`
    - `NUM_OVERLAP`
    - 以及常用 memory / embed / scheduler 相关参数
  - 本地可用 `7B` 离线路径：
    - `/mnt/data1/home/jiangjiajun/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-7B-Instruct`
  - 2026-04-18 在 `17` 上用 `8卡` 对 `Qwen2.5-VL-7B` baseline 做过实测：
    - 配置：`satnav`, `overlap=0`, `per_frame`, `history`, `num_history=8`, `log_base=1.0`
    - `bsz=8` 稳定
    - `bsz=10` 稳定
    - `bsz=12` 也可稳定进入训练
  - 同日正式长跑中，`bsz=12` 在第 1 个 step 后触发过
    `pin_memory` 线程里的 `torch.AcceleratorError: CUDA error: invalid argument`
    （`rank4`，17 机）
  - 当前主训练脚本已支持 env 覆盖：
    - `DATALOADER_NUM_WORKERS`
    - `DATALOADER_PIN_MEMORY`
  - 当前在 `17` 上重启 `7B baseline` 的更稳妥口径：
    - `BATCH_SIZE=10`
    - `LEARNING_RATE=2e-5`
    - `DATALOADER_PIN_MEMORY=false`
  - 当前推荐的 `7B baseline` 起训口径：
    - `BATCH_SIZE=12`
    - `LEARNING_RATE=2e-5`
    - 其余 baseline 配置保持不变
- SwiftVLN `Qwen2.5-VL-32B` 17 机启动约定（Updated: 2026-05-05）：
  - 本地离线模型路径：
    - `/mnt/data1/home/jiangjiajun/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-32B-Instruct`
  - 17 机 `streamvln-container` 内没有 `tmux`，32B 长跑采用 **17 host tmux + docker exec 容器训练**：
    - launcher: `runtime/tmp/launch_qwen25_32b_17.sh`
    - config: `runtime/train_queue/qwen25_vl_32b_0418.env`
  - 2026-05-05 实测：
    - `BATCH_SIZE=1`
    - `GRAD_ACCUM_STEPS=8`（8 卡有效 batch=64）
    - `USE_DEEPSPEED=true`
    - `DEEPSPEED_CONFIG=zero3`
    - `MAX_LENGTH=24576`
    - `FREEZE_VIT=true`
    - `DATALOADER_PIN_MEMORY=false`
    - `DATALOADER_NUM_WORKERS=2`
    - `DATALOADER_PREFETCH_FACTOR=2`
    - `DATASET_NUM_PROC=1`
  - 保持 swiftvln 当前 default 语义配置：
    - `SatNav0418`
    - `NUM_FRAMES=32`
    - `NUM_FUTURE_STEPS=4`
    - `NUM_OVERLAP=0`
    - `HISTORY_PROCESSOR_TYPE=per_frame`
    - `NUM_HISTORY=8`
    - `COMPRESS_STRIDE=2`
    - no pixel/pose/uav embed
  - `zero3_offload` 在当前 17 环境会触发
    `AttributeError: 'DeepSpeedCPUAdam' object has no attribute 'ds_opt_adam'`，
    因此不作为默认选择。
  - 训练脚本已修正 CUDA allocator 环境变量：
    - 使用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
    - 不再使用无效的 `PYTORCH_ALLOC_CONF`
  - 不带 `FREEZE_VIT=true` 或未修正 allocator 时，32B 会在前 1-2 step 附近 OOM。
  - 2026-05-05 73 机重启 32B 时新增队列配置：
    - `runtime/train_queue/qwen25_vl_32b_0418_73_nomidckpt.env`
    - 训练超参与 17/原 73 配置一致，仅将 `SAVE_STEPS=5000`，跳过 step-1000
      中间 checkpoint，避免再次在中间保存阶段中止。
    - 已启动 session：`train_q25vl32b73_restart_224552`
    - 日志：`logs/train_launch/train_q25vl32b73_restart_224552.log`
    - 输出：`output/swiftvln/swiftvln-satnav-32b-1ep-f32s4-overlap0-pf-h8-b1.0-pool-s2-noembed-bs64-lr2e-5-224602`
- SwiftVLN no-memory / per-frame naming 配置约定（Updated: 2026-04-20）：
  - 训练脚本：`src/swiftvln/model/script/train/train_swiftvln_qwen2_5_vl.sh`
  - 数据集：`src/swiftvln/model/dataset.py`
  - 队列展示/解析：`src/swiftvln/scripts/train/train_queue.sh`
  - 评测按名解析说明：`src/swiftvln/scripts/eval/eval_by_name.sh`
  - 当前没有独立的 `USE_MEMORY=false` 开关；如需 no-memory，使用：
    - `HISTORY_PROCESSOR_TYPE=per_frame`
    - `NUM_HISTORY=0`
  - 该配置会让 dataset 采样 `0` 张历史帧，并且 system prompt 不插入 `<history_memory>`
  - 统一实验名约定：`pf-h0-nomem-b{log_base}-{method}-s{stride}`
  - `per_frame` 且 `USE_RANDOM=true` 且 `NUM_HISTORY>0` 时，实验名会额外带：
    - `pf-h{num_history}-random-b{log_base}-{method}-s{stride}`
  - `per_frame` 非 random 路径保持原格式不变：
    - `pf-h{num_history}-b{log_base}-{method}-s{stride}`
  - `src/swiftvln/scripts/train/train_queue.sh` 与
    `src/swiftvln/scripts/eval/eval_by_name.sh`
    当前都兼容解析旧的非 random 名字与新的 `-random-` 名字
  - 2026-04-20 起，`per_frame` 的 `USE_RANDOM=true` 已在**评测端**完整接通：
    - Python CLI：`src/swiftvln/model/eval.py`
    - evaluator 实际采样：`src/swiftvln/model/evaluator.py`
    - distributed eval 脚本：`src/swiftvln/model/script/eval/eval_swiftvln_qwen2_5_vl_distributed.sh`
    - `eval_by_name.sh` 解析出的 `USE_RANDOM=true` 不再只是命名元数据，而会真实影响 eval 的 history sampling
  - `log_base` / `use_random` 在 `NUM_HISTORY=0` 时保留为配置元数据，但不会实际影响采样
  - `gtc` / `segment_gtc` 不适用该 no-memory 约定，因为其历史采样逻辑不看 `NUM_HISTORY`
- SwiftVLN 本地 deploy 默认模型（Updated: 2026-04-21）：
  - CLI：`src/swiftvln/cli.py`
  - 解析常量：`src/swiftvln/deployment/model_resolver.py`
  - wrapper：
    - `src/swiftvln/scripts/deploy/start_swiftvln_deploy.sh`
    - `src/swiftvln/scripts/deploy/run_deploy_session.sh`
  - 当前若 deploy 未显式传 `--model-name` / `model_name`，默认使用：
    - `output/swiftvln/swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-b1.0-pool-s2-noembed-bs64-lr2e-5-182323`
  - `start_swiftvln_deploy.sh` 现支持：
    - `bash .../start_swiftvln_deploy.sh`
    - `bash .../start_swiftvln_deploy.sh <model_name> [session_root]`
  - `run_deploy_session.sh` 现支持：
    - `bash .../run_deploy_session.sh <requests_jsonl>`
    - `bash .../run_deploy_session.sh <requests_jsonl> [model_name] [session_root]`
- SwiftVLN `map` memory 配置约定（Updated: 2026-04-17）：
  - 共享实现：
    - `src/swiftvln/model/map_memory.py`
  - 训练链路：
    - 参数定义：`src/swiftvln/model/arguments.py`
    - dataset 接线：`src/swiftvln/model/dataset.py`
    - trainer 透传/校验：`src/swiftvln/model/trainer.py`
    - template pose 兼容：`src/swiftvln/model/template.py`
  - 评测链路：
    - CLI 参数：`src/swiftvln/model/eval.py`
    - evaluator window 刷新：`src/swiftvln/model/evaluator.py`
    - 单次 eval 脚本：`src/swiftvln/model/script/eval/eval_swiftvln_qwen2_5_vl_distributed.sh`
    - 按名评测解析：`src/swiftvln/scripts/eval/eval_by_name.sh`
  - 队列透传：
    - `src/swiftvln/scripts/train/train_queue.sh`
  - 当前 `map` 的产品定义：
    - `memory_method=map` 表示**替换原 history frame memory**，不是并存
    - 当前仅支持 `vln_env_type=satnav`
    - 当前仅支持 `history_processor_type=per_frame`
    - 当前要求 `use_tome=false`
    - **当前禁用所有 embed 增强**：`use_pixel_embed` / `use_pose_embed` / `use_uav_adapter` 必须同时为 `false`
      - 原因：map 是合成的俯视图，不是真实 RGB 相机帧，RGB-frame 对齐的 embed 语义不适用
      - 训练/评测在 `trainer._validate_memory_method` 与 `evaluator.__init__` 中硬校验，shell 脚本入口也会提前 `exit 1`
    - system prompt 仍复用统一的 `<history_memory>` 占位；只是视觉来源从历史帧切换为 `global map + local map`
    - explored map 未探索区域直接置 `0`
    - 当前默认中心策略：
      - `global` 为 `north-up`，默认使用 `adaptive_start`：
        初始以起点为中心；当历史轨迹/已探索区域接近边界时，按 `25m` 量化步长做最小必要平移，尽量让更多历史留在图内
      - `local` 为 `north-up`，始终以当前位置为中心
    - 默认参数：
      - `map_global_side_m=1000`
      - `map_local_side_m=400`
      - `map_render_px=448`
      - `map_mask_method=dilate20`
    - 训练与评测都按 **window cadence** 更新 map：
      - 训练样本里，map 固定锚定在 `start_idx` 之前的历史
      - 评测时，仅在 Overlap window 滑动时刷新 map cache；窗口内不更新
    - **首窗口行为**（与 `history` 模式不对称，需注意）：
      - `history` 模式在首窗口（`time_ids[0]==0` / eval `step_id==0`）**不插入** `<history_memory>`
      - `map` 模式在首窗口**仍会生成** global + local 两张图，但 `observed_poses` 为空 →
        mask 全黑、仅显示起点蓝点；system prompt 也始终带 `<history_memory>` 占位
      - 即 map 模式的 history slot 语义是"起点 + 已探索区域"，在首窗口退化为"仅起点"
  - 统一实验名约定：
    - `map-g{global}-l{local}-r{render}-{mask}-s{compress_stride}`
    - 例：`map-g1000-l400-r448-d20-s2`
  - 依赖说明：
    - `map_memory.py` 运行时依赖可选 geo 包：`rasterio` 与 `pyproj`
    - 当前仓库实现已做惰性导入；若环境缺依赖，只会在 `memory_method=map` 真正执行到地图渲染时报错
  - **On-disk render cache**（Updated: 2026-04-17）：
    - 位置：默认 `{dataset_root}/map_cache`（例：`/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260418/map_cache`）
      - 训练入口（`dataset.py`）从 `SatNavTrajectoryMetadataResolver.dataset_root` 推导
      - 评测入口（`evaluator.py`）从 habitat `DATA_PATH` 回溯找第一个 `ver_*` 目录
    - 开关（优先级从高到低）：
      1. 环境变量 `SWIFTVLN_MAP_CACHE_DIR=<path>` 强制指定路径
      2. 环境变量 `SWIFTVLN_MAP_CACHE_DIR=off|false|none|0|disable|disabled|no` 关闭缓存
      3. 训练/评测脚本 `MAP_CACHE_DIR`（default `"auto"` → 使用代码推导值）
      4. 都未设 → 使用推导的 `{dataset_root}/map_cache`
    - Cache key 由 render config digest + `scene_id` + `window_start` + 全部 poses 的
      `(x, y, altitude, heading_deg)` 保留 6 位小数的 bytes 组合而成（content-addressable）
    - 文件布局：`<cache_dir>/<key[:2]>/<key[2:4]>/<key>_{global,local}.png`
      - 两级 sharding 防止单目录 inode 爆炸
      - 原子写入（`tempfile` + `os.replace`），多进程 / 多 epoch 并发安全
    - 失效 / 清理：
      - `map_memory._MAP_CACHE_FORMAT_VERSION` 或任一 render 配置（`global_side_m` /
        `local_side_m` / `render_px` / `mask_method` / `hfov` / `sensor_width` /
        `sensor_height` / `global_center_mode` / `global_shift_*`）变化
        → key digest 自动改变 → 新旧缓存自然共存，不会读到过期结果
      - 如需回收磁盘，直接 `rm -rf {cache_dir}` 安全（会懒加载重建）
    - Debug：`SWIFTVLN_DEBUG=1` 会在首 N 次 render 打印 `[MAP DEBUG][builder] cache_hit/render ... cache=miss(M/wW)` 统计
    - 性能目标：rasterio 渲染 ~2.4s/次 → cache hit ~10ms（PNG decode），命中率稳定后
      DataLoader 近乎零 CPU 渲染成本（首 epoch 负责 warm-up）
- SwanLab 直连默认（Updated: 2026-04-08）：
  - `src/swiftvln/scripts/train/train_queue.sh`
  - `src/swiftvln/model/script/train/train_swiftvln_qwen2_5_vl.sh`
  - 当 `USE_SWANLAB=true` 时，训练脚本默认会清理 `http_proxy/https_proxy/HTTP_PROXY/HTTPS_PROXY/all_proxy/ALL_PROXY`
  - 目的：避免误继承本地 `127.0.0.1:7890` 一类代理，导致 SwanLab 登录失败
  - 如需保留代理，可显式设置 `SWANLAB_DIRECT_NETWORK=false`
  - `train_queue.sh` 现在也会把全局 `QA_DATASET` 显式写入临时训练脚本，避免 `qa*` 实验回退到模型脚本内的旧默认 QA 路径
- SwiftVLN 训练恢复支持（Updated: 2026-04-08）：
  - `src/swiftvln/model/script/train/train_swiftvln_qwen2_5_vl.sh`
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
- Eval todo 锁修复（Updated: 2026-04-15）：
  - `src/swiftvln/scripts/eval/eval_queue.sh`
  - `src/swiftvln/scripts/eval/enqueue_eval.sh`
  - 旧逻辑里 `eval_queue.sh` 在 `remove_line_from_todo()` 中使用
    `exec 201>"$TODO_LOCK_FILE"; flock 201`，FD 会保留在长生命周期 shell 中，
    并被后续 `eval_by_name.sh` / `torchrun` 子进程继承，导致整条 eval queue
    运行期间持续占用 `eval_todo.txt.lock`
  - 当前已改为短生命周期 subshell 持锁，更新完 todo 后立即释放
  - `enqueue_eval.sh` 现在对 `eval_todo.txt.lock` 使用带超时的 `flock -w`
    ，避免训练队列在“准备自动入评测队列”阶段无限阻塞
- 评测 worker：`src/swiftvln/scripts/eval/start_eval_worker.sh`
- 评测 monitor：`src/swiftvln/scripts/eval/start_eval_monitor.sh`
- 评测 watchdog：`src/swiftvln/scripts/eval/eval_watchdog.sh`
- 数据处理：`src/swiftvln/scripts/data_process/*.py`
- 数据集 merge：`src/swiftvln/scripts/data_process/merge_satnav_data.py`
- 数据同步：`src/swiftvln/scripts/data_sync/*.sh`
- SatNav merge skill：`.codex/skills/merge-satnav-data/SKILL.md`
- 主线评测 summary 指标（Updated: 2026-04-18）：
  - 公共汇总逻辑：`src/swiftvln/common/eval/runner.py`
  - 统计 helper：`src/swiftvln/common/eval/reporting.py`
  - 顶层 `success_rate / mean_spl / oracle_success / navigation_error / avg_steps`
    为当前 split 内所有 episode 的**直接平均**，无任何 trajectory-type 重加权
  - `evaluation_summary.json` 仍会写出 `by_trajectory_type` 细分，便于分类查看
  - `collect_eval_results.py` 的 `ALL_*` 列直接读顶层指标
  - 同等清理已同步至所有 baseline：
    `baseline/{streamvln,navila,uninavid,openfly}/src/eval_satnav.py`
- StreamVLN baseline 默认训练口径（Updated: 2026-06-29）：
  - 训练脚本：`baseline/streamvln/scripts/train_satnav.sh`
  - 当前默认 SatNav 正式训练配置：
    - `SATNAV_DATASET=SatNav-v0.1`
    - 训练目录：`/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/trajectory_data`
    - 可用 `SATNAV_TRAIN_DATA_DIR=<trajectory_data_dir>` 显式覆盖训练 trajectory 目录
    - `NUM_EPOCHS=1`
    - `LEARNING_RATE=2e-5`
    - `BATCH_SIZE=3`
    - `GRAD_ACCUM=2`
    - `GPUS_PER_NODE=8`
  - 即默认有效 batch size 为 `48`
  - `scratch` 与 `continue` 都复用同一组默认 batch 配置；如需更保守，可显式覆盖 `BATCH_SIZE=2`
- SatNav 0404 eval 子集管理工具（Updated: 2026-04-18）：
  - eval 子集生成脚本：`src/swiftvln/scripts/eval/generate_satnav_keep_lists.py`
  - 将生成的子集写回数据盘（先备份再覆盖）：
    `src/swiftvln/scripts/eval/trim_satnav_eval_episodes_by_keep_csv.py`
  - 分析产物目录：`runtime/analysis/satnav_keep_lists/`
- Baseline eval 结果子集回填脚本（Updated: 2026-04-18）：
  - `src/swiftvln/scripts/eval/apply_keep_subset_to_baselines.py`

### Baseline StreamVLN Layout (Updated: 2026-03-09)

`baseline/streamvln` 已按“入口脚本 / 源码实现”分层：

- 入口脚本：`baseline/streamvln/scripts/*.sh`
  - `scripts/train_satnav.sh` -> 调用 `baseline/streamvln/src/train_satnav.py`
  - `scripts/eval_satnav.sh` -> 调用 `baseline/streamvln/src/eval_satnav.py`
  - `scripts/train_eval_satnav.sh` -> 串行执行 train 后自动 eval；默认不带 webhook，
    仅当显式设置 `TRAIN_WEBHOOK_URL` / `EVAL_WEBHOOK_URL` 时发送通知
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
- StreamVLN baseline eval 默认：`8` 卡；eval split 和数据只由
  `baseline/streamvln/configs/satnav_task.yaml` 控制
  - 当前默认：
    - `SPLIT: all`
    - `DATA_PATH: /mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/episodes/eval`
    - `SCENES_DIR: /mnt/data3/jiangjiajun/dataset/satnav_datasets/scenes`
  - `SPLIT: all` 顺序展开为 `val_seen` + `val_unseen`
  - `eval_satnav.sh` 和 `train_eval_satnav.sh` 均不接受位置 split 参数
  - 评测结果目录约定（0319 起）：
    - SwiftVLN 主线模型（swiftvln）：
      `results/eval/<arch>/<model_name>/<split>/<timestamp>/`
      例：`results/eval/swiftvln/<model>/val_seen/20260319_143025/`
    - SatNav 默认同时跑 `val_seen` + `val_unseen`（不设置 `EVAL_SPLIT`），Habitat 默认 `EVAL_SPLIT=val_unseen`
- StreamVLN SatNav eval 约定（Updated: 2026-04-01）：
  - 多卡汇总改为 rank0 从 `result.jsonl` 离线去重汇总，不再用末尾 `all_gather(...)` 汇总本地 `results`
  - resume / 去重唯一键使用 `scene_id + episode_id`，避免仅按 `episode_id` 导致跨 scene 冲突
  - `--max_episodes` 语义为“先截断总 episode，再做分布式切分”
  - `evaluation_summary.json` 当前仅保留直接平均指标；不再额外写重加权字段

### Baseline NaVILA Layout (Updated: 2026-06-29)

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

NaVILA SatNav eval 约定：

- `baseline/navila/scripts/eval_satnav.sh` 已按 StreamVLN 方式收敛为命名参数主路径：
  - 只支持 `--model_dir`、`--model_name`、`--gpus`、`--max_episodes`、`--model_base`、`--action_format`、`--dry_run`
  - 不再支持位置参数、`--checkpoint_path`、`--split`、`--satnav_version`
  - split 和 eval 数据只从 `baseline/navila/configs/satnav_task.yaml` 读取
  - 当前默认 `DATASET.SPLIT: all`，`DATASET.DATA_PATH` 指向
    `/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/episodes/eval`
  - model zoo 归档根目录若有 `config.json`，直接用根目录加载；否则才选择最新
    `checkpoint-*`
  - 当前 dry-run 验证通过：
    `navila-continue0404-r2-20260427-141143-sample-hk7-fs7-stopx4`、
    `navila-scratch0404-r1-20260407-195154-sample-hk7-fs7-stopx4`
- NaVILA Hugging Face upload-ready 目录已生成在：
  - `output/model_zoo/baseline/HF_model/navila-satnav-continue-1ep-8f-sample-hk7-fs7-stopx4`
  - `output/model_zoo/baseline/HF_model/navila-satnav-scratch-1ep-8f-sample-hk7-fs7-stopx4`
  - 目录只保留 `config.json`、`llm/`、`mm_projector/`、`vision_tower/` 和 README；
    不包含训练日志、trainer state、RNG/scheduler state 或 `checkpoint-*`
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
  - `evaluation_summary.json` 当前仅保留直接平均指标；不再额外写重加权字段
- 生成停止条件与 dtype 稳定性修复（Updated: 2026-04-07）：
  - `baseline/navila/src/eval_satnav.py` 不再直接使用上游 `KeywordsStoppingCriteria`
  - 当前改为本地 `SafeKeywordsStoppingCriteria`，只匹配**生成后缀**，避免 Llama 3 prompt 内自带 `<|eot_id|>` 时在 `0 token` 阶段被误判为 stop
  - 评测新增 `--debug_generation` / `--debug_generation_limit`，可打印 `stop_str`、prompt 命中、首步 token/top scores、是否立即终止
  - 评测新增 `--eval_dtype {auto,float16,bfloat16,float32}`
  - `auto` 默认在支持时优先使用 `bfloat16`，用于规避部分 NaVILA checkpoint 在 `float16` 视觉前向下首步 logits 变成 `NaN`、输出 `!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!` 的问题
- NaVILA eval checkpoint 缓存约定（Updated: 2026-06-29）：
  - `baseline/navila/scripts/eval_satnav.sh`
  - `LOCAL_CACHE_DIR=""` 会禁用本地缓存
  - 若 checkpoint 位于 NFS 且容器内缺少 `rsync`，脚本会直接回退到源 checkpoint，不再写出空缓存并伪造 `.cache_complete`

NaVILA SatNav train 补充约定（Updated: 2026-03-25）：

- `baseline/navila/scripts/train_satnav.sh` 现在支持两种初始化模式：
  - `scratch`：从 `baseline/navila/model/navila-siglip-llama3-8b-v1.5-pretrain` 起训
  - `continue`：从 `baseline/navila/model/navila-llama3-8b-8f` 继续训练
  - 无参默认 `scratch`
  - 兼容旧调用：若只传一个非模式参数，则视为 `EXP_NAME`
- `baseline/navila/scripts/train_satnav.sh` 当前默认训练数据为
  `SATNAV_DATASET=SatNav-v0.1`，默认读取：
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/trajectory_data`
  可用 `SATNAV_TRAIN_DATA_DIR=/path/to/trajectory_data` 显式覆盖
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

### Baseline OpenFly Layout (Updated: 2026-06-29)

`baseline/openfly` 已新增 SatNav-only baseline 分层：

- 入口脚本：`baseline/openfly/scripts/*.sh`
  - `scripts/setup_env.sh`
  - `scripts/download_model.sh`
  - `scripts/train_satnav.sh`
  - `scripts/eval_satnav.sh`
- 源码目录：`baseline/openfly/src/*`
  - `src/train_satnav.py`
  - `src/eval_satnav.py`
  - `src/action_formats.py`
  - `src/dataset/satnav_dataset.py`
  - `src/openfly_core/*`（本地注册的 OpenFly HF config/model/processor）
- 配置目录：
  - `baseline/openfly/configs/satnav_task.yaml`
  - `baseline/openfly/configs/zero2.json`
- 模型目录：
  - `baseline/openfly/model/openfly-agent-7b/`（下载后落点）
  - `baseline/openfly/model/openvlaopenvla-7b-prismatic/`（scratch native OpenVLA/Prismatic
    checkpoint 落点，必须包含 `checkpoints/*.pt`）

OpenFly SatNav baseline 约定：

- 不依赖外部 `OpenFly-Platform` repo 运行时路径；训练与评测使用 `baseline/openfly/src/openfly_core/*`
  中本地注册的 HF 组件
- `baseline/openfly/scripts/download_model.sh` 默认下载 continue 模型；`--backend scratch`
  下载 `openvla/openvla-7b-prismatic` 到
  `baseline/openfly/model/openvlaopenvla-7b-prismatic`；`--backend all` 同时准备两套起训资产
- `baseline/openfly/scripts/train_satnav.sh` 当前默认训练数据为
  `SATNAV_DATASET=SatNav-v0.1`，默认读取：
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/trajectory_data`
  可用 `SATNAV_TRAIN_DATA_DIR=/path/to/trajectory_data` 显式覆盖
- `baseline/openfly/scripts/train_satnav.sh` 当前默认 `NUM_GPUS=8`
- `baseline/openfly/scripts/eval_satnav.sh` 已按 StreamVLN/NaVILA/UniNaVid 方式收敛为命名参数主路径：
  - 只支持 `--model_dir`、`--model_name`、`--gpus`、`--max_episodes`、`--action_format`、
    `--action_history_limit`、`--processor_path`、`--dry_run`
  - 不再支持位置参数、`--checkpoint_path`、`--split`、`--satnav_version`
  - split 和 eval 数据只从 `baseline/openfly/configs/satnav_task.yaml` 读取
  - 当前默认 `DATASET.SPLIT: all`，`DATASET.DATA_PATH` 指向
    `/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/episodes/eval`
  - 旧 model zoo 归档根目录没有根权重时，脚本会选择该根下最新 `checkpoint-*`
  - eval 仍会从模型名中的 `actcompact|actoriginal` 和 `hist<N>` 解析动作格式与历史长度；
    不再从 `data<N>` 解析 eval 数据版本
  - 当前 dry-run 验证通过：
    `openfly-baseline-1ep-data260418-bkcontinue-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-bs96-lr2e-5-20260420-095357`、
    `openfly-baseline-1ep-data260418-bkscratch-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-bs96-lr2e-5-20260420-233110`
- OpenFly Hugging Face upload-ready 目录已生成在：
  - `output/model_zoo/baseline/HF_model/openfly-satnav-continue-1ep-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-lr2e-5`
  - `output/model_zoo/baseline/HF_model/openfly-satnav-scratch-1ep-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-lr2e-5`
  - 目录把 `checkpoint-40055` 中的 HF 权重、config、processor/tokenizer 文件提升到根目录；
    不包含训练状态、RNG/scheduler state、trainer state 或 `zero_to_fp32.py`
- 支持两种 backend（Updated: 2026-04-17）：
  - `continue`：直接加载 HF OpenFly checkpoint (`openfly-agent-7b`)，在 OpenFly 已完成 VLN 训练的基础上继续训练
  - `scratch`：从 Prismatic/OpenVLA `.pt` checkpoint 初始化权重（OpenFly 任务训练前的原始 OpenVLA），
    复用当前 HF `Trainer` / `checkpoint-*` 训练链路，相当于从 VLN 训练起点重新开始
- backend 通过环境变量切换：
  - `OPENFLY_BACKEND=continue|scratch`
  - 当前默认是 `continue`
- `scratch` backend 约定（Updated: 2026-04-17）：
  - 默认模型路径：
    `baseline/openfly/model/openvlaopenvla-7b-prismatic`
  - 默认 processor/tokenizer 来源：
    `baseline/openfly/model/openfly-agent-7b`
  - 当前实现并不会在运行时 import 外部 `OpenFly-Platform` repo
  - 而是通过
    `baseline/openfly/src/native_core/checkpoint_conversion.py`
    把 Prismatic checkpoint 映射到当前 `openfly_core` HF 模型结构后再训练
  - `build_native_hf_model(...)` 当前会按请求的 `torch_dtype`
    临时设置默认 dtype 后构模，避免 `scratch` 路径在启用 `flash_attention_2`
    时仍以 `float32` 构建 7B 模型并触发不兼容 warning
  - 因此 `scratch` 训练产物仍然是标准 HF `checkpoint-*` 目录，可直接复用现有
    `baseline/openfly/src/eval_satnav.py`
- 支持两种动作格式（Updated: 2026-04-15）：
  - `compact`：四动作文本 supervision / decode
    `stop / forward / left / right`
  - `original`：保留 OpenFly action-token 表征；SatNav 仅开放 4 个合法 8 维动作模板
- `original` 模式的 SatNav 8 维动作模板约定：
  - `stop -> [1, 0, 0, 0, 0, 0, 0, 0]`
  - `forward -> [0, 10, 0, 0, 0, 0, 0, 0]`
  - `left -> [0, 0, 15, 0, 0, 0, 0, 0]`
  - `right -> [0, 0, 0, 15, 0, 0, 0, 0]`
  - 其中 SatNav forward 固定 `10m`，左/右转固定 `15deg`
- 训练/评测脚本通过环境变量切换：
  - `OPENFLY_BACKEND=continue|scratch`
  - `OPENFLY_ACTION_FORMAT=compact|original`
  - `OPENFLY_UNNORM_KEY` 默认 `satnav_original`
- `scratch` backend 训练修复（Updated: 2026-04-16）：
  - `baseline/openfly/src/native_core/checkpoint_conversion.py`
  - `baseline/openfly/src/backends/native_backend.py`
  - `baseline/openfly/src/openfly_core/modeling_prismatic.py`
  - `baseline/openfly/configs/zero1.json`
  - `baseline/openfly/configs/zero2.json`
  - Prismatic → HF runtime 构模现在会：
    - 按请求的 `torch_dtype` 建图，避免 `flash_attention_2 + fp32` 组合
    - 在 `no_init_weights()` 下实例化 7B HF 模型，规避随机初始化导致的超长卡顿
  - vision 前向现在会把 `pixel_values` cast 到视觉 backbone 的参数 dtype，
    修复 `Input type (float) and bias type (c10::BFloat16)` 错误
  - OpenFly DeepSpeed 配置已显式设置 `torch_adam=true`，避免在 H100 上 JIT 构建
    `FusedAdam` 时触发 `nvcc fatal: Unsupported gpu architecture 'compute_90'`
  - 98 服务器 smoke 结果：
    - 单卡 `scratch + original + no-DeepSpeed` 已能正常进入训练并产出 loss / checkpoint
    - 8 卡 `scratch + original + zero2` 已能正常完成 `max_steps=2` smoke
    - 8 卡 smoke 的 `checkpoint-2` 落盘后仍会有一段较长尾部收尾；日志需看到
      最终 `train_runtime / train_loss` 统计，不能只看 checkpoint 文件是否已出现
- `baseline/openfly/scripts/train_satnav.sh` 默认实验名会显式追加动作模式后缀：
  - `-bkcontinue`
  - `-bkscratch`
  - `-actcompact`
  - `-actoriginal`
  并追加 SatNav 采样标签：
  - `-sample-hk<HEAD_KEEP>-fs<SAMPLE_STRIDE>-stopx<STOP_REPEAT>-stoph<STOP_HISTORY_AUG>`
- OpenFly SatNav 训练收尾兼容（Updated: 2026-04-15）：
  - `baseline/openfly/src/train_satnav.py`
  - `baseline/openfly/scripts/train_satnav.sh`
  - 训练入口会在进程内 monkey-patch `AcceleratedOptimizer.train/eval`：
    若底层 optimizer 没有对应方法，则退化为 no-op，避免
    `DeepSpeedZeroOptimizer has no attribute train`
  - 多卡 DeepSpeed 训练结束后不再额外执行
    `trainer.save_model(output_dir)` 根目录全量导出；默认以 `checkpoint-*`
    作为可评测产物，规避 Zero2 收尾长时间卡住
  - root 输出目录与每个 `checkpoint-*` 目录现在都会额外写
    `backend_meta.json`，记录 `backend / model_name_or_path / processor_source`
    等来源信息
  - 新增 `OUTPUT_DIR_OVERRIDE=<abs_path>`，可把训练输出直接落到本地盘
    （如 73 的 `/mnt/data3/...`），减少往 NFS 工作区写大 checkpoint 时的卡顿风险
  - 默认 `SAVE_TOTAL_LIMIT=1`，训练过程中最多保留最新一个 `checkpoint-*`
  - 训练脚本默认禁用 wandb：
    - `--report_to none`
    - `WANDB_DISABLED=true`
    - `WANDB_MODE=disabled`
  - 训练脚本新增可调开关：
    - `USE_FLASH_ATTENTION_2`
    - `WEIGHT_DECAY`
    - `WARMUP_RATIO`
    - `LR_SCHEDULER_TYPE`
    - `MAX_GRAD_NORM`
    - `DATALOADER_NUM_WORKERS`
    - `REPORT_TO`
  - 当前默认训练配置（Updated: 2026-04-16）：
    - `TRAIN_BSZ=12`
    - `GRAD_ACCUM=1`
    - `TORCH_DTYPE=bfloat16`
    - `USE_FLASH_ATTENTION_2=true`
    - `LEARNING_RATE=2e-5`
    - `SAVE_STEPS=10000`
    - `LR_SCHEDULER_TYPE=linear`
    - `WEIGHT_DECAY=0.0`
    - 该默认值来自 73 上 8xH100 短程 benchmark；目标是无 acc-grad 前提下提高吞吐
  - OpenFly SatNav 默认采样策略（Updated: 2026-04-17）：
    - `baseline/openfly/src/dataset/satnav_dataset.py`
    - `baseline/openfly/scripts/train_satnav.sh`
    - 现已切到统一全局参数的 `head + denser forward + tail keep + stop window` 采样与监督
      逻辑；不再按 `Boundary/LandmarkSet/Road` 分类型设置不同采样参数
    - 当前默认值：
      - `SATNAV_HEAD_KEEP=7`
      - `SATNAV_SAMPLE_STRIDE=3`
      - `SATNAV_STOP_REPEAT=2`
      - `SATNAV_STOP_WINDOW=0`
      - `SATNAV_TAIL_KEEP=5`
      - `SATNAV_STOP_HISTORY_AUG=1`
    - 语义：
      - 保留每条轨迹前 `7` 步
      - 所有 `left/right` 全保留
      - 连续 `forward` 段按 stride=`3` 更密地保留
      - near-goal tail 始终保留，但默认不启用额外 stop window 重标
      - `stop_history_aug` 默认保持 `1`，不再靠多 terminal history 变体做 stop 增强
    - 0404 近似统计（Updated: 2026-04-17）：
      - 原始全量：`5.396M`
      - 当前默认：约 `3.825M`
      - 全局动作占比约：
        - `stop 5.5%`
        - `forward 57.9%`
        - `left 19.1%`
        - `right 17.5%`
      - 98 上最近一次 8xH100 实测吞吐为 `33430 steps / 26032.96s`
        （约 `0.779s/step`，global batch `96`）
      - 按此估算，当前默认配置 1 epoch 约 `8.8h`，实务上按 `9.0-9.5h`
        预留更稳妥
    - 2026-04-17 修复：
      - `baseline/openfly/src/backends/base.py` 现兼容 `hf/native` 与
        `continue/scratch` 两套 backend 字符串；旧环境变量值不会再导致训练初始化阶段
        直接报 `Unsupported OpenFly backend`
    - 训练产物中的 `dataset_statistics.json` 现会额外记录 `sampling_env`
      与 `trajectory_type_counts`，便于回溯具体采样配置
  - OpenFly `original` 训练监督（Updated: 2026-04-16）：
    - `baseline/openfly/src/dataset/satnav_dataset.py`
    - `baseline/openfly/src/train_satnav.py`
    - 训练时仅监督 `original` 8 维动作 token 中前 `4` 个有效动作维度 token
    - 后 `4` 个恒定 inactive 维度 token 与结尾 `eos` 不再参与 CE loss
    - 训练默认通过自定义 `OpenFlyTrainer` 对这 4 个 token 施加位置加权：
      - `OPENFLY_ORIGINAL_DIM_LOSS_WEIGHTS=0.4,1.2,1.2,1.2`
    - 含义：
      - 下调第 1 个动作 token 的权重
      - 上调第 2/3/4 个动作 token 的权重
    - 目的：削弱 `stop` 在第 1 个 supervised token 位置上的结构性优势，减少
      “会走会转，但收尾时过早 stop” 的偏置
- prompt 保持 OpenFly 原问句风格：
  `What action should the robot take to ...?`
- 训练输出目录：
  `output/openfly-baseline/<EXP_NAME>/`
- 评测结果目录：
  `results/openfly-baseline/<EXP_NAME_or_subpath>/<split>/`
- `baseline/openfly/src/eval_satnav.py` 的 `result.jsonl` 现在会显式写出：
  - `action`
  - `parsed_action`
  - `generated_text`
  - `action_trace`
  便于直接检查 compact/original 两种动作模式的逐步输出；原始 `raw_outputs`
  仍保留用于调试
- 多卡 train/eval 启动约定：
  - `baseline/openfly/scripts/train_satnav.sh`
  - `baseline/openfly/scripts/eval_satnav.sh`
  - 默认使用显式 `MASTER_ADDR=127.0.0.1` + `MASTER_PORT`，不使用 `torchrun --standalone`
- 环境名：
  `openfly-baseline`
- 该 baseline 只面向 SatNav 离线数据与 SatNav 平台评测，不包含 AirSim / UE / GTAV / ROS2 / TFDS 工具链依赖

### Baseline UniNaVid Train Modes (Updated: 2026-06-29)

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
- `baseline/uninavid/scripts/train_satnav.sh` 默认 `SAVE_TOTAL_LIMIT=1`，
  训练过程中最多保留最新一个 `checkpoint-*`
- `baseline/uninavid/scripts/train_satnav.sh` 当前默认训练数据为
  `SATNAV_DATASET=SatNav-v0.1`，默认读取：
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/trajectory_data`
  可用 `SATNAV_TRAIN_DATA_DIR=/path/to/trajectory_data` 显式覆盖

UniNaVid SatNav eval 约定（Updated: 2026-06-29）：

- `baseline/uninavid/scripts/eval_satnav.sh` 已按 StreamVLN/NaVILA 方式收敛为命名参数主路径：
  - 只支持 `--model_dir`、`--model_name`、`--gpus`、`--max_episodes`、`--model_base`、`--dry_run`
  - 不再支持位置参数、`--checkpoint_path`、`--split`、`--satnav_version`
  - split 和 eval 数据只从 `baseline/uninavid/configs/satnav_task.yaml` 读取
  - 当前默认 `DATASET.SPLIT: all`，`DATASET.DATA_PATH` 指向
    `/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/episodes/eval`
  - model zoo 归档根目录若有 `config.json` 和 `pytorch_model*.bin`，直接用根目录加载；
    否则才选择最新 `checkpoint-*`
  - 当前 dry-run 验证通过：
    `uninavid-baseline-continue-1ep-data260418-bs192-lr1e-5-20260418-203618`、
    `uninavid-baseline-scratch-1ep-data260418-bs192-lr1e-5-20260418-203618`
- UniNaVid Hugging Face upload-ready 目录已生成在：
  - `output/model_zoo/baseline/HF_model/uninavid-satnav-continue-1ep-lr1e-5`
  - `output/model_zoo/baseline/HF_model/uninavid-satnav-scratch-1ep-lr1e-5`
  - 目录只保留 `config.json`、`pytorch_model*.bin`、index、tokenizer 相关文件和 README；
    不包含 trainer state、training args 或 `checkpoint-*`
- `baseline/uninavid/src/eval_satnav.py` 的断点续跑与离线汇总唯一键使用 `scene_id + episode_id`
- 避免仅按 `episode_id` 去重时，`val_seen` 中跨 scene 重复 episode id 导致的误跳过与汇总失真
- 多卡汇总为 rank0 在 `dist.barrier()` 后从 `result.jsonl` 按联合键离线去重并写 `evaluation_summary.json`
- 分布式收尾使用带 `device_ids=[local_rank]` 的 barrier，并在 `main()` 退出时显式 `destroy_process_group()`，避免 NCCL barrier / process group 清理 warning
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

### Train Watchdog 异步回调机制（Updated: 2026-04-21）

训练同样使用 **tmux + watchdog** 事件驱动模式，支持多服务器并行启动：

- 训练在 tmux session 中运行（命名：`train_<short_desc>_<HHMMSS>`）
- `train_watchdog.sh` 后台监控，通过 `train_events.log` 消费实验事件
- 2026-04-21 修复：`cleanup_old_runs()` 里的计数从 `((count++))` 改为 `((count += 1))`
  旧写法在 `set -e` 下清理到首个旧 run 目录时会直接退出，表现为 watchdog 启动后立刻静默结束
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
- 当前四个 baseline 默认发布数据集：`SatNav-v0.1`
  - train trajectory root: `/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/trajectory_data`
  - eval root: `/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/episodes/eval`
  - eval 默认 split: `all` (`val_seen` + `val_unseen`)
- `ver_260418` 为 0418 历史训练 / eval 快照，仍保留如下路径记录：
- Legacy eval episodes (val_seen):
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260418/episodes/eval/val_seen/all_episodes.json`
  （当前：4574 条；2026-04-20 在 2026-04-19 重建 split 基础上移除了 `27` 个与 train 路线重复的 episodes）
- Legacy eval episodes (val_unseen):
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260418/episodes/eval/val_unseen/all_episodes.json`
  （当前：8756 条；`val_unseen` 默认评测集）
- **注意**：`episodes/eval/` 下当前默认只有 `val_seen/` 和 `val_unseen/` 两个子目录
- `val_seen_update/` 已不再作为当前 0418 默认 eval 目录使用；如果历史脚本仍引用它，需要先改回 `val_seen`
- QA JSONL:
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260418/data/qa_swift.jsonl`
  （2026-04-26 已重写，JSONL 内图片绝对路径均指向 `ver_260418`，不再指向 `ver_260404`）
- Trajectory data:
  - train 主集：`/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260418/trajectory_data`
  - val_seen standalone 子集：`/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260418/trajectory_data_val_seen`
- Scene maps:
  - active: `/mnt/data3/jiangjiajun/dataset/satnav_datasets/scenes`
  - backup(old): `/mnt/data3/jiangjiajun/dataset/satnav_datasets/old_scenes`

### SatNav ver_260418 Snapshot (Updated: 2026-05-01)

- 这是 0418 历史快照；四个 baseline 当前默认已切到 `SatNav-v0.1`
- 当时默认路径已同步到：
  - `src/swiftvln/model/script/train/train_swiftvln_qwen2_5_vl.sh`
  - `src/swiftvln/scripts/train/train_queue.sh`
  - `src/swiftvln/configs/satnav_task.yaml`
  - `baseline/{streamvln,navila,uninavid,openfly}/configs/satnav_task.yaml`
  - `baseline/{streamvln,navila,uninavid,openfly}/scripts/*satnav.sh`
- 新增 `trajectory-group` 级别 seen split 构建脚本：
  - `src/swiftvln/scripts/data_process/build_similarity_val_seen_from_train.py`
  - 用途：从当前 `train` 中按 `(scene_id, trajectory_id)` 选完整 traj group，构建 `100% route-disjoint` 的 `val_seen`，同时抽取 standalone `trajectory_data_val_seen` 并从主 train / 主 `trajectory_data` 中同步移除
  - 默认参考 split：`val_seen_update`
  - 当前默认参数口径：`scene_coverage_min=45`、`size_tolerance=5%`、`ratio_tolerance=2pp`、`top_k=5`
- 2026-04-19 重建后的 0418 split / trajdata 状态：
  - `val_seen`: `4574` episodes, `56` scenes
    - `2026-04-20` 额外移除了 `27` 个与 train 路线重复的 episodes（9 条唯一轨迹）
  - `val_unseen`: `8756` episodes
    - `2026-04-26` 已基于 `val_unseen/all_episodes.json` 重建类型拆分文件：
      `boundary=2863`、`landmark=2872`、`road=3021`，三者 union 与 all 严格一致
  - `trajectory_data_val_seen`: 历史 standalone 子集仍为 `4601` trajectories；本次仅修正 `episodes/eval/val_seen/*` 与相关 baseline `val_seen` 结果
  - train after cleanup: `105164` episodes
  - main `trajectory_data`: `105164` trajectories
  - train 与主 `trajectory_data` 现在已严格对齐（`0` missing / `0` extra）
  - 清理 manifest：`/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260418/episodes/train/remove_missing_traj_manifest.json`
- 新 `val_seen` 与 train 在 episode / `(scene_id, trajectory_id)` 两级都 `0 overlap`
- 2026-04-19 当前评测口径补充：
  - 0418 实际默认 eval split 只有 `val_seen` 与 `val_unseen`
  - `val_seen_update` 不再作为当前默认目录存在；若要复现实验历史，需要显式提供对应文件而不是继续假定默认脚本可直接找到
- 2026-04-26 数据一致性修复：
  - `data/qa_swift.jsonl` 已重新生成，共 `273120` 行，图片路径版本计数为 `ver_260418: 273120`，缺图数为 `0`
  - `episodes/eval/val_unseen/{boundary,landmark,road}_episodes.json` 已从当前 `all_episodes.json` 定点重建
  - 注意：`merge_manifest.json` 仍是原始 merge 产物记录，不代表后续 0418 split 清理 / QA 路径修复后的当前状态
- 2026-05-01 release 准备：
  - episode-only release 目录：
    `/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-Episodes-v0.1`
  - 该目录只保留 `episodes/`（含 `train/`, `eval/val_seen/`, `eval/val_unseen/`）以及 release 文档，不包含 `data/`, `trajectory_data/`, `raw_data/`, `check_trajectory_data/`
  - `SatNav-Episodes-v0.1/episodes` 与 `ver_260418/episodes` 已同步更新 Boundary subtype 命名：
    - 原缺失 `trajectory_subtype` -> `loop`
    - `overlap` -> `extended`
    - `arc` 保持不变
  - 更新后 Boundary subtype 条目计数（跨 all/type 文件重复计数）：`loop=43340`, `extended=8826`, `arc=8814`；无缺失字段
  - `SatNav-Episodes-v0.1/episodes` 与 `ver_260418/episodes` 已同步更新 LandmarkSet subtype 命名：
    - `aux_info.turn_count=2` -> `trajectory_subtype=one_turn`
    - `aux_info.turn_count=3` -> `trajectory_subtype=two_turn`
  - 更新后 LandmarkSet subtype 条目计数（跨 all/type 文件重复计数）：`one_turn=73920`, `two_turn=15528`；无缺失字段
  - `SatNav-Episodes-v0.1/episodes` 与 `ver_260418/episodes` 已同步更新 Road subtype 命名：
    - `Highway` -> `road`
    - `Multiway` -> `hybrid`
    - `Waterway` -> `waterway`
  - 更新后 Road subtype 条目计数（跨 all/type 文件重复计数）：`road=67936`, `hybrid=7398`, `waterway=11226`；无缺失字段
  - `ver_260418/trajectory_data` 中未存储 `trajectory_type` / `trajectory_subtype` 字段，因此本次 subtype 命名不涉及 trajectory cache 内容改写

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
- SwiftVLN 维护自己的 SatNav 轨迹生成配置：
  `src/swiftvln/configs/satnav_trajectory_generation.yaml`
  - 该文件用于调用 SatNav repo 的
    `applications/trajectory_generation/generate_parallel.py`
  - `satnav-data` skill 会基于该 YAML 生成临时配置，并只覆盖
    `DATASET.DATA_PATH` 与 `DATASET.SCENES_DIR`
  - 生产默认：`MAX_EPISODE_STEPS=500`、`FORWARD_STEP_SIZE=10`、
    `TURN_ANGLE=15`、`RGB_SENSOR=448x448/HFOV90`、
    `LandmarkSet SUCCESS_DISTANCE=3.0`
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

- 共享工作区挂载点：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN-refactor`
- **SSH 仅用于**：远端 GPU/进程状态检查、远程 tmux 启动/停止。
- **文件操作**（脚本、队列写入、输出读写）：始终是本地操作，无需 SSH。
- server 17 Docker 容器名查询：`ssh 10.246.132.17 "docker ps"`

### Conda Environments

| Env | Purpose |
|---|---|
| `swift-vln-train` | SwiftVLN 主线训练（SwiftVLN / StreamVLN / CompressVLN） |
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
- `EXPERIMENTS` 数组（格式：`model|config|changes|ds_names|ds_paths|reserved|qa_ratio`；第 6 列保留为空）
- `ENV_TYPE`（`satnav` 或 `habitat`）

`train_queue.sh` 的 SwanLab 约定（Updated: 2026-04-21）：
- 交互式默认启用 SwanLab
- 非交互模式默认仍为 `USE_SWANLAB=true`
- 非交互配置文件现在可显式写 `USE_SWANLAB=false` 关闭 SwanLab；`train_queue.sh` 不再强制改回 `true`
- 默认 `SWANLAB_PROJECT=SatNav`

由 `orchestrate-plan` skill 在运行时通过 Write tool 生成，放在 `runtime/plans/generated/` 下。

### Offline Model Convention (Updated: 2026-03-17)

- 后续 VLN 训练任务默认使用**离线本地模型**，不依赖在线下载（避免 DNS/外网波动导致训练失败）。
- SwiftVLN 的默认基座模型路径：
  `/mnt/data1/home/jiangjiajun/.cache/modelscope/models/Qwen/Qwen2___5-VL-3B-Instruct`
- 启动训练前必须先检查该路径存在且非空；若缺失，先修复模型路径/缓存，再启动训练。
- 若脚本默认值仍是 `Qwen/Qwen2.5-VL-3B-Instruct`（在线 ID），运行时需显式覆盖为上述本地绝对路径。

### tmux Session Naming Convention

所有长时间运行的任务（训练/评测）必须在 tmux 中启动：

```
train_<short_desc>_<HHMMSS>   # 例: train_overlap_smoke_143025
eval_<short_desc>_<HHMMSS>    # 例: eval_overlap_smoke_150200
```

## Smoke Test Convention

- 使用 skill：`.codex/skills/swiftvln-smoke-test/SKILL.md`
- 原则：小规模、可复现、不可破坏。
- 默认范围：SatNav-only。
- 训练 smoke 仅使用多卡多 GPU（不做单卡 smoke）。
- 主线 smoke 覆盖模型：`swiftvln`，要求 train+eval 全链路可用。
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
- 目标：基于真实 `UAV ↔ Satellite` 配对图做视觉对齐，产出后续可接入 SwiftVLN 的 adapter

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
- teacher 默认对齐到目标 SwiftVLN checkpoint 的 visual tower 输出空间
- `TeacherVisionTower` 必须使用 `Qwen2_5_VLForConditionalGeneration.from_pretrained(...)`
  加载 Qwen2.5-VL checkpoint；不要再走 `AutoModelForCausalLM`，否则会在
  `qwen2_5_vl` config 上报模型类型不识别
- 当前推荐 teacher：
  `/mnt/data4/jiangjiajun/archive/swiftvln/data0317/train/swiftvln-satnav-3b-1ep-f32s4-overlap16-gtc-k512-noembed-bs64-lr2e-5-202149/v0-20260318-202212/checkpoint-3957`
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

## SwiftVLN UAV Adapter Stage-B (Updated: 2026-04-07)

SwiftVLN 已接入 `uav_adapter` 的 Stage-B 最小链路：

- 在线 enhancement 模块：`src/swiftvln/common/embedding_enhancement/uav_adapter.py`
- pipeline 工厂：`src/swiftvln/common/embedding_enhancement/__init__.py`
- SwiftVLN 模型加载与本地/外部权重恢复：
  `src/swiftvln/model/model.py`
- SwiftVLN trainer 参数透传与 pipeline 重建：
  `src/swiftvln/model/trainer.py`
- SwiftVLN eval 参数透传：
  `src/swiftvln/model/eval.py`
- SwiftVLN 分布式评测脚本参数透传：
  `src/swiftvln/model/script/eval/eval_swiftvln_qwen2_5_vl_distributed.sh`
- 训练脚本参数透传：
  `src/swiftvln/model/script/train/train_swiftvln_qwen2_5_vl.sh`

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
  `src/swiftvln/model/script/test/test_uav_adapter_strategy.py`
- 现有 pixel/pose loader smoke：
  `src/swiftvln/model/script/test/test_pixel_embed_strategy.py`
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

## Local Deployment (Updated: 2026-04-21)

- CLI 入口：`python -m swiftvln deploy --model swiftvln --model-name <EXP_NAME>`
- CLI 接线：`src/swiftvln/cli.py`
- runner：`src/swiftvln/runners/deploy.py`
- 部署包：`src/swiftvln/deployment/`
  - `model_resolver.py`：解析 swiftvln baseline 实验名并查找最新 checkpoint
  - `gpu.py`：默认自动选择本机一张空闲 H100；若已设置 `CUDA_VISIBLE_DEVICES`，则沿用现有可见卡
  - `loader.py`：加载 swiftvln checkpoint
  - `policy.py`：baseline-only 的 window/history/overlap 推理逻辑
  - `session.py`：单会话状态机、图片落盘、events/summary 记录
  - `server.py`：JSONL stdin/stdout 协议入口
- README：`src/swiftvln/deployment/README.md`
- 启动 wrapper：`src/swiftvln/scripts/deploy/start_swiftvln_deploy.sh`
- 单次 session wrapper：`src/swiftvln/scripts/deploy/run_deploy_session.sh`
- smoke 脚本：`src/swiftvln/scripts/deploy/deploy_smoke.sh`
- 单测：`tests/test_swiftvln_deployment.py`

当前部署产品定义：

- 只支持 `swiftvln` baseline `per_frame + pool + noembed`
- 当前只适配名字里的 `NUM_OVERLAP`、`NUM_HISTORY`、`LOG_BASE`、`USE_RANDOM`
- 明确不支持：`map` / `gtc` / `segment_gtc` / `tome` / `initial` / embedding enhancement
- 协议是**单进程、单会话** JSONL CLI，不提供 HTTP
- 输入命令：
  - `{"type":"start","instruction":"...","session_id":"optional"}`
  - `{"type":"image","image_path":"/abs/path/to/image.jpg"}`
  - `{"type":"end","reason":"optional"}`
- 状态机：`waiting_start -> waiting_image -> waiting_feedback -> closed`
- 会话目录默认写到：`runtime/deploy/sessions/<session_id>/`
  - 固定产物：`session_meta.json` / `events.jsonl` / `session_summary.json` / `images/`
- 行为与 eval baseline 对齐：
  - 第一张图必定触发一次推理
  - 之后每张图表示刚完成一个动作
  - 若动作队列在这张反馈图上耗尽，则立刻用该图再次推理
  - 若模型输出无法解析动作，fallback 为 `[STOP]`

## ms-swift 4.x Adaptation (Updated: 2026-05-12)

- 本仓库当前 `ms-swift-refactor` 分支正在适配最新 ms-swift。
- 保护约定：
  - 不修改旧环境 `swift-vln-train` / `swift-vln-eval`
  - 不修改旧仓库 `/mnt/data1/home/jiangjiajun/workspace/ms-swift`
  - 仅使用 update 环境：`swift-vln-train-update` / `swift-vln-eval-update`
- `README.md` 现在是 train/eval 两套 from-scratch 安装文档，不再要求从旧环境 clone：
  - conda 基础层：`environment-train.yml`
  - pip 训练依赖：`requirements-train.txt`
  - PyTorch CUDA 12.8 与 `flash-attn` 在 README 中单独按顺序安装
    - 若在 `/tmp` 等不同挂载点构建 `flash-attn` 遇到 `Invalid cross-device link`，
      需把 `TMPDIR` 与 `PIP_CACHE_DIR` 放到同一文件系统后重试
  - 当前目标训练环境仍是 `swift-vln-train-update`
  - eval conda 基础层：`environment-eval.yml`
  - eval pip 依赖：`requirements-eval.txt`
    - 包含 `qwen-vl-utils==0.0.14`，与 README 的 Qwen-VL 验证口径一致
  - 当前目标 eval 环境仍是 `swift-vln-eval-update`
  - eval 从零安装 `SwiftVLN` / `ms-swift-lateset` / `SatNav` / `habitat-lab-0.2.4`
  - eval 使用 Python `3.9` + Habitat `0.2.4` stack；不能直接复用 train 的 Python `3.10` 环境
  - `pyproject.toml` 的 `requires-python` 已放宽到 `>=3.9`，以支持 eval 环境 `pip install -e .`
- update 环境当前使用：
  - `/mnt/data1/home/jiangjiajun/workspace/ms-swift-lateset`
  - 版本：`ms-swift 4.2.0.dev0`
  - commit：`ad7d5c515 [docs] fix docs (#9244)`
- SwiftVLN 当前直接使用 ms-swift 4.x API：
  - `swift.arguments.SftArguments`
  - `swift.dataset.*`
  - `swift.model.*`
  - `swift.template.*`
  - `swift.pipelines.train.sft.SwiftSft`
  - 不再保留 `swift.llm.*` / ms-swift 3.x fallback
- SwiftVLN 模型注册：
  - 直接使用 `SwiftVLNQwen25VLLoader`
  - `ModelMeta` 使用 `loader=SwiftVLNQwen25VLLoader`
  - 特殊 token 仍为 `<history_image>` / `<history_memory>` / `<current_image>`
- 默认脚本环境：
  - train 脚本默认激活 `swift-vln-train-update`
  - eval distributed 脚本默认激活 `swift-vln-eval-update`
  - 可分别通过 `SWIFTVLN_TRAIN_CONDA_ENV` / `SWIFTVLN_EVAL_CONDA_ENV` 覆盖回其他环境
- Qwen3/Qwen3.5 可行性报告：
  - `reports/ms_swift_qwen3_future.md`
  - 结论：可适配，但需要模型族抽象，不能只替换 `model_type` / `model_path`
- Qwen3-VL 适配重构（Updated: 2026-05-01）：
  - 当前主线已保留 `swiftvln_qwen2_5_vl` 并新增 `swiftvln_qwen3_vl`
  - 共享逻辑：
    - streaming KV-cache 状态管理
    - `<history_image>` / `<history_memory>` / `<current_image>` 注入
    - `embed_enhance` pipeline 创建、迁移到目标 device/dtype、checkpoint 权重恢复
  - 模板实现：
    - Qwen2.5 继续基于 ms-swift `Qwen2_5VLTemplate`
    - Qwen3 基于 ms-swift `Qwen3VLTemplate`
    - history/current 图像展开与压缩逻辑通过 SwiftVLN mixin 复用
    - Qwen3 visual encoder 返回 `pooler_output/deepstack_features` 或 tuple；训练模板与 evaluator 均需先归一化为 pooled visual tokens
    - 2026-05-12 起训练模板不再强制 `processor.image_processor(..., do_resize=False)`；
      由 Qwen processor 自行按 patch/merge 规则规整图像尺寸。Habitat 训练集中存在
      640x480 / 644x476 等尺寸，Qwen3/Qwen2.5 fast image processor 在强制
      `do_resize=False` 时会因 `patches.view(...)` shape 不整除而失败。
  - Qwen3 DeepSpeed 兼容：
    - ms-swift Qwen3-VL DeepSpeed patch 在 `inputs_embeds` 路径仍会访问 `input_ids.device`
    - `SwiftVLNQwen3VLLoader` 会给 `model.model.forward` 加窄补丁：仅当 SwiftVLN 已经把视觉特征注入 `inputs_embeds` 且没有原始 pixel media 时，直接进入 Qwen3 language stack
    - 2026-05-12 在 98 上验证 Qwen3-VL 2B SatNav smoke 时，默认
      `USE_LIGER_KERNEL=true` 会在 Qwen3 vision RoPE 的 Triton kernel 编译阶段报
      `numel exceeds triton maximum tensor numel`；Qwen3 smoke/训练需显式
      `USE_LIGER_KERNEL=false`
  - 脚本选择：
    - `MODEL_FAMILY=qwen2_5_vl`（默认） -> `MODEL_TYPE=swiftvln_qwen2_5_vl`
    - `MODEL_FAMILY=qwen3_vl` -> `MODEL_TYPE=swiftvln_qwen3_vl`
    - Qwen3 默认 base model：
      `/mnt/data1/home/jiangjiajun/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct`
    - Qwen3 8B 可通过 `STAGE1_MODEL_PATH=/mnt/data1/home/jiangjiajun/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct` 覆盖
  - 实验命名：
    - Qwen2.5 保持旧格式，不额外加 family tag
    - Qwen3 名字包含 `qwen3vl-`，例如 `swiftvln-satnav-qwen3vl-2b-...`
  - `eval_by_name.sh` 会从模型名中的 `qwen3vl` 解析 `MODEL_FAMILY=qwen3_vl`
  - README 当前只保留 conda 环境安装与 smoke 命令
  - Qwen3.5 仍未注册为可运行模型；当前 update 环境缺少 `transformers.models.qwen3_5`，后续必须使用独立环境
  - Qwen3 2B 下载/同步状态：
    - 2026-05-01 98 上 `Qwen3-VL-2B-Instruct` 已下载完成并同步到 73：
      `/mnt/data1/home/jiangjiajun/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct`
    - 73 上已验证 `swiftvln_qwen3_vl` + `load_model=False` 可正常加载 `Qwen3VLProcessor` / `SwiftVLNQwen3VLTemplate`
    - 同目录存在 Hugging Face 下载残留 `.cache/huggingface/download/*.incomplete`；实际加载所需 `model.safetensors` 与 `config.json` 已和 98 hash 一致
  - 已验证 smoke：
  - 2026-05-12 98 机 Qwen2.5 3B / Qwen3 2B SatNav smoke（产物已按用户要求从
    `output/swiftvln` 与 `results/eval/swiftvln` 清理）：
    - Qwen2.5 3B train：
      `MAX_SAMPLES=16 SAVE_STEPS=1 SAVE_TOTAL_LIMIT=1 TRAIN_NUM_GPUS=2`
    - Qwen2.5 3B output：
      `output/swiftvln/swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-b1.0-pool-s2-noembed-bs16-lr2e-5-090746/v0-20260512-090755/checkpoint-1`
    - Qwen3 2B train：
      `MODEL_FAMILY=qwen3_vl USE_LIGER_KERNEL=false MAX_SAMPLES=16 SAVE_STEPS=1 SAVE_TOTAL_LIMIT=1 TRAIN_NUM_GPUS=2`
    - Qwen3 2B output：
      `output/swiftvln/swiftvln-satnav-qwen3vl-2b-1ep-f32s4-overlap0-pf-h8-b1.0-pool-s2-noembed-bs16-lr2e-5-091236/v0-20260512-091245/checkpoint-1`
    - 两个模型均已用 `eval_by_name.sh` 完成 `val_seen` + `val_unseen`
      各 1 episode smoke，并写出 `evaluation_summary.json`
  - 2026-05-12 98 机 Habitat smoke（当前代码，SatNav smoke 清理后重跑）：
    - Qwen2.5 3B train：
      `VLN_ENV_TYPE=habitat MAX_SAMPLES=16 SAVE_STEPS=1 SAVE_TOTAL_LIMIT=1 TRAIN_NUM_GPUS=2`
    - Qwen2.5 3B output：
      `output/swiftvln/swiftvln-habitat-3b-1ep-f32s4-overlap0-pf-h8-b1.0-pool-s2-noembed-bs16-lr2e-5-093946/v0-20260512-093955/checkpoint-1`
    - Qwen2.5 3B eval：
      `ENV_TYPE=habitat EVAL_SPLIT=val_unseen MAX_EPISODES=1 CUDA_DEVICES=0 SAVE_VIDEO=false`
      -> `results/eval/swiftvln/swiftvln-habitat-3b-1ep-f32s4-overlap0-pf-h8-b1.0-pool-s2-noembed-bs16-lr2e-5-093946/val_unseen/20260512_094211/evaluation_summary.json`
    - Qwen3 2B train：
      `MODEL_FAMILY=qwen3_vl USE_LIGER_KERNEL=false VLN_ENV_TYPE=habitat MAX_SAMPLES=16 SAVE_STEPS=1 SAVE_TOTAL_LIMIT=1 TRAIN_NUM_GPUS=2`
    - Qwen3 2B output：
      `output/swiftvln/swiftvln-habitat-qwen3vl-2b-1ep-f32s4-overlap0-pf-h8-b1.0-pool-s2-noembed-bs16-lr2e-5-093641/v0-20260512-093650/checkpoint-1`
    - Qwen3 2B eval：
      `ENV_TYPE=habitat EVAL_SPLIT=val_unseen MAX_EPISODES=1 CUDA_DEVICES=0 SAVE_VIDEO=false`
      -> `results/eval/swiftvln/swiftvln-habitat-qwen3vl-2b-1ep-f32s4-overlap0-pf-h8-b1.0-pool-s2-noembed-bs16-lr2e-5-093641/val_unseen/20260512_093838/evaluation_summary.json`
  - 2026-05-01 Qwen2.5 73 机 train/eval smoke：
    - train：`MODEL_FAMILY=qwen2_5_vl MAX_SAMPLES=16 MAX_STEPS=2 TRAIN_NUM_GPUS=2`
    - loss：step1 `1.28262424` -> step2 `0.92245096`，有限、无 NaN/inf
    - eval：`MODEL_FAMILY=qwen2_5_vl MAX_EPISODES=1 CUDA_DEVICES=0,1 EVAL_SPLIT=val_seen`
    - summary：`total_episodes=1`，`world_size=2`
    - 本轮 smoke output/results/log 已清理
  - 2026-05-01 Qwen3-VL 8B 73 机 train/eval smoke：
    - base model：`/mnt/data1/home/jiangjiajun/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct`
    - train：`MODEL_FAMILY=qwen3_vl NUM_FRAMES=8 NUM_HISTORY=2 NUM_FUTURE_STEPS=2 MAX_SAMPLES=8 MAX_STEPS=1 TRAIN_NUM_GPUS=8`
    - loss：step1 `5.48239994`，有限、无 NaN/inf；单步约 245s，保存后总 runtime 约 478s
    - eval：`MODEL_FAMILY=qwen3_vl NUM_FRAMES=8 NUM_HISTORY=2 NUM_FUTURE_STEPS=2 MAX_EPISODES=1 CUDA_DEVICES=0,1 EVAL_SPLIT=val_seen`
    - summary：`total_episodes=1`，`world_size=2`
    - 本轮 smoke output/results/log 已清理
  - 73 机 8 卡 train smoke：
    - `MAX_SAMPLES=128 MAX_STEPS=2 SAVE_STEPS=1 SAVE_TOTAL_LIMIT=1`
    - 输出：`output/swiftvln/swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-b1.0-pool-s2-noembed-bs64-lr2e-5-173751/v0-20260430-173826/checkpoint-2`
    - 训练日志：`/tmp/smoke_swiftvln_ms_swift4_direct_train.log`
    - loss：step1 `1.22701669` -> step2 `1.15895748`，有限、无 NaN/inf
  - 73 机 eval smoke：
    - `MAX_EPISODES=2 EVAL_SPLIT=val_seen CUDA_DEVICES=0,1`
    - 结果：`results/eval/swiftvln/swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-b1.0-pool-s2-noembed-bs64-lr2e-5-173751/val_seen/20260430_174342/evaluation_summary.json`
    - eval 日志：`/tmp/smoke_swiftvln_ms_swift4_direct_eval.log`
    - summary：`total_episodes=2`，`world_size=2`

## Commit Style

- 使用 conventional commit：`feat/fix/refactor/docs/test/perf/chore`
- commit message 优先简洁中文
- 保持原子提交，避免混入无关改动

## Webhook

- 企业微信 / Codex webhook 不在仓库文档中硬编码。
- 如需发送 webhook 通知，必须由调用方通过环境变量或外部 secret 注入 URL。

## Operating Rules for Codex

- 优先使用现有脚本，不拼装临时一次性流程。
- 高风险步骤（远程/资源密集）需先说明预期和阻塞点。
- 除非用户明确要求，不创建 commit。
- 不回滚用户已有的无关改动。
- 默认失败策略：单项失败可继续后续队列项，并记录失败原因。
