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

- 训练队列：`src/swiftvln/scripts/train/train_queue.sh`
- 评测单模型：`src/swiftvln/scripts/eval/eval_by_name.sh`
- 评测队列：`src/swiftvln/scripts/eval/eval_queue.sh`
- 评测入队：`src/swiftvln/scripts/eval/enqueue_eval.sh`
- 评测 worker：`src/swiftvln/scripts/eval/start_eval_worker.sh`
- 评测 monitor：`src/swiftvln/scripts/eval/start_eval_monitor.sh`
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

## Eval Queue Path Convention

评测队列文件统一放在：

- `runtime/eval_queue/eval_todo.txt`
- `runtime/eval_queue/eval_done.txt`
- `runtime/eval_queue/eval_failed_todo.txt`

注意：不再使用旧路径 `src/swiftvln/scripts/eval/*.txt`。

## Current SatNav Dataset Defaults

- Dataset root: `/mnt/data3/jiangjiajun/dataset/satnav_datasets`
- 当前常用版本：`ver_260306`
- Eval episodes:
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260306/episodes/eval/all_episodes.json`
- QA JSONL:
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260306/data/qa_swift.jsonl`
- Trajectory data:
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260306/trajectory_data`
- Scene maps:
  - active: `/mnt/data3/jiangjiajun/dataset/satnav_datasets/scenes`
  - backup(old): `/mnt/data3/jiangjiajun/dataset/satnav_datasets/old_scenes`

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
