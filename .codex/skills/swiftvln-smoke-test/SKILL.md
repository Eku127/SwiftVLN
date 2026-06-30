---
name: swiftvln-smoke-test
description: "Run SwiftVLN swiftvln SatNav multi-GPU smoke tests, verifying train -> checkpoint -> eval end to end."
---

# SwiftVLN Smoke Test Skill

Use this skill when the user asks for主线 SwiftVLN 的 smoke test / 冒烟测试 / train+eval 快速打通验证。

## Scope

- 主线模型仅限：`swiftvln`
- 环境仅限：`SatNav`
- 目标：最小代价验证当前代码仍能
  - 启动训练
  - 产出 checkpoint
  - 从该 checkpoint 完成评测
  - 写出 `evaluation_summary.json`

## Fixed Paths

| Purpose | Path |
|---|---|
| SwiftVLN train | `src/swiftvln/model/script/train/train_swiftvln_qwen_vl.sh` |
| Eval entry | `src/swiftvln/scripts/eval/eval_by_name.sh` |
| Eval queue | `src/swiftvln/scripts/eval/eval_queue.sh` |
| Enqueue eval | `src/swiftvln/scripts/eval/enqueue_eval.sh` |
| Eval worker | `src/swiftvln/scripts/eval/start_eval_worker.sh` |
| Eval monitor | `src/swiftvln/scripts/eval/start_eval_monitor.sh` |

## Fixed Environments

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate swift-vln-train
conda activate swift-vln-eval
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN-refactor
```

## Preflight

Before touching anything, verify:

- repo root exists
- fixed script paths above exist
- `swift-vln-train` and `swift-vln-eval` can activate
- target SatNav dataset path exists
- `nvidia-smi` sees the planned GPUs

If any item fails, stop and report the exact missing path / dependency.

## Smoke Values

`train_swiftvln_qwen_vl.sh` 的仓库默认值是正式训练口径；smoke 必须显式覆盖为下面这些值：

| Variable | Smoke Value |
|---|---|
| `VLN_ENV_TYPE` | `satnav` |
| `MAX_SAMPLES` | `16` |
| `SAVE_STEPS` | `1` |
| `SAVE_TOTAL_LIMIT` | `1` |
| `USE_SWANLAB` | `false` |

## Train Smoke

推荐 2 卡：

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate swift-vln-train
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN-refactor

MAX_SAMPLES=16 SAVE_STEPS=1 SAVE_TOTAL_LIMIT=1 \
USE_SWANLAB=false \
TRAIN_NUM_GPUS=2 \
  bash src/swiftvln/model/script/train/train_swiftvln_qwen_vl.sh \
  2>&1 | tee /tmp/smoke_swiftvln_train.log
```

训练完成后确认：

- `output/swiftvln/<exp_name>/v0-*/checkpoint-*` 存在
- checkpoint 下至少有 `config.json`
- 且存在 `.safetensors` 或 `.bin`
- 日志中没有 `Traceback` / `RuntimeError` / `torchrun` 启动失败

## Eval Smoke

默认同时跑 `val_seen + val_unseen`，每个 split 各 `10` 个 episode：

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate swift-vln-eval
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN-refactor

MAX_EPISODES=10 ENV_TYPE=satnav CUDA_DEVICES=0,1 \
  bash src/swiftvln/scripts/eval/eval_by_name.sh <swiftvln_exp_name> \
  2>&1 | tee /tmp/smoke_swiftvln_eval.log
```

确认结果：

- `results/eval/swiftvln/<exp_name>/val_seen/<timestamp>/evaluation_summary.json`
- `results/eval/swiftvln/<exp_name>/val_unseen/<timestamp>/evaluation_summary.json`
- eval log 无 fatal error

## Cleanup

仅在用户明确要求清理 smoke 产物时执行，且只删除本次 smoke 对应目录：

```bash
rm -rf output/swiftvln/<smoke_exp_name>
rm -rf results/eval/swiftvln/<smoke_exp_name>
```

## Report Checklist

至少汇报：

- 使用的 train / eval 命令
- train log 路径
- eval log 路径
- 输出 checkpoint 路径
- `val_seen` / `val_unseen` summary 路径
- 关键指标：`success_rate`, `mean_spl`, `oracle_success`, `navigation_error`, `avg_steps`
