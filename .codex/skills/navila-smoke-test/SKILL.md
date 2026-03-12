---
name: navila-smoke-test
description: "Run NaVILA baseline smoke tests on SatNav: 8-GPU train smoke on server 98, optional eval smoke from the saved output, and verification that train/eval artifacts are usable."
---

# NaVILA Smoke Test

Use this skill when user asks for `navila` smoke test, `NaVILA` baseline 冒烟测试, or requests checking whether `baseline/navila` can still train and evaluate on SatNav after a change.

## Goal

Validate that `baseline/navila` can:

- run an 8-GPU SatNav smoke train on server `98`,
- save a usable merged model under `output/`,
- optionally hand that output to eval,
- and confirm losses are finite with no launcher/model traceback.

## Fixed Paths

| Purpose | Path |
|---|---|
| Train launcher | `baseline/navila/scripts/train_satnav.sh` |
| Eval launcher | `baseline/navila/scripts/eval_satnav.sh` |
| Train entry | `baseline/navila/src/train_satnav.py` |
| Eval entry | `baseline/navila/src/eval_satnav.py` |
| Base pretrain model | `baseline/navila/model/navila-siglip-llama3-8b-v1.5-pretrain` |
| Smoke annotations | `baseline/smoke_test_data/annotations.json` |
| Smoke image root | `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260306/trajectory_data` |
| Smoke train outputs | `output/navila-baseline/smoketest/<run_name>/` |
| Smoke eval results | `results/navila-baseline/smoketest/<run_name>/<split>/` |
| Shared repo root | `/mnt/data1/home/jiangjiajun/workspace/SwiftVLN` |
| Upstream repo | `/mnt/data1/home/jiangjiajun/workspace/NaVILA` |

## Environment

- Preferred server: `98` (local host)
- Conda env: `navila-baseline`
- Train smoke must use `8` GPUs.
- Eval smoke should use `1` GPU unless user explicitly asks for multi-GPU eval.
- Prefer `ZeRO-2` for smoke because it matched the current stable path during local validation.

## Rules

- Do not invent ad-hoc Python launchers when the fixed shell scripts already cover the flow.
- Do not use the full dataset for smoke; use `baseline/smoke_test_data/annotations.json`.
- Do not use aggressive default train params for smoke. Override them explicitly.
- Treat merged output dir `output/navila-baseline/smoketest/<run_name>/` as the eval input when it contains `config.json`, `llm/`, `vision_tower/`, and `mm_projector/`.
- If train fails with CUDA/NCCL errors, report that as a stability failure, not a data-format failure.

## Step 1 — Preflight

Before running anything:

- confirm the fixed train/eval scripts exist;
- confirm `baseline/smoke_test_data/annotations.json` exists;
- confirm `baseline/navila/model/navila-siglip-llama3-8b-v1.5-pretrain` exists;
- confirm `nvidia-smi -L` on server 98 shows 8 GPUs;
- confirm `conda activate navila-baseline` works on server 98.

Stop immediately if any item fails.

## Step 2 — Train Smoke

Always use an explicit smoke subpath.

Recommended command:

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate navila-baseline
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN

DATA_PATH=/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/baseline/smoke_test_data/annotations.json \
IMAGE_FOLDER=/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260306/trajectory_data \
DS_CONFIG=/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/baseline/navila/configs/zero2.json \
NUM_GPUS=8 \
TRAIN_BSZ=1 \
GRAD_ACCUM=1 \
MAX_STEPS=8 \
SAVE_STEPS=8 \
SAVE_TOTAL_LIMIT=1 \
REPORT_TO=none \
DATALOADER_WORKERS=0 \
LEARNING_RATE=5e-5 \
MASTER_PORT=29640 \
bash baseline/navila/scripts/train_satnav.sh smoketest/<run_name>
```

Required checks after train:

- `output/navila-baseline/smoketest/<run_name>/config.json` exists;
- `output/navila-baseline/smoketest/<run_name>/llm/` exists;
- log contains finite step losses;
- log contains no uncaught `Traceback`.

## Step 3 — Eval Smoke

Run eval from the merged smoke output dir.

Recommended command:

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate navila-baseline
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN

bash baseline/navila/scripts/eval_satnav.sh smoketest/<run_name> val_seen 1 5
```

Required checks after eval:

- `results/navila-baseline/smoketest/<run_name>/<split>/evaluation_summary.json` exists;
- eval log reaches `[OK] Evaluation completed!`;
- eval input resolves to the smoke output dir rather than a base model path.

## Step 4 — Report

Report all of the following:

- exact train command and eval command;
- smoke run name;
- train step losses and final `train_loss`;
- whether merged model artifacts were written under `output/`;
- eval summary path and whether eval completed;
- residual risk if smoke covered only train or only a very small eval subset.
