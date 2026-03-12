---
name: uninavid-smoke-test
description: "Run Uni-NaVid baseline smoke tests on SatNav: 8-GPU train smoke, checkpoint handoff to eval smoke, feature-change pre/post loss comparison, and mandatory cleanup of smoke outputs/results."
---

# Uni-NaVid Smoke Test

Use this skill when user asks for `uninavid` smoke test, `Uni-NaVid` baseline 冒烟测试, baseline train+eval quick validation, or requests checking whether a new feature keeps loss behavior reasonable before and after a change.

## Goal

Validate that `baseline/uninavid` can:

- run an 8-GPU SatNav smoke train,
- save a usable checkpoint under `output/`,
- hand the checkpoint to eval,
- write eval outputs under `results/`,
- compare pre/post loss when a feature change is involved,
- and delete smoke artifacts after verification.

## Fixed Paths

| Purpose | Path |
|---|---|
| Train launcher | `baseline/uninavid/scripts/train_satnav.sh` |
| Eval launcher | `baseline/uninavid/scripts/eval_satnav.sh` |
| Train entry | `baseline/uninavid/src/train_satnav.py` |
| Eval entry | `baseline/uninavid/src/eval_satnav.py` |
| Base model | `baseline/uninavid/model/Uni-Navid` |
| Smoke annotations | `baseline/smoke_test_data/annotations.json` |
| Smoke outputs | `output/uninavid-baseline/smoketest/<run_name>/` |
| Smoke results | `results/uninavid-baseline/smoketest/<run_name>/<split>/` |
| Shared repo root | `/mnt/data1/home/jiangjiajun/workspace/SwiftVLN` |
| Upstream repo | `/mnt/data1/home/jiangjiajun/workspace/Uni-NaVid` |

## Environment

- Preferred server: `73` (`ssh 10.246.152.73`)
- Conda env: `uninavid-baseline`
- Train smoke must use `8` GPUs.
- Eval smoke may use `1` GPU unless user explicitly asks for multi-GPU eval.
- If a previous distributed smoke was interrupted, clear stale Uni-NaVid train processes on server 73 and rerun with a fresh `MASTER_PORT`.

## Rules

- Do not invent ad-hoc Python launchers when the fixed shell scripts already cover the flow.
- Do not use full dataset or long training for smoke.
- Do not leave smoke checkpoints or results behind after verification.
- Do not delete non-smoke outputs.
- During eval, SatNav may emit caught `Camera view bounds exceed image bounds` tracebacks on weak smoke checkpoints. Treat this as a known environment-side boundary failure only if eval still completes and writes `evaluation_summary.json`.
- If a feature modifies training logic, run pre-change and post-change smoke with the same seed and the same smoke dataset, then compare step losses and final `train_loss`.
- For feature validation, losses should be either:
  - exactly identical when the feature should be behavior-preserving, or
  - very close when the feature changes kernels or numerics only (for example flash-attn vs standard attention).

## Step 1 — Preflight

Before running anything:

- confirm the fixed train/eval scripts exist;
- confirm `baseline/smoke_test_data/annotations.json` exists;
- confirm `baseline/uninavid/model/Uni-Navid` exists;
- confirm `nvidia-smi -L` on server 73 shows 8 GPUs;
- confirm `conda activate uninavid-baseline` works on server 73.

Stop immediately if any item fails.

## Step 2 — Train Smoke

Always use an explicit smoke subpath so artifacts are easy to delete.

If a previous smoke run failed or was interrupted, do this before rerun:

```bash
ssh 10.246.152.73 "ps -eo pid,cmd | grep -E 'deepspeed.launcher.launch|baseline/uninavid/src/train_satnav.py' | grep -v grep || true"
```

If any stale Uni-NaVid train process remains, terminate it before relaunching. Then use a fresh master port, for example `MASTER_PORT=29618`.

Example:

```bash
ssh 10.246.152.73 '
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate uninavid-baseline
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN

DATA_PATH=/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/baseline/smoke_test_data/annotations.json \
NUM_GPUS=8 \
TRAIN_BSZ=1 \
EVAL_BSZ=1 \
GRAD_ACCUM=1 \
MAX_STEPS=8 \
SAVE_STRATEGY=steps \
SAVE_STEPS=8 \
SAVE_TOTAL_LIMIT=1 \
REPORT_TO=none \
DATALOADER_WORKERS=0 \
MODEL_MAX_LENGTH=1536 \
SEED=1234 \
MASTER_PORT=29618 \
UNINAVID_USE_FLASH_ATTN=1 \
bash baseline/uninavid/scripts/train_satnav.sh smoketest/<run_name>
'
```

Required checks after train:

- `output/uninavid-baseline/smoketest/<run_name>/checkpoint-*` exists;
- log contains finite step losses;
- log contains no `Traceback`;
- if this run is for behavior-preserving feature validation, record every printed step loss and final `train_loss`.

## Step 3 — Eval Smoke

Run eval from the smoke checkpoint by name or subpath.

Example:

```bash
ssh 10.246.152.73 '
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate uninavid-baseline
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN

bash baseline/uninavid/scripts/eval_satnav.sh smoketest/<run_name> val_seen 1 5
'
```

Required checks after eval:

- `results/uninavid-baseline/smoketest/<run_name>/<split>/evaluation_summary.json` exists;
- eval log reaches final summary and `[OK] Evaluation completed!`;
- checkpoint handoff happened from the smoke output, not from a base model path.
- if tracebacks appear, confirm they are the known caught SatNav boundary error rather than an uncaught launcher/model failure.

## Step 4 — Feature Change Validation

Use this when user asks whether a new feature keeps training behavior normal.

Procedure:

1. run smoke train on the pre-change code with fixed seed and fixed smoke data;
2. apply the feature change;
3. run the same smoke train again with the same seed and config;
4. compare:
   - step losses,
   - final `train_loss`,
   - presence/absence of warnings relevant to the feature.

Interpretation:

- Behavior-preserving fixes such as label-length accounting or warning cleanup should keep losses identical.
- Kernel/runtime swaps such as flash-attn may produce slightly different losses, but they should stay close and stable, without NaN, divergence, or obvious regression.

## Step 5 — Cleanup

Cleanup is mandatory after smoke verification succeeds or fails.

Delete only the confirmed smoke paths:

```bash
rm -rf output/uninavid-baseline/smoketest/<run_name>
rm -rf results/uninavid-baseline/smoketest/<run_name>
```

If temporary logs were written outside those directories, delete them too.

## Step 6 — Report

Report all of the following:

- exact train command and eval command;
- smoke run name;
- checkpoint path used for eval;
- step losses and final `train_loss` when applicable;
- eval summary path and key metrics;
- whether artifacts were cleaned up successfully;
- residual risk, if smoke did not cover it.
