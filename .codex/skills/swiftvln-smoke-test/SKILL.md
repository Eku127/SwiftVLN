---
name: swiftvln-smoke-test
description: "Run SwiftVLN SatNav-only multi-GPU smoke tests for StreamVLN/OverlapVLN baselines, ensuring checkpoint+eval verification and post-test cleanup."
---

# SwiftVLN Smoke Test Skill

Use this skill when user asks for: smoke test, 冒烟测试, 快速验证 train/eval 是否可跑通, or requests minimal end-to-end verification in this repo.

## Goal

Quickly verify that the current code version can still start training, produce a checkpoint, run evaluation from it, and write expected outputs — without running expensive full experiments.

**Coverage: StreamVLN baseline + OverlapVLN baseline, SatNav only, multi-card.**

---

## Constraints (read before doing anything)

**Must NOT:**
- Re-download model weights or datasets if paths already exist.
- Run full-scale long training as smoke test.
- Change production config/script defaults permanently.
- Modify dataset version/path blindly.
- Delete historical/non-smoke outputs, checkpoints, or logs. For confirmed smoke artifacts, use `rm -rf` directly — do NOT stage to a tmp/cleanup directory.
- Use destructive git/file commands unless explicitly requested.
- Use `find/rg/ls` to locate scripts — use fixed paths in this skill.
- Leave SwanLab cloud uploads or WXWork notifications active during smoke.
- Skip restoring patched script variables after smoke completes.

**Fixed paths (do not search for these):**

| Purpose | Path |
|---|---|
| StreamVLN train | `src/swiftvln/models/streamvln/script/train/train_streamvln_qwen2_5_vl_single_node.sh` |
| OverlapVLN train | `src/swiftvln/models/overlapvln/script/train/train_overlapvln_qwen2_5_vl.sh` |
| Eval entry | `src/swiftvln/scripts/eval/eval_by_name.sh` |
| Eval queue | `src/swiftvln/scripts/eval/eval_queue.sh` |
| Enqueue eval | `src/swiftvln/scripts/eval/enqueue_eval.sh` |
| Eval worker | `src/swiftvln/scripts/eval/start_eval_worker.sh` |
| Eval monitor | `src/swiftvln/scripts/eval/start_eval_monitor.sh` |

**Fixed conda setup:**

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate swift-vln-train   # for training
conda activate swift-vln-eval    # for evaluation
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN
```

---

## Step 1 — Preflight

Check all of the following before touching any script. If any item fails, **stop and report the exact missing item/path**.

- [ ] Repo root exists: `/mnt/data1/home/jiangjiajun/workspace/SwiftVLN`
- [ ] All fixed script paths listed above exist on disk
- [ ] Conda init script exists: `/mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh`
- [ ] Conda envs are activatable: `swift-vln-train`, `swift-vln-eval`
- [ ] SatNav dataset path check:
  - Open each training script and confirm `SATNAV_DATA_PATHS` uses `ver_260227`.
  - Cross-check with `CODEX_CONTEXT.md` → `Current SatNav Dataset Defaults`.
  - If version still differs (e.g., newer `ver_26XXXX` available), update the scripts permanently in this step — **do not defer to Step 2**.
- [ ] GPUs are visible (`nvidia-smi` shows expected devices)

---

## Step 2 — Patch Training Scripts for Smoke

Training scripts define key parameters as **internal shell variables** (not environment variables).
Apply the patches below to each training script before running. **Both scripts need the same patches.**

### Patch table

| Variable | Production default | Smoke value | Reason |
|---|---|---|---|
| `VLN_ENV_TYPE` | `habitat` | `satnav` | Smoke must use SatNav, not Habitat |
| `MAX_SAMPLES` | `0` (all) | `16` | Minimal data, just verify flow |
| `SAVE_STEPS` | `1000` | `1` | Ensure checkpoint is saved before training ends |
| `SAVE_TOTAL_LIMIT` | `1` | `1` | Keep only 1 checkpoint |
| `USE_SWANLAB` | `true` | `false` | No cloud tracking during smoke |
| `USE_WXWORK_NOTIFICATION` | `true` | `false` | No enterprise WeChat alerts during smoke |
| `SATNAV_DATA_PATHS` | `ver_260227/...` | `ver_260227/...` | Scripts already updated; verify they match current version |

### Patch commands — StreamVLN

```bash
SCRIPT_STREAM="src/swiftvln/models/streamvln/script/train/train_streamvln_qwen2_5_vl_single_node.sh"

sed -i 's/^VLN_ENV_TYPE=.*/VLN_ENV_TYPE="satnav"/' "$SCRIPT_STREAM"
sed -i 's/^MAX_SAMPLES=.*/MAX_SAMPLES="16"/' "$SCRIPT_STREAM"
sed -i 's/^SAVE_STEPS=.*/SAVE_STEPS=1/' "$SCRIPT_STREAM"
sed -i 's/^USE_SWANLAB=.*/USE_SWANLAB=false/' "$SCRIPT_STREAM"
sed -i 's/^USE_WXWORK_NOTIFICATION=.*/USE_WXWORK_NOTIFICATION=false/' "$SCRIPT_STREAM"
```

### Patch commands — OverlapVLN

```bash
SCRIPT_OVERLAP="src/swiftvln/models/overlapvln/script/train/train_overlapvln_qwen2_5_vl.sh"

sed -i 's/^VLN_ENV_TYPE=.*/VLN_ENV_TYPE="satnav"/' "$SCRIPT_OVERLAP"
sed -i 's/^MAX_SAMPLES=.*/MAX_SAMPLES="16"/' "$SCRIPT_OVERLAP"
sed -i 's/^SAVE_STEPS=.*/SAVE_STEPS=1/' "$SCRIPT_OVERLAP"
sed -i 's/^USE_SWANLAB=.*/USE_SWANLAB=false/' "$SCRIPT_OVERLAP"
sed -i 's/^USE_WXWORK_NOTIFICATION=.*/USE_WXWORK_NOTIFICATION=false/' "$SCRIPT_OVERLAP"
```

> After patching, **record the expected `OUTPUT_DIR`** for each model.
> The `OUTPUT_DIR` is printed by the script when it starts. Capture it before training ends.
> You will need these paths for Step 5 (cleanup).

---

## Step 3 — Run Training Smoke

Run each model sequentially (or one at a time). Multi-card only — no single-card smoke.

### StreamVLN

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate swift-vln-train
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN

bash src/swiftvln/models/streamvln/script/train/train_streamvln_qwen2_5_vl_single_node.sh \
  2>&1 | tee /tmp/smoke_streamvln_train.log
```

### OverlapVLN

```bash
bash src/swiftvln/models/overlapvln/script/train/train_overlapvln_qwen2_5_vl.sh \
  2>&1 | tee /tmp/smoke_overlapvln_train.log
```

### Verify training artifacts (per model)

- [ ] `output/<arch>/<exp_name>/v0-*/checkpoint-*` directory exists
- [ ] Checkpoint contains `config.json` and `.safetensors` or `.bin` weight files
- [ ] Log (`/tmp/smoke_*_train.log`) has no `Traceback`, `RuntimeError`, or launcher failure

If verification fails: record failure reason, **still run restore in Step 4**, then proceed to report.

---

## Step 4 — Restore Training Scripts (run regardless of pass/fail)

**This step is mandatory even if training failed.**

### Restore — StreamVLN

```bash
sed -i 's/^VLN_ENV_TYPE=.*/VLN_ENV_TYPE="habitat"/' "$SCRIPT_STREAM"
sed -i 's/^MAX_SAMPLES=.*/MAX_SAMPLES="0"/' "$SCRIPT_STREAM"
sed -i 's/^SAVE_STEPS=.*/SAVE_STEPS=1000/' "$SCRIPT_STREAM"
sed -i 's/^USE_SWANLAB=.*/USE_SWANLAB=true/' "$SCRIPT_STREAM"
sed -i 's/^USE_WXWORK_NOTIFICATION=.*/USE_WXWORK_NOTIFICATION=true/' "$SCRIPT_STREAM"
```

### Restore — OverlapVLN

```bash
sed -i 's/^VLN_ENV_TYPE=.*/VLN_ENV_TYPE="habitat"/' "$SCRIPT_OVERLAP"
sed -i 's/^MAX_SAMPLES=.*/MAX_SAMPLES="0"/' "$SCRIPT_OVERLAP"
sed -i 's/^SAVE_STEPS=.*/SAVE_STEPS=1000/' "$SCRIPT_OVERLAP"
sed -i 's/^USE_SWANLAB=.*/USE_SWANLAB=true/' "$SCRIPT_OVERLAP"
sed -i 's/^USE_WXWORK_NOTIFICATION=.*/USE_WXWORK_NOTIFICATION=true/' "$SCRIPT_OVERLAP"
```

> **Note on `SATNAV_DATA_PATHS`**: dataset version is **not** restored after smoke — it is a permanent production fix, not a smoke-only temporary change. If you updated it in Step 2, leave it as-is after smoke.

Confirm restore by spot-checking `grep VLN_ENV_TYPE` in each script — should show `habitat` again.

---

## Step 5 — Run Eval Smoke

Use `eval_by_name.sh` with the checkpoint paths recorded in Step 2/3. Limit episodes for speed.

### StreamVLN eval

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate swift-vln-eval
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN

# 不设置 EVAL_SPLIT → 默认同时跑 val_seen + val_unseen（每个 split 各 10 episodes）
MAX_EPISODES=10 ENV_TYPE=satnav \
  bash src/swiftvln/scripts/eval/eval_by_name.sh <streamvln_exp_name> \
  2>&1 | tee /tmp/smoke_streamvln_eval.log
```

### OverlapVLN eval

```bash
MAX_EPISODES=10 ENV_TYPE=satnav \
  bash src/swiftvln/scripts/eval/eval_by_name.sh <overlapvln_exp_name> \
  2>&1 | tee /tmp/smoke_overlapvln_eval.log
```

### Verify eval artifacts (per model)

- [ ] `results/eval/<arch>/<exp_name>/val_seen/<timestamp>/evaluation_summary.json` exists
- [ ] `results/eval/<arch>/<exp_name>/val_unseen/<timestamp>/evaluation_summary.json` exists
- [ ] Log has no `Traceback`, `RuntimeError`, or fatal errors

---

## Step 6 — Cleanup Smoke Artifacts

Use the paths recorded during Step 2/3 and **directly delete** — no intermediate moves, no tmp staging.

```bash
# Replace with actual recorded paths from Step 2/3
rm -rf output/streamvln/<smoke_exp_name>
rm -rf output/overlapvln/<smoke_exp_name>
rm -rf results/eval/streamvln/<smoke_exp_name>
rm -rf results/eval/overlapvln/<smoke_exp_name>
rm -f /tmp/smoke_*.log

# If queue entries were created, remove the specific lines:
sed -i "/<smoke_model_name>/d" runtime/eval_queue/eval_todo.txt
sed -i "/<smoke_model_name>/d" runtime/eval_queue/eval_done.txt
```

> **Do NOT use `mv` to a staging/tmp directory** — if the path is confirmed as a smoke artifact (recorded in Step 2/3), `rm -rf` directly. Moving to tmp just wastes space and leaves cleanup unfinished.

---

## Step 7 — Report

Provide the following at completion:

### Pass/Fail Matrix

| Model | Env | Split | Train | Checkpoint Path | Eval | Summary Path | Failure Reason |
|---|---|---|---|---|---|---|---|
| streamvln baseline | SatNav | val_seen | ✅/❌ | `output/...` | ✅/❌ | `results/.../val_seen/<ts>/` | — |
| | | val_unseen | — | | ✅/❌ | `results/.../val_unseen/<ts>/` | — |
| overlapvln baseline | SatNav | val_seen | ✅/❌ | `output/...` | ✅/❌ | `results/.../val_seen/<ts>/` | — |
| | | val_unseen | — | | ✅/❌ | `results/.../val_unseen/<ts>/` | — |

### Also include:
1. Exact commands or script entrypoints that were run
2. Paths to key logs/checkpoints/summaries
3. Residual risk (what was not covered by this smoke)
