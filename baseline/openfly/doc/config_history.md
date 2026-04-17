# OpenFly SatNav Config History

## Config v1 (deprecated — SR=0.0)

### Train

- `OPENFLY_BACKEND=continue`
- `OPENFLY_ACTION_FORMAT=original`
- `DATA_PATH=/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260404/trajectory_data/annotations.json`
- `IMAGE_FOLDER=/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260404/trajectory_data`
- `NUM_GPUS=8`
- `TRAIN_BSZ=12`
- `GRAD_ACCUM=1`
- `NUM_EPOCHS=1`
- `LEARNING_RATE=2e-5`
- `SAVE_STEPS=10000`
- `LOGGING_STEPS=20`
- `TORCH_DTYPE=bfloat16`
- `DEEPSPEED_MODE=zero2`
- `USE_FLASH_ATTENTION_2=true`
- `GRID_SIZE=16`

### Sampling

- `SATNAV_HEAD_KEEP=7`
- `SATNAV_SAMPLE_STRIDE=7`
- `SATNAV_STOP_REPEAT=6`
- `SATNAV_STOP_HISTORY_AUG=3`

### Known Issues

- 8D `original` action format with 4 unsupervised trailing dimensions added noise to training
- No action history in prompt → model had no temporal memory
- No per-dimension loss reweighting
- Result: SR=0.0 across all checkpoints (10k, 20k, 33430 steps)

---

## Config v2 (deprecated — with stop_window=2)

### Train

- `OPENFLY_BACKEND=continue`
- `OPENFLY_ACTION_FORMAT=original`
- same optimizer/hardware settings as v1

### Sampling

- `SATNAV_HEAD_KEEP=7`
- `SATNAV_SAMPLE_STRIDE=3`
- `SATNAV_STOP_REPEAT=2`
- `SATNAV_STOP_WINDOW=2`
- `SATNAV_TAIL_KEEP=5`
- `SATNAV_STOP_HISTORY_AUG=1`

### Original Supervision

- supervise only the first `4` active action-dimension tokens
- inactive trailing action tokens and final `eos` are masked from loss
- weighted CE over the first `4` action-dimension tokens:
  - `OPENFLY_ORIGINAL_DIM_LOSS_WEIGHTS=0.4,1.2,1.2,1.2`

### Known Issues

- `SATNAV_STOP_WINDOW=2` relabels near-goal steps as `stop` during training,
  creating a strong premature-stop bias — model stops as soon as it visually
  resembles the goal region, not when it actually reaches it
- No action history in prompt
- Result: SR=0.0 (Boundary oracle_success high ~0.47, but SR=0 — model reached
  goal area then failed to stop correctly)

### 0404 Sample Distribution (stop_window=2)

- total: `3,929,473`
- `stop=419,816` (`10.68%`)
- `forward=2,107,790` (`53.64%`)
- `left=730,805` (`18.60%`)
- `right=671,062` (`17.08%`)

---

## Current Config (v3) — default as of 2026-04-17

This is the production-default configuration. All new training runs use these
settings unless overridden via environment variables.

### Key Changes vs v2

| Parameter | v2 | v3 | Rationale |
|---|---|---|---|
| `OPENFLY_ACTION_FORMAT` | `original` | **`compact`** | Removes 8D vector noise; simpler output space |
| `SATNAV_STOP_WINDOW` | `2` | **`0`** | Teach stop only at the true end; eliminates premature-stop bias |
| `OPENFLY_ACTION_HISTORY_LIMIT` | — | **`16`** | Past-action clause in prompt gives model temporal memory |

### Train

- `OPENFLY_BACKEND=continue`
- `OPENFLY_ACTION_FORMAT=compact`
- `DATA_PATH=/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260404/trajectory_data/annotations.json`
- `IMAGE_FOLDER=/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260404/trajectory_data`
- `NUM_GPUS=8`
- `TRAIN_BSZ=12`
- `GRAD_ACCUM=1`
- `NUM_EPOCHS=1`
- `LEARNING_RATE=2e-5`
- `SAVE_STEPS=10000`
- `LOGGING_STEPS=20`
- `TORCH_DTYPE=bfloat16`
- `DEEPSPEED_MODE=zero2`
- `USE_FLASH_ATTENTION_2=true`
- `GRID_SIZE=16`
- `WEIGHT_DECAY=0.0`
- `WARMUP_RATIO=0.0`
- `LR_SCHEDULER_TYPE=linear`
- `MAX_GRAD_NORM=1.0`
- `DATALOADER_NUM_WORKERS=4`
- `REPORT_TO=none`

### Sampling

- `SATNAV_HEAD_KEEP=7`
- `SATNAV_SAMPLE_STRIDE=3`
- `SATNAV_STOP_REPEAT=2`
- `SATNAV_STOP_WINDOW=0`
- `SATNAV_TAIL_KEEP=5`
- `SATNAV_STOP_HISTORY_AUG=1`

### Prompt

- `OPENFLY_ACTION_HISTORY_LIMIT=16`
- Format: `What action should the robot take to {instruction}? Past actions (N so far): forward, left, ...`
- At step 0 (no prior actions), the clause reads: `Past actions: none.`

### 0404 Sample Distribution (stop_window=0)

- total: `3,824,519`
- `stop=209,908` (`5.49%`)
- `forward=2,212,744` (`57.85%`)
- `left=730,805` (`19.10%`)
- `right=671,062` (`17.54%`)

### Backend Compatibility

Both `continue` and `scratch` backends support both `compact` and `original` action formats.

- `continue` default: `OPENFLY_BACKEND=continue` → loads `openfly-agent-7b` HF checkpoint
- `scratch` default: `OPENFLY_BACKEND=scratch` → initializes from OpenVLA Prismatic `.pt` checkpoint

### Eval Results (val_seen, ver_260404, 6338 episodes)

| Ckpt (steps) | Overall SR | Overall SPL | Boundary SR | Road SR | LandmarkSet SR | OS |
|---|---|---|---|---|---|---|
| 6000 | 11.94% | 11.92% | 15.28% | 18.18% | 5.59% | 22.81% |
| 8000 | **12.28%** | **12.25%** | **17.36%** | **18.08%** | **5.23%** | **23.27%** |

- 8k steps = 0.2 epoch; Boundary SR still rising (+2pp per 2k steps), Road/LandmarkSet plateaued
- Premature stop (final action=stop on failed episodes): ~85% of failures
- Boundary near-failures (<30m from goal): 316 (20%) — recoverable with more training
- LandmarkSet failure pattern: template trajectory shape (no visual landmark search) — architectural limit of 3-frame context
- Runtime (8×H100, ~1.16 s/step): smoke run 8k steps ≈ 2h40m; full 1 epoch ~9h

### Throughput

- 8×H100, bfloat16, flash_attention2, DeepSpeed Zero-2
- ~1.15–1.20 s/step
- Full epoch (33 430 steps): ~10–11h
