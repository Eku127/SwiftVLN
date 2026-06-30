---
name: swiftvln-eval
description: "Run VLN evaluation by model name or queue using tmux-based execution. Supports queue consumption and CSV result collection."
---

# SwiftVLN Eval Skill

Evaluate trained SwiftVLN models by name or through the shared eval queue. Eval runs in tmux so it survives terminal disconnects.

Related skills:
- **`swiftvln-train`**: upstream producer — trains models and enqueues them to `eval_todo.txt`.
- **`server-train-eval-monitor`**: cluster-wide status overview across 98/73/17.

---

## Inputs

| Parameter | Default | Notes |
|---|---|---|
| Model name(s) | — | One or more model names to evaluate, or use queue-based consumption |
| Eval split | SatNav: unset means `val_seen` + `val_unseen`; Habitat: `val_unseen` | Can be overridden with `EVAL_SPLIT` |
| CUDA devices | `0,1,2,3,4,5,6,7` | GPU device list |
| Save video | `false` | Whether to save evaluation videos |
| Auto resume | `AUTO_RESUME_EVAL=true` | Reuse the latest incomplete result dir for the same model/split unless `OUTPUT_DIR` is set |
| Conda env | `swift-vln-eval` | Must be activated before running eval scripts |

---

## Eval Script Decision Tree

| Scenario | Script | Description |
|---|---|---|
| One-off eval of a single model | `eval_by_name.sh` | Run once and exit |
| Serial queue consumption | `eval_queue.sh` | Consume current queue entries and exit unless configured otherwise |
| Long-lived queue worker | `start_eval_worker.sh` | Keeps polling `eval_todo.txt`, waits for new tasks |

## Map-Memory Parsing Notes

- `eval_by_name.sh` supports parsing map-memory config from SwiftVLN experiment names.
- Naming block:
  - `map-g{global}-l{local}-r{render}-{mask}-s{compress_stride}`
  - Example: `map-g1000-l400-r384-d20-s2`
- Expected parsed values:
  - `MEMORY_METHOD=map`
  - `MAP_GLOBAL_SIDE_M`
  - `MAP_LOCAL_SIDE_M`
  - `MAP_RENDER_PX`
  - `MAP_MASK_METHOD`
  - `HISTORY_PROCESSOR_TYPE=per_frame`
  - `USE_TOME=false`
- Queue mode passes model names to `eval_by_name.sh`, so map parsing depends on the model name and `eval_by_name.sh`.

Before real eval, validate parsing when useful:

```bash
CHECK_ONLY=true bash src/swiftvln/scripts/eval/eval_by_name.sh <model_name>
```

For map experiments, confirm `MEMORY_METHOD=map` and the expected `MAP_*` values appear.

---

## Resume Behavior

- The unified eval script defaults to `AUTO_RESUME_EVAL=true`.
- If `OUTPUT_DIR` is not explicitly set, it reuses the newest incomplete
  `results/eval/swiftvln/<model>/<split>/<timestamp>/` directory that has
  partial resume evidence (`result.jsonl` or `.dist_sync`) and no
  `evaluation_summary.json`.
- Completed eval directories are not reused. Set `AUTO_RESUME_EVAL=false` to
  force a fresh timestamp directory, or set `OUTPUT_DIR=<path>` to resume or
  write to a specific directory.

---

## Model Zoo Names

- SwiftVLN HF upload-ready model directories live under
  `output/model_zoo/swiftvln/HF_model/`; `output/model_zoo/swiftvln/` should not
  contain old long-name model directories outside `HF_model`.
- Mainline `eval_by_name.sh <model_name>` resolves
  `output/model_zoo/swiftvln/HF_model/<model_name>` directly before falling
  back to old training-output `checkpoint-*` discovery.
- Current short-name SwiftVLN HF models:
  `swiftvln-satnav-3b-1ep-f32s4-overlap0-gtc-k512-noembed`,
  `swiftvln-satnav-3b-1ep-f32s4-overlap0-map-g1000-l400-r448-d20-s2-noembed`,
  `swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h0-nomem-pool-s2-noembed`,
  `swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-b2.0-pool-s2-noembed`,
  `swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-initial-noembed`,
  `swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed`,
  `swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-posefilm`,
  `swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-random-pool-s2-noembed`,
  `swiftvln-satnav-3b-1ep-f32s4-overlap0-sgtc-k512-noembed`,
  `swiftvln-satnav-3b-1ep-f32s4-overlap16-pf-h8-pool-s2-noembed`,
  `swiftvln-satnav-3b-1ep-f32s4-overlap4-pf-h8-pool-s2-noembed`.

---

## Step 1 → Check Servers & Pick Eval Host

1. Eval hosts allowed: **98 and 73 only**. Never run eval on 17.
2. Inspect GPU availability:
   - 98: local `nvidia-smi` plus `pgrep -af "torchrun|train_queue|eval_queue|eval_by_name"`
   - 73: SSH with explicit host, then same checks
3. Treat a GPU as available only when memory/use are low and no relevant train/eval process is running.
4. Pick an available host. If both hosts are busy, start `start_eval_worker.sh` in tmux on one eval host so it waits for queue tasks and free GPUs.

---

## Step 2 → Launch Eval In Tmux

All eval should run inside tmux.

### Mode A: Single Model Eval

```bash
session_name="eval_$(echo "$MODEL_NAME" | cut -c1-30)_$(date +%H%M%S)"
run_log="logs/train_launch/${session_name}.log"
mkdir -p logs/train_launch

tmux new-session -d -s "${session_name}" \
  "source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh && \
   conda activate swift-vln-eval && \
   bash src/swiftvln/scripts/eval/eval_by_name.sh ${MODEL_NAME} 2>&1 | tee ${run_log}"
```

### Mode B: Queue Eval

```bash
session_name="eval_queue_$(date +%H%M%S)"
run_log="logs/train_launch/${session_name}.log"
mkdir -p logs/train_launch

tmux new-session -d -s "${session_name}" \
  "source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh && \
   conda activate swift-vln-eval && \
   AUTO_TODO=true DYNAMIC_TODO=true \
   bash src/swiftvln/scripts/eval/eval_queue.sh 2>&1 | tee ${run_log}"
```

For a long-lived worker:

```bash
session_name="eval_worker_$(date +%H%M%S)"
run_log="logs/train_launch/${session_name}.log"
mkdir -p logs/train_launch

tmux new-session -d -s "${session_name}" \
  "source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh && \
   conda activate swift-vln-eval && \
   bash src/swiftvln/scripts/eval/start_eval_worker.sh 2>&1 | tee ${run_log}"
```

### Remote Launch On 73

```bash
ssh 10.246.152.73 "tmux new-session -d -s '${session_name}' \
  'source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh && \
   conda activate swift-vln-eval && \
   bash /mnt/data1/home/jiangjiajun/workspace/SwiftVLN-refactor/src/swiftvln/scripts/eval/eval_queue.sh 2>&1 | \
   tee /mnt/data1/home/jiangjiajun/workspace/SwiftVLN-refactor/${run_log}'"
```

Record tmux session name, host, log path, and start time.

---

## Step 3 → Quick Health Check

After launching, do 2-3 quick checks:

```bash
tmux has-session -t "${session_name}" 2>/dev/null && echo "OK" || echo "GONE"
tmux capture-pane -pt "${session_name}" -S -30 2>/dev/null | tail -30
tmux capture-pane -pt "${session_name}" -S -80 2>/dev/null | \
  grep -iE "Traceback|RuntimeError|CUDA out of memory|Address already in use" || echo "No immediate fatal errors"
```

If the user asked to monitor until done, keep polling tmux/logs and queue files manually.

---

## Enqueue Models

Prefer `enqueue_eval.sh`:

```bash
bash src/swiftvln/scripts/eval/enqueue_eval.sh <model_name>
bash src/swiftvln/scripts/eval/enqueue_eval.sh <model_name> --skip-checkpoint
```

Rules:
- Queue writes are local file operations in the shared workspace.
- Deduplicate before appending: skip if already in `eval_todo.txt` or `eval_done.txt`.
- Verify enqueue result by recounting todo/done/failed.

---

## Results

| Type | Path |
|---|---|
| Per-model results | `results/eval/<model_arch>/<model_name>/<split>/<timestamp>/` |
| Queue logs | `logs/eval_queue_*.log` |
| Queue summaries | `logs/eval_queue_results/eval_results_*.txt` |
| Per-host completion status | `runtime/eval_queue/eval_queue_last_run_<hostname>.json` |
| Collected CSV | `results/eval_collected/<split>/eval_results.csv` |

CSV collection is triggered by `src/swiftvln/scripts/eval/collect_eval_results.py` after each successful eval.

---

## Default Script Paths

| Purpose | Path |
|---|---|
| Single eval by name | `src/swiftvln/scripts/eval/eval_by_name.sh` |
| Queue engine | `src/swiftvln/scripts/eval/eval_queue.sh` |
| Long-lived worker | `src/swiftvln/scripts/eval/start_eval_worker.sh` |
| Local enqueue helper | `src/swiftvln/scripts/eval/enqueue_eval.sh` |
| CSV collector | `src/swiftvln/scripts/eval/collect_eval_results.py` |
| Eval todo queue | `runtime/eval_queue/eval_todo.txt` |
| Eval done list | `runtime/eval_queue/eval_done.txt` |
| Eval failed list | `runtime/eval_queue/eval_failed_todo.txt` |
| Per-host completion status | `runtime/eval_queue/eval_queue_last_run_<hostname>.json` |

---

## Known Issues & Quick Fixes

### 1. `ssh 73` timeout / wrong host resolution

Use the explicit host:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 10.246.152.73
```

### 2. Eval log has NCCL / generation warnings but no crash

Treat as non-fatal if `torchrun` is alive, rank 0 progress is increasing, and there is no `Traceback`.

### 3. Concurrent eval worker

During preflight:

```bash
pgrep -af "start_eval_worker.sh|eval_queue.sh|eval_by_name.sh"
```

Stop an existing worker only if it is not intended for the current run.

---

## Operating Rules

1. Eval hosts are `98` and `73` only. Never eval on `17`.
2. Launch eval in tmux. Use `eval_<short_desc>_<HHMMSS>`.
3. Prefer existing project scripts over ad-hoc commands.
4. Use `eval_by_name.sh` for a single model, `eval_queue.sh` or `start_eval_worker.sh` for queue consumption.
5. Eval enqueue is always a local file operation. Prefer `enqueue_eval.sh`.
6. Activate `swift-vln-eval` before running eval scripts.
7. Do not auto-fallback to single-GPU on OOM unless the user explicitly requests.
8. Keep train and eval in separate tmux sessions.
9. Do not commit unless the user explicitly asks.
