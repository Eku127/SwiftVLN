---
name: overlapvln-eval
description: "Run and supervise VLN evaluation with queue-based execution on servers 98/73. Auto-consume newly trained models via eval_todo, human-like interactive session supervision, auto-recovery, webhook notifications, and CSV result collection."
---

# OverlapVLN Eval Skill

Evaluate trained VLN models by name or via a persistent queue worker. Codex acts as a **human-like operator**: pick an eval host, launch workers, stay in the session watching evaluation progress, handle issues in real-time, and report results — all within a single interactive session.

Related skills:
- **`overlapvln-train`**: upstream producer — trains models and enqueues them to `eval_todo.txt`.
- **`server-train-eval-monitor`**: cluster-wide status overview across 98/73/17.

---

## Architecture: Shared Workspace Mount

Servers `98`, `73`, and `17` all mount the same workspace at:

```
/mnt/data1/home/jiangjiajun/workspace/SwiftVLN
```

Implications:

- **Queue files** (`eval_todo.txt`, `eval_done.txt`, `eval_failed_todo.txt`), eval outputs, scripts, and collected CSVs are **locally accessible from every host**.
- Eval enqueue from any host (including `17` after training) is always a **local file append** — no SSH, no remote transfer.
- SSH is only needed for **GPU status checks** (`nvidia-smi`), **process inspection** (is `eval_queue.sh` alive?), and **remote process launch/kill**.

---

## Inputs

Before starting, confirm the following with the user:

| Parameter | Default | Notes |
|---|---|---|
| Model name(s) | — | One or more model names to evaluate, or use queue-based consumption |
| Eval split | `val_unseen` | Options: `val_unseen`, `val_seen`, `test` |
| CUDA devices | `0,1,2,3,4,5,6,7` | GPU device list |
| Save video | `false` | Whether to save evaluation videos |
| Conda env | `swift-vln-eval` | Must be activated before running eval scripts |
| Monitor interval | `70s` | Normal pause between supervision cycles |
| Alert interval | `35s` | Shorter pause after anomaly detection |

---

## Eval Script Decision Tree

Choose the right script based on the scenario:

| Scenario | Script | Description |
|---|---|---|
| One-off eval of a single model | `eval_by_name.sh` | Run once and exit |
| Serial queue consumption (long-lived) | `start_eval_worker.sh` | Keeps polling `eval_todo.txt`, waits for new tasks |
| Auto-stop queue consumption | `start_eval_monitor.sh` | Polls `eval_todo.txt`, stops when all hosts idle |
| Low-level queue runner | `eval_queue.sh` | Internal queue engine, usually invoked by worker/monitor |

---

## Step 1 → Check Servers & Pick Eval Host

1. Eval hosts allowed: **`98` and `73` only**. Never run eval on `17`.
2. Inspect GPU availability on both hosts:
   - `98` — local `nvidia-smi`
   - `73` (`10.246.152.73`) — SSH for GPU/process checks
3. Pick a host with available GPUs. If `98` is busy with training, use `73`.
4. **Wait for user confirmation** on selected host.

## Step 2 → Launch Eval

Two modes depending on the task:

### Mode A: Single Model Eval

```bash
conda activate swift-vln-eval
bash src/swiftvln/scripts/eval/eval_by_name.sh <model_name>
```

Optional env vars: `EVAL_SPLIT`, `CUDA_DEVICES`, `SAVE_VIDEO`, `ENV_TYPE`.

For parsing validation before real eval (recommended for new naming slots like `pose` / `posefilm`):

```bash
CHECK_ONLY=true bash src/swiftvln/scripts/eval/eval_by_name.sh <model_name>
```

For smoke eval (not full benchmark), add:

```bash
CUDA_DEVICES=0 MAX_EPISODES=2 EVAL_SPLIT=val_unseen SAVE_VIDEO=false \
  bash src/swiftvln/scripts/eval/eval_by_name.sh <model_name>
```

### Mode B: Queue Worker (continuous consumption)

Choose based on desired behavior:

**Long-lived worker** (waits for new tasks when queue empties):

```bash
conda activate swift-vln-eval
bash src/swiftvln/scripts/eval/start_eval_worker.sh
```

**Auto-stop monitor** (stops when all 98/73/17 hosts are idle):

```bash
conda activate swift-vln-eval
bash src/swiftvln/scripts/eval/start_eval_monitor.sh
```

Both consume `eval_todo.txt` and run `eval_queue.sh` internally.

## Step 3 → Enqueue Models

When a trained model needs evaluation, enqueue via **local file operation**:

**Preferred: use the existing `enqueue_eval.sh`** (has locking + dedup + checkpoint check):

```bash
bash src/swiftvln/scripts/eval/enqueue_eval.sh <model_name>
# Or skip checkpoint verification:
bash src/swiftvln/scripts/eval/enqueue_eval.sh <model_name> --skip-checkpoint
```

**Alternative: direct file append** (when `enqueue_eval.sh` is not suitable):

```bash
model="<model_name>"
todo="runtime/eval_queue/eval_todo.txt"
done="runtime/eval_queue/eval_done.txt"
if ! grep -Fxq "$model" "$todo" 2>/dev/null && ! grep -Fxq "$model" "$done" 2>/dev/null; then
  echo "$model" >> "$todo"
fi
```

**Key rules**:
- Always a local file operation (shared filesystem, no SSH needed).
- Deduplicate before appending: skip if in `eval_todo.txt` or `eval_done.txt`.
- If in `eval_failed_todo.txt`: keep one canonical entry in `eval_todo.txt`.
- Verify enqueue result by recounting todo/done/failed.

## Step 4 → Supervise Eval (Human-Like Session Loop)

> **Core principle**: Codex stays in the session and acts like a human operator. **No new monitoring script is created.** Codex itself repeatedly: sleeps → wakes up → checks status → takes action if needed → reports → sleeps again — until evaluation completes or the user interrupts.

### How the loop works

```
┌──────────────────────────────────────────────┐
│  1. Wake up (after sleeping N seconds)       │
│  2. Collect status signals (see checklist)   │
│  3. Evaluate: healthy / anomaly / complete?  │
│  4. If anomaly → attempt auto-fix            │
│  5. Report concise status update to user     │
│  6. Determine next sleep duration:           │
│     - Normal: 70s                            │
│     - Alert:  35s (after anomaly)            │
│  7. Sleep and repeat                         │
└──────────────────────────────────────────────┘
```

This is **not** a background script. It is Codex executing shell commands interactively, pausing with `sleep`, then running the next round of checks — exactly as a human would do.

### Per-cycle status checklist

- **Queue status**: count lines in `eval_todo.txt` / `eval_done.txt` / `eval_failed_todo.txt`
- **Process presence**: check `eval_queue.sh`, `eval_by_name.sh`, `torchrun ... -m swiftvln.models.*.eval` on eval host
- **Eval progress**: tail latest per-model eval log for `Rank 0 (... eps ...)` progress line
- **Runtime health**: scan for NCCL / CUDA / OOM / port conflict errors
- **Pose embed verification** (if applicable): confirm log shows `Pose Embed: true (fusion=...)` and `Restored embed_enhance weights from checkpoint`

### Exit conditions

- Target model appears in `eval_done.txt` → **success**
- Target model appears in `eval_failed_todo.txt` → **failed**
- All target models resolved → stop loop

### Anomaly response

- On detection (OOM, port conflict, crash): **shorten** polling to `35s`.
- After recovery: **restore** to `70s`.
- Auto-fix attempts before failure: port conflict resolution, bf16 fallback.
- **Do not** auto-fallback to single-GPU on OOM unless user explicitly requests.

### Queue failure policy

- If one eval fails, **skip it and continue** remaining queue tasks.
- Send webhook notification with issue, attempted fix, and outcome.

## Step 5 → Results & Notifications

### Eval outputs

| Type | Path |
|---|---|
| Per-model results | `results/eval/<model_arch>/<model_name>/...` |
| Queue logs | `logs/eval_queue_*.log` |
| Queue summaries | `logs/eval_queue_results/eval_results_*.txt` |
| Collected CSV | `results/eval_collected/eval_results_data<version>.csv` |

### CSV collection

- Automatically triggered by `src/swiftvln/scripts/eval/collect_eval_results.py` after each successful eval.
- CSV columns include: model_name, model_type, plan, collected_time, plus ALL / Boundary / LandmarkSet / Road metrics (SR, SPL, OS, NE, Steps, Episodes).
- Sorted by inferred test plan order (baseline → baseline + gtc → baseline + sgtc, etc.).

### Webhook notifications

- Built into `eval_queue.sh` (uses env var `WEBHOOK_URL`).
- Default URL: `https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=87cd9c07-52f0-4cec-a7a8-9586a9dc68c8`
- Triggers: eval started, eval finished (success/failed), queue completed.
- For failure webhooks: include issue, attempted solution, and status.

---

## Default Script Paths

| Purpose | Path |
|---|---|
| Single eval by name | `src/swiftvln/scripts/eval/eval_by_name.sh` |
| Queue engine | `src/swiftvln/scripts/eval/eval_queue.sh` |
| Long-lived worker | `src/swiftvln/scripts/eval/start_eval_worker.sh` |
| Auto-stop monitor | `src/swiftvln/scripts/eval/start_eval_monitor.sh` |
| Local enqueue helper | `src/swiftvln/scripts/eval/enqueue_eval.sh` |
| CSV collector | `src/swiftvln/scripts/eval/collect_eval_results.py` |
| Eval todo queue | `runtime/eval_queue/eval_todo.txt` |
| Eval done list | `runtime/eval_queue/eval_done.txt` |
| Eval failed list | `runtime/eval_queue/eval_failed_todo.txt` |

---

## Known Issues & Quick Fixes (Updated: 2026-03-04)

### 1) `ssh 73` timeout / wrong host resolution

Symptom:
- `ssh 73` hangs or resolves to unexpected address (e.g., `0.0.0.73`), causing startup failure before eval begins.

Root cause:
- Local SSH alias/config is not guaranteed across machines.

Fix:
- Always prefer explicit host/IP in automation:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 10.246.152.73 'hostname; date'
```

- In scripts, rely on `HOST_73=10.246.152.73` instead of shorthand alias.

### 2) Eval log has NCCL / generation warnings but no crash

Symptom:
- Warnings such as:
  - `ProcessGroupNCCL ... using GPU x ... currently unknown`
  - `generation flags ... ignored: ['temperature']`
- But eval still shows advancing progress (`Rank 0 ... n/N`), GPUs occupied, no `Traceback`.

Root cause:
- Runtime warning (environment/framework behavior), not necessarily a blocking error.

Fix / decision rule:
- Treat as **non-fatal** if all conditions hold:
  1. `torchrun` process still alive
  2. `Rank 0` progress keeps increasing
  3. No `Traceback|RuntimeError|Exception|CUDA out of memory|Address already in use`
- Continue monitoring; only escalate if progress stalls or fatal errors appear.

### 3) Potential interference from concurrent eval monitor/worker

Symptom:
- Another session already has `start_eval_monitor.sh` / `start_eval_worker.sh` / `eval_queue.sh` running.

Risk:
- Unintended concurrent queue consumption, duplicated eval, or status confusion.

Fix:
- During preflight, confirm process cwd/cmdline and keep current run isolated:

```bash
pgrep -af "start_eval_monitor.sh|eval_queue.sh|eval_by_name.sh"
readlink -f /proc/<pid>/cwd
```

- If an existing monitor/worker is not intended for the current run, stop it explicitly before queue-mode eval.

---

## Operating Rules

1. **Eval hosts are `98` and `73` only**. Never eval on `17`.
2. **Prefer existing project scripts** over ad-hoc one-off logic.
3. **Use `eval_by_name.sh`** for single model evaluation, **`start_eval_worker.sh`** or **`start_eval_monitor.sh`** for queue-based consumption.
4. **Supervision is interactive and session-driven**:
   - Codex runs the sleep-check-act loop in the current session, like a human operator.
   - Never create a new standalone monitoring script for Codex supervision.
   - The `start_eval_monitor.sh` script is for launching the eval consumer process itself, not for Codex's own supervision loop.
5. **Continue supervision until completion**: when user says "monitor until done", keep the loop until all target models reach terminal status, or user interrupts.
6. **Eval enqueue is always a local file operation**: shared workspace means no SSH for queue-file writes. Prefer `enqueue_eval.sh` (has locking + dedup + checkpoint verification).
7. **Conda env**: activate `swift-vln-eval` before running any eval script.
8. **Do not auto-fallback to single-GPU on OOM** unless user explicitly requests.
9. **Keep train and eval in separate sessions** (or tmux windows) to avoid blocking.
10. **Do not commit** unless user explicitly asks.
11. **Clean up** temporary test outputs/artifacts after verification.
