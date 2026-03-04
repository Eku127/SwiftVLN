---
name: overlapvln-train
description: "Launch, supervise, and complete VLN training pipelines. Covers OverlapVLN (default) and optionally StreamVLN/CompressVLN. Includes queue-first workflow, human-like interactive session supervision (sleep-check-act loop), auto-recovery, eval enqueue via shared workspace mount, and webhook completion notifications."
---

# OverlapVLN Train Skill

A step-by-step training pipeline where Codex acts as a **human-like operator**: confirm plans, launch training, stay in the session watching over it by periodically checking status, handle issues in real-time, and bridge results to eval — all within a single interactive session.

---

## Architecture: Shared Workspace Mount

Servers `98`, `73`, and `17` all mount the same workspace at:

```
/mnt/data1/home/jiangjiajun/workspace/SwiftVLN
```

Implications:

- **Queue files** (`eval_todo.txt`, `eval_done.txt`, `eval_failed_todo.txt`), training outputs, scripts, and configs are **locally accessible from every host**.
- After training completes on any host, eval enqueue is always a **local file append** — no SSH, no remote transfer.
- SSH is only needed for **GPU status checks** (`nvidia-smi`), **process inspection** (is `torchrun` alive?), and **remote process launch/kill**.

---

## Inputs

Before starting, confirm the following with the user:

| Parameter | Default | Notes |
|---|---|---|
| Model(s) | `overlapvln` | Include `streamvln`/`compressvln` only when user explicitly names them. Treat `baseline`/`basleine`/ambiguous wording as `overlapvln`. |
| Stage | — | `stage1` or `stage2` |
| Environment | — | `satnav` or `habitat` |
| QA mixed training | — | Whether to mix QA data |
| SwanLab | Always enabled | Project fixed to `satnav` |
| Monitor interval | `90s` | Normal pause between supervision cycles |
| Alert interval | `45s` | Shorter pause after anomaly detection |
| Max idle cycles | `10` | Escalate and notify user if no progress for this many consecutive cycles |

---

## Step 1 → Confirm Plan & Run Checklist

1. Read current scripts to catch recent changes:
- `src/swiftvln/scripts/train/train_queue.sh`
- `src/swiftvln/models/overlapvln/script/train/train_overlapvln_qwen2_5_vl.sh`
2. Produce a **run checklist**:
   - Model set, stage, environment
   - Data version plan
   - Launch mode (`queue` or `single`)
   - Expected output directory naming
3. Apply default model rule: `baseline`/`basleine`/unspecified → `overlapvln`; explicit names override.
4. **Wait for user confirmation** before proceeding.

## Step 2 → Check Servers & Pick Host

1. Inspect all three servers:
   - `98` — local, direct execution
   - `73` (`10.246.152.73`) — SSH for GPU/process status
   - `17` (`10.246.132.17`) — Docker container; also check container + tmux readiness
2. Run `nvidia-smi` on each to assess GPU availability.
3. Pick an actually idle 8-GPU node and explain the reasoning.
4. **Wait for user confirmation** on selected host.

## Step 3 → Pin Dataset Version & Config

1. Discover latest SatNav dataset version under `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_*`.
2. Resolve data paths:
   - Trajectory data: `<latest>/trajectory_data`
- QA data: `<latest>/data/qa_swift.jsonl`
3. If user requests config sync, update:
- `src/swiftvln/configs/satnav_task.yaml`
- `src/swiftvln/scripts/train/train_queue.sh`
4. **Fail fast** if any data path is missing — report the exact blocking path.
5. **Wait for user confirmation** to proceed.

## Step 4 → Launch Training (Always in tmux)

> **All training must be launched inside a persistent tmux session.** Never run training in a bare shell. tmux ensures the training survives Codex session disconnection and allows Codex to inspect it via `tmux capture-pane` during supervision.

### tmux session naming convention

```
train_<short_exp_desc>_<HHMMSS>
```

Example: `train_baseline_ovlpvln_ver260227_161820`

### Launch pattern

```bash
# Generate session name
session_name="train_<short_desc>_$(date +%H%M%S)"
run_log="logs/train_launch/${session_name}.log"
mkdir -p logs/train_launch

# Launch inside tmux (non-blocking)
tmux new-session -d -s "${session_name}" \
  "bash src/swiftvln/scripts/train/train_queue.sh 2>&1 | tee ${run_log}"

# Verify the session is alive
tmux ls | grep "${session_name}"
```

For training on `73` or `17`, wrap the `tmux new-session` command inside `ssh <host>`.

1. Choose launch mode:
   - **Queue** (multi-experiment / multi-model): `src/swiftvln/scripts/train/train_queue.sh`
   - **Single** (one run or deep tuning): `src/swiftvln/models/overlapvln/script/train/train_overlapvln_qwen2_5_vl.sh`
2. Verify tmux session is alive and training process (`train_queue.sh` / `torchrun`) is running before proceeding.
3. Record: tmux session name, start time, log path, selected host, full command.
4. Mark any temporary patched scripts for post-run cleanup.
5. **Immediately proceed to Step 5** — no user confirmation needed, no pause.

### Pose Embed smoke-test preset (SatNav)

When the user asks to verify pose embedding adoption (not full training), use a tiny run:

- `VLN_ENV_TYPE=satnav`
- `MAX_SAMPLES<=32`, `NUM_EPOCHS=1`, `BATCH_SIZE=1`, single GPU
- `SAVE_STEPS=5` (ensure checkpoint is written during smoke test)
- `USE_POSE_EMBED=true` and run both `POSE_FUSION_METHOD=additive` and `POSE_FUSION_METHOD=film`

Pass signals in log:

- `EmbeddingEnhancementPipeline(...pose...)`
- Dataset sample keys include `frame_poses`
- `Training completed!` and `Model saved to: ...`

## Step 5 → Supervise Training (Human-Like Session Loop)

> **This step starts immediately after Step 4 with no pause or user prompt.** Codex stays in the session and acts like a human operator sitting at the terminal. **No standalone monitoring script is created.** Codex itself repeatedly: sleeps → wakes up → checks status → takes action if needed → reports → sleeps again — continuing this loop until all training completes or the user interrupts.
>
> **Do NOT stop after launching.** After verifying the tmux session is alive, enter this loop immediately and stay in it.

### How the loop works

```
┌──────────────────────────────────────────────┐
│  1. Wake up (after sleeping N seconds)       │
│  2. Collect status signals (see checklist)   │
│  3. Evaluate: healthy / anomaly / complete?  │
│  4. If anomaly → attempt auto-fix            │
│  5. Report concise status update to user     │
│  6. Determine next sleep duration:           │
│     - Normal: MONITOR_INTERVAL_SEC (90s)     │
│     - Alert:  ALERT_MONITOR_INTERVAL_SEC     │
│               (45s, after anomaly)            │
│  7. Sleep and repeat                         │
└──────────────────────────────────────────────┘
```

This is **not** a background script or cron job. It is Codex executing shell commands interactively, pausing with `sleep`, then running the next round of checks — exactly as a human would do in a terminal session.

### Per-cycle status checklist

- **tmux session alive**: `tmux ls | grep <session_name>` — if gone, escalate immediately
- **Training progress**: `tmux capture-pane -pt <session_name> | tail -30` for latest output; also tail `logs/train_launch/<session_name>.log`
- **Queue status**: check for `SUCCESS` / `FAILED` markers in queue result files
- **Runtime health**: scan for NCCL / CUDA / OOM / port conflict / dataloader errors
- **Process presence**: verify `train_queue.sh` / `torchrun` workers are alive on the execution host
- **Eval linkage** (if active): check `eval_todo.txt` / `eval_done.txt` / `eval_failed_todo.txt`

### Anomaly response

- On detection (OOM, NCCL hang, restart): **shorten** polling to `ALERT_MONITOR_INTERVAL_SEC`.
- After recovery and stabilization: **restore** to `MONITOR_INTERVAL_SEC`.
- If no useful progress for `MAX_IDLE_CYCLES` consecutive cycles: run deeper diagnostics and **notify user** with evidence.

### Auto-fix attempts (before declaring failure)

- Dataloader workers adjustment
- Port conflict resolution
- OOM → reduce batch size
- bf16 fallback
- `safe_serialization` fallback

### Queue failure policy

- If one queue experiment fails, **skip it and continue** remaining runs.
- Send webhook notification with issue description, attempted fix, and outcome.

## Step 6 → Verify Checkpoint & Enqueue to Eval

1. For each completed run, parse logs for `Model saved to: <path>`.
2. Verify checkpoint: directory exists and is non-empty.
3. Generate **run summary**: model, config deltas, data version, output path, duration, status.
4. **Enqueue to eval** — append to `src/swiftvln/scripts/eval/eval_todo.txt`:
   - One model name per line.
   - This is a **local file write** on the current host (shared workspace mount).
   - No SSH required. No enqueue wrapper scripts. No remote target host assumption.
5. **Deduplicate** before appending:
   - Already in `eval_todo.txt` or `eval_done.txt` → skip.
   - In `eval_failed_todo.txt` → keep one canonical entry in `eval_todo.txt`.
6. **Verify** enqueue result: recount todo/done/failed, confirm inserted line or skip reason.

> **Why no SSH?** Because `98`, `73`, and `17` all see the same filesystem. Writing `eval_todo.txt` on the training host is immediately visible to eval workers on any other host.

## Step 7 → Final Summary & Webhook

1. Send completion webhook via `.codex/skills/overlapvln-train/scripts/send_wecom_webhook.sh`:
   - URL: `https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=503b5488-4d70-455d-a5b9-29fc8d7fb797`
2. Required fields: `exp_name`, `duration`, `success_rate`, `output_dir`.
3. Return final report:
   - Total runs, success / failed counts
   - Per-run output path
   - Eval queue updates for each successful model

---

## Default Script Paths

| Purpose | Path |
|---|---|
| Training queue | `src/swiftvln/scripts/train/train_queue.sh` |
| OverlapVLN single run | `src/swiftvln/models/overlapvln/script/train/train_overlapvln_qwen2_5_vl.sh` |
| Eval todo queue | `src/swiftvln/scripts/eval/eval_todo.txt` |
| Webhook helper | `.codex/skills/overlapvln-train/scripts/send_wecom_webhook.sh` |

---

## Operating Rules

1. **Pause for user confirmation** before each major step transition (Step 1→2→3→4). Once user confirms Step 4 launch, **proceed through Steps 4→5→6→7 automatically without stopping**.
2. **Always launch training in tmux**. Never run training in a bare shell. Use the naming convention `train_<short_desc>_<HHMMSS>`.
3. **After launching, immediately enter the supervision loop (Step 5)**. Do not pause, do not ask the user if they want monitoring. Just start monitoring.
4. **Prefer existing project scripts** over ad-hoc one-off logic.
5. **Supervision is interactive and session-driven**:
   - Codex itself runs the sleep-check-act loop in the current session, like a human operator watching the terminal.
   - Never create or launch a standalone monitoring script / daemon / cron job.
   - Use `sleep <N>` between cycles, then run status-checking commands, exactly as a person would.
6. **Continue supervision until completion**: keep the loop running until all target runs reach terminal status (`SUCCESS`/`FAILED`), or user explicitly interrupts.
5. **Eval enqueue is always a local file operation**: all servers share the mounted workspace. Never SSH for queue-file writes. Never use enqueue wrapper scripts. Append directly to `eval_todo.txt`.
6. **Do not commit** unless user explicitly asks.
7. **Keep docs minimal** unless user explicitly asks.
8. **Clean up** temporary run-only scripts and artifacts after verification.
9. **Fail fast** on missing dataset paths, checkpoints, or conda envs — report the exact blocking item.
