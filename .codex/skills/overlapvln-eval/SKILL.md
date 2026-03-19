---
name: overlapvln-eval
description: "Run and supervise VLN evaluation with tmux-based async execution, watchdog monitoring, and Codex callback via `codex exec resume`. Supports queue consumption, auto-recovery, webhook notifications, and CSV result collection."
---

# OverlapVLN Eval Skill

Evaluate trained VLN models by name or via a persistent queue. Codex acts as a **launch operator**: pick an eval host, launch eval in tmux, register a watchdog for async completion callback, and optionally stay for initial health checks. When the watchdog detects eval completion or failure, it resumes the Codex session for intelligent result reporting or error recovery.

Related skills:
- **`overlapvln-train`**: upstream producer — trains models and enqueues them to `eval_todo.txt`.
- **`server-train-eval-monitor`**: cluster-wide status overview across 98/73/17.

---

## Architecture: Async Execution with Watchdog Callback

### Execution Model

```
Codex Session (interactive):
  1. Pick eval host, verify GPUs
  2. Launch eval in tmux (non-blocking)
  3. Quick health check (2-3 cycles)
  4. Register watchdog (background)
  5. Report to user → session can safely end

Watchdog (background, nohup):
  - Polls tmux session status every 30s
  - On completion → codex exec resume → Codex comes back for reporting
  - On failure → codex exec resume → Codex comes back for error analysis & fix
  - On timeout/stall → webhook + optional Codex callback
```

**Key benefit**: eval runs in tmux (survives session disconnect), watchdog ensures Codex is notified asynchronously. No more session timeout issues.

### Shared Workspace Mount

Servers `98`, `73`, and `17` all mount the same workspace at:

```
/mnt/data1/home/jiangjiajun/workspace/SwiftVLN
```

- **Queue files**, eval outputs, scripts, and CSVs are **locally accessible from every host**.
- Eval enqueue from any host is always a **local file append** — no SSH.
- SSH is only needed for **GPU status checks**, **process inspection**, and **remote tmux launch**.

---

## Inputs

Before starting, confirm the following with the user:

| Parameter | Default | Notes |
|---|---|---|
| Model name(s) | — | One or more model names to evaluate, or use queue-based consumption |
| Eval split | SatNav: 不指定则同时跑 `val_seen` + `val_unseen`；Habitat: `val_unseen` | 显式设置 `EVAL_SPLIT=val_seen` 可只跑单个 split |
| CUDA devices | `0,1,2,3,4,5,6,7` | GPU device list |
| Save video | `false` | Whether to save evaluation videos |
| Conda env | `swift-vln-eval` | Must be activated before running eval scripts |
| Async mode | `watchdog` | `watchdog` (default) or `stay` (Codex stays in supervision loop) |

---

## Eval Script Decision Tree

| Scenario | Script | Description |
|---|---|---|
| One-off eval of a single model | `eval_by_name.sh` | Run once and exit |
| Serial queue consumption (long-lived) | `start_eval_worker.sh` | Keeps polling `eval_todo.txt`, waits for new tasks |
| Auto-stop queue consumption | `start_eval_monitor.sh` | Polls `eval_todo.txt`, stops when all hosts idle |
| Low-level queue runner | `eval_queue.sh` | Internal queue engine, usually invoked by worker/monitor |
| Async completion monitor | `eval_watchdog.sh` | Background watchdog, triggers Codex callback |

---

## Step 1 → Check Servers & Pick Eval Host

1. Eval hosts allowed: **`98` and `73` only**. Never run eval on `17`.
2. Inspect GPU availability on both hosts:
   - `98` — local `nvidia-smi` + `pgrep -af "torchrun|train_queue"` 检查是否有训练进程
   - `73` (`10.246.152.73`) — SSH 同上
3. **判断 GPU 是否真正空闲**：不仅看 `nvidia-smi` 显存，还要确认没有 `torchrun` / `train_queue.sh` 进程在运行。被训练占用的 GPU 不能用于 eval。
4. Pick a host with available GPUs. If `98` is busy with training, use `73`.

### 如果两台服务器都没有空闲 GPU

**不要直接启动 eval**（会因 GPU 被占用而 OOM/CUDA 错误）。改为：

1. 在其中一台服务器上以 tmux 启动 `start_eval_worker.sh`：

```bash
session_name="eval_worker_$(date +%H%M%S)"
run_log="logs/train_launch/${session_name}.log"
mkdir -p logs/train_launch

tmux new-session -d -s "${session_name}" \
  "source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh && \
   conda activate swift-vln-eval && \
   bash src/swiftvln/scripts/eval/start_eval_worker.sh 2>&1 | tee ${run_log}"
```

2. Worker 会持续轮询 `eval_todo.txt`（每 60s），当训练结束 GPU 释放后自动开始 eval。
3. 注册 `eval_watchdog` 监控该 worker 的 tmux session。
4. 向用户报告："两台服务器均在训练中，已启动 eval worker 等待，训练完成后将自动开始评测。"

### 如果是用户交互式触发（非 watchdog 自动触发）

5. **Wait for user confirmation** on selected host.

## Step 2 → Launch Eval in tmux

> **All eval must be launched inside a persistent tmux session.** This ensures the eval survives Codex session disconnect and allows the watchdog to monitor it.
>
> **⚠️ Step 2、Step 3、Step 4 是原子操作：启动 tmux → health check → 注册 watchdog，不可跳过任何一步。**

### tmux session naming convention

```
eval_<short_desc>_<HHMMSS>
```

Example: `eval_queue_val_seen_153025`

### Mode A: Single Model Eval

```bash
session_name="eval_$(echo $MODEL_NAME | cut -c1-30)_$(date +%H%M%S)"
run_log="logs/train_launch/${session_name}.log"
mkdir -p logs/train_launch

tmux new-session -d -s "${session_name}" \
  "source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh && \
   conda activate swift-vln-eval && \
   bash src/swiftvln/scripts/eval/eval_by_name.sh ${MODEL_NAME} 2>&1 | tee ${run_log}"

tmux ls | grep "${session_name}"
```

Optional env vars: `EVAL_SPLIT`（不设置时 SatNav 自动跑 val_seen+val_unseen 两个 split）, `CUDA_DEVICES`, `SAVE_VIDEO`, `ENV_TYPE`.

For parsing validation before real eval:

```bash
CHECK_ONLY=true bash src/swiftvln/scripts/eval/eval_by_name.sh <model_name>
```

### Mode B: Queue Eval (multiple models)

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

For long-lived worker (waits for new tasks):

```bash
tmux new-session -d -s "${session_name}" \
  "source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh && \
   conda activate swift-vln-eval && \
   bash src/swiftvln/scripts/eval/start_eval_worker.sh 2>&1 | tee ${run_log}"
```

### Remote launch (server 73)

```bash
ssh 10.246.152.73 "tmux new-session -d -s '${session_name}' \
  'source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh && \
   conda activate swift-vln-eval && \
   bash /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/src/swiftvln/scripts/eval/eval_queue.sh 2>&1 | \
   tee /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/${run_log}'"
```

**Record**: tmux session name, start time, log path, eval host.

## Step 3 → Quick Health Check

After launching, do 2-3 rounds of health checks (every 30s) to catch early failures:

```bash
# Round 1: verify tmux session exists
tmux has-session -t "${session_name}" 2>/dev/null && echo "OK" || echo "GONE"

# Round 2: verify eval process started
tmux capture-pane -pt "${session_name}" -S -20 2>/dev/null | tail -20

# Round 3: check for immediate fatal errors
tmux capture-pane -pt "${session_name}" -S -50 2>/dev/null | \
  grep -iE "Traceback|RuntimeError|CUDA out of memory|Address already in use" || echo "No errors"
```

If early errors detected, attempt to fix before proceeding to Step 4.

## Step 4 → Register Watchdog & Report (MANDATORY)

> **🚨 此步骤为 MANDATORY（强制），不可跳过。**
> Watchdog 负责：eval 结束/崩溃/停滞时发 webhook 通知，eval 结束后自动退出。

### 4.1 — 启动 watchdog

Watchdog 必须运行在**与 tmux session 相同的服务器上**（才能检测 tmux）。

**eval 在 98 上：**

```bash
SWIFTVLN_ROOT="/mnt/data1/home/jiangjiajun/workspace/SwiftVLN"
nohup bash "${SWIFTVLN_ROOT}/src/swiftvln/scripts/eval/eval_watchdog.sh" \
  --tmux-session "${session_name}" \
  --eval-log "${run_log}" \
  --cleanup-days 7 \
  > /dev/null 2>&1 &
WATCHDOG_PID=$!
echo "Watchdog PID=${WATCHDOG_PID}"
```

**eval 在 73 上（SSH 启动）：**

```bash
SWIFTVLN_ROOT="/mnt/data1/home/jiangjiajun/workspace/SwiftVLN"
ssh 10.246.152.73 "cd ${SWIFTVLN_ROOT} && \
  nohup bash src/swiftvln/scripts/eval/eval_watchdog.sh \
    --tmux-session '${session_name}' \
    --eval-log '${run_log}' \
    --cleanup-days 7 \
    > /dev/null 2>&1 & echo \$!"
```

> Watchdog 逻辑：每 30s 检查 tmux session 是否存活；eval 结束（tmux 退出）后发 webhook 并**自动退出**；停滞 ~10 分钟发 webhook 告警。

### 4.2 — 验证 watchdog 存活

```bash
sleep 2
kill -0 "$WATCHDOG_PID" 2>/dev/null && echo "✅ Watchdog alive (PID=${WATCHDOG_PID})" || echo "❌ Watchdog failed"
```

如果失败：检查脚本路径和 tmux session 名称是否正确，重新注册。

### 4.3 — 报告给用户

- Eval 运行在：tmux session `<name>`，服务器 `<host>`
- Watchdog PID：`<pid>`（eval 结束后自动退出）
- 进度查看：`tmux attach -t <name>`
- 完成/报错时：webhook 通知

### Mode: Stay (user explicitly requests "monitor until done")

> **即使 stay 模式，也必须先完成上面的 4.1 + 4.2 注册 watchdog**，作为兜底。

If the user says "monitor until done" or "stay and watch", 先注册 watchdog，然后进入 supervision loop:

```
┌──────────────────────────────────────────────┐
│  1. Wake up (after sleeping 70s)             │
│  2. Check tmux session alive                 │
│  3. Tail log for progress / errors           │
│  4. Check eval_done/eval_failed files        │
│  5. If anomaly → attempt auto-fix            │
│  6. Report concise status update to user     │
│  7. Sleep and repeat                         │
└──────────────────────────────────────────────┘
```

Exit conditions:
- Target model appears in `eval_done.txt` → **success**
- Target model appears in `eval_failed_todo.txt` → **failed**
- All target models resolved → stop loop

Anomaly response:
- On detection (OOM, port conflict, crash): **shorten** polling to `35s`.
- After recovery: **restore** to `70s`.
- Auto-fix: port conflict resolution, bf16 fallback.
- **Do not** auto-fallback to single-GPU on OOM unless user explicitly requests.

## Step 5 → Enqueue Models (if needed)

When a trained model needs evaluation, enqueue via **local file operation**:

**Preferred: use `enqueue_eval.sh`** (has locking + dedup + checkpoint check):

```bash
bash src/swiftvln/scripts/eval/enqueue_eval.sh <model_name>
# Or skip checkpoint verification:
bash src/swiftvln/scripts/eval/enqueue_eval.sh <model_name> --skip-checkpoint
```

**Key rules**:
- Always a local file operation (shared filesystem, no SSH needed).
- Deduplicate before appending: skip if in `eval_todo.txt` or `eval_done.txt`.
- Verify enqueue result by recounting todo/done/failed.

## Step 6 → Results & Notifications

### Watchdog callback results

When the watchdog triggers `codex exec resume`, Codex resumes and should:

1. **On success**: read `eval_queue_last_run.json` and result files, summarize metrics, collect CSV.
2. **On failure**: analyze error context from the injected prompt, attempt fix, optionally restart eval.
3. **On timeout/stall**: check tmux session, diagnose, decide whether to wait or intervene.

### Eval outputs

**全局文件（多服务器共享，按 hostname 隔离）：**

| Type | Path |
|---|---|
| Per-model results | `results/eval/<model_arch>/<model_name>/<split>/<timestamp>/` |
| Queue logs | `logs/eval_queue_*.log` |
| Queue summaries | `logs/eval_queue_results/eval_results_*.txt` |
| Per-host completion status | `runtime/eval_queue/eval_queue_last_run_<hostname>.json` |
| Collected CSV | `results/eval_collected/eval_results_data<version>.csv` |

**Per-run 目录（每次 eval 独立，多服务器并发安全）：**

```
runtime/eval_queue/runs/<hostname>_<session_name>/
├── watchdog_result.json      watchdog 最终结果
├── watchdog.log              watchdog 日志
├── codex_response.txt        Codex 回调响应（如有）
└── eval_queue_status.json    eval_queue.sh 写入的完成状态
```

- 自动清理：watchdog 启动时默认清理 7 天前的 run 目录（`--cleanup-days`）
- `EVAL_RUN_DIR` 环境变量由 watchdog 自动 export，eval_queue.sh 会读取并写入 per-run 状态

### CSV collection

- Triggered by `src/swiftvln/scripts/eval/collect_eval_results.py` after each successful eval.
- CSV columns: model_name, model_type, plan, **swanlab_url**, collected_time, plus ALL / Boundary / LandmarkSet / Road metrics.
- `swanlab_url` 来源优先级：`evaluation_summary.json` 中的 `swanlab_url` → fallback 到训练输出目录下的 `train_metadata.json`。

### Webhook notifications

- Built into `eval_queue.sh` (uses env var `WEBHOOK_URL`).
- Default URL: `https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=87cd9c07-52f0-4cec-a7a8-9586a9dc68c8`
- Triggers: eval started, eval finished (success/failed), queue completed.
- Watchdog also sends independent notifications on completion/failure/stall.

---

## Default Script Paths

| Purpose | Path |
|---|---|
| Single eval by name | `src/swiftvln/scripts/eval/eval_by_name.sh` |
| Queue engine | `src/swiftvln/scripts/eval/eval_queue.sh` |
| Long-lived worker | `src/swiftvln/scripts/eval/start_eval_worker.sh` |
| Auto-stop monitor | `src/swiftvln/scripts/eval/start_eval_monitor.sh` |
| **Async watchdog** | `src/swiftvln/scripts/eval/eval_watchdog.sh` |
| Local enqueue helper | `src/swiftvln/scripts/eval/enqueue_eval.sh` |
| CSV collector | `src/swiftvln/scripts/eval/collect_eval_results.py` |
| Eval todo queue | `runtime/eval_queue/eval_todo.txt` |
| Eval done list | `runtime/eval_queue/eval_done.txt` |
| Eval failed list | `runtime/eval_queue/eval_failed_todo.txt` |
| Per-host completion status | `runtime/eval_queue/eval_queue_last_run_<hostname>.json` |
| Per-run directory | `runtime/eval_queue/runs/<hostname>_<session>/` |

---

## Codex Session Recovery via `codex exec resume`

When the watchdog or an external script triggers `codex exec resume`, the resumed Codex session receives a prompt with context about what happened. Codex should:

1. **Read the injected prompt carefully** — it contains outcome summary and error context.
2. **Check queue files** (`eval_done.txt`, `eval_failed_todo.txt`) for authoritative status.
3. **Read `eval_queue_last_run.json`** for structured completion data.
4. **For failures**: analyze error, check if auto-fixable, apply fix, re-enqueue and re-launch.
5. **For success**: collect CSV, summarize results, notify user via webhook.
6. **For timeouts**: check if eval is still running (`tmux has-session`), diagnose stall.

### Resumed session behavior

The resumed session runs with `--full-auto` (workspace-write sandbox). It can:
- Read/write files in the workspace
- Run shell commands (git, python, bash scripts)
- Launch new tmux sessions for retry

It **cannot** (without user override):
- SSH to remote servers (use local shared filesystem instead)
- Modify system-level configuration

---

## Known Issues & Quick Fixes

### 1) `ssh 73` timeout / wrong host resolution

Fix: Always use `ssh -o BatchMode=yes -o ConnectTimeout=8 10.246.152.73`.

### 2) Eval log has NCCL / generation warnings but no crash

Treat as **non-fatal** if: `torchrun` alive, `Rank 0` progress increasing, no `Traceback`.

### 3) Concurrent eval monitor/worker

During preflight, check:

```bash
pgrep -af "start_eval_monitor.sh|eval_queue.sh|eval_by_name.sh"
```

Stop existing monitor/worker if not intended for current run.

### 4) Codex session ID capture fails

If `ls ~/.codex/sessions/...` returns empty, fall back to `--last` flag:

```bash
codex exec resume --last --full-auto "..."
```

Or skip Codex callback and rely on webhook-only mode.

---

## Operating Rules

1. **Eval hosts are `98` and `73` only**. Never eval on `17`.
2. **Always launch eval in tmux**. Never run eval in a bare shell. Use naming convention `eval_<short_desc>_<HHMMSS>`.
3. **🚨 MANDATORY: tmux 启动后必须立即注册 watchdog（Step 4）并验证存活。** Watchdog 负责 webhook 通知和自动退出，不可跳过。
4. **Watchdog 在 tmux session 结束后自动退出**，无需手动清理。
5. **报告中必须包含 watchdog PID**。没有 PID 说明 Step 4 被跳过了。
6. **Prefer existing project scripts** over ad-hoc one-off logic.
7. **Use `eval_by_name.sh`** for single model, **`eval_queue.sh`** / **`start_eval_worker.sh`** for queue-based consumption.
8. **Eval enqueue is always a local file operation**: shared workspace, no SSH for queue writes. Prefer `enqueue_eval.sh`.
9. **Conda env**: activate `swift-vln-eval` before running any eval script.
10. **Do not auto-fallback to single-GPU on OOM** unless user explicitly requests.
11. **Keep train and eval in separate tmux sessions** to avoid blocking.
12. **Do not commit** unless user explicitly asks.
