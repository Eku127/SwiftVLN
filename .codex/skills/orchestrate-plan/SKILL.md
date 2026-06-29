---
name: orchestrate-plan
description: "Read a natural-language experiment plan file, interpret the experiments, confirm with the user, schedule training across servers 98/73/17 non-interactively, chain eval automatically, and return a results summary CSV. Use when the user asks to 'run a plan', '执行实验计划', '跑 runtime/plans/…', or 'orchestrate experiments'."
---

# Orchestrate Plan Skill

Codex acts as a **full-pipeline orchestrator**: read a natural-language plan → extract experiments → confirm → allocate servers → launch training non-interactively → wait for eval to complete → return results.

This skill coordinates the existing `swiftvln-train` and `swiftvln-eval` skills without duplicating their internal steps. Follow each step in order.

---

## Related Skills & Scripts

- **Training**: `swiftvln-train` skill (Step 2/3/4/5 conventions reused here)
- **Evaluation**: `swiftvln-eval` skill (eval launch, watchdog, results collection)
- **Train queue**: `src/swiftvln/scripts/train/train_queue.sh` (non-interactive mode via `TRAIN_EXPERIMENTS_FILE`)
- **Eval queue**: `runtime/eval_queue/eval_todo.txt` (auto-populated by train_queue after each training)
- **Results**: `src/swiftvln/scripts/eval/collect_eval_results.py`

---

## EXPERIMENTS File Format (Critical Reference)

When Codex generates the EXPERIMENTS bash file, each array entry must follow this exact pipe-separated format:

```
model|config|changes|ds_names|ds_paths
```

| Field | Description | Example |
|---|---|---|
| `model` | Model name | `swiftvln` |
| `config` | Param overrides (comma-separated) or `default` | `default` or `NUM_OVERLAP=32,BATCH_SIZE=8` |
| `changes` | Human-readable description of non-default params | `defaults` or `NUM_OVERLAP=32 BATCH_SIZE=8` |
| `ds_names` | Dataset name(s), comma-separated | `SatNav` |
| `ds_paths` | Trajectory data path(s), comma-separated | `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/trajectory_data` |

**Default SatNav paths (ver_260317):**
- Trajectory data: `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/trajectory_data`

---

## Step 1 → Read & Interpret Plan

1. Read the plan file specified by the user (e.g. `runtime/plans/my_plan.md`).
2. Extract the following from natural language:
   - **Model**: `swiftvln` (default: `swiftvln`)
   - **Env**: `satnav` or `habitat` (default: `satnav`)
   - **Data version**: e.g. `ver_260317` (default: latest in `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_*`)
   - **Experiment list**: each experiment's name/id and any non-default hyperparameter overrides
3. Build a structured experiment list. For each experiment, assign:
   - A short `exp_id` (will appear in the model name for traceability)
   - The `config` string (or `default` if no overrides)

### Critical interpretation rules

**"X 组" / "X 次" / "跑 X 个" 永远指总实验数，不是每台服务器的数量。**
- "跑两组" = 整个计划共 2 个实验，最终会产出 2 个模型
- 服务器分配是内部细节（Step 3），对用户不可见
- 例：4 个实验分配到 2 台服务器 → 每台跑 2 个，但用户描述的是"4 组"

**当实验描述有歧义时，必须在 Step 2 确认阶段列出你的理解并向用户确认，不要自行假设。**

常见歧义情况：
- "baseline 跑两组" → 两组具体是什么？需要问：是 2 组相同重复实验？还是 2 组不同超参？
- "改一个参数跑两组" → 两组是 [改了的] 和 [没改的]？还是 2 种不同的改法？
- "和上次一样再跑两组" → 需要确认"上次"具体是哪个配置

**Example interpretation:**

> "跑 swiftvln satnav 四组实验：纯 baseline，overlap=16，log_base=2.0，以及 overlap=32"

此例描述完整，直接解析为：
```
Experiment 1: exp_id=baseline,   model=swiftvln, config=default
Experiment 2: exp_id=ovlp16,     model=swiftvln, config=NUM_OVERLAP=16
Experiment 3: exp_id=log2,       model=swiftvln, config=LOG_BASE=2.0
Experiment 4: exp_id=ovlp32,     model=swiftvln, config=NUM_OVERLAP=32
```

> "baseline 基础上 log base 改成 2.0 跑两组"

此例有歧义：指定了 LOG_BASE=2.0，但"两组"的具体差异不明确。应在 Step 2 问用户：
```
我理解你要在 baseline 基础上把 LOG_BASE 改成 2.0，但"两组"具体是指：

  A) 两组重复实验？例如：LOG_BASE=2.0 重复跑两次
  B) LOG_BASE=2.0 和 LOG_BASE=1.0（默认）各跑一次做对比？
  C) 其他？

请确认后我再生成实验列表。
```

---

## Step 2 → Confirm with User

Present a clear table of all extracted experiments. **Wait for user confirmation before proceeding.**

If there is **any ambiguity** in the plan (see Step 1 rules above), ask the clarifying questions first, then show the finalized table after the user answers.

Example output after clarification:
```
我理解你要跑以下 4 个实验（swiftvln, satnav, ver_260317）：

 #  | exp_id        | 超参覆盖
----|---------------|----------
 1  | baseline      | (defaults)
 2  | ovlp16        | NUM_OVERLAP=16
 3  | log2          | LOG_BASE=2.0
 4  | ovlp32        | NUM_OVERLAP=32

共 4 个实验，将分配到可用服务器（服务器分配在下一步）。是否继续？
```

If the user requests changes, update the experiment list and re-confirm.

---

## Step 3 → Check Servers & Allocate

Check GPU availability on all three servers:

```bash
# Server 98 (local)
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# Server 73
ssh -o BatchMode=yes -o ConnectTimeout=8 10.246.152.73 nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# Server 17 (Docker)
ssh -o BatchMode=yes -o ConnectTimeout=8 10.246.132.17 "docker ps" 2>/dev/null
# Then check GPU inside container
```

**Allocation rules:**
- A server is "available" if it has 8 free GPUs (memory.used < 500 MiB on all 8).
- Distribute experiments greedily: fill one server first (serial queue), then overflow to next.
- Prefer 98 → 73 → 17.
- If fewer servers available than needed, batch more experiments onto each available server.
- Each server's batch runs **serially** inside that server (train_queue.sh default behavior).

**Wait for user to confirm allocation plan.**

---

## Step 4 → Generate EXPERIMENTS Files & Launch Training

> **Steps 4 and 5 are atomic. Must execute both without interruption.**

For each server with allocated experiments:

### 4.1 — Generate the EXPERIMENTS bash file

Use the Write tool to create `/tmp/train_experiments_<server>_<HHMMSS>.sh`. Content template:

```bash
#!/bin/bash
# Auto-generated by orchestrate-plan skill — DO NOT EDIT MANUALLY
# Plan: <plan_file_path>
# Generated: <timestamp>

ENV_TYPE="satnav"
USE_SWANLAB="false"
SWANLAB_PROJECT=""

# Dataset config
_SATNAV_TRAJ="/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/trajectory_data"
DATASET_CONFIGS=("SatNav|${_SATNAV_TRAJ}")

# EXPERIMENTS array: model|config|changes|ds_names|ds_paths
EXPERIMENTS=(
  "swiftvln|default|defaults|SatNav|${_SATNAV_TRAJ}"
  "swiftvln|NUM_OVERLAP=16|NUM_OVERLAP=16|SatNav|${_SATNAV_TRAJ}"
)
```

Rules:
- One file per server, placed in `/tmp/` (or `runtime/plans/generated/` for persistence).
- `config` field: `default` for all-default, or comma-separated overrides like `NUM_OVERLAP=32,BATCH_SIZE=8`.
- `changes` field: human-readable, e.g. `NUM_OVERLAP=32` or `defaults`.

### 4.2 — Verify offline base model exists

```bash
test -d /mnt/data1/home/jiangjiajun/.cache/modelscope/models/Qwen/Qwen2___5-VL-3B-Instruct && echo OK
```

Fail fast if missing.

### 4.3 — Launch train_queue.sh in tmux (per server)

Session naming: `train_plan_<short_plan_name>_<HHMMSS>`

**Server 98 (local):**
```bash
session_name="train_plan_<short_name>_$(date +%H%M%S)"
run_log="logs/train_launch/${session_name}.log"
mkdir -p logs/train_launch

tmux new-session -d -s "${session_name}" \
  "source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh && \
   conda activate swift-vln-train && \
   TRAIN_EXPERIMENTS_FILE='/tmp/train_experiments_98_HHMMSS.sh' \
   bash src/swiftvln/scripts/train/train_queue.sh 2>&1 | tee ${run_log}"
```

**Server 73 (remote):**
```bash
ssh 10.246.152.73 "tmux new-session -d -s '${session_name}' \
  'source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh && \
   conda activate swift-vln-train && \
   TRAIN_EXPERIMENTS_FILE=\"/tmp/train_experiments_73_HHMMSS.sh\" \
   bash /mnt/data1/home/jiangjiajun/workspace/SwiftVLN-refactor/src/swiftvln/scripts/train/train_queue.sh \
   2>&1 | tee /mnt/data1/home/jiangjiajun/workspace/SwiftVLN-refactor/${run_log}'"
```

---

## Step 5 → Register Train Watchdog (MANDATORY)

For each server, register a watchdog **immediately** after tmux launch. Must run on the same server as the tmux session.

```bash
SWIFTVLN_ROOT="/mnt/data1/home/jiangjiajun/workspace/SwiftVLN-refactor"
nohup bash "${SWIFTVLN_ROOT}/src/swiftvln/scripts/train/train_watchdog.sh" \
  --tmux-session "${session_name}" \
  --train-log "${run_log}" \
  --on-all-done eval \
  --cleanup-days 7 \
  > /dev/null 2>&1 &
WATCHDOG_PID=$!
sleep 2
kill -0 "$WATCHDOG_PID" 2>/dev/null && echo "Watchdog alive PID=${WATCHDOG_PID}" || echo "ERROR: watchdog failed"
```

`--on-all-done eval` means: when training finishes on this server, the watchdog will automatically:
1. Collect successful model names
2. Trigger a new Codex session to start eval (via `swiftvln-eval` skill)

**Record per server**: tmux session name, host, log path, watchdog PID, start time.

---

## Step 6 → Report & Exit (Wait for Eval)

After all servers are launched:

1. Do a quick health check on each tmux session:
   ```bash
   tmux capture-pane -pt "<session_name>" -S -20 2>/dev/null | tail -20
   ```
2. Report to user:
   - Training running on server(s) `<hosts>` in tmux session(s) `<names>`
   - Watchdog PID(s): `<pids>`
   - Progress: `tmux attach -t <name>`
   - What happens next: "训练完成后 watchdog 自动触发评测，评测完成后 Codex 将自动回调并返回结果"

**The Codex session can safely end here.** Everything from this point is handled automatically.

---

## Step 7 → Eval Completion Callback (Codex Resume)

When all evals finish, `eval_watchdog` resumes this Codex session. The injected prompt will say evaluation is done.

Upon resume:

1. Read `runtime/eval_queue/eval_done.txt` to confirm which models finished.
2. Check `runtime/eval_queue/eval_failed_todo.txt` for any failures.
3. Run CSV collection:
   ```bash
   cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN-refactor
   source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
   conda activate swift-vln-eval
   python src/swiftvln/scripts/eval/collect_eval_results.py
   ```
4. Read the output CSV at `results/eval_collected/eval_results_data<version>.csv`.
5. Present results to the user as a formatted table showing SR / SPL / NE for each experiment.
6. If there are failures, diagnose and offer to requeue.

---

## Operating Rules

1. **Always wait for user confirmation** at Step 2 (experiment list) and Step 3 (server allocation).
2. **Steps 4 and 5 are atomic** — register watchdog immediately after tmux launch, never skip.
3. **Report watchdog PID** — missing PID means Step 5 was skipped.
4. **Train hosts**: 98, 73, 17 all supported. **Eval hosts**: 98 and 73 only (never eval on 17).
5. **TRAIN_EXPERIMENTS_FILE** must exist on the target server's local filesystem before tmux launch. Since all three servers share `/mnt/data1/...`, generating to any subpath there works for all.
6. **Prefer `runtime/plans/generated/`** over `/tmp/` for EXPERIMENTS files — persists across reboots for debugging.
7. **Do not modify existing eval_todo.txt entries** — train_queue.sh's `enqueue_model_for_eval` handles auto-enqueue after each successful training.
8. **Data version**: always verify latest `ver_*` under `/mnt/data3/jiangjiajun/dataset/satnav_datasets/` unless plan specifies otherwise.
9. **Fail fast** on missing base model, data paths, or conda envs.
10. **Do not commit** unless user explicitly asks.

---

## Quick Reference: EXPERIMENTS Array Examples

### Baseline
```bash
EXPERIMENTS=(
  "swiftvln|default|defaults|SatNav|/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/trajectory_data"
)
```

### Baseline + custom override (e.g. NUM_OVERLAP=32)
```bash
EXPERIMENTS=(
  "swiftvln|NUM_OVERLAP=32|NUM_OVERLAP=32|SatNav|/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/trajectory_data"
)
```

### Multi-experiment file example (4 experiments across one server)
```bash
ENV_TYPE="satnav"
USE_SWANLAB="false"
SWANLAB_PROJECT=""

_T="/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/trajectory_data"
EXPERIMENTS=(
  "swiftvln|default|defaults|SatNav|${_T}"
  "swiftvln|NUM_OVERLAP=16|NUM_OVERLAP=16|SatNav|${_T}"
  "swiftvln|LOG_BASE=2.0|LOG_BASE=2.0|SatNav|${_T}"
  "swiftvln|NUM_OVERLAP=32|NUM_OVERLAP=32|SatNav|${_T}"
)
```
