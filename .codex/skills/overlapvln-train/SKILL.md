---
name: overlapvln-train
description: "Launch VLN training with tmux-based async execution and webhook watchdog monitoring. Supports multi-server parallel launch and webhook notifications on completion/failure/stall."
---

# OverlapVLN Train Skill

Launch VLN training on one or more servers. Codex acts as a **launch operator**: confirm plan, pick hosts, start training in tmux, register watchdog, do quick health check, then exit. The watchdog runs in background and sends webhook notifications on key events.

Related skills:
- **`overlapvln-eval`**: run eval after training completes (triggered manually or by user).
- **`server-train-eval-monitor`**: cluster-wide status overview.

## No-Memory Convention (OverlapVLN)

- OverlapVLN 没有单独的 `USE_MEMORY=false` 开关。
- 当前仓库约定的 **effective no-memory** 配置是：
  - `HISTORY_PROCESSOR_TYPE=per_frame`
  - `NUM_HISTORY=0`
- 该配置下 dataset 会采样 `0` 张历史帧，system prompt 不会插入 `<history_memory>`。
- `per_frame` 的实验名会显式带上 `pf-h0-nomem-...`，便于和普通 `pf-h8/...` 区分。
- `log_base` / `use_random` 仍可保留在配置中，但在 `NUM_HISTORY=0` 时不会实际影响采样。
- `gtc` / `segment_gtc` 不适用这套 no-memory 约定；它们的历史采样逻辑不依赖 `NUM_HISTORY`。

## Map-Memory Convention (OverlapVLN)

- OverlapVLN 现在支持 `MEMORY_METHOD=map`，表示用 `global map + local map` 替换历史帧 memory。
- 当前约束：
  - `VLN_ENV_TYPE=satnav`
  - `HISTORY_PROCESSOR_TYPE=per_frame`
  - `USE_TOME=false`
- 训练脚本会把 map 参数写进实验名，格式为：
  - `map-g{global}-l{local}-r{render}-{mask}-s{compress_stride}`
  - 例：`map-g1000-l400-r384-d20-s2`
- 训练脚本中的 map 相关变量：
  - `MEMORY_METHOD`
  - `MAP_GLOBAL_SIDE_M`
  - `MAP_LOCAL_SIDE_M`
  - `MAP_RENDER_PX`
  - `MAP_MASK_METHOD`
- `train_queue.sh` 会透传并覆写这些变量；如需入队训练 map memory，必须把这几个变量一起明确写进配置。
- 当前推荐把 `OVERLAPVLN_DEBUG=1` 一并透传，用于检查 dataset prompt、template tokenize、history token 注入是否符合预期。

---

## Shared Workspace Mount

Servers `98`, `73`, and `17` all mount:

```
/mnt/data1/home/jiangjiajun/workspace/SwiftVLN
```

Queue files, outputs, logs are all local file operations. SSH only for GPU checks and remote tmux launch.

---

## Inputs

Before starting, confirm with the user:

| Parameter | Default | Notes |
|---|---|---|
| Model(s) | `overlapvln` | `baseline`/ambiguous → `overlapvln` |
| Stage | — | `stage1` or `stage2` |
| Environment | — | `satnav` or `habitat` |
| Server(s) | — | One or more of: `98`, `73`, `17` |
| QA mixed training | — | Whether to mix QA data |
| Stage1 base model path | `/mnt/data1/home/jiangjiajun/.cache/modelscope/models/Qwen/Qwen2___5-VL-3B-Instruct` | Default script path; use this absolute local cache path to avoid ModelScope hub resolution |

---

## Step 1 → Confirm Plan & Run Checklist

1. Read current scripts:
   - `src/swiftvln/scripts/train/train_queue.sh`
   - `src/swiftvln/model/script/train/train_overlapvln_qwen2_5_vl.sh`
2. Produce **run checklist**: model set, stage, environment, data version, offline model path, launch mode, expected output naming.
   For `stage1`, confirm the resolved path is the absolute local cache path above, not `Qwen/Qwen2.5-VL-3B-Instruct`.
   If `MEMORY_METHOD=map`, checklist 里必须额外确认：
   - `global/local/render/mask`
   - 约束 `satnav + per_frame + no ToMe`
   - 预期实验名中是否包含 `map-g...-l...-r...-...-s...`
3. Apply default model rule: `baseline`/unspecified → `overlapvln`.
4. **Wait for user confirmation**.

## Step 2 → Check Servers & Pick Host(s)

1. Inspect GPU availability on requested servers:
   - `98` — local `nvidia-smi`
   - `73` — `ssh -o BatchMode=yes -o ConnectTimeout=8 10.246.152.73 nvidia-smi`
   - `17` — `ssh -o BatchMode=yes -o ConnectTimeout=8 10.246.132.17 nvidia-smi` + Docker check
2. For multi-server launch, verify each requested server has enough **currently visible** GPUs for the planned run.
   - 默认不再假设必须 8 卡。
   - 默认行为：训练脚本自动使用当前环境里**可见的全部 GPU**。
   - 若需要指定卡数：设置 `TRAIN_NUM_GPUS=<N>`。
   - 若需要指定具体卡列表：设置 `TRAIN_CUDA_DEVICES=0,1,3,5`。
3. **Wait for user confirmation**.

## Step 3 → Pin Dataset Version & Config

1. Discover latest SatNav dataset under `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_*`.
2. Resolve data paths (trajectory, QA).
3. Verify offline base model: `test -d /mnt/data1/home/jiangjiajun/.cache/modelscope/models/Qwen/Qwen2___5-VL-3B-Instruct && echo OK`
   If missing, fail fast instead of falling back to a remote `model_id`.
4. If user requests, sync `satnav_task.yaml` and `train_queue.sh`.
5. **Fail fast** if any path is missing.
6. **Wait for user confirmation**.

## Step 4 → Launch Training in tmux (per server)

> **All training must be launched in tmux.** For each server, create a separate tmux session.
> **⚠️ Step 4 和 Step 5 是原子操作：启动 tmux 后必须立即注册 watchdog，不可跳过。**

### tmux naming

```
train_<short_desc>_<HHMMSS>
```

### Launch pattern (local server 98)

```bash
session_name="train_ovlpvln_$(date +%H%M%S)"
run_log="logs/train_launch/${session_name}.log"
mkdir -p logs/train_launch

tmux new-session -d -s "${session_name}" \
  "source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh && \
   conda activate swift-vln-train && \
   TRAIN_NUM_GPUS=${TRAIN_NUM_GPUS:-} \
   TRAIN_CUDA_DEVICES=${TRAIN_CUDA_DEVICES:-} \
   bash src/swiftvln/scripts/train/train_queue.sh 2>&1 | tee ${run_log}"
```

### Launch pattern (remote server 73)

```bash
ssh 10.246.152.73 "tmux new-session -d -s '${session_name}' \
  'source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh && \
   conda activate swift-vln-train && \
   TRAIN_NUM_GPUS=${TRAIN_NUM_GPUS:-} \
   TRAIN_CUDA_DEVICES=${TRAIN_CUDA_DEVICES:-} \
   bash /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/src/swiftvln/scripts/train/train_queue.sh 2>&1 | \
   tee /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/${run_log}'"
```

### Launch pattern (remote server 17 with Docker)

```bash
ssh 10.246.132.17 "docker exec -d streamvln-container bash -c \
  'tmux new-session -d -s ${session_name} \
    \"source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh && \
     conda activate swift-vln-train && \
     TRAIN_NUM_GPUS=${TRAIN_NUM_GPUS:-} \
     TRAIN_CUDA_DEVICES=${TRAIN_CUDA_DEVICES:-} \
     bash /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/src/swiftvln/scripts/train/train_queue.sh 2>&1 | \
     tee /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/${run_log}\"'"
```

**Record**: tmux session name, host, log path, start time for each server.

## Step 5 → Register Watchdog & Quick Health Check (MANDATORY)

> **🚨 此步骤为 MANDATORY（强制），不可跳过。**
> Watchdog 负责：训练结束/崩溃/停滞时发 webhook 通知，训练结束后**自动退出**。

### 5.1 — 注册 watchdog（每台服务器各一个）

Watchdog 必须运行在**与 tmux session 相同的服务器上**。

**训练在 98 上（本地启动）：**

```bash
SWIFTVLN_ROOT="/mnt/data1/home/jiangjiajun/workspace/SwiftVLN"
nohup bash "${SWIFTVLN_ROOT}/src/swiftvln/scripts/train/train_watchdog.sh" \
  --tmux-session "${session_name}" \
  --train-log "${run_log}" \
  --on-all-done notify \
  --cleanup-days 7 \
  > /dev/null 2>&1 &
WATCHDOG_PID=$!
echo "Watchdog PID=${WATCHDOG_PID}"
```

**训练在 73 或 17 上（SSH 到对应服务器启动）：**

```bash
SWIFTVLN_ROOT="/mnt/data1/home/jiangjiajun/workspace/SwiftVLN"
ssh 10.246.152.73 "cd ${SWIFTVLN_ROOT} && \
  nohup bash src/swiftvln/scripts/train/train_watchdog.sh \
    --tmux-session '${session_name}' \
    --train-log '${run_log}' \
    --on-all-done notify \
    --cleanup-days 7 \
    > /dev/null 2>&1 & echo \$!"
```

### 5.2 — 验证 watchdog 存活

```bash
sleep 2
kill -0 "$WATCHDOG_PID" 2>/dev/null && echo "✅ Watchdog alive (PID=${WATCHDOG_PID})" || echo "❌ Watchdog failed"
```

如果失败：检查脚本路径和 tmux session 名称是否正确，重新注册。

### 5.3 — Quick health check（1-2 轮）

```bash
# tmux alive?
tmux has-session -t "${session_name}" 2>/dev/null && echo "OK"

# Training output (check for start or fatal errors)
tmux capture-pane -pt "${session_name}" -S -30 2>/dev/null | tail -30
```

## Step 6 → Report & Exit

Report to user:

- Training running on server(s) `<hosts>` in tmux session(s) `<names>`
- **Watchdog PID**: `<pid>`（若为空则说明 Step 5 未执行）
- 进度查看：`tmux attach -t <name>`
- Watchdog 日志：`runtime/train_queue/runs/<hostname>_<session>/watchdog.log`
- 训练完成/崩溃/停滞时：webhook 通知

**The Codex session can safely end here.**

---

## Per-Run Directory Structure

```
runtime/train_queue/runs/<hostname>_<session_name>/
├── watchdog_result.json       watchdog 最终状态
├── watchdog.log               watchdog 运行日志
├── train_queue_status.json    train_queue.sh 写入
└── train_events.log           实验事件（SUCCESS/FAILED/QUEUE_DONE）
```

Auto-cleanup: watchdog cleans dirs older than 7 days at startup.

---

## Default Script Paths

| Purpose | Path |
|---|---|
| Training queue | `src/swiftvln/scripts/train/train_queue.sh` |
| **Train watchdog** | `src/swiftvln/scripts/train/train_watchdog.sh` |
| OverlapVLN single run | `src/swiftvln/model/script/train/train_overlapvln_qwen2_5_vl.sh` |
| Eval todo queue | `runtime/eval_queue/eval_todo.txt` |
| Eval enqueue helper | `src/swiftvln/scripts/eval/enqueue_eval.sh` |

---

## Training Metadata (`train_metadata.json`)

After training completes, `train_queue.sh` writes `train_metadata.json` to `$OUTPUT_DIR`:

```json
{
  "swanlab_url": "https://swanlab.cn/@eku127/StreamVLN/runs/...",
  "swanlab_project": "SatNav",
  "swanlab_exp_name": "<EXP_NAME>"
}
```

Downstream consumers: eval runner (`runner.py`) and CSV collector (`collect_eval_results.py`) both read this file for SwanLab URL.

---

## Operating Rules

1. **Pause for user confirmation** on Step 1→2→3 transitions. Once confirmed, **proceed through Steps 4→5→6 automatically**.
2. **Always launch training in tmux**. Use naming: `train_<short_desc>_<HHMMSS>`.
3. 默认不指定 GPU 参数时，训练脚本会自动使用当前环境里全部可见 GPU；不要再默认假设是 8 卡。
4. 若用户指定卡数，用 `TRAIN_NUM_GPUS=<N>`；若用户指定具体卡位，用 `TRAIN_CUDA_DEVICES=<csv>`。
5. **🚨 MANDATORY: tmux 启动后必须立即注册 watchdog（Step 5）并验证存活。** 不可跳过。
6. **Watchdog 在 tmux session 结束后自动退出**，无需手动清理。
7. **报告中必须包含 watchdog PID**。没有 PID 说明 Step 5 被跳过了。
8. **Multi-server**: each server gets its own tmux session + watchdog. They run independently.
9. **Prefer existing project scripts** over ad-hoc logic.
10. **Eval enqueue is always a local file operation** (shared filesystem).
11. **Do not commit** unless user explicitly asks.
10. **Fail fast** on missing datasets, checkpoints, or conda envs.
