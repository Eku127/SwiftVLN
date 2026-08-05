---
name: swiftvln-train
description: "Launch VLN training with tmux-based async execution and direct status checks. Supports multi-server parallel launch and automatic eval enqueue after successful training."
---

# SwiftVLN Train Skill

Launch VLN training on one or more servers. Codex acts as a **launch operator**: confirm the plan, pick hosts, start training in tmux, and verify the process through tmux output and queue status files.

Related skills:
- **`swiftvln-eval`**: run eval after training completes (triggered manually or by user).
- **`server-train-eval-monitor`**: cluster-wide status overview.

## Current Baseline Defaults

- 主训练脚本当前 baseline 默认值：
  - `MAX_SAMPLES=0`（显式全量）
  - `NUM_OVERLAP=0`
  - `NUM_OVERLAP>0` 时固定使用 stride-aligned tail window，不再提供 tail-adjust 控制参数
  - `SAVE_STEPS=1000`
  - `SAVE_TOTAL_LIMIT=1`
  - SatNav 默认训练数据：`SatNav-v0.1`
  - 当前 0418 默认 eval split 口径：
    - `val_seen = 4574`
    - `val_unseen = 8756`
    - `2026-04-20` 已从 `val_seen` 中移除 `27` 个与 train 路线重复的 episodes
    - `val_seen_update` 已不再作为当前默认 eval 目录存在
- 历史 baseline 默认曾使用 `NUM_OVERLAP=16`；如果用户没有特别说明，当前应按 `overlap=0` 理解 baseline。
- 主训练脚本默认 base model 仍是本地 `3B`：
  - `/mnt/data1/home/jiangjiajun/.cache/modelscope/models/Qwen/Qwen2___5-VL-3B-Instruct`
- 如需训练 `7B`，现在可以直接通过环境变量覆盖，而不必改脚本默认值：
  - `BASE_MODEL_PATH=/mnt/data1/home/jiangjiajun/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-7B-Instruct`
  - 或 `MODEL_PATH=/mnt/data1/home/jiangjiajun/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-7B-Instruct`
  - 关键训练超参也支持 env 覆盖，如：
    - `BATCH_SIZE`
    - `LEARNING_RATE`
    - `NUM_EPOCHS`
    - `GRAD_ACCUM_STEPS`
    - `NUM_HISTORY`
    - `HISTORY_PROCESSOR_TYPE`
    - `NUM_OVERLAP`
- 已在 `17` 上做过 `Qwen2.5-VL-7B` baseline 实测：
  - `8卡`
  - `per_frame + history + overlap=0`
  - `bsz=12` 可稳定训练
- 若在 `17` 上用 `7B` 正式长跑遇到 `pin_memory` 线程里的
  `torch.AcceleratorError: CUDA error: invalid argument`，
  当前推荐直接覆盖：
  - `DATALOADER_PIN_MEMORY=false`
  - 并优先使用更保守的 `BATCH_SIZE=10`

## No-Memory Convention (SwiftVLN)

- SwiftVLN 没有单独的 `USE_MEMORY=false` 开关。
- 当前仓库约定的 **effective no-memory** 配置是：
  - `HISTORY_PROCESSOR_TYPE=per_frame`
  - `NUM_HISTORY=0`
- 该配置下 dataset 会采样 `0` 张历史帧，system prompt 不会插入 `<history_memory>`。
- `per_frame` 的实验名会显式带上 `pf-h0-nomem-...`，便于和普通 `pf-h8/...` 区分。
- `log_base` / `use_random` 仍可保留在配置中，但在 `NUM_HISTORY=0` 时不会实际影响采样。
- `gtc` / `segment_gtc` 不适用这套 no-memory 约定；它们的历史采样逻辑不依赖 `NUM_HISTORY`。

## Map-Memory Convention (SwiftVLN)

- SwiftVLN 现在支持 `MEMORY_METHOD=map`，表示用 `global map + local map` 替换历史帧 memory。
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
- 当前推荐把 `SWIFTVLN_DEBUG=1` 一并透传，用于检查 dataset prompt、template tokenize、history token 注入是否符合预期。

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
| Model(s) | `swiftvln` | `baseline`/ambiguous → `swiftvln` |
| Environment | — | `satnav` or `habitat` |
| Server(s) | — | One or more of: `98`, `73`, `17` |
| Base model path | `/mnt/data1/home/jiangjiajun/.cache/modelscope/models/Qwen/Qwen2___5-VL-3B-Instruct` | Default script path; use this absolute local cache path to avoid ModelScope hub resolution |

---

## Step 1 → Confirm Plan & Run Checklist

1. Read current scripts:
   - `src/swiftvln/scripts/train/train_queue.sh`
   - `src/swiftvln/model/script/train/train_swiftvln_qwen_vl.sh`
2. Produce **run checklist**: model set, environment, offline model path, launch mode, expected output naming.
   Confirm the resolved base model path is the absolute local cache path above, not `Qwen/Qwen2.5-VL-3B-Instruct`.
   If `MEMORY_METHOD=map`, checklist 里必须额外确认：
   - `global/local/render/mask`
   - 约束 `satnav + per_frame + no ToMe`
   - 预期实验名中是否包含 `map-g...-l...-r...-...-s...`
3. Apply default model rule: `baseline`/unspecified → `swiftvln`.
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
2. Resolve trajectory data paths.
3. Verify offline base model: `test -d /mnt/data1/home/jiangjiajun/.cache/modelscope/models/Qwen/Qwen2___5-VL-3B-Instruct && echo OK`
   If missing, fail fast instead of falling back to a remote `model_id`.
4. If user requests, sync `satnav_task.yaml` and `train_queue.sh`.
5. **Fail fast** if any path is missing.
6. **Wait for user confirmation**.

## Step 4 → Launch Training in tmux (per server)

> **All training must be launched in tmux.** For each server, create a separate tmux session.

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
   conda activate swift-vln-train-update && \
   TRAIN_NUM_GPUS=${TRAIN_NUM_GPUS:-} \
   TRAIN_CUDA_DEVICES=${TRAIN_CUDA_DEVICES:-} \
   bash src/swiftvln/scripts/train/train_queue.sh 2>&1 | tee ${run_log}"
```

### Launch pattern (remote server 73)

```bash
ssh 10.246.152.73 "tmux new-session -d -s '${session_name}' \
  'source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh && \
   conda activate swift-vln-train-update && \
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
     conda activate swift-vln-train-update && \
     TRAIN_NUM_GPUS=${TRAIN_NUM_GPUS:-} \
     TRAIN_CUDA_DEVICES=${TRAIN_CUDA_DEVICES:-} \
     bash /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/src/swiftvln/scripts/train/train_queue.sh 2>&1 | \
     tee /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/${run_log}\"'"
```

**Record**: tmux session name, host, log path, start time for each server.

## Step 5 → Quick Health Check

启动后检查 1–2 轮，确认 tmux 存活、训练日志开始推进且没有立即报错：

```bash
# tmux alive?
tmux has-session -t "${session_name}" 2>/dev/null && echo "OK"

# Training output (check for start or fatal errors)
tmux capture-pane -pt "${session_name}" -S -30 2>/dev/null | tail -30
```

远程任务通过对应 SSH 主机执行同样的 `tmux has-session` 和 `capture-pane` 检查。若用户要求持续监控，继续轮询 tmux、训练日志以及 `runtime/train_queue/train_queue_last_run_<hostname>.json`；不要启动额外后台监控脚本。

## Step 6 → Report & Exit

Report to user:

- Training running on server(s) `<hosts>` in tmux session(s) `<names>`
- 进度查看：`tmux attach -t <name>`
- 启动日志：`logs/train_launch/<session>.log`
- 完成状态：`runtime/train_queue/train_queue_last_run_<hostname>.json`

训练成功后 `train_queue.sh` 会继续自动加入 eval todo；启动 eval worker 是独立操作，不依赖训练 sidecar。

---

## Default Script Paths

| Purpose | Path |
|---|---|
| Training queue | `src/swiftvln/scripts/train/train_queue.sh` |
| SwiftVLN single run | `src/swiftvln/model/script/train/train_swiftvln_qwen_vl.sh` |
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

The eval result recorder reads this file to carry the SwanLab URL into `evaluation_summary.json`.

---

## Operating Rules

1. **Pause for user confirmation** on Step 1→2→3 transitions. Once confirmed, **proceed through Steps 4→5→6 automatically**.
2. **Always launch training in tmux**. Use naming: `train_<short_desc>_<HHMMSS>`.
3. 默认不指定 GPU 参数时，训练脚本会自动使用当前环境里全部可见 GPU；不要再默认假设是 8 卡。
4. 若用户指定卡数，用 `TRAIN_NUM_GPUS=<N>`；若用户指定具体卡位，用 `TRAIN_CUDA_DEVICES=<csv>`。
5. tmux 启动后必须执行 Step 5 的快速健康检查。
6. **Multi-server**: each server gets its own tmux session and log file. They run independently.
7. **Prefer existing project scripts** over ad-hoc logic.
8. **Eval enqueue is always a local file operation** (shared filesystem).
9. **Fail fast** on missing datasets, checkpoints, or conda envs.
10. **Do not commit** unless user explicitly asks.
