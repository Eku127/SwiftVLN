---
name: server-train-eval-monitor
description: Monitor training/evaluation status across servers 98, 73, and 17 (Docker). Use when user asks to "看一圈服务器", "监控使用情况", "训练评估进度", "现在啥进度", "哪些任务在跑", or requests GPU usage, running jobs, current progress, and total experiment-group counts.
---

# Server Train Eval Monitor

巡检 98/73/17 三台机器的训练与评估状态，输出统一摘要：GPU 使用、正在运行任务、训练/评估进度、以及总实验组数。

## Inputs

- 默认：单次巡检（1 sweep）。
- 若用户明确要求持续监控，可启用循环模式：
  - `sleep_seconds`：每轮巡检间隔秒数（例如 30 / 60 / 300）。
  - `rounds`：轮数；未指定时默认 `10` 轮。
  - `rounds=0` 表示持续监控，直到用户叫停。

## Workflow

1. Check all servers.
- `98`: local host.
- `73`: SSH host `10.246.152.73`.
- `17`: SSH host `10.246.132.17`, then inspect inside `streamvln-container`.

2. Collect GPU usage.
- Run `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`.
- For `17`, run inside Docker container.

3. Collect running training/eval jobs.
- Inspect `tmux ls` and relevant panes with `tmux capture-pane -pt <session> | tail -n 120`.
- Inspect processes with:
`ps -eo pid,ppid,etime,cmd | grep -E "train_queue.sh|eval_queue.sh|torchrun|swift sft|eval_by_name.sh|start_eval_worker.sh" | grep -v grep`.

4. Collect queue files and totals.
- Training queue totals:
  - Read `logs/train_queue_*.log` and parse `实验列表 (共 X 个实验)` and latest `进度: a / b`.
- Eval queue totals:
  - Check:
    - `runtime/eval_queue/eval_todo.txt`
    - `runtime/eval_queue/eval_done.txt`
    - `runtime/eval_queue/eval_failed_todo.txt`
  - Report counts and key model names (first few lines).

5. For each active training task, report progress.
- If queue-style: use `进度: a / b`.
- If trainer-style: parse latest `global_step/max_steps` or `Train: xx%`.
- Always include whether currently healthy (loss lines advancing) or stalled/erroring.

6. If loop mode is enabled:
- Run the full sweep every `sleep_seconds`.
- For round 1: output full snapshot.
- For round 2+ : output delta-focused updates first (new process, finished job, queue变化, GPU明显波动), then current summary.
- Sleep with `sleep <sleep_seconds>` between rounds.
- Stop when rounds reached, or user asks to stop.

7. Return concise summary first, then details.
- Put high-level status first:
  - `98`: running/idle + current run progress + total groups.
  - `73`: running/idle + eval worker/queue summary.
  - `17`: running/idle + current run progress + total groups.
- Then include notable risks/errors (port conflict, dataloader crash, no enqueue, etc.).

## Required Checks

- Always check all 3 servers in one sweep, even if user only mentions one.
- Always include total experiment-group count when detectable (from queue summary or progress `a/b`).
- Always include evaluation queue counts (`todo/done/failed`) when eval files exist.
- If data is stale or ambiguous, say so explicitly and state what was missing.
- In loop mode, every round must still include all 3 servers; do not reduce to partial checks.
- If any SSH/权限失败，标记该轮为 `partial` 并继续下一轮（除非用户要求立即停止）。

## Output Template

Use this structure:

1. Overall
- One-line health summary of 98/73/17.

2. Server 98
- GPU summary.
- Running tasks.
- Current progress (`a/b` or `%`) and total groups.
- Eval queue counts if available.

3. Server 73
- GPU summary.
- Running tasks.
- Eval queue worker and `todo/done/failed` counts.

4. Server 17 (Docker)
- GPU summary from container.
- Running tasks from container/tmux.
- Current progress and total groups.

5. Risks / Follow-ups
- Errors seen, retries, and whether enqueue-to-eval succeeded.
