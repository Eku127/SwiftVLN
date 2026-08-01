# Eval Flow (Train → Queue → Eval)

## Architecture

All three servers (`98`, `73`, `17`) share the same NFS-mounted workspace at:

```
/mnt/data1/home/jiangjiajun/workspace/SwiftVLN
```

Queue files, eval outputs, and scripts are locally accessible from every host.
No SSH is required for queue-file operations.

## Contract

- **Producer** (train side): append finished `model_name` to `runtime/eval_queue/eval_todo.txt` — always a local file write.
- **Consumer** (eval side): `start_eval_worker.sh` continuously polls todo and runs serial eval.
- Queue state transition:
  - success: `eval_todo.txt` → `eval_done.txt`
  - failure: `eval_todo.txt` → `eval_failed_todo.txt`

## Host Rules

- Training hosts: `98`, `73`, `17`.
- Eval hosts: `98` and `73` only. Never eval on `17`.
- Enqueue from any host is a local file append (shared filesystem). No SSH needed.

## Key Files

| Purpose | Path |
|---|---|
| Eval queue file | `runtime/eval_queue/eval_todo.txt` |
| Eval done file | `runtime/eval_queue/eval_done.txt` |
| Eval failed file | `runtime/eval_queue/eval_failed_todo.txt` |
| Local enqueue helper | `src/swiftvln/scripts/eval/enqueue_eval.sh` |
| Eval worker starter | `src/swiftvln/scripts/eval/start_eval_worker.sh` |
| Single eval by name | `src/swiftvln/scripts/eval/eval_by_name.sh` |
| Queue runner | `src/swiftvln/scripts/eval/eval_queue.sh` |
| CSV collector | `src/swiftvln/scripts/eval/collect_eval_results.py` |
| Collected CSV | `results/eval_collected/<split>/eval_results.csv` |

## Failure Policy

- Eval queue continues after single-task failure.
- Failed tasks move to `eval_failed_todo.txt` and are recorded in queue logs.
