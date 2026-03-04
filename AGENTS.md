# AGENTS.md

## Startup Context

- At the beginning of each new task in this repository, read:
  - `.codex/CODEX_CONTEXT.md`
- Treat that file as the default project background and workflow context.

## Repo Skills

- If task matches SatNav data processing, use:
  - `.codex/skills/satnav-data/SKILL.md`
- If task matches VLN training (OverlapVLN / StreamVLN / CompressVLN), use:
  - `.codex/skills/overlapvln-train/SKILL.md`
- If task matches VLN evaluation (eval by name, eval queue, eval monitoring), use:
  - `.codex/skills/overlapvln-eval/SKILL.md`
- If user asks for cross-server status巡检, use:
  - `.codex/skills/server-train-eval-monitor/SKILL.md`

## Execution Rules

- Prefer existing scripts under `src/swiftvln/scripts/` over ad-hoc one-off logic.
- After SatNav data version updates, sync dataset paths in:
  - `src/swiftvln/configs/satnav_task.yaml`
  - `src/swiftvln/scripts/train/train_queue.sh`
