# AGENTS.md

## Startup Context (Mandatory)

- **每次启动一个新的 agent 时，必须先阅读：**
  - `.codex/CODEX_CONTEXT.md`
- 把该文件视为当前仓库的默认事实来源（路径、脚本、环境、约定）。

## Context Maintenance Rule (Mandatory)

- 若本次任务包含**重大改动**，必须在结束前同步更新 `.codex/CODEX_CONTEXT.md`。
- 重大改动包括但不限于：
  - 目录/包结构重构
  - 训练或评测主流程变化
  - 队列路径、关键脚本路径变化
  - 默认数据版本或关键环境约定变化

## Repo Skills

- If task matches SatNav data processing, use:
  - `.codex/skills/satnav-data/SKILL.md`
- If user asks to merge two SatNav dataset versions, analyze overlap/complementarity, remap conflicting episode IDs, or create a new merged version like 0327 + 0403 -> 0404, use:
  - `.codex/skills/merge-satnav-data/SKILL.md`
- If task matches mainline VLN training (OverlapVLN), use:
  - `.codex/skills/overlapvln-train/SKILL.md`
- If task matches VLN evaluation (eval by name, eval queue, eval monitoring), use:
  - `.codex/skills/overlapvln-eval/SKILL.md`
- If user asks to run StreamVLN baseline training/evaluation on SatNav (基线训练、启动baseline、评测baseline), use:
  - `.codex/skills/run-streamvln-baseline/SKILL.md`
- If user asks for cross-server status巡检, use:
  - `.codex/skills/server-train-eval-monitor/SKILL.md`
- If user asks to periodically check GPU health across 98/73/17, 定时巡检显卡, 持续检查掉卡, or monitor three-server GPU-only status, use:
  - `.codex/skills/gpu-health-monitor/SKILL.md`
- If task is mainline smoke test / 冒烟测试 for OverlapVLN train+eval validation, use:
  - `.codex/skills/swiftvln-smoke-test/SKILL.md`
- If task is any baseline smoke test (streamvln / navila / uninavid — 冒烟测试、smoke test、baseline train+eval quick validation), use:
  - `.codex/skills/baseline-smoke-test/SKILL.md`
- If user asks to run an experiment plan, 执行实验计划, 跑 runtime/plans/…, orchestrate experiments, or run multiple experiment settings end-to-end (train + eval + results), use:
  - `.codex/skills/orchestrate-plan/SKILL.md`

## Execution Rules

- Prefer existing scripts under `src/swiftvln/scripts/` over ad-hoc one-off logic.
- After SatNav data version updates, sync dataset paths in:
  - `src/swiftvln/configs/satnav_task.yaml`
  - `src/swiftvln/scripts/train/train_queue.sh`
