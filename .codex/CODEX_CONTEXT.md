# Codex Repo Context (SwiftVLN)

This file is the local working context for Codex in this repository.
Prefer this file over generic defaults when making decisions.

## Repository Scope

- Repo: `SwiftVLN`
- Primary active domain in this workspace: `src/swiftvln/*` (especially OverlapVLN training/eval and SatNav data workflow).
- Research goal: low-altitude UAV visual-language navigation (city-scale, paper-oriented research).
- Current primary model target: `src/swiftvln/models/overlapvln`.

## Research Task Characteristics

- Viewpoint shift: use satellite-map crops to mimic UAV top-down view instead of common ego-view VLN.
- Long-horizon challenge: city-scale navigation introduces naturally long trajectories and harder end-to-end planning.
- Multi-type task design:
  - `boundary`: loop-closure ability (returning to/around start-relevant area)
  - `LandmarkSet`: orientation and spatial relation understanding
  - `Road`: counting and instruction grounding on numbers/intersections
- Memory design is a current key research direction for improving end-to-end VLN.

## Key Directories

- SwiftVLN package root: `src/swiftvln`
- OverlapVLN core: `src/swiftvln/models/overlapvln`
- SatNav task config: `src/swiftvln/configs/satnav_task.yaml`
- Training queue script: `src/swiftvln/scripts/train/train_queue.sh`
- Eval scripts:
  - `src/swiftvln/scripts/eval/eval_by_name.sh`
  - `src/swiftvln/scripts/eval/eval_queue.sh`
- Eval results root: `results/eval`
- Docker helper scripts: `src/swiftvln/scripts/docker`
- Data processing scripts: `src/swiftvln/scripts/data_process/*.py`
- Data sync scripts: `src/swiftvln/scripts/data_sync/*.sh`
- Local repo skills: `.codex/skills/*`

## Current SatNav Dataset Defaults

- Dataset root: `/mnt/data3/jiangjiajun/dataset/satnav_datasets`
- Latest version in active use: `ver_260227`
- Eval episodes path:
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260227/episodes/eval/all_episodes.json`
- QA JSONL path:
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260227/data/qa_swift.jsonl`
- Trajectory data path:
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260227/trajectory_data`

## Workflow Conventions

- When user asks to process SatNav data, use `.codex/skills/satnav-data/SKILL.md`.
- Execute SatNav pipeline in staged confirmations:
  1) data processing
  2) trajectory generation
  3) sync + verify
  4) sync project config paths
- After finishing a new SatNav version, also update:
  - `src/swiftvln/configs/satnav_task.yaml` (`DATASET.DATA_PATH`)
  - `src/swiftvln/scripts/train/train_queue.sh` (`QA_DATASET`, `default_satnav_path`)
- Prefer training/eval orchestration through existing queue scripts instead of ad-hoc command assembly.
- Before launching actual training settings, provide a pre-run confirmation checklist to user and wait for approval.
- Before each training launch, check 3-server status and pick an idle node:
  - `98` (current host): direct training.
  - `73` (`10.246.152.73`): same mounted workflow as 98 via SSH.
  - `17` (`10.246.132.17`): run training inside Docker container (often managed from tmux).
- Workspace mount note: `98`/`73`/`17` share the **same NFS-mounted workspace** at `/mnt/data1/home/jiangjiajun/workspace/SwiftVLN`.
  - All workspace files (scripts, queue files, outputs, configs) are **locally accessible from every host**.
  - SSH is only needed for process/GPU/container status checks and remote process launch/kill.
  - **No SSH is required for file operations** (reading queue files, writing eval_todo, etc.).
- Eval execution hosts: `98` and `73` only.
- Recommended eval mode: run a long-lived worker via `src/swiftvln/scripts/eval/start_eval_worker.sh` and feed model names through `src/swiftvln/scripts/eval/eval_todo.txt`.
- Train→eval bridge: after successful checkpoint save, **directly append** model name to `src/swiftvln/scripts/eval/eval_todo.txt` (local file write, no wrapper script, no SSH).
- Eval queue lifecycle: success moves model name to `src/swiftvln/scripts/eval/eval_done.txt`; failure moves to `src/swiftvln/scripts/eval/eval_failed_todo.txt`.
- Queue scripts should try automatic fixes/retries on common runtime failures before final failure.
- Send webhook notifications for retry/failure events and final completion summaries (train and eval).
- Eval auto-fix must not downgrade to single-GPU on OOM unless user explicitly requests.

## Dependency Note

- SwiftVLN has been migrated as an independent repo layout.
- Current runtime environment may still import `swift` package from external installation/workspace.
- Do not modify external repos unless user explicitly requests; prefer adapting SwiftVLN-side paths/scripts first.

## Commit Style (Project Preference)

- Prefer conventional commit types: `feat`, `fix`, `refactor`, `docs`, `test`, `perf`, `chore`.
- Use concise Chinese commit subject lines when committing project changes.
- Keep commits atomic and avoid bundling unrelated dirty files.

## Operating Rules for Codex in This Repo

- Prefer existing project scripts over ad-hoc one-off scripts.
- Verify key outputs before reporting completion (files, counts, sizes).
- For risky or environment-dependent steps (remote sync, cluster ops), report exact commands and blocking reason on failure.
- Do not revert user pre-existing unrelated changes.
- Do not create commits unless user explicitly requests committing.
- Minimize documentation output unless user explicitly asks for docs.
- Remove temporary test/debug scripts after use when they are only for the current task.
- Remove temporary test artifacts/outputs after validation (lock files, scratch logs/results/checkpoints, temporary queue entries) unless user explicitly asks to retain them.
- Default baseline model is `overlapvln` when user does not explicitly specify model.
- Queue failure policy: skip failed runs and continue remaining runs unless user overrides.
