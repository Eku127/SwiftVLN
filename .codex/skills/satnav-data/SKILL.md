---
name: satnav-data
description: "Process SatNav dataset versions end to end, including unzip data.zip, inspect and classify city/type changes, generate episodes, run trajectory generation, and sync/verify across servers. Use when the user asks to process SatNav data, process data, 处理数据, 处理 SatNav 数据, or mentions SatNav dataset preparation."
---

# Process SatNav Data

Execute a 4-step pipeline **continuously without pausing between steps**. Only stop if a step fails or if a decision requires user input (e.g., new city/type classification). Otherwise, proceed automatically through all steps.

## Inputs

- Require a dataset version string, e.g. `ver_260211`.
- Assume dataset root: `/mnt/data3/jiangjiajun/dataset/satnav_datasets/<version>/`.

## Step 1: Data Processing

1. Unzip `<version>/data.zip` into `<version>/data/`.
2. Inspect available cities and episode statistics.
3. If new cities or episode types appear, ask user how to classify them:
- New city to `train` or `eval`.
- New episode type to include or skip.
4. Update data process configs accordingly.
   - 当前默认划分（0327 起）：
     - `eval`: `LosAngeles-1`, `Rome-1`, `NewYork-1`, `Auckland-1`, `Orlando-1`, `Rotterdam-1`
     - `train`: 其余全部城市（含 `Amsterdam-1`, `Dube-1`）
   - 当前自动分类结果（由 `classify_eval_cities()` 判断）：
     - `val_seen`: `LosAngeles-1`, `Rome-1`, `NewYork-1`
     - `val_unseen`: `Auckland-1`, `Orlando-1`, `Rotterdam-1`
   - 输入的 `VLN_episodes.json` 必须已经使用公开数据规范中的 canonical trajectory types；数据处理脚本不会原地迁移或删除原始城市数据。
5. Run episodes processing to produce grouped outputs:
   - `episodes/train/` — 训练集（all_episodes.json + 各类型）
   - `episodes/eval/val_seen/` — seen eval 城市（基础城市名在 train 中有 TIF）
   - `episodes/eval/val_unseen/` — unseen eval 城市（基础城市名完全不在 train 中）
   - **注意**：`episodes/eval/` 下没有顶层扁平文件，只有 `val_seen/` 和 `val_unseen/` 子目录
   - 分类逻辑由 `config.py` 中 `classify_eval_cities()` 自动判断，无需手动维护
6. Report summary and **proceed to Step 2 automatically** unless the user explicitly asks to stop after Step 1.

## Step 2: Trajectory Generation

Use the SatNav repository application, but use the trajectory-generation config
maintained in SwiftVLN.

1. Build a temporary config from:
- `/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/src/swiftvln/configs/satnav_trajectory_generation.yaml`
- Set `DATASET.DATA_PATH` to:
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/<version>/episodes/train/all_episodes.json`
- Set `DATASET.SCENES_DIR` to:
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/scenes`
- Keep the other production parameters from this YAML:
  `MAX_EPISODE_STEPS=500`, `FORWARD_STEP_SIZE=10`, `TURN_ANGLE=15`,
  `RGB_SENSOR=448x448/HFOV90`, and `LandmarkSet` success distance `3.0`.

2. Run in `satnav` conda env (在 tmux 中启动):
```bash
cd /mnt/data1/home/jiangjiajun/workspace/SatNav
python -m applications.trajectory_generation.generate_parallel \
    --config <temp_config> \
    --output_dir /mnt/data3/jiangjiajun/dataset/satnav_datasets/<version>/trajectory_data
```
- 默认 `--num_workers` 自动计算：`min(num_scenes, cpu_count//4, 72)`，无需手动指定
- 默认开启 Scene Affinity（按城市分组）以降低内存和 I/O 压力
- 如需强制指定并发数：加 `--num_workers N`

3. **Resume 机制说明（重要）**：
   - 每个 episode 完成后在目录内写 `.done` + `.annotation.json`
   - 重跑时：有 `.annotation.json` → 直接读缓存；有 `.done` 无 `.annotation.json` → 重模拟取 actions；无 `.done` → 完整重跑
   - 中途被 kill 的 half-killed episodes（有图片但 actions 不一致）会被自动检测并重新生成

4. Verify outputs exist:
- `trajectory_data/images/`
- `trajectory_data/annotations.json`
- `trajectory_data/summary.json`

5. Verify quality stats:
- `annotations.json` 的条数 = 成功生成的 episodes 数（< 总 episode 数，失败/超步的会被排除）
- `summary.json` 行数 == `annotations.json` 条数
- 查看生成器日志中的 failed / discarded (max_steps) 统计
- 失败的 episode 不写 `.annotation.json`，不进入训练数据，属预期行为

6. Report runtime and throughput estimate, then **proceed to Step 3 automatically**.

## Step 3: Sync and Verify

Sync the **entire version directory** (`ver_<version>/`) to both remote servers — this includes `episodes/`, `data/`, and `trajectory_data/`.

Use the existing script:

```bash
bash src/swiftvln/scripts/data_sync/sync_and_verify.sh <version>
```

The script handles:
1. `rsync -avP --update` of the full `ver_<version>/` to `10.246.152.73`.
2. `rsync -avP --update` of the full `ver_<version>/` to `10.246.132.17`.
3. Verification across local + both remote servers:
   - Total file counts (98 vs 73 vs 17)
   - Key file sizes:
     - `episodes/train/all_episodes.json`
     - `trajectory_data/annotations.json`
   - `trajectory_data/images/` subdirectory counts
4. Report verification result and any mismatch details.

## Step 4: Sync Training Config Paths

After Step 3 is successful, synchronize latest SatNav dataset paths in project configs:

1. Update `src/swiftvln/configs/satnav_task.yaml`:
- `DATASET.DATA_PATH` -> `/mnt/data3/jiangjiajun/dataset/satnav_datasets/<version>/episodes/eval/{split}/all_episodes.json`
2. Update `src/swiftvln/scripts/train/train_queue.sh`:
- `default_satnav_path` -> `/mnt/data3/jiangjiajun/dataset/satnav_datasets/<version>/trajectory_data`
3. Print changed lines and **proceed automatically** (no confirmation needed).

## Default Script Paths

- `src/swiftvln/scripts/data_process/inspect_data.py`
- `src/swiftvln/scripts/data_process/run_all.py`
- `src/swiftvln/scripts/data_process/process_episodes.py`
- `src/swiftvln/scripts/data_process/config.py`
- `/mnt/data1/home/jiangjiajun/workspace/SatNav/applications/trajectory_generation/generate_parallel.py`
- `src/swiftvln/configs/satnav_trajectory_generation.yaml`
- `src/swiftvln/configs/satnav_task.yaml`
- `src/swiftvln/scripts/data_sync/sync_and_verify.sh`
- `src/swiftvln/scripts/data_sync/sync_data.sh`

## Operating Rules

- **Run all steps continuously without pausing for confirmation.** Only stop and ask the user if:
  - A step fails with an unrecoverable error.
  - A decision is genuinely ambiguous (e.g., new city or episode type discovered that has no precedent).
- After each step, report a brief summary and immediately proceed to the next step.
- Prefer existing project scripts over ad-hoc one-off logic.
- Preserve and restore temporary config edits when possible.
- Fail fast on missing files or SSH permission issues; show exact blocking command and path.
