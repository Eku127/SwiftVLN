# SwiftVLN 迁移与 Smoke Test 报告（2026-03-04）

## 1. 目标与约束
- 迁移来源：历史 ms-swift 3.x 仓库中的 `examples/vln`；当前主线运行依赖使用规范路径 `/mnt/data1/home/jiangjiajun/workspace/ms-swift` 下的 4.x 源码。
- 迁移目标：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/examples/vln`
- 约束（2026-03-04 当时）：不修改旧 3.x `ms-swift` 目录内容；运行继续使用已有 conda 环境（`swift-vln-base` / `swift-vln-train` / `swift-vln-eval`）。

## 2. 迁移动作
1. 新建独立仓库目录：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN`
2. 完整复制 `examples/vln` 到新仓库。
3. 新建迁移后运行所需目录：`logs/`、`output/`、`results/`、`reports/`。
4. 路径重构：将脚本中的仓库根变量统一到 `SWIFTVLN_ROOT`，并将硬编码仓库路径替换为 `SwiftVLN`。
5. 语法检查：`examples/vln` 下 shell 脚本 `bash -n` 通过。

## 3. 依赖检查结论

### 3.1 代码内路径依赖扫描
- 命令：`rg -n "workspace/ms-swift|MSSWIFT_ROOT|MS_SWIFT_ROOT" examples/vln -S`
- 结果：无匹配（迁移代码中不存在旧仓库路径/旧根变量名）。

### 3.2 运行时依赖检查
- `swift-vln-train`：`import swift` 成功。
- `swift-vln-eval`：`import swift, habitat, habitat_baselines, satnav` 成功。
- 关键事实（报告生成时）：conda 环境中的 `swift` 仍来自旧 3.x editable 安装；
  2026-08-01 已冻结为普通 wheel，当前主线使用规范路径 `ms-swift` 下的 4.x 源码。

### 3.3 结论
- **本次迁移后，`examples/vln` 功能可以在 `SwiftVLN` 路径下运行。**
- **但运行时仍依赖外部 `ms-swift` 的 `swift` 包安装（editable/source 引用）。**
- 这不影响当前 smoke 训练/评测通过，但代表“代码目录迁移已完成，底层 Python 包依赖尚未完全去耦”。

## 4. Smoke Train 结果

环境：`swift-vln-train`，`CUDA_VISIBLE_DEVICES=0`，最小步数训练（`max_steps=1`）验证可训练与可落盘。

| Case | 状态 | Checkpoint |
|---|---|---|
| streamvln_habitat | PASS | `output/smoke_train/streamvln_habitat/v0-20260304-115310/checkpoint-1` |
| streamvln_satnav | PASS | `output/smoke_train/streamvln_satnav/v0-20260304-115412/checkpoint-1` |
| overlapvln_habitat_perframe | PASS | `output/smoke_train/overlapvln_habitat_perframe/v0-20260304-115458/checkpoint-1` |
| overlapvln_satnav_perframe_pixel | PASS | `output/smoke_train/overlapvln_satnav_perframe_pixel/v0-20260304-115546/checkpoint-1` |
| overlapvln_satnav_gtc | PASS | `output/smoke_train/overlapvln_satnav_gtc/v0-20260304-115635/checkpoint-1` |
| overlapvln_satnav_sgtc | PASS | `output/smoke_train/overlapvln_satnav_sgtc/v0-20260304-115720/checkpoint-1` |

训练日志：`logs/smoke/train_*.log`

## 5. Smoke Eval 结果

环境：`swift-vln-eval`，`CUDA_VISIBLE_DEVICES=0`。

为了加速 smoke：
- `examples/vln/config/vln_r2r_smoke.yaml`：`max_episode_steps=5`
- `examples/vln/config/satnav_task_smoke.yaml`：`MAX_EPISODE_STEPS=5`，`SPLIT=val_unseen`

### 5.1 评测矩阵

| Case | Env | 状态 | 输出目录 |
|---|---|---|---|
| streamvln_habitat | habitat | PASS | `results/smoke_eval/streamvln_habitat` |
| streamvln_satnav | satnav | PASS | `results/smoke_eval/streamvln_satnav` |
| overlapvln_habitat_perframe | habitat | PASS | `results/smoke_eval/overlapvln_habitat_perframe` |
| overlapvln_satnav_perframe_pixel | satnav | PASS | `results/smoke_eval/overlapvln_satnav_perframe_pixel` |
| overlapvln_satnav_gtc | satnav | PASS | `results/smoke_eval/overlapvln_satnav_gtc` |
| overlapvln_satnav_sgtc | satnav | PASS | `results/smoke_eval/overlapvln_satnav_sgtc` |

每个输出目录均包含：
- `all_results.jsonl`
- `evaluation_summary.json`
- `timing_summary.json`

批量 eval 退出码记录：`logs/smoke/eval_smoke_status.tsv`（5 组串行 + 1 组 habitat 单跑，均成功）。

评测日志：`logs/smoke/eval_*.log`

## 6. 对原仓库不修改验证

- `ms-swift HEAD`：`fbd22e9b5dcc4b3ddef4645cc98ec2f4a2adc35e`
- 当时对旧 3.x 仓库执行 `git status --porcelain`：空（无改动）

结论：满足“迁移前后不修改原 `ms-swift` 目录内容”的要求。

后续状态（2026-08-01）：上述旧 3.x 仓库已按用户确认永久删除；当前规范路径
`/mnt/data1/home/jiangjiajun/workspace/ms-swift` 指向固定在 `ad7d5c515` 的 4.x 源码仓库。

## 7. 当前可用性结论

- 迁移后的 `SwiftVLN/examples/vln` 已可独立作为代码目录运行。
- 训练与评测 smoke（6 train + 6 eval）全部通过，路径与功能可用性得到验证。
- 当前残余耦合点为 `swift` 包来源仍是 `ms-swift`，后续若要“完全独立 repo”，需补齐该依赖去耦方案。

## 8. 后续建议（去耦阶段）
1. 明确 `swift` 依赖策略：
   - 方案A：在 `SwiftVLN` 内 vendor 必要子模块；
   - 方案B：固定版本 wheel/pip 依赖并移除本地 editable 绑定。
2. 增加 `requirements`/`environment` 锁定文件（你当前要求可暂缓）。
3. 在 CI 中固化最小 train/eval smoke，防止路径与依赖回归。
