# Uni-NaVid 训练与评测约定

## 入口脚本

| 用途 | 路径 |
|---|---|
| 训练 | `baseline/uninavid/scripts/train_satnav.sh` |
| 评测 | `baseline/uninavid/scripts/eval_satnav.sh` |

## 训练配置约定

- 训练目标为结构化四步动作文本（如 `1. forward 2. left 3. right 4. stop`），eval 仍可兼容，评测端按动作词正则提取
- 默认 `GROUP_BY_MODALITY_LENGTH=False`：避免 SatNav 长度分组把大量易样本连续堆在一起导致 loss 快速塌到日志 `0.0`
- `train_satnav.py` 额外记录 `loss_raw`，用于区分真实异常和四舍五入显示
- 默认 deepspeed 配置：`baseline/uninavid/configs/zero1.json`（ZeRO-2 存在 NaN 问题，详见 `deepspeed_zero2_nan_analysis.md`）
- `train_satnav.sh` 支持两种初始化模式：
  - `continue`：从 `baseline/uninavid/model/Uni-Navid` 继续训练
  - `scratch`：从 `baseline/uninavid/model/vicuna-7b-v1.5` 起训
- 调用方式：
  - `bash baseline/uninavid/scripts/train_satnav.sh continue`
  - `bash baseline/uninavid/scripts/train_satnav.sh scratch`
  - 仍兼容旧调用：只传 `EXP_NAME` 时默认按 `continue`
- 训练脚本对齐 `streamvln` 的实验管理方式：
  - 默认生成结构化 `EXP_NAME`
  - 支持 `USE_SWANLAB=true` 时通过 `--report_to swanlab` 上报
  - `REPORT_TO` 显式传值时优先级高于 `USE_SWANLAB`

## 正式训练默认参数（Updated: 2026-03-12）

- **服务器**：98（8× H100 80GB）
- **`TRAIN_BSZ=24`**：在 ver_260306 全量数据上实测吞吐最高且不 OOM
- 已验证：`bs32`、`bs40` 可跑通但吞吐低于 `bs24`
- 保守配置可手动覆盖为 `TRAIN_BSZ=16`
- 默认实验名自动组织为：
  - `uninavid-baseline-{mode}-{epochs}ep-data{ver}-bs{effective_bs}-lr{lr}-{timestamp}`
- 仍支持手工传入自定义 `EXP_NAME`
- 推荐正式训练使用默认结构化命名，便于 `eval_satnav.sh` 直接按名字解析 `data260306 -> ver_260306`
- 推荐正式训练上报方式：
  - 默认关闭：`USE_SWANLAB=false`
  - 需要实验面板时手动开启：`USE_SWANLAB=true`
  - 调试/压测也可显式传：`REPORT_TO=none`

## 评测配置约定（Updated: 2026-04-02）

- `baseline/uninavid/src/eval_satnav.py` 默认使用确定性解码（`do_sample=False`, `temperature=0.0`），用于稳定复现实验指标
- prompt 语义是“预测 next four actions”，评测端会将解析出的动作词截断为最多 4 个（`forward/left/right/stop`）
- 多卡 + resume 汇总使用 `result.jsonl` 去重（按 `episode_id` 最后写入覆盖），避免历史结果被每个 rank 重复计入 summary
- `--max_episodes` 语义为“先截断总 episode，再做分布式切分”，不是“每卡独立上限”
- `baseline/uninavid/scripts/eval_satnav.sh` 的本地 checkpoint cache 不再只按 `checkpoint-6000` 命名
- cache key 现在包含父实验目录和源路径 hash，避免不同实验同名 checkpoint 发生缓存冲突

## Smoke Test 约定（Updated: 2026-03-11）

使用 skill：`.codex/skills/uninavid-smoke-test/SKILL.md`

**固定要求**：73 服务器、8 卡、SatNav smoke annotations

**推荐参数**：

| 参数 | 值 |
|---|---|
| `DATA_PATH` | `baseline/smoke_test_data/annotations.json` |
| `NUM_GPUS` | `8` |
| `TRAIN_BSZ` | `1` |
| `EVAL_BSZ` | `1` |
| `GRAD_ACCUM` | `1` |
| `MAX_STEPS` | `8` |
| `SAVE_STRATEGY` | `steps` |
| `SAVE_STEPS` | `8` |
| `SAVE_TOTAL_LIMIT` | `1` |
| `REPORT_TO` | `none` |
| `DATALOADER_WORKERS` | `0` |
| `MODEL_MAX_LENGTH` | `1536` |
| `SEED` | `1234` |
| `MASTER_PORT` | 重试时更换新端口 |

**产物目录**：

- `output/uninavid-baseline/smoketest/<run_name>/`
- `results/uninavid-baseline/smoketest/<run_name>/<split>/`

**流程必须覆盖**：

1. 训练保存 checkpoint
2. checkpoint 送给 eval
3. feature 修改前后 loss 对比（若任务涉及训练行为变更）
4. smoke 结束后删除本次 smoke 产物

**注意事项**：

- `train_satnav.sh` 支持 `MASTER_PORT` 环境变量；分布式 smoke 重试前先检查 73 上是否存在残留 Uni-NaVid 训练进程
- eval smoke 可能打印已捕获的 SatNav `Camera view bounds exceed image bounds` traceback；若最终 summary 保存且打印 `Evaluation completed!`，则视为流程通过
