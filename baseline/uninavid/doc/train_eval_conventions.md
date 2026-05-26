# Uni-NaVid 训练与评测约定

本文档只记录开源使用时需要知道的训练与评测接口约定。环境安装见 `env_setup.md`，已知问题见 `troubleshooting.md`。

## 入口脚本

| 用途 | 路径 |
|---|---|
| 训练 | `baseline/uninavid/scripts/train_satnav.sh` |
| 评测 | `baseline/uninavid/scripts/eval_satnav.sh` |

## 训练

训练脚本支持两种初始化模式：

| 模式 | 默认模型 | 说明 |
|---|---|---|
| `continue` | `baseline/uninavid/model/Uni-Navid` | 从 Uni-NaVid checkpoint 继续训练，默认模式 |
| `scratch` | `baseline/uninavid/model/vicuna-7b-v1.5` | 从 Vicuna-7B 底座起训 |

常用调用：

```bash
# 默认 continue
bash baseline/uninavid/scripts/train_satnav.sh

# 显式指定模式
bash baseline/uninavid/scripts/train_satnav.sh continue
bash baseline/uninavid/scripts/train_satnav.sh scratch

# 自定义实验名
bash baseline/uninavid/scripts/train_satnav.sh continue my_exp_name
```

常用环境变量：

| 变量 | 说明 |
|---|---|
| `DATA_PATH` | SatNav `annotations.json` 路径 |
| `VIDEO_FOLDER` | SatNav `trajectory_data` 根目录 |
| `NUM_GPUS` | 训练 GPU 数 |
| `TRAIN_BSZ` | 单卡 batch size |
| `GRAD_ACCUM` | 梯度累积步数 |
| `NUM_EPOCHS` | 训练 epoch 数 |
| `LEARNING_RATE` | 学习率 |
| `MAX_STEPS` | 可选，限制训练步数 |
| `SAVE_STRATEGY` / `SAVE_STEPS` | checkpoint 保存策略 |
| `DS_CONFIG` | DeepSpeed 配置路径，默认使用 `configs/zero1.json` |
| `REPORT_TO` | HuggingFace Trainer 上报后端，例如 `none` / `wandb` |

输出目录：

```text
output/uninavid-baseline/<EXP_NAME>/
```

默认实验名格式：

```text
uninavid-baseline-{mode}-{epochs}ep-data{version}-bs{effective_bs}-lr{lr}-{timestamp}
```

## 训练数据约定

当前 SatNav baseline 直接读取 SatNav trajectory 数据，不需要提前转换成 mp4 或 Uni-NaVid 原始 JSON 格式。

- 输入标注：`DATA_PATH`
- 图像根目录：`VIDEO_FOLDER`
- 数据集实现：`baseline/uninavid/src/dataset/satnav_dataset.py`
- 每个训练样本预测 4 个动作，动作词为 `forward`、`left`、`right`、`stop`

训练目标使用结构化四步动作文本，例如：

```text
1. forward 2. left 3. right 4. stop
```

评测端按动作词解析，因此结构化文本和普通空格分隔动作都可兼容。

## 评测

评测脚本支持两种输入：

```bash
# 按实验名评测，自动查找 output/uninavid-baseline/<EXP_NAME>/checkpoint-*
bash baseline/uninavid/scripts/eval_satnav.sh <exp_name> [split] [gpus] [max_episodes]

# 按实验名从自定义模型根目录评测，例如 model zoo
bash baseline/uninavid/scripts/eval_satnav.sh \
  --model_dir output/model_zoo/baseline \
  --model_name <exp_name> \
  --split val_seen \
  --gpus 8

# 按 checkpoint 路径评测
bash baseline/uninavid/scripts/eval_satnav.sh /path/to/checkpoint [split] [gpus] [max_episodes]
```

split 约定：

- 不传 `split`：顺序运行 `val_seen` 和 `val_unseen`
- 传 `val_seen` / `val_unseen` / `test`：只运行指定 split

常用环境变量：

| 变量 | 说明 |
|---|---|
| `SATNAV_VERSION` | 指定 SatNav 数据版本；不设置时会尝试从实验名解析 |
| `MODEL_BASE` | adapter-only checkpoint 的底座模型路径，可选 |
| `LOCAL_CACHE_DIR` | 本地 checkpoint cache 目录，可选 |

输出目录：

```text
results/uninavid-baseline/<EXP_NAME>/<split>/
results/uninavid-baseline/by-path/<checkpoint_name>/<split>/
```

评测结果包含：

- `eval.log`：评测日志
- `result.jsonl`：逐 episode 结果
- summary 文件：整体指标汇总

## 注意事项

- 默认 DeepSpeed 配置优先使用 `configs/zero1.json`；如果改用 ZeRO-2，建议先看 `troubleshooting.md`。
- `--max_episodes` 表示先截断总 episode，再进行分布式切分，不是每张 GPU 各自的 episode 上限。
- 多卡或断点续跑时，结果汇总应按 `episode_id` 去重。
