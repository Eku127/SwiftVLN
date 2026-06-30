# Baseline Eval Refactor Guide

本文档记录 StreamVLN baseline eval 重构后的目标状态，并给后续
NaVILA、UniNaVid、OpenFly 等 baseline 的 eval 脚本改造提供可执行模板。

核心目标不是“统一所有模型内部实现”，而是统一外层 eval 入口：

- 模型选择只通过 `--model_dir` + `--model_name`
- eval 数据和 split 只通过 `baseline/<baseline>/configs/satnav_task.yaml`
- eval 脚本不再从模型名中的 `data...` 推断数据版本
- eval 脚本不再默认支持位置参数、by-path、`--checkpoint_path`、`--split`、`--satnav_version`
- README 和 skill 文档只描述推荐主路径
- train+eval pipeline 必须调用新的 by-name eval 入口

## Current Baseline Target State

StreamVLN 已经作为本次 refactor 的参考实现；NaVILA、UniNaVid、OpenFly
也已按同一外层入口和数据默认值完成第一阶段收敛。

### Data Defaults

四个 baseline 的训练和评测默认都指向同一个发布数据集标识：

```text
SATNAV_DATASET=SatNav-v0.1
```

训练默认路径：

```text
/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/trajectory_data
```

评测默认路径：

```text
/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/episodes/eval
```

当前 `SatNav-v0.1/trajectory_data` 和 `SatNav-v0.1/episodes/eval` 已存在；
如果临时需要使用其他 trajectory export，只能显式设置：

```bash
SATNAV_TRAIN_DATA_DIR=/path/to/trajectory_data
```

不要在脚本内部自动 fallback 到 `ver_260418` 之类的历史目录；fallback 会让训练和评测口径重新变得不可追踪。

### Eval Data Config

StreamVLN eval 数据配置在：

```text
baseline/streamvln/configs/satnav_task.yaml
```

推荐格式：

```yaml
DATASET:
  TYPE: SatNav
  SPLIT: all
  DATA_PATH: /mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/episodes/eval
  SCENES_DIR: /mnt/data3/jiangjiajun/dataset/satnav_datasets/scenes
```

字段语义：

- `SPLIT: all`：脚本顺序评测 `val_seen` 和 `val_unseen`
- `SPLIT: val_seen`：只评测 `val_seen`
- `SPLIT: val_unseen`：只评测 `val_unseen`
- `DATA_PATH`：eval split 父目录，脚本解析为 `<DATA_PATH>/<split>/all_episodes.json`
- `SCENES_DIR`：SatNav scenes 目录

### Model Directories

StreamVLN 当前可用模型名：

```text
streamvln-baseline-continue-1ep-f32h8s4-lr2e-5
streamvln-baseline-scratch-1ep-f32h8s4-lr2e-5
streamvln-satnav-continue-1ep-f32h8s4-lr2e-5
streamvln-satnav-scratch-1ep-f32h8s4-lr2e-5
```

两组目录的区别：

- `streamvln-baseline-*`：本地 model zoo 归档版，保留训练日志、`trainer_state.json`、历史 `checkpoint-*`
- `streamvln-satnav-*`：Hugging Face upload-ready 精简版，只保留 eval/inference 所需文件

两组都必须能通过同一个 eval 命令运行：

```bash
bash baseline/streamvln/scripts/eval_satnav.sh \
  --model_dir /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/model_zoo/baseline \
  --model_name streamvln-satnav-continue-1ep-f32h8s4-lr2e-5 \
  --gpus 8
```

StreamVLN eval 当前直接加载 `<model_dir>/<model_name>` 这个 Hugging Face 模型目录。
目录根必须包含：

```text
config.json
model.safetensors.index.json
model-*.safetensors 或 pytorch_model*.bin
tokenizer_config.json
```

不再查找或加载：

```text
<model_dir>/<model_name>/checkpoint-*
```

### HF Upload-Ready Model Production

所有 baseline 的 model zoo 统一放在：

```text
/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/model_zoo/baseline
```

后续 refactor 其他 baseline 时，模型命名和公开版 checkpoint 生产也应在这个目录内完成。

#### Naming Rules

每个 baseline 至少区分两类目录：

```text
<baseline>-baseline-...   # 本地归档版
<baseline>-satnav-...     # HF upload-ready 公开版
```

以 StreamVLN 为例：

```text
streamvln-baseline-continue-1ep-f32h8s4-lr2e-5
streamvln-baseline-scratch-1ep-f32h8s4-lr2e-5
streamvln-satnav-continue-1ep-f32h8s4-lr2e-5
streamvln-satnav-scratch-1ep-f32h8s4-lr2e-5
```

命名原则：

- 保留会影响 eval 行为或人类复现判断的字段：
  - baseline 名称
  - training mode：`scratch` / `continue`
  - training epochs：如 `1ep`
  - eval 窗口或历史参数：如 `f32h8s4`
  - 关键学习率：如 `lr2e-5`
- 删除不应影响 eval 入口的字段：
  - 训练数据版本：如 `data260418`、`data260418p80`
  - batch size：如 `bs64`
  - timestamp：如 `20260421-051524`
  - 机器、重试轮次、临时链路标记
- 如果某个 baseline 的 eval 必须依赖 action format 或 history length，应保留：
  - OpenFly 示例：`actcompact`、`hist16`
- 发布版目录名使用 `satnav`，表示“面向 SatNav 公开复现的模型包”，而不是训练数据版本。

不要让 eval 脚本从发布版模型名中恢复数据版本。数据选择只能来自
`configs/satnav_task.yaml`。

#### Directory Roles

本地归档版保留训练上下文，便于排查和追溯：

```text
<baseline>-baseline-.../
  config.json
  model-*.safetensors
  model.safetensors.index.json
  tokenizer_config.json
  trainer_state.json
  training_args.bin
  train.log
  checkpoint-*/
  ...
```

HF upload-ready 公开版只保留 eval/inference 必需文件：

```text
<baseline>-satnav-.../
  README.md
  .gitattributes
  config.json
  generation_config.json
  model-*.safetensors
  model.safetensors.index.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  added_tokens.json
  merges.txt
  vocab.json
```

公开版应删除：

```text
checkpoint-*/
global_step*/
rng_state_*.pth
optimizer.pt
scheduler.pt
trainer_state.json
training_args.bin
zero_to_fp32.py
train.log
eval.log
*.tmp
```

如果某个 baseline 的公开加载还需要 processor、image processor、adapter config、vocab
或其他文件，应把这些文件列入该 baseline 的 release manifest，并在 README 说明。

#### Config Rewriting

公开版不能保留只在本机有效的绝对路径。

StreamVLN 的实际处理：

- 本地归档版 `config.json` 中的 vision tower 是本地路径：
  `/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/baseline/streamvln/model/siglip-so400m-patch14-384`
- 公开版 `config.json` 改成：
  `google/siglip-so400m-patch14-384`

这样外部加载时可以根据 Hugging Face id 构建 SigLIP base architecture。StreamVLN 的 fine-tuned
vision tower 参数已保存在主 `model-*.safetensors` shards 中，README 需要明确说明这一点。

通用规则：

- `config.json`、processor config、tokenizer config 中不要保留 `/mnt/...`、`/home/...` 这类本地路径
- 可公开解析的上游模型用 Hugging Face id
- 必须随包发布的本地资产放入发布目录，并用相对路径引用
- 不要原地修改本地归档版 config；生成公开版时复制后再改

#### README For Public Model Directory

每个 HF upload-ready 目录必须有自己的 `README.md`。最少包含：

- 模型类型和训练模式
- 它对应的本地归档版模型名
- 目录只包含 eval/inference artifacts
- 哪些训练状态被省略
- 关键 config 改写说明
- eval 需要的代码依赖
- 推荐 eval 命令
- 窗口参数解释

StreamVLN README 中应明确：

```text
This directory is a Hugging Face upload-ready copy of:
streamvln-baseline-continue-1ep-f32h8s4-lr2e-5
```

并说明：

```text
DeepSpeed optimizer state, RNG state, training logs, and duplicated checkpoint-* files are intentionally omitted.
```

#### Production Checklist

生成公开版 checkpoint 时，按下面顺序执行。

1. 确认源目录：

```bash
MODEL_ZOO=/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/model_zoo/baseline
SRC="${MODEL_ZOO}/streamvln-baseline-continue-1ep-f32h8s4-lr2e-5"
DST="${MODEL_ZOO}/streamvln-satnav-continue-1ep-f32h8s4-lr2e-5"
```

2. 确认源目录根已经是可加载 HF 模型目录：

```bash
ls "${SRC}/config.json"
ls "${SRC}/model.safetensors.index.json"
ls "${SRC}"/model-*.safetensors
ls "${SRC}/tokenizer_config.json"
```

3. 创建公开版目录，只复制 release manifest 中的文件。

4. 改写 `config.json` 中的本地路径，例如 `mm_vision_tower` / `vision_tower`。

5. 写公开版 `README.md` 和 `.gitattributes`。

6. 验证权重没有意外变化：

```bash
cmp -s "${SRC}/model.safetensors.index.json" "${DST}/model.safetensors.index.json"
cmp -s "${SRC}/model-00001-of-00004.safetensors" "${DST}/model-00001-of-00004.safetensors"
```

对大模型可只抽查 index 和一个 shard；正式发布前建议对所有 shard 计算 sha256。

7. 验证公开版目录不含训练状态：

```bash
find "${DST}" -maxdepth 2 \( \
  -name 'checkpoint-*' -o \
  -name 'global_step*' -o \
  -name 'rng_state_*.pth' -o \
  -name 'optimizer.pt' -o \
  -name 'scheduler.pt' -o \
  -name 'trainer_state.json' -o \
  -name 'training_args.bin' -o \
  -name 'train.log' \
\) -print
```

命令应无输出。

8. 用统一 eval 入口 dry-run：

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate streamvln-baseline

bash baseline/streamvln/scripts/eval_satnav.sh \
  --model_dir "${MODEL_ZOO}" \
  --model_name streamvln-satnav-continue-1ep-f32h8s4-lr2e-5 \
  --gpus 8 \
  --dry_run
```

9. dry-run 通过后，至少做一次小样本真实 eval：

```bash
bash baseline/streamvln/scripts/eval_satnav.sh \
  --model_dir "${MODEL_ZOO}" \
  --model_name streamvln-satnav-continue-1ep-f32h8s4-lr2e-5 \
  --gpus 1 \
  --max_episodes 2
```

小样本真实 eval 用来确认 Python 入口、模型类、tokenizer、vision tower、SatNav 环境都能实际加载。

#### Applying This To Other Baselines

迁移其他 baseline 时，先为每个模型确定两个名字：

```text
<baseline>-baseline-<mode>-<training-summary>
<baseline>-satnav-<mode>-<training-summary>
```

如果现有 model zoo 目录还带有历史数据版本和 timestamp，先不要急着改 eval 脚本。
先完成以下三件事：

1. 选定最终公开版模型名
2. 生成 `<baseline>-satnav-*` 公开目录
3. 证明该公开目录可以被新的 `--model_dir + --model_name` eval 入口 dry-run

只有公开目录和 eval 入口都跑通后，README 和报告才应切换到新名字。

### Train Eval Pipeline

StreamVLN 的一键脚本：

```text
baseline/streamvln/scripts/train_eval_satnav.sh
```

必须按新入口调用 eval：

```bash
bash baseline/streamvln/scripts/eval_satnav.sh \
  --model_dir output/streamvln-baseline \
  --model_name <EXP_NAME_OR_SUBPATH> \
  --gpus <N>
```

不允许再调用旧形式：

```bash
bash baseline/streamvln/scripts/eval_satnav.sh <EXP_NAME> <split> <gpus>
```

`train_eval_satnav.sh` 也不再接受位置 split 参数。需要单 split 时，修改
`baseline/streamvln/configs/satnav_task.yaml` 的 `DATASET.SPLIT`。

## Refactor Procedure For Other Baselines

下面步骤用于迁移 NaVILA、UniNaVid、OpenFly 等 baseline。

### Step 1: Identify Model Loading Contract

先确认该 baseline 的 Python eval 入口到底接受什么模型格式。

需要回答：

- Python eval 是否能直接加载 Hugging Face root 目录？
- 是否必须传 Trainer `checkpoint-*` 目录？
- tokenizer / processor 是否必须和模型目录同源？
- 模型 config 中是否含有需要本地化的路径，如 vision tower、processor、base model？
- 多卡 eval 是否需要额外 `--world_size`、`--rank`、`--gpu` 参数？

不要在 shell 里同时支持所有历史格式。先选定一种 model zoo 规范，再让 shell 只服务这条主路径。

### Step 2: Normalize CLI

目标 CLI：

```bash
bash baseline/<baseline>/scripts/eval_satnav.sh \
  --model_dir /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/model_zoo/baseline \
  --model_name <model_name> \
  --gpus 8
```

保留可选参数：

```text
--max_episodes <N>
--dry_run
```

仅在模型确实需要时保留 baseline-specific 参数，例如：

```text
--vision_tower <path-or-hf-id>
--action_format <compact|original>
```

但这些参数不应重新引入数据版本选择。

删除或停止推荐：

```text
位置参数：eval_satnav.sh <exp_or_checkpoint> [split] [gpus]
--checkpoint_path
--split
--satnav_version
SATNAV_VERSION 影响 eval 数据
results/.../by-path/... 作为默认输出路径
```

如果为了过渡必须保留历史参数，应满足两点：

- README 和 skill 不再展示历史参数
- 新主路径必须不依赖历史参数

### Step 3: Move Eval Data Selection To YAML

每个 baseline 都应有自己的配置：

```text
baseline/<baseline>/configs/satnav_task.yaml
```

脚本读取：

```text
DATASET.SPLIT
DATASET.DATA_PATH
DATASET.SCENES_DIR
```

推荐 shell 逻辑：

```bash
CONFIG_SPLIT="$(read_config_value SPLIT)"
if [ "$CONFIG_SPLIT" = "all" ]; then
  SPLITS_LIST="val_seen val_unseen"
else
  SPLITS_LIST="$CONFIG_SPLIT"
fi
```

每个 split 运行前生成临时 config：

```text
baseline/<baseline>/configs/.satnav_task_eval_<split>_<pid>.yaml
```

临时 config 应落地为具体 episode 文件：

```yaml
DATASET:
  SPLIT: val_seen
  DATA_PATH: /mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/episodes/eval/val_seen/all_episodes.json
```

正常结束、失败退出、`--dry_run` 后都应清理临时 config。至少要在普通路径中显式 `rm -f`，更稳妥是加 `trap`。

### Step 4: Restrict Model Name Parsing

只解析 eval 必需参数。

StreamVLN 示例：

```text
f32h8s4 -> frames=32, history=8, future_steps=4
```

允许：

- 解析窗口长度、history 长度、action history 长度等 eval 实际需要的参数
- 解析 action format 这类模型行为参数
- 解析失败时使用明确默认值并打印 warning

禁止：

- 从 `data260418` / `data260418p80` 推断 eval 数据
- 从时间戳、学习率、batch size 决定 eval 数据或模型路径
- 根据模型名 fallback 到历史结果目录

### Step 5: Resolve Local Assets Explicitly

如果模型 config 中包含外部资源路径，应采用可解释的解析顺序。

StreamVLN vision tower 规则：

1. 用户显式 `--vision_tower` 优先
2. 如果 config 值本身是本地目录，直接用
3. 如果是相对路径，先尝试 repo root 下解析
4. 尝试 baseline 本地模型目录，如 `baseline/streamvln/model/<basename>`
5. 尝试 model zoo 下同名目录
6. 都不存在时保留原始 Hugging Face id，让 Transformers 使用 cache 或下载

如需临时改写模型 config，不要修改原模型目录。应创建隐藏临时目录：

```text
baseline/<baseline>/configs/.<baseline>_model_eval_<split>_<pid>/
```

将模型文件软链进去，只写一份临时 `config.json`，eval 后删除临时目录。

`.gitignore` 应忽略这些临时文件：

```gitignore
baseline/*/configs/.satnav_task_eval_*.yaml
baseline/<baseline>/configs/.<baseline>_model_eval_*/
```

### Step 6: Update Train Eval Pipeline

如果 baseline 有 train+eval 一键脚本，必须同步新 eval 入口。

错误形式：

```bash
bash scripts/eval_satnav.sh "$EXP_NAME" "$SPLIT" "$GPUS"
```

正确形式：

```bash
bash scripts/eval_satnav.sh \
  --model_dir "$MODEL_ROOT" \
  --model_name "$EXP_NAME" \
  --gpus "$GPUS"
```

一键脚本不应该再自己循环 split。split 展开由 eval 脚本根据 YAML 处理。

如果需要单 split：

```text
edit baseline/<baseline>/configs/satnav_task.yaml
DATASET.SPLIT: val_seen
```

### Step 7: Update Docs And Skills Together

改脚本时必须同步：

- `baseline/<baseline>/README.md`
- `.codex/CODEX_CONTEXT.md`
- 对应 `.codex/skills/*/SKILL.md`
- 如有报告引用 model zoo 路径，也同步 report/raw data

README eval 部分只保留三块：

1. 配置评测数据：说明 `satnav_task.yaml`
2. 启动评测：只给 `--model_dir + --model_name`
3. 行为说明：模型目录、解析参数、输出路径、日志路径

不要在主 README 中继续展示 by-path、位置参数、`SATNAV_VERSION` 覆盖 eval 等历史用法。

## Validation Checklist

每个 baseline 改完至少执行：

```bash
bash -n baseline/<baseline>/scripts/eval_satnav.sh
```

如果有 train+eval 脚本：

```bash
bash -n baseline/<baseline>/scripts/train_eval_satnav.sh
```

dry-run：

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate <baseline-conda-env>

bash baseline/<baseline>/scripts/eval_satnav.sh \
  --model_dir /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/model_zoo/baseline \
  --model_name <model_name> \
  --gpus 8 \
  --dry_run
```

检查 dry-run 输出：

- 模型目录正确
- split 展开正确
- `DATA_PATH` 被解析为 `<eval_root>/<split>/all_episodes.json`
- `SCENES_DIR` 正确
- eval 必需参数解析正确
- 输出目录是 `results/<baseline>-baseline/<model_name>/<split>/`
- `--dry_run` 不进入 Python eval
- 没有残留 `.satnav_task_eval_*.yaml`
- 如果生成了临时模型目录，没有残留 `.<baseline>_model_eval_*`

检查工作区：

```bash
find baseline/<baseline>/configs -maxdepth 1 -name '.satnav_task_eval_*.yaml' -print
git diff --check
```

### Current SatNav-v0.1 8-GPU Smoke Evidence

2026-06-29 已按 refactor 标准完成四个 baseline 在 `SatNav-v0.1` 上的
8 卡 train/eval smoke。该证据用于判断入口、默认数据路径、HF upload-ready
模型目录和分布式收尾是否可用；不用于报告模型精度。

Train smoke：

```text
logs/baseline_satnav_v01_smoke/train_smoke_73_20260629_143641.csv
logs/baseline_satnav_v01_smoke/train_streamvln_local98_20260629_150840.status
```

覆盖：

- StreamVLN continue，`SATNAV_DATASET=SatNav-v0.1`，
  `SATNAV_MAX_EPISODES=64 SATNAV_MAX_SAMPLES=64 MAX_STEPS=2`，
  `train_rc=0 verify_rc=0`
- NaVILA continue，`SATNAV_DATASET=SatNav-v0.1`，
  `MAX_STEPS=2 SATNAV_MAX_SAMPLES=32`，`train_rc=0 verify_rc=0`
- UniNaVid continue，`SATNAV_DATASET=SatNav-v0.1`，
  `MAX_STEPS=2`，`train_rc=0 verify_rc=0`
- OpenFly continue，`SATNAV_DATASET=SatNav-v0.1`，
  `MAX_STEPS=2 SATNAV_MAX_SAMPLES=32`，`train_rc=0 verify_rc=0`

Eval smoke：

```text
logs/baseline_satnav_v01_smoke/local98_eval_smoke_20260629_142339.csv
logs/baseline_satnav_v01_smoke/local98_uninavid_retry_20260629_143332.status
```

覆盖 HF upload-ready continue 模型：

```text
streamvln-satnav-continue-1ep-f32h8s4-lr2e-5
navila-satnav-continue-1ep-8f-sample-hk7-fs7-stopx4
uninavid-satnav-continue-1ep-lr1e-5
openfly-satnav-continue-1ep-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-lr2e-5
```

统一 eval 命令形态：

```bash
bash baseline/<baseline>/scripts/eval_satnav.sh \
  --model_dir /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/model_zoo/baseline/HF_model \
  --model_name <baseline-satnav-continue-model-name> \
  --gpus 8 \
  --max_episodes 2
```

结果摘要副本已写出：

```text
logs/baseline_satnav_v01_smoke/local98_streamvln_20260629_142339_summaries.jsonl
logs/baseline_satnav_v01_smoke/local98_navila_20260629_142339_summaries.jsonl
logs/baseline_satnav_v01_smoke/local98_uninavid_retry_20260629_143332_summaries.jsonl
logs/baseline_satnav_v01_smoke/local98_openfly_20260629_142339_summaries.jsonl
```

本轮修复中需要复用到其他 baseline 的经验：

- StreamVLN 训练入口必须优先使用自己的 conda Python，并把该 env 的 `bin`
  放到 `PATH` 前面；否则 DeepSpeed JIT 可能找不到 conda 内的 `ninja`
- StreamVLN 在 import 阶段会触发 tokenizer/config 读取，训练脚本和 Python 入口都要设置
  `STREAMVLN_OFFLINE=true` 时的 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
- shell 中可变长度参数不要用字符串拼接后放在多行命令尾部；使用数组，如
  `SWANLAB_ARGS=(--report_to none)`，再用 `"${SWANLAB_ARGS[@]}"`
- 分布式 eval 聚合时不要等待没有分到 episode 的 rank；需要按 deterministic shard
  计算 expected ranks，且 rank0 清理同步 marker 后要 barrier，再让其他 rank 写 marker

历史 0418 smoke 仍可作为 refactor 初期证据参考：

```text
logs/baseline_refactor_smoke/train_smoke_20260629_110010.log
logs/baseline_refactor_smoke/eval_smoke_20260629_112321.log
```

## Migration Notes Per Baseline

### NaVILA

NaVILA 已按 StreamVLN 方式完成第一阶段 eval 入口收敛：

- 公开入口只接受 `--model_dir + --model_name`
- 不再接受位置参数、`--checkpoint_path`、`--split`、`--satnav_version`
- eval split 和数据只从 `baseline/navila/configs/satnav_task.yaml` 读取
- `DATASET.DATA_PATH` 是 eval split 父目录，当前默认：
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/episodes/eval`
- 训练默认使用 `SATNAV_DATASET=SatNav-v0.1`，并展开到
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/trajectory_data`
- model zoo 旧归档根目录如果已有 `config.json`，直接用根目录加载；否则才在该根下选择最新
  `checkpoint-*`
- NFS checkpoint cache 逻辑保留，但只影响模型加载路径，不影响 eval 数据选择
- `MODEL_BASE` 作为可选参数保留，用于 adapter-only checkpoint 的显式回退

当前已 dry-run 通过的模型名：

```text
navila-continue0404-r2-20260427-141143-sample-hk7-fs7-stopx4
navila-scratch0404-r1-20260407-195154-sample-hk7-fs7-stopx4
```

验证命令：

```bash
bash baseline/navila/scripts/eval_satnav.sh \
  --model_dir /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/model_zoo/baseline \
  --model_name navila-continue0404-r2-20260427-141143-sample-hk7-fs7-stopx4 \
  --gpus 8 \
  --dry_run
```

### UniNaVid

UniNaVid 已按 StreamVLN/NaVILA 的方式完成第一阶段 eval 入口收敛：

- 公开入口只接受 `--model_dir + --model_name`
- 不再接受位置参数、`--checkpoint_path`、`--split`、`--satnav_version`
- eval split 和数据只从 `baseline/uninavid/configs/satnav_task.yaml` 读取
- `DATASET.DATA_PATH` 是 eval split 父目录，当前默认：
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/episodes/eval`
- 训练默认使用 `SATNAV_DATASET=SatNav-v0.1`，并展开到
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/trajectory_data`
- 保留启动前 GPU 可见性检查，8 卡 eval 会在进入 `torchrun` 前确认当前 host 可见 GPU 数足够
- model zoo 归档根目录已有 `config.json` 和 `pytorch_model*.bin`，直接用根目录加载

当前已 dry-run 通过的归档模型名：

```text
uninavid-baseline-continue-1ep-data260418-bs192-lr1e-5-20260418-203618
uninavid-baseline-scratch-1ep-data260418-bs192-lr1e-5-20260418-203618
```

当前已生成并 dry-run 通过的 HF upload-ready 模型名：

```text
uninavid-satnav-continue-1ep-lr1e-5
uninavid-satnav-scratch-1ep-lr1e-5
```

验证命令：

```bash
bash baseline/uninavid/scripts/eval_satnav.sh \
  --model_dir /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/model_zoo/baseline/HF_model \
  --model_name uninavid-satnav-continue-1ep-lr1e-5 \
  --gpus 8 \
  --dry_run
```

### OpenFly

OpenFly 已按 StreamVLN/NaVILA/UniNaVid 的方式完成第一阶段 eval 入口收敛：

- 公开入口只接受 `--model_dir + --model_name`
- 不再接受位置参数、`--checkpoint_path`、`--split`、`--satnav_version`
- eval split 和数据只从 `baseline/openfly/configs/satnav_task.yaml` 读取
- `DATASET.DATA_PATH` 是 eval split 父目录，当前默认：
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/episodes/eval`
- 训练默认使用 `SATNAV_DATASET=SatNav-v0.1`，并展开到
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/trajectory_data`
- 保留 `-actcompact` / `-actoriginal` 解析，因为这是 eval 行为参数
- 保留 `-hist<N>` 解析，因为这是 eval 行为参数
- 不再从 `data<N>` 解析 eval 数据版本
- 旧 model zoo 归档根目录没有根权重时，脚本会选择该根下最新 `checkpoint-*`
- HF upload-ready 目录把 `checkpoint-*` 中的 HF 权重文件提升到目录根，eval 直接加载公开目录根

当前已 dry-run 通过的归档模型名：

```text
openfly-baseline-1ep-data260418-bkcontinue-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-bs96-lr2e-5-20260420-095357
openfly-baseline-1ep-data260418-bkscratch-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-bs96-lr2e-5-20260420-233110
```

当前已生成并 dry-run 通过的 HF upload-ready 模型名：

```text
openfly-satnav-continue-1ep-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-lr2e-5
openfly-satnav-scratch-1ep-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-lr2e-5
```

验证命令：

```bash
bash baseline/openfly/scripts/eval_satnav.sh \
  --model_dir /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/model_zoo/baseline/HF_model \
  --model_name openfly-satnav-continue-1ep-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-lr2e-5 \
  --gpus 8 \
  --dry_run
```

## Done Definition

一个 baseline 的 eval refactor 完成，需要同时满足：

- 新命令 `--model_dir + --model_name` 可 dry-run
- README 只展示新命令
- skill/context 不再让 Codex 调用旧位置参数
- eval 数据只由 YAML 控制
- train+eval pipeline 不再传位置 split
- model zoo 中推荐模型名明确列出
- `bash -n` 通过
- `git diff --check` 无新增 whitespace 问题
