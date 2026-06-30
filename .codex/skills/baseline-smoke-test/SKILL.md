---
name: baseline-smoke-test
description: "Run smoke tests for any baseline (streamvln / navila / uninavid / openfly) on SatNav: 8-GPU train smoke with the current baseline data default, eval smoke by --model_dir/--model_name, loss trend verification, and post-test cleanup."
---

# Baseline Smoke Test

## 触发条件

当用户说以下任意形式时使用：

- "跑一下 baseline `<name>` 的 smoke test"
- "baseline `<name>` 冒烟测试"
- "`<name>` baseline smoke"

其中 `<name>` ∈ `streamvln` | `navila` | `uninavid` | `openfly`

**第一步：从用户输入中识别 baseline 名称，然后跳转到对应分支。**

---

## 通用原则（所有 baseline 一致）

- 全部固定 **8 GPU** 训练。
- 当前 refactor 验证口径：`streamvln` / `navila` / `uninavid` / `openfly`
  均默认使用 `SatNav-v0.1`。
- 不使用 `smoke_test_data/` 专用数据。
- 训练：`MAX_STEPS=8`，`SAVE_STRATEGY=steps`，`SAVE_STEPS=8`，`LOGGING_STEPS=1`。
- 评测：尽量贴近正式默认路径；若 baseline 支持则优先用正式 split，仅缩小 episode 数量。
- baseline eval 入口：
  `baseline/streamvln/scripts/eval_satnav.sh`
  `baseline/navila/scripts/eval_satnav.sh`
  `baseline/uninavid/scripts/eval_satnav.sh`
  `baseline/openfly/scripts/eval_satnav.sh`
- eval 入口统一使用 `--model_dir <root> --model_name <name>`；不要再传位置参数、
  `--checkpoint_path`、`--split` 或 `--satnav_version`。
- eval split 和数据只由 `baseline/<baseline>/configs/satnav_task.yaml` 控制；
  当前 `SPLIT: all` 会顺序运行 `val_seen` 和 `val_unseen`。
- 验证训练 loss 是否有限且呈下降趋势（见 [Loss 验证](#loss-验证) 节）。
- 验证评测 summary 写入成功。
- 完成后**必须清理** smoke 输出和结果目录。

---

## Step 0 — 当前数据口径

所有 baseline 在本轮 refactor 后固定使用 `SatNav-v0.1`：

```bash
SATNAV_DATA_ROOT="/mnt/data3/jiangjiajun/dataset/satnav_datasets"
SATNAV_DATASET="SatNav-v0.1"
SATNAV_TRAIN_DATA_DIR="${SATNAV_DATA_ROOT}/${SATNAV_DATASET}/trajectory_data"
SATNAV_ANNOTATIONS="${SATNAV_TRAIN_DATA_DIR}/annotations.json"
SATNAV_EVAL_ROOT="${SATNAV_DATA_ROOT}/${SATNAV_DATASET}/episodes/eval"
echo "SatNav dataset: ${SATNAV_DATASET}"
echo "Train annotations: ${SATNAV_ANNOTATIONS}"
echo "Eval root: ${SATNAV_EVAL_ROOT}"
```

确认对应 baseline 的 `configs/satnav_task.yaml` 中：

```yaml
DATASET:
  SPLIT: all
  DATA_PATH: /mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/episodes/eval
```

---

## Baseline: StreamVLN

### 环境

- 服务器：`98`（本机）
- Conda 环境：`streamvln-baseline`
- 脚本：`baseline/streamvln/scripts/train_satnav.sh` / `eval_satnav.sh`

### Preflight

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate streamvln-baseline
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN

python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
ls baseline/streamvln/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3/config.json
ls /mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/trajectory_data/annotations.json
```

### Full-Path Smoke（缩步数，保留正式主路径）

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate streamvln-baseline
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN

SMOKE_PIPE_LOG="/tmp/streamvln_smoke_train_eval_$(date +%Y%m%d_%H%M%S).log"

SMOKE_TEST=true \
SATNAV_DATASET=SatNav-v0.1 \
MAX_STEPS=8 \
SAVE_STRATEGY=steps \
SAVE_STEPS=8 \
LOGGING_STEPS=1 \
TRAIN_GPUS=8 \
EVAL_GPUS=8 \
EVAL_MAX_EPISODES=10 \
bash baseline/streamvln/scripts/train_eval_satnav.sh continue \
  2>&1 | tee "${SMOKE_PIPE_LOG}"

# 训练结束后捕获自动生成的 EXP_NAME
SMOKE_EXP=$(ls -t output/streamvln-baseline/smoketest/ | head -1)
echo "Smoke EXP: ${SMOKE_EXP}"
```

> 这里不是“先 train，再手工单独 eval”，而是直接验证正式联动链路：
> `train_satnav.sh` -> `EXP_NAME 提取` -> `eval_satnav.sh smoketest/<EXP_NAME>` -> `evaluation_summary.json`
> 是否能一口气打通。
>
> 与正式全量相比，仅保留两处 smoke 缩减：
> - 训练只跑 `8` steps
> - eval 只跑 `10` 个 episodes（每个 split）
>
> 其余关键路径尽量保持与正式一致：
> - `continue` 模式
> - `train_eval_satnav.sh` 联动入口
> - **默认 val_seen + val_unseen 两个 split**（0319 起 episodes/eval/ 下只有 val_seen/ 和 val_unseen/ 子目录）
> - `8 GPU eval`

### 验证训练产物

```bash
# checkpoint 或合并模型在根目录（safetensors），eval 使用根目录
ls output/streamvln-baseline/smoketest/${SMOKE_EXP}/
# 预期：含 safetensors 或 checkpoint-8/ 子目录（eval 脚本自动处理）
```

### 验证联动切换成功

```bash
grep -nE "Post-train summary|Eval target subpath|Start eval|Pipeline finished successfully" "${SMOKE_PIPE_LOG}"
```

**预期至少包含：**
- `Post-train summary: train_rc=0`
- `Eval target subpath: smoketest/${SMOKE_EXP}`
- `[INFO] Start eval...`
- `[OK] Pipeline finished successfully.`

### Eval Smoke 验证

```bash
# val_seen
ls results/streamvln-baseline/smoketest/${SMOKE_EXP}/val_seen/
tail -100 results/streamvln-baseline/smoketest/${SMOKE_EXP}/val_seen/eval.log
cat results/streamvln-baseline/smoketest/${SMOKE_EXP}/val_seen/evaluation_summary.json

# val_unseen
ls results/streamvln-baseline/smoketest/${SMOKE_EXP}/val_unseen/
tail -100 results/streamvln-baseline/smoketest/${SMOKE_EXP}/val_unseen/eval.log
cat results/streamvln-baseline/smoketest/${SMOKE_EXP}/val_unseen/evaluation_summary.json
```

> 如果联动失败，这里通常会表现为：
> `results/streamvln-baseline/smoketest/${SMOKE_EXP}/val_seen/` 不存在，或者 pipeline log 中没有 `Start eval`。

### Cleanup

```bash
REPO=/mnt/data1/home/jiangjiajun/workspace/SwiftVLN
rm -rf ${REPO}/output/streamvln-baseline/smoketest/${SMOKE_EXP}
rm -rf ${REPO}/results/streamvln-baseline/smoketest/${SMOKE_EXP}
rm -f "${SMOKE_PIPE_LOG}"
```

---

## Baseline: NaVILA

### 环境

- 服务器：`98`（本机）
- Conda 环境：`navila-baseline`
- 脚本：`baseline/navila/scripts/train_satnav.sh` / `eval_satnav.sh`

### Preflight

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate navila-baseline
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN

nvidia-smi -L | wc -l        # 应为 8
ls baseline/navila/model/navila-llama3-8b-8f/config.json
ls /mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/trajectory_data/annotations.json
```

### Train Smoke

```bash
SMOKE_NAME="smoke_navila_$(date +%Y%m%d_%H%M%S)"

source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate navila-baseline
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN

SATNAV_DATASET=SatNav-v0.1 \
MAX_STEPS=8 \
SAVE_STEPS=8 \
SAVE_TOTAL_LIMIT=1 \
TRAIN_BSZ=1 \
GRAD_ACCUM=1 \
DATALOADER_WORKERS=2 \
REPORT_TO=none \
MASTER_PORT=29640 \
bash baseline/navila/scripts/train_satnav.sh continue "smoketest/${SMOKE_NAME}" \
  2>&1 | tee /tmp/navila_smoke_train.log
```

> `CUSTOM_EXP_NAME = smoketest/${SMOKE_NAME}`，脚本会自动追加采样标签；
> 输出至 `output/navila-baseline/smoketest/${SMOKE_NAME}-sample-hk7-fs7-stopx4/`

### 验证训练产物

验证 `output/navila-baseline/smoketest/${SMOKE_NAME}-sample-hk7-fs7-stopx4/` 下包含：
- `config.json`
- `llm/`（NaVILA 特有的合并输出格式）
- 无 `Traceback` 字样

如果只有 `checkpoint-8/` 子目录（DeepSpeed 格式），NaVILA eval 脚本会自动选取它。

### Eval Smoke

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate navila-baseline
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN

SCRIPT_MODEL_DIR="/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/navila-baseline"
bash baseline/navila/scripts/eval_satnav.sh \
  --model_dir "${SCRIPT_MODEL_DIR}" \
  --model_name "smoketest/${SMOKE_NAME}-sample-hk7-fs7-stopx4" \
  --gpus 8 \
  --max_episodes 10 \
  2>&1 | tee "/tmp/navila_smoke_eval.log"
```

`eval_satnav.sh` 只支持命名参数主路径：
`--model_dir <root> --model_name <name> --gpus <n>`。

### Cleanup

```bash
REPO=/mnt/data1/home/jiangjiajun/workspace/SwiftVLN
rm -rf ${REPO}/output/navila-baseline/smoketest/${SMOKE_NAME}-sample-hk7-fs7-stopx4
rm -rf ${REPO}/results/navila-baseline/smoketest/${SMOKE_NAME}-sample-hk7-fs7-stopx4
```

---

## Baseline: Uni-NaVid

### 环境

- 服务器：`73`（`ssh 10.246.152.73`，远程执行）
- Conda 环境：`uninavid-baseline`
- 脚本：`baseline/uninavid/scripts/train_satnav.sh` / `eval_satnav.sh`

### Preflight（在 73 上执行）

```bash
ssh 10.246.152.73 '
  nvidia-smi -L | wc -l        # 应为 8
  ls /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/baseline/uninavid/model/Uni-Navid/config.json
  ls /mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/trajectory_data/annotations.json
'
```

### 清理残留进程（如有）

```bash
ssh 10.246.152.73 "ps -eo pid,cmd | grep -E 'deepspeed.launcher|train_satnav.py' | grep -v grep || true"
# 如发现残留进程，kill 后使用新的 MASTER_PORT
```

### Train Smoke（在 73 上执行）

```bash
SMOKE_NAME="smoke_uninavid_$(date +%Y%m%d_%H%M%S)"

ssh 10.246.152.73 "
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate uninavid-baseline
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN

SATNAV_DATASET=SatNav-v0.1 \
MAX_STEPS=8 \
SAVE_STRATEGY=steps \
SAVE_STEPS=8 \
SAVE_TOTAL_LIMIT=1 \
TRAIN_BSZ=1 \
GRAD_ACCUM=1 \
DATALOADER_WORKERS=0 \
REPORT_TO=none \
MASTER_PORT=29618 \
bash baseline/uninavid/scripts/train_satnav.sh continue 'smoketest/${SMOKE_NAME}'
" 2>&1 | tee /tmp/uninavid_smoke_train.log
```

> 输出至 `output/uninavid-baseline/smoketest/${SMOKE_NAME}/`

### Eval Smoke（在 73 上执行）

```bash
ssh 10.246.152.73 "
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate uninavid-baseline
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN
bash baseline/uninavid/scripts/eval_satnav.sh \
  --model_dir /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/uninavid-baseline \
  --model_name 'smoketest/${SMOKE_NAME}' \
  --gpus 8 \
  --max_episodes 10
" 2>&1 | tee "/tmp/uninavid_smoke_eval.log"
```

`eval_satnav.sh` 只支持命名参数主路径：
`--model_dir <root> --model_name <name> --gpus <n>`。

### Cleanup

```bash
ssh 10.246.152.73 "
  rm -rf /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/uninavid-baseline/smoketest/${SMOKE_NAME}
  rm -rf /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/uninavid-baseline/smoketest/${SMOKE_NAME}
"
```

---

## Baseline: OpenFly

### 环境

- 服务器：`73`（`ssh 10.246.152.73`，远程执行）
- Conda 环境：`openfly-baseline`
- 脚本：`baseline/openfly/scripts/train_satnav.sh` / `eval_satnav.sh`

### Preflight（在 73 上执行）

```bash
ssh 10.246.152.73 '
  nvidia-smi -L | wc -l        # 应为 8
  ls /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/baseline/openfly/model/openfly-agent-7b/config.json
  ls /mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/trajectory_data/annotations.json
'
```

### Train Smoke（在 73 上执行）

```bash
SMOKE_NAME="smoke_openfly_$(date +%Y%m%d_%H%M%S)"

ssh 10.246.152.73 "
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate openfly-baseline
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN

SATNAV_DATASET=SatNav-v0.1 \
OPENFLY_BACKEND=continue \
EXP_NAME='smoketest/${SMOKE_NAME}' \
MAX_STEPS=8 \
SAVE_STEPS=8 \
SAVE_TOTAL_LIMIT=1 \
NUM_GPUS=8 \
TRAIN_BSZ=1 \
GRAD_ACCUM=1 \
DATALOADER_NUM_WORKERS=2 \
REPORT_TO=none \
MASTER_PORT=29619 \
bash baseline/openfly/scripts/train_satnav.sh
" 2>&1 | tee /tmp/openfly_smoke_train.log
```

> 输出至 `output/openfly-baseline/smoketest/${SMOKE_NAME}/`。

### Eval Smoke（在 73 上执行）

```bash
ssh 10.246.152.73 "
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate openfly-baseline
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN
bash baseline/openfly/scripts/eval_satnav.sh \
  --model_dir /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/openfly-baseline \
  --model_name 'smoketest/${SMOKE_NAME}' \
  --gpus 8 \
  --max_episodes 10
" 2>&1 | tee "/tmp/openfly_smoke_eval.log"
```

OpenFly eval 会从模型名解析 `-act<format>` 与 `-hist<N>`；smoke 名没有这些字段时默认
`action_format=auto -> compact`、`action_history_limit=16`。

### Cleanup

```bash
ssh 10.246.152.73 "
  rm -rf /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/openfly-baseline/smoketest/${SMOKE_NAME}
  rm -rf /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/openfly-baseline/smoketest/${SMOKE_NAME}
"
```

---

## Loss 验证

所有 baseline 训练完成后执行以下检查（将 `LOGFILE` 替换为对应的 train log 路径）：

```bash
python3 -c "
import re, sys

log = open('LOGFILE').read()
losses = [float(x) for x in re.findall(r\"'loss': ([0-9.eE+\-]+)\", log)]
print('Step losses:', losses)

assert len(losses) > 0, 'ERROR: No loss data in log'

for i, l in enumerate(losses):
    assert l == l, f'ERROR: NaN at step {i+1}'
    assert l < 1e9, f'ERROR: Overflow at step {i+1}: {l}'

print('PASS: all losses finite')

if len(losses) >= 2:
    first = losses[0]
    last = losses[-1]
    print(f'First loss: {first:.4f} → Last loss: {last:.4f}')
    if last < first:
        print('PASS: loss is decreasing')
    elif last < first * 1.5:
        print('WARN: loss did not decrease but stayed stable (acceptable for warmup)')
    else:
        print('FAIL: loss increased significantly — check for training issues')
        sys.exit(1)
"
```

**判断标准：**

| 情况 | 结论 |
|---|---|
| 所有 loss 有限，final < initial | ✅ PASS |
| 所有 loss 有限，final ≈ initial（< 1.5x） | ⚠️ 可接受，记录并说明 |
| 出现 NaN / Inf / final > 1.5x initial | ❌ FAIL，需排查 |

---

## Eval 验证

所有 baseline eval 完成后检查：

```bash
# streamvln
for SPLIT in val_seen val_unseen; do
  echo "=== streamvln ${SPLIT} ===" && \
  cat results/streamvln-baseline/smoketest/${SMOKE_EXP}/${SPLIT}/evaluation_summary.json
done

# navila
NAVILA_SMOKE_MODEL="${SMOKE_NAME}-sample-hk7-fs7-stopx4"
for SPLIT in val_seen val_unseen; do
  echo "=== navila ${SPLIT} ===" && \
  cat results/navila-baseline/smoketest/${NAVILA_SMOKE_MODEL}/${SPLIT}/evaluation_summary.json
done

# uninavid
for SPLIT in val_seen val_unseen; do
  echo "=== uninavid ${SPLIT} ===" && \
  ssh 10.246.152.73 "cat /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/uninavid-baseline/smoketest/${SMOKE_NAME}/${SPLIT}/evaluation_summary.json"
done

# openfly
for SPLIT in val_seen val_unseen; do
  echo "=== openfly ${SPLIT} ===" && \
  ssh 10.246.152.73 "cat /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/openfly-baseline/smoketest/${SMOKE_NAME}/${SPLIT}/evaluation_summary.json"
done
```

**验收标准：**

- `evaluation_summary.json` 文件存在；
- eval log 包含 10/10 episodes 完成（无中途 crash）；
- 可能出现 `Camera view bounds exceed image bounds` 被 except 捕获，这是已知行为，不影响通过；
- SR / NE 数值不作性能要求，仅验证流程完整性。

---

## Report 模板

完成后报告以下内容：

| 项目 | 状态 |
|---|---|
| Baseline | streamvln / navila / uninavid / openfly |
| 数据版本 | `SatNav-v0.1` |
| 训练步数 | 8 步 |
| Step losses | `[x.xx, x.xx, ...]` |
| Loss 趋势 | ✅ PASS / ⚠️ 稳定 / ❌ FAIL |
| Checkpoint 存在 | ✅/❌ |
| Eval val_seen 10 eps 完成 | ✅/❌ |
| Eval val_unseen 10 eps 完成 | ✅/❌ |
| Eval Summary 写入（两 split） | ✅/❌ |
| 清理完成 | ✅/❌ |
