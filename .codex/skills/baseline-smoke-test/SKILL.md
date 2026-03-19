---
name: baseline-smoke-test
description: "Run smoke tests for any baseline (streamvln / navila / uninavid) on SatNav: 8-GPU train smoke with latest production data (limited steps), checkpoint handoff to eval smoke (limited episodes), loss trend verification, and post-test cleanup."
---

# Baseline Smoke Test

## 触发条件

当用户说以下任意形式时使用：

- "跑一下 baseline `<name>` 的 smoke test"
- "baseline `<name>` 冒烟测试"
- "`<name>` baseline smoke"

其中 `<name>` ∈ `streamvln` | `navila` | `uninavid`

**第一步：从用户输入中识别 baseline 名称，然后跳转到对应分支。**

---

## 通用原则（所有 baseline 一致）

- 全部固定 **8 GPU** 训练。
- 全部使用 **生产数据最新版本**（auto-detect），不使用 `smoke_test_data/` 专用数据。
- 训练：`MAX_STEPS=8`，`SAVE_STRATEGY=steps`，`SAVE_STEPS=8`，`LOGGING_STEPS=1`。
- 评测：尽量贴近正式默认路径；若 baseline 支持则优先用正式 split，仅缩小 episode 数量。
- baseline eval 入口：
  `baseline/streamvln/scripts/eval_satnav.sh`
  `baseline/navila/scripts/eval_satnav.sh`
  `baseline/uninavid/scripts/eval_satnav.sh`
- 上述三个脚本在 **不显式传 split** 时，均默认顺序运行 `val_seen` 和 `val_unseen`。
- 验证训练 loss 是否有限且呈下降趋势（见 [Loss 验证](#loss-验证) 节）。
- 验证评测 summary 写入成功。
- 完成后**必须清理** smoke 输出和结果目录。

---

## Step 0 — 自动检测最新数据版本

**适用于所有 baseline。** 在构建命令前先确定数据路径：

```bash
SATNAV_DATA_ROOT="/mnt/data3/jiangjiajun/dataset/satnav_datasets"
LATEST_VER=$(ls -d "${SATNAV_DATA_ROOT}"/ver_* | sort | tail -1 | xargs basename)
LATEST_DATA_DIR="${SATNAV_DATA_ROOT}/${LATEST_VER}/trajectory_data"
LATEST_ANNOTATIONS="${LATEST_DATA_DIR}/annotations.json"
echo "Latest SatNav version: ${LATEST_VER}"
echo "Annotations: ${LATEST_ANNOTATIONS}"
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
ls "${LATEST_ANNOTATIONS}"
```

### Full-Path Smoke（缩步数，保留正式主路径）

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate streamvln-baseline
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN

SMOKE_PIPE_LOG="/tmp/streamvln_smoke_train_eval_$(date +%Y%m%d_%H%M%S).log"

SMOKE_TEST=true \
SATNAV_VERSION="${LATEST_VER}" \
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
ls baseline/navila/model/navila-siglip-llama3-8b-v1.5-pretrain/config.json
ls "${LATEST_ANNOTATIONS}"
```

### Train Smoke

```bash
SMOKE_NAME="smoke_navila_$(date +%Y%m%d_%H%M%S)"

source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate navila-baseline
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN

DATA_PATH="${LATEST_ANNOTATIONS}" \
IMAGE_FOLDER="${LATEST_DATA_DIR}" \
MAX_STEPS=8 \
SAVE_STEPS=8 \
SAVE_TOTAL_LIMIT=1 \
TRAIN_BSZ=1 \
GRAD_ACCUM=1 \
DATALOADER_WORKERS=2 \
REPORT_TO=none \
MASTER_PORT=29640 \
bash baseline/navila/scripts/train_satnav.sh "smoketest/${SMOKE_NAME}" \
  2>&1 | tee /tmp/navila_smoke_train.log
```

> `CUSTOM_EXP_NAME = smoketest/${SMOKE_NAME}` → 输出至 `output/navila-baseline/smoketest/${SMOKE_NAME}/`

### 验证训练产物

验证 `output/navila-baseline/smoketest/${SMOKE_NAME}/` 下包含：
- `config.json`
- `llm/`（NaVILA 特有的合并输出格式）
- 无 `Traceback` 字样

如果只有 `checkpoint-8/` 子目录（DeepSpeed 格式），NaVILA eval 脚本会自动选取它。

### Eval Smoke

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate navila-baseline
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN

# smoke 为了分别保留两边日志，这里显式逐 split 执行；
# 正式调用若不传 split，脚本默认也会顺序跑 val_seen + val_unseen。
for SPLIT in val_seen val_unseen; do
  bash baseline/navila/scripts/eval_satnav.sh \
    "smoketest/${SMOKE_NAME}" "${SPLIT}" 1 10 \
    2>&1 | tee "/tmp/navila_smoke_eval_${SPLIT}.log"
done
```

### Cleanup

```bash
REPO=/mnt/data1/home/jiangjiajun/workspace/SwiftVLN
rm -rf ${REPO}/output/navila-baseline/smoketest/${SMOKE_NAME}
rm -rf ${REPO}/results/navila-baseline/smoketest/${SMOKE_NAME}
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
  ls '"${LATEST_ANNOTATIONS}"'
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

DATA_PATH='${LATEST_ANNOTATIONS}' \
VIDEO_FOLDER='${LATEST_DATA_DIR}' \
MAX_STEPS=8 \
SAVE_STRATEGY=steps \
SAVE_STEPS=8 \
SAVE_TOTAL_LIMIT=1 \
TRAIN_BSZ=1 \
GRAD_ACCUM=1 \
DATALOADER_WORKERS=0 \
REPORT_TO=none \
MASTER_PORT=29618 \
bash baseline/uninavid/scripts/train_satnav.sh 'smoketest/${SMOKE_NAME}'
" 2>&1 | tee /tmp/uninavid_smoke_train.log
```

> 输出至 `output/uninavid-baseline/smoketest/${SMOKE_NAME}/`

### Eval Smoke（在 73 上执行）

```bash
# smoke 为了分别保留两边日志，这里显式逐 split 执行；
# 正式调用若不传 split，脚本默认也会顺序跑 val_seen + val_unseen。
for SPLIT in val_seen val_unseen; do
  ssh 10.246.152.73 "
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate uninavid-baseline
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN
bash baseline/uninavid/scripts/eval_satnav.sh \
  'smoketest/${SMOKE_NAME}' '${SPLIT}' 1 10
" 2>&1 | tee "/tmp/uninavid_smoke_eval_${SPLIT}.log"
done
```

### Cleanup

```bash
ssh 10.246.152.73 "
  rm -rf /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/uninavid-baseline/smoketest/${SMOKE_NAME}
  rm -rf /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/uninavid-baseline/smoketest/${SMOKE_NAME}
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
for SPLIT in val_seen val_unseen; do
  echo "=== navila ${SPLIT} ===" && \
  cat results/navila-baseline/smoketest/${SMOKE_NAME}/${SPLIT}/evaluation_summary.json
done

# uninavid
for SPLIT in val_seen val_unseen; do
  echo "=== uninavid ${SPLIT} ===" && \
  ssh 10.246.152.73 "cat /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/uninavid-baseline/smoketest/${SMOKE_NAME}/${SPLIT}/evaluation_summary.json"
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
| Baseline | streamvln / navila / uninavid |
| 数据版本 | `ver_XXXXXX` |
| 训练步数 | 8 步 |
| Step losses | `[x.xx, x.xx, ...]` |
| Loss 趋势 | ✅ PASS / ⚠️ 稳定 / ❌ FAIL |
| Checkpoint 存在 | ✅/❌ |
| Eval val_seen 10 eps 完成 | ✅/❌ |
| Eval val_unseen 10 eps 完成 | ✅/❌ |
| Eval Summary 写入（两 split） | ✅/❌ |
| 清理完成 | ✅/❌ |
