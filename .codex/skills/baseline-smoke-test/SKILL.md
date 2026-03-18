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
- 评测：`val_seen` split，`max_episodes=10`，1 GPU。
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

### Train Smoke

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate streamvln-baseline
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN

SMOKE_TEST=true \
MAX_STEPS=8 \
SAVE_STRATEGY=steps \
SAVE_STEPS=8 \
LOGGING_STEPS=1 \
bash baseline/streamvln/scripts/train_satnav.sh continue \
  2>&1 | tee /tmp/streamvln_smoke_train.log

# 训练结束后捕获自动生成的 EXP_NAME
SMOKE_EXP=$(ls -t output/streamvln-baseline/smoketest/ | head -1)
echo "Smoke EXP: ${SMOKE_EXP}"
```

> 注意：streamvln 的 train_satnav.sh 自动使用 `SATNAV_DATA_ROOT` 下的最新数据版本，无需手动传入。

### 验证训练产物

```bash
# checkpoint 或合并模型在根目录（safetensors），eval 使用根目录
ls output/streamvln-baseline/smoketest/${SMOKE_EXP}/
# 预期：含 safetensors 或 checkpoint-8/ 子目录（eval 脚本自动处理）
```

### Eval Smoke

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate streamvln-baseline
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN

SATNAV_VERSION="" bash baseline/streamvln/scripts/eval_satnav.sh \
  smoketest/${SMOKE_EXP} val_seen 1 10 \
  2>&1 | tee /tmp/streamvln_smoke_eval.log
```

> 注意：当前 `baseline/streamvln/scripts/eval_satnav.sh` 在 `set -u` 下会直接读取 `SATNAV_VERSION`，若未定义可能报 `unbound variable`。smoke 命令建议显式加 `SATNAV_VERSION=""`。

### Cleanup

```bash
REPO=/mnt/data1/home/jiangjiajun/workspace/SwiftVLN
rm -rf ${REPO}/output/streamvln-baseline/smoketest/${SMOKE_EXP}
rm -rf ${REPO}/results/streamvln-baseline/smoketest/${SMOKE_EXP}
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

bash baseline/navila/scripts/eval_satnav.sh \
  "smoketest/${SMOKE_NAME}" val_seen 1 10 \
  2>&1 | tee /tmp/navila_smoke_eval.log
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
ssh 10.246.152.73 "
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate uninavid-baseline
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN

bash baseline/uninavid/scripts/eval_satnav.sh \
  'smoketest/${SMOKE_NAME}' val_seen 1 10
" 2>&1 | tee /tmp/uninavid_smoke_eval.log
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
cat results/streamvln-baseline/smoketest/${SMOKE_EXP}/val_seen/evaluation_summary.json

# navila
cat results/navila-baseline/smoketest/${SMOKE_NAME}/val_seen/evaluation_summary.json

# uninavid
ssh 10.246.152.73 "cat /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/uninavid-baseline/smoketest/${SMOKE_NAME}/val_seen/evaluation_summary.json"
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
| Eval 10 episodes 完成 | ✅/❌ |
| Eval Summary 写入 | ✅/❌ |
| 清理完成 | ✅/❌ |
