---
name: run-streamvln-baseline
description: "在 SatNav 数据上执行 StreamVLN baseline 的训练与评测（8卡全参数），包含服务器选择、数据版本选择、规范化实验命名、checkpoint 校验与按名称评测。适用于：run streamvln baseline / streamvln基线训练 / 用satnav数据训练streamvln / 启动streamvln基线 / 评测streamvln baseline。"
---

# StreamVLN Baseline 训练与评测

---

## 概览

### 本 skill 提供能力

- 8 卡全量训练 StreamVLN baseline on SatNav data
- 两种训练模式：`continue`（从官方 checkpoint fine-tune）或 `scratch`（从 LLaVA-Video-7B-Qwen2 开始）
- 支持选择 SatNav 数据版本（默认最新）
- 支持指定训练服务器（98 / 73 / 17）
- 训练完成后 eval by name

### 服务器与 Conda 约定

> 服务器详细配置（IP、访问方式、GPU 数量、conda envs、tmux 命名规范）统一维护在：
> `.codex/CODEX_CONTEXT.md` → `Runtime/Infra Conventions`

本 skill 使用的 conda env：`streamvln-baseline`

### Conda 初始化

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate streamvln-baseline
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN
```

### Webhook（强制）

本 skill 强制发送 4 个通知：`训练开始`、`训练结束`、`评测开始`、`评测结束`。

Webhook URL（与现有 skill 保持一致）：

- Train webhook  
  `https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=2504fe89-9e8a-4767-9e12-61383bbe456e`
- Eval webhook  
  `https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=2504fe89-9e8a-4767-9e12-61383bbe456e`

建议先定义一个通用发送函数（后续步骤直接复用）：

```bash
send_wecom_markdown() {
  local webhook_url="$1"
  local content="$2"
  curl -sS -X POST "$webhook_url" \
    -H "Content-Type: application/json" \
    -d "{\"msgtype\":\"markdown\",\"markdown\":{\"content\":\"${content//$'\n'/\\n}\"}}"
}
```

### 固定路径（无需额外搜索）

| 用途 | 路径 |
|---|---|
| Train script | `baseline/streamvln/scripts/train_satnav.sh` |
| Eval script | `baseline/streamvln/scripts/eval_satnav.sh` |
| Train+Eval pipeline script | `baseline/streamvln/scripts/train_eval_satnav.sh` |
| Train implementation | `baseline/streamvln/src/train_satnav.py` |
| Eval implementation | `baseline/streamvln/src/eval_satnav.py` |
| SatNav config template | `baseline/streamvln/configs/satnav_task.yaml` |
| DeepSpeed config | `baseline/streamvln/configs/zero2.json` |
| Official checkpoint | `baseline/streamvln/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3` |
| Base model (scratch) | `baseline/streamvln/model/LLaVA-Video-7B-Qwen2` |
| Vision model | `baseline/streamvln/model/siglip-so400m-patch14-384` |
| SatNav data root | `/mnt/data3/jiangjiajun/dataset/satnav_datasets` |
| StreamVLN project | `/mnt/data1/home/jiangjiajun/workspace/StreamVLN` |

### 输出路径

所有路径相对于 SwiftVLN repo root（`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN`）：

| 用途 | 路径 |
|---|---|
| Training output dir | `output/streamvln-baseline/<EXP_NAME>/` |
| Smoke training output dir | `output/streamvln-baseline/smoketest/<EXP_NAME>/` |
| Training log | `output/streamvln-baseline/.../train.log` |
| Checkpoint | `output/streamvln-baseline/.../checkpoint-*` |
| Eval results | `results/streamvln-baseline/<EXP_NAME_or_subpath>/<split>/` |
| Eval log | `results/streamvln-baseline/<EXP_NAME_or_subpath>/<split>/eval.log` |

### EXP_NAME 命名规范

```
streamvln-baseline-{mode}-{epochs}ep-f{frames}h{history}s{future}-data{ver}-bs{eff_bs}-lr{lr}-{timestamp}
```

示例：
```
streamvln-baseline-continue-1ep-f32h8s4-data260306-bs32-lr2e-5-20260309-143000
streamvln-baseline-scratch-1ep-f32h8s4-data260306-bs32-lr2e-5-20260309-150000
```

---

## 步骤 1 — 参数确认与预检查

### 与用户确认

开始前确认：

| 参数 | 默认值 | 说明 |
|---|---|---|
| Training mode | `continue` | `continue` or `scratch` |
| Target server | — | `98` / `73` / `17`，需用户明确指定 |
| SatNav data version | auto-detect latest | 如不指定则自动用最新版本 |
| SwanLab | `false` | 是否开启 SwanLab 上报 |
| Webhook notification | `true` | 强制开启，发送 train/eval 开始和结束通知 |

**继续前需等待用户确认。**

### 预检查（在目标服务器执行）

若目标是 73/17，SSH 进去后执行检查：

```bash
# GPU 可用性
nvidia-smi

# 检查 conda env
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate streamvln-baseline

# 检查关键路径
ls baseline/streamvln/scripts/train_satnav.sh
ls baseline/streamvln/scripts/eval_satnav.sh
ls baseline/streamvln/configs/zero2.json
```

**模型权重检查：**

- continue 模式：`baseline/streamvln/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3/`
- scratch 模式：`baseline/streamvln/model/LLaVA-Video-7B-Qwen2/`
- 通用：`baseline/streamvln/model/siglip-so400m-patch14-384/`

若任何检查失败，**立即停止并报告缺失项**。

---

## 步骤 2 — 选择服务器

若用户已指定服务器，跳过 GPU 调查，直接确认该服务器可用。

若用户未指定，检查三台服务器的 GPU 占用情况：

```bash
# 98 (本机)
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader

# 73
ssh 10.246.152.73 nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader

# 17 (Docker)
ssh 10.246.132.17 "docker exec <container_name> nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader"
```

选出实际空闲的 8-GPU 节点，说明选择理由。

**选定服务器后需等待用户确认。**

---

## 步骤 3 — 选择数据版本

SatNav 数据位于 `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_XXXXXX/`。

### 自动检测（默认）

训练脚本会自动选最新版本，也可手动确认当前最新：

```bash
ls -d /mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_* | sort | tail -1
```

### 手动指定版本

```bash
export SATNAV_VERSION=ver_260306
```

### 校验数据完整性

```bash
VERSION_DIR="/mnt/data3/jiangjiajun/dataset/satnav_datasets/${SATNAV_VERSION}"
ls "${VERSION_DIR}/trajectory_data/annotations.json"        # 训练数据
ls "${VERSION_DIR}/trajectory_data/images/"
ls "${VERSION_DIR}/episodes/eval/val_seen/all_episodes.json"   # eval episodes (val_seen)
```

---

## 步骤 4 — 启动训练

> **所有训练必须在 tmux session 中启动**，确保训练在 Codex session 断开后继续运行，并可通过 `tmux capture-pane` 检查状态。

### 默认方式：一键串行 train + eval（推荐）

**除非用户明确要求"只训练不评测"，否则默认使用 `train_eval_satnav.sh`。**

该脚本自动完成全流程：训练 → 解析 EXP_NAME → 评测 → 4 条 webhook 全自动发送。训练失败时不进入 eval。

### tmux 会话命名

```
train_streamvln_baseline_<HHMMSS>
```

示例：`train_streamvln_baseline_143025`

### 在 98 启动（本机）

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate streamvln-baseline
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN

SESSION="train_streamvln_baseline_$(date +%H%M%S)"
LOG="/tmp/${SESSION}.log"

tmux new-session -d -s "${SESSION}" \
  "source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh && \
   conda activate streamvln-baseline && \
   cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN && \
   SATNAV_VERSION=ver_260306 TRAIN_GPUS=8 EVAL_GPUS=8 \
   bash baseline/streamvln/scripts/train_eval_satnav.sh continue \
   2>&1 | tee ${LOG}"

tmux ls | grep "${SESSION}"
```

### 在 73 启动（远程 SSH）

```bash
SESSION="train_streamvln_baseline_$(date +%H%M%S)"
LOG="/tmp/${SESSION}.log"

ssh 10.246.152.73 "
  tmux new-session -d -s '${SESSION}' \
  'source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh && \
   conda activate streamvln-baseline && \
   cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN && \
   SATNAV_VERSION=ver_260306 TRAIN_GPUS=8 EVAL_GPUS=8 \
   bash baseline/streamvln/scripts/train_eval_satnav.sh continue \
   2>&1 | tee ${LOG}'
"

ssh 10.246.152.73 "tmux ls | grep '${SESSION}'"
```

### 在 17 启动（远程 SSH + Docker）

```bash
SESSION="train_streamvln_baseline_$(date +%H%M%S)"
LOG="/tmp/${SESSION}.log"

ssh 10.246.132.17 "
  docker exec \$(docker ps --format '{{.Names}}' | head -1) bash -c \"
    tmux new-session -d -s '${SESSION}' \
    'source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh && \
     conda activate streamvln-baseline && \
     cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN && \
     SATNAV_VERSION=ver_260306 TRAIN_GPUS=8 EVAL_GPUS=8 \
     bash baseline/streamvln/scripts/train_eval_satnav.sh continue \
     2>&1 | tee ${LOG}'
  \"
"
```

### 仅训练模式（不自动 eval）

当用户明确要求"只训练"时，改用 `train_satnav.sh`，需手动发送 webhook：

```bash
SESSION="train_streamvln_baseline_$(date +%H%M%S)"
LOG="/tmp/${SESSION}.log"

TRAIN_WEBHOOK_URL="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=2504fe89-9e8a-4767-9e12-61383bbe456e"
TRAIN_START_TS="$(date +%s)"
TRAIN_START_MSG="## StreamVLN Baseline Train Started
server: <98|73|17>
mode: <continue|scratch>
satnav_version: <ver_xxxxxx>
tmux_session: ${SESSION}
time: $(date '+%Y-%m-%d %H:%M:%S')"
send_wecom_markdown "${TRAIN_WEBHOOK_URL}" "${TRAIN_START_MSG}"

tmux new-session -d -s "${SESSION}" \
  "source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh && \
   conda activate streamvln-baseline && \
   cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN && \
   bash baseline/streamvln/scripts/train_satnav.sh continue \
   2>&1 | tee ${LOG}"

tmux ls | grep "${SESSION}"
```

### 使用自定义参数

```bash
SATNAV_VERSION=ver_260306 \
NUM_EPOCHS=1 \
LEARNING_RATE=2e-5 \
BATCH_SIZE=2 \
GRAD_ACCUM=2 \
USE_SWANLAB=true \
USE_WXWORK_NOTIFICATION=true \
  bash baseline/streamvln/scripts/train_satnav.sh continue
```

### 可用环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SATNAV_VERSION` | auto-detect latest | SatNav data version (e.g. `ver_260306`) |
| `NUM_EPOCHS` | `1` | Training epochs |
| `LEARNING_RATE` | `2e-5` | Learning rate |
| `BATCH_SIZE` | `2` | Per-device batch size |
| `GRAD_ACCUM` | `2` | Gradient accumulation steps |
| `GPUS_PER_NODE` | `8` | Number of GPUs |
| `USE_SWANLAB` | `false` | Enable SwanLab cloud logging |
| `USE_WXWORK_NOTIFICATION` | `false` | Enable WXWork webhook on completion |
| `SMOKE_TEST` | `false` | Save under `output/streamvln-baseline/smoketest/` when `true` |
| `SAVE_STRATEGY` | `epoch` | `epoch` or `steps` |
| `SAVE_STEPS` | `1000` | Save interval (when `SAVE_STRATEGY=steps`) |

> **脚本硬编码的训练超参**（不可通过环境变量覆盖）：
> - `warmup_ratio=0.075`
> - `lr_scheduler_type=cosine_with_min_lr`（`min_lr=1.85e-05`）
> - `mm_vision_tower_lr=5e-6`
> - `weight_decay=0.0`
>
> 如需修改这些参数，直接编辑 `baseline/streamvln/scripts/train_satnav.sh`。

### 监控循环

启动后立即进入监控循环。Codex 自身执行 sleep-check-act，如同人在终端前操作。

**循环参数：**

| 参数 | 值 | 说明 |
|---|---|---|
| MONITOR_INTERVAL | `90s` | 正常轮询间隔 |
| ALERT_INTERVAL | `45s` | 检测到异常后缩短的轮询间隔 |
| MAX_IDLE_CYCLES | `10` | 连续无进展轮次上限，超过则通知用户 |

**每轮检查清单：**

```bash
# 1. session 是否存活
tmux has-session -t "${SESSION}" 2>/dev/null && echo "RUNNING" || echo "DONE/CRASHED"

# 2. 最新训练进度
tmux capture-pane -pt "${SESSION}" | tail -30

# 3. 日志错误扫描
grep -E "Traceback|RuntimeError|CUDA out of memory|NCCL" "${LOG}" | tail -10

# 4. 进程存在性（训练主机上执行）
pgrep -f "train_satnav" >/dev/null && echo "PROCESS_ALIVE" || echo "PROCESS_GONE"
```

**异常处理与模式切换：**

- 检测到异常（OOM / NCCL / session 消失）→ **切换到 ALERT_INTERVAL（45s）**，尝试自动修复（见下）
- 自动修复成功且连续 3 轮正常 → **恢复到 MONITOR_INTERVAL（90s）**
- 连续 MAX_IDLE_CYCLES（10）轮无新 loss 输出 → 运行深度诊断并**通知用户**，附带最近日志证据
- session 消失 + 训练未完成 → 检查 `train.log` 确认崩溃原因

**自动修复尝试（失败前先试）：**

| 异常信号 | 修复动作 |
|---|---|
| `CUDA out of memory` | 减小 `BATCH_SIZE`，重新启动 |
| `NCCL` / `torchrun` 崩溃 | 检查端口冲突，重启 tmux session |
| session 消失但训练未完成 | 检查 `train.log` 排查原因，修复后重启 |

**退出条件：**

- 训练正常完成（日志出现 `Training completed!`）→ 退出循环，进入校验步骤
- 训练失败且自动修复无效 → 退出循环，发送失败 webhook，报告用户
- 用户主动中断 → 退出循环

### 校验训练产物

训练完成后从日志中记录 EXP_NAME，然后验证：

```bash
ls output/streamvln-baseline/<EXP_NAME>/checkpoint-*/config.json
ls output/streamvln-baseline/<EXP_NAME>/checkpoint-*/*.safetensors 2>/dev/null || \
ls output/streamvln-baseline/<EXP_NAME>/checkpoint-*/*.bin
```

- [ ] checkpoint 目录存在
- [ ] 包含 `config.json` + 权重文件（`.safetensors` 或 `.bin`）
- [ ] `train.log` 无 `Traceback` / `RuntimeError`
- [ ] 训练 loss 有下降趋势

### 发送训练结束 webhook（仅训练模式需要，一键串行模式自动发送）

训练结束后（无论成功失败）发送：

```bash
TRAIN_END_TS="$(date +%s)"
TRAIN_DURATION_SEC=$((TRAIN_END_TS - TRAIN_START_TS))
TRAIN_STATUS="SUCCESS"   # 失败时改为 FAILED
TRAIN_OUTPUT_DIR="output/streamvln-baseline/<EXP_NAME>"
TRAIN_LOG_PATH="${TRAIN_OUTPUT_DIR}/train.log"

TRAIN_END_MSG="## StreamVLN Baseline Train Finished
status: ${TRAIN_STATUS}
exp_name: <EXP_NAME>
duration_sec: ${TRAIN_DURATION_SEC}
output_dir: ${TRAIN_OUTPUT_DIR}
log: ${TRAIN_LOG_PATH}
time: $(date '+%Y-%m-%d %H:%M:%S')"

send_wecom_markdown "${TRAIN_WEBHOOK_URL}" "${TRAIN_END_MSG}"
```

---

## 步骤 5 — 执行评测（默认 `val_seen` + `val_unseen` 都跑）

> **必须发送 eval 开始 / 结束通知。**
> 本 skill 当前约定：**默认 val_seen 和 val_unseen 两个 split 均跑（8卡）**。若用户明确指定了单个 split，则只跑指定的那个。
> 0319 起 `episodes/eval/` 下只有 `val_seen/` 和 `val_unseen/` 子目录，不再有顶层扁平文件。
>
> 如果使用了一键串行模式（`train_eval_satnav.sh`），eval 已自动执行且 webhook 已自动发送，可跳过本步骤中的手动 eval 和手动 webhook 部分，直接进入"校验评测产物"。

### 评测前清理（必做）

清理本次评测目标的旧结果和相关后台进程，**不影响其他实验的历史结果**：

```bash
# 1) 停掉所有 eval 相关后台
tmux ls | grep streamvln_eval | awk -F: '{print $1}' | xargs -r -n1 tmux kill-session -t
pkill -f "baseline/streamvln/scripts/eval_satnav.sh|baseline/streamvln/src/eval_satnav.py|streamvln_eval" || true

# 2) 仅清理当前实验的 eval 结果（保留其他实验的历史结果）
rm -rf "results/streamvln-baseline/<EXP_NAME>/val_seen"
rm -rf "results/streamvln-baseline/<EXP_NAME>/val_unseen"
rm -rf "results/streamvln-baseline/<EXP_NAME>/test"
```

> **禁止**使用 `rm -rf results/streamvln-baseline/by-path` 或 `find ... -name val_unseen ... -exec rm` 等全局清理命令——这会误删其他实验的已有评测结果。

### 评测数据约束（按 split 子目录路由）

默认直接使用：

`/mnt/data3/jiangjiajun/dataset/satnav_datasets/<ver_xxxxxx>/episodes/eval/val_seen/all_episodes.json`

**eval_satnav.sh 根据 `SPLIT` 参数自动路由到对应子目录**（`val_seen/` 或 `val_unseen/`）。
**若不传 `SPLIT` 参数，则脚本会顺序运行 `val_seen` 和 `val_unseen` 两个 split。**
不做城市过滤，使用全量 eval episodes。

### 按实验名评测（推荐）

从训练日志末尾获取 EXP_NAME。若未指定单个 split，直接调用一次即可默认顺序跑完 `val_seen` 和 `val_unseen`：

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate streamvln-baseline
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN

# 默认两个 split 都跑（除非用户明确指定了单个 split）
bash baseline/streamvln/scripts/eval_satnav.sh <EXP_NAME> "" 8
```

如果需要为每个 split 单独发送 webhook，可按下面形式显式循环：

```bash
EVAL_WEBHOOK_URL="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=2504fe89-9e8a-4767-9e12-61383bbe456e"

for SPLIT in val_seen val_unseen; do
  EVAL_START_TS="$(date +%s)"

  EVAL_START_MSG="## StreamVLN Baseline Eval Started
exp_name: <EXP_NAME>
split: ${SPLIT}
gpus: 8
time: $(date '+%Y-%m-%d %H:%M:%S')"
  send_wecom_markdown "${EVAL_WEBHOOK_URL}" "${EVAL_START_MSG}"

  if bash baseline/streamvln/scripts/eval_satnav.sh <EXP_NAME> "${SPLIT}" 8; then
    eval_rc=0
  else
    eval_rc=$?
  fi

  EVAL_END_TS="$(date +%s)"
  EVAL_DURATION_SEC=$((EVAL_END_TS - EVAL_START_TS))
  EVAL_STATUS="SUCCESS"
  [ $eval_rc -ne 0 ] && EVAL_STATUS="FAILED"

  EVAL_END_MSG="## StreamVLN Baseline Eval Finished
status: ${EVAL_STATUS}
exp_name: <EXP_NAME>
split: ${SPLIT}
duration_sec: ${EVAL_DURATION_SEC}
eval_log: results/streamvln-baseline/<EXP_NAME>/${SPLIT}/eval.log
time: $(date '+%Y-%m-%d %H:%M:%S')"
  send_wecom_markdown "${EVAL_WEBHOOK_URL}" "${EVAL_END_MSG}"
done
```

### 评测时覆盖数据版本

```bash
SATNAV_VERSION=ver_260306 \
  bash baseline/streamvln/scripts/eval_satnav.sh <EXP_NAME> val_seen 8
```

### 按 checkpoint 路径评测（兼容旧方式）

```bash
bash baseline/streamvln/scripts/eval_satnav.sh /path/to/checkpoint val_seen 8
```

### 校验评测产物

- [ ] `results/streamvln-baseline/<EXP_NAME_or_subpath>/<split>/eval.log` 存在
- [ ] 日志无 `Traceback` / `RuntimeError`
- [ ] 指标出现在日志中（SR, SPL, nDTW 等）

---

## 步骤 6 — 结果汇报

### 通过/失败矩阵

| 服务器 | 模式 | 数据版本 | 训练 | EXP_NAME | Checkpoint | Split | 评测 | SR / SPL | 失败原因 |
|---|---|---|---|---|---|---|---|---|---|
| 98/73/17 | continue | ver_XXXXXX | ✅/❌ | `...` | `output/...` | val_seen | ✅/❌ | X.X / X.X | — |
| | | | — | | | val_unseen | ✅/❌ | X.X / X.X | — |
| 98/73/17 | scratch | ver_XXXXXX | ✅/❌ | `...` | `output/...` | val_seen | ✅/❌ | X.X / X.X | — |
| | | | — | | | val_unseen | ✅/❌ | X.X / X.X | — |

### 报告还应包含

1. 实际执行的命令
2. 关键日志 / checkpoint / 结果路径
3. 训练耗时和 GPU 利用情况
4. 任何异常或告警
5. 4 条 webhook 已发送的证据（时间戳 + 阶段）

---

## 故障排查

| 问题 | 解决方案 |
|---|---|
| `No SatNav data versions found` | 检查 `/mnt/data3/jiangjiajun/dataset/satnav_datasets/` 是否有 `ver_*` 目录 |
| `Official checkpoint not found` | 运行 `bash baseline/streamvln/scripts/download_model.sh` |
| `Base model not found` | 检查 `baseline/streamvln/model/LLaVA-Video-7B-Qwen2/` 是否存在 |
| `CUDA out of memory` | 减小 `BATCH_SIZE`（默认 2）或 `GRAD_ACCUM` |
| `No checkpoint found in exp dir` | 训练可能未完成，检查 `train.log` |
| `tokenizer_config.json not found` | 官方 checkpoint 正常现象，eval 脚本自动 fallback 到本地 `LLaVA-Video-7B-Qwen2` |
| `torchrun` 端口冲突 | 脚本用随机端口，重试即可 |
| server 17 Docker 容器名未知 | `ssh 10.246.132.17 "docker ps"` 查看容器名 |

## 报错处理与 Skill 自修复（强制）

出现任何报错时，必须先定位根因，再决定修复动作：

1. 先判断错误类型：
   - 运行环境/资源问题（GPU、端口、路径、权限、依赖）
   - 数据或checkpoint问题
   - 命令参数/流程问题
   - **skill 本身问题**（文档命令错误、步骤顺序错误、缺少关键前置条件、webhook示例不正确等）

2. 若是运行问题：
   - 先修复并重试当前任务；
   - 在最终报告中记录：错误现象、根因、修复动作、结果。

3. 若确认是 **skill 本身问题**：
   - 先修复当前任务使其可继续执行；
   - **必须同步修复本 skill 文档本身**（必要时连同相关脚本一起修复）；
   - 在最终报告中明确列出已修复的 skill 条目与对应文件路径。

4. 禁止仅"临时绕过"而不回写 skill：
   - 任何可复现的 skill 缺陷都应固化修复，避免下次重复踩坑。

### 训练失败恢复

1. 训练中途崩溃且有 partial checkpoint → 删除输出目录后重跑，或手动指定 checkpoint 继续
2. tmux session 消失 → 检查 `train.log` 确认失败原因，修复后重新启动 tmux session
