---
name: baseline-streamvln-smoke-test
description: "Smoke test for baseline/streamvln training + evaluation pipeline on SatNav data. Supports lite (single-GPU, adapter-only) and full (8-GPU, full-param) modes. Use when user asks to smoke test / 冒烟测试 the streamvln baseline, verify baseline/streamvln can train+eval on SatNav, or validate baseline pipeline changes."
---

# Baseline StreamVLN Smoke Test

Validates that `baseline/streamvln/` can load model, process SatNav data,
run training to completion, and evaluate the resulting checkpoint.

---

## Mode Selection (MUST confirm before running)

| 关键词 | 模式 | 说明 |
|---|---|---|
| 单卡 / 简单 / lite / adapter | **lite** | 1 GPU, 仅训 mm_mlp_adapter, 资源少, ~2 min |
| 多卡 / 8卡 / 复杂 / full / 全参 | **full** | 8 GPU, 全参数训练, 验证完整管线, ~5 min |

**如果用户未指定模式，必须先询问：**

> 请问你要跑哪种 smoke test？
> - **lite**（单卡，仅训 adapter，快速验证）
> - **full**（8卡全参数，完整验证）

---

## Key Quirks (do not skip)

- Launcher: **两种模式都用** `python -m torch.distributed.run`（`torchrun` 在此环境不在 PATH）
- nvcc: must prepend `/usr/local/cuda-13.0/bin` to `PATH`（system nvcc is CUDA 11.5, doesn't support sm_90）
- Conda env: `streamvln-baseline`
- Vision tower delay_load: config 中 `delay_load: true`，eval 时需显式调用 `vision_tower.load_model()`
  （已在 `src/eval_satnav.py` 中修复，但如果代码被覆盖需重新加入）
- lite 模式 checkpoint 只含 `mm_projector.bin`，eval 前需与基础模型合并
- **CUDA_VISIBLE_DEVICES 污染**：lite 设置了 `CUDA_VISIBLE_DEVICES=7`，若同一 shell 中续跑 full，必须先 `unset CUDA_VISIBLE_DEVICES`，否则 full 模式 rank 1-7 找不到 GPU 而报 `CUDA error: invalid device ordinal`
- **full 模式 eval checkpoint 路径**：full 训练保存的可直接用于 eval 的模型在**根目录** `output/streamvln-baseline/smoketest/smoke_full/`（含 safetensors）；子目录 `checkpoint-5/` 是 DeepSpeed 格式，无法直接用于 eval。
- **full 模式 eval camera 边界错误**：full 模型只训 5 步后导航能力很差，会让 agent 走到地图边缘，触发 SatNav 的 `Camera view bounds exceed image bounds`。此错误会被 `src/eval_satnav.py` 的 except 块捕获，每条 episode 得到 `distance_to_goal=inf`。smoke test 中这是**可接受的已知现象**，流程本身（10/10 episodes 跑完、summary 保存）验证通过即可。

---

## Fixed Paths

| Purpose | Path |
|---|---|
| Train script | `baseline/streamvln/src/train_satnav.py` |
| Train launch script | `baseline/streamvln/scripts/train_satnav.sh` |
| Eval script | `baseline/streamvln/src/eval_satnav.py` |
| Eval launch script | `baseline/streamvln/scripts/eval_satnav.sh` |
| DeepSpeed cfg | `baseline/streamvln/configs/zero2.json` |
| SatNav eval config | `baseline/streamvln/configs/satnav_task.yaml` |
| Model checkpoint | `baseline/streamvln/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3` |
| SigLIP vision tower | `baseline/streamvln/model/siglip-so400m-patch14-384` |
| SatNav data | `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260306/trajectory_data` |

---

## Mode Differences

| 参数 | lite | full |
|---|---|---|
| GPU 数 | 1 (CUDA_VISIBLE_DEVICES=7) | 8 (all, 须先 unset CUDA_VISIBLE_DEVICES) |
| mm_tunable_parts | `mm_mlp_adapter` | `mm_vision_tower,mm_mlp_adapter,mm_language_model` |
| per_device_train_batch_size | 1 | 2 |
| gradient_accumulation_steps | 1 | 2 |
| data_augmentation | False | True |
| torch_compile | False | True |
| model_max_length | 4096 | 32768 |
| dataloader_num_workers | 2 | 8 |
| report_to | none | none |
| Smoke output dir | `output/streamvln-baseline/smoketest/smoke_lite/` | `output/streamvln-baseline/smoketest/smoke_full/` |
| Checkpoint 内容 | 仅 mm_projector.bin + config.json | 完整模型权重（根目录 safetensors）|
| Eval 前需合并 | **是** | 否（直接用根目录）|

**共同参数**：max_steps=5, save_steps=5, save_total_limit=1, logging_steps=1, bf16=True, gradient_checkpointing=True, warmup_ratio=0.0, lr=2e-5, mm_vision_tower_lr=5e-6

---

## Step 1 — Preflight

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate streamvln-baseline
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN

python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
ls baseline/streamvln/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3/config.json
ls /mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260306/trajectory_data/annotations.json
```

Stop if any check fails.

---

## Step 2 — Run Smoke Training

### Step 2a — lite 模式（单卡 adapter-only）

```bash
REPO=/mnt/data1/home/jiangjiajun/workspace/SwiftVLN
BASELINE=${REPO}/baseline/streamvln
PYTHON=/mnt/data1/home/jiangjiajun/miniconda3/envs/streamvln-baseline/bin/python
OUTPUT=${REPO}/output/streamvln-baseline/smoketest/smoke_lite

mkdir -p "${OUTPUT}"

export PATH="/usr/local/cuda-13.0/bin:${PATH}"
export PYTHONPATH="/mnt/data1/home/jiangjiajun/workspace/StreamVLN:/mnt/data1/home/jiangjiajun/workspace/StreamVLN/streamvln:${BASELINE}"
export CUDA_VISIBLE_DEVICES=7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

${PYTHON} -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port=29556 \
  ${BASELINE}/src/train_satnav.py \
  --deepspeed ${BASELINE}/configs/zero2.json \
  --model_name_or_path ${BASELINE}/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3 \
  --version qwen_1_5 \
  --video_folder /mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260306/trajectory_data \
  --group_by_task False \
  --num_history 8 --num_future_steps 4 --num_frames 32 \
  --data_augmentation False \
  --mm_tunable_parts='mm_mlp_adapter' \
  --vision_tower ${BASELINE}/model/siglip-so400m-patch14-384 \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False --mm_use_im_patch_token False \
  --image_aspect_ratio anyres_max_9 \
  --image_grid_pinpoints '(1x1),...,(6x6)' \
  --bf16 True \
  --run_name smoke_lite \
  --output_dir ${OUTPUT} \
  --max_steps 5 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --evaluation_strategy 'no' \
  --save_strategy 'steps' --save_steps 5 --save_total_limit 1 \
  --learning_rate 2e-5 --mm_vision_tower_lr 5e-6 \
  --weight_decay 0.0 --warmup_ratio 0.0 \
  --lr_scheduler_type 'cosine' \
  --logging_steps 1 --tf32 True \
  --model_max_length 4096 \
  --gradient_checkpointing True \
  --dataloader_num_workers 2 \
  --lazy_preprocess True --torch_compile False \
  --dataloader_drop_last True \
  --report_to none \
  2>&1 | tee ${OUTPUT}/smoke.log
```

### Step 2b — full 模式（8卡全参数）

> ⚠️ 必须先 `unset CUDA_VISIBLE_DEVICES`（如果 lite 先跑了，该变量仍为 7，会导致 full 的 rank 1-7 找不到设备）

```bash
unset CUDA_VISIBLE_DEVICES   # 关键：清除 lite 留下的环境变量

REPO=/mnt/data1/home/jiangjiajun/workspace/SwiftVLN
BASELINE=${REPO}/baseline/streamvln
PYTHON=/mnt/data1/home/jiangjiajun/miniconda3/envs/streamvln-baseline/bin/python
OUTPUT=${REPO}/output/streamvln-baseline/smoketest/smoke_full

mkdir -p "${OUTPUT}"

export PATH="/usr/local/cuda-13.0/bin:${PATH}"
export PYTHONPATH="/mnt/data1/home/jiangjiajun/workspace/StreamVLN:/mnt/data1/home/jiangjiajun/workspace/StreamVLN/streamvln:${BASELINE}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

${PYTHON} -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_port=$((RANDOM % 10000 + 20000)) \
  ${BASELINE}/src/train_satnav.py \
  --deepspeed ${BASELINE}/configs/zero2.json \
  --model_name_or_path ${BASELINE}/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3 \
  --version qwen_1_5 \
  --video_folder /mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260306/trajectory_data \
  --group_by_task False \
  --num_history 8 --num_future_steps 4 --num_frames 32 \
  --data_augmentation True \
  --mm_tunable_parts='mm_vision_tower,mm_mlp_adapter,mm_language_model' \
  --vision_tower ${BASELINE}/model/siglip-so400m-patch14-384 \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False --mm_use_im_patch_token False \
  --image_aspect_ratio anyres_max_9 \
  --image_grid_pinpoints '(1x1),...,(6x6)' \
  --bf16 True \
  --run_name smoke_full \
  --output_dir ${OUTPUT} \
  --max_steps 5 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 2 \
  --evaluation_strategy 'no' \
  --save_strategy 'steps' --save_steps 5 --save_total_limit 1 \
  --learning_rate 2e-5 --mm_vision_tower_lr 5e-6 \
  --weight_decay 0.0 --warmup_ratio 0.0 \
  --lr_scheduler_type 'cosine' \
  --logging_steps 1 --tf32 True \
  --model_max_length 32768 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --lazy_preprocess True --torch_compile True \
  --torch_compile_backend inductor \
  --dataloader_drop_last True \
  --report_to none \
  2>&1 | tee ${OUTPUT}/smoke.log
```

---

## Step 3 — Verify Training

Check the log for:

- [ ] `len train_dataset: XXXXX`（数据加载成功，应 > 30000）
- [ ] `{'loss': X.XXXX, ...}` 出现 5 次（每 step 打印一次）
- [ ] `Model saved to ...` 出现（保存成功）
- [ ] 无 `Traceback`、`RuntimeError`、`FAILED` 关键字

```bash
# 替换 $OUTPUT_NAME 为 smoke_lite 或 smoke_full
grep -E "len train_dataset|loss.*grad_norm|Model saved|Traceback|RuntimeError|FAILED" \
  output/streamvln-baseline/smoketest/${OUTPUT_NAME}/smoke.log
```

---

## Step 4 — Prepare Checkpoint for Eval

### lite 模式 — 需要合并 checkpoint

lite 模式只保存了 `mm_projector.bin`，eval 需要完整模型。
通过 symlink 基础模型 + 覆盖微调产物创建合并目录：

```bash
REPO=/mnt/data1/home/jiangjiajun/workspace/SwiftVLN
BASE_MODEL=${REPO}/baseline/streamvln/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3
SMOKE_CKPT=${REPO}/output/streamvln-baseline/smoketest/smoke_lite
MERGED=${REPO}/output/streamvln-baseline/smoketest/smoke_lite_merged

mkdir -p "${MERGED}"

# Symlink all base model files
for f in "${BASE_MODEL}"/*; do
    ln -sf "$f" "${MERGED}/$(basename $f)"
done

# Overwrite with fine-tuned projector and config
rm -f "${MERGED}/config.json" "${MERGED}/mm_projector.bin"
cp "${SMOKE_CKPT}/config.json" "${MERGED}/config.json"
cp "${SMOKE_CKPT}/mm_projector.bin" "${MERGED}/mm_projector.bin"
```

EVAL_CKPT 设为 `output/streamvln-baseline/smoketest/smoke_lite_merged`。

### full 模式 — 直接使用根目录

full 训练保存的完整模型 safetensors 在**根目录** `output/streamvln-baseline/smoketest/smoke_full/`。
子目录 `checkpoint-5/` 是 DeepSpeed 格式，**不可直接用于 eval**。

EVAL_CKPT 设为 `output/streamvln-baseline/smoketest/smoke_full`（根目录，非 checkpoint-5 子目录）。

---

## Step 5 — Run Smoke Eval

使用 `eval_satnav.sh`，限制 10 个 episodes 快速验证：

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate streamvln-baseline
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN

export CUDA_VISIBLE_DEVICES=7
export PATH="/usr/local/cuda-13.0/bin:${PATH}"

# 替换 $EVAL_CKPT 为 Step 4 确定的路径
bash baseline/streamvln/scripts/eval_satnav.sh \
  ${EVAL_CKPT} val_seen 1 10
```

### 已知问题：Vision Tower delay_load

`src/eval_satnav.py` 已包含显式加载 vision tower 的修复代码。
如果遇到 `'SigLipVisionTower' object has no attribute 'vision_tower'`，
检查 `src/eval_satnav.py` 的 `main()` 函数中是否有以下代码（在 `model.to()` 之前）：

```python
vision_tower = model.get_vision_tower()
if not getattr(vision_tower, "is_loaded", False):
    if get_rank() == 0:
        print(f"[INFO] Vision tower not loaded (delay_load). Loading from: {vision_tower.vision_tower_name}")
    vision_tower.load_model()
vision_tower.to(device=args.local_rank, dtype=torch.bfloat16)
```

---

## Step 6 — Verify Eval

Check eval output for:

- [ ] `[Eval] split=val_seen, total=10` 出现（环境和数据加载成功）
- [ ] 进度条跑完 10/10（无中途 crash）
- [ ] `Evaluation Summary` 正常打印（SR / SPL / OS / NE 有输出，无 Python 崩溃）
- [ ] `evaluation_summary.json` 已保存到 results 目录

```bash
cat results/streamvln-baseline/by-path/$(basename ${EVAL_CKPT})/val_seen/evaluation_summary.json
```

**预期结果**（smoke test 只训 5 步，指标不代表模型能力）：

| 模式 | SR | NE | 说明 |
|---|---|---|---|
| lite | 0% | 50~200m | adapter-only 5步，接近预训练模型行为，不触发边界错误 |
| full | 0% | 0.00m / NaN | 全参 5步模型导航失控，触发 camera 边界错误，episode 报 error 但流程完成 |

**full 模式 camera 边界错误说明**：
- 全参数训练 5 步后，模型生成的导航动作会让 agent 走到场景地图边缘
- SatNav 的 `render_image()` 抛出 `Camera view bounds exceed image bounds`
- src/eval_satnav.py 的 `except` 块捕获此错误，episode 记为 `distance_to_goal=inf`
- summary 中 NE 显示为 0.00m（inf 被过滤后均值为空 → NaN → 0）
- **pipeline 本身功能正常**，全部 10 episodes 能跑完、summary 能保存，即视为 PASS

---

## Step 7 — Cleanup

```bash
REPO=/mnt/data1/home/jiangjiajun/workspace/SwiftVLN

# lite 模式
rm -rf ${REPO}/output/streamvln-baseline/smoketest/smoke_lite
rm -rf ${REPO}/output/streamvln-baseline/smoketest/smoke_lite_merged
rm -rf ${REPO}/results/streamvln-baseline/by-path/smoke_lite_merged

# full 模式
rm -rf ${REPO}/output/streamvln-baseline/smoketest/smoke_full
rm -rf ${REPO}/results/streamvln-baseline/by-path/smoke_full
```

---

## Report

| 项目 | 状态 |
|---|---|
| 模式 | lite / full |
| 数据加载 | ✅/❌ |
| 5 步训练完成 | ✅/❌ |
| Loss 正常（无 NaN）| ✅/❌ |
| 模型保存 | ✅/❌ |
| Checkpoint 合并（仅 lite）| ✅/❌/N/A |
| Eval 环境初始化 | ✅/❌ |
| Eval 10 episodes 跑完（无 crash）| ✅/❌ |
| Eval Summary 保存 | ✅/❌ |
| 清理完成 | ✅/❌ |
