# S2R Stage-A 训练

S2R Stage-A 使用 UAV–Satellite 配对图像训练 sim-to-real adapter。Qwen2.5-VL
视觉塔在训练期间保持冻结，adapter 仅作用于 UAV 视觉 token，使其与对应的卫星图像
特征对齐。

| 组件 | Stage-A 中的作用 |
| --- | --- |
| Qwen2.5-VL 视觉塔 | 同时编码 UAV 与 Satellite 图像，参数冻结 |
| S2R adapter | 变换 UAV 视觉 token，参与训练 |
| Projection head | 计算对比学习与 retrieval 指标，参与训练 |

训练目标为双向 UAV–Satellite 对比损失与全局 cosine alignment loss 之和。Retrieval
评测使用共享的 projection head 计算配对图像相似度。

## 1. 准备环境与 Teacher

按照[安装](../getting-started/INSTALLATION.md)创建 `swiftvln-train` 环境，并完成
[模型与 Checkpoint](../getting-started/CHECKPOINTS.md)中的默认 SatNav 模型下载。

```bash
cd /path/to/SwiftVLN
export SWIFTVLN_ROOT="${PWD}"
source .local/env.sh
source "${SWIFTVLN_CONDA_SH}"
conda activate swiftvln-train

python -m pip install -e ".[s2r-data]"
```

默认实验使用已训练的 SwiftVLN SatNav 模型作为冻结 teacher：

```bash
export TEACHER_MODEL_PATH="${SWIFTVLN_SATNAV_MODEL_PATH}"
```

## 2. 准备 SatDronePair

Stage-A 使用以下四个数据源生成 SatDronePair：

- DenseUAV
- GTA-UAV
- SUES-200
- UAV-VisLoc

上游数据下载、转换命令与图像质量检查见
[SatDronePair 数据生产](../data/SATDRONEPAIR.md)。转换后的目录为：

```text
SatDronePair/
├── denseuav/{drone,satellite,pairs.csv,dataset_info.json}
├── gta/{drone,satellite,pairs.csv,dataset_info.json}
├── sues/{drone,satellite,pairs.csv,dataset_info.json}
└── uavvisloc/{drone,satellite,pairs.csv,dataset_info.json}
```

## 3. 构建 Manifest

设置数据与 manifest 路径：

```bash
export PAIR_ROOT=/path/to/SatDronePair
export MANIFEST_PATH="${SWIFTVLN_ROOT}/runtime/s2r/manifests/manifest_v1.jsonl"
```

构建训练与验证 split：

```bash
python -m tools.s2r.scripts.build_manifest \
  --data_root "${PAIR_ROOT}" \
  --output_path "${MANIFEST_PATH}" \
  --val_ratio 0.1 \
  --seed 42 \
  --skip_missing false
```

Manifest 按位置划分 `train` 与 `val`，同一地点的配对图像不会跨 split：

| 数据源 | Split 分组 |
| --- | --- |
| DenseUAV | 基础位置 ID |
| GTA-UAV | Satellite tile |
| SUES-200 | Scene ID |
| UAV-VisLoc | Sequence ID |

命令完成后会输出 manifest 总记录数，以及按数据源和 split 汇总的记录数。

## 4. 训练 Stage-A Adapter

### 4.1 默认训练配置

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--batch_size` | `8` | 每张 GPU 的配对图像数量 |
| `--epochs` | `10` | 训练轮数 |
| `--learning_rate` | `1e-4` | Adapter 与 projection head 的 learning rate |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--warmup_ratio` | `0.05` | Linear warmup 占总步数的比例 |
| `--grad_accum_steps` | `1` | 梯度累积步数 |
| `--temperature` | `0.07` | 双向对比损失温度 |
| `--adapter_layers` | `2` | Transformer adapter 层数 |
| `--adapter_heads` | `8` | Adapter attention heads |
| `--adapter_mlp_ratio` | `4.0` | Adapter MLP expansion ratio |
| `--projection_dim` | `512` | Retrieval projection 维度 |
| `--teacher_dtype` | `auto` | CUDA 上使用 BF16，CPU 上使用 FP32 |


### 4.2 单卡训练

`train_s2r_stagea.sh` 使用一张 GPU，并自动加载 `.local/env.sh` 与
`swiftvln-train` 环境：

```bash
export OUTPUT_DIR="${SWIFTVLN_ROOT}/output/s2r/s2r-stagea-swiftvln-3b-10ep-bs8-lr1e-4"

bash scripts/train/train_s2r_stagea.sh
```

### 4.3 多卡训练

使用 `torchrun` 启动多卡训练。以下配置使用 8 张 GPU，有效 batch size 为 64：

```bash
export OUTPUT_DIR="${SWIFTVLN_ROOT}/output/s2r/s2r-stagea-swiftvln-3b-10ep-bs64-lr1e-4"

torchrun --standalone --nproc_per_node=8 -m tools.s2r.trainer \
  --manifest_path "${MANIFEST_PATH}" \
  --teacher_model_path "${TEACHER_MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}"
```

Stage-A 的双向对比损失会汇总全部 GPU 的图像特征，并保留跨 rank 的梯度。

## 5. 输出与 Checkpoint

训练输出保存在 `OUTPUT_DIR`：

```text
<OUTPUT_DIR>/
├── best.pt
├── latest.pt
├── train_args.json
├── progress.json
├── metrics.jsonl
└── checkpoints/
    └── step_XXXXXXX.pt
```

| 文件 | 内容 |
| --- | --- |
| `best.pt` | `val` split 上 `U2S R@1` 最高的 checkpoint |
| `latest.pt` | 最近一次保存的 checkpoint |
| `train_args.json` | 本次训练参数 |
| `progress.json` | 当前步数、训练状态与最终指标 |
| `metrics.jsonl` | 每次 retrieval 评测的指标 |

默认在训练结束时执行一次 `val` retrieval 评测。设置 `--eval_every_steps <N>` 可按步数
评测并更新 `best.pt`；设置 `--save_every_steps <N>` 可额外保存周期 checkpoint。

加载已有 Stage-A 权重继续训练：

```bash
OUTPUT_DIR="${SWIFTVLN_ROOT}/output/s2r/<new-run-name>" \
MANIFEST_PATH="${MANIFEST_PATH}" \
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH}" \
bash scripts/train/train_s2r_stagea.sh \
  --resume_checkpoint /path/to/stage-a/latest.pt
```

`resume_checkpoint` 加载 adapter、projection head、global step 与 best metric；optimizer、
scheduler 和随机数状态会重新初始化。

## 6. Retrieval 评测

使用 `best.pt` 在 `val` split 上执行检索评测：

```bash
python -m tools.s2r.eval \
  --manifest_path "${MANIFEST_PATH}" \
  --checkpoint_path "${OUTPUT_DIR}/best.pt" \
  --split val \
  --batch_size 8 \
  --num_workers 4 \
  | tee "${OUTPUT_DIR}/eval_val.json"
```

Teacher 路径保存在 Stage-A checkpoint 中。需要切换本地模型目录时，增加
`--teacher_model_path /path/to/teacher`。

| 指标 | 含义 |
| --- | --- |
| `u2s_r@1/5/10` | 以 UAV 图像检索对应 Satellite 图像的 Recall@K |
| `s2u_r@1/5/10` | 以 Satellite 图像检索对应 UAV 图像的 Recall@K |
| `paired_cosine_mean` | 成对图像投影特征的平均 cosine similarity |
| `by_source` | DenseUAV、GTA-UAV、SUES-200 与 UAV-VisLoc 的分数据源指标 |
