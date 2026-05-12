# SwiftVLN Environment Setup

This repository uses the `ms-swift` 4.x API. Do not use the old
`swift-vln-train` / `swift-vln-eval` environments for current SwiftVLN work.

## Train Environment

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda create -n swift-vln-train-update --clone swift-vln-train
conda activate swift-vln-train-update

cd /mnt/data1/home/jiangjiajun/workspace/ms-swift-lateset
pip install -e .

pip install "transformers>=4.57,<5.0" "qwen-vl-utils>=0.0.14" decord -U

cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN-refactor
pip install -e .
```

Expected key versions:

```bash
python - <<'PY'
import importlib.metadata as m
for p in ["ms-swift", "transformers", "qwen-vl-utils", "decord", "flash-attn", "deepspeed"]:
    print(p, m.version(p))
PY
```

Current validated baseline:

```text
ms-swift 4.2.0.dev0
transformers 4.57.3
qwen-vl-utils 0.0.14
```

## Eval Environment

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda create -n swift-vln-eval-update --clone swift-vln-eval
conda activate swift-vln-eval-update

cd /mnt/data1/home/jiangjiajun/workspace/ms-swift-lateset
pip install -e .

pip install "transformers>=4.57,<5.0" "qwen-vl-utils>=0.0.14" decord -U

cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN-refactor
pip install -e .
```

## Model Cache

Qwen2.5-VL default:

```text
/mnt/data1/home/jiangjiajun/.cache/modelscope/models/Qwen/Qwen2___5-VL-3B-Instruct
```

Qwen3-VL default after the 2B model is fully cached:

```text
/mnt/data1/home/jiangjiajun/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct
```

Current validated Qwen3-VL 8B smoke path on 73:

```text
/mnt/data1/home/jiangjiajun/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct
```

## Smoke Commands

Qwen2.5-VL train smoke:

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate swift-vln-train-update
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN-refactor

MODEL_FAMILY=qwen2_5_vl VLN_ENV_TYPE=satnav \
MAX_SAMPLES=16 MAX_STEPS=2 SAVE_STEPS=1 SAVE_TOTAL_LIMIT=1 \
USE_SWANLAB=false USE_WXWORK_NOTIFICATION=false TRAIN_NUM_GPUS=2 \
bash src/swiftvln/model/script/train/train_swiftvln_qwen2_5_vl.sh
```

Qwen3-VL 8B train smoke:

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate swift-vln-train-update
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN-refactor

MODEL_FAMILY=qwen3_vl \
STAGE1_MODEL_PATH=/mnt/data1/home/jiangjiajun/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct \
VLN_ENV_TYPE=satnav NUM_FRAMES=8 NUM_HISTORY=2 NUM_FUTURE_STEPS=2 \
MAX_SAMPLES=8 MAX_STEPS=1 SAVE_STEPS=1 SAVE_TOTAL_LIMIT=1 \
USE_SWANLAB=false USE_WXWORK_NOTIFICATION=false TRAIN_NUM_GPUS=8 \
bash src/swiftvln/model/script/train/train_swiftvln_qwen2_5_vl.sh
```

Eval smoke for either model:

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate swift-vln-eval-update
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN-refactor

MODEL_FAMILY=<qwen2_5_vl_or_qwen3_vl> ENV_TYPE=satnav EVAL_SPLIT=val_seen \
MAX_EPISODES=2 CUDA_DEVICES=0,1 \
bash src/swiftvln/scripts/eval/eval_by_name.sh <swiftvln_exp_name>
```

For the Qwen3-VL 8B smoke above, keep eval parameters aligned with training:

```bash
MODEL_FAMILY=qwen3_vl ENV_TYPE=satnav EVAL_SPLIT=val_seen \
NUM_FRAMES=8 NUM_HISTORY=2 NUM_FUTURE_STEPS=2 MAX_EPISODES=1 CUDA_DEVICES=0,1 \
MODEL_PATH=<checkpoint_path> \
bash src/swiftvln/model/script/eval/eval_swiftvln_qwen2_5_vl_distributed.sh
```

Qwen3.5 is not supported by these update environments. Use a separate
Qwen3.5 environment with a Transformers build that provides
`transformers.models.qwen3_5`.
