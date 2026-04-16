# OpenFly SatNav Config History

## Previous Config

### Train

- `OPENFLY_BACKEND=hf`
- `OPENFLY_ACTION_FORMAT=original`
- `DATA_PATH=/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260404/trajectory_data/annotations.json`
- `IMAGE_FOLDER=/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260404/trajectory_data`
- `NUM_GPUS=8`
- `TRAIN_BSZ=12`
- `GRAD_ACCUM=1`
- `NUM_EPOCHS=1`
- `LEARNING_RATE=2e-5`
- `SAVE_STEPS=10000`
- `LOGGING_STEPS=20`
- `TORCH_DTYPE=bfloat16`
- `DEEPSPEED_MODE=zero2`
- `USE_FLASH_ATTENTION_2=true`
- `GRID_SIZE=16`
- `WEIGHT_DECAY=0.0`
- `WARMUP_RATIO=0.0`
- `LR_SCHEDULER_TYPE=linear`
- `MAX_GRAD_NORM=1.0`
- `DATALOADER_NUM_WORKERS=4`
- `REPORT_TO=none`

### Sampling

- `SATNAV_HEAD_KEEP=7`
- `SATNAV_SAMPLE_STRIDE=7`
- `SATNAV_STOP_REPEAT=6`
- `SATNAV_STOP_HISTORY_AUG=3`

### Original Supervision

- supervise only the first `4` active action-dimension tokens
- inactive trailing action tokens and final `eos` are masked from loss
- no per-dimension loss reweighting

## Current Config

### Train

- `OPENFLY_BACKEND=hf`
- `OPENFLY_ACTION_FORMAT=original`
- `DATA_PATH=/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260404/trajectory_data/annotations.json`
- `IMAGE_FOLDER=/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260404/trajectory_data`
- `NUM_GPUS=8`
- `TRAIN_BSZ=12`
- `GRAD_ACCUM=1`
- `NUM_EPOCHS=1`
- `LEARNING_RATE=2e-5`
- `SAVE_STEPS=10000`
- `LOGGING_STEPS=20`
- `TORCH_DTYPE=bfloat16`
- `DEEPSPEED_MODE=zero2`
- `USE_FLASH_ATTENTION_2=true`
- `GRID_SIZE=16`
- `WEIGHT_DECAY=0.0`
- `WARMUP_RATIO=0.0`
- `LR_SCHEDULER_TYPE=linear`
- `MAX_GRAD_NORM=1.0`
- `DATALOADER_NUM_WORKERS=4`
- `REPORT_TO=none`

### Sampling

- `SATNAV_HEAD_KEEP=7`
- `SATNAV_SAMPLE_STRIDE=7`
- `SATNAV_STOP_REPEAT=4`
- `SATNAV_STOP_HISTORY_AUG=1`

### Original Supervision

- supervise only the first `4` active action-dimension tokens
- inactive trailing action tokens and final `eos` are masked from loss
- weighted CE over the first `4` action-dimension tokens:
  - `OPENFLY_ORIGINAL_DIM_LOSS_WEIGHTS=0.4,1.2,1.2,1.2`
