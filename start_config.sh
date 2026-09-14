cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN

source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate swift-vln-train-update

# 防止当前 shell 遗留旧的 Qwen3 恢复训练参数
unset RESUME_FROM_CHECKPOINT
unset RESUME_ONLY_MODEL

# 环境与 GPU
export SWIFTVLN_TRAIN_CONDA_ENV=swift-vln-train-update
export TRAIN_CUDA_DEVICES=1,2,3,4,5,6,7
export MASTER_PORT=29625

# 使用 Qwen2.5-VL-3B 初始权重
export MODEL_FAMILY=qwen2_5_vl
export MODEL_TYPE=swiftvln_qwen2_5_vl
export MODEL_PATH=/mnt/data1/home/jiangjiajun/.cache/modelscope/models/Qwen/Qwen2___5-VL-3B-Instruct

# 原语动作组数据
export VLN_ENV_TYPE=satnav
export VLN_DATA_PATH=/mnt/data1/home/jiangjiajun/workspace/SatNav/Primitive/output/swiftvln_move_along_val_seen_along_v2

# 每个动作组独立；history 仅保留该动作组内部的历史帧
export MEMORY_METHOD=history
export HISTORY_PROCESSOR_TYPE=per_frame
export NUM_FRAMES=8
export NUM_HISTORY=8
export NUM_FUTURE_STEPS=1
export NUM_OVERLAP=0
export COMPRESS_STRIDE=2
export LOG_BASE=1.0
export USE_RANDOM=false
export SYSTEM_PROMPT_SETTING=vanilla
export EMBEDDING_MODE=none

# 训练与优化
export NUM_EPOCHS=15
export BATCH_SIZE=4
export GRAD_ACCUM_STEPS=1
export LEARNING_RATE=1e-5
export LR_SCHEDULER_KWARGS='{"min_lr":2e-6}'
export USE_DEEPSPEED=true
export DEEPSPEED_CONFIG=zero2
export GRADIENT_CHECKPOINTING=true
export USE_LIGER_KERNEL=true
export ATTN_IMPL=flash_attn
export TF32=true

# 保存与日志
export SAVE_STEPS=500
export SAVE_TOTAL_LIMIT=3
export LOGGING_STEPS=10
export USE_SWANLAB=false

# 训练后单独评测，避免训练结束后立即长时间生成评测结果
export PRIMITIVE_METRICS_ENABLED=false

# 与 Qwen3 实验完全隔离
export OUTPUT_DIR_OVERRIDE=/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/swiftvln/primitive_along

bash scripts/train/train_swiftvln_qwen_vl.sh