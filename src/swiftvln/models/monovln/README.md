# MonoVLN

MonoVLN 是一个基于单轮对话的视觉语言导航系统，使用历史帧压缩技术减少计算开销。与 StreamVLN 的多轮对话不同，MonoVLN 将所有输入（指令 + 历史观察 + 当前观察）整合到单个用户消息中，模型直接输出动作序列。

## Key Features

- **单轮对话**：每个训练/推理样本是一个 user-assistant 对话
- **历史帧压缩**：使用 2D Average Pooling 压缩历史帧，当前帧保持原始分辨率
- **灵活采样**：每个 episode 均匀采样固定数量的训练样本
- **无 KV Cache**：评估时每步独立调用，无需管理 KV cache
- **可配置压缩**：通过 `--compress_stride` 参数调整压缩比例

## Architecture

```
单轮对话格式:
User: "Task: {instruction}. Historical observations: <image> <image> .... Current observation: <image>. Output actions: ↑/←/→/STOP."
Assistant: "↑ ↑ ← → STOP"

数据流:
Dataset.__getitem__()
    ├── 采样历史帧（均匀采样 num_history 帧）
    ├── 获取当前帧（1 帧）
    └── 构建单轮对话
    ↓
Template._encode()
    ├── 历史帧 → <history_image> tokens
    └── 当前帧 → <current_image> tokens
    ↓
Template._post_encode()
    ├── 历史帧 → 2D Avg Pool → 压缩
    └── 当前帧 → 保持原样
    ↓
Model.forward() / Model.generate()
```

## Compression Details

### Default Configuration

- **Stride**: 2 (default)
- **Compression Ratio**: 4:1
- **Example**: 256 tokens/frame → 64 tokens/frame

### Other Stride Options

| Stride | Compression Ratio | Tokens/Frame |
|--------|-------------------|--------------|
| 1 | 1:1 (no compression) | 256 |
| 2 | 4:1 | 64 |
| 3 | 9:1 | 28 |
| 4 | 16:1 | 16 |

## Installation

MonoVLN 使用与 StreamVLN 相同的环境：

```bash
conda activate swift-vln-train  # for training
conda activate swift-vln-eval   # for evaluation
```

## Usage

### Training

```bash
# Single-node training with default stride (2)
bash src/swiftvln/models/monovln/script/train/train_monovln_qwen2_5_vl.sh

# With custom stride and samples per episode
COMPRESS_STRIDE=3 SAMPLES_PER_EPISODE=16 bash src/swiftvln/models/monovln/script/train/train_monovln_qwen2_5_vl.sh
```

Training script parameters:
- `COMPRESS_STRIDE`: Pooling stride (default: 2)
- `SAMPLES_PER_EPISODE`: Number of training samples per episode (default: 8)
- `MODEL_PATH`: Base model path
- `VLN_DATA_PATHS`: Training data paths
- All other parameters similar to StreamVLN

### Evaluation

```bash
# Habitat evaluation (default)
ENV_TYPE=habitat MODEL_PATH=/path/to/checkpoint bash src/swiftvln/models/monovln/script/eval/eval_monovln_qwen2_5_vl_distributed.sh

# SatNav evaluation
ENV_TYPE=satnav MODEL_PATH=/path/to/checkpoint bash src/swiftvln/models/monovln/script/eval/eval_monovln_qwen2_5_vl_distributed.sh

# With custom stride
COMPRESS_STRIDE=3 ENV_TYPE=habitat MODEL_PATH=/path/to/checkpoint bash src/swiftvln/models/monovln/script/eval/eval_monovln_qwen2_5_vl_distributed.sh
```

## Key Differences from StreamVLN

| Feature | StreamVLN | MonoVLN |
|---------|-----------|---------|
| 对话格式 | 多轮对话 | 单轮对话 |
| 历史帧位置 | System prompt | User message |
| 当前帧 | 多帧（窗口内） | 1 帧 |
| KV Cache | 需要管理 | 不需要 |
| 历史帧处理 | 原始分辨率 | 压缩 |
| Dataset 继承 | 独立实现 | 独立实现 |

## Directory Structure

```
monovln/
├── __init__.py              # Model + Template registration
├── model.py                 # MonoVLN Model (no KV cache)
├── template.py              # MonoVLN Template (compression)
├── compressor.py            # 2D Pooling implementation
├── dataset.py               # Independent dataset (single-turn)
├── arguments.py             # Arguments with samples_per_episode
├── trainer.py               # Training entry point
├── eval.py                  # Evaluation entry point
├── evaluator.py             # MonoVLN Evaluator (single-turn)
├── README.md                # This file
├── doc/
│   ├── OVERVIEW.md          # Logic overview
│   └── README_CN.md         # Chinese documentation
└── script/
    ├── train/
    │   └── train_monovln_qwen2_5_vl.sh
    └── eval/
        └── eval_monovln_qwen2_5_vl_distributed.sh
```

## Implementation Details

### Core Components

1. **dataset.py**: 独立的数据集实现
   - 继承 `torch.utils.data.Dataset`
   - 每个 episode 均匀采样 `samples_per_episode` 个训练样本
   - 保证首尾样本（无历史 / STOP 动作）被包含

2. **template.py**: 差异化图像处理
   - `<history_image>`: 历史帧，压缩处理
   - `<current_image>`: 当前帧，保持原样

3. **evaluator.py**: 单轮对话评估
   - 每步独立构建完整 messages
   - 不需要 KV cache 管理

### Episode Sampling

每个 episode 采样 `samples_per_episode` 个时间点：

```python
# samples_per_episode = 8, episode 有 100 个 action
# 采样点: [0, 13, 27, 41, 54, 68, 82, 96]
# 
# - 0: 第一个样本（无历史）
# - 96: 最后一个样本（预测包含 STOP）
```

## Performance Expectations

### Computational Savings

For typical configuration (8 history frames + 1 current frame):

| Component | StreamVLN | MonoVLN (stride=2) | Savings |
|-----------|-----------|---------------------|---------|
| History tokens | 8 × 256 = 2048 | 8 × 64 = 512 | 75% |
| Current tokens | 1 × 256 = 256 | 1 × 256 = 256 | 0% |
| Total | 2304 | 768 | 67% |

### Expected Impact

- **Training Speed**: ~1.5-2x faster
- **GPU Memory**: ~30-40% reduction
- **Simpler inference**: No KV cache management

## Troubleshooting

### Issue: Compression not applied

**Solution**: Make sure you're using `monovln_qwen2_5_vl` model_type:

```bash
--model_type monovln_qwen2_5_vl  # Correct
--model_type streamvln_qwen2_5_vl    # Wrong (no compression)
```

### Issue: Model loading error

**Solution**: Ensure you've registered MonoVLN:

```bash
--custom_register_path src/swiftvln/models/monovln
```

### Issue: Different results between training and eval

**Solution**: Use the same `compress_stride` value:

```bash
# Training
COMPRESS_STRIDE=2 bash train_script.sh

# Evaluation  
COMPRESS_STRIDE=2 bash eval_script.sh
```

## Citation

If you use MonoVLN in your research, please cite:

```bibtex
@inproceedings{monovln2026,
  title={MonoVLN: Single-Turn Visual Language Navigation with History Compression},
  author={Your Name},
  year={2026}
}
```

## Acknowledgments

MonoVLN is built on top of:
- [StreamVLN](../streamvln/): Base VLN concepts
- [ms-swift](https://github.com/modelscope/ms-swift): Training framework
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2-VL): Vision-language model
