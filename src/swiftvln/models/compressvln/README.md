# CompressVLN

CompressVLN extends StreamVLN with history frame compression using 2D Average Pooling. This reduces computational cost during training and inference while maintaining model performance.

## Key Features

- **History Frame Compression**: Compresses history frames (in System prompt) using 2D Average Pooling
- **Current Frame Preservation**: Keeps current frames (in User turns) at full resolution
- **No Additional Parameters**: Uses parameter-free pooling - fully compatible with StreamVLN weights
- **Configurable Compression**: Adjustable compression ratio via `--compress_stride` parameter
- **Multi-Environment Support**: Works with both Habitat (indoor) and SatNav (satellite map) environments

## Architecture

```
Dataset (same as StreamVLN)
    ↓
Template._encode() (standard Qwen2.5-VL)
    ↓
Template._post_encode() [NEW: Compression applied here]
    │
    ├─ History frames (System prompt) → 2D Avg Pool → Compressed
    └─ Current frames (User turns) → No compression
    ↓
Model.forward() (same as StreamVLN)
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

CompressVLN uses the same environment as StreamVLN:

```bash
conda activate swift-vln-train  # for training
conda activate swift-vln-eval   # for evaluation
```

## Usage

### Training

```bash
# Single-node training with default stride (2)
bash src/swiftvln/models/compressvln/script/train/train_compressvln_qwen2_5_vl.sh

# With custom stride
COMPRESS_STRIDE=3 bash src/swiftvln/models/compressvln/script/train/train_compressvln_qwen2_5_vl.sh
```

Training script parameters:
- `COMPRESS_STRIDE`: Pooling stride (default: 2)
- `MODEL_PATH`: Base model path
- `VLN_DATA_PATHS`: Training data paths
- All other parameters same as StreamVLN

### Evaluation

```bash
# Habitat evaluation (default)
ENV_TYPE=habitat MODEL_PATH=/path/to/checkpoint bash src/swiftvln/models/compressvln/script/eval/eval_compressvln_qwen2_5_vl_distributed.sh

# SatNav evaluation
ENV_TYPE=satnav MODEL_PATH=/path/to/checkpoint bash src/swiftvln/models/compressvln/script/eval/eval_compressvln_qwen2_5_vl_distributed.sh

# With custom stride
COMPRESS_STRIDE=3 ENV_TYPE=habitat MODEL_PATH=/path/to/checkpoint bash src/swiftvln/models/compressvln/script/eval/eval_compressvln_qwen2_5_vl_distributed.sh
```

**Note**: CompressVLN reuses StreamVLN's evaluator. Compression is automatically applied by the template.

## Model Compatibility

CompressVLN models are **fully compatible** with StreamVLN:

| Scenario | Supported | Notes |
|----------|-----------|-------|
| Train with CompressVLN, Eval with CompressVLN | ✅ Yes | Normal usage |
| Train with StreamVLN, Eval with CompressVLN | ✅ Yes | Compression applied at eval time |
| Train with CompressVLN, Eval with StreamVLN | ✅ Yes | No compression at eval time |
| Load CompressVLN checkpoint in StreamVLN | ✅ Yes | Same model architecture |

This is possible because:
1. CompressVLN uses the same model architecture as StreamVLN (no new parameters)
2. Compression is implemented purely in the template (data preprocessing)
3. The model only sees the final compressed/uncompressed embeddings

## Directory Structure

```
compressvln/
├── __init__.py              # Model + Template registration
├── template.py              # CompressVLN Template (core)
├── compressor.py            # 2D Pooling implementation
├── arguments.py             # Extended arguments with --compress_stride
├── trainer.py               # Training entry point
├── README.md                # This file
└── script/
    ├── train/
    │   └── train_compressvln_qwen2_5_vl.sh
    └── eval/
        └── eval_compressvln_qwen2_5_vl_distributed.sh
```

## Implementation Details

### Core Components

1. **compressor.py**: Implements 2D Average Pooling
   - Input: `[num_tokens, hidden_size]` + `grid_thw` 
   - Output: `[compressed_tokens, hidden_size]`
   - Method: Reshape to 2D → avg_pool2d → Flatten

2. **template.py**: Extends `Qwen2VLTemplate`
   - Overrides `_post_encode()` to apply compression
   - Identifies history vs current frames
   - Rebuilds sequence with new length

3. **arguments.py**: Adds `compress_stride` parameter
   - Extends `StreamVLNTrainArguments`
   - Default: stride=2

### History Frame Identification

CompressVLN determines which frames are history based on the dataset structure:

- **History Frames**: Images in the System prompt (first N frames)
- **Current Frames**: Images in User turns (remaining frames)

In StreamVLN dataset format:
```python
{
    'messages': [
        {'role': 'system', 'content': 'Task... <image> <image> ...'},  # History
        {'role': 'user', 'content': 'you can see <image>.'},           # Current
        {'role': 'assistant', 'content': '↑↑→'},
        ...
    ]
}
```

## Performance Expectations

### Computational Savings

For typical StreamVLN configuration (8 history frames + 1 current frame):

| Component | StreamVLN | CompressVLN (stride=2) | Savings |
|-----------|-----------|------------------------|---------|
| History tokens | 8 × 256 = 2048 | 8 × 64 = 512 | 75% |
| Current tokens | 1 × 256 = 256 | 1 × 256 = 256 | 0% |
| Total | 2304 | 768 | 67% |

### Expected Impact

- **Training Speed**: ~1.5-2x faster (due to reduced sequence length)
- **GPU Memory**: ~30-40% reduction
- **Navigation Performance**: Minimal impact (history still represented)

## Troubleshooting

### Issue: Compression not applied

**Solution**: Make sure you're using `compressvln_qwen2_5_vl` model_type:

```bash
--model_type compressvln_qwen2_5_vl  # Correct
--model_type streamvln_qwen2_5_vl    # Wrong (no compression)
```

### Issue: Model loading error

**Solution**: Ensure you've registered CompressVLN:

```bash
--custom_register_path src/swiftvln/models/compressvln
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

If you use CompressVLN in your research, please cite:

```bibtex
@inproceedings{compressvln2026,
  title={CompressVLN: Efficient Visual Language Navigation with History Frame Compression},
  author={Your Name},
  year={2026}
}
```

## Acknowledgments

CompressVLN is built on top of:
- [StreamVLN](../streamvln/): Base VLN model
- [ms-swift](https://github.com/modelscope/ms-swift): Training framework
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2-VL): Vision-language model
