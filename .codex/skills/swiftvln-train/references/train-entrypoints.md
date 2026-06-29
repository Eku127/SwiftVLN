# SwiftVLN Train Entrypoints

## Core Scripts

- Queue mode: `src/swiftvln/scripts/train/train_queue.sh`
- SwiftVLN single run: `src/swiftvln/model/script/train/train_swiftvln_qwen2_5_vl.sh`

## Important Variables in `train_swiftvln_qwen2_5_vl.sh`

- Conda env: `conda activate swift-vln-train-update`
- Model family: `MODEL_FAMILY="qwen2_5_vl"` or `"qwen3_vl"`
- Model type: auto-derived from `MODEL_FAMILY`, override with `MODEL_TYPE` only when needed
- Base model: override with `BASE_MODEL_PATH` or `MODEL_PATH`
- Environment selector: `VLN_ENV_TYPE="habitat"` or `"satnav"`
- SatNav data list: `SATNAV_DATA_PATHS=(...)`
- Completion signal in logs: `Model saved to: <output_dir>`

## Queue Script Capability

- Supported models: `swiftvln`
- Can patch per-run params into temporary scripts and run serially
- Writes logs under `logs/` and summary files under `logs/train_queue_results/`
