# OverlapVLN Train Entrypoints

## Core Scripts

- Queue mode: `src/swiftvln/scripts/train/train_queue.sh`
- OverlapVLN single run: `src/swiftvln/model/script/train/train_overlapvln_qwen2_5_vl.sh`

## Important Variables in `train_overlapvln_qwen2_5_vl.sh`

- Conda env: `conda activate swift-vln-train`
- Model type: `MODEL_TYPE="overlapvln_qwen2_5_vl"`
- Environment selector: `VLN_ENV_TYPE="habitat"` or `"satnav"`
- SatNav data list: `SATNAV_DATA_PATHS=(...)`
- QA path: `QA_DATASET=.../qa_swift.jsonl`
- Completion signal in logs: `Model saved to: <output_dir>`

## Queue Script Capability

- Supported models: `overlapvln`
- Can patch per-run params into temporary scripts and run serially
- Writes logs under `logs/` and summary files under `logs/train_queue_results/`
