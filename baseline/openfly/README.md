# OpenFly Baseline

SatNav-only baseline integration for OpenFly.

## Layout

- `scripts/train_satnav.sh`: launch training on SatNav trajectory data
- `scripts/eval_satnav.sh`: evaluate on SatNav `val_seen` / `val_unseen`
- `scripts/setup_env.sh`: create `openfly-baseline` conda env
- `src/dataset/satnav_dataset.py`: SatNav step-wise dataset loader
- `src/train_satnav.py`: Transformers Trainer entrypoint
- `src/eval_satnav.py`: SatNav evaluator

## Notes

- Prompt format keeps OpenFly's original query style:
  `What action should the robot take to ...?`
- Supported action formats:
  - `compact`: 4 text actions `stop / forward / left / right`
  - `original`: OpenFly-style action tokens over an 8D action vector
- Supported backends:
  - `hf`: load an existing HF OpenFly checkpoint directory
  - `native`: initialize from a Prismatic/OpenVLA `.pt` checkpoint, then continue training in the current HF/Trainer pipeline
- Select the backend with `OPENFLY_BACKEND=hf|native`
- `native` currently only supports `OPENFLY_ACTION_FORMAT=original`
- Native backend defaults:
  - `MODEL_PATH=baseline/openfly/model/openvlaopenvla-7b-prismatic`
  - `PROCESSOR_PATH=baseline/openfly/model/openfly-agent-7b`
  - The processor/tokenizer/image preprocessor come from `PROCESSOR_PATH`
  - The model weights come from the native Prismatic checkpoint under `MODEL_PATH`
- Native backend output is still a normal HF checkpoint layout under `output/openfly-baseline/.../checkpoint-*`
  so `scripts/eval_satnav.sh` keeps reusing the existing HF eval path
- `original` SatNav templates only enable 4 legal actions:
  - `stop -> [1, 0, 0, 0, 0, 0, 0, 0]`
  - `forward -> [0, 10, 0, 0, 0, 0, 0, 0]`
  - `left -> [0, 0, 15, 0, 0, 0, 0, 0]`
  - `right -> [0, 0, 0, 15, 0, 0, 0, 0]`
- Select the mode with `OPENFLY_ACTION_FORMAT=compact|original`
- Default experiment names always include the action mode suffix:
  - `-actcompact`
  - `-actoriginal`
- Experiment names also include the backend suffix:
  - `-bkhf`
  - `-bknative`
- Default train names also include the SatNav sampling tag:
  - `-sample-hk3-fs2-stopx5`
- Training defaults do not use Weights & Biases:
  - `--report_to none`
  - `WANDB_DISABLED=true`
  - `WANDB_MODE=disabled`
- Current default train config is tuned for 8xH100 SatNav runs:
  - `TRAIN_BSZ=12`
  - `GRAD_ACCUM=1`
  - `TORCH_DTYPE=bfloat16`
  - `USE_FLASH_ATTENTION_2=true`
  - `LEARNING_RATE=2e-5`
  - `SAVE_STEPS=5000`
  - `LR_SCHEDULER_TYPE=linear`
  - `WEIGHT_DECAY=0.0`
- Current default sample policy is a forward-reduced, light stop-boost variant:
  - `SATNAV_HEAD_KEEP=3`
  - `SATNAV_SAMPLE_STRIDE=2`
  - `SATNAV_STOP_REPEAT=5`
  - Keep all turns, keep every 2nd forward inside each consecutive forward run, and repeat stop samples 5x
  - On 0404 this keeps the first 3 steps, cuts forward to roughly half, and keeps stop slightly below single-class left/right volume
- `train_satnav.sh` also exposes system / optimizer knobs for H100 tuning:
  - `USE_FLASH_ATTENTION_2=true|false`
  - `WEIGHT_DECAY`
  - `WARMUP_RATIO`
  - `LR_SCHEDULER_TYPE`
  - `MAX_GRAD_NORM`
  - `DATALOADER_NUM_WORKERS`
  - `REPORT_TO`
  - `SATNAV_HEAD_KEEP`
  - `SATNAV_SAMPLE_STRIDE`
  - `SATNAV_STOP_REPEAT`
- Eval outputs now record explicit action fields in `result.jsonl`:
  - `action`: final action id
  - `parsed_action`: final action name
  - `generated_text`: final decoded output
  - `action_trace`: per-step action ids / names / raw outputs
- This baseline does not depend on the external `OpenFly-Platform` repo at runtime.
- Native backend writes `backend_meta.json` alongside training outputs and checkpoints so the checkpoint source stays traceable.
