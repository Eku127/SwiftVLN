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
  - `-sample-hk7-fs7-stopx4-stoph1`
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
  - `SAVE_STEPS=10000`
  - `LR_SCHEDULER_TYPE=linear`
  - `WEIGHT_DECAY=0.0`
- Current default sample policy is a NaVILA-style forward-reduced, stop-augmented variant:
  - `SATNAV_HEAD_KEEP=7`
  - `SATNAV_SAMPLE_STRIDE=7`
  - `SATNAV_STOP_REPEAT=4`
  - `SATNAV_STOP_HISTORY_AUG=1`
  - Keep all turns, keep every 7th forward inside each consecutive forward run, repeat terminal stop samples 4x,
    and no longer widen stop supervision across multiple terminal-history variants by default
  - On 0404 this yields about `3.209M` samples:
    - `stop=419,816` (`13.08%`)
    - `forward=1,387,530` (`43.24%`)
    - `left=730,805` (`22.77%`)
    - `right=671,062` (`20.91%`)
  - Relative to full raw `5.396M` samples, this is about `-40.5%`, with 8xH100 1 epoch roughly `10.0h`
- `original` action-token supervision is now trimmed to the first 4 active action-dimension tokens, and those
  4 positions use a weighted CE profile by default:
  - `OPENFLY_ORIGINAL_DIM_LOSS_WEIGHTS=0.4,1.2,1.2,1.2`
  - This down-weights the first action token and up-weights the later three, reducing the structural bias where
    `stop` is decided by the first supervised token position
  - The trailing invariant inactive-dimension tokens and final `eos` no longer contribute to CE loss
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
  - `SATNAV_STOP_HISTORY_AUG`
  - `OPENFLY_ORIGINAL_DIM_LOSS_WEIGHTS`
- Eval outputs now record explicit action fields in `result.jsonl`:
  - `action`: final action id
  - `parsed_action`: final action name
  - `generated_text`: final decoded output
  - `action_trace`: per-step action ids / names / raw outputs
- This baseline does not depend on the external `OpenFly-Platform` repo at runtime.
- Native backend writes `backend_meta.json` alongside training outputs and checkpoints so the checkpoint source stays traceable.
