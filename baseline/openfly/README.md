# OpenFly Baseline

SatNav-only baseline integration for OpenFly.

## Layout

- `scripts/train_satnav.sh`: launch training on SatNav trajectory data
- `scripts/eval_satnav.sh`: evaluate on SatNav `val_seen` / `val_unseen`
- `scripts/setup_env.sh`: create `openfly-baseline` conda env
- `src/dataset/satnav_dataset.py`: SatNav step-wise dataset loader
- `src/train_satnav.py`: Transformers Trainer entrypoint
- `src/eval_satnav.py`: SatNav evaluator
- `src/prompting.py`: prompt builder with action-history support

## Evaluation Results (val_seen, ver_260404)

| Config | Ckpt | Overall SR | Boundary SR | Road SR | LandmarkSet SR | OS |
|---|---|---|---|---|---|---|
| original format, stop_window=2, no hist | 33430 | 0.0% | 0.0% | — | — | 18.8% |
| **compact + stop_window=0 + hist16** | **8000** | **12.3%** | **17.4%** | **18.1%** | **5.2%** | **23.3%** |
| compact + stop_window=0 + hist16 | 6000 | 11.9% | 15.3% | 18.2% | 5.6% | 22.8% |

The key recipe change that broke the SR=0 deadlock:
1. Switch action format from `original` to `compact` (removes 8D vector noise)
2. Set `SATNAV_STOP_WINDOW=0` (teach stop only at the true trajectory end)
3. Add action history to prompt (`OPENFLY_ACTION_HISTORY_LIMIT=16`)

## Notes

- Prompt format: `What action should the robot take to ...? Past actions: forward, forward, left, ...`
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
  - `-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16`
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
- Current default sample policy is global and training-only:
  - `SATNAV_HEAD_KEEP=7`
  - `SATNAV_SAMPLE_STRIDE=3`
  - `SATNAV_STOP_REPEAT=2`
  - `SATNAV_STOP_WINDOW=0`  ← disabled to avoid premature-stop bias
  - `SATNAV_TAIL_KEEP=5`
  - `SATNAV_STOP_HISTORY_AUG=1`
  - semantics:
    - keep the first `7` steps
    - keep all turns
    - keep every 3rd forward inside each consecutive forward run
    - always preserve a near-goal tail before the final stop
    - stop window disabled: only the true final step is supervised as `stop`
    - keep stop-history augmentation at `1x`
  - On 0404 with `stop_window=0` this yields about `3.825M` samples:
    - `stop=209,908` (`5.49%`)
    - `forward=2,212,744` (`57.85%`)
    - `left=730,805` (`19.10%`)
    - `right=671,062` (`17.54%`)
  - With the latest measured 8xH100 throughput on 98, 1 epoch is about `8.8h`
    and should be budgeted as roughly `9.0-9.5h` including save overhead
- Action history prompt (new in v2):
  - `OPENFLY_ACTION_HISTORY_LIMIT=16` controls how many past actions appear in the prompt
  - Past actions are appended as: `Past actions: forward, forward, left, ...`
  - If no actions have been taken yet, the clause is omitted
  - Both train and eval use the same limit; set `OPENFLY_ACTION_HISTORY_LIMIT=0` to disable
- `original` action-token supervision is trimmed to the first 4 active action-dimension tokens, and those
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
  - `SATNAV_STOP_WINDOW`
  - `SATNAV_TAIL_KEEP`
  - `SATNAV_STOP_HISTORY_AUG`
  - `OPENFLY_ACTION_HISTORY_LIMIT`
  - `OPENFLY_ORIGINAL_DIM_LOSS_WEIGHTS`
- Eval outputs record explicit action fields in `result.jsonl`:
  - `action`: final action id
  - `parsed_action`: final action name
  - `generated_text`: final decoded output
  - `action_trace`: per-step action ids / names / raw outputs
- Eval is robust to simulator out-of-bounds errors; such steps are recorded as episode failures
  rather than crashing the full evaluation run
- This baseline does not depend on the external `OpenFly-Platform` repo at runtime.
- Native backend writes `backend_meta.json` alongside training outputs and checkpoints so the checkpoint source stays traceable.
