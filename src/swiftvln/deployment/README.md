# OverlapVLN Deployment

This package provides a local deployment entrypoint for the `overlapvln` baseline.
The current implementation is a single-process, single-session JSONL stdin/stdout server.
By default it auto-selects one idle local H100. If `CUDA_VISIBLE_DEVICES` is already set,
the deployment server uses the existing visible device set.

## Quick Start

Recommended environment:

```bash
conda activate swift-vln-eval
```

### One-Shot Session

If you just want to verify one session end to end, prepare a JSONL request file and run the wrapper script.

Create `requests.jsonl`:

```json
{"type":"start","instruction":"Drive to the bank.","session_id":"demo"}
{"type":"image","image_path":"/abs/path/to/frame_0001.jpg"}
{"type":"end","reason":"done"}
```

Run it:

```bash
bash src/swiftvln/scripts/deploy/run_deploy_session.sh \
  <EXP_NAME> \
  /abs/path/to/requests.jsonl
```

This is the simplest way to smoke or manually validate the deployment pipeline.

### Long-Running Server

Start the deploy server:

```bash
bash src/swiftvln/scripts/deploy/start_overlapvln_deploy.sh \
  overlapvln-satnav-stage1-3b-1ep-f32s4-overlap8-pf-h8-random-b1.0-pool-s2-noembed-bs64-lr2e-5-20260421-123456
```

After the process prints a `ready` JSON line, keep the process alive and continue writing JSON commands to its stdin.

## Raw CLI Start

```bash
python -m swiftvln deploy \
  --model overlapvln \
  --model-name overlapvln-satnav-stage1-3b-1ep-f32s4-overlap8-pf-h8-random-b1.0-pool-s2-noembed-bs64-lr2e-5-20260421-123456
```

Optional session root:

```bash
python -m swiftvln deploy \
  --model overlapvln \
  --model-name <EXP_NAME> \
  --session-root runtime/deploy/sessions
```

## Foolproof Flow

The practical flow is:

1. Activate `swift-vln-eval`.
2. Start the server with `start_overlapvln_deploy.sh`, or run a one-shot session with `run_deploy_session.sh`.
3. Send `start` once with the task instruction.
4. Send the first `image`; the model returns one action sequence.
5. After the robot finishes one action, send the returned image back as the next `image` command.
6. If the queue is empty on that feedback image, the server immediately runs a new inference on that same image.
7. Send `end` when the task is over.

## JSONL Protocol

Input commands:

```json
{"type":"start","instruction":"Drive to the gas station.","session_id":"demo"}
{"type":"image","image_path":"/abs/path/to/frame_0001.jpg"}
{"type":"image","image_path":"/abs/path/to/frame_0002.jpg"}
{"type":"end","reason":"task finished"}
```

Behavior:

- The process emits one `ready` message after model load.
- `start` switches the state to `waiting_image`.
- The first `image` always triggers inference and returns `actions`, `next_action`, and `remaining_actions`.
- Every later `image` means one previously issued action has finished.
- If the action queue becomes empty on a feedback image, that same image is used immediately for the next inference.
- `end` writes the summary, closes logs, and releases the model.

Minimal raw example:

```bash
python -m swiftvln deploy --model overlapvln --model-name <EXP_NAME> <<'EOF'
{"type":"start","instruction":"Drive to the bank.","session_id":"demo"}
{"type":"image","image_path":"/abs/path/to/frame_0001.jpg"}
{"type":"image","image_path":"/abs/path/to/frame_0002.jpg"}
{"type":"end","reason":"done"}
EOF
```

## Session Directory

Each session is persisted under:

```text
runtime/deploy/sessions/<session_id>/
  session_meta.json
  events.jsonl
  session_summary.json
  images/
```

Input images are copied into `images/` with numbered names, for example:

- `000001_infer_input.jpg`
- `000002_feedback.jpg`
- `000003_feedback_infer_input.jpg`

## Current Limits

- Only the `overlapvln` baseline per-frame/pool/noembed path is supported.
- The parser currently handles baseline sampling settings and `num_overlap` only.
- `map`, `gtc`, `segment_gtc`, `tome`, `initial`, and embedding enhancement variants are rejected.
- Only single-session local CLI deployment is supported. There is no HTTP server.
- Image input is limited to local image paths.

## Helper Scripts

- `src/swiftvln/scripts/deploy/start_overlapvln_deploy.sh`
  - Long-running deploy server wrapper.
- `src/swiftvln/scripts/deploy/run_deploy_session.sh`
  - One-shot wrapper that reads a JSONL request file and runs a full session.
- `src/swiftvln/scripts/deploy/deploy_smoke.sh`
  - Smoke helper for `start -> image -> end`.
