# SatNav Real-time Ground Station / 实机导航地面站

[中文](#中文) | [English](#english)

---

# English

`satnav/` is the full **SatNav ground-station** stack for real-world VLN (Vision-and-Language Navigation). It wires **SwiftVLN model inference**, **RC RTMP video**, and a **flight-control CloudSDK** into one operable console.

Operators enter a navigation instruction in the web UI, inspect the live stream and model input, then trigger inference and flight actions — a **see → infer → execute** loop. The current release is **operator-stepped** (the frontend owns the pacing). Server-side fully automatic flight orchestration is planned; REST APIs are already split into composable steps.

| Path | Role | Stack |
|---|---|---|
| `satnav/frontend/` | Ground-station web console (UI / interaction) | React 18 + TypeScript + Vite |
| `satnav/backend/` | Orchestration API (model / RTMP / flight proxy) | Python + FastAPI |
| `satnav/runtime/` | Runtime data (model registry, session images, operator logs; not versioned) | — |

The frontend talks to the backend **only** over REST. The backend starts and talks to the model long-running process, consumes RTMP, and proxies flight login / DRC / stick / OSD. Training and core model code live under `src/swiftvln/`; `satnav/` holds real-world integration only.

---

## 1. In-repo vs out-of-repo dependencies

### 1.1 In this repository

| Component | Path | Notes |
|---|---|---|
| SatNav Frontend | `satnav/frontend/` | Console UI |
| SatNav Backend | `satnav/backend/` | FastAPI orchestration |
| SwiftVLN Deploy | `src/swiftvln/deployment/`, `src/swiftvln/scripts/deploy/` | Model process (started by Backend) |

### 1.2 Out of this repository

A full ground station also depends on these **out-of-repo** pieces (deployment / release details are not expanded here):

| Component | Notes |
|---|---|
| Inference environment | Install the Eval env from [`docs/installation.md`](../docs/installation.md) (PyTorch CUDA, `ms-swift`, `flash-attn`, …) |
| Model weights | Hugging Face / local checkpoint dir via `SATNAV_MODEL_PATH` (no need to place weights under repo `output/` first) |
| RTMP ingest | Receives DJI RC push; configure `SATNAV_RTMP_URL` on Backend. **You may self-host RTMP** (e.g. Nginx-RTMP / SRS); this repo does not ship a pusher |
| Flight CloudSDK Backend | Separate HTTP service (login / DRC / stick / OSD) via `SATNAV_BACKEND_HOST` / `PORT`. **Official flight-backend release is coming soon**; deploy docs will follow |

A complete closed loop needs both RTMP and the flight backend. Neither is documented in depth in this README or in `satnav/frontend`. For **model-load / inference UI** debugging only, the flight backend may be absent (**SatNav frontend / Backend still start**), but **flight-related features are unavailable**.

---

## 2. Glossary

| Term | Meaning |
|---|---|
| VLN | Vision-and-Language Navigation |
| Deploy / JSONL | SwiftVLN model long process; SatNav Backend sends `start` / `image` JSONL over stdin/stdout |
| RTMP | RC video stream; Backend pulls frames and resizes to 448×448 model input |
| CloudSDK | Flight-side HTTP backend (out of repo; official release coming soon) |
| DRC | Acquire flight control (connect + enter); stick commands need this first |
| stick-task | Async flight task (forward / turn); poll by `task_id` until terminal |
| OSD | Aircraft pose / position snapshot (height, lat/lng, heading, …) |
| `performed_inference` | Whether this `image` step ran real model inference; often with non-empty `actions` when `true` |

### Action codes

| Value | Name | Meaning |
|----|------|------|
| `0` | `STOP` | Stop |
| `1` | `FORWARD` | Forward |
| `2` | `TURN_LEFT` | Turn left |
| `3` | `TURN_RIGHT` | Turn right |

`actions` / `next_action` / `remaining_actions` / `completed_action` from the model all use the table above.

---

## 3. Architecture

> Each numbered box is an independent service.

```text
┌──────────────────────────────┐
│ ① SatNav Web                │
│ （即./frontend）              |
| React + TypeScript           │
│ 独立前端服务                   │
└──────────────┬───────────────┘
               ⇅ REST
┌─────────────────────────────────┐
│ ② SatNav API                   │
│ （即./backend）                  |
| Python + FastAPI                │
│ REST 编排（本阶段分步触发）         |  ◀───────────┐
| + RTMP Adapter（抽帧 → 448×448） |              |
└───────┬──────────┬──────────────┘              |
        |          |                             |
       响应        状态                           |
        |          |                             |
        ▲          ▲                             |
        |          |                             |
        │          │                             |
        │         HTTP（CloudSDK）                |
      JSONL命令     └──────────────┐              |
        │                         │              |
┌───────▼──────────────────────┐  │              |
│ ③ SwiftVLN 模型长进程         │  |              |
│ start_swiftvln_deploy.sh     │  │              │
│ → server.py → session.py     │  │              │
└──────────────────────────────┘  │              │
                                  │              │
                         ┌────────▼─────────┐    │
                         │ ④ 飞控 CloudSDK  │    │
                         │ login/DRC/stick  │    │
                         └──────────────────┘    │
                                                 │
┌──────────────────────────────────┐             │
│ ⑤ RTMP服务接 DJI RC(遥控器)推流    │             │
└──────────────────────────────────┘             │
               │                                 │
               └──视频流──────────────────────────┘
```

Ownership:

| # | Ownership |
|---|---|
| ① ② | This repo `satnav/` |
| ③ | This repo `src/swiftvln/` (started by ②) |
| ④ ⑤ | Out of repo (self-host RTMP; flight backend coming soon) |

Boundaries:

- UI calls SatNav FastAPI REST only — never RTMP, the model subprocess, or CloudSDK directly.
- Current release: frontend steps inference and flight; server-side auto flight orchestration is planned (APIs already composable).
- Model path: `src/swiftvln/scripts/deploy/start_swiftvln_deploy.sh` → `src/swiftvln/deployment/server.py` → `src/swiftvln/deployment/session.py` (JSONL).
- Core model code stays in `src/swiftvln/`; `satnav/` is integration only.

Model process IPC (not HTTP):

```text
FastAPI starts → src/swiftvln/scripts/deploy/start_swiftvln_deploy.sh
→ stdin  JSONL (start / image)
→ stdout JSONL responses
```

Flight traffic is proxied by SatNav API to CloudSDK (see [`backend/API_DOCS.md`](backend/API_DOCS.md)).

---

## 4. Directory layout

```text
satnav/
├── README.md                       # this file (ground-station overview)
├── backend/
│   ├── requirements.txt
│   ├── start_api.sh
│   ├── API_DOCS.md
│   ├── tests/
│   └── app/
│       ├── main.py                 # REST entry
│       ├── schemas.py
│       ├── media_cache.py
│       ├── time_utils.py
│       ├── services/
│       │   └── model_inference_service.py
│       └── adapters/
│           ├── swiftvln_deploy_client.py
│           ├── rtmp_stream_checker.py
│           ├── rtmp_frame_utils.py
│           ├── backend_health_checker.py
│           ├── flight_control_auth_client.py
│           ├── flight_control_device_registry.py
│           ├── flight_control_drc_client.py
│           └── flight_control_flight_client.py
│
├── frontend/
│   ├── design/ui-mockup.png        # UI mockup
│   ├── README.md
│   ├── package.json
│   ├── .env.development            # VITE_API_BASE
│   └── src/
│       ├── api/                    # API layer (no React)
│       │   ├── clients/            # health / model / rtmp / flight / inference / media
│       │   ├── workflows/          # flightSetup / navStep / stickPoll
│       │   └── types/
│       └── ui/                     # UI layer
│           ├── pages/ConsolePage.tsx
│           ├── hooks/useConsoleController.ts
│           ├── utils/actionQueue.ts  # action-queue state machine
│           └── components/         # Header / ActionPanel / ActionQueue / …
│
└── runtime/                        # runtime data (model_registry / model_sessions / logs / …)
```

Notes:

- Frontend details: [`frontend/README.md`](frontend/README.md) — `src/api/` vs `src/ui/`.
- API details: [`backend/API_DOCS.md`](backend/API_DOCS.md).

---

## 5. Quick Start

Assumptions:

- Repo root is `$SWIFTVLN_ROOT` (e.g. `/path/to/SwiftVLN`)
- Eval inference conda env is ready per [`docs/installation.md`](../docs/installation.md)
- Model weights dir is `$MODEL_PATH` (`config.json`, safetensors, …)
- RTMP is deployed and the RC pushes to `$RTMP_URL` (e.g. `rtmp://127.0.0.1/live/satnav`); self-host RTMP — not documented here
- Flight CloudSDK listens on `$FC_HOST:$FC_PORT` (e.g. `127.0.0.1:6789`); official release coming soon
- GPU via `CUDA_VISIBLE_DEVICES`

### 5.1 Startup order

A full ground station needs all of the following (RTMP / flight are not expanded in this repo or `frontend` docs):

1. Start RTMP and ensure the RC pushes to `$RTMP_URL`
2. Start flight CloudSDK Backend (coming soon)
3. Start SatNav Backend (starts the model process)
4. Start SatNav Frontend
5. Open the console and follow §6

### 5.2 Start SatNav Backend (full env)

```bash
cd "$SWIFTVLN_ROOT"
# Activate your Eval / Deploy inference env, e.g.:
# conda activate <your-swiftvln-eval-env>

SATNAV_MODEL_NAME="<model-name-used-for-deploy-param-resolve>" \
SATNAV_MODEL_PATH="$MODEL_PATH" \
SATNAV_DEPLOY_PYTHON="$(which python)" \
SATNAV_MODEL_STARTUP_TIMEOUT_SECONDS=600 \
SATNAV_MODEL_LOG_BUFFER_SIZE=2000 \
SATNAV_RTMP_URL="$RTMP_URL" \
SATNAV_RTMP_PROBE_TIMEOUT_SECONDS=8 \
SATNAV_RTMP_REFRESH_WAIT_SECONDS=10 \
SATNAV_RTMP_FRAME_TIMEOUT_SECONDS=10 \
SATNAV_MODEL_DEPLOY_RESPONSE_TIMEOUT_SECONDS=120 \
SATNAV_MODEL_INPUT_RESIZE_MODE=center-crop \
SATNAV_BACKEND_HOST="$FC_HOST" \
SATNAV_BACKEND_PORT="$FC_PORT" \
SATNAV_BACKEND_API_DOCS_PATH=/v3/api-docs \
SATNAV_BACKEND_PROBE_TIMEOUT_SECONDS=3 \
SATNAV_BACKEND_EXPECTED_TITLE="CloudSDK API" \
SATNAV_BACKEND_REQUEST_TIMEOUT_SECONDS=10 \
SATNAV_FLIGHT_DEFAULT_YAW_DEG=15 \
SATNAV_FLIGHT_DEFAULT_FORWARD_DISTANCE_M=10 \
SATNAV_FLIGHT_DEFAULT_YAW_TOLERANCE_DEG=0.2 \
SATNAV_FLIGHT_DEFAULT_FORWARD_TOLERANCE_M=0.3 \
SATNAV_FLIGHT_DEFAULT_TIMEOUT_MS=30000 \
CUDA_VISIBLE_DEVICES=0 \
SATNAV_API_HOST=0.0.0.0 \
SATNAV_API_PORT=8000 \
SATNAV_CORS_ORIGINS=http://127.0.0.1:5173 \
bash satnav/backend/start_api.sh
```

Required: `SATNAV_MODEL_NAME`, `SATNAV_MODEL_PATH`, `SATNAV_RTMP_URL`.  
`SATNAV_MODEL_PATH` may be any HF model folder; the API creates a deploy mapping under `satnav/runtime/model_registry/`.  
Flight `workspace_id` comes from `POST .../flight-rc-backend/login` (cached; no env var).

Install deps:

```bash
pip install -r satnav/backend/requirements.txt
```

Check:

```bash
curl -sS "http://127.0.0.1:8000/api/satnav/health"
# Swagger: http://127.0.0.1:8000/docs
```

Environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `SATNAV_MODEL_NAME` | Full model name for deploy param resolve | (required) |
| `SATNAV_MODEL_PATH` | Absolute HF model directory | (required) |
| `SATNAV_DEPLOY_PYTHON` | Python for the model process | same as FastAPI |
| `SATNAV_MODEL_STARTUP_TIMEOUT_SECONDS` | Model load timeout (s) | `600` |
| `SATNAV_MODEL_LOG_BUFFER_SIZE` | In-memory log line cap | `2000` |
| `SATNAV_RTMP_URL` | RTMP pull URL | (required) |
| `SATNAV_RTMP_PROBE_TIMEOUT_SECONDS` | RTMP connect timeout (s) | `8` |
| `SATNAV_RTMP_REFRESH_WAIT_SECONDS` | Wait for reconnect after refresh | `probe_timeout + 2` |
| `SATNAV_RTMP_FRAME_TIMEOUT_SECONDS` | Wait for fresh frame on inference (s) | `10` |
| `SATNAV_MODEL_DEPLOY_RESPONSE_TIMEOUT_SECONDS` | Deploy JSONL response timeout (s) | `120` |
| `SATNAV_MODEL_INPUT_RESIZE_MODE` | `center-crop` / `stretch` | `center-crop` |
| `SATNAV_MODEL_INFERENCE_INPUT_ROOT` | JPEG output dir | `satnav/runtime/model_sessions/inference_inputs` |
| `SATNAV_OPERATOR_LOG_ROOT` | Operator session log directory (absolute path) | `satnav/runtime/logs` |
| `SATNAV_BACKEND_HOST` | Flight CloudSDK host | — |
| `SATNAV_BACKEND_PORT` | Flight CloudSDK port | `6789` |
| `SATNAV_BACKEND_API_DOCS_PATH` | OpenAPI docs path | `/v3/api-docs` |
| `SATNAV_BACKEND_PROBE_TIMEOUT_SECONDS` | Backend probe timeout (s) | `3` |
| `SATNAV_BACKEND_EXPECTED_TITLE` | Expected OpenAPI title | `CloudSDK API` |
| `SATNAV_BACKEND_REQUEST_TIMEOUT_SECONDS` | Flight HTTP timeout (s) | `10` |
| `SATNAV_FLIGHT_DEFAULT_YAW_DEG` | Default yaw magnitude (deg) | `15` |
| `SATNAV_FLIGHT_DEFAULT_FORWARD_DISTANCE_M` | Default forward distance (m) | `10` |
| `SATNAV_FLIGHT_DEFAULT_YAW_TOLERANCE_DEG` | Default yaw tolerance (deg) | `0.2` |
| `SATNAV_FLIGHT_DEFAULT_FORWARD_TOLERANCE_M` | Default forward tolerance (m) | `0.3` |
| `SATNAV_FLIGHT_DEFAULT_TIMEOUT_MS` | Default stick-task timeout (ms) | `30000` |
| `SATNAV_API_HOST` | FastAPI bind host | `0.0.0.0` |
| `SATNAV_API_PORT` | FastAPI bind port | `8000` |
| `SATNAV_CORS_ORIGINS` | Allowed frontend Origins (comma-separated) | `http://127.0.0.1:5173` |

**Operator session logs** (from the first **Inference** click): appended to `SATNAV_OPERATOR_LOG_ROOT` (default `satnav/runtime/logs/`). **One log file per FastAPI process** (created at API startup, closed on API shutdown), e.g. `operator-20260805T141830+08.log`. Use an **absolute path** to override:

```bash
export SATNAV_OPERATOR_LOG_ROOT=/data/satnav/operator-logs
```

### 5.3 Start SatNav Frontend

```bash
cd "$SWIFTVLN_ROOT/satnav/frontend"
npm install
VITE_API_BASE=http://127.0.0.1:8000 npm run dev
# open http://127.0.0.1:5173
```

Production build:

```bash
cd "$SWIFTVLN_ROOT/satnav/frontend"
npm run build     # dist/
npm run preview
```

Notes:

- Backend `SATNAV_CORS_ORIGINS` must match the frontend Origin; prefer `127.0.0.1` over mixing with `localhost`.
- `Ctrl+C` on `npm run dev` does not stop in-tab JS polls — close the browser tab.
- Frontend details: [`frontend/README.md`](frontend/README.md).

### 5.4 Model process lifecycle

```text
FastAPI starts
  → starts src/swiftvln/scripts/deploy/start_swiftvln_deploy.sh in background
  → waits until model ready (does not block /health)
  → first inference sends deploy start (with instruction)
  → each inference sends image
  → killing the API process stops the model subprocess
```

- No public deploy `end` yet; multiple `inference` calls in one API process share a session.
- The model subprocess follows FastAPI lifetime, not single UI actions.

---

## 6. Operator main loop

```text
register_device → login → drc
  → inference (actions / next_action / remaining_actions)
  → frontend action queue records the current slot
  → execute one step: forward or turn (task_id)
  → poll stick-task until COMPLETED / FAILED
  → inference again (feedback frame, advance currentIndex)
  → loop until next_action == 0 or emergency STOP
```

Rules:

- **Inference** only calls `POST .../model/inference`; **execute one step** only calls `forward` / `turn` (decoupled).
- Do not infer again while flight is incomplete (frontend `canRunInference` gate).
- First frame with `performed_inference=true` fills 1–4 slots; feedback frames have `actions=[]`, `currentIndex = queueLength - remaining_actions.length`.
- After STOP (`next_action=0` or emergency STOP), re-acquire flight control; inference remains available for debug.

Console layout:

```text
[Header: health badges + Session + settings]
[Instruction + Infer]     [Flight: bind / login / DRC + OSD]
[RTMP raw] [448×448] [Action ×4 + Execute / Emergency STOP]
[Logs]
```

Action-queue / button gating details: [`frontend/README.md`](frontend/README.md).

---

## 7. FastAPI APIs for the frontend

Full request / response / samples: [`backend/API_DOCS.md`](backend/API_DOCS.md). With the service up: `http://127.0.0.1:8000/docs` (Swagger UI).

| Method | Path | Purpose |
| ------ | ------------------------------- | ------------------- |
| `GET`  | `/api/satnav/health`            | FastAPI liveness |
| `GET`  | `/api/satnav/system/model/status` | Model load / run status |
| `GET`  | `/api/satnav/system/rtmp/status` | Configured RTMP stream health |
| `POST` | `/api/satnav/system/rtmp/refresh` | Refresh RTMP connection |
| `GET`  | `/api/satnav/system/model/logs`  | Incremental model process logs |
| `GET`  | `/api/satnav/system/flight-rc-backend/status` | Flight backend reachability |
| `POST` | `/api/satnav/system/flight-rc-backend/login` | Login or refresh flight token |
| `POST` | `/api/satnav/system/flight-rc-backend/register_device` | Register `rc_sn` / `device_sn` |
| `GET` | `/api/satnav/system/flight-rc-backend/cur_device_info` | Current `rc_sn` / `device_sn` |
| `POST` | `/api/satnav/system/flight-rc-backend/drc` | Acquire flight control (DRC connect + enter) |
| `POST` | `/api/satnav/system/flight-rc-backend/flight/forward` | Forward (pitch-by-distance) |
| `POST` | `/api/satnav/system/flight-rc-backend/flight/turn` | Turn (TURN_LEFT / TURN_RIGHT) |
| `GET` | `/api/satnav/system/flight-rc-backend/flight/stick-task/{task_id}` | Stick-task status |
| `GET` | `/api/satnav/system/flight-rc-backend/flight/osd/latest` | Latest OSD snapshot |
| `POST` | `/api/satnav/model/inference`   | Grab RTMP frame + one model step |
| `GET`  | `/api/satnav/media/raw_img`         | Latest raw RTMP JPEG |
| `GET`  | `/api/satnav/media/model_input_img` | Latest 448×448 model-input JPEG |
| `POST` | `/api/satnav/operator/logs/start` | Start operator session log file |
| `POST` | `/api/satnav/operator/logs/append` | Append lines to operator log |
| `POST` | `/api/satnav/operator/logs/close` | Close operator log file |
| `GET`  | `/api/satnav/operator/logs/status` | Operator log file status |

Inference body sample:

```json
{
  "instruction": "Fly to the lake and fly around the lake clockwise"
}
```

Frontend pattern: preview `raw_img` / `model_input_img` → `inference` → `forward` / `turn` → poll `stick-task`.

Backend chain (see code / API_DOCS for details):

```text
inference → RTMP frame → 448×448 → Deploy JSONL (start/image)
flight    → DRC → forward/turn → stick-task / osd
```

---

## 8. Related docs

| Doc | Content |
|---|---|
| [`satnav/backend/API_DOCS.md`](backend/API_DOCS.md) | REST contracts / curl samples |
| [`satnav/frontend/README.md`](frontend/README.md) | Frontend structure & console behavior |
| [`docs/installation.md`](../docs/installation.md) | Eval / inference env setup |
| `src/swiftvln/deployment/` | Deploy protocol & implementation |
| `http://127.0.0.1:8000/docs` | Live Swagger UI |

---

## 9. Current scope / Roadmap

**Current release**

- Operator-stepped frontend loop: inference and flight are decoupled
- Backend: health, RTMP, model inference, flight proxy (login / DRC / stick / OSD), media preview
- Frontend action queue tracks multi-step actions and stick-task status

**Planned**

- Server-side automatic flight orchestration (new layer on existing REST steps; no change to model / RTMP / flight boundaries)


---

# 中文

`satnav/` 是整套 **SatNav 地面站**代码：面向实机 VLN（Vision-and-Language Navigation）任务，把 **SwiftVLN 模型推理**、**遥控器 RTMP 视频流** 与 **飞控 CloudSDK** 串成一台可操作的一体机控制台。

操作员在网页上输入导航指令、查看推流画面与模型输入，并触发推理与飞控动作，形成「看图 → 推理 → 执行」的闭环。当前发布以**操作员分步触发**为主（前端控制节奏）；服务端全自动飞控编排列为后续能力，现有 REST 已按可编排步骤拆分。

| 目录 | 角色 | 技术栈 |
|---|---|---|
| `satnav/frontend/` | 地面站 Web 控制台（展示与人机交互） | React 18 + TypeScript + Vite |
| `satnav/backend/` | 地面站编排 API（代理模型 / RTMP / 飞控） | Python + FastAPI |
| `satnav/runtime/` | 运行时数据（模型映射、session 图片、操作复盘日志等，不入代码仓） | — |

前端只通过 REST 调用后端；后端负责拉起并对接模型长进程、消费 RTMP、代理飞控 login / DRC / stick / OSD。模型训练与核心算法仍在仓库 `src/swiftvln/`；本目录只放实机集成与地面站相关代码。

---

## 1. 仓库内 / 仓外依赖

### 1.1 本仓库内

| 组件 | 路径 | 说明 |
|---|---|---|
| SatNav Frontend | `satnav/frontend/` | 控制台 UI |
| SatNav Backend | `satnav/backend/` | FastAPI 编排层 |
| SwiftVLN Deploy | `src/swiftvln/deployment/`、`src/swiftvln/scripts/deploy/` | 模型长进程（由 Backend 拉起） |

### 1.2 仓外

完整地面站还依赖下列**非本仓**组件（部署与发布不在本文展开）：

| 组件 | 说明 |
|---|---|
| 推理环境 | 按仓库 [`docs/installation.md`](../docs/installation.md) 的 Eval 环境安装（含 PyTorch CUDA、`ms-swift`、`flash-attn` 等） |
| 模型权重 | HuggingFace / 本地 checkpoint 目录；通过 `SATNAV_MODEL_PATH` 指向，无需预先放进仓库 `output/` |
| RTMP 推流服务 | 接收 DJI RC 推流；Backend 启动时配置 `SATNAV_RTMP_URL`。**RTMP 服务可自行部署**（常见 Nginx-RTMP / SRS 等），本仓不内置推流端 |
| 飞控 CloudSDK Backend | 独立 HTTP 服务（login / DRC / stick / OSD）；通过 `SATNAV_BACKEND_HOST` / `PORT` 连接。**飞控后端待正式 release（coming soon）**，届时另附部署与对接文档 |

完整实机闭环需要 RTMP 与飞控后端均可用；二者均不在本文中展开。仅做「模型加载 / 推理 UI」调试时，可暂时没有飞控（**不影响 SatNav 前端 / Backend 启动**），但**飞控相关能力不可用**。

---

## 2. 术语表

| 术语 | 含义 |
|---|---|
| VLN | Vision-and-Language Navigation，视觉语言导航 |
| Deploy / JSONL | SwiftVLN 模型长进程；SatNav Backend 经 stdin/stdout 发送 `start` / `image` 等 JSONL 命令 |
| RTMP | 遥控器推流协议；Backend 拉流抽帧并预处理为 448×448 模型输入 |
| CloudSDK | 飞控侧 HTTP 后端（非本仓；待正式 release） |
| DRC | 获取飞行控制（connect + enter）；未完成前无法下发 stick 动作 |
| stick-task | 飞控异步任务（前进 / 转向）；用 `task_id` 轮询至完成 |
| OSD | 飞机姿态与位置快照（高度、经纬度、航向等） |
| `performed_inference` | 本次 `image` 是否触发真实模型推理；`true` 时常伴随非空 `actions` |

### 动作编码

| 值 | 名称 | 含义 |
|----|------|------|
| `0` | `STOP` | 停止 |
| `1` | `FORWARD` | 前进 |
| `2` | `TURN_LEFT` | 左转 |
| `3` | `TURN_RIGHT` | 右转 |

模型返回的 `actions` / `next_action` / `remaining_actions` / `completed_action` 均使用上表整数编码。

---

## 3. 系统架构

> 标号+方框的为一个独立服务。

```text
┌──────────────────────────────┐
│ ① SatNav Web                │
│ （即./frontend）              |
| React + TypeScript           │
│ 独立前端服务                   │
└──────────────┬───────────────┘
               ⇅ REST
┌─────────────────────────────────┐
│ ② SatNav API                   │
│ （即./backend）                  |
| Python + FastAPI                │
│ REST 编排（本阶段分步触发）     |  ◀───────────┐
| + RTMP Adapter（抽帧 → 448×448） |              |
└───────┬──────────┬──────────────┘              |
        |          |                             |
       响应        状态                           |
        |          |                             |
        ▲          ▲                             |
        |          |                             |
        │          │                             |
        │         HTTP（CloudSDK）                |
      JSONL命令     └──────────────┐              |
        │                         │              |
┌───────▼──────────────────────┐  │              |
│ ③ SwiftVLN 模型长进程         │  |              |
│ start_swiftvln_deploy.sh     │  │              │
│ → server.py → session.py     │  │              │
└──────────────────────────────┘  │              │
                                  │              │
                         ┌────────▼─────────┐    │
                         │ ④ 飞控 CloudSDK  │    │
                         │ login/DRC/stick  │    │
                         └──────────────────┘    │
                                                 │
┌──────────────────────────────────┐             │
│ ⑤ RTMP服务接 DJI RC(遥控器)推流    │             │
└──────────────────────────────────┘             │
               │                                 │
               └──视频流──────────────────────────┘
```

图中组件归属：

| 标号 | 归属 |
|---|---|
| ① ② | 本仓库 `satnav/` |
| ③ | 本仓库 `src/swiftvln/`（由 ② 启动） |
| ④ ⑤ | 非本仓独立服务（RTMP 可自行部署；飞控 coming soon） |

边界原则：

- UI 只调用 SatNav FastAPI REST，不直接连 RTMP、模型子进程或飞控 CloudSDK。
- 当前发布由前端分步触发推理与飞控；服务端全自动飞控编排为后续规划（接口已按可编排步骤拆分）。
- 模型调用走正式链路：`src/swiftvln/scripts/deploy/start_swiftvln_deploy.sh` → `src/swiftvln/deployment/server.py` → `src/swiftvln/deployment/session.py`（JSONL）。
- 模型核心代码在 `src/swiftvln/`；`satnav/` 只放实机集成代码。

模型子进程通信（非 HTTP）：

```text
FastAPI 启动 → src/swiftvln/scripts/deploy/start_swiftvln_deploy.sh
→ stdin  发送 JSONL（start / image）
→ stdout 读取 JSONL 响应
```

飞控经 SatNav API 代理调用 CloudSDK backend（详见 [`backend/API_DOCS.md`](backend/API_DOCS.md)）。

---

## 4. 目录说明

```text
satnav/
├── README.md                       # 本文件（地面站总览）
├── backend/
│   ├── requirements.txt
│   ├── start_api.sh
│   ├── API_DOCS.md
│   ├── tests/
│   └── app/
│       ├── main.py                 # REST 入口
│       ├── schemas.py
│       ├── media_cache.py
│       ├── time_utils.py
│       ├── services/
│       │   └── model_inference_service.py
│       └── adapters/
│           ├── swiftvln_deploy_client.py
│           ├── rtmp_stream_checker.py
│           ├── rtmp_frame_utils.py
│           ├── backend_health_checker.py
│           ├── flight_control_auth_client.py
│           ├── flight_control_device_registry.py
│           ├── flight_control_drc_client.py
│           └── flight_control_flight_client.py
│
├── frontend/
│   ├── design/ui-mockup.png        # UI 设计参考图
│   ├── README.md
│   ├── package.json
│   ├── .env.development            # VITE_API_BASE
│   └── src/
│       ├── api/                    # 调用侧（无 React 依赖）
│       │   ├── clients/            # health / model / rtmp / flight / inference / media
│       │   ├── workflows/          # flightSetup / navStep / stickPoll
│       │   └── types/
│       └── ui/                     # 显示侧
│           ├── pages/ConsolePage.tsx
│           ├── hooks/useConsoleController.ts
│           ├── utils/actionQueue.ts  # Action 队列状态机
│           └── components/         # Header / ActionPanel / ActionQueue 等
│
└── runtime/                        # 运行时数据（model_registry / model_sessions / logs 等）
```

说明：

- 前端细节：[`frontend/README.md`](frontend/README.md)；调用侧 `src/api/`，显示侧 `src/ui/`。
- 接口细节：[`backend/API_DOCS.md`](backend/API_DOCS.md)。

---

## 5. Quick Start

以下假设：

- 仓库根目录为 `$SWIFTVLN_ROOT`（例如 `/path/to/SwiftVLN`）
- Eval 推理 conda 环境已按 [`docs/installation.md`](../docs/installation.md) 装好
- 模型权重目录为 `$MODEL_PATH`（含 `config.json` 与 safetensors 等）
- RTMP 服务已部署，遥控器推流到 `$RTMP_URL`（例如 `rtmp://127.0.0.1/live/satnav`）；RTMP 可自行部署，本仓不展开
- 飞控 CloudSDK 已在 `$FC_HOST:$FC_PORT` 运行（例如 `127.0.0.1:6789`）；飞控后端待正式 release（coming soon），部署细节届时另附
- GPU 使用 `CUDA_VISIBLE_DEVICES` 指定

### 5.1 启动顺序

完整地面站需下列服务均就绪（RTMP、飞控不在本仓 / `frontend` 文档展开）：

1. 启动 RTMP 服务，并保证遥控器推流到 `$RTMP_URL`
2. 启动飞控 CloudSDK Backend（coming soon）
3. 启动 SatNav Backend（会拉起模型长进程）
4. 启动 SatNav Frontend
5. 浏览器打开控制台，按 §6 操作主流程

### 5.2 启动 SatNav Backend（完整环境变量）

```bash
cd "$SWIFTVLN_ROOT"
# 激活你的 Eval / Deploy 推理环境，例如：
# conda activate <your-swiftvln-eval-env>

SATNAV_MODEL_NAME="<model-name-used-for-deploy-param-resolve>" \
SATNAV_MODEL_PATH="$MODEL_PATH" \
SATNAV_DEPLOY_PYTHON="$(which python)" \
SATNAV_MODEL_STARTUP_TIMEOUT_SECONDS=600 \
SATNAV_MODEL_LOG_BUFFER_SIZE=2000 \
SATNAV_RTMP_URL="$RTMP_URL" \
SATNAV_RTMP_PROBE_TIMEOUT_SECONDS=8 \
SATNAV_RTMP_REFRESH_WAIT_SECONDS=10 \
SATNAV_RTMP_FRAME_TIMEOUT_SECONDS=10 \
SATNAV_MODEL_DEPLOY_RESPONSE_TIMEOUT_SECONDS=120 \
SATNAV_MODEL_INPUT_RESIZE_MODE=center-crop \
SATNAV_BACKEND_HOST="$FC_HOST" \
SATNAV_BACKEND_PORT="$FC_PORT" \
SATNAV_BACKEND_API_DOCS_PATH=/v3/api-docs \
SATNAV_BACKEND_PROBE_TIMEOUT_SECONDS=3 \
SATNAV_BACKEND_EXPECTED_TITLE="CloudSDK API" \
SATNAV_BACKEND_REQUEST_TIMEOUT_SECONDS=10 \
SATNAV_FLIGHT_DEFAULT_YAW_DEG=15 \
SATNAV_FLIGHT_DEFAULT_FORWARD_DISTANCE_M=10 \
SATNAV_FLIGHT_DEFAULT_YAW_TOLERANCE_DEG=0.2 \
SATNAV_FLIGHT_DEFAULT_FORWARD_TOLERANCE_M=0.3 \
SATNAV_FLIGHT_DEFAULT_TIMEOUT_MS=30000 \
CUDA_VISIBLE_DEVICES=0 \
SATNAV_API_HOST=0.0.0.0 \
SATNAV_API_PORT=8000 \
SATNAV_CORS_ORIGINS=http://127.0.0.1:5173 \
bash satnav/backend/start_api.sh
```

必填项：`SATNAV_MODEL_NAME`、`SATNAV_MODEL_PATH`、`SATNAV_RTMP_URL`。  
`SATNAV_MODEL_PATH` 可为任意 HF 模型目录；API 会在 `satnav/runtime/model_registry/` 下建立 deployment 映射。  
飞控 `workspace_id` 由 `POST .../flight-rc-backend/login` 返回并缓存，无需环境变量。

依赖安装：

```bash
pip install -r satnav/backend/requirements.txt
```

验证：

```bash
curl -sS "http://127.0.0.1:8000/api/satnav/health"
# 交互式文档：http://127.0.0.1:8000/docs
```

环境变量含义一览：

| 变量 | 含义 | 默认 |
|---|---|---|
| `SATNAV_MODEL_NAME` | 用于解析推理参数的完整模型名称 | （必填） |
| `SATNAV_MODEL_PATH` | HF 模型文件夹绝对路径 | （必填） |
| `SATNAV_DEPLOY_PYTHON` | 模型进程使用的 Python | 与 FastAPI 相同 |
| `SATNAV_MODEL_STARTUP_TIMEOUT_SECONDS` | 模型加载超时（秒） | `600` |
| `SATNAV_MODEL_LOG_BUFFER_SIZE` | 内存日志条数上限 | `2000` |
| `SATNAV_RTMP_URL` | RTMP 拉流地址 | （必填） |
| `SATNAV_RTMP_PROBE_TIMEOUT_SECONDS` | RTMP 建连超时（秒） | `8` |
| `SATNAV_RTMP_REFRESH_WAIT_SECONDS` | 刷新 RTMP 时等待重连结果的超时 | `probe_timeout + 2` |
| `SATNAV_RTMP_FRAME_TIMEOUT_SECONDS` | 推理时等待新 RTMP 帧超时（秒） | `10` |
| `SATNAV_MODEL_DEPLOY_RESPONSE_TIMEOUT_SECONDS` | Deploy JSONL 响应超时（秒） | `120` |
| `SATNAV_MODEL_INPUT_RESIZE_MODE` | 模型输入缩放：`center-crop` / `stretch` | `center-crop` |
| `SATNAV_MODEL_INFERENCE_INPUT_ROOT` | 推理 JPEG 落盘目录 | `satnav/runtime/model_sessions/inference_inputs` |
| `SATNAV_OPERATOR_LOG_ROOT` | 操作复盘日志目录（绝对路径） | `satnav/runtime/logs` |
| `SATNAV_BACKEND_HOST` | 飞控 CloudSDK 地址 | — |
| `SATNAV_BACKEND_PORT` | 飞控 CloudSDK 端口 | `6789` |
| `SATNAV_BACKEND_API_DOCS_PATH` | OpenAPI 文档路径 | `/v3/api-docs` |
| `SATNAV_BACKEND_PROBE_TIMEOUT_SECONDS` | backend 探测超时（秒） | `3` |
| `SATNAV_BACKEND_EXPECTED_TITLE` | 期望的 OpenAPI 服务名 | `CloudSDK API` |
| `SATNAV_BACKEND_REQUEST_TIMEOUT_SECONDS` | 飞控 HTTP 请求超时（秒） | `10` |
| `SATNAV_FLIGHT_DEFAULT_YAW_DEG` | 默认转向角度（绝对值） | `15` |
| `SATNAV_FLIGHT_DEFAULT_FORWARD_DISTANCE_M` | 默认前进距离（米） | `10` |
| `SATNAV_FLIGHT_DEFAULT_YAW_TOLERANCE_DEG` | 默认转向容差（度） | `0.2` |
| `SATNAV_FLIGHT_DEFAULT_FORWARD_TOLERANCE_M` | 默认前进容差（米） | `0.3` |
| `SATNAV_FLIGHT_DEFAULT_TIMEOUT_MS` | 默认 Stick 任务超时（毫秒） | `30000` |
| `SATNAV_API_HOST` | FastAPI 监听地址 | `0.0.0.0` |
| `SATNAV_API_PORT` | FastAPI 监听端口 | `8000` |
| `SATNAV_CORS_ORIGINS` | 允许跨域的前端 Origin（逗号分隔） | `http://127.0.0.1:5173` |

**操作复盘日志**（从首次点击「推理」起）：由后端写入 `SATNAV_OPERATOR_LOG_ROOT`（默认 `satnav/runtime/logs/`）。**每次 FastAPI 服务启动一个文件**（启动时创建、进程退出时关闭），例如 `operator-20260805T141830+08.log`。自定义目录请设置**绝对路径**：

```bash
export SATNAV_OPERATOR_LOG_ROOT=/data/satnav/operator-logs
```

### 5.3 启动 SatNav Frontend

```bash
cd "$SWIFTVLN_ROOT/satnav/frontend"
npm install
VITE_API_BASE=http://127.0.0.1:8000 npm run dev
# 浏览器打开 http://127.0.0.1:5173
```

生产构建：

```bash
cd "$SWIFTVLN_ROOT/satnav/frontend"
npm run build     # 输出 dist/
npm run preview
```

注意：

- Backend 的 `SATNAV_CORS_ORIGINS` 须与前端 Origin 一致；建议统一使用 `127.0.0.1`，避免与 `localhost` 混用。
- 仅 `Ctrl+C` 停止 `npm run dev` 不会结束浏览器页内轮询；关闭标签页即可。
- 前端细节见 [`frontend/README.md`](frontend/README.md)。

### 5.4 模型进程生命周期

```text
FastAPI 启动
  → 后台拉起 src/swiftvln/scripts/deploy/start_swiftvln_deploy.sh
  → 等待模型 ready（不阻塞 /health）
  → 首次 inference 发送 deploy start（带 instruction）
  → 每次 inference 发送 image
  → API 进程退出时关闭模型子进程
```

- 当前未对外暴露 deploy `end`；同一 API 进程内可多次 `inference` 共用 session。
- 模型子进程随 FastAPI 启停，不随单次操作单独重启。

---

## 6. 操作员主流程

```text
register_device → login → drc
  → inference（得 actions / next_action / remaining_actions）
  → 前端 Action 队列登记当前槽
  → 执行一步：forward 或 turn（得 task_id）
  → 轮询 stick-task 至 COMPLETED / FAILED
  → 再次 inference（反馈帧，推进 currentIndex）
  → 循环；next_action == 0 或应急 STOP 时结束
```

关键约束：

- **推理**只调 `POST .../model/inference`；**执行一步**只调 `forward` / `turn`（二者解耦）。
- 飞控未完成前勿再次推理（前端 `canRunInference` 门控）。
- 首帧 `performed_inference=true` 填充 1~4 槽；反馈帧 `actions=[]`，`currentIndex = queueLength - remaining_actions.length`。
- STOP（模型 `next_action=0` 或应急 STOP）后需重新「获取飞行控制」；推理仍可用于调试。

控制台布局概览：

```text
[Header：健康徽章 + Session + 设置]
[导航指令 + 推理]     [飞控：绑定 / 登录 / DRC + OSD]
[RTMP 原始帧] [448×448] [Action 四槽 + 执行一步 / 应急 STOP]
[日志]
```

Action 队列与按钮门控细节见 [`frontend/README.md`](frontend/README.md)。

---

## 7. FastAPI 对前端提供的接口

完整入参 / 出参 / 示例见 [`backend/API_DOCS.md`](backend/API_DOCS.md)。服务启动后也可访问 `http://127.0.0.1:8000/docs`（Swagger UI）。

| 方法     | 接口                              | 用途                  |
| ------ | ------------------------------- | ------------------- |
| `GET`  | `/api/satnav/health`            | FastAPI 存活检查        |
| `GET`  | `/api/satnav/system/model/status` | 模型加载和运行状态          |
| `GET`  | `/api/satnav/system/rtmp/status` | 检查部署时配置的 RTMP 流是否有效 |
| `POST` | `/api/satnav/system/rtmp/refresh` | 刷新 RTMP 连接（推流就绪后重连） |
| `GET`  | `/api/satnav/system/model/logs`  | 增量获取模型加载和运行日志      |
| `GET`  | `/api/satnav/system/flight-rc-backend/status` | 飞控 backend 服务可达性检查  |
| `POST` | `/api/satnav/system/flight-rc-backend/login` | 登录或刷新飞控 backend token |
| `POST` | `/api/satnav/system/flight-rc-backend/register_device` | 登记遥控器 `rc_sn` 与飞行器 `device_sn` |
| `GET` | `/api/satnav/system/flight-rc-backend/cur_device_info` | 查询当前登记的 `rc_sn` / `device_sn` |
| `POST` | `/api/satnav/system/flight-rc-backend/drc` | 获取飞行控制（DRC connect + enter） |
| `POST` | `/api/satnav/system/flight-rc-backend/flight/forward` | 执行前进（FORWARD / pitch-by-distance） |
| `POST` | `/api/satnav/system/flight-rc-backend/flight/turn` | 执行转向（TURN_LEFT / TURN_RIGHT） |
| `GET` | `/api/satnav/system/flight-rc-backend/flight/stick-task/{task_id}` | 查询 Stick 闭环任务状态（飞控执行进度/结果） |
| `GET` | `/api/satnav/system/flight-rc-backend/flight/osd/latest` | 获取当前 OSD 快照（高度 / 经纬度 / 航向等） |
| `POST` | `/api/satnav/model/inference`   | 从 RTMP 抽一帧并执行一次模型推理 |
| `GET`  | `/api/satnav/media/raw_img`         | 获取最新 RTMP 原始帧（JPEG）       |
| `GET`  | `/api/satnav/media/model_input_img` | 获取最近一次推理的 448×448 模型输入（JPEG）   |
| `POST` | `/api/satnav/operator/logs/start` | 开始写入操作复盘日志文件 |
| `POST` | `/api/satnav/operator/logs/append` | 追加操作复盘日志行 |
| `POST` | `/api/satnav/operator/logs/close` | 关闭操作复盘日志文件 |
| `GET`  | `/api/satnav/operator/logs/status` | 查询操作复盘日志状态 |

推理请求示例：

```json
{
  "instruction": "Fly to the lake and fly around the lake clockwise"
}
```

前端调用方式：预览 `raw_img` / `model_input_img` → `inference` 得 `next_action` → `forward` / `turn` → 轮询 `stick-task`。

Backend 内部链路摘要（实现细节见代码与 API_DOCS）：

```text
inference → RTMP 抽帧 → 448×448 → Deploy JSONL（start/image）
flight    → DRC → forward/turn → stick-task / osd
```

---

## 8. 相关文档

| 文档 | 内容 |
|---|---|
| [`satnav/backend/API_DOCS.md`](backend/API_DOCS.md) | REST 入参 / 出参 / curl 示例 |
| [`satnav/frontend/README.md`](frontend/README.md) | 前端开发、分层、控制台行为 |
| [`docs/installation.md`](../docs/installation.md) | Eval / 推理环境安装 |
| `src/swiftvln/deployment/` | 模型 Deploy 协议与实现 |
| `http://127.0.0.1:8000/docs` | 运行中 Swagger UI |

---

## 9. Current scope / Roadmap

**Current release**

- 操作员在前端分步触发：推理与飞控解耦
- Backend 提供健康检查、RTMP、模型推理、飞控代理（login / DRC / stick / OSD）、媒体预览
- Frontend Action 队列维护多步动作与 stick-task 状态

**Planned**

- 服务端全自动飞控编排（在现有 REST 步骤之上增加编排层，不改变模型 / RTMP / 飞控对接边界）
