#!/usr/bin/env bash

set -euo pipefail

# Configurable at startup (model / RTMP / flight / API); 启动时可配置（模型 / RTMP / 飞控 / API）：
# SATNAV_MODEL_NAME — required full model name for inference parameter resolution; 必填，用于解析推理参数的完整模型名称。
# SATNAV_MODEL_PATH — required absolute path to the HF model directory; 必填，实际 HF 模型文件夹的绝对路径。
# SATNAV_DEPLOY_PYTHON — Python for the model subprocess; defaults to the FastAPI interpreter; 模型子进程 Python，默认与 FastAPI 相同。
# SATNAV_MODEL_STARTUP_TIMEOUT_SECONDS — model load timeout in seconds; default 600; 模型加载超时（秒），默认 600。
# SATNAV_MODEL_LOG_BUFFER_SIZE — in-memory log line cap; default 2000; 内存日志条数上限，默认 2000。
# SATNAV_RTMP_URL — required RTMP ingest URL; 必填，RTMP 拉流地址。
# SATNAV_RTMP_PROBE_TIMEOUT_SECONDS — RTMP connect timeout in seconds; default 8; RTMP 建连超时（秒），默认 8。
# SATNAV_RTMP_REFRESH_WAIT_SECONDS — wait for RTMP refresh result; default probe_timeout+2; 刷新 RTMP 等待重连结果超时，默认 probe_timeout+2。
# SATNAV_BACKEND_HOST / SATNAV_BACKEND_PORT — flight-control backend host and port; default port 6789; 飞控 backend 地址与端口，默认端口 6789。
# SATNAV_BACKEND_API_DOCS_PATH — OpenAPI docs path; default /v3/api-docs; OpenAPI 文档路径，默认 /v3/api-docs。
# SATNAV_BACKEND_PROBE_TIMEOUT_SECONDS — backend health probe timeout in seconds; default 3; backend 探测超时（秒），默认 3。
# SATNAV_BACKEND_EXPECTED_TITLE — expected OpenAPI service title; default CloudSDK API; 期望 OpenAPI 服务名，默认 CloudSDK API。
# SATNAV_BACKEND_REQUEST_TIMEOUT_SECONDS — flight backend HTTP timeout in seconds; default 10; 飞控 backend HTTP 请求超时（秒），默认 10。
# SATNAV_FLIGHT_DEFAULT_YAW_DEG — default turn angle (absolute degrees); default 15; 默认转向角度（绝对值，度），默认 15。
# SATNAV_FLIGHT_DEFAULT_FORWARD_DISTANCE_M — default forward distance in meters; default 10; 默认前进距离（米），默认 10。
# SATNAV_FLIGHT_DEFAULT_YAW_TOLERANCE_DEG — default yaw tolerance in degrees; default 0.2; 默认转向容差（度），默认 0.2。
# SATNAV_FLIGHT_DEFAULT_FORWARD_TOLERANCE_M — default forward tolerance in meters; default 0.3; 默认前进容差（米），默认 0.3。
# SATNAV_FLIGHT_DEFAULT_TIMEOUT_MS — default stick-task timeout in milliseconds; default 30000; 默认 Stick 任务超时（毫秒），默认 30000。
# SATNAV_CORS_ORIGINS — comma-separated allowed frontend Origins; default http://127.0.0.1:5173; 允许跨域的前端 Origin（逗号分隔），默认 http://127.0.0.1:5173。
# SATNAV_API_HOST / SATNAV_API_PORT — FastAPI bind host and port; default 0.0.0.0:8000; FastAPI 监听地址与端口，默认 0.0.0.0:8000。
# CUDA_VISIBLE_DEVICES — GPU index for the model process, e.g. 0; 指定模型使用的 GPU，例如 0。
# Dependencies: pip install -r satnav/backend/requirements.txt (fastapi, opencv, Pillow, numpy, …); 依赖：pip install -r satnav/backend/requirements.txt（含 fastapi / opencv / Pillow / numpy 等）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${SATNAV_API_HOST:-0.0.0.0}"
PORT="${SATNAV_API_PORT:-8000}"

: "${SATNAV_MODEL_NAME:?SATNAV_MODEL_NAME must be set}"
: "${SATNAV_MODEL_PATH:?SATNAV_MODEL_PATH must be set}"
: "${SATNAV_RTMP_URL:?SATNAV_RTMP_URL must be set, e.g. SATNAV_RTMP_URL=rtmp://<host>/<app>/<stream> bash satnav/backend/start_api.sh}"

cd "$SCRIPT_DIR"
exec python3 -m uvicorn app.main:app --host "$HOST" --port "$PORT"
