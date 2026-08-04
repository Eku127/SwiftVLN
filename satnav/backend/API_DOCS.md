# SatNav Backend API Docs / 文档

[中文](#中文) | [English](#english)

---

# English

Service version: `0.1.9`  
Default Base URL: `http://127.0.0.1:8000` (controlled by `SATNAV_API_HOST` / `SATNAV_API_PORT`)

FastAPI interactive docs:

- Swagger UI: `GET /docs`
- OpenAPI JSON: `GET /openapi.json`

This document lists **all currently implemented** REST endpoints, including request/response fields and examples.

---

## Conventions

| Item | Description |
|------|-------------|
| Content-Type | Use `application/json` when the request body is JSON |
| Timestamps | Shanghai timezone ISO 8601 strings, e.g. `2026-08-02T21:46:18.757010+08:00` |
| Error responses | FastAPI standard format: `{"detail": "error description"}` |
| CORS | Allowed frontend Origins from `SATNAV_CORS_ORIGINS` (comma-separated); default `http://127.0.0.1:5173` |
| Action codes | See the action code table below; integer values `0`–`3` |

### Action codes

Model and deploy session fields `actions` / `next_action` / `remaining_actions` / `completed_action` all use the following integer codes:

| Value | Name | Meaning |
|----|------|------|
| `0` | `STOP` | Stop |
| `1` | `FORWARD` | Forward |
| `2` | `TURN_LEFT` | Turn left |
| `3` | `TURN_RIGHT` | Turn right |

`raw_action_text` is the model’s raw output string, usually a space-separated code sequence, e.g. `"0 1 2 3"`.

### HTTP status codes (may appear on any endpoint)

| Status | Meaning |
|--------|---------|
| `200` | Success |
| `400` | Request parameters or business preconditions not satisfied |
| `422` | Request body validation failed (Pydantic) |
| `503` | Dependency not configured or deploy process unavailable |
| `504` | Timed out waiting for RTMP frame or deploy response |

---

## Endpoint index

| Method | Path | Description |
|------|------|------|
| `GET` | `/api/satnav/health` | API process liveness check |
| `GET` | `/api/satnav/system/model/status` | Model load and RTMP reader summary status |
| `GET` | `/api/satnav/system/model/logs` | Incrementally fetch model subprocess logs |
| `GET` | `/api/satnav/system/rtmp/status` | RTMP pull stream connection status |
| `POST` | `/api/satnav/system/rtmp/refresh` | Refresh RTMP connection (restart background reader) |
| `GET` | `/api/satnav/system/flight-rc-backend/status` | Flight-control CloudSDK backend reachability |
| `POST` | `/api/satnav/system/flight-rc-backend/login` | Login or refresh flight-control token |
| `POST` | `/api/satnav/system/flight-rc-backend/register_device` | Register RC `rc_sn` and aircraft `device_sn` |
| `GET` | `/api/satnav/system/flight-rc-backend/cur_device_info` | Query currently registered `rc_sn` / `device_sn` |
| `POST` | `/api/satnav/system/flight-rc-backend/drc` | Acquire flight control (DRC connect + enter) |
| `POST` | `/api/satnav/system/flight-rc-backend/flight/forward` | Execute forward (FORWARD / pitch-by-distance) |
| `POST` | `/api/satnav/system/flight-rc-backend/flight/turn` | Execute turn (TURN_LEFT / TURN_RIGHT) |
| `GET` | `/api/satnav/system/flight-rc-backend/flight/stick-task/{task_id}` | Query Stick closed-loop task status |
| `GET` | `/api/satnav/system/flight-rc-backend/flight/osd/latest` | Get current OSD info (LiveStore snapshot) |
| `POST` | `/api/satnav/model/inference` | RTMP frame capture + one deploy image step |
| `GET` | `/api/satnav/media/raw_img` | Latest RTMP raw frame JPEG preview |
| `GET` | `/api/satnav/media/model_input_img` | Most recent inference 448×448 model input JPEG |

---

## 1. Health check

### `GET /api/satnav/health`

Checks whether the SatNav FastAPI process is alive. Does **not** check whether the model, RTMP, or flight control is ready.

#### Request

None (no Query, no Body).

#### Response

| Field | Type | Description |
|------|------|------|
| `status` | string | Fixed `"ok"` when the API process is healthy |
| `service` | string | Service name, fixed `"satnav-api"` |
| `version` | string | API version, currently `"0.1.9"` |
| `timestamp` | string | Response time (Shanghai timezone) |

#### Examples

**Request**

```http
GET /api/satnav/health HTTP/1.1
Host: 127.0.0.1:8000
```

**Response `200`**

```json
{
  "status": "ok",
  "service": "satnav-api",
  "version": "0.1.9",
  "timestamp": "2026-08-02T21:46:18.757010+08:00"
}
```

---

## 2. Model and component status

### `GET /api/satnav/system/model/status`

Returns combined status of the API, RTMP reader, and SwiftVLN model subprocess. The endpoint does **not** block waiting for model load to finish.

#### Request

None.

#### Response

| Field | Type | Description |
|------|------|------|
| `status` | string | Overall status: `starting` (loading), `ready` (model ready), `error` (model load failed or process exited) |
| `api.ready` | boolean | Whether the API process is available; fixed `true` |
| `rtmp` | object | RTMP reader config and runtime summary (see table below) |
| `model` | object | SwiftVLN deploy subprocess status (see table below) |
| `timestamp` | string | Response time |

**`rtmp` object**

| Field | Type | Description |
|------|------|------|
| `rtmp_url` | string | Environment variable `SATNAV_RTMP_URL` |
| `probe_timeout_s` | number | Connection timeout in seconds |
| `probe_method` | string | Fixed `"opencv"` |
| `reader_state` | string | `connecting` / `connected` / `error` / `stopped` / `idle` |
| `frame_sequence` | integer | Cumulative decoded frame sequence number |
| `last_error` | string \| null | Reader’s most recent error |

**`model` object**

| Field | Type | Description |
|------|------|------|
| `state` | string | `not_started` / `loading` / `ready` / `error` / `stopped` |
| `loaded` | boolean | Whether the model has finished loading |
| `process_running` | boolean | Whether the deploy subprocess is running |
| `model_name` | string \| null | `SATNAV_MODEL_NAME` |
| `checkpoint_path` | string \| null | Checkpoint path after ready |
| `gpu` | object \| null | GPU metadata (from deploy `ready` response) |
| `error` | string \| null | Failure reason |
| `started_at` | string \| null | Load start time |
| `ready_at` | string \| null | Ready time |
| `stderr_tail` | string[] | Optional recent stderr tail lines |

#### Examples

**Request**

```http
GET /api/satnav/system/model/status HTTP/1.1
Host: 127.0.0.1:8000
```

**Response `200` (model loading)**

```json
{
  "status": "starting",
  "api": { "ready": true },
  "rtmp": {
    "rtmp_url": "rtmp://127.0.0.1/live/satnav",
    "probe_timeout_s": 8.0,
    "probe_method": "opencv",
    "reader_state": "connecting",
    "frame_sequence": 0,
    "last_error": null
  },
  "model": {
    "state": "loading",
    "loaded": false,
    "process_running": true,
    "model_name": "swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-b1.0-pool-s2-noembed",
    "checkpoint_path": null,
    "gpu": null,
    "error": null,
    "started_at": "2026-08-02T21:40:00.000000+08:00",
    "ready_at": null
  },
  "timestamp": "2026-08-02T21:40:05.000000+08:00"
}
```

**Response `200` (all ready)**

```json
{
  "status": "ready",
  "api": { "ready": true },
  "rtmp": {
    "rtmp_url": "rtmp://127.0.0.1/live/satnav",
    "probe_timeout_s": 8.0,
    "probe_method": "opencv",
    "reader_state": "connected",
    "frame_sequence": 1284,
    "last_error": null
  },
  "model": {
    "state": "ready",
    "loaded": true,
    "process_running": true,
    "model_name": "swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-b1.0-pool-s2-noembed",
    "checkpoint_path": "/path/to/checkpoint-1",
    "gpu": { "cuda_visible_devices": "0" },
    "error": null,
    "started_at": "2026-08-02T21:40:00.000000+08:00",
    "ready_at": "2026-08-02T21:42:30.000000+08:00"
  },
  "timestamp": "2026-08-02T21:46:18.757010+08:00"
}
```

---

## 3. Model logs

### `GET /api/satnav/system/model/logs`

Incrementally pull SwiftVLN deploy subprocess logs (stdout / stderr / system).

#### Query params

| Param | Type | Required | Default | Description |
|------|------|------|------|------|
| `after_sequence` | integer | No | `0` | Return only logs with `sequence` greater than this value |
| `limit` | integer | No | `500` | Max entries per request, range `1`–`2000` |

#### Response

| Field | Type | Description |
|------|------|------|
| `logs` | array | Log records (ascending by `sequence`) |
| `latest_sequence` | integer | Maximum sequence in the current buffer |
| `has_more` | boolean | Whether more unpulled logs remain |

**Each `logs[]` entry**

| Field | Type | Description |
|------|------|------|
| `sequence` | integer | Monotonically increasing sequence |
| `timestamp` | string | Record time |
| `source` | string | Fixed `"swiftvln"` |
| `stream` | string | `stdout` / `stderr` / `system` |
| `level` | string | `info` / `error` |
| `message` | string | Log body |

#### Examples

**Request**

```http
GET /api/satnav/system/model/logs?after_sequence=0&limit=100 HTTP/1.1
Host: 127.0.0.1:8000
```

**Response `200`**

```json
{
  "logs": [
    {
      "sequence": 1,
      "timestamp": "2026-08-02T21:40:01.000000+08:00",
      "source": "swiftvln",
      "stream": "system",
      "level": "info",
      "message": "开始后台加载 SwiftVLN 模型"
    },
    {
      "sequence": 2,
      "timestamp": "2026-08-02T21:40:02.000000+08:00",
      "source": "swiftvln",
      "stream": "stderr",
      "level": "info",
      "message": "Loading checkpoint shards: 100%"
    }
  ],
  "latest_sequence": 42,
  "has_more": true
}
```

---

## 4. RTMP stream status

### `GET /api/satnav/system/rtmp/status`

Returns the **cached status** of the background RTMP reader created at API startup (milliseconds-level; does not re-`open` the stream).

> Note: The reader attempts one connection at API startup; if the push stream was not ready, it enters `error` and **does not auto-reconnect**. After the stream is ready, call `POST /api/satnav/system/rtmp/refresh` to reconnect without restarting the API.

#### Request

None.

#### Response

| Field | Type | Description |
|------|------|------|
| `rtmp_url` | string | Configured RTMP URL |
| `connected` | boolean | Whether OpenCV successfully `open`ed and decoded at least one frame |
| `stream_active` | boolean | Same as `connected`; indicates frames can be captured |
| `probe_method` | string | Fixed `"opencv"` |
| `checked_at` | string | Query time |
| `video` | object | Optional; present when stream is active |
| `video.codec` | string | Fixed `"unknown"` |
| `video.width` | integer | Latest frame width (pixels) |
| `video.height` | integer | Latest frame height (pixels) |
| `error` | string | Optional reason when not connected |

**Common `error` values**

| Value | Meaning |
|----|------|
| `RTMP reader is still connecting` | First connection in progress |
| `unable to open RTMP stream` | `VideoCapture.open()` failed |
| `RTMP frame read failed N consecutive time(s)` | Connected but consecutive frame reads failed |
| `RTMP URL must start with rtmp:// or rtmps://` | Invalid URL scheme |

#### Examples

**Request**

```http
GET /api/satnav/system/rtmp/status HTTP/1.1
Host: 127.0.0.1:8000
```

**Response `200` (healthy)**

```json
{
  "rtmp_url": "rtmp://127.0.0.1/live/satnav",
  "connected": true,
  "stream_active": true,
  "probe_method": "opencv",
  "checked_at": "2026-08-02T21:46:18.757010+08:00",
  "video": {
    "codec": "unknown",
    "width": 1920,
    "height": 1080
  }
}
```

**Response `200` (connection failed)**

```json
{
  "rtmp_url": "rtmp://127.0.0.1/live/satnav",
  "connected": false,
  "stream_active": false,
  "probe_method": "opencv",
  "checked_at": "2026-08-02T21:46:18.757010+08:00",
  "error": "unable to open RTMP stream"
}
```

---

### `POST /api/satnav/system/rtmp/refresh`

Stops the current background RTMP reader and re-`open`s the pull stream. Use when the API started before RTMP push, or when the initial connection failed but the stream is now ready.

The endpoint blocks until this reconnect attempt completes, up to `SATNAV_RTMP_REFRESH_WAIT_SECONDS` (default `SATNAV_RTMP_PROBE_TIMEOUT_SECONDS + 2` seconds), then returns the same structure as `GET .../rtmp/status`, plus:

| Field | Type | Description |
|------|------|------|
| `refreshed` | boolean | Fixed `true` |
| `reader_state` | string | Reader state after refresh: `connecting` / `connected` / `error` |

#### Request

None.

#### Examples

**Request**

```http
POST /api/satnav/system/rtmp/refresh HTTP/1.1
Host: 127.0.0.1:8000
```

**Response `200` (connected after refresh)**

```json
{
  "rtmp_url": "rtmp://127.0.0.1/live/satnav",
  "connected": true,
  "stream_active": true,
  "probe_method": "opencv",
  "checked_at": "2026-08-03T09:56:00.000000+08:00",
  "video": {
    "codec": "unknown",
    "width": 1920,
    "height": 1080
  },
  "refreshed": true,
  "reader_state": "connected"
}
```

**Response `200` (still failed after refresh)**

```json
{
  "rtmp_url": "rtmp://127.0.0.1/live/satnav",
  "connected": false,
  "stream_active": false,
  "probe_method": "opencv",
  "checked_at": "2026-08-03T09:56:00.000000+08:00",
  "error": "unable to open RTMP stream",
  "refreshed": true,
  "reader_state": "error"
}
```

---

## 5. Flight-control backend reachability

### `GET /api/satnav/system/flight-rc-backend/status`

Probes whether the flight-control backend is reachable via CloudSDK OpenAPI `api-docs`.

**Prerequisites**: Environment variable `SATNAV_BACKEND_HOST` must be set; returns `503` if not configured.

#### Request

None.

#### Response

| Field | Type | Description |
|------|------|------|
| `backend_host` | string | `SATNAV_BACKEND_HOST` |
| `backend_port` | integer | `SATNAV_BACKEND_PORT`, default `6789` |
| `api_docs_path` | string | Probe path, default `/v3/api-docs` |
| `healthy` | boolean | Whether probe succeeded and service name matches |
| `probe_method` | string | Fixed `"api_docs"` |
| `checked_at` | string | Probe time |
| `status_code` | integer | Optional HTTP status code |
| `service_title` | string | Optional OpenAPI `info.title` |
| `error` | string | Optional reason when `healthy=false` |

#### Examples

**Request**

```http
GET /api/satnav/system/flight-rc-backend/status HTTP/1.1
Host: 127.0.0.1:8000
```

**Response `200` (healthy)**

```json
{
  "backend_host": "127.0.0.1",
  "backend_port": 6789,
  "api_docs_path": "/v3/api-docs",
  "healthy": true,
  "probe_method": "api_docs",
  "status_code": 200,
  "service_title": "CloudSDK API",
  "checked_at": "2026-08-02T21:46:18.757010+08:00"
}
```

**Response `503` (host not configured)**

```json
{
  "detail": "SATNAV_BACKEND_HOST is not configured"
}
```

---

## 6. Flight-control backend login / refresh token

### `POST /api/satnav/system/flight-rc-backend/login`

Logs into CloudSDK or refreshes the cached `x-auth-token`. The token is stored in **API process memory**, not on disk; cleared when the API exits.

**Behavior rules**

| Scenario | Behavior |
|------|------|
| No token in memory | Must provide `username` + `password`; calls `POST /manage/api/v1/login` |
| Token already in memory | Ignores username/password in body; calls `POST /manage/api/v1/token/refresh` |
| Refresh returns 401 | Clears cache, returns `400`; login required again |

#### JSON Body

| Field | Type | Required | Default | Description |
|------|------|------|------|------|
| `username` | string \| null | Required when no token | `null` | CloudSDK username |
| `password` | string \| null | Required when no token | `null` | CloudSDK password |
| `flag` | integer | No | `1` | Login flag forwarded to backend; must be `>= 1` |

#### Response

| Field | Type | Description |
|------|------|------|
| `auth_method` | string | `"login"` or `"refresh"` |
| `logged_in` | boolean | Fixed `true` on success |
| `workspace_id` | string \| null | Workspace ID from login response, cached server-side (same store as device registration) |
| `backend` | object | Raw CloudSDK JSON response |
| `backend.code` | integer | `0` means success |
| `backend.message` | string | e.g. `"success"` |
| `backend.data` | object | Includes `access_token`, `user_id`, `username`, etc. |
| `timestamp` | string | Completion time |

> The API does **not** return `x-auth-token` separately in the response; the token and `workspace_id` are cached server-side only for subsequent flight-control calls. Endpoints that need `workspace_id` (e.g. DRC) read this cache first; if not logged in, returns `请先登录飞控系统（workspace_id 未缓存）`.

#### Examples

**Request (first login)**

```http
POST /api/satnav/system/flight-rc-backend/login HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "username": "adminPC",
  "password": "adminPC",
  "flag": 1
}
```

**Response `200`**

```json
{
  "auth_method": "login",
  "logged_in": true,
  "workspace_id": "00000000-0000-4000-8000-000000000001",
  "backend": {
    "code": 0,
    "message": "success",
    "data": {
      "user_id": "user-1",
      "username": "adminPC",
      "workspace_id": "00000000-0000-4000-8000-000000000001",
      "user_type": 1,
      "mqtt_username": "admin",
      "mqtt_addr": "mqtt://host:1883",
      "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    }
  },
  "timestamp": "2026-08-02T21:46:18.757010+08:00"
}
```

**Request (token exists, refresh)**

```http
POST /api/satnav/system/flight-rc-backend/login HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{}
```

**Response `200`**

```json
{
  "auth_method": "refresh",
  "logged_in": true,
  "workspace_id": "00000000-0000-4000-8000-000000000001",
  "backend": {
    "code": 0,
    "message": "success",
    "data": {
      "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
      "username": "adminPC"
    }
  },
  "timestamp": "2026-08-02T21:50:00.000000+08:00"
}
```

**Response `400` (no token and no credentials)**

```json
{
  "detail": "username and password are required when no cached x-auth-token exists"
}
```

---

## 7. Register flight-control device SN

### `POST /api/satnav/system/flight-rc-backend/register_device`

Called after the frontend dialog collects RC SN and aircraft device SN. Caches both in **API process memory** (not on disk). Subsequent “acquire flight control” reads these values automatically.

**Prerequisites**: `SATNAV_BACKEND_HOST` must be configured (same as login).

#### JSON Body

| Field | Type | Required | Description |
|------|------|------|------|
| `rc_sn` | string | Yes | RC Plus 2 remote controller serial number |
| `device_sn` | string | Yes | Aircraft device serial number |

#### Response

| Field | Type | Description |
|------|------|------|
| `registered` | boolean | Fixed `true` |
| `workspace_id` | string \| null | Current cached workspace ID (from login) |
| `client_id` | string \| null | Cached DRC `client_id` if flight control was acquired |
| `drc_ready` | boolean | `true` when `client_id` is cached |
| `rc_sn` | string | Registered RC SN |
| `device_sn` | string | Registered aircraft SN |
| `registered_at` | string | Registration time |
| `timestamp` | string | Response time |

#### Examples

**Request**

```http
POST /api/satnav/system/flight-rc-backend/register_device HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "rc_sn": "RCPLUS2_SN_EXAMPLE",
  "device_sn": "AIRCRAFT_SN_EXAMPLE"
}
```

**Response `200`**

```json
{
  "registered": true,
  "workspace_id": "00000000-0000-4000-8000-000000000001",
  "client_id": null,
  "drc_ready": false,
  "rc_sn": "RCPLUS2_SN_EXAMPLE",
  "device_sn": "AIRCRAFT_SN_EXAMPLE",
  "registered_at": "2026-08-02T22:30:00.000000+08:00",
  "timestamp": "2026-08-02T22:30:00.000000+08:00"
}
```

---

## 8. Query flight-control device SN

### `GET /api/satnav/system/flight-rc-backend/cur_device_info`

Returns cached `workspace_id` (from login) and the **most recently registered** `rc_sn` and `device_sn` in the API process.

#### Request

None.

#### Response

| Field | Type | Description |
|------|------|------|
| `logged_in` | boolean | `true` when `workspace_id` is cached |
| `workspace_id` | string \| null | Workspace ID from login |
| `registered` | boolean | `true` when both `rc_sn` and `device_sn` are registered |
| `rc_sn` | string \| null | RC SN; `null` if not registered |
| `device_sn` | string \| null | Aircraft SN; `null` if not registered |
| `client_id` | string \| null | DRC connect cached `client_id`; `null` if flight control not acquired |
| `drc_ready` | boolean | `true` when `client_id` is cached |
| `registered_at` | string \| null | Most recent registration time |
| `timestamp` | string | Response time |

#### Examples

**Request**

```http
GET /api/satnav/system/flight-rc-backend/cur_device_info HTTP/1.1
Host: 127.0.0.1:8000
```

**Response `200` (registered)**

```json
{
  "logged_in": true,
  "workspace_id": "00000000-0000-4000-8000-000000000001",
  "registered": true,
  "rc_sn": "RCPLUS2_SN_EXAMPLE",
  "device_sn": "AIRCRAFT_SN_EXAMPLE",
  "client_id": "drc-client-abc123",
  "drc_ready": true,
  "registered_at": "2026-08-02T22:30:00.000000+08:00",
  "timestamp": "2026-08-02T22:35:00.000000+08:00"
}
```

**Response `200` (not registered)**

```json
{
  "logged_in": false,
  "workspace_id": null,
  "registered": false,
  "rc_sn": null,
  "device_sn": null,
  "client_id": null,
  "drc_ready": false,
  "registered_at": null,
  "timestamp": "2026-08-02T22:35:00.000000+08:00"
}
```

---

## 9. Acquire flight control (DRC)

### `POST /api/satnav/system/flight-rc-backend/drc`

Maps to the frontend “**Acquire flight control**” button. Uses login-cached `x-auth-token` and **§7 registered `rc_sn` / `device_sn`**, and **serially** calls the flight-control backend:

1. `POST .../drc/connect`
2. `POST .../drc/enter` (`rc_sn` from device registration cache)

**Prerequisites**

1. `POST /api/satnav/system/flight-rc-backend/login` succeeded (cached `workspace_id`)
2. `POST /api/satnav/system/flight-rc-backend/register_device` registered `rc_sn` / `device_sn`

#### Related config

| Variable | Default | Description |
|------|------|------|
| `SATNAV_BACKEND_HOST` | — | Flight-control backend address (required) |
| `SATNAV_BACKEND_PORT` | `6789` | Flight-control backend port |

> `workspace_id` is **no longer** set via environment variable; always read from login response cache.  
> After DRC connect succeeds, `client_id` is written to process cache for §10–§11 flight action endpoints.

#### JSON Body

| Field | Type | Required | Default | Description |
|------|------|------|------|------|
| `expire_sec` | integer | No | `3600` | Session TTL (seconds), range `[1800, 86400]` |

> No need to pass `rc_sn`; read from device registration cache. `device_sn` validates registration and is returned in the response for subsequent flight-control API use.

#### Response

| Field | Type | Description |
|------|------|------|
| `drc_ready` | boolean | Fixed `true` on success |
| `client_id` | string | `data.client_id` from connect |
| `rc_sn` | string | RC SN from registration cache |
| `device_sn` | string | Aircraft SN from registration cache |
| `expire_sec` | integer | Session TTL |
| `workspace_id` | string | Workspace ID |
| `connect` | object | Raw backend `drc/connect` response |
| `enter` | object | Raw backend `drc/enter` response |
| `pub` | string \| array \| object \| null | `data.pub` from enter response |
| `sub` | string \| array \| object \| null | `data.sub` from enter response |
| `timestamp` | string | Completion time |

#### Backend call details (SatNav internal serial flow)

**Step 1 — connect**

```http
POST http://{SATNAV_BACKEND_HOST}:{SATNAV_BACKEND_PORT}/control/api/v1/pilot/rc-plus-2/workspaces/{workspace_id}/drc/connect
x-auth-token: <login cached access_token>
Content-Type: application/json

{"expire_sec": 3600}
```

**Step 2 — enter** (`rc_sn` from §7 registration cache)

```http
POST http://{SATNAV_BACKEND_HOST}:{SATNAV_BACKEND_PORT}/control/api/v1/pilot/rc-plus-2/workspaces/{workspace_id}/drc/enter
x-auth-token: <login cached access_token>
Content-Type: application/json

{
  "client_id": "<connect.data.client_id>",
  "rc_sn": "<cached rc_sn>",
  "expire_sec": 3600
}
```

After entering DRC mode, the RC shows a confirmation dialog; control is available only after user confirmation.

#### Examples

**Request**

```http
POST /api/satnav/system/flight-rc-backend/drc HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "expire_sec": 3600
}
```

**Response `200`**

```json
{
  "drc_ready": true,
  "client_id": "drc-client-abc123",
  "rc_sn": "RCPLUS2_SN_EXAMPLE",
  "device_sn": "AIRCRAFT_SN_EXAMPLE",
  "expire_sec": 3600,
  "workspace_id": "00000000-0000-4000-8000-000000000001",
  "connect": {
    "code": 0,
    "message": "success",
    "data": {
      "client_id": "drc-client-abc123"
    }
  },
  "enter": {
    "code": 0,
    "message": "success",
    "data": {
      "pub": ["thing/product/xxx/drc/down"],
      "sub": ["thing/product/xxx/drc/up"]
    }
  },
  "pub": ["thing/product/xxx/drc/down"],
  "sub": ["thing/product/xxx/drc/up"],
  "timestamp": "2026-08-02T22:14:00.000000+08:00"
}
```

**Response `400` (not logged in, no workspace_id)**

```json
{
  "detail": "请先登录飞控系统（workspace_id 未缓存）"
}
```

**Response `400` (device not registered)**

```json
{
  "detail": "rc_sn, device_sn not registered; call POST /api/satnav/system/flight-rc-backend/register_device first"
}
```

**Response `400` (login not done first)**

```json
{
  "detail": "x-auth-token is not available; login first"
}
```

**Response `400` (backend business failure)**

```json
{
  "detail": "some backend error (code=10001)"
}
```

---

## 10. Execute forward (FORWARD)

### `POST /api/satnav/system/flight-rc-backend/flight/forward`

Maps to model action `next_action=1` (FORWARD); calls flight-control backend `pitch-by-distance` (CloudSDK 3.6).

**Prerequisites**

1. `POST .../login` succeeded (cached `workspace_id`)
2. `POST .../register_device` registered `rc_sn` / `device_sn`
3. `POST .../drc` succeeded (cached `client_id`)

#### Related config (environment variables, all have defaults)

| Variable | Default | Description |
|------|------|------|
| `SATNAV_FLIGHT_DEFAULT_FORWARD_DISTANCE_M` | `10` | Default forward distance (meters) |
| `SATNAV_FLIGHT_DEFAULT_FORWARD_TOLERANCE_M` | `0.3` | Default forward tolerance (meters) |
| `SATNAV_FLIGHT_DEFAULT_TIMEOUT_MS` | `30000` | Default Stick task timeout (milliseconds) |

#### JSON Body

All fields optional; omitted values use environment variable defaults above.

| Field | Type | Required | Description |
|------|------|------|------|
| `distance_m` | number | No | Forward distance (meters) |
| `tolerance_m` | number | No | Arrival tolerance (meters), range `[0, 50]` |
| `timeout_ms` | integer | No | Task timeout (milliseconds), range `[1000, 300000]` |

Empty body `{}` is valid and uses all defaults.

#### Response

| Field | Type | Description |
|------|------|------|
| `action` | integer | Fixed `1` (FORWARD) |
| `task_id` | string | Stick closed-loop task ID for §12 polling |
| `status` | string | Status at submission, usually `PENDING` |
| `parameters` | object | Actual parameters used (`distance_m` / `tolerance_m` / `timeout_ms`) |
| `backend` | object | Raw flight-control backend response |
| `timestamp` | string | Response time |

#### Examples

**Request (default 10m / 0.3m / 30s)**

```http
POST /api/satnav/system/flight-rc-backend/flight/forward HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{}
```

**Response `200`**

```json
{
  "action": 1,
  "task_id": "stick-task-abc123",
  "status": "PENDING",
  "parameters": {
    "distance_m": 10,
    "tolerance_m": 0.3,
    "timeout_ms": 30000
  },
  "backend": {
    "code": 0,
    "message": "success",
    "data": {
      "task_id": "stick-task-abc123",
      "status": "PENDING"
    }
  },
  "timestamp": "2026-08-03T10:00:00.000000+08:00"
}
```

**Response `400` (flight control not acquired)**

```json
{
  "detail": "请先获取飞行控制（client_id 未缓存）"
}
```

---

## 11. Execute turn (TURN_LEFT / TURN_RIGHT)

### `POST /api/satnav/system/flight-rc-backend/flight/turn`

Maps to model action `next_action=2` (turn left) or `3` (turn right); calls flight-control backend `yaw-by-degree` (CloudSDK 3.7).

Uses **scheme A**: `action` aligns with inference `next_action`; `degree` is **absolute**, sign determined by `action` (`2` → negative angle left, `3` → positive angle right).

**Prerequisites**: Same as §10.

#### Related config (environment variables, all have defaults)

| Variable | Default | Description |
|------|------|------|
| `SATNAV_FLIGHT_DEFAULT_YAW_DEG` | `15` | Default turn angle (absolute, degrees) |
| `SATNAV_FLIGHT_DEFAULT_YAW_TOLERANCE_DEG` | `0.2` | Default turn tolerance (degrees) |
| `SATNAV_FLIGHT_DEFAULT_TIMEOUT_MS` | `30000` | Default Stick task timeout (milliseconds) |

#### JSON Body

| Field | Type | Required | Description |
|------|------|------|------|
| `action` | integer | Yes | `2`=TURN_LEFT, `3`=TURN_RIGHT |
| `degree` | number | No | Absolute turn angle (degrees); default `15` if omitted |
| `tolerance_deg` | number | No | Arrival tolerance (degrees), range `[0, 10]` |
| `timeout_ms` | integer | No | Task timeout (milliseconds), range `[1000, 300000]` |

#### Response

| Field | Type | Description |
|------|------|------|
| `action` | integer | `2` or `3` |
| `task_id` | string | Stick closed-loop task ID |
| `status` | string | Status at submission |
| `parameters` | object | `degree` (absolute), `signed_degree` (actually sent), `tolerance_deg`, `timeout_ms` |
| `backend` | object | Raw flight-control backend response |
| `timestamp` | string | Response time |

#### Examples

**Request (turn right 15°, aligned with `next_action=3`)**

```http
POST /api/satnav/system/flight-rc-backend/flight/turn HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "action": 3,
  "degree": 15
}
```

**Response `200`**

```json
{
  "action": 3,
  "task_id": "stick-task-def456",
  "status": "PENDING",
  "parameters": {
    "degree": 15,
    "signed_degree": 15,
    "tolerance_deg": 0.2,
    "timeout_ms": 30000
  },
  "backend": {
    "code": 0,
    "message": "success",
    "data": {
      "task_id": "stick-task-def456",
      "status": "PENDING"
    }
  },
  "timestamp": "2026-08-03T10:01:00.000000+08:00"
}
```

**Turn left example**: `{"action": 2}` (omit `degree` for default 15°, actually sends `signed_degree=-15`).

---

## 12. Query Stick closed-loop task status

### `GET /api/satnav/system/flight-rc-backend/flight/stick-task/{task_id}`

Polls `task_id` returned from §10 / §11; maps to CloudSDK 3.9. Recommended: wait until `status` is `COMPLETED` or `FAILED` before calling `POST .../model/inference` for the next feedback frame.

**Prerequisites**: `login` succeeded (needs `workspace_id`); `client_id` not required.

#### Path params

| Param | Type | Description |
|------|------|------|
| `task_id` | string | Task ID from §10 or §11 |

#### Response

| Field | Type | Description |
|------|------|------|
| `task_id` | string | Task ID |
| `kind` | string | Task type, e.g. `YAW` / `PITCH` |
| `status` | string | `PENDING` / `RUNNING` / `COMPLETED` / `FAILED`, etc. |
| `backend` | object | Raw flight-control backend response |
| `timestamp` | string | Response time |

#### Examples

**Request**

```http
GET /api/satnav/system/flight-rc-backend/flight/stick-task/stick-task-abc123 HTTP/1.1
Host: 127.0.0.1:8000
```

**Response `200`**

```json
{
  "task_id": "stick-task-abc123",
  "kind": "PITCH",
  "status": "COMPLETED",
  "backend": {
    "code": 0,
    "message": "success",
    "data": {
      "task_id": "stick-task-abc123",
      "kind": "PITCH",
      "status": "COMPLETED"
    }
  },
  "timestamp": "2026-08-03T10:02:00.000000+08:00"
}
```

#### Recommended closed-loop sequence

```
login → register_device → drc (cache client_id)
  → inference (get next_action)
  → forward / turn (get task_id)
  → poll stick-task until COMPLETED
  → inference (feedback frame) → …
```

> `next_action=0` (STOP) has no corresponding flight-control REST endpoint; the frontend ends the current step when received.

---

## 12.1 Get current OSD info (LiveStore snapshot)

### `GET /api/satnav/system/flight-rc-backend/flight/osd/latest`

Proxies CloudSDK `POST .../flight/osd/latest`; returns the latest OSD snapshot for the aircraft in LiveStore.

**Prerequisites** (same as forward / turn):

1. `POST .../login` succeeded (cached `workspace_id`)
2. `POST .../register_device` registered `rc_sn` / `device_sn`
3. `POST .../drc` succeeded (cached `client_id`)

If `drc/enter` was not completed or `osd_info_push` has not arrived, the flight-control backend returns `code != 0`; this endpoint forwards `detail` as `400`.

#### Request

None (`client_id`, `rc_sn`, `device_sn` injected from server cache).

#### Response

| Field | Type | Description |
|------|------|------|
| `device_sn` | string | Aircraft SN |
| `attitude_head` | number | Heading (degrees) |
| `latitude` | number | Latitude |
| `longitude` | number | Longitude |
| `height` | number | Height above ground (meters) |
| `speed_x` / `speed_y` / `speed_z` | number | Velocity components |
| `gimbal_pitch` / `gimbal_roll` / `gimbal_yaw` | number | Gimbal attitude |
| `received_at_ms` | integer | OSD push receive time (Unix milliseconds) |
| `age_ms` | integer | Snapshot age (milliseconds) |
| `backend` | object | Raw flight-control backend response |
| `timestamp` | string | This API response time |

#### Examples

**Request**

```http
GET /api/satnav/system/flight-rc-backend/flight/osd/latest HTTP/1.1
Host: 127.0.0.1:8000
```

**Response `200`**

```json
{
  "device_sn": "AIRCRAFT_SN_EXAMPLE",
  "attitude_head": 42.5,
  "latitude": 22.6070293,
  "longitude": 114.0561159,
  "height": 25.3,
  "speed_x": 0.1,
  "speed_y": 0.0,
  "speed_z": 0.0,
  "gimbal_pitch": -10.0,
  "gimbal_roll": 0.5,
  "gimbal_yaw": 15.0,
  "received_at_ms": 1743518000456,
  "age_ms": 120,
  "backend": {
    "code": 0,
    "message": "success",
    "data": {
      "device_sn": "AIRCRAFT_SN_EXAMPLE",
      "attitude_head": 42.5,
      "latitude": 22.6070293,
      "longitude": 114.0561159,
      "height": 25.3,
      "received_at_ms": 1743518000456,
      "age_ms": 120
    }
  },
  "timestamp": "2026-08-03T19:26:27.000000+08:00"
}
```

**Response `400` (no OSD cache yet)**

```json
{
  "detail": "No LiveStore OSD snapshot for device_sn=.... Complete drc/enter and wait for osd_info_push."
}
```

---

## 13. Model inference (RTMP capture + Deploy image)

### `POST /api/satnav/model/inference`

Runs one full perception step:

1. Check model `ready`, RTMP `stream_active`
2. If deploy session not started → send `start` (with `instruction`)
3. Wait for one **new** RTMP frame (`request_fresh_frame`)
4. Preprocess to 448×448 RGB JPEG (default `center-crop`)
5. Send `image` JSONL command to deploy subprocess
6. Return deploy response and timing

Whether **real model inference** runs is determined by the deploy session internal queue (same as `test_rtmp_model_latency.py`):

| Scenario | `performed_inference` |
|------|------------------------|
| Action queue empty (usually first frame) | `true` |
| Queue still has unconsumed actions | `false` (save image only and pop completed action) |

**Session constraints**

- Within the same API lifetime, if session already `start`ed and `instruction` differs from this request → `400`
- Do not call deploy `end` (terminates model subprocess)

#### JSON Body

| Field | Type | Required | Description |
|------|------|------|------|
| `instruction` | string | Yes | Navigation natural-language instruction; trimmed non-empty, min length 1 |

#### Response

| Field | Type | Description |
|------|------|------|
| `instruction` | string | Instruction actually used (after trim) |
| `session_id` | string | Deploy session ID |
| `started_new_session` | boolean | Whether this request sent a new `start` |
| `performed_inference` | boolean | Whether this `image` triggered real model inference |
| `raw_action_text` | string | Raw model action text, e.g. `"0 1 2 3"` (see action codes) |
| `actions` | integer[] | Actions generated this inference; elements `0`–`3` (non-empty only when `performed_inference=true`) |
| `next_action` | integer \| null | Head of queue to execute (`0`–`3`) |
| `remaining_actions` | integer[] | Remaining action queue (`0`–`3`) |
| `completed_action` | integer \| null | Action consumed as feedback this step (`0`–`3`; `null` on first frame) |
| `deploy_state` | string | Deploy session state, e.g. `waiting_feedback` / `waiting_image` |
| `frame_sequence` | integer | RTMP frame sequence used this step |
| `image_path` | string | Absolute path of saved 448×448 JPEG |
| `timing` | object | Per-stage timing (milliseconds) |
| `timing.capture_ms` | number | Wait and capture new RTMP frame |
| `timing.preprocess_ms` | number | BGR → 448×448 RGB |
| `timing.input_save_ms` | number | JPEG write to disk |
| `timing.deploy_response_ms` | number | Send `image` until deploy response |
| `timing.capture_to_action_response_ms` | number | End-to-end from requesting new frame to deploy response |
| `timestamp` | string | Completion time |

#### Related environment variables

| Variable | Default | Description |
|------|------|------|
| `SATNAV_RTMP_FRAME_TIMEOUT_SECONDS` | `10` | New frame wait timeout |
| `SATNAV_MODEL_DEPLOY_RESPONSE_TIMEOUT_SECONDS` | `120` | Deploy JSONL response timeout |
| `SATNAV_MODEL_INPUT_RESIZE_MODE` | `center-crop` | Or `stretch` |
| `SATNAV_MODEL_INFERENCE_INPUT_ROOT` | `{session_root}/inference_inputs` | JPEG output directory |

#### Examples

**Request (first frame, triggers real inference)**

```http
POST /api/satnav/model/inference HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "instruction": "Fly to the lake and fly around the lake clockwise"
}
```

**Response `200`**

```json
{
  "instruction": "Fly to the lake and fly around the lake clockwise",
  "session_id": "satnav-infer-a1b2c3d4",
  "started_new_session": true,
  "performed_inference": true,
  "raw_action_text": "0 1 2 3",
  "actions": [0, 1, 2, 3],
  "next_action": 0,
  "remaining_actions": [0, 1, 2, 3],
  "completed_action": null,
  "deploy_state": "waiting_feedback",
  "frame_sequence": 1290,
  "image_path": "/path/to/SwiftVLN/satnav/runtime/model_sessions/inference_inputs/infer_f3e2d1c0b9a8_448.jpg",
  "timing": {
    "capture_ms": 45.231,
    "preprocess_ms": 12.104,
    "input_save_ms": 3.552,
    "deploy_response_ms": 856.443,
    "capture_to_action_response_ms": 917.330
  },
  "timestamp": "2026-08-02T21:46:18.757010+08:00"
}
```

**Request (queue still has actions, consume queue only)**

```http
POST /api/satnav/model/inference HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "instruction": "Fly to the lake and fly around the lake clockwise"
}
```

**Response `200`**

```json
{
  "instruction": "Fly to the lake and fly around the lake clockwise",
  "session_id": "satnav-infer-a1b2c3d4",
  "started_new_session": false,
  "performed_inference": false,
  "raw_action_text": "",
  "actions": [],
  "next_action": 1,
  "remaining_actions": [1, 2, 3],
  "completed_action": 0,
  "deploy_state": "waiting_feedback",
  "frame_sequence": 1295,
  "image_path": "/path/to/SwiftVLN/satnav/runtime/model_sessions/inference_inputs/infer_8a7b6c5d4e3f_448.jpg",
  "timing": {
    "capture_ms": 38.102,
    "preprocess_ms": 11.887,
    "input_save_ms": 3.201,
    "deploy_response_ms": 12.554,
    "capture_to_action_response_ms": 65.744
  },
  "timestamp": "2026-08-02T21:47:05.000000+08:00"
}
```

**Response `400` (model not ready)**

```json
{
  "detail": "model is not ready (state=loading)"
}
```

**Response `400` (RTMP not active)**

```json
{
  "detail": "unable to open RTMP stream"
}
```

**Response `400` (instruction conflicts with existing session)**

```json
{
  "detail": "deploy session already active with a different instruction"
}
```

**Response `422` (empty instruction)**

```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "instruction"],
      "msg": "String should have at least 1 character",
      "input": "",
      "ctx": { "min_length": 1 }
    }
  ]
}
```

**Response `504` (RTMP new frame timeout)**

```json
{
  "detail": "timed out waiting for a fresh RTMP frame: RTMP frame read failed 3 consecutive time(s)"
}
```

**Response `503` (deploy process not running)**

```json
{
  "detail": "deploy process is not running"
}
```

---

## 14. Raw RTMP frame preview

### `GET /api/satnav/media/raw_img`

Returns the **latest** JPEG of the background RTMP reader’s current frame for frontend live preview. Does not trigger inference or wait for a new frame.

#### Request

None.

#### Response

| Type | Description |
|------|------|
| `image/jpeg` | JPEG binary body |

Response headers:

| Header | Description |
|----|------|
| `X-Frame-Sequence` | Current frame sequence (same as RTMP reader) |
| `Cache-Control` | `no-store` |

#### Examples

**Request**

```http
GET /api/satnav/media/raw_img HTTP/1.1
Host: 127.0.0.1:8000
```

**Response `200`**

Body is JPEG binary (not JSON).

**Response `404` (no decoded frame yet)**

```json
{
  "detail": "no RTMP frame available yet"
}
```

---

## 15. Model input preview

### `GET /api/satnav/media/model_input_img`

Returns the **most recent** 448×448 RGB JPEG from `POST /api/satnav/model/inference` preprocessing (same image sent to deploy).

#### Request

None.

#### Response

| Type | Description |
|------|------|
| `image/jpeg` | 448×448 JPEG binary body |

Response headers:

| Header | Description |
|----|------|
| `X-Frame-Sequence` | Corresponding RTMP frame sequence |
| `X-Image-Path` | Local absolute path on disk |
| `X-Updated-At` | Cache update time |
| `Cache-Control` | `no-store` |

#### Examples

**Request**

```http
GET /api/satnav/media/model_input_img HTTP/1.1
Host: 127.0.0.1:8000
```

**Response `404` (inference not called yet)**

```json
{
  "detail": "no model input image available yet; call POST /api/satnav/model/inference first"
}
```

---

## Changelog

| Date | Version | Notes |
|------|------|------|
| 2026-08-02 | 0.1.0 | Initial release: 7 implemented REST endpoints |
| 2026-08-02 | 0.1.1 | Added `POST /api/satnav/system/flight-rc-backend/drc` (DRC acquire flight control) |
| 2026-08-02 | 0.1.3 | Added device SN register/query APIs; DRC reads registration cache |
| 2026-08-03 | 0.1.5 | Login caches `workspace_id`; flight endpoints read cache first; removed `SATNAV_BACKEND_WORKSPACE_ID` |
| 2026-08-03 | 0.1.6 | Added three flight action endpoints (forward / turn / stick-task); default 15°/10m and tolerances via env vars; DRC caches `client_id` |
| 2026-08-03 | 0.1.7 | Device API path rename: `register_device` (register), `cur_device_info` (query) |
| 2026-08-03 | 0.1.8 | Added `media/raw_img`, `media/model_input_img`; completed `requirements.txt` |
| 2026-08-03 | 0.1.9 | Added `SATNAV_CORS_ORIGINS`, default allows `http://127.0.0.1:5173` |
| 2026-08-03 | 0.2.0 | Added `GET .../flight/osd/latest` (LiveStore OSD snapshot) |

---

# 中文

服务版本：`0.1.9`  
默认 Base URL：`http://127.0.0.1:8000`（由 `SATNAV_API_HOST` / `SATNAV_API_PORT` 控制）

FastAPI 自带交互式文档：

- Swagger UI：`GET /docs`
- OpenAPI JSON：`GET /openapi.json`

本文档登记 **当前已实现** 的全部 REST 接口，含入参、出参说明与示例。

---

## 通用约定

| 项目 | 说明 |
|------|------|
| Content-Type | 请求体为 JSON 时使用 `application/json` |
| 时间戳 | 均为上海时区 ISO 8601 字符串，如 `2026-08-02T21:46:18.757010+08:00` |
| 错误响应 | FastAPI 标准格式：`{"detail": "错误描述"}` |
| CORS | 由 `SATNAV_CORS_ORIGINS` 配置允许的前端 Origin（逗号分隔）；默认 `http://127.0.0.1:5173` |
| Action 编码 | 见下方「动作编码表」；取值为 `0`–`3` 的整数 |

### 动作编码表

模型与 deploy session 返回的 `actions` / `next_action` / `remaining_actions` / `completed_action` 均使用下列整数编码：

| 值 | 名称 | 含义 |
|----|------|------|
| `0` | `STOP` | 停止 |
| `1` | `FORWARD` | 前进 |
| `2` | `TURN_LEFT` | 左转 |
| `3` | `TURN_RIGHT` | 右转 |

`raw_action_text` 为模型原始输出字符串，通常为空格分隔的编码序列，例如 `"0 1 2 3"`。

### HTTP 状态码（各接口可能返回）

| 状态码 | 含义 |
|--------|------|
| `200` | 成功 |
| `400` | 请求参数或业务前置条件不满足 |
| `422` | 请求体校验失败（Pydantic） |
| `503` | 依赖未配置或 deploy 进程不可用 |
| `504` | 等待 RTMP 帧或 deploy 响应超时 |

---

## 接口一览

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/satnav/health` | API 进程存活检查 |
| `GET` | `/api/satnav/system/model/status` | 模型加载与 RTMP reader 摘要状态 |
| `GET` | `/api/satnav/system/model/logs` | 增量获取模型子进程日志 |
| `GET` | `/api/satnav/system/rtmp/status` | RTMP 拉流连接状态 |
| `POST` | `/api/satnav/system/rtmp/refresh` | 刷新 RTMP 连接（重启后台 reader） |
| `GET` | `/api/satnav/system/flight-rc-backend/status` | 飞控 CloudSDK backend 可达性 |
| `POST` | `/api/satnav/system/flight-rc-backend/login` | 登录或刷新飞控 token |
| `POST` | `/api/satnav/system/flight-rc-backend/register_device` | 登记遥控器 `rc_sn` 与飞行器 `device_sn` |
| `GET` | `/api/satnav/system/flight-rc-backend/cur_device_info` | 查询当前登记的 `rc_sn` / `device_sn` |
| `POST` | `/api/satnav/system/flight-rc-backend/drc` | 获取飞行控制（DRC connect + enter） |
| `POST` | `/api/satnav/system/flight-rc-backend/flight/forward` | 执行前进（FORWARD / pitch-by-distance） |
| `POST` | `/api/satnav/system/flight-rc-backend/flight/turn` | 执行转向（TURN_LEFT / TURN_RIGHT） |
| `GET` | `/api/satnav/system/flight-rc-backend/flight/stick-task/{task_id}` | 查询 Stick 闭环任务状态 |
| `GET` | `/api/satnav/system/flight-rc-backend/flight/osd/latest` | 获取当前 OSD 信息（LiveStore 快照） |
| `POST` | `/api/satnav/model/inference` | RTMP 抽帧 + 一次 deploy image 步骤 |
| `GET` | `/api/satnav/media/raw_img` | 最新 RTMP 原始帧 JPEG 预览 |
| `GET` | `/api/satnav/media/model_input_img` | 最近一次推理的 448×448 模型输入 JPEG |

---

## 1. 健康检查

### `GET /api/satnav/health`

检查 SatNav FastAPI 进程是否存活。**不**检查模型、RTMP、飞控是否就绪。

#### 入参

无（无 Query、无 Body）。

#### 出参

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | 固定为 `"ok"` 表示 API 进程正常 |
| `service` | string | 服务名，固定 `"satnav-api"` |
| `version` | string | API 版本，当前 `"0.1.9"` |
| `timestamp` | string | 响应时间（上海时区） |

#### 示例

**请求**

```http
GET /api/satnav/health HTTP/1.1
Host: 127.0.0.1:8000
```

**响应 `200`**

```json
{
  "status": "ok",
  "service": "satnav-api",
  "version": "0.1.9",
  "timestamp": "2026-08-02T21:46:18.757010+08:00"
}
```

---

## 2. 模型与组件状态

### `GET /api/satnav/system/model/status`

返回 API、RTMP reader、SwiftVLN 模型子进程的综合状态。接口**不阻塞**等待模型加载完成。

#### 入参

无。

#### 出参

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | 整体状态：`starting`（加载中）、`ready`（模型就绪）、`error`（模型加载失败或进程退出） |
| `api.ready` | boolean | API 进程是否可用，固定 `true` |
| `rtmp` | object | RTMP reader 配置与运行时摘要（见下表） |
| `model` | object | SwiftVLN deploy 子进程状态（见下表） |
| `timestamp` | string | 响应时间 |

**`rtmp` 对象**

| 字段 | 类型 | 说明 |
|------|------|------|
| `rtmp_url` | string | 环境变量 `SATNAV_RTMP_URL` |
| `probe_timeout_s` | number | 建连超时秒数 |
| `probe_method` | string | 固定 `"opencv"` |
| `reader_state` | string | `connecting` / `connected` / `error` / `stopped` / `idle` |
| `frame_sequence` | integer | 已累计解码帧序号 |
| `last_error` | string \| null | reader 最近一次错误 |

**`model` 对象**

| 字段 | 类型 | 说明 |
|------|------|------|
| `state` | string | `not_started` / `loading` / `ready` / `error` / `stopped` |
| `loaded` | boolean | 模型是否加载完成 |
| `process_running` | boolean | deploy 子进程是否在运行 |
| `model_name` | string \| null | `SATNAV_MODEL_NAME` |
| `checkpoint_path` | string \| null | 就绪后的 checkpoint 路径 |
| `gpu` | object \| null | GPU 元数据（由 deploy `ready` 响应提供） |
| `error` | string \| null | 失败原因 |
| `started_at` | string \| null | 开始加载时间 |
| `ready_at` | string \| null | 就绪时间 |
| `stderr_tail` | string[] | 可选，最近 stderr 尾部行 |

#### 示例

**请求**

```http
GET /api/satnav/system/model/status HTTP/1.1
Host: 127.0.0.1:8000
```

**响应 `200`（模型加载中）**

```json
{
  "status": "starting",
  "api": { "ready": true },
  "rtmp": {
    "rtmp_url": "rtmp://127.0.0.1/live/satnav",
    "probe_timeout_s": 8.0,
    "probe_method": "opencv",
    "reader_state": "connecting",
    "frame_sequence": 0,
    "last_error": null
  },
  "model": {
    "state": "loading",
    "loaded": false,
    "process_running": true,
    "model_name": "swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-b1.0-pool-s2-noembed",
    "checkpoint_path": null,
    "gpu": null,
    "error": null,
    "started_at": "2026-08-02T21:40:00.000000+08:00",
    "ready_at": null
  },
  "timestamp": "2026-08-02T21:40:05.000000+08:00"
}
```

**响应 `200`（全部就绪）**

```json
{
  "status": "ready",
  "api": { "ready": true },
  "rtmp": {
    "rtmp_url": "rtmp://127.0.0.1/live/satnav",
    "probe_timeout_s": 8.0,
    "probe_method": "opencv",
    "reader_state": "connected",
    "frame_sequence": 1284,
    "last_error": null
  },
  "model": {
    "state": "ready",
    "loaded": true,
    "process_running": true,
    "model_name": "swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-b1.0-pool-s2-noembed",
    "checkpoint_path": "/path/to/checkpoint-1",
    "gpu": { "cuda_visible_devices": "0" },
    "error": null,
    "started_at": "2026-08-02T21:40:00.000000+08:00",
    "ready_at": "2026-08-02T21:42:30.000000+08:00"
  },
  "timestamp": "2026-08-02T21:46:18.757010+08:00"
}
```

---

## 3. 模型日志

### `GET /api/satnav/system/model/logs`

增量拉取 SwiftVLN deploy 子进程日志（stdout / stderr / system）。

#### 入参（Query）

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `after_sequence` | integer | 否 | `0` | 只返回 `sequence` 大于该值的日志 |
| `limit` | integer | 否 | `500` | 单次最多返回条数，范围 `1`–`2000` |

#### 出参

| 字段 | 类型 | 说明 |
|------|------|------|
| `logs` | array | 日志记录列表（按 `sequence` 升序） |
| `latest_sequence` | integer | 当前缓冲区中最大序号 |
| `has_more` | boolean | 是否还有未拉取的更新日志 |

**`logs[]` 每条记录**

| 字段 | 类型 | 说明 |
|------|------|------|
| `sequence` | integer | 单调递增序号 |
| `timestamp` | string | 记录时间 |
| `source` | string | 固定 `"swiftvln"` |
| `stream` | string | `stdout` / `stderr` / `system` |
| `level` | string | `info` / `error` |
| `message` | string | 日志正文 |

#### 示例

**请求**

```http
GET /api/satnav/system/model/logs?after_sequence=0&limit=100 HTTP/1.1
Host: 127.0.0.1:8000
```

**响应 `200`**

```json
{
  "logs": [
    {
      "sequence": 1,
      "timestamp": "2026-08-02T21:40:01.000000+08:00",
      "source": "swiftvln",
      "stream": "system",
      "level": "info",
      "message": "开始后台加载 SwiftVLN 模型"
    },
    {
      "sequence": 2,
      "timestamp": "2026-08-02T21:40:02.000000+08:00",
      "source": "swiftvln",
      "stream": "stderr",
      "level": "info",
      "message": "Loading checkpoint shards: 100%"
    }
  ],
  "latest_sequence": 42,
  "has_more": true
}
```

---

## 4. RTMP 流状态

### `GET /api/satnav/system/rtmp/status`

返回 API 启动时创建的后台 RTMP reader 的**缓存状态**（毫秒级，不重新 `open` 流）。

> 注意：reader 在 API 启动时尝试连接一次；若当时推流未就绪，会进入 `error` 且**不会自动重连**。推流就绪后请调用 `POST /api/satnav/system/rtmp/refresh` 重新连接，无需重启 API。

#### 入参

无。

#### 出参

| 字段 | 类型 | 说明 |
|------|------|------|
| `rtmp_url` | string | 配置的 RTMP 地址 |
| `connected` | boolean | OpenCV 是否已成功 `open` 且至少解码过一帧 |
| `stream_active` | boolean | 与 `connected` 同义，表示可用于抽帧 |
| `probe_method` | string | 固定 `"opencv"` |
| `checked_at` | string | 本次查询时间 |
| `video` | object | 可选，流已激活时存在 |
| `video.codec` | string | 固定 `"unknown"` |
| `video.width` | integer | 最近一帧宽度（像素） |
| `video.height` | integer | 最近一帧高度（像素） |
| `error` | string | 可选，未连接时的原因 |

**常见 `error` 值**

| 值 | 含义 |
|----|------|
| `RTMP reader is still connecting` | 正在首次连接 |
| `unable to open RTMP stream` | `VideoCapture.open()` 失败 |
| `RTMP frame read failed N consecutive time(s)` | 已连接但连续读帧失败 |
| `RTMP URL must start with rtmp:// or rtmps://` | URL  scheme 错误 |

#### 示例

**请求**

```http
GET /api/satnav/system/rtmp/status HTTP/1.1
Host: 127.0.0.1:8000
```

**响应 `200`（正常）**

```json
{
  "rtmp_url": "rtmp://127.0.0.1/live/satnav",
  "connected": true,
  "stream_active": true,
  "probe_method": "opencv",
  "checked_at": "2026-08-02T21:46:18.757010+08:00",
  "video": {
    "codec": "unknown",
    "width": 1920,
    "height": 1080
  }
}
```

**响应 `200`（连接失败）**

```json
{
  "rtmp_url": "rtmp://127.0.0.1/live/satnav",
  "connected": false,
  "stream_active": false,
  "probe_method": "opencv",
  "checked_at": "2026-08-02T21:46:18.757010+08:00",
  "error": "unable to open RTMP stream"
}
```

---

### `POST /api/satnav/system/rtmp/refresh`

停止当前后台 RTMP reader 并重新 `open` 拉流。适用于 API 先于 RTMP 推流启动、或启动时连接失败后推流已就绪的场景。

接口会阻塞等待本次重连结果，最长 `SATNAV_RTMP_REFRESH_WAIT_SECONDS`（默认 `SATNAV_RTMP_PROBE_TIMEOUT_SECONDS + 2` 秒），然后返回与 `GET .../rtmp/status` 相同结构的状态，并额外附带：

| 字段 | 类型 | 说明 |
|------|------|------|
| `refreshed` | boolean | 固定 `true` |
| `reader_state` | string | 刷新后的 reader 状态：`connecting` / `connected` / `error` |

#### 入参

无。

#### 示例

**请求**

```http
POST /api/satnav/system/rtmp/refresh HTTP/1.1
Host: 127.0.0.1:8000
```

**响应 `200`（刷新后已连接）**

```json
{
  "rtmp_url": "rtmp://127.0.0.1/live/satnav",
  "connected": true,
  "stream_active": true,
  "probe_method": "opencv",
  "checked_at": "2026-08-03T09:56:00.000000+08:00",
  "video": {
    "codec": "unknown",
    "width": 1920,
    "height": 1080
  },
  "refreshed": true,
  "reader_state": "connected"
}
```

**响应 `200`（刷新后仍失败）**

```json
{
  "rtmp_url": "rtmp://127.0.0.1/live/satnav",
  "connected": false,
  "stream_active": false,
  "probe_method": "opencv",
  "checked_at": "2026-08-03T09:56:00.000000+08:00",
  "error": "unable to open RTMP stream",
  "refreshed": true,
  "reader_state": "error"
}
```

---

## 5. 飞控 Backend 可达性

### `GET /api/satnav/system/flight-rc-backend/status`

通过 CloudSDK OpenAPI `api-docs` 探测飞控 backend 是否可达。

**前置条件**：需配置环境变量 `SATNAV_BACKEND_HOST`；未配置时返回 `503`。

#### 入参

无。

#### 出参

| 字段 | 类型 | 说明 |
|------|------|------|
| `backend_host` | string | `SATNAV_BACKEND_HOST` |
| `backend_port` | integer | `SATNAV_BACKEND_PORT`，默认 `6789` |
| `api_docs_path` | string | 探测路径，默认 `/v3/api-docs` |
| `healthy` | boolean | 探测是否成功且服务名匹配 |
| `probe_method` | string | 固定 `"api_docs"` |
| `checked_at` | string | 探测时间 |
| `status_code` | integer | 可选，HTTP 状态码 |
| `service_title` | string | 可选，OpenAPI `info.title` |
| `error` | string | 可选，`healthy=false` 时的原因 |

#### 示例

**请求**

```http
GET /api/satnav/system/flight-rc-backend/status HTTP/1.1
Host: 127.0.0.1:8000
```

**响应 `200`（健康）**

```json
{
  "backend_host": "127.0.0.1",
  "backend_port": 6789,
  "api_docs_path": "/v3/api-docs",
  "healthy": true,
  "probe_method": "api_docs",
  "status_code": 200,
  "service_title": "CloudSDK API",
  "checked_at": "2026-08-02T21:46:18.757010+08:00"
}
```

**响应 `503`（未配置 host）**

```json
{
  "detail": "SATNAV_BACKEND_HOST is not configured"
}
```

---

## 6. 飞控 Backend 登录 / 刷新 Token

### `POST /api/satnav/system/flight-rc-backend/login`

登录 CloudSDK 或刷新已缓存的 `x-auth-token`。Token 保存在 **API 进程内存**，不落盘；API 退出时清空。

**行为规则**

| 场景 | 行为 |
|------|------|
| 内存中无 token | 必须提供 `username` + `password`，调用 `POST /manage/api/v1/login` |
| 内存中已有 token | 忽略 body 中的账号密码，调用 `POST /manage/api/v1/token/refresh` |
| refresh 返回 401 | 清空缓存，返回 `400`，需重新 login |

#### 入参（JSON Body）

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `username` | string \| null | 无 token 时必填 | `null` | CloudSDK 用户名 |
| `password` | string \| null | 无 token 时必填 | `null` | CloudSDK 密码 |
| `flag` | integer | 否 | `1` | 登录标志，透传给 backend，须 `>= 1` |

#### 出参

| 字段 | 类型 | 说明 |
|------|------|------|
| `auth_method` | string | `"login"` 或 `"refresh"` |
| `logged_in` | boolean | 固定 `true`（成功时） |
| `workspace_id` | string \| null | 登录返回的 workspace ID，并缓存在服务端（与 device 登记同处） |
| `backend` | object | CloudSDK 原始 JSON 响应 |
| `backend.code` | integer | `0` 表示成功 |
| `backend.message` | string | 如 `"success"` |
| `backend.data` | object | 含 `access_token`、`user_id`、`username` 等 |
| `timestamp` | string | 完成时间 |

> API **不**在响应中单独返回 `x-auth-token` 字段；token 与 `workspace_id` 仅缓存在服务端，供后续飞控调用使用。后续需要 `workspace_id` 的接口（如 DRC）优先读取该缓存；未登录时返回 `请先登录飞控系统（workspace_id 未缓存）`。

#### 示例

**请求（首次登录）**

```http
POST /api/satnav/system/flight-rc-backend/login HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "username": "adminPC",
  "password": "adminPC",
  "flag": 1
}
```

**响应 `200`**

```json
{
  "auth_method": "login",
  "logged_in": true,
  "workspace_id": "00000000-0000-4000-8000-000000000001",
  "backend": {
    "code": 0,
    "message": "success",
    "data": {
      "user_id": "user-1",
      "username": "adminPC",
      "workspace_id": "00000000-0000-4000-8000-000000000001",
      "user_type": 1,
      "mqtt_username": "admin",
      "mqtt_addr": "mqtt://host:1883",
      "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    }
  },
  "timestamp": "2026-08-02T21:46:18.757010+08:00"
}
```

**请求（已有 token，刷新）**

```http
POST /api/satnav/system/flight-rc-backend/login HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{}
```

**响应 `200`**

```json
{
  "auth_method": "refresh",
  "logged_in": true,
  "workspace_id": "00000000-0000-4000-8000-000000000001",
  "backend": {
    "code": 0,
    "message": "success",
    "data": {
      "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
      "username": "adminPC"
    }
  },
  "timestamp": "2026-08-02T21:50:00.000000+08:00"
}
```

**响应 `400`（无 token 且未提供账号密码）**

```json
{
  "detail": "username and password are required when no cached x-auth-token exists"
}
```

---

## 7. 登记飞控设备 SN

### `POST /api/satnav/system/flight-rc-backend/register_device`

前端弹框录入遥控器 SN 与飞行器 device SN 后调用本接口，将二者缓存在 **API 进程内存**（不落盘）。后续「获取飞行控制」将自动读取此处登记的值。

**前置条件**：需配置 `SATNAV_BACKEND_HOST`（与 login 相同）。

#### 入参（JSON Body）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `rc_sn` | string | 是 | RC Plus 2 遥控器序列号 |
| `device_sn` | string | 是 | 飞行器设备序列号 |

#### 出参

| 字段 | 类型 | 说明 |
|------|------|------|
| `registered` | boolean | 固定 `true` |
| `workspace_id` | string \| null | 当前缓存的 workspace ID（来自 login） |
| `client_id` | string \| null | 已获取飞行控制时缓存的 DRC `client_id` |
| `drc_ready` | boolean | 已缓存 `client_id` 时为 `true` |
| `rc_sn` | string | 登记后的遥控器 SN |
| `device_sn` | string | 登记后的飞行器 SN |
| `registered_at` | string | 登记时间 |
| `timestamp` | string | 响应时间 |

#### 示例

**请求**

```http
POST /api/satnav/system/flight-rc-backend/register_device HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "rc_sn": "RCPLUS2_SN_EXAMPLE",
  "device_sn": "AIRCRAFT_SN_EXAMPLE"
}
```

**响应 `200`**

```json
{
  "registered": true,
  "workspace_id": "00000000-0000-4000-8000-000000000001",
  "client_id": null,
  "drc_ready": false,
  "rc_sn": "RCPLUS2_SN_EXAMPLE",
  "device_sn": "AIRCRAFT_SN_EXAMPLE",
  "registered_at": "2026-08-02T22:30:00.000000+08:00",
  "timestamp": "2026-08-02T22:30:00.000000+08:00"
}
```

---

## 8. 查询飞控设备 SN

### `GET /api/satnav/system/flight-rc-backend/cur_device_info`

返回当前 API 进程中缓存的 `workspace_id`（login）以及**最新一次登记**的 `rc_sn` 与 `device_sn`。

#### 入参

无。

#### 出参

| 字段 | 类型 | 说明 |
|------|------|------|
| `logged_in` | boolean | 已缓存 `workspace_id` 时为 `true` |
| `workspace_id` | string \| null | 来自 login 的 workspace ID |
| `registered` | boolean | `rc_sn` 与 `device_sn` 均已登记时为 `true` |
| `rc_sn` | string \| null | 遥控器 SN；未登记时为 `null` |
| `device_sn` | string \| null | 飞行器 SN；未登记时为 `null` |
| `client_id` | string \| null | DRC connect 缓存的 `client_id`；未获取飞行控制时为 `null` |
| `drc_ready` | boolean | 已缓存 `client_id` 时为 `true` |
| `registered_at` | string \| null | 最近登记时间 |
| `timestamp` | string | 响应时间 |

#### 示例

**请求**

```http
GET /api/satnav/system/flight-rc-backend/cur_device_info HTTP/1.1
Host: 127.0.0.1:8000
```

**响应 `200`（已登记）**

```json
{
  "logged_in": true,
  "workspace_id": "00000000-0000-4000-8000-000000000001",
  "registered": true,
  "rc_sn": "RCPLUS2_SN_EXAMPLE",
  "device_sn": "AIRCRAFT_SN_EXAMPLE",
  "client_id": "drc-client-abc123",
  "drc_ready": true,
  "registered_at": "2026-08-02T22:30:00.000000+08:00",
  "timestamp": "2026-08-02T22:35:00.000000+08:00"
}
```

**响应 `200`（未登记）**

```json
{
  "logged_in": false,
  "workspace_id": null,
  "registered": false,
  "rc_sn": null,
  "device_sn": null,
  "client_id": null,
  "drc_ready": false,
  "registered_at": null,
  "timestamp": "2026-08-02T22:35:00.000000+08:00"
}
```

---

## 9. 获取飞行控制（DRC）

### `POST /api/satnav/system/flight-rc-backend/drc`

前端按钮「**获取飞行控制**」对应此接口。使用 login 缓存的 `x-auth-token`，以及 **§7 登记的 `rc_sn` / `device_sn`**，**串行**调用飞控 backend：

1. `POST .../drc/connect`
2. `POST .../drc/enter`（`rc_sn` 来自设备登记缓存）

**前置条件**

1. `POST /api/satnav/system/flight-rc-backend/login` 成功（缓存 `workspace_id`）
2. `POST /api/satnav/system/flight-rc-backend/register_device` 已登记 `rc_sn` / `device_sn`

#### 相关配置

| 变量 | 默认 | 说明 |
|------|------|------|
| `SATNAV_BACKEND_HOST` | — | 飞控 backend 地址（必填） |
| `SATNAV_BACKEND_PORT` | `6789` | 飞控 backend 端口 |

> `workspace_id` **不再**通过环境变量配置，统一从 login 响应缓存读取。  
> DRC connect 成功后，`client_id` 会写入进程缓存，供 §10–§11 飞行动作接口使用。

#### 入参（JSON Body）

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `expire_sec` | integer | 否 | `3600` | 会话有效期（秒），范围 `[1800, 86400]` |

> 无需再传 `rc_sn`；从设备登记缓存读取。`device_sn` 用于校验已登记，并随响应返回，供后续飞控 API 使用。

#### 出参

| 字段 | 类型 | 说明 |
|------|------|------|
| `drc_ready` | boolean | 固定 `true`（成功时） |
| `client_id` | string | connect 返回的 `data.client_id` |
| `rc_sn` | string | 登记缓存中的遥控器 SN |
| `device_sn` | string | 登记缓存中的飞行器 SN |
| `expire_sec` | integer | 会话有效期 |
| `workspace_id` | string | workspace ID |
| `connect` | object | backend `drc/connect` 原始响应 |
| `enter` | object | backend `drc/enter` 原始响应 |
| `pub` | string \| array \| object \| null | enter 响应 `data.pub` |
| `sub` | string \| array \| object \| null | enter 响应 `data.sub` |
| `timestamp` | string | 完成时间 |

#### 后端调用详情（SatNav 内部串行）

**Step 1 — connect**

```http
POST http://{SATNAV_BACKEND_HOST}:{SATNAV_BACKEND_PORT}/control/api/v1/pilot/rc-plus-2/workspaces/{workspace_id}/drc/connect
x-auth-token: <login 缓存的 access_token>
Content-Type: application/json

{"expire_sec": 3600}
```

**Step 2 — enter**（`rc_sn` 来自 §7 登记缓存）

```http
POST http://{SATNAV_BACKEND_HOST}:{SATNAV_BACKEND_PORT}/control/api/v1/pilot/rc-plus-2/workspaces/{workspace_id}/drc/enter
x-auth-token: <login 缓存的 access_token>
Content-Type: application/json

{
  "client_id": "<connect.data.client_id>",
  "rc_sn": "<cached rc_sn>",
  "expire_sec": 3600
}
```

进入 DRC 模式后，手柄会弹出确认框，用户确认后方可控制。

#### 示例

**请求**

```http
POST /api/satnav/system/flight-rc-backend/drc HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "expire_sec": 3600
}
```

**响应 `200`**

```json
{
  "drc_ready": true,
  "client_id": "drc-client-abc123",
  "rc_sn": "RCPLUS2_SN_EXAMPLE",
  "device_sn": "AIRCRAFT_SN_EXAMPLE",
  "expire_sec": 3600,
  "workspace_id": "00000000-0000-4000-8000-000000000001",
  "connect": {
    "code": 0,
    "message": "success",
    "data": {
      "client_id": "drc-client-abc123"
    }
  },
  "enter": {
    "code": 0,
    "message": "success",
    "data": {
      "pub": ["thing/product/xxx/drc/down"],
      "sub": ["thing/product/xxx/drc/up"]
    }
  },
  "pub": ["thing/product/xxx/drc/down"],
  "sub": ["thing/product/xxx/drc/up"],
  "timestamp": "2026-08-02T22:14:00.000000+08:00"
}
```

**响应 `400`（未登录，无 workspace_id）**

```json
{
  "detail": "请先登录飞控系统（workspace_id 未缓存）"
}
```

**响应 `400`（未登记设备）**

```json
{
  "detail": "rc_sn, device_sn not registered; call POST /api/satnav/system/flight-rc-backend/register_device first"
}
```

**响应 `400`（未先 login）**

```json
{
  "detail": "x-auth-token is not available; login first"
}
```

**响应 `400`（backend 业务失败）**

```json
{
  "detail": "some backend error (code=10001)"
}
```

---

## 10. 执行前进（FORWARD）

### `POST /api/satnav/system/flight-rc-backend/flight/forward`

对应模型动作 `next_action=1`（FORWARD），调用飞控 backend `pitch-by-distance`（CloudSDK 3.6）。

**前置条件**

1. `POST .../login` 成功（缓存 `workspace_id`）
2. `POST .../register_device` 已登记 `rc_sn` / `device_sn`
3. `POST .../drc` 成功（缓存 `client_id`）

#### 相关配置（环境变量，均有默认值）

| 变量 | 默认 | 说明 |
|------|------|------|
| `SATNAV_FLIGHT_DEFAULT_FORWARD_DISTANCE_M` | `10` | 默认前进距离（米） |
| `SATNAV_FLIGHT_DEFAULT_FORWARD_TOLERANCE_M` | `0.3` | 默认前进容差（米） |
| `SATNAV_FLIGHT_DEFAULT_TIMEOUT_MS` | `30000` | 默认 Stick 任务超时（毫秒） |

#### 入参（JSON Body）

全部字段可选；省略时使用上表环境变量默认值。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `distance_m` | number | 否 | 前进距离（米） |
| `tolerance_m` | number | 否 | 到达容差（米），范围 `[0, 50]` |
| `timeout_ms` | integer | 否 | 任务超时（毫秒），范围 `[1000, 300000]` |

空 body `{}` 合法，表示全部使用默认参数。

#### 出参

| 字段 | 类型 | 说明 |
|------|------|------|
| `action` | integer | 固定 `1`（FORWARD） |
| `task_id` | string | Stick 闭环任务 ID，用于 §12 轮询 |
| `status` | string | 提交时状态，通常为 `PENDING` |
| `parameters` | object | 实际使用的参数（`distance_m` / `tolerance_m` / `timeout_ms`） |
| `backend` | object | 飞控 backend 原始响应 |
| `timestamp` | string | 响应时间 |

#### 示例

**请求（使用默认 10m / 1m / 30s）**

```http
POST /api/satnav/system/flight-rc-backend/flight/forward HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{}
```

**响应 `200`**

```json
{
  "action": 1,
  "task_id": "stick-task-abc123",
  "status": "PENDING",
  "parameters": {
    "distance_m": 10,
    "tolerance_m": 0.3,
    "timeout_ms": 30000
  },
  "backend": {
    "code": 0,
    "message": "success",
    "data": {
      "task_id": "stick-task-abc123",
      "status": "PENDING"
    }
  },
  "timestamp": "2026-08-03T10:00:00.000000+08:00"
}
```

**响应 `400`（未获取飞行控制）**

```json
{
  "detail": "请先获取飞行控制（client_id 未缓存）"
}
```

---

## 11. 执行转向（TURN_LEFT / TURN_RIGHT）

### `POST /api/satnav/system/flight-rc-backend/flight/turn`

对应模型动作 `next_action=2`（左转）或 `3`（右转），调用飞控 backend `yaw-by-degree`（CloudSDK 3.7）。

采用**方案 A**：`action` 与推理 `next_action` 对齐；`degree` 为**绝对值**，符号由 `action` 决定（`2` → 负角度左转，`3` → 正角度右转）。

**前置条件**：同 §10。

#### 相关配置（环境变量，均有默认值）

| 变量 | 默认 | 说明 |
|------|------|------|
| `SATNAV_FLIGHT_DEFAULT_YAW_DEG` | `15` | 默认转向角度（绝对值，度） |
| `SATNAV_FLIGHT_DEFAULT_YAW_TOLERANCE_DEG` | `0.2` | 默认转向容差（度） |
| `SATNAV_FLIGHT_DEFAULT_TIMEOUT_MS` | `30000` | 默认 Stick 任务超时（毫秒） |

#### 入参（JSON Body）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `action` | integer | 是 | `2`=TURN_LEFT，`3`=TURN_RIGHT |
| `degree` | number | 否 | 绝对转角（度），省略则用默认 `15` |
| `tolerance_deg` | number | 否 | 到达容差（度），范围 `[0, 10]` |
| `timeout_ms` | integer | 否 | 任务超时（毫秒），范围 `[1000, 300000]` |

#### 出参

| 字段 | 类型 | 说明 |
|------|------|------|
| `action` | integer | `2` 或 `3` |
| `task_id` | string | Stick 闭环任务 ID |
| `status` | string | 提交时状态 |
| `parameters` | object | `degree`（绝对值）、`signed_degree`（实际下发）、`tolerance_deg`、`timeout_ms` |
| `backend` | object | 飞控 backend 原始响应 |
| `timestamp` | string | 响应时间 |

#### 示例

**请求（右转 15°，与 `next_action=3` 对齐）**

```http
POST /api/satnav/system/flight-rc-backend/flight/turn HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "action": 3,
  "degree": 15
}
```

**响应 `200`**

```json
{
  "action": 3,
  "task_id": "stick-task-def456",
  "status": "PENDING",
  "parameters": {
    "degree": 15,
    "signed_degree": 15,
    "tolerance_deg": 0.2,
    "timeout_ms": 30000
  },
  "backend": {
    "code": 0,
    "message": "success",
    "data": {
      "task_id": "stick-task-def456",
      "status": "PENDING"
    }
  },
  "timestamp": "2026-08-03T10:01:00.000000+08:00"
}
```

**左转示例**：`{"action": 2}`（省略 `degree` 时默认 15°，实际下发 `signed_degree=-15`）。

---

## 12. 查询 Stick 闭环任务状态

### `GET /api/satnav/system/flight-rc-backend/flight/stick-task/{task_id}`

轮询 §10 / §11 返回的 `task_id`，对应 CloudSDK 3.9。建议在 `status` 变为 `COMPLETED` 或 `FAILED` 后再调用 `POST .../model/inference` 获取下一帧反馈。

**前置条件**：`login` 成功（需 `workspace_id`）；无需 `client_id`。

#### 路径参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `task_id` | string | §10 或 §11 返回的任务 ID |

#### 出参

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_id` | string | 任务 ID |
| `kind` | string | 任务类型，如 `YAW` / `PITCH` |
| `status` | string | `PENDING` / `RUNNING` / `COMPLETED` / `FAILED` 等 |
| `backend` | object | 飞控 backend 原始响应 |
| `timestamp` | string | 响应时间 |

#### 示例

**请求**

```http
GET /api/satnav/system/flight-rc-backend/flight/stick-task/stick-task-abc123 HTTP/1.1
Host: 127.0.0.1:8000
```

**响应 `200`**

```json
{
  "task_id": "stick-task-abc123",
  "kind": "PITCH",
  "status": "COMPLETED",
  "backend": {
    "code": 0,
    "message": "success",
    "data": {
      "task_id": "stick-task-abc123",
      "kind": "PITCH",
      "status": "COMPLETED"
    }
  },
  "timestamp": "2026-08-03T10:02:00.000000+08:00"
}
```

#### 推荐闭环顺序

```
login → register_device → drc（缓存 client_id）
  → inference（得 next_action）
  → forward / turn（得 task_id）
  → 轮询 stick-task 至 COMPLETED
  → inference（反馈帧）→ …
```

> `next_action=0`（STOP）无对应飞控 REST 接口；前端收到后结束当前步骤即可。

---

## 12.1 获取当前 OSD 信息（LiveStore 快照）

### `GET /api/satnav/system/flight-rc-backend/flight/osd/latest`

代理 CloudSDK `POST .../flight/osd/latest`，返回 LiveStore 中该飞机最新的 OSD 快照。

**前置条件**（与 forward / turn 一致）：

1. `POST .../login` 成功（缓存 `workspace_id`）
2. `POST .../register_device` 已登记 `rc_sn` / `device_sn`
3. `POST .../drc` 成功（缓存 `client_id`）

未完成 `drc/enter` 或尚未收到 `osd_info_push` 时，飞控 backend 返回 `code != 0`，本接口以 `400` 透传 `detail`。

#### 入参

无（`client_id`、`rc_sn`、`device_sn` 由服务端从缓存注入）。

#### 出参

| 字段 | 类型 | 说明 |
|------|------|------|
| `device_sn` | string | 飞机 SN |
| `attitude_head` | number | 航向角（度） |
| `latitude` | number | 纬度 |
| `longitude` | number | 经度 |
| `height` | number | 对地高（米） |
| `speed_x` / `speed_y` / `speed_z` | number | 速度分量 |
| `gimbal_pitch` / `gimbal_roll` / `gimbal_yaw` | number | 云台姿态 |
| `received_at_ms` | integer | OSD 推送接收时间（Unix 毫秒） |
| `age_ms` | integer | 快照年龄（毫秒） |
| `backend` | object | 飞控 backend 原始响应 |
| `timestamp` | string | 本 API 响应时间 |

#### 示例

**请求**

```http
GET /api/satnav/system/flight-rc-backend/flight/osd/latest HTTP/1.1
Host: 127.0.0.1:8000
```

**响应 `200`**

```json
{
  "device_sn": "AIRCRAFT_SN_EXAMPLE",
  "attitude_head": 42.5,
  "latitude": 22.6070293,
  "longitude": 114.0561159,
  "height": 25.3,
  "speed_x": 0.1,
  "speed_y": 0.0,
  "speed_z": 0.0,
  "gimbal_pitch": -10.0,
  "gimbal_roll": 0.5,
  "gimbal_yaw": 15.0,
  "received_at_ms": 1743518000456,
  "age_ms": 120,
  "backend": {
    "code": 0,
    "message": "success",
    "data": {
      "device_sn": "AIRCRAFT_SN_EXAMPLE",
      "attitude_head": 42.5,
      "latitude": 22.6070293,
      "longitude": 114.0561159,
      "height": 25.3,
      "received_at_ms": 1743518000456,
      "age_ms": 120
    }
  },
  "timestamp": "2026-08-03T19:26:27.000000+08:00"
}
```

**响应 `400`（尚无 OSD 缓存）**

```json
{
  "detail": "No LiveStore OSD snapshot for device_sn=.... Complete drc/enter and wait for osd_info_push."
}
```

---

## 13. 算法推理（RTMP 抽帧 + Deploy Image）

### `POST /api/satnav/model/inference`

执行一次完整的感知步骤：

1. 检查模型 `ready`、RTMP `stream_active`
2. 若 deploy session 未启动 → 发送 `start`（带 `instruction`）
3. 等待一张**新** RTMP 帧（`request_fresh_frame`）
4. 预处理为 448×448 RGB JPEG（默认 `center-crop`）
5. 向 deploy 子进程发送 `image` JSONL 命令
6. 返回 deploy 响应及耗时

**是否真实调用模型推理**由 deploy session 内部队列决定（与 `test_rtmp_model_latency.py` 一致）：

| 场景 | `performed_inference` |
|------|------------------------|
| 动作队列为空（通常为首帧） | `true` |
| 队列仍有未消费 action | `false`（仅存图并弹出已完成动作） |

**Session 约束**

- 同一 API 生命周期内，若 session 已 `start` 且 `instruction` 与本次请求不同 → `400`
- 不要调用 deploy `end`（会终止模型子进程）

#### 入参（JSON Body）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `instruction` | string | 是 | 导航自然语言指令，去首尾空白后不能为空，最短 1 字符 |

#### 出参

| 字段 | 类型 | 说明 |
|------|------|------|
| `instruction` | string | 实际使用的指令（trim 后） |
| `session_id` | string | deploy session ID |
| `started_new_session` | boolean | 本次是否新发了 `start` |
| `performed_inference` | boolean | 本次 `image` 是否触发真实模型推理 |
| `raw_action_text` | string | 模型原始动作文本，如 `"0 1 2 3"`（见动作编码表） |
| `actions` | integer[] | 本次推理生成的动作列表，元素为 `0`–`3`（仅 `performed_inference=true` 时非空） |
| `next_action` | integer \| null | 队列头部待执行动作（`0`–`3`） |
| `remaining_actions` | integer[] | 剩余动作队列（`0`–`3`） |
| `completed_action` | integer \| null | 本次作为反馈消费的动作（`0`–`3`；首帧为 `null`） |
| `deploy_state` | string | deploy session 状态，如 `waiting_feedback` / `waiting_image` |
| `frame_sequence` | integer | 本次使用的 RTMP 帧序号 |
| `image_path` | string | 本地保存的 448×448 JPEG 绝对路径 |
| `timing` | object | 各阶段耗时（毫秒） |
| `timing.capture_ms` | number | 等待并获取新 RTMP 帧 |
| `timing.preprocess_ms` | number | BGR → 448×448 RGB |
| `timing.input_save_ms` | number | JPEG 落盘 |
| `timing.deploy_response_ms` | number | 发送 `image` 至收到 deploy 响应 |
| `timing.capture_to_action_response_ms` | number | 从请求新帧到拿到 deploy 响应的端到端时间 |
| `timestamp` | string | 完成时间 |

#### 相关环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `SATNAV_RTMP_FRAME_TIMEOUT_SECONDS` | `10` | 等待新帧超时 |
| `SATNAV_MODEL_DEPLOY_RESPONSE_TIMEOUT_SECONDS` | `120` | deploy JSONL 响应超时 |
| `SATNAV_MODEL_INPUT_RESIZE_MODE` | `center-crop` | 或 `stretch` |
| `SATNAV_MODEL_INFERENCE_INPUT_ROOT` | `{session_root}/inference_inputs` | JPEG 输出目录 |

#### 示例

**请求（首帧，触发真实推理）**

```http
POST /api/satnav/model/inference HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "instruction": "Fly to the lake and fly around the lake clockwise"
}
```

**响应 `200`**

```json
{
  "instruction": "Fly to the lake and fly around the lake clockwise",
  "session_id": "satnav-infer-a1b2c3d4",
  "started_new_session": true,
  "performed_inference": true,
  "raw_action_text": "0 1 2 3",
  "actions": [0, 1, 2, 3],
  "next_action": 0,
  "remaining_actions": [0, 1, 2, 3],
  "completed_action": null,
  "deploy_state": "waiting_feedback",
  "frame_sequence": 1290,
  "image_path": "/path/to/SwiftVLN/satnav/runtime/model_sessions/inference_inputs/infer_f3e2d1c0b9a8_448.jpg",
  "timing": {
    "capture_ms": 45.231,
    "preprocess_ms": 12.104,
    "input_save_ms": 3.552,
    "deploy_response_ms": 856.443,
    "capture_to_action_response_ms": 917.330
  },
  "timestamp": "2026-08-02T21:46:18.757010+08:00"
}
```

**请求（队列仍有 action，仅消费队列）**

```http
POST /api/satnav/model/inference HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "instruction": "Fly to the lake and fly around the lake clockwise"
}
```

**响应 `200`**

```json
{
  "instruction": "Fly to the lake and fly around the lake clockwise",
  "session_id": "satnav-infer-a1b2c3d4",
  "started_new_session": false,
  "performed_inference": false,
  "raw_action_text": "",
  "actions": [],
  "next_action": 1,
  "remaining_actions": [1, 2, 3],
  "completed_action": 0,
  "deploy_state": "waiting_feedback",
  "frame_sequence": 1295,
  "image_path": "/path/to/SwiftVLN/satnav/runtime/model_sessions/inference_inputs/infer_8a7b6c5d4e3f_448.jpg",
  "timing": {
    "capture_ms": 38.102,
    "preprocess_ms": 11.887,
    "input_save_ms": 3.201,
    "deploy_response_ms": 12.554,
    "capture_to_action_response_ms": 65.744
  },
  "timestamp": "2026-08-02T21:47:05.000000+08:00"
}
```

**响应 `400`（模型未就绪）**

```json
{
  "detail": "model is not ready (state=loading)"
}
```

**响应 `400`（RTMP 未激活）**

```json
{
  "detail": "unable to open RTMP stream"
}
```

**响应 `400`（instruction 与已有 session 冲突）**

```json
{
  "detail": "deploy session already active with a different instruction"
}
```

**响应 `422`（instruction 为空）**

```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "instruction"],
      "msg": "String should have at least 1 character",
      "input": "",
      "ctx": { "min_length": 1 }
    }
  ]
}
```

**响应 `504`（等待 RTMP 新帧超时）**

```json
{
  "detail": "timed out waiting for a fresh RTMP frame: RTMP frame read failed 3 consecutive time(s)"
}
```

**响应 `503`（deploy 进程未运行）**

```json
{
  "detail": "deploy process is not running"
}
```

---

## 14. 原始 RTMP 帧预览

### `GET /api/satnav/media/raw_img`

返回后台 RTMP reader **当前最新一帧**的 JPEG 图像，供前端实时预览。不触发推理、不等待新帧。

#### 入参

无。

#### 出参

| 类型 | 说明 |
|------|------|
| `image/jpeg` | JPEG 二进制 body |

响应头：

| 头 | 说明 |
|----|------|
| `X-Frame-Sequence` | 当前帧序号（与 RTMP reader 一致） |
| `Cache-Control` | `no-store` |

#### 示例

**请求**

```http
GET /api/satnav/media/raw_img HTTP/1.1
Host: 127.0.0.1:8000
```

**响应 `200`**

Body 为 JPEG 二进制（非 JSON）。

**响应 `404`（尚无解码帧）**

```json
{
  "detail": "no RTMP frame available yet"
}
```

---

## 15. 模型输入预览

### `GET /api/satnav/media/model_input_img`

返回 **最近一次** `POST /api/satnav/model/inference` 预处理得到的 448×448 RGB JPEG（与送入 deploy 的图像一致）。

#### 入参

无。

#### 出参

| 类型 | 说明 |
|------|------|
| `image/jpeg` | 448×448 JPEG 二进制 body |

响应头：

| 头 | 说明 |
|----|------|
| `X-Frame-Sequence` | 对应 RTMP 帧序号 |
| `X-Image-Path` | 本地落盘绝对路径 |
| `X-Updated-At` | 缓存更新时间 |
| `Cache-Control` | `no-store` |

#### 示例

**请求**

```http
GET /api/satnav/media/model_input_img HTTP/1.1
Host: 127.0.0.1:8000
```

**响应 `404`（尚未调用 inference）**

```json
{
  "detail": "no model input image available yet; call POST /api/satnav/model/inference first"
}
```

---

## 变更记录

| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-08-02 | 0.1.0 | 初版：登记 7 个已实现 REST 接口 |
| 2026-08-02 | 0.1.1 | 新增 `POST /api/satnav/system/flight-rc-backend/drc`（DRC 获取飞行控制） |
| 2026-08-02 | 0.1.3 | 新增设备 SN 登记/查询 API；DRC 改为读取登记缓存 |
| 2026-08-03 | 0.1.5 | login 缓存 `workspace_id`；飞控接口优先读缓存，移除 `SATNAV_BACKEND_WORKSPACE_ID` |
| 2026-08-03 | 0.1.6 | 新增飞行动作三接口（forward / turn / stick-task）；默认 15°/10m 及容差支持环境变量；DRC 缓存 `client_id` |
| 2026-08-03 | 0.1.7 | 设备 API 路径重命名：`register_device`（登记）、`cur_device_info`（查询） |
| 2026-08-03 | 0.1.8 | 新增 `media/raw_img`、`media/model_input_img`；补全 `requirements.txt` |
| 2026-08-03 | 0.1.9 | 新增 `SATNAV_CORS_ORIGINS`，默认允许 `http://127.0.0.1:5173` |
| 2026-08-03 | 0.2.0 | 新增 `GET .../flight/osd/latest`（LiveStore OSD 快照） |