"""FastAPI entrypoint for the SatNav orchestration service."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import Response

from app.adapters.backend_health_checker import BackendHealthChecker
from app.adapters.flight_control_auth_client import (
    FlightControlAuthClient,
    FlightControlAuthError,
)
from app.adapters.flight_control_device_registry import FlightControlDeviceRegistry
from app.adapters.flight_control_drc_client import FlightControlDrcClient
from app.adapters.flight_control_flight_client import FlightControlFlightClient
from app.adapters.rtmp_frame_utils import encode_bgr_as_jpeg
from app.adapters.rtmp_stream_checker import RtmpStreamChecker
from app.adapters.swiftvln_deploy_client import SwiftVLNDeployClient
from app.cors import configure_cors
from app.media_cache import model_input_cache
from app.schemas import (
    BackendLoginRequest,
    FlightForwardRequest,
    FlightRcBackendDeviceRegisterRequest,
    FlightRcBackendDrcRequest,
    FlightTurnRequest,
    ModelInferenceRequest,
    OperatorLogAppendRequest,
)
from app.services.model_inference_service import (
    ModelInferenceError,
    ModelInferenceService,
)
from app.services.operator_session_log import OperatorSessionLogWriter
from app.time_utils import now_shanghai_iso

SERVICE_NAME = "satnav-api"
SERVICE_VERSION = "0.2.1"

# This SatNav API process manages a single SwiftVLN model subprocess.
model_client = SwiftVLNDeployClient.from_environment()
# RTMP URL comes from SATNAV_RTMP_URL; the checker is created during lifespan startup.
rtmp_checker: Optional[RtmpStreamChecker] = None
backend_health_checker: Optional[BackendHealthChecker] = None
flight_control_auth: Optional[FlightControlAuthClient] = None
flight_control_device_registry = FlightControlDeviceRegistry()
operator_session_log = OperatorSessionLogWriter.from_environment()


def _require_rtmp_checker() -> RtmpStreamChecker:
    if rtmp_checker is None:
        raise RuntimeError("RTMP checker is not initialized")
    return rtmp_checker


@asynccontextmanager
async def lifespan(_: FastAPI):
    global rtmp_checker, backend_health_checker, flight_control_auth
    # FastAPI startup -> start persistent RTMP consumer -> start model subprocess.
    rtmp_checker = RtmpStreamChecker.from_environment()
    backend_health_checker = BackendHealthChecker.from_environment()
    flight_control_auth = FlightControlAuthClient.from_environment()
    rtmp_checker.start()
    model_client.start_in_background()
    operator_session_log.start()
    try:
        yield
    finally:
        # FastAPI shutdown -> clear flight token -> stop RTMP consumer -> close model subprocess.
        if flight_control_auth is not None:
            flight_control_auth.clear()
        flight_control_device_registry.clear()
        model_input_cache.clear()
        rtmp_checker.close()
        model_client.close()
        operator_session_log.close()


app = FastAPI(
    title="SatNav API",
    description="SatNav real-time navigation orchestration service.",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)
configure_cors(app)


@app.get(
    "/api/satnav/health",
    tags=["system"],
    summary="Check if SatNav API is alive; 检查 SatNav API 是否存活",
)
async def health() -> Dict[str, str]:
    """Return the API process liveness status."""
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "timestamp": now_shanghai_iso(),
    }


@app.post(
    "/api/satnav/operator/logs/start",
    tags=["operator"],
    summary="Start operator console log file; 开始写入操作复盘日志文件",
)
async def operator_logs_start() -> Dict[str, Any]:
    """Return current log file (opened automatically at API startup)."""
    return await asyncio.to_thread(operator_session_log.start)


@app.post(
    "/api/satnav/operator/logs/append",
    tags=["operator"],
    summary="Append lines to operator log file; 追加操作复盘日志行",
)
async def operator_logs_append(body: OperatorLogAppendRequest) -> Dict[str, Any]:
    """Append pre-formatted console lines to the active operator log file."""
    try:
        return await asyncio.to_thread(operator_session_log.append, body.lines)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post(
    "/api/satnav/operator/logs/close",
    tags=["operator"],
    summary="Close operator log file; 关闭操作复盘日志文件",
)
async def operator_logs_close() -> Dict[str, Any]:
    """Flush and close the active log file (normally only on API process shutdown)."""
    return await asyncio.to_thread(operator_session_log.close)


@app.get(
    "/api/satnav/operator/logs/status",
    tags=["operator"],
    summary="Get operator log file status; 查询操作复盘日志文件状态",
)
async def operator_logs_status() -> Dict[str, Any]:
    """Return whether an operator log file is open and its path."""
    return operator_session_log.status()


@app.get(
    "/api/satnav/system/model/status",
    tags=["system"],
    summary="Check model load and runtime status; 检查模型加载和运行状态",
)
async def model_status() -> Dict[str, Any]:
    """Return component readiness without blocking on model loading."""
    model_status = model_client.status()
    model_state = model_status["state"]
    if model_state == "ready":
        overall_status = "ready"
    elif model_state == "error":
        overall_status = "error"
    else:
        overall_status = "starting"
    return {
        "status": overall_status,
        "api": {"ready": True},
        "rtmp": _require_rtmp_checker().config(),
        "model": model_status,
        "timestamp": now_shanghai_iso(),
    }


@app.get(
    "/api/satnav/system/model/logs",
    tags=["system"],
    summary="Get model load and runtime logs; 获取模型加载和运行日志",
)
async def model_logs(
    after_sequence: int = Query(default=0, ge=0),
    limit: int = Query(default=500, ge=1, le=2000),
) -> Dict[str, Any]:
    """Return incremental SwiftVLN logs for the control page."""
    return model_client.logs(
        after_sequence=after_sequence,
        limit=limit,
    )


@app.get(
    "/api/satnav/system/rtmp/status",
    tags=["system"],
    summary="Check whether the configured RTMP stream is active; 检查部署时配置的 RTMP 流是否有效",
)
async def rtmp_status() -> Dict[str, Any]:
    """Return the latest RTMP status from the startup reader."""
    return _require_rtmp_checker().check()


@app.post(
    "/api/satnav/system/rtmp/refresh",
    tags=["system"],
    summary="Refresh the RTMP connection; 刷新 RTMP 连接",
)
async def rtmp_refresh() -> Dict[str, Any]:
    """Restart the background RTMP reader after the stream becomes available."""
    return await asyncio.to_thread(_require_rtmp_checker().refresh_connection)


@app.get(
    "/api/satnav/system/flight-rc-backend/status",
    tags=["system"],
    summary="Check whether the flight-control backend is reachable; 检查飞控 backend 服务是否可达",
)
async def flight_rc_backend_status() -> Dict[str, Any]:
    """Probe CloudSDK backend via configured OpenAPI api-docs path."""
    if backend_health_checker is None:
        raise HTTPException(
            status_code=503,
            detail="SATNAV_BACKEND_HOST is not configured",
        )
    return await asyncio.to_thread(backend_health_checker.check)


def _require_flight_control_auth() -> FlightControlAuthClient:
    if flight_control_auth is None:
        raise HTTPException(
            status_code=503,
            detail="SATNAV_BACKEND_HOST is not configured",
        )
    return flight_control_auth


def _flight_client_or_400() -> FlightControlFlightClient:
    auth_client = _require_flight_control_auth()
    try:
        flight_client = FlightControlFlightClient.from_registry(
            auth_client,
            flight_control_device_registry,
        )
    except FlightControlAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if flight_client is None:
        raise HTTPException(
            status_code=503,
            detail="SATNAV_BACKEND_HOST is not configured",
        )
    return flight_client


@app.post(
    "/api/satnav/system/flight-rc-backend/login",
    tags=["system"],
    summary="Login or refresh flight-control backend token; 登录或刷新飞控 backend（登陆飞控系统）",
)
async def flight_rc_backend_login(body: BackendLoginRequest) -> Dict[str, Any]:
    """Login with username/password, or refresh when a cached token exists."""
    client = _require_flight_control_auth()
    try:
        payload = await asyncio.to_thread(
            client.login_or_refresh,
            username=body.username,
            password=body.password,
            flag=body.flag,
        )
    except FlightControlAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    workspace_id = payload.get("workspace_id")
    if isinstance(workspace_id, str) and workspace_id.strip():
        flight_control_device_registry.set_workspace_id(workspace_id)
    return payload


@app.post(
    "/api/satnav/system/flight-rc-backend/register_device",
    tags=["system"],
    summary="Register RC and aircraft serial numbers; 登记飞控遥控器与飞行器 SN",
)
async def flight_rc_backend_device_register(
    body: FlightRcBackendDeviceRegisterRequest,
) -> Dict[str, Any]:
    """Store rc_sn and device_sn in API process memory for later DRC calls."""
    _require_flight_control_auth()
    try:
        return await asyncio.to_thread(
            flight_control_device_registry.register,
            rc_sn=body.rc_sn,
            device_sn=body.device_sn,
        )
    except FlightControlAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get(
    "/api/satnav/system/flight-rc-backend/cur_device_info",
    tags=["system"],
    summary="Get the currently registered RC and aircraft SNs; 查询当前登记的遥控器与飞行器 SN",
)
async def flight_rc_backend_device_current() -> Dict[str, Any]:
    """Return cached workspace_id, rc_sn and device_sn."""
    _require_flight_control_auth()
    return flight_control_device_registry.current()


@app.post(
    "/api/satnav/system/flight-rc-backend/drc",
    tags=["system"],
    summary="Acquire flight control (DRC); 获取飞行控制",
)
async def flight_rc_backend_drc(
    body: FlightRcBackendDrcRequest = Body(default_factory=FlightRcBackendDrcRequest),
) -> Dict[str, Any]:
    """Serially call DRC connect then enter using cached token and device SNs."""
    auth_client = _require_flight_control_auth()
    try:
        drc_client = FlightControlDrcClient.from_registry(
            auth_client,
            flight_control_device_registry,
        )
    except FlightControlAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if drc_client is None:
        raise HTTPException(
            status_code=503,
            detail="SATNAV_BACKEND_HOST is not configured",
        )
    try:
        return await asyncio.to_thread(
            drc_client.acquire_control,
            expire_sec=body.expire_sec,
        )
    except FlightControlAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/api/satnav/system/flight-rc-backend/flight/forward",
    tags=["system"],
    summary="Execute forward action (FORWARD); 执行前进动作（FORWARD）",
)
async def flight_rc_backend_forward(
    body: FlightForwardRequest = Body(default_factory=FlightForwardRequest),
) -> Dict[str, Any]:
    """Submit pitch-by-distance using cached flight-control session context."""
    flight_client = _flight_client_or_400()
    try:
        return await asyncio.to_thread(
            flight_client.forward,
            distance_m=body.distance_m,
            tolerance_m=body.tolerance_m,
            timeout_ms=body.timeout_ms,
        )
    except FlightControlAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/api/satnav/system/flight-rc-backend/flight/turn",
    tags=["system"],
    summary="Execute turn action (TURN_LEFT / TURN_RIGHT); 执行转向动作（TURN_LEFT / TURN_RIGHT）",
)
async def flight_rc_backend_turn(body: FlightTurnRequest) -> Dict[str, Any]:
    """Submit yaw-by-degree using cached flight-control session context."""
    flight_client = _flight_client_or_400()
    try:
        return await asyncio.to_thread(
            flight_client.turn,
            action=body.action,
            degree=body.degree,
            tolerance_deg=body.tolerance_deg,
            timeout_ms=body.timeout_ms,
        )
    except FlightControlAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get(
    "/api/satnav/system/flight-rc-backend/flight/stick-task/{task_id}",
    tags=["system"],
    summary="Query stick closed-loop task status; 查询 Stick 闭环任务状态",
)
async def flight_rc_backend_stick_task(task_id: str) -> Dict[str, Any]:
    """Poll yaw/pitch stick task status by task_id."""
    flight_client = _flight_client_or_400()
    try:
        return await asyncio.to_thread(flight_client.get_stick_task, task_id)
    except FlightControlAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get(
    "/api/satnav/system/flight-rc-backend/flight/osd/latest",
    tags=["system"],
    summary="Get latest OSD snapshot (LiveStore); 获取当前 OSD 信息（LiveStore 快照）",
)
async def flight_rc_backend_osd_latest() -> Dict[str, Any]:
    """Fetch the latest LiveStore OSD snapshot for the registered aircraft."""
    flight_client = _flight_client_or_400()
    try:
        return await asyncio.to_thread(flight_client.get_osd_latest)
    except FlightControlAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/api/satnav/model/inference",
    tags=["model"],
    summary="Capture an RTMP frame and run one model inference; 从 RTMP 抽帧并执行一次模型推理",
)
async def model_inference(body: ModelInferenceRequest) -> Dict[str, Any]:
    """Capture a fresh RTMP frame, preprocess to 448x448, and send deploy image."""
    service = ModelInferenceService.from_environment(
        model_client,
        _require_rtmp_checker(),
    )
    try:
        return await asyncio.to_thread(service.run, body.instruction)
    except ModelInferenceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get(
    "/api/satnav/media/raw_img",
    tags=["media"],
    summary="Get the latest raw RTMP frame as JPEG; 获取最新 RTMP 原始帧 JPEG",
    response_class=Response,
)
async def media_raw_img() -> Response:
    """Return the latest decoded RTMP frame as JPEG for UI preview."""
    checker = _require_rtmp_checker()
    snapshot = checker.snapshot_latest_frame()
    if snapshot is None:
        raise HTTPException(
            status_code=404,
            detail="no RTMP frame available yet",
        )
    frame_sequence, frame_bgr = snapshot
    try:
        jpeg_bytes = await asyncio.to_thread(encode_bgr_as_jpeg, frame_bgr)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Frame-Sequence": str(frame_sequence),
        },
    )


@app.get(
    "/api/satnav/media/model_input_img",
    tags=["media"],
    summary="Get the latest 448×448 model-input JPEG; 获取最新 448×448 模型输入 JPEG",
    response_class=Response,
)
async def media_model_input_img() -> Response:
    """Return the model-input JPEG from the most recent inference call."""
    snapshot = model_input_cache.snapshot()
    if snapshot is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "no model input image available yet; "
                "call POST /api/satnav/model/inference first"
            ),
        )
    headers = {
        "Cache-Control": "no-store",
        "X-Image-Path": snapshot["image_path"] or "",
    }
    if snapshot["frame_sequence"] is not None:
        headers["X-Frame-Sequence"] = str(snapshot["frame_sequence"])
    if snapshot["updated_at"] is not None:
        headers["X-Updated-At"] = snapshot["updated_at"]
    return Response(
        content=snapshot["jpeg_bytes"],
        media_type="image/jpeg",
        headers=headers,
    )
