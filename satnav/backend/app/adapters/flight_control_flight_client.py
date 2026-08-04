"""CloudSDK stick closed-loop flight actions (forward / turn / task status)."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.adapters.flight_control_auth_client import (
    FlightControlAuthClient,
    FlightControlAuthError,
)
from app.adapters.flight_control_device_registry import FlightControlDeviceRegistry
from app.time_utils import now_shanghai_iso


@dataclass(frozen=True)
class FlightControlFlightDefaults:
    """Default yaw/forward distances and tolerances from env; 从环境变量读取默认转向/前进距离与容差。"""

    yaw_deg: float = 15.0
    forward_distance_m: float = 10.0
    yaw_tolerance_deg: float = 0.2
    forward_tolerance_m: float = 0.3
    timeout_ms: int = 30000

    @classmethod
    def from_environment(cls) -> "FlightControlFlightDefaults":
        return cls(
            yaw_deg=float(os.environ.get("SATNAV_FLIGHT_DEFAULT_YAW_DEG", "15")),
            forward_distance_m=float(
                os.environ.get("SATNAV_FLIGHT_DEFAULT_FORWARD_DISTANCE_M", "10")
            ),
            yaw_tolerance_deg=float(
                os.environ.get("SATNAV_FLIGHT_DEFAULT_YAW_TOLERANCE_DEG", "0.2")
            ),
            forward_tolerance_m=float(
                os.environ.get("SATNAV_FLIGHT_DEFAULT_FORWARD_TOLERANCE_M", "0.3")
            ),
            timeout_ms=int(os.environ.get("SATNAV_FLIGHT_DEFAULT_TIMEOUT_MS", "30000")),
        )


class FlightControlFlightClient:
    """Submit forward/turn stick tasks and poll stick-task / OSD status; 下发前进/转向 Stick 任务并查询 stick-task / OSD 状态。"""

    def __init__(
        self,
        auth_client: FlightControlAuthClient,
        device_registry: FlightControlDeviceRegistry,
        *,
        defaults: Optional[FlightControlFlightDefaults] = None,
    ) -> None:
        self.auth_client = auth_client
        self.device_registry = device_registry
        self.defaults = defaults or FlightControlFlightDefaults.from_environment()

    @classmethod
    def from_registry(
        cls,
        auth_client: Optional[FlightControlAuthClient],
        device_registry: FlightControlDeviceRegistry,
    ) -> Optional["FlightControlFlightClient"]:
        if auth_client is None:
            return None
        device_registry.require_workspace_id()
        return cls(auth_client, device_registry)

    def forward(
        self,
        *,
        distance_m: Optional[float] = None,
        tolerance_m: Optional[float] = None,
        timeout_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        session = self._session_context()
        resolved_distance_m = (
            self.defaults.forward_distance_m
            if distance_m is None
            else distance_m
        )
        resolved_tolerance_m = (
            self.defaults.forward_tolerance_m
            if tolerance_m is None
            else tolerance_m
        )
        resolved_timeout_ms = (
            self.defaults.timeout_ms if timeout_ms is None else timeout_ms
        )
        backend_response = self._post_json(
            path_suffix="pitch-by-distance",
            payload={
                "client_id": session["client_id"],
                "rc_sn": session["rc_sn"],
                "device_sn": session["device_sn"],
                "distance_m": resolved_distance_m,
                "tolerance_m": resolved_tolerance_m,
                "timeout_ms": resolved_timeout_ms,
            },
        )
        return self._wrap_action_response(
            backend_response,
            action=1,
            parameters={
                "distance_m": resolved_distance_m,
                "tolerance_m": resolved_tolerance_m,
                "timeout_ms": resolved_timeout_ms,
            },
        )

    def turn(
        self,
        *,
        action: int,
        degree: Optional[float] = None,
        tolerance_deg: Optional[float] = None,
        timeout_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        if action not in {2, 3}:
            raise FlightControlAuthError("action must be 2 (TURN_LEFT) or 3 (TURN_RIGHT)")

        session = self._session_context()
        magnitude = self.defaults.yaw_deg if degree is None else abs(degree)
        if magnitude <= 0:
            raise FlightControlAuthError("degree must be greater than 0")
        signed_degree = magnitude if action == 3 else -magnitude
        resolved_tolerance_deg = (
            self.defaults.yaw_tolerance_deg
            if tolerance_deg is None
            else tolerance_deg
        )
        resolved_timeout_ms = (
            self.defaults.timeout_ms if timeout_ms is None else timeout_ms
        )
        backend_response = self._post_json(
            path_suffix="yaw-by-degree",
            payload={
                "client_id": session["client_id"],
                "rc_sn": session["rc_sn"],
                "device_sn": session["device_sn"],
                "degree": signed_degree,
                "tolerance_deg": resolved_tolerance_deg,
                "timeout_ms": resolved_timeout_ms,
            },
        )
        return self._wrap_action_response(
            backend_response,
            action=action,
            parameters={
                "degree": magnitude,
                "signed_degree": signed_degree,
                "tolerance_deg": resolved_tolerance_deg,
                "timeout_ms": resolved_timeout_ms,
            },
        )

    def get_osd_latest(self) -> Dict[str, Any]:
        session = self._session_context()
        backend_response = self._post_json(
            path_suffix="osd/latest",
            payload={
                "client_id": session["client_id"],
                "rc_sn": session["rc_sn"],
                "device_sn": session["device_sn"],
            },
        )
        data = backend_response.get("data")
        if not isinstance(data, dict):
            raise FlightControlAuthError("osd/latest response missing data object")
        return {
            "device_sn": data.get("device_sn"),
            "attitude_head": data.get("attitude_head"),
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "height": data.get("height"),
            "speed_x": data.get("speed_x"),
            "speed_y": data.get("speed_y"),
            "speed_z": data.get("speed_z"),
            "gimbal_pitch": data.get("gimbal_pitch"),
            "gimbal_roll": data.get("gimbal_roll"),
            "gimbal_yaw": data.get("gimbal_yaw"),
            "received_at_ms": data.get("received_at_ms"),
            "age_ms": data.get("age_ms"),
            "backend": backend_response,
            "timestamp": now_shanghai_iso(),
        }

    def get_stick_task(self, task_id: str) -> Dict[str, Any]:
        resolved_task_id = task_id.strip()
        if not resolved_task_id:
            raise FlightControlAuthError("task_id must not be empty")
        workspace_id = self.device_registry.require_workspace_id()
        backend_response = self._get_json(
            path=(
                f"/control/api/v1/pilot/rc-plus-2/workspaces/{workspace_id}"
                f"/flight/stick-task/{resolved_task_id}"
            ),
        )
        self._ensure_backend_success(backend_response, step="stick-task")
        data = backend_response.get("data")
        if not isinstance(data, dict):
            raise FlightControlAuthError("stick-task response missing data object")
        return {
            "task_id": data.get("task_id", resolved_task_id),
            "kind": data.get("kind"),
            "status": data.get("status"),
            "backend": backend_response,
            "timestamp": now_shanghai_iso(),
        }

    def _session_context(self) -> Dict[str, str]:
        workspace_id = self.device_registry.require_workspace_id()
        rc_sn, device_sn = self.device_registry.require_both()
        client_id = self.device_registry.require_client_id()
        return {
            "workspace_id": workspace_id,
            "rc_sn": rc_sn,
            "device_sn": device_sn,
            "client_id": client_id,
        }

    def _flight_path(self, suffix: str) -> str:
        workspace_id = self.device_registry.require_workspace_id()
        return (
            "/control/api/v1/pilot/rc-plus-2/workspaces/"
            f"{workspace_id}/flight/{suffix}"
        )

    def _post_json(self, *, path_suffix: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = self.auth_client.get_auth_headers()
        headers["Content-Type"] = "application/json"
        response, status_code = self.auth_client._post_json(
            url=self.auth_client._url(self._flight_path(path_suffix)),
            headers=headers,
            payload=payload,
        )
        self._handle_http_status(status_code, response, step=path_suffix)
        self._ensure_backend_success(response, step=path_suffix)
        return response

    def _get_json(self, *, path: str) -> Dict[str, Any]:
        headers = self.auth_client.get_auth_headers()
        request = Request(path if path.startswith("http") else self.auth_client._url(path), method="GET", headers=headers)
        try:
            with urlopen(request, timeout=self.auth_client.request_timeout_s) as response:
                status_code = response.getcode()
                raw = response.read()
        except HTTPError as exc:
            status_code = exc.code
            raw = exc.read()
        except URLError as exc:
            reason = exc.reason
            raise FlightControlAuthError(
                f"{type(reason).__name__}: {reason}"
            ) from exc
        except OSError as exc:
            raise FlightControlAuthError(f"{type(exc).__name__}: {exc}") from exc

        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FlightControlAuthError("backend returned non-JSON response") from exc
        if not isinstance(parsed, dict):
            raise FlightControlAuthError("backend returned invalid JSON object")
        self._handle_http_status(status_code, parsed, step="stick-task")
        return parsed

    def _handle_http_status(
        self,
        status_code: int,
        response: Dict[str, Any],
        *,
        step: str,
    ) -> None:
        if status_code == 401:
            self.auth_client.clear()
            self.device_registry.clear_workspace_id()
            self.device_registry.clear_client_id()
            raise FlightControlAuthError(
                "backend returned 401 during flight request; login required"
            )
        if status_code < 200 or status_code >= 300:
            message = response.get("message") or f"unexpected HTTP status {status_code}"
            raise FlightControlAuthError(f"{step} failed: {message} (http={status_code})")

    @staticmethod
    def _ensure_backend_success(response: Dict[str, Any], *, step: str) -> None:
        if response.get("code") != 0:
            message = response.get("message") or f"{step} failed"
            raise FlightControlAuthError(f"{message} (code={response.get('code')})")

    def _wrap_action_response(
        self,
        backend_response: Dict[str, Any],
        *,
        action: int,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        data = backend_response.get("data")
        if not isinstance(data, dict):
            raise FlightControlAuthError("flight action response missing data object")
        return {
            "action": action,
            "task_id": data.get("task_id"),
            "status": data.get("status"),
            "parameters": parameters,
            "backend": backend_response,
            "timestamp": now_shanghai_iso(),
        }
