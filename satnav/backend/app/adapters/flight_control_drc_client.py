"""Serial CloudSDK DRC connect + enter calls for RC Plus 2 flight control."""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.adapters.flight_control_device_registry import FlightControlDeviceRegistry
from app.adapters.flight_control_auth_client import (
    FlightControlAuthClient,
    FlightControlAuthError,
)
from app.time_utils import now_shanghai_iso


class FlightControlDrcClient:
    """Acquire DRC control via CloudSDK connect then enter; 串行调用 CloudSDK DRC connect 与 enter 以获取飞行控制。"""

    def __init__(
        self,
        auth_client: FlightControlAuthClient,
        device_registry: FlightControlDeviceRegistry,
        *,
        workspace_id: str,
    ) -> None:
        self.auth_client = auth_client
        self.device_registry = device_registry
        self.workspace_id = workspace_id

    @classmethod
    def from_registry(
        cls,
        auth_client: Optional[FlightControlAuthClient],
        device_registry: FlightControlDeviceRegistry,
    ) -> Optional["FlightControlDrcClient"]:
        if auth_client is None:
            return None
        workspace_id = device_registry.require_workspace_id()
        return cls(
            auth_client,
            device_registry,
            workspace_id=workspace_id,
        )

    def config(self) -> Dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "connect_path": self._drc_path("connect"),
            "enter_path": self._drc_path("enter"),
        }

    def acquire_control(self, *, expire_sec: int = 3600) -> Dict[str, Any]:
        rc_sn, device_sn = self.device_registry.require_both()

        connect_response = self._post_backend(
            path_suffix="connect",
            payload={"expire_sec": expire_sec},
        )
        self._ensure_backend_success(connect_response, step="drc/connect")

        connect_data = connect_response.get("data")
        if not isinstance(connect_data, dict):
            raise FlightControlAuthError("drc/connect response missing data object")

        client_id = connect_data.get("client_id")
        if not isinstance(client_id, str) or not client_id.strip():
            raise FlightControlAuthError("drc/connect response missing data.client_id")

        enter_response = self._post_backend(
            path_suffix="enter",
            payload={
                "client_id": client_id,
                "rc_sn": rc_sn,
                "expire_sec": expire_sec,
            },
        )
        self._ensure_backend_success(enter_response, step="drc/enter")

        enter_data = enter_response.get("data")
        if not isinstance(enter_data, dict):
            raise FlightControlAuthError("drc/enter response missing data object")

        self.device_registry.set_client_id(client_id)

        return {
            "drc_ready": True,
            "client_id": client_id,
            "workspace_id": self.workspace_id,
            "rc_sn": rc_sn,
            "device_sn": device_sn,
            "expire_sec": expire_sec,
            "connect": connect_response,
            "enter": enter_response,
            "pub": enter_data.get("pub"),
            "sub": enter_data.get("sub"),
            "timestamp": now_shanghai_iso(),
        }

    def _drc_path(self, action: str) -> str:
        return (
            "/control/api/v1/pilot/rc-plus-2/workspaces/"
            f"{self.workspace_id}/drc/{action}"
        )

    def _post_backend(
        self,
        *,
        path_suffix: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        headers = self.auth_client.get_auth_headers()
        headers["Content-Type"] = "application/json"
        response, status_code = self.auth_client._post_json(
            url=self.auth_client._url(self._drc_path(path_suffix)),
            headers=headers,
            payload=payload,
        )
        if status_code == 401:
            self.auth_client.clear()
            self.device_registry.clear_workspace_id()
            self.device_registry.clear_client_id()
            raise FlightControlAuthError(
                "backend returned 401 during DRC request; login required"
            )
        if status_code < 200 or status_code >= 300:
            message = response.get("message") or f"unexpected HTTP status {status_code}"
            raise FlightControlAuthError(
                f"drc/{path_suffix} failed: {message} (http={status_code})"
            )
        return response

    @staticmethod
    def _ensure_backend_success(response: Dict[str, Any], *, step: str) -> None:
        if response.get("code") != 0:
            message = response.get("message") or f"{step} failed"
            raise FlightControlAuthError(f"{message} (code={response.get('code')})")
