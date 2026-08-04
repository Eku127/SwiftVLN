"""In-memory registry for flight-control session context."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any, Dict, Optional, Tuple

from app.adapters.flight_control_auth_client import FlightControlAuthError
from app.time_utils import now_shanghai_iso


@dataclass
class FlightControlDeviceRegistry:
    """In-memory registry for workspace_id / client_id / rc_sn / device_sn; 内存登记 workspace_id、client_id、遥控器与飞行器 SN。"""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _workspace_id: Optional[str] = field(default=None, init=False, repr=False)
    _client_id: Optional[str] = field(default=None, init=False, repr=False)
    _rc_sn: Optional[str] = field(default=None, init=False, repr=False)
    _device_sn: Optional[str] = field(default=None, init=False, repr=False)
    _registered_at: Optional[str] = field(default=None, init=False, repr=False)

    def set_workspace_id(self, workspace_id: Optional[str]) -> None:
        resolved = (workspace_id or "").strip()
        with self._lock:
            self._workspace_id = resolved or None

    def require_workspace_id(self) -> str:
        with self._lock:
            workspace_id = self._workspace_id
        if not workspace_id:
            raise FlightControlAuthError(
                "请先登录飞控系统（workspace_id 未缓存）"
            )
        return workspace_id

    def set_client_id(self, client_id: Optional[str]) -> None:
        resolved = (client_id or "").strip()
        with self._lock:
            self._client_id = resolved or None

    def require_client_id(self) -> str:
        with self._lock:
            client_id = self._client_id
        if not client_id:
            raise FlightControlAuthError(
                "请先获取飞行控制（client_id 未缓存）"
            )
        return client_id

    def register(self, *, rc_sn: str, device_sn: str) -> Dict[str, Any]:
        resolved_rc_sn = rc_sn.strip()
        resolved_device_sn = device_sn.strip()
        if not resolved_rc_sn:
            raise FlightControlAuthError("rc_sn must not be empty")
        if not resolved_device_sn:
            raise FlightControlAuthError("device_sn must not be empty")

        registered_at = now_shanghai_iso()
        with self._lock:
            self._rc_sn = resolved_rc_sn
            self._device_sn = resolved_device_sn
            self._registered_at = registered_at
            workspace_id = self._workspace_id
            client_id = self._client_id

        return {
            "registered": True,
            "workspace_id": workspace_id,
            "client_id": client_id,
            "drc_ready": bool(client_id),
            "rc_sn": resolved_rc_sn,
            "device_sn": resolved_device_sn,
            "registered_at": registered_at,
            "timestamp": now_shanghai_iso(),
        }

    def current(self) -> Dict[str, Any]:
        with self._lock:
            workspace_id = self._workspace_id
            client_id = self._client_id
            rc_sn = self._rc_sn
            device_sn = self._device_sn
            registered_at = self._registered_at

        return {
            "logged_in": bool(workspace_id),
            "workspace_id": workspace_id,
            "client_id": client_id,
            "drc_ready": bool(client_id),
            "registered": bool(rc_sn and device_sn),
            "rc_sn": rc_sn,
            "device_sn": device_sn,
            "registered_at": registered_at,
            "timestamp": now_shanghai_iso(),
        }

    def require_both(self) -> Tuple[str, str]:
        with self._lock:
            rc_sn = self._rc_sn
            device_sn = self._device_sn

        missing = []
        if not rc_sn:
            missing.append("rc_sn")
        if not device_sn:
            missing.append("device_sn")
        if missing:
            raise FlightControlAuthError(
                f"{', '.join(missing)} not registered; "
                "call POST /api/satnav/system/flight-rc-backend/register_device first"
            )
        return rc_sn, device_sn

    def clear_workspace_id(self) -> None:
        with self._lock:
            self._workspace_id = None

    def clear_client_id(self) -> None:
        with self._lock:
            self._client_id = None

    def clear_devices(self) -> None:
        with self._lock:
            self._rc_sn = None
            self._device_sn = None
            self._registered_at = None

    def clear(self) -> None:
        with self._lock:
            self._workspace_id = None
            self._client_id = None
            self._rc_sn = None
            self._device_sn = None
            self._registered_at = None
