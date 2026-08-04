"""Cache and refresh CloudSDK backend x-auth-token for flight-control APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import threading
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.time_utils import now_shanghai_iso


class FlightControlAuthError(RuntimeError):
    """Raised when CloudSDK login or token refresh fails; CloudSDK 登录或 token 刷新失败时抛出。"""


@dataclass
class FlightControlAuthClient:
    """Cache and refresh in-memory x-auth-token for CloudSDK flight APIs; 缓存并刷新飞控 CloudSDK 的内存 x-auth-token。"""

    host: str
    port: int
    request_timeout_s: float = 10.0
    login_path: str = "/manage/api/v1/login"
    refresh_path: str = "/manage/api/v1/token/refresh"

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _access_token: Optional[str] = field(default=None, init=False, repr=False)
    _profile: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _last_auth_method: Optional[str] = field(default=None, init=False, repr=False)
    _authenticated_at: Optional[str] = field(default=None, init=False, repr=False)

    @classmethod
    def from_environment(cls) -> Optional["FlightControlAuthClient"]:
        host = os.environ.get("SATNAV_BACKEND_HOST", "").strip()
        if not host:
            return None
        port = int(os.environ.get("SATNAV_BACKEND_PORT", "6789"))
        request_timeout_s = float(
            os.environ.get("SATNAV_BACKEND_REQUEST_TIMEOUT_SECONDS", "10")
        )
        return cls(host=host, port=port, request_timeout_s=request_timeout_s)

    def config(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "host": self.host,
                "port": self.port,
                "request_timeout_s": self.request_timeout_s,
                "login_path": self.login_path,
                "refresh_path": self.refresh_path,
                "logged_in": self._access_token is not None,
            }

    def clear(self) -> None:
        with self._lock:
            self._access_token = None
            self._profile = {}
            self._last_auth_method = None
            self._authenticated_at = None

    def get_workspace_id(self) -> Optional[str]:
        with self._lock:
            value = self._profile.get("workspace_id")
            return value if isinstance(value, str) and value.strip() else None

    def status(self) -> Dict[str, Any]:
        with self._lock:
            payload: Dict[str, Any] = {
                "logged_in": self._access_token is not None,
                "backend_host": self.host,
                "backend_port": self.port,
                "checked_at": now_shanghai_iso(),
            }
            if self._last_auth_method is not None:
                payload["auth_method"] = self._last_auth_method
            if self._authenticated_at is not None:
                payload["authenticated_at"] = self._authenticated_at
            if self._profile:
                payload["profile"] = dict(self._profile)
            return payload

    def get_auth_headers(self) -> Dict[str, str]:
        with self._lock:
            if not self._access_token:
                raise FlightControlAuthError("x-auth-token is not available; login first")
            return {"x-auth-token": self._access_token}

    def login_or_refresh(
        self,
        *,
        username: Optional[str] = None,
        password: Optional[str] = None,
        flag: int = 1,
    ) -> Dict[str, Any]:
        with self._lock:
            has_token = self._access_token is not None

        if has_token:
            backend_response, status_code = self._refresh_token()
            auth_method = "refresh"
        else:
            if not username or not password:
                raise FlightControlAuthError(
                    "username and password are required when no cached x-auth-token exists"
                )
            backend_response, status_code = self._login(
                username=username,
                password=password,
                flag=flag,
            )
            auth_method = "login"

        if status_code == 401 and auth_method == "refresh":
            self.clear()
            raise FlightControlAuthError(
                "token refresh returned 401; cached token cleared, login required"
            )

        if backend_response.get("code") != 0:
            message = backend_response.get("message") or "backend authentication failed"
            raise FlightControlAuthError(f"{message} (code={backend_response.get('code')})")

        data = backend_response.get("data")
        if not isinstance(data, dict):
            raise FlightControlAuthError("backend response missing data object")

        access_token = data.get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            raise FlightControlAuthError("backend response missing data.access_token")

        self._store_session(access_token=access_token, data=data, auth_method=auth_method)
        with self._lock:
            workspace_id = self._profile.get("workspace_id")
        return {
            "auth_method": auth_method,
            "logged_in": True,
            "workspace_id": workspace_id,
            "backend": backend_response,
            "timestamp": now_shanghai_iso(),
        }

    def _store_session(
        self,
        *,
        access_token: str,
        data: Dict[str, Any],
        auth_method: str,
    ) -> None:
        profile = {
            key: data[key]
            for key in (
                "user_id",
                "username",
                "workspace_id",
                "user_type",
                "mqtt_username",
                "mqtt_addr",
            )
            if key in data
        }
        with self._lock:
            if "workspace_id" not in profile and self._profile.get("workspace_id"):
                profile["workspace_id"] = self._profile["workspace_id"]
            self._access_token = access_token
            self._profile = profile
            self._last_auth_method = auth_method
            self._authenticated_at = now_shanghai_iso()

    def _login(
        self,
        *,
        username: str,
        password: str,
        flag: int,
    ) -> tuple[Dict[str, Any], int]:
        url = self._url(self.login_path)
        payload = {
            "username": username,
            "password": password,
            "flag": flag,
        }
        return self._post_json(url=url, headers={"Content-Type": "application/json"}, payload=payload)

    def _refresh_token(self) -> tuple[Dict[str, Any], int]:
        with self._lock:
            token = self._access_token
        if not token:
            raise FlightControlAuthError("x-auth-token is not available; login first")
        url = self._url(self.refresh_path)
        return self._post_json(url=url, headers={"x-auth-token": token}, payload=None)

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = f"/{path}"
        return f"http://{self.host}:{self.port}{path}"

    def _post_json(
        self,
        *,
        url: str,
        headers: Dict[str, str],
        payload: Optional[Dict[str, Any]],
    ) -> tuple[Dict[str, Any], int]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(url, data=body, method="POST", headers=headers)
        try:
            with urlopen(request, timeout=self.request_timeout_s) as response:
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
        return parsed, status_code
