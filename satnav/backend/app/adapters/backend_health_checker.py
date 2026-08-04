"""Probe CloudSDK backend reachability via OpenAPI api-docs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.time_utils import now_shanghai_iso


@dataclass
class BackendHealthChecker:
    """Probe flight-control CloudSDK reachability via OpenAPI api-docs; 通过 OpenAPI api-docs 探测飞控 CloudSDK 是否可达。"""

    host: str
    port: int
    api_docs_path: str = "/v3/api-docs"
    probe_timeout_s: float = 3.0
    expected_service_title: str = "CloudSDK API"

    @classmethod
    def from_environment(cls) -> Optional["BackendHealthChecker"]:
        host = os.environ.get("SATNAV_BACKEND_HOST", "").strip()
        if not host:
            return None
        port = int(os.environ.get("SATNAV_BACKEND_PORT", "6789"))
        api_docs_path = os.environ.get(
            "SATNAV_BACKEND_API_DOCS_PATH", "/v3/api-docs"
        ).strip()
        if not api_docs_path.startswith("/"):
            api_docs_path = f"/{api_docs_path}"
        probe_timeout_s = float(
            os.environ.get("SATNAV_BACKEND_PROBE_TIMEOUT_SECONDS", "3")
        )
        expected_service_title = os.environ.get(
            "SATNAV_BACKEND_EXPECTED_TITLE", "CloudSDK API"
        ).strip()
        return cls(
            host=host,
            port=port,
            api_docs_path=api_docs_path,
            probe_timeout_s=probe_timeout_s,
            expected_service_title=expected_service_title,
        )

    def config(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "api_docs_path": self.api_docs_path,
            "probe_timeout_s": self.probe_timeout_s,
            "expected_service_title": self.expected_service_title,
            "probe_method": "api_docs",
        }

    def check(self) -> Dict[str, Any]:
        url = f"http://{self.host}:{self.port}{self.api_docs_path}"
        request = Request(url, method="GET")
        try:
            with urlopen(request, timeout=self.probe_timeout_s) as response:
                status_code = response.getcode()
                body = response.read()
        except HTTPError as exc:
            return self._result(
                healthy=False,
                status_code=exc.code,
                error=f"unexpected api-docs status code: {exc.code}",
            )
        except URLError as exc:
            reason = exc.reason
            return self._result(
                healthy=False,
                error=f"{type(reason).__name__}: {reason}",
            )
        except OSError as exc:
            return self._result(
                healthy=False,
                error=f"{type(exc).__name__}: {exc}",
            )

        if status_code < 200 or status_code >= 300:
            return self._result(
                healthy=False,
                status_code=status_code,
                error=f"unexpected api-docs status code: {status_code}",
            )

        service_title = self._parse_service_title(body)
        if service_title is None:
            return self._result(
                healthy=False,
                status_code=status_code,
                error="response is not a valid OpenAPI document",
            )
        if (
            self.expected_service_title
            and service_title != self.expected_service_title
        ):
            return self._result(
                healthy=False,
                status_code=status_code,
                service_title=service_title,
                error=(
                    "unexpected service title: "
                    f"{service_title!r} (expected {self.expected_service_title!r})"
                ),
            )
        return self._result(
            healthy=True,
            status_code=status_code,
            service_title=service_title,
        )

    def _parse_service_title(self, body: bytes) -> Optional[str]:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        if not payload.get("openapi"):
            return None
        info = payload.get("info")
        if not isinstance(info, dict):
            return None
        title = info.get("title")
        if not isinstance(title, str) or not title.strip():
            return None
        return title

    def _result(
        self,
        *,
        healthy: bool,
        status_code: Optional[int] = None,
        service_title: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "backend_host": self.host,
            "backend_port": self.port,
            "api_docs_path": self.api_docs_path,
            "healthy": healthy,
            "probe_method": "api_docs",
            "checked_at": now_shanghai_iso(),
        }
        if status_code is not None:
            payload["status_code"] = status_code
        if service_title is not None:
            payload["service_title"] = service_title
        if error is not None:
            payload["error"] = error
        return payload
