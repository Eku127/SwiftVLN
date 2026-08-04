"""Persistent RTMP consumer for SatNav API."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

from app.time_utils import now_shanghai_iso


@dataclass
class RtmpStreamChecker:
    """Persistent RTMP consumer for the API lifetime (status, refresh, frame grab); API 生命周期内常驻 RTMP 消费（状态、刷新、抽帧）。"""

    rtmp_url: str
    probe_timeout_s: float = 8.0

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _refresh_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)
    _capture: Any = field(default=None, init=False, repr=False)
    _state: str = field(default="idle", init=False, repr=False)
    _frame_shape: Optional[Tuple[int, int]] = field(default=None, init=False, repr=False)
    _frame: Any = field(default=None, init=False, repr=False)
    _frame_condition: threading.Condition = field(init=False, repr=False)
    _frame_sequence: int = field(default=0, init=False, repr=False)
    _last_error: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_frame_condition",
            threading.Condition(self._lock),
        )

    @classmethod
    def from_environment(cls) -> "RtmpStreamChecker":
        rtmp_url = os.environ.get("SATNAV_RTMP_URL", "").strip()
        if not rtmp_url:
            raise ValueError(
                "必须设置 SATNAV_RTMP_URL。"
                "启动示例："
                "SATNAV_RTMP_URL=rtmp://<host>/<app>/<stream> "
                "bash satnav/backend/start_api.sh"
            )
        probe_timeout_s = float(
            os.environ.get("SATNAV_RTMP_PROBE_TIMEOUT_SECONDS", "8")
        )
        return cls(rtmp_url=rtmp_url, probe_timeout_s=probe_timeout_s)

    @classmethod
    def refresh_wait_timeout_s(cls) -> float:
        configured = os.environ.get("SATNAV_RTMP_REFRESH_WAIT_SECONDS", "").strip()
        if configured:
            return float(configured)
        probe_timeout_s = float(
            os.environ.get("SATNAV_RTMP_PROBE_TIMEOUT_SECONDS", "8")
        )
        return probe_timeout_s + 2.0

    def config(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "rtmp_url": self.rtmp_url,
                "probe_timeout_s": self.probe_timeout_s,
                "probe_method": "opencv",
                "reader_state": self._state,
                "frame_sequence": self._frame_sequence,
                "last_error": self._last_error,
            }

    def start(self) -> None:
        with self._refresh_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._begin_reader()

    def refresh_connection(
        self,
        wait_timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Stop the current reader and open RTMP again."""
        timeout_s = (
            wait_timeout_s
            if wait_timeout_s is not None
            else self.refresh_wait_timeout_s()
        )
        with self._refresh_lock:
            self._stop_reader()
            self._begin_reader()

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                state = self._state
            if state in {"connected", "error"}:
                break
            time.sleep(0.05)

        payload = self.check()
        payload["refreshed"] = True
        payload["reader_state"] = self.config()["reader_state"]
        return payload

    def close(self) -> None:
        with self._refresh_lock:
            self._stop_reader()
            with self._lock:
                self._state = "stopped"

    def _begin_reader(self) -> None:
        self._stop_event.clear()
        with self._frame_condition:
            self._state = "connecting"
            self._frame_shape = None
            self._frame = None
            self._frame_sequence = 0
            self._last_error = None
            self._frame_condition.notify_all()
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="satnav-rtmp-reader",
            daemon=True,
        )
        self._thread.start()

    def _stop_reader(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=self.probe_timeout_s + 2.0)
        capture = self._capture
        if capture is not None:
            capture.release()
            self._capture = None
        self._thread = None
        with self._frame_condition:
            self._frame_condition.notify_all()

    def check(self) -> Dict[str, Any]:
        """Return the latest RTMP status from the startup reader."""
        if not self.rtmp_url.startswith(("rtmp://", "rtmps://")):
            return self._result(
                connected=False,
                stream_active=False,
                error="RTMP URL must start with rtmp:// or rtmps://",
            )

        with self._lock:
            state = self._state
            frame_shape = self._frame_shape
            last_error = self._last_error

        if state == "connected" and frame_shape is not None:
            height, width = frame_shape
            video: Dict[str, Any] = {"codec": "unknown"}
            if width > 0:
                video["width"] = width
            if height > 0:
                video["height"] = height
            return self._result(
                connected=True,
                stream_active=True,
                video=video,
            )

        if state == "connecting":
            return self._result(
                connected=False,
                stream_active=False,
                error="RTMP reader is still connecting",
            )

        if state == "error":
            return self._result(
                connected=False,
                stream_active=False,
                error=last_error or "RTMP reader failed",
            )

        return self._result(
            connected=False,
            stream_active=False,
            error=last_error or "RTMP reader is not running",
        )

    def request_fresh_frame(self, timeout_s: float) -> tuple[int, Any]:
        """Return the first decoded frame newer than this call."""
        deadline = time.monotonic() + timeout_s
        with self._frame_condition:
            baseline_sequence = self._frame_sequence
            while self._frame_sequence <= baseline_sequence:
                if self._stop_event.is_set():
                    raise RuntimeError("RTMP reader is stopping")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = f": {self._last_error}" if self._last_error else ""
                    raise TimeoutError(
                        f"timed out waiting for a fresh RTMP frame{detail}"
                    )
                self._frame_condition.wait(timeout=min(remaining, 0.1))
            return self._frame_sequence, self._frame.copy()

    def snapshot_latest_frame(self) -> Optional[tuple[int, Any]]:
        """Return a copy of the most recent decoded frame, if any."""
        with self._frame_condition:
            if self._frame is None or self._frame_sequence <= 0:
                return None
            return self._frame_sequence, self._frame.copy()

    def _reader_loop(self) -> None:
        try:
            import cv2
        except ImportError:
            with self._frame_condition:
                self._state = "error"
                self._last_error = "opencv-python is required for RTMP status checks"
                self._frame_condition.notify_all()
            return

        timeout_ms = max(1, int(self.probe_timeout_s * 1000))
        capture = cv2.VideoCapture()
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.open(self.rtmp_url, cv2.CAP_FFMPEG):
            capture.release()
            with self._frame_condition:
                self._state = "error"
                self._last_error = "unable to open RTMP stream"
                self._frame_condition.notify_all()
            return

        self._capture = capture
        with self._frame_condition:
            self._state = "connected"
            self._last_error = None
            self._frame_condition.notify_all()

        consecutive_failures = 0
        while not self._stop_event.is_set():
            ok, frame = capture.read()
            if not ok or frame is None:
                consecutive_failures += 1
                with self._frame_condition:
                    self._last_error = (
                        "RTMP frame read failed "
                        f"{consecutive_failures} consecutive time(s)"
                    )
                    self._frame_condition.notify_all()
                if self._stop_event.wait(0.05):
                    break
                continue

            consecutive_failures = 0
            height, width = frame.shape[:2]
            with self._frame_condition:
                self._frame = frame
                self._frame_shape = (int(height), int(width))
                self._frame_sequence += 1
                self._last_error = None
                self._frame_condition.notify_all()

    def _result(
        self,
        *,
        connected: bool,
        stream_active: bool,
        video: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "rtmp_url": self.rtmp_url,
            "connected": connected,
            "stream_active": stream_active,
            "probe_method": "opencv",
            "checked_at": now_shanghai_iso(),
        }
        if video is not None:
            payload["video"] = video
        if error is not None:
            payload["error"] = error
        return payload
