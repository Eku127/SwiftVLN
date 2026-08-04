"""In-memory cache for the latest model-input preview image."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any, Dict, Optional

from app.time_utils import now_shanghai_iso


@dataclass
class ModelInputCache:
    """Store the JPEG bytes from the most recent inference preprocessing step."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _jpeg_bytes: Optional[bytes] = field(default=None, init=False, repr=False)
    _frame_sequence: Optional[int] = field(default=None, init=False, repr=False)
    _image_path: Optional[str] = field(default=None, init=False, repr=False)
    _updated_at: Optional[str] = field(default=None, init=False, repr=False)

    def update(
        self,
        *,
        jpeg_bytes: bytes,
        frame_sequence: int,
        image_path: str,
    ) -> None:
        with self._lock:
            self._jpeg_bytes = jpeg_bytes
            self._frame_sequence = frame_sequence
            self._image_path = image_path
            self._updated_at = now_shanghai_iso()

    def snapshot(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self._jpeg_bytes:
                return None
            return {
                "jpeg_bytes": self._jpeg_bytes,
                "frame_sequence": self._frame_sequence,
                "image_path": self._image_path,
                "updated_at": self._updated_at,
            }

    def clear(self) -> None:
        with self._lock:
            self._jpeg_bytes = None
            self._frame_sequence = None
            self._image_path = None
            self._updated_at = None


model_input_cache = ModelInputCache()
