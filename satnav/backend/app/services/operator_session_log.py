"""Append-only operator console logs; directory from SATNAV_OPERATOR_LOG_ROOT."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import threading
from typing import IO, Any, Dict, List, Optional

from app.time_utils import SHANGHAI_TZ, now_shanghai_iso


class OperatorSessionLogWriter:
    """One operator log file per SatNav API process (from startup until shutdown)."""

    def __init__(self, log_root: Path, repo_root: Path) -> None:
        self.log_root = log_root.resolve()
        self.repo_root = repo_root.resolve()
        self._lock = threading.Lock()
        self._file: Optional[IO[str]] = None
        self._path: Optional[Path] = None
        self._started_at: Optional[str] = None

    @classmethod
    def from_environment(cls) -> "OperatorSessionLogWriter":
        default_repo_root = Path(__file__).resolve().parents[4]
        configured_repo = os.environ.get("SATNAV_REPO_ROOT")
        repo_root = Path(configured_repo) if configured_repo else default_repo_root

        configured_log_root = os.environ.get("SATNAV_OPERATOR_LOG_ROOT")
        if configured_log_root:
            log_root = Path(configured_log_root)
        else:
            log_root = repo_root / "satnav" / "runtime" / "logs"
        return cls(log_root=log_root, repo_root=repo_root)

    @property
    def active(self) -> bool:
        with self._lock:
            return self._file is not None

    def start(self) -> Dict[str, Any]:
        with self._lock:
            if self._file is not None:
                return self._status_unlocked()
            self.log_root.mkdir(parents=True, exist_ok=True)
            path = self.log_root / _suggest_log_filename()
            self._path = path
            self._file = path.open("a", encoding="utf-8", buffering=1)
            self._started_at = now_shanghai_iso()
            return self._status_unlocked()

    def append(self, lines: List[str]) -> Dict[str, Any]:
        if not lines:
            return self.status()
        with self._lock:
            if self._file is None:
                raise RuntimeError("operator log session is not started")
            for line in lines:
                self._file.write(line if line.endswith("\n") else f"{line}\n")
            self._file.flush()
            return self._status_unlocked()

    def close(self) -> Dict[str, Any]:
        with self._lock:
            self._close_unlocked()
            return self._status_unlocked()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return self._status_unlocked()

    def _status_unlocked(self) -> Dict[str, Any]:
        path = self._path
        return {
            "active": self._file is not None,
            "log_path": str(path) if path is not None else None,
            "log_path_relative": _relative_to_repo(path, self.repo_root) if path else None,
            "started_at": self._started_at,
            "timestamp": now_shanghai_iso(),
        }

    def _close_unlocked(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
        self._file = None
        self._path = None
        self._started_at = None


def _suggest_log_filename(now: Optional[datetime] = None) -> str:
    current = now or datetime.now(SHANGHAI_TZ)
    stamp = current.strftime("%Y%m%dT%H%M%S")
    return f"operator-{stamp}+08.log"


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
