"""Manage the production SwiftVLN JSONL deployment process."""

from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Deque, Dict, Optional, Set
from uuid import uuid4

from app.time_utils import now_shanghai_iso


class SwiftVLNDeployClient:
    """Manage the SwiftVLN JSONL deploy subprocess and expose load/inference IO; 管理 SwiftVLN JSONL 部署子进程并提供加载状态与推理 IO。"""

    def __init__(
        self,
        repo_root: Path,
        model_name: Optional[str] = None,
        model_path: Optional[Path] = None,
        session_root: Optional[Path] = None,
        startup_timeout_s: float = 600.0,
        deploy_python: Optional[str] = None,
        log_buffer_size: int = 2000,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.model_name = model_name or None
        self.model_path = model_path.resolve() if model_path else None
        self.session_root = (
            session_root or self.repo_root / "satnav" / "runtime" / "model_sessions"
        ).resolve()
        self.model_registry_root = (
            self.repo_root / "satnav" / "runtime" / "model_registry"
        ).resolve()
        self.startup_timeout_s = startup_timeout_s
        self.deploy_python = deploy_python or sys.executable
        self.script_path = (
            self.repo_root
            / "src"
            / "swiftvln"
            / "scripts"
            / "deploy"
            / "start_swiftvln_deploy.sh"
        )

        self.process: Optional[subprocess.Popen[str]] = None
        self._loader_thread: Optional[threading.Thread] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stdout_queue: queue.Queue[Optional[str]] = queue.Queue()
        self._stderr_tail: Deque[str] = deque(maxlen=20)
        self._logs: Deque[Dict[str, Any]] = deque(maxlen=log_buffer_size)
        self._log_sequence = 0
        self._lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._stop_event = threading.Event()
        self.deploy_response_timeout_s = 120.0
        self._session_started = False
        self._active_instruction: Optional[str] = None
        self._active_session_id: Optional[str] = None
        self._deploy_session_state: Optional[str] = None
        self._status: Dict[str, Any] = {
            "state": "not_started",
            "loaded": False,
            "process_running": False,
            "model_name": self.model_name,
            "checkpoint_path": None,
            "gpu": None,
            "error": None,
            "started_at": None,
            "ready_at": None,
        }

    @classmethod
    def from_environment(cls) -> "SwiftVLNDeployClient":
        repo_root = Path(__file__).resolve().parents[4]
        configured_root = os.environ.get("SATNAV_REPO_ROOT")
        model_name = os.environ.get("SATNAV_MODEL_NAME") or None
        model_path_value = os.environ.get("SATNAV_MODEL_PATH")
        session_root_value = os.environ.get("SATNAV_MODEL_SESSION_ROOT")
        startup_timeout_s = float(
            os.environ.get("SATNAV_MODEL_STARTUP_TIMEOUT_SECONDS", "600")
        )
        log_buffer_size = int(
            os.environ.get("SATNAV_MODEL_LOG_BUFFER_SIZE", "2000")
        )
        deploy_response_timeout_s = float(
            os.environ.get("SATNAV_MODEL_DEPLOY_RESPONSE_TIMEOUT_SECONDS", "120")
        )
        client = cls(
            repo_root=Path(configured_root) if configured_root else repo_root,
            model_name=model_name,
            model_path=Path(model_path_value) if model_path_value else None,
            session_root=Path(session_root_value) if session_root_value else None,
            startup_timeout_s=startup_timeout_s,
            deploy_python=os.environ.get("SATNAV_DEPLOY_PYTHON") or sys.executable,
            log_buffer_size=log_buffer_size,
        )
        client.deploy_response_timeout_s = deploy_response_timeout_s
        return client

    def start_in_background(self) -> None:
        """Begin model loading without blocking FastAPI startup."""
        with self._lock:
            if self._loader_thread is not None and self._loader_thread.is_alive():
                return
            self._status.update(
                {
                    "state": "loading",
                    "loaded": False,
                    "process_running": False,
                    "error": None,
                    "started_at": now_shanghai_iso(),
                    "ready_at": None,
                }
            )
        self._stop_event.clear()
        self._append_log(
            stream="system",
            level="info",
            message="开始后台加载 SwiftVLN 模型",
        )
        self._loader_thread = threading.Thread(
            target=self._load_process,
            name="swiftvln-model-loader",
            daemon=True,
        )
        self._loader_thread.start()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            process = self.process
            process_running = process is not None and process.poll() is None
            if self._status["state"] == "ready" and not process_running:
                return_code = process.returncode if process is not None else None
                self._status.update(
                    {
                        "state": "error",
                        "loaded": False,
                        "process_running": False,
                        "error": (
                            "deploy process exited after model load; "
                            f"exit code={return_code}"
                        ),
                    }
                )
            payload = dict(self._status)
            payload["process_running"] = process_running
            if self._stderr_tail:
                payload["stderr_tail"] = list(self._stderr_tail)
            return payload

    def logs(self, after_sequence: int = 0, limit: int = 500) -> Dict[str, Any]:
        """Return ordered model-process logs for API and WebSocket consumers."""
        with self._lock:
            records = [
                dict(record)
                for record in self._logs
                if record["sequence"] > after_sequence
            ][:limit]
            latest_sequence = self._log_sequence
        return {
            "logs": records,
            "latest_sequence": latest_sequence,
            "has_more": bool(records)
            and records[-1]["sequence"] < latest_sequence,
        }

    def session_info(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "session_started": self._session_started,
                "session_id": self._active_session_id,
                "instruction": self._active_instruction,
                "state": self._deploy_session_state,
            }

    def start_session(
        self,
        instruction: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved_session_id = session_id or f"satnav-{uuid4().hex[:8]}"
        response = self.send_command(
            {
                "type": "start",
                "instruction": instruction,
                "session_id": resolved_session_id,
            },
            expected_type="start",
        )
        with self._lock:
            self._session_started = True
            self._active_instruction = instruction
            self._active_session_id = response.get("session_id", resolved_session_id)
            self._deploy_session_state = response.get("state")
        return response

    def send_image(self, image_path: str) -> Dict[str, Any]:
        response = self.send_command(
            {"type": "image", "image_path": image_path},
            expected_type="image",
        )
        with self._lock:
            self._deploy_session_state = response.get("state")
        return response

    def send_command(
        self,
        command: Dict[str, Any],
        expected_type: str,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        with self._command_lock:
            if self._stop_event.is_set():
                raise RuntimeError("deploy client is stopping")
            process = self.process
            if process is None or process.stdin is None or process.poll() is not None:
                raise RuntimeError("deploy process is not running")
            payload = json.dumps(command, ensure_ascii=False)
            process.stdin.write(payload + "\n")
            process.stdin.flush()
            response = self._wait_for_message(
                {expected_type},
                timeout_s or self.deploy_response_timeout_s,
            )
            return response

    def _load_process(self) -> None:
        try:
            if not self.script_path.is_file():
                raise FileNotFoundError(
                    f"SwiftVLN deploy script not found: {self.script_path}"
                )
            deploy_output_root = self._prepare_model_registry()
            self.session_root.mkdir(parents=True, exist_ok=True)
            command = ["bash", str(self.script_path)]
            if self.model_name:
                command.extend([self.model_name, str(self.session_root)])
            self._append_log(
                stream="system",
                level="info",
                message=f"启动模型进程：{' '.join(command)}",
            )

            environment = os.environ.copy()
            environment["DEPLOY_PYTHON"] = self.deploy_python
            environment["DEPLOY_OUTPUT_ROOT"] = str(deploy_output_root)
            environment["PYTHONUNBUFFERED"] = "1"
            self.process = subprocess.Popen(
                command,
                cwd=self.repo_root,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            with self._lock:
                self._status["process_running"] = True

            self._stdout_thread = threading.Thread(
                target=self._read_stdout,
                name="swiftvln-deploy-stdout",
                daemon=True,
            )
            self._stderr_thread = threading.Thread(
                target=self._read_stderr,
                name="swiftvln-deploy-stderr",
                daemon=True,
            )
            self._stdout_thread.start()
            self._stderr_thread.start()

            ready = self._wait_until_ready()
            self._append_log(
                stream="system",
                level="info",
                message=(
                    "SwiftVLN 模型加载完成："
                    f"model={ready.get('model_name')}, "
                    f"checkpoint={ready.get('checkpoint_path')}"
                ),
            )
            with self._lock:
                self._status.update(
                    {
                        "state": "ready",
                        "loaded": True,
                        "process_running": True,
                        "model_name": ready.get("model_name"),
                        "checkpoint_path": ready.get("checkpoint_path"),
                        "gpu": ready.get("gpu"),
                        "error": None,
                        "ready_at": now_shanghai_iso(),
                    }
                )
        except Exception as exc:
            if self._stop_event.is_set():
                return
            self._terminate_process(timeout_s=5.0)
            self._append_log(
                stream="system",
                level="error",
                message=f"SwiftVLN 模型加载失败：{type(exc).__name__}: {exc}",
            )
            with self._lock:
                self._status.update(
                    {
                        "state": "error",
                        "loaded": False,
                        "process_running": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    def _prepare_model_registry(self) -> Path:
        """Map an explicitly supplied flat HF directory into deploy layout."""
        if not self.model_name:
            raise ValueError("SATNAV_MODEL_NAME is required")
        if self.model_path is None:
            raise ValueError("SATNAV_MODEL_PATH is required")
        if not self.model_path.is_dir():
            raise FileNotFoundError(
                f"SATNAV_MODEL_PATH does not exist: {self.model_path}"
            )
        if not (self.model_path / "config.json").is_file():
            raise FileNotFoundError(
                f"model config.json not found: {self.model_path}"
            )
        has_weights = any(self.model_path.glob("*.safetensors")) or any(
            self.model_path.glob("*.bin")
        )
        if not has_weights:
            raise FileNotFoundError(
                f"model weights not found: {self.model_path}"
            )

        model_dir = self.model_registry_root / "swiftvln" / self.model_name
        checkpoint_link = model_dir / "checkpoint-1"
        model_dir.mkdir(parents=True, exist_ok=True)
        if checkpoint_link.is_symlink():
            if checkpoint_link.resolve() != self.model_path:
                checkpoint_link.unlink()
        elif checkpoint_link.exists():
            raise FileExistsError(
                "runtime model checkpoint path exists and is not a symlink: "
                f"{checkpoint_link}"
            )
        if not checkpoint_link.exists():
            checkpoint_link.symlink_to(
                self.model_path,
                target_is_directory=True,
            )
        self._append_log(
            stream="system",
            level="info",
            message=(
                f"模型目录映射完成：{checkpoint_link} -> {self.model_path}"
            ),
        )
        return self.model_registry_root

    def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            for line in self.process.stdout:
                text = line.rstrip("\n")
                if text:
                    self._append_log(
                        stream="stdout",
                        level="info",
                        message=text,
                    )
                self._stdout_queue.put(text)
        finally:
            self._stdout_queue.put(None)

    def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for line in self.process.stderr:
            text = line.rstrip("\n")
            if text:
                with self._lock:
                    self._stderr_tail.append(text)
                self._append_log(
                    stream="stderr",
                    level="info",
                    message=text,
                )

    def _append_log(self, stream: str, level: str, message: str) -> None:
        with self._lock:
            self._log_sequence += 1
            self._logs.append(
                {
                    "sequence": self._log_sequence,
                    "timestamp": now_shanghai_iso(),
                    "source": "swiftvln",
                    "stream": stream,
                    "level": level,
                    "message": message,
                }
            )
        output = sys.stderr if level == "error" else sys.stdout
        print(
            f"[SwiftVLN][{stream}][{level}] {message}",
            file=output,
            flush=True,
        )

    def _wait_until_ready(self) -> Dict[str, Any]:
        return self._wait_for_message({"ready"}, self.startup_timeout_s)

    def _wait_for_message(
        self,
        expected_types: Set[str],
        timeout_s: float,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while not self._stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"timed out waiting for deploy message: {sorted(expected_types)}"
                )
            try:
                line = self._stdout_queue.get(timeout=min(remaining, 0.2))
            except queue.Empty:
                if self.process is not None and self.process.poll() is not None:
                    raise RuntimeError(
                        f"deploy process exited with code {self.process.returncode}"
                    )
                continue
            if line is None:
                return_code = self.process.poll() if self.process is not None else None
                raise RuntimeError(
                    f"deploy process closed stdout; exit code={return_code}"
                )
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            message_type = str(payload.get("type", ""))
            if message_type in {"startup_error", "error"} or payload.get("ok") is False:
                raise RuntimeError(
                    f"deploy {message_type or 'error'}: {payload.get('error', payload)}"
                )
            if message_type in expected_types:
                return payload
        raise RuntimeError("deploy client stopped")

    def close(self, timeout_s: float = 10.0) -> None:
        """Stop the child model process during FastAPI shutdown."""
        self._append_log(
            stream="system",
            level="info",
            message="正在关闭 SwiftVLN 模型进程",
        )
        self._stop_event.set()
        self._terminate_process(timeout_s=timeout_s)
        with self._lock:
            self._status.update(
                {
                    "state": "stopped",
                    "loaded": False,
                    "process_running": False,
                }
            )

    def _terminate_process(self, timeout_s: float) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
