from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any, Dict, Optional
from uuid import uuid4

from PIL import Image

from swiftvln.deployment.model_resolver import SwiftVLNDeploySpec
from swiftvln.deployment.policy import SwiftVLNBaselinePolicy


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class DeploymentSessionError(RuntimeError):
    """Raised for protocol or state errors inside the deployment session."""


@dataclass
class SessionPaths:
    root: Path
    images_dir: Path
    meta_path: Path
    events_path: Path
    summary_path: Path


class SwiftVLNDeploySession:
    def __init__(
        self,
        spec: SwiftVLNDeploySpec,
        policy: SwiftVLNBaselinePolicy,
        session_root: str | Path,
        gpu_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.spec = spec
        self.policy = policy
        self.session_root = Path(session_root).resolve()
        self.gpu_metadata = gpu_metadata or {}

        self.state = "waiting_start"
        self.session_id: Optional[str] = None
        self.session_paths: Optional[SessionPaths] = None
        self.instruction: Optional[str] = None
        self.pending_actions: list[int] = []
        self.event_handle = None
        self.event_count = 0
        self.image_count = 0
        self.inference_count = 0
        self.started_at: Optional[str] = None
        self.ended_at: Optional[str] = None

    def current_session_id(self) -> Optional[str]:
        return self.session_id

    def start(self, instruction: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        if self.state != "waiting_start":
            raise DeploymentSessionError("start is only allowed in waiting_start state")
        if not instruction or not instruction.strip():
            raise DeploymentSessionError("instruction must be a non-empty string")

        resolved_session_id = session_id or self._generate_session_id()
        resolved_instruction = instruction.strip()
        started_at = _utc_now()
        self.policy.reset()
        try:
            session_paths = self._create_session_paths(resolved_session_id)
        except FileExistsError as exc:
            raise DeploymentSessionError(
                f"session_id already exists on disk: {resolved_session_id}"
            ) from exc

        self.session_id = resolved_session_id
        self.instruction = resolved_instruction
        self.started_at = started_at
        self.state = "waiting_image"
        self.pending_actions = []
        self.session_paths = session_paths
        self._write_session_meta()
        self.event_handle = self.session_paths.events_path.open("a", encoding="utf-8", buffering=1)

        response = {
            "type": "start",
            "ok": True,
            "session_id": self.session_id,
            "state": self.state,
            "session_dir": str(self.session_paths.root),
        }
        self._write_event(
            kind="start",
            command={"type": "start", "instruction": self.instruction, "session_id": self.session_id},
            response=response,
        )
        return response

    def handle_image(self, image_path: str) -> Dict[str, Any]:
        if self.state not in {"waiting_image", "waiting_feedback"}:
            raise DeploymentSessionError("image is only allowed after start and before end")
        if self.session_paths is None or self.session_id is None or self.instruction is None:
            raise DeploymentSessionError("session is not initialized")

        source_path = Path(image_path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"image_path does not exist: {source_path}")

        completed_action = self.pending_actions[0] if self.pending_actions else None
        will_infer_after_feedback = completed_action is not None and len(self.pending_actions) == 1
        if completed_action is None:
            image_role = "infer_input"
        elif will_infer_after_feedback:
            image_role = "feedback_infer_input"
        else:
            image_role = "feedback"

        stored_image_path = self._store_image(source_path, image_role=image_role)
        with Image.open(stored_image_path) as image:
            current_image = image.convert("RGB")

        if completed_action is not None:
            self.pending_actions.pop(0)
            self.policy.record_completed_action(completed_action)

        self.policy.observe_image(current_image)
        performed_inference = False
        raw_action_text = ""
        actions: list[int] = []
        if not self.pending_actions:
            inference = self.policy.infer_actions(self.instruction, current_image)
            performed_inference = True
            raw_action_text = inference.raw_action_text
            actions = list(inference.actions)
            self.pending_actions = list(actions)
            self.inference_count += 1

        next_action = self.pending_actions[0] if self.pending_actions else None
        self.state = "waiting_feedback" if self.pending_actions else "waiting_image"

        response = {
            "type": "image",
            "ok": True,
            "session_id": self.session_id,
            "state": self.state,
            "performed_inference": performed_inference,
            "raw_action_text": raw_action_text,
            "actions": actions,
            "next_action": next_action,
            "remaining_actions": list(self.pending_actions),
            "completed_action": completed_action,
            "stored_image_path": str(stored_image_path),
        }
        self._write_event(
            kind="image",
            command={"type": "image", "image_path": str(source_path)},
            response=response,
            extra={
                "source_image_path": str(source_path),
                "policy": self.policy.snapshot(),
            },
        )
        return response

    def end(self, reason: Optional[str] = None) -> Dict[str, Any]:
        if self.state == "waiting_start":
            raise DeploymentSessionError("end is only allowed after start")
        if self.state == "closed":
            raise DeploymentSessionError("session is already closed")
        if self.session_paths is None or self.session_id is None:
            raise DeploymentSessionError("session is not initialized")

        self.ended_at = _utc_now()
        summary = {
            "session_id": self.session_id,
            "state": "closed",
            "reason": reason or "",
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "instruction": self.instruction,
            "image_count": self.image_count,
            "inference_count": self.inference_count,
            "pending_actions": list(self.pending_actions),
            "executed_actions": self.policy.snapshot()["executed_actions"],
            "model": self.spec.to_dict(),
            "gpu": self.gpu_metadata,
            "policy": self.policy.snapshot(),
        }
        self.session_paths.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        self.state = "closed"
        response = {
            "type": "end",
            "ok": True,
            "session_id": self.session_id,
            "state": "closed",
            "reason": reason or "",
            "summary_path": str(self.session_paths.summary_path),
        }
        self._write_event(
            kind="end",
            command={"type": "end", "reason": reason or ""},
            response=response,
            extra={"policy": self.policy.snapshot()},
        )

        if self.event_handle is not None:
            self.event_handle.close()
            self.event_handle = None

        self.policy.close()
        return response

    def close_if_needed(self) -> None:
        if self.event_handle is not None:
            self.event_handle.close()
            self.event_handle = None
        if getattr(self.policy, "model", None) is not None:
            self.policy.close()

    def _generate_session_id(self) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        return f"deploy-{timestamp}-{uuid4().hex[:8]}"

    def _create_session_paths(self, session_id: str) -> SessionPaths:
        root = self.session_root / session_id
        images_dir = root / "images"
        images_dir.mkdir(parents=True, exist_ok=False)
        return SessionPaths(
            root=root,
            images_dir=images_dir,
            meta_path=root / "session_meta.json",
            events_path=root / "events.jsonl",
            summary_path=root / "session_summary.json",
        )

    def _write_session_meta(self) -> None:
        if self.session_paths is None:
            raise DeploymentSessionError("session paths are not initialized")
        meta = {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "instruction": self.instruction,
            "model": self.spec.to_dict(),
            "gpu": self.gpu_metadata,
        }
        self.session_paths.meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _store_image(self, source_path: Path, image_role: str) -> Path:
        if self.session_paths is None:
            raise DeploymentSessionError("session paths are not initialized")
        self.image_count += 1
        suffix = source_path.suffix or ".jpg"
        stored_name = f"{self.image_count:06d}_{image_role}{suffix.lower()}"
        stored_path = self.session_paths.images_dir / stored_name
        shutil.copy2(source_path, stored_path)
        return stored_path

    def _write_event(
        self,
        kind: str,
        command: Dict[str, Any],
        response: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.event_handle is None:
            return
        self.event_count += 1
        payload = {
            "timestamp": _utc_now(),
            "event_index": self.event_count,
            "kind": kind,
            "state": self.state,
            "command": command,
            "response": response,
        }
        if extra:
            payload.update(extra)
        self.event_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
