"""Single-shot model inference: RTMP capture -> 448x448 -> deploy image."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any, Dict
from uuid import uuid4

from app.adapters.rtmp_frame_utils import prepare_model_image
from app.adapters.rtmp_stream_checker import RtmpStreamChecker
from app.adapters.swiftvln_deploy_client import SwiftVLNDeployClient
from app.media_cache import ModelInputCache, model_input_cache
from app.time_utils import now_shanghai_iso


class ModelInferenceError(Exception):
    """User-facing inference precondition failure."""


class ModelInferenceService:
    """Capture one RTMP frame and send it to the deploy JSONL session."""

    def __init__(
        self,
        model_client: SwiftVLNDeployClient,
        rtmp_checker: RtmpStreamChecker,
        *,
        frame_timeout_s: float = 10.0,
        resize_mode: str = "center-crop",
        input_root: Path | None = None,
        media_cache: ModelInputCache | None = None,
    ) -> None:
        self.model_client = model_client
        self.rtmp_checker = rtmp_checker
        self.frame_timeout_s = frame_timeout_s
        self.resize_mode = resize_mode
        self.input_root = (
            input_root
            or model_client.session_root / "inference_inputs"
        ).resolve()
        self.media_cache = media_cache or model_input_cache

    @classmethod
    def from_environment(
        cls,
        model_client: SwiftVLNDeployClient,
        rtmp_checker: RtmpStreamChecker,
    ) -> "ModelInferenceService":
        frame_timeout_s = float(
            os.environ.get("SATNAV_RTMP_FRAME_TIMEOUT_SECONDS", "10")
        )
        resize_mode = os.environ.get(
            "SATNAV_MODEL_INPUT_RESIZE_MODE",
            "center-crop",
        ).strip()
        input_root_value = os.environ.get("SATNAV_MODEL_INFERENCE_INPUT_ROOT")
        return cls(
            model_client=model_client,
            rtmp_checker=rtmp_checker,
            frame_timeout_s=frame_timeout_s,
            resize_mode=resize_mode,
            input_root=Path(input_root_value) if input_root_value else None,
        )

    def run(self, instruction: str) -> Dict[str, Any]:
        resolved_instruction = instruction.strip()
        if not resolved_instruction:
            raise ModelInferenceError("instruction must not be empty")

        model_status = self.model_client.status()
        if model_status.get("state") != "ready":
            raise ModelInferenceError(
                f"model is not ready (state={model_status.get('state')})"
            )

        rtmp_status = self.rtmp_checker.check()
        if not rtmp_status.get("stream_active"):
            error = rtmp_status.get("error") or "RTMP stream is not active"
            raise ModelInferenceError(error)

        session_info = self.model_client.session_info()
        started_new_session = False
        if not session_info["session_started"]:
            session_id = f"satnav-infer-{uuid4().hex[:8]}"
            self.model_client.start_session(
                resolved_instruction,
                session_id=session_id,
            )
            started_new_session = True
        elif session_info["instruction"] != resolved_instruction:
            raise ModelInferenceError(
                "deploy session already active with a different instruction"
            )

        capture_start = time.perf_counter()
        frame_sequence, frame_bgr = self.rtmp_checker.request_fresh_frame(
            self.frame_timeout_s
        )
        capture_ms = (time.perf_counter() - capture_start) * 1000.0

        preprocess_start = time.perf_counter()
        model_image = prepare_model_image(frame_bgr, self.resize_mode)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        self.input_root.mkdir(parents=True, exist_ok=True)
        image_path = self.input_root / f"infer_{uuid4().hex[:12]}_448.jpg"
        save_start = time.perf_counter()
        model_image.save(image_path, format="JPEG", quality=95)
        input_save_ms = (time.perf_counter() - save_start) * 1000.0
        self.media_cache.update(
            jpeg_bytes=image_path.read_bytes(),
            frame_sequence=frame_sequence,
            image_path=str(image_path),
        )

        deploy_start = time.perf_counter()
        deploy_response = self.model_client.send_image(str(image_path))
        deploy_response_ms = (time.perf_counter() - deploy_start) * 1000.0

        return {
            "instruction": resolved_instruction,
            "session_id": deploy_response.get("session_id"),
            "started_new_session": started_new_session,
            "performed_inference": bool(deploy_response.get("performed_inference")),
            "raw_action_text": deploy_response.get("raw_action_text", ""),
            "actions": deploy_response.get("actions", []),
            "next_action": deploy_response.get("next_action"),
            "remaining_actions": deploy_response.get("remaining_actions", []),
            "completed_action": deploy_response.get("completed_action"),
            "deploy_state": deploy_response.get("state"),
            "frame_sequence": frame_sequence,
            "image_path": str(image_path),
            "timing": {
                "capture_ms": round(capture_ms, 3),
                "preprocess_ms": round(preprocess_ms, 3),
                "input_save_ms": round(input_save_ms, 3),
                "deploy_response_ms": round(deploy_response_ms, 3),
                "capture_to_action_response_ms": round(
                    (time.perf_counter() - capture_start) * 1000.0,
                    3,
                ),
            },
            "timestamp": now_shanghai_iso(),
        }
