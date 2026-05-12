from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

from swiftvln.deployment.loader import load_swiftvln_policy
from swiftvln.deployment.model_resolver import (
    DeploymentModelSpecError,
    resolve_swiftvln_deploy_spec,
)
from swiftvln.deployment.session import DeploymentSessionError, SwiftVLNDeploySession


class JsonlDeploymentServer:
    def __init__(self, session: SwiftVLNDeploySession, spec: Any):
        self.session = session
        self.spec = spec

    def run(self, input_stream=sys.stdin, output_stream=sys.stdout) -> int:
        self._emit(
            {
                "type": "ready",
                "ok": True,
                "session_id": None,
                "state": self.session.state,
                "model_name": self.spec.model_name,
                "checkpoint_path": self.spec.checkpoint_path,
                "gpu": self.session.gpu_metadata,
            },
            output_stream=output_stream,
        )

        for raw_line in input_stream:
            line = raw_line.strip()
            if not line:
                continue
            response = self._handle_line(line)
            self._emit(response, output_stream=output_stream)
            if response.get("state") == "closed":
                return 0
        return 0

    def _handle_line(self, line: str) -> Dict[str, Any]:
        try:
            command = json.loads(line)
        except json.JSONDecodeError as exc:
            return self._error_response(f"invalid json: {exc.msg}")

        if not isinstance(command, dict):
            return self._error_response("command must be a JSON object")

        command_type = command.get("type")
        try:
            if command_type == "start":
                return self.session.start(
                    instruction=str(command.get("instruction", "")),
                    session_id=command.get("session_id"),
                )
            if command_type == "image":
                return self.session.handle_image(str(command.get("image_path", "")))
            if command_type == "end":
                return self.session.end(reason=command.get("reason"))
            return self._error_response(f"unsupported command type: {command_type!r}")
        except (DeploymentSessionError, FileNotFoundError, ValueError) as exc:
            return self._error_response(str(exc))
        except Exception as exc:  # pragma: no cover
            return self._error_response(f"internal error: {exc}")

    def _error_response(self, message: str) -> Dict[str, Any]:
        return {
            "type": "error",
            "ok": False,
            "session_id": self.session.current_session_id(),
            "state": self.session.state,
            "error": message,
        }

    @staticmethod
    def _emit(payload: Dict[str, Any], output_stream) -> None:
        output_stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        output_stream.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SwiftVLN local deployment JSONL server")
    parser.add_argument("--model-name", required=True, help="SwiftVLN experiment directory name")
    parser.add_argument(
        "--session-root",
        default="runtime/deploy/sessions",
        help="Directory for persisted deploy sessions",
    )
    parser.add_argument(
        "--output-root",
        default="output",
        help="Root directory that contains output/swiftvln/<model-name>",
    )
    return parser


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _gpu_metadata_from_env() -> Dict[str, Any]:
    data: Dict[str, Any] = {"mode": os.environ.get("SWIFTVLN_DEPLOY_GPU_MODE", "unknown")}
    if os.environ.get("SWIFTVLN_DEPLOY_PHYSICAL_GPU"):
        data["physical_index"] = os.environ["SWIFTVLN_DEPLOY_PHYSICAL_GPU"]
    if os.environ.get("SWIFTVLN_DEPLOY_GPU_UUID"):
        data["uuid"] = os.environ["SWIFTVLN_DEPLOY_GPU_UUID"]
    if os.environ.get("SWIFTVLN_DEPLOY_GPU_NAME"):
        data["name"] = os.environ["SWIFTVLN_DEPLOY_GPU_NAME"]
    data["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return data


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    try:
        spec = resolve_swiftvln_deploy_spec(
            repo_root=repo_root,
            model_name=args.model_name,
            output_root=repo_root / args.output_root,
        )
        policy = load_swiftvln_policy(spec)
        session = SwiftVLNDeploySession(
            spec=spec,
            policy=policy,
            session_root=repo_root / args.session_root,
            gpu_metadata=_gpu_metadata_from_env(),
        )
    except DeploymentModelSpecError as exc:
        sys.stdout.write(
            json.dumps(
                {
                    "type": "startup_error",
                    "ok": False,
                    "session_id": None,
                    "state": "waiting_start",
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        sys.stdout.flush()
        return 1
    except Exception as exc:
        sys.stdout.write(
            json.dumps(
                {
                    "type": "startup_error",
                    "ok": False,
                    "session_id": None,
                    "state": "waiting_start",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        sys.stdout.flush()
        return 1

    server = JsonlDeploymentServer(session=session, spec=spec)
    try:
        return server.run()
    finally:
        session.close_if_needed()


if __name__ == "__main__":
    raise SystemExit(main())
