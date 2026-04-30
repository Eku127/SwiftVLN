from __future__ import annotations

import importlib
import os
from typing import Sequence

from swiftvln.common.registry import MODEL_TO_PACKAGE, ensure_supported_model
from swiftvln.deployment.gpu import DeploymentGPUError, select_idle_h100_from_system


def _configure_deploy_gpu() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ.setdefault("SWIFTVLN_DEPLOY_GPU_MODE", "preconfigured")
        return

    gpu = select_idle_h100_from_system()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu.index)
    os.environ["SWIFTVLN_DEPLOY_GPU_MODE"] = "auto_h100"
    os.environ["SWIFTVLN_DEPLOY_PHYSICAL_GPU"] = str(gpu.index)
    os.environ["SWIFTVLN_DEPLOY_GPU_UUID"] = gpu.uuid
    os.environ["SWIFTVLN_DEPLOY_GPU_NAME"] = gpu.name


def run_deploy(
    model: str,
    model_name: str,
    session_root: str,
    extra_args: Sequence[str],
) -> int:
    ensure_supported_model(model)

    if model != "overlapvln":
        raise ValueError("Deployment currently supports only overlapvln.")

    try:
        _configure_deploy_gpu()
    except DeploymentGPUError as exc:
        print(f"[deploy] GPU selection failed: {exc}", flush=True)
        return 1

    # Ensure model/template registration side effects are loaded after GPU pinning.
    importlib.import_module(MODEL_TO_PACKAGE[model])

    server_module = importlib.import_module("swiftvln.deployment.server")
    server_main = getattr(server_module, "main")
    deploy_args = [
        "--model-name",
        model_name,
        "--session-root",
        session_root,
        *list(extra_args),
    ]
    return int(server_main(deploy_args))
