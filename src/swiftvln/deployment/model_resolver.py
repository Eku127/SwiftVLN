from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Dict


class DeploymentModelSpecError(ValueError):
    """Raised when a model name cannot be used for deployment."""


_BASELINE_PATTERN = re.compile(
    r"^overlapvln-"
    r"(?:(?P<env_type>habitat|satnav)-)?"
    r"(?P<stage>stage\d+)-"
    r"(?P<model_size>\d+[bB])-"
    r"(?P<num_epochs>\d+)ep-"
    r"f(?P<num_frames>\d+)s(?P<num_future_steps>\d+)-"
    r"overlap(?P<num_overlap>\d+)-"
    r"pf-h(?P<num_history>\d+)"
    r"(?P<random>-random)?"
    r"(?P<nomem>-nomem)?-"
    r"b(?P<log_base>[0-9.]+)-"
    r"(?P<method>pool|tome)-"
    r"s(?P<compress_stride>\d+)-"
    r"(?P<embed_slot>[^-]+)"
    r"(?P<suffix>.*)$"
)

_UNSUPPORTED_MARKERS = {
    "-map-g": "map memory deployment is not supported yet",
    "-gtc-k": "gtc deployment is not supported yet",
    "-sgtc-k": "segment gtc deployment is not supported yet",
    "-initial-": "initial system prompt deployment is not supported yet",
}


@dataclass(frozen=True)
class OverlapVLNDeploySpec:
    model_name: str
    model_dir: str
    checkpoint_path: str
    env_type: str
    stage: str
    model_size: str
    num_epochs: int
    num_frames: int
    num_future_steps: int
    num_overlap: int
    num_history: int
    log_base: float
    use_random: bool
    compress_stride: int
    history_method: str
    embed_slot: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _checkpoint_step(path: Path) -> int:
    match = re.match(r"checkpoint-(\d+)$", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def _find_latest_checkpoint(model_dir: Path) -> Path:
    latest_checkpoint: Path | None = None

    version_dirs = sorted(
        [path for path in model_dir.iterdir() if path.is_dir() and path.name.startswith("v")],
        key=lambda path: path.name,
    )
    for version_dir in version_dirs:
        checkpoints = sorted(
            [
                path
                for path in version_dir.iterdir()
                if path.is_dir() and path.name.startswith("checkpoint-")
            ],
            key=_checkpoint_step,
        )
        if checkpoints:
            latest_checkpoint = checkpoints[-1]

    if latest_checkpoint is None:
        checkpoints = sorted(
            [
                path
                for path in model_dir.iterdir()
                if path.is_dir() and path.name.startswith("checkpoint-")
            ],
            key=_checkpoint_step,
        )
        if checkpoints:
            latest_checkpoint = checkpoints[-1]

    if latest_checkpoint is None:
        raise DeploymentModelSpecError(
            f"No checkpoint-* directory found under model directory: {model_dir}"
        )
    return latest_checkpoint


def _check_checkpoint_integrity(checkpoint_path: Path) -> None:
    config_path = checkpoint_path / "config.json"
    if not config_path.is_file():
        raise DeploymentModelSpecError(
            f"Checkpoint is incomplete: missing config.json at {checkpoint_path}"
        )

    has_weights = any(checkpoint_path.glob("*.safetensors")) or any(checkpoint_path.glob("*.bin"))
    if not has_weights:
        raise DeploymentModelSpecError(
            f"Checkpoint is incomplete: missing model weights at {checkpoint_path}"
        )


def _reject_unsupported_model_name(model_name: str) -> None:
    for marker, message in _UNSUPPORTED_MARKERS.items():
        if marker in model_name:
            raise DeploymentModelSpecError(message)


def resolve_overlapvln_deploy_spec(
    repo_root: str | Path,
    model_name: str,
    output_root: str | Path | None = None,
) -> OverlapVLNDeploySpec:
    repo_root = Path(repo_root).resolve()
    output_root = Path(output_root).resolve() if output_root is not None else repo_root / "output"

    if not model_name.startswith("overlapvln-"):
        raise DeploymentModelSpecError(
            f"Unsupported model name: {model_name}. Expected overlapvln-*"
        )

    _reject_unsupported_model_name(model_name)
    match = _BASELINE_PATTERN.match(model_name)
    if match is None:
        raise DeploymentModelSpecError(
            "Unsupported overlapvln deploy name. "
            "Only baseline per_frame names with pool/noembed are supported."
        )

    values = match.groupdict()
    env_type = values["env_type"] or "habitat"
    history_method = values["method"]
    embed_slot = values["embed_slot"]
    num_frames = int(values["num_frames"])
    num_future_steps = int(values["num_future_steps"])
    num_overlap = int(values["num_overlap"])
    num_history = int(values["num_history"])

    if history_method != "pool":
        raise DeploymentModelSpecError(
            f"Unsupported history compression '{history_method}'. Only pool is supported."
        )
    if embed_slot != "noembed":
        raise DeploymentModelSpecError(
            f"Unsupported embedding slot '{embed_slot}'. Only noembed is supported."
        )
    if num_future_steps <= 0 or num_frames <= 0:
        raise DeploymentModelSpecError("num_frames and num_future_steps must be positive.")
    if num_frames % num_future_steps != 0:
        raise DeploymentModelSpecError(
            "num_frames must be divisible by num_future_steps for deployment."
        )
    if num_overlap < 0 or num_overlap >= num_frames:
        raise DeploymentModelSpecError("num_overlap must satisfy 0 <= num_overlap < num_frames.")
    if num_overlap % num_future_steps != 0:
        raise DeploymentModelSpecError(
            "num_overlap must be divisible by num_future_steps for deployment."
        )

    model_dir = output_root / "overlapvln" / model_name
    if not model_dir.is_dir():
        raise DeploymentModelSpecError(f"Model directory does not exist: {model_dir}")

    checkpoint_path = _find_latest_checkpoint(model_dir)
    _check_checkpoint_integrity(checkpoint_path)

    return OverlapVLNDeploySpec(
        model_name=model_name,
        model_dir=str(model_dir),
        checkpoint_path=str(checkpoint_path),
        env_type=env_type,
        stage=values["stage"],
        model_size=values["model_size"].lower(),
        num_epochs=int(values["num_epochs"]),
        num_frames=num_frames,
        num_future_steps=num_future_steps,
        num_overlap=num_overlap,
        num_history=num_history,
        log_base=float(values["log_base"]),
        use_random=bool(values["random"]),
        compress_stride=int(values["compress_stride"]),
        history_method=history_method,
        embed_slot=embed_slot,
    )
