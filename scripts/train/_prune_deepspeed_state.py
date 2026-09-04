#!/usr/bin/env python3
"""Validate completed model checkpoints and prune DeepSpeed resume state."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CHECKPOINT_RE = re.compile(r"checkpoint-([1-9][0-9]*)")
VERSION_RE = re.compile(r"v[0-9]+-.+")
DEEPSPEED_STATE_RE = re.compile(r"global_step[0-9]+")
DEEPSPEED_HELPERS = ("latest", "zero_to_fp32.py")


class ValidationError(RuntimeError):
    """Raised when a checkpoint is not safe to prune."""


@dataclass(frozen=True)
class PrunePlan:
    checkpoint: Path
    state_dirs: tuple[Path, ...]
    helper_files: tuple[Path, ...]
    apparent_bytes: int
    model_artifact: str
    tensor_count: int | None


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValidationError(f"required JSON file is missing or empty: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"invalid JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValidationError(f"expected a JSON object: {path}")
    return value


def validate_safetensors(path: Path) -> set[str]:
    if not path.is_file() or path.stat().st_size <= 8:
        raise ValidationError(f"safetensors file is missing or empty: {path}")
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValidationError(f"truncated safetensors header length: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length <= 0 or header_length > path.stat().st_size - 8:
            raise ValidationError(f"invalid safetensors header length in {path}")
        if header_length > 128 * 1024 * 1024:
            raise ValidationError(f"unreasonably large safetensors header in {path}")
        try:
            header = json.loads(handle.read(header_length))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValidationError(f"invalid safetensors header in {path}: {error}") from error
    if not isinstance(header, dict):
        raise ValidationError(f"safetensors header is not an object: {path}")
    tensor_keys = {str(key) for key in header if key != "__metadata__"}
    if not tensor_keys:
        raise ValidationError(f"safetensors file contains no tensors: {path}")
    return tensor_keys


def validate_model_artifacts(checkpoint: Path) -> tuple[str, int | None]:
    config = read_json(checkpoint / "config.json")
    if not config.get("model_type"):
        raise ValidationError(f"config.json has no model_type: {checkpoint}")

    index_names = (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    for index_name in index_names:
        index_path = checkpoint / index_name
        if not index_path.exists():
            continue
        index = read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValidationError(f"weight_map is missing or empty: {index_path}")
        tensor_count = 0
        shard_keys: dict[str, set[str] | None] = {}
        for shard_name in sorted({str(name) for name in weight_map.values()}):
            if Path(shard_name).name != shard_name:
                raise ValidationError(f"unsafe shard path in {index_path}: {shard_name}")
            shard = checkpoint / shard_name
            if not shard.is_file() or shard.stat().st_size == 0:
                raise ValidationError(f"missing model shard: {shard}")
            shard_keys[shard_name] = (
                validate_safetensors(shard) if shard.suffix == ".safetensors" else None
            )
        for tensor_name, shard_name_value in weight_map.items():
            shard_name = str(shard_name_value)
            keys = shard_keys[shard_name]
            if keys is not None and str(tensor_name) not in keys:
                raise ValidationError(
                    f"tensor {tensor_name!r} is absent from indexed shard {shard_name}"
                )
            tensor_count += 1
        return index_name, tensor_count

    single_names = (
        "model.safetensors",
        "adapter_model.safetensors",
        "pytorch_model.bin",
        "adapter_model.bin",
    )
    for name in single_names:
        model_path = checkpoint / name
        if not model_path.exists():
            continue
        if not model_path.is_file() or model_path.stat().st_size == 0:
            raise ValidationError(f"model artifact is missing or empty: {model_path}")
        tensor_count = (
            len(validate_safetensors(model_path))
            if model_path.suffix == ".safetensors"
            else None
        )
        return name, tensor_count
    raise ValidationError(f"no standard model artifact found in {checkpoint}")


def apparent_size(path: Path) -> int:
    total = 0
    for root, directories, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in directories + files:
            total += (root_path / name).lstat().st_size
    return total


def discover_checkpoints(output_dir: Path, not_before_epoch: float | None) -> list[Path]:
    output_dir = output_dir.resolve(strict=True)
    candidates: list[Path] = []
    for candidate in output_dir.rglob("checkpoint-*"):
        match = CHECKPOINT_RE.fullmatch(candidate.name)
        if match is None or not candidate.is_dir():
            continue
        if candidate.is_symlink():
            raise ValidationError(f"checkpoint directory must not be a symlink: {candidate}")
        relative_parts = candidate.relative_to(output_dir).parts
        if len(relative_parts) == 1:
            pass
        elif len(relative_parts) == 2 and VERSION_RE.fullmatch(relative_parts[0]):
            pass
        else:
            continue
        if not_before_epoch is not None and candidate.stat().st_mtime + 5 < not_before_epoch:
            continue
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(output_dir)
        except ValueError as error:
            raise ValidationError(f"checkpoint escapes output directory: {candidate}") from error
        candidates.append(resolved)
    candidates.sort(
        key=lambda path: (
            int(CHECKPOINT_RE.fullmatch(path.name).group(1)),
            path.stat().st_mtime,
        )
    )
    if not candidates:
        raise ValidationError(
            f"no checkpoint created during this training invocation was found under {output_dir}"
        )
    return candidates


def build_plan(checkpoint: Path) -> PrunePlan:
    match = CHECKPOINT_RE.fullmatch(checkpoint.name)
    if match is None:
        raise ValidationError(f"unexpected checkpoint directory name: {checkpoint}")
    trainer_state = read_json(checkpoint / "trainer_state.json")
    if int(trainer_state.get("global_step", -1)) != int(match.group(1)):
        raise ValidationError(
            f"trainer_state global_step does not match {checkpoint.name}: {checkpoint}"
        )
    model_artifact, tensor_count = validate_model_artifacts(checkpoint)

    state_dirs: list[Path] = []
    for child in checkpoint.iterdir():
        if DEEPSPEED_STATE_RE.fullmatch(child.name) is None:
            continue
        if child.is_symlink() or not child.is_dir():
            raise ValidationError(f"DeepSpeed state target is not a real directory: {child}")
        resolved = child.resolve(strict=True)
        if resolved.parent != checkpoint:
            raise ValidationError(f"DeepSpeed state escapes checkpoint: {child}")
        state_dirs.append(resolved)

    helper_files: list[Path] = []
    for name in DEEPSPEED_HELPERS:
        helper = checkpoint / name
        if not helper.exists() and not helper.is_symlink():
            continue
        if helper.is_dir() and not helper.is_symlink():
            raise ValidationError(f"DeepSpeed helper unexpectedly is a directory: {helper}")
        helper_files.append(helper)

    return PrunePlan(
        checkpoint=checkpoint,
        state_dirs=tuple(sorted(state_dirs)),
        helper_files=tuple(helper_files),
        apparent_bytes=sum(apparent_size(path) for path in state_dirs),
        model_artifact=model_artifact,
        tensor_count=tensor_count,
    )


def apply_plan(plan: PrunePlan) -> None:
    for state_dir in plan.state_dirs:
        shutil.rmtree(state_dir)
    for helper in plan.helper_files:
        helper.unlink(missing_ok=True)
    for target in (*plan.state_dirs, *plan.helper_files):
        if target.exists() or target.is_symlink():
            raise ValidationError(f"cleanup verification failed: {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--not-before-epoch", type=float)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform deletion. Without this flag, only print the validated plan.",
    )
    args = parser.parse_args()

    try:
        checkpoints = discover_checkpoints(args.output_dir, args.not_before_epoch)
        # Validate every candidate before deleting anything.
        plans = [build_plan(checkpoint) for checkpoint in checkpoints]
        for plan in plans:
            tensor_text = "unknown" if plan.tensor_count is None else str(plan.tensor_count)
            print(
                f"[DeepSpeed cleanup] validated checkpoint={plan.checkpoint} "
                f"model={plan.model_artifact} tensors={tensor_text} "
                f"state_dirs={len(plan.state_dirs)} "
                f"apparent_gib={plan.apparent_bytes / (1024 ** 3):.2f}",
                flush=True,
            )
        if not args.apply:
            print("[DeepSpeed cleanup] dry run only; nothing was deleted.", flush=True)
            return 0
        for plan in plans:
            apply_plan(plan)
        total_bytes = sum(plan.apparent_bytes for plan in plans)
        print(
            f"[DeepSpeed cleanup] complete checkpoints={len(plans)} "
            f"freed_apparent_gib={total_bytes / (1024 ** 3):.2f}",
            flush=True,
        )
        return 0
    except (OSError, TypeError, ValueError, ValidationError) as error:
        print(f"[DeepSpeed cleanup] refused: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
