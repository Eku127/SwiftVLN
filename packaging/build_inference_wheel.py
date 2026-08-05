#!/usr/bin/env python3
"""Build the inference wheel without offline S2R data-generation code."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = Path(__file__).resolve().parent / "inference" / "pyproject.toml"
S2R_DATA_PREFIX = "swiftvln/s2r/data_generation/"
REQUIRED_RUNTIME_FILES = {
    "swiftvln/common/embedding_enhancement/uav_adapter.py",
    "swiftvln/common/env/habitat.py",
    "swiftvln/common/utils/video_utils.py",
    "swiftvln/configs/satnav_task.yaml",
    "swiftvln/configs/vln_r2r.yaml",
    "swiftvln/habitat_extensions/measures.py",
    "swiftvln/s2r/model.py",
}


def _copy_ignore(source: str, names: Iterable[str]) -> set[str]:
    ignored = {
        name
        for name in names
        if name == "__pycache__" or name.endswith((".pyc", ".pyo"))
    }
    source_path = Path(source).resolve()
    if source_path == (REPO_ROOT / "src/swiftvln/s2r").resolve():
        ignored.add("data_generation")
    return ignored


def _validate_wheel(wheel_path: Path) -> None:
    with zipfile.ZipFile(wheel_path) as archive:
        members = set(archive.namelist())
        forbidden = sorted(
            member for member in members if member.startswith(S2R_DATA_PREFIX)
        )
        if forbidden:
            raise RuntimeError(
                "Inference wheel contains S2R data-generation files: "
                + ", ".join(forbidden)
            )

        missing = sorted(REQUIRED_RUNTIME_FILES - members)
        if missing:
            raise RuntimeError(
                "Inference wheel is missing required runtime files: "
                + ", ".join(missing)
            )


def build_inference_wheel(output_dir: Path) -> Path:
    """Build and validate one ``swiftvln-inference`` wheel."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="swiftvln-inference-build-") as tmpdir:
        staging_root = Path(tmpdir) / "project"
        staging_root.mkdir()
        shutil.copy2(PROFILE_PATH, staging_root / "pyproject.toml")
        shutil.copy2(REPO_ROOT / "README.md", staging_root / "README.md")
        shutil.copytree(
            REPO_ROOT / "src",
            staging_root / "src",
            ignore=_copy_ignore,
        )

        before = set(output_dir.glob("swiftvln_inference-*.whl"))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--disable-pip-version-check",
                "--no-build-isolation",
                "--no-deps",
                "--wheel-dir",
                str(output_dir),
                str(staging_root),
            ],
            check=True,
        )
        candidates = set(output_dir.glob("swiftvln_inference-*.whl"))
        if not candidates:
            raise RuntimeError("pip completed without producing an inference wheel")
        created = candidates - before
        wheel_path = max(created or candidates, key=lambda path: path.stat().st_mtime)

    _validate_wheel(wheel_path)
    return wheel_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "dist/inference",
        help="Directory that receives the validated wheel",
    )
    args = parser.parse_args(argv)
    wheel_path = build_inference_wheel(args.output_dir)
    print(f"Built inference wheel: {wheel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
