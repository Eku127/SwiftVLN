"""Installed SwiftVLN evaluation entrypoint."""

from __future__ import annotations

import os

os.environ.setdefault(
    "__EGL_VENDOR_LIBRARY_FILENAMES",
    "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
)

from .arguments import build_summary_extras, parse_eval_args
from .runner import SwiftVLNEvaluationRunner


def main(argv: list[str] | None = None) -> None:
    args = parse_eval_args(argv)
    SwiftVLNEvaluationRunner(args, build_summary_extras(args)).run()
