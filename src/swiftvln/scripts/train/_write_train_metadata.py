#!/usr/bin/env python3
"""Persist train metadata shared by train entrypoints."""

import json
import os
from pathlib import Path


def main() -> int:
    output_dir = os.environ.get("_OUTPUT_DIR", "")
    if not output_dir:
        return 0

    meta = {}
    url = os.environ.get("_SWANLAB_URL", "")
    if url:
        meta["swanlab_url"] = url
    project = os.environ.get("_SWANLAB_PROJECT", "")
    if project:
        meta["swanlab_project"] = project
    exp = os.environ.get("_SWANLAB_EXP", "")
    if exp:
        meta["swanlab_exp_name"] = exp

    if not meta:
        return 0

    out = Path(output_dir) / "train_metadata.json"
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved train metadata: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
