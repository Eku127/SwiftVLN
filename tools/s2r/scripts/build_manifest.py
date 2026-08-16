#!/usr/bin/env python
"""Build Stage-A manifest for sim-to-real alignment."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tools.s2r.arguments import parse_manifest_args  # noqa: E402
from tools.s2r.dataset import build_manifest_records, write_manifest  # noqa: E402


def main(argv=None):
    args = parse_manifest_args(argv)
    records = build_manifest_records(
        args.data_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
        skip_missing=args.skip_missing,
    )
    write_manifest(records, args.output_path)

    by_source = Counter(record["dataset"] for record in records)
    by_split = Counter(record["split"] for record in records)
    print(json.dumps({
        "output_path": args.output_path,
        "num_records": len(records),
        "by_source": dict(by_source),
        "by_split": dict(by_split),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
