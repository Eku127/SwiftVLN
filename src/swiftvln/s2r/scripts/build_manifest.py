#!/usr/bin/env python
"""Build Stage-A manifest for sim-to-real alignment."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_CURRENT_DIR)))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from swiftvln.s2r.arguments import parse_manifest_args
from swiftvln.s2r.dataset import build_manifest_records, write_manifest


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
