#!/usr/bin/env python3
"""Tail a training log and forward parsed metrics to SwanLab.

This sidecar is intentionally decoupled from the training conda env.
It watches the plain-text training log, extracts dict-style metric lines
emitted by HuggingFace Trainer, and logs them to SwanLab from a separate
Python environment.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import pathlib
import re
import sys
import time
from typing import Dict, Optional


DICT_LINE_RE = re.compile(r"\{.*\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", default="cloud")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--idle-timeout", type=float, default=600.0)
    return parser.parse_args()


def wait_for_log(path: pathlib.Path, poll_interval: float, idle_timeout: float) -> None:
    deadline = time.time() + idle_timeout
    while not path.exists():
        if time.time() > deadline:
            raise TimeoutError(f"log file not found within timeout: {path}")
        time.sleep(poll_interval)


def extract_metrics(line: str) -> Optional[Dict[str, float]]:
    match = DICT_LINE_RE.search(line)
    if not match:
        return None
    try:
        data = ast.literal_eval(match.group(0))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if "loss" not in data:
        return None
    metrics: Dict[str, float] = {}
    for key, value in data.items():
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics


def main() -> int:
    args = parse_args()
    log_path = pathlib.Path(args.log_path)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import swanlab
    except Exception as exc:  # pragma: no cover
        print(f"failed to import swanlab: {exc}", file=sys.stderr, flush=True)
        return 1

    wait_for_log(log_path, args.poll_interval, args.idle_timeout)

    run = swanlab.init(
        project=args.project,
        experiment_name=args.run_name,
        mode=args.mode,
        config={
            "output_dir": str(output_dir),
            "log_path": str(log_path),
            "sidecar": True,
        },
    )

    run_url = getattr(run, "url", "") or ""
    metadata = {
        "swanlab_project": args.project,
        "swanlab_exp_name": args.run_name,
    }
    if run_url:
        metadata["swanlab_url"] = run_url
    (output_dir / "train_metadata.json").write_text(json.dumps(metadata, indent=2))

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        f.seek(0, os.SEEK_END)
        last_activity = time.time()
        while True:
            line = f.readline()
            if line:
                last_activity = time.time()
                metrics = extract_metrics(line)
                if metrics:
                    swanlab.log(metrics)
                continue

            if time.time() - last_activity > args.idle_timeout:
                break
            time.sleep(args.poll_interval)

    swanlab.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
