from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]


def _has_s2r_data_tools() -> bool:
    """Return whether this installation includes offline S2R data tooling."""
    try:
        return importlib.util.find_spec("swiftvln.s2r.data_generation") is not None
    except ModuleNotFoundError:
        return False


def _run_train(extra_args: Sequence[str]) -> int:
    from swiftvln.model.trainer import train_main

    train_main(list(extra_args))
    return 0


def _run_eval(extra_args: Sequence[str]) -> int:
    from swiftvln.model.eval import main as eval_main

    old_argv = sys.argv[:]
    try:
        sys.argv = ["swiftvln.model.eval", *extra_args]
        eval_main()
    finally:
        sys.argv = old_argv
    return 0


def _run_queue(target: str, extra_args: Sequence[str]) -> int:
    script = REPO_ROOT / "src" / "swiftvln" / "scripts" / target / f"{target}_queue.sh"
    completed = subprocess.run(
        ["bash", str(script), *extra_args],
        cwd=REPO_ROOT,
        check=False,
    )
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swiftvln", description="SwiftVLN command line")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Run SwiftVLN training")
    subparsers.add_parser("eval", help="Run SwiftVLN evaluation")

    p_queue = subparsers.add_parser("queue", help="Run queue orchestrations")
    p_queue.add_argument("target", choices=["train", "eval"], help="Queue type")

    if _has_s2r_data_tools():
        subparsers.add_parser(
            "s2r-data",
            help="Generate SatDronePair data for S2R alignment training",
            add_help=False,
        )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)

    if args.command == "train":
        return _run_train(extra)
    if args.command == "eval":
        return _run_eval(extra)
    if args.command == "queue":
        return _run_queue(args.target, extra)
    if args.command == "s2r-data":
        from swiftvln.s2r.data_generation import main as run_s2r_data

        return run_s2r_data(extra)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
