from __future__ import annotations

import argparse
import sys
from typing import Sequence


def _run_train(extra_args: Sequence[str]) -> int:
    from swiftvln.training.sft.trainer import train_main

    train_main(list(extra_args))
    return 0


def _run_eval(extra_args: Sequence[str]) -> int:
    from swiftvln.evaluation.cli import main as eval_main

    eval_main(list(extra_args))
    return 0


def _run_generate_habitat_trajectories(extra_args: Sequence[str]) -> int:
    from swiftvln.data.habitat_trajectory import main as generate_main

    return generate_main(list(extra_args))


def _run_validate_habitat_trajectories(extra_args: Sequence[str]) -> int:
    from swiftvln.data.habitat_trajectory_validation import main as validate_main

    return validate_main(list(extra_args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swiftvln", description="SwiftVLN command line")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "train",
        add_help=False,
        help="Run SwiftVLN training",
    )
    subparsers.add_parser(
        "eval",
        add_help=False,
        help="Run SwiftVLN evaluation",
    )
    subparsers.add_parser(
        "generate-habitat-trajectories",
        add_help=False,
        help="Generate offline Habitat training trajectories",
    )
    subparsers.add_parser(
        "validate-habitat-trajectories",
        add_help=False,
        help="Validate offline Habitat training trajectories",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)

    if args.command == "train":
        return _run_train(extra)
    if args.command == "eval":
        return _run_eval(extra)
    if args.command == "generate-habitat-trajectories":
        return _run_generate_habitat_trajectories(extra)
    if args.command == "validate-habitat-trajectories":
        return _run_validate_habitat_trajectories(extra)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
