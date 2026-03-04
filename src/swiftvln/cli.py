import argparse
import sys

from swiftvln.common.registry import MODEL_CHOICES
from swiftvln.runners.train import run_train
from swiftvln.runners.eval import run_eval
from swiftvln.runners.queue import run_queue_train, run_queue_eval

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swiftvln", description="SwiftVLN command line")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train", help="Run model training")
    p_train.add_argument("--model", required=True, choices=list(MODEL_CHOICES))

    p_eval = subparsers.add_parser("eval", help="Run model evaluation")
    p_eval.add_argument("--model", required=True, choices=list(MODEL_CHOICES))

    p_queue = subparsers.add_parser("queue", help="Run queue orchestrations")
    p_queue.add_argument("target", choices=["train", "eval"], help="Queue type")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)

    if args.command == "train":
        return run_train(args.model, extra)
    if args.command == "eval":
        return run_eval(args.model, extra)
    if args.command == "queue":
        if args.target == "train":
            return run_queue_train(extra)
        return run_queue_eval(extra)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
