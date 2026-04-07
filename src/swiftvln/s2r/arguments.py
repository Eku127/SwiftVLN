"""Argument helpers for Stage-A sim-to-real alignment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass


def str2bool(value):
    """Parse bool-like CLI values."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


@dataclass
class ManifestConfig:
    data_root: str
    output_path: str
    val_ratio: float = 0.1
    seed: int = 42
    skip_missing: bool = True

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    manifest_path: str
    teacher_model_path: str
    output_dir: str
    train_split: str = "train"
    val_split: str = "val"
    batch_size: int = 8
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    epochs: int = 10
    max_steps: int = 0
    grad_accum_steps: int = 1
    grad_clip_norm: float = 1.0
    temperature: float = 0.07
    adapter_layers: int = 2
    adapter_heads: int = 8
    adapter_mlp_ratio: float = 4.0
    adapter_dropout: float = 0.0
    projection_dim: int = 512
    projection_hidden_dim: int = 0
    seed: int = 42
    max_train_samples: int = 0
    max_val_samples: int = 0
    device: str = "auto"
    teacher_dtype: str = "auto"
    save_every_steps: int = 0
    eval_every_steps: int = 0
    num_keep_checkpoints: int = 2
    resume_checkpoint: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class EvalConfig:
    manifest_path: str
    checkpoint_path: str
    teacher_model_path: str = ""
    split: str = "val"
    batch_size: int = 8
    num_workers: int = 4
    max_samples: int = 0
    device: str = "auto"
    teacher_dtype: str = "auto"

    def to_dict(self):
        return asdict(self)


def build_manifest_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage-A s2r manifest.")
    parser.add_argument("--data_root", required=True, help="SatDronePair root directory.")
    parser.add_argument("--output_path", required=True, help="Output manifest .jsonl path.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation group ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split generation.")
    parser.add_argument(
        "--skip_missing",
        type=str2bool,
        default=True,
        help="Skip rows with missing exported images.",
    )
    return parser


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Stage-A sim-to-real adapter.")
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--teacher_model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--adapter_layers", type=int, default=2)
    parser.add_argument("--adapter_heads", type=int, default=8)
    parser.add_argument("--adapter_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--adapter_dropout", type=float, default=0.0)
    parser.add_argument("--projection_dim", type=int, default=512)
    parser.add_argument("--projection_hidden_dim", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--teacher_dtype", default="auto")
    parser.add_argument("--save_every_steps", type=int, default=0)
    parser.add_argument("--eval_every_steps", type=int, default=0)
    parser.add_argument("--num_keep_checkpoints", type=int, default=2)
    parser.add_argument("--resume_checkpoint", default="")
    return parser


def build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Stage-A sim-to-real adapter.")
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--teacher_model_path", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--teacher_dtype", default="auto")
    return parser


def parse_manifest_args(argv=None) -> ManifestConfig:
    args = build_manifest_parser().parse_args(argv)
    return ManifestConfig(**vars(args))


def parse_train_args(argv=None) -> TrainConfig:
    args = build_train_parser().parse_args(argv)
    return TrainConfig(**vars(args))


def parse_eval_args(argv=None) -> EvalConfig:
    args = build_eval_parser().parse_args(argv)
    return EvalConfig(**vars(args))
