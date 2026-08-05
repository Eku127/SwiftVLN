# Copyright (c) Alibaba, Inc. and its affiliates.
"""SwiftVLN multi-environment evaluation CLI."""

from __future__ import annotations

# Set NVIDIA EGL before any optional Habitat import.
import os

os.environ.setdefault(
    "__EGL_VENDOR_LIBRARY_FILENAMES",
    "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
)

import argparse
from typing import Any

from swiftvln.experiment import EMBEDDING_MODES, SwiftVLNExperimentSpec
from swiftvln.model.eval_runner import SwiftVLNEvaluationRunner

DEFAULT_MODEL_TYPE = "swiftvln_qwen2_5_vl"


def create_eval_parser() -> argparse.ArgumentParser:
    """Build the single supported SwiftVLN evaluation interface."""
    parser = argparse.ArgumentParser(
        description="SwiftVLN Multi-Environment Evaluation"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to a SwiftVLN checkpoint or HF model directory",
    )
    parser.add_argument(
        "--model_type",
        default=DEFAULT_MODEL_TYPE,
        choices=["swiftvln_qwen2_5_vl", "swiftvln_qwen3_vl"],
        help="Registered SwiftVLN model type",
    )
    environment = parser.add_argument_group("environment")
    environment.add_argument(
        "--env-type",
        default="habitat",
        choices=["habitat", "satnav"],
        help="Evaluation environment",
    )
    environment.add_argument(
        "--habitat_config_path",
        default="configs/vln_r2r.yaml",
        help="Habitat YAML path relative to the SwiftVLN package or repository",
    )
    environment.add_argument(
        "--satnav-config",
        default="configs/satnav_task.yaml",
        help="SatNav YAML path relative to the SwiftVLN package or repository",
    )
    environment.add_argument(
        "--eval_split",
        default="val_unseen",
        help="Dataset split to evaluate",
    )

    window = parser.add_argument_group("window and memory")
    window.add_argument("--num_frames", type=int, default=32)
    window.add_argument("--num_history", type=int, default=8)
    window.add_argument("--num_future_steps", type=int, default=4)
    window.add_argument("--num_overlap", type=int, default=0)
    window.add_argument(
        "--memory_method",
        default="history",
        choices=["history", "map"],
        help="Use history frames or SatNav explored maps as memory",
    )
    window.add_argument(
        "--history_processor_type",
        default="per_frame",
        choices=["per_frame", "gtc", "segment_gtc"],
    )
    window.add_argument("--compress_stride", type=int, default=2)
    window.add_argument("--log_base", type=float, default=1.0)
    window.add_argument("--use_random", action="store_true")
    window.add_argument("--use_tome", action="store_true")
    window.add_argument("--gtc_output_tokens", type=int, default=512)
    window.add_argument("--gtc_temperature", type=float, default=0.1)
    window.add_argument("--gtc_num_iterations", type=int, default=1)
    window.add_argument(
        "--system_prompt_setting",
        default="vanilla",
        choices=["vanilla", "initial"],
    )

    map_group = parser.add_argument_group("map memory")
    map_group.add_argument("--map_global_side_m", type=float, default=1000.0)
    map_group.add_argument("--map_local_side_m", type=float, default=400.0)
    map_group.add_argument("--map_render_px", type=int, default=448)
    map_group.add_argument("--map_mask_method", default="dilate20")

    embedding = parser.add_argument_group("embedding enhancement")
    embedding.add_argument(
        "--embedding_mode",
        "--embedding-mode",
        default="none",
        choices=EMBEDDING_MODES,
        help="Exactly one of none, pose, posefilm, or uav",
    )
    embedding.add_argument("--uav_adapter_path", default="")
    embedding.add_argument("--uav_adapter_type", default="transformer_v1")
    embedding.add_argument("--uav_adapter_apply_scope", default="all_images")
    embedding.add_argument("--pose_norm_scale", type=float, default=100.0)

    output = parser.add_argument_group("output and execution")
    output.add_argument(
        "--output_dir",
        default=f"./results/eval/{DEFAULT_MODEL_TYPE}",
    )
    output.add_argument("--save_video", action="store_true")
    output.add_argument("--video_compression", action="store_true")
    output.add_argument("--distributed", action="store_true")
    output.add_argument("--max_episodes", type=int)
    output.add_argument("--debug_timing", action="store_true")
    output.add_argument("--verbose", action="store_true")
    return parser


def validate_eval_args(args: argparse.Namespace) -> None:
    """Apply the shared train/name/eval cross-field rules."""
    SwiftVLNExperimentSpec(
        env_type=args.env_type,
        model_family=(
            "qwen3_vl" if args.model_type == "swiftvln_qwen3_vl" else "qwen2_5_vl"
        ),
        num_frames=args.num_frames,
        num_future_steps=args.num_future_steps,
        num_overlap=args.num_overlap,
        memory_method=args.memory_method,
        history_processor_type=args.history_processor_type,
        num_history=args.num_history,
        log_base=args.log_base,
        use_random=args.use_random,
        compress_stride=args.compress_stride,
        use_tome=args.use_tome,
        gtc_output_tokens=args.gtc_output_tokens,
        gtc_temperature=args.gtc_temperature,
        gtc_num_iterations=args.gtc_num_iterations,
        map_global_side_m=args.map_global_side_m,
        map_local_side_m=args.map_local_side_m,
        map_render_px=args.map_render_px,
        map_mask_method=args.map_mask_method,
        system_prompt_setting=args.system_prompt_setting,
        embedding=args.embedding_mode,
    )


def build_summary_extras(args: Any) -> dict[str, Any]:
    """Select configuration fields persisted in ``evaluation_summary.json``."""
    fields = (
        "model_type",
        "num_frames",
        "compress_stride",
        "num_overlap",
        "history_processor_type",
        "system_prompt_setting",
        "memory_method",
        "embedding_mode",
    )
    extras = {field: getattr(args, field) for field in fields}
    if args.memory_method == "map":
        extras.update(
            {
                "map_global_side_m": args.map_global_side_m,
                "map_local_side_m": args.map_local_side_m,
                "map_render_px": args.map_render_px,
                "map_mask_method": args.map_mask_method,
            }
        )
    if args.embedding_mode in {"pose", "posefilm"}:
        extras["pose_norm_scale"] = args.pose_norm_scale
    elif args.embedding_mode == "uav":
        extras.update(
            {
                "uav_adapter_type": args.uav_adapter_type,
                "uav_adapter_apply_scope": args.uav_adapter_apply_scope,
            }
        )
        if args.uav_adapter_path:
            extras["uav_adapter_path"] = args.uav_adapter_path
    if args.history_processor_type in {"gtc", "segment_gtc"}:
        extras.update(
            {
                "gtc_output_tokens": args.gtc_output_tokens,
                "gtc_temperature": args.gtc_temperature,
                "gtc_num_iterations": args.gtc_num_iterations,
            }
        )
    else:
        extras.update(
            {
                "log_base": args.log_base,
                "use_random": args.use_random,
                "use_tome": args.use_tome,
            }
        )
    return extras


def parse_eval_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = create_eval_parser()
    args = parser.parse_args(argv)
    try:
        validate_eval_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse_eval_args()
    SwiftVLNEvaluationRunner(args, build_summary_extras(args)).run()


if __name__ == "__main__":
    main()
