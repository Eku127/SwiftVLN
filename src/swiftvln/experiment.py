"""Validated SwiftVLN experiment configuration and model-name codec.

The train and eval shell entrypoints use this module as their shared source of
truth.  It intentionally has no Torch, ms-swift, Habitat, or SatNav imports so
name parsing and validation stay fast and usable in lightweight tooling.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Dict, Optional, Sequence


MODEL_FAMILIES = ("qwen2_5_vl", "qwen3_vl")
ENV_TYPES = ("habitat", "satnav")
MEMORY_METHODS = ("history", "map")
HISTORY_PROCESSORS = ("per_frame", "gtc", "segment_gtc")
SYSTEM_PROMPTS = ("vanilla", "initial")
EMBEDDING_MODES = ("none", "pose", "posefilm", "uav")


class ExperimentNameError(ValueError):
    """Raised when a model name is malformed or describes an invalid run."""


_NAME_PREFIX_RE = re.compile(
    r"^swiftvln-(?P<env>habitat|satnav)-"
    r"(?:(?P<qwen3>qwen3vl)-)?"
    r"(?P<size>\d+b)-(?P<epochs>\d+)ep-"
    r"f(?P<frames>\d+)s(?P<future>\d+)-"
    r"overlap(?P<overlap>\d+)-(?P<body>.+)$"
)
_RUN_SUFFIX_RE = re.compile(
    r"^(?P<body>.+)-bs(?P<batch>\d+)-"
    r"lr(?P<lr>\d+(?:\.\d+)?(?:e[+-]?\d+)?)-"
    r"(?P<timestamp>(?:\d{8}-)?\d{6})$",
    re.IGNORECASE,
)
_EMBEDDING_RE = re.compile(
    r"^(?P<body>.+)-"
    r"(?P<embedding>noembed|pose|posefilm|uav|pose\+uav|posefilm\+uav)$"
)
_PER_FRAME_RE = re.compile(
    r"^pf-h(?P<history>\d+)"
    r"(?P<nomem>-nomem)?(?P<random>-random)?"
    r"(?:-b(?P<log_base>\d+(?:\.\d+)?))?"
    r"-(?P<method>pool|tome)-s(?P<stride>\d+)$"
)
_GTC_RE = re.compile(
    r"^(?P<kind>s?gtc)-k(?P<tokens>\d+)"
    r"(?:-t(?P<temperature>\d+(?:\.\d+)?))?"
    r"(?:-i(?P<iterations>\d+))?$"
)
_MAP_RE = re.compile(
    r"^map-g(?P<global>\d+(?:\.\d+)?)-"
    r"l(?P<local>\d+(?:\.\d+)?)-"
    r"r(?P<render>\d+)-(?P<mask>[a-zA-Z0-9.]+)-"
    r"s(?P<stride>\d+)$"
)


def _format_number(value: float) -> str:
    return format(value, ".15g")


def _format_log_base(value: float) -> str:
    if float(value).is_integer():
        return f"{value:.1f}"
    return _format_number(value)


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def normalize_embedding_mode(value: str) -> str:
    """Return the canonical mutually-exclusive embedding mode."""
    if not isinstance(value, str):
        raise ExperimentNameError(
            f"embedding mode must be exactly one of {EMBEDDING_MODES}, got {value!r}"
        )
    mode = value.strip().lower()
    if mode not in EMBEDDING_MODES:
        raise ExperimentNameError(
            f"embedding mode must be exactly one of {EMBEDDING_MODES}, got {value!r}"
        )
    return mode


def embedding_uses_pose(value: str) -> bool:
    """Whether a mode requires pose metadata and a pose embedding module."""
    return normalize_embedding_mode(value) in {"pose", "posefilm"}


def embedding_uses_uav(value: str) -> bool:
    """Whether a mode requires the Stage-A UAV adapter."""
    return normalize_embedding_mode(value) == "uav"


def pose_fusion_for_embedding_mode(value: str) -> str:
    """Resolve the implementation-level pose fusion from the public mode."""
    return "film" if normalize_embedding_mode(value) == "posefilm" else "additive"


@dataclass(frozen=True)
class SwiftVLNExperimentSpec:
    """The stable configuration encoded in a SwiftVLN experiment name."""

    env_type: str
    model_family: str = "qwen2_5_vl"
    model_size: str = "3b"
    num_epochs: int = 1
    num_frames: int = 32
    num_future_steps: int = 4
    num_overlap: int = 0

    memory_method: str = "history"
    history_processor_type: str = "per_frame"
    num_history: int = 8
    log_base: float = 1.0
    use_random: bool = False
    compress_stride: int = 2
    use_tome: bool = False
    gtc_output_tokens: int = 512
    gtc_temperature: float = 0.1
    gtc_num_iterations: int = 1

    map_global_side_m: float = 1000.0
    map_local_side_m: float = 400.0
    map_render_px: int = 448
    map_mask_method: str = "dilate20"

    system_prompt_setting: str = "vanilla"
    embedding: str = "none"

    effective_batch_size: Optional[int] = None
    learning_rate: Optional[str] = None
    timestamp: Optional[str] = None

    _source_name: Optional[str] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "embedding", normalize_embedding_mode(self.embedding))
        if self.env_type not in ENV_TYPES:
            raise ExperimentNameError(
                f"env_type must be one of {ENV_TYPES}, got {self.env_type!r}"
            )
        if self.model_family not in MODEL_FAMILIES:
            raise ExperimentNameError(
                f"model_family must be one of {MODEL_FAMILIES}, "
                f"got {self.model_family!r}"
            )
        if not re.fullmatch(r"\d+b", self.model_size):
            raise ExperimentNameError(
                f"model_size must look like '3b', got {self.model_size!r}"
            )
        for field_name in (
            "num_epochs",
            "num_frames",
            "num_future_steps",
            "compress_stride",
        ):
            if getattr(self, field_name) <= 0:
                raise ExperimentNameError(f"{field_name} must be positive")
        if not 0 <= self.num_overlap < self.num_frames:
            raise ExperimentNameError(
                "num_overlap must satisfy 0 <= num_overlap < num_frames"
            )
        if self.num_overlap % self.num_future_steps != 0:
            raise ExperimentNameError(
                "num_overlap must be aligned to num_future_steps"
            )
        if self.memory_method not in MEMORY_METHODS:
            raise ExperimentNameError(
                f"memory_method must be one of {MEMORY_METHODS}"
            )
        if self.history_processor_type not in HISTORY_PROCESSORS:
            raise ExperimentNameError(
                f"history_processor_type must be one of {HISTORY_PROCESSORS}"
            )
        if self.system_prompt_setting not in SYSTEM_PROMPTS:
            raise ExperimentNameError(
                f"system_prompt_setting must be one of {SYSTEM_PROMPTS}"
            )
        if self.memory_method == "map":
            if self.env_type != "satnav":
                raise ExperimentNameError("map memory is supported only for SatNav")
            if self.history_processor_type != "per_frame":
                raise ExperimentNameError(
                    "map memory requires history_processor_type='per_frame'"
                )
            if self.use_tome:
                raise ExperimentNameError("map memory requires pooling, not ToMe")
            if self.use_random:
                raise ExperimentNameError(
                    "map memory does not use random history sampling"
                )
            if self.embedding != "none":
                raise ExperimentNameError(
                    "map memory cannot use pose or UAV embedding enhancement"
                )
            if min(
                self.map_global_side_m,
                self.map_local_side_m,
                self.map_render_px,
            ) <= 0:
                raise ExperimentNameError("map dimensions must be positive")
            if self.map_local_side_m > self.map_global_side_m:
                raise ExperimentNameError(
                    "map_local_side_m cannot exceed map_global_side_m"
                )
            if not (
                self.map_mask_method == "strict"
                or re.fullmatch(r"dilate\d+(?:\.\d+)?", self.map_mask_method)
            ):
                raise ExperimentNameError(
                    "map_mask_method must be 'strict' or 'dilate<N>'"
                )
        elif self.history_processor_type == "per_frame":
            if self.num_history < 0:
                raise ExperimentNameError("num_history cannot be negative")
            if self.log_base < 1.0:
                raise ExperimentNameError("log_base must be >= 1.0")
            if self.num_history == 0 and self.use_random:
                raise ExperimentNameError(
                    "random sampling is invalid when num_history=0"
                )
        else:
            if self.use_random or self.use_tome:
                raise ExperimentNameError(
                    "GTC processors do not use per-frame random/ToMe options"
                )
            if self.gtc_output_tokens <= 0:
                raise ExperimentNameError("gtc_output_tokens must be positive")
            if self.gtc_temperature <= 0:
                raise ExperimentNameError("gtc_temperature must be positive")
            if self.gtc_num_iterations <= 0:
                raise ExperimentNameError("gtc_num_iterations must be positive")

        metadata = (
            self.effective_batch_size,
            self.learning_rate,
            self.timestamp,
        )
        if any(value is not None for value in metadata) and not all(
            value is not None for value in metadata
        ):
            raise ExperimentNameError(
                "run names require effective_batch_size, learning_rate, "
                "and timestamp together"
            )
        if self.effective_batch_size is not None and self.effective_batch_size <= 0:
            raise ExperimentNameError("effective_batch_size must be positive")
        if self.learning_rate is not None:
            try:
                learning_rate = Decimal(self.learning_rate)
            except InvalidOperation as exc:
                raise ExperimentNameError(
                    f"invalid learning_rate: {self.learning_rate!r}"
                ) from exc
            if learning_rate <= 0:
                raise ExperimentNameError("learning_rate must be positive")
        if self.timestamp is not None and not re.fullmatch(
            r"(?:\d{8}-)?\d{6}", self.timestamp
        ):
            raise ExperimentNameError(
                "timestamp must be HHMMSS or YYYYMMDD-HHMMSS"
            )

    def _memory_name(self, *, include_default_log_base: bool) -> str:
        if self.memory_method == "map":
            mask = self.map_mask_method
            match = re.fullmatch(r"dilate(\d+(?:\.\d+)?)", mask)
            if match:
                mask = f"d{_format_number(float(match.group(1)))}"
            return (
                f"map-g{_format_number(self.map_global_side_m)}"
                f"-l{_format_number(self.map_local_side_m)}"
                f"-r{self.map_render_px}-{mask}-s{self.compress_stride}"
            )
        if self.history_processor_type == "gtc":
            memory = f"gtc-k{self.gtc_output_tokens}"
        elif self.history_processor_type == "segment_gtc":
            memory = f"sgtc-k{self.gtc_output_tokens}"
        else:
            memory = ""
        if memory:
            if self.gtc_temperature != 0.1:
                memory += f"-t{_format_number(self.gtc_temperature)}"
            if self.gtc_num_iterations != 1:
                memory += f"-i{self.gtc_num_iterations}"
            return memory

        memory = f"pf-h{self.num_history}"
        if self.num_history == 0:
            memory += "-nomem"
        elif self.use_random:
            memory += "-random"
        if include_default_log_base or self.log_base != 1.0:
            memory += f"-b{_format_log_base(self.log_base)}"
        method = "tome" if self.use_tome else "pool"
        return f"{memory}-{method}-s{self.compress_stride}"

    def to_model_name(self, style: str = "short") -> str:
        """Serialize to a canonical short/run name or return the source name."""
        if style == "source":
            if self._source_name is None:
                raise ExperimentNameError("this spec was not parsed from a name")
            return self._source_name
        if style not in {"short", "run"}:
            raise ExperimentNameError("style must be 'short', 'run', or 'source'")
        if style == "run" and self.effective_batch_size is None:
            raise ExperimentNameError("run style requires batch/lr/timestamp metadata")

        family = "qwen3vl-" if self.model_family == "qwen3_vl" else ""
        name = (
            f"swiftvln-{self.env_type}-{family}{self.model_size}-"
            f"{self.num_epochs}ep-f{self.num_frames}s{self.num_future_steps}-"
            f"overlap{self.num_overlap}-"
            f"{self._memory_name(include_default_log_base=style == 'run')}"
        )
        if self.system_prompt_setting != "vanilla":
            name += f"-{self.system_prompt_setting}"
        embedding_tag = "noembed" if self.embedding == "none" else self.embedding
        name += f"-{embedding_tag}"
        if style == "run":
            name += (
                f"-bs{self.effective_batch_size}-lr{self.learning_rate}-"
                f"{self.timestamp}"
            )
        return name

    def to_dict(self) -> Dict[str, object]:
        values = asdict(self)
        values.pop("_source_name", None)
        return values

    def to_shell_assignments(self) -> Dict[str, str]:
        """Return the variables consumed by eval_by_name.sh."""
        return {
            "MODEL_ARCH": "swiftvln",
            "PARSED_ENV_TYPE": self.env_type,
            "MODEL_FAMILY": self.model_family,
            "MODEL_SIZE": self.model_size,
            "NUM_EPOCHS": str(self.num_epochs),
            "NUM_FRAMES": str(self.num_frames),
            "NUM_HISTORY": str(self.num_history),
            "NUM_FUTURE_STEPS": str(self.num_future_steps),
            "NUM_OVERLAP": str(self.num_overlap),
            "MEMORY_METHOD": self.memory_method,
            "HISTORY_PROCESSOR_TYPE": self.history_processor_type,
            "LOG_BASE": _format_log_base(self.log_base),
            "USE_RANDOM": str(self.use_random).lower(),
            "COMPRESS_STRIDE": str(self.compress_stride),
            "USE_TOME": str(self.use_tome).lower(),
            "GTC_OUTPUT_TOKENS": (
                str(self.gtc_output_tokens)
                if self.history_processor_type in {"gtc", "segment_gtc"}
                else ""
            ),
            "GTC_TEMPERATURE": (
                _format_number(self.gtc_temperature)
                if self.history_processor_type in {"gtc", "segment_gtc"}
                else ""
            ),
            "GTC_NUM_ITERATIONS": (
                str(self.gtc_num_iterations)
                if self.history_processor_type in {"gtc", "segment_gtc"}
                else ""
            ),
            "MAP_GLOBAL_SIDE_M": (
                _format_number(self.map_global_side_m)
                if self.memory_method == "map"
                else ""
            ),
            "MAP_LOCAL_SIDE_M": (
                _format_number(self.map_local_side_m)
                if self.memory_method == "map"
                else ""
            ),
            "MAP_RENDER_PX": (
                str(self.map_render_px) if self.memory_method == "map" else ""
            ),
            "MAP_MASK_METHOD": (
                self.map_mask_method if self.memory_method == "map" else ""
            ),
            "SYSTEM_PROMPT_SETTING": self.system_prompt_setting,
            "EMBEDDING_MODE": self.embedding,
            "BATCH_SIZE": (
                str(self.effective_batch_size)
                if self.effective_batch_size is not None
                else ""
            ),
            "LEARNING_RATE": self.learning_rate or "",
        }


def parse_model_name(model_name: str) -> SwiftVLNExperimentSpec:
    """Parse current short names and historical run names."""
    prefix = _NAME_PREFIX_RE.fullmatch(model_name)
    if prefix is None:
        raise ExperimentNameError(
            "model name must match swiftvln-{habitat|satnav}-..."
        )

    body = prefix.group("body")
    effective_batch_size: Optional[int] = None
    learning_rate: Optional[str] = None
    timestamp: Optional[str] = None
    run_suffix = _RUN_SUFFIX_RE.fullmatch(body)
    if run_suffix:
        body = run_suffix.group("body")
        effective_batch_size = int(run_suffix.group("batch"))
        learning_rate = run_suffix.group("lr")
        timestamp = run_suffix.group("timestamp")

    embedding = "none"
    embedding_match = _EMBEDDING_RE.fullmatch(body)
    if embedding_match:
        body = embedding_match.group("body")
        embedding_tag = embedding_match.group("embedding")
        if "+" in embedding_tag:
            raise ExperimentNameError(
                "combined pose+UAV experiment names are no longer supported"
            )
        embedding = "none" if embedding_tag == "noembed" else embedding_tag

    system_prompt_setting = "vanilla"
    if body.endswith("-initial"):
        body = body[: -len("-initial")]
        system_prompt_setting = "initial"

    common = {
        "env_type": prefix.group("env"),
        "model_family": "qwen3_vl" if prefix.group("qwen3") else "qwen2_5_vl",
        "model_size": prefix.group("size"),
        "num_epochs": int(prefix.group("epochs")),
        "num_frames": int(prefix.group("frames")),
        "num_future_steps": int(prefix.group("future")),
        "num_overlap": int(prefix.group("overlap")),
        "system_prompt_setting": system_prompt_setting,
        "embedding": embedding,
        "effective_batch_size": effective_batch_size,
        "learning_rate": learning_rate,
        "timestamp": timestamp,
        "_source_name": model_name,
    }

    map_match = _MAP_RE.fullmatch(body)
    if map_match:
        mask = map_match.group("mask")
        short_mask = re.fullmatch(r"d(\d+(?:\.\d+)?)", mask)
        if short_mask:
            mask = f"dilate{short_mask.group(1)}"
        return SwiftVLNExperimentSpec(
            **common,
            memory_method="map",
            compress_stride=int(map_match.group("stride")),
            map_global_side_m=float(map_match.group("global")),
            map_local_side_m=float(map_match.group("local")),
            map_render_px=int(map_match.group("render")),
            map_mask_method=mask,
        )

    gtc_match = _GTC_RE.fullmatch(body)
    if gtc_match:
        processor = (
            "segment_gtc" if gtc_match.group("kind") == "sgtc" else "gtc"
        )
        return SwiftVLNExperimentSpec(
            **common,
            history_processor_type=processor,
            gtc_output_tokens=int(gtc_match.group("tokens")),
            gtc_temperature=float(gtc_match.group("temperature") or 0.1),
            gtc_num_iterations=int(gtc_match.group("iterations") or 1),
        )

    per_frame = _PER_FRAME_RE.fullmatch(body)
    if per_frame:
        num_history = int(per_frame.group("history"))
        if bool(per_frame.group("nomem")) != (num_history == 0):
            raise ExperimentNameError(
                "the nomem tag must be present exactly when num_history=0"
            )
        return SwiftVLNExperimentSpec(
            **common,
            num_history=num_history,
            log_base=float(per_frame.group("log_base") or 1.0),
            use_random=bool(per_frame.group("random")),
            compress_stride=int(per_frame.group("stride")),
            use_tome=per_frame.group("method") == "tome",
        )

    raise ExperimentNameError(f"unsupported memory block in model name: {body!r}")


def _build_spec(args: argparse.Namespace) -> SwiftVLNExperimentSpec:
    return SwiftVLNExperimentSpec(
        env_type=args.env_type,
        model_family=args.model_family,
        model_size=args.model_size.lower(),
        num_epochs=args.num_epochs,
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
        effective_batch_size=args.effective_batch_size,
        learning_rate=args.learning_rate,
        timestamp=args.timestamp,
    )


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser("parse-name")
    parse_parser.add_argument("model_name")
    parse_parser.add_argument(
        "--format", choices=("shell", "json", "value"), default="json"
    )
    parse_parser.add_argument("--field")

    build_parser = subparsers.add_parser("build-name")
    build_parser.add_argument("--env-type", required=True, choices=ENV_TYPES)
    build_parser.add_argument("--model-family", required=True, choices=MODEL_FAMILIES)
    build_parser.add_argument("--model-size", required=True)
    build_parser.add_argument("--num-epochs", required=True, type=int)
    build_parser.add_argument("--num-frames", required=True, type=int)
    build_parser.add_argument("--num-future-steps", required=True, type=int)
    build_parser.add_argument("--num-overlap", required=True, type=int)
    build_parser.add_argument("--memory-method", required=True, choices=MEMORY_METHODS)
    build_parser.add_argument(
        "--history-processor-type", required=True, choices=HISTORY_PROCESSORS
    )
    build_parser.add_argument("--num-history", required=True, type=int)
    build_parser.add_argument("--log-base", required=True, type=float)
    build_parser.add_argument("--use-random", required=True, type=_parse_bool)
    build_parser.add_argument("--compress-stride", required=True, type=int)
    build_parser.add_argument("--use-tome", required=True, type=_parse_bool)
    build_parser.add_argument("--gtc-output-tokens", required=True, type=int)
    build_parser.add_argument("--gtc-temperature", required=True, type=float)
    build_parser.add_argument("--gtc-num-iterations", required=True, type=int)
    build_parser.add_argument("--map-global-side-m", required=True, type=float)
    build_parser.add_argument("--map-local-side-m", required=True, type=float)
    build_parser.add_argument("--map-render-px", required=True, type=int)
    build_parser.add_argument("--map-mask-method", required=True)
    build_parser.add_argument(
        "--system-prompt-setting", required=True, choices=SYSTEM_PROMPTS
    )
    build_parser.add_argument(
        "--embedding-mode", required=True, choices=EMBEDDING_MODES
    )
    build_parser.add_argument("--effective-batch-size", required=True, type=int)
    build_parser.add_argument("--learning-rate", required=True)
    build_parser.add_argument("--timestamp", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _create_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "parse-name":
            spec = parse_model_name(args.model_name)
            if args.format == "json":
                print(json.dumps(spec.to_dict(), ensure_ascii=False, sort_keys=True))
            elif args.format == "shell":
                for key, value in spec.to_shell_assignments().items():
                    print(f"{key}={shlex.quote(value)}")
            else:
                values = spec.to_dict()
                if not args.field or args.field not in values:
                    parser.error(
                        "--format value requires --field from: "
                        + ", ".join(sorted(values))
                    )
                value = values[args.field]
                if isinstance(value, bool):
                    value = str(value).lower()
                elif value is None:
                    value = ""
                print(value)
            return 0

        spec = _build_spec(args)
        print(spec.to_model_name(style="run"))
        return 0
    except ExperimentNameError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
