"""Offline action metrics and a self-contained report for primitive episodes.

The regular SwiftVLN SFT loop records token loss, but it does not know whether
the generated navigation symbols match the action-group trajectory.  This
module evaluates a checkpoint on the same frame-by-frame, autoregressive
protocol used at inference time and records four task-level metrics:

* micro action accuracy across all actions, including STOP;
* per-class and combined movement accuracy for FORWARD / LEFT / RIGHT;
* STOP accuracy;
* exact match rate for the complete action sequence of an action group.

The evaluator intentionally consumes the ordinary SwiftVLN ``annotations.json
and images/<sample>/rgb`` layout.  It therefore works for the primitive dataset
builder without introducing a second data format.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


ACTION_NAMES = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
}
MOVEMENT_ACTIONS = (1, 2, 3)
METRIC_FIELDS = (
    "action_accuracy",
    "movement_accuracy",
    "move_forward_accuracy",
    "turn_left_accuracy",
    "turn_right_accuracy",
    "stop_accuracy",
    "exact_match_rate",
)


@dataclass(frozen=True)
class PrimitiveMetricConfig:
    """Inference settings which must match the primitive training recipe."""

    model_path: Path
    dataset_path: Path
    output_dir: Path
    model_type: str
    num_frames: int
    num_history: int
    num_future_steps: int
    num_overlap: int
    history_processor_type: str
    compress_stride: int
    log_base: float
    use_random: bool
    use_tome: bool
    system_prompt_setting: str
    embedding_mode: str
    max_samples: int | None
    max_new_tokens: int
    seed: int
    attn_impl: str


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return result


def _load_annotations(dataset_path: Path) -> list[dict[str, Any]]:
    annotations_path = dataset_path / "annotations.json"
    if not annotations_path.is_file():
        raise FileNotFoundError(f"annotations.json does not exist: {annotations_path}")
    try:
        records = json.loads(annotations_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid annotations JSON: {annotations_path}") from error
    if not isinstance(records, list):
        raise ValueError(f"annotations must be an array: {annotations_path}")
    if not records:
        raise ValueError(f"annotations are empty: {annotations_path}")
    return records


def _normalise_annotation(record: Mapping[str, Any], position: int, dataset_path: Path) -> dict[str, Any]:
    """Validate one regular SwiftVLN action-group episode."""

    actions = record.get("actions")
    instructions = record.get("instructions")
    video = record.get("video")
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"annotation {position}: actions must be a non-empty array")
    if not all(isinstance(action, int) and action in {-1, 0, 1, 2, 3} for action in actions):
        raise ValueError(f"annotation {position}: unsupported action ID")
    if actions[0] != -1:
        raise ValueError(f"annotation {position}: first action must be INIT (-1)")
    if not isinstance(instructions, list) or not instructions or not isinstance(instructions[0], str):
        raise ValueError(f"annotation {position}: instructions must contain text")
    if not isinstance(video, str) or not video:
        raise ValueError(f"annotation {position}: video must be a relative directory")

    video_path = (dataset_path / video).resolve()
    try:
        video_path.relative_to(dataset_path.resolve())
    except ValueError as error:
        raise ValueError(f"annotation {position}: video escapes dataset root") from error
    rgb_path = video_path / "rgb"
    if not rgb_path.is_dir():
        raise FileNotFoundError(f"annotation {position}: RGB directory missing: {rgb_path}")
    frames = sorted(path for path in rgb_path.iterdir() if path.is_file())
    if len(frames) != len(actions):
        raise ValueError(
            f"annotation {position}: frame/action mismatch "
            f"({len(frames)} frames, {len(actions)} actions)"
        )
    return {
        "id": record.get("id", position),
        "trajectory_id": str(record.get("trajectory_id", position)),
        "instruction": instructions[0],
        "raw_actions": list(actions),
        "frame_paths": frames,
    }


def _gold_actions(raw_actions: Sequence[int]) -> list[int]:
    """Mirror the target shift in :class:`SwiftVLNDataset` exactly."""

    return [int(action) for action in raw_actions[1:]] + [0]


def _load_pil_images(paths: Sequence[Path]) -> list[Any]:
    from PIL import Image

    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def _new_timing_stats() -> dict[str, float]:
    return {
        "build_embeds": 0.0,
        "model_generate": 0.0,
        "decode": 0.0,
        "parse_actions": 0.0,
        "update_cache": 0.0,
        "window_slide": 0.0,
    }


def _session_args(config: PrimitiveMetricConfig) -> SimpleNamespace:
    """Give the shared inference session only the fields it consumes."""

    return SimpleNamespace(
        num_frames=config.num_frames,
        num_overlap=config.num_overlap,
        history_processor_type=config.history_processor_type,
        compress_stride=config.compress_stride,
        use_tome=config.use_tome,
        log_base=config.log_base,
        use_random=config.use_random,
        gtc_output_tokens=512,
        gtc_temperature=0.1,
        gtc_num_iterations=1,
        system_prompt_setting=config.system_prompt_setting,
        memory_method="history",
        map_global_side_m=1000.0,
        map_local_side_m=400.0,
        map_render_px=448,
        map_mask_method="dilate20",
        embedding_mode=config.embedding_mode,
        verbose=False,
        debug_timing=False,
        max_new_tokens=config.max_new_tokens,
    )


def _load_inference_session(config: PrimitiveMetricConfig):
    """Load a checkpoint through the same registered wrapper used by eval."""

    import torch

    from swift.model import get_model_processor
    from swiftvln.backends.specs import get_environment_spec
    from swiftvln.evaluation.diagnostics import DiagnosticsObserver
    from swiftvln.evaluation.inference import SwiftVLNInferenceSession
    from swiftvln.modeling import register_swiftvln_models

    register_swiftvln_models()
    model, processor = get_model_processor(
        model_id_or_path=str(config.model_path),
        model_type=config.model_type,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_impl=config.attn_impl,
        embedding_mode=config.embedding_mode,
    )
    if model is None:
        raise RuntimeError("model loader returned no model")
    model.eval()
    if not torch.cuda.is_available():
        raise RuntimeError("primitive metrics require a CUDA-enabled SwiftVLN environment")
    # Match the ordinary evaluation runner: visual inputs and prompt IDs enter
    # through the first visible CUDA device, even when Accelerate dispatches
    # language-model blocks across several devices.
    device = torch.device("cuda")
    args = _session_args(config)
    # Map memory is disabled, so the lightweight config placeholder is never
    # dereferenced by the session.  It preserves the public constructor shape.
    runtime_config = SimpleNamespace(DATASET=SimpleNamespace(DATA_PATH="", SCENES_DIR=""))
    diagnostics = DiagnosticsObserver(args, str(config.output_dir))
    session = SwiftVLNInferenceSession(
        model=model,
        processor=processor,
        args=args,
        config=runtime_config,
        environment_spec=get_environment_spec("satnav"),
        device=device,
        num_history=config.num_history,
        num_future_steps=config.num_future_steps,
        diagnostics=diagnostics,
    )
    return model, session


def _parse_actions(output: str) -> list[int]:
    """Parse exactly the navigation symbols accepted by the SatNav backend."""

    import re

    return [
        {"STOP": 0, "↑": 1, "←": 2, "→": 3}[symbol]
        for symbol in re.findall(r"STOP|↑|←|→", output)
    ]


def _evaluate_episode(session: Any, item: Mapping[str, Any]) -> tuple[list[int], list[dict[str, Any]], dict[str, float]]:
    """Autoregress over fixed expert frames and retain raw model responses."""

    import torch

    images = _load_pil_images(item["frame_paths"])
    gold_actions = _gold_actions(item["raw_actions"])
    if len(images) != len(gold_actions):
        raise RuntimeError("validated annotation lost action/frame alignment")

    predicted: list[int] = []
    turns: list[dict[str, Any]] = []
    timing_stats = _new_timing_stats()
    session.reset()
    session.start_first_window(images, item)
    with torch.inference_mode():
        for step_id, image in enumerate(images):
            while step_id >= session.window_start_step + session.num_frames:
                started_at = time.monotonic()
                session.slide_window(
                    images,
                    session.pose_history,
                    session.window_start_step + session.stride,
                    item,
                )
                timing_stats["window_slide"] += time.monotonic() - started_at
            current_pose = session.observe_pose()
            actions = session.predict(
                instruction=item["instruction"],
                current_image=image,
                current_pose=current_pose,
                step_id=step_id,
                parse_actions=_parse_actions,
                timing_stats=timing_stats,
            )
            prediction = int(actions[0]) if actions else 0
            predicted.append(prediction)
            session.record_action(prediction)
            turns.append(
                {
                    "step": step_id,
                    "gold_action": ACTION_NAMES[gold_actions[step_id]],
                    "predicted_action": ACTION_NAMES[prediction],
                    "extra_predicted_actions": [ACTION_NAMES[action] for action in actions[1:]],
                }
            )
    return predicted, turns, timing_stats


def _empty_summary() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "exact_match_count": 0,
        "total_action_count": 0,
        "correct_action_count": 0,
        "movement_action_count": 0,
        "correct_movement_action_count": 0,
        "class_totals": Counter(),
        "class_correct": Counter(),
        "confusion": Counter(),
    }


def _accumulate(summary: dict[str, Any], gold: Sequence[int], predicted: Sequence[int]) -> bool:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted action sequences must have equal length")
    exact = list(gold) == list(predicted)
    summary["sample_count"] += 1
    summary["exact_match_count"] += int(exact)
    for expected, actual in zip(gold, predicted):
        if expected not in ACTION_NAMES or actual not in ACTION_NAMES:
            raise ValueError("unsupported action ID while accumulating primitive metrics")
        summary["total_action_count"] += 1
        summary["correct_action_count"] += int(expected == actual)
        summary["class_totals"][expected] += 1
        summary["class_correct"][expected] += int(expected == actual)
        summary["confusion"][(expected, actual)] += 1
        if expected in MOVEMENT_ACTIONS:
            summary["movement_action_count"] += 1
            summary["correct_movement_action_count"] += int(expected == actual)
    return exact


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _finish_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    class_totals: Counter[int] = summary["class_totals"]
    class_correct: Counter[int] = summary["class_correct"]
    confusion: Counter[tuple[int, int]] = summary["confusion"]
    per_action = {
        ACTION_NAMES[action]: {
            "correct": class_correct[action],
            "total": class_totals[action],
            "accuracy": _ratio(class_correct[action], class_totals[action]),
        }
        for action in sorted(ACTION_NAMES)
    }
    confusion_rows = {
        ACTION_NAMES[gold]: {
            ACTION_NAMES[predicted]: confusion[(gold, predicted)]
            for predicted in sorted(ACTION_NAMES)
        }
        for gold in sorted(ACTION_NAMES)
    }
    action_accuracy = _ratio(summary["correct_action_count"], summary["total_action_count"])
    movement_accuracy = _ratio(
        summary["correct_movement_action_count"], summary["movement_action_count"]
    )
    result = {
        "sample_count": summary["sample_count"],
        "exact_match_count": summary["exact_match_count"],
        "exact_match_rate": _ratio(summary["exact_match_count"], summary["sample_count"]),
        "total_action_count": summary["total_action_count"],
        "correct_action_count": summary["correct_action_count"],
        "action_accuracy": action_accuracy,
        "movement_action_count": summary["movement_action_count"],
        "correct_movement_action_count": summary["correct_movement_action_count"],
        "movement_accuracy": movement_accuracy,
        "per_action": per_action,
        "move_forward_accuracy": per_action["MOVE_FORWARD"]["accuracy"],
        "turn_left_accuracy": per_action["TURN_LEFT"]["accuracy"],
        "turn_right_accuracy": per_action["TURN_RIGHT"]["accuracy"],
        "stop_accuracy": per_action["STOP"]["accuracy"],
        "confusion_matrix": confusion_rows,
    }
    return result


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(row, handle, ensure_ascii=False)
        handle.write("\n")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _safe_float(value: Any) -> str:
    return "—" if value is None else f"{float(value) * 100:.2f}%"


def _svg_line_chart(rows: Sequence[Mapping[str, Any]], field: str, colour: str) -> str:
    values = [row.get("metrics", {}).get(field) for row in rows]
    points = [(index, value) for index, value in enumerate(values) if isinstance(value, (int, float))]
    width, height, padding = 640, 190, 34
    if not points:
        return "<p>暂无可绘制数据。</p>"
    max_index = max(1, len(rows) - 1)
    coordinates = []
    for index, value in points:
        x = padding + (width - 2 * padding) * index / max_index
        y = height - padding - (height - 2 * padding) * float(value)
        coordinates.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(coordinates)
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(field)} curve">'
        f'<rect width="{width}" height="{height}" fill="#fff" stroke="#d8dee9"/>'
        f'<line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" stroke="#8f9bb3"/>'
        f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height-padding}" stroke="#8f9bb3"/>'
        f'<polyline fill="none" stroke="{colour}" stroke-width="3" points="{polyline}"/>'
        '<text x="6" y="16" font-size="12">100%</text>'
        f'<text x="9" y="{height-padding+4}" font-size="12">0%</text>'
        f'<text x="{width-110}" y="{height-8}" font-size="12">checkpoint order</text>'
        '</svg>'
    )


def _render_html(rows: Sequence[Mapping[str, Any]]) -> str:
    latest = rows[-1] if rows else {}
    metrics = latest.get("metrics", {}) if isinstance(latest, Mapping) else {}
    cards = "".join(
        f'<section class="card"><h3>{html.escape(field.replace("_", " "))}</h3><strong>{_safe_float(metrics.get(field))}</strong></section>'
        for field in METRIC_FIELDS
    )
    per_action = metrics.get("per_action", {}) if isinstance(metrics, Mapping) else {}
    action_rows = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{details.get('correct', 0)}</td><td>{details.get('total', 0)}</td>"
        f"<td>{_safe_float(details.get('accuracy'))}</td>"
        "</tr>"
        for name, details in per_action.items()
    )
    confusion = metrics.get("confusion_matrix", {}) if isinstance(metrics, Mapping) else {}
    names = list(ACTION_NAMES.values())
    matrix_header = "".join(f"<th>预测 {html.escape(name)}</th>" for name in names)
    matrix_rows = "".join(
        f"<tr><th>真实 {html.escape(gold)}</th>"
        + "".join(f"<td>{confusion.get(gold, {}).get(predicted, 0)}</td>" for predicted in names)
        + "</tr>"
        for gold in names
    )
    history_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('checkpoint_name', 'unknown')))}</td>"
        f"<td>{_safe_float(row.get('metrics', {}).get('action_accuracy'))}</td>"
        f"<td>{_safe_float(row.get('metrics', {}).get('movement_accuracy'))}</td>"
        f"<td>{_safe_float(row.get('metrics', {}).get('stop_accuracy'))}</td>"
        f"<td>{_safe_float(row.get('metrics', {}).get('exact_match_rate'))}</td>"
        "</tr>"
        for row in rows
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>SwiftVLN Primitive Metrics</title>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;margin:28px;background:#f5f7fb;color:#172033}}
h1,h2{{margin:0 0 12px}} .muted{{color:#60708a}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:18px 0 28px}}
.chart-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px;margin:8px 0 28px}} .chart-grid section{{min-width:0}}
.card,table,svg{{background:#fff;border-radius:10px;box-shadow:0 1px 3px #1d2a4420}} .card{{padding:16px}} .card h3{{font-size:14px;text-transform:capitalize;color:#60708a;margin:0 0 8px}} .card strong{{font-size:28px}}
table{{border-collapse:collapse;width:100%;margin:10px 0 28px;overflow:hidden}} th,td{{padding:9px 11px;text-align:left;border-bottom:1px solid #e8ecf3}} th{{background:#edf2fb}} svg{{width:100%;height:auto;margin:8px 0 22px}}
</style></head><body>
<h1>Primitive executor metrics</h1><p class="muted">最近 checkpoint：{html.escape(str(latest.get('checkpoint_name', '—')))}</p>
<div class="grid">{cards}</div>
<h2>Action accuracy</h2>{_svg_line_chart(rows, 'action_accuracy', '#2463eb')}
<h2>Exact-match rate</h2>{_svg_line_chart(rows, 'exact_match_rate', '#0e9f6e')}
<h2>Class accuracy trends</h2><div class="chart-grid">
<section><h3>Movement actions</h3>{_svg_line_chart(rows, 'movement_accuracy', '#7c3aed')}</section>
<section><h3>MOVE_FORWARD</h3>{_svg_line_chart(rows, 'move_forward_accuracy', '#0f766e')}</section>
<section><h3>TURN_LEFT</h3>{_svg_line_chart(rows, 'turn_left_accuracy', '#c2410c')}</section>
<section><h3>TURN_RIGHT</h3>{_svg_line_chart(rows, 'turn_right_accuracy', '#be123c')}</section>
<section><h3>STOP</h3>{_svg_line_chart(rows, 'stop_accuracy', '#1d4ed8')}</section>
</div>
<h2>Per-action performance</h2><table><thead><tr><th>动作</th><th>正确</th><th>总数</th><th>准确率</th></tr></thead><tbody>{action_rows}</tbody></table>
<h2>Confusion matrix</h2><table><thead><tr><th>真实 / 预测</th>{matrix_header}</tr></thead><tbody>{matrix_rows}</tbody></table>
<h2>Checkpoint history</h2><table><thead><tr><th>Checkpoint</th><th>全部动作</th><th>运动动作</th><th>STOP</th><th>整段 exact match</th></tr></thead><tbody>{history_rows}</tbody></table>
</body></html>"""


def _read_metric_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid metrics JSONL") from error
            if not isinstance(row, dict) or "metrics" not in row:
                raise ValueError(f"{path}:{line_number}: invalid metric record")
            rows.append(row)
    return rows


def write_visual_report(metrics_dir: Path) -> Path:
    """Render all metric records to a dependency-free HTML report."""

    rows = _read_metric_rows(metrics_dir / "metrics.jsonl")
    report_path = metrics_dir / "report.html"
    report_path.write_text(_render_html(rows), encoding="utf-8")
    return report_path


def evaluate(config: PrimitiveMetricConfig) -> dict[str, Any]:
    """Run primitive action evaluation and append one durable metric row."""

    if config.num_future_steps != 1:
        raise ValueError(
            "primitive metrics require --num-future-steps=1 so one predicted "
            "action is aligned to each source frame"
        )
    if config.num_overlap != 0:
        raise ValueError("primitive metrics require --num-overlap=0")
    if config.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if not config.model_path.is_dir():
        raise FileNotFoundError(f"model path does not exist: {config.model_path}")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(config.seed)
    annotations = [
        _normalise_annotation(row, position, config.dataset_path)
        for position, row in enumerate(_load_annotations(config.dataset_path))
    ]
    if config.max_samples is not None:
        annotations = annotations[: config.max_samples]
    if not annotations:
        raise ValueError("no annotations selected for primitive metrics")

    _, session = _load_inference_session(config)
    summary_state = _empty_summary()
    checkpoint_name = config.model_path.name
    prediction_path = config.output_dir / f"predictions_{checkpoint_name}.jsonl"
    if prediction_path.exists():
        raise FileExistsError(f"refusing to overwrite predictions: {prediction_path}")
    started_at = time.time()
    with prediction_path.open("x", encoding="utf-8") as handle:
        for position, item in enumerate(annotations, start=1):
            predicted, turns, timing = _evaluate_episode(session, item)
            gold = _gold_actions(item["raw_actions"])
            exact_match = _accumulate(summary_state, gold, predicted)
            row = {
                "sample_id": item["id"],
                "trajectory_id": item["trajectory_id"],
                "instruction": item["instruction"],
                "gold_actions": [ACTION_NAMES[action] for action in gold],
                "predicted_actions": [ACTION_NAMES[action] for action in predicted],
                "exact_match": exact_match,
                "turns": turns,
                "timing_seconds": timing,
            }
            json.dump(row, handle, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            print(
                f"[primitive metrics] {position}/{len(annotations)} "
                f"exact={int(exact_match)} actions={len(gold)}",
                flush=True,
            )

    metrics = _finish_summary(summary_state)
    config_payload = asdict(config)
    config_payload.update(
        {
            "model_path": str(config.model_path),
            "dataset_path": str(config.dataset_path),
            "output_dir": str(config.output_dir),
        }
    )
    record = {
        "record_type": "primitive_action_metrics",
        "created_at_unix": time.time(),
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": str(config.model_path.resolve()),
        "dataset_path": str(config.dataset_path.resolve()),
        "prediction_path": str(prediction_path.resolve()),
        "config": config_payload,
        "elapsed_seconds": time.time() - started_at,
        "metrics": metrics,
    }
    _append_jsonl(config.output_dir / "metrics.jsonl", record)
    _write_json(config.output_dir / "latest.json", record)
    report_path = write_visual_report(config.output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"Metrics ledger: {config.output_dir / 'metrics.jsonl'}", flush=True)
    print(f"Visual report: {report_path}", flush=True)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one SwiftVLN checkpoint on primitive action-group episodes."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate", help="generate predictions and append metrics")
    evaluate_parser.add_argument("--model-path", type=Path, required=True)
    evaluate_parser.add_argument("--dataset-path", type=Path, required=True)
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument(
        "--model-type",
        choices=("swiftvln_qwen2_5_vl", "swiftvln_qwen3_vl"),
        required=True,
    )
    evaluate_parser.add_argument("--num-frames", type=_positive_int, default=8)
    evaluate_parser.add_argument("--num-history", type=_nonnegative_int, default=0)
    evaluate_parser.add_argument("--num-future-steps", type=_positive_int, default=1)
    evaluate_parser.add_argument("--num-overlap", type=_nonnegative_int, default=0)
    evaluate_parser.add_argument(
        "--history-processor-type",
        choices=("per_frame", "gtc", "segment_gtc"),
        default="per_frame",
    )
    evaluate_parser.add_argument("--compress-stride", type=_positive_int, default=2)
    evaluate_parser.add_argument("--log-base", type=float, default=1.0)
    evaluate_parser.add_argument("--use-random", action="store_true")
    evaluate_parser.add_argument("--use-tome", action="store_true")
    evaluate_parser.add_argument(
        "--system-prompt-setting", choices=("vanilla", "initial"), default="vanilla"
    )
    evaluate_parser.add_argument(
        "--embedding-mode", choices=("none", "pose", "posefilm", "uav"), default="none"
    )
    evaluate_parser.add_argument("--max-samples", type=_positive_int)
    evaluate_parser.add_argument("--max-new-tokens", type=_positive_int, default=16)
    evaluate_parser.add_argument("--seed", type=int, default=42)
    evaluate_parser.add_argument(
        "--attn-impl", choices=("flash_attn", "sdpa", "eager"), default="flash_attn"
    )
    visualise_parser = subparsers.add_parser("visualize", help="render metrics.jsonl to report.html")
    visualise_parser.add_argument("--metrics-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "visualize":
            path = write_visual_report(args.metrics_dir)
            print(f"Visual report: {path}", flush=True)
            return 0
        config = PrimitiveMetricConfig(
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            model_type=args.model_type,
            num_frames=args.num_frames,
            num_history=args.num_history,
            num_future_steps=args.num_future_steps,
            num_overlap=args.num_overlap,
            history_processor_type=args.history_processor_type,
            compress_stride=args.compress_stride,
            log_base=args.log_base,
            use_random=args.use_random,
            use_tome=args.use_tome,
            system_prompt_setting=args.system_prompt_setting,
            embedding_mode=args.embedding_mode,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            attn_impl=args.attn_impl,
        )
        evaluate(config)
    except (FileExistsError, FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        print(f"Primitive metric evaluation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
