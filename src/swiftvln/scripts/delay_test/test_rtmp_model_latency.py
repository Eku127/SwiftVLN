#!/usr/bin/env python3
"""Measure repeated RTMP capture -> 448x448 input -> real model inference latency.

This test follows the production deployment-session queue. The first frame of
each round runs real model inference and fills a four-action queue. After each
simulated flight action, a fresh 448x448 RTMP frame is sent as feedback; the
next three interactions consume the queue without running model inference.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Sequence

import cv2
from PIL import Image


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

MODEL_INPUT_SIZE = (448, 448)
ACTION_NAMES = {
    0: "STOP",
    1: "FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
}
SUMMARY_METRIC_DETAILS = {
    "capture_ms": "请求后等待并取得一张最新RTMP解码帧",
    "preprocess_ms": "原始BGR帧转换、裁剪并缩放为448×448 RGB",
    "input_save_ms": "将448×448模型输入编码并保存为JPEG",
    "image_to_action_response_ms": "448×448图像进入推理流程至模型Action解析完成",
    "true_inference_response_ms": "每次真实调用SwiftVLN模型的黑盒响应时间",
    "queue_response_ms": "反馈图进入Session后弹出已完成动作并返回队列下一动作",
    "capture_to_action_response_ms": "请求RTMP新帧至获得模型Action的端到端感知时间",
    "flight_action_ms": "模拟飞控执行当前Action的持续时间",
    "node_total_ms": "单节点从请求抽帧到当前飞控Action结束的总时间",
    "completed_round_total_ms": "完整四节点测试轮次的总时间",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def console_log(stage: str, message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{stage}] {message}", flush=True)


def elapsed_ms(start_ns: int, end_ns: int | None = None) -> float:
    if end_ns is None:
        end_ns = time.perf_counter_ns()
    return (end_ns - start_ns) / 1_000_000.0


class EventLogger:
    def __init__(self, path: Path, run_id: str):
        self.path = path
        self.run_id = run_id
        self._handle = path.open("a", encoding="utf-8", buffering=1)

    def emit(self, event: str, **fields: Any) -> None:
        payload = {
            "run_id": self.run_id,
            "event": event,
            "wall_time": utc_now(),
            "monotonic_ns": time.perf_counter_ns(),
            **fields,
        }
        self._handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class DeployProcessClient:
    """JSONL client for the production start_swiftvln_deploy.sh entrypoint."""

    def __init__(
        self,
        script_path: Path,
        model_name: str,
        session_root: Path,
        event_logger: EventLogger,
    ):
        self.script_path = script_path
        self.model_name = model_name
        self.session_root = session_root
        self.event_logger = event_logger
        self.process: subprocess.Popen[str] | None = None
        self._stdout_queue: queue.Queue[str | None] = queue.Queue()
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self.session_started = False

    def start(
        self,
        startup_timeout_s: float,
        stop_event: threading.Event,
    ) -> dict[str, Any]:
        environment = os.environ.copy()
        environment["DEPLOY_PYTHON"] = sys.executable
        environment["PYTHONUNBUFFERED"] = "1"
        command = [
            "bash",
            str(self.script_path),
            self.model_name,
            str(self.session_root),
        ]
        self.process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            name="swiftvln-deploy-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            name="swiftvln-deploy-stderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        return self._wait_for_message(
            expected_types={"ready"},
            timeout_s=startup_timeout_s,
            stop_event=stop_event,
        )

    def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            for line in self.process.stdout:
                self._stdout_queue.put(line.rstrip("\n"))
        finally:
            self._stdout_queue.put(None)

    def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for line in self.process.stderr:
            text = line.rstrip("\n")
            if text:
                console_log("DEPLOY-ERR", text)

    def _wait_for_message(
        self,
        expected_types: set[str],
        timeout_s: float,
        stop_event: threading.Event,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            if stop_event.is_set():
                raise InterruptedError("Stop requested while waiting for Deploy response")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for Deploy message: {sorted(expected_types)}"
                )
            try:
                line = self._stdout_queue.get(timeout=min(remaining, 0.1))
            except queue.Empty:
                if self.process is not None and self.process.poll() is not None:
                    raise RuntimeError(
                        f"Deploy process exited with code {self.process.returncode}"
                    )
                continue
            if line is None:
                return_code = self.process.poll() if self.process is not None else None
                raise RuntimeError(f"Deploy stdout closed; exit code={return_code}")
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                console_log("DEPLOY", line)
                continue
            if not isinstance(payload, dict):
                console_log("DEPLOY", line)
                continue

            message_type = str(payload.get("type", ""))
            self.event_logger.emit("deploy_response", payload=payload)
            if message_type in {"startup_error", "error"} or not payload.get("ok", True):
                raise RuntimeError(
                    f"Deploy {message_type or 'error'}: {payload.get('error', payload)}"
                )
            if message_type in expected_types:
                return payload
            console_log("DEPLOY", f"忽略非目标响应：{payload}")

    def send(
        self,
        command: dict[str, Any],
        expected_type: str,
        timeout_s: float,
        stop_event: threading.Event,
    ) -> dict[str, Any]:
        if (
            self.process is None
            or self.process.stdin is None
            or self.process.poll() is not None
        ):
            raise RuntimeError("Deploy process is not running")
        serialized = json.dumps(command, ensure_ascii=False)
        self.event_logger.emit("deploy_command", command=command)
        self.process.stdin.write(serialized + "\n")
        self.process.stdin.flush()
        response = self._wait_for_message(
            expected_types={expected_type},
            timeout_s=timeout_s,
            stop_event=stop_event,
        )
        if expected_type == "start":
            self.session_started = True
        return response

    def close(self, reason: str, timeout_s: float = 30.0) -> None:
        if self.process is None:
            return
        if self.process.poll() is None and self.session_started:
            try:
                self.send(
                    {"type": "end", "reason": reason},
                    expected_type="end",
                    timeout_s=timeout_s,
                    stop_event=threading.Event(),
                )
            except Exception as exc:
                console_log("DEPLOY", f"发送end失败：{exc}")
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        try:
            self.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5.0)


class LatestFrameReader:
    """Continuously drain RTMP and retain only the newest decoded frame."""

    def __init__(self, rtmp_url: str, open_timeout_ms: int, read_timeout_ms: int):
        self.rtmp_url = rtmp_url
        self.open_timeout_ms = open_timeout_ms
        self.read_timeout_ms = read_timeout_ms
        self._capture: cv2.VideoCapture | None = None
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame: Any = None
        self._sequence = 0
        self._last_error: str | None = None

    def start(self) -> None:
        capture = cv2.VideoCapture()
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_timeout_ms)
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout_ms)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.open(self.rtmp_url, cv2.CAP_FFMPEG):
            capture.release()
            raise RuntimeError(f"Unable to open RTMP stream: {self.rtmp_url}")

        self._capture = capture
        self._thread = threading.Thread(
            target=self._read_loop,
            name="rtmp-latest-frame-reader",
            daemon=True,
        )
        self._thread.start()

    def _read_loop(self) -> None:
        assert self._capture is not None
        consecutive_failures = 0
        while not self._stop_event.is_set():
            ok, frame = self._capture.read()
            if not ok:
                consecutive_failures += 1
                with self._condition:
                    self._last_error = (
                        f"RTMP frame read failed {consecutive_failures} consecutive time(s)"
                    )
                    self._condition.notify_all()
                self._stop_event.wait(0.05)
                continue

            consecutive_failures = 0
            with self._condition:
                self._frame = frame
                self._sequence += 1
                self._last_error = None
                self._condition.notify_all()

    def request_fresh_frame(
        self,
        timeout_s: float,
        external_stop_event: threading.Event,
    ) -> tuple[int, Any]:
        """Return the first decoded frame newer than this method invocation."""
        deadline = time.monotonic() + timeout_s
        with self._condition:
            baseline_sequence = self._sequence
            while self._sequence <= baseline_sequence:
                if external_stop_event.is_set():
                    raise InterruptedError("Stop requested while waiting for RTMP frame")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = f": {self._last_error}" if self._last_error else ""
                    raise TimeoutError(f"Timed out waiting for a fresh RTMP frame{detail}")
                self._condition.wait(timeout=min(remaining, 0.1))

            return self._sequence, self._frame.copy()

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._capture is not None:
            self._capture.release()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class SimulatedFlightController:
    """Replace this adapter with the real async flight-control API later."""

    def __init__(self, action_duration_s: float, stop_event: threading.Event):
        self.action_duration_s = action_duration_s
        self.stop_event = stop_event

    def execute(self, action: int) -> tuple[bool, float]:
        started_ns = time.perf_counter_ns()
        completed = not self.stop_event.wait(self.action_duration_s)
        return completed, elapsed_ms(started_ns)


def prepare_model_image(frame_bgr: Any, resize_mode: str) -> Image.Image:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    target_width, target_height = MODEL_INPUT_SIZE

    if resize_mode == "stretch":
        resized = cv2.resize(rgb, MODEL_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    elif resize_mode == "center-crop":
        height, width = rgb.shape[:2]
        side = min(height, width)
        top = (height - side) // 2
        left = (width - side) // 2
        cropped = rgb[top : top + side, left : left + side]
        resized = cv2.resize(cropped, MODEL_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    else:
        raise ValueError(f"Unsupported resize mode: {resize_mode}")

    image = Image.fromarray(resized, mode="RGB")
    if image.size != MODEL_INPUT_SIZE or image.mode != "RGB":
        raise RuntimeError(
            f"Model input must be 448x448 RGB, got size={image.size}, mode={image.mode}"
        )
    return image


def percentile(values: Sequence[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean_ms": None,
            "std_ms": None,
            "min_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
        }
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "std_ms": statistics.pstdev(values),
        "min_ms": min(values),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "max_ms": max(values),
    }


def format_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def write_reports(
    run_dir: Path,
    metadata: dict[str, Any],
    rounds: list[dict[str, Any]],
    stop_reason: str,
) -> None:
    nodes = [node for round_record in rounds for node in round_record["nodes"]]
    action_latencies = [
        node["flight_action_ms"]
        for node in nodes
        if node["flight_action_ms"] is not None
    ]
    true_inference_latencies = [
        node["image_to_action_response_ms"]
        for node in nodes
        if node["performed_inference"]
    ]
    queue_response_latencies = [
        node["image_to_action_response_ms"]
        for node in nodes
        if not node["performed_inference"]
    ]
    metric_values = {
        "capture_ms": [node["capture_ms"] for node in nodes],
        "preprocess_ms": [node["preprocess_ms"] for node in nodes],
        "input_save_ms": [node["input_save_ms"] for node in nodes],
        "image_to_action_response_ms": [
            node["image_to_action_response_ms"] for node in nodes
        ],
        "true_inference_response_ms": true_inference_latencies,
        "queue_response_ms": queue_response_latencies,
        "capture_to_action_response_ms": [
            node["capture_to_action_response_ms"] for node in nodes
        ],
        "flight_action_ms": action_latencies,
        "node_total_ms": [node["node_total_ms"] for node in nodes],
        "completed_round_total_ms": [
            record["round_total_ms"]
            for record in rounds
            if record["completed"] and record["round_total_ms"] is not None
        ],
    }
    summary = {name: summarize(values) for name, values in metric_values.items()}
    report = {
        "metadata": metadata,
        "stop_reason": stop_reason,
        "updated_at": utc_now(),
        "started_rounds": len(rounds),
        "completed_rounds": sum(record["completed"] for record in rounds),
        "image_action_interactions": len(nodes),
        "true_inference_count": len(true_inference_latencies),
        "completed_actions": len(action_latencies),
        "summary": summary,
        "summary_details": SUMMARY_METRIC_DETAILS,
        "rounds": rounds,
    }

    report_suffix = metadata["report_suffix"]
    json_path = run_dir / f"report_{report_suffix}.json"
    markdown_path = run_dir / f"report_{report_suffix}.md"
    csv_path = run_dir / f"interactions_{report_suffix}.csv"

    json_tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    json_tmp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    json_tmp.replace(json_path)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "round",
            "node",
            "frame_sequence",
            "capture_ms",
            "preprocess_ms",
            "input_save_ms",
            "image_to_action_response_ms",
            "performed_inference",
            "capture_to_action_response_ms",
            "generated_actions",
            "remaining_actions",
            "next_action",
            "flight_action_ms",
            "node_total_ms",
            "node_stop_reason",
            "stop_image_path",
            "inference_frame_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in rounds:
            for node in record["nodes"]:
                writer.writerow(
                    {
                        "round": record["round"],
                        "node": node["node"],
                        "frame_sequence": node["frame_sequence"],
                        "capture_ms": node["capture_ms"],
                        "preprocess_ms": node["preprocess_ms"],
                        "input_save_ms": node["input_save_ms"],
                        "image_to_action_response_ms": node[
                            "image_to_action_response_ms"
                        ],
                        "performed_inference": node["performed_inference"],
                        "capture_to_action_response_ms": node[
                            "capture_to_action_response_ms"
                        ],
                        "generated_actions": json.dumps(node["generated_actions"]),
                        "remaining_actions": json.dumps(node["remaining_actions"]),
                        "next_action": node["next_action"],
                        "flight_action_ms": node["flight_action_ms"],
                        "node_total_ms": node["node_total_ms"],
                        "node_stop_reason": node["node_stop_reason"],
                        "stop_image_path": node["stop_image_path"],
                        "inference_frame_path": node["inference_frame_path"],
                    }
                )

    lines = [
        "# RTMP → 448×448 → SwiftVLN 时延报告",
        "",
        f"- Run ID：`{metadata['run_id']}`",
        f"- Report suffix：`{report_suffix}`",
        f"- 停止原因：`{stop_reason}`",
        f"- 已开始/完成轮数：{len(rounds)}/{report['completed_rounds']}",
        f"- Image→Action交互次数：{len(nodes)}",
        f"- 真实模型推理次数：{len(true_inference_latencies)}",
        f"- 完成飞控动作数：{len(action_latencies)}",
        f"- 模型：`{metadata['model_name']}`",
        f"- 指令：`{metadata['instruction']}`",
        f"- RTMP：`{metadata['rtmp_url']}`",
        f"- 模型输入：`448×448 RGB`（{metadata['resize_mode']}）",
        f"- 推理帧额外归档："
        f"`{metadata.get('inference_frame_dir') if metadata.get('save_inference_frames') else '未启用'}`",
        f"- STOP触发图片：`{metadata.get('stop_image_path') or '-'}`",
        "",
        "## 汇总",
        "",
        "| 环节 | 详解 | N | Mean(ms) | Std(ms) | P50(ms) | P95(ms) | P99(ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric_name, values in summary.items():
        lines.append(
            f"| {metric_name} | {SUMMARY_METRIC_DETAILS[metric_name]} | "
            f"{values['count']} | "
            f"{format_ms(values['mean_ms'])} | {format_ms(values['std_ms'])} | "
            f"{format_ms(values['p50_ms'])} | {format_ms(values['p95_ms'])} | "
            f"{format_ms(values['p99_ms'])} |"
        )

    lines.extend(
        [
            "",
            "## 每次 Image → Action 记录",
            "",
            "| 轮次 | 节点 | 抽帧(ms) | 448处理(ms) | Image→响应(ms) | "
            "真推理 | 新生成Actions | 剩余队列 | 本次执行 | 飞控(ms) | 节点总计(ms) |",
            "|---:|---:|---:|---:|---:|:---:|---|---|---:|---:|---:|",
        ]
    )
    for record in rounds:
        for node in record["nodes"]:
            lines.append(
                f"| {record['round']} | {node['node']} | {node['capture_ms']:.3f} | "
                f"{node['preprocess_ms']:.3f} | "
                f"{node['image_to_action_response_ms']:.3f} | "
                f"{'是' if node['performed_inference'] else '否'} | "
                f"`{node['generated_actions']}` | `{node['remaining_actions']}` | "
                f"{node['next_action']} | {format_ms(node['flight_action_ms'])} | "
                f"{node['node_total_ms']:.3f} |"
            )

    markdown_tmp = markdown_path.with_suffix(markdown_path.suffix + ".tmp")
    markdown_tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    markdown_tmp.replace(markdown_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repeatedly capture RTMP frames, prepare exact 448x448 RGB inputs, "
            "send them through the production SwiftVLN deploy-session queue, "
            "and simulate flight-control execution."
        )
    )
    parser.add_argument("--rtmp-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument(
        "--result-root",
        default=str(REPO_ROOT / "results" / "delay_test"),
    )
    parser.add_argument(
        "--report-suffix",
        default=None,
        help="Suffix for report files; defaults to time.time_ns().",
    )
    parser.add_argument(
        "--stop-image-dir",
        default="~/tmp",
        help="Directory for the exact 448x448 input that triggers model STOP.",
    )
    parser.add_argument(
        "--save-inference-frames",
        action="store_true",
        help="Archive each 448x448 frame that actually triggers model inference.",
    )
    parser.add_argument(
        "--inference-frame-dir",
        default="~/tmp",
        help="Archive directory used with --save-inference-frames.",
    )
    parser.add_argument(
        "--resize-mode",
        choices=("center-crop", "stretch"),
        default="center-crop",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=0,
        help="Stop after this many rounds; 0 means no limit.",
    )
    parser.add_argument("--frame-timeout", type=float, default=10.0)
    parser.add_argument("--rtmp-open-timeout-ms", type=int, default=10_000)
    parser.add_argument("--rtmp-read-timeout-ms", type=int, default=5_000)
    parser.add_argument("--deploy-startup-timeout", type=float, default=600.0)
    parser.add_argument("--deploy-response-timeout", type=float, default=120.0)
    parser.add_argument(
        "--flight-action-seconds",
        type=float,
        default=5.0,
        help="Simulated duration of every non-STOP flight action.",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Also persist each raw RTMP frame inside the run result directory.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.max_rounds < 0:
        raise ValueError("--max-rounds must be >= 0")
    if args.frame_timeout <= 0:
        raise ValueError("--frame-timeout must be > 0")
    if args.flight_action_seconds < 0:
        raise ValueError("--flight-action-seconds must be >= 0")
    if args.rtmp_open_timeout_ms <= 0 or args.rtmp_read_timeout_ms <= 0:
        raise ValueError("RTMP timeout values must be > 0")
    if args.deploy_startup_timeout <= 0 or args.deploy_response_timeout <= 0:
        raise ValueError("Deploy timeout values must be > 0")
    if args.report_suffix is not None:
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
        if not args.report_suffix or any(char not in allowed for char in args.report_suffix):
            raise ValueError(
                "--report-suffix may contain only letters, numbers, '-', '_' and '.'"
            )


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)

    run_id = datetime.now().strftime("rtmp_model_%Y%m%d_%H%M%S")
    run_dir = Path(args.result_root).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model_inputs_dir = run_dir / "model_inputs_448"
    model_inputs_dir.mkdir()
    raw_frames_dir = run_dir / "raw_frames"
    if args.save_frames:
        raw_frames_dir.mkdir()
    inference_frame_dir = Path(args.inference_frame_dir).expanduser().resolve()
    if args.save_inference_frames:
        inference_frame_dir.mkdir(parents=True, exist_ok=True)
    report_suffix = args.report_suffix or str(time.time_ns())

    metadata = {
        "run_id": run_id,
        "report_suffix": report_suffix,
        "started_at": utc_now(),
        "repo_root": str(REPO_ROOT),
        "rtmp_url": args.rtmp_url,
        "model_name": args.model_name,
        "deploy_entrypoint": str(
            REPO_ROOT / "src/swiftvln/scripts/deploy/start_swiftvln_deploy.sh"
        ),
        "instruction": args.instruction,
        "model_input": {"width": 448, "height": 448, "mode": "RGB"},
        "resize_mode": args.resize_mode,
        "flight_action_seconds": args.flight_action_seconds,
        "max_rounds": args.max_rounds,
        "save_frames": args.save_frames,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "stop_image_dir": str(Path(args.stop_image_dir).expanduser().resolve()),
        "stop_image_path": None,
        "save_inference_frames": args.save_inference_frames,
        "inference_frame_dir": str(inference_frame_dir),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    event_logger = EventLogger(run_dir / "events.jsonl", run_id)
    stop_event = threading.Event()
    stop_reason_holder = {"reason": "running"}
    rounds: list[dict[str, Any]] = []
    frame_reader: LatestFrameReader | None = None
    deploy_client: DeployProcessClient | None = None

    def request_stop(reason: str) -> None:
        if not stop_event.is_set():
            stop_reason_holder["reason"] = reason
            event_logger.emit("stop_requested", reason=reason)
            stop_event.set()
            console_log("STOP", f"收到停止信号：{reason}；正在整理报告")

    def handle_signal(signum: int, _frame: Any) -> None:
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        signal_name = signal.Signals(signum).name.lower()
        request_stop(signal_name)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        report_path = run_dir / f"report_{report_suffix}.md"
        write_reports(run_dir, metadata, rounds, "running")
        console_log("START", f"Run ID：{run_id}")
        console_log("START", f"结果目录：{run_dir}")
        console_log("START", f"增量报告：{report_path}")
        console_log(
            "CONFIG",
            f"448×448 RGB；模拟飞控每个动作 {args.flight_action_seconds:.3f}s",
        )
        event_logger.emit("run_started", metadata=metadata)

        deploy_script = (
            REPO_ROOT / "src/swiftvln/scripts/deploy/start_swiftvln_deploy.sh"
        )
        console_log("DEPLOY", f"启动正式部署入口：{deploy_script}")
        deploy_client = DeployProcessClient(
            script_path=deploy_script,
            model_name=args.model_name,
            session_root=run_dir / "deploy_sessions",
            event_logger=event_logger,
        )
        model_load_start_ns = time.perf_counter_ns()
        ready_response = deploy_client.start(
            startup_timeout_s=args.deploy_startup_timeout,
            stop_event=stop_event,
        )
        metadata["model_load_ms"] = elapsed_ms(model_load_start_ns)
        metadata["checkpoint_path"] = ready_response.get("checkpoint_path")
        metadata["gpu"] = ready_response.get("gpu")
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        event_logger.emit(
            "model_ready",
            checkpoint_path=metadata["checkpoint_path"],
            model_load_ms=metadata["model_load_ms"],
        )
        console_log(
            "MODEL",
            f"模型加载完成，耗时 {metadata['model_load_ms']:.3f}ms",
        )
        if stop_event.is_set():
            raise InterruptedError("Stop requested after model loading")
        start_response = deploy_client.send(
            {
                "type": "start",
                "instruction": args.instruction,
                "session_id": run_id,
            },
            expected_type="start",
            timeout_s=args.deploy_response_timeout,
            stop_event=stop_event,
        )
        event_logger.emit("deploy_session_started", response=start_response)
        console_log("SESSION", f"正式Deploy session已启动：{run_id}")

        console_log("RTMP", f"正在连接：{args.rtmp_url}")
        frame_reader = LatestFrameReader(
            rtmp_url=args.rtmp_url,
            open_timeout_ms=args.rtmp_open_timeout_ms,
            read_timeout_ms=args.rtmp_read_timeout_ms,
        )
        frame_reader.start()
        event_logger.emit("rtmp_reader_ready")
        console_log("RTMP", "连接成功，后台持续读取最新帧")

        flight_controller = SimulatedFlightController(
            action_duration_s=args.flight_action_seconds,
            stop_event=stop_event,
        )

        round_index = 0
        last_inference_input_path: Path | None = None
        while not stop_event.is_set():
            if args.max_rounds and round_index >= args.max_rounds:
                request_stop("max_rounds")
                break

            round_index += 1
            round_start_ns = time.perf_counter_ns()
            round_record = {
                "round": round_index,
                "completed": False,
                "nodes": [],
                "round_total_ms": None,
                "round_stop_reason": None,
            }
            rounds.append(round_record)
            event_logger.emit("round_started", round=round_index)
            console_log("ROUND", f"开始第 {round_index} 轮")

            for node_index in range(1, 5):
                if stop_event.is_set():
                    break

                node_start_ns = time.perf_counter_ns()
                console_log(
                    "CAPTURE",
                    f"第 {round_index} 轮节点 {node_index}：等待新RTMP帧",
                )
                capture_start_ns = time.perf_counter_ns()
                frame_sequence, frame_bgr = frame_reader.request_fresh_frame(
                    timeout_s=args.frame_timeout,
                    external_stop_event=stop_event,
                )
                capture_ms = elapsed_ms(capture_start_ns)
                event_logger.emit(
                    "frame_captured",
                    round=round_index,
                    node=node_index,
                    frame_sequence=frame_sequence,
                    width=int(frame_bgr.shape[1]),
                    height=int(frame_bgr.shape[0]),
                    capture_ms=capture_ms,
                )
                console_log(
                    "CAPTURE",
                    f"获得帧 #{frame_sequence}，"
                    f"{frame_bgr.shape[1]}×{frame_bgr.shape[0]}，"
                    f"耗时 {capture_ms:.3f}ms",
                )

                preprocess_start_ns = time.perf_counter_ns()
                model_image = prepare_model_image(frame_bgr, args.resize_mode)
                preprocess_ms = elapsed_ms(preprocess_start_ns)
                if model_image.size != MODEL_INPUT_SIZE or model_image.mode != "RGB":
                    raise RuntimeError(
                        "Model input must be exactly 448x448 RGB, got "
                        f"{model_image.size} {model_image.mode}"
                    )

                model_input_path = (
                    model_inputs_dir
                    / f"round_{round_index:04d}_node_{node_index}_448.jpg"
                )
                input_save_start_ns = time.perf_counter_ns()
                model_image.save(model_input_path, format="JPEG", quality=95)
                input_save_ms = elapsed_ms(input_save_start_ns)
                event_logger.emit(
                    "model_input_ready",
                    round=round_index,
                    node=node_index,
                    width=448,
                    height=448,
                    mode=model_image.mode,
                    path=str(model_input_path),
                    preprocess_ms=preprocess_ms,
                    input_save_ms=input_save_ms,
                )
                console_log(
                    "INPUT",
                    f"448×448 RGB 已准备：{model_input_path.name}；"
                    f"处理 {preprocess_ms:.3f}ms，落盘 {input_save_ms:.3f}ms",
                )

                console_log(
                    "DEPLOY",
                    f"第 {round_index} 轮节点 {node_index}：发送image并等待响应",
                )
                response_start_ns = time.perf_counter_ns()
                assert deploy_client is not None
                response = deploy_client.send(
                    {"type": "image", "image_path": str(model_input_path)},
                    expected_type="image",
                    timeout_s=args.deploy_response_timeout,
                    stop_event=stop_event,
                )
                image_to_action_response_ms = elapsed_ms(response_start_ns)
                capture_to_action_response_ms = elapsed_ms(capture_start_ns)

                performed_inference = bool(response["performed_inference"])
                generated_actions = [int(action) for action in response["actions"]]
                remaining_actions = [
                    int(action) for action in response["remaining_actions"]
                ]
                next_action_value = response["next_action"]
                next_action = (
                    int(next_action_value) if next_action_value is not None else None
                )
                if any(
                    action not in ACTION_NAMES
                    for action in generated_actions + remaining_actions
                ):
                    raise RuntimeError(
                        "Deploy returned unsupported actions: "
                        f"generated={generated_actions}, remaining={remaining_actions}"
                    )

                node_stop_reason: str | None = None
                if node_index == 1 and not performed_inference:
                    node_stop_reason = "missing_round_start_inference"
                elif node_index > 1 and performed_inference:
                    node_stop_reason = f"unexpected_inference_at_node_{node_index}"
                elif node_index == 1:
                    has_terminal_stop = (
                        0 in remaining_actions and len(remaining_actions) <= 4
                    )
                    if len(remaining_actions) != 4 and not has_terminal_stop:
                        node_stop_reason = (
                            f"invalid_action_count_{len(remaining_actions)}"
                        )
                else:
                    expected_remaining = 5 - node_index
                    has_terminal_stop = (
                        0 in remaining_actions
                        and len(remaining_actions) <= expected_remaining
                    )
                    if len(remaining_actions) != expected_remaining and not has_terminal_stop:
                        node_stop_reason = (
                            f"invalid_remaining_count_node_{node_index}_"
                            f"got_{len(remaining_actions)}"
                        )

                inference_frame_path: str | None = None
                if performed_inference:
                    last_inference_input_path = model_input_path
                if args.save_inference_frames and performed_inference:
                    archived_path = (
                        inference_frame_dir
                        / (
                            f"swiftvln_inference_{report_suffix}_"
                            f"round{round_index:04d}_node{node_index}_448.jpg"
                        )
                    )
                    shutil.copy2(model_input_path, archived_path)
                    inference_frame_path = str(archived_path)
                    event_logger.emit(
                        "inference_frame_archived",
                        round=round_index,
                        node=node_index,
                        path=inference_frame_path,
                    )
                    console_log("INPUT", f"推理帧已额外归档：{archived_path}")

                node_record = {
                    "node": node_index,
                    "frame_sequence": frame_sequence,
                    "capture_ms": capture_ms,
                    "preprocess_ms": preprocess_ms,
                    "input_save_ms": input_save_ms,
                    "image_to_action_response_ms": image_to_action_response_ms,
                    "performed_inference": performed_inference,
                    "capture_to_action_response_ms": capture_to_action_response_ms,
                    "raw_action_text": response["raw_action_text"],
                    "generated_actions": generated_actions,
                    "remaining_actions": remaining_actions,
                    "next_action": next_action,
                    "completed_previous_action": response["completed_action"],
                    "flight_action_ms": None,
                    "flight_action_completed": False,
                    "node_total_ms": elapsed_ms(node_start_ns),
                    "node_stop_reason": node_stop_reason,
                    "stop_image_path": None,
                    "inference_frame_path": inference_frame_path,
                }
                round_record["nodes"].append(node_record)

                if args.save_frames:
                    cv2.imwrite(
                        str(
                            raw_frames_dir
                            / f"round_{round_index:04d}_node_{node_index}_raw.jpg"
                        ),
                        frame_bgr,
                    )

                event_logger.emit(
                    "deploy_image_response",
                    round=round_index,
                    node=node_index,
                    image_to_action_response_ms=image_to_action_response_ms,
                    performed_inference=performed_inference,
                    generated_actions=generated_actions,
                    remaining_actions=remaining_actions,
                    next_action=next_action,
                )
                console_log(
                    "DEPLOY",
                    f"响应 {image_to_action_response_ms:.3f}ms；"
                    f"真实推理={performed_inference}；"
                    f"新生成={generated_actions}；剩余={remaining_actions}；"
                    f"本次执行={next_action}",
                )
                write_reports(
                    run_dir,
                    metadata,
                    rounds,
                    node_stop_reason or stop_reason_holder["reason"],
                )

                if node_stop_reason is not None:
                    round_record["round_stop_reason"] = node_stop_reason
                    request_stop(node_stop_reason)
                    break
                if next_action is None:
                    node_stop_reason = "missing_next_action"
                    node_record["node_stop_reason"] = node_stop_reason
                    round_record["round_stop_reason"] = node_stop_reason
                    request_stop(node_stop_reason)
                    break
                if next_action == 0:
                    node_stop_reason = "model_stop"
                    node_record["node_stop_reason"] = node_stop_reason
                    round_record["round_stop_reason"] = node_stop_reason
                    stop_image_dir = Path(args.stop_image_dir).expanduser().resolve()
                    try:
                        stop_image_dir.mkdir(parents=True, exist_ok=True)
                        stop_source_path = (
                            last_inference_input_path or model_input_path
                        )
                        stop_image_path = (
                            stop_image_dir
                            / (
                                f"swiftvln_model_stop_{report_suffix}_"
                                f"round{round_index:04d}_node{node_index}_448.jpg"
                            )
                        )
                        shutil.copy2(stop_source_path, stop_image_path)
                        node_record["stop_image_path"] = str(stop_image_path)
                        metadata["stop_image_path"] = str(stop_image_path)
                        (run_dir / "metadata.json").write_text(
                            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        console_log(
                            "STOP",
                            f"已保存生成STOP队列的448×448推理输入：{stop_image_path}",
                        )
                    except OSError as exc:
                        metadata["stop_image_error"] = str(exc)
                        console_log("STOP", f"保存STOP图片失败：{exc}")
                    event_logger.emit(
                        "model_stop",
                        round=round_index,
                        node=node_index,
                        generated_actions=generated_actions,
                        remaining_actions=remaining_actions,
                        stop_image_path=node_record["stop_image_path"],
                    )
                    write_reports(run_dir, metadata, rounds, node_stop_reason)
                    request_stop(node_stop_reason)
                    break

                event_logger.emit(
                    "flight_action_started",
                    round=round_index,
                    node=node_index,
                    action=next_action,
                    action_name=ACTION_NAMES[next_action],
                )
                console_log(
                    "FLIGHT",
                    f"开始模拟 {ACTION_NAMES[next_action]}({next_action})，"
                    f"预计 {args.flight_action_seconds:.3f}s",
                )
                completed, execution_ms = flight_controller.execute(next_action)
                node_record["flight_action_ms"] = execution_ms
                node_record["flight_action_completed"] = completed
                node_record["node_total_ms"] = elapsed_ms(node_start_ns)
                event_logger.emit(
                    "flight_action_finished",
                    round=round_index,
                    node=node_index,
                    action=next_action,
                    action_name=ACTION_NAMES[next_action],
                    completed=completed,
                    execution_ms=execution_ms,
                )
                console_log(
                    "FLIGHT",
                    f"动作 {'完成' if completed else '中断'}，耗时 {execution_ms:.3f}ms",
                )

                if not completed:
                    node_stop_reason = stop_reason_holder["reason"]
                    node_record["node_stop_reason"] = node_stop_reason
                    round_record["round_stop_reason"] = node_stop_reason

                write_reports(
                    run_dir,
                    metadata,
                    rounds,
                    node_stop_reason or stop_reason_holder["reason"],
                )
                print(
                    f"[round {round_index} node {node_index}] "
                    f"capture={capture_ms:.3f}ms preprocess={preprocess_ms:.3f}ms "
                    f"image_to_actions={image_to_action_response_ms:.3f}ms "
                    f"real_inference={performed_inference} "
                    f"generated={generated_actions} remaining={remaining_actions} "
                    f"next={next_action} "
                    f"flight={execution_ms:.3f}ms",
                    flush=True,
                )
                if not completed:
                    break

            round_record["completed"] = (
                len(round_record["nodes"]) == 4
                and all(
                    node["flight_action_completed"]
                    for node in round_record["nodes"]
                )
                and not stop_event.is_set()
            )
            round_record["round_total_ms"] = elapsed_ms(round_start_ns)
            write_reports(
                run_dir,
                metadata,
                rounds,
                round_record["round_stop_reason"] or stop_reason_holder["reason"],
            )
            event_logger.emit(
                "round_finished",
                round=round_index,
                completed=round_record["completed"],
                round_total_ms=round_record["round_total_ms"],
                stop_reason=round_record["round_stop_reason"],
            )

    except KeyboardInterrupt:
        request_stop("sigint")
    except InterruptedError:
        if not stop_event.is_set():
            request_stop("interrupted")
    except Exception as exc:
        stop_reason_holder["reason"] = f"error:{type(exc).__name__}"
        stop_event.set()
        event_logger.emit(
            "run_error",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
    finally:
        final_reason = stop_reason_holder["reason"]
        if final_reason == "running":
            final_reason = "completed"

        if frame_reader is not None:
            frame_reader.stop()
        if deploy_client is not None:
            deploy_client.close(reason=final_reason)

        metadata["finished_at"] = utc_now()
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        console_log("REPORT", "正在写入最终统计报告")
        write_reports(run_dir, metadata, rounds, final_reason)
        report_path = run_dir / f"report_{report_suffix}.md"
        event_logger.emit(
            "report_written",
            report_path=str(report_path),
            stop_reason=final_reason,
            completed_rounds=len(rounds),
        )
        event_logger.close()
        print(f"Stop reason: {final_reason}", flush=True)
        print(f"Report: {report_path}", flush=True)

    return 0 if not stop_reason_holder["reason"].startswith("error:") else 1


if __name__ == "__main__":
    raise SystemExit(main())
