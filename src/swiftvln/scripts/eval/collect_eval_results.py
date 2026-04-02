#!/usr/bin/env python3
"""Collect eval summaries into versioned CSV files.

Output CSV naming:
  results/eval_collected/<split>/eval_results_data<version>.csv

The split (val_seen / val_unseen / test) is taken from --eval-split.
If --eval-split is not provided, it is inferred from the result_path directory
structure: results/eval/<arch>/<model>/<split>/<timestamp>/ → parent.name.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def pct(v: Any) -> str:
    if isinstance(v, (int, float)):
        return f"{v * 100:.2f}%"
    return ""


def f4(v: Any) -> str:
    if isinstance(v, (int, float)):
        return f"{v:.4f}"
    return ""


def f2(v: Any) -> str:
    if isinstance(v, (int, float)):
        return f"{v:.2f}"
    return ""


def parse_model_type(model_name: str) -> str:
    for t in ("overlapvln", "streamvln", "compressvln", "navila", "monovln", "uninavid"):
        if model_name.startswith(f"{t}-"):
            return t
    return "unknown"


def parse_data_version(model_name: str, model_path: str) -> str:
    m = re.search(r"data(\d{6})", model_name)
    if m:
        return m.group(1)
    m = re.search(r"ver_(\d{6})", model_path or "")
    if m:
        return m.group(1)
    return "unknown"


def infer_plan(model_name: str, model_type: str) -> str:
    if model_type == "overlapvln":
        # History processor type determines the base method
        if "-sgtc-k" in model_name:
            m = re.search(r"-sgtc-k(\d+)", model_name)
            base = f"baseline + sgtc-k{m.group(1)}" if m else "baseline + sgtc"
        elif "-gtc-k" in model_name:
            m = re.search(r"-gtc-k(\d+)", model_name)
            base = f"baseline + gtc-k{m.group(1)}" if m else "baseline + gtc"
        elif re.search(r"-tome-s\d+", model_name):
            # per_frame + GridToMe compression
            base = "baseline + tome"
        else:
            # per_frame + avg pool (default path)
            # Naming format: -pf-h{H}-b{B}-pool-s{S}-
            m_log = re.search(r"-b([0-9]+\.[0-9]+)-", model_name)
            log_base = m_log.group(1) if m_log else "1.0"
            m_stride = re.search(r"-(?:pool|tome)-s(\d+)", model_name)
            stride = m_stride.group(1) if m_stride else "2"

            if log_base not in ("1.0", "1"):
                base = f"baseline + log{log_base}"
            elif stride != "2":
                base = f"baseline + s{stride}"
            else:
                base = "baseline"

        # Additive modifiers stacked on top of the base method
        if "-initial-" in model_name:
            base += " + initial"
        m_qa = re.search(r"-qa(\d+)(?:-|$)", model_name)
        if m_qa:
            base += f" + qa{m_qa.group(1)}"
        return base

    if model_type in ("streamvln", "navila"):
        return f"{model_type} baseline"
    if model_type == "compressvln":
        m = re.search(r"-stride(\d+)-", model_name)
        return f"compress stride{m.group(1)}" if m else "compress baseline"
    if model_type == "uninavid":
        return "uninavid baseline"
    return f"{model_type} run"


def plan_rank(plan: str) -> int:
    order = {
        # overlapvln variants (ascending complexity)
        "baseline": 10,
        "baseline + tome": 20,
        "baseline + s3": 25,
        "baseline + s4": 27,
        "baseline + log2.0": 30,
        "baseline + log3.0": 32,
        "baseline + initial": 35,
        "baseline + qa15": 40,
        "baseline + qa30": 50,
        "baseline + gtc-k256": 55,
        "baseline + gtc-k512": 60,
        "baseline + gtc": 62,
        "baseline + sgtc-k256": 65,
        "baseline + sgtc-k512": 70,
        "baseline + sgtc": 72,
        # other model types
        "streamvln baseline": 80,
        "compress baseline": 85,
        "compress stride2": 86,
        "compress stride3": 87,
        "compress stride4": 88,
        "navila baseline": 90,
        "uninavid baseline": 95,
    }
    return order.get(plan, 999)


def strip_run_timestamp(model_name: str) -> str:
    # e.g. ...-20260214-143037
    return re.sub(r"-\d{8}-\d{6}$", "", model_name)


def overlap_variant_rank(model_name: str) -> int:
    # Keep baseline first inside the same setting group; higher = later in table.
    if "-sgtc-k" in model_name:
        return 70
    if "-gtc-k" in model_name:
        return 60
    if re.search(r"-qa\d+", model_name):
        return 50
    if "-initial-" in model_name:
        return 40
    if re.search(r"-tome-s\d+", model_name):
        return 30
    # Non-default log_base (e.g. -b2.0-)
    m_log = re.search(r"-b([0-9]+\.[0-9]+)-", model_name)
    if m_log and m_log.group(1) not in ("1.0", "1"):
        return 20
    # Non-default stride (e.g. -pool-s4-)
    m_stride = re.search(r"-pool-s(\d+)", model_name)
    if m_stride and m_stride.group(1) != "2":
        return 15
    return 10  # baseline: pool + b1.0 + s2


def normalize_overlap_setting_key(model_name: str) -> str:
    key = strip_run_timestamp(model_name)
    # Remove variant markers so experiments with the same base config are grouped.
    key = re.sub(r"-sgtc-k\d+", "", key)
    key = re.sub(r"-gtc-k\d+", "", key)
    key = re.sub(r"-qa\d+", "", key)
    key = key.replace("-initial", "")
    # Normalize method slot: -tome-s{N} and -pool-s{N} both → -s{N}
    key = re.sub(r"-(tome|pool)-s(\d+)", r"-s\2", key)
    key = re.sub(r"--+", "-", key).strip("-")
    return key


def sort_key(row: Dict[str, str]) -> tuple:
    model_name = row.get("model_name", "")
    model_type = row.get("model_type") or parse_model_type(model_name)
    data_version = parse_data_version(model_name, "")

    if model_type == "overlapvln":
        setting_key = normalize_overlap_setting_key(model_name)
        variant_rank = overlap_variant_rank(model_name)
        return (data_version, model_type, setting_key, variant_rank, model_name)

    plan = row.get("plan", infer_plan(model_name, model_type))
    return (data_version, model_type, int(plan_rank(plan)), plan, strip_run_timestamp(model_name), model_name)


def dedupe_rows_by_model_name(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    # Keep the latest record for the same model_name.
    seen: Dict[str, Dict[str, str]] = {}
    for row in rows:
        name = row.get("model_name", "")
        if not name:
            continue
        seen[name] = row
    return list(seen.values())


def pick_type_block(summary: Dict[str, Any], type_name: str) -> Dict[str, Any]:
    by_type = summary.get("by_trajectory_type", {}) or {}
    if type_name in by_type:
        return by_type[type_name] or {}
    if type_name == "LandmarkSet" and "Landmark" in by_type:
        return by_type["Landmark"] or {}
    return {}


def resolve_swanlab_url(summary: Dict[str, Any]) -> str:
    """Get SwanLab URL from evaluation_summary or fall back to train_metadata.json."""
    url = summary.get("swanlab_url", "")
    if url:
        return url
    model_path = summary.get("model_path", "")
    if not model_path:
        return ""
    p = Path(model_path)
    for ancestor in [p.parent.parent, p.parent, p]:
        candidate = ancestor / "train_metadata.json"
        if candidate.is_file():
            try:
                meta = json.loads(candidate.read_text(encoding="utf-8"))
                url = meta.get("swanlab_url", "")
                if url:
                    return url
            except Exception:
                pass
    return ""


def build_row(model_name: str, result_path: Path, summary: Dict[str, Any]) -> Dict[str, str]:
    model_type = parse_model_type(model_name)
    plan = infer_plan(model_name, model_type)
    swanlab_url = resolve_swanlab_url(summary)

    boundary = pick_type_block(summary, "Boundary")
    landmark = pick_type_block(summary, "LandmarkSet")
    road = pick_type_block(summary, "Road")

    row = {
        "model_name": model_name,
        "model_type": model_type,
        "plan": plan,
        "swanlab_url": swanlab_url,
        "collected_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ALL_SR": pct(summary.get("success_rate")),
        "ALL_SPL": f4(summary.get("mean_spl")),
        "ALL_OS": pct(summary.get("oracle_success")),
        "ALL_NE": f2(summary.get("navigation_error")),
        "ALL_Steps": f2(summary.get("avg_steps")),
        "ALL_Episodes": str(summary.get("total_episodes", "")),
        "Boundary_SR": pct(boundary.get("success_rate")),
        "Boundary_SPL": f4(boundary.get("mean_spl")),
        "Boundary_OS": pct(boundary.get("oracle_success")),
        "Boundary_NE": f2(boundary.get("navigation_error")),
        "Boundary_Steps": f2(boundary.get("avg_steps")),
        "Boundary_Episodes": str(boundary.get("total_episodes", "")),
        "LandmarkSet_SR": pct(landmark.get("success_rate")),
        "LandmarkSet_SPL": f4(landmark.get("mean_spl")),
        "LandmarkSet_OS": pct(landmark.get("oracle_success")),
        "LandmarkSet_NE": f2(landmark.get("navigation_error")),
        "LandmarkSet_Steps": f2(landmark.get("avg_steps")),
        "LandmarkSet_Episodes": str(landmark.get("total_episodes", "")),
        "Road_SR": pct(road.get("success_rate")),
        "Road_SPL": f4(road.get("mean_spl")),
        "Road_OS": pct(road.get("oracle_success")),
        "Road_NE": f2(road.get("navigation_error")),
        "Road_Steps": f2(road.get("avg_steps")),
        "Road_Episodes": str(road.get("total_episodes", "")),
    }
    return row


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(csv_path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = [
        "model_name",
        "model_type",
        "plan",
        "swanlab_url",
        "collected_time",
        "ALL_SR",
        "ALL_SPL",
        "ALL_OS",
        "ALL_NE",
        "ALL_Steps",
        "ALL_Episodes",
        "Boundary_SR",
        "Boundary_SPL",
        "Boundary_OS",
        "Boundary_NE",
        "Boundary_Steps",
        "Boundary_Episodes",
        "LandmarkSet_SR",
        "LandmarkSet_SPL",
        "LandmarkSet_OS",
        "LandmarkSet_NE",
        "LandmarkSet_Steps",
        "LandmarkSet_Episodes",
        "Road_SR",
        "Road_SPL",
        "Road_OS",
        "Road_NE",
        "Road_Steps",
        "Road_Episodes",
    ]
    rows = dedupe_rows_by_model_name(rows)
    rows = sorted(rows, key=sort_key)
    rows = [{k: r.get(k, "") for k in fieldnames} for r in rows]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


KNOWN_SPLITS = {"val_seen", "val_unseen", "test"}


def infer_split_from_path(result_path: Path) -> str:
    """Infer eval split from result directory structure.

    Expected layout: results/eval/<arch>/<model>/<split>/<timestamp>/
    The split is the parent directory name of the timestamp folder.
    """
    candidate = result_path.parent.name
    if candidate in KNOWN_SPLITS:
        return candidate
    # Also check grandparent in case result_path points directly to split dir
    candidate2 = result_path.name
    if candidate2 in KNOWN_SPLITS:
        return candidate2
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--result-path", required=True, help="Path to eval result dir that contains evaluation_summary.json")
    parser.add_argument("--output-dir", default="results/eval_collected")
    parser.add_argument("--eval-split", default=None, help="Eval split (val_seen/val_unseen/test). Inferred from result-path if not provided.")
    args = parser.parse_args()

    result_path = Path(args.result_path)
    summary_path = result_path / "evaluation_summary.json"
    if not summary_path.exists():
        print(f"[WARN] evaluation_summary.json not found: {summary_path}")
        return 0

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    eval_split = args.eval_split or infer_split_from_path(result_path)
    if eval_split == "unknown":
        print(f"[WARN] could not determine eval_split from path: {result_path}; writing to root output dir")

    row = build_row(args.model_name, result_path, summary)
    data_version = parse_data_version(args.model_name, str(summary.get("model_path", "")))

    split_dir = Path(args.output_dir) / eval_split if eval_split != "unknown" else Path(args.output_dir)
    csv_path = split_dir / f"eval_results_data{data_version}.csv"

    rows = read_csv_rows(csv_path)
    rows = [r for r in rows if r.get("model_name") != args.model_name]
    rows.append(row)
    write_csv_rows(csv_path, rows)

    print(f"[INFO] collected: {args.model_name} (split={eval_split})")
    print(f"[INFO] csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
