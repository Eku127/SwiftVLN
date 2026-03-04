#!/usr/bin/env python3
"""Collect eval summaries into versioned CSV files.

Output CSV naming:
  results/eval_collected/eval_results_data<version>.csv
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
    for t in ("overlapvln", "streamvln", "compressvln", "monovln", "uninavid"):
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
        if "-sgtc-k" in model_name:
            return "baseline + sgtc"
        if "-gtc-k" in model_name:
            return "baseline + gtc"
        if "-qa" in model_name:
            m = re.search(r"-qa(\d+)-", model_name)
            return f"baseline + qa{m.group(1)}" if m else "baseline + qa"
        if "-initial-" in model_name:
            return "baseline + initial"
        if "-tome-" in model_name:
            return "baseline + tome"
        return "baseline"
    if model_type == "streamvln":
        return "stream baseline"
    if model_type == "compressvln":
        if "-stride" in model_name:
            m = re.search(r"-stride(\d+)-", model_name)
            return f"compress stride{m.group(1)}" if m else "compress baseline"
        return "compress baseline"
    return f"{model_type} run"


def plan_rank(plan: str) -> int:
    order = {
        "baseline": 10,
        "baseline + initial": 20,
        "baseline + tome": 30,
        "baseline + qa15": 40,
        "baseline + qa30": 50,
        "baseline + gtc": 60,
        "baseline + sgtc": 70,
        "stream baseline": 80,
        "compress baseline": 90,
    }
    return order.get(plan, 999)


def strip_run_timestamp(model_name: str) -> str:
    # e.g. ...-20260214-143037
    return re.sub(r"-\d{8}-\d{6}$", "", model_name)


def overlap_variant_rank(model_name: str) -> int:
    # Keep baseline first inside the same setting group.
    if "-sgtc-k" in model_name:
        return 60
    if "-gtc-k" in model_name:
        return 50
    if re.search(r"-qa\d*-", model_name):
        return 40
    if "-initial-" in model_name:
        return 30
    if "-tome-" in model_name:
        return 20
    return 10


def normalize_overlap_setting_key(model_name: str) -> str:
    key = strip_run_timestamp(model_name)
    # Remove experiment variant markers so same setting stays grouped.
    key = re.sub(r"-sgtc-k\d+", "", key)
    key = re.sub(r"-gtc-k\d+", "", key)
    key = re.sub(r"-qa\d*", "", key)
    key = key.replace("-initial", "")
    key = key.replace("-tome", "")
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


def build_row(model_name: str, result_path: Path, summary: Dict[str, Any]) -> Dict[str, str]:
    model_type = parse_model_type(model_name)
    plan = infer_plan(model_name, model_type)

    boundary = pick_type_block(summary, "Boundary")
    landmark = pick_type_block(summary, "LandmarkSet")
    road = pick_type_block(summary, "Road")

    row = {
        "model_name": model_name,
        "model_type": model_type,
        "plan": plan,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--result-path", required=True, help="Path to eval result dir that contains evaluation_summary.json")
    parser.add_argument("--output-dir", default="results/eval_collected")
    args = parser.parse_args()

    result_path = Path(args.result_path)
    summary_path = result_path / "evaluation_summary.json"
    if not summary_path.exists():
        print(f"[WARN] evaluation_summary.json not found: {summary_path}")
        return 0

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    row = build_row(args.model_name, result_path, summary)
    data_version = parse_data_version(args.model_name, str(summary.get("model_path", "")))
    csv_path = Path(args.output_dir) / f"eval_results_data{data_version}.csv"

    rows = read_csv_rows(csv_path)
    rows = [r for r in rows if r.get("model_name") != args.model_name]
    rows.append(row)
    write_csv_rows(csv_path, rows)

    print(f"[INFO] collected: {args.model_name}")
    print(f"[INFO] csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
