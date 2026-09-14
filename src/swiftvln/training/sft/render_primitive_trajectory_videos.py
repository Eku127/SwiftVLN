"""Re-render primitive predictions as real SatNav trajectory videos.

``evaluate_primitive_categories`` evaluates each action group with fixed expert
observations.  Its prediction JSONL is sufficient to replay both the source
actions and the predicted actions in SatNav without loading the VLM again.
This utility reconstructs the action-group start state, executes the two
action sequences in separate environments, and saves the usual SatNav
RGB/top-down videos.  The map is a clean local satellite crop: it overlays
only the current action-group gold path in green and prediction path in red.
For turn-only groups, green/red heading arrows make the otherwise stationary
trajectory difference visible. Source scene/trajectory IDs are not necessarily
unique in legacy datasets: replay resolves the collected full instruction and
requires a single matching geometry, never silently selecting the last episode.

It can additionally build one *stitched action-group rollout* per source
episode.  The source action groups are replayed in their original order.  A
group present in the prediction JSONL uses the model's predicted actions;
groups that were filtered out of the primitive dataset use their source
actions as an explicitly labelled oracle bridge.  Segment-level ``STOP``
tokens are delimiters for the next primitive instruction and are not issued to
the simulator, so that a video can continue into the next action group.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ACTION_IDS = {
    "STOP": 0,
    "MOVE_FORWARD": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
}
CATEGORIES = ("boundary", "landmark", "road")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def _parse_segment_dirs(values: Sequence[str]) -> dict[str, Path]:
    directories: dict[str, Path] = {}
    for value in values:
        category, separator, raw_path = value.partition("=")
        if not separator or category not in CATEGORIES or not raw_path:
            raise ValueError("--segment-dir must be CATEGORY=PATH for boundary, landmark, or road")
        if category in directories:
            raise ValueError(f"duplicate --segment-dir for {category}")
        directory = Path(raw_path)
        if not (directory / "segments.jsonl").is_file():
            raise FileNotFoundError(f"missing segments.jsonl in {directory}")
        directories[category] = directory
    return directories


def _episode_key(scene_id: Any, trajectory_id: Any) -> tuple[str, str]:
    return str(scene_id), str(trajectory_id)


def _episode_directory_name(scene_id: Any, trajectory_id: Any) -> str:
    safe_scene = re.sub(r"[^A-Za-z0-9._-]+", "_", str(scene_id))
    safe_trajectory = re.sub(r"[^A-Za-z0-9._-]+", "_", str(trajectory_id))
    return f"{safe_scene}__{safe_trajectory}"


def _index_segments(segment_dirs: Mapping[str, Path]) -> dict[str, list[dict[str, Any]]]:
    return {
        category: _read_jsonl(directory / "segments.jsonl")
        for category, directory in segment_dirs.items()
    }


def _load_predictions(prediction_dir: Path, categories: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for category in categories:
        path = prediction_dir / f"predictions_{category}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing prediction file: {path}")
        rows = _read_jsonl(path)
        for row in rows:
            if row.get("category") != category:
                raise ValueError(f"{path}: row category does not match {category}")
            if not isinstance(row.get("source_segment_index"), int):
                raise ValueError(f"{path}: source_segment_index is required")
            if not isinstance(row.get("gold_actions"), list) or not isinstance(row.get("predicted_actions"), list):
                raise ValueError(f"{path}: action sequences are required")
        result[category] = rows
    return result


def _load_environment(config_path: Path, episodes_path: Path, scenes_dir: Path):
    from omegaconf import OmegaConf
    from satnav.core.env import Env
    from satnav.dataset.satnav_dataset import SatNavDataset

    config = OmegaConf.load(config_path)
    OmegaConf.set_struct(config, False)
    config.DATASET.DATA_PATH = str(episodes_path)
    config.DATASET.SCENES_DIR = str(scenes_dir)
    config.DATASET.SPLIT = "eval"
    dataset = SatNavDataset(config.DATASET)
    episodes = defaultdict(list)
    for episode in dataset.episodes:
        episodes[_episode_key(episode.scene_id, episode.trajectory_id)].append(episode)
    if not episodes:
        raise ValueError(f"no episodes loaded from {episodes_path}")
    return config, episodes, Env


def _episode_geometry(episode: Any) -> str:
    return json.dumps({name: getattr(episode, name, None) for name in
                       ("start_position", "start_rotation", "reference_path", "waypoints")}, sort_keys=True)


def _resolve_episode(episodes: Mapping, segment: Mapping[str, Any]) -> Any:
    """Trajectory IDs may collide; bind replay to the collected instruction."""
    key = _episode_key(segment["scene_id"], segment["trajectory_id"])
    instruction = segment.get("instruction")
    if not isinstance(instruction, str) or not instruction:
        raise ValueError(f"{key}: missing source instruction")
    matches = [ep for ep in episodes.get(key, []) if ep.instruction.instruction_text == instruction]
    if segment.get("episode_id") is not None:
        matches = [ep for ep in matches if str(ep.episode_id) == str(segment["episode_id"])]
    if not matches or len({_episode_geometry(ep) for ep in matches}) != 1:
        raise ValueError(f"{key}: source instruction has missing or ambiguous geometry; refusing replay")
    return matches[0]


def _episode_identity(episode: Any) -> dict[str, Any]:
    return {"source_episode_id": str(episode.episode_id),
            "source_start_position": episode.start_position,
            "source_start_rotation": episode.start_rotation,
            "source_geometry_sha256": hashlib.sha256(_episode_geometry(episode).encode()).hexdigest(),
            "episode_resolution": "exact_source_instruction_and_unique_geometry"}


def _actions_to_ids(actions: Iterable[Any], *, context: str) -> list[int]:
    converted: list[int] = []
    for action in actions:
        if not isinstance(action, str) or action not in ACTION_IDS:
            raise ValueError(f"{context}: unsupported action {action!r}")
        converted.append(ACTION_IDS[action])
    return converted


def _source_segment_actions(segment: Mapping[str, Any], *, context: str) -> list[int]:
    """Return executable source actions, dropping the episode-final STOP token.

    ``segments.jsonl`` normally carries movement/turn actions only, but the
    last source segment of an episode can contain STOP.  STOP terminates a
    SatNav episode, while the episode video needs a final visible pose; it is
    therefore represented in the rollout manifest but not stepped in either
    environment.
    """

    actions = _actions_to_ids(segment.get("actions", []), context=context)
    if ACTION_IDS["STOP"] in actions:
        return actions[: actions.index(ACTION_IDS["STOP"])]
    return actions


def _predicted_segment_actions(row: Mapping[str, Any], *, context: str) -> tuple[list[int], int | None]:
    """Return executable model actions and the first segment-local STOP index.

    Primitive evaluation is teacher-forced after each decoded token, so a row
    can contain extra actions after a model-predicted STOP.  For a concatenated
    rollout those later tokens have no deployment meaning: the next primitive
    instruction would be selected after STOP.  We therefore execute only the
    prefix before the first STOP.
    """

    actions = _actions_to_ids(row["predicted_actions"], context=context)
    try:
        stop_index = actions.index(ACTION_IDS["STOP"])
    except ValueError:
        return actions, None
    return actions[:stop_index], stop_index


def _prefix_actions(segments: Sequence[Mapping[str, Any]], target_index: int) -> list[int]:
    if not 0 <= target_index < len(segments):
        raise ValueError(f"source_segment_index {target_index} is outside segments")
    target = segments[target_index]
    target_key = _episode_key(target.get("scene_id"), target.get("trajectory_id"))
    target_start = target.get("image_indices", [None])[0]
    if not isinstance(target_start, int):
        raise ValueError("target segment has no valid image_indices")
    prefix: list[int] = []
    for index, segment in enumerate(segments):
        if index == target_index:
            break
        if _episode_key(segment.get("scene_id"), segment.get("trajectory_id")) != target_key:
            continue
        if segment.get("instruction") != target.get("instruction"):
            continue
        indices = segment.get("image_indices")
        if not isinstance(indices, list) or not indices or not isinstance(indices[-1], int):
            raise ValueError(f"segment {index}: invalid image_indices")
        if indices[-1] <= target_start:
            prefix.extend(_actions_to_ids(segment.get("actions", []), context=f"segment {index}"))
    return prefix


def _agent_pose(env: Any) -> tuple[list[float], float]:
    """Return the position and north-up clockwise heading in degrees."""

    state = env.agent_state
    return [float(value) for value in state.position.tolist()], float(state.rotation)


def _step(env: Any, action: int, observation: Mapping[str, Any], done: bool):
    if done:
        return observation, done, env.last_step_info
    next_observation, next_done, info = env.step(action)
    return next_observation, next_done, info


def _trajectory_map(
    base_map: Any,
    bounds: Mapping[str, Any],
    *,
    gold_positions: Sequence[Sequence[float]],
    predicted_positions: Sequence[Sequence[float]],
    gold_headings: Sequence[float],
    predicted_headings: Sequence[float],
    step: int,
    gold_action: str,
    predicted_action: str,
) -> Any:
    """Draw only the two action-group paths on a clean local satellite crop."""

    import cv2
    from satnav.utils.maps import draw_path, geo_to_pixel

    map_bgr = cv2.cvtColor(base_map, cv2.COLOR_RGB2BGR)
    # Green is the replayed source action group; red is the model's execution.
    map_shape = map_bgr.shape[:2]
    gold_pixels = [
        geo_to_pixel(position[0], position[1], bounds, map_shape)
        for position in gold_positions
    ]
    predicted_pixels = [
        geo_to_pixel(position[0], position[1], bounds, map_shape)
        for position in predicted_positions
    ]
    draw_path(map_bgr, gold_pixels, color=(0, 220, 0), thickness=5)
    draw_path(map_bgr, predicted_pixels, color=(0, 0, 255), thickness=5)

    def draw_heading(
        pixel: tuple[int, int],
        heading: float,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        """Draw a north-up heading arrow; a zero-degree heading points up."""

        import math

        row, col = pixel
        angle = math.radians(heading)
        arrow_length = 38
        endpoint = (
            int(round(col + arrow_length * math.sin(angle))),
            int(round(row - arrow_length * math.cos(angle))),
        )
        cv2.arrowedLine(map_bgr, (col, row), endpoint, color, thickness, tipLength=0.32)

    # Draw gold as a broad green outline and prediction as a narrower red core.
    # Thus both remain visible when the two poses agree exactly.
    draw_heading(gold_pixels[-1], gold_headings[-1], (0, 220, 0), 8)
    draw_heading(predicted_pixels[-1], predicted_headings[-1], (0, 0, 255), 4)
    start_row, start_col = gold_pixels[0]
    gold_row, gold_col = gold_pixels[-1]
    predicted_row, predicted_col = predicted_pixels[-1]
    cv2.circle(map_bgr, (start_col, start_row), 12, (255, 255, 255), thickness=-1)
    cv2.circle(map_bgr, (gold_col, gold_row), 10, (0, 220, 0), thickness=-1)
    cv2.circle(map_bgr, (predicted_col, predicted_row), 6, (0, 0, 255), thickness=-1)
    legend = (
        ("green: gold", (0, 220, 0)),
        ("red: prediction", (0, 0, 255)),
        (f"step {step + 1}: {gold_action} / {predicted_action}", (255, 255, 255)),
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    text_sizes = [cv2.getTextSize(text, font, font_scale, 1)[0] for text, _ in legend]
    panel_width = min(map_bgr.shape[1], max(width for width, _ in text_sizes) + 20)
    panel_height = 24 * len(legend) + 10
    cv2.rectangle(map_bgr, (0, 0), (panel_width, panel_height), (0, 0, 0), thickness=-1)
    for line, (text, color) in enumerate(legend, start=1):
        cv2.putText(map_bgr, text, (10, 6 + line * 24), font, font_scale, color, 1)
    return cv2.cvtColor(map_bgr, cv2.COLOR_BGR2RGB)


def _render_one(
    *,
    config: Any,
    Env: Any,
    episode: Any,
    prefix: Sequence[int],
    gold_actions: Sequence[int],
    predicted_actions: Sequence[int],
    instruction: str,
    output_path: Path,
) -> None:
    from satnav.utils.maps import make_video

    if len(gold_actions) != len(predicted_actions):
        raise ValueError("gold and predicted action counts must be equal")
    gold_env = Env(config, cycle=False)
    predicted_env = Env(config, cycle=False)
    try:
        gold_obs = gold_env.reset_to_episode(episode)
        predicted_obs = predicted_env.reset_to_episode(episode)
        gold_done = False
        predicted_done = False
        for action in prefix:
            gold_obs, gold_done, _ = _step(gold_env, action, gold_obs, gold_done)
            predicted_obs, predicted_done, _ = _step(predicted_env, action, predicted_obs, predicted_done)
        gold_position, gold_heading = _agent_pose(gold_env)
        predicted_position, predicted_heading = _agent_pose(predicted_env)
        gold_positions = [gold_position]
        predicted_positions = [predicted_position]
        gold_headings = [gold_heading]
        predicted_headings = [predicted_heading]
        rgb_frames: list[Any] = []
        per_step: list[tuple[int, int]] = []
        for step, (gold_action, predicted_action) in enumerate(zip(gold_actions, predicted_actions)):
            gold_obs, gold_done, _ = _step(gold_env, gold_action, gold_obs, gold_done)
            predicted_obs, predicted_done, _ = _step(
                predicted_env, predicted_action, predicted_obs, predicted_done
            )
            gold_position, gold_heading = _agent_pose(gold_env)
            predicted_position, predicted_heading = _agent_pose(predicted_env)
            gold_positions.append(gold_position)
            predicted_positions.append(predicted_position)
            gold_headings.append(gold_heading)
            predicted_headings.append(predicted_heading)
            rgb_frames.append(predicted_obs["rgb"].copy())
            per_step.append((gold_action, predicted_action))

        from satnav.utils.maps import crop_satellite_map

        simulator = predicted_env.simulator
        sat_tif = simulator._satsim._current_scene
        if sat_tif is None:
            raise RuntimeError("SatNav satellite scene is unavailable for trajectory rendering")
        base_map, bounds = crop_satellite_map(
            sat_tif=sat_tif,
            reference_path=[*gold_positions, *predicted_positions],
            goal_position=gold_positions[-1],
            start_position=gold_positions[0],
            padding_meters=30.0,
            max_resolution=1024,
        )
        topdown_frames: list[Any] = []
        for step, (gold_action, predicted_action) in enumerate(per_step):
            topdown_frames.append(
                _trajectory_map(
                    base_map,
                    bounds,
                    gold_positions=gold_positions[: step + 2],
                    predicted_positions=predicted_positions[: step + 2],
                    gold_headings=gold_headings[: step + 2],
                    predicted_headings=predicted_headings[: step + 2],
                    step=step,
                    gold_action=next(name for name, ident in ACTION_IDS.items() if ident == gold_action),
                    predicted_action=next(name for name, ident in ACTION_IDS.items() if ident == predicted_action),
                )
            )
        make_video(
            rgb_frames=rgb_frames,
            topdown_frames=topdown_frames,
            instruction_text=instruction,
            output_path=output_path,
            fps=5,
            frame_width=2048,
            quality=5,
        )
    finally:
        gold_env.close()
        predicted_env.close()


def _action_name(action: int | None) -> str:
    if action is None:
        # OpenCV's Hershey font only handles ASCII reliably.  This state is
        # reached after a model-predicted segment STOP, while gold still has
        # source actions left in the same group.
        return "STOPPED"
    return next(name for name, ident in ACTION_IDS.items() if ident == action)


def _episode_trajectory_map(
    base_map: Any,
    bounds: Mapping[str, Any],
    *,
    gold_positions: Sequence[Sequence[float]],
    predicted_positions: Sequence[Sequence[float]],
    gold_headings: Sequence[float],
    predicted_headings: Sequence[float],
    boundary_markers: Sequence[Mapping[str, Any]],
    step: int,
    total_steps: int,
    source_segment_index: int,
    action_source: str,
    gold_action: int | None,
    predicted_action: int | None,
) -> Any:
    """Draw the accumulated source and stitched-prediction episode routes."""

    import math

    import cv2
    from satnav.utils.maps import draw_path, geo_to_pixel

    map_bgr = cv2.cvtColor(base_map, cv2.COLOR_RGB2BGR)
    map_shape = map_bgr.shape[:2]
    gold_pixels = [
        geo_to_pixel(position[0], position[1], bounds, map_shape)
        for position in gold_positions
    ]
    predicted_pixels = [
        geo_to_pixel(position[0], position[1], bounds, map_shape)
        for position in predicted_positions
    ]
    draw_path(map_bgr, gold_pixels, color=(0, 220, 0), thickness=5)
    draw_path(map_bgr, predicted_pixels, color=(0, 0, 255), thickness=5)

    # Small paired rings show the end of every source action group that has
    # already completed.  Their number/order is recorded in the JSON manifest.
    for marker in boundary_markers:
        position_index = int(marker["position_index"])
        if position_index > step + 1:
            continue
        gold_row, gold_col = gold_pixels[position_index]
        predicted_row, predicted_col = predicted_pixels[position_index]
        cv2.circle(map_bgr, (gold_col, gold_row), 7, (0, 220, 0), thickness=2)
        cv2.circle(map_bgr, (predicted_col, predicted_row), 5, (0, 0, 255), thickness=2)

    def draw_heading(
        pixel: tuple[int, int], heading: float, color: tuple[int, int, int], thickness: int
    ) -> None:
        row, col = pixel
        angle = math.radians(heading)
        endpoint = (
            int(round(col + 38 * math.sin(angle))),
            int(round(row - 38 * math.cos(angle))),
        )
        cv2.arrowedLine(map_bgr, (col, row), endpoint, color, thickness, tipLength=0.32)

    # Green outline plus red core keeps both headings visible when they agree.
    draw_heading(gold_pixels[-1], gold_headings[-1], (0, 220, 0), 8)
    draw_heading(predicted_pixels[-1], predicted_headings[-1], (0, 0, 255), 4)
    start_row, start_col = gold_pixels[0]
    gold_row, gold_col = gold_pixels[-1]
    predicted_row, predicted_col = predicted_pixels[-1]
    cv2.circle(map_bgr, (start_col, start_row), 12, (255, 255, 255), thickness=-1)
    cv2.circle(map_bgr, (gold_col, gold_row), 10, (0, 220, 0), thickness=-1)
    cv2.circle(map_bgr, (predicted_col, predicted_row), 6, (0, 0, 255), thickness=-1)

    action_source_label = "model prediction" if action_source == "model_prediction" else "gold bridge"
    legend = (
        ("green: full source route", (0, 220, 0)),
        ("red: stitched predicted route", (0, 0, 255)),
        (f"segment {source_segment_index} ({action_source_label})", (255, 255, 0)),
        (
            f"step {step + 1}/{total_steps}: {_action_name(gold_action)} / {_action_name(predicted_action)}",
            (255, 255, 255),
        ),
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    text_sizes = [cv2.getTextSize(text, font, font_scale, 1)[0] for text, _ in legend]
    panel_width = min(map_bgr.shape[1], max(width for width, _ in text_sizes) + 20)
    panel_height = 24 * len(legend) + 10
    cv2.rectangle(map_bgr, (0, 0), (panel_width, panel_height), (0, 0, 0), thickness=-1)
    for line, (text, color) in enumerate(legend, start=1):
        cv2.putText(map_bgr, text, (10, 6 + line * 24), font, font_scale, color, 1)
    return cv2.cvtColor(map_bgr, cv2.COLOR_BGR2RGB)


def _render_episode_rollout(
    *,
    config: Any,
    Env: Any,
    episode: Any,
    category: str,
    source_segments: Sequence[tuple[int, Mapping[str, Any]]],
    prediction_rows: Mapping[int, Mapping[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    """Render one continuous route by stitching its source action groups.

    This intentionally does not re-run the VLM on observations after a model
    deviation.  Its prediction rows were generated on each segment's expert
    start observation.  The output is therefore an accumulated action-trace
    comparison, not a fresh closed-loop policy evaluation.
    """

    from satnav.utils.maps import crop_satellite_map, make_video

    if not source_segments:
        raise ValueError("cannot render an episode without source action segments")
    gold_env = Env(config, cycle=False)
    predicted_env = Env(config, cycle=False)
    try:
        gold_obs = gold_env.reset_to_episode(episode)
        predicted_obs = predicted_env.reset_to_episode(episode)
        gold_done = False
        predicted_done = False
        gold_position, gold_heading = _agent_pose(gold_env)
        predicted_position, predicted_heading = _agent_pose(predicted_env)
        gold_positions = [gold_position]
        predicted_positions = [predicted_position]
        gold_headings = [gold_heading]
        predicted_headings = [predicted_heading]
        rgb_frames: list[Any] = []
        step_records: list[dict[str, Any]] = []
        boundary_markers: list[dict[str, Any]] = []
        segment_records: list[dict[str, Any]] = []

        for source_index, segment in source_segments:
            gold_actions = _source_segment_actions(segment, context=f"segment {source_index}")
            row = prediction_rows.get(source_index)
            if row is None:
                predicted_actions = list(gold_actions)
                predicted_stop_index = None
                action_source = "gold_bridge"
            else:
                predicted_actions, predicted_stop_index = _predicted_segment_actions(
                    row, context=f"prediction segment {source_index}"
                )
                action_source = "model_prediction"

            first_video_step = len(step_records)
            for local_step in range(max(len(gold_actions), len(predicted_actions))):
                gold_action = gold_actions[local_step] if local_step < len(gold_actions) else None
                predicted_action = (
                    predicted_actions[local_step] if local_step < len(predicted_actions) else None
                )
                if gold_action is not None:
                    gold_obs, gold_done, _ = _step(gold_env, gold_action, gold_obs, gold_done)
                if predicted_action is not None:
                    predicted_obs, predicted_done, _ = _step(
                        predicted_env, predicted_action, predicted_obs, predicted_done
                    )
                gold_position, gold_heading = _agent_pose(gold_env)
                predicted_position, predicted_heading = _agent_pose(predicted_env)
                gold_positions.append(gold_position)
                predicted_positions.append(predicted_position)
                gold_headings.append(gold_heading)
                predicted_headings.append(predicted_heading)
                rgb_frames.append(predicted_obs["rgb"].copy())
                step_records.append(
                    {
                        "source_segment_index": source_index,
                        "action_source": action_source,
                        "gold_action": gold_action,
                        "predicted_action": predicted_action,
                    }
                )

            boundary_markers.append(
                {
                    "source_segment_index": source_index,
                    "position_index": len(gold_positions) - 1,
                    "action_source": action_source,
                }
            )
            segment_records.append(
                {
                    "source_segment_index": source_index,
                    "primitive_instruction": row.get("instruction") if row is not None else None,
                    "action_source": action_source,
                    "source_actions": [_action_name(action) for action in gold_actions],
                    "predicted_actions_raw": row.get("predicted_actions") if row is not None else None,
                    "predicted_actions_executed": [_action_name(action) for action in predicted_actions],
                    "first_predicted_stop_index": predicted_stop_index,
                    "source_terminal_stop_omitted": ACTION_IDS["STOP"]
                    in _actions_to_ids(segment.get("actions", []), context=f"segment {source_index}"),
                    "first_video_step": first_video_step,
                    "last_video_step": len(step_records) - 1,
                }
            )

        if not rgb_frames:
            raise ValueError("source episode contains no executable actions")
        simulator = gold_env.simulator
        sat_tif = simulator._satsim._current_scene
        if sat_tif is None:
            raise RuntimeError("SatNav satellite scene is unavailable for trajectory rendering")
        base_map, bounds = crop_satellite_map(
            sat_tif=sat_tif,
            reference_path=[*gold_positions, *predicted_positions],
            goal_position=gold_positions[-1],
            start_position=gold_positions[0],
            padding_meters=30.0,
            max_resolution=1024,
        )
        topdown_frames = [
            _episode_trajectory_map(
                base_map,
                bounds,
                gold_positions=gold_positions[: step + 2],
                predicted_positions=predicted_positions[: step + 2],
                gold_headings=gold_headings[: step + 2],
                predicted_headings=predicted_headings[: step + 2],
                boundary_markers=boundary_markers,
                step=step,
                total_steps=len(step_records),
                source_segment_index=int(step_records[step]["source_segment_index"]),
                action_source=str(step_records[step]["action_source"]),
                gold_action=step_records[step]["gold_action"],
                predicted_action=step_records[step]["predicted_action"],
            )
            for step in range(len(step_records))
        ]
        source_instruction = str(source_segments[0][1].get("instruction", ""))
        make_video(
            rgb_frames=rgb_frames,
            topdown_frames=topdown_frames,
            instruction_text=(
                f"Full episode instruction: {source_instruction}\n"
                "Green: source trajectory. Red: stored predictions replayed (not closed-loop); missing groups use gold bridges."
            ),
            output_path=output_path,
            fps=5,
            frame_width=2048,
            quality=5,
        )
        model_segment_count = sum(record["action_source"] == "model_prediction" for record in segment_records)
        return {
            "category": category,
            **_episode_identity(episode),
            "gold_positions": gold_positions,
            "predicted_positions": predicted_positions,
            "gold_headings": gold_headings,
            "predicted_headings": predicted_headings,
            "source_scene_id": str(source_segments[0][1]["scene_id"]),
            "source_trajectory_id": str(source_segments[0][1]["trajectory_id"]),
            "source_full_instruction": source_instruction,
            "video": str(output_path.resolve()),
            "source_segment_count": len(segment_records),
            "model_predicted_segment_count": model_segment_count,
            "gold_bridge_segment_count": len(segment_records) - model_segment_count,
            "video_step_count": len(step_records),
            "source_action_count": sum(len(record["source_actions"]) for record in segment_records),
            "predicted_executed_action_count": sum(
                len(record["predicted_actions_executed"]) for record in segment_records
            ),
            "action_segments": segment_records,
            "interpretation": (
                "Stitched action-trace comparison. Model-predicted action groups use their stored "
                "expert-start predictions; source groups without a primitive prediction are replayed "
                "as gold bridges. Segment-level STOP is not executed because it delimitates the next group."
            ),
        }
    finally:
        gold_env.close()
        predicted_env.close()


def _source_segments_for_episode(
    segments: Sequence[Mapping[str, Any]], episode_key: tuple[str, str]
) -> list[tuple[int, Mapping[str, Any]]]:
    return [
        (index, segment)
        for index, segment in enumerate(segments)
        if _episode_key(segment.get("scene_id"), segment.get("trajectory_id")) == episode_key
    ]


def render(args: argparse.Namespace) -> None:
    if args.episode_rollouts_only and not args.render_episode_rollouts:
        raise ValueError("--episode-rollouts-only requires --render-episode-rollouts")
    segment_dirs = _parse_segment_dirs(args.segment_dir)
    missing = set(args.categories).difference(segment_dirs)
    if missing:
        raise ValueError(f"missing source segment directories for: {sorted(missing)}")
    segments = _index_segments(segment_dirs)
    predictions = _load_predictions(args.prediction_dir, args.categories)
    for category, rows in predictions.items():
        for row in rows:
            index = row["source_segment_index"]
            if not 0 <= index < len(segments[category]):
                raise ValueError(f"{category}: source segment index {index} out of bounds")
            source_text = row.get("source_full_instruction")
            if source_text is not None and source_text != segments[category][index].get("instruction"):
                raise ValueError(f"{category} segment {index}: prediction/source instruction mismatch")
    config, episodes, Env = _load_environment(args.config, args.episodes_path, args.scenes_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.episode_rollouts_only:
        completed = 0
        episode_videos: dict[tuple[str, str, str], dict[str, Any]] = {}
        for category in args.categories:
            rows = predictions[category]
            limit = min(args.max_videos_per_category, len(rows))
            for ordinal, row in enumerate(rows[:limit], start=1):
                source_index = int(row["source_segment_index"])
                segment = segments[category][source_index]
                episode_key = _episode_key(segment["scene_id"], segment["trajectory_id"])
                episode = _resolve_episode(episodes, segment)
                output_directory = args.output_dir / category
                if args.group_by_episode:
                    output_directory = output_directory / _episode_directory_name(*episode_key)
                output_path = output_directory / f"{int(row['sample_id']):06d}.mp4"
                manifest_key = (category, *episode_key)
                episode_record = episode_videos.setdefault(
                    manifest_key,
                    {
                        "category": category,
                        **_episode_identity(episode),
                        "source_scene_id": episode_key[0],
                        "source_trajectory_id": episode_key[1],
                        "source_full_instruction": row.get("source_full_instruction"),
                        "action_segments": [],
                    },
                )
                episode_record["action_segments"].append(
                    {
                        "sample_id": int(row["sample_id"]),
                        "source_segment_index": source_index,
                        "primitive_instruction": row.get("instruction"),
                        "gold_actions": row["gold_actions"],
                        "predicted_actions": row["predicted_actions"],
                        "diagnostic_video": row.get("diagnostic_video"),
                        "trajectory_video": str(output_path.resolve()),
                    }
                )
                if output_path.exists() and not args.overwrite:
                    print(f"[trajectory video] {category} {ordinal}/{limit}: exists, skipping", flush=True)
                    continue
                output_path.parent.mkdir(parents=True, exist_ok=True)
                _render_one(
                    config=config,
                    Env=Env,
                    episode=episode,
                    prefix=_prefix_actions(segments[category], source_index),
                    gold_actions=_actions_to_ids(row["gold_actions"], context="gold_actions"),
                    predicted_actions=_actions_to_ids(row["predicted_actions"], context="predicted_actions"),
                    instruction=str(row.get("instruction", "")),
                    output_path=output_path,
                )
                completed += 1
                print(f"[trajectory video] {category} {ordinal}/{limit}: saved", flush=True)
        serialised_episodes: dict[str, list[dict[str, Any]]] = {}
        for record in episode_videos.values():
            record["action_segments"].sort(
                key=lambda item: (item["source_segment_index"], item["sample_id"])
            )
            serialised_episodes.setdefault(record["category"], []).append(record)
        for records in serialised_episodes.values():
            records.sort(key=lambda item: (item["source_scene_id"], item["source_trajectory_id"]))
        (args.output_dir / "episode_video_manifest.json").write_text(
            json.dumps(serialised_episodes, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Action-group trajectory videos rendered: {completed}", flush=True)

    if not args.render_episode_rollouts:
        return

    rollout_directory = args.output_dir / "episode_rollouts"
    rollout_manifest: dict[str, list[dict[str, Any]]] = {}
    rollout_completed = 0
    for category in args.categories:
        rows_by_episode: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
        for row in predictions[category]:
            source_index = int(row["source_segment_index"])
            segment = segments[category][source_index]
            episode_key = _episode_key(segment["scene_id"], segment["trajectory_id"])
            if source_index in rows_by_episode[episode_key]:
                raise ValueError(
                    f"duplicate prediction for {category} source_segment_index {source_index}"
                )
            rows_by_episode[episode_key][source_index] = row

        records: list[dict[str, Any]] = []
        for ordinal, (episode_key, rows_by_source_index) in enumerate(
            sorted(rows_by_episode.items()), start=1
        ):
            source_segments = _source_segments_for_episode(segments[category], episode_key)
            if len({segment.get("instruction") for _, segment in source_segments}) != 1:
                raise ValueError(f"{episode_key}: source segments mix different instructions")
            episode = _resolve_episode(episodes, source_segments[0][1])
            output_path = (
                rollout_directory
                / category
                / f"{_episode_directory_name(*episode_key)}.mp4"
            )
            if output_path.exists() and not args.overwrite:
                print(
                    f"[episode rollout] {category} {ordinal}/{len(rows_by_episode)}: exists, skipping",
                    flush=True,
                )
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            record = _render_episode_rollout(
                config=config,
                Env=Env,
                episode=episode,
                category=category,
                source_segments=source_segments,
                prediction_rows=rows_by_source_index,
                output_path=output_path,
            )
            records.append(record)
            rollout_completed += 1
            print(
                f"[episode rollout] {category} {ordinal}/{len(rows_by_episode)}: saved",
                flush=True,
            )
        rollout_manifest[category] = records
    rollout_directory.mkdir(parents=True, exist_ok=True)
    (rollout_directory / "episode_rollout_manifest.json").write_text(
        json.dumps(rollout_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Complete episode rollout videos rendered: {rollout_completed}", flush=True)


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render real gold/predicted SatNav trajectories from primitive predictions.")
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-path", type=Path, required=True)
    parser.add_argument("--scenes-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path,
        default=Path("/mnt/data1/home/jiangjiajun/workspace/SatNav/applications/episode_processing/configs/trajectory_generation.yaml"),
    )
    parser.add_argument("--segment-dir", action="append", required=True, metavar="CATEGORY=PATH")
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument("--max-videos-per-category", type=_positive_int, default=50)
    parser.add_argument(
        "--group-by-episode",
        action="store_true",
        help="Store trajectory videos below category/scene_id__trajectory_id/.",
    )
    parser.add_argument(
        "--render-episode-rollouts",
        action="store_true",
        help=(
            "Also create one stitched full-episode rollout video per source episode below "
            "episode_rollouts/. Source groups without primitive predictions are explicitly replayed "
            "as gold bridges."
        ),
    )
    parser.add_argument(
        "--episode-rollouts-only",
        action="store_true",
        help="Skip existing per-action-group videos and render only full-episode rollout videos.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        render(args)
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"Trajectory video rendering failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
