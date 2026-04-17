import re
from typing import Optional

import numpy as np


COMPACT = "compact"
ORIGINAL = "original"
AUTO = "auto"
SUPPORTED_ACTION_FORMATS = {COMPACT, ORIGINAL}

ACTION_TO_NAME = {
    0: "stop",
    1: "forward",
    2: "left",
    3: "right",
}

NAME_TO_ACTION = {name: action for action, name in ACTION_TO_NAME.items()}

ACTION_PATTERNS = {
    0: re.compile(r"\bstop\b", re.IGNORECASE),
    1: re.compile(r"\bforward\b", re.IGNORECASE),
    2: re.compile(r"\bleft\b", re.IGNORECASE),
    3: re.compile(r"\bright\b", re.IGNORECASE),
}

ORIGINAL_ACTION_DIM = 8
ORIGINAL_UNNORM_KEY = "satnav_original"
ORIGINAL_ACTIVE_MASK = np.array([True, True, True, True, False, False, False, False], dtype=bool)
ORIGINAL_Q01 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
ORIGINAL_Q99 = np.array([1.0, 10.0, 15.0, 15.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# SatNav-adapted original OpenFly action templates:
# - forward is fixed to 10 meters
# - left / right are fixed to 15 degrees
ORIGINAL_ACTION_VECTORS = {
    0: np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    1: np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    2: np.array([0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    3: np.array([0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
}


def resolve_action_format(requested: str, fallback: Optional[str] = None) -> str:
    if requested == AUTO:
        requested = fallback or COMPACT
    if requested not in SUPPORTED_ACTION_FORMATS:
        raise ValueError(f"Unsupported OpenFly action format: {requested}")
    return requested


def action_to_text(action: int) -> str:
    if action not in ACTION_TO_NAME:
        raise ValueError(f"Unsupported SatNav action: {action}")
    return ACTION_TO_NAME[action]


def parse_action_text(text: str) -> int | None:
    for action, pattern in ACTION_PATTERNS.items():
        if pattern.search(text):
            return action
    return None


def get_original_action_vector(action: int) -> np.ndarray:
    if action not in ORIGINAL_ACTION_VECTORS:
        raise ValueError(f"Unsupported SatNav action for original OpenFly format: {action}")
    return ORIGINAL_ACTION_VECTORS[action].copy()


def normalize_original_action(raw_action: np.ndarray) -> np.ndarray:
    raw_action = np.asarray(raw_action, dtype=np.float32)
    normalized = np.zeros_like(raw_action, dtype=np.float32)
    active = ORIGINAL_ACTIVE_MASK
    normalized[active] = 2.0 * (raw_action[active] - ORIGINAL_Q01[active]) / (ORIGINAL_Q99[active] - ORIGINAL_Q01[active]) - 1.0
    return normalized


def get_normalized_original_action_vector(action: int) -> np.ndarray:
    return normalize_original_action(get_original_action_vector(action))


def get_original_norm_stats(unnorm_key: str = ORIGINAL_UNNORM_KEY, num_transitions: Optional[int] = None) -> dict:
    stats = {
        unnorm_key: {
            "action": {
                "q01": ORIGINAL_Q01.tolist(),
                "q99": ORIGINAL_Q99.tolist(),
                "mask": ORIGINAL_ACTIVE_MASK.astype(bool).tolist(),
            }
        }
    }
    if num_transitions is not None:
        stats[unnorm_key]["num_transitions"] = int(num_transitions)
    return stats


def get_original_action_templates() -> dict[str, list[float]]:
    return {str(action): vector.tolist() for action, vector in ORIGINAL_ACTION_VECTORS.items()}


def convert_original_action_vector_to_action(action_vector: np.ndarray) -> tuple[int, np.ndarray, float]:
    raw = np.asarray(action_vector, dtype=np.float32)
    rounded = np.round(raw).astype(np.int32)

    for action, template in ORIGINAL_ACTION_VECTORS.items():
        if np.array_equal(rounded, template.astype(np.int32)):
            return action, rounded, 0.0

    active_idx = np.where(ORIGINAL_ACTIVE_MASK)[0]
    nearest_action = 0
    nearest_distance = float("inf")
    for action, template in ORIGINAL_ACTION_VECTORS.items():
        distance = float(np.linalg.norm(raw[active_idx] - template[active_idx]))
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_action = action

    return nearest_action, rounded, nearest_distance
