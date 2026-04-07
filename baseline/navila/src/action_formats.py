import os
import re


DEFAULT_ACTION_FORMAT = "compact"
ACTION_TO_NAME = {
    0: "stop",
    1: "forward",
    2: "left",
    3: "right",
}
ACTION_TO_TEXT_BY_FORMAT = {
    "sentence": {
        0: "The next action is stop.",
        1: "The next action is move forward 10 meters.",
        2: "The next action is turn left 15 degree.",
        3: "The next action is turn right 15 degree.",
    },
    "compact": ACTION_TO_NAME.copy(),
}
ACTION_PATTERNS = {
    0: re.compile(r"\bstop\b", re.IGNORECASE),
    1: re.compile(r"\b(?:move forward|forward)\b", re.IGNORECASE),
    2: re.compile(r"\b(?:turn left|left)\b", re.IGNORECASE),
    3: re.compile(r"\b(?:turn right|right)\b", re.IGNORECASE),
}

PROMPT_TEMPLATE_SENTENCE = (
    "Imagine you are a robot programmed for navigation tasks. You have been given a video "
    'of historical observations {history_tokens}, and current observation <image>\n. Your assigned task is: "{instruction}" '
    "Analyze this series of images to decide your next action, which could be turning left or right by a specific "
    "degree, moving forward a certain distance, or stop if the task is completed."
)
PROMPT_TEMPLATE_COMPACT = (
    "Imagine you are a robot programmed for navigation tasks. You have been given a video "
    'of historical observations {history_tokens}, and current observation <image>\n. Your assigned task is: "{instruction}" '
    "Analyze this series of images and reply with exactly one word for the next action: "
    "stop, forward, left, or right."
)


def normalize_action_format(value: str | None = None) -> str:
    raw = value
    if raw is None:
        raw = os.getenv("SATNAV_ACTION_FORMAT", DEFAULT_ACTION_FORMAT)
    normalized = str(raw).strip().lower().replace("-", "_")
    alias_map = {
        "": DEFAULT_ACTION_FORMAT,
        "default": DEFAULT_ACTION_FORMAT,
        "natural": "sentence",
        "legacy": "sentence",
        "sentence": "sentence",
        "compact": "compact",
        "token": "compact",
        "word": "compact",
    }
    if normalized not in alias_map:
        choices = ", ".join(sorted(alias_map))
        raise ValueError(f"Unsupported SATNAV action format: {raw!r}. Expected one of: {choices}")
    return alias_map[normalized]


def get_action_to_text(action_format: str | None = None) -> dict[int, str]:
    return ACTION_TO_TEXT_BY_FORMAT[normalize_action_format(action_format)]


def build_prompt(instruction: str, history_count: int, action_format: str | None = None) -> str:
    normalized = normalize_action_format(action_format)
    history_tokens = "<image>\n" * max(history_count, 0)
    template = PROMPT_TEMPLATE_COMPACT if normalized == "compact" else PROMPT_TEMPLATE_SENTENCE
    return template.format(
        history_tokens=history_tokens,
        instruction=instruction,
    )


def parse_action_text(text: str) -> int | None:
    for action, pattern in ACTION_PATTERNS.items():
        if pattern.search(text):
            return action
    return None
