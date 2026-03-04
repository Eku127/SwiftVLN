# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Shared constants for SwiftVLN.
"""

from __future__ import annotations

# Standard image token (ms-swift compatible)
DEFAULT_IMAGE_TOKEN = "<image>"

# Special tokens for memory/compression variants
HISTORY_MEMORY_TOKEN = "<history_memory>"
CURRENT_IMAGE_TOKEN = "<current_image>"

# Action vocabulary
DEFAULT_ACTION_MAP = {
    0: "STOP",
    1: "↑",  # MOVE_FORWARD
    2: "←",  # TURN_LEFT
    3: "→",  # TURN_RIGHT
}

# Prompt conjunction templates
DEFAULT_CONJUNCTIONS = [
    "you can see ",
    "in front of you is ",
    "there is ",
    "you can spot ",
    "you are toward the ",
    "ahead of you is ",
    "in your sight is ",
]

# Environment-specific navigation prompt templates
PROMPT_TEMPLATE_HABITAT = (
    "You are an autonomous navigation assistant. Your task is to {instruction}. "
    "Based on your observations, output a sequence of actions using: "
    "↑ (forward 0.25m), ← (turn left), → (turn right), or STOP (when goal is reached). "
    "Output actions directly without explanation."
)

PROMPT_TEMPLATE_SATNAV = (
    "You are an autonomous navigation assistant. Your task is to {instruction}. "
    "Based on your observations, output a sequence of actions using: "
    "↑ (forward 10m), ← (turn left), → (turn right), or STOP (when goal is reached). "
    "Output actions directly without explanation."
)


def get_navigation_prompt_template(env_type: str) -> str:
    """Get environment-specific navigation prompt template."""
    return PROMPT_TEMPLATE_SATNAV if str(env_type).lower() == "satnav" else PROMPT_TEMPLATE_HABITAT


def format_navigation_prompt(instruction: str, env_type: str) -> str:
    """Format navigation prompt with instruction and environment type."""
    return get_navigation_prompt_template(env_type).format(instruction=instruction)
