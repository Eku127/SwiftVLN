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
