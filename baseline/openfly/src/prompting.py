import os
from typing import Iterable, Optional, Sequence


SYS_PROMPT = (
    "You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language."
)

_ACTION_NAME_BY_ID = {0: "stop", 1: "forward", 2: "left", 3: "right"}
DEFAULT_ACTION_HISTORY_LIMIT = 16


def _resolve_history_limit(limit: Optional[int]) -> int:
    if limit is not None:
        return max(int(limit), 0)
    env_value = os.getenv("OPENFLY_ACTION_HISTORY_LIMIT", "").strip()
    if env_value:
        try:
            return max(int(env_value), 0)
        except ValueError:
            pass
    return DEFAULT_ACTION_HISTORY_LIMIT


def _normalize_action_history(
    action_history: Optional[Sequence[int]],
) -> list[str]:
    if not action_history:
        return []
    names: list[str] = []
    for raw in action_history:
        try:
            aid = int(raw)
        except (TypeError, ValueError):
            continue
        if aid == -1:
            continue
        name = _ACTION_NAME_BY_ID.get(aid)
        if name is None:
            continue
        names.append(name)
    return names


def _format_action_history_clause(
    action_history: Optional[Sequence[int]],
    limit: Optional[int] = None,
) -> str:
    names = _normalize_action_history(action_history)
    if not names:
        return "Past actions: none."
    max_keep = _resolve_history_limit(limit)
    truncated = max_keep > 0 and len(names) > max_keep
    kept = names[-max_keep:] if max_keep > 0 else names
    step_count = len(names)
    joined = ", ".join(kept)
    prefix = "Past actions"
    if truncated:
        prefix = f"Past actions (last {len(kept)} of {step_count})"
    else:
        prefix = f"Past actions ({step_count} so far)"
    return f"{prefix}: {joined}."


def _format_system_prompt(system_prompt: str) -> str:
    return f"<<SYS>\n{system_prompt.strip()}\n<</SYS>>\n\n"


class LLaMa2ChatPromptBuilder:
    """Minimal OpenFly-compatible prompt builder."""

    def __init__(self, system_prompt: Optional[str] = None) -> None:
        self.system_prompt = _format_system_prompt(SYS_PROMPT if system_prompt is None else system_prompt)
        self.prompt = ""
        self.turn_count = 0

    def add_turn(self, role: str, message: str) -> str:
        expected_role = "human" if (self.turn_count % 2 == 0) else "gpt"
        if role != expected_role:
            raise ValueError(f"Unexpected role={role!r} at turn={self.turn_count}, expected={expected_role!r}")

        message = message.replace("<image>", "").strip()
        if self.turn_count == 0:
            wrapped = f"[INST] {self.system_prompt}{message} [/INST] "
        elif role == "human":
            wrapped = f"[INST] {message} [/INST] "
        else:
            wrapped = f"{message if message != '' else ' '}</s>"

        self.prompt += wrapped
        self.turn_count += 1
        return wrapped

    def get_prompt(self) -> str:
        return self.prompt.removeprefix("<s>").rstrip()


def normalize_instruction(text: str) -> str:
    return " ".join(text.replace("\r\n", " ").replace("\n", " ").strip().split()).lower()


def _build_instruction_message(
    instruction: str,
    action_history: Optional[Sequence[int]] = None,
    history_limit: Optional[int] = None,
) -> str:
    base = f"What action should the robot take to {normalize_instruction(instruction)}?"
    clause = _format_action_history_clause(action_history, limit=history_limit)
    return f"{base} {clause}"


def build_openfly_prompt(
    instruction: str,
    action_history: Optional[Sequence[int]] = None,
    history_limit: Optional[int] = None,
) -> str:
    prompt_builder = LLaMa2ChatPromptBuilder()
    prompt_builder.add_turn(
        "human",
        _build_instruction_message(instruction, action_history, history_limit),
    )
    return prompt_builder.get_prompt()


def build_openfly_prompt_with_answer(
    instruction: str,
    answer: str,
    action_history: Optional[Sequence[int]] = None,
    history_limit: Optional[int] = None,
) -> str:
    prompt_builder = LLaMa2ChatPromptBuilder()
    prompt_builder.add_turn(
        "human",
        _build_instruction_message(instruction, action_history, history_limit),
    )
    prompt_builder.add_turn("gpt", answer)
    return prompt_builder.get_prompt()
