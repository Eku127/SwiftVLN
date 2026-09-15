"""Dependency-free static semantics for supported navigation backends."""

from __future__ import annotations

from dataclasses import dataclass


NAVIGATION_PROMPT_STYLES = ("standard", "primitive")


PRIMITIVE_SATNAV_PROMPT_TEMPLATE = """You are an autonomous navigation assistant. Your task is to {instruction}.

Based on your current visual observation and historical observations, output exactly ONE action using:
↑ (forward 10m), ← (turn left), → (turn right), or STOP (when the current task is completed).

Treat the center of the current image as your current heading direction.
Use the position and direction of the relevant route, reference, or target relative to the image center to determine whether your heading needs adjustment.
Pay particular attention to whether the intended navigation direction is aligned with the central region of the current image.
If the intended direction is approximately aligned with the image center, avoid unnecessary corrective turns.

Use historical observations to understand navigation progress and how the scene has changed over time, but use the current image as the primary evidence for selecting the current action.

For a Turn instruction, rotate in the specified direction. If no target position is explicitly specified, treat the center of the current image as the desired target position.

Output exactly ONE action from ↑, ←, →, or STOP.
Output the action directly without explanation."""


@dataclass(frozen=True)
class EnvironmentSpec:
    """Static backend semantics safe to use without importing a simulator."""

    name: str
    forward_step_m: float
    turn_angle_deg: float
    prompt_template: str
    config_argument: str
    supports_map_memory: bool = False
    reports_trajectory_types: bool = False
    action_symbols: tuple[str, ...] = ("STOP", "↑", "←", "→")

    @property
    def action_map(self) -> dict[int, str]:
        return dict(enumerate(self.action_symbols))

    @property
    def forward_distance_label(self) -> str:
        return f"{self.forward_step_m:g}m"

    def format_prompt(
        self,
        instruction: str,
        navigation_prompt_style: str = "standard",
    ) -> str:
        """Format one navigation prompt using a registered prompt style."""
        style = str(navigation_prompt_style).strip().lower()
        if style not in NAVIGATION_PROMPT_STYLES:
            raise ValueError(
                "navigation_prompt_style must be one of "
                f"{NAVIGATION_PROMPT_STYLES}, got {navigation_prompt_style!r}"
            )
        if style == "primitive":
            if self.name != "satnav":
                raise ValueError(
                    "navigation_prompt_style='primitive' supports only satnav"
                )
            template = PRIMITIVE_SATNAV_PROMPT_TEMPLATE
        else:
            template = self.prompt_template
        formatted_instruction = str(instruction).strip()
        if style == "primitive":
            formatted_instruction = formatted_instruction.rstrip(".")
        return template.format(instruction=formatted_instruction)


HABITAT_SPEC = EnvironmentSpec(
    name="habitat",
    forward_step_m=0.25,
    turn_angle_deg=15.0,
    config_argument="habitat_config_path",
    prompt_template=(
        "You are an autonomous navigation assistant. Your task is to {instruction}. "
        "Based on your observations, output a sequence of actions using: "
        "↑ (forward 0.25m), ← (turn left), → (turn right), or STOP (when goal is reached). "
        "Output actions directly without explanation."
    ),
)

SATNAV_SPEC = EnvironmentSpec(
    name="satnav",
    forward_step_m=10.0,
    turn_angle_deg=15.0,
    config_argument="satnav_config",
    supports_map_memory=True,
    reports_trajectory_types=True,
    prompt_template=(
        "You are an autonomous navigation assistant. Your task is to {instruction}. "
        "Based on your observations, output a sequence of actions using: "
        "↑ (forward 10m), ← (turn left), → (turn right), or STOP (when goal is reached). "
        "Output actions directly without explanation."
    ),
)

_ENVIRONMENT_SPECS = {
    HABITAT_SPEC.name: HABITAT_SPEC,
    SATNAV_SPEC.name: SATNAV_SPEC,
}


def get_environment_spec(name: str) -> EnvironmentSpec:
    normalized = str(name).strip().lower()
    try:
        return _ENVIRONMENT_SPECS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_ENVIRONMENT_SPECS))
        raise ValueError(
            f"Unknown env_type: {name}. Supported backends: {supported}."
        ) from exc
