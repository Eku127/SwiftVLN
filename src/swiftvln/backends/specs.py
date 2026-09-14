"""Dependency-free static semantics for supported navigation backends."""

from __future__ import annotations

from dataclasses import dataclass


NAVIGATION_PROMPT_STYLES = ("standard", "primitive")


PRIMITIVE_SATNAV_PROMPT_TEMPLATE = """You are an autonomous navigation assistant. Your task is to {instruction}.

At each step, select exactly ONE action based on the navigation instruction, your historical observations, and the current visual observation.

Available actions:
↑ : move forward 10m
← : turn left
→ : turn right
STOP : stop when the current navigation instruction has been completed

The navigation instruction follows one of these primitive forms:

1. Move toward the [ordinal] <target> [in your <position>].
2. Move along the <reference> until you see the [ordinal] <target> [in your <position>].
3. Turn <left/right> until the <target> [in your <position>].

For augmented MOVE_ALONG samples, the instruction may instead be:

4. Move along the <reference> until reaching the [ordinal] <target>.

The ordinal and position terms may be absent.

Interpret and execute the primitives according to the following rules:

MOVE_TOWARD:

- Treat the <target> as the object that guides the movement.
- Move toward the specified target while continuously adjusting the heading according to its visual location.
- If an ordinal is provided, use it to identify the intended instance among multiple objects of the same type.
- If a position is provided, use it to identify the intended target and its spatial relationship to the agent.
- Do not confuse another visually similar object with the specified target.

MOVE_ALONG:

- Treat the <reference> as the route or structure that continuously guides movement.
- Stay aligned with and follow the <reference>, using left or right turns when necessary to correct the heading.
- The <target> defines the completion condition rather than the direction of travel.
- For "until you see" instructions, continue following the <reference> until the specified target is visually observed.
- If an ordinal is provided, count or distinguish occurrences in navigation order and do not stop at an earlier matching target.
- If a position is provided, the target must satisfy the specified visual position as described by the instruction.
- For augmented "until reaching" instructions, continue following the <reference> until the specified target has been reached rather than merely observed.
- Do not leave the <reference> simply to move toward the termination target.

TURN:

- Rotate only in the direction explicitly specified by the instruction.
- Use the changing visual location of the <target> to determine whether further rotation is required.
- Do not move forward while executing a TURN primitive.
- If a position is explicitly provided, stop turning when the target reaches that specified position.
- If no position is written in a TURN instruction, interpret the required position as center.
- Do not reverse the prescribed turning direction merely because the target appears on the opposite side of the image.

GENERAL EXECUTION RULES:

- Use historical observations to understand navigation progress and how the route, target locations, and viewpoint have changed over time.
- Historical observations are ordered from earlier observations to more recent observations.
- Use the current observation as the primary evidence for selecting the action at the current step.
- Compare historical and current observations to determine whether the current action is making progress or causing deviation.
- Follow the current primitive according to its semantics rather than simply matching words such as "left", "right", or "forward" to actions.
- Do not repeat a navigation stage or correction that has already been completed.
- When an ordinal is specified, make sure the correct occurrence is used before completing the instruction.
- Output STOP only when the completion condition of the current primitive has been satisfied.
- Output exactly ONE action from: ↑, ←, →, STOP.
- Output the action directly without explanation or any additional text."""


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
