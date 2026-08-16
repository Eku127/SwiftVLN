"""Dependency-free static semantics for supported navigation backends."""

from __future__ import annotations

from dataclasses import dataclass


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

    def format_prompt(self, instruction: str) -> str:
        return self.prompt_template.format(instruction=instruction)


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
