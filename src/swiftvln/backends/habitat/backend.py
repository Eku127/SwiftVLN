"""Habitat evaluation backend composition."""

from __future__ import annotations

from typing import Any

import numpy as np

from swiftvln.backends.base import BackendDependencyError, EvaluationBackend, EnvWrapper
from swiftvln.backends.specs import HABITAT_SPEC

from .config import load_habitat_config


class HabitatBackend(EvaluationBackend):
    spec = HABITAT_SPEC

    def load_config(self, config_path: str, args: Any):
        return load_habitat_config(config_path, args)

    def create_wrapper(self) -> EnvWrapper:
        try:
            from habitat import Env
        except ModuleNotFoundError as exc:
            raise BackendDependencyError(
                "Habitat backend requires habitat-lab and habitat-sim. "
                "Use the validated evaluation environment before selecting "
                "--env-type habitat."
            ) from exc
        from .wrapper import HabitatEnvWrapper

        environment = Env(config=self.config)
        max_steps = self.config.habitat.environment.max_episode_steps
        return HabitatEnvWrapper(environment, max_episode_steps=max_steps)

    def create_video_state(self) -> list[np.ndarray]:
        return []

    def capture_before_step(
        self,
        state: list[np.ndarray],
        *,
        observations: dict[str, Any],
        instruction: str,
        env_wrapper: EnvWrapper,
        episode: Any,
        rgb: np.ndarray,
    ) -> None:
        del episode, rgb
        from .video import collect_frame

        frame = collect_frame(observations, instruction, env_wrapper)
        if frame is not None:
            state.append(frame)

    def save_episode_video(
        self,
        state: list[np.ndarray],
        *,
        episode_id: str,
        instruction: str,
        metrics: dict[str, Any],
    ) -> None:
        del instruction
        from .video import save_video

        save_video(
            output_path=self.output_path,
            episode_id=episode_id,
            frames=state,
            metrics=metrics,
        )
