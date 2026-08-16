"""SatNav evaluation backend composition."""

from __future__ import annotations

from typing import Any

import numpy as np

from swiftvln.backends.base import BackendDependencyError, EvaluationBackend, EnvWrapper
from swiftvln.backends.specs import SATNAV_SPEC

from .config import load_satnav_config
from .video import SatNavVideoState


class SatNavBackend(EvaluationBackend):
    spec = SATNAV_SPEC
    captures_after_step = True

    def load_config(self, config_path: str, args: Any):
        return load_satnav_config(config_path, args)

    def create_wrapper(self) -> EnvWrapper:
        try:
            from satnav.core.env import Env
            from satnav.dataset.satnav_dataset import SatNavDataset
        except ModuleNotFoundError as exc:
            raise BackendDependencyError(
                "SatNav backend is unavailable because optional dependency "
                f"{exc.name!r} is not installed. Install the SatNav repository "
                "in the evaluation environment before selecting --env-type satnav."
            ) from exc
        from .wrapper import SatNavEnvWrapper

        dataset = SatNavDataset(self.config.DATASET)
        environment = Env(self.config, dataset=dataset, cycle=False)
        return SatNavEnvWrapper(environment, config=self.config)

    def create_video_state(self) -> SatNavVideoState:
        return SatNavVideoState()

    def capture_before_step(
        self,
        state: SatNavVideoState,
        *,
        observations: dict[str, Any],
        instruction: str,
        env_wrapper: EnvWrapper,
        episode: Any,
        rgb: np.ndarray,
    ) -> None:
        del observations, instruction, env_wrapper, episode
        state.rgb_frames.append(rgb.copy())

    def capture_after_step(
        self,
        state: SatNavVideoState,
        *,
        env_wrapper: EnvWrapper,
        episode: Any,
        action: int,
        step_id: int,
        rgb_fallback: np.ndarray,
    ) -> None:
        from .video import collect_topdown

        collect_topdown(
            state,
            env_wrapper=env_wrapper,
            episode=episode,
            action_symbol=self.idx2actions.get(action, "STOP"),
            step_id=step_id,
            rgb_fallback=rgb_fallback,
        )

    def save_episode_video(
        self,
        state: SatNavVideoState,
        *,
        episode_id: str,
        instruction: str,
        metrics: dict[str, Any],
    ) -> None:
        from .video import save_video

        save_video(
            state,
            output_path=self.output_path,
            episode_id=episode_id,
            instruction=instruction,
            metrics=metrics,
        )
