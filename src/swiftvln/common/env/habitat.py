"""Habitat environment wrapper for VLN evaluation.

This module provides a wrapper around habitat.Env that implements the
unified EnvWrapper interface.
"""

from typing import Any, Dict, Tuple
import numpy as np

from habitat import Env as HabitatEnv

from .base import EnvWrapper


class HabitatEnvWrapper(EnvWrapper):
    """Wrapper for Habitat VLN environment.

    This wrapper adapts habitat.Env to the unified EnvWrapper interface,
    allowing it to be used interchangeably with other environment backends.
    """

    def __init__(self, habitat_env: HabitatEnv, max_episode_steps: int = 500):
        """Initialize Habitat environment wrapper.

        Args:
            habitat_env: Initialized habitat.Env instance
            max_episode_steps: Maximum steps per episode (optional, will try to read from config)
        """
        self._env = habitat_env
        self._current_episode = None

        # Try to get max_steps from config, fallback to parameter
        try:
            self._max_steps = habitat_env._config.habitat.environment.max_episode_steps
        except (AttributeError, KeyError):
            try:
                self._max_steps = habitat_env._config.environment.max_episode_steps
            except (AttributeError, KeyError):
                self._max_steps = max_episode_steps

    def reset(self, episode: Any) -> Dict[str, Any]:
        """Reset environment to a specific episode.

        Args:
            episode: Habitat episode object

        Returns:
            Dictionary of initial observations
        """
        self._current_episode = episode
        self._env.current_episode = episode
        observations = self._env.reset()
        return observations

    def step(self, action: int) -> Tuple[Dict[str, Any], bool]:
        """Execute an action in the environment.

        Args:
            action: Action index (0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT)

        Returns:
            Tuple of (observations, done)
        """
        observations = self._env.step(action)
        done = self._env.episode_over
        return observations, done

    def get_metrics(self) -> Dict[str, float]:
        """Get current evaluation metrics.

        Returns:
            Dictionary containing metrics (success, spl, distance_to_goal, etc.)
        """
        return self._env.get_metrics()

    def get_rgb(self, obs: Dict[str, Any]) -> np.ndarray:
        """Extract RGB image from observation.

        Args:
            obs: Observation dictionary

        Returns:
            RGB image as numpy array (H, W, 3), dtype=uint8
        """
        return obs["rgb"]

    def get_instruction(self, episode: Any) -> str:
        """Extract instruction text from episode.

        Args:
            episode: Habitat episode object

        Returns:
            Natural language instruction string
        """
        # Habitat episodes have instruction as InstructionData object
        instruction = episode.instruction
        if hasattr(instruction, "instruction_text"):
            return instruction.instruction_text
        elif isinstance(instruction, dict):
            return instruction.get("instruction_text", "")
        else:
            return str(instruction)

    @property
    def max_steps(self) -> int:
        """Maximum number of steps per episode.

        Returns:
            Maximum steps from habitat config
        """
        return self._max_steps

    @property
    def env_type(self) -> str:
        """Environment type identifier.

        Returns:
            'habitat'
        """
        return "habitat"

    @property
    def episode_over(self) -> bool:
        """Check if current episode is over.

        Returns:
            True if episode is finished
        """
        return self._env.episode_over

    @property
    def episodes(self) -> list[Any]:
        """Return Habitat dataset episodes."""
        return self._env.episodes

    def close(self) -> None:
        """Clean up environment resources."""
        self._env.close()

    @property
    def env(self) -> HabitatEnv:
        """Access underlying Habitat environment.

        Returns:
            The wrapped habitat.Env instance
        """
        return self._env
