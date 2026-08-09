"""SatNav environment wrapper for VLN evaluation.

This module provides a wrapper around satnav.core.env.Env that implements the
unified EnvWrapper interface.
"""

from typing import TYPE_CHECKING, Any, Dict, Tuple
import numpy as np

if TYPE_CHECKING:
    from satnav.core.env import Env as SatNavEnv
else:
    SatNavEnv = Any

from ..base import EnvWrapper


class SatNavEnvWrapper(EnvWrapper):
    """Wrapper for SatNav VLN environment.

    This wrapper adapts satnav.core.env.Env to the unified EnvWrapper interface,
    allowing it to be used interchangeably with other environment backends.
    """

    def __init__(self, satnav_env: SatNavEnv, config: Any = None):
        """Initialize SatNav environment wrapper.

        Args:
            satnav_env: Initialized satnav.core.env.Env instance
            config: SatNav configuration object (needed for video generation)
        """
        self._env = satnav_env
        self._current_episode = None
        self._last_info = {}  # Store last step info for video generation
        self._config = config  # Store config for annotate_topdown_map

    def reset(self, episode: Any) -> Dict[str, Any]:
        """Reset environment to a specific episode.

        Args:
            episode: SatNav episode object

        Returns:
            Dictionary of initial observations
        """
        self._current_episode = episode
        # SatNav uses reset_to_episode for direct episode control
        observations = self._env.reset_to_episode(episode)
        return observations

    def step(self, action: int) -> Tuple[Dict[str, Any], bool]:
        """Execute an action in the environment.

        Args:
            action: Action index (0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT)

        Returns:
            Tuple of (observations, done)
        """
        observations, done, info = self._env.step(action)
        self._last_info = info  # Store info for video generation
        return observations, done

    def get_last_step_info(self) -> Dict[str, Any]:
        """Get info dictionary from the last step.

        Returns:
            Info dictionary containing metrics, top_down_map, etc.
        """
        return self._last_info

    def get_agent_state(self) -> Any:
        """Get current agent state (position, rotation).

        Returns:
            Agent state object from simulator
        """
        return self._env._task._sim.get_agent_state()

    @property
    def config(self) -> Any:
        """Get configuration object.

        Returns:
            SatNav configuration object
        """
        return self._config

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
            episode: SatNav episode object

        Returns:
            Natural language instruction string
        """
        # SatNav episodes have instruction as dict with 'text' key
        instruction = episode.instruction
        if isinstance(instruction, dict):
            return instruction.get("text", instruction.get("instruction_text", ""))
        elif hasattr(instruction, "text"):
            return instruction.text
        elif hasattr(instruction, "instruction_text"):
            return instruction.instruction_text
        else:
            return str(instruction)

    @property
    def max_steps(self) -> int:
        """Maximum number of steps per episode.

        Returns:
            Maximum steps from SatNav environment
        """
        return self._env.max_episode_steps

    @property
    def env_type(self) -> str:
        """Environment type identifier.

        Returns:
            'satnav'
        """
        return "satnav"

    @property
    def episode_over(self) -> bool:
        """Check if current episode is over.

        Returns:
            True if episode is finished
        """
        return self._env.episode_over

    @property
    def episodes(self) -> list[Any]:
        """Return SatNav dataset episodes."""
        return self._env._dataset.episodes

    def close(self) -> None:
        """Clean up environment resources."""
        # SatNav Env doesn't have explicit close method currently
        # but we provide this for interface consistency
        pass

    @property
    def env(self) -> SatNavEnv:
        """Access underlying SatNav environment.

        Returns:
            The wrapped satnav.core.env.Env instance
        """
        return self._env
