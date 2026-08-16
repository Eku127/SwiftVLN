"""Environment wrapper abstract interface for VLN evaluation.

This module defines a unified interface for VLN environments that abstracts
away differences between Habitat and SatNav simulators.
"""

from abc import ABC, abstractmethod
import random
import re
from typing import Any, Dict, Tuple
import numpy as np
import torch

from .specs import EnvironmentSpec


DEFAULT_EVAL_SEED = 42


class BackendDependencyError(ImportError):
    """Raised when the selected backend's optional simulator is unavailable."""


class EvaluationBackend(ABC):
    """Protocol-like base for config, environment, actions, and video hooks."""

    spec: EnvironmentSpec
    captures_after_step = False

    def __init__(self, config_path: str, args: Any):
        self.args = args
        self.config_path = config_path
        self.save_video = bool(getattr(args, "save_video", False))
        self.output_path = getattr(args, "output_dir", "./results/eval")
        self.config = self.load_config(config_path, args)
        self.idx2actions = self.spec.action_map
        self.actions2idx = {
            symbol: index for index, symbol in self.idx2actions.items()
        }

    @abstractmethod
    def load_config(self, config_path: str, args: Any) -> Any:
        """Load and apply runtime overrides for this backend."""

    @abstractmethod
    def create_wrapper(self) -> "EnvWrapper":
        """Create the selected simulator and wrap it behind ``EnvWrapper``."""

    def set_seed(self, seed: int = DEFAULT_EVAL_SEED) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def parse_actions(self, output: str) -> list[int]:
        patterns = "|".join(re.escape(action) for action in self.actions2idx)
        return [self.actions2idx[item] for item in re.findall(patterns, output)]

    def create_video_state(self) -> Any:
        return None

    def capture_before_step(
        self,
        state: Any,
        *,
        observations: dict[str, Any],
        instruction: str,
        env_wrapper: "EnvWrapper",
        episode: Any,
        rgb: np.ndarray,
    ) -> None:
        del state, observations, instruction, env_wrapper, episode, rgb

    def capture_after_step(
        self,
        state: Any,
        *,
        env_wrapper: "EnvWrapper",
        episode: Any,
        action: int,
        step_id: int,
        rgb_fallback: np.ndarray,
    ) -> None:
        del state, env_wrapper, episode, action, step_id, rgb_fallback

    def save_episode_video(
        self,
        state: Any,
        *,
        episode_id: str,
        instruction: str,
        metrics: dict[str, Any],
    ) -> None:
        del state, episode_id, instruction, metrics


class EnvWrapper(ABC):
    """Abstract wrapper for VLN environments.

    This class provides a unified interface for different VLN simulation
    backends (Habitat, SatNav, etc.), allowing the same evaluation code
    to work across different environments.
    """

    @abstractmethod
    def reset(self, episode: Any) -> Dict[str, Any]:
        """Reset environment to a specific episode.

        Args:
            episode: Episode object containing start position, goal, instruction, etc.

        Returns:
            Dictionary of initial observations (e.g., {'rgb': image})
        """
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[Dict[str, Any], bool]:
        """Execute an action in the environment.

        Args:
            action: Action index (0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT)

        Returns:
            Tuple of:
                - observations: Dictionary of observations
                - done: Boolean indicating if episode is over
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Get current evaluation metrics.

        Returns:
            Dictionary containing metrics like:
                - 'success': 1.0 if successful, 0.0 otherwise
                - 'spl': Success weighted by Path Length
                - 'distance_to_goal': Distance to goal in meters
                - 'oracle_success': Oracle success (if available)
        """
        pass

    @abstractmethod
    def get_rgb(self, obs: Dict[str, Any]) -> np.ndarray:
        """Extract RGB image from observation.

        Args:
            obs: Observation dictionary from reset() or step()

        Returns:
            RGB image as numpy array (H, W, 3), dtype=uint8
        """
        pass

    @abstractmethod
    def get_instruction(self, episode: Any) -> str:
        """Extract instruction text from episode.

        Args:
            episode: Episode object

        Returns:
            Natural language instruction as string
        """
        pass

    @property
    @abstractmethod
    def max_steps(self) -> int:
        """Maximum number of steps per episode.

        Returns:
            Maximum steps allowed
        """
        pass

    @property
    @abstractmethod
    def env_type(self) -> str:
        """Environment type identifier for prompt template selection.

        Returns:
            'habitat' or 'satnav'
        """
        pass

    @property
    @abstractmethod
    def episode_over(self) -> bool:
        """Check if current episode is over.

        Returns:
            True if episode is finished
        """
        pass

    @property
    @abstractmethod
    def episodes(self) -> list[Any]:
        """Return the episodes exposed by this environment's dataset."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up environment resources."""
        pass
