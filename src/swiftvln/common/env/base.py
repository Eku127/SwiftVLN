"""Environment wrapper abstract interface for VLN evaluation.

This module defines a unified interface for VLN environments that abstracts
away differences between Habitat and SatNav simulators.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np


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
