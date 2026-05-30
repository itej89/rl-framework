"""Abstract base class for all rl_framework environments.

Every concrete environment (GridWorld, ConnectFour, SortingNetwork, …) must
implement this interface. The train loop and eval loop depend ONLY on this
interface — they never import a concrete environment class directly.

This is the Dependency Inversion Principle (SWG-001 §5 — DIP):
    train() depends on BaseEnv (abstraction), not GridWorld (concrete).
"""

from abc import ABC, abstractmethod

__all__ = ["BaseEnv"]


class BaseEnv(ABC):
    """Interface every environment must satisfy.

    State and action spaces are discrete integers.  Continuous environments
    (CartPole, LunarLander) will be wrapped to produce integer observations
    via discretisation or feature extraction in later phases.
    """

    @property
    @abstractmethod
    def n_states(self) -> int:
        """Total number of distinct states in this environment."""

    @property
    @abstractmethod
    def n_actions(self) -> int:
        """Total number of distinct actions available at every state."""

    @abstractmethod
    def reset(self) -> int:
        """Reset to the initial state and return it.

        Must clear all episode-internal state so no information leaks
        between episodes.
        """

    @abstractmethod
    def step(self, action: int) -> tuple[int, float, bool, dict]:
        """Apply action and return (next_state, reward, done, info).

        Args:
            action: Integer index in [0, n_actions).

        Returns:
            next_state: Integer state reached after the action.
            reward:     Scalar reward signal.
            done:       True if the episode has ended.
            info:       Diagnostic dict (may be empty).
        """

    @abstractmethod
    def render(self) -> str:
        """Return a human-readable string representation of the current state."""
