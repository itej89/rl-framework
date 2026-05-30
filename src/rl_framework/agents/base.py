"""Abstract base class for all rl_framework agents.

Every agent (QTableAgent, ReinforceAgent, PPOAgent, AlphaZero, …) must
implement this interface.  The train loop calls ONLY these methods.

DIP (SWG-001 §5): train() depends on BaseAgent, not QTableAgent.
"""

from abc import ABC, abstractmethod

from rl_framework.config import TrainConfig

__all__ = ["BaseAgent"]


class BaseAgent(ABC):
    """Interface every agent must satisfy.

    Attributes:
        n_states:  Number of states in the environment.
        n_actions: Number of actions in the environment.
    """

    def __init__(
        self, n_states: int, n_actions: int, config: TrainConfig
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions

    @abstractmethod
    def select_action(self, state: int) -> int:
        """Choose an action for the given state.

        During training this may be stochastic (exploration).
        During evaluation the caller is responsible for forcing ε=0
        before calling this method.
        """

    @abstractmethod
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """Update internal parameters from one (s, a, r, s', done) transition."""

    def on_episode_end(self) -> None:
        """Hook called by the train loop at the end of each episode.

        Override to implement ε-decay, learning rate schedules, etc.
        Default: no-op.
        """

    @property
    def epsilon(self) -> float:
        """Current exploration rate. Override in agents that use ε-greedy."""
        return 0.0
