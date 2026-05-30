"""Training configuration dataclass.

Centralises all hyperparameters so experiment scripts can pass a single
object to train() instead of a long list of keyword arguments.
"""

from dataclasses import dataclass

__all__ = ["TrainConfig"]


@dataclass
class TrainConfig:
    """Hyperparameters for the generic training loop.

    Attributes:
        n_episodes:              Total number of training episodes.
        max_steps_per_episode:   Hard cap on steps per episode (safety valve).
        alpha:                   Learning rate α for Q-learning / value updates.
        gamma:                   Discount factor γ ∈ (0, 1].
        epsilon_start:           Initial exploration rate ε.
        epsilon_decay:           Multiplicative decay applied each episode end.
        epsilon_min:             Floor — ε never drops below this value.
    """

    n_episodes: int
    max_steps_per_episode: int
    alpha: float
    gamma: float
    epsilon_start: float
    epsilon_decay: float
    epsilon_min: float
