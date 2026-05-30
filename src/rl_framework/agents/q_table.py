"""Q-table agent — tabular Q-learning with ε-greedy exploration.

Algorithm (Watkins, 1989):

    Select action:
        with prob ε → random action           (explore)
        otherwise   → argmax_a Q(s, a)        (exploit)

    Update (Bellman residual step):
        Q(s,a) ← Q(s,a) + α · [r + γ·max_a' Q(s',a')·(1-done)  −  Q(s,a)]
                                └──────────────────────────────────────────┘
                                              TD error

    Episode end:
        ε ← max(ε_min, ε · ε_decay)

Requirements: RLF-AGT-010 through RLF-AGT-013
"""

import numpy as np

from rl_framework.agents.base import BaseAgent
from rl_framework.config import TrainConfig

__all__ = ["QTableAgent"]


class QTableAgent(BaseAgent):
    """Tabular Q-learning agent for discrete state/action spaces."""

    def __init__(
        self, n_states: int, n_actions: int, config: TrainConfig
    ) -> None:
        super().__init__(n_states, n_actions, config)
        # Q-table: rows = states, cols = actions. Init to zero (RLF-AGT-010).
        self._q: np.ndarray = np.zeros((n_states, n_actions))
        self._alpha = config.alpha
        self._gamma = config.gamma
        self._epsilon = config.epsilon_start
        self._epsilon_decay = config.epsilon_decay
        self._epsilon_min = config.epsilon_min

    # ── BaseAgent interface ───────────────────────────────────────────────

    def select_action(self, state: int) -> int:
        """ε-greedy action selection. RLF-AGT-011."""
        if np.random.random() < self._epsilon:
            return int(np.random.randint(self.n_actions))
        return self._greedy_action(state)

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """Q-learning Bellman update. RLF-AGT-012."""
        future_value = self._gamma * np.max(self._q[next_state]) * (1 - int(done))
        td_error = reward + future_value - self._q[state, action]
        self._q[state, action] += self._alpha * td_error

    def on_episode_end(self) -> None:
        """Decay ε after every episode. RLF-AGT-013."""
        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)

    # ── Public property ───────────────────────────────────────────────────

    @property
    def epsilon(self) -> float:
        """Current exploration rate (used by train loop for logging)."""
        return self._epsilon

    # ── Private helpers ───────────────────────────────────────────────────

    def _greedy_action(self, state: int) -> int:
        """Return the action with the highest Q-value in this state."""
        return int(np.argmax(self._q[state]))
