"""GridWorld 4×4 environment.

A deterministic grid with 16 states (0–15, row-major).

Layout:
     col 0  col 1  col 2  col 3
row 0:  0      1      2      3
row 1:  4      5(X)   6      7
row 2:  8      9     10     11(X)
row 3: 12     13     14     15(G)

Actions: 0=up  1=down  2=left  3=right
Terminal states: 15 (goal, +1)  |  5, 11 (pits, -1)
Off-grid move: agent stays in current cell, reward = 0.

Requirements: RLF-ENV-010 through RLF-ENV-013
"""

from rl_framework.envs.base import BaseEnv

__all__ = ["GridWorld"]

_ROWS = 4
_COLS = 4
_GOAL: int = 15
_PITS: frozenset[int] = frozenset({5, 11})

# Action → (row_delta, col_delta)
_DELTAS: dict[int, tuple[int, int]] = {
    0: (-1, 0),   # up
    1: (1, 0),    # down
    2: (0, -1),   # left
    3: (0, 1),    # right
}


class GridWorld(BaseEnv):
    """Deterministic 4×4 grid world. See module docstring for full spec."""

    def __init__(self) -> None:
        self._state: int = 0

    # ── BaseEnv properties ────────────────────────────────────────────────

    @property
    def n_states(self) -> int:
        return _ROWS * _COLS

    @property
    def n_actions(self) -> int:
        return len(_DELTAS)

    # ── Core interface ────────────────────────────────────────────────────

    def reset(self) -> int:
        """Return to state 0 (top-left). RLF-ENV-004."""
        self._state = 0
        return self._state

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        """Transition the agent and return (next_state, reward, done, info).

        RLF-ENV-003, RLF-ENV-011, RLF-ENV-012.
        """
        next_state = self._next_state(self._state, action)
        self._state = next_state
        reward, done = self._outcome(next_state)
        return next_state, reward, done, {}

    def render(self) -> str:
        """Return a 4-row ASCII grid. RLF-ENV-013."""
        rows = []
        for r in range(_ROWS):
            cells = [self._cell_symbol(r * _COLS + c) for c in range(_COLS)]
            rows.append(" ".join(cells))
        return "\n".join(rows)

    # ── Private helpers ───────────────────────────────────────────────────

    def _next_state(self, state: int, action: int) -> int:
        """Compute next state, clamping at grid edges. RLF-ENV-011."""
        row, col = divmod(state, _COLS)
        dr, dc = _DELTAS[action]
        new_row = row + dr
        new_col = col + dc
        if not (0 <= new_row < _ROWS and 0 <= new_col < _COLS):
            return state          # off-grid: stay put
        return new_row * _COLS + new_col

    def _outcome(self, state: int) -> tuple[float, bool]:
        """Map a state to (reward, done). RLF-ENV-012."""
        if state == _GOAL:
            return 1.0, True
        if state in _PITS:
            return -1.0, True
        return 0.0, False

    def _cell_symbol(self, state: int) -> str:
        """Single character label for a cell."""
        if state == self._state:
            return "A"
        if state == _GOAL:
            return "G"
        if state in _PITS:
            return "X"
        return "."
