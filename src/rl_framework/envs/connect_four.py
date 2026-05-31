"""Connect Four game environment — optimised for MCTS throughput.

Board layout:
    board[row][col]: row 0 = top, row 5 = bottom.
    0 = empty, 1 = player 0's piece, 2 = player 1's piece.

Performance choices:
    - _heights[col] tracks pieces in each column — O(1) drop, no row scan.
    - Win check only inspects cells around the last-placed piece.
    - clone() copies lists without deepcopy overhead.
    - step() returns self.board reference (not a copy) — callers must not mutate.

Requirements: RLF-CF-001 through RLF-CF-006, RLF-ENV-020
"""

from __future__ import annotations

from rl_framework.exceptions import EnvironmentError

__all__ = ["ConnectFour"]

_ROWS = 6
_COLS = 7
_WIN  = 4
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


class ConnectFour:
    """Two-player Connect Four — fast clone-friendly implementation."""

    rows: int = _ROWS
    cols: int = _COLS

    def __init__(self) -> None:
        self.board:    list[list[int]] = [[0] * _COLS for _ in range(_ROWS)]
        self._heights: list[int]       = [0] * _COLS   # pieces in each column
        self._current_player: int      = 0
        self._done:           bool     = False

    # ── Public interface ──────────────────────────────────────────────────────

    def reset(self) -> list[list[int]]:
        """Clear board, set player 0 to move. RLF-ENV-020."""
        self.board    = [[0] * _COLS for _ in range(_ROWS)]
        self._heights = [0] * _COLS
        self._current_player = 0
        self._done = False
        return self.board

    def step(self, action: int) -> tuple[list[list[int]], float, bool, dict]:
        """Drop a piece in column `action`. RLF-CF-001/002/003/004."""
        if self._heights[action] >= _ROWS:
            raise EnvironmentError(f"Column {action} is full.")
        row = self._drop(action, self._current_player + 1)
        if self._wins_at(row, action):
            self._done = True
            return self.board, 1.0, True, {}
        if all(h == _ROWS for h in self._heights):
            self._done = True
            return self.board, 0.0, True, {}
        self._current_player ^= 1
        return self.board, 0.0, False, {}

    def legal_actions(self) -> list[int]:
        """Columns that are not full, in ascending order. RLF-CF-005."""
        return [c for c in range(_COLS) if self._heights[c] < _ROWS]

    def current_player(self) -> int:
        """Index of player to move (0 or 1). RLF-ENV-020."""
        return self._current_player

    def render(self) -> str:
        """Human-readable board string. RLF-CF-006."""
        symbols = {0: ".", 1: "X", 2: "O"}
        rows = [
            " ".join(symbols[self.board[r][c]] for c in range(_COLS))
            for r in range(_ROWS)
        ]
        rows.append(" ".join(str(c) for c in range(_COLS)))
        return "\n".join(rows)

    def clone(self) -> "ConnectFour":
        """Fast copy for MCTS (avoids deepcopy)."""
        env = ConnectFour.__new__(ConnectFour)
        env.board    = [row[:] for row in self.board]
        env._heights = self._heights[:]
        env._current_player = self._current_player
        env._done           = self._done
        return env

    # ── Internals ─────────────────────────────────────────────────────────────

    def _drop(self, col: int, marker: int) -> int:
        """Place marker in col using height counter; return landing row."""
        row = _ROWS - 1 - self._heights[col]
        self.board[row][col] = marker
        self._heights[col] += 1
        return row

    def _wins_at(self, row: int, col: int) -> bool:
        """Check if the piece at (row, col) completes a 4-in-a-row."""
        marker = self.board[row][col]
        for dr, dc in _DIRECTIONS:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < _ROWS and 0 <= c < _COLS and self.board[r][c] == marker:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= _WIN:
                return True
        return False

    def _check_winner(self) -> int:
        """Scan full board for winner. Returns marker (1 or 2) or 0."""
        board = self.board
        for r in range(_ROWS):
            for c in range(_COLS):
                m = board[r][c]
                if m == 0:
                    continue
                for dr, dc in _DIRECTIONS:
                    if all(
                        0 <= r + dr * i < _ROWS
                        and 0 <= c + dc * i < _COLS
                        and board[r + dr * i][c + dc * i] == m
                        for i in range(1, _WIN)
                    ):
                        return m
        return 0
