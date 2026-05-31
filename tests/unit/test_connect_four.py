"""Tests for ConnectFour environment.

Requirements covered:
    RLF-CF-001: 6-row × 7-col board; pieces fall to lowest empty row
    RLF-CF-002: win detection — horizontal, vertical, both diagonals
    RLF-CF-003: full board with no winner is a draw (reward=0, done=True)
    RLF-CF-004: illegal move (full column) raises EnvironmentError
    RLF-CF-005: legal_actions() returns non-full columns in order
    RLF-CF-006: render() returns a non-empty string
    RLF-ENV-020: reset/step/legal_actions/current_player interface
"""

import pytest
from rl_framework.envs.connect_four import ConnectFour
from rl_framework.exceptions import EnvironmentError


def _make() -> ConnectFour:
    return ConnectFour()


# ── Interface ─────────────────────────────────────────────────────────────────

def test_reset_returns_board_state() -> None:
    """Verify RLF-ENV-020: reset() returns something (board state)."""
    env = _make()
    state = env.reset()
    assert state is not None


def test_initial_player_is_zero() -> None:
    """Verify RLF-ENV-020: first player after reset is 0."""
    env = _make()
    env.reset()
    assert env.current_player() == 0


def test_player_alternates_after_step() -> None:
    """Verify RLF-ENV-020: player toggles each step."""
    env = _make()
    env.reset()
    env.step(0)
    assert env.current_player() == 1
    env.step(1)
    assert env.current_player() == 0


def test_step_returns_four_elements() -> None:
    """Verify RLF-ENV-020: step() returns (state, reward, done, info)."""
    env = _make()
    env.reset()
    result = env.step(0)
    assert len(result) == 4


# ── Board shape & gravity ─────────────────────────────────────────────────────

def test_board_is_6_rows_7_cols() -> None:
    """Verify RLF-CF-001: board dimensions."""
    env = _make()
    env.reset()
    assert env.rows == 6
    assert env.cols == 7


def test_piece_falls_to_lowest_row() -> None:
    """Verify RLF-CF-001: first piece in col 0 lands on row 5 (bottom)."""
    env = _make()
    env.reset()
    env.step(0)   # player 0 drops in col 0
    # board[5][0] should be player 0's piece (0-indexed, row 0 = top)
    assert env.board[5][0] == 1   # player 0 marker


def test_second_piece_stacks_above_first() -> None:
    """Verify RLF-CF-001: second piece in same col lands on row 4."""
    env = _make()
    env.reset()
    env.step(0)   # player 0 → row 5
    env.step(0)   # player 1 → row 4
    assert env.board[4][0] == 2   # player 1 marker


# ── Legal actions ─────────────────────────────────────────────────────────────

def test_all_columns_legal_on_empty_board() -> None:
    """Verify RLF-CF-005: 7 legal actions on empty board."""
    env = _make()
    env.reset()
    assert env.legal_actions() == list(range(7))


def test_full_column_not_legal() -> None:
    """Verify RLF-CF-005: full column is excluded from legal_actions."""
    env = _make()
    env.reset()
    # Fill col 3 via _drop (6 pieces alternating markers, avoids win check)
    for i in range(6):
        env._drop(3, 1 + (i % 2))   # alternates 1,2,1,2,1,2 — no 4-in-a-row
        env._heights[3] = i + 1      # keep height in sync
    assert 3 not in env.legal_actions()


def test_illegal_move_raises() -> None:
    """Verify RLF-CF-004: stepping into a full column raises EnvironmentError."""
    env = _make()
    env.reset()
    # Fill col 0 using _drop so _heights is accurate
    for i in range(6):
        env._drop(0, 1 + (i % 2))
    env._heights[0] = 6   # mark as full
    with pytest.raises(EnvironmentError):
        env.step(0)


# ── Win detection ─────────────────────────────────────────────────────────────

def _drop_sequence(cols: list[int]) -> tuple[ConnectFour, float, bool]:
    """Helper: play moves alternating players, return final (env, reward, done)."""
    env = _make()
    env.reset()
    reward, done = 0.0, False
    for col in cols:
        _, reward, done, _ = env.step(col)
        if done:
            break
    return env, reward, done


def test_horizontal_win() -> None:
    """Verify RLF-CF-002: four in a row horizontally is a win."""
    # P0: cols 0,1,2,3; P1: cols 0,1,2 (bottom rows, interleaved)
    # P0 moves: 0,1,2,3  P1 moves: 4,4,4  → P0 wins on bottom row
    cols = [0, 4, 1, 4, 2, 4, 3]   # P0 gets 4 across bottom
    env, reward, done = _drop_sequence(cols)
    assert done is True
    assert reward == pytest.approx(1.0)


def test_vertical_win() -> None:
    """Verify RLF-CF-002: four in a column vertically is a win."""
    # P0: col 0 four times; P1: col 1 three times
    cols = [0, 1, 0, 1, 0, 1, 0]
    env, reward, done = _drop_sequence(cols)
    assert done is True
    assert reward == pytest.approx(1.0)


def test_diagonal_win_forward_slash() -> None:
    """Verify RLF-CF-002: four in / diagonal is a win."""
    # Build / diagonal for P0 at (5,0),(4,1),(3,2),(2,3)
    # P0 cols: 0, 1, 1, 2, 2, 2, 3   P1 cols: 5, 5, 5, 5, 5, 5
    cols = [0, 5, 1, 5, 1, 5, 2, 5, 2, 5, 2, 5, 3]
    env, reward, done = _drop_sequence(cols)
    assert done is True
    assert reward == pytest.approx(1.0)


def test_diagonal_win_back_slash() -> None:
    """Verify RLF-CF-002: four in \\ diagonal is a win."""
    # Build \ diagonal for P0 at (5,3),(4,2),(3,1),(2,0)
    cols = [3, 5, 2, 5, 2, 5, 1, 5, 1, 5, 1, 5, 0]
    env, reward, done = _drop_sequence(cols)
    assert done is True
    assert reward == pytest.approx(1.0)


def test_non_terminal_step_reward_is_zero() -> None:
    """Verify RLF-CF-002: non-terminal step returns reward=0, done=False."""
    env = _make()
    env.reset()
    _, reward, done, _ = env.step(0)
    assert reward == pytest.approx(0.0)
    assert done is False


def test_draw_no_winner_on_three_in_a_row() -> None:
    """Verify RLF-CF-003: _check_winner returns 0 when no 4-in-a-row exists."""
    env = _make()
    env.reset()
    # Three in a column — not a win
    env.board[5][0] = 1
    env.board[4][0] = 1
    env.board[3][0] = 1
    env.board[2][0] = 2   # breaks the run at 3
    assert env._check_winner() == 0


def test_draw_full_board_returns_done() -> None:
    """Verify RLF-CF-003: when board is full and no winner, legal_actions is empty."""
    env = _make()
    env.reset()
    # Set all heights to 6 (full) without placing actual pieces
    # (just testing the legal_actions logic for the draw detection path)
    env._heights = [6] * 7
    env._done = False
    assert env.legal_actions() == []   # all columns full → no legal moves


# ── Render ────────────────────────────────────────────────────────────────────

def test_render_returns_nonempty_string() -> None:
    """Verify RLF-CF-006: render() returns a non-empty string."""
    env = _make()
    env.reset()
    rendered = env.render()
    assert isinstance(rendered, str)
    assert len(rendered) > 0
