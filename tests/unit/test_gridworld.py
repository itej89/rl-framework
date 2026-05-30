"""Tests for GridWorld environment.

Requirements covered:
    RLF-ENV-001, RLF-ENV-002, RLF-ENV-003, RLF-ENV-004,
    RLF-ENV-010, RLF-ENV-011, RLF-ENV-012, RLF-ENV-013
"""

import pytest
from rl_framework.envs import GridWorld
from rl_framework.envs.base import BaseEnv


# ── Interface compliance ──────────────────────────────────────────────────────

def test_gridworld_is_instance_of_base_env() -> None:
    """Verify RLF-ENV-001: GridWorld implements the BaseEnv interface."""
    env = GridWorld()
    assert isinstance(env, BaseEnv)


def test_gridworld_n_states_is_16() -> None:
    """Verify RLF-ENV-002: GridWorld exposes n_states = 16."""
    assert GridWorld().n_states == 16


def test_gridworld_n_actions_is_4() -> None:
    """Verify RLF-ENV-002: GridWorld exposes n_actions = 4."""
    assert GridWorld().n_actions == 4


# ── reset() ───────────────────────────────────────────────────────────────────

def test_reset_returns_int() -> None:
    """Verify RLF-ENV-004: reset() returns an int."""
    state = GridWorld().reset()
    assert isinstance(state, int)


def test_reset_returns_initial_state_zero() -> None:
    """Verify RLF-ENV-004: reset() always returns state 0 (top-left cell)."""
    env = GridWorld()
    env.step(1)  # move somewhere
    assert env.reset() == 0


def test_reset_clears_episode_state() -> None:
    """Verify RLF-ENV-004: episode state is cleared on reset — done is False."""
    env = GridWorld()
    env.reset()
    # Walk into goal
    for _ in range(20):
        _, _, done, _ = env.step(1)
        if done:
            break
    env.reset()
    # After reset, a step should not immediately return done=True (unless we
    # happen to step into a terminal — but starting state 0 stepping right is safe)
    _, _, done, _ = env.step(3)  # right from cell 0 → cell 1 (not terminal)
    assert not done


# ── step() return signature ───────────────────────────────────────────────────

def test_step_returns_four_elements() -> None:
    """Verify RLF-ENV-003: step() returns exactly 4 elements."""
    env = GridWorld()
    env.reset()
    result = env.step(3)  # right
    assert len(result) == 4


def test_step_returns_correct_types() -> None:
    """Verify RLF-ENV-003: step() returns (int, float, bool, dict)."""
    env = GridWorld()
    env.reset()
    next_state, reward, done, info = env.step(3)
    assert isinstance(next_state, int)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


# ── Grid structure ────────────────────────────────────────────────────────────

def test_gridworld_has_16_states() -> None:
    """Verify RLF-ENV-010: all reachable states are in range [0, 15]."""
    env = GridWorld()
    env.reset()
    visited = set()
    # BFS over all states via all actions
    frontier = [0]
    while frontier:
        s = frontier.pop()
        if s in visited:
            continue
        visited.add(s)
        if s in (5, 11, 15):  # terminal states — skip stepping further
            continue
        env._state = s  # direct state injection for coverage
        for a in range(4):
            env._state = s
            ns, _, _, _ = env.step(a)
            if ns not in visited:
                frontier.append(ns)
    assert all(0 <= s <= 15 for s in visited)


def test_move_up_from_top_row_stays_in_place() -> None:
    """Verify RLF-ENV-011: off-grid move leaves agent in current cell."""
    env = GridWorld()
    env.reset()  # state = 0, top-left
    next_state, _, _, _ = env.step(0)  # up from row 0
    assert next_state == 0


def test_move_left_from_left_column_stays_in_place() -> None:
    """Verify RLF-ENV-011: off-grid move (left from col 0) stays in place."""
    env = GridWorld()
    env.reset()  # state = 0
    next_state, _, _, _ = env.step(2)  # left from col 0
    assert next_state == 0


def test_move_right_increases_state_by_one() -> None:
    """Verify RLF-ENV-010: moving right from state 0 goes to state 1."""
    env = GridWorld()
    env.reset()
    next_state, _, _, _ = env.step(3)  # right
    assert next_state == 1


def test_move_down_increases_state_by_four() -> None:
    """Verify RLF-ENV-010: moving down from state 0 goes to state 4."""
    env = GridWorld()
    env.reset()
    next_state, _, _, _ = env.step(1)  # down
    assert next_state == 4


# ── Reward structure ──────────────────────────────────────────────────────────

def test_goal_state_gives_positive_reward_and_done() -> None:
    """Verify RLF-ENV-012: goal state (cell 15) yields reward=+1, done=True."""
    env = GridWorld()
    env._state = 14  # one left of goal
    next_state, reward, done, _ = env.step(3)  # right → 15
    assert next_state == 15
    assert reward == pytest.approx(1.0)
    assert done is True


def test_pit_cell_5_gives_negative_reward_and_done() -> None:
    """Verify RLF-ENV-012: pit state 5 yields reward=-1, done=True."""
    env = GridWorld()
    env._state = 4  # left of pit 5
    next_state, reward, done, _ = env.step(3)  # right → 5
    assert next_state == 5
    assert reward == pytest.approx(-1.0)
    assert done is True


def test_pit_cell_11_gives_negative_reward_and_done() -> None:
    """Verify RLF-ENV-012: pit state 11 yields reward=-1, done=True."""
    env = GridWorld()
    env._state = 10  # left of pit 11
    next_state, reward, done, _ = env.step(3)  # right → 11
    assert next_state == 11
    assert reward == pytest.approx(-1.0)
    assert done is True


def test_non_terminal_step_gives_zero_reward_and_not_done() -> None:
    """Verify RLF-ENV-012: non-terminal transition yields reward=0, done=False."""
    env = GridWorld()
    env.reset()  # state 0
    _, reward, done, _ = env.step(3)  # right → 1 (not terminal)
    assert reward == pytest.approx(0.0)
    assert done is False


# ── render() ──────────────────────────────────────────────────────────────────

def test_render_returns_string() -> None:
    """Verify RLF-ENV-013: render() returns a str."""
    env = GridWorld()
    env.reset()
    assert isinstance(env.render(), str)


def test_render_contains_agent_marker() -> None:
    """Verify RLF-ENV-013: render output contains 'A' for agent position."""
    env = GridWorld()
    env.reset()
    assert "A" in env.render()


def test_render_contains_goal_marker() -> None:
    """Verify RLF-ENV-013: render output contains 'G' for goal."""
    env = GridWorld()
    env.reset()
    assert "G" in env.render()


def test_render_contains_pit_marker() -> None:
    """Verify RLF-ENV-013: render output contains 'X' for pits."""
    env = GridWorld()
    env.reset()
    assert "X" in env.render()
