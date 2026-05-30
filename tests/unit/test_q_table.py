"""Tests for QTableAgent.

Requirements covered:
    RLF-AGT-001, RLF-AGT-002, RLF-AGT-010, RLF-AGT-011,
    RLF-AGT-012, RLF-AGT-013
"""

import numpy as np
import pytest
from rl_framework.agents import QTableAgent
from rl_framework.agents.base import BaseAgent
from rl_framework.config import TrainConfig


def _make_agent(epsilon: float = 0.0) -> QTableAgent:
    cfg = TrainConfig(
        n_episodes=100,
        max_steps_per_episode=200,
        alpha=0.1,
        gamma=0.9,
        epsilon_start=epsilon,
        epsilon_decay=0.99,
        epsilon_min=0.01,
    )
    return QTableAgent(n_states=16, n_actions=4, config=cfg)


# ── Interface compliance ──────────────────────────────────────────────────────

def test_q_table_agent_is_instance_of_base_agent() -> None:
    """Verify RLF-AGT-001: QTableAgent implements the BaseAgent interface."""
    agent = _make_agent()
    assert isinstance(agent, BaseAgent)


# ── Q-table initialisation ────────────────────────────────────────────────────

def test_q_table_initialised_to_zero() -> None:
    """Verify RLF-AGT-010: all Q-values start at zero."""
    agent = _make_agent()
    assert np.all(agent._q == 0.0)


def test_q_table_shape_is_states_by_actions() -> None:
    """Verify RLF-AGT-010: Q-table shape is (n_states, n_actions)."""
    agent = _make_agent()
    assert agent._q.shape == (16, 4)


# ── select_action: exploration ────────────────────────────────────────────────

def test_select_action_explores_when_epsilon_is_one() -> None:
    """Verify RLF-AGT-011: with ε=1, actions are random (not always argmax)."""
    agent = _make_agent(epsilon=1.0)
    # Artificially make action 0 always the argmax
    agent._q[:, 0] = 100.0
    actions = {agent.select_action(0) for _ in range(50)}
    # With ε=1, should see multiple different actions, not just 0
    assert len(actions) > 1


def test_select_action_exploits_when_epsilon_is_zero() -> None:
    """Verify RLF-AGT-011: with ε=0, always picks argmax action."""
    agent = _make_agent(epsilon=0.0)
    agent._q[0, 2] = 5.0  # action 2 is best in state 0
    actions = [agent.select_action(0) for _ in range(20)]
    assert all(a == 2 for a in actions)


def test_select_action_returns_valid_action_index() -> None:
    """Verify RLF-AGT-011: returned action is always in [0, n_actions)."""
    agent = _make_agent(epsilon=0.5)
    for state in range(16):
        a = agent.select_action(state)
        assert 0 <= a < 4


# ── update: Bellman Q-learning ────────────────────────────────────────────────

def test_update_moves_q_value_toward_bellman_target() -> None:
    """Verify RLF-AGT-012: Q(s,a) moves toward r + γ·max Q(s',a')."""
    agent = _make_agent()
    # Set a known next-state value so target is predictable
    agent._q[5, :] = [0.0, 0.0, 10.0, 0.0]  # max Q(s'=5) = 10
    # Q(0,1) starts at 0; target = 0 + 0.9*10 = 9; update: 0 + 0.1*(9-0) = 0.9
    agent.update(state=0, action=1, reward=0.0, next_state=5, done=False)
    assert agent._q[0, 1] == pytest.approx(0.9)


def test_update_zeroes_future_value_at_terminal_state() -> None:
    """Verify RLF-AGT-012: (1-done) zeroes future value when done=True."""
    agent = _make_agent()
    agent._q[15, :] = [100.0, 100.0, 100.0, 100.0]  # high Q at terminal
    # done=True: target = r + γ*max*0 = 1.0; update: 0 + 0.1*(1.0-0) = 0.1
    agent.update(state=14, action=3, reward=1.0, next_state=15, done=True)
    assert agent._q[14, 3] == pytest.approx(0.1)


def test_update_with_negative_reward_decreases_q_value() -> None:
    """Verify RLF-AGT-012: negative reward drives Q values negative."""
    agent = _make_agent()
    agent.update(state=4, action=3, reward=-1.0, next_state=5, done=True)
    assert agent._q[4, 3] < 0.0


# ── ε decay ───────────────────────────────────────────────────────────────────

def test_epsilon_decays_after_on_episode_end() -> None:
    """Verify RLF-AGT-013: ε is multiplied by epsilon_decay on episode end."""
    agent = _make_agent(epsilon=1.0)
    initial_eps = agent._epsilon
    agent.on_episode_end()
    assert agent._epsilon == pytest.approx(initial_eps * 0.99)


def test_epsilon_does_not_go_below_epsilon_min() -> None:
    """Verify RLF-AGT-013: ε is floored at epsilon_min after many episodes."""
    agent = _make_agent(epsilon=1.0)
    for _ in range(10_000):
        agent.on_episode_end()
    assert agent._epsilon >= 0.01
