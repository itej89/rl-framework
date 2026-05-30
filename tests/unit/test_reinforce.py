"""Tests for ReinforceAgent.

Requirements covered:
    RLF-PG-001: select_action returns valid action and records log_prob
    RLF-PG-002: store_reward accumulates rewards during episode
    RLF-PG-003: compute_returns produces correct discounted sums
    RLF-PG-004: update clears episode buffers after applying gradient
    RLF-PG-005: baseline='none' uses raw returns as advantage
    RLF-PG-006: baseline='mean' subtracts mean return as constant baseline
    RLF-PG-007: baseline='learned' uses value network V(s) as baseline
"""

import torch
import numpy as np
import pytest
from rl_framework.agents.reinforce import ReinforceAgent


def _make_agent(baseline: str = "none") -> ReinforceAgent:
    return ReinforceAgent(
        obs_dim=4,
        n_actions=2,
        gamma=0.99,
        lr=1e-3,
        baseline=baseline,
    )


# ── select_action ─────────────────────────────────────────────────────────────

def test_select_action_returns_valid_action() -> None:
    """Verify RLF-PG-001: action is in {0, 1}."""
    agent = _make_agent()
    obs = np.zeros(4, dtype=np.float32)
    action = agent.select_action(obs)
    assert action in (0, 1)


def test_select_action_records_log_prob() -> None:
    """Verify RLF-PG-001: each call appends one log_prob to buffer."""
    agent = _make_agent()
    obs = np.zeros(4, dtype=np.float32)
    assert len(agent._log_probs) == 0
    agent.select_action(obs)
    assert len(agent._log_probs) == 1
    agent.select_action(obs)
    assert len(agent._log_probs) == 2


def test_select_action_records_state_for_learned_baseline() -> None:
    """Verify RLF-PG-007: learned baseline stores states for V(s) computation."""
    agent = _make_agent(baseline="learned")
    obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    agent.select_action(obs)
    assert len(agent._states) == 1


# ── store_reward ──────────────────────────────────────────────────────────────

def test_store_reward_accumulates_rewards() -> None:
    """Verify RLF-PG-002: rewards accumulate in buffer across steps."""
    agent = _make_agent()
    assert len(agent._rewards) == 0
    agent.store_reward(1.0)
    agent.store_reward(1.0)
    assert len(agent._rewards) == 2


# ── compute_returns ───────────────────────────────────────────────────────────

def test_compute_returns_single_reward() -> None:
    """Verify RLF-PG-003: single reward → G_0 = reward."""
    agent = _make_agent()
    returns = agent.compute_returns([5.0], gamma=0.99)
    assert returns[0] == pytest.approx(5.0)


def test_compute_returns_two_rewards() -> None:
    """Verify RLF-PG-003: G_0 = r_0 + γ·r_1."""
    agent = _make_agent()
    returns = agent.compute_returns([1.0, 1.0], gamma=0.9)
    # G_1 = 1.0;  G_0 = 1.0 + 0.9*1.0 = 1.9
    assert returns[1] == pytest.approx(1.0)
    assert returns[0] == pytest.approx(1.9)


def test_compute_returns_three_rewards() -> None:
    """Verify RLF-PG-003: discounting is correct over 3 steps."""
    agent = _make_agent()
    returns = agent.compute_returns([1.0, 1.0, 1.0], gamma=0.9)
    # G_2 = 1.0
    # G_1 = 1.0 + 0.9*1.0 = 1.9
    # G_0 = 1.0 + 0.9*1.9 = 2.71
    assert returns[0] == pytest.approx(2.71)
    assert returns[1] == pytest.approx(1.9)
    assert returns[2] == pytest.approx(1.0)


def test_compute_returns_length_matches_rewards() -> None:
    """Verify RLF-PG-003: one return per reward."""
    agent = _make_agent()
    rewards = [1.0] * 10
    returns = agent.compute_returns(rewards, gamma=0.99)
    assert len(returns) == 10


# ── update: buffer management ─────────────────────────────────────────────────

def test_update_clears_episode_buffers() -> None:
    """Verify RLF-PG-004: after update, all episode buffers are empty."""
    agent = _make_agent()
    obs = np.zeros(4, dtype=np.float32)
    for _ in range(5):
        agent.select_action(obs)
        agent.store_reward(1.0)
    agent.update()
    assert len(agent._log_probs) == 0
    assert len(agent._rewards) == 0


def test_update_with_learned_baseline_clears_states() -> None:
    """Verify RLF-PG-007: states buffer cleared after update."""
    agent = _make_agent(baseline="learned")
    obs = np.zeros(4, dtype=np.float32)
    for _ in range(3):
        agent.select_action(obs)
        agent.store_reward(1.0)
    agent.update()
    assert len(agent._states) == 0


# ── baseline variants ─────────────────────────────────────────────────────────

def test_no_baseline_advantage_equals_returns() -> None:
    """Verify RLF-PG-005: advantage = G_t when baseline='none'."""
    agent = _make_agent(baseline="none")
    returns = torch.tensor([3.0, 2.0, 1.0])
    advantage = agent._compute_advantage(returns, states=None)
    assert torch.allclose(advantage, returns)


def test_mean_baseline_advantage_is_zero_mean() -> None:
    """Verify RLF-PG-006: mean-subtracted advantage has zero mean."""
    agent = _make_agent(baseline="mean")
    returns = torch.tensor([1.0, 2.0, 3.0])
    advantage = agent._compute_advantage(returns, states=None)
    assert advantage.mean().abs() < 1e-5


def test_learned_baseline_advantage_shape_matches_returns() -> None:
    """Verify RLF-PG-007: learned advantage has same shape as returns."""
    agent = _make_agent(baseline="learned")
    returns = torch.tensor([1.0, 2.0, 3.0])
    states = torch.zeros(3, 4)
    advantage = agent._compute_advantage(returns, states=states)
    assert advantage.shape == returns.shape
