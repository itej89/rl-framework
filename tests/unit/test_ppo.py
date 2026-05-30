"""Tests for PPOAgent.

Requirements covered:
    RLF-PPO-001: select_action records (obs, action, log_prob, value) in buffer
    RLF-PPO-002: compute_gae_advantages produces correct δ_t and A_t
    RLF-PPO-003: buffer triggers gradient update when full (n_steps reached)
    RLF-PPO-004: clipped ratio is clamped to [1-ε, 1+ε]
    RLF-PPO-005: buffer is cleared after each gradient update
    RLF-PPO-006: returns G_t = A_t + V(s_t) (used for value loss)
"""

import torch
import numpy as np
import pytest
from rl_framework.agents.ppo import PPOAgent


def _make_agent(n_steps: int = 8) -> PPOAgent:
    return PPOAgent(
        obs_dim=8,
        n_actions=4,
        n_steps=n_steps,
        n_epochs=2,
        minibatch_size=4,
        clip_eps=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        lr=3e-4,
        entropy_coeff=0.01,
        value_coeff=0.5,
    )


# ── select_action ─────────────────────────────────────────────────────────────

def test_select_action_returns_valid_action() -> None:
    """Verify RLF-PPO-001: action is in [0, n_actions)."""
    agent = _make_agent()
    obs = np.zeros(8, dtype=np.float32)
    action = agent.select_action(obs)
    assert 0 <= action < 4


def test_select_action_fills_buffer_incrementally() -> None:
    """Verify RLF-PPO-001: buffer grows by one step per call."""
    agent = _make_agent(n_steps=8)
    obs = np.zeros(8, dtype=np.float32)
    assert agent._buffer_size == 0
    agent.select_action(obs)
    agent.update(obs, 0, 1.0, obs, False)
    assert agent._buffer_size == 1


# ── GAE advantages ────────────────────────────────────────────────────────────

def test_compute_gae_single_step_no_future() -> None:
    """Verify RLF-PPO-002: single done step → A_0 = r_0 + γ*0 - V(s_0)."""
    agent = _make_agent()
    rewards = torch.tensor([1.0])
    values = torch.tensor([0.5])
    next_value = torch.tensor(0.0)
    dones = torch.tensor([1.0])   # episode ended
    adv = agent.compute_gae(rewards, values, next_value, dones)
    # δ_0 = 1.0 + 0.99*0.0*(1-1) - 0.5 = 0.5;  A_0 = 0.5
    assert adv[0].item() == pytest.approx(0.5, abs=1e-4)


def test_compute_gae_two_steps_correct_discount() -> None:
    """Verify RLF-PPO-002: two steps propagate A backward correctly."""
    agent = _make_agent()
    rewards = torch.tensor([1.0, 1.0])
    values = torch.tensor([0.0, 0.0])
    next_value = torch.tensor(0.0)
    dones = torch.tensor([0.0, 1.0])
    adv = agent.compute_gae(rewards, values, next_value, dones)
    # δ_1 = 1.0 + 0.99*0*(1-1) - 0 = 1.0;  A_1 = 1.0
    # δ_0 = 1.0 + 0.99*0*(1-0) - 0 = 1.0;  A_0 = 1.0 + 0.99*0.95*1.0
    assert adv[1].item() == pytest.approx(1.0, abs=1e-4)
    assert adv[0].item() == pytest.approx(1.0 + 0.99 * 0.95 * 1.0, abs=1e-4)


def test_compute_gae_returns_same_length_as_rewards() -> None:
    """Verify RLF-PPO-002: one advantage per step."""
    agent = _make_agent()
    T = 6
    rewards = torch.ones(T)
    values = torch.zeros(T)
    adv = agent.compute_gae(rewards, values, torch.tensor(0.0), torch.zeros(T))
    assert adv.shape == (T,)


# ── Clipping ──────────────────────────────────────────────────────────────────

def test_clip_ratio_clamps_large_ratio() -> None:
    """Verify RLF-PPO-004: ratio > 1+ε is clamped when advantage > 0."""
    agent = _make_agent()
    # ratio = 2.0 (policy changed a lot), advantage = +1
    ratio = torch.tensor([2.0])
    advantage = torch.tensor([1.0])
    loss = agent.clipped_policy_loss(ratio, advantage)
    # unclipped = 2.0 * 1.0 = 2.0; clipped = 1.2 * 1.0 = 1.2 → min = 1.2
    assert loss.item() == pytest.approx(-1.2, abs=1e-4)


def test_clip_ratio_clamps_small_ratio() -> None:
    """Verify RLF-PPO-004: ratio < 1-ε is clamped when advantage < 0."""
    agent = _make_agent()
    # ratio = 0.5 (policy shrank), advantage = -1
    ratio = torch.tensor([0.5])
    advantage = torch.tensor([-1.0])
    loss = agent.clipped_policy_loss(ratio, advantage)
    # unclipped = 0.5*(-1) = -0.5; clipped = 0.8*(-1) = -0.8 → min = -0.8
    assert loss.item() == pytest.approx(0.8, abs=1e-4)


def test_clip_does_not_clip_ratio_near_one() -> None:
    """Verify RLF-PPO-004: ratio close to 1 is not clipped."""
    agent = _make_agent()
    ratio = torch.tensor([1.05])   # within [0.8, 1.2]
    advantage = torch.tensor([2.0])
    loss = agent.clipped_policy_loss(ratio, advantage)
    # both unclipped and clipped = 1.05*2 → loss = -2.1
    assert loss.item() == pytest.approx(-1.05 * 2.0, abs=1e-4)


# ── Buffer management ─────────────────────────────────────────────────────────

def test_buffer_clears_after_update() -> None:
    """Verify RLF-PPO-005: buffer size resets to 0 after n_steps reached."""
    agent = _make_agent(n_steps=4)
    obs = np.zeros(8, dtype=np.float32)
    for _ in range(4):
        agent.select_action(obs)
        agent.update(obs, 0, 1.0, obs, False)
    # After 4 steps the buffer should have been flushed
    assert agent._buffer_size == 0


def test_returns_equal_advantage_plus_value() -> None:
    """Verify RLF-PPO-006: G_t = A_t + V(s_t) used for value target."""
    agent = _make_agent()
    advantages = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.5, 0.5, 0.5])
    returns = agent.compute_returns_from_advantages(advantages, values)
    expected = torch.tensor([1.5, 2.5, 3.5])
    assert torch.allclose(returns, expected, atol=1e-5)
