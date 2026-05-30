"""Tests for train() and eval() loops.

Requirements covered: RLF-TRN-001, RLF-TRN-002, RLF-TRN-003,
                      RLF-EVL-001, RLF-EVL-002
"""

import pytest
from rl_framework.envs import GridWorld
from rl_framework.agents import QTableAgent
from rl_framework.config import TrainConfig
from rl_framework.train import train
from rl_framework.eval import eval as rl_eval


def _make_env_and_agent(n_episodes: int = 5, epsilon: float = 0.5) -> tuple:
    env = GridWorld()
    cfg = TrainConfig(
        n_episodes=n_episodes,
        max_steps_per_episode=50,
        alpha=0.1,
        gamma=0.9,
        epsilon_start=epsilon,
        epsilon_decay=0.99,
        epsilon_min=0.01,
    )
    agent = QTableAgent(n_states=env.n_states, n_actions=env.n_actions, config=cfg)
    return env, agent, cfg


# ── train() ───────────────────────────────────────────────────────────────────

def test_train_returns_a_list() -> None:
    """Verify RLF-TRN-003: train() returns a list."""
    env, agent, cfg = _make_env_and_agent()
    metrics = train(env, agent, cfg)
    assert isinstance(metrics, list)


def test_train_returns_one_dict_per_episode() -> None:
    """Verify RLF-TRN-001 + RLF-TRN-003: list length equals n_episodes."""
    env, agent, cfg = _make_env_and_agent(n_episodes=7)
    metrics = train(env, agent, cfg)
    assert len(metrics) == 7


def test_train_metrics_contain_required_keys() -> None:
    """Verify RLF-TRN-003: each dict has episode, total_reward, steps, epsilon."""
    env, agent, cfg = _make_env_and_agent()
    metrics = train(env, agent, cfg)
    for m in metrics:
        assert "episode" in m
        assert "total_reward" in m
        assert "steps" in m
        assert "epsilon" in m


def test_train_episode_indices_are_sequential() -> None:
    """Verify RLF-TRN-003: episode key is 0-indexed and sequential."""
    env, agent, cfg = _make_env_and_agent(n_episodes=5)
    metrics = train(env, agent, cfg)
    assert [m["episode"] for m in metrics] == list(range(5))


def test_train_steps_does_not_exceed_max_steps() -> None:
    """Verify RLF-TRN-001: episodes are capped at max_steps_per_episode."""
    env, agent, cfg = _make_env_and_agent()
    metrics = train(env, agent, cfg)
    for m in metrics:
        assert m["steps"] <= cfg.max_steps_per_episode


# ── eval() ────────────────────────────────────────────────────────────────────

def test_eval_returns_dict() -> None:
    """Verify RLF-EVL-002: eval() returns a dict."""
    env, agent, cfg = _make_env_and_agent()
    train(env, agent, cfg)
    result = rl_eval(env, agent, n_episodes=3)
    assert isinstance(result, dict)


def test_eval_returns_required_keys() -> None:
    """Verify RLF-EVL-002: dict contains mean_return, std_return, mean_steps."""
    env, agent, cfg = _make_env_and_agent()
    train(env, agent, cfg)
    result = rl_eval(env, agent, n_episodes=3)
    assert "mean_return" in result
    assert "std_return" in result
    assert "mean_steps" in result


def test_eval_runs_greedy_policy_not_random() -> None:
    """Verify RLF-EVL-001: eval uses greedy policy (ε=0).

    Strategy: set Q-values so greedy policy always moves right, then
    verify agent consistently takes action 3 (right) from state 0.
    """
    env, agent, cfg = _make_env_and_agent(epsilon=0.0)
    # Make action 3 (right) strongly preferred in every state
    agent._q[:, 3] = 100.0
    actions_taken: list[int] = []

    original_select = agent.select_action

    def spy_select(state: int) -> int:
        a = original_select(state)
        actions_taken.append(a)
        return a

    agent.select_action = spy_select  # type: ignore[method-assign]
    rl_eval(env, agent, n_episodes=3)

    # Every recorded action should be 3 (greedy choice given Q[:,3]=100)
    assert all(a == 3 for a in actions_taken)
