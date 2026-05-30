"""Greedy evaluation loop.

Runs the agent with ε=0 (pure exploitation) to measure the true
quality of the current policy, independent of exploration noise.

Requirements: RLF-EVL-001, RLF-EVL-002
"""

import numpy as np

from rl_framework.agents.base import BaseAgent
from rl_framework.envs.base import BaseEnv

__all__ = ["eval"]

_MAX_STEPS = 1_000   # safety cap per eval episode


def eval(
    env: BaseEnv, agent: BaseAgent, n_episodes: int
) -> dict[str, float]:
    """Run n_episodes greedy episodes and return summary statistics.

    Temporarily forces ε=0 for agents that use ε-greedy exploration,
    then restores the original ε afterward.

    Returns:
        Dict with keys: mean_return, std_return, mean_steps.
    """
    saved_epsilon = _disable_exploration(agent)
    returns, step_counts = _collect_returns(env, agent, n_episodes)
    _restore_exploration(agent, saved_epsilon)
    return _summarise(returns, step_counts)


def _disable_exploration(agent: BaseAgent) -> float:
    """Set _epsilon to 0 and return the saved value."""
    saved = getattr(agent, "_epsilon", 0.0)
    if hasattr(agent, "_epsilon"):
        agent._epsilon = 0.0      # type: ignore[attr-defined]
    return saved


def _restore_exploration(agent: BaseAgent, saved: float) -> None:
    """Restore the previously saved _epsilon value."""
    if hasattr(agent, "_epsilon"):
        agent._epsilon = saved    # type: ignore[attr-defined]


def _collect_returns(
    env: BaseEnv, agent: BaseAgent, n_episodes: int
) -> tuple[list[float], list[int]]:
    """Run episodes and collect total returns and step counts."""
    returns: list[float] = []
    step_counts: list[int] = []
    for _ in range(n_episodes):
        total, steps = _run_greedy_episode(env, agent)
        returns.append(total)
        step_counts.append(steps)
    return returns, step_counts


def _run_greedy_episode(
    env: BaseEnv, agent: BaseAgent
) -> tuple[float, int]:
    """Run one episode with no exploration. Return (total_reward, steps)."""
    state = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    while not done and steps < _MAX_STEPS:
        action = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
    return total_reward, steps


def _summarise(
    returns: list[float], step_counts: list[int]
) -> dict[str, float]:
    """Compute mean ± std of returns and mean steps."""
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_steps": float(np.mean(step_counts)),
    }
