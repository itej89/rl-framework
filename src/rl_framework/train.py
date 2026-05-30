"""Generic training loop.

Works with any (BaseEnv, BaseAgent) pair — no knowledge of the specific
algorithm or environment.  This is the algorithm-agnostic outer loop:

    for episode in range(n_episodes):
        state = env.reset()
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
        agent.on_episode_end()      ← ε-decay, lr schedules, etc.

Requirements: RLF-TRN-001, RLF-TRN-002, RLF-TRN-003
"""

from rl_framework.agents.base import BaseAgent
from rl_framework.config import TrainConfig
from rl_framework.envs.base import BaseEnv

__all__ = ["train"]


def train(
    env: BaseEnv, agent: BaseAgent, config: TrainConfig
) -> list[dict[str, float | int]]:
    """Run training for config.n_episodes episodes.

    Returns:
        List of per-episode metric dicts with keys:
        episode, total_reward, steps, epsilon.
    """
    metrics = []
    for episode in range(config.n_episodes):
        episode_metrics = _run_episode(env, agent, config, episode)
        agent.on_episode_end()
        metrics.append(episode_metrics)
    return metrics


def _run_episode(
    env: BaseEnv, agent: BaseAgent, config: TrainConfig, episode: int
) -> dict[str, float | int]:
    """Run a single episode and return its metrics."""
    state = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    while not done and steps < config.max_steps_per_episode:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        steps += 1
    return {
        "episode": episode,
        "total_reward": total_reward,
        "steps": steps,
        "epsilon": agent.epsilon,
    }
