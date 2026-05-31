"""Arena: pit two agents against each other for N games.

Requirements: RLF-ARENA-001, RLF-ARENA-002
"""

from __future__ import annotations

import random
from typing import Callable
from rl_framework.envs.connect_four import ConnectFour

__all__ = ["arena", "random_action"]

AgentFn = Callable[[ConnectFour], int]


def arena(
    agent_a: AgentFn,
    agent_b: AgentFn,
    env: ConnectFour,
    n_games: int,
) -> dict[str, int]:
    """Play n_games games, alternating first-mover. RLF-ARENA-001.

    Args:
        agent_a: Callable (env) → action for player A.
        agent_b: Callable (env) → action for player B.
        env:     ConnectFour instance (reset() is called each game).
        n_games: Total number of games to play.

    Returns:
        dict with keys wins_a, wins_b, draws.
    """
    wins_a = wins_b = draws = 0
    for game_idx in range(n_games):
        a_is_first = (game_idx % 2 == 0)
        result = _play_one(agent_a, agent_b, env, a_is_first)
        if result == 1:
            wins_a += 1
        elif result == -1:
            wins_b += 1
        else:
            draws += 1
    return {"wins_a": wins_a, "wins_b": wins_b, "draws": draws}


def random_action(env: ConnectFour) -> int:
    """Baseline agent: pick a random legal action."""
    return random.choice(env.legal_actions())


# ── Internal ──────────────────────────────────────────────────────────────────

def _play_one(
    agent_a: AgentFn,
    agent_b: AgentFn,
    env: ConnectFour,
    a_is_first: bool,
) -> int:
    """Play one game. Return +1 if A wins, -1 if B wins, 0 for draw."""
    env.reset()
    agents = [agent_a, agent_b] if a_is_first else [agent_b, agent_a]
    # agents[0] moves when current_player==0, agents[1] when current_player==1
    # Map back to A/B: if a_is_first, player0=A, player1=B
    player0_is_a = a_is_first

    while True:
        player = env.current_player()
        action = agents[player](env)
        _, reward, done, _ = env.step(action)
        if done:
            if reward == 0.0:
                return 0   # draw
            # The player who just moved won
            winner_is_player0 = (player == 0)
            winner_is_a = winner_is_player0 == player0_is_a
            return 1 if winner_is_a else -1
