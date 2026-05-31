"""MCTS search using standard negamax backpropagation.

Value convention:
    +1 = the player who JUST MOVED (into this node) won.
    -1 = they lost.
     0 = draw.

    Node.W stores the sum of values from the perspective of the player
    who MOVES FROM this node.  Since backprop negates at each step,
    a win for the child's mover becomes -1 at the parent (opponent loses).

Requirements: RLF-MCTS-002 through RLF-MCTS-005
"""

from __future__ import annotations

import random
from rl_framework.mcts.node import MCTSNode
from rl_framework.envs.connect_four import ConnectFour

__all__ = ["mcts_action"]


def mcts_action(env: ConnectFour, n_simulations: int, c: float = 1.41) -> int:
    """Run MCTS from current state, return highest-N child's action. RLF-MCTS-005."""
    root = MCTSNode(parent=None, action=None)
    root.N = 1   # avoid log(0) on first UCB1 call
    for _ in range(n_simulations):
        sim_env = env.clone()
        _simulate(root, sim_env, c)
    return max(root.children.values(), key=lambda n: n.N).action  # type: ignore[return-value]


def _simulate(root: MCTSNode, env: ConnectFour, c: float) -> None:
    """One simulation: select leaf → expand → rollout → backprop."""
    path: list[MCTSNode] = [root]

    # ── 1. Selection: follow UCB1 to a leaf ──────────────────────────────────
    node = root
    while not node.is_leaf() and not env._done:
        legal = env.legal_actions()
        unexplored = [a for a in legal if a not in node.children]
        if unexplored:
            break
        node = node.best_child(c)
        path.append(node)
        _, reward, done, _ = env.step(node.action)  # type: ignore[arg-type]
        if done:
            # Terminal found in selection. reward=+1 means the player who took
            # node.action (= node's parent's mover) won. That player's perspective
            # is stored in node's PARENT. _backprop walks reversed(path):
            #   node.W   += reward  → node gets credit (this is actually the loser's slot)
            #   parent.W += -reward → parent gets -1 (but parent won — wrong?)
            # Actually the correct interpretation: reward is for the player who
            # moved INTO node. node's W accumulates for the player who moves FROM node.
            # Since node is terminal, there's no player moving from it. We want to
            # credit the winning move for the parent's mover. So node.W should store
            # the value FROM THE PARENT'S MOVER'S PERSPECTIVE = reward = +1.
            # parent.W should store -1 (parent's parent's mover loses).
            # This is exactly _backprop(path, reward) — node gets +1, parent gets -1.
            _backprop(path, reward)
            return

    if env._done:
        _backprop(path, 0.0)
        return

    # ── 2. Expansion: add one new child ──────────────────────────────────────
    legal = env.legal_actions()
    if not legal:
        _backprop(path, 0.0)
        return
    unexplored = [a for a in legal if a not in node.children]
    action = random.choice(unexplored)
    child = MCTSNode(parent=node, action=action)
    node.children[action] = child
    path.append(child)
    _, reward, done, _ = env.step(action)

    if done:
        # reward=+1 → the player who took `action` (node's mover) won.
        # child.W should reflect child's mover's position.
        # child's mover is the OPPONENT of node's mover who just won.
        # child's mover faces a lost position → -reward.
        _backprop(path, -reward)
        return

    # ── 3. Rollout ────────────────────────────────────────────────────────────
    value = _rollout(env)
    # value = +1 if the player-to-move at rollout start wins.
    # rollout starts from child's perspective (child's mover moves first).
    # child.W += value → child's mover gets credit. Correct.
    _backprop(path, value)


def _backprop(path: list[MCTSNode], value: float) -> None:
    """Walk path from leaf to root, negating value at each step. RLF-MCTS-004."""
    for node in reversed(path):
        node.N += 1
        node.W += value
        value = -value


def _rollout(env: ConnectFour) -> float:
    """Random rollout to terminal.

    Returns +1 if player-to-move-at-start wins, -1 if they lose, 0 for draw.
    RLF-MCTS-003.
    """
    first_mover = env.current_player()
    while not env._done:
        mover = env.current_player()
        action = random.choice(env.legal_actions())
        _, reward, done, _ = env.step(action)
        if done:
            if reward == 0.0:
                return 0.0
            return 1.0 if mover == first_mover else -1.0
    return 0.0
