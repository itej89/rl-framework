"""MCTS tree node.

Requirements: RLF-MCTS-001, RLF-MCTS-002, RLF-MCTS-004
"""

from __future__ import annotations

import math

__all__ = ["MCTSNode"]


class MCTSNode:
    """One node in the MCTS search tree.

    Args:
        parent: Parent node, or None for root.
        action: The action that led from parent to this node.
    """

    __slots__ = ("parent", "action", "N", "W", "children")

    def __init__(self, parent: MCTSNode | None, action: int | None) -> None:
        self.parent:   MCTSNode | None        = parent
        self.action:   int | None             = action
        self.N:        int                    = 0
        self.W:        float                  = 0.0
        self.children: dict[int, MCTSNode]   = {}

    # ── Selection helpers ──────────────────────────────────────────────────────

    def ucb1(self, c: float) -> float:
        """UCB1 score. Returns +inf for unvisited nodes. RLF-MCTS-002."""
        if self.N == 0:
            return math.inf
        assert self.parent is not None
        exploit = self.W / self.N
        explore = c * math.sqrt(math.log(self.parent.N) / self.N)
        return exploit + explore

    def is_leaf(self) -> bool:
        """True when this node has no expanded children."""
        return len(self.children) == 0

    def best_child(self, c: float) -> "MCTSNode":
        """Return child with highest UCB1 score."""
        return max(self.children.values(), key=lambda n: n.ucb1(c))

    # ── Backpropagation ────────────────────────────────────────────────────────

    def backprop(self, value: float) -> None:
        """Walk to root, updating N and W, negating value each level.

        RLF-MCTS-004: Connect Four is zero-sum; a win for the mover is
        a loss for the parent (opponent's perspective), so value flips.
        """
        self.N += 1
        self.W += value
        if self.parent is not None:
            self.parent.backprop(-value)
