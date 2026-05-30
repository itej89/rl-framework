"""REINFORCE policy gradient agent with optional value baseline.

Algorithm (Williams, 1992):

    COLLECT episode:
        for each step t:
            a_t ~ π_θ(·|s_t)          sample action from policy network
            record log π_θ(a_t|s_t)   needed for gradient
            record r_t                 reward from environment

    COMPUTE returns (backward pass through time):
        G_T = r_T
        G_t = r_t + γ · G_{t+1}       discounted sum from t onward

    COMPUTE advantage (subtract baseline to reduce variance):
        none:    A_t = G_t
        mean:    A_t = G_t - mean(G)
        learned: A_t = G_t - V_φ(s_t)  where V_φ is a value network

    UPDATE networks (gradient ascent on J):
        Policy loss:  L_π  = -E[ log π_θ(a_t|s_t) · A_t ]
        Value loss:   L_V  =  E[ (G_t - V_φ(s_t))² ]       (if learned baseline)

        θ ← θ - α · ∇_θ L_π
        φ ← φ - α · ∇_φ L_V

Key insight: the negative sign on L_π turns gradient DESCENT into ASCENT.
We want to *maximise* J, but PyTorch optimisers do *minimise* — so we negate.

Requirements: RLF-PG-001 through RLF-PG-007
"""

import torch
import torch.nn as nn
import numpy as np

__all__ = ["ReinforceAgent"]

# ── Shared MLP builder ────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    """Two-layer MLP with ReLU activation."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


# ── Agent ─────────────────────────────────────────────────────────────────────

class ReinforceAgent:
    """REINFORCE with configurable baseline.

    Args:
        obs_dim:   Dimension of the observation vector.
        n_actions: Number of discrete actions.
        gamma:     Discount factor γ.
        lr:        Learning rate for both policy and value networks.
        baseline:  One of 'none', 'mean', 'learned'.
        hidden:    Hidden layer size for MLP networks.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        baseline: str = "none",
        hidden: int = 128,
    ) -> None:
        self._gamma = gamma
        self._baseline = baseline

        # Policy network: obs → logits over actions
        self._policy = _mlp(obs_dim, hidden, n_actions)
        self._policy_opt = torch.optim.Adam(self._policy.parameters(), lr=lr)

        # Value network only needed for learned baseline
        self._value: nn.Sequential | None = None
        self._value_opt: torch.optim.Adam | None = None
        if baseline == "learned":
            self._value = _mlp(obs_dim, hidden, 1)
            self._value_opt = torch.optim.Adam(
                self._value.parameters(), lr=lr
            )

        # Episode buffers — cleared after every update()
        self._log_probs: list[torch.Tensor] = []
        self._rewards: list[float] = []
        self._states: list[torch.Tensor] = []

    # ── Public interface ──────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray) -> int:
        """Sample action from π_θ(·|obs). Records log_prob. RLF-PG-001."""
        obs_t = torch.tensor(obs, dtype=torch.float32)
        logits = self._policy(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        self._log_probs.append(dist.log_prob(action))
        if self._baseline == "learned":
            self._states.append(obs_t)
        return int(action.item())

    def store_reward(self, reward: float) -> None:
        """Append one step's reward to the episode buffer. RLF-PG-002."""
        self._rewards.append(reward)

    def update(self) -> dict[str, float]:
        """Apply REINFORCE gradient update at end of episode. RLF-PG-004."""
        returns = self.compute_returns(self._rewards, self._gamma)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        states_t = torch.stack(self._states) if self._states else None
        advantage = self._compute_advantage(returns_t, states_t)
        policy_loss = self._policy_loss(advantage)
        value_loss = self._value_loss(returns_t, states_t)
        self._clear_buffers()
        return {"policy_loss": policy_loss, "value_loss": value_loss}

    def compute_returns(self, rewards: list[float], gamma: float) -> list[float]:
        """Discounted return G_t = r_t + γ·G_{t+1}. RLF-PG-003."""
        returns: list[float] = []
        g = 0.0
        for r in reversed(rewards):
            g = r + gamma * g
            returns.insert(0, g)
        return returns

    # ── Advantage computation (one method per baseline type) ──────────────

    def _compute_advantage(
        self,
        returns: torch.Tensor,
        states: torch.Tensor | None,
    ) -> torch.Tensor:
        """Dispatch to the correct baseline variant."""
        if self._baseline == "none":
            return returns
        if self._baseline == "mean":
            return returns - returns.mean()
        return self._learned_advantage(returns, states)

    def _learned_advantage(
        self,
        returns: torch.Tensor,
        states: torch.Tensor | None,
    ) -> torch.Tensor:
        """Advantage = G_t - V_φ(s_t). Detach so policy grad is clean."""
        assert self._value is not None and states is not None
        values = self._value(states).squeeze(-1)
        return returns - values.detach()

    # ── Loss computation and optimiser steps ──────────────────────────────

    def _policy_loss(self, advantage: torch.Tensor) -> float:
        """Negate because we maximise J but minimise loss. RLF-PG-001."""
        log_probs = torch.stack(self._log_probs)
        loss = -(log_probs * advantage.detach()).mean()
        self._policy_opt.zero_grad()
        loss.backward()
        self._policy_opt.step()
        return float(loss.item())

    def _value_loss(
        self,
        returns: torch.Tensor,
        states: torch.Tensor | None,
    ) -> float:
        """MSE between V(s_t) and G_t. Only runs for learned baseline."""
        if self._value is None or states is None:
            return 0.0
        assert self._value_opt is not None
        values = self._value(states).squeeze(-1)
        loss = nn.functional.mse_loss(values, returns)
        self._value_opt.zero_grad()
        loss.backward()
        self._value_opt.step()
        return float(loss.item())

    def _clear_buffers(self) -> None:
        """Reset all episode buffers. Called at end of update(). RLF-PG-004."""
        self._log_probs.clear()
        self._rewards.clear()
        self._states.clear()
