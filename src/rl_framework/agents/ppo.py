"""Proximal Policy Optimisation (PPO-clip) agent.

Algorithm (Schulman et al., 2017 — arXiv:1707.06347):

    COLLECT n_steps of on-policy experience:
        a_t ~ π_θ(·|s_t),  record (s_t, a_t, log π_θ(a_t|s_t), V_φ(s_t), r_t, done_t)

    COMPUTE advantages via Generalised Advantage Estimation (GAE):
        δ_t = r_t + γ · V(s_{t+1}) · (1 − done_t) − V(s_t)
        A_t = δ_t + (γλ) · A_{t+1}    (backward sweep, A_T = δ_T)

    COMPUTE returns:
        G_t = A_t + V(s_t)             (used as value-function targets)

    UPDATE for n_epochs epochs over random minibatches:
        ratio       = π_θ_new(a_t|s_t) / π_θ_old(a_t|s_t)   (prob ratio)
        L_CLIP      = E[min(ratio·A_t, clip(ratio, 1−ε, 1+ε)·A_t)]
        L_V         = E[(V_φ(s_t) − G_t)²]
        L_entropy   = E[H[π_θ(·|s_t)]]                        (exploration bonus)
        loss = −L_CLIP + c_v·L_V − c_e·L_entropy
        Update θ, φ jointly.
        Clear buffer after each update cycle.

Requirements: RLF-PPO-001 through RLF-PPO-006
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = ["PPOAgent"]


# ── Shared MLP builder ────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    """Two-layer MLP with ReLU hidden activation."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


# ── Shared actor-critic network ───────────────────────────────────────────────

class _ActorCritic(nn.Module):
    """Shared-trunk policy + value network."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


# ── PPO agent ─────────────────────────────────────────────────────────────────

class PPOAgent:
    """PPO-clip with GAE advantage estimation.

    Args:
        obs_dim:        Observation vector dimension.
        n_actions:      Number of discrete actions.
        n_steps:        Rollout length before each gradient update.
        n_epochs:       Number of optimisation epochs per rollout.
        minibatch_size: Minibatch size for gradient updates.
        clip_eps:       PPO clip threshold ε.
        gamma:          Discount factor γ.
        gae_lambda:     GAE smoothing parameter λ.
        lr:             Optimiser learning rate.
        entropy_coeff:  Entropy bonus coefficient c_e.
        value_coeff:    Value loss coefficient c_v.
        hidden:         Hidden layer width.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_steps: int = 2048,
        n_epochs: int = 10,
        minibatch_size: int = 64,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        lr: float = 3e-4,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        hidden: int = 128,
    ) -> None:
        self._n_steps = n_steps
        self._n_epochs = n_epochs
        self._minibatch_size = minibatch_size
        self._clip_eps = clip_eps
        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._entropy_coeff = entropy_coeff
        self._value_coeff = value_coeff

        self._net = _ActorCritic(obs_dim, n_actions, hidden)
        self._opt = torch.optim.Adam(self._net.parameters(), lr=lr)

        # Rollout buffer — filled one step at a time
        self._obs_buf:      list[torch.Tensor] = []
        self._act_buf:      list[int]          = []
        self._logp_buf:     list[torch.Tensor] = []
        self._val_buf:      list[torch.Tensor] = []
        self._rew_buf:      list[float]        = []
        self._done_buf:     list[float]        = []
        self._last_obs:     torch.Tensor | None = None

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def _buffer_size(self) -> int:
        """Number of completed (obs, action, …, reward, done) tuples stored."""
        return len(self._rew_buf)

    def select_action(self, obs: np.ndarray) -> int:
        """Sample action from π_θ; record obs, log_prob, value. RLF-PPO-001."""
        obs_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            logits, value = self._net(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        self._last_obs = obs_t
        self._obs_buf.append(obs_t)
        self._act_buf.append(int(action.item()))
        self._logp_buf.append(dist.log_prob(action))
        self._val_buf.append(value)
        return int(action.item())

    def update(
        self,
        next_obs: np.ndarray,
        action: int,
        reward: float,
        next_next_obs: np.ndarray,
        done: bool,
    ) -> dict[str, float] | None:
        """Store one transition; run PPO update when buffer is full.

        Returns loss dict after an update, else None. RLF-PPO-003/005.
        """
        self._rew_buf.append(reward)
        self._done_buf.append(float(done))

        if len(self._rew_buf) < self._n_steps:
            return None

        # Buffer full — run update
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32)
        with torch.no_grad():
            _, next_value = self._net(next_obs_t)

        rewards  = torch.tensor(self._rew_buf,  dtype=torch.float32)
        dones    = torch.tensor(self._done_buf, dtype=torch.float32)
        values   = torch.stack(self._val_buf)
        obs_t    = torch.stack(self._obs_buf)
        acts_t   = torch.tensor(self._act_buf,  dtype=torch.long)
        old_logp = torch.stack(self._logp_buf)

        advantages = self.compute_gae(rewards, values.detach(), next_value, dones)
        returns    = self.compute_returns_from_advantages(advantages, values.detach())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        losses = self._optimise(obs_t, acts_t, old_logp.detach(), advantages, returns)
        self._clear_buffer()
        return losses

    # ── GAE / returns ─────────────────────────────────────────────────────────

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_value: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Generalised Advantage Estimation — backward sweep. RLF-PPO-002."""
        T = len(rewards)
        advantages = torch.zeros(T)
        gae = 0.0
        for t in reversed(range(T)):
            v_next = next_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self._gamma * v_next * (1.0 - dones[t]) - values[t]
            gae = float(delta) + self._gamma * self._gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
        return advantages

    def compute_returns_from_advantages(
        self,
        advantages: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """G_t = A_t + V(s_t) — value target for critic. RLF-PPO-006."""
        return advantages + values

    # ── PPO objective ─────────────────────────────────────────────────────────

    def clipped_policy_loss(
        self,
        ratio: torch.Tensor,
        advantage: torch.Tensor,
    ) -> torch.Tensor:
        """Pessimistic clipped surrogate loss (negated objective). RLF-PPO-004."""
        surr_unclipped = ratio * advantage
        surr_clipped   = ratio.clamp(1.0 - self._clip_eps, 1.0 + self._clip_eps) * advantage
        return -torch.min(surr_unclipped, surr_clipped).mean()

    # ── Optimisation ──────────────────────────────────────────────────────────

    def _optimise(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_logp: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict[str, float]:
        T = len(obs)
        total_policy = total_value = total_entropy = 0.0
        steps = 0

        for _ in range(self._n_epochs):
            perm = torch.randperm(T)
            for start in range(0, T, self._minibatch_size):
                idx = perm[start : start + self._minibatch_size]
                mb_loss, mb_pol, mb_val, mb_ent = self._mb_update(
                    obs[idx], actions[idx], old_logp[idx],
                    advantages[idx], returns[idx],
                )
                total_policy  += mb_pol
                total_value   += mb_val
                total_entropy += mb_ent
                steps += 1

        n = max(steps, 1)
        return {
            "policy_loss":  total_policy  / n,
            "value_loss":   total_value   / n,
            "entropy_loss": total_entropy / n,
        }

    def _mb_update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_logp: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> tuple[float, float, float, float]:
        logits, values = self._net(obs)
        dist     = torch.distributions.Categorical(logits=logits)
        new_logp = dist.log_prob(actions)
        entropy  = dist.entropy().mean()

        ratio    = (new_logp - old_logp).exp()
        pol_loss = self.clipped_policy_loss(ratio, advantages)
        val_loss = nn.functional.mse_loss(values, returns)
        loss     = pol_loss + self._value_coeff * val_loss - self._entropy_coeff * entropy

        self._opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=0.5)
        self._opt.step()

        return (
            float(loss.item()),
            float(pol_loss.item()),
            float(val_loss.item()),
            float(entropy.item()),
        )

    def _clear_buffer(self) -> None:
        """Reset all rollout buffers. RLF-PPO-005."""
        self._obs_buf.clear()
        self._act_buf.clear()
        self._logp_buf.clear()
        self._val_buf.clear()
        self._rew_buf.clear()
        self._done_buf.clear()
        self._last_obs = None
