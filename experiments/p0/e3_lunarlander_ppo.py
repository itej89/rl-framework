"""P0-E3: PPO on LunarLander-v2.

What this experiment shows:
  1. PPO handles continuous observations without a Q-table (curse of dimensionality)
  2. GAE reduces variance relative to Monte-Carlo returns while keeping low bias
  3. Clipped surrogate prevents large policy updates that destabilise training

Acceptance criterion (issue #4):
    Mean return ≥ 200 over a 100-episode greedy evaluation window,
    achieved within 1000 training episodes in ≥ 3 of 5 seeds.

LunarLander-v2:
    Observation: 8-dim vector (position, velocity, angle, angular-velocity, leg contacts)
    Actions:     4 discrete (do nothing, left engine, main engine, right engine)
    Solved at:   mean return ≥ 200 over 100 consecutive episodes

Usage:
    python experiments/p0/e3_lunarlander_ppo.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym

from rl_framework.agents.ppo import PPOAgent

OUT_DIR        = Path("results/p0_e3_lunarlander_ppo")
N_TRAIN_EPS    = 1000
N_SEEDS        = 5
PASS_THRESHOLD = 200.0
EVAL_WINDOW    = 100   # rolling window for the acceptance criterion


# ── Hyper-parameters ──────────────────────────────────────────────────────────

PPO_KWARGS = dict(
    obs_dim        = 8,
    n_actions      = 4,
    n_steps        = 1024,
    n_epochs       = 4,
    minibatch_size = 128,
    clip_eps       = 0.2,
    gamma          = 0.99,
    gae_lambda     = 0.95,
    lr             = 3e-4,
    entropy_coeff  = 0.01,
    value_coeff    = 0.5,
    hidden         = 128,
)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*65}")
    print(f"  P0-E3: PPO — LunarLander-v2")
    print(f"  {N_TRAIN_EPS} training episodes × {N_SEEDS} seeds")
    print(f"  Acceptance: mean return ≥ {PASS_THRESHOLD} "
          f"(rolling {EVAL_WINDOW}) in ≥ 3/5 seeds")
    print(f"{'='*65}")

    all_returns: list[list[float]] = []
    for seed in range(N_SEEDS):
        returns = _train_one_seed(seed)
        all_returns.append(returns)
        rolling_max = _rolling_max(returns, EVAL_WINDOW)
        print(f"  seed {seed}: best rolling-{EVAL_WINDOW} = {rolling_max:.1f}")

    _plot_learning_curves(all_returns)
    _save_results(all_returns)
    _print_verdict(all_returns)


# ── Training ──────────────────────────────────────────────────────────────────

def _train_one_seed(seed: int) -> list[float]:
    """Train one PPO agent for N_TRAIN_EPS episodes; return per-episode returns."""
    env   = gym.make("LunarLander-v3")
    agent = PPOAgent(**PPO_KWARGS)
    np.random.seed(seed)
    torch_seed = seed
    import torch
    torch.manual_seed(torch_seed)

    returns: list[float] = []
    for ep in range(N_TRAIN_EPS):
        ep_return = _run_episode(env, agent)
        returns.append(ep_return)
        if (ep + 1) % 100 == 0:
            recent = np.mean(returns[-EVAL_WINDOW:])
            print(f"    seed {seed}  ep {ep+1:>5}:  "
                  f"rolling-{EVAL_WINDOW} avg = {recent:7.1f}")
    env.close()
    return returns


def _run_episode(env: gym.Env, agent: PPOAgent) -> float:
    """Run one episode: collect steps, let agent update when buffer fills."""
    obs, _ = env.reset()
    total  = 0.0
    done   = False
    while not done:
        action   = agent.select_action(obs.astype(np.float32))
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done      = terminated or truncated
        agent.update(next_obs.astype(np.float32), action, float(reward), next_obs.astype(np.float32), done)
        obs       = next_obs
        total    += float(reward)
    return total


# ── Metrics ───────────────────────────────────────────────────────────────────

def _rolling_max(returns: list[float], window: int) -> float:
    arr = np.array(returns)
    if len(arr) < window:
        return float(arr.mean())
    rolling = np.convolve(arr, np.ones(window) / window, mode="valid")
    return float(rolling.max())


def _print_verdict(all_returns: list[list[float]]) -> None:
    print(f"\n{'='*65}")
    print("  VERDICT")
    print(f"{'='*65}")
    seeds_passed = 0
    for s, returns in enumerate(all_returns):
        best = _rolling_max(returns, EVAL_WINDOW)
        passed = best >= PASS_THRESHOLD
        seeds_passed += int(passed)
        mark = "✅" if passed else "❌"
        print(f"  seed {s}: best rolling-{EVAL_WINDOW} = {best:.1f}  {mark}")

    joint = seeds_passed >= 3
    print(f"\n  Seeds passed: {seeds_passed}/5  (need ≥ 3)")
    if joint:
        print("  ✅  PASSED — PPO solves LunarLander-v2")
    else:
        print("  ❌  FAILED — consider more episodes or tuning PPO hyper-params")


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_learning_curves(all_returns: list[list[float]]) -> None:
    arr    = np.array(all_returns)          # (N_SEEDS, N_TRAIN_EPS)
    window = EVAL_WINDOW
    mean_c = arr.mean(axis=0)
    std_c  = arr.std(axis=0)

    rolling_mean = np.convolve(mean_c, np.ones(window) / window, mode="valid")
    rolling_std  = np.convolve(std_c,  np.ones(window) / window, mode="valid")
    episodes     = np.arange(len(rolling_mean)) + window

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(episodes, rolling_mean, color="steelblue", linewidth=2, label="mean across seeds")
    ax.fill_between(
        episodes,
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        color="steelblue", alpha=0.2, label="±1 std across seeds",
    )
    ax.axhline(PASS_THRESHOLD, color="gray", linestyle="--",
               linewidth=1, label=f"target ({PASS_THRESHOLD})")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Rolling avg return (w={window})")
    ax.set_title("P0-E3 PPO — LunarLander-v2\n(shaded = ±1 std across seeds)")
    ax.legend()
    plt.tight_layout()
    path = OUT_DIR / "learning_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Saved: {path}")


# ── I/O ───────────────────────────────────────────────────────────────────────

def _save_results(all_returns: list[list[float]]) -> None:
    arr = np.array(all_returns)
    summary = {
        "mean_best_rolling_return": float(
            np.mean([_rolling_max(r, EVAL_WINDOW) for r in all_returns])
        ),
        "std_best_rolling_return": float(
            np.std([_rolling_max(r, EVAL_WINDOW) for r in all_returns])
        ),
        "mean_final_return": float(arr[:, -EVAL_WINDOW:].mean()),
        "std_final_return":  float(arr[:, -EVAL_WINDOW:].std()),
        "n_seeds":           N_SEEDS,
        "n_episodes":        N_TRAIN_EPS,
        "ppo_kwargs":        {k: v for k, v in PPO_KWARGS.items()},
    }
    path = OUT_DIR / "results.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
