"""P0-E2: REINFORCE with baseline on CartPole-v1.

What this experiment shows:
  1. REINFORCE can solve CartPole without a Q-table (continuous obs space)
  2. The baseline reduces variance: same expected gradient, less noise
  3. Learned V(s) baseline converges faster and more reliably than no baseline

Three variants compared over 5 independent seeds:
  none    — advantage = G_t                    (high variance)
  mean    — advantage = G_t - mean(G)          (simple constant baseline)
  learned — advantage = G_t - V_θ(s_t)        (state-dependent baseline)

Acceptance criterion (issue #3):
    'learned' baseline: mean return ≥ 450 on CartPole-v1 within 500 episodes,
    achieved in ≥ 3 of 5 seeds.

Usage:
    python experiments/p0/e2_cartpole_reinforce.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym

from rl_framework.agents.reinforce import ReinforceAgent

OUT_DIR = Path("results/p0_e2_cartpole_reinforce")
N_EPISODES = 500
N_SEEDS = 5
GAMMA = 0.99
LR = 3e-3
HIDDEN = 128
PASS_THRESHOLD = 450.0   # CartPole-v1 max = 500


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*65}")
    print(f"  P0-E2: REINFORCE with baseline — CartPole-v1")
    print(f"  {N_EPISODES} episodes × {N_SEEDS} seeds × 3 baselines")
    print(f"{'='*65}")

    all_results: dict[str, list] = {}
    for baseline in ("none", "mean", "learned"):
        print(f"\n  Baseline: '{baseline}'")
        seed_returns = _run_baseline(baseline)
        all_results[baseline] = seed_returns
        _print_baseline_summary(baseline, seed_returns)

    _plot_comparison(all_results)
    _plot_variance_comparison(all_results)
    _save_results(all_results)
    _print_verdict(all_results)


# ── Training ──────────────────────────────────────────────────────────────────

def _run_baseline(baseline: str) -> list[list[float]]:
    """Train N_SEEDS agents with this baseline. Return per-seed return curves."""
    seed_returns = []
    for seed in range(N_SEEDS):
        returns = _train_one_seed(baseline, seed)
        seed_returns.append(returns)
        best = max(returns[-50:])
        print(f"    seed {seed}: best last-50 avg = {best:.1f}")
    return seed_returns


def _train_one_seed(baseline: str, seed: int) -> list[float]:
    """Run one full training run, return episode return per episode."""
    env = gym.make("CartPole-v1")
    agent = ReinforceAgent(
        obs_dim=4, n_actions=2,
        gamma=GAMMA, lr=LR,
        baseline=baseline, hidden=HIDDEN,
    )
    returns = []
    for _ in range(N_EPISODES):
        ep_return = _run_episode(env, agent)
        agent.update()
        returns.append(ep_return)
    env.close()
    return returns


def _run_episode(env: gym.Env, agent: ReinforceAgent) -> float:
    """Run one episode: collect transitions, store rewards. Return total reward."""
    obs, _ = env.reset()
    total = 0.0
    done = False
    while not done:
        action = agent.select_action(obs.astype(np.float32))
        obs, reward, terminated, truncated, _ = env.step(action)
        agent.store_reward(float(reward))
        total += float(reward)
        done = terminated or truncated
    return total


# ── Diagnostics ───────────────────────────────────────────────────────────────

def _print_baseline_summary(baseline: str, seed_returns: list[list[float]]) -> None:
    """Print rolling-avg at 25%, 50%, 100% of training across seeds."""
    arr = np.array(seed_returns)          # shape (N_SEEDS, N_EPISODES)
    window = 50
    checkpoints = [N_EPISODES // 4, N_EPISODES // 2, N_EPISODES - 1]
    print(f"    Rolling avg return (window={window}, mean ± std across seeds):")
    for ep in checkpoints:
        start = max(0, ep - window)
        segment = arr[:, start:ep + 1]
        means = segment.mean(axis=1)
        print(f"      ep {ep+1:>4}: {means.mean():6.1f} ± {means.std():5.1f}")

    # Variance comparison: std of returns in last 50 episodes
    last_std = arr[:, -50:].std(axis=1).mean()
    print(f"    Return std dev (last 50 eps, avg across seeds): {last_std:.1f}")


def _print_verdict(all_results: dict[str, list]) -> None:
    print(f"\n{'='*65}")
    print(f"  VERDICT")
    print(f"{'='*65}")
    arr = np.array(all_results["learned"])   # (N_SEEDS, N_EPISODES)
    window = 50
    seeds_passed = 0
    for s in range(N_SEEDS):
        avg_last = np.mean(arr[s, -window:])
        passed = avg_last >= PASS_THRESHOLD
        seeds_passed += int(passed)
        mark = "✅" if passed else "❌"
        print(f"  seed {s}: avg last-{window} = {avg_last:.1f}  {mark}")

    joint = seeds_passed >= 3
    print(f"\n  Seeds passed: {seeds_passed}/5  (need ≥ 3)")
    if joint:
        print(f"  ✅  PASSED — REINFORCE + learned baseline solves CartPole-v1")
    else:
        print(f"  ❌  FAILED — try more episodes or tune LR")


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_comparison(all_results: dict[str, list]) -> None:
    """Rolling-avg return curves for all three baselines (mean across seeds)."""
    fig, ax = plt.subplots(figsize=(11, 5))
    window = 50
    colors = {"none": "coral", "mean": "steelblue", "learned": "limegreen"}

    for baseline, seed_returns in all_results.items():
        arr = np.array(seed_returns)
        mean_curve = arr.mean(axis=0)
        std_curve = arr.std(axis=0)
        rolling_mean = np.convolve(
            mean_curve, np.ones(window) / window, mode="valid"
        )
        rolling_std = np.convolve(
            std_curve, np.ones(window) / window, mode="valid"
        )
        episodes = np.arange(len(rolling_mean)) + window
        ax.plot(episodes, rolling_mean, color=colors[baseline],
                linewidth=2, label=f"baseline='{baseline}'")
        ax.fill_between(
            episodes,
            rolling_mean - rolling_std,
            rolling_mean + rolling_std,
            color=colors[baseline], alpha=0.15,
        )

    ax.axhline(PASS_THRESHOLD, color="gray", linestyle="--",
               linewidth=1, label=f"target ({PASS_THRESHOLD})")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Rolling avg return (w={window})")
    ax.set_title("P0-E2 REINFORCE — baseline comparison\n"
                 "(shaded = ±1 std across seeds)")
    ax.legend()
    ax.set_ylim(0, 520)
    plt.tight_layout()
    path = OUT_DIR / "baseline_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Saved: {path}")


def _plot_variance_comparison(all_results: dict[str, list]) -> None:
    """Box plot of return std-dev (last 100 eps) per baseline — shows variance."""
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = []
    stds = []
    for baseline, seed_returns in all_results.items():
        arr = np.array(seed_returns)
        per_seed_std = arr[:, -100:].std(axis=1)
        labels.append(f"'{baseline}'")
        stds.append(per_seed_std)

    ax.boxplot(stds, labels=labels, patch_artist=True,
               boxprops=dict(facecolor="lightblue"))
    ax.set_ylabel("Return std-dev (last 100 episodes)")
    ax.set_title("P0-E2 — Variance reduction from baseline\n"
                 "(lower = more stable training)")
    plt.tight_layout()
    path = OUT_DIR / "variance_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── I/O ───────────────────────────────────────────────────────────────────────

def _save_results(all_results: dict[str, list]) -> None:
    summary = {}
    for baseline, seed_returns in all_results.items():
        arr = np.array(seed_returns)
        summary[baseline] = {
            "mean_final_return": float(arr[:, -50:].mean()),
            "std_final_return": float(arr[:, -50:].std()),
            "n_seeds": N_SEEDS,
        }
    path = OUT_DIR / "results.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
