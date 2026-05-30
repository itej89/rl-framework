"""P0-E1: Q-learning on GridWorld 4×4.

What this experiment shows:
  1. The Bellman update propagating value outward from the goal state
  2. How ε-decay shifts the agent from exploration to exploitation
  3. The learned Q-table — you can read off V*(s) = max_a Q(s,a) for each cell
  4. The learned policy — the arrow the agent would take from each cell
  5. Convergence: acceptance criterion is ≥95% of eval episodes reach goal in ≤20 steps

Acceptance criterion (issue #2):
    eval over 100 episodes → success rate ≥ 95%
    (success = episode ends at goal, not pit)

Usage:
    python experiments/p0/e1_gridworld_qlearning.py
    python experiments/p0/e1_gridworld_qlearning.py --sweep   # hyperparameter sweep
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed — saves to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from rl_framework.envs import GridWorld
from rl_framework.agents import QTableAgent
from rl_framework.config import TrainConfig
from rl_framework.train import train
from rl_framework.eval import eval as rl_eval

# ── Constants ─────────────────────────────────────────────────────────────────

OUT_DIR = Path("results/p0_e1_gridworld")
ACTION_ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}
GRID_ROWS, GRID_COLS = 4, 4
GOAL, PITS = 15, {5, 11}

# Default hyperparameters — chosen to converge reliably on GridWorld
DEFAULT_CFG = TrainConfig(
    n_episodes=3_000,
    max_steps_per_episode=200,
    alpha=0.3,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
)

# ── Main entry point ───────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        _run_sweep()
    else:
        _run_single(DEFAULT_CFG, label="default")


# ── Single run ─────────────────────────────────────────────────────────────────

def _run_single(cfg: TrainConfig, label: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  P0-E1: GridWorld Q-learning  [{label}]")
    print(f"{'='*60}")
    _print_config(cfg)

    env = GridWorld()
    agent = QTableAgent(
        n_states=env.n_states, n_actions=env.n_actions, config=cfg
    )

    print(f"\n  Training for {cfg.n_episodes} episodes...")
    metrics = train(env, agent, cfg)
    _print_training_summary(metrics)

    print("\n  Learned Q-table (max Q per state = V*(s) estimate):")
    _print_value_table(agent)

    print("\n  Learned policy (best action per cell):")
    _print_policy(agent)

    print("\n  Evaluating (100 greedy episodes)...")
    eval_results = rl_eval(env, agent, n_episodes=100)
    success_rate = _compute_success_rate(env, agent)
    _print_eval_results(eval_results, success_rate)

    _plot_learning_curve(metrics, label)
    _plot_q_table(agent, label)
    _plot_policy(agent, label)

    result = {
        "label": label,
        "config": cfg.__dict__,
        "eval": eval_results,
        "success_rate": success_rate,
        "passed": success_rate >= 0.95,
    }
    _save_result(result, label)
    _print_verdict(success_rate)
    return result


# ── Training diagnostics ───────────────────────────────────────────────────────

def _print_config(cfg: TrainConfig) -> None:
    print(f"  α={cfg.alpha}  γ={cfg.gamma}  "
          f"ε: {cfg.epsilon_start}→{cfg.epsilon_min} (decay {cfg.epsilon_decay})")
    print(f"  {cfg.n_episodes} episodes, max {cfg.max_steps_per_episode} steps each")


def _print_training_summary(metrics: list[dict]) -> None:
    """Print rolling-average reward at 10%, 50%, and 100% of training."""
    n = len(metrics)
    checkpoints = [n // 10, n // 2, n - 1]
    window = 100
    print(f"\n  Rolling avg reward (window={window}):")
    for i in checkpoints:
        start = max(0, i - window)
        avg = np.mean([m["total_reward"] for m in metrics[start : i + 1]])
        eps = metrics[i]["epsilon"]
        print(f"    ep {i+1:>5}: avg_reward={avg:+.3f}  ε={eps:.3f}")


def _print_value_table(agent: QTableAgent) -> None:
    """Print V*(s) ≈ max_a Q(s,a) as a 4×4 grid.

    This is the Bellman equation made visible: the value of each state
    is exactly what the agent estimated through experience.
    """
    print()
    for r in range(GRID_ROWS):
        row_str = ""
        for c in range(GRID_COLS):
            s = r * GRID_COLS + c
            v = np.max(agent._q[s])
            if s == GOAL:
                row_str += "  [GOAL]"
            elif s in PITS:
                row_str += "  [ PIT]"
            else:
                row_str += f"  {v:+.3f}"
        print(row_str)
    print()
    print("  Reading: positive = good state, negative = near pit,")
    print("           values discount toward goal by γ=0.99 per step.")


def _print_policy(agent: QTableAgent) -> None:
    """Print the greedy policy as arrows on a 4×4 grid."""
    print()
    for r in range(GRID_ROWS):
        row_str = ""
        for c in range(GRID_COLS):
            s = r * GRID_COLS + c
            if s == GOAL:
                row_str += "  G"
            elif s in PITS:
                row_str += "  X"
            else:
                best_a = int(np.argmax(agent._q[s]))
                row_str += f"  {ACTION_ARROWS[best_a]}"
        print(row_str)
    print()
    print("  Reading: arrow = best action from each cell (optimal policy π*)")


# ── Evaluation ────────────────────────────────────────────────────────────────

def _compute_success_rate(env: GridWorld, agent: QTableAgent) -> float:
    """Fraction of 100 greedy episodes that end at goal (not pit/timeout)."""
    successes = 0
    saved_eps = agent._epsilon
    agent._epsilon = 0.0
    for _ in range(100):
        state = env.reset()
        for _ in range(20):          # acceptance criterion: ≤20 steps
            action = agent.select_action(state)
            state, _, done, _ = env.step(action)
            if done:
                break
        if state == GOAL:
            successes += 1
    agent._epsilon = saved_eps
    return successes / 100


def _print_eval_results(eval_results: dict, success_rate: float) -> None:
    print(f"    mean_return : {eval_results['mean_return']:+.3f}")
    print(f"    std_return  : {eval_results['std_return']:.3f}")
    print(f"    mean_steps  : {eval_results['mean_steps']:.1f}")
    print(f"    success_rate: {success_rate:.0%}  (goal reached in ≤20 steps)")


def _print_verdict(success_rate: float) -> None:
    if success_rate >= 0.95:
        print(f"\n  ✅  PASSED — success rate {success_rate:.0%} ≥ 95%")
    else:
        print(f"\n  ❌  FAILED — success rate {success_rate:.0%} < 95%")


# ── Hyperparameter sweep ───────────────────────────────────────────────────────

def _run_sweep() -> None:
    """Sweep α and γ to show their effect on convergence speed."""
    print("\nRunning hyperparameter sweep (α × γ)...")
    alphas = [0.1, 0.3, 0.5]
    gammas = [0.90, 0.99]
    summary = []
    for alpha in alphas:
        for gamma in gammas:
            cfg = TrainConfig(
                n_episodes=3_000,
                max_steps_per_episode=200,
                alpha=alpha,
                gamma=gamma,
                epsilon_start=1.0,
                epsilon_decay=0.995,
                epsilon_min=0.01,
            )
            label = f"a{alpha}_g{gamma}"
            result = _run_single(cfg, label)
            summary.append(result)

    print("\n" + "="*60)
    print("  SWEEP SUMMARY")
    print("="*60)
    print(f"  {'label':20s}  {'success':>8}  {'mean_ret':>10}  {'passed':>6}")
    for r in summary:
        print(f"  {r['label']:20s}  {r['success_rate']:8.0%}  "
              f"{r['eval']['mean_return']:+10.3f}  {'✅' if r['passed'] else '❌':>6}")


# ── Plotting ───────────────────────────────────────────────────────────────────

def _plot_learning_curve(metrics: list[dict], label: str) -> None:
    """Rolling-average reward curve + ε-decay on twin axis."""
    window = 100
    rewards = [m["total_reward"] for m in metrics]
    epsilons = [m["epsilon"] for m in metrics]
    rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
    episodes = np.arange(len(rolling)) + window

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(episodes, rolling, color="steelblue", linewidth=1.5,
             label=f"Rolling avg reward (w={window})")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Avg total reward", color="steelblue")
    ax1.set_title(f"P0-E1 GridWorld Q-learning — {label}")

    ax2 = ax1.twinx()
    ax2.plot(range(len(epsilons)), epsilons, color="coral",
             linewidth=1.0, alpha=0.7, label="ε (exploration rate)")
    ax2.set_ylabel("ε", color="coral")
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    plt.tight_layout()
    path = OUT_DIR / f"learning_curve_{label}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def _plot_q_table(agent: QTableAgent, label: str) -> None:
    """Heatmap of V*(s) = max_a Q(s,a)."""
    v_star = np.max(agent._q, axis=1).reshape(GRID_ROWS, GRID_COLS)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(v_star, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="V*(s) = max_a Q(s,a)")

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            s = r * GRID_COLS + c
            label_txt = "GOAL" if s == GOAL else ("PIT" if s in PITS else
                        f"{v_star[r, c]:+.2f}")
            ax.text(c, r, label_txt, ha="center", va="center",
                    fontsize=9, color="black")

    ax.set_xticks(range(GRID_COLS))
    ax.set_yticks(range(GRID_ROWS))
    ax.set_title(f"Learned V*(s) — {label}\n(green=high value, red=low)")
    plt.tight_layout()
    path = OUT_DIR / f"q_table_{label}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def _plot_policy(agent: QTableAgent, label: str) -> None:
    """Arrow grid showing the greedy policy π*(s) = argmax_a Q(s,a)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.5, GRID_COLS - 0.5)
    ax.set_ylim(-0.5, GRID_ROWS - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks(range(GRID_COLS))
    ax.set_yticks(range(GRID_ROWS))
    ax.grid(True, linewidth=0.8)

    # Arrow deltas for each action (col_delta, row_delta) in plot coords
    arrow_d = {0: (0, -0.35), 1: (0, 0.35), 2: (-0.35, 0), 3: (0.35, 0)}

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            s = r * GRID_COLS + c
            if s == GOAL:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (c - 0.4, r - 0.4), 0.8, 0.8,
                    boxstyle="round,pad=0.1", color="limegreen", zorder=2))
                ax.text(c, r, "G", ha="center", va="center",
                        fontsize=12, fontweight="bold")
            elif s in PITS:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (c - 0.4, r - 0.4), 0.8, 0.8,
                    boxstyle="round,pad=0.1", color="tomato", zorder=2))
                ax.text(c, r, "X", ha="center", va="center",
                        fontsize=12, fontweight="bold")
            else:
                best_a = int(np.argmax(agent._q[s]))
                dx, dy = arrow_d[best_a]
                ax.annotate("", xy=(c + dx, r + dy), xytext=(c, r),
                            arrowprops=dict(arrowstyle="->", color="steelblue",
                                            lw=2.0))

    ax.set_title(f"Learned policy π*(s) — {label}\n(arrow = greedy action)")
    plt.tight_layout()
    path = OUT_DIR / f"policy_{label}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── I/O ────────────────────────────────────────────────────────────────────────

def _save_result(result: dict, label: str) -> None:
    path = OUT_DIR / f"result_{label}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P0-E1: GridWorld Q-learning")
    p.add_argument("--sweep", action="store_true",
                   help="Run α×γ hyperparameter sweep")
    return p.parse_args()


if __name__ == "__main__":
    main()
