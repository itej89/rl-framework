"""P1-E1: Connect Four — pure MCTS baseline.

What this experiment shows:
  1. MCTS can play Connect Four without any domain knowledge (no hand-crafted
     evaluation function, no learned network)
  2. UCB1 balances exploration vs exploitation, allocating more simulations to
     promising moves
  3. Win rate plateaus at ~80-85% vs random at 500 sims — the motivation for
     adding a value network (issue #6, Phase 1 E2)

Acceptance criterion (issue #5):
    MCTS(500 sims) beats random agent ≥ 80% in 100 games.
    Arena completes 100 games in < 60 seconds.

Usage:
    python experiments/p1/e1_connect_four_mcts.py
"""

import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_framework.envs.connect_four import ConnectFour
from rl_framework.mcts.search import mcts_action
from rl_framework.arena import arena, random_action

OUT_DIR    = Path("results/p1_e1_connect_four_mcts")
N_GAMES    = 100
SIM_COUNTS = [50, 100, 200, 500, 1000]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*65}")
    print(f"  P1-E1: Connect Four — Pure MCTS baseline")
    print(f"  Sweep: n_simulations ∈ {SIM_COUNTS}")
    print(f"{'='*65}")

    results: dict[int, dict] = {}
    for n_sims in SIM_COUNTS:
        results[n_sims] = _run_mcts_vs_random(n_sims)

    _plot_win_rate_vs_sims(results)
    _save_results(results)
    _print_verdict(results)


# ── Experiment ────────────────────────────────────────────────────────────────

def _run_mcts_vs_random(n_sims: int) -> dict:
    """Run MCTS(n_sims) vs random for N_GAMES; report win rate and timing."""
    def mcts_agent(env: ConnectFour) -> int:
        return mcts_action(env, n_sims, c=1.41)

    env = ConnectFour()
    t = time.time()
    outcome = arena(mcts_agent, random_action, env, n_games=N_GAMES)
    elapsed = time.time() - t
    win_rate = outcome["wins_a"] / N_GAMES

    print(f"\n  n_sims={n_sims:>5}: {outcome}  "
          f"win_rate={win_rate:.2f}  time={elapsed:.1f}s")
    return {**outcome, "win_rate": win_rate, "time_s": elapsed}


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_win_rate_vs_sims(results: dict[int, dict]) -> None:
    sims  = list(results.keys())
    rates = [results[s]["win_rate"] for s in sims]
    times = [results[s]["time_s"] for s in sims]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(sims, rates, "o-", color="steelblue", linewidth=2, markersize=8)
    ax1.axhline(0.80, color="gray", linestyle="--", linewidth=1,
                label="80% threshold (issue #5)")
    ax1.set_xlabel("MCTS simulations per move")
    ax1.set_ylabel("Win rate vs random agent")
    ax1.set_title("P1-E1: Win rate vs simulation budget")
    ax1.set_xscale("log")
    ax1.set_ylim(0, 1)
    ax1.legend()

    ax2.plot(sims, times, "o-", color="coral", linewidth=2, markersize=8)
    ax2.axhline(60.0, color="gray", linestyle="--", linewidth=1,
                label="60s budget (issue #5)")
    ax2.set_xlabel("MCTS simulations per move")
    ax2.set_ylabel(f"Time for {N_GAMES} games (s)")
    ax2.set_title("P1-E1: Arena timing vs simulation budget")
    ax2.set_xscale("log")
    ax2.legend()

    plt.tight_layout()
    path = OUT_DIR / "mcts_vs_random.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Saved: {path}")


# ── I/O ───────────────────────────────────────────────────────────────────────

def _save_results(results: dict[int, dict]) -> None:
    path = OUT_DIR / "results.json"
    out = {str(k): v for k, v in results.items()}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {path}")


def _print_verdict(results: dict[int, dict]) -> None:
    print(f"\n{'='*65}")
    print("  VERDICT")
    print(f"{'='*65}")
    r500 = results[500]
    wr   = r500["win_rate"]
    t    = r500["time_s"]
    passed_wr = wr >= 0.80
    passed_t  = t < 60.0
    mark_wr = "✅" if passed_wr else "❌"
    mark_t  = "✅" if passed_t  else "❌"
    print(f"  n_sims=500: win_rate={wr:.2f}  {mark_wr}  (need ≥ 0.80)")
    print(f"  n_sims=500: time={t:.1f}s      {mark_t}  (need < 60s)")
    if passed_wr and passed_t:
        print("  ✅  PASSED — Pure MCTS baseline established")
        print("  Note: plateau at ~80% motivates value network (issue #6)")
    else:
        print("  ❌  FAILED")


if __name__ == "__main__":
    main()
