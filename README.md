# RL Framework: AlphaGo → AlphaFold Arc

A learning + experimenting + framework-building journey through reinforcement learning, tracing DeepMind's research arc from AlphaGo (2016) to AlphaTensor/AlphaDev (2022–23).

## Goal

Build a reusable RL framework by implementing progressively complex experiments — starting from Q-learning on GridWorld, ending with RL-driven algorithm discovery (sorting networks).

## Phase Overview

| Phase | Analog | Core Concept | Milestone |
|---|---|---|---|
| 0 | Pre-AlphaGo | Q-learning, PPO, policy gradients | PPO solves LunarLander (return ≥ 200) |
| 1 | AlphaGo (2016) | MCTS + value/policy network | MCTS+net beats pure MCTS by ≥10 Elo |
| 2 | AlphaZero (2018) | Self-play, no human data | Near-perfect Connect Four in 200 iterations |
| 3 | AlphaCode (2022) | RL on structured discrete output | DSL synthesis pass@10 ≥ 80% |
| 4 | AlphaFold (2021) | RL + differentiable physics | RNA contact F1 ≥ 0.60 |
| 5 | AlphaTensor/Dev | RL for algorithm discovery | Rediscover optimal sorting network n=6 |

## Framework Structure (target)

```
rl_framework/
  envs/          # GridWorld, Connect Four, TicTacToe, DSL synthesis, sorting network
  agents/        # Q-table, REINFORCE, PPO, AlphaZero, GRPO, Expert Iteration
  mcts/          # Reusable tree search, UCB1, PUCT
  networks/      # CNN, Transformer, Evoformer-lite
  algorithms/    # PPO, ExIt, AlphaZero loop, structure RL
  verifiers/     # Game winner, code correctness, RNA F1, sorting network validity
  train.py       # Generic training loop
  eval.py        # Elo, pass@k, F1, returns
  cluster/       # AMD MI300X / SLURM scripts
```

## Compute

AMD MI300X GPUs (8× per node), ROCm, SLURM. Single-node experiments run in Docker.

## References

- AlphaGo: Silver et al. (2016) — [Nature](https://www.nature.com/articles/nature16961)
- AlphaZero: Silver et al. (2018) — [arXiv:1712.01815](https://arxiv.org/abs/1712.01815)
- AlphaCode: Li et al. (2022) — [Science](https://www.science.org/doi/10.1126/science.abq1158)
- AlphaFold2: Jumper et al. (2021) — [Nature](https://www.nature.com/articles/s41586-021-03819-2)
- AlphaTensor: Fawzi et al. (2022) — [Nature](https://www.nature.com/articles/s41586-022-05172-4)
- AlphaDev: Mankowitz et al. (2023) — [Nature](https://www.nature.com/articles/s41586-023-06004-9)
- Expert Iteration (ExIt): Anthony et al. (2017) — [arXiv:1705.08439](https://arxiv.org/abs/1705.08439)
- PPO: Schulman et al. (2017) — [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
