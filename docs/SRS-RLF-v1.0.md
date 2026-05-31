# SRS-RLF-v1.0
# Software Requirements Specification — RL Framework

**Project**: rl_framework — Reusable Reinforcement Learning Library
**Standard ID**: SRS-RLF-v1.0
**Version**: 1.0
**Date**: 2026-05-30
**Status**: Draft
**Authors**: itej89
**Reviewers**: TBD
**Approved by**: TBD

---

## Changelog

| Version | Date | Author | Summary |
|---------|------|--------|---------|
| 1.1 | 2026-05-31 | itej89 | Add Phase 1 requirements: ConnectFour env, MCTS, Arena |
| 1.0 | 2026-05-30 | itej89 | Initial draft — Phase 0 scope |

---

## 1. Introduction

### 1.1 Purpose

This document specifies the software requirements for `rl_framework`, a reusable
Python library for building and experimenting with reinforcement learning agents.
It is the contractual reference for design, implementation, and verification across
all project phases (Phase 0 through Phase 5).

**Intended audience**: The engineer implementing the library, tech leads reviewing PRs,
and anyone writing experiments that depend on this library.

### 1.2 Scope

**Product name**: `rl_framework`

**What it does**: Provides abstract interfaces for environments, agents, search
algorithms, neural networks, and verifiers. Provides concrete implementations
for Phase 0: GridWorld environment, Q-table agent, REINFORCE agent, PPO agent,
and a generic train/eval loop.

**What it does NOT do**:
- Does not implement experiment scripts (those live in `experiments/`)
- Does not manage cluster compute or job scheduling
- Does not provide a web interface or REST API
- Does not implement Phase 1–5 components in this version

**Goals**:
- Provide a clean, stable Python API that experiment scripts depend on
- Enable new environments and agents to be added without modifying existing code
- Be fully type-annotated, linted, and tested

### 1.3 Definitions

| Term | Definition |
|------|-----------|
| MDP | Markov Decision Process — the mathematical framework for RL problems |
| State (s) | A complete description of the world at a point in time |
| Action (a) | A choice the agent makes in a given state |
| Reward (r) | A scalar signal the environment returns after each step |
| Episode | A complete sequence from env.reset() to a terminal state |
| Return (G) | Discounted sum of rewards: G_t = Σ γ^k · r_{t+k} |
| Policy (π) | A mapping from states to actions (or action probabilities) |
| Q-value | Q(s,a) — expected return starting from s, taking a, then following π |
| TD error | r + γ·max_a' Q(s',a') − Q(s,a) — the Bellman residual |
| ε-greedy | Exploration strategy: random action with prob ε, greedy otherwise |

### 1.4 References

| ID | Title | Version | Date |
|----|-------|---------|------|
| IEEE-830 | IEEE Std 830-1998 — SRS Recommended Practice | 1998 | 1998-10-20 |
| SWG-001 | Software Development Guidelines | v3.9 | 2026-05-30 |
| Sutton-Barto | Reinforcement Learning: An Introduction | 2nd ed. | 2018 |
| Schulman-PPO | Proximal Policy Optimization Algorithms | arXiv:1707.06347 | 2017 |

### 1.5 Overview

Section 2 describes the library in system context.
Section 3 contains all functional and non-functional requirements.
Section 4 lists open issues.
Appendix A provides the traceability matrix.

---

## 2. Overall Description

### 2.1 Product Perspective

`rl_framework` is a standalone Python library. It is not part of a larger system
but is the foundation that all experiment scripts (in `experiments/`) import.

```
experiments/
  p0_e1_gridworld_qlearning.py   ←→   rl_framework.envs.GridWorld
                                  ←→   rl_framework.agents.QTableAgent
                                  ←→   rl_framework.train / eval
```

**Inputs from**: Experiment scripts (function calls)
**Outputs to**: Metrics dictionaries, trained agent objects
**Operated by**: Python experiment scripts, either locally or via cluster Docker containers

### 2.2 Product Functions

1. **Environment abstraction**: Define a standard interface all environments implement
2. **GridWorld environment**: Concrete 4×4 grid environment for Phase 0
3. **Agent abstraction**: Define a standard interface all agents implement
4. **Q-table agent**: Tabular Q-learning with ε-greedy exploration
5. **REINFORCE agent**: Policy gradient with value baseline (Phase 0, E2)
6. **PPO agent**: Proximal Policy Optimization (Phase 0, E3)
7. **Training loop**: Generic episode-based training that works with any env+agent pair
8. **Evaluation loop**: Greedy rollout for measuring agent performance

### 2.3 User Classes

| User Class | Description | Technical Level |
|---|---|---|
| Experiment author | Writes scripts in `experiments/` using this library | Intermediate Python |
| Library contributor | Adds new envs, agents, or algorithms | Advanced Python + RL knowledge |

### 2.4 Operating Environment

| Attribute | Value |
|-----------|-------|
| OS | Linux (Ubuntu 22.04+) |
| Runtime | Python 3.11+ |
| Hardware | CPU (Phase 0); AMD MI300X GPU via ROCm (Phase 1+) |
| Key dependencies | numpy, torch, gymnasium |

### 2.5 Design and Implementation Constraints

- Language: Python 3.11+
- Package layout: `src/` layout (SWG-001 §7)
- Type annotations required on all public functions (SWG-001 §14)
- Functions ≤ 20 lines (SWG-001 §5.2.1)
- No circular imports between modules
- `__all__` declared in every `__init__.py`

### 2.6 Assumptions and Dependencies

**Assumptions**:
- Experiment scripts always call `env.reset()` before the first `env.step()`
- Reward values are finite scalars
- Episode lengths are finite (environments always reach a terminal state)

**External dependencies**:
- `numpy` — array operations for Q-tables and embeddings
- `torch` — neural networks for REINFORCE and PPO
- `gymnasium` — optional; used as reference for env interface design

---

## 3. Specific Requirements

### 3.1 External Interface Requirements

#### 3.1.1 Python API

The public API is imported as:

```python
from rl_framework.envs import GridWorld
from rl_framework.agents import QTableAgent, ReinforceAgent, PPOAgent
from rl_framework import train, eval
```

All public classes and functions are declared in their module's `__all__`.

#### 3.1.2 Hardware Interfaces

Phase 0 runs entirely on CPU. No GPU requirement.

---

### 3.2 Functional Requirements

#### 3.2.1 Environment Interface (BaseEnv)

```
RLF-ENV-001  The system SHALL define a BaseEnv abstract base class with the
             following abstract methods: reset() -> int, step(action: int) ->
             tuple[int, float, bool, dict], render() -> str.

Priority:   Critical
Status:     Draft
Rationale:  All agents and the train loop depend on this interface. Changing
            the interface breaks all downstream code — define it once, clearly.
```

```
RLF-ENV-002  BaseEnv SHALL declare read-only properties n_states: int and
             n_actions: int that concrete environments must implement.

Priority:   Critical
Status:     Draft
Rationale:  Agents use n_states and n_actions to size their internal structures
            (e.g., Q-table shape = (n_states, n_actions)).
```

```
RLF-ENV-003  env.step(action) SHALL return a named tuple or plain tuple of
             exactly four elements: (next_state: int, reward: float,
             done: bool, info: dict).

Priority:   Critical
Status:     Draft
Rationale:  The train loop unpacks exactly four values. Inconsistent returns
            cause silent bugs.
```

```
RLF-ENV-004  env.reset() SHALL return the initial state as an int and reset
             all internal episode state.

Priority:   Critical
Status:     Draft
Rationale:  Each episode starts fresh; state from a previous episode must not
            leak into the next.
```

#### 3.2.2 Two-Player Game Environment Interface

```
RLF-ENV-020  Two-player game environments SHALL expose: reset() → board state,
             step(action) → (next_state, reward, done, info), legal_actions() →
             list[int], and current_player() → int (0 or 1).

Priority:   Critical
Status:     Approved
Rationale:  MCTS and self-play loops need to know whose turn it is and which
            moves are legal without inspecting board internals.
```

```
RLF-ENV-021  reset() SHALL randomise which player moves first when
             random_start=True (default False).

Priority:   Low
Status:     Approved
Rationale:  Self-play training benefits from both players seeing both colours.
```

#### 3.2.3 ConnectFour Environment

```
RLF-CF-001  ConnectFour SHALL model a 6-row × 7-column board. Pieces fall to
            the lowest empty row in the chosen column.

Priority:   Critical
Status:     Approved
Rationale:  Official Connect Four rules; board shape determines all win-check logic.
```

```
RLF-CF-002  ConnectFour SHALL detect wins horizontally, vertically, and along
            both diagonals (\ and /). A win for the player who just moved SHALL
            return reward=+1.0, done=True from step().

Priority:   Critical
Status:     Approved
Rationale:  All four win directions must be checked; missing one is a silent bug.
```

```
RLF-CF-003  A full board with no winner SHALL return reward=0.0, done=True (draw).

Priority:   Critical
Status:     Approved
Rationale:  Draw is a valid terminal state; treating it as a win causes incorrect
            MCTS backpropagation.
```

```
RLF-CF-004  step() called with an illegal action (full column) SHALL raise
            EnvironmentError.

Priority:   High
Status:     Approved
Rationale:  Silent illegal moves corrupt board state silently; fail-fast is safer.
```

```
RLF-CF-005  legal_actions() SHALL return exactly the columns that are not full,
            in ascending column order.

Priority:   Critical
Status:     Approved
Rationale:  MCTS expansion iterates legal_actions(); incorrect set causes invalid
            tree branches.
```

```
RLF-CF-006  render() SHALL return a human-readable string with '.' for empty,
            'X' for player 0, 'O' for player 1, and column indices on the bottom
            row.

Priority:   Low
Status:     Approved
Rationale:  Debugging aid; exact format is not contractual but must be non-empty.
```

#### 3.2.4 MCTS — Monte Carlo Tree Search

```
RLF-MCTS-001  MCTSNode SHALL store: visit_count (N), total_value (W), children
              dict mapping action→MCTSNode, and a reference to its parent.

Priority:   Critical
Status:     Approved
Rationale:  UCB1, backprop, and best-child selection all require N and W; parent
            pointer is needed for the backpropagation walk.
```

```
RLF-MCTS-002  UCB1 selection SHALL use score = W/N + C·√(ln N_parent / N).
              Unvisited children (N=0) SHALL always be selected before visited
              ones (treated as score=+∞).

Priority:   Critical
Status:     Approved
Rationale:  Standard UCB1 formula (Kocsis & Szepesvári, 2006); unvisited nodes
            must be explored before exploitation begins.
```

```
RLF-MCTS-003  Each simulation SHALL follow the sequence: selection → expansion
              (add one unvisited child) → random rollout → backpropagation.

Priority:   Critical
Status:     Approved
Rationale:  The four-phase structure is the defining property of MCTS; deviating
            from it changes the algorithm's convergence properties.
```

```
RLF-MCTS-004  Backpropagation SHALL negate the value at each step (value for
              current player = -value for opponent) to correctly propagate
              zero-sum outcomes.

Priority:   Critical
Status:     Approved
Rationale:  Connect Four is zero-sum; a win for player 0 is a loss for player 1.
            Without negation, both players maximise the same objective.
```

```
RLF-MCTS-005  mcts_action(env, n_simulations, c) SHALL return the action
              corresponding to the child of root with the highest visit count N.

Priority:   Critical
Status:     Approved
Rationale:  Highest-N child is the standard MCTS policy (more robust than
            highest-W/N which is noisier at low counts).
```

#### 3.2.5 Arena

```
RLF-ARENA-001  arena(agent_a, agent_b, env, n_games) SHALL play n_games games,
               alternating which agent moves first each game, and return a dict
               with keys wins_a, wins_b, draws.

Priority:   High
Status:     Approved
Rationale:  Alternating first-move removes first-player advantage bias from
            win-rate estimates.
```

```
RLF-ARENA-002  MCTS(500 simulations) SHALL win ≥ 80% of 100 games against a
               random agent. arena() SHALL complete those 100 games in under 60
               seconds on a single CPU core.

Priority:   High
Status:     Approved
Rationale:  Pure MCTS with random rollouts plateaus ~80-85% at 500 sims; ≥95%
            requires a learned value function (Phase 1 E2). The 60s budget
            makes iterative self-play training practical.
```

#### 3.2.2 GridWorld Environment

```
RLF-ENV-010  GridWorld SHALL be a 4×4 deterministic grid with 16 states
             numbered 0–15 in row-major order (top-left=0, bottom-right=15).

Priority:   Critical
Status:     Draft
Rationale:  Phase 0-E1 experiment; must match the analytical example in lesson
            material so results are verifiable by hand.
```

```
RLF-ENV-011  GridWorld SHALL support 4 actions: 0=up, 1=down, 2=left, 3=right.
             An action that would move the agent off the grid SHALL leave the
             agent in its current cell.

Priority:   Critical
Status:     Draft
Rationale:  Boundary handling is a classic source of off-by-one bugs; explicit
            requirement makes testing unambiguous.
```

```
RLF-ENV-012  GridWorld SHALL have one goal state (cell 15, reward=+1.0, done=True)
             and two pit states (cells 5 and 11, reward=-1.0, done=True).
             All other transitions SHALL yield reward=0.0, done=False.

Priority:   Critical
Status:     Draft
Rationale:  Defines the reward structure that Q-learning must learn to navigate.
```

```
RLF-ENV-013  GridWorld.render() SHALL return a string representation of the
             4×4 grid showing the agent position (A), goal (G), and pits (X).

Priority:   Medium
Status:     Draft
Rationale:  Human-readable debugging during development and teaching.
```

#### 3.2.3 Agent Interface (BaseAgent)

```
RLF-AGT-001  The system SHALL define a BaseAgent abstract base class with the
             following abstract methods: select_action(state: int) -> int,
             update(state, action, reward, next_state, done) -> None.

Priority:   Critical
Status:     Draft
Rationale:  The train loop calls these two methods and nothing else. Defining
            the interface here decouples train loop from any specific algorithm.
```

```
RLF-AGT-002  BaseAgent.__init__ SHALL accept n_states: int, n_actions: int,
             and a config object, so the train loop can construct any agent
             uniformly.

Priority:   High
Status:     Draft
Rationale:  Enables the train loop to instantiate agents generically without
            knowing the concrete class.
```

#### 3.2.4 Q-Table Agent

```
RLF-AGT-010  QTableAgent SHALL store Q-values in a numpy array of shape
             (n_states, n_actions) initialised to zero.

Priority:   Critical
Status:     Draft
Rationale:  Zero initialisation is the standard for tabular Q-learning; it is
            optimistic under negative rewards and neutral under zero rewards.
```

```
RLF-AGT-011  QTableAgent.select_action SHALL implement ε-greedy exploration:
             with probability ε return a uniformly random action; otherwise
             return argmax_a Q(state, a).

Priority:   Critical
Status:     Draft
Rationale:  ε-greedy is the standard exploration strategy for tabular RL.
            Without exploration the agent never discovers better actions.
```

```
RLF-AGT-012  QTableAgent.update SHALL implement the Q-learning Bellman update:
             Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') · (1-done) − Q(s,a)]
             using learning rate α and discount factor γ from config.

Priority:   Critical
Status:     Draft
Rationale:  This is the core algorithm. The (1-done) term correctly zeroes
            the future value at terminal states.
```

```
RLF-AGT-013  QTableAgent SHALL decay ε after each episode by multiplying it
             by config.epsilon_decay, with a minimum floor of config.epsilon_min.

Priority:   High
Status:     Draft
Rationale:  Decaying ε shifts the agent from exploration early in training to
            exploitation later, which is required for convergence.
```

#### 3.2.5 Training Loop

```
RLF-TRN-001  train(env, agent, config) SHALL run config.n_episodes episodes,
             where each episode begins with env.reset() and ends when done=True
             or the step count exceeds config.max_steps_per_episode.

Priority:   Critical
Status:     Draft
Rationale:  The outer loop is identical for all agents; factoring it out avoids
            code duplication across every experiment.
```

```
RLF-TRN-002  train() SHALL call agent.select_action(state) and agent.update(
             state, action, reward, next_state, done) on every step.

Priority:   Critical
Status:     Draft
Rationale:  These are the only two agent methods the train loop needs; using
            only BaseAgent interface methods keeps train() algorithm-agnostic.
```

```
RLF-TRN-003  train() SHALL return a list of episode metrics dicts, each
             containing at minimum: {"episode": int, "total_reward": float,
             "steps": int, "epsilon": float}.

Priority:   High
Status:     Draft
Rationale:  Experiment scripts use these metrics to plot learning curves and
            verify convergence. Consistent schema enables shared plotting code.
```

#### 3.2.6 Evaluation Loop

```
RLF-EVL-001  eval(env, agent, n_episodes) SHALL run n_episodes episodes with
             agent.select_action in greedy mode (ε=0, no exploration).

Priority:   High
Status:     Draft
Rationale:  Training metrics are noisy due to exploration. Evaluation under
            greedy policy gives the true current policy performance.
```

```
RLF-EVL-002  eval() SHALL return a dict containing mean_return: float,
             std_return: float, and mean_steps: float computed over all
             n_episodes evaluation episodes.

Priority:   High
Status:     Draft
Rationale:  Mean ± std return is the standard way to report RL performance.
```

---

### 3.3 Non-Functional Requirements

```
RLF-NFR-001  Unit test line coverage SHALL be ≥ 90% on all source files
             under src/rl_framework/.

Priority:   High
Status:     Draft
Rationale:  SWG-001 §6.4 — coverage requirement for library code.
```

```
RLF-NFR-002  All public functions and methods SHALL carry full type annotations
             on parameters and return values. mypy --strict SHALL pass with
             zero errors.

Priority:   High
Status:     Draft
Rationale:  SWG-001 §14 — type annotations catch interface mismatches at
            development time rather than at runtime.
```

```
RLF-NFR-003  No production function SHALL exceed 20 lines of code (excluding
             blank lines and docstrings). ruff SHALL be the enforced linter.

Priority:   High
Status:     Draft
Rationale:  SWG-001 §5.2.1 — small functions are easier to test and understand.
```

```
RLF-NFR-004  Unit tests SHALL complete in under 2 seconds total on a CPU-only
             machine (no GPU required for Phase 0 tests).

Priority:   Medium
Status:     Draft
Rationale:  SWG-001 §6.4.1 T9 — fast tests encourage frequent test runs.
```

```
RLF-NFR-005  The library SHALL have no circular imports. Running
             python -c 'import rl_framework' SHALL succeed with zero warnings.

Priority:   High
Status:     Draft
Rationale:  Circular imports cause intermittent import failures that are
            difficult to debug.
```

---

## 4. Open Issues and Risks

| ID | Description | Owner | Target Resolution |
|----|-------------|-------|------------------|
| OI-001 | REINFORCE and PPO agents (Phase 0 E2/E3) — requirements to be added in SRS v1.1 | itej89 | Before issue #3 starts |
| OI-002 | Gymnasium compatibility: should BaseEnv mirror gymnasium.Env interface exactly? | itej89 | ADR-001 |

---

## Appendices

### Appendix A — Traceability Matrix

| Req ID | Module | Test File | Test Name | Status |
|--------|--------|-----------|-----------|--------|
| RLF-ENV-001 | envs/base.py | tests/unit/test_gridworld.py | test_gridworld_implements_base_env | Not written |
| RLF-ENV-002 | envs/base.py | tests/unit/test_gridworld.py | test_gridworld_n_states_is_16 | Not written |
| RLF-ENV-003 | envs/gridworld.py | tests/unit/test_gridworld.py | test_step_returns_four_elements | Not written |
| RLF-ENV-004 | envs/gridworld.py | tests/unit/test_gridworld.py | test_reset_returns_initial_state | Not written |
| RLF-ENV-010 | envs/gridworld.py | tests/unit/test_gridworld.py | test_gridworld_has_16_states | Not written |
| RLF-ENV-011 | envs/gridworld.py | tests/unit/test_gridworld.py | test_move_off_grid_stays_in_place | Not written |
| RLF-ENV-012 | envs/gridworld.py | tests/unit/test_gridworld.py | test_goal_state_gives_positive_reward | Not written |
| RLF-ENV-012 | envs/gridworld.py | tests/unit/test_gridworld.py | test_pit_state_gives_negative_reward | Not written |
| RLF-ENV-013 | envs/gridworld.py | tests/unit/test_gridworld.py | test_render_returns_string | Not written |
| RLF-AGT-001 | agents/base.py | tests/unit/test_q_table.py | test_q_table_implements_base_agent | Not written |
| RLF-AGT-010 | agents/q_table.py | tests/unit/test_q_table.py | test_q_table_initialised_to_zero | Not written |
| RLF-AGT-011 | agents/q_table.py | tests/unit/test_q_table.py | test_select_action_explores_when_epsilon_is_one | Not written |
| RLF-AGT-011 | agents/q_table.py | tests/unit/test_q_table.py | test_select_action_exploits_when_epsilon_is_zero | Not written |
| RLF-AGT-012 | agents/q_table.py | tests/unit/test_q_table.py | test_update_moves_q_value_toward_target | Not written |
| RLF-AGT-012 | agents/q_table.py | tests/unit/test_q_table.py | test_update_zeroes_future_value_at_terminal | Not written |
| RLF-AGT-013 | agents/q_table.py | tests/unit/test_q_table.py | test_epsilon_decays_after_each_episode | Not written |
| RLF-TRN-001 | train.py | tests/unit/test_train.py | test_train_runs_correct_number_of_episodes | Not written |
| RLF-TRN-003 | train.py | tests/unit/test_train.py | test_train_returns_metrics_list | Not written |
| RLF-EVL-001 | eval.py | tests/unit/test_eval.py | test_eval_runs_greedy_policy | Not written |
| RLF-EVL-002 | eval.py | tests/unit/test_eval.py | test_eval_returns_mean_and_std | Not written |

### Appendix B — Supporting Diagrams

| Diagram | Location |
|---------|----------|
| Class diagram | `docs/uml/class-diagram.md` |
| Sequence diagram | `docs/uml/sequence-diagram.md` |
| Component diagram | `docs/uml/component-diagram.md` |
