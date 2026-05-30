# ADR-001 — src/ Layout and Abstract Interface Design

**Date:** 2026-05-30
**Status:** Accepted
**Deciders:** itej89

---

## Context

Two decisions must be made before writing any code:

1. **Package layout**: flat layout (`rl_framework/`) vs. `src/` layout (`src/rl_framework/`)
2. **Agent/env coupling**: Should agents know about environments, or communicate only through a defined interface?

### Option A: Flat layout
Simple. `import rl_framework` works immediately with no `pip install`.
Risk: tests can import the source directory directly, masking packaging errors.
Experiments running in Docker may fail even though local tests pass.

### Option B: src/ layout
Slightly more setup. Requires `pip install -e .` before use.
Benefit: tests always import the installed package, catching packaging problems early.
Standard for production Python libraries (SWG-001 §7).

### Option C: Tight coupling (agents call env methods)
Simple but fragile. Changing GridWorld breaks all agents.

### Option D: Abstract interfaces (BaseEnv, BaseAgent)
Agents only call `select_action` / `update` via the BaseAgent interface.
Train loop only calls `reset` / `step` via the BaseEnv interface.
Adding a new env (TicTacToe in Phase 2) requires zero changes to agents or train loop.

---

## Decision

**Use `src/` layout** (Option B).
**Use abstract interfaces** (Option D) for both environments and agents.

---

## Consequences

**Positive**:
- `src/` layout catches packaging errors before they reach the cluster
- Abstract interfaces mean Phase 1–5 additions never break Phase 0 code
- Train loop is genuinely algorithm-agnostic — the same function runs Q-learning, PPO, and AlphaZero

**Negative**:
- `pip install -e .` required before running anything — minor friction for new contributors
- Abstract base classes add a small amount of boilerplate

**Neutral**:
- `gymnasium.Env` interface was considered as the base. Decision: NOT adopted. Gymnasium's
  `obs_space` / `act_space` objects are overkill for Phase 0 tabular envs. We define our own
  minimal interface and add Gymnasium compatibility in a later phase if needed (OI-002).
