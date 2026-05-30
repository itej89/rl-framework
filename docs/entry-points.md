# Entry Point Registry

> Last updated: 2026-05-30 — SRS-RLF-v1.0

`rl_framework` is a library — it has no CLI entry points or long-running processes.
It is invoked exclusively by experiment scripts.

## Library Import Entry Point

| Attribute | Value |
|-----------|-------|
| Trigger | `import rl_framework` from an experiment script |
| Inputs | None (library import) |
| Outputs | Exposes public API per `__all__` in each module |
| Error behaviour | `ImportError` if dependencies missing; `ConfigurationError` if env invalid |

## Experiment Scripts (consumers of this library)

| Script | Entry point | Cluster script |
|--------|------------|----------------|
| `experiments/p0_e1_gridworld_qlearning.py` | `python experiments/p0_e1_gridworld_qlearning.py` | `cluster/run_p0_e1_gridworld.sh` |
| `experiments/p0_e2_cartpole_reinforce.py` | `python experiments/p0_e2_cartpole_reinforce.py` | TBD |
| `experiments/p0_e3_lunarlander_ppo.py` | `python experiments/p0_e3_lunarlander_ppo.py` | TBD |

## Environment Variables

None required for Phase 0. Phase 1+ cluster scripts use:

| Variable | Required by | Purpose |
|----------|------------|---------|
| `CACHE_DIR` | Phase 1+ | Path to embedding / checkpoint cache |
| `OUTPUT_DIR` | Phase 1+ | Path to results output |
