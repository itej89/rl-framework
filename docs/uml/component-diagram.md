# Component Diagram

> Last updated: 2026-05-30 — reflects SRS-RLF-v1.0 Phase 0 scope

---

```
┌─────────────────────────────────────────────────────────────────┐
│                        experiments/                             │
│   p0_e1_gridworld_qlearning.py                                  │
│   p0_e2_cartpole_reinforce.py                                   │
│   p0_e3_lunarlander_ppo.py                                      │
└─────────────────┬──────────────────────────────────────────────┘
                  │  imports
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     rl_framework  (src/)                        │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   │
│  │   envs/      │   │   agents/    │   │  train.py        │   │
│  │              │   │              │   │  eval.py         │   │
│  │  BaseEnv     │   │  BaseAgent   │   │                  │   │
│  │  GridWorld   │   │  QTableAgent │   │  train(env,      │   │
│  │              │   │  Reinforce   │   │    agent, cfg)   │   │
│  │              │   │  PPOAgent    │   │  eval(env,       │   │
│  └──────┬───────┘   └──────┬───────┘     agent, n)       │   │
│         │                  │           └──────────────────┘   │
│         └──────────────────┘                  │               │
│                  │ uses                        │ uses          │
│                  ▼                             ▼               │
│          ┌───────────────────────────────────────────┐        │
│          │          exceptions.py                     │        │
│          │   RLFrameworkError                         │        │
│          │   EnvironmentError / AgentError            │        │
│          └───────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

**Dependency rules (no violations allowed):**
- `experiments/` → `rl_framework` (one direction only)
- `train.py` → `envs/`, `agents/` (via BaseEnv / BaseAgent interfaces)
- `agents/` → NO dependency on `envs/` (agents are env-agnostic)
- `envs/` → NO dependency on `agents/`
