# Class Diagram

> Last updated: 2026-05-30 — reflects SRS-RLF-v1.0 Phase 0 scope

---

```
┌──────────────────────────────────────┐
│  <<abstract>>                        │
│  BaseEnv                             │
│──────────────────────────────────────│
│  + n_states: int  (abstract)         │
│  + n_actions: int (abstract)         │
│──────────────────────────────────────│
│  + reset() -> int               (abs)│
│  + step(action: int)                 │
│      -> tuple[int,float,bool,dict]   │
│                                 (abs)│
│  + render() -> str              (abs)│
└──────────────────┬───────────────────┘
                   │ inherits
                   ▼
┌──────────────────────────────────────┐
│  GridWorld                           │
│──────────────────────────────────────│
│  + n_states: int = 16               │
│  + n_actions: int = 4               │
│  - _state: int                       │
│  - _GOAL: int = 15                   │
│  - _PITS: frozenset = {5, 11}        │
│──────────────────────────────────────│
│  + reset() -> int                    │
│  + step(action) -> tuple             │
│  + render() -> str                   │
│  - _next_state(state, action) -> int │
└──────────────────────────────────────┘


┌──────────────────────────────────────┐
│  <<abstract>>                        │
│  BaseAgent                           │
│──────────────────────────────────────│
│  + n_states: int                     │
│  + n_actions: int                    │
│──────────────────────────────────────│
│  + select_action(state: int)         │
│      -> int                     (abs)│
│  + update(state, action, reward,     │
│           next_state, done)          │
│      -> None                    (abs)│
│  + on_episode_end() -> None          │
└──────────────────┬───────────────────┘
                   │ inherits
                   ▼
┌──────────────────────────────────────┐
│  QTableAgent                         │
│──────────────────────────────────────│
│  - _q: ndarray  shape(n_s, n_a)      │
│  - _alpha: float  (learning rate)    │
│  - _gamma: float  (discount)         │
│  - _epsilon: float                   │
│  - _epsilon_decay: float             │
│  - _epsilon_min: float               │
│──────────────────────────────────────│
│  + select_action(state) -> int       │
│  + update(s,a,r,s',done) -> None     │
│  + on_episode_end() -> None          │
│  - _greedy_action(state) -> int      │
└──────────────────────────────────────┘


┌──────────────────────────────────────┐
│  TrainConfig  (dataclass)            │
│──────────────────────────────────────│
│  + n_episodes: int                   │
│  + max_steps_per_episode: int        │
│  + alpha: float                      │
│  + gamma: float                      │
│  + epsilon_start: float              │
│  + epsilon_decay: float              │
│  + epsilon_min: float                │
└──────────────────────────────────────┘
```
