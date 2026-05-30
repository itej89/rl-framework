"""rl_framework — reusable reinforcement learning library.

Public API:
    from rl_framework.envs import GridWorld
    from rl_framework.agents import QTableAgent
    from rl_framework.config import TrainConfig
    from rl_framework.train import train
    from rl_framework.eval import eval
"""

from rl_framework.config import TrainConfig
from rl_framework.eval import eval
from rl_framework.train import train

__all__ = ["TrainConfig", "train", "eval"]
__version__ = "0.1.0"
