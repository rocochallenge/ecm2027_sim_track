"""Random-seed helpers with no task or asset-registry side effects."""

import random

import numpy as np
import torch


def set_seed(seed, env=None, torch_deterministic: bool = True):
    """Seed Python, NumPy, Torch, and an optional IsaacLab environment."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        if env is not None:
            env.seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


__all__ = ["set_seed"]
