"""Deterministic seeding for reproducibility."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all RNGs the framework touches.

    With ``deterministic=True`` we also force cuDNN deterministic mode and
    enable ``torch.use_deterministic_algorithms`` with ``warn_only=True``
    (some sparse ops have no deterministic kernel; we warn rather than crash).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # CUBLAS workspace config is required for full determinism on CUDA >= 10.2
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (RuntimeError, AttributeError):
            # Older torch or unavailable backend. Best-effort only.
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
