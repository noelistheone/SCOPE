"""Early stopping helper used by the Trainer."""

from __future__ import annotations

import math
from typing import Literal


class EarlyStopper:
    """Patience-based early stopper.

    Parameters
    ----------
    patience : int
        Stop after this many evaluations without improvement.
    mode : "max" or "min"
        Direction of improvement.
    min_delta : float
        Minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int = 20,
                 mode: Literal["max", "min"] = "max",
                 min_delta: float = 0.0) -> None:
        if patience < 1:
            raise ValueError(f"patience must be >= 1 (got {patience})")
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min' (got {mode})")
        self.patience = patience
        self.mode = mode
        self.min_delta = abs(min_delta)
        self.best: float = -math.inf if mode == "max" else math.inf
        self.counter: int = 0
        self.should_stop: bool = False
        self.improved: bool = False

    def step(self, value: float) -> bool:
        """Record a new metric. Returns True if it was an improvement."""
        if math.isnan(value):
            # NaN never improves; count toward patience.
            self.improved = False
            self.counter += 1
            self.should_stop = self.counter >= self.patience
            return False

        if self.mode == "max":
            improved = value > self.best + self.min_delta
        else:
            improved = value < self.best - self.min_delta

        if improved:
            self.best = value
            self.counter = 0
            self.improved = True
        else:
            self.counter += 1
            self.improved = False

        self.should_stop = self.counter >= self.patience
        return improved

    def reset(self) -> None:
        self.best = -math.inf if self.mode == "max" else math.inf
        self.counter = 0
        self.should_stop = False
        self.improved = False
