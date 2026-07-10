"""Parameter initialization utilities."""

from __future__ import annotations

import torch.nn as nn


def xavier_uniform_initialization(module: nn.Module) -> None:
    """Apply Xavier-uniform init to Linear/Embedding weights.

    Use as ``model.apply(xavier_uniform_initialization)``.
    """
    if isinstance(module, nn.Embedding):
        nn.init.xavier_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)


def xavier_normal_initialization(module: nn.Module) -> None:
    """Apply Xavier-normal init to Linear/Embedding weights."""
    if isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
