"""Common loss functions: BPR, L2 regularization, InfoNCE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking loss.

    L = -mean(log_sigmoid(pos_score - neg_score))
    """

    def __init__(self, gamma: float = 1.0e-10) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, pos_score: torch.Tensor,
                neg_score: torch.Tensor) -> torch.Tensor:
        # Use logsigmoid for numerical stability (avoids log(sigmoid(x)) → -inf).
        return -F.logsigmoid(pos_score - neg_score + self.gamma).mean()


class EmbLoss(nn.Module):
    """L2 regularization on a list of embedding tensors.

    Normalizes by batch size (first dim of the first tensor) like MMRec's reg.
    """

    def __init__(self, norm: int = 2) -> None:
        super().__init__()
        if norm not in (1, 2):
            raise ValueError(f"norm must be 1 or 2 (got {norm})")
        self.norm = norm

    def forward(self, *embeddings: torch.Tensor) -> torch.Tensor:
        if not embeddings:
            raise ValueError("EmbLoss called with no tensors")
        batch_size = embeddings[0].shape[0]
        if batch_size == 0:
            return embeddings[0].new_zeros(())
        total = embeddings[0].new_zeros(())
        for emb in embeddings:
            total = total + torch.norm(emb, p=self.norm).pow(self.norm)
        return total / batch_size


class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE loss for contrastive learning.

    Given paired views z1, z2 of N items, treat (z1_i, z2_i) as positives and
    all other (z1_i, z2_j) with j != i as negatives. Returns the average of
    both directions for symmetry.
    """

    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0 (got {temperature})")
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if z1.shape != z2.shape:
            raise ValueError(
                f"z1 and z2 must have same shape (got {z1.shape} vs {z2.shape})")
        if z1.dim() != 2:
            raise ValueError(f"Expected 2D inputs, got {z1.dim()}D")
        n = z1.size(0)
        if n == 0:
            return z1.new_zeros(())

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = z1 @ z2.t() / self.temperature
        labels = torch.arange(n, device=z1.device)
        # Symmetric: average both rows->cols and cols->rows directions.
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_a + loss_b)
