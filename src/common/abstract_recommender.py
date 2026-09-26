"""Abstract base classes for recommender models.

Hierarchy:
    nn.Module
      └── AbstractRecommender         (defines calculate_loss / predict / full_sort_predict)
          └── GeneralRecommender      (adds n_users, n_items, device, embedding_size)
              └── GeneralGraphRecommender   (adds norm_adj, n_layers)
                  └── MultimodalRecommender (adds v_feat, t_feat, feat_embed_dim)

Models inherit from the most specific level they need. Concrete subclasses
must implement ``forward``, ``calculate_loss``, and ``full_sort_predict``.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn


class AbstractRecommender(nn.Module):
    """Common interface for every recommender model."""

    def __init__(self) -> None:
        super().__init__()

    # ---- methods subclasses must implement ----

    def calculate_loss(self, interaction: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return a scalar loss for the given training batch.

        ``interaction`` carries: 'user', 'pos_item', 'neg_item' (LongTensors).
        Subclass may return a tuple ``(total_loss, dict_of_components)`` if it
        wants per-component logging; the Trainer accepts either form.
        """
        raise NotImplementedError

    def predict(self, interaction: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Predict score for (user, item) pairs. Used only by pointwise evaluators."""
        raise NotImplementedError

    def full_sort_predict(self, interaction: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return a [batch_users, n_items] score matrix.

        Used by top-K evaluation. Implementations should NOT mask training items
        — the Trainer/evaluator handles that.
        """
        raise NotImplementedError

    # ---- optional hooks ----

    def pre_epoch_processing(self, epoch: int) -> None:
        """Hook called before each training epoch."""
        return None

    def post_epoch_processing(self, epoch: int) -> None:
        """Hook called after each training epoch."""
        return None

    def __str__(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (f"{self.__class__.__name__}("
                f"params={total:,}, trainable={trainable:,})")


class GeneralRecommender(AbstractRecommender):
    """CF baseline: stores user/item counts, embedding size, device."""

    def __init__(self,
                 config: Mapping[str, Any],
                 n_users: int,
                 n_items: int) -> None:
        super().__init__()
        if n_users <= 0 or n_items <= 0:
            raise ValueError(
                f"n_users and n_items must be positive (got {n_users}, {n_items})")
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.embedding_size = int(config.get("embedding_size", 64))
        self.reg_weight = float(config.get("reg_weight", 0.0))


class GeneralGraphRecommender(GeneralRecommender):
    """Graph CF baseline: adds a normalized adjacency matrix and depth."""

    def __init__(self,
                 config: Mapping[str, Any],
                 n_users: int,
                 n_items: int,
                 norm_adj: Optional[torch.Tensor] = None) -> None:
        super().__init__(config, n_users, n_items)
        self.n_layers = int(config.get("n_layers", 2))
        # norm_adj is a sparse COO tensor of shape [n_users+n_items, n_users+n_items].
        # Kept on CPU here; the Trainer moves the whole model to device once.
        self.register_buffer("norm_adj", norm_adj if norm_adj is not None else torch.empty(0))


class MultimodalRecommender(GeneralGraphRecommender):
    """Multimodal recommender: image + text features.

    Subclasses access ``self.v_feat`` / ``self.t_feat`` directly. Either may be
    ``None`` if the model is single-modality; concrete models that require both
    should assert their presence in ``__init__``.
    """

    def __init__(self,
                 config: Mapping[str, Any],
                 n_users: int,
                 n_items: int,
                 norm_adj: Optional[torch.Tensor] = None,
                 v_feat: Optional[torch.Tensor] = None,
                 t_feat: Optional[torch.Tensor] = None) -> None:
        super().__init__(config, n_users, n_items, norm_adj=norm_adj)
        self.feat_embed_dim = int(config.get("feat_embed_dim", 64))

        # Multimodal features. Registered as non-persistent buffers so they
        # ride along with the model on .to(device) but aren't saved in checkpoints
        # (they're regenerated from disk each run).
        if v_feat is not None:
            if v_feat.shape[0] != self.n_items:
                raise ValueError(
                    f"v_feat first dim {v_feat.shape[0]} != n_items {self.n_items}")
            self.register_buffer("v_feat", v_feat.float(), persistent=False)
            self.v_feat_dim = v_feat.shape[1]
        else:
            self.v_feat = None
            self.v_feat_dim = 0

        if t_feat is not None:
            if t_feat.shape[0] != self.n_items:
                raise ValueError(
                    f"t_feat first dim {t_feat.shape[0]} != n_items {self.n_items}")
            self.register_buffer("t_feat", t_feat.float(), persistent=False)
            self.t_feat_dim = t_feat.shape[1]
        else:
            self.t_feat = None
            self.t_feat_dim = 0

    def has_modalities(self) -> bool:
        return self.v_feat is not None or self.t_feat is not None
