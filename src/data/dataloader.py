"""Train and evaluation dataloaders.

Train side: uniform negative sampling with rejection; capped retries.
Eval side: yields user chunks with their training history (for masking)
and ground-truth positives.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import RecDataset

LOGGER = logging.getLogger("recsys.data")


# =====================================================================
# Training: per-interaction (user, pos_item, neg_item) triples
# =====================================================================


class _TrainTripletDataset(Dataset):
    """Indexable dataset of training interactions. Negative sampling is done
    in ``__getitem__`` (each DataLoader worker holds its own RNG)."""

    def __init__(self,
                 rec: RecDataset,
                 max_tries: int = 100) -> None:
        self.n_items = rec.n_items
        self.users = rec.train_users
        self.items = rec.train_items
        self.user_pos = rec.user_pos_dict
        if max_tries < 1:
            raise ValueError(f"max_tries must be >= 1 (got {max_tries})")
        self.max_tries = int(max_tries)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> tuple[int, int, int]:
        u = int(self.users[idx])
        i_pos = int(self.items[idx])
        pos_set = self.user_pos.get(u, ())
        # Per-process RNG; numpy's default_rng would re-seed across workers.
        for _ in range(self.max_tries):
            i_neg = int(np.random.randint(0, self.n_items))
            if i_neg not in pos_set:
                return u, i_pos, i_neg
        # Fallback: accept the last draw even if a collision (extremely rare).
        return u, i_pos, i_neg


def _train_collate(batch: list[tuple[int, int, int]]) -> dict[str, torch.Tensor]:
    users = torch.tensor([b[0] for b in batch], dtype=torch.long)
    pos = torch.tensor([b[1] for b in batch], dtype=torch.long)
    neg = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return {"user": users, "pos_item": pos, "neg_item": neg}


class TrainDataLoader(DataLoader):
    """DataLoader yielding {'user', 'pos_item', 'neg_item'} batches.

    Wraps ``_TrainTripletDataset`` with safe defaults: bounded workers, pinned
    memory when CUDA is available, persistent workers to amortize startup cost.
    """

    def __init__(self,
                 rec: RecDataset,
                 batch_size: int,
                 num_workers: int = 4,
                 pin_memory: bool | None = None,
                 max_neg_tries: int = 100,
                 shuffle: bool = True) -> None:
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        num_workers = max(0, int(num_workers))
        ds = _TrainTripletDataset(rec, max_tries=max_neg_tries)
        super().__init__(
            dataset=ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=_train_collate,
            persistent_workers=(num_workers > 0),
            drop_last=False,
        )


# =====================================================================
# Evaluation: chunked user batches with training-history mask
# =====================================================================


class EvalDataLoader:
    """Iterable yielding eval chunks.

    Each chunk is a dict with:
        user_ids: LongTensor [B]
        history_indices: LongTensor [B, max_hist]  (-1 padding)
        history_values:  LongTensor [B, max_hist]  (1 valid / 0 pad)
        positive_items:  list[np.ndarray]          length B (variable)
        positive_lengths: LongTensor [B]
    """

    def __init__(self,
                 rec: RecDataset,
                 phase: str,
                 batch_size: int = 1024) -> None:
        if phase not in ("valid", "test"):
            raise ValueError(f"phase must be 'valid' or 'test' (got {phase!r})")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1 (got {batch_size})")
        self.rec = rec
        self.batch_size = int(batch_size)
        eval_map = (rec.valid_user_to_items if phase == "valid"
                    else rec.test_user_to_items)
        # Eval users are those who have at least one positive in this split.
        self.user_ids: np.ndarray = np.array(
            sorted(eval_map.keys()), dtype=np.int64)
        self.user_to_pos = eval_map
        self.user_pos_train = rec.user_pos_dict
        self.phase = phase

    def __len__(self) -> int:
        # Number of batches.
        return (len(self.user_ids) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[dict[str, Any]]:
        n = len(self.user_ids)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            users = self.user_ids[start:end]

            # Training history (to mask during scoring).
            hist_lists = [list(self.user_pos_train.get(int(u), ())) for u in users]
            max_hist = max((len(h) for h in hist_lists), default=0)
            if max_hist == 0:
                history_indices = torch.zeros((len(users), 0), dtype=torch.long)
                history_values = torch.zeros((len(users), 0), dtype=torch.long)
            else:
                history_indices = torch.full(
                    (len(users), max_hist), -1, dtype=torch.long)
                history_values = torch.zeros(
                    (len(users), max_hist), dtype=torch.long)
                for r, hl in enumerate(hist_lists):
                    if hl:
                        history_indices[r, :len(hl)] = torch.as_tensor(
                            hl, dtype=torch.long)
                        history_values[r, :len(hl)] = 1

            # Ground-truth positives (variable length per user).
            positive_items = [self.user_to_pos[int(u)] for u in users]
            positive_lengths = torch.tensor(
                [len(p) for p in positive_items], dtype=torch.long)

            yield {
                "user_ids": torch.from_numpy(users),
                "history_indices": history_indices,
                "history_values": history_values,
                "positive_items": positive_items,
                "positive_lengths": positive_lengths,
            }
