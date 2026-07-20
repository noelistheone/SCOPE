"""Dataset loader for MMRec-style preprocessed data.

Expected on-disk layout under ``data/<dataset>/``:
    <dataset>.inter        TSV with columns: userID, itemID, x_label (0/1/2)
    image_feat.npy         float32 [n_items, D_v]    (often 4096)
    text_feat.npy          float32 [n_items, D_t]    (often 384)

The .inter file uses **already 0-indexed** integer IDs for users and items
(MMRec convention). We treat IDs as contiguous; gaps would still load but
``n_users`` / ``n_items`` is taken as ``max_id + 1`` so the embedding tables
size correctly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

LOGGER = logging.getLogger("scope.data")

_TRAIN_LABEL = 0
_VALID_LABEL = 1
_TEST_LABEL = 2


class RecDataset:
    """Loads .inter + multimodal features. CPU-only; the model moves tensors later.

    Attributes
    ----------
    n_users, n_items : int
    train_df, valid_df, test_df : pandas.DataFrame
        Each holds the rows of the corresponding split.
    train_matrix : scipy.sparse.csr_matrix
        Shape [n_users, n_items], 1 where a train interaction exists.
    user_pos_dict : dict[int, set[int]]
        For each user, the set of training positive items (for negative sampling).
    valid_user_to_items, test_user_to_items : dict[int, np.ndarray]
        Ground-truth items per user for eval.
    v_feat, t_feat : np.memmap | None
        Lazily mmap'd feature arrays; ``None`` if file is absent.
    """

    def __init__(self,
                 config: Mapping[str, Any],
                 project_root: Optional[Path] = None) -> None:
        self.config = config
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]
        self.project_root = project_root

        data_path = self.project_root / str(config["data_path"])
        self.data_path = data_path

        inter_file = data_path / str(config["inter_file_name"])
        if not inter_file.is_file():
            raise FileNotFoundError(
                f".inter file not found: {inter_file}. "
                "Did you run `python scripts/download_data.py`?")

        sep = config.get("inter_separator", "\t")
        user_field = str(config["user_id_field"])
        item_field = str(config["item_id_field"])
        split_field = str(config["split_field"])

        LOGGER.info(f"Loading interactions from {inter_file}")
        df = pd.read_csv(inter_file, sep=sep, engine="c")
        for col in (user_field, item_field, split_field):
            if col not in df.columns:
                raise KeyError(
                    f"Column {col!r} not in {inter_file}. Found: {list(df.columns)}")
        df = df.astype({user_field: np.int64,
                        item_field: np.int64,
                        split_field: np.int8})

        self.user_field = user_field
        self.item_field = item_field
        self.split_field = split_field

        self.n_users = int(df[user_field].max()) + 1
        self.n_items = int(df[item_field].max()) + 1
        LOGGER.info(f"n_users={self.n_users}, n_items={self.n_items}, "
                    f"n_interactions={len(df)}")

        self.train_df = df[df[split_field] == _TRAIN_LABEL].reset_index(drop=True)
        self.valid_df = df[df[split_field] == _VALID_LABEL].reset_index(drop=True)
        self.test_df = df[df[split_field] == _TEST_LABEL].reset_index(drop=True)

        self._build_train_matrix()
        self._build_eval_maps()
        self._load_features()

    # ------------------------------------------------------------------

    def _build_train_matrix(self) -> None:
        train = self.train_df
        rows = train[self.user_field].to_numpy(dtype=np.int64)
        cols = train[self.item_field].to_numpy(dtype=np.int64)
        vals = np.ones(len(train), dtype=np.float32)
        self.train_matrix: sp.csr_matrix = sp.coo_matrix(
            (vals, (rows, cols)), shape=(self.n_users, self.n_items)
        ).tocsr()
        # Deduplicate (multiple training interactions of same (u, i)).
        self.train_matrix.sum_duplicates()
        self.train_matrix.data[:] = 1.0

        # user_pos_dict for fast negative-sampling rejection.
        pos_dict: dict[int, set[int]] = {}
        for u, i in zip(rows.tolist(), cols.tolist()):
            pos_dict.setdefault(u, set()).add(i)
        self.user_pos_dict = pos_dict

        # Training (user, item) pair index for the DataLoader.
        self.train_users = rows
        self.train_items = cols

    def _build_eval_maps(self) -> None:
        def to_user_map(df: pd.DataFrame) -> dict[int, np.ndarray]:
            mapping: dict[int, list[int]] = {}
            for u, i in zip(df[self.user_field].to_numpy(dtype=np.int64).tolist(),
                            df[self.item_field].to_numpy(dtype=np.int64).tolist()):
                mapping.setdefault(u, []).append(i)
            return {u: np.array(items, dtype=np.int64)
                    for u, items in mapping.items()}

        self.valid_user_to_items = to_user_map(self.valid_df)
        self.test_user_to_items = to_user_map(self.test_df)

    def _load_features(self) -> None:
        v_path = self.data_path / str(self.config.get("vision_feature_file", ""))
        t_path = self.data_path / str(self.config.get("text_feature_file", ""))

        self.v_feat: Optional[np.ndarray] = None
        self.t_feat: Optional[np.ndarray] = None

        if v_path.is_file():
            arr = np.load(v_path, mmap_mode="r")
            if arr.shape[0] != self.n_items:
                LOGGER.warning(
                    f"v_feat n_items mismatch: feat has {arr.shape[0]}, "
                    f"interaction implies {self.n_items}. Using feat dim.")
            self.v_feat = arr
            LOGGER.info(f"Loaded v_feat: shape {arr.shape}, dtype {arr.dtype}")
        else:
            LOGGER.info(f"No v_feat at {v_path} (skipping visual modality)")

        if t_path.is_file():
            arr = np.load(t_path, mmap_mode="r")
            if arr.shape[0] != self.n_items:
                LOGGER.warning(
                    f"t_feat n_items mismatch: feat has {arr.shape[0]}, "
                    f"interaction implies {self.n_items}. Using feat dim.")
            self.t_feat = arr
            LOGGER.info(f"Loaded t_feat: shape {arr.shape}, dtype {arr.dtype}")
        else:
            LOGGER.info(f"No t_feat at {t_path} (skipping textual modality)")

    # ---- helpers ----

    def num_train_interactions(self) -> int:
        return len(self.train_users)

    def __repr__(self) -> str:
        return (f"RecDataset(name={self.config.get('dataset')!r}, "
                f"n_users={self.n_users}, n_items={self.n_items}, "
                f"n_train={len(self.train_df)}, n_valid={len(self.valid_df)}, "
                f"n_test={len(self.test_df)})")


def make_synthetic_dataset(n_users: int = 50,
                           n_items: int = 100,
                           n_inter: int = 500,
                           feat_dim_v: int = 16,
                           feat_dim_t: int = 8,
                           seed: int = 0) -> RecDataset:
    """Build a tiny in-memory dataset for smoke tests (no disk I/O)."""
    rng = np.random.default_rng(seed)
    users = rng.integers(0, n_users, size=n_inter)
    items = rng.integers(0, n_items, size=n_inter)
    labels = rng.integers(0, 3, size=n_inter)
    # Ensure at least one user-item per (split, user) where possible.
    df = pd.DataFrame({
        "userID": users,
        "itemID": items,
        "x_label": labels,
    })

    obj = RecDataset.__new__(RecDataset)
    obj.config = {
        "dataset": "synthetic",
        "data_path": "synthetic",
        "inter_file_name": "synthetic.inter",
    }
    obj.project_root = Path(".")
    obj.data_path = Path(".")
    obj.user_field = "userID"
    obj.item_field = "itemID"
    obj.split_field = "x_label"
    obj.n_users = n_users
    obj.n_items = n_items
    obj.train_df = df[df["x_label"] == 0].reset_index(drop=True)
    obj.valid_df = df[df["x_label"] == 1].reset_index(drop=True)
    obj.test_df = df[df["x_label"] == 2].reset_index(drop=True)
    obj._build_train_matrix()
    obj._build_eval_maps()
    obj.v_feat = rng.standard_normal((n_items, feat_dim_v)).astype(np.float32)
    obj.t_feat = rng.standard_normal((n_items, feat_dim_t)).astype(np.float32)
    return obj
