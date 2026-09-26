"""Miscellaneous helpers: path management, time formatting, dict utilities."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create directory if missing. Returns a Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_local_time() -> str:
    """Timestamp suitable for run names: 'YYYYMMDD_HHMMSS'."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def dict_to_str(d: Mapping[str, Any], precision: int = 4) -> str:
    """Pretty one-line dict for metric logging."""
    parts = []
    for k, v in d.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.{precision}f}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def deep_update(base: dict, override: Mapping[str, Any]) -> dict:
    """Recursively merge ``override`` into ``base`` (in-place on base). Returns base.

    Lists and scalars are replaced wholesale; only dicts are merged recursively.
    """
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, Mapping):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base
