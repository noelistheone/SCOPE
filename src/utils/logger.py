"""Logger factory: file + console + (optional) TensorBoard."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_LOGGER_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def build_logger(name: str = "recsys",
                 log_dir: Optional[str | Path] = None,
                 level: int = logging.INFO,
                 console: bool = True) -> logging.Logger:
    """Build (or fetch) a logger.

    If ``log_dir`` is given, a ``run.log`` file is also opened in that directory.
    Repeated calls with the same name return the same logger without duplicating
    handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # avoid duplicates if root logger also configured

    if logger.handlers:
        return logger

    formatter = logging.Formatter(_LOGGER_FORMAT, datefmt=_DATE_FORMAT)

    if console:
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
