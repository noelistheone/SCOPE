"""CPU and GPU safety guards.

Run ``configure_runtime`` at process entry to cap CPU thread usage (so
multi-process dataloaders + BLAS don't trash the system) and ``check_gpu_available``
before allocating any CUDA memory so we fail fast rather than racing another job.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Callable, Mapping, Tuple, TypeVar

import torch

_T = TypeVar("_T")


def configure_runtime(config: Mapping[str, object] | None = None) -> None:
    """Apply CPU thread caps. Must be called before any heavy import that
    spins up BLAS thread pools (e.g. first ``import numpy`` operation).

    Reads ``cpu_threads`` from ``config`` (default 4). Sets OMP/MKL/OPENBLAS
    environment variables (only if not already set) and torch's internal
    thread counts.
    """
    n_threads = 4
    if config is not None:
        try:
            n_threads = int(config.get("cpu_threads", 4))  # type: ignore[call-arg]
        except (TypeError, ValueError):
            n_threads = 4
    n_threads = max(1, n_threads)

    # setdefault: respect explicit user override from shell
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(n_threads))

    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(min(2, n_threads))


def check_gpu_available(min_free_mb: int = 4000,
                        device_id: int = 0) -> Tuple[bool, str]:
    """Check that the GPU has at least ``min_free_mb`` of free memory.

    Returns ``(ok, info_string)``. Use to gate training when another process
    is hogging the card.
    """
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    try:
        with torch.cuda.device(device_id):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
    except (RuntimeError, AttributeError) as e:
        return False, f"mem_get_info failed: {e}"

    free_mb = free_bytes / (1024 ** 2)
    total_mb = total_bytes / (1024 ** 2)
    name = torch.cuda.get_device_name(device_id)
    info = (f"GPU {device_id} ({name}): "
            f"{free_mb:.0f} MB free / {total_mb:.0f} MB total")
    return free_mb >= min_free_mb, info


def oom_safe(fn: Callable[..., _T]) -> Callable[..., _T]:
    """Decorator that intercepts CUDA OOM, dumps a memory summary, then re-raises.

    Useful around training step / full-sort eval to give an actionable error
    message rather than a cryptic CUDA stack trace.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
            log = logging.getLogger("scope.resource")
            log.error("CUDA OOM in %s. Memory summary:\n%s",
                      fn.__qualname__, torch.cuda.memory_summary())
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"CUDA OOM in {fn.__qualname__}. "
                "Lower train_batch_size or eval_batch_size_users in your config."
            ) from e
    return wrapper
