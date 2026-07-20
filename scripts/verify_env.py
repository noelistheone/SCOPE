#!/usr/bin/env python
"""Environment verification: imports + CUDA + sparse ops.

Exits 0 only if all imports succeed and a tiny sparse-mm smoke test passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when running as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> int:
    rows = []

    # ---- core imports ----
    try:
        import torch
        rows.append(("torch", torch.__version__))
    except Exception as e:
        print(f"FATAL: torch import failed: {e}", file=sys.stderr)
        return 1
    try:
        import numpy as np
        rows.append(("numpy", np.__version__))
    except Exception as e:
        print(f"FATAL: numpy import failed: {e}", file=sys.stderr)
        return 1
    for mod_name in ("pandas", "scipy", "sklearn", "yaml", "optuna",
                     "tensorboard", "einops", "tqdm", "torch_geometric", "matplotlib"):
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "?")
            rows.append((mod_name, ver))
        except Exception as e:
            print(f"FATAL: {mod_name} import failed: {e}", file=sys.stderr)
            return 1

    # torch_scatter is optional but expected.
    try:
        import torch_scatter
        rows.append(("torch_scatter", torch_scatter.__version__))
    except Exception as e:
        print(f"WARN: torch_scatter not installed ({e}) — some models may fail")

    # ---- project imports ----
    try:
        from src.utils.configurator import Config  # noqa: F401
        from src.data.dataset import RecDataset  # noqa: F401
        from src.data.graph_utils import build_norm_adj  # noqa: F401
        from src.models import MODEL_REGISTRY
        from src.common.trainer import Trainer  # noqa: F401
        from src.evaluation.topk_evaluator import TopKEvaluator  # noqa: F401
    except Exception as e:
        print(f"FATAL: src.* import failed: {e}", file=sys.stderr)
        return 1
    rows.append(("models registered", str(sorted(MODEL_REGISTRY))))

    # ---- CUDA detection ----
    cuda_ok = torch.cuda.is_available()
    rows.append(("CUDA available", str(cuda_ok)))
    if cuda_ok:
        rows.append(("CUDA device 0", torch.cuda.get_device_name(0)))
        try:
            free, total = torch.cuda.mem_get_info(0)
            rows.append(("CUDA free", f"{free/2**20:.0f} MB"))
            rows.append(("CUDA total", f"{total/2**20:.0f} MB"))
        except Exception as e:
            rows.append(("mem_get_info", f"failed: {e}"))

    # ---- sparse-mm smoke test (CPU; tiny) ----
    try:
        i = torch.tensor([[0, 1, 2], [1, 0, 2]], dtype=torch.long)
        v = torch.tensor([1., 1., 1.], dtype=torch.float32)
        sp = torch.sparse_coo_tensor(i, v, (3, 3)).coalesce()
        out = torch.sparse.mm(sp, torch.eye(3))
        assert out.shape == (3, 3)
        rows.append(("sparse-mm", "ok"))
    except Exception as e:
        print(f"FATAL: sparse-mm smoke test failed: {e}", file=sys.stderr)
        return 1

    # ---- report ----
    width = max(len(k) for k, _ in rows)
    print("Environment check:")
    print("-" * (width + 32))
    for k, v in rows:
        print(f"  {k.ljust(width)}  {v}")
    print("-" * (width + 32))
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
