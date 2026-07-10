"""Layered YAML config loader.

Merge order (later overrides earlier):
  1. ``configs/overall.yaml``     — global defaults
  2. ``configs/dataset/<ds>.yaml`` — dataset-specific settings
  3. ``configs/model/<m>.yaml``    — model-specific hyperparameters
  4. CLI overrides (passed as a dict)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping, Optional

import yaml

from src.utils.misc import deep_update

# Project root: parent of `src/` — three levels up from this file.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"


class Config(MutableMapping[str, Any]):
    """Dict-like view of a resolved config.

    Use ``Config(model, dataset, cli_overrides)`` to build. Access settings via
    ``cfg['key']`` or ``cfg.get('key', default)``.
    """

    def __init__(self,
                 model: str,
                 dataset: str,
                 cli_overrides: Optional[Mapping[str, Any]] = None,
                 config_root: Optional[Path] = None) -> None:
        self.model_name = model.lower()
        self.dataset_name = dataset.lower()
        self._root = Path(config_root) if config_root else CONFIG_ROOT
        self._data: dict[str, Any] = {}
        self._load()
        if cli_overrides:
            deep_update(self._data, dict(cli_overrides))

        # Stash names for downstream consumers.
        self._data["model"] = self.model_name
        self._data["dataset"] = self.dataset_name

    # ------- loading -------

    def _load(self) -> None:
        overall = self._load_yaml(self._root / "overall.yaml")
        dataset = self._load_yaml(
            self._root / "dataset" / f"{self.dataset_name}.yaml")
        model = self._load_yaml(
            self._root / "model" / f"{self.model_name}.yaml")

        self._data = {}
        deep_update(self._data, overall)
        deep_update(self._data, dataset)
        deep_update(self._data, model)

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Config root in {path} must be a mapping, got {type(loaded).__name__}")
        return loaded

    # ------- MutableMapping interface -------

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    # ------- helpers -------

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow copy as a plain dict."""
        return dict(self._data)

    def dump_to(self, path: str | Path) -> None:
        """Write the resolved config to ``path`` as YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self._data, f, sort_keys=True,
                           allow_unicode=True, default_flow_style=False)

    def __repr__(self) -> str:
        return f"Config(model={self.model_name!r}, dataset={self.dataset_name!r}, n_keys={len(self)})"
