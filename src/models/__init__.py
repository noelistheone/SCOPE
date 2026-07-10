"""Model registry. Add new models here.

All models conform to the AbstractRecommender interface defined in
``src.common.abstract_recommender``.
"""

from typing import Type

from src.common.abstract_recommender import AbstractRecommender
from src.models.bm3 import BM3
from src.models.cohesion import COHESION
from src.models.damrs import DAMRS
from src.models.diffmm import DiffMM
from src.models.dragon import DRAGON
from src.models.freedom import FREEDOM
from src.models.grcn import GRCN
from src.models.gume import GUME
from src.models.lattice import LATTICE
from src.models.lgmrec import LGMRec
from src.models.lightgcn import LightGCN
from src.models.llmrec import LLMRec
from src.models.mentor import MENTOR
from src.models.mgcn import MGCN
from src.models.mllmrec import MLLMRec
from src.models.mmgcn import MMGCN
from src.models.rlmrec import RLMRec
from src.models.smore import SMORE
from src.models.vbpr import VBPR

MODEL_REGISTRY: dict[str, Type[AbstractRecommender]] = {
    # Linear / graph collaborative filtering
    "lightgcn": LightGCN,
    # Multimodal recommenders
    "vbpr": VBPR,
    "mmgcn": MMGCN,
    "lattice": LATTICE,
    "freedom": FREEDOM,
    "grcn": GRCN,
    "bm3": BM3,
    "mgcn": MGCN,
    "mentor": MENTOR,
    "lgmrec": LGMRec,
    "diffmm": DiffMM,
    "smore": SMORE,
    "gume": GUME,
    "dragon": DRAGON,
    "damrs": DAMRS,
    "cohesion": COHESION,
    # LLM-augmented recommenders
    "llmrec": LLMRec,
    "rlmrec": RLMRec,
    "mllmrec": MLLMRec,
}


def get_model(name: str) -> Type[AbstractRecommender]:
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {name!r}. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key]


__all__ = [
    "MODEL_REGISTRY", "get_model",
    "LightGCN", "VBPR", "MMGCN", "LATTICE", "FREEDOM", "GRCN",
    "BM3", "MGCN", "MENTOR", "LGMRec", "DiffMM", "SMORE",
    "GUME", "DRAGON", "DAMRS", "COHESION",
    "LLMRec", "RLMRec", "MLLMRec",
]
