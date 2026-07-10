"""Common base classes, losses, and the training loop."""

from src.common.abstract_recommender import (
    AbstractRecommender,
    GeneralRecommender,
    GeneralGraphRecommender,
    MultimodalRecommender,
)
from src.common.init import xavier_uniform_initialization
from src.common.loss import BPRLoss, EmbLoss, InfoNCELoss
from src.common.trainer import Trainer

__all__ = [
    "AbstractRecommender",
    "GeneralRecommender",
    "GeneralGraphRecommender",
    "MultimodalRecommender",
    "xavier_uniform_initialization",
    "BPRLoss",
    "EmbLoss",
    "InfoNCELoss",
    "Trainer",
]
