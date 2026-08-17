"""规范 train 门面 / Canonical training facade.

中文：集中公开 dataset、sampling、loss 与 trainer。English: Export datasets, sampling, losses and the unified trainer.
"""

from .datasets import FeatureMatrixDataset, FeatureVectorDataset, FileBagDataset, RawWindowDataset, SampleIdentity
from .losses import inverse_frequency_class_weights
from .selection import validate_epoch_selection
from .trainer import FrozenOuterSplit, InnerGroupedSplit, TrainingConfig, TrainingResult, UnifiedTrainer

__all__ = [
    "FeatureMatrixDataset", "FeatureVectorDataset", "FileBagDataset", "FrozenOuterSplit",
    "InnerGroupedSplit", "RawWindowDataset", "SampleIdentity", "TrainingConfig",
    "TrainingResult", "UnifiedTrainer", "inverse_frequency_class_weights",
    "validate_epoch_selection",
]
