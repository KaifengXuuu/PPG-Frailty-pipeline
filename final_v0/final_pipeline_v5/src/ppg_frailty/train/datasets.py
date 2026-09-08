"""训练数据集规范重导出 / Canonical training dataset re-exports.

中文：复用带不可变身份的数据集。English: Re-export datasets carrying immutable row identities.
"""

from ..training.datasets import (
    FeatureMatrixDataset,
    FeatureVectorDataset,
    FileBagDataset,
    RawWindowDataset,
    SampleIdentity,
    collate_samples,
)

__all__ = [
    "FeatureMatrixDataset",
    "FeatureVectorDataset",
    "FileBagDataset",
    "RawWindowDataset",
    "SampleIdentity",
    "collate_samples",
]
