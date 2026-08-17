"""V1 特征层公共入口。

English: Public facade for engineering extraction, frozen registries, vectors, and
ordered feature matrices.
"""

from .engineering import (
    EngineeringExtraction,
    FoldFeatureTransform,
    engineering_feature_names,
    extract_engineering_features,
    fit_fold_feature_transform,
    transform_engineering,
)
from .registry import (
    FeatureDefinition,
    FeatureRegistry,
    build_feature_vector,
    build_ordered_matrix,
    default_registry,
    summarize_engineering,
)

__all__ = [
    "EngineeringExtraction",
    "FoldFeatureTransform",
    "engineering_feature_names",
    "extract_engineering_features",
    "fit_fold_feature_transform",
    "transform_engineering",
    "FeatureDefinition",
    "FeatureRegistry",
    "default_registry",
    "summarize_engineering",
    "build_feature_vector",
    "build_ordered_matrix",
]
