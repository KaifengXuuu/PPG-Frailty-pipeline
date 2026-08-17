"""V2 特征层公共入口 / V2 feature-layer public facade.

中文：公开工程特征、冻结注册表、外层训练折向量变换及固定 PPI 后端对照。
English: Exposes engineering features, frozen registries, outer-train vector transforms,
and function-only fixed-PPI backend comparisons.
"""

from .engineering import (
    EngineeringExtraction,
    FoldFeatureTransform,
    engineering_feature_names,
    extract_engineering_features,
    fit_fold_feature_transform,
    transform_engineering,
)
from .vector_transform import (
    FoldFeatureVectorTransform,
    FoldTransformedFeatureBatch,
    fit_fold_feature_vector_transform,
    transform_feature_vector,
    transform_feature_vector_batch,
)
from .prv_backend_compare import (
    PRV_BACKEND_COMPARISON_SCHEMA,
    PrvBackendResult,
    SUPPORTED_PRV_BACKENDS,
    evaluate_prv_backend,
    fixed_ppi_fixtures,
    run_prv_backend_comparison,
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
    "PRV_BACKEND_COMPARISON_SCHEMA",
    "PrvBackendResult",
    "SUPPORTED_PRV_BACKENDS",
    "evaluate_prv_backend",
    "fixed_ppi_fixtures",
    "run_prv_backend_comparison",
    "FoldFeatureTransform",
    "FoldFeatureVectorTransform",
    "FoldTransformedFeatureBatch",
    "engineering_feature_names",
    "extract_engineering_features",
    "fit_fold_feature_transform",
    "fit_fold_feature_vector_transform",
    "transform_engineering",
    "transform_feature_vector",
    "transform_feature_vector_batch",
    "FeatureDefinition",
    "FeatureRegistry",
    "default_registry",
    "summarize_engineering",
    "build_feature_vector",
    "build_ordered_matrix",
]
