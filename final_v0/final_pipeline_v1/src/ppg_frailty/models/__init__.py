"""Public model construction facade for the final pipeline.

English: This package exposes strict representation-aware constructors for the
reviewed compact CNN, full/small InceptionTime, NumPy ROCKET, feature baselines,
experimental effect-size ShapeFormer and file-bag fusion routes.  Optional deep
dependencies are imported only when their model route is requested.

中文：本包为最终管线公开严格且感知 representation 的模型构造入口，覆盖已审查的
紧凑 CNN、完整/小型 InceptionTime、NumPy ROCKET、特征基线、实验性效应量
ShapeFormer 与文件袋融合路线。仅在请求对应深度模型时加载可选深度依赖。
"""

from .factory import (
    CANONICAL_MODEL_REGISTRY,
    PYTORCH_DEPENDENCY_STATUS,
    ModelInputSpec,
    build_model,
    create_model,
    normalize_model_config,
    normalize_model_id,
)
from .feature_baselines import FeatureVectorBaseline
from .rocket import (
    MaskedChannelRobustScaler,
    MiniRocketAblation,
    RocketRidgeClassifier,
    RocketTransformer,
)

__all__ = [
    "CANONICAL_MODEL_REGISTRY",
    "PYTORCH_DEPENDENCY_STATUS",
    "FeatureVectorBaseline",
    "MaskedChannelRobustScaler",
    "MiniRocketAblation",
    "ModelInputSpec",
    "RocketRidgeClassifier",
    "RocketTransformer",
    "build_model",
    "create_model",
    "normalize_model_config",
    "normalize_model_id",
]
