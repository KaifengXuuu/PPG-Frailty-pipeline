"""Public model construction facade for the final pipeline.

English: This package exposes strict representation-aware constructors for the
reviewed compact CNN, full/small InceptionTime, NumPy ROCKET, feature baselines,
PISDPort-reference/effect-size-ablation ShapeFormer and file-bag fusion routes. Optional deep
dependencies are imported only when their model route is requested.

中文：本包为最终管线公开严格且感知 representation 的模型构造入口，覆盖已审查的
紧凑 CNN、完整/小型 InceptionTime、NumPy ROCKET、特征基线、PISDPort 参考线/
效应量消融 ShapeFormer 与文件袋融合路线。仅在请求对应深度模型时加载可选依赖。
"""

from .factory import (
    CANONICAL_MODEL_REGISTRY,
    CHANNEL_SPECIFIC_SCALAR_DISTANCE_ABLATION,
    FROZEN_MODEL_RUN_PROVENANCE_FIELDS,
    FIVE_MEMBER_ENSEMBLE_COMPARISONS,
    MATRIX_FIVE_MEMBER_ENSEMBLE_COMPARISON,
    NONENSEMBLE_MODEL_CANDIDATES,
    PYTORCH_DEPENDENCY_STATUS,
    RAW_FIVE_MEMBER_ENSEMBLE_COMPARISON,
    ModelCandidate,
    ModelInputSpec,
    PreparedModelFactory,
    build_model,
    create_model,
    model_candidate,
    materialize_architecture_parameters,
    normalize_model_config,
    normalize_model_id,
    prepare_model_factory,
    resolved_architecture_hash,
    resolved_architecture_parameters,
    validate_resolved_architecture,
    validate_frozen_model_run_provenance,
)
from .feature_baselines import FeatureVectorBaseline
from .motion import (
    HISTORICAL_LIGHT_CNN_CHANNELS,
    LightCnnArchitecture,
    LightCnnMotionDetector,
    build_historical_light_cnn_backup,
    build_parameterized_light_cnn,
    count_trainable_parameters,
)
from .rocket import (
    MaskedChannelRobustScaler,
    MiniRocketAblation,
    RocketRidgeClassifier,
    RocketTransformer,
)

__all__ = [
    "CANONICAL_MODEL_REGISTRY",
    "CHANNEL_SPECIFIC_SCALAR_DISTANCE_ABLATION",
    "FROZEN_MODEL_RUN_PROVENANCE_FIELDS",
    "FIVE_MEMBER_ENSEMBLE_COMPARISONS",
    "MATRIX_FIVE_MEMBER_ENSEMBLE_COMPARISON",
    "NONENSEMBLE_MODEL_CANDIDATES",
    "PYTORCH_DEPENDENCY_STATUS",
    "RAW_FIVE_MEMBER_ENSEMBLE_COMPARISON",
    "FeatureVectorBaseline",
    "HISTORICAL_LIGHT_CNN_CHANNELS",
    "LightCnnArchitecture",
    "LightCnnMotionDetector",
    "build_historical_light_cnn_backup",
    "build_parameterized_light_cnn",
    "count_trainable_parameters",
    "MaskedChannelRobustScaler",
    "MiniRocketAblation",
    "ModelCandidate",
    "ModelInputSpec",
    "PreparedModelFactory",
    "RocketRidgeClassifier",
    "RocketTransformer",
    "build_model",
    "create_model",
    "model_candidate",
    "materialize_architecture_parameters",
    "normalize_model_config",
    "normalize_model_id",
    "prepare_model_factory",
    "resolved_architecture_hash",
    "resolved_architecture_parameters",
    "validate_resolved_architecture",
    "validate_frozen_model_run_provenance",
]
