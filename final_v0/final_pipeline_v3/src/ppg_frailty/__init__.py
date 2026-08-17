"""PPG frailty final pipeline V2 / PPG 衰弱度最终流程 V2.

中文：公共 API 只从本包导出；V1 与根目录历史脚本均不可成为 V2 活动依赖。
English: Public APIs are exported only from this package. V1 and historical root
scripts remain provenance-only and cannot become active V2 dependencies.
"""

from .config import PipelineConfig, load_config
from .contracts import (
    ArtifactReductionResult,
    EngineeringFeatureSequence,
    FeatureVectorV1,
    ManifestRow,
    OrderedFeatureMatrixV1,
    PredictionBundle,
    PulseResult,
    QualityEndpoint,
    QualityResult,
    QualityState,
    RepresentationMode,
    SignalRoute,
    SignalViews,
)

__all__ = [
    "ArtifactReductionResult",
    "EngineeringFeatureSequence",
    "FeatureVectorV1",
    "ManifestRow",
    "OrderedFeatureMatrixV1",
    "PipelineConfig",
    "PredictionBundle",
    "PulseResult",
    "QualityEndpoint",
    "QualityResult",
    "QualityState",
    "RepresentationMode",
    "SignalRoute",
    "SignalViews",
    "load_config",
]

