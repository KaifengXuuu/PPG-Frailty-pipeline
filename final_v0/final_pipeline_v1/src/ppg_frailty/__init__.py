"""PPG frailty final pipeline V1 / PPG 衰弱度最终流程 V1.

中文：公共 API 只从本包导出；根目录历史脚本不得成为未来活动依赖。
English: Public APIs are exported only from this package; historical root scripts
cannot become future-active dependencies.
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

