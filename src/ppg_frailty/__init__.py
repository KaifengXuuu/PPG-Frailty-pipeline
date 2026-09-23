"""PPG frailty final pipeline V6, retaining V2-compatible algorithm contracts.

Public entry points are self-contained; historical implementations are not runtime dependencies.
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
