"""V1 伪影削减公共入口。

English: Public facade for deterministic reducers and fail-closed V1 routing.
"""

from .base import ArtifactReducer, validate_result
from .bss import (
    BssConfig,
    FastIcaBssConfig,
    FastIcaBssReducer,
    NmfBssConfig,
    NmfBssReducer,
    PcaBssConfig,
    PcaBssReducer,
)
from .decomposition import SsaConfig, SsaReducer
from .identity import IdentityReducer
from .legacy import (
    CeemdLiteNlmsLegacyConfig,
    CeemdLiteNlmsLegacyReducer,
    DwtA2LegacyConfig,
    DwtA2LegacyReducer,
    EmdSiftingConfig,
    EmdSiftingRateOnlyReducer,
)
from .nlms import NlmsConfig, NlmsReducer
from .router import ArtifactRouteOutcome, get_reducer, reducer_audit_metadata, run_artifact_route
from .spectral import SpectralMaskConfig, SpectralMaskReducer

__all__ = [
    "ArtifactReducer",
    "IdentityReducer",
    "EmdSiftingConfig",
    "EmdSiftingRateOnlyReducer",
    "CeemdLiteNlmsLegacyConfig",
    "CeemdLiteNlmsLegacyReducer",
    "DwtA2LegacyConfig",
    "DwtA2LegacyReducer",
    "NlmsConfig",
    "NlmsReducer",
    "SsaConfig",
    "SsaReducer",
    "SpectralMaskConfig",
    "SpectralMaskReducer",
    "BssConfig",
    "PcaBssConfig",
    "FastIcaBssConfig",
    "NmfBssConfig",
    "PcaBssReducer",
    "FastIcaBssReducer",
    "NmfBssReducer",
    "ArtifactRouteOutcome",
    "get_reducer",
    "reducer_audit_metadata",
    "run_artifact_route",
    "validate_result",
]
