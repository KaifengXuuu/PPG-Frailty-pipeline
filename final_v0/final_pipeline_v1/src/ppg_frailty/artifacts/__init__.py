"""V1 伪影削减公共入口。

English: Public facade for deterministic reducers and fail-closed V1 routing.
"""

from .base import ArtifactReducer, validate_result
from .bss import BssConfig, FastIcaBssReducer, NmfBssReducer, PcaBssReducer
from .decomposition import SsaConfig, SsaReducer
from .identity import IdentityReducer
from .nlms import NlmsConfig, NlmsReducer
from .router import ArtifactRouteOutcome, get_reducer, run_artifact_route
from .spectral import SpectralMaskConfig, SpectralMaskReducer

__all__ = [
    "ArtifactReducer",
    "IdentityReducer",
    "NlmsConfig",
    "NlmsReducer",
    "SsaConfig",
    "SsaReducer",
    "SpectralMaskConfig",
    "SpectralMaskReducer",
    "BssConfig",
    "PcaBssReducer",
    "FastIcaBssReducer",
    "NmfBssReducer",
    "ArtifactRouteOutcome",
    "get_reducer",
    "run_artifact_route",
    "validate_result",
]
