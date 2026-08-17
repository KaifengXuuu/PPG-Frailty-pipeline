"""规范 singular artifact 门面 / Canonical singular artifact facade.

中文：历史实现目录名为 ``artifacts``；规范 API 固定为 singular ``artifact``。
English: Implementations remain under ``artifacts`` while the contract exposes the
singular ``artifact`` boundary.
"""

from .base import ArtifactReducer, validate_result
from .bss import BssConfig, FastIcaBssReducer, NmfBssReducer, PcaBssReducer
from .decomposition import SsaConfig, SsaReducer
from .identity import IdentityReducer
from .nlms import NlmsConfig, NlmsReducer
from .router import ArtifactRouteOutcome, UnsupportedReducer, get_reducer, run_artifact_route
from .spectral import SpectralMaskConfig, SpectralMaskReducer

__all__ = [
    "ArtifactReducer", "BssConfig", "FastIcaBssReducer", "IdentityReducer",
    "NlmsConfig", "NlmsReducer", "NmfBssReducer", "PcaBssReducer", "SpectralMaskConfig",
    "SpectralMaskReducer", "SsaConfig", "SsaReducer", "ArtifactRouteOutcome",
    "UnsupportedReducer", "get_reducer", "run_artifact_route", "validate_result",
]
