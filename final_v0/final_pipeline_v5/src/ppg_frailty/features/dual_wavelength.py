"""双波长特征规范门面 / Canonical dual-wavelength feature facade.

中文：复用 RED/IR 特征提取。English: Re-export the tested RED/IR feature extractor.
"""

from ..signal.optical import (
    OPTICAL_SCHEMA_VERSION,
    OpticalBeatAudit,
    OpticalFeatureResult,
    extract_dual_optical,
)

__all__ = [
    "OPTICAL_SCHEMA_VERSION",
    "OpticalBeatAudit",
    "OpticalFeatureResult",
    "extract_dual_optical",
]
