"""Identity reducer for the direct control."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ..contracts import ArtifactReductionResult
from ..signal.views import CANONICAL_FS_HZ
from .base import ArtifactReducer, failure_result, success_result, validate_ppg

class IdentityReducer(ArtifactReducer):
    """Exact no-op that remains direct."""

    reducer_id = "identity"
    reducer_version = "identity_exact_v1"
    algorithm_kernel_description = "Copy dual-wavelength PPG sample by sample without estimating or suppressing artifacts. " "Kernel: identity mapping and shared-time-grid validation, serving as the undenoised direct control."
    is_identity = True

    def reduce(
        self,
        ppg: np.ndarray,
        imu_processed: Mapping[str, np.ndarray] | None,
        *,
        fs_hz: float = CANONICAL_FS_HZ,
    ) -> ArtifactReductionResult:
        """Copy input to isolate mutable aliases while preserving every sample value."""

        try:
            source = validate_ppg(ppg, fs_hz=fs_hz)
            return success_result(
                self,
                source.copy(),
                input_ppg=source,
                confidence=1.0,
                parameters={},
                diagnostics={"max_absolute_change": 0.0},
            )
        except ValueError as exc:
            return failure_result(self, str(exc))
