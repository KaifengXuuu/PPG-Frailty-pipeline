"""恒等 reducer：direct control / Identity reducer for the direct control."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ..contracts import ArtifactReductionResult
from ..signal.views import CANONICAL_FS_HZ
from .base import ArtifactReducer, failure_result, success_result, validate_ppg

class IdentityReducer(ArtifactReducer):
    """不改变任何样本；仍属于 direct branch / Exact no-op that remains direct."""

    reducer_id = "identity"
    reducer_version = "identity_exact_v1"
    algorithm_kernel_description = "逐样本复制双波长 PPG，不估计或抑制伪影；内核：恒等映射与同时间网格校验，" "作为未去噪直接对照。"
    is_identity = True

    def reduce(
        self,
        ppg: np.ndarray,
        imu_processed: Mapping[str, np.ndarray] | None,
        *,
        fs_hz: float = CANONICAL_FS_HZ,
    ) -> ArtifactReductionResult:
        """复制输入以隔离可变别名，数值保持逐样本相等 / Exact copied output."""

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
