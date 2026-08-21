"""显式信号视图与唯一窗口合同 / Explicit signal views and sole window contract.

中文：本模块冻结 400 Hz 信号视图及 direct/identity/non-identity 路线边界。
窗口类型仅从 ``ppg_frailty.data.windows`` 重新导出；仓库中不再维护第二套
不兼容的 ``WindowPlan`` 或 padding 语义。

English: This module freezes the 400 Hz views and route boundary. Window types
are re-exported solely from ``ppg_frailty.data.windows``; no second incompatible
planner or padding contract is maintained here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..contracts import ArtifactReductionResult, SignalRoute, SignalViews
from ..data.windows import WindowPlan, WindowSlice, extract_window


CANONICAL_FS_HZ = 400.0


def _matrix(value: np.ndarray, *, name: str, columns: int = 2) -> np.ndarray:
    """转成有限二维矩阵 / Convert to a finite two-dimensional matrix."""

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != columns:
        raise ValueError(f"{name} must have shape (samples, {columns})")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite")
    return matrix


@dataclass(frozen=True)
class CanonicalSignalViews:
    """V1 四视图合同 / V1 four-view signal contract.

    中文：``x_analysis_rate`` 在 direct/identity 路线等于 ``x_filter``；只有
    non-identity 成功时 ``x_ar`` 才存在，而且只能进入 rate/PPI/PRV 路径。

    English: ``x_analysis_rate`` equals ``x_filter`` on direct/identity routes.
    A non-identity ``x_ar`` exists only after successful reduction and is eligible
    solely for rate/PPI/PRV processing.
    """

    x_native: np.ndarray
    x_filter: np.ndarray
    x_analysis_rate: np.ndarray
    imu_processed: dict[str, np.ndarray]
    metadata: dict[str, Any]
    source_valid_mask: np.ndarray
    repair_mask: np.ndarray
    x_ar: np.ndarray | None = None
    route: SignalRoute = SignalRoute.DIRECT

    @property
    def processed_imu_physical(self) -> dict[str, np.ndarray]:
        """Return the physical-unit IMU analysis view, never the DL tensor."""

        return self.imu_processed

    @property
    def x_analysis(self) -> np.ndarray:
        """Return the amplitude-preserving direct analysis PPG view.

        This alias is deliberately distinct from ``x_analysis_rate``: the
        latter may be replaced by a rate-only artifact-reduced signal, whereas
        morphology, optical engineering, and the DL-window source retain the
        amplitude-preserving ``x_filter`` samples.
        """

        return np.asarray(self.x_filter, dtype=np.float64)

    def validate(self) -> None:
        """验证形状、对齐、采样率和 rate-only 边界 / Validate all invariants."""

        native = _matrix(self.x_native, name="x_native")
        filtered = _matrix(self.x_filter, name="x_filter")
        rate = _matrix(self.x_analysis_rate, name="x_analysis_rate")
        if filtered.shape != native.shape or rate.shape != native.shape:
            raise ValueError("all PPG views must share the native time grid")
        if not np.array_equal(rate, filtered):
            raise ValueError("direct x_analysis_rate must equal x_filter in V1")
        valid = np.asarray(self.source_valid_mask, dtype=bool)
        repaired = np.asarray(self.repair_mask, dtype=bool)
        if valid.shape != native.shape or repaired.shape != native.shape:
            raise ValueError("source_valid_mask and repair_mask must match PPG shape")
        if float(self.metadata.get("fs_hz", 0.0)) != CANONICAL_FS_HZ:
            raise ValueError("canonical views must remain on the exact 400 Hz grid")

        if self.route is SignalRoute.ARTIFACT_RATE_ONLY:
            artifact = _matrix(self.x_ar, name="x_ar") if self.x_ar is not None else None
            if artifact is None or artifact.shape != native.shape:
                raise ValueError("successful non-identity route requires aligned x_ar")
            if not bool(self.metadata.get("rate_only", False)):
                raise ValueError("non-identity x_ar must be marked rate_only")
            if self.metadata.get("q_morph_state") != "not_applicable":
                raise ValueError("non-identity x_ar requires q_morph_state=not_applicable")
            artifact_valid = np.asarray(
                self.metadata.get("artifact_output_valid_mask"), dtype=bool
            )
            if artifact_valid.shape != (native.shape[0],) or not np.any(artifact_valid):
                raise ValueError("non-identity x_ar requires a non-empty aligned validity mask")
        elif self.route is SignalRoute.IDENTITY:
            if self.x_ar is not None and not np.array_equal(np.asarray(self.x_ar), filtered):
                raise ValueError("identity x_ar must be byte-value identical to x_filter")
        elif self.x_ar is not None:
            raise ValueError("direct/dropped views cannot carry a non-identity x_ar")

    @property
    def analysis_signal(self) -> np.ndarray:
        """返回合法 rate 输入，绝不失败回退 / Return the legal rate input without fallback."""

        self.validate()
        if self.route is SignalRoute.ARTIFACT_RATE_ONLY:
            # 中文：validate 已证明存在；assert 仅帮助静态类型收窄。
            # English: Validation proves existence; assert only narrows the type.
            assert self.x_ar is not None
            return np.asarray(self.x_ar, dtype=np.float64)
        return np.asarray(self.x_filter, dtype=np.float64)

    @property
    def rate_valid_mask(self) -> np.ndarray:
        """返回 rate waveform 的显式逐样本有效性 / Return rate-sample validity."""

        self.validate()
        if self.route is SignalRoute.ARTIFACT_RATE_ONLY:
            return np.asarray(self.metadata["artifact_output_valid_mask"], dtype=bool).copy()
        return np.ones(np.asarray(self.x_filter).shape[0], dtype=bool)

    def to_contract(self) -> SignalViews:
        """适配仓库公共 ``SignalViews`` / Adapt to the repository-wide contract."""

        self.validate()
        metadata = dict(self.metadata)
        metadata.update(
            {
                "route": self.route.value,
                "non_identity_artifact_reduction": self.route
                is SignalRoute.ARTIFACT_RATE_ONLY,
                "rate_only": self.route is SignalRoute.ARTIFACT_RATE_ONLY,
            }
        )
        value = SignalViews(
            x_native=np.asarray(self.x_native, dtype=np.float64),
            x_filter=np.asarray(self.x_filter, dtype=np.float64),
            x_analysis=self.analysis_signal,
            imu_processed={key: np.asarray(item) for key, item in self.imu_processed.items()},
            metadata=metadata,
        )
        value.validate()
        return value

    def with_artifact_result(
        self, result: ArtifactReductionResult
    ) -> "CanonicalSignalViews":
        """绑定 reducer 结果；失败时抛错而非偷回 direct / Bind without fallback."""

        if result.status != "success" or result.x_ar is None:
            raise RuntimeError(
                "artifact reducer did not succeed; direct fallback is scientifically forbidden"
            )
        artifact = _matrix(result.x_ar, name="artifact_result.x_ar")
        if artifact.shape != np.asarray(self.x_filter).shape:
            raise ValueError("artifact result lost sample/channel alignment")
        if result.is_identity:
            if not np.array_equal(artifact, np.asarray(self.x_filter)):
                raise ValueError("identity reducer changed x_filter")
            route = SignalRoute.IDENTITY
            metadata = {
                **self.metadata,
                "reducer_id": result.reducer_id,
                "reducer_version": result.reducer_version,
                "rate_only": False,
                "q_morph_state": "available",
            }
        else:
            route = SignalRoute.ARTIFACT_RATE_ONLY
            output_valid = np.asarray(
                result.diagnostics.get(
                    "output_valid_mask",
                    np.ones(artifact.shape[0], dtype=bool),
                ),
                dtype=bool,
            )
            if output_valid.shape != (artifact.shape[0],) or not np.any(output_valid):
                raise ValueError("artifact reducer returned an invalid output_valid_mask")
            metadata = {
                **self.metadata,
                "reducer_id": result.reducer_id,
                "reducer_version": result.reducer_version,
                "rate_only": True,
                "q_morph_state": "not_applicable",
                "artifact_output_valid_mask": output_valid.copy(),
            }
        updated = replace(self, x_ar=artifact.copy(), route=route, metadata=metadata)
        updated.validate()
        return updated


__all__ = [
    "CANONICAL_FS_HZ",
    "CanonicalSignalViews",
    "WindowPlan",
    "WindowSlice",
    "extract_window",
]
