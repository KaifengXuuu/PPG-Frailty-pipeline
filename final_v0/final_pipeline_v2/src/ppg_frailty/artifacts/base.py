"""ArtifactReducer 公共接口与失败闭合 / Common reducer interface and fail-closed rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

import numpy as np

from ..contracts import ArtifactReductionResult
from ..signal.views import CANONICAL_FS_HZ


IMU_REFERENCE_AXES6_PROFILE_ID = "imu_axes6_reference_v2"
IMU_REFERENCE_DERIVED9_AUGMENTATION_PROFILE_ID = (
    "imu_axes6_plus_derived3_augmentation_ablation_v2"
)


class ArtifactReducer(ABC):
    """所有 reducer 的确定性 CPU 接口 / Deterministic CPU interface for reducers."""

    reducer_id: str
    reducer_version: str
    algorithm_kernel_description: str = ""
    is_identity: bool = False

    @abstractmethod
    def reduce(
        self,
        ppg: np.ndarray,
        imu_processed: Mapping[str, np.ndarray] | None,
        *,
        fs_hz: float = CANONICAL_FS_HZ,
    ) -> ArtifactReductionResult:
        """返回 aligned result 或显式 failure / Return aligned output or explicit failure."""


def validate_ppg(
    ppg: np.ndarray,
    *,
    fs_hz: float,
    allow_single_channel: bool = False,
) -> np.ndarray:
    """验证 reducer 输入，不做隐式重采样 / Validate without implicit resampling."""

    if float(fs_hz) != CANONICAL_FS_HZ:
        raise ValueError("artifact reducers require the exact 400 Hz grid")
    values = np.asarray(ppg, dtype=np.float64)
    allowed = (1, 2) if allow_single_channel else (2,)
    if values.ndim != 2 or values.shape[1] not in allowed or values.shape[0] == 0:
        raise ValueError(f"PPG must have shape (samples, {allowed})")
    if not np.isfinite(values).all():
        raise ValueError("artifact reducer input must be finite")
    return values


def imu_reference_matrix(
    imu_processed: Mapping[str, np.ndarray] | None,
    n_samples: int,
    *,
    profile_id: str = IMU_REFERENCE_AXES6_PROFILE_ID,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """构建标准化 IMU reference matrix / Build a standardized IMU reference matrix.

    reference 只使用 dynamic ACC 与 gyro 六轴；三个派生量仅在具名 augmentation
    profile 中加入。The reference uses six physical axes; derived channels require
    the explicitly named augmentation profile.
    """

    if imu_processed is None:
        raise ValueError("IMU references are required for this reducer")
    if "imu_valid_mask" in imu_processed:
        valid_rows = np.asarray(
            imu_processed["imu_valid_mask"], dtype=bool
        ).ravel()
        if valid_rows.shape != (n_samples,):
            raise ValueError("imu_valid_mask must align with PPG")
    else:
        valid_rows = np.ones(n_samples, dtype=bool)
    if np.count_nonzero(valid_rows) < 32:
        raise ValueError("too few valid processed IMU samples")
    if profile_id not in {
        IMU_REFERENCE_AXES6_PROFILE_ID,
        IMU_REFERENCE_DERIVED9_AUGMENTATION_PROFILE_ID,
    }:
        raise ValueError(f"unknown IMU reference profile: {profile_id}")
    arrays: list[np.ndarray] = []
    names: list[str] = []
    for key, axis_names in (
        ("dynamic_acc_mps2", ("acc_x", "acc_y", "acc_z")),
        ("gyro_rads", ("gyro_x", "gyro_y", "gyro_z")),
    ):
        if key not in imu_processed:
            raise ValueError(f"{profile_id} requires processed field {key}")
        value = np.asarray(imu_processed[key], dtype=np.float64)
        if value.shape != (n_samples, 3):
            raise ValueError(f"{key} must have shape (samples, 3)")
        if not np.isfinite(value[valid_rows]).all():
            raise ValueError(f"{key} is nonfinite inside its valid mask")
        arrays.append(value)
        names.extend(axis_names)
    if profile_id == IMU_REFERENCE_DERIVED9_AUGMENTATION_PROFILE_ID:
        for key in ("dynamic_magnitude", "gyro_magnitude", "jerk_magnitude"):
            if key not in imu_processed:
                raise ValueError(f"{profile_id} requires processed field {key}")
            value = np.asarray(imu_processed[key], dtype=np.float64).reshape(-1, 1)
            if value.shape[0] != n_samples:
                raise ValueError(f"{key} must align with PPG")
            if not np.isfinite(value[valid_rows]).all():
                raise ValueError(f"{key} is nonfinite inside its valid mask")
            arrays.append(value)
            names.append(key)
    if not arrays:
        raise ValueError("no supported processed IMU references are available")
    matrix = np.column_stack(arrays)
    center = np.mean(matrix[valid_rows], axis=0)
    scale = np.std(matrix[valid_rows], axis=0)
    keep = scale > 1e-12
    if not np.any(keep):
        raise ValueError("all IMU references are constant")
    standardized = np.zeros((n_samples, int(np.count_nonzero(keep))))
    standardized[valid_rows] = (
        matrix[valid_rows][:, keep] - center[keep]
    ) / scale[keep]
    kept_names = tuple(name for name, selected in zip(names, keep) if selected)
    # English: Returning the mask is mandatory: reducers must either propagate it
    # to their output-validity evidence or fail; neutralized rows are never evidence.
    # 中文：必须返回 mask；reducer 要么传播到输出有效性，要么失败。置零行不是证据。
    return np.asarray(standardized, dtype=np.float64), kept_names, valid_rows.copy()


def parameters_dict(config: Any) -> dict[str, Any]:
    """将 frozen dataclass 参数转审计字典 / Convert frozen config to audit dictionary."""

    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    return {}


def failure_result(
    reducer: ArtifactReducer,
    reason: str,
    *,
    status: str = "failed",
    parameters: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> ArtifactReductionResult:
    """失败结果必须 `x_ar=None` / Failure results must never contain a waveform."""

    return ArtifactReductionResult(
        x_ar=None,
        reducer_id=reducer.reducer_id,
        reducer_version=reducer.reducer_version,
        is_identity=reducer.is_identity,
        status=status,
        confidence=0.0,
        diagnostics=dict(diagnostics or {}),
        parameters=dict(parameters or {}),
        channel_available=(False, False),
        alignment={"fs_hz": CANONICAL_FS_HZ, "same_time_grid": False},
        reasons=(str(reason),),
    )


def success_result(
    reducer: ArtifactReducer,
    x_ar: np.ndarray,
    *,
    input_ppg: np.ndarray,
    confidence: float,
    parameters: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> ArtifactReductionResult:
    """构造并验证成功结果 / Construct and validate a successful result."""

    output = np.asarray(x_ar, dtype=np.float64)
    source = np.asarray(input_ppg, dtype=np.float64)
    if output.shape != source.shape or output.ndim != 2 or output.shape[1] != 2:
        raise ValueError("successful x_ar must preserve samples-by-two-channel shape")
    if not np.isfinite(output).all():
        raise ValueError("successful x_ar must be finite")
    if reducer.is_identity and not np.array_equal(output, source):
        raise ValueError("identity reducer changed its input")
    channel_available = tuple(bool(np.std(output[:, index]) > 1e-12) for index in range(2))
    return ArtifactReductionResult(
        x_ar=output,
        reducer_id=reducer.reducer_id,
        reducer_version=reducer.reducer_version,
        is_identity=reducer.is_identity,
        status="success",
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        diagnostics=dict(diagnostics),
        parameters=dict(parameters),
        channel_available=(channel_available[0], channel_available[1]),
        alignment={
            "fs_hz": CANONICAL_FS_HZ,
            "same_time_grid": True,
            "input_samples": int(source.shape[0]),
            "output_samples": int(output.shape[0]),
        },
        reasons=(),
    )


def validate_result(input_ppg: np.ndarray, result: ArtifactReductionResult) -> None:
    """外部审计 reducer 合同 / Audit a reducer result at the router boundary."""

    source = np.asarray(input_ppg, dtype=np.float64)
    if result.status == "success":
        if result.x_ar is None:
            raise ValueError("successful reducer returned no x_ar")
        output = np.asarray(result.x_ar, dtype=np.float64)
        if output.shape != source.shape or not np.isfinite(output).all():
            raise ValueError("successful reducer violated alignment/finite contract")
        if not bool(result.alignment.get("same_time_grid", False)):
            raise ValueError("successful reducer did not declare same_time_grid")
        if result.is_identity and not np.array_equal(output, source):
            raise ValueError("identity reducer changed input")
    elif result.x_ar is not None:
        raise ValueError("failed/unsupported reducer must return x_ar=None")
