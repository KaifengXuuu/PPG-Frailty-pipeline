"""Outer-train-only IMU scaling for raw/fusion / raw/fusion 的外层训练折 IMU 缩放。

English: RED/IR have already passed their selected per-window strategy. This
artifact fits the selected transform for the six physical IMU axes used by the
frailty tensor on declared outer-train participants and preserves a hash-bound roster.
OOF fitting is rejected.
中文：RED/IR 已执行所选窗口策略。本产物仅用声明的 outer-train participant
拟合六个 IMU 轴，并将 roster 绑定到哈希；任何 OOF 拟合都会被拒绝。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

import numpy as np

from ..normalization import (
    FALLBACK_MAD,
    FALLBACK_ONE,
    IMU_MEAN_STD,
    IMU_NONE,
    IMU_ROBUST,
    RawNormalizationConfig,
)
from ..provenance import assert_training_only, stable_payload_sha256
from .raw import RawWindows

IMU_TRANSFORM_SCHEMA_VERSION = "raw_frailty_imu_axes6_outer_train_configurable_v4"
IQR_NORMAL_CONSISTENCY_DIVISOR = 1.349
RAW_CHANNEL_SCHEMA = (
    "RED",
    "IR",
    "A_dyn_x",
    "A_dyn_y",
    "A_dyn_z",
    "GX",
    "GY",
    "GZ",
)
IMU_CHANNEL_SCHEMA = RAW_CHANNEL_SCHEMA[2:]


def _artifact_hash(
    center: np.ndarray,
    scale: np.ndarray,
    valid_count: np.ndarray,
    fitted_ids: tuple[str, ...],
    *,
    strategy: str,
    iqr_fallback: str,
    robust_iqr_divisor: float,
    mad_consistency_divisor: float,
    scale_epsilon: float,
    standard_ddof: int,
) -> str:
    return stable_payload_sha256({
        "schema_version": IMU_TRANSFORM_SCHEMA_VERSION,
        "channel_schema": list(IMU_CHANNEL_SCHEMA),
        "center": np.asarray(center, dtype=np.float64).tolist(),
        "scale": np.asarray(scale, dtype=np.float64).tolist(),
        "valid_count": np.asarray(valid_count, dtype=np.int64).tolist(),
        "fitted_on_participant_ids": list(fitted_ids),
        "strategy": strategy,
        "iqr_fallback": iqr_fallback,
        "robust_iqr_divisor": float(robust_iqr_divisor),
        "mad_consistency_divisor": float(mad_consistency_divisor),
        "scale_epsilon": float(scale_epsilon),
        "standard_ddof": int(standard_ddof),
        "per_window_imu_scaling": False,
    })


@dataclass(frozen=True)
class FoldImuChannelTransform:
    """Hash-bound six-axis fold transform / 绑定哈希的六轴折内变换。"""

    center: np.ndarray
    scale: np.ndarray
    valid_count: np.ndarray
    fitted_on_participant_ids: tuple[str, ...]
    artifact_sha256: str
    strategy: str = IMU_ROBUST
    iqr_fallback: str = "standard_deviation_then_finite_one"
    robust_iqr_divisor: float = IQR_NORMAL_CONSISTENCY_DIVISOR
    mad_consistency_divisor: float = 0.6744897501960817
    scale_epsilon: float = 1e-8
    standard_ddof: int = 0
    channel_schema: tuple[str, ...] = IMU_CHANNEL_SCHEMA
    schema_version: str = IMU_TRANSFORM_SCHEMA_VERSION

    def validate(self) -> None:
        """Audit alignment and identity / 审计对齐及身份。"""

        center = np.asarray(self.center, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        count = np.asarray(self.valid_count, dtype=np.int64)
        expected_shape = (len(IMU_CHANNEL_SCHEMA), )
        if center.shape != expected_shape or scale.shape != expected_shape or count.shape != expected_shape:
            raise ValueError("frailty IMU transform arrays must align with six axes")
        if not np.isfinite(center).all() or not np.isfinite(scale).all():
            raise ValueError("IMU transform statistics must be finite")
        if np.any(scale <= 0.0) or np.any(count <= 0):
            raise ValueError("IMU transform scale/count is invalid")
        if not self.fitted_on_participant_ids:
            raise ValueError("IMU transform requires an outer-train roster")
        resolved = RawNormalizationConfig.from_mapping({
            "raw_imu": self.strategy,
            "iqr_fallback": self.iqr_fallback,
            "robust_iqr_divisor": self.robust_iqr_divisor,
            "mad_consistency_divisor": self.mad_consistency_divisor,
            "scale_epsilon": self.scale_epsilon,
            "standard_ddof": self.standard_ddof,
        })
        expected = _artifact_hash(
            center,
            scale,
            count,
            self.fitted_on_participant_ids,
            strategy=resolved.raw_imu,
            iqr_fallback=resolved.iqr_fallback,
            robust_iqr_divisor=resolved.robust_iqr_divisor,
            mad_consistency_divisor=resolved.mad_consistency_divisor,
            scale_epsilon=resolved.scale_epsilon,
            standard_ddof=resolved.standard_ddof,
        )
        if (self.schema_version != IMU_TRANSFORM_SCHEMA_VERSION or self.channel_schema != IMU_CHANNEL_SCHEMA
                or self.strategy != resolved.raw_imu or self.iqr_fallback != resolved.iqr_fallback
                or self.artifact_sha256 != expected):
            raise ValueError("IMU transform artifact identity drift")


def _validate_raw_tensor(values: np.ndarray) -> np.ndarray:
    tensor = np.asarray(values, dtype=np.float64)
    if tensor.ndim != 3 or tensor.shape[1] != len(RAW_CHANNEL_SCHEMA) or tensor.shape[0] == 0 or tensor.shape[2] == 0:
        raise ValueError("frailty raw tensor must have shape [windows,8,samples]")
    if not np.isfinite(tensor).all():
        raise ValueError("raw tensor must be finite")
    return tensor


def _valid_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    result = np.asarray(mask, dtype=bool)
    if result.shape != shape:
        raise ValueError("raw valid_mask must have shape [windows,samples]")
    return result


def fit_fold_imu_channel_transform(
    values: np.ndarray,
    participant_ids: Iterable[str],
    *,
    fitted_on_participant_ids: Iterable[str],
    outer_train_participant_ids: Iterable[str],
    outer_oof_participant_ids: Iterable[str],
    valid_mask: np.ndarray | None = None,
    normalization: Mapping[str, Any] | None = None,
) -> FoldImuChannelTransform:
    """Fit six IMU axes on declared train rows / 仅在声明训练行拟合六轴。"""

    normalization_config = RawNormalizationConfig.from_mapping(normalization)
    fitted = assert_training_only(
        fitted_on_participant_ids,
        outer_train_participant_ids,
        outer_oof_participant_ids,
    )
    tensor = _validate_raw_tensor(values)
    ids = tuple(str(item) for item in participant_ids)
    if len(ids) != tensor.shape[0]:
        raise ValueError("participant_ids must align with raw windows")
    fitted_set = set(fitted)
    selected = np.asarray([item in fitted_set for item in ids], dtype=bool)
    if {item for item, keep in zip(ids, selected) if keep} != fitted_set:
        raise ValueError("not every declared fitted participant has a raw window")
    mask = _valid_mask(valid_mask, (tensor.shape[0], tensor.shape[2]))
    channel_count = len(IMU_CHANNEL_SCHEMA)
    center = np.empty(channel_count, dtype=np.float64)
    scale = np.empty(channel_count, dtype=np.float64)
    count = np.empty(channel_count, dtype=np.int64)
    selected_mask = mask[selected]
    for channel in range(channel_count):
        samples = tensor[selected, channel + 2, :][selected_mask]
        count[channel] = samples.size
        if not samples.size:
            raise ValueError(f"no valid train samples for IMU channel {IMU_CHANNEL_SCHEMA[channel]}")
        if normalization_config.raw_imu == IMU_NONE:
            center[channel] = 0.0
            scale[channel] = 1.0
            continue
        if normalization_config.raw_imu == IMU_MEAN_STD:
            center[channel] = float(np.mean(samples))
            candidate = (float(np.std(samples, ddof=normalization_config.standard_ddof))
                         if samples.size > normalization_config.standard_ddof else float("nan"))
        elif normalization_config.raw_imu == IMU_ROBUST:
            center[channel] = float(np.median(samples))
            q25, q75 = np.percentile(samples, [25.0, 75.0])
            candidate = float(q75 - q25) / float(normalization_config.robust_iqr_divisor)
            if not np.isfinite(candidate) or candidate <= normalization_config.scale_epsilon:
                if normalization_config.iqr_fallback == FALLBACK_ONE:
                    candidate = 1.0
                elif normalization_config.iqr_fallback == FALLBACK_MAD:
                    candidate = float(np.median(np.abs(samples - center[channel]))) / float(
                        normalization_config.mad_consistency_divisor)
                else:
                    candidate = (float(np.std(
                        samples,
                        ddof=normalization_config.standard_ddof,
                    )) if samples.size > normalization_config.standard_ddof else float("nan"))
        else:  # defensive: RawNormalizationConfig owns strategy registration.
            raise ValueError(f"unsupported raw IMU normalization: {normalization_config.raw_imu}")
        scale[channel] = candidate if np.isfinite(candidate) and candidate > normalization_config.scale_epsilon else 1.0
    artifact_sha256 = _artifact_hash(
        center,
        scale,
        count,
        fitted,
        strategy=normalization_config.raw_imu,
        iqr_fallback=normalization_config.iqr_fallback,
        robust_iqr_divisor=normalization_config.robust_iqr_divisor,
        mad_consistency_divisor=normalization_config.mad_consistency_divisor,
        scale_epsilon=normalization_config.scale_epsilon,
        standard_ddof=normalization_config.standard_ddof,
    )
    artifact = FoldImuChannelTransform(
        center=center,
        scale=scale,
        valid_count=count,
        fitted_on_participant_ids=fitted,
        artifact_sha256=artifact_sha256,
        strategy=normalization_config.raw_imu,
        iqr_fallback=normalization_config.iqr_fallback,
        robust_iqr_divisor=normalization_config.robust_iqr_divisor,
        mad_consistency_divisor=normalization_config.mad_consistency_divisor,
        scale_epsilon=normalization_config.scale_epsilon,
        standard_ddof=normalization_config.standard_ddof,
    )
    artifact.validate()
    return artifact


def apply_fold_imu_channel_transform(
    values: np.ndarray,
    transform: FoldImuChannelTransform,
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply without touching RED/IR / 应用变换且不触碰 RED/IR。"""

    tensor = _validate_raw_tensor(values)
    transform.validate()
    mask = _valid_mask(valid_mask, (tensor.shape[0], tensor.shape[2]))
    output = tensor.copy()
    standardized = (tensor[:, 2:, :] - transform.center.reshape(
        1, len(IMU_CHANNEL_SCHEMA), 1)) / transform.scale.reshape(1, len(IMU_CHANNEL_SCHEMA), 1)
    output[:, 2:, :] = np.where(mask[:, None, :], standardized, 0.0)
    if not np.isfinite(output).all():
        raise ValueError("fold IMU transform produced nonfinite output")
    return output.astype(np.float32)


def transform_raw_windows_imu(
    windows: RawWindows,
    transform: FoldImuChannelTransform,
) -> RawWindows:
    """Apply artifact and bind provenance / 应用产物并绑定 provenance。"""

    values = apply_fold_imu_channel_transform(
        windows.values,
        transform,
        valid_mask=windows.valid_mask,
    )
    provenance = dict(windows.provenance)
    provenance.update({
        "imu_fold_standardized": transform.strategy != IMU_NONE,
        "imu_transform_applied": transform.strategy != IMU_NONE,
        "imu_transform_schema": transform.schema_version,
        "imu_transform_sha256": transform.artifact_sha256,
        "imu_transform_fitted_on_participant_ids": list(transform.fitted_on_participant_ids),
        "imu_normalization": transform.strategy,
        "imu_normalization_parameters": {
            "iqr_fallback": transform.iqr_fallback,
            "robust_iqr_divisor": transform.robust_iqr_divisor,
            "mad_consistency_divisor": transform.mad_consistency_divisor,
            "scale_epsilon": transform.scale_epsilon,
            "standard_ddof": transform.standard_ddof,
        },
    })
    return replace(windows, values=values, provenance=provenance)


__all__ = [
    "FoldImuChannelTransform",
    "IMU_CHANNEL_SCHEMA",
    "IMU_TRANSFORM_SCHEMA_VERSION",
    "IQR_NORMAL_CONSISTENCY_DIVISOR",
    "apply_fold_imu_channel_transform",
    "fit_fold_imu_channel_transform",
    "transform_raw_windows_imu",
]
