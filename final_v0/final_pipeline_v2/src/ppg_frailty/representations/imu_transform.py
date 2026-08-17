"""Outer-train-only IMU scaling for raw/fusion / raw/fusion 的外层训练折 IMU 缩放。

English: RED/IR are already window-scaled. This artifact fits the six physical
IMU axes used by the frailty tensor on declared outer-train participants and
preserves a hash-bound roster.
OOF fitting is rejected.
中文：RED/IR 已在窗口内缩放。本产物仅用声明的 outer-train participant 拟合
六个 IMU 轴，并将 roster 绑定到哈希；任何 OOF 拟合都会被拒绝。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from ..provenance import assert_training_only, stable_payload_sha256
from .raw import RawWindows


IMU_TRANSFORM_SCHEMA_VERSION = (
    "raw_frailty_imu_axes6_outer_train_median_iqr_population_sd_v2"
)
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
) -> str:
    return stable_payload_sha256({
        "schema_version": IMU_TRANSFORM_SCHEMA_VERSION,
        "channel_schema": list(IMU_CHANNEL_SCHEMA),
        "center": np.asarray(center, dtype=np.float64).tolist(),
        "scale": np.asarray(scale, dtype=np.float64).tolist(),
        "valid_count": np.asarray(valid_count, dtype=np.int64).tolist(),
        "fitted_on_participant_ids": list(fitted_ids),
        "per_window_imu_scaling": False,
    })


@dataclass(frozen=True)
class FoldImuChannelTransform:
    """Six-axis robust scaling artifact / 六轴稳健缩放产物。"""

    center: np.ndarray
    scale: np.ndarray
    valid_count: np.ndarray
    fitted_on_participant_ids: tuple[str, ...]
    artifact_sha256: str
    channel_schema: tuple[str, ...] = IMU_CHANNEL_SCHEMA
    schema_version: str = IMU_TRANSFORM_SCHEMA_VERSION

    def validate(self) -> None:
        """Audit alignment and identity / 审计对齐及身份。"""

        center = np.asarray(self.center, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        count = np.asarray(self.valid_count, dtype=np.int64)
        expected_shape = (len(IMU_CHANNEL_SCHEMA),)
        if center.shape != expected_shape or scale.shape != expected_shape or count.shape != expected_shape:
            raise ValueError("frailty IMU transform arrays must align with six axes")
        if not np.isfinite(center).all() or not np.isfinite(scale).all():
            raise ValueError("IMU transform statistics must be finite")
        if np.any(scale <= 0.0) or np.any(count <= 0):
            raise ValueError("IMU transform scale/count is invalid")
        if not self.fitted_on_participant_ids:
            raise ValueError("IMU transform requires an outer-train roster")
        expected = _artifact_hash(center, scale, count, self.fitted_on_participant_ids)
        if (
            self.schema_version != IMU_TRANSFORM_SCHEMA_VERSION
            or self.channel_schema != IMU_CHANNEL_SCHEMA
            or self.artifact_sha256 != expected
        ):
            raise ValueError("IMU transform artifact identity drift")


def _validate_raw_tensor(values: np.ndarray) -> np.ndarray:
    tensor = np.asarray(values, dtype=np.float64)
    if (
        tensor.ndim != 3
        or tensor.shape[1] != len(RAW_CHANNEL_SCHEMA)
        or tensor.shape[0] == 0
        or tensor.shape[2] == 0
    ):
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
) -> FoldImuChannelTransform:
    """Fit six IMU axes on declared train rows / 仅在声明训练行拟合六轴。"""

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
        center[channel] = float(np.median(samples))
        q25, q75 = np.percentile(samples, [25.0, 75.0])
        robust_scale = float(q75 - q25)
        if not np.isfinite(robust_scale) or robust_scale <= 1e-12:
            robust_scale = float(np.std(samples, ddof=0))
        scale[channel] = (
            robust_scale
            if np.isfinite(robust_scale) and robust_scale > 1e-12
            else 1.0
        )
    artifact = FoldImuChannelTransform(
        center=center,
        scale=scale,
        valid_count=count,
        fitted_on_participant_ids=fitted,
        artifact_sha256=_artifact_hash(center, scale, count, fitted),
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
    standardized = (
        tensor[:, 2:, :] - transform.center.reshape(1, len(IMU_CHANNEL_SCHEMA), 1)
    ) / transform.scale.reshape(1, len(IMU_CHANNEL_SCHEMA), 1)
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
        "imu_fold_standardized": True,
        "imu_transform_schema": transform.schema_version,
        "imu_transform_sha256": transform.artifact_sha256,
        "imu_transform_fitted_on_participant_ids": list(transform.fitted_on_participant_ids),
        "imu_normalization": (
            "outer_train_median_iqr_population_sd_then_one"
        ),
    })
    return replace(windows, values=values, provenance=provenance)


__all__ = [
    "FoldImuChannelTransform",
    "IMU_CHANNEL_SCHEMA",
    "IMU_TRANSFORM_SCHEMA_VERSION",
    "apply_fold_imu_channel_transform",
    "fit_fold_imu_channel_transform",
    "transform_raw_windows_imu",
]
