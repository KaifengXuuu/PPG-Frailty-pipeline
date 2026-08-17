"""Motion 8-channel reference plus named 11-channel derived augmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..provenance import assert_training_only, stable_payload_sha256
from ..signal.motion_imu import (
    MOTION_IMU_CHANNEL_SCHEMA,
    MOTION_IMU_CHANNEL_UNITS,
    MotionImuResult,
)
from ..signal.views import CANONICAL_FS_HZ


MOTION_REFERENCE_PROFILE_ID = "motion_8ch_axes_reference_v2"
MOTION_DERIVED_AUGMENTATION_PROFILE_ID = (
    "motion_11ch_derived_augmentation_ablation_v2"
)
MOTION_REFERENCE_IMU_CHANNEL_SCHEMA = MOTION_IMU_CHANNEL_SCHEMA[:6]
MOTION_REFERENCE_IMU_CHANNEL_UNITS = MOTION_IMU_CHANNEL_UNITS[:6]
MOTION_NETWORK_CHANNEL_SCHEMA = (
    "RED", "IR", *MOTION_REFERENCE_IMU_CHANNEL_SCHEMA,
)
MOTION_NETWORK_CHANNEL_UNITS = (
    "window_robust_z", "window_robust_z", *MOTION_REFERENCE_IMU_CHANNEL_UNITS,
)
MOTION_AUGMENTED_CHANNEL_SCHEMA = ("RED", "IR", *MOTION_IMU_CHANNEL_SCHEMA)
MOTION_AUGMENTED_CHANNEL_UNITS = (
    "window_robust_z", "window_robust_z", *MOTION_IMU_CHANNEL_UNITS,
)
MOTION_WINDOW_SECONDS = 8.0
MOTION_HOP_SECONDS = 2.0
MOTION_WINDOW_SAMPLES = 3200
MOTION_HOP_SAMPLES = 800
MOTION_REFERENCE_SCALER_SCHEMA = (
    "ppg_frailty.motion_axes6_outer_train_median_iqr_population_sd.v2"
)
MOTION_AUGMENTED_SCALER_SCHEMA = (
    "ppg_frailty.motion_derived9_outer_train_median_iqr_population_sd.v2"
)
MOTION_SCALER_SCHEMA = MOTION_REFERENCE_SCALER_SCHEMA


def _profile_components(
    profile_id: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], str]:
    if profile_id == MOTION_REFERENCE_PROFILE_ID:
        return (
            MOTION_NETWORK_CHANNEL_SCHEMA,
            MOTION_NETWORK_CHANNEL_UNITS,
            MOTION_REFERENCE_IMU_CHANNEL_SCHEMA,
            MOTION_REFERENCE_SCALER_SCHEMA,
        )
    if profile_id == MOTION_DERIVED_AUGMENTATION_PROFILE_ID:
        return (
            MOTION_AUGMENTED_CHANNEL_SCHEMA,
            MOTION_AUGMENTED_CHANNEL_UNITS,
            MOTION_IMU_CHANNEL_SCHEMA,
            MOTION_AUGMENTED_SCALER_SCHEMA,
        )
    raise ValueError(f"unknown motion tensor profile: {profile_id}")


def motion_network_schema_payload(
    profile_id: str = MOTION_REFERENCE_PROFILE_ID,
) -> dict[str, object]:
    channel_schema, channel_units, imu_schema, _ = _profile_components(profile_id)
    is_augmentation = profile_id == MOTION_DERIVED_AUGMENTATION_PROFILE_ID
    return {
        "schema_version": "ppg_frailty.motion_network_tensor.v2",
        "profile_id": profile_id,
        "profile_role": (
            "named_derived_signal_augmentation_ablation"
            if is_augmentation
            else "canonical_reference"
        ),
        "channel_schema": list(channel_schema),
        "channel_units": list(channel_units),
        "fs_hz": CANONICAL_FS_HZ,
        "window_s": MOTION_WINDOW_SECONDS,
        "hop_s": MOTION_HOP_SECONDS,
        "window_samples": MOTION_WINDOW_SAMPLES,
        "hop_samples": MOTION_HOP_SAMPLES,
        "ppg_normalization": "per_window_median_iqr_mad_then_one",
        "imu_normalization": (
            "outer_training_participant_only_median_iqr_"
            "population_sd_then_one"
        ),
        "imu_per_window_amplitude_normalization": False,
        "imu_channel_count": len(imu_schema),
        "derived_motion_channels_included": is_augmentation,
        "derived_motion_channels_are_frailty_predictors": False,
        "silent_channel_derivation": False,
        "tensor_layout": "window_channel_sample",
    }


MOTION_NETWORK_SCHEMA_SHA256 = stable_payload_sha256(motion_network_schema_payload())
MOTION_AUGMENTED_SCHEMA_SHA256 = stable_payload_sha256(
    motion_network_schema_payload(MOTION_DERIVED_AUGMENTATION_PROFILE_ID)
)


def _profile_schema_sha256(profile_id: str) -> str:
    _profile_components(profile_id)
    return (
        MOTION_NETWORK_SCHEMA_SHA256
        if profile_id == MOTION_REFERENCE_PROFILE_ID
        else MOTION_AUGMENTED_SCHEMA_SHA256
    )


@dataclass(frozen=True)
class MotionWindowTensors:
    """Unscaled-SI IMU windows; only RED/IR are normalized at materialization."""

    values: np.ndarray
    start_samples: np.ndarray
    record_id: str
    participant_id: str
    role_or_activity: str
    dataset_id: str
    profile_id: str = MOTION_REFERENCE_PROFILE_ID
    channel_schema: tuple[str, ...] = MOTION_NETWORK_CHANNEL_SCHEMA
    schema_sha256: str = MOTION_NETWORK_SCHEMA_SHA256

    def validate(self) -> None:
        tensor = np.asarray(self.values)
        starts = np.asarray(self.start_samples)
        channel_schema, _, _, _ = _profile_components(self.profile_id)
        expected_shape = (len(channel_schema), MOTION_WINDOW_SAMPLES)
        if tensor.ndim != 3 or tensor.shape[1:] != expected_shape:
            raise ValueError(
                "motion tensor shape does not match its declared reference/augmentation profile"
            )
        if tensor.shape[0] == 0 or starts.shape != (tensor.shape[0],):
            raise ValueError("motion tensor windows/start samples are misaligned")
        if not np.isfinite(tensor).all() or not np.issubdtype(starts.dtype, np.integer):
            raise ValueError("motion tensor must be finite with integer starts")
        if not self.record_id or not self.participant_id or not self.dataset_id:
            raise ValueError("motion tensor identity fields must be non-empty")
        if (
            self.channel_schema != channel_schema
            or self.schema_sha256 != _profile_schema_sha256(self.profile_id)
        ):
            raise ValueError("motion network tensor schema hash drift")


def _window_robust_scale(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    median = np.median(matrix, axis=0)
    q25, q75 = np.percentile(matrix, [25.0, 75.0], axis=0)
    scale = q75 - q25
    mad_scale = 1.4826 * np.median(np.abs(matrix - median), axis=0)
    scale = np.where(scale > 1e-12, scale, np.where(mad_scale > 1e-12, mad_scale, 1.0))
    return (matrix - median) / scale


def build_motion_window_tensors(
    ppg_red_ir: np.ndarray,
    motion_imu: MotionImuResult,
    *,
    record_id: str,
    participant_id: str,
    role_or_activity: str,
    dataset_id: str,
    fs_hz: float = CANONICAL_FS_HZ,
    profile_id: str = MOTION_REFERENCE_PROFILE_ID,
) -> MotionWindowTensors:
    """Materialize exact 8-s/2-s windows without per-window IMU normalization."""

    if float(fs_hz) != CANONICAL_FS_HZ:
        raise ValueError("formal motion windows require exactly 400 Hz")
    motion_imu.validate()
    channel_schema, _, imu_schema, _ = _profile_components(profile_id)
    ppg = np.asarray(ppg_red_ir, dtype=np.float64)
    imu = np.asarray(motion_imu.values, dtype=np.float64)
    if ppg.ndim != 2 or ppg.shape[1] != 2 or ppg.shape[0] != imu.shape[0]:
        raise ValueError("motion PPG/IMU must be aligned samples-by-[2/9]")
    if not np.isfinite(ppg).all():
        raise ValueError("motion PPG source must be finite")
    if ppg.shape[0] < MOTION_WINDOW_SAMPLES:
        raise ValueError("motion recording is shorter than the frozen 8-second window")
    starts = np.arange(
        0,
        ppg.shape[0] - MOTION_WINDOW_SAMPLES + 1,
        MOTION_HOP_SAMPLES,
        dtype=np.int64,
    )
    tensor = np.empty(
        (starts.size, len(channel_schema), MOTION_WINDOW_SAMPLES),
        dtype=np.float32,
    )
    for index, start in enumerate(starts):
        stop = int(start + MOTION_WINDOW_SAMPLES)
        tensor[index, :2] = _window_robust_scale(ppg[start:stop]).T.astype(np.float32)
        tensor[index, 2:] = imu[start:stop, :len(imu_schema)].T.astype(np.float32)
    result = MotionWindowTensors(
        values=tensor,
        start_samples=starts,
        record_id=str(record_id),
        participant_id=str(participant_id),
        role_or_activity=str(role_or_activity),
        dataset_id=str(dataset_id),
        profile_id=profile_id,
        channel_schema=channel_schema,
        schema_sha256=_profile_schema_sha256(profile_id),
    )
    result.validate()
    return result


def _scaler_hash(
    center: np.ndarray,
    scale: np.ndarray,
    count: np.ndarray,
    participant_ids: tuple[str, ...],
    *,
    profile_id: str,
    schema_version: str,
    channel_schema: tuple[str, ...],
) -> str:
    return stable_payload_sha256(
        {
            "profile_id": profile_id,
            "schema_version": schema_version,
            "imu_channel_schema": list(channel_schema),
            "center": center.tolist(),
            "scale": scale.tolist(),
            "valid_count": count.tolist(),
            "fitted_on_participant_ids": list(participant_ids),
            "per_window_imu_scaling": False,
        }
    )


@dataclass(frozen=True)
class MotionFoldImuTransform:
    center: np.ndarray
    scale: np.ndarray
    valid_count: np.ndarray
    fitted_on_participant_ids: tuple[str, ...]
    artifact_sha256: str
    profile_id: str = MOTION_REFERENCE_PROFILE_ID
    schema_version: str = MOTION_REFERENCE_SCALER_SCHEMA
    channel_schema: tuple[str, ...] = MOTION_REFERENCE_IMU_CHANNEL_SCHEMA

    def validate(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        count = np.asarray(self.valid_count, dtype=np.int64)
        _, _, expected_channels, expected_schema = _profile_components(self.profile_id)
        expected_shape = (len(expected_channels),)
        if (
            center.shape != expected_shape
            or scale.shape != expected_shape
            or count.shape != expected_shape
        ):
            raise ValueError("motion fold scaler does not match its tensor profile")
        if not np.isfinite(center).all() or not np.isfinite(scale).all():
            raise ValueError("motion fold scaler statistics must be finite")
        if np.any(scale <= 0.0) or np.any(count <= 0) or not self.fitted_on_participant_ids:
            raise ValueError("motion fold scaler has invalid scale/count/roster")
        if (
            self.schema_version != expected_schema
            or self.channel_schema != expected_channels
            or self.artifact_sha256
            != _scaler_hash(
                center,
                scale,
                count,
                self.fitted_on_participant_ids,
                profile_id=self.profile_id,
                schema_version=self.schema_version,
                channel_schema=self.channel_schema,
            )
        ):
            raise ValueError("motion fold scaler artifact identity drift")


def fit_motion_fold_imu_transform(
    values: np.ndarray,
    participant_ids: Iterable[str],
    *,
    fitted_on_participant_ids: Iterable[str],
    outer_train_participant_ids: Iterable[str],
    outer_oof_participant_ids: Iterable[str],
    profile_id: str | None = None,
) -> MotionFoldImuTransform:
    """Fit the declared reference or augmentation IMU channels on train participants."""

    fitted = assert_training_only(
        fitted_on_participant_ids,
        outer_train_participant_ids,
        outer_oof_participant_ids,
    )
    tensor = np.asarray(values, dtype=np.float64)
    ids = tuple(str(value) for value in participant_ids)
    inferred_profile = (
        MOTION_REFERENCE_PROFILE_ID
        if tensor.ndim == 3 and tensor.shape[1] == len(MOTION_NETWORK_CHANNEL_SCHEMA)
        else MOTION_DERIVED_AUGMENTATION_PROFILE_ID
        if tensor.ndim == 3 and tensor.shape[1] == len(MOTION_AUGMENTED_CHANNEL_SCHEMA)
        else None
    )
    selected_profile = profile_id or inferred_profile
    if selected_profile is None or inferred_profile != selected_profile:
        raise ValueError("motion scaler input does not match a declared tensor profile")
    channel_schema, _, imu_schema, scaler_schema = _profile_components(selected_profile)
    if tensor.shape[1:] != (len(channel_schema), MOTION_WINDOW_SAMPLES):
        raise ValueError("motion scaler input shape/profile mismatch")
    if tensor.shape[0] != len(ids) or not np.isfinite(tensor).all():
        raise ValueError("motion scaler input IDs/data are invalid")
    selected = np.asarray([value in set(fitted) for value in ids], dtype=bool)
    if {value for value, keep in zip(ids, selected) if keep} != set(fitted):
        raise ValueError("not every declared train participant has a motion window")
    imu_count = len(imu_schema)
    center = np.empty(imu_count, dtype=np.float64)
    scale = np.empty(imu_count, dtype=np.float64)
    count = np.empty(imu_count, dtype=np.int64)
    for channel in range(imu_count):
        samples = tensor[selected, channel + 2, :].reshape(-1)
        count[channel] = samples.size
        center[channel] = float(np.median(samples))
        q25, q75 = np.percentile(samples, [25.0, 75.0])
        candidate = float(q75 - q25)
        if not np.isfinite(candidate) or candidate <= 1e-12:
            candidate = float(np.std(samples, ddof=0))
        scale[channel] = (
            candidate
            if np.isfinite(candidate) and candidate > 1e-12
            else 1.0
        )
    result = MotionFoldImuTransform(
        center=center,
        scale=scale,
        valid_count=count,
        fitted_on_participant_ids=fitted,
        artifact_sha256=_scaler_hash(
            center,
            scale,
            count,
            fitted,
            profile_id=selected_profile,
            schema_version=scaler_schema,
            channel_schema=imu_schema,
        ),
        profile_id=selected_profile,
        schema_version=scaler_schema,
        channel_schema=imu_schema,
    )
    result.validate()
    return result


def apply_motion_fold_imu_transform(
    values: np.ndarray,
    transform: MotionFoldImuTransform,
) -> np.ndarray:
    """Scale profile IMU channels while preserving window-scaled RED/IR exactly."""

    tensor = np.asarray(values, dtype=np.float64)
    transform.validate()
    channel_schema, _, imu_schema, _ = _profile_components(transform.profile_id)
    if tensor.ndim != 3 or tensor.shape[1:] != (
        len(channel_schema),
        MOTION_WINDOW_SAMPLES,
    ):
        raise ValueError("motion transform input shape/profile mismatch")
    output = tensor.copy()
    output[:, 2:, :] = (
        tensor[:, 2:, :] - transform.center.reshape(1, len(imu_schema), 1)
    ) / transform.scale.reshape(1, len(imu_schema), 1)
    if not np.isfinite(output).all():
        raise ValueError("motion fold IMU transform produced nonfinite output")
    return output.astype(np.float32)


__all__ = [
    "MOTION_HOP_SAMPLES",
    "MOTION_HOP_SECONDS",
    "MOTION_AUGMENTED_CHANNEL_SCHEMA",
    "MOTION_AUGMENTED_CHANNEL_UNITS",
    "MOTION_AUGMENTED_SCHEMA_SHA256",
    "MOTION_AUGMENTED_SCALER_SCHEMA",
    "MOTION_DERIVED_AUGMENTATION_PROFILE_ID",
    "MOTION_NETWORK_CHANNEL_SCHEMA",
    "MOTION_NETWORK_CHANNEL_UNITS",
    "MOTION_NETWORK_SCHEMA_SHA256",
    "MOTION_SCALER_SCHEMA",
    "MOTION_REFERENCE_IMU_CHANNEL_SCHEMA",
    "MOTION_REFERENCE_IMU_CHANNEL_UNITS",
    "MOTION_REFERENCE_PROFILE_ID",
    "MOTION_REFERENCE_SCALER_SCHEMA",
    "MOTION_WINDOW_SAMPLES",
    "MOTION_WINDOW_SECONDS",
    "MotionFoldImuTransform",
    "MotionWindowTensors",
    "apply_motion_fold_imu_transform",
    "build_motion_window_tensors",
    "fit_motion_fold_imu_transform",
    "motion_network_schema_payload",
]
