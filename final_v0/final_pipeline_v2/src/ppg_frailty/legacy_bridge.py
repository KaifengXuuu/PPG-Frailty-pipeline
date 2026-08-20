"""Executable L0--L7 contracts for the historical-to-V2 bridge study.

This module is intentionally outside the canonical configuration validator.
It implements only the frozen nine-case protocol recorded by
``stage3_alter.yaml`` and does not add new general-purpose pipeline defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any, Mapping

import numpy as np
from scipy import signal

from .representations.raw import RawWindows
from .training.aggregation import LINE_A_EQUAL_FILES, LINE_B_EQUAL_ROLE_FAMILIES
from .training.legacy_bridge import LegacyBridgeTrainingConfig


@dataclass(frozen=True)
class LegacyBridgeProfile:
    """One cumulative profile in the reviewed CompactCNN L0--L7 chain."""

    profile_id: str
    preprocessing: str
    target_fs_hz: float
    window_seconds: float
    hop_seconds: float
    historical_retained_fraction: float | None
    max_windows_per_file: int | None
    imu_normalization: str
    sampler: str
    class_weighting: str
    optimizer: str
    batch_size: int
    expected_aggregation_rule: str
    fixed_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        if self.profile_id not in {f"L{value}" for value in range(8)}:
            raise ValueError("legacy bridge profile must be L0..L7")
        if self.preprocessing not in {"legacy_raw_filtered", "v2_calibrated_ekf"}:
            raise ValueError("unsupported bridge preprocessing")
        if self.target_fs_hz not in {64.0, 400.0}:
            raise ValueError("bridge target fs must be 64 or 400 Hz")
        if self.imu_normalization not in {
            "per_window_all_eight_channels",
            "outer_train_axes6_fold_robust",
        }:
            raise ValueError("unsupported bridge IMU normalization")
        if self.fixed_epochs != 10:
            raise ValueError("bridge profile must remain fixed at ten epochs")

    @property
    def uses_legacy_preprocessing(self) -> bool:
        return self.preprocessing == "legacy_raw_filtered"

    @property
    def uses_fold_imu_transform(self) -> bool:
        return self.imu_normalization == "outer_train_axes6_fold_robust"

    @property
    def channel_schema(self) -> tuple[str, ...]:
        if self.uses_legacy_preprocessing:
            return ("RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ")
        return (
            "RED",
            "IR",
            "A_dyn_x",
            "A_dyn_y",
            "A_dyn_z",
            "GX",
            "GY",
            "GZ",
        )

    def training_config(self, *, device: str = "cpu") -> LegacyBridgeTrainingConfig:
        balance = (
            "equal_role_families"
            if self.sampler == "balance_line_weighted_v2"
            else (
                "uniform_replacement"
                if self.sampler == "uniform_replacement"
                else "legacy_exhaustive"
            )
        )
        return LegacyBridgeTrainingConfig(
            profile_id=self.profile_id,
            sampler=self.sampler,
            class_weighting=self.class_weighting,
            optimizer=self.optimizer,
            training_balance=balance,
            expected_aggregation_rule=self.expected_aggregation_rule,
            batch_size=self.batch_size,
            fixed_epochs=self.fixed_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            device=device,
            seed=42,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_legacy_bridge_profile(profile_id: str) -> LegacyBridgeProfile:
    """Resolve one profile without accepting arbitrary bridge combinations."""

    profile = str(profile_id).upper()
    if profile not in {f"L{value}" for value in range(8)}:
        raise ValueError(f"unknown legacy bridge profile: {profile_id!r}")
    level = int(profile[1:])
    return LegacyBridgeProfile(
        profile_id=profile,
        preprocessing=("legacy_raw_filtered" if level <= 2 else "v2_calibrated_ekf"),
        target_fs_hz=(64.0 if level <= 1 else 400.0),
        window_seconds=(15.0 if level == 0 else 5.0),
        hop_seconds=(3.0 if level == 0 else 2.5),
        historical_retained_fraction=(0.9 if level == 0 else None),
        max_windows_per_file=(None if level == 0 else 128),
        imu_normalization=(
            "per_window_all_eight_channels"
            if level <= 3
            else "outer_train_axes6_fold_robust"
        ),
        sampler=(
            "exhaustive_shuffle_without_replacement"
            if level <= 4
            else "uniform_replacement"
            if level == 5
            else "balance_line_weighted_v2"
        ),
        class_weighting=(
            "outer_train_window_inverse_frequency"
            if level <= 5
            else "outer_train_inverse_frequency"
        ),
        optimizer=("adam" if level == 7 else "adamw"),
        batch_size=(64 if level == 7 else 32),
        expected_aggregation_rule=(
            LINE_B_EQUAL_ROLE_FAMILIES if level >= 6 else LINE_A_EQUAL_FILES
        ),
    )


def bridge_profile_from_case(
    case_id: str,
    profiles: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
) -> LegacyBridgeProfile:
    """Bind a study case to its reviewed display profile exactly once."""

    matches = [row for row in profiles if str(row.get("catalog_case_id")) == str(case_id)]
    if len(matches) != 1:
        raise ValueError(f"bridge case must map to exactly one profile: {case_id}")
    return resolve_legacy_bridge_profile(str(matches[0]["profile_id"]))


def _filter_sos(values: np.ndarray, *, fs_hz: float, cutoff: Any, kind: str) -> np.ndarray:
    sos = signal.butter(3, cutoff, btype=kind, fs=float(fs_hz), output="sos")
    return signal.sosfiltfilt(sos, np.asarray(values, dtype=np.float64), axis=0)


def _robust_scale_all_channels(segment: np.ndarray) -> np.ndarray:
    values = np.asarray(segment, dtype=np.float32)
    center = np.median(values, axis=0, keepdims=True)
    q25, q75 = np.percentile(values, [25.0, 75.0], axis=0)
    scale = (q75 - q25) / 1.349
    standard_deviation = np.std(values, axis=0, keepdims=False)
    scale = np.where(scale > 1e-6, scale, standard_deviation)
    normalized = (values - center) / (scale.reshape(1, -1) + 1e-6)
    normalized = np.where(np.isfinite(normalized), normalized, 0.0)
    return np.clip(normalized, -8.0, 8.0).astype(np.float32)


def _window_starts(
    n_samples: int,
    *,
    fs_hz: float,
    profile: LegacyBridgeProfile,
) -> np.ndarray:
    window = int(round(profile.window_seconds * float(fs_hz)))
    hop = int(round(profile.hop_seconds * float(fs_hz)))
    if window <= 0 or hop <= 0 or n_samples <= 0:
        raise ValueError("bridge recording/window dimensions must be positive")
    if n_samples < window:
        if profile.profile_id != "L0":
            return np.empty(0, dtype=np.int64)
        return np.asarray([0], dtype=np.int64)
    if n_samples == window:
        return np.asarray([0], dtype=np.int64)
    starts = np.arange(0, n_samples - window + 1, hop, dtype=np.int64)
    right = n_samples - window
    if starts.size == 0 or int(starts[-1]) != right:
        starts = np.append(starts, right)
    cap: int | None = profile.max_windows_per_file
    if profile.historical_retained_fraction is not None:
        cap = max(
            1,
            int(np.ceil(starts.size * float(profile.historical_retained_fraction))),
        )
    if cap is not None and starts.size > cap:
        selected = np.linspace(0, starts.size - 1, int(cap)).round().astype(np.int64)
        if np.unique(selected).size != int(cap):
            raise RuntimeError("bridge uniform window retention produced duplicates")
        starts = starts[selected]
    return starts.astype(np.int64)


def _raw_windows_from_matrix(
    matrix: np.ndarray,
    *,
    fs_hz: float,
    profile: LegacyBridgeProfile,
    valid_mask: np.ndarray | None,
    provenance: Mapping[str, Any],
) -> RawWindows:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 8 or not np.isfinite(values).all():
        raise ValueError("bridge matrix must be finite samples x 8 channels")
    valid = (
        np.ones(values.shape[0], dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != (values.shape[0],):
        raise ValueError("bridge validity mask must align with samples")
    starts = _window_starts(values.shape[0], fs_hz=fs_hz, profile=profile)
    window = int(round(profile.window_seconds * float(fs_hz)))
    rows: list[np.ndarray] = []
    retained_starts: list[int] = []
    dropped = 0
    for start in starts:
        stop = min(int(start) + window, values.shape[0])
        if not np.all(valid[int(start) : stop]):
            dropped += 1
            continue
        segment = _robust_scale_all_channels(values[int(start) : stop])
        if segment.shape[0] < window:
            if profile.profile_id != "L0":
                raise ValueError(
                    "only historical L0 permits right-zero padding of a short record"
                )
            segment = np.pad(
                segment,
                ((0, window - segment.shape[0]), (0, 0)),
                mode="constant",
            )
        rows.append(segment.T)
        retained_starts.append(int(start))
    if not rows:
        raise ValueError("bridge window materialization produced no valid windows")
    return RawWindows(
        values=np.stack(rows).astype(np.float32),
        valid_mask=np.ones((len(rows), window), dtype=bool),
        start_samples=np.asarray(retained_starts, dtype=np.int64),
        candidate_count=int(starts.size),
        dropped_invalid_count=int(dropped),
        provenance={
            "schema_version": "ppg_frailty.legacy_bridge_raw_windows.v1",
            "profile": profile.to_dict(),
            "normalization": (
                "per_window_all_8ch_median_iqr_over_1p349_sd_fallback_clip_-8_8"
            ),
            "short_record_policy": (
                "historical_L0_scale_available_rows_then_right_zero_pad"
                if profile.profile_id == "L0"
                else "complete_windows_only_no_padding"
            ),
            **dict(provenance),
        },
    )


def build_legacy_bridge_raw_windows(
    record: Mapping[str, Any],
    profile: LegacyBridgeProfile,
) -> RawWindows:
    """Freshly preprocess one L0--L2 raw record and build retained windows."""

    if not profile.uses_legacy_preprocessing:
        raise ValueError("legacy raw preprocessing is valid only for L0..L2")
    source_fs = float(record["fs_hz"])
    if source_fs != 400.0:
        raise ValueError("legacy bridge source grid must be the audited 400 Hz grid")
    ppg = np.asarray(record["ppg"], dtype=np.float64)
    acc = np.asarray(record["acc"], dtype=np.float64)
    gyro = np.asarray(record["gyro"], dtype=np.float64)
    if (
        ppg.ndim != 2
        or ppg.shape[1] != 2
        or acc.shape != (ppg.shape[0], 3)
        or gyro.shape != (ppg.shape[0], 3)
        or not np.isfinite(np.column_stack((ppg, acc, gyro))).all()
    ):
        raise ValueError("legacy bridge raw channels are not aligned finite 8-channel data")
    ppg_filtered = _filter_sos(
        signal.detrend(ppg, axis=0, type="linear"),
        fs_hz=source_fs,
        cutoff=(0.2, 8.0),
        kind="bandpass",
    )
    acc_filtered = _filter_sos(
        acc,
        fs_hz=source_fs,
        cutoff=20.0,
        kind="lowpass",
    )
    gyro_filtered = _filter_sos(
        gyro,
        fs_hz=source_fs,
        cutoff=40.0,
        kind="lowpass",
    )
    # Historical cache materialization rounded the filtered 8-channel matrix
    # to float32 before resampling, then rounded the resampled output again.
    matrix = np.column_stack(
        (ppg_filtered, acc_filtered, gyro_filtered)
    ).astype(np.float32)
    target_fs = float(profile.target_fs_hz)
    if target_fs != source_fs:
        ratio = Fraction(str(target_fs / source_fs)).limit_denominator(10_000)
        matrix = signal.resample_poly(
            matrix,
            up=ratio.numerator,
            down=ratio.denominator,
            axis=0,
        ).astype(np.float32)
    return _raw_windows_from_matrix(
        matrix,
        fs_hz=target_fs,
        profile=profile,
        valid_mask=None,
        provenance={
            "preprocessing": (
                "PPG_detrend_order3_zero_phase_0p2_8Hz;"
                "ACC_order3_LPF20Hz;GYRO_order3_LPF40Hz;"
                "no_SI_no_B_calibration_no_EKF_no_gravity_removal"
            ),
            "resampling": (
                "not_applied_native_400Hz"
                if target_fs == source_fs
                else "historical_scipy_resample_poly_default_constant_pad_400_to_64Hz"
            ),
            "channel_schema": profile.channel_schema,
        },
    )


def build_v2_window_scaled_bridge_raw_windows(
    views: Any,
    profile: LegacyBridgeProfile,
) -> RawWindows:
    """Build L3 windows: V2 IMU semantics with legacy per-window 8ch scaling."""

    if profile.profile_id != "L3":
        raise ValueError("V2 per-window bridge materialization is specific to L3")
    views.validate()
    dynamic = np.asarray(views.imu_processed["dynamic_acc_mps2"], dtype=np.float64)
    gyro = np.asarray(views.imu_processed["gyro_rads"], dtype=np.float64)
    matrix = np.column_stack((np.asarray(views.x_filter), dynamic, gyro))
    valid = np.asarray(
        views.imu_processed.get(
            "imu_valid_mask", np.ones(matrix.shape[0], dtype=bool)
        ),
        dtype=bool,
    )
    return _raw_windows_from_matrix(
        matrix,
        fs_hz=400.0,
        profile=profile,
        valid_mask=valid,
        provenance={
            "preprocessing": "canonical_V2_SI_B_calibration_roll_pitch_EKF",
            "channel_schema": profile.channel_schema,
            "imu_fold_transform": "not_applied_L3_per_window_ablation",
        },
    )


__all__ = [
    "LegacyBridgeProfile",
    "bridge_profile_from_case",
    "build_legacy_bridge_raw_windows",
    "build_v2_window_scaled_bridge_raw_windows",
    "resolve_legacy_bridge_profile",
]
