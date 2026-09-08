"""Executable historical-to-V2 bridge profiles.

This module is intentionally outside the canonical configuration validator.
It preserves the frozen L0--L7 cumulative protocol and also accepts complete,
hash-bound, field-driven profiles for Stage-3 centred-star and focused
follow-up plans. A profile identifier is provenance only: algorithms dispatch
on declared fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from typing import Any, Mapping

import numpy as np
from scipy import signal

from .representations.raw import RawWindows
from .training.aggregation import LINE_A_EQUAL_FILES, LINE_B_EQUAL_ROLE_FAMILIES
from .training.legacy_bridge import (
    BRIDGE_CLASS_WEIGHTING,
    BRIDGE_SAMPLERS,
    LegacyBridgeTrainingConfig,
)

FIELD_DRIVEN_PROTOCOL_DESIGNS = frozenset({"centered_star_v1", "field_driven_followup_v1"})


@dataclass(frozen=True)
class LegacyBridgeProfile:
    """One resolved bridge profile with field-driven runtime controls."""

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
    protocol_design: str = "cumulative_chain_v1"
    ppg_preprocessing: str | None = None
    imu_preprocessing: str | None = None
    allow_short_record_padding: bool | None = None
    primary_report_aggregation_view: str | None = None
    declared_controls: Mapping[str, Any] | None = None
    profile_definition_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.profile_id or not self.profile_id.replace("_", "").isalnum():
            raise ValueError("bridge profile_id must be a non-empty safe identifier")
        if self.protocol_design not in {
                "cumulative_chain_v1",
                *FIELD_DRIVEN_PROTOCOL_DESIGNS,
        }:
            raise ValueError("unsupported bridge protocol design")
        if self.protocol_design == "cumulative_chain_v1" and self.profile_id not in {f"L{value}" for value in range(8)}:
            raise ValueError("cumulative legacy bridge profile must be L0..L7")
        if self.preprocessing not in {
                "legacy_raw_filtered",
                "v2_calibrated_ekf",
                "field_driven",
        }:
            raise ValueError("unsupported bridge preprocessing")
        if not np.isfinite(self.target_fs_hz) or self.target_fs_hz <= 0.0:
            raise ValueError("bridge target fs must be finite and positive")
        if self.window_seconds <= 0.0 or self.hop_seconds <= 0.0:
            raise ValueError("bridge window and hop must be positive")
        if self.historical_retained_fraction is not None and not (0.0 < self.historical_retained_fraction <= 1.0):
            raise ValueError("historical retained fraction must lie in (0,1]")
        if self.max_windows_per_file is not None and self.max_windows_per_file <= 0:
            raise ValueError("bridge window cap must be positive when supplied")
        if self.imu_normalization not in {
                "per_window_all_eight_channels",
                "outer_train_axes6_fold_robust",
        }:
            raise ValueError("unsupported bridge IMU normalization")
        if self.sampler not in BRIDGE_SAMPLERS:
            raise ValueError("unsupported bridge sampler")
        if self.class_weighting not in BRIDGE_CLASS_WEIGHTING:
            raise ValueError("unsupported bridge class weighting")
        if self.optimizer not in {"adamw", "adam"}:
            raise ValueError("unsupported bridge optimizer")
        if self.fixed_epochs <= 0 or self.batch_size <= 0:
            raise ValueError("bridge epochs and batch size must be positive")
        if (not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0 or not np.isfinite(self.weight_decay)
                or self.weight_decay < 0.0):
            raise ValueError("bridge learning rate/weight decay are invalid")
        if self.resolved_ppg_preprocessing not in {
                "legacy_detrend_bandpass_0p2_8",
                "canonical_v2",
        }:
            raise ValueError("unsupported bridge PPG preprocessing")
        if self.resolved_imu_preprocessing not in {
                "legacy_filtered_axes",
                "calibrated_ekf_adyn",
        }:
            raise ValueError("unsupported bridge IMU preprocessing")
        if self.expected_aggregation_rule not in {
                LINE_A_EQUAL_FILES,
                LINE_B_EQUAL_ROLE_FAMILIES,
        }:
            raise ValueError("unsupported training-metric aggregation rule")
        if self.resolved_primary_report_aggregation_view not in {
                "window_balanced_to_participant",
                LINE_A_EQUAL_FILES,
                LINE_B_EQUAL_ROLE_FAMILIES,
        }:
            raise ValueError("unsupported primary report aggregation view")

    @property
    def resolved_ppg_preprocessing(self) -> str:
        if self.ppg_preprocessing is not None:
            return str(self.ppg_preprocessing)
        return "legacy_detrend_bandpass_0p2_8" if self.preprocessing == "legacy_raw_filtered" else "canonical_v2"

    @property
    def resolved_imu_preprocessing(self) -> str:
        if self.imu_preprocessing is not None:
            return str(self.imu_preprocessing)
        return "legacy_filtered_axes" if self.preprocessing == "legacy_raw_filtered" else "calibrated_ekf_adyn"

    @property
    def resolved_allow_short_record_padding(self) -> bool:
        if self.allow_short_record_padding is not None:
            return bool(self.allow_short_record_padding)
        return self.profile_id == "L0"

    @property
    def resolved_primary_report_aggregation_view(self) -> str:
        if self.primary_report_aggregation_view is not None:
            return str(self.primary_report_aggregation_view)
        return self.expected_aggregation_rule

    @property
    def uses_legacy_preprocessing(self) -> bool:
        return (self.resolved_ppg_preprocessing == "legacy_detrend_bandpass_0p2_8"
                and self.resolved_imu_preprocessing == "legacy_filtered_axes")

    @property
    def builds_windows_from_raw_record(self) -> bool:
        return self.resolved_ppg_preprocessing == "legacy_detrend_bandpass_0p2_8"

    @property
    def requires_calibrated_imu_views(self) -> bool:
        return self.builds_windows_from_raw_record and self.resolved_imu_preprocessing == "calibrated_ekf_adyn"

    @property
    def uses_canonical_all_channel_window_scaling(self) -> bool:
        return not self.builds_windows_from_raw_record and self.imu_normalization == "per_window_all_eight_channels"

    @property
    def uses_fold_imu_transform(self) -> bool:
        return self.imu_normalization == "outer_train_axes6_fold_robust"

    @property
    def channel_schema(self) -> tuple[str, ...]:
        if self.resolved_imu_preprocessing == "legacy_filtered_axes":
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
        balance = ("equal_role_families" if self.sampler == "balance_line_weighted_v2" else
                   ("uniform_replacement" if self.sampler == "uniform_replacement" else "legacy_exhaustive"))
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
            protocol_design=self.protocol_design,
        )

    def to_dict(self) -> dict[str, Any]:
        if self.protocol_design in FIELD_DRIVEN_PROTOCOL_DESIGNS:
            return {
                "profile_id": self.profile_id,
                "protocol_design": self.protocol_design,
                "controls": dict(self.declared_controls or {}),
                "profile_definition_sha256": self.profile_definition_sha256,
                "training_identity_sha256": self.training_identity_sha256,
            }
        # Preserve the exact historical L-profile payload and therefore its
        # existing effective-config hashes/provenance schema.
        return {
            "profile_id": self.profile_id,
            "preprocessing": self.preprocessing,
            "target_fs_hz": self.target_fs_hz,
            "window_seconds": self.window_seconds,
            "hop_seconds": self.hop_seconds,
            "historical_retained_fraction": self.historical_retained_fraction,
            "max_windows_per_file": self.max_windows_per_file,
            "imu_normalization": self.imu_normalization,
            "sampler": self.sampler,
            "class_weighting": self.class_weighting,
            "optimizer": self.optimizer,
            "batch_size": self.batch_size,
            "expected_aggregation_rule": self.expected_aggregation_rule,
            "fixed_epochs": self.fixed_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }

    @property
    def training_identity_payload(self) -> dict[str, Any]:
        """Controls that can affect materialization, training, or prediction."""

        return {
            "ppg_preprocessing": self.resolved_ppg_preprocessing,
            "imu_preprocessing": self.resolved_imu_preprocessing,
            "target_fs_hz": float(self.target_fs_hz),
            "window_seconds": float(self.window_seconds),
            "hop_seconds": float(self.hop_seconds),
            "historical_retained_fraction": self.historical_retained_fraction,
            "max_windows_per_file": self.max_windows_per_file,
            "allow_short_record_padding": self.resolved_allow_short_record_padding,
            "normalization": self.imu_normalization,
            "sampler": self.sampler,
            "class_weighting": self.class_weighting,
            "optimizer": self.optimizer,
            "batch_size": int(self.batch_size),
            "fixed_epochs": int(self.fixed_epochs),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "training_metric_aggregation_rule": self.expected_aggregation_rule,
        }

    @property
    def training_identity_sha256(self) -> str:
        return _stable_mapping_sha256(self.training_identity_payload)


_STAR_CONTROL_KEYS = frozenset({
    "ppg_preprocessing",
    "imu_preprocessing",
    "target_fs_hz",
    "window_seconds",
    "hop_seconds",
    "historical_retained_fraction",
    "max_windows_per_file",
    "allow_short_record_padding",
    "normalization",
    "sampler",
    "class_weighting",
    "optimizer",
    "batch_size",
    "fixed_epochs",
    "learning_rate",
    "weight_decay",
    "training_metric_aggregation_rule",
    "primary_report_aggregation_view",
})


def _stable_mapping_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_legacy_bridge_profile(
    profile_id: str,
    *,
    protocol_design: str = "cumulative_chain_v1",
    profile_definition: Mapping[str, Any] | None = None,
    profile_definition_sha256: str | None = None,
) -> LegacyBridgeProfile:
    """Resolve a frozen cumulative or complete hash-bound explicit profile."""

    profile = str(profile_id).upper()
    if protocol_design in FIELD_DRIVEN_PROTOCOL_DESIGNS:
        if not isinstance(profile_definition, Mapping):
            raise ValueError("field-driven profile requires a complete definition")
        if str(profile_definition.get("profile_id", "")).upper() != profile:
            raise ValueError("field-driven profile definition identifier mismatch")
        controls = profile_definition.get("controls")
        if not isinstance(controls, Mapping):
            raise ValueError("field-driven profile controls must be a mapping")
        observed_keys = frozenset(map(str, controls))
        if observed_keys != _STAR_CONTROL_KEYS:
            missing = sorted(_STAR_CONTROL_KEYS - observed_keys)
            extra = sorted(observed_keys - _STAR_CONTROL_KEYS)
            raise ValueError("field-driven profile controls must be complete:" f"missing={missing}:extra={extra}")
        if not isinstance(controls["allow_short_record_padding"], bool):
            raise ValueError("allow_short_record_padding must be boolean")
        numeric_keys = {
            "target_fs_hz",
            "window_seconds",
            "hop_seconds",
            "learning_rate",
            "weight_decay",
        }
        numeric_types = (int, float, np.integer, np.floating)
        if any(isinstance(controls[key], bool) or not isinstance(controls[key], numeric_types) for key in numeric_keys):
            raise ValueError("field-driven numeric controls must be numbers")
        if any(
                isinstance(controls[key], bool) or not isinstance(controls[key], (int, np.integer))
                for key in ("batch_size", "fixed_epochs")):
            raise ValueError("batch_size and fixed_epochs must be integers")
        retained_fraction = controls["historical_retained_fraction"]
        if retained_fraction is not None and (isinstance(retained_fraction, bool)
                                              or not isinstance(retained_fraction, numeric_types)):
            raise ValueError("historical_retained_fraction must be numeric or null")
        window_cap = controls["max_windows_per_file"]
        if window_cap is not None and (isinstance(window_cap, bool) or not isinstance(window_cap, (int, np.integer))):
            raise ValueError("max_windows_per_file must be an integer or null")
        observed_sha256 = _stable_mapping_sha256(controls)
        if profile_definition_sha256 != observed_sha256:
            raise ValueError("field-driven profile definition SHA mismatch:"
                             f"expected={profile_definition_sha256}:observed={observed_sha256}")
        normalization = {
            "per_window_all_eight": "per_window_all_eight_channels",
            "ppg_window_imu_outer_train_fold": "outer_train_axes6_fold_robust",
        }.get(str(controls["normalization"]))
        if normalization is None:
            raise ValueError("unsupported field-driven normalization")
        return LegacyBridgeProfile(
            profile_id=profile,
            preprocessing="field_driven",
            target_fs_hz=float(controls["target_fs_hz"]),
            window_seconds=float(controls["window_seconds"]),
            hop_seconds=float(controls["hop_seconds"]),
            historical_retained_fraction=(None if controls["historical_retained_fraction"] is None else float(
                controls["historical_retained_fraction"])),
            max_windows_per_file=(None if controls["max_windows_per_file"] is None else int(
                controls["max_windows_per_file"])),
            imu_normalization=normalization,
            sampler=str(controls["sampler"]),
            class_weighting=str(controls["class_weighting"]),
            optimizer=str(controls["optimizer"]),
            batch_size=int(controls["batch_size"]),
            expected_aggregation_rule=str(controls["training_metric_aggregation_rule"]),
            fixed_epochs=int(controls["fixed_epochs"]),
            learning_rate=float(controls["learning_rate"]),
            weight_decay=float(controls["weight_decay"]),
            protocol_design=protocol_design,
            ppg_preprocessing=str(controls["ppg_preprocessing"]),
            imu_preprocessing=str(controls["imu_preprocessing"]),
            allow_short_record_padding=bool(controls["allow_short_record_padding"]),
            primary_report_aggregation_view=str(controls["primary_report_aggregation_view"]),
            declared_controls=dict(controls),
            profile_definition_sha256=observed_sha256,
        )
    if protocol_design != "cumulative_chain_v1":
        raise ValueError(f"unknown bridge protocol design: {protocol_design!r}")
    if profile_definition is not None or profile_definition_sha256 is not None:
        raise ValueError("cumulative bridge does not accept inline profile controls")
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
        imu_normalization=("per_window_all_eight_channels" if level <= 3 else "outer_train_axes6_fold_robust"),
        sampler=("exhaustive_shuffle_without_replacement"
                 if level <= 4 else "uniform_replacement" if level == 5 else "balance_line_weighted_v2"),
        class_weighting=("outer_train_window_inverse_frequency" if level <= 5 else "outer_train_inverse_frequency"),
        optimizer=("adam" if level == 7 else "adamw"),
        batch_size=(64 if level == 7 else 32),
        expected_aggregation_rule=(LINE_B_EQUAL_ROLE_FAMILIES if level >= 6 else LINE_A_EQUAL_FILES),
    )


def bridge_profile_from_case(
    case_id: str,
    profiles: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    *,
    protocol_design: str = "cumulative_chain_v1",
) -> LegacyBridgeProfile:
    """Bind a study case to its reviewed display profile exactly once."""

    matches = [row for row in profiles if str(row.get("catalog_case_id")) == str(case_id)]
    if len(matches) != 1:
        raise ValueError(f"bridge case must map to exactly one profile: {case_id}")
    definition = matches[0]
    return resolve_legacy_bridge_profile(
        str(definition["profile_id"]),
        protocol_design=protocol_design,
        profile_definition=(definition if protocol_design in FIELD_DRIVEN_PROTOCOL_DESIGNS else None),
        profile_definition_sha256=(str(definition.get("controls_sha256"))
                                   if protocol_design in FIELD_DRIVEN_PROTOCOL_DESIGNS else None),
    )


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
        if not profile.resolved_allow_short_record_padding:
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
    valid = np.ones(values.shape[0], dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != (values.shape[0], ):
        raise ValueError("bridge validity mask must align with samples")
    starts = _window_starts(values.shape[0], fs_hz=fs_hz, profile=profile)
    window = int(round(profile.window_seconds * float(fs_hz)))
    rows: list[np.ndarray] = []
    retained_starts: list[int] = []
    dropped = 0
    for start in starts:
        stop = min(int(start) + window, values.shape[0])
        if not np.all(valid[int(start):stop]):
            dropped += 1
            continue
        segment = values[int(start):stop]
        if profile.imu_normalization == "per_window_all_eight_channels":
            segment = _robust_scale_all_channels(segment)
        else:
            normalized = segment.astype(np.float32, copy=True)
            normalized[:, :2] = _robust_scale_all_channels(segment[:, :2])
            segment = normalized
        if segment.shape[0] < window:
            if not profile.resolved_allow_short_record_padding:
                raise ValueError("this bridge profile forbids right-zero padding of a short record")
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
            "schema_version":
            "ppg_frailty.legacy_bridge_raw_windows.v1",
            "profile":
            profile.to_dict(),
            "normalization":
            ("per_window_all_8ch_median_iqr_over_1p349_sd_fallback_clip_-8_8" if profile.imu_normalization
             == "per_window_all_eight_channels" else "per_window_red_ir_only_then_outer_train_axes6_fold_robust"),
            "short_record_policy":
            (("historical_L0_scale_available_rows_then_right_zero_pad"
              if profile.protocol_design == "cumulative_chain_v1" else "scale_available_rows_then_right_zero_pad")
             if profile.resolved_allow_short_record_padding else "complete_windows_only_no_padding"),
            **dict(provenance),
        },
    )


def build_legacy_bridge_raw_windows(
    record: Mapping[str, Any],
    profile: LegacyBridgeProfile,
    *,
    calibrated_views: Any | None = None,
) -> RawWindows:
    """Preprocess one raw record using only the profile's declared fields."""

    if not profile.builds_windows_from_raw_record:
        raise ValueError("raw-record builder requires legacy PPG preprocessing")
    source_fs = float(record["fs_hz"])
    if source_fs != 400.0:
        raise ValueError("legacy bridge source grid must be the audited 400 Hz grid")
    ppg = np.asarray(record["ppg"], dtype=np.float64)
    acc = np.asarray(record["acc"], dtype=np.float64)
    gyro = np.asarray(record["gyro"], dtype=np.float64)
    if (ppg.ndim != 2 or ppg.shape[1] != 2 or acc.shape != (ppg.shape[0], 3) or gyro.shape != (ppg.shape[0], 3)
            or not np.isfinite(np.column_stack((ppg, acc, gyro))).all()):
        raise ValueError("legacy bridge raw channels are not aligned finite 8-channel data")
    ppg_filtered = _filter_sos(
        signal.detrend(ppg, axis=0, type="linear"),
        fs_hz=source_fs,
        cutoff=(0.2, 8.0),
        kind="bandpass",
    )
    if profile.resolved_imu_preprocessing == "legacy_filtered_axes":
        imu_acc = _filter_sos(
            acc,
            fs_hz=source_fs,
            cutoff=20.0,
            kind="lowpass",
        )
        imu_gyro = _filter_sos(
            gyro,
            fs_hz=source_fs,
            cutoff=40.0,
            kind="lowpass",
        )
        valid_mask: np.ndarray | None = None
        imu_provenance = "legacy_axes_no_SI_no_B_calibration_no_EKF"
    else:
        if calibrated_views is None:
            raise ValueError("calibrated EKF IMU profile requires canonical views")
        calibrated_views.validate()
        imu_acc = np.asarray(
            calibrated_views.imu_processed["dynamic_acc_mps2"],
            dtype=np.float64,
        )
        imu_gyro = np.asarray(
            calibrated_views.imu_processed["gyro_rads"],
            dtype=np.float64,
        )
        valid_mask = np.asarray(
            calibrated_views.imu_processed.get("imu_valid_mask", np.ones(ppg.shape[0], dtype=bool)),
            dtype=bool,
        )
        if imu_acc.shape != acc.shape or imu_gyro.shape != gyro.shape:
            raise ValueError("calibrated IMU views must align with the raw record")
        imu_provenance = "canonical_SI_B_calibration_roll_pitch_EKF_A_dyn"
    # Historical cache materialization rounded the filtered 8-channel matrix
    # to float32 before resampling, then rounded the resampled output again.
    matrix = np.column_stack((ppg_filtered, imu_acc, imu_gyro)).astype(np.float32)
    target_fs = float(profile.target_fs_hz)
    if target_fs != source_fs:
        ratio = Fraction(str(target_fs / source_fs)).limit_denominator(10_000)
        matrix = signal.resample_poly(
            matrix,
            up=ratio.numerator,
            down=ratio.denominator,
            axis=0,
        ).astype(np.float32)
        if valid_mask is not None:
            source_positions = np.minimum(
                np.rint(np.arange(matrix.shape[0], dtype=np.float64) * source_fs / target_fs).astype(np.int64),
                valid_mask.size - 1,
            )
            valid_mask = valid_mask[source_positions]
    return _raw_windows_from_matrix(
        matrix,
        fs_hz=target_fs,
        profile=profile,
        valid_mask=valid_mask,
        provenance={
            "preprocessing":
            ("PPG_detrend_order3_zero_phase_0p2_8Hz;"
             "ACC_order3_LPF20Hz;GYRO_order3_LPF40Hz;"
             "no_SI_no_B_calibration_no_EKF_no_gravity_removal" if profile.protocol_design == "cumulative_chain_v1" else
             ("PPG_detrend_order3_zero_phase_0p2_8Hz;"
              f"IMU={imu_provenance}")),
            "resampling": ("not_applied_native_400Hz" if target_fs == source_fs else
                           ("historical_scipy_resample_poly_default_constant_pad_400_to_64Hz" if profile.protocol_design
                            == "cumulative_chain_v1" else ("scipy_resample_poly_default_constant_pad_"
                                                           f"{source_fs:g}_to_{target_fs:g}Hz"))),
            "channel_schema":
            profile.channel_schema,
        },
    )


def build_v2_window_scaled_bridge_raw_windows(
    views: Any,
    profile: LegacyBridgeProfile,
) -> RawWindows:
    """Build canonical-view windows with legacy per-window 8ch scaling."""

    if not profile.uses_canonical_all_channel_window_scaling:
        raise ValueError("canonical all-channel bridge scaling was not requested")
    views.validate()
    dynamic = np.asarray(views.imu_processed["dynamic_acc_mps2"], dtype=np.float64)
    gyro = np.asarray(views.imu_processed["gyro_rads"], dtype=np.float64)
    matrix = np.column_stack((np.asarray(views.x_filter), dynamic, gyro))
    valid = np.asarray(
        views.imu_processed.get("imu_valid_mask", np.ones(matrix.shape[0], dtype=bool)),
        dtype=bool,
    )
    return _raw_windows_from_matrix(
        matrix,
        fs_hz=400.0,
        profile=profile,
        valid_mask=valid,
        provenance={
            "preprocessing":
            "canonical_V2_SI_B_calibration_roll_pitch_EKF",
            "channel_schema":
            profile.channel_schema,
            "imu_fold_transform": ("not_applied_L3_per_window_ablation" if profile.protocol_design
                                   == "cumulative_chain_v1" else "not_applied_per_window_all_eight_profile"),
        },
    )


__all__ = [
    "LegacyBridgeProfile",
    "bridge_profile_from_case",
    "build_legacy_bridge_raw_windows",
    "build_v2_window_scaled_bridge_raw_windows",
    "resolve_legacy_bridge_profile",
]
