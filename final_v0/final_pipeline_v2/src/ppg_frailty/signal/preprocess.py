"""V2 400-Hz PPG quality checks and zero-phase preprocessing.

M3 数学迁移来源（复制并按 V2 合同审计，绝不 import）：
`final_v0/M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core/ppg.py`,
SHA-256 `d68a807893b02b590341199822425e83ec3fcba5e44e2f6597b944e21001abe5`；
quality.py SHA-256
`b246d0bfae4afdf6275b92d9579ded7fde063f3d0d33b49c77d6af43d5386bdc`。

M3 mathematical source was copied conceptually and audited for the V2 contract;
there is deliberately no runtime import from M3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy import signal

from .imu import ImuPreprocessResult, preprocess_imu
from .motion_imu import (
    MotionImuCalibration,
    RollPitchEkfConfig,
    preprocess_motion_imu_calibrated_ekf,
    preprocess_motion_imu_lpf_ablation,
)
from .resample import validate_dl_resampling_config
from .views import CANONICAL_FS_HZ, CanonicalSignalViews


M3_PPG_SOURCE_SHA256 = (
    "d68a807893b02b590341199822425e83ec3fcba5e44e2f6597b944e21001abe5"
)
M3_QUALITY_SOURCE_SHA256 = (
    "b246d0bfae4afdf6275b92d9579ded7fde063f3d0d33b49c77d6af43d5386bdc"
)
REFERENCE_PPG_FILTER_PROFILE_ID = "butterworth_0p2_8hz_reference"
ABLATION_PPG_FILTER_PROFILE_ID = "butterworth_0p5_5hz_ablation"
CANONICAL_AXES6_NORMALIZATION_ID = (
    "outer_training_participant_only_robust_scaler_axes6"
)


@dataclass(frozen=True)
class PpgFilterProfile:
    """Registered filter identity; registration does not execute an ablation."""

    profile_id: str
    low_hz: float
    high_hz: float
    order: int
    registry_role: str

    def validate(self) -> None:
        if self.profile_id not in {
            REFERENCE_PPG_FILTER_PROFILE_ID,
            ABLATION_PPG_FILTER_PROFILE_ID,
        }:
            raise ValueError("unregistered PPG filter profile")
        if self.order != 3 or not 0.0 < self.low_hz < self.high_hz < 200.0:
            raise ValueError("invalid registered PPG filter profile")
        if self.registry_role not in {"reference", "ablation"}:
            raise ValueError("filter registry_role must be reference or ablation")


_PPG_FILTER_PROFILES = {
    REFERENCE_PPG_FILTER_PROFILE_ID: PpgFilterProfile(
        REFERENCE_PPG_FILTER_PROFILE_ID, 0.2, 8.0, 3, "reference"
    ),
    ABLATION_PPG_FILTER_PROFILE_ID: PpgFilterProfile(
        ABLATION_PPG_FILTER_PROFILE_ID, 0.5, 5.0, 3, "ablation"
    ),
}


def get_ppg_filter_profile(profile_id: str) -> PpgFilterProfile:
    """Return one named profile; no implicit nearest-band matching is allowed."""

    try:
        profile = _PPG_FILTER_PROFILES[str(profile_id).strip().lower()]
    except KeyError as exc:
        raise KeyError(f"unknown PPG filter profile: {profile_id}") from exc
    profile.validate()
    return profile


@dataclass(frozen=True)
class InputQC:
    """滤波前输入检查结果 / Result of pre-filter input inspection."""

    repaired: np.ndarray
    source_valid_mask: np.ndarray
    repair_mask: np.ndarray
    status: str
    reasons: tuple[str, ...]
    metrics: dict[str, Any]


def validate_timestamp_grid(
    timestamps_s: np.ndarray | None,
    n_samples: int,
    *,
    fs_hz: float = CANONICAL_FS_HZ,
    jitter_tolerance_fraction: float = 0.05,
) -> None:
    """验证显式同步时间轴 / Validate an explicit synchronized time grid."""

    if timestamps_s is None:
        return
    time = np.asarray(timestamps_s, dtype=np.float64).ravel()
    if time.size != n_samples or not np.isfinite(time).all():
        raise ValueError("timestamps must be finite and match sample count")
    if time.size < 2:
        return
    intervals = np.diff(time)
    expected = 1.0 / float(fs_hz)
    if np.any(intervals <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    relative = np.abs(intervals - expected) / expected
    if float(np.percentile(relative, 99.0)) > jitter_tolerance_fraction:
        raise ValueError("timestamp jitter exceeds the frozen five-percent tolerance")


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """返回布尔真值的右开连续区间 / Return right-open runs of true values."""

    values = np.asarray(mask, dtype=bool).ravel()
    padded = np.concatenate(([False], values, [False]))
    changes = np.diff(padded.astype(np.int8))
    return list(
        zip(
            np.flatnonzero(changes == 1).astype(int).tolist(),
            np.flatnonzero(changes == -1).astype(int).tolist(),
        )
    )


def _longest_constant_run(values: np.ndarray) -> int:
    """计算严格相等平线长度 / Compute the longest exactly constant run."""

    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size < 2:
        return int(x.size)
    return max((stop - start + 1 for start, stop in _true_runs(np.diff(x) == 0.0)), default=1)


def inspect_and_repair(
    values: np.ndarray,
    *,
    fs_hz: float = CANONICAL_FS_HZ,
    max_gap_sec: float = 0.25,
    flatline_sec: float = 1.0,
) -> InputQC:
    """仅修复有限内部短缺口 / Repair only bounded internal short gaps.

    边界缺口、长缺口、过多非有限值和平线均 fail closed。The function fails
    closed for boundary/long gaps, excessive non-finite values, and flatlines.
    """

    source = np.asarray(values, dtype=np.float64)
    if source.ndim == 1:
        source = source[:, None]
    if source.ndim != 2 or source.shape[0] == 0:
        raise ValueError("signal must be a non-empty samples-by-channels matrix")
    if float(fs_hz) != CANONICAL_FS_HZ:
        raise ValueError("V2 canonical input QC requires exactly 400 Hz")

    valid = np.isfinite(source)
    repaired = source.copy()
    repair_mask = np.zeros_like(valid)
    reasons: list[str] = []
    metrics: dict[str, Any] = {"n_samples": int(source.shape[0]), "channels": {}}
    max_gap = int(round(max_gap_sec * fs_hz))
    fatal = False
    for column in range(source.shape[1]):
        finite = valid[:, column]
        fraction = float(np.count_nonzero(~finite) / finite.size)
        gap_runs = _true_runs(~finite)
        info: dict[str, Any] = {
            "nonfinite_fraction": fraction,
            "longest_nonfinite_gap_samples": max(
                (stop - start for start, stop in gap_runs),
                default=0,
            ),
        }
        # English: Total missing fraction is not governed by a hidden threshold.
        # Every gap obeys the explicit max length; full source coverage is retained
        # for downstream SQI, where the registered 0.80 endpoint gate applies.
        # 中文：不使用隐藏的总缺失比例阈值。每个 gap 受显式最大长度约束；完整
        # source coverage 传入下游 SQI，并由注册的 0.80 coverage gate 判定。
        if not finite.any():
            fatal = True
            reasons.append(f"channel_{column}:all_nonfinite")
        for start, stop in gap_runs:
            length = stop - start
            internal = start > 0 and stop < source.shape[0]
            if internal and length <= max_gap and np.isfinite(
                [repaired[start - 1, column], repaired[stop, column]]
            ).all():
                repaired[start:stop, column] = np.linspace(
                    repaired[start - 1, column],
                    repaired[stop, column],
                    length + 2,
                    dtype=np.float64,
                )[1:-1]
                repair_mask[start:stop, column] = True
                reasons.append(f"channel_{column}:short_gap_interpolated:{length}")
            else:
                fatal = True
                reasons.append(
                    f"channel_{column}:{'boundary_gap' if not internal else 'long_gap'}:{length}"
                )
        if np.isfinite(repaired[:, column]).all():
            longest = _longest_constant_run(repaired[:, column])
            info["longest_constant_run"] = int(longest)
            if longest >= int(round(flatline_sec * fs_hz)):
                fatal = True
                reasons.append(f"channel_{column}:flatline:{longest}")
            minimum = float(np.min(repaired[:, column]))
            maximum = float(np.max(repaired[:, column]))
            info["min_occupancy"] = float(np.mean(repaired[:, column] == minimum))
            info["max_occupancy"] = float(np.mean(repaired[:, column] == maximum))
        metrics["channels"][str(column)] = info
    status = "failed" if fatal or not np.isfinite(repaired).all() else "success"
    return InputQC(repaired, valid, repair_mask, status, tuple(reasons), metrics)


def design_ppg_sos(
    low_hz: float = 0.2,
    high_hz: float = 8.0,
    *,
    fs_hz: float = CANONICAL_FS_HZ,
    order: int = 3,
) -> np.ndarray:
    """设计冻结三阶 Butterworth SOS / Design the frozen third-order SOS."""

    if float(fs_hz) != CANONICAL_FS_HZ:
        raise ValueError("PPG filter design is frozen to 400 Hz")
    if not 0.0 < low_hz < high_hz < fs_hz / 2.0:
        raise ValueError("band edges must lie strictly within Nyquist")
    return signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs_hz, output="sos")


def preprocess_ppg_pair(
    ppg: np.ndarray,
    *,
    fs_hz: float = CANONICAL_FS_HZ,
    timestamps_s: np.ndarray | None = None,
    max_gap_samples: int = 100,
    flatline_sec: float = 1.0,
    filter_low_hz: float = 0.2,
    filter_high_hz: float = 8.0,
    filter_order: int = 3,
) -> tuple[np.ndarray, np.ndarray, InputQC]:
    """生成保基线 native 和 direct x_filter / Build native and direct views."""

    source = np.asarray(ppg, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("PPG must be samples-by-[RED,IR]")
    validate_timestamp_grid(timestamps_s, source.shape[0], fs_hz=fs_hz)
    if max_gap_samples < 0:
        raise ValueError("max_gap_samples must be non-negative")
    qc = inspect_and_repair(
        source,
        fs_hz=fs_hz,
        max_gap_sec=float(max_gap_samples) / fs_hz,
        flatline_sec=flatline_sec,
    )
    if qc.status != "success":
        raise ValueError("PPG input failed closed QC: " + ";".join(qc.reasons))
    native = np.asarray(qc.repaired, dtype=np.float64)
    sos = design_ppg_sos(
        filter_low_hz,
        filter_high_hz,
        fs_hz=fs_hz,
        order=filter_order,
    )
    # 中文：离线科学分析固定 linear detrend + sosfiltfilt，短输入绝不切 causal。
    # English: Offline analysis is fixed to linear detrend + zero phase; no causal fallback.
    working = signal.detrend(native, axis=0, type="linear")
    try:
        filtered = signal.sosfiltfilt(sos, working, axis=0)
    except ValueError as exc:
        raise ValueError(f"zero_phase_filter_insufficient_length:{exc}") from exc
    return native, np.asarray(filtered, dtype=np.float64), qc


def roll_pitch_ekf_config_from_resolved(
    imu_config: Mapping[str, Any],
) -> RollPitchEkfConfig:
    """Build the calibrated roll-pitch EKF object from one resolved config."""

    if not isinstance(imu_config, Mapping):
        raise ValueError("resolved signal.imu must be a mapping")
    required = {
        "process_covariance_diagonal_per_second",
        "observation_covariance_diagonal_rad2",
        "initial_covariance_diagonal",
        "dynamic_observation_scale",
        "sensor_lowpass_acc_hz",
        "sensor_lowpass_gyro_hz",
        "sensor_filter_order",
        "calibration_start_s",
        "calibration_stop_s",
        "gravity_lowpass_hz",
        "gravity_filter_order",
    }
    missing = sorted(required - set(imu_config))
    if missing:
        raise ValueError(f"resolved signal.imu lacks EKF parameters: {missing}")
    result = RollPitchEkfConfig(
        process_covariance_diagonal_per_second=tuple(
            float(value)
            for value in imu_config["process_covariance_diagonal_per_second"]
        ),
        observation_covariance_diagonal_rad2=tuple(
            float(value)
            for value in imu_config["observation_covariance_diagonal_rad2"]
        ),
        initial_covariance_diagonal=tuple(
            float(value)
            for value in imu_config["initial_covariance_diagonal"]
        ),
        dynamic_observation_scale=float(imu_config["dynamic_observation_scale"]),
        accelerometer_lowpass_hz=float(imu_config["sensor_lowpass_acc_hz"]),
        gyroscope_lowpass_hz=float(imu_config["sensor_lowpass_gyro_hz"]),
        sensor_filter_order=int(imu_config["sensor_filter_order"]),
        calibration_start_s=float(imu_config["calibration_start_s"]),
        calibration_stop_s=float(imu_config["calibration_stop_s"]),
        gravity_lowpass_hz=float(imu_config["gravity_lowpass_hz"]),
        gravity_filter_order=int(imu_config["gravity_filter_order"]),
    )
    result.validate()
    return result


def build_signal_views(
    record: Mapping[str, Any] | Any,
    config: Mapping[str, Any],
) -> CanonicalSignalViews:
    """Build the canonical V2 400-Hz signal views.

    `record` 可提供 `ppg`，或分别提供 `red` 与 `ir`；IMU 必须提供 `acc`、`gyro`
    及明确单位。``record`` may provide ``ppg`` or separate ``red``/``ir`` arrays.
    """

    def required_field(name: str) -> Any:
        if isinstance(record, Mapping):
            if name not in record:
                raise ValueError(f"record missing required field: {name}")
            return record[name]
        if not hasattr(record, name):
            raise ValueError(f"record missing required field: {name}")
        return getattr(record, name)

    def optional_field(name: str, default: Any = None) -> Any:
        if isinstance(record, Mapping):
            return record.get(name, default)
        return getattr(record, name, default)

    if not isinstance(config, Mapping) or not isinstance(config.get("signal"), Mapping):
        raise ValueError("formal build_signal_views requires resolved config['signal']")
    signal_config = config["signal"]
    expected_signal_keys = {
        "internal_fs_hz", "channel_order", "ppg_native_unit",
        "accelerometer_input_unit", "gyroscope_input_unit", "ppg_filter",
        "analysis_view", "gap_repair", "imu", "dl_resampling", "normalization",
    }
    if set(signal_config) != expected_signal_keys:
        raise ValueError(
            "resolved signal key mismatch: "
            f"missing={sorted(expected_signal_keys-set(signal_config))}, "
            f"unknown={sorted(set(signal_config)-expected_signal_keys)}"
        )
    fs_hz = float(required_field("fs_hz"))
    configured_fs = float(signal_config.get("internal_fs_hz", float("nan")))
    if fs_hz != CANONICAL_FS_HZ or configured_fs != CANONICAL_FS_HZ:
        raise ValueError("record/config internal sampling grid must both equal 400 Hz")
    record_id = str(required_field("record_id"))
    if not record_id:
        raise ValueError("record_id must be non-empty")
    ppg = required_field("ppg")
    acc, gyro = required_field("acc"), required_field("gyro")
    timestamps = (
        record.get("timestamps_s")
        if isinstance(record, Mapping)
        else getattr(record, "timestamps_s", None)
    )
    ppg_filter = signal_config.get("ppg_filter")
    if not isinstance(ppg_filter, Mapping):
        raise ValueError("resolved signal.ppg_filter is required")
    expected_filter = {
        "family": "butterworth_sos",
        "order": 3,
        "phase": "zero_phase",
        "short_signal_policy": "reject",
        "notch_enabled": False,
    }
    if any(ppg_filter.get(key) != value for key, value in expected_filter.items()):
        raise ValueError("resolved PPG filter differs from the frozen V2 profiles")
    filter_pair = (float(ppg_filter.get("low_hz")), float(ppg_filter.get("high_hz")))
    if filter_pair not in {(0.2, 8.0), (0.5, 5.0)}:
        raise ValueError("resolved PPG filter is not a registered V2 comparison profile")
    if set(ppg_filter) != set(expected_filter) | {"low_hz", "high_hz"}:
        raise ValueError("resolved PPG filter contains missing or unknown keys")
    if signal_config.get("channel_order") != ["RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ"]:
        raise ValueError("resolved channel_order differs from the frozen eight-channel schema")
    if signal_config.get("ppg_native_unit") != "raw_counts":
        raise ValueError("formal PPG native unit must be raw_counts")
    expected_analysis = {
        "direct_source": (
            "x_filter_0p2_to_8hz"
            if filter_pair == (0.2, 8.0) else "x_filter_0p5_to_5hz"
        ),
        "non_identity_source": "aligned_x_ar",
        "non_identity_semantics": "rate_only",
        "additional_filter": "none",
    }
    analysis_view = signal_config.get("analysis_view")
    if not isinstance(analysis_view, Mapping) or dict(analysis_view) != expected_analysis:
        raise ValueError("resolved signal.analysis_view differs from the frozen V2 contract")
    gap_repair = signal_config.get("gap_repair")
    expected_gap_keys = {
        "method", "max_gap_samples", "edge_extrapolation", "all_missing_channel_action"
    }
    if not isinstance(gap_repair, Mapping) or set(gap_repair) != expected_gap_keys:
        raise ValueError("resolved signal.gap_repair key mismatch")
    if (
        gap_repair.get("method") != "linear_inside_only"
        or gap_repair.get("edge_extrapolation") is not False
        or gap_repair.get("all_missing_channel_action") != "reject_record"
        or not isinstance(gap_repair.get("max_gap_samples"), int)
        or int(gap_repair["max_gap_samples"]) < 0
    ):
        raise ValueError("resolved signal.gap_repair differs from the supported policy")
    quality_config = config.get("quality")
    if not isinstance(quality_config, Mapping):
        raise ValueError("formal build_signal_views requires resolved config['quality']")
    if int(quality_config.get("long_gap_max_samples", -1)) != int(gap_repair["max_gap_samples"]):
        raise ValueError("signal gap repair and quality long-gap thresholds must match")
    flatline_sec = float(quality_config.get("flatline_duration_s", float("nan")))
    if not np.isfinite(flatline_sec) or flatline_sec <= 0.0:
        raise ValueError("quality.flatline_duration_s must be explicit and positive")
    native, filtered, ppg_qc = preprocess_ppg_pair(
        ppg,
        fs_hz=fs_hz,
        timestamps_s=timestamps,
        max_gap_samples=int(gap_repair["max_gap_samples"]),
        flatline_sec=flatline_sec,
        filter_low_hz=float(ppg_filter["low_hz"]),
        filter_high_hz=float(ppg_filter["high_hz"]),
        filter_order=int(ppg_filter["order"]),
    )

    acc_unit = signal_config.get("accelerometer_input_unit")
    gyro_unit = signal_config.get("gyroscope_input_unit")
    imu_config = signal_config.get("imu")
    if not isinstance(acc_unit, str) or not isinstance(gyro_unit, str) or not isinstance(imu_config, Mapping):
        raise ValueError("resolved signal units and signal.imu profile are required")
    gravity_profile = str(imu_config.get("gravity_method"))
    calibrated_profiles = {
        "calibrated_roll_pitch_ekf",
        "profile_a_lowpass_0p3hz",
    }
    legacy_profiles = {"quaternion_error_state_ekf", "low_pass_0p3hz"}
    if gravity_profile not in calibrated_profiles | legacy_profiles:
        raise ValueError("formal IMU gravity method is not a registered V2 profile")
    common_imu_keys = {
        "gravity_method", "initialization", "comparison_method",
        "sensor_lowpass_acc_hz", "sensor_lowpass_gyro_hz", "gravity_lowpass_hz",
        "output_units", "required_axes", "failure_action",
    }
    if (
        float(imu_config.get("sensor_lowpass_acc_hz", float("nan"))) != 20.0
        or float(imu_config.get("sensor_lowpass_gyro_hz", float("nan"))) != 40.0
        or float(imu_config.get("gravity_lowpass_hz", float("nan"))) != 0.3
        or imu_config.get("failure_action") != "fail_closed"
        or imu_config.get("required_axes") != 6
        or imu_config.get("output_units")
        != {"acceleration": "m/s^2", "gyroscope": "rad/s", "jerk": "m/s^3"}
    ):
        raise ValueError("resolved signal.imu parameters differ from the V2 contract")
    if gravity_profile in calibrated_profiles:
        calibrated_keys = common_imu_keys | {
            "sensor_filter_order",
            "gravity_filter_order",
            "calibration_start_s",
            "calibration_stop_s",
            "process_covariance_diagonal_per_second",
            "observation_covariance_diagonal_rad2",
            "initial_covariance_diagonal",
            "dynamic_observation_scale",
        }
        if set(imu_config) != calibrated_keys:
            raise ValueError("calibrated signal.imu contains missing or unknown keys")
        if (
            imu_config.get("initialization")
            != "same_participant_static_calibration"
            or imu_config.get("comparison_method") != "profile_a_lowpass_0p3hz"
        ):
            raise ValueError("calibrated EKF initialization/comparator identity drift")
    else:
        if set(imu_config) != common_imu_keys:
            raise ValueError("legacy signal.imu contains missing or unknown keys")
        if (
            imu_config.get("initialization") != "online_no_precalibration"
            or imu_config.get("comparison_method") != "lowpass_0p3hz"
        ):
            raise ValueError("legacy IMU profile declaration drift")
    dl_resampling = validate_dl_resampling_config(signal_config.get("dl_resampling"))
    workflow_normalization = {
        "raw_ppg": "per_window_median_iqr_over_1p349_sd_finite",
        "raw_imu": CANONICAL_AXES6_NORMALIZATION_ID,
        "iqr_fallback": "standard_deviation_then_finite_one",
        "clip_after_scale": [-8.0, 8.0],
    }
    observed_normalization = signal_config.get("normalization")
    normalized_observed = (
        tuple(
            sorted(
                (
                    key,
                    tuple(value) if isinstance(value, list) else value,
                )
                for key, value in observed_normalization.items()
            )
        )
        if isinstance(observed_normalization, Mapping)
        else ()
    )
    expected_normalization = tuple(
        sorted(
            (
                key,
                tuple(value) if isinstance(value, list) else value,
            )
            for key, value in workflow_normalization.items()
        )
    )
    if normalized_observed != expected_normalization:
        raise ValueError(
            "frailty signal.normalization requires the canonical axes6 profile; "
            "motion9 belongs to the separate motion augmentation profile"
        )
    for name, expected in (("acc_unit", acc_unit), ("gyro_unit", gyro_unit)):
        observed = (
            record.get(name)
            if isinstance(record, Mapping)
            else getattr(record, name, None)
        )
        if observed is not None and str(observed) != expected:
            raise ValueError(f"record {name} conflicts with resolved signal profile")
    if gravity_profile in calibrated_profiles:
        calibration = optional_field("imu_calibration")
        participant_id = str(required_field("participant_id"))
        if not isinstance(calibration, MotionImuCalibration):
            raise ValueError(
                "calibrated EKF requires an explicit same-participant "
                "MotionImuCalibration; no fallback is permitted"
            )
        ekf_config = roll_pitch_ekf_config_from_resolved(imu_config)
        processor = (
            preprocess_motion_imu_calibrated_ekf
            if gravity_profile == "calibrated_roll_pitch_ekf"
            else preprocess_motion_imu_lpf_ablation
        )
        motion_imu = processor(
            acc,
            gyro,
            fs_hz=fs_hz,
            acceleration_unit=acc_unit,
            gyroscope_unit=gyro_unit,
            participant_id=participant_id,
            calibration=calibration,
            config=ekf_config,
        )
        dynamic = np.asarray(motion_imu.values[:, :3], dtype=np.float64)
        gyro_si = np.asarray(motion_imu.values[:, 3:6], dtype=np.float64)
        jerk_axes = np.diff(dynamic, axis=0, prepend=dynamic[:1]) * fs_hz
        imu = ImuPreprocessResult(
            processed={
                "acc_mps2": dynamic + motion_imu.gravity_mps2,
                "gravity_mps2": motion_imu.gravity_mps2,
                "dynamic_acc_mps2": dynamic,
                "gyro_rads": gyro_si,
                "dynamic_magnitude": motion_imu.values[:, 6],
                "gyro_magnitude": motion_imu.values[:, 7],
                "jerk_mps3": jerk_axes,
                "jerk_magnitude": motion_imu.values[:, 8],
                "imu_valid_mask": motion_imu.valid_mask,
            },
            status="success",
            reasons=(),
            gravity_method=motion_imu.profile_id,
            diagnostics=motion_imu.diagnostics,
            valid_mask=motion_imu.valid_mask,
        )
    else:
        imu = preprocess_imu(
            acc,
            gyro,
            fs_hz=fs_hz,
            acc_unit=acc_unit,
            gyro_unit=gyro_unit,
            gravity_method=(
                "no_precalibration_ekf"
                if gravity_profile == "quaternion_error_state_ekf"
                else "lpf_0p3"
            ),
            timestamps_s=timestamps,
        )
    if imu.status in {"failed", "no_estimate", "initialization_pending"}:
        raise ValueError("IMU preprocessing failed closed: " + ";".join(imu.reasons))
    if native.shape[0] != imu.processed["acc_mps2"].shape[0]:
        raise ValueError("PPG and IMU must be sample-synchronous")
    metadata = {
        "fs_hz": CANONICAL_FS_HZ,
        "record_id": record_id,
        "ppg_channel_schema": ("RED", "IR"),
        "ppg_unit": str(signal_config.get("ppg_native_unit")),
        "gravity_method": imu.gravity_method,
        "rate_only": False,
        "q_morph_state": "available",
        "source_hashes": {
            "m3_ppg_source_sha256": M3_PPG_SOURCE_SHA256,
            "m3_quality_source_sha256": M3_QUALITY_SOURCE_SHA256,
        },
        "qc_reasons": ppg_qc.reasons + imu.reasons,
        "ppg_qc_metrics": ppg_qc.metrics,
        "imu_status": imu.status,
        "imu_diagnostics": imu.diagnostics,
        "imu_valid_fraction": float(np.mean(imu.valid_mask)),
        "dl_resampling": dl_resampling,
        "canonical_feature_grid_hz": CANONICAL_FS_HZ,
    }
    views = CanonicalSignalViews(
        x_native=native,
        x_filter=filtered,
        # 中文：V2 direct rate 输入按冻结合同明确等于 x_filter。
        # English: ADR-003 freezes the direct rate view to x_filter.
        x_analysis_rate=filtered.copy(),
        imu_processed=imu.processed,
        metadata=metadata,
        source_valid_mask=ppg_qc.source_valid_mask,
        repair_mask=ppg_qc.repair_mask,
    )
    views.validate()
    return views
