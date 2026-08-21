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
        if (
            isinstance(self.order, bool)
            or not isinstance(self.order, (int, np.integer))
            or not 1 <= int(self.order) <= 20
            or not np.isfinite([self.low_hz, self.high_hz]).all()
            or not 0.0 < self.low_hz < self.high_hz < CANONICAL_FS_HZ / 2.0
        ):
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
    """设计运行时配置的 Butterworth SOS / Design runtime-configured SOS."""

    if float(fs_hz) != CANONICAL_FS_HZ:
        raise ValueError("PPG filter design requires the canonical 400 Hz grid")
    if (
        not np.isfinite([low_hz, high_hz]).all()
        or not 0.0 < float(low_hz) < float(high_hz) < fs_hz / 2.0
    ):
        raise ValueError("band edges must be finite and lie strictly within Nyquist")
    if (
        isinstance(order, bool)
        or not isinstance(order, (int, np.integer))
        or not 1 <= int(order) <= 20
    ):
        raise ValueError("PPG filter order must be an integer in [1,20]")
    return signal.butter(
        int(order),
        [float(low_hz), float(high_hz)],
        btype="bandpass",
        fs=fs_hz,
        output="sos",
    )


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
    defaults = RollPitchEkfConfig()
    result = RollPitchEkfConfig(
        process_covariance_diagonal_per_second=tuple(
            float(value)
            for value in imu_config.get(
                "process_covariance_diagonal_per_second",
                defaults.process_covariance_diagonal_per_second,
            )
        ),
        observation_covariance_diagonal_rad2=tuple(
            float(value)
            for value in imu_config.get(
                "observation_covariance_diagonal_rad2",
                defaults.observation_covariance_diagonal_rad2,
            )
        ),
        initial_covariance_diagonal=tuple(
            float(value)
            for value in imu_config.get(
                "initial_covariance_diagonal",
                defaults.initial_covariance_diagonal,
            )
        ),
        dynamic_observation_scale=float(
            imu_config.get("dynamic_observation_scale", defaults.dynamic_observation_scale)
        ),
        gravity_mps2=float(
            imu_config.get("gravity_mps2", defaults.gravity_mps2)
        ),
        accelerometer_lowpass_hz=float(
            imu_config.get("sensor_lowpass_acc_hz", defaults.accelerometer_lowpass_hz)
        ),
        gyroscope_lowpass_hz=float(
            imu_config.get("sensor_lowpass_gyro_hz", defaults.gyroscope_lowpass_hz)
        ),
        sensor_filter_order=imu_config.get(
            "sensor_filter_order", defaults.sensor_filter_order
        ),
        calibration_start_s=float(
            imu_config.get("calibration_start_s", defaults.calibration_start_s)
        ),
        calibration_stop_s=float(
            imu_config.get("calibration_stop_s", defaults.calibration_stop_s)
        ),
        gravity_lowpass_hz=float(
            imu_config.get("gravity_lowpass_hz", defaults.gravity_lowpass_hz)
        ),
        gravity_filter_order=imu_config.get(
            "gravity_filter_order", defaults.gravity_filter_order
        ),
    )
    result.validate()
    return result


def canonical_ppg_direct_source(low_hz: float, high_hz: float) -> str:
    """Derive the analysis-view identity from the executable filter band."""

    def token(value: float) -> str:
        return format(float(value), ".15g").replace("-", "m").replace(".", "p")

    return f"x_filter_{token(low_hz)}_to_{token(high_hz)}hz"


def _canonical_acceleration_unit(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("signal.accelerometer_input_unit must be a supported string")
    normalized = value.strip().lower().replace("²", "2")
    aliases = {"g": "g", "mg": "mg", "m/s2": "m/s2", "m/s^2": "m/s2"}
    if normalized not in aliases:
        raise ValueError(f"unsupported signal.accelerometer_input_unit: {value}")
    return aliases[normalized]


def _canonical_gyroscope_unit(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("signal.gyroscope_input_unit must be a supported string")
    normalized = value.strip().lower()
    if normalized not in {"deg/s", "rad/s"}:
        raise ValueError(f"unsupported signal.gyroscope_input_unit: {value}")
    return normalized


def _finite_number(value: Any, *, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite numeric")
    return float(value)


def _positive_vector(
    value: Any,
    *,
    name: str,
    length: int,
) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    result = [_finite_number(item, name=name) for item in value]
    if any(item <= 0.0 for item in result):
        raise ValueError(f"{name} values must be positive")
    return result


def _materialize_imu_profile(
    value: Any,
    *,
    fs_hz: float,
) -> dict[str, Any]:
    """Resolve one flat legacy input into a profile-specific effective mapping."""

    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("signal.imu must be a mapping")
    declared = dict(value)
    structural_keys = {
        "gravity_method",
        "initialization",
        "comparison_method",
        "output_units",
        "required_axes",
        "failure_action",
    }
    numeric_keys = (
        "sensor_lowpass_acc_hz",
        "sensor_lowpass_gyro_hz",
        "sensor_filter_order",
        "gravity_lowpass_hz",
        "gravity_filter_order",
        "calibration_start_s",
        "calibration_stop_s",
        "process_covariance_diagonal_per_second",
        "observation_covariance_diagonal_rad2",
        "initial_covariance_diagonal",
        "dynamic_observation_scale",
        "gravity_mps2",
    )
    unknown = sorted(set(declared) - structural_keys - set(numeric_keys))
    if unknown:
        raise ValueError(f"signal.imu contains unknown fields: {unknown}")

    method = str(declared.get("gravity_method", "calibrated_roll_pitch_ekf"))
    calibrated_methods = {
        "calibrated_roll_pitch_ekf",
        "profile_a_lowpass_0p3hz",
    }
    legacy_methods = {"quaternion_error_state_ekf", "low_pass_0p3hz"}
    if method not in calibrated_methods | legacy_methods:
        raise ValueError("signal.imu.gravity_method is not a registered V2 profile")

    derived_initialization = (
        "same_participant_static_calibration"
        if method in calibrated_methods
        else "online_no_precalibration"
    )
    derived_comparison = (
        "profile_a_lowpass_0p3hz"
        if method in calibrated_methods
        else "lowpass_0p3hz"
    )
    if (
        "initialization" in declared
        and declared["initialization"] != derived_initialization
    ):
        raise ValueError(
            "signal.imu.initialization disagrees with the selected gravity_method"
        )
    if (
        "comparison_method" in declared
        and declared["comparison_method"] != derived_comparison
    ):
        raise ValueError(
            "signal.imu.comparison_method disagrees with the selected gravity_method"
        )
    output_units = {
        "acceleration": "m/s^2",
        "gyroscope": "rad/s",
        "jerk": "m/s^3",
    }
    if "output_units" in declared and declared["output_units"] != output_units:
        raise ValueError("signal.imu.output_units is a derived SI-unit contract")
    if "required_axes" in declared and declared["required_axes"] != 6:
        raise ValueError("signal.imu.required_axes is the derived axes6 contract")
    if "failure_action" in declared and declared["failure_action"] != "fail_closed":
        raise ValueError("signal.imu.failure_action must be fail_closed")

    defaults = RollPitchEkfConfig()
    numeric_defaults: dict[str, Any] = {
        "sensor_lowpass_acc_hz": defaults.accelerometer_lowpass_hz,
        "sensor_lowpass_gyro_hz": defaults.gyroscope_lowpass_hz,
        "sensor_filter_order": defaults.sensor_filter_order,
        "gravity_lowpass_hz": defaults.gravity_lowpass_hz,
        "gravity_filter_order": (
            2 if method in legacy_methods else defaults.gravity_filter_order
        ),
        "calibration_start_s": defaults.calibration_start_s,
        "calibration_stop_s": defaults.calibration_stop_s,
        "process_covariance_diagonal_per_second": list(
            defaults.process_covariance_diagonal_per_second
        ),
        "observation_covariance_diagonal_rad2": list(
            defaults.observation_covariance_diagonal_rad2
        ),
        "initial_covariance_diagonal": list(defaults.initial_covariance_diagonal),
        "dynamic_observation_scale": defaults.dynamic_observation_scale,
        "gravity_mps2": defaults.gravity_mps2,
    }
    active_by_method = {
        "calibrated_roll_pitch_ekf": {
            "sensor_lowpass_acc_hz",
            "sensor_lowpass_gyro_hz",
            "sensor_filter_order",
            "calibration_start_s",
            "calibration_stop_s",
            "process_covariance_diagonal_per_second",
            "observation_covariance_diagonal_rad2",
            "initial_covariance_diagonal",
            "dynamic_observation_scale",
            "gravity_mps2",
        },
        "profile_a_lowpass_0p3hz": {
            "sensor_lowpass_acc_hz",
            "sensor_lowpass_gyro_hz",
            "sensor_filter_order",
            "gravity_lowpass_hz",
            "gravity_filter_order",
            "calibration_start_s",
            "calibration_stop_s",
            "gravity_mps2",
        },
        "quaternion_error_state_ekf": {
            "sensor_lowpass_acc_hz",
            "sensor_lowpass_gyro_hz",
            "sensor_filter_order",
        },
        "low_pass_0p3hz": {
            "sensor_lowpass_acc_hz",
            "sensor_lowpass_gyro_hz",
            "sensor_filter_order",
            "gravity_lowpass_hz",
            "gravity_filter_order",
        },
    }
    active = active_by_method[method]
    normalized: dict[str, Any] = {}
    vector_lengths = {
        "process_covariance_diagonal_per_second": 5,
        "observation_covariance_diagonal_rad2": 2,
        "initial_covariance_diagonal": 5,
    }
    order_fields = {"sensor_filter_order", "gravity_filter_order"}
    for name in numeric_keys:
        if name not in declared and name not in active:
            continue
        raw = declared.get(name, numeric_defaults[name])
        if name in vector_lengths:
            resolved: Any = _positive_vector(
                raw,
                name=f"signal.imu.{name}",
                length=vector_lengths[name],
            )
        elif name in order_fields:
            if (
                isinstance(raw, (bool, np.bool_))
                or not isinstance(raw, (int, np.integer))
                or not 1 <= int(raw) <= 20
            ):
                raise ValueError(f"signal.imu.{name} must be an integer in [1,20]")
            resolved = int(raw)
        else:
            resolved = _finite_number(raw, name=f"signal.imu.{name}")
        if name not in active:
            if resolved != numeric_defaults[name]:
                raise ValueError(
                    f"signal.imu.{name} is inactive for gravity_method={method}; "
                    "remove the non-default value"
                )
            continue
        normalized[name] = resolved

    effective = {
        "gravity_method": method,
        "initialization": derived_initialization,
        "comparison_method": derived_comparison,
        **normalized,
        "output_units": output_units,
        "required_axes": 6,
        "failure_action": "fail_closed",
    }
    if method in calibrated_methods:
        roll_pitch_ekf_config_from_resolved(effective).validate(fs_hz)
    else:
        from .imu import ImuProfile

        ImuProfile(
            gravity_method=(
                "no_precalibration_ekf"
                if method == "quaternion_error_state_ekf"
                else "lpf_0p3"
            ),
            acceleration_lowpass_hz=float(effective["sensor_lowpass_acc_hz"]),
            gyroscope_lowpass_hz=float(effective["sensor_lowpass_gyro_hz"]),
            sensor_filter_order=int(effective["sensor_filter_order"]),
            gravity_lowpass_hz=float(effective.get("gravity_lowpass_hz", 0.3)),
            gravity_filter_order=int(effective.get("gravity_filter_order", 2)),
        ).validate(fs_hz)
    return effective


def materialize_signal_preprocessing_config(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Canonicalize all executable signal preprocessing controls before hashing."""

    if not isinstance(value, Mapping):
        raise ValueError("signal must be a mapping")
    declared = dict(value)
    allowed = {
        "internal_fs_hz",
        "channel_order",
        "ppg_native_unit",
        "accelerometer_input_unit",
        "gyroscope_input_unit",
        "ppg_filter",
        "peak_detector",
        "analysis_view",
        "gap_repair",
        "imu",
        "dl_resampling",
        "normalization",
    }
    unknown = sorted(set(declared) - allowed)
    if unknown:
        raise ValueError(f"signal contains unknown fields: {unknown}")

    fs_hz = _finite_number(
        declared.get("internal_fs_hz", CANONICAL_FS_HZ),
        name="signal.internal_fs_hz",
    )
    if fs_hz != CANONICAL_FS_HZ:
        raise ValueError(
            "signal.internal_fs_hz must equal the implemented 400 Hz internal grid; "
            "configure signal.dl_resampling.target_fs_hz for raw model input"
        )
    channel_order = declared.get(
        "channel_order", ["RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ"]
    )
    if not isinstance(channel_order, (list, tuple)) or list(channel_order) != [
        "RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ"
    ]:
        raise ValueError("signal.channel_order must be the canonical eight-channel schema")
    ppg_unit = str(declared.get("ppg_native_unit", "raw_counts")).strip().lower()
    if ppg_unit != "raw_counts":
        raise ValueError("signal.ppg_native_unit must be raw_counts")

    raw_filter = declared.get("ppg_filter", {})
    if raw_filter is None:
        raw_filter = {}
    if not isinstance(raw_filter, Mapping):
        raise ValueError("signal.ppg_filter must be a mapping")
    raw_filter = dict(raw_filter)
    filter_defaults = {
        "family": "butterworth_sos",
        "order": 3,
        "low_hz": 0.2,
        "high_hz": 8.0,
        "phase": "zero_phase",
        "short_signal_policy": "reject",
        "notch_enabled": False,
    }
    unknown_filter = sorted(set(raw_filter) - set(filter_defaults))
    if unknown_filter:
        raise ValueError(f"signal.ppg_filter contains unknown fields: {unknown_filter}")
    for name in ("family", "phase", "short_signal_policy"):
        if name in raw_filter and raw_filter[name] != filter_defaults[name]:
            raise ValueError(f"signal.ppg_filter.{name} requests an unimplemented policy")
    if "notch_enabled" in raw_filter and raw_filter["notch_enabled"] is not False:
        raise ValueError("signal.ppg_filter.notch_enabled requests an unimplemented policy")
    order = raw_filter.get("order", filter_defaults["order"])
    low_hz = _finite_number(
        raw_filter.get("low_hz", filter_defaults["low_hz"]),
        name="signal.ppg_filter.low_hz",
    )
    high_hz = _finite_number(
        raw_filter.get("high_hz", filter_defaults["high_hz"]),
        name="signal.ppg_filter.high_hz",
    )
    design_ppg_sos(low_hz, high_hz, fs_hz=fs_hz, order=order)
    ppg_filter = {
        **filter_defaults,
        "order": int(order),
        "low_hz": float(low_hz),
        "high_hz": float(high_hz),
    }

    raw_analysis = declared.get("analysis_view", {})
    if raw_analysis is None:
        raw_analysis = {}
    if not isinstance(raw_analysis, Mapping):
        raise ValueError("signal.analysis_view must be a mapping")
    raw_analysis = dict(raw_analysis)
    analysis_defaults = {
        "non_identity_source": "aligned_x_ar",
        "non_identity_semantics": "rate_only",
        "additional_filter": "none",
    }
    allowed_analysis = {"direct_source", *analysis_defaults}
    unknown_analysis = sorted(set(raw_analysis) - allowed_analysis)
    if unknown_analysis:
        raise ValueError(f"signal.analysis_view contains unknown fields: {unknown_analysis}")
    expected_direct_source = canonical_ppg_direct_source(
        ppg_filter["low_hz"], ppg_filter["high_hz"]
    )
    direct_source = raw_analysis.get("direct_source")
    if direct_source is not None and str(direct_source) not in {
        "x_filter",
        "configured_ppg_filter",
        expected_direct_source,
    }:
        raise ValueError(
            "signal.analysis_view.direct_source numeric alias disagrees with "
            "signal.ppg_filter"
        )
    for name, expected in analysis_defaults.items():
        if name in raw_analysis and raw_analysis[name] != expected:
            raise ValueError(f"signal.analysis_view.{name} requests unsupported semantics")
    analysis_view = {
        "direct_source": expected_direct_source,
        **analysis_defaults,
    }

    raw_gap = declared.get("gap_repair", {})
    if raw_gap is None:
        raw_gap = {}
    if not isinstance(raw_gap, Mapping):
        raise ValueError("signal.gap_repair must be a mapping")
    raw_gap = dict(raw_gap)
    gap_defaults = {
        "method": "linear_inside_only",
        "max_gap_samples": 100,
        "edge_extrapolation": False,
        "all_missing_channel_action": "reject_record",
    }
    unknown_gap = sorted(set(raw_gap) - set(gap_defaults))
    if unknown_gap:
        raise ValueError(f"signal.gap_repair contains unknown fields: {unknown_gap}")
    for name in ("method", "all_missing_channel_action"):
        if name in raw_gap and raw_gap[name] != gap_defaults[name]:
            raise ValueError(f"signal.gap_repair.{name} requests an unsupported policy")
    if "edge_extrapolation" in raw_gap and raw_gap["edge_extrapolation"] is not False:
        raise ValueError("signal.gap_repair.edge_extrapolation requests an unsupported policy")
    max_gap = raw_gap.get("max_gap_samples", gap_defaults["max_gap_samples"])
    if (
        isinstance(max_gap, (bool, np.bool_))
        or not isinstance(max_gap, (int, np.integer))
        or int(max_gap) < 0
    ):
        raise ValueError("signal.gap_repair.max_gap_samples must be a non-negative integer")
    gap_repair = {**gap_defaults, "max_gap_samples": int(max_gap)}

    effective = dict(declared)
    effective.update(
        {
            "internal_fs_hz": CANONICAL_FS_HZ,
            "channel_order": ["RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ"],
            "ppg_native_unit": "raw_counts",
            "accelerometer_input_unit": _canonical_acceleration_unit(
                declared.get("accelerometer_input_unit", "g")
            ),
            "gyroscope_input_unit": _canonical_gyroscope_unit(
                declared.get("gyroscope_input_unit", "deg/s")
            ),
            "ppg_filter": ppg_filter,
            "analysis_view": analysis_view,
            "gap_repair": gap_repair,
            "imu": _materialize_imu_profile(declared.get("imu"), fs_hz=fs_hz),
        }
    )
    return effective


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
    signal_config = materialize_signal_preprocessing_config(config["signal"])
    expected_signal_keys = {
        "internal_fs_hz", "channel_order", "ppg_native_unit",
        "accelerometer_input_unit", "gyroscope_input_unit", "ppg_filter",
        "peak_detector", "analysis_view", "gap_repair", "imu",
        "dl_resampling", "normalization",
    }
    if set(signal_config) != expected_signal_keys:
        raise ValueError(
            "resolved signal key mismatch: "
            f"missing={sorted(expected_signal_keys-set(signal_config))}, "
            f"unknown={sorted(set(signal_config)-expected_signal_keys)}"
        )
    from ..module_registry import resolve_peak_detector_config

    resolve_peak_detector_config(signal_config)
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
    implemented_filter_controls = {
        "family": "butterworth_sos",
        "phase": "zero_phase",
        "short_signal_policy": "reject",
        "notch_enabled": False,
    }
    if any(
        ppg_filter.get(key) != value
        for key, value in implemented_filter_controls.items()
    ):
        raise ValueError("resolved PPG filter requests an unimplemented family or phase")
    filter_pair = (float(ppg_filter.get("low_hz")), float(ppg_filter.get("high_hz")))
    design_ppg_sos(
        *filter_pair,
        fs_hz=fs_hz,
        order=ppg_filter.get("order"),
    )
    if set(ppg_filter) != set(implemented_filter_controls) | {
        "order",
        "low_hz",
        "high_hz",
    }:
        raise ValueError("resolved PPG filter contains missing or unknown keys")
    if signal_config.get("channel_order") != ["RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ"]:
        raise ValueError("resolved channel_order differs from the frozen eight-channel schema")
    if signal_config.get("ppg_native_unit") != "raw_counts":
        raise ValueError("formal PPG native unit must be raw_counts")
    analysis_view = signal_config.get("analysis_view")
    if not isinstance(analysis_view, Mapping) or set(analysis_view) != {
        "direct_source",
        "non_identity_source",
        "non_identity_semantics",
        "additional_filter",
    }:
        raise ValueError("resolved signal.analysis_view key mismatch")
    direct_source = str(analysis_view.get("direct_source"))
    if direct_source != canonical_ppg_direct_source(*filter_pair):
        raise ValueError(
            "analysis_view.direct_source must name the configured x_filter"
        )
    if (
        analysis_view.get("non_identity_source") != "aligned_x_ar"
        or analysis_view.get("non_identity_semantics") != "rate_only"
        or analysis_view.get("additional_filter") != "none"
    ):
        raise ValueError("resolved signal.analysis_view requests unsupported semantics")
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
        "sensor_filter_order", "gravity_filter_order",
        "output_units", "required_axes", "failure_action",
    }
    calibrated_extra_keys = {
        "calibration_start_s",
        "calibration_stop_s",
        "process_covariance_diagonal_per_second",
        "observation_covariance_diagonal_rad2",
        "initial_covariance_diagonal",
        "dynamic_observation_scale",
        "gravity_mps2",
    }
    allowed_imu_keys = common_imu_keys | calibrated_extra_keys
    unknown_imu_keys = set(imu_config) - allowed_imu_keys
    if unknown_imu_keys:
        raise ValueError(
            f"signal.imu contains unknown keys: {sorted(unknown_imu_keys)}"
        )
    if (
        imu_config.get("failure_action") != "fail_closed"
        or imu_config.get("required_axes") != 6
        or imu_config.get("output_units")
        != {"acceleration": "m/s^2", "gyroscope": "rad/s", "jerk": "m/s^3"}
    ):
        raise ValueError("resolved signal.imu structural contract is invalid")
    if gravity_profile in calibrated_profiles:
        if (
            imu_config.get("initialization")
            != "same_participant_static_calibration"
            or imu_config.get("comparison_method") != "profile_a_lowpass_0p3hz"
        ):
            raise ValueError(
                "calibrated EKF initialization/comparator identity drift"
            )
    else:
        if set(imu_config) - common_imu_keys:
            raise ValueError("legacy signal.imu contains calibrated-only keys")
        if (
            imu_config.get("initialization") != "online_no_precalibration"
            or imu_config.get("comparison_method") != "lowpass_0p3hz"
        ):
            raise ValueError("legacy IMU profile declaration drift")
    dl_resampling = validate_dl_resampling_config(signal_config.get("dl_resampling"))
    from ..normalization import RawNormalizationConfig

    resolved_normalization = RawNormalizationConfig.from_mapping(
        signal_config.get("normalization")
    ).to_mapping()
    for name, expected in (("acc_unit", acc_unit), ("gyro_unit", gyro_unit)):
        observed = (
            record.get(name)
            if isinstance(record, Mapping)
            else getattr(record, name, None)
        )
        if observed is not None:
            canonical_observed = (
                _canonical_acceleration_unit(observed)
                if name == "acc_unit"
                else _canonical_gyroscope_unit(observed)
            )
        else:
            canonical_observed = None
        if canonical_observed is not None and canonical_observed != expected:
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
                "roll_rad": motion_imu.roll_rad,
                "pitch_rad": motion_imu.pitch_rad,
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
            acceleration_lowpass_hz=float(
                imu_config.get("sensor_lowpass_acc_hz", 20.0)
            ),
            gyroscope_lowpass_hz=float(
                imu_config.get("sensor_lowpass_gyro_hz", 40.0)
            ),
            sensor_filter_order=imu_config.get("sensor_filter_order", 3),
            gravity_lowpass_hz=float(
                imu_config.get("gravity_lowpass_hz", 0.3)
            ),
            gravity_filter_order=imu_config.get("gravity_filter_order", 2),
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
        "ppg_filter": {
            "family": "butterworth_sos",
            "phase": "zero_phase",
            "order": int(ppg_filter["order"]),
            "low_hz": float(ppg_filter["low_hz"]),
            "high_hz": float(ppg_filter["high_hz"]),
            "short_signal_policy": "reject",
            "notch_enabled": False,
        },
        "analysis_view": dict(analysis_view),
        "imu_status": imu.status,
        "imu_diagnostics": imu.diagnostics,
        "imu_valid_fraction": float(np.mean(imu.valid_mask)),
        "dl_resampling": dl_resampling,
        "raw_normalization": resolved_normalization,
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
