"""注册表驱动的 stateful IMU runtime / Registry-bound stateful IMU runtime.

中文：公共入口只接受 profile ID；采样率、重力路线、相位模式和 EKF 参数全部
由版本化注册表解析。Causal processor 持久保存 SOS、ESKF、LPF、jerk 和时间轴
状态，因此任意合法分块与整段处理一致。

English: Public execution is selected only by profile ID. Sampling rate, gravity
route, phase mode, and EKF parameters come from the registry. The causal processor
retains filter, estimator, jerk, and time-grid state, so valid chunking is equivalent
to one-shot processing.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .contracts import ImuPreprocessResult, ProcessingStatus, QualityAssessment, QualityIssue
from .imu import IMU_CHANNELS, _invalid_shape_result
from .imu_math import (
    EskfConfiguration,
    NoPrecalibrationEskf,
    convert_imu_to_si,
    filter_axes,
)
from .quality import inspect_and_repair_signal, validate_timestamp_grid, with_contract_issues
from .registry import get_profile, registry_sha256


def _eskf_config(profile: dict[str, Any]) -> EskfConfiguration:
    """从 registry 构造全部 ESKF 参数 / Resolve every active ESKF parameter."""

    return EskfConfiguration(
        gyro_noise_density=float(profile["gyro_noise_density"]),
        gyro_bias_random_walk=float(profile["gyro_bias_random_walk"]),
        base_accel_angle_noise_rad=float(np.deg2rad(profile["base_accel_angle_noise_deg"])),
        nis_gate=float(profile["nis_gate_2d_99p9"]),
        update_decimation=int(profile["gravity_update_decimation"]),
        min_tracking_sec=float(profile["minimum_tracking_sec"]),
        min_accepted_updates=int(profile["minimum_accepted_updates"]),
        tracking_tilt_sigma_deg=float(profile["tracking_tilt_sigma_deg"]),
        max_prediction_only_sec=float(profile["maximum_prediction_only_sec"]),
        max_tracking_tilt_sigma_deg=float(profile["maximum_tracking_tilt_sigma_deg"]),
        max_bias_rad_s=float(profile["maximum_gyro_bias_rad_s"]),
    )


def _resolve_profile(profile_id: str, observed_fs_hz: float) -> dict[str, Any]:
    """解析并强制 IMU profile / Resolve and enforce an IMU profile."""

    profile = get_profile(profile_id)
    if profile.get("modality") != "imu":
        raise ValueError(f"profile_mismatch:not_imu:{profile_id}")
    expected = float(profile["sampling_rate_hz"])
    if not np.isclose(float(observed_fs_hz), expected, rtol=0.0, atol=1e-12):
        raise ValueError(f"profile_mismatch:fs_hz={observed_fs_hz}, expected={expected}")
    if profile.get("phase_mode") != "causal_stateful":
        raise ValueError(f"profile_mismatch:only_registered_causal_imu_is_active:{profile_id}")
    if profile.get("algorithm_key") not in {"ekf", "lpf_0p3"}:
        raise ValueError(f"profile_mismatch:algorithm_key:{profile_id}")
    return profile


def _failed_result(
    quality: QualityAssessment,
    profile_id: str,
    sample_count: int,
    reason: str,
) -> ImuPreprocessResult:
    """构造不改变 runtime state 的失败结果 / Build a state-preserving failure."""

    return ImuPreprocessResult(
        quality.status,
        None,
        None,
        None,
        None,
        np.zeros(sample_count, dtype=bool),
        quality,
        {"reason": reason, "silent_fallback": False},
        profile_id,
    )


def _common_jerk(
    dynamic: np.ndarray,
    route_valid: np.ndarray,
    fs_hz: float,
    previous_dynamic: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """计算跨 chunk 连续 jerk 与公共 mask / Compute continuous jerk and mask."""

    jerk = np.full_like(dynamic, np.nan)
    if dynamic.shape[0] == 0:
        return jerk, np.zeros(0, dtype=bool)
    if previous_dynamic is not None and route_valid[0] and np.isfinite(previous_dynamic).all():
        jerk[0] = (dynamic[0] - previous_dynamic) * float(fs_hz)
    if dynamic.shape[0] > 1:
        adjacent_valid = route_valid[1:] & route_valid[:-1]
        differences = np.diff(dynamic, axis=0) * float(fs_hz)
        jerk[1:][adjacent_valid] = differences[adjacent_valid]
    common = route_valid & np.isfinite(dynamic).all(axis=1) & np.isfinite(jerk).all(axis=1)
    return jerk, common


class CausalImuProcessor:
    """持久 causal IMU processor / Persistent causal IMU processor."""

    def __init__(
        self,
        profile_id: str,
        *,
        fs_hz: float,
        acceleration_unit: str,
        gyroscope_unit: str,
    ) -> None:
        """绑定 profile 并创建空 runtime state / Bind profile and create state."""

        self.profile_id = profile_id
        self.profile = _resolve_profile(profile_id, fs_hz)
        self.fs_hz = float(fs_hz)
        self.acceleration_unit = acceleration_unit
        self.gyroscope_unit = gyroscope_unit
        self.acc_filter_state: np.ndarray | None = None
        self.gyro_filter_state: np.ndarray | None = None
        self.gravity_filter_state: np.ndarray | None = None
        self.last_dynamic: np.ndarray | None = None
        self.previous_timestamp_s: float | None = None
        self.global_sample_index = 0
        self.estimator = (
            NoPrecalibrationEskf(self.fs_hz, _eskf_config(self.profile))
            if self.profile["algorithm_key"] == "ekf"
            else None
        )

    def reset_for_new_session(self) -> None:
        """显式重置全部状态 / Explicitly reset the complete session state."""

        self.acc_filter_state = None
        self.gyro_filter_state = None
        self.gravity_filter_state = None
        self.last_dynamic = None
        self.previous_timestamp_s = None
        self.global_sample_index = 0
        self.estimator = (
            NoPrecalibrationEskf(self.fs_hz, _eskf_config(self.profile))
            if self.profile["algorithm_key"] == "ekf"
            else None
        )

    def process_chunk(
        self,
        acceleration: np.ndarray,
        gyroscope: np.ndarray,
        *,
        timestamps_s: np.ndarray | None = None,
    ) -> ImuPreprocessResult:
        """处理一个连续 chunk / Process one contiguous chunk."""

        acc_raw = np.asarray(acceleration, dtype=np.float64)
        gyro_raw = np.asarray(gyroscope, dtype=np.float64)
        if (
            acc_raw.ndim != 2
            or gyro_raw.ndim != 2
            or acc_raw.shape[1] != 3
            or gyro_raw.shape[1] != 3
            or acc_raw.shape[0] != gyro_raw.shape[0]
        ):
            return _invalid_shape_result(acc_raw, self.fs_hz, self.profile_id)
        stacked = np.column_stack([acc_raw, gyro_raw])
        quality = inspect_and_repair_signal(
            stacked,
            self.fs_hz,
            channel_names=IMU_CHANNELS,
            profile_id=self.profile_id,
            max_gap_sec=0.25,
            max_nonfinite_fraction=0.01,
            min_duration_sec=0.0,
            flatline_channels=[],
        )
        quality = with_contract_issues(
            quality,
            validate_timestamp_grid(
                timestamps_s,
                self.fs_hz,
                stacked.shape[0],
                previous_timestamp_s=self.previous_timestamp_s,
            ),
        )
        if quality.status in {ProcessingStatus.INVALID, ProcessingStatus.INSUFFICIENT}:
            return _failed_result(quality, self.profile_id, stacked.shape[0], "quality_gate_failed")
        try:
            repaired = np.asarray(quality.signal, dtype=np.float64)
            acc_si, gyro_si = convert_imu_to_si(
                repaired[:, :3],
                repaired[:, 3:],
                acceleration_unit=self.acceleration_unit,
                gyroscope_unit=self.gyroscope_unit,
            )
        except ValueError as exc:
            quality.issues.append(QualityIssue("unit_unknown", "fatal", str(exc)))
            quality.status = ProcessingStatus.INVALID
            return _failed_result(quality, self.profile_id, stacked.shape[0], "unit_unknown")

        order = int(self.profile["sensor_filter_order"])
        acc_filtered, next_acc_state = filter_axes(
            acc_si,
            self.fs_hz,
            float(self.profile["acceleration_lowpass_hz"]),
            order=order,
            phase_mode="causal_stateful",
            initial_state=self.acc_filter_state,
        )
        gyro_filtered, next_gyro_state = filter_axes(
            gyro_si,
            self.fs_hz,
            float(self.profile["gyroscope_lowpass_hz"]),
            order=order,
            phase_mode="causal_stateful",
            initial_state=self.gyro_filter_state,
        )

        state_names: list[str] = []
        route_diagnostics: dict[str, Any] = {}
        if self.profile["algorithm_key"] == "ekf":
            assert self.estimator is not None
            count = stacked.shape[0]
            gravity = np.full((count, 3), np.nan)
            route_valid = np.zeros(count, dtype=bool)
            quaternions = np.full((count, 4), np.nan)
            biases = np.full((count, 3), np.nan)
            tilt = np.full(count, np.nan)
            nis = np.full(count, np.nan)
            accepted = np.zeros(count, dtype=bool)
            downweighted = np.zeros(count, dtype=bool)
            for local_index in range(count):
                sample = self.estimator.step(
                    acc_filtered[local_index],
                    gyro_filtered[local_index],
                    self.global_sample_index + local_index,
                )
                state_names.append(str(sample["state"]))
                route_valid[local_index] = bool(sample["valid"])
                gravity[local_index] = sample["gravity"]
                if "quaternion" in sample:
                    quaternions[local_index] = sample["quaternion"]
                    biases[local_index] = sample["bias"]
                    tilt[local_index] = sample["tilt_sigma_deg"]
                    nis[local_index] = sample["nis"]
                    accepted[local_index] = sample["accepted"]
                    downweighted[local_index] = sample["downweighted"]
            route_diagnostics.update(
                {
                    "quaternion_wxyz": quaternions,
                    "gyro_bias_rad_s": biases,
                    "statistical_tilt_sigma_deg": tilt,
                    "nis": nis,
                    "accepted_gravity_update": accepted,
                    "dynamic_accel_downweighted": downweighted,
                    "no_static_precalibration": True,
                    "physical_observability_status": "unverified_no_static_precalibration",
                    "bias_observability": "partial_full_unverified",
                    "yaw_reference": "unobservable_relative_only",
                    "accelerometer_bias_status": "not_estimated",
                }
            )
        else:
            gravity, next_gravity_state = filter_axes(
                acc_filtered,
                self.fs_hz,
                0.3,
                order=int(self.profile["gravity_filter_order"]),
                phase_mode="causal_stateful",
                initial_state=self.gravity_filter_state,
            )
            self.gravity_filter_state = next_gravity_state
            route_valid = np.isfinite(gravity).all(axis=1)
            state_names = ["tracking" if value else "no_estimate" for value in route_valid]

        dynamic = acc_filtered - gravity
        dynamic[~route_valid] = np.nan
        jerk, common_valid = _common_jerk(dynamic, route_valid, self.fs_hz, self.last_dynamic)
        terminal_state = state_names[-1] if state_names else "initialization_pending"
        if terminal_state == "no_estimate":
            overall = ProcessingStatus.NO_ESTIMATE
        elif not np.any(common_valid):
            overall = ProcessingStatus.INITIALIZATION_PENDING
        elif np.all(common_valid):
            overall = quality.status
        else:
            overall = ProcessingStatus.PARTIAL

        # 中文：只有成功计算后才提交 state；失败 chunk 不污染后续。
        # English: Commit state only after successful processing.
        self.acc_filter_state = next_acc_state
        self.gyro_filter_state = next_gyro_state
        self.last_dynamic = dynamic[-1].copy() if route_valid.size and route_valid[-1] else None
        self.global_sample_index += stacked.shape[0]
        if timestamps_s is not None and np.asarray(timestamps_s).size:
            self.previous_timestamp_s = float(np.asarray(timestamps_s).ravel()[-1])
        diagnostics = {
            "registry_id": "m3_preprocessing_profiles_corrected_v1",
            "registry_sha256": registry_sha256(),
            "resolved_profile": self.profile,
            "gravity_method": self.profile["gravity_method"],
            "phase_mode": "causal_stateful",
            "state_per_sample": state_names,
            "state_counts": dict(sorted(Counter(state_names).items())),
            "terminal_state": terminal_state,
            "valid_fraction": float(np.mean(common_valid)) if common_valid.size else 0.0,
            "first_valid_index": int(np.flatnonzero(common_valid)[0]) if np.any(common_valid) else None,
            "last_valid_index": int(np.flatnonzero(common_valid)[-1]) if np.any(common_valid) else None,
            "silent_fallback": False,
            **route_diagnostics,
        }
        return ImuPreprocessResult(
            overall,
            gravity,
            dynamic,
            gyro_filtered,
            jerk,
            common_valid,
            quality,
            diagnostics,
            self.profile_id,
        )


def preprocess_imu(
    acceleration: np.ndarray,
    gyroscope: np.ndarray,
    fs_hz: float,
    *,
    acceleration_unit: str,
    gyroscope_unit: str,
    profile_id: str,
    timestamps_s: np.ndarray | None = None,
) -> ImuPreprocessResult:
    """唯一 one-shot facade；执行参数由 profile 决定 / Sole registry facade."""

    _resolve_profile(profile_id, fs_hz)
    processor = CausalImuProcessor(
        profile_id,
        fs_hz=fs_hz,
        acceleration_unit=acceleration_unit,
        gyroscope_unit=gyroscope_unit,
    )
    return processor.process_chunk(acceleration, gyroscope, timestamps_s=timestamps_s)


__all__ = ["CausalImuProcessor", "preprocess_imu"]
