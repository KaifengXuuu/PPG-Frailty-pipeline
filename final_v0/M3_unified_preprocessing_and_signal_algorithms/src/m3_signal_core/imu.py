"""统一 IMU 公共入口 / Unified public IMU entry point.

中文：本入口让 ESKF 主路线和 0.3 Hz LPF 对照复用同一数据清洗、显式单位转换、
20/40 Hz 低通和 backward-difference vector jerk。任何路线失败都返回自己的
状态；严禁路线间静默 fallback。

English: The ESKF primary and 0.3 Hz LPF comparator share cleaning, explicit unit
conversion, 20/40 Hz filtering, and backward-difference vector jerk. Each route reports
its own failure state and never silently falls back to the other.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import (
    ImuPreprocessResult,
    ProcessingStatus,
    QualityAssessment,
    QualityIssue,
)
from .imu_math import (
    EskfConfiguration,
    NoPrecalibrationEskf,
    STANDARD_GRAVITY_MPS2,
    convert_imu_to_si,
    filter_axes,
    run_ekf,
)
from .quality import inspect_and_repair_signal


IMU_CHANNELS = ["AX", "AY", "AZ", "GX", "GY", "GZ"]


def vector_jerk(dynamic_acceleration: np.ndarray, fs_hz: float) -> np.ndarray:
    """唯一 jerk 定义：Δa_dyn/Δt / Canonical backward-difference vector jerk."""

    acceleration = np.asarray(dynamic_acceleration, dtype=np.float64)
    jerk = np.full_like(acceleration, np.nan)
    if acceleration.ndim == 2 and acceleration.shape[0] > 1:
        jerk[1:] = np.diff(acceleration, axis=0) * float(fs_hz)
    return jerk


def _invalid_shape_result(
    signal: np.ndarray,
    fs_hz: float,
    profile_id: str,
) -> ImuPreprocessResult:
    """构造缺六轴的 fail-closed 结果 / Build a fail-closed missing-axis result."""

    values = np.asarray(signal, dtype=np.float64)
    length = values.shape[0] if values.ndim else 0
    quality = QualityAssessment(
        status=ProcessingStatus.INVALID,
        signal=values,
        valid_mask=np.isfinite(values),
        repair_mask=np.zeros_like(values, dtype=bool),
        issues=[
            QualityIssue(
                "missing_channel",
                "fatal",
                "Exactly six aligned AX,AY,AZ,GX,GY,GZ channels are required.",
            )
        ],
        metrics={"fs_hz": float(fs_hz)},
        profile_id=profile_id,
    )
    return ImuPreprocessResult(
        ProcessingStatus.INVALID,
        None,
        None,
        None,
        None,
        np.zeros(length, dtype=bool),
        quality,
        {"reason": "missing_channel", "silent_fallback": False},
        profile_id,
    )


def preprocess_imu(
    acceleration: np.ndarray,
    gyroscope: np.ndarray,
    fs_hz: float,
    *,
    acceleration_unit: str,
    gyroscope_unit: str,
    gravity_method: str,
    phase_mode: str,
    profile_id: str,
) -> ImuPreprocessResult:
    """执行 ESKF 或 LPF 且保持同一输出 schema / Execute one gravity route."""

    acc_raw = np.asarray(acceleration, dtype=np.float64)
    gyro_raw = np.asarray(gyroscope, dtype=np.float64)
    if (
        acc_raw.ndim != 2
        or gyro_raw.ndim != 2
        or acc_raw.shape[1] != 3
        or gyro_raw.shape[1] != 3
        or acc_raw.shape[0] != gyro_raw.shape[0]
    ):
        return _invalid_shape_result(acc_raw, fs_hz, profile_id)

    stacked = np.column_stack([acc_raw, gyro_raw])
    quality = inspect_and_repair_signal(
        stacked,
        fs_hz,
        channel_names=IMU_CHANNELS,
        profile_id=profile_id,
        max_gap_sec=0.25,
        max_nonfinite_fraction=0.01,
        min_duration_sec=3.0,
        flatline_channels=[],
    )
    if quality.status in {ProcessingStatus.INVALID, ProcessingStatus.INSUFFICIENT}:
        return ImuPreprocessResult(
            quality.status,
            None,
            None,
            None,
            None,
            np.zeros(stacked.shape[0], dtype=bool),
            quality,
            {"reason": "quality_gate_failed", "silent_fallback": False},
            profile_id,
        )

    repaired = np.asarray(quality.signal, dtype=np.float64)
    try:
        acc_si, gyro_si = convert_imu_to_si(
            repaired[:, :3],
            repaired[:, 3:],
            acceleration_unit=acceleration_unit,
            gyroscope_unit=gyroscope_unit,
        )
    except ValueError as exc:
        quality.issues.append(QualityIssue("unit_unknown", "fatal", str(exc)))
        quality.status = ProcessingStatus.INVALID
        return ImuPreprocessResult(
            quality.status,
            None,
            None,
            None,
            None,
            np.zeros(stacked.shape[0], dtype=bool),
            quality,
            {"reason": "unit_unknown", "silent_fallback": False},
            profile_id,
        )

    try:
        acc_filtered, acc_state = filter_axes(
            acc_si, fs_hz, 20.0, order=3, phase_mode=phase_mode
        )
        gyro_filtered, gyro_state = filter_axes(
            gyro_si, fs_hz, 40.0, order=3, phase_mode=phase_mode
        )
    except ValueError as exc:
        quality.issues.append(
            QualityIssue("filter_input_too_short", "insufficient", str(exc))
        )
        quality.status = ProcessingStatus.INSUFFICIENT
        return ImuPreprocessResult(
            quality.status,
            None,
            None,
            None,
            None,
            np.zeros(stacked.shape[0], dtype=bool),
            quality,
            {"reason": "filter_input_too_short", "silent_fallback": False},
            profile_id,
        )

    diagnostics: dict[str, Any] = {
        "acceleration_input_unit": acceleration_unit,
        "gyroscope_input_unit": gyroscope_unit,
        "acceleration_output_unit": "m/s^2",
        "gyroscope_output_unit": "rad/s",
        "acceleration_lowpass_hz": 20.0,
        "gyroscope_lowpass_hz": 40.0,
        "phase_mode": phase_mode,
        "acc_filter_final_state": acc_state,
        "gyro_filter_final_state": gyro_state,
        "silent_fallback": False,
    }
    if gravity_method == "ekf":
        gravity, sample_valid, route_diagnostics = run_ekf(
            acc_filtered, gyro_filtered, fs_hz
        )
        diagnostics.update(route_diagnostics)
        diagnostics["gravity_method"] = "quaternion_error_state_ekf_without_precalibration"
        overall = (
            quality.status
            if np.any(sample_valid)
            else ProcessingStatus.INITIALIZATION_PENDING
        )
    elif gravity_method == "lpf_0p3":
        try:
            gravity, gravity_state = filter_axes(
                acc_filtered, fs_hz, 0.3, order=2, phase_mode=phase_mode
            )
        except ValueError as exc:
            quality.issues.append(
                QualityIssue("gravity_filter_input_too_short", "insufficient", str(exc))
            )
            quality.status = ProcessingStatus.INSUFFICIENT
            return ImuPreprocessResult(
                quality.status,
                None,
                None,
                None,
                None,
                np.zeros(stacked.shape[0], dtype=bool),
                quality,
                {"reason": "gravity_filter_input_too_short", "silent_fallback": False},
                profile_id,
            )
        sample_valid = np.ones(stacked.shape[0], dtype=bool)
        diagnostics["gravity_method"] = "second_order_lowpass_0p3_hz"
        diagnostics["gravity_filter_final_state"] = gravity_state
        diagnostics["gravity_norm_error_mps2"] = (
            np.linalg.norm(gravity, axis=1) - STANDARD_GRAVITY_MPS2
        )
        overall = quality.status
    else:
        raise ValueError("gravity_method must be 'ekf' or 'lpf_0p3'")

    dynamic = acc_filtered - gravity
    dynamic[~sample_valid] = np.nan
    jerk = vector_jerk(dynamic, fs_hz)
    sample_valid = sample_valid & np.isfinite(dynamic).all(axis=1)
    if sample_valid.size:
        sample_valid[0] = False
    return ImuPreprocessResult(
        overall,
        gravity,
        dynamic,
        gyro_filtered,
        jerk,
        sample_valid,
        quality,
        diagnostics,
        profile_id,
    )


__all__ = [
    "EskfConfiguration",
    "NoPrecalibrationEskf",
    "STANDARD_GRAVITY_MPS2",
    "convert_imu_to_si",
    "preprocess_imu",
    "vector_jerk",
]

