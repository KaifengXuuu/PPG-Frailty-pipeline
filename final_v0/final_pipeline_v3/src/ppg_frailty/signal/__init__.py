"""V1 信号层公共 facade / Public facade for the V1 signal layer."""

from __future__ import annotations

from typing import Any

from ..contracts import PulseResult, SignalRoute
from .imu import (
    CausalImuProcessor,
    EskfConfiguration,
    GravityComparisonResult,
    ImuProfile,
    ImuPreprocessResult,
    compare_ekf_lpf_gravity,
    estimate_gravity_lpf,
    estimate_gravity_no_precalibration_ekf,
    preprocess_imu,
)
from .morphology import MorphologyResult, extract_morphology, require_direct_route
from .motion_imu import (
    CALIBRATED_ROLL_PITCH_EKF_PROFILE_ID,
    MOTION_IMU_CHANNEL_SCHEMA,
    MOTION_IMU_CHANNEL_UNITS,
    FORMAL_STATIC_CALIBRATION_ROLES,
    PROFILE_A_LPF_ID,
    PTT_STATIC_CALIBRATION_ROLE,
    MotionImuCalibration,
    MotionImuResult,
    RollPitchEkfConfig,
    fit_motion_imu_calibration,
    preprocess_motion_imu_calibrated_ekf,
    preprocess_motion_imu_lpf_ablation,
)
from .optical import OpticalFeatureResult, extract_dual_optical
from .peaks import MIN_BASIC_RATE_PEAKS, detect_pulses
from .preprocess import (
    ABLATION_PPG_FILTER_PROFILE_ID,
    REFERENCE_PPG_FILTER_PROFILE_ID,
    InputQC,
    PpgFilterProfile,
    build_signal_views,
    get_ppg_filter_profile,
    inspect_and_repair,
    preprocess_ppg_pair,
)
from .prv import MIN_TIME_DOMAIN_PRV_INTERVALS, PrvResult, compute_prv
from .resample import (
    DlResampleResult,
    SynchronizedResampleResult,
    V2_DL_RESAMPLING_TARGETS_HZ,
    resample_dl_view,
    resample_synchronized_channels,
    validate_dl_resampling_config,
)
from .sqi import (
    SqiCalibrator,
    SqiConfig,
    SqiDiagnosticComponent,
    SqiDiagnosticConfig,
    SqiDiagnostics,
    evaluate_quality,
    evaluate_quality_diagnostics,
    fit_sqi_calibrator,
    quality_component_scores,
)
from .views import (
    CANONICAL_FS_HZ,
    CanonicalSignalViews,
    WindowPlan,
    WindowSlice,
    extract_window,
)


def extract_direct_features(
    views: CanonicalSignalViews,
    *,
    pulse: PulseResult | None = None,
) -> dict[str, Any]:
    """统一 direct-only 形态与双波长入口 / Unified direct-only feature entry.

    中文：route guard 在任何形态函数前执行；non-identity `x_ar` 会立即失败，
    从而不能误取 `x_filter` 值。English: The route guard executes before any
    waveform feature function, preventing accidental copying from ``x_filter``.
    """

    require_direct_route(views.route)
    views.validate()
    detected = pulse if pulse is not None else detect_pulses(views)
    return {
        "morphology": extract_morphology(
            views.x_filter, detected, route=views.route, fs_hz=CANONICAL_FS_HZ
        ),
        "optical": extract_dual_optical(
            views.x_native,
            views.x_filter,
            detected,
            route=views.route,
            fs_hz=CANONICAL_FS_HZ,
        ),
    }


__all__ = [
    "CANONICAL_FS_HZ",
    "CanonicalSignalViews",
    "WindowPlan",
    "WindowSlice",
    "extract_window",
    "ABLATION_PPG_FILTER_PROFILE_ID",
    "REFERENCE_PPG_FILTER_PROFILE_ID",
    "InputQC",
    "PpgFilterProfile",
    "get_ppg_filter_profile",
    "inspect_and_repair",
    "preprocess_ppg_pair",
    "build_signal_views",
    "ImuPreprocessResult",
    "EskfConfiguration",
    "GravityComparisonResult",
    "ImuProfile",
    "CausalImuProcessor",
    "preprocess_imu",
    "compare_ekf_lpf_gravity",
    "estimate_gravity_no_precalibration_ekf",
    "estimate_gravity_lpf",
    "detect_pulses",
    "MIN_BASIC_RATE_PEAKS",
    "PrvResult",
    "compute_prv",
    "MIN_TIME_DOMAIN_PRV_INTERVALS",
    "DlResampleResult",
    "SynchronizedResampleResult",
    "V2_DL_RESAMPLING_TARGETS_HZ",
    "resample_dl_view",
    "resample_synchronized_channels",
    "validate_dl_resampling_config",
    "MorphologyResult",
    "CALIBRATED_ROLL_PITCH_EKF_PROFILE_ID",
    "MOTION_IMU_CHANNEL_SCHEMA",
    "MOTION_IMU_CHANNEL_UNITS",
    "FORMAL_STATIC_CALIBRATION_ROLES",
    "PROFILE_A_LPF_ID",
    "PTT_STATIC_CALIBRATION_ROLE",
    "MotionImuCalibration",
    "MotionImuResult",
    "RollPitchEkfConfig",
    "fit_motion_imu_calibration",
    "preprocess_motion_imu_calibrated_ekf",
    "preprocess_motion_imu_lpf_ablation",
    "extract_morphology",
    "OpticalFeatureResult",
    "extract_dual_optical",
    "SqiConfig",
    "SqiCalibrator",
    "SqiDiagnosticComponent",
    "SqiDiagnosticConfig",
    "SqiDiagnostics",
    "fit_sqi_calibrator",
    "quality_component_scores",
    "evaluate_quality",
    "evaluate_quality_diagnostics",
    "extract_direct_features",
]
