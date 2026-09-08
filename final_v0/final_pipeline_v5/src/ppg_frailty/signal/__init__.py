"""V1 信号层公共 facade / Public facade for the V1 signal layer."""

from __future__ import annotations

from typing import Any, Mapping

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
    MOTION_IMU_CALIBRATION_SCHEMA,
    FORMAL_STATIC_CALIBRATION_ROLES,
    PROFILE_A_LPF_ID,
    PTT_STATIC_CALIBRATION_ROLE,
    MotionImuCalibration,
    MotionImuResult,
    RollPitchEkfConfig,
    fit_motion_imu_calibration,
    preprocess_motion_imu_calibrated_ekf,
    preprocess_motion_imu_profile_a_lpf,
    preprocess_motion_imu_lpf_ablation,
)
from .optical import (
    OPTICAL_SCHEMA_VERSION,
    OpticalBeatAudit,
    OpticalFeatureResult,
    extract_dual_optical,
)
from .peaks import MIN_BASIC_RATE_PEAKS
from .preprocess import (
    ABLATION_PPG_FILTER_PROFILE_ID,
    REFERENCE_PPG_FILTER_PROFILE_ID,
    InputQC,
    PpgFilterProfile,
    build_signal_views,
    get_ppg_filter_profile,
    inspect_and_repair,
    preprocess_ppg_pair,
    roll_pitch_ekf_config_from_resolved,
)
from .prv import MIN_TIME_DOMAIN_PRV_INTERVALS, PrvConfig, PrvResult, compute_prv
from .resample import (
    DlResampleResult,
    SynchronizedResampleResult,
    V2_DL_RESAMPLING_TARGETS_HZ,
    prepare_configured_dl_input,
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


CANONICAL_DETECTOR_ID = "msptdfast_v2_3_python_port"

def detect_pulses(*args: Any, detector_id: str, **kwargs: Any) -> PulseResult:
    """Lazy compatibility facade that still requires one explicit detector ID."""

    from ..peaks.resolver import detect_pulses as implementation

    return implementation(*args, detector_id=detector_id, **kwargs)

def detect_pulses_per_wavelength(
    *args: Any,
    detector_id: str,
    **kwargs: Any,
) -> dict[str, PulseResult]:
    """Lazy per-wavelength facade without introducing a package import cycle."""

    from ..peaks.resolver import detect_pulses_per_wavelength as implementation

    return implementation(*args, detector_id=detector_id, **kwargs)

def extract_direct_features(
    views: CanonicalSignalViews,
    *,
    pulse: PulseResult | None = None,
    detector_id: str | None = None,
    min_observation_sec: float = 8.0,
    min_peaks: int = 5,
    detector_parameters: Mapping[str, Any] | None = None,
    pulses_per_wavelength: Mapping[str, PulseResult] | None = None,
) -> dict[str, Any]:
    """统一 direct-only 形态与双波长入口 / Unified direct-only feature entry.

    中文：route guard 在任何形态函数前执行；non-identity `x_ar` 会立即失败，
    从而不能误取 `x_filter` 值。English: The route guard executes before any
    waveform feature function, preventing accidental copying from ``x_filter``.
    """

    require_direct_route(views.route)
    views.validate()
    if pulses_per_wavelength is None and detector_id is None:
        raise ValueError("extract_direct_features requires independent RED/IR pulses " "or a persisted detector_id")
    dual_pulses = (
        dict(pulses_per_wavelength)
        if pulses_per_wavelength is not None
        else detect_pulses_per_wavelength(
            views,
            detector_id=str(detector_id),
            min_observation_sec=min_observation_sec,
            min_peaks=min_peaks,
            detector_parameters=detector_parameters,
        )
    )
    if detector_id is not None and any(result.detector_id != detector_id for result in dual_pulses.values()):
        raise ValueError("provided RED/IR pulses disagree with detector_id")
    from ..peaks.pairing import select_reference_wavelength

    detected = pulse if pulse is not None else dual_pulses[select_reference_wavelength(dual_pulses)]
    if any(result.detector_id != detected.detector_id for result in dual_pulses.values()):
        raise ValueError("morphology pulse and RED/IR pulses use different detectors")
    return {
        "morphology": extract_morphology(views.x_filter, detected, route=views.route, fs_hz=CANONICAL_FS_HZ),
        "optical": extract_dual_optical(
            views.x_native,
            views.x_filter,
            dual_pulses,
            route=views.route,
            fs_hz=CANONICAL_FS_HZ,
        ),
    }


__all__ = [
    "CANONICAL_FS_HZ",
    "CanonicalSignalViews",
    "SignalRoute",
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
    "roll_pitch_ekf_config_from_resolved",
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
    "detect_pulses_per_wavelength",
    "CANONICAL_DETECTOR_ID",
    "MIN_BASIC_RATE_PEAKS",
    "PrvConfig",
    "PrvResult",
    "compute_prv",
    "MIN_TIME_DOMAIN_PRV_INTERVALS",
    "DlResampleResult",
    "SynchronizedResampleResult",
    "V2_DL_RESAMPLING_TARGETS_HZ",
    "prepare_configured_dl_input",
    "resample_dl_view",
    "resample_synchronized_channels",
    "validate_dl_resampling_config",
    "MorphologyResult",
    "CALIBRATED_ROLL_PITCH_EKF_PROFILE_ID",
    "MOTION_IMU_CHANNEL_SCHEMA",
    "MOTION_IMU_CHANNEL_UNITS",
    "MOTION_IMU_CALIBRATION_SCHEMA",
    "FORMAL_STATIC_CALIBRATION_ROLES",
    "PROFILE_A_LPF_ID",
    "PTT_STATIC_CALIBRATION_ROLE",
    "MotionImuCalibration",
    "MotionImuResult",
    "RollPitchEkfConfig",
    "fit_motion_imu_calibration",
    "preprocess_motion_imu_calibrated_ekf",
    "preprocess_motion_imu_profile_a_lpf",
    "preprocess_motion_imu_lpf_ablation",
    "extract_morphology",
    "OPTICAL_SCHEMA_VERSION",
    "OpticalBeatAudit",
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
