"""V1 信号层公共 facade / Public facade for the V1 signal layer."""

from __future__ import annotations

from typing import Any

from ..contracts import PulseResult, SignalRoute
from .imu import (
    CausalImuProcessor,
    EskfConfiguration,
    ImuProfile,
    ImuPreprocessResult,
    estimate_gravity_lpf,
    estimate_gravity_no_precalibration_ekf,
    preprocess_imu,
)
from .morphology import MorphologyResult, extract_morphology, require_direct_route
from .optical import OpticalFeatureResult, extract_dual_optical
from .peaks import detect_pulses
from .preprocess import InputQC, build_signal_views, inspect_and_repair, preprocess_ppg_pair
from .prv import PrvResult, compute_prv
from .sqi import (
    SqiCalibrator,
    SqiConfig,
    evaluate_quality,
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
    "InputQC",
    "inspect_and_repair",
    "preprocess_ppg_pair",
    "build_signal_views",
    "ImuPreprocessResult",
    "EskfConfiguration",
    "ImuProfile",
    "CausalImuProcessor",
    "preprocess_imu",
    "estimate_gravity_no_precalibration_ekf",
    "estimate_gravity_lpf",
    "detect_pulses",
    "PrvResult",
    "compute_prv",
    "MorphologyResult",
    "extract_morphology",
    "OpticalFeatureResult",
    "extract_dual_optical",
    "SqiConfig",
    "SqiCalibrator",
    "fit_sqi_calibrator",
    "quality_component_scores",
    "evaluate_quality",
    "extract_direct_features",
]
