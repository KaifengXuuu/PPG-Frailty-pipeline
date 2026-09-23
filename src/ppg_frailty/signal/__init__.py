"""Signal implementations plus the route-safe direct-feature composition."""

from __future__ import annotations

from typing import Any, Mapping

from ..contracts import PulseResult
from .morphology import extract_morphology, require_direct_route
from .optical import extract_dual_optical
from .views import CANONICAL_FS_HZ, CanonicalSignalViews


CANONICAL_DETECTOR_ID = "msptdfast_v2_3_python_port"


def detect_pulses(*args: Any, detector_id: str, **kwargs: Any) -> PulseResult:
    """Dispatch one explicitly selected pulse detector without an import cycle."""

    from ..peaks.resolver import detect_pulses as implementation

    return implementation(*args, detector_id=detector_id, **kwargs)


def detect_pulses_per_wavelength(
    *args: Any,
    detector_id: str,
    **kwargs: Any,
) -> dict[str, PulseResult]:
    """Dispatch the selected detector independently for each wavelength."""

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
    """Extract route-safe morphology and dual-wavelength features."""

    require_direct_route(views.route)
    views.validate()
    if pulses_per_wavelength is None and detector_id is None:
        raise ValueError("extract_direct_features requires independent RED/IR pulses or a persisted detector_id")
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
    "CANONICAL_DETECTOR_ID",
    "detect_pulses",
    "detect_pulses_per_wavelength",
    "extract_direct_features",
]
