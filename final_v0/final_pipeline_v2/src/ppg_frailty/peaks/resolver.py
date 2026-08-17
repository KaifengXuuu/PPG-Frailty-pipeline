"""Fail-closed public resolver for registered pulse detectors."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import PulseResult, SignalRoute
from ..signal.views import CANONICAL_FS_HZ, CanonicalSignalViews
from .aboy_project import (
    DETECTOR_ID as CANONICAL_DETECTOR_ID,
    detect_pulses_per_wavelength_aboy_project,
)

ABLATION_DETECTOR_ID = "dual_polarity_prominence_v1_ablation"

REGISTERED_DETECTOR_IDS = (
    CANONICAL_DETECTOR_ID,
    ABLATION_DETECTOR_ID,
)


def resolve_detector_id(detector_id: str) -> str:
    """Require one exact persisted ID; aliases and omission are forbidden."""

    requested = str(detector_id).strip()
    if requested not in REGISTERED_DETECTOR_IDS:
        raise ValueError(
            f"pulse detector ID is not registered: {detector_id!r}; "
            f"expected one of {REGISTERED_DETECTOR_IDS}"
        )
    return requested


def _available_labels(values: np.ndarray | CanonicalSignalViews) -> tuple[str, ...]:
    matrix = (
        np.asarray(values.analysis_signal)
        if isinstance(values, CanonicalSignalViews)
        else np.asarray(values)
    )
    channel_count = 1 if matrix.ndim == 1 else (
        int(matrix.shape[1]) if matrix.ndim == 2 else 0
    )
    if channel_count not in (1, 2):
        raise ValueError("pulse input must be samples-by-one/two channels")
    return ("RED", "IR")[:channel_count]


def detect_pulses_per_wavelength(
    values: np.ndarray | CanonicalSignalViews,
    *,
    detector_id: str,
    fs_hz: float = CANONICAL_FS_HZ,
    min_observation_sec: float = 8.0,
    min_peaks: int = 5,
    source_route: SignalRoute | str | None = None,
    run_id: str | None = None,
) -> dict[str, PulseResult]:
    """Detect RED and IR independently through one explicitly resolved detector."""

    resolved = resolve_detector_id(detector_id)
    if resolved == CANONICAL_DETECTOR_ID:
        return detect_pulses_per_wavelength_aboy_project(
            values,
            fs_hz=fs_hz,
            min_observation_sec=min_observation_sec,
            min_peaks=min_peaks,
            source_route=source_route,
            run_id=run_id,
        )
    from ..signal.peaks import _detect_pulses_dual_polarity_ablation

    results: dict[str, PulseResult] = {}
    for label in _available_labels(values):
        results[label] = _detect_pulses_dual_polarity_ablation(
            values,
            fs_hz=fs_hz,
            wavelength=label,
            min_observation_sec=min_observation_sec,
            min_peaks=min_peaks,
            source_route=source_route,
            run_id=run_id,
        )
    return results


def detect_pulses(
    values: np.ndarray | CanonicalSignalViews,
    *,
    detector_id: str,
    fs_hz: float = CANONICAL_FS_HZ,
    wavelength: str = "auto",
    min_observation_sec: float = 8.0,
    min_peaks: int = 5,
    source_route: SignalRoute | str | None = None,
    run_id: str | None = None,
) -> PulseResult:
    """Return one result; canonical AUTO uses score/coverage/RED tie-break."""

    requested_wavelength = str(wavelength).strip().upper()
    labels = _available_labels(values)
    if requested_wavelength != "AUTO" and requested_wavelength not in labels:
        raise ValueError(f"wavelength must be auto or one of {labels}")
    resolved = resolve_detector_id(detector_id)
    if resolved == ABLATION_DETECTOR_ID:
        from ..signal.peaks import _detect_pulses_dual_polarity_ablation

        return _detect_pulses_dual_polarity_ablation(
            values,
            fs_hz=fs_hz,
            wavelength=requested_wavelength,
            min_observation_sec=min_observation_sec,
            min_peaks=min_peaks,
            source_route=source_route,
            run_id=run_id,
        )
    results = detect_pulses_per_wavelength(
        values,
        detector_id=resolved,
        fs_hz=fs_hz,
        min_observation_sec=min_observation_sec,
        min_peaks=min_peaks,
        source_route=source_route,
        run_id=run_id,
    )
    if requested_wavelength != "AUTO":
        return results[requested_wavelength]
    return max(
        results.values(),
        key=lambda row: (
            float(row.detector_score),
            float(row.detector_coverage),
            row.wavelength == "RED",
        ),
    )


__all__ = [
    "ABLATION_DETECTOR_ID",
    "CANONICAL_DETECTOR_ID",
    "REGISTERED_DETECTOR_IDS",
    "detect_pulses",
    "detect_pulses_per_wavelength",
    "resolve_detector_id",
]
