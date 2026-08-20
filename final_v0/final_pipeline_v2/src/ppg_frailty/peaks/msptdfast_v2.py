"""Registered MSPTDfast v2 peak-detector ablation.

The numerical detector below is the single equation-level Python port used by
Stage-ablation-01 and by the ordinary V2 peak-detector resolver.  It is bound
to the reviewed ppg-beats v2.3 MATLAB source, but does not claim bitwise MATLAB
parity.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np
from scipy import signal

from ..contracts import PulseResult, SignalRoute
from ..signal.views import CANONICAL_FS_HZ, CanonicalSignalViews


DETECTOR_ID = "msptdfast_v2_3_python_port"
DETECTOR_VERSION = "msptdfast_v2_equation_level_python_port_v2_native_direct_input"
DETECTOR_NAME = "MSPTDfast v2 equation-level Python port"
IMPLEMENTATION_PATH = "ppg_frailty.peaks.msptdfast_v2.detect_msptdfast_v2"
AUTHOR_SOURCE_SHA256 = (
    "39f5010f1d485f2dc180bffbbe662dc9bd16e8116bd1ae630761f1e2b58bcabd"
)
DEFAULT_PARAMETERS: dict[str, float] = {
    "target_downsample_hz": 20.0,
    "minimum_heart_rate_bpm": 30.0,
    "window_s": 6.0,
    "overlap_fraction": 0.2,
}


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(
            f"{name} must be finite" + (" and positive" if positive else "")
        )
    return result


def resolve_parameters(value: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Materialize the complete executable MSPTDfast parameter mapping."""

    declared = {} if value is None else dict(value)
    unknown = sorted(set(declared) - set(DEFAULT_PARAMETERS))
    if unknown:
        raise ValueError(f"MSPTDfast parameters contain unknown keys: {unknown}")
    resolved = {**DEFAULT_PARAMETERS, **declared}
    for key in ("target_downsample_hz", "minimum_heart_rate_bpm", "window_s"):
        resolved[key] = _finite(resolved[key], f"MSPTDfast.{key}", positive=True)
    resolved["overlap_fraction"] = _finite(
        resolved["overlap_fraction"], "MSPTDfast.overlap_fraction"
    )
    if not 0.0 <= resolved["overlap_fraction"] < 1.0:
        raise ValueError("MSPTDfast.overlap_fraction must be in [0, 1)")
    return resolved


def _window_starts(length: int, window: int, hop: int) -> tuple[int, ...]:
    if length <= 0 or window <= 0 or hop <= 0:
        raise ValueError("window inputs must be positive")
    if length <= window:
        return (0,)
    starts = list(range(0, length - window + 1, hop))
    right = length - window
    if starts[-1] != right:
        starts.append(right)
    return tuple(starts)


def _msptd_window_peaks(
    values: np.ndarray,
    fs_hz: float,
    *,
    minimum_heart_rate_bpm: float,
) -> np.ndarray:
    """Equation-level Python port of the author's MSPTD core for one window."""

    x = signal.detrend(np.asarray(values, dtype=np.float64), type="linear")
    n = x.size
    if n < 5:
        return np.empty(0, dtype=np.int64)
    largest_scale = int(math.ceil(n / 2.0) - 1)
    duration = n / float(fs_hz)
    scales = np.arange(1, largest_scale + 1, dtype=np.int64)
    scale_frequencies = (largest_scale / scales) / duration
    included = scales[scale_frequencies >= minimum_heart_rate_bpm / 60.0]
    if included.size == 0:
        return np.empty(0, dtype=np.int64)
    max_scale = int(included[-1])
    local_max = np.zeros((max_scale, n), dtype=bool)
    for scale_index, k in enumerate(range(1, max_scale + 1)):
        centers = np.arange(k, n - k)
        local_max[scale_index, centers] = (
            (x[centers] > x[centers - k]) & (x[centers] > x[centers + k])
        )
    lambda_max = int(np.argmax(np.sum(local_max, axis=1))) + 1
    return np.flatnonzero(np.all(local_max[:lambda_max], axis=0)).astype(np.int64)


def detect_msptdfast_v2(
    values: np.ndarray,
    fs_hz: float,
    *,
    target_downsample_hz: float = 20.0,
    minimum_heart_rate_bpm: float = 30.0,
    window_s: float = 6.0,
    overlap_fraction: float = 0.2,
) -> np.ndarray:
    """Detect peaks with the Stage-ablation-01 MSPTDfast v2 port unchanged."""

    source = np.asarray(values, dtype=np.float64).reshape(-1)
    if source.size < 3 or not np.isfinite(source).all() or fs_hz <= 0.0:
        raise ValueError("MSPTDfast requires one finite non-empty PPG signal and fs")
    parameters = resolve_parameters(
        {
            "target_downsample_hz": target_downsample_hz,
            "minimum_heart_rate_bpm": minimum_heart_rate_bpm,
            "window_s": window_s,
            "overlap_fraction": overlap_fraction,
        }
    )
    nominal = int(round(parameters["window_s"] * fs_hz))
    hop = int(round(nominal * (1.0 - parameters["overlap_fraction"])))
    # MATLAB reference uses inclusive win_ends=start+6*fs.
    window = nominal + 1
    detected: list[int] = []
    factor = max(1, int(math.floor(fs_hz / parameters["target_downsample_hz"])))
    relative_fs = fs_hz / factor
    tolerance_s = (
        0.2 if relative_fs < 10.0 else 0.1 if relative_fs < 20.0 else 0.05
    )
    tolerance = int(math.ceil(fs_hz * tolerance_s))
    for start in _window_starts(source.size, min(window, source.size), max(hop, 1)):
        segment = source[start : start + window]
        downsampled = segment[::factor]
        for peak in _msptd_window_peaks(
            downsampled,
            relative_fs,
            minimum_heart_rate_bpm=parameters["minimum_heart_rate_bpm"],
        ):
            # Translate the author's one-based p*factor before local refinement.
            approximate = int((int(peak) + 1) * factor - 1)
            left = max(0, approximate - tolerance)
            right = min(segment.size, approximate + tolerance + 1)
            if right > left:
                detected.append(start + left + int(np.argmax(segment[left:right])))
    return np.unique(np.asarray(detected, dtype=np.int64))


def _prepare_input(
    values: np.ndarray | CanonicalSignalViews,
    *,
    fs_hz: float,
    source_route: SignalRoute | str | None,
) -> tuple[np.ndarray, int, SignalRoute, str, str]:
    if float(fs_hz) != CANONICAL_FS_HZ:
        raise ValueError("pipeline pulse detection requires the exact 400 Hz time grid")
    if isinstance(values, CanonicalSignalViews):
        route = values.route
        if source_route is not None and SignalRoute(source_route) is not route:
            raise ValueError("declared pulse source_route disagrees with signal views")
        if route is SignalRoute.ARTIFACT_RATE_ONLY:
            matrix = np.asarray(values.analysis_signal, dtype=np.float64)
            valid = np.asarray(values.rate_valid_mask, dtype=bool)
            source_view = "artifact_rate_only"
        else:
            # The author's MSPTDfast core owns its per-window detrending and
            # therefore consumes repaired native PPG on direct/identity routes.
            matrix = np.asarray(values.x_native, dtype=np.float64)
            valid = np.ones(matrix.shape[0], dtype=bool)
            source_view = "repaired_native_ppg"
        record_id = str(values.metadata.get("record_id", "")).strip()
    else:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        valid = np.ones(matrix.shape[0], dtype=bool)
        route = SignalRoute.DIRECT if source_route is None else SignalRoute(source_route)
        record_id = ""
        source_view = "caller_declared_raw_ppg"
    if route is SignalRoute.DROPPED:
        raise ValueError("pulse detection cannot run on a dropped route")
    if matrix.ndim != 2 or matrix.shape[1] not in (1, 2) or not np.isfinite(matrix).all():
        raise ValueError("pulse input must be finite samples-by-one/two channels")
    if valid.shape != (matrix.shape[0],):
        raise ValueError("rate_valid_mask must align with pulse samples")
    offset = 0
    if not np.all(valid):
        padded = np.concatenate(([False], valid, [False]))
        changes = np.diff(padded.astype(np.int8))
        runs = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
        if not runs:
            raise ValueError("rate waveform has no artifact-valid samples")
        start, stop = max(runs, key=lambda bounds: bounds[1] - bounds[0])
        matrix = matrix[start:stop]
        offset = int(start)
    return matrix, offset, route, record_id, source_view


def detect_pulses_per_wavelength_msptdfast_v2(
    values: np.ndarray | CanonicalSignalViews,
    *,
    fs_hz: float = CANONICAL_FS_HZ,
    min_observation_sec: float = 8.0,
    min_peaks: int = 5,
    source_route: SignalRoute | str | None = None,
    run_id: str | None = None,
    detector_parameters: Mapping[str, Any] | None = None,
) -> dict[str, PulseResult]:
    """Adapt exact MSPTDfast peak indices to the shared V2 PulseResult contract."""

    from .resolver import validate_peak_detection_parameters

    min_observation_sec, min_peaks = validate_peak_detection_parameters(
        min_observation_sec, min_peaks
    )
    parameters = resolve_parameters(detector_parameters)
    matrix, offset, route, record_id, source_view = _prepare_input(
        values, fs_hz=fs_hz, source_route=source_route
    )
    if matrix.shape[0] / fs_hz < min_observation_sec:
        raise ValueError(
            f"HR/PPI requires at least {min_observation_sec:g} seconds of observation"
        )
    labels = ("RED", "IR")[: matrix.shape[1]]
    output: dict[str, PulseResult] = {}
    for channel, label in enumerate(labels):
        local_peaks = detect_msptdfast_v2(matrix[:, channel], fs_hz, **parameters)
        peaks = local_peaks + offset
        if peaks.size < min_peaks:
            raise ValueError(
                f"{DETECTOR_ID} requires at least {min_peaks} detected peaks for {label}"
            )
        timestamps = peaks.astype(np.float64) / fs_hz
        ppi = np.diff(timestamps)
        valid = np.isfinite(ppi) & (ppi >= 60.0 / 210.0) & (ppi <= 60.0 / 35.0)
        accepted = np.zeros(peaks.size, dtype=bool)
        if valid.size:
            accepted[:-1] |= valid
            accepted[1:] |= valid
        starts = np.arange(ppi.size, dtype=np.int64)
        parameter_payload = {
            "algorithm": DETECTOR_VERSION,
            "author_source_sha256": AUTHOR_SOURCE_SHA256,
            "parameters": parameters,
            "valid_run_offset": offset,
            "source_view": source_view,
            "wavelength": label,
            "min_observation_sec": min_observation_sec,
            "min_peaks": min_peaks,
        }
        provenance = (parameter_payload,)
        provenance_hash = hashlib.sha256(
            json.dumps(
                provenance, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
        if run_id is not None:
            resolved_run_id = str(run_id).strip()
        elif record_id:
            resolved_run_id = (
                f"{record_id}::{route.value}::{DETECTOR_ID}::{label}::"
                f"{provenance_hash[:16]}"
            )
        else:
            digest = hashlib.sha256(
                np.ascontiguousarray(matrix[:, channel], dtype="<f8").tobytes()
            ).hexdigest()[:20]
            resolved_run_id = f"array::{route.value}::{DETECTOR_ID}::{label}::{digest}"
        if not resolved_run_id:
            raise ValueError("pulse detection run_id cannot be empty")
        span = float(timestamps[-1] - timestamps[0]) if timestamps.size >= 2 else 0.0
        coverage = float(np.clip(np.sum(ppi[valid]) / span, 0.0, 1.0)) if span else 0.0
        result = PulseResult(
            peaks=peaks,
            peak_timestamps_s=timestamps,
            accepted_peak_mask=accepted,
            interval_start_peak_indices=starts,
            interval_stop_peak_indices=starts + 1,
            ppi_s=ppi,
            valid_interval_mask=valid,
            adjacency_mask=np.ones(ppi.size, dtype=bool),
            wavelength=label,
            detector_version=DETECTOR_VERSION,
            confidence=np.ones(peaks.size, dtype=np.float64),
            source_route=route,
            detection_run_id=resolved_run_id,
            interval_run_ids=np.full(
                ppi.shape,
                resolved_run_id,
                dtype=f"<U{max(1, len(resolved_run_id))}",
            ),
            detector_id=DETECTOR_ID,
            selected_polarity=1,
            block_hri_provenance_hash=provenance_hash,
            block_provenance=provenance,
            interval_rejection_reasons=tuple(
                "" if item else "outside_project_35_210_bpm" for item in valid.tolist()
            ),
            peak_ordinals=np.arange(peaks.size, dtype=np.int64),
            detector_score=float(np.count_nonzero(valid) + 0.5 * coverage),
            detector_coverage=coverage,
        )
        result.validate_identity()
        output[label] = result
    return output


__all__ = [
    "AUTHOR_SOURCE_SHA256",
    "DEFAULT_PARAMETERS",
    "DETECTOR_ID",
    "DETECTOR_NAME",
    "DETECTOR_VERSION",
    "IMPLEMENTATION_PATH",
    "detect_msptdfast_v2",
    "detect_pulses_per_wavelength_msptdfast_v2",
    "resolve_parameters",
]
