"""Canonical project Aboy++-inspired pulse detector.

This is a project-specific, thesis-aligned implementation. It does not claim
bit-for-bit parity with the published upstream Aboy++ implementation. The
detector works on a private copy of the legal 400 Hz analysis view; morphology
and optical feature extraction continue to receive the unchanged analysis view.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np
from scipy import signal

from ..contracts import PulseResult, SignalRoute
from ..signal.views import CANONICAL_FS_HZ, CanonicalSignalViews


DETECTOR_ID = "aboy_project_v1"
DETECTOR_VERSION = "project_aboy_inspired_block_adaptive_v1"
DETECTOR_NAME = "project Aboy++-inspired detector"
IMPLEMENTATION_PATH = (
    "ppg_frailty.peaks.aboy_project.detect_pulses_per_wavelength_aboy_project"
)
BLOCK_SECONDS = 10.0
INITIAL_HRI = 0.0
INITIAL_HRI_RULE = "historical_project_fixed_zero_no_label_tuning"
MIN_BPM = 35.0
MAX_BPM = 210.0
MIN_BASIC_RATE_PEAKS = 5


@dataclass(frozen=True)
class _BlockCandidate:
    """One polarity candidate from one complete detector block."""

    block_index: int
    block_start_sample: int
    polarity: int
    peaks: np.ndarray
    prominence: np.ndarray
    hri_in: float
    hri_out: float
    provenance: dict[str, Any]


@dataclass(frozen=True)
class _PolarityCandidate:
    """Chronological block merge for one wavelength and polarity."""

    polarity: int
    peaks: np.ndarray
    prominence: np.ndarray
    score: float
    n_clean_ppi: int
    coverage: float
    cv: float
    block_rows: tuple[dict[str, Any], ...]


def _provenance_hash(rows: tuple[dict[str, Any], ...]) -> str:
    """Hash JSON-safe detector decisions with stable ordering."""

    encoded = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _block_parameters(
    hri_in: float,
    hri_out: float,
    fs_hz: float,
) -> dict[str, float | int]:
    """Return the thesis equations without hidden rounding conventions."""

    d_210 = max(1, int(round(float(fs_hz) * 60.0 / MAX_BPM)))
    high_hz = min(8.0, max(1.5, 3.0 * (1.0 + float(hri_in))))
    hrwin_samples = float(fs_hz) / (3.0 * (1.0 + float(hri_out)))
    final_distance = max(int(round(2.0 * hrwin_samples)), d_210)
    return {
        "d_210_samples": d_210,
        "f_high_hz": float(high_hz),
        "hrwin_samples": hrwin_samples,
        "final_distance_samples": final_distance,
    }


def _bandpass_block(
    values: np.ndarray,
    *,
    fs_hz: float,
    high_hz: float,
) -> np.ndarray:
    """Apply the detector-only second-order 0.5--adaptive-Hz band-pass."""

    sos = signal.butter(
        2,
        [0.5, float(high_hz)],
        btype="bandpass",
        fs=float(fs_hz),
        output="sos",
    )
    return np.asarray(signal.sosfiltfilt(sos, values), dtype=np.float64)


def _upper_30_percent_mean(values: np.ndarray) -> float:
    """Mean the upper 30% using the historical sorted-index convention."""

    x = np.sort(np.asarray(values, dtype=np.float64))
    if not x.size:
        raise ValueError("upper-30-percent mean requires at least one value")
    start = int(np.floor(0.70 * x.size)) if x.size >= 3 else 0
    return float(np.mean(x[start:]))


def _score_peak_train(
    peaks: np.ndarray,
    *,
    fs_hz: float,
) -> tuple[float, int, float, float]:
    """Score a train as N_clean + 0.5*coverage - min(CV, 2)."""

    _accepted, valid, _reasons, _reference = _clean_intervals(
        peaks,
        fs_hz=fs_hz,
    )
    ppi = np.diff(np.asarray(peaks, dtype=np.int64)) / float(fs_hz)
    selected = ppi[valid]
    n_clean = int(selected.size)
    span = (
        (int(peaks[-1]) - int(peaks[0])) / float(fs_hz)
        if np.asarray(peaks).size >= 2
        else 0.0
    )
    coverage = float(np.sum(selected) / span) if span > 0.0 else 0.0
    coverage = float(np.clip(coverage, 0.0, 1.0))
    if selected.size >= 2 and float(np.mean(selected)) > 0.0:
        cv = float(np.std(selected, ddof=0) / np.mean(selected))
    else:
        cv = 2.0
    score = float(n_clean + 0.5 * coverage - min(cv, 2.0))
    return score, n_clean, coverage, cv


def _clean_intervals(
    peaks: np.ndarray,
    *,
    fs_hz: float,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], float | None]:
    """Clean PPIs without deleting peaks or compressing the original timeline.

    Apply inclusive 35--210 bpm bounds, then MAD cleaning, then define the
    ratio reference as the cleaned median. A ratio failure rejects the
    following peak; both adjacent intervals remain present but invalid.
    """

    peak_array = np.asarray(peaks, dtype=np.int64)
    if peak_array.size < 2:
        return (
            np.ones(peak_array.size, dtype=bool),
            np.empty(0, dtype=bool),
            (),
            None,
        )
    ppi = np.diff(peak_array).astype(np.float64) / float(fs_hz)
    lower = 60.0 / MAX_BPM
    upper = 60.0 / MIN_BPM
    physiological = np.isfinite(ppi) & (ppi >= lower) & (ppi <= upper)
    mad_valid = np.ones(ppi.size, dtype=bool)
    if int(np.count_nonzero(physiological)) >= 5:
        selected = ppi[physiological]
        median = float(np.median(selected))
        mad = float(np.median(np.abs(selected - median)))
        robust_sigma = 1.4826 * mad
        if robust_sigma > 0.0:
            mad_valid[physiological] = (
                np.abs(ppi[physiological] - median)
                <= 4.0 * robust_sigma
            )

    cleaned = physiological & mad_valid
    reference = (
        float(np.median(ppi[cleaned]))
        if np.any(cleaned)
        else None
    )
    ratio = np.ones(ppi.size, dtype=bool)
    if reference is not None:
        ratio = (ppi >= 0.5 * reference) & (ppi <= 1.8 * reference)

    accepted_peaks = np.ones(peak_array.size, dtype=bool)
    ratio_rejected_stops = np.flatnonzero(~ratio) + 1
    accepted_peaks[ratio_rejected_stops] = False
    endpoints_accepted = accepted_peaks[:-1] & accepted_peaks[1:]
    valid = physiological & ratio & mad_valid & endpoints_accepted

    reasons: list[str] = []
    for index in range(ppi.size):
        row: list[str] = []
        if not physiological[index]:
            row.append("outside_35_210_bpm")
        if not ratio[index]:
            row.append("outside_0p5_to_1p8_reference_ppi")
        if not mad_valid[index]:
            row.append("outside_4x_1p4826_mad")
        if not endpoints_accepted[index]:
            row.append("rejected_peak_endpoint")
        reasons.append(";".join(row))
    return accepted_peaks, valid, tuple(reasons), reference


def _block_candidate(
    block: np.ndarray,
    *,
    block_index: int,
    block_start_sample: int,
    polarity: int,
    hri_in: float,
    fs_hz: float,
) -> _BlockCandidate:
    """Run one complete block at one polarity and persist adaptive values."""

    initial = _block_parameters(hri_in, hri_in, fs_hz)
    filtered = _bandpass_block(
        np.asarray(block, dtype=np.float64) * float(polarity),
        fs_hz=fs_hz,
        high_hz=float(initial["f_high_hz"]),
    )
    preliminary, _ = signal.find_peaks(
        filtered,
        distance=int(initial["d_210_samples"]),
    )
    preliminary = np.asarray(preliminary, dtype=np.int64)
    preliminary_ppi = np.diff(preliminary).astype(np.float64) / float(fs_hz)
    retained = np.empty(0, dtype=np.float64)
    hri_out = float(hri_in)
    update_reason = "insufficient_retained_ppi"
    if preliminary_ppi.size:
        threshold = float(np.percentile(preliminary_ppi, 30.0))
        retained = preliminary_ppi[preliminary_ppi > threshold]
        if retained.size >= 2:
            retained_mean = float(np.mean(retained))
            preliminary_median = float(np.median(preliminary_ppi))
            if (
                0.5 * preliminary_median
                <= retained_mean
                <= 1.5 * preliminary_median
                and retained_mean > 0.0
            ):
                hri_out = float(
                    10.0 * np.std(retained, ddof=0) / retained_mean
                )
                update_reason = "updated_population_sd_ddof0"
            else:
                update_reason = (
                    "retained_mean_outside_0p5_to_1p5_preliminary_median"
                )

    parameters = _block_parameters(hri_in, hri_out, fs_hz)
    if preliminary.size:
        upper_mean = _upper_30_percent_mean(filtered[preliminary])
    else:
        upper_mean = float("-inf")
    block_sd = float(np.std(filtered, ddof=0))
    prominence_threshold = 0.25 * max(upper_mean, block_sd)
    prominence_threshold = max(float(prominence_threshold), 1e-12)
    peaks, properties = signal.find_peaks(
        filtered,
        distance=int(parameters["final_distance_samples"]),
        prominence=prominence_threshold,
    )
    peaks = np.asarray(peaks, dtype=np.int64)
    prominences = np.asarray(
        properties.get("prominences", np.zeros(peaks.size)),
        dtype=np.float64,
    )
    block_score, n_clean, coverage, cv = _score_peak_train(
        peaks,
        fs_hz=fs_hz,
    )
    provenance: dict[str, Any] = {
        "block_index": int(block_index),
        "block_start_sample": int(block_start_sample),
        "block_stop_sample": int(block_start_sample + block.size),
        "block_complete": True,
        "polarity": int(polarity),
        "initial_hri": float(INITIAL_HRI),
        "initial_hri_rule": INITIAL_HRI_RULE,
        "hri_in": float(hri_in),
        "hri_out": float(hri_out),
        "hri_update_reason": update_reason,
        "hri_sd_ddof": 0,
        "preliminary_peak_count": int(preliminary.size),
        "preliminary_ppi_30th_percentile_s": (
            float(threshold) if preliminary_ppi.size else None
        ),
        "preliminary_ppi_median_s": (
            float(np.median(preliminary_ppi))
            if preliminary_ppi.size
            else None
        ),
        "retained_pd_count": int(retained.size),
        "d_210_samples": int(parameters["d_210_samples"]),
        "f_high_hz": float(parameters["f_high_hz"]),
        "hrwin_samples": float(parameters["hrwin_samples"]),
        "final_distance_samples": int(parameters["final_distance_samples"]),
        "upper_30_percent_preliminary_amplitude_mean": (
            None if not np.isfinite(upper_mean) else float(upper_mean)
        ),
        "block_population_sd": block_sd,
        "final_prominence": float(prominence_threshold),
        "final_peak_count": int(peaks.size),
        "n_clean_ppi": int(n_clean),
        "coverage": float(coverage),
        "cv": float(cv),
        "score": float(block_score),
    }
    return _BlockCandidate(
        block_index=block_index,
        block_start_sample=block_start_sample,
        polarity=polarity,
        peaks=peaks + int(block_start_sample),
        prominence=prominences,
        hri_in=float(hri_in),
        hri_out=float(hri_out),
        provenance=provenance,
    )


def _merge_block_peaks(
    blocks: tuple[_BlockCandidate, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Chronologically merge blocks with historical exact-sample uniqueness."""

    if not blocks:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    peak_parts = [np.asarray(row.peaks, dtype=np.int64) for row in blocks]
    prominence_parts = [
        np.asarray(row.prominence, dtype=np.float64) for row in blocks
    ]
    for peaks, prominence in zip(peak_parts, prominence_parts):
        if peaks.shape != prominence.shape:
            raise ValueError("block peaks and prominences must align")
        if peaks.size and np.any(np.diff(peaks) <= 0):
            raise ValueError("block peaks must be strictly chronological")
    peak_parts = [part for part in peak_parts if part.size]
    prominence_parts = [part for part in prominence_parts if part.size]
    if not peak_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    peaks = np.concatenate(peak_parts).astype(np.int64, copy=False)
    prominence = np.concatenate(prominence_parts).astype(np.float64, copy=False)
    unique_peaks, first_indices = np.unique(peaks, return_index=True)
    return unique_peaks, prominence[first_indices]


def _polarity_candidate(
    values: np.ndarray,
    *,
    polarity: int,
    fs_hz: float,
    sample_offset: int,
) -> _PolarityCandidate:
    """Run complete blocks for one polarity with block-to-block HRI carry."""

    block_samples = int(round(BLOCK_SECONDS * float(fs_hz)))
    complete_blocks = int(values.size // block_samples)
    hri = float(INITIAL_HRI)
    rows: list[_BlockCandidate] = []
    for block_index in range(complete_blocks):
        local_start = block_index * block_samples
        local_stop = local_start + block_samples
        candidate = _block_candidate(
            values[local_start:local_stop],
            block_index=block_index,
            block_start_sample=sample_offset + local_start,
            polarity=polarity,
            hri_in=hri,
            fs_hz=fs_hz,
        )
        rows.append(candidate)
        hri = candidate.hri_out
    blocks = tuple(rows)
    peaks, prominence = _merge_block_peaks(blocks)
    score, n_clean, coverage, cv = _score_peak_train(peaks, fs_hz=fs_hz)
    provenance = tuple(
        {
            **row.provenance,
            "polarity_aggregate_n_clean_ppi": int(n_clean),
            "polarity_aggregate_coverage": float(coverage),
            "polarity_aggregate_cv": float(cv),
            "polarity_aggregate_score": float(score),
        }
        for row in blocks
    )
    return _PolarityCandidate(
        polarity=polarity,
        peaks=peaks,
        prominence=prominence,
        score=score,
        n_clean_ppi=n_clean,
        coverage=coverage,
        cv=cv,
        block_rows=provenance,
    )


def _prepare_input(
    values: np.ndarray | CanonicalSignalViews,
    *,
    fs_hz: float,
    source_route: SignalRoute | str | None,
) -> tuple[np.ndarray, int, SignalRoute, str]:
    """Resolve the legal analysis signal and one contiguous valid run."""

    if float(fs_hz) != CANONICAL_FS_HZ:
        raise ValueError("pulse detection requires the exact 400 Hz time grid")
    if isinstance(values, CanonicalSignalViews):
        matrix = np.asarray(values.analysis_signal, dtype=np.float64)
        valid_samples = np.asarray(values.rate_valid_mask, dtype=bool)
        route = values.route
        if source_route is not None and SignalRoute(source_route) is not route:
            raise ValueError("declared pulse source_route disagrees with signal views")
        record_id = str(values.metadata.get("record_id", "")).strip()
    else:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        valid_samples = np.ones(matrix.shape[0], dtype=bool)
        route = (
            SignalRoute.DIRECT
            if source_route is None
            else SignalRoute(source_route)
        )
        record_id = ""
    if route is SignalRoute.DROPPED:
        raise ValueError("pulse detection cannot run on a dropped route")
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if (
        matrix.ndim != 2
        or matrix.shape[1] not in (1, 2)
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("pulse input must be finite samples-by-one/two channels")
    if valid_samples.shape != (matrix.shape[0],):
        raise ValueError("rate_valid_mask must align with pulse samples")
    sample_offset = 0
    if not np.all(valid_samples):
        padded = np.concatenate(([False], valid_samples, [False]))
        changes = np.diff(padded.astype(np.int8))
        runs = list(
            zip(
                np.flatnonzero(changes == 1),
                np.flatnonzero(changes == -1),
            )
        )
        if not runs:
            raise ValueError("rate waveform has no artifact-valid samples")
        start, stop = max(runs, key=lambda bounds: bounds[1] - bounds[0])
        matrix = matrix[start:stop]
        sample_offset = int(start)
    return matrix, sample_offset, route, record_id


def _result_for_wavelength(
    values: np.ndarray,
    *,
    label: str,
    fs_hz: float,
    sample_offset: int,
    source_route: SignalRoute,
    record_id: str,
    run_id: str | None,
    min_peaks: int,
    min_observation_sec: float = 8.0,
) -> PulseResult:
    """Detect one wavelength and build the public result contract."""

    candidates = tuple(
        _polarity_candidate(
            values,
            polarity=polarity,
            fs_hz=fs_hz,
            sample_offset=sample_offset,
        )
        for polarity in (1, -1)
    )
    best = max(
        candidates,
        key=lambda item: (
            item.score,
            item.n_clean_ppi,
            item.peaks.size,
            item.polarity,
        ),
    )
    if best.peaks.size < int(min_peaks):
        raise ValueError(
            f"{DETECTOR_ID} requires at least {min_peaks} detected peaks for {label}"
        )
    all_rows = tuple(
        {
            **row,
            "wavelength": label,
            "selected_polarity": int(best.polarity),
            "polarity_selected": bool(candidate.polarity == best.polarity),
            "min_observation_sec": float(min_observation_sec),
            "min_peaks": int(min_peaks),
        }
        for candidate in candidates
        for row in candidate.block_rows
    )
    provenance_hash = _provenance_hash(all_rows)
    if run_id is None:
        if record_id:
            resolved_run_id = (
                f"{record_id}::{source_route.value}::{DETECTOR_ID}::"
                f"{label}::{provenance_hash[:16]}"
            )
        else:
            digest = hashlib.sha256(
                np.ascontiguousarray(values, dtype="<f8").tobytes(order="C")
            ).hexdigest()[:20]
            resolved_run_id = (
                f"array::{source_route.value}::{DETECTOR_ID}::{label}::{digest}"
            )
    else:
        resolved_run_id = str(run_id).strip()
    if not resolved_run_id:
        raise ValueError("pulse detection run_id cannot be empty")

    accepted_peaks, valid_intervals, rejection_reasons, _reference = (
        _clean_intervals(best.peaks, fs_hz=fs_hz)
    )
    timestamps = best.peaks.astype(np.float64) / float(fs_hz)
    starts = np.arange(max(0, best.peaks.size - 1), dtype=np.int64)
    ppi = np.diff(timestamps)
    median_prominence = max(float(np.median(best.prominence)), 1e-12)
    confidence = np.clip(
        best.prominence / (2.0 * median_prominence),
        0.0,
        1.0,
    )
    result = PulseResult(
        peaks=best.peaks,
        peak_timestamps_s=timestamps,
        accepted_peak_mask=accepted_peaks,
        interval_start_peak_indices=starts,
        interval_stop_peak_indices=starts + 1,
        ppi_s=ppi,
        valid_interval_mask=valid_intervals,
        adjacency_mask=np.ones(ppi.size, dtype=bool),
        wavelength=label,
        detector_version=DETECTOR_VERSION,
        confidence=np.asarray(confidence, dtype=np.float64),
        source_route=source_route,
        detection_run_id=resolved_run_id,
        interval_run_ids=np.full(
            ppi.shape,
            resolved_run_id,
            dtype=f"<U{max(1, len(resolved_run_id))}",
        ),
        detector_id=DETECTOR_ID,
        selected_polarity=int(best.polarity),
        block_hri_provenance_hash=provenance_hash,
        block_provenance=all_rows,
        interval_rejection_reasons=rejection_reasons,
        peak_ordinals=np.arange(best.peaks.size, dtype=np.int64),
        detector_score=float(best.score),
        detector_coverage=float(best.coverage),
    )
    result.validate_identity()
    return result


def detect_pulses_per_wavelength_aboy_project(
    values: np.ndarray | CanonicalSignalViews,
    *,
    fs_hz: float = CANONICAL_FS_HZ,
    min_observation_sec: float = 8.0,
    min_peaks: int = MIN_BASIC_RATE_PEAKS,
    source_route: SignalRoute | str | None = None,
    run_id: str | None = None,
) -> dict[str, PulseResult]:
    """Run the same implementation independently for RED and IR."""

    from .resolver import validate_peak_detection_parameters

    min_observation_sec, min_peaks = validate_peak_detection_parameters(
        min_observation_sec,
        min_peaks,
    )
    matrix, sample_offset, route, record_id = _prepare_input(
        values,
        fs_hz=fs_hz,
        source_route=source_route,
    )
    if matrix.shape[0] / float(fs_hz) < min_observation_sec:
        raise ValueError(
            "HR/PPI requires at least "
            f"{min_observation_sec:g} seconds of observation"
        )
    block_samples = int(round(BLOCK_SECONDS * float(fs_hz)))
    if matrix.shape[0] < block_samples:
        raise ValueError(
            f"{DETECTOR_ID} requires at least one complete 10-second block"
        )
    labels = ("RED", "IR")[: matrix.shape[1]]
    return {
        label: _result_for_wavelength(
            matrix[:, channel],
            label=label,
            fs_hz=fs_hz,
            sample_offset=sample_offset,
            source_route=route,
            record_id=record_id,
            run_id=run_id,
            min_peaks=min_peaks,
            min_observation_sec=min_observation_sec,
        )
        for channel, label in enumerate(labels)
    }


__all__ = [
    "BLOCK_SECONDS",
    "DETECTOR_ID",
    "DETECTOR_NAME",
    "DETECTOR_VERSION",
    "INITIAL_HRI",
    "INITIAL_HRI_RULE",
    "IMPLEMENTATION_PATH",
    "MAX_BPM",
    "MIN_BASIC_RATE_PEAKS",
    "MIN_BPM",
    "detect_pulses_per_wavelength_aboy_project",
]
