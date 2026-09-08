"""Seven-step project Aboy peak-detector contract (version 2).

Unlike :mod:`ppg_frailty.peaks.aboy_project`, this module implements the
authoritative seven-step project contract literally.  Version 1 remains
available so existing runs keep their original algorithm identity.
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

DETECTOR_ID = "aboy_project_v2"
DETECTOR_VERSION = "project_aboy_seven_step_authority_v2"
DETECTOR_NAME = "project Aboy seven-step detector v2"
IMPLEMENTATION_PATH = "ppg_frailty.peaks.aboy_project_v2." "detect_pulses_per_wavelength_aboy_project_v2"
BLOCK_SECONDS = 10.0
HIGHPASS_HZ = 0.2
HIGHPASS_ORDER = 2
INITIAL_HRI = 0.0
MIN_BPM = 35.0
MAX_BPM = 210.0
MIN_BASIC_RATE_PEAKS = 5


@dataclass(frozen=True)
class _Candidate:
    polarity: int
    peaks: np.ndarray
    prominence: np.ndarray
    hri_out: float
    n_clean_ppi: int
    coverage: float
    cv: float
    score: float
    provenance: dict[str, Any]


def _stable_hash(rows: tuple[dict[str, Any], ...]) -> str:
    encoded = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepare_input(
    values: np.ndarray | CanonicalSignalViews,
    *,
    fs_hz: float,
    source_route: SignalRoute | str | None,
) -> tuple[np.ndarray, int, SignalRoute, str, str]:
    """Return the detector input before the v2-owned 0.2-Hz high-pass."""

    if float(fs_hz) != CANONICAL_FS_HZ:
        raise ValueError("Aboy project v2 requires the exact 400 Hz time grid")
    if isinstance(values, CanonicalSignalViews):
        values.validate()
        route = values.route
        if source_route is not None and SignalRoute(source_route) is not route:
            raise ValueError("declared pulse source_route disagrees with signal views")
        if route is SignalRoute.ARTIFACT_RATE_ONLY:
            matrix = np.asarray(values.analysis_signal, dtype=np.float64)
            valid_samples = np.asarray(values.rate_valid_mask, dtype=bool)
            source_view = "artifact_rate_only"
        else:
            # The v2 detector owns the seven-step 0.2-Hz high-pass.  Feeding
            # x_filter here would apply the initial preprocessing twice.
            matrix = np.asarray(values.x_native, dtype=np.float64)
            valid_samples = np.ones(matrix.shape[0], dtype=bool)
            source_view = "repaired_native_ppg"
        record_id = str(values.metadata.get("record_id", "")).strip()
    else:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        route = SignalRoute.DIRECT if source_route is None else SignalRoute(source_route)
        valid_samples = np.ones(matrix.shape[0], dtype=bool)
        record_id = ""
        source_view = "caller_declared_raw_ppg"
    if route is SignalRoute.DROPPED:
        raise ValueError("pulse detection cannot run on a dropped route")
    if (matrix.ndim != 2 or matrix.shape[1] not in (1, 2) or not np.isfinite(matrix).all() or valid_samples.shape !=
        (matrix.shape[0], )):
        raise ValueError("pulse input must be finite samples-by-one/two channels")
    offset = 0
    if not np.all(valid_samples):
        padded = np.concatenate(([False], valid_samples, [False]))
        changes = np.diff(padded.astype(np.int8))
        runs = tuple(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
        if not runs:
            raise ValueError("rate waveform has no artifact-valid samples")
        start, stop = max(runs, key=lambda bounds: bounds[1] - bounds[0])
        matrix = matrix[start:stop]
        offset = int(start)
    return matrix, offset, route, record_id, source_view


def _highpass(values: np.ndarray, fs_hz: float) -> np.ndarray:
    sos = signal.butter(
        HIGHPASS_ORDER,
        HIGHPASS_HZ,
        btype="highpass",
        fs=float(fs_hz),
        output="sos",
    )
    try:
        output = signal.sosfiltfilt(sos, np.asarray(values, dtype=np.float64))
    except ValueError as exc:
        raise ValueError(f"aboy_project_v2_highpass_failed:{exc}") from exc
    if not np.isfinite(output).all():
        raise FloatingPointError("aboy_project_v2_highpass_returned_nonfinite")
    return np.asarray(output, dtype=np.float64)


def _bandpass(values: np.ndarray, fs_hz: float, high_hz: float) -> np.ndarray:
    sos = signal.butter(
        2,
        [0.5, float(high_hz)],
        btype="bandpass",
        fs=float(fs_hz),
        output="sos",
    )
    return np.asarray(signal.sosfiltfilt(sos, values), dtype=np.float64)


def _physiology_and_mad(peaks: np.ndarray, fs_hz: float) -> tuple[np.ndarray, tuple[str, ...]]:
    """Apply step 7 bounds first and then four robust-SD MAD cleaning."""

    ppi = np.diff(np.asarray(peaks, dtype=np.int64)) / float(fs_hz)
    physiological = np.isfinite(ppi) & (ppi >= 60.0 / MAX_BPM) & (ppi <= 60.0 / MIN_BPM)
    mad_valid = np.ones(ppi.size, dtype=bool)
    if int(np.count_nonzero(physiological)) >= 5:
        selected = ppi[physiological]
        median = float(np.median(selected))
        robust_sigma = 1.4826 * float(np.median(np.abs(selected - median)))
        if robust_sigma > 0.0:
            mad_valid[physiological] = np.abs(ppi[physiological] - median) <= 4.0 * robust_sigma
    valid = physiological & mad_valid
    reasons = tuple(";".join(part for part, failed in (
        ("outside_35_210_bpm", not physiological[index]),
        ("outside_4x_1p4826_mad", not mad_valid[index]),
    ) if failed) for index in range(ppi.size))
    return valid, reasons


def _score(peaks: np.ndarray, fs_hz: float) -> tuple[float, int, float, float]:
    valid, _ = _physiology_and_mad(peaks, fs_hz)
    ppi = np.diff(np.asarray(peaks, dtype=np.int64)) / float(fs_hz)
    selected = ppi[valid]
    n_clean = int(selected.size)
    span = (int(peaks[-1]) - int(peaks[0])) / float(fs_hz) if np.asarray(peaks).size >= 2 else 0.0
    coverage = float(np.clip(np.sum(selected) / span, 0.0, 1.0)) if span else 0.0
    cv = (float(np.std(selected, ddof=0) /
                np.mean(selected)) if selected.size >= 2 and float(np.mean(selected)) > 0.0 else 2.0)
    return float(n_clean + 0.5 * coverage - min(cv, 2.0)), n_clean, coverage, cv


def _candidate(
    block: np.ndarray,
    *,
    block_index: int,
    block_start: int,
    polarity: int,
    hri_in: float,
    fs_hz: float,
) -> _Candidate:
    high_hz = min(8.0, max(1.5, 3.0 * (1.0 + float(hri_in))))
    filtered = _bandpass(np.asarray(block) * float(polarity), fs_hz, high_hz)
    d_210 = max(1, int(round(float(fs_hz) * 60.0 / MAX_BPM)))
    preliminary, _ = signal.find_peaks(filtered, distance=d_210)
    preliminary = np.asarray(preliminary, dtype=np.int64)
    preliminary_ppi = np.diff(preliminary) / float(fs_hz)
    retained = np.empty(0, dtype=np.float64)
    hri_out = float(hri_in)
    update_reason = "insufficient_retained_ppi"
    threshold: float | None = None
    if preliminary_ppi.size:
        threshold = float(np.percentile(preliminary_ppi, 30.0))
        retained = preliminary_ppi[preliminary_ppi > threshold]
        if retained.size >= 2:
            retained_mean = float(np.mean(retained))
            retained_median = float(np.median(retained))
            if 0.5 * retained_median <= retained_mean <= 1.5 * retained_median and retained_mean > 0.0:
                hri_out = float(10.0 * np.std(retained, ddof=0) / retained_mean)
                update_reason = "updated_from_retained_pd_population_sd"
            else:
                update_reason = "retained_mean_outside_retained_median_guard"
    hrwin = float(fs_hz) / (3.0 * (1.0 + hri_out))
    distance = max(int(round(2.0 * hrwin)), d_210)
    if preliminary.size:
        amplitudes = np.sort(filtered[preliminary])
        upper_count = max(1, int(np.ceil(0.30 * amplitudes.size)))
        upper_mean = float(np.mean(amplitudes[-upper_count:]))
    else:
        upper_mean = float("-inf")
    block_sd = float(np.std(filtered, ddof=0))
    prominence_threshold = max(0.25 * max(upper_mean, block_sd), 1e-12)
    peaks, properties = signal.find_peaks(
        filtered,
        distance=distance,
        prominence=prominence_threshold,
    )
    peaks = np.asarray(peaks, dtype=np.int64) + int(block_start)
    prominence = np.asarray(properties.get("prominences", np.zeros(peaks.size)), dtype=np.float64)
    score, n_clean, coverage, cv = _score(peaks, fs_hz)
    provenance = {
        "block_index": int(block_index),
        "block_start_sample": int(block_start),
        "block_stop_sample": int(block_start + block.size),
        "polarity": int(polarity),
        "hri_in": float(hri_in),
        "hri_out": float(hri_out),
        "hri_update_reason": update_reason,
        "preliminary_peak_count": int(preliminary.size),
        "preliminary_ppi_30th_percentile_s": threshold,
        "retained_pd_count": int(retained.size),
        "retained_pd_median_s": float(np.median(retained)) if retained.size else None,
        "retained_pd_mean_s": float(np.mean(retained)) if retained.size else None,
        "adaptive_high_hz": float(high_hz),
        "d_210_samples": int(d_210),
        "hrwin_samples": float(hrwin),
        "final_distance_samples": int(distance),
        "upper_30_percent_preliminary_amplitude_mean": (float(upper_mean) if np.isfinite(upper_mean) else None),
        "block_population_sd": block_sd,
        "final_prominence": float(prominence_threshold),
        "final_peak_count": int(peaks.size),
        "n_clean_ppi": int(n_clean),
        "coverage": float(coverage),
        "cv": float(cv),
        "score": float(score),
    }
    return _Candidate(polarity, peaks, prominence, hri_out, n_clean, coverage, cv, score, provenance)


def _remove_ratio_outlier_peaks(peaks: np.ndarray, prominence: np.ndarray,
                                fs_hz: float) -> tuple[np.ndarray, np.ndarray, float | None, tuple[int, ...]]:
    """Remove the following peak before physiological/MAD interval cleaning."""

    source = np.asarray(peaks, dtype=np.int64)
    if source.size < 3:
        return source, np.asarray(prominence), None, ()
    ppi = np.diff(source) / float(fs_hz)
    reference = float(np.median(ppi))
    bad = (ppi < 0.5 * reference) | (ppi > 1.8 * reference)
    rejected_indices = tuple((np.flatnonzero(bad) + 1).astype(int).tolist())
    keep = np.ones(source.size, dtype=bool)
    keep[list(rejected_indices)] = False
    return source[keep], np.asarray(prominence)[keep], reference, rejected_indices


def _result_for_channel(
    values: np.ndarray,
    *,
    label: str,
    fs_hz: float,
    sample_offset: int,
    route: SignalRoute,
    record_id: str,
    source_view: str,
    run_id: str | None,
    min_peaks: int,
    min_observation_sec: float,
) -> PulseResult:
    filtered = _highpass(values, fs_hz)
    block_samples = int(round(BLOCK_SECONDS * fs_hz))
    hri = float(INITIAL_HRI)
    selected_rows: list[_Candidate] = []
    audit_rows: list[dict[str, Any]] = []
    for block_index in range(filtered.size // block_samples):
        local_start = block_index * block_samples
        absolute_start = sample_offset + local_start
        block = filtered[local_start:local_start + block_samples]
        candidates = tuple(
            _candidate(
                block,
                block_index=block_index,
                block_start=absolute_start,
                polarity=polarity,
                hri_in=hri,
                fs_hz=fs_hz,
            ) for polarity in (1, -1))
        selected = max(
            candidates,
            key=lambda row: (row.score, row.n_clean_ppi, row.peaks.size, row.polarity),
        )
        selected_rows.append(selected)
        hri = selected.hri_out
        audit_rows.extend({
            **candidate.provenance,
            "selected_block_polarity": int(selected.polarity),
            "polarity_selected": candidate is selected,
            "source_view_before_highpass": source_view,
            "initial_highpass_hz": HIGHPASS_HZ,
            "initial_highpass_order": HIGHPASS_ORDER,
        } for candidate in candidates)
    peak_parts = [row.peaks for row in selected_rows if row.peaks.size]
    prominence_parts = [row.prominence for row in selected_rows if row.peaks.size]
    peaks = np.concatenate(peak_parts) if peak_parts else np.empty(0, dtype=np.int64)
    prominence = np.concatenate(prominence_parts) if prominence_parts else np.empty(0)
    peaks, prominence, ratio_reference, rejected = _remove_ratio_outlier_peaks(peaks, prominence, fs_hz)
    if peaks.size < int(min_peaks):
        raise ValueError(f"{DETECTOR_ID} requires at least {min_peaks} detected peaks for {label}")
    valid, reasons = _physiology_and_mad(peaks, fs_hz)
    score, _n_clean, coverage, _cv = _score(peaks, fs_hz)
    final_row = {
        "stage": "merge_ratio_remove_then_interval_clean",
        "wavelength": label,
        "ratio_reference_ppi_s": ratio_reference,
        "ratio_rejected_peak_indices_before_removal": list(rejected),
        "final_peak_count": int(peaks.size),
        "min_observation_sec": float(min_observation_sec),
        "min_peaks": int(min_peaks),
    }
    provenance = tuple(audit_rows + [final_row])
    provenance_hash = _stable_hash(provenance)
    if run_id is not None:
        resolved_run_id = str(run_id).strip()
    elif record_id:
        resolved_run_id = f"{record_id}::{route.value}::{DETECTOR_ID}::{label}::" f"{provenance_hash[:16]}"
    else:
        digest = hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes(order="C")).hexdigest()[:20]
        resolved_run_id = f"array::{route.value}::{DETECTOR_ID}::{label}::{digest}"
    if not resolved_run_id:
        raise ValueError("pulse detection run_id cannot be empty")
    timestamps = peaks.astype(np.float64) / fs_hz
    ppi = np.diff(timestamps)
    starts = np.arange(ppi.size, dtype=np.int64)
    median_prominence = max(float(np.median(prominence)), 1e-12)
    confidence = np.clip(prominence / (2.0 * median_prominence), 0.0, 1.0)
    polarity_sum = sum(row.polarity for row in selected_rows)
    dominant_polarity = 1 if polarity_sum >= 0 else -1
    result = PulseResult(
        peaks=peaks,
        peak_timestamps_s=timestamps,
        accepted_peak_mask=np.ones(peaks.size, dtype=bool),
        interval_start_peak_indices=starts,
        interval_stop_peak_indices=starts + 1,
        ppi_s=ppi,
        valid_interval_mask=valid,
        adjacency_mask=np.ones(ppi.size, dtype=bool),
        wavelength=label,
        detector_version=DETECTOR_VERSION,
        confidence=confidence,
        source_route=route,
        detection_run_id=resolved_run_id,
        interval_run_ids=np.full(
            ppi.shape,
            resolved_run_id,
            dtype=f"<U{max(1, len(resolved_run_id))}",
        ),
        detector_id=DETECTOR_ID,
        selected_polarity=dominant_polarity,
        block_hri_provenance_hash=provenance_hash,
        block_provenance=provenance,
        interval_rejection_reasons=reasons,
        peak_ordinals=np.arange(peaks.size, dtype=np.int64),
        detector_score=float(score),
        detector_coverage=float(coverage),
    )
    result.validate_identity()
    return result


def detect_pulses_per_wavelength_aboy_project_v2(
    values: np.ndarray | CanonicalSignalViews,
    *,
    fs_hz: float = CANONICAL_FS_HZ,
    min_observation_sec: float = 8.0,
    min_peaks: int = MIN_BASIC_RATE_PEAKS,
    source_route: SignalRoute | str | None = None,
    run_id: str | None = None,
) -> dict[str, PulseResult]:
    """Execute all seven contract steps independently for RED and IR."""

    from .resolver import validate_peak_detection_parameters

    min_observation_sec, min_peaks = validate_peak_detection_parameters(min_observation_sec, min_peaks)
    matrix, offset, route, record_id, source_view = _prepare_input(values, fs_hz=fs_hz, source_route=source_route)
    if matrix.shape[0] / fs_hz < min_observation_sec:
        raise ValueError(f"HR/PPI requires at least {min_observation_sec:g} seconds of observation")
    if matrix.shape[0] < int(round(BLOCK_SECONDS * fs_hz)):
        raise ValueError(f"{DETECTOR_ID} requires one complete 10-second block")
    labels = ("RED", "IR")[:matrix.shape[1]]
    return {
        label: _result_for_channel(
            matrix[:, channel],
            label=label,
            fs_hz=fs_hz,
            sample_offset=offset,
            route=route,
            record_id=record_id,
            source_view=source_view,
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
    "HIGHPASS_HZ",
    "HIGHPASS_ORDER",
    "IMPLEMENTATION_PATH",
    "MAX_BPM",
    "MIN_BASIC_RATE_PEAKS",
    "MIN_BPM",
    "detect_pulses_per_wavelength_aboy_project_v2",
]
