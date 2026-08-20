"""双极性脉搏峰、PPI 与邻接合同 / Dual-polarity pulse, PPI, and adjacency.

拒绝的间期不会从时间轴删除；所有 interval endpoint 都指向原 peak ordinal。
Rejected intervals remain on the original time axis and keep explicit endpoints.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from scipy import signal

from ..contracts import PulseResult, SignalRoute
from .views import CANONICAL_FS_HZ, CanonicalSignalViews


DETECTOR_ID = "dual_polarity_prominence_v1_ablation"
DETECTOR_VERSION = "whole_record_dual_polarity_prominence_v1"
MIN_BASIC_RATE_PEAKS = 5


@dataclass(frozen=True)
class _Candidate:
    """内部极性/波长候选 / Internal polarity and wavelength candidate."""

    peaks: np.ndarray
    prominence: np.ndarray
    score: float
    polarity: int
    channel_index: int


def _robust_scale(values: np.ndarray) -> float:
    """稳健 IQR scale，退化时回退标准差 / Robust IQR scale with SD fallback."""

    x = np.asarray(values, dtype=np.float64)
    q25, q75 = np.percentile(x, [25.0, 75.0])
    scale = float((q75 - q25) / 1.349)
    return scale if scale > 1e-12 else max(float(np.std(x)), 1e-12)


def _candidate(
    values: np.ndarray,
    *,
    channel_index: int,
    polarity: int,
    fs_hz: float,
) -> _Candidate:
    """评分一个通道/极性 / Score one channel-polarity alternative."""

    oriented = np.asarray(values, dtype=np.float64) * float(polarity)
    scale = _robust_scale(oriented)
    peaks, properties = signal.find_peaks(
        oriented,
        distance=max(1, int(round(0.30 * fs_hz))),
        prominence=max(0.15 * scale, 1e-12),
    )
    prominence = np.asarray(properties.get("prominences", np.zeros(peaks.size)), dtype=np.float64)
    if peaks.size >= 2:
        ppi = np.diff(peaks) / fs_hz
        plausible = float(np.mean((ppi >= 0.30) & (ppi <= 2.00)))
        interval_cv = float(np.std(ppi) / max(np.mean(ppi), 1e-12))
    else:
        plausible, interval_cv = 0.0, 1.0
    duration = oriented.size / fs_hz
    rate_hz = peaks.size / max(duration, 1e-12)
    density = float(np.exp(-((rate_hz - 1.4) / 1.4) ** 2))
    prominence_score = float(np.tanh(np.median(prominence) / scale)) if prominence.size else 0.0
    score = 0.55 * plausible + 0.20 * density + 0.20 * prominence_score + 0.05 * np.exp(-interval_cv)
    return _Candidate(peaks.astype(np.int64), prominence, float(score), polarity, channel_index)


def _detect_pulses_dual_polarity_ablation(
    values: np.ndarray | CanonicalSignalViews,
    *,
    fs_hz: float = CANONICAL_FS_HZ,
    wavelength: str = "auto",
    min_observation_sec: float = 8.0,
    min_peaks: int = MIN_BASIC_RATE_PEAKS,
    source_route: SignalRoute | str | None = None,
    run_id: str | None = None,
) -> PulseResult:
    """公共峰检测入口 / Public pulse-detection entry point.

    `CanonicalSignalViews` 自动选择合法 rate signal；失败 reducer 无法构造该视图，
    因此这里不存在静默 direct fallback。A canonical view automatically selects its
    legal rate signal, so a failed reducer can never silently reach this function.
    """

    from ..peaks.resolver import validate_peak_detection_parameters

    min_observation_sec, min_peaks = validate_peak_detection_parameters(
        min_observation_sec,
        min_peaks,
    )
    if float(fs_hz) != CANONICAL_FS_HZ:
        raise ValueError("pulse detection requires the exact 400 Hz time grid")
    if isinstance(values, CanonicalSignalViews):
        matrix = values.analysis_signal
        valid_samples = values.rate_valid_mask
        resolved_route = values.route
        if source_route is not None and SignalRoute(source_route) is not resolved_route:
            raise ValueError("declared pulse source_route disagrees with signal views")
        record_id = str(values.metadata.get("record_id", "")).strip()
    else:
        matrix = np.asarray(values, dtype=np.float64)
        valid_samples = np.ones(matrix.shape[0], dtype=bool)
        resolved_route = (
            SignalRoute.DIRECT
            if source_route is None
            else SignalRoute(source_route)
        )
        record_id = ""
    if resolved_route is SignalRoute.DROPPED:
        raise ValueError("pulse detection cannot run on a dropped route")
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[1] not in (1, 2) or not np.isfinite(matrix).all():
        raise ValueError("pulse input must be finite samples-by-one/two channels")
    if run_id is None:
        if record_id:
            resolved_run_id = (
                f"{record_id}::{resolved_route.value}::{DETECTOR_ID}"
            )
        else:
            digest = hashlib.sha256(
                np.ascontiguousarray(matrix, dtype="<f8").tobytes(order="C")
            ).hexdigest()[:20]
            resolved_run_id = (
                f"array::{resolved_route.value}::{DETECTOR_ID}::{digest}"
            )
    else:
        resolved_run_id = str(run_id).strip()
    if not resolved_run_id:
        raise ValueError("pulse detection run_id cannot be empty")
    if matrix.shape[0] / fs_hz < min_observation_sec:
        raise ValueError(
            "HR/PPI requires at least "
            f"{min_observation_sec:g} seconds of observation"
        )
    sample_offset = 0
    if not np.all(valid_samples):
        # 中文：选择最长连续 artifact-valid run，绝不把 invalid gap 两侧拼接。
        # English: Use the longest contiguous valid run; never bridge an invalid gap.
        padded = np.concatenate(([False], valid_samples, [False]))
        changes = np.diff(padded.astype(np.int8))
        runs = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
        if not runs:
            raise ValueError("rate waveform has no artifact-valid samples")
        start, stop = max(runs, key=lambda bounds: bounds[1] - bounds[0])
        if (stop - start) / fs_hz < min_observation_sec:
            raise ValueError(
                "longest artifact-valid run is shorter than configured "
                f"{min_observation_sec:g} seconds"
            )
        matrix = matrix[start:stop]
        sample_offset = int(start)
    labels = ("RED", "IR")[: matrix.shape[1]]
    if wavelength.upper() == "AUTO":
        channel_indices = range(matrix.shape[1])
    else:
        if wavelength.upper() not in labels:
            raise ValueError(f"wavelength must be auto or one of {labels}")
        channel_indices = (labels.index(wavelength.upper()),)

    candidates = [
        _candidate(matrix[:, channel], channel_index=channel, polarity=polarity, fs_hz=fs_hz)
        for channel in channel_indices
        for polarity in (1, -1)
    ]
    best = max(candidates, key=lambda item: (item.score, item.peaks.size, -item.channel_index, item.polarity))
    peaks = best.peaks + sample_offset
    if peaks.size < min_peaks:
        raise ValueError(f"HR/PPI requires at least {min_peaks} detected peaks")
    timestamps = peaks.astype(np.float64) / fs_hz
    start = np.arange(max(0, peaks.size - 1), dtype=np.int64)
    stop = start + 1
    ppi = np.diff(timestamps)
    valid_intervals = np.isfinite(ppi) & (ppi >= 0.30) & (ppi <= 2.00)
    # 中文：intervals 来自连续原始峰，故 adjacency 恒真；validity 单独表达拒绝。
    # English: Intervals join consecutive original peaks; validity remains a separate mask.
    adjacency = np.ones(ppi.size, dtype=bool)
    accepted_peaks = np.zeros(peaks.size, dtype=bool)
    if valid_intervals.size:
        accepted_peaks[:-1] |= valid_intervals
        accepted_peaks[1:] |= valid_intervals
    median_prominence = max(float(np.median(best.prominence)), 1e-12)
    confidence = np.clip(best.prominence / (2.0 * median_prominence), 0.0, 1.0)
    valid_ppi = ppi[valid_intervals]
    peak_span = float(timestamps[-1] - timestamps[0]) if peaks.size >= 2 else 0.0
    detector_coverage = (
        float(np.clip(np.sum(valid_ppi) / peak_span, 0.0, 1.0))
        if peak_span > 0.0
        else 0.0
    )
    provenance = (
        {
            "algorithm": "legacy_whole_record_fixed_distance_prominence",
            "detector_id": DETECTOR_ID,
            "channel_index": int(best.channel_index),
            "polarity": int(best.polarity),
            "score": float(best.score),
            "valid_run_offset": int(sample_offset),
            "min_observation_sec": float(min_observation_sec),
            "min_peaks": int(min_peaks),
        },
    )
    provenance_hash = hashlib.sha256(
        json.dumps(
            provenance,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    result = PulseResult(
        peaks=peaks,
        peak_timestamps_s=timestamps,
        accepted_peak_mask=accepted_peaks,
        interval_start_peak_indices=start,
        interval_stop_peak_indices=stop,
        ppi_s=ppi,
        valid_interval_mask=valid_intervals,
        adjacency_mask=adjacency,
        wavelength=labels[best.channel_index],
        detector_version=DETECTOR_VERSION,
        confidence=np.asarray(confidence, dtype=np.float64),
        source_route=resolved_route,
        detection_run_id=resolved_run_id,
        interval_run_ids=np.full(
            ppi.shape,
            resolved_run_id,
            dtype=f"<U{max(1, len(resolved_run_id))}",
        ),
        detector_id=DETECTOR_ID,
        selected_polarity=int(best.polarity),
        block_hri_provenance_hash=provenance_hash,
        block_provenance=provenance,
        interval_rejection_reasons=tuple(
            "" if valid else "outside_legacy_0p30_to_2p00_ppi"
            for valid in valid_intervals.tolist()
        ),
        peak_ordinals=np.arange(peaks.size, dtype=np.int64),
        detector_score=float(best.score),
        detector_coverage=detector_coverage,
    )
    result.validate_identity()
    return result


__all__: tuple[str, ...] = ()
