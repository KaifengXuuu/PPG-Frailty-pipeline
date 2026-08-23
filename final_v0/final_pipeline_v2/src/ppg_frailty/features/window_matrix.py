"""Pure 146-channel chronological window feature matrix.

The extractor reuses recording-global pulse/morphology identities and assigns
events to configurable complete engineering windows by timestamp.  Routing,
quality, motion, coverage, and validity remain QC/provenance, never predictors.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from ..contracts import (
    EngineeringFeatureSequence,
    OrderedFeatureMatrixV1,
    PulseResult,
    RoutingTimeline,
    SignalRoute,
)
from ..data.windows import WindowPlan
from ..provenance import assert_training_only
from ..quality.routing_timeline import matrix_row_route, overlapping_cells
from ..signal.morphology import MORPHOLOGY_NAMES, MorphologyResult
from ..signal.views import CANONICAL_FS_HZ, CanonicalSignalViews
from .engineering import engineering_feature_names, extract_engineering_features


WINDOW_FEATURE_SCHEMA_VERSION = "window_feature_set_d146_v1"
ORDERED_WINDOW_MATRIX_SCHEMA_VERSION = (
    "ordered_window_feature_matrix_d146_variable_k_v1"
)

MORPHOLOGY_WINDOW_NAMES = tuple(
    f"local_morphology.{name}.{statistic}"
    for name in MORPHOLOGY_NAMES
    for statistic in ("median", "mad")
)
INTERVAL_WINDOW_NAMES = (
    "local_interval.ppi_mean_s",
    "local_interval.ppi_median_s",
    "local_interval.ppi_population_sd_s",
    "local_interval.ppi_iqr_s",
    "local_interval.ppi_mad_s",
    "local_interval.ppi_cv",
    "local_interval.hr_mean_bpm",
    "local_interval.hr_median_bpm",
    "local_interval.hr_population_sd_bpm",
)
SUCCESSIVE_WINDOW_NAMES = (
    "local_successive.delta_ppi_mean_s",
    "local_successive.delta_ppi_median_s",
    "local_successive.delta_ppi_population_sd_s",
    "local_successive.delta_ppi_mad_s",
    "local_successive.abs_delta_ppi_mean_s",
    "local_successive.abs_delta_ppi_median_s",
)
LOCAL_PRV_WINDOW_NAMES = (
    "local_prv.mean_squared_delta_ppi_s2",
    "local_prv.pnn50_fraction",
)
RATE_WINDOW_NAMES = (
    *INTERVAL_WINDOW_NAMES,
    *SUCCESSIVE_WINDOW_NAMES,
    *LOCAL_PRV_WINDOW_NAMES,
)


def window_feature_names() -> tuple[str, ...]:
    names = (
        *engineering_feature_names(),
        *MORPHOLOGY_WINDOW_NAMES,
        *RATE_WINDOW_NAMES,
    )
    if len(names) != 146 or len(set(names)) != 146:
        raise RuntimeError("window feature schema must contain 146 unique predictors")
    forbidden = ("sqi", "motion", "route", "coverage", "technical_metadata")
    if any(any(token in name.lower() for token in forbidden) for name in names):
        raise RuntimeError("routing/QC metadata leaked into matrix predictors")
    return names


@dataclass(frozen=True)
class WindowFeatureExtraction:
    """Chronological 146-feature rows plus non-predictor validity and tiers."""

    sequence: EngineeringFeatureSequence
    value_validity: np.ndarray
    row_tiers: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FoldWindowFeatureTransform:
    """Outer-training-only robust transform for the 146 window predictors."""

    center: np.ndarray
    scale: np.ndarray
    valid_count: np.ndarray
    feature_names: tuple[str, ...]
    fitted_on_participant_ids: tuple[str, ...]


def validate_window_feature_extraction(
    extraction: WindowFeatureExtraction, *, transformed: bool
) -> None:
    names = window_feature_names()
    sequence = extraction.sequence
    values = np.asarray(sequence.values)
    validity = np.asarray(extraction.value_validity)
    expected = WINDOW_FEATURE_SCHEMA_VERSION + (
        "+outer_train_robust_v1" if transformed else ""
    )
    if (
        tuple(sequence.channel_schema) != names
        or sequence.schema_version != expected
        or values.ndim != 2
        or values.shape[1] != len(names)
        or validity.shape != values.shape
        or np.asarray(sequence.start_samples).shape != (values.shape[0],)
        or np.asarray(sequence.valid_row_mask).shape != (values.shape[0],)
        or len(extraction.row_tiers) != values.shape[0]
    ):
        raise ValueError("146-channel window feature extraction schema drift")


def _source_route_name(pulse: PulseResult) -> str:
    value = getattr(pulse.source_route, "value", str(pulse.source_route))
    if value in {"direct_x_filter", "identity_direct", "direct"}:
        return "direct"
    if value in {"non_identity_x_ar_rate_only", "processed"}:
        return "processed"
    raise ValueError("pulse result does not originate from a usable source route")


def _cell_for_sample(timeline: RoutingTimeline, sample: float):
    index = int(np.floor(sample))
    if np.isclose(sample, timeline.n_samples):
        index = timeline.n_samples - 1
    for cell in timeline.cells:
        if cell.start_sample_400 <= index < cell.stop_sample_400:
            return cell
    return None


def _eligible_event_segment(
    timeline: RoutingTimeline,
    *,
    start_sample: float,
    stop_sample: float,
    source_route: str,
) -> bool:
    left = _cell_for_sample(timeline, start_sample)
    right = _cell_for_sample(timeline, max(start_sample, stop_sample - 1e-9))
    if left is None or right is None or left.cell_id != right.cell_id:
        return False
    return (
        left.source_route == source_route
        and left.final_tier in {"excellent", "acceptable"}
    )


def build_route_eligible_rate_pulse(
    timeline: RoutingTimeline,
    direct_pulse: PulseResult,
    processed_pulse: PulseResult | None = None,
) -> PulseResult:
    """Combine eligible PPI segments without crossing ownership/source boundaries."""

    timeline.validate()
    pulses = (direct_pulse,) + (() if processed_pulse is None else (processed_pulse,))
    segments: list[tuple[float, PulseResult, tuple[int, ...], str]] = []
    for pulse in pulses:
        pulse.validate_identity()
        source_route = _source_route_name(pulse)
        starts = np.asarray(pulse.interval_start_peak_indices, dtype=np.int64)
        stops = np.asarray(pulse.interval_stop_peak_indices, dtype=np.int64)
        times = np.asarray(pulse.peak_timestamps_s, dtype=np.float64)
        candidate_rows: list[tuple[int, str]] = []
        for index, (left_index, right_index) in enumerate(zip(starts, stops)):
            left_sample = float(times[left_index] * CANONICAL_FS_HZ)
            right_sample = float(times[right_index] * CANONICAL_FS_HZ)
            left_cell = _cell_for_sample(timeline, left_sample)
            right_cell = _cell_for_sample(
                timeline, max(left_sample, right_sample - 1e-9)
            )
            if (
                left_cell is not None
                and right_cell is not None
                and left_cell.cell_id == right_cell.cell_id
                and left_cell.source_route == source_route
                and left_cell.final_tier in {"excellent", "acceptable"}
            ):
                candidate_rows.append((index, left_cell.cell_id))
        current: list[int] = []
        current_cell: str | None = None
        for index, cell_id in candidate_rows:
            contiguous = bool(
                current
                and index == current[-1] + 1
                and cell_id == current_cell
                and int(stops[current[-1]]) == int(starts[index])
                and str(pulse.interval_run_ids[current[-1]])
                == str(pulse.interval_run_ids[index])
            )
            if current and not contiguous:
                segments.append(
                    (float(times[starts[current[0]]]), pulse, tuple(current), source_route)
                )
                current = []
            current.append(index)
            current_cell = cell_id
        if current:
            segments.append(
                (float(times[starts[current[0]]]), pulse, tuple(current), source_route)
            )
    if not segments:
        raise ValueError("routing timeline contains no eligible rate/PPI segment")
    segments.sort(key=lambda item: (item[0], item[3]))

    peak_times: list[float] = []
    accepted_peaks: list[bool] = []
    confidence: list[float] = []
    interval_starts: list[int] = []
    interval_stops: list[int] = []
    ppi_values: list[float] = []
    interval_validity: list[bool] = []
    adjacency: list[bool] = []
    interval_run_ids: list[str] = []
    interval_routes: list[str] = []
    previous_peak_index: int | None = None
    wavelengths = {str(pulse.wavelength) for _, pulse, _, _ in segments}
    if len(wavelengths) != 1:
        raise ValueError("routing rate pulse sources use different wavelengths")

    for _, pulse, indices, source_route in segments:
        original_times = np.asarray(pulse.peak_timestamps_s, dtype=np.float64)
        original_accepted = np.asarray(pulse.accepted_peak_mask, dtype=bool)
        original_confidence = np.asarray(pulse.confidence, dtype=np.float64)
        original_starts = np.asarray(
            pulse.interval_start_peak_indices, dtype=np.int64
        )
        original_stops = np.asarray(
            pulse.interval_stop_peak_indices, dtype=np.int64
        )
        first_original = int(original_starts[indices[0]])
        first_new = len(peak_times)
        peak_times.append(float(original_times[first_original]))
        accepted_peaks.append(bool(original_accepted[first_original]))
        confidence.append(float(original_confidence[first_original]))
        if previous_peak_index is not None:
            interval_starts.append(previous_peak_index)
            interval_stops.append(first_new)
            ppi_values.append(float("nan"))
            interval_validity.append(False)
            adjacency.append(False)
            interval_run_ids.append("routing_boundary")
            interval_routes.append("routing_boundary")
        local_left = first_new
        for index in indices:
            original_stop = int(original_stops[index])
            local_right = len(peak_times)
            peak_times.append(float(original_times[original_stop]))
            accepted_peaks.append(bool(original_accepted[original_stop]))
            confidence.append(float(original_confidence[original_stop]))
            interval_starts.append(local_left)
            interval_stops.append(local_right)
            ppi_values.append(float(pulse.ppi_s[index]))
            interval_validity.append(bool(pulse.valid_interval_mask[index]))
            adjacency.append(bool(pulse.adjacency_mask[index]))
            interval_run_ids.append(str(pulse.interval_run_ids[index]))
            interval_routes.append(
                SignalRoute.DIRECT.value
                if source_route == "direct"
                else SignalRoute.ARTIFACT_RATE_ONLY.value
            )
            local_left = local_right
        previous_peak_index = local_left

    route_set = set(interval_routes) - {"routing_boundary"}
    composite_route = (
        SignalRoute.ARTIFACT_RATE_ONLY
        if route_set == {SignalRoute.ARTIFACT_RATE_ONLY.value}
        else SignalRoute.DIRECT
    )
    timestamps = np.asarray(peak_times, dtype=np.float64)
    result = PulseResult(
        peaks=np.rint(timestamps * CANONICAL_FS_HZ).astype(np.int64),
        peak_timestamps_s=timestamps,
        accepted_peak_mask=np.asarray(accepted_peaks, dtype=bool),
        interval_start_peak_indices=np.asarray(interval_starts, dtype=np.int64),
        interval_stop_peak_indices=np.asarray(interval_stops, dtype=np.int64),
        ppi_s=np.asarray(ppi_values, dtype=np.float64),
        valid_interval_mask=np.asarray(interval_validity, dtype=bool),
        adjacency_mask=np.asarray(adjacency, dtype=bool),
        wavelength=next(iter(wavelengths)),
        detector_version="routing_composite_from_global_runs_v1",
        confidence=np.asarray(confidence, dtype=np.float64),
        source_route=composite_route,
        detection_run_id="routing_composite_global_runs_v1",
        interval_run_ids=np.asarray(interval_run_ids),
        interval_source_routes=np.asarray(interval_routes),
    )
    result.validate_identity()
    return result


def route_eligible_morphology_aggregates(
    pulse: PulseResult,
    morphology: MorphologyResult,
    timeline: RoutingTimeline,
) -> tuple[dict[str, float], dict[str, bool]]:
    """Aggregate only complete globally detected beats in Excellent direct cells."""

    values, validity = _morphology_features(
        pulse, morphology, timeline, 0, timeline.n_samples
    )
    names = tuple(
        f"{name}_{statistic}"
        for name in MORPHOLOGY_NAMES
        for statistic in ("median", "mad")
    )
    return (
        {name: float(value) for name, value in zip(names, values)},
        {name: bool(value) for name, value in zip(names, validity)},
    )


def _pulse_intervals(
    pulse: PulseResult,
    timeline: RoutingTimeline,
    start_sample: int,
    stop_sample: int,
) -> tuple[np.ndarray, np.ndarray]:
    pulse.validate_identity()
    source_route = _source_route_name(pulse)
    peaks_s = np.asarray(pulse.peak_timestamps_s, dtype=np.float64)
    start_indices = np.asarray(pulse.interval_start_peak_indices, dtype=np.int64)
    stop_indices = np.asarray(pulse.interval_stop_peak_indices, dtype=np.int64)
    ppi = np.asarray(pulse.ppi_s, dtype=np.float64)
    valid = np.asarray(pulse.valid_interval_mask, dtype=bool)
    adjacency = np.asarray(pulse.adjacency_mask, dtype=bool)
    if not (
        start_indices.shape == stop_indices.shape == ppi.shape == valid.shape
        == adjacency.shape
    ):
        raise ValueError("pulse interval identity arrays are misaligned")
    starts = peaks_s[start_indices] * CANONICAL_FS_HZ
    stops = peaks_s[stop_indices] * CANONICAL_FS_HZ
    mids = (starts + stops) / 2.0
    allowed = np.asarray(
        [
            bool(
                valid[index]
                and np.isfinite(ppi[index])
                and ppi[index] > 0.0
                and start_sample <= mids[index] < stop_sample
                and _eligible_event_segment(
                    timeline,
                    start_sample=starts[index],
                    stop_sample=stops[index],
                    source_route=source_route,
                )
            )
            for index in range(ppi.size)
        ],
        dtype=bool,
    )
    return np.flatnonzero(allowed), adjacency


def _rate_features(
    pulses: Iterable[PulseResult],
    timeline: RoutingTimeline,
    start_sample: int,
    stop_sample: int,
) -> tuple[np.ndarray, np.ndarray]:
    interval_values: list[float] = []
    deltas: list[float] = []
    for pulse in pulses:
        indices, adjacency = _pulse_intervals(
            pulse, timeline, start_sample, stop_sample
        )
        ppi = np.asarray(pulse.ppi_s, dtype=np.float64)
        interval_values.extend(map(float, ppi[indices]))
        selected = set(map(int, indices))
        for left in sorted(selected):
            right = left + 1
            if (
                right in selected
                and bool(adjacency[left])
                and bool(adjacency[right])
                and int(pulse.interval_stop_peak_indices[left])
                == int(pulse.interval_start_peak_indices[right])
                and str(pulse.interval_run_ids[left])
                == str(pulse.interval_run_ids[right])
            ):
                pair_start = float(
                    pulse.peak_timestamps_s[
                        pulse.interval_start_peak_indices[left]
                    ]
                ) * CANONICAL_FS_HZ
                pair_stop = float(
                    pulse.peak_timestamps_s[
                        pulse.interval_stop_peak_indices[right]
                    ]
                ) * CANONICAL_FS_HZ
                pair_midpoint = 0.5 * (
                    float(pulse.peak_timestamps_s[
                        pulse.interval_start_peak_indices[left]
                    ])
                    + float(pulse.peak_timestamps_s[
                        pulse.interval_stop_peak_indices[right]
                    ])
                ) * CANONICAL_FS_HZ
                if (
                    start_sample <= pair_midpoint < stop_sample
                    and _eligible_event_segment(
                        timeline,
                        start_sample=pair_start,
                        stop_sample=pair_stop,
                        source_route=_source_route_name(pulse),
                    )
                ):
                    deltas.append(float(ppi[right] - ppi[left]))

    values = np.full(len(RATE_WINDOW_NAMES), np.nan, dtype=np.float64)
    validity = np.zeros(len(RATE_WINDOW_NAMES), dtype=bool)
    intervals = np.asarray(interval_values, dtype=np.float64)
    if intervals.size >= 4:
        q25, q75 = np.percentile(intervals, (25.0, 75.0))
        median = float(np.median(intervals))
        mean = float(np.mean(intervals))
        sd = float(np.std(intervals, ddof=0))
        hr = 60.0 / intervals
        values[:9] = (
            mean,
            median,
            sd,
            float(q75 - q25),
            float(np.median(np.abs(intervals - median))),
            float(sd / mean) if mean > 0.0 else np.nan,
            float(np.mean(hr)),
            float(np.median(hr)),
            float(np.std(hr, ddof=0)),
        )
        validity[:9] = np.isfinite(values[:9])
    differences = np.asarray(deltas, dtype=np.float64)
    if differences.size >= 3:
        delta_median = float(np.median(differences))
        absolute = np.abs(differences)
        values[9:] = (
            float(np.mean(differences)),
            delta_median,
            float(np.std(differences, ddof=0)),
            float(np.median(np.abs(differences - delta_median))),
            float(np.mean(absolute)),
            float(np.median(absolute)),
            float(np.mean(np.square(differences))),
            float(np.mean(absolute > 0.050)),
        )
        validity[9:] = np.isfinite(values[9:])
    return values, validity


def _morphology_features(
    pulse: PulseResult,
    morphology: MorphologyResult,
    timeline: RoutingTimeline,
    start_sample: int,
    stop_sample: int,
) -> tuple[np.ndarray, np.ndarray]:
    if _source_route_name(pulse) != "direct":
        raise ValueError("matrix morphology requires the global direct pulse run")
    peaks = np.asarray(pulse.peak_timestamps_s, dtype=np.float64) * CANONICAL_FS_HZ
    output = np.full(len(MORPHOLOGY_WINDOW_NAMES), np.nan, dtype=np.float64)
    validity = np.zeros(len(MORPHOLOGY_WINDOW_NAMES), dtype=bool)
    for name_index, name in enumerate(MORPHOLOGY_NAMES):
        beat_values = np.asarray(morphology.beat_values[name], dtype=np.float64)
        beat_validity = np.asarray(morphology.beat_validity[name], dtype=bool)
        if beat_values.shape != peaks.shape or beat_validity.shape != peaks.shape:
            raise ValueError("morphology beat identities are not aligned to global peaks")
        eligible = np.zeros(peaks.shape, dtype=bool)
        for ordinal in range(1, peaks.size - 1):
            beat_start = 0.5 * (peaks[ordinal - 1] + peaks[ordinal])
            beat_stop = 0.5 * (peaks[ordinal] + peaks[ordinal + 1])
            cell = _cell_for_sample(timeline, peaks[ordinal])
            eligible[ordinal] = bool(
                beat_validity[ordinal]
                and start_sample <= peaks[ordinal] < stop_sample
                and cell is not None
                and cell.final_tier == "excellent"
                and cell.source_route == "direct"
                and _eligible_event_segment(
                    timeline,
                    start_sample=beat_start,
                    stop_sample=beat_stop,
                    source_route="direct",
                )
            )
        selected = beat_values[eligible]
        if selected.size >= 3:
            median = float(np.median(selected))
            offset = 2 * name_index
            output[offset] = median
            output[offset + 1] = float(np.median(np.abs(selected - median)))
            validity[offset:offset + 2] = np.isfinite(output[offset:offset + 2])
    return output, validity


def extract_window_features(
    direct_views: CanonicalSignalViews,
    *,
    plan: WindowPlan,
    timeline: RoutingTimeline,
    direct_pulse: PulseResult,
    direct_morphology: MorphologyResult,
    processed_pulse: PulseResult | None = None,
) -> WindowFeatureExtraction:
    """Build all complete 146-feature rows after final RoutingTimeline exists."""

    direct_views.validate()
    timeline.validate()
    base = extract_engineering_features(direct_views, plan=plan)
    starts = np.asarray(base.sequence.start_samples, dtype=np.int64)
    planned = plan.plan(direct_views.x_filter.shape[0], CANONICAL_FS_HZ)
    if len(planned) != starts.size or any(
        int(item.start_sample) != int(start)
        for item, start in zip(planned, starts)
    ):
        raise RuntimeError("engineering rows and complete-window plan diverged")
    names = window_feature_names()
    values = np.full((starts.size, len(names)), np.nan, dtype=np.float64)
    validity = np.zeros(values.shape, dtype=bool)
    row_mask = np.zeros(starts.size, dtype=bool)
    tiers: list[str] = []
    reasons: list[str] = []
    direct_rate_pulses = (direct_pulse,)
    mixed_rate_pulses = (
        direct_rate_pulses
        if processed_pulse is None
        else (direct_pulse, processed_pulse)
    )
    for row_index, item in enumerate(planned):
        start, stop = int(item.start_sample), int(item.end_sample)
        tier, eligible = matrix_row_route(timeline, start, stop)
        tiers.append(tier)
        if not eligible:
            reasons.append(f"row_{row_index}:routing_excluded")
            continue
        row_mask[row_index] = True
        if tier == "excellent":
            values[row_index, :115] = base.sequence.values[row_index]
            validity[row_index, :115] = base.value_validity[row_index]
            morph_values, morph_validity = _morphology_features(
                direct_pulse, direct_morphology, timeline, start, stop
            )
            values[row_index, 115:129] = morph_values
            validity[row_index, 115:129] = morph_validity
            rate_pulses = direct_rate_pulses
        else:
            rate_pulses = mixed_rate_pulses
        rate_values, rate_validity = _rate_features(
            rate_pulses, timeline, start, stop
        )
        values[row_index, 129:] = rate_values
        validity[row_index, 129:] = rate_validity
    sequence = EngineeringFeatureSequence(
        values=values,
        start_samples=starts,
        valid_row_mask=row_mask,
        channel_schema=names,
        schema_version=WINDOW_FEATURE_SCHEMA_VERSION,
    )
    result = WindowFeatureExtraction(
        sequence=sequence,
        value_validity=validity,
        row_tiers=tuple(tiers),
        reasons=tuple(reasons),
    )
    validate_window_feature_extraction(result, transformed=False)
    return result


def fit_fold_window_feature_transform(
    extractions: Iterable[WindowFeatureExtraction],
    *,
    fitted_on_participant_ids: Iterable[str],
    outer_train_participant_ids: Iterable[str],
    outer_oof_participant_ids: Iterable[str],
) -> FoldWindowFeatureTransform:
    """Fit median/(IQR/1.349), with population-SD then one fallback, on train only."""

    fitted = assert_training_only(
        fitted_on_participant_ids,
        outer_train_participant_ids,
        outer_oof_participant_ids,
    )
    items = tuple(extractions)
    if not items:
        raise ValueError("at least one outer-training matrix extraction is required")
    for item in items:
        validate_window_feature_extraction(item, transformed=False)
    matrix = np.vstack([item.sequence.values for item in items])
    valid = np.vstack([item.value_validity for item in items])
    row_valid = np.concatenate([item.sequence.valid_row_mask for item in items])
    center = np.zeros(146, dtype=np.float64)
    scale = np.ones(146, dtype=np.float64)
    count = np.zeros(146, dtype=np.int64)
    for column in range(146):
        selected = matrix[:, column][
            row_valid & valid[:, column] & np.isfinite(matrix[:, column])
        ]
        count[column] = selected.size
        if selected.size:
            center[column] = float(np.median(selected))
            q25, q75 = np.percentile(selected, (25.0, 75.0))
            candidate = float((q75 - q25) / 1.349)
            if not np.isfinite(candidate) or candidate <= 1e-8:
                candidate = float(np.std(selected, ddof=0))
            scale[column] = candidate if candidate > 1e-8 else 1.0
    return FoldWindowFeatureTransform(
        center=center,
        scale=scale,
        valid_count=count,
        feature_names=window_feature_names(),
        fitted_on_participant_ids=fitted,
    )


def transform_window_features(
    extraction: WindowFeatureExtraction,
    transform: FoldWindowFeatureTransform,
) -> WindowFeatureExtraction:
    """Apply a train-only transform and map unavailable values to neutral zero."""

    validate_window_feature_extraction(extraction, transformed=False)
    if (
        transform.feature_names != window_feature_names()
        or np.asarray(transform.center).shape != (146,)
        or np.asarray(transform.scale).shape != (146,)
        or not np.isfinite(transform.center).all()
        or not np.isfinite(transform.scale).all()
        or np.any(transform.scale <= 0.0)
    ):
        raise ValueError("window feature transform schema drift")
    valid = np.asarray(extraction.value_validity, dtype=bool)
    values = (
        np.asarray(extraction.sequence.values, dtype=np.float64) - transform.center
    ) / transform.scale
    values[~valid] = 0.0
    values[~np.asarray(extraction.sequence.valid_row_mask, dtype=bool)] = 0.0
    result = WindowFeatureExtraction(
        sequence=EngineeringFeatureSequence(
            values=values,
            start_samples=extraction.sequence.start_samples.copy(),
            valid_row_mask=extraction.sequence.valid_row_mask.copy(),
            channel_schema=extraction.sequence.channel_schema,
            schema_version=WINDOW_FEATURE_SCHEMA_VERSION + "+outer_train_robust_v1",
        ),
        value_validity=valid.copy(),
        row_tiers=extraction.row_tiers,
        reasons=extraction.reasons,
    )
    validate_window_feature_extraction(result, transformed=True)
    return result


def build_ordered_window_matrix(
    extraction: WindowFeatureExtraction,
    *,
    provenance: Mapping[str, object],
) -> OrderedFeatureMatrixV1:
    """Store one recording as finite [146,K_i] without cropping or padding."""

    validate_window_feature_extraction(extraction, transformed=True)
    values = np.asarray(extraction.sequence.values, dtype=np.float64)
    row_mask = np.asarray(extraction.sequence.valid_row_mask, dtype=bool)
    if values.shape[0] == 0 or not np.any(row_mask) or not np.isfinite(values).all():
        raise ValueError("matrix requires at least one finite route-eligible row")
    channel_schema = window_feature_names()
    validity = np.asarray(extraction.value_validity, dtype=bool)
    metadata = dict(provenance)
    metadata.update(
        {
            "matrix_k": int(values.shape[0]),
            "matrix_length_policy": "all_complete_windows_variable_k",
            "matrix_schema_version": ORDERED_WINDOW_MATRIX_SCHEMA_VERSION,
            "matrix_channel_schema_sha256": hashlib.sha256(
                "\n".join(channel_schema).encode("utf-8")
            ).hexdigest(),
            "feature_validity_sha256": hashlib.sha256(
                np.packbits(validity, bitorder="little").tobytes()
            ).hexdigest(),
            "validity_encoding": "provenance_only_not_predictor_channels_v1",
            "padding_policy": "none_at_record_storage_batch_only",
            "unavailable_after_transform": "outer_train_center_zero",
            "source_row_indices": list(range(values.shape[0])),
        }
    )
    return OrderedFeatureMatrixV1(
        values=values.T,
        row_mask=row_mask.copy(),
        channel_schema=channel_schema,
        context_schema=(),
        schema_version=ORDERED_WINDOW_MATRIX_SCHEMA_VERSION,
        provenance=metadata,
    )


__all__ = [
    "FoldWindowFeatureTransform",
    "INTERVAL_WINDOW_NAMES",
    "LOCAL_PRV_WINDOW_NAMES",
    "MORPHOLOGY_WINDOW_NAMES",
    "ORDERED_WINDOW_MATRIX_SCHEMA_VERSION",
    "RATE_WINDOW_NAMES",
    "SUCCESSIVE_WINDOW_NAMES",
    "WINDOW_FEATURE_SCHEMA_VERSION",
    "WindowFeatureExtraction",
    "build_ordered_window_matrix",
    "build_route_eligible_rate_pulse",
    "extract_window_features",
    "fit_fold_window_feature_transform",
    "transform_window_features",
    "validate_window_feature_extraction",
    "route_eligible_morphology_aggregates",
    "window_feature_names",
]
