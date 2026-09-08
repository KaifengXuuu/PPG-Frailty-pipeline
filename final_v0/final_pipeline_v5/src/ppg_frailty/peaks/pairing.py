"""One-to-one event and canonical RED/IR beat-cycle pairing contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..contracts import PulseResult, SignalRoute

DUAL_WAVELENGTH_PAIRING_SCHEMA_VERSION = "dual_wavelength_midpoint_cycle_pairing_v3"


@dataclass(frozen=True)
class EventMatchMetrics:
    """事件匹配量化结果 / Quantitative event-matching result."""

    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    timing_mae_s: float | None


@dataclass(frozen=True)
class BeatPairAudit:
    """One retained audit row for a paired, unpaired, or invalid pulse event."""

    reference_wavelength: str
    secondary_wavelength: str
    reference_peak_ordinal: int | None
    reference_peak_sample: int | None
    red_peak_ordinal: int | None
    red_peak_sample: int | None
    ir_peak_ordinal: int | None
    ir_peak_sample: int | None
    lag_samples_ir_minus_red: int | None
    lag_s_ir_minus_red: float | None
    pair_valid: bool
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class BeatPairingResult:
    """Auditable deterministic RED/IR pairing on reference beat cycles."""

    detector_id: str
    reference_wavelength: str
    secondary_wavelength: str
    reference_score: float
    reference_coverage: float
    secondary_score: float
    secondary_coverage: float
    red_detection_run_id: str
    ir_detection_run_id: str
    red_detector_version: str
    ir_detector_version: str
    red_selected_polarity: int
    ir_selected_polarity: int
    red_block_hri_provenance_hash: str
    ir_block_hri_provenance_hash: str
    rows: tuple[BeatPairAudit, ...]
    schema_version: str = DUAL_WAVELENGTH_PAIRING_SCHEMA_VERSION
    reference_selection_rule: str = "highest_detector_score_then_coverage_then_red"
    cycle_interval_policy: str = "midpoint_left_inclusive_right_exclusive"
    ambiguity_tie_break: str = "nearest_sample_then_earlier_sample_then_lower_peak_ordinal"

    @property
    def paired_rows(self) -> tuple[BeatPairAudit, ...]:
        """Return valid pairs in chronological reference order."""

        return tuple(
            sorted(
                (row for row in self.rows if row.pair_valid),
                key=lambda row: (
                    int(row.reference_peak_sample),
                    int(row.red_peak_sample),
                    int(row.ir_peak_sample),
                ),
            ))


def _validated_pulses(pulses: Mapping[str, PulseResult], ) -> dict[str, PulseResult]:
    """Validate the two independent PulseResults required by formal pairing."""

    normalized = {str(name).upper(): pulse for name, pulse in pulses.items()}
    if set(normalized) != {"RED", "IR"}:
        raise ValueError("dual-wavelength pairing requires exactly RED and IR PulseResults")
    for wavelength, pulse in normalized.items():
        pulse.validate_identity()
        if str(pulse.wavelength).upper() != wavelength:
            raise ValueError("PulseResult wavelength disagrees with its mapping key")
        peaks = np.asarray(pulse.peaks, dtype=np.int64)
        accepted = np.asarray(pulse.accepted_peak_mask, dtype=bool)
        if peaks.ndim != 1 or peaks.shape != accepted.shape:
            raise ValueError("PulseResult peaks and accepted mask must be aligned vectors")
        if peaks.size and np.any(np.diff(peaks) <= 0):
            raise ValueError("PulseResult peaks must be strictly chronological")
    detector_ids = {str(pulse.detector_id).strip() for pulse in normalized.values()}
    if len(detector_ids) != 1 or not next(iter(detector_ids), ""):
        raise ValueError("RED and IR must use the same explicit detector_id")
    routes = {SignalRoute(pulse.source_route) for pulse in normalized.values()}
    if len(routes) != 1:
        raise ValueError("RED and IR PulseResults must share one signal route")
    return normalized


def _score_and_coverage(pulse: PulseResult) -> tuple[float, float]:
    """Read, rather than reconstruct, the detector's persisted selection evidence."""

    score = float(pulse.detector_score)
    coverage = float(pulse.detector_coverage)
    if not np.isfinite(score) or not np.isfinite(coverage):
        raise ValueError("formal RED/IR selection requires persisted finite score/coverage")
    return score, coverage


def select_reference_wavelength(pulses: Mapping[str, PulseResult]) -> str:
    """Select by persisted score, then coverage, with a final RED tie-break."""

    normalized = _validated_pulses(pulses)
    evidence = {wavelength: _score_and_coverage(pulse) for wavelength, pulse in normalized.items()}
    return max(
        ("RED", "IR"),
        key=lambda wavelength: (
            evidence[wavelength][0],
            evidence[wavelength][1],
            wavelength == "RED",
        ),
    )


def _peak_ordinals(pulse: PulseResult) -> np.ndarray:
    """Return persisted original-grid peak ordinals with a strict shape check."""

    ordinals = np.asarray(pulse.peak_ordinals, dtype=np.int64)
    peaks = np.asarray(pulse.peaks, dtype=np.int64)
    if ordinals.shape != peaks.shape:
        raise ValueError("PulseResult peak_ordinals must align one-to-one with peaks")
    if ordinals.size and np.unique(ordinals).size != ordinals.size:
        raise ValueError("PulseResult peak_ordinals must be unique")
    return ordinals


def pair_dual_wavelength_beats(
    pulses: Mapping[str, PulseResult],
    *,
    fs_hz: float,
) -> BeatPairingResult:
    """Pair accepted peaks inside non-overlapping reference midpoint cycles.

    No numerical tolerance is introduced. Every peak remains represented either in
    a reference-cycle row or in an explicit secondary/unaccepted audit row.
    """

    if not np.isfinite(fs_hz) or fs_hz <= 0.0:
        raise ValueError("fs_hz must be finite and positive")
    normalized = _validated_pulses(pulses)
    reference_wavelength = select_reference_wavelength(normalized)
    secondary_wavelength = "IR" if reference_wavelength == "RED" else "RED"
    reference = normalized[reference_wavelength]
    secondary = normalized[secondary_wavelength]
    reference_score, reference_coverage = _score_and_coverage(reference)
    secondary_score, secondary_coverage = _score_and_coverage(secondary)

    reference_peaks = np.asarray(reference.peaks, dtype=np.int64)
    reference_accepted = np.asarray(reference.accepted_peak_mask, dtype=bool)
    reference_ordinals = _peak_ordinals(reference)
    secondary_peaks = np.asarray(secondary.peaks, dtype=np.int64)
    secondary_accepted = np.asarray(secondary.accepted_peak_mask, dtype=bool)
    secondary_ordinals = _peak_ordinals(secondary)
    accepted_reference_positions = np.flatnonzero(reference_accepted)
    used_secondary_positions: set[int] = set()
    rows: list[BeatPairAudit] = []

    red_ordinals = _peak_ordinals(normalized["RED"])
    ir_ordinals = _peak_ordinals(normalized["IR"])

    def row_for(
        *,
        reference_position: int | None,
        secondary_position: int | None,
        pair_valid: bool,
        reasons: tuple[str, ...],
    ) -> BeatPairAudit:
        red_position = reference_position if reference_wavelength == "RED" else secondary_position
        ir_position = reference_position if reference_wavelength == "IR" else secondary_position
        red_pulse = normalized["RED"]
        ir_pulse = normalized["IR"]
        red_sample = int(red_pulse.peaks[red_position]) if red_position is not None else None
        ir_sample = int(ir_pulse.peaks[ir_position]) if ir_position is not None else None
        lag_samples = int(ir_sample - red_sample) if red_sample is not None and ir_sample is not None else None
        return BeatPairAudit(
            reference_wavelength=reference_wavelength,
            secondary_wavelength=secondary_wavelength,
            reference_peak_ordinal=(int(reference_ordinals[reference_position])
                                    if reference_position is not None else None),
            reference_peak_sample=(int(reference_peaks[reference_position])
                                   if reference_position is not None else None),
            red_peak_ordinal=(int(red_ordinals[red_position]) if red_position is not None else None),
            red_peak_sample=red_sample,
            ir_peak_ordinal=(int(ir_ordinals[ir_position]) if ir_position is not None else None),
            ir_peak_sample=ir_sample,
            lag_samples_ir_minus_red=lag_samples,
            lag_s_ir_minus_red=(float(lag_samples / fs_hz) if lag_samples is not None else None),
            pair_valid=bool(pair_valid),
            reason_codes=reasons,
        )

    accepted_reference_rank = {
        int(position): rank
        for rank, position in enumerate(accepted_reference_positions.tolist())
    }
    for reference_position in range(reference_peaks.size):
        if not reference_accepted[reference_position]:
            rows.append(
                row_for(
                    reference_position=reference_position,
                    secondary_position=None,
                    pair_valid=False,
                    reasons=("reference_peak_rejected", ),
                ))
            continue
        rank = accepted_reference_rank[reference_position]
        if rank == 0 or rank == accepted_reference_positions.size - 1:
            rows.append(
                row_for(
                    reference_position=reference_position,
                    secondary_position=None,
                    pair_valid=False,
                    reasons=("reference_boundary_has_no_complete_cycle", ),
                ))
            continue
        previous_position = int(accepted_reference_positions[rank - 1])
        following_position = int(accepted_reference_positions[rank + 1])
        left_midpoint = (float(reference_peaks[previous_position]) + float(reference_peaks[reference_position])) / 2.0
        right_midpoint = (float(reference_peaks[reference_position]) + float(reference_peaks[following_position])) / 2.0
        candidates = np.flatnonzero(secondary_accepted
                                    & (secondary_peaks.astype(np.float64) >= left_midpoint)
                                    & (secondary_peaks.astype(np.float64) < right_midpoint))
        candidates = np.asarray(
            [int(position) for position in candidates if int(position) not in used_secondary_positions],
            dtype=np.int64,
        )
        if not candidates.size:
            rows.append(
                row_for(
                    reference_position=reference_position,
                    secondary_position=None,
                    pair_valid=False,
                    reasons=("no_secondary_peak_in_reference_cycle", ),
                ))
            continue
        chosen = min(
            candidates.tolist(),
            key=lambda position: (
                abs(int(secondary_peaks[position]) - int(reference_peaks[reference_position])),
                int(secondary_peaks[position]),
                int(secondary_ordinals[position]),
            ),
        )
        used_secondary_positions.add(int(chosen))
        reasons = ("paired", "multiple_secondary_candidates_nearest_selected") if candidates.size > 1 else ("paired", )
        rows.append(
            row_for(
                reference_position=reference_position,
                secondary_position=int(chosen),
                pair_valid=True,
                reasons=reasons,
            ))

    for secondary_position in range(secondary_peaks.size):
        if secondary_position in used_secondary_positions:
            continue
        reason = "secondary_peak_unpaired" if secondary_accepted[secondary_position] else "secondary_peak_rejected"
        rows.append(
            row_for(
                reference_position=None,
                secondary_position=secondary_position,
                pair_valid=False,
                reasons=(reason, ),
            ))

    rows.sort(key=lambda row: (
        min(sample for sample in (row.red_peak_sample, row.ir_peak_sample) if sample is not None),
        row.reference_peak_sample is None,
        row.red_peak_sample is None,
        row.ir_peak_sample is None,
    ))
    paired = [row for row in rows if row.pair_valid]
    if any(current.reference_peak_sample >= following.reference_peak_sample
           or current.red_peak_sample >= following.red_peak_sample or current.ir_peak_sample >= following.ir_peak_sample
           for current, following in zip(paired, paired[1:])):
        raise RuntimeError("dual-wavelength pairing lost monotonic one-to-one order")
    return BeatPairingResult(
        detector_id=str(reference.detector_id),
        reference_wavelength=reference_wavelength,
        secondary_wavelength=secondary_wavelength,
        reference_score=reference_score,
        reference_coverage=reference_coverage,
        secondary_score=secondary_score,
        secondary_coverage=secondary_coverage,
        red_detection_run_id=str(normalized["RED"].detection_run_id),
        ir_detection_run_id=str(normalized["IR"].detection_run_id),
        red_detector_version=str(normalized["RED"].detector_version),
        ir_detector_version=str(normalized["IR"].detector_version),
        red_selected_polarity=int(normalized["RED"].selected_polarity),
        ir_selected_polarity=int(normalized["IR"].selected_polarity),
        red_block_hri_provenance_hash=str(normalized["RED"].block_hri_provenance_hash),
        ir_block_hri_provenance_hash=str(normalized["IR"].block_hri_provenance_hash),
        rows=tuple(rows),
    )


def match_events(reference_s: np.ndarray, predicted_s: np.ndarray, *, tolerance_s: float) -> EventMatchMetrics:
    """按时间排序贪心一对一匹配 / Greedy chronological one-to-one matching."""

    reference = np.sort(np.asarray(reference_s, dtype=np.float64))
    predicted = np.sort(np.asarray(predicted_s, dtype=np.float64))
    if reference.ndim != 1 or predicted.ndim != 1 or tolerance_s <= 0.0:
        raise ValueError("events must be one-dimensional and tolerance_s positive")
    if not np.isfinite(reference).all() or not np.isfinite(predicted).all():
        raise ValueError("event timestamps must be finite")
    used = np.zeros(predicted.size, dtype=bool)
    errors: list[float] = []
    for event in reference:
        candidates = np.flatnonzero((~used) & (np.abs(predicted - event) <= tolerance_s))
        if candidates.size:
            chosen = int(candidates[np.argmin(np.abs(predicted[candidates] - event))])
            used[chosen] = True
            errors.append(abs(float(predicted[chosen] - event)))
    tp = len(errors)
    fp = int(predicted.size - tp)
    fn = int(reference.size - tp)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-15)
    return EventMatchMetrics(tp, fp, fn, float(precision), float(recall), float(f1),
                             float(np.mean(errors)) if errors else None)


__all__ = [
    "BeatPairAudit",
    "BeatPairingResult",
    "DUAL_WAVELENGTH_PAIRING_SCHEMA_VERSION",
    "EventMatchMetrics",
    "match_events",
    "pair_dual_wavelength_beats",
    "select_reference_wavelength",
]
