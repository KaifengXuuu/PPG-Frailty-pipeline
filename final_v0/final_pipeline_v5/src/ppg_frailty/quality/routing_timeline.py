"""Window-level SQI/motion routing on the canonical 400 Hz time grid.

This module owns time boundaries and route decisions only.  It never creates a
hybrid waveform and never fits SQI, motion, or reducer parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from ..contracts import RoutingCell, RoutingTimeline, RoutingWindow
from .routing import QualityTier, route_quality_tier

ROUTING_TIMELINE_SCHEMA = "ppg_frailty.routing_timeline.v1"
ROUTING_FS_HZ = 400.0
ROUTING_WINDOW_SECONDS = 8.0
ROUTING_HOP_SECONDS = 2.0


@dataclass(frozen=True)
class RoutingEvidence:
    """Evidence and final route for one native routing window."""

    window: RoutingWindow
    sqi_mode: str
    sqi_assessed: bool
    direct_q_rate_score: float | None = None
    direct_q_rate_state: str | None = None
    direct_q_morph_score: float | None = None
    direct_q_morph_state: str | None = None
    motion_detector_enabled: bool = False
    motion_probability: float | None = None
    motion_threshold: float | None = None
    motion_state: str = "off"
    structural_failure: bool = False
    denoiser_enabled: bool = False
    denoiser_requested: bool = False
    denoiser_status: str = "not_requested"
    post_q_rate_score: float | None = None
    post_q_rate_state: str | None = None
    pre_route_tier: str = "pending"
    final_tier: str = "pending"
    source_route: str = "none"
    source_view: str = "none"
    reason_codes: tuple[str, ...] = ()


def build_routing_windows(
    record_id: str,
    n_samples: int,
    *,
    fs_hz: float = ROUTING_FS_HZ,
    window_s: float = ROUTING_WINDOW_SECONDS,
    hop_s: float = ROUTING_HOP_SECONDS,
) -> tuple[RoutingWindow, ...]:
    """Return all complete shared SQI/motion evidence windows."""

    if not record_id:
        raise ValueError("routing windows require record_id")
    if float(fs_hz) != ROUTING_FS_HZ:
        raise ValueError("routing evidence must use the canonical 400 Hz grid")
    if n_samples <= 0 or window_s <= 0.0 or hop_s <= 0.0:
        raise ValueError("routing length, window and hop must be positive")
    window_samples = int(round(window_s * fs_hz))
    hop_samples = int(round(hop_s * fs_hz))
    if not np.isclose(window_samples / fs_hz, window_s) or not np.isclose(hop_samples / fs_hz, hop_s):
        raise ValueError("routing window/hop must map exactly to the 400 Hz grid")
    if n_samples < window_samples:
        return ()
    starts = np.arange(0, n_samples - window_samples + 1, hop_samples, dtype=np.int64)
    return tuple(
        RoutingWindow(
            record_id=record_id,
            routing_window_id=f"{record_id}::routing_{index:06d}",
            start_s=float(start / fs_hz),
            stop_s=float((start + window_samples) / fs_hz),
            centre_s=float((start + window_samples / 2.0) / fs_hz),
            start_sample_400=int(start),
            stop_sample_400=int(start + window_samples),
        ) for index, start in enumerate(starts))


def resolve_routing_evidence(
    evidence: RoutingEvidence,
    *,
    role: str,
    allow_rate_feature_recovery_without_direct_sqi: bool = False,
) -> RoutingEvidence:
    """Apply the authoritative truth table and the post-reducer promotion rule.

    ``allow_rate_feature_recovery_without_direct_sqi`` is an explicit
    representation capability, not a hidden SQI switch.  It permits a
    motion-high/SQI-off window to become Acceptable only after a successful
    reducer and a passing post-reduction Q_rate assessment.  Raw, matrix and
    fusion callers leave it false; the feature-vector rate-feature route opts
    in explicitly.
    """

    mode = str(evidence.sqi_mode)
    if mode not in {"off", "diagnostics_only", "route"}:
        raise ValueError("unknown SQI mode in routing evidence")
    reasons = list(evidence.reason_codes)
    if evidence.structural_failure:
        return replace(
            evidence,
            pre_route_tier=QualityTier.EXCLUDED.value,
            final_tier=QualityTier.EXCLUDED.value,
            source_route="none",
            source_view="none",
            denoiser_requested=False,
            reason_codes=tuple(dict.fromkeys((*reasons, "structural_hard_failure"))),
        )

    if evidence.motion_detector_enabled:
        motion_high = False if evidence.motion_state == "low" else True if evidence.motion_state == "high" else None
    else:
        if evidence.motion_state != "off":
            raise ValueError("disabled motion detector must expose motion_state=off")
        motion_high = None

    if mode == "route":
        decision = route_quality_tier(
            sqi_enabled=True,
            q_rate_state=evidence.direct_q_rate_state,
            q_morph_state=evidence.direct_q_morph_state,
            motion_enabled=evidence.motion_detector_enabled,
            motion_high=motion_high,
        )
        pre_tier = decision.tier
        reasons.extend(decision.reasons)
    else:
        # ``off`` and ``diagnostics_only`` must not introduce an implicit
        # protocol-role gate.  Every role already selected by the input config
        # is Excellent when motion is off; with motion enabled, low/high maps
        # to Excellent/Unfit.  Keep this path on the same authoritative truth
        # table as the public routing helper.
        decision = route_quality_tier(
            sqi_enabled=False,
            q_rate_state=None,
            q_morph_state=None,
            motion_enabled=evidence.motion_detector_enabled,
            motion_high=motion_high,
        )
        pre_tier = decision.tier
        reasons.extend(decision.reasons)

    denoiser_requested = bool(evidence.denoiser_enabled and pre_tier is QualityTier.UNFIT
                              and (mode == "route" or evidence.motion_detector_enabled))
    post_reduction_promotion_authorized = bool(mode == "route"
                                               or (mode == "off" and allow_rate_feature_recovery_without_direct_sqi))
    if pre_tier is QualityTier.EXCELLENT:
        final_tier, source_route, source_view = pre_tier, "direct", "x_filter_400"
    elif pre_tier is QualityTier.ACCEPTABLE:
        final_tier, source_route, source_view = pre_tier, "direct", "x_filter_400"
    elif (denoiser_requested and post_reduction_promotion_authorized and evidence.denoiser_status == "success"
          and evidence.post_q_rate_state == "pass"):
        final_tier = QualityTier.ACCEPTABLE
        source_route, source_view = "processed", "x_ar_400"
        reasons.append("post_q_rate_pass_promoted_acceptable_processed")
    else:
        final_tier = QualityTier.EXCLUDED
        source_route, source_view = "none", "none"
        if denoiser_requested and not post_reduction_promotion_authorized:
            reasons.append("denoiser_diagnostic_only_without_sqi_route")
        elif denoiser_requested:
            reasons.append("post_reducer_not_promoted")
        else:
            reasons.append("unfit_without_recovery")

    return replace(
        evidence,
        sqi_assessed=bool(mode in {"diagnostics_only", "route"}),
        denoiser_requested=denoiser_requested,
        pre_route_tier=pre_tier.value,
        final_tier=final_tier.value,
        source_route=source_route,
        source_view=source_view,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def _excluded_edge_cell(
    *,
    record_id: str,
    participant_id: str,
    role: str,
    start: int,
    stop: int,
    fs_hz: float,
    cell_index: int,
    config_sha256: str,
) -> RoutingCell:
    return RoutingCell(
        record_id=record_id,
        participant_id=participant_id,
        role=role,
        routing_window_id="evidence_unavailable",
        cell_id=f"{record_id}::cell_{cell_index:06d}",
        cell_start_s=float(start / fs_hz),
        cell_stop_s=float(stop / fs_hz),
        start_sample_400=start,
        stop_sample_400=stop,
        sqi_mode="not_assessed",
        sqi_assessed=False,
        direct_q_rate_score=None,
        direct_q_rate_state=None,
        direct_q_morph_score=None,
        direct_q_morph_state=None,
        motion_detector_enabled=False,
        motion_probability=None,
        motion_threshold=None,
        motion_state="unavailable",
        pre_route_tier=QualityTier.EXCLUDED.value,
        denoiser_enabled=False,
        denoiser_requested=False,
        denoiser_status="not_requested",
        post_q_rate_score=None,
        post_q_rate_state=None,
        final_tier=QualityTier.EXCLUDED.value,
        source_route="none",
        source_view="none",
        reason_codes=("routing_evidence_unavailable", ),
        config_sha256=config_sha256,
        sqi_calibrator_sha256=None,
        motion_model_sha256=None,
        motion_input_schema_sha256=None,
        reducer_sha256=None,
    )


def build_routing_timeline(
    *,
    record_id: str,
    participant_id: str,
    role: str,
    n_samples: int,
    evidence: Iterable[RoutingEvidence],
    config_sha256: str,
    sqi_calibrator_sha256: str | None = None,
    motion_model_sha256: str | None = None,
    motion_input_schema_sha256: str | None = None,
    reducer_sha256: str | None = None,
    fs_hz: float = ROUTING_FS_HZ,
) -> RoutingTimeline:
    """Convert overlapping evidence windows into unique centre-midpoint cells."""

    if float(fs_hz) != ROUTING_FS_HZ:
        raise ValueError("routing timeline requires the canonical 400 Hz grid")
    rows = tuple(evidence)
    windows = tuple(row.window for row in rows)
    if any(window.record_id != record_id for window in windows):
        raise ValueError("routing evidence record identity drift")
    if any(left.start_sample_400 >= right.start_sample_400 for left, right in zip(windows, windows[1:])):
        raise ValueError("routing evidence windows must be chronological and unique")

    cells: list[RoutingCell] = []
    if not rows:
        cells.append(
            _excluded_edge_cell(
                record_id=record_id,
                participant_id=participant_id,
                role=role,
                start=0,
                stop=n_samples,
                fs_hz=fs_hz,
                cell_index=0,
                config_sha256=config_sha256,
            ))
    else:
        centres = np.asarray(
            [(row.window.start_sample_400 + row.window.stop_sample_400) / 2.0 for row in rows],
            dtype=np.float64,
        )
        boundaries = np.rint((centres[:-1] + centres[1:]) / 2.0).astype(np.int64)
        starts = np.concatenate(([rows[0].window.start_sample_400], boundaries)).astype(np.int64)
        stops = np.concatenate((boundaries, [rows[-1].window.stop_sample_400])).astype(np.int64)
        if starts[0] > 0:
            cells.append(
                _excluded_edge_cell(
                    record_id=record_id,
                    participant_id=participant_id,
                    role=role,
                    start=0,
                    stop=int(starts[0]),
                    fs_hz=fs_hz,
                    cell_index=0,
                    config_sha256=config_sha256,
                ))
        for row, start, stop in zip(rows, starts, stops):
            if start >= stop:
                raise ValueError("routing ownership cell has zero support")
            cells.append(
                RoutingCell(
                    record_id=record_id,
                    participant_id=participant_id,
                    role=role,
                    routing_window_id=row.window.routing_window_id,
                    cell_id=f"{record_id}::cell_{len(cells):06d}",
                    cell_start_s=float(start / fs_hz),
                    cell_stop_s=float(stop / fs_hz),
                    start_sample_400=int(start),
                    stop_sample_400=int(stop),
                    sqi_mode=row.sqi_mode,
                    sqi_assessed=row.sqi_assessed,
                    direct_q_rate_score=row.direct_q_rate_score,
                    direct_q_rate_state=row.direct_q_rate_state,
                    direct_q_morph_score=row.direct_q_morph_score,
                    direct_q_morph_state=row.direct_q_morph_state,
                    motion_detector_enabled=row.motion_detector_enabled,
                    motion_probability=row.motion_probability,
                    motion_threshold=row.motion_threshold,
                    motion_state=row.motion_state,
                    pre_route_tier=row.pre_route_tier,
                    denoiser_enabled=row.denoiser_enabled,
                    denoiser_requested=row.denoiser_requested,
                    denoiser_status=row.denoiser_status,
                    post_q_rate_score=row.post_q_rate_score,
                    post_q_rate_state=row.post_q_rate_state,
                    final_tier=row.final_tier,
                    source_route=row.source_route,
                    source_view=row.source_view,
                    reason_codes=row.reason_codes,
                    config_sha256=config_sha256,
                    sqi_calibrator_sha256=sqi_calibrator_sha256,
                    motion_model_sha256=motion_model_sha256,
                    motion_input_schema_sha256=motion_input_schema_sha256,
                    reducer_sha256=reducer_sha256,
                ))
        if stops[-1] < n_samples:
            cells.append(
                _excluded_edge_cell(
                    record_id=record_id,
                    participant_id=participant_id,
                    role=role,
                    start=int(stops[-1]),
                    stop=n_samples,
                    fs_hz=fs_hz,
                    cell_index=len(cells),
                    config_sha256=config_sha256,
                ))

    timeline = RoutingTimeline(
        record_id=record_id,
        participant_id=participant_id,
        role=role,
        fs_hz=float(fs_hz),
        n_samples=int(n_samples),
        windows=windows,
        cells=tuple(cells),
        schema_version=ROUTING_TIMELINE_SCHEMA,
    )
    timeline.validate()
    return timeline


def overlapping_cells(timeline: RoutingTimeline, start_sample: int, stop_sample: int) -> tuple[RoutingCell, ...]:
    """Return ownership cells with positive overlap with a target interval."""

    timeline.validate()
    if not 0 <= start_sample < stop_sample <= timeline.n_samples:
        raise ValueError("routing query bounds are outside the recording")
    return tuple(cell for cell in timeline.cells
                 if cell.start_sample_400 < stop_sample and cell.stop_sample_400 > start_sample)


def matrix_row_route(timeline: RoutingTimeline, start_sample: int, stop_sample: int) -> tuple[str, bool]:
    """Resolve Excellent/Acceptable/excluded eligibility for one matrix row."""

    cells = overlapping_cells(timeline, start_sample, stop_sample)
    if not cells or any(cell.final_tier == QualityTier.EXCLUDED.value for cell in cells):
        return QualityTier.EXCLUDED.value, False
    if all(cell.final_tier == QualityTier.EXCELLENT.value and cell.source_route == "direct" for cell in cells):
        return QualityTier.EXCELLENT.value, True
    if any(cell.final_tier == QualityTier.ACCEPTABLE.value for cell in cells):
        return QualityTier.ACCEPTABLE.value, True
    return QualityTier.EXCLUDED.value, False


__all__ = [
    "ROUTING_FS_HZ",
    "ROUTING_HOP_SECONDS",
    "ROUTING_TIMELINE_SCHEMA",
    "ROUTING_WINDOW_SECONDS",
    "RoutingEvidence",
    "build_routing_timeline",
    "build_routing_windows",
    "matrix_row_route",
    "overlapping_cells",
    "resolve_routing_evidence",
]
