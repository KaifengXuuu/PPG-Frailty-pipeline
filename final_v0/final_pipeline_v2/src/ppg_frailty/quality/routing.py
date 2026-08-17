"""V2 quality modes and signal-route constraints.

``off`` is the default and computes no SQI. ``diagnostics_only`` computes and
returns component evidence but is structurally forbidden from changing retention,
aggregation, artifact reduction, or prediction. ``route`` remains fail-closed
until a supervised routing artifact and policy are designed and approved.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

from ..contracts import (
    ArtifactReductionResult,
    QualityEndpoint,
    QualityResult,
    QualityState,
    RouteResult,
    RouteState,
    SignalRoute,
    SourceSignalView,
)
from .endpoint_sqi import SqiDiagnostics, evaluate_quality_diagnostics


class QualityMode(str, Enum):
    OFF = "off"
    DIAGNOSTICS_ONLY = "diagnostics_only"
    ROUTE = "route"


class QualityRoutingDisabledError(RuntimeError):
    """Raised when the deliberately unavailable supervised SQI router is requested."""


@dataclass(frozen=True)
class SegmentIntegrity:
    """Hard, segment-local integrity result evaluated before endpoint SQI."""

    pass_: bool
    segment_id: str
    start_sample: int
    end_sample: int
    reasons: tuple[str, ...] = ()

    def validate(self) -> None:
        if not str(self.segment_id).strip() or self.start_sample < 0:
            raise ValueError("segment integrity identity is invalid")
        if self.end_sample <= self.start_sample:
            raise ValueError("segment integrity bounds are empty or reversed")
        if not self.pass_ and not self.reasons:
            raise ValueError("hard-invalid integrity results require an explicit reason")


@dataclass(frozen=True)
class QualityModeOutcome:
    """Auditable outcome whose default modes cannot affect classification."""

    mode: QualityMode
    computed: bool
    result: QualityResult | SqiDiagnostics | Any | None
    classification_action: str
    affects_retention: bool
    affects_aggregation: bool
    affects_prediction: bool
    reasons: tuple[str, ...]

    def validate(self) -> None:
        if self.mode in {QualityMode.OFF, QualityMode.DIAGNOSTICS_ONLY}:
            if (
                self.classification_action != "keep_unchanged"
                or self.affects_retention
                or self.affects_aggregation
                or self.affects_prediction
            ):
                raise ValueError("non-routing quality mode changed classification behavior")
        if self.mode is QualityMode.OFF and (self.computed or self.result is not None):
            raise ValueError("quality.mode=off must not compute SQI")
        if self.mode is QualityMode.DIAGNOSTICS_ONLY and (
            not self.computed or self.result is None
        ):
            raise ValueError("diagnostics_only must retain its computed SQI result")


def resolve_quality_mode(value: QualityMode | str | None) -> QualityMode:
    """Resolve an explicit mode; omission means the frozen V2 default ``off``."""

    if value is None:
        return QualityMode.OFF
    if isinstance(value, QualityMode):
        return value
    try:
        return QualityMode(str(value).strip().lower())
    except ValueError as exc:
        raise ValueError("quality.mode must be off, diagnostics_only, or route") from exc


def quality_mode_from_config(config: Mapping[str, Any]) -> QualityMode:
    """Read the optional V2 quality mode without inventing routing parameters."""

    quality = config.get("quality", {})
    if not isinstance(quality, Mapping):
        raise ValueError("config['quality'] must be a mapping")
    return resolve_quality_mode(quality.get("mode", QualityMode.OFF.value))


def run_quality_mode(
    values: Any,
    *,
    mode: QualityMode | str | None = QualityMode.OFF,
    evaluator: Callable[..., QualityResult | SqiDiagnostics | Any] = evaluate_quality_diagnostics,
    **evaluation_kwargs: Any,
) -> QualityModeOutcome:
    """Run only the confirmed semantics; the supervised router is not implemented."""

    resolved = resolve_quality_mode(mode)
    if resolved is QualityMode.ROUTE:
        raise QualityRoutingDisabledError(
            "quality.mode=route is disabled until a supervised routing artifact, "
            "frozen component weights, and thresholds are approved"
        )
    if resolved is QualityMode.OFF:
        outcome = QualityModeOutcome(
            mode=resolved,
            computed=False,
            result=None,
            classification_action="keep_unchanged",
            affects_retention=False,
            affects_aggregation=False,
            affects_prediction=False,
            reasons=("sqi_disabled_v2_default",),
        )
    else:
        result = evaluator(values, **evaluation_kwargs)
        outcome = QualityModeOutcome(
            mode=resolved,
            computed=True,
            result=result,
            classification_action="keep_unchanged",
            affects_retention=False,
            affects_aggregation=False,
            affects_prediction=False,
            reasons=("diagnostics_saved_separately_not_a_predictor_or_gate",),
        )
    outcome.validate()
    return outcome


def route_segment_pre_reduction(
    integrity: SegmentIntegrity,
    *,
    q_pre: QualityResult | None,
    recoverable_motion: bool,
    reducer_enabled: bool,
) -> RouteResult:
    """Resolve A1 through the explicit reducer-candidate transition.

    This function consumes already-fitted endpoint states only. It does not fit or
    activate a threshold, so operational supervised quality routing stays disabled.
    """

    integrity.validate()
    if not integrity.pass_:
        result = RouteResult(
            state=RouteState.HARD_INVALID,
            source_signal=None,
            segment_id=integrity.segment_id,
            start_sample=integrity.start_sample,
            end_sample=integrity.end_sample,
            reasons=integrity.reasons,
        )
        result.validate()
        return result
    if q_pre is None:
        raise ValueError("integrity-valid segments require direct endpoint quality")
    rate_pass = q_pre.q_rate.state is QualityState.PASS
    shape_pass = q_pre.shape_endpoint.state is QualityState.PASS
    if rate_pass:
        state = RouteState.FULL_DIRECT if shape_pass else RouteState.RATE_ONLY_DIRECT
        result = RouteResult(
            state=state,
            source_signal=SourceSignalView.X_FILTER,
            segment_id=integrity.segment_id,
            start_sample=integrity.start_sample,
            end_sample=integrity.end_sample,
            q_pre=q_pre,
            reasons=(
                ()
                if shape_pass
                else ("q_rate_pass_q_shape_not_pass_rate_only_direct",)
            ),
        )
        result.validate()
        return result
    if not recoverable_motion or not reducer_enabled:
        reason = (
            "q_rate_fail_not_recoverable"
            if not recoverable_motion
            else "q_rate_fail_reducer_disabled"
        )
        result = RouteResult(
            state=RouteState.DEGRADED_DROP,
            source_signal=None,
            segment_id=integrity.segment_id,
            start_sample=integrity.start_sample,
            end_sample=integrity.end_sample,
            q_pre=q_pre,
            reasons=(reason,),
        )
        result.validate()
        return result
    result = RouteResult(
        state=RouteState.RATE_RECOVERY_CANDIDATE,
        source_signal=None,
        segment_id=integrity.segment_id,
        start_sample=integrity.start_sample,
        end_sample=integrity.end_sample,
        q_pre=q_pre,
        reasons=("q_rate_fail_recoverable_motion",),
    )
    result.validate()
    return result


def finalize_rate_recovery(
    candidate: RouteResult,
    *,
    reduction: ArtifactReductionResult,
    q_rate_post: QualityEndpoint | None,
) -> RouteResult:
    """Resolve A2 after one non-identity reducer attempt and Q_rate-only reassessment."""

    candidate.validate()
    if candidate.state is not RouteState.RATE_RECOVERY_CANDIDATE:
        raise ValueError("only rate_recovery_candidate can enter reducer finalization")
    reduction_succeeded = (
        reduction.status == "success"
        and reduction.x_ar is not None
        and not reduction.is_identity
    )
    if not reduction_succeeded:
        reasons = tuple(
            dict.fromkeys(
                (
                    *candidate.reasons,
                    *reduction.reasons,
                    (
                        "identity_reducer_cannot_create_rate_recovery"
                        if reduction.is_identity
                        else "artifact_reducer_failed"
                    ),
                )
            )
        )
        result = RouteResult(
            state=RouteState.REJECTED_AFTER_REDUCTION,
            source_signal=None,
            segment_id=candidate.segment_id,
            start_sample=candidate.start_sample,
            end_sample=candidate.end_sample,
            q_pre=candidate.q_pre,
            reducer_name=reduction.reducer_id,
            reducer_status=reduction.status,
            reasons=reasons,
        )
        result.validate()
        return result
    if q_rate_post is None:
        raise ValueError("successful non-identity reduction requires Q_rate_post")
    passed = q_rate_post.state is QualityState.PASS
    result = RouteResult(
        state=(
            RouteState.RATE_ONLY_PROCESSED
            if passed
            else RouteState.REJECTED_AFTER_REDUCTION
        ),
        source_signal=SourceSignalView.X_AR if passed else None,
        segment_id=candidate.segment_id,
        start_sample=candidate.start_sample,
        end_sample=candidate.end_sample,
        q_pre=candidate.q_pre,
        q_post=q_rate_post,
        reducer_name=reduction.reducer_id,
        reducer_status=reduction.status,
        reasons=tuple(
            dict.fromkeys(
                (
                    *candidate.reasons,
                    "q_morph_post_not_applicable",
                    "q_rate_post_pass" if passed else "q_rate_post_not_pass",
                )
            )
        ),
    )
    result.validate()
    return result


def assert_quality_route(result: QualityResult, route: SignalRoute | str) -> None:
    """Enforce not-applicable morphology for non-identity artifact outputs."""

    resolved = route if isinstance(route, SignalRoute) else SignalRoute(route)
    result.validate_for_route(resolved)


__all__ = [
    "QualityMode",
    "QualityModeOutcome",
    "QualityRoutingDisabledError",
    "SegmentIntegrity",
    "assert_quality_route",
    "finalize_rate_recovery",
    "quality_mode_from_config",
    "resolve_quality_mode",
    "route_segment_pre_reduction",
    "run_quality_mode",
]
