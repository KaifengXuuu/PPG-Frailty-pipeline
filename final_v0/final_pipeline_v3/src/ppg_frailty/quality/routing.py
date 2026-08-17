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

from ..contracts import QualityResult, SignalRoute
from .endpoint_sqi import SqiDiagnostics, evaluate_quality_diagnostics


class QualityMode(str, Enum):
    OFF = "off"
    DIAGNOSTICS_ONLY = "diagnostics_only"
    ROUTE = "route"


class QualityRoutingDisabledError(RuntimeError):
    """Raised when the deliberately unavailable supervised SQI router is requested."""


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


def assert_quality_route(result: QualityResult, route: SignalRoute | str) -> None:
    """Enforce not-applicable morphology for non-identity artifact outputs."""

    resolved = route if isinstance(route, SignalRoute) else SignalRoute(route)
    result.validate_for_route(resolved)


__all__ = [
    "QualityMode",
    "QualityModeOutcome",
    "QualityRoutingDisabledError",
    "assert_quality_route",
    "quality_mode_from_config",
    "resolve_quality_mode",
    "run_quality_mode",
]
