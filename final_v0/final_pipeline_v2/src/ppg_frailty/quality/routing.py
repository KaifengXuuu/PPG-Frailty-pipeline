"""V2 quality modes and signal-route constraints.

``off`` is the default and computes no SQI. ``diagnostics_only`` computes and
returns component evidence but is structurally forbidden from changing retention,
aggregation, artifact reduction, or prediction. ``route`` computes the configured
endpoint quality and hands the result to the explicit route state machine.  It is
an ordinary optional runtime module, not a readiness-gated special case.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

from ..contracts import (
    QualityResult,
    QualityState,
    SignalRoute,
)
from .endpoint_sqi import SqiDiagnostics, evaluate_quality_diagnostics


class QualityMode(str, Enum):
    OFF = "off"
    DIAGNOSTICS_ONLY = "diagnostics_only"
    ROUTE = "route"


class QualityTier(str, Enum):
    """Classifier input tier produced by the explicit SQI/motion state table."""

    EXCELLENT = "excellent"
    ACCEPTABLE = "acceptable"
    UNFIT = "unfit"
    EXCLUDED = "excluded"


@dataclass(frozen=True)
class QualityTierDecision:
    """One label-free tier decision; recovery remains a separate module."""

    tier: QualityTier
    sqi_enabled: bool
    motion_enabled: bool
    reasons: tuple[str, ...]

    @property
    def eligible_for_direct_input(self) -> bool:
        return self.tier in {QualityTier.EXCELLENT, QualityTier.ACCEPTABLE}


@dataclass(frozen=True)
class RouteModuleSwitches:
    """Three independent runtime switches used by the route orchestrator."""

    sqi_enabled: bool
    motion_detector_enabled: bool
    denoiser_enabled: bool


_SQI_TIER_TABLE: dict[tuple[bool, str], QualityTier] = {
    # q_morph pass/fail, motion state.  Q_rate failure is handled before lookup.
    (True, "off"): QualityTier.EXCELLENT,
    (True, "low"): QualityTier.EXCELLENT,
    (True, "high"): QualityTier.UNFIT,
    (False, "off"): QualityTier.ACCEPTABLE,
    (False, "low"): QualityTier.ACCEPTABLE,
    (False, "high"): QualityTier.UNFIT,
}


def _endpoint_pass(state: QualityState | str | None) -> bool:
    if state is None:
        return False
    try:
        resolved = state if isinstance(state, QualityState) else QualityState(str(state))
    except ValueError:
        return False
    return resolved is QualityState.PASS


def _motion_state(*, enabled: bool, high: bool | None) -> str:
    if not enabled:
        return "off"
    if high is None:
        return "unavailable"
    if not isinstance(high, bool):
        raise TypeError("motion_high must be boolean or None")
    return "high" if high else "low"


def route_quality_tier(
    *,
    sqi_enabled: bool,
    q_rate_state: QualityState | str | None,
    q_morph_state: QualityState | str | None,
    motion_enabled: bool,
    motion_high: bool | None,
) -> QualityTierDecision:
    """Apply the user-authoritative SQI/motion truth table.

    ``UNFIT`` is intentionally not converted to ``EXCLUDED`` here.  A caller may
    independently enable one denoiser attempt and then reassess Q_rate; this pure
    function neither runs nor selects a reducer. When SQI and motion are both
    disabled, every record already admitted by the configurable ``roles`` input
    selector is Excellent; static-only versus all-role scope is therefore an
    explicit data input choice rather than a hidden tier gate.
    """

    for name, value in (
        ("sqi_enabled", sqi_enabled),
        ("motion_enabled", motion_enabled),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean")
    motion_state = _motion_state(enabled=motion_enabled, high=motion_high)
    if motion_state == "unavailable":
        return QualityTierDecision(
            tier=QualityTier.UNFIT,
            sqi_enabled=sqi_enabled,
            motion_enabled=True,
            reasons=("motion_detector_enabled_but_evidence_unavailable",),
        )

    if not sqi_enabled:
        if motion_state == "high":
            tier = QualityTier.UNFIT
            reason = "sqi_off_high_motion_unfit"
        elif motion_state == "low":
            tier = QualityTier.EXCELLENT
            reason = "sqi_off_low_motion_excellent"
        else:
            tier = QualityTier.EXCELLENT
            reason = "sqi_and_motion_off_selected_role_excellent"
        return QualityTierDecision(
            tier=tier,
            sqi_enabled=False,
            motion_enabled=motion_enabled,
            reasons=(reason,),
        )

    rate_pass = _endpoint_pass(q_rate_state)
    if not rate_pass:
        return QualityTierDecision(
            tier=QualityTier.UNFIT,
            sqi_enabled=True,
            motion_enabled=motion_enabled,
            reasons=("q_rate_not_pass_unfit",),
        )
    morph_pass = _endpoint_pass(q_morph_state)
    tier = _SQI_TIER_TABLE[(morph_pass, motion_state)]
    return QualityTierDecision(
        tier=tier,
        sqi_enabled=True,
        motion_enabled=motion_enabled,
        reasons=(
            "q_rate_pass",
            "q_morph_pass" if morph_pass else "q_morph_not_pass",
            f"motion_{motion_state}",
            f"tier_{tier.value}",
        ),
    )


def route_module_switches_from_config(config: Mapping[str, Any]) -> RouteModuleSwitches:
    """Resolve SQI, motion-detector, and denoiser switches independently."""

    artifact = config.get("artifact", {})
    if not isinstance(artifact, Mapping):
        raise ValueError("config['artifact'] must be a mapping")
    motion_enabled = artifact.get("motion_detector_enabled", False)
    if not isinstance(motion_enabled, bool):
        raise ValueError("artifact.motion_detector_enabled must be boolean")
    reducer = str(artifact.get("reducer", "identity"))
    denoiser_enabled = artifact.get("denoiser_enabled", reducer != "identity")
    if not isinstance(denoiser_enabled, bool):
        raise ValueError("artifact.denoiser_enabled must be boolean")
    return RouteModuleSwitches(
        sqi_enabled=quality_mode_from_config(config) is QualityMode.ROUTE,
        motion_detector_enabled=motion_enabled,
        denoiser_enabled=denoiser_enabled,
    )


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
        if self.mode is QualityMode.ROUTE:
            if not self.computed or self.result is None:
                raise ValueError("route mode must retain its computed SQI result")
            if (
                self.classification_action != "apply_explicit_route_policy"
                or not self.affects_retention
                or not self.affects_aggregation
                or not self.affects_prediction
            ):
                raise ValueError("route mode must declare its classification effects")


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
    """Execute one quality mode without using metadata as an authorization gate.

    ``route`` evaluates the endpoint score and marks the result for the explicit
    route state machine.  Artifact reduction remains a separate step because it
    requires reducer configuration and segment integrity inputs that are not part
    of this small facade.
    """

    resolved = resolve_quality_mode(mode)
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
    elif resolved is QualityMode.DIAGNOSTICS_ONLY:
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
    else:
        result = evaluator(values, **evaluation_kwargs)
        if not isinstance(result, QualityResult):
            raise TypeError(
                "quality route evaluator must return an endpoint QualityResult"
            )
        outcome = QualityModeOutcome(
            mode=resolved,
            computed=True,
            result=result,
            classification_action="apply_explicit_route_policy",
            affects_retention=True,
            affects_aggregation=True,
            affects_prediction=True,
            reasons=("endpoint_quality_ready_for_explicit_route_state_machine",),
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
    "QualityTier",
    "QualityTierDecision",
    "RouteModuleSwitches",
    "assert_quality_route",
    "quality_mode_from_config",
    "resolve_quality_mode",
    "route_module_switches_from_config",
    "route_quality_tier",
    "run_quality_mode",
]
