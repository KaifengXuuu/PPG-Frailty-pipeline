"""Pure A1/A2 seven-state routing tests; supervised activation remains separate."""

from __future__ import annotations

import unittest

import numpy as np

from ppg_frailty.contracts import (
    ArtifactReductionResult,
    QualityEndpoint,
    QualityResult,
    QualityState,
    RouteState,
    SourceSignalView,
)
from ppg_frailty.quality import (
    SegmentIntegrity,
    finalize_rate_recovery,
    route_segment_pre_reduction,
)


def endpoint(state: QualityState) -> QualityEndpoint:
    return QualityEndpoint(
        score=1.0 if state is QualityState.PASS else 0.0,
        state=state,
        threshold=0.5,
        components={},
        reasons=(),
        coverage=1.0,
    )


def quality(rate: QualityState, shape: QualityState) -> QualityResult:
    q_rate = endpoint(rate)
    q_shape = endpoint(shape)
    morph_pass = rate is QualityState.PASS and shape is QualityState.PASS
    q_morph = endpoint(QualityState.PASS if morph_pass else QualityState.FAIL)
    return QualityResult(
        q_rate=q_rate,
        q_morph=q_morph,
        state="pass" if morph_pass else "fail",
        components={},
        reasons=(),
        coverage=1.0,
        q_shape=q_shape,
    )


def integrity(pass_: bool = True) -> SegmentIntegrity:
    return SegmentIntegrity(
        pass_=pass_,
        segment_id="record::000000-004000",
        start_sample=0,
        end_sample=4000,
        reasons=() if pass_ else ("flatline",),
    )


def reduction(*, success: bool = True) -> ArtifactReductionResult:
    return ArtifactReductionResult(
        x_ar=np.ones((4000, 2)) if success else None,
        reducer_id="test_nonidentity",
        reducer_version="test_v1",
        is_identity=False,
        status="success" if success else "failed",
        confidence=1.0 if success else 0.0,
        diagnostics={},
        parameters={},
        channel_available=(True, True),
        alignment={"same_time_grid": success},
        reasons=() if success else ("synthetic_failure",),
    )


class SegmentRouteStateMachineTest(unittest.TestCase):
    def test_all_seven_typed_states_and_source_views(self) -> None:
        hard = route_segment_pre_reduction(
            integrity(False),
            q_pre=None,
            recoverable_motion=False,
            reducer_enabled=False,
        )
        full = route_segment_pre_reduction(
            integrity(),
            q_pre=quality(QualityState.PASS, QualityState.PASS),
            recoverable_motion=False,
            reducer_enabled=False,
        )
        rate_direct = route_segment_pre_reduction(
            integrity(),
            q_pre=quality(QualityState.PASS, QualityState.FAIL),
            recoverable_motion=True,
            reducer_enabled=True,
        )
        degraded = route_segment_pre_reduction(
            integrity(),
            q_pre=quality(QualityState.FAIL, QualityState.FAIL),
            recoverable_motion=False,
            reducer_enabled=True,
        )
        candidate = route_segment_pre_reduction(
            integrity(),
            q_pre=quality(QualityState.FAIL, QualityState.FAIL),
            recoverable_motion=True,
            reducer_enabled=True,
        )
        processed = finalize_rate_recovery(
            candidate,
            reduction=reduction(),
            q_rate_post=endpoint(QualityState.PASS),
        )
        rejected = finalize_rate_recovery(
            candidate,
            reduction=reduction(),
            q_rate_post=endpoint(QualityState.FAIL),
        )
        self.assertEqual(
            {
                item.state
                for item in (
                    hard,
                    full,
                    rate_direct,
                    candidate,
                    processed,
                    degraded,
                    rejected,
                )
            },
            set(RouteState),
        )
        self.assertIs(full.source_signal, SourceSignalView.X_FILTER)
        self.assertIs(rate_direct.source_signal, SourceSignalView.X_FILTER)
        self.assertIs(processed.source_signal, SourceSignalView.X_AR)
        self.assertTrue(rate_direct.is_terminal)
        self.assertFalse(candidate.is_terminal)

    def test_shape_failure_never_requests_reducer_when_rate_passes(self) -> None:
        result = route_segment_pre_reduction(
            integrity(),
            q_pre=quality(QualityState.PASS, QualityState.FAIL),
            recoverable_motion=True,
            reducer_enabled=True,
        )
        self.assertIs(result.state, RouteState.RATE_ONLY_DIRECT)
        self.assertEqual(result.reducer_status, "not_run")

    def test_reducer_failure_is_explicit_and_has_no_fallback_source(self) -> None:
        candidate = route_segment_pre_reduction(
            integrity(),
            q_pre=quality(QualityState.FAIL, QualityState.FAIL),
            recoverable_motion=True,
            reducer_enabled=True,
        )
        result = finalize_rate_recovery(
            candidate,
            reduction=reduction(success=False),
            q_rate_post=None,
        )
        self.assertIs(result.state, RouteState.REJECTED_AFTER_REDUCTION)
        self.assertIsNone(result.source_signal)
        self.assertIn("artifact_reducer_failed", result.reasons)


if __name__ == "__main__":
    unittest.main()
