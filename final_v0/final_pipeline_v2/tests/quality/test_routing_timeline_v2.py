"""Truth-table, switch and ownership tests for common window routing."""

from __future__ import annotations

from dataclasses import replace
import itertools
import unittest

from ppg_frailty.quality.routing_timeline import (
    RoutingEvidence,
    build_routing_timeline,
    build_routing_windows,
    resolve_routing_evidence,
)


class RoutingTimelineV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.windows = build_routing_windows("r", 20 * 400)
        self.assertEqual(len(self.windows), 7)

    def _row(self, **changes) -> RoutingEvidence:
        base = RoutingEvidence(
            window=self.windows[0],
            sqi_mode="route",
            sqi_assessed=True,
            direct_q_rate_state="pass",
            direct_q_morph_state="pass",
        )
        return replace(base, **changes)

    def test_authoritative_high_motion_row_is_unfit_then_excluded(self) -> None:
        row = resolve_routing_evidence(
            self._row(
                motion_detector_enabled=True,
                motion_state="high",
                motion_probability=0.9,
                motion_threshold=0.5,
            ),
            role="B",
        )
        self.assertEqual(row.pre_route_tier, "unfit")
        self.assertEqual(row.final_tier, "excluded")

    def test_every_route_mode_truth_table_row(self) -> None:
        cases = (
            ("pass", "pass", False, "off", "excellent"),
            ("pass", "pass", True, "low", "excellent"),
            ("pass", "pass", True, "high", "unfit"),
            ("pass", "fail", False, "off", "acceptable"),
            ("pass", "unavailable", True, "low", "acceptable"),
            ("pass", "fail", True, "high", "unfit"),
            ("fail", "pass", False, "off", "unfit"),
            ("unavailable", "unavailable", True, "low", "unfit"),
            ("pass", "pass", True, "unavailable", "unfit"),
        )
        for rate, morph, motion_enabled, motion_state, expected in cases:
            with self.subTest(
                rate=rate,
                morph=morph,
                motion_enabled=motion_enabled,
                motion_state=motion_state,
            ):
                result = resolve_routing_evidence(
                    self._row(
                        direct_q_rate_state=rate,
                        direct_q_morph_state=morph,
                        motion_detector_enabled=motion_enabled,
                        motion_state=motion_state,
                    ),
                    role="B",
                )
                self.assertEqual(result.pre_route_tier, expected)

        structural = resolve_routing_evidence(
            self._row(structural_failure=True, denoiser_enabled=True),
            role="B",
        )
        self.assertEqual(structural.final_tier, "excluded")
        self.assertFalse(structural.denoiser_requested)

    def test_post_reduction_promotes_only_route_mode_unfit_to_acceptable(self) -> None:
        promoted = resolve_routing_evidence(
            self._row(
                direct_q_rate_state="fail",
                denoiser_enabled=True,
                denoiser_status="success",
                post_q_rate_state="pass",
            ),
            role="B",
        )
        self.assertEqual(promoted.final_tier, "acceptable")
        self.assertEqual(promoted.source_route, "processed")
        self.assertEqual(promoted.source_view, "x_ar_400")

        diagnostic_only = resolve_routing_evidence(
            replace(
                promoted,
                sqi_mode="diagnostics_only",
                motion_detector_enabled=True,
                motion_state="high",
            ),
            role="B",
        )
        self.assertEqual(diagnostic_only.final_tier, "excluded")
        self.assertEqual(diagnostic_only.source_view, "none")

    def test_feature_vector_can_explicitly_recover_rate_features_with_sqi_off(
        self,
    ) -> None:
        recovered = resolve_routing_evidence(
            self._row(
                sqi_mode="off",
                sqi_assessed=False,
                direct_q_rate_state=None,
                direct_q_morph_state=None,
                motion_detector_enabled=True,
                motion_state="high",
                denoiser_enabled=True,
                denoiser_status="success",
                post_q_rate_state="pass",
                post_q_rate_score=0.8,
            ),
            role="S1",
            allow_rate_feature_recovery_without_direct_sqi=True,
        )
        self.assertFalse(recovered.sqi_assessed)
        self.assertTrue(recovered.denoiser_requested)
        self.assertEqual(recovered.pre_route_tier, "unfit")
        self.assertEqual(recovered.final_tier, "acceptable")
        self.assertEqual(recovered.source_route, "processed")
        self.assertEqual(recovered.source_view, "x_ar_400")
        self.assertIn(
            "post_q_rate_pass_promoted_acceptable_processed",
            recovered.reason_codes,
        )

        diagnostics_only = resolve_routing_evidence(
            replace(recovered, sqi_mode="diagnostics_only"),
            role="S1",
            allow_rate_feature_recovery_without_direct_sqi=True,
        )
        self.assertEqual(diagnostics_only.final_tier, "excluded")
        self.assertEqual(diagnostics_only.source_view, "none")

    def test_independent_boolean_switches_and_no_unauthorised_promotion(self) -> None:
        for sqi_active, motion_enabled, denoiser_enabled in itertools.product(
            (False, True), repeat=3
        ):
            with self.subTest(
                sqi=sqi_active, motion=motion_enabled, denoiser=denoiser_enabled
            ):
                row = self._row(
                    sqi_mode="route" if sqi_active else "off",
                    sqi_assessed=sqi_active,
                    direct_q_rate_state="fail" if sqi_active else None,
                    direct_q_morph_state="fail" if sqi_active else None,
                    motion_detector_enabled=motion_enabled,
                    motion_state="high" if motion_enabled else "off",
                    denoiser_enabled=denoiser_enabled,
                    denoiser_status="success",
                    post_q_rate_state="pass",
                    post_q_rate_score=0.9,
                )
                result = resolve_routing_evidence(row, role="S1")
                expected_request = denoiser_enabled and (sqi_active or motion_enabled)
                self.assertEqual(result.denoiser_requested, expected_request)
                if not sqi_active:
                    self.assertEqual(
                        result.final_tier,
                        "excluded" if motion_enabled else "excellent",
                    )

    def test_diagnostics_only_cannot_route_from_q_states(self) -> None:
        result = resolve_routing_evidence(
            self._row(
                sqi_mode="diagnostics_only",
                direct_q_rate_state="fail",
                direct_q_morph_state="fail",
            ),
            role="B",
        )
        self.assertEqual(result.final_tier, "excellent")
        self.assertTrue(result.sqi_assessed)

    def test_sqi_and_motion_off_admits_every_configured_role(self) -> None:
        for role in ("B", "R1", "S1", "W2"):
            with self.subTest(role=role):
                result = resolve_routing_evidence(
                    self._row(
                        sqi_mode="off",
                        sqi_assessed=False,
                        direct_q_rate_state=None,
                        direct_q_morph_state=None,
                    ),
                    role=role,
                )
                self.assertEqual(result.final_tier, "excellent")
                self.assertIn(
                    "sqi_and_motion_off_selected_role_excellent",
                    result.reason_codes,
                )

    def test_midpoint_cells_are_unique_and_cover_every_sample(self) -> None:
        rows = tuple(
            resolve_routing_evidence(
                self._row(window=window),
                role="B",
            )
            for window in self.windows
        )
        timeline = build_routing_timeline(
            record_id="r",
            participant_id="p1",
            role="B",
            n_samples=20 * 400,
            evidence=rows,
            config_sha256="a" * 64,
        )
        timeline.validate()
        self.assertEqual(timeline.cells[0].start_sample_400, 0)
        self.assertEqual(timeline.cells[-1].stop_sample_400, 20 * 400)
        for left, right in zip(timeline.cells, timeline.cells[1:]):
            self.assertEqual(left.stop_sample_400, right.start_sample_400)


if __name__ == "__main__":
    unittest.main()
