"""Truth-table tests for independent SQI, motion, and denoiser routing switches."""

from __future__ import annotations

import unittest

from ppg_frailty.contracts import QualityComponent, QualityState
from ppg_frailty.module_registry import (
    list_modules,
    resolve_artifact_config,
)
from ppg_frailty.quality.routing import (
    QualityTier,
    route_module_switches_from_config,
    route_quality_tier,
)
from ppg_frailty.signal.sqi import SqiConfig, _endpoint


def tier(
    *,
    sqi: bool,
    rate: QualityState | None = QualityState.PASS,
    morph: QualityState | None = QualityState.PASS,
    motion: bool = False,
    high: bool | None = None,
    static: bool = True,
) -> QualityTier:
    return route_quality_tier(
        sqi_enabled=sqi,
        q_rate_state=rate,
        q_morph_state=morph,
        motion_enabled=motion,
        motion_high=high,
        static_role=static,
    ).tier


class SqiTierStateMachineTest(unittest.TestCase):
    def test_sqi_on_truth_table(self) -> None:
        cases = (
            # Q_rate, Q_morph, motion enabled, high motion, expected tier
            (QualityState.PASS, QualityState.PASS, False, None, QualityTier.EXCELLENT),
            (QualityState.PASS, QualityState.PASS, True, False, QualityTier.EXCELLENT),
            (QualityState.PASS, QualityState.PASS, True, True, QualityTier.UNFIT),
            (QualityState.PASS, QualityState.FAIL, False, None, QualityTier.ACCEPTABLE),
            (QualityState.PASS, QualityState.FAIL, True, False, QualityTier.ACCEPTABLE),
            (QualityState.PASS, QualityState.FAIL, True, True, QualityTier.UNFIT),
            (QualityState.FAIL, QualityState.PASS, False, None, QualityTier.UNFIT),
            (QualityState.FAIL, QualityState.FAIL, True, False, QualityTier.UNFIT),
            (QualityState.UNAVAILABLE, QualityState.PASS, True, True, QualityTier.UNFIT),
        )
        for rate, morph, motion, high, expected in cases:
            with self.subTest(
                rate=rate, morph=morph, motion=motion, high=high
            ):
                self.assertIs(
                    tier(
                        sqi=True,
                        rate=rate,
                        morph=morph,
                        motion=motion,
                        high=high,
                    ),
                    expected,
                )

    def test_sqi_off_truth_table(self) -> None:
        self.assertIs(tier(sqi=False, motion=False, static=True), QualityTier.EXCELLENT)
        self.assertIs(tier(sqi=False, motion=False, static=False), QualityTier.UNFIT)
        self.assertIs(tier(sqi=False, motion=True, high=False), QualityTier.EXCELLENT)
        self.assertIs(tier(sqi=False, motion=True, high=True), QualityTier.UNFIT)

    def test_missing_motion_evidence_fails_closed_to_unfit(self) -> None:
        decision = route_quality_tier(
            sqi_enabled=True,
            q_rate_state=QualityState.PASS,
            q_morph_state=QualityState.PASS,
            motion_enabled=True,
            motion_high=None,
            static_role=True,
        )
        self.assertIs(decision.tier, QualityTier.UNFIT)
        self.assertIn("evidence_unavailable", decision.reasons[0])

    def test_unfit_is_not_silently_converted_to_excluded(self) -> None:
        decision = route_quality_tier(
            sqi_enabled=True,
            q_rate_state=QualityState.FAIL,
            q_morph_state=QualityState.PASS,
            motion_enabled=False,
            motion_high=None,
            static_role=True,
        )
        self.assertIs(decision.tier, QualityTier.UNFIT)
        self.assertFalse(decision.eligible_for_direct_input)

    def test_three_switches_resolve_independently(self) -> None:
        switches = route_module_switches_from_config(
            {
                "quality": {"mode": "off"},
                "artifact": {
                    "reducer": "pca_bss",
                    "motion_detector_enabled": True,
                    "denoiser_enabled": False,
                },
            }
        )
        self.assertFalse(switches.sqi_enabled)
        self.assertTrue(switches.motion_detector_enabled)
        self.assertFalse(switches.denoiser_enabled)

    def test_artifact_resolver_decouples_motion_and_denoising(self) -> None:
        identity_motion = {
            "reducer": "identity",
            "reducer_version": "identity_v1",
            "selection_scope": "run_before_evaluation",
            "degraded_policy": "drop",
            "motion_detector_enabled": True,
            "motion_detector": {
                "evidence_path": "artifacts/example/motion_internal_evidence.json",
                "expected_evidence_sha256": "a" * 64,
                "device": "cuda",
                "batch_size": 64,
                "window_probability_aggregation": "median",
                "threshold_source": "bundle_frozen",
            },
            "denoiser_enabled": False,
            "non_identity_output_contract": "rate_only",
            "failure_action": "no_result_no_fallback",
            "parameters": {},
        }
        resolved = resolve_artifact_config(identity_motion)
        self.assertTrue(resolved["motion_detector_enabled"])
        self.assertFalse(resolved["denoiser_enabled"])

        duplicate_switch = dict(identity_motion)
        duplicate_switch["motion_detector"] = {
            **identity_motion["motion_detector"],
            "enabled": False,
        }
        with self.assertRaisesRegex(ValueError, "duplicate switch"):
            resolve_artifact_config(duplicate_switch)

        preconfigured = dict(identity_motion)
        preconfigured.update(
            {
                "reducer": "pca_bss",
                "reducer_version": "pca_component_select_v2",
                "motion_detector_enabled": False,
                "motion_detector": {
                    "evidence_path": None,
                    "expected_evidence_sha256": None,
                    "device": "cuda",
                    "batch_size": 64,
                    "window_probability_aggregation": "median",
                    "threshold_source": "bundle_frozen",
                },
            }
        )
        with self.assertRaisesRegex(ValueError, "inactive non-identity"):
            resolve_artifact_config(preconfigured)

        enabled = dict(preconfigured)
        enabled.update(
            {
                "denoiser_enabled": True,
                "degraded_policy": "denoise_then_extract_rate_features",
            }
        )
        resolved = resolve_artifact_config(enabled)
        self.assertTrue(resolved["denoiser_enabled"])
        self.assertFalse(resolved["motion_detector_enabled"])

    def test_default_and_ablation_reducers_are_registered(self) -> None:
        registered = {
            row["module_id"]: row["scientific_status"]
            for row in list_modules("artifact")
        }
        self.assertEqual(registered["pca_bss"], "preferred_rate_recovery")
        self.assertEqual(
            registered["fastica_bss"], "parallel_rate_recovery_ablation"
        )

    def test_motion_energy_is_retained_by_default_and_zero_is_a_valid_ablation(self) -> None:
        defaults = SqiConfig()
        self.assertEqual(defaults.calibrator, "fixed_formula_thresholds_v1")
        self.assertEqual(defaults.q_rate_threshold, 0.50)
        self.assertEqual(defaults.q_morph_threshold, 0.65)
        self.assertEqual(defaults.minimum_coverage, 0.80)
        self.assertGreater(defaults.rate_component_weights["motion_energy_rms"], 0.0)
        ablation_weights = dict(defaults.rate_component_weights)
        ablation_weights["motion_energy_rms"] = 0.0
        ablation = SqiConfig(rate_component_weights=ablation_weights)
        ablation.validate()
        self.assertEqual(ablation.rate_component_weights["motion_energy_rms"], 0.0)

    def test_score_and_coverage_threshold_boundaries_are_inclusive(self) -> None:
        component = QualityComponent(
            raw_value=0.5,
            normalized_value=0.5,
            state=QualityState.PASS,
            reason="boundary_fixture",
        )
        result = _endpoint(
            {"component": component},
            {"component": 1.0},
            threshold=0.50,
            coverage=0.80,
            minimum_coverage=0.80,
        )
        self.assertIs(result.state, QualityState.PASS)


if __name__ == "__main__":
    unittest.main()
