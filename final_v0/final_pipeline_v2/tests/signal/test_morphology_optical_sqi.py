"""Direct-only morphology/optical and endpoint SQI tests / 形态、双波长与SQI测试。"""

from __future__ import annotations

import unittest

import numpy as np

from ppg_frailty.contracts import PulseResult, QualityState, SignalRoute
from ppg_frailty.peaks import CANONICAL_DETECTOR_ID
from ppg_frailty.signal import (
    fit_sqi_calibrator,
    SqiConfig,
    detect_pulses,
    detect_pulses_per_wavelength,
    evaluate_quality,
    extract_dual_optical,
    extract_morphology,
)


def synthetic_signals(seconds: float = 20.0) -> tuple[np.ndarray, np.ndarray, object]:
    """构造带 acquisition baseline 的双波长波形 / Build baseline-preserving signals."""

    time = np.arange(0.0, seconds, 1.0 / 400.0)
    red_filtered = 20.0 * np.sin(2 * np.pi * 1.2 * time)
    ir_filtered = 15.0 * np.sin(2 * np.pi * 1.2 * time + 0.05)
    filtered = np.column_stack((red_filtered, ir_filtered))
    native = filtered + np.array([1000.0, 1200.0])
    return native, filtered, detect_pulses(
        filtered,
        detector_id=CANONICAL_DETECTOR_ID,
    )


class DirectFeatureTest(unittest.TestCase):
    """验证 direct 边界和 rate-only 禁止调用 / Verify strict route boundary."""

    def test_morphology_and_dual_optical_are_available_direct(self) -> None:
        native, filtered, pulse = synthetic_signals()
        pulses = detect_pulses_per_wavelength(
            filtered,
            detector_id=CANONICAL_DETECTOR_ID,
        )
        morphology = extract_morphology(filtered, pulse, route=SignalRoute.DIRECT)
        optical = extract_dual_optical(
            native,
            filtered,
            pulses,
            route=SignalRoute.DIRECT,
        )
        self.assertTrue(morphology.aggregate_validity["amplitude_median"])
        self.assertTrue(optical.aggregate_validity["ratio_of_ratios_median"])
        self.assertGreater(optical.aggregate_values["red_ir_max_xcorr"], 0.9)
        self.assertNotIn(
            "red_ir_cardiac_coherence",
            optical.aggregate_values,
        )

    def test_optical_uses_local_valley_baselines_and_canonical_ratios(self) -> None:
        """解析波形锁定 AC/DC/ratio 公式 / Analytic waveform locks formulas."""

        samples = 2001
        index = np.arange(samples)
        red = -20.0 * np.cos(2.0 * np.pi * index / 400.0)
        infrared = -15.0 * np.cos(2.0 * np.pi * index / 400.0)
        filtered = np.column_stack((red, infrared))
        native = filtered + np.array([1000.0, 1200.0])
        peaks = np.array([200, 600, 1000, 1400, 1800], dtype=np.int64)
        starts = np.arange(peaks.size - 1, dtype=np.int64)
        def analytic_pulse(wavelength: str, score: float) -> PulseResult:
            run_id = f"analytic_direct_run_{wavelength.lower()}"
            return PulseResult(
                peaks=peaks,
                peak_timestamps_s=peaks / 400.0,
                accepted_peak_mask=np.ones(peaks.size, dtype=bool),
                interval_start_peak_indices=starts,
                interval_stop_peak_indices=starts + 1,
                ppi_s=np.ones(peaks.size - 1),
                valid_interval_mask=np.ones(peaks.size - 1, dtype=bool),
                adjacency_mask=np.ones(peaks.size - 1, dtype=bool),
                wavelength=wavelength,
                detector_version="analytic:polarity=+1",
                confidence=np.ones(peaks.size),
                source_route=SignalRoute.DIRECT,
                detection_run_id=run_id,
                interval_run_ids=np.full(peaks.size - 1, run_id),
                detector_id=CANONICAL_DETECTOR_ID,
                selected_polarity=1,
                block_hri_provenance_hash="0" * 64,
                interval_rejection_reasons=tuple(
                    "accepted" for _ in range(starts.size)
                ),
                peak_ordinals=np.arange(peaks.size, dtype=np.int64),
                detector_score=score,
                detector_coverage=1.0,
            )

        pulses = {
            "RED": analytic_pulse("RED", 2.0),
            "IR": analytic_pulse("IR", 1.0),
        }
        optical = extract_dual_optical(
            native, filtered, pulses, route=SignalRoute.DIRECT
        )
        epsilon = 1e-12
        expected_r = (
            40.0 / (980.0 + epsilon)
        ) / (
            30.0 / (1185.0 + epsilon)
        )
        self.assertAlmostEqual(optical.aggregate_values["red_ac_median"], 40.0, places=10)
        self.assertAlmostEqual(optical.aggregate_values["ir_ac_median"], 30.0, places=10)
        self.assertAlmostEqual(optical.aggregate_values["red_dc_median"], 980.0, places=10)
        self.assertAlmostEqual(optical.aggregate_values["ir_dc_median"], 1185.0, places=10)
        self.assertAlmostEqual(
            optical.aggregate_values["red_ir_ac_ratio_median"], 4.0 / 3.0, places=10
        )
        self.assertAlmostEqual(
            optical.aggregate_values["red_ir_dc_ratio_median"], 980.0 / 1185.0, places=10
        )
        self.assertAlmostEqual(
            optical.aggregate_values["ratio_of_ratios_median"], expected_r, places=10
        )

    def test_morphology_uses_explicit_negative_detector_polarity(self) -> None:
        samples = 2001
        index = np.arange(samples)
        negative_pulses = 20.0 * np.cos(2.0 * np.pi * index / 400.0)
        filtered = np.column_stack((negative_pulses, negative_pulses))
        peaks = np.array([200, 600, 1000, 1400, 1800], dtype=np.int64)
        starts = np.arange(peaks.size - 1, dtype=np.int64)
        run_id = "negative_polarity_run"
        pulse = PulseResult(
            peaks=peaks,
            peak_timestamps_s=peaks / 400.0,
            accepted_peak_mask=np.ones(peaks.size, dtype=bool),
            interval_start_peak_indices=starts,
            interval_stop_peak_indices=starts + 1,
            ppi_s=np.ones(peaks.size - 1),
            valid_interval_mask=np.ones(peaks.size - 1, dtype=bool),
            adjacency_mask=np.ones(peaks.size - 1, dtype=bool),
            wavelength="RED",
            detector_version="project_aboy_inspired_block_adaptive_v1",
            confidence=np.ones(peaks.size),
            source_route=SignalRoute.DIRECT,
            detection_run_id=run_id,
            interval_run_ids=np.full(peaks.size - 1, run_id),
            detector_id=CANONICAL_DETECTOR_ID,
            selected_polarity=-1,
            block_hri_provenance_hash="0" * 64,
            interval_rejection_reasons=tuple(
                "accepted" for _ in range(starts.size)
            ),
            peak_ordinals=np.arange(peaks.size, dtype=np.int64),
            detector_score=5.0,
            detector_coverage=1.0,
        )
        morphology = extract_morphology(
            filtered, pulse, route=SignalRoute.DIRECT
        )
        self.assertTrue(morphology.aggregate_validity["amplitude_median"])
        self.assertAlmostEqual(
            morphology.aggregate_values["amplitude_median"], 40.0, places=10
        )

    def test_nonidentity_is_rejected_before_morphology(self) -> None:
        native, filtered, pulse = synthetic_signals()
        with self.assertRaises(PermissionError):
            extract_morphology(filtered, pulse, route=SignalRoute.ARTIFACT_RATE_ONLY)
        with self.assertRaises(PermissionError):
            extract_dual_optical(native, filtered, pulse, route=SignalRoute.ARTIFACT_RATE_ONLY)

    def test_endpoint_sqi_marks_q_morph_not_applicable(self) -> None:
        _, filtered, pulse = synthetic_signals()
        config = SqiConfig()
        direct = evaluate_quality(
            filtered, route=SignalRoute.DIRECT, pulse=pulse, config=config
        )
        artifact = evaluate_quality(
            filtered,
            route=SignalRoute.ARTIFACT_RATE_ONLY,
            pulse=pulse,
            config=config,
        )
        self.assertIsNot(direct.q_morph.state, QualityState.NOT_APPLICABLE)
        self.assertIs(artifact.q_morph.state, QualityState.NOT_APPLICABLE)
        self.assertIsNone(artifact.q_morph.score)
        self.assertIn("rate.normalized_spectral_entropy", artifact.components)
        self.assertIn("rate.saturation", artifact.components)
        self.assertIs(
            artifact.components["rate.saturation"].state,
            QualityState.UNAVAILABLE,
        )

    def test_qc_evidence_exposes_all_required_components(self) -> None:
        _, filtered, pulse = synthetic_signals()
        evidence = {
            "channels": {
                "0": {
                    "longest_constant_run": 1,
                    "min_occupancy": 0.001,
                    "max_occupancy": 0.001,
                    "longest_nonfinite_gap_samples": 0,
                },
                "1": {
                    "longest_constant_run": 1,
                    "min_occupancy": 0.001,
                    "max_occupancy": 0.001,
                    "longest_nonfinite_gap_samples": 0,
                },
            }
        }
        quality = evaluate_quality(
            filtered,
            route=SignalRoute.DIRECT,
            pulse=pulse,
            config=SqiConfig(),
            qc_evidence=evidence,
        )
        for name in ("flatline", "clipping", "saturation", "long_gap"):
            self.assertIn(f"rate.{name}", quality.components)
        self.assertIs(
            quality.components["rate.saturation"].state,
            QualityState.UNAVAILABLE,
        )
        self.assertIs(
            quality.components["rate.long_gap"].state,
            QualityState.PASS,
        )

    def test_empirical_calibrator_ignores_heldout_mutation(self) -> None:
        rows = [
            {"rate.a": 0.2, "morph.b": 0.4},
            {"rate.a": 0.8, "morph.b": 0.6},
            {"rate.a": 0.1, "morph.b": 0.1},
        ]
        kwargs = {
            "participant_ids": ["train1", "train2", "heldout"],
            "fitted_on_participant_ids": ["train1", "train2"],
            "outer_train_participant_ids": ["train1", "train2"],
            "outer_oof_participant_ids": ["heldout"],
        }
        first = fit_sqi_calibrator(rows, **kwargs)
        mutated = [rows[0], rows[1], {"rate.a": 999.0, "morph.b": -999.0}]
        second = fit_sqi_calibrator(mutated, **kwargs)
        self.assertEqual(first.bounds, second.bounds)
        self.assertEqual(
            first.fitted_on_participant_ids,
            ("train1", "train2"),
        )


if __name__ == "__main__":
    unittest.main()
