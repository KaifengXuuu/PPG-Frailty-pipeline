"""Correction-D tests for independent RED/IR pairing and optical formulas."""

from __future__ import annotations

from dataclasses import asdict
import unittest

import numpy as np

from ppg_frailty.contracts import PulseResult, SignalRoute
from ppg_frailty.features.registry import default_registry
from ppg_frailty.peaks.pairing import (
    pair_dual_wavelength_beats,
    select_reference_wavelength,
)
from ppg_frailty.signal.optical import (
    _standardized_waveform_agreement,
    extract_dual_optical,
)


FS_HZ = 400.0
DETECTOR_ID = "aboy_project_v1"


def pulse_result(
    wavelength: str,
    peaks: list[int],
    *,
    accepted: list[bool] | None = None,
    polarity: int = 1,
    score: float = 10.0,
    coverage: float = 0.9,
    detector_id: str = DETECTOR_ID,
) -> PulseResult:
    """Build a complete independent PulseResult fixture."""

    peak_array = np.asarray(peaks, dtype=np.int64)
    accepted_array = (
        np.ones(peak_array.size, dtype=bool)
        if accepted is None
        else np.asarray(accepted, dtype=bool)
    )
    start = np.arange(max(peak_array.size - 1, 0), dtype=np.int64)
    run_id = f"{wavelength.lower()}-run"
    return PulseResult(
        peaks=peak_array,
        peak_timestamps_s=peak_array.astype(np.float64) / FS_HZ,
        accepted_peak_mask=accepted_array,
        interval_start_peak_indices=start,
        interval_stop_peak_indices=start + 1,
        ppi_s=np.diff(peak_array.astype(np.float64)) / FS_HZ,
        valid_interval_mask=np.ones(start.size, dtype=bool),
        adjacency_mask=np.ones(start.size, dtype=bool),
        wavelength=wavelength,
        detector_version=f"{detector_id}:polarity={polarity:+d}",
        confidence=np.ones(peak_array.size, dtype=np.float64),
        source_route=SignalRoute.DIRECT,
        detection_run_id=run_id,
        interval_run_ids=np.full(start.size, run_id),
        detector_id=detector_id,
        selected_polarity=polarity,
        block_hri_provenance_hash="0" * 64,
        block_provenance=(),
        interval_rejection_reasons=tuple("accepted" for _ in range(start.size)),
        peak_ordinals=np.arange(peak_array.size, dtype=np.int64),
        detector_score=score,
        detector_coverage=coverage,
    )


class DualWavelengthPairingTest(unittest.TestCase):
    """Pairing stays chronological, non-reusing, explicit, and deterministic."""

    def test_reference_selection_uses_persisted_score_coverage_then_red(self) -> None:
        tied = {
            "RED": pulse_result("RED", [100, 500, 900], score=5.0, coverage=0.8),
            "IR": pulse_result("IR", [120, 520, 920], score=5.0, coverage=0.8),
        }
        self.assertEqual(select_reference_wavelength(tied), "RED")
        higher_ir_coverage = {
            "RED": tied["RED"],
            "IR": pulse_result("IR", [120, 520, 920], score=5.0, coverage=0.9),
        }
        self.assertEqual(select_reference_wavelength(higher_ir_coverage), "IR")

    def test_pairing_records_missing_extra_and_ambiguity_without_reuse(self) -> None:
        pulses = {
            "RED": pulse_result("RED", [100, 500, 900, 1300, 1700], score=9.0),
            "IR": pulse_result(
                "IR",
                [120, 480, 520, 1320, 1720],
                score=8.0,
                polarity=-1,
            ),
        }
        result = pair_dual_wavelength_beats(pulses, fs_hz=FS_HZ)
        paired = result.paired_rows
        self.assertEqual(
            [(row.red_peak_sample, row.ir_peak_sample) for row in paired],
            [(500, 480), (1300, 1320)],
        )
        self.assertEqual(len({row.ir_peak_ordinal for row in paired}), len(paired))
        ambiguous = next(row for row in paired if row.red_peak_sample == 500)
        self.assertIn(
            "multiple_secondary_candidates_nearest_selected",
            ambiguous.reason_codes,
        )
        missing = next(
            row
            for row in result.rows
            if row.reference_peak_sample == 900
        )
        self.assertFalse(missing.pair_valid)
        self.assertEqual(
            missing.reason_codes,
            ("no_secondary_peak_in_reference_cycle",),
        )
        unpaired = [
            row
            for row in result.rows
            if row.reason_codes == ("secondary_peak_unpaired",)
        ]
        self.assertTrue(any(row.ir_peak_sample == 520 for row in unpaired))

    def test_pairing_serializes_both_detector_runs_polarities_and_hri_hashes(
        self,
    ) -> None:
        pulses = {
            "RED": pulse_result(
                "RED", [100, 500, 900, 1300], polarity=1, score=9.0
            ),
            "IR": pulse_result(
                "IR", [120, 520, 920, 1320], polarity=-1, score=8.0
            ),
        }
        payload = asdict(pair_dual_wavelength_beats(pulses, fs_hz=FS_HZ))
        self.assertEqual(payload["red_detection_run_id"], "red-run")
        self.assertEqual(payload["ir_detection_run_id"], "ir-run")
        self.assertEqual(payload["red_selected_polarity"], 1)
        self.assertEqual(payload["ir_selected_polarity"], -1)
        self.assertEqual(payload["red_block_hri_provenance_hash"], "0" * 64)
        self.assertEqual(payload["ir_block_hri_provenance_hash"], "0" * 64)
        self.assertIn(DETECTOR_ID, payload["red_detector_version"])
        self.assertIn(DETECTOR_ID, payload["ir_detector_version"])

    def test_rejected_events_remain_explicit_audit_rows(self) -> None:
        pulses = {
            "RED": pulse_result(
                "RED",
                [100, 500, 900, 1300],
                accepted=[True, False, True, True],
                score=9.0,
            ),
            "IR": pulse_result(
                "IR",
                [120, 520, 920, 1320],
                accepted=[True, True, False, True],
                score=8.0,
            ),
        }
        result = pair_dual_wavelength_beats(pulses, fs_hz=FS_HZ)
        reasons = [reason for row in result.rows for reason in row.reason_codes]
        self.assertIn("reference_peak_rejected", reasons)
        self.assertIn("secondary_peak_rejected", reasons)

    def test_detector_identity_must_match(self) -> None:
        pulses = {
            "RED": pulse_result("RED", [100, 500, 900], detector_id=DETECTOR_ID),
            "IR": pulse_result(
                "IR",
                [120, 520, 920],
                detector_id="dual_polarity_prominence_v1_ablation",
            ),
        }
        with self.assertRaisesRegex(ValueError, "same explicit detector_id"):
            pair_dual_wavelength_beats(pulses, fs_hz=FS_HZ)


class DualWavelengthOpticalTest(unittest.TestCase):
    """Optical formulas use own peaks/valleys and the common paired-valid set."""

    @staticmethod
    def _sparse_peak_signals(
        *,
        negative_dc: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, PulseResult]]:
        samples = 1100
        peaks = [100, 300, 500, 700, 900]
        red_amplitudes = [5.0, 10.0, 10.0, 100.0, 5.0]
        ir_amplitudes = [5.0, 1.0, 10.0, 20.0, 5.0]
        filtered = np.zeros((samples, 2), dtype=np.float64)
        for peak, red_amplitude, ir_amplitude in zip(
            peaks, red_amplitudes, ir_amplitudes
        ):
            filtered[peak, 0] = red_amplitude
            filtered[peak, 1] = -ir_amplitude
        baseline = np.array(
            [-1000.0, -1200.0] if negative_dc else [1000.0, 1200.0]
        )
        native = filtered + baseline
        pulses = {
            "RED": pulse_result("RED", peaks, score=9.0, polarity=1),
            "IR": pulse_result("IR", peaks, score=8.0, polarity=-1),
        }
        return native, filtered, pulses

    def test_opposite_polarities_use_own_peaks_and_own_valleys(self) -> None:
        samples = 2001
        index = np.arange(samples)
        red_filtered = -20.0 * np.cos(2.0 * np.pi * index / 400.0)
        ir_filtered = 15.0 * np.cos(2.0 * np.pi * (index - 20) / 400.0)
        filtered = np.column_stack((red_filtered, ir_filtered))
        native = filtered + np.array([1000.0, 1200.0])
        pulses = {
            "RED": pulse_result(
                "RED", [200, 600, 1000, 1400, 1800], score=9.0, polarity=1
            ),
            "IR": pulse_result(
                "IR", [220, 620, 1020, 1420, 1820], score=8.0, polarity=-1
            ),
        }
        result = extract_dual_optical(
            native, filtered, pulses, route=SignalRoute.DIRECT
        )
        self.assertEqual(len(result.pairing.paired_rows), 3)
        self.assertAlmostEqual(result.aggregate_values["red_ac_median"], 40.0)
        self.assertAlmostEqual(result.aggregate_values["ir_ac_median"], 30.0)
        valid_audit = [row for row in result.beat_audit if row.optical_valid]
        self.assertEqual(len(valid_audit), 3)
        self.assertTrue(
            all(
                row.pairing.lag_samples_ir_minus_red == 20
                for row in valid_audit
            )
        )

    def test_recording_ratios_follow_median_ac_dc_not_beat_ratio_median(self) -> None:
        native, filtered, pulses = self._sparse_peak_signals()
        result = extract_dual_optical(
            native, filtered, pulses, route=SignalRoute.DIRECT
        )
        aggregate_ac_ratio = (
            result.aggregate_values["red_ac_median"]
            / result.aggregate_values["ir_ac_median"]
        )
        beatwise = result.beat_values["red_ir_ac_ratio"]
        beatwise_median = float(
            np.median(beatwise[result.beat_validity["red_ir_ac_ratio"]])
        )
        self.assertAlmostEqual(
            result.aggregate_values["red_ir_ac_ratio_median"],
            aggregate_ac_ratio,
            places=10,
        )
        self.assertNotAlmostEqual(aggregate_ac_ratio, beatwise_median, places=6)
        self.assertTrue(
            result.diagnostics["canonical_ratios_from_recording_median_ac_dc"]
        )
        self.assertFalse(
            result.diagnostics["beatwise_ratios_affect_prediction"]
        )

    def test_negative_dc_does_not_change_dc_dependent_ratios(self) -> None:
        positive = self._sparse_peak_signals(negative_dc=False)
        negative = self._sparse_peak_signals(negative_dc=True)
        positive_result = extract_dual_optical(
            *positive[:2], positive[2], route=SignalRoute.DIRECT
        )
        negative_result = extract_dual_optical(
            *negative[:2], negative[2], route=SignalRoute.DIRECT
        )
        for name in (
            "red_pi_median",
            "ir_pi_median",
            "red_ir_dc_ratio_median",
            "ratio_of_ratios_median",
        ):
            self.assertAlmostEqual(
                positive_result.aggregate_values[name],
                negative_result.aggregate_values[name],
                places=12,
            )

    def test_standardized_xcorr_recovers_point_four_second_shift(self) -> None:
        rng = np.random.default_rng(42)
        red = rng.normal(size=4000)
        infrared = np.zeros_like(red)
        infrared[160:] = 7.0 + 3.0 * red[:-160]
        filtered = np.column_stack((red, infrared))
        native = filtered + np.array([1000.0, 1200.0])
        pulses = {
            "RED": pulse_result("RED", [400, 1000, 1600, 2200, 2800], score=9.0),
            "IR": pulse_result("IR", [560, 1160, 1760, 2360, 2960], score=8.0),
        }
        result = extract_dual_optical(
            native, filtered, pulses, route=SignalRoute.DIRECT
        )
        self.assertAlmostEqual(
            result.aggregate_values["red_ir_xcorr_lag_s"], 0.4
        )
        agreement = result.diagnostics["waveform_agreement"]
        self.assertEqual(agreement["max_lag_samples"], 200)
        self.assertEqual(
            agreement["search_bounds_inclusive_samples"], (-200, 200)
        )
        expected_zero = float(np.corrcoef(red, infrared)[0, 1])
        self.assertAlmostEqual(
            result.aggregate_values["red_ir_zero_lag_correlation"],
            expected_zero,
            places=12,
        )

    def test_xcorr_equal_maxima_use_abs_then_signed_lag_tie_break(self) -> None:
        red = np.zeros(401, dtype=np.float64)
        infrared = np.zeros(401, dtype=np.float64)
        red[200] = 1.0
        infrared[199] = 1.0
        infrared[201] = 1.0
        _, _, lag = _standardized_waveform_agreement(
            red,
            infrared,
            max_lag_samples=200,
        )
        self.assertEqual(lag, -1)

    def test_coherence_is_absent_from_formal_aggregates_and_registry(self) -> None:
        native, filtered, pulses = self._sparse_peak_signals()
        result = extract_dual_optical(
            native, filtered, pulses, route=SignalRoute.DIRECT
        )
        self.assertNotIn("red_ir_cardiac_coherence", result.aggregate_values)
        self.assertNotIn(
            "optical.red_ir_cardiac_coherence",
            default_registry().names,
        )
        self.assertFalse(result.diagnostics["coherence"]["computed"])
        self.assertFalse(result.diagnostics["coherence"]["affects_prediction"])

    def test_direct_identity_route_restriction_remains(self) -> None:
        native, filtered, pulses = self._sparse_peak_signals()
        with self.assertRaisesRegex(TypeError, "independent RED and IR"):
            extract_dual_optical(
                native,
                filtered,
                pulses["RED"],
                route=SignalRoute.DIRECT,
            )
        with self.assertRaises(PermissionError):
            extract_dual_optical(
                native,
                filtered,
                pulses,
                route=SignalRoute.ARTIFACT_RATE_ONLY,
            )


if __name__ == "__main__":
    unittest.main()
