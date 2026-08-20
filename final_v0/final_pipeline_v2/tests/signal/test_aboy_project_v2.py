"""Thesis-parity contracts for the project Aboy++-inspired detector."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy import signal

from ppg_frailty.config import (
    canonical_json_bytes,
    load_config,
    materialize_formal_ablation_config,
    validate_config_payload,
)
from ppg_frailty.contracts import SignalRoute
from ppg_frailty.peaks import (
    ABLATION_DETECTOR_ID,
    CANONICAL_DETECTOR_ID,
    detect_pulses,
    detect_pulses_per_wavelength,
)
from ppg_frailty.peaks.aboy_project import (
    INITIAL_HRI_RULE,
    _BlockCandidate,
    _PolarityCandidate,
    _bandpass_block,
    _block_candidate,
    _block_parameters,
    _clean_intervals,
    _merge_block_peaks,
    _result_for_wavelength,
    _score_peak_train,
    _upper_30_percent_mean,
)


FS = 400.0


def pulse_train(
    times_s: np.ndarray,
    *,
    seconds: float = 20.0,
    width_s: float = 0.04,
) -> np.ndarray:
    """Build deterministic narrow positive pulses on the 400 Hz grid."""

    time = np.arange(int(round(seconds * FS))) / FS
    output = np.zeros(time.size, dtype=np.float64)
    for event in np.asarray(times_s, dtype=np.float64):
        output += np.exp(-0.5 * np.square((time - event) / width_s))
    return output


class AboyProjectDetectorTest(unittest.TestCase):
    def test_public_thresholds_are_validated_and_forwarded(self) -> None:
        sentinel = {"RED": object()}
        matrix = np.zeros((4000, 1), dtype=np.float64)
        with patch(
            "ppg_frailty.peaks.resolver.detect_pulses_per_wavelength_aboy_project",
            return_value=sentinel,
        ) as implementation:
            observed = detect_pulses_per_wavelength(
                matrix,
                detector_id=CANONICAL_DETECTOR_ID,
                min_observation_sec=6.5,
                min_peaks=3,
            )
        self.assertIs(observed, sentinel)
        self.assertEqual(implementation.call_args.kwargs["min_observation_sec"], 6.5)
        self.assertEqual(implementation.call_args.kwargs["min_peaks"], 3)
        for kwargs, message in (
            ({"min_observation_sec": 0.0}, "min_observation_sec"),
            ({"min_observation_sec": float("inf")}, "min_observation_sec"),
            ({"min_observation_sec": True}, "min_observation_sec"),
            ({"min_peaks": 0}, "min_peaks"),
            ({"min_peaks": 2.5}, "min_peaks"),
            ({"min_peaks": True}, "min_peaks"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    detect_pulses_per_wavelength(
                        matrix,
                        detector_id=CANONICAL_DETECTOR_ID,
                        **kwargs,
                    )

    def test_thresholds_bind_detector_provenance_without_changing_peaks(self) -> None:
        waveform = pulse_train(np.arange(0.8, 19.5, 1.0))
        matrix = np.column_stack((waveform, 0.9 * waveform))
        default = detect_pulses(
            matrix,
            detector_id=CANONICAL_DETECTOR_ID,
        )
        configured = detect_pulses(
            matrix,
            detector_id=CANONICAL_DETECTOR_ID,
            min_observation_sec=6.5,
            min_peaks=3,
        )
        np.testing.assert_array_equal(configured.peaks, default.peaks)
        self.assertNotEqual(
            configured.block_hri_provenance_hash,
            default.block_hri_provenance_hash,
        )
        self.assertTrue(configured.block_provenance)
        self.assertTrue(
            all(
                row["min_observation_sec"] == 6.5
                and row["min_peaks"] == 3
                for row in configured.block_provenance
            )
        )

    def test_positive_and_inverted_sixty_bpm_have_same_events(self) -> None:
        waveform = pulse_train(np.arange(0.8, 19.5, 1.0))
        positive = detect_pulses(
            np.column_stack((waveform, 0.9 * waveform)),
            detector_id=CANONICAL_DETECTOR_ID,
        )
        inverted = detect_pulses(
            np.column_stack((-waveform, -0.9 * waveform)),
            detector_id=CANONICAL_DETECTOR_ID,
        )
        np.testing.assert_allclose(
            positive.peak_timestamps_s,
            inverted.peak_timestamps_s,
            rtol=0.0,
            atol=1.0 / FS,
        )
        self.assertEqual(positive.selected_polarity, 1)
        self.assertEqual(inverted.selected_polarity, -1)

    def test_hri_carries_to_next_block_and_controls_high_cut(self) -> None:
        first_times = np.array(
            [0.5, 1.2, 2.05, 2.8, 3.7, 4.4, 5.25, 6.0, 6.9, 7.6, 8.45, 9.2]
        )
        first = pulse_train(first_times, seconds=10.0, width_s=0.035)
        second = pulse_train(
            np.arange(0.6, 9.8, 0.8),
            seconds=10.0,
            width_s=0.035,
        )
        block0 = _block_candidate(
            first,
            block_index=0,
            block_start_sample=0,
            polarity=1,
            hri_in=0.0,
            fs_hz=FS,
        )
        self.assertGreater(block0.hri_out, 0.0)
        block1 = _block_candidate(
            second,
            block_index=1,
            block_start_sample=4000,
            polarity=1,
            hri_in=block0.hri_out,
            fs_hz=FS,
        )
        self.assertEqual(block1.hri_in, block0.hri_out)
        self.assertAlmostEqual(
            block1.provenance["f_high_hz"],
            min(8.0, max(1.5, 3.0 * (1.0 + block0.hri_out))),
            places=12,
        )
        self.assertEqual(
            block1.provenance["initial_hri_rule"],
            INITIAL_HRI_RULE,
        )

    def test_hri_uses_strict_upper_30_percent_and_full_pd_median(self) -> None:
        preliminary = np.array([0, 100, 200, 300, 500, 900], dtype=np.int64)
        final = np.array([100, 500, 900], dtype=np.int64)
        with (
            patch(
                "ppg_frailty.peaks.aboy_project._bandpass_block",
                side_effect=lambda values, **_kwargs: np.asarray(values),
            ),
            patch(
                "ppg_frailty.peaks.aboy_project.signal.find_peaks",
                side_effect=[
                    (preliminary, {}),
                    (final, {"prominences": np.ones(final.size)}),
                ],
            ),
        ):
            candidate = _block_candidate(
                np.linspace(0.0, 1.0, 1000),
                block_index=0,
                block_start_sample=0,
                polarity=1,
                hri_in=0.25,
                fs_hz=100.0,
            )
        self.assertEqual(candidate.provenance["retained_pd_count"], 2)
        self.assertEqual(
            candidate.provenance["preliminary_ppi_30th_percentile_s"],
            1.0,
        )
        self.assertEqual(
            candidate.provenance["preliminary_ppi_median_s"],
            1.0,
        )
        self.assertEqual(candidate.hri_out, 0.25)
        self.assertEqual(
            candidate.provenance["hri_update_reason"],
            "retained_mean_outside_0p5_to_1p5_preliminary_median",
        )

    def test_exact_block_equations_and_prominence(self) -> None:
        block = pulse_train(np.arange(0.8, 9.8, 1.0), seconds=10.0)
        candidate = _block_candidate(
            block,
            block_index=0,
            block_start_sample=0,
            polarity=1,
            hri_in=0.0,
            fs_hz=FS,
        )
        expected = _block_parameters(0.0, candidate.hri_out, FS)
        self.assertEqual(expected["d_210_samples"], round(FS * 60.0 / 210.0))
        self.assertEqual(expected["f_high_hz"], 3.0)
        self.assertAlmostEqual(
            expected["hrwin_samples"],
            FS / (3.0 * (1.0 + candidate.hri_out)),
            places=12,
        )
        self.assertEqual(
            expected["final_distance_samples"],
            max(
                round(2.0 * expected["hrwin_samples"]),
                expected["d_210_samples"],
            ),
        )
        filtered = _bandpass_block(block, fs_hz=FS, high_hz=3.0)
        preliminary, _ = signal.find_peaks(
            filtered,
            distance=expected["d_210_samples"],
        )
        expected_prominence = 0.25 * max(
            _upper_30_percent_mean(filtered[preliminary]),
            float(np.std(filtered, ddof=0)),
        )
        self.assertAlmostEqual(
            candidate.provenance["final_prominence"],
            expected_prominence,
            places=12,
        )

    def test_polarity_score_terms_are_frozen(self) -> None:
        peaks = np.arange(6, dtype=np.int64) * int(FS)
        score, n_clean, coverage, cv = _score_peak_train(peaks, fs_hz=FS)
        self.assertEqual(n_clean, 5)
        self.assertEqual(coverage, 1.0)
        self.assertEqual(cv, 0.0)
        self.assertEqual(score, 5.5)

    def test_block_merge_uses_historical_exact_sample_uniqueness(self) -> None:
        base = {
            "block_index": 0,
            "polarity": 1,
            "hri_in": 0.0,
            "hri_out": 0.0,
        }
        left = _BlockCandidate(
            block_start_sample=0,
            peaks=np.array([400, 3998, 4000], dtype=np.int64),
            prominence=np.array([1.0, 0.5, 0.7]),
            provenance=dict(base),
            **base,
        )
        right_base = {**base, "block_index": 1}
        right = _BlockCandidate(
            block_start_sample=4000,
            peaks=np.array([4000, 4002, 7998], dtype=np.int64),
            prominence=np.array([0.8, 0.9, 0.6]),
            provenance=dict(right_base),
            **right_base,
        )
        peaks, prominence = _merge_block_peaks((left, right))
        np.testing.assert_array_equal(peaks, [400, 3998, 4000, 4002, 7998])
        np.testing.assert_allclose(prominence, [1.0, 0.5, 0.7, 0.9, 0.6])

    def test_rate_limits_are_inclusive_and_exact(self) -> None:
        for interval, expected in ((120, True), (720, True), (119, False), (721, False)):
            with self.subTest(interval=interval):
                peaks = np.arange(7, dtype=np.int64) * interval
                _accepted, valid, reasons, _reference = _clean_intervals(
                    peaks,
                    fs_hz=420.0,
                )
                self.assertEqual(bool(np.all(valid)), expected)
                if not expected:
                    self.assertTrue(
                        all("outside_35_210_bpm" in row for row in reasons)
                    )

    def test_ratio_and_mad_cleaning_thresholds(self) -> None:
        def result(intervals: list[int]):
            return _clean_intervals(
                np.concatenate(([0], np.cumsum(intervals))).astype(np.int64),
                fs_hz=100.0,
            )

        for value, expected in ((40, True), (144, True), (39, False), (145, False)):
            with self.subTest(ratio_interval=value):
                intervals = [80] * 5 + [value] + [80] * 5
                accepted, valid, reasons, reference = result(intervals)
                self.assertEqual(reference, 0.8)
                self.assertEqual(bool(valid[5]), expected)
                if not expected:
                    self.assertFalse(accepted[6])
                    self.assertIn(
                        "outside_0p5_to_1p8_reference_ppi",
                        reasons[5],
                    )
                    self.assertIn("rejected_peak_endpoint", reasons[6])

        _accepted, valid, reasons, _reference = result(
            [99, 100, 101, 99, 100, 101, 107, 100]
        )
        self.assertFalse(valid[6])
        self.assertIn("outside_4x_1p4826_mad", reasons[6])

    def test_cleaning_orders_mad_before_reference_ratio(self) -> None:
        intervals = np.array([40, 40, 41, 60, 80], dtype=np.int64)
        peaks = np.concatenate(([0], np.cumsum(intervals)))
        _accepted, valid, reasons, reference = _clean_intervals(
            peaks,
            fs_hz=100.0,
        )
        self.assertEqual(reference, 0.4)
        self.assertFalse(valid[3])
        self.assertIn("outside_4x_1p4826_mad", reasons[3])
        self.assertNotIn("outside_0p5_to_1p8_reference_ppi", reasons[3])
        self.assertFalse(valid[4])
        self.assertIn("outside_4x_1p4826_mad", reasons[4])
        self.assertIn("outside_0p5_to_1p8_reference_ppi", reasons[4])

    def test_rejections_preserve_timestamps_ordinals_and_adjacency(self) -> None:
        intervals = np.array(
            [390, 400, 410, 390, 400, 410, 160, 520, 400, 410],
            dtype=np.int64,
        )
        peaks = np.concatenate(([0], np.cumsum(intervals))).astype(np.int64)
        selected = _PolarityCandidate(
            polarity=1,
            peaks=peaks,
            prominence=np.ones(peaks.size, dtype=np.float64),
            score=10.0,
            n_clean_ppi=7,
            coverage=0.8,
            cv=0.1,
            block_rows=({"block_index": 0, "polarity": 1},),
        )
        rejected = _PolarityCandidate(
            polarity=-1,
            peaks=peaks,
            prominence=np.ones(peaks.size, dtype=np.float64),
            score=0.0,
            n_clean_ppi=0,
            coverage=0.0,
            cv=2.0,
            block_rows=({"block_index": 0, "polarity": -1},),
        )
        with patch(
            "ppg_frailty.peaks.aboy_project._polarity_candidate",
            side_effect=[selected, rejected],
        ):
            result = _result_for_wavelength(
                np.zeros(4000, dtype=np.float64),
                label="RED",
                fs_hz=FS,
                sample_offset=0,
                source_route=SignalRoute.DIRECT,
                record_id="timeline_rejection_fixture",
                run_id=None,
                min_peaks=5,
            )
        np.testing.assert_array_equal(result.peaks, peaks)
        np.testing.assert_array_equal(
            result.peak_ordinals,
            np.arange(peaks.size),
        )
        np.testing.assert_allclose(
            result.peak_timestamps_s,
            peaks / FS,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            result.interval_start_peak_indices,
            np.arange(peaks.size - 1),
        )
        np.testing.assert_array_equal(
            result.interval_stop_peak_indices,
            np.arange(1, peaks.size),
        )
        self.assertEqual(result.ppi_s.size, peaks.size - 1)
        self.assertTrue(np.all(result.adjacency_mask))
        self.assertTrue(
            any(
                "outside_0p5_to_1p8_reference_ppi" in reason
                for reason in result.interval_rejection_reasons
            )
        )
        self.assertTrue(
            any(
                "outside_4x_1p4826_mad" in reason
                for reason in result.interval_rejection_reasons
            )
        )
        self.assertFalse(np.all(result.valid_interval_mask))
        self.assertEqual(
            len(result.interval_rejection_reasons),
            result.ppi_s.size,
        )
        result.validate_identity()

    def test_red_and_ir_run_independently(self) -> None:
        time = np.arange(0.0, 20.0, 1.0 / FS)
        red = np.sin(2.0 * np.pi * 1.0 * time)
        infrared = np.sin(2.0 * np.pi * 1.25 * time + 0.2)
        results = detect_pulses_per_wavelength(
            np.column_stack((red, infrared)),
            detector_id=CANONICAL_DETECTOR_ID,
        )
        self.assertEqual(set(results), {"RED", "IR"})
        self.assertEqual(results["RED"].wavelength, "RED")
        self.assertEqual(results["IR"].wavelength, "IR")
        self.assertNotEqual(
            results["RED"].peaks.size,
            results["IR"].peaks.size,
        )
        self.assertNotEqual(
            results["RED"].detection_run_id,
            results["IR"].detection_run_id,
        )

    def test_old_detector_characterization_is_explicit_ablation_only(self) -> None:
        time = np.arange(0.0, 20.0, 1.0 / FS)
        waveform = (
            np.sin(2.0 * np.pi * 1.25 * time)
            + 0.15 * np.sin(2.0 * np.pi * 2.5 * time + 0.3)
        )
        result = detect_pulses(
            np.column_stack((waveform, 0.8 * waveform)),
            detector_id=ABLATION_DETECTOR_ID,
        )
        digest = hashlib.sha256(
            result.peaks.astype("<i8").tobytes()
        ).hexdigest()
        self.assertEqual(
            digest,
            "f0d4460fdf808a676019752de11b976b1d47219a6958a8768b3190e420c38b1c",
        )
        self.assertEqual(result.detector_id, ABLATION_DETECTOR_ID)

    def test_old_detector_auto_preserves_legacy_channel_tie_break(self) -> None:
        from ppg_frailty.signal.peaks import _Candidate

        red_peaks = np.arange(6, dtype=np.int64) * 400 + 100
        ir_peaks = np.arange(7, dtype=np.int64) * 120 + 100

        def candidate(
            _values: np.ndarray,
            *,
            channel_index: int,
            polarity: int,
            fs_hz: float,
        ) -> _Candidate:
            del fs_hz
            if polarity < 0:
                peaks = np.arange(5, dtype=np.int64) * 400 + 100
                score = 0.0
            elif channel_index == 0:
                peaks = red_peaks
                score = 1.0
            else:
                peaks = ir_peaks
                score = 1.0
            return _Candidate(
                peaks=peaks,
                prominence=np.ones(peaks.size, dtype=np.float64),
                score=score,
                polarity=polarity,
                channel_index=channel_index,
            )

        matrix = np.zeros((8000, 2), dtype=np.float64)
        with patch(
            "ppg_frailty.signal.peaks._candidate",
            side_effect=candidate,
        ):
            result = detect_pulses(
                matrix,
                detector_id=ABLATION_DETECTOR_ID,
                wavelength="auto",
            )
        self.assertEqual(result.wavelength, "IR")
        np.testing.assert_array_equal(result.peaks, ir_peaks)

    def test_missing_unknown_and_canonical_failure_fail_closed(self) -> None:
        waveform = pulse_train(np.arange(0.8, 19.5, 1.0))
        matrix = np.column_stack((waveform, waveform))
        with self.assertRaises(TypeError):
            detect_pulses(matrix)  # type: ignore[call-arg]
        with self.assertRaisesRegex(ValueError, "not registered"):
            detect_pulses(matrix, detector_id="unknown_detector")
        with patch(
            "ppg_frailty.peaks.resolver.detect_pulses_per_wavelength_aboy_project",
            side_effect=RuntimeError("injected canonical failure"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "injected canonical failure",
            ):
                detect_pulses(
                    matrix,
                    detector_id=CANONICAL_DETECTOR_ID,
                )

    def test_active_configs_and_named_ablation_materialization(self) -> None:
        import tempfile

        root = Path(__file__).resolve().parents[2]
        names = (
            "reference_static_role_aware_v2.yaml",
            "reference_static_feature_vector_v2.yaml",
            "reference_static_feature_matrix_v2.yaml",
            "reference_static_fusion_v2.yaml",
        )
        for name in names:
            config = load_config(root / "configs" / name)
            self.assertEqual(
                config.payload["signal"]["peak_detector"]["detector_id"],
                CANONICAL_DETECTOR_ID,
            )
            self.assertEqual(
                config.payload["signal"]["peak_detector"][
                    "min_observation_sec"
                ],
                8.0,
            )
            self.assertEqual(
                config.payload["signal"]["peak_detector"]["min_peaks"],
                5,
            )
            missing = config.to_dict()
            del missing["signal"]["peak_detector"]
            with self.assertRaisesRegex(ValueError, "peak_detector"):
                validate_config_payload(missing)

        with tempfile.TemporaryDirectory(
            dir=root / "tests" / "signal"
        ) as directory:
            output = Path(directory) / "old_detector.yaml"
            ablation = materialize_formal_ablation_config(
                root / "configs/reference_static_role_aware_v2.yaml",
                family="peak_detector",
                profile_id=ABLATION_DETECTOR_ID,
                output_path=output,
                profiles_path=root / "configs/formal_ablation_profiles_v2.yaml",
            )
            self.assertEqual(
                ablation.payload["signal"]["peak_detector"]["detector_id"],
                ABLATION_DETECTOR_ID,
            )
            identity = ablation.payload["output"][
                "formal_ablation_materialization"
            ]
            self.assertEqual(identity["family"], "peak_detector")
            self.assertEqual(identity["catalog_role"], "ablation")

    def test_peak_thresholds_bind_effective_hash_and_invalid_ranges_fail(self) -> None:
        root = Path(__file__).resolve().parents[2]
        base = load_config(root / "configs/reference_static_role_aware_v2.yaml")
        custom = base.to_dict()
        custom["signal"]["peak_detector"].update(
            {"min_observation_sec": 6.5, "min_peaks": 3}
        )
        resolved = validate_config_payload(custom)
        digest = hashlib.sha256(canonical_json_bytes(resolved)).hexdigest()
        self.assertNotEqual(digest, base.sha256)
        self.assertEqual(
            resolved["signal"]["peak_detector"],
            {
                "detector_id": CANONICAL_DETECTOR_ID,
                "failure_action": "fail_closed_no_fallback",
                "min_observation_sec": 6.5,
                "min_peaks": 3,
            },
        )
        for field, value, message in (
            ("min_observation_sec", 0.0, "min_observation_sec"),
            ("min_observation_sec", float("nan"), "min_observation_sec"),
            ("min_observation_sec", False, "min_observation_sec"),
            ("min_peaks", 0, "min_peaks"),
            ("min_peaks", 3.5, "min_peaks"),
            ("min_peaks", False, "min_peaks"),
        ):
            with self.subTest(field=field, value=value):
                invalid = copy.deepcopy(base.to_dict())
                invalid["signal"]["peak_detector"][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_config_payload(invalid)


if __name__ == "__main__":
    unittest.main()
