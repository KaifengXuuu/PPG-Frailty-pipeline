"""Peak/PPI/PRV eligibility tests / 峰、间期与 PRV 准入测试。"""

from __future__ import annotations

import unittest

import numpy as np

from ppg_frailty.contracts import PulseResult, SignalRoute
from ppg_frailty.peaks import CANONICAL_DETECTOR_ID
from ppg_frailty.signal import (
    MIN_BASIC_RATE_PEAKS,
    MIN_TIME_DOMAIN_PRV_INTERVALS,
    PrvConfig,
    compute_prv,
    detect_pulses,
)
from ppg_frailty.signal.prv import _sample_entropy


def regular_pulse_result(
    seconds: int,
    ppi_s: float = 1.0,
    *,
    route: SignalRoute = SignalRoute.DIRECT,
) -> PulseResult:
    """构建不压缩时间的规则间期 / Build a regular uncompressed interval timeline."""

    timestamps = np.arange(0.0, float(seconds) + 1e-9, ppi_s)
    peaks = np.rint(timestamps * 400.0).astype(np.int64)
    intervals = np.diff(timestamps)
    starts = np.arange(intervals.size, dtype=np.int64)
    run_id = f"test_regular::{route.value}::{seconds}::{ppi_s}"
    return PulseResult(
        peaks=peaks,
        peak_timestamps_s=timestamps,
        accepted_peak_mask=np.ones(peaks.size, dtype=bool),
        interval_start_peak_indices=starts,
        interval_stop_peak_indices=starts + 1,
        ppi_s=intervals,
        valid_interval_mask=np.ones(intervals.size, dtype=bool),
        adjacency_mask=np.ones(intervals.size, dtype=bool),
        wavelength="RED",
        detector_version="test:polarity=+1",
        confidence=np.ones(peaks.size, dtype=np.float64),
        source_route=route,
        detection_run_id=run_id,
        interval_run_ids=np.full(intervals.shape, run_id),
    )


class PeakPrvTest(unittest.TestCase):
    """验证 rate 路径保留时间/邻接 / Verify rate time and adjacency contract."""

    def test_dual_polarity_detects_inverted_pulse(self) -> None:
        time = np.arange(0, 20.0, 1.0 / 400.0)
        inverted = -np.sin(2.0 * np.pi * 1.25 * time)
        matrix = np.column_stack((inverted, 0.8 * inverted))
        result = detect_pulses(
            matrix,
            detector_id=CANONICAL_DETECTOR_ID,
        )
        self.assertGreaterEqual(result.peaks.size, 20)
        self.assertTrue(np.all(result.adjacency_mask))
        self.assertGreater(float(np.mean(result.valid_interval_mask)), 0.95)

    def test_time_prv_at_sixty_seconds(self) -> None:
        pulse = regular_pulse_result(60)
        result = compute_prv(
            pulse, observation_duration_s=60.0, role="B",
            route=SignalRoute.DIRECT, q_rate_qualified=True,
        )
        self.assertTrue(result.time_domain_eligible)
        self.assertAlmostEqual(result.values["hr_mean_bpm"], 60.0, places=8)
        self.assertAlmostEqual(result.values["rmssd_s"], 0.0, places=8)
        self.assertFalse(result.frequency_domain_eligible)

    def test_nondefault_prv_config_changes_real_eligibility_and_is_persisted(self) -> None:
        """Configured thresholds/bands must reach computation, not only YAML."""

        pulse = regular_pulse_result(60)
        default = compute_prv(
            pulse,
            observation_duration_s=60.0,
            role="B",
            route=SignalRoute.DIRECT,
            q_rate_qualified=True,
        )
        configured = PrvConfig.from_mapping(
            {
                "time_prv_min_duration_s": 30.0,
                "time_prv_min_coverage": 0.5,
                "time_prv_min_intervals": 20,
                "spectral_prv_min_duration_s": 45.0,
                "spectral_prv_min_coverage": 0.5,
                "spectral_prv_min_intervals": 40,
                "tachogram_fs_hz": 8.0,
                "spectral_bands_hz": {
                    "vlf": [0.01, 0.10],
                    "lf": [0.10, 0.30],
                    "hf": [0.30, 0.80],
                },
                "sample_entropy": {
                    "m": 3,
                    "r_sd_fraction": 0.3,
                    "min_intervals": 40,
                },
            }
        )
        changed = compute_prv(
            pulse,
            observation_duration_s=60.0,
            role="B",
            route=SignalRoute.DIRECT,
            q_rate_qualified=True,
            config=configured,
        )
        self.assertFalse(default.frequency_domain_eligible)
        self.assertTrue(changed.frequency_domain_eligible, changed.reasons)
        self.assertTrue(changed.sample_entropy_eligible)
        self.assertEqual(changed.configuration, configured.to_dict())
        self.assertEqual(changed.configuration["tachogram_fs_hz"], 8.0)
        self.assertEqual(changed.configuration["sample_entropy"]["m"], 3)

    def test_prv_config_defaults_missing_fields_and_rejects_bad_ranges(self) -> None:
        defaults = PrvConfig.from_mapping({})
        self.assertEqual(defaults, PrvConfig())
        self.assertEqual(
            PrvConfig.from_mapping({"time_prv_min_duration_s": 45.0}).time_prv_min_intervals,
            MIN_TIME_DOMAIN_PRV_INTERVALS,
        )
        invalid = (
            {"time_prv_min_coverage": 1.1},
            {"time_prv_min_intervals": 1},
            {"tachogram_fs_hz": 0.0},
            {
                "tachogram_fs_hz": 1.0,
                "spectral_bands_hz": {"hf": [0.15, 0.6]},
            },
            {"sample_entropy": {"m": 4, "min_intervals": 5}},
        )
        for payload in invalid:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    PrvConfig.from_mapping(payload)

    def test_prv_parameters_are_independent_and_alias_conflicts_fail(self) -> None:
        configured = PrvConfig.from_mapping({"time_prv_min_coverage": 0.5})
        self.assertEqual(configured.time_prv_min_coverage, 0.5)
        self.assertEqual(
            configured.spectral_prv_min_coverage,
            PrvConfig().spectral_prv_min_coverage,
        )
        with self.assertRaisesRegex(ValueError, "conflicts with its deprecated"):
            PrvConfig.from_mapping(
                {
                    "rate_prv_min_peaks": 6,
                    "time_prv_min_accepted_peaks": 5,
                }
            )
        self.assertEqual(
            PrvConfig.from_mapping(
                {
                    "rate_prv_min_peaks": 6,
                    "time_prv_min_accepted_peaks": 6,
                }
            ).rate_prv_min_peaks,
            6,
        )

    def test_basic_and_time_prv_use_distinct_frozen_count_boundaries(self) -> None:
        self.assertEqual(MIN_BASIC_RATE_PEAKS, 5)
        self.assertEqual(MIN_TIME_DOMAIN_PRV_INTERVALS, 30)
        basic_only = compute_prv(
            regular_pulse_result(8, ppi_s=2.0),
            observation_duration_s=8.0,
            role="B",
            route=SignalRoute.DIRECT,
            q_rate_qualified=True,
        )
        self.assertTrue(basic_only.validity["hr_mean_bpm"])
        self.assertFalse(basic_only.time_domain_eligible)
        below_time = compute_prv(
            regular_pulse_result(60, ppi_s=60.0 / 29.0),
            observation_duration_s=60.0,
            role="B",
            route=SignalRoute.DIRECT,
            q_rate_qualified=True,
        )
        self.assertEqual(below_time.values["accepted_interval_count"], 29.0)
        self.assertFalse(below_time.time_domain_eligible)
        at_time = compute_prv(
            regular_pulse_result(60, ppi_s=2.0),
            observation_duration_s=60.0,
            role="B",
            route=SignalRoute.DIRECT,
            q_rate_qualified=True,
        )
        self.assertEqual(at_time.values["accepted_interval_count"], 30.0)
        self.assertTrue(at_time.time_domain_eligible)

    def test_frequency_prv_accepts_direct_and_rate_only(self) -> None:
        pulse = regular_pulse_result(300)
        direct = compute_prv(
            pulse, observation_duration_s=300.0, role="B",
            route=SignalRoute.DIRECT, q_rate_qualified=True,
        )
        artifact = compute_prv(
            regular_pulse_result(300, route=SignalRoute.ARTIFACT_RATE_ONLY),
            observation_duration_s=300.0,
            role="B",
            route=SignalRoute.ARTIFACT_RATE_ONLY,
            q_rate_qualified=True,
        )
        self.assertTrue(direct.frequency_domain_eligible)
        self.assertTrue(artifact.frequency_domain_eligible)
        self.assertTrue(artifact.validity["lf_power_s2"])

    def test_frequency_prv_requires_canonical_role_family(self) -> None:
        """仅规范 B/R 具备长时静态资格 / Only canonical B/R are eligible."""

        pulse = regular_pulse_result(
            300, route=SignalRoute.ARTIFACT_RATE_ONLY
        )
        canonical = compute_prv(
            pulse,
            observation_duration_s=300.0,
            role="R",
            route=SignalRoute.ARTIFACT_RATE_ONLY,
            q_rate_qualified=True,
        )
        self.assertTrue(canonical.frequency_domain_eligible, canonical.reasons)
        for noncanonical in ("R1", "R2", "R3", "R4", "relax", "baseline"):
            with self.subTest(role=noncanonical):
                result = compute_prv(
                    pulse,
                    observation_duration_s=300.0,
                    role=noncanonical,
                    route=SignalRoute.ARTIFACT_RATE_ONLY,
                    q_rate_qualified=True,
                )
                self.assertFalse(result.frequency_domain_eligible)

    def test_sample_entropy_uses_normalized_match_probabilities(self) -> None:
        """锁定正确概率归一 / Lock the correctly normalized SampEn estimator."""

        x = 1.0 + 0.08 * np.sin(np.linspace(0.0, 12.0 * np.pi, 80))
        tolerance = 0.2 * np.std(x, ddof=1)

        def probability(length: int) -> float:
            templates = x.size - length + 1
            matches = 0
            for left in range(templates - 1):
                for right in range(left + 1, templates):
                    if np.max(
                        np.abs(
                            x[left : left + length]
                            - x[right : right + length]
                        )
                    ) <= tolerance:
                        matches += 1
            return matches / (templates * (templates - 1) / 2.0)

        expected = -np.log(probability(3) / probability(2))
        self.assertAlmostEqual(_sample_entropy(x), expected, places=12)

    def test_rmssd_does_not_cross_a_rejected_interval(self) -> None:
        """拒绝区间两侧不得形成差分 / A rejected interval breaks differencing."""

        pulse = regular_pulse_result(70)
        ppi = 1.0 + 0.03 * np.sin(np.arange(pulse.ppi_s.size) * 0.37)
        valid = np.ones_like(ppi, dtype=bool)
        valid[31] = False
        broken = PulseResult(
            peaks=pulse.peaks,
            peak_timestamps_s=pulse.peak_timestamps_s,
            accepted_peak_mask=pulse.accepted_peak_mask,
            interval_start_peak_indices=pulse.interval_start_peak_indices,
            interval_stop_peak_indices=pulse.interval_stop_peak_indices,
            ppi_s=ppi,
            valid_interval_mask=valid,
            adjacency_mask=pulse.adjacency_mask,
            wavelength=pulse.wavelength,
            detector_version=pulse.detector_version,
            confidence=pulse.confidence,
            source_route=pulse.source_route,
            detection_run_id=pulse.detection_run_id,
            interval_run_ids=pulse.interval_run_ids,
        )
        result = compute_prv(
            broken,
            observation_duration_s=70.0,
            role="B",
            route=SignalRoute.DIRECT,
            q_rate_qualified=True,
        )
        pair_mask = valid[:-1] & valid[1:]
        expected = np.sqrt(np.mean(np.square(np.diff(ppi)[pair_mask])))
        self.assertAlmostEqual(result.values["rmssd_s"], expected, places=12)

    def test_rejected_interval_blocks_differences_and_spectral_bridge(self) -> None:
        pulse = regular_pulse_result(
            400, route=SignalRoute.ARTIFACT_RATE_ONLY
        )
        valid = pulse.valid_interval_mask.copy()
        valid[199] = False
        broken = PulseResult(
            peaks=pulse.peaks,
            peak_timestamps_s=pulse.peak_timestamps_s,
            accepted_peak_mask=pulse.accepted_peak_mask,
            interval_start_peak_indices=pulse.interval_start_peak_indices,
            interval_stop_peak_indices=pulse.interval_stop_peak_indices,
            ppi_s=pulse.ppi_s,
            valid_interval_mask=valid,
            adjacency_mask=pulse.adjacency_mask,
            wavelength=pulse.wavelength,
            detector_version=pulse.detector_version,
            confidence=pulse.confidence,
            source_route=pulse.source_route,
            detection_run_id=pulse.detection_run_id,
            interval_run_ids=pulse.interval_run_ids,
        )
        result = compute_prv(
            broken,
            observation_duration_s=400.0,
            role="B",
            route=SignalRoute.ARTIFACT_RATE_ONLY,
            q_rate_qualified=True,
        )
        self.assertFalse(result.frequency_domain_eligible)
        self.assertIn(
            "frequency_prv_configured_requirements_not_met",
            result.reasons,
        )
        # 所有保留PPI相同；如果错误跨gap，RMSSD数值仍可能为0，因此检查pair数派生pNN。
        self.assertAlmostEqual(result.values["rmssd_s"], 0.0)
        self.assertIn(
            "sample_entropy_uses_longest_contiguous_run",
            result.reasons,
        )

    def test_prv_rejects_route_or_run_identity_drift(self) -> None:
        pulse = regular_pulse_result(60)
        with self.assertRaisesRegex(ValueError, "route must match"):
            compute_prv(
                pulse,
                observation_duration_s=60.0,
                role="B",
                route=SignalRoute.ARTIFACT_RATE_ONLY,
                q_rate_qualified=True,
            )
        mismatched = PulseResult(
            **{
                **pulse.__dict__,
                "interval_run_ids": np.full(
                    pulse.ppi_s.shape, "different_detection_run"
                ),
            }
        )
        with self.assertRaisesRegex(ValueError, "cannot cross"):
            compute_prv(
                mismatched,
                observation_duration_s=60.0,
                role="B",
                route=SignalRoute.DIRECT,
                q_rate_qualified=True,
            )


if __name__ == "__main__":
    unittest.main()
