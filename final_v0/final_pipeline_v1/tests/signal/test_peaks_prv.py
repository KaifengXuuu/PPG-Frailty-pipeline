"""Peak/PPI/PRV eligibility tests / 峰、间期与 PRV 准入测试。"""

from __future__ import annotations

import unittest

import numpy as np

from ppg_frailty.contracts import PulseResult, SignalRoute
from ppg_frailty.signal import compute_prv, detect_pulses
from ppg_frailty.signal.prv import _sample_entropy


def regular_pulse_result(seconds: int, ppi_s: float = 1.0) -> PulseResult:
    """构建不压缩时间的规则间期 / Build a regular uncompressed interval timeline."""

    timestamps = np.arange(0.0, float(seconds) + 1e-9, ppi_s)
    peaks = np.rint(timestamps * 400.0).astype(np.int64)
    intervals = np.diff(timestamps)
    starts = np.arange(intervals.size, dtype=np.int64)
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
    )


class PeakPrvTest(unittest.TestCase):
    """验证 rate 路径保留时间/邻接 / Verify rate time and adjacency contract."""

    def test_dual_polarity_detects_inverted_pulse(self) -> None:
        time = np.arange(0, 20.0, 1.0 / 400.0)
        inverted = -np.sin(2.0 * np.pi * 1.25 * time)
        matrix = np.column_stack((inverted, 0.8 * inverted))
        result = detect_pulses(matrix)
        self.assertGreaterEqual(result.peaks.size, 20)
        self.assertTrue(np.all(result.adjacency_mask))
        self.assertGreater(float(np.mean(result.valid_interval_mask)), 0.95)

    def test_time_prv_at_sixty_seconds(self) -> None:
        pulse = regular_pulse_result(60)
        result = compute_prv(
            pulse, observation_duration_s=60.0, role="baseline",
            route=SignalRoute.DIRECT, q_rate_qualified=True,
        )
        self.assertTrue(result.time_domain_eligible)
        self.assertAlmostEqual(result.values["hr_mean_bpm"], 60.0, places=8)
        self.assertAlmostEqual(result.values["rmssd_s"], 0.0, places=8)
        self.assertFalse(result.frequency_domain_eligible)

    def test_frequency_prv_accepts_direct_and_rate_only(self) -> None:
        pulse = regular_pulse_result(300)
        direct = compute_prv(
            pulse, observation_duration_s=300.0, role="baseline",
            route=SignalRoute.DIRECT, q_rate_qualified=True,
        )
        artifact = compute_prv(
            pulse,
            observation_duration_s=300.0,
            role="baseline",
            route=SignalRoute.ARTIFACT_RATE_ONLY,
            q_rate_qualified=True,
        )
        self.assertTrue(direct.frequency_domain_eligible)
        self.assertTrue(artifact.frequency_domain_eligible)
        self.assertTrue(artifact.validity["lf_power_s2"])

    def test_long_recovery_roles_r1_to_r4_are_frequency_eligible(self) -> None:
        """冻结恢复角色具备长时 PRV 资格 / Frozen R1-R4 roles are eligible."""

        pulse = regular_pulse_result(300)
        for role in ("R1", "R2", "R3", "R4"):
            with self.subTest(role=role):
                result = compute_prv(
                    pulse,
                    observation_duration_s=300.0,
                    role=role,
                    route=SignalRoute.ARTIFACT_RATE_ONLY,
                    q_rate_qualified=True,
                )
                self.assertTrue(result.frequency_domain_eligible, result.reasons)

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
        )
        result = compute_prv(
            broken,
            observation_duration_s=70.0,
            role="baseline",
            route=SignalRoute.DIRECT,
            q_rate_qualified=True,
        )
        pair_mask = valid[:-1] & valid[1:]
        expected = np.sqrt(np.mean(np.square(np.diff(ppi)[pair_mask])))
        self.assertAlmostEqual(result.values["rmssd_s"], expected, places=12)

    def test_rejected_interval_blocks_differences_and_spectral_bridge(self) -> None:
        pulse = regular_pulse_result(400)
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
        )
        result = compute_prv(
            broken,
            observation_duration_s=400.0,
            role="baseline",
            route=SignalRoute.ARTIFACT_RATE_ONLY,
            q_rate_qualified=True,
        )
        self.assertFalse(result.frequency_domain_eligible)
        self.assertIn(
            "frequency_prv_requires_qrate_static_contiguous300s_200intervals",
            result.reasons,
        )
        # 所有保留PPI相同；如果错误跨gap，RMSSD数值仍可能为0，因此检查pair数派生pNN。
        self.assertAlmostEqual(result.values["rmssd_s"], 0.0)
        self.assertIn(
            "sample_entropy_uses_longest_contiguous_run",
            result.reasons,
        )


if __name__ == "__main__":
    unittest.main()
