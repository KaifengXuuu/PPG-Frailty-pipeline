"""V2 named historical reducer smoke tests / V2 具名历史 reducer 冒烟测试。"""

from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from ppg_frailty.artifact.legacy import (
    CeemdLiteNlmsLegacyConfig,
    DwtA2LegacyReducer,
    EmdSiftingConfig,
)
from ppg_frailty.artifacts import get_reducer


def fixture(samples: int = 800) -> np.ndarray:
    """Synthetic two-channel PPG / 合成双通道 PPG。"""

    time = np.arange(samples, dtype=np.float64) / 400.0
    slow = 0.35 * np.sin(2.0 * np.pi * 0.25 * time)
    return np.column_stack((
        np.sin(2.0 * np.pi * 1.2 * time) + slow,
        0.8 * np.sin(2.0 * np.pi * 1.15 * time + 0.2) - slow,
    ))


class LegacyReducerTests(unittest.TestCase):
    """Verify named, nonidentity, fail-closed routes / 验证具名故障闭合支线。"""

    def test_emd_named_route_is_nonidentity_rate_only(self) -> None:
        source = fixture()
        reducer = get_reducer(
            "emd_sifting_rate_only",
            {"max_imfs": 2, "max_sift": 3, "sd_threshold": 0.2},
        )
        self.assertIsInstance(reducer.config, EmdSiftingConfig)
        result = reducer.reduce(source, None)
        self.assertEqual(result.status, "success", result.reasons)
        self.assertFalse(result.is_identity)
        self.assertEqual(result.x_ar.shape, source.shape)
        self.assertFalse(np.array_equal(result.x_ar, source))
        self.assertTrue(result.diagnostics["rate_only"])
        self.assertEqual(result.diagnostics["representation"], "feature_vector")

    def test_ceemd_nlms_is_deterministic_with_frozen_seed(self) -> None:
        source = fixture()
        parameters = {
            "pairs": 1,
            "noise_ratio": 0.2,
            "max_imfs": 2,
            "max_sift": 2,
            "sd_threshold": 0.2,
            "protect_bandwidth_hz": 0.25,
            "protect_harmonics": 2,
            "low_motion_hz": 0.4,
            "high_motion_hz": 6.0,
            "nlms_length": 8,
            "nlms_mu": 0.1,
            "nlms_leak": 1e-4,
            "random_seed": 2025,
        }
        reducer = get_reducer("ceemd_lite_nlms_legacy", parameters)
        self.assertIsInstance(reducer.config, CeemdLiteNlmsLegacyConfig)
        first = reducer.reduce(source, None)
        second = reducer.reduce(source, None)
        self.assertEqual(first.status, "success", first.reasons)
        self.assertTrue(np.array_equal(first.x_ar, second.x_ar))
        self.assertFalse(np.array_equal(first.x_ar, source))

    def test_dwt_never_identity_fallback_and_ans_is_absent(self) -> None:
        source = fixture()
        result = DwtA2LegacyReducer().reduce(source, None)
        self.assertIn(result.status, {"success", "unsupported"})
        if result.status == "success":
            self.assertFalse(np.array_equal(result.x_ar, source))
            self.assertTrue(result.diagnostics["rate_only"])
        else:
            self.assertIsNone(result.x_ar)
            self.assertTrue(result.diagnostics["identity_fallback_forbidden"])
        with self.assertRaises(KeyError):
            get_reducer("ans")


if __name__ == "__main__":
    unittest.main()
