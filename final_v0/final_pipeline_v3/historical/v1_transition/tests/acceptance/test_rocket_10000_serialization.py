"""ROCKET 10,000-kernel 正式配置序列化 / Full ROCKET serialization test.

中文：与 reduced 64-kernel synthetic 比较不同，本测试真实生成 10,000 个核，
仅用 outer-training fixture 拟合 scaler/kernel/ridge，经 joblib 保存与加载后比较
held-out matrix 的 transform 和概率逐元素一致。它是序列化/隔离合约测试，
不是 frailty 性能基准。

English: unlike the reduced 64-kernel comparison, this test really generates 10,000
kernels, fits scaler/kernels/ridge on an outer-training fixture only, then verifies
element-wise held-out transform and probability parity after joblib save/load. It is
a serialization/isolation contract test, not a frailty benchmark.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from ppg_frailty.models.rocket_ridge import RocketRidgeClassifier
from tools.acceptance_gate import PIPELINE_ROOT


class RocketTenThousandKernelSerializationTest(unittest.TestCase):
    """验证 reference kernel count 与完整状态 parity / Verify full state parity."""

    def test_rocket_10000_full_config_serialization_parity(self) -> None:
        """10,000 个 learned kernels 必须可复现 / All 10,000 learned kernels round-trip."""

        rng = np.random.default_rng(10_000)
        train_values = rng.normal(size=(6, 1, 32)).astype(np.float32)
        train_labels = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
        train_ids = tuple(f"outer_train_{index}" for index in range(6))
        train_mask = np.ones((6, 32), dtype=bool)
        train_mask[:, -2:] = False
        heldout_values = rng.normal(size=(1, 1, 32)).astype(np.float32)
        heldout_mask = np.ones((1, 32), dtype=bool)
        heldout_mask[:, -2:] = False
        heldout_id = "outer_heldout_never_fit"

        model = RocketRidgeClassifier(n_kernels=10_000, alpha=1.0, seed=42)
        model.fit(
            train_values,
            train_labels,
            mask=train_mask,
            participant_ids=train_ids,
        )
        self.assertEqual(model.n_kernels, 10_000)
        self.assertIsNotNone(model.transformer.kernels_)
        self.assertEqual(len(model.transformer.kernels_), 10_000)
        self.assertEqual(model.fitted_participant_ids_, tuple(sorted(train_ids)))
        self.assertNotIn(heldout_id, model.fitted_participant_ids_)
        for state in model.fitted_object_provenance_.values():
            self.assertEqual(tuple(state["fitted_participant_ids"]), tuple(sorted(train_ids)))
            self.assertNotIn(heldout_id, state["fitted_participant_ids"])

        scaled_before = model.scaler.transform(heldout_values, heldout_mask)
        transformed_before = model.transformer.transform(scaled_before, heldout_mask)
        probability_before = model.predict_proba(heldout_values, heldout_mask)
        self.assertEqual(transformed_before.shape, (1, 20_000))
        self.assertEqual(probability_before.shape, (1, 3))
        self.assertTrue(np.isfinite(transformed_before).all())
        self.assertTrue(np.isfinite(probability_before).all())
        np.testing.assert_allclose(probability_before.sum(axis=1), 1.0, atol=1e-12)

        temporary_parent = PIPELINE_ROOT / "artifacts/acceptance/tmp"
        temporary_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="rocket_10000_", dir=temporary_parent) as directory:
            target = Path(directory) / "rocket_10000.joblib"
            joblib.dump(model, target, compress=3)
            self.assertGreater(target.stat().st_size, 0)
            loaded = joblib.load(target)
            self.assertEqual(len(loaded.transformer.kernels_), 10_000)
            scaled_after = loaded.scaler.transform(heldout_values, heldout_mask)
            transformed_after = loaded.transformer.transform(scaled_after, heldout_mask)
            probability_after = loaded.predict_proba(heldout_values, heldout_mask)

        np.testing.assert_array_equal(transformed_after, transformed_before)
        np.testing.assert_allclose(probability_after, probability_before, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()

