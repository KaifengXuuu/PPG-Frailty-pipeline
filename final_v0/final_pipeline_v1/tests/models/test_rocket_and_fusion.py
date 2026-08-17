"""ROCKET, feature baseline and FileBag fusion tests.

ROCKET、特征基线与 FileBag 融合测试。
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from ppg_frailty.models.feature_baselines import FeatureVectorBaseline
from ppg_frailty.models.fusion import FileBagFusionClassifier
from ppg_frailty.models.rocket import MiniRocketAblation, RocketRidgeClassifier, RocketTransformer


class _MeanSignalEncoder(nn.Module):
    """English: Tiny deterministic encoder used only by tests.

    中文：仅供测试使用的微型确定性编码器。
    """

    feature_dim = 2

    def forward_features(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        del mask
        return x.mean(dim=-1)


class RocketAndFusionTests(unittest.TestCase):
    """English: Exercise classical and multi-input model boundaries.

    中文：验证经典模型与多输入模型的边界。
    """

    def test_numpy_rocket_is_deterministic_and_mask_aware(self) -> None:
        """English: Invalid padded values cannot change ROCKET features.

        中文：无效补齐值不能改变 ROCKET 特征。
        """

        rng = np.random.default_rng(12)
        x = rng.normal(size=(4, 2, 24)).astype(np.float32)
        mask = np.ones((4, 24), dtype=bool)
        mask[:, 20:] = False
        changed = x.copy()
        changed[:, :, 20:] = 100_000.0
        first = RocketTransformer(n_kernels=12, seed=5).fit(x)
        second = RocketTransformer(n_kernels=12, seed=5).fit(x)
        np.testing.assert_allclose(first.transform(x, mask), second.transform(changed, mask))
        self.assertEqual(first.transform(x, mask).shape, (4, 24))

    def test_rocket_classifier_and_named_mini_ablation(self) -> None:
        """English: Main and low-cost ablation remain visibly distinct.

        中文：主路线与低成本消融保持明确区分。
        """

        rng = np.random.default_rng(4)
        x = rng.normal(size=(9, 2, 20)).astype(np.float32)
        y = np.asarray([0, 1, 2] * 3)
        ids = [f"P{index}" for index in range(9)]
        model = RocketRidgeClassifier(n_kernels=16, seed=2).fit(x, y, participant_ids=ids)
        probability = model.predict_proba(x)
        self.assertEqual(probability.shape, (9, 3))
        np.testing.assert_allclose(probability.sum(axis=1), 1.0)
        mini = MiniRocketAblation(n_kernels=8)
        self.assertEqual(mini.model_id, "minirocket_engineering_ablation")
        self.assertIn("not_reference", mini.scientific_status)

    def test_feature_baseline_freezes_feature_width(self) -> None:
        """English: Feature schema mismatch fails closed.

        中文：特征 schema 不匹配时关闭失败。
        """

        x = np.arange(27, dtype=np.float64).reshape(9, 3)
        y = np.asarray([0, 1, 2] * 3)
        model = FeatureVectorBaseline("logistic_regression", ("a", "b", "c"))
        model.fit(x, y, participant_ids=[f"P{i}" for i in range(9)])
        self.assertEqual(model.predict_proba(x).shape, (9, 3))
        with self.assertRaises(ValueError):
            model.predict_proba(x[:, :2])

    def test_file_features_are_encoded_once_per_file(self) -> None:
        """English: File features never acquire a window dimension.

        中文：文件特征绝不会获得窗口维。
        """

        model = FileBagFusionClassifier(
            _MeanSignalEncoder(),
            signal_feature_dim=2,
            n_file_features=3,
            n_classes=3,
            feature_hidden_dim=4,
            fusion_hidden_dim=5,
        ).eval()
        observed: list[tuple[int, ...]] = []
        hook = model.feature_encoder[0].register_forward_pre_hook(
            lambda module, arguments: observed.append(tuple(arguments[0].shape))
        )
        try:
            logits = model(
                torch.randn(2, 4, 2, 10),
                torch.tensor([[True, True, False, False], [True, True, True, True]]),
                torch.randn(2, 3),
                torch.ones(2, 4, 10, dtype=torch.bool),
            )
        finally:
            hook.remove()
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(observed, [(2, 3)])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

