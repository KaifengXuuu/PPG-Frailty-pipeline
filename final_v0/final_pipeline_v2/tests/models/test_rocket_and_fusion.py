"""Feature baseline and FileBag fusion tests.

特征基线与 FileBag 融合测试。
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from ppg_frailty.models.feature_baselines import FeatureVectorBaseline
from ppg_frailty.models.factory import normalize_model_id
from ppg_frailty.models.fusion import FileBagFusionClassifier


class _MeanSignalEncoder(nn.Module):
    """English: Tiny deterministic encoder used only by tests.

    中文：仅供测试使用的微型确定性编码器。
    """

    feature_dim = 2

    def forward_features(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        del mask
        return x.mean(dim=-1)


class FeatureAndFusionTests(unittest.TestCase):
    """English: Exercise classical and multi-input model boundaries.

    中文：验证经典模型与多输入模型的边界。
    """

    def test_retired_rocket_id_is_not_executable(self) -> None:
        for model_id in ("ROCKET", "MiniROCKET", "rocket_numpy"):
            with self.subTest(model_id=model_id):
                with self.assertRaisesRegex(ValueError, "unsupported model"):
                    normalize_model_id(model_id)

    def test_feature_baseline_freezes_feature_width(self) -> None:
        """English: Feature schema mismatch fails closed.

        中文：特征 schema 不匹配时关闭失败。
        """

        x = np.arange(27, dtype=np.float64).reshape(9, 3)
        y = np.asarray([0, 1, 2] * 3)
        model = FeatureVectorBaseline(
            "logistic_regression",
            ("a", "b", "c"),
            logistic_c=1.0,
            logistic_max_iter=5000,
            logistic_solver="lbfgs",
        )
        model.fit(x, y, participant_ids=[f"P{i}" for i in range(9)])
        self.assertEqual(model.predict_proba(x).shape, (9, 3))
        with self.assertRaises(ValueError):
            model.predict_proba(x[:, :2])

    def test_sparse_screening_parameters_reach_sklearn_estimators(self) -> None:
        """The bounded search fields must alter the actual estimator."""

        logistic = FeatureVectorBaseline(
            "logistic_regression",
            ("a", "b"),
            logistic_c=10.0,
            logistic_max_iter=5000,
            logistic_solver="lbfgs",
        )
        self.assertEqual(logistic.pipeline.named_steps["model"].C, 10.0)

        trees = FeatureVectorBaseline(
            "extra_trees",
            ("a", "b"),
            extra_trees_n_estimators=200,
            extra_trees_n_jobs=1,
            extra_trees_max_features=0.5,
            extra_trees_min_samples_leaf=5,
        )
        estimator = trees.pipeline.named_steps["model"]
        self.assertEqual(estimator.max_features, 0.5)
        self.assertEqual(estimator.min_samples_leaf, 5)
        with self.assertRaisesRegex(ValueError, "max_features"):
            FeatureVectorBaseline(
                "extra_trees",
                ("a", "b"),
                extra_trees_n_estimators=200,
                extra_trees_n_jobs=1,
                extra_trees_max_features="auto",
                extra_trees_min_samples_leaf=1,
            )

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
