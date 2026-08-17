"""Architecture, masking and factory contract tests.

架构、掩码与模型工厂契约测试。
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from ppg_frailty.models import ModelInputSpec, create_model
from ppg_frailty.models.compact_cnn import CompactCNN1D, trainable_parameter_count
from ppg_frailty.models.inception import (
    FullInceptionTimeSingleNetwork,
    InceptionTimeFiveMemberProbabilityEnsemble,
    SmallInceptionTimeSingleNetwork,
)
from ppg_frailty.models.shapeformer import (
    ExperimentalShapeFormer,
    discover_effect_size_shapelets,
)


class ReviewedArchitectureTests(unittest.TestCase):
    """English: Lock reviewed architectures against accidental drift.

    中文：锁定已审查架构，防止参数规模意外漂移。
    """

    def test_exact_parameter_counts(self) -> None:
        """English: Counts are part of the reviewed model identity.

        中文：参数量是已审查模型身份的一部分。
        """

        self.assertEqual(trainable_parameter_count(CompactCNN1D(8, 3)), 79_139)
        self.assertEqual(trainable_parameter_count(FullInceptionTimeSingleNetwork(8, 3)), 456_579)
        self.assertEqual(trainable_parameter_count(SmallInceptionTimeSingleNetwork(8, 3)), 57_027)

    def test_output_shapes_and_matrix_mask(self) -> None:
        """English: Padded matrix columns cannot alter valid predictions.

        中文：特征矩阵补齐列的占位值不得改变有效预测。
        """

        compact = CompactCNN1D(8, 3).eval()
        self.assertEqual(tuple(compact(torch.randn(2, 8, 64)).shape), (2, 3))

        small = SmallInceptionTimeSingleNetwork(4, 3).eval()
        x_first = torch.randn(2, 4, 32)
        x_second = x_first.clone()
        mask = torch.zeros(2, 32, dtype=torch.bool)
        mask[:, :24] = True
        x_second[:, :, 24:] = torch.randn(2, 4, 8) * 1_000.0
        with torch.no_grad():
            first = small(x_first, mask)
            second = small(x_second, mask)
        torch.testing.assert_close(first, second, atol=1e-6, rtol=0.0)

    def test_five_member_probability_ensemble(self) -> None:
        """English: Five distinct members average probabilities, not logits.

        中文：五个不同成员平均概率，而不是平均 logits。
        """

        members = [SmallInceptionTimeSingleNetwork(2, 3) for _ in range(5)]
        ensemble = InceptionTimeFiveMemberProbabilityEnsemble(members, [1, 2, 3, 4, 5]).eval()
        x = torch.randn(3, 2, 48)
        with torch.no_grad():
            member_probability = ensemble.member_probabilities(x)
            average = ensemble.predict_probabilities(x)
        self.assertEqual(tuple(member_probability.shape), (5, 3, 3))
        torch.testing.assert_close(average, member_probability.mean(dim=0))
        torch.testing.assert_close(average.sum(dim=1), torch.ones(3))

    @staticmethod
    def _effect_size_fixture() -> tuple[np.ndarray, object]:
        """Build one explicit outer-train bank / 构建一个显式 outer-train 库。"""

        rng = np.random.default_rng(7)
        x = rng.normal(size=(6, 2, 32)).astype(np.float32)
        y = np.asarray([0, 0, 1, 1, 2, 2])
        ids = [f"P{index}" for index in range(6)]
        shapelets = discover_effect_size_shapelets(
            x,
            y,
            ids,
            discovery_method="effect_size_shapelets_v1",
            input_fs_hz=100.0,
            shapelet_length=4,
            outer_repeat_index=2,
            outer_fold_index=3,
            per_class=1,
            stride=4,
            max_candidates_per_class=4,
        )
        return x, shapelets

    def test_effect_size_shapeformer_is_self_contained_experimental(self) -> None:
        """English: Discovery records fold/time identity and needs no PISD.

        中文：发现过程记录 fold/时间身份，且不依赖 PISD。
        """

        x, shapelets = self._effect_size_fixture()
        self.assertEqual(shapelets.fitted_participant_ids, tuple(f"P{i}" for i in range(6)))
        self.assertEqual(shapelets.discovery_method, "effect_size_shapelets_v1")
        self.assertEqual(shapelets.shapelet_length_samples, 4)
        self.assertAlmostEqual(shapelets.shapelet_length_seconds, 0.04)
        self.assertEqual((shapelets.outer_repeat_index, shapelets.outer_fold_index), (2, 3))

        model = ExperimentalShapeFormer(
            2,
            3,
            shapelets,
            hidden_channels=8,
            attention_heads=2,
            patch_size_samples=4,
            input_fs_hz=100.0,
        ).eval()
        self.assertEqual(model.model_status, "experimental")
        self.assertFalse(model.external_pisd_supported)
        self.assertFalse(model.raw_sample_token_attention)
        self.assertEqual(model.attention_input_route, "non_overlapping_patch_embedding")
        self.assertEqual(model.patch_embedding.kernel_size, (4,))
        self.assertEqual(model.patch_embedding.stride, (4,))
        self.assertEqual(tuple(model(torch.from_numpy(x[:2])).shape), (2, 3))
        self.assertEqual(model.provenance()["shapelet_length_seconds"], 0.04)

    def test_shapeformer_patch_route_is_mask_aware(self) -> None:
        """English: Invalid tail values cannot change patch/shapelet output.

        中文：无效尾部占位值不能改变 patch 或 shapelet 输出。
        """

        x, shapelets = self._effect_size_fixture()
        model = ExperimentalShapeFormer(
            2,
            3,
            shapelets,
            hidden_channels=8,
            attention_heads=2,
            patch_size_samples=4,
            input_fs_hz=100.0,
        ).eval()
        first = torch.from_numpy(x[:2].copy())
        second = first.clone()
        mask = torch.zeros(2, 32, dtype=torch.bool)
        mask[:, :24] = True
        second[:, :, 24:] = 10_000.0
        with torch.no_grad():
            expected = model(first, mask)
            observed = model(second, mask)
        torch.testing.assert_close(expected, observed, atol=1e-6, rtol=0.0)

    def test_shapeformer_discovery_and_factory_fail_closed(self) -> None:
        """English: PISD/unknown/missing identities never use effect-size fallback.

        中文：PISD、未知或缺失身份绝不回退到效应量路线。
        """

        x, shapelets = self._effect_size_fixture()
        y = np.asarray([0, 0, 1, 1, 2, 2])
        ids = [f"P{index}" for index in range(6)]
        with self.assertRaisesRegex(ValueError, "never fall back"):
            discover_effect_size_shapelets(
                x,
                y,
                ids,
                discovery_method="pisd",
                input_fs_hz=100.0,
                shapelet_length=4,
                outer_repeat_index=0,
                outer_fold_index=0,
            )
        input_spec = ModelInputSpec("raw", n_channels=2, n_classes=3)
        provenance_config = {
            "model_id": "shapeformer_effect_size",
            "shapelets": shapelets,
            "discovery_method": "effect_size_shapelets_v1",
            "input_fs_hz": 100.0,
            "outer_repeat_index": 2,
            "outer_fold_index": 3,
            "outer_train_participant_hash": shapelets.outer_train_participant_hash,
        }
        with self.assertRaisesRegex(ValueError, "missing required options"):
            create_model(
                {"model_id": "shapeformer_effect_size", "shapelets": shapelets},
                input_spec,
            )
        with self.assertRaisesRegex(ValueError, "never fall back"):
            create_model(
                {**provenance_config, "discovery_method": "pisd"},
                input_spec,
            )
        with self.assertRaisesRegex(ValueError, "input_fs_hz does not match"):
            create_model(
                {**provenance_config, "input_fs_hz": 200.0},
                input_spec,
            )
        with self.assertRaisesRegex(ValueError, "outer repeat/fold does not match"):
            create_model(
                {**provenance_config, "outer_fold_index": 4},
                input_spec,
            )
        with self.assertRaisesRegex(ValueError, "roster hash does not match"):
            create_model(
                {**provenance_config, "outer_train_participant_hash": "0" * 64},
                input_spec,
            )
        with self.assertRaisesRegex(ValueError, "raw sample-token attention"):
            create_model(
                {
                    **provenance_config,
                    "patch_size_samples": 1,
                    "hidden_channels": 8,
                    "attention_heads": 2,
                },
                input_spec,
            )
        model = create_model(
            {
                **provenance_config,
                "patch_size_samples": 4,
                "hidden_channels": 8,
                "attention_heads": 2,
            },
            input_spec,
        )
        self.assertEqual(model.outer_fold_index, 3)
        mapped_model = create_model(
            {
                **provenance_config,
                "shapelets": dict(shapelets.__dict__),
                "patch_size_samples": 4,
                "hidden_channels": 8,
                "attention_heads": 2,
            },
            input_spec,
        )
        self.assertEqual(mapped_model.provenance(), model.provenance())

    def test_factory_covers_four_representation_modes(self) -> None:
        """English: Every frozen representation has an executable route.

        中文：每个冻结 representation 都有可执行路线。
        """

        raw = create_model(
            {"model_id": "compact_cnn"},
            ModelInputSpec("raw", n_channels=8, n_classes=3),
        )
        vector = create_model(
            {"model_id": "logistic_regression"},
            ModelInputSpec("feature_vector", feature_names=("a", "b"), n_classes=3),
        )
        matrix = create_model(
            {"model_name": "InceptionTimeMatrix", "variant": "small"},
            ModelInputSpec("feature_matrix", n_channels=4, n_classes=3),
        )
        fusion = create_model(
            {"model_id": "fusion_compact"},
            ModelInputSpec("fusion", n_channels=8, n_file_features=5, n_classes=3),
        )
        self.assertIsInstance(raw, CompactCNN1D)
        self.assertEqual(vector.model_id, "logistic_regression")
        self.assertIsInstance(matrix, SmallInceptionTimeSingleNetwork.__bases__[0])
        self.assertEqual(fusion.feature_encoder[0].in_features, 5)

    def test_registry_and_strict_route_boundaries(self) -> None:
        """English: Human names normalise once; invalid aliases/routes fail.

        中文：人类名称只规范化一次；无效别名与路线关闭失败。
        """

        model = create_model(
            {"model_name": "CompactCNN1D"},
            ModelInputSpec("raw", n_channels=8, n_classes=3),
        )
        self.assertEqual(model.model_id, "compact_cnn")
        self.assertEqual(model.canonical_model_name, "CompactCNN1D")
        with self.assertRaises(ValueError):
            create_model(
                {"model_id": "rocket_numpy", "n_kernels": 4},
                ModelInputSpec("raw", n_channels=8, n_classes=3),
            )
        with self.assertRaises(ValueError):
            create_model(
                {
                    "model_id": "inception_five_member_ensemble",
                    "member_seeds": [1, 2, 3, 4, 4],
                },
                ModelInputSpec("raw", n_channels=8, n_classes=3),
            )
        with self.assertRaises(ValueError):
            create_model(
                {"model_name": "UnknownModel"},
                ModelInputSpec("raw", n_channels=8, n_classes=3),
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
