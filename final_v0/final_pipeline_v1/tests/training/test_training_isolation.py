"""Frozen membership, epoch selection and evaluation isolation tests.

冻结成员、epoch 选择与评估隔离测试。
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from ppg_frailty.models.feature_baselines import FeatureVectorBaseline
from ppg_frailty.training import (
    FeatureVectorDataset,
    FrozenOuterSplit,
    InnerGroupedSplit,
    RawWindowDataset,
    SampleIdentity,
    TrainingConfig,
    UnifiedTrainer,
    evaluate_predictions,
    predict_torch_dataset,
)
from ppg_frailty.training.trainer import (
    outer_train_inverse_frequency_weights,
    participant_file_window_sampling_weights,
)


class _TinyClassifier(nn.Module):
    """English: Small network used to test protocol behavior, not accuracy.

    中文：用于测试协议行为而非准确率的微型网络。
    """

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(2, 3)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            feature = x.mean(dim=-1)
        else:
            weights = mask.to(x.dtype).unsqueeze(1)
            feature = (x * weights).sum(dim=-1) / weights.sum(dim=-1)
        return self.classifier(feature)


def _identity(participant: str, label: int, suffix: str = "0") -> SampleIdentity:
    """English: Build deterministic test metadata / 中文：构造确定性测试元数据。"""

    return SampleIdentity(
        participant_id=participant,
        file_id=f"{participant}_B_{suffix}",
        role="B",
        label=label,
        signal_route="direct",
        window_id=f"{participant}_w_{suffix}",
    )


class TrainingIsolationTests(unittest.TestCase):
    """English: Prove outer labels cannot reach fitting decisions.

    中文：证明 outer 标签无法进入拟合决策。
    """

    def setUp(self) -> None:
        rng = np.random.default_rng(10)
        self.identities = tuple(_identity(f"P{i}", i % 3) for i in range(6))
        self.dataset = RawWindowDataset(
            rng.normal(size=(6, 2, 12)).astype(np.float32), self.identities
        )
        self.split = FrozenOuterSplit(
            repeat=0,
            fold=0,
            seed=42,
            train_participant_ids=tuple(f"P{i}" for i in range(6)),
            oof_participant_ids=("P6", "P7", "P8"),
            registry_hash="registry-test",
            fold_hash="fold-test",
        )

    def test_fit_api_has_no_outer_label_argument(self) -> None:
        """English: Structural guard against the historical leakage API.

        中文：从结构上阻止历史上的泄漏式 API。
        """

        parameters = set(inspect.signature(UnifiedTrainer.fit).parameters)
        self.assertFalse({"outer_labels", "outer_y", "outer_oof_dataset", "validation_dataset"} & parameters)

    def test_actual_reference_yaml_constructs_training_config(self) -> None:
        """English: Resolved YAML fields are consumed without renaming or loss.

        中文：resolved YAML 字段未经重命名或丢失即可构造训练配置。
        """

        pipeline_root = Path(__file__).resolve().parents[2]
        payload = yaml.safe_load(
            (pipeline_root / "configs" / "reference_static_v1.yaml").read_text(encoding="utf-8")
        )
        config = TrainingConfig.from_mapping(payload["training"])
        self.assertEqual(config.epoch_rule, "fixed_epoch")
        self.assertEqual(config.optimizer, "adam")
        self.assertEqual(config.sampler, "participant_file_window_balanced_v1")
        self.assertEqual(config.class_weighting, "outer_train_inverse_frequency")
        legacy = TrainingConfig(epoch_rule="fixed")
        self.assertEqual(legacy.epoch_rule, "fixed_epoch")
        self.assertEqual(legacy.legacy_epoch_rule_alias, "fixed")

    def test_participant_file_window_sampler_and_class_weights(self) -> None:
        """English: Training probability mass is balanced at declared levels.

        中文：训练抽样概率质量按照声明的层级均衡。
        """

        identities = (
            _identity("P0", 0, "0"),
            _identity("P0", 0, "0"),
            _identity("P0", 0, "1"),
            _identity("P1", 1, "0"),
        )
        # English/中文: Make the first two rows one file and the third another.
        identities = (
            identities[0],
            SampleIdentity(**{**identities[1].__dict__, "window_id": "P0_w_second"}),
            SampleIdentity(**{**identities[2].__dict__, "file_id": "P0_B_other"}),
            identities[3],
        )
        dataset = RawWindowDataset(np.ones((4, 2, 12), dtype=np.float32), identities)
        weights = participant_file_window_sampling_weights(dataset)
        np.testing.assert_allclose(weights, (0.125, 0.125, 0.25, 0.5))
        np.testing.assert_allclose(
            outer_train_inverse_frequency_weights(dataset, 3),
            (1.0, 1.0, 0.0),
        )

    def test_fixed_epoch_fit_records_exact_membership(self) -> None:
        """English: Fixed epochs train only frozen outer-train subjects.

        中文：固定 epoch 只训练冻结 outer-train subject。
        """

        trainer = UnifiedTrainer(
            TrainingConfig(fixed_epochs=1, batch_size=3, learning_rate=1e-2, seed=8)
        )
        result = trainer.fit(_TinyClassifier, self.dataset, self.split)
        self.assertEqual(result.selected_epoch, 1)
        self.assertEqual(result.provenance.fitted_participant_ids, tuple(f"P{i}" for i in range(6)))
        self.assertIn("outer_labels_not_accepted_by_fit_api", result.provenance.notes)

    def test_outer_subject_in_training_dataset_fails(self) -> None:
        """English: Membership is checked before optimiser construction.

        中文：在构造优化器之前校验成员关系。
        """

        bad_identities = (*self.identities[:-1], _identity("P6", 2))
        bad_dataset = RawWindowDataset(self.dataset.values.copy(), bad_identities)
        trainer = UnifiedTrainer(TrainingConfig(fixed_epochs=1, batch_size=3))
        with self.assertRaises(ValueError):
            trainer.fit(_TinyClassifier, bad_dataset, self.split)

    def test_inner_selection_discards_model_then_refits_full_outer_train(self) -> None:
        """English: Selection and final-fit models are distinct instances.

        中文：选择模型与最终拟合模型是不同实例。
        """

        calls: list[_TinyClassifier] = []

        def factory() -> _TinyClassifier:
            model = _TinyClassifier()
            calls.append(model)
            return model

        trainer = UnifiedTrainer(
            TrainingConfig(
                epoch_rule="inner_grouped_selection",
                inner_grouped_folds=2,
                maximum_inner_epochs=1,
                inner_patience=1,
                batch_size=3,
                seed=9,
            )
        )
        result = trainer.fit(
            factory,
            self.dataset,
            self.split,
            inner_split=InnerGroupedSplit(
                train_participant_ids=("P0", "P1", "P2"),
                validation_participant_ids=("P3", "P4", "P5"),
            ),
        )
        self.assertEqual(len(calls), 2)
        self.assertIs(result.model, calls[1])
        self.assertEqual(result.provenance.fitted_participant_ids, tuple(f"P{i}" for i in range(6)))
        self.assertTrue(
            any(
                row.get("inner_selection_unit") == "participant"
                for row in result.history
            )
        )

    def test_estimator_uses_same_membership_guard(self) -> None:
        """English: Classical models preserve exact fitted subject provenance.

        中文：经典模型同样保留精确拟合 subject provenance。
        """

        values = np.arange(18, dtype=np.float32).reshape(6, 3)
        dataset = FeatureVectorDataset(values, ("a", "b", "c"), self.identities)
        estimator = FeatureVectorBaseline("logistic_regression", ("a", "b", "c"))
        result = UnifiedTrainer(TrainingConfig(fixed_epochs=1)).fit_estimator(
            estimator, dataset, self.split
        )
        self.assertEqual(result.provenance.fitted_participant_ids, tuple(f"P{i}" for i in range(6)))

    def test_prediction_only_evaluation_checks_oof_membership(self) -> None:
        """English: Evaluator accepts OOF rows and computes strict metrics.

        中文：评估器只接受 OOF 行并计算严格指标。
        """

        identities = tuple(_identity(f"P{i}", i % 3) for i in range(6, 9))
        dataset = RawWindowDataset(np.ones((3, 2, 12), dtype=np.float32), identities)
        probability, labels, returned = predict_torch_dataset(
            _TinyClassifier(), dataset, self.split, batch_size=2
        )
        self.assertEqual(len(returned), 3)
        metrics = evaluate_predictions(labels, probability, class_order=(0, 1, 2))
        self.assertEqual(metrics.n_rows, 3)
        self.assertEqual(len(metrics.confusion_matrix), 3)

    def test_estimator_path_imports_and_fits_when_torch_is_blocked(self) -> None:
        """English: Optional deep dependency cannot disable classical routes.

        中文：缺少可选深度依赖时，经典路线仍可导入并拟合。
        """

        source_root = Path(__file__).resolve().parents[2] / "src"
        script = r"""
import builtins
import numpy as np
original_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("simulated missing torch")
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
from ppg_frailty.models.feature_baselines import FeatureVectorBaseline
from ppg_frailty.training import FeatureVectorDataset, FrozenOuterSplit, SampleIdentity, TrainingConfig, UnifiedTrainer
identities = tuple(
    SampleIdentity(f"P{i}", f"F{i}", "B", i % 3, "direct", window_id=f"W{i}")
    for i in range(6)
)
dataset = FeatureVectorDataset(np.arange(18, dtype=np.float32).reshape(6, 3), ("a", "b", "c"), identities)
split = FrozenOuterSplit(0, 0, 42, tuple(f"P{i}" for i in range(6)), ("P6",), "registry", "fold")
result = UnifiedTrainer(TrainingConfig(fixed_epochs=1)).fit_estimator(
    FeatureVectorBaseline("logistic_regression", ("a", "b", "c")),
    dataset,
    split,
)
assert len(result.provenance.fitted_participant_ids) == 6
print("NO_TORCH_ESTIMATOR_OK")
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(source_root)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("NO_TORCH_ESTIMATOR_OK", completed.stdout)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
