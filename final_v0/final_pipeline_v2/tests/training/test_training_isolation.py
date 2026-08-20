"""Frozen membership, epoch selection and evaluation isolation tests.

冻结成员、epoch 选择与评估隔离测试。
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score
from torch import nn

from ppg_frailty.models.feature_baselines import FeatureVectorBaseline
from ppg_frailty.config import validate_config_payload
from ppg_frailty.training import (
    FeatureVectorDataset,
    FrozenOuterSplit,
    RawWindowDataset,
    SampleIdentity,
    TrainingConfig,
    UnifiedTrainer,
    build_inner_grouped_split,
    evaluate_predictions,
    predict_torch_dataset,
)
from ppg_frailty.training.trainer import (
    configured_class_weight_vector,
    outer_train_effective_number_weights,
    outer_train_inverse_frequency_weights,
    participant_file_window_sampling_weights,
    resolve_torch_training_device,
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

    def test_cuda_device_preflight_is_deterministic_and_never_falls_back(self) -> None:
        previous = os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        self.addCleanup(
            lambda: (
                os.environ.__setitem__("CUBLAS_WORKSPACE_CONFIG", previous)
                if previous is not None
                else os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            )
        )
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CPU fallback is forbidden"):
                resolve_torch_training_device(
                    "cuda", deterministic_algorithms=True
                )
        self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = "invalid"
        with self.assertRaisesRegex(RuntimeError, "CUBLAS_WORKSPACE_CONFIG"):
            resolve_torch_training_device("cuda", deterministic_algorithms=True)
        self.assertEqual(
            resolve_torch_training_device(
                "cpu", deterministic_algorithms=True
            ).type,
            "cpu",
        )

    def test_actual_reference_yaml_constructs_training_config(self) -> None:
        """English: Resolved YAML fields are consumed without renaming or loss.

        中文：resolved YAML 字段未经重命名或丢失即可构造训练配置。
        """

        pipeline_root = Path(__file__).resolve().parents[2]
        payload = yaml.safe_load(
            (pipeline_root / "configs" / "reference_static_role_aware_v2.yaml").read_text(encoding="utf-8")
        )
        config = TrainingConfig.from_mapping(payload["training"])
        self.assertEqual(config.epoch_rule, "fixed_epoch")
        self.assertEqual(config.optimizer, "adam")
        self.assertEqual(config.sampler, "balance_line_weighted_v2")
        self.assertEqual(config.class_weighting, "inverse_frequency")
        self.assertEqual(config.class_count_basis, "participant")
        legacy = TrainingConfig(epoch_rule="fixed")
        self.assertEqual(legacy.epoch_rule, "fixed_epoch")
        self.assertEqual(legacy.legacy_epoch_rule_alias, "fixed")

        derived = TrainingConfig(
            fixed_epochs=7,
            execution_mode="smoke",
            epoch_profile="caller_supplied_label",
        )
        self.assertEqual(derived.execution_mode, "formal")
        self.assertEqual(derived.epoch_profile, "ablation_7")
        self.assertEqual(
            derived.to_mapping(),
            TrainingConfig(fixed_epochs=7).to_mapping(),
        )
        override = derived._with_epoch_override(1)
        self.assertEqual(override.fixed_epochs, 1)
        self.assertEqual(override.execution_mode, "smoke")
        self.assertEqual(override.epoch_profile, "smoke")

        for field in ("optimizer", "sampler", "class_weighting"):
            payload["training"].pop(field)
        effective = validate_config_payload(payload)
        resolved = TrainingConfig.from_mapping(effective["training"])
        self.assertEqual(resolved.device, "cuda")
        result = UnifiedTrainer(
            replace(
                resolved,
                execution_mode="smoke",
                epoch_profile="smoke",
                fixed_epochs=1,
                # This unit test verifies configuration consumption, not the
                # host GPU.  Production reference YAML remains CUDA-first and
                # the dedicated resolver test above covers fail-closed CUDA.
                device="cpu",
            )
        ).fit(_TinyClassifier, self.dataset, self.split)
        self.assertEqual(result.provenance.optimizer, "adam")
        self.assertEqual(result.provenance.sampler, "balance_line_weighted_v2")
        self.assertEqual(
            result.provenance.class_weighting,
            "inverse_frequency",
        )

    def test_training_algorithm_controls_are_independent_and_executable(self) -> None:
        """Registered alternatives dispatch real runtime implementations."""

        config = TrainingConfig(
            optimizer="adamw",
            sampler="uniform_replacement",
            class_weighting="outer_train_window_inverse_frequency",
        )
        trainer = UnifiedTrainer(config)
        optimizer = trainer._optimizer(_TinyClassifier())
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        loader = trainer._loader(self.dataset, shuffle=True)
        self.assertIsInstance(loader.sampler, torch.utils.data.WeightedRandomSampler)
        self.assertTrue(loader.sampler.replacement)
        np.testing.assert_allclose(
            configured_class_weight_vector(
                self.dataset,
                class_weighting="outer_train_window_inverse_frequency",
                n_classes=3,
            ),
            (1.0, 1.0, 1.0),
        )

        exhaustive = UnifiedTrainer(
            TrainingConfig(
                sampler="exhaustive_shuffle_without_replacement",
                class_weighting="none",
            )
        )
        exhaustive_loader = exhaustive._loader(self.dataset, shuffle=True)
        self.assertIsInstance(
            exhaustive_loader.sampler, torch.utils.data.RandomSampler
        )
        self.assertFalse(exhaustive_loader.sampler.replacement)
        self.assertIsNone(exhaustive._criterion(self.dataset).weight)

        nondeterministic = UnifiedTrainer(
            TrainingConfig(
                deterministic_algorithms=False,
                cache_policy="disabled",
            )
        )
        nondeterministic._set_seed()
        self.assertFalse(torch.are_deterministic_algorithms_enabled())
        # Restore the process-global torch setting for later tests.
        UnifiedTrainer(TrainingConfig())._set_seed()
        self.assertTrue(torch.are_deterministic_algorithms_enabled())

    def test_training_numeric_ranges_and_enums_fail_closed(self) -> None:
        for kwargs in (
            {"optimizer": "invented"},
            {"sampler": "invented"},
            {"class_weighting": "invented"},
            {"loss": "invented"},
            {"loss": "weighted_ce", "class_weighting": "none"},
            {"loss": "balanced_softmax", "class_weighting": "effective_number"},
            {"focal_gamma": -1.0},
            {"focal_gamma": float("nan")},
            {"loss": "cross_entropy", "focal_gamma": 1.5},
            {"class_weight_beta": -0.1},
            {"class_weight_beta": 1.0},
            {"class_weight_beta": float("inf")},
            {"class_weighting": "none", "class_weight_beta": 0.9},
            {"deterministic_algorithms": "true"},
            {"cache_policy": "invented"},
            {"learning_rate": 0.0},
            {"learning_rate": float("nan")},
            {"learning_rate": "0.001"},
            {"weight_decay": -1.0},
            {"weight_decay": float("inf")},
            {"weight_decay": "0.0"},
            {"batch_size": True},
            {"num_workers": -1},
            {"seed": -1},
            {"seed": 2**32},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrainingConfig(**kwargs)

    def test_epoch_seed_above_uint32_maps_numpy_deterministically(self) -> None:
        trainer = UnifiedTrainer(TrainingConfig(seed=2**32 - 1))
        absolute_seed = (2**32 - 1) + 1_000_000
        trainer._set_absolute_seed(absolute_seed)
        first_numpy = np.random.random(4)
        first_torch = torch.rand(4)
        trainer._set_absolute_seed(absolute_seed)
        np.testing.assert_array_equal(first_numpy, np.random.random(4))
        torch.testing.assert_close(first_torch, torch.rand(4), rtol=0.0, atol=0.0)

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
            TrainingConfig(
                fixed_epochs=2,
                batch_size=3,
                learning_rate=1e-2,
                seed=8,
                execution_mode="smoke",
                epoch_profile="smoke",
            )
        )
        result = trainer.fit(_TinyClassifier, self.dataset, self.split)
        self.assertEqual(result.selected_epoch, 2)
        self.assertEqual(result.provenance.fitted_participant_ids, tuple(f"P{i}" for i in range(6)))
        self.assertIn("outer_labels_not_accepted_by_fit_api", result.provenance.notes)
        metric_rows = [
            row
            for row in result.history
            if "training_participant_balanced_accuracy" in row
        ]
        self.assertEqual([row["epoch"] for row in metric_rows], [1, 2])
        probability, labels, identities = trainer.predict_probabilities(
            result.model, self.dataset
        )
        participant_probability, participant_labels = (
            trainer._participant_training_predictions(
                probability,
                labels,
                identities,
                balance_line=result.provenance.expected_aggregation_rule,
            )
        )
        expected = balanced_accuracy_score(
            participant_labels, participant_probability.argmax(axis=1)
        )
        self.assertAlmostEqual(
            metric_rows[-1]["training_participant_balanced_accuracy"], expected
        )
        self.assertTrue(
            all(row["training_data_scope"] == "full_outer_train_only" for row in metric_rows)
        )
        self.assertTrue(all(not row["outer_heldout_used"] for row in metric_rows))
        self.assertTrue(
            all(not row["metric_used_for_selection_or_checkpoint"] for row in metric_rows)
        )

    def test_train_balanced_accuracy_respects_declared_line_a_and_line_b(self) -> None:
        """Train BA follows file- or role-balanced participant aggregation."""

        identities = (
            _identity("P0", 0, "b"),
            SampleIdentity("P0", "P0_R_1", "R1", 0, "direct", window_id="r1"),
            SampleIdentity("P0", "P0_R_2", "R2", 0, "direct", window_id="r2"),
            SampleIdentity("P0", "P0_R_3", "R3", 0, "direct", window_id="r3"),
            SampleIdentity("P0", "P0_R_4", "R4", 0, "direct", window_id="r4"),
            _identity("P1", 1, "b"),
        )
        probability = np.asarray(
            (
                (1.0, 0.0),
                (0.2, 0.8),
                (0.2, 0.8),
                (0.2, 0.8),
                (0.2, 0.8),
                (0.0, 1.0),
            ),
            dtype=np.float64,
        )
        labels = np.asarray([identity.label for identity in identities], dtype=np.int64)
        line_a, line_a_labels = UnifiedTrainer._participant_training_predictions(
            probability,
            labels,
            identities,
            balance_line="line_a_equal_files",
        )
        line_b, line_b_labels = UnifiedTrainer._participant_training_predictions(
            probability,
            labels,
            identities,
            balance_line="line_b_equal_role_families",
        )
        np.testing.assert_array_equal(line_a_labels, (0, 1))
        np.testing.assert_array_equal(line_b_labels, (0, 1))
        self.assertEqual(line_a.argmax(axis=1).tolist(), [1, 1])
        self.assertEqual(line_b.argmax(axis=1).tolist(), [0, 1])

    def test_outer_subject_in_training_dataset_fails(self) -> None:
        """English: Membership is checked before optimiser construction.

        中文：在构造优化器之前校验成员关系。
        """

        bad_identities = (*self.identities[:-1], _identity("P6", 2))
        bad_dataset = RawWindowDataset(self.dataset.values.copy(), bad_identities)
        trainer = UnifiedTrainer(
            TrainingConfig(
                fixed_epochs=1,
                batch_size=3,
                execution_mode="smoke",
                epoch_profile="smoke",
            )
        )
        with self.assertRaises(ValueError):
            trainer.fit(_TinyClassifier, bad_dataset, self.split)

    def test_inner_selection_is_outer_train_only_then_fresh_full_refit(self) -> None:
        """The selected epoch never sees OOF participants and refits a new model."""

        config = TrainingConfig(
            epoch_rule="inner_grouped_selection",
            epoch_profile="inner_grouped_selection",
            inner_grouped_folds=2,
            maximum_inner_epochs=2,
            inner_patience=1,
            batch_size=3,
            training_balance="equal_files",
        )
        inner = build_inner_grouped_split(
            self.dataset,
            self.split,
            n_folds=config.inner_grouped_folds,
            seed=config.seed,
        )
        self.assertEqual(
            inner,
            build_inner_grouped_split(
                self.dataset,
                self.split,
                n_folds=config.inner_grouped_folds,
                seed=config.seed,
            ),
        )
        all_validation_folds = tuple(
            build_inner_grouped_split(
                self.dataset,
                self.split,
                n_folds=2,
                seed=config.seed,
                validation_fold_index=index,
            ).validation_participant_ids
            for index in range(2)
        )
        self.assertFalse(set(all_validation_folds[0]) & set(all_validation_folds[1]))
        self.assertEqual(
            set(all_validation_folds[0]) | set(all_validation_folds[1]),
            set(self.split.train_participant_ids),
        )
        with self.assertRaisesRegex(ValueError, "smallest outer-train"):
            build_inner_grouped_split(
                self.dataset,
                self.split,
                n_folds=3,
                seed=config.seed,
            )
        self.assertEqual(
            set(inner.train_participant_ids) | set(inner.validation_participant_ids),
            set(self.split.train_participant_ids),
        )
        self.assertFalse(
            set(inner.train_participant_ids) & set(self.split.oof_participant_ids)
        )
        self.assertFalse(
            set(inner.validation_participant_ids) & set(self.split.oof_participant_ids)
        )
        constructed: list[_TinyClassifier] = []

        def factory() -> _TinyClassifier:
            model = _TinyClassifier()
            constructed.append(model)
            return model

        result = UnifiedTrainer(config).fit(
            factory,
            self.dataset,
            self.split,
            inner_split=inner,
        )
        self.assertEqual(len(constructed), 2)
        self.assertIs(result.model, constructed[1])
        self.assertIsNot(result.model, constructed[0])
        self.assertIn(result.selected_epoch, (1, 2))
        self.assertEqual(
            set(result.provenance.fitted_participant_ids),
            set(self.split.train_participant_ids),
        )
        self.assertEqual(result.provenance.inner_grouped_folds, 2)
        self.assertEqual(result.provenance.inner_membership_hash, inner.membership_hash)
        self.assertEqual(
            result.provenance.inner_train_participant_ids,
            tuple(sorted(inner.train_participant_ids)),
        )
        self.assertEqual(
            result.provenance.inner_validation_participant_ids,
            tuple(sorted(inner.validation_participant_ids)),
        )
        self.assertTrue(
            any("inner_participant_balanced_accuracy" in row for row in result.history)
        )
        self.assertEqual(
            {
                row["inner_selection_aggregation_rule"]
                for row in result.history
                if "inner_selection_aggregation_rule" in row
            },
            {config.expected_aggregation_rule},
        )
        self.assertTrue(
            all(
                row.get("outer_heldout_used") is False
                for row in result.history
                if "outer_heldout_used" in row
            )
        )

    def test_loss_strategies_and_effective_number_weights_execute(self) -> None:
        identities = tuple(
            _identity(participant, label)
            for participant, label in (
                ("A", 0),
                ("B", 1),
                ("C", 1),
                ("D", 2),
                ("E", 2),
                ("F", 2),
            )
        )
        dataset = RawWindowDataset(
            np.ones((6, 2, 12), dtype=np.float32),
            identities,
        )
        effective = outer_train_effective_number_weights(
            dataset,
            3,
            beta=0.9,
        )
        expected = (1.0 - 0.9) / (1.0 - np.power(0.9, np.asarray((1, 2, 3))))
        expected /= expected.mean()
        np.testing.assert_allclose(effective, expected, rtol=1e-6)

        logits = torch.tensor(
            [[1.2, -0.5, 0.2], [-0.4, 0.8, 0.1], [0.1, -0.2, 1.1]],
            dtype=torch.float32,
        )
        targets = torch.tensor((0, 1, 2), dtype=torch.long)
        balanced = UnifiedTrainer(
            TrainingConfig(loss="balanced_softmax", class_weighting="none")
        )._criterion(dataset)
        expected_balanced = nn.functional.cross_entropy(
            logits + torch.log(torch.tensor((1.0, 2.0, 3.0))),
            targets,
        )
        torch.testing.assert_close(balanced(logits, targets), expected_balanced)

        weighted = UnifiedTrainer(
            TrainingConfig(
                loss="weighted_ce",
                class_weighting="effective_number",
                class_weight_beta=0.9,
            )
        )._criterion(dataset)
        weighted_config = TrainingConfig(
            loss="weighted_ce",
            class_weighting="effective_number",
            class_weight_beta=0.9,
        )
        self.assertEqual(weighted_config.loss, "cross_entropy")
        self.assertEqual(weighted_config.legacy_loss_alias, "weighted_ce")
        self.assertEqual(
            weighted_config.to_mapping(),
            TrainingConfig(
                loss="cross_entropy",
                class_weighting="effective_number",
                class_weight_beta=0.9,
            ).to_mapping(),
        )
        expected_weighted = nn.functional.cross_entropy(
            logits,
            targets,
            weight=torch.as_tensor(effective),
        )
        torch.testing.assert_close(weighted(logits, targets), expected_weighted)

        focal = UnifiedTrainer(
            TrainingConfig(
                loss="focal_loss",
                class_weighting="effective_number",
                class_weight_beta=0.9,
                focal_gamma=1.5,
            )
        )._criterion(dataset)
        focal_value = focal(logits, targets)
        self.assertTrue(torch.isfinite(focal_value))
        self.assertGreater(float(focal_value), 0.0)
        split = FrozenOuterSplit(
            repeat=0,
            fold=0,
            seed=42,
            train_participant_ids=tuple("ABCDEF"),
            oof_participant_ids=("OOF",),
            registry_hash="registry-loss-test",
            fold_hash="fold-loss-test",
        )
        result = UnifiedTrainer(
            TrainingConfig(
                execution_mode="smoke",
                epoch_profile="smoke",
                fixed_epochs=1,
                batch_size=3,
                loss="focal_loss",
                class_weighting="effective_number",
                class_weight_beta=0.9,
                focal_gamma=1.5,
            )
        ).fit(_TinyClassifier, dataset, split)
        self.assertEqual(result.provenance.loss, "focal_loss")
        self.assertEqual(result.provenance.class_weighting, "effective_number")
        self.assertEqual(result.provenance.class_weight_beta, 0.9)
        self.assertEqual(result.provenance.focal_gamma, 1.5)
        self.assertEqual(result.provenance.class_counts, (1.0, 2.0, 3.0))
        self.assertEqual(
            result.provenance.class_weight_count_basis,
            "participant",
        )
        np.testing.assert_allclose(result.provenance.class_weight_vector, effective)

    def test_estimator_uses_same_membership_guard(self) -> None:
        """English: Classical models preserve exact fitted subject provenance.

        中文：经典模型同样保留精确拟合 subject provenance。
        """

        values = np.arange(18, dtype=np.float32).reshape(6, 3)
        dataset = FeatureVectorDataset(values, ("a", "b", "c"), self.identities)
        estimator = FeatureVectorBaseline(
            "logistic_regression",
            ("a", "b", "c"),
            logistic_c=1.0,
            logistic_max_iter=5000,
            logistic_solver="lbfgs",
        )
        result = UnifiedTrainer(
            TrainingConfig()
        ).fit_estimator(
            estimator, dataset, self.split
        )
        self.assertEqual(result.provenance.fitted_participant_ids, tuple(f"P{i}" for i in range(6)))
        self.assertEqual(result.provenance.sampler, "balance_line_weighted_v2")
        self.assertEqual(
            result.provenance.class_weighting,
            "inverse_frequency",
        )
        self.assertEqual(result.provenance.optimizer, "not_applicable")
        self.assertEqual(result.provenance.epoch_profile, "not_applicable")

    def test_estimator_rejects_torch_only_training_fields_before_fit(self) -> None:
        config = TrainingConfig(fixed_epochs=37)
        with self.assertRaisesRegex(
            ValueError,
            "execution_backend=estimator does not support.*fixed_epochs",
        ):
            config.validate_for_execution_backend("estimator")
        config.validate_for_execution_backend("torch")

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
result = UnifiedTrainer(TrainingConfig()).fit_estimator(
    FeatureVectorBaseline(
        "logistic_regression",
        ("a", "b", "c"),
        logistic_c=1.0,
        logistic_max_iter=5000,
        logistic_solver="lbfgs",
    ),
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
