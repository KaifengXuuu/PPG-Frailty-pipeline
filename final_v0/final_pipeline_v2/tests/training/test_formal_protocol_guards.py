"""Formal §5.12–§5.14 protocol regression tests.

正式 §5.12–§5.14 协议回归测试。
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ppg_frailty.bundle.save import REQUIRED_V2_METADATA, save_bundle_strict
from ppg_frailty.evaluate.aggregate import aggregate_hierarchy_strict
from ppg_frailty.evaluate.metrics import evaluate_participant_probabilities
from ppg_frailty.evaluate.oof import validate_oof_contract
from ppg_frailty.models import ModelInputSpec
from ppg_frailty.training import (
    REQUIRED_METADATA,
    FrozenOuterSplit,
    OofPredictionRow,
    RawWindowDataset,
    SampleIdentity,
    TrainingConfig,
    UnifiedTrainer,
    aggregate_hierarchy,
    assert_repeated_bundle_parity,
    current_runtime_environment,
    evaluate_predictions,
    load_bundle,
    input_spec_sha256,
    paired_fold_seed_deltas,
    predict_bundle_raw,
    read_oof_parquet,
    save_bundle,
    summarize_repeat_metric,
    validate_dataset_identity_coherence,
    validate_expected_oof_roster,
    write_oof_parquet,
)


class _TinyEstimator:
    """Deterministic lightweight estimator / 确定性的轻量 estimator。"""

    model_id = "logistic_regression"

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return fixed valid probabilities / 返回固定的有效概率。"""

        rows = int(np.asarray(x).shape[0])
        return np.tile(np.asarray((0.2, 0.3, 0.5), dtype=np.float64), (rows, 1))


class _RawAdapter:
    """Serializable raw-record boundary / 可序列化 raw-record 边界。"""

    representation_mode = "feature_vector"
    input_schema_hash = input_spec_sha256(
        ModelInputSpec(
            "feature_vector",
            n_classes=3,
            feature_names=("feature_a", "feature_b"),
        )
    )
    allowed_role_families = ("B", "R")
    boundary = "already_preprocessed_file_record_to_model_input"

    def transform_record(self, raw_record: np.ndarray) -> dict[str, np.ndarray]:
        """Map one raw payload to model inputs / 将 raw payload 映射为模型输入。"""

        return {"x": np.asarray(raw_record, dtype=np.float32).reshape(1, 2)}


class _BrokenAdapter:
    """Adapter that proves staging cleanup / 用于证明 staging 清理的 adapter。"""

    def __getstate__(self) -> dict[str, object]:
        """Fail during serialization / 在序列化期间主动失败。"""

        raise RuntimeError("intentional serialization failure")


class _TinyTorchClassifier(nn.Module):
    """Small deterministic training-isolation model / 小型确定性隔离测试模型。"""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(2, 3)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Pool time then classify / 时间池化后分类。"""

        return self.classifier(x.mean(dim=-1))


def _identity(participant: str, label: int, suffix: str = "0") -> SampleIdentity:
    """Build one row identity / 构造单行身份。"""

    return SampleIdentity(
        participant_id=participant,
        file_id=f"{participant}_B_{suffix}",
        role="B",
        label=label,
        signal_route="direct",
        window_id=f"{participant}_W_{suffix}",
    )


def _window_row(
    participant: str,
    *,
    retained: bool = True,
    manifest_hash: str = "manifest",
) -> OofPredictionRow:
    """Build one fully identified aggregation row / 构造完整身份聚合行。"""

    return OofPredictionRow(
        participant_id=participant,
        file_id=f"{participant}_B",
        role="B",
        label=0,
        probabilities=(0.8, 0.1, 0.1) if retained else (),
        repeat=0,
        fold=0,
        split_seed=42,
        training_seed=42,
        config_hash="config",
        manifest_hash=manifest_hash,
        fold_hash="fold",
        preprocessing_hash="preprocess",
        feature_hash="feature",
        model_hash="model",
        representation_mode="raw",
        signal_route="direct",
        quality_score=1.0 if retained else 0.0,
        retained=retained,
        level="window",
        window_id=f"{participant}_W",
        rejection_reason=None if retained else "drop_route",
    )


def _formal_participant_row(
    participant: str,
    fold: int,
    *,
    member_index: int | None = None,
    prediction_kind: str | None = None,
    retained: bool = True,
) -> OofPredictionRow:
    """Build one formal participant-level OOF row / 构造正式 participant OOF 行。"""

    kind = prediction_kind or (
        "ensemble_member" if member_index is not None else "single_model"
    )
    member_seeds = (42, 10042, 20042, 30042, 40042)
    training_seed = (
        member_seeds[member_index]
        if kind == "ensemble_member" and member_index is not None
        else 42 if kind == "single_model" else None
    )
    return OofPredictionRow(
        participant_id=participant,
        file_id=f"participant::{participant}",
        role="participant",
        label=fold % 3,
        probabilities=(0.7, 0.2, 0.1) if retained else (),
        repeat=0,
        fold=fold,
        split_seed=42,
        training_seed=training_seed,
        config_hash="formal_config",
        manifest_hash="manifest",
        fold_hash="fold_registry",
        preprocessing_hash="preprocessing",
        feature_hash="features",
        model_hash="model",
        representation_mode="raw",
        signal_route="direct",
        quality_score=1.0 if retained else 0.0,
        retained=retained,
        level="participant",
        member_index=member_index,
        prediction_kind=kind,
        member_training_seeds=member_seeds if kind == "ensemble_average" else (),
        ensemble_base_model_id=(
            "inception_full" if kind.startswith("ensemble_") else ""
        ),
        class_order=(0, 1, 2) if retained else (),
        code_commit="commit",
        data_schema_id="record_schema_v1",
        feature_schema_id="feature_schema_v1",
        model_version="model_v1",
        aggregation_rule="window_file_role_participant_equal",
        environment_hash="environment",
        manifest_version="manifest_v1",
        fold_registry_version="fold_v1",
        artifact_reducer_name="identity",
        artifact_reducer_version="identity_v1",
        route_status="retained" if retained else "dropped",
        source_snapshot_hash="snapshot",
        rejection_reason=None if retained else "drop_route",
    )


def _formal_metadata() -> dict[str, object]:
    """Return complete §5.14 metadata / 返回完整 §5.14 metadata。"""

    return {
        "model_identity": {
            "name": "LogisticRegressionL2",
            "machine_id": "logistic_regression",
            "version": "test_v1",
        },
        "representation_mode": "feature_vector",
        "signal_route": "direct",
        "class_order": [0, 1, 2],
        "channel_schema": ["feature_a", "feature_b"],
        "preprocessing": {"name": "unit_test", "version": "v1"},
        "preprocessing_hash": "preprocessing",
        "resampling": {"status": "not_applicable", "method": "none"},
        "window_plan": {"status": "not_applicable", "name": "file_level"},
        "feature_registry": {"name": "FeatureRegistryV1", "hash": "features"},
        "feature_hash": "features",
        "feature_vector_schema": {"columns": ["feature_a", "feature_b"]},
        "ordered_matrix_schema": {"status": "not_applicable"},
        "mask_semantics": {"feature_validity": "true_is_valid"},
        "validity_policy": {"unavailable": "nan_and_false"},
        "fitted_objects": ["model:TinyEstimator"],
        "representation_state": {"kind": "not_applicable"},
        "pooling_rule": "window_file_role_participant_equal",
        "aggregation_rule": "window_file_role_participant_equal",
        "manifest_hash": "manifest",
        "fold_hash": "fold",
        "manifest_version": "manifest_v1",
        "fold_registry_version": "fold_v1",
        "pipeline_generation": "final_pipeline_v2",
        "config_hash": "a" * 64,
        "balance_hash": "b" * 64,
        "run_hash": "c" * 64,
        "source_snapshot_hash": "d" * 64,
        "code_version": "commit",
        "environment": current_runtime_environment(),
        "dependency_status": "approved_test_dependencies",
        "serialization_trust": {
            "trusted_local_only": True,
            "authenticated_signature": False,
        },
        "golden_case": {"id": "tiny_case", "n_samples": 1},
    }


class AggregationAndMetricsProtocolTests(unittest.TestCase):
    """Full identity, coverage and scientific metric guards.

    完整实验身份、coverage 与科学指标守卫。
    """

    def test_different_experiment_identity_is_never_mixed(self) -> None:
        """Every declared identity field must survive every hierarchy level.

        每个声明的实验身份字段都必须保留到每个聚合层级。
        """

        first = _window_row("P1")
        changed_values = {
            "config_hash": "other_config",
            "manifest_hash": "other_manifest",
            "fold_hash": "other_fold",
            "preprocessing_hash": "other_preprocess",
            "feature_hash": "other_feature",
            "model_hash": "other_model",
            "representation_mode": "feature_matrix",
            "signal_route": "spectral_mask",
        }
        for field_name, changed_value in changed_values.items():
            with self.subTest(identity_field=field_name):
                second = replace(
                    first,
                    probabilities=(0.1, 0.1, 0.8),
                    **{field_name: changed_value},
                )
                result = aggregate_hierarchy((first, second))
                self.assertEqual(len(result.file_rows), 2)
                self.assertEqual(len(result.participant_rows), 2)

    def test_drop_rows_remain_in_denominator(self) -> None:
        """No-result participants remain visible in coverage / 无结果 participant 留在分母。"""

        result = aggregate_hierarchy((_window_row("P1"), _window_row("P2", retained=False)))
        self.assertEqual(len(result.source_rows), 2)
        self.assertEqual(len(result.dropped_rows), 1)
        self.assertEqual(len(result.participant_rows), 1)
        participant_coverage = next(item for item in result.coverage if item.level == "participant")
        self.assertEqual((participant_coverage.n_total, participant_coverage.n_retained), (2, 1))
        self.assertAlmostEqual(participant_coverage.coverage_rate, 0.5)
        all_dropped = aggregate_hierarchy((_window_row("P3", retained=False),))
        self.assertEqual(all_dropped.participant_rows, ())
        self.assertEqual(all_dropped.coverage[-1].coverage_rate, 0.0)

    def test_metrics_include_per_class_calibration_and_coverage(self) -> None:
        """Dropped probability rows are excluded only after coverage is fixed.

        drop 概率行只会在 coverage 分母固定之后被排除。
        """

        labels = np.asarray((0, 1, 2, 2))
        probabilities = np.asarray(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (np.nan, np.nan, np.nan))
        )
        metrics = evaluate_predictions(
            labels,
            probabilities,
            class_order=(0, 1, 2),
            retained_mask=np.asarray((True, True, True, False)),
        )
        self.assertEqual(len(metrics.per_class), 3)
        self.assertEqual(metrics.worst_class_f1, 1.0)
        self.assertEqual(metrics.multiclass_brier, 0.0)
        self.assertEqual(metrics.expected_calibration_error, 0.0)
        self.assertEqual((metrics.n_total, metrics.n_retained, metrics.n_dropped), (4, 3, 1))
        self.assertAlmostEqual(metrics.coverage_rate, 0.75)

    def test_repeat_ci_and_paired_keys_are_explicit(self) -> None:
        """Dispersion and pairing conventions are frozen / 冻结离散度与配对约定。"""

        summary = summarize_repeat_metric((1.0, 2.0, 3.0))
        self.assertAlmostEqual(summary.mean, 2.0)
        self.assertAlmostEqual(summary.population_sd, np.sqrt(2.0 / 3.0))
        self.assertAlmostEqual(summary.sample_sd, 1.0)
        self.assertLess(summary.ci95_lower, summary.mean)
        paired = paired_fold_seed_deltas(
            {(0, 0, 42): 0.5, (0, 1, 42): 0.6},
            {(0, 0, 42): 0.7, (0, 1, 42): 0.5},
        )
        np.testing.assert_allclose(paired.deltas, (0.2, -0.1))
        with self.assertRaises(ValueError):
            paired_fold_seed_deltas({(0, 0, 42): 0.5}, {(0, 1, 42): 0.5})


class FormalOofProtocolTests(unittest.TestCase):
    """Exact held-out Cartesian product and ensemble guards.

    精确 held-out 笛卡尔积与 ensemble 守卫。
    """

    def test_exact_single_model_roster(self) -> None:
        """Every expected subject appears exactly once / 每个预期 subject 恰好出现一次。"""

        rows = (_formal_participant_row("P1", 0), _formal_participant_row("P2", 1))
        roster = {(0, 0, 42): ("P1",), (0, 1, 42): ("P2",)}
        validate_expected_oof_roster(
            rows, roster, expected_config_hashes=("formal_config",)
        )
        with self.assertRaises(ValueError):
            validate_expected_oof_roster(
                rows[:1], roster, expected_config_hashes=("formal_config",)
            )

    def test_five_member_completeness(self) -> None:
        """Ensemble indices must be exactly 0..4 / ensemble 编号必须恰为 0..4。"""

        member_rows = tuple(
            _formal_participant_row("P1", 0, member_index=index) for index in range(5)
        )
        rows = member_rows + (
            _formal_participant_row(
                "P1", 0, prediction_kind="ensemble_average"
            ),
        )
        roster = {(0, 0, 42): ("P1",)}
        validate_expected_oof_roster(
            rows,
            roster,
            expected_config_hashes=("formal_config",),
            expected_member_count=5,
        )
        with self.assertRaises(ValueError):
            validate_expected_oof_roster(
                rows[:-2] + rows[-1:],
                roster,
                expected_config_hashes=("formal_config",),
                expected_member_count=5,
            )
        trace_drift = (
            replace(member_rows[0], manifest_hash="different_manifest"),
            *member_rows[1:],
            rows[-1],
        )
        with self.assertRaisesRegex(ValueError, "truth or trace drift"):
            validate_expected_oof_roster(
                trace_drift,
                roster,
                expected_config_hashes=("formal_config",),
                expected_member_count=5,
            )

    def test_rejected_subject_still_satisfies_roster(self) -> None:
        """A drop is an explicit row rather than absence / drop 是显式行而不是缺行。"""

        row = _formal_participant_row("P1", 0, retained=False)
        validate_expected_oof_roster(
            (row,),
            {(0, 0, 42): ("P1",)},
            expected_config_hashes=("formal_config",),
        )

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional pyarrow unavailable")
    def test_all_dropped_ensemble_is_complete_and_parquet_roundtrips(self) -> None:
        """All five members plus the average preserve one coherent dropped state."""

        rows = tuple(
            _formal_participant_row(
                "P1", 0, member_index=index, retained=False
            )
            for index in range(5)
        ) + (
            _formal_participant_row(
                "P1",
                0,
                prediction_kind="ensemble_average",
                retained=False,
            ),
        )
        roster = {(0, 0, 42): ("P1",)}
        validate_expected_oof_roster(
            rows,
            roster,
            expected_config_hashes=("formal_config",),
            expected_member_count=5,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = write_oof_parquet(rows, Path(temporary) / "ensemble.parquet")
            self.assertEqual(read_oof_parquet(path), rows)


class TrainingIdentityProtocolTests(unittest.TestCase):
    """Exact refit roster and data-to-identity binding.

    精确 refit roster 与数据-身份绑定。
    """

    def setUp(self) -> None:
        rng = np.random.default_rng(101)
        self.identities = tuple(_identity(f"P{index}", index % 3) for index in range(6))
        self.values = rng.normal(size=(6, 2, 8)).astype(np.float32)
        self.dataset = RawWindowDataset(self.values, self.identities)
        self.split = FrozenOuterSplit(
            repeat=0,
            fold=0,
            seed=42,
            train_participant_ids=tuple(f"P{index}" for index in range(6)),
            oof_participant_ids=("P6", "P7"),
            registry_hash="registry",
            fold_hash="fold",
        )

    def test_partial_final_refit_is_rejected(self) -> None:
        """Final fitting cannot silently omit train subjects / 最终拟合不得静默漏 subject。"""

        partial = RawWindowDataset(self.values[:5], self.identities[:5])
        trainer = UnifiedTrainer(TrainingConfig(fixed_epochs=1, batch_size=3, execution_mode="smoke", epoch_profile="smoke"))
        with self.assertRaises(ValueError):
            trainer.fit(_TinyTorchClassifier, partial, self.split)

    def test_bound_split_rejects_disguised_values(self) -> None:
        """Changing values under unchanged IDs is detected / ID 不变而数值变化会被识别。"""

        bound = self.split.bind_training_dataset(self.dataset)
        changed = self.values.copy()
        changed[0, 0, 0] += 100.0
        disguised = RawWindowDataset(changed, self.identities)
        with self.assertRaises(ValueError):
            bound.assert_training_dataset(disguised)

    def test_identity_mix_is_rejected(self) -> None:
        """One file cannot belong to two participants / 同一 file 不得属于两个 participant。"""

        mixed = (
            _identity("P0", 0),
            replace(_identity("P1", 1), file_id="P0_B_0"),
        )
        dataset = RawWindowDataset(self.values[:2], mixed)
        with self.assertRaises(ValueError):
            validate_dataset_identity_coherence(dataset)

    def test_heldout_roster_mutation_cannot_change_fitted_state(self) -> None:
        """Trainer never receives held-out data or labels / trainer 不接收 held-out 数据或标签。"""

        config = TrainingConfig(fixed_epochs=1, batch_size=3, seed=77, execution_mode="smoke", epoch_profile="smoke")
        first = UnifiedTrainer(config).fit(_TinyTorchClassifier, self.dataset, self.split)
        changed_oof = replace(self.split, oof_participant_ids=("P8", "P9"))
        second = UnifiedTrainer(config).fit(
            _TinyTorchClassifier, self.dataset, changed_oof
        )
        self.assertEqual(first.provenance.state_hash, second.provenance.state_hash)
        self.assertNotEqual(
            first.provenance.outer_membership_hash,
            second.provenance.outer_membership_hash,
        )


class FormalBundleProtocolTests(unittest.TestCase):
    """Transactional, schema-bound and raw-adapter bundle tests.

    事务、schema 绑定及 raw-adapter bundle 测试。
    """

    @staticmethod
    def _save(directory: Path, *, adapter: object | None = None) -> Path:
        """Save one tiny formal bundle / 保存一个微型正式 bundle。"""

        return save_bundle(
            _TinyEstimator(),
            directory,
            model_config={"model_id": "logistic_regression", "seed": 42},
            input_spec=ModelInputSpec(
                "feature_vector",
                n_classes=3,
                feature_names=("feature_a", "feature_b"),
            ),
            metadata=_formal_metadata(),
            golden_inputs={"x": np.asarray(((1.0, 2.0),), dtype=np.float32)},
            pipeline_adapter=adapter,
        )

    def test_raw_adapter_and_stale_schema_guard(self) -> None:
        """Raw prediction works and stale expectations fail closed.

        raw 推理可运行，陈旧 schema 期望关闭失败。
        """

        with tempfile.TemporaryDirectory() as temporary:
            target = self._save(Path(temporary) / "bundle", adapter=_RawAdapter())
            probability = predict_bundle_raw(target, np.asarray((3.0, 4.0)))
            np.testing.assert_allclose(probability, ((0.2, 0.3, 0.5),))
            with self.assertRaises(ValueError):
                load_bundle(target, expected_metadata={"feature_hash": "stale"})

    def test_failed_serialization_leaves_no_partial_target(self) -> None:
        """Only atomic rename can make the target visible / 仅原子 rename 可暴露目标。"""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "bundle"
            with self.assertRaises(RuntimeError):
                self._save(target, adapter=_BrokenAdapter())
            self.assertFalse(target.exists())
            self.assertEqual(tuple(root.glob(".bundle.staging-*")), ())

    def test_missing_metadata_fails_before_target_creation(self) -> None:
        """Incomplete deployment metadata cannot create files / 不完整 metadata 不得创建文件。"""

        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "bundle"
            metadata = _formal_metadata()
            metadata.pop("validity_policy")
            with self.assertRaises(ValueError):
                save_bundle(
                    _TinyEstimator(),
                    target,
                    model_config={"model_id": "logistic_regression"},
                    input_spec=ModelInputSpec(
                        "feature_vector",
                        n_classes=3,
                        feature_names=("feature_a", "feature_b"),
                    ),
                    metadata=metadata,
                    golden_inputs={"x": np.ones((1, 2), dtype=np.float32)},
                )
            self.assertFalse(target.exists())

    def test_nonfinite_ndarray_metadata_is_rejected(self) -> None:
        """Nested arrays cannot bypass strict JSON / 嵌套数组不能绕过严格 JSON。"""

        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "bundle"
            metadata = _formal_metadata()
            metadata["environment"]["numeric_probe"] = np.asarray((np.nan,))
            with self.assertRaisesRegex(TypeError, "NaN or infinity"):
                save_bundle(
                    _TinyEstimator(),
                    target,
                    model_config={"model_id": "logistic_regression"},
                    input_spec=ModelInputSpec(
                        "feature_vector",
                        n_classes=3,
                        feature_names=("feature_a", "feature_b"),
                    ),
                    metadata=metadata,
                    golden_inputs={"x": np.ones((1, 2), dtype=np.float32)},
                )
            self.assertFalse(target.exists())

    def test_ten_thousand_load_predict_roundtrips(self) -> None:
        """Exercise 10,000 load/predict rounds without repeated save.

        不重复 save，执行 10,000 轮 load/predict。
        """

        with tempfile.TemporaryDirectory() as temporary:
            target = self._save(Path(temporary) / "bundle")
            assert_repeated_bundle_parity(target, iterations=10_000)


class CanonicalFacadeParityTests(unittest.TestCase):
    """Canonical singular packages must delegate to plural authorities.

    canonical 单数包必须委托 plural 唯一实现。
    """

    def test_aggregate_and_metric_facades_preserve_formal_semantics(self) -> None:
        """Facade calls retain drop coverage and metric formulas.

        facade 调用必须保留 drop coverage 与指标公式。
        """

        aggregation = aggregate_hierarchy_strict(
            (_window_row("P1"), _window_row("P2", retained=False))
        )
        self.assertEqual(len(aggregation.dropped_rows), 1)
        self.assertEqual(aggregation.coverage[-1].coverage_rate, 0.5)
        metrics = evaluate_participant_probabilities(
            np.asarray((0, 1, 2)),
            np.eye(3, dtype=np.float64),
        )
        self.assertEqual(metrics.multiclass_brier, 0.0)
        self.assertEqual(metrics.coverage_rate, 1.0)

    def test_oof_facade_delegates_exact_roster(self) -> None:
        """Canonical OOF facade uses complete trace rules / canonical OOF 使用完整 trace。"""

        rows = (_formal_participant_row("P1", 0), _formal_participant_row("P2", 1))
        audit = validate_oof_contract(
            rows,
            {
                (0, 0, 42, "formal_config"): {"P1"},
                (0, 1, 42, "formal_config"): {"P2"},
            },
        )
        self.assertTrue(audit.exact_once)
        self.assertEqual(audit.subject_rows, 2)

    def test_bundle_facade_has_one_schema_and_explicit_strict_gate(self) -> None:
        """Canonical bundle schema is the exact training schema.

        canonical bundle schema 与 training schema 必须精确相同。
        """

        self.assertEqual(set(REQUIRED_V2_METADATA), set(REQUIRED_METADATA))
        arguments = {
            "model_config": {"model_id": "logistic_regression", "seed": 42},
            "input_spec": ModelInputSpec(
                "feature_vector",
                n_classes=3,
                feature_names=("feature_a", "feature_b"),
            ),
            "metadata": _formal_metadata(),
            "golden_inputs": {"x": np.asarray(((1.0, 2.0),), dtype=np.float32)},
            "pipeline_adapter": _RawAdapter(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                save_bundle_strict(
                    _TinyEstimator(),
                    root / "rejected",
                    strict_metadata=False,
                    **arguments,
                )
            target = save_bundle_strict(
                _TinyEstimator(),
                root / "accepted",
                strict_metadata=True,
                **arguments,
            )
            np.testing.assert_allclose(
                predict_bundle_raw(target, np.asarray((1.0, 2.0))),
                ((0.2, 0.3, 0.5),),
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
