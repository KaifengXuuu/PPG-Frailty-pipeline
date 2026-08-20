"""OOF hierarchy, ablation and deployable bundle tests.

OOF 层级、消融与可部署 bundle 测试。
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ppg_frailty.models import (
    ModelInputSpec,
    create_model,
    materialize_architecture_parameters,
)
from ppg_frailty.models.factory import FRAILTY_RAW_CHANNEL_SCHEMA
from ppg_frailty.training import (
    AblationCase,
    OofPredictionRow,
    OofWriter,
    aggregate_hierarchy,
    assert_golden_parity,
    current_runtime_environment,
    load_bundle,
    paired_subject_deltas,
    predict_bundle,
    read_oof_parquet,
    read_oof_parquet_metadata,
    run_ablation_matrix,
    save_bundle,
)


def _row(
    participant: str,
    file_id: str,
    role: str,
    window: str,
    probability: tuple[float, float, float],
    *,
    quality_score: float = 1.0,
) -> OofPredictionRow:
    """English: Construct a fully provenance-bound OOF row.

    中文：构造完整绑定 provenance 的 OOF 行。
    """

    return OofPredictionRow(
        participant_id=participant,
        file_id=file_id,
        role=role,
        label=0,
        probabilities=probability,
        repeat=0,
        fold=1,
        split_seed=42,
        training_seed=42,
        config_hash="config",
        manifest_hash="manifest",
        fold_hash="fold",
        preprocessing_hash="preprocessing",
        feature_hash="feature",
        model_hash="model",
        representation_mode="raw",
        signal_route="direct",
        quality_score=quality_score,
        retained=True,
        level="window",
        window_id=window,
    )


class OofAggregationTests(unittest.TestCase):
    """English: Lock equal-weight hierarchy and Parquet behavior.

    中文：锁定等权层级与 Parquet 行为。
    """

    def setUp(self) -> None:
        self.rows = (
            _row("P1", "F1", "B", "W1", (0.8, 0.1, 0.1)),
            _row("P1", "F1", "B", "W2", (0.6, 0.2, 0.2)),
            _row("P1", "F2", "R", "W3", (0.2, 0.4, 0.4)),
            _row("P1", "F2", "R", "W4", (0.4, 0.3, 0.3)),
        )

    def test_window_file_role_participant_equal_weight_hierarchy(self) -> None:
        """English: Each hierarchy level averages its direct children.

        中文：每个层级均等权平均其直接子节点。
        """

        result = aggregate_hierarchy(
            self.rows, balance_line="line_b_equal_role_families"
        )
        self.assertEqual(len(result.file_rows), 2)
        self.assertEqual(len(result.role_rows), 2)
        self.assertEqual(len(result.participant_rows), 1)
        np.testing.assert_allclose(result.file_rows[0].probabilities, (0.7, 0.15, 0.15))
        np.testing.assert_allclose(result.participant_rows[0].probabilities, (0.5, 0.25, 0.25))

    def test_quality_weighting_changes_each_selected_hierarchy(self) -> None:
        """SQI weighting is an executable modifier, not a hash-only switch."""

        rows = (
            _row(
                "P1", "F1", "B", "W1", (0.9, 0.05, 0.05),
                quality_score=0.9,
            ),
            _row(
                "P1", "F2", "R", "W2", (0.1, 0.45, 0.45),
                quality_score=0.1,
            ),
        )
        for line in ("line_a_equal_files", "line_b_equal_role_families"):
            with self.subTest(balance_line=line):
                ordinary = aggregate_hierarchy(rows, balance_line=line)
                weighted = aggregate_hierarchy(
                    rows,
                    balance_line=line,
                    quality_weighted=True,
                )
                np.testing.assert_allclose(
                    ordinary.participant_rows[0].probabilities,
                    (0.5, 0.25, 0.25),
                )
                np.testing.assert_allclose(
                    weighted.participant_rows[0].probabilities,
                    (0.82, 0.09, 0.09),
                )

    def test_explicit_weight_sources_apply_at_their_real_prediction_level(self) -> None:
        """Route Q_rate is file-level; migrated legacy SQI is window-level."""

        route_rows = (
            _row(
                "P1", "F1", "B", "W1", (1.0, 0.0, 0.0),
                quality_score=0.4,
            ),
            _row(
                "P1", "F1", "B", "W2", (0.0, 1.0, 0.0),
                quality_score=0.4,
            ),
        )
        route = aggregate_hierarchy(
            route_rows,
            balance_line="line_a_equal_files",
            quality_weighted=True,
            quality_weight_source="route_file_q_rate",
        )
        np.testing.assert_allclose(
            route.file_rows[0].probabilities,
            (0.5, 0.5, 0.0),
        )
        direct_files = (
            replace(
                _row(
                    "P1", "F1", "B", "W1", (0.9, 0.05, 0.05),
                    quality_score=0.9,
                ),
                level="file",
                window_id=None,
                representation_mode="fusion",
            ),
            replace(
                _row(
                    "P1", "F2", "R", "W2", (0.1, 0.45, 0.45),
                    quality_score=0.1,
                ),
                level="file",
                window_id=None,
                representation_mode="fusion",
            ),
        )
        fusion_route = aggregate_hierarchy(
            direct_files,
            balance_line="line_a_equal_files",
            quality_weighted=True,
            quality_weight_source="route_file_q_rate",
        )
        np.testing.assert_allclose(
            fusion_route.participant_rows[0].probabilities,
            (0.82, 0.09, 0.09),
        )

        legacy_rows = (
            _row(
                "P1", "F1", "B", "W1", (1.0, 0.0, 0.0),
                quality_score=0.9,
            ),
            _row(
                "P1", "F1", "B", "W2", (0.0, 1.0, 0.0),
                quality_score=0.1,
            ),
        )
        legacy = aggregate_hierarchy(
            legacy_rows,
            balance_line="line_a_equal_files",
            quality_weighted=True,
            quality_weight_source="legacy_window_sqi",
        )
        np.testing.assert_allclose(
            legacy.file_rows[0].probabilities,
            (0.9, 0.1, 0.0),
        )

        inconsistent_route = (
            route_rows[0],
            _row(
                "P1", "F1", "B", "W2", (0.0, 1.0, 0.0),
                quality_score=0.8,
            ),
        )
        with self.assertRaisesRegex(ValueError, "constant across windows"):
            aggregate_hierarchy(
                inconsistent_route,
                balance_line="line_a_equal_files",
                quality_weighted=True,
                quality_weight_source="route_file_q_rate",
            )

    def test_legacy_all_zero_weights_match_historical_mean_fallback(self) -> None:
        rows = (
            _row(
                "P1", "F1", "B", "W1", (0.8, 0.1, 0.1),
                quality_score=0.0,
            ),
            _row(
                "P1", "F1", "B", "W2", (0.2, 0.4, 0.4),
                quality_score=0.0,
            ),
        )
        result = aggregate_hierarchy(
            rows,
            balance_line="line_a_equal_files",
            quality_weighted=True,
            quality_weight_source="legacy_window_sqi",
        )
        np.testing.assert_allclose(
            result.file_rows[0].probabilities,
            (0.5, 0.25, 0.25),
        )
        np.testing.assert_allclose(
            result.participant_rows[0].probabilities,
            (0.5, 0.25, 0.25),
        )

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional pyarrow is unavailable")
    def test_oof_writer_emits_parquet(self) -> None:
        """English: Writer uses Parquet only and never falls back to CSV.

        中文：写入器只使用 Parquet，绝不回退为 CSV。
        """

        with tempfile.TemporaryDirectory() as temporary:
            target = OofWriter().write(self.rows, Path(temporary) / "oof.parquet")
            self.assertTrue(target.is_file())
            self.assertGreater(target.stat().st_size, 0)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional pyarrow is unavailable")
    def test_empty_oof_writer_preserves_exact_typed_schema_and_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = OofWriter().write_empty(
                Path(temporary) / "empty.parquet",
                "participant_level_deliberately_absent",
            )
            self.assertEqual(read_oof_parquet(target), ())
            metadata = read_oof_parquet_metadata(target)
            self.assertEqual(metadata["schema_version"], "ppg_frailty_oof_v2")
            self.assertEqual(metadata["artifact_state"], "empty")
            self.assertEqual(
                metadata["empty_reason"],
                "participant_level_deliberately_absent",
            )


class AblationAndBundleTests(unittest.TestCase):
    """English: Exercise comparison API and golden bundle parity.

    中文：验证对照 API 与 bundle golden parity。
    """

    def test_one_factor_ablation_and_paired_delta(self) -> None:
        """English: Runner receives baseline and one changed field per case.

        中文：runner 接收基线，并且每个案例只改变一个字段。
        """

        seen: dict[str, int] = {}

        def runner(config: dict, case_id: str) -> int:
            seen[case_id] = config["model"]["width"]
            return config["model"]["width"]

        result = run_ablation_matrix(
            {"model": {"width": 8, "depth": 2}, "seed": 42},
            (AblationCase("wide", "model.width", 16, "capacity control"),),
            runner,
        )
        self.assertEqual(result, {"baseline": 8, "wide": 16})
        comparison = paired_subject_deltas({"P1": 0.5, "P2": 0.6}, {"P1": 0.7, "P2": 0.5})
        self.assertAlmostEqual(comparison.mean_delta, 0.05)

    def test_torch_bundle_roundtrip_and_integrity_guard(self) -> None:
        """English: Reloaded predictions match and tampering is rejected.

        中文：重载预测一致，且任何文件篡改均被拒绝。
        """

        model_config = {
            "model_id": "compact_cnn",
            "seed": 7,
            "dropout": 0.2,
            "kernel_sizes": [9, 9, 7],
            "dilations": [1, 1, 1],
            "pool_sizes": [4, 4],
        }
        input_spec = ModelInputSpec(
            "raw",
            n_channels=8,
            n_classes=3,
            channel_schema=FRAILTY_RAW_CHANNEL_SCHEMA,
        )
        model_config["architecture_parameters"] = materialize_architecture_parameters(
            model_config, input_spec
        )
        model = create_model(model_config, input_spec).eval()
        inputs = {
            "x": np.random.default_rng(3)
            .normal(size=(2, 8, 64))
            .astype(np.float32)
        }
        metadata = {
            "model_identity": {
                "name": "CompactCNN1D",
                "machine_id": "compact_cnn",
                "version": "test",
            },
            "representation_mode": "raw",
            "signal_route": "direct",
            "class_order": [0, 1, 2],
            "channel_schema": list(FRAILTY_RAW_CHANNEL_SCHEMA),
            "preprocessing": {"name": "test_preprocessing", "version": "test"},
            "preprocessing_hash": "preprocessing",
            "resampling": {"method": "not_applied", "status": "not_applicable"},
            "window_plan": {"name": "test_window", "length_samples": 64},
            "feature_registry": {"status": "not_applicable", "version": "v1"},
            "feature_hash": "not-applicable",
            "feature_vector_schema": {"status": "not_applicable"},
            "ordered_matrix_schema": {"status": "not_applicable"},
            "mask_semantics": {"sample_mask": "true_is_valid"},
            "validity_policy": {"unavailable": "nan_and_false"},
            "fitted_objects": ["model"],
            "representation_state": {"kind": "raw_model_weights"},
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
            "code_version": "test",
            "environment": current_runtime_environment(),
            "dependency_status": "decision_pending",
            "serialization_trust": {
                "trusted_local_only": True,
                "authenticated_signature": False,
            },
            "golden_case": {"id": "unit_test", "n_samples": 2},
        }
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "bundle"
            save_bundle(
                model,
                target,
                model_config=model_config,
                input_spec=input_spec,
                metadata=metadata,
                golden_inputs=inputs,
            )
            loaded = load_bundle(target)
            assert_golden_parity(loaded)
            probability = predict_bundle(loaded, inputs)
            self.assertEqual(probability.shape, (2, 3))
            with (target / "state.pt").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaises(ValueError):
                load_bundle(target)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
