"""OOF hierarchy, ablation and deployable bundle tests.

OOF 层级、消融与可部署 bundle 测试。
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ppg_frailty.models import ModelInputSpec, create_model
from ppg_frailty.training import (
    AblationCase,
    OofPredictionRow,
    OofWriter,
    aggregate_hierarchy,
    assert_golden_parity,
    load_bundle,
    paired_subject_deltas,
    predict_bundle,
    run_ablation_matrix,
    save_bundle,
)


def _row(
    participant: str,
    file_id: str,
    role: str,
    window: str,
    probability: tuple[float, float, float],
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
        seed=42,
        config_hash="config",
        manifest_hash="manifest",
        fold_hash="fold",
        preprocessing_hash="preprocessing",
        feature_hash="feature",
        model_hash="model",
        representation_mode="raw",
        signal_route="direct",
        quality_score=1.0,
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

        result = aggregate_hierarchy(self.rows)
        self.assertEqual(len(result.file_rows), 2)
        self.assertEqual(len(result.role_rows), 2)
        self.assertEqual(len(result.participant_rows), 1)
        np.testing.assert_allclose(result.file_rows[0].probabilities, (0.7, 0.15, 0.15))
        np.testing.assert_allclose(result.participant_rows[0].probabilities, (0.5, 0.25, 0.25))

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional pyarrow is unavailable")
    def test_oof_writer_emits_parquet(self) -> None:
        """English: Writer uses Parquet only and never falls back to CSV.

        中文：写入器只使用 Parquet，绝不回退为 CSV。
        """

        with tempfile.TemporaryDirectory() as temporary:
            target = OofWriter().write(self.rows, Path(temporary) / "oof.parquet")
            self.assertTrue(target.is_file())
            self.assertGreater(target.stat().st_size, 0)


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

        model_config = {"model_id": "compact_cnn", "seed": 7}
        input_spec = ModelInputSpec("raw", n_channels=2, n_classes=3, channel_schema=("red", "ir"))
        model = create_model(model_config, input_spec).eval()
        inputs = {"x": np.random.default_rng(3).normal(size=(2, 2, 64)).astype(np.float32)}
        metadata = {
            "model_identity": {
                "name": "CompactCNN1D",
                "machine_id": "compact_cnn",
                "version": "test",
            },
            "representation_mode": "raw",
            "signal_route": "direct",
            "class_order": [0, 1, 2],
            "channel_schema": ["red", "ir"],
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
            "code_version": "test",
            "environment": {"python": "test", "platform": "cpu"},
            "dependency_status": "decision_pending",
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
