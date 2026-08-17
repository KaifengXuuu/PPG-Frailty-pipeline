"""Canonical facade、strict config 与 coverage 集成测试。

English: These tests prove that exact spec paths import, formal configuration uses
one reducer/planner authority, dropped rows remain in coverage, and bundle inference
delegates to the serialised raw-record adapter.
"""

from __future__ import annotations

import importlib
import inspect
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ppg_frailty.bundle.infer import infer_raw_record
from ppg_frailty.bundle.save import (
    REQUIRED_V1_METADATA,
    save_bundle_strict,
    validate_bundle_metadata,
)
from ppg_frailty.config import load_config
from ppg_frailty.evaluate.aggregate import aggregate_hierarchy_strict
from ppg_frailty.evaluate.metrics import evaluate_participant_probabilities
from ppg_frailty.module_registry import (
    list_modules,
    resolve_artifact_config,
    resolve_artifact_module_id,
    resolve_window_config,
)
from ppg_frailty.pipeline import validate_installation
from ppg_frailty.signal.resample import resample_dl_view
from ppg_frailty.training.bundle import LoadedBundle, REQUIRED_METADATA
from ppg_frailty.training.oof import OofPredictionRow


ROOT = Path(__file__).resolve().parents[2]


class CanonicalFacadeTests(unittest.TestCase):
    """验证规范路径与唯一 authority / Validate exact paths and sole authorities."""

    def test_every_exact_spec_facade_imports(self) -> None:
        """§4 的精确文件名必须可导入 / Every exact §4 filename must import."""

        modules = (
            "ppg_frailty.signal.ppg_preprocess",
            "ppg_frailty.signal.imu_preprocess",
            "ppg_frailty.signal.window_plan",
            "ppg_frailty.signal.resample",
            "ppg_frailty.features.prv",
            "ppg_frailty.features.morphology",
            "ppg_frailty.features.dual_wavelength",
            "ppg_frailty.features.spectral",
            "ppg_frailty.features.file_vector",
            "ppg_frailty.features.ordered_matrix",
            "ppg_frailty.models.inception_time_port",
            "ppg_frailty.models.inception_ensemble",
            "ppg_frailty.models.shapeformer_port",
            "ppg_frailty.models.feature_models",
            "ppg_frailty.models.rocket_ridge",
            "ppg_frailty.models.file_fusion",
        )
        for name in modules:
            with self.subTest(module=name):
                self.assertIsNotNone(importlib.import_module(name))

    def test_registry_points_only_to_importable_canonical_paths(self) -> None:
        """注册表不能暴露 plural artifact 实现 / Registry exposes canonical paths."""

        for descriptor in list_modules():
            implementation = descriptor["implementation"]
            self.assertNotIn("ppg_frailty.artifacts.", implementation)
            module_name, _ = implementation.rsplit(".", 1)
            with self.subTest(implementation=implementation):
                self.assertIsNotNone(importlib.import_module(module_name))

    def test_all_four_materialized_configs_preflight(self) -> None:
        """四份正式配置均只读通过冻结 preflight / All four configs pass."""

        names = (
            "feature_matrix_v1.yaml",
            "motion_benchmark_v1.yaml",
            "reference_all_roles_v1.yaml",
            "reference_static_v1.yaml",
        )
        for name in names:
            with self.subTest(config=name):
                result = validate_installation(config_path=ROOT / "configs" / name)
                self.assertEqual(result["status"], "passed")

    def test_motion_config_is_exact_and_unknown_parameter_fails(self) -> None:
        """spectral_mask 原名可用，未知参数关闭失败 / Exact name passes; unknown fails."""

        config = load_config(ROOT / "configs" / "motion_benchmark_v1.yaml").to_dict()
        resolved = resolve_artifact_config(config["artifact"])
        self.assertEqual(resolved["declared_reducer"], "spectral_mask")
        self.assertEqual(resolved["runtime_reducer"], "spectral_mask")
        invalid = dict(config["artifact"])
        invalid["parameters"] = dict(invalid["parameters"], invented_parameter=1)
        with self.assertRaisesRegex(ValueError, "unknown reducer parameters"):
            resolve_artifact_config(invalid)

    def test_artifact_comparison_ids_are_canonical_or_explicit_legacy(self) -> None:
        """comparison 不静默翻译短名 / Comparison never silently translates IDs."""

        canonical = resolve_artifact_module_id("spectral_mask")
        legacy = resolve_artifact_module_id("spectral")
        self.assertEqual(canonical["canonical_module_id"], "spectral_mask")
        self.assertFalse(canonical["legacy_alias_used"])
        self.assertEqual(legacy["canonical_module_id"], "spectral_mask")
        self.assertTrue(legacy["legacy_alias_used"])

    def test_window_adapter_reuses_the_data_windowplan_class(self) -> None:
        """canonical window facade 与 data authority 是同一类 / One planner class."""

        from ppg_frailty.data.windows import WindowPlan as DataWindowPlan
        from ppg_frailty.signal.window_plan import WindowPlan as FacadeWindowPlan

        config = load_config(ROOT / "configs" / "reference_static_v1.yaml").to_dict()
        profiles = resolve_window_config(config["windows"])
        plan = FacadeWindowPlan(source_record_id="fixture", **profiles["raw_dl"])
        self.assertIs(FacadeWindowPlan, DataWindowPlan)
        self.assertEqual(plan.window_seconds, 5.0)

    def test_dl_resampling_is_separate_and_anti_aliased(self) -> None:
        """400→100 Hz 生成独立 view 且不改 input / Create a separate DL view."""

        source = np.vstack((np.arange(400.0), np.arange(400.0) ** 2))
        snapshot = source.copy()
        result = resample_dl_view(source, target_fs_hz=100.0)
        self.assertEqual(result.values.shape, (2, 100))
        self.assertEqual((result.up, result.down), (1, 4))
        np.testing.assert_array_equal(source, snapshot)


class AuthorityContractTests(unittest.TestCase):
    """验证 bundle/evaluate 不复制弱逻辑 / Verify bundle/evaluate delegation."""

    def test_bundle_facade_uses_training_metadata_authority(self) -> None:
        """required keys 必须逐项相同且 strict flag 必填 / One exact schema."""

        self.assertEqual(set(REQUIRED_V1_METADATA), set(REQUIRED_METADATA))
        self.assertIs(
            inspect.signature(save_bundle_strict).parameters["strict_metadata"].default,
            inspect.Parameter.empty,
        )
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            validate_bundle_metadata({})

    def test_raw_bundle_facade_delegates_to_serialized_adapter(self) -> None:
        """raw inference 不再从 metadata 重建第二条 pipeline / Delegate adapter."""

        loaded = LoadedBundle(
            model=object(),
            transforms=None,
            manifest={"metadata": {"representation_mode": "raw"}},
            directory=ROOT,
            pipeline_adapter=object(),
        )
        window_probability = np.asarray(((0.8, 0.1, 0.1), (0.2, 0.3, 0.5)))
        with patch(
            "ppg_frailty.bundle.infer.predict_bundle_raw",
            return_value=window_probability,
        ) as delegated:
            output = infer_raw_record(loaded, {"record_id": "fixture"})
        delegated.assert_called_once()
        np.testing.assert_allclose(output["window_probabilities"], window_probability)
        np.testing.assert_allclose(output["file_probability"], (0.5, 0.2, 0.3))

    def test_dropped_probability_nan_is_filtered_before_metrics(self) -> None:
        """drop 行可含 NaN 且仍进入 coverage denominator / Dropped NaN is legal."""

        metrics = evaluate_participant_probabilities(
            np.asarray((0, 1, 2, 0)),
            np.asarray(
                (
                    (0.8, 0.1, 0.1),
                    (0.1, 0.8, 0.1),
                    (0.1, 0.1, 0.8),
                    (np.nan, np.nan, np.nan),
                )
            ),
            retained_mask=np.asarray((True, True, True, False)),
        )
        self.assertEqual(metrics.n_retained, 3)
        self.assertEqual(metrics.n_dropped, 1)
        self.assertEqual(metrics.coverage_rate, 0.75)

    def test_all_dropped_rows_remain_auditable(self) -> None:
        """all-drop 不应在聚合前消失 / All-drop remains in source coverage."""

        dropped = OofPredictionRow(
            participant_id="P01",
            file_id="P01:B",
            role="B",
            label=0,
            probabilities=(),
            repeat=0,
            fold=0,
            seed=42,
            config_hash="config",
            manifest_hash="manifest",
            fold_hash="fold",
            preprocessing_hash="preprocess",
            feature_hash="feature",
            model_hash="model",
            representation_mode="raw",
            signal_route="dropped",
            quality_score=0.0,
            retained=False,
            level="file",
            rejection_reason="quality_reject",
        )
        result = aggregate_hierarchy_strict((dropped,))
        self.assertEqual(len(result.source_rows), 1)
        self.assertEqual(len(result.dropped_rows), 1)
        self.assertEqual(len(result.participant_rows), 0)
        self.assertTrue(result.coverage)


if __name__ == "__main__":
    unittest.main()
