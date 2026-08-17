"""V2 配置、决策与依赖合同 / V2 config, decision, and dependency contracts."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ppg_frailty.config import (
    PipelineConfig,
    dependency_availability_report,
    load_config,
    load_formal_ablation_profiles,
    load_formal_experiment_catalog,
    load_v2_decision_profile,
    materialize_formal_ablation_config,
    required_runtime_modules,
    validate_config_payload,
)
from ppg_frailty.module_registry import list_modules, validate_model_config
from ppg_frailty.experiment import _model_input_sampling_rate_hz


ROOT = Path(__file__).resolve().parents[2]


class V2ConfigurationTests(unittest.TestCase):
    """验证默认关闭门与显式comparison / Check defaults and explicit comparisons."""

    def test_default_formal_config_is_sqi_off_br_role_aware_fixed10(self) -> None:
        """默认不得偷偷启用未监督路线 / Default must not activate an unready route."""

        config = load_config(ROOT / "configs/reference_static_feature_vector_v2.yaml")
        self.assertEqual(config.schema_version, "ppg_frailty.pipeline_config.v2")
        self.assertEqual(config.payload["roles"], ["B", "R1", "R2", "R3", "R4"])
        self.assertEqual(config.payload["quality"]["mode"], "off")
        self.assertIs(config.payload["quality"]["supervised_route_ready"], False)
        self.assertEqual(
            config.payload["training"]["training_balance"],
            "equal_role_families",
        )
        self.assertEqual(config.payload["training"]["epoch_profile"], "default_10")
        self.assertEqual(config.payload["training"]["fixed_epochs"], 10)
        self.assertEqual(
            config.payload["aggregation"]["balance_line"],
            "line_b_equal_role_families",
        )

    def test_four_representation_formal_entrypoints_preflight(self) -> None:
        """四表征配置均为role-aware/SQI-off/fixed10 / Four formal entrypoints."""

        expected = {
            "reference_static_role_aware_v2.yaml": ("raw", "CompactCNN1D"),
            "reference_static_feature_vector_v2.yaml": (
                "feature_vector",
                "LogisticRegressionL2",
            ),
            "reference_static_feature_matrix_v2.yaml": (
                "feature_matrix",
                "InceptionTimeMatrix",
            ),
            "reference_static_fusion_v2.yaml": ("fusion", "FileBagFusionCompact"),
        }
        for filename, (mode, model_id) in expected.items():
            config = load_config(ROOT / "configs" / filename)
            self.assertEqual(config.representation_mode, mode)
            self.assertEqual(config.payload["model"]["model_id"], model_id)
            self.assertEqual(config.payload["quality"]["mode"], "off")
            self.assertEqual(config.payload["training"]["fixed_epochs"], 10)
            self.assertEqual(
                config.payload["aggregation"]["balance_line"],
                "line_b_equal_role_families",
            )
            validate_model_config(config.payload["model"], mode)

    def test_formal_catalog_freezes_matched_member0_and_ensemble_seed_axes(self) -> None:
        catalog = load_formal_experiment_catalog(
            ROOT / "configs/formal_experiment_catalog_v2.yaml"
        )
        entries = {entry["entry_id"]: entry for entry in catalog["entries"]}
        models = {entry_id: entry["model"] for entry_id, entry in entries.items()}
        for entry_id in ("inception_full", "inception_matrix"):
            self.assertEqual(
                models[entry_id]["seed_policy"],
                "outer_cv_repeat_seed_equals_split_seed",
            )
        comparator_pairs = {
            "inception_full_member0_comparator": "inception_full",
            "inception_matrix_member0_comparator": "inception_matrix",
        }
        for comparator_id, ordinary_id in comparator_pairs.items():
            self.assertEqual(
                entries[comparator_id]["catalog_role"],
                "matched_comparator",
            )
            self.assertEqual(
                models[comparator_id]["seed_policy"],
                "cv_fixed_member0_seed_50042_comparator",
            )
            comparator_model = dict(models[comparator_id])
            comparator_model["seed_policy"] = (
                "outer_cv_repeat_seed_equals_split_seed"
            )
            self.assertEqual(comparator_model, models[ordinary_id])
        for entry_id in (
            "inception_full_five_member_ensemble",
            "inception_matrix_five_member_ensemble",
        ):
            self.assertEqual(
                models[entry_id]["seed_policy"],
                "cv_fixed_five_member_seed_roster",
            )
            self.assertEqual(
                models[entry_id]["member_seeds"],
                [50042, 60042, 70042, 80042, 90042],
            )
            self.assertEqual(
                models[entry_id]["architecture_parameters"]["member_seeds"],
                [50042, 60042, 70042, 80042, 90042],
            )
        self.assertEqual(
            models["compact_cnn"]["seed_policy"],
            "outer_cv_repeat_seed_equals_split_seed",
        )

    def test_shapeformer_reference_is_variable_length_and_formally_materialized(self) -> None:
        """Reference OSD has no fixed-length or stride controls."""

        decision = load_v2_decision_profile(
            ROOT / "configs/v2_decision_profile.yaml"
        )["confirmed_defaults"]["shapeformer"]
        self.assertEqual(
            decision["formal_config_status"],
            "materialized_in_formal_experiment_catalog_v2",
        )
        self.assertIsNone(decision["fixed_shapelet_length_samples"])
        self.assertIsNone(decision["candidate_stride"])
        self.assertEqual(decision["num_pip_ratio"], 0.20)
        self.assertEqual(decision["shapelets_per_class"], 3)
        self.assertEqual(decision["max_discovery_windows"], 180)
        statuses = {
            row["module_id"]: row["scientific_status"]
            for row in list_modules("model")
            if row["module_id"].startswith("ShapeFormer")
        }
        self.assertEqual(
            statuses,
            {
                "ShapeFormerChannelSpecificOSD":
                    "implemented_not_benchmarked_high_compute",
                "ShapeFormerEffectSizeFixedV1":
                    "fixed_length_effect_size_ablation",
            },
        )

        catalog = load_formal_experiment_catalog(
            ROOT / "configs/formal_experiment_catalog_v2.yaml"
        )
        reference = next(
            entry["model"]
            for entry in catalog["entries"]
            if entry["entry_id"] == "shapeformer_channel_specific_osd"
        )
        validate_model_config(reference, "raw")
        self.assertNotIn("shapelet_length_samples", reference)
        self.assertNotIn("discovery_stride_samples", reference)
        self.assertNotIn("num_pips", reference)
        self.assertEqual(reference["position_search_neighbourhood_samples"], 128)
        self.assertEqual(
            reference["information_gain_split_rule"],
            "upstream_positive_recall_grid_0p2",
        )
        self.assertEqual(reference["sequence_length_samples"], 2000)
        self.assertEqual(
            reference["architecture_parameters"]["information_gain_split_rule"],
            "upstream_positive_recall_grid_0p2",
        )
        self.assertEqual(reference["architecture_parameters"]["generic_branch_channel_count"], 8)
        self.assertEqual(
            reference["input_channel_order"],
            ["RED", "IR", "A_dyn_x", "A_dyn_y", "A_dyn_z", "GX", "GY", "GZ"],
        )
        self.assertEqual(
            reference["input_channels_resolution"],
            "canonical_frailty_raw_8",
        )
        self.assertNotIn("input_projection", reference)
        for entry in catalog["entries"]:
            if entry["representation_mode"] not in {"raw", "fusion"}:
                continue
            model = entry["model"]
            self.assertEqual(model["input_channels"], 8)
            self.assertEqual(
                model["input_channels_resolution"],
                "canonical_frailty_raw_8",
            )
            self.assertEqual(model["input_channel_order"], reference["input_channel_order"])
            self.assertNotIn("input_projection", model)
        fixed_length_reference = dict(reference)
        fixed_length_reference["shapelet_length_samples"] = 128
        with self.assertRaisesRegex(ValueError, "unknown=.*shapelet_length_samples"):
            validate_model_config(fixed_length_reference, "raw")

    def test_canonical_feature_vector_is_role_aware(self) -> None:
        """Canonical training and aggregation remain role-aware and paired."""

        canonical = load_config(ROOT / "configs/reference_static_feature_vector_v2.yaml")
        self.assertEqual(
            canonical.payload["training"]["training_balance"],
            "equal_role_families",
        )
        self.assertEqual(
            canonical.payload["aggregation"]["balance_line"],
            "line_b_equal_role_families",
        )
        mutated = canonical.to_dict()
        mutated["training"]["training_balance"] = "equal_files"
        with self.assertRaisesRegex(ValueError, "matched A-A or B-B"):
            validate_config_payload(mutated)

        motion9 = canonical.to_dict()
        motion9["signal"]["normalization"]["raw_imu"] = (
            "outer_training_participant_only_robust_scaler_motion9_augmentation"
        )
        with self.assertRaisesRegex(ValueError, "canonical axes6"):
            validate_config_payload(motion9)

    def test_formal_ablation_profiles_are_exact_and_never_auto_run(self) -> None:
        catalog = load_formal_ablation_profiles(
            ROOT / "configs/formal_ablation_profiles_v2.yaml"
        )
        self.assertEqual(
            catalog["execution_policy"],
            {
                "auto_run": False,
                "materialization_only": True,
                "allow_cartesian_product": False,
                "safe_suite_execution": False,
            },
        )
        families = catalog["families"]
        self.assertEqual(
            [row["fixed_epochs"] for row in families["deep_fixed_epoch"]["entries"]],
            [7, 10, 15],
        )
        self.assertEqual(len(families["fixed_kernel_samples"]["cases"]), 12)
        self.assertEqual(
            {row["case_id"] for row in families["fixed_kernel_samples"]["cases"]},
            {
                f"{model}__{case}"
                for model in ("compactcnn1d", "inceptiontimefull")
                for case in (
                    "reference", "context_10s", "fs_100", "fs_160",
                    "fs_200", "dilation_2",
                )
            },
        )
        self.assertEqual(
            families["direct_filter"]["entries"][1]["profile_id"],
            "direct_filter_0p5_to_5hz_ablation",
        )
        self.assertIs(families["imu_gravity"]["silent_fallback_forbidden"], True)

    def test_single_factor_configs_materialize_without_execution(self) -> None:
        profiles = ROOT / "configs/formal_ablation_profiles_v2.yaml"
        raw = ROOT / "configs/reference_static_role_aware_v2.yaml"
        with tempfile.TemporaryDirectory(dir=ROOT / "tests/contracts") as directory:
            output = Path(directory)
            epoch = materialize_formal_ablation_config(
                raw,
                family="deep_fixed_epoch",
                profile_id="epoch_7_ablation",
                output_path=output / "epoch7.yaml",
                profiles_path=profiles,
            )
            self.assertEqual(epoch.payload["training"]["fixed_epochs"], 7)
            self.assertFalse(
                epoch.payload["output"]["formal_ablation_materialization"][
                    "scientific_execution_completed"
                ]
            )
            fixed = materialize_formal_ablation_config(
                raw,
                family="fixed_kernel_samples",
                profile_id="compactcnn1d__fs_100",
                output_path=output / "fixed.yaml",
                profiles_path=profiles,
            )
            self.assertEqual(
                fixed.payload["signal"]["dl_resampling"]["case_id"],
                "compactcnn1d__fs_100",
            )
            self.assertEqual(fixed.payload["model"]["kernel_sizes"], [9, 9, 7])
            self.assertEqual(fixed.payload["windows"]["raw_dl"]["length_s"], 5.0)
            self.assertEqual(_model_input_sampling_rate_hz(fixed), 100.0)
            self.assertEqual(_model_input_sampling_rate_hz(load_config(raw)), 400.0)
            for profile_id, expected_hz in (
                ("compactcnn1d__reference", 400.0),
                ("compactcnn1d__fs_160", 160.0),
                ("compactcnn1d__fs_200", 200.0),
            ):
                sampled = materialize_formal_ablation_config(
                    raw,
                    family="fixed_kernel_samples",
                    profile_id=profile_id,
                    output_path=output / f"{profile_id}.yaml",
                    profiles_path=profiles,
                )
                self.assertEqual(
                    _model_input_sampling_rate_hz(sampled),
                    expected_hz,
                )
            filtered = materialize_formal_ablation_config(
                raw,
                family="direct_filter",
                profile_id="direct_filter_0p5_to_5hz_ablation",
                output_path=output / "filter.yaml",
                profiles_path=profiles,
            )
            self.assertEqual(
                filtered.payload["signal"]["analysis_view"]["direct_source"],
                "x_filter_0p5_to_5hz",
            )
            gravity = materialize_formal_ablation_config(
                raw,
                family="imu_gravity",
                profile_id="imu_lpf_0p3hz_ablation",
                output_path=output / "gravity.yaml",
                profiles_path=profiles,
            )
            self.assertEqual(
                gravity.payload["signal"]["imu"]["gravity_method"],
                "low_pass_0p3hz",
            )
            with self.assertRaisesRegex(ValueError, "single-factor"):
                materialize_formal_ablation_config(
                    Path(epoch.source_path),
                    family="direct_filter",
                    profile_id="direct_filter_0p5_to_5hz_ablation",
                    output_path=output / "cartesian.yaml",
                    profiles_path=profiles,
                )

    def test_epoch_profile_rejects_classical_config(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests/contracts") as directory:
            with self.assertRaisesRegex(ValueError, "deep-model-only"):
                materialize_formal_ablation_config(
                    ROOT / "configs/reference_static_feature_vector_v2.yaml",
                    family="deep_fixed_epoch",
                    profile_id="epoch_7_ablation",
                    output_path=Path(directory) / "invalid.yaml",
                    profiles_path=ROOT / "configs/formal_ablation_profiles_v2.yaml",
                )

    def test_sqi_off_rejects_non_br_or_duplicate_role_ids(self) -> None:
        """SQI关闭时只允许精确B/R文件ID / Off mode is exactly B/R-only."""

        payload = load_config(
            ROOT / "configs/reference_static_feature_vector_v2.yaml"
        ).to_dict()
        payload["roles"].append("S1")
        with self.assertRaisesRegex(ValueError, "exactly B,R1,R2,R3,R4"):
            validate_config_payload(payload)
        payload["roles"] = ["B", "R1", "R1", "R3", "R4"]
        with self.assertRaisesRegex(ValueError, "exactly B,R1,R2,R3,R4"):
            validate_config_payload(payload)

    def test_quality_route_fails_until_supervised_ready(self) -> None:
        """未监督阈值不得路由 / Routing is gated by supervised readiness."""

        payload = load_config(ROOT / "configs/reference_static_feature_vector_v2.yaml").to_dict()
        payload["quality"]["mode"] = "route"
        payload["quality"]["rate_threshold"] = 0.5
        payload["quality"]["morph_threshold"] = 0.5
        with self.assertRaisesRegex(ValueError, "supervised authority artifact"):
            validate_config_payload(payload)
        payload["quality"]["supervised_route_ready"] = True
        with self.assertRaisesRegex(ValueError, "YAML boolean cannot authorize"):
            validate_config_payload(payload)

    def test_legacy_requires_explicit_provenance_mode(self) -> None:
        """复制 V1 config 不得进入 formal loader / V1 requires explicit legacy mode."""

        path = ROOT / "historical/v1_transition/configs/reference_static_v1.yaml"
        with self.assertRaisesRegex(ValueError, "provenance-only"):
            load_config(path)
        legacy = load_config(path, allow_legacy=True)
        self.assertTrue(legacy.is_legacy)

    def test_decision_and_runtime_dependencies_are_machine_readable(self) -> None:
        decision = load_v2_decision_profile(ROOT / "configs/v2_decision_profile.yaml")
        self.assertEqual(
            decision["confirmed_defaults"]["quality"]["default_mode"],
            "off",
        )
        prv_contract = decision["confirmed_defaults"]["prv"]["comparison_contract"]
        self.assertEqual(prv_contract["input_unit"], "milliseconds")
        self.assertEqual(len(prv_contract["fixture_ids"]), 5)
        self.assertEqual(prv_contract["input_cleaning"], "none")
        self.assertIs(prv_contract["classifier_integrated"], False)
        config = load_config(
            ROOT / "configs/reference_static_feature_vector_v2.yaml"
        )
        report = dependency_availability_report(config)
        self.assertEqual(
            report["policy"],
            "ordinary_import_availability_no_version_or_origin_lock",
        )
        self.assertEqual(
            [row["module"] for row in report["modules"]],
            list(required_runtime_modules(config)),
        )

    def test_shapeformer_runtime_dependencies_include_torch(self) -> None:
        base = load_config(ROOT / "configs/reference_static_role_aware_v2.yaml")
        payload = base.to_dict()
        payload["model"]["model_id"] = (
            "ShapeFormerChannelSpecificScalarDistanceAblation"
        )
        config = PipelineConfig(
            payload=payload,
            source_path="synthetic_dependency_contract_only",
            sha256="0" * 64,
        )
        self.assertIn("torch", required_runtime_modules(config))


if __name__ == "__main__":
    unittest.main()
