"""V2 配置、决策与依赖合同 / V2 config, decision, and dependency contracts."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ppg_frailty.config import (
    _live_exact_lock_evidence,
    PipelineConfig,
    V2_DEPENDENCY_PROFILE_IDS,
    dependency_gate_report,
    load_config,
    load_dependency_profiles,
    load_formal_ablation_profiles,
    load_formal_experiment_catalog,
    load_v2_decision_profile,
    materialize_formal_ablation_config,
    required_dependency_profile_ids,
    validate_config_payload,
)
from ppg_frailty.module_registry import list_modules, validate_model_config
from ppg_frailty.experiment import _model_input_sampling_rate_hz


ROOT = Path(__file__).resolve().parents[2]


class V2ConfigurationTests(unittest.TestCase):
    """验证默认关闭门与显式comparison / Check defaults and explicit comparisons."""

    def test_default_formal_config_is_sqi_off_br_line_a_fixed10(self) -> None:
        """默认不得偷偷启用未监督路线 / Default must not activate an unready route."""

        config = load_config(ROOT / "configs/reference_static_feature_vector_v2.yaml")
        self.assertEqual(config.schema_version, "ppg_frailty.pipeline_config.v2")
        self.assertEqual(config.payload["roles"], ["B", "R1", "R2", "R3", "R4"])
        self.assertEqual(config.payload["quality"]["mode"], "off")
        self.assertIs(config.payload["quality"]["supervised_route_ready"], False)
        self.assertEqual(config.payload["training"]["training_balance"], "equal_files")
        self.assertEqual(config.payload["training"]["epoch_profile"], "default_10")
        self.assertEqual(config.payload["training"]["fixed_epochs"], 10)
        self.assertEqual(config.payload["aggregation"]["balance_line"], "line_a_equal_files")

    def test_four_representation_formal_entrypoints_preflight(self) -> None:
        """四表征配置均为Line A/SQI-off/fixed10 / Four formal entrypoints."""

        expected = {
            "reference_static_line_a_v2.yaml": ("raw", "CompactCNN1D"),
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
                "line_a_equal_files",
            )
            validate_model_config(config.payload["model"], mode)

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
        fixed_length_reference = dict(reference)
        fixed_length_reference["shapelet_length_samples"] = 128
        with self.assertRaisesRegex(ValueError, "unknown=.*shapelet_length_samples"):
            validate_model_config(fixed_length_reference, "raw")

    def test_line_b_is_matched_and_one_factor(self) -> None:
        """Line B 训练与聚合不可拆开 / Line B train and aggregation remain paired."""

        line_a = load_config(ROOT / "configs/reference_static_feature_vector_v2.yaml")
        line_b = load_config(ROOT / "configs/reference_static_feature_vector_line_b_v2.yaml")
        self.assertEqual(line_b.payload["training"]["training_balance"], "equal_role_families")
        self.assertEqual(line_b.payload["aggregation"]["balance_line"], "line_b_equal_role_families")
        mutated = line_b.to_dict()
        mutated["training"]["training_balance"] = "equal_files"
        with self.assertRaisesRegex(ValueError, "matched A-A or B-B"):
            validate_config_payload(mutated)
        ignored = {"config_id", "training", "aggregation"}
        self.assertEqual(
            {key: value for key, value in line_a.payload.items() if key not in ignored},
            {key: value for key, value in line_b.payload.items() if key not in ignored},
        )

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
        raw = ROOT / "configs/reference_static_line_a_v2.yaml"
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

    def test_decision_and_six_dependency_profiles_are_machine_readable(self) -> None:
        """延期 lock 不得伪装已验证 / Pending locks cannot claim resolved packages."""

        decision = load_v2_decision_profile(ROOT / "configs/v2_decision_profile.yaml")
        profiles, locks = load_dependency_profiles(
            ROOT / "requirements/profiles.json", ROOT / "locks/profiles.lock.json"
        )
        self.assertEqual(decision["confirmed_defaults"]["quality"]["default_mode"], "off")
        prv_contract = decision["confirmed_defaults"]["prv"]["comparison_contract"]
        self.assertEqual(prv_contract["input_unit"], "milliseconds")
        self.assertEqual(len(prv_contract["fixture_ids"]), 5)
        self.assertEqual(prv_contract["input_cleaning"], "none")
        self.assertIs(prv_contract["identical_input_sha256_required"], True)
        self.assertIs(prv_contract["classifier_integrated"], False)
        self.assertEqual(
            prv_contract["missing_optional_dependency_status"],
            "unavailable_optional_dependency",
        )
        self.assertEqual(
            {row["profile_id"] for row in profiles["profiles"]},
            V2_DEPENDENCY_PROFILE_IDS,
        )
        lock_by_id = {row["profile_id"]: row for row in locks["profiles"]}
        pending_ids = {
            "prv_rhenan_legacy_compare",
        }
        self.assertTrue(
            all(
                lock_by_id[profile_id]["status"]
                == "pending_profile_install_and_full_regression"
                and not lock_by_id[profile_id]["resolved_packages"]
                for profile_id in pending_ids
            )
        )
        self.assertEqual(
            {
                profile_id
                for profile_id in (
                    "core", "deep", "formal_benchmark", "onnx_winner_gate"
                )
                if lock_by_id[profile_id]["status"] == "validated_exact_lock"
            },
            {"core", "deep", "formal_benchmark", "onnx_winner_gate"},
        )
        self.assertEqual(
            lock_by_id["formal_benchmark"]["status"],
            "validated_exact_lock",
        )
        self.assertIn(
            "pyarrow==25.0.1",
            lock_by_id["formal_benchmark"]["resolved_packages"],
        )
        self.assertEqual(
            lock_by_id["prv_aura_compare"]["status"],
            "validated_exact_lock",
        )
        self.assertIn(
            "hrv-analysis==1.0.2",
            lock_by_id["prv_aura_compare"]["resolved_packages"],
        )
        self.assertIn(
            "nolds==0.6.2",
            lock_by_id["prv_aura_compare"]["resolved_packages"],
        )
        self.assertIn(
            "astropy==5.2.2",
            lock_by_id["prv_aura_compare"]["resolved_packages"],
        )
        self.assertIn(
            "numpy==1.26.4",
            lock_by_id["prv_aura_compare"]["resolved_packages"],
        )
        self.assertEqual(
            lock_by_id["prv_aura_compare"]["environment_inventory_path"],
            "locks/prv_aura_hrv102_py311_v2.json",
        )
        resolution_policy = profiles["resolution_policy"]
        self.assertIn("do_not_modify_conda_ml", resolution_policy["prv_aura_compare"])
        self.assertIn("do_not_modify_conda_ml", resolution_policy["onnx_winner_gate"])
        aura_inventory = json.loads(
            (
                ROOT
                / lock_by_id["prv_aura_compare"]["environment_inventory_path"]
            ).read_text(encoding="utf-8")
        )
        aura_exact_set = aura_inventory["exact_installed_distribution_set"]
        self.assertEqual(
            aura_exact_set["policy"],
            "exact_records_no_unknown_distributions",
        )
        self.assertEqual(aura_exact_set["profile_closure_count"], 22)
        self.assertEqual(aura_exact_set["record_count"], 24)
        aura_exact_records = tuple(
            (ROOT / aura_exact_set["records_path"]).read_text(
                encoding="utf-8"
            ).splitlines()
        )
        with patch(
            "ppg_frailty.config._installed_distribution_records",
            return_value=tuple(
                (*aura_exact_records, "unexpected==9.9@isolated_prefix")
            ),
        ):
            aura_live = _live_exact_lock_evidence(
                lock_by_id["prv_aura_compare"],
                pipeline_root=ROOT,
                run_pip_check=False,
            )
        self.assertIs(aura_live["installed_distribution_set_match"], False)
        self.assertEqual(
            aura_live["installed_distribution_evidence"]["unexpected_records"],
            ["unexpected==9.9@isolated_prefix"],
        )
        onnx_lock = lock_by_id["onnx_winner_gate"]
        self.assertEqual(onnx_lock["status"], "validated_exact_lock")
        self.assertEqual(
            onnx_lock["environment_inventory_path"],
            "locks/onnx_winner_gate_py311_v2.json",
        )
        inventory_path = ROOT / onnx_lock["environment_inventory_path"]
        probe_path = ROOT / onnx_lock["isolated_probe_path"]
        self.assertEqual(
            hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
            onnx_lock["environment_inventory_sha256"],
        )
        self.assertEqual(
            hashlib.sha256(probe_path.read_bytes()).hexdigest(),
            onnx_lock["isolated_probe_sha256"],
        )
        onnx_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        self.assertEqual(
            onnx_inventory["python"]["executable_path_policy"],
            "lexical_with_resolved_base_binding",
        )
        self.assertEqual(
            onnx_inventory["python"]["executable_resolved_relative_to_base_prefix"],
            "bin/python3.11",
        )
        onnx_probe = json.loads(probe_path.read_text(encoding="utf-8"))
        self.assertEqual(
            onnx_probe["status"],
            "passed_non_scientific_probe_exact_isolated_lock_validated",
        )
        self.assertIs(onnx_probe["probe"]["sklearn_logistic"]["parity_passed"], True)
        self.assertIs(onnx_probe["probe"]["torch_module"]["parity_passed"], True)
        self.assertIs(onnx_probe["probe"]["training_executed"], False)
        self.assertIs(onnx_probe["probe"]["cross_validation_executed"], False)
        self.assertIn("onnxscript==0.5.7", onnx_lock["resolved_packages"])
        self.assertIn("onnx-ir==0.1.13", onnx_lock["resolved_packages"])
        exact_set = onnx_inventory["exact_installed_distribution_set"]
        self.assertEqual(exact_set["policy"], "exact_records_no_unknown_distributions")
        self.assertEqual(exact_set["profile_closure_count"], 41)
        self.assertEqual(exact_set["record_count"], 131)
        exact_records = tuple(
            (ROOT / exact_set["records_path"]).read_text(
                encoding="utf-8"
            ).splitlines()
        )
        with patch(
            "ppg_frailty.config._installed_distribution_records",
            return_value=tuple((*exact_records, "unexpected==9.9@isolated_prefix")),
        ):
            live = _live_exact_lock_evidence(
                onnx_lock,
                pipeline_root=ROOT,
                run_pip_check=False,
            )
        self.assertIs(live["installed_distribution_set_match"], False)
        self.assertEqual(
            live["installed_distribution_evidence"]["unexpected_records"],
            ["unexpected==9.9@isolated_prefix"],
        )
        self.assertNotIn(
            ".fit(",
            (ROOT / "tools/validate_onnx_winner_profile.py").read_text(
                encoding="utf-8"
            ),
        )
        self.assertEqual(
            required_dependency_profile_ids(
                load_config(ROOT / "configs/reference_static_feature_vector_v2.yaml"),
                operation="onnx_winner_gate",
            ),
            ("onnx_winner_gate",),
        )
        supplemental = profiles["supplemental_optional_inputs"]
        self.assertEqual([row["input_id"] for row in supplemental], ["artifact_legacy_ablation"])
        self.assertIs(supplemental[0]["blocks_formal_benchmark"], False)
        formal_text = (ROOT / "requirements/requirements-formal-benchmark.txt").read_text(encoding="utf-8")
        self.assertNotIn("PyWavelets", formal_text)
        self.assertIn(
            "PyWavelets",
            (ROOT / "requirements/requirements-artifact-legacy-ablation.txt").read_text(encoding="utf-8"),
        )

    def test_scalar_distance_shapeformer_ablation_requires_deep_profile(self) -> None:
        base = load_config(ROOT / "configs/reference_static_line_a_v2.yaml")
        payload = base.to_dict()
        payload["model"]["model_id"] = (
            "ShapeFormerChannelSpecificScalarDistanceAblation"
        )
        config = PipelineConfig(
            payload=payload,
            source_path="synthetic_dependency_contract_only",
            sha256="0" * 64,
        )
        self.assertEqual(
            required_dependency_profile_ids(config, operation="preflight"),
            ("core", "deep"),
        )

    def test_exact_dependency_lock_rejects_inventory_hash_tamper(self) -> None:
        lock = json.loads(
            (ROOT / "locks/profiles.lock.json").read_text(encoding="utf-8")
        )
        exact = next(
            row for row in lock["profiles"] if row["profile_id"] == "core"
        )
        exact["environment_inventory_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            path = Path(directory) / "tampered.lock.json"
            path.write_text(
                json.dumps(lock, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "inventory hash drift"):
                load_dependency_profiles(
                    ROOT / "requirements/profiles.json",
                    path,
                )

    def test_formal_dependency_gate_rejects_live_package_drift(self) -> None:
        import importlib.metadata

        config = load_config(ROOT / "configs/reference_static_feature_vector_v2.yaml")
        original = importlib.metadata.version

        def drift(name: str) -> str:
            if name.lower() == "numpy":
                return "0.0.0-drift"
            return original(name)

        with patch("importlib.metadata.version", side_effect=drift):
            with self.assertRaisesRegex(RuntimeError, "exact-lock gate is closed"):
                dependency_gate_report(
                    config,
                    operation="formal_benchmark",
                    profiles_path=ROOT / "requirements/profiles.json",
                    lock_path=ROOT / "locks/profiles.lock.json",
                    require_exact_lock=True,
                )


if __name__ == "__main__":
    unittest.main()
