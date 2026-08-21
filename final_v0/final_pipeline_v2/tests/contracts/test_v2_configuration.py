"""V2 配置、决策与依赖合同 / V2 config, decision, and dependency contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ppg_frailty.config import (
    PipelineConfig,
    canonical_json_bytes,
    dependency_availability_report,
    load_config,
    load_formal_ablation_profiles,
    load_formal_experiment_catalog,
    load_v2_decision_profile,
    materialize_formal_ablation_config,
    required_runtime_modules,
    validate_config_payload,
)
from ppg_frailty.module_registry import (
    list_modules,
    normalize_window_config,
    resolve_window_config,
    validate_model_config,
)
from ppg_frailty.experiment import _classifier_role_ids, _model_input_sampling_rate_hz


ROOT = Path(__file__).resolve().parents[2]


class V2ConfigurationTests(unittest.TestCase):
    """验证默认关闭门与显式comparison / Check defaults and explicit comparisons."""

    def test_default_formal_config_is_sqi_off_br_role_aware_fixed10(self) -> None:
        """默认不得偷偷启用未监督路线 / Default must not activate an unready route."""

        config = load_config(ROOT / "configs/reference_static_feature_vector_v2.yaml")
        self.assertEqual(config.schema_version, "ppg_frailty.pipeline_config.v2")
        self.assertEqual(config.payload["roles"], ["B", "R1", "R2", "R3", "R4"])
        self.assertEqual(config.payload["quality"]["mode"], "off")
        self.assertNotIn("supervised_route_ready", config.payload["quality"])
        self.assertEqual(config.payload["quality"]["flatline_duration_s"], 1.0)
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

    def test_formal_training_controls_accept_registered_alternatives(self) -> None:
        canonical = load_config(
            ROOT / "configs" / "reference_static_role_aware_v2.yaml"
        ).to_dict()
        for field, value in (
            ("optimizer", "adamw"),
            ("sampler", "uniform_replacement"),
            ("sampler", "exhaustive_shuffle_without_replacement"),
            ("class_count_basis", "row"),
            ("class_weighting", "none"),
        ):
            with self.subTest(field=field, value=value):
                varied = json.loads(json.dumps(canonical))
                varied["training"][field] = value
                validate_config_payload(varied)

    def test_config_id_is_identity_not_an_algorithm_profile_gate(self) -> None:
        canonical = load_config(
            ROOT / "configs" / "reference_static_role_aware_v2.yaml"
        ).to_dict()
        canonical["config_id"] = "user_selected_optimizer_and_window_trial"
        resolved = validate_config_payload(canonical)
        self.assertEqual(
            resolved["config_id"],
            "user_selected_optimizer_and_window_trial",
        )

        for unsafe in ("", "../escape", "nested/case", "nested\\case"):
            with self.subTest(unsafe=unsafe):
                varied = json.loads(json.dumps(canonical))
                varied["config_id"] = unsafe
                with self.assertRaisesRegex(ValueError, "path-safe identifier"):
                    validate_config_payload(varied)

    def test_raw_normalization_strategies_are_materialized_and_hash_bound(self) -> None:
        source = load_config(
            ROOT / "configs" / "reference_static_role_aware_v2.yaml"
        ).to_dict()
        hashes: set[str] = set()
        with tempfile.TemporaryDirectory() as directory:
            for index, (ppg, imu) in enumerate((
                ("robust", "outer_train_robust"),
                ("standard_zscore", "outer_train_mean_std"),
                ("none", "none"),
            )):
                varied = json.loads(json.dumps(source))
                varied["signal"]["normalization"]["raw_ppg"] = ppg
                varied["signal"]["normalization"]["raw_imu"] = imu
                path = Path(directory) / f"normalization_{index}.yaml"
                path.write_text(json.dumps(varied), encoding="utf-8")
                loaded = load_config(path)
                effective = loaded.payload["signal"]["normalization"]
                self.assertEqual(
                    set(effective),
                    {
                        "raw_ppg", "raw_imu", "iqr_fallback",
                        "clip_after_scale", "robust_iqr_divisor",
                        "mad_consistency_divisor", "scale_epsilon",
                        "standard_ddof",
                    },
                )
                hashes.add(loaded.sha256)
        self.assertEqual(len(hashes), 3)
        self.assertEqual(
            {row["module_id"] for row in list_modules("normalization")},
            {
                "ppg_per_window_robust",
                "ppg_per_window_standard_zscore",
                "ppg_none",
                "imu_outer_train_robust",
                "imu_outer_train_mean_std",
                "imu_none",
            },
        )

    def test_non_raw_representations_reject_inactive_normalization_axes(self) -> None:
        for filename in (
            "reference_static_feature_vector_v2.yaml",
            "reference_static_feature_matrix_v2.yaml",
        ):
            with self.subTest(filename=filename):
                payload = load_config(ROOT / "configs" / filename).to_dict()
                payload["signal"]["normalization"]["raw_ppg"] = "none"
                with self.assertRaisesRegex(ValueError, "requires representation_mode"):
                    validate_config_payload(payload)

                payload = load_config(ROOT / "configs" / filename).to_dict()
                payload["signal"]["normalization"]["robust_iqr_divisor"] = 1.5
                with self.assertRaisesRegex(ValueError, "requires representation_mode"):
                    validate_config_payload(payload)

    def test_parameterized_signal_and_window_modules_are_registry_discoverable(self) -> None:
        expected = {
            "ppg_filter": {"butterworth_sos"},
            "gap_repair": {"linear_inside_only"},
            "imu_gravity": {
                "calibrated_roll_pitch_ekf",
                "profile_a_lowpass_0p3hz",
                "quaternion_error_state_ekf",
                "low_pass_0p3hz",
            },
            "dl_resampling": {
                "off_identity_source_grid",
                "polyphase_anti_alias",
            },
            "window_profile": {"engineering", "raw_dl"},
        }
        for family, module_ids in expected.items():
            with self.subTest(family=family):
                rows = list_modules(family)
                self.assertEqual(
                    {str(row["module_id"]) for row in rows},
                    module_ids,
                )
                self.assertTrue(
                    all("runtime" in str(row["scientific_status"]) for row in rows)
                )

    def test_signal_structural_fields_fail_during_config_materialization(self) -> None:
        reference = load_config(
            ROOT / "configs" / "reference_static_role_aware_v2.yaml"
        ).to_dict()
        invalid_cases = (
            (("signal", "ppg_filter", "family"), "chebyshev"),
            (("signal", "ppg_filter", "phase"), "causal"),
            (("signal", "ppg_filter", "notch_enabled"), True),
            (("signal", "analysis_view", "direct_source"), "x_filter_99_to_100hz"),
            (("signal", "analysis_view", "additional_filter"), "bandpass_again"),
            (("signal", "gap_repair", "method"), "cubic"),
            (("signal", "gap_repair", "edge_extrapolation"), True),
            (("signal", "accelerometer_input_unit"), "raw_counts"),
            (("signal", "gyroscope_input_unit"), "rpm"),
            (("signal", "channel_order"), ["IR", "RED"]),
            (("signal", "imu", "gravity_method"), "automatic"),
            (("signal", "imu", "initialization"), "automatic"),
            (
                ("signal", "imu", "initialization"),
                "online_no_precalibration",
            ),
            (("signal", "imu", "comparison_method"), "automatic"),
            (("signal", "imu", "comparison_method"), "lowpass_0p3hz"),
            (("signal", "imu", "required_axes"), 9),
        )
        for path, value in invalid_cases:
            with self.subTest(path=".".join(path)):
                payload = json.loads(json.dumps(reference))
                target = payload
                for name in path[:-1]:
                    target = target[name]
                target[path[-1]] = value
                with self.assertRaises(ValueError):
                    validate_config_payload(payload)

    def test_signal_preprocessing_defaults_materialize_before_hashing(self) -> None:
        payload = load_config(
            ROOT / "configs" / "reference_static_role_aware_v2.yaml"
        ).to_dict()
        for name in (
            "internal_fs_hz",
            "channel_order",
            "ppg_native_unit",
            "accelerometer_input_unit",
            "gyroscope_input_unit",
            "ppg_filter",
            "analysis_view",
            "gap_repair",
            "imu",
        ):
            payload["signal"].pop(name)
        resolved = validate_config_payload(payload)["signal"]
        self.assertEqual(resolved["internal_fs_hz"], 400.0)
        self.assertEqual(resolved["accelerometer_input_unit"], "g")
        self.assertEqual(resolved["gyroscope_input_unit"], "deg/s")
        self.assertEqual(resolved["ppg_filter"]["low_hz"], 0.2)
        self.assertEqual(
            resolved["analysis_view"]["direct_source"],
            "x_filter_0p2_to_8hz",
        )
        self.assertEqual(resolved["gap_repair"]["max_gap_samples"], 100)
        self.assertEqual(
            resolved["imu"]["gravity_method"],
            "calibrated_roll_pitch_ekf",
        )

    def test_signal_derived_aliases_have_one_effective_hash(self) -> None:
        reference = load_config(
            ROOT / "configs" / "reference_static_role_aware_v2.yaml"
        )
        for alias in (
            "x_filter",
            "configured_ppg_filter",
            "x_filter_0p2_to_8hz",
        ):
            with self.subTest(direct_source=alias):
                payload = reference.to_dict()
                payload["signal"]["analysis_view"]["direct_source"] = alias
                resolved = validate_config_payload(payload)
                self.assertEqual(
                    resolved["signal"]["analysis_view"]["direct_source"],
                    "x_filter_0p2_to_8hz",
                )
                self.assertEqual(
                    hashlib.sha256(canonical_json_bytes(resolved)).hexdigest(),
                    reference.sha256,
                )

    def test_imu_effective_identity_contains_only_active_profile_parameters(self) -> None:
        from ppg_frailty.signal import roll_pitch_ekf_config_from_resolved

        reference = load_config(
            ROOT / "configs" / "reference_static_role_aware_v2.yaml"
        )
        reference_payload = reference.to_dict()
        reference_imu = reference_payload["signal"]["imu"]
        self.assertNotIn("gravity_lowpass_hz", reference_imu)
        self.assertNotIn("gravity_filter_order", reference_imu)

        default_inactive = reference.to_dict()
        default_inactive["signal"]["imu"]["gravity_lowpass_hz"] = 0.3
        default_inactive["signal"]["imu"]["gravity_filter_order"] = 4
        resolved_default = validate_config_payload(default_inactive)
        self.assertEqual(
            hashlib.sha256(canonical_json_bytes(resolved_default)).hexdigest(),
            reference.sha256,
        )

        invalid_inactive = reference.to_dict()
        invalid_inactive["signal"]["imu"]["gravity_lowpass_hz"] = 0.8
        with self.assertRaisesRegex(ValueError, "inactive"):
            validate_config_payload(invalid_inactive)

        ekf_varied = reference.to_dict()
        ekf_varied["signal"]["imu"][
            "process_covariance_diagonal_per_second"
        ][0] = 7.0
        resolved_ekf = validate_config_payload(ekf_varied)
        self.assertNotEqual(
            hashlib.sha256(canonical_json_bytes(resolved_ekf)).hexdigest(),
            reference.sha256,
        )
        self.assertEqual(
            roll_pitch_ekf_config_from_resolved(
                resolved_ekf["signal"]["imu"]
            ).process_covariance_diagonal_per_second[0],
            7.0,
        )

        lpf_default = reference.to_dict()
        lpf_default["signal"]["imu"]["gravity_method"] = (
            "profile_a_lowpass_0p3hz"
        )
        resolved_lpf_default = validate_config_payload(lpf_default)
        lpf_imu = resolved_lpf_default["signal"]["imu"]
        self.assertNotIn("process_covariance_diagonal_per_second", lpf_imu)
        self.assertEqual(lpf_imu["gravity_lowpass_hz"], 0.3)
        self.assertEqual(lpf_imu["gravity_filter_order"], 4)

        lpf_varied = json.loads(json.dumps(resolved_lpf_default))
        lpf_varied["signal"]["imu"]["gravity_lowpass_hz"] = 0.8
        lpf_varied["signal"]["imu"]["gravity_filter_order"] = 3
        resolved_lpf_varied = validate_config_payload(lpf_varied)
        self.assertNotEqual(
            hashlib.sha256(canonical_json_bytes(resolved_lpf_varied)).hexdigest(),
            hashlib.sha256(canonical_json_bytes(resolved_lpf_default)).hexdigest(),
        )
        runtime_lpf = roll_pitch_ekf_config_from_resolved(
            resolved_lpf_varied["signal"]["imu"]
        )
        self.assertEqual(runtime_lpf.gravity_lowpass_hz, 0.8)
        self.assertEqual(runtime_lpf.gravity_filter_order, 3)

        inactive_lpf = json.loads(json.dumps(resolved_lpf_default))
        inactive_lpf["signal"]["imu"][
            "process_covariance_diagonal_per_second"
        ] = [7.0, 5.0, 0.05, 0.05, 0.05]
        with self.assertRaisesRegex(ValueError, "inactive"):
            validate_config_payload(inactive_lpf)

    def test_partial_module_defaults_are_persisted_in_effective_config(self) -> None:
        source = json.loads(
            json.dumps(
                load_config(
                    ROOT / "configs" / "reference_static_role_aware_v2.yaml"
                ).to_dict()
            )
        )
        for field in (
            "optimizer",
            "sampler",
            "class_weighting",
            "learning_rate",
            "weight_decay",
            "seed",
            "classifier_role_families",
            "n_classes",
            "epoch_rule",
            "execution_mode",
            "outer_labels_visible_to_trainer",
        ):
            source["training"].pop(field, None)
        source["windows"] = {"raw_dl": {"length_s": 4.0}}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial_v2.yaml"
            path.write_text(json.dumps(source), encoding="utf-8")
            config = load_config(path)

        effective = config.to_dict()
        self.assertEqual(effective["training"]["optimizer"], "adam")
        self.assertEqual(
            effective["training"]["sampler"], "balance_line_weighted_v2"
        )
        self.assertEqual(
            effective["training"]["class_weighting"],
            "inverse_frequency",
        )
        self.assertEqual(effective["training"]["class_count_basis"], "participant")
        self.assertEqual(effective["training"]["classifier_role_families"], ["B", "R"])
        self.assertEqual(effective["training"]["cache_policy"], "disabled")
        self.assertEqual(effective["training"]["n_classes"], 3)
        self.assertEqual(effective["windows"]["engineering"]["length_s"], 10.0)
        self.assertEqual(effective["windows"]["raw_dl"]["length_s"], 4.0)
        self.assertEqual(
            effective["windows"]["raw_dl"]["end_alignment"],
            "include_right_aligned_if_distinct",
        )
        self.assertEqual(effective["windows"]["raw_dl"]["cap_per_file"], 128)
        self.assertEqual(
            effective["signal"]["normalization"]["raw_ppg"],
            "per_window_robust",
        )
        self.assertEqual(
            effective["signal"]["normalization"]["raw_imu"],
            "none",
        )
        self.assertEqual(
            effective["signal"]["normalization"]["robust_iqr_divisor"],
            1.349,
        )
        self.assertEqual(
            config.sha256,
            hashlib.sha256(canonical_json_bytes(effective)).hexdigest(),
        )

    def test_empty_and_partial_window_mappings_match_reference_defaults(self) -> None:
        expected = {
            "shared_planner_version": "window_plan_v1",
            "engineering": {
                "length_s": 10.0,
                "hop_s": 2.0,
                "end_alignment": "left_start_regular_grid",
                "padding": "none_complete_windows_only",
                "cap_per_file": None,
                "cap_fraction_per_file": None,
                "min_valid_fraction": 1.0,
            },
            "raw_dl": {
                "length_s": 5.0,
                "hop_s": 2.5,
                "end_alignment": "include_right_aligned_if_distinct",
                "padding": "none_complete_windows_only",
                "cap_per_file": 128,
                "cap_fraction_per_file": None,
                "min_valid_fraction": 1.0,
            },
        }
        self.assertEqual(normalize_window_config({}), expected)

        partial = normalize_window_config({"raw_dl": {"length_s": 4.0}})
        self.assertEqual(partial["engineering"], expected["engineering"])
        self.assertEqual(
            partial["raw_dl"],
            {**expected["raw_dl"], "length_s": 4.0},
        )

    def test_dl_resampling_accepts_partial_config_and_arbitrary_valid_rate(self) -> None:
        payload = load_config(
            ROOT / "configs/reference_static_role_aware_v2.yaml"
        ).to_dict()
        payload["signal"]["dl_resampling"] = {
            "enabled": True,
            "target_fs_hz": 128.0,
        }
        resolved = validate_config_payload(payload)
        self.assertEqual(
            resolved["signal"]["dl_resampling"],
            {
                "enabled": True,
                "target_fs_hz": 128.0,
                "method": "polyphase_anti_alias",
                "preserve_feature_grid_hz": 400.0,
            },
        )

    def test_internal_grid_and_unimplemented_training_cache_fail_closed(self) -> None:
        payload = load_config(
            ROOT / "configs/reference_static_role_aware_v2.yaml"
        ).to_dict()
        payload["signal"]["internal_fs_hz"] = 200.0
        with self.assertRaisesRegex(ValueError, "implemented 400 Hz internal grid"):
            validate_config_payload(payload)

        payload = load_config(
            ROOT / "configs/reference_static_role_aware_v2.yaml"
        ).to_dict()
        payload["training"]["cache_policy"] = "content_addressed_strict"
        with self.assertRaisesRegex(ValueError, "cache_policy must be one of"):
            validate_config_payload(payload)

    def test_evaluation_resampling_budgets_and_seed_are_configurable(self) -> None:
        payload = load_config(
            ROOT / "configs/reference_static_feature_vector_v2.yaml"
        ).to_dict()
        payload["evaluation"] = {
            "statistics": {
                "bootstrap_replicates": 321,
                "paired_permutation_replicates": 654,
                "seed": 123,
            },
        }
        resolved = validate_config_payload(payload)
        self.assertEqual(
            resolved["evaluation"]["statistics"]["bootstrap_replicates"], 321
        )
        self.assertEqual(
            resolved["evaluation"]["statistics"]["paired_permutation_replicates"],
            654,
        )
        self.assertEqual(
            resolved["evaluation"]["statistics"]["seed"], 123
        )

    def test_formal_catalog_freezes_matched_member0_and_ensemble_seed_axes(self) -> None:
        catalog = load_formal_experiment_catalog(
            ROOT / "configs/formal_experiment_catalog_v2.yaml"
        )
        entries = {entry["entry_id"]: entry for entry in catalog["entries"]}
        models = {entry_id: entry["model"] for entry_id, entry in entries.items()}
        for entry_id in ("inception_full",):
            self.assertEqual(
                models[entry_id]["seed_policy"],
                "outer_cv_repeat_seed_equals_split_seed",
            )
        comparator_pairs = {
            "inception_full_member0_comparator": "inception_full",
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

    def test_feature_vector_parameters_accept_valid_runtime_ranges(self) -> None:
        catalog = load_formal_experiment_catalog(
            ROOT / "configs/formal_experiment_catalog_v2.yaml"
        )
        models = {
            entry["entry_id"]: entry["model"] for entry in catalog["entries"]
        }
        self.assertEqual(models["logistic_regression"]["logistic_c"], 1.0)
        self.assertEqual(models["extra_trees"]["extra_trees_max_features"], "sqrt")
        self.assertEqual(models["extra_trees"]["extra_trees_min_samples_leaf"], 1)

        custom_logistic = dict(models["logistic_regression"])
        custom_logistic["logistic_c"] = 0.2
        custom_logistic["logistic_max_iter"] = 321
        custom_logistic["logistic_solver"] = "saga"
        custom_logistic.pop("architecture_parameters")
        validate_model_config(custom_logistic, "feature_vector")

        custom_trees = dict(models["extra_trees"])
        custom_trees["extra_trees_n_estimators"] = 137
        custom_trees["extra_trees_max_features"] = "log2"
        custom_trees["extra_trees_min_samples_leaf"] = 0.125
        custom_trees.pop("architecture_parameters")
        validate_model_config(custom_trees, "feature_vector")

        invalid_logistic = dict(models["logistic_regression"])
        invalid_logistic["logistic_c"] = -0.2
        with self.assertRaisesRegex(ValueError, "logistic_c"):
            validate_model_config(invalid_logistic, "feature_vector")

        invalid_trees = dict(models["extra_trees"])
        invalid_trees["extra_trees_min_samples_leaf"] = 0.75
        with self.assertRaisesRegex(ValueError, "min_samples_leaf"):
            validate_model_config(invalid_trees, "feature_vector")

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
                "ShapeFormerChannelSpecificScalarDistanceAblation":
                    "optional_scalar_distance_ablation_not_literature_reference",
                "ShapeFormerEffectSizeFixedV1":
                    "parameterized_effect_size_ablation_legacy_name",
                "ShapeFormerLegacyEffectSizePort":
                    "legacy_parallel_ablation_not_osd_parity",
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

    def test_canonical_defaults_are_role_aware_but_modules_are_independent(self) -> None:
        """Reference defaults remain Line B without turning them into a gate."""

        canonical = load_config(ROOT / "configs/reference_static_feature_vector_v2.yaml")
        self.assertEqual(
            canonical.payload["training"]["training_balance"],
            "equal_role_families",
        )
        self.assertEqual(
            canonical.payload["aggregation"]["balance_line"],
            "line_b_equal_role_families",
        )
        self.assertEqual(
            canonical.payload["features"]["engineering_sequence_schema"],
            "engineering_10s_hop2s_thesis_115_v3",
        )
        mutated = canonical.to_dict()
        mutated["training"]["training_balance"] = "equal_files"
        validate_config_payload(mutated)
        self.assertEqual(
            mutated["aggregation"]["balance_line"],
            "line_b_equal_role_families",
        )

        motion9 = canonical.to_dict()
        motion9["signal"]["normalization"]["raw_imu"] = (
            "outer_training_participant_only_robust_scaler_motion9_augmentation"
        )
        with self.assertRaisesRegex(ValueError, "separate model input module"):
            validate_config_payload(motion9)

        stale_iqr = canonical.to_dict()
        stale_iqr["signal"]["normalization"]["raw_imu"] = (
            "outer_training_participant_only_robust_scaler_axes6"
        )
        with self.assertRaisesRegex(ValueError, "non-default signal.normalization"):
            validate_config_payload(stale_iqr)

        stale_engineering = canonical.to_dict()
        stale_engineering["features"][
            "engineering_sequence_schema"
        ] = "EngineeringFeatureSequenceV1"
        with self.assertRaisesRegex(ValueError, "engineering_sequence_schema"):
            validate_config_payload(stale_engineering)

        matrix = load_config(
            ROOT / "configs/reference_static_feature_matrix_v2.yaml"
        ).to_dict()
        self.assertEqual(matrix["features"]["matrix_k"], 150)
        self.assertEqual(matrix["windows"]["engineering"]["length_s"], 10.0)
        self.assertEqual(matrix["windows"]["engineering"]["hop_s"], 2.0)
        wrong_k = copy.deepcopy(matrix)
        wrong_k["features"]["matrix_k"] = 149
        with self.assertRaisesRegex(ValueError, "fixed at 150"):
            validate_config_payload(wrong_k)
        wrong_hop = copy.deepcopy(matrix)
        wrong_hop["windows"]["engineering"]["hop_s"] = 5.0
        with self.assertRaisesRegex(ValueError, "fixed 10 s/2 s"):
            validate_config_payload(wrong_hop)

        changed_feature_formula = canonical.to_dict()
        changed_feature_formula["features"]["tachogram_fs_hz"] = 8.0
        validate_config_payload(changed_feature_formula)

        unknown_feature_control = canonical.to_dict()
        unknown_feature_control["features"]["silent_algorithm_knob"] = True
        with self.assertRaisesRegex(ValueError, "features contains unknown"):
            validate_config_payload(unknown_feature_control)

        changed_aggregation_operator = canonical.to_dict()
        changed_aggregation_operator["aggregation"]["quality_weighting"] = True
        changed_aggregation_operator["quality"]["mode"] = "route"
        weighted = validate_config_payload(changed_aggregation_operator)
        self.assertEqual(
            weighted["aggregation"]["quality_weight_source"],
            "route_file_q_rate",
        )

        inert_quality_weighting = canonical.to_dict()
        inert_quality_weighting["aggregation"]["quality_weighting"] = True
        with self.assertRaisesRegex(ValueError, "route_file_q_rate requires"):
            validate_config_payload(inert_quality_weighting)

        unknown_aggregation_control = canonical.to_dict()
        unknown_aggregation_control["aggregation"]["silent_pooling_knob"] = "mean"
        with self.assertRaisesRegex(ValueError, "aggregation contains unknown"):
            validate_config_payload(unknown_aggregation_control)

        # The selected line is the sole user input; its hierarchy fields are
        # derived provenance, so changing A/B never requires five synchronized
        # edits to a copied effective config.
        line_a = canonical.to_dict()
        line_a["aggregation"]["balance_line"] = "line_a_equal_files"
        line_a = validate_config_payload(line_a)
        self.assertEqual(
            line_a["aggregation"]["hierarchy"],
            ["window", "file", "participant"],
        )
        self.assertEqual(line_a["aggregation"]["file_to_role"], "not_applicable")

        for section, field, value in (
            ("engineering", "length_s", 5.0),
            ("engineering", "hop_s", 1.0),
            ("raw_dl", "length_s", 10.0),
            ("raw_dl", "hop_s", 1.0),
        ):
            with self.subTest(window_section=section, field=field):
                varied_window = (
                    canonical.to_dict()
                    if section == "engineering"
                    else load_config(
                        ROOT / "configs/reference_static_role_aware_v2.yaml"
                    ).to_dict()
                )
                varied_window["windows"][section][field] = value
                validate_config_payload(varied_window)
                resolved = resolve_window_config(varied_window["windows"])
                runtime_field = (
                    "window_seconds" if field == "length_s" else "hop_seconds"
                )
                self.assertEqual(resolved[section][runtime_field], value)

        defaults = resolve_window_config({})
        self.assertEqual(
            defaults,
            resolve_window_config(canonical.to_dict()["windows"]),
        )
        self.assertEqual(defaults["engineering"]["window_seconds"], 10.0)
        self.assertEqual(defaults["engineering"]["hop_seconds"], 2.0)
        self.assertEqual(defaults["engineering"]["end_alignment"], "start")
        self.assertIsNone(defaults["engineering"]["max_windows"])
        self.assertEqual(defaults["engineering"]["cap_policy"], "not_applicable")
        self.assertIsNone(defaults["engineering"]["max_window_fraction"])
        self.assertEqual(defaults["raw_dl"]["window_seconds"], 5.0)
        self.assertEqual(defaults["raw_dl"]["hop_seconds"], 2.5)
        self.assertEqual(
            defaults["raw_dl"]["end_alignment"],
            "include_right_aligned_if_distinct",
        )
        self.assertEqual(defaults["raw_dl"]["max_windows"], 128)
        self.assertIsNone(defaults["raw_dl"]["max_window_fraction"])
        self.assertEqual(defaults["raw_dl"]["cap_policy"], "uniform_progress")
        partial = resolve_window_config({"raw_dl": {"length_s": 4.0}})
        self.assertEqual(partial["raw_dl"]["window_seconds"], 4.0)
        self.assertEqual(partial["raw_dl"]["hop_seconds"], 2.5)
        self.assertEqual(
            partial["raw_dl"]["end_alignment"],
            "include_right_aligned_if_distinct",
        )
        self.assertEqual(partial["raw_dl"]["max_windows"], 128)
        fractional = load_config(
            ROOT / "configs/reference_static_role_aware_v2.yaml"
        ).to_dict()
        fractional["windows"]["raw_dl"]["cap_per_file"] = None
        fractional["windows"]["raw_dl"]["cap_fraction_per_file"] = 0.4
        fractional = validate_config_payload(fractional)
        fractional_runtime = resolve_window_config(fractional["windows"])["raw_dl"]
        self.assertIsNone(fractional_runtime["max_windows"])
        self.assertEqual(fractional_runtime["max_window_fraction"], 0.4)
        self.assertEqual(fractional_runtime["cap_policy"], "uniform_progress")
        ambiguous = load_config(
            ROOT / "configs/reference_static_role_aware_v2.yaml"
        ).to_dict()
        ambiguous["windows"]["raw_dl"]["cap_fraction_per_file"] = 0.5
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            validate_config_payload(ambiguous)
        for invalid in (0.0, -1.0, float("nan"), float("inf"), True, "5"):
            with self.subTest(invalid_window_seconds=invalid):
                with self.assertRaisesRegex(ValueError, "finite and positive|numeric"):
                    resolve_window_config({"raw_dl": {"length_s": invalid}})

        for field, value in (
            ("sensor_lowpass_acc_hz", 30.0),
            ("sensor_lowpass_gyro_hz", 50.0),
            ("sensor_filter_order", 4),
        ):
            with self.subTest(imu_field=field):
                varied_imu = canonical.to_dict()
                varied_imu["signal"]["imu"][field] = value
                validate_config_payload(varied_imu)

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
                "profile_a_lowpass_0p3hz",
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

    def test_role_selection_accepts_registered_auxiliary_roles_but_not_duplicates(self) -> None:
        """Role selection is configurable; duplicate selectors remain invalid."""

        payload = load_config(
            ROOT / "configs/reference_static_feature_vector_v2.yaml"
        ).to_dict()
        payload["roles"].append("S1")
        payload["training"]["classifier_role_families"] = ["B", "R", "S"]
        resolved = validate_config_payload(payload)
        effective = PipelineConfig(
            resolved,
            "synthetic",
            hashlib.sha256(canonical_json_bytes(resolved)).hexdigest(),
        )
        self.assertEqual(
            _classifier_role_ids(effective),
            ("B", "R1", "R2", "R3", "R4", "S1"),
        )
        resolved["training"]["classifier_role_families"] = ["S"]
        s_only = validate_config_payload(resolved)
        effective_s_only = PipelineConfig(
            s_only,
            "synthetic",
            hashlib.sha256(canonical_json_bytes(s_only)).hexdigest(),
        )
        self.assertEqual(_classifier_role_ids(effective_s_only), ("S1",))
        payload["training"]["classifier_role_families"] = ["B", "W"]
        with self.assertRaisesRegex(ValueError, "represented by roles"):
            validate_config_payload(payload)
        payload["roles"] = ["B", "R1", "R1", "R3", "R4"]
        payload["training"]["classifier_role_families"] = ["B", "R"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_config_payload(payload)

    def test_role_set_permutations_have_one_effective_identity(self) -> None:
        source = load_config(
            ROOT / "configs/reference_static_feature_vector_v2.yaml"
        ).to_dict()
        source["roles"].append("S1")
        source["training"]["classifier_role_families"] = ["B", "R", "S"]
        permuted = json.loads(json.dumps(source))
        permuted["roles"] = ["S1", "R4", "R2", "B", "R3", "R1"]
        permuted["training"]["classifier_role_families"] = ["S", "R", "B"]

        resolved = validate_config_payload(source)
        resolved_permuted = validate_config_payload(permuted)

        self.assertEqual(
            resolved["roles"], ["B", "R1", "R2", "R3", "R4", "S1"]
        )
        self.assertEqual(
            resolved["training"]["classifier_role_families"],
            ["B", "R", "S"],
        )
        self.assertEqual(resolved, resolved_permuted)
        self.assertEqual(
            hashlib.sha256(canonical_json_bytes(resolved)).hexdigest(),
            hashlib.sha256(canonical_json_bytes(resolved_permuted)).hexdigest(),
        )

    def test_quality_route_is_an_executable_optional_module(self) -> None:
        """Route selection uses configured thresholds without a readiness gate."""

        payload = load_config(ROOT / "configs/reference_static_feature_vector_v2.yaml").to_dict()
        payload["quality"]["mode"] = "route"
        payload["quality"]["rate_threshold"] = 0.5
        payload["quality"]["morph_threshold"] = 0.5
        payload["quality"]["fit_scope"] = "stale_source_annotation"
        payload["quality"]["components"] = ["stale_source_annotation"]
        payload["quality"]["high_quality_rule"] = "stale_source_annotation"
        resolved = validate_config_payload(payload)
        self.assertEqual(
            resolved["quality"]["fit_scope"],
            "outer_training_participants_only",
        )
        self.assertIn("motion_energy_rms", resolved["quality"]["components"])
        self.assertEqual(
            resolved["quality"]["high_quality_rule"],
            "configured_endpoint_thresholds",
        )
        payload["quality"]["supervised_route_ready"] = True
        with self.assertRaisesRegex(ValueError, "is retired"):
            validate_config_payload(payload)
        payload["quality"]["supervised_route_ready"] = False
        with self.assertRaisesRegex(ValueError, "is retired"):
            validate_config_payload(payload)

    def test_inactive_quality_parameters_cannot_change_effective_hash(self) -> None:
        """Off/diagnostics persist only fields their runtime consumes."""

        canonical = load_config(
            ROOT / "configs/reference_static_role_aware_v2.yaml"
        )
        off = canonical.to_dict()
        off["quality"].update(
            {
                "rate_threshold": 0.91,
                "morph_threshold": 0.13,
                "minimum_coverage": 0.25,
                "calibrator": "fixed_formula_thresholds_v1",
                "calibrator_quantiles": [0.2, 0.8],
                "rate_component_weights": {"ignored_while_off": 99.0},
            }
        )
        resolved_off = validate_config_payload(off)
        self.assertEqual(
            hashlib.sha256(canonical_json_bytes(resolved_off)).hexdigest(),
            canonical.sha256,
        )
        self.assertNotIn("rate_threshold", resolved_off["quality"])
        self.assertNotIn("calibrator", resolved_off["quality"])

        diagnostics = canonical.to_dict()
        diagnostics["quality"]["mode"] = "diagnostics_only"
        resolved_diagnostics = validate_config_payload(diagnostics)
        inert_diagnostics = json.loads(json.dumps(resolved_diagnostics))
        inert_diagnostics["quality"].update(
            {
                "rate_threshold": 0.99,
                "morph_threshold": 0.01,
                "calibrator": "fixed_formula_thresholds_v1",
                "calibrator_quantiles": [0.3, 0.7],
                "rate_component_weights": {"ignored_while_diagnostic": 2.0},
                "morph_component_weights": {"ignored_while_diagnostic": 3.0},
            }
        )
        resolved_inert = validate_config_payload(inert_diagnostics)
        self.assertEqual(
            hashlib.sha256(canonical_json_bytes(resolved_inert)).hexdigest(),
            hashlib.sha256(
                canonical_json_bytes(resolved_diagnostics)
            ).hexdigest(),
        )
        self.assertNotIn("rate_threshold", resolved_inert["quality"])
        self.assertNotIn("calibrator", resolved_inert["quality"])

        physical = json.loads(json.dumps(resolved_diagnostics))
        physical["quality"]["cardiac_band_hz"] = [0.6, 3.1]
        resolved_physical = validate_config_payload(physical)
        self.assertNotEqual(
            hashlib.sha256(canonical_json_bytes(resolved_physical)).hexdigest(),
            hashlib.sha256(
                canonical_json_bytes(resolved_diagnostics)
            ).hexdigest(),
        )

    def test_quality_off_keeps_configurable_physical_flatline_admission(self) -> None:
        """Physical PPG admission runs even when endpoint SQI is disabled."""

        canonical = load_config(
            ROOT / "configs/reference_static_role_aware_v2.yaml"
        )
        self.assertEqual(canonical.payload["quality"]["mode"], "off")
        self.assertEqual(canonical.payload["quality"]["flatline_duration_s"], 1.0)

        changed = canonical.to_dict()
        changed["quality"]["flatline_duration_s"] = 2.5
        resolved = validate_config_payload(changed)
        self.assertEqual(resolved["quality"]["flatline_duration_s"], 2.5)
        self.assertNotEqual(
            hashlib.sha256(canonical_json_bytes(resolved)).hexdigest(),
            canonical.sha256,
        )

        invalid = canonical.to_dict()
        invalid["quality"]["flatline_duration_s"] = 0.0
        with self.assertRaisesRegex(ValueError, "flatline_duration_s"):
            validate_config_payload(invalid)

    def test_quality_weight_source_is_explicit_and_representation_compatible(self) -> None:
        self.assertEqual(
            {row["module_id"] for row in list_modules("quality_weight_source")},
            {"none", "route_file_q_rate", "legacy_window_sqi"},
        )
        raw = load_config(
            ROOT / "configs/reference_static_role_aware_v2.yaml"
        ).to_dict()
        raw["quality"]["window_selection"] = {
            "policy": "legacy_per_file_top_fraction",
            "keep_fraction": 0.7,
            "application_scope": "outer_train_only",
        }
        raw["aggregation"].update(
            {
                "quality_weighting": True,
                "quality_weight_source": "legacy_window_sqi",
            }
        )
        resolved = validate_config_payload(raw)
        self.assertEqual(
            resolved["aggregation"]["quality_weight_levels"],
            ["window_to_file", "file_to_role", "role_to_participant"],
        )
        self.assertEqual(
            resolved["aggregation"]["window_to_file"],
            "quality_weighted_mean",
        )
        self.assertEqual(
            resolved["aggregation"]["file_to_role"],
            "quality_weighted_mean",
        )
        self.assertEqual(
            resolved["aggregation"]["role_to_participant"],
            "quality_weighted_mean",
        )
        self.assertEqual(resolved["quality"]["mode"], "off")

        route = load_config(
            ROOT / "configs/reference_static_feature_vector_v2.yaml"
        ).to_dict()
        route["quality"]["mode"] = "route"
        route["aggregation"].update(
            {
                "quality_weighting": True,
                "quality_weight_source": "route_file_q_rate",
            }
        )
        resolved_route = validate_config_payload(route)
        self.assertEqual(
            resolved_route["aggregation"]["window_to_file"],
            "ordinary_mean",
        )
        self.assertEqual(
            resolved_route["aggregation"]["file_to_role"],
            "quality_weighted_mean",
        )
        self.assertEqual(
            resolved_route["aggregation"]["role_to_participant"],
            "quality_weighted_mean",
        )

        missing_scorer = load_config(
            ROOT / "configs/reference_static_role_aware_v2.yaml"
        ).to_dict()
        missing_scorer["aggregation"].update(
            {
                "quality_weighting": True,
                "quality_weight_source": "legacy_window_sqi",
            }
        )
        with self.assertRaisesRegex(ValueError, "window_selection.policy"):
            validate_config_payload(missing_scorer)

        fusion = load_config(
            ROOT / "configs/reference_static_fusion_v2.yaml"
        ).to_dict()
        fusion["quality"]["window_selection"] = {
            "policy": "legacy_per_file_top_fraction",
            "keep_fraction": 0.7,
        }
        fusion["aggregation"].update(
            {
                "quality_weighting": True,
                "quality_weight_source": "legacy_window_sqi",
            }
        )
        with self.assertRaisesRegex(ValueError, "fusion starts at file level"):
            validate_config_payload(fusion)

        fusion_selection_view = load_config(
            ROOT / "configs/reference_static_fusion_v2.yaml"
        ).to_dict()
        fusion_selection_view["quality"]["window_selection"] = {
            "policy": "legacy_per_file_top_fraction",
            "keep_fraction": 0.7,
            "application_scope": "legacy_train_and_aggregation",
        }
        with self.assertRaisesRegex(ValueError, "raw window-level OOF"):
            validate_config_payload(fusion_selection_view)

        inactive = load_config(
            ROOT / "configs/reference_static_role_aware_v2.yaml"
        ).to_dict()
        inactive["aggregation"]["quality_weight_source"] = "legacy_window_sqi"
        resolved_inactive = validate_config_payload(inactive)
        self.assertEqual(resolved_inactive["aggregation"]["quality_weight_source"], "none")

    def test_artifact_module_requires_an_executable_route_combination(self) -> None:
        payload = load_config(
            ROOT / "configs/reference_static_feature_vector_v2.yaml"
        ).to_dict()
        payload["artifact"]["degraded_policy"] = "invented"
        with self.assertRaisesRegex(ValueError, "degraded_policy must be"):
            validate_config_payload(payload)

        payload = load_config(
            ROOT / "configs/reference_static_feature_vector_v2.yaml"
        ).to_dict()
        payload["artifact"]["motion_detector_enabled"] = "true"
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            validate_config_payload(payload)
        payload["artifact"]["motion_detector_enabled"] = True
        payload["artifact"]["motion_detector"].update(
            {
                "evidence_path": "artifacts/example/motion_internal_evidence.json",
                "expected_evidence_sha256": "a" * 64,
            }
        )
        resolved_motion_only = validate_config_payload(payload)
        self.assertTrue(resolved_motion_only["artifact"]["motion_detector_enabled"])
        self.assertFalse(resolved_motion_only["artifact"]["denoiser_enabled"])

        payload = load_config(
            ROOT / "configs/reference_static_feature_vector_v2.yaml"
        ).to_dict()
        payload["artifact"].update(
            {
                "reducer": "spectral_mask",
                "reducer_version": "spectral_mask_v1",
                "denoiser_enabled": True,
                "degraded_policy": "denoise_then_extract_rate_features",
                "parameters": {},
            }
        )
        resolved_denoiser_only = validate_config_payload(payload)
        self.assertTrue(resolved_denoiser_only["artifact"]["denoiser_enabled"])
        self.assertFalse(resolved_denoiser_only["artifact"]["motion_detector_enabled"])
        diagnostics_recovery = json.loads(json.dumps(payload))
        diagnostics_recovery["quality"]["mode"] = "diagnostics_only"
        resolved_diagnostics_recovery = validate_config_payload(
            diagnostics_recovery
        )
        self.assertEqual(
            resolved_diagnostics_recovery["quality"]["calibrator"],
            "fixed_formula_thresholds_v1",
        )
        self.assertEqual(
            resolved_diagnostics_recovery["quality"]["high_quality_rule"],
            "direct_diagnostics_only_post_denoise_q_rate_fixed_formula_only",
        )
        payload["quality"]["mode"] = "route"
        payload["quality"]["rate_threshold"] = 0.5
        payload["quality"]["morph_threshold"] = 0.5
        validate_config_payload(payload)

        raw_nonidentity = load_config(
            ROOT / "configs/reference_static_role_aware_v2.yaml"
        ).to_dict()
        raw_nonidentity["quality"]["mode"] = "route"
        raw_nonidentity["artifact"].update(
            {
                "reducer": "pca_bss",
                "reducer_version": "pca_component_select_v2",
                "denoiser_enabled": True,
                "degraded_policy": "denoise_then_extract_rate_features",
                "parameters": {},
            }
        )
        with self.assertRaisesRegex(ValueError, "feature_vector"):
            validate_config_payload(raw_nonidentity)

    def test_sqi_motion_and_denoiser_switches_accept_all_combinations(self) -> None:
        for sqi_enabled in (False, True):
            for motion_enabled in (False, True):
                for denoiser_enabled in (False, True):
                    with self.subTest(
                        sqi=sqi_enabled,
                        motion=motion_enabled,
                        denoiser=denoiser_enabled,
                    ):
                        payload = load_config(
                            ROOT / "configs/reference_static_feature_vector_v2.yaml"
                        ).to_dict()
                        if sqi_enabled:
                            payload["quality"].update(
                                {
                                    "mode": "route",
                                    "rate_threshold": 0.50,
                                    "morph_threshold": 0.65,
                                }
                            )
                        payload["artifact"]["motion_detector_enabled"] = motion_enabled
                        if motion_enabled:
                            payload["artifact"]["motion_detector"].update(
                                {
                                    "evidence_path": "artifacts/example/motion_internal_evidence.json",
                                    "expected_evidence_sha256": "a" * 64,
                                }
                            )
                        payload["artifact"]["denoiser_enabled"] = denoiser_enabled
                        if denoiser_enabled:
                            payload["artifact"].update(
                                {
                                    "reducer": "pca_bss",
                                    "reducer_version": "pca_component_select_v2",
                                    "degraded_policy": "denoise_then_extract_rate_features",
                                    "parameters": {},
                                }
                            )
                        resolved = validate_config_payload(payload)
                        self.assertEqual(
                            resolved["quality"]["mode"] == "route", sqi_enabled
                        )
                        self.assertIs(
                            resolved["artifact"]["motion_detector_enabled"],
                            motion_enabled,
                        )
                        self.assertIs(
                            resolved["artifact"]["denoiser_enabled"],
                            denoiser_enabled,
                        )

    def test_mandatory_output_contract_is_not_exposed_as_fake_switches(self) -> None:
        reference = load_config(
            ROOT / "configs/reference_static_feature_vector_v2.yaml"
        ).to_dict()
        for field, value in (
            ("overwrite_existing", True),
            ("strict_json", False),
            ("write_parquet", False),
            ("parquet_missing_dependency_action", "skip"),
            ("write_window_oof", False),
            ("write_file_oof", False),
            ("write_subject_oof", False),
        ):
            with self.subTest(field=field):
                payload = json.loads(json.dumps(reference))
                payload["output"][field] = value
                with self.assertRaisesRegex(ValueError, "mandatory writer invariant"):
                    validate_config_payload(payload)

        member_toggle = json.loads(json.dumps(reference))
        member_toggle["output"]["write_member_oof"] = True
        with self.assertRaisesRegex(ValueError, "ensemble capability"):
            validate_config_payload(member_toggle)

    def test_motion_evidence_distinguishes_runtime_bundle_from_external_audits(self) -> None:
        evidence = list_modules("motion_evidence")
        self.assertEqual(
            {row["module_id"] for row in evidence},
            {
                "reused_frailty29_all29_bundle",
                "sqi_only",
                "sqi_plus_motion_override",
                "historical_light_cnn_backup",
            },
        )
        by_id = {row["module_id"]: row for row in evidence}
        self.assertEqual(
            set(by_id["reused_frailty29_all29_bundle"]["representation_modes"]),
            {"raw", "feature_vector", "feature_matrix", "fusion"},
        )
        self.assertIn(
            "in-sample auxiliary evidence",
            by_id["reused_frailty29_all29_bundle"]["notes"],
        )
        self.assertTrue(
            all(
                not by_id[module_id]["representation_modes"]
                for module_id in (
                    "sqi_only",
                    "sqi_plus_motion_override",
                    "historical_light_cnn_backup",
                )
            )
        )
        with self.assertRaisesRegex(ValueError, "unknown module family"):
            list_modules("motion_option")

    def test_estimator_backend_rejects_torch_only_training_switches(self) -> None:
        reference = load_config(
            ROOT / "configs/reference_static_feature_vector_v2.yaml"
        ).to_dict()
        changes = (
            {"fixed_epochs": 7},
            {"batch_size": 32},
            {"learning_rate": 0.002},
            {"weight_decay": 0.0},
            {"device": "cuda"},
            {"num_workers": 1},
            {"optimizer": "adamw"},
            {"optimizer_parameters": {"eps": 1e-7}},
            {"loss": "focal_loss"},
            {"loss": "focal_loss", "focal_gamma": 1.5},
            {"label_smoothing": 0.1},
            {"gradient_clip_norm": 1.0},
            {"deterministic_algorithms": False},
            {"samples_per_epoch": 100},
            {
                "epoch_rule": "inner_grouped_selection",
                "maximum_inner_epochs": 20,
                "inner_patience": 3,
                "inner_grouped_folds": 3,
            },
        )
        for update in changes:
            with self.subTest(update=update):
                payload = json.loads(json.dumps(reference))
                payload["training"].update(update)
                with self.assertRaisesRegex(
                    ValueError,
                    "execution_backend=estimator does not support",
                ):
                    validate_config_payload(payload)

        derived_only = json.loads(json.dumps(reference))
        derived_only["training"].update(
            {
                "execution_mode": "smoke",
                "epoch_profile": "unused_estimator_label",
            }
        )
        resolved = validate_config_payload(derived_only)
        self.assertEqual(resolved["training"]["execution_mode"], "formal")
        self.assertEqual(resolved["training"]["epoch_profile"], "default_10")

    def test_estimator_backend_keeps_shared_training_modules_configurable(self) -> None:
        payload = load_config(
            ROOT / "configs/reference_static_feature_vector_v2.yaml"
        ).to_dict()
        payload["training"].update(
            {
                "sampler": "class_subject_balanced",
                "participant_window_quota": "50%",
                "class_weighting": "effective_number",
                "class_weight_beta": 0.9,
                "training_balance": "equal_files",
                "classifier_role_families": ["B"],
                "seed": 7,
            }
        )
        resolved = validate_config_payload(payload)
        self.assertEqual(resolved["training"]["sampler"], "class_subject_balanced")
        self.assertEqual(resolved["training"]["participant_window_quota"], "50%")
        self.assertEqual(resolved["training"]["class_weighting"], "effective_number")
        self.assertEqual(resolved["training"]["class_weight_beta"], 0.9)
        self.assertEqual(resolved["training"]["training_balance"], "equal_files")
        self.assertEqual(resolved["training"]["classifier_role_families"], ["B"])
        self.assertEqual(resolved["training"]["seed"], 7)

        payload = load_config(
            ROOT / "configs/reference_static_feature_vector_v2.yaml"
        ).to_dict()
        payload["training"]["sampler"] = (
            "exhaustive_shuffle_without_replacement"
        )
        validate_config_payload(payload)

        payload["training"]["sampler"] = "uniform_replacement"
        with self.assertRaisesRegex(
            ValueError,
            "estimator does not support sampler=uniform_replacement",
        ):
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
            decision["schema_version"], "ppg_frailty.v2_decision_profile.v3"
        )
        self.assertEqual(
            decision["confirmed_defaults"]["quality"]["default_mode"],
            "off",
        )
        prv_contract = decision["confirmed_defaults"]["prv"]["comparison_contract"]
        self.assertEqual(prv_contract["input_unit"], "milliseconds")
        self.assertEqual(
            decision["confirmed_defaults"]["balance"][
                "training_and_reporting_selection"
            ],
            "independent",
        )
        self.assertIn("sqi_supervised_route", decision["deferred_evidence"])
        self.assertNotIn("deferred_gates", decision)
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
