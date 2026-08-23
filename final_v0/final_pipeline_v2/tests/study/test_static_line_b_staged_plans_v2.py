"""No-training contracts for the staged Static Line B study plans."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from ppg_frailty.study import (
    StudyRunner,
    load_study_plan,
    validate_canonical_expansion,
)


ROOT = Path(__file__).resolve().parents[2]
PLAN_DIR = ROOT / "configs" / "studies" / "static_line_b_staged_v2"
ORIGINAL_PLAN = ROOT / "configs" / "studies" / "static_line_b_all_models_v2.yaml"


def _load(name: str):
    plan = load_study_plan(PLAN_DIR / name)
    expansion = validate_canonical_expansion(
        StudyRunner(pipeline_root=ROOT).expand(plan)
    )
    return plan, expansion


class StaticLineBStagedPlanTests(unittest.TestCase):
    def test_non_bridge_same_model_drift_is_only_declared_experimental_factors(self) -> None:
        """Guard the cross-plan audit requested for the staged root YAML files."""

        _, stage1 = _load("01_representation_baselines_v2.yaml")
        _, stage4 = _load("04_selected_inception_ensemble_v2.yaml")
        _, stage5 = _load("05_sqi_motion_finalists_v2.yaml")
        stage6_plan, stage6 = _load(
            "06_sequential_single_factor_ablation_v2.yaml"
        )
        stage1_by_model = {
            case.config["model"]["model_id"]: case.config
            for case in stage1.cases
        }

        compact_reference = stage1_by_model["CompactCNN1D"]
        stage6_reference = next(
            case.config
            for case in stage6.cases
            if case.case_id == stage6_plan.study.reference_case_id
        )
        for case in stage6.cases:
            for section in (
                "manifest", "roles", "signal", "windows", "quality", "artifact",
                "features", "model", "aggregation", "evaluation",
            ):
                self.assertEqual(case.config[section], stage6_reference[section])
            changed_training = {
                key
                for key in set(case.config["training"]) | set(stage6_reference["training"])
                if case.config["training"].get(key)
                != stage6_reference["training"].get(key)
            }
            self.assertLessEqual(changed_training, {"learning_rate"})

        stage5_signal_reference = dict(compact_reference["signal"])
        stage5_signal_reference["dl_resampling"] = {
            **stage5_signal_reference["dl_resampling"],
            "enabled": True,
            "target_fs_hz": 64.0,
        }
        stage5_signal_reference["imu"] = {
            **stage5_signal_reference["imu"],
            "gravity_method": "calibrated_roll_pitch_ekf",
            "comparison_method": "profile_a_lowpass_0p3hz",
        }
        for name in ("gravity_lowpass_hz", "gravity_filter_order"):
            stage5_signal_reference["imu"].pop(name, None)
        for name, value in (
            ("process_covariance_diagonal_per_second", [5.0, 5.0, 0.05, 0.05, 0.05]),
            ("observation_covariance_diagonal_rad2", [0.5, 0.5]),
            ("initial_covariance_diagonal", [1.0, 1.0, 0.5, 0.5, 0.5]),
            ("dynamic_observation_scale", 3.0),
        ):
            stage5_signal_reference["imu"][name] = value
        stage5_reference = next(
            case.config
            for case in stage5.cases
            if case.case_id == "raw_compact_cnn__static_only"
        )
        for section in (
            "manifest", "windows", "features", "model", "aggregation",
            "evaluation",
        ):
            self.assertEqual(stage5_reference[section], compact_reference[section])
        self.assertEqual(stage5_reference["signal"], stage5_signal_reference)
        stage5_reference_training_changes = {
            key
            for key in set(stage5_reference["training"])
            | set(compact_reference["training"])
            if stage5_reference["training"].get(key)
            != compact_reference["training"].get(key)
        }
        self.assertEqual(
            stage5_reference_training_changes,
            {"optimizer", "batch_size"},
        )

        for case in stage5.cases:
            for section in (
                "manifest", "windows", "features", "model", "aggregation",
                "evaluation",
            ):
                self.assertEqual(case.config[section], stage5_reference[section])
            self.assertEqual(case.config["signal"], stage5_reference["signal"])
            changed_training = {
                key
                for key in set(case.config["training"])
                | set(stage5_reference["training"])
                if case.config["training"].get(key)
                != stage5_reference["training"].get(key)
            }
            self.assertLessEqual(
                changed_training, {"classifier_role_families"}
            )
            if case.case_id == "raw_compact_cnn__static_only":
                self.assertEqual(
                    case.config["roles"], stage5_reference["roles"]
                )
                self.assertFalse(changed_training)
            else:
                self.assertEqual(
                    case.config["roles"],
                    [
                        "B", "R1", "R2", "R3", "R4",
                        "S1", "S2", "W1", "W2",
                    ],
                )
                self.assertEqual(
                    case.config["training"]["classifier_role_families"],
                    ["B", "R", "S", "W"],
                )

        member0, ensemble = stage4.cases
        for section in (
            "manifest", "roles", "signal", "windows", "quality", "artifact",
            "features", "training", "aggregation", "evaluation",
        ):
            self.assertEqual(member0.config[section], ensemble.config[section])

    def test_stage_01_includes_small_variable_k_matrix_representative(self) -> None:
        plan, expansion = _load("01_representation_baselines_v2.yaml")
        self.assertEqual(plan.catalog.scope, "selected_ordinary")
        self.assertEqual(
            {case.catalog_entry for case in expansion.cases},
            {
                "compact_cnn",
                "logistic_regression",
                "inception_matrix_small",
                "inception_full",
            },
        )
        self.assertEqual(
            Counter(case.output_group for case in expansion.cases),
            {
                "raw": 2,
                "feature_vector": 1,
                "feature_matrix": 1,
            },
        )
        self.assertEqual(plan.study.reference_case_id, "raw__compact_cnn")
        self.assertEqual(plan.execution.repeats, (0, 1, 2, 3, 4))
        self.assertEqual(plan.execution.folds, (0, 1, 2, 3, 4))
        self.assertEqual(plan.execution.jobs, 1)
        self.assertFalse(plan.execution.allow_parallel_deep)
        matrix = next(
            case.config
            for case in expansion.cases
            if case.output_group == "feature_matrix"
        )
        self.assertEqual(matrix["model"]["variant"], "small")
        self.assertEqual(matrix["model"]["input_channels"], 146)
        self.assertEqual(matrix["model"]["ensemble_size"], 1)
        self.assertEqual(
            matrix["features"]["matrix_schema"],
            "ordered_window_feature_matrix_d146_variable_k_v1",
        )
        self.assertNotIn("matrix_k", matrix["features"])
        self.assertEqual(matrix["training"]["fixed_epochs"], 10)
        self.assertEqual(matrix["windows"]["engineering"]["length_s"], 10.0)
        self.assertEqual(matrix["windows"]["engineering"]["hop_s"], 2.0)
        self.assertEqual(
            matrix["windows"]["engineering"]["padding"],
            "none_complete_windows_only",
        )

        by_id = {case.case_id: case for case in expansion.cases}
        compact = by_id["raw__compact_cnn"].config
        self.assertEqual(compact["windows"]["raw_dl"]["length_s"], 5.0)
        self.assertEqual(compact["windows"]["raw_dl"]["hop_s"], 2.5)
        self.assertFalse(compact["signal"]["dl_resampling"]["enabled"])
        self.assertEqual(compact["training"]["fixed_epochs"], 10)
        self.assertEqual(compact["training"]["batch_size"], 64)
        self.assertEqual(compact["training"]["learning_rate"], 0.001)

        logistic = by_id["feature_vector__logistic"].config
        self.assertEqual(logistic["model"]["logistic_solver"], "lbfgs")
        self.assertEqual(logistic["model"]["logistic_max_iter"], 5000)
        self.assertEqual(logistic["training"]["device"], "cpu")

        configured = by_id["raw__inception_full_configured"].config
        self.assertEqual(configured["model"]["model_id"], "InceptionTimeFull")
        self.assertEqual(configured["model"]["input_channels"], 8)
        self.assertEqual(configured["model"]["dropout"], 0.5)
        self.assertTrue(configured["signal"]["dl_resampling"]["enabled"])
        self.assertEqual(
            configured["signal"]["dl_resampling"]["target_fs_hz"], 64.0
        )
        self.assertEqual(configured["windows"]["raw_dl"]["length_s"], 5.0)
        self.assertEqual(configured["windows"]["raw_dl"]["hop_s"], 2.5)
        self.assertEqual(configured["training"]["fixed_epochs"], 10)
        self.assertEqual(configured["training"]["batch_size"], 16)
        self.assertEqual(configured["training"]["learning_rate"], 0.0003)
        self.assertEqual(configured["training"]["weight_decay"], 0.005)
        self.assertEqual(configured["training"]["label_smoothing"], 0.2)

    def test_stage_01_jobs_override_is_reduced_for_deep_cases(self) -> None:
        plan = load_study_plan(
            PLAN_DIR / "01_representation_baselines_v2.yaml"
        )
        plan = replace(
            plan,
            execution=replace(plan.execution, jobs=3),
        )

        def fake_executor(case, config_path, case_directory, plan, progress_sink):
            del case, config_path, case_directory, plan, progress_sink
            return {"status": "passed", "cell_results": []}

        with tempfile.TemporaryDirectory() as temporary:
            result = StudyRunner(
                pipeline_root=ROOT,
                executor=fake_executor,
            ).run(plan, output_root=temporary)
        self.assertEqual(result.effective_jobs, 1)

    def test_stage_02_is_the_four_case_selected_state_r0_supplement(self) -> None:
        plan, expansion = _load("02_competitive_routes_models_v2.yaml")
        self.assertEqual(plan.catalog.balance_line, "line_b")
        self.assertEqual(plan.catalog.scope, "selected_ordinary")
        self.assertEqual(len(expansion.cases), 4)
        self.assertEqual(
            Counter(case.output_group for case in expansion.cases),
            {"raw": 2, "feature_vector": 2},
        )
        entries = {str(case.catalog_entry) for case in expansion.cases}
        self.assertEqual(
            entries,
            {
                "inception_full",
                "inception_small",
                "rbf_svm",
                "extra_trees",
            },
        )
        self.assertEqual(plan.execution.repeats, (0,))
        self.assertEqual(plan.execution.folds, (0, 1, 2, 3, 4))
        self.assertEqual(plan.execution.jobs, 1)
        self.assertFalse(plan.execution.allow_parallel_deep)
        self.assertEqual(
            len(expansion.cases)
            * len(plan.execution.repeats)
            * len(plan.execution.folds),
            20,
        )
        self.assertEqual(
            {
                case.case_id: case.screen_profile_id
                for case in plan.cases
            },
            {
                "raw__inception_full": "v2_core_b0_b2_b7_selected",
                "raw__inception_small": "v2_core_b0_b2_b7_selected",
                "feature_vector__rbf_svm": "canonical",
                "feature_vector__extra_trees": "canonical",
            },
        )
        self.assertTrue(all(case.formal_profile is None for case in plan.cases))
        self.assertEqual(
            {case.case_id for case in expansion.cases},
            {
                "raw__inception_full",
                "raw__inception_small",
                "feature_vector__rbf_svm",
                "feature_vector__extra_trees",
            },
        )
        self.assertEqual(
            {
                case.case_id: case.config["config_id"]
                for case in expansion.cases
            },
            {
                "raw__inception_full":
                    "formal_inception_full_line_b_v2__v2_core_b0_b2_b7_selected",
                "raw__inception_small":
                    "formal_inception_small_line_b_v2__v2_core_b0_b2_b7_selected",
                "feature_vector__rbf_svm":
                    "formal_rbf_svm_line_b_v2__canonical",
                "feature_vector__extra_trees":
                    "formal_extra_trees_line_b_v2__canonical",
            },
        )
        for case in expansion.cases:
            self.assertEqual(
                case.config["training"]["training_balance"],
                "equal_role_families",
            )
            self.assertEqual(
                case.config["aggregation"]["balance_line"],
                "line_b_equal_role_families",
            )
            self.assertEqual(case.config["quality"]["mode"], "off")
            self.assertEqual(case.config["artifact"]["reducer"], "identity")
            self.assertFalse(
                case.config["artifact"]["motion_detector_enabled"]
            )

    def test_stage_last_defaults_to_one_shapeformer_outer_cell(self) -> None:
        plan, expansion = _load("stage_last_shapeformer_stability_v2.yaml")
        self.assertEqual(len(expansion.cases), 1)
        self.assertEqual(
            plan.study.study_id,
            "staged_static_stage_last_shapeformer_stability_v2",
        )
        self.assertIn("Stage last", plan.study.flow_position)
        self.assertEqual(
            expansion.cases[0].catalog_entry,
            "shapeformer_channel_specific_osd",
        )
        self.assertEqual(plan.cases[0].screen_profile_id, "canonical")
        self.assertEqual(plan.cases[0].overrides, {})
        self.assertIsNone(plan.cases[0].formal_profile)
        self.assertEqual(plan.execution.repeats, (0,))
        self.assertEqual(plan.execution.folds, (0,))
        self.assertEqual(plan.execution.jobs, 1)

    def test_shapeformer_has_no_numbered_stage_03_plan(self) -> None:
        self.assertFalse((PLAN_DIR / "03_shapeformer_stability_v2.yaml").exists())

    def test_stage_04_is_one_exact_raw_matched_ensemble_pair(self) -> None:
        plan, expansion = _load("04_selected_inception_ensemble_v2.yaml")
        self.assertEqual(plan.catalog.scope, "matched_ensemble_pair")
        by_entry = {str(case.catalog_entry): case.config for case in expansion.cases}
        self.assertEqual(
            set(by_entry),
            {
                "inception_full_member0_comparator",
                "inception_full_five_member_ensemble",
            },
        )
        comparator = by_entry["inception_full_member0_comparator"]["model"]
        ensemble = by_entry["inception_full_five_member_ensemble"]["model"]
        self.assertEqual(comparator["ensemble_size"], 1)
        self.assertEqual(
            comparator["seed_policy"],
            "cv_fixed_member0_seed_50042_comparator",
        )
        self.assertEqual(ensemble["ensemble_size"], 5)
        self.assertEqual(
            ensemble["member_seeds"],
            [50042, 60042, 70042, 80042, 90042],
        )
        self.assertEqual(
            ensemble["architecture_parameters"]["probability_aggregation"],
            "arithmetic_mean",
        )

    def test_matched_pair_scope_rejects_retired_matrix_pair(self) -> None:
        plan = load_study_plan(
            PLAN_DIR / "04_selected_inception_ensemble_v2.yaml"
        )
        cases = list(plan.cases)
        cases[1] = replace(
            cases[1],
            catalog_entry="inception_matrix_five_member_ensemble",
            output_group="feature_matrix",
        )
        with self.assertRaisesRegex(ValueError, "unknown entries"):
            StudyRunner(pipeline_root=ROOT).expand(
                replace(plan, cases=tuple(cases))
            )

    def test_matched_pair_scope_rejects_extra_factor_override(self) -> None:
        plan = load_study_plan(
            PLAN_DIR / "04_selected_inception_ensemble_v2.yaml"
        )
        cases = list(plan.cases)
        cases[0] = replace(
            cases[0],
            overrides={"training.learning_rate": 0.0003},
        )
        with self.assertRaisesRegex(ValueError, "cannot add unequal"):
            replace(plan, cases=tuple(cases))

    def test_selected_ordinary_scope_rejects_ensemble_entry(self) -> None:
        plan = load_study_plan(
            PLAN_DIR / "01_representation_baselines_v2.yaml"
        )
        cases = list(plan.cases)
        cases[0] = replace(
            cases[0],
            catalog_entry="inception_full_five_member_ensemble",
        )
        with self.assertRaisesRegex(ValueError, "registered ordinary"):
            StudyRunner(pipeline_root=ROOT).expand(
                replace(plan, cases=tuple(cases))
            )

    def test_stage_05_is_the_eight_case_compact_cnn_route_screen(self) -> None:
        plan, expansion = _load("05_sqi_motion_finalists_v2.yaml")
        self.assertEqual(
            plan.study.reference_case_id,
            "raw_compact_cnn__off_off_all_roles",
        )
        self.assertEqual(plan.execution.repeats, (0,))
        self.assertEqual(plan.execution.folds, (0, 1, 2, 3, 4))
        self.assertEqual(plan.execution.device, "cuda")
        self.assertEqual(len(expansion.cases), 8)
        self.assertEqual(
            {str(case.catalog_entry) for case in expansion.cases},
            {"compact_cnn"},
        )
        self.assertEqual(
            {case.output_group for case in expansion.cases},
            {"raw"},
        )
        by_id = {case.case_id: case.config for case in expansion.cases}
        self.assertEqual(
            set(by_id),
            {
                "raw_compact_cnn__off_off_all_roles",
                "raw_compact_cnn__static_only",
                "raw_compact_cnn__sqi_only",
                "raw_compact_cnn__sqi_motion_matching_fold",
                "raw_compact_cnn__sqi_motion_pca",
                "raw_compact_cnn__sqi_motion_fastica",
                "raw_compact_cnn__sqi_off_motion_pca",
                "raw_compact_cnn__sqi_off_motion_fastica",
            },
        )
        off = by_id["raw_compact_cnn__off_off_all_roles"]
        self.assertEqual(off["model"]["model_id"], "CompactCNN1D")
        self.assertEqual(off["representation_mode"], "raw")
        self.assertEqual(off["windows"]["raw_dl"]["length_s"], 5.0)
        self.assertEqual(off["windows"]["raw_dl"]["hop_s"], 2.5)
        self.assertTrue(off["signal"]["dl_resampling"]["enabled"])
        self.assertEqual(off["signal"]["dl_resampling"]["target_fs_hz"], 64.0)
        self.assertEqual(off["signal"]["normalization"]["raw_ppg"], "per_window_robust")
        self.assertEqual(off["signal"]["normalization"]["raw_imu"], "none")
        self.assertEqual(off["training"]["device"], "cuda")
        self.assertEqual(off["quality"]["mode"], "off")
        self.assertFalse(off["artifact"]["motion_detector_enabled"])
        self.assertFalse(off["artifact"]["denoiser_enabled"])
        self.assertEqual(
            off["roles"],
            ["B", "R1", "R2", "R3", "R4", "S1", "S2", "W1", "W2"],
        )
        self.assertEqual(
            off["training"]["classifier_role_families"],
            ["B", "R", "S", "W"],
        )

        static_only = by_id["raw_compact_cnn__static_only"]
        self.assertEqual(static_only["roles"], ["B", "R1", "R2", "R3", "R4"])
        self.assertEqual(
            static_only["training"]["classifier_role_families"], ["B", "R"]
        )
        self.assertEqual(static_only["quality"]["mode"], "off")
        self.assertFalse(static_only["artifact"]["motion_detector_enabled"])
        self.assertFalse(static_only["artifact"]["denoiser_enabled"])

        sqi = by_id["raw_compact_cnn__sqi_only"]
        self.assertEqual(sqi["quality"]["calibrator"], "fixed_formula_thresholds_v1")
        self.assertEqual(sqi["quality"]["rate_threshold"], 0.50)
        self.assertEqual(sqi["quality"]["morph_threshold"], 0.65)
        self.assertEqual(sqi["quality"]["minimum_coverage"], 0.80)
        self.assertGreater(
            sqi["quality"]["rate_component_weights"]["motion_energy_rms"],
            0.0,
        )

        motion = by_id["raw_compact_cnn__sqi_motion_matching_fold"]["artifact"]
        self.assertTrue(motion["motion_detector_enabled"])
        self.assertFalse(motion["denoiser_enabled"])
        self.assertEqual(
            motion["motion_detector"]["expected_evidence_sha256"],
            "10f02a9d784e06471c7109ff8dc92d28f1a8d7753f8fdf179bebce5699fb446c",
        )
        self.assertEqual(
            motion["motion_detector"]["window_probability_aggregation"],
            "native_windows_file_median_diagnostics_only",
        )
        self.assertEqual(
            motion["motion_detector"]["reuse_scope"],
            "matching_outer_fold_or_all29_final",
        )
        self.assertEqual(
            motion["motion_detector"]["expected_split_registry_sha256"],
            "130b2887eb29a5a534397b4ce4dc7032f9de30ae46533fa0b2c41559ff4a1284",
        )
        self.assertEqual(motion["motion_detector"]["device"], "cuda")
        self.assertEqual(
            by_id["raw_compact_cnn__sqi_motion_pca"]["artifact"]["reducer"],
            "pca_bss",
        )
        self.assertEqual(
            by_id["raw_compact_cnn__sqi_motion_fastica"]["artifact"]["reducer"],
            "fastica_bss",
        )
        sqi_off_pca = by_id["raw_compact_cnn__sqi_off_motion_pca"]
        self.assertEqual(sqi_off_pca["quality"]["mode"], "off")
        self.assertTrue(sqi_off_pca["artifact"]["motion_detector_enabled"])
        self.assertTrue(sqi_off_pca["artifact"]["denoiser_enabled"])
        self.assertEqual(sqi_off_pca["artifact"]["reducer"], "pca_bss")
        self.assertEqual(sqi_off_pca["quality"]["rate_threshold"], 0.50)
        sqi_off_fastica = by_id["raw_compact_cnn__sqi_off_motion_fastica"]
        self.assertEqual(sqi_off_fastica["quality"]["mode"], "off")
        self.assertTrue(sqi_off_fastica["artifact"]["motion_detector_enabled"])
        self.assertTrue(sqi_off_fastica["artifact"]["denoiser_enabled"])
        self.assertEqual(sqi_off_fastica["artifact"]["reducer"], "fastica_bss")
        self.assertTrue(
            all(
                case.config["training"]["device"] == "cuda"
                and case.config["signal"]["imu"]["gravity_method"]
                == "calibrated_roll_pitch_ekf"
                for case in expansion.cases
            )
        )

    def test_catalog_new_leaf_override_still_rejects_unknown_parameters(self) -> None:
        plan = load_study_plan(PLAN_DIR / "05_sqi_motion_finalists_v2.yaml")
        cases = list(plan.cases)
        cases[0] = replace(
            cases[0],
            overrides={**cases[0].overrides, "quality.invented_parameter": 1},
        )
        with self.assertRaisesRegex(ValueError, "quality contains unknown fields"):
            StudyRunner(pipeline_root=ROOT).expand(
                replace(plan, cases=tuple(cases))
            )

    def test_catalog_rejects_unregistered_top_level_override(self) -> None:
        plan = load_study_plan(PLAN_DIR / "05_sqi_motion_finalists_v2.yaml")
        cases = list(plan.cases)
        with self.assertRaisesRegex(ValueError, "roles selector"):
            cases[0] = replace(
                cases[0],
                overrides={**cases[0].overrides, "invented_top_level": 1},
            )

    def test_retained_stage_06_changes_one_learning_rate_only(self) -> None:
        plan, expansion = _load(
            "06_sequential_single_factor_ablation_v2.yaml"
        )
        self.assertEqual(plan.study.kind, "catalog_sweep")
        self.assertEqual(plan.study.decision_role, "ablation")
        self.assertEqual(len(expansion.cases), 3)
        self.assertEqual(
            {case.config["training"]["learning_rate"] for case in expansion.cases},
            {0.0003, 0.001, 0.003},
        )

    def test_retained_stage_06_uses_selected_b0_b2_b7_controls(self) -> None:
        _, expansion = _load("06_sequential_single_factor_ablation_v2.yaml")
        for case in expansion.cases:
            self.assertEqual(case.config["training"]["optimizer"], "adamw")
            self.assertEqual(case.config["training"]["batch_size"], 32)
            self.assertEqual(
                case.config["training"]["sampler"],
                "exhaustive_shuffle_without_replacement",
            )
            self.assertEqual(
                case.config["training"]["class_weighting"],
                "inverse_frequency",
            )
            self.assertEqual(case.config["training"]["class_count_basis"], "row")
            self.assertEqual(case.config["signal"]["dl_resampling"]["target_fs_hz"], 64.0)
            self.assertEqual(case.config["windows"]["raw_dl"]["length_s"], 5.0)
            self.assertEqual(case.config["windows"]["raw_dl"]["hop_s"], 2.5)
            self.assertEqual(
                case.config["aggregation"]["balance_line"],
                "line_b_equal_role_families",
            )

    def test_original_mega_study_excludes_retired_rocket_profiles(self) -> None:
        plan = load_study_plan(ORIGINAL_PLAN)
        expansion = validate_canonical_expansion(
            StudyRunner(pipeline_root=ROOT).expand(plan)
        )
        self.assertEqual(len(expansion.cases), 31)
        self.assertEqual(plan.catalog.scope, "ordinary_active")


if __name__ == "__main__":
    unittest.main()
