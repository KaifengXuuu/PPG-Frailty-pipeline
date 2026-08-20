"""No-training contracts for the staged Static Line B study plans."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from ppg_frailty.study import (
    AxisSpec,
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
    def test_stage_01_has_four_canonical_representation_baselines(self) -> None:
        plan, expansion = _load("01_representation_baselines_v2.yaml")
        self.assertEqual(plan.catalog.scope, "selected_ordinary")
        self.assertEqual(
            {case.catalog_entry for case in expansion.cases},
            {
                "compact_cnn",
                "logistic_regression",
                "rocket_numpy",
                "fusion_compact",
            },
        )
        self.assertEqual(
            Counter(case.output_group for case in expansion.cases),
            {
                "raw": 1,
                "feature_vector": 1,
                "feature_matrix": 1,
                "fusion": 1,
            },
        )
        self.assertEqual(plan.execution.repeats, (0,))
        self.assertEqual(plan.execution.folds, (0, 1, 2, 3, 4))
        self.assertFalse(plan.execution.allow_parallel_deep)

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

    def test_stage_02_is_the_three_case_r0_supplement(self) -> None:
        plan, expansion = _load("02_competitive_routes_models_v2.yaml")
        self.assertEqual(plan.catalog.balance_line, "line_b")
        self.assertEqual(plan.catalog.scope, "selected_ordinary")
        self.assertEqual(len(expansion.cases), 3)
        self.assertEqual(
            Counter(case.output_group for case in expansion.cases),
            {"raw": 1, "feature_vector": 2},
        )
        entries = {str(case.catalog_entry) for case in expansion.cases}
        self.assertEqual(
            entries,
            {
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
            15,
        )
        self.assertTrue(
            all(case.screen_profile_id == "canonical" for case in plan.cases)
        )
        self.assertTrue(all(case.overrides == {} for case in plan.cases))
        self.assertTrue(all(case.formal_profile is None for case in plan.cases))
        self.assertEqual(
            {case.case_id for case in expansion.cases},
            {
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
                "raw__inception_small":
                    "formal_inception_small_line_b_v2__canonical",
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

    def test_matched_pair_scope_rejects_cross_representation_pair(self) -> None:
        plan = load_study_plan(
            PLAN_DIR / "04_selected_inception_ensemble_v2.yaml"
        )
        cases = list(plan.cases)
        cases[1] = replace(
            cases[1],
            catalog_entry="inception_matrix_five_member_ensemble",
            output_group="feature_matrix",
        )
        with self.assertRaisesRegex(ValueError, "one representation route"):
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
        with self.assertRaisesRegex(ValueError, "cannot add overrides"):
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

    def test_stage_05_contains_only_paired_off_and_diagnostic_modes(self) -> None:
        _plan, expansion = _load("05_sqi_motion_finalists_v2.yaml")
        by_entry: dict[str, list[object]] = defaultdict(list)
        for case in expansion.cases:
            by_entry[str(case.catalog_entry)].append(case)
            self.assertEqual(case.config["artifact"]["reducer"], "identity")
            self.assertFalse(case.config["artifact"]["motion_detector_enabled"])
        self.assertEqual(
            set(by_entry),
            {
                "compact_cnn",
                "logistic_regression",
                "rocket_numpy",
                "fusion_compact",
            },
        )
        for cases in by_entry.values():
            self.assertEqual(
                {case.config["quality"]["mode"] for case in cases},
                {"off", "diagnostics_only"},
            )

    def test_stage_06_has_one_learning_rate_axis_only(self) -> None:
        plan, expansion = _load(
            "06_sequential_single_factor_ablation_v2.yaml"
        )
        self.assertEqual(plan.study.kind, "ablation")
        self.assertEqual(len(plan.axes), 1)
        self.assertEqual(plan.axes[0].path, "training.learning_rate")
        self.assertEqual(plan.axes[0].values, (0.0003, 0.001, 0.003))
        self.assertEqual(len(expansion.cases), 3)
        self.assertEqual(
            {case.config["training"]["learning_rate"] for case in expansion.cases},
            {0.0003, 0.001, 0.003},
        )

    def test_stage_06_documented_deep_axes_remain_single_factor(self) -> None:
        plan = load_study_plan(
            PLAN_DIR / "06_sequential_single_factor_ablation_v2.yaml"
        )
        for path, values, reference in (
            ("training.batch_size", (32, 64, 128), 64),
            ("training.fixed_epochs", (7, 10, 15), 10),
        ):
            with self.subTest(path=path):
                changed = replace(
                    plan,
                    axes=(
                        AxisSpec(
                            path=path,
                            values=values,
                            reference=reference,
                        ),
                    ),
                )
                expansion = validate_canonical_expansion(
                    StudyRunner(pipeline_root=ROOT).expand(changed)
                )
                self.assertEqual(len(expansion.cases), 3)
                self.assertEqual(
                    {case.config[path.split(".")[0]][path.split(".")[1]]
                     for case in expansion.cases},
                    set(values),
                )

    def test_stage_06_classical_axis_updates_runtime_and_provenance(self) -> None:
        plan = load_study_plan(
            PLAN_DIR / "06_sequential_single_factor_ablation_v2.yaml"
        )
        logistic_plan = replace(
            plan,
            base_config="configs/reference_static_feature_vector_v2.yaml",
            axes=(
                AxisSpec(
                    path="model.logistic_c",
                    values=(0.1, 1.0, 10.0),
                    reference=1.0,
                ),
            ),
        )
        expansion = validate_canonical_expansion(
            StudyRunner(pipeline_root=ROOT).expand(logistic_plan)
        )
        for case in expansion.cases:
            model = case.config["model"]
            self.assertEqual(
                model["logistic_c"],
                model["architecture_parameters"]["C"],
            )

    def test_original_mega_study_remains_loadable_with_39_cases(self) -> None:
        plan = load_study_plan(ORIGINAL_PLAN)
        expansion = validate_canonical_expansion(
            StudyRunner(pipeline_root=ROOT).expand(plan)
        )
        self.assertEqual(len(expansion.cases), 39)
        self.assertEqual(plan.catalog.scope, "ordinary_13")


if __name__ == "__main__":
    unittest.main()
