"""No-training contracts for Stage 6 tuning and channel ablation."""

from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
from xml.etree import ElementTree
from zipfile import ZipFile

import yaml

from ppg_frailty.experiment import _bind_raw_dataset_for_model, _model_input_spec
from ppg_frailty.models import create_model
from ppg_frailty.study import StudyRunner, validate_canonical_expansion
from ppg_frailty.study.hyperparameter import (
    _candidate_component_rows,
    _completion_candidates,
    _copy_root_profile_tables,
    _design_scope_conclusion,
    _merge_equal_resource_rankings,
    _narrow_table_views,
    _nested_phase_frozen_membership,
    _phase_plan,
    _persisted_reporting_selection_seed,
    _root_ranking_selection_role,
    _write_root_report,
    complete_successive_halving_study,
    load_hyperparameter_plan,
)
from ppg_frailty.training import RawWindowDataset, SampleIdentity

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PLAN_DIR = ROOT / "configs" / "studies" / "static_line_b_staged_v2"


class HyperparameterStudyTests(unittest.TestCase):
    def test_hyperparameter_narrow_views_are_lossless_and_bounded(self) -> None:
        rows = [
            {
                "case_id": "candidate",
                "repeat": 0,
                **{f"metric_{index}": index for index in range(11)},
            }
        ]
        views = _narrow_table_views(
            rows,
            identity_fields=("case_id", "repeat"),
            semantic_groups=(
                (
                    "Primary metrics",
                    tuple(f"metric_{index}" for index in range(5)),
                ),
            ),
        )
        self.assertTrue(all(len(view[0]) <= 8 for _title, view in views))
        displayed_fields = {
            field for _title, view in views for row in view for field in row
        }
        self.assertEqual(displayed_fields, set(rows[0]))

    def _persist_expansion_contract(self, phase: Path, expansion) -> dict[str, object]:
        """Persist the same resolved-config/TEST_COMPONENTS contract as a phase."""

        cases = []
        configs: dict[str, object] = {}
        for case in expansion.cases:
            relative = Path("raw") / case.case_id / "resolved_config.yaml"
            target = phase / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            config = dict(case.config)
            target.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            configs[case.case_id] = config
            cases.append(
                {
                    "case_id": case.case_id,
                    "resolved_config_path": relative.as_posix(),
                }
            )
        manifest = {"status": "running", "cases": cases}
        (phase / "study_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        tables = phase / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        component_rows = []
        for case_id, raw_config in configs.items():
            config = dict(raw_config)
            model = dict(config["model"])
            training = dict(config["training"])
            signal = dict(config["signal"])
            windows = dict(config["windows"])
            input_data = {
                "channels": list(model["input_channel_order"]),
                "pipeline_fs_hz": float(signal["internal_fs_hz"]),
                "representation_mode": config["representation_mode"],
                "roles": list(config["roles"]),
                "signal_view": "x_dl_all8_window_norm",
                "window": dict(windows["raw_dl"]),
            }
            shared = {
                "participating_cases": case_id,
                "execution_state": "enabled",
                "algorithm_kernel_description": "persisted test fixture",
                "reporter_profile_id": "multiclass_participant_oof_v1",
                "model_reporter_extension_id": "inceptiontime_single_network_model_v1",
                "algorithm_references": "persisted fixture reference",
            }
            component_rows.extend(
                (
                    {
                        **shared,
                        "component_role": "classifier",
                        "module_id": model["model_id"],
                        "input_data": json.dumps(input_data, sort_keys=True),
                        "fixed_parameters": json.dumps(model, sort_keys=True),
                    },
                    {
                        **shared,
                        "component_role": "trainer",
                        "module_id": training["optimizer"],
                        "input_data": json.dumps(
                            {"model_input": input_data}, sort_keys=True
                        ),
                        "fixed_parameters": json.dumps(training, sort_keys=True),
                    },
                )
            )
        (tables / "test_components.json").write_text(
            json.dumps(component_rows),
            encoding="utf-8",
        )
        return configs

    def test_root_ranking_role_is_derived_from_declared_study_design(self) -> None:
        regularization = load_hyperparameter_plan(
            PLAN_DIR / "stage6_regula_search.yaml"
        )
        halving = load_hyperparameter_plan(
            PLAN_DIR / "stage6_batch_LR_search.yaml"
        )
        self.assertEqual(
            _root_ranking_selection_role(regularization, "full_cv_ranking"),
            "declared_full_cv_equal_weight_fold_cell_ranking",
        )
        self.assertEqual(
            _root_ranking_selection_role(halving, "screen_ranking"),
            "reduced_resource_screening_evidence_not_full_cv_selection",
        )
        self.assertEqual(
            _root_ranking_selection_role(
                halving, "all_candidates_full_cv_ranking"
            ),
            "exhaustive_full_grid_selection_evidence_after_completion",
        )

    def test_joint_grid_design_scope_forbids_single_factor_claims(self) -> None:
        regularization = load_hyperparameter_plan(
            PLAN_DIR / "stage6_regula_search.yaml"
        )
        batch_lr = load_hyperparameter_plan(
            PLAN_DIR / "stage6_batch_LR_search.yaml"
        )
        regularization_scope = _design_scope_conclusion(
            regularization, selected_case_id="r2_wd001_do05_ls02"
        )
        batch_lr_scope = _design_scope_conclusion(
            batch_lr, selected_case_id="b16_lr3e-4"
        )
        self.assertEqual(
            regularization_scope["confidence"],
            "design_scope_joint_profile_nonfactorial",
        )
        self.assertIn("training.weight_decay", regularization_scope["finding"])
        self.assertIn("model.dropout", regularization_scope["finding"])
        self.assertIn("training.label_smoothing", regularization_scope["finding"])
        self.assertIn("training.batch_size", batch_lr_scope["finding"])
        self.assertIn("training.learning_rate", batch_lr_scope["finding"])
        self.assertEqual(
            batch_lr_scope["selection_effect"],
            "profile_level_selection_only_no_single_factor_claim",
        )

    def test_reporting_seed_comes_from_persisted_phase_plan(self) -> None:
        plan = load_hyperparameter_plan(
            PLAN_DIR / "stage6_regula_search.yaml"
        )
        self.assertEqual(plan["search"]["selection_seed"], 42)
        plan["search"]["selection_seed"] = 777
        phase_plan = _phase_plan(
            plan,
            phase_id="seed_contract",
            candidates=plan["candidates"][:2],
            repeats=[0],
            folds=[0],
            epochs=1,
            inherited={},
            device="cuda",
            jobs=1,
        )
        self.assertEqual(phase_plan.search.selection_seed, 777)
        with tempfile.TemporaryDirectory() as temporary:
            phase = Path(temporary)
            (phase / "study_plan.yaml").write_text(
                yaml.safe_dump({"search": {"selection_seed": 777}}),
                encoding="utf-8",
            )
            self.assertEqual(
                _persisted_reporting_selection_seed(
                    (("full_cv", phase),), default=42
                ),
                777,
            )

    def test_profile_required_table_serializations_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "root"
            phase = Path(temporary) / "phase"
            (phase / "tables").mkdir(parents=True)
            (phase / "tables" / "case_summary.csv").write_text(
                "case_id\nexample\n", encoding="utf-8"
            )
            rows = _copy_root_profile_tables(
                output,
                (("full_cv", phase),),
                required_by_name={"case_summary": ("profile",)},
            )
        self.assertEqual(rows[0]["status"], "N/A_fail_closed")
        self.assertEqual(rows[0]["missing_suffixes"], [".json"])
        self.assertIn(".json", rows[0]["reason"])

    def _expand(self, name: str, inherited: dict[str, object]):
        plan = load_hyperparameter_plan(PLAN_DIR / name)
        resource = plan["resource"]
        successive = plan["study"]["study_type"] == "successive_halving"
        phase = _phase_plan(
            plan,
            phase_id="contract_test",
            candidates=plan["candidates"],
            repeats=(
                resource["screen_repeats"] if successive else resource["repeats"]
            ),
            folds=resource["screen_folds"] if successive else resource["folds"],
            epochs=resource["screen_epochs"] if successive else resource["epochs"],
            inherited=inherited,
            device="cuda",
            jobs=1,
        )
        return plan, validate_canonical_expansion(
            StudyRunner(pipeline_root=ROOT).expand(phase)
        )

    def test_batch_lr_halving_budget_and_grid(self) -> None:
        plan, expansion = self._expand("stage6_batch_LR_search.yaml", {})
        pairs = {
            (
                case.config["training"]["batch_size"],
                case.config["training"]["learning_rate"],
            )
            for case in expansion.cases
        }
        self.assertEqual(
            pairs,
            {
                (16, 0.0001), (16, 0.0003), (16, 0.001),
                (32, 0.0001), (32, 0.0003), (32, 0.001),
            },
        )
        self.assertTrue(
            all(
                case.config["training"]["sampler"]
                == "exhaustive_shuffle_without_replacement"
                and case.config["training"]["class_weighting"]
                == "inverse_frequency"
                and case.config["training"]["class_count_basis"] == "row"
                for case in expansion.cases
            )
        )
        resource = plan["resource"]
        screen_epochs = (
            len(plan["candidates"])
            * len(resource["screen_repeats"])
            * len(resource["screen_folds"])
            * resource["screen_epochs"]
        )
        promoted_epochs = (
            resource["promote_count"]
            * len(resource["promotion_repeats"])
            * len(resource["promotion_folds"])
            * resource["promotion_epochs"]
        )
        direct_epochs = (
            len(plan["candidates"])
            * len(resource["promotion_repeats"])
            * len(resource["promotion_folds"])
            * resource["promotion_epochs"]
        )
        self.assertEqual((screen_epochs, promoted_epochs), (150, 750))
        self.assertEqual(screen_epochs + promoted_epochs, 900)
        self.assertEqual(direct_epochs, 1500)

    def test_regularization_grid_inherits_batch_and_lr(self) -> None:
        _, expansion = self._expand(
            "stage6_regula_search.yaml",
            {"training.batch_size": 16, "training.learning_rate": 0.0003},
        )
        self.assertEqual(len(expansion.cases), 9)
        self.assertTrue(
            all(
                case.config["training"]["batch_size"] == 16
                and case.config["training"]["learning_rate"] == 0.0003
                for case in expansion.cases
            )
        )
        observed = {
            (
                case.config["training"]["weight_decay"],
                case.config["model"]["dropout"],
                case.config["training"]["label_smoothing"],
            )
            for case in expansion.cases
        }
        self.assertEqual(len(observed), 9)

    def test_halving_completion_selects_only_three_unpromoted_cases(self) -> None:
        plan = load_hyperparameter_plan(PLAN_DIR / "stage6_batch_LR_search.yaml")
        promoted = [
            {"case_id": "b16_lr1e-3"},
            {"case_id": "b32_lr3e-4"},
            {"case_id": "b16_lr3e-4"},
        ]
        remaining = _completion_candidates(plan, promoted)
        self.assertEqual(
            [row["case_id"] for row in remaining],
            ["b16_lr1e-4", "b32_lr1e-4", "b32_lr1e-3"],
        )

    def test_complete_grid_ranking_merges_disjoint_equal_resources(self) -> None:
        def row(case_id: str, balanced_accuracy: float, macro_f1: float):
            return {
                "case_id": case_id,
                "cell_count": 25,
                "balanced_accuracy_mean": balanced_accuracy,
                "balanced_accuracy_sd": 0.05,
                "balanced_accuracy_percent_mean_sd": "50.0 ± 5.0",
                "macro_f1_mean": macro_f1,
                "macro_f1_sd": 0.04,
                "macro_f1_percent_mean_sd": "50.0 ± 4.0",
                "rank": 1,
            }

        merged = _merge_equal_resource_rankings(
            (
                [row("a", 0.60, 0.50), row("b", 0.55, 0.60)],
                [row("c", 0.70, 0.40)],
            ),
            metric="balanced_accuracy",
            tie_break="macro_f1",
            expected_case_ids=["a", "b", "c"],
        )
        self.assertEqual([item["case_id"] for item in merged], ["c", "a", "b"])
        self.assertEqual([item["rank"] for item in merged], [1, 2, 3])
        self.assertTrue(
            all(
                item["selection_role"]
                == "exhaustive_full_grid_selection_evidence_after_completion"
                for item in merged
            )
        )

    def test_completion_reuses_promoted_and_updates_selection_after_success(self) -> None:
        _plan, expansion = self._expand("stage6_batch_LR_search.yaml", {})

        def row(case_id: str, balanced_accuracy: float, macro_f1: float):
            return {
                "case_id": case_id,
                "cell_count": 25,
                "balanced_accuracy_mean": balanced_accuracy,
                "balanced_accuracy_sd": 0.05,
                "balanced_accuracy_percent_mean_sd": "50.0 ± 5.0",
                "macro_f1_mean": macro_f1,
                "macro_f1_sd": 0.04,
                "macro_f1_percent_mean_sd": "50.0 ± 4.0",
                "rank": 1,
            }

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "halving"
            tables = output / "tables"
            screen_dir = output / "phases" / "screen" / "run"
            promotion_dir = output / "phases" / "promoted_full_cv" / "run"
            completion_dir = output / "phases" / "nonpromoted_full_cv" / "run"
            tables.mkdir(parents=True)
            screen_dir.mkdir(parents=True)
            promotion_dir.mkdir(parents=True)
            completion_dir.mkdir(parents=True)
            (output / "study_plan.yaml").write_text(
                (PLAN_DIR / "stage6_batch_LR_search.yaml").read_text(
                    encoding="utf-8"
                ),
                encoding="utf-8",
            )
            manifest = {
                "status": "passed",
                "device": "cuda",
                "jobs": 1,
                "phase_directories": {
                    "screen": "phases/screen/run",
                    "promotion": "phases/promoted_full_cv/run",
                },
                "ranking_tables": ["screen_ranking", "promotion_ranking"],
                "selected_case_id": "b16_lr3e-4",
            }
            (output / "study_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (output / "selected_configuration.json").write_text(
                json.dumps({"case_id": "b16_lr3e-4"}), encoding="utf-8"
            )
            promoted = [
                row("b16_lr3e-4", 0.61, 0.54),
                row("b32_lr3e-4", 0.56, 0.48),
                row("b16_lr1e-3", 0.55, 0.49),
            ]
            completed = [
                row("b16_lr1e-4", 0.50, 0.45),
                row("b32_lr1e-4", 0.52, 0.46),
                row("b32_lr1e-3", 0.70, 0.60),
            ]
            (tables / "screen_ranking.json").write_text(
                json.dumps(promoted), encoding="utf-8"
            )
            (tables / "promotion_ranking.json").write_text(
                json.dumps(promoted), encoding="utf-8"
            )
            self._persist_expansion_contract(screen_dir, expansion)
            self._persist_expansion_contract(promotion_dir, expansion)
            self._persist_expansion_contract(completion_dir, expansion)

            def write_case_summary(directory: Path, rows):
                (directory / "tables").mkdir(exist_ok=True)
                (directory / "tables" / "case_summary.json").write_text(
                    json.dumps([
                        {
                            "case_id": item["case_id"],
                            "participant_mean_balanced_accuracy": item[
                                "balanced_accuracy_mean"
                            ],
                            "repeat_balanced_accuracy_population_sd": item[
                                "balanced_accuracy_sd"
                            ],
                            "participant_mean_macro_f1": item["macro_f1_mean"],
                            "repeat_macro_f1_population_sd": item["macro_f1_sd"],
                            "participant_mean_macro_roc_auc_ovr": 0.70,
                            "participant_mean_macro_pr_auc_ovr": 0.65,
                            "expected_calibration_error": 0.15,
                            "worst_fold_balanced_accuracy": 0.33,
                            "worst_class_f1": 0.45,
                            "balanced_accuracy_lcb95": 0.40,
                            "fold_cell_count": item["cell_count"],
                        }
                        for item in rows
                    ]),
                    encoding="utf-8",
                )

            write_case_summary(screen_dir, promoted)
            write_case_summary(promotion_dir, promoted)
            write_case_summary(completion_dir, completed)
            resolved_path = completion_dir / "resolved_config.yaml"
            resolved_path.write_text("model: {}\n", encoding="utf-8")
            with (
                patch(
                    "ppg_frailty.study.hyperparameter._run_phase",
                    return_value=(None, completion_dir, completed),
                ) as run_phase,
                patch(
                    "ppg_frailty.study.hyperparameter._resolved_config",
                    return_value=(resolved_path, {"model": {}}),
                ),
            ):
                result = complete_successive_halving_study(
                    output, pipeline_root=ROOT, device="cuda", jobs=1
                )
            self.assertEqual(result, output)
            trained = run_phase.call_args.kwargs["candidates"]
            self.assertEqual(
                [candidate["case_id"] for candidate in trained],
                ["b16_lr1e-4", "b32_lr1e-4", "b32_lr1e-3"],
            )
            self.assertEqual(run_phase.call_args.kwargs["epochs"], 10)
            self.assertEqual(run_phase.call_args.kwargs["folds"], [0, 1, 2, 3, 4])
            updated_manifest = json.loads(
                (output / "study_manifest.json").read_text(encoding="utf-8")
            )
            updated_selection = json.loads(
                (output / "selected_configuration.json").read_text(encoding="utf-8")
            )
            self.assertEqual(updated_selection["case_id"], "b32_lr1e-3")
            self.assertEqual(updated_manifest["selected_case_id"], "b32_lr1e-3")
            self.assertEqual(
                updated_manifest["successive_halving_completion"][
                    "full_cv_fold_cell_count"
                ],
                150,
            )
            self.assertTrue(
                (output / "precompletion_selected_configuration.json").is_file()
            )
            self.assertTrue(
                (tables / "all_candidates_full_cv_ranking.csv").is_file()
            )
            self.assertTrue(
                (tables / "all_candidates_full_cv_participant_oof_ranking.csv").is_file()
            )
            summary = (output / "STUDY_SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn("all_candidates_full_cv_ranking", summary)
            self.assertIn("nonpromoted_full_cv", summary)
            self.assertTrue(
                (output / "result_backup" / "precompletion_study_manifest.json").is_file()
            )

    def test_channel_ablation_is_explicit_dl_tensor_slicing(self) -> None:
        inherited = {
            "training.batch_size": 32,
            "training.learning_rate": 0.0003,
            "training.weight_decay": 0.001,
            "training.label_smoothing": 0.1,
            "model.dropout": 0.3,
        }
        plan, expansion = self._expand(
            "stage_ablation_channels.yaml",
            inherited,
        )
        orders = {
            case.case_id: tuple(case.config["model"]["input_channel_order"])
            for case in expansion.cases
        }
        self.assertEqual(orders["channels_ppg_red_ir"], ("RED", "IR"))
        self.assertEqual(
            orders["channels_imu_acc_gyro"],
            ("A_dyn_x", "A_dyn_y", "A_dyn_z", "GX", "GY", "GZ"),
        )
        identities = (
            SampleIdentity("p0", "f0", "B", 0, "direct", window_id="w0"),
            SampleIdentity("p1", "f1", "R1", 1, "direct", window_id="w1"),
        )
        values = np.arange(2 * 8 * 32, dtype=np.float32).reshape(2, 8, 32)
        dataset = RawWindowDataset(values, identities)
        bound, provenance = _bind_raw_dataset_for_model(
            dataset,
            "inception_full",
            declared_channel_order=("RED", "IR"),
        )
        self.assertEqual(bound.values.shape, (2, 2, 32))
        self.assertEqual(bound.channel_schema, ("RED", "IR"))
        self.assertEqual(provenance["source_indices"], (0, 1))
        self.assertFalse(provenance["silent_channel_slicing"])
        spec = _model_input_spec(bound, "raw")
        selected = next(
            case.config["model"]
            for case in expansion.cases
            if case.case_id == "channels_ppg_red_ir"
        )
        model = create_model(
            {
                "model_id": "inception_full",
                "seed": 42,
                "dropout": selected["dropout"],
                "kernel_sizes": selected["kernel_sizes"],
                "dilation": selected["dilation"],
            },
            spec,
        )
        self.assertEqual(model.n_channels, 2)
        with tempfile.TemporaryDirectory() as temporary:
            phase = Path(temporary) / "full_cv"
            phase.mkdir(parents=True)
            self._persist_expansion_contract(phase, expansion)
            component_rows = _candidate_component_rows(
                plan,
                {"full_cv": phase},
                inherited=inherited,
            )
            component_path = phase / "tables" / "test_components.json"
            tampered = json.loads(component_path.read_text(encoding="utf-8"))
            classifier = next(
                row
                for row in tampered
                if row["component_role"] == "classifier"
                and row["participating_cases"] == "channels_ppg_red_ir"
            )
            classifier_input = json.loads(classifier["input_data"])
            classifier_input["channels"] = ["RED", "IR", "GX"]
            classifier["input_data"] = json.dumps(classifier_input, sort_keys=True)
            component_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "channel provenance differs"):
                _candidate_component_rows(
                    plan,
                    {"full_cv": phase},
                    inherited=inherited,
                )
        by_case = {row["participating_cases"]: row for row in component_rows}
        expected = {
            "channels_full8_reference": (
                "RED", "IR", "A_dyn_x", "A_dyn_y", "A_dyn_z", "GX", "GY", "GZ",
            ),
            "channels_ppg_red_ir": ("RED", "IR"),
            "channels_imu_acc_gyro": (
                "A_dyn_x", "A_dyn_y", "A_dyn_z", "GX", "GY", "GZ",
            ),
        }
        self.assertEqual(set(by_case), set(expected))
        resolved_paths = set()
        for case_id, channel_order in expected.items():
            input_data = json.loads(by_case[case_id]["input_data"])
            fixed = json.loads(by_case[case_id]["fixed_parameters"])
            self.assertEqual(tuple(input_data["channels"]), channel_order)
            self.assertEqual(input_data["signal_view"], "x_dl_all8_window_norm")
            self.assertEqual(input_data["sampling_rate_hz"], 64.0)
            self.assertEqual(input_data["pipeline_fs_hz"], 400.0)
            self.assertEqual(input_data["window"]["length_s"], 5.0)
            self.assertEqual(input_data["window"]["hop_s"], 2.5)
            self.assertEqual(fixed["model"]["input_channels"], len(channel_order))
            self.assertEqual(
                tuple(fixed["model"]["input_channel_order"]), channel_order
            )
            self.assertEqual(fixed["training"]["batch_size"], 32)
            self.assertEqual(fixed["persisted_provenance"]["phase"], "full_cv")
            resolved_paths.add(
                fixed["persisted_provenance"]["resolved_config_path"]
            )
        self.assertEqual(len(resolved_paths), 3)

    def test_root_report_uses_latest_portable_outputs(self) -> None:
        plan, expansion = self._expand("stage6_batch_LR_search.yaml", {})
        ranking = [
            {
                "case_id": "b16_lr1e-4",
                "cell_count": 5,
                "balanced_accuracy_mean": 0.726,
                "balanced_accuracy_sd": 0.060,
                "balanced_accuracy_percent_mean_sd": "72.6 ± 6.0",
                "macro_f1_mean": 0.701,
                "macro_f1_sd": 0.050,
                "macro_f1_percent_mean_sd": "70.1 ± 5.0",
                "rank": 1,
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            phase = output / "phases/screen/run"
            phase.mkdir(parents=True)
            (phase / "STUDY_SUMMARY.md").write_text("# nested\n", encoding="utf-8")
            configs = self._persist_expansion_contract(phase, expansion)
            (phase / "tables/case_summary.json").write_text(
                json.dumps(
                    [
                        {
                            "case_id": "b16_lr1e-4",
                            "participant_mean_balanced_accuracy": 0.70,
                            "repeat_balanced_accuracy_population_sd": 0.04,
                            "participant_mean_macro_f1": 0.68,
                            "repeat_macro_f1_population_sd": 0.05,
                            "participant_mean_macro_roc_auc_ovr": 0.80,
                            "participant_mean_macro_pr_auc_ovr": 0.75,
                            "expected_calibration_error": 0.10,
                            "worst_fold_balanced_accuracy": 0.40,
                            "worst_class_f1": 0.50,
                            "balanced_accuracy_lcb95": 0.60,
                            "fold_cell_count": 5,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            _write_root_report(
                output,
                plan=plan,
                phase_directories={"screen": phase},
                ranking_tables={"screen_ranking": ranking},
                selected={
                    "case_id": "b16_lr1e-4",
                    "resolved_config": configs["b16_lr1e-4"],
                },
                inherited={},
            )
            required = (
                "STUDY_SUMMARY.md", "STUDY_SUMMARY.html", "TEST_COMPONENTS.md",
                "REPORT_METHODS.md", "RESULT_INTERPRETATION.md",
                "outputs_index.json", "tables/screen_ranking.csv",
                "tables/screen_ranking.json", "tables/reproducibility.csv",
                "tables/test_components.csv", "tables/table_figure_pairs.csv",
                "tables/report_tables.xlsx",
                "tables/screen_participant_oof_ranking.csv",
                "tables/reporter_profiles.csv",
                "tables/comprehensive_model_comparison.csv",
                "tables/model_comparison_performance.csv",
                "tables/model_comparison_uncertainty.csv",
                "tables/model_comparison_inference.csv",
                "tables/model_comparison_robustness.csv",
                "tables/selection_conclusions.csv",
                "tables/root_reporter_artifact_status.csv",
            )
            self.assertTrue(all((output / name).is_file() for name in required))
            summary = (output / "STUDY_SUMMARY.md").read_text(encoding="utf-8")
            interpretation = (output / "RESULT_INTERPRETATION.md").read_text(
                encoding="utf-8"
            )
            components = (output / "TEST_COMPONENTS.md").read_text(encoding="utf-8")
            self.assertIn("72.6 ± 6.0", summary)
            self.assertIn("outer_cv_repeat_seed_equals_split_seed", summary)
            self.assertIn("40042", summary)
            self.assertIn("equal-weight mean of declared fold-cell", summary)
            self.assertIn("Model/module-owned reporter methods and literature", summary)
            self.assertIn("P values are null-hypothesis tail probabilities", summary)
            self.assertIn("### Ranking and performance", summary)
            self.assertIn(
                "### Uncertainty and 95% confidence intervals", summary
            )
            self.assertIn("### Ranking and performance", interpretation)
            self.assertIn("### Paired inference", interpretation)
            self.assertNotIn(
                "| rank | case_id | status | complete_for_requested_execution |",
                summary,
            )
            self.assertIn("profile_a_lowpass_0p3hz", components)
            self.assertIn(components.split("\n\n", 1)[1].strip(), summary)
            self.assertIn(
                "### Profile identity and participating components", summary
            )
            self.assertIn("### Required outputs", summary)
            self.assertIn("### Methods, limitations, and provenance", summary)
            markdown_lines = summary.splitlines()
            markdown_table_widths = [
                markdown_lines[index + 1].count("|") - 1
                for index, line in enumerate(markdown_lines[:-1])
                if line.startswith("| ")
                and markdown_lines[index + 1].startswith("|---")
            ]
            self.assertTrue(markdown_table_widths)
            self.assertTrue(
                all(width <= 8 for width in markdown_table_widths),
                markdown_table_widths,
            )
            html_summary = (output / "STUDY_SUMMARY.html").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "<h3>Profile identity and participating components</h3>",
                html_summary,
            )
            self.assertIn("<h3>Input data and fixed parameters</h3>", html_summary)
            html_table_widths = [
                len(re.findall(r"<th(?:\s|>)", table))
                for table in re.findall(
                    r"<table(?:\s[^>]*)?>.*?</table>",
                    html_summary,
                    flags=re.DOTALL,
                )
            ]
            self.assertTrue(html_table_widths)
            self.assertTrue(
                all(width <= 8 for width in html_table_widths),
                html_table_widths,
            )
            candidate_rows = json.loads(
                (output / "tables/test_components.json").read_text(
                    encoding="utf-8"
                )
            )
            persisted_batch_lr = {
                (
                    json.loads(row["fixed_parameters"])["training"]["batch_size"],
                    json.loads(row["fixed_parameters"])["training"][
                        "learning_rate"
                    ],
                )
                for row in candidate_rows
            }
            self.assertEqual(
                persisted_batch_lr,
                {
                    (16, 0.0001), (16, 0.0003), (16, 0.001),
                    (32, 0.0001), (32, 0.0003), (32, 0.001),
                },
            )
            reporter_profile_rows = json.loads(
                (output / "tables/reporter_profiles.json").read_text(
                    encoding="utf-8"
                )
            )
            per_class_rows = json.loads(
                (output / "tables/classifier_per_class_results.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertGreater(len(candidate_rows[0]), 8)
            self.assertGreater(len(reporter_profile_rows[0]), 8)
            self.assertGreater(len(per_class_rows[0]), 8)
            artifact_status = json.loads(
                (output / "tables/root_reporter_artifact_status.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(artifact_status)
            self.assertTrue(
                all(
                    row["status"] in {
                        "copied_from_nested_phase",
                        "generated_from_nested_phase",
                        "generated_but_unpaired_fail_closed",
                        "N/A_fail_closed",
                    }
                    for row in artifact_status
                )
            )
            for table_name in (
                "model_comparison_performance",
                "model_comparison_uncertainty",
                "model_comparison_inference",
                "model_comparison_robustness",
            ):
                header = (
                    output / "tables" / f"{table_name}.csv"
                ).read_text(encoding="utf-8").splitlines()[0]
                self.assertLessEqual(len(header.split(",")), 8)
            raw_comparison = json.loads(
                (
                    output / "tables/comprehensive_model_comparison.json"
                ).read_text(encoding="utf-8")
            )
            self.assertIsInstance(raw_comparison, list)
            if raw_comparison:
                self.assertIn("balanced_accuracy_mean", raw_comparison[0])
            inference_rows = json.loads(
                (
                    output
                    / "tables/exploratory_selected_paired_inference.json"
                ).read_text(encoding="utf-8")
            )
            repeat_delta_rows = json.loads(
                (output / "tables/pairwise_repeat_metric_deltas.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(inference_rows), 15)
            self.assertEqual(len(repeat_delta_rows), 25)
            self.assertTrue(
                all(
                    row["comparison_contract_status"]
                    == "N/A_frozen_split_registry_no_full_resource_phase"
                    and row["candidate_minus_reference"] is None
                    for row in inference_rows
                )
            )
            self.assertTrue(
                all(
                    row["comparison_contract_status"]
                    == "N/A_frozen_split_registry_no_full_resource_phase"
                    and row["comparison_role"]
                    == "exploratory_post_selection_model_comparison"
                    and row["balanced_accuracy_delta"] is None
                    and row["macro_f1_delta"] is None
                    and row["macro_roc_auc_ovr_delta"] is None
                    for row in repeat_delta_rows
                )
            )
            with ZipFile(output / "tables/report_tables.xlsx") as archive:
                self.assertIsNone(archive.testzip())
                workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
            namespace = {
                "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
            }
            sheet_count = len(workbook.findall(".//m:sheet", namespace))
            csv_count = len(tuple((output / "tables").glob("*.csv")))
            self.assertEqual(sheet_count, csv_count)
            index = json.loads((output / "outputs_index.json").read_text())
            self.assertTrue(index)
            self.assertTrue(all(len(row["sha256"]) == 64 for row in index))

    def test_nested_full_resource_registry_must_pass_and_match_across_phases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def persist(phase: str, *, fold_for_p2: int = 1, status: str = "PASS") -> Path:
                directory = root / phase
                tables = directory / "tables"
                tables.mkdir(parents=True)
                (tables / "reproducibility_summary.json").write_text(
                    json.dumps([{"audit_status": status}]), encoding="utf-8"
                )
                (tables / "reproducibility_splits.json").write_text(
                    json.dumps(
                        [
                            {
                                "audit_status": status,
                                "repeat": 0,
                                "fold": 0,
                                "split_seed": 42,
                                "oof_participant_count": 1,
                                "oof_participant_ids": ["P1"],
                                "train_oof_overlap_count": 0,
                            },
                            {
                                "audit_status": status,
                                "repeat": 0,
                                "fold": fold_for_p2,
                                "split_seed": 42,
                                "oof_participant_count": 1,
                                "oof_participant_ids": ["P2"],
                                "train_oof_overlap_count": 0,
                            },
                        ]
                    ),
                    encoding="utf-8",
                )
                return directory

            promotion = persist("promotion")
            completion = persist("completion")
            membership, reason = _nested_phase_frozen_membership(
                (("promotion", promotion), ("completion", completion))
            )
            self.assertIsNone(reason)
            self.assertEqual(
                membership,
                {("P1", 0): (0, 42), ("P2", 0): (1, 42)},
            )

            mismatched = persist("mismatched", fold_for_p2=2)
            membership, reason = _nested_phase_frozen_membership(
                (("promotion", promotion), ("completion", mismatched))
            )
            self.assertIsNone(membership)
            self.assertEqual(
                reason,
                "frozen_split_registry_cross_phase_roster_mismatch",
            )

            failed = persist("failed", status="FAIL")
            membership, reason = _nested_phase_frozen_membership(
                (("full_cv", failed),)
            )
            self.assertIsNone(membership)
            self.assertEqual(
                reason,
                "frozen_split_registry_not_verifiable__full_cv",
            )


if __name__ == "__main__":
    unittest.main()
