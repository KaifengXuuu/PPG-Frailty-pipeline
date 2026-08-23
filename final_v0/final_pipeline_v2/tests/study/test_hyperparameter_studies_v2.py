"""No-training contracts for Stage 6 tuning and channel ablation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from xml.etree import ElementTree
from zipfile import ZipFile

from ppg_frailty.experiment import _bind_raw_dataset_for_model, _model_input_spec
from ppg_frailty.models import create_model
from ppg_frailty.study import StudyRunner, validate_canonical_expansion
from ppg_frailty.study.hyperparameter import (
    _completion_candidates,
    _merge_equal_resource_rankings,
    _phase_plan,
    _write_root_report,
    complete_successive_halving_study,
    load_hyperparameter_plan,
)
from ppg_frailty.training import RawWindowDataset, SampleIdentity

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PLAN_DIR = ROOT / "configs" / "studies" / "static_line_b_staged_v2"


class HyperparameterStudyTests(unittest.TestCase):
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
            def write_case_summary(directory: Path, rows):
                (directory / "tables").mkdir()
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
        _, expansion = self._expand(
            "stage_ablation_channels.yaml",
            {
                "training.batch_size": 32,
                "training.learning_rate": 0.0003,
                "training.weight_decay": 0.001,
                "training.label_smoothing": 0.1,
                "model.dropout": 0.3,
            },
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

    def test_root_report_uses_latest_portable_outputs(self) -> None:
        plan = load_hyperparameter_plan(PLAN_DIR / "stage6_batch_LR_search.yaml")
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
            (phase / "tables").mkdir()
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
                    "resolved_config": {
                        "signal": {
                            "imu": {
                                "gravity_method": "profile_a_lowpass_0p3hz"
                            }
                        },
                        "model": {
                            "input_channel_order": [
                                "RED", "IR", "A_dyn_x", "A_dyn_y", "A_dyn_z",
                                "GX", "GY", "GZ",
                            ]
                        },
                    },
                },
                inherited={},
            )
            required = (
                "STUDY_SUMMARY.md", "STUDY_SUMMARY.html", "TEST_COMPONENTS.md",
                "outputs_index.json", "tables/screen_ranking.csv",
                "tables/screen_ranking.json", "tables/reproducibility.csv",
                "tables/test_components.csv", "tables/table_figure_pairs.csv",
                "tables/report_tables.xlsx",
                "tables/screen_participant_oof_ranking.csv",
            )
            self.assertTrue(all((output / name).is_file() for name in required))
            summary = (output / "STUDY_SUMMARY.md").read_text(encoding="utf-8")
            components = (output / "TEST_COMPONENTS.md").read_text(encoding="utf-8")
            self.assertIn("72.6 ± 6.0", summary)
            self.assertIn("outer_cv_repeat_seed_equals_split_seed", summary)
            self.assertIn("40042", summary)
            self.assertIn("equal-weight mean of declared fold-cell", summary)
            self.assertIn("profile_a_lowpass_0p3hz", components)
            self.assertIn(components.split("\n\n", 1)[1].strip(), summary)
            with ZipFile(output / "tables/report_tables.xlsx") as archive:
                self.assertIsNone(archive.testzip())
                workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
            namespace = {
                "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
            }
            self.assertEqual(len(workbook.findall(".//m:sheet", namespace)), 5)
            index = json.loads((output / "outputs_index.json").read_text())
            self.assertTrue(index)
            self.assertTrue(all(len(row["sha256"]) == 64 for row in index))


if __name__ == "__main__":
    unittest.main()
