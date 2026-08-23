"""Synthetic, no-training checks for the two-model Stage-3 centered star."""

from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path

from ppg_frailty.reporting.analyze import (
    _stage3_star_presentation_tables,
    _stage3_star_report_tables,
    analyze_study,
)
from ppg_frailty.provenance import stable_payload_sha256
from ppg_frailty.reporting.collect import CollectedStudy
from ppg_frailty.reporting.plots import (
    _stage3_star_fold_delta_heatmap,
    _stage3_star_model_deltas,
)
from ppg_frailty.reporting.report import generate_study_report
from ppg_frailty.study import load_study_plan
from ppg_frailty.training.aggregation import (
    LINE_A_EQUAL_FILES,
    LINE_B_EQUAL_ROLE_FAMILIES,
)


W = "window_balanced_to_participant"
MODELS = ("CompactCNN1D", "InceptionTimeFull")
CHANGES = {
    1: (("target_fs_hz", 400),),
    2: (("window_seconds", 5.0), ("hop_seconds", 2.5)),
    3: (("imu_preprocessing", "calibrated_ekf_adyn"),),
    4: (("normalization", "ppg_window_imu_outer_train_fold"),),
    5: (("sampler", "balance_line_weighted_v2"),),
    6: (("optimizer", "adam"), ("batch_size", 64)),
    7: (("primary_report_aggregation_view", LINE_B_EQUAL_ROLE_FAMILIES),),
}


def _fixture(
    repeat_indices: tuple[int, ...] = (0,),
) -> tuple[CollectedStudy, list[dict], list[dict], list[dict]]:
    profiles, execution = [], []
    baseline = {
        "target_fs_hz": 64,
        "window_seconds": 15.0,
        "hop_seconds": 3.0,
        "imu_preprocessing": "legacy_filtered_axes",
        "normalization": "per_window_all_eight",
        "sampler": "exhaustive_shuffle_without_replacement",
        "optimizer": "adamw",
        "batch_size": 32,
        "primary_report_aggregation_view": W,
        "training_metric_aggregation_rule": LINE_A_EQUAL_FILES,
    }
    for level in range(8):
        for model in MODELS:
            stem = "compact" if model == MODELS[0] else "inception"
            display = f"{stem}__B{level}"
            controls = dict(baseline)
            if level:
                for path, value in CHANGES[level]:
                    controls[path] = value
            profiles.append(
                {
                    "case_id": display,
                    "catalog_case_id": display.lower(),
                    "model_id": model,
                    "profile_id": f"B{level}",
                    "factor_id": f"factor_{level}",
                    "reference_case_id": None if level == 0 else f"{stem}__B0",
                    "changed_control_paths": (
                        [] if level == 0 else [path for path, _ in CHANGES[level]]
                    ),
                    "controls": controls,
                    "controls_sha256": stable_payload_sha256(controls),
                    "interpretation": "synthetic",
                }
            )
            execution.append(display)
    plan = {
        "legacy_bridge": {
            "design": "centered_star_v1",
            "profiles": profiles,
            "execution_order": execution,
            "centered_comparisons": [
                {
                    "model_id": row["model_id"],
                    "reference_case_id": row["reference_case_id"],
                    "variant_case_id": row["case_id"],
                    "profile_id": row["profile_id"],
                    "factor_id": row["factor_id"],
                    "changed_control_paths": row["changed_control_paths"],
                }
                for row in profiles
                if row["reference_case_id"] is not None
            ],
            "budget": {
                "repeat_indices": list(repeat_indices),
                "fold_indices": list(range(5)),
            },
        },
        "report": {"write_static_figures": True, "write_html": True},
    }
    summaries, views, folds, cells, windows = [], [], [], [], []
    for profile in profiles:
        level = int(profile["profile_id"][1:])
        model_offset = 0.1 if profile["model_id"] == MODELS[1] else 0.0
        effect = 0.0 if level in {0, 7} else level / 100.0
        summaries.append(
            {
                "case_id": profile["catalog_case_id"],
                "status": "passed",
                "complete_for_requested_execution": True,
            }
        )
        for view_index, view in enumerate((W, LINE_A_EQUAL_FILES, LINE_B_EQUAL_ROLE_FAMILIES)):
            score = 0.50 + model_offset + effect + view_index / 1000.0
            views.append(
                {
                    "case_id": profile["catalog_case_id"],
                    "aggregation_view": view,
                    "participant_mean_balanced_accuracy": score,
                    "participant_mean_macro_f1": score - 0.05,
                    "worst_class_f1": score - 0.10,
                }
            )
            for repeat in repeat_indices:
                for fold in range(5):
                    folds.append(
                        {
                            "case_id": profile["catalog_case_id"],
                            "aggregation_view": view,
                            "repeat": repeat,
                            "fold": fold,
                            "balanced_accuracy": score + fold / 10000.0,
                            "macro_f1": score - 0.05,
                            "worst_class_f1": score - 0.10,
                        }
                    )
        for repeat in repeat_indices:
            for fold in range(5):
                cells.append(
                    {
                        "case_id": profile["catalog_case_id"],
                        "repeat": repeat,
                        "fold": fold,
                        "status": "passed",
                        "split_seed": 42 + repeat,
                        "training_seed": 42,
                    }
                )
                probability = [0.7, 0.2, 0.1]
                windows.append(
                    {
                        "case_id": profile["catalog_case_id"],
                        "repeat": repeat,
                        "fold": fold,
                        "split_seed": 42 + repeat,
                        "training_seed": 42,
                        "manifest_hash": "manifest",
                        "fold_hash": f"repeat-{repeat}-fold-{fold}",
                        "participant_id": f"P{fold}",
                        "file_id": f"P{fold}_B",
                        "role": "B",
                        "window_id": f"P{fold}_B::0",
                        "label": 0,
                        "retained": True,
                        "class_order": [0, 1, 2],
                        "prediction_kind": "single_model",
                        "member_index": None,
                        "probabilities": probability,
                    }
                )
    bundle = CollectedStudy(
        root=Path("."), plan=plan, manifest={"cases": []}, case_records=(),
        varied_parameters=(), controlled_parameters=(), cell_rows=tuple(cells),
        history_rows=(), file_oof_rows=(), subject_oof_rows=(), role_oof_rows=(),
        quality_rows=(), trusted_config_metrics=(), limitations=(),
        window_oof_rows=tuple(windows),
    )
    return bundle, summaries, views, folds


def _repeat_metrics(
    bundle: CollectedStudy,
    views: list[dict],
) -> list[dict]:
    repeat_indices = tuple(
        int(value)
        for value in bundle.plan["legacy_bridge"]["budget"]["repeat_indices"]
    )
    rows: list[dict] = []
    for view in views:
        for repeat in repeat_indices:
            offset = repeat / 1000.0
            rows.append(
                {
                    "case_id": view["case_id"],
                    "aggregation_view": view["aggregation_view"],
                    "repeat": repeat,
                    "balanced_accuracy": (
                        view["participant_mean_balanced_accuracy"] + offset
                    ),
                    "macro_f1": view["participant_mean_macro_f1"] + offset,
                    "worst_class_f1": view["worst_class_f1"] + offset,
                }
            )
    return rows


class Stage3StarReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as pyplot
        except ImportError as error:
            raise unittest.SkipTest(str(error)) from error
        cls.pyplot = pyplot

    def tearDown(self) -> None:
        self.pyplot.close("all")

    def _tables(self, bundle: CollectedStudy | None = None):
        fixture, summaries, views, folds = _fixture()
        return _stage3_star_report_tables(
            bundle or fixture, summaries, views, folds
        )

    def test_materialized_repository_plan_is_accepted_without_running(self) -> None:
        bundle, _summaries, _views, _folds = _fixture()
        path = Path(__file__).resolve().parents[2] / "configs/studies/static_line_b_staged_v2/stage3_star.yaml"
        plan = load_study_plan(path).to_dict()
        absolute, contrasts, fold_rows, execution, notes = _stage3_star_report_tables(
            replace(bundle, plan=plan, cell_rows=(), window_oof_rows=()), [], [], []
        )
        self.assertEqual((len(absolute), len(contrasts), len(fold_rows), len(execution)), (16, 14, 350, 16))
        self.assertFalse(any("design contract is invalid" in note for note in notes))

    def test_exact_counts_signs_and_no_cross_model_contrasts(self) -> None:
        absolute, contrasts, fold_rows, execution, notes = self._tables()
        self.assertEqual((len(absolute), len(contrasts), len(fold_rows), len(execution)), (16, 14, 70, 16))
        self.assertTrue(all(row["reference_case_id"].split("__")[0] == row["variant_case_id"].split("__")[0] for row in contrasts))
        b1 = next(row for row in contrasts if row["model"] == MODELS[0] and row["variant_profile"] == "B1")
        self.assertAlmostEqual(b1["delta_native_balanced_accuracy"], 0.01)
        self.assertTrue(b1["contrast_metrics_available"])
        self.assertTrue(all(not any(key.startswith("delta_") for key in row) for row in execution))
        self.assertIn("80 repeat/fold cells", notes[-1])

    def test_full_five_repeat_budget_produces_complete_350_fold_contrasts(self) -> None:
        bundle, summaries, views, folds = _fixture(tuple(range(5)))
        absolute, contrasts, fold_rows, execution, notes = (
            _stage3_star_report_tables(bundle, summaries, views, folds)
        )
        self.assertEqual(
            (len(absolute), len(contrasts), len(fold_rows), len(execution)),
            (16, 14, 350, 16),
        )
        self.assertTrue(all(row["contrast_metrics_available"] for row in contrasts))
        b7 = next(
            row
            for row in contrasts
            if row["model"] == MODELS[0] and row["variant_profile"] == "B7"
        )
        self.assertEqual(b7["matched_window_oof_row_count"], 25)
        self.assertTrue(
            b7["report_view_factor_window_oof_probabilities_identical"]
        )
        self.assertIn("400 repeat/fold cells", notes[-1])
        analysis = replace(
            analyze_study(replace(bundle, plan={"report": {}})),
            stage3_star_fold_contrasts=tuple(fold_rows),
        )
        self.assertIsNotNone(
            _stage3_star_fold_delta_heatmap(analysis, self.pyplot)
        )

    def test_two_model_tables_and_horizontal_profile_matches_use_paired_repeats(
        self,
    ) -> None:
        bundle, summaries, views, folds = _fixture(tuple(range(5)))
        absolute, contrasts, _fold_rows, _execution, _notes = (
            _stage3_star_report_tables(bundle, summaries, views, folds)
        )
        inception, cnn, side_by_side, notes = _stage3_star_presentation_tables(
            bundle,
            absolute,
            contrasts,
            _repeat_metrics(bundle, views),
        )
        self.assertEqual((len(inception), len(cnn), len(side_by_side)), (8, 8, 8))
        self.assertFalse(notes)
        inception_b2 = next(row for row in inception if row["profile"] == "B2")
        self.assertAlmostEqual(inception_b2["native_balanced_accuracy"], 0.622)
        self.assertAlmostEqual(inception_b2["delta_vs_B0_balanced_accuracy"], 0.02)
        self.assertAlmostEqual(inception_b2["delta_vs_B0_balanced_accuracy_sd"], 0.0)
        self.assertEqual(
            inception_b2["comparison_type"],
            "within_model_B0_centered_ablation",
        )
        cnn_b7 = next(row for row in cnn if row["profile"] == "B7")
        self.assertAlmostEqual(cnn_b7["delta_vs_B0_balanced_accuracy"], 0.002)
        cross_b2 = next(row for row in side_by_side if row["profile"] == "B2")
        self.assertAlmostEqual(
            cross_b2["inception_minus_cnn_balanced_accuracy"], 0.1
        )
        self.assertEqual(
            cross_b2["comparison_type"],
            "matched_architecture_comparison_not_ablation",
        )
        self.assertTrue(cross_b2["comparison_metrics_available"])

    def test_b7_uses_native_W_to_B_and_exact_row_identity_audit(self) -> None:
        _absolute, contrasts, _folds, _execution, _notes = self._tables()
        b7 = next(row for row in contrasts if row["model"] == MODELS[0] and row["variant_profile"] == "B7")
        self.assertEqual(b7["native_comparison_semantics"], "B0_window_endpoint_to_variant_line_b_endpoint")
        self.assertAlmostEqual(b7["delta_native_balanced_accuracy"], 0.002)
        self.assertEqual(b7["matched_window_oof_row_count"], 5)
        self.assertEqual(b7["window_oof_probability_max_abs_diff"], 0.0)
        self.assertTrue(b7["report_view_factor_training_controls_identical"])
        self.assertTrue(b7["report_view_factor_window_oof_probabilities_identical"])

    def test_seed_or_single_factor_mismatch_suppresses_entire_contrast(self) -> None:
        bundle, summaries, views, folds = _fixture()
        bad_cells = [dict(row) for row in bundle.cell_rows]
        target = next(row for row in bad_cells if row["case_id"] == "compact__b1" and row["fold"] == 3)
        target["training_seed"] = 99
        bad = replace(bundle, cell_rows=tuple(bad_cells))
        _absolute, contrasts, fold_rows, _execution, _notes = _stage3_star_report_tables(bad, summaries, views, folds)
        b1 = next(row for row in contrasts if row["model"] == MODELS[0] and row["variant_profile"] == "B1")
        self.assertFalse(b1["contrast_metrics_available"])
        self.assertTrue(all(row["delta_native_balanced_accuracy"] is None for row in fold_rows if row["model"] == MODELS[0] and row["variant_profile"] == "B1"))

    def test_cross_model_reference_is_rejected(self) -> None:
        bundle, summaries, views, folds = _fixture()
        plan = dict(bundle.plan)
        bridge = dict(plan["legacy_bridge"])
        profiles = [dict(row) for row in bridge["profiles"]]
        target = next(row for row in profiles if row["case_id"] == "compact__B1")
        target["reference_case_id"] = "inception__B0"
        bridge["profiles"] = profiles
        plan["legacy_bridge"] = bridge
        absolute, contrasts, fold_rows, execution, notes = _stage3_star_report_tables(
            replace(bundle, plan=plan), summaries, views, folds
        )
        self.assertEqual((absolute, contrasts, fold_rows, execution), ([], [], [], []))
        self.assertIn("contrasts must be B0-centred within each model", notes[0])

    def test_generic_single_reference_pairing_is_disabled(self) -> None:
        bundle, _summaries, _views, _folds = _fixture()
        analysis = analyze_study(
            replace(bundle, manifest={"cases": [], "reference_case_id": "compact__b0"})
        )
        self.assertEqual(analysis.paired_deltas, ())
        self.assertTrue(
            any("single-reference paired deltas are disabled" in note for note in analysis.notes)
        )

    def test_missing_cell_and_fold_hash_mismatch_fail_closed(self) -> None:
        for mutation in ("missing_cell", "fold_hash"):
            with self.subTest(mutation=mutation):
                bundle, summaries, views, folds = _fixture()
                if mutation == "missing_cell":
                    cells = tuple(
                        row for row in bundle.cell_rows
                        if not (
                            row["case_id"] == "compact__b1" and row["fold"] == 4
                        )
                    )
                    bundle = replace(bundle, cell_rows=cells)
                else:
                    windows = [dict(row) for row in bundle.window_oof_rows]
                    for row in windows:
                        if row["case_id"] == "compact__b1" and row["fold"] == 4:
                            row["fold_hash"] = "mismatched-fold-hash"
                    bundle = replace(bundle, window_oof_rows=tuple(windows))
                _absolute, contrasts, fold_rows, _execution, _notes = (
                    _stage3_star_report_tables(bundle, summaries, views, folds)
                )
                b1 = next(
                    row for row in contrasts
                    if row["model"] == MODELS[0] and row["variant_profile"] == "B1"
                )
                self.assertFalse(b1["contrast_metrics_available"])
                self.assertTrue(
                    all(
                        row["delta_native_balanced_accuracy"] is None
                        for row in fold_rows
                        if row["model"] == MODELS[0]
                        and row["variant_profile"] == "B1"
                    )
                )

    def test_star_plots_and_report_outputs_are_registered(self) -> None:
        bundle, _summaries, views, _folds = _fixture()
        absolute, contrasts, fold_rows, execution, notes = self._tables(bundle)
        inception, cnn, model_comparison, presentation_notes = (
            _stage3_star_presentation_tables(
                bundle,
                absolute,
                contrasts,
                _repeat_metrics(bundle, views),
            )
        )
        base = analyze_study(replace(bundle, plan={"report": {}}))
        analysis = replace(
            base,
            stage3_star_absolute=tuple(absolute),
            stage3_star_contrasts=tuple(contrasts),
            stage3_star_fold_contrasts=tuple(fold_rows),
            stage3_star_execution=tuple(execution),
            stage3_star_inception_comparison=tuple(inception),
            stage3_star_cnn_comparison=tuple(cnn),
            stage3_star_model_comparison=tuple(model_comparison),
            notes=tuple(notes + presentation_notes),
        )
        self.assertIsNotNone(_stage3_star_model_deltas(analysis, self.pyplot))
        self.assertIsNotNone(_stage3_star_fold_delta_heatmap(analysis, self.pyplot))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_bundle = replace(bundle, root=root)
            from unittest.mock import patch
            with patch("ppg_frailty.reporting.report.analyze_study", return_value=analysis):
                result = generate_study_report(root, collected=result_bundle)
            self.assertTrue((root / "tables/stage3_star_fold_contrasts.json").is_file())
            self.assertTrue(
                (root / "tables/stage3_star_inception_comparison.csv").is_file()
            )
            self.assertTrue(
                (root / "tables/stage3_star_cnn_comparison.csv").is_file()
            )
            self.assertTrue(
                (root / "tables/stage3_star_model_comparison.csv").is_file()
            )
            self.assertTrue((root / "figures/stage3_star_model_deltas.png").is_file())
            markdown = result.summary_markdown.read_text(encoding="utf-8")
            self.assertIn("InceptionTime B0–B7 comparison", markdown)
            self.assertIn("CompactCNN B0–B7 comparison", markdown)
            self.assertIn("InceptionTime versus CompactCNN", markdown)
            paths = {
                row["path"]
                for row in json.loads(result.output_index.read_text(encoding="utf-8"))[
                    "artifacts"
                ]
            }
            self.assertIn("tables/stage3_star_contrasts.csv", paths)
            self.assertIn("tables/stage3_star_model_comparison.csv", paths)

    def test_unavailable_star_delta_plot_is_warning_free(self) -> None:
        bundle, _summaries, _views, _folds = _fixture()
        _absolute, contrasts, _fold_rows, _execution, _notes = self._tables(bundle)
        unavailable = tuple(
            {
                **row,
                "delta_native_balanced_accuracy": None,
                "delta_native_macro_f1": None,
                "delta_native_worst_class_f1": None,
            }
            for row in contrasts
        )
        analysis = replace(
            analyze_study(replace(bundle, plan={"report": {}})),
            stage3_star_contrasts=unavailable,
        )
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            with self.assertRaisesRegex(ValueError, "no available"):
                _stage3_star_model_deltas(analysis, self.pyplot)
        self.assertFalse(
            any("No artists with labels" in str(item.message) for item in observed)
        )


if __name__ == "__main__":
    unittest.main()
