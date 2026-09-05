from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from ppg_frailty.artifact import get_reducer as canonical_get_reducer
from ppg_frailty.peaks import detect_pulses
from ppg_frailty.peaks.aboy_project_v2 import (
    DETECTOR_ID as ABOY_V2_ID,
    IMPLEMENTATION_PATH as ABOY_V2_IMPLEMENTATION_PATH,
)
from ppg_frailty.peaks.msptdfast_v2 import (
    DETECTOR_ID as MSPTDFAST_V2_ID,
    IMPLEMENTATION_PATH as MSPTDFAST_IMPLEMENTATION_PATH,
    detect_msptdfast_v2,
)
from ppg_frailty.quality.stage5_pre import (
    PEAK_ABLATION_SCHEMA,
    STAGE5_SCHEMA,
    _aggregate_benchmark,
    _annotate_denoiser_uncertainty,
    _detector_report_rows,
    _denoiser_activity_result_rows,
    _file_prediction_rows,
    _holm_adjusted_p_values,
    _holm_sidak_step_down,
    _motion_detector_conclusion_rows,
    _paired_participant_sign_flip_p,
    _participant_mean_percentile_ci,
    _static_peak_rank_sum_comparisons,
    _stage5_reporter_output_status,
    _stage_directory,
    _rank_and_mark_denoiser_rows,
    align_and_score_beats,
    generate_motion_peak_report,
    get_reducer as stage5_get_reducer,
    load_motion_peak_plan,
)
from ppg_frailty.quality.motion_adapters import (
    fit_formal_motion_model,
    write_formal_motion_input_schema,
)
from ppg_frailty.quality.motion_runner import (
    MotionFitContext,
    MotionFittedArtifact,
    MotionWindowExample,
    _run_internal_reverse_evaluation_impl,
    _run_ptt_motion_training_ablation_impl,
    participant_macro_motion_metrics,
)
from ppg_frailty.quality.motion import load_motion_fold_jobs
from ppg_frailty.models.motion import FORMAL_MOTION_MODEL_ID
from ppg_frailty.data.external_folds import load_formal_ptt_repeated_folds
from ppg_frailty.representations.motion import MOTION_NETWORK_SCHEMA_SHA256


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PIPELINE_ROOT.parents[1]
PLAN_ROOT = PIPELINE_ROOT / "configs/studies/static_line_b_staged_v2"


def _cuda_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return bool(torch.cuda.is_available())


class Stage5PreAndStaticPeakAblationTests(unittest.TestCase):
    def test_participant_bootstrap_sign_flip_and_holm_are_deterministic(self) -> None:
        values = [0.70, 0.75, 0.80, 0.85, 0.90]
        first_ci = _participant_mean_percentile_ci(
            values, n_resamples=2000, seed=42
        )
        second_ci = _participant_mean_percentile_ci(
            values, n_resamples=2000, seed=42
        )
        self.assertEqual(first_ci, second_ci)
        self.assertLessEqual(first_ci[0], np.mean(values))
        self.assertGreaterEqual(first_ci[1], np.mean(values))

        p_value = _paired_participant_sign_flip_p(
            [1.0] * 8, n_resamples=100_000, seed=42
        )
        self.assertLess(p_value, 0.02)
        self.assertEqual(
            _paired_participant_sign_flip_p(
                [0.0] * 8, n_resamples=100_000, seed=42
            ),
            1.0,
        )
        adjusted = _holm_adjusted_p_values(
            {"a": 0.01, "b": 0.03, "c": 0.20}
        )
        self.assertEqual(adjusted, {"a": 0.03, "b": 0.06, "c": 0.2})

    def test_motion_detector_conclusions_never_cross_endpoint_strata(self) -> None:
        def rows(*, correct: bool) -> list[dict[str, object]]:
            return [
                {
                    "participant_id": participant,
                    "file_id": f"{participant}_{label}",
                    "activity_label": label,
                    "repeat_index": 0,
                    "fold_index": 0,
                    "p_active": (
                        0.9 if (label == 1) == correct else 0.1
                    ),
                    "threshold": 0.5,
                    "predicted_activity": label if correct else 1 - label,
                }
                for participant in ("p1", "p2")
                for label in (0, 1)
            ]

        metrics, _ = _detector_report_rows(
            [
                (
                    "model_a",
                    "frailty29_outer_oof",
                    "source_grouped_oof",
                    rows(correct=False),
                ),
                (
                    "model_b",
                    "frailty29_outer_oof",
                    "source_grouped_oof",
                    rows(correct=True),
                ),
                (
                    "model_cross_dataset",
                    "frailty29_trained_to_ptt22",
                    "frozen_cross_dataset",
                    rows(correct=True),
                ),
            ],
            participant_cluster_bootstrap_resamples=31,
            participant_cluster_bootstrap_seed=17,
        )
        conclusions = _motion_detector_conclusion_rows(
            metrics,
            denoiser_enabled=False,
        )
        endpoints = {
            (
                row["target_dataset"],
                row["evaluation_scope"],
                row["aggregation_level"],
            ): row
            for row in conclusions
            if row["angle"].startswith("motion_detector_endpoint::")
        }
        self.assertEqual(len(endpoints), 4)
        self.assertEqual(
            endpoints[("frailty29", "source_grouped_oof", "window")][
                "leading_or_selected_case"
            ],
            "model_b",
        )
        self.assertEqual(
            endpoints[("ptt22", "frozen_cross_dataset", "window")][
                "leading_or_selected_case"
            ],
            "model_cross_dataset",
        )
        self.assertEqual(
            endpoints[("frailty29", "source_grouped_oof", "window")][
                "within_stratum_candidate_count"
            ],
            2,
        )
        self.assertTrue(
            all(
                "No comparison is made across target datasets" in row["finding"]
                for row in endpoints.values()
            )
        )
        self.assertTrue(
            all(
                row["participant_cluster_bootstrap_resamples"] == 31
                and row["participant_cluster_bootstrap_seed"] == 17
                for row in metrics
            )
        )

    def test_obsolete_subject_confusions_require_complete_window_file_replacements(
        self,
    ) -> None:
        obsolete_ids = (
            "motion_internal_subject_confusion_matrix",
            "motion_ptt_subject_confusion_matrix",
            "motion_ptt_training_oof_subject_confusion_matrix",
            "motion_internal_reverse_subject_confusion_matrix",
        )
        datasets = (
            "frailty29_outer_oof",
            "frailty29_trained_to_ptt22",
            "ptt22_outer_oof",
            "ptt22_trained_to_frailty29",
        )
        replacement_figures = (
            "motion_internal_confusion_matrix",
            "motion_internal_file_confusion_matrix",
            "motion_ptt_confusion_matrix",
            "motion_ptt_file_confusion_matrix",
            "motion_ptt_training_oof_confusion_matrix",
            "motion_ptt_training_oof_file_confusion_matrix",
            "motion_internal_reverse_confusion_matrix",
            "motion_internal_reverse_file_confusion_matrix",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tables = root / "tables"
            figures = root / "figures"
            tables.mkdir()
            figures.mkdir()
            window_rows = [
                {"dataset": dataset, "aggregation_level": "window"}
                for dataset in datasets
            ]
            file_rows = [
                {
                    "dataset": dataset,
                    "aggregation_level": "file_median_window_probability",
                }
                for dataset in datasets
            ]
            for table_name, rows in (
                ("motion_detector_window_confusion", window_rows),
                ("motion_detector_file_confusion", file_rows),
            ):
                (tables / f"{table_name}.json").write_text(
                    json.dumps(rows), encoding="utf-8"
                )
                (tables / f"{table_name}.csv").write_text(
                    "dataset,aggregation_level\n", encoding="utf-8"
                )
            for figure_id in replacement_figures:
                (figures / f"{figure_id}.png").touch()

            rows = _stage5_reporter_output_status(
                profile_rows=(),
                report_config={"required_detector_figures": obsolete_ids},
                tables=tables,
                figures=figures,
                reverse_available=True,
                denoiser_enabled=False,
            )

            self.assertEqual({row["output_id"] for row in rows}, set(obsolete_ids))
            self.assertTrue(
                all(
                    row["status"] == "N/A"
                    and row["reason"]
                    == "superseded_by_window_and_file_level_contract"
                    and row["replacement_status"] == "generated"
                    for row in rows
                )
            )
            self.assertTrue(
                all(
                    not (figures / f"{output_id}.png").exists()
                    for output_id in obsolete_ids
                )
            )

            (figures / "motion_ptt_file_confusion_matrix.png").unlink()
            with self.assertRaisesRegex(
                ValueError,
                "motion_ptt_subject_confusion_matrix",
            ):
                _stage5_reporter_output_status(
                    profile_rows=(),
                    report_config={"required_detector_figures": obsolete_ids},
                    tables=tables,
                    figures=figures,
                    reverse_available=True,
                    denoiser_enabled=False,
                )

    def test_stage5_uses_the_canonical_pipeline_reducer_factory(self) -> None:
        self.assertIs(stage5_get_reducer, canonical_get_reducer)

    def test_detector_auc_ties_are_order_invariant(self) -> None:
        rows = [
            {
                "participant_id": "p1",
                "activity_label": label,
                "p_active": 0.5,
                "threshold": 0.5,
            }
            for label in (1, 0)
        ]
        forward = participant_macro_motion_metrics(rows)
        backward = participant_macro_motion_metrics(list(reversed(rows)))
        self.assertEqual(forward["roc_auc"], 0.5)
        self.assertEqual(forward["pr_auc"], 0.5)
        self.assertEqual(forward, backward)

    def test_stage5_plan_is_complete_ptt_and_uses_only_real_reducers(self) -> None:
        plan = load_motion_peak_plan(PLAN_ROOT / "stage5_pre.yaml")
        self.assertEqual(plan.schema_version, STAGE5_SCHEMA)
        self.assertEqual(plan.payload["ptt_dataset"]["record_count"], 66)
        self.assertEqual(plan.payload["motion_detector"]["internal_participant_count"], 29)
        self.assertEqual(plan.payload["motion_detector"]["training_device"], "cuda")
        self.assertFalse(
            any(
                "subject_confusion_matrix" in figure_id
                for figure_id in plan.payload["report"]["required_detector_figures"]
            )
        )
        self.assertEqual(
            plan.payload["motion_detector"]["split"]["method"],
            "StratifiedGroupKFold",
        )
        self.assertEqual(
            plan.payload["motion_detector"]["reverse_ablation"]["repeat_indices"],
            [0],
        )
        self.assertEqual(
            plan.payload["motion_detector"]["reverse_ablation"]["folds"],
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(
            plan.payload["motion_model_comparison"]["candidates"],
            [
                "frailty29_trained_legacy_reference",
                "ptt22_trained_reverse_ablation",
            ],
        )
        self.assertNotIn("learned_denoiser", plan.payload["denoiser_benchmark"]["reducers"])
        self.assertEqual(
            plan.payload["denoiser_benchmark"]["scoring_peak_detector"],
            MSPTDFAST_V2_ID,
        )
        serialized = json.dumps(plan.payload, sort_keys=True).lower()
        self.assertNotIn("training_gate", serialized)
        self.assertNotIn("phase0", serialized)

    def test_peak_ablation_is_strictly_static_and_paired(self) -> None:
        plan = load_motion_peak_plan(
            PLAN_ROOT / "stage_ablation_01_static_peak_detectors.yaml"
        )
        self.assertEqual(plan.schema_version, PEAK_ABLATION_SCHEMA)
        self.assertEqual(plan.payload["activities"], ["sit"])
        self.assertEqual(plan.payload["ptt_dataset"]["selected_record_count"], 22)
        self.assertEqual(plan.payload["motion_denoiser"], "not_used_pure_static_experiment")
        self.assertEqual(
            {row["algorithm_id"] for row in plan.payload["algorithms"]},
            {"aboy_project", "msptdfast_v2_3_python_port"},
        )
        implementations = {
            row["algorithm_id"]: row["implementation"]
            for row in plan.payload["algorithms"]
        }
        self.assertEqual(
            implementations[MSPTDFAST_V2_ID], MSPTDFAST_IMPLEMENTATION_PATH
        )
        self.assertEqual(
            implementations["aboy_project"], ABOY_V2_IMPLEMENTATION_PATH
        )
        modules = {
            row["algorithm_id"]: row["module_id"]
            for row in plan.payload["algorithms"]
        }
        self.assertEqual(modules["aboy_project"], ABOY_V2_ID)
        self.assertNotIn("aboy_project_v1", modules.values())
        self.assertEqual(plan.payload["validation"]["lag_window_s"], 300.0)
        self.assertEqual(plan.payload["validation"]["beat_tolerance_s"], 0.15)
        self.assertEqual(
            plan.payload["detector_input"],
            "repaired_native_ppg_each_registered_module_owns_preprocessing",
        )
        statistical = plan.payload["validation"]["statistical_comparison"]
        self.assertEqual(
            statistical["metrics"],
            [
                "recording_f1_percent",
                "recording_sensitivity_percent",
                "recording_positive_predictive_value_percent",
                "recording_ibi_ppi_rmse_ms",
                "execution_time_percent_of_ppg_signal_duration",
            ],
        )
        self.assertEqual(
            statistical["family_definition"],
            "all_selected_metrics_channels_and_reference_comparators",
        )

    def test_peak_ablation_rejects_a_study_local_detector_copy(self) -> None:
        source = PLAN_ROOT / "stage_ablation_01_static_peak_detectors.yaml"
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "copied_detector.yaml"
            copied.write_text(
                source.read_text(encoding="utf-8").replace(
                    MSPTDFAST_IMPLEMENTATION_PATH,
                    "ppg_frailty.quality.stage5_pre.detect_msptdfast_v2",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "registered default MSPTDfast module"):
                load_motion_peak_plan(copied)

    def test_delay_shift_does_not_change_interval_error(self) -> None:
        reference = np.arange(0.5, 20.0, 0.8)
        predicted = reference + 0.34
        result = align_and_score_beats(reference, predicted)
        self.assertAlmostEqual(result["lag_s"], 0.16, places=8)
        self.assertAlmostEqual(result["f1"], 1.0)
        self.assertAlmostEqual(result["ibi_ppi_rmse_ms"], 0.0, places=9)

    def test_recording_lag_can_change_at_300_seconds(self) -> None:
        reference = np.arange(1.0, 599.0, 1.0)
        predicted = reference + np.where(reference < 300.0, 0.2, 0.46)
        result = align_and_score_beats(
            reference,
            predicted,
            max_lag_s=1.0,
            lag_step_s=0.01,
            tolerance_s=0.15,
            lag_window_s=300.0,
            recording_duration_s=600.0,
        )
        self.assertEqual(len(result["lag_windows"]), 2)
        self.assertNotEqual(
            result["lag_windows"][0]["lag_s"],
            result["lag_windows"][1]["lag_s"],
        )
        self.assertEqual(result["ncorrect"], result["nref"])
        self.assertAlmostEqual(result["f1_percent"], 100.0)

    def test_holm_sidak_step_down_is_monotone(self) -> None:
        adjusted, rejected, ranks = _holm_sidak_step_down(
            [0.01, 0.04, 0.20], alpha=0.05
        )
        self.assertEqual(ranks, [1, 2, 3])
        self.assertLessEqual(adjusted[0], adjusted[1])
        self.assertLessEqual(adjusted[1], adjusted[2])
        self.assertEqual(rejected, [True, False, False])

    def test_static_peak_rank_sum_unifies_all_endpoint_channel_tests(self) -> None:
        rows = []
        for participant_index in range(4):
            for algorithm, offset in (
                ("msptdfast_v2_3_python_port", 0.0),
                ("aboy_project", 2.0),
            ):
                for channel in ("RED", "IR"):
                    rows.append(
                        {
                            "participant_id": f"p{participant_index}",
                            "record_id": f"p{participant_index}_sit",
                            "algorithm_or_reducer": algorithm,
                            "channel": channel,
                            "status": "passed",
                            "f1_percent": 99.0 - offset - participant_index,
                            "sensitivity_percent": (
                                99.0 - offset - participant_index
                            ),
                            "positive_predictive_value_percent": (
                                98.0 - offset - participant_index
                            ),
                            "ibi_ppi_rmse_ms": (
                                10.0 + offset + participant_index
                            ),
                            "execution_time_percent": (
                                0.01 + 0.01 * offset + participant_index * 0.001
                            ),
                        }
                    )
        comparisons = _static_peak_rank_sum_comparisons(
            rows,
            reference_algorithm_id="msptdfast_v2_3_python_port",
            alpha=0.05,
            registered_metric_ids=["recording_f1_percent"],
        )
        self.assertEqual(len(comparisons), 10)
        self.assertEqual(
            {row["metric"] for row in comparisons},
            {
                "recording_f1_percent",
                "recording_sensitivity_percent",
                "recording_positive_predictive_value_percent",
                "recording_ibi_ppi_rmse_ms",
                "execution_time_percent_of_ppg_signal_duration",
            },
        )
        self.assertEqual(
            {row["holm_sidak_family_size"] for row in comparisons}, {10}
        )
        self.assertTrue(
            all(row["pairing_used_by_test"] is False for row in comparisons)
        )
        self.assertEqual(
            {
                row["analysis_registration"]
                for row in comparisons
                if row["metric"] == "recording_f1_percent"
            },
            {"prespecified_in_resolved_plan"},
        )
        self.assertEqual(
            {
                row["analysis_registration"]
                for row in comparisons
                if row["metric"] != "recording_f1_percent"
            },
            {"retrospective_supplement_requested_2026-08-24"},
        )
        self.assertEqual(
            {
                row["registered_family_size"]
                for row in comparisons
                if row["metric"] == "recording_f1_percent"
            },
            {2},
        )

    def test_interval_metric_excludes_nonconsecutive_matches(self) -> None:
        reference = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
        predicted = np.asarray([0.2, 1.2, 3.2, 4.2])
        result = align_and_score_beats(reference, predicted)
        self.assertEqual(result["matched_interval_count"], 2)
        self.assertAlmostEqual(result["ibi_ppi_rmse_ms"], 0.0, places=9)

    def test_msptdfast_port_detects_regular_synthetic_beats(self) -> None:
        fs_hz = 400.0
        time_s = np.arange(int(18 * fs_hz)) / fs_hz
        values = np.zeros_like(time_s)
        expected = np.arange(1.0, 17.1, 1.0)
        for peak in expected:
            values += np.exp(-0.5 * np.square((time_s - peak) / 0.035))
        observed = detect_msptdfast_v2(values, fs_hz) / fs_hz
        matched = align_and_score_beats(expected, observed, max_lag_s=0.5)
        self.assertGreaterEqual(matched["f1"], 0.90)
        self.assertEqual(
            hashlib.sha256(
                np.rint(observed * fs_hz).astype("<i8").tobytes()
            ).hexdigest(),
            "2be812f579d630bcecd0f8efa44f606ca7ec1cf1bdfcf757191b67dbfa32d567",
        )

    def test_msptdfast_runtime_consumes_nondefault_parameters(self) -> None:
        fs_hz = 100.0
        time_s = np.arange(int(12 * fs_hz)) / fs_hz
        values = np.sin(2.0 * np.pi * time_s)
        peaks = detect_msptdfast_v2(
            values,
            fs_hz,
            target_downsample_hz=25.0,
            minimum_heart_rate_bpm=40.0,
            window_s=4.0,
            overlap_fraction=0.5,
        )
        self.assertGreater(len(peaks), 5)

    def test_pipeline_resolver_uses_the_same_msptdfast_peak_indices(self) -> None:
        fs_hz = 400.0
        time_s = np.arange(int(18 * fs_hz)) / fs_hz
        values = np.sin(2.0 * np.pi * 1.1 * time_s)
        direct = detect_msptdfast_v2(values, fs_hz)
        resolved = detect_pulses(
            values,
            detector_id=MSPTDFAST_V2_ID,
            fs_hz=fs_hz,
            wavelength="RED",
        )
        np.testing.assert_array_equal(resolved.peaks, direct)
        self.assertEqual(resolved.detector_id, MSPTDFAST_V2_ID)
        custom = {
            "target_downsample_hz": 25.0,
            "minimum_heart_rate_bpm": 40.0,
            "window_s": 4.0,
            "overlap_fraction": 0.5,
        }
        configured = detect_pulses(
            values,
            detector_id=MSPTDFAST_V2_ID,
            fs_hz=fs_hz,
            wavelength="RED",
            detector_parameters=custom,
        )
        self.assertEqual(configured.block_provenance[0]["parameters"], custom)

    def test_msptdfast_runs_on_local_ptt_static_excerpt(self) -> None:
        source = (
            REPOSITORY_ROOT
            / "physionet.org/files/pulse-transit-time-ppg/1.1.0/csv/s1_sit.csv"
        )
        values = np.loadtxt(
            source,
            delimiter=",",
            skiprows=1,
            usecols=(3,),
            max_rows=30_000,
        )
        peaks = detect_msptdfast_v2(values, 500.0)
        self.assertGreater(len(peaks), 30)
        self.assertTrue(np.all(np.diff(peaks) > 0))

    def test_static_report_writes_plots_index_and_small_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = load_motion_peak_plan(
                PLAN_ROOT / "stage_ablation_01_static_peak_detectors.yaml"
            )
            (root / "resolved_plan.yaml").write_text(
                plan.path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            (root / "study_manifest.json").write_text(
                json.dumps({
                    "study_id": plan.study_id,
                    "study_type": plan.study_type,
                    "status": "passed",
                    "scientific_scope": "synthetic report fixture",
                }),
                encoding="utf-8",
            )
            rows = []
            for participant_index in range(2):
                for algorithm, f1, rmse in (
                    ("aboy_project", 91.0, 19.0),
                    ("msptdfast_v2_3_python_port", 92.0, 18.0),
                ):
                    for channel in ("RED", "IR"):
                        rows.append({
                            "participant_id": f"p{participant_index}",
                            "record_id": f"p{participant_index}_sit",
                            "algorithm_or_reducer": algorithm,
                            "activity_group": "static",
                            "channel": channel,
                            "status": "passed",
                            "f1_percent": f1 + participant_index,
                            "sensitivity_percent": f1 + participant_index,
                            "positive_predictive_value_percent": f1 + participant_index,
                            "ibi_ppi_rmse_ms": rmse + participant_index,
                            "execution_time_percent": 0.1 + participant_index * 0.01,
                        })
            (root / "static_peak_ablation.json").write_text(
                json.dumps({
                    "schema_version":
                        "ppg_frailty.stage_ablation_01_static_peak_result.v3",
                    "rows": rows,
                    "statistical_comparisons": [],
                }),
                encoding="utf-8",
            )
            result = generate_motion_peak_report(root)
            self.assertEqual(result["figure_count"], 5)
            static_report = (root / "STUDY_SUMMARY.md").read_text(
                encoding="utf-8"
            )
            static_headers = [
                line
                for line in static_report.splitlines()
                if line.startswith("| ") and not line.startswith("| ---")
            ]
            self.assertTrue(static_headers)
            self.assertTrue(
                all(line.count("|") - 1 <= 8 for line in static_headers)
            )
            self.assertTrue((root / "tables" / "report_tables.xlsx").is_file())
            self.assertTrue((root / "tables" / "table_figure_pairs.csv").is_file())
            self.assertTrue((root / "tables" / "test_components.csv").is_file())
            self.assertTrue(
                (root / "tables/static_peak_detector_recording_metrics.csv").is_file()
            )
            self.assertTrue(
                (root / "tables/static_peak_detector_rank_sum_holm_sidak.csv").is_file()
            )
            self.assertTrue(
                (root / "tables/static_peak_detector_significance_summary.csv").is_file()
            )
            comparison_rows = json.loads(
                (root / "tables/result_comparison.json").read_text(
                    encoding="utf-8"
                )
            )
            rank_sum_rows = [
                row
                for row in comparison_rows
                if row.get("test") == "wilcoxon_rank_sum_two_sided"
            ]
            self.assertTrue(rank_sum_rows)
            self.assertEqual(len(rank_sum_rows), 10)
            self.assertEqual(
                {row["holm_sidak_family_size"] for row in rank_sum_rows},
                {10},
            )
            self.assertEqual(
                {row["evidence_type"] for row in rank_sum_rows},
                {"recording_rank_sum_endpoint_test"},
            )
            self.assertTrue((root / "TEST_COMPONENTS.md").is_file())
            self.assertTrue((root / "STUDY_SUMMARY.html").is_file())
            self.assertIn(
                "<table>",
                (root / "STUDY_SUMMARY.html").read_text(encoding="utf-8"),
            )
            index = json.loads((root / "outputs_index.json").read_text(encoding="utf-8"))
            self.assertTrue(
                any(
                    row["path"].endswith("static_peak_detector_f1.png")
                    for row in index["files"]
                )
            )
            backup = json.loads(
                (root / "result_backup/backup_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(backup["entries"])
            for row in backup["entries"]:
                copied = root / row["backup"]
                self.assertEqual(hashlib.sha256(copied.read_bytes()).hexdigest(), row["sha256"])

    @unittest.skipUnless(_cuda_available(), "formal motion training requires CUDA")
    def test_motion_training_records_loss_and_train_ba_without_selection(self) -> None:
        rng = np.random.default_rng(4)
        examples = tuple(
            MotionWindowExample(
                window_id=f"window-{index}",
                participant_id=participant,
                file_id=f"file-{index}",
                role_or_activity=role,
                activity_label=label,
                values=rng.normal(size=(8, 3200)).astype(np.float32),
                dataset_id="internal",
            )
            for index, (participant, role, label) in enumerate(
                (("p1", "B", 0), ("p1", "S", 1), ("p2", "R", 0), ("p2", "W", 1))
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = MotionFitContext(
                execution_mode="formal",
                repeat_index=0,
                fold_index=0,
                split_seed=42,
                training_seed=42,
                final_fit=False,
                training_participant_ids=("p1", "p2"),
                held_out_participant_ids=(),
                model_input_schema_sha256=MOTION_NETWORK_SCHEMA_SHA256,
                artifact_directory=root / "cell",
            )
            fitted = fit_formal_motion_model(examples, context)
            self.assertEqual(fitted.runtime_model.device, "cuda")
            self.assertEqual(
                next(fitted.runtime_model.model.parameters()).device.type,
                "cuda",
            )
            history = json.loads(
                (root / "cell/motion_training_history.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(history["rows"]), 10)
            self.assertTrue(all(not row["outer_heldout_used"] for row in history["rows"]))
            self.assertTrue(
                all(not row["used_for_epoch_selection_or_checkpoint"] for row in history["rows"])
            )
            self.assertTrue(
                all(0.0 <= row["training_balanced_accuracy"] <= 1.0 for row in history["rows"])
            )

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow not installed")
    def test_stage5_report_contains_confusions_learning_curves_and_backup(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = load_motion_peak_plan(PLAN_ROOT / "stage5_pre.yaml")
            resolved_payload = yaml.safe_load(plan.path.read_text(encoding="utf-8"))
            resolved_payload["report"]["participant_cluster_bootstrap_resamples"] = 257
            resolved_payload["report"]["participant_cluster_bootstrap_seed"] = 7
            resolved_payload["report"]["required_detector_figures"].extend(
                [
                    "motion_internal_subject_confusion_matrix",
                    "motion_ptt_subject_confusion_matrix",
                ]
            )
            (root / "resolved_plan.yaml").write_text(
                yaml.safe_dump(resolved_payload, sort_keys=False), encoding="utf-8"
            )
            (root / "study_manifest.json").write_text(
                json.dumps({
                    "study_id": plan.study_id,
                    "study_type": plan.study_type,
                    "status": "passed",
                    "scientific_scope": "synthetic Stage5 report fixture",
                    "stages": {
                        "internal_motion_oof": {
                            "status": "passed", "artifact_dir": "motion_internal"
                        },
                        "ptt_motion_external": {
                            "status": "passed", "artifact_dir": "motion_external"
                        },
                        "ptt_denoiser_benchmark": {
                            "status": "passed", "artifact_dir": "denoiser"
                        },
                    },
                }),
                encoding="utf-8",
            )
            internal = root / "motion_internal"
            external = root / "motion_external"
            denoiser = root / "denoiser"
            internal.mkdir()
            external.mkdir()
            denoiser.mkdir()
            metrics = {
                "participant_macro_balanced_accuracy": 0.8,
                "worst_fold_balanced_accuracy": 0.7,
                "participant_macro_f1": 0.79,
                "ece": 0.1,
                "parameter_count": 100,
                "inference_cost": {},
            }
            (internal / "motion_internal_evidence.json").write_text(
                json.dumps({"major_metrics": metrics}), encoding="utf-8"
            )
            (external / "motion_ptt_external_report.json").write_text(
                json.dumps({"major_metrics": metrics}), encoding="utf-8"
            )
            prediction_rows = [
                {
                    "participant_id": "p1", "file_id": "p1_static",
                    "activity_label": 0, "repeat_index": 0, "fold_index": 0,
                    "p_active": 0.1, "threshold": 0.5, "predicted_activity": 0,
                },
                {
                    "participant_id": "p1", "file_id": "p1_motion",
                    "activity_label": 1, "repeat_index": 0, "fold_index": 0,
                    "p_active": 0.9, "threshold": 0.5, "predicted_activity": 1,
                },
                {
                    "participant_id": "p2", "file_id": "p2_static",
                    "activity_label": 0, "repeat_index": 0, "fold_index": 0,
                    "p_active": 0.8, "threshold": 0.5, "predicted_activity": 1,
                },
                {
                    "participant_id": "p2", "file_id": "p2_motion",
                    "activity_label": 1, "repeat_index": 0, "fold_index": 0,
                    "p_active": 0.2, "threshold": 0.5, "predicted_activity": 0,
                },
            ]
            pq.write_table(
                pa.Table.from_pylist(prediction_rows),
                internal / "motion_window_oof.parquet",
            )
            pq.write_table(
                pa.Table.from_pylist([
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"repeat_index", "fold_index"}
                    }
                    for row in prediction_rows
                ]),
                external / "motion_ptt_window_predictions.parquet",
            )
            history_dir = internal / "repeat_0/fold_0"
            history_dir.mkdir(parents=True)
            (history_dir / "motion_training_history.json").write_text(
                json.dumps({
                    "fold_index": 0,
                    "final_fit": False,
                    "rows": [
                        {"epoch": 1, "training_loss": 0.7, "training_balanced_accuracy": 0.6},
                        {"epoch": 2, "training_loss": 0.5, "training_balanced_accuracy": 0.8},
                    ],
                }),
                encoding="utf-8",
            )
            summary = [
                {
                    "algorithm_or_reducer": "identity",
                    "activity_group": activity_group,
                    "channel": "RED",
                    "participant_count": 22,
                    "segment_count": 44,
                    "participant_macro_f1": 0.8,
                    "participant_macro_ibi_ppi_rmse_ms": 30.0,
                    "total_runtime_s": 2.0,
                }
                for activity_group in ("static", "dynamic")
            ]
            (denoiser / "denoiser_benchmark.json").write_text(
                json.dumps({"summary_rows": summary}), encoding="utf-8"
            )
            result = generate_motion_peak_report(root)
            self.assertEqual(result["figure_count"], 17)
            self.assertTrue((root / "tables" / "report_tables.xlsx").is_file())
            self.assertTrue(
                (root / "tables" / "inference_configuration.csv").is_file()
            )
            self.assertTrue((root / "tables" / "table_figure_pairs.csv").is_file())
            self.assertTrue((root / "tables" / "denoiser_algorithms.csv").is_file())
            component_table = (root / "TEST_COMPONENTS.md").read_text(encoding="utf-8")
            component_header = next(
                line for line in component_table.splitlines() if line.startswith("| ")
            )
            self.assertTrue(component_header.startswith("| Model / module |"))
            self.assertIn("Algorithm and kernel (≤300 chars)", component_table)
            self.assertIn("pca_bss", component_table)
            generate_motion_peak_report(root)
            self.assertTrue((root / "figures/motion_training_learning_curves.png").is_file())
            self.assertTrue((root / "result_backup/backup_manifest.json").is_file())
            self.assertIn(
                "<table>",
                (root / "STUDY_SUMMARY.html").read_text(encoding="utf-8"),
            )
            report_text = (root / "STUDY_SUMMARY.md").read_text(encoding="utf-8")
            markdown_headers = [
                line
                for line in report_text.splitlines()
                if line.startswith("| ") and not line.startswith("| ---")
            ]
            self.assertTrue(markdown_headers)
            self.assertTrue(
                all(line.count("|") - 1 <= 8 for line in markdown_headers)
            )
            html_text = (root / "STUDY_SUMMARY.html").read_text(
                encoding="utf-8"
            )
            html_headings = re.findall(
                r"<thead><tr>(.*?)</tr></thead>", html_text
            )
            self.assertTrue(html_headings)
            self.assertTrue(
                all(heading.count("<th>") <= 8 for heading in html_headings)
            )
            detector_section = report_text.split(
                "#### Detector — Balanced accuracy", 1
            )[1]
            detector_header = next(
                line for line in detector_section.splitlines() if line.startswith("| ")
            )
            self.assertTrue(detector_header.startswith("| model_id |"))
            self.assertEqual(detector_header.count("|"), 7)
            denoiser_section = report_text.split(
                "### Denoiser results: static", 1
            )[1]
            denoiser_header = next(
                line for line in denoiser_section.splitlines() if line.startswith("| ")
            )
            self.assertEqual(
                denoiser_header,
                "| denoiser | IR/RED | RMSE ± SD (ms) | F1 ± SD (%) | RMSE P versus identity |",
            )
            self.assertEqual(denoiser_header.count("|"), 6)
            self.assertIn("mean ± SD", report_text)
            self.assertIn("Detector P is N/A because", report_text)
            self.assertIn(
                "displayed RMSE P is the retrospective exploratory",
                report_text,
            )
            with (root / "tables/motion_detector_balanced_accuracy.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                self.assertEqual(len(next(csv.reader(handle))), 6)
            for activity_group in ("static", "dynamic"):
                with (root / f"tables/denoiser_{activity_group}.csv").open(
                    encoding="utf-8", newline=""
                ) as handle:
                    denoiser_table = list(csv.DictReader(handle))
                self.assertTrue(denoiser_table)
                self.assertEqual(
                    list(denoiser_table[0]),
                    [
                        "denoiser",
                        "IR/RED",
                        "RMSE ± SD (ms)",
                        "F1 ± SD (%)",
                        "RMSE P versus identity",
                    ],
                )
                rmse_means = [
                    float(row["RMSE ± SD (ms)"].split(" ± ", 1)[0].rstrip("*"))
                    for row in denoiser_table
                    if row["RMSE ± SD (ms)"] != "N/A"
                ]
                self.assertEqual(rmse_means, sorted(rmse_means))
            for retired in (
                "denoiser_beat_f1_red.csv",
                "denoiser_beat_f1_ir.csv",
                "denoiser_sensitivity_red.csv",
                "denoiser_sensitivity_ir.csv",
                "denoiser_ppv_red.csv",
                "denoiser_ppv_ir.csv",
                "denoiser_ibi_ppi_rmse_red.csv",
                "denoiser_ibi_ppi_rmse_ir.csv",
            ):
                self.assertFalse((root / "tables" / retired).exists())
            with (root / "tables/result_comparison.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                self.assertEqual(len(next(csv.reader(handle))), 8)
            for table_name in (
                "motion_detector_training_source_inference.csv",
                "denoiser_paired_inference.csv",
            ):
                with (root / "tables" / table_name).open(
                    encoding="utf-8", newline=""
                ) as handle:
                    inference_rows = list(csv.reader(handle))
                self.assertGreaterEqual(len(inference_rows), 2)
                self.assertEqual(len(inference_rows[0]), 8)
            detector = json.loads(
                (root / "tables/motion_detector_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(detector), 4)
            self.assertTrue(
                {
                    "model_id",
                    "dataset",
                    "target_dataset",
                    "evaluation_scope",
                    "aggregation_level",
                    "file_score_aggregation",
                    "observation_count",
                    "participant_count",
                    "file_count",
                    "window_count",
                    "balanced_accuracy",
                    "macro_f1",
                    "sensitivity",
                    "specificity",
                    "roc_auc",
                    "pr_auc",
                    "worst_fold_balanced_accuracy",
                    "worst_fold_balanced_accuracy_applicability",
                    "worst_fold_balanced_accuracy_reason",
                    "repeat_uncertainty_applicability",
                    "repeat_uncertainty_reason",
                    "balanced_accuracy_repeat_sample_sd",
                    "balanced_accuracy_repeat_t_ci95_low",
                    "balanced_accuracy_repeat_t_ci95_high",
                    "balanced_accuracy_participant_cluster_ci95_low",
                    "balanced_accuracy_participant_cluster_ci95_high",
                    "macro_f1_participant_cluster_ci95_low",
                    "macro_f1_participant_cluster_ci95_high",
                    "sensitivity_participant_cluster_ci95_low",
                    "sensitivity_participant_cluster_ci95_high",
                    "specificity_participant_cluster_ci95_low",
                    "specificity_participant_cluster_ci95_high",
                    "roc_auc_participant_cluster_ci95_low",
                    "roc_auc_participant_cluster_ci95_high",
                    "roc_auc_participant_cluster_ci95_applicability",
                    "roc_auc_participant_cluster_ci95_reason",
                    "participant_cluster_ci95_applicability",
                    "participant_cluster_ci95_method",
                    "participant_cluster_bootstrap_resamples",
                    "participant_cluster_bootstrap_seed",
                    "roc_pr_participant_cluster_ci95_applicability",
                    "roc_pr_participant_cluster_ci95_reason",
                    "paired_inference_applicability",
                    "paired_inference_reason",
                    "balanced_accuracy_p_value",
                    "macro_f1_p_value",
                }.issubset(detector[0]),
            )
            self.assertEqual(
                {row["aggregation_level"] for row in detector}, {"window", "file"}
            )
            self.assertEqual(
                {
                    (row["dataset"], row["target_dataset"])
                    for row in detector
                },
                {
                    ("frailty29_outer_oof", "frailty29"),
                    ("frailty29_trained_to_ptt22", "ptt22"),
                },
            )
            self.assertTrue(
                all(
                    row["repeat_uncertainty_applicability"] == "N/A"
                    and row["balanced_accuracy_repeat_sample_sd"] is None
                    and row["balanced_accuracy_repeat_t_ci95_low"] is None
                    and row["balanced_accuracy_repeat_t_ci95_high"] is None
                    and row["paired_inference_applicability"] == "N/A"
                    and row["balanced_accuracy_p_value"] is None
                    and row["macro_f1_p_value"] is None
                    for row in detector
                )
            )
            self.assertTrue(
                all(
                    row["participant_cluster_ci95_applicability"] == "available"
                    and row["participant_cluster_bootstrap_resamples"] == 257
                    and row["participant_cluster_bootstrap_seed"] == 7
                    and row["balanced_accuracy_participant_cluster_ci95_low"]
                    <= row["balanced_accuracy"]
                    <= row["balanced_accuracy_participant_cluster_ci95_high"]
                    and row["macro_f1_participant_cluster_ci95_low"]
                    <= row["macro_f1"]
                    <= row["macro_f1_participant_cluster_ci95_high"]
                    and row["sensitivity_participant_cluster_ci95_low"]
                    <= row["sensitivity"]
                    <= row["sensitivity_participant_cluster_ci95_high"]
                    and row["specificity_participant_cluster_ci95_low"]
                    <= row["specificity"]
                    <= row["specificity_participant_cluster_ci95_high"]
                    and row["roc_auc_participant_cluster_ci95_low"]
                    <= row["roc_auc"]
                    <= row["roc_auc_participant_cluster_ci95_high"]
                    and row["roc_auc_participant_cluster_ci95_applicability"]
                    == "available"
                    and row["roc_auc_participant_cluster_ci95_reason"] == ""
                    and row["roc_pr_participant_cluster_ci95_applicability"]
                    == "N/A"
                    and bool(row["roc_pr_participant_cluster_ci95_reason"])
                    for row in detector
                )
            )
            internal_metrics = [
                row for row in detector
                if row["evaluation_scope"] == "source_grouped_oof"
            ]
            transfer_metrics = [
                row for row in detector
                if row["evaluation_scope"] == "frozen_cross_dataset"
            ]
            self.assertTrue(
                all(
                    row["worst_fold_balanced_accuracy_applicability"]
                    == "available"
                    for row in internal_metrics
                )
            )
            self.assertTrue(
                all(
                    row["worst_fold_balanced_accuracy_applicability"] == "N/A"
                    and row["worst_fold_balanced_accuracy_reason"]
                    == "frozen_target_evaluation_has_no_training_fold_axis"
                    for row in transfer_metrics
                )
            )
            conclusions = json.loads(
                (root / "tables/result_conclusions.json").read_text(
                    encoding="utf-8"
                )
            )
            endpoint_conclusions = [
                row for row in conclusions
                if row["angle"].startswith("motion_detector_endpoint::")
            ]
            self.assertEqual(len(endpoint_conclusions), 4)
            self.assertEqual(
                {
                    (
                        row["target_dataset"],
                        row["evaluation_scope"],
                        row["aggregation_level"],
                    )
                    for row in endpoint_conclusions
                },
                {
                    ("frailty29", "source_grouped_oof", "window"),
                    ("frailty29", "source_grouped_oof", "file"),
                    ("ptt22", "frozen_cross_dataset", "window"),
                    ("ptt22", "frozen_cross_dataset", "file"),
                },
            )
            self.assertTrue(
                all(
                    "no within-stratum candidate family" in row["finding"]
                    and "No comparison is made across target datasets" in row["finding"]
                    for row in endpoint_conclusions
                )
            )
            uncertainty_conclusion = next(
                row for row in conclusions
                if row["angle"] == "uncertainty_and_inference"
            )
            self.assertIn("seed=7", uncertainty_conclusion["finding"])
            self.assertIn("resamples=257", uncertainty_conclusion["finding"])
            denoiser_rows = json.loads(
                (root / "tables/denoiser_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                denoiser_rows[0]["denoiser_sd_interpretation"],
                "descriptive_between_subject_variability_not_repeat_training_uncertainty",
            )
            self.assertEqual(denoiser_rows[0]["denoiser_sd_applicability"], "N/A")
            self.assertEqual(
                denoiser_rows[0]["repeat_uncertainty_applicability"], "N/A"
            )
            output_status = json.loads(
                (root / "tables/reporter_output_status.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(output_status)
            self.assertTrue(
                all(row["status"] in {"generated", "N/A"} for row in output_status)
            )
            self.assertTrue(
                all(row["reason"] for row in output_status if row["status"] == "N/A")
            )
            obsolete_status = [
                row
                for row in output_status
                if row["output_id"]
                in {
                    "motion_internal_subject_confusion_matrix",
                    "motion_ptt_subject_confusion_matrix",
                }
            ]
            self.assertEqual(len(obsolete_status), 2)
            self.assertTrue(
                all(
                    row["status"] == "N/A"
                    and row["reason"]
                    == "superseded_by_window_and_file_level_contract"
                    and row["replacement_status"] == "generated"
                    for row in obsolete_status
                )
            )
            self.assertFalse(
                (root / "figures/motion_internal_subject_confusion_matrix.png").exists()
            )
            self.assertFalse(
                (root / "figures/motion_ptt_subject_confusion_matrix.png").exists()
            )
            self.assertEqual(
                {
                    row["status"]
                    for row in output_status
                    if "reporter_profile:" in row["required_by"]
                },
                {"generated"},
            )
            reproducibility = json.loads(
                (root / "tables/reproducibility_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            bootstrap_audit = next(
                row for row in reproducibility
                if row["evidence_scope"] == "participant_cluster_bootstrap_report"
            )
            self.assertIsNone(bootstrap_audit["repeat_count"])
            self.assertEqual(bootstrap_audit["resample_count"], 257)
            self.assertIsNone(bootstrap_audit["split_seed"])
            self.assertEqual(bootstrap_audit["resampling_seed"], 7)
            file_confusion = json.loads(
                (root / "tables/motion_detector_file_confusion.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(file_confusion), 2)
            self.assertEqual(
                {row["model_id"] for row in file_confusion},
                {"frailty29_trained_motion_detector"},
            )
            self.assertEqual(
                file_confusion[0]["aggregation_level"],
                "file_median_window_probability",
            )
            self.assertTrue(
                (root / "figures/motion_internal_file_confusion_matrix.png").is_file()
            )
            self.assertTrue(
                (root / "figures/frailty29_trained_window_score_distribution.png").is_file()
            )
            self.assertTrue(
                (root / "figures/frailty29_trained_file_score_distribution.png").is_file()
            )
            self.assertTrue(
                (root / "figures/frailty29_trained_window_prediction_tsne.png").is_file()
            )
            self.assertTrue(
                (root / "figures/frailty29_trained_file_prediction_tsne.png").is_file()
            )
            self.assertTrue(
                (root / "figures/frailty29_trained_window_roc_auc_curve.png").is_file()
            )
            self.assertTrue(
                (root / "figures/frailty29_trained_file_roc_auc_curve.png").is_file()
            )
            roc_rows = json.loads(
                (root / "tables/motion_detector_roc_curves.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(roc_rows)
            self.assertEqual(
                {row["aggregation_level"] for row in roc_rows},
                {"window", "file"},
            )
            self.assertTrue(
                all("false_positive_rate" in row for row in roc_rows)
            )
            per_class_rows = json.loads(
                (root / "tables/motion_detector_per_class_results.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(per_class_rows), 8)
            self.assertEqual(
                {row["class_name"] for row in per_class_rows},
                {"static", "motion"},
            )
            self.assertEqual(
                {row["aggregation_level"] for row in per_class_rows},
                {"window", "file"},
            )
            self.assertTrue(
                all(
                    row["prediction_rule_source"]
                    == "normalized_predicted_label_preserves_frozen_threshold"
                    for row in per_class_rows
                )
            )
            self.assertTrue(
                (root / "tables/TABLE_COLUMN_DEFINITIONS.md").is_file()
            )
            tsne_rows = json.loads(
                (root / "tables/motion_detector_prediction_tsne.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(tsne_rows)
            self.assertEqual(
                {row["tsne_input_space"] for row in tsne_rows},
                {"persisted_prediction_probability_vector"},
            )
            index = json.loads(
                (root / "outputs_index.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("outputs_index.json", {row["path"] for row in index["files"]})
            for row in index["files"]:
                indexed_file = root / row["path"]
                self.assertEqual(
                    hashlib.sha256(indexed_file.read_bytes()).hexdigest(), row["sha256"]
                )

    def test_denoiser_tables_sort_by_rmse_then_f1_and_mark_optima(self) -> None:
        rows = [
            {
                "algorithm_or_reducer": "a",
                "activity_group": "static",
                "channel": "RED",
                "participant_macro_f1": 0.95,
                "participant_macro_ibi_ppi_rmse_ms": 10.0,
            },
            {
                "algorithm_or_reducer": "b",
                "activity_group": "static",
                "channel": "RED",
                "participant_macro_f1": 0.9,
                "participant_macro_ibi_ppi_rmse_ms": 8.0,
            },
            {
                "algorithm_or_reducer": "c",
                "activity_group": "dynamic",
                "channel": "IR",
                "participant_macro_f1": 0.95,
                "participant_macro_ibi_ppi_rmse_ms": 4.0,
            },
        ]
        numeric, display = _rank_and_mark_denoiser_rows(rows, "static")
        self.assertEqual([row["algorithm_or_reducer"] for row in numeric], ["b", "a"])
        self.assertEqual(display[0]["algorithm_or_reducer"], "b*")
        self.assertEqual(display[0]["activity_group"], "static")
        self.assertEqual(
            display[0]["participant_macro_ibi_ppi_rmse_ms"], "8.0*"
        )
        self.assertEqual(display[1]["algorithm_or_reducer"], "a*")
        self.assertEqual(display[1]["participant_macro_f1"], "0.95*")
        _, single_display = _rank_and_mark_denoiser_rows([rows[0]], "static")
        self.assertEqual(single_display[0]["algorithm_or_reducer"], "a**")

    def test_denoiser_activity_table_has_five_columns_rmse_order_and_marks(self) -> None:
        summary = [
            {
                "algorithm_or_reducer": "identity",
                "activity_group": "static",
                "channel": "RED",
                "participant_macro_f1": 0.80,
                "participant_macro_f1_sd": 0.03,
                "participant_macro_ibi_ppi_rmse_ms": 5.0,
                "participant_macro_ibi_ppi_rmse_ms_sd": 1.5,
            },
            {
                "algorithm_or_reducer": "fastica_bss",
                "activity_group": "static",
                "channel": "IR",
                "participant_macro_f1": 0.70,
                "participant_macro_f1_sd": 0.02,
                "participant_macro_ibi_ppi_rmse_ms": 4.0,
                "participant_macro_ibi_ppi_rmse_ms_sd": 1.0,
            },
            {
                "algorithm_or_reducer": "pca_bss",
                "activity_group": "static",
                "channel": "IR",
                "participant_macro_f1": 0.95,
                "participant_macro_f1_sd": 0.01,
                "participant_macro_ibi_ppi_rmse_ms": 6.0,
                "participant_macro_ibi_ppi_rmse_ms_sd": 2.0,
            },
        ]
        inference = [
            {
                "candidate_denoiser": candidate,
                "activity_group": "static",
                "channel": "IR",
                "metric": "participant_macro_ibi_ppi_rmse_ms",
                "holm_adjusted_p_value": p_value,
            }
            for candidate, p_value in (("fastica_bss", 0.01), ("pca_bss", 0.20))
        ]
        rows = _denoiser_activity_result_rows(
            summary,
            inference,
            activity_group="static",
            reference_id="identity",
        )
        self.assertTrue(all(len(row) == 5 for row in rows))
        self.assertEqual(
            [row["denoiser"] for row in rows],
            ["fastica_bss*", "identity", "pca_bss*"],
        )
        self.assertEqual(rows[0]["RMSE ± SD (ms)"], "4.0 ± 1.0*")
        self.assertEqual(rows[2]["F1 ± SD (%)"], "95.0 ± 1.0*")
        self.assertEqual(rows[0]["RMSE P versus identity"], "0.0100")
        self.assertEqual(rows[1]["RMSE P versus identity"], "Reference")

        single = _denoiser_activity_result_rows(
            summary[:1],
            (),
            activity_group="static",
            reference_id="identity",
        )
        self.assertEqual(single[0]["denoiser"], "identity**")
        self.assertTrue(single[0]["RMSE ± SD (ms)"].endswith("*"))
        self.assertTrue(single[0]["F1 ± SD (%)"].endswith("*"))

    def test_denoiser_sd_is_explicitly_between_subject_sample_sd(self) -> None:
        rows = _annotate_denoiser_uncertainty([
            {
                "algorithm_or_reducer": "pca_bss",
                "activity_group": "dynamic",
                "channel": "IR",
                "participant_macro_f1": 0.9,
                "participant_macro_f1_sd": 0.05,
            }
        ])
        self.assertEqual(
            rows[0]["denoiser_sd_estimator"],
            "between_subject_sample_sd_ddof1",
        )
        self.assertEqual(rows[0]["denoiser_sd_applicability"], "available")
        self.assertEqual(rows[0]["denoiser_sd_unit"], "subject")
        self.assertEqual(rows[0]["repeat_uncertainty_applicability"], "N/A")

    def test_denoiser_subject_macro_pools_counts_and_interval_sse(self) -> None:
        def passed(
            participant_id: str,
            *,
            true_positive: int,
            false_positive: int,
            false_negative: int,
            matched_intervals: int,
            rmse_ms: float,
            runtime_s: float,
        ) -> dict[str, object]:
            return {
                "participant_id": participant_id,
                "algorithm_or_reducer": "pca_bss",
                "activity_group": "dynamic",
                "channel": "RED",
                "status": "passed",
                "true_positives": true_positive,
                "false_positives": false_positive,
                "false_negatives": false_negative,
                "matched_interval_count": matched_intervals,
                "ibi_ppi_rmse_ms": rmse_ms,
                "runtime_s": runtime_s,
            }

        def failed(participant_id: str, runtime_s: float) -> dict[str, object]:
            return {
                "participant_id": participant_id,
                "algorithm_or_reducer": "pca_bss",
                "activity_group": "dynamic",
                "channel": "RED",
                "status": "failed",
                "runtime_s": runtime_s,
            }

        rows = [
            passed(
                "p1",
                true_positive=8,
                false_positive=2,
                false_negative=2,
                matched_intervals=4,
                rmse_ms=10.0,
                runtime_s=1.0,
            ),
            passed(
                "p1",
                true_positive=2,
                false_positive=0,
                false_negative=8,
                matched_intervals=1,
                rmse_ms=20.0,
                runtime_s=2.0,
            ),
            passed(
                "p2",
                true_positive=3,
                false_positive=1,
                false_negative=1,
                matched_intervals=4,
                rmse_ms=30.0,
                runtime_s=3.0,
            ),
            failed("p2", 4.0),
            failed("p3", 5.0),
        ]
        summary = _aggregate_benchmark(rows)
        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertEqual(row["attempted_participant_count"], 3)
        self.assertEqual(row["passed_participant_count"], 2)
        self.assertEqual(row["failed_participant_count"], 2)
        self.assertEqual(row["all_failed_participant_count"], 1)
        self.assertEqual(row["partially_failed_participant_count"], 1)
        self.assertAlmostEqual(row["participant_coverage_rate"], 2.0 / 3.0)
        self.assertEqual(row["attempted_segment_count"], 5)
        self.assertEqual(row["passed_segment_count"], 3)
        self.assertEqual(row["failed_segment_count"], 2)
        self.assertAlmostEqual(row["segment_coverage_rate"], 3.0 / 5.0)
        self.assertEqual(row["participant_count"], 2)
        self.assertEqual(row["segment_count"], 3)
        self.assertAlmostEqual(row["participant_macro_f1"], 0.6875)
        self.assertAlmostEqual(row["participant_macro_sensitivity"], 0.625)
        self.assertAlmostEqual(
            row["participant_macro_positive_predictive_value"],
            (10.0 / 12.0 + 3.0 / 4.0) / 2.0,
        )
        p1_rmse = np.sqrt((10.0**2 * 4 + 20.0**2) / 5.0)
        self.assertAlmostEqual(
            row["participant_macro_ibi_ppi_rmse_ms"],
            (p1_rmse + 30.0) / 2.0,
        )
        self.assertEqual(row["rmse_evaluable_participant_count"], 2)
        self.assertEqual(row["rmse_evaluable_segment_count"], 3)
        self.assertEqual(row["matched_interval_count"], 9)
        self.assertEqual(row["passed_runtime_s"], 6.0)
        self.assertEqual(row["failed_runtime_s"], 9.0)
        self.assertEqual(row["total_runtime_s"], 15.0)

    def test_denoiser_all_failed_group_is_na_and_sorts_last(self) -> None:
        rows = [
            {
                "participant_id": "p1",
                "algorithm_or_reducer": "good",
                "activity_group": "static",
                "channel": "RED",
                "status": "passed",
                "true_positives": 4,
                "false_positives": 1,
                "false_negatives": 1,
                "matched_interval_count": 3,
                "ibi_ppi_rmse_ms": 12.0,
                "runtime_s": 1.0,
            },
            {
                "participant_id": "p1",
                "algorithm_or_reducer": "all_failed",
                "activity_group": "static",
                "channel": "RED",
                "status": "failed",
                "runtime_s": 2.0,
            },
            {
                "participant_id": "p2",
                "algorithm_or_reducer": "all_failed",
                "activity_group": "static",
                "channel": "RED",
                "status": "failed",
                "runtime_s": 3.0,
            },
        ]
        summary = _aggregate_benchmark(rows)
        by_algorithm = {
            row["algorithm_or_reducer"]: row for row in summary
        }
        failed = by_algorithm["all_failed"]
        self.assertEqual(failed["attempted_participant_count"], 2)
        self.assertEqual(failed["all_failed_participant_count"], 2)
        self.assertEqual(failed["participant_coverage_rate"], 0.0)
        self.assertEqual(failed["segment_coverage_rate"], 0.0)
        self.assertIsNone(failed["participant_macro_f1"])
        self.assertIsNone(failed["participant_macro_ibi_ppi_rmse_ms"])
        numeric, display = _rank_and_mark_denoiser_rows(summary, "static")
        self.assertEqual(
            [row["algorithm_or_reducer"] for row in numeric],
            ["good", "all_failed"],
        )
        self.assertEqual(display[-1]["activity_group"], "static")

    def test_denoiser_aggregation_rejects_inconsistent_passed_evidence(self) -> None:
        row = {
            "participant_id": "p1",
            "algorithm_or_reducer": "pca_bss",
            "activity_group": "static",
            "channel": "RED",
            "status": "passed",
            "true_positives": 4,
            "false_positives": 1,
            "false_negatives": 1,
            "nref": 99,
            "matched_interval_count": 3,
            "ibi_ppi_rmse_ms": 12.0,
            "runtime_s": 1.0,
        }
        with self.assertRaisesRegex(ValueError, "nref disagrees"):
            _aggregate_benchmark([row])

    def test_file_rows_use_median_probability_and_frozen_threshold(
        self,
    ) -> None:
        rows = [
            {
                "participant_id": "p1", "file_id": "p1_static", "activity_label": 0,
                "p_active": probability, "threshold": 0.5,
            }
            for probability in (0.1, 0.2, 0.9)
        ] + [
            {
                "participant_id": "p1", "file_id": "p1_motion", "activity_label": 1,
                "p_active": probability, "threshold": 0.5,
            }
            for probability in (0.6, 0.8)
        ]
        file_rows = _file_prediction_rows(rows)
        self.assertEqual(len(file_rows), 2)
        by_id = {row["file_id"]: row for row in file_rows}
        self.assertEqual(by_id["p1_static"]["p_active"], 0.2)
        self.assertEqual(by_id["p1_static"]["predicted_activity"], 0)
        self.assertEqual(by_id["p1_motion"]["p_active"], 0.7)
        self.assertEqual(by_id["p1_motion"]["predicted_activity"], 1)
        mean_rows = _file_prediction_rows(rows, score_aggregation="mean")
        mean_by_id = {row["file_id"]: row for row in mean_rows}
        self.assertAlmostEqual(mean_by_id["p1_static"]["p_active"], 0.4)
        self.assertEqual(mean_by_id["p1_static"]["score_aggregation"], "mean")

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow not installed")
    def test_reverse_ablation_trains_five_ptt_folds_and_one_final_model(self) -> None:
        split_path = PIPELINE_ROOT / "splits/ptt_formal_repeated_grouped_5x5_v2.csv"
        split_rows = load_formal_ptt_repeated_folds(split_path)
        participants = sorted({str(row["subject_id"]) for row in split_rows})
        examples = tuple(
            MotionWindowExample(
                window_id=f"{participant}-{activity}",
                participant_id=participant,
                file_id=f"{participant}-{activity}",
                role_or_activity=activity,
                activity_label=label,
                values=np.asarray([probability], dtype=np.float32),
                dataset_id="ptt_ppg_1_1_0_local",
            )
            for participant in participants
            for activity, label, probability in (
                ("sit", 0, 0.1),
                ("walk", 1, 0.9),
            )
        )
        contexts: list[MotionFitContext] = []

        def fit_model(rows, context):
            contexts.append(context)
            context.artifact_directory.mkdir(parents=True, exist_ok=True)
            model_path = context.artifact_directory / "model.mock"
            model_path.write_bytes(
                f"{context.fold_index}:{context.final_fit}".encode("ascii")
            )
            return MotionFittedArtifact(
                runtime_model=object(),
                model_id=FORMAL_MOTION_MODEL_ID,
                artifact_path=str(model_path),
                artifact_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                model_input_schema_sha256=MOTION_NETWORK_SCHEMA_SHA256,
                training_participant_ids=tuple(context.training_participant_ids),
                parameter_count=1,
                inference_cost={
                    "device": "mock",
                    "batch_size": 1,
                    "window_samples": 3200,
                    "warmup_iterations": 1,
                    "timed_iterations": 1,
                    "latency_ms_per_window_p50": 1.0,
                    "latency_ms_per_window_p95": 1.0,
                    "throughput_windows_per_second": 1.0,
                },
            )

        def predict_probability(runtime, rows):
            del runtime
            return [float(np.asarray(row.values).reshape(-1)[0]) for row in rows]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schema_path, schema_file_sha = write_formal_motion_input_schema(
                root / "formal_motion_input_schema.json"
            )
            with patch(
                "ppg_frailty.quality.motion_reference.verify_formal_ptt_source_evidence",
                return_value=(),
            ):
                result = _run_ptt_motion_training_ablation_impl(
                    examples,
                    ptt_split_csv=split_path,
                    fit_model=fit_model,
                    predict_probability=predict_probability,
                    model_input_schema_path=schema_path,
                    expected_model_input_schema_sha256=schema_file_sha,
                    output_dir=root / "run",
                    formal_source_evidence={"synthetic": True},
                )
            self.assertEqual(len(contexts), 6)
            self.assertTrue(all(row.training_dataset_kind == "ptt" for row in contexts))
            self.assertEqual(sum(row.final_fit for row in contexts), 1)
            self.assertEqual(len(result.oof_rows), 44)
            self.assertEqual(
                result.evidence["deployment_threshold"]["score_origin"],
                "strict_outer_oof_model_predictions_only",
            )
            self.assertEqual(
                result.evidence["major_metrics"]["participant_macro_roc_auc"],
                1.0,
            )
            internal_jobs = load_motion_fold_jobs(
                PIPELINE_ROOT / "splits/sgkf5_seed42_v2.csv"
            )
            internal_participants = sorted(
                set(internal_jobs[0].train_participant_ids)
                | set(internal_jobs[0].oof_participant_ids)
            )
            internal_examples = tuple(
                MotionWindowExample(
                    window_id=f"{participant}-{role}",
                    participant_id=participant,
                    file_id=f"{participant}-{role}",
                    role_or_activity=role,
                    activity_label=label,
                    values=np.asarray([probability], dtype=np.float32),
                    dataset_id="frailty29",
                )
                for participant in internal_participants
                for role, label, probability in (
                    ("B", 0, 0.1),
                    ("S1", 1, 0.9),
                )
            )
            with patch(
                "ppg_frailty.quality.motion_reference.verify_formal_internal_source_evidence",
                return_value=(),
            ), patch(
                "ppg_frailty.quality.motion_reference.verify_formal_ptt_source_evidence",
                return_value=(),
            ):
                reverse = _run_internal_reverse_evaluation_impl(
                    internal_examples,
                    ptt_training_evidence_path=result.evidence_path,
                    expected_ptt_training_evidence_sha256=result.evidence_sha256,
                    internal_fold_jobs=internal_jobs,
                    load_frozen_model=lambda path, metadata: object(),
                    predict_probability=predict_probability,
                    output_dir=root / "reverse",
                    formal_source_evidence={"synthetic": True},
                )
            self.assertEqual(len(reverse.prediction_rows), 58)
            self.assertFalse(reverse.report["fit_or_recalibration_performed"])
            self.assertTrue(reverse.report["ablation_executed"])
            self.assertEqual(
                reverse.report["major_metrics"]["participant_macro_balanced_accuracy"],
                1.0,
            )

    def test_resume_preserves_partial_stage_and_allocates_new_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            partial = root / "motion_internal"
            partial.mkdir()
            (partial / "repeat_0_fold_0_partial.pt").write_bytes(b"partial")
            attempt, complete = _stage_directory(
                root, {"stages": {}}, "internal_motion_oof", "motion_internal",
                "motion_internal_evidence.json",
            )
            self.assertFalse(complete)
            self.assertEqual(attempt.name, "motion_internal_attempt_002")
            self.assertTrue((partial / "repeat_0_fold_0_partial.pt").is_file())
            attempt.mkdir()
            (attempt / "motion_internal_evidence.json").write_text("{}", encoding="utf-8")
            resolved, complete = _stage_directory(
                root,
                {"stages": {"internal_motion_oof": {"artifact_dir": attempt.name}}},
                "internal_motion_oof", "motion_internal", "motion_internal_evidence.json",
            )
            self.assertTrue(complete)
            self.assertEqual(resolved, attempt)


if __name__ == "__main__":
    unittest.main()
