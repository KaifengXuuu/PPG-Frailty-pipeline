from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

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
    _stage_directory,
    _rank_and_mark_denoiser_rows,
    _subject_activity_prediction_rows,
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
            {"aboy_project_v1", ABOY_V2_ID, "msptdfast_v2_3_python_port"},
        )
        implementations = {
            row["algorithm_id"]: row["implementation"]
            for row in plan.payload["algorithms"]
        }
        self.assertEqual(
            implementations[MSPTDFAST_V2_ID], MSPTDFAST_IMPLEMENTATION_PATH
        )
        self.assertEqual(implementations[ABOY_V2_ID], ABOY_V2_IMPLEMENTATION_PATH)
        self.assertEqual(
            plan.payload["detector_input"],
            "repaired_native_ppg_each_registered_module_owns_preprocessing",
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
            with self.assertRaisesRegex(ValueError, "registered MSPTDfast module"):
                load_motion_peak_plan(copied)

    def test_delay_shift_does_not_change_interval_error(self) -> None:
        reference = np.arange(0.5, 20.0, 0.8)
        predicted = reference + 0.34
        result = align_and_score_beats(reference, predicted)
        # The official tie rule chooses the smallest absolute lag that still
        # places every beat inside the 0.2-s tolerance, not the physical delay.
        self.assertAlmostEqual(result["lag_s"], 0.16, places=8)
        self.assertAlmostEqual(result["f1"], 1.0)
        self.assertAlmostEqual(result["ibi_ppi_rmse_ms"], 0.0, places=9)

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
            for algorithm, f1, rmse in (
                ("aboy_project_v1", 0.90, 20.0),
                (ABOY_V2_ID, 0.91, 19.0),
                ("msptdfast_v2_3_python_port", 0.92, 18.0),
            ):
                for channel in ("RED", "IR"):
                    rows.append({
                        "algorithm_or_reducer": algorithm,
                        "activity_group": "static",
                        "channel": channel,
                        "participant_count": 22,
                        "segment_count": 22,
                        "participant_macro_f1": f1,
                        "participant_macro_ibi_ppi_rmse_ms": rmse,
                        "total_runtime_s": 1.0,
                    })
            (root / "static_peak_ablation.json").write_text(
                json.dumps({"summary_rows": rows}), encoding="utf-8"
            )
            result = generate_motion_peak_report(root)
            self.assertEqual(result["figure_count"], 3)
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
            (root / "resolved_plan.yaml").write_text(
                plan.path.read_text(encoding="utf-8"), encoding="utf-8"
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
                    "participant_id": "p1", "activity_label": 0,
                    "p_active": 0.1, "threshold": 0.5, "predicted_activity": 0,
                },
                {
                    "participant_id": "p1", "activity_label": 1,
                    "p_active": 0.9, "threshold": 0.5, "predicted_activity": 1,
                },
                {
                    "participant_id": "p2", "activity_label": 0,
                    "p_active": 0.8, "threshold": 0.5, "predicted_activity": 1,
                },
                {
                    "participant_id": "p2", "activity_label": 1,
                    "p_active": 0.2, "threshold": 0.5, "predicted_activity": 0,
                },
            ]
            pq.write_table(
                pa.Table.from_pylist(prediction_rows),
                internal / "motion_window_oof.parquet",
            )
            pq.write_table(
                pa.Table.from_pylist(prediction_rows),
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
            summary = [{
                "algorithm_or_reducer": "identity",
                "activity_group": "dynamic",
                "channel": "RED",
                "participant_count": 22,
                "segment_count": 44,
                "participant_macro_f1": 0.8,
                "participant_macro_ibi_ppi_rmse_ms": 30.0,
                "total_runtime_s": 2.0,
            }]
            (denoiser / "denoiser_benchmark.json").write_text(
                json.dumps({"summary_rows": summary}), encoding="utf-8"
            )
            result = generate_motion_peak_report(root)
            self.assertEqual(result["figure_count"], 8)
            generate_motion_peak_report(root)
            self.assertTrue((root / "figures/motion_training_learning_curves.png").is_file())
            self.assertTrue((root / "result_backup/backup_manifest.json").is_file())
            self.assertIn(
                "<table>",
                (root / "STUDY_SUMMARY.html").read_text(encoding="utf-8"),
            )
            detector = json.loads(
                (root / "tables/motion_detector_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(detector), 2)
            self.assertEqual(
                set(detector[0]),
                {
                    "dataset",
                    "participant_macro_balanced_accuracy",
                    "participant_macro_f1",
                    "participant_macro_sensitivity",
                    "participant_macro_specificity",
                    "participant_macro_roc_auc",
                    "participant_macro_pr_auc",
                    "worst_fold_balanced_accuracy",
                },
            )
            subject_confusion = json.loads(
                (root / "tables/motion_detector_subject_confusion.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(subject_confusion), 2)
            self.assertEqual(
                subject_confusion[0]["aggregation_level"],
                "participant_by_activity_class_median_probability",
            )
            self.assertEqual(subject_confusion[0]["participant_count"], 2)
            self.assertEqual(subject_confusion[0]["participant_activity_class_count"], 4)
            self.assertTrue(
                (root / "figures/motion_internal_subject_confusion_matrix.png").is_file()
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
        self.assertEqual(display[0]["activity_group"], "static*")
        self.assertEqual(
            display[0]["participant_macro_ibi_ppi_rmse_ms"], "8.0*"
        )
        self.assertEqual(display[1]["activity_group"], "static*")
        self.assertEqual(display[1]["participant_macro_f1"], "0.95*")
        _, single_display = _rank_and_mark_denoiser_rows([rows[0]], "static")
        self.assertEqual(single_display[0]["activity_group"], "static**")

    def test_subject_activity_rows_use_median_probability_and_frozen_threshold(
        self,
    ) -> None:
        rows = [
            {
                "participant_id": "p1", "activity_label": 0,
                "p_active": probability, "threshold": 0.5,
            }
            for probability in (0.1, 0.2, 0.9)
        ] + [
            {
                "participant_id": "p1", "activity_label": 1,
                "p_active": probability, "threshold": 0.5,
            }
            for probability in (0.6, 0.8)
        ]
        subject_rows = _subject_activity_prediction_rows(rows)
        self.assertEqual(len(subject_rows), 2)
        self.assertEqual(subject_rows[0]["median_p_active"], 0.2)
        self.assertEqual(subject_rows[0]["predicted_activity"], 0)
        self.assertEqual(subject_rows[1]["median_p_active"], 0.7)
        self.assertEqual(subject_rows[1]["predicted_activity"], 1)

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
