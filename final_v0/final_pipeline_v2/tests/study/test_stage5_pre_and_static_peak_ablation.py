from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ppg_frailty.quality.stage5_pre import (
    PEAK_ABLATION_SCHEMA,
    STAGE5_SCHEMA,
    _stage_directory,
    align_and_score_beats,
    detect_msptdfast_v2,
    generate_motion_peak_report,
    load_motion_peak_plan,
)
from ppg_frailty.quality.motion_adapters import fit_formal_motion_model
from ppg_frailty.quality.motion_runner import MotionFitContext, MotionWindowExample
from ppg_frailty.representations.motion import MOTION_NETWORK_SCHEMA_SHA256


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PIPELINE_ROOT.parents[1]
PLAN_ROOT = PIPELINE_ROOT / "configs/studies/static_line_b_staged_v2"


class Stage5PreAndStaticPeakAblationTests(unittest.TestCase):
    def test_stage5_plan_is_complete_ptt_and_uses_only_real_reducers(self) -> None:
        plan = load_motion_peak_plan(PLAN_ROOT / "stage5_pre.yaml")
        self.assertEqual(plan.schema_version, STAGE5_SCHEMA)
        self.assertEqual(plan.payload["ptt_dataset"]["record_count"], 66)
        self.assertEqual(plan.payload["motion_detector"]["internal_participant_count"], 29)
        self.assertEqual(
            plan.payload["motion_detector"]["split"]["method"],
            "StratifiedGroupKFold",
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
            {"aboy_project_v1", "msptdfast_v2_3_python_port"},
        )

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
            fit_formal_motion_model(examples, context)
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
                {"activity_label": 0, "predicted_activity": 0},
                {"activity_label": 0, "predicted_activity": 1},
                {"activity_label": 1, "predicted_activity": 1},
                {"activity_label": 1, "predicted_activity": 0},
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
            self.assertEqual(result["figure_count"], 6)
            self.assertTrue((root / "figures/motion_training_learning_curves.png").is_file())
            self.assertTrue((root / "result_backup/backup_manifest.json").is_file())
            self.assertIn(
                "<table>",
                (root / "STUDY_SUMMARY.html").read_text(encoding="utf-8"),
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
