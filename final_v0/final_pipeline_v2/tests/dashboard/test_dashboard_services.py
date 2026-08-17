from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
import zipfile

import numpy as np

from ppg_frailty.dashboard.downloads import preview_csv_bytes, study_zip_bytes
from ppg_frailty.dashboard.app import _control_study_job, create_app
from ppg_frailty.dashboard.job_manager import StudyJobManager
from ppg_frailty.dashboard.preview_service import PipelinePreviewService


class DashboardServiceTests(unittest.TestCase):
    def test_stop_without_active_job_never_starts_a_study(self) -> None:
        class FakeJobs:
            def __init__(self) -> None:
                self.started: list[list[str]] = []

            def start(self, arguments):
                self.started.append(list(arguments))
                return "unexpected"

            def terminate(self, job_id):
                raise AssertionError(f"unexpected terminate: {job_id}")

        jobs = FakeJobs()
        active, disabled, message = _control_study_job(
            jobs,
            trigger="stop-job-button",
            study_plan="configs/studies/single_config_v2.yaml",
            jobs_value=1,
            resume_directory=None,
            arguments=None,
            job_id=None,
        )
        self.assertIsNone(active)
        self.assertTrue(disabled)
        self.assertIn("No active", message)
        self.assertEqual(jobs.started, [])

    def test_start_refuses_to_replace_a_running_job(self) -> None:
        class FakeJobs:
            def __init__(self) -> None:
                self.started: list[list[str]] = []

            def status(self, job_id):
                return {"job_id": job_id, "state": "running"}

            def start(self, arguments):
                self.started.append(list(arguments))
                return "unexpected"

        jobs = FakeJobs()
        active, disabled, message = _control_study_job(
            jobs,
            trigger="start-job-button",
            study_plan="configs/studies/single_config_v2.yaml",
            jobs_value=2,
            resume_directory=None,
            arguments=None,
            job_id="existing-job",
        )
        self.assertEqual(active, "existing-job")
        self.assertFalse(disabled)
        self.assertIn("already running", message)
        self.assertEqual(jobs.started, [])

    def test_preview_csv_has_synchronized_columns(self) -> None:
        payload = preview_csv_bytes(
            np.asarray([0.0, 0.5]),
            {"red": np.asarray([1.0, 2.0]), "ir": np.asarray([3.0, 4.0])},
        ).decode("utf-8")
        self.assertEqual(payload.splitlines()[0], "time_s,red,ir")
        self.assertEqual(len(payload.splitlines()), 3)

    def test_study_zip_is_confined_and_contains_relative_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            studies = Path(temporary) / "studies"
            study = studies / "20260817_ablation_epoch"
            study.mkdir(parents=True)
            (study / "STUDY_SUMMARY.md").write_text("# test\n", encoding="utf-8")
            data = study_zip_bytes(study, studies_root=studies)
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                self.assertEqual(
                    archive.namelist(),
                    ["20260817_ablation_epoch/STUDY_SUMMARY.md"],
                )
            with self.assertRaises(ValueError):
                study_zip_bytes(Path(temporary), studies_root=studies)

    def test_completed_study_figures_are_confined_and_embeddable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study = root / "artifacts" / "studies" / "20260817_grid_test"
            figure = study / "figures" / "leaderboard.png"
            figure.parent.mkdir(parents=True)
            figure.write_bytes(b"synthetic-png-fixture")
            service = PipelinePreviewService(root)
            relative_study = study.relative_to(root)
            self.assertEqual(
                service.study_figure_paths(relative_study),
                ("figures/leaderboard.png",),
            )
            self.assertTrue(
                service.study_figure_data_uri(
                    relative_study,
                    "figures/leaderboard.png",
                ).startswith("data:image/png;base64,")
            )
            with self.assertRaises(ValueError):
                service.study_figure_data_uri(relative_study, "../../outside.png")

    def test_nested_dashboard_study_is_discovered_by_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study = (
                root
                / "artifacts"
                / "studies"
                / "dashboard_job_123"
                / "20260817_ablation_test"
            )
            study.mkdir(parents=True)
            (study / "study_manifest.json").write_text("{}\n", encoding="utf-8")
            self.assertEqual(
                PipelinePreviewService(root).study_directories(),
                (study.resolve(),),
            )

    def test_service_discovers_configs_without_loading_data(self) -> None:
        root = Path(__file__).resolve().parents[2]
        service = PipelinePreviewService(root)
        names = {path.name for path in service.config_paths()}
        self.assertEqual(
            names,
            {
                "reference_static_role_aware_v2.yaml",
                "reference_static_feature_vector_v2.yaml",
                "reference_static_feature_matrix_v2.yaml",
                "reference_static_fusion_v2.yaml",
            },
        )
        self.assertNotIn("formal_ablation_profiles_v2.yaml", names)
        self.assertNotIn("formal_experiment_catalog_v2.yaml", names)
        self.assertIn(
            "configs/studies/single_config_v2.yaml",
            service.study_plan_paths(),
        )

    def test_dash_application_constructs_without_starting_a_server(self) -> None:
        root = Path(__file__).resolve().parents[2]
        app = create_app(root)
        response = app.server.test_client().get("/")
        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(len(app.callback_map), 10)

    def test_job_manager_reads_latest_structured_progress_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "artifacts" / "studies" / "20260817_grid_test"
            output.mkdir(parents=True)
            events = (
                {"event": "case_started", "current": 0, "total": 2}
                ,
                {"event": "case_finished", "current": 1, "total": 2}
            )
            (output / "progress_events.jsonl").write_text(
                "\n".join(json.dumps(value) for value in events) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                StudyJobManager(root)._latest_progress(),
                events[-1],
            )


if __name__ == "__main__":
    unittest.main()
