from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ppg_frailty.provenance import sha256_file
from ppg_frailty.quality.stage5_pre import (
    _stage_progress,
    run_motion_peak_study,
)
from ppg_frailty.study import ProgressEvent, TerminalProgressSink
from motion_peak_studies_v2 import build_parser


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
PLAN_ROOT = PIPELINE_ROOT / "configs/studies/static_line_b_staged_v2"


class Stage5PreProgressTests(unittest.TestCase):
    def test_cli_can_disable_only_the_denoiser_stage(self) -> None:
        args = build_parser().parse_args(
            ["run", "--plan", "stage5_pre.yaml", "--no-denoiser"]
        )
        self.assertTrue(args.no_denoiser)
        self.assertEqual(args.command, "run")

    def test_existing_terminal_sink_renders_total_and_subtask_without_ansi(self) -> None:
        stream = io.StringIO()
        sink = TerminalProgressSink(stream=stream, refresh_interval=0.0)
        sink(ProgressEvent(event="start", current=0, total=7, message="Stage5-pre"))
        _stage_progress(sink, 0, 7, "internal motion OOF")(
            3, 29, "preprocess participant p03"
        )
        sink.close()

        rendered = stream.getvalue()
        self.assertIn("TOTAL", rendered)
        self.assertIn("SUB", rendered)
        self.assertIn("elapsed", rendered)
        self.assertIn("ETA~", rendered)
        self.assertIn("3/29", rendered)
        self.assertNotIn("\x1b", rendered)

    def test_stage5_orchestration_relays_runtime_discovered_subtask_counts(self) -> None:
        events: list[ProgressEvent] = []

        observed_training_devices: list[str] = []

        def internal(
            repository: Path,
            *,
            output_dir: Path,
            progress_callback,
            training_device: str,
        ):
            del repository
            observed_training_devices.append(training_device)
            progress_callback(7, 29, "internal participant")
            evidence = output_dir / "motion_internal_evidence.json"
            evidence.parent.mkdir(parents=True, exist_ok=True)
            evidence.write_text("{}", encoding="utf-8")
            return SimpleNamespace(evidence_sha256=sha256_file(evidence))

        def external(repository: Path, *, output_dir: Path, progress_callback, **kwargs):
            del repository, kwargs
            progress_callback(11, 22, "PTT participant")
            output_dir.mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(report_sha256="e" * 64)

        def ptt_training(
            repository: Path, *, output_dir: Path, progress_callback, **kwargs
        ):
            del repository, kwargs
            progress_callback(5, 6, "PTT fold")
            output_dir.mkdir(parents=True, exist_ok=True)
            evidence = output_dir / "motion_ptt_training_evidence.json"
            evidence.write_text("{}", encoding="utf-8")
            return SimpleNamespace(evidence_sha256=sha256_file(evidence))

        def reverse(repository: Path, *, output_dir: Path, progress_callback, **kwargs):
            del repository, kwargs
            progress_callback(17, 29, "reverse participant")
            output_dir.mkdir(parents=True, exist_ok=True)
            report = output_dir / "motion_internal_reverse_evaluation_report.json"
            report.write_text("{}", encoding="utf-8")
            return SimpleNamespace(report_sha256=sha256_file(report))

        def comparison(*, output_dir: Path, **kwargs):
            del kwargs
            output_dir.mkdir(parents=True, exist_ok=True)
            result = output_dir / "motion_model_comparison_manifest.json"
            result.write_text("{}", encoding="utf-8")
            return result

        def denoiser(repository: Path, *, progress_callback, **kwargs):
            del repository, kwargs
            progress_callback(5, 22, "denoiser participant")
            return {"status": "passed", "rows": [], "summary_rows": []}

        with tempfile.TemporaryDirectory() as temporary, patch(
            "ppg_frailty.quality.stage5_pre.run_formal_internal_motion_reference",
            side_effect=internal,
        ), patch(
            "ppg_frailty.quality.stage5_pre.run_formal_ptt_motion_reference",
            side_effect=external,
        ), patch(
            "ppg_frailty.quality.stage5_pre.run_formal_ptt_motion_training_ablation",
            side_effect=ptt_training,
        ), patch(
            "ppg_frailty.quality.stage5_pre.run_formal_internal_reverse_evaluation",
            side_effect=reverse,
        ), patch(
            "ppg_frailty.quality.stage5_pre._write_motion_model_comparison_package",
            side_effect=comparison,
        ), patch(
            "ppg_frailty.quality.stage5_pre.run_ptt_denoiser_benchmark",
            side_effect=denoiser,
        ), patch(
            "ppg_frailty.quality.stage5_pre.generate_motion_peak_report",
            return_value={},
        ):
            run_motion_peak_study(
                PLAN_ROOT / "stage5_pre.yaml",
                pipeline_root=PIPELINE_ROOT,
                output_root=temporary,
                progress_sink=events.append,
            )

        self.assertTrue(all(event.total == 7 for event in events))
        self.assertEqual(events[0].event, "motion_peak_study_started")
        self.assertEqual(events[-1].event, "motion_peak_study_finished")
        self.assertEqual(events[-1].current, 7)
        observed = {
            (event.detail_current, event.detail_total, event.case_id)
            for event in events
        }
        self.assertIn((7, 29, "internal motion OOF"), observed)
        self.assertIn((11, 22, "PTT motion evaluation"), observed)
        self.assertIn((5, 6, "PTT motion training ablation"), observed)
        self.assertIn((17, 29, "Frailty29 reverse evaluation"), observed)
        self.assertIn((5, 22, "PTT denoiser benchmark"), observed)
        self.assertEqual(observed_training_devices, ["cuda"])

    def test_no_denoiser_skips_only_denoiser_after_both_detector_directions(self) -> None:
        events: list[ProgressEvent] = []

        def internal(repository, *, output_dir, **kwargs):
            del repository, kwargs
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / "motion_internal_evidence.json"
            path.write_text("{}", encoding="utf-8")
            return SimpleNamespace(evidence_sha256=sha256_file(path))

        def external(repository, *, output_dir, **kwargs):
            del repository, kwargs
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / "motion_ptt_external_report.json"
            path.write_text("{}", encoding="utf-8")
            return SimpleNamespace(report_sha256=sha256_file(path))

        def ptt_training(repository, *, output_dir, **kwargs):
            del repository, kwargs
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / "motion_ptt_training_evidence.json"
            path.write_text("{}", encoding="utf-8")
            return SimpleNamespace(evidence_sha256=sha256_file(path))

        def reverse(repository, *, output_dir, **kwargs):
            del repository, kwargs
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / "motion_internal_reverse_evaluation_report.json"
            path.write_text("{}", encoding="utf-8")
            return SimpleNamespace(report_sha256=sha256_file(path))

        def comparison(*, output_dir, **kwargs):
            del kwargs
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / "motion_model_comparison_manifest.json"
            path.write_text("{}", encoding="utf-8")
            return path

        denoiser = Mock(side_effect=AssertionError("denoiser must be skipped"))
        with tempfile.TemporaryDirectory() as temporary, patch(
            "ppg_frailty.quality.stage5_pre.run_formal_internal_motion_reference",
            side_effect=internal,
        ), patch(
            "ppg_frailty.quality.stage5_pre.run_formal_ptt_motion_reference",
            side_effect=external,
        ), patch(
            "ppg_frailty.quality.stage5_pre.run_formal_ptt_motion_training_ablation",
            side_effect=ptt_training,
        ), patch(
            "ppg_frailty.quality.stage5_pre.run_formal_internal_reverse_evaluation",
            side_effect=reverse,
        ), patch(
            "ppg_frailty.quality.stage5_pre._write_motion_model_comparison_package",
            side_effect=comparison,
        ), patch(
            "ppg_frailty.quality.stage5_pre.run_ptt_denoiser_benchmark",
            denoiser,
        ), patch(
            "ppg_frailty.quality.stage5_pre.generate_motion_peak_report",
            return_value={},
        ):
            root = run_motion_peak_study(
                PLAN_ROOT / "stage5_pre.yaml",
                pipeline_root=PIPELINE_ROOT,
                output_root=temporary,
                progress_sink=events.append,
                include_denoiser=False,
            )
            manifest = json.loads(
                (root / "study_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["stages"]["ptt_denoiser_benchmark"]["status"],
                "skipped_by_cli",
            )
            self.assertEqual(
                manifest["stages"]["frailty29_reverse_evaluation"]["status"],
                "passed",
            )
        self.assertFalse(denoiser.called)
        self.assertTrue(all(event.total == 6 for event in events))


if __name__ == "__main__":
    unittest.main()
