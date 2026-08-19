"""No-training tests for the compact two-level study progress display."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from frailty_3class_sweep_v2 import _generate_report_with_progress
from ppg_frailty.study import ProgressEvent, TerminalProgressSink
from ppg_frailty.study.runner import (
    _ExecutorEventRelay,
    _executor_progress_adapter,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class _TtyBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


class ProgressDisplayTests(unittest.TestCase):
    def test_refresh_thread_can_pause_for_process_fork_and_restart(self) -> None:
        sink = TerminalProgressSink(
            stream=_TtyBuffer(),
            refresh_interval=60.0,
            ansi=True,
        )
        sink(ProgressEvent(event="program_started", message="loading"))
        first_thread = sink._thread
        self.assertIsNotNone(first_thread)
        sink.suspend_refresh()
        self.assertIsNone(sink._thread)
        sink(ProgressEvent(event="study_running", message="running"))
        self.assertIsNotNone(sink._thread)
        self.assertIsNot(sink._thread, first_thread)
        sink.close()

    def test_two_level_display_has_elapsed_eta_and_collapses_subline(self) -> None:
        clock = _Clock()
        stream = _TtyBuffer()
        sink = TerminalProgressSink(
            stream=stream,
            width=10,
            refresh_interval=0,
            ansi=True,
            clock=clock,
        )
        sink(ProgressEvent(event="program_started", message="loading plan"))
        clock.value = 10.0
        sink(
            ProgressEvent(
                event="study_started",
                current=0,
                total=2,
                unit_current=0,
                unit_total=2,
                message="jobs=2",
            )
        )
        sink(
            ProgressEvent(
                event="cell_start",
                case_id="raw_model",
                repeat=0,
                fold=0,
                unit_current=0,
                unit_total=2,
                detail_current=0,
                detail_total=2,
                detail_label="CV repeat 1/2 · fold 1/2",
            )
        )
        clock.value = 70.0
        sink(
            ProgressEvent(
                event="cell_complete",
                case_id="raw_model",
                repeat=0,
                fold=1,
                unit_current=1,
                unit_total=2,
                detail_current=2,
                detail_total=2,
                detail_label="CV repeat 1/2 · fold 2/2",
            )
        )
        total_line, detail_line = sink._lines()
        self.assertIn("1/4 case-repeats", total_line)
        self.assertIn("elapsed 00:01:10", total_line)
        self.assertIn("ETA~ 00:03:30", total_line)
        self.assertIsNotNone(detail_line)
        self.assertIn("elapsed 00:01:00", str(detail_line))
        self.assertIn("\x1b[1A", stream.getvalue())
        sink(
            ProgressEvent(
                event="case_finished",
                current=1,
                total=2,
                case_id="raw_model",
                unit_current=2,
                unit_total=2,
                message="passed",
            )
        )
        self.assertIsNone(sink._lines()[1])
        sink.close()
        self.assertTrue(stream.getvalue().endswith("\n"))

    def test_failed_case_is_terminal_and_partial_status_is_preserved(self) -> None:
        clock = _Clock()
        sink = TerminalProgressSink(
            stream=_TtyBuffer(),
            width=10,
            refresh_interval=0,
            ansi=True,
            clock=clock,
        )
        sink(
            ProgressEvent(
                event="study_started",
                current=0,
                total=4,
                unit_current=0,
                unit_total=1,
                message="jobs=3",
            )
        )
        for index, (case_id, status) in enumerate(
            (
                ("raw", "passed"),
                ("vector", "passed"),
                ("matrix", "passed"),
                ("fusion", "failed"),
            ),
            start=1,
        ):
            sink(
                ProgressEvent(
                    event="case_finished",
                    current=index,
                    total=4,
                    case_id=case_id,
                    unit_current=1,
                    unit_total=1,
                    message=status,
                )
            )
        sink(
            ProgressEvent(
                event="study_finished",
                current=4,
                total=4,
                message="partial",
            )
        )
        total_line, detail_line = sink._lines()
        self.assertIn("4/4 case-repeats", total_line)
        self.assertIn("1 failed", total_line)
        self.assertIn("3 passed", total_line)
        self.assertIn("partial", total_line)
        self.assertIn("ETA~ 00:00:00", total_line)
        self.assertIsNone(detail_line)

    def test_report_path_starts_after_closed_progress_line(self) -> None:
        stream = _TtyBuffer()
        sink = TerminalProgressSink(
            stream=stream,
            refresh_interval=0,
            ansi=False,
        )
        sink(
            ProgressEvent(
                event="study_started",
                current=0,
                total=1,
                unit_current=0,
                unit_total=1,
            )
        )
        sink(
            ProgressEvent(
                event="study_finished",
                current=1,
                total=1,
                message="partial",
            )
        )
        with patch(
            "frailty_3class_sweep_v2.generate_study_report",
            return_value=SimpleNamespace(summary_markdown=Path("/tmp/SUMMARY.md")),
        ), redirect_stdout(stream):
            _generate_report_with_progress(
                "/tmp/study",
                sink,
                study_status="partial",
            )
        sink.close()
        rendered = stream.getvalue()
        self.assertIn("report complete · study partial", rendered)
        self.assertIn("\nReport: /tmp/SUMMARY.md\n", rendered)
        self.assertNotIn("report completeReport:", rendered)

    def test_adapter_uses_repeat_as_unit_and_fold_as_detail(self) -> None:
        events: list[ProgressEvent] = []
        emit = _executor_progress_adapter(
            events.append,
            "case_a",
            repeats=(0, 1),
            folds=(0, 1),
        )
        emit(
            {
                "stage": "cell_start",
                "current_cell": 1,
                "total_cells": 4,
                "repeat_index": 0,
                "fold_index": 0,
            }
        )
        emit(
            {
                "stage": "cell_complete",
                "current_cell": 2,
                "total_cells": 4,
                "repeat_index": 0,
                "fold_index": 1,
            }
        )
        emit({"stage": "run_complete", "total_cells": 4})
        self.assertEqual(
            (events[0].unit_current, events[0].unit_total),
            (0, 2),
        )
        self.assertEqual(
            (events[0].detail_current, events[0].detail_total),
            (0, 2),
        )
        self.assertEqual(
            (events[1].unit_current, events[1].detail_current),
            (1, 2),
        )
        self.assertEqual(
            (events[2].unit_current, events[2].unit_total),
            (2, 2),
        )

    def test_child_jsonl_relay_reads_only_complete_rows_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "executor_events.jsonl"
            first = json.dumps(
                ProgressEvent(
                    event="cell_start",
                    case_id="case_a",
                    unit_current=0,
                    unit_total=2,
                    detail_current=0,
                    detail_total=2,
                ).to_dict(),
                sort_keys=True,
            ).encode("utf-8")
            second = json.dumps(
                ProgressEvent(
                    event="cell_complete",
                    case_id="case_a",
                    unit_current=1,
                    unit_total=2,
                    detail_current=2,
                    detail_total=2,
                ).to_dict(),
                sort_keys=True,
            ).encode("utf-8")
            split = len(second) // 2
            path.write_bytes(first + b"\n" + second[:split])
            events: list[ProgressEvent] = []
            relay = _ExecutorEventRelay(events.append)
            relay.register("case_a", path)
            self.assertEqual(relay.drain(), 1)
            with path.open("ab") as stream:
                stream.write(second[split:] + b"\n")
            self.assertEqual(relay.drain(), 1)
            self.assertEqual(relay.drain(), 0)
            self.assertEqual(
                [event.event for event in events],
                ["cell_start", "cell_complete"],
            )


if __name__ == "__main__":
    unittest.main()
