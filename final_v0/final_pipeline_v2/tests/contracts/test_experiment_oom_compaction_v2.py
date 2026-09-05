"""Bounded-memory contracts for Stage 5 per-fold experiment artifacts."""

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from ppg_frailty import experiment


def _full_sqi_result(state: str) -> dict[str, object]:
    return {
        "state": state,
        "reasons": ["fixture_reason"],
        "components": {"top_level_trace": list(range(100))},
        "q_rate": {
            "state": "positive",
            "score": 0.75,
            "coverage": 0.8,
            "threshold": 0.5,
            "reasons": [],
            "components": {"per_peak_trace": list(range(200))},
        },
        "q_morph": {
            "state": "negative",
            "score": 0.6,
            "coverage": 0.9,
            "threshold": 0.65,
            "reasons": ["below_threshold"],
            "components": {"template_trace": list(range(300))},
        },
    }


class ExperimentOomCompactionTests(unittest.TestCase):
    def test_compact_window_evidence_keeps_scalars_not_components(self) -> None:
        evidence = {
            "window-0": {
                "direct": _full_sqi_result("unfit"),
                "post_reduction": _full_sqi_result("acceptable"),
                "unused_stage": {"components": {"large": [1, 2, 3]}},
            }
        }

        compact = experiment._compact_window_sqi_evidence(evidence)

        self.assertEqual(set(compact["window-0"]), {"direct", "post_reduction"})
        direct = compact["window-0"]["direct"]
        self.assertEqual(direct["state"], "unfit")
        self.assertEqual(direct["q_rate"]["score"], 0.75)
        self.assertEqual(direct["q_rate"]["coverage"], 0.8)
        self.assertEqual(direct["q_morph"]["threshold"], 0.65)
        self.assertNotIn("components", direct)
        self.assertNotIn("components", direct["q_rate"])
        self.assertNotIn("components", direct["q_morph"])
        self.assertIn("components", evidence["window-0"]["direct"])

    def test_full_window_evidence_gzip_is_streamed_and_deterministic(self) -> None:
        summary = {
            "repeat_index": 2,
            "fold_index": 4,
            "config_hash": "a" * 64,
            "route_window_sqi_evidence": [
                {
                    "record_id": "participant-01_B",
                    "participant_id": "participant-01",
                    "role": "B",
                    "windows": {
                        "window-1": {"direct": _full_sqi_result("excellent")},
                        "window-0": {"direct": _full_sqi_result("unfit")},
                    },
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()

            first_count, first_sha = experiment._write_route_window_sqi_evidence(
                first,
                summary,
            )
            second_count, second_sha = experiment._write_route_window_sqi_evidence(
                second,
                summary,
            )

            first_path = first / "route_window_sqi_evidence.jsonl.gz"
            second_path = second / "route_window_sqi_evidence.jsonl.gz"
            self.assertEqual((first_count, second_count), (2, 2))
            self.assertEqual(first_sha, second_sha)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())

            with gzip.open(first_path, "rt", encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream]
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["record_type"], "header")
            self.assertFalse(rows[0]["report_consumed"])
            self.assertEqual(
                {row["routing_window_id"] for row in rows[1:]},
                {"window-0", "window-1"},
            )
            self.assertIn(
                "components",
                rows[1]["evidence"]["direct"]["q_rate"],
            )

    def test_artifact_index_externalizes_inline_payloads_idempotently(self) -> None:
        prefix = "repeat_02_fold_04"
        summary = {
            "quality_diagnostics": [{"record_id": "record-1"}],
            "training_history": [{"epoch": 1}],
            "sampling_diagnostics": [{"class": 0}],
            "physical_recording_qc": [{"record_id": "record-1"}],
            "route_artifacts": [{"record_id": "record-1"}],
            "route_window_sqi_evidence": [
                {
                    "record_id": "record-1",
                    "windows": {
                        "window-0": {"direct": _full_sqi_result("unfit")},
                        "window-1": {"direct": _full_sqi_result("excellent")},
                    },
                }
            ],
            "preprocessing_cache": {
                "mode": "read_write",
                "counts": {"hit": 1},
                "logical_array_bytes": 123,
                "elapsed_seconds": 0.5,
                "large_event_log": list(range(100)),
            },
        }

        compact = experiment._artifact_index_cell_summary(
            summary,
            artifact_prefix=prefix,
        )
        second_pass = experiment._artifact_index_cell_summary(
            compact,
            artifact_prefix=prefix,
        )

        for inline_name in (
            "quality_diagnostics",
            "training_history",
            "sampling_diagnostics",
            "physical_recording_qc",
            "route_artifacts",
            "route_window_sqi_evidence",
            "preprocessing_cache",
        ):
            self.assertNotIn(inline_name, compact)
        self.assertEqual(
            compact["route_window_sqi_evidence_artifact"],
            f"{prefix}/route_window_sqi_evidence.jsonl.gz",
        )
        self.assertEqual(compact["route_window_sqi_evidence_row_count"], 2)
        self.assertEqual(
            compact["quality_diagnostics_artifact"],
            f"{prefix}/quality_diagnostics.json",
        )
        self.assertNotIn(
            "large_event_log",
            compact["preprocessing_cache_summary"],
        )
        self.assertEqual(second_pass, compact)
        self.assertIn("route_window_sqi_evidence", summary)

    def test_root_aggregation_keeps_oof_and_drops_full_route_payloads(self) -> None:
        file_rows = ({"record_id": "record-1"},)
        subject_rows = ({"participant_id": "participant-01"},)
        window_rows = ({"window_id": "window-0"},)
        role_rows = ({"role": "B"},)
        member_rows = ({"member_index": 0},)
        cell = experiment._CellResult(
            summary={
                "quality_diagnostics": [
                    {
                        "record_id": "record-1",
                        "route_status": "retained",
                    }
                ],
                "route_artifacts": [
                    {"record_id": "record-1", "cells": list(range(100))}
                ],
                "route_window_sqi_evidence": [
                    {
                        "record_id": "record-1",
                        "windows": {
                            "window-0": {"direct": _full_sqi_result("unfit")}
                        },
                    }
                ],
                "physical_recording_qc": [{"trace": list(range(100))}],
            },
            file_rows=file_rows,
            subject_rows=subject_rows,
            window_rows=window_rows,
            role_rows=role_rows,
            member_rows=member_rows,
        )

        compact = experiment._cell_for_root_aggregation(
            cell,
            artifact_prefix="repeat_00_fold_00",
        )

        self.assertNotIn("route_artifacts", compact.summary)
        self.assertNotIn("route_window_sqi_evidence", compact.summary)
        self.assertNotIn("physical_recording_qc", compact.summary)
        self.assertEqual(
            compact.summary["quality_diagnostics"],
            cell.summary["quality_diagnostics"],
        )
        self.assertNotIn(
            "route_artifact",
            compact.summary["quality_diagnostics"][0],
        )
        self.assertIs(compact.file_rows, file_rows)
        self.assertIs(compact.subject_rows, subject_rows)
        self.assertIs(compact.window_rows, window_rows)
        self.assertIs(compact.role_rows, role_rows)
        self.assertIs(compact.member_rows, member_rows)


if __name__ == "__main__":
    unittest.main()
