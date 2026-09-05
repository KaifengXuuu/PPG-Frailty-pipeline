"""Study-report aggregation tests for recording preprocessing-cache audits."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ppg_frailty.reporting import generate_study_report
from ppg_frailty.reporting.cache_audit import collect_preprocessing_cache_rows
from ppg_frailty.reporting.collect import CollectedStudy


class PreprocessingCacheReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _audit_rows(self) -> tuple[dict, ...]:
        cell = self.root / "case" / "repeat_03_fold_04"
        cell.mkdir(parents=True)
        events = [
            {
                "namespace": "canonical_signal_views",
                "layer": "canonical_physical_signal_views",
                "recording_id": "record_a",
                "cache_key": "a" * 64,
                "disposition": "hit",
                "elapsed_seconds": 0.10,
                "logical_array_bytes": 100,
                "entry_path": "canonical_signal_views/a",
                "module_chain": ["load_recording", "build_signal_views"],
                "affects_predictions": False,
            },
            {
                "namespace": "canonical_signal_views",
                "layer": "canonical_physical_signal_views",
                "recording_id": "record_b",
                "cache_key": "b" * 64,
                "disposition": "written",
                "elapsed_seconds": 0.20,
                "logical_array_bytes": 200,
                "entry_path": "canonical_signal_views/b",
                "module_chain": ["load_recording", "build_signal_views"],
                "affects_predictions": False,
            },
            {
                "namespace": "canonical_signal_views",
                "layer": "canonical_physical_signal_views",
                "recording_id": "record_c",
                "cache_key": "c" * 64,
                "disposition": "namespace_bypassed",
                "elapsed_seconds": 0.05,
                "logical_array_bytes": 50,
                "entry_path": None,
                "module_chain": ["load_recording", "build_signal_views"],
                "affects_predictions": False,
            },
            {
                "namespace": "raw_windows",
                "layer": "pristine_pre_routing_raw_dl_windows",
                "recording_id": "record_a",
                "cache_key": "d" * 64,
                "disposition": "read_only_miss_computed",
                "elapsed_seconds": 0.40,
                "logical_array_bytes": 80,
                "entry_path": None,
                "module_chain": ["raw_window_plan", "x_dl_all8_window_norm"],
                "affects_predictions": False,
            },
        ]
        (cell / "preprocessing_cache.json").write_text(
            json.dumps(
                {
                    "schema_version": "ppg_frailty.preprocessing_cache_audit.v1",
                    "mode": "read_write",
                    "root": "artifacts/studies/cache",
                    "namespaces": ["canonical_signal_views", "raw_windows"],
                    "source_verification": "sha256",
                    "affects_predictions": False,
                    "labels_cached": False,
                    "fold_local_artifacts_cached": False,
                    "route_masks_cached": False,
                    "counts": {},
                    "logical_array_bytes": 430,
                    "elapsed_seconds": 0.75,
                    "events": events,
                    "identities": {},
                }
            ),
            encoding="utf-8",
        )
        rows, limitations = collect_preprocessing_cache_rows(
            "case_001", self.root / "case"
        )
        self.assertEqual(limitations, ())
        return tuple(dict(row) for row in rows)

    def test_collects_per_cell_layer_hit_write_bypass_bytes_and_time(self) -> None:
        rows = self._audit_rows()
        self.assertEqual(len(rows), 2)
        by_namespace = {row["namespace"]: row for row in rows}
        views = by_namespace["canonical_signal_views"]
        self.assertEqual((views["repeat"], views["fold"]), (3, 4))
        self.assertEqual(views["event_count"], 3)
        self.assertEqual(views["hit_count"], 1)
        self.assertEqual(views["write_count"], 1)
        self.assertEqual(views["bypass_count"], 1)
        self.assertEqual(views["logical_array_bytes"], 350)
        self.assertAlmostEqual(views["elapsed_seconds"], 0.35)
        self.assertEqual(views["unique_cache_key_count"], 3)
        self.assertEqual(
            views["disposition_counts"],
            {"hit": 1, "namespace_bypassed": 1, "written": 1},
        )
        raw = by_namespace["raw_windows"]
        self.assertEqual(raw["bypass_count"], 1)
        self.assertEqual(raw["write_count"], 0)
        self.assertEqual(raw["logical_array_bytes"], 80)

    def test_report_writes_cache_csv_json_and_inventory_entries(self) -> None:
        rows = self._audit_rows()
        report_root = self.root / "report"
        report_root.mkdir()
        bundle = CollectedStudy(
            root=report_root,
            plan={
                "study": {"study_id": "cache_reporting_fixture"},
                "execution": {"repeats": [3], "folds": [4]},
                "report": {
                    "write_static_figures": False,
                    "write_html": False,
                    "write_excel_workbook": False,
                },
            },
            manifest={"cases": [], "planned_case_count": 0},
            case_records=(),
            varied_parameters=(),
            controlled_parameters=(),
            cell_rows=(),
            history_rows=(),
            file_oof_rows=(),
            subject_oof_rows=(),
            role_oof_rows=(),
            quality_rows=(),
            trusted_config_metrics=(),
            limitations=(),
            preprocessing_cache_rows=rows,
        )
        result = generate_study_report(report_root, collected=bundle)
        csv_path = report_root / "tables" / "preprocessing_cache.csv"
        json_path = report_root / "tables" / "preprocessing_cache.json"
        self.assertTrue(csv_path.is_file())
        self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), list(rows))
        inventory = json.loads(result.output_index.read_text(encoding="utf-8"))
        paths = {row["path"] for row in inventory["artifacts"]}
        self.assertIn("tables/preprocessing_cache.csv", paths)
        self.assertIn("tables/preprocessing_cache.json", paths)
        summary = json.loads(
            (report_root / "study_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["preprocessing_cache"], list(rows))


if __name__ == "__main__":
    unittest.main()
