from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

from ppg_frailty.reporting.tabular import (
    ReportTable,
    compact_rows,
    write_excel_workbook,
)


class ReportingTabularExportTests(unittest.TestCase):
    def test_compact_rows_collapses_score_mean_and_sd_without_losing_json_source(self) -> None:
        source = [
            {
                "case_id": "case-a",
                "metric": "balanced_accuracy",
                "n": 5,
                "mean": 0.726,
                "sample_sd": 0.060,
                "population_sd": 0.054,
                "ci95_low": 0.65,
                "ci95_high": 0.80,
            }
        ]

        displayed = compact_rows(source)

        self.assertEqual(
            displayed,
            [
                {
                    "case_id": "case-a",
                    "metric": "balanced_accuracy",
                    "n": 5,
                    "mean_sd": "72.6 ± 6.0",
                }
            ],
        )
        self.assertEqual(source[0]["mean"], 0.726)
        self.assertIn("ci95_low", source[0])

    def test_workbook_has_one_valid_xml_worksheet_per_registered_table(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = write_excel_workbook(
                Path(temporary) / "report_tables.xlsx",
                (
                    ReportTable(
                        "metric_distribution_summary",
                        [{"metric": "macro_f1", "mean": 0.8, "sample_sd": 0.1}],
                    ),
                    ReportTable("empty_evidence_table", ()),
                ),
            )

            with ZipFile(path) as archive:
                self.assertIsNone(archive.testzip())
                worksheets = [
                    name
                    for name in archive.namelist()
                    if name.startswith("xl/worksheets/sheet")
                ]
                self.assertEqual(len(worksheets), 2)
                for name in archive.namelist():
                    if name.endswith((".xml", ".rels")):
                        ElementTree.fromstring(archive.read(name))

    def test_workbook_is_readable_by_installed_openpyxl_when_available(self) -> None:
        try:
            from openpyxl import load_workbook
        except ImportError:
            self.skipTest("openpyxl is not installed")
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "tables.xlsx"
            write_excel_workbook(
                target,
                (ReportTable("metrics", ({"case": "a", "score": 0.75},)),),
            )
            workbook = load_workbook(target, read_only=True, data_only=True)
            self.assertEqual(workbook.sheetnames, ["metrics"])
            self.assertEqual(workbook["metrics"]["A2"].value, "a")


if __name__ == "__main__":
    unittest.main()
