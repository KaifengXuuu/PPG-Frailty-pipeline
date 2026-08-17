"""已物化 CSV/报告回读测试 / Materialized CSV and report read-back tests."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_ROOT / "src"))

from ppg_frailty.data import (  # noqa: E402
    FrozenFoldRegistry,
    load_external_manifest,
    load_manifest,
    load_provisional_external_split,
)
from ppg_frailty.provenance import sha256_file  # noqa: E402


class MaterializedOutputTests(unittest.TestCase):
    """所有生成物必须可回读且与报告一致 / Generated outputs round-trip."""

    def test_internal_manifest_and_fold_csvs(self) -> None:
        manifest_path = PIPELINE_ROOT / "manifests/internal_records_v1.csv"
        primary_path = PIPELINE_ROOT / "splits/sgkf5_v1.csv"
        repeated_path = PIPELINE_ROOT / "splits/sgkf5_repeats_v1.csv"
        self.assertEqual(len(load_manifest(manifest_path)), 261)
        primary = FrozenFoldRegistry.from_csv(primary_path)
        repeated = FrozenFoldRegistry.from_csv(repeated_path)
        self.assertEqual(len(primary.assignments), 29)
        self.assertEqual(len(repeated.assignments), 145)
        self.assertEqual(len(primary.get_split(0, 0)["oof_participant_ids"]), 6)

    def test_external_manifest_and_provisional_split(self) -> None:
        external_path = PIPELINE_ROOT / "manifests/external_records_v1.csv"
        provisional_path = (
            PIPELINE_ROOT
            / "splits/v1_provisional_external_grouped_split_seed42.csv"
        )
        external = load_external_manifest(external_path)
        split_rows = load_provisional_external_split(provisional_path)
        self.assertEqual(len(external), 80)
        self.assertEqual(
            sum(row.inclusion_status == "included" for row in external),
            79,
        )
        self.assertEqual(len(split_rows), 110)

    def test_reports_hash_their_generated_artifacts(self) -> None:
        internal_report = json.loads(
            (PIPELINE_ROOT / "reports/data_contract_report.json").read_text(
                encoding="utf-8"
            )
        )
        external_report = json.loads(
            (
                PIPELINE_ROOT / "reports/external_data_contract_report.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(internal_report["status"], "pass")
        self.assertEqual(
            external_report["status"],
            "pass_with_provisional_split_pending_confirmation",
        )
        for report in (internal_report, external_report):
            for artifact in report["generated_artifacts"].values():
                path = PIPELINE_ROOT / artifact["path"]
                self.assertEqual(sha256_file(path), artifact["sha256"])
                self.assertEqual(path.stat().st_size, artifact["bytes"])


if __name__ == "__main__":
    unittest.main()
