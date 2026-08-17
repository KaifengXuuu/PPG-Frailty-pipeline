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
    load_formal_ptt_repeated_folds,
    load_manifest,
)
from ppg_frailty.provenance import sha256_file  # noqa: E402


class MaterializedOutputTests(unittest.TestCase):
    """所有生成物必须可回读且与报告一致 / Generated outputs round-trip."""

    def test_internal_manifest_and_fold_csvs(self) -> None:
        manifest_path = PIPELINE_ROOT / "manifests/internal_records_v2.csv"
        primary_path = PIPELINE_ROOT / "splits/sgkf5_seed42_v2.csv"
        repeated_path = PIPELINE_ROOT / "splits/sgkf5_repeated_grouped_5x5_v2.csv"
        manifest = load_manifest(manifest_path)
        self.assertEqual(len(manifest), 261)
        self.assertEqual(
            {row.class_name_provenance_alias for row in manifest},
            {"pre_frail", "robust_non_frail", "young"},
        )
        self.assertEqual(
            {row.class_source for row in manifest},
            {"frailty_status_2", "frailty_status_3", "cohort_override_young"},
        )
        scored_rows = [row for row in manifest if row.class_id in {0, 1}]
        young_rows = [row for row in manifest if row.class_id == 2]
        self.assertTrue(all(row.label_record_id for row in scored_rows))
        self.assertTrue(young_rows)
        self.assertTrue(any(not row.label_record_id for row in young_rows))
        self.assertTrue(any(row.label_record_id for row in young_rows))
        self.assertTrue(
            all(row.class_source == "cohort_override_young" for row in young_rows)
        )
        self.assertTrue(
            all(
                row.class_id == 2 and row.class_source == "cohort_override_young"
                for row in manifest
                if not row.label_record_id
            )
        )
        primary = FrozenFoldRegistry.from_csv(primary_path)
        repeated = FrozenFoldRegistry.from_csv(repeated_path)
        self.assertEqual(len(primary.assignments), 29)
        self.assertEqual(len(repeated.assignments), 145)
        self.assertEqual(len(primary.get_split(0, 0)["oof_participant_ids"]), 6)

    def test_external_manifest_and_formal_ptt_split(self) -> None:
        external_path = PIPELINE_ROOT / "manifests/external_records_v2.csv"
        formal_path = PIPELINE_ROOT / "splits/ptt_formal_repeated_grouped_5x5_v2.csv"
        external = load_external_manifest(external_path)
        split_rows = load_formal_ptt_repeated_folds(formal_path)
        self.assertEqual(len(external), 80)
        self.assertEqual(
            sum(row.inclusion_status == "included" for row in external),
            79,
        )
        self.assertEqual(len(split_rows), 110)
        self.assertEqual({int(row["repeat_index"]) for row in split_rows}, set(range(5)))

    def test_reports_hash_their_generated_artifacts(self) -> None:
        internal_report = json.loads(
            (PIPELINE_ROOT / "reports/data_contract_report_v2.json").read_text(
                encoding="utf-8"
            )
        )
        external_report = json.loads(
            (
                PIPELINE_ROOT / "reports/external_data_contract_report_v2.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(internal_report["status"], "passed")
        self.assertEqual(external_report["status"], "passed")
        for report in (internal_report, external_report):
            for artifact in report["generated_artifacts"].values():
                path = PIPELINE_ROOT / artifact["path"]
                self.assertEqual(sha256_file(path), artifact["sha256"])
                self.assertEqual(path.stat().st_size, artifact["bytes"])


if __name__ == "__main__":
    unittest.main()
