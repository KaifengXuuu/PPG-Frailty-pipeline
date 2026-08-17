"""内部/外部 manifest 与 QC 测试 / Manifest and QC contract tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PIPELINE_ROOT.parents[1]
sys.path.insert(0, str(PIPELINE_ROOT / "src"))

from ppg_frailty.data import (  # noqa: E402
    QCReason,
    QCStatus,
    QCThresholds,
    assess_numeric_record,
    audit_external_manifest,
    audit_manifest,
    load_m2_internal_manifest,
)
from ppg_frailty.data.external_manifest import (  # noqa: E402
    M2_EXTERNAL_RELATIVE_PATH,
    PTT_DATASET_ID,
    PTT_WAVELENGTH_STATUS,
    load_m2_external_manifest,
)
from ppg_frailty.data.qc import parse_failure_assessment  # noqa: E402
from ppg_frailty.data.schema import CANONICAL_CHANNEL_SCHEMA  # noqa: E402


class ManifestContractTests(unittest.TestCase):
    """验证权威 roster 与语义 / Validate authoritative rosters and semantics."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.internal_rows = load_m2_internal_manifest(
            REPOSITORY_ROOT,
            verify_sources=False,
        )
        cls.external_rows = load_m2_external_manifest(
            REPOSITORY_ROOT / M2_EXTERNAL_RELATIVE_PATH
        )

    def test_internal_manifest_exact_roster(self) -> None:
        """内部 roster 不得静默少行 / Internal roster cannot silently shrink."""

        summary = audit_manifest(self.internal_rows)
        self.assertEqual(summary["record_count"], 261)
        self.assertEqual(summary["participant_count"], 29)
        self.assertEqual(
            summary["class_participant_counts"],
            {"Pre-Frail": 9, "Robust/Non-Frail": 12, "Young": 8},
        )
        self.assertTrue(
            all(
                tuple(row.channel_schema) == CANONICAL_CHANNEL_SCHEMA
                for row in self.internal_rows
            )
        )

    def test_external_manifest_preserves_unresolved_ptt_wavelength(self) -> None:
        """禁止把 PTT pleth 猜成 RED/IR / Never infer PTT wavelength mapping."""

        summary = audit_external_manifest(self.external_rows)
        self.assertEqual(summary["record_count_total"], 80)
        self.assertEqual(summary["record_count_included"], 79)
        self.assertEqual(summary["sim_included_count"], 13)
        ptt = [
            row for row in self.external_rows if row.dataset_id == PTT_DATASET_ID
        ]
        self.assertEqual(len(ptt), 66)
        self.assertEqual({row.subject_id for row in ptt}, {f"s{i}" for i in range(1, 23)})
        self.assertTrue(
            all(
                row.ppg_wavelength_status == PTT_WAVELENGTH_STATUS
                for row in ptt
            )
        )


class QualityControlTests(unittest.TestCase):
    """验证显式阈值与 fail-closed 原因 / Validate explicit fail-closed QC."""

    def setUp(self) -> None:
        channels = CANONICAL_CHANNEL_SCHEMA
        self.thresholds = QCThresholds(
            minimum_duration_s=3.0,
            maximum_nonfinite_gap_s=0.05,
            flatline_std_floor_by_channel={name: 1e-6 for name in channels},
            robust_span_floor_by_channel={name: 1e-5 for name in channels},
            absolute_limit_by_channel={name: 10.0 for name in channels},
            saturation_fraction_limit=0.05,
            timestamp_relative_tolerance=1e-6,
            timestamps_required=True,
        )
        self.fs = 100.0
        self.timestamps = np.arange(400, dtype=np.float64) / self.fs
        self.values = np.column_stack(
            [
                np.sin(
                    2.0
                    * np.pi
                    * (1.0 + 0.07 * index)
                    * self.timestamps
                    + 0.13 * index
                )
                for index in range(len(CANONICAL_CHANNEL_SCHEMA))
            ]
        )

    def test_valid_numeric_record_passes(self) -> None:
        result = assess_numeric_record(
            self.values,
            CANONICAL_CHANNEL_SCHEMA,
            fs=self.fs,
            thresholds=self.thresholds,
            timestamps_s=self.timestamps,
        )
        self.assertEqual(result.status, QCStatus.PASS)
        self.assertEqual(result.reasons, ())

    def test_missing_values_are_reported_not_silently_imputed(self) -> None:
        broken = self.values.copy()
        broken[:, 0] = np.nan
        result = assess_numeric_record(
            broken,
            CANONICAL_CHANNEL_SCHEMA,
            fs=self.fs,
            thresholds=self.thresholds,
            timestamps_s=self.timestamps,
        )
        self.assertEqual(result.status, QCStatus.FAIL)
        self.assertIn(QCReason.ALL_NONFINITE_CHANNEL.value, result.reasons)

    def test_parse_failure_has_machine_reason(self) -> None:
        result = parse_failure_assessment("bad numeric token")
        self.assertEqual(result.status, QCStatus.FAIL)
        self.assertEqual(result.reasons, (QCReason.PARSE_FAILURE.value,))

    def test_unknown_channel_returns_reason_instead_of_crashing(self) -> None:
        channels = ("UNKNOWN", *CANONICAL_CHANNEL_SCHEMA[1:])
        result = assess_numeric_record(
            self.values,
            channels,
            fs=self.fs,
            thresholds=self.thresholds,
            timestamps_s=self.timestamps,
        )
        self.assertEqual(result.status, QCStatus.FAIL)
        self.assertIn(QCReason.MISSING_REQUIRED_CHANNEL.value, result.reasons)


if __name__ == "__main__":
    unittest.main()
