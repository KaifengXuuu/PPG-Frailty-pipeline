"""内部/外部 manifest 与 QC 测试 / Manifest and QC contract tests."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PIPELINE_ROOT.parents[1]
sys.path.insert(0, str(PIPELINE_ROOT / "src"))

from ppg_frailty.data import (  # noqa: E402
    QCReason,
    QCStatus,
    QCThresholds,
    assess_manifest_record,
    assess_numeric_record,
    audit_external_manifest,
    audit_manifest,
    load_m2_internal_manifest,
)
from ppg_frailty.data.external_manifest import (  # noqa: E402
    M2_EXTERNAL_RELATIVE_PATH,
    PTT_DATASET_ID,
    PTT_DISTAL_CHANNEL_MAPPING,
    PTT_IMU_UNIT_CONFLICT_PROVENANCE,
    PTT_SOURCE_WAVELENGTH_STATUS,
    PTT_WAVELENGTH_STATUS,
    adapt_ptt_synchronized_channels,
    load_m2_external_manifest,
    select_ptt_distal_red_ir,
)
from ppg_frailty.data.qc import parse_failure_assessment  # noqa: E402
from ppg_frailty.data.schema import (  # noqa: E402
    CANONICAL_CHANNEL_SCHEMA,
    canonicalize_role_family,
    is_default_classifier_role,
)


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
        self.assertEqual(
            summary["role_family_record_counts"],
            {"B": 29, "R": 116, "S": 58, "W": 58},
        )
        self.assertEqual(summary["default_classifier_record_count"], 145)
        self.assertEqual(canonicalize_role_family("R4"), "R")
        self.assertTrue(is_default_classifier_role("B"))
        self.assertTrue(is_default_classifier_role("R2"))
        self.assertFalse(is_default_classifier_role("S1"))

    def test_external_manifest_records_adopted_distal_ptt_mapping(self) -> None:
        """Preserve source conflict while applying the confirmed project mapping."""

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
        self.assertEqual(
            summary["ptt_source_wavelength_status"],
            PTT_SOURCE_WAVELENGTH_STATUS,
        )
        self.assertEqual(
            summary["ptt_distal_channel_mapping"],
            PTT_DISTAL_CHANNEL_MAPPING,
        )
        conflict = PTT_IMU_UNIT_CONFLICT_PROVENANCE
        self.assertEqual(
            conflict["status"],
            "project_resolved_v2_036_source_manifest_conflict_retained",
        )
        self.assertEqual(
            conflict["wfdb_header_declaration"]["sha256"],
            "8f4570b9f58c43d9f0bf1dac3eba71d91acd0d85c6bff33067b7db897ae7209f",
        )
        self.assertEqual(
            conflict["canonical_csv_numeric_evidence"]["sha256"],
            "25601a60611e01fe138a88829b202c95f047764f5ee232b4d3457df460133391",
        )
        self.assertEqual(
            conflict["project_adopted_units"]["acceleration_unit"],
            "m/s^2",
        )
        self.assertEqual(
            conflict["project_adopted_units"]["acceleration_conversion"],
            "identity_m_per_s2_no_scale",
        )
        self.assertEqual(
            conflict["project_adopted_units"]["gyroscope_unit"],
            "deg/s",
        )
        pair = select_ptt_distal_red_ir(
            {"pleth_1": np.array([1.0, 2.0]), "pleth_2": np.array([3.0, 4.0])}
        )
        self.assertTrue(np.array_equal(pair[:, 0], [1.0, 2.0]))
        self.assertTrue(np.array_equal(pair[:, 1], [3.0, 4.0]))

        source_time = np.arange(1000, dtype=np.float64) / 500.0
        adapted = adapt_ptt_synchronized_channels(
            {
                "pleth_1": np.sin(2.0 * np.pi * source_time),
                "pleth_2": np.cos(2.0 * np.pi * source_time),
                "ecg": source_time,
                "acc_x": 2.0 * source_time,
            },
            record_id="ptt_unit_test",
            observed_source_file_sha256="0" * 64,
            additional_channel_order=("ecg", "acc_x"),
        )
        adapted.validate()
        self.assertEqual(adapted.channel_schema, ("RED", "IR", "ecg", "acc_x"))
        self.assertEqual(adapted.values.shape, (800, 4))
        self.assertEqual((adapted.up, adapted.down), (4, 5))
        self.assertTrue(np.array_equal(adapted.ppg_red_ir, adapted.values[:, :2]))
        self.assertTrue(np.array_equal(adapted.timestamps_s, np.arange(800) / 400.0))
        self.assertEqual(len(adapted.source_values_sha256), 64)
        self.assertEqual(len(adapted.output_values_sha256), 64)
        with self.assertRaisesRegex(ValueError, "output_values_sha256"):
            replace(adapted, output_values_sha256="0" * 64).validate()

        bound_record = next(
            row
            for row in self.external_rows
            if row.dataset_id == PTT_DATASET_ID and row.record_id == "s1_sit"
        )
        bound = adapt_ptt_synchronized_channels(
            {
                "pleth_1": np.sin(2.0 * np.pi * source_time),
                "pleth_2": np.cos(2.0 * np.pi * source_time),
            },
            external_record=bound_record,
            observed_source_file_sha256=bound_record.checksum_sha256,
        )
        self.assertEqual(bound.record_id, "s1_sit")
        self.assertEqual(bound.source_file_sha256, bound_record.checksum_sha256)
        with self.assertRaisesRegex(ValueError, "differs from manifest"):
            adapt_ptt_synchronized_channels(
                {
                    "pleth_1": np.sin(2.0 * np.pi * source_time),
                    "pleth_2": np.cos(2.0 * np.pi * source_time),
                },
                external_record=bound_record,
                observed_source_file_sha256="0" * 64,
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
            absolute_limit_by_channel={name: None for name in channels},
            saturation_fraction_limit=None,
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

    def test_device_dependent_limits_are_deferred_until_verified(self) -> None:
        high_scale = self.values * 100.0
        deferred = assess_numeric_record(
            high_scale,
            CANONICAL_CHANNEL_SCHEMA,
            fs=self.fs,
            thresholds=self.thresholds,
            timestamps_s=self.timestamps,
        )
        self.assertNotIn(QCReason.CLIPPING.value, deferred.reasons)
        self.assertEqual(
            deferred.metrics["device_dependent_qc_status"],
            "deferred_missing_device_metadata",
        )
        enabled = assess_numeric_record(
            high_scale,
            CANONICAL_CHANNEL_SCHEMA,
            fs=self.fs,
            thresholds=replace(
                self.thresholds,
                device_limits_verified=True,
                absolute_limit_by_channel={
                    name: 10.0 for name in CANONICAL_CHANNEL_SCHEMA
                },
                saturation_fraction_limit=0.05,
            ),
            timestamps_s=self.timestamps,
        )
        self.assertIn(QCReason.CLIPPING.value, enabled.reasons)

    def test_device_limit_contract_rejects_placeholder_numbers_and_missing_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit None"):
            replace(
                self.thresholds,
                absolute_limit_by_channel={
                    name: 10.0 for name in CANONICAL_CHANNEL_SCHEMA
                },
            ).validate()
        with self.assertRaisesRegex(ValueError, "finite positive"):
            replace(
                self.thresholds,
                device_limits_verified=True,
            ).validate()

    def test_deferred_timestamp_and_saturation_parameters_are_none(self) -> None:
        deferred = replace(
            self.thresholds,
            timestamps_required=False,
            timestamp_relative_tolerance=None,
        )
        deferred.validate()
        with self.assertRaisesRegex(ValueError, "saturation limit"):
            replace(deferred, saturation_fraction_limit=0.05).validate()
        with self.assertRaisesRegex(ValueError, "timestamped QC"):
            assess_numeric_record(
                self.values,
                CANONICAL_CHANNEL_SCHEMA,
                fs=self.fs,
                thresholds=deferred,
                timestamps_s=self.timestamps,
            )

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

    def test_manifest_bound_qc_enforces_identity_but_defers_device_limits(self) -> None:
        row = replace(
            ManifestContractTests.internal_rows[0],
            n_samples=self.values.shape[0],
            fs=self.fs,
            duration_s=self.values.shape[0] / self.fs,
        )
        admitted = assess_manifest_record(
            row,
            self.values,
            observed_channel_names=CANONICAL_CHANNEL_SCHEMA,
            observed_fs=self.fs,
            thresholds=self.thresholds,
            timestamps_s=self.timestamps,
        )
        self.assertTrue(admitted.admitted)
        self.assertFalse(admitted.evidence["metrics"]["device_dependent_checks_executed"])
        rejected = assess_manifest_record(
            row,
            self.values[:-1],
            observed_channel_names=CANONICAL_CHANNEL_SCHEMA,
            observed_fs=self.fs,
            thresholds=self.thresholds,
            timestamps_s=self.timestamps[:-1],
        )
        self.assertFalse(rejected.admitted)
        self.assertIn("manifest_sample_count_mismatch", rejected.assessment.reasons)


if __name__ == "__main__":
    unittest.main()
