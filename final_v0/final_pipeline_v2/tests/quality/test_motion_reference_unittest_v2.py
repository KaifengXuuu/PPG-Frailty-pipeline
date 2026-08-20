"""Non-scientific tests for canonical source-bound motion entry boundaries."""

from __future__ import annotations

import inspect
import hashlib
import copy
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ppg_frailty.data.external_manifest import (
    M2_EXTERNAL_RELATIVE_PATH,
    PTT_ADOPTED_ACCELERATION_CONVERSION,
    PTT_ADOPTED_ACCELERATION_UNIT,
    PTT_ADOPTED_GYROSCOPE_CONVERSION,
    PTT_ADOPTED_GYROSCOPE_UNIT,
    PTT_DATASET_ID,
    PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH,
    PTT_IMU_UNIT_EVIDENCE_SHA256,
    load_m2_external_manifest,
)
from ppg_frailty.quality.motion import MotionFoldJob
from ppg_frailty.quality.motion_reference import (
    PttImuUnitEvidenceRequired,
    load_ptt_imu_unit_evidence,
    run_formal_internal_motion_reference,
    run_formal_ptt_motion_reference,
    _load_internal_numeric_source,
    _validate_ptt_unit_conflict,
    verify_formal_internal_source_evidence,
    verify_formal_ptt_source_evidence,
)
from ppg_frailty.data.qc import physical_recording_qc_profile_v2
from ppg_frailty.provenance import stable_payload_sha256
from ppg_frailty.signal.imu import convert_acceleration, convert_gyro
import ppg_frailty.quality.motion_reference as motion_reference
import ppg_frailty.quality.motion_runner as motion_runner
from ppg_frailty.contracts import ManifestRow
from ppg_frailty.quality.motion_runner import (
    FormalMotionEntryRequiredError,
    MotionWindowExample,
    run_internal_motion_oof,
    run_ptt_external_evaluation,
)


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PIPELINE_ROOT.parents[1]


class MotionReferenceBoundaryTests(unittest.TestCase):
    @staticmethod
    def _valid_formal_ptt_source_evidence_fixture() -> dict[str, object]:
        records = tuple(
            row
            for row in load_m2_external_manifest(
                REPOSITORY_ROOT / M2_EXTERNAL_RELATIVE_PATH
            )
            if row.dataset_id == PTT_DATASET_ID
        )
        participants = {row.subject_id for row in records}
        record_ids = {row.record_id for row in records}
        unit_evidence_path = (
            REPOSITORY_ROOT / PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH
        ).resolve()
        unit_evidence = load_ptt_imu_unit_evidence(
            unit_evidence_path,
            expected_sha256=PTT_IMU_UNIT_EVIDENCE_SHA256,
            expected_records=records,
        )

        def per_record(value: str) -> dict[str, str]:
            return {record_id: value for record_id in record_ids}

        mapping_hash = stable_payload_sha256(
            dict(motion_reference.PTT_CHANNEL_MAPPING_PROVENANCE)
        )
        resampling_hash = stable_payload_sha256(
            {
                "source_fs_hz": 500.0,
                "target_fs_hz": 400.0,
                "up": 4,
                "down": 5,
                "method": "scipy_signal_resample_poly_anti_alias_line_pad_v2",
                "axis": 0,
            }
        )
        source_schema_hash = stable_payload_sha256(
            ["pleth_1", "pleth_2", "AX", "AY", "AZ", "GX", "GY", "GZ"]
        )
        target_schema_hash = stable_payload_sha256(
            ["RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ"]
        )
        payload: dict[str, object] = {
            "schema_version": "ppg_frailty.formal_ptt_motion_source_evidence.v2",
            "formal_entry_id": motion_reference.FORMAL_PTT_MOTION_ENTRY_ID,
            "repository_root": str(REPOSITORY_ROOT.resolve()),
            "source_manifest_path": str(
                (REPOSITORY_ROOT / M2_EXTERNAL_RELATIVE_PATH).resolve()
            ),
            "source_manifest_sha256": motion_reference.M2_EXTERNAL_MANIFEST_SHA256,
            "dataset_id": PTT_DATASET_ID,
            "record_count": len(records),
            "participant_count": len(participants),
            "activity_counts": {"sit": 22, "walk": 22, "run": 22},
            "manifest_roster_sha256":
                motion_reference._ptt_manifest_roster_sha256(records),
            "record_source_sha256_by_id": {
                row.record_id: row.checksum_sha256 for row in records
            },
            "distal_mapping_sha256_by_id": per_record(mapping_hash),
            "resampling_config_sha256_by_id": per_record(resampling_hash),
            "adapted_source_values_sha256_by_id": per_record("a" * 64),
            "adapted_output_values_sha256_by_id": per_record("b" * 64),
            "source_channel_schema_sha256_by_id":
                per_record(source_schema_hash),
            "target_channel_schema_sha256_by_id":
                per_record(target_schema_hash),
            "calibration_source_role": motion_reference.PTT_STATIC_CALIBRATION_ROLE,
            "calibration_artifact_sha256_by_participant": {
                participant: "c" * 64 for participant in participants
            },
            "record_ekf_lineage_sha256_by_id": per_record("d" * 64),
            "unit_evidence_artifact_sha256": unit_evidence.artifact_sha256,
            "unit_evidence_artifact_path": str(unit_evidence_path),
            "unit_evidence_schema_version": unit_evidence.schema_version,
            "unit_evidence_decision_id": unit_evidence.decision_id,
            "unit_evidence_decision_date": unit_evidence.decision_date,
            "unit_evidence_decision_authority": unit_evidence.decision_authority,
            "unit_evidence_resolution_basis": unit_evidence.resolution_basis,
            "unit_evidence_acceleration_conversion":
                unit_evidence.acceleration_conversion,
            "unit_evidence_acceleration_decision_basis":
                unit_evidence.acceleration_decision_basis,
            "unit_evidence_gyroscope_conversion":
                unit_evidence.gyroscope_conversion,
            "unit_evidence_gyroscope_decision_basis":
                unit_evidence.gyroscope_decision_basis,
            "unit_evidence_wfdb_header_sha256":
                unit_evidence.wfdb_header_sha256,
            "unit_evidence_numeric_evidence_sha256":
                unit_evidence.numeric_evidence_sha256,
            "unit_evidence_historical_transform_sha256":
                unit_evidence.historical_transform_sha256,
            "acceleration_unit": unit_evidence.acceleration_unit,
            "gyroscope_unit": unit_evidence.gyroscope_unit,
            "ekf_config_sha256": "e" * 64,
            "tensor_schema_sha256":
                motion_reference.MOTION_NETWORK_SCHEMA_SHA256,
            "materialized_window_count": 1,
            "materialized_window_values_sha256": "f" * 64,
        }
        payload["source_evidence_sha256"] = stable_payload_sha256(payload)
        return payload

    @staticmethod
    def _valid_formal_source_evidence_fixture() -> dict[str, object]:
        rows = tuple(
            motion_reference.load_m2_internal_manifest(
                REPOSITORY_ROOT, verify_sources=True
            )
        )
        participants = sorted({row.participant_id for row in rows})
        record_ids = {row.record_id for row in rows}
        baseline_by_participant = {
            participant: next(
                row.record_id
                for row in rows
                if row.participant_id == participant and row.role == "B"
            )
            for participant in participants
        }
        physical_by_id = {
            row.record_id: {
                "schema_version": "ppg_frailty.recording_qc_admission.v2",
                "record_id": row.record_id,
                "manifest_version": row.manifest_version,
                "source_hash": row.source_hash,
                "admitted": True,
                "status": "pass",
                "reasons": [],
                "metrics": {"device_dependent_checks_executed": False},
                "device_dependent_qc_status":
                    "deferred_missing_device_metadata",
                "sqi_or_classifier_effect":
                    "none_recording_safety_admission_only",
            }
            for row in rows
        }
        payload: dict[str, object] = {
            "schema_version":
                motion_reference.FORMAL_INTERNAL_SOURCE_EVIDENCE_SCHEMA,
            "formal_entry_id": motion_reference.FORMAL_INTERNAL_MOTION_ENTRY_ID,
            "repository_root": str(REPOSITORY_ROOT.resolve()),
            "source_manifest_path": str(
                (REPOSITORY_ROOT / motion_reference.M2_FILE_MANIFEST).resolve()
            ),
            "source_manifest_sha256":
                motion_reference.M2_FILE_MANIFEST_SHA256,
            "dataset_version_id": motion_reference.M2_DATASET_VERSION_ID,
            "record_count": len(rows),
            "participant_count": len(participants),
            "role_counts": {
                role: sum(row.role == role for row in rows)
                for role in sorted({row.role for row in rows})
            },
            "manifest_roster_sha256":
                motion_reference._manifest_roster_sha256(rows),
            "record_source_sha256_by_id": {
                row.record_id: row.source_hash for row in rows
            },
            "physical_recording_qc_profile":
                physical_recording_qc_profile_v2(),
            "physical_recording_qc_evidence_by_id": physical_by_id,
            "all_records_physical_qc_admitted": True,
            "calibration_source_role": "same_participant_B_only",
            "calibration_source_record_id_by_participant":
                baseline_by_participant,
            "calibration_artifact_sha256_by_participant": {
                participant: "a" * 64 for participant in participants
            },
            "record_ekf_lineage_sha256_by_id": {
                record_id: "b" * 64 for record_id in record_ids
            },
            "record_motion_values_sha256_by_id": {
                record_id: "c" * 64 for record_id in record_ids
            },
            "ekf_config_sha256": "d" * 64,
            "tensor_schema_sha256":
                motion_reference.MOTION_NETWORK_SCHEMA_SHA256,
            "materialized_window_count": 1,
            "materialized_window_values_sha256": "e" * 64,
            "source_loader_id":
                "same_hashed_bytes_csv_header_shape_finite_manifest_physical_qc_v2",
        }
        payload["source_evidence_sha256"] = stable_payload_sha256(payload)
        return payload

    @staticmethod
    def _fixture_row(path: Path, payload: bytes) -> ManifestRow:
        n_samples = payload.count(b"\n") - 1
        return ManifestRow(
            record_id="fixture",
            participant_id="P01",
            class_id=0,
            class_name="Pre-Frail",
            class_name_provenance_alias="pre_frail",
            class_source="fixture",
            label_record_id="fixture",
            role="B",
            source_path=path.name,
            source_hash=hashlib.sha256(payload).hexdigest(),
            source_version="fixture",
            fs=400.0,
            n_samples=n_samples,
            duration_s=n_samples / 400.0,
            channel_schema=("RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ"),
            channel_units={
                "RED": "count", "IR": "count",
                "AX": "g_source_declared", "AY": "g_source_declared",
                "AZ": "g_source_declared",
                "GX": "degree_per_second_source_declared",
                "GY": "degree_per_second_source_declared",
                "GZ": "degree_per_second_source_declared",
            },
            synchrony_status="fixture",
            reference_available=False,
            qc_status="fixture",
            qc_reasons=(),
            manifest_version="fixture",
        )

    @staticmethod
    def _fixture_payload() -> bytes:
        rows = ["RED,IR,AX,AY,AZ,GX,GY,GZ"]
        rows.extend(
            ",".join(str(index + offset) for offset in range(1, 9))
            for index in range(2000)
        )
        return ("\n".join(rows) + "\n").encode("utf-8")

    def test_internal_source_parses_the_exact_hashed_bytes(self) -> None:
        payload = self._fixture_payload()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            source.write_bytes(payload)
            row = self._fixture_row(source, payload)
            original_loadtxt = np.loadtxt

            def mutate_path_then_parse(buffer, *args, **kwargs):
                source.write_bytes(payload.replace(b"1,2,3", b"7,7,7"))
                return original_loadtxt(buffer, *args, **kwargs)

            with patch(
                "ppg_frailty.quality.motion_reference.np.loadtxt",
                side_effect=mutate_path_then_parse,
            ):
                values, evidence = _load_internal_numeric_source(root, row)
            np.testing.assert_array_equal(values[0], np.arange(1.0, 9.0))
            self.assertIs(evidence["admitted"], True)

    def test_internal_source_tamper_before_read_fails_hash_validation(self) -> None:
        payload = self._fixture_payload()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            source.write_bytes(payload)
            row = self._fixture_row(source, payload)
            source.write_bytes(payload.replace(b"1,2,3", b"7,7,7"))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                _load_internal_numeric_source(root, row)

    def test_canonical_signatures_have_no_injected_data_or_model_hooks(self) -> None:
        forbidden = {
            "examples",
            "fold_jobs",
            "fit_model",
            "predict_probability",
            "load_frozen_model",
            "model",
            "tensor",
        }
        internal = set(
            inspect.signature(run_formal_internal_motion_reference).parameters
        )
        external = set(
            inspect.signature(run_formal_ptt_motion_reference).parameters
        )
        self.assertFalse(forbidden & internal)
        self.assertFalse(forbidden & external)

        obsolete_authorization = {
            "formal_run_authorization",
            "_canonical_entry_token",
        }
        internal_core = set(
            inspect.signature(motion_runner._run_internal_motion_oof_impl).parameters
        )
        external_core = set(
            inspect.signature(
                motion_runner._run_ptt_external_evaluation_impl
            ).parameters
        )
        self.assertFalse(obsolete_authorization & internal_core)
        self.assertFalse(obsolete_authorization & external_core)

    def test_injected_internal_and_ptt_runners_cannot_enter_formal(self) -> None:
        example = MotionWindowExample(
            window_id="arbitrary",
            participant_id="arbitrary",
            file_id="arbitrary",
            role_or_activity="B",
            activity_label=0,
            values=np.zeros((11, 3200), dtype=np.float32),
            dataset_id="arbitrary",
        )
        job = MotionFoldJob(
            repeat_index=0,
            fold_index=0,
            split_seed=42,
            training_seed=42,
            train_participant_ids=("arbitrary",),
            oof_participant_ids=(),
        )

        def must_not_run(*_args, **_kwargs):
            raise AssertionError("injected callback reached")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(
                FormalMotionEntryRequiredError, "injected examples"
            ):
                run_internal_motion_oof(
                    [example],
                    [job],
                    fit_model=must_not_run,
                    predict_probability=must_not_run,
                    model_input_schema_path=root / "arbitrary.json",
                    expected_model_input_schema_sha256="0" * 64,
                    output_dir=root / "internal",
                    execution_mode="formal",
                )
            with self.assertRaisesRegex(
                FormalMotionEntryRequiredError, "injected PTT"
            ):
                run_ptt_external_evaluation(
                    [example],
                    internal_evidence_path=root / "arbitrary.json",
                    expected_internal_evidence_sha256="0" * 64,
                    ptt_split_csv=root / "arbitrary.csv",
                    load_frozen_model=must_not_run,
                    predict_probability=must_not_run,
                    output_dir=root / "ptt",
                )

    def test_ptt_stops_at_structured_unit_conflict_before_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(PttImuUnitEvidenceRequired) as captured:
                run_formal_ptt_motion_reference(
                    REPOSITORY_ROOT,
                    internal_evidence_path=root / "not_reached.json",
                    expected_internal_evidence_sha256="0" * 64,
                    output_dir=root / "not_reached",
                )
        payload = captured.exception.payload
        self.assertFalse(payload["ready"])
        self.assertFalse(payload["unit_guessing_allowed"])
        self.assertEqual(
            payload["concrete_conflict_evidence"]["status"],
            "project_resolved_v2_036_source_manifest_conflict_retained",
        )
        self.assertEqual(
            payload["required_evidence_sha256"],
            PTT_IMU_UNIT_EVIDENCE_SHA256,
        )

    def test_canonical_v2_036_units_are_hash_bound_and_acceleration_is_identity(
        self,
    ) -> None:
        records = [
            row
            for row in load_m2_external_manifest(
                REPOSITORY_ROOT / M2_EXTERNAL_RELATIVE_PATH
            )
            if row.dataset_id == PTT_DATASET_ID
        ]
        evidence = load_ptt_imu_unit_evidence(
            REPOSITORY_ROOT / PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH,
            expected_sha256=PTT_IMU_UNIT_EVIDENCE_SHA256,
            expected_records=records,
        )
        self.assertEqual(evidence.acceleration_unit, PTT_ADOPTED_ACCELERATION_UNIT)
        self.assertEqual(
            evidence.acceleration_conversion,
            PTT_ADOPTED_ACCELERATION_CONVERSION,
        )
        self.assertEqual(evidence.gyroscope_unit, PTT_ADOPTED_GYROSCOPE_UNIT)
        self.assertEqual(
            evidence.gyroscope_conversion,
            PTT_ADOPTED_GYROSCOPE_CONVERSION,
        )
        source_acceleration = np.asarray([[4.298409, 1.371349, -8.450766]])
        np.testing.assert_array_equal(
            convert_acceleration(source_acceleration, evidence.acceleration_unit),
            source_acceleration,
        )
        source_gyroscope = np.asarray([[0.007759, -0.000482, 0.004583]])
        np.testing.assert_allclose(
            convert_gyro(source_gyroscope, evidence.gyroscope_unit),
            source_gyroscope * (np.pi / 180.0),
            rtol=0.0,
            atol=1e-15,
        )

    def test_g_acceleration_or_tampered_artifact_cannot_bypass_v2_036(self) -> None:
        records = [
            row
            for row in load_m2_external_manifest(
                REPOSITORY_ROOT / M2_EXTERNAL_RELATIVE_PATH
            )
            if row.dataset_id == PTT_DATASET_ID
        ]
        canonical_path = REPOSITORY_ROOT / PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH
        evidence = load_ptt_imu_unit_evidence(
            canonical_path,
            expected_sha256=PTT_IMU_UNIT_EVIDENCE_SHA256,
            expected_records=records,
        )
        with self.assertRaisesRegex(ValueError, r"source m/s\^2"):
            replace(
                evidence,
                acceleration_unit="g",
                acceleration_conversion="g_to_m_per_s2",
            ).validate(records)
        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / "unit_evidence.json"
            tampered.write_bytes(
                canonical_path.read_bytes().replace(
                    b'"identity_m_per_s2_no_scale"',
                    b'"g_to_m_per_s2"',
                )
            )
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_ptt_imu_unit_evidence(
                    tampered,
                    expected_sha256=PTT_IMU_UNIT_EVIDENCE_SHA256,
                    expected_records=records,
                )

    def test_exact_unit_evidence_reaches_materialization_without_readiness_audit(
        self,
    ) -> None:
        marker = RuntimeError("materialization reached")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(
                motion_runner,
                "audit_ptt_external_readiness",
                side_effect=AssertionError(
                    "readiness audit must not authorize execution"
                ),
            ) as readiness, patch.object(
                motion_reference,
                "_build_ptt_materialization",
                side_effect=marker,
            ) as materialize:
                with self.assertRaisesRegex(RuntimeError, "materialization reached"):
                    run_formal_ptt_motion_reference(
                        REPOSITORY_ROOT,
                        internal_evidence_path=root / "not_read.json",
                        expected_internal_evidence_sha256="0" * 64,
                        output_dir=root / "not_reached",
                        unit_evidence_path=(
                            REPOSITORY_ROOT / PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH
                        ),
                        expected_unit_evidence_sha256=(
                            PTT_IMU_UNIT_EVIDENCE_SHA256
                        ),
                    )
            readiness.assert_not_called()
            materialize.assert_called_once()

    def test_ptt_builder_shape_and_verifier_are_coherent_and_tamper_closed(
        self,
    ) -> None:
        evidence = self._valid_formal_ptt_source_evidence_fixture()
        self.assertEqual(verify_formal_ptt_source_evidence(evidence), ())
        tampered = copy.deepcopy(evidence)
        tampered["unit_evidence_acceleration_conversion"] = "g_to_m_per_s2"
        unsigned = {
            key: value
            for key, value in tampered.items()
            if key != "source_evidence_sha256"
        }
        tampered["source_evidence_sha256"] = stable_payload_sha256(unsigned)
        self.assertIn(
            "formal_ptt_resolved_acceleration_unit_invalid",
            verify_formal_ptt_source_evidence(tampered),
        )

    def test_ptt_derived_norm_validation_is_ulp_aware(self) -> None:
        records = [
            row
            for row in load_m2_external_manifest(
                REPOSITORY_ROOT / M2_EXTERNAL_RELATIVE_PATH
            )
            if row.dataset_id == PTT_DATASET_ID
        ]
        self.assertEqual(
            _validate_ptt_unit_conflict(REPOSITORY_ROOT, records),
            {"declared_g_but_values_and_code_inference_conflict": 66},
        )

    def test_formal_source_evidence_rejects_resealed_physical_qc_tamper(self) -> None:
        evidence = self._valid_formal_source_evidence_fixture()
        self.assertEqual(verify_formal_internal_source_evidence(evidence), ())
        tampered = copy.deepcopy(evidence)
        qc_by_id = tampered["physical_recording_qc_evidence_by_id"]
        record_id = next(iter(qc_by_id))
        qc_by_id[record_id]["admitted"] = False
        unsigned = {
            key: value
            for key, value in tampered.items()
            if key != "source_evidence_sha256"
        }
        tampered["source_evidence_sha256"] = stable_payload_sha256(unsigned)
        self.assertIn(
            "formal_source_physical_qc_evidence_drift",
            verify_formal_internal_source_evidence(tampered),
        )


if __name__ == "__main__":
    unittest.main()
