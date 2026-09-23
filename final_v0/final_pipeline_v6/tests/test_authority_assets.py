"""Packaged source rosters and frozen folds need no retired checkout files."""

import csv
import hashlib
import io

import pytest

from ppg_frailty.data import external_manifest, folds, manifest
from ppg_frailty.paths import pipeline_resource
from ppg_frailty.provenance import sha256_file
from ppg_frailty.quality import motion_reference


@pytest.mark.parametrize(("relative", "expected_sha256"), [
    (manifest.M2_FILE_MANIFEST, manifest.M2_FILE_MANIFEST_SHA256),
    (external_manifest.M2_EXTERNAL_RELATIVE_PATH, external_manifest.M2_EXTERNAL_MANIFEST_SHA256),
    (folds.M2_SPLIT_RELATIVE_PATH, folds.M2_SPLIT_FILE_SHA256),
])
def test_authority_asset_preserves_original_bytes(relative, expected_sha256):
    assert relative.parts[:2] == ("assets", "authority")
    assert sha256_file(pipeline_resource(relative)) == expected_sha256


def test_internal_authority_and_folds_do_not_need_old_tree(tmp_path):
    rows = manifest.load_m2_internal_manifest(tmp_path, verify_sources=False)
    expected = manifest.load_internal_manifest(pipeline_resource("manifests/internal_records_v2.csv"))
    assert rows == expected
    registry = folds.load_m2_frozen_registry(tmp_path)
    audit = folds.validate_frozen_memberships(registry, rows)
    assert audit.train_oof_disjoint
    assert audit.oof_partition_exact


def test_external_authority_preserves_materialized_records():
    source = pipeline_resource(external_manifest.M2_EXTERNAL_RELATIVE_PATH)
    rows = external_manifest.load_m2_external_manifest(source)
    expected = external_manifest.load_external_manifest(pipeline_resource("manifests/external_records_v2.csv"))
    assert rows == expected


def test_ptt_unit_evidence_checks_raw_sources_not_retired_script(tmp_path, monkeypatch):
    records = [row for row in external_manifest.load_m2_external_manifest(
        pipeline_resource(external_manifest.M2_EXTERNAL_RELATIVE_PATH)
    ) if row.dataset_id == external_manifest.PTT_DATASET_ID]
    provenance = external_manifest.PTT_IMU_UNIT_CONFLICT_PROVENANCE
    header = provenance["wfdb_header_declaration"]
    numeric = provenance["canonical_csv_numeric_evidence"]
    header_bytes = "\n".join(
        f"sample 16 1/{unit} 0 {channel}"
        for unit, channels in (("g", ("a_x", "a_y", "a_z")), ("deg/s", ("g_x", "g_y", "g_z")))
        for channel in channels
    ).encode()
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=motion_reference.PTT_CSV_COLUMNS)
    writer.writeheader()
    row = dict.fromkeys(motion_reference.PTT_CSV_COLUMNS, 0)
    row.update(zip(("a_x", "a_y", "a_z"), numeric["first_acceleration_xyz"], strict=True))
    row.update(zip(("g_x", "g_y", "g_z"), numeric["first_gyroscope_xyz"], strict=True))
    writer.writerow(row)
    sources = {header["relative_path"]: (header["sha256"], header_bytes),
               numeric["relative_path"]: (numeric["sha256"], stream.getvalue().encode())}
    reads = []

    def read_source(root, relative_path, expected_sha256):
        assert root == tmp_path
        digest, payload = sources[relative_path]
        assert expected_sha256 == digest
        reads.append(relative_path)
        return root / relative_path, payload

    monkeypatch.setattr(motion_reference, "_read_bound_source_bytes", read_source)
    assert motion_reference._validate_ptt_unit_conflict(tmp_path, records) == {
        motion_reference.PTT_UNRESOLVED_IMU_UNIT_STATUS: 66
    }
    assert reads == [header["relative_path"], numeric["relative_path"]]
    evidence = motion_reference.load_ptt_imu_unit_evidence(
        pipeline_resource(external_manifest.PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH),
        expected_sha256=external_manifest.PTT_IMU_UNIT_EVIDENCE_SHA256,
        expected_records=records,
    )
    assert evidence.historical_transform_sha256 == provenance["historical_code_transform"]["sha256"]
    assert evidence.acceleration_conversion == "identity_m_per_s2_no_scale"
    assert evidence.gyroscope_conversion == "degrees_per_second_to_radians_per_second"


def test_actual_source_hash_verification_is_not_relaxed(tmp_path):
    source = tmp_path / "record.csv"
    original = b"a_x,a_y,a_z\n1,2,3\n"
    source.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()
    assert motion_reference._read_bound_source_bytes(tmp_path, source.name, digest)[1] == original
    source.write_bytes(original + b"4,5,6\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        motion_reference._read_bound_source_bytes(tmp_path, source.name, digest)
