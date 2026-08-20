from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import signal as scipy_signal

from ppg_frailty.audit.legacy_v2_bridge import (
    PHASE0_RESULT_SCHEMA_VERSION,
    Phase0Result,
    _audit_sources,
    _decision,
    _discover_static,
    _fresh_legacy_windows,
    _selected_npz_rows,
    run_legacy_v2_phase0,
)
from ppg_frailty.data.schema import CANONICAL_CHANNEL_SCHEMA
from ppg_frailty.legacy_bridge import (
    build_legacy_bridge_raw_windows,
    resolve_legacy_bridge_profile,
)


OUTPUTS = [
    "artifacts/audit/legacy_v2_manifest_record_diff.csv",
    "artifacts/audit/legacy_v2_source_hash_audit.csv",
    "artifacts/audit/legacy_v2_source_hash_audit.json",
    "artifacts/audit/legacy_v2_channel_qc.csv",
    "artifacts/audit/legacy_v2_participant_alias_map.csv",
    "artifacts/audit/legacy_v2_imu_unit_ekf_audit.csv",
    "artifacts/audit/legacy_v2_cache_audit.json",
    "artifacts/audit/legacy_v2_split_audit.json",
    "artifacts/audit/LEGACY_V2_PHASE0_DATA_AUDIT.md",
]


@dataclass(frozen=True)
class _Row:
    record_id: str
    participant_id: str
    class_id: int
    class_name: str
    role: str
    source_path: str
    source_hash: str
    fs: float
    n_samples: int
    duration_s: float
    channel_schema: tuple[str, ...] = CANONICAL_CHANNEL_SCHEMA
    channel_units: dict[str, str] | None = None
    manifest_version: str = "fixture"


def _write_signal(path: Path, samples: int, offset: float = 0.0) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(samples, dtype=np.float64) / 400.0
    matrix = np.column_stack(
        (
            np.sin(2 * np.pi * time) + offset,
            np.cos(2 * np.pi * time) + offset,
            np.zeros(samples),
            np.zeros(samples),
            np.ones(samples),
            np.zeros(samples),
            np.zeros(samples),
            np.zeros(samples),
        )
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CANONICAL_CHANNEL_SCHEMA)
        writer.writerows(matrix)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Phase0ResultTests(unittest.TestCase):
    def test_result_schema_and_phase0_hash_alias_are_stable(self) -> None:
        result = Phase0Result(
            decision="PASS",
            advisory_checks_passed=True,
            stop_reasons=(),
            limitations=(),
            outputs={"one": "/tmp/one"},
            source_specification="spec.md",
            source_specification_sha256="a" * 64,
            audit_spec_sha256="b" * 64,
            manifest_sha256="c" * 64,
            split_sha256="d" * 64,
        )
        payload = result.to_dict()
        self.assertEqual(payload["schema_version"], PHASE0_RESULT_SCHEMA_VERSION)
        self.assertEqual(payload["audit_spec_sha256"], payload["phase0_spec_sha256"])
        self.assertTrue(payload["advisory_checks_passed"])
        self.assertNotIn("training_allowed", payload)

    def test_decision_is_fail_closed(self) -> None:
        self.assertEqual(_decision(["hash_mismatch"], ["cache_unproven"])[0], "STOP")
        self.assertEqual(_decision([], ["cache_unproven"])[0], "PASS_WITH_DECLARED_LIMITATIONS")
        self.assertEqual(_decision([], [])[0], "PASS")


class Phase0ComponentTests(unittest.TestCase):
    def test_independent_discovery_uses_explicit_suffix_aliases_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            data = repository / "PPG_Testing_05_01_2026"
            for role in ("B", "R1", "R2", "R3", "R4"):
                _write_signal(data / "StudyData" / f"OLD01_03_{role}.csv", 40)
                _write_signal(data / "TestDataYoungers" / f"YNG_02_{role}.csv", 40, 1.0)
            discovered, aliases, issues = _discover_static(
                repository, data, {"OLD01": 0}
            )
            self.assertFalse(issues)
            self.assertEqual(len(discovered), 10)
            alias_by_id = {
                row["historical_file_participant_id"]: row for row in aliases
            }
            self.assertEqual(
                alias_by_id["OLD01_03"]["historical_participant_alias"], "OLD01"
            )
            self.assertEqual(
                alias_by_id["YNG_02"]["historical_participant_alias"],
                "YNG_02",
            )
            self.assertTrue(all(row["one_to_one"] == "true" for row in aliases))

    def test_source_audit_rehashes_and_reports_channel_qc_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            relative = "PPG_Testing_05_01_2026/TestDataYoungers/YNG_02_B.csv"
            source = repository / relative
            digest = _write_signal(source, 40)
            row = _Row(
                record_id="frailty3:YNG_02:B",
                participant_id="YNG_02",
                class_id=2,
                class_name="Young",
                role="B",
                source_path=relative,
                source_hash=digest,
                fs=400.0,
                n_samples=40,
                duration_s=0.1,
            )
            before = source.read_bytes()
            source_rows, channels, stops, limitations = _audit_sources(
                repository, [row], {relative: 2}, CANONICAL_CHANNEL_SCHEMA
            )
            self.assertFalse(stops)
            self.assertIn("sampling_rate_not_independently_observable_no_timestamp_column", limitations)
            self.assertEqual(source_rows[0]["source_hash_match"], "true")
            self.assertEqual(len(channels), 8)
            self.assertEqual(source.read_bytes(), before)

    def test_declared_legacy_window_regeneration_is_64_hz_15_second(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "P_01_B.csv"
            _write_signal(source, 8000)
            original_resample_poly = scipy_signal.resample_poly
            with patch(
                "ppg_frailty.legacy_bridge.signal.resample_poly",
                side_effect=original_resample_poly,
            ) as resample_poly:
                windows = _fresh_legacy_windows(source)
            self.assertNotIn("padtype", resample_poly.call_args.kwargs)
            self.assertEqual(
                np.asarray(resample_poly.call_args.args[0]).dtype,
                np.dtype("float32"),
            )
            matrix = np.loadtxt(source, delimiter=",", skiprows=1, dtype=np.float64)
            authoritative = build_legacy_bridge_raw_windows(
                {
                    "fs_hz": 400.0,
                    "ppg": matrix[:, :2],
                    "acc": matrix[:, 2:5],
                    "gyro": matrix[:, 5:8],
                },
                resolve_legacy_bridge_profile("L0"),
            ).values
            self.assertEqual(windows.dtype, np.dtype("float32"))
            self.assertEqual(windows.shape[1:], (8, 960))
            self.assertGreaterEqual(windows.shape[0], 2)
            self.assertTrue(np.isfinite(windows).all())
            np.testing.assert_array_equal(windows, authoritative)

    def test_cache_x_stream_selects_rows_and_hashes_complete_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "cache.npz"
            values = np.arange(6 * 2 * 3, dtype=np.float32).reshape(6, 2, 3)
            np.savez(target, x=values)
            selected, digest = _selected_npz_rows(
                target,
                {
                    10: np.asarray([1, 4], dtype=np.int64),
                    20: np.asarray([0, 5], dtype=np.int64),
                },
            )
            np.testing.assert_array_equal(selected[10], values[[1, 4]])
            np.testing.assert_array_equal(selected[20], values[[0, 5]])
            self.assertEqual(digest, hashlib.sha256(values.tobytes()).hexdigest())


class Phase0OrchestrationTests(unittest.TestCase):
    def test_run_writes_exact_nine_outputs_and_hash_bound_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            pipeline = repository / "final_v0/final_pipeline_v2"
            (pipeline / "manifests").mkdir(parents=True)
            (pipeline / "splits").mkdir(parents=True)
            (pipeline / "manifests/internal_records_v2.csv").write_text("fixture\n")
            (pipeline / "splits/sgkf5_repeated_grouped_5x5_v2.csv").write_text("fixture\n")
            specification = repository / "spec.md"
            specification.write_text("registered phase zero\n", encoding="utf-8")
            source_sha = hashlib.sha256(specification.read_bytes()).hexdigest()
            phase0 = {
                "manifest_path": "manifests/internal_records_v2.csv",
                "manifest_expected_rows": 0,
                "split_path": "splits/sgkf5_repeated_grouped_5x5_v2.csv",
                "required_channel_order": list(CANONICAL_CHANNEL_SCHEMA),
                "audit_outputs": OUTPUTS,
            }
            cache_payload = {
                "schema_version": "fixture.cache.v1",
                "status": "historical_cache_not_available",
                "cache_files": [],
            }
            split_payload = {"schema_version": "fixture.split.v1", "status": "PASS"}
            with (
                patch(
                    "ppg_frailty.audit.legacy_v2_bridge.load_internal_manifest",
                    return_value=[],
                ),
                patch(
                    "ppg_frailty.audit.legacy_v2_bridge._read_label_table",
                    return_value=({}, []),
                ),
                patch(
                    "ppg_frailty.audit.legacy_v2_bridge._discover_static",
                    return_value=([], [], []),
                ),
                patch(
                    "ppg_frailty.audit.legacy_v2_bridge._audit_sources",
                    return_value=([], [], [], []),
                ),
                patch(
                    "ppg_frailty.audit.legacy_v2_bridge._manifest_diff",
                    return_value=([], []),
                ),
                patch(
                    "ppg_frailty.audit.legacy_v2_bridge._audit_imu",
                    return_value=([], [], []),
                ),
                patch(
                    "ppg_frailty.audit.legacy_v2_bridge._audit_cache",
                    return_value=(cache_payload, ["historical_cache_not_available"]),
                ),
                patch(
                    "ppg_frailty.audit.legacy_v2_bridge._audit_split",
                    return_value=(split_payload, []),
                ),
            ):
                result = run_legacy_v2_phase0(
                    repository,
                    pipeline_root=pipeline,
                    phase0_spec=phase0,
                    source_specification="spec.md",
                    source_specification_sha256=source_sha,
                )
            self.assertEqual(result.decision, "PASS_WITH_DECLARED_LIMITATIONS")
            self.assertTrue(result.advisory_checks_passed)
            self.assertEqual(result.source_specification_sha256, source_sha)
            expected_spec_sha = hashlib.sha256(
                json.dumps(
                    phase0,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest()
            self.assertEqual(result.audit_spec_sha256, expected_spec_sha)
            self.assertEqual(len(result.outputs), 9)
            self.assertTrue(all(Path(path).is_file() for path in result.outputs.values()))
            source_json = json.loads(
                Path(result.outputs["legacy_v2_source_hash_audit.json"]).read_text()
            )
            self.assertEqual(
                source_json["phase0_result"]["schema_version"],
                PHASE0_RESULT_SCHEMA_VERSION,
            )
            markdown = Path(
                result.outputs["LEGACY_V2_PHASE0_DATA_AUDIT.md"]
            ).read_text(encoding="utf-8")
            self.assertIn("Static 145-record set identical:", markdown)
            self.assertIn("Current source bytes match V2 manifest:", markdown)
            self.assertIn("Historical cache traceability:", markdown)
            self.assertIn("IMU unit / calibration / EKF red flags", markdown)


if __name__ == "__main__":
    unittest.main()
