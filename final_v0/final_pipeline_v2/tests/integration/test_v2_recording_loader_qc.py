"""Non-scientific integration checks for manifest-bound physical recording QC."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ppg_frailty.data.schema import CANONICAL_CHANNEL_SCHEMA
from ppg_frailty.pipeline import PipelinePaths, _load_record


_UNITS = {
    "RED": "raw_device_counts_adc_scale_unknown",
    "IR": "raw_device_counts_adc_scale_unknown",
    "AX": "g_source_declared",
    "AY": "g_source_declared",
    "AZ": "g_source_declared",
    "GX": "degree_per_second_source_declared",
    "GY": "degree_per_second_source_declared",
    "GZ": "degree_per_second_source_declared",
}


class FormalRecordingLoaderQCTests(unittest.TestCase):
    def _fixture(self, values: np.ndarray, *, fs: float = 2.0):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        root = Path(temporary.name)
        source = root / "raw" / "fixture.csv"
        source.parent.mkdir(parents=True)
        np.savetxt(
            source,
            values,
            delimiter=",",
            header=",".join(CANONICAL_CHANNEL_SCHEMA),
            comments="",
        )
        payload = source.read_bytes()
        row = SimpleNamespace(
            record_id="fixture_record",
            source_path="raw/fixture.csv",
            source_hash=hashlib.sha256(payload).hexdigest(),
            channel_schema=tuple(CANONICAL_CHANNEL_SCHEMA),
            channel_units=tuple(_UNITS.items()),
            n_samples=int(values.shape[0]),
            fs=float(fs),
            duration_s=float(values.shape[0] / fs),
            manifest_version="fixture_manifest_v2",
        )
        return temporary, row, PipelinePaths(root, root)

    @staticmethod
    def _valid_values() -> np.ndarray:
        sample = np.arange(20, dtype=np.float64)
        return np.column_stack(
            [sample * (index + 1.0) + 0.01 * sample**2 for index in range(8)]
        )

    def test_full_record_is_qc_checked_before_reduced_slice(self) -> None:
        temporary, row, paths = self._fixture(self._valid_values())
        self.addCleanup(temporary.cleanup)
        loaded = _load_record(row, paths, max_samples=4)
        self.assertEqual(loaded["ppg"].shape, (4, 2))
        self.assertEqual(loaded["full_record_n_samples_before_slice"], 20)
        self.assertEqual(loaded["returned_n_samples_after_slice"], 4)
        self.assertEqual(loaded["recording_qc"]["status"], "pass")
        self.assertEqual(
            loaded["recording_qc"]["metrics"]["observed_n_samples"], 20
        )
        self.assertEqual(
            loaded["recording_qc_profile"]["minimum_duration_s"], 5.0
        )
        self.assertFalse(
            loaded["recording_qc"]["metrics"]["device_dependent_checks_executed"]
        )

    def test_flatline_is_rejected_by_physical_admission(self) -> None:
        values = self._valid_values()
        values[:, 0] = 1.0
        temporary, row, paths = self._fixture(values)
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(ValueError, "flatline"):
            _load_record(row, paths, max_samples=None)

    def test_any_nonfinite_run_is_rejected(self) -> None:
        values = self._valid_values()
        values[10, 3] = np.nan
        temporary, row, paths = self._fixture(values)
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(ValueError, "excessive_nonfinite_gap"):
            _load_record(row, paths, max_samples=4)

    def test_swap_restore_during_numeric_parse_cannot_change_hashed_buffer(self) -> None:
        temporary, row, paths = self._fixture(self._valid_values())
        self.addCleanup(temporary.cleanup)
        source = paths.repository_root / row.source_path
        original_payload = source.read_bytes()
        original_loadtxt = np.loadtxt

        def swap_path_then_restore(buffer, *args, **kwargs):
            source.write_bytes(b"tampered path bytes")
            try:
                return original_loadtxt(buffer, *args, **kwargs)
            finally:
                source.write_bytes(original_payload)

        with patch(
            "ppg_frailty.pipeline.np.loadtxt",
            side_effect=swap_path_then_restore,
        ):
            loaded = _load_record(row, paths, max_samples=4)
        np.testing.assert_array_equal(loaded["ppg"][0], (0.0, 0.0))
        identity = loaded["recording_qc"]["source_byte_identity"]
        self.assertEqual(identity["read_operation_count"], 1)
        self.assertTrue(identity["header_parsed_from_same_buffer"])
        self.assertTrue(identity["numeric_values_parsed_from_same_buffer"])


if __name__ == "__main__":
    unittest.main()
