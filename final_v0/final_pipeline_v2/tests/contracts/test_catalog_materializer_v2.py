"""Atomic, non-executing formal catalogue materialization contracts."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import materialize_reference_configs as materializer


ROOT = Path(__file__).resolve().parents[2]


class CatalogMaterializerContracts(unittest.TestCase):
    def test_catalog_is_complete_and_nonoverwriting(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests/contracts") as raw:
            target = Path(raw) / "line_a"
            manifest = materializer.materialize_catalog(
                line="line_a",
                output_dir=target,
            )
            self.assertEqual(manifest["config_count"], 13)
            self.assertEqual(manifest["candidate_count"], 11)
            self.assertEqual(manifest["matched_comparator_count"], 1)
            self.assertEqual(manifest["ensemble_comparison_count"], 1)
            self.assertEqual(len(tuple(target.glob("*.yaml"))), 13)
            self.assertTrue(
                (
                    target
                    / "comparison_inception_full_member0_comparator_line_a_v2.yaml"
                ).is_file()
            )
            self.assertFalse(
                (target / "comparison_inception_matrix_member0_comparator_line_a_v2.yaml").exists()
            )
            self.assertTrue((target / "catalog_manifest.json").is_file())
            with self.assertRaises(FileExistsError):
                materializer.materialize_catalog(
                    line="line_a",
                    output_dir=target,
                )

    def test_mid_write_failure_leaves_no_target_or_stage(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests/contracts") as raw:
            parent = Path(raw)
            target = parent / "failed"
            original = materializer._atomic_text
            calls = 0

            def fail_third(path: Path, content: str) -> None:
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError("injected_catalog_write_failure")
                original(path, content)

            with patch.object(materializer, "_atomic_text", side_effect=fail_third):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "injected_catalog_write_failure",
                ):
                    materializer.materialize_catalog(
                        line="line_a",
                        output_dir=target,
                    )
            self.assertFalse(target.exists())
            self.assertEqual(tuple(parent.glob(".failed.stage-*")), ())


if __name__ == "__main__":
    unittest.main()
