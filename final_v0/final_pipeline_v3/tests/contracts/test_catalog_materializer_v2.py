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
            self.assertEqual(manifest["config_count"], 15)
            self.assertEqual(len(tuple(target.glob("*.yaml"))), 15)
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
