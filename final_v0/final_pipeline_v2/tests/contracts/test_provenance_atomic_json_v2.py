"""Bounded-memory and atomicity contracts for strict provenance JSON."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ppg_frailty.provenance import atomic_write_json


class AtomicWriteJsonTests(unittest.TestCase):
    def test_streamed_output_matches_the_strict_sorted_pretty_contract(self) -> None:
        payload = {
            "zeta": [3, 2, 1],
            "alpha": {"文本": "非 ASCII", "enabled": True},
        }
        expected = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "nested" / "payload.json"
            atomic_write_json(target, payload, root=root)

            self.assertEqual(target.read_text(encoding="utf-8"), expected)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), payload)
            self.assertFalse(target.with_suffix(".json.tmp").exists())

    def test_large_payload_does_not_use_dumps_or_path_write_text(self) -> None:
        payload = {
            "rows": [
                {"index": index, "value": f"row-{index:06d}"}
                for index in range(20_000)
            ]
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "large.json"
            with (
                patch(
                    "ppg_frailty.provenance.json.dumps",
                    side_effect=AssertionError("whole-payload json.dumps forbidden"),
                ),
                patch.object(
                    Path,
                    "write_text",
                    side_effect=AssertionError("whole-payload write_text forbidden"),
                ),
            ):
                atomic_write_json(target, payload, root=root)

            self.assertGreater(target.stat().st_size, 1_000_000)
            with target.open("r", encoding="utf-8") as stream:
                restored = json.load(stream)
            self.assertEqual(restored, payload)

    def test_late_strict_encoding_error_preserves_target_and_cleans_temp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "payload.json"
            original = '{"status":"previous"}\n'
            target.write_text(original, encoding="utf-8")

            with self.assertRaises(ValueError):
                atomic_write_json(
                    target,
                    {"valid_prefix": list(range(1_000)), "invalid": float("nan")},
                    root=root,
                )

            self.assertEqual(target.read_text(encoding="utf-8"), original)
            self.assertFalse(target.with_suffix(".json.tmp").exists())

    def test_success_commits_with_path_replace(self) -> None:
        calls: list[tuple[Path, Path]] = []
        original_replace = Path.replace

        def tracked_replace(source: Path, target: Path) -> Path:
            calls.append((source, target))
            return original_replace(source, target)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "payload.json"
            target.write_text('{"status":"previous"}\n', encoding="utf-8")
            with patch.object(Path, "replace", new=tracked_replace):
                atomic_write_json(target, {"status": "current"}, root=root)

            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")),
                {"status": "current"},
            )
            self.assertEqual(calls, [(target.with_suffix(".json.tmp"), target)])
            self.assertFalse(target.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
