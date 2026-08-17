"""CLI discovery tests; help paths must never import data or start training."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class StudyCliHelpTests(unittest.TestCase):
    def invoke(self, script: str, *arguments: str) -> str:
        completed = subprocess.run(
            [sys.executable, str(ROOT / script), *arguments],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return completed.stdout

    def test_single_pipeline_help(self) -> None:
        output = self.invoke("frailty_3class_pipeline_v2.py", "--help")
        self.assertIn("--config", output)
        self.assertIn("--resume", output)
        self.assertIn("--jobs", output)

    def test_sweep_and_grid_help(self) -> None:
        output = self.invoke("frailty_3class_sweep_v2.py", "--help")
        self.assertIn("ablation", output)
        self.assertIn("grid", output)
        self.assertIn("report", output)
        grid = self.invoke("frailty_3class_sweep_v2.py", "grid", "--help")
        self.assertIn("--vary", grid)
        self.assertIn("--output-root", grid)


if __name__ == "__main__":
    unittest.main()
