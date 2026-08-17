"""V2 must not consume copied V1 acceptance evidence / V2 禁止消费 V1 验收证据。"""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

class AcceptanceGenerationBoundaryTests(unittest.TestCase):
    """V1 harnesses exist only in the immutable historical namespace."""

    def _assert_historical_only(self, name: str) -> None:
        self.assertFalse((ROOT / "tools" / f"{name}.py").exists())
        self.assertTrue(
            (
                ROOT
                / "historical"
                / "v1_transition"
                / "tools"
                / f"{name}.py"
            ).is_file()
        )

    def test_v1_acceptance_harness_is_historical_only(self) -> None:
        self._assert_historical_only("acceptance_gate")

    def test_v1_cpu_ci_is_historical_only(self) -> None:
        self._assert_historical_only("run_cpu_ci")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
