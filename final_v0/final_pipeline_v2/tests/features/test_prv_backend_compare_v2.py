"""Fixed-PPI backend comparison tests / 固定 PPI 后端对照测试。"""

from __future__ import annotations

import json
import unittest
from unittest import mock

from ppg_frailty.features import (
    evaluate_prv_backend,
    fixed_ppi_fixtures,
    run_prv_backend_comparison,
)


class PrvBackendComparisonTests(unittest.TestCase):
    """No cleaner/classifier and per-backend failure / 无清洗、无分类接线。"""

    def test_local_comparison_is_strict_json_and_uses_exact_vector(self) -> None:
        report = run_prv_backend_comparison(
            backends=("local",),
            fixture_ids=("dual_modulated",),
        )
        self.assertEqual(report["schema_version"], "ppg_frailty.prv_backend_comparison.v2")
        self.assertEqual(
            report["status"],
            "diagnostic_success_not_exact_profile_evidence",
        )
        self.assertIs(
            report["execution_authority"]["formal_optional_profile_evidence"],
            False,
        )
        self.assertFalse(report["cleaner_applied"])
        self.assertFalse(report["classifier_integrated"])
        fixture = report["fixtures"][0]
        row = fixture["backends"][0]
        self.assertEqual(row["status"], "success")
        self.assertEqual(row["input_sha256"], fixture["input_sha256"])
        self.assertEqual(row["interval_count"], len(fixed_ppi_fixtures()["dual_modulated"]))
        json.dumps(report, allow_nan=False)

    def test_missing_optional_backends_do_not_block_local(self) -> None:
        original_import = __import__("importlib").import_module

        def import_side_effect(name: str):
            if name in {"hrvanalysis", "hrv.rri", "hrv.classical"}:
                raise ModuleNotFoundError(name)
            return original_import(name)

        with mock.patch(
            "ppg_frailty.features.prv_backend_compare.importlib.import_module",
            side_effect=import_side_effect,
        ):
            report = run_prv_backend_comparison(
                backends=("local", "aura_hrv_analysis", "rhenan_hrv"),
                fixture_ids=("alternating_75bpm",),
            )
        statuses = {row["backend"]: row["status"] for row in report["fixtures"][0]["backends"]}
        self.assertEqual(statuses["local"], "success")
        self.assertEqual(statuses["aura_hrv_analysis"], "unavailable_optional_dependency")
        self.assertEqual(statuses["rhenan_hrv"], "unavailable_optional_dependency")

    def test_unknown_backend_fails_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown PRV"):
            evaluate_prv_backend([800.0, 810.0, 790.0, 805.0], "unknown")


if __name__ == "__main__":
    unittest.main()
