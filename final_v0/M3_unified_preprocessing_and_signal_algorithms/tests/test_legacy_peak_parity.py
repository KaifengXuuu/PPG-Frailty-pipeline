"""历史 peak 同输入 parity / Same-input parity for legacy peak implementations."""

from __future__ import annotations

import unittest

from tools.legacy_peak_parity import run_legacy_peak_parity


class LegacyPeakParityTests(unittest.TestCase):
    """冻结重复实现的一致性及已知分叉 / Freeze parity and known divergence."""

    @classmethod
    def setUpClass(cls) -> None:
        """只运行一次 AST 隔离审计 / Run the isolated AST audit once."""

        cls.report = run_legacy_peak_parity()

    def test_funcs_and_ppg_duplicates_are_exact(self) -> None:
        """funcs.py 与 ppg.py 重复函数必须逐值一致 / Duplicates remain exact."""

        duplicate = self.report["funcs_ppg_duplicate_parity"]
        self.assertTrue(duplicate["exact"])
        self.assertEqual(duplicate["peak_count"], 36)
        self.assertEqual(
            duplicate["peak_indices_sha256"],
            "6c473708ce464984ab63f629afe6cc36abce0f3f53e12c5f2fccc7d9f5913504",
        )

    def test_classifier_alias_is_exact_and_adaptation_is_distinct(self) -> None:
        """分类器 alias 相同，但跨实现差异须显式 / Alias exact; adaptation distinct."""

        alias = self.report["classifier_alias_parity"]
        comparison = self.report["cross_implementation_comparison"]
        self.assertTrue(alias["exact"])
        self.assertEqual(alias["peak_count"], 35)
        self.assertEqual(
            alias["peak_indices_sha256"],
            "595782a5d6843b716add69752e932f749505c36b76277090aef833e0c78c4bb2",
        )
        self.assertFalse(comparison["exact"])
        self.assertEqual(comparison["funcs_ppg_only_indices"], [8318])
        self.assertEqual(comparison["classifier_only_indices"], [])
        self.assertEqual(
            self.report["status"], "pass_with_expected_cross_implementation_difference"
        )


if __name__ == "__main__":
    # 中文：兼容 unittest 直接执行 / English: Preserve direct unittest execution.
    unittest.main()
