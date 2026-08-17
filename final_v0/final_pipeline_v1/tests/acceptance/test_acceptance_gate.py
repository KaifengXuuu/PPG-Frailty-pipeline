"""严格验收器的正例与负例 / Positive and negative tests for the strict gate.

中文：这些测试刻意破坏临时 fixture，证明门禁不是“永远通过”的清单脚本。
临时文件仅创建在 V1 ``artifacts/acceptance/tmp`` 下，并由测试安全清理。

English: these tests deliberately corrupt temporary fixtures so the gate proves it
is not an always-green checklist. Temporary files stay below the V1 acceptance
artifact root and are safely cleaned by ``TemporaryDirectory``.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.acceptance_gate import (
    AcceptanceFailure,
    PIPELINE_ROOT,
    REPOSITORY_ROOT,
    _validate_real_reduced_experiment,
    active_source_snapshot,
    check_real_reduced_experiment,
    check_no_fabricated_scientific_results,
    check_no_legacy_imports_or_unfinished,
    check_spec_lock,
    check_strict_json_tree,
    check_target_package,
    check_typed_containers,
    python_tree_snapshot,
)


class AcceptanceGateNegativeTests(unittest.TestCase):
    """用受控损坏验证 fail-closed 行为 / Verify fail-closed behavior."""

    def setUp(self) -> None:
        """在 V1 内建立隔离临时根 / Create an isolated temporary root in V1."""

        parent = PIPELINE_ROOT / "artifacts/acceptance/tmp"
        parent.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(prefix="gate_test_", dir=parent)
        self.root = Path(self._temporary.name)

    def tearDown(self) -> None:
        """精确清理本测试目录 / Remove exactly this test's temporary directory."""

        self._temporary.cleanup()

    def test_target_boundary_rejects_missing_files(self) -> None:
        """空树不能伪装成第 4 节实现 / An empty tree cannot satisfy section 4."""

        with self.assertRaises(AcceptanceFailure):
            check_target_package(self.root, self.root)

    def test_spec_lock_rejects_wrong_source_bytes(self) -> None:
        """任一字节变化都破坏规范锁 / Any changed byte breaks the specification lock."""

        source = self.root / "AA_TODO/3/spec.md"
        source.parent.mkdir(parents=True)
        source.write_text("changed specification\n", encoding="utf-8")
        lock = self.root / "pipeline/docs/spec/SPEC_LOCK.json"
        lock.parent.mkdir(parents=True)
        lock.write_text(
            json.dumps(
                {
                    "source_path": "AA_TODO/3/spec.md",
                    "source_sha256": "0" * 64,
                    "source_bytes": 22,
                    "source_lines": 1,
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaises(AcceptanceFailure):
            check_spec_lock(self.root / "pipeline", self.root)

    def test_strict_json_rejects_nan(self) -> None:
        """JSON NaN 必须关闭失败 / JSON NaN must fail closed."""

        artifact = self.root / "bad.json"
        artifact.write_text('{"value": NaN}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            check_strict_json_tree(self.root, self.root)

    def test_unfinished_function_is_detected(self) -> None:
        """pass-only 函数不能进入活动代码 / Pass-only functions cannot ship."""

        source = self.root / "src/ppg_frailty/unimplemented.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            '"""中英文说明 / Bilingual explanation."""\n\ndef unfinished():\n    pass\n',
            encoding="utf-8",
        )
        (self.root / "tools").mkdir()
        with self.assertRaises(AcceptanceFailure):
            check_no_legacy_imports_or_unfinished(self.root, self.root)

    def test_unsupported_metric_claim_is_rejected(self) -> None:
        """无范围的 BA 不能被称为结果 / Unscoped BA cannot be called evidence."""

        artifact = self.root / "artifacts/unsupported.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_text(
            json.dumps({"status": "passed", "balanced_accuracy": 0.99}),
            encoding="utf-8",
        )
        with self.assertRaises(AcceptanceFailure):
            check_no_fabricated_scientific_results(self.root, self.root)

    def test_synthetic_metric_scope_is_accepted(self) -> None:
        """明确 synthetic 非基准声明可通过 / Explicit non-benchmark synthetic scope passes."""

        artifact = self.root / "artifacts/synthetic.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "scientific_scope": "synthetic_contract_test_not_frailty_benchmark",
                    "balanced_accuracy": 0.5,
                }
            ),
            encoding="utf-8",
        )
        result = check_no_fabricated_scientific_results(self.root, self.root)
        self.assertEqual(result.status, "passed")

    def test_source_snapshot_changes_after_edit(self) -> None:
        """测试编辑必须使树 hash 失效 / Editing a test must invalidate the tree hash."""

        tests = self.root / "tests"
        tests.mkdir()
        path = tests / "test_example.py"
        path.write_text("# 第一版 / first version\n", encoding="utf-8")
        before = python_tree_snapshot(tests)
        path.write_text("# 第二版 / second version\n", encoding="utf-8")
        after = python_tree_snapshot(tests)
        self.assertNotEqual(before["tree_sha256"], after["tree_sha256"])

    def test_real_experiment_rejects_benchmark_scope_promotion(self) -> None:
        """reduced smoke 不得改称 benchmark / Reduced smoke cannot be promoted."""

        registry = json.loads(
            (PIPELINE_ROOT / "artifacts/experiments/reference_registry.json").read_text(
                encoding="utf-8"
            )
        )
        source = (
            PIPELINE_ROOT
            / "artifacts/experiments"
            / registry["current_passing_reference"]["path"]
        )
        target = self.root / "corrupt_experiment"
        shutil.copytree(source, target)
        result_path = target / "experiment_result.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["scientific_scope"] = "frozen_5x5_scientific_benchmark"
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(AcceptanceFailure, "smoke-only"):
            _validate_real_reduced_experiment(target, PIPELINE_ROOT)

    def test_real_experiment_rejects_duplicate_oof_participant(self) -> None:
        """subject OOF 必须精确一次 / Subject OOF must be exact-once."""

        import pyarrow as pa
        import pyarrow.parquet as pq

        registry = json.loads(
            (PIPELINE_ROOT / "artifacts/experiments/reference_registry.json").read_text(
                encoding="utf-8"
            )
        )
        source = (
            PIPELINE_ROOT
            / "artifacts/experiments"
            / registry["current_passing_reference"]["path"]
        )
        target = self.root / "duplicate_oof_experiment"
        shutil.copytree(source, target)
        subject_path = target / "oof_subject_predictions.parquet"
        table = pq.read_table(subject_path)
        identities = table.column("participant_id").to_pylist()
        identities[1] = identities[0]
        table = table.set_column(
            table.column_names.index("participant_id"),
            "participant_id",
            pa.array(identities, type=table.schema.field("participant_id").type),
        )
        pq.write_table(table, subject_path)
        with self.assertRaisesRegex(AcceptanceFailure, "exact-once"):
            _validate_real_reduced_experiment(target, PIPELINE_ROOT)


class AcceptanceGateLiveContractTests(unittest.TestCase):
    """验证当前树的基础契约 / Verify stable contracts in the current tree."""

    def test_current_spec_lock_is_byte_exact(self) -> None:
        """当前附件必须保持逐字节身份 / The attached specification remains byte exact."""

        result = check_spec_lock(PIPELINE_ROOT, REPOSITORY_ROOT)
        self.assertEqual(result.status, "passed")

    def test_typed_container_contract_is_complete(self) -> None:
        """八个规范容器保持 dataclass 字段 / Eight typed containers retain required fields."""

        result = check_typed_containers(PIPELINE_ROOT, REPOSITORY_ROOT)
        self.assertEqual(result.status, "passed")

    def test_real_reduced_feature_vector_reference_is_complete(self) -> None:
        """冻结参考须具备 exact OOF 与 train-only 证据 / Validate real evidence."""

        result = check_real_reduced_experiment(PIPELINE_ROOT, REPOSITORY_ROOT)
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.evidence["train_participants"], 23)
        self.assertEqual(result.evidence["oof_participants"], 6)
        self.assertGreater(result.evidence["retained_oof_participants"], 0)
        self.assertIs(result.evidence["outcome_metric_values_locked"], False)
        self.assertEqual(
            result.evidence["formal_runner_supported_representation"],
            "feature_vector",
        )

    def test_cpu_writer_and_gate_use_identical_source_snapshot(self) -> None:
        """CI 写入器与 gate 必须逐项同构 / Writer and gate hashes must match."""

        from tools.run_cpu_ci import _active_source_snapshot

        self.assertEqual(
            _active_source_snapshot(),
            active_source_snapshot(PIPELINE_ROOT),
        )


if __name__ == "__main__":
    unittest.main()
