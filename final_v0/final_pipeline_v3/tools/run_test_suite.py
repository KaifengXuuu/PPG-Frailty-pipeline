#!/usr/bin/env python3
"""标准库模块化测试入口 / Standard-library modular test entry point.

中文：环境不要求 pytest。调用者可用 ``--suite`` 选择单模块量化测试或全部 CPU
验收，并可将逐测试状态写入 strict JSON。

English: pytest is not required. ``--suite`` selects one quantitative module suite or
the complete CPU gate, with per-test status optionally written as strict JSON.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import os
import sys
import time
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"

# 中文：在导入测试前固定本包为唯一活动源码；English: bind the V2 package first.
sys.path.insert(0, str(SRC))


SUITES = {
    # 中文：验收器本身必须有可单独执行的负例测试。
    # English: the acceptance gate has its own independently selectable negatives.
    'acceptance': TESTS / 'acceptance',
    "all": TESTS,
    "safe": TESTS,
    "data": TESTS / "data",
    "signal": TESTS / "signal",
    "artifacts": TESTS / "artifacts",
    "features": TESTS / "features",
    "models": TESTS / "models",
    "training": TESTS / "training",
    "integration": TESTS / "integration",
    "cli": TESTS / "cli",
    "contracts": TESTS / "contracts",
}

SAFE_DIRECTORIES = (
    "data",
    "signal",
    "artifacts",
    "features",
    "models",
    "training",
    "integration",
    "contracts",
    "quality",
)
SAFE_EXCLUDED_MODULE_REASONS = {
    "tests.artifacts.test_legacy_reducers_v2": "ablation_module",
    "tests.features.test_prv_backend_compare_v2": "comparison_module",
    "tests.integration.test_experiment_runner": "real_experiment_runner",
    "tests.models.test_time_scale_ablation": "ablation_module",
    "tests.models.test_time_scale_v2": "ablation_module",
    "tests.training.test_oof_aggregation_ablation_bundle": "ablation_module",
    "tests.training.test_v2_statistics_archive": "comparison_statistics_module",
}


class RecordingResult(unittest.TextTestResult):
    """记录逐测试状态 / Record an auditable status for every test case."""

    def startTest(self, test: unittest.case.TestCase) -> None:  # noqa: N802
        """记录开始时间 / Store per-test start time."""

        self._case_start = time.perf_counter()
        super().startTest(test)

    def _record(self, test: unittest.case.TestCase, status: str, detail: str | None) -> None:
        """追加 strict-JSON 兼容记录 / Append a strict-JSON-compatible row."""

        elapsed = time.perf_counter() - getattr(self, "_case_start", time.perf_counter())
        self.case_records.append(
            {
                "test_id": test.id(),
                "status": status,
                "duration_s": round(max(0.0, elapsed), 9),
                "detail": detail,
            }
        )

    def addSuccess(self, test: unittest.case.TestCase) -> None:  # noqa: N802
        super().addSuccess(test)
        self._record(test, "passed", None)

    def addFailure(self, test: unittest.case.TestCase, err: Any) -> None:  # noqa: N802
        super().addFailure(test, err)
        self._record(test, "failed", self._exc_info_to_string(err, test))

    def addError(self, test: unittest.case.TestCase, err: Any) -> None:  # noqa: N802
        super().addError(test, err)
        self._record(test, "error", self._exc_info_to_string(err, test))

    def addSkip(self, test: unittest.case.TestCase, reason: str) -> None:  # noqa: N802
        super().addSkip(test, reason)
        self._record(test, "skipped", reason)


class RecordingRunner(unittest.TextTestRunner):
    """构造带记录能力的结果对象 / Create the recording result class."""

    resultclass = RecordingResult

    def _makeResult(self) -> RecordingResult:  # noqa: N802
        result = super()._makeResult()
        result.case_records = []
        return result


def _discover(path: Path, pattern: str) -> unittest.TestSuite:
    """在指定子树发现测试 / Discover tests below one registered suite root."""

    if not path.is_dir():
        raise FileNotFoundError(f"test suite directory does not exist: {path}")
    return unittest.defaultTestLoader.discover(
        start_dir=str(path), pattern=pattern, top_level_dir=str(ROOT)
    )


def _module_name(path: Path) -> str:
    '''Return the import name for a test path / 将测试路径转换为导入名。'''

    return ".".join(path.relative_to(ROOT).with_suffix("").parts)


def _imports_pytest(path: Path) -> bool:
    '''Detect a direct pytest import without importing the test module.

    中文：safe suite 只承诺标准库 unittest。这里在测试模块导入前解析 AST，
    因此缺少 pytest 的环境不会先生成 _FailedTest 再被事后过滤。
    '''

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError):
        # 中文：读取或语法错误必须由 unittest 正常暴露，不能伪装成 pytest 排除。
        # English: ordinary source failures remain visible to unittest discovery.
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name == "pytest" or alias.name.startswith("pytest.")
                for alias in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "pytest" or module.startswith("pytest."):
                return True
    return False


def _safe_test_files(pattern: str):
    '''Yield safe candidates before any test module import / 导入前枚举候选文件。'''

    for directory in SAFE_DIRECTORIES:
        suite_root = TESTS / directory
        if not suite_root.is_dir():
            raise FileNotFoundError(f"test suite directory does not exist: {suite_root}")
        for path in sorted(suite_root.rglob("*.py")):
            if fnmatch.fnmatchcase(path.name, pattern):
                yield path


def _safe_suite(pattern: str) -> tuple[unittest.TestSuite, list[dict[str, str]]]:
    '''Build a prefiltered non-scientific safe smoke suite.

    中文：科学/heavy comparison、ablation、真实实验 runner 及 pytest 模块在逐文件
    discovery 前排除；可信的非科学 comparison contract negative tests 保留在 safe
    suite 中，验证归档边界但不执行科学 comparison。
    '''

    combined = unittest.TestSuite()
    excluded: list[dict[str, str]] = []
    for path in _safe_test_files(pattern):
        module = _module_name(path)
        explicit_reason = SAFE_EXCLUDED_MODULE_REASONS.get(module)
        if explicit_reason is not None:
            excluded.append({"module": module, "reason": explicit_reason})
            continue
        if _imports_pytest(path):
            excluded.append({"module": module, "reason": "requires_pytest"})
            continue
        # Exact filename discovery preserves unittest import/error behaviour while
        # preventing excluded siblings from being imported by directory discovery.
        combined.addTests(_discover(path.parent, path.name))
    return combined, sorted(excluded, key=lambda row: (row["module"], row["reason"]))


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    """原子写 strict JSON / Atomically write a strict JSON report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _test_source_snapshot() -> dict[str, Any]:
    '''绑定报告与当前测试源码 / Bind the report to the current test source tree.

    中文：仅保存路径、字节数和逐文件 SHA-256，不复制源码。树 hash 使用
    canonical strict JSON，因此主机路径分隔符不会改变身份。

    English: only relative paths, byte sizes, and per-file SHA-256 values are stored.
    Canonical strict JSON makes the tree identity independent of host path separators.
    '''

    rows = []
    for path in sorted(TESTS.rglob('*.py')):
        payload = path.read_bytes()
        rows.append(
            {
                'path': path.relative_to(TESTS).as_posix(),
                'bytes': len(payload),
                'sha256': hashlib.sha256(payload).hexdigest(),
            }
        )
    encoded = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')
    return {
        'algorithm': 'sha256(canonical_json(file_path,bytes,sha256))',
        'file_count': len(rows),
        'tree_sha256': hashlib.sha256(encoded).hexdigest(),
        'files': rows,
    }


def main(argv: list[str] | None = None) -> int:
    """运行选定套件并返回 POSIX 状态 / Run a selected suite and return its status."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(SUITES), default="all")
    parser.add_argument("--pattern", default="test_*.py")
    parser.add_argument("--verbosity", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args(argv)

    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    started = time.perf_counter()
    safe_exclusions: list[dict[str, str]] = []
    if arguments.suite == "safe":
        suite, safe_exclusions = _safe_suite(arguments.pattern)
    else:
        suite = _discover(SUITES[arguments.suite], arguments.pattern)
    runner = RecordingRunner(verbosity=arguments.verbosity)
    result = runner.run(suite)
    elapsed = time.perf_counter() - started
    counts = {
        "run": result.testsRun,
        "passed": sum(row["status"] == "passed" for row in result.case_records),
        "failed": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
    }
    report = {
        "schema_version": "ppg_frailty.test_report.v2",
        "pipeline_generation": "final_pipeline_v2",
        "suite": arguments.suite,
        "pattern": arguments.pattern,
        "status": "passed" if result.wasSuccessful() else "failed",
        "duration_s": round(elapsed, 9),
        "counts": counts,
        "tests": sorted(result.case_records, key=lambda row: row["test_id"]),
    }
    report['test_source_snapshot'] = _test_source_snapshot()
    if arguments.suite == "safe":
        report['excluded_modules'] = safe_exclusions
        report['excluded_comparison_modules'] = [
            row['module']
            for row in safe_exclusions
            if row['reason'] in {
                'ablation_module',
                'comparison_module',
                'comparison_statistics_module',
            }
        ]
        report['excluded_pytest_modules'] = [
            row['module'] for row in safe_exclusions if row['reason'] == 'requires_pytest'
        ]
        report['excluded_execution_modules'] = [
            row['module']
            for row in safe_exclusions
            if row['reason'] == 'real_experiment_runner'
        ]
        report['excluded_suites'] = ['acceptance', 'cli']
    # 中文：CPU CI 使用 error，因此任一意外 warning 会成为失败而非静默文本。
    # English: CPU CI sets error so every unexpected warning becomes a failure.
    report['warnings_policy'] = os.environ.get('PYTHONWARNINGS', 'default')
    if arguments.report is not None:
        _write_report(arguments.report, report)
    print(json.dumps({key: report[key] for key in ["suite", "status", "counts"]}, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
