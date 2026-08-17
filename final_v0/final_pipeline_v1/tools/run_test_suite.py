#!/usr/bin/env python3
"""标准库模块化测试入口 / Standard-library modular test entry point.

中文：环境不要求 pytest。调用者可用 ``--suite`` 选择单模块量化测试或全部 CPU
验收，并可将逐测试状态写入 strict JSON。

English: pytest is not required. ``--suite`` selects one quantitative module suite or
the complete CPU gate, with per-test status optionally written as strict JSON.
"""

from __future__ import annotations

import argparse
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

# 中文：在导入测试前固定本包为唯一活动源码；English: bind the V1 package first.
sys.path.insert(0, str(SRC))


SUITES = {
    # 中文：验收器本身必须有可单独执行的负例测试。
    # English: the acceptance gate has its own independently selectable negatives.
    'acceptance': TESTS / 'acceptance',
    # English: Baseline byte-identity is a first-class acceptance suite.
    # 中文：历史基线逐字节身份是正式验收套件，不是一次性辅助检查。
    'audit': TESTS / 'audit',
    "all": TESTS,
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
        "schema_version": "ppg_frailty.test_report.v1",
        "suite": arguments.suite,
        "pattern": arguments.pattern,
        "status": "passed" if result.wasSuccessful() else "failed",
        "duration_s": round(elapsed, 9),
        "counts": counts,
        "tests": sorted(result.case_records, key=lambda row: row["test_id"]),
    }
    report['test_source_snapshot'] = _test_source_snapshot()
    # 中文：CPU CI 使用 error，因此任一意外 warning 会成为失败而非静默文本。
    # English: CPU CI sets error so every unexpected warning becomes a failure.
    report['warnings_policy'] = os.environ.get('PYTHONWARNINGS', 'default')
    if arguments.report is not None:
        _write_report(arguments.report, report)
    print(json.dumps({key: report[key] for key in ["suite", "status", "counts"]}, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
