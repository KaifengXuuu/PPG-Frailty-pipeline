#!/usr/bin/env python3
"""运行 M3 unittest 并可写 strict JSON 报告 / Run M3 tests and write a report."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
import time
import unittest
from pathlib import Path
from typing import Any

import numpy
import scipy
import sklearn


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
TEST_ROOT = PACKAGE_ROOT / "tests"
REPORT_PATH = PACKAGE_ROOT / "M3_REFERENCE_TEST_RESULTS.json"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))
sys.dont_write_bytecode = True


def sha256_file(path: Path) -> str:
    """计算测试源哈希 / Compute a test-source digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_input_snapshot() -> tuple[dict[str, str], str]:
    """冻结测试所见实现/合同/fixture / Freeze all implementation inputs seen."""

    patterns = (
        "src/**/*.py",
        "tests/test_*.py",
        "registries/*.json",
        "schemas/*.json",
        "fixtures/*",
    )
    paths = sorted(
        {
            path
            for pattern in patterns
            for path in PACKAGE_ROOT.glob(pattern)
            if path.is_file()
        },
        key=lambda path: path.relative_to(PACKAGE_ROOT).as_posix(),
    )
    digests = {
        path.relative_to(PACKAGE_ROOT).as_posix(): sha256_file(path) for path in paths
    }
    # 中文：对排序后的相对路径和文件哈希再次摘要，单一值即可检测陈旧报告。
    # English: Hash the sorted path/digest map so one value detects stale reports.
    canonical = json.dumps(
        digests, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return digests, hashlib.sha256(canonical).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    """只在 M3 包内原子写报告 / Atomically write a package-local report."""

    target = path.resolve(strict=False)
    target.relative_to(PACKAGE_ROOT.resolve())
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(target)


def run_suite() -> tuple[unittest.result.TestResult, str, float]:
    """发现并执行全部 reference tests / Discover and execute reference tests."""

    suite = unittest.defaultTestLoader.discover(
        str(TEST_ROOT),
        pattern="test_*.py",
        top_level_dir=str(PACKAGE_ROOT),
    )
    stream = io.StringIO()
    start = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    elapsed = time.perf_counter() - start
    return result, stream.getvalue(), elapsed


def main() -> int:
    """运行测试并按请求保存机器报告 / Run tests and optionally save JSON."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    result, transcript, elapsed = run_suite()
    test_files = sorted(TEST_ROOT.glob("test_*.py"))
    input_digests, input_snapshot_sha256 = build_input_snapshot()
    report = {
        "schema_version": "m3.reference_test_report.v2",
        "report_id": "m3_reference_tests_v1",
        "status": "pass" if result.wasSuccessful() else "fail",
        "tests_run": int(result.testsRun),
        "failure_count": len(result.failures),
        "error_count": len(result.errors),
        "skipped_count": len(result.skipped),
        "elapsed_sec": float(elapsed),
        "failures": [
            {"test": str(test), "traceback": traceback}
            for test, traceback in result.failures
        ],
        "errors": [
            {"test": str(test), "traceback": traceback}
            for test, traceback in result.errors
        ],
        "test_source_sha256": {
            path.name: sha256_file(path) for path in test_files
        },
        "input_file_sha256": input_digests,
        "input_snapshot_sha256": input_snapshot_sha256,
        "runtime": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "transcript": transcript,
    }
    if args.write_report:
        atomic_write_json(REPORT_PATH, report)
    print(transcript, end="")
    print(json.dumps({key: report[key] for key in (
        "status", "tests_run", "failure_count", "error_count", "skipped_count"
    )}, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
