#!/usr/bin/env python3
"""验证 workspace 扫描证据的完整性和内部一致性。

Verify completeness and internal consistency of workspace scan evidence.

该工具不重新读取原始数据，只核对 ``final_v0/records/generated`` 中已经生成的
manifest、summary 和运行账本，并将结果写入同一生成目录。

The tool does not re-read source data. It validates manifests, summaries, and the run ledger
already stored under ``final_v0/records/generated`` and writes its report there.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# 中文：校验器只读取final_v0生成证据；路径固定在本目录，避免触碰只读源项目。
# English: Validate generated final_v0 evidence only; anchored paths protect the read-only source project.
FINAL_ROOT = Path(__file__).resolve().parents[1]
GENERATED_ROOT = FINAL_ROOT / "records" / "generated"
REPORT_PATH = GENERATED_ROOT / "SCAN_VERIFICATION.json"

INPUT_NAMES = (
    "datasets",
    "PPG_Testing_05_01_2026",
    "physionet.org",
    "train_raw",
    "train_labeled",
    "train_val",
    "train_window",
)
OUTPUT_NAMES = (
    ".CNN_results",
    "denoiser_preview_output",
    "models",
    "results",
    "results_detector_v8",
    "results_denoiser_v8",
    "results_frailty3",
    "results_hybrid_denoiser",
    "results_hybrid_denoiser_raw_imu",
    "results_hybrid_denoiser_raw_imu_baseline",
    "results_stage1",
    "results_stage2",
    "results_v7_3",
    "results_v7_4",
    "results_v72_noleak",
    "results_v8_audit",
    "test_asa_classifier",
)


def read_json(path: Path) -> Any:
    """读取 JSON；read a JSON document."""

    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取非空 JSONL 行；read non-empty JSONL rows."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check(condition: bool, message: str, failures: list[str]) -> None:
    """累积失败而不中断后续核验；accumulate failures without stopping later checks."""

    if not condition:
        failures.append(message)


def verify_baseline(failures: list[str]) -> dict[str, Any]:
    """核验根文件、代码、路径引用和文件树基线；verify root, code, reference, and tree baselines."""

    summary = read_json(GENERATED_ROOT / "BASELINE_SUMMARY.json")
    roots = read_jsonl(GENERATED_ROOT / "ROOT_FILES.jsonl")
    code = read_jsonl(GENERATED_ROOT / "CODE_FILES.jsonl")
    refs = read_jsonl(GENERATED_ROOT / "CODE_PATH_REFERENCES.jsonl")
    files = read_jsonl(GENERATED_ROOT / "WORKSPACE_FILES.jsonl")

    check(len(roots) == summary["root_file_count"], "baseline root count mismatch", failures)
    check(len(code) == summary["code_full_read_count"], "baseline code count mismatch", failures)
    check(len(refs) == summary["path_reference_count"], "baseline path-reference count mismatch", failures)
    check(len(files) == summary["workspace_file_count"], "baseline workspace-file count mismatch", failures)
    check(sum(int(row.get("bytes") or 0) for row in files) == summary["workspace_byte_count"],
          "baseline workspace byte count mismatch", failures)
    check(all(row.get("sha256") for row in roots if row.get("read_mode") == "full_bytes"),
          "one or more fully read root files lack SHA-256", failures)
    check(all(row.get("sha256") and row.get("read_mode") == "full_bytes" for row in code),
          "one or more code files lack full-read SHA-256 evidence", failures)
    env_rows = [row for row in roots if row.get("path") == ".env"]
    check(bool(env_rows) and env_rows[0].get("structure", {}).get("values_redacted") is True,
          ".env redaction evidence missing", failures)
    check(summary.get("error_count") == 0, "baseline reports scan errors", failures)
    return {
        "root_files": len(roots),
        "fully_read_root_files": sum(row.get("read_mode") == "full_bytes" for row in roots),
        "fully_read_code_files": len(code),
        "path_references": len(refs),
        "workspace_files": len(files),
        "workspace_bytes": summary["workspace_byte_count"],
    }


def verify_inputs(failures: list[str]) -> dict[str, Any]:
    """核验所有输入头部清单；verify every input-head manifest."""

    totals = {"directories": 0, "files": 0, "bytes": 0, "errors": 0}
    details: dict[str, Any] = {}
    for name in INPUT_NAMES:
        manifest = read_jsonl(GENERATED_ROOT / "inputs" / f"{name}.jsonl")
        summary = read_json(GENERATED_ROOT / "inputs" / f"{name}.summary.json")
        errors = sum(bool(row.get("error")) for row in manifest)
        byte_count = sum(int(row.get("bytes") or 0) for row in manifest)
        check(len(manifest) == summary["file_count"], f"{name}: input file count mismatch", failures)
        check(byte_count == summary["byte_count"], f"{name}: input byte count mismatch", failures)
        check(errors == summary["error_count"], f"{name}: input error count mismatch", failures)
        for row in manifest:
            if row.get("read_mode") == "bounded_head":
                check(bool(row.get("head_sha256")), f"{row.get('path')}: missing head SHA-256", failures)
                check(0 <= row.get("head_bytes_read", -1) <= row.get("head_bytes_requested", -2),
                      f"{row.get('path')}: invalid head byte count", failures)
        totals["directories"] += 1
        totals["files"] += len(manifest)
        totals["bytes"] += byte_count
        totals["errors"] += errors
        details[name] = {"files": len(manifest), "bytes": byte_count, "errors": errors}
    totals["details"] = details
    return totals


def verify_outputs(failures: list[str]) -> dict[str, Any]:
    """核验输出文本 EOF 证据和二进制登记；verify output-text EOF evidence and binary registration."""

    totals = {
        "directories": 0, "files": 0, "bytes": 0, "text_full_read": 0,
        "binary_metadata_only": 0, "errors": 0,
    }
    details: dict[str, Any] = {}
    for name in OUTPUT_NAMES:
        manifest = read_jsonl(GENERATED_ROOT / "outputs" / f"{name}.jsonl")
        summary = read_json(GENERATED_ROOT / "outputs" / f"{name}.summary.json")
        errors = sum(bool(row.get("error")) for row in manifest)
        byte_count = sum(int(row.get("bytes") or 0) for row in manifest)
        text_rows = [row for row in manifest if row.get("read_mode") == "full_bytes"]
        binary_rows = [row for row in manifest if row.get("read_mode") == "filename_metadata_only"]
        check(len(manifest) == summary["file_count"], f"{name}: output file count mismatch", failures)
        check(byte_count == summary["byte_count"], f"{name}: output byte count mismatch", failures)
        check(len(text_rows) == summary["text_full_read_count"], f"{name}: text count mismatch", failures)
        check(len(binary_rows) == summary["binary_metadata_only_count"], f"{name}: binary count mismatch", failures)
        check(errors == summary["error_count"], f"{name}: output error count mismatch", failures)
        for row in text_rows:
            check(row.get("bytes_read") == row.get("bytes"),
                  f"{row.get('path')}: full text did not reach declared EOF", failures)
            check(bool(row.get("sha256")), f"{row.get('path')}: missing full-text SHA-256", failures)
        totals["directories"] += 1
        totals["files"] += len(manifest)
        totals["bytes"] += byte_count
        totals["text_full_read"] += len(text_rows)
        totals["binary_metadata_only"] += len(binary_rows)
        totals["errors"] += errors
        details[name] = {
            "files": len(manifest), "bytes": byte_count, "text_full_read": len(text_rows),
            "binary_metadata_only": len(binary_rows), "errors": errors,
        }
    totals["details"] = details
    return totals


def verify_ledger(failures: list[str]) -> dict[str, Any]:
    """核验运行账本覆盖所有预期事务；verify ledger coverage for every expected transaction."""

    rows = read_jsonl(GENERATED_ROOT / "SCAN_RUNS.jsonl")
    keys = {(row.get("section"), row.get("target")) for row in rows}
    expected = {("baseline", "workspace")}
    expected.update(("input", name) for name in INPUT_NAMES)
    expected.update(("output", name) for name in OUTPUT_NAMES)
    missing = sorted(expected - keys)
    check(not missing, f"scan ledger missing transactions: {missing}", failures)
    check(all(row.get("error_count") == 0 for row in rows), "scan ledger contains errors", failures)
    return {"rows": len(rows), "expected_transactions": len(expected), "missing": missing}


def main() -> int:
    """运行全部一致性检查并写报告；run all consistency checks and write the report."""

    failures: list[str] = []
    report = {
        "baseline": verify_baseline(failures),
        "inputs": verify_inputs(failures),
        "outputs": verify_outputs(failures),
        "ledger": verify_ledger(failures),
    }
    report["failures"] = failures
    report["status"] = "pass" if not failures else "fail"
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"status": report["status"], "failure_count": len(failures)}, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

