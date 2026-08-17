#!/usr/bin/env python3
"""Verify the final_v0 delivery boundary, documentation, and generated evidence.

中文：检查 ``final_v0`` 的Python语法与中英文说明、扫描/算法图验证状态、必需记录、
详细文件树覆盖和路径边界，并把严格JSON结果写入 ``records/generated``。本工具不读取
或修改 final_v0 之外的任何项目文件。

English: Check Python syntax and bilingual documentation, scan/diagram verification
status, required records, detailed-tree coverage, and path boundaries inside
``final_v0``. Write a strict JSON result below ``records/generated`` and do not read
or modify any project file outside final_v0.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path


FINAL_ROOT = Path(__file__).resolve().parents[1]
TREE_PATH = FINAL_ROOT / "FINAL_V0_TREE.md"
REPORT_PATH = FINAL_ROOT / "records" / "generated" / "FINAL_V0_VERIFICATION.json"
SCAN_REPORT = FINAL_ROOT / "records" / "generated" / "SCAN_VERIFICATION.json"
DIAGRAM_REPORT = FINAL_ROOT / "records" / "generated" / "ALGORITHM_DIAGRAM_VERIFICATION.json"

REQUIRED_PATHS = (
    "README.md",
    "FINAL_V0_TREE.md",
    "records/WORK_LOG.md",
    "records/SCAN_PROTOCOL.md",
    "records/PENDING_AGENT_UPDATES.md",
    "records/ROOT_FILE_IO_INVENTORY.md",
    "records/M0_EXECUTIVE_REPORT.md",
    "records/M0_METHOD_REGISTRY.md",
    "records/M0_CODE_OUTPUT_CROSSWALK.md",
    "records/M0_PAPER_EVIDENCE.md",
    "records/M0_RISK_REGISTER.md",
    "records/HUMAN_DECISION_GATES.md",
    "algorithm_diagrams/README.md",
    "algorithm_diagrams/00_PROJECT_HISTORICAL_SIGNAL_FLOW.md",
    "algorithm_diagrams/01_PROJECT_END_TO_END_PIPELINE.md",
)


def sha256_bytes(payload: bytes) -> str:
    """Return a SHA-256 evidence digest / 返回 SHA-256 证据摘要。"""

    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    """Load one strict JSON object / 读取一份严格 JSON 对象。"""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object in {path}")
    return value


def inspect_python(path: Path) -> dict[str, object]:
    """Check AST and bilingual module guidance / 检查 AST 和中英文模块说明。"""

    payload = path.read_bytes()
    text = payload.decode("utf-8", errors="replace")
    failures: list[str] = []
    try:
        tree = ast.parse(text, filename=str(path))
        doc = ast.get_docstring(tree) or ""
    except SyntaxError as exc:
        tree = None
        doc = ""
        failures.append(f"syntax_error_line_{exc.lineno}:{exc.msg}")

    # 中文：模块说明必须同时出现中文和英文，满足用户对新代码详细双语注释的要求。
    # English: Require both Chinese and English in module guidance for bilingual auditability.
    if not re.search(r"[\u4e00-\u9fff]", doc):
        failures.append("module_docstring_missing_chinese")
    if not re.search(r"[A-Za-z]{3,}", doc):
        failures.append("module_docstring_missing_english")

    comment_lines = [
        line.lstrip()[1:].strip()
        for line in text.splitlines()
        if line.lstrip().startswith("#") and not line.startswith("#!")
    ]
    if not any(re.search(r"[\u4e00-\u9fff]", line) for line in comment_lines):
        failures.append("inline_comments_missing_chinese")
    if not any(re.search(r"[A-Za-z]{3,}", line) for line in comment_lines):
        failures.append("inline_comments_missing_english")

    symbols = []
    if tree is not None:
        symbols = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
    return {
        "path": path.relative_to(FINAL_ROOT).as_posix(),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "top_level_symbol_count": len(symbols),
        "failures": failures,
    }


def tree_indexed_paths() -> set[str]:
    """Extract file rows from the detailed tree / 从详细文件树提取文件行。"""

    if not TREE_PATH.exists():
        return set()
    text = TREE_PATH.read_text(encoding="utf-8")
    return set(re.findall(r"^\| `([^`]+)` \|", text, flags=re.MULTILINE))


def main() -> None:
    """Run delivery checks and atomically write the report / 执行交付检查并原子写报告。"""

    failures: list[str] = []
    all_files = sorted(path for path in FINAL_ROOT.rglob("*") if path.is_file())

    # 中文：解析后的绝对路径必须仍在 final_v0 内，拒绝意外符号链接越界。
    # English: Resolved paths must remain inside final_v0, rejecting symlink escapes.
    boundary_failures: list[str] = []
    root_resolved = FINAL_ROOT.resolve()
    for path in all_files:
        try:
            path.resolve().relative_to(root_resolved)
        except ValueError:
            boundary_failures.append(path.as_posix())
    failures.extend(f"path_outside_final_v0:{path}" for path in boundary_failures)

    missing_required = [rel for rel in REQUIRED_PATHS if not (FINAL_ROOT / rel).is_file()]
    failures.extend(f"missing_required:{rel}" for rel in missing_required)

    python_reports = [inspect_python(path) for path in all_files if path.suffix == ".py"]
    failures.extend(
        f"{item['path']}:{failure}"
        for item in python_reports
        for failure in item["failures"]
    )

    indexed = tree_indexed_paths()
    # 中文：索引和本报告允许自引用例外，其余永久文件必须出现在详细表中。
    # English: Exempt the self-index and this report; every other file must be indexed.
    expected_indexed = {
        path.relative_to(FINAL_ROOT).as_posix()
        for path in all_files
        if path not in {TREE_PATH, REPORT_PATH}
    }
    missing_from_tree = sorted(expected_indexed - indexed)
    failures.extend(f"tree_missing:{rel}" for rel in missing_from_tree)

    try:
        scan = load_json(SCAN_REPORT)
        scan_status = scan.get("status")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        scan_status = "unreadable"
        failures.append(f"scan_report_unreadable:{type(exc).__name__}")
    if scan_status != "pass":
        failures.append(f"scan_status:{scan_status}")

    try:
        diagrams = load_json(DIAGRAM_REPORT)
        diagram_status = diagrams.get("status")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        diagram_status = "unreadable"
        failures.append(f"diagram_report_unreadable:{type(exc).__name__}")
    if diagram_status != "pass":
        failures.append(f"diagram_status:{diagram_status}")

    report = {
        "status": "pass" if not failures else "fail",
        "write_boundary": "final_v0_only",
        "permanent_file_count_seen": len(all_files),
        "python_file_count": len(python_reports),
        "python_files": python_reports,
        "required_path_count": len(REQUIRED_PATHS),
        "missing_required_paths": missing_required,
        "tree_missing_paths": missing_from_tree,
        "boundary_failures": boundary_failures,
        "scan_verification_status": scan_status,
        "algorithm_diagram_verification_status": diagram_status,
        "failures": failures,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # 中文：严格JSON禁用NaN；同目录临时文件保证报告完整写入。
    # English: Disable NaN and use a sibling temporary file for an atomic strict-JSON report.
    temp_path = REPORT_PATH.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(REPORT_PATH)

    if failures:
        raise SystemExit("final_v0 delivery verification failed: " + "; ".join(failures))
    print(
        f"PASS: {len(all_files)} files seen, {len(python_reports)} Python files checked, "
        "scan and diagram evidence valid."
    )


if __name__ == "__main__":
    main()

