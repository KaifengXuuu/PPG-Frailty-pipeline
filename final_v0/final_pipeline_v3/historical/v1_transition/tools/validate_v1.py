#!/usr/bin/env python3
"""V1 结构、合同和自审验证器 / V1 structural, contract, and review validator."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


@dataclass(frozen=True)
class Check:
    """一个确定性检查 / One deterministic validation check."""

    check_id: str
    run: Callable[[], str]


def _sha(path: Path) -> str:
    """流式计算 hash / Hash a file as a byte stream."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_paths() -> str:
    """核对规范要求的顶层交付 / Check specification-level deliverables."""

    paths = [
        "README.md",
        "MIGRATION.md",
        "pyproject.toml",
        "configs/reference_static_v1.yaml",
        "configs/reference_all_roles_v1.yaml",
        "configs/motion_benchmark_v1.yaml",
        "configs/feature_matrix_v1.yaml",
        "artifacts/audit/baseline_inventory.json",
        "artifacts/audit/legacy_characterization.json",
        "src/ppg_frailty/config.py",
        "src/ppg_frailty/contracts.py",
        "src/ppg_frailty/provenance.py",
    ]
    missing = [relative for relative in paths if not (ROOT / relative).is_file()]
    if missing:
        raise AssertionError(f"missing required paths: {missing}")
    return f"required_paths={len(paths)}"


def _spec_lock() -> str:
    """复算附件 hash / Recompute the attached specification hash."""

    lock = json.loads((ROOT / "docs/spec/SPEC_LOCK.json").read_text(encoding="utf-8"))
    source = REPO / lock["source_path"]
    observed = _sha(source)
    # 中文：SPEC_LOCK 的权威字段为 source_sha256；English: use the locked schema field.
    if observed != lock["source_sha256"]:
        raise AssertionError(f"spec hash mismatch: {observed}")
    return f"spec_sha256={observed}"


def _python_ast_and_bilingual() -> str:
    """核对 AST 与中英文说明 / Check syntax and bilingual documentation."""

    failures: list[str] = []
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "tools").glob("*.py"))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        try:
            ast.parse(text, filename=str(path))
        except SyntaxError as error:
            failures.append(f"{path.relative_to(ROOT)}: syntax={error}")
            continue
        if re.search(r"[\u4e00-\u9fff]", text) is None:
            failures.append(f"{path.relative_to(ROOT)}: missing Chinese documentation")
        if re.search(r"\b(the|and|return|validate|English|strict)\b", text, re.I) is None:
            failures.append(f"{path.relative_to(ROOT)}: missing English documentation")
    if failures:
        raise AssertionError("; ".join(failures))
    return f"python_files={len(paths)}"


def _no_legacy_runtime_imports() -> str:
    """阻止活动包导入根历史脚本 / Reject runtime imports of historical scripts."""

    forbidden = {
        "funcs",
        "ppg",
        "frailty_3class_classifier",
        "frailty_3class_overfitting_sweep",
        "frailty_3class_holdout_eval",
        "frailty_3class_cnn_fusion",
        "shapeformer_port",
        "pttppg_denoiser_hybrid_core",
    }
    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            overlap = forbidden.intersection(names)
            if overlap:
                violations.append(f"{path.relative_to(ROOT)}:{sorted(overlap)}")
    if violations:
        raise AssertionError("legacy imports: " + ", ".join(violations))
    return "legacy_runtime_imports=0"


def _strict_json() -> str:
    """解析所有机器 JSON 并拒绝 NaN / Parse machine JSON and reject NaN."""

    count = 0
    for path in sorted(ROOT.rglob("*.json")):
        def reject(token: str) -> None:
            raise ValueError(f"non-finite JSON token {token}")

        json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
        count += 1
    return f"strict_json_files={count}"


def _configs() -> str:
    """通过唯一加载器验证配置 / Validate configs through the sole loader."""

    from ppg_frailty.config import load_config

    paths = sorted((ROOT / "configs").glob("*.yaml"))
    configs = [load_config(path) for path in paths]
    identifiers = [config.config_id for config in configs]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("duplicate config_id")
    return f"configs={len(configs)}"


def _write(path: Path, payload: dict[str, object]) -> None:
    """原子写验证报告 / Atomically write the validation report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    """执行全部检查 / Execute every registered check."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", type=Path)
    arguments = parser.parse_args(argv)
    checks = [
        Check("required_paths", _required_paths),
        Check("spec_lock", _spec_lock),
        Check("python_ast_and_bilingual", _python_ast_and_bilingual),
        Check("no_legacy_runtime_imports", _no_legacy_runtime_imports),
        Check("strict_json", _strict_json),
        Check("configs", _configs),
    ]
    rows: list[dict[str, str]] = []
    for check in checks:
        try:
            detail = check.run()
            rows.append({"check_id": check.check_id, "status": "passed", "detail": detail})
        except Exception as error:  # 每项都记录，避免首错隐藏后续缺陷 / Record all failures.
            rows.append({"check_id": check.check_id, "status": "failed", "detail": repr(error)})
    status = "passed" if all(row["status"] == "passed" for row in rows) else "failed"
    payload: dict[str, object] = {
        "schema_version": "ppg_frailty.v1_validation.v1",
        "status": status,
        "checks_run": len(rows),
        "checks": rows,
    }
    if arguments.write_report is not None:
        _write(arguments.write_report, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
