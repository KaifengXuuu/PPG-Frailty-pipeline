#!/usr/bin/env python3
"""隔离执行历史 peak 函数并生成 parity 证据 / Isolated legacy peak parity.

中文：本工具只读取根目录历史脚本，以 AST 提取所需函数，避免导入 Dash、Torch
或 HRV 可视化依赖以及模块级副作用。它不把历史代码变成 future-active 入口。

English: This audit reads root historical scripts and executes only selected AST
functions, avoiding UI/ML imports and module-level side effects. It never promotes
legacy code into a future-active runtime.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
FIXTURE_PATH = PACKAGE_ROOT / "fixtures" / "ppg_reference_v1.npy"
REPORT_PATH = PACKAGE_ROOT / "evidence" / "legacy_peak_parity.json"
BUILD_REPORT_PATH = PACKAGE_ROOT / "M3_BUILD_REPORT.json"
sys.dont_write_bytecode = True

LEGACY_COMMON_NAMES = {
    "highpass_filter",
    "bandpass_filter",
    "robust_std",
    "estimate_hr",
    "reject_artifacts",
    "calculate_hrv",
    "window_indices",
    "_detect_maxima_adaptive",
    "aboypp_peak_hr",
}
CLASSIFIER_NAMES = {
    "interp_nan",
    "iter_windows",
    "clean_pp_intervals",
    "_highpass_ppg",
    "_aboy_bandpass",
    "_detect_maxima_adaptive",
    "_score_peak_train",
    "aboypp_detect_peaks",
    "detect_ppg_peaks",
}


def sha256_file(path: Path) -> str:
    """逐字节文件摘要 / Hash a file byte-for-byte."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_int64(values: np.ndarray) -> str:
    """以固定 little/native int64 序列冻结 peak 结果 / Hash an int64 peak train."""

    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def _load_selected_functions(path: Path, names: set[str]) -> dict[str, Any]:
    """只编译白名单函数节点 / Compile only allow-listed function AST nodes."""

    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    found = {node.name for node in selected}
    if found != names:
        raise RuntimeError(f"legacy_function_set_mismatch:{path.name}:{sorted(names-found)}")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    # 中文：类型名也必须提供，因为旧脚本没有 postponed-annotation 保证。
    # English: Supply typing names because legacy annotations may evaluate eagerly.
    namespace: dict[str, Any] = {
        "np": np,
        "signal": signal,
        "FS": 400,
        "MIN_BPM": 40,
        "MAX_BPM": 180,
        "Any": Any,
        "Dict": Dict,
        "Iterable": Iterable,
        "List": List,
        "Optional": Optional,
        "Sequence": Sequence,
        "Tuple": Tuple,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def _arrays_equal(left: Any, right: Any) -> bool:
    """比较历史返回数组并允许对应 NaN / Compare arrays with aligned NaN."""

    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape:
        return False
    return bool(np.allclose(a, b, rtol=0.0, atol=0.0, equal_nan=True))


def run_legacy_peak_parity() -> dict[str, Any]:
    """运行固定同输入审计 / Run the frozen same-input audit."""

    sources = {
        "funcs": REPOSITORY_ROOT / "funcs.py",
        "ppg": REPOSITORY_ROOT / "ppg.py",
        "classifier": REPOSITORY_ROOT / "frailty_3class_classifier.py",
    }
    if not FIXTURE_PATH.is_file() or not all(path.is_file() for path in sources.values()):
        raise FileNotFoundError("legacy_peak_parity_input_missing")
    fixture = np.load(FIXTURE_PATH, allow_pickle=False)
    funcs = _load_selected_functions(sources["funcs"], LEGACY_COMMON_NAMES)
    ppg = _load_selected_functions(sources["ppg"], LEGACY_COMMON_NAMES)
    classifier = _load_selected_functions(sources["classifier"], CLASSIFIER_NAMES)

    funcs_result = funcs["aboypp_peak_hr"](fixture, fs=400)
    ppg_result = ppg["aboypp_peak_hr"](fixture, fs=400)
    classifier_peaks = np.asarray(
        classifier["aboypp_detect_peaks"](fixture, 400.0), dtype=np.int64
    )
    alias_peaks = np.asarray(
        classifier["detect_ppg_peaks"](fixture, 400.0), dtype=np.int64
    )
    funcs_peaks = np.asarray(funcs_result["peaks_all"], dtype=np.int64)
    ppg_peaks = np.asarray(ppg_result["peaks_all"], dtype=np.int64)

    duplicate_keys = ("peaks_all", "hr_series", "HRi_series", "rr")
    duplicate_exact = all(
        _arrays_equal(funcs_result[key], ppg_result[key]) for key in duplicate_keys
    ) and all(
        bool(np.isclose(funcs_result[key], ppg_result[key], equal_nan=True))
        for key in ("hr_global", "hrv_ms")
    )
    alias_exact = bool(np.array_equal(classifier_peaks, alias_peaks))
    cross_exact = bool(np.array_equal(funcs_peaks, classifier_peaks))
    common = np.intersect1d(funcs_peaks, classifier_peaks)
    return {
        "evidence_id": "m3_legacy_peak_same_input_parity_v1",
        "status": (
            "pass_with_expected_cross_implementation_difference"
            if duplicate_exact and alias_exact and not cross_exact
            else "fail"
        ),
        "fixture": str(FIXTURE_PATH.relative_to(REPOSITORY_ROOT)),
        "fixture_sha256": sha256_file(FIXTURE_PATH),
        "sampling_rate_hz": 400,
        "source_sha256": {
            name: sha256_file(path) for name, path in sources.items()
        },
        "funcs_ppg_duplicate_parity": {
            "exact": duplicate_exact,
            "peak_count": int(funcs_peaks.size),
            "peak_indices_sha256": sha256_int64(funcs_peaks),
        },
        "classifier_alias_parity": {
            "exact": alias_exact,
            "peak_count": int(classifier_peaks.size),
            "peak_indices_sha256": sha256_int64(classifier_peaks),
        },
        "cross_implementation_comparison": {
            "exact": cross_exact,
            "matched_exact_index_count": int(common.size),
            "funcs_ppg_only_indices": np.setdiff1d(funcs_peaks, classifier_peaks).tolist(),
            "classifier_only_indices": np.setdiff1d(classifier_peaks, funcs_peaks).tolist(),
            "interpretation": (
                "The classifier adaptation is not the same algorithm as funcs.py/ppg.py; "
                "the difference is preserved as historical evidence, not corrected-v1 parity."
            ),
        },
        "authority_boundary": (
            "audit_only_ast_execution; all future-active callers use m3_signal_core"
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """仅向 M3 evidence 原子写 strict JSON / Atomically write M3 evidence only."""

    path.resolve(strict=False).relative_to(PACKAGE_ROOT.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def update_build_report() -> None:
    """把 parity 证据登记到总构建报告 / Register parity evidence in build report."""

    report = json.loads(BUILD_REPORT_PATH.read_text(encoding="utf-8"))
    entry = {
        "path": "evidence/legacy_peak_parity.json",
        "sha256": sha256_file(REPORT_PATH),
        "bytes": REPORT_PATH.stat().st_size,
    }
    report["outputs"] = [
        item for item in report["outputs"] if item["path"] != entry["path"]
    ] + [entry]
    report["outputs"] = sorted(report["outputs"], key=lambda item: item["path"])
    report["report_id"] = "m3_complete_evidence_build_v1"
    write_json(BUILD_REPORT_PATH, report)


def main() -> int:
    """执行审计并按需保存 / Run the audit and optionally save evidence."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = run_legacy_peak_parity()
    if args.write_report:
        write_json(REPORT_PATH, report)
        update_build_report()
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if report["status"].startswith("pass") else 1


if __name__ == "__main__":
    # 中文：命令行只读默认；显式 --write-report 才落证据。
    # English: CLI is read-only unless --write-report is explicitly provided.
    raise SystemExit(main())
