#!/usr/bin/env python3
"""Verify that every scanned code/Notebook path has a per-script diagram entry.

中文：读取 ``CODE_FILES.jsonl`` 的52个真实路径，并检查它们分别出现在M0根图册、
非M0根图册或归档图册的反引号标题中。工具只读取final_v0证据并写严格JSON报告。

English: Read the 52 real code/Notebook paths from ``CODE_FILES.jsonl`` and require
each path in the appropriate M0-root, non-M0-root, or archived per-script atlas.
The tool reads final_v0 evidence only and writes a strict JSON report.
"""

from __future__ import annotations

import json
from pathlib import Path


FINAL_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = FINAL_ROOT / "records" / "generated" / "CODE_FILES.jsonl"
REPORT = FINAL_ROOT / "records" / "generated" / "CODE_DIAGRAM_COVERAGE.json"
M0_ATLAS = FINAL_ROOT / "algorithm_diagrams" / "m0" / "05_SCRIPT_ALGORITHM_ATLAS.md"
ROOT_ATLAS = FINAL_ROOT / "algorithm_diagrams" / "baseline" / "01_NON_M0_ROOT_SCRIPT_ATLAS.md"
ARCHIVE_ATLAS = FINAL_ROOT / "algorithm_diagrams" / "baseline" / "02_ARCHIVED_SCRIPT_ATLAS.md"

M0_ROOT_NAMES = {
    "funcs.py",
    "ppg.py",
    "cnnppg_v7.py",
    "pttppg_pipeline_v7.py",
    "pttppg_pipeline_v7_4_noleak_viz_ae.py",
    "pttppg_denoiser_v8_masknet.py",
    "pttppg_stage2_denoiser.py",
    "pttppg_detector_v8_scores_audit_fix9.py",
    "pttppg_denoiser_hybrid_core.py",
    "pttppg_denoiser_hybrid_train.py",
    "pttppg_denoiser_hybrid_preview.py",
    "pttppg_denoiser_hybrid_ab_compare.py",
    "pttppg_denoiser_hybrid_export_onnx.py",
    "pttppg_denoiser_onnx_runtime.py",
    "ppg_denoiser_dash_utils.py",
    "ppg_peak_hr_gating_train.py",
}


def read_manifest_paths() -> list[str]:
    """Load exact code paths from JSONL / 从JSONL读取精确代码路径。"""

    paths: list[str] = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            paths.append(str(json.loads(line)["path"]))
    return paths


def main() -> None:
    """Check all groups and atomically write strict JSON / 检查全分组并原子写严格JSON。"""

    paths = read_manifest_paths()
    atlas_texts = {
        "m0_root": M0_ATLAS.read_text(encoding="utf-8"),
        "non_m0_root": ROOT_ATLAS.read_text(encoding="utf-8"),
        "archived": ARCHIVE_ATLAS.read_text(encoding="utf-8"),
    }
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    group_counts = {"m0_root": 0, "non_m0_root": 0, "archived": 0}

    for path in paths:
        if "/" in path:
            group = "archived"
            label = path
        elif path in M0_ROOT_NAMES:
            group = "m0_root"
            label = path
        else:
            group = "non_m0_root"
            label = path
        group_counts[group] += 1
        found = f"`{label}`" in atlas_texts[group]
        if not found:
            failures.append(f"missing_diagram_entry:{group}:{path}")
        rows.append({"path": path, "group": group, "diagram_entry_found": found})

    if len(paths) != len(set(paths)):
        failures.append("duplicate_paths_in_code_manifest")
    if len(paths) != 52:
        failures.append(f"unexpected_code_manifest_count:{len(paths)}")
    expected_counts = {"m0_root": 16, "non_m0_root": 13, "archived": 23}
    if group_counts != expected_counts:
        failures.append(f"unexpected_group_counts:{group_counts}")

    report_data = {
        "status": "pass" if not failures else "fail",
        "code_manifest_count": len(paths),
        "group_counts": group_counts,
        "expected_group_counts": expected_counts,
        "covered_count": sum(bool(row["diagram_entry_found"]) for row in rows),
        "missing_count": sum(not bool(row["diagram_entry_found"]) for row in rows),
        "failures": failures,
        "files": rows,
    }

    # 中文：禁用NaN并使用同目录临时文件，保证报告严格且完整。
    # English: Disable NaN and use a sibling temporary file for a complete strict report.
    temp_path = REPORT.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(REPORT)
    if failures:
        raise SystemExit("Code-diagram coverage failed: " + "; ".join(failures))
    print(f"PASS: {len(paths)}/{len(paths)} code and Notebook paths have diagram entries.")


if __name__ == "__main__":
    main()

