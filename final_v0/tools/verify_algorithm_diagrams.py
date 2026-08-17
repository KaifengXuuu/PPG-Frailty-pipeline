#!/usr/bin/env python3
"""Verify Markdown/Mermaid diagram completeness and script coverage.

中文：对算法图执行确定性静态检查：标题存在、Mermaid fence 成对、图类型受支持、
图体非空，并确认 M0 逐脚本图册列出了所有预期入口。结果写入 ``records/generated``，
便于后续审计重跑和内容校验。

English: Perform deterministic static checks on the algorithm diagrams: headings,
balanced Mermaid fences, supported diagram types, non-empty bodies, and complete M0
script-atlas coverage. Write a machine-readable report below ``records/generated``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


# 中文：所有路径均锚定 final_v0；验证器不会写到项目其他位置。
# English: Anchor every path in final_v0; the verifier writes nowhere else.
FINAL_ROOT = Path(__file__).resolve().parents[1]
DIAGRAM_ROOT = FINAL_ROOT / "algorithm_diagrams"
REPORT_PATH = FINAL_ROOT / "records" / "generated" / "ALGORITHM_DIAGRAM_VERIFICATION.json"
ATLAS_PATH = DIAGRAM_ROOT / "m0" / "05_SCRIPT_ALGORITHM_ATLAS.md"

SUPPORTED_STARTS = (
    "flowchart ",
    "graph ",
    "sequenceDiagram",
    "stateDiagram",
    "classDiagram",
)

EXPECTED_M0_SCRIPTS = (
    "funcs.py",
    "ppg.py",
    "pttppg_pipeline_v7.py",
    "cnnppg_v7.py",
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
)


def digest(payload: bytes) -> str:
    """Return SHA-256 for evidence identity / 返回证据身份 SHA-256。"""

    return hashlib.sha256(payload).hexdigest()


def inspect_markdown(path: Path) -> dict[str, object]:
    """Inspect headings and Mermaid blocks in one file / 检查单文件标题和 Mermaid 块。"""

    payload = path.read_bytes()
    text = payload.decode("utf-8", errors="replace")
    lines = text.splitlines()
    failures: list[str] = []
    mermaid_blocks: list[list[str]] = []
    current: list[str] | None = None

    if not any(line.startswith("# ") for line in lines):
        failures.append("missing_level_1_heading")

    for line_no, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if current is None and stripped == "```mermaid":
            current = []
            continue
        if current is not None and stripped == "```":
            mermaid_blocks.append(current)
            current = None
            continue
        if current is not None:
            current.append(raw_line)
        elif stripped.startswith("```mermaid"):
            failures.append(f"invalid_mermaid_fence_at_line_{line_no}")

    if current is not None:
        failures.append("unclosed_mermaid_fence")
    if not mermaid_blocks:
        failures.append("no_mermaid_blocks")

    for index, block in enumerate(mermaid_blocks, start=1):
        nonempty = [line.strip() for line in block if line.strip()]
        if not nonempty:
            failures.append(f"empty_mermaid_block_{index}")
            continue
        if not nonempty[0].startswith(SUPPORTED_STARTS):
            failures.append(f"unsupported_mermaid_start_{index}:{nonempty[0]}")

    return {
        "path": path.relative_to(FINAL_ROOT).as_posix(),
        "bytes": len(payload),
        "sha256": digest(payload),
        "mermaid_block_count": len(mermaid_blocks),
        "failures": failures,
    }


def main() -> None:
    """Run all checks and write a strict JSON report / 执行全部检查并写严格 JSON。"""

    diagram_paths = [
        path
        for path in sorted(DIAGRAM_ROOT.rglob("*.md"))
        if path.name != "README.md"
    ]
    file_reports = [inspect_markdown(path) for path in diagram_paths]

    atlas_text = ATLAS_PATH.read_text(encoding="utf-8") if ATLAS_PATH.exists() else ""
    missing_scripts = [name for name in EXPECTED_M0_SCRIPTS if f"`{name}`" not in atlas_text]
    total_blocks = sum(int(item["mermaid_block_count"]) for item in file_reports)
    failures = [
        f"{item['path']}:{failure}"
        for item in file_reports
        for failure in item["failures"]
    ]
    failures.extend(f"atlas_missing_script:{name}" for name in missing_scripts)

    report = {
        "status": "pass" if not failures else "fail",
        "diagram_file_count": len(diagram_paths),
        "mermaid_block_count": total_blocks,
        "expected_m0_script_count": len(EXPECTED_M0_SCRIPTS),
        "missing_m0_scripts": missing_scripts,
        "failures": failures,
        "files": file_reports,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # 中文：使用严格 JSON；不允许 NaN，确保其他工具可安全读取。
    # English: Emit strict JSON with NaN disabled for safe downstream parsing.
    temp_path = REPORT_PATH.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(REPORT_PATH)

    if failures:
        raise SystemExit("Algorithm diagram verification failed: " + "; ".join(failures))
    print(
        f"PASS: {len(diagram_paths)} diagram files, {total_blocks} Mermaid blocks, "
        f"{len(EXPECTED_M0_SCRIPTS)} M0 scripts covered."
    )


if __name__ == "__main__":
    main()

