#!/usr/bin/env python3
"""构建并验证 M0 历史归档包；build and verify the M0 history package.

中文
----
该工具只读取 ``final_v0`` 内已经存在的 M0 报告、算法图和机器摘要，并只写
``final_v0/M0_history_MA_denoising_detector_HR_feature``。它按字节生成不可变快照、
源—快照 SHA-256 清单、包级验证 JSON，以及本包全部永久文件的详细树状索引。

默认模式不会覆盖内容不同的既有快照；只有显式 ``--refresh`` 才允许刷新。因此后续
TODO 不会静默改写已归档的 M0 历史。``--verify-only`` 完全只读，用于复核快照内容、
源文件漂移和必需文档。

English
-------
This tool reads existing M0 reports, diagrams, and machine summaries inside ``final_v0``
and writes only the dedicated M0 history package. It creates byte-for-byte snapshots,
a source-to-snapshot SHA-256 manifest, a package verification JSON, and a detailed tree
for every permanent package file.

The default mode refuses to overwrite a differing snapshot; ``--refresh`` is required
for an intentional refresh. ``--verify-only`` is read-only and checks snapshot integrity,
source drift, and required documents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


# 中文：所有根路径均由脚本位置解析，禁止依赖调用者当前目录。
# English: Resolve every root from this script; never depend on the caller's cwd.
FINAL_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = FINAL_ROOT / "M0_history_MA_denoising_detector_HR_feature"
# 中文：v2 使用追加式文件名，保留既有 v1 清单、验证和文件树不被覆盖。
# English: Use additive v2 filenames so the existing v1 manifest, verification, and tree remain immutable.
MANIFEST_PATH = PACKAGE_ROOT / "M0_SOURCE_SNAPSHOT_MANIFEST_V2.json"
VERIFICATION_PATH = PACKAGE_ROOT / "M0_PACKAGE_VERIFICATION_V2.json"
TREE_PATH = PACKAGE_ROOT / "08_M0_PACKAGE_TREE_V2.md"
PACKAGE_ID = "M0-20260803-local-audit-v2"


# 中文：这些人工可读报告构成 M0 结论的完整历史快照。
# English: These human-readable reports form the complete historical M0 conclusion set.
RECORD_SNAPSHOTS: tuple[tuple[str, str], ...] = (
    ("records/M0_EXECUTIVE_REPORT.md", "snapshots/records/M0_EXECUTIVE_REPORT.md"),
    ("records/M0_METHOD_REGISTRY.md", "snapshots/records/M0_METHOD_REGISTRY.md"),
    ("records/M0_CODE_OUTPUT_CROSSWALK.md", "snapshots/records/M0_CODE_OUTPUT_CROSSWALK.md"),
    ("records/M0_PAPER_EVIDENCE.md", "snapshots/records/M0_PAPER_EVIDENCE.md"),
    ("records/M0_RISK_REGISTER.md", "snapshots/records/M0_RISK_REGISTER.md"),
    ("records/M0_ARCHIVED_LINEAGE_EVIDENCE.md", "snapshots/records/M0_ARCHIVED_LINEAGE_EVIDENCE.md"),
    ("records/PROJECT_WIDE_SCAN_FINDINGS.md", "snapshots/records/PROJECT_WIDE_SCAN_FINDINGS.md"),
    ("records/ROOT_FILE_IO_INVENTORY.md", "snapshots/records/ROOT_FILE_IO_INVENTORY.md"),
    ("records/ARCHIVED_CODE_IO_INVENTORY.md", "snapshots/records/ARCHIVED_CODE_IO_INVENTORY.md"),
    ("records/CODE_IO_MASTER_INDEX.md", "snapshots/records/CODE_IO_MASTER_INDEX.md"),
    ("records/HUMAN_DECISION_GATES.md", "snapshots/records/HUMAN_DECISION_GATES.md"),
    ("records/SCAN_PROTOCOL.md", "snapshots/records/SCAN_PROTOCOL.md"),
    # 中文：保存用户确认的 MAdenoiser 后续路线决定，避免把计划误作历史结果。
    # English: Snapshot the confirmed MAdenoiser route decision without treating it as a result.
    ("records/decisions/20260803_m0_madenoiser_route.md",
     "snapshots/records/decisions/20260803_m0_madenoiser_route.md"),
)


# 中文：集中快照 M0 主流程图和逐路线算法图，包括本轮新增的五族统一图。
# English: Snapshot the M0 overview and route diagrams, including the new five-family map.
DIAGRAM_SNAPSHOTS: tuple[tuple[str, str], ...] = (
    (
        "algorithm_diagrams/00_PROJECT_HISTORICAL_SIGNAL_FLOW.md",
        "snapshots/algorithm_diagrams/00_PROJECT_HISTORICAL_SIGNAL_FLOW.md",
    ),
    (
        "algorithm_diagrams/m0/01_FOUNDATION_FUNCS_PPG.md",
        "snapshots/algorithm_diagrams/01_FOUNDATION_FUNCS_PPG.md",
    ),
    (
        "algorithm_diagrams/m0/02_V7_TO_STAGE2_EVOLUTION.md",
        "snapshots/algorithm_diagrams/02_V7_TO_STAGE2_EVOLUTION.md",
    ),
    (
        "algorithm_diagrams/m0/03_HYBRID_SUITE.md",
        "snapshots/algorithm_diagrams/03_HYBRID_SUITE.md",
    ),
    (
        "algorithm_diagrams/m0/04_HEARTBEAT_AND_MOTION_AB.md",
        "snapshots/algorithm_diagrams/04_HEARTBEAT_AND_MOTION_AB.md",
    ),
    (
        "algorithm_diagrams/m0/05_SCRIPT_ALGORITHM_ATLAS.md",
        "snapshots/algorithm_diagrams/05_SCRIPT_ALGORITHM_ATLAS.md",
    ),
    (
        "algorithm_diagrams/m0/06_FIVE_METHOD_FAMILIES_IMPLEMENTATION_AND_BENCHMARK.md",
        "snapshots/algorithm_diagrams/06_FIVE_METHOD_FAMILIES_IMPLEMENTATION_AND_BENCHMARK.md",
    ),
    # 中文：保存用户确认路线到 frailty 特征选择的完整依赖图。
    # English: Snapshot the confirmed route-to-frailty feature-selection dependency map.
    (
        "algorithm_diagrams/m0/07_MADENOISER_ROUTE_TO_FRAILTY_FEATURE_SELECTION.md",
        "snapshots/algorithm_diagrams/07_MADENOISER_ROUTE_TO_FRAILTY_FEATURE_SELECTION.md",
    ),
)


# 中文：只复制小型验证、源码清单和相关输入/输出 summary。完整大 manifest 继续以
# ``records/generated`` 为单一事实源，并由 05_EVIDENCE_INDEX_AND_PROVENANCE.md 指向。
# English: Copy selected compact verification files, code inventories, and relevant
# input/output summaries. Large manifests remain canonical under ``records/generated``.
VERIFICATION_SNAPSHOTS: tuple[tuple[str, str], ...] = (
    ("records/generated/BASELINE_SUMMARY.json", "snapshots/verification/BASELINE_SUMMARY.json"),
    ("records/generated/TOP_LEVEL_DIRECTORIES.json", "snapshots/verification/TOP_LEVEL_DIRECTORIES.json"),
    ("records/generated/SCAN_RUNS.jsonl", "snapshots/verification/SCAN_RUNS.jsonl"),
    ("records/generated/SCAN_VERIFICATION.json", "snapshots/verification/SCAN_VERIFICATION.json"),
    (
        "records/generated/ALGORITHM_DIAGRAM_VERIFICATION.json",
        "snapshots/verification/ALGORITHM_DIAGRAM_VERIFICATION_ROUTE_V2.json",
    ),
    ("records/generated/CODE_DIAGRAM_COVERAGE.json", "snapshots/verification/CODE_DIAGRAM_COVERAGE.json"),
    ("records/generated/ROOT_FILES.jsonl", "snapshots/verification/ROOT_FILES.jsonl"),
    ("records/generated/CODE_FILES.jsonl", "snapshots/verification/CODE_FILES.jsonl"),
    (
        "records/generated/CODE_PATH_REFERENCES.jsonl",
        "snapshots/verification/CODE_PATH_REFERENCES.jsonl",
    ),
    (
        "records/generated/inputs/physionet.org.summary.json",
        "snapshots/verification/inputs/physionet.org.summary.json",
    ),
    (
        "records/generated/inputs/PPG_Testing_05_01_2026.summary.json",
        "snapshots/verification/inputs/PPG_Testing_05_01_2026.summary.json",
    ),
    ("records/generated/outputs/results.summary.json", "snapshots/verification/outputs/results.summary.json"),
    (
        "records/generated/outputs/results_v72_noleak.summary.json",
        "snapshots/verification/outputs/results_v72_noleak.summary.json",
    ),
    (
        "records/generated/outputs/results_v7_4.summary.json",
        "snapshots/verification/outputs/results_v7_4.summary.json",
    ),
    (
        "records/generated/outputs/results_denoiser_v8.summary.json",
        "snapshots/verification/outputs/results_denoiser_v8.summary.json",
    ),
    (
        "records/generated/outputs/results_stage2.summary.json",
        "snapshots/verification/outputs/results_stage2.summary.json",
    ),
    (
        "records/generated/outputs/results_stage1.summary.json",
        "snapshots/verification/outputs/results_stage1.summary.json",
    ),
    (
        "records/generated/outputs/results_v8_audit.summary.json",
        "snapshots/verification/outputs/results_v8_audit.summary.json",
    ),
    (
        "records/generated/outputs/results_hybrid_denoiser_raw_imu.summary.json",
        "snapshots/verification/outputs/results_hybrid_denoiser_raw_imu.summary.json",
    ),
    (
        "records/generated/outputs/results_hybrid_denoiser_raw_imu_baseline.summary.json",
        "snapshots/verification/outputs/results_hybrid_denoiser_raw_imu_baseline.summary.json",
    ),
    (
        "records/generated/outputs/results_hybrid_denoiser.summary.json",
        "snapshots/verification/outputs/results_hybrid_denoiser.summary.json",
    ),
    (
        "records/generated/outputs/denoiser_preview_output.summary.json",
        "snapshots/verification/outputs/denoiser_preview_output.summary.json",
    ),
    (
        "records/generated/outputs/.CNN_results.summary.json",
        "snapshots/verification/outputs/CNN_RESULTS.summary.json",
    ),
    (
        "records/generated/outputs/results_frailty3.summary.json",
        "snapshots/verification/outputs/results_frailty3.summary.json",
    ),
)

SNAPSHOT_SPECS = RECORD_SNAPSHOTS + DIAGRAM_SNAPSHOTS + VERIFICATION_SNAPSHOTS


# 中文：这些是用户要求的人工可读核心文件；包验证必须逐一确认。
# English: These are the required human-readable package documents.
REQUIRED_PACKAGE_DOCS: tuple[str, ...] = (
    "README.md",
    "01_M0_COMPLETE_RESULTS_AND_DECISIONS.md",
    "02_CANDIDATE_SCRIPTS_MOTION_DENOISING_DYNAMIC_HR.md",
    "03_FIVE_METHOD_FAMILIES_CODE_TEST_IMPLEMENTATION_AUDIT.md",
    "04_UNIFIED_IMPLEMENTATION_AND_BENCHMARK_CONTRACT.md",
    "05_EVIDENCE_INDEX_AND_PROVENANCE.md",
    # 中文：用户确认路线必须作为包级必需文档接受验证。
    # English: Validate the user-confirmed route as a required package document.
    "07_CONFIRMED_MADENOISER_FOLLOWUP_ROADMAP.md",
)


def sha256_bytes(payload: bytes) -> str:
    """返回稳定 SHA-256；return a stable SHA-256 digest."""

    return hashlib.sha256(payload).hexdigest()


def relative_to_checked(path: Path, root: Path) -> str:
    """确认路径位于根目录内并返回POSIX相对路径；validate containment and return a POSIX path."""

    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"Path escapes allowed root: {resolved_path} not under {resolved_root}") from exc


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    """同目录临时文件后原子替换；atomically replace a file through a same-directory temporary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    """以UTF-8/LF原子写文本；atomically write UTF-8 text with LF newlines."""

    write_bytes_atomic(path, text.replace("\r\n", "\n").encode("utf-8"))


def write_json_atomic(path: Path, data: Any) -> None:
    """写严格、可读、稳定排序JSON；write strict, readable, deterministically sorted JSON."""

    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    write_text_atomic(path, payload)


def snapshot_one(source_rel: str, snapshot_rel: str, refresh: bool) -> dict[str, Any]:
    """复制或验证一个快照；copy or verify one snapshot according to the refresh policy."""

    source = FINAL_ROOT / source_rel
    snapshot = PACKAGE_ROOT / snapshot_rel
    relative_to_checked(source, FINAL_ROOT)
    relative_to_checked(snapshot, PACKAGE_ROOT)
    if not source.is_file():
        raise FileNotFoundError(f"Missing snapshot source: {source_rel}")

    source_payload = source.read_bytes()
    if snapshot.exists():
        current_payload = snapshot.read_bytes()
        if current_payload != source_payload and not refresh:
            raise RuntimeError(
                "Historical snapshot differs from its source. "
                f"Use --refresh only after explicit approval: {snapshot_rel}"
            )
    if not snapshot.exists() or snapshot.read_bytes() != source_payload:
        write_bytes_atomic(snapshot, source_payload)

    snapshot_payload = snapshot.read_bytes()
    source_hash = sha256_bytes(source_payload)
    snapshot_hash = sha256_bytes(snapshot_payload)
    return {
        "source": source_rel,
        "snapshot": snapshot_rel,
        "bytes": len(snapshot_payload),
        "source_sha256": source_hash,
        "snapshot_sha256": snapshot_hash,
        "byte_equal": source_payload == snapshot_payload,
    }


def build_snapshot_manifest(refresh: bool) -> dict[str, Any]:
    """构建全部快照并返回manifest；build every declared snapshot and return its manifest."""

    records = [snapshot_one(source, target, refresh=refresh) for source, target in SNAPSHOT_SPECS]
    failures = [record["snapshot"] for record in records if not record["byte_equal"]]
    manifest = {
        "package_id": PACKAGE_ID,
        "status": "pass" if not failures else "fail",
        "snapshot_count": len(records),
        "snapshot_total_bytes": sum(int(record["bytes"]) for record in records),
        "large_manifest_policy": (
            "Large canonical manifests remain under final_v0/records/generated and are indexed "
            "by 05_EVIDENCE_INDEX_AND_PROVENANCE.md."
        ),
        "failures": failures,
        "snapshots": records,
    }
    write_json_atomic(MANIFEST_PATH, manifest)
    return manifest


def load_manifest() -> dict[str, Any]:
    """读取并做最小schema检查；load the manifest and perform a minimal schema check."""

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}")
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("snapshots"), list):
        raise RuntimeError("Snapshot manifest has an invalid top-level schema.")
    return data


def verify_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """验证必需文档、快照哈希、当前源漂移和Mermaid图块；verify package requirements."""

    missing_docs = [name for name in REQUIRED_PACKAGE_DOCS if not (PACKAGE_ROOT / name).is_file()]
    snapshot_failures: list[dict[str, str]] = []
    source_drift: list[dict[str, str]] = []
    total_snapshot_bytes = 0

    for record in manifest.get("snapshots", []):
        source_rel = str(record.get("source", ""))
        snapshot_rel = str(record.get("snapshot", ""))
        expected_source_hash = str(record.get("source_sha256", ""))
        expected_snapshot_hash = str(record.get("snapshot_sha256", ""))
        source = FINAL_ROOT / source_rel
        snapshot = PACKAGE_ROOT / snapshot_rel

        if not snapshot.is_file():
            snapshot_failures.append({"snapshot": snapshot_rel, "reason": "missing"})
            continue
        snapshot_payload = snapshot.read_bytes()
        total_snapshot_bytes += len(snapshot_payload)
        actual_snapshot_hash = sha256_bytes(snapshot_payload)
        if actual_snapshot_hash != expected_snapshot_hash:
            snapshot_failures.append({"snapshot": snapshot_rel, "reason": "sha256_mismatch"})

        if not source.is_file():
            source_drift.append({"source": source_rel, "reason": "missing"})
        else:
            actual_source_hash = sha256_bytes(source.read_bytes())
            if actual_source_hash != expected_source_hash:
                source_drift.append({"source": source_rel, "reason": "source_changed_since_snapshot"})

    diagram_files = sorted((PACKAGE_ROOT / "snapshots/algorithm_diagrams").glob("*.md"))
    mermaid_blocks = sum(path.read_text(encoding="utf-8").count("```mermaid") for path in diagram_files)
    status = "pass" if not missing_docs and not snapshot_failures and not source_drift else "fail"
    return {
        "package_id": PACKAGE_ID,
        "status": status,
        "required_document_count": len(REQUIRED_PACKAGE_DOCS),
        "missing_documents": missing_docs,
        "snapshot_count": len(manifest.get("snapshots", [])),
        "snapshot_total_bytes": total_snapshot_bytes,
        "snapshot_failures": snapshot_failures,
        "source_drift": source_drift,
        "algorithm_diagram_file_count": len(diagram_files),
        "mermaid_block_count": mermaid_blocks,
    }


def first_markdown_title(payload: bytes, fallback: str) -> str:
    """提取Markdown首个一级标题；extract the first level-one Markdown heading."""

    text = payload.decode("utf-8", errors="replace")
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def describe_file(relative_path: str, payload: bytes) -> str:
    """生成简明内容感知说明；build a concise content-aware description."""

    path = Path(relative_path)
    suffix = path.suffix.lower()
    if suffix == ".md":
        title = first_markdown_title(payload, path.stem)
        blocks = payload.decode("utf-8", errors="replace").count("```mermaid")
        extra = f"；Mermaid图块={blocks}" if blocks else ""
        return f"Markdown《{title}》{extra}"
    if suffix == ".json":
        try:
            data = json.loads(payload.decode("utf-8"))
            if isinstance(data, dict):
                fields = ",".join(list(data)[:8])
                return f"机器JSON；顶层字段={fields}"
        except (UnicodeDecodeError, json.JSONDecodeError):
            return "JSON解析警告；详见包验证"
        return "机器JSON"
    if suffix == ".jsonl":
        # 中文：先单独计算，避免旧版 Python 禁止 f-string 表达式中的反斜线。
        # English: Compute separately because older Python rejects backslashes in f-string expressions.
        line_count = len(payload.splitlines())
        return f"逐行机器证据；记录数={line_count}"
    return f"{suffix.lstrip('.').upper() or '无扩展名'} 文件"


def build_tree(relative_files: Iterable[str]) -> list[str]:
    """构建稳定Unicode树；build a deterministic Unicode tree."""

    tree: dict[str, dict[str, Any]] = {}
    for relative_path in relative_files:
        node = tree
        for part in Path(relative_path).parts:
            node = node.setdefault(part, {})

    lines = [PACKAGE_ROOT.name + "/"]

    def visit(node: dict[str, dict[str, Any]], prefix: str) -> None:
        names = sorted(node, key=lambda name: (not bool(node[name]), name.lower()))
        for index, name in enumerate(names):
            is_last = index == len(names) - 1
            lines.append(prefix + ("└── " if is_last else "├── ") + name)
            if node[name]:
                visit(node[name], prefix + ("    " if is_last else "│   "))

    visit(tree, "")
    return lines


def render_package_tree() -> str:
    """渲染本包树、字节、SHA与说明；render the package tree and per-file integrity table."""

    files = sorted(path for path in PACKAGE_ROOT.rglob("*") if path.is_file() and path != TREE_PATH)
    relative_files = [path.relative_to(PACKAGE_ROOT).as_posix() for path in files]
    tree_files = sorted(relative_files + [TREE_PATH.name])
    lines = [
        "# M0 历史归档文件树与逐文件说明 / M0 Package Tree and Per-file Descriptions",
        "",
        "> 本文件由 `final_v0/tools/build_m0_history_package.py` 自动生成；索引自身不记录哈希以避免自引用。",
        "",
        "## 树状结构 / Tree",
        "",
        "```text",
        *build_tree(tree_files),
        "```",
        "",
        "## 完整性与内容 / Integrity and content",
        "",
        "| 文件 / File | 字节 / Bytes | SHA-256 | 内容 / Content |",
        "|---|---:|---|---|",
    ]
    for path, relative_path in zip(files, relative_files):
        payload = path.read_bytes()
        description = describe_file(relative_path, payload).replace("|", "¦")
        lines.append(
            f"| `{relative_path}` | {len(payload)} | `{sha256_bytes(payload)}` | {description} |"
        )
    lines.append(
        # 中文：使用动态文件名，确保 v2 树不会伪装成或覆盖 v1 的 06 号树。
        # English: Use the dynamic name so the v2 tree never masquerades as or overwrites the v1 tree.
        f"| `{TREE_PATH.name}` | self | intentionally omitted | 自动生成的本包树和完整性索引自身。 |"
    )
    lines.extend(
        [
            "",
            f"- 永久文件数（含本索引）/ Permanent files including this index：**{len(tree_files)}**。",
            "- 历史快照刷新必须显式 `--refresh` 且应先取得用户确认。",
            "- 总项目树继续由 `final_v0/FINAL_V0_TREE.md` 维护。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """解析安全、非破坏性命令行选项；parse safe, non-destructive CLI options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Intentionally refresh differing historical snapshots after explicit approval.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Read-only verification; do not write snapshots, manifests, or indexes.",
    )
    args = parser.parse_args()
    if args.refresh and args.verify_only:
        parser.error("--refresh and --verify-only are mutually exclusive")
    return args


def main() -> int:
    """构建或只读验证归档；build the package or verify it read-only."""

    args = parse_args()
    relative_to_checked(PACKAGE_ROOT, FINAL_ROOT)

    if args.verify_only:
        verification = verify_manifest(load_manifest())
        print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
        return 0 if verification["status"] == "pass" else 1

    PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = build_snapshot_manifest(refresh=bool(args.refresh))
    verification = verify_manifest(manifest)
    write_json_atomic(VERIFICATION_PATH, verification)
    write_text_atomic(TREE_PATH, render_package_tree())
    print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0 if verification["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
