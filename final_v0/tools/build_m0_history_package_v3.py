#!/usr/bin/env python3
"""追加构建并验证 M0 v3 归档；additively build and verify the M0 v3 archive.

中文
----
本工具以 v2 快照清单为冻结基线，只为本轮 Activity/Motion 监督决定、算法图和
最新算法图验证创建新的 v3 快照。它绝不覆盖内容不同的既有快照，也不修改 v1/v2
manifest、verification 或 tree。全部写入都限制在专用 M0 包目录。

English
-------
This tool uses the frozen v2 snapshot manifest as its baseline and adds v3 snapshots
for the confirmed Activity/Motion decision, the new algorithm diagram, and the current
diagram verification. It never overwrites a differing historical snapshot and never
modifies v1/v2 manifests, verifications, or trees. Every write stays inside the M0 package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


# 中文：从脚本位置解析允许写入的 final_v0 与 M0 包根，避免调用目录注入。
# English: Resolve the writable final_v0 and package roots from the script location.
FINAL_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = FINAL_ROOT / "M0_history_MA_denoising_detector_HR_feature"
BASE_MANIFEST_PATH = PACKAGE_ROOT / "M0_SOURCE_SNAPSHOT_MANIFEST_V2.json"
MANIFEST_PATH = PACKAGE_ROOT / "M0_SOURCE_SNAPSHOT_MANIFEST_V3.json"
VERIFICATION_PATH = PACKAGE_ROOT / "M0_PACKAGE_VERIFICATION_V3.json"
TREE_PATH = PACKAGE_ROOT / "10_M0_PACKAGE_TREE_V3.md"
PACKAGE_ID = "M0-20260803-local-audit-v3-activity-motion"


# 中文：新快照使用从未使用过的目标路径；同名历史文件保持不可变。
# English: New snapshots use fresh targets so same-named historical files stay immutable.
NEW_SNAPSHOT_SPECS: tuple[tuple[str, str], ...] = (
    (
        "records/decisions/20260803_m0_activity_motion_supervision.md",
        "snapshots/records/decisions/20260803_m0_activity_motion_supervision.md",
    ),
    (
        "algorithm_diagrams/m0/08_ACTIVITY_MOTION_TRANSFER_AND_RECOVERY_FEATURES.md",
        "snapshots/algorithm_diagrams/08_ACTIVITY_MOTION_TRANSFER_AND_RECOVERY_FEATURES.md",
    ),
)


# 中文：这些文档是 v3 可读入口；验证缺一即失败。
# English: These documents are the required human-readable v3 entry points.
REQUIRED_PACKAGE_DOCS: tuple[str, ...] = (
    "00_CURRENT_STATUS_V3.md",
    "README.md",
    "01_M0_COMPLETE_RESULTS_AND_DECISIONS.md",
    "02_CANDIDATE_SCRIPTS_MOTION_DENOISING_DYNAMIC_HR.md",
    "03_FIVE_METHOD_FAMILIES_CODE_TEST_IMPLEMENTATION_AUDIT.md",
    "04_UNIFIED_IMPLEMENTATION_AND_BENCHMARK_CONTRACT.md",
    "05_EVIDENCE_INDEX_AND_PROVENANCE.md",
    "07_CONFIRMED_MADENOISER_FOLLOWUP_ROADMAP.md",
    "09_ACTIVITY_MOTION_SUPERVISION_THREE_CLASS_HISTORY_AND_RECOVERY.md",
    "evidence/EARLY_MULTICLASS_SEARCH_AUDIT.json",
    "evidence/MOTION29_DATA_AUDIT.json",
)


def sha256_bytes(payload: bytes) -> str:
    """返回稳定 SHA-256；return a stable SHA-256 digest."""

    return hashlib.sha256(payload).hexdigest()


def relative_to_checked(path: Path, root: Path) -> str:
    """验证路径未逃逸并返回POSIX相对路径；validate containment and return POSIX relative path."""

    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"Path escapes allowed root: {resolved_path} not under {resolved_root}") from exc


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    """在同目录原子写入；atomically write through a same-directory temporary file."""

    relative_to_checked(path, PACKAGE_ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    """写UTF-8/LF文本；write UTF-8 text with LF newlines."""

    write_bytes_atomic(path, text.replace("\r\n", "\n").encode("utf-8"))


def write_json_atomic(path: Path, data: Any) -> None:
    """写稳定排序、禁止NaN的JSON；write deterministic JSON and reject NaN."""

    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    write_text_atomic(path, payload)


def load_base_specs() -> list[tuple[str, str]]:
    """从冻结v2清单导出快照规范；derive snapshot specifications from the frozen v2 manifest."""

    if not BASE_MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing frozen v2 manifest: {BASE_MANIFEST_PATH}")
    base = json.loads(BASE_MANIFEST_PATH.read_text(encoding="utf-8"))
    records = base.get("snapshots")
    if not isinstance(records, list):
        raise RuntimeError("The v2 manifest has an invalid snapshots field.")

    specs: list[tuple[str, str]] = []
    for record in records:
        source = str(record.get("source", ""))
        snapshot = str(record.get("snapshot", ""))
        if not source or not snapshot:
            raise RuntimeError("The v2 manifest contains an incomplete snapshot record.")

        # 中文：算法图验证在新增图后必然变化；v3必须写新目标，不能覆盖v2验证快照。
        # English: Diagram verification changes after a new diagram, so v3 uses a new target.
        if source == "records/generated/ALGORITHM_DIAGRAM_VERIFICATION.json":
            snapshot = "snapshots/verification/ALGORITHM_DIAGRAM_VERIFICATION_ROUTE_V3.json"
        specs.append((source, snapshot))

    specs.extend(NEW_SNAPSHOT_SPECS)
    targets = [target for _, target in specs]
    if len(targets) != len(set(targets)):
        raise RuntimeError("Duplicate v3 snapshot target detected.")
    return specs


def snapshot_one(source_rel: str, snapshot_rel: str) -> dict[str, Any]:
    """按字节创建或核验单个快照；create or verify one byte-for-byte snapshot."""

    source = FINAL_ROOT / source_rel
    snapshot = PACKAGE_ROOT / snapshot_rel
    relative_to_checked(source, FINAL_ROOT)
    relative_to_checked(snapshot, PACKAGE_ROOT)
    if not source.is_file():
        raise FileNotFoundError(f"Missing snapshot source: {source_rel}")

    source_payload = source.read_bytes()
    if snapshot.exists():
        snapshot_payload = snapshot.read_bytes()
        if snapshot_payload != source_payload:
            raise RuntimeError(
                "Refusing to overwrite a differing historical snapshot: "
                f"{snapshot_rel}. Create a new versioned target instead."
            )
    else:
        write_bytes_atomic(snapshot, source_payload)

    snapshot_payload = snapshot.read_bytes()
    return {
        "source": source_rel,
        "snapshot": snapshot_rel,
        "bytes": len(snapshot_payload),
        "source_sha256": sha256_bytes(source_payload),
        "snapshot_sha256": sha256_bytes(snapshot_payload),
        "byte_equal": source_payload == snapshot_payload,
    }


def build_manifest() -> dict[str, Any]:
    """构建v3快照和源—快照清单；build v3 snapshots and their source mapping."""

    records = [snapshot_one(source, target) for source, target in load_base_specs()]
    failures = [record["snapshot"] for record in records if not record["byte_equal"]]
    manifest = {
        "package_id": PACKAGE_ID,
        "status": "pass" if not failures else "fail",
        "base_manifest": BASE_MANIFEST_PATH.name,
        "snapshot_count": len(records),
        "snapshot_total_bytes": sum(int(record["bytes"]) for record in records),
        "failures": failures,
        "snapshots": records,
    }
    write_json_atomic(MANIFEST_PATH, manifest)
    return manifest


def load_manifest() -> dict[str, Any]:
    """读取v3清单并检查最小schema；load the v3 manifest and validate its minimal schema."""

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing v3 manifest: {MANIFEST_PATH}")
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("snapshots"), list):
        raise RuntimeError("The v3 manifest has an invalid top-level schema.")
    return data


def verify_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """验证文档、哈希、源漂移和算法图；verify documents, hashes, source drift, and diagrams."""

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
        if sha256_bytes(snapshot_payload) != expected_snapshot_hash:
            snapshot_failures.append({"snapshot": snapshot_rel, "reason": "sha256_mismatch"})

        if not source.is_file():
            source_drift.append({"source": source_rel, "reason": "missing"})
        elif sha256_bytes(source.read_bytes()) != expected_source_hash:
            source_drift.append({"source": source_rel, "reason": "source_changed_since_snapshot"})

    diagrams = sorted((PACKAGE_ROOT / "snapshots/algorithm_diagrams").glob("*.md"))
    mermaid_blocks = sum(path.read_text(encoding="utf-8").count("```mermaid") for path in diagrams)
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
        "algorithm_diagram_file_count": len(diagrams),
        "mermaid_block_count": mermaid_blocks,
    }


def first_markdown_title(payload: bytes, fallback: str) -> str:
    """提取第一个一级标题；extract the first level-one Markdown heading."""

    text = payload.decode("utf-8", errors="replace")
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def describe_file(relative_path: str, payload: bytes) -> str:
    """生成内容感知的短说明；build a concise content-aware description."""

    path = Path(relative_path)
    suffix = path.suffix.lower()
    if suffix == ".md":
        title = first_markdown_title(payload, path.stem)
        blocks = payload.decode("utf-8", errors="replace").count("```mermaid")
        return f"Markdown《{title}》" + (f"；Mermaid图块={blocks}" if blocks else "")
    if suffix == ".json":
        try:
            data = json.loads(payload.decode("utf-8"))
            if isinstance(data, dict):
                return "机器JSON；顶层字段=" + ",".join(list(data)[:8])
        except (UnicodeDecodeError, json.JSONDecodeError):
            return "JSON解析警告；详见v3验证"
        return "机器JSON"
    if suffix == ".jsonl":
        return f"逐行机器证据；记录数={len(payload.splitlines())}"
    if suffix == ".png":
        return "历史结果PNG证据；内容与来源见专题和证据索引"
    return f"{suffix.lstrip('.').upper() or '无扩展名'} 文件"


def build_tree(relative_files: Iterable[str]) -> list[str]:
    """生成稳定Unicode树；build a deterministic Unicode tree."""

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
    """渲染v3包树、字节、哈希与说明；render the v3 tree, bytes, hashes, and descriptions."""

    files = sorted(path for path in PACKAGE_ROOT.rglob("*") if path.is_file() and path != TREE_PATH)
    relative_files = [path.relative_to(PACKAGE_ROOT).as_posix() for path in files]
    all_tree_files = sorted(relative_files + [TREE_PATH.name])
    lines = [
        "# M0 v3 文件树与逐文件说明 / M0 v3 Tree and Per-file Descriptions",
        "",
        "> 由 `final_v0/tools/build_m0_history_package_v3.py` 自动生成；本索引不自哈希。",
        "",
        "## 树状结构 / Tree",
        "",
        "```text",
        *build_tree(all_tree_files),
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
        lines.append(f"| `{relative_path}` | {len(payload)} | `{sha256_bytes(payload)}` | {description} |")
    lines.extend(
        [
            f"| `{TREE_PATH.name}` | self | intentionally omitted | 自动生成的v3包树自身。 |",
            "",
            f"- 永久文件数（含本索引）：**{len(all_tree_files)}**。",
            "- v1/v2历史文件保持原字节；v3只追加新决定、算法图、验证和证据。",
            "- 总项目树由 `final_v0/FINAL_V0_TREE.md` 维护。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """只暴露构建与只读验证；expose build and read-only verification only."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Read-only verification; do not write snapshots, manifests, or tree indexes.",
    )
    return parser.parse_args()


def main() -> int:
    """构建或只读验证v3；build or read-only verify v3."""

    args = parse_args()
    relative_to_checked(PACKAGE_ROOT, FINAL_ROOT)
    if args.verify_only:
        verification = verify_manifest(load_manifest())
        print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
        return 0 if verification["status"] == "pass" else 1

    PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    verification = verify_manifest(manifest)
    write_json_atomic(VERIFICATION_PATH, verification)
    write_text_atomic(TREE_PATH, render_package_tree())
    print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0 if verification["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
