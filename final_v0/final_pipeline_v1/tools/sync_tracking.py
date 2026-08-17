#!/usr/bin/env python3
"""同步 V1 工作记录、算法索引与详细文件树 / Sync V1 tracking artifacts.

中文：本脚本只写 final_pipeline_v1 内三个生成文档；跟踪文档自身更新不递归产生
新日志。所有排序使用相对路径的 UTF-8 字节序，确保跨运行稳定。

English: This script writes only three generated documents inside final_pipeline_v1.
Tracking updates do not recursively create new log entries. Relative paths are sorted
by UTF-8 bytes for deterministic output.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "records" / "log_entries"
DIAGRAM_ROOT = ROOT / "docs" / "algorithms"


def _sha256(path: Path) -> str:
    """逐字节哈希 / Return a byte-exact SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    """在 V1 根内原子写文本 / Atomically write text inside the V1 root."""

    target = path.resolve(strict=False)
    target.relative_to(ROOT.resolve())
    temporary = target.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")
    temporary.replace(target)


def _sorted_files(root: Path) -> list[Path]:
    """稳定列出文件且排除缓存/临时文件 / List stable non-cache files."""

    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".tmp"}
        ),
        key=lambda item: item.relative_to(root).as_posix().encode("utf-8"),
    )


def build_work_log() -> str:
    """聚合不可变阶段日志 / Aggregate immutable phase fragments."""

    fragments = _sorted_files(LOG_ROOT) if LOG_ROOT.exists() else []
    lines = [
        "# Final Pipeline V1 work log / 工作日志",
        "",
        "> Auto-generated from records/log_entries; tracking updates are not project events.",
        "> 由 records/log_entries 自动生成；追踪文档更新本身不计项目事件。",
        "",
    ]
    for fragment in fragments:
        lines.extend([fragment.read_text(encoding="utf-8").strip(), "", "---", ""])
    return "\n".join(lines)


def build_algorithm_index() -> str:
    """生成算法图索引 / Build the algorithm-document index."""

    paths = [
        path
        for path in (_sorted_files(DIAGRAM_ROOT) if DIAGRAM_ROOT.exists() else [])
        if path.name != "README.md"
    ]
    lines = [
        "# Algorithm diagrams / 算法图",
        "",
        "| Path / 路径 | Bytes / 字节 | SHA-256 |",
        "|---|---:|---|",
    ]
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        lines.append(f"| `{rel}` | {path.stat().st_size} | `{_sha256(path)}` |")
    return "\n".join(lines)


def build_tree() -> str:
    """生成逐文件树和完整性说明 / Build a detailed file tree and integrity index."""

    excluded = {ROOT / "PROJECT_TREE.md"}
    paths = [path for path in _sorted_files(ROOT) if path not in excluded]
    lines = [
        "# Final Pipeline V1 detailed tree / 详细文件树",
        "",
        "| Path / 路径 | Bytes / 字节 | SHA-256 |",
        "|---|---:|---|",
    ]
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        lines.append(f"| `{rel}` | {path.stat().st_size} | `{_sha256(path)}` |")
    lines.extend(
        [
            "",
            "说明 / Note: PROJECT_TREE.md omits its own hash to avoid recursive self-reference.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """执行三项同步 / Run all three tracking synchronizations."""

    _atomic_write(ROOT / "WORK_LOG.md", build_work_log())
    _atomic_write(DIAGRAM_ROOT / "README.md", build_algorithm_index())
    _atomic_write(ROOT / "PROJECT_TREE.md", build_tree())


if __name__ == "__main__":
    main()

