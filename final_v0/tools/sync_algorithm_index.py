#!/usr/bin/env python3
"""Rebuild the algorithm-diagram index from Markdown sources.

中文：扫描 ``final_v0/algorithm_diagrams`` 中除 README 外的 Markdown，提取标题、
首段说明、字节数和 SHA-256，并机械重建算法图索引。该索引属于追踪文档；更新它本身
不会再次触发日志更新，避免递归写入。

English: Scan Markdown diagrams below ``final_v0/algorithm_diagrams`` (excluding
README), extract the title, first descriptive paragraph, byte size, and SHA-256,
then mechanically rebuild the diagram index. The index is a tracking document,
so rewriting it does not recursively create another log entry.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


# 中文：从脚本位置解析根目录，避免依赖当前工作目录。
# English: Resolve roots from this script so execution is independent of cwd.
FINAL_ROOT = Path(__file__).resolve().parents[1]
DIAGRAM_ROOT = FINAL_ROOT / "algorithm_diagrams"
INDEX_PATH = DIAGRAM_ROOT / "README.md"


def sha256_bytes(payload: bytes) -> str:
    """Return a stable SHA-256 digest / 返回稳定的 SHA-256 摘要。"""

    return hashlib.sha256(payload).hexdigest()


def extract_title_and_summary(text: str, fallback: str) -> tuple[str, str]:
    """Extract the first heading and prose paragraph / 提取首标题与首个说明段。"""

    title = fallback
    summary_lines: list[str] = []
    in_fence = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line:
            if summary_lines:
                break
            continue
        if line.startswith("# ") and title == fallback:
            title = line[2:].strip()
            continue
        if line.startswith("#") or line.startswith("-") or line.startswith("|"):
            if summary_lines:
                break
            continue
        summary_lines.append(line)
    summary = " ".join(summary_lines) if summary_lines else "Mermaid algorithm structure and audit annotations."
    return title, summary


def main() -> None:
    """Build and atomically replace the Markdown index / 构建并原子替换索引。"""

    entries: list[tuple[str, str, str, int, str]] = []
    for path in sorted(DIAGRAM_ROOT.rglob("*.md")):
        if path == INDEX_PATH:
            continue
        payload = path.read_bytes()
        text = payload.decode("utf-8", errors="replace")
        rel = path.relative_to(FINAL_ROOT).as_posix()
        title, summary = extract_title_and_summary(text, path.stem)
        entries.append((rel, title, summary, len(payload), sha256_bytes(payload)))

    lines = [
        "# Algorithm Diagram Registry / 算法图索引",
        "",
        "> 此文件由 `tools/sync_algorithm_index.py` 自动生成；请编辑具体图文件，不要手工编辑本索引。",
        "> Generated mechanically; edit the source diagram files rather than this index.",
        "",
        f"- Diagram documents / 图文档数量：{len(entries)}",
        "- Format / 格式：Markdown + Mermaid",
        "- Convention / 约定：实线为运行数据流；虚线为监督、评价、风险或审计引用。",
        "",
        "## Diagram files / 图文件",
        "",
    ]
    for rel, title, summary, size, digest in entries:
        lines.extend(
            [
                f"### `{rel}`",
                "",
                f"- 标题 / Title：{title}",
                f"- 内容 / Content：{summary}",
                f"- 大小 / Bytes：{size}",
                f"- SHA-256：`{digest}`",
                "",
            ]
        )

    # 中文：先写同目录临时文件，再 replace，防止中断留下半个索引。
    # English: Write a same-directory temporary file before replace to avoid partial output.
    temp_path = INDEX_PATH.with_suffix(".md.tmp")
    temp_path.write_text("\n".join(lines), encoding="utf-8")
    temp_path.replace(INDEX_PATH)


if __name__ == "__main__":
    main()

