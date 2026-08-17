#!/usr/bin/env python3
"""Build a content-aware file tree and per-file description for final_v0.

中文：为 ``final_v0`` 的每个永久文件生成树状结构、字节数、SHA-256和内容感知说明。
Markdown 会提取标题与首个实质段落；JSON/JSONL 会说明证据范围和记录数；Python 会
提取模块文档和主要入口。工具只读取 ``final_v0``，并只重写 ``FINAL_V0_TREE.md``。

English: Generate a tree, byte size, SHA-256, and content-aware description for every
permanent file in ``final_v0``. Markdown descriptions use headings and substantive
content; JSON/JSONL descriptions state evidence scope and record counts; Python
descriptions use module documentation and entry points. Only ``FINAL_V0_TREE.md`` is
rewritten.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any


FINAL_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = FINAL_ROOT / "FINAL_V0_TREE.md"
MAX_DESCRIPTION_CHARS = 420


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 digest / 返回 SHA-256 摘要。"""

    return hashlib.sha256(payload).hexdigest()


def clean_table_text(value: str) -> str:
    """Normalize prose for a Markdown table cell / 规范化表格单元中的说明文本。"""

    compact = " ".join(value.replace("|", "¦").split())
    if len(compact) > MAX_DESCRIPTION_CHARS:
        return compact[: MAX_DESCRIPTION_CHARS - 1].rstrip() + "…"
    return compact


def first_markdown_content(text: str) -> tuple[str, str]:
    """Return the first title and substantive prose / 返回首标题与首个实质说明。"""

    title = "Markdown document"
    candidates: list[str] = []
    in_fence = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line:
            continue
        if line.startswith("# ") and title == "Markdown document":
            title = line[2:].strip()
            continue
        if line.startswith("#") or line.startswith(">"):
            continue
        if line.startswith("|"):
            continue
        # 中文：保留首个普通段落或前两个要点，使日志/报告说明具有具体内容。
        # English: Keep the first prose line or two bullets for a specific description.
        candidates.append(line.lstrip("- "))
        if len(candidates) >= 2:
            break
    return title, "；".join(candidates)


def describe_markdown(relative_path: str, payload: bytes) -> str:
    """Describe a Markdown report, log, or diagram / 描述 Markdown 报告、日志或图。"""

    text = payload.decode("utf-8", errors="replace")
    title, detail = first_markdown_content(text)
    if relative_path.startswith("algorithm_diagrams/"):
        block_count = text.count("```mermaid")
        prefix = f"算法图《{title}》；含 {block_count} 个 Mermaid 图块"
    elif relative_path.startswith("records/log_entries/"):
        prefix = f"不可变工作日志《{title}》"
    elif relative_path.startswith("records/pending_agent_updates/"):
        prefix = f"待用户要求后处理的 `_agent` 候选《{title}》"
    else:
        prefix = f"文档《{title}》"
    return clean_table_text(prefix + (f"；{detail}" if detail else ""))


def compact_json_fields(data: Any) -> str:
    """Summarize useful top-level JSON fields / 汇总有信息量的 JSON 顶层字段。"""

    if not isinstance(data, dict):
        size = len(data) if isinstance(data, list) else 1
        return f"top-level={type(data).__name__}, items={size}"
    preferred = (
        "status",
        "directory",
        "mode",
        "file_count",
        "total_files",
        "total_bytes",
        "text_files",
        "binary_files",
        "error_count",
        "diagram_file_count",
        "mermaid_block_count",
        "failures",
    )
    parts: list[str] = []
    for key in preferred:
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, list):
            value = f"list[{len(value)}]"
        elif isinstance(value, dict):
            value = f"object[{len(value)}]"
        parts.append(f"{key}={value}")
    if not parts:
        parts.append("keys=" + ",".join(list(data)[:10]))
    return "; ".join(parts)


def describe_json(relative_path: str, payload: bytes) -> str:
    """Describe one strict JSON evidence file / 描述一份严格 JSON 证据。"""

    try:
        data = json.loads(payload.decode("utf-8"))
        fields = compact_json_fields(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fields = f"JSON parse warning={type(exc).__name__}"
    name = Path(relative_path).name
    return clean_table_text(f"机器可读 JSON 证据 `{name}`；{fields}")


def describe_jsonl(relative_path: str, payload: bytes) -> str:
    """Describe a line-delimited scan manifest / 描述逐行扫描 manifest。"""

    line_count = payload.count(b"\n")
    name = Path(relative_path).name
    if "/inputs/" in relative_path:
        scope = f"输入目录 `{name.removesuffix('.jsonl')}` 的逐文件头部/schema manifest"
    elif "/outputs/" in relative_path:
        scope = f"输出目录 `{name.removesuffix('.jsonl')}` 的文本EOF/二进制元数据 manifest"
    else:
        special_scopes = {
            "WORKSPACE_FILES.jsonl": "workspace全文件树元数据 manifest",
            "ROOT_FILES.jsonl": "根目录逐文件完整读取 manifest",
            "CODE_FILES.jsonl": "全部代码/notebook逐字节读取与结构 manifest",
            "CODE_PATH_REFERENCES.jsonl": "代码静态输入/输出路径字符串引用清单",
            "SCAN_RUNS.jsonl": "baseline、输入和输出扫描事务账本",
        }
        scope = special_scopes.get(name, "逐行机器证据 manifest")
    return clean_table_text(f"{scope}；{line_count} 条记录；每行保留路径、读取模式、结构和完整性字段")


def describe_python(relative_path: str, payload: bytes) -> str:
    """Describe a Python audit tool from its AST / 从 AST 描述 Python 审计工具。"""

    text = payload.decode("utf-8", errors="replace")
    try:
        tree = ast.parse(text, filename=relative_path)
        doc = ast.get_docstring(tree) or "No module docstring."
        first_doc_line = next((line.strip() for line in doc.splitlines() if line.strip()), doc)
        names = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        entries = ", ".join(names[:8]) if names else "script entry only"
        return clean_table_text(f"带中英文说明的 final_v0 工具；{first_doc_line}；主要入口：{entries}")
    except SyntaxError as exc:
        return clean_table_text(f"Python工具；AST解析失败：line {exc.lineno}: {exc.msg}")


def describe_file(relative_path: str, payload: bytes) -> str:
    """Dispatch to a format-aware description / 按格式分派内容说明。"""

    suffix = Path(relative_path).suffix.lower()
    if suffix == ".md":
        return describe_markdown(relative_path, payload)
    if suffix == ".json":
        return describe_json(relative_path, payload)
    if suffix == ".jsonl":
        return describe_jsonl(relative_path, payload)
    if suffix == ".py":
        return describe_python(relative_path, payload)
    return clean_table_text(
        f"{suffix.lstrip('.').upper() or '无扩展名'} 文件；用途由路径和相邻审计记录定义"
    )


def build_tree(relative_files: list[str]) -> list[str]:
    """Build a deterministic Unicode tree / 构建稳定排序的 Unicode 树。"""

    tree: dict[str, dict[str, Any]] = {}
    for relative_path in relative_files:
        node = tree
        for part in Path(relative_path).parts:
            node = node.setdefault(part, {})

    lines = ["final_v0/"]

    def visit(node: dict[str, dict[str, Any]], prefix: str) -> None:
        names = sorted(node, key=lambda name: (not bool(node[name]), name.lower()))
        for index, name in enumerate(names):
            last = index == len(names) - 1
            lines.append(prefix + ("└── " if last else "├── ") + name)
            if node[name]:
                visit(node[name], prefix + ("    " if last else "│   "))

    visit(tree, "")
    return lines


def render_index() -> str:
    """Render the complete detailed index / 渲染完整详细索引。"""

    files = sorted(path for path in FINAL_ROOT.rglob("*") if path.is_file() and path != INDEX_PATH)
    relative_files = [path.relative_to(FINAL_ROOT).as_posix() for path in files]
    all_tree_files = sorted(relative_files + [INDEX_PATH.name])
    lines = [
        "# `final_v0` 文件树与逐文件详细说明 / File Tree and Detailed Per-file Descriptions",
        "",
        "> 本文件由 `tools/update_final_v0_index_detailed.py` 自动生成。索引自身不记录 SHA-256，以避免自引用。",
        "",
        "## 树状结构 / Tree",
        "",
        "```text",
        *build_tree(all_tree_files),
        "```",
        "",
        "## 逐文件内容与完整性 / Per-file content and integrity",
        "",
        "| 文件 / File | 字节 / Bytes | SHA-256 | 内容详细说明 / Detailed content description |",
        "|---|---:|---|---|",
    ]

    for path, relative_path in zip(files, relative_files):
        payload = path.read_bytes()
        lines.append(
            f"| `{relative_path}` | {len(payload)} | `{sha256_bytes(payload)}` | "
            f"{describe_file(relative_path, payload)} |"
        )
    lines.append(
        "| `FINAL_V0_TREE.md` | self | intentionally omitted | 自动生成的本文件树、内容说明和完整性索引自身。 |"
    )
    lines.extend(
        [
            "",
            "## 完整性与更新规则 / Integrity and update rules",
            "",
            f"- 永久文件总数（含本索引）：**{len(all_tree_files)}**。",
            "- 每次逻辑写入后运行本工具；索引自身的更新不递归产生日志。",
            "- 所有非索引文件必须同时具有字节数、SHA-256和内容感知说明；缺一项即视为未验证。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Atomically rebuild the index / 原子重建索引。"""

    temp_path = INDEX_PATH.with_suffix(".md.tmp")
    temp_path.write_text(render_index(), encoding="utf-8", newline="\n")
    temp_path.replace(INDEX_PATH)


if __name__ == "__main__":
    main()

