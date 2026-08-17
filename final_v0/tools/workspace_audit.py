#!/usr/bin/env python3
"""对只读项目执行可追溯扫描，并将全部证据写入 ``final_v0``。

Perform traceable scans of the read-only project and write all evidence to ``final_v0``.

``baseline`` 完整读取根目录文本和 workspace 内全部代码；``input`` 读取指定输入
目录中每个文件的头部；``output`` 完整读取指定输出目录的文本文件，并只登记
非文本文件名与格式元数据。

``baseline`` fully reads root text and every code file in the workspace; ``input`` reads
the bounded head of every file in one input directory; ``output`` fully reads every text
file in one output directory while recording only filename/format metadata for binaries.
"""

from __future__ import annotations

import argparse
import ast
import codecs
import csv
import hashlib
import json
import os
import re
import sys
import zipfile
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
FINAL_ROOT = WORKSPACE_ROOT / "final_v0"
GENERATED_ROOT = FINAL_ROOT / "records" / "generated"
EXCLUDED_TOPS = {".git", "final_v0"}

INPUT_TOPS = {
    "datasets",
    "PPG_Testing_05_01_2026",
    "physionet.org",
    "train_raw",
    "train_labeled",
    "train_val",
    "train_window",
}
OUTPUT_TOPS = {
    ".CNN_results",
    "denoiser_preview_output",
    "models",
    "results",
    "results_detector_v8",
    "results_denoiser_v8",
    "results_frailty3",
    "results_hybrid_denoiser",
    "results_hybrid_denoiser_raw_imu",
    "results_hybrid_denoiser_raw_imu_baseline",
    "results_stage1",
    "results_stage2",
    "results_v7_3",
    "results_v7_4",
    "results_v72_noleak",
    "results_v8_audit",
    "test_asa_classifier",
}
CODE_SUFFIXES = {
    ".py", ".ipynb", ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
    ".js", ".ts", ".tsx", ".jsx", ".r", ".R", ".jl", ".c", ".cc",
    ".cpp", ".h", ".hpp",
}
ROOT_TEXT_SUFFIXES = {
    ".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".jsonl",
    ".csv", ".tsv", ".dockerignore", ".gitignore", ".example",
}
ROOT_TEXT_NAMES = {
    "AGENTS.md", "Dockerfile", "LICENSE", "README.md", "docker-compose.yml",
    "pyproject.toml", ".dockerignore", ".gitignore", ".env", ".env.example",
    ".codex",
}
OUTPUT_TEXT_SUFFIXES = {
    ".csv", ".tsv", ".json", ".jsonl", ".txt", ".md", ".log", ".yaml",
    ".yml", ".toml", ".xml", ".html", ".htm", ".tex", ".py", ".ipynb",
}
SECRET_RE = re.compile(r"(?i)(token|secret|password|passwd|api[_-]?key|credential)")
METRIC_RE = re.compile(
    r"(?i)(balanced.?acc|macro.?f1|accuracy|precision|recall|\bf1\b|\bmae\b|"
    r"\brmse\b|\bbias\b|coverage|loss|snr|correlation|latency|memory|epoch|fold|"
    r"seed|subject|threshold|scorecard|confusion)"
)
PATH_RE = re.compile(
    r"(?i)(dataset|data|input|output|result|model|checkpoint|manifest|train_|"
    r"physionet|ppg_testing|\.csv$|\.json$|\.npz$|\.npy$|\.pt$|\.pth$|"
    r"\.onnx$|\.png$|\.pdf$|\.txt$|\.log$|\.pkl$|\.joblib$)"
)


def now_utc() -> str:
    """返回 ISO UTC 时间；return an ISO UTC timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(path: Path) -> str:
    """返回 workspace 相对路径；return a workspace-relative path."""

    return path.relative_to(WORKSPACE_ROOT).as_posix()


def iter_files(base: Path) -> Iterator[Path]:
    """稳定遍历文件且不跟随目录链接；walk files deterministically without following directory links."""

    for directory, subdirs, names in os.walk(base, followlinks=False):
        subdirs[:] = sorted(subdirs)
        for name in sorted(names):
            path = Path(directory) / name
            if path.is_file():
                yield path


def iter_source_files() -> Iterator[Path]:
    """遍历项目文件并排除 Git 与生成物；walk project files excluding Git and generated artifacts."""

    for child in sorted(WORKSPACE_ROOT.iterdir(), key=lambda item: item.name.lower()):
        if child.name in EXCLUDED_TOPS:
            continue
        if child.is_file():
            yield child
        elif child.is_dir():
            yield from iter_files(child)


def safe_top(name: str, allowed: set[str]) -> Path:
    """验证顶层目标，防止路径越界；validate a top-level target and prevent path escape."""

    if name not in allowed or Path(name).name != name:
        raise ValueError(f"unregistered target: {name!r}")
    path = (WORKSPACE_ROOT / name).resolve()
    if path.parent != WORKSPACE_ROOT.resolve() or not path.is_dir():
        raise ValueError(f"invalid target directory: {name!r}")
    return path


def sha256(data: bytes) -> str:
    """计算不可逆字节摘要；calculate an irreversible byte digest."""

    return hashlib.sha256(data).hexdigest()


def decode(data: bytes) -> tuple[str, str]:
    """解码文本并记录回退编码；decode text and report the fallback encoding."""

    for encoding in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace"), "utf-8-replacement"


def sanitize(text: str, limit: int = 500) -> str:
    """压缩文本并遮蔽疑似凭据；compact text and redact likely credentials."""

    compact = " ".join(text.replace("\x00", " ").split())
    if "=" in compact:
        key, value = compact.split("=", 1)
        if SECRET_RE.search(key):
            compact = f"{key}=<REDACTED:{len(value)} chars>"
    return compact[:limit]


def write_json(path: Path, payload: Any) -> None:
    """在 final_v0 内原子写 JSON；atomically write JSON inside final_v0."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temp.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """在 final_v0 内原子写 JSONL；atomically write JSONL inside final_v0."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temp.replace(path)


def append_run(section: str, target: str, files: int, size: int,
               evidence: list[Path], errors: int) -> dict[str, Any]:
    """追加机器可读扫描账本；append the machine-readable scan ledger."""

    row = {
        "timestamp_utc": now_utc(),
        "section": section,
        "target": target,
        "status": "complete" if errors == 0 else "complete_with_errors",
        "file_count": files,
        "byte_count": size,
        "evidence_files": [rel(path) for path in evidence],
        "error_count": errors,
    }
    ledger = GENERATED_ROOT / "SCAN_RUNS.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return row


def path_like(value: str) -> bool:
    """保守判断字符串是否表达路径；conservatively decide whether a string expresses a path."""

    value = value.strip()
    return bool(value and len(value) <= 500 and "\n" not in value and
                (PATH_RE.search(value) or "/" in value or "\\" in value))


def io_role(context: str) -> str:
    """从代码上下文推断输入或输出；infer input/output role from code context."""

    lowered = context.lower()
    out_hit = any(term in lowered for term in ("save", "write", "output", "result", "mkdir", "export", "dump", "to_csv"))
    in_hit = any(term in lowered for term in ("read", "load", "input", "dataset", "glob", "rglob"))
    if out_hit and not in_hit:
        return "output"
    if in_hit and not out_hit:
        return "input"
    return "unknown"


def resolve_literal(value: str) -> dict[str, Any]:
    """尝试解析静态路径字面量；try to resolve a static path literal."""

    normalized = value.strip().replace("\\", "/")
    if normalized.startswith(("http://", "https://")):
        return {"normalized": normalized, "path_kind": "url", "exists": None, "resolved": None}
    if any(token in normalized for token in ("{", "}", "$", "*", "?", "<", ">")):
        return {"normalized": normalized, "path_kind": "dynamic", "exists": None, "resolved": None}
    candidate = Path(normalized)
    if not candidate.is_absolute():
        candidate = WORKSPACE_ROOT / candidate
    try:
        resolved = candidate.resolve()
        inside = resolved == WORKSPACE_ROOT.resolve() or WORKSPACE_ROOT.resolve() in resolved.parents
        return {
            "normalized": normalized,
            "path_kind": "workspace" if inside else "external",
            "exists": resolved.exists(),
            "resolved": rel(resolved) if inside else str(resolved),
        }
    except (OSError, ValueError):
        return {"normalized": normalized, "path_kind": "unresolved", "exists": None, "resolved": None}


def python_structure(path: Path, text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """解析 Python 定义、依赖和路径；parse Python definitions, dependencies, and paths."""

    lines = text.splitlines()
    functions: set[str] = set()
    classes: set[str] = set()
    imports: set[str] = set()
    refs: list[dict[str, Any]] = []
    try:
        tree = ast.parse(text, filename=rel(path))
        parse_error = None
    except SyntaxError as exc:
        tree = None
        parse_error = f"line {exc.lineno}: {exc.msg}"
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.add(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.add(node.name)
            elif isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and path_like(node.value):
                number = int(getattr(node, "lineno", 0) or 0)
                context = lines[number - 1] if 0 < number <= len(lines) else ""
                refs.append({
                    "source": rel(path), "line": number, "literal": node.value,
                    "io_role": io_role(context), "context": sanitize(context, 300),
                    **resolve_literal(node.value),
                })
    return {
        "functions": sorted(functions), "classes": sorted(classes),
        "imports": sorted(imports), "parse_error": parse_error,
    }, refs


def notebook_structure(path: Path, text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """解析 notebook 单元与路径字符串；parse notebook cells and path strings."""

    try:
        book = json.loads(text)
    except json.JSONDecodeError as exc:
        return {"parse_error": str(exc)}, []
    refs: list[dict[str, Any]] = []
    cells = book.get("cells", [])
    for cell_index, cell in enumerate(cells):
        source = "".join(cell.get("source", []))
        for number, line in enumerate(source.splitlines(), start=1):
            for match in re.finditer(r"(['\"])([^'\"\r\n]{1,500})\1", line):
                value = match.group(2)
                if path_like(value):
                    refs.append({
                        "source": rel(path), "cell_index": cell_index, "line": number,
                        "literal": value, "io_role": io_role(line), "context": sanitize(line, 300),
                        **resolve_literal(value),
                    })
    return {
        "cell_count": len(cells),
        "code_cells": sum(cell.get("cell_type") == "code" for cell in cells),
        "markdown_cells": sum(cell.get("cell_type") == "markdown" for cell in cells),
        "cells_with_outputs": sum(bool(cell.get("outputs")) for cell in cells),
        "kernel": book.get("metadata", {}).get("kernelspec", {}).get("name"),
        "parse_error": None,
    }, refs


def full_text_record(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """完整读取一个文本/代码文件；fully read one text or code file."""

    data = path.read_bytes()
    text, encoding = decode(data)
    if path.name == ".env":
        names = [line.split("=", 1)[0].strip() for line in text.splitlines()
                 if line.strip() and not line.lstrip().startswith("#") and "=" in line]
        structure, refs = {"variable_names": sorted(set(names)), "values_redacted": True}, []
    elif path.suffix == ".py":
        structure, refs = python_structure(path, text)
    elif path.suffix == ".ipynb":
        structure, refs = notebook_structure(path, text)
    else:
        structure, refs = {}, []
    return {
        "path": rel(path), "bytes": len(data), "sha256": sha256(data),
        "encoding": encoding, "line_count": len(text.splitlines()),
        "suffix": path.suffix or "<none>", "read_mode": "full_bytes",
        "structure": structure, "error": None,
    }, refs


def top_summaries() -> list[dict[str, Any]]:
    """统计各顶层目录的文件数、字节数和类型；summarize file counts, bytes, and types per top directory."""

    rows: list[dict[str, Any]] = []
    for child in sorted(WORKSPACE_ROOT.iterdir(), key=lambda item: item.name.lower()):
        if not child.is_dir() or child.name in EXCLUDED_TOPS:
            continue
        count = total = errors = 0
        suffixes: Counter[str] = Counter()
        for path in iter_files(child):
            try:
                size = path.stat().st_size
                count += 1
                total += size
                suffixes[path.suffix.lower() or "<none>"] += 1
            except OSError:
                errors += 1
        role = "input" if child.name in INPUT_TOPS else "output" if child.name in OUTPUT_TOPS else "other"
        rows.append({
            "path": child.name, "role": role, "file_count": count,
            "byte_count": total, "suffix_counts": dict(sorted(suffixes.items())),
            "metadata_errors": errors,
        })
    return rows


def run_baseline() -> dict[str, Any]:
    """运行根文本、全部代码和完整文件树基线；run root-text, all-code, and full-tree baseline."""

    root_rows: list[dict[str, Any]] = []
    code_rows: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    seen: set[str] = set()
    errors = 0
    for path in sorted((item for item in WORKSPACE_ROOT.iterdir() if item.is_file()), key=lambda item: item.name.lower()):
        is_text = path.name in ROOT_TEXT_NAMES or path.suffix in ROOT_TEXT_SUFFIXES or path.suffix in CODE_SUFFIXES
        if is_text:
            try:
                row, refs = full_text_record(path)
            except Exception as exc:  # noqa: BLE001 - 审计保留任意单文件失败。
                row, refs = {"path": rel(path), "bytes": path.stat().st_size,
                             "read_mode": "failed", "error": f"{type(exc).__name__}: {exc}"}, []
                errors += 1
        else:
            row, refs = {"path": rel(path), "bytes": path.stat().st_size,
                         "suffix": path.suffix or "<none>", "read_mode": "metadata_only",
                         "sha256": None, "error": None}, []
        root_rows.append(row)
        references.extend(refs)
        if path.suffix in CODE_SUFFIXES and row.get("read_mode") == "full_bytes":
            code_rows.append(row)
            seen.add(rel(path))

    for path in iter_source_files():
        if path.suffix not in CODE_SUFFIXES or rel(path) in seen:
            continue
        try:
            row, refs = full_text_record(path)
        except Exception as exc:  # noqa: BLE001
            row, refs = {"path": rel(path), "bytes": path.stat().st_size,
                         "read_mode": "failed", "error": f"{type(exc).__name__}: {exc}"}, []
            errors += 1
        code_rows.append(row)
        references.extend(refs)

    unique = {(row.get("source"), row.get("cell_index"), row.get("line"), row.get("literal")): row
              for row in references}
    reference_rows = sorted(unique.values(), key=lambda row: (row["source"], row.get("line", 0), row["literal"]))
    file_rows = []
    total = 0
    for path in iter_source_files():
        try:
            size = path.stat().st_size
            parts = Path(rel(path)).parts
            top = parts[0] if len(parts) > 1 else "<root>"
            role = "input" if top in INPUT_TOPS else "output" if top in OUTPUT_TOPS else "other"
            file_rows.append({"path": rel(path), "top": top, "role": role,
                              "bytes": size, "suffix": path.suffix.lower() or "<none>",
                              "is_symlink": path.is_symlink()})
            total += size
        except OSError as exc:
            file_rows.append({"path": rel(path), "error": f"{type(exc).__name__}: {exc}"})
            errors += 1

    outputs = [
        GENERATED_ROOT / "ROOT_FILES.jsonl", GENERATED_ROOT / "CODE_FILES.jsonl",
        GENERATED_ROOT / "CODE_PATH_REFERENCES.jsonl", GENERATED_ROOT / "TOP_LEVEL_DIRECTORIES.json",
        GENERATED_ROOT / "WORKSPACE_FILES.jsonl", GENERATED_ROOT / "BASELINE_SUMMARY.json",
    ]
    write_jsonl(outputs[0], root_rows)
    write_jsonl(outputs[1], code_rows)
    write_jsonl(outputs[2], reference_rows)
    write_json(outputs[3], top_summaries())
    write_jsonl(outputs[4], file_rows)
    write_json(outputs[5], {
        "timestamp_utc": now_utc(), "root_file_count": len(root_rows),
        "root_full_read_count": sum(row.get("read_mode") == "full_bytes" for row in root_rows),
        "code_full_read_count": sum(row.get("read_mode") == "full_bytes" for row in code_rows),
        "path_reference_count": len(reference_rows), "workspace_file_count": len(file_rows),
        "workspace_byte_count": total, "error_count": errors,
        "security": {"env_values_recorded": False, "excluded_tops": sorted(EXCLUDED_TOPS)},
    })
    return append_run("baseline", "workspace", len(file_rows), total, outputs, errors)


def text_head(path: Path, head: bytes) -> dict[str, Any]:
    """从输入头部推断文本表结构；infer text-table structure from an input head."""

    text, encoding = decode(head)
    lines = [line for line in text.splitlines() if line.strip()]
    delimiter = None
    columns: list[str] = []
    if lines:
        try:
            dialect = csv.Sniffer().sniff("\n".join(lines[:20]), delimiters=",;\t|")
            delimiter = dialect.delimiter
            columns = next(csv.reader([lines[0]], delimiter=delimiter))
        except csv.Error:
            pass
    return {
        "kind": "text", "encoding": encoding, "head_line_count": len(lines),
        "delimiter": delimiter, "column_count": len(columns) or None,
        "columns": [sanitize(value, 200) for value in columns[:100]],
        "head_preview": [sanitize(line, 500) for line in lines[:5]],
    }


def numpy_structure(path: Path) -> dict[str, Any]:
    """只读取 NPY 头部 shape/dtype；read only the NPY header shape/dtype."""

    try:
        import numpy as np  # 延迟依赖 / Lazy dependency.
        with path.open("rb") as handle:
            version = np.lib.format.read_magic(handle)
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = reader(handle)
        return {"kind": "npy", "shape": list(shape), "dtype": str(dtype), "fortran_order": bool(fortran)}
    except Exception as exc:  # noqa: BLE001
        return {"kind": "npy", "header_error": f"{type(exc).__name__}: {exc}"}


def archive_structure(path: Path) -> dict[str, Any]:
    """读取 ZIP/NPZ 目录但不加载数组；read ZIP/NPZ entries without loading arrays."""

    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            return {
                "kind": "npz" if path.suffix.lower() == ".npz" else "zip",
                "entry_count": len(entries),
                "entries": [{"name": item.filename, "bytes": item.file_size,
                             "compressed_bytes": item.compress_size} for item in entries[:500]],
                "entries_truncated": len(entries) > 500,
            }
    except Exception as exc:  # noqa: BLE001
        return {"kind": "archive", "header_error": f"{type(exc).__name__}: {exc}"}


def input_structure(path: Path, head: bytes) -> dict[str, Any]:
    """识别输入文件头和数据结构；identify an input header and data structure."""

    suffix = path.suffix.lower()
    if suffix == ".npy":
        return numpy_structure(path)
    if suffix in {".npz", ".zip", ".xlsx"}:
        return archive_structure(path)
    if suffix in {".csv", ".tsv", ".txt", ".hea", ".json", ".jsonl", ".md"}:
        return text_head(path, head)
    if head.startswith(b"\x89HDF\r\n\x1a\n"):
        return {"kind": "hdf5", "magic": head[:8].hex()}
    if head.startswith(b"MATLAB"):
        return {"kind": "mat", "header": sanitize(head[:128].decode("latin-1", errors="replace"))}
    return {"kind": "binary_or_unknown", "magic_hex": head[:32].hex()}


def run_input(name: str, head_bytes: int) -> dict[str, Any]:
    """扫描指定输入目录中每个文件的头部；scan every file head in one input directory."""

    target = safe_top(name, INPUT_TOPS)
    rows: list[dict[str, Any]] = []
    total = errors = 0
    for path in iter_files(target):
        try:
            size = path.stat().st_size
            with path.open("rb") as handle:
                head = handle.read(head_bytes)
            rows.append({
                "path": rel(path), "bytes": size, "suffix": path.suffix.lower() or "<none>",
                "read_mode": "bounded_head", "head_bytes_requested": head_bytes,
                "head_bytes_read": len(head), "head_sha256": sha256(head),
                "structure": input_structure(path, head), "error": None,
            })
            total += size
        except Exception as exc:  # noqa: BLE001
            rows.append({"path": rel(path), "read_mode": "failed", "error": f"{type(exc).__name__}: {exc}"})
            errors += 1
    manifest = GENERATED_ROOT / "inputs" / f"{name}.jsonl"
    summary = GENERATED_ROOT / "inputs" / f"{name}.summary.json"
    write_jsonl(manifest, rows)
    write_json(summary, {
        "timestamp_utc": now_utc(), "target": name, "file_count": len(rows),
        "byte_count": total, "head_bytes_per_file": head_bytes, "error_count": errors,
        "format_counts": dict(Counter(row.get("structure", {}).get("kind", "error") for row in rows)),
    })
    return append_run("input", name, len(rows), total, [manifest, summary], errors)


def probably_text(path: Path) -> bool:
    """识别输出文本；identify output text using suffix or a tiny extensionless probe."""

    if path.suffix.lower() in OUTPUT_TEXT_SUFFIXES or path.name.lower().startswith("readme"):
        return True
    if path.stat().st_size == 0:
        return True
    if path.suffix:
        return False
    with path.open("rb") as handle:
        probe = handle.read(8192)
    if b"\x00" in probe:
        return False
    try:
        decoded = probe.decode("utf-8")
    except UnicodeDecodeError:
        return False
    printable = sum(char.isprintable() or char.isspace() for char in decoded)
    return not decoded or printable / len(decoded) >= 0.90


def stream_output_text(path: Path) -> dict[str, Any]:
    """流式读至 EOF 并提取输出 schema/指标行；stream to EOF and extract output schema/metric lines."""

    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    head = bytearray()
    tail = bytearray()
    first: list[str] = []
    last: deque[str] = deque(maxlen=5)
    metrics: list[str] = []
    headings: list[str] = []
    carry = ""
    byte_count = newline_count = 0

    def process(line: str) -> None:
        clean = sanitize(line, 1000)
        if len(first) < 5:
            first.append(clean)
        last.append(clean)
        if len(metrics) < 250 and METRIC_RE.search(line):
            metrics.append(clean)
        if len(headings) < 100 and line.lstrip().startswith("#"):
            headings.append(clean)

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            byte_count += len(block)
            newline_count += block.count(b"\n")
            if len(head) < 65536:
                head.extend(block[: 65536 - len(head)])
            tail.extend(block)
            if len(tail) > 65536:
                del tail[: len(tail) - 65536]
            decoded = decoder.decode(block)
            lines = (carry + decoded).splitlines(keepends=True)
            carry = ""
            if lines and not lines[-1].endswith(("\n", "\r")):
                carry = lines.pop()
            for line in lines:
                process(line)
    carry += decoder.decode(b"", final=True)
    if carry:
        process(carry)

    structure: dict[str, Any] = {
        "line_count": newline_count + int(byte_count > 0 and not bytes(tail).endswith((b"\n", b"\r"))),
        "first_lines": first, "last_lines": list(last), "metric_lines": metrics,
        "headings": headings,
    }
    if path.suffix.lower() in {".csv", ".tsv"}:
        lines = [line for line in decode(bytes(head))[0].splitlines() if line.strip()]
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        if lines:
            try:
                delimiter = csv.Sniffer().sniff("\n".join(lines[:20]), delimiters=",;\t|").delimiter
            except csv.Error:
                pass
            try:
                columns = next(csv.reader([lines[0]], delimiter=delimiter))
            except (csv.Error, StopIteration):
                columns = []
        else:
            columns = []
        structure.update({"delimiter": delimiter, "columns": columns[:500], "column_count": len(columns)})
    return {
        "read_mode": "full_bytes", "bytes_read": byte_count,
        "sha256": digest.hexdigest(), "encoding": "utf-8-with-replacement-on-error",
        "structure": structure,
    }


def run_output(name: str) -> dict[str, Any]:
    """扫描指定输出目录并完整读取全部文本；scan one output directory and fully read all text."""

    target = safe_top(name, OUTPUT_TOPS)
    rows: list[dict[str, Any]] = []
    total = errors = text_count = binary_count = 0
    for path in iter_files(target):
        try:
            size = path.stat().st_size
            row: dict[str, Any] = {"path": rel(path), "bytes": size,
                                   "suffix": path.suffix.lower() or "<none>", "error": None}
            if probably_text(path):
                row.update(stream_output_text(path))
                text_count += 1
            else:
                # 非文本输出不读取载荷。/ Binary output payloads are intentionally not read.
                row.update({"read_mode": "filename_metadata_only", "format": path.suffix.lower() or "unknown"})
                binary_count += 1
            rows.append(row)
            total += size
        except Exception as exc:  # noqa: BLE001
            rows.append({"path": rel(path), "read_mode": "failed", "error": f"{type(exc).__name__}: {exc}"})
            errors += 1
    manifest = GENERATED_ROOT / "outputs" / f"{name}.jsonl"
    summary = GENERATED_ROOT / "outputs" / f"{name}.summary.json"
    write_jsonl(manifest, rows)
    write_json(summary, {
        "timestamp_utc": now_utc(), "target": name, "file_count": len(rows),
        "byte_count": total, "text_full_read_count": text_count,
        "binary_metadata_only_count": binary_count, "error_count": errors,
        "suffix_counts": dict(Counter(row.get("suffix", "<error>") for row in rows)),
    })
    return append_run("output", name, len(rows), total, [manifest, summary], errors)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """解析分段扫描参数；parse sectioned scan arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("section", choices=("baseline", "input", "output"))
    parser.add_argument("--name")
    parser.add_argument("--head-bytes", type=int, default=65536)
    args = parser.parse_args(argv)
    if args.section in {"input", "output"} and not args.name:
        parser.error("--name is required for input/output scans")
    if not 1024 <= args.head_bytes <= 1024 * 1024:
        parser.error("--head-bytes must be between 1024 and 1048576")
    return args


def main(argv: list[str] | None = None) -> int:
    """运行一个扫描事务并登记结果；run one scan transaction and register its result."""

    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.section == "baseline":
        result = run_baseline()
    elif args.section == "input":
        result = run_input(args.name, args.head_bytes)
    else:
        result = run_output(args.name)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["error_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

