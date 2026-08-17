#!/usr/bin/env python3
"""从不可变片段重建 final_v0 的两份追踪主文档。

Rebuild the two final_v0 tracking documents from immutable fragments.

日志和待录入 `_agent` 草稿采用 append-only 片段保存，避免并发或沙箱故障导致
既有记录被覆盖。该工具只写 ``records/WORK_LOG.md`` 和
``records/PENDING_AGENT_UPDATES.md``。

Work-log and pending-agent-update entries are stored as append-only fragments to prevent
loss during concurrent or sandbox-constrained updates. This tool writes only the two
tracking documents named above.
"""

from __future__ import annotations

from pathlib import Path


FINAL_ROOT = Path(__file__).resolve().parents[1]
RECORDS_ROOT = FINAL_ROOT / "records"
LOG_PARTS = RECORDS_ROOT / "log_entries"
PENDING_PARTS = RECORDS_ROOT / "pending_agent_updates"
WORK_LOG = RECORDS_ROOT / "WORK_LOG.md"
PENDING_LOG = RECORDS_ROOT / "PENDING_AGENT_UPDATES.md"


def read_fragments(directory: Path) -> list[str]:
    """按文件名稳定读取 Markdown 片段；read Markdown fragments in stable filename order."""

    if not directory.exists():
        return []
    return [path.read_text(encoding="utf-8").strip() for path in sorted(directory.glob("*.md"))]


def write_document(path: Path, header: str, fragments: list[str], empty: str) -> None:
    """原子重建一份追踪文档；atomically rebuild one tracking document."""

    body = "\n\n---\n\n".join(fragments) if fragments else empty
    content = f"{header.rstrip()}\n\n{body.rstrip()}\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def main() -> None:
    """同步工作日志和待录入草稿；synchronize the work log and pending-update draft."""

    work_header = """# 工作日志 / Work Log

> 本文件由 `tools/sync_tracking_docs.py` 从 `records/log_entries/*.md` 自动重建。
> 逻辑工作使用不可变条目保存；追踪文档自身的同步不另记日志。"""
    pending_header = """# 待录入 `_agent` 内容草稿簿 / Pending `_agent` Update Drafts

> 本文件由 `tools/sync_tracking_docs.py` 从 `records/pending_agent_updates/*.md`
> 自动重建。除非用户明确要求草拟或展示，否则不得写入 `_agent/`。

## 强制规则 / Mandatory rules

- 候选内容默认为 `draft`，必须注明目标文档、来源、证据和待确认项。
- 用户明确要求后才整理成逐文档可审核正文。
- 只有用户明确回复“确认录入”或“同意录入”后才可写入 `_agent/`。
- 本会话当前写入边界仍限制为 `final_v0/`。"""
    write_document(WORK_LOG, work_header, read_fragments(LOG_PARTS), "暂无日志条目。")
    write_document(PENDING_LOG, pending_header, read_fragments(PENDING_PARTS), "## 当前候选\n\n暂无。")


if __name__ == "__main__":
    main()

