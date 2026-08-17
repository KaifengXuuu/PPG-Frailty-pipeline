## 2026-08-02 — 内容感知文件树索引器

- 原因：旧索引虽有完整树、字节数和SHA，但对部分报告/manifest的说明过于通用。
- 写入：`tools/update_final_v0_index_detailed.py`；Markdown提取标题/实质段落，JSON/JSONL提取范围/记录数，Python提取模块职责/入口。
- 边界：只读 `final_v0`，只重写 `FINAL_V0_TREE.md`；索引更新不递归生成日志。
- 目标：保证 `final_v0` 每个永久文件都在一张树中拥有文件名、内容说明、字节数和SHA-256。

