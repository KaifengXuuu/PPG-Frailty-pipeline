#!/usr/bin/env python3
"""Add one audited bilingual inline-comment block to the scan verifier.

中文：在 ``verify_scan_evidence.py`` 的路径常量前加入一组解释“只核对final_v0证据、
不重读项目源数据”的中英文行内注释。目标锚点必须唯一，且旧注释不得已存在。

English: Insert one bilingual inline-comment block before the scan verifier's path
constants, explaining that it validates final_v0 evidence without rereading project
source data. The anchor must be unique and the comment must not already exist.
"""

from __future__ import annotations

from pathlib import Path


FINAL_ROOT = Path(__file__).resolve().parents[1]
TARGET = FINAL_ROOT / "tools" / "verify_scan_evidence.py"
ANCHOR = "FINAL_ROOT = Path(__file__).resolve().parents[1]\n"
INSERT = (
    "# 中文：校验器只读取final_v0生成证据；路径固定在本目录，避免触碰只读源项目。\n"
    "# English: Validate generated final_v0 evidence only; anchored paths protect the read-only source project.\n"
)


def main() -> None:
    """Validate the anchor and atomically insert comments / 校验锚点并原子插入注释。"""

    text = TARGET.read_text(encoding="utf-8")
    if text.count(ANCHOR) != 1:
        raise SystemExit(f"Expected one anchor in {TARGET}.")
    if INSERT.strip() in text:
        raise SystemExit("Bilingual inline-comment block already exists.")
    corrected = text.replace(ANCHOR, INSERT + ANCHOR, 1)

    # 中文：使用同目录临时文件，保证注释更改不会留下半写Python源文件。
    # English: Use a sibling temporary file so the comment edit cannot leave partial Python source.
    temp_path = TARGET.with_suffix(".py.tmp")
    temp_path.write_text(corrected, encoding="utf-8")
    temp_path.replace(TARGET)
    print(f"Inserted bilingual inline comments into {TARGET.name}.")


if __name__ == "__main__":
    main()

