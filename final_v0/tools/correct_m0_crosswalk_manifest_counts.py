#!/usr/bin/env python3
"""Correct M0 crosswalk file counts from verified output manifests.

中文：把 M0 crosswalk 汇总表中的人工估算数量替换为输出 ``*.summary.json`` 的
实际 ``file_count``。每个旧文本必须唯一命中，否则在写入前停止；脚本只修改
``final_v0/records/M0_CODE_OUTPUT_CROSSWALK.md``。

English: Replace manually estimated M0 crosswalk counts with the verified
``file_count`` values in output summary manifests. Every old string must match once,
or the script stops before writing. Only the final_v0 crosswalk is modified.
"""

from __future__ import annotations

from pathlib import Path


FINAL_ROOT = Path(__file__).resolve().parents[1]
TARGET = FINAL_ROOT / "records" / "M0_CODE_OUTPUT_CROSSWALK.md"
REPLACEMENTS = (
    (
        "`results/`：29 文件、结构与 v7 写出项相符",
        "`results/`：5 文件、结构与 v7 写出项相符",
    ),
    (
        "目录存在，49 文件；文本/二进制角色与代码相符",
        "目录存在，16 文件；文本/二进制角色与代码相符",
    ),
    (
        "`results_v7_4/` 63 文件为显式参数运行；`results_v7_3/` 另有旧/不同批次 30 文件",
        "`results_v7_4/` 55 文件为显式参数运行；`results_v7_3/` 另有旧/不同批次 33 文件",
    ),
    (
        "52 文件；三组 summary/audit/NPZ 均存在",
        "30 文件；三组 summary/audit/NPZ 均存在",
    ),
    (
        "`results_hybrid_denoiser_raw_imu/` 与 `_baseline/` 各 9–10 文件；契约相符",
        "`results_hybrid_denoiser_raw_imu/` 与 `_baseline/` 各 8 文件；契约相符",
    ),
    (
        "7 文件；best val/schema 与后两组不同",
        "6 文件；best val/schema 与后两组不同",
    ),
)


def main() -> None:
    """Validate unique matches and replace atomically / 校验唯一命中并原子替换。"""

    text = TARGET.read_text(encoding="utf-8")
    corrected = text
    for old, new in REPLACEMENTS:
        count = corrected.count(old)
        if count != 1:
            raise SystemExit(f"Expected one match for {old!r}, found {count}.")
        corrected = corrected.replace(old, new, 1)

    # 中文：同目录临时文件保证替换是完整的，不留下半写 crosswalk。
    # English: A sibling temporary file prevents a partially written crosswalk.
    temp_path = TARGET.with_suffix(".md.tmp")
    temp_path.write_text(corrected, encoding="utf-8")
    temp_path.replace(TARGET)
    print(f"Corrected {len(REPLACEMENTS)} manifest-count statements in {TARGET.name}.")


if __name__ == "__main__":
    main()

