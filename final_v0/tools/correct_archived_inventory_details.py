#!/usr/bin/env python3
"""Apply four evidence-backed corrections to the archived-code inventory.

中文：根据第二次逐字节/保存输出复核，精确补充旧SVM Notebook编译问题、Esther列名
不匹配、FilteredWalkTest采样率异常和16July SpO₂越界。每个旧文本必须唯一命中，
工具只修改 ``final_v0/records/ARCHIVED_CODE_IO_INVENTORY.md``。

English: Apply four exact, evidence-backed clarifications covering the old SVM
Notebook compile issue, Esther column mismatch, FilteredWalkTest sampling anomaly,
and 16July SpO2 range. Every old string must match exactly once; only the final_v0
archived inventory is modified.
"""

from __future__ import annotations

from pathlib import Path


FINAL_ROOT = Path(__file__).resolve().parents[1]
TARGET = FINAL_ROOT / "records" / "ARCHIVED_CODE_IO_INVENTORY.md"
REPLACEMENTS = (
    (
        "对应早期552/791样本；被根`svm2_dataset_train.ipynb/.py`扩展；保存类别缺失警告",
        "对应早期552/791样本；被根`svm2_dataset_train.ipynb/.py`扩展；首cell future-import位置错误，保存大量类别缺失/recall警告",
    ),
    (
        "不可移植；workspace输出无法唯一关联；SpO2未校准",
        "不可移植；脚本要求pleth1/2而本地镜像只有Ir1/Red1，会全部跳过；未找到对应产物；SpO2未校准",
    ),
    (
        "历史探索，无固定文件输出",
        "历史探索，无固定文件输出；保存125Hz只取首个8ms间隔，数据含大量零间隔，采样率估计不可靠",
    ),
    (
        "summary 12行但PNG含多轮陈旧产物；未放Archive但已被`funcs.py/ppg.py`替代",
        "summary 12行但PNG含多轮陈旧产物；SpO2含107.294与78.727等越界/异常值；未放Archive但已被`funcs.py/ppg.py`替代",
    ),
)


def main() -> None:
    """Validate and atomically apply corrections / 校验并原子应用更正。"""

    text = TARGET.read_text(encoding="utf-8")
    corrected = text
    for old, new in REPLACEMENTS:
        count = corrected.count(old)
        if count != 1:
            raise SystemExit(f"Expected one match for {old!r}, found {count}.")
        corrected = corrected.replace(old, new, 1)

    # 中文：同目录临时文件防止审计表出现部分写入。
    # English: Use a sibling temporary file so the audit table cannot be partially written.
    temp_path = TARGET.with_suffix(".md.tmp")
    temp_path.write_text(corrected, encoding="utf-8")
    temp_path.replace(TARGET)
    print(f"Applied {len(REPLACEMENTS)} archived-inventory clarifications.")


if __name__ == "__main__":
    main()

