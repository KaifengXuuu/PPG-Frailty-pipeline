#!/usr/bin/env python3
"""Apply one audited filename-reference correction inside final_v0.

中文：把三份 M0 文档/校验器中的误写 ``dash_denoiser_utils.py`` 精确替换为
真实根文件名 ``ppg_denoiser_dash_utils.py``。每个目标必须且只能命中一次，否则停止，
从而避免无范围的文本重写。

English: Correct the misspelled ``dash_denoiser_utils.py`` reference to the real
root filename ``ppg_denoiser_dash_utils.py`` in exactly three audited targets.
Each target must match exactly once; otherwise the script stops before rewriting.
"""

from __future__ import annotations

from pathlib import Path


FINAL_ROOT = Path(__file__).resolve().parents[1]
OLD = "dash_denoiser_utils.py"
NEW = "ppg_denoiser_dash_utils.py"
TARGETS = (
    FINAL_ROOT / "records" / "M0_CODE_OUTPUT_CROSSWALK.md",
    FINAL_ROOT / "algorithm_diagrams" / "m0" / "05_SCRIPT_ALGORITHM_ATLAS.md",
    FINAL_ROOT / "tools" / "verify_algorithm_diagrams.py",
)


def main() -> None:
    """Validate exact matches, then replace atomically / 校验唯一命中后原子替换。"""

    prepared: list[tuple[Path, str]] = []
    for path in TARGETS:
        text = path.read_text(encoding="utf-8")
        count = text.count(OLD)
        if count != 1:
            raise SystemExit(f"Expected exactly one {OLD!r} in {path}, found {count}.")
        prepared.append((path, text.replace(OLD, NEW, 1)))

    for path, corrected in prepared:
        # 中文：临时文件与目标同目录，replace 不会跨文件系统并能避免半写文件。
        # English: Use a sibling temporary file for atomic replacement and no partial writes.
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(corrected, encoding="utf-8")
        temp_path.replace(path)
        print(f"Corrected: {path.relative_to(FINAL_ROOT)}")


if __name__ == "__main__":
    main()

