"""规范 PPG 预处理门面 / Canonical PPG-preprocessing facade.

中文：只重导出已测试实现，不复制滤波或缺口修复算法。
English: This exact-path facade re-exports the tested implementation only.
"""

from .preprocess import (
    InputQC,
    build_signal_views,
    design_ppg_sos,
    inspect_and_repair,
    preprocess_ppg_pair,
    validate_timestamp_grid,
)

__all__ = [
    "InputQC",
    "build_signal_views",
    "design_ppg_sos",
    "inspect_and_repair",
    "preprocess_ppg_pair",
    "validate_timestamp_grid",
]
