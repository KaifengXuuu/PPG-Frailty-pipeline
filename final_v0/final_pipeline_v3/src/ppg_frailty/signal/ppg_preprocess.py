"""规范 PPG 预处理门面 / Canonical PPG-preprocessing facade.

中文：只重导出已测试实现，不复制滤波或缺口修复算法。
English: This exact-path facade re-exports the tested implementation only.
"""

from .preprocess import (
    ABLATION_PPG_FILTER_PROFILE_ID,
    REFERENCE_PPG_FILTER_PROFILE_ID,
    InputQC,
    PpgFilterProfile,
    build_signal_views,
    design_ppg_sos,
    get_ppg_filter_profile,
    inspect_and_repair,
    preprocess_ppg_pair,
    validate_timestamp_grid,
)

__all__ = [
    "ABLATION_PPG_FILTER_PROFILE_ID",
    "REFERENCE_PPG_FILTER_PROFILE_ID",
    "InputQC",
    "PpgFilterProfile",
    "build_signal_views",
    "design_ppg_sos",
    "get_ppg_filter_profile",
    "inspect_and_repair",
    "preprocess_ppg_pair",
    "validate_timestamp_grid",
]
