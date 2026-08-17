"""唯一窗口计划的规范路径 / Canonical path for the sole window planner.

中文：直接重导出 data authority，禁止产生第二套 WindowPlan。
English: Re-export the data authority directly; no second planner is defined.
"""

from ..data.windows import ShortRecordError, WindowPlan, WindowSlice, extract_window

__all__ = ["ShortRecordError", "WindowPlan", "WindowSlice", "extract_window"]
