"""V1 time-scale names must fail closed in V2 / V1 时间尺度名称在 V2 中关闭失败。"""

from __future__ import annotations

import pytest

from ppg_frailty.models.time_scale import build_physical_time_cases, create_time_scaled_model


def test_v1_physical_time_helpers_are_not_v2_evidence() -> None:
    with pytest.raises(RuntimeError, match="not V2 evidence"):
        build_physical_time_cases()
    with pytest.raises(RuntimeError, match="disabled in V2"):
        create_time_scaled_model()
