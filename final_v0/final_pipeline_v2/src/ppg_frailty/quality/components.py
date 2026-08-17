"""SQI component 类型与表格化 / SQI component types and tabulation."""

from __future__ import annotations

from typing import Any

from ..contracts import QualityComponent, QualityEndpoint, QualityResult, QualityState


def component_rows(result: QualityResult) -> list[dict[str, Any]]:
    """按名称输出可审计 component / Return auditable components in name order."""

    return [
        {
            "name": name,
            "raw_value": component.raw_value,
            "normalized_value": component.normalized_value,
            "state": component.state.value,
            "reason": component.reason,
        }
        for name, component in sorted(result.components.items())
    ]


__all__ = ["QualityComponent", "QualityEndpoint", "QualityResult", "QualityState", "component_rows"]
