"""V2 decision contract helpers / V2 决策合同辅助函数。

English: This module centralises small, dependency-free identities that must be
shared by configuration, runners, reports, and tests. It performs no training and
never starts an ablation automatically.

中文：本模块集中定义配置、runner、报告和测试共同使用的V2身份；它不执行训练，
也绝不会自动启动消融实验。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


QUALITY_MODES = ("off", "diagnostics_only", "route")
EPOCH_PROFILES: Mapping[str, int | None] = {
    "default_10": 10,
    "ablation_7": 7,
    "ablation_15": 15,
    "smoke": None,
}
BALANCE_LINES = {
    "line_a_equal_files": ("equal_files", "equal_files_no_role_layer"),
    "line_b_equal_role_families": (
        "equal_role_families",
        "equal_role_families",
    ),
}


@dataclass(frozen=True)
class ResolvedBalanceLine:
    """Resolved matched training/aggregation policy / 匹配的训练和聚合策略。"""

    line_id: str
    training_balance: str
    aggregation: str


def resolve_balance_line(
    line_id: str,
    *,
    training_balance: str,
    aggregation: str,
) -> ResolvedBalanceLine:
    """Reject mismatched Line A/B declarations / 拒绝A/B声明错配。"""

    if line_id not in BALANCE_LINES:
        raise ValueError(f"unknown balance line: {line_id}")
    expected_training, expected_aggregation = BALANCE_LINES[line_id]
    if (training_balance, aggregation) != (
        expected_training,
        expected_aggregation,
    ):
        raise ValueError(
            "balance line mismatch: "
            f"{line_id} requires training={expected_training}, "
            f"aggregation={expected_aggregation}"
        )
    return ResolvedBalanceLine(line_id, training_balance, aggregation)


def validate_quality_mode(mode: str, *, supervised_route_ready: bool) -> str:
    """Keep route disabled before supervision / 监督完成前禁用route。"""

    if mode not in QUALITY_MODES:
        raise ValueError(f"unknown quality mode: {mode}")
    if mode == "route":
        raise ValueError(
            "quality route is disabled until a frozen supervised artifact ID/hash exists"
        )
    return mode


__all__ = [
    "BALANCE_LINES",
    "EPOCH_PROFILES",
    "QUALITY_MODES",
    "ResolvedBalanceLine",
    "resolve_balance_line",
    "validate_quality_mode",
]
