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
BALANCE_LINES = ("line_a_equal_files", "line_b_equal_role_families")
TRAINING_BALANCES = ("equal_files", "equal_role_families")
AGGREGATION_BALANCES = ("equal_files_no_role_layer", "equal_role_families")


@dataclass(frozen=True)
class ResolvedBalanceLine:
    """Compatibility view of independently selected balance strategies."""

    line_id: str
    training_balance: str
    aggregation: str


def resolve_balance_line(
    line_id: str,
    *,
    training_balance: str,
    aggregation: str,
) -> ResolvedBalanceLine:
    """Validate, but never couple, training and reporting balance choices.

    ``line_id`` is retained as a display/profile label for old callers.  The
    executable training sampler and OOF aggregation are independent modules;
    neither is an authorization condition for the other.
    """

    if line_id not in BALANCE_LINES:
        raise ValueError(f"unknown balance line: {line_id}")
    if training_balance not in TRAINING_BALANCES:
        raise ValueError(f"unknown training balance: {training_balance}")
    if aggregation not in AGGREGATION_BALANCES:
        raise ValueError(f"unknown aggregation balance: {aggregation}")
    return ResolvedBalanceLine(line_id, training_balance, aggregation)


def validate_quality_mode(mode: str) -> str:
    """Validate the directly selectable quality module."""

    if mode not in QUALITY_MODES:
        raise ValueError(f"unknown quality mode: {mode}")
    return mode


__all__ = [
    "BALANCE_LINES",
    "TRAINING_BALANCES",
    "AGGREGATION_BALANCES",
    "EPOCH_PROFILES",
    "QUALITY_MODES",
    "ResolvedBalanceLine",
    "resolve_balance_line",
    "validate_quality_mode",
]
