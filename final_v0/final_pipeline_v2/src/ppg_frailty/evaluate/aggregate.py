"""严格层级聚合的 canonical facade / Canonical strict-aggregation facade.

中文：所有 identity 分组、drop/no-result coverage 与层级平均只由
training.aggregation 实现；本文件不预过滤 retained 行，也不复制公式。
English: Identity grouping, rejected-row coverage and hierarchy arithmetic have one
authority in training.aggregation; this facade never pre-filters source rows.
"""

from __future__ import annotations

from typing import Iterable

from ..training.aggregation import (
    CANONICAL_BALANCE_LINE,
    CoverageSummary,
    ExperimentIdentity,
    HierarchyAggregation,
    aggregate_hierarchy,
    experiment_identity,
)
from ..training.oof import OofPredictionRow


# English: The legacy facade type name remains an exact alias, not a second container.
# 中文：保留旧门面类型名，但它是精确别名而不是第二个数据容器。
StrictAggregationResult = HierarchyAggregation


def aggregate_hierarchy_strict(
    rows: Iterable[OofPredictionRow],
    *,
    balance_line: str = CANONICAL_BALANCE_LINE,
    quality_weighted: bool = False,
) -> HierarchyAggregation:
    """直接调用唯一聚合器 / Delegate directly to the sole aggregator."""

    return aggregate_hierarchy(
        tuple(rows),
        balance_line=balance_line,
        quality_weighted=quality_weighted,
    )


__all__ = [
    "CoverageSummary",
    "ExperimentIdentity",
    "HierarchyAggregation",
    "StrictAggregationResult",
    "aggregate_hierarchy",
    "aggregate_hierarchy_strict",
    "experiment_identity",
]
