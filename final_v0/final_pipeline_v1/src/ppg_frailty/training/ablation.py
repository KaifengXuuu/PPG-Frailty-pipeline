"""One-factor ablation and paired-comparison API / 单因素消融与配对比较 API。"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class AblationCase:
    """One named dotted-path change / 单个具名点路径改动。"""

    case_id: str
    parameter_path: str
    value: Any
    rationale: str


@dataclass(frozen=True)
class PairedComparison:
    """Paired per-subject deltas / 逐 subject 配对差值。"""

    subject_ids: tuple[str, ...]
    deltas: tuple[float, ...]
    mean_delta: float
    median_delta: float


def _set_dotted(config: dict[str, Any], path: str, value: Any) -> None:
    """Set exactly one nested field / 设置且仅设置一个嵌套字段。"""

    fields = path.split(".")
    if not fields or any(not field for field in fields):
        raise ValueError("parameter_path must be a non-empty dotted path")
    cursor = config
    for field in fields[:-1]:
        if field not in cursor or not isinstance(cursor[field], dict):
            raise KeyError(f"unknown nested configuration path: {path}")
        cursor = cursor[field]
    if fields[-1] not in cursor:
        raise KeyError(f"unknown configuration field: {path}")
    if cursor[fields[-1]] == value:
        raise ValueError(f"ablation does not change the baseline value: {path}")
    cursor[fields[-1]] = value


def run_ablation_matrix(
    base_config: Mapping[str, Any],
    cases: list[AblationCase] | tuple[AblationCase, ...],
    runner: Callable[[dict[str, Any], str], Any],
) -> dict[str, Any]:
    """Run baseline plus strict one-factor cases using a caller-supplied runner.

    使用调用方 runner 执行基线和严格单因素案例。runner 可以直接连接 CLI 的指定
    模块量化测试；本函数不改变 fold、seed 或数据成员关系。
    """

    frozen_base = copy.deepcopy(dict(base_config))
    results = {"baseline": runner(copy.deepcopy(frozen_base), "baseline")}
    seen = {"baseline"}
    for case in cases:
        if not case.case_id or case.case_id in seen:
            raise ValueError("ablation case ids must be non-empty and unique")
        seen.add(case.case_id)
        candidate = copy.deepcopy(frozen_base)
        _set_dotted(candidate, case.parameter_path, case.value)
        results[case.case_id] = runner(candidate, case.case_id)
    return results


def paired_subject_deltas(
    reference: Mapping[str, float], candidate: Mapping[str, float]
) -> PairedComparison:
    """Compare identical subject keys only / 只比较完全相同的 subject 键。"""

    if set(reference) != set(candidate) or not reference:
        raise ValueError("paired comparison requires identical non-empty subject sets")
    subjects = tuple(sorted(reference))
    deltas = np.asarray([candidate[key] - reference[key] for key in subjects], dtype=np.float64)
    if not np.isfinite(deltas).all():
        raise ValueError("paired deltas must be finite")
    return PairedComparison(
        subject_ids=subjects,
        deltas=tuple(float(value) for value in deltas),
        mean_delta=float(deltas.mean()),
        median_delta=float(np.median(deltas)),
    )
