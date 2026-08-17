"""V2 balance-line aggregation with window→file as the invariant first step.

V2 平衡线路聚合；window→file 始终是不可变的第一步。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import groupby
import re
from typing import Callable, Iterable

import numpy as np

from .oof import OofPredictionRow


LINE_A_EQUAL_FILES = "line_a_equal_files"
LINE_B_EQUAL_ROLE_FAMILIES = "line_b_equal_role_families"
BALANCE_LINES = (LINE_A_EQUAL_FILES, LINE_B_EQUAL_ROLE_FAMILIES)
_ROLE_PATTERN = re.compile(r"^(?P<family>[BRSW])(?:[_-]?[0-9]+)?$", re.IGNORECASE)


def canonical_role_family(role: str) -> str:
    """Return the only physiological role families B/R/S/W, or fail closed.

    R1--R4 and analogous numeric suffixes are file identifiers, not roles. The
    function intentionally does not guess aliases such as relax so raw naming
    changes cannot silently alter Line B weights.
    """

    match = _ROLE_PATTERN.fullmatch(str(role).strip())
    if match is None:
        raise ValueError(
            f"unsupported role; expected B/R/S/W with optional file suffix: {role!r}"
        )
    return str(match.group("family")).upper()


def aggregation_rule_for_training_balance(training_balance: str) -> str:
    """Map the training-side balance identity to its required aggregation line."""

    mapping = {
        "equal_files": LINE_A_EQUAL_FILES,
        "equal_role_families": LINE_B_EQUAL_ROLE_FAMILIES,
    }
    try:
        return mapping[str(training_balance)]
    except KeyError as exc:
        raise ValueError(f"unsupported training_balance: {training_balance!r}") from exc


@dataclass(frozen=True)
class HierarchyAggregation:
    """Canonical retained predictions plus explicit coverage accounting.

    English: source_rows deliberately keeps rejected/no-result rows so that
    downstream reports cannot silently turn rejection into a smaller dataset.
    中文：source_rows 会刻意保留被拒绝/无结果行，避免下游报告把拒绝样本静默
    变成一个更小的数据集。
    """

    file_rows: tuple[OofPredictionRow, ...]
    role_rows: tuple[OofPredictionRow, ...]
    participant_rows: tuple[OofPredictionRow, ...]
    source_rows: tuple[OofPredictionRow, ...] = ()
    dropped_rows: tuple[OofPredictionRow, ...] = ()
    coverage: tuple["CoverageSummary", ...] = ()
    balance_line: str = LINE_A_EQUAL_FILES


@dataclass(frozen=True)
class ExperimentIdentity:
    """Full experiment identity required at every aggregation level.

    每个聚合层级必须携带的完整实验身份。缺少任意哈希或路线字段时，来自不同
    实验的概率可能被错误平均，因此这些字段必须共同参与分组键。
    """

    repeat: int
    fold: int
    seed: int
    config_hash: str
    manifest_hash: str
    fold_hash: str
    preprocessing_hash: str
    feature_hash: str
    model_hash: str
    representation_mode: str
    signal_route: str


@dataclass(frozen=True)
class CoverageSummary:
    """Coverage for one experiment and one hierarchy level / 单实验单层级覆盖率。"""

    experiment: ExperimentIdentity
    level: str
    n_total: int
    n_retained: int
    n_dropped: int
    coverage_rate: float


def experiment_identity(row: OofPredictionRow) -> ExperimentIdentity:
    """Return the fail-closed identity used by every grouping operation.

    返回所有分组操作共同使用的关闭失败身份键。
    """

    identity = ExperimentIdentity(
        repeat=row.repeat,
        fold=row.fold,
        seed=row.seed,
        config_hash=row.config_hash,
        manifest_hash=row.manifest_hash,
        fold_hash=row.fold_hash,
        preprocessing_hash=row.preprocessing_hash,
        feature_hash=row.feature_hash,
        model_hash=row.model_hash,
        representation_mode=row.representation_mode,
        signal_route=row.signal_route,
    )
    textual = (
        identity.config_hash,
        identity.manifest_hash,
        identity.fold_hash,
        identity.preprocessing_hash,
        identity.feature_hash,
        identity.model_hash,
        identity.representation_mode,
        identity.signal_route,
    )
    if any(not str(value).strip() for value in textual):
        raise ValueError("aggregation identity fields must be non-empty")
    return identity


def _identity_key(row: OofPredictionRow) -> tuple[object, ...]:
    """Tuple form used for deterministic sorting / 用于确定性排序的元组形式。"""

    identity = experiment_identity(row)
    return tuple(identity.__dict__.values())


def _coverage_summaries(
    rows: tuple[OofPredictionRow, ...], *, balance_line: str
) -> tuple[CoverageSummary, ...]:
    """Count retained units using the declared Line A/Line B hierarchy."""

    output: list[CoverageSummary] = []
    ordered = sorted(rows, key=_identity_key)
    for _, grouped in groupby(ordered, key=_identity_key):
        values = tuple(grouped)
        identity = experiment_identity(values[0])
        unit_keys: dict[str, list[tuple[object, ...]]] = {
            "source": [
                (
                    row.level,
                    row.participant_id,
                    row.file_id,
                    row.role,
                    row.window_id,
                    row.member_index,
                    index,
                )
                for index, row in enumerate(values)
            ],
            "file": [(row.participant_id, row.file_id) for row in values],
            "participant": [(row.participant_id,) for row in values],
        }
        retained_keys: dict[str, set[tuple[object, ...]]] = {
            "source": {
                unit_keys["source"][index]
                for index, row in enumerate(values)
                if row.retained
            },
            "file": {
                (row.participant_id, row.file_id) for row in values if row.retained
            },
            "participant": {(row.participant_id,) for row in values if row.retained},
        }
        if balance_line == LINE_B_EQUAL_ROLE_FAMILIES:
            unit_keys["role_family"] = [
                (row.participant_id, canonical_role_family(row.role)) for row in values
            ]
            retained_keys["role_family"] = {
                (row.participant_id, canonical_role_family(row.role))
                for row in values
                if row.retained
            }
        levels = (
            ("source", "file", "participant")
            if balance_line == LINE_A_EQUAL_FILES
            else ("source", "file", "role_family", "participant")
        )
        for level in levels:
            total_units = unit_keys[level]
            total = len(total_units) if level == "source" else len(set(total_units))
            retained = len(retained_keys[level])
            output.append(
                CoverageSummary(
                    experiment=identity,
                    level=level,
                    n_total=total,
                    n_retained=retained,
                    n_dropped=total - retained,
                    coverage_rate=float(retained / total) if total else 0.0,
                )
            )
    return tuple(output)


def _mean_probabilities(rows: list[OofPredictionRow], quality_weighted: bool) -> tuple[float, ...]:
    """Average probabilities with optional declared SQI weights.

    平均概率；仅在显式声明时使用 SQI 权重。
    """

    values = np.asarray([row.probabilities for row in rows], dtype=np.float64)
    if quality_weighted:
        weights = np.asarray([row.quality_score for row in rows], dtype=np.float64)
        if weights.sum() <= 0:
            raise ValueError("quality-weighted aggregation requires a positive total weight")
        result = np.average(values, axis=0, weights=weights)
    else:
        result = values.mean(axis=0)
    result /= result.sum()
    return tuple(float(value) for value in result)


def _group_aggregate(
    rows: Iterable[OofPredictionRow],
    key: Callable[[OofPredictionRow], tuple],
    *,
    level: str,
    quality_weighted: bool,
    file_id_factory: Callable[[list[OofPredictionRow]], str],
    role_factory: Callable[[list[OofPredictionRow]], str],
) -> tuple[OofPredictionRow, ...]:
    """Aggregate deterministic sorted groups / 聚合确定性排序后的分组。"""

    ordered = sorted(rows, key=key)
    output: list[OofPredictionRow] = []
    for _, grouped in groupby(ordered, key=key):
        values = list(grouped)
        if len({row.label for row in values}) != 1:
            raise ValueError("labels disagree within one aggregation unit")
        reference = values[0]
        output.append(
            replace(
                reference,
                probabilities=_mean_probabilities(values, quality_weighted),
                file_id=file_id_factory(values),
                role=role_factory(values),
                level=level,
                window_id=None,
                member_index=None,
                quality_score=float(np.mean([row.quality_score for row in values])),
            )
        )
    return tuple(output)


def aggregate_hierarchy(
    rows: Iterable[OofPredictionRow],
    *,
    balance_line: str = LINE_A_EQUAL_FILES,
    quality_weighted: bool = False,
) -> HierarchyAggregation:
    """Apply one explicit V2 balance line without automatic route selection.

    Line A is window→file→participant with equal retained files. Line B is
    window→file→canonical-role-family→participant with equal available families.
    Missing files/families are naturally renormalised by ordinary means.
    """

    source_rows = tuple(rows)
    if balance_line not in BALANCE_LINES:
        raise ValueError(f"unsupported aggregation balance_line: {balance_line!r}")
    if not source_rows:
        raise ValueError("no OOF rows are available for aggregation")
    if any(row.level not in {"window", "file"} for row in source_rows):
        raise ValueError("aggregate_hierarchy accepts only window- or file-level source rows")
    for row in source_rows:
        experiment_identity(row)
        canonical_role_family(row.role)
        if row.aggregation_rule and row.aggregation_rule != balance_line:
            raise ValueError(
                "OOF aggregation_rule metadata disagrees with the requested balance_line"
            )
    retained = [row for row in source_rows if row.retained]
    duplicate_keys = [
        (
            *_identity_key(row),
            row.participant_id,
            row.file_id,
            row.role,
            row.window_id,
            row.member_index,
        )
        for row in retained
        if row.level == "window"
    ]
    if len(duplicate_keys) != len(set(duplicate_keys)):
        raise ValueError("duplicate window prediction detected")
    windows = [row for row in retained if row.level == "window"]
    direct_files = [row for row in retained if row.level == "file"]
    generated_files: tuple[OofPredictionRow, ...] = ()
    if windows:
        generated_files = _group_aggregate(
            windows,
            lambda row: (
                *_identity_key(row),
                row.participant_id,
                row.file_id,
                canonical_role_family(row.role),
            ),
            level="file",
            quality_weighted=quality_weighted,
            file_id_factory=lambda values: values[0].file_id,
            role_factory=lambda values: canonical_role_family(values[0].role),
        )
    direct_files = [
        replace(row, role=canonical_role_family(row.role), aggregation_rule=balance_line)
        for row in direct_files
    ]
    generated_files = tuple(
        replace(row, aggregation_rule=balance_line) for row in generated_files
    )
    file_rows = tuple(sorted((*generated_files, *direct_files), key=lambda row: (
        *_identity_key(row), row.participant_id, row.role, row.file_id
    )))
    unique_file_keys = [
        (*_identity_key(row), row.participant_id, row.file_id)
        for row in file_rows
    ]
    if len(unique_file_keys) != len(set(unique_file_keys)):
        raise ValueError("a file was supplied both directly and through window rows")

    if balance_line == LINE_A_EQUAL_FILES:
        role_rows: tuple[OofPredictionRow, ...] = ()
        participant_inputs = file_rows
    else:
        role_rows = _group_aggregate(
            file_rows,
            lambda row: (
                *_identity_key(row),
                row.participant_id,
                canonical_role_family(row.role),
            ),
            level="role",
            quality_weighted=False,
            file_id_factory=lambda values: (
                f"role::{values[0].participant_id}::{canonical_role_family(values[0].role)}"
            ),
            role_factory=lambda values: canonical_role_family(values[0].role),
        )
        participant_inputs = role_rows
    participant_rows = _group_aggregate(
        participant_inputs,
        lambda row: (
            *_identity_key(row),
            row.participant_id,
        ),
        level="participant",
        quality_weighted=False,
        file_id_factory=lambda values: f"participant::{values[0].participant_id}",
        role_factory=lambda values: "participant",
    )
    participant_rows = tuple(
        replace(row, aggregation_rule=balance_line) for row in participant_rows
    )
    return HierarchyAggregation(
        file_rows=file_rows,
        role_rows=role_rows,
        participant_rows=participant_rows,
        source_rows=source_rows,
        dropped_rows=tuple(row for row in source_rows if not row.retained),
        coverage=_coverage_summaries(source_rows, balance_line=balance_line),
        balance_line=balance_line,
    )


# Descriptive internal alias / 描述性内部别名。
aggregate_oof_rows = aggregate_hierarchy
