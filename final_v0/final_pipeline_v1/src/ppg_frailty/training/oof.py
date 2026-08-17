"""Strict out-of-fold prediction rows and Parquet writer.

严格 OOF 预测行与 Parquet 写入器。
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class OofPredictionRow:
    """One auditable prediction before or after hierarchy aggregation.

    层级聚合前或后的单条可审计预测。
    """

    participant_id: str
    file_id: str
    role: str
    label: int
    probabilities: tuple[float, ...]
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
    quality_score: float
    retained: bool
    level: str = "window"
    window_id: str | None = None
    member_index: int | None = None
    class_order: tuple[int, ...] = ()
    code_commit: str = ""
    data_schema_id: str = ""
    feature_schema_id: str = ""
    model_version: str = ""
    aggregation_rule: str = ""
    environment_hash: str = ""
    manifest_version: str = ""
    fold_registry_version: str = ""
    artifact_reducer_name: str = ""
    artifact_reducer_version: str = ""
    route_status: str = ""
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        base_text = (
            self.participant_id,
            self.file_id,
            self.role,
            self.config_hash,
            self.manifest_hash,
            self.fold_hash,
            self.preprocessing_hash,
            self.feature_hash,
            self.model_hash,
            self.representation_mode,
            self.signal_route,
        )
        if any(not str(value).strip() for value in base_text):
            raise ValueError("OOF identity and provenance fields must be non-empty")
        probability = np.asarray(self.probabilities, dtype=np.float64)
        if probability.ndim != 1:
            raise ValueError("probabilities must be one-dimensional")
        if self.retained or probability.size:
            if probability.size < 2:
                raise ValueError("retained probabilities must contain at least two classes")
            if not np.isfinite(probability).all() or np.any(probability < 0.0):
                raise ValueError("probabilities must be finite and non-negative")
            if not np.isclose(probability.sum(), 1.0, atol=1e-6):
                raise ValueError("probabilities must sum to one")
        if self.class_order and (
            len(self.class_order) != probability.size
            or len(self.class_order) != len(set(self.class_order))
        ):
            raise ValueError("class_order must be unique and match the probability vector")
        if self.class_order and self.label not in self.class_order:
            raise ValueError("OOF label must occur in class_order")
        if self.level not in {"window", "file", "role", "participant"}:
            raise ValueError("invalid aggregation level")
        if self.level == "window" and not self.window_id:
            raise ValueError("window-level OOF rows require window_id")
        if self.member_index is not None and self.member_index < 0:
            raise ValueError("member_index must be non-negative")
        if not self.retained and not str(self.rejection_reason or "").strip():
            raise ValueError("dropped/no-result OOF rows require rejection_reason")
        if not np.isfinite(self.quality_score) or not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("quality_score must be finite in [0,1]")


def validate_unique_subject_oof(rows: Iterable[OofPredictionRow]) -> None:
    """Ensure each subject appears in one fold per repeat/config.

    确保每个 subject 在每个 repeat/config 中只属于一个 fold。
    """

    membership: dict[tuple[object, ...], int] = {}
    prediction_keys: set[tuple[object, ...]] = set()
    for row in rows:
        key = (
            row.repeat,
            row.seed,
            row.config_hash,
            row.manifest_hash,
            row.fold_hash,
            row.preprocessing_hash,
            row.feature_hash,
            row.model_hash,
            row.representation_mode,
            row.signal_route,
            row.participant_id,
        )
        previous = membership.setdefault(key, row.fold)
        if previous != row.fold:
            raise ValueError("one participant appears in multiple OOF folds for the same repeat/config")
        prediction_key = (
            row.repeat,
            row.fold,
            row.seed,
            row.config_hash,
            row.manifest_hash,
            row.fold_hash,
            row.preprocessing_hash,
            row.feature_hash,
            row.model_hash,
            row.representation_mode,
            row.signal_route,
            row.participant_id,
            row.file_id,
            row.role,
            row.level,
            row.window_id,
            row.member_index,
        )
        if prediction_key in prediction_keys:
            raise ValueError("duplicate OOF prediction row")
        prediction_keys.add(prediction_key)


FORMAL_TRACE_FIELDS = (
    "config_hash",
    "manifest_hash",
    "fold_hash",
    "preprocessing_hash",
    "feature_hash",
    "model_hash",
    "representation_mode",
    "signal_route",
    "code_commit",
    "data_schema_id",
    "feature_schema_id",
    "model_version",
    "aggregation_rule",
    "environment_hash",
    "manifest_version",
    "fold_registry_version",
    "artifact_reducer_name",
    "artifact_reducer_version",
    "route_status",
)


def validate_expected_oof_roster(
    rows: Iterable[OofPredictionRow],
    expected_heldout_roster: Mapping[tuple[int, int, int], Iterable[str]],
    *,
    expected_config_hashes: Iterable[str],
    expected_level: str = "participant",
    expected_member_count: int = 1,
    require_trace: bool = True,
) -> None:
    """Validate the complete formal frozen-OOF Cartesian product.

    English: For every (repeat, fold, seed), configuration and expected held-out
    subject, exactly one participant prediction must exist for a single model,
    or exactly member indices 0..N-1 for an ensemble. Rejected subjects remain
    rows with retained=False, so completeness and coverage are independent.

    中文：对每个 (repeat, fold, seed)、配置与冻结 held-out subject，单模型必须
    恰好有一行；集成模型必须恰好具有 0..N-1 的成员编号。被拒绝 subject 仍以
    retained=False 行存在，因此完整性校验与 coverage 统计彼此独立。
    """

    frozen = tuple(rows)
    roster = {
        tuple(int(value) for value in key): tuple(sorted(set(str(item) for item in values)))
        for key, values in expected_heldout_roster.items()
    }
    configurations = tuple(sorted(set(str(value) for value in expected_config_hashes)))
    if not frozen or not roster or not configurations:
        raise ValueError("formal OOF validation requires rows, roster and configurations")
    if any(len(key) != 3 or not values for key, values in roster.items()):
        raise ValueError("roster keys must be (repeat,fold,seed) with non-empty subjects")
    if expected_member_count <= 0:
        raise ValueError("expected_member_count must be positive")
    validate_unique_subject_oof(frozen)

    selected = tuple(row for row in frozen if row.level == expected_level)
    if not selected:
        raise ValueError(f"OOF table has no {expected_level!r} rows")
    expected_combinations = {
        (repeat, fold, seed, config_hash)
        for repeat, fold, seed in roster
        for config_hash in configurations
    }
    observed_combinations = {
        (row.repeat, row.fold, row.seed, row.config_hash) for row in selected
    }
    if observed_combinations != expected_combinations:
        missing = sorted(expected_combinations - observed_combinations)
        extra = sorted(observed_combinations - expected_combinations)
        raise ValueError(f"OOF repeat/fold/seed/config mismatch; missing={missing}, extra={extra}")

    # English: The frozen roster itself must assign a subject once per repeat/seed.
    # 中文：冻结 roster 自身也必须保证每个 subject 在每个 repeat/seed 仅出现一次。
    roster_membership: dict[tuple[int, int, str], int] = {}
    for (repeat, fold, seed), participants in roster.items():
        for participant in participants:
            key = (repeat, seed, participant)
            previous = roster_membership.setdefault(key, fold)
            if previous != fold:
                raise ValueError("expected roster assigns one subject to multiple folds")

    for row in selected:
        roster_key = (row.repeat, row.fold, row.seed)
        if row.participant_id not in roster[roster_key]:
            raise ValueError("OOF table contains a subject outside the frozen held-out roster")
        if require_trace:
            missing_trace = [
                name
                for name in FORMAL_TRACE_FIELDS
                if getattr(row, name) is None or not str(getattr(row, name)).strip()
            ]
            if missing_trace:
                raise ValueError(f"OOF row is missing formal trace fields: {missing_trace}")
            if not row.class_order or len(row.class_order) != len(row.probabilities):
                if row.retained:
                    raise ValueError("retained formal OOF rows require probability class_order")

    grouped: dict[tuple[int, int, int, str, str], list[OofPredictionRow]] = {}
    for row in selected:
        key = (row.repeat, row.fold, row.seed, row.config_hash, row.participant_id)
        grouped.setdefault(key, []).append(row)
    for repeat, fold, seed in roster:
        for config_hash in configurations:
            for participant in roster[(repeat, fold, seed)]:
                key = (repeat, fold, seed, config_hash, participant)
                values = grouped.get(key, [])
                if len(values) != expected_member_count:
                    raise ValueError(
                        "OOF subject/member completeness failed for "
                        f"{key}: expected {expected_member_count}, observed {len(values)}"
                    )
                member_indices = {row.member_index for row in values}
                expected_indices = (
                    {None}
                    if expected_member_count == 1
                    else set(range(expected_member_count))
                )
                if member_indices != expected_indices:
                    raise ValueError(
                        f"OOF ensemble member indices are incomplete for {key}: {member_indices}"
                    )


# Descriptive formal alias / 描述性的正式协议别名。
validate_formal_oof = validate_expected_oof_roster


class OofWriter:
    """Fail-closed optional-pyarrow Parquet writer / 缺少 pyarrow 时关闭失败的写入器。"""

    def write(self, rows: Iterable[OofPredictionRow], path: str | Path) -> Path:
        """Atomically write validated rows / 原子写入已校验行。"""

        frozen = tuple(rows)
        if not frozen:
            raise ValueError("cannot write an empty OOF table")
        validate_unique_subject_oof(frozen)
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "OOF Parquet output requires optional dependency pyarrow; CSV fallback is forbidden"
            ) from exc
        target = Path(path)
        if target.suffix.lower() != ".parquet":
            raise ValueError("OOF output path must end with .parquet")
        target.parent.mkdir(parents=True, exist_ok=True)
        records = [asdict(row) for row in frozen]
        table = pa.Table.from_pylist(records)
        temporary = target.with_name(f".{target.name}.tmp")
        try:
            pq.write_table(table, temporary, compression="zstd")
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        return target


def write_oof_parquet(rows: Iterable[OofPredictionRow], path: str | Path) -> Path:
    """Functional writer facade / 函数式写入门面。"""

    return OofWriter().write(rows, path)
