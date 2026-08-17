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
    split_seed: int
    training_seed: int | None
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
    prediction_kind: str = "single_model"
    member_training_seeds: tuple[int, ...] = ()
    ensemble_base_model_id: str = ""
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
    source_snapshot_hash: str = ""
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
        if self.split_seed < 0:
            raise ValueError("split_seed must be non-negative")
        if self.training_seed is not None and self.training_seed < 0:
            raise ValueError("training_seed must be null or non-negative")
        if self.prediction_kind not in {"single_model", "ensemble_member", "ensemble_average"}:
            raise ValueError("invalid prediction_kind")
        member_seeds = tuple(int(value) for value in self.member_training_seeds)
        if self.prediction_kind == "single_model":
            if self.training_seed is None or self.member_index is not None or member_seeds:
                raise ValueError("single_model requires one training_seed and no member semantics")
        elif self.prediction_kind == "ensemble_member":
            if self.training_seed is None or self.member_index is None:
                raise ValueError("ensemble_member requires member_index and its training_seed")
            if member_seeds or not str(self.ensemble_base_model_id).strip():
                raise ValueError("ensemble_member stores its own seed and the base-model identity")
        else:
            if self.training_seed is not None or self.member_index is not None:
                raise ValueError("ensemble_average cannot be represented as one training seed/member")
            if member_seeds != (50042, 60042, 70042, 80042, 90042):
                raise ValueError("ensemble_average requires the exact five V2 member seeds")
            if not str(self.ensemble_base_model_id).strip():
                raise ValueError("ensemble_average requires ensemble_base_model_id")
        object.__setattr__(self, "member_training_seeds", member_seeds)
        if not self.retained and not str(self.rejection_reason or "").strip():
            raise ValueError("dropped/no-result OOF rows require rejection_reason")
        if not self.retained and (probability.size or self.class_order):
            raise ValueError("dropped OOF rows must keep empty probabilities and class_order")
        if not np.isfinite(self.quality_score) or not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("quality_score must be finite in [0,1]")

    @property
    def seed(self) -> int:
        """Compatibility read alias; persisted schema uses explicit split_seed."""

        return self.split_seed


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
            row.prediction_kind,
            row.training_seed,
        )
        if prediction_key in prediction_keys:
            raise ValueError("duplicate OOF prediction row")
        prediction_keys.add(prediction_key)


def validate_role_level_oof(rows: Iterable[OofPredictionRow]) -> None:
    """Validate canonical role rows before Parquet/JSON persistence."""

    selected = tuple(row for row in rows if row.level == "role")
    if not selected:
        raise ValueError("canonical hierarchy requires role-level OOF rows")
    seen: set[tuple[object, ...]] = set()
    for row in selected:
        role = str(row.role).strip().upper()
        if role not in {"B", "R", "S", "W"}:
            raise ValueError("role-level OOF requires canonical B/R/S/W roles")
        if row.file_id != f"role::{row.participant_id}::{role}":
            raise ValueError("role-level OOF synthetic file identity drift")
        if row.aggregation_rule != "line_b_equal_role_families":
            raise ValueError("role-level OOF requires canonical role-aware aggregation")
        key = (
            row.repeat,
            row.fold,
            row.split_seed,
            row.config_hash,
            row.manifest_hash,
            row.fold_hash,
            row.preprocessing_hash,
            row.feature_hash,
            row.model_hash,
            row.representation_mode,
            row.signal_route,
            row.participant_id,
            role,
            row.prediction_kind,
            row.member_index,
            row.training_seed,
        )
        if key in seen:
            raise ValueError("duplicate canonical role-level OOF row")
        seen.add(key)


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
    "source_snapshot_hash",
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
                expected_rows = 1 if expected_member_count == 1 else expected_member_count + 1
                if len(values) != expected_rows:
                    raise ValueError(
                        "OOF subject/member completeness failed for "
                        f"{key}: expected {expected_rows}, observed {len(values)}"
                    )
                if expected_member_count == 1:
                    if values[0].prediction_kind != "single_model" or values[0].member_index is not None:
                        raise ValueError(f"single-model OOF semantics are invalid for {key}")
                else:
                    members = [row for row in values if row.prediction_kind == "ensemble_member"]
                    averages = [row for row in values if row.prediction_kind == "ensemble_average"]
                    if {row.member_index for row in members} != set(range(expected_member_count)):
                        raise ValueError(f"OOF ensemble member indices are incomplete for {key}")
                    if len(averages) != 1:
                        raise ValueError(f"OOF ensemble requires exactly one average row for {key}")
                    average = averages[0]
                    coherent_fields = (
                        "participant_id", "file_id", "role", "label", "repeat",
                        "fold", "split_seed", "config_hash", "manifest_hash",
                        "fold_hash", "preprocessing_hash", "feature_hash",
                        "representation_mode", "signal_route", "quality_score",
                        "retained", "level", "class_order", "code_commit",
                        "data_schema_id", "feature_schema_id", "model_version",
                        "aggregation_rule", "environment_hash",
                        "manifest_version", "fold_registry_version",
                        "artifact_reducer_name", "artifact_reducer_version",
                        "route_status", "source_snapshot_hash",
                        "rejection_reason", "ensemble_base_model_id",
                    )
                    for member in members:
                        drift = [
                            name for name in coherent_fields
                            if getattr(member, name) != getattr(average, name)
                        ]
                        if drift:
                            raise ValueError(
                                "OOF ensemble member/average truth or trace drift for "
                                f"{key}: {drift}"
                            )
                    retained_states = {row.retained for row in values}
                    if len(retained_states) != 1:
                        raise ValueError("ensemble completeness cannot mix retained and dropped rows")
                    if True in retained_states:
                        expected_average = np.asarray(
                            [
                                row.probabilities
                                for row in sorted(members, key=lambda row: row.member_index)
                            ],
                            dtype=np.float64,
                        ).mean(axis=0)
                        if not np.allclose(
                            expected_average,
                            np.asarray(averages[0].probabilities, dtype=np.float64),
                            rtol=0.0,
                            atol=1e-12,
                        ):
                            raise ValueError(
                                f"ensemble average OOF row is not the exact member mean for {key}"
                            )
                    else:
                        reasons = {str(row.rejection_reason) for row in values}
                        if len(reasons) != 1:
                            raise ValueError("all-dropped ensemble rows require one coherent reason")
                        if any(row.probabilities or row.class_order for row in values):
                            raise ValueError("all-dropped ensemble rows must not carry predictions")


# Descriptive formal alias / 描述性的正式协议别名。
validate_formal_oof = validate_expected_oof_roster


OOF_SCHEMA_VERSION = "ppg_frailty_oof_v2"


def _arrow_schema(pa: object, *, empty_reason: str | None = None) -> object:
    """Build the one explicit nullable/non-nullable V2 Arrow schema."""

    fields = [
        pa.field("participant_id", pa.string(), nullable=False),
        pa.field("file_id", pa.string(), nullable=False),
        pa.field("role", pa.string(), nullable=False),
        pa.field("label", pa.int64(), nullable=False),
        pa.field(
            "probabilities",
            pa.list_(pa.field("element", pa.float64(), nullable=False)),
            nullable=False,
        ),
        pa.field("repeat", pa.int64(), nullable=False),
        pa.field("fold", pa.int64(), nullable=False),
        pa.field("split_seed", pa.int64(), nullable=False),
        pa.field("training_seed", pa.int64(), nullable=True),
        pa.field("config_hash", pa.string(), nullable=False),
        pa.field("manifest_hash", pa.string(), nullable=False),
        pa.field("fold_hash", pa.string(), nullable=False),
        pa.field("preprocessing_hash", pa.string(), nullable=False),
        pa.field("feature_hash", pa.string(), nullable=False),
        pa.field("model_hash", pa.string(), nullable=False),
        pa.field("representation_mode", pa.string(), nullable=False),
        pa.field("signal_route", pa.string(), nullable=False),
        pa.field("quality_score", pa.float64(), nullable=False),
        pa.field("retained", pa.bool_(), nullable=False),
        pa.field("level", pa.string(), nullable=False),
        pa.field("window_id", pa.string(), nullable=True),
        pa.field("member_index", pa.int64(), nullable=True),
        pa.field("prediction_kind", pa.string(), nullable=False),
        pa.field(
            "member_training_seeds",
            pa.list_(pa.field("element", pa.int64(), nullable=False)),
            nullable=False,
        ),
        pa.field("ensemble_base_model_id", pa.string(), nullable=False),
        pa.field(
            "class_order",
            pa.list_(pa.field("element", pa.int64(), nullable=False)),
            nullable=False,
        ),
        pa.field("code_commit", pa.string(), nullable=False),
        pa.field("data_schema_id", pa.string(), nullable=False),
        pa.field("feature_schema_id", pa.string(), nullable=False),
        pa.field("model_version", pa.string(), nullable=False),
        pa.field("aggregation_rule", pa.string(), nullable=False),
        pa.field("environment_hash", pa.string(), nullable=False),
        pa.field("manifest_version", pa.string(), nullable=False),
        pa.field("fold_registry_version", pa.string(), nullable=False),
        pa.field("artifact_reducer_name", pa.string(), nullable=False),
        pa.field("artifact_reducer_version", pa.string(), nullable=False),
        pa.field("route_status", pa.string(), nullable=False),
        pa.field("source_snapshot_hash", pa.string(), nullable=False),
        pa.field("rejection_reason", pa.string(), nullable=True),
    ]
    metadata = {
        b"schema_version": OOF_SCHEMA_VERSION.encode("ascii"),
        b"seed_semantics": (
            b"ordinary_single_outer_cv_training_seed_equals_repeat_split_seed;"
            b"ensemble_member_seeds_are_fixed_and_split_seed_independent;"
            b"matched_single_comparator_uses_member0_seed_50042"
        ),
        b"ensemble_semantics": b"five_member_rows_plus_exact_probability_average",
        b"artifact_state": b"empty" if empty_reason is not None else b"populated",
        b"empty_reason": (
            str(empty_reason).strip().encode("utf-8") if empty_reason is not None else b""
        ),
    }
    return pa.schema(
        fields,
        metadata=metadata,
    )


def _schema_for_readback(pa: object, table: object) -> object:
    """Validate metadata state and reconstruct the exact expected V2 schema."""

    metadata = table.schema.metadata or {}
    state = metadata.get(b"artifact_state")
    reason_bytes = metadata.get(b"empty_reason")
    if state == b"populated":
        if reason_bytes != b"" or table.num_rows == 0:
            raise ValueError("populated OOF artifacts require rows and an empty empty_reason")
        return _arrow_schema(pa)
    if state == b"empty":
        try:
            reason = (reason_bytes or b"").decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("OOF empty_reason metadata is not valid UTF-8") from exc
        if not reason or table.num_rows != 0:
            raise ValueError("empty OOF artifacts require zero rows and a non-empty reason")
        return _arrow_schema(pa, empty_reason=reason)
    raise ValueError("OOF artifact_state metadata must be populated or empty")


def _rows_from_arrow_table(table: object) -> tuple[OofPredictionRow, ...]:
    rows = []
    for record in table.to_pylist():
        record["probabilities"] = tuple(record["probabilities"])
        record["class_order"] = tuple(record["class_order"])
        record["member_training_seeds"] = tuple(record["member_training_seeds"])
        rows.append(OofPredictionRow(**record))
    frozen = tuple(rows)
    validate_unique_subject_oof(frozen)
    return frozen


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
        schema = _arrow_schema(pa)
        table = pa.Table.from_pylist(records, schema=schema)
        if not table.schema.equals(schema, check_metadata=True):
            raise RuntimeError("OOF Arrow schema drifted before write")
        temporary = target.with_name(f".{target.name}.tmp")
        try:
            pq.write_table(table, temporary, compression="zstd")
            observed = pq.read_table(temporary)
            if not observed.schema.equals(schema, check_metadata=True):
                raise RuntimeError("OOF Parquet schema/metadata changed during round-trip")
            reconstructed = _rows_from_arrow_table(observed)
            if reconstructed != frozen:
                raise RuntimeError("OOF Parquet values changed during round-trip")
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        return target

    def write_empty(self, path: str | Path, reason: str) -> Path:
        """Atomically write a zero-row artifact with the exact typed V2 schema.

        Empty/failed/deliberately absent levels remain machine-readable Parquet,
        never an incompatible ad-hoc table. The non-empty reason is persisted in
        schema metadata and verified during the write/read round-trip.
        """

        empty_reason = str(reason).strip()
        if not empty_reason:
            raise ValueError("empty OOF artifacts require a non-empty reason")
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
        schema = _arrow_schema(pa, empty_reason=empty_reason)
        table = pa.Table.from_pylist([], schema=schema)
        temporary = target.with_name(f".{target.name}.tmp")
        try:
            pq.write_table(table, temporary, compression="zstd")
            observed = pq.read_table(temporary)
            expected = _schema_for_readback(pa, observed)
            if not observed.schema.equals(expected, check_metadata=True):
                raise RuntimeError("empty OOF Parquet schema/metadata changed during round-trip")
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        return target


def write_oof_parquet(rows: Iterable[OofPredictionRow], path: str | Path) -> Path:
    """Functional writer facade / 函数式写入门面。"""

    return OofWriter().write(rows, path)


def write_empty_oof_parquet(path: str | Path, reason: str) -> Path:
    """Functional facade for an exact typed zero-row V2 OOF artifact."""

    return OofWriter().write_empty(path, reason)


def read_oof_parquet(path: str | Path) -> tuple[OofPredictionRow, ...]:
    """Read only an exact V2 OOF schema and revalidate every row."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("OOF Parquet input requires optional dependency pyarrow") from exc
    source = Path(path)
    if source.suffix.lower() != ".parquet" or not source.is_file():
        raise FileNotFoundError("OOF input must be an existing .parquet file")
    table = pq.read_table(source)
    schema = _schema_for_readback(pa, table)
    if not table.schema.equals(schema, check_metadata=True):
        raise ValueError("OOF Parquet does not match the exact V2 schema/metadata")
    return _rows_from_arrow_table(table)


def read_oof_parquet_metadata(path: str | Path) -> dict[str, str]:
    """Return validated V2 schema metadata, including an empty-artifact reason."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("OOF Parquet input requires optional dependency pyarrow") from exc
    source = Path(path)
    if source.suffix.lower() != ".parquet" or not source.is_file():
        raise FileNotFoundError("OOF input must be an existing .parquet file")
    table = pq.read_table(source)
    expected = _schema_for_readback(pa, table)
    if not table.schema.equals(expected, check_metadata=True):
        raise ValueError("OOF Parquet does not match the exact V2 schema/metadata")
    return {
        key.decode("ascii"): value.decode("utf-8")
        for key, value in (table.schema.metadata or {}).items()
    }
