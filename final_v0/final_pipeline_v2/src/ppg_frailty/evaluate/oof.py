"""OOF 完整性 canonical facade / Canonical OOF-integrity facade.

中文：exact-once roster、完整 trace 与 ensemble 成员集合只由
training.oof.validate_expected_oof_roster 校验；此处只转换旧键结构。
English: The training validator is the sole authority; this file only adapts keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from ..training.oof import (
    FORMAL_TRACE_FIELDS,
    OofPredictionRow,
    OOF_SCHEMA_VERSION,
    read_oof_parquet,
    validate_expected_oof_roster,
    validate_unique_subject_oof,
)


@dataclass(frozen=True)
class OofContractAudit:
    """成功校验后的机器可读计数 / Machine-readable counts after validation."""

    subject_rows: int
    expected_subject_rows: int
    exact_once: bool
    member_complete: bool
    trace_complete: bool


def validate_oof_contract(
    rows: Iterable[OofPredictionRow],
    expected_oof_participants: Mapping[tuple[int, int, int, str], set[str]],
    *,
    ensemble_size: int = 1,
) -> OofContractAudit:
    """转换含 config 的旧 roster 键并调用唯一 validator / Adapt and delegate."""

    frozen = tuple(rows)
    roster: dict[tuple[int, int, int], set[str]] = {}
    configurations: set[str] = set()
    for key, participants in expected_oof_participants.items():
        if len(key) != 4:
            raise ValueError("expected OOF key must be (repeat,fold,seed,config_hash)")
        repeat, fold, seed, config_hash = key
        triple = (int(repeat), int(fold), int(seed))
        values = set(str(value) for value in participants)
        previous = roster.setdefault(triple, values)
        if previous != values:
            raise ValueError("held-out roster differs across configurations for one split")
        configurations.add(str(config_hash))
    validate_expected_oof_roster(
        frozen,
        roster,
        expected_config_hashes=configurations,
        expected_level="participant",
        expected_member_count=ensemble_size,
        require_trace=True,
    )
    selected = tuple(row for row in frozen if row.level == "participant")
    rows_per_subject = 1 if ensemble_size == 1 else ensemble_size + 1
    expected_count = sum(len(values) for values in roster.values()) * len(configurations) * rows_per_subject
    return OofContractAudit(
        subject_rows=len(selected),
        expected_subject_rows=expected_count,
        exact_once=True,
        member_complete=True,
        trace_complete=True,
    )


__all__ = [
    "FORMAL_TRACE_FIELDS",
    "OofContractAudit",
    "OofPredictionRow",
    "OOF_SCHEMA_VERSION",
    "read_oof_parquet",
    "validate_expected_oof_roster",
    "validate_oof_contract",
    "validate_unique_subject_oof",
]
