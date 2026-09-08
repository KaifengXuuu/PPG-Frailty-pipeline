"""Frozen formal repeated grouped folds for the 22-participant PTT dataset.

The PTT source has no frailty class target. Every included participant supplies the
same sit/walk/run activity roster, so this registry is truthfully described as
participant-grouped and activity-balanced, never as class-stratified.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

from .external_manifest import (
    INDEPENDENCE_CLAIM,
    M2_EXTERNAL_MANIFEST_SHA256,
    PTT_DATASET_ID,
    ExternalManifestError,
    ExternalRecord,
    audit_external_manifest,
)


PTT_FORMAL_REPEAT_SEEDS = (42, 10042, 20042, 30042, 40042)
PTT_FORMAL_FOLD_SIZES = (5, 5, 4, 4, 4)
PTT_FORMAL_REGISTRY_ID = "ptt_formal_repeated_grouped_activity_balanced_5x5_v2"
PTT_FORMAL_ALGORITHM = "sha256_seed_rank_grouped_activity_balanced_no_class_target_v1"
PTT_FORMAL_COLUMNS = (
    "registry_id",
    "registry_status",
    "assignment_algorithm",
    "repeat_index",
    "split_seed",
    "n_splits",
    "fold_index",
    "fold_number",
    "subject_id",
    "record_ids",
    "activity_raw",
    "activity_binary",
    "runtime_split_recomputation_allowed",
    "source_manifest_sha256",
    "independence_claim",
)

def _strict_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

def _ptt_by_subject(
    records: Iterable[ExternalRecord],
) -> dict[str, tuple[ExternalRecord, ...]]:
    materialized = list(records)
    audit_external_manifest(materialized)
    ptt = [row for row in materialized if row.dataset_id == PTT_DATASET_ID and row.inclusion_status == "included"]
    subjects = sorted({row.subject_id for row in ptt})
    if len(subjects) != 22:
        raise ExternalManifestError(["formal PTT folds require exactly 22 subjects"])
    grouped = {
        subject: tuple(
            sorted(
                (row for row in ptt if row.subject_id == subject),
                key=lambda row: row.record_id,
            )
        )
        for subject in subjects
    }
    for subject, rows in grouped.items():
        if len(rows) != 3 or {row.activity_raw for row in rows} != {
            "sit",
            "walk",
            "run",
        }:
            raise ExternalManifestError([f"formal PTT subject lacks sit/walk/run roster: {subject}"])
    return grouped

def _fold_by_subject(subjects: Iterable[str], seed: int) -> dict[str, int]:
    ordered = sorted(
        (str(subject) for subject in subjects),
        key=lambda subject: (
            hashlib.sha256(f"{seed}:{subject}".encode("utf-8")).hexdigest(),
            subject,
        ),
    )
    if len(ordered) != sum(PTT_FORMAL_FOLD_SIZES):
        raise ExternalManifestError(["formal PTT fold-size contract requires 22 subjects"])
    assignments: dict[str, int] = {}
    cursor = 0
    for fold_index, size in enumerate(PTT_FORMAL_FOLD_SIZES):
        for subject in ordered[cursor : cursor + size]:
            assignments[subject] = fold_index
        cursor += size
    return assignments

def build_formal_ptt_fold_rows(
    records: Iterable[ExternalRecord],
) -> list[dict[str, str]]:
    """Build the frozen-row payload; callers must materialize it before use."""

    grouped = _ptt_by_subject(records)
    output: list[dict[str, str]] = []
    for repeat_index, seed in enumerate(PTT_FORMAL_REPEAT_SEEDS):
        assignments = _fold_by_subject(grouped, seed)
        for subject in sorted(grouped):
            rows = grouped[subject]
            fold_index = assignments[subject]
            output.append(
                {
                    "registry_id": PTT_FORMAL_REGISTRY_ID,
                    "registry_status": "frozen_formal_benchmark",
                    "assignment_algorithm": PTT_FORMAL_ALGORITHM,
                    "repeat_index": str(repeat_index),
                    "split_seed": str(seed),
                    "n_splits": "5",
                    "fold_index": str(fold_index),
                    "fold_number": str(fold_index + 1),
                    "subject_id": subject,
                    "record_ids": _strict_json([row.record_id for row in rows]),
                    "activity_raw": _strict_json(sorted({row.activity_raw for row in rows})),
                    "activity_binary": _strict_json(sorted({row.activity_binary for row in rows})),
                    "runtime_split_recomputation_allowed": "false",
                    "source_manifest_sha256": M2_EXTERNAL_MANIFEST_SHA256,
                    "independence_claim": INDEPENDENCE_CLAIM,
                }
            )
    validate_formal_ptt_fold_rows(output)
    return output

def validate_formal_ptt_fold_rows(rows: Iterable[Mapping[str, str]]) -> None:
    """Audit repeat seeds, grouping, fold sizes, activities, and fixed identity."""

    materialized = [dict(row) for row in rows]
    issues: list[str] = []
    if len(materialized) != 22 * len(PTT_FORMAL_REPEAT_SEEDS):
        issues.append(f"expected 110 formal assignment rows, observed {len(materialized)}")
    subjects = {row.get("subject_id", "") for row in materialized}
    if len(subjects) != 22 or "" in subjects:
        issues.append("formal PTT registry must contain exactly 22 named subjects")
    keys = [(row.get("repeat_index", ""), row.get("subject_id", "")) for row in materialized]
    if len(set(keys)) != len(keys):
        issues.append("duplicate repeat/subject assignment")
    expected_common = {
        "registry_id": PTT_FORMAL_REGISTRY_ID,
        "registry_status": "frozen_formal_benchmark",
        "assignment_algorithm": PTT_FORMAL_ALGORITHM,
        "n_splits": "5",
        "runtime_split_recomputation_allowed": "false",
        "source_manifest_sha256": M2_EXTERNAL_MANIFEST_SHA256,
        "independence_claim": INDEPENDENCE_CLAIM,
    }
    for index, row in enumerate(materialized):
        if any(row.get(key) != value for key, value in expected_common.items()):
            issues.append(f"formal registry identity drift at row {index}")
        try:
            activities = set(json.loads(row.get("activity_raw", "")))
            record_ids = json.loads(row.get("record_ids", ""))
        except (TypeError, json.JSONDecodeError):
            issues.append(f"invalid JSON fields at row {index}")
            continue
        if activities != {"sit", "walk", "run"} or not isinstance(record_ids, list) or len(record_ids) != 3:
            issues.append(f"incomplete activity/record roster at row {index}")
    for repeat_index, seed in enumerate(PTT_FORMAL_REPEAT_SEEDS):
        repeat = [row for row in materialized if row.get("repeat_index") == str(repeat_index)]
        if len(repeat) != 22 or {row.get("subject_id") for row in repeat} != subjects:
            issues.append(f"repeat {repeat_index} participant partition drift")
        if {row.get("split_seed") for row in repeat} != {str(seed)}:
            issues.append(f"repeat {repeat_index} seed drift")
        counts = Counter(int(row["fold_index"]) for row in repeat)
        observed = tuple(counts[index] for index in range(5))
        if observed != PTT_FORMAL_FOLD_SIZES:
            issues.append(f"repeat {repeat_index} fold sizes {observed} != {PTT_FORMAL_FOLD_SIZES}")
    if issues:
        raise ExternalManifestError(issues)

def _checked_target(path: str | Path, *, output_root: str | Path) -> Path:
    root = Path(output_root).resolve()
    target = Path(path).resolve(strict=False)
    target.relative_to(root)
    return target

def materialize_formal_ptt_repeated_folds(
    records: Iterable[ExternalRecord],
    output_csv: str | Path,
    *,
    output_root: str | Path,
) -> list[dict[str, str]]:
    """Atomically save the sole formal registry; runtime regeneration is forbidden."""

    rows = build_formal_ptt_fold_rows(records)
    target = _checked_target(output_csv, output_root=output_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PTT_FORMAL_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)
    return rows

def load_formal_ptt_repeated_folds(path: str | Path) -> list[dict[str, str]]:
    """Load and audit saved assignments without recomputing them."""

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PTT_FORMAL_COLUMNS:
            raise ExternalManifestError(["formal PTT registry column order drift"])
        rows = [dict(row) for row in reader]
    validate_formal_ptt_fold_rows(rows)
    return rows

def resolve_formal_ptt_split(
    rows: Iterable[Mapping[str, str]],
    *,
    repeat_index: int,
    fold_index: int,
) -> dict[str, tuple[str, ...] | int]:
    """Resolve train/OOF subjects solely from materialized assignments."""

    materialized = [dict(row) for row in rows]
    validate_formal_ptt_fold_rows(materialized)
    if repeat_index not in range(5) or fold_index not in range(5):
        raise ValueError("repeat_index and fold_index must lie in [0,4]")
    repeat = [row for row in materialized if int(row["repeat_index"]) == repeat_index]
    oof = tuple(sorted(row["subject_id"] for row in repeat if int(row["fold_index"]) == fold_index))
    train = tuple(sorted(row["subject_id"] for row in repeat if int(row["fold_index"]) != fold_index))
    if set(train) & set(oof) or len(train) + len(oof) != 22:
        raise ExternalManifestError(["resolved formal PTT split is not disjoint/exhaustive"])
    return {
        "repeat_index": repeat_index,
        "split_seed": PTT_FORMAL_REPEAT_SEEDS[repeat_index],
        "fold_index": fold_index,
        "train_subject_ids": train,
        "oof_subject_ids": oof,
    }


__all__ = [
    "PTT_FORMAL_ALGORITHM",
    "PTT_FORMAL_COLUMNS",
    "PTT_FORMAL_FOLD_SIZES",
    "PTT_FORMAL_REGISTRY_ID",
    "PTT_FORMAL_REPEAT_SEEDS",
    "build_formal_ptt_fold_rows",
    "load_formal_ptt_repeated_folds",
    "materialize_formal_ptt_repeated_folds",
    "resolve_formal_ptt_split",
    "validate_formal_ptt_fold_rows",
]
