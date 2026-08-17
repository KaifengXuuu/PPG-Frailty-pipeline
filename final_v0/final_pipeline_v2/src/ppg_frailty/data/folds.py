"""只读导入并物化 outer-fold membership / Frozen outer-fold memberships.

中文：本模块从 M2 corrected SGKF JSON 读取已物化 membership。任何运行时
splitter 调用都不属于公共 API。CSV 每行只表达 participant 的 OOF fold，训练集
始终是同一 repeat 全体 participant 的补集。

English: This module imports the already-materialized corrected M2 memberships.
No runtime splitter belongs to the public API. Each CSV row records a participant's
OOF fold; training membership is the complement within that repeat.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from ppg_frailty.contracts import ManifestRow
from ppg_frailty.provenance import sha256_file

from .manifest import M2_DATASET_VERSION_ID
from .schema import (
    CANONICAL_CLASS_NAMES,
    FOLD_COLUMNS,
    FoldAssignment,
    fold_assignment_from_csv,
    fold_assignment_to_csv,
)


M2_SPLIT_RELATIVE_PATH = Path(
    "final_v0/M2_data_manifest_and_evaluation_protocol/"
    "splits/frailty3_future_corrected_sgkf5_v2.json"
)
M2_SPLIT_FILE_SHA256 = (
    "c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c"
)
M2_SPLIT_PAYLOAD_SHA256 = (
    "0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46"
)
M2_SUBJECT_INPUT_SHA256 = (
    "6a9c5e057200f32fefa44ed7c77cd6c5ef4f1aa61af36dc2e5f8d786861a5e10"
)
M2_SPLIT_REGISTRY_ID = "frailty3_future_corrected_sgkf5_v2"
M2_SEEDS = (42, 10042, 20042, 30042, 40042)
OUTER_CV_SINGLE_MODEL_SEED_POLICY = "outer_cv_repeat_seed_equals_split_seed"


def outer_cv_single_model_training_seed(
    repeat_index: int,
    split_seed: int,
) -> int:
    """Return the frozen model RNG for one repeat; folds never offset it."""

    repeat = int(repeat_index)
    observed = int(split_seed)
    if repeat < 0 or repeat >= len(M2_SEEDS):
        raise ValueError(f"outer CV repeat index is outside 0..4: {repeat}")
    expected = int(M2_SEEDS[repeat])
    if observed != expected:
        raise ValueError(
            f"outer CV split seed drift for repeat {repeat}: {observed} != {expected}"
        )
    return observed


@dataclass(frozen=True)
class FrozenFoldAudit:
    """冻结 registry 的机器验收摘要 / Machine audit of frozen memberships."""

    registry_id: str
    registry_file_sha256: str
    registry_payload_sha256: str
    participant_count: int
    repeat_count: int
    fold_count_per_repeat: int
    assignment_count: int
    all_classes_present: bool
    class_balance_spread_at_most_one: bool
    train_oof_disjoint: bool
    oof_partition_exact: bool


@dataclass(frozen=True)
class FrozenFoldRegistry:
    """CSV-backed frozen fold registry / CSV 支持的冻结分折注册表."""

    assignments: tuple[FoldAssignment, ...]

    @classmethod
    def from_csv(cls, path: str | Path) -> "FrozenFoldRegistry":
        """从物化 CSV 加载 / Load from one materialized CSV."""

        rows: list[FoldAssignment] = []
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != FOLD_COLUMNS:
                raise ValueError("frozen fold CSV column order drift")
            for line_number, raw in enumerate(reader, start=2):
                try:
                    rows.append(fold_assignment_from_csv(raw))
                except Exception as exc:  # noqa: BLE001 - include line identity.
                    raise ValueError(
                        f"invalid fold row at line {line_number}: {exc}"
                    ) from exc
        if not rows:
            raise ValueError("frozen fold CSV is empty")
        _validate_assignment_rows(rows)
        return cls(tuple(rows))

    @property
    def participant_ids(self) -> tuple[str, ...]:
        """返回稳定 participant roster / Return stable participant roster."""

        return tuple(sorted({row.participant_id for row in self.assignments}))

    def get_split(self, repeat_index: int, fold_index: int) -> dict[str, Any]:
        """解析 train/OOF membership / Resolve one train/OOF partition."""

        repeat_rows = [
            row for row in self.assignments if row.repeat_index == int(repeat_index)
        ]
        if not repeat_rows:
            raise KeyError(f"repeat_index_not_found:{repeat_index}")
        oof = sorted(
            row.participant_id
            for row in repeat_rows
            if row.fold_index == int(fold_index)
        )
        if not oof:
            raise KeyError(f"fold_index_not_found:{fold_index}")
        all_participants = {row.participant_id for row in repeat_rows}
        train = sorted(all_participants - set(oof))
        seeds = {row.split_seed for row in repeat_rows}
        if len(seeds) != 1:
            raise ValueError("repeat contains multiple split seeds")
        split_seed = int(next(iter(seeds)))
        return {
            "repeat_index": int(repeat_index),
            "split_seed": split_seed,
            "fold_index": int(fold_index),
            # Outer-CV model RNG is repeat-local: all five folds in a repeat
            # share that repeat's split seed. Final-refit seed 42 is separate.
            "training_seed": outer_cv_single_model_training_seed(
                int(repeat_index),
                split_seed,
            ),
            "train_participant_ids": train,
            "oof_participant_ids": oof,
        }


def _canonical_registry_payload(registry: Mapping[str, Any]) -> dict[str, Any]:
    """移除自引用 hash 字段 / Remove the self-referential hash field."""

    return {key: value for key, value in registry.items() if key != "payload_sha256"}


def load_frozen_memberships(path: str | Path) -> dict[str, Any]:
    """加载并验证唯一 M2 corrected JSON / Load the authoritative M2 JSON."""

    source = Path(path)
    file_digest = sha256_file(source)
    if file_digest != M2_SPLIT_FILE_SHA256:
        raise ValueError(
            f"frozen registry file SHA drift: {file_digest} != {M2_SPLIT_FILE_SHA256}"
        )
    registry = json.loads(source.read_text(encoding="utf-8"))
    # 中文：M2 payload identity 使用其 builder 的 pretty strict JSON bytes，
    # 不是 V1 provenance.py 的 compact JSON identity。这里忠实复现来源规则。
    # English: M2 payload identity hashes builder-style pretty strict JSON bytes,
    # not V1's compact JSON identity. Reproduce the authority rule exactly.
    canonical_text = json.dumps(
        _canonical_registry_payload(registry),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    payload_digest = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
    if registry.get("payload_sha256") != M2_SPLIT_PAYLOAD_SHA256:
        raise ValueError("declared M2 registry payload SHA drift")
    if payload_digest != M2_SPLIT_PAYLOAD_SHA256:
        raise ValueError("computed M2 registry payload SHA drift")
    required = {
        "registry_id": M2_SPLIT_REGISTRY_ID,
        "dataset_version_id": M2_DATASET_VERSION_ID,
        "subject_input_sha256": M2_SUBJECT_INPUT_SHA256,
        "runtime_split_recomputation_allowed": False,
        "n_splits": 5,
        "n_repeats": 5,
        "seeds": list(M2_SEEDS),
        "class_missing_fold_count": 0,
        "invariants_pass": True,
    }
    for key, expected in required.items():
        if registry.get(key) != expected:
            raise ValueError(f"frozen registry invariant drift: {key}")
    return registry


def load_m2_frozen_registry(repository_root: str | Path) -> dict[str, Any]:
    """从 repository root 解析 M2 registry / Resolve M2 from repository root."""

    return load_frozen_memberships(
        Path(repository_root).resolve() / M2_SPLIT_RELATIVE_PATH
    )


def _participant_class_map(rows: Iterable[ManifestRow]) -> dict[str, int]:
    """建立 participant→class 并拒绝冲突 / Build a conflict-free class map."""

    mapping: dict[str, int] = {}
    for row in rows:
        previous = mapping.setdefault(row.participant_id, row.class_id)
        if previous != row.class_id:
            raise ValueError(f"participant class conflict: {row.participant_id}")
    return mapping


def validate_frozen_memberships(
    registry: Mapping[str, Any],
    manifest_rows: Iterable[ManifestRow],
) -> FrozenFoldAudit:
    """验证每个 repeat 的 partition/class/file inheritance / Audit memberships."""

    rows = list(manifest_rows)
    classes = _participant_class_map(rows)
    participant_set = set(classes)
    expected_files_by_participant = {
        participant: {row.record_id for row in rows if row.participant_id == participant}
        for participant in participant_set
    }
    train_oof_disjoint = True
    oof_partition_exact = True
    all_classes_present = True
    class_balance = True
    assignment_count = 0
    repeats = registry.get("repeats", [])
    if len(repeats) != 5:
        raise ValueError("frozen registry must contain five repeats")
    for expected_repeat_index, repeat in enumerate(repeats):
        if repeat.get("repeat_index") != expected_repeat_index:
            raise ValueError("repeat index/order drift")
        if repeat.get("split_seed") != M2_SEEDS[expected_repeat_index]:
            raise ValueError("repeat seed drift")
        folds = repeat.get("folds", [])
        if len(folds) != 5:
            raise ValueError("repeat must contain five folds")
        seen: list[str] = []
        class_counts_by_fold = {class_id: [] for class_id in CANONICAL_CLASS_NAMES}
        for expected_fold_index, fold in enumerate(folds):
            if fold.get("fold_index") != expected_fold_index:
                raise ValueError("fold index/order drift")
            train = set(fold.get("train_subject_ids", []))
            oof = set(fold.get("oof_validation_subject_ids", []))
            train_oof_disjoint &= not bool(train & oof)
            oof_partition_exact &= train | oof == participant_set
            seen.extend(sorted(oof))
            assignment_count += len(oof)
            observed_class_ids = {classes[participant] for participant in oof}
            all_classes_present &= observed_class_ids == set(CANONICAL_CLASS_NAMES)
            for class_id in CANONICAL_CLASS_NAMES:
                class_counts_by_fold[class_id].append(
                    sum(classes[participant] == class_id for participant in oof)
                )
            # 中文：验证 JSON 中的 file membership 确实继承 participant fold。
            # English: Verify file membership is inherited from participant membership.
            expected_oof_files = {
                record_id
                for participant in oof
                for record_id in expected_files_by_participant[participant]
            }
            if set(fold.get("oof_validation_file_ids_all_roles", [])) != expected_oof_files:
                raise ValueError("OOF file membership does not inherit participant fold")
        oof_partition_exact &= len(seen) == len(participant_set)
        oof_partition_exact &= set(seen) == participant_set
        for values in class_counts_by_fold.values():
            class_balance &= max(values) - min(values) <= 1
    audit = FrozenFoldAudit(
        registry_id=str(registry["registry_id"]),
        registry_file_sha256=M2_SPLIT_FILE_SHA256,
        registry_payload_sha256=M2_SPLIT_PAYLOAD_SHA256,
        participant_count=len(participant_set),
        repeat_count=len(repeats),
        fold_count_per_repeat=5,
        assignment_count=assignment_count,
        all_classes_present=bool(all_classes_present),
        class_balance_spread_at_most_one=bool(class_balance),
        train_oof_disjoint=bool(train_oof_disjoint),
        oof_partition_exact=bool(oof_partition_exact),
    )
    if not all(
        (
            audit.all_classes_present,
            audit.class_balance_spread_at_most_one,
            audit.train_oof_disjoint,
            audit.oof_partition_exact,
        )
    ):
        raise ValueError(f"frozen membership audit failed: {audit}")
    return audit


def materialize_assignments(
    registry: Mapping[str, Any],
    manifest_rows: Iterable[ManifestRow],
) -> list[FoldAssignment]:
    """将 M2 JSON 转成每 participant 一行 / Convert JSON to long assignments."""

    rows = list(manifest_rows)
    validate_frozen_memberships(registry, rows)
    classes = _participant_class_map(rows)
    assignments: list[FoldAssignment] = []
    for repeat in registry["repeats"]:
        for fold in repeat["folds"]:
            for participant in fold["oof_validation_subject_ids"]:
                class_id = classes[str(participant)]
                assignments.append(
                    FoldAssignment(
                        source_registry_id=M2_SPLIT_REGISTRY_ID,
                        source_registry_file_sha256=M2_SPLIT_FILE_SHA256,
                        source_registry_payload_sha256=M2_SPLIT_PAYLOAD_SHA256,
                        dataset_version_id=M2_DATASET_VERSION_ID,
                        repeat_index=int(repeat["repeat_index"]),
                        split_seed=int(repeat["split_seed"]),
                        fold_index=int(fold["fold_index"]),
                        fold_number=int(fold["fold_number"]),
                        participant_id=str(participant),
                        class_id=class_id,
                        class_name=CANONICAL_CLASS_NAMES[class_id],
                    )
                )
    assignments.sort(
        key=lambda row: (
            row.repeat_index,
            row.fold_index,
            row.participant_id.encode("utf-8"),
        )
    )
    _validate_assignment_rows(assignments)
    return assignments


def _validate_assignment_rows(rows: Iterable[FoldAssignment]) -> None:
    """验证长表唯一性 / Validate long-table uniqueness."""

    materialized = list(rows)
    keys = [(row.repeat_index, row.participant_id) for row in materialized]
    if len(set(keys)) != len(keys):
        raise ValueError("participant has multiple folds within one repeat")
    for repeat_index in sorted({row.repeat_index for row in materialized}):
        repeat_rows = [row for row in materialized if row.repeat_index == repeat_index]
        if len(repeat_rows) != 29:
            raise ValueError(f"repeat {repeat_index} does not contain 29 participants")
        if {row.fold_index for row in repeat_rows} != set(range(5)):
            raise ValueError(f"repeat {repeat_index} does not contain five folds")


def _checked_target(path: str | Path, *, output_root: str | Path) -> Path:
    """限制 CSV 写入 final pipeline root / Restrict CSV writes."""

    root = Path(output_root).resolve()
    target = Path(path).resolve(strict=False)
    target.relative_to(root)
    return target


def _write_fold_csv(
    path: str | Path,
    rows: Iterable[FoldAssignment],
    *,
    output_root: str | Path,
) -> None:
    """原子写 fold CSV / Atomically write one fold CSV."""

    target = _checked_target(path, output_root=output_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FOLD_COLUMNS))
        writer.writeheader()
        writer.writerows(fold_assignment_to_csv(row) for row in rows)
    temporary.replace(target)


def materialize_fold_csvs(
    registry: Mapping[str, Any],
    manifest_rows: Iterable[ManifestRow],
    primary_output_csv: str | Path,
    repeats_output_csv: str | Path,
    *,
    output_root: str | Path,
) -> tuple[list[FoldAssignment], list[FoldAssignment]]:
    """写 seed42 主表与完整5×5表 / Write primary and complete repeat tables."""

    repeated = materialize_assignments(registry, manifest_rows)
    primary = [row for row in repeated if row.repeat_index == 0]
    _write_fold_csv(primary_output_csv, primary, output_root=output_root)
    _write_fold_csv(repeats_output_csv, repeated, output_root=output_root)
    return primary, repeated


def resolve_outer_fold(
    assignments: Iterable[FoldAssignment],
    repeat_index: int,
    fold_index: int,
) -> dict[str, Any]:
    """从内存 assignment 解析一个 split / Resolve one split in memory."""

    return FrozenFoldRegistry(tuple(assignments)).get_split(
        repeat_index=repeat_index,
        fold_index=fold_index,
    )


__all__ = [
    "FrozenFoldAudit",
    "FrozenFoldRegistry",
    "M2_SEEDS",
    "OUTER_CV_SINGLE_MODEL_SEED_POLICY",
    "M2_SPLIT_FILE_SHA256",
    "M2_SPLIT_PAYLOAD_SHA256",
    "load_frozen_memberships",
    "load_m2_frozen_registry",
    "materialize_assignments",
    "materialize_fold_csvs",
    "outer_cv_single_model_training_seed",
    "resolve_outer_fold",
    "validate_frozen_memberships",
]
