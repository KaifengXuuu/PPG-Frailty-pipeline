#!/usr/bin/env python3
"""构建 M2 数据 manifests、外部记录清单与双 fold 注册表。

中文：完整读取 Frailty3 261 个 CSV 的每个字节，校验八通道 header，把全部数据
token 解析为有限浮点数，并物化历史错误/未来修正两套 subject-level SGKF registry。
English: Read every byte/token of all 261 Frailty3 CSVs and materialize both the
historical defective and future corrected subject-level SGKF registries.

Safety / 安全：源数据只读；所有写入路径硬限制在本 M2 包内。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# 中文：路径从脚本位置解析，避免当前目录重定向输出。
# English: Resolve paths from this script so cwd cannot redirect outputs.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = FINAL_ROOT.parent
FRAILTY_ROOT = REPO_ROOT / "PPG_Testing_05_01_2026"
OLDER_ROOT = FRAILTY_ROOT / "StudyData"
YOUNG_ROOT = FRAILTY_ROOT / "TestDataYoungers"
LABEL_PATH = FRAILTY_ROOT / "StudyData_frailtyScored" / "StudyData_V7_standard.csv"
PHYSIONET_ROOT = REPO_ROOT / "physionet.org" / "files"
PTT_ROOT = PHYSIONET_ROOT / "pulse-transit-time-ppg" / "1.1.0"
SIM_ROOT = PHYSIONET_ROOT / "simultaneous-measurements" / "1.0.0"
MANIFEST_ROOT = PACKAGE_ROOT / "manifests"
SPLIT_ROOT = PACKAGE_ROOT / "splits"
EXAMPLE_ROOT = PACKAGE_ROOT / "examples"
BUILD_REPORT = PACKAGE_ROOT / "M2_BUILD_REPORT.json"

EXPECTED_HEADER = ("RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ")
EXPECTED_ROLES = ("B", "R1", "R2", "R3", "R4", "S1", "S2", "W1", "W2")
STATIC_ROLES = frozenset(("B", "R1", "R2", "R3", "R4"))
SEEDS = (42, 10042, 20042, 30042, 40042)
N_SPLITS = 5
RAW_FS_HZ = 400.0
CLASS_NAMES = {0: "pre_frail", 1: "robust_non_frail", 2: "young"}
ROLE_PATTERN = re.compile(r"^(?P<subject>.+)_(?P<role>B|R[1-4]|S[1-2]|W[1-2])\.csv$")
PTT_PATTERN = re.compile(r"^(s(?P<number>\d+))_(?P<activity>sit|walk|run)\.csv$")
AUDITED_LABEL_SHA256 = "fed31cf6b576fd7cc138b7e6101b43a8bd8b31312c97fa2839cb9942095ce8e3"
AUDITED_RAW_TREE_SHA256 = "8b17c299a67e88aa39f273203f00ffaba4902fc6c5b1e05c05de7f6798ec5e13"
AUDITED_RAW_LABEL_TREE_SHA256 = "d7bca800beb041bba390c63d685797cecb81ba022ff9d83f1aaeff816047b9a7"
UNITS = {
    "RED": "raw_device_counts_adc_scale_unknown",
    "IR": "raw_device_counts_adc_scale_unknown",
    "AX": "g_source_declared",
    "AY": "g_source_declared",
    "AZ": "g_source_declared",
    "GX": "degree_per_second_source_declared",
    "GY": "degree_per_second_source_declared",
    "GZ": "degree_per_second_source_declared",
}


def sha256_bytes(payload: bytes) -> str:
    """返回字节 SHA-256；return the SHA-256 digest of bytes."""

    return hashlib.sha256(payload).hexdigest()


def stable_json_bytes(value: Any) -> bytes:
    """生成确定性严格 JSON；render deterministic strict JSON bytes."""

    text = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    )
    return (text + "\n").encode("utf-8")


def checked_target(path: Path) -> Path:
    """拒绝 M2 包外写入；reject every write outside the M2 package."""

    package = PACKAGE_ROOT.resolve()
    target = path.resolve(strict=False)
    try:
        target.relative_to(package)
    except ValueError as exc:
        raise RuntimeError(f"Refusing write outside M2 package: {target}") from exc
    return target


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """在包内原子写入；atomically write inside the guarded package."""

    target = checked_target(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)


def write_json(path: Path, value: Any) -> None:
    """写严格 JSON；write strict deterministic JSON."""

    atomic_write_bytes(path, stable_json_bytes(value))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """原子写 UTF-8 CSV；atomically write a UTF-8 CSV."""

    target = checked_target(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)


def relative_repo(path: Path) -> str:
    """返回仓库相对路径；return a stable repository-relative path."""

    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def permission_snapshot(root: Path) -> dict[str, int]:
    """记录源树只读权限位；record read-only permission bits for a source tree."""

    files = [path for path in root.rglob("*") if path.is_file()]
    directories = [root] + [path for path in root.rglob("*") if path.is_dir()]
    return {
        "file_count": len(files),
        "directory_count": len(directories),
        "files_with_write_bits": sum(
            bool(path.stat().st_mode & 0o222) for path in files
        ),
        "directories_with_write_bits": sum(
            bool(path.stat().st_mode & 0o222) for path in directories
        ),
    }


def load_labels() -> tuple[dict[str, dict[str, str]], str]:
    """读取当前权威标签 CSV；read the active authoritative label CSV."""

    payload = LABEL_PATH.read_bytes()
    digest = sha256_bytes(payload)
    if digest != AUDITED_LABEL_SHA256:
        raise RuntimeError(f"Label SHA drift: {digest}")
    with LABEL_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = {row["ID"].strip(): row for row in csv.DictReader(handle)}
    return rows, digest


def role_metadata(role: str) -> tuple[str, str, str, str]:
    """映射仅获确认的 role 语义；map only confirmed role-family semantics."""

    family = role[0]
    stage = {
        "B": "baseline",
        "R": "relax_or_recovery",
        "S": "stand_and_sit",
        "W": "walk",
    }[family]
    activity = "static" if family in {"B", "R"} else "motion"
    index_status = "not_applicable" if role == "B" else "unverified"
    return family, stage, activity, index_status


def classify_subject(
    subject_id: str, cohort: str, labels: dict[str, dict[str, str]]
) -> tuple[int, str, str, str]:
    """按 cohort/标签合同赋类；assign class under the frozen label contract."""

    label_id = subject_id.rsplit("_", 1)[0]
    if cohort == "young":
        # 中文：Young 是 cohort class；标签表 status 不得覆盖它。
        # English: Young is a cohort class; label-table status cannot override it.
        linked = label_id if label_id in labels else ""
        return 2, CLASS_NAMES[2], "cohort_override_young", linked
    if label_id not in labels:
        raise RuntimeError(f"Missing older label row: {subject_id} -> {label_id}")
    status = int(float(labels[label_id]["FRAILTY-STATUS"]))
    if status == 2:
        return 0, CLASS_NAMES[0], "frailty_status_2", label_id
    if status == 3:
        return 1, CLASS_NAMES[1], "frailty_status_3", label_id
    raise RuntimeError(f"Unsupported FRAILTY-STATUS={status} for {subject_id}")


def parse_numeric_csv(path: Path) -> tuple[str, int, int]:
    """完整读取、哈希并解析所有数值；read, hash, and parse every numeric token."""

    payload = path.read_bytes()
    digest = sha256_bytes(payload)
    header_bytes, separator, body = payload.partition(b"\n")
    if not separator:
        raise RuntimeError(f"Missing data body: {path}")
    header = tuple(
        part.decode("ascii") for part in header_bytes.rstrip(b"\r").split(b",")
    )
    if header != EXPECTED_HEADER:
        raise RuntimeError(f"Header mismatch in {path}: {header}")
    # 中文：行终止符转成分隔符，NumPy 消费并验证每个 token。
    # English: Convert row endings so NumPy consumes and validates every token.
    row_count = body.count(b"\n") + (0 if body.endswith(b"\n") else 1)
    flattened = body.replace(b"\r\n", b",").replace(b"\n", b",").rstrip(b",")
    values = np.fromstring(flattened.decode("ascii"), dtype=np.float64, sep=",")
    expected_values = row_count * len(EXPECTED_HEADER)
    if values.size != expected_values:
        raise RuntimeError(
            f"Numeric shape mismatch in {path}: {values.size} != {expected_values}"
        )
    if not bool(np.isfinite(values).all()):
        raise RuntimeError(f"NaN/Inf found in {path}")
    return digest, len(payload), row_count


def scan_frailty_files() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """扫描 261 files 并形成 subject rows；scan all files and derive subject rows."""

    labels, label_sha = load_labels()
    file_rows: list[dict[str, Any]] = []
    snapshot_lines: list[str] = []
    subject_files: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cohort, directory in (("older", OLDER_ROOT), ("young", YOUNG_ROOT)):
        paths = sorted(directory.glob("*.csv"), key=lambda item: item.name.encode("utf-8"))
        for path in paths:
            match = ROLE_PATTERN.fullmatch(path.name)
            if not match:
                raise RuntimeError(f"Unexpected Frailty3 filename: {path.name}")
            subject_id, role = match.group("subject"), match.group("role")
            class_id, class_name, class_source, label_id = classify_subject(
                subject_id, cohort, labels
            )
            digest, size, samples = parse_numeric_csv(path)
            duration = samples / RAW_FS_HZ
            family, stage, activity, index_status = role_metadata(role)
            warnings: list[str] = []
            if duration < 30.0:
                warnings.append("shorter_than_30s_window")
            if role == "B" and duration > 330.0:
                warnings.append("baseline_duration_above_330s_review")
            if role in {"R1", "R2"} and duration < 250.0:
                warnings.append("recovery_duration_below_250s_review")
            relative = relative_repo(path)
            row = {
                "dataset_version_id": "PENDING_SNAPSHOT_DIGEST",
                "file_id": f"frailty3:{subject_id}:{role}",
                "subject_id": subject_id,
                "cohort": cohort,
                "class_id": class_id,
                "class_name": class_name,
                "class_source": class_source,
                "label_record_id": label_id,
                "relative_path": relative,
                "role": role,
                "role_family": family,
                "stage_name": stage,
                "activity_state": activity,
                "role_family_semantics_status": "confirmed",
                "role_instance_index_semantics": index_status,
                "confirmed_partial_order": (
                    "precedes_R_family" if family in {"S", "W"}
                    else "follows_S_or_W_family" if family == "R" else "not_encoded"
                ),
                "channels": json.dumps(EXPECTED_HEADER, separators=(",", ":")),
                "raw_fs_hz": f"{RAW_FS_HZ:g}",
                "sampling_rate_source": "user_confirmed_and_first_party_code_no_csv_timestamp",
                "units": json.dumps(UNITS, sort_keys=True, separators=(",", ":")),
                "n_samples": samples,
                "duration_seconds": f"{duration:.6f}",
                "file_bytes": size,
                "sha256": digest,
                "numeric_full_scan": "passed_finite_8_columns",
                "inclusion_status": "included",
                "inclusion_reason": "frozen_29_subject_roster",
                "reference_available": "false",
                "reference_type": "none",
                "warning_codes": ";".join(warnings),
            }
            file_rows.append(row)
            subject_files[subject_id].append(row)
            snapshot_lines.append(f"{relative}\0{size}\0{digest}\n")

    if len(file_rows) != 261 or len(subject_files) != 29:
        raise RuntimeError(
            f"Unexpected roster: files={len(file_rows)}, subjects={len(subject_files)}"
        )
    snapshot_lines.append(
        f"{relative_repo(LABEL_PATH)}\0{LABEL_PATH.stat().st_size}\0{label_sha}\n"
    )
    snapshot_digest = sha256_bytes("".join(sorted(snapshot_lines)).encode("utf-8"))
    dataset_version = f"frailty3_m2_20260815_{snapshot_digest[:16]}"
    for row in file_rows:
        row["dataset_version_id"] = dataset_version

    subject_rows: list[dict[str, Any]] = []
    ordered_subjects = sorted(subject_files, key=lambda value: value.encode("utf-8"))
    for subject_id in ordered_subjects:
        rows = sorted(
            subject_files[subject_id],
            key=lambda item: EXPECTED_ROLES.index(item["role"]),
        )
        roles = tuple(row["role"] for row in rows)
        if roles != EXPECTED_ROLES:
            raise RuntimeError(f"Role coverage mismatch for {subject_id}: {roles}")
        first = rows[0]
        subject_rows.append(
            {
                "dataset_version_id": dataset_version,
                "subject_id": subject_id,
                "cohort": first["cohort"],
                "class_id": first["class_id"],
                "class_name": first["class_name"],
                "class_source": first["class_source"],
                "label_record_id": first["label_record_id"],
                "n_files_all_roles": len(rows),
                "n_files_static_roles": sum(
                    row["role"] in STATIC_ROLES for row in rows
                ),
                "roles": ";".join(roles),
                "file_ids": ";".join(row["file_id"] for row in rows),
                "inclusion_status": "included",
                "inclusion_reason": "frozen_29_subject_roster",
            }
        )
    summary = {
        "dataset_version_id": dataset_version,
        "m2_snapshot_digest_algorithm": "sha256_sorted_relative_path_nul_size_nul_file_sha_newline",
        "m2_snapshot_sha256": snapshot_digest,
        "audited_label_sha256": AUDITED_LABEL_SHA256,
        "audited_raw_tree_sha256": AUDITED_RAW_TREE_SHA256,
        "audited_raw_plus_label_tree_sha256": AUDITED_RAW_LABEL_TREE_SHA256,
        "file_count": len(file_rows),
        "subject_count": len(subject_rows),
        "total_data_rows": sum(int(row["n_samples"]) for row in file_rows),
        "total_raw_bytes": sum(int(row["file_bytes"]) for row in file_rows),
        "class_subject_counts": dict(
            sorted(Counter(row["class_name"] for row in subject_rows).items())
        ),
        "role_file_counts": dict(
            sorted(Counter(row["role"] for row in file_rows).items())
        ),
        "shorter_than_30s_file_count": sum(
            "shorter_than_30s_window" in row["warning_codes"] for row in file_rows
        ),
    }
    return file_rows, subject_rows, summary


def evaluate_fold(fold_counts: np.ndarray, class_totals: np.ndarray) -> float:
    """复刻 SGKF 比例离散度目标；replicate SGKF's proportion objective."""

    proportions = fold_counts / class_totals.reshape(1, -1)
    return float(np.mean(np.std(proportions, axis=0)))


def assign_groups(
    subject_rows: list[dict[str, Any]], seed: int, corrected: bool
) -> list[list[str]]:
    """执行历史错误或修正 SGKF 分配；run buggy or corrected SGKF assignment."""

    ordered = sorted(
        subject_rows, key=lambda row: row["subject_id"].encode("utf-8")
    )
    group_ids = [str(row["subject_id"]) for row in ordered]
    counts = np.zeros((len(ordered), len(CLASS_NAMES)), dtype=np.float64)
    for index, row in enumerate(ordered):
        counts[index, int(row["class_id"])] = 1.0
    class_totals = counts.sum(axis=0)
    rng = np.random.RandomState(seed)
    if corrected:
        # 中文：关键修复：group ID 与 class-count row 共用同一个 permutation。
        # English: Critical fix: jointly permute group IDs and class-count rows.
        permutation = np.arange(len(group_ids))
        rng.shuffle(permutation)
        work_counts = counts[permutation]
        work_groups = [group_ids[int(index)] for index in permutation]
    else:
        # 中文：历史 1.4.2 只 shuffle counts，仍把 row index 当原 group ID。
        # English: Historical 1.4.2 shuffles counts but reuses original group IDs.
        work_counts = counts.copy()
        rng.shuffle(work_counts)
        work_groups = list(group_ids)

    order = np.argsort(-np.std(work_counts, axis=1), kind="mergesort")
    fold_counts = np.zeros((N_SPLITS, len(CLASS_NAMES)), dtype=np.float64)
    fold_groups: list[list[str]] = [[] for _ in range(N_SPLITS)]
    for row_index in order:
        best_fold = 0
        best_score = math.inf
        best_size = math.inf
        group_count = work_counts[int(row_index)]
        for fold_index in range(N_SPLITS):
            fold_counts[fold_index] += group_count
            score = evaluate_fold(fold_counts, class_totals)
            fold_counts[fold_index] -= group_count
            fold_size = float(fold_counts[fold_index].sum())
            tied = math.isclose(
                score, best_score, rel_tol=1e-12, abs_tol=1e-12
            )
            if score < best_score or (tied and fold_size < best_size):
                best_fold = fold_index
                best_score = score
                best_size = fold_size
        fold_counts[best_fold] += group_count
        fold_groups[best_fold].append(work_groups[int(row_index)])
    return [
        sorted(fold, key=lambda value: value.encode("utf-8"))
        for fold in fold_groups
    ]


def class_count(
    subject_ids: Iterable[str], classes: dict[str, int]
) -> dict[str, int]:
    """统计 subject 集合类别；count classes for a subject set."""

    count = Counter(CLASS_NAMES[classes[subject]] for subject in subject_ids)
    return {name: int(count.get(name, 0)) for name in CLASS_NAMES.values()}


def build_registry(
    subject_rows: list[dict[str, Any]],
    file_rows: list[dict[str, Any]],
    corrected: bool,
) -> dict[str, Any]:
    """物化五次五折 registry；materialize a five-repeat/five-fold registry."""

    subjects = [str(row["subject_id"]) for row in subject_rows]
    subject_set = set(subjects)
    classes = {
        str(row["subject_id"]): int(row["class_id"]) for row in subject_rows
    }
    files_by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in file_rows:
        files_by_subject[str(row["subject_id"])].append(row)
    split_input = [
        {"subject_id": subject, "class_id": classes[subject]}
        for subject in sorted(subject_set, key=lambda value: value.encode("utf-8"))
    ]
    repeats: list[dict[str, Any]] = []
    class_missing_folds = 0
    all_invariants_pass = True
    for repeat_index, seed in enumerate(SEEDS):
        oof_folds = assign_groups(subject_rows, seed, corrected=corrected)
        oof_sets = [set(fold) for fold in oof_folds]
        if set().union(*oof_sets) != subject_set:
            raise RuntimeError(f"OOF union failure for seed {seed}")
        if sum(len(fold) for fold in oof_folds) != len(subjects):
            raise RuntimeError(f"Duplicate OOF subject for seed {seed}")
        fold_records: list[dict[str, Any]] = []
        for fold_index, oof_subjects in enumerate(oof_folds):
            oof_set = set(oof_subjects)
            train_subjects = sorted(
                subject_set - oof_set, key=lambda value: value.encode("utf-8")
            )
            oof_counts = class_count(oof_subjects, classes)
            train_counts = class_count(train_subjects, classes)
            coverage = all(value > 0 for value in oof_counts.values())
            class_missing_folds += int(not coverage)
            all_invariants_pass &= not (set(train_subjects) & oof_set)
            fold_records.append(
                {
                    "fold_index": fold_index,
                    "fold_number": fold_index + 1,
                    "training_seed": seed + fold_index,
                    "train_subject_ids": train_subjects,
                    "oof_validation_subject_ids": oof_subjects,
                    "train_class_counts": train_counts,
                    "oof_validation_class_counts": oof_counts,
                    "all_three_classes_present": coverage,
                    "train_file_ids_all_roles": sorted(
                        row["file_id"]
                        for subject in train_subjects
                        for row in files_by_subject[subject]
                    ),
                    "oof_validation_file_ids_all_roles": sorted(
                        row["file_id"]
                        for subject in oof_subjects
                        for row in files_by_subject[subject]
                    ),
                    "oof_validation_file_ids_static_roles": sorted(
                        row["file_id"]
                        for subject in oof_subjects
                        for row in files_by_subject[subject]
                        if row["role"] in STATIC_ROLES
                    ),
                }
            )
        # 中文：未来主协议要求每类在五折中的数量差不超过 1。
        # English: Future folds require a per-class spread no greater than one.
        if corrected:
            for class_name in CLASS_NAMES.values():
                values = [
                    fold["oof_validation_class_counts"][class_name]
                    for fold in fold_records
                ]
                if max(values) - min(values) > 1:
                    all_invariants_pass = False
        repeats.append(
            {"repeat_index": repeat_index, "split_seed": seed, "folds": fold_records}
        )

    status = (
        "active_future_benchmark_only"
        if corrected
        else "historical_reproduction_only"
    )
    registry_id = (
        "frailty3_future_corrected_sgkf5_v2"
        if corrected
        else "frailty3_historical_sgkf5_sklearn142_bug_v1"
    )
    registry: dict[str, Any] = {
        "schema_version": "m2.fold_registry.v1",
        "registry_id": registry_id,
        "status": status,
        "dataset_version_id": subject_rows[0]["dataset_version_id"],
        "split_unit": "subject_id",
        "subject_input_order": "stable_utf8_bytewise",
        "subject_input_sha256": sha256_bytes(stable_json_bytes(split_input)),
        "class_label_map": {
            str(class_id): class_name for class_id, class_name in CLASS_NAMES.items()
        },
        "n_subjects": len(subject_rows),
        "n_splits": N_SPLITS,
        "n_repeats": len(SEEDS),
        "seeds": list(SEEDS),
        "training_seed_rule": "split_seed_plus_zero_based_fold_index",
        "available_roles": list(EXPECTED_ROLES),
        "current_static_evaluation_roles": sorted(STATIC_ROLES),
        "algorithm": (
            "corrected_sgkf_joint_group_count_permutation_then_greedy_balance"
            if corrected
            else "sklearn_1_4_2_sgkf_shuffle_counts_without_group_remap_reproduction"
        ),
        "semantic_parent": "sklearn.model_selection.StratifiedGroupKFold",
        "group_constraint": "all_files_and_windows_follow_subject_membership",
        "numpy_version": np.__version__,
        "runtime_split_recomputation_allowed": False,
        "all_candidates_must_rerun": bool(corrected),
        "class_missing_fold_count": class_missing_folds,
        "invariants_pass": bool(
            all_invariants_pass
            and (class_missing_folds == 0 if corrected else True)
        ),
        "repeats": repeats,
    }
    registry["payload_sha256"] = sha256_bytes(stable_json_bytes(registry))
    return registry


def verify_historical_against_sklearn(
    subject_rows: list[dict[str, Any]], registry: dict[str, Any]
) -> tuple[str, bool]:
    """对照本地 sklearn；cross-check the local sklearn implementation."""

    try:
        import sklearn
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError:
        return "not_installed", False
    ordered = sorted(
        subject_rows, key=lambda row: row["subject_id"].encode("utf-8")
    )
    y = np.asarray([int(row["class_id"]) for row in ordered])
    groups = np.asarray([str(row["subject_id"]) for row in ordered])
    x = np.zeros((len(ordered), 1), dtype=np.float64)
    for repeat in registry["repeats"]:
        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=int(repeat["split_seed"]),
        )
        observed = [
            sorted(groups[oof].tolist(), key=lambda value: value.encode("utf-8"))
            for _, oof in splitter.split(x, y, groups)
        ]
        expected = [
            fold["oof_validation_subject_ids"] for fold in repeat["folds"]
        ]
        if observed != expected:
            return sklearn.__version__, False
    return sklearn.__version__, True


def read_checksums(path: Path) -> dict[str, str]:
    """读取 PhysioNet SHA 清单；read a PhysioNet SHA manifest."""

    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        checksums[relative.strip().lstrip("*")] = digest
    return checksums


def build_external_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """构建外部 dataset/record manifests；build external manifests."""

    registry_path = (
        PACKAGE_ROOT / "registries" / "external_dataset_registry.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    dataset_rows = [dict(row) for row in registry["datasets"]]
    ptt_checksums = read_checksums(PTT_ROOT / "SHA256SUMS.txt")
    sim_checksums = read_checksums(SIM_ROOT / "SHA256SUMS.txt")
    records: list[dict[str, Any]] = []

    ptt_csv_root = PTT_ROOT / "csv"
    ptt_paths = [
        path
        for path in ptt_csv_root.glob("*.csv")
        if PTT_PATTERN.fullmatch(path.name)
    ]
    ptt_paths.sort(
        key=lambda path: (
            int(PTT_PATTERN.fullmatch(path.name).group("number")),
            ("sit", "walk", "run").index(
                PTT_PATTERN.fullmatch(path.name).group("activity")
            ),
        )
    )
    for path in ptt_paths:
        match = PTT_PATTERN.fullmatch(path.name)
        assert match is not None
        subject, activity = match.group(1), match.group("activity")
        stem = path.stem
        source_files = [
            f"csv/{path.name}",
            f"{stem}.hea",
            f"{stem}.dat",
            f"{stem}.atr",
        ]
        flags = (
            "known_noisy_ecg_and_possible_imu_single_sample_artifact"
            if stem == "s13_walk"
            else ""
        )
        records.append(
            {
                "dataset_id": "ptt_ppg_1_1_0_local",
                "record_id": stem,
                "subject_id": subject,
                "source_files": json.dumps(source_files, separators=(",", ":")),
                "canonical_representation": f"csv/{path.name}",
                "activity_raw": activity,
                "activity_binary": "static" if activity == "sit" else "motion",
                "activity_label_source": "record_name_official_protocol",
                "container_grid_fs_hz": 500,
                "channel_rate_detail": (
                    "PPG_acquisition_1000_to_export_500;ECG_IMU_500;"
                    "loadcell_80;temperature_10"
                ),
                "ppg_channels": "pleth_1..pleth_6",
                "ppg_placement": "left_index_finger_distal_and_proximal",
                "ppg_wavelength_status": "unresolved_red_ir_mapping_conflict",
                "ecg_channels": "three_lead_ra_la_ll_exported_ecg",
                "ecg_reference_type": "manually_verified_r_peaks",
                "imu_channels": "accelerometer_xyz;gyroscope_xyz",
                "imu_unit_status": (
                    "declared_g_but_values_and_code_inference_conflict"
                ),
                "checksum_sha256": ptt_checksums.get(f"csv/{path.name}", ""),
                "checksum_status": "official_entry_and_full_snapshot_audit_pass",
                "inclusion_status": "included",
                "inclusion_reason": (
                    "synchronized_ppg_ecg_imu_and_manual_verified_peaks"
                ),
                "known_quality_flags": flags,
            }
        )

    generated = SIM_ROOT / "generated_data"
    for hea in sorted(generated.glob("x*.hea"), key=lambda path: path.stem):
        stem = hea.stem
        text = hea.read_text(encoding="utf-8", errors="replace")
        has_pleth = "SOT/Pleth" in text
        components = [
            f"generated_data/{stem}{suffix}"
            for suffix in (".hea", ".dat", ".atr", ".aux")
            if (generated / f"{stem}{suffix}").is_file()
        ]
        component_hashes = {
            item: sim_checksums.get(item, "") for item in components
        }
        records.append(
            {
                "dataset_id": "simultaneous_measurements_1_0_0_local",
                "record_id": stem,
                "subject_id": stem if stem != "x001a" else "x001_variant",
                "source_files": json.dumps(components, separators=(",", ":")),
                "canonical_representation": f"generated_data/{stem}.hea",
                "activity_raw": (
                    "multi_stage_rest_walk_2back_uphill_walk"
                    if has_pleth
                    else "unavailable"
                ),
                "activity_binary": (
                    "interval_labeled_static_and_motion"
                    if has_pleth
                    else "unavailable"
                ),
                "activity_label_source": (
                    "manually_revised_aux_phase_markers"
                    if has_pleth
                    else "none"
                ),
                "container_grid_fs_hz": 256,
                "channel_rate_detail": (
                    "generated_256;SOT_native_512;FAROS_1000;NEXUS_8000;"
                    "POLAR_1000;HEXOSKIN_256"
                ),
                "ppg_channels": "SOT/Pleth" if has_pleth else "none",
                "ppg_placement": (
                    "right_hand_fingertip_schematic_exact_digit_unknown"
                    if has_pleth
                    else "none"
                ),
                "ppg_wavelength_status": (
                    "unknown" if has_pleth else "not_applicable"
                ),
                "ecg_channels": (
                    "SOT/EKG_filtered" if has_pleth else "none"
                ),
                "ecg_reference_type": (
                    "manually_revised_consensus_r_peaks"
                    if has_pleth
                    else "none"
                ),
                "imu_channels": (
                    "FAROS_accelerometer_xyz_no_gyroscope"
                    if has_pleth
                    else "none"
                ),
                "imu_unit_status": (
                    "mg_source_declared" if has_pleth else "not_applicable"
                ),
                "checksum_sha256": json.dumps(
                    component_hashes, sort_keys=True, separators=(",", ":")
                ),
                "checksum_status": "official_full_snapshot_audit_pass",
                "inclusion_status": "included" if has_pleth else "excluded",
                "inclusion_reason": (
                    "synchronized_ppg_ecg_accel_manual_peaks_and_phase_markers"
                    if has_pleth
                    else "missing_SOT_Pleth_and_required_annotations"
                ),
                "known_quality_flags": "",
            }
        )
    return dataset_rows, records


def build_source_anomalies(file_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总源异常和未知元数据；summarize source anomalies and unknowns."""

    durations: dict[str, list[float]] = defaultdict(list)
    short: Counter[str] = Counter()
    for row in file_rows:
        role = str(row["role"])
        durations[role].append(float(row["duration_seconds"]))
        if "shorter_than_30s_window" in row["warning_codes"]:
            short[role] += 1
    return {
        "schema_version": "m2.source_anomalies.v1",
        "label_source_conflicts": [
            {
                "active": "StudyData_V7_standard.csv:STE072",
                "deprecated_conflicting": [
                    "StudyData_V7_standard.xlsx:STE02",
                    "StudyData_V7.csv:STE02",
                ],
                "reason": (
                    "active CSV matches STE072_01 raw filenames and current code"
                ),
            }
        ],
        "label_only_ids_without_raw_files": [
            "BAE28",
            "NRE29",
            "PSR16",
            "PSS22",
        ],
        "duration_seconds_by_role": {
            role: {"min": min(values), "max": max(values)}
            for role, values in sorted(durations.items())
        },
        "shorter_than_30s_counts": dict(sorted(short.items())),
        "metadata_unknown": [
            "ppg_adc_unit_and_scale",
            "frailty3_ppg_wavelength_nm",
            "frailty3_ppg_placement",
            "frailty3_ppg_polarity",
            "R1_R4_exact_semantics_and_order",
            "S1_S2_W1_W2_exact_semantics_and_total_order",
        ],
        "reference_limit": (
            "frailty3_has_no_synchronized_ecg_or_manual_peak_ground_truth"
        ),
    }


def main() -> int:
    """生成全部 M2 机器产物；generate every M2 machine artifact."""

    file_rows, subject_rows, frailty_summary = scan_frailty_files()
    file_manifest_path = MANIFEST_ROOT / "frailty3_file_manifest.csv"
    subject_manifest_path = MANIFEST_ROOT / "frailty3_subject_manifest.csv"
    write_csv(file_manifest_path, file_rows, list(file_rows[0]))
    write_csv(subject_manifest_path, subject_rows, list(subject_rows[0]))
    write_json(
        MANIFEST_ROOT / "frailty3_source_anomalies.json",
        build_source_anomalies(file_rows),
    )

    historical = build_registry(subject_rows, file_rows, corrected=False)
    future = build_registry(subject_rows, file_rows, corrected=True)
    sklearn_version, historical_matches = verify_historical_against_sklearn(
        subject_rows, historical
    )
    historical["local_sklearn_version"] = sklearn_version
    historical["local_sklearn_membership_exact_match"] = historical_matches
    historical["payload_sha256"] = sha256_bytes(
        stable_json_bytes(
            {
                key: value
                for key, value in historical.items()
                if key != "payload_sha256"
            }
        )
    )
    if sklearn_version == "1.4.2" and not historical_matches:
        raise RuntimeError(
            "Historical implementation does not match local sklearn 1.4.2"
        )
    if historical["class_missing_fold_count"] != 6:
        count = historical["class_missing_fold_count"]
        raise RuntimeError(f"Historical missing-class fold drift: {count} != 6")
    if not future["invariants_pass"] or future["class_missing_fold_count"] != 0:
        raise RuntimeError("Corrected future registry failed invariants")
    historical_path = (
        SPLIT_ROOT / "frailty3_historical_sgkf5_sklearn142_bug_v1.json"
    )
    future_path = SPLIT_ROOT / "frailty3_future_corrected_sgkf5_v2.json"
    write_json(historical_path, historical)
    write_json(future_path, future)

    dataset_rows, external_records = build_external_records()
    dataset_fields = sorted({key for row in dataset_rows for key in row})
    write_csv(
        MANIFEST_ROOT / "external_dataset_manifest.csv",
        dataset_rows,
        dataset_fields,
    )
    write_csv(
        MANIFEST_ROOT / "external_record_manifest.csv",
        external_records,
        list(external_records[0]),
    )

    file_manifest_sha = sha256_bytes(file_manifest_path.read_bytes())
    example_fold = future["repeats"][0]["folds"][0]
    example = {
        "schema_version": "m2.result_provenance.v1",
        "result_status": "template_not_run",
        "dataset_version_id": frailty_summary["dataset_version_id"],
        "dataset_manifest_sha256": file_manifest_sha,
        "fold_registry_id": future["registry_id"],
        "fold_registry_sha256": future["payload_sha256"],
        "protocol_id": "frailty3_fixed_epoch_oof_v2_corrected_sgkf",
        "repeat_index": 0,
        "split_seed": SEEDS[0],
        "fold_index": 0,
        "training_seed": SEEDS[0],
        "train_subject_ids": example_fold["train_subject_ids"],
        "oof_validation_subject_ids": example_fold[
            "oof_validation_subject_ids"
        ],
        "fixed_epoch": "MUST_BE_LOCKED_BEFORE_RUN",
        "early_stopping": False,
        "outer_oof_visible_during_training": False,
        "preprocessing_version": "M3_NOT_YET_FROZEN_EXAMPLE_ONLY",
        "feature_schema_version": "M4_ROUTE_SPECIFIC_NOT_YET_FROZEN",
        "config_hash": "TEMPLATE_NOT_A_RESULT",
        "evaluation_role": "oof_validation",
        "metric_prefix": "oof_validation_",
        "independent_test": False,
        "metrics": {},
        "coverage": {},
    }
    write_json(
        EXAMPLE_ROOT / "result_provenance_fixed_epoch_oof_template.json",
        example,
    )

    ptt_included = sum(
        row["dataset_id"] == "ptt_ppg_1_1_0_local"
        and row["inclusion_status"] == "included"
        for row in external_records
    )
    sim_included = sum(
        row["dataset_id"] == "simultaneous_measurements_1_0_0_local"
        and row["inclusion_status"] == "included"
        for row in external_records
    )
    sim_excluded = sum(
        row["dataset_id"] == "simultaneous_measurements_1_0_0_local"
        and row["inclusion_status"] == "excluded"
        for row in external_records
    )
    build_report = {
        "schema_version": "m2.build_report.v1",
        "status": "pass",
        "date": "2026-08-15",
        "write_boundary": relative_repo(PACKAGE_ROOT),
        "source_policy": "read_only_PPG_Testing_and_physionet",
        "generator_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "subject_manifest_sha256": sha256_bytes(subject_manifest_path.read_bytes()),
        "source_permission_snapshot": {
            "PPG_Testing_05_01_2026": permission_snapshot(FRAILTY_ROOT),
            "physionet.org": permission_snapshot(REPO_ROOT / "physionet.org"),
        },
        "frailty3": frailty_summary,
        "numeric_scan": {
            "method": (
                "full_file_bytes_sha256_exact_header_numpy_all_tokens_finite"
            ),
            "files_passed": len(file_rows),
            "values_parsed": (
                frailty_summary["total_data_rows"] * len(EXPECTED_HEADER)
            ),
            "files_failed": 0,
        },
        "fold_registries": {
            "historical_registry_id": historical["registry_id"],
            "historical_payload_sha256": historical["payload_sha256"],
            "historical_class_missing_folds": historical[
                "class_missing_fold_count"
            ],
            "historical_matches_local_sklearn": historical_matches,
            "local_sklearn_version": sklearn_version,
            "future_registry_id": future["registry_id"],
            "future_payload_sha256": future["payload_sha256"],
            "future_class_missing_folds": future[
                "class_missing_fold_count"
            ],
            "future_invariants_pass": future["invariants_pass"],
            "seeds": list(SEEDS),
            "repeats": len(SEEDS),
            "folds": N_SPLITS,
        },
        "external_manifest": {
            "dataset_rows": len(dataset_rows),
            "record_rows": len(external_records),
            "included_ptt_records": ptt_included,
            "included_sim_records": sim_included,
            "excluded_sim_records": sim_excluded,
        },
        "generated_files": [
            "manifests/frailty3_file_manifest.csv",
            "manifests/frailty3_subject_manifest.csv",
            "manifests/frailty3_source_anomalies.json",
            "manifests/external_dataset_manifest.csv",
            "manifests/external_record_manifest.csv",
            "splits/frailty3_historical_sgkf5_sklearn142_bug_v1.json",
            "splits/frailty3_future_corrected_sgkf5_v2.json",
            "examples/result_provenance_fixed_epoch_oof_template.json",
            "M2_BUILD_REPORT.json",
        ],
    }
    write_json(BUILD_REPORT, build_report)
    print(json.dumps(build_report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    """保留稳定 CLI；retain a stable CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
    raise SystemExit(main())
