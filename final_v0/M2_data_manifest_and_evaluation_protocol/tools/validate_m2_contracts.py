#!/usr/bin/env python3
"""验证 M2 manifests、双 fold registry、协议和结果溯源合同。

Validate M2 manifests, both fold registries, protocol semantics, and result
provenance. With --write-report, writes only this M2 package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


# 中文：验证器与生成器使用相同写入边界；English: share one write boundary.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = FINAL_ROOT.parent
REPORT_PATH = PACKAGE_ROOT / "M2_CONTRACT_VERIFICATION.json"
TREE_PATH = PACKAGE_ROOT / "M2_PACKAGE_TREE.md"
SEEDS = [42, 10042, 20042, 30042, 40042]
ROLES = ["B", "R1", "R2", "R3", "R4", "S1", "S2", "W1", "W2"]
CLASS_NAMES = ["pre_frail", "robust_non_frail", "young"]


def sha256_bytes(payload: bytes) -> str:
    """返回 SHA-256；return a SHA-256 digest."""

    return hashlib.sha256(payload).hexdigest()


def stable_json_bytes(value: Any) -> bytes:
    """生成与 builder 相同的 JSON；render builder-compatible strict JSON."""

    text = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    )
    return (text + "\n").encode("utf-8")


def checked_target(path: Path) -> Path:
    """拒绝 M2 包外写入；reject writes outside the M2 package."""

    target = path.resolve(strict=False)
    try:
        target.relative_to(PACKAGE_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Refusing write outside M2 package: {target}") from exc
    return target


def atomic_write(path: Path, payload: bytes) -> None:
    """原子写验证产物；atomically write a verification artifact."""

    target = checked_target(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)


def load_json(relative: str) -> Any:
    """读取包内 JSON；read package-local JSON."""

    return json.loads((PACKAGE_ROOT / relative).read_text(encoding="utf-8"))


def load_csv(relative: str) -> list[dict[str, str]]:
    """读取包内 CSV；read a package-local CSV."""

    with (PACKAGE_ROOT / relative).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def check(
    condition: bool,
    rule: str,
    detail: str,
    checks: list[dict[str, Any]],
    failures: list[dict[str, str]],
) -> None:
    """记录一条检查；record one validation check."""

    checks.append({"rule": rule, "status": "pass" if condition else "fail"})
    if not condition:
        failures.append({"rule": rule, "detail": detail})


def validate_manifests(
    checks: list[dict[str, Any]], failures: list[dict[str, str]]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """验证 Frailty3 manifests；validate Frailty3 manifests."""

    files = load_csv("manifests/frailty3_file_manifest.csv")
    subjects = load_csv("manifests/frailty3_subject_manifest.csv")
    check(len(files) == 261, "frailty_file_count_261", str(len(files)), checks, failures)
    check(len(subjects) == 29, "frailty_subject_count_29", str(len(subjects)), checks, failures)
    check(
        len({row["file_id"] for row in files}) == 261,
        "frailty_file_ids_unique", "duplicate file_id", checks, failures,
    )
    check(
        len({row["subject_id"] for row in subjects}) == 29,
        "frailty_subject_ids_unique", "duplicate subject_id", checks, failures,
    )
    role_counts = Counter(row["role"] for row in files)
    check(
        role_counts == Counter({role: 29 for role in ROLES}),
        "each_role_has_29_files", repr(role_counts), checks, failures,
    )
    class_counts = Counter(row["class_name"] for row in subjects)
    expected = Counter({"pre_frail": 9, "robust_non_frail": 12, "young": 8})
    check(
        class_counts == expected,
        "subject_class_counts_9_12_8", repr(class_counts), checks, failures,
    )
    by_subject: dict[str, list[str]] = {}
    for row in files:
        by_subject.setdefault(row["subject_id"], []).append(row["role"])
    check(
        all(set(values) == set(ROLES) and len(values) == 9 for values in by_subject.values()),
        "each_subject_has_exact_nine_roles", "role coverage mismatch", checks, failures,
    )
    check(
        all(row["numeric_full_scan"] == "passed_finite_8_columns" for row in files),
        "all_numeric_full_scans_pass", "numeric scan failure", checks, failures,
    )
    check(
        sum(int(row["n_samples"]) for row in files) == 18152248,
        "frailty_data_rows_18152248", "data row drift", checks, failures,
    )
    check(
        sum(int(row["file_bytes"]) for row in files) == 969008078,
        "frailty_raw_bytes_969008078", "byte count drift", checks, failures,
    )
    versions = {row["dataset_version_id"] for row in files + subjects}
    check(
        len(versions) == 1
        and next(iter(versions)).startswith("frailty3_m2_20260815_"),
        "single_dataset_version", repr(versions), checks, failures,
    )
    source_paths = [REPO_ROOT / row["relative_path"] for row in files]
    check(
        all(path.is_file() for path in source_paths),
        "all_manifest_sources_exist", "missing source file", checks, failures,
    )
    check(
        all("final_v0" not in path.parts for path in source_paths),
        "source_paths_outside_final_v0", "source inside final_v0", checks, failures,
    )
    check(
        all(row["reference_available"] == "false" for row in files),
        "frailty_reference_unavailable_explicit",
        "Frailty3 reference incorrectly claimed", checks, failures,
    )
    return files, subjects


def verify_registry_hash(registry: dict[str, Any]) -> bool:
    """验证 registry payload hash；verify the registry payload digest."""

    payload = {
        key: value for key, value in registry.items() if key != "payload_sha256"
    }
    return sha256_bytes(stable_json_bytes(payload)) == registry["payload_sha256"]


def validate_one_registry(
    registry: dict[str, Any],
    subject_ids: set[str],
    future: bool,
    checks: list[dict[str, Any]],
    failures: list[dict[str, str]],
) -> None:
    """验证一套 5×5 membership；validate one materialized 5x5 membership."""

    prefix = "future" if future else "historical"
    check(registry["n_splits"] == 5, f"{prefix}_five_folds", "not 5", checks, failures)
    check(registry["n_repeats"] == 5, f"{prefix}_five_repeats", "not 5", checks, failures)
    check(registry["seeds"] == SEEDS, f"{prefix}_fixed_seeds", repr(registry["seeds"]), checks, failures)
    check(
        len(registry["subject_input_sha256"]) == 64,
        f"{prefix}_subject_input_hash",
        registry["subject_input_sha256"], checks, failures,
    )
    check(
        registry["semantic_parent"]
        == "sklearn.model_selection.StratifiedGroupKFold",
        f"{prefix}_sgkf_semantic_parent",
        registry["semantic_parent"], checks, failures,
    )
    check(verify_registry_hash(registry), f"{prefix}_payload_hash", "hash mismatch", checks, failures)
    check(len(registry["repeats"]) == 5, f"{prefix}_repeat_records", "wrong repeat count", checks, failures)
    for repeat in registry["repeats"]:
        seed = int(repeat["split_seed"])
        seen: list[str] = []
        class_fold_counts = {name: [] for name in CLASS_NAMES}
        check(len(repeat["folds"]) == 5, f"{prefix}_seed_{seed}_five_folds", "wrong folds", checks, failures)
        for fold in repeat["folds"]:
            train = set(fold["train_subject_ids"])
            oof = set(fold["oof_validation_subject_ids"])
            check(
                not (train & oof),
                f"{prefix}_{seed}_{fold['fold_index']}_disjoint",
                "train/oof overlap", checks, failures,
            )
            check(
                train | oof == subject_ids,
                f"{prefix}_{seed}_{fold['fold_index']}_complete",
                "fold union drift", checks, failures,
            )
            seen.extend(fold["oof_validation_subject_ids"])
            for name in CLASS_NAMES:
                class_fold_counts[name].append(
                    int(fold["oof_validation_class_counts"][name])
                )
            if future:
                check(
                    fold["all_three_classes_present"] is True,
                    f"future_{seed}_{fold['fold_index']}_all_classes",
                    repr(fold["oof_validation_class_counts"]), checks, failures,
                )
        check(
            len(seen) == len(subject_ids) and set(seen) == subject_ids,
            f"{prefix}_seed_{seed}_oof_partition",
            "OOF is not an exact partition", checks, failures,
        )
        if future:
            for name, values in class_fold_counts.items():
                check(
                    max(values) - min(values) <= 1,
                    f"future_seed_{seed}_{name}_balanced",
                    repr(values), checks, failures,
                )
    expected_missing = 0 if future else 6
    check(
        registry["class_missing_fold_count"] == expected_missing,
        f"{prefix}_missing_class_fold_count_{expected_missing}",
        str(registry["class_missing_fold_count"]), checks, failures,
    )


def validate_folds(
    subjects: list[dict[str, str]],
    checks: list[dict[str, Any]],
    failures: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """验证双注册表；validate both fold registries."""

    historical = load_json(
        "splits/frailty3_historical_sgkf5_sklearn142_bug_v1.json"
    )
    future = load_json("splits/frailty3_future_corrected_sgkf5_v2.json")
    subject_ids = {row["subject_id"] for row in subjects}
    validate_one_registry(historical, subject_ids, False, checks, failures)
    validate_one_registry(future, subject_ids, True, checks, failures)
    check(
        historical.get("local_sklearn_version") == "1.4.2"
        and historical.get("local_sklearn_membership_exact_match") is True,
        "historical_exact_sklearn142_reproduction",
        "local sklearn reproduction failed", checks, failures,
    )
    check(
        historical["registry_id"] != future["registry_id"],
        "dual_registry_ids_distinct", "registry IDs collide", checks, failures,
    )
    return historical, future


def validate_stage_external_protocol(
    checks: list[dict[str, Any]], failures: list[dict[str, str]]
) -> None:
    """验证阶段、外部数据和协议；validate stage/external/protocol contracts."""

    stage = load_json("registries/stage_role_registry.json")
    families = {row["family"]: row for row in stage["roles"]}
    observed = {key: families[key]["stage_name"] for key in families}
    expected = {
        "B": "baseline",
        "R": "relax_or_recovery",
        "S": "stand_and_sit",
        "W": "walk",
    }
    check(
        observed == expected,
        "confirmed_stage_family_mapping", repr(observed), checks, failures,
    )
    partial = stage["confirmed_partial_order"]
    check(
        len(partial) == 1
        and partial[0]["before_families"] == ["S", "W"]
        and partial[0]["after_family"] == "R",
        "only_confirmed_partial_order_S_W_before_R",
        repr(partial), checks, failures,
    )
    check(
        all(
            families[key]["instance_index_semantics"] == "unverified"
            for key in ("R", "S", "W")
        ),
        "indexed_stage_semantics_unverified",
        "an index was over-interpreted", checks, failures,
    )

    external_datasets = load_csv("manifests/external_dataset_manifest.csv")
    check(
        len(external_datasets) == 5,
        "external_dataset_rows_5", str(len(external_datasets)), checks, failures,
    )
    external = load_csv("manifests/external_record_manifest.csv")
    ptt = [
        row for row in external if row["dataset_id"] == "ptt_ppg_1_1_0_local"
    ]
    sim = [
        row for row in external
        if row["dataset_id"] == "simultaneous_measurements_1_0_0_local"
    ]
    check(len(ptt) == 66, "external_ptt_66_records", str(len(ptt)), checks, failures)
    sim_counts = Counter(row["inclusion_status"] for row in sim)
    check(
        sim_counts["included"] == 13,
        "external_sim_13_included", repr(sim_counts), checks, failures,
    )
    check(
        sim_counts["excluded"] == 1,
        "external_sim_x001a_excluded", repr(sim_counts), checks, failures,
    )
    check(
        all("unresolved" in row["ppg_wavelength_status"] for row in ptt),
        "ptt_wavelength_conflict_not_hidden",
        "PTT wavelength asserted", checks, failures,
    )
    check(
        all(row["ecg_reference_type"] == "manually_verified_r_peaks" for row in ptt),
        "ptt_manual_reference_explicit", "PTT reference drift", checks, failures,
    )
    protocol = load_json("registries/protocol_registry.json")
    active = next(
        item for item in protocol["protocols"]
        if item["protocol_id"] == protocol["active_protocol_id"]
    )
    check(
        active["fold_registry_id"] == "frailty3_future_corrected_sgkf5_v2",
        "active_protocol_uses_future_registry", repr(active), checks, failures,
    )
    check(
        active["early_stopping"] is False
        and active["outer_oof_visible_during_training"] is False,
        "active_fixed_epoch_no_outer_oof_training", repr(active), checks, failures,
    )
    check(
        active["metric_prefix"] == "oof_validation_",
        "active_oof_naming", active["metric_prefix"], checks, failures,
    )
    example = load_json(
        "examples/result_provenance_fixed_epoch_oof_template.json"
    )
    check(
        example["evaluation_role"] == "oof_validation"
        and example["independent_test"] is False,
        "example_not_independent_test", repr(example), checks, failures,
    )
    check(
        not any(key.startswith("test_") for key in example),
        "example_has_no_test_prefix", repr(list(example)), checks, failures,
    )


def validate_contracts(require_generated: bool = False) -> dict[str, Any]:
    """运行全部检查；run the complete validation suite."""

    checks: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    required = [
        "README.md",
        "00_CURRENT_STATUS.md",
        "01_DATASET_MANIFEST_AND_PROVENANCE.md",
        "02_STAGE_ROLE_MAPPING.md",
        "03_DUAL_FOLD_REGISTRY_AND_MAIN_PROTOCOL.md",
        "04_EXTERNAL_SYNCHRONIZED_DATA_MANIFEST.md",
        "05_RESULT_PROVENANCE_AND_NAMING_CONTRACT.md",
        "schemas/dataset_manifest.schema.json",
        "schemas/fold_registry.schema.json",
        "schemas/result_provenance.schema.json",
        "M2_BUILD_REPORT.json",
    ]
    for relative in required:
        check(
            (PACKAGE_ROOT / relative).is_file(),
            f"required_{relative}", "missing", checks, failures,
        )
    for relative in [item for item in required if item.endswith(".json")]:
        try:
            load_json(relative)
            valid = True
        except (OSError, json.JSONDecodeError):
            valid = False
        check(
            valid, f"json_parse_{relative}", "invalid JSON", checks, failures,
        )
    files, subjects = validate_manifests(checks, failures)
    _, future = validate_folds(subjects, checks, failures)
    validate_stage_external_protocol(checks, failures)
    build = load_json("M2_BUILD_REPORT.json")
    check(
        build["status"] == "pass",
        "build_report_pass", repr(build), checks, failures,
    )
    permissions = build["source_permission_snapshot"]
    check(
        all(
            item["files_with_write_bits"] == 0
            and item["directories_with_write_bits"] == 0
            for item in permissions.values()
        ),
        "raw_source_trees_have_no_write_bits",
        repr(permissions), checks, failures,
    )
    check(
        build["fold_registries"]["future_payload_sha256"]
        == future["payload_sha256"],
        "build_report_future_hash_matches",
        "future hash drift", checks, failures,
    )
    if require_generated:
        for path in (REPORT_PATH, TREE_PATH):
            check(
                path.is_file(),
                f"generated_{path.name}",
                "missing generated output", checks, failures,
            )
    return {
        "schema_version": "m2.contract_verification.v1",
        "status": "pass" if not failures else "fail",
        "check_count": len(checks),
        "pass_count": sum(item["status"] == "pass" for item in checks),
        "failure_count": len(failures),
        "failures": failures,
        "dataset_version_id": files[0]["dataset_version_id"],
        "active_fold_registry_id": future["registry_id"],
        "active_fold_registry_sha256": future["payload_sha256"],
        "checks": checks,
    }


def render_tree() -> str:
    """渲染包内完整性树；render the package integrity tree."""

    excluded = {REPORT_PATH, TREE_PATH}
    files = sorted(
        path
        for path in PACKAGE_ROOT.rglob("*")
        if path.is_file() and path not in excluded
    )
    lines = [
        "# M2 包文件树与完整性 / Package Tree and Integrity",
        "",
        "> Generated by tools/validate_m2_contracts.py --write-report; self hashes are omitted.",
        "",
        "| File | Bytes | SHA-256 | Content |",
        "|---|---:|---|---|",
    ]
    for path in files:
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        payload = path.read_bytes()
        content = "M2 package artifact"
        if relative.startswith("manifests/"):
            content = "dataset or record manifest"
        elif relative.startswith("splits/"):
            content = "materialized subject fold registry"
        elif relative.startswith("schemas/"):
            content = "machine-readable contract"
        elif relative.startswith("tools/"):
            content = "bilingual build/validation tool"
        lines.append(
            f"| {relative} | {len(payload)} | {sha256_bytes(payload)} | {content} |"
        )
    lines.extend(
        [
            "| M2_CONTRACT_VERIFICATION.json | self | omitted | machine validation report |",
            "| M2_PACKAGE_TREE.md | self | omitted | this integrity tree |",
            "",
            f"- 永久文件 / Permanent files：{len(files) + 2}。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """解析 CLI；parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true")
    return parser.parse_args()


def main() -> int:
    """验证并返回确定退出码；validate and return a deterministic exit code."""

    args = parse_args()
    if args.write_report:
        preflight = validate_contracts(require_generated=False)
        atomic_write(TREE_PATH, render_tree().encode("utf-8"))
        atomic_write(REPORT_PATH, stable_json_bytes({"status": "generating"}))
        report = validate_contracts(require_generated=True)
        report["preflight_status"] = preflight["status"]
        atomic_write(REPORT_PATH, stable_json_bytes(report))
    else:
        report = validate_contracts(require_generated=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
