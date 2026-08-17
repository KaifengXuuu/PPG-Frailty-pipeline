#!/usr/bin/env python3
"""物化 V1 数据与协议合同 / Materialize V1 data and protocol contracts.

中文：此工具只从 M2 已冻结的 manifest、corrected fold JSON 和协议注册表导入。
内部 261 个来源文件会逐一重新计算 SHA-256；outer split 只复制冻结 membership，
绝不调用运行时 splitter。外部 PTT/SIM 保留原始语义，provisional PTT 五折只作为
待 V2 人工确认的 grouped-CV 资产，不宣称独立 test。

English: This tool imports only the frozen M2 manifests, corrected fold JSON, and
protocol registry. All 261 internal source files are re-hashed; outer memberships
are copied without invoking a runtime splitter. External PTT/SIM semantics remain
unchanged, and the provisional PTT five-fold registry is a grouped-CV asset pending
V2 human confirmation, never an independent-test claim.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PIPELINE_ROOT.parents[1]
SRC_ROOT = PIPELINE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ppg_frailty.data import (  # noqa: E402 - local src path is deliberate.
    QCReason,
    audit_external_manifest,
    audit_manifest,
    build_external_manifest,
    build_internal_manifest,
    load_frozen_memberships,
    materialize_fold_csvs,
    materialize_provisional_external_grouped_split,
)
from ppg_frailty.data.external_manifest import (  # noqa: E402
    INDEPENDENCE_CLAIM,
    M2_EXTERNAL_MANIFEST_SHA256,
    M2_EXTERNAL_RELATIVE_PATH,
    PROVISIONAL_EXTERNAL_SPLIT_REGISTRY_ID,
)
from ppg_frailty.data.folds import (  # noqa: E402
    M2_SEEDS,
    M2_SPLIT_FILE_SHA256,
    M2_SPLIT_PAYLOAD_SHA256,
    M2_SPLIT_REGISTRY_ID,
    M2_SPLIT_RELATIVE_PATH,
    validate_frozen_memberships,
)
from ppg_frailty.data.manifest import (  # noqa: E402
    M2_DATASET_VERSION_ID,
    M2_FILE_MANIFEST,
    M2_FILE_MANIFEST_SHA256,
)
from ppg_frailty.provenance import (  # noqa: E402
    atomic_write_json,
    sha256_file,
)


SPEC_RELATIVE_PATH = Path(
    "AA_TODO/3/CODEX_IMPLEMENTATION_SPEC_PPG_FRAILTY_DEV0_MERGED.md"
)
SPEC_SHA256 = (
    "cd7c4907a8beccd301048dea07ae6fdc4e9d2dc839759cb8dddd4a461e3c5000"
)
PROTOCOL_REGISTRY_RELATIVE_PATH = Path(
    "final_v0/M2_data_manifest_and_evaluation_protocol/"
    "registries/protocol_registry.json"
)
PROTOCOL_REGISTRY_SHA256 = (
    "beae2a6922ae0ca840cec1a5c501cde6b6fc029afed16fc798aa2ef8e05fa394"
)
ACTIVE_PROTOCOL_ID = "frailty3_fixed_epoch_oof_v2_corrected_sgkf"
HISTORICAL_SGKF_RELATIVE_PATH = Path(
    "final_v0/M2_data_manifest_and_evaluation_protocol/"
    "splits/frailty3_historical_sgkf5_sklearn142_bug_v1.json"
)
LEGACY_EXTERNAL_SPLIT_RELATIVE_PATH = Path(
    "results_hybrid_denoiser/splits.json"
)
LEGACY_EXTERNAL_SPLIT_SHA256 = (
    "6350387f086dfb289b541ff61832572d55a0bc33fa7b6fc0a2428aaec61c687f"
)
PRODUCER_RELATIVE_PATHS = (
    Path("src/ppg_frailty/provenance.py"),
    Path("src/ppg_frailty/data/schema.py"),
    Path("src/ppg_frailty/data/manifest.py"),
    Path("src/ppg_frailty/data/external_manifest.py"),
    Path("src/ppg_frailty/data/qc.py"),
    Path("src/ppg_frailty/data/folds.py"),
    Path("src/ppg_frailty/data/windows.py"),
    Path("src/ppg_frailty/data/cache.py"),
    Path("tools/materialize_data_contracts.py"),
)


def _verified_sha(path: Path, expected: str, *, identity: str) -> str:
    """逐字节验证权威文件 / Byte-verify one authority file."""

    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{identity} SHA drift: {observed} != {expected}")
    return observed


def _load_active_protocol() -> dict[str, Any]:
    """加载并严验活动协议 / Load and strictly validate the active protocol."""

    path = REPOSITORY_ROOT / PROTOCOL_REGISTRY_RELATIVE_PATH
    _verified_sha(path, PROTOCOL_REGISTRY_SHA256, identity="protocol registry")
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("active_protocol_id") != ACTIVE_PROTOCOL_ID:
        raise ValueError("active protocol ID drift")
    matches = [
        item
        for item in registry.get("protocols", [])
        if item.get("protocol_id") == ACTIVE_PROTOCOL_ID
    ]
    if len(matches) != 1:
        raise ValueError("active protocol must resolve exactly once")
    protocol = matches[0]
    required = {
        "status": "active_future_benchmark_only",
        "family": "fixed_epoch_oof_cv",
        "fold_registry_id": M2_SPLIT_REGISTRY_ID,
        "folds": 5,
        "repeats": 5,
        "seeds": list(M2_SEEDS),
        "subject_grouped": True,
        "early_stopping": False,
        "outer_oof_visible_during_training": False,
        "metric_prefix": "oof_validation_",
        "independent_test_claim_allowed": False,
    }
    for key, expected in required.items():
        if protocol.get(key) != expected:
            raise ValueError(f"active protocol invariant drift: {key}")
    return protocol


def _artifact(path: Path) -> dict[str, object]:
    """生成文件 provenance 单元 / Build one artifact provenance entry."""

    return {
        "path": path.relative_to(PIPELINE_ROOT).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _producer_sources() -> dict[str, str]:
    """哈希全部数据合同生产者 / Hash every data-contract producer source."""

    return {
        relative.as_posix(): sha256_file(PIPELINE_ROOT / relative)
        for relative in PRODUCER_RELATIVE_PATHS
    }


def main() -> int:
    """物化全部数据合同并 fail closed / Materialize all contracts fail closed."""

    _verified_sha(
        REPOSITORY_ROOT / SPEC_RELATIVE_PATH,
        SPEC_SHA256,
        identity="implementation specification",
    )
    active_protocol = _load_active_protocol()

    internal_output = PIPELINE_ROOT / "manifests/internal_records_v1.csv"
    external_output = PIPELINE_ROOT / "manifests/external_records_v1.csv"
    primary_split_output = PIPELINE_ROOT / "splits/sgkf5_v1.csv"
    repeated_split_output = PIPELINE_ROOT / "splits/sgkf5_repeats_v1.csv"
    provisional_output = (
        PIPELINE_ROOT
        / "splits/v1_provisional_external_grouped_split_seed42.csv"
    )
    internal_report_output = PIPELINE_ROOT / "reports/data_contract_report.json"
    external_report_output = (
        PIPELINE_ROOT / "reports/external_data_contract_report.json"
    )

    # 中文：先撤销旧 pass 状态。若任一后续步骤失败，磁盘上只能看到明确的
    # incomplete 状态，不能把上一次成功报告误配给本次部分输出。
    # English: Revoke any prior pass state first. If a later step fails, disk
    # state is explicitly incomplete rather than a stale success over partial files.
    incomplete = {
        "schema_version": "ppg_frailty.materialization_state.v1",
        "status": "materializing_incomplete_fail_closed",
        "materializer_sha256": sha256_file(__file__),
    }
    atomic_write_json(
        internal_report_output,
        incomplete,
        root=PIPELINE_ROOT,
    )
    atomic_write_json(
        external_report_output,
        incomplete,
        root=PIPELINE_ROOT,
    )

    # 中文：build_internal_manifest 内部会重哈希全部 261 个来源文件。
    # English: build_internal_manifest re-hashes all 261 source recordings.
    internal_rows = build_internal_manifest(
        REPOSITORY_ROOT / M2_FILE_MANIFEST,
        internal_output,
    )
    external_rows = build_external_manifest(
        REPOSITORY_ROOT / M2_EXTERNAL_RELATIVE_PATH,
        external_output,
    )
    frozen_registry = load_frozen_memberships(
        REPOSITORY_ROOT / M2_SPLIT_RELATIVE_PATH
    )
    fold_audit = validate_frozen_memberships(
        frozen_registry,
        internal_rows,
    )
    primary_assignments, repeated_assignments = materialize_fold_csvs(
        frozen_registry,
        internal_rows,
        primary_split_output,
        repeated_split_output,
        output_root=PIPELINE_ROOT,
    )
    provisional_rows = materialize_provisional_external_grouped_split(
        external_rows,
        provisional_output,
        output_root=PIPELINE_ROOT,
    )

    legacy_path = REPOSITORY_ROOT / LEGACY_EXTERNAL_SPLIT_RELATIVE_PATH
    legacy_sha = _verified_sha(
        legacy_path,
        LEGACY_EXTERNAL_SPLIT_SHA256,
        identity="legacy external 15/3/4 split",
    )
    historical_sgkf_path = REPOSITORY_ROOT / HISTORICAL_SGKF_RELATIVE_PATH
    historical_sgkf_sha = sha256_file(historical_sgkf_path)

    internal_report = {
        "schema_version": "ppg_frailty.data_contract_report.v1",
        "status": "pass",
        "spec_authority": {
            "path": SPEC_RELATIVE_PATH.as_posix(),
            "sha256": SPEC_SHA256,
            "mapped_sections": [
                "2_invariants",
                "5.2_manifest_qc_frozen_folds",
                "5.3_unified_window_plan",
                "5.12_oof_contract",
                "5.13_hierarchical_aggregation",
                "5.14_provenance_cache",
                "8_testing",
                "9_no_go_rules",
            ],
        },
        "internal_manifest_authority": {
            "dataset_version_id": M2_DATASET_VERSION_ID,
            "path": M2_FILE_MANIFEST.as_posix(),
            "sha256": M2_FILE_MANIFEST_SHA256,
            "source_file_rehash_count": len(internal_rows),
            "source_file_rehash_status": "all_pass",
            "silent_skip_allowed": False,
        },
        "internal_manifest_audit": audit_manifest(internal_rows),
        "frozen_fold_authority": {
            "registry_id": M2_SPLIT_REGISTRY_ID,
            "path": M2_SPLIT_RELATIVE_PATH.as_posix(),
            "file_sha256": M2_SPLIT_FILE_SHA256,
            "payload_sha256": M2_SPLIT_PAYLOAD_SHA256,
            "runtime_split_recomputation_allowed": False,
            "primary_assignment_count": len(primary_assignments),
            "repeated_assignment_count": len(repeated_assignments),
            "audit": asdict(fold_audit),
        },
        "active_evaluation_protocol": {
            "protocol_registry_path": PROTOCOL_REGISTRY_RELATIVE_PATH.as_posix(),
            "protocol_registry_sha256": PROTOCOL_REGISTRY_SHA256,
            **active_protocol,
        },
        "qc_contract": {
            "silent_skip_allowed": False,
            "reason_codes": sorted(reason.value for reason in QCReason),
            "thresholds_have_hidden_defaults": False,
            "parser_failure_is_explicit": True,
        },
        "window_contract": {
            "single_public_planner": "ppg_frailty.data.WindowPlan",
            "explicit_fields": [
                "source_record_id",
                "window_seconds",
                "hop_seconds",
                "end_alignment",
                "short_record_action",
                "include_padded_tail",
                "max_windows",
                "cap_policy",
            ],
            "padding_mask_required": True,
        },
        "cache_contract": {
            "implementation": "ppg_frailty.data.ContentAddressedCache",
            "identity_fields": [
                "source_sha256",
                "config_sha256",
                "schema_sha256",
                "producer_sha256",
                "fold_file_sha256",
            ],
            "payload_hash_algorithm": "sha256_raw_bytes",
            "pickle_allowed": False,
        },
        "historical_only_registries": [
            {
                "path": HISTORICAL_SGKF_RELATIVE_PATH.as_posix(),
                "sha256": historical_sgkf_sha,
                "status": "historical_reproduction_only_forbidden_as_main",
            },
            {
                "path": LEGACY_EXTERNAL_SPLIT_RELATIVE_PATH.as_posix(),
                "sha256": legacy_sha,
                "status": "historical_15_3_4_only_not_active",
            },
        ],
        "generated_artifacts": {
            "internal_manifest": _artifact(internal_output),
            "primary_seed42_folds": _artifact(primary_split_output),
            "five_repeat_folds": _artifact(repeated_split_output),
        },
        "producer_source_sha256": _producer_sources(),
        "materializer": {
            "path": Path(__file__).resolve().relative_to(PIPELINE_ROOT).as_posix(),
            "sha256": sha256_file(__file__),
        },
    }
    atomic_write_json(
        internal_report_output,
        internal_report,
        root=PIPELINE_ROOT,
    )

    external_audit = audit_external_manifest(external_rows)
    fold_oof_counts = {
        str(fold_index): sum(
            row["split"] == "oof"
            and int(row["fold_index"]) == fold_index
            for row in provisional_rows
        )
        for fold_index in range(5)
    }
    external_report = {
        "schema_version": "ppg_frailty.external_data_contract_report.v1",
        "status": "pass_with_provisional_split_pending_confirmation",
        "source_authority": {
            "path": M2_EXTERNAL_RELATIVE_PATH.as_posix(),
            "sha256": M2_EXTERNAL_MANIFEST_SHA256,
            "source_snapshot_rehash_status": (
                "not_performed_paths_are_dataset_relative_and_not_in_contract"
            ),
        },
        "manifest_audit": external_audit,
        "usage_contract": {
            "intended_scope": "heartbeat_and_motion_development_benchmark",
            "independence_claim": INDEPENDENCE_CLAIM,
            "ptt_wavelength_policy": "preserve_unresolved_never_infer_red_ir",
            "sim_included_records": 13,
            "resampling_policy": (
                "explicit_time_axis_resampling_required_before_internal_400hz_use"
            ),
        },
        "provisional_grouped_split": {
            "registry_id": PROVISIONAL_EXTERNAL_SPLIT_REGISTRY_ID,
            "status": "provisional_pending_v2_human_confirmation",
            "seed": 42,
            "n_splits": 5,
            "group": "subject_id",
            "algorithm": (
                "sha256_rank_of_seed_colon_subject_then_round_robin_five_folds"
            ),
            "ptt_subject_count": 22,
            "activity_coverage_required_each_oof": ["run", "sit", "walk"],
            "oof_subject_count_by_fold": fold_oof_counts,
            "runtime_split_recomputation_allowed": False,
            "independent_test_claim_allowed": False,
            "v2_human_confirmation_point": True,
        },
        "legacy_split": {
            "path": LEGACY_EXTERNAL_SPLIT_RELATIVE_PATH.as_posix(),
            "sha256": legacy_sha,
            "partition_sizes": [15, 3, 4],
            "status": "historical_only_not_active_not_independent_test",
        },
        "generated_artifacts": {
            "external_manifest": _artifact(external_output),
            "provisional_ptt_grouped_split": _artifact(provisional_output),
        },
        "producer_source_sha256": _producer_sources(),
        "materializer": {
            "path": Path(__file__).resolve().relative_to(PIPELINE_ROOT).as_posix(),
            "sha256": sha256_file(__file__),
        },
    }
    atomic_write_json(
        external_report_output,
        external_report,
        root=PIPELINE_ROOT,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "internal_records": len(internal_rows),
                "external_records_total": len(external_rows),
                "external_records_included": sum(
                    row.inclusion_status == "included"
                    for row in external_rows
                ),
                "repeated_fold_assignments": len(repeated_assignments),
                "provisional_external_rows": len(provisional_rows),
                "reports": [
                    internal_report_output.relative_to(PIPELINE_ROOT).as_posix(),
                    external_report_output.relative_to(PIPELINE_ROOT).as_posix(),
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
