#!/usr/bin/env python3
"""Verify that copied numerical modules still match their V2 source.

Only repository-path relocation is allowed in two source files.  V5 application
and reporting facades are new files and therefore excluded from byte parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

V5_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = V5_ROOT.parent / "final_pipeline_v2"
NEW_PREFIXES = (
    "ppg_frailty/v5/",
    "ppg_frailty/v5_reporting/",
)
NON_NUMERICAL_PREFIXES = ("ppg_frailty/dashboard/", )
# V2's monolithic command dispatcher contained presentation, synthetic demos,
# subprocess launchers, and formal execution in one 1,099-line file.  V5 keeps
# every called algorithm in its original module but retires this application
# facade in favour of the typed V5 service/CLI boundary.
RETIRED_APPLICATION_FILES = {
    "ppg_frailty/cli.py": "replaced_by_ppg_frailty.v5.cli_no_algorithm_removed",
}
PATH_ONLY_FILES = {
    "ppg_frailty/data/external_manifest.py",
    "ppg_frailty/quality/motion_reference.py",
}
APPROVED_INTEGRATION_FILES = {
    "ppg_frailty/audit/legacy_v2_bridge.py": {
        "v2_sha256": "bb803531748375f33adf80213eb482e029e27a5350bf12ce6393a631bb465ebd",
        "v5_sha256": "d479e20dc7cfdd1f80b5c619f42f435835b09fa178e2be8449cc5b9111fb5211",
        "reason": "default_true_phase0_markdown_gate_for_v5_data_only_publication",
    },
    "ppg_frailty/experiment.py": {
        "v2_sha256": "b1c0e4f1a98bc5f3539065fc966b6e31a23cab286475c49d656a358477976db3",
        "v5_sha256": "504802ed6738b23bdc630cad29b147fc7fc058c6e9813cba1afea5afad56638b",
        "reason": "post_fit_checkpoint_payload_and_physical_nested_output_paths",
    },
    "ppg_frailty/study/expand.py": {
        "v2_sha256": "a7f87779b99974431587257cf088a0fd4bc8074d3137ede3693a86b1478d9363",
        "v5_sha256": "42dd2dbfcce1bf7568454271fb83d26fc9f4ccee7b70afb3e53ac59be13d9d78",
        "reason": "preserve_base_config_identity_when_no_comparison_axis_exists",
    },
    "ppg_frailty/study/recovery.py": {
        "v2_sha256":
        "51bacb3c6707d5024972119e4e7a6700454265d555b4614d1701443a6ddcc658",
        "v5_sha256":
        "9aba0be80046608c8ec22dafd445aa7f298253020ad844037c14bde4c2b04dfa",
        "reason": ("validate_and_recover_complete_nested_v5_publications_without_refit_"
                   "including_full_nested_checkpoint_integrity_and_idempotent_externalized_sqi"),
    },
    "ppg_frailty/study/runner.py": {
        "v2_sha256": "3305c11ce4bc43d643dc5cd4afbb61d10a224a5951a87478e5feef53efa7bbd6",
        "v5_sha256": "53071c6808569043345a41e93853704c39f39c1d7a7fa2bcb006b6641843a3ca",
        "reason": "v5_output_layout_atomic_publication_interruption_recovery_and_run_scoped_phase0_data",
    },
    "ppg_frailty/quality/stage5_pre.py": {
        "v2_sha256":
        "a2ae703fa743f1819e2022c4c0f50e1a194816016705e871f2c1909aa37eede8",
        "v5_sha256":
        "e2b7e9a5d229c9240f137ea2153fbccbe79fb32692cfd659913319b51b769545",
        "reason": ("default_true_report_gate_only_for_v5_data_only_specialized_adapter_"
                   "with_unchanged_stage5_and_static_peak_computation"),
    },
    "ppg_frailty/study/hyperparameter.py": {
        "v2_sha256":
        "8d0c4a7b6f5d16cc72b784dcfbfa50c056e8d05dac3dd2ca3f05b62ddc6ac9a2",
        "v5_sha256":
        "11837814cae0fd36a7aa8376987377c5d6376cb9204d5a6067a3bc4c71e78e3f",
        "reason": ("default_true_report_gates_plus_data_only_ranking_tables_and_run_name_"
                   "allocation_plus_v5_only_phase_runner_injection_and_resume_routing_"
                   "without_hyperparameter_math_or_promotion_changes"),
    },
    "ppg_frailty/training/bundle.py": {
        "v2_sha256":
        "aef24734ff448e21858d66a6d8c4190f9ea912970a13b7ea80294ef24403ee9c",
        "v5_sha256":
        "95e5b01b6171e50451f5f3bdc57742eea8c4304360e7a19be58de9cd658a8b6f",
        "reason": ("serialize_explicit_raw_noop_run_golden_parity_on_training_device_"
                   "and_attach_v5_request_metadata_without_rng_or_numerical_changes"),
    },
}
PRESET_SOURCES = {
    "baseline.yaml":
    V2_ROOT / "configs/reference_static_role_aware_v2.yaml",
    "feature_vector.yaml":
    V2_ROOT / "configs/reference_static_feature_vector_v2.yaml",
    "feature_matrix.yaml":
    V2_ROOT / "configs/reference_static_feature_matrix_v2.yaml",
    "fusion.yaml":
    V2_ROOT / "configs/reference_static_fusion_v2.yaml",
    "finalcase.yaml": (V2_ROOT / "artifacts/studies/static_line_b_staged_v2/"
                       "20260824_final_case_inception_small_no_gravity_merged_v2/"
                       "cases/small_no_gravity__raw__"
                       "tuned_all_roles__inception_small_no_gravity/resolved_config.yaml"),
}
V5_ONLY_STATIC_FILES = {
    "configs/studies/finalcase.yaml",
    "requirements/environment-finalcase-lock.yaml",
    "requirements/requirements-finalcase-lock.txt",
}
APPROVED_STATIC_FILES = {
    "configs/studies/README.md": {
        "v2_sha256": "efa930b92de9ad7ad0ccc3da2219b867278c6b058501a3fb69a2fc4ccb06a8c7",
        "v5_sha256": "d0e44732240ea2841a83dad8f6f561e57dee125a0a5d5b49715ba5dfaae7821f",
        "reason": "replace_obsolete_v2_commands_with_v5_data_only_entry_points",
    },
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_source_bytes(path: Path) -> bytes:
    value = path.read_text(encoding="utf-8")
    return value.replace("final_pipeline_v5", "final_pipeline_v2").encode("utf-8")


def audit() -> dict[str, Any]:
    source_files = {
        path.relative_to(V2_ROOT / "src").as_posix(): path
        for path in (V2_ROOT / "src").rglob("*.py") if not any(
            path.relative_to(V2_ROOT / "src").as_posix().startswith(prefix) for prefix in NON_NUMERICAL_PREFIXES)
    }
    copied_files = {
        path.relative_to(V5_ROOT / "src").as_posix(): path
        for path in (V5_ROOT / "src").rglob("*.py") if not any(
            path.relative_to(V5_ROOT / "src").as_posix().startswith(prefix)
            for prefix in (*NEW_PREFIXES, *NON_NUMERICAL_PREFIXES))
    }
    missing = sorted(set(source_files) - set(copied_files) - set(RETIRED_APPLICATION_FILES))
    unexpected = sorted(set(copied_files) - set(source_files))
    exact: list[str] = []
    path_only: list[str] = []
    approved_integration: list[dict[str, str]] = []
    mismatched: list[str] = []
    for relative in sorted(set(source_files) & set(copied_files)):
        source, copied = source_files[relative], copied_files[relative]
        if source.read_bytes() == copied.read_bytes():
            exact.append(relative)
        elif relative in PATH_ONLY_FILES and source.read_bytes() == _normalized_source_bytes(copied):
            path_only.append(relative)
        elif relative in APPROVED_INTEGRATION_FILES:
            contract = APPROVED_INTEGRATION_FILES[relative]
            if _sha(source) == contract["v2_sha256"] and _sha(copied) == contract["v5_sha256"]:
                approved_integration.append({"path": relative, "reason": contract["reason"]})
            else:
                mismatched.append(relative)
        else:
            mismatched.append(relative)

    preset_rows: list[dict[str, Any]] = []
    preset_failures: list[str] = []
    for filename, source in PRESET_SOURCES.items():
        target = V5_ROOT / "configs/presets" / filename
        matches = source.is_file() and target.is_file() and _sha(source) == _sha(target)
        preset_rows.append({
            "preset": filename.removesuffix(".yaml"),
            "source": str(source),
            "source_sha256": _sha(source) if source.is_file() else None,
            "v5_sha256": _sha(target) if target.is_file() else None,
            "exact": matches,
        })
        if not matches:
            preset_failures.append(filename)

    config_failures: list[str] = []
    source_configs = {path.name: path for path in (V2_ROOT / "configs").glob("*.yaml")}
    target_configs = {path.name: path for path in (V5_ROOT / "configs").glob("*.yaml")}
    for filename in sorted(set(source_configs) | set(target_configs)):
        source = source_configs.get(filename)
        target = target_configs.get(filename)
        if source is None or target is None:
            config_failures.append(filename)
            continue
        if source.read_bytes() == target.read_bytes():
            continue
        if filename in {"motion_detector_contract_v2.yaml", "v2_decision_profile.yaml"}:
            if source.read_bytes() == _normalized_source_bytes(target):
                continue
        config_failures.append(filename)

    static_failures: list[str] = []
    approved_static: list[dict[str, str]] = []
    static_count = 0
    for directory in (
            "manifests",
            "splits",
            "reports",
            "docs/spec",
            "model_cards",
            "requirements",
            "configs/studies",
    ):
        source_root, target_root = V2_ROOT / directory, V5_ROOT / directory
        source_items = {
            path.relative_to(source_root).as_posix(): path
            for path in source_root.rglob("*") if path.is_file()
        }
        target_items = {
            path.relative_to(target_root).as_posix(): path
            for path in target_root.rglob("*") if path.is_file()
        }
        for relative in sorted(set(source_items) | set(target_items)):
            static_count += 1
            joined = f"{directory}/{relative}"
            if source_items.get(relative) is None and joined in V5_ONLY_STATIC_FILES:
                continue
            if joined in APPROVED_STATIC_FILES:
                contract = APPROVED_STATIC_FILES[joined]
                if (relative in source_items and relative in target_items
                        and _sha(source_items[relative]) == contract["v2_sha256"]
                        and _sha(target_items[relative]) == contract["v5_sha256"]):
                    approved_static.append({"path": joined, "reason": contract["reason"]})
                    continue
            if (relative not in source_items or relative not in target_items
                    or _sha(source_items[relative]) != _sha(target_items[relative])):
                static_failures.append(f"{directory}/{relative}")

    passed = not any((
        missing,
        unexpected,
        mismatched,
        preset_failures,
        config_failures,
        static_failures,
    ))
    return {
        "schema_version": "ppg_frailty.v5_v2_parity_audit.v1",
        "status": "passed" if passed else "failed",
        "v2_root": str(V2_ROOT),
        "v5_root": str(V5_ROOT),
        "source_python_file_count": len(source_files),
        "exact_python_file_count": len(exact),
        "path_only_python_files": path_only,
        "approved_hash_bound_integration_files": approved_integration,
        "missing_python_files": missing,
        "retired_application_files": RETIRED_APPLICATION_FILES,
        "unexpected_python_files": unexpected,
        "mismatched_python_files": mismatched,
        "static_file_count": static_count,
        "approved_hash_bound_static_files": approved_static,
        "static_failures": static_failures,
        "presets": preset_rows,
        "preset_failures": preset_failures,
        "top_level_config_count": len(source_configs),
        "top_level_config_failures": config_failures,
        "excluded_new_prefixes": list(NEW_PREFIXES),
        "excluded_non_numerical_prefixes": list(NON_NUMERICAL_PREFIXES),
        "v5_only_static_files": sorted(V5_ONLY_STATIC_FILES),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", help="Optional JSON output inside V5.")
    args = parser.parse_args()
    result = audit()
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.write:
        target = Path(args.write)
        target = target.resolve() if target.is_absolute() else (V5_ROOT / target).resolve()
        target.relative_to(V5_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
