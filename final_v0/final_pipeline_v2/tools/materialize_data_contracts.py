#!/usr/bin/env python3
"""Materialize V2 data identities from frozen authorities without splitting.

The default target is the pipeline root. Acceptance may instead use an empty
isolated target below the pipeline root with the isolated-nonoverwrite flag.
Both routes re-hash all 261 raw sources, copy corrected M2 memberships, build
the external manifest and materialize already-confirmed PTT assignments. No
splitter, training, ablation, or benchmark is invoked.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path
from typing import Any


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PIPELINE_ROOT.parents[1]
SRC_ROOT = PIPELINE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ppg_frailty.data import (  # noqa: E402
    audit_external_manifest,
    audit_manifest,
    build_external_manifest,
    build_internal_manifest,
    load_frozen_memberships,
    materialize_fold_csvs,
    materialize_formal_ptt_repeated_folds,
)
from ppg_frailty.data.external_manifest import (  # noqa: E402
    M2_EXTERNAL_MANIFEST_SHA256,
    M2_EXTERNAL_RELATIVE_PATH,
    PTT_CHANNEL_MAPPING_PROVENANCE,
    PTT_DISTAL_CHANNEL_MAPPING,
    PTT_SOURCE_PAGE_URL,
)
from ppg_frailty.data.folds import (  # noqa: E402
    M2_SPLIT_FILE_SHA256,
    M2_SPLIT_PAYLOAD_SHA256,
    M2_SPLIT_RELATIVE_PATH,
    validate_frozen_memberships,
)
from ppg_frailty.data.manifest import (  # noqa: E402
    M2_FILE_MANIFEST,
    M2_FILE_MANIFEST_SHA256,
)
from ppg_frailty.provenance import atomic_write_json, sha256_file  # noqa: E402


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "internal_manifest": output_root / "manifests/internal_records_v2.csv",
        "external_manifest": output_root / "manifests/external_records_v2.csv",
        "primary_folds": output_root / "splits/sgkf5_seed42_v2.csv",
        "repeated_folds": output_root / "splits/sgkf5_repeated_grouped_5x5_v2.csv",
        "ptt_folds": output_root / "splits/ptt_formal_repeated_grouped_5x5_v2.csv",
        "internal_report": output_root / "reports/internal_manifest_v2_report.json",
        "data_report": output_root / "reports/data_contract_report_v2.json",
        "external_report": output_root / "reports/external_data_contract_report_v2.json",
        "index_report": output_root / "reports/materialization_index_v2.json",
    }


def _artifact(path: Path, output_root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(output_root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _write_incomplete(paths: dict[str, Path], output_root: Path) -> None:
    payload = {
        "schema_version": "ppg_frailty.data_materialization_state.v2",
        "pipeline_generation": "final_pipeline_v2",
        "status": "materializing_incomplete_fail_closed",
        "training_executed": False,
        "ablation_executed": False,
        "ptt_benchmark_executed": False,
    }
    for key in ("internal_report", "data_report", "external_report", "index_report"):
        atomic_write_json(paths[key], payload, root=output_root)


def materialize(output_root: Path, *, isolated_nonoverwrite: bool) -> dict[str, Any]:
    """Build all data contracts under one verified output boundary."""

    root = output_root.resolve()
    root.relative_to(PIPELINE_ROOT.resolve())
    authoritative = root == PIPELINE_ROOT.resolve()
    if not authoritative and not isolated_nonoverwrite:
        raise ValueError("non-authoritative output requires --isolated-nonoverwrite")
    if not authoritative:
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise FileExistsError("isolated materialization target must be empty")
    paths = _paths(root)
    _write_incomplete(paths, root)

    internal_rows = build_internal_manifest(
        REPOSITORY_ROOT / M2_FILE_MANIFEST,
        paths["internal_manifest"],
    )
    registry = load_frozen_memberships(REPOSITORY_ROOT / M2_SPLIT_RELATIVE_PATH)
    fold_audit = validate_frozen_memberships(registry, internal_rows)
    primary, repeated = materialize_fold_csvs(
        registry,
        internal_rows,
        paths["primary_folds"],
        paths["repeated_folds"],
        output_root=root,
    )
    external_rows = build_external_manifest(
        REPOSITORY_ROOT / M2_EXTERNAL_RELATIVE_PATH,
        paths["external_manifest"],
    )
    ptt_rows = materialize_formal_ptt_repeated_folds(
        external_rows,
        paths["ptt_folds"],
        output_root=root,
    )
    generated = {
        "internal_manifest": _artifact(paths["internal_manifest"], root),
        "primary_seed42_folds": _artifact(paths["primary_folds"], root),
        "five_repeat_folds": _artifact(paths["repeated_folds"], root),
        "external_manifest": _artifact(paths["external_manifest"], root),
        "ptt_formal_repeated_folds": _artifact(paths["ptt_folds"], root),
    }
    internal_summary = audit_manifest(internal_rows)
    external_summary = audit_external_manifest(external_rows)
    internal_report = {
        "schema_version": "ppg_frailty.internal_manifest_materialization.v2",
        "pipeline_generation": "final_pipeline_v2",
        "status": "passed",
        "source_manifest_path": M2_FILE_MANIFEST.as_posix(),
        "source_manifest_sha256": sha256_file(REPOSITORY_ROOT / M2_FILE_MANIFEST),
        "source_manifest_sha256_expected": M2_FILE_MANIFEST_SHA256,
        "all_261_source_hashes_verified": True,
        "summary": internal_summary,
        "generated_artifact": generated["internal_manifest"],
        "producer_sha256": sha256_file(__file__),
    }
    data_report = {
        "schema_version": "ppg_frailty.data_contract_report.v2",
        "pipeline_generation": "final_pipeline_v2",
        "status": "passed",
        "authority": {
            "manifest_sha256": M2_FILE_MANIFEST_SHA256,
            "fold_file_sha256": M2_SPLIT_FILE_SHA256,
            "fold_payload_sha256": M2_SPLIT_PAYLOAD_SHA256,
        },
        "internal_manifest_audit": internal_summary,
        "frozen_fold_audit": asdict(fold_audit),
        "primary_assignment_count": len(primary),
        "repeated_assignment_count": len(repeated),
        "generated_artifacts": {
            key: generated[key]
            for key in (
                "internal_manifest",
                "primary_seed42_folds",
                "five_repeat_folds",
            )
        },
        "training_executed": False,
        "scientific_metrics_emitted": False,
    }
    external_report = {
        "schema_version": "ppg_frailty.external_data_contract_report.v2",
        "pipeline_generation": "final_pipeline_v2",
        "status": "passed",
        "source_manifest": {
            "path": M2_EXTERNAL_RELATIVE_PATH.as_posix(),
            "sha256": M2_EXTERNAL_MANIFEST_SHA256,
        },
        "manifest_audit": external_summary,
        "ptt_distal_mapping": dict(PTT_DISTAL_CHANNEL_MAPPING),
        "mapping_provenance": PTT_CHANNEL_MAPPING_PROVENANCE,
        "source_page": PTT_SOURCE_PAGE_URL,
        "formal_assignment_count": len(ptt_rows),
        "generated_artifacts": {
            "external_manifest": generated["external_manifest"],
            "ptt_formal_repeated_folds": generated["ptt_formal_repeated_folds"],
        },
        "independent_test_claim": False,
        "ptt_benchmark_executed": False,
    }
    atomic_write_json(paths["internal_report"], internal_report, root=root)
    atomic_write_json(paths["data_report"], data_report, root=root)
    atomic_write_json(paths["external_report"], external_report, root=root)
    index = {
        "schema_version": "ppg_frailty.data_materialization_index.v2",
        "pipeline_generation": "final_pipeline_v2",
        "status": "passed",
        "output_mode": (
            "authoritative_replace" if authoritative else "isolated_nonoverwrite"
        ),
        "generated_artifacts": generated,
        "reports": {
            "internal": _artifact(paths["internal_report"], root),
            "data_contract": _artifact(paths["data_report"], root),
            "external": _artifact(paths["external_report"], root),
        },
        "producer": {
            "path": Path(__file__).relative_to(PIPELINE_ROOT).as_posix(),
            "sha256": sha256_file(__file__),
            "bytes": Path(__file__).stat().st_size,
        },
        "training_executed": False,
        "ablation_executed": False,
        "ptt_benchmark_executed": False,
    }
    atomic_write_json(paths["index_report"], index, root=root)
    return index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PIPELINE_ROOT,
        help="pipeline root (default) or an empty isolated directory below it",
    )
    parser.add_argument("--isolated-nonoverwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    if PIPELINE_ROOT.name != "final_pipeline_v2":
        raise SystemExit("materializer root is not final_pipeline_v2")
    arguments = build_parser().parse_args(argv)
    output_root = arguments.output_root
    if not output_root.is_absolute():
        output_root = PIPELINE_ROOT / output_root
    index = materialize(
        output_root,
        isolated_nonoverwrite=arguments.isolated_nonoverwrite,
    )
    print(json.dumps(index, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
