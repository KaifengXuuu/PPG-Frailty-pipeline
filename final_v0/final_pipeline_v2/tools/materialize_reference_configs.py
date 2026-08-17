#!/usr/bin/env python3
"""Materialize, but never execute, the formal V2 model catalogue.

The source catalogue contains 13 single-model candidates and two explicit
five-member ensemble comparisons.  A caller selects Line A or Line B and an
empty output directory.  This tool writes fully resolved, validated YAML plus
a SHA/byte index; it never trains, evaluates, recomputes folds, or overwrites.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppg_frailty.config import (  # noqa: E402
    canonical_json_bytes,
    load_formal_ablation_profiles,
    load_formal_experiment_catalog,
    validate_config_payload,
)


BASE_CONFIGS = {
    "raw": "reference_static_role_aware_v2.yaml",
    "feature_vector": "reference_static_feature_vector_v2.yaml",
    "feature_matrix": "reference_static_feature_matrix_v2.yaml",
    "fusion": "reference_static_fusion_v2.yaml",
}


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return dict(value)


def _load_yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), str(path))


def _line_sections(line: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if line not in {"line_a", "line_b"}:
        raise ValueError("line must be line_a or line_b")
    source = _load_yaml(ROOT / "configs/reference_static_feature_vector_v2.yaml")
    training = copy.deepcopy(source["training"])
    aggregation = copy.deepcopy(source["aggregation"])
    if line == "line_a":
        training["training_balance"] = "equal_files"
        aggregation.update(
            {
                "balance_line": "line_a_equal_files",
                "hierarchy": ["window", "file", "participant"],
                "window_to_file": "ordinary_mean",
                "file_to_role": "not_applicable",
                "role_to_participant": "not_applicable",
                "missing_role_policy": "not_applicable",
                "quality_weighting": False,
                "direct_all_window_participant_mean": False,
            }
        )
    return training, aggregation


def resolved_catalog_payloads(
    *,
    line: str,
    catalog_path: Path | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return all 15 fully resolved configs without writing or executing them."""

    catalog = load_formal_experiment_catalog(
        catalog_path or ROOT / "configs/formal_experiment_catalog_v2.yaml"
    )
    ablation_catalog = (
        load_formal_ablation_profiles(
            ROOT / "configs/formal_ablation_profiles_v2.yaml"
        )
        if line == "line_a"
        else None
    )
    line_training, line_aggregation = _line_sections(line)
    output: list[dict[str, Any]] = []
    for entry in catalog["entries"]:
        mode = str(entry["representation_mode"])
        base_path = ROOT / "configs" / BASE_CONFIGS[mode]
        base = _load_yaml(base_path)
        payload = copy.deepcopy(base)
        payload["config_id"] = f"{entry['config_stem']}_{line}_v2"
        payload["representation_mode"] = mode
        payload["model"] = copy.deepcopy(entry["model"])
        payload["training"] = copy.deepcopy(line_training)
        payload["aggregation"] = copy.deepcopy(line_aggregation)
        is_ensemble = entry["catalog_role"] == "ensemble_comparison"
        payload["output"]["write_member_oof"] = is_ensemble
        if line == "line_a":
            assert ablation_catalog is not None
            payload["output"]["formal_ablation_materialization"] = {
                "schema_version": "ppg_frailty.formal_ablation_materialization.v2",
                "family": "aggregation_balance",
                "profile_id": "equal_files_line_a_ablation",
                "catalog_role": "ablation",
                "base_config_path": base_path.relative_to(ROOT).as_posix(),
                "base_config_sha256": hashlib.sha256(
                    canonical_json_bytes(base)
                ).hexdigest(),
                "profile_catalog_sha256": ablation_catalog["catalog_sha256"],
                "single_factor_only": True,
                "automatic_execution": False,
                "scientific_execution_completed": False,
            }
        checked = validate_config_payload(payload)
        output.append(checked)
    if len(output) != 15 or len({row["config_id"] for row in output}) != 15:
        raise RuntimeError("formal catalogue did not resolve to 15 unique configs")
    return tuple(output)


def _atomic_text(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"catalog output already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def materialize_catalog(
    *,
    line: str,
    output_dir: Path,
    catalog_path: Path | None = None,
) -> dict[str, Any]:
    """Write validated configs and an immutable SHA/byte index to a new directory."""

    target = output_dir.resolve()
    target.relative_to(ROOT.resolve())
    if target.exists():
        raise FileExistsError(f"catalog output directory already exists: {target}")
    catalog_source = (
        catalog_path or ROOT / "configs/formal_experiment_catalog_v2.yaml"
    ).resolve()
    configs = resolved_catalog_payloads(line=line, catalog_path=catalog_source)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent)
    ).resolve()
    staging.relative_to(ROOT.resolve())
    try:
        rows: list[dict[str, Any]] = []
        for payload in configs:
            filename = f"{payload['config_id']}.yaml"
            path = staging / filename
            rendered = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
            _atomic_text(path, rendered)
            encoded = path.read_bytes()
            rows.append(
                {
                    "config_id": payload["config_id"],
                    "path": filename,
                    "bytes": len(encoded),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                    "canonical_payload_sha256": hashlib.sha256(
                        canonical_json_bytes(payload)
                    ).hexdigest(),
                    "model_id": payload["model"]["model_id"],
                    "representation_mode": payload["representation_mode"],
                    "ensemble_size": payload["model"]["ensemble_size"],
                }
            )
        manifest = {
            "schema_version": "ppg_frailty.materialized_formal_catalog.v2",
            "pipeline_generation": "final_pipeline_v2",
            "status": "materialized_not_executed",
            "balance_line": line,
            "source_catalog_path": catalog_source.relative_to(ROOT).as_posix(),
            "source_catalog_sha256": hashlib.sha256(
                catalog_source.read_bytes()
            ).hexdigest(),
            "config_count": len(rows),
            "candidate_count": sum(row["ensemble_size"] == 1 for row in rows),
            "ensemble_comparison_count": sum(
                row["ensemble_size"] == 5 for row in rows
            ),
            "auto_run": False,
            "training_executed": False,
            "ablation_executed": False,
            "full_5x5_executed": False,
            "ptt_benchmark_executed": False,
            "artifacts": rows,
        }
        manifest["artifact_payload_sha256"] = hashlib.sha256(
            canonical_json_bytes(rows)
        ).hexdigest()
        _atomic_text(
            staging / "catalog_manifest.json",
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
        )
        for payload, row in zip(configs, rows):
            materialized = staging / str(row["path"])
            if (
                materialized.stat().st_size != int(row["bytes"])
                or hashlib.sha256(materialized.read_bytes()).hexdigest()
                != row["sha256"]
                or validate_config_payload(_load_yaml(materialized)) != payload
            ):
                raise RuntimeError("staged formal catalog verification failed")
        if target.exists():
            raise FileExistsError(f"catalog output directory already exists: {target}")
        staging.rename(target)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line", choices=("line_a", "line_b"), required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory below final_pipeline_v2; existing paths are rejected",
    )
    parser.add_argument("--catalog", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    catalog_path = arguments.catalog
    if catalog_path is not None and not catalog_path.is_absolute():
        catalog_path = ROOT / catalog_path
    output_dir = arguments.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    manifest = materialize_catalog(
        line=arguments.line,
        output_dir=output_dir,
        catalog_path=catalog_path,
    )
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
