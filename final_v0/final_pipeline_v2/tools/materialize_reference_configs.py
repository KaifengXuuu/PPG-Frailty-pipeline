#!/usr/bin/env python3
"""Materialize, but never execute, the formal V2 model catalogue.

The source catalogue contains 13 ordinary candidates, two explicit member-0
comparators and two five-member ensemble comparisons. A caller selects Line A
or Line B and an empty output directory. This tool writes fully resolved,
validated YAML plus a SHA/byte index; it never trains, evaluates, recomputes
folds, or overwrites.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppg_frailty.catalog import resolved_catalog_payloads  # noqa: E402
from ppg_frailty.config import (  # noqa: E402
    canonical_json_bytes,
    load_formal_experiment_catalog,
    validate_config_payload,
)




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
    configs = resolved_catalog_payloads(
        pipeline_root=ROOT,
        line=line,
        catalog_path=catalog_source,
    )
    catalog = load_formal_experiment_catalog(catalog_source)
    role_by_config_id = {
        f"{entry['config_stem']}_{line}_v2": str(entry["catalog_role"])
        for entry in catalog["entries"]
    }
    if set(role_by_config_id) != {str(row["config_id"]) for row in configs}:
        raise RuntimeError("resolved catalog config identities drifted from entries")
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
                    "seed_policy": payload["model"]["seed_policy"],
                    "catalog_role": role_by_config_id[str(payload["config_id"])],
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
            "candidate_count": sum(
                row["catalog_role"] in {"reference_candidate", "ablation_candidate"}
                for row in rows
            ),
            "matched_comparator_count": sum(
                row["catalog_role"] == "matched_comparator" for row in rows
            ),
            "ensemble_comparison_count": sum(
                row["catalog_role"] == "ensemble_comparison" for row in rows
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
                or validate_config_payload(
                    yaml.safe_load(materialized.read_text(encoding="utf-8"))
                )
                != payload
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
