"""Index a completed run, optionally refit each case, then export its models."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import atomic_json, file_sha256, resolve_path
from .model_config_export import export_model_config
from .results import build_study_data_index

@dataclass(frozen=True)
class RefitOptions:
    """Optional full-cohort model stage; outer-fold training remains the default."""

    enabled: bool = False
    purpose: str = "configured_full_cohort_refit"

def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON root must be a mapping: {path}")
    return value

def _study_cases(study: Path) -> tuple[Mapping[str, Any], ...]:
    cases = _read_json(study / "study_manifest.json").get("cases", ())
    if not isinstance(cases, list) or not cases or not all(isinstance(case, Mapping) for case in cases):
        raise ValueError("study manifest contains no valid cases")
    return tuple(cases)

def _case_path(study: Path, case: Mapping[str, Any], field: str) -> Path:
    return resolve_path(str(case[field]), base=study, within=study, must_exist=True, label=field.replace("_", " "))

def preflight_refit_request(
    options: RefitOptions,
    *,
    pipeline_root: str | Path,
    cases: Sequence[Mapping[str, Any]],
    repeats: tuple[int, ...],
    folds: tuple[int, ...],
    resume_directory: str | Path | None,
) -> dict[str, Any] | None:
    """Describe the requested stage without adding a second scientific validator."""

    del pipeline_root, resume_directory
    if not options.enabled:
        return None
    return {
        "status": "enabled_after_outer_fold_training",
        "default_refit": False,
        "purpose": options.purpose,
        "case_count": len(cases),
        "cases": [
            {
                "case_id": str(case.get("case_id", "")),
                "config_id": str(case.get("config_id", "")),
                "config_sha256": str(case.get("config_sha256", "")),
            }
            for case in cases
        ],
        "outer_cells": len(repeats) * len(folds),
        "refit_scope": "complete eligible participant cohort for each resolved case",
    }

def _adopt_refit_bundle(bundle: Path, *, config_hash: str | None = None) -> Path:
    """Reuse an interrupted/resumed refit only after reload and golden parity."""

    from ..training.bundle import assert_golden_parity, load_bundle

    loaded = load_bundle(bundle)
    assert_golden_parity(loaded)
    metadata = loaded.manifest.get("metadata")
    observed = metadata.get("config_hash") if isinstance(metadata, Mapping) else None
    if config_hash and observed != config_hash:
        raise ValueError(f"existing all-29 refit config differs: {observed!r} != {config_hash!r}")
    return loaded.directory.resolve()

def _run_refits(study: Path, options: RefitOptions) -> list[dict[str, Any]]:
    if not options.enabled:
        return []
    from ..experiment import execute_final_refit, final_refit_preflight

    published: list[dict[str, Any]] = []
    for case in _study_cases(study):
        case_id = str(case["case_id"])
        config = _case_path(study, case, "resolved_config_path")
        evidence = final_refit_preflight(config, purpose=options.purpose)
        bundle = study / "models" / case_id / "all29_refit"
        directory = (
            _adopt_refit_bundle(bundle, config_hash=str(evidence["config_hash"]))
            if bundle.exists()
            else Path(execute_final_refit(config, bundle_directory=bundle, purpose=options.purpose)).resolve()
        )
        manifest = directory / "manifest.json"
        published.append(
            {
                "case_id": case_id,
                "status": "all29_refit_published",
                "purpose": options.purpose,
                "performance_evidence": "outer_oof_only_no_refit_self_evaluation",
                "bundle_manifest": manifest.relative_to(study).as_posix(),
                "bundle_manifest_sha256": file_sha256(manifest),
                "config_sha256": evidence["config_hash"],
                "participant_count": evidence["participant_count"],
                "participant_ids": list(evidence["participant_ids"]),
                "manifest_hash": evidence["manifest_hash"],
                "fold_registry_hash": evidence["fold_registry_hash"],
            }
        )
    return published

def post_run_finalize(
    study_directory: str | Path,
    *,
    pipeline_root: str | Path,
    hash_prediction_files: bool = False,
    export_configuration: bool = True,
    refit: RefitOptions | None = None,
) -> dict[str, Any]:
    """Build indexes, optional full-cohort weights, and the reusable export."""

    root = Path(pipeline_root).resolve()
    study = resolve_path(study_directory, base=root, within=root, must_exist=True, label="pipeline output")
    data = build_study_data_index(study, hash_prediction_files=hash_prediction_files)
    refit_cases = _run_refits(study, refit or RefitOptions())
    if refit_cases:
        atomic_json(
            study / "v5_refit_manifest.json",
            {
                "schema_version": "ppg_frailty.v5_refit_manifest.v1",
                "default_refit": False,
                "cases": refit_cases,
            },
        )
    export = (
        export_model_config(
            study,
            pipeline_root=root,
            replace_existing=(root / "model_config" / study.name).exists(),
        )
        if export_configuration
        else None
    )
    return {
        "study_directory": str(study),
        "data_manifest": data,
        "refit": refit_cases or None,
        "model_config_export": export,
    }


__all__ = ["RefitOptions", "post_run_finalize", "preflight_refit_request"]
