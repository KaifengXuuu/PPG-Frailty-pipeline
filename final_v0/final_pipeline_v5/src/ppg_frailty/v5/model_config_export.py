"""Export reusable configs and one selected learned bundle per study case."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json, os
from pathlib import Path
import re, shutil, time
from typing import Any, Mapping

import yaml

from ..config import load_config
from ..module_registry import list_modules, registry_sha256
from ..training.bundle import verify_bundle
from .io import atomic_json, file_sha256, resolve_path
from .results import _read_csv, _read_mapping, _write_csv


EXPORT_SCHEMA = "ppg_frailty.v5_model_config_export.v1"
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")

def _write(path: Path, value: Any) -> None:
    if path.suffix.lower() in {".yaml", ".yml"}:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:
        atomic_json(path, value)

def _dotted(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for name in path.split("."):
        if not isinstance(value, Mapping) or name not in value:
            return None
        value = value[name]
    return value

def _derived_module_defaults(config: Mapping[str, Any]) -> list[dict[str, str]]:
    """Map fields back to selectors only when the mapping is unambiguous."""
    paths = dict(
        class_weighting="training.class_weighting", class_count_basis="training.class_count_basis",
        epoch_selection="training.epoch_rule", gap_repair="signal.gap_repair.method",
        imu_gravity="signal.imu.gravity_method", loss="training.loss", model="model.model_id",
        optimizer="training.optimizer", ppg_filter="signal.ppg_filter.family",
        peak_detector="signal.peak_detector.detector_id", quality_mode="quality.mode",
        quality_weight_source="aggregation.quality_weight_source", representation="representation_mode",
        sampler="training.sampler", training_balance="training.training_balance",
        window_quality_selection="quality.window_selection.policy", artifact="artifact.reducer",
        aggregation="aggregation.balance_line", shapeformer_discovery_balance="model.discovery_balance",
    )
    candidates = [(family, str(value), path) for family, path in paths.items()
                  if (value := _dotted(config, path)) is not None]
    enabled, method = (_dotted(config, f"signal.dl_resampling.{name}") for name in ("enabled", "method"))
    candidates += [("dl_resampling", "off_identity_source_grid" if enabled is False else str(method),
                    "signal.dl_resampling")]
    candidates += [("normalization", prefix + str(value), path)
                   for path, prefix in (("signal.normalization.raw_ppg", "ppg_"),
                                        ("signal.normalization.raw_imu", "imu_"))
                   if (value := _dotted(config, path)) is not None]
    candidates += [(family, "enabled" if value else "disabled", path)
                   for path, family in (("artifact.motion_detector_enabled", "motion_detector_switch"),
                                        ("artifact.denoiser_enabled", "denoiser_switch"))
                   if isinstance(value := _dotted(config, path), bool)]
    groups = _dotted(config, "features.enabled_groups")
    if isinstance(groups, list):
        candidates.extend(("feature_group", str(value), "features.enabled_groups") for value in groups)
    available = {(str(row["family"]), str(row["module_id"])) for row in list_modules()}
    return [{"family": family, "module_id": module, "source_path": path,
             "derivation": "exact_resolved_config_field"} for family, module, path in sorted(candidates)
            if (family, module) in available]

def _selected_models(study: Path, data: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    def usable(row: Any) -> bool:
        return isinstance(row, Mapping) and bool(row.get("case_id") and row.get("bundle_manifest"))

    selected = {str(row["case_id"]): dict(row) for row in data.get("published_models", ()) if usable(row)}
    refit_path = study / "v5_refit_manifest.json"
    if refit_path.is_file():
        selected.update({
            str(row["case_id"]): {
                **dict(row), "model_role": "all29_full_cohort_refit",
                "deployment_status": "full_cohort_refit_no_internal_self_evaluation",
            }
            for row in _read_mapping(refit_path).get("cases", ()) if usable(row)
        })
    return selected

def _copy_bundle(study: Path, target: Path, selection: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    raw = selection.get("bundle_manifest")
    if not isinstance(raw, str):
        return None, None
    manifest = resolve_path(raw, base=study, within=study, must_exist=True, label="selected bundle")
    bundle = manifest.parent
    try:
        verified = verify_bundle(bundle, load_model=False)
        if selection.get("bundle_manifest_sha256") not in (None, "", file_sha256(manifest)):
            raise ValueError("selected bundle manifest hash differs from the index")
    except (FileNotFoundError, TypeError, ValueError) as error:
        return None, str(error)
    shutil.copytree(bundle, target)
    return dict(verified), None

def _raw_inference_capability(config: Mapping[str, Any], bundle: Mapping[str, Any] | None,
                              config_hash: str | None) -> bool:
    expected = {
        "signal.normalization.raw_imu": "none", "quality.mode": "off",
        "quality.window_selection.policy": "none", "artifact.reducer": "identity",
    }
    disabled = ("artifact.motion_detector_enabled", "artifact.denoiser_enabled", "aggregation.quality_weighting")
    return bool(bundle and config_hash == bundle.get("config_hash") and config.get("representation_mode") == "raw"
                and all(_dotted(config, path) == value for path, value in expected.items())
                and all(_dotted(config, path) is False for path in disabled))

def _case_name(case_id: str, used: set[str]) -> str:
    base = _SAFE.sub("_", case_id).strip("._") or "case"
    name, number = base, 2
    while name in used:
        name, number = f"{base}_{number:02d}", number + 1
    used.add(name)
    return name

def export_model_config(study_directory: str | Path, *, pipeline_root: str | Path,
                        replace_existing: bool = False) -> dict[str, Any]:
    """Export one completed run without modifying it or loading model weights."""
    root = Path(pipeline_root).resolve()
    study = resolve_path(study_directory, base=root, within=root / "pipeline_output", must_exist=True,
                         label="pipeline output")
    required = (study / "study_manifest.json", study / "v5_data_manifest.json", study / "tables/v5_fold_models.csv")
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError("missing model-export inputs: " + ", ".join(missing))
    study_manifest, data = _read_mapping(required[0]), _read_mapping(required[1])
    cases = study_manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("study manifest contains no cases")
    target = root / "model_config" / study.name
    if target.exists() and not replace_existing:
        raise FileExistsError(f"refusing to overwrite existing model configuration export: {target}")
    staging = target.parent / f".{target.name}.staging-{time.time_ns()}"
    staging.mkdir(parents=True)
    request = study / "v5_run_request.json"
    request_data = _read_mapping(request) if request.is_file() else {}
    requested_modules = _dotted(request_data, "configuration_resolution.module_selections") or []
    selections = _selected_models(study, data)
    exported: list[dict[str, Any]] = []
    used: set[str] = set()
    registry_hash = registry_sha256()
    try:
        _write(staging / "available_modules.json", {"module_registry_sha256": registry_hash, "modules": list_modules()})
        (staging / "README.md").write_text(
            "# Model reuse export\n\nEach case contains its resolved pipeline config, module defaults, fold provenance "
            "and selected learned bundle.\n", encoding="utf-8",
        )
        for index, raw_case in enumerate(cases):
            if not isinstance(raw_case, Mapping):
                raise TypeError("study case must be a mapping")
            case_id = str(raw_case.get("case_id") or f"case_{index + 1:03d}")
            name = _case_name(case_id, used)
            directory = staging / "cases" / name
            source_config = resolve_path(str(raw_case["resolved_config_path"]), base=study, within=study,
                                         must_exist=True, label="resolved config")
            config = _read_mapping(source_config)
            try:
                config_hash = load_config(source_config).sha256
            except (KeyError, TypeError, ValueError):
                config_hash = None
            _write(directory / "resolved_pipeline_config.yaml", config)
            fields, all_rows = _read_csv(required[2])
            rows = [row for row in all_rows if row.get("case_id") == case_id]
            _write_csv(directory / "fold_model_parameters.csv", rows, fields)
            selection = selections.get(case_id)
            bundle, error = ((None, None) if selection is None else
                             _copy_bundle(study, directory / "learned_model", selection))
            inferable = _raw_inference_capability(config, bundle, config_hash)
            role = None if selection is None else selection.get(
                "model_role", "research_outer_fold_median_for_dashboard_trial")
            reason = error or (
                "resolved config and bundle config hashes differ"
                if bundle and config_hash != bundle.get("config_hash") else
                "ready" if inferable else "configuration or verified bundle is not supported for raw inference"
            )
            capability = {
                "configuration_reuse": True, "learned_weights_available": bundle is not None,
                "new_participant_inference": inferable, "reason": reason,
            }
            _write(directory / "pipeline_module_defaults.yaml", {
                "schema_version": "ppg_frailty.v5_pipeline_module_defaults.v1",
                "requested_module_selections": requested_modules,
                "derived_module_defaults": _derived_module_defaults(config), "parameter_defaults": config,
            })
            _write(directory / "model_reuse_parameters.yaml", {
                "schema_version": "ppg_frailty.v5_model_reuse_parameters.v1",
                "source": {"pipeline_output": str(study.relative_to(root)), "case_id": case_id},
                "model": config.get("model", {}), "training": config.get("training", {}),
                "representation_mode": config.get("representation_mode"),
                "fold_provenance": {"fold_model_row_count": len(rows), "file": "fold_model_parameters.csv"},
                "deployment_capability": capability, "selected_model": selection,
            })
            prefix = Path("cases") / name
            exported.append({
                "case_id": case_id, "directory": str(prefix),
                "resolved_config": str(prefix / "resolved_pipeline_config.yaml"),
                "fold_model_parameters": str(prefix / "fold_model_parameters.csv"),
                "fold_model_row_count": len(rows),
                "bundle_path": str(prefix / "learned_model") if bundle else None,
                "learned_model": str(prefix / "learned_model/manifest.json") if bundle else None,
                "model_role": role if bundle else None, "new_participant_inference": inferable,
                "inference_validation": {"status": "ready" if inferable else "unavailable", "reason": reason},
            })
        learned = [row for row in exported if row["bundle_path"]]
        inferable = [row for row in learned if row["new_participant_inference"]]
        manifest = {
            "schema_version": EXPORT_SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": {
                "pipeline_output": str(study.relative_to(root)),
                "study_manifest_sha256": file_sha256(required[0]), "v5_data_manifest_sha256": file_sha256(required[1]),
                "v5_data_status": data.get("status"),
            },
            "capabilities": {
                "configuration_reuse_only": not learned, "learned_weights_available": bool(learned),
                "new_participant_inference_available": bool(inferable), "new_participant_inference_status":
                "available" if inferable else "unavailable_without_compatible_verified_bundle",
            },
            "module_registry": {"file": "available_modules.json", "sha256": registry_hash},
            "case_count": len(exported), "cases": exported,
        }
        _write(staging / "export_manifest.json", manifest)
        target.parent.mkdir(parents=True, exist_ok=True)
        backup = target.parent / f".{target.name}.backup-{time.time_ns()}"
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(staging, target)
        except BaseException:
            if backup.exists():
                os.replace(backup, target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {**manifest, "output_directory": str(target.relative_to(root))}

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export reusable model/config parameters from one V5 run.")
    parser.add_argument("--pipeline-output", required=True)
    parser.add_argument("--replace", action="store_true")
    return parser

def main(argv: list[str] | None = None, *, pipeline_root: str | Path | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(pipeline_root or Path(__file__).resolve().parents[3]).resolve()
    result = export_model_config(args.pipeline_output, pipeline_root=root, replace_existing=args.replace)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


__all__ = ["_derived_module_defaults", "build_parser", "export_model_config", "main"]
