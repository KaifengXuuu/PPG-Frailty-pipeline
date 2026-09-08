"""Compact data and learned-weight exports for specialized computations."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import yaml

from ppg_frailty.reporting.tabular import ReportTable, write_excel_workbook

# Kept as a module-level compatibility seam for callers/tests that isolate the
# canonical pipeline root by monkeypatching this historical name.
from .output_contract import MODEL_CONFIG_ROOT, PIPELINE_OUTPUT_ROOT, V5_ROOT  # noqa: F401

_MOTION_EXPORT_SCHEMA = "ppg_frailty.v5_motion_model_config_export.v1"
_SPECIALIZED_EXCEL_SCHEMA = "ppg_frailty.v5_specialized_pipeline_workbook.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(payload: object) -> str:
    value = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(value).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a JSON object: {path}")
    return dict(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(dict.fromkeys(str(key) for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value,
                                                                                         (dict, list, tuple)) else value
                for key, value in row.items()
            })


def _source(root: Path, base: Path, value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def export_specialized_data_excel(output: str | Path, *, replace: bool = False) -> Mapping[str, Any]:
    """Create one economical workbook view over persisted CSV data."""

    root = Path(output).resolve()
    target = root / "tables" / "pipeline_data.xlsx"
    if target.exists() and not replace:
        raise FileExistsError(target)
    files = [{
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    } for path in sorted(root.rglob("*")) if path.is_file() and path != target]
    tables = [ReportTable("artifact_inventory", files, compact=False)]
    included = []
    for path in sorted(root.rglob("*.csv")):
        if path.stat().st_size > 32 * 1024 * 1024 or len(included) >= 64:
            continue
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) > 100_000:
            continue
        relative = path.relative_to(root).as_posix()
        tables.append(ReportTable(relative[:-4], rows, compact=False))
        included.append({"path": relative, "rows": len(rows)})
    target.parent.mkdir(parents=True, exist_ok=True)
    write_excel_workbook(target, tables)
    result = {
        "schema_version": _SPECIALIZED_EXCEL_SCHEMA,
        "status": "complete",
        "workbook": target.relative_to(root).as_posix(),
        "artifact_count": len(files),
        "included_csv_tables": included,
    }
    _write_json(root / "pipeline_excel_status.json", result)
    return result


def motion_model_sources(output: str | Path, *, inspect_checkpoints: bool = True) -> list[dict[str, Any]]:
    """Read all ten outer-fold and two final Stage5 models from evidence."""

    del inspect_checkpoints
    root = Path(output).resolve()
    manifest = _json(root / "study_manifest.json")
    specifications = (
        (
            "frailty29",
            "internal_motion_oof",
            "motion_internal_evidence.json",
            "final_threshold",
            "final_threshold_artifact_sha256",
        ),
        (
            "ptt22",
            "ptt_motion_training_ablation",
            "motion_ptt_training_evidence.json",
            "deployment_threshold",
            "deployment_threshold_artifact_sha256",
        ),
    )
    rows = []
    for dataset, stage_name, evidence_name, threshold_key, threshold_hash_key in specifications:
        stage = manifest["stages"][stage_name]
        directory = root / str(stage["artifact_dir"])
        evidence_path = directory / evidence_name
        evidence = _json(evidence_path)
        schema_path = _source(root, directory, evidence["model_input_schema_path"])
        entries = [("outer_fold", cell, cell["threshold"], cell["threshold_artifact_sha256"])
                   for cell in evidence["cell_evidence"]]
        entries.append((
            "final_all_participants",
            evidence["final_model"],
            evidence[threshold_key],
            evidence[threshold_hash_key],
        ))
        for role, entry, threshold, threshold_sha in entries:
            model = _source(
                root,
                directory,
                entry.get("model_artifact_path", entry.get("artifact_path")),
            )
            model_sha = str(entry.get("model_artifact_sha256", entry.get("artifact_sha256")))
            if _sha256(model) != model_sha:
                raise ValueError(f"model checksum mismatch: {model}")
            history = model.parent / "motion_training_history.json"
            rows.append({
                "dataset": dataset,
                "model_role": role,
                "repeat": int(entry.get("repeat_index", -1)),
                "fold": int(entry.get("fold_index", -1)),
                "model_id": evidence.get("model_id"),
                "source_model": model.relative_to(root).as_posix(),
                "model_sha256": model_sha,
                "source_training_history": history.relative_to(root).as_posix(),
                "source_evidence": evidence_path.relative_to(root).as_posix(),
                "source_input_schema": schema_path.relative_to(root).as_posix(),
                "threshold": dict(threshold),
                "threshold_sha256": str(threshold_sha),
                "parameter_count": entry.get("parameter_count"),
                "training_participant_ids": entry.get("training_participant_ids"),
                "training_participant_count": entry.get("training_participant_count"),
                "inference_cost": entry.get("inference_cost"),
                "loader_metadata": {
                    "artifact_sha256": model_sha,
                    "training_participant_ids": entry.get("training_participant_ids"),
                    "parameter_count": entry.get("parameter_count"),
                    "inference_cost": entry.get("inference_cost"),
                    "model_input_schema_sha256": evidence.get("model_input_schema_sha256"),
                },
            })
    if len(rows) != 12:
        raise ValueError(f"expected 12 Stage5 models, found {len(rows)}")
    return rows


def export_motion_model_config(
    output: str | Path,
    *,
    request_bindings: Mapping[str, Mapping[str, str]] | None = None,
) -> Mapping[str, Any]:
    """Export all fold/final weights and their reusable module parameters."""

    del request_bindings
    root = Path(output).resolve()
    rows = motion_model_sources(root)
    target = MODEL_CONFIG_ROOT / root.name
    target.mkdir(parents=True, exist_ok=True)
    plan = yaml.safe_load((root / "resolved_plan.yaml").read_text(encoding="utf-8"))
    exported = []
    shared = {}
    for row in rows:
        dataset = str(row["dataset"])
        if dataset not in shared:
            evidence = target / "training_evidence" / f"{dataset}.json"
            schema = target / "input_schemas" / f"{dataset}.json"
            evidence.parent.mkdir(parents=True, exist_ok=True)
            schema.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / str(row["source_evidence"]), evidence)
            shutil.copy2(root / str(row["source_input_schema"]), schema)
            shared[dataset] = {
                "exported_evidence": evidence.relative_to(target).as_posix(),
                "exported_input_schema": schema.relative_to(target).as_posix(),
            }
        coordinate = ("final" if row["model_role"] == "final_all_participants" else
                      f"repeat_{int(row['repeat']):02d}/fold_{int(row['fold']):02d}")
        directory = target / "learned_models" / dataset / coordinate
        directory.mkdir(parents=True, exist_ok=True)
        model = directory / "formal_motion_model.pt"
        history = directory / "motion_training_history.json"
        threshold = directory / "threshold.json"
        shutil.copy2(root / str(row["source_model"]), model)
        shutil.copy2(root / str(row["source_training_history"]), history)
        _write_json(threshold, row["threshold"])
        exported.append({
            **row,
            **shared[dataset],
            "exported_model": model.relative_to(target).as_posix(),
            "exported_training_history": history.relative_to(target).as_posix(),
            "exported_threshold": threshold.relative_to(target).as_posix(),
        })
    shutil.copy2(root / "resolved_plan.yaml", target / "resolved_plan.yaml")
    defaults = {
        "schema_version": "ppg_frailty.v5_specialized_module_defaults.v1",
        "source_pipeline_output": root.relative_to(V5_ROOT).as_posix(),
        "module_switches": {
            "motion_detector": True,
            "denoiser": bool(_json(root / "study_manifest.json").get("denoiser_enabled")),
        },
        "parameter_defaults": plan,
    }
    (target / "pipeline_module_defaults.yaml").write_text(yaml.safe_dump(defaults, sort_keys=False), encoding="utf-8")
    reuse = {
        "schema_version": "ppg_frailty.v5_motion_model_reuse_parameters.v1",
        "loader": "ppg_frailty.quality.motion_adapters.load_formal_motion_model",
        "model_count": len(exported),
        "models": exported,
    }
    (target / "model_reuse_parameters.yaml").write_text(yaml.safe_dump(reuse, sort_keys=False), encoding="utf-8")
    _write_csv(target / "model_artifacts.csv", exported)
    manifest = {
        "schema_version": _MOTION_EXPORT_SCHEMA,
        "status": "complete",
        "source_pipeline_output": root.relative_to(V5_ROOT).as_posix(),
        "model_count": len(exported),
        "outer_fold_model_count": sum(r["model_role"] == "outer_fold" for r in exported),
        "final_model_count": sum(r["model_role"] == "final_all_participants" for r in exported),
        "models": exported,
    }
    _write_json(target / "export_manifest.json", manifest)
    return {
        key: manifest[key]
        for key in (
            "schema_version",
            "status",
            "model_count",
            "outer_fold_model_count",
            "final_model_count",
        )
    } | {
        "output_directory": target.relative_to(V5_ROOT).as_posix()
    }


__all__ = [
    "PIPELINE_OUTPUT_ROOT",
    "export_motion_model_config",
    "export_specialized_data_excel",
    "motion_model_sources",
]
