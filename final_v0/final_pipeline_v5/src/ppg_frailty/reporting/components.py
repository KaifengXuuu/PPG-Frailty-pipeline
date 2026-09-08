"""Compact component inventory derived from persisted resolved configuration."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .profiles import annotate_component_row, annotate_component_rows
from .tabular import markdown_column_definitions_block

TEST_COMPONENT_COLUMNS = (
    ("module_id", "Model / module"),
    ("component_role", "Component role"),
    ("participating_cases", "Cases / phases"),
    ("execution_state", "State"),
    ("input_data", "Input data"),
    ("fixed_parameters", "Fixed parameters"),
    ("algorithm_kernel_description", "Algorithm and kernel"),
    ("reporter_profile_id", "Reporter profile"),
    ("model_reporter_extension_id", "Reporter extension"),
    ("algorithm_references", "Sources"),
)
TOP_MODEL_CONFIGURATION_COLUMNS = (
    ("predictive_rank", "Rank"),
    ("case_id", "Case"),
    ("model_id", "Model"),
    ("representation_mode", "Representation"),
    ("resolved_config_path", "Resolved config"),
    ("config_section", "Section"),
    ("parameter_path", "Parameter"),
    ("resolved_value", "Value"),
)
TEST_COMPONENT_VIEW_SCHEMAS = (
    ("Participation", TEST_COMPONENT_COLUMNS[:4]),
    ("Inputs and parameters", (TEST_COMPONENT_COLUMNS[0], *TEST_COMPONENT_COLUMNS[4:6])),
    ("Methods and provenance", (TEST_COMPONENT_COLUMNS[0], *TEST_COMPONENT_COLUMNS[6:])),
)
_HASH = re.compile(r"(?:^|_)(?:sha\d*|hash|checksum)(?:_|$)", re.I)


def without_hashes(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): without_hashes(v) for k, v in value.items() if not _HASH.search(str(k))}
    if isinstance(value, (list, tuple)):
        return [without_hashes(v) for v in value]
    return value.as_posix() if isinstance(value, Path) else value


def _cell(value: Any) -> str:
    return json.dumps(without_hashes(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _row(
    case: str,
    role: str,
    module: Any,
    state: str,
    inputs: Mapping[str, Any],
    parameters: Any,
    *,
    description: str = "",
    reporter_profile_id: str | None = None,
) -> dict[str, Any]:
    return annotate_component_row({
        "participating_cases": case,
        "component_role": role,
        "module_id": str(module or "not_declared"),
        "execution_state": state,
        "input_data": _cell(inputs),
        "fixed_parameters": _cell(parameters),
        "algorithm_kernel_description": description,
        **({
            "reporter_profile_id": reporter_profile_id
        } if reporter_profile_id else {}),
    })


def _group(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = tuple(name for name, _ in TEST_COMPONENT_COLUMNS if name != "participating_cases")
    grouped: dict[tuple[str, ...], list[str]] = {}
    originals: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = tuple(_cell(row.get(field)) for field in fields)
        grouped.setdefault(key, []).append(str(row.get("participating_cases", "")))
        originals[key] = row
    return [{
        "participating_cases": "; ".join(sorted(set(cases))),
        **{field: originals[key].get(field, "")
           for field in fields},
    } for key, cases in grouped.items()]


def _configs(root: Path, manifest: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any], str]]:
    output = []
    for case in manifest.get("cases", ()):
        if not isinstance(case, Mapping):
            continue
        relative = Path(str(case.get("resolved_config_path", "")))
        target = (root / relative).resolve()
        if relative and not relative.is_absolute() and target.is_relative_to(root.resolve()) and target.is_file():
            config = yaml.safe_load(target.read_text(encoding="utf-8"))
            if isinstance(config, Mapping):
                output.append((str(case.get("case_id", target.parent.name)), config, relative.as_posix()))
    return output


def _get(value: Mapping[str, Any], path: str, default: Any = None) -> Any:
    current: Any = value
    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


_PIPELINE_COMPONENTS = (
    ("dataset_adapter", "manifest", "dataset_id", "audit_provenance_v1"),
    ("split_registry", "splits", "registry_id", "audit_provenance_v1"),
    ("ppg_preprocessing", "signal.ppg_filter", "family", "audit_provenance_v1"),
    ("imu_preprocessing", "signal.imu", "gravity_method", "audit_provenance_v1"),
    ("peak_detector", "signal.peak_detector", "detector_id", "audit_provenance_v1"),
    ("window_planner", "windows", "shared_planner_version", "audit_provenance_v1"),
    ("sqi", "quality", "mode", "sqi_route_coverage_v1"),
    ("motion_detector", "artifact.motion_detector", "model_id", "motion_route_component_v1"),
    ("denoiser", "artifact", "reducer", "frailty_denoiser_route_v1"),
    ("feature_extractor", "features", "registry_id", "audit_provenance_v1"),
    ("representation", "", "representation_mode", "audit_provenance_v1"),
    ("classifier", "model", "model_id", "multiclass_participant_oof_v1"),
    ("trainer", "training", "optimizer", "audit_provenance_v1"),
    ("aggregation", "aggregation", "balance_line", "audit_provenance_v1"),
    ("evaluation", "evaluation", "primary_metric", "audit_provenance_v1"),
)


def build_pipeline_test_component_rows(root: str | Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for case_id, config, _ in _configs(Path(root), manifest):
        context = {
            "dataset": _get(config, "manifest.dataset_id"),
            "roles": _get(config, "manifest.roles"),
            "fs_hz": _get(config, "signal.target_fs_hz"),
        }
        for role, section_path, id_key, profile in _PIPELINE_COMPONENTS:
            section = _get(config, section_path, config if not section_path else {})
            section = section if isinstance(section, Mapping) else {"value": section}
            module = section.get(id_key, _get(config, id_key))
            active = module not in (None, "", "off", False)
            rows.append(
                _row(
                    case_id,
                    role,
                    module,
                    "enabled" if active else "not_executed_not_declared",
                    context,
                    section,
                    reporter_profile_id=profile if active else "audit_provenance_v1",
                ))
    return _group(rows)


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    value = without_hashes(value)
    if isinstance(value, Mapping):
        return ([(prefix, {})] if not value else [
            item for key in sorted(value) for item in _flatten(value[key], f"{prefix}.{key}" if prefix else str(key))
        ])
    return [(prefix, value)]


def build_top_model_configuration_rows(
    root: str | Path,
    manifest: Mapping[str, Any],
    predictive_leaderboard: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    configs = {case: (config, path) for case, config, path in _configs(Path(root), manifest)}
    ranked = sorted(predictive_leaderboard,
                    key=lambda r: (int(r.get("predictive_rank", 10**9)), str(r.get("case_id", ""))))[:top_k]
    output = []
    for fallback, row in enumerate(ranked, 1):
        case = str(row.get("case_id", ""))
        if case not in configs:
            continue
        config, path = configs[case]
        for key, value in _flatten(config):
            output.append({
                "predictive_rank": int(row.get("predictive_rank", fallback)),
                "case_id": case,
                "model_id": _get(config, "model.model_id", "N/A"),
                "representation_mode": config.get("representation_mode", "N/A"),
                "resolved_config_path": path,
                "config_section": key.split(".", 1)[0],
                "parameter_path": key,
                "resolved_value": _cell(value),
            })
    return output


def build_motion_peak_test_component_rows(resolved_plan: Mapping[str, Any],
                                          manifest: Mapping[str, Any],
                                          *,
                                          study_root: str | Path | None = None) -> list[dict[str, Any]]:
    rows = []
    dataset = resolved_plan.get("ptt_dataset", {})
    dataset = dataset if isinstance(dataset, Mapping) else {}
    rows.append(
        _row(
            "all",
            "dataset_adapter",
            dataset.get("dataset_id"),
            "enabled",
            dataset,
            dataset,
            reporter_profile_id="audit_provenance_v1",
        ))
    for declared in resolved_plan.get("algorithms", ()):
        if isinstance(declared, Mapping):
            module = declared.get("module_id", declared.get("algorithm_id"))
            rows.append(
                _row(
                    "PTT peak ablation",
                    "peak_detector",
                    module,
                    "executed",
                    dataset,
                    declared,
                    reporter_profile_id="beat_detector_recording_v1",
                ))
    detector = resolved_plan.get("motion_detector")
    if isinstance(detector, Mapping):
        rows.append(
            _row(
                "motion study",
                "motion_detector",
                detector.get("model_id"),
                "executed",
                dataset,
                detector,
                reporter_profile_id="motion_route_component_v1",
            ))
    benchmark = resolved_plan.get("denoiser_benchmark")
    if isinstance(benchmark, Mapping):
        for reducer in benchmark.get("reducers", ()):
            rows.append(
                _row(
                    "denoiser benchmark",
                    "denoiser",
                    reducer,
                    "executed",
                    dataset,
                    benchmark,
                    reporter_profile_id="stage5_ecg_ppg_denoiser_v1",
                ))
    validation = resolved_plan.get("validation")
    if isinstance(validation, Mapping):
        rows.append(
            _row(
                "peak validation",
                "peak_validation",
                validation.get("alignment"),
                "executed",
                dataset,
                validation,
                reporter_profile_id="beat_detector_recording_v1",
            ))
    return _group(rows)


def markdown_test_component_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "N/A — no persisted component configuration was available."
    sections = []
    for title, schema in TEST_COMPONENT_VIEW_SCHEMAS:
        fields, labels = zip(*schema, strict=True)
        lines = [f"### {title}", "", "| " + " | ".join(labels) + " |", "|" + "|".join("---" for _ in fields) + "|"]
        lines += [
            "| " + " | ".join(str(row.get(field, "")).replace("|", r"\|").replace("\n", " ") for field in fields) + " |"
            for row in annotate_component_rows(rows)
        ]
        lines += ["", markdown_column_definitions_block(fields, display_labels=labels)]
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def write_test_component_markdown(root: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    target = Path(root) / "TEST_COMPONENTS.md"
    target.write_text(
        "# Test models, modules, inputs, and parameters\n\n" + markdown_test_component_table(rows) + "\n",
        encoding="utf-8",
    )
    return target


__all__ = [
    "TEST_COMPONENT_COLUMNS",
    "TEST_COMPONENT_VIEW_SCHEMAS",
    "TOP_MODEL_CONFIGURATION_COLUMNS",
    "build_motion_peak_test_component_rows",
    "build_pipeline_test_component_rows",
    "build_top_model_configuration_rows",
    "markdown_test_component_table",
    "without_hashes",
    "write_test_component_markdown",
]
