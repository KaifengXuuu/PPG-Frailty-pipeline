"""Auditable multi-resource tuning studies built from ordinary study phases."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
from html import escape as html_escape
import json
import math
from pathlib import Path
import shutil
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

import yaml

from .expand import parse_study_plan
from .progress import NullProgressSink, ProgressSink
from .runner import StudyRunner


_SCHEMA = "ppg_frailty.hyperparameter_study_plan.v1"
_STUDY_TYPES = {
    "successive_halving",
    "dependent_regularization_grid",
    "dependent_channel_ablation",
}


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} keys must be strings")
    return dict(value)


def _strict(value: Any, label: str, fields: set[str]) -> dict[str, Any]:
    result = _mapping(value, label)
    if set(result) != fields:
        raise ValueError(
            f"{label} key mismatch: missing={sorted(fields-set(result))}, "
            f"unknown={sorted(set(result)-fields)}"
        )
    return result


def _indices(value: Any, label: str) -> list[int]:
    result = [int(item) for item in value]
    if not result or len(result) != len(set(result)) or not set(result) <= set(range(5)):
        raise ValueError(f"{label} must be a unique non-empty subset of 0..4")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _validate_overrides(value: Any, label: str) -> dict[str, Any]:
    overrides = _mapping(value, label)
    for path in overrides:
        if len(path.split(".")) < 2 or any(not part for part in path.split(".")):
            raise ValueError(f"{label} has invalid dotted path: {path}")
    json.dumps(overrides, allow_nan=False)
    return overrides


def load_hyperparameter_plan(path: str | Path) -> dict[str, Any]:
    """Load and fully validate the small orchestration schema."""

    source = Path(path).resolve()
    loaded = _mapping(
        yaml.safe_load(source.read_text(encoding="utf-8")), "plan"
    )
    from ..reporting.conclusions import DEFAULT_REPORTING_RANDOM_SEED

    raw = _strict(
        {
            **loaded,
            "search": loaded.get(
                "search",
                {"selection_seed": DEFAULT_REPORTING_RANDOM_SEED},
            ),
        },
        "plan",
        {
            "schema_version", "study", "catalog", "base", "candidates",
            "resource", "execution", "output", "report", "search",
        },
    )
    if raw["schema_version"] != _SCHEMA:
        raise ValueError(f"schema_version must equal {_SCHEMA}")
    study = _strict(
        raw["study"], "study",
        {
            "study_id", "study_type", "purpose", "flow_position",
            "decision_role", "thesis_sections",
        },
    )
    if study["study_type"] not in _STUDY_TYPES:
        raise ValueError(f"unsupported study_type: {study['study_type']}")
    if not str(study["study_id"]).strip() or not str(study["purpose"]).strip():
        raise ValueError("study_id and purpose must be non-empty")
    if not isinstance(study["thesis_sections"], list):
        raise TypeError("study.thesis_sections must be a list")
    catalog = _strict(raw["catalog"], "catalog", {"path", "balance_line"})
    if catalog["balance_line"] not in {"line_a", "line_b"}:
        raise ValueError("catalog.balance_line must be line_a or line_b")
    base = _strict(
        raw["base"], "base",
        {"catalog_entry", "output_group", "profile_id", "common_overrides"},
    )
    base["common_overrides"] = _validate_overrides(
        base["common_overrides"], "base.common_overrides"
    )
    candidates = raw["candidates"]
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("candidates must contain at least two cases")
    normalized_candidates: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for index, value in enumerate(candidates):
        candidate = _strict(
            value, f"candidates[{index}]",
            {"case_id", "label", "overrides", "rationale"},
        )
        case_id = str(candidate["case_id"])
        if not case_id or case_id in identifiers:
            raise ValueError("candidate case_id values must be non-empty and unique")
        identifiers.add(case_id)
        candidate["overrides"] = _validate_overrides(
            candidate["overrides"], f"candidates[{index}].overrides"
        )
        normalized_candidates.append(candidate)
    execution = _strict(
        raw["execution"], "execution",
        {
            "jobs", "device", "parallel_level", "continue_on_error",
            "allow_parallel_deep", "measure_operational_costs",
        },
    )
    _positive_int(execution["jobs"], "execution.jobs")
    if execution["parallel_level"] != "cases":
        raise ValueError("execution.parallel_level must be cases")
    if not str(execution["device"]).startswith("cuda"):
        raise ValueError("deep hyperparameter studies require a CUDA device")
    search = _strict(raw["search"], "search", {"selection_seed"})
    selection_seed = search["selection_seed"]
    if (
        isinstance(selection_seed, bool)
        or not isinstance(selection_seed, int)
        or selection_seed < 0
    ):
        raise ValueError("search.selection_seed must be a non-negative integer")
    search["selection_seed"] = int(selection_seed)
    report_defaults = {
        "classification_tsne_random_state": 42,
        "classification_tsne_perplexity": 30.0,
        "classification_tsne_max_samples": 5000,
        "classification_roc_macro_grid_points": 201,
        "classification_score_histogram_bins": 40,
    }
    report = _strict(
        {**report_defaults, **_mapping(raw["report"], "report")}, "report",
        {
            "top_k", "write_html", "write_static_figures",
            "calibration_bins", "figure_modules", "compact_mean_sd",
            "write_excel_workbook", "classification_tsne_random_state",
            "classification_tsne_perplexity", "classification_tsne_max_samples",
            "classification_roc_macro_grid_points",
            "classification_score_histogram_bins",
        },
    )
    resource = _mapping(raw["resource"], "resource")
    if study["study_type"] == "successive_halving":
        resource = _strict(
            resource, "resource",
            {
                "screen_epochs", "screen_repeats", "screen_folds",
                "promotion_epochs", "promotion_repeats", "promotion_folds",
                "promote_count", "ranking_metric", "tie_break_metric",
            },
        )
        _positive_int(resource["screen_epochs"], "resource.screen_epochs")
        _positive_int(resource["promotion_epochs"], "resource.promotion_epochs")
        resource["screen_repeats"] = _indices(
            resource["screen_repeats"], "resource.screen_repeats"
        )
        resource["screen_folds"] = _indices(
            resource["screen_folds"], "resource.screen_folds"
        )
        resource["promotion_repeats"] = _indices(
            resource["promotion_repeats"], "resource.promotion_repeats"
        )
        resource["promotion_folds"] = _indices(
            resource["promotion_folds"], "resource.promotion_folds"
        )
        promote = _positive_int(resource["promote_count"], "resource.promote_count")
        if promote >= len(normalized_candidates):
            raise ValueError("promote_count must be smaller than candidate count")
    else:
        resource = _strict(
            resource, "resource",
            {"epochs", "repeats", "folds", "ranking_metric", "tie_break_metric"},
        )
        _positive_int(resource["epochs"], "resource.epochs")
        resource["repeats"] = _indices(resource["repeats"], "resource.repeats")
        resource["folds"] = _indices(resource["folds"], "resource.folds")
    output = _strict(raw["output"], "output", {"root"})
    return {
        **raw,
        "study": study,
        "catalog": catalog,
        "base": base,
        "candidates": normalized_candidates,
        "resource": resource,
        "execution": execution,
        "search": search,
        "output": output,
        "report": report,
        "plan_path": str(source),
    }


def _safe_slug(value: str) -> str:
    result = "".join(character if character.isalnum() else "-" for character in value)
    return "-".join(part for part in result.lower().split("-") if part)[:96]


def _new_output(root: Path, study_id: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now():%Y%m%d_%H%M%S}_hyperparameter_{_safe_slug(study_id)}"
    for index in range(1, 1000):
        target = root / (stem if index == 1 else f"{stem}_{index:02d}")
        try:
            target.mkdir()
            return target
        except FileExistsError:
            continue
    raise RuntimeError("could not allocate a unique hyperparameter study directory")


def _dotted_get(value: Mapping[str, Any], path: str) -> Any:
    cursor: Any = value
    for part in path.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            raise KeyError(f"upstream selected configuration lacks {path}")
        cursor = cursor[part]
    return cursor


def _load_upstream(
    directory: str | Path | None,
    *,
    expected_type: str,
) -> dict[str, Any]:
    if directory is None:
        raise ValueError(f"{expected_type} requires --upstream-study")
    root = Path(directory).resolve()
    selected = root / "selected_configuration.json"
    if not selected.is_file():
        raise FileNotFoundError(selected)
    payload = _mapping(json.loads(selected.read_text(encoding="utf-8")), str(selected))
    config = _mapping(payload.get("resolved_config"), "upstream resolved_config")
    if str(config.get("model", {}).get("model_id")) != "InceptionTimeFull":
        raise ValueError("upstream selected model must be InceptionTimeFull")
    return payload


def _phase_plan(
    plan: Mapping[str, Any],
    *,
    phase_id: str,
    candidates: Sequence[Mapping[str, Any]],
    repeats: Sequence[int],
    folds: Sequence[int],
    epochs: int,
    inherited: Mapping[str, Any],
    device: str,
    jobs: int,
) -> Any:
    base = plan["base"]
    cases = []
    for candidate in candidates:
        overrides = {
            **dict(base["common_overrides"]),
            **dict(inherited),
            **dict(candidate["overrides"]),
            "training.fixed_epochs": int(epochs),
        }
        cases.append({
            "case_id": str(candidate["case_id"]),
            "catalog_entry": str(base["catalog_entry"]),
            "screen_profile_id": str(base["profile_id"]),
            "output_group": str(base["output_group"]),
            "overrides": overrides,
            "rationale": str(candidate["rationale"]),
            "formal_profile": None,
        })
    payload = {
        "schema_version": "ppg_frailty.study_plan.v2",
        "study": {
            "study_id": f"{plan['study']['study_id']}__{phase_id}",
            "kind": "catalog_sweep",
            "purpose": f"{plan['study']['purpose']} Phase={phase_id}.",
            "flow_position": str(plan["study"]["flow_position"]),
            "decision_role": str(plan["study"]["decision_role"]),
            "reference_case_id": None,
            "thesis_sections": list(plan["study"]["thesis_sections"]),
        },
        "catalog": {
            "path": str(plan["catalog"]["path"]),
            "balance_line": str(plan["catalog"]["balance_line"]),
            "scope": "selected_ordinary",
        },
        "search": {
            "method": "deterministic_sparse_profiles",
            "selection_seed": int(plan["search"]["selection_seed"]),
            "runtime_sampling": False,
            "interpretation": (
                f"Declared {phase_id} tuning phase; no final-test claim and no "
                "automatic final-model selection."
            ),
            "controlled_factors": [
                "Participant-grouped registered splits and split seeds",
                "V2-core plus selected B0/B2/B7 DL state",
                f"Fixed resource: epochs={epochs}, repeats={list(repeats)}, folds={list(folds)}",
            ],
            "notes": [
                "Candidate ranking is tuning-only and cannot be reported as independent validation.",
                "Every candidate uses the same declared resource within this phase.",
            ],
        },
        "cases": cases,
        "execution": {
            "repeats": list(repeats),
            "folds": list(folds),
            "jobs": int(jobs),
            "device": device,
            "parallel_level": "cases",
            "continue_on_error": False,
            "allow_parallel_deep": False,
            "measure_operational_costs": bool(
                plan["execution"]["measure_operational_costs"]
            ),
        },
        "output": {"root": "."},
        "report": dict(plan["report"]),
    }
    return parse_study_plan(payload)


def _run_phase(
    plan: Mapping[str, Any],
    *,
    output: Path,
    phase_id: str,
    candidates: Sequence[Mapping[str, Any]],
    repeats: Sequence[int],
    folds: Sequence[int],
    epochs: int,
    inherited: Mapping[str, Any],
    pipeline_root: Path,
    device: str,
    jobs: int,
    progress_sink: ProgressSink,
) -> tuple[Any, Path, list[dict[str, Any]]]:
    standard = _phase_plan(
        plan,
        phase_id=phase_id,
        candidates=candidates,
        repeats=repeats,
        folds=folds,
        epochs=epochs,
        inherited=inherited,
        device=device,
        jobs=jobs,
    )
    runner = StudyRunner(pipeline_root=pipeline_root, progress_sink=progress_sink)
    result = runner.run(standard, output_root=output / "phases" / phase_id)
    from ppg_frailty.reporting import generate_study_report

    # A failed phase still receives the complete ordinary report before the
    # orchestration fails closed and refuses to rank incomplete candidates.
    generate_study_report(result.output_directory)
    if result.status != "passed":
        raise RuntimeError(f"{phase_id} did not complete: {result.status}")
    ranked = _rank_case_records(
        result.case_records,
        metric=str(plan["resource"]["ranking_metric"]),
        tie_break=str(plan["resource"]["tie_break_metric"]),
        expected_cells=len(repeats) * len(folds),
    )
    return result, result.output_directory, ranked


def _rank_case_records(
    records: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    tie_break: str,
    expected_cells: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        result = record.get("result")
        cells = result.get("cell_results", []) if isinstance(result, Mapping) else []
        if record.get("status") != "passed" or len(cells) != expected_cells:
            raise RuntimeError(
                f"candidate {record.get('case_id')} lacks complete ranking evidence"
            )
        primary = [float(cell["metrics"][metric]) for cell in cells]
        secondary = [float(cell["metrics"][tie_break]) for cell in cells]
        if not all(math.isfinite(value) for value in (*primary, *secondary)):
            raise RuntimeError("candidate ranking metrics must be finite")
        rows.append({
            "case_id": str(record["case_id"]),
            "cell_count": len(cells),
            f"{metric}_mean": mean(primary),
            f"{metric}_sd": pstdev(primary),
            f"{metric}_percent_mean_sd": (
                f"{100.0*mean(primary):.1f} ± {100.0*pstdev(primary):.1f}"
            ),
            f"{tie_break}_mean": mean(secondary),
            f"{tie_break}_sd": pstdev(secondary),
            f"{tie_break}_percent_mean_sd": (
                f"{100.0*mean(secondary):.1f} ± {100.0*pstdev(secondary):.1f}"
            ),
        })
    rows.sort(
        key=lambda row: (
            -float(row[f"{metric}_mean"]),
            -float(row[f"{tie_break}_mean"]),
            str(row["case_id"]),
        )
    )
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def _completion_candidates(
    plan: Mapping[str, Any],
    promoted_ranking: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return declared candidates that lack the full promotion resource."""

    if str(plan["study"]["study_type"]) != "successive_halving":
        raise ValueError("full-grid completion requires a successive-halving study")
    declared = {
        str(candidate["case_id"]): dict(candidate)
        for candidate in plan["candidates"]
    }
    promoted_ids = [str(row["case_id"]) for row in promoted_ranking]
    unknown = sorted(set(promoted_ids) - set(declared))
    if unknown:
        raise ValueError(f"promotion ranking contains unknown cases: {unknown}")
    expected_promoted = int(plan["resource"]["promote_count"])
    if len(promoted_ids) != expected_promoted or len(set(promoted_ids)) != len(
        promoted_ids
    ):
        raise ValueError(
            "promotion ranking does not match the declared promote_count"
        )
    promoted_set = set(promoted_ids)
    return [
        declared[str(candidate["case_id"])]
        for candidate in plan["candidates"]
        if str(candidate["case_id"]) not in promoted_set
    ]


def _merge_equal_resource_rankings(
    groups: Sequence[Sequence[Mapping[str, Any]]],
    *,
    metric: str,
    tie_break: str,
    expected_case_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Merge disjoint complete-CV rankings that used the identical resource."""

    merged: list[dict[str, Any]] = []
    for group in groups:
        merged.extend(dict(row) for row in group)
    observed_ids = [str(row["case_id"]) for row in merged]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("full-CV ranking groups overlap")
    if set(observed_ids) != set(str(value) for value in expected_case_ids):
        raise ValueError("full-CV ranking groups do not cover every declared candidate")
    cell_counts = {int(row["cell_count"]) for row in merged}
    if len(cell_counts) != 1:
        raise ValueError("full-CV ranking groups used unequal cell resources")
    for row in merged:
        if not math.isfinite(float(row[f"{metric}_mean"])) or not math.isfinite(
            float(row[f"{tie_break}_mean"])
        ):
            raise ValueError("full-CV ranking metrics must be finite")
        row["metric_source"] = "equal_weight_fold_cell_mean_for_selection"
        row["selection_role"] = (
            "exhaustive_full_grid_selection_evidence_after_completion"
        )
    merged.sort(
        key=lambda row: (
            -float(row[f"{metric}_mean"]),
            -float(row[f"{tie_break}_mean"]),
            str(row["case_id"]),
        )
    )
    for rank, row in enumerate(merged, start=1):
        row["rank"] = rank
    return merged


def _resolved_config(phase_dir: Path, case_id: str) -> tuple[Path, dict[str, Any]]:
    manifest = json.loads((phase_dir / "study_manifest.json").read_text(encoding="utf-8"))
    matches = [row for row in manifest["cases"] if row["case_id"] == case_id]
    if len(matches) != 1:
        raise RuntimeError(f"cannot resolve selected case {case_id}")
    path = phase_dir / matches[0]["resolved_config_path"]
    config = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), str(path))
    return path, config


def _component_case_ids(value: Any) -> set[str]:
    """Normalize the persisted TEST_COMPONENTS case-membership cell."""

    if isinstance(value, str):
        return {item.strip() for item in value.split(";") if item.strip()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return {str(item).strip() for item in value if str(item).strip()}
    return set()


def _component_json_mapping(value: Any, label: str) -> dict[str, Any]:
    """Decode one lossless TEST_COMPONENTS JSON cell and fail closed."""

    payload = json.loads(value) if isinstance(value, str) else value
    return _mapping(payload, label)


def _candidate_component_rows(
    plan: Mapping[str, Any],
    phase_directories: Mapping[str, Path],
    *,
    inherited: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Project every candidate from its own persisted nested execution contract.

    Full-resource phases take precedence over reduced screening.  Both the
    resolved config and the nested ordinary report's TEST_COMPONENTS table are
    required and cross-checked; a selected winner is never used as a template
    for another candidate.
    """

    declared = {
        str(candidate["case_id"]): dict(candidate)
        for candidate in plan["candidates"]
    }
    phase_priority = ("full_cv", "completion", "promotion", "screen")
    ordered_phases = [
        (phase, phase_directories[phase])
        for phase in phase_priority
        if phase in phase_directories
    ]
    ordered_phases.extend(
        (phase, directory)
        for phase, directory in sorted(phase_directories.items())
        if phase not in phase_priority
    )
    resolved: dict[str, dict[str, Any]] = {}
    for phase, raw_directory in ordered_phases:
        directory = Path(raw_directory).resolve()
        manifest_path = directory / "study_manifest.json"
        components_path = directory / "tables" / "test_components.json"
        if not manifest_path.is_file():
            continue
        manifest = _mapping(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            str(manifest_path),
        )
        manifest_cases = [
            _mapping(item, f"{manifest_path}:case")
            for item in manifest.get("cases", ())
            if isinstance(item, Mapping)
        ]
        relevant = [
            item
            for item in manifest_cases
            if str(item.get("case_id")) in declared
            and str(item.get("case_id")) not in resolved
        ]
        if not relevant:
            continue
        if not components_path.is_file():
            raise FileNotFoundError(
                f"candidate provenance requires persisted {components_path}"
            )
        component_payload = json.loads(components_path.read_text(encoding="utf-8"))
        if not isinstance(component_payload, list):
            raise TypeError(f"{components_path} must contain a list")
        components = [
            _mapping(item, f"{components_path}:row")
            for item in component_payload
            if isinstance(item, Mapping)
        ]
        for manifest_case in relevant:
            case_id = str(manifest_case["case_id"])
            relative_config = Path(str(manifest_case.get("resolved_config_path", "")))
            if relative_config.is_absolute() or not relative_config.parts:
                raise ValueError(
                    f"candidate {case_id} has an invalid resolved_config_path"
                )
            config_path = (directory / relative_config).resolve()
            try:
                config_path.relative_to(directory)
            except ValueError as error:
                raise ValueError(
                    f"candidate {case_id} resolved config escapes its phase"
                ) from error
            if not config_path.is_file():
                raise FileNotFoundError(config_path)
            config = _mapping(
                yaml.safe_load(config_path.read_text(encoding="utf-8")),
                str(config_path),
            )

            def one_component(role: str) -> dict[str, Any]:
                matches = [
                    item
                    for item in components
                    if str(item.get("component_role")) == role
                    and case_id in _component_case_ids(
                        item.get("participating_cases")
                    )
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"candidate {case_id} requires exactly one persisted "
                        f"{role} TEST_COMPONENTS row in {phase}; found {len(matches)}"
                    )
                return matches[0]

            classifier = one_component("classifier")
            trainer = one_component("trainer")
            classifier_input = _component_json_mapping(
                classifier.get("input_data"),
                f"{components_path}:{case_id}:classifier input_data",
            )
            classifier_fixed = _component_json_mapping(
                classifier.get("fixed_parameters"),
                f"{components_path}:{case_id}:classifier fixed_parameters",
            )
            trainer_fixed = _component_json_mapping(
                trainer.get("fixed_parameters"),
                f"{components_path}:{case_id}:trainer fixed_parameters",
            )
            model = _mapping(config.get("model"), f"{config_path}:model")
            training = _mapping(config.get("training"), f"{config_path}:training")
            signal = _mapping(config.get("signal"), f"{config_path}:signal")
            windows = _mapping(config.get("windows"), f"{config_path}:windows")
            raw_window = _mapping(
                windows.get("raw_dl"), f"{config_path}:windows.raw_dl"
            )
            if classifier_fixed != model:
                raise ValueError(
                    f"candidate {case_id} classifier TEST_COMPONENTS parameters "
                    "differ from its resolved model config"
                )
            if trainer_fixed != training:
                raise ValueError(
                    f"candidate {case_id} trainer TEST_COMPONENTS parameters "
                    "differ from its resolved training config"
                )
            channels = list(model.get("input_channel_order", ()))
            component_channels = list(classifier_input.get("channels", ()))
            if not channels or channels != component_channels:
                raise ValueError(
                    f"candidate {case_id} channel provenance differs between "
                    "resolved config and TEST_COMPONENTS"
                )
            if int(model.get("input_channels", -1)) != len(channels):
                raise ValueError(
                    f"candidate {case_id} input_channels does not match channel order"
                )
            component_window = classifier_input.get("window")
            if not isinstance(component_window, Mapping) or dict(
                component_window
            ) != raw_window:
                raise ValueError(
                    f"candidate {case_id} raw window differs between resolved "
                    "config and TEST_COMPONENTS"
                )
            signal_view = str(classifier_input.get("signal_view", "")).strip()
            if not signal_view:
                raise ValueError(
                    f"candidate {case_id} classifier input view is not persisted"
                )
            dl_resampling = _mapping(
                signal.get("dl_resampling"),
                f"{config_path}:signal.dl_resampling",
            )
            internal_fs_hz = float(signal.get("internal_fs_hz"))
            sampling_rate_hz = (
                float(dl_resampling["target_fs_hz"])
                if bool(dl_resampling.get("enabled"))
                else internal_fs_hz
            )
            if not math.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0.0:
                raise ValueError(
                    f"candidate {case_id} has invalid persisted model sampling rate"
                )
            input_data = {
                **classifier_input,
                "sampling_rate_hz": sampling_rate_hz,
                "model_sampling_rate_hz": sampling_rate_hz,
                "pipeline_fs_hz": internal_fs_hz,
                "provenance_phase": phase,
                "resolved_config_path": config_path.relative_to(directory).as_posix(),
                "test_components_path": "tables/test_components.json",
            }
            fixed_parameters = {
                "persisted_provenance": {
                    "phase": phase,
                    "phase_directory": directory.name,
                    "resolved_config_path": config_path.relative_to(
                        directory
                    ).as_posix(),
                    "test_components_path": "tables/test_components.json",
                },
                "representation_mode": config.get("representation_mode"),
                "roles": config.get("roles"),
                "model": model,
                "training": training,
                "windows": windows,
                "signal": signal,
                "aggregation": config.get("aggregation"),
                "hyperparameter_declaration": {
                    "common_overrides": dict(plan["base"]["common_overrides"]),
                    "inherited": dict(inherited),
                    "candidate_overrides": dict(declared[case_id]["overrides"]),
                    "resource_contract": dict(plan["resource"]),
                },
            }
            resolved[case_id] = {
                "participating_cases": case_id,
                "component_role": "classifier_tuning_candidate",
                "module_id": str(classifier.get("module_id")),
                "execution_state": str(classifier.get("execution_state")),
                "input_data": json.dumps(
                    input_data, ensure_ascii=False, sort_keys=True
                ),
                "fixed_parameters": json.dumps(
                    fixed_parameters, ensure_ascii=False, sort_keys=True
                ),
                "algorithm_kernel_description": str(
                    classifier.get("algorithm_kernel_description", "")
                ),
                "reporter_profile_id": str(
                    classifier.get("reporter_profile_id", "")
                ),
                "model_reporter_extension_id": str(
                    classifier.get("model_reporter_extension_id", "")
                ),
                "algorithm_references": str(
                    classifier.get("algorithm_references", "")
                ),
            }
    missing = sorted(set(declared) - set(resolved))
    if missing:
        raise ValueError(
            "candidate provenance is missing persisted resolved config/"
            f"TEST_COMPONENTS evidence: {missing}"
        )
    return [resolved[str(candidate["case_id"])] for candidate in plan["candidates"]]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = list(rows)
    fields = list(dict.fromkeys(str(key) for row in values for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in values:
            writer.writerow({
                key: (
                    json.dumps(row.get(key), ensure_ascii=False, sort_keys=True)
                    if isinstance(row.get(key), (dict, list, tuple))
                    else row.get(key)
                )
                for key in fields
            })


def _write_table_pair(
    output: Path,
    *,
    name: str,
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    selection_role: str | None = None,
) -> list[dict[str, Any]]:
    tables = output / "tables"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    normalized_rows = [
        {
            **dict(row),
            "metric_source": row.get(
                "metric_source", "equal_weight_fold_cell_mean_for_selection"
            ),
            "selection_role": selection_role
            or row.get(
                "selection_role", "declared_resource_orchestration_selection_evidence"
            ),
        }
        for row in rows
    ]
    display_rows = [
        {
            "rank": row["rank"],
            "case_id": row["case_id"],
            "cell_count": row["cell_count"],
            "metric_source": row.get(
                "metric_source", "equal_weight_fold_cell_mean_for_selection"
            ),
            "selection_role": row.get(
                "selection_role", "declared_resource_orchestration_selection_evidence"
            ),
            f"{metric}_mean_sd_percent": row[f"{metric}_percent_mean_sd"],
            **{
                key: value
                for key, value in row.items()
                if key.endswith("_percent_mean_sd")
                and key != f"{metric}_percent_mean_sd"
            },
            **{
                key: row.get(key)
                for key in (
                    "macro_roc_auc_ovr",
                    "macro_pr_auc_ovr",
                    "expected_calibration_error",
                    "worst_fold_balanced_accuracy",
                    "worst_class_f1",
                    "balanced_accuracy_lcb95",
                )
                if key in row
            },
        }
        for row in normalized_rows
    ]
    _write_csv(tables / f"{name}.csv", display_rows)
    (tables / f"{name}.json").write_text(
        json.dumps(normalized_rows, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [str(row["case_id"]) for row in normalized_rows]
        values = [100.0 * float(row[f"{metric}_mean"]) for row in normalized_rows]
        raw_errors = [row.get(f"{metric}_sd") for row in normalized_rows]
        errors = [
            0.0 if value is None else 100.0 * float(value)
            for value in raw_errors
        ]
        figure, axis = plt.subplots(
            figsize=(max(7.0, 0.8 * len(normalized_rows)), 4.8)
        )
        axis.bar(
            range(len(rows)),
            values,
            yerr=errors if any(value is not None for value in raw_errors) else None,
            capsize=4,
        )
        axis.set_xticks(range(len(rows)), labels, rotation=35, ha="right")
        axis.set_ylabel(f"{metric} (%)")
        axis.set_title(name.replace("_", " ").title())
        figure.tight_layout()
        figure.savefig(figures / f"{name}.png", dpi=160)
        plt.close(figure)
    except Exception as error:  # Plot absence is explicit, never silent.
        (figures / f"{name}.NA.txt").write_text(
            f"{type(error).__name__}: {error}\n", encoding="utf-8"
        )
    return display_rows


def _root_ranking_selection_role(
    plan: Mapping[str, Any], table_name: str
) -> str:
    """Label root ranking evidence from the declared study design, not stale output."""

    study_type = str(plan["study"]["study_type"])
    if study_type == "successive_halving":
        if table_name == "screen_ranking":
            return "reduced_resource_screening_evidence_not_full_cv_selection"
        if table_name == "promotion_ranking":
            return "promoted_subset_full_cv_selection_evidence"
        if table_name in {"completion_ranking", "nonpromoted_full_cv_ranking"}:
            return "completion_subset_full_cv_evidence_not_standalone_selection"
        if table_name == "all_candidates_full_cv_ranking":
            return "exhaustive_full_grid_selection_evidence_after_completion"
        return "declared_successive_halving_resource_orchestration_evidence"
    if study_type == "dependent_regularization_grid":
        return "declared_full_cv_equal_weight_fold_cell_ranking"
    if study_type == "dependent_channel_ablation":
        return "declared_full_cv_channel_ablation_ranking"
    return "declared_full_cv_equal_weight_fold_cell_ranking"


def _design_scope_conclusion(
    plan: Mapping[str, Any], *, selected_case_id: str
) -> dict[str, Any]:
    """Describe whether candidate contrasts identify profiles or one factor."""

    candidates = list(plan["candidates"])
    common = dict(plan["base"]["common_overrides"])

    def effective_override(
        candidate: Mapping[str, Any], path: str
    ) -> tuple[str, str]:
        overrides = candidate["overrides"]
        if path in overrides:
            value = overrides[path]
        elif path in common:
            value = common[path]
        else:
            return ("not_declared", "")
        return (
            "value",
            json.dumps(value, ensure_ascii=False, sort_keys=True),
        )

    paths = sorted(
        {
            str(path)
            for candidate in candidates
            for path in candidate["overrides"]
        }
    )
    varying_paths = [
        path
        for path in paths
        if len(
            {
                effective_override(candidate, path)
                for candidate in candidates
            }
        )
        > 1
    ]
    if len(varying_paths) > 1:
        finding = (
            "Candidate overrides simultaneously vary "
            + ", ".join(f"`{path}`" for path in varying_paths)
            + ". This is a joint profile/grid comparison: it estimates whole "
            "declared-profile differences and does not identify the causal effect "
            "of any single field."
        )
        confidence = "design_scope_joint_profile_nonfactorial"
        selection_effect = "profile_level_selection_only_no_single_factor_claim"
    elif varying_paths:
        finding = (
            f"Candidate overrides vary only `{varying_paths[0]}`; the declared "
            "contrast is a single-factor comparison within the persisted shared contract."
        )
        confidence = "design_scope_single_factor"
        selection_effect = "single_declared_factor_interpretation_allowed"
    else:
        finding = (
            "No candidate-varying override field was found; factor-level "
            "interpretation is unavailable."
        )
        confidence = "design_scope_unresolved"
        selection_effect = "no_factor_claim"
    return {
        "angle": "design_scope",
        "leading_or_selected_case": selected_case_id,
        "finding": finding,
        "confidence": confidence,
        "selection_effect": selection_effect,
    }


def _participant_oof_rankings(
    phase_directories: Mapping[str, Path],
) -> dict[str, list[dict[str, Any]]]:
    """Read nested report evidence without changing orchestration selection."""

    def resolved_sd(
        item: Mapping[str, Any], sample_key: str, population_key: str
    ) -> tuple[float | None, str]:
        sample = item.get(sample_key)
        if sample is not None and math.isfinite(float(sample)):
            return float(sample), "sample_sd_ddof1_across_repeats"
        population = item.get(population_key)
        if population is not None and math.isfinite(float(population)):
            return float(population), "population_sd_legacy_fallback"
        return None, "unavailable_fewer_than_two_repeats"

    def display_mean_sd(value: float, sd: float | None, repeats: int) -> str:
        if sd is None:
            suffix = f"; n={repeats} repeat" if repeats == 1 else ""
            return f"{100.0 * value:.1f} (SD N/A{suffix})"
        return f"{100.0 * value:.1f} ± {100.0 * sd:.1f}"

    output: dict[str, list[dict[str, Any]]] = {}
    for phase, directory in phase_directories.items():
        source = directory / "tables" / "case_summary.json"
        if not source.is_file():
            continue
        rows: list[dict[str, Any]] = []
        for item in json.loads(source.read_text(encoding="utf-8")):
            if not isinstance(item, Mapping):
                continue
            balanced_accuracy = float(item["participant_mean_balanced_accuracy"])
            balanced_accuracy_sd, ba_sd_estimator = resolved_sd(
                item,
                "repeat_balanced_accuracy_sample_sd",
                "repeat_balanced_accuracy_population_sd",
            )
            macro_f1 = float(item["participant_mean_macro_f1"])
            macro_f1_sd, f1_sd_estimator = resolved_sd(
                item,
                "repeat_macro_f1_sample_sd",
                "repeat_macro_f1_population_sd",
            )
            repeat_count = int(item.get("repeat_count", 0) or 0)
            rows.append(
                {
                    "case_id": str(item["case_id"]),
                    "cell_count": int(item["fold_cell_count"]),
                    "balanced_accuracy_mean": balanced_accuracy,
                    "balanced_accuracy_sd": balanced_accuracy_sd,
                    "balanced_accuracy_percent_mean_sd": display_mean_sd(
                        balanced_accuracy, balanced_accuracy_sd, repeat_count
                    ),
                    "macro_f1_mean": macro_f1,
                    "macro_f1_sd": macro_f1_sd,
                    "macro_f1_percent_mean_sd": display_mean_sd(
                        macro_f1, macro_f1_sd, repeat_count
                    ),
                    "macro_roc_auc_ovr": item.get(
                        "participant_mean_macro_roc_auc_ovr"
                    ),
                    "macro_pr_auc_ovr": item.get(
                        "participant_mean_macro_pr_auc_ovr"
                    ),
                    "expected_calibration_error": item.get(
                        "expected_calibration_error"
                    ),
                    "worst_fold_balanced_accuracy": item.get(
                        "worst_fold_balanced_accuracy"
                    ),
                    "worst_class_f1": item.get("worst_class_f1"),
                    "balanced_accuracy_lcb95": item.get(
                        "balanced_accuracy_lcb95"
                    ),
                    "metric_source": (
                        "participant_oof_recomputed_equal_repeat_mean"
                    ),
                    "selection_role": (
                        "descriptive_sensitivity_not_orchestration_selection"
                    ),
                    "balanced_accuracy_sd_estimator": ba_sd_estimator,
                    "macro_f1_sd_estimator": f1_sd_estimator,
                    "sd_estimator": (
                        ba_sd_estimator
                        if ba_sd_estimator == f1_sd_estimator
                        else "metric_specific_see_estimator_columns"
                    ),
                }
            )
        rows.sort(
            key=lambda row: (
                -float(row["balanced_accuracy_mean"]),
                -float(row["macro_f1_mean"]),
                str(row["case_id"]),
            )
        )
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        output[f"{phase}_participant_oof_ranking"] = rows
    if "promotion_participant_oof_ranking" in output and (
        "completion_participant_oof_ranking" in output
    ):
        combined = [
            dict(row)
            for name in (
                "promotion_participant_oof_ranking",
                "completion_participant_oof_ranking",
            )
            for row in output[name]
        ]
        case_ids = [str(row["case_id"]) for row in combined]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("participant-OOF completion rankings overlap")
        combined.sort(
            key=lambda row: (
                -float(row["balanced_accuracy_mean"]),
                -float(row["macro_f1_mean"]),
                str(row["case_id"]),
            )
        )
        for rank, row in enumerate(combined, start=1):
            row["rank"] = rank
            row["selection_role"] = (
                "descriptive_full_grid_sensitivity_not_orchestration_selection"
            )
        output["all_candidates_full_cv_participant_oof_ranking"] = combined
    return output


def _write_workbook(output: Path, tables: Sequence[str]) -> None:
    from ..reporting.tabular import ReportTable, write_excel_workbook

    report_tables = []
    for name in tables:
        path = output / "tables" / f"{name}.csv"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        report_tables.append(ReportTable(name=name, rows=rows, compact=False))
    write_excel_workbook(output / "tables" / "report_tables.xlsx", report_tables)


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    from ..reporting.tabular import markdown_column_definitions_block

    values = list(rows)
    if not values:
        return "N/A — no rows."
    fields = list(dict.fromkeys(str(key) for row in values for key in row))
    if len(fields) > 8:
        raise ValueError(
            "human-facing hyperparameter table has "
            f"{len(fields)} columns; maximum is 8"
        )
    lines = [
        "| " + " | ".join(fields) + " |",
        "|" + "|".join("---" for _ in fields) + "|",
    ]
    for row in values:
        lines.append(
            "| "
            + " | ".join(
                str(row.get(field, "")).replace("|", r"\|").replace("\n", " ")
                for field in fields
            )
            + " |"
        )
    lines.extend(("", markdown_column_definitions_block(fields)))
    return "\n".join(lines)


def _html_table(rows: Sequence[Mapping[str, Any]]) -> str:
    from ..reporting.tabular import html_column_definitions_block

    values = list(rows)
    if not values:
        return "<p>N/A — no rows.</p>"
    fields = list(dict.fromkeys(str(key) for row in values for key in row))
    if len(fields) > 8:
        raise ValueError(
            "human-facing hyperparameter table has "
            f"{len(fields)} columns; maximum is 8"
        )
    headings = "".join(f"<th>{html_escape(field)}</th>" for field in fields)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{html_escape(str(row.get(field, '')))}</td>" for field in fields
        )
        + "</tr>"
        for row in values
    )
    return (
        f"<table><thead><tr>{headings}</tr></thead><tbody>{body}</tbody></table>"
        + html_column_definitions_block(fields)
    )


def _project_table_rows(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> list[dict[str, Any]]:
    """Return a presentation-only fixed-schema projection."""

    selected = tuple(dict.fromkeys(str(field) for field in fields))
    if len(selected) > 8:
        raise ValueError(
            f"hyperparameter table projection has {len(selected)} columns; maximum is 8"
        )
    return [
        {field: row.get(field, "") for field in selected}
        for row in rows
    ]


def _narrow_table_views(
    rows: Sequence[Mapping[str, Any]],
    *,
    identity_fields: Sequence[str],
    semantic_groups: Sequence[tuple[str, Sequence[str]]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Split one lossless row schema into semantic views of at most eight columns.

    Identity columns are repeated across views. Every source field is represented;
    fields not assigned to a semantic group are emitted in explicit additional
    audit views rather than being silently dropped.
    """

    values = list(rows)
    if not values:
        return [("", [])]
    all_fields = list(
        dict.fromkeys(str(key) for row in values for key in row)
    )
    identities = [field for field in identity_fields if field in all_fields]
    if len(identities) >= 8:
        raise ValueError(
            "hyperparameter table identity projection must use <8 columns"
        )
    covered = set(identities)
    views: list[tuple[str, list[dict[str, Any]]]] = []
    width = 8 - len(identities)

    def append_chunks(title: str, candidates: Sequence[str]) -> None:
        fields = [
            field
            for field in candidates
            if field in all_fields and field not in covered
        ]
        for offset in range(0, len(fields), width):
            chunk = fields[offset : offset + width]
            suffix = (
                f" ({offset // width + 1}/{math.ceil(len(fields) / width)})"
                if len(fields) > width
                else ""
            )
            views.append(
                (
                    title + suffix,
                    _project_table_rows(values, [*identities, *chunk]),
                )
            )
            covered.update(chunk)

    for title, fields in semantic_groups:
        append_chunks(title, fields)
    append_chunks(
        "Additional audit fields",
        [field for field in all_fields if field not in covered],
    )
    if not views:
        views.append(("", _project_table_rows(values, identities)))
    return views


def _markdown_narrow_views(
    rows: Sequence[Mapping[str, Any]],
    *,
    identity_fields: Sequence[str],
    semantic_groups: Sequence[tuple[str, Sequence[str]]],
    heading_level: int = 3,
) -> str:
    views = _narrow_table_views(
        rows,
        identity_fields=identity_fields,
        semantic_groups=semantic_groups,
    )
    lines: list[str] = []
    for title, projected in views:
        if title:
            lines.extend((f"{'#' * heading_level} {title}", ""))
        lines.extend((_markdown_table(projected), ""))
    return "\n".join(lines).rstrip()


def _html_narrow_views(
    rows: Sequence[Mapping[str, Any]],
    *,
    identity_fields: Sequence[str],
    semantic_groups: Sequence[tuple[str, Sequence[str]]],
    heading_level: int = 3,
) -> str:
    views = _narrow_table_views(
        rows,
        identity_fields=identity_fields,
        semantic_groups=semantic_groups,
    )
    return "\n".join(
        (
            f"<h{heading_level}>{html_escape(title)}</h{heading_level}>"
            if title
            else ""
        )
        + _html_table(projected)
        for title, projected in views
    )


def _html_fixed_schema_views(
    rows: Sequence[Mapping[str, Any]],
    schemas: Sequence[tuple[str, Sequence[tuple[str, str]]]],
    *,
    heading_level: int = 3,
) -> str:
    """Render shared reporter/component schemas as fixed HTML projections."""

    if not rows:
        return "<p>N/A — no rows.</p>"
    output: list[str] = []
    for title, schema in schemas:
        fields = [field for field, _label in schema]
        output.extend(
            (
                f"<h{heading_level}>{html_escape(title)}</h{heading_level}>",
                _html_table(_project_table_rows(rows, fields)),
            )
        )
    return "\n".join(output)


def _ranking_view_groups(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    fields = list(dict.fromkeys(str(key) for row in rows for key in row))
    return (
        (
            "Performance",
            tuple(
                field
                for field in fields
                if field == "cell_count" or field.endswith("_percent_mean_sd")
            ),
        ),
        (
            "Robustness and discrimination",
            (
                "macro_roc_auc_ovr",
                "macro_pr_auc_ovr",
                "expected_calibration_error",
                "worst_fold_balanced_accuracy",
                "worst_class_f1",
                "balanced_accuracy_lcb95",
            ),
        ),
        ("Evidence role", ("metric_source", "selection_role")),
    )


_PER_CLASS_VIEW_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Evaluation context",
        (
            "evaluation_id",
            "aggregation_level",
            "class_label",
            "result_applicability",
        ),
    ),
    (
        "Confusion counts and support",
        (
            "true_positive",
            "false_positive",
            "true_negative",
            "false_negative",
            "support",
            "predicted_support",
        ),
    ),
    (
        "Per-class rates",
        (
            "precision",
            "sensitivity",
            "recall",
            "specificity",
            "balanced_accuracy_ovr",
            "f1",
        ),
    ),
    (
        "Discrimination and retained observations",
        (
            "roc_auc_ovr",
            "pr_auc_ovr",
            "observation_count",
            "input_observation_count",
            "retained_observation_count",
            "excluded_observation_count",
        ),
    ),
    (
        "Metric applicability and provenance",
        (
            "probability_metric_applicability",
            "metric_scope",
            "metric_source",
            "prediction_rule_source",
        ),
    ),
)


_REPEAT_DELTA_VIEW_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Matched roster and contract",
        (
            "reference_case_id",
            "candidate_case_id",
            "split_seed",
            "matched_participant_count",
            "comparison_contract_status",
            "difference_direction",
        ),
    ),
    (
        "Balanced-accuracy difference",
        (
            "reference_balanced_accuracy",
            "candidate_balanced_accuracy",
            "balanced_accuracy_delta",
        ),
    ),
    (
        "Macro-F1 difference",
        (
            "reference_macro_f1",
            "candidate_macro_f1",
            "macro_f1_delta",
        ),
    ),
    (
        "Macro ROC-AUC difference",
        (
            "reference_macro_roc_auc_ovr",
            "candidate_macro_roc_auc_ovr",
            "macro_roc_auc_ovr_delta",
        ),
    ),
    (
        "Comparison provenance",
        (
            "comparison_family",
            "comparison_role",
            "matched_roster_sha256",
            "automatic_selection",
        ),
    ),
)


def _reproducibility_rows(
    plan: Mapping[str, Any],
    phase_directories: Mapping[str, Path] | None = None,
) -> list[dict[str, Any]]:
    resource = plan["resource"]
    if plan["study"]["study_type"] == "successive_halving":
        phases = (
            (
                "screen_5epoch_reduced_cv", resource["screen_epochs"],
                resource["screen_repeats"], resource["screen_folds"],
            ),
            (
                "promoted_full_cv", resource["promotion_epochs"],
                resource["promotion_repeats"], resource["promotion_folds"],
            ),
        )
        if phase_directories is not None and "completion" in phase_directories:
            phases = (*phases, (
                "nonpromoted_full_cv",
                resource["promotion_epochs"],
                resource["promotion_repeats"],
                resource["promotion_folds"],
            ))
    else:
        phases = (("full_cv", resource["epochs"], resource["repeats"], resource["folds"]),)
    return [
        {
            "phase": phase,
            "fixed_epochs": int(epochs),
            "repeat_indices": list(repeats),
            "fold_indices": list(folds),
            "split_seeds": [42 + 10000 * int(repeat) for repeat in repeats],
            "training_seeds": [42 + 10000 * int(repeat) for repeat in repeats],
            "training_seed_policy": "outer_cv_repeat_seed_equals_split_seed",
            "split_group": "participant_id",
            "selection_scope": "development_tuning_only_not_final_test",
        }
        for phase, epochs, repeats, folds in phases
    ]


def _persisted_reporting_selection_seed(
    phases: Sequence[tuple[str, Path]],
    *,
    default: int,
) -> int:
    """Resolve the report RNG seed from persisted nested phase plans."""

    seeds: set[int] = set()
    for phase, directory in phases:
        source = Path(directory) / "study_plan.yaml"
        if not source.is_file():
            continue
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError(f"{source} must contain a mapping")
        search = payload.get("search")
        if not isinstance(search, Mapping) or "selection_seed" not in search:
            continue
        value = search["selection_seed"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"phase {phase} has an invalid persisted search.selection_seed"
            )
        seeds.add(int(value))
    if len(seeds) > 1:
        raise ValueError(
            f"equal/full-resource phases disagree on search.selection_seed: {sorted(seeds)}"
        )
    return next(iter(seeds), int(default))


def _final_classification_evidence(
    phase_directories: Mapping[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[tuple[str, Path]]]:
    """Collect only equal/full-resource phase evidence for root conclusions."""

    if "full_cv" in phase_directories:
        selected_phases = [("full_cv", phase_directories["full_cv"])]
    elif "completion" in phase_directories and "promotion" in phase_directories:
        selected_phases = [
            ("promotion", phase_directories["promotion"]),
            ("completion", phase_directories["completion"]),
        ]
    elif "promotion" in phase_directories:
        selected_phases = [("promotion", phase_directories["promotion"])]
    else:
        selected_phases = []
    summaries: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    seen_cases: set[str] = set()
    expected_full_resource_roster: frozenset[tuple[Any, ...]] | None = None
    for phase, directory in selected_phases:
        summary_path = directory / "tables" / "case_summary.json"
        prediction_path = directory / "tables" / "classification_prediction_scores.json"
        if not summary_path.is_file() or not prediction_path.is_file():
            nested_manifest_path = directory / "study_manifest.json"
            if nested_manifest_path.is_file():
                nested_manifest = json.loads(
                    nested_manifest_path.read_text(encoding="utf-8")
                )
                if nested_manifest.get("status") == "passed":
                    raise ValueError(
                        f"passed phase {phase} lacks case-summary or classification-prediction evidence"
                    )
            continue
        phase_summaries = [
            dict(row) for row in json.loads(summary_path.read_text(encoding="utf-8"))
        ]
        phase_case_id_list = [str(row.get("case_id")) for row in phase_summaries]
        phase_case_ids = set(phase_case_id_list)
        if (
            not phase_case_ids
            or "None" in phase_case_ids
            or len(phase_case_id_list) != len(phase_case_ids)
        ):
            raise ValueError(f"phase {phase} has invalid or duplicate case summaries")
        phase_predictions = [
            dict(row)
            for row in json.loads(prediction_path.read_text(encoding="utf-8"))
        ]
        prediction_case_ids = {
            str(row.get("classifier_id", row.get("case_id")))
            for row in phase_predictions
        }
        if prediction_case_ids != phase_case_ids:
            raise ValueError(
                f"phase {phase} summary/prediction case sets differ: "
                f"summary={sorted(phase_case_ids)}, prediction={sorted(prediction_case_ids)}"
            )
        roster_by_case: dict[str, set[tuple[Any, ...]]] = {}
        for row in phase_predictions:
            case_id = str(row.get("classifier_id", row.get("case_id")))
            if "split_seed" not in row:
                raise ValueError(
                    f"phase {phase} prediction evidence lacks persisted split_seed"
                )
            participant_id = str(row.get("participant_id", "")).strip()
            repeat = int(row.get("repeat", -1))
            fold = int(row.get("fold", -1))
            split_seed = int(row["split_seed"])
            label = int(row.get("true_label", row.get("label", -1)))
            if (
                not participant_id
                or participant_id == "None"
                or repeat < 0
                or fold < 0
                or split_seed < 0
                or label not in (0, 1, 2)
            ):
                raise ValueError(
                    f"phase {phase} prediction evidence has an incomplete roster row"
                )
            roster_key = (
                participant_id,
                repeat,
                fold,
                split_seed,
                label,
            )
            if roster_key in roster_by_case.setdefault(case_id, set()):
                raise ValueError(
                    f"phase {phase} has duplicate participant-repeat prediction for {case_id}: {roster_key}"
                )
            roster_by_case[case_id].add(roster_key)
        roster_signatures = {frozenset(value) for value in roster_by_case.values()}
        if len(roster_signatures) != 1:
            raise ValueError(
                f"phase {phase} candidates do not share one participant/fold/split/label roster"
            )
        phase_roster = next(iter(roster_signatures))
        if expected_full_resource_roster is None:
            expected_full_resource_roster = phase_roster
        elif phase_roster != expected_full_resource_roster:
            raise ValueError(
                "full-resource promotion/completion phases do not share one "
                "participant/fold/split/label roster"
            )
        overlap = seen_cases.intersection(phase_case_ids)
        if overlap:
            raise ValueError(
                f"full-resource hyperparameter evidence overlaps cases: {sorted(overlap)}"
            )
        seen_cases.update(phase_case_ids)
        summaries.extend({**row, "evidence_phase": phase} for row in phase_summaries)
        predictions.extend(
            {**row, "evidence_phase": phase} for row in phase_predictions
        )
    return summaries, predictions, selected_phases


def _nested_phase_frozen_membership(
    selected_phases: Sequence[tuple[str, Path]],
) -> tuple[dict[tuple[str, int], tuple[int, int]] | None, str | None]:
    """Verify one authoritative frozen split roster across full-resource phases.

    Hyperparameter root reports combine ordinary nested studies.  Their OOF
    rows may agree with one another while all being detached from the declared
    split registry, so agreement of prediction rows alone is insufficient.
    Only PASS ordinary reproducibility audits and their persisted split tables
    can authorize numeric root-level pairwise comparisons.
    """

    if not selected_phases:
        return None, "frozen_split_registry_no_full_resource_phase"
    expected: dict[tuple[str, int], tuple[int, int]] | None = None
    for phase, raw_directory in selected_phases:
        directory = Path(raw_directory)
        summary_path = directory / "tables" / "reproducibility_summary.json"
        splits_path = directory / "tables" / "reproducibility_splits.json"
        if not summary_path.is_file() or not splits_path.is_file():
            return None, f"frozen_split_registry_not_verifiable__{phase}"
        try:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            split_payload = json.loads(splits_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            return None, f"frozen_split_registry_not_verifiable__{phase}"
        summary_rows = (
            summary_payload
            if isinstance(summary_payload, list)
            else [summary_payload]
            if isinstance(summary_payload, Mapping)
            else []
        )
        split_rows = split_payload if isinstance(split_payload, list) else []
        if (
            len(summary_rows) != 1
            or not isinstance(summary_rows[0], Mapping)
            or str(summary_rows[0].get("audit_status")) != "PASS"
            or not split_rows
        ):
            return None, f"frozen_split_registry_not_verifiable__{phase}"
        membership: dict[tuple[str, int], tuple[int, int]] = {}
        try:
            for raw_row in split_rows:
                if not isinstance(raw_row, Mapping):
                    raise TypeError("split row is not a mapping")
                if str(raw_row.get("audit_status")) != "PASS":
                    raise ValueError("split row did not pass reproducibility audit")
                repeat = int(raw_row["repeat"])
                fold = int(raw_row["fold"])
                split_seed = int(raw_row["split_seed"])
                participant_ids = raw_row["oof_participant_ids"]
                if (
                    repeat < 0
                    or fold < 0
                    or split_seed < 0
                    or not isinstance(participant_ids, list)
                    or not participant_ids
                    or int(raw_row.get("oof_participant_count", -1))
                    != len(participant_ids)
                    or int(raw_row.get("train_oof_overlap_count", -1)) != 0
                ):
                    raise ValueError("split row has an incomplete roster contract")
                for participant_id in participant_ids:
                    participant = str(participant_id).strip()
                    if not participant:
                        raise ValueError("empty participant ID in split roster")
                    key = (participant, repeat)
                    assignment = (fold, split_seed)
                    previous = membership.setdefault(key, assignment)
                    if previous != assignment:
                        raise ValueError(
                            "participant has conflicting frozen fold assignment"
                        )
        except (KeyError, TypeError, ValueError):
            return None, f"frozen_split_registry_not_verifiable__{phase}"
        if not membership:
            return None, f"frozen_split_registry_not_verifiable__{phase}"
        if expected is None:
            expected = membership
        elif membership != expected:
            return None, "frozen_split_registry_cross_phase_roster_mismatch"
    return expected, None


def _fail_closed_root_pairwise_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    reason: str,
) -> list[dict[str, Any]]:
    """Preserve the declared root comparison schema but remove numeric results."""

    unavailable = f"N/A_{reason}"
    output: list[dict[str, Any]] = []
    for raw_row in rows:
        row = dict(raw_row)
        for field in tuple(row):
            if (
                field in {
                    "candidate_minus_reference",
                    "participant_cluster_delta_ci95_low",
                    "participant_cluster_delta_ci95_high",
                    "bootstrap_valid_resamples",
                    "raw_two_sided_p_value",
                    "n_resamples",
                    "holm_adjusted_p_value",
                    "holm_rank",
                    "holm_family_size",
                    "reject_null_after_holm",
                    "participant_count",
                    "repeat_count",
                    "split_seed",
                    "matched_participant_count",
                    "matched_roster_sha256",
                }
                or field.startswith("reference_balanced_accuracy")
                or field.startswith("candidate_balanced_accuracy")
                or field.startswith("balanced_accuracy_delta")
                or field.startswith("reference_macro_f1")
                or field.startswith("candidate_macro_f1")
                or field.startswith("macro_f1_delta")
                or field.startswith("reference_macro_roc_auc_ovr")
                or field.startswith("candidate_macro_roc_auc_ovr")
                or field.startswith("macro_roc_auc_ovr_delta")
            ):
                row[field] = None
        row["comparison_contract_status"] = unavailable
        row["frozen_split_registry_status"] = unavailable
        if "p_value_applicability" in row:
            row["p_value_applicability"] = unavailable
        if "test_method" in row:
            row["test_method"] = unavailable
        if "interpretation" in row:
            row["interpretation"] = (
                "N/A: root-level numeric comparison was not computed because "
                "the nested frozen split registry could not be verified."
            )
        output.append(row)
    return output


def _copy_root_diagnostic_figures(
    output: Path,
    selected_phases: Sequence[tuple[str, Path]],
    *,
    required_names: Sequence[str],
    required_by_name: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    """Copy required diagnostics or persist an explicit fail-closed N/A."""

    from ..reporting.plots import FIGURE_TABLE_SOURCES

    rows: list[dict[str, Any]] = []
    target = output / "figures"
    target.mkdir(parents=True, exist_ok=True)
    target_tables = output / "tables"
    target_tables.mkdir(parents=True, exist_ok=True)
    phase_sources = list(selected_phases) or [("no_full_resource_phase", None)]
    for phase, raw_directory in phase_sources:
        directory = Path(raw_directory) if raw_directory is not None else None
        for name in required_names:
            source = (
                directory / "figures" / f"{name}.png"
                if directory is not None
                else None
            )
            destination = target / f"{phase}_{name}.png"
            if source is not None and source.is_file():
                shutil.copy2(source, destination)
                status = "generated_from_nested_phase"
                path = destination.relative_to(output).as_posix()
                reason = ""
            else:
                status = "N/A_fail_closed"
                marker = target / f"{phase}_{name}.NA.txt"
                reason = (
                    "no equal/full-resource phase is available for this "
                    "profile-required figure"
                    if directory is None
                    else "nested phase did not generate this profile-required figure"
                )
                marker.write_text(reason + "\n", encoding="utf-8")
                path = marker.relative_to(output).as_posix()
            source_tables = list(FIGURE_TABLE_SOURCES.get(name, ()))
            source_table_paths: list[str] = []
            missing_source_tables: list[str] = []
            for table_name in source_tables:
                copied = False
                for suffix in (".csv", ".json"):
                    table_source = (
                        directory / "tables" / f"{table_name}{suffix}"
                        if directory is not None
                        else None
                    )
                    if table_source is None or not table_source.is_file():
                        continue
                    table_destination = (
                        target_tables / f"{phase}_{table_name}{suffix}"
                    )
                    shutil.copy2(table_source, table_destination)
                    source_table_paths.append(
                        table_destination.relative_to(output).as_posix()
                    )
                    copied = True
                if not copied:
                    missing_source_tables.append(table_name)
            if (
                status == "generated_from_nested_phase"
                and missing_source_tables
            ):
                status = "generated_but_unpaired_fail_closed"
                reason = (
                    "figure exists but its declared source tables are missing: "
                    f"{missing_source_tables}"
                )
            rows.append(
                {
                    "phase": phase,
                    "figure": name,
                    "required_by_profiles": list(
                        (required_by_name or {}).get(name, ())
                    ),
                    "status": status,
                    "path": path,
                    "source_tables": source_tables,
                    "source_table_paths": source_table_paths,
                    "missing_source_tables": missing_source_tables,
                    "reason": reason,
                }
            )
    expected = {
        (phase, name)
        for phase, _directory in phase_sources
        for name in required_names
    }
    observed = {(str(row["phase"]), str(row["figure"])) for row in rows}
    if observed != expected:
        raise RuntimeError("profile-required root figure status is incomplete")
    return rows


def _copy_root_profile_tables(
    output: Path,
    selected_phases: Sequence[tuple[str, Path]],
    *,
    required_by_name: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    """Copy every profile-required nested table or register explicit N/A."""

    target = output / "tables"
    target.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    phase_sources = list(selected_phases) or [("no_full_resource_phase", None)]
    for phase, raw_directory in phase_sources:
        directory = Path(raw_directory) if raw_directory is not None else None
        for name in sorted(required_by_name):
            paths: list[str] = []
            missing_suffixes: list[str] = []
            for suffix in (".csv", ".json"):
                source = (
                    directory / "tables" / f"{name}{suffix}"
                    if directory is not None
                    else None
                )
                if source is None or not source.is_file():
                    missing_suffixes.append(suffix)
                    continue
                destination = target / f"{phase}_{name}{suffix}"
                shutil.copy2(source, destination)
                paths.append(destination.relative_to(output).as_posix())
            complete = not missing_suffixes
            rows.append(
                {
                    "phase": phase,
                    "table": name,
                    "required_by_profiles": list(required_by_name[name]),
                    "status": (
                        "copied_from_nested_phase"
                        if complete
                        else "N/A_fail_closed"
                    ),
                    "paths": paths,
                    "missing_suffixes": missing_suffixes,
                    "reason": (
                        ""
                        if complete
                        else (
                            "no equal/full-resource phase is available for this "
                            "profile-required table"
                            if directory is None
                            else (
                                "nested phase did not generate every required "
                                f"table serialization: {missing_suffixes}"
                            )
                        )
                    ),
                }
            )
    expected = {
        (phase, name)
        for phase, _directory in phase_sources
        for name in required_by_name
    }
    observed = {(str(row["phase"]), str(row["table"])) for row in rows}
    if observed != expected:
        raise RuntimeError("profile-required root table status is incomplete")
    return rows


def _write_root_report(
    output: Path,
    *,
    plan: Mapping[str, Any],
    phase_directories: Mapping[str, Path],
    ranking_tables: Mapping[str, Sequence[Mapping[str, Any]]],
    selected: Mapping[str, Any],
    inherited: Mapping[str, Any],
) -> None:
    from ..reporting.components import (
        TEST_COMPONENT_VIEW_SCHEMAS,
        markdown_test_component_table,
    )
    from ..reporting.conclusions import (
        DEFAULT_REPORTING_RANDOM_SEED,
        classification_comparison_rows,
        classification_comparison_table_views,
        classification_conclusion_rows,
        paired_inference_against_reference,
        paired_repeat_deltas_against_reference,
        write_result_interpretation,
    )
    from ..reporting.classification_diagnostics import (
        classification_per_class_metric_rows,
    )
    from ..data.schema import CANONICAL_CLASS_NAMES
    from ..reporting.profiles import (
        REPORTER_PROFILE_VIEW_SCHEMAS,
        markdown_reporter_profile_tables,
        reporter_profile_rows,
        required_figure_modules,
        write_reporter_methods,
    )

    metric = str(plan["resource"]["ranking_metric"])
    table_names: list[str] = []
    display_rankings: dict[str, list[dict[str, Any]]] = {}
    for name, rows in ranking_tables.items():
        display_rankings[name] = _write_table_pair(
            output,
            name=name,
            rows=rows,
            metric=metric,
            selection_role=_root_ranking_selection_role(plan, name),
        )
        table_names.append(name)
    participant_rankings = _participant_oof_rankings(phase_directories)
    display_participant_rankings: dict[str, list[dict[str, Any]]] = {}
    for name, rows in participant_rankings.items():
        display_participant_rankings[name] = _write_table_pair(
            output,
            name=name,
            rows=rows,
            metric="balanced_accuracy",
        )
        table_names.append(name)
    candidate_rows = _candidate_component_rows(
        plan,
        phase_directories,
        inherited=inherited,
    )
    profile_rows = reporter_profile_rows(candidate_rows)
    required_tables_by_name = {
        name: sorted(
            str(profile["profile_id"])
            for profile in profile_rows
            if name in profile.get("required_tables", ())
        )
        for name in sorted(
            {
                str(name)
                for profile in profile_rows
                for name in profile.get("required_tables", ())
            }
        )
    }
    required_figures_by_name = {
        name: sorted(
            str(profile["profile_id"])
            for profile in profile_rows
            if name in profile.get("required_figures", ())
        )
        for name in required_figure_modules(candidate_rows)
    }
    evidence_summaries, evidence_predictions, evidence_phases = (
        _final_classification_evidence(phase_directories)
    )
    frozen_membership, frozen_registry_failure_reason = (
        _nested_phase_frozen_membership(evidence_phases)
    )
    frozen_membership_guard: Mapping[tuple[str, int], tuple[int, int]] = (
        frozen_membership if frozen_membership is not None else {}
    )
    reporting_selection_seed = _persisted_reporting_selection_seed(
        evidence_phases,
        default=int(
            plan.get("search", {}).get(
                "selection_seed", DEFAULT_REPORTING_RANDOM_SEED
            )
        ),
    )
    declared_candidate_ids = tuple(
        str(row["case_id"]) for row in plan.get("candidates", ())
    )
    resource_contract = plan.get("resource", {})
    raw_full_repeats = resource_contract.get(
        "repeats", resource_contract.get("promotion_repeats", ())
    )
    expected_full_repeats = (
        tuple(int(value) for value in raw_full_repeats)
        if isinstance(raw_full_repeats, (list, tuple))
        else ()
    )
    exploratory_inference = paired_inference_against_reference(
        evidence_predictions,
        reference_case_id=str(selected["case_id"]),
        comparison_family=(
            f"{plan['study']['study_id']}__post_selection_selected_reference"
        ),
        inference_role="exploratory_post_selection_same_tuning_cv",
        candidate_case_ids=declared_candidate_ids,
        expected_repeats=expected_full_repeats or None,
        expected_membership=frozen_membership_guard,
        seed=reporting_selection_seed,
    )
    exploratory_repeat_deltas = paired_repeat_deltas_against_reference(
        evidence_predictions,
        reference_case_id=str(selected["case_id"]),
        comparison_family=(
            f"{plan['study']['study_id']}__post_selection_selected_reference"
        ),
        comparison_role="exploratory_post_selection_model_comparison",
        candidate_case_ids=declared_candidate_ids,
        expected_repeats=expected_full_repeats or None,
        expected_membership=frozen_membership_guard,
    )
    if frozen_registry_failure_reason is not None:
        exploratory_inference = _fail_closed_root_pairwise_rows(
            exploratory_inference,
            reason=frozen_registry_failure_reason,
        )
        exploratory_repeat_deltas = _fail_closed_root_pairwise_rows(
            exploratory_repeat_deltas,
            reason=frozen_registry_failure_reason,
        )
    classifier_per_class_rows = list(
        classification_per_class_metric_rows(
            evidence_predictions,
            class_names=CANONICAL_CLASS_NAMES,
        )
    )
    represented_classifier_ids = {
        str(row.get("classifier_id")) for row in classifier_per_class_rows
    }
    for candidate_id in sorted(
        set(declared_candidate_ids) - represented_classifier_ids
    ):
        for class_label, class_name in sorted(CANONICAL_CLASS_NAMES.items()):
            classifier_per_class_rows.append(
                {
                    "classifier_id": candidate_id,
                    "evaluation_id": "full_resource_participant_outer_oof",
                    "aggregation_level": "participant",
                    "class_label": class_label,
                    "class_name": class_name,
                    "true_positive": None,
                    "false_positive": None,
                    "true_negative": None,
                    "false_negative": None,
                    "support": None,
                    "predicted_support": None,
                    "observation_count": 0,
                    "input_observation_count": 0,
                    "retained_observation_count": 0,
                    "excluded_observation_count": 0,
                    "precision": None,
                    "sensitivity": None,
                    "recall": None,
                    "specificity": None,
                    "balanced_accuracy_ovr": None,
                    "f1": None,
                    "roc_auc_ovr": None,
                    "pr_auc_ovr": None,
                    "probability_metric_applicability": (
                        "N/A_no_full_resource_participant_oof_evidence"
                    ),
                    "result_applicability": (
                        "N/A_no_full_resource_participant_oof_evidence"
                    ),
                    "metric_scope": "one_vs_rest_not_computable",
                    "metric_source": "N/A_no_classifier_evidence",
                    "prediction_rule_source": "N/A_no_classifier_evidence",
                }
            )
    comprehensive_rows = classification_comparison_rows(
        evidence_summaries,
        paired_inference=exploratory_inference,
    )
    comparison_views = classification_comparison_table_views(
        comprehensive_rows,
        paired_inference=exploratory_inference,
    )
    conclusion_rows = classification_conclusion_rows(
        comprehensive_rows,
        selected_case_id=str(selected["case_id"]),
        selection_basis=(
            "persisted equal-weight fold-cell orchestration ranking; participant-OOF ranking is a descriptive sensitivity analysis"
        ),
        study_role="development_tuning_not_independent_final_test",
        planned_case_count=len(plan["candidates"]),
        incomplete_case_count=max(0, len(plan["candidates"]) - len(evidence_summaries)),
    )
    conclusion_rows.append(
        _design_scope_conclusion(
            plan,
            selected_case_id=str(selected["case_id"]),
        )
    )
    root_diagnostic_figures = _copy_root_diagnostic_figures(
        output,
        evidence_phases,
        required_names=required_figure_modules(candidate_rows),
        required_by_name=required_figures_by_name,
    )
    root_profile_tables = _copy_root_profile_tables(
        output,
        evidence_phases,
        required_by_name=required_tables_by_name,
    )
    copied_reporter_table_names = sorted(
        {
            Path(path).stem
            for rows, key in (
                (root_diagnostic_figures, "source_table_paths"),
                (root_profile_tables, "paths"),
            )
            for row in rows
            for path in row.get(key, ())
            if str(path).endswith(".csv")
        }
    )
    table_names.extend(copied_reporter_table_names)
    root_reporter_artifact_status = [
        {
            "artifact_kind": "table",
            "phase": row["phase"],
            "artifact_name": row["table"],
            "required_by_profiles": row["required_by_profiles"],
            "status": row["status"],
            "paths": row["paths"],
            "reason": row["reason"],
        }
        for row in root_profile_tables
    ] + [
        {
            "artifact_kind": "figure",
            "phase": row["phase"],
            "artifact_name": row["figure"],
            "required_by_profiles": row["required_by_profiles"],
            "status": row["status"],
            "paths": [row["path"]] if row["path"] else [],
            "reason": row["reason"],
        }
        for row in root_diagnostic_figures
    ]
    _write_csv(output / "tables" / "test_components.csv", candidate_rows)
    (output / "tables" / "test_components.json").write_text(
        json.dumps(candidate_rows, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    table_names.append("test_components")
    for table_name, table_rows in (
        ("reporter_profiles", profile_rows),
        ("comprehensive_model_comparison", comprehensive_rows),
        (
            "model_comparison_performance",
            comparison_views["ranking_performance"],
        ),
        (
            "model_comparison_uncertainty",
            comparison_views["uncertainty_ci"],
        ),
        (
            "model_comparison_inference",
            comparison_views["paired_inference"],
        ),
        (
            "model_comparison_robustness",
            comparison_views["robustness"],
        ),
        ("exploratory_selected_paired_inference", exploratory_inference),
        ("pairwise_repeat_metric_deltas", exploratory_repeat_deltas),
        ("classifier_per_class_results", classifier_per_class_rows),
        ("selection_conclusions", conclusion_rows),
        ("root_diagnostic_figures", root_diagnostic_figures),
        ("root_reporter_artifact_status", root_reporter_artifact_status),
    ):
        _write_csv(output / "tables" / f"{table_name}.csv", table_rows)
        (output / "tables" / f"{table_name}.json").write_text(
            json.dumps(
                table_rows,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        table_names.append(table_name)
    write_reporter_methods(output, candidate_rows)
    write_result_interpretation(
        output,
        comparison_rows=comprehensive_rows,
        conclusion_rows=conclusion_rows,
        paired_inference=exploratory_inference,
        split_classification_comparison=True,
    )
    reproducibility_rows = _reproducibility_rows(plan, phase_directories)
    _write_csv(output / "tables" / "reproducibility.csv", reproducibility_rows)
    (output / "tables" / "reproducibility.json").write_text(
        json.dumps(
            reproducibility_rows, indent=2, ensure_ascii=False, allow_nan=False
        ),
        encoding="utf-8",
    )
    table_names.append("reproducibility")
    paired_plot_tables = tuple(ranking_tables) + tuple(participant_rankings)
    pair_rows = [
        {
            "table": name,
            "table_path": f"tables/{name}.csv",
            "table_status": "available",
            "figure": name,
            "figure_status": (
                "generated"
                if (output / "figures" / f"{name}.png").is_file()
                else "N/A"
            ),
            "figure_path": (
                f"figures/{name}.png"
                if (output / "figures" / f"{name}.png").is_file()
                else f"figures/{name}.NA.txt"
            ),
            "reason": "",
        }
        for name in paired_plot_tables
    ]
    pair_rows.extend(
        {
            "table": "; ".join(row.get("source_tables", ())),
            "table_path": "; ".join(row.get("source_table_paths", ())),
            "table_status": (
                "copied_from_nested_phase"
                if row.get("source_table_paths")
                and not row.get("missing_source_tables")
                else "partial_or_N/A"
            ),
            "figure": f"{row['phase']}_{row['figure']}",
            "figure_status": row["status"],
            "figure_path": row["path"],
            "reason": row["reason"],
        }
        for row in root_diagnostic_figures
    )
    _write_csv(output / "tables" / "table_figure_pairs.csv", pair_rows)
    table_names.append("table_figure_pairs")
    from ..reporting.tabular import write_table_column_definitions

    write_table_column_definitions(
        output / "tables",
        csv_directory=output / "tables",
    )
    table_names.append("table_column_definitions")
    _write_workbook(output, table_names)
    selected_case = str(selected["case_id"])
    selection_evidence = str(
        selected.get(
            "selection_evidence",
            (
                "successive_halving_promoted_full_cv"
                if plan["study"]["study_type"] == "successive_halving"
                else "declared_full_cv_equal_weight_fold_cell_ranking"
            ),
        )
    )
    phase_lines = [
        f"- `{name}`: [{path.name}]({path.relative_to(output).as_posix()}/STUDY_SUMMARY.md)"
        for name, path in phase_directories.items()
    ]
    ranking_sections: list[str] = []
    for name, rows in display_rankings.items():
        ranking_sections.extend(
            (
                f"### {name}",
                "",
                _markdown_narrow_views(
                    rows,
                    identity_fields=("rank", "case_id"),
                    semantic_groups=_ranking_view_groups(rows),
                    heading_level=4,
                ),
                "",
            )
        )
    participant_sections: list[str] = []
    for name, rows in display_participant_rankings.items():
        participant_sections.extend(
            (
                f"### {name}",
                "",
                _markdown_narrow_views(
                    rows,
                    identity_fields=("rank", "case_id"),
                    semantic_groups=_ranking_view_groups(rows),
                    heading_level=4,
                ),
                "",
            )
        )
    final_participant_name = next(
        (
            name
            for name in (
                "all_candidates_full_cv_participant_oof_ranking",
                "promotion_participant_oof_ranking",
                "full_cv_participant_oof_ranking",
            )
            if name in participant_rankings and participant_rankings[name]
        ),
        None,
    )
    if final_participant_name is None and participant_rankings:
        final_participant_name = next(reversed(participant_rankings))
    participant_selected_case = (
        str(participant_rankings[final_participant_name][0]["case_id"])
        if final_participant_name is not None
        else "N/A"
    )
    selection_agreement = (
        participant_selected_case == selected_case
        if final_participant_name is not None
        else "N/A"
    )
    available_inference_row = next(
        (
            row
            for row in exploratory_inference
            if row.get("repeat_count") is not None
            and row.get("bootstrap_resamples") is not None
            and row.get("bootstrap_seed") is not None
        ),
        None,
    )
    paired_inference_method = (
        f"Whole participant clusters ({int(available_inference_row['repeat_count'])} "
        f"repeats per sampled participant) use "
        f"{int(available_inference_row['bootstrap_resamples']):,} shared-draw "
        "paired bootstrap resamples for BA, Macro-F1, and Macro ROC-AUC "
        f"CIs (seed={int(available_inference_row['bootstrap_seed'])}). "
        "BA and Macro-F1 additionally use participant-cluster permutations "
        "with metric-wise Holm correction."
        if available_inference_row is not None
        else (
            "Paired participant-cluster inference is explicit N/A because no "
            "selected-reference pair had a complete compatible participant, "
            "repeat, fold, split, label, and probability roster."
        )
    )
    component_markdown = markdown_test_component_table(candidate_rows)
    summary = "\n".join([
        f"# {plan['study']['study_id']}",
        "",
        str(plan["study"]["purpose"]),
        "",
        "## Selection outcome",
        "",
        f"Selected tuning configuration: `{selected_case}`.",
        "",
        f"Selection evidence: `{selection_evidence}`.",
        "",
        "This is development/tuning evidence, not a final or independent test. "
        "The orchestration ranking is the equal-weight mean of declared fold-cell "
        "metrics. Participant-OOF repeat summaries are reported separately as a "
        "descriptive sensitivity view and do not rewrite the completed selection.",
        "",
        f"Selection agreement check: participant-OOF descriptive top is "
        f"`{participant_selected_case}`; agreement with the orchestration winner "
        f"is `{selection_agreement}`.",
        "",
        "## Phase reports",
        "",
        *phase_lines,
        "",
        "## Paired tables and plots",
        "",
        *[
            f"- [{name}.csv](tables/{name}.csv) · "
            f"[plot](figures/{name}.png)"
            for name in ranking_tables
        ],
        *[
            f"- [{name}.csv](tables/{name}.csv) · "
            f"[plot](figures/{name}.png)"
            for name in participant_rankings
        ],
        "",
        *ranking_sections,
        "## Participant-OOF descriptive sensitivity rankings",
        "",
        "These tables recompute participant-level OOF metrics within repeat, "
        "then report the equal-weight repeat mean. They explain why the full-CV "
        "BA value differs from the fold-cell selection mean.",
        "",
        *participant_sections,
        "## Test models, modules, inputs, and fixed parameters",
        "",
        component_markdown,
        "",
        "## Model/module-owned reporter methods and literature",
        "",
        "The shared core registry resolved the report contract from the "
        "persisted `InceptionTimeFull` module identity. See "
        "[REPORT_METHODS.md](REPORT_METHODS.md) and "
        "[reporter_profiles.csv](tables/reporter_profiles.csv). Reporter profiles "
        "change presentation only.",
        "",
        markdown_reporter_profile_tables(profile_rows),
        "",
        "## Reporter required-artifact status",
        "",
        "Every profile-required table and figure is copied/generated or recorded "
        "as an explicit fail-closed N/A; no requirement is silently omitted. See "
        "[root_reporter_artifact_status.csv]"
        "(tables/root_reporter_artifact_status.csv).",
        "",
        _markdown_narrow_views(
            root_reporter_artifact_status,
            identity_fields=("artifact_name",),
            semantic_groups=(
                (
                    "Status and requirement",
                    (
                        "artifact_kind",
                        "phase",
                        "required_by_profiles",
                        "status",
                        "reason",
                    ),
                ),
                ("Artifact paths", ("artifact_kind", "phase", "paths")),
            ),
        ),
        "",
        "## Comprehensive participant-OOF comparison",
        "",
        "These narrow tables use equal-repeat participant-OOF endpoints. The "
        "historical wide compatibility evidence remains available in "
        "[comprehensive_model_comparison.json]"
        "(tables/comprehensive_model_comparison.json) and its CSV companion, "
        "but is not rendered as one expanding report table.",
        "Participant-cluster CIs resample participant IDs within true-class strata "
        "and carry every repeat OOF row; paired CIs apply the same draw to both "
        "classifiers before taking candidate minus reference.",
        "",
        "### Ranking and performance",
        "",
        _markdown_table(comparison_views["ranking_performance"]),
        "",
        "### Uncertainty and 95% confidence intervals",
        "",
        _markdown_table(comparison_views["uncertainty_ci"]),
        "",
        "### Robustness",
        "",
        _markdown_table(comparison_views["robustness"]),
        "",
        "## Per-class classifier results",
        "",
        "Every full-resource classifier is reported for every frailty class "
        "from its persisted participant OOF probabilities and decision labels.",
        "",
        _markdown_narrow_views(
            classifier_per_class_rows,
            identity_fields=("classifier_id", "class_name"),
            semantic_groups=_PER_CLASS_VIEW_GROUPS,
        ),
        "",
        "## Exploratory paired P values against the persisted selection",
        "",
        "Because the reference was chosen on this same tuning CV, these are "
        "post-selection exploratory P values—not confirmatory evidence. "
        + paired_inference_method
        + " "
        "P values are null-hypothesis tail probabilities, not the probability "
        "that a candidate is best.",
        "",
        _markdown_table(comparison_views["paired_inference"]),
        "",
        "The lossless inference contract, raw P values, resampling counts, seeds, "
        "family identifiers, and N/A reasons remain in "
        "[exploratory_selected_paired_inference.json]"
        "(tables/exploratory_selected_paired_inference.json).",
        "",
        "## Matched per-repeat differences against the persisted selection",
        "",
        "Each row is candidate minus selected reference on the exact matched "
        "participant/fold/split roster. These post-selection comparisons are "
        "exploratory, not causal ablations.",
        "",
        _markdown_narrow_views(
            exploratory_repeat_deltas,
            identity_fields=("comparison_id", "repeat"),
            semantic_groups=_REPEAT_DELTA_VIEW_GROUPS,
        ),
        "",
        "## Conclusions and selection confidence",
        "",
        _markdown_table(conclusion_rows),
        "",
        "The standalone detailed interpretation is "
        "[RESULT_INTERPRETATION.md](RESULT_INTERPRETATION.md).",
        "",
        "## Core reporter diagnostic figures",
        "",
        *[
            f"![{row['phase']} {row['figure']}]({row['path']})"
            for row in root_diagnostic_figures
            if row["status"] == "generated_from_nested_phase"
        ],
        "",
        "## Seeds and data splits",
        "",
        _markdown_narrow_views(
            reproducibility_rows,
            identity_fields=("phase",),
            semantic_groups=(
                (
                    "Execution indices and seeds",
                    (
                        "fixed_epochs",
                        "repeat_indices",
                        "fold_indices",
                        "split_seeds",
                        "training_seeds",
                        "training_seed_policy",
                    ),
                ),
                ("Split and decision scope", ("split_group", "selection_scope")),
            ),
        ),
        "",
        "Fold-cell orchestration tables use `mean ± population SD` over declared "
        "cells. Participant-OOF comparison tables use five-repeat sample SD and "
        "separately labeled repeat t-CI/participant-cluster bootstrap CI. Raw "
        "numeric columns remain in JSON; each CSV table occupies one workbook sheet.",
        "",
        "## Reproducibility",
        "",
        "The nested phase reports contain split seeds, training seeds, data splits, "
        "model/module names, actual input descriptions, and resolved fixed parameters.",
        "",
    ])
    (output / "STUDY_SUMMARY.md").write_text(summary, encoding="utf-8")
    (output / "TEST_COMPONENTS.md").write_text(
        "# Test models, modules, inputs, and fixed parameters\n\n"
        + component_markdown
        + "\n",
        encoding="utf-8",
    )
    html_ranking = "\n".join(
        f"<h2>{html_escape(name)}</h2>"
        + _html_narrow_views(
            rows,
            identity_fields=("rank", "case_id"),
            semantic_groups=_ranking_view_groups(rows),
        )
        for name, rows in display_rankings.items()
    )
    html_participant_ranking = "\n".join(
        f"<h2>{html_escape(name)}</h2>"
        + _html_narrow_views(
            rows,
            identity_fields=("rank", "case_id"),
            semantic_groups=_ranking_view_groups(rows),
        )
        for name, rows in display_participant_rankings.items()
    )
    html = "\n".join((
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Hyperparameter study</title>",
        "<style>body{font-family:sans-serif;max-width:1600px;margin:auto;padding:1rem}"
        "table{border-collapse:collapse;margin-bottom:1.5rem}th,td{border:1px solid #bbb;"
        "padding:.3rem;vertical-align:top}th{background:#eee}</style></head><body>",
        f"<h1>{html_escape(str(plan['study']['study_id']))}</h1>",
        f"<p>Selected tuning configuration: <code>{html_escape(selected_case)}</code>.</p>",
        f"<p>Selection evidence: <code>{html_escape(selection_evidence)}</code>.</p>",
        "<p>This is development/tuning evidence, not a final test.</p>",
        "<p>Orchestration selection uses equal-weight fold-cell means. "
        "Participant-OOF repeat summaries are a descriptive sensitivity view.</p>",
        "<p><a href='STUDY_SUMMARY.md'>Complete Markdown report</a></p>",
        "<p><a href='tables/report_tables.xlsx'>Excel workbook</a></p>",
        html_ranking,
        html_participant_ranking,
        "<h2>Test models, modules, inputs, and fixed parameters</h2>",
        _html_fixed_schema_views(candidate_rows, TEST_COMPONENT_VIEW_SCHEMAS),
        "<h2>Model/module-owned reporter methods and literature</h2>",
        "<p>See <a href='REPORT_METHODS.md'>REPORT_METHODS.md</a>. Profiles are "
        "presentation-only and resolved from persisted module identities.</p>",
        _html_fixed_schema_views(profile_rows, REPORTER_PROFILE_VIEW_SCHEMAS),
        "<h2>Reporter required-artifact status</h2>",
        "<p>Every required table and figure is available or explicitly "
        "N/A/fail-closed.</p>",
        _html_narrow_views(
            root_reporter_artifact_status,
            identity_fields=("artifact_name",),
            semantic_groups=(
                (
                    "Status and requirement",
                    (
                        "artifact_kind",
                        "phase",
                        "required_by_profiles",
                        "status",
                        "reason",
                    ),
                ),
                ("Artifact paths", ("artifact_kind", "phase", "paths")),
            ),
        ),
        "<h2>Comprehensive participant-OOF comparison</h2>",
        "<p>The historical wide compatibility evidence remains in "
        "<a href='tables/comprehensive_model_comparison.json'>lossless JSON</a> "
        "and its CSV companion; the report renders separate narrow views.</p>",
        "<h3>Ranking and performance</h3>",
        _html_table(comparison_views["ranking_performance"]),
        "<h3>Uncertainty and 95% confidence intervals</h3>",
        _html_table(comparison_views["uncertainty_ci"]),
        "<h3>Robustness</h3>",
        _html_table(comparison_views["robustness"]),
        "<h2>Per-class classifier results</h2>",
        _html_narrow_views(
            classifier_per_class_rows,
            identity_fields=("classifier_id", "class_name"),
            semantic_groups=_PER_CLASS_VIEW_GROUPS,
        ),
        "<h2>Exploratory paired P values against the persisted selection</h2>",
        "<p>Post-selection exploratory only: "
        + html_escape(paired_inference_method)
        + " P is not posterior confidence.</p>",
        _html_table(comparison_views["paired_inference"]),
        "<p>Lossless inference provenance remains in "
        "<a href='tables/exploratory_selected_paired_inference.json'>the raw "
        "inference JSON</a>.</p>",
        "<h2>Matched per-repeat differences against the persisted selection</h2>",
        _html_narrow_views(
            exploratory_repeat_deltas,
            identity_fields=("comparison_id", "repeat"),
            semantic_groups=_REPEAT_DELTA_VIEW_GROUPS,
        ),
        "<h2>Conclusions and selection confidence</h2>",
        _html_table(conclusion_rows),
        "<p><a href='RESULT_INTERPRETATION.md'>Detailed interpretation</a></p>",
        "<h2>Core reporter diagnostic figures</h2>",
        "".join(
            f"<figure><img style='max-width:100%' src='{html_escape(row['path'])}' "
            f"alt='{html_escape(row['figure'])}'><figcaption>"
            f"{html_escape(row['phase'] + ' ' + row['figure'])}</figcaption></figure>"
            for row in root_diagnostic_figures
            if row["status"] == "generated_from_nested_phase"
        ),
        "<h2>Seeds and data splits</h2>",
        _html_narrow_views(
            reproducibility_rows,
            identity_fields=("phase",),
            semantic_groups=(
                (
                    "Execution indices and seeds",
                    (
                        "fixed_epochs",
                        "repeat_indices",
                        "fold_indices",
                        "split_seeds",
                        "training_seeds",
                        "training_seed_policy",
                    ),
                ),
                ("Split and decision scope", ("split_group", "selection_scope")),
            ),
        ),
        "</body></html>",
    ))
    (output / "STUDY_SUMMARY.html").write_text(html, encoding="utf-8")
    index_rows = [
        {
            "path": path.relative_to(output).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and "result_backup" not in path.parts
        and path != output / "outputs_index.json"
    ]
    (output / "outputs_index.json").write_text(
        json.dumps(index_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _backup_report_outputs(output: Path) -> None:
    """Create a self-indexed root-report snapshot without duplicating training evidence."""

    backup = output / "result_backup"
    backup.mkdir(exist_ok=True)
    for name in (
        "study_plan.yaml", "study_manifest.json", "selected_configuration.json",
        "precompletion_study_manifest.json",
        "precompletion_selected_configuration.json",
        "STUDY_SUMMARY.md", "STUDY_SUMMARY.html", "TEST_COMPONENTS.md",
        "REPORT_METHODS.md", "RESULT_INTERPRETATION.md",
    ):
        source = output / name
        if source.is_file():
            shutil.copy2(source, backup / name)
    for directory in ("tables", "figures"):
        source = output / directory
        target = backup / directory
        if target.exists():
            shutil.rmtree(target)
        if source.is_dir():
            shutil.copytree(source, target)
    (backup / "BACKUP_SCOPE.md").write_text(
        "# Backup scope\n\n"
        "This directory is a self-indexed snapshot of the root report, its "
        "tables, copied table sources, and figures. It intentionally excludes "
        "nested phase reports, training outputs, checkpoints, raw predictions, "
        "and datasets. Links from the copied root report to nested phase paths "
        "therefore resolve only from the original study directory; those source "
        "artifacts remain indexed by the root `outputs_index.json`.\n",
        encoding="utf-8",
    )
    backup_index = [
        {
            "path": path.relative_to(backup).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(backup.rglob("*"))
        if path.is_file() and path != backup / "outputs_index.json"
    ]
    (backup / "outputs_index.json").write_text(
        json.dumps(backup_index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_hyperparameter_study(
    plan_path: str | Path,
    *,
    pipeline_root: str | Path,
    upstream_study: str | Path | None = None,
    output_root: str | Path | None = None,
    device: str | None = None,
    jobs: int | None = None,
    progress_sink: ProgressSink | None = None,
) -> Path:
    """Run one plan, select deterministically, and archive every phase."""

    plan = load_hyperparameter_plan(plan_path)
    root = Path(pipeline_root).resolve()
    raw_output = Path(output_root or plan["output"]["root"])
    output_parent = raw_output if raw_output.is_absolute() else root / raw_output
    output = _new_output(output_parent.resolve(), str(plan["study"]["study_id"]))
    with (output / "study_plan.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            {key: value for key, value in plan.items() if key != "plan_path"},
            stream,
            sort_keys=False,
            allow_unicode=True,
        )
    sink = progress_sink or NullProgressSink()
    resolved_device = str(device or plan["execution"]["device"])
    if not resolved_device.startswith("cuda"):
        raise ValueError("hyperparameter training must use CUDA")
    resolved_jobs = _positive_int(
        jobs if jobs is not None else plan["execution"]["jobs"], "jobs"
    )
    study_type = str(plan["study"]["study_type"])
    inherited: dict[str, Any] = {}
    upstream_payload: dict[str, Any] | None = None
    if study_type != "successive_halving":
        upstream_payload = _load_upstream(upstream_study, expected_type=study_type)
        upstream_config = upstream_payload["resolved_config"]
        paths = (
            ("training.batch_size", "training.learning_rate")
            if study_type == "dependent_regularization_grid"
            else (
                "training.batch_size", "training.learning_rate",
                "training.weight_decay", "training.label_smoothing", "model.dropout",
            )
        )
        inherited = {path: _dotted_get(upstream_config, path) for path in paths}
    phase_directories: dict[str, Path] = {}
    ranking_tables: dict[str, list[dict[str, Any]]] = {}
    if study_type == "successive_halving":
        resource = plan["resource"]
        _, screen_dir, screen_ranking = _run_phase(
            plan,
            output=output,
            phase_id="screen_5epoch_reduced_cv",
            candidates=plan["candidates"],
            repeats=resource["screen_repeats"],
            folds=resource["screen_folds"],
            epochs=int(resource["screen_epochs"]),
            inherited={},
            pipeline_root=root,
            device=resolved_device,
            jobs=resolved_jobs,
            progress_sink=sink,
        )
        phase_directories["screen"] = screen_dir
        ranking_tables["screen_ranking"] = screen_ranking
        promoted_ids = {
            row["case_id"]
            for row in screen_ranking[: int(resource["promote_count"])]
        }
        promoted = [
            row for row in plan["candidates"] if row["case_id"] in promoted_ids
        ]
        _, full_dir, full_ranking = _run_phase(
            plan,
            output=output,
            phase_id="promoted_full_cv",
            candidates=promoted,
            repeats=resource["promotion_repeats"],
            folds=resource["promotion_folds"],
            epochs=int(resource["promotion_epochs"]),
            inherited={},
            pipeline_root=root,
            device=resolved_device,
            jobs=resolved_jobs,
            progress_sink=sink,
        )
        phase_directories["promotion"] = full_dir
        ranking_tables["promotion_ranking"] = full_ranking
        selected_row = full_ranking[0]
        selected_path, selected_config = _resolved_config(
            full_dir, str(selected_row["case_id"])
        )
    else:
        resource = plan["resource"]
        _, full_dir, full_ranking = _run_phase(
            plan,
            output=output,
            phase_id="full_cv",
            candidates=plan["candidates"],
            repeats=resource["repeats"],
            folds=resource["folds"],
            epochs=int(resource["epochs"]),
            inherited=inherited,
            pipeline_root=root,
            device=resolved_device,
            jobs=resolved_jobs,
            progress_sink=sink,
        )
        phase_directories["full_cv"] = full_dir
        ranking_tables["full_cv_ranking"] = full_ranking
        selected_row = full_ranking[0]
        selected_path, selected_config = _resolved_config(
            full_dir, str(selected_row["case_id"])
        )
    selected = {
        "schema_version": "ppg_frailty.tuning_selection.v1",
        "study_id": plan["study"]["study_id"],
        "study_type": study_type,
        "selection_scope": "development_tuning_only_not_final_test",
        "case_id": selected_row["case_id"],
        "ranking": selected_row,
        "resolved_config_source": selected_path.relative_to(output).as_posix(),
        "resolved_config": selected_config,
        "upstream_selection": upstream_payload,
    }
    (output / "selected_configuration.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "ppg_frailty.hyperparameter_study_manifest.v1",
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "study": plan["study"],
        "plan_sha256": hashlib.sha256(
            Path(plan["plan_path"]).read_bytes()
        ).hexdigest(),
        "device": resolved_device,
        "jobs": resolved_jobs,
        "phase_directories": {
            key: value.relative_to(output).as_posix()
            for key, value in phase_directories.items()
        },
        "selected_case_id": selected_row["case_id"],
        "ranking_tables": list(ranking_tables),
    }
    (output / "study_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    _write_root_report(
        output,
        plan=plan,
        phase_directories=phase_directories,
        ranking_tables=ranking_tables,
        selected=selected,
        inherited=inherited,
    )
    _backup_report_outputs(output)
    return output


def inspect_successive_halving_completion(study_dir: str | Path) -> dict[str, Any]:
    """Describe the unpromoted full-CV work without starting training."""

    root = Path(study_dir).resolve()
    manifest = _mapping(
        json.loads((root / "study_manifest.json").read_text(encoding="utf-8")),
        "study_manifest",
    )
    plan = load_hyperparameter_plan(root / "study_plan.yaml")
    if str(plan["study"]["study_type"]) != "successive_halving":
        raise ValueError("completion is only valid for successive-halving studies")
    if "completion" in manifest.get("phase_directories", {}):
        remaining: list[dict[str, Any]] = []
        status = "already_complete"
    else:
        promotion_path = root / "tables" / "promotion_ranking.json"
        promoted = list(json.loads(promotion_path.read_text(encoding="utf-8")))
        remaining = _completion_candidates(plan, promoted)
        status = "ready"
    resource = plan["resource"]
    return {
        "status": status,
        "study_dir": str(root),
        "candidate_ids": [str(row["case_id"]) for row in remaining],
        "candidate_count": len(remaining),
        "repeats": list(resource["promotion_repeats"]),
        "folds": list(resource["promotion_folds"]),
        "fixed_epochs": int(resource["promotion_epochs"]),
        "fold_cell_count": (
            len(remaining)
            * len(resource["promotion_repeats"])
            * len(resource["promotion_folds"])
        ),
        "reuses_existing_promoted_full_cv": True,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.completion.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def complete_successive_halving_study(
    study_dir: str | Path,
    *,
    pipeline_root: str | Path,
    device: str | None = None,
    jobs: int | None = None,
    progress_sink: ProgressSink | None = None,
) -> Path:
    """Full-CV only the unpromoted cases and then rank the complete grid."""

    output = Path(study_dir).resolve()
    manifest_path = output / "study_manifest.json"
    selected_path = output / "selected_configuration.json"
    original_manifest_text = manifest_path.read_text(encoding="utf-8")
    original_selected_text = selected_path.read_text(encoding="utf-8")
    manifest = _mapping(json.loads(original_manifest_text), "study_manifest")
    original_selected = _mapping(
        json.loads(original_selected_text), "selected_configuration"
    )
    if str(manifest.get("status")) != "passed":
        raise RuntimeError("the source successive-halving study is not passed")
    if "completion" in manifest.get("phase_directories", {}):
        raise RuntimeError("the successive-halving study is already complete")
    plan = load_hyperparameter_plan(output / "study_plan.yaml")
    if str(plan["study"]["study_type"]) != "successive_halving":
        raise ValueError("completion is only valid for successive-halving studies")
    existing_rankings = {
        str(name): list(
            json.loads(
                (output / "tables" / f"{name}.json").read_text(encoding="utf-8")
            )
        )
        for name in manifest["ranking_tables"]
    }
    promotion_ranking = existing_rankings.get("promotion_ranking")
    if not promotion_ranking:
        raise RuntimeError("the source study lacks a promotion ranking")
    remaining = _completion_candidates(plan, promotion_ranking)
    if not remaining:
        raise RuntimeError("no unpromoted candidates remain")
    resolved_device = str(
        device or manifest.get("device") or plan["execution"]["device"]
    )
    if not resolved_device.startswith("cuda"):
        raise ValueError("hyperparameter training must use CUDA")
    resolved_jobs = _positive_int(
        jobs if jobs is not None else manifest.get("jobs", plan["execution"]["jobs"]),
        "jobs",
    )
    resource = plan["resource"]
    sink = progress_sink or NullProgressSink()
    _, completion_dir, completion_ranking = _run_phase(
        plan,
        output=output,
        phase_id="nonpromoted_full_cv",
        candidates=remaining,
        repeats=resource["promotion_repeats"],
        folds=resource["promotion_folds"],
        epochs=int(resource["promotion_epochs"]),
        inherited={},
        pipeline_root=Path(pipeline_root).resolve(),
        device=resolved_device,
        jobs=resolved_jobs,
        progress_sink=sink,
    )
    for row in completion_ranking:
        row["metric_source"] = "equal_weight_fold_cell_mean_for_selection"
        row["selection_role"] = (
            "completion_subset_full_cv_evidence_not_standalone_selection"
        )
    combined_ranking = _merge_equal_resource_rankings(
        (promotion_ranking, completion_ranking),
        metric=str(resource["ranking_metric"]),
        tie_break=str(resource["tie_break_metric"]),
        expected_case_ids=[str(row["case_id"]) for row in plan["candidates"]],
    )
    selected_row = combined_ranking[0]
    completion_ids = {str(row["case_id"]) for row in remaining}
    phase_directories = {
        str(name): output / str(relative)
        for name, relative in manifest["phase_directories"].items()
    }
    phase_directories["completion"] = completion_dir
    selected_phase = (
        completion_dir
        if str(selected_row["case_id"]) in completion_ids
        else phase_directories["promotion"]
    )
    resolved_path, resolved_config = _resolved_config(
        selected_phase, str(selected_row["case_id"])
    )
    selected = {
        "schema_version": "ppg_frailty.tuning_selection.v1",
        "study_id": plan["study"]["study_id"],
        "study_type": "successive_halving",
        "selection_scope": "development_tuning_only_not_final_test",
        "case_id": selected_row["case_id"],
        "ranking": selected_row,
        "resolved_config_source": resolved_path.relative_to(output).as_posix(),
        "resolved_config": resolved_config,
        "upstream_selection": None,
        "precompletion_case_id": original_selected.get("case_id"),
        "selection_evidence": "complete_six_candidate_full_5x5_grid",
    }
    ranking_tables = {
        **existing_rankings,
        "nonpromoted_full_cv_ranking": completion_ranking,
        "all_candidates_full_cv_ranking": combined_ranking,
    }
    completed_manifest = {
        **manifest,
        "status": "passed",
        "device": resolved_device,
        "jobs": resolved_jobs,
        "phase_directories": {
            name: path.relative_to(output).as_posix()
            for name, path in phase_directories.items()
        },
        "selected_case_id": selected_row["case_id"],
        "ranking_tables": list(ranking_tables),
        "completion_created_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "successive_halving_completion": {
            "status": "passed",
            "reused_promoted_case_ids": [
                str(row["case_id"]) for row in promotion_ranking
            ],
            "newly_trained_case_ids": sorted(completion_ids),
            "full_cv_candidate_count": len(combined_ranking),
            "full_cv_fold_cell_count": sum(
                int(row["cell_count"]) for row in combined_ranking
            ),
            "selection_recomputed_after_complete_grid": True,
        },
    }
    (output / "precompletion_selected_configuration.json").write_text(
        original_selected_text, encoding="utf-8"
    )
    (output / "precompletion_study_manifest.json").write_text(
        original_manifest_text, encoding="utf-8"
    )
    _write_json_atomic(selected_path, selected)
    _write_json_atomic(manifest_path, completed_manifest)
    try:
        _write_root_report(
            output,
            plan=plan,
            phase_directories=phase_directories,
            ranking_tables=ranking_tables,
            selected=selected,
            inherited={},
        )
    except Exception:
        _write_json_atomic(selected_path, original_selected)
        _write_json_atomic(manifest_path, manifest)
        raise
    _backup_report_outputs(output)
    return output


def regenerate_hyperparameter_report(study_dir: str | Path) -> dict[str, Any]:
    """Regenerate every nested and root report without retraining."""

    root = Path(study_dir).resolve()
    manifest = _mapping(
        json.loads((root / "study_manifest.json").read_text(encoding="utf-8")),
        "study_manifest",
    )
    from ppg_frailty.reporting import generate_study_report

    outputs: dict[str, str] = {}
    phase_directories = {
        str(phase): root / str(relative)
        for phase, relative in manifest["phase_directories"].items()
    }
    for phase, directory in phase_directories.items():
        report = generate_study_report(directory)
        outputs[str(phase)] = str(report.summary_markdown)
    plan = load_hyperparameter_plan(root / "study_plan.yaml")
    selected = _mapping(
        json.loads((root / "selected_configuration.json").read_text(encoding="utf-8")),
        "selected_configuration",
    )
    ranking_tables = {
        str(name): list(
            json.loads((root / "tables" / f"{name}.json").read_text(encoding="utf-8"))
        )
        for name in manifest["ranking_tables"]
    }
    inherited: dict[str, Any] = {}
    upstream = selected.get("upstream_selection")
    if isinstance(upstream, Mapping) and isinstance(upstream.get("resolved_config"), Mapping):
        upstream_config = upstream["resolved_config"]
        paths = (
            ("training.batch_size", "training.learning_rate")
            if plan["study"]["study_type"] == "dependent_regularization_grid"
            else (
                "training.batch_size", "training.learning_rate",
                "training.weight_decay", "training.label_smoothing", "model.dropout",
            )
        )
        inherited = {path: _dotted_get(upstream_config, path) for path in paths}
    _write_root_report(
        root,
        plan=plan,
        phase_directories=phase_directories,
        ranking_tables=ranking_tables,
        selected=selected,
        inherited=inherited,
    )
    _backup_report_outputs(root)
    return {
        "status": "regenerated",
        "root_report": str(root / "STUDY_SUMMARY.md"),
        "phase_reports": outputs,
    }


__all__ = [
    "complete_successive_halving_study",
    "inspect_successive_halving_completion",
    "load_hyperparameter_plan",
    "regenerate_hyperparameter_report",
    "run_hyperparameter_study",
]
