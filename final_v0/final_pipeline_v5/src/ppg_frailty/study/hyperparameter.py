"""Auditable multi-resource tuning studies built from ordinary study phases."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Mapping, Sequence

import yaml

from .expand import parse_study_plan
from .progress import NullProgressSink, ProgressSink
from .runner import StudyRunner


_SCHEMA = "ppg_frailty.hyperparameter_study_plan.v1"
_STUDY_TYPES = {"successive_halving", "dependent_regularization_grid", "dependent_channel_ablation"}
_PLAN_FIELDS = {
    "schema_version", "study", "catalog", "base", "candidates", "resource", "execution", "output", "report", "search",
}
_STUDY_FIELDS = {"study_id", "study_type", "purpose", "flow_position", "decision_role", "thesis_sections"}
_BASE_FIELDS = {"catalog_entry", "output_group", "profile_id", "common_overrides"}
_CANDIDATE_FIELDS = {"case_id", "label", "overrides", "rationale"}
_EXECUTION_FIELDS = {
    "jobs", "device", "parallel_level", "continue_on_error", "allow_parallel_deep", "measure_operational_costs",
}
_REPORT_FIELDS = {
    "top_k", "write_html", "write_static_figures", "calibration_bins", "figure_modules", "compact_mean_sd",
    "write_excel_workbook", "classification_tsne_random_state", "classification_tsne_perplexity",
    "classification_tsne_max_samples", "classification_roc_macro_grid_points",
    "classification_score_histogram_bins",
}
_HALVING_RESOURCE_FIELDS = {
    "screen_epochs", "screen_repeats", "screen_folds", "promotion_epochs", "promotion_repeats",
    "promotion_folds", "promote_count", "ranking_metric", "tie_break_metric",
}
_FULL_RESOURCE_FIELDS = {"epochs", "repeats", "folds", "ranking_metric", "tie_break_metric"}
_REPORT_DEFAULTS = {
    "classification_tsne_random_state": 42,
    "classification_tsne_perplexity": 30.0,
    "classification_tsne_max_samples": 5000,
    "classification_roc_macro_grid_points": 201,
    "classification_score_histogram_bins": 40,
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
        raise ValueError(f"{label} key mismatch: missing={sorted(fields-set(result))}, "
                         f"unknown={sorted(set(result)-fields)}")
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
    loaded = _mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "plan")
    from ..reporting.conclusions import DEFAULT_REPORTING_RANDOM_SEED

    raw = _strict(
        {**loaded, "search": loaded.get("search", {"selection_seed": DEFAULT_REPORTING_RANDOM_SEED})},
        "plan",
        _PLAN_FIELDS,
    )
    if raw["schema_version"] != _SCHEMA:
        raise ValueError(f"schema_version must equal {_SCHEMA}")
    study = _strict(raw["study"], "study", _STUDY_FIELDS)
    if study["study_type"] not in _STUDY_TYPES:
        raise ValueError(f"unsupported study_type: {study['study_type']}")
    if not str(study["study_id"]).strip() or not str(study["purpose"]).strip():
        raise ValueError("study_id and purpose must be non-empty")
    if not isinstance(study["thesis_sections"], list):
        raise TypeError("study.thesis_sections must be a list")
    catalog = _strict(raw["catalog"], "catalog", {"path", "balance_line"})
    if catalog["balance_line"] not in {"line_a", "line_b"}:
        raise ValueError("catalog.balance_line must be line_a or line_b")
    base = _strict(raw["base"], "base", _BASE_FIELDS)
    base["common_overrides"] = _validate_overrides(base["common_overrides"], "base.common_overrides")
    candidates = raw["candidates"]
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("candidates must contain at least two cases")
    normalized_candidates: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for index, value in enumerate(candidates):
        candidate = _strict(value, f"candidates[{index}]", _CANDIDATE_FIELDS)
        case_id = str(candidate["case_id"])
        if not case_id or case_id in identifiers:
            raise ValueError("candidate case_id values must be non-empty and unique")
        identifiers.add(case_id)
        candidate["overrides"] = _validate_overrides(candidate["overrides"], f"candidates[{index}].overrides")
        normalized_candidates.append(candidate)
    execution = _strict(raw["execution"], "execution", _EXECUTION_FIELDS)
    _positive_int(execution["jobs"], "execution.jobs")
    if execution["parallel_level"] != "cases":
        raise ValueError("execution.parallel_level must be cases")
    if not str(execution["device"]).startswith("cuda"):
        raise ValueError("deep hyperparameter studies require a CUDA device")
    search = _strict(raw["search"], "search", {"selection_seed"})
    selection_seed = search["selection_seed"]
    if isinstance(selection_seed, bool) or not isinstance(selection_seed, int) or selection_seed < 0:
        raise ValueError("search.selection_seed must be a non-negative integer")
    search["selection_seed"] = int(selection_seed)
    report = _strict(
        {**_REPORT_DEFAULTS, **_mapping(raw["report"], "report")},
        "report",
        _REPORT_FIELDS,
    )
    resource = _mapping(raw["resource"], "resource")
    if study["study_type"] == "successive_halving":
        resource = _strict(resource, "resource", _HALVING_RESOURCE_FIELDS)
        for field in ("screen_epochs", "promotion_epochs"):
            _positive_int(resource[field], f"resource.{field}")
        for field in ("screen_repeats", "screen_folds", "promotion_repeats", "promotion_folds"):
            resource[field] = _indices(resource[field], f"resource.{field}")
        promote = _positive_int(resource["promote_count"], "resource.promote_count")
        if promote >= len(normalized_candidates):
            raise ValueError("promote_count must be smaller than candidate count")
    else:
        resource = _strict(resource, "resource", _FULL_RESOURCE_FIELDS)
        _positive_int(resource["epochs"], "resource.epochs")
        resource["repeats"] = _indices(resource["repeats"], "resource.repeats")
        resource["folds"] = _indices(resource["folds"], "resource.folds")
    output = _strict(raw["output"], "output", {"root"})
    raw.update(
        study=study, catalog=catalog, base=base, candidates=normalized_candidates,
        resource=resource, execution=execution, search=search, output=output, report=report,
    )
    raw["plan_path"] = str(source)
    return raw

def _safe_slug(value: str) -> str:
    result = "".join(character if character.isalnum() else "-" for character in value)
    return "-".join(part for part in result.lower().split("-") if part)[:96]

def _new_output(root: Path, study_id: str, run_name: str | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if run_name is not None:
        requested = Path(run_name)
        if requested.is_absolute() or len(requested.parts) != 1 or requested.name in {"", ".", ".."}:
            raise ValueError("run_name must be one portable directory name")
        target = root / requested.name
        target.mkdir()
        return target
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

def _load_upstream(directory: str | Path | None, *, expected_type: str) -> dict[str, Any]:
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
    phase_id: str, candidates: Sequence[Mapping[str, Any]],
    repeats: Sequence[int], folds: Sequence[int], epochs: int,
    inherited: Mapping[str, Any], device: str, jobs: int,
) -> Any:
    base = plan["base"]
    cases = []
    for candidate in candidates:
        overrides = {
            **dict(base["common_overrides"]), **dict(inherited), **dict(candidate["overrides"]),
            "training.fixed_epochs": int(epochs),
        }
        cases.append(
            {
                "case_id": str(candidate["case_id"]),
                "catalog_entry": str(base["catalog_entry"]),
                "screen_profile_id": str(base["profile_id"]),
                "output_group": str(base["output_group"]),
                "overrides": overrides, "rationale": str(candidate["rationale"]), "formal_profile": None,
            }
        )
    payload = {
        "schema_version": "ppg_frailty.study_plan.v2",
        "study": {
            "study_id": f"{plan['study']['study_id']}__{phase_id}",
            "kind": "catalog_sweep", "purpose": f"{plan['study']['purpose']} Phase={phase_id}.",
            "flow_position": str(plan["study"]["flow_position"]),
            "decision_role": str(plan["study"]["decision_role"]),
            "reference_case_id": None, "thesis_sections": list(plan["study"]["thesis_sections"]),
        },
        "catalog": {
            "path": str(plan["catalog"]["path"]), "balance_line": str(plan["catalog"]["balance_line"]),
            "scope": "selected_ordinary",
        },
        "search": {
            "method": "deterministic_sparse_profiles", "selection_seed": int(plan["search"]["selection_seed"]),
            "runtime_sampling": False,
            "interpretation": (
                f"Declared {phase_id} tuning phase; no final-test claim and no automatic final-model selection."
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
            "repeats": list(repeats), "folds": list(folds), "jobs": int(jobs), "device": device,
            "parallel_level": "cases", "continue_on_error": False, "allow_parallel_deep": False,
            "measure_operational_costs": bool(plan["execution"]["measure_operational_costs"]),
        },
        "output": {"root": "."}, "report": dict(plan["report"]),
    }
    return parse_study_plan(payload)

def _run_phase(
    plan: Mapping[str, Any],
    *,
    output: Path, phase_id: str, candidates: Sequence[Mapping[str, Any]],
    repeats: Sequence[int], folds: Sequence[int], epochs: int,
    inherited: Mapping[str, Any], pipeline_root: Path, device: str, jobs: int,
    progress_sink: ProgressSink,
    runner_factory: Callable[[Any, str, Path | None], StudyRunner] | None = None,
    resume_directory: Path | None = None, run_name: str | None = None,
) -> tuple[Any, Path, list[dict[str, Any]]]:
    standard = _phase_plan(
        plan, phase_id=phase_id, candidates=candidates, repeats=repeats, folds=folds,
        epochs=epochs, inherited=inherited, device=device, jobs=jobs,
    )
    runner = (
        runner_factory(standard, phase_id, resume_directory)
        if runner_factory is not None
        else StudyRunner(pipeline_root=pipeline_root, progress_sink=progress_sink)
    )
    result = runner.run(
        standard,
        output_root=output / "phases" / phase_id,
        resume_directory=resume_directory,
        run_name=None if resume_directory is not None else run_name,
    )
    if result.status != "passed":
        raise RuntimeError(f"{phase_id} did not complete: {result.status}")
    ranked = _rank_case_records(
        result.case_records, metric=str(plan["resource"]["ranking_metric"]),
        tie_break=str(plan["resource"]["tie_break_metric"]),
        expected_cells=len(repeats) * len(folds),
    )
    return result, result.output_directory, ranked

def _phase_resume_directory(output: Path, phase_id: str) -> Path | None:
    """Resolve one existing phase run without changing its persisted layout."""

    parent = output / "phases" / phase_id
    if not parent.is_dir():
        return None
    if (parent / "study_plan.yaml").is_file():
        return parent.resolve()
    candidates = tuple(
        path.resolve() for path in sorted(parent.iterdir(), key=lambda value: value.name)
        if path.is_dir() and (path / "study_plan.yaml").is_file()
    )
    if len(candidates) > 1:
        raise RuntimeError(f"resume phase {phase_id} has multiple study runs; refusing to guess")
    return candidates[0] if candidates else None

def _phase_run_name(output: Path, phase_id: str) -> str:
    """Return a unique, portable name for one V5 phase-level public run."""

    identity = f"{output.name}\0{phase_id}"
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    return f"{_safe_slug(output.name)[:72]}__{_safe_slug(phase_id)[:72]}__{suffix}"

@dataclass(frozen=True)
class _PhaseContext:
    plan: Mapping[str, Any]
    output: Path
    pipeline_root: Path
    device: str
    jobs: int
    progress_sink: ProgressSink
    runner_factory: Callable[[Any, str, Path | None], StudyRunner] | None
    resume: bool

    def run_resource(
        self, phase_id: str, candidates: Sequence[Mapping[str, Any]], prefix: str,
        inherited: Mapping[str, Any],
    ) -> tuple[Any, Path, list[dict[str, Any]]]:
        resource = self.plan["resource"]
        stem = f"{prefix}_" if prefix else ""
        resume_directory = _phase_resume_directory(self.output, phase_id) if self.resume else None
        run_name = (
            _phase_run_name(self.output, phase_id)
            if self.runner_factory is not None and resume_directory is None
            else None
        )
        return _run_phase(
            self.plan, output=self.output, phase_id=phase_id, candidates=candidates,
            repeats=resource[f"{stem}repeats"], folds=resource[f"{stem}folds"],
            epochs=int(resource[f"{stem}epochs"]), inherited=inherited,
            pipeline_root=self.pipeline_root, device=self.device, jobs=self.jobs,
            progress_sink=self.progress_sink, runner_factory=self.runner_factory,
            resume_directory=resume_directory, run_name=run_name,
        )

def _run_initial_phases(
    plan: Mapping[str, Any], phases: _PhaseContext, inherited: Mapping[str, Any]
) -> tuple[dict[str, Path], dict[str, list[dict[str, Any]]], Path, dict[str, Any]]:
    directories: dict[str, Path] = {}
    rankings: dict[str, list[dict[str, Any]]] = {}
    if str(plan["study"]["study_type"]) == "successive_halving":
        _, screen_dir, screen_rows = phases.run_resource(
            "screen_5epoch_reduced_cv", plan["candidates"], "screen", {}
        )
        directories["screen"] = screen_dir
        rankings["screen_ranking"] = screen_rows
        promoted_ids = {
            row["case_id"] for row in screen_rows[: int(plan["resource"]["promote_count"])]
        }
        candidates = [row for row in plan["candidates"] if row["case_id"] in promoted_ids]
        phase_id, prefix, directory_key, table_name = (
            "promoted_full_cv", "promotion", "promotion", "promotion_ranking"
        )
    else:
        candidates = plan["candidates"]
        phase_id, prefix, directory_key, table_name = "full_cv", "", "full_cv", "full_cv_ranking"
    _, selected_phase, selected_rows = phases.run_resource(
        phase_id, candidates, prefix, inherited
    )
    directories[directory_key] = selected_phase
    rankings[table_name] = selected_rows
    return directories, rankings, selected_phase, selected_rows[0]

def _metric_fields(name: str, values: Sequence[float]) -> dict[str, Any]:
    average, deviation = mean(values), pstdev(values)
    return {
        f"{name}_mean": average, f"{name}_sd": deviation,
        f"{name}_percent_mean_sd": f"{100.0*average:.1f} ± {100.0*deviation:.1f}",
    }

def _sort_rankings(
    rows: list[dict[str, Any]], metric: str, tie_break: str
) -> list[dict[str, Any]]:
    rows.sort(
        key=lambda row: (
            -float(row[f"{metric}_mean"]), -float(row[f"{tie_break}_mean"]),
            str(row["case_id"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows

def _rank_case_records(
    records: Sequence[Mapping[str, Any]], *, metric: str, tie_break: str, expected_cells: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        result = record.get("result")
        cells = result.get("cell_results", []) if isinstance(result, Mapping) else []
        if record.get("status") != "passed" or len(cells) != expected_cells:
            raise RuntimeError(f"candidate {record.get('case_id')} lacks complete ranking evidence")
        primary = [float(cell["metrics"][metric]) for cell in cells]
        secondary = [float(cell["metrics"][tie_break]) for cell in cells]
        if not all(math.isfinite(value) for value in (*primary, *secondary)):
            raise RuntimeError("candidate ranking metrics must be finite")
        rows.append(
            {
                "case_id": str(record["case_id"]), "cell_count": len(cells),
                **_metric_fields(metric, primary), **_metric_fields(tie_break, secondary),
            }
        )
    return _sort_rankings(rows, metric, tie_break)

def _completion_candidates(
    plan: Mapping[str, Any], promoted_ranking: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Return declared candidates that lack the full promotion resource."""

    if str(plan["study"]["study_type"]) != "successive_halving":
        raise ValueError("full-grid completion requires a successive-halving study")
    declared = {str(candidate["case_id"]): dict(candidate) for candidate in plan["candidates"]}
    promoted_ids = [str(row["case_id"]) for row in promoted_ranking]
    unknown = sorted(set(promoted_ids) - set(declared))
    if unknown:
        raise ValueError(f"promotion ranking contains unknown cases: {unknown}")
    expected_promoted = int(plan["resource"]["promote_count"])
    if len(promoted_ids) != expected_promoted or len(set(promoted_ids)) != len(promoted_ids):
        raise ValueError("promotion ranking does not match the declared promote_count")
    promoted_set = set(promoted_ids)
    return [declared[str(candidate["case_id"])] for candidate in plan["candidates"]
            if str(candidate["case_id"]) not in promoted_set]

def _merge_equal_resource_rankings(
    groups: Sequence[Sequence[Mapping[str, Any]]], *, metric: str, tie_break: str,
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
        if not math.isfinite(float(row[f"{metric}_mean"])) or not math.isfinite(float(row[f"{tie_break}_mean"])):
            raise ValueError("full-CV ranking metrics must be finite")
        row["metric_source"] = "equal_weight_fold_cell_mean_for_selection"
        row["selection_role"] = "exhaustive_full_grid_selection_evidence_after_completion"
    return _sort_rankings(merged, metric, tie_break)

def _resolved_config(phase_dir: Path, case_id: str) -> tuple[Path, dict[str, Any]]:
    manifest = json.loads((phase_dir / "study_manifest.json").read_text(encoding="utf-8"))
    matches = [row for row in manifest["cases"] if row["case_id"] == case_id]
    if len(matches) != 1:
        raise RuntimeError(f"cannot resolve selected case {case_id}")
    path = phase_dir / matches[0]["resolved_config_path"]
    config = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), str(path))
    return path, config

def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = list(rows)
    fields = list(dict.fromkeys(str(key) for row in values for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in values:
            writer.writerow({
                key: json.dumps(row.get(key), ensure_ascii=False, sort_keys=True)
                if isinstance(row.get(key), (dict, list, tuple)) else row.get(key)
                for key in fields
            })

def _write_table_pair(
    output: Path, *, name: str, rows: Sequence[Mapping[str, Any]], metric: str,
    selection_role: str | None = None,
) -> list[dict[str, Any]]:
    """Persist complete ranking data; figures are generated by analyse_report."""

    normalized = [
        {
            **dict(row),
            "metric_source": row.get("metric_source", "equal_weight_fold_cell_mean_for_selection"),
            "selection_role": selection_role or row.get(
                "selection_role", "declared_resource_orchestration_selection_evidence"
            ),
        }
        for row in rows
    ]
    _write_csv(output / "tables" / f"{name}.csv", normalized)
    _write_json(output / "tables" / f"{name}.json", normalized)
    return normalized

def _root_ranking_selection_role(plan: Mapping[str, Any], table_name: str) -> str:
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

def _selection_payload(
    plan: Mapping[str, Any], output: Path, selected_row: Mapping[str, Any],
    selected_phase: Path, upstream: Mapping[str, Any] | None,
) -> dict[str, Any]:
    resolved_path, resolved_config = _resolved_config(selected_phase, str(selected_row["case_id"]))
    return {
        "schema_version": "ppg_frailty.tuning_selection.v1",
        "study_id": plan["study"]["study_id"], "study_type": str(plan["study"]["study_type"]),
        "selection_scope": "development_tuning_only_not_final_test",
        "case_id": selected_row["case_id"], "ranking": selected_row,
        "resolved_config_source": resolved_path.relative_to(output).as_posix(),
        "resolved_config": resolved_config, "upstream_selection": upstream,
    }

def _write_ranking_tables(
    output: Path, plan: Mapping[str, Any], tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    metric = str(plan["resource"]["ranking_metric"])
    for name, rows in tables.items():
        _write_table_pair(output, name=name, rows=rows, metric=metric,
                          selection_role=_root_ranking_selection_role(plan, name))

def run_hyperparameter_study(
    plan_path: str | Path,
    *,
    pipeline_root: str | Path, upstream_study: str | Path | None = None,
    output_root: str | Path | None = None, device: str | None = None, jobs: int | None = None,
    progress_sink: ProgressSink | None = None, run_name: str | None = None,
    resume: str | Path | None = None,
    phase_runner_factory: (Callable[[Any, str, Path | None], StudyRunner] | None) = None,
) -> Path:
    """Run one plan, select deterministically, and archive every phase."""

    plan = load_hyperparameter_plan(plan_path)
    root = Path(pipeline_root).resolve()
    raw_output = Path(output_root or plan["output"]["root"])
    output_parent = raw_output if raw_output.is_absolute() else root / raw_output
    resumed = resume is not None
    if resumed and run_name is not None:
        raise ValueError("run_name cannot be combined with resume")
    output = Path(resume).resolve() if resumed else _new_output(
        output_parent.resolve(), str(plan["study"]["study_id"]), run_name=run_name
    )
    persisted_plan_path = output / "study_plan.yaml"
    normalized_plan = {key: value for key, value in plan.items() if key != "plan_path"}
    if resumed:
        if not output.is_dir() or not persisted_plan_path.is_file():
            raise FileNotFoundError(f"hyperparameter resume lacks study_plan.yaml: {output}")
        persisted_plan = load_hyperparameter_plan(persisted_plan_path)
        normalized_persisted = {key: value for key, value in persisted_plan.items() if key != "plan_path"}
        if normalized_persisted != normalized_plan:
            raise ValueError("hyperparameter resume plan differs from the persisted study plan")
    else:
        with persisted_plan_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(normalized_plan, stream, sort_keys=False, allow_unicode=True)
    sink = progress_sink or NullProgressSink()
    resolved_device = str(device or plan["execution"]["device"])
    if not resolved_device.startswith("cuda"):
        raise ValueError("hyperparameter training must use CUDA")
    resolved_jobs = _positive_int(jobs if jobs is not None else plan["execution"]["jobs"], "jobs")
    study_type = str(plan["study"]["study_type"])
    inherited: dict[str, Any] = {}
    upstream_payload: dict[str, Any] | None = None
    if study_type != "successive_halving":
        upstream_payload = _load_upstream(upstream_study, expected_type=study_type)
        upstream_config = upstream_payload["resolved_config"]
        paths = ("training.batch_size", "training.learning_rate")
        if study_type != "dependent_regularization_grid":
            paths = (
                "training.batch_size",
                "training.learning_rate",
                "training.weight_decay",
                "training.label_smoothing",
                "model.dropout",
            )
        inherited = {path: _dotted_get(upstream_config, path) for path in paths}
    phases = _PhaseContext(plan, output, root, resolved_device, resolved_jobs, sink, phase_runner_factory, resumed)
    phase_directories, ranking_tables, selected_phase, selected_row = _run_initial_phases(
        plan, phases, inherited
    )
    selected = _selection_payload(plan, output, selected_row, selected_phase, upstream_payload)
    _write_json(output / "selected_configuration.json", selected)
    manifest = {
        "schema_version": "ppg_frailty.hyperparameter_study_manifest.v1",
        "status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "study": plan["study"],
        "plan_sha256": hashlib.sha256(Path(plan["plan_path"]).read_bytes()).hexdigest(),
        "device": resolved_device, "jobs": resolved_jobs,
        "phase_directories": {key: value.relative_to(output).as_posix() for key, value in phase_directories.items()},
        "selected_case_id": selected_row["case_id"], "ranking_tables": list(ranking_tables),
    }
    if phase_runner_factory is not None:
        manifest["phase_output_layout"] = "comparison/repeat/fold"
    _write_json(output / "study_manifest.json", manifest)
    _write_ranking_tables(output, plan, ranking_tables)
    return output

def inspect_successive_halving_completion(study_dir: str | Path) -> dict[str, Any]:
    """Describe the unpromoted full-CV work without starting training."""

    root = Path(study_dir).resolve()
    manifest_path = root / "study_manifest.json"
    manifest = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")), "study_manifest")
    plan = load_hyperparameter_plan(root / "study_plan.yaml")
    if str(plan["study"]["study_type"]) != "successive_halving":
        raise ValueError("completion is only valid for successive-halving studies")
    if "completion" in manifest.get("phase_directories", {}):
        remaining: list[dict[str, Any]] = []
        status = "already_complete"
    else:
        promoted = list(json.loads((root / "tables" / "promotion_ranking.json").read_text(encoding="utf-8")))
        remaining = _completion_candidates(plan, promoted)
        status = "ready"
    resource = plan["resource"]
    return {
        "status": status, "study_dir": str(root),
        "candidate_ids": [str(row["case_id"]) for row in remaining],
        "candidate_count": len(remaining), "repeats": list(resource["promotion_repeats"]),
        "folds": list(resource["promotion_folds"]), "fixed_epochs": int(resource["promotion_epochs"]),
        "fold_cell_count": (len(remaining) * len(resource["promotion_repeats"]) * len(resource["promotion_folds"])),
        "reuses_existing_promoted_full_cv": True,
    }

def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.completion.tmp")
    _write_json(temporary, payload)
    temporary.replace(path)

def complete_successive_halving_study(
    study_dir: str | Path,
    *,
    pipeline_root: str | Path, device: str | None = None, jobs: int | None = None,
    progress_sink: ProgressSink | None = None,
    phase_runner_factory: (Callable[[Any, str, Path | None], StudyRunner] | None) = None,
) -> Path:
    """Full-CV only the unpromoted cases and then rank the complete grid."""

    output = Path(study_dir).resolve()
    manifest_path = output / "study_manifest.json"
    selected_path = output / "selected_configuration.json"
    original_manifest_text = manifest_path.read_text(encoding="utf-8")
    original_selected_text = selected_path.read_text(encoding="utf-8")
    manifest = _mapping(json.loads(original_manifest_text), "study_manifest")
    original_selected = _mapping(json.loads(original_selected_text), "selected_configuration")
    if str(manifest.get("status")) != "passed":
        raise RuntimeError("the source successive-halving study is not passed")
    if "completion" in manifest.get("phase_directories", {}):
        raise RuntimeError("the successive-halving study is already complete")
    plan = load_hyperparameter_plan(output / "study_plan.yaml")
    if str(plan["study"]["study_type"]) != "successive_halving":
        raise ValueError("completion is only valid for successive-halving studies")
    existing_rankings = {
        str(name): list(json.loads((output / "tables" / f"{name}.json").read_text(encoding="utf-8")))
        for name in manifest["ranking_tables"]
    }
    promotion_ranking = existing_rankings.get("promotion_ranking")
    if not promotion_ranking:
        raise RuntimeError("the source study lacks a promotion ranking")
    remaining = _completion_candidates(plan, promotion_ranking)
    if not remaining:
        raise RuntimeError("no unpromoted candidates remain")
    resolved_device = str(device or manifest.get("device") or plan["execution"]["device"])
    if not resolved_device.startswith("cuda"):
        raise ValueError("hyperparameter training must use CUDA")
    job_value = jobs if jobs is not None else manifest.get("jobs", plan["execution"]["jobs"])
    resolved_jobs = _positive_int(job_value, "jobs")
    resource = plan["resource"]
    sink = progress_sink or NullProgressSink()
    phases = _PhaseContext(
        plan, output, Path(pipeline_root).resolve(), resolved_device, resolved_jobs,
        sink, phase_runner_factory, True,
    )
    _, completion_dir, completion_ranking = phases.run_resource("nonpromoted_full_cv", remaining, "promotion", {})
    for row in completion_ranking:
        row["metric_source"] = "equal_weight_fold_cell_mean_for_selection"
        row["selection_role"] = "completion_subset_full_cv_evidence_not_standalone_selection"
    combined_ranking = _merge_equal_resource_rankings(
        (promotion_ranking, completion_ranking), metric=str(resource["ranking_metric"]),
        tie_break=str(resource["tie_break_metric"]),
        expected_case_ids=[str(row["case_id"]) for row in plan["candidates"]],
    )
    selected_row = combined_ranking[0]
    completion_ids = {str(row["case_id"]) for row in remaining}
    phase_directories = {str(name): output / str(relative) for name, relative in manifest["phase_directories"].items()}
    phase_directories["completion"] = completion_dir
    selected_phase = (
        completion_dir if str(selected_row["case_id"]) in completion_ids else phase_directories["promotion"]
    )
    selected = {
        **_selection_payload(plan, output, selected_row, selected_phase, None),
        "precompletion_case_id": original_selected.get("case_id"),
        "selection_evidence": "complete_six_candidate_full_5x5_grid",
    }
    ranking_tables = {
        **existing_rankings, "nonpromoted_full_cv_ranking": completion_ranking,
        "all_candidates_full_cv_ranking": combined_ranking,
    }
    completed_manifest = {
        **manifest,
        "status": "passed", "device": resolved_device, "jobs": resolved_jobs,
        "phase_directories": {name: path.relative_to(output).as_posix() for name, path in phase_directories.items()},
        "selected_case_id": selected_row["case_id"], "ranking_tables": list(ranking_tables),
        "completion_created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "successive_halving_completion": {
            "status": "passed", "reused_promoted_case_ids": [str(row["case_id"]) for row in promotion_ranking],
            "newly_trained_case_ids": sorted(completion_ids),
            "full_cv_candidate_count": len(combined_ranking),
            "full_cv_fold_cell_count": sum(int(row["cell_count"]) for row in combined_ranking),
            "selection_recomputed_after_complete_grid": True,
        },
    }
    if phase_runner_factory is not None:
        completed_manifest["phase_output_layout"] = "comparison/repeat/fold"
    (output / "precompletion_selected_configuration.json").write_text(original_selected_text, encoding="utf-8")
    (output / "precompletion_study_manifest.json").write_text(original_manifest_text, encoding="utf-8")
    _write_json_atomic(selected_path, selected)
    _write_json_atomic(manifest_path, completed_manifest)
    _write_ranking_tables(output, plan, ranking_tables)
    return output

def generate_hyperparameter_report(output: str | Path, **context: Any) -> dict[str, Any]:
    """Compatibility adapter for the centralized specialized reporter."""

    from ..reporting.specialized import generate_hyperparameter_report as generate

    return generate(output, **context)

def regenerate_hyperparameter_report(study_dir: str | Path) -> dict[str, Any]:
    """Rebuild the root and nested reports without retraining."""

    from ..reporting.specialized import rebuild_hyperparameter_report

    return rebuild_hyperparameter_report(study_dir)


__all__ = [
    "complete_successive_halving_study",
    "generate_hyperparameter_report",
    "inspect_successive_halving_completion",
    "load_hyperparameter_plan",
    "regenerate_hyperparameter_report",
    "run_hyperparameter_study",
]
