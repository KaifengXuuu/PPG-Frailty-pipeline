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
    raw = _strict(
        yaml.safe_load(source.read_text(encoding="utf-8")),
        "plan",
        {
            "schema_version", "study", "catalog", "base", "candidates",
            "resource", "execution", "output", "report",
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
            "selection_seed": 42,
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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = list(rows)
    fields = list(dict.fromkeys(str(key) for row in values for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
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
            "selection_role": row.get(
                "selection_role", "successive_halving_selection_evidence"
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
                "selection_role", "successive_halving_selection_evidence"
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
        errors = [100.0 * float(row[f"{metric}_sd"]) for row in normalized_rows]
        figure, axis = plt.subplots(
            figsize=(max(7.0, 0.8 * len(normalized_rows)), 4.8)
        )
        axis.bar(range(len(rows)), values, yerr=errors, capsize=4)
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


def _participant_oof_rankings(
    phase_directories: Mapping[str, Path],
) -> dict[str, list[dict[str, Any]]]:
    """Read nested report evidence without changing orchestration selection."""

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
            balanced_accuracy_sd = float(
                item["repeat_balanced_accuracy_population_sd"]
            )
            macro_f1 = float(item["participant_mean_macro_f1"])
            macro_f1_sd = float(item["repeat_macro_f1_population_sd"])
            rows.append(
                {
                    "case_id": str(item["case_id"]),
                    "cell_count": int(item["fold_cell_count"]),
                    "balanced_accuracy_mean": balanced_accuracy,
                    "balanced_accuracy_sd": balanced_accuracy_sd,
                    "balanced_accuracy_percent_mean_sd": (
                        f"{100.0 * balanced_accuracy:.1f} ± "
                        f"{100.0 * balanced_accuracy_sd:.1f}"
                    ),
                    "macro_f1_mean": macro_f1,
                    "macro_f1_sd": macro_f1_sd,
                    "macro_f1_percent_mean_sd": (
                        f"{100.0 * macro_f1:.1f} ± {100.0 * macro_f1_sd:.1f}"
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
    values = list(rows)
    if not values:
        return "N/A — no rows."
    fields = list(dict.fromkeys(str(key) for row in values for key in row))
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
    return "\n".join(lines)


def _html_table(rows: Sequence[Mapping[str, Any]]) -> str:
    values = list(rows)
    if not values:
        return "<p>N/A — no rows.</p>"
    fields = list(dict.fromkeys(str(key) for row in values for key in row))
    headings = "".join(f"<th>{html_escape(field)}</th>" for field in fields)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{html_escape(str(row.get(field, '')))}</td>" for field in fields
        )
        + "</tr>"
        for row in values
    )
    return f"<table><thead><tr>{headings}</tr></thead><tbody>{body}</tbody></table>"


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


def _write_root_report(
    output: Path,
    *,
    plan: Mapping[str, Any],
    phase_directories: Mapping[str, Path],
    ranking_tables: Mapping[str, Sequence[Mapping[str, Any]]],
    selected: Mapping[str, Any],
    inherited: Mapping[str, Any],
) -> None:
    from ..reporting.components import markdown_test_component_table

    metric = str(plan["resource"]["ranking_metric"])
    table_names: list[str] = []
    display_rankings: dict[str, list[dict[str, Any]]] = {}
    for name, rows in ranking_tables.items():
        display_rankings[name] = _write_table_pair(
            output, name=name, rows=rows, metric=metric
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
    selected_config = (
        dict(selected["resolved_config"])
        if isinstance(selected.get("resolved_config"), Mapping)
        else {}
    )
    selected_signal = (
        dict(selected_config["signal"])
        if isinstance(selected_config.get("signal"), Mapping)
        else {}
    )
    selected_imu = (
        dict(selected_signal["imu"])
        if isinstance(selected_signal.get("imu"), Mapping)
        else {}
    )
    selected_model = (
        dict(selected_config["model"])
        if isinstance(selected_config.get("model"), Mapping)
        else {}
    )
    gravity_method = str(
        selected_imu.get("gravity_method", "not_available_in_report_fixture")
    )
    model_channels = list(selected_model.get("input_channel_order", ()))
    candidate_rows = [
        {
            "participating_cases": row["case_id"],
            "component_role": "classifier_tuning_candidate",
            "module_id": "InceptionTimeFull",
            "execution_state": "executed",
            "input_data": json.dumps(
                {
                    "dataset": "Frailty29 static roles B/R1-R4",
                    "signal_views": [
                        "RED/IR amplitude-preserving analysis view",
                        f"{gravity_method} processed physical A_dyn/GX/GY/GZ",
                        {
                            "dl_only_model_input_channel_order": model_channels,
                        },
                    ],
                    "sampling_rate_hz": 64.0,
                    "window_s": 5.0,
                    "hop_s": 2.5,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "fixed_parameters": json.dumps(
                {
                    **dict(plan["base"]["common_overrides"]),
                    **dict(inherited),
                    **dict(row["overrides"]),
                    "resource_contract": plan["resource"],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "algorithm_kernel_description": (
                "InceptionTimeFull raw-DL candidate; candidate-specific values "
                "are combined with the selected B0+B2+B7 signal/training state; "
                f"IMU gravity method={gravity_method}."
            ),
        }
        for row in plan["candidates"]
    ]
    _write_csv(output / "tables" / "test_components.csv", candidate_rows)
    (output / "tables" / "test_components.json").write_text(
        json.dumps(candidate_rows, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    table_names.append("test_components")
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
    _write_csv(output / "tables" / "table_figure_pairs.csv", pair_rows)
    table_names.append("table_figure_pairs")
    _write_workbook(output, table_names)
    selected_case = str(selected["case_id"])
    selection_evidence = str(
        selected.get(
            "selection_evidence", "successive_halving_promoted_full_cv"
        )
    )
    phase_lines = [
        f"- `{name}`: [{path.name}]({path.relative_to(output).as_posix()}/STUDY_SUMMARY.md)"
        for name, path in phase_directories.items()
    ]
    ranking_sections: list[str] = []
    for name, rows in display_rankings.items():
        ranking_sections.extend((f"### {name}", "", _markdown_table(rows), ""))
    participant_sections: list[str] = []
    for name, rows in display_participant_rankings.items():
        participant_sections.extend(
            (f"### {name}", "", _markdown_table(rows), "")
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
        "## Seeds and data splits",
        "",
        _markdown_table(reproducibility_rows),
        "",
        "All compact percentages use `mean ± population SD`; raw numeric columns "
        "remain available in JSON and each displayed CSV table occupies one workbook sheet.",
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
        f"<h2>{html_escape(name)}</h2>{_html_table(rows)}"
        for name, rows in display_rankings.items()
    )
    html_participant_ranking = "\n".join(
        f"<h2>{html_escape(name)}</h2>{_html_table(rows)}"
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
        _html_table(candidate_rows),
        "<h2>Seeds and data splits</h2>",
        _html_table(reproducibility_rows),
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
    backup = output / "result_backup"
    backup.mkdir(exist_ok=True)
    for name in (
        "study_plan.yaml", "study_manifest.json", "selected_configuration.json",
        "precompletion_study_manifest.json",
        "precompletion_selected_configuration.json",
        "STUDY_SUMMARY.md", "STUDY_SUMMARY.html", "TEST_COMPONENTS.md",
        "outputs_index.json",
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
