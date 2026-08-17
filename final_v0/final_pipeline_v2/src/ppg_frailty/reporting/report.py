"""Write complete, human-readable study reports and an output inventory."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .analyze import StudyAnalysis, analyze_study
from .collect import CollectedStudy, collect_study
from .plots import clear_static_figure_artifacts, generate_static_figures


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).replace("|", r"\|")


def _markdown_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[tuple[str, str]],
) -> list[str]:
    if not rows:
        return ["N/A — no rows were available.", ""]
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_fmt(row.get(field)) for field, _ in columns)
            + " |"
        )
    lines.append("")
    return lines


def _study_info(collected: CollectedStudy) -> Mapping[str, Any]:
    value = collected.plan.get("study", {})
    return value if isinstance(value, Mapping) else {}


def _report_markdown(
    collected: CollectedStudy,
    analysis: StudyAnalysis,
    figures: Sequence[Mapping[str, Any]],
) -> str:
    study = _study_info(collected)
    manifest = collected.manifest
    execution = collected.plan.get("execution", {})
    axes = collected.plan.get("axes", ())
    catalog = collected.plan.get("catalog", {})
    search = collected.plan.get("search", {})
    is_catalog_sweep = study.get("kind") == "catalog_sweep"
    config_source_label = (
        f"Catalog: {catalog.get('path', 'N/A')} "
        f"(scope={catalog.get('scope', 'N/A')}, "
        f"balance={catalog.get('balance_line', 'N/A')})"
        if is_catalog_sweep and isinstance(catalog, Mapping)
        else f"Base pipeline config: {collected.plan.get('base_config', 'N/A')}"
    )
    lines = [
        f"# V2 study summary — {study.get('study_id', collected.root.name)}",
        "",
        "> This report is descriptive evidence for manual review. It does not "
        "automatically select a final use case or winner.",
        "",
        "## Scientific context",
        "",
        f"- Study kind: {study.get('kind', 'N/A')}",
        f"- Purpose: {study.get('purpose', 'N/A')}",
        f"- Position in use-case selection flow: {study.get('flow_position', 'N/A')}",
        f"- Decision role: {study.get('decision_role', 'N/A')}",
        f"- Thesis sections: {_fmt(study.get('thesis_sections', []))}",
        f"- {config_source_label}",
        f"- Reference case: {manifest.get('reference_case_id') or 'N/A'}",
        "",
        "## Run controls and completeness",
        "",
        f"- Repeats requested: {_fmt(execution.get('repeats', []))}",
        f"- Folds requested: {_fmt(execution.get('folds', []))}",
        f"- Case-level jobs requested: {execution.get('jobs', 'N/A')}",
        f"- Effective jobs: {manifest.get('effective_jobs', 'N/A')}",
        f"- Planned / passed / failed / not-run cases: "
        f"{manifest.get('planned_case_count', 'N/A')} / "
        f"{manifest.get('passed_case_count', 'N/A')} / "
        f"{manifest.get('failed_case_count', 'N/A')} / "
        f"{manifest.get('not_run_case_count', 'N/A')}",
        f"- Planned / reported / passed / failed / not-run cells: "
        f"{manifest.get('planned_cell_count', 'N/A')} / "
        f"{manifest.get('reported_cell_count', 'N/A')} / "
        f"{manifest.get('passed_cell_count', 'N/A')} / "
        f"{manifest.get('failed_cell_count', 'N/A')} / "
        f"{manifest.get('not_run_cell_count', 'N/A')}",
        f"- Resume-skipped passed cases: {manifest.get('resumed_case_count', 0)}",
        "",
        "## Varied and controlled parameters",
        "",
    ]
    if axes:
        lines.extend(
            [
                f"- {axis.get('path')}: values={_fmt(axis.get('values'))}; "
                f"reference={_fmt(axis.get('reference'))}"
                for axis in axes
                if isinstance(axis, Mapping)
            ]
        )
    elif is_catalog_sweep and isinstance(search, Mapping):
        lines.extend(
            [
                "- Explicit deterministic sparse catalog profiles; this is a "
                "screening comparison, not a single-factor causal ablation.",
                f"- Search method: {search.get('method', 'N/A')}",
                f"- Runtime parameter sampling: {search.get('runtime_sampling', 'N/A')}",
                f"- Profile-design seed: {search.get('selection_seed', 'N/A')}",
                f"- Interpretation: {search.get('interpretation', 'N/A')}",
            ]
        )
    else:
        lines.append("- No scientific axis: this is a single-config run.")
    lines.extend(
        [
            "",
            "The complete resolved varied/controlled tables are "
            "[varied_parameters.csv](tables/varied_parameters.csv) and "
            "[controlled_parameters.csv](tables/controlled_parameters.csv). "
            "Execution controls such as jobs are not scientific grid variables.",
            "",
            "<details><summary>Complete controlled-parameter list "
            f"({len(collected.controlled_parameters)} rows)</summary>",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            collected.controlled_parameters,
            (
                ("parameter_path", "Controlled parameter"),
                ("value", "Resolved value"),
            ),
        )
    )
    lines.extend(
        [
            "</details>",
            "",
            "## Predictive ranking",
            "",
            "Ranking is by participant-level mean balanced accuracy. Macro-F1 and "
            "both lower-bound columns remain visible; deployment measurements do "
            "not filter this table.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            analysis.predictive_leaderboard,
            (
                ("predictive_rank", "Rank"),
                ("case_id", "Case"),
                ("participant_mean_balanced_accuracy", "BA"),
                ("participant_mean_macro_f1", "Macro-F1"),
                ("balanced_accuracy_lcb95", "BA LCB95"),
                ("macro_f1_lcb95", "Macro-F1 LCB95"),
                ("repeat_balanced_accuracy_ci95_low", "BA CI95 low"),
                ("repeat_balanced_accuracy_ci95_high", "BA CI95 high"),
                ("repeat_macro_f1_ci95_low", "Macro-F1 CI95 low"),
                ("repeat_macro_f1_ci95_high", "Macro-F1 CI95 high"),
                ("worst_fold_balanced_accuracy", "Worst-fold BA"),
                ("worst_class_recall", "Worst recall"),
                ("worst_class_f1", "Worst F1"),
                ("metric_source", "Source"),
            ),
        )
    )
    lines.extend(
        [
            "## Worst-class F1 stability review",
            "",
            "This secondary view reorders the BA-ranked complete cases by worst-class "
            "F1, then repeat variability. It remains descriptive and does not select "
            "a winner.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            analysis.worst_class_f1_stability,
            (
                ("worst_class_f1_stability_rank", "Stability rank"),
                ("predictive_rank", "BA rank"),
                ("case_id", "Case"),
                ("worst_class_f1", "Worst F1"),
                ("worst_class_recall", "Worst recall"),
                ("participant_mean_balanced_accuracy", "Mean BA"),
                ("repeat_balanced_accuracy_population_sd", "Repeat BA SD"),
                ("balanced_accuracy_lcb95", "BA LCB95"),
            ),
        )
    )
    lines.extend(["## Incomplete cases excluded from ranking", ""])
    lines.extend(
        _markdown_table(
            analysis.incomplete_cases,
            (
                ("case_id", "Case"),
                ("status", "Status"),
                ("incompleteness_reasons", "Reasons"),
                ("repeat_count", "Reported repeats"),
                ("expected_repeat_count", "Expected repeats"),
                ("passed_fold_cell_count", "Passed cells"),
                ("expected_fold_cell_count", "Expected cells"),
            ),
        )
    )
    lines.extend(
        [
            "## Deployment measurements (separate from predictive ranking)",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            analysis.deployment_table,
            (
                ("case_id", "Case"),
                ("parameter_count", "Parameters"),
                ("inference_cost", "Inference cost"),
                ("deployment_readiness", "Status"),
                ("reported_exclusion_reason", "Reported note"),
            ),
        )
    )
    lines.extend(
        [
            "## Route × role coverage and feature availability",
            "",
            "This table separates direct and processed rate paths, retained coverage, "
            "unavailable predictors, and reducer failures for each role/route state.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            analysis.route_role_coverage,
            (
                ("case_id", "Case"),
                ("role", "Role"),
                ("route_state", "Route state"),
                ("signal_route", "Signal route"),
                ("retained_coverage", "Retained coverage"),
                ("direct_rate_record_count", "Direct"),
                ("processed_rate_record_count", "Processed"),
                ("unavailable_predictor_rate", "Unavailable predictors"),
                ("reducer_failure_count", "Reducer failures"),
            ),
        )
    )
    lines.extend(
        [
            "## Quality-component distributions by route and role",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            analysis.quality_distributions,
            (
                ("case_id", "Case"),
                ("role", "Role"),
                ("route_state", "Route state"),
                ("component", "Component"),
                ("valid_count", "Valid n"),
                ("unavailable_rate", "Unavailable"),
                ("mean", "Mean"),
                ("population_sd", "SD"),
                ("minimum", "Min"),
                ("maximum", "Max"),
            ),
        )
    )
    failed = [
        row
        for row in collected.case_records
        if str(row.get("status")) not in {"passed"}
    ]
    lines.extend(["## Failed or incomplete cases", ""])
    lines.extend(
        _markdown_table(
            failed,
            (
                ("case_id", "Case"),
                ("status", "Status"),
                ("error_type", "Error type"),
                ("error", "Message"),
            ),
        )
    )
    lines.extend(["## Figure status", ""])
    lines.extend(
        _markdown_table(
            figures,
            (
                ("figure", "Figure"),
                ("status", "Status"),
                ("path", "Path"),
                ("reason", "Reason"),
            ),
        )
    )
    lines.extend(["## Limitations and N/A items", ""])
    if analysis.notes:
        lines.extend(f"- {note}" for note in analysis.notes)
    else:
        lines.append("- No collection limitation was recorded.")
    lines.extend(
        [
            "",
            "## Output navigation",
            "",
            "- [outputs_index.json](outputs_index.json): machine-readable inventory",
            "- [study_summary.json](study_summary.json): report context and tables",
            "- [tables/predictive_leaderboard.csv](tables/predictive_leaderboard.csv)",
            "- [tables/metric_distribution_summary.csv](tables/metric_distribution_summary.csv)",
            "- [tables/worst_class_f1_stability.csv](tables/worst_class_f1_stability.csv)",
            "- [tables/incomplete_cases.csv](tables/incomplete_cases.csv)",
            "- [tables/confusion_counts.csv](tables/confusion_counts.csv)",
            "- [tables/confusion_row_normalized.csv](tables/confusion_row_normalized.csv)",
            "- [tables/top_confusion_matrices/](tables/top_confusion_matrices/): top-case count and row-normalized CSVs",
            "- [tables/deployment_measurements.csv](tables/deployment_measurements.csv)",
            "- [figures/plot_status.json](figures/plot_status.json)",
            "",
        ]
    )
    return "\n".join(lines)


def _html_table(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]
) -> str:
    if not rows:
        return "<p><em>N/A — no rows were available.</em></p>"
    header = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(_fmt(row.get(field)))}</td>"
                for field, _ in columns
            )
            + "</tr>"
        )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _report_html(
    collected: CollectedStudy,
    analysis: StudyAnalysis,
    figures: Sequence[Mapping[str, Any]],
) -> str:
    study = _study_info(collected)
    generated = [row for row in figures if row.get("status") == "generated"]
    figure_html = "".join(
        f"<figure><img src='{html.escape(str(row['path']))}' alt='"
        f"{html.escape(str(row['figure']))}'><figcaption>"
        f"{html.escape(str(row['figure']))}</figcaption></figure>"
        for row in generated
    )
    limitations = "".join(f"<li>{html.escape(note)}</li>" for note in analysis.notes)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>V2 study — {html.escape(str(study.get("study_id", collected.root.name)))}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1280px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%;font-size:.9rem}}th,td{{border:1px solid #ccc;padding:.4rem;text-align:left}}
th{{background:#f0f3f6}}img{{max-width:100%;height:auto}}figure{{margin:2rem 0}}
.notice{{padding:1rem;background:#fff5cc;border-left:4px solid #c49000}}
</style></head><body>
<h1>V2 study — {html.escape(str(study.get("study_id", collected.root.name)))}</h1>
<p class="notice">Descriptive manual-review report; no automatic winner is selected.</p>
<p><strong>Purpose:</strong> {html.escape(str(study.get("purpose", "N/A")))}</p>
<p><strong>Flow position:</strong> {html.escape(str(study.get("flow_position", "N/A")))}</p>
<h2>Predictive leaderboard</h2>
{_html_table(analysis.predictive_leaderboard, (
    ("predictive_rank", "Rank"), ("case_id", "Case"),
    ("participant_mean_balanced_accuracy", "BA"),
    ("participant_mean_macro_f1", "Macro-F1"),
    ("balanced_accuracy_lcb95", "BA LCB95"),
    ("macro_f1_lcb95", "Macro-F1 LCB95"),
    ("repeat_balanced_accuracy_ci95_low", "BA CI95 low"),
    ("repeat_balanced_accuracy_ci95_high", "BA CI95 high"),
    ("repeat_macro_f1_ci95_low", "Macro-F1 CI95 low"),
    ("repeat_macro_f1_ci95_high", "Macro-F1 CI95 high"),
    ("worst_fold_balanced_accuracy", "Worst-fold BA"),
    ("worst_class_f1", "Worst F1"),
))}
<h2>Worst-class F1 stability review</h2>
{_html_table(analysis.worst_class_f1_stability, (
    ("worst_class_f1_stability_rank", "Stability rank"),
    ("predictive_rank", "BA rank"), ("case_id", "Case"),
    ("worst_class_f1", "Worst F1"),
    ("participant_mean_balanced_accuracy", "Mean BA"),
    ("repeat_balanced_accuracy_population_sd", "Repeat BA SD"),
))}
<h2>Incomplete cases excluded from ranking</h2>
{_html_table(analysis.incomplete_cases, (
    ("case_id", "Case"), ("status", "Status"),
    ("incompleteness_reasons", "Reasons"),
    ("repeat_count", "Reported repeats"),
    ("expected_repeat_count", "Expected repeats"),
    ("passed_fold_cell_count", "Passed cells"),
    ("expected_fold_cell_count", "Expected cells"),
))}
<h2>Deployment measurements (not a predictive filter)</h2>
{_html_table(analysis.deployment_table, (
    ("case_id", "Case"), ("parameter_count", "Parameters"),
    ("inference_cost", "Inference cost"), ("deployment_readiness", "Status"),
))}
<h2>Route × role coverage and feature availability</h2>
{_html_table(analysis.route_role_coverage, (
    ("case_id", "Case"), ("role", "Role"), ("route_state", "Route state"),
    ("signal_route", "Signal route"), ("retained_coverage", "Retained coverage"),
    ("direct_rate_record_count", "Direct"), ("processed_rate_record_count", "Processed"),
    ("unavailable_predictor_rate", "Unavailable predictors"),
    ("reducer_failure_count", "Reducer failures"),
))}
<h2>Quality-component distributions</h2>
{_html_table(analysis.quality_distributions, (
    ("case_id", "Case"), ("role", "Role"), ("route_state", "Route state"),
    ("component", "Component"), ("valid_count", "Valid n"),
    ("unavailable_rate", "Unavailable"), ("mean", "Mean"),
    ("population_sd", "SD"), ("minimum", "Min"), ("maximum", "Max"),
))}
<h2>Figures</h2>{figure_html or "<p><em>N/A — no generated figures.</em></p>"}
<h2>Limitations</h2><ul>{limitations or "<li>None recorded.</li>"}</ul>
<p>See <a href="outputs_index.json">outputs_index.json</a> for every artifact.</p>
</body></html>
"""


@dataclass(frozen=True)
class ReportResult:
    study_directory: Path
    summary_markdown: Path
    summary_html: Path | None
    output_index: Path
    table_count: int
    generated_figure_count: int
    na_figure_count: int

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


def _index_entry(
    root: Path,
    path: Path,
    *,
    artifact_type: str,
    description: str,
    status: str = "available",
) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "type": artifact_type,
        "status": status,
        "description": description,
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file()
            else None
        ),
    }


def _safe_filename(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.")
    return cleaned[:100] or "case"


def _write_top_confusion_matrix_files(
    root: Path,
    tables: Path,
    analysis: StudyAnalysis,
) -> tuple[list[dict[str, Any]], int]:
    """Write top-case count and row-normalized matrices as standalone CSVs."""

    target = tables / "top_confusion_matrices"
    target.mkdir(exist_ok=True)
    for old in target.glob("*.csv"):
        if old.is_file() or old.is_symlink():
            old.unlink()
    matrices = {
        str(row.get("case_id")): row for row in analysis.confusion_matrices
    }
    entries: list[dict[str, Any]] = []
    written = 0
    for ranked in analysis.predictive_leaderboard:
        case_id = str(ranked["case_id"])
        matrix_row = matrices.get(case_id)
        if matrix_row is None:
            continue
        order = list(matrix_row.get("class_order", ()))
        matrix = matrix_row.get("confusion_matrix")
        if not isinstance(matrix, (list, tuple)) or len(matrix) != len(order):
            continue
        try:
            numeric = [[float(value) for value in row] for row in matrix]
        except (TypeError, ValueError):
            continue
        if any(len(row) != len(order) for row in numeric):
            continue
        rank = int(ranked["predictive_rank"])
        stem = f"rank_{rank:02d}_{_safe_filename(case_id)}"
        count_rows = [
            {
                "true_class": order[row_index],
                **{
                    f"predicted_{label}": numeric[row_index][column_index]
                    for column_index, label in enumerate(order)
                },
            }
            for row_index in range(len(order))
        ]
        normalized_rows: list[Mapping[str, Any]] = []
        for row_index, row in enumerate(numeric):
            total = sum(row)
            normalized_rows.append(
                {
                    "true_class": order[row_index],
                    **{
                        f"predicted_{label}": (
                            row[column_index] / total if total > 0.0 else None
                        )
                        for column_index, label in enumerate(order)
                    },
                }
            )
        outputs = (
            (
                target / f"{stem}_counts.csv",
                count_rows,
                f"Top-rank {rank} confusion counts for {case_id}",
            ),
            (
                target / f"{stem}_row_normalized.csv",
                normalized_rows,
                f"Top-rank {rank} row-normalized confusion matrix for {case_id}",
            ),
        )
        for path, rows, description in outputs:
            _write_csv(path, rows)
            entries.append(
                _index_entry(
                    root,
                    path,
                    artifact_type="table_csv",
                    description=description,
                )
            )
            written += 1
    return entries, written


def _artifact_type(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    if relative == Path("study_plan.yaml"):
        return "study_plan"
    if relative == Path("study_manifest.json"):
        return "study_manifest"
    if relative == Path("study_run_result.json"):
        return "study_run_result"
    if relative == Path("progress_events.jsonl"):
        return "progress_log"
    if relative.parts and relative.parts[0] == "resolved_configs":
        return "resolved_config"
    if relative.parts and relative.parts[0] in {
        "cases",
        "raw",
        "fusion",
        "feature_vector",
        "feature_matrix",
    }:
        return "case_artifact"
    if relative.parts and relative.parts[0] == "tables":
        return "report_table"
    if relative.parts and relative.parts[0] == "figures":
        return "report_figure"
    if relative.name.startswith("STUDY_SUMMARY") or relative.name == "study_summary.json":
        return "study_summary"
    return "study_artifact"


def _complete_inventory(
    root: Path,
    generated_entries: Sequence[Mapping[str, Any]],
    *,
    output_index: Path,
) -> list[dict[str, Any]]:
    """Index every regular study artifact, not only generated report files."""

    generated = {
        str(row.get("path")): dict(row)
        for row in generated_entries
        if row.get("path")
    }
    inventory: list[dict[str, Any]] = []
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        if path == output_index or not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        known = generated.get(relative, {})
        inventory.append(
            {
                "path": relative,
                "type": known.get("type", _artifact_type(path, root)),
                "status": known.get("status", "available"),
                "description": known.get(
                    "description", "Study input, execution, case, or report artifact"
                ),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    inventory.append(
        {
            "path": output_index.name,
            "type": "output_index",
            "status": "available",
            "description": "Machine-readable complete inventory of this study folder",
            "bytes": None,
            "sha256": None,
            "self_hash_policy": "omitted_to_avoid_recursive_self_hash",
        }
    )
    return inventory


def generate_study_report(
    study_directory: str | Path,
    *,
    collected: CollectedStudy | None = None,
) -> ReportResult:
    """Collect, analyze, and report one existing study directory."""

    root = Path(study_directory).resolve()
    bundle = collected or collect_study(root)
    analysis = analyze_study(bundle)
    tables = root / "tables"
    figures_dir = root / "figures"
    tables.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    table_payloads: tuple[tuple[str, Sequence[Mapping[str, Any]], str], ...] = (
        ("case_summary", analysis.case_summary, "One descriptive row per case"),
        (
            "metric_distribution_summary",
            analysis.metric_distribution_summary,
            "Repeat mean/SD/t-CI95/min/max by case and predictive metric",
        ),
        ("varied_parameters", bundle.varied_parameters, "Declared variables and resolved case values"),
        ("controlled_parameters", bundle.controlled_parameters, "Complete non-variable resolved parameter list"),
        ("predictive_leaderboard", analysis.predictive_leaderboard, "BA-ranked manual review table"),
        ("deployment_measurements", analysis.deployment_table, "Operational measurements, separate from ranking"),
        ("repeat_metrics", analysis.repeat_metrics, "Participant OOF or labeled cell fallback per repeat"),
        ("fold_metrics", analysis.fold_metrics, "Per repeat/fold cell metrics"),
        (
            "per_class_metrics",
            analysis.per_class_metrics,
            "Pooled participant OOF or labeled cell-fallback class metrics",
        ),
        (
            "confusion_matrices",
            analysis.confusion_matrices,
            "Pooled participant OOF or labeled cell-fallback confusion matrices",
        ),
        (
            "confusion_counts",
            analysis.confusion_counts,
            "Long-form pooled confusion counts",
        ),
        (
            "confusion_row_normalized",
            analysis.confusion_row_normalized,
            "Long-form row-normalized pooled confusion matrices",
        ),
        ("calibration_bins", analysis.calibration_bins, "Top-label participant OOF reliability bins"),
        ("paired_deltas", analysis.paired_deltas, "Repeat-paired deltas versus declared reference"),
        ("coverage", analysis.coverage, "Coverage and quality diagnostic counts"),
        (
            "route_role_coverage",
            analysis.route_role_coverage,
            "Retained/direct/processed/unavailable/reducer-failure summaries by route and role",
        ),
        (
            "quality_distributions",
            analysis.quality_distributions,
            "Quality-component distributions by route and role",
        ),
        (
            "worst_class_f1_stability",
            analysis.worst_class_f1_stability,
            "Top-10 worst-class-F1 stability review",
        ),
        (
            "incomplete_cases",
            analysis.incomplete_cases,
            "Cases excluded from ranking because requested execution was incomplete",
        ),
        ("cell_metrics_raw", bundle.cell_rows, "Normalized raw cell metrics"),
        ("training_history_raw", bundle.history_rows, "Normalized training history"),
        ("quality_diagnostics_raw", bundle.quality_rows, "Normalized quality diagnostics"),
        ("case_records", bundle.case_records, "Case pass/fail/resume records"),
    )
    index: list[dict[str, Any]] = []
    table_file_count = 0
    for name, rows, description in table_payloads:
        csv_path = tables / f"{name}.csv"
        json_path = tables / f"{name}.json"
        _write_csv(csv_path, rows)
        _write_json(json_path, list(rows))
        status = "available" if rows else "N/A_no_rows"
        index.extend(
            (
                _index_entry(
                    root,
                    csv_path,
                    artifact_type="table_csv",
                    description=description,
                    status=status,
                ),
                _index_entry(
                    root,
                    json_path,
                    artifact_type="table_json",
                    description=description,
                    status=status,
                ),
            )
        )
        table_file_count += 2
    confusion_entries, confusion_file_count = _write_top_confusion_matrix_files(
        root,
        tables,
        analysis,
    )
    index.extend(confusion_entries)
    table_file_count += confusion_file_count
    write_figures = bool(bundle.plan.get("report", {}).get("write_static_figures", True))
    if not write_figures:
        clear_static_figure_artifacts(figures_dir)
    figures = (
        generate_static_figures(bundle, analysis, figures_dir)
        if write_figures
        else (
            {
                "figure": "all_static_figures",
                "status": "disabled",
                "path": "",
                "reason": "write_static_figures=false",
            },
        )
    )
    plot_status = figures_dir / "plot_status.json"
    _write_json(plot_status, list(figures))
    index.append(
        _index_entry(
            root,
            plot_status,
            artifact_type="figure_index",
            description="Generated/N/A status for every requested figure",
        )
    )
    for figure in figures:
        raw_path = figure.get("path")
        if raw_path and (root / str(raw_path)).is_file():
            index.append(
                _index_entry(
                    root,
                    root / str(raw_path),
                    artifact_type="figure" if figure["status"] == "generated" else "na_marker",
                    description=str(figure["figure"]),
                    status=str(figure["status"]),
                )
            )
    summary_payload = {
        "schema_version": "ppg_frailty.study_report.v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "study_directory": str(root),
        "plan": bundle.plan,
        "manifest": bundle.manifest,
        "varied_parameters": bundle.varied_parameters,
        "controlled_parameters": bundle.controlled_parameters,
        "analysis": asdict(analysis),
        "figure_status": list(figures),
    }
    summary_json = root / "study_summary.json"
    _write_json(summary_json, summary_payload)
    index.append(
        _index_entry(
            root,
            summary_json,
            artifact_type="summary_json",
            description="Machine-readable complete study summary",
        )
    )
    markdown_path = root / "STUDY_SUMMARY.md"
    markdown_path.write_text(
        _report_markdown(bundle, analysis, figures), encoding="utf-8"
    )
    index.append(
        _index_entry(
            root,
            markdown_path,
            artifact_type="summary_markdown",
            description="Primary human-readable study summary",
        )
    )
    write_html = bool(bundle.plan.get("report", {}).get("write_html", True))
    html_path = root / "STUDY_SUMMARY.html" if write_html else None
    if html_path is not None:
        html_path.write_text(
            _report_html(bundle, analysis, figures), encoding="utf-8"
        )
        index.append(
            _index_entry(
                root,
                html_path,
                artifact_type="summary_html",
                description="Portable HTML summary with figures",
            )
        )
    output_index = root / "outputs_index.json"
    complete_index = _complete_inventory(
        root,
        index,
        output_index=output_index,
    )
    _write_json(
        output_index,
        {
            "schema_version": "ppg_frailty.study_output_index.v2",
            "study_directory": str(root),
            "inventory_scope": "all_regular_files_below_study_directory",
            "artifacts": complete_index,
        },
    )
    return ReportResult(
        study_directory=root,
        summary_markdown=markdown_path,
        summary_html=html_path,
        output_index=output_index,
        table_count=table_file_count,
        generated_figure_count=sum(
            row.get("status") == "generated" for row in figures
        ),
        na_figure_count=sum(row.get("status") == "N/A" for row in figures),
    )


__all__ = ["ReportResult", "generate_study_report"]
