"""Compatibility facade for the unified V5 report engine.

The V2 module mixed statistics, table declarations, HTML templates, plotting
and publication in one 4,000-line file. Numerical work now stays in the shared
analysis modules; this file only preserves ``generate_study_report`` for old
callers. New workflows use ``analyse_report.py`` and never write into pipeline
outputs.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .collect import CollectedStudy, collect_study
from .tabular import compact_rows, write_csv, write_excel_workbook_from_csv_directory


@dataclass(frozen=True)
class ReportResult:
    study_directory: Path
    summary_markdown: Path
    summary_html: Path | None
    output_index: Path
    table_count: int
    generated_figure_count: int
    na_figure_count: int


def _json(path: Path, value: Any) -> None:
    from ppg_frailty.v5_reporting.writer import _jsonable

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _mode_and_factors(collected: CollectedStudy) -> tuple[str, str | None, tuple[str, ...]]:
    manifest = collected.manifest
    raw_reference = manifest.get("reference_case_id", manifest.get("reference_case"))
    reference = None if raw_reference in (None, "") else str(raw_reference)
    axes = collected.plan.get("axes", ())
    factors = tuple(str(row["path"]) for row in axes if isinstance(row, Mapping) and row.get("path") not in (None, ""))
    role = str(collected.plan.get("study", {}).get("decision_role", "")).lower()
    if reference and factors and "ablation" in role:
        return "ablation", reference, factors
    return ("comparison" if reference else "single"), reference, factors


def _loaded(collected: CollectedStudy) -> Any:
    from ppg_frailty.v5_reporting.contracts import LoadedReportData

    case_ids = tuple(
        str(row["case_id"]) for row in collected.manifest.get("cases", ())
        if isinstance(row, Mapping) and row.get("case_id") not in (None, ""))
    return LoadedReportData(
        collected=collected,
        layer_rows={
            "window": tuple(collected.window_oof_rows),
            "file": tuple(collected.file_oof_rows),
            "role": tuple(collected.role_oof_rows),
            "participant": tuple(collected.subject_oof_rows),
            "member": (),
        },
        artifact_records=(),
        evaluation_scope_by_case={case_id: "outer_oof"
                                  for case_id in case_ids},
        source_root_by_case={case_id: collected.root
                             for case_id in case_ids},
        source_kind="v2_study_compatibility",
    )


def _summary(
    directory: Path,
    table_names: Sequence[str],
    figures: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    title = f"PPG Frailty analysis — {directory.name}"
    tables = "\n".join(f"- [{name}.csv](tables/{name}.csv)" for name in table_names)
    plots = "\n".join(f"- `{row.get('figure')}`: {row.get('status')}" for row in figures)
    markdown = directory / "STUDY_SUMMARY.md"
    markdown.write_text(
        f"# {title}\n\n## Tables\n\n{tables}\n\n## Figures\n\n{plots}\n",
        encoding="utf-8",
    )
    html = directory / "STUDY_SUMMARY.html"
    html.write_text(
        "<!doctype html><meta charset='utf-8'><title>" + title + "</title><h1>" + title +
        "</h1><p>Tables and figures are indexed in outputs_index.json.</p>",
        encoding="utf-8",
    )
    return markdown, html


def generate_study_report(
    study_directory: str | Path,
    *,
    collected: CollectedStudy | None = None,
) -> ReportResult:
    """Deprecated V2 adapter that renders into ``study_directory`` in place.

    The supported V5 entry point is ``analyse_report.py run``; it treats the
    pipeline directory as read-only and publishes below ``report_output``.
    """

    warnings.warn(
        "reporting.generate_study_report() is an in-place V2 compatibility "
        "adapter; use analyse_report.py run for read-only pipeline inputs",
        DeprecationWarning,
        stacklevel=2,
    )

    from ppg_frailty.v5_reporting.analysis import build_analysis
    from ppg_frailty.v5_reporting.contracts import ReportRequest, RunSpec
    from ppg_frailty.v5_reporting.plots import generate_selected_figures
    from ppg_frailty.v5_reporting.registry import resolve_selection
    from ppg_frailty.v5_reporting.writer import _write_outputs_index

    root = Path(study_directory).resolve()
    bundle = collected or collect_study(root)
    mode, reference, factors = _mode_and_factors(bundle)
    request = ReportRequest(
        mode=mode,
        runs=(RunSpec(root.name, root), ),
        reference_case=reference,
        factor_paths=factors,
        presets=("full", ),
        on_missing="na",
    )
    selection = resolve_selection(
        mode=mode,
        presets=request.presets,
        modules=(),
        figures=None,
        tables=None,
    )
    data = _loaded(bundle)
    products = build_analysis(data, request, selection)
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for name in selection.tables:
        rows = tuple(products.tables.get(name, ()))
        write_csv(tables / f"{name}.csv", compact_rows(rows))
        _json(tables / f"{name}.json", rows)
    write_excel_workbook_from_csv_directory(tables / "report_tables.xlsx", tables)
    figures = generate_selected_figures(data, products, request, selection.figures, root / "figures")
    _json(root / "study_summary.json", {"analysis": asdict(products.analysis)})
    _json(root / "figure_status.json", figures)
    markdown, html = _summary(root, selection.tables, figures)
    _write_outputs_index(root)
    return ReportResult(
        study_directory=root,
        summary_markdown=markdown,
        summary_html=html,
        output_index=root / "outputs_index.json",
        table_count=len(selection.tables) * 2 + 1,
        generated_figure_count=sum(row.get("status") == "generated" for row in figures),
        na_figure_count=sum(row.get("status") == "N/A" for row in figures),
    )


__all__ = ["ReportResult", "generate_study_report"]
