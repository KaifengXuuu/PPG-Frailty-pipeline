"""Artifact-only analysis of historical sweeps that predate OOF Parquet."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable, Mapping, Sequence

from .tabular import ReportTable, write_csv, write_excel_workbook

METRICS = (
    "subject_balanced_accuracy",
    "subject_macro_f1",
    "file_balanced_accuracy",
    "file_macro_f1",
    "window_balanced_accuracy",
    "window_macro_f1",
)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _prepare_output_directory(path: str | Path) -> Path:
    target = Path(path).resolve()
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FileExistsError(f"historical output must be a new or empty directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _walk(value: Any, context: Mapping[str, Any] | None = None) -> Iterable[dict[str, Any]]:
    """Yield mapping leaves while carrying scalar identifiers from ancestors."""
    if isinstance(value, Mapping):
        scalars = {str(k): v for k, v in value.items() if not isinstance(v, (Mapping, list, tuple))}
        inherited = {**(context or {}), **scalars}
        if any(_number(value.get(metric)) is not None for metric in METRICS):
            yield inherited
        for child in value.values():
            if isinstance(child, (Mapping, list, tuple)):
                yield from _walk(child, inherited)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk(child, context)


def _source_rows(source: Path, source_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inventory, rows = [], []
    for path in sorted(source.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        extracted = list(_walk(payload))
        inventory.append({
            "source_id": source_id,
            "path": path.relative_to(source).as_posix(),
            "bytes": path.stat().st_size,
            "metric_row_count": len(extracted),
        })
        for index, row in enumerate(extracted):
            rows.append({
                "source_id": source_id,
                "source_file": path.relative_to(source).as_posix(),
                "source_row": index,
                **row
            })
    return inventory, rows


def _configuration(row: Mapping[str, Any]) -> str:
    return str(row.get("config_id", row.get("case_id", row.get("model", row.get("resolved_model", "unknown")))))


def _summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        for metric in METRICS:
            if (value := _number(row.get(metric))) is not None:
                grouped.setdefault((str(row["source_id"]), _configuration(row), metric), []).append(value)
    output = []
    for (source, config, metric), values in sorted(grouped.items()):
        mean = fmean(values)
        spread = stdev(values) if len(values) > 1 else 0.0
        output.append({
            "source_id": source,
            "configuration": config,
            "metric": metric,
            "n_repeats": len(values),
            "mean": mean,
            "sample_sd": spread,
            "minimum": min(values),
            "maximum": max(values),
            "participant_cluster_ci95": None,
            "participant_cluster_ci_status": "N/A_no_participant_OOF_in_historical_archive",
        })
    return output


def _pairwise(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Descriptive matched-repeat differences; no P value is invented."""
    groups: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in rows:
        for metric in METRICS:
            if (value := _number(row.get(metric))) is not None:
                groups.setdefault((str(row["source_id"]), metric), {}).setdefault(_configuration(row), []).append(value)
    output = []
    for (source, metric), configurations in sorted(groups.items()):
        names = sorted(configurations)
        if len(names) < 2:
            continue
        reference = names[0]
        for candidate in names[1:]:
            left, right = configurations[reference], configurations[candidate]
            count = min(len(left), len(right))
            deltas = [right[i] - left[i] for i in range(count)]
            output.append({
                "source_id": source,
                "metric": metric,
                "reference": reference,
                "candidate": candidate,
                "matched_repeat_count": count,
                "candidate_minus_reference_mean": fmean(deltas) if deltas else None,
                "sample_sd": stdev(deltas) if len(deltas) > 1 else 0.0 if deltas else None,
                "p_value": None,
                "inference_status": "N/A_historical_descriptive_only",
            })
    return output


def _plot(summary: Sequence[Mapping[str, Any]], target: Path) -> Path | None:
    if not summary:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except ImportError:
        return None
    selected = [row for row in summary if row["metric"] == "subject_balanced_accuracy"] or list(summary)
    figure, axis = pyplot.subplots(figsize=(max(7, len(selected) * 0.45), 4.8))
    labels = [f"{row['source_id']}:{row['configuration']}" for row in selected]
    axis.bar(labels, [row["mean"] for row in selected], yerr=[row["sample_sd"] for row in selected])
    axis.set_ylabel(str(selected[0]["metric"]))
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(target, dpi=170, bbox_inches="tight")
    pyplot.close(figure)
    return target


def run_historical_analysis(sources: Sequence[Path], output_dir: str | Path) -> Path:
    missing = [str(path) for path in sources if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"historical source directories are missing: {missing}")
    target = _prepare_output_directory(output_dir)
    tables = target / "tables"
    figures = target / "figures"
    tables.mkdir()
    figures.mkdir()
    inventory, raw = [], []
    for index, source in enumerate(sources, 1):
        found, rows = _source_rows(source, f"{index:02d}_{source.name}")
        inventory += found
        raw += rows
    summary, pairwise = _summaries(raw), _pairwise(raw)
    products = {
        "source_inventory":
        inventory,
        "historical_metric_rows":
        raw,
        "historical_metric_summary":
        summary,
        "pairwise_repeat_metric_deltas":
        pairwise,
        "missing_statistics": [{
            "statistic": "participant-cluster CI / ROC-AUC / paired participant inference",
            "status": "N/A",
            "reason": "participant-level probabilities were not archived",
        }],
    }
    for name, rows in products.items():
        write_csv(tables / f"{name}.csv", rows)
        (tables / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str) + "\n",
                                             encoding="utf-8")
    write_excel_workbook(tables / "historical_report.xlsx",
                         [ReportTable(name, rows, compact=False) for name, rows in products.items()])
    plot = _plot(summary, figures / "historical_performance.png")
    lines = [
        "# Historical sweep analysis",
        "",
        "Historical aggregate artifacts are analyzed descriptively; unavailable participant-level inference remains explicit.",
        "",
        f"- Sources: {len(sources)}",
        f"- Metric rows: {len(raw)}",
        f"- Summaries: {len(summary)}",
        f"- Figure: {plot.relative_to(target).as_posix() if plot else 'N/A'}",
    ]
    (target / "STUDY_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (target / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "ppg_frailty.historical_analysis.v2",
                "sources": [str(path) for path in sources],
                "tables": list(products),
                "figure": None if plot is None else plot.relative_to(target).as_posix(),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return target


def run_historical_major_report(
    *,
    early_source: str | Path,
    shapeformer_source: str | Path,
    fixed_epoch_source: str | Path,
    extension_source: str | Path,
    output_dir: str | Path,
    window_seconds: float = 5.0,
    overlap_percent: float = 50.0,
    patience: int = 20,
    extra_input: str = "0",
) -> Path:
    return run_historical_analysis(
        tuple(
            Path(value).resolve()
            for value in (early_source, shapeformer_source, fixed_epoch_source, extension_source)),
        output_dir,
    )


__all__ = ["run_historical_analysis", "run_historical_major_report"]
