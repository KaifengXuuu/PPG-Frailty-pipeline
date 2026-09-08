"""Exact figure dispatcher; selected names are the complete output contract."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any, Mapping

from ppg_frailty.reporting.plots import STATIC_FIGURE_NAMES, generate_static_figures

from .contracts import AnalysisProducts, LoadedReportData, ReportContractError, ReportRequest


def _na(directory: Path, name: str, reason: str) -> dict[str, Any]:
    target = directory / f"{name}.NA.txt"
    temporary = directory / f".{name}.tmp-{os.getpid()}-{time.time_ns()}"
    temporary.write_text(f"N/A: {reason.strip()}\n", encoding="utf-8")
    os.replace(temporary, target)
    return {
        "figure": name,
        "status": "N/A",
        "path": str(target.relative_to(directory.parent)),
        "reason": reason.strip(),
    }


def _ensemble_member_metrics(products: AnalysisProducts, directory: Path) -> dict[str, Any]:
    rows = [row for row in products.tables["ensemble_member_metrics"] if row.get("status") == "available"]
    if not rows:
        return _na(directory, "ensemble_member_metrics", "no ensemble-member metrics")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except Exception as error:  # pragma: no cover - environment dependent.
        return _na(directory, "ensemble_member_metrics", f"matplotlib unavailable: {error}")
    figure = None
    temporary = directory / f".ensemble_member_metrics.tmp-{os.getpid()}-{time.time_ns()}.png"
    target = directory / "ensemble_member_metrics.png"
    try:
        case_ids = sorted({str(row["case_id"]) for row in rows})
        figure, axes = pyplot.subplots(1, 2, figsize=(11.0, 4.4), squeeze=False)
        for axis, metric, title in (
            (axes[0, 0], "balanced_accuracy", "Member balanced accuracy"),
            (axes[0, 1], "macro_f1", "Member macro F1"),
        ):
            values = [[float(row[metric]) for row in rows if str(row["case_id"]) == case_id] for case_id in case_ids]
            axis.boxplot(values, labels=case_ids, showmeans=True)
            axis.set_title(title)
            axis.set_ylim(0.0, 1.0)
            axis.tick_params(axis="x", rotation=25)
            axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(temporary, format="png", dpi=170, bbox_inches="tight")
        os.replace(temporary, target)
        return {
            "figure": "ensemble_member_metrics",
            "status": "generated",
            "path": str(target.relative_to(directory.parent)),
            "reason": "",
        }
    except Exception as error:  # noqa: BLE001 - preserve a usable report.
        return _na(directory, "ensemble_member_metrics", f"{type(error).__name__}: {error}")
    finally:
        if temporary.exists():
            temporary.unlink()
        if figure is not None:
            pyplot.close(figure)


def generate_selected_figures(
    data: LoadedReportData,
    products: AnalysisProducts,
    request: ReportRequest,
    names: tuple[str, ...],
    directory: Path,
) -> tuple[Mapping[str, Any], ...]:
    """Generate exactly ``names`` in caller order, never an implicit extra."""

    if not names:
        return ()
    directory.mkdir(parents=True, exist_ok=True)
    standard = tuple(name for name in names if name in STATIC_FIGURE_NAMES)
    statuses: dict[str, Mapping[str, Any]] = {}
    if standard:
        for row in generate_static_figures(
            data.collected,
            products.analysis,
            directory,
            modules=standard,
        ):
            figure = str(row.get("figure"))
            if figure not in standard:
                raise ReportContractError(f"figure dispatcher generated an unrequested output: {figure!r}")
            statuses[figure] = row
    if "ensemble_member_metrics" in names:
        statuses["ensemble_member_metrics"] = _ensemble_member_metrics(products, directory)
    for name in names:
        if name not in statuses:
            statuses[name] = _na(
                directory,
                name,
                "selected figure is not applicable to the collected study profile",
            )

    output: list[Mapping[str, Any]] = []
    for name in names:
        status = dict(statuses[name])
        if status.get("status") != "generated":
            if request.on_missing == "error":
                raise ReportContractError(f"selected figure {name!r} is unavailable: {status.get('reason')}")
            if request.on_missing == "skip":
                raw_path = status.get("path")
                if isinstance(raw_path, str):
                    path = directory.parent / raw_path
                    if path.is_file():
                        path.unlink()
                status["status"] = "skipped"
                status["path"] = ""
        output.append(status)
    return tuple(output)


__all__ = ["generate_selected_figures"]
