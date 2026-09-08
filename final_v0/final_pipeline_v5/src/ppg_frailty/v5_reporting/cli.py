"""Command-line entry point for listing, validating and running V5 reports."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from .analysis import build_analysis
from .collect import load_report_data
from .contracts import REPORT_MODES, ReportContractError, ReportRequest, RunSpec
from .registry import (
    KNOWN_FIGURES,
    KNOWN_TABLES,
    MODULES,
    PRESETS,
    resolve_selection,
)
from .validate import validate_report_data
from .writer import export_report_excel, resolve_output_path, write_report
from ppg_frailty.v5.output_contract import (
    PIPELINE_OUTPUT_ROOT,
    REPORT_OUTPUT_ROOT,
    safe_output_name,
)
from ppg_frailty.v5.specialized import (
    rebuild_specialized_report,
    run_specialized_analysis,
    validate_specialized_plan,
)


def _tokens(values: Iterable[str] | None, *, optional: bool = False) -> tuple[str, ...] | None:
    if values is None:
        return None if optional else ()
    output = tuple(token.strip() for value in values for token in value.split(",") if token.strip())
    lowered = {value.lower() for value in output}
    if "none" in lowered:
        if len(output) != 1:
            raise ReportContractError("'none' cannot be combined with other selections")
        return ()
    return tuple(dict.fromkeys(output))


def _run_spec(value: str) -> RunSpec:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--run must be NAME=PATH")
    try:
        return RunSpec(name.strip(), Path(raw_path.strip()))
    except ReportContractError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _add_request_arguments(parser: argparse.ArgumentParser, *, output: bool) -> None:
    parser.add_argument("--mode", choices=sorted(REPORT_MODES), default="single")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        type=Path,
        help=("One pipeline_output run. Its directory name becomes the default "
              "report_output child name."),
    )
    source.add_argument(
        "--run",
        action="append",
        type=_run_spec,
        metavar="NAME=PATH",
        help="Advanced named input; repeat for a multi-run comparison.",
    )
    parser.add_argument("--include-case", action="append", default=[])
    parser.add_argument("--exclude-case", action="append", default=[])
    parser.add_argument("--reference-case")
    parser.add_argument("--comparison-family", default="declared_comparison")
    parser.add_argument("--factor-path", action="append", default=[])
    parser.add_argument("--preset", "--presets", action="append", default=[])
    parser.add_argument("--module", "--modules", action="append", default=[])
    parser.add_argument(
        "--figure",
        "--figures",
        action="append",
        default=None,
        help="Exact comma-separated output list; use 'none' for no figures.",
    )
    parser.add_argument(
        "--table",
        "--tables",
        action="append",
        default=None,
        help="Exact comma-separated output list; use 'none' for no tables.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--permutation-resamples", type=int, default=100_000)
    parser.add_argument("--statistics-seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--validation-depth", choices=("selected", "full"), default="full")
    parser.add_argument("--on-missing", choices=("error", "na", "skip"), default="na")
    parser.add_argument(
        "--v2-compatibility",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    if output:
        destination = parser.add_mutually_exclusive_group()
        destination.add_argument(
            "--output-name",
            help="Optional new child name below report_output.",
        )
        destination.add_argument(
            "--output-dir",
            type=Path,
            help="Exact new path below report_output (advanced compatibility form).",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyse_report.py",
        description="Composable, artifact-only V5 analysis and report runner.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("list", help="List registered presets/modules/outputs.")
    validate = subcommands.add_parser("validate", help="Validate inputs without writing.")
    _add_request_arguments(validate, output=False)
    run = subcommands.add_parser("run", help="Validate, analyse, then atomically write.")
    _add_request_arguments(run, output=True)
    excel = subcommands.add_parser(
        "export-excel",
        help="Regenerate report_tables.xlsx from an existing report's CSV tables.",
    )
    excel.add_argument("--report-output", required=True)
    excel.add_argument("--replace", action="store_true")
    specialized_validate = subcommands.add_parser(
        "specialized-validate",
        help="Validate any preserved non-canonical study YAML.",
    )
    specialized_validate.add_argument("--plan", required=True)
    specialized_validate.add_argument("--source-root", default=str(PIPELINE_OUTPUT_ROOT.parent))
    specialized_run = subcommands.add_parser(
        "specialized-run",
        help="Run a preserved artifact-only oracle or role-scope analysis.",
    )
    specialized_run.add_argument("--plan", required=True)
    specialized_run.add_argument("--source-root", default=str(PIPELINE_OUTPUT_ROOT.parent))
    specialized_run.add_argument("--output-name")
    specialized_run.add_argument("--study-dir")
    specialized_run.add_argument("--case-id")
    specialized_run.add_argument("--prediction-file")
    specialized_run.add_argument("--step", type=float)
    specialized_report = subcommands.add_parser(
        "specialized-report",
        help="Rebuild a Stage5/static-peak/hyperparameter report from pipeline_output.",
    )
    specialized_report.add_argument("--input", required=True)
    specialized_report.add_argument("--output-name")
    return parser


def _request(namespace: argparse.Namespace) -> ReportRequest:
    if namespace.input is not None:
        input_path = namespace.input.expanduser().resolve()
        runs = (RunSpec(input_path.name, input_path), )
    else:
        runs = tuple(namespace.run or ())
    output_dir = getattr(namespace, "output_dir", None)
    output_name = getattr(namespace, "output_name", None)
    if output_name is not None:
        output_dir = REPORT_OUTPUT_ROOT / safe_output_name(output_name, label="report output name")
    elif output_dir is None and hasattr(namespace, "output_dir"):
        canonical_names: set[str] = set()
        canonical = True
        for run in runs:
            try:
                relative = run.path.resolve().relative_to(PIPELINE_OUTPUT_ROOT.resolve())
            except ValueError:
                canonical = False
                break
            if not relative.parts:
                canonical = False
                break
            run_root = PIPELINE_OUTPUT_ROOT / relative.parts[0]
            if not (run_root / "study_manifest.json").is_file():
                canonical = False
                break
            canonical_names.add(relative.parts[0])
        if not canonical or len(canonical_names) != 1:
            raise ReportContractError("automatic report naming requires all inputs to belong to the same "
                                      "pipeline_output/<run>; use --output-name/--output-dir for legacy "
                                      "or unrelated inputs")
        output_dir = REPORT_OUTPUT_ROOT / next(iter(canonical_names))
    return ReportRequest(
        mode=namespace.mode,
        runs=runs,
        output_dir=output_dir,
        include_cases=tuple(_tokens(namespace.include_case) or ()),
        exclude_cases=tuple(_tokens(namespace.exclude_case) or ()),
        reference_case=namespace.reference_case,
        comparison_family=namespace.comparison_family,
        factor_paths=tuple(_tokens(namespace.factor_path) or ()),
        presets=tuple(_tokens(namespace.preset) or ()),
        modules=tuple(_tokens(namespace.module) or ()),
        figures=_tokens(namespace.figure, optional=True),
        tables=_tokens(namespace.table, optional=True),
        bootstrap_resamples=namespace.bootstrap_resamples,
        permutation_resamples=namespace.permutation_resamples,
        statistics_seed=namespace.statistics_seed,
        alpha=namespace.alpha,
        calibration_bins=namespace.calibration_bins,
        validation_depth=namespace.validation_depth,
        on_missing=namespace.on_missing,
        allow_v2_compatibility=namespace.v2_compatibility,
    )


def _print(value: Any, *, stream: Any | None = None) -> None:
    stream = sys.stdout if stream is None else stream
    print(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        file=stream,
    )


def _listing() -> Mapping[str, Any]:
    return {
        "modes": sorted(REPORT_MODES),
        "presets": {
            key: (list(value) if key != "full" else ["<all mode-compatible modules>"])
            for key, value in sorted(PRESETS.items())
        },
        "modules": [{
            **asdict(module),
            "modes": sorted(module.modes),
        } for module in MODULES],
        "figures": sorted(KNOWN_FIGURES),
        "tables": sorted(KNOWN_TABLES),
        "selection_rule": ("explicit --figures/--tables replace defaults exactly; 'none' selects zero"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    namespace = parser.parse_args(argv)
    if namespace.command == "list":
        _print(_listing())
        return 0
    try:
        if namespace.command == "specialized-validate":
            _print(validate_specialized_plan(
                namespace.plan,
                source_root=namespace.source_root,
            ))
            return 0
        if namespace.command == "specialized-run":
            output = run_specialized_analysis(
                namespace.plan,
                source_root=namespace.source_root,
                output_name=namespace.output_name,
                source_study_dir=namespace.study_dir,
                case_id=namespace.case_id,
                prediction_file=namespace.prediction_file,
                step=namespace.step,
            )
            _print({"status": "complete", "output_dir": str(output)})
            return 0
        if namespace.command == "specialized-report":
            output = rebuild_specialized_report(
                namespace.input,
                output_name=namespace.output_name,
            )
            _print({"status": "complete", "output_dir": str(output)})
            return 0
        if namespace.command == "export-excel":
            status = export_report_excel(
                namespace.report_output,
                replace=bool(namespace.replace),
            )
            _print(status)
            return 0
        request = _request(namespace)
        selection = resolve_selection(
            mode=request.mode,
            presets=request.presets,
            modules=request.modules,
            figures=request.figures,
            tables=request.tables,
        )
        if namespace.command == "run":
            # Reject an unsafe/existing path before performing expensive analysis.
            resolve_output_path(request.output_dir)  # type: ignore[arg-type]
        data = load_report_data(request)
        validation = validate_report_data(data, request)
        if namespace.command == "validate":
            _print({
                "status": validation.status,
                "selection": asdict(selection),
                "validation": asdict(validation),
                "source_kind": data.source_kind,
            })
            return 0
        products = build_analysis(data, request, selection)
        output = write_report(data, products, request, selection, validation)
        _print({
            "status": "complete",
            "output_dir": str(output),
            "selection": asdict(selection),
        })
        return 0
    except (ReportContractError, FileNotFoundError, OSError, TypeError, ValueError) as error:
        _print(
            {
                "status": "error",
                "error": f"{type(error).__name__}: {error}"
            },
            stream=sys.stderr,
        )
        return 2


__all__ = ["build_parser", "main"]
