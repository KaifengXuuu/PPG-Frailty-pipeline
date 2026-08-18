#!/usr/bin/env python3
"""Configuration-driven V2 study runner with reports and resume."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml


PIPELINE_ROOT = Path(__file__).resolve().parent
SRC = PIPELINE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ppg_frailty.reporting import generate_study_report
from ppg_frailty.study import (
    AxisSpec,
    ExecutionSpec,
    OutputSpec,
    ProgressEvent,
    ReportSpec,
    StudyInfo,
    StudyPlan,
    StudyRunner,
    TerminalProgressSink,
    load_study_plan,
    validate_canonical_expansion,
)


def _indices(value: str) -> tuple[int, ...]:
    if value.strip().lower() == "all":
        return tuple(range(5))
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("use all or comma-separated indices 0..4") from error


def _yaml_value(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError as error:
        raise argparse.ArgumentTypeError(f"invalid YAML value: {value}") from error


def _assignment(value: str, *, require_list: bool) -> tuple[str, Any]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected DOTTED.PATH=YAML_VALUE")
    path, raw = value.split("=", 1)
    path = path.strip()
    if not path:
        raise argparse.ArgumentTypeError("configuration path cannot be empty")
    parsed = _yaml_value(raw)
    if require_list and (not isinstance(parsed, list) or len(parsed) < 2):
        raise argparse.ArgumentTypeError("grid values must be a YAML list of 2+ values")
    return path, parsed


def _vary(value: str) -> tuple[str, list[Any]]:
    path, parsed = _assignment(value, require_list=True)
    return path, list(parsed)


def _reference(value: str) -> tuple[str, Any]:
    return _assignment(value, require_list=False)


def _add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repeats", type=_indices, help="all or comma-separated 0..4")
    parser.add_argument("--folds", type=_indices, help="all or comma-separated 0..4")
    parser.add_argument(
        "--jobs",
        type=int,
        help="Case-level CPU parallelism; deep cases remain jobs=1 by default.",
    )
    parser.add_argument(
        "--output-root",
        help="Parent directory for a new dated, study-specific folder.",
    )
    parser.add_argument(
        "--resume",
        help="Existing study directory; passed cases skip and failed cases rerun.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve/print only.")
    parser.add_argument("--no-report", action="store_true", help="Skip report generation.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run canonical V2 single-config, ablation, Cartesian-grid, or explicit "
            "catalog-screening plans. Each case uses one resolved pipeline YAML; "
            "parallelism is only across cases."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run a YAML study plan.")
    run.add_argument("--plan", required=True, help="Study-plan YAML")
    _add_execution_arguments(run)

    ablation = subparsers.add_parser(
        "ablation", help="Build and run a single-factor ablation."
    )
    ablation.add_argument("--base-config", required=True)
    ablation.add_argument("--factor", required=True, help="Dotted config path")
    ablation.add_argument(
        "--values", required=True, nargs="+", type=_yaml_value, help="YAML scalar values"
    )
    ablation.add_argument(
        "--reference-value", required=True, type=_yaml_value
    )
    ablation.add_argument("--study-id", required=True)
    ablation.add_argument("--purpose", required=True)
    ablation.add_argument("--flow-position", required=True)
    ablation.add_argument("--thesis-section", action="append", default=[])
    _add_execution_arguments(ablation)

    grid = subparsers.add_parser("grid", help="Build and run a Cartesian grid.")
    grid.add_argument("--base-config", required=True)
    grid.add_argument(
        "--vary",
        required=True,
        action="append",
        type=_vary,
        metavar="PATH=[VALUES]",
        help="Repeat for each dotted-path axis.",
    )
    grid.add_argument(
        "--reference",
        action="append",
        type=_reference,
        default=[],
        metavar="PATH=VALUE",
        help="Repeat for every axis to enable paired deltas.",
    )
    grid.add_argument("--study-id", required=True)
    grid.add_argument("--purpose", required=True)
    grid.add_argument("--flow-position", required=True)
    grid.add_argument("--thesis-section", action="append", default=[])
    _add_execution_arguments(grid)

    report = subparsers.add_parser(
        "report", help="Regenerate reports from an existing study folder."
    )
    report.add_argument("--study-dir", required=True)
    return parser


def _execution(args: argparse.Namespace, base: ExecutionSpec | None = None) -> ExecutionSpec:
    source = base or ExecutionSpec()
    return replace(
        source,
        repeats=args.repeats if args.repeats is not None else source.repeats,
        folds=args.folds if args.folds is not None else source.folds,
        jobs=args.jobs if args.jobs is not None else source.jobs,
    )


def _run_plan(args: argparse.Namespace) -> StudyPlan:
    plan = load_study_plan(args.plan)
    plan = replace(plan, execution=_execution(args, plan.execution))
    if args.output_root is not None:
        plan = replace(plan, output=OutputSpec(root=args.output_root))
    return plan


def _ablation_plan(args: argparse.Namespace) -> StudyPlan:
    return StudyPlan(
        schema_version="ppg_frailty.study_plan.v2",
        study=StudyInfo(
            study_id=args.study_id,
            kind="ablation",
            purpose=args.purpose,
            flow_position=args.flow_position,
            decision_role="ablation",
            thesis_sections=tuple(args.thesis_section),
        ),
        base_config=str(Path(args.base_config).resolve()),
        axes=(
            AxisSpec(
                path=args.factor,
                values=tuple(args.values),
                reference=args.reference_value,
            ),
        ),
        execution=_execution(args),
        output=OutputSpec(root=args.output_root or "artifacts/studies"),
        report=ReportSpec(),
    )


def _grid_plan(args: argparse.Namespace) -> StudyPlan:
    references = dict(args.reference)
    paths = [path for path, _ in args.vary]
    unknown = sorted(set(references) - set(paths))
    if unknown:
        raise ValueError(f"reference paths are not declared axes: {unknown}")
    if references and set(references) != set(paths):
        missing = sorted(set(paths) - set(references))
        raise ValueError(f"paired reference requires a value for every axis: {missing}")
    return StudyPlan(
        schema_version="ppg_frailty.study_plan.v2",
        study=StudyInfo(
            study_id=args.study_id,
            kind="grid",
            purpose=args.purpose,
            flow_position=args.flow_position,
            decision_role="screening",
            thesis_sections=tuple(args.thesis_section),
        ),
        base_config=str(Path(args.base_config).resolve()),
        axes=tuple(
            AxisSpec(path=path, values=tuple(values), reference=references.get(path))
            for path, values in args.vary
        ),
        execution=_execution(args),
        output=OutputSpec(root=args.output_root or "artifacts/studies"),
        report=ReportSpec(),
    )


def _print_expansion(runner: StudyRunner, plan: StudyPlan) -> None:
    expansion = validate_canonical_expansion(runner.expand(plan))
    print(
        json.dumps(
            {
                "study": plan.to_dict(),
                "base_config": str(expansion.base_config_path),
                "reference_case_id": expansion.reference_case_id,
                "cases": [case.to_dict() for case in expansion.cases],
                "varied_parameters": list(expansion.varied_parameters),
                "controlled_parameter_count": len(expansion.controlled_parameters),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _generate_report_with_progress(
    study_dir: str | Path,
    sink: TerminalProgressSink,
) -> None:
    sink(ProgressEvent(event="report_started", current=0, total=1))
    report = generate_study_report(study_dir)
    sink(
        ProgressEvent(
            event="report_finished",
            current=1,
            total=1,
            message=str(report.summary_markdown),
        )
    )
    print(f"Report: {report.summary_markdown}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sink = TerminalProgressSink()
    sink(
        ProgressEvent(
            event="program_started",
            message="loading command and study plan",
        )
    )
    try:
        if args.command == "report":
            _generate_report_with_progress(args.study_dir, sink)
            return 0
        plan = (
            _run_plan(args)
            if args.command == "run"
            else _ablation_plan(args)
            if args.command == "ablation"
            else _grid_plan(args)
        )
        sink(ProgressEvent(event="plan_loaded", message="study plan loaded"))
        runner = StudyRunner(
            pipeline_root=PIPELINE_ROOT,
            progress_sink=sink,
        )
        if args.dry_run:
            _print_expansion(runner, plan)
            return 0
        run_result = runner.run(
            plan,
            output_root=args.output_root,
            resume_directory=args.resume,
        )
        if not args.no_report:
            _generate_report_with_progress(run_result.output_directory, sink)
        print(f"Study output: {run_result.output_directory}")
        return 0 if run_result.status == "passed" else 2
    finally:
        sink.close()


if __name__ == "__main__":
    raise SystemExit(main())
