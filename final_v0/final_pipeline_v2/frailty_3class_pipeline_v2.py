#!/usr/bin/env python3
"""Run one complete, canonical V2 configuration and generate its report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent
SRC = PIPELINE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ppg_frailty.reporting import generate_study_report
from ppg_frailty.study import (
    ExecutionSpec,
    OutputSpec,
    ProgressEvent,
    ReportSpec,
    StudyInfo,
    StudyPlan,
    StudyRunner,
    TerminalProgressSink,
    validate_canonical_expansion,
)


def _indices(value: str) -> tuple[int, ...]:
    if value.strip().lower() == "all":
        return tuple(range(5))
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("use all or comma-separated indices 0..4") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one full V2 pipeline configuration through the canonical executor, "
            "then write CSV/JSON/Markdown/HTML tables and figures."
        )
    )
    parser.add_argument("--config", required=True, help="V2 pipeline YAML configuration")
    parser.add_argument("--study-id", default="manual_single_config_v2")
    parser.add_argument(
        "--purpose",
        default="Manual end-to-end verification of one selected V2 configuration.",
    )
    parser.add_argument(
        "--flow-position",
        default="Single-config evidence before manual final-use-case confirmation.",
    )
    parser.add_argument("--repeats", type=_indices, default=tuple(range(5)))
    parser.add_argument("--folds", type=_indices, default=tuple(range(5)))
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Case-level CPU parallelism; deep-model studies are reduced to 1 by default.",
    )
    parser.add_argument(
        "--measure-operational-costs",
        action="store_true",
        help="Measure per-outer-cell operational timing and memory costs.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/studies",
        help="Parent directory; a dated single-config child is created.",
    )
    parser.add_argument(
        "--resume",
        help="Existing study directory; passed cases are skipped and failed cases rerun.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the case without writing or training.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Run only; a report can be generated later with the sweep script.",
    )
    return parser


def plan_from_args(args: argparse.Namespace) -> StudyPlan:
    return StudyPlan(
        schema_version="ppg_frailty.study_plan.v2",
        study=StudyInfo(
            study_id=args.study_id,
            kind="single",
            purpose=args.purpose,
            flow_position=args.flow_position,
            decision_role="single_run",
        ),
        base_config=str(Path(args.config).resolve()),
        execution=ExecutionSpec(
            repeats=args.repeats,
            folds=args.folds,
            jobs=args.jobs,
            measure_operational_costs=args.measure_operational_costs,
        ),
        output=OutputSpec(root=args.output_root),
        report=ReportSpec(),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = plan_from_args(args)
    runner = StudyRunner(
        pipeline_root=PIPELINE_ROOT,
        progress_sink=TerminalProgressSink(),
    )
    if args.dry_run:
        expansion = validate_canonical_expansion(runner.expand(plan))
        print(
            json.dumps(
                {
                    "study": plan.to_dict(),
                    "base_config": str(expansion.base_config_path),
                    "cases": [case.to_dict() for case in expansion.cases],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    run_result = runner.run(
        plan,
        output_root=args.output_root,
        resume_directory=args.resume,
    )
    if not args.no_report:
        report_progress = TerminalProgressSink()
        report_progress(
            ProgressEvent(event="report_started", current=0, total=1)
        )
        report = generate_study_report(run_result.output_directory)
        report_progress(
            ProgressEvent(
                event="report_finished",
                current=1,
                total=1,
                message=str(report.summary_markdown),
            )
        )
        report_progress.close()
        print(f"Report: {report.summary_markdown}")
    print(f"Study output: {run_result.output_directory}")
    return 0 if run_result.status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
