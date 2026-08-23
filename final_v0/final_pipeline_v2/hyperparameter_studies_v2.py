#!/usr/bin/env python3
"""Run auditable staged hyperparameter and channel studies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent
SRC = PIPELINE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ppg_frailty.study.hyperparameter import (
    complete_successive_halving_study,
    inspect_successive_halving_completion,
    load_hyperparameter_plan,
    regenerate_hyperparameter_report,
    run_hyperparameter_study,
)
from ppg_frailty.study import TerminalProgressSink


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run successive-halving, dependent regularization, or channel "
            "ablation plans without treating tuning evidence as a final test."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate one plan YAML.")
    validate.add_argument("--plan", required=True)
    run = commands.add_parser("run", help="Run one plan YAML.")
    run.add_argument("--plan", required=True)
    run.add_argument("--upstream-study")
    run.add_argument("--output-root")
    run.add_argument("--device")
    run.add_argument("--jobs", type=int)
    complete = commands.add_parser(
        "complete",
        help="Full-CV only the unpromoted cases of a completed halving study.",
    )
    complete.add_argument("--study-dir", required=True)
    complete.add_argument("--device")
    complete.add_argument("--jobs", type=int)
    complete.add_argument(
        "--dry-run", action="store_true", help="Print the missing work only."
    )
    report = commands.add_parser("report", help="Regenerate nested and root reports.")
    report.add_argument("--study-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        plan = load_hyperparameter_plan(args.plan)
        print(json.dumps({
            "status": "valid",
            "study_id": plan["study"]["study_id"],
            "study_type": plan["study"]["study_type"],
            "candidate_count": len(plan["candidates"]),
        }, sort_keys=True))
        return 0
    if args.command == "report":
        result = regenerate_hyperparameter_report(args.study_dir)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "complete" and args.dry_run:
        print(json.dumps(
            inspect_successive_halving_completion(args.study_dir), sort_keys=True
        ))
        return 0
    progress = TerminalProgressSink()
    try:
        if args.command == "complete":
            output = complete_successive_halving_study(
                args.study_dir,
                pipeline_root=PIPELINE_ROOT,
                device=args.device,
                jobs=args.jobs,
                progress_sink=progress,
            )
        else:
            output = run_hyperparameter_study(
                args.plan,
                pipeline_root=PIPELINE_ROOT,
                upstream_study=args.upstream_study,
                output_root=args.output_root,
                device=args.device,
                jobs=args.jobs,
                progress_sink=progress,
            )
    finally:
        progress.close()
    print(f"Study output: {output}")
    print(f"Report: {output / 'STUDY_SUMMARY.md'}")
    print(f"Selected configuration: {output / 'selected_configuration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
