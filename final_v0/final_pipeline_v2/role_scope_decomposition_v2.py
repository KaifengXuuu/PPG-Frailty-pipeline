#!/usr/bin/env python3
"""Validate or run a prediction-locked training/aggregation role-scope decomposition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PIPELINE_ROOT = Path(__file__).resolve().parent
SRC = PIPELINE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ppg_frailty.evaluate.role_scope_decomposition import (
    load_role_scope_plan,
    run_role_scope_decomposition,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate the decomposition YAML.")
    validate.add_argument("--plan", required=True)
    run = commands.add_parser("run", help="Run the prediction-locked decomposition.")
    run.add_argument("--plan", required=True)
    run.add_argument(
        "--output-root",
        default="artifacts/studies/static_line_b_staged_v2",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        plan = load_role_scope_plan(args.plan, pipeline_root=PIPELINE_ROOT)
        print(
            json.dumps(
                {
                    "status": "valid",
                    "study_id": plan.study_id,
                    "static_case_id": plan.static_source.case_id,
                    "all_role_case_id": plan.all_role_source.case_id,
                    "static_aggregation_roles": plan.static_aggregation_roles,
                    "all_aggregation_roles": plan.all_aggregation_roles,
                    "metrics": plan.metrics,
                },
                sort_keys=True,
            )
        )
        return 0
    output = run_role_scope_decomposition(
        args.plan,
        pipeline_root=PIPELINE_ROOT,
        output_root=args.output_root,
    )
    print(f"Role-scope decomposition output: {output}")
    print(f"Report: {output / 'STUDY_SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
