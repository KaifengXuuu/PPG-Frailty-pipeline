#!/usr/bin/env python3
"""Validate or run the leakage-explicit Stage 0 decision-bias oracle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent
SRC = PIPELINE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ppg_frailty.evaluate.decision_bias_oracle import (
    load_decision_bias_oracle_plan,
    run_decision_bias_oracle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate the Stage 0 YAML.")
    validate.add_argument("--plan", required=True)
    run = commands.add_parser("run", help="Run Stage 0 on completed participant OOF.")
    run.add_argument("--plan", required=True)
    run.add_argument("--study-dir", help="Override source.study_dir.")
    run.add_argument("--case-id", help="Override source.case_id.")
    run.add_argument("--prediction-file", help="Override the completed root OOF parquet.")
    run.add_argument("--step", type=float, help="Override the simplex grid step.")
    run.add_argument(
        "--output-root",
        default="artifacts/studies/static_line_b_staged_v2",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        plan = load_decision_bias_oracle_plan(args.plan, pipeline_root=PIPELINE_ROOT)
        print(
            json.dumps(
                {
                    "status": "valid",
                    "study_id": plan.study_id,
                    "source_study_dir": str(plan.source_study_dir),
                    "case_id": plan.case_id,
                    "expected_participants": plan.expected_participants,
                    "expected_repeats": list(plan.expected_repeats),
                    "bias_step": plan.bias_step,
                },
                sort_keys=True,
            )
        )
        return 0
    output = run_decision_bias_oracle(
        args.plan,
        pipeline_root=PIPELINE_ROOT,
        output_root=args.output_root,
        source_study_dir=args.study_dir,
        case_id=args.case_id,
        prediction_file=args.prediction_file,
        step=args.step,
    )
    print(f"Stage 0 output: {output}")
    print(f"Report: {output / 'STUDY_SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
