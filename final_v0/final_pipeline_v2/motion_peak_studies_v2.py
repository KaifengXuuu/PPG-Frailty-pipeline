#!/usr/bin/env python3
"""Run/report the isolated Stage5-pre and Stage-ablation-01 studies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent
SRC = PIPELINE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ppg_frailty.quality.stage5_pre import (
    generate_motion_peak_report,
    load_motion_peak_plan,
    run_motion_peak_study,
)
from ppg_frailty.study import TerminalProgressSink


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage5-pre motion/PTT or Stage-ablation-01 static peaks."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate one study YAML.")
    validate.add_argument("--plan", required=True)
    run = commands.add_parser("run", help="Run one study YAML.")
    run.add_argument("--plan", required=True)
    run.add_argument(
        "--output-root",
        default="artifacts/studies/static_line_b_staged_v2",
    )
    run.add_argument("--resume")
    run.add_argument(
        "--device",
        help="Stage5 motion-training CUDA device override (cuda or cuda:N).",
    )
    run.add_argument(
        "--no-denoiser",
        action="store_true",
        help="Run Stage5 detector training/evaluation only and skip the PTT denoiser benchmark.",
    )
    report = commands.add_parser("report", help="Rebuild report and result backup.")
    report.add_argument("--study-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        plan = load_motion_peak_plan(args.plan)
        print(json.dumps({
            "status": "valid",
            "study_id": plan.study_id,
            "study_type": plan.study_type,
        }, sort_keys=True))
        return 0
    if args.command == "report":
        print(json.dumps(generate_motion_peak_report(args.study_dir), sort_keys=True))
        return 0
    progress = TerminalProgressSink()
    try:
        output = run_motion_peak_study(
            args.plan,
            pipeline_root=PIPELINE_ROOT,
            output_root=args.output_root,
            resume=args.resume,
            progress_sink=progress,
            device=args.device,
            include_denoiser=not args.no_denoiser,
        )
    finally:
        progress.close()
    print(f"Study output: {output}")
    print(f"Report: {output / 'STUDY_SUMMARY.md'}")
    print(f"Result backup: {output / 'result_backup'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
