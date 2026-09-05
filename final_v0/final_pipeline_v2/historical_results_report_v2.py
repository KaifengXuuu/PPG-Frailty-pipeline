#!/usr/bin/env python3
"""Build the selected historical-search evidence bundle without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ppg_frailty.reporting.historical import run_historical_major_report
from ppg_frailty.reporting.historical_suite import run_historical_report_suite


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    repository_root = ROOT.parent.parent
    parser.add_argument(
        "--early-sweep",
        default=str(
            repository_root
            / "results_frailty3/_overfitting_sweep/20260527_1320_cnn_inceptionTime"
        ),
    )
    parser.add_argument(
        "--shapeformer-sweep",
        default=str(
            repository_root
            / "results_frailty3/_overfitting_sweep/20260528_1045_shapeformer_0extra"
        ),
    )
    parser.add_argument(
        "--fixed-epoch-sweep",
        default=str(
            repository_root
            / "results_frailty3/_overfitting_sweep/20260608_1206_overfitting_sweep_stage1_rank2"
        ),
    )
    parser.add_argument(
        "--extension-sweep",
        default=str(
            repository_root
            / "results_frailty3/_overfitting_sweep/20260625_2320_overfitting_sweep_stage1_rank2"
        ),
    )
    parser.add_argument(
        "--generalization-sweep",
        default=str(
            repository_root
            / "results_frailty3/_overfitting_sweep/20260630_0630_overfitting_sweep_generalization_rank2"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            repository_root
            / "results_frailty3/_v2_reanalysis/20260824_historical_search_reports_v2"
        ),
        help="Immutable empty destination; the reporter refuses non-empty directories.",
    )
    parser.add_argument(
        "--layout",
        choices=("split", "legacy-combined"),
        default="split",
        help=(
            "split writes the requested early merged report plus three independent "
            "June reports; legacy-combined retains the prior four-source Markdown bundle"
        ),
    )
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--overlap-percent", type=float, default=50.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--extra-input", default="0")
    return parser


def main() -> int:
    args = _parser().parse_args()
    common = {
        "early_source": args.early_sweep,
        "shapeformer_source": args.shapeformer_sweep,
        "fixed_epoch_source": args.fixed_epoch_sweep,
        "extension_source": args.extension_sweep,
        "output_dir": args.output_dir,
        "window_seconds": args.window_seconds,
        "overlap_percent": args.overlap_percent,
        "patience": args.patience,
        "extra_input": args.extra_input,
    }
    if args.layout == "split":
        output = run_historical_report_suite(
            **common,
            generalization_source=args.generalization_sweep,
        )
    else:
        output = run_historical_major_report(**common)
    print(f"Historical V2-oriented report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
