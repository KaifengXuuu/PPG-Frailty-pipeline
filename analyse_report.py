#!/usr/bin/env python3
"""Run the V5 artifact-only report CLI without installing the package."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from ppg_frailty.v5_reporting.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
