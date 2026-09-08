#!/usr/bin/env python3
"""Validate or run one materialized Dashboard comparison sequence."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from ppg_frailty.dashboard.sequence_cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
