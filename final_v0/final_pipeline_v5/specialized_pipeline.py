#!/usr/bin/env python3
"""Run preserved non-canonical study schemas with V5 output boundaries."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from ppg_frailty.v5.specialized import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
