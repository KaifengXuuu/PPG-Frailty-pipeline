#!/usr/bin/env python3
"""Export reusable V5 configuration defaults from one completed pipeline output."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ppg_frailty.v5.model_config_export import main


if __name__ == "__main__":
    raise SystemExit(main())
