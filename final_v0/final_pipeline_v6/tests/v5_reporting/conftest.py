from __future__ import annotations

from pathlib import Path
import sys


V5_ROOT = Path(__file__).resolve().parents[2]
SOURCE = V5_ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
