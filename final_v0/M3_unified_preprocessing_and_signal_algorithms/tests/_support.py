"""测试路径和 fixture helpers / Test paths and fixture helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
FIXTURE_ROOT = PACKAGE_ROOT / "fixtures"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
sys.dont_write_bytecode = True


def load_fixture(name: str) -> np.ndarray:
    """只读载入一个 NPY fixture / Load one NPY fixture read-only."""

    return np.load(FIXTURE_ROOT / name, allow_pickle=False)

