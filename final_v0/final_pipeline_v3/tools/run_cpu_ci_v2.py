#!/usr/bin/env python3
"""Run the V2 non-scientific acceptance gate with warnings promoted to errors."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    output = arguments.output.resolve()
    output.relative_to(ROOT.resolve())
    environment = dict(os.environ)
    environment["PYTHONWARNINGS"] = "error"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/acceptance_gate_v2.py"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
