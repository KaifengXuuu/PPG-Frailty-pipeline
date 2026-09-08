"""Compatibility entry point for the unified historical analyzer."""

from pathlib import Path
from typing import Any

from .historical import run_historical_analysis


def run_historical_report_suite(
    *,
    early_source: str | Path,
    shapeformer_source: str | Path,
    fixed_epoch_source: str | Path,
    extension_source: str | Path,
    generalization_source: str | Path,
    output_dir: str | Path,
    **_: Any,
) -> Path:
    sources = (early_source, shapeformer_source, fixed_epoch_source, extension_source, generalization_source)
    return run_historical_analysis(tuple(Path(value).resolve() for value in sources), output_dir)


__all__ = ["run_historical_report_suite"]
