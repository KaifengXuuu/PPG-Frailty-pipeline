"""Composable, artifact-only reporting facade for pipeline V5."""

from .analysis import build_analysis
from .collect import load_report_data
from .contracts import ReportContractError, ReportRequest, RunSpec
from .registry import MODULES, PRESETS, resolve_selection
from .validate import validate_report_data
from .writer import write_report

__all__ = [
    "MODULES",
    "PRESETS",
    "ReportContractError",
    "ReportRequest",
    "RunSpec",
    "build_analysis",
    "load_report_data",
    "resolve_selection",
    "validate_report_data",
    "write_report",
]
