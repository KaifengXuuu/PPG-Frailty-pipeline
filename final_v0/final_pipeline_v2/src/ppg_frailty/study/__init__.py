"""Configuration-driven grid and ablation studies.

The study package is an orchestration layer only.  It never implements signal,
feature, representation, model, or evaluation algorithms; resolved cases are
delegated to the canonical pipeline executor.
"""

from .expand import (
    ResolvedCase,
    StudyExpansion,
    expand_study,
    flatten_mapping,
    load_study_plan,
    parse_study_plan,
    validate_canonical_expansion,
)
from .progress import (
    CompositeProgressSink,
    JsonlProgressSink,
    NullProgressSink,
    ProgressEvent,
    TerminalProgressSink,
)
from .runner import StudyRunResult, StudyRunner, default_experiment_executor
from .schema import AxisSpec, ExecutionSpec, OutputSpec, ReportSpec, StudyInfo, StudyPlan

__all__ = [
    "AxisSpec",
    "CompositeProgressSink",
    "ExecutionSpec",
    "JsonlProgressSink",
    "NullProgressSink",
    "OutputSpec",
    "ProgressEvent",
    "ReportSpec",
    "ResolvedCase",
    "StudyExpansion",
    "StudyInfo",
    "StudyPlan",
    "StudyRunResult",
    "StudyRunner",
    "TerminalProgressSink",
    "default_experiment_executor",
    "expand_study",
    "flatten_mapping",
    "load_study_plan",
    "parse_study_plan",
    "validate_canonical_expansion",
]
