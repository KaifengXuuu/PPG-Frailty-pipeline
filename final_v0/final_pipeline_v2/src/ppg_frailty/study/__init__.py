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
from .schema import (
    AxisSpec,
    CatalogCaseSpec,
    CatalogSpec,
    ExecutionSpec,
    FormalProfileSpec,
    LegacyBridgeSpec,
    OutputSpec,
    ReportSpec,
    SparseSearchSpec,
    StudyInfo,
    StudyPlan,
    catalog_case_spec_from_mapping,
    catalog_cases_from_mapping,
    catalog_spec_from_mapping,
    formal_profile_spec_from_mapping,
    legacy_bridge_spec_from_mapping,
    sparse_search_spec_from_mapping,
)

__all__ = [
    "AxisSpec",
    "CatalogCaseSpec",
    "CatalogSpec",
    "CompositeProgressSink",
    "ExecutionSpec",
    "FormalProfileSpec",
    "LegacyBridgeSpec",
    "JsonlProgressSink",
    "NullProgressSink",
    "OutputSpec",
    "ProgressEvent",
    "ReportSpec",
    "ResolvedCase",
    "SparseSearchSpec",
    "StudyExpansion",
    "StudyInfo",
    "StudyPlan",
    "StudyRunResult",
    "StudyRunner",
    "TerminalProgressSink",
    "default_experiment_executor",
    "catalog_case_spec_from_mapping",
    "catalog_cases_from_mapping",
    "catalog_spec_from_mapping",
    "expand_study",
    "flatten_mapping",
    "formal_profile_spec_from_mapping",
    "legacy_bridge_spec_from_mapping",
    "load_study_plan",
    "parse_study_plan",
    "sparse_search_spec_from_mapping",
    "validate_canonical_expansion",
]
