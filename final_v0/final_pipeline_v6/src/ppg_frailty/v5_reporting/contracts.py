"""Small, typed contracts for the V5 artifact-only report runner.

The report layer deliberately owns no training code.  It consumes immutable
prediction artifacts and delegates all scientific calculations to the copied,
tested :mod:`ppg_frailty.training` and :mod:`ppg_frailty.reporting` modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPORT_MODES = frozenset({"single", "comparison", "ablation", "test"})
PREDICTION_LAYERS = ("window", "file", "role", "participant", "member")
EVALUATION_SCOPES = frozenset({"outer_oof", "independent_test"})


class ReportContractError(ValueError):
    """Raised when inputs or requested report semantics are not auditable."""


@dataclass(frozen=True)
class RunSpec:
    """One named immutable run or V2-compatible study directory."""

    case_id: str
    path: Path

    def __post_init__(self) -> None:
        case_id = str(self.case_id).strip()
        if not case_id:
            raise ReportContractError("run case_id must be non-empty")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve())


@dataclass(frozen=True)
class ArtifactRecord:
    """Inventory row for one input prediction artifact."""

    case_id: str
    layer: str
    path: Path
    repeat: int | None
    fold: int | None
    row_count: int
    byte_count: int
    artifact_state: str
    empty_reason: str
    sha256: str


@dataclass(frozen=True)
class LoadedReportData:
    """Normalized input shared by the old analysis code and V5 additions."""

    collected: Any
    layer_rows: Mapping[str, tuple[Mapping[str, Any], ...]]
    artifact_records: tuple[ArtifactRecord, ...]
    evaluation_scope_by_case: Mapping[str, str]
    source_root_by_case: Mapping[str, Path]
    config_by_case: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    manifest_by_case: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    legacy_v2_cases: frozenset[str] = frozenset()
    source_kind: str = "run"

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.evaluation_scope_by_case))


@dataclass(frozen=True)
class ValidationIssue:
    """One deterministic validation observation."""

    severity: str
    code: str
    message: str
    case_id: str | None = None
    repeat: int | None = None
    fold: int | None = None


@dataclass(frozen=True)
class ValidationReport:
    """Validation result; errors are normally raised before this is returned."""

    status: str
    issues: tuple[ValidationIssue, ...]
    cases: tuple[str, ...]
    artifact_count: int
    prediction_row_count: int


@dataclass(frozen=True)
class ReportRequest:
    """Fully resolved CLI request persisted in ``analysis_manifest.json``."""

    mode: str
    runs: tuple[RunSpec, ...]
    output_dir: Path | None = None
    include_cases: tuple[str, ...] = ()
    exclude_cases: tuple[str, ...] = ()
    reference_case: str | None = None
    comparison_family: str = "declared_comparison"
    factor_paths: tuple[str, ...] = ()
    presets: tuple[str, ...] = ()
    modules: tuple[str, ...] = ()
    figures: tuple[str, ...] | None = None
    tables: tuple[str, ...] | None = None
    bootstrap_resamples: int = 10_000
    permutation_resamples: int = 100_000
    statistics_seed: int = 42
    alpha: float = 0.05
    calibration_bins: int = 10
    validation_depth: str = "full"
    on_missing: str = "na"
    allow_v2_compatibility: bool = True

    def __post_init__(self) -> None:
        for name in ("include_cases", "exclude_cases", "factor_paths"):
            normalized = tuple(dict.fromkeys(str(value).strip() for value in getattr(self, name) if str(value).strip()))
            object.__setattr__(self, name, normalized)
        family = str(self.comparison_family).strip()
        if not family:
            raise ReportContractError("comparison_family must be non-empty")
        object.__setattr__(self, "comparison_family", family)
        if self.reference_case is not None:
            reference = str(self.reference_case).strip()
            if not reference:
                raise ReportContractError("reference_case must be non-empty when provided")
            object.__setattr__(self, "reference_case", reference)
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.mode not in REPORT_MODES:
            raise ReportContractError(f"unsupported report mode: {self.mode!r}")
        if not self.runs:
            raise ReportContractError("at least one --run is required")
        if len({run.case_id for run in self.runs}) != len(self.runs):
            raise ReportContractError("--run names must be unique")
        if set(self.include_cases) & set(self.exclude_cases):
            raise ReportContractError("a case cannot be both included and excluded")
        if self.validation_depth not in {"selected", "full"}:
            raise ReportContractError("validation_depth must be selected or full")
        if self.on_missing not in {"error", "na", "skip"}:
            raise ReportContractError("on_missing must be error, na, or skip")
        if self.bootstrap_resamples <= 0 or self.permutation_resamples <= 0:
            raise ReportContractError("resample budgets must be positive")
        if self.statistics_seed < 0:
            raise ReportContractError("statistics_seed must be non-negative")
        if not 0.0 < float(self.alpha) < 1.0:
            raise ReportContractError("alpha must be in (0,1)")
        if self.calibration_bins < 2:
            raise ReportContractError("calibration_bins must be at least two")
        if self.mode in {"comparison", "ablation"} and not self.reference_case:
            raise ReportContractError(f"{self.mode} mode requires --reference-case")
        if self.mode == "ablation" and not self.factor_paths:
            raise ReportContractError("ablation mode requires --factor-path")


@dataclass(frozen=True)
class ModuleSpec:
    """Static module registration; no discovery or implicit figure injection."""

    name: str
    modes: frozenset[str]
    tables: tuple[str, ...]
    figures: tuple[str, ...]
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedSelection:
    """Exact output selection after preset expansion and strict validation."""

    modules: tuple[str, ...]
    tables: tuple[str, ...]
    figures: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisProducts:
    """Existing V2 analysis plus V5-only normalized tables."""

    analysis: Any
    tables: Mapping[str, tuple[Mapping[str, Any], ...]]
    notes: tuple[str, ...] = ()


TableGetter = Callable[[LoadedReportData, AnalysisProducts], Sequence[Mapping[str, Any]]]
