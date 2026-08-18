"""Typed study-plan contracts used by CLI, runner, reports, and Dash."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


_STUDY_KINDS = frozenset({"single", "ablation", "grid", "catalog_sweep"})
_DECISION_ROLES = frozenset(
    {"single_run", "screening", "ablation", "robustness", "candidate_comparison"}
)
_CATALOG_BALANCE_LINES = frozenset({"line_b"})
_CATALOG_SCOPES = frozenset(
    {"ordinary_13", "selected_ordinary", "matched_ensemble_pair"}
)
_CATALOG_OUTPUT_GROUPS = frozenset(
    {"raw", "fusion", "feature_vector", "feature_matrix"}
)
_SPARSE_SEARCH_METHODS = frozenset({"deterministic_sparse_profiles"})
_FORMAL_PROFILE_FAMILIES = frozenset({"fixed_kernel_samples"})
_SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")


def _strict_mapping(
    value: Any,
    *,
    label: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    payload = dict(value)
    if any(not isinstance(key, str) for key in payload):
        raise TypeError(f"{label} keys must be strings")
    missing = required - set(payload)
    unknown = set(payload) - required - optional
    if missing or unknown:
        raise ValueError(
            f"{label} key mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    return payload


def _safe_identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID_RE.fullmatch(value):
        raise ValueError(
            f"{label} must match {_SAFE_ID_RE.pattern} and be filesystem-safe"
        )
    return value


def _validate_override_paths(overrides: Mapping[str, Any]) -> None:
    paths: list[str] = []
    for raw_path in overrides:
        if not isinstance(raw_path, str):
            raise TypeError("catalog case override paths must be strings")
        fields = raw_path.split(".")
        if len(fields) < 2 or any(not field.strip() for field in fields):
            raise ValueError(
                "catalog case override paths must be non-empty dotted paths"
            )
        paths.append(raw_path)
    for index, path in enumerate(paths):
        for other in paths[index + 1 :]:
            if path.startswith(f"{other}.") or other.startswith(f"{path}."):
                raise ValueError(
                    "catalog case overrides cannot contain parent/child path "
                    f"collisions: {path!r}, {other!r}"
                )


def _unique_int_tuple(values: Any, *, label: str) -> tuple[int, ...]:
    if values in (None, "all"):
        return tuple(range(5))
    result = tuple(int(value) for value in values)
    if not result or len(result) != len(set(result)) or not set(result) <= set(range(5)):
        raise ValueError(f"{label} must be a unique non-empty subset of 0..4")
    return result


@dataclass(frozen=True)
class AxisSpec:
    """One dotted configuration path and its candidate values."""

    path: str
    values: tuple[Any, ...]
    reference: Any = None

    def __post_init__(self) -> None:
        fields = self.path.split(".")
        if not fields or any(not field.strip() for field in fields):
            raise ValueError("axis path must be a non-empty dotted path")
        if len(self.values) < 2:
            raise ValueError(f"axis {self.path} requires at least two values")
        if len({repr(value) for value in self.values}) != len(self.values):
            raise ValueError(f"axis {self.path} values must be unique")
        if self.reference is not None and all(self.reference != value for value in self.values):
            raise ValueError(f"axis {self.path} reference must be one of its values")


@dataclass(frozen=True)
class CatalogSpec:
    """Immutable source and scope for a formal Line B model catalogue."""

    path: str
    balance_line: str = "line_b"
    scope: str = "ordinary_13"

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("catalog path must be non-empty")
        if self.balance_line not in _CATALOG_BALANCE_LINES:
            raise ValueError(
                f"catalog balance_line must be one of "
                f"{sorted(_CATALOG_BALANCE_LINES)}"
            )
        if self.scope not in _CATALOG_SCOPES:
            raise ValueError(
                f"catalog scope must be one of {sorted(_CATALOG_SCOPES)}"
            )


@dataclass(frozen=True)
class SparseSearchSpec:
    """Deterministic, predeclared sparse profiles; never runtime sampling."""

    method: str
    selection_seed: int
    interpretation: str
    runtime_sampling: bool = False
    controlled_factors: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.method not in _SPARSE_SEARCH_METHODS:
            raise ValueError(
                f"sparse search method must be one of "
                f"{sorted(_SPARSE_SEARCH_METHODS)}"
            )
        if (
            isinstance(self.selection_seed, bool)
            or not isinstance(self.selection_seed, int)
            or not 0 <= self.selection_seed <= 0xFFFFFFFF
        ):
            raise ValueError("selection_seed must be an integer in 0..2^32-1")
        if self.runtime_sampling is not False:
            raise ValueError("catalog sweep runtime_sampling must be false")
        if (
            not isinstance(self.interpretation, str)
            or not self.interpretation.strip()
        ):
            raise ValueError("sparse search interpretation must be non-empty")
        for label, values in (
            ("controlled_factors", self.controlled_factors),
            ("notes", self.notes),
        ):
            if isinstance(values, (str, bytes)) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(
                    f"sparse search {label} must contain only non-empty strings"
                )
        if len(self.controlled_factors) != len(set(self.controlled_factors)):
            raise ValueError("sparse search controlled_factors must be unique")


@dataclass(frozen=True)
class FormalProfileSpec:
    """One registered formal single-factor profile, distinct from screen IDs."""

    family: str
    profile_id: str

    def __post_init__(self) -> None:
        if self.family not in _FORMAL_PROFILE_FAMILIES:
            raise ValueError(
                f"formal profile family must be one of "
                f"{sorted(_FORMAL_PROFILE_FAMILIES)}"
            )
        _safe_identifier(self.profile_id, label="formal profile_id")


@dataclass(frozen=True)
class CatalogCaseSpec:
    """One explicit, non-Cartesian catalogue screening profile."""

    case_id: str
    catalog_entry: str
    screen_profile_id: str
    output_group: str
    overrides: Mapping[str, Any]
    rationale: str
    formal_profile: FormalProfileSpec | None = None

    def __post_init__(self) -> None:
        _safe_identifier(self.case_id, label="catalog case_id")
        _safe_identifier(self.catalog_entry, label="catalog entry")
        _safe_identifier(self.screen_profile_id, label="screen_profile_id")
        if self.output_group not in _CATALOG_OUTPUT_GROUPS:
            raise ValueError(
                f"catalog case output_group must be one of "
                f"{sorted(_CATALOG_OUTPUT_GROUPS)}"
            )
        if not isinstance(self.overrides, Mapping):
            raise TypeError("catalog case overrides must be a mapping")
        _validate_override_paths(self.overrides)
        if self.formal_profile is not None and not isinstance(
            self.formal_profile, FormalProfileSpec
        ):
            raise TypeError("formal_profile must be FormalProfileSpec or None")
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise ValueError("catalog case rationale must be non-empty")


@dataclass(frozen=True)
class StudyInfo:
    """Human-readable scientific context required in every summary."""

    study_id: str
    kind: str
    purpose: str
    flow_position: str
    decision_role: str
    reference_case_id: str | None = None
    thesis_sections: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in _STUDY_KINDS:
            raise ValueError(f"study kind must be one of {sorted(_STUDY_KINDS)}")
        if self.decision_role not in _DECISION_ROLES:
            raise ValueError(
                f"decision_role must be one of {sorted(_DECISION_ROLES)}"
            )
        for value, name in (
            (self.study_id, "study_id"),
            (self.purpose, "purpose"),
            (self.flow_position, "flow_position"),
        ):
            if not str(value).strip():
                raise ValueError(f"study {name} must be non-empty")


@dataclass(frozen=True)
class ExecutionSpec:
    """Execution controls which never define a scientific grid axis."""

    repeats: tuple[int, ...] = tuple(range(5))
    folds: tuple[int, ...] = tuple(range(5))
    jobs: int = 1
    parallel_level: str = "cases"
    continue_on_error: bool = True
    allow_parallel_deep: bool = False

    def __post_init__(self) -> None:
        _unique_int_tuple(self.repeats, label="repeats")
        _unique_int_tuple(self.folds, label="folds")
        if isinstance(self.jobs, bool) or int(self.jobs) <= 0:
            raise ValueError("execution jobs must be a positive integer")
        if self.parallel_level != "cases":
            raise ValueError("only case-level parallelism is supported")


@dataclass(frozen=True)
class OutputSpec:
    """Study output root; each new run creates its own timestamped child."""

    root: str = "artifacts/studies"

    def __post_init__(self) -> None:
        if not str(self.root).strip():
            raise ValueError("output root must be non-empty")


@dataclass(frozen=True)
class ReportSpec:
    """Presentation settings; these do not alter model fitting or predictions."""

    top_k: int = 10
    write_html: bool = True
    write_static_figures: bool = True
    calibration_bins: int = 10

    def __post_init__(self) -> None:
        if not 1 <= int(self.top_k) <= 100:
            raise ValueError("report top_k must lie in 1..100")
        if not 2 <= int(self.calibration_bins) <= 100:
            raise ValueError("calibration_bins must lie in 2..100")


@dataclass(frozen=True)
class StudyPlan:
    """Complete configuration-independent study definition."""

    schema_version: str
    study: StudyInfo
    base_config: str | None = None
    axes: tuple[AxisSpec, ...] = ()
    catalog: CatalogSpec | None = None
    search: SparseSearchSpec | None = None
    cases: tuple[CatalogCaseSpec, ...] = ()
    execution: ExecutionSpec = field(default_factory=ExecutionSpec)
    output: OutputSpec = field(default_factory=OutputSpec)
    report: ReportSpec = field(default_factory=ReportSpec)
    plan_path: Path | None = None

    def __post_init__(self) -> None:
        if self.schema_version != "ppg_frailty.study_plan.v2":
            raise ValueError("unsupported study plan schema")
        paths = tuple(axis.path for axis in self.axes)
        if len(paths) != len(set(paths)):
            raise ValueError("study axis paths must be unique")
        if self.study.kind == "catalog_sweep":
            if self.base_config not in (None, ""):
                raise ValueError("catalog_sweep cannot define base_config")
            if self.axes:
                raise ValueError("catalog_sweep cannot define Cartesian axes")
            if not isinstance(self.catalog, CatalogSpec):
                raise ValueError("catalog_sweep requires a CatalogSpec")
            if not isinstance(self.search, SparseSearchSpec):
                raise ValueError("catalog_sweep requires a SparseSearchSpec")
            if not self.cases:
                raise ValueError("catalog_sweep requires explicit cases")
            if any(not isinstance(case, CatalogCaseSpec) for case in self.cases):
                raise TypeError("catalog_sweep cases must be CatalogCaseSpec values")
            case_ids = tuple(case.case_id for case in self.cases)
            if len(case_ids) != len(set(case_ids)):
                raise ValueError("catalog_sweep case_id values must be unique")
            if self.catalog.scope == "ordinary_13":
                entries = {case.catalog_entry for case in self.cases}
                if len(entries) != 13:
                    raise ValueError(
                        "ordinary_13 catalog_sweep requires 13 distinct "
                        "catalog_entry values"
                    )
            if self.catalog.scope == "matched_ensemble_pair":
                entries = {case.catalog_entry for case in self.cases}
                if len(self.cases) != 2 or len(entries) != 2:
                    raise ValueError(
                        "matched_ensemble_pair requires exactly two distinct "
                        "catalog entries"
                    )
                if any(
                    case.overrides or case.formal_profile is not None
                    for case in self.cases
                ):
                    raise ValueError(
                        "matched_ensemble_pair cannot add overrides or formal "
                        "profiles; the registered ensemble factor must be isolated"
                    )
            return
        if not str(self.base_config or "").strip():
            raise ValueError("base_config must be non-empty")
        if self.catalog is not None or self.search is not None or self.cases:
            raise ValueError(
                "single/ablation/grid studies cannot define catalog/search/cases"
            )
        if self.study.kind == "single" and self.axes:
            raise ValueError("single studies cannot define grid axes")
        if self.study.kind == "ablation" and len(self.axes) != 1:
            raise ValueError("ablation studies require exactly one axis")
        if self.study.kind == "grid" and not self.axes:
            raise ValueError("grid studies require one or more axes")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "study": {
                "study_id": self.study.study_id,
                "kind": self.study.kind,
                "purpose": self.study.purpose,
                "flow_position": self.study.flow_position,
                "decision_role": self.study.decision_role,
                "reference_case_id": self.study.reference_case_id,
                "thesis_sections": list(self.study.thesis_sections),
            },
            "execution": {
                "repeats": list(self.execution.repeats),
                "folds": list(self.execution.folds),
                "jobs": self.execution.jobs,
                "parallel_level": self.execution.parallel_level,
                "continue_on_error": self.execution.continue_on_error,
                "allow_parallel_deep": self.execution.allow_parallel_deep,
            },
            "output": {"root": self.output.root},
            "report": {
                "top_k": self.report.top_k,
                "write_html": self.report.write_html,
                "write_static_figures": self.report.write_static_figures,
                "calibration_bins": self.report.calibration_bins,
            },
        }
        if self.study.kind != "catalog_sweep":
            return {
                "schema_version": payload["schema_version"],
                "study": payload["study"],
                "base_config": self.base_config,
                "axes": [
                    {
                        "path": axis.path,
                        "values": list(axis.values),
                        "reference": axis.reference,
                    }
                    for axis in self.axes
                ],
                "execution": payload["execution"],
                "output": payload["output"],
                "report": payload["report"],
            }
        assert self.catalog is not None
        assert self.search is not None
        payload["catalog"] = {
            "path": self.catalog.path,
            "balance_line": self.catalog.balance_line,
            "scope": self.catalog.scope,
        }
        payload["search"] = {
            "method": self.search.method,
            "selection_seed": self.search.selection_seed,
            "runtime_sampling": self.search.runtime_sampling,
            "interpretation": self.search.interpretation,
            "controlled_factors": list(self.search.controlled_factors),
            "notes": list(self.search.notes),
        }
        cases = [
            {
                "case_id": case.case_id,
                "catalog_entry": case.catalog_entry,
                "screen_profile_id": case.screen_profile_id,
                "output_group": case.output_group,
                "overrides": copy.deepcopy(dict(case.overrides)),
                "rationale": case.rationale,
                "formal_profile": (
                    None
                    if case.formal_profile is None
                    else {
                        "family": case.formal_profile.family,
                        "profile_id": case.formal_profile.profile_id,
                    }
                ),
            }
            for case in self.cases
        ]
        return {
            "schema_version": payload["schema_version"],
            "study": payload["study"],
            "catalog": payload["catalog"],
            "search": payload["search"],
            "cases": cases,
            "execution": payload["execution"],
            "output": payload["output"],
            "report": payload["report"],
        }


def execution_from_mapping(value: Mapping[str, Any] | None) -> ExecutionSpec:
    payload = dict(value or {})
    return ExecutionSpec(
        repeats=_unique_int_tuple(payload.get("repeats"), label="repeats"),
        folds=_unique_int_tuple(payload.get("folds"), label="folds"),
        jobs=int(payload.get("jobs", 1)),
        parallel_level=str(payload.get("parallel_level", "cases")),
        continue_on_error=bool(payload.get("continue_on_error", True)),
        allow_parallel_deep=bool(payload.get("allow_parallel_deep", False)),
    )


def catalog_spec_from_mapping(value: Mapping[str, Any]) -> CatalogSpec:
    """Parse one strict catalog section for the study-plan parser."""

    payload = _strict_mapping(
        value,
        label="catalog",
        required=frozenset({"path"}),
        optional=frozenset({"balance_line", "scope"}),
    )
    return CatalogSpec(
        path=payload["path"],
        balance_line=payload.get("balance_line", "line_b"),
        scope=payload.get("scope", "ordinary_13"),
    )


def sparse_search_spec_from_mapping(
    value: Mapping[str, Any],
) -> SparseSearchSpec:
    """Parse deterministic sparse-search metadata without sampling."""

    payload = _strict_mapping(
        value,
        label="search",
        required=frozenset({"method", "selection_seed", "interpretation"}),
        optional=frozenset(
            {"runtime_sampling", "controlled_factors", "notes"}
        ),
    )
    controlled = payload.get("controlled_factors", ())
    notes = payload.get("notes", ())
    if isinstance(controlled, (str, bytes)) or not isinstance(
        controlled, (list, tuple)
    ):
        raise TypeError("search controlled_factors must be a list")
    if isinstance(notes, (str, bytes)) or not isinstance(notes, (list, tuple)):
        raise TypeError("search notes must be a list")
    return SparseSearchSpec(
        method=payload["method"],
        selection_seed=payload["selection_seed"],
        runtime_sampling=payload.get("runtime_sampling", False),
        interpretation=payload["interpretation"],
        controlled_factors=tuple(controlled),
        notes=tuple(notes),
    )


def formal_profile_spec_from_mapping(
    value: Mapping[str, Any] | None,
) -> FormalProfileSpec | None:
    """Parse an optional formal profile without conflating screen identity."""

    if value is None:
        return None
    payload = _strict_mapping(
        value,
        label="formal_profile",
        required=frozenset({"family", "profile_id"}),
    )
    return FormalProfileSpec(
        family=payload["family"],
        profile_id=payload["profile_id"],
    )


def catalog_case_spec_from_mapping(value: Mapping[str, Any]) -> CatalogCaseSpec:
    """Parse one strict explicit catalogue case."""

    payload = _strict_mapping(
        value,
        label="catalog case",
        required=frozenset(
            {
                "case_id",
                "catalog_entry",
                "screen_profile_id",
                "output_group",
                "overrides",
                "rationale",
                "formal_profile",
            }
        ),
    )
    overrides = payload["overrides"]
    if not isinstance(overrides, Mapping):
        raise TypeError("catalog case overrides must be a mapping")
    return CatalogCaseSpec(
        case_id=payload["case_id"],
        catalog_entry=payload["catalog_entry"],
        screen_profile_id=payload["screen_profile_id"],
        output_group=payload["output_group"],
        overrides=copy.deepcopy(dict(overrides)),
        rationale=payload["rationale"],
        formal_profile=formal_profile_spec_from_mapping(
            payload["formal_profile"]
        ),
    )


def catalog_cases_from_mapping(value: Any) -> tuple[CatalogCaseSpec, ...]:
    """Parse the non-empty explicit case list for a catalogue sweep."""

    if isinstance(value, (str, bytes, Mapping)) or not isinstance(
        value, (list, tuple)
    ):
        raise TypeError("catalog cases must be a list")
    if not value:
        raise ValueError("catalog cases must be non-empty")
    return tuple(catalog_case_spec_from_mapping(item) for item in value)
