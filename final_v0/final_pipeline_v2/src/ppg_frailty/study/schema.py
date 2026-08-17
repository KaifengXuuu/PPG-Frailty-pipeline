"""Typed study-plan contracts used by CLI, runner, reports, and Dash."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


_STUDY_KINDS = frozenset({"single", "ablation", "grid"})
_DECISION_ROLES = frozenset(
    {"single_run", "screening", "ablation", "robustness", "candidate_comparison"}
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
    base_config: str
    axes: tuple[AxisSpec, ...] = ()
    execution: ExecutionSpec = field(default_factory=ExecutionSpec)
    output: OutputSpec = field(default_factory=OutputSpec)
    report: ReportSpec = field(default_factory=ReportSpec)
    plan_path: Path | None = None

    def __post_init__(self) -> None:
        if self.schema_version != "ppg_frailty.study_plan.v2":
            raise ValueError("unsupported study plan schema")
        if not str(self.base_config).strip():
            raise ValueError("base_config must be non-empty")
        paths = tuple(axis.path for axis in self.axes)
        if len(paths) != len(set(paths)):
            raise ValueError("study axis paths must be unique")
        if self.study.kind == "single" and self.axes:
            raise ValueError("single studies cannot define grid axes")
        if self.study.kind == "ablation" and len(self.axes) != 1:
            raise ValueError("ablation studies require exactly one axis")
        if self.study.kind == "grid" and not self.axes:
            raise ValueError("grid studies require one or more axes")

    def to_dict(self) -> dict[str, Any]:
        return {
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
            "base_config": self.base_config,
            "axes": [
                {
                    "path": axis.path,
                    "values": list(axis.values),
                    "reference": axis.reference,
                }
                for axis in self.axes
            ],
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
