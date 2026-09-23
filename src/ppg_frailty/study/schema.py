"""Typed study-plan contracts used by CLI, runner, reports, and Dash."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

_STUDY_KINDS = frozenset("single ablation grid catalog_sweep".split())
_DECISION_ROLES = frozenset("single_run screening ablation robustness candidate_comparison".split())
_CATALOG_BALANCE_LINES = frozenset("line_a line_b".split())
_CATALOG_SCOPES = frozenset("ordinary_active ordinary_13 selected_ordinary matched_ensemble_pair".split())
_CATALOG_OUTPUT_GROUPS = frozenset("raw fusion feature_vector feature_matrix".split())
_SPARSE_SEARCH_METHODS = frozenset({"deterministic_sparse_profiles"})
_FORMAL_PROFILE_FAMILIES = frozenset({"fixed_kernel_samples"})
_PREPROCESSING_CACHE_MODES = frozenset("off read_only read_write".split())
_PREPROCESSING_CACHE_NAMESPACES = frozenset("imu_calibration canonical_signal_views motion_windows raw_windows".split())
_SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FIELD_DRIVEN_BRIDGE_DESIGNS = frozenset({"centered_star_v1", "field_driven_followup_v1"})
_BRIDGE_DESIGNS = frozenset({"cumulative_chain_v1", *_FIELD_DRIVEN_BRIDGE_DESIGNS})
_STAR_CONTROL_KEYS = frozenset(
    "ppg_preprocessing imu_preprocessing target_fs_hz window_seconds hop_seconds "
    "historical_retained_fraction max_windows_per_file allow_short_record_padding "
    "normalization sampler class_weighting optimizer batch_size fixed_epochs "
    "learning_rate weight_decay training_metric_aggregation_rule "
    "primary_report_aggregation_view".split()
)
_STAR_FACTORS = {
    "B1": ("sampling_rate", frozenset({"target_fs_hz"})),
    "B2": ("window_plan", frozenset({"window_seconds", "hop_seconds"})),
    "B3": ("imu_preprocessing", frozenset({"imu_preprocessing"})),
    "B4": ("normalization", frozenset({"normalization"})),
    "B5": ("sampler", frozenset({"sampler"})),
    "B6": ("optimizer_and_batch_size", frozenset({"optimizer", "batch_size"})),
    "B7": ("primary_aggregation", frozenset({"primary_report_aggregation_view"})),
}
_FACTOR_FIELDS = frozenset({"factor_id", "overrides", "expected_changed_paths", "interpretation"})
_PROFILE_FIELDS = frozenset({"case_id", "catalog_case_id", "model_id", "profile_id"})
_PROFILE_OPTIONAL = frozenset({"factor_id", "reference_case_id", "changed_control_paths", "controls", "controls_sha256", "interpretation"})
_BRIDGE_SEQUENCE_FIELDS = tuple("aggregation_views primary_table_columns sampling_diagnostics restrictions required_runtime_capabilities".split())

def _strict_mapping(value: Any, *, label: str, required: frozenset[str], optional: frozenset[str] = frozenset()) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    payload = dict(value)
    if any(not isinstance(key, str) for key in payload):
        raise TypeError(f"{label} keys must be strings")
    missing, unknown = required - set(payload), set(payload) - required - optional
    if missing or unknown:
        raise ValueError(f"{label} key mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}")
    return payload

def _safe_identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID_RE.fullmatch(value):
        raise ValueError(f"{label} must match {_SAFE_ID_RE.pattern} and be filesystem-safe")
    return value

def _validate_override_paths(overrides: Mapping[str, Any]) -> None:
    paths: list[str] = []
    for raw_path in overrides:
        if not isinstance(raw_path, str):
            raise TypeError("catalog case override paths must be strings")
        fields = raw_path.split(".")
        if (len(fields) < 2 and raw_path != "roles") or any(not field.strip() for field in fields):
            raise ValueError("catalog case override paths must be non-empty dotted paths; " "the registered top-level roles selector is also configurable")
        paths.append(raw_path)
    for index, path in enumerate(paths):
        for other in paths[index + 1 :]:
            if path.startswith(f"{other}.") or other.startswith(f"{path}."):
                raise ValueError(f"catalog case overrides cannot contain parent/child path collisions: {path!r}, {other!r}")

def _unique_int_tuple(values: Any, *, label: str) -> tuple[int, ...]:
    if values in (None, "all"):
        return tuple(range(5))
    result = tuple(int(value) for value in values)
    if not result or len(result) != len(set(result)) or not set(result) <= set(range(5)):
        raise ValueError(f"{label} must be a unique non-empty subset of 0..4")
    return result

def _string_tuple(values: Any, *, label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise TypeError(f"{label} must be a list")
    result = tuple(values)
    if not result or any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must contain unique values")
    return result

def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _record(value: Any, *, sequences: tuple[str, ...] = (), mappings: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        name: copy.deepcopy(list(item))
        if name in sequences
        else copy.deepcopy(dict(item))
        if name in mappings
        else item
        for name, item in vars(value).items()
    }

def _validate_star_controls(value: Any, *, label: str) -> dict[str, Any]:
    controls = _strict_mapping(value, label=label, required=_STAR_CONTROL_KEYS)
    _canonical_sha256(controls)
    numeric = ("target_fs_hz", "window_seconds", "hop_seconds", "learning_rate")
    integers = ("target_fs_hz", "batch_size", "fixed_epochs")
    if any(isinstance(controls[key], bool) or not isinstance(controls[key], (int, float)) or controls[key] <= 0 for key in numeric):
        raise ValueError(f"{label} rates, windows, and learning_rate must be positive")
    if any(isinstance(controls[key], bool) or not isinstance(controls[key], int) or controls[key] <= 0 for key in integers):
        raise ValueError(f"{label} target_fs_hz, batch_size, and fixed_epochs must be positive integers")
    exempt = {*numeric, *integers, "historical_retained_fraction", "max_windows_per_file", "allow_short_record_padding", "weight_decay"}
    if any(not isinstance(controls[key], str) or not controls[key] for key in _STAR_CONTROL_KEYS - exempt):
        raise TypeError(f"{label} module and strategy controls must be non-empty strings")
    fraction, cap, decay = (controls[key] for key in ("historical_retained_fraction", "max_windows_per_file", "weight_decay"))
    if fraction is not None and (isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction <= 1):
        raise ValueError(f"{label}.historical_retained_fraction must lie in (0, 1]")
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0):
        raise ValueError(f"{label}.max_windows_per_file must be null or positive")
    if not isinstance(controls["allow_short_record_padding"], bool):
        raise TypeError(f"{label}.allow_short_record_padding must be boolean")
    if isinstance(decay, bool) or not isinstance(decay, (int, float)) or decay < 0:
        raise ValueError(f"{label}.weight_decay must be non-negative")
    return copy.deepcopy(controls)

def _materialize_field_contract(
    baseline_value: Any, factors_value: Any, profiles_value: Any, *, centered: bool
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]], tuple[Mapping[str, Any], ...], dict[str, Mapping[str, Any]]]:
    baseline = _validate_star_controls(baseline_value, label="legacy_bridge.baseline_controls")
    if centered:
        raw_factors = _strict_mapping(
            factors_value,
            label="legacy_bridge.factor_overrides",
            required=frozenset(f"B{index}" for index in range(1, 8)),
        )
    elif not isinstance(factors_value, Mapping) or not factors_value:
        raise ValueError("follow-up factor_overrides must be a non-empty mapping")
    else:
        raw_factors = factors_value
        if isinstance(profiles_value, (str, bytes, Mapping)) or not isinstance(profiles_value, (list, tuple)):
            raise TypeError("legacy_bridge.profiles must be a list")
    factors: dict[str, Mapping[str, Any]] = {}
    effective: dict[str, Mapping[str, Any]] = {"B0": baseline} if centered else {}
    for raw_profile_id, value in raw_factors.items():
        profile_id = str(raw_profile_id)
        if not centered and (not profile_id or not profile_id.replace("_", "").isalnum()):
            raise ValueError("follow-up profile identifiers must be safe")
        row = _strict_mapping(value, label=f"legacy_bridge.factor_overrides.{profile_id}", required=_FACTOR_FIELDS)
        factor_id = str(row["factor_id"]) if centered else row["factor_id"]
        overrides = _strict_mapping(
            row["overrides"],
            label=f"legacy_bridge.factor_overrides.{profile_id}.overrides",
            required=frozenset(),
            optional=_STAR_CONTROL_KEYS,
        )
        paths = _string_tuple(
            row["expected_changed_paths"],
            label=f"legacy_bridge.factor_overrides.{profile_id}.expected_changed_paths",
        )
        declared_paths = {f"controls.{key}" for key in overrides}
        if centered:
            expected_factor, expected_keys = _STAR_FACTORS[profile_id]
            if factor_id != expected_factor or set(overrides) != set(expected_keys) or set(paths) != declared_paths:
                raise ValueError(f"centered-star {profile_id}/{factor_id} is not its declared single-factor change")
            if not isinstance(row["interpretation"], str) or not row["interpretation"].strip():
                raise ValueError("centered-star factor interpretation must be non-empty")
        else:
            if not overrides or set(paths) != declared_paths:
                raise ValueError(f"follow-up {profile_id} must declare every changed control exactly")
            if not isinstance(factor_id, str) or not factor_id.strip():
                raise ValueError("follow-up factor_id must be non-empty")
            if not isinstance(row["interpretation"], str) or not row["interpretation"].strip():
                raise ValueError("follow-up interpretation must be non-empty")
        controls = _validate_star_controls(
            {**copy.deepcopy(baseline), **copy.deepcopy(overrides)},
            label=f"legacy_bridge.effective_controls.{profile_id}",
        )
        if {key for key in controls if controls[key] != baseline[key]} != set(overrides):
            suffix = "a no-op override" if centered else "a no-op or undeclared override"
            raise ValueError(f"{'centered-star' if centered else 'follow-up'} {profile_id} contains {suffix}")
        factors[profile_id] = {
            "factor_id": factor_id,
            "overrides": copy.deepcopy(overrides),
            "expected_changed_paths": list(paths),
            "interpretation": row["interpretation"],
        }
        effective[profile_id] = controls

    if centered and (isinstance(profiles_value, (str, bytes, Mapping)) or not isinstance(profiles_value, (list, tuple))):
        raise TypeError("legacy_bridge.profiles must be a list")
    prefix = "centered-star" if centered else "follow-up"
    compact: list[dict[str, Any]] = []
    for index, value in enumerate(profiles_value):
        row = _strict_mapping(
            value,
            label=f"legacy_bridge.profiles[{index}]",
            required=_PROFILE_FIELDS,
            optional=_PROFILE_OPTIONAL,
        )
        if not isinstance(row["case_id"], str) or not row["case_id"].strip():
            raise ValueError(f"{prefix} profile case_id must be non-empty")
        _safe_identifier(row["catalog_case_id"], label=f"{prefix} profile catalog_case_id")
        if row["model_id"] not in {"CompactCNN1D", "InceptionTimeFull"}:
            raise ValueError(f"{prefix} model_id is unsupported")
        if row["profile_id"] not in effective:
            detail = "must be B0..B7" if centered else "lacks declared controls"
            raise ValueError(f"{prefix} profile_id {detail}")
        compact.append(row)
    identities = tuple((str(row["model_id"]), str(row["profile_id"])) for row in compact)
    if centered:
        roster = tuple((model_id, f"B{index}") for index in range(8) for model_id in ("CompactCNN1D", "InceptionTimeFull"))
        if tuple((row["model_id"], row["profile_id"]) for row in compact) != roster:
            raise ValueError("centered-star profiles must be profile-major paired CompactCNN/InceptionTime B0..B7")
        references = {row["model_id"]: row["case_id"] for row in compact if row["profile_id"] == "B0"}
    else:
        if not compact or len(identities) != len(set(identities)):
            raise ValueError("follow-up model/profile identities must be non-empty and unique")
        if {str(row["profile_id"]) for row in compact} != set(effective):
            raise ValueError("follow-up profiles must consume every declared control set")
        references = {}

    profiles: list[Mapping[str, Any]] = []
    for index, row in enumerate(compact):
        model_id, profile_id = str(row["model_id"]), str(row["profile_id"])
        factor = factors.get(profile_id)
        if centered:
            reference = None if profile_id == "B0" else references[model_id]
        else:
            first = references.setdefault(model_id, row["case_id"])
            reference = None if first == row["case_id"] else first
        materialized = {
            "case_id": row["case_id"],
            "catalog_case_id": row["catalog_case_id"],
            "model_id": row["model_id"] if centered else model_id,
            "profile_id": profile_id,
            "factor_id": "baseline" if factor is None else factor["factor_id"],
            "reference_case_id": reference,
            "changed_control_paths": [] if factor is None else list(factor["expected_changed_paths"]),
            "controls": copy.deepcopy(effective[profile_id]),
            "controls_sha256": _canonical_sha256(effective[profile_id]),
            "interpretation": row.get(
                "interpretation",
                "complete legacy baseline" if factor is None else factor["interpretation"],
            ),
        }
        for key in _PROFILE_OPTIONAL - {"interpretation"}:
            if key in row and row[key] != materialized[key]:
                design = "centered-star" if centered else "follow-up"
                raise ValueError(f"legacy_bridge.profiles[{index}].{key} differs from materialized {design} controls")
        if centered and (not isinstance(materialized["interpretation"], str) or not materialized["interpretation"].strip()):
            raise ValueError("centered-star profile interpretation must be non-empty")
        profiles.append(materialized)
    return baseline, factors, tuple(profiles), effective

def _materialize_centered_star_contract(
    baseline_value: Any, factors_value: Any, profiles_value: Any
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]], tuple[Mapping[str, Any], ...], dict[str, Mapping[str, Any]],]:
    return _materialize_field_contract(baseline_value, factors_value, profiles_value, centered=True)

def _materialize_field_driven_followup_contract(
    baseline_value: Any, factors_value: Any, profiles_value: Any
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]], tuple[Mapping[str, Any], ...], dict[str, Mapping[str, Any]],]:
    return _materialize_field_contract(baseline_value, factors_value, profiles_value, centered=False)

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
    scope: str = "ordinary_active"
    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("catalog path must be non-empty")
        if self.balance_line not in _CATALOG_BALANCE_LINES:
            raise ValueError(f"catalog balance_line must be one of {sorted(_CATALOG_BALANCE_LINES)}")
        if self.scope not in _CATALOG_SCOPES:
            raise ValueError(f"catalog scope must be one of {sorted(_CATALOG_SCOPES)}")

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
            raise ValueError(f"sparse search method must be one of {sorted(_SPARSE_SEARCH_METHODS)}")
        if isinstance(self.selection_seed, bool) or not isinstance(self.selection_seed, int) or not 0 <= self.selection_seed <= 0xFFFFFFFF:
            raise ValueError("selection_seed must be an integer in 0..2^32-1")
        if self.runtime_sampling is not False:
            raise ValueError("catalog sweep runtime_sampling must be false")
        if not isinstance(self.interpretation, str) or not self.interpretation.strip():
            raise ValueError("sparse search interpretation must be non-empty")
        for label, values in (("controlled_factors", self.controlled_factors), ("notes", self.notes)):
            if isinstance(values, (str, bytes)) or any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"sparse search {label} must contain only non-empty strings")
        if len(self.controlled_factors) != len(set(self.controlled_factors)):
            raise ValueError("sparse search controlled_factors must be unique")

@dataclass(frozen=True)
class LegacyBridgeSpec:
    """Auditable contract for the isolated legacy/V2 bridge."""
    schema_version: str
    protocol_id: str
    source_specification: str
    source_specification_sha256: str
    runtime_status: str
    phase0: Mapping[str, Any]
    budget: Mapping[str, Any]
    execution_order: tuple[str, ...]
    numeric_profile_order: tuple[str, ...]
    adjacent_comparisons: tuple[str, ...]
    profiles: tuple[Mapping[str, Any], ...]
    aggregation_views: tuple[str, ...]
    primary_table_columns: tuple[str, ...]
    sampling_diagnostics: tuple[str, ...]
    restrictions: tuple[str, ...]
    required_runtime_capabilities: tuple[str, ...]
    design: str = "cumulative_chain_v1"
    baseline_controls: Mapping[str, Any] | None = None
    factor_overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    def __post_init__(self) -> None:
        if self.schema_version != "ppg_frailty.legacy_v2_bridge_protocol.v1":
            raise ValueError("unsupported legacy bridge protocol schema")
        if self.design not in _BRIDGE_DESIGNS:
            raise ValueError(f"unsupported legacy bridge design: {self.design}")
        _safe_identifier(self.protocol_id, label="legacy bridge protocol_id")
        if not str(self.source_specification).strip() or not _SHA256_RE.fullmatch(self.source_specification_sha256):
            raise ValueError("legacy bridge source specification/hash is invalid")
        for label in ("execution_order", "numeric_profile_order", *_BRIDGE_SEQUENCE_FIELDS):
            values = getattr(self, label)
            if not values or len(values) != len(set(values)) or not all(str(value).strip() for value in values):
                raise ValueError(f"legacy bridge {label} must contain unique values")
        profile_ids = tuple(str(row.get("case_id", "")) for row in self.profiles)
        if profile_ids != self.execution_order or set(self.numeric_profile_order) != set(profile_ids):
            raise ValueError("legacy bridge profile and execution rosters differ")
        if self.uses_inline_profiles:
            if self.baseline_controls is None:
                raise ValueError("field-driven bridge requires baseline controls")
            materializer = _materialize_centered_star_contract if self.design == "centered_star_v1" else _materialize_field_driven_followup_contract
            baseline, factors, profiles, effective = materializer(self.baseline_controls, self.factor_overrides, self.profiles)
            if baseline != self.baseline_controls or factors != self.factor_overrides or profiles != self.profiles:
                raise ValueError("field-driven bridge controls are not canonical")
            if _canonical_sha256(effective) != self.source_specification_sha256:
                raise ValueError("field-driven bridge controls hash differs")
        elif self.baseline_controls is not None or self.factor_overrides:
            raise ValueError("cumulative bridge cannot define field-driven controls")
    @property
    def uses_inline_profiles(self) -> bool:
        return self.design in _FIELD_DRIVEN_BRIDGE_DESIGNS
    @property
    def centered_comparisons(self) -> tuple[Mapping[str, Any], ...]:
        if self.design != "centered_star_v1":
            return ()
        return tuple(
            {
                "model_id": profile["model_id"],
                "reference_case_id": profile["reference_case_id"],
                "variant_case_id": profile["case_id"],
                "profile_id": profile["profile_id"],
                "factor_id": profile["factor_id"],
                "changed_control_paths": list(profile["changed_control_paths"]),
            }
            for profile in self.profiles
            if profile["profile_id"] != "B0"
        )
    def controls_sha256(self, profile: Mapping[str, Any]) -> str:
        value = str(profile.get("controls_sha256", ""))
        if not self.uses_inline_profiles or not _SHA256_RE.fullmatch(value):
            raise ValueError("controls_sha256 requires a field-driven profile")
        return value
    def to_dict(self) -> dict[str, Any]:
        sequences = ("execution_order", "numeric_profile_order", "adjacent_comparisons", "profiles", *_BRIDGE_SEQUENCE_FIELDS)
        result = _record(self, sequences=sequences, mappings=("phase0", "budget", "factor_overrides"))
        result["profiles"] = [copy.deepcopy(dict(value)) for value in self.profiles]
        if not self.uses_inline_profiles:
            for name in ("design", "baseline_controls", "factor_overrides"):
                result.pop(name)
        else:
            result["baseline_controls"] = copy.deepcopy(dict(self.baseline_controls or {}))
            if self.design == "centered_star_v1":
                result["centered_comparisons"] = [copy.deepcopy(dict(value)) for value in self.centered_comparisons]
        return result

@dataclass(frozen=True)
class FormalProfileSpec:
    """One registered formal single-factor profile."""
    family: str
    profile_id: str
    def __post_init__(self) -> None:
        if self.family not in _FORMAL_PROFILE_FAMILIES:
            raise ValueError(f"formal profile family must be one of {sorted(_FORMAL_PROFILE_FAMILIES)}")
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
        for value, label in (
            (self.case_id, "catalog case_id"),
            (self.catalog_entry, "catalog entry"),
            (self.screen_profile_id, "screen_profile_id"),
        ):
            _safe_identifier(value, label=label)
        if self.output_group not in _CATALOG_OUTPUT_GROUPS:
            raise ValueError(f"catalog case output_group must be one of {sorted(_CATALOG_OUTPUT_GROUPS)}")
        if not isinstance(self.overrides, Mapping):
            raise TypeError("catalog case overrides must be a mapping")
        _validate_override_paths(self.overrides)
        if self.formal_profile is not None and not isinstance(self.formal_profile, FormalProfileSpec):
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
            raise ValueError(f"decision_role must be one of {sorted(_DECISION_ROLES)}")
        for value, name in (
            (self.study_id, "study_id"),
            (self.purpose, "purpose"),
            (self.flow_position, "flow_position"),
        ):
            if not str(value).strip():
                raise ValueError(f"study {name} must be non-empty")

@dataclass(frozen=True)
class PreprocessingCacheSpec:
    """Operational recording cache; never a scientific grid axis."""
    mode: str = "off"
    root: str = "artifacts/studies/cache"
    namespaces: tuple[str, ...] = ("imu_calibration", "canonical_signal_views", "motion_windows", "raw_windows")
    verify_source_sha256: bool = True
    def __post_init__(self) -> None:
        if self.mode not in _PREPROCESSING_CACHE_MODES:
            raise ValueError("preprocessing cache mode must be off, read_only, or read_write")
        if not isinstance(self.root, str) or not self.root.strip():
            raise ValueError("preprocessing cache root must be non-empty")
        if not self.namespaces or len(self.namespaces) != len(set(self.namespaces)) or not set(self.namespaces) <= _PREPROCESSING_CACHE_NAMESPACES:
            raise ValueError("preprocessing cache namespaces must be a unique non-empty subset " f"of {sorted(_PREPROCESSING_CACHE_NAMESPACES)}")
        if self.verify_source_sha256 is not True:
            raise ValueError("preprocessing cache requires source SHA-256 verification")
    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "root": self.root,
            "namespaces": list(self.namespaces),
            "verify_source_sha256": self.verify_source_sha256,
        }

def preprocessing_cache_from_mapping(value: Mapping[str, Any] | None) -> PreprocessingCacheSpec:
    payload = _strict_mapping(
        value or {},
        label="execution.preprocessing_cache",
        required=frozenset(),
        optional=frozenset({"mode", "root", "namespaces", "verify_source_sha256"}),
    )
    namespaces = payload.get("namespaces", ["imu_calibration", "canonical_signal_views", "motion_windows", "raw_windows"])
    return PreprocessingCacheSpec(
        mode=str(payload.get("mode", "off")),
        root=str(payload.get("root", "artifacts/studies/cache")),
        namespaces=_string_tuple(namespaces, label="execution.preprocessing_cache.namespaces"),
        verify_source_sha256=payload.get("verify_source_sha256", True),
    )

@dataclass(frozen=True)
class ExecutionSpec:
    """Execution controls which never define a scientific grid axis."""
    repeats: tuple[int, ...] = tuple(range(5))
    folds: tuple[int, ...] = tuple(range(5))
    jobs: int = 1
    device: str | None = None
    parallel_level: str = "cases"
    continue_on_error: bool = True
    allow_parallel_deep: bool = False
    measure_operational_costs: bool = False
    preprocessing_cache: PreprocessingCacheSpec = field(default_factory=PreprocessingCacheSpec)
    def __post_init__(self) -> None:
        _unique_int_tuple(self.repeats, label="repeats")
        _unique_int_tuple(self.folds, label="folds")
        if isinstance(self.jobs, bool) or int(self.jobs) <= 0:
            raise ValueError("execution jobs must be a positive integer")
        if self.device is not None and (not isinstance(self.device, str) or not self.device.strip()):
            raise ValueError("execution device must be null or a non-empty string")
        if self.parallel_level != "cases":
            raise ValueError("only case-level parallelism is supported")
        for name in ("continue_on_error", "allow_parallel_deep", "measure_operational_costs"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"execution {name} must be boolean")
        if not isinstance(self.preprocessing_cache, PreprocessingCacheSpec):
            raise TypeError("execution preprocessing_cache must be PreprocessingCacheSpec")

@dataclass(frozen=True)
class OutputSpec:
    """Study output root; each new run creates its own timestamped child."""
    root: str = "artifacts/studies"
    def __post_init__(self) -> None:
        if not str(self.root).strip():
            raise ValueError("output root must be non-empty")

@dataclass(frozen=True)
class ReportSpec:
    """Presentation settings; these do not alter fitting or predictions."""
    top_k: int = 10
    detailed_configuration_top_k: int = 0
    write_html: bool = True
    write_static_figures: bool = True
    calibration_bins: int = 10
    figure_modules: tuple[str, ...] = ("all",)
    compact_mean_sd: bool = True
    write_excel_workbook: bool = True
    classification_tsne_random_state: int = 42
    classification_tsne_perplexity: float = 30.0
    classification_tsne_max_samples: int = 5000
    classification_roc_macro_grid_points: int = 201
    classification_score_histogram_bins: int = 40
    def __post_init__(self) -> None:
        if not 1 <= int(self.top_k) <= 100:
            raise ValueError("report top_k must lie in 1..100")
        if not 0 <= int(self.detailed_configuration_top_k) <= 100:
            raise ValueError("report detailed_configuration_top_k must lie in 0..100")
        if not 2 <= int(self.calibration_bins) <= 100:
            raise ValueError("calibration_bins must lie in 2..100")
        if not self.figure_modules or any(not str(value).strip() for value in self.figure_modules):
            raise ValueError("figure_modules must contain non-empty module names")
        if not math.isfinite(float(self.classification_tsne_perplexity)) or float(self.classification_tsne_perplexity) <= 0:
            raise ValueError("classification_tsne_perplexity must be positive")
        for name, minimum in (
            ("classification_tsne_max_samples", 3),
            ("classification_roc_macro_grid_points", 2),
            ("classification_score_histogram_bins", 2),
        ):
            if int(getattr(self, name)) < minimum:
                suffix = name.removeprefix("classification_")
                raise ValueError(f"classification_{suffix} must be at least {minimum}")

def _case_dict(case: CatalogCaseSpec) -> dict[str, Any]:
    result = _record(case, mappings=("overrides",))
    formal = result["formal_profile"]
    result["formal_profile"] = None if formal is None else vars(formal).copy()
    return result

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
    legacy_bridge: LegacyBridgeSpec | None = None
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
            if self.base_config not in (None, "") or self.axes:
                raise ValueError("catalog_sweep uses catalog cases, not base_config/axes")
            if not isinstance(self.catalog, CatalogSpec) or not isinstance(self.search, SparseSearchSpec):
                raise ValueError("catalog_sweep requires catalog and search sections")
            if not self.cases or any(not isinstance(case, CatalogCaseSpec) for case in self.cases):
                raise ValueError("catalog_sweep requires explicit typed cases")
            case_ids = tuple(case.case_id for case in self.cases)
            if len(case_ids) != len(set(case_ids)):
                raise ValueError("catalog_sweep case IDs must be unique")
            if self.legacy_bridge is not None:
                profile_ids = tuple(str(row["catalog_case_id"]) for row in self.legacy_bridge.profiles)
                if case_ids != profile_ids:
                    raise ValueError("legacy bridge profiles and catalog cases differ")
            return
        if self.legacy_bridge is not None or self.catalog is not None or self.search is not None or self.cases:
            raise ValueError("non-catalog studies cannot define catalog-only sections")
        if not str(self.base_config or "").strip():
            raise ValueError("base_config must be non-empty")
        expected_axes = {"single": 0, "ablation": 1}
        if self.study.kind in expected_axes and len(self.axes) != expected_axes[self.study.kind]:
            raise ValueError(f"{self.study.kind} study has the wrong axis count")
        if self.study.kind == "grid" and not self.axes:
            raise ValueError("grid studies require at least one axis")
    def to_dict(self) -> dict[str, Any]:
        execution = _record(self.execution, sequences=("repeats", "folds"))
        device = execution.pop("device")
        execution["preprocessing_cache"] = self.execution.preprocessing_cache.to_dict()
        if device is not None:
            execution["device"] = device
        report = _record(self.report, sequences=("figure_modules",))
        common = {"execution": execution, "output": {"root": self.output.root}, "report": report}
        if self.study.kind != "catalog_sweep":
            return {
                "schema_version": self.schema_version,
                "study": _record(self.study, sequences=("thesis_sections",)),
                "base_config": self.base_config,
                "axes": [{"path": axis.path, "values": list(axis.values), "reference": axis.reference} for axis in self.axes],
                **common,
            }
        assert self.catalog is not None and self.search is not None
        result = {
            "schema_version": self.schema_version,
            "study": _record(self.study, sequences=("thesis_sections",)),
            "catalog": vars(self.catalog).copy(),
            "search": {name: list(value) if name in {"controlled_factors", "notes"} else value for name, value in vars(self.search).items()},
            "cases": [_case_dict(case) for case in self.cases],
            **common,
        }
        if self.legacy_bridge is not None:
            result["legacy_bridge"] = self.legacy_bridge.to_dict()
        return result

def execution_from_mapping(value: Mapping[str, Any] | None) -> ExecutionSpec:
    optional = frozenset(
        "repeats folds jobs device parallel_level continue_on_error allow_parallel_deep " "measure_operational_costs preprocessing_cache".split()
    )
    payload = _strict_mapping(value or {}, label="execution", required=frozenset(), optional=optional)
    return ExecutionSpec(
        repeats=_unique_int_tuple(payload.get("repeats"), label="repeats"),
        folds=_unique_int_tuple(payload.get("folds"), label="folds"),
        jobs=int(payload.get("jobs", 1)),
        device=payload.get("device"),
        parallel_level=str(payload.get("parallel_level", "cases")),
        continue_on_error=payload.get("continue_on_error", True),
        allow_parallel_deep=payload.get("allow_parallel_deep", False),
        measure_operational_costs=payload.get("measure_operational_costs", False),
        preprocessing_cache=preprocessing_cache_from_mapping(payload.get("preprocessing_cache")),
    )

def catalog_spec_from_mapping(value: Mapping[str, Any]) -> CatalogSpec:
    payload = _strict_mapping(
        value,
        label="catalog",
        required=frozenset({"path"}),
        optional=frozenset({"balance_line", "scope"}),
    )
    return CatalogSpec(
        path=payload["path"],
        balance_line=payload.get("balance_line", "line_b"),
        scope=payload.get("scope", "ordinary_active"),
    )

def sparse_search_spec_from_mapping(value: Mapping[str, Any]) -> SparseSearchSpec:
    payload = _strict_mapping(
        value,
        label="search",
        required=frozenset({"method", "selection_seed", "interpretation"}),
        optional=frozenset({"runtime_sampling", "controlled_factors", "notes"}),
    )
    controlled, notes = payload.get("controlled_factors", ()), payload.get("notes", ())
    if isinstance(controlled, (str, bytes)) or not isinstance(controlled, (list, tuple)):
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

def legacy_bridge_spec_from_mapping(value: Mapping[str, Any]) -> LegacyBridgeSpec:
    """Normalize transport types while materializers retain numerical controls."""

    if not isinstance(value, Mapping):
        raise TypeError("legacy_bridge must be a mapping")
    payload = copy.deepcopy(dict(value))
    design = str(payload.get("design", "cumulative_chain_v1"))
    phase0 = {
        "enabled": False,
        "advisory_only": True,
        "mandatory": False,
        "affects_training_execution": False,
        **dict(payload.get("phase0", {})),
    }
    if not isinstance(phase0["enabled"], bool):
        raise TypeError("legacy_bridge.phase0.enabled must be boolean")
    budget_value = payload.get("budget")
    if not isinstance(budget_value, Mapping):
        raise TypeError("legacy_bridge.budget must be a mapping")
    budget = copy.deepcopy(dict(budget_value))
    profiles_value = payload.get("profiles")
    if isinstance(profiles_value, (str, bytes, Mapping)) or not isinstance(profiles_value, (list, tuple)):
        raise TypeError("legacy_bridge.profiles must be a list")
    baseline: Mapping[str, Any] | None = None
    factors: Mapping[str, Mapping[str, Any]] = {}
    if design in _FIELD_DRIVEN_BRIDGE_DESIGNS:
        materializer = _materialize_centered_star_contract if design == "centered_star_v1" else _materialize_field_driven_followup_contract
        baseline, factors, profiles, _ = materializer(payload.get("baseline_controls"), payload.get("factor_overrides"), profiles_value)
    else:
        profiles = tuple(copy.deepcopy(dict(row)) for row in profiles_value)
        for index, row in enumerate(profiles):
            if not row.get("case_id") or not row.get("catalog_case_id"):
                raise ValueError(f"legacy_bridge.profiles[{index}] requires case identifiers")
    execution = _string_tuple(payload.get("execution_order"), label="legacy_bridge.execution_order")
    numeric = _string_tuple(payload.get("numeric_profile_order", execution), label="legacy_bridge.numeric_profile_order")
    adjacent_value = payload.get("adjacent_comparisons", ())
    adjacent = () if not adjacent_value else _string_tuple(adjacent_value, label="legacy_bridge.adjacent_comparisons")
    sequences = {name: _string_tuple(payload.get(name), label=f"legacy_bridge.{name}") for name in _BRIDGE_SEQUENCE_FIELDS}
    return LegacyBridgeSpec(
        schema_version=str(payload.get("schema_version", "")),
        protocol_id=str(payload.get("protocol_id", "")),
        source_specification=str(payload.get("source_specification", "")),
        source_specification_sha256=str(payload.get("source_specification_sha256", "")),
        runtime_status=str(payload.get("runtime_status", "")),
        phase0=phase0,
        budget=budget,
        execution_order=execution,
        numeric_profile_order=numeric,
        adjacent_comparisons=adjacent,
        profiles=tuple(profiles),
        design=design,
        baseline_controls=copy.deepcopy(baseline),
        factor_overrides=copy.deepcopy(factors),
        **sequences,
    )

def formal_profile_spec_from_mapping(value: Mapping[str, Any] | None) -> FormalProfileSpec | None:
    if value is None:
        return None
    payload = _strict_mapping(value, label="formal_profile", required=frozenset({"family", "profile_id"}))
    return FormalProfileSpec(family=payload["family"], profile_id=payload["profile_id"])

def catalog_case_spec_from_mapping(value: Mapping[str, Any]) -> CatalogCaseSpec:
    payload = _strict_mapping(
        value,
        label="catalog case",
        required=frozenset({"case_id", "catalog_entry", "screen_profile_id", "output_group", "overrides", "rationale", "formal_profile"}),
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
        formal_profile=formal_profile_spec_from_mapping(payload["formal_profile"]),
    )

def catalog_cases_from_mapping(value: Any) -> tuple[CatalogCaseSpec, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, (list, tuple)):
        raise TypeError("catalog cases must be a list")
    if not value:
        raise ValueError("catalog cases must be non-empty")
    return tuple(catalog_case_spec_from_mapping(item) for item in value)
