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


_STUDY_KINDS = frozenset({"single", "ablation", "grid", "catalog_sweep"})
_DECISION_ROLES = frozenset(
    {"single_run", "screening", "ablation", "robustness", "candidate_comparison"}
)
_CATALOG_BALANCE_LINES = frozenset({"line_a", "line_b"})
_CATALOG_SCOPES = frozenset(
    {"ordinary_active", "ordinary_13", "selected_ordinary", "matched_ensemble_pair"}
)
_CATALOG_OUTPUT_GROUPS = frozenset(
    {"raw", "fusion", "feature_vector", "feature_matrix"}
)
_SPARSE_SEARCH_METHODS = frozenset({"deterministic_sparse_profiles"})
_FORMAL_PROFILE_FAMILIES = frozenset({"fixed_kernel_samples"})
_LEGACY_BRIDGE_RUNTIME_STATUS = "implemented_advisory_audit_v1"
_SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BRIDGE_INCEPTION_L0 = "inception_full__L0_legacy64_w15_fixed10"
_BRIDGE_STUDY_ID = "staged_static_03_legacy_v2_bridge_v2"
_BRIDGE_STAR_STUDY_ID = "staged_static_03_centered_star_v1"
_BRIDGE_V3_STUDY_ID = "staged_static_03_v3_b1_b2_followup_v1"
_FIELD_DRIVEN_BRIDGE_DESIGNS = frozenset(
    {"centered_star_v1", "field_driven_followup_v1"}
)
_BRIDGE_DESIGNS = frozenset(
    {"cumulative_chain_v1", *_FIELD_DRIVEN_BRIDGE_DESIGNS}
)
_BRIDGE_COMPACT_BY_LEVEL = {
    0: "compact_cnn__L0_legacy64_w15_fixed10",
    1: "compact_cnn__L1_legacy64_w5_fixed10",
    2: "compact_cnn__L2_legacy400_w5_fixed10",
    3: "compact_cnn__L3_v2_imu_window_scaled_fixed10",
    4: "compact_cnn__L4_v2_imu_fold_scaled_fixed10",
    5: "compact_cnn__L5_uniform_replacement_fixed10",
    6: "compact_cnn__L6_v2_line_b_balance_fixed10",
    7: "compact_cnn__L7_v2_training_bundle_fixed10",
}
_BRIDGE_EXECUTION_ORDER = (
    _BRIDGE_INCEPTION_L0,
    _BRIDGE_COMPACT_BY_LEVEL[7],
    _BRIDGE_COMPACT_BY_LEVEL[5],
    _BRIDGE_COMPACT_BY_LEVEL[6],
    _BRIDGE_COMPACT_BY_LEVEL[4],
    _BRIDGE_COMPACT_BY_LEVEL[3],
    _BRIDGE_COMPACT_BY_LEVEL[2],
    _BRIDGE_COMPACT_BY_LEVEL[1],
    _BRIDGE_COMPACT_BY_LEVEL[0],
)
_BRIDGE_NUMERIC_ORDER = (
    _BRIDGE_INCEPTION_L0,
    *(_BRIDGE_COMPACT_BY_LEVEL[level] for level in range(8)),
)
_BRIDGE_ADJACENT_COMPARISONS = tuple(
    f"{_BRIDGE_COMPACT_BY_LEVEL[level]}->{_BRIDGE_COMPACT_BY_LEVEL[level + 1]}"
    for level in range(7)
)
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
    "B6": (
        "optimizer_and_batch_size",
        frozenset({"optimizer", "batch_size"}),
    ),
    "B7": ("primary_aggregation", frozenset({"primary_report_aggregation_view"})),
}


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
        if (
            (len(fields) < 2 and raw_path != "roles")
            or any(not field.strip() for field in fields)
        ):
            raise ValueError(
                "catalog case override paths must be non-empty dotted paths; "
                "the registered top-level roles selector is also configurable"
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


def _string_tuple(values: Any, *, label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise TypeError(f"{label} must be a list")
    result = tuple(values)
    if not result or any(
        not isinstance(value, str) or not value.strip() for value in result
    ):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must contain unique values")
    return result


def _canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible protocol data without YAML formatting ambiguity."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_star_controls(value: Any, *, label: str) -> dict[str, Any]:
    controls = _strict_mapping(value, label=label, required=_STAR_CONTROL_KEYS)
    _canonical_sha256(controls)  # Reject non-finite/non-JSON parameters.
    numeric = ("target_fs_hz", "window_seconds", "hop_seconds", "learning_rate")
    integers = ("target_fs_hz", "batch_size", "fixed_epochs")
    if any(
        isinstance(controls[key], bool)
        or not isinstance(controls[key], (int, float))
        or controls[key] <= 0
        for key in numeric
    ):
        raise ValueError(f"{label} rates, windows, and learning_rate must be positive")
    if any(
        isinstance(controls[key], bool)
        or not isinstance(controls[key], int)
        or controls[key] <= 0
        for key in integers
    ):
        raise ValueError(
            f"{label} target_fs_hz, batch_size, and fixed_epochs must be "
            "positive integers"
        )
    text_keys = _STAR_CONTROL_KEYS - set(numeric) - set(integers) - {
        "historical_retained_fraction", "max_windows_per_file",
        "allow_short_record_padding", "weight_decay",
    }
    if any(
        not isinstance(controls[key], str) or not controls[key]
        for key in text_keys
    ):
        raise TypeError(
            f"{label} module and strategy controls must be non-empty strings"
        )
    fraction, cap, decay = (
        controls[key]
        for key in (
            "historical_retained_fraction",
            "max_windows_per_file",
            "weight_decay",
        )
    )
    if fraction is not None and (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not 0 < fraction <= 1
    ):
        raise ValueError(f"{label}.historical_retained_fraction must lie in (0, 1]")
    if cap is not None and (
        isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0
    ):
        raise ValueError(f"{label}.max_windows_per_file must be null or positive")
    if not isinstance(controls["allow_short_record_padding"], bool):
        raise TypeError(f"{label}.allow_short_record_padding must be boolean")
    if isinstance(decay, bool) or not isinstance(decay, (int, float)) or decay < 0:
        raise ValueError(f"{label}.weight_decay must be non-negative")
    return copy.deepcopy(controls)


def _materialize_centered_star_contract(
    baseline_value: Any,
    factors_value: Any,
    profiles_value: Any,
) -> tuple[
    dict[str, Any],
    dict[str, Mapping[str, Any]],
    tuple[Mapping[str, Any], ...],
    dict[str, Mapping[str, Any]],
]:
    """Validate one star declaration and materialize model-neutral controls."""

    baseline = _validate_star_controls(
        baseline_value, label="legacy_bridge.baseline_controls"
    )
    raw_factors = _strict_mapping(
        factors_value,
        label="legacy_bridge.factor_overrides",
        required=frozenset({f"B{index}" for index in range(1, 8)}),
    )
    factors: dict[str, Mapping[str, Any]] = {}
    effective: dict[str, Mapping[str, Any]] = {"B0": baseline}
    for profile_id, value in raw_factors.items():
        row = _strict_mapping(
            value,
            label=f"legacy_bridge.factor_overrides.{profile_id}",
            required=frozenset(
                {"factor_id", "overrides", "expected_changed_paths", "interpretation"}
            ),
        )
        factor_id = str(row["factor_id"])
        expected_factor_id, expected_keys = _STAR_FACTORS[profile_id]
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
        if (
            factor_id != expected_factor_id
            or set(overrides) != set(expected_keys)
            or set(paths) != {f"controls.{key}" for key in overrides}
        ):
            raise ValueError(
                f"centered-star {profile_id}/{factor_id} is not its declared "
                "single-factor change"
            )
        if not isinstance(row["interpretation"], str) or not row[
            "interpretation"
        ].strip():
            raise ValueError("centered-star factor interpretation must be non-empty")
        controls = copy.deepcopy(baseline)
        controls.update(copy.deepcopy(overrides))
        controls = _validate_star_controls(
            controls, label=f"legacy_bridge.effective_controls.{profile_id}"
        )
        if {key for key in controls if controls[key] != baseline[key]} != set(
            overrides
        ):
            raise ValueError(f"centered-star {profile_id} contains a no-op override")
        factors[profile_id] = {
            "factor_id": factor_id,
            "overrides": copy.deepcopy(overrides),
            "expected_changed_paths": list(paths),
            "interpretation": row["interpretation"],
        }
        effective[profile_id] = controls
    if isinstance(profiles_value, (str, bytes, Mapping)) or not isinstance(
        profiles_value, (list, tuple)
    ):
        raise TypeError("legacy_bridge.profiles must be a list")
    compact: list[dict[str, Any]] = []
    optional = frozenset(
        {
            "factor_id",
            "reference_case_id",
            "changed_control_paths",
            "controls",
            "controls_sha256",
            "interpretation",
        }
    )
    for index, value in enumerate(profiles_value):
        row = _strict_mapping(
            value,
            label=f"legacy_bridge.profiles[{index}]",
            required=frozenset(
                {"case_id", "catalog_case_id", "model_id", "profile_id"}
            ),
            optional=optional,
        )
        if not isinstance(row["case_id"], str) or not row["case_id"].strip():
            raise ValueError("centered-star profile case_id must be non-empty")
        _safe_identifier(
            row["catalog_case_id"], label="centered-star profile catalog_case_id"
        )
        if row["model_id"] not in {"CompactCNN1D", "InceptionTimeFull"}:
            raise ValueError("centered-star model_id is unsupported")
        if row["profile_id"] not in effective:
            raise ValueError("centered-star profile_id must be B0..B7")
        compact.append(row)
    roster = tuple(
        (model_id, f"B{index}")
        for index in range(8)
        for model_id in ("CompactCNN1D", "InceptionTimeFull")
    )
    if tuple((row["model_id"], row["profile_id"]) for row in compact) != roster:
        raise ValueError(
            "centered-star profiles must be profile-major paired "
            "CompactCNN/InceptionTime B0..B7"
        )
    b0_case = {
        row["model_id"]: row["case_id"] for row in compact if row["profile_id"] == "B0"
    }
    profiles: list[Mapping[str, Any]] = []
    for index, row in enumerate(compact):
        profile_id = str(row["profile_id"])
        factor = factors.get(profile_id)
        materialized = {
            "case_id": row["case_id"],
            "catalog_case_id": row["catalog_case_id"],
            "model_id": row["model_id"],
            "profile_id": profile_id,
            "factor_id": "baseline" if factor is None else factor["factor_id"],
            "reference_case_id": (
                None if profile_id == "B0" else b0_case[row["model_id"]]
            ),
            "changed_control_paths": (
                [] if factor is None else list(factor["expected_changed_paths"])
            ),
            "controls": copy.deepcopy(effective[profile_id]),
            "controls_sha256": _canonical_sha256(effective[profile_id]),
            "interpretation": row.get(
                "interpretation",
                "complete legacy baseline"
                if factor is None
                else factor["interpretation"],
            ),
        }
        for key in optional - {"interpretation"}:
            if key in row and row[key] != materialized[key]:
                raise ValueError(
                    f"legacy_bridge.profiles[{index}].{key} differs from "
                    "materialized centered-star controls"
                )
        if not isinstance(materialized["interpretation"], str) or not materialized[
            "interpretation"
        ].strip():
            raise ValueError("centered-star profile interpretation must be non-empty")
        profiles.append(materialized)
    return baseline, factors, tuple(profiles), effective


def _materialize_field_driven_followup_contract(
    baseline_value: Any,
    factors_value: Any,
    profiles_value: Any,
) -> tuple[
    dict[str, Any],
    dict[str, Mapping[str, Any]],
    tuple[Mapping[str, Any], ...],
    dict[str, Mapping[str, Any]],
]:
    """Materialize arbitrary reviewed combinations through the shared runtime."""

    baseline = _validate_star_controls(
        baseline_value, label="legacy_bridge.baseline_controls"
    )
    if not isinstance(factors_value, Mapping) or not factors_value:
        raise ValueError("follow-up factor_overrides must be a non-empty mapping")
    if isinstance(profiles_value, (str, bytes, Mapping)) or not isinstance(
        profiles_value, (list, tuple)
    ):
        raise TypeError("legacy_bridge.profiles must be a list")
    factors: dict[str, Mapping[str, Any]] = {}
    effective: dict[str, Mapping[str, Any]] = {}
    for raw_profile_id, value in factors_value.items():
        profile_id = str(raw_profile_id)
        if not profile_id or not profile_id.replace("_", "").isalnum():
            raise ValueError("follow-up profile identifiers must be safe")
        row = _strict_mapping(
            value,
            label=f"legacy_bridge.factor_overrides.{profile_id}",
            required=frozenset(
                {"factor_id", "overrides", "expected_changed_paths", "interpretation"}
            ),
        )
        overrides = _strict_mapping(
            row["overrides"],
            label=f"legacy_bridge.factor_overrides.{profile_id}.overrides",
            required=frozenset(),
            optional=_STAR_CONTROL_KEYS,
        )
        paths = _string_tuple(
            row["expected_changed_paths"],
            label=(
                f"legacy_bridge.factor_overrides.{profile_id}."
                "expected_changed_paths"
            ),
        )
        if not overrides or set(paths) != {f"controls.{key}" for key in overrides}:
            raise ValueError(
                f"follow-up {profile_id} must declare every changed control exactly"
            )
        if not isinstance(row["factor_id"], str) or not row["factor_id"].strip():
            raise ValueError("follow-up factor_id must be non-empty")
        if not isinstance(row["interpretation"], str) or not row[
            "interpretation"
        ].strip():
            raise ValueError("follow-up interpretation must be non-empty")
        controls = copy.deepcopy(baseline)
        controls.update(copy.deepcopy(overrides))
        controls = _validate_star_controls(
            controls, label=f"legacy_bridge.effective_controls.{profile_id}"
        )
        changed = {key for key in controls if controls[key] != baseline[key]}
        if changed != set(overrides):
            raise ValueError(
                f"follow-up {profile_id} contains a no-op or undeclared override"
            )
        factors[profile_id] = {
            "factor_id": row["factor_id"],
            "overrides": copy.deepcopy(overrides),
            "expected_changed_paths": list(paths),
            "interpretation": row["interpretation"],
        }
        effective[profile_id] = controls

    compact: list[dict[str, Any]] = []
    optional = frozenset(
        {
            "factor_id",
            "reference_case_id",
            "changed_control_paths",
            "controls",
            "controls_sha256",
            "interpretation",
        }
    )
    for index, value in enumerate(profiles_value):
        row = _strict_mapping(
            value,
            label=f"legacy_bridge.profiles[{index}]",
            required=frozenset(
                {"case_id", "catalog_case_id", "model_id", "profile_id"}
            ),
            optional=optional,
        )
        if not isinstance(row["case_id"], str) or not row["case_id"].strip():
            raise ValueError("follow-up profile case_id must be non-empty")
        _safe_identifier(
            row["catalog_case_id"], label="follow-up profile catalog_case_id"
        )
        if row["model_id"] not in {"CompactCNN1D", "InceptionTimeFull"}:
            raise ValueError("follow-up model_id is unsupported")
        if row["profile_id"] not in effective:
            raise ValueError("follow-up profile_id lacks declared controls")
        compact.append(row)
    identities = tuple(
        (str(row["model_id"]), str(row["profile_id"])) for row in compact
    )
    if not compact or len(identities) != len(set(identities)):
        raise ValueError("follow-up model/profile identities must be non-empty and unique")
    if {str(row["profile_id"]) for row in compact} != set(effective):
        raise ValueError("follow-up profiles must consume every declared control set")

    reference_by_model: dict[str, str] = {}
    profiles: list[Mapping[str, Any]] = []
    for index, row in enumerate(compact):
        model_id = str(row["model_id"])
        profile_id = str(row["profile_id"])
        factor = factors[profile_id]
        reference_case_id = reference_by_model.setdefault(model_id, row["case_id"])
        materialized = {
            "case_id": row["case_id"],
            "catalog_case_id": row["catalog_case_id"],
            "model_id": model_id,
            "profile_id": profile_id,
            "factor_id": factor["factor_id"],
            "reference_case_id": (
                None if reference_case_id == row["case_id"] else reference_case_id
            ),
            "changed_control_paths": list(factor["expected_changed_paths"]),
            "controls": copy.deepcopy(effective[profile_id]),
            "controls_sha256": _canonical_sha256(effective[profile_id]),
            "interpretation": row.get("interpretation", factor["interpretation"]),
        }
        for key in optional - {"interpretation"}:
            if key in row and row[key] != materialized[key]:
                raise ValueError(
                    f"legacy_bridge.profiles[{index}].{key} differs from "
                    "materialized follow-up controls"
                )
        profiles.append(materialized)
    return baseline, factors, tuple(profiles), effective


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
class LegacyBridgeSpec:
    """Auditable contract for the isolated legacy/V2 bridge.

    This metadata is deliberately separate from pipeline overrides.  The
    historical sampler, class-weight, IMU, and window profiles are implemented
    by a dedicated bridge runtime; treating them as ordinary catalog overrides
    would silently weaken the canonical V2 configuration contract.
    """

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
            raise ValueError(
                f"legacy bridge design must be one of {sorted(_BRIDGE_DESIGNS)}"
            )
        _safe_identifier(self.protocol_id, label="legacy bridge protocol_id")
        if (
            not isinstance(self.source_specification, str)
            or not self.source_specification.strip()
        ):
            raise ValueError("legacy bridge source_specification must be non-empty")
        if not _SHA256_RE.fullmatch(self.source_specification_sha256):
            raise ValueError(
                "legacy bridge source_specification_sha256 must be lowercase SHA-256"
            )
        expected_runtime = (
            "implemented_field_driven_centered_star_v1"
            if self.design == "centered_star_v1"
            else "implemented_field_driven_followup_v1"
            if self.design == "field_driven_followup_v1"
            else _LEGACY_BRIDGE_RUNTIME_STATUS
        )
        if self.runtime_status != expected_runtime:
            raise ValueError(
                "legacy bridge runtime_status must identify the selected "
                f"runtime {expected_runtime!r}"
            )
        for label, values in (
            ("execution_order", self.execution_order),
            ("numeric_profile_order", self.numeric_profile_order),
            ("aggregation_views", self.aggregation_views),
            ("primary_table_columns", self.primary_table_columns),
            ("sampling_diagnostics", self.sampling_diagnostics),
            ("restrictions", self.restrictions),
            ("required_runtime_capabilities", self.required_runtime_capabilities),
        ):
            if not values or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(
                    f"legacy bridge {label} must contain non-empty strings"
                )
            if len(values) != len(set(values)):
                raise ValueError(f"legacy bridge {label} must be unique")
        if self.design == "centered_star_v1":
            self._validate_centered_star()
            return
        if self.design == "field_driven_followup_v1":
            self._validate_field_driven_followup()
            return
        if self.baseline_controls is not None or self.factor_overrides:
            raise ValueError(
                "cumulative legacy bridge cannot define centered-star controls"
            )
        if not self.adjacent_comparisons or len(
            self.adjacent_comparisons
        ) != len(set(self.adjacent_comparisons)):
            raise ValueError(
                "legacy bridge adjacent_comparisons must be non-empty and unique"
            )
        if len(self.execution_order) != 9 or len(self.profiles) != 9:
            raise ValueError("legacy bridge requires exactly 9 ordered profiles")
        if self.execution_order != _BRIDGE_EXECUTION_ORDER:
            raise ValueError("legacy bridge execution order drifted")
        if self.numeric_profile_order != _BRIDGE_NUMERIC_ORDER:
            raise ValueError("legacy bridge numeric profile order drifted")
        if self.adjacent_comparisons != _BRIDGE_ADJACENT_COMPARISONS:
            raise ValueError("legacy bridge adjacent comparison chain drifted")
        if any(not isinstance(profile, Mapping) for profile in self.profiles):
            raise TypeError("legacy bridge profiles must be mappings")
        if any(
            "case_id" not in profile or "catalog_case_id" not in profile
            for profile in self.profiles
        ):
            raise ValueError(
                "legacy bridge profiles require case_id and catalog_case_id"
            )
        display_ids = tuple(str(profile["case_id"]) for profile in self.profiles)
        if display_ids != self.execution_order:
            raise ValueError(
                "legacy bridge profiles must be listed in exact execution order"
            )
        if set(self.numeric_profile_order) != set(self.execution_order):
            raise ValueError(
                "legacy bridge numeric_profile_order must cover the exact 9 cases"
            )
        if len(self.adjacent_comparisons) != 7:
            raise ValueError("legacy bridge requires the seven L0-to-L7 comparisons")
        model_profiles = {
            (str(profile.get("model_id")), str(profile.get("profile_id")))
            for profile in self.profiles
        }
        expected_model_profiles = {("InceptionTimeFull", "L0")} | {
            ("CompactCNN1D", f"L{level}") for level in range(8)
        }
        if model_profiles != expected_model_profiles:
            raise ValueError("legacy bridge model/profile roster drifted")
        for profile in self.profiles:
            model_id = str(profile["model_id"])
            profile_id = str(profile["profile_id"])
            level = int(profile_id[1:])
            expected_case = (
                _BRIDGE_INCEPTION_L0
                if model_id == "InceptionTimeFull"
                else _BRIDGE_COMPACT_BY_LEVEL[level]
            )
            expected_predecessor = (
                None
                if model_id == "InceptionTimeFull" or level == 0
                else _BRIDGE_COMPACT_BY_LEVEL[level - 1]
            )
            if profile["case_id"] != expected_case:
                raise ValueError("legacy bridge case/profile identity drifted")
            if profile.get("predecessor_case_id") != expected_predecessor:
                raise ValueError("legacy bridge numeric predecessor chain drifted")

    def _validate_centered_star(self) -> None:
        if self.source_specification != "inline:legacy_bridge.effective_controls":
            raise ValueError(
                "centered star source specification must identify inline controls"
            )
        if self.runtime_status != "implemented_field_driven_centered_star_v1":
            raise ValueError("centered star runtime_status drifted")
        if dict(self.phase0) != {
            "enabled": False,
            "advisory_only": True,
            "mandatory": False,
            "affects_training_execution": False,
        }:
            raise ValueError("centered star permanently disables legacy Phase 0")
        if dict(self.budget) != {
            "case_count": 16,
            "repeat_indices": [0, 1, 2, 3, 4],
            "fold_indices": [0, 1, 2, 3, 4],
            "training_seed": 42,
            "fixed_epochs": 10,
            "early_stopping": False,
            "outer_label_checkpoint_selection": False,
            "fit_count": 400,
            "model_epoch_count": 4000,
            "phase0_fit_count": 0,
            "phase0_model_epoch_count": 0,
        }:
            raise ValueError(
                "centered star budget must be 16 cases/400 fits/4000 epochs"
            )
        if self.baseline_controls is None:
            raise ValueError("centered star requires explicit baseline_controls")
        baseline, factors, profiles, effective = (
            _materialize_centered_star_contract(
                self.baseline_controls,
                self.factor_overrides,
                self.profiles,
            )
        )
        if baseline != self.baseline_controls or factors != self.factor_overrides:
            raise ValueError("centered star stores a non-canonical factor contract")
        if profiles != self.profiles:
            raise ValueError("centered star profiles are not fully materialized")
        if _canonical_sha256(effective) != self.source_specification_sha256:
            raise ValueError("centered star inline effective-controls SHA-256 drifted")
        fits = (
            len(profiles)
            * len(self.budget["repeat_indices"])
            * len(self.budget["fold_indices"])
        )
        epochs = int(self.budget["fixed_epochs"])
        if (
            any(int(row["controls"]["fixed_epochs"]) != epochs for row in profiles)
            or int(self.budget["fit_count"]) != fits
            or int(self.budget["model_epoch_count"]) != fits * epochs
        ):
            raise ValueError(
                "centered star fit/epoch budget is internally inconsistent"
            )
        if (
            len(self.execution_order) != 16
            or tuple(row["case_id"] for row in profiles) != self.execution_order
            or self.numeric_profile_order != self.execution_order
            or self.adjacent_comparisons
        ):
            raise ValueError("centered star execution/report order contract drifted")

    def _validate_field_driven_followup(self) -> None:
        if self.source_specification != "inline:legacy_bridge.effective_controls":
            raise ValueError("field-driven follow-up requires inline controls")
        if dict(self.phase0) != {
            "enabled": False,
            "advisory_only": True,
            "mandatory": False,
            "affects_training_execution": False,
        }:
            raise ValueError("field-driven follow-up permanently disables Phase 0")
        if self.baseline_controls is None:
            raise ValueError("field-driven follow-up requires baseline_controls")
        baseline, factors, profiles, effective = (
            _materialize_field_driven_followup_contract(
                self.baseline_controls,
                self.factor_overrides,
                self.profiles,
            )
        )
        if baseline != self.baseline_controls or factors != self.factor_overrides:
            raise ValueError("field-driven follow-up stores non-canonical controls")
        if profiles != self.profiles:
            raise ValueError("field-driven follow-up profiles are not materialized")
        if _canonical_sha256(effective) != self.source_specification_sha256:
            raise ValueError("field-driven follow-up effective-controls SHA drifted")
        required_budget = {
            "case_count",
            "repeat_indices",
            "fold_indices",
            "training_seed",
            "fixed_epochs",
            "early_stopping",
            "outer_label_checkpoint_selection",
            "fit_count",
            "model_epoch_count",
            "phase0_fit_count",
            "phase0_model_epoch_count",
        }
        if set(self.budget) != required_budget:
            raise ValueError("field-driven follow-up budget key set drifted")
        repeats = _unique_int_tuple(
            self.budget["repeat_indices"], label="legacy_bridge.budget.repeat_indices"
        )
        folds = _unique_int_tuple(
            self.budget["fold_indices"], label="legacy_bridge.budget.fold_indices"
        )
        epochs = int(self.budget["fixed_epochs"])
        fits = len(profiles) * len(repeats) * len(folds)
        if (
            int(self.budget["case_count"]) != len(profiles)
            or int(self.budget["training_seed"]) != 42
            or epochs <= 0
            or self.budget["early_stopping"] is not False
            or self.budget["outer_label_checkpoint_selection"] is not False
            or int(self.budget["fit_count"]) != fits
            or int(self.budget["model_epoch_count"]) != fits * epochs
            or int(self.budget["phase0_fit_count"]) != 0
            or int(self.budget["phase0_model_epoch_count"]) != 0
            or any(int(row["controls"]["fixed_epochs"]) != epochs for row in profiles)
        ):
            raise ValueError("field-driven follow-up fit/epoch budget is inconsistent")
        if (
            tuple(row["case_id"] for row in profiles) != self.execution_order
            or self.numeric_profile_order != self.execution_order
            or self.adjacent_comparisons
        ):
            raise ValueError("field-driven follow-up execution/report order drifted")

    @property
    def uses_inline_profiles(self) -> bool:
        """Whether algorithms are selected exclusively by hash-bound controls."""

        return self.design in _FIELD_DRIVEN_BRIDGE_DESIGNS

    @property
    def centered_comparisons(self) -> tuple[Mapping[str, Any], ...]:
        """Return the fourteen legal same-model B0-to-Bk contrasts."""

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
        """Return the declared effective-controls identity for one profile."""

        value = str(profile.get("controls_sha256", ""))
        if not self.uses_inline_profiles or not _SHA256_RE.fullmatch(value):
            raise ValueError("controls_sha256 requires a field-driven profile")
        return value

    def to_dict(self) -> dict[str, Any]:
        """Return a detached serialization preserving the reviewed protocol."""

        result = {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "source_specification": self.source_specification,
            "source_specification_sha256": self.source_specification_sha256,
            "runtime_status": self.runtime_status,
            "phase0": copy.deepcopy(dict(self.phase0)),
            "budget": copy.deepcopy(dict(self.budget)),
            "execution_order": list(self.execution_order),
            "numeric_profile_order": list(self.numeric_profile_order),
            "adjacent_comparisons": list(self.adjacent_comparisons),
            "profiles": [copy.deepcopy(dict(value)) for value in self.profiles],
            "aggregation_views": list(self.aggregation_views),
            "primary_table_columns": list(self.primary_table_columns),
            "sampling_diagnostics": list(self.sampling_diagnostics),
            "restrictions": list(self.restrictions),
            "required_runtime_capabilities": list(
                self.required_runtime_capabilities
            ),
        }
        if self.uses_inline_profiles:
            result["design"] = self.design
            result["baseline_controls"] = copy.deepcopy(
                dict(self.baseline_controls or {})
            )
            result["factor_overrides"] = copy.deepcopy(dict(self.factor_overrides))
            if self.design == "centered_star_v1":
                result["centered_comparisons"] = [
                    copy.deepcopy(dict(value)) for value in self.centered_comparisons
                ]
        return result


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
    device: str | None = None
    parallel_level: str = "cases"
    continue_on_error: bool = True
    allow_parallel_deep: bool = False
    measure_operational_costs: bool = False

    def __post_init__(self) -> None:
        _unique_int_tuple(self.repeats, label="repeats")
        _unique_int_tuple(self.folds, label="folds")
        if isinstance(self.jobs, bool) or int(self.jobs) <= 0:
            raise ValueError("execution jobs must be a positive integer")
        if self.device is not None and (
            not isinstance(self.device, str) or not self.device.strip()
        ):
            raise ValueError("execution device must be null or a non-empty string")
        if self.parallel_level != "cases":
            raise ValueError("only case-level parallelism is supported")
        for name in (
            "continue_on_error",
            "allow_parallel_deep",
            "measure_operational_costs",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"execution {name} must be boolean")


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
        if not 2 <= int(self.calibration_bins) <= 100:
            raise ValueError("calibration_bins must lie in 2..100")
        if not self.figure_modules or any(
            not str(value).strip() for value in self.figure_modules
        ):
            raise ValueError("figure_modules must contain non-empty module names")
        if not math.isfinite(float(self.classification_tsne_perplexity)) or float(
            self.classification_tsne_perplexity
        ) <= 0:
            raise ValueError("classification_tsne_perplexity must be positive")
        if int(self.classification_tsne_max_samples) < 3:
            raise ValueError("classification_tsne_max_samples must be at least 3")
        if int(self.classification_roc_macro_grid_points) < 2:
            raise ValueError("classification_roc_macro_grid_points must be at least 2")
        if int(self.classification_score_histogram_bins) < 2:
            raise ValueError("classification_score_histogram_bins must be at least 2")


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
        if self.study.study_id in {
            _BRIDGE_STUDY_ID,
            _BRIDGE_STAR_STUDY_ID,
            _BRIDGE_V3_STUDY_ID,
        } and self.legacy_bridge is None:
            raise ValueError(
                "Stage 3 legacy bridge study requires legacy bridge protocol metadata"
            )
        if (
            self.legacy_bridge is not None
            and self.study.study_id
            not in {
                _BRIDGE_STUDY_ID,
                _BRIDGE_STAR_STUDY_ID,
                _BRIDGE_V3_STUDY_ID,
            }
        ):
            raise ValueError(
                "legacy_bridge metadata is reserved for the registered Stage 3 study"
            )
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
            if self.catalog.scope in {"ordinary_active", "ordinary_13"}:
                entries = {case.catalog_entry for case in self.cases}
                if not entries:
                    raise ValueError("ordinary catalog_sweep requires catalog entries")
            if self.catalog.scope == "matched_ensemble_pair":
                entries = {case.catalog_entry for case in self.cases}
                if len(self.cases) != 2 or len(entries) != 2:
                    raise ValueError(
                        "matched_ensemble_pair requires exactly two distinct "
                        "catalog entries"
                    )
                if any(case.formal_profile is not None for case in self.cases):
                    raise ValueError(
                        "matched_ensemble_pair cannot add formal profiles; the "
                        "registered ensemble factor must be isolated"
                    )
                matched_overrides = tuple(
                    copy.deepcopy(dict(case.overrides)) for case in self.cases
                )
                allowed_matched_paths = {
                    "signal.dl_resampling.enabled",
                    "signal.dl_resampling.target_fs_hz",
                    "signal.normalization.raw_ppg",
                    "signal.normalization.raw_imu",
                    "windows.raw_dl.length_s",
                    "windows.raw_dl.hop_s",
                    "training.optimizer",
                    "training.batch_size",
                    "training.sampler",
                    "training.class_weighting",
                    "training.class_count_basis",
                    "aggregation.balance_line",
                }
                if (
                    any(value != matched_overrides[0] for value in matched_overrides[1:])
                    or not set(matched_overrides[0]) <= allowed_matched_paths
                ):
                    raise ValueError(
                        "matched_ensemble_pair cannot add unequal or model-changing "
                        "overrides; identical registered-input profile overrides "
                        "are required"
                    )
            if self.legacy_bridge is not None:
                if not isinstance(self.legacy_bridge, LegacyBridgeSpec):
                    raise TypeError("legacy_bridge must be LegacyBridgeSpec or None")
                if self.catalog.scope != "selected_ordinary":
                    raise ValueError(
                        "legacy bridge requires catalog scope=selected_ordinary"
                    )
                expected_study_id, expected_reference = {
                    "cumulative_chain_v1": (
                        _BRIDGE_STUDY_ID,
                        "compact_cnn__l0_legacy64_w15_fixed10",
                    ),
                    "centered_star_v1": (
                        _BRIDGE_STAR_STUDY_ID,
                        "compact_cnn__b0_star_fixed10",
                    ),
                    "field_driven_followup_v1": (
                        _BRIDGE_V3_STUDY_ID,
                        "compact_cnn__b0_b2_r5_fixed10",
                    ),
                }[self.legacy_bridge.design]
                if (
                    self.study.study_id != expected_study_id
                    or self.study.decision_role != "ablation"
                    or self.study.reference_case_id != expected_reference
                ):
                    raise ValueError(
                        "legacy bridge requires the frozen ablation reference case"
                    )
                budget = self.legacy_bridge.budget
                if len(self.cases) != int(budget["case_count"]):
                    raise ValueError(
                        "legacy bridge case count differs from its frozen budget"
                    )
                if self.execution.repeats != tuple(budget["repeat_indices"]):
                    raise ValueError(
                        "legacy bridge execution repeats differ from its frozen budget"
                    )
                if self.execution.folds != tuple(budget["fold_indices"]):
                    raise ValueError(
                        "legacy bridge execution folds differ from its frozen budget"
                    )
                if self.search.selection_seed != int(budget["training_seed"]):
                    raise ValueError(
                        "legacy bridge selection/training seed declaration drifted"
                    )
                if self.execution.jobs != 1 or self.execution.continue_on_error:
                    raise ValueError(
                        "legacy bridge requires serial fail-fast case execution"
                    )
                if self.legacy_bridge.uses_inline_profiles and (
                    not re.fullmatch(r"cuda(?::\d+)?", str(self.execution.device))
                    or self.execution.allow_parallel_deep
                    or self.execution.measure_operational_costs
                ):
                    raise ValueError(
                        "field-driven bridge requires one serial CUDA device and no "
                        "operational-cost reruns"
                    )
                profile_case_ids = tuple(
                    str(profile["catalog_case_id"])
                    for profile in self.legacy_bridge.profiles
                )
                declared_case_ids = tuple(case.case_id for case in self.cases)
                if profile_case_ids != declared_case_ids:
                    raise ValueError(
                        "legacy bridge catalog cases must match protocol profiles "
                        "in exact execution order"
                    )
                for case, profile in zip(
                    self.cases, self.legacy_bridge.profiles
                ):
                    expected_entry = (
                        "inception_full"
                        if profile["model_id"] == "InceptionTimeFull"
                        else "compact_cnn"
                    )
                    if (
                        case.catalog_entry != expected_entry
                        or case.output_group != "raw"
                    ):
                        raise ValueError(
                            "legacy bridge catalog architecture/route differs "
                            "from its protocol profile"
                        )
                if any(
                    case.overrides or case.formal_profile is not None
                    for case in self.cases
                ):
                    raise ValueError(
                        "legacy bridge profiles cannot masquerade as canonical "
                        "catalog overrides"
                    )
            return
        if self.legacy_bridge is not None:
            raise ValueError("legacy_bridge is valid only for catalog_sweep")
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
                "measure_operational_costs": (
                    self.execution.measure_operational_costs
                ),
            },
            "output": {"root": self.output.root},
            "report": {
                "top_k": self.report.top_k,
                "write_html": self.report.write_html,
                "write_static_figures": self.report.write_static_figures,
                "calibration_bins": self.report.calibration_bins,
                "figure_modules": list(self.report.figure_modules),
                "compact_mean_sd": self.report.compact_mean_sd,
                "write_excel_workbook": self.report.write_excel_workbook,
                "classification_tsne_random_state": (
                    self.report.classification_tsne_random_state
                ),
                "classification_tsne_perplexity": (
                    self.report.classification_tsne_perplexity
                ),
                "classification_tsne_max_samples": (
                    self.report.classification_tsne_max_samples
                ),
                "classification_roc_macro_grid_points": (
                    self.report.classification_roc_macro_grid_points
                ),
                "classification_score_histogram_bins": (
                    self.report.classification_score_histogram_bins
                ),
            },
        }
        if self.execution.device is not None:
            payload["execution"]["device"] = self.execution.device
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
        result = {
            "schema_version": payload["schema_version"],
            "study": payload["study"],
            "catalog": payload["catalog"],
            "search": payload["search"],
            "cases": cases,
            "execution": payload["execution"],
            "output": payload["output"],
            "report": payload["report"],
        }
        if self.legacy_bridge is not None:
            result["legacy_bridge"] = self.legacy_bridge.to_dict()
        return result


def execution_from_mapping(value: Mapping[str, Any] | None) -> ExecutionSpec:
    payload = _strict_mapping(
        value or {},
        label="execution",
        required=frozenset(),
        optional=frozenset(
            {
                "repeats",
                "folds",
                "jobs",
                "device",
                "parallel_level",
                "continue_on_error",
                "allow_parallel_deep",
                "measure_operational_costs",
            }
        ),
    )
    return ExecutionSpec(
        repeats=_unique_int_tuple(payload.get("repeats"), label="repeats"),
        folds=_unique_int_tuple(payload.get("folds"), label="folds"),
        jobs=int(payload.get("jobs", 1)),
        device=payload.get("device"),
        parallel_level=str(payload.get("parallel_level", "cases")),
        continue_on_error=payload.get("continue_on_error", True),
        allow_parallel_deep=payload.get("allow_parallel_deep", False),
        measure_operational_costs=payload.get("measure_operational_costs", False),
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
        scope=payload.get("scope", "ordinary_active"),
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


def _field_driven_spec_from_mapping(
    value: Mapping[str, Any], *, design: str
) -> LegacyBridgeSpec:
    payload = _strict_mapping(
        value,
        label="legacy_bridge",
        required=frozenset(
            {
                "schema_version",
                "design",
                "protocol_id",
                "source_specification",
                "source_specification_sha256",
                "runtime_status",
                "budget",
                "execution_order",
                "profiles",
                "baseline_controls",
                "factor_overrides",
                "aggregation_views",
                "primary_table_columns",
                "sampling_diagnostics",
                "restrictions",
                "required_runtime_capabilities",
            }
        ),
        optional=frozenset(
            {
                "phase0",
                "numeric_profile_order",
                "adjacent_comparisons",
                "centered_comparisons",
            }
        ),
    )
    phase0 = _strict_mapping(
        payload.get(
            "phase0",
            {
                "enabled": False,
                "advisory_only": True,
                "mandatory": False,
                "affects_training_execution": False,
            },
        ),
        label="legacy_bridge.phase0",
        required=frozenset(
            {
                "enabled",
                "advisory_only",
                "mandatory",
                "affects_training_execution",
            }
        ),
    )
    materializer = (
        _materialize_centered_star_contract
        if design == "centered_star_v1"
        else _materialize_field_driven_followup_contract
    )
    baseline, factors, profiles, _effective = materializer(
        payload["baseline_controls"], payload["factor_overrides"], payload["profiles"]
    )
    execution_order = _string_tuple(
        payload["execution_order"],
        label="legacy_bridge.execution_order",
    )
    numeric_order = _string_tuple(
        payload.get("numeric_profile_order", list(execution_order)),
        label="legacy_bridge.numeric_profile_order",
    )
    adjacent_raw = payload.get("adjacent_comparisons", [])
    adjacent = (
        ()
        if adjacent_raw == []
        else _string_tuple(
            adjacent_raw,
            label="legacy_bridge.adjacent_comparisons",
        )
    )
    spec = LegacyBridgeSpec(
        schema_version=payload["schema_version"],
        protocol_id=payload["protocol_id"],
        source_specification=payload["source_specification"],
        source_specification_sha256=payload["source_specification_sha256"],
        runtime_status=payload["runtime_status"],
        phase0=copy.deepcopy(phase0),
        budget=copy.deepcopy(dict(payload["budget"])),
        execution_order=execution_order,
        numeric_profile_order=numeric_order,
        adjacent_comparisons=adjacent,
        profiles=tuple(copy.deepcopy(profiles)),
        aggregation_views=_string_tuple(
            payload["aggregation_views"],
            label="legacy_bridge.aggregation_views",
        ),
        primary_table_columns=_string_tuple(
            payload["primary_table_columns"],
            label="legacy_bridge.primary_table_columns",
        ),
        sampling_diagnostics=_string_tuple(
            payload["sampling_diagnostics"],
            label="legacy_bridge.sampling_diagnostics",
        ),
        restrictions=_string_tuple(
            payload["restrictions"],
            label="legacy_bridge.restrictions",
        ),
        required_runtime_capabilities=_string_tuple(
            payload["required_runtime_capabilities"],
            label="legacy_bridge.required_runtime_capabilities",
        ),
        design=design,
        baseline_controls=copy.deepcopy(baseline),
        factor_overrides=copy.deepcopy(factors),
    )
    if "centered_comparisons" in payload:
        if design != "centered_star_v1" or payload[
            "centered_comparisons"
        ] != [dict(value) for value in spec.centered_comparisons]:
            raise ValueError("legacy bridge centered_comparisons drifted")
    return spec


def legacy_bridge_spec_from_mapping(
    value: Mapping[str, Any],
) -> LegacyBridgeSpec:
    """Parse the strict executable legacy/V2 bridge protocol metadata."""

    if isinstance(value, Mapping) and value.get("design") in (
        _FIELD_DRIVEN_BRIDGE_DESIGNS
    ):
        return _field_driven_spec_from_mapping(
            value, design=str(value["design"])
        )

    payload = _strict_mapping(
        value,
        label="legacy_bridge",
        required=frozenset(
            {
                "schema_version",
                "protocol_id",
                "source_specification",
                "source_specification_sha256",
                "runtime_status",
                "phase0",
                "budget",
                "execution_order",
                "numeric_profile_order",
                "adjacent_comparisons",
                "profiles",
                "aggregation_views",
                "primary_table_columns",
                "sampling_diagnostics",
                "restrictions",
                "required_runtime_capabilities",
            }
        ),
    )
    phase0 = _strict_mapping(
        payload["phase0"],
        label="legacy_bridge.phase0",
        required=frozenset(
            {
                "advisory_only",
                "mandatory",
                "affects_training_execution",
                "manifest_path",
                "manifest_expected_rows",
                "split_path",
                "static_discovery_globs",
                "historical_label_mapping",
                "static_expected_record_count",
                "static_expected_participant_count",
                "static_expected_class_counts",
                "static_required_roles",
                "required_channel_order",
                "manifest_audit_checks",
                "historical_discovery_checks",
                "imu_audit_checks",
                "unit_correction_policy",
                "cache_audit_checks",
                "split_audit_checks",
                "split_runtime_recompute",
                "audit_decisions",
                "training_input_source",
                "historical_cache_use",
                "historical_cache_mismatch_training_effect",
                "audit_outputs",
                "advisory_findings",
            }
        ),
        optional=frozenset({"enabled"}),
    )
    phase0.setdefault("enabled", False)
    if not isinstance(phase0["enabled"], bool):
        raise TypeError("legacy bridge Phase 0 enabled must be boolean")
    if phase0["advisory_only"] is not True:
        raise ValueError("legacy bridge Phase 0 must be advisory-only")
    if phase0["mandatory"] is not False:
        raise ValueError("legacy bridge Phase 0 must not be mandatory")
    if phase0["affects_training_execution"] is not False:
        raise ValueError("legacy bridge Phase 0 must not affect training execution")
    expected_phase0_scalars = {
        "manifest_path": "manifests/internal_records_v2.csv",
        "manifest_expected_rows": 261,
        "split_path": "splits/sgkf5_repeated_grouped_5x5_v2.csv",
        "static_expected_record_count": 145,
        "static_expected_participant_count": 29,
        "training_input_source": "fresh_current_raw_csv_bytes_independent_of_phase0",
        "historical_cache_use": "audit_only_never_training",
    }
    for key, expected in expected_phase0_scalars.items():
        observed = phase0[key]
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(
                f"legacy bridge Phase 0 {key} must be {expected!r}"
            )
    if phase0["static_discovery_globs"] != [
        "StudyData/*.csv",
        "TestDataYoungers/*.csv",
    ]:
        raise ValueError("legacy bridge Phase 0 discovery globs drifted")
    if phase0["static_required_roles"] != ["B", "R1", "R2", "R3", "R4"]:
        raise ValueError("legacy bridge Phase 0 static roles drifted")
    if phase0["required_channel_order"] != [
        "RED",
        "IR",
        "AX",
        "AY",
        "AZ",
        "GX",
        "GY",
        "GZ",
    ]:
        raise ValueError("legacy bridge Phase 0 channel order drifted")
    historical_labels = _strict_mapping(
        phase0["historical_label_mapping"],
        label="legacy_bridge.phase0.historical_label_mapping",
        required=frozenset(
            {
                "StudyData/FRAILTY-STATUS=2",
                "StudyData/FRAILTY-STATUS=3",
                "TestDataYoungers",
            }
        ),
    )
    if historical_labels != {
        "StudyData/FRAILTY-STATUS=2": "Pre-Frail",
        "StudyData/FRAILTY-STATUS=3": "Robust/Non-Frail",
        "TestDataYoungers": "Young",
    }:
        raise ValueError("legacy bridge Phase 0 historical label mapping drifted")
    for key in (
        "manifest_audit_checks",
        "historical_discovery_checks",
        "imu_audit_checks",
        "cache_audit_checks",
        "split_audit_checks",
    ):
        _string_tuple(phase0[key], label=f"legacy_bridge.phase0.{key}")
    if phase0["unit_correction_policy"] != "report_only_no_automatic_correction":
        raise ValueError("legacy bridge Phase 0 unit correction policy drifted")
    if phase0["split_runtime_recompute"] is not False:
        raise ValueError("legacy bridge Phase 0 split runtime recompute must be false")
    if phase0["audit_decisions"] != [
        "PASS",
        "STOP",
        "PASS_WITH_DECLARED_LIMITATIONS",
    ]:
        raise ValueError("legacy bridge Phase 0 audit decisions drifted")
    if phase0["historical_cache_mismatch_training_effect"] != "none":
        raise ValueError(
            "legacy bridge historical cache mismatch must not affect fresh training"
        )
    class_counts = _strict_mapping(
        phase0["static_expected_class_counts"],
        label="legacy_bridge.phase0.static_expected_class_counts",
        required=frozenset({"Pre-Frail", "Robust/Non-Frail", "Young"}),
    )
    if class_counts != {"Pre-Frail": 9, "Robust/Non-Frail": 12, "Young": 8}:
        raise ValueError("legacy bridge Phase 0 expected class counts drifted")
    expected_audit_outputs = (
        "artifacts/audit/legacy_v2_manifest_record_diff.csv",
        "artifacts/audit/legacy_v2_source_hash_audit.csv",
        "artifacts/audit/legacy_v2_source_hash_audit.json",
        "artifacts/audit/legacy_v2_channel_qc.csv",
        "artifacts/audit/legacy_v2_participant_alias_map.csv",
        "artifacts/audit/legacy_v2_imu_unit_ekf_audit.csv",
        "artifacts/audit/legacy_v2_cache_audit.json",
        "artifacts/audit/legacy_v2_split_audit.json",
        "artifacts/audit/LEGACY_V2_PHASE0_DATA_AUDIT.md",
    )
    if _string_tuple(
        phase0["audit_outputs"], label="legacy_bridge.phase0.audit_outputs"
    ) != expected_audit_outputs:
        raise ValueError("legacy bridge Phase 0 audit output contract drifted")
    advisory_findings = _string_tuple(
        phase0["advisory_findings"],
        label="legacy_bridge.phase0.advisory_findings",
    )
    if len(advisory_findings) != 7:
        raise ValueError("legacy bridge Phase 0 requires seven advisory findings")

    budget = _strict_mapping(
        payload["budget"],
        label="legacy_bridge.budget",
        required=frozenset(
            {
                "case_count",
                "repeat_indices",
                "fold_indices",
                "training_seed",
                "fixed_epochs",
                "early_stopping",
                "outer_label_checkpoint_selection",
                "fit_count",
                "model_epoch_count",
                "phase0_fit_count",
                "phase0_model_epoch_count",
            }
        ),
    )
    expected_budget = {
        "case_count": 9,
        "repeat_indices": [0],
        "fold_indices": [0, 1, 2, 3, 4],
        "training_seed": 42,
        "fixed_epochs": 10,
        "early_stopping": False,
        "outer_label_checkpoint_selection": False,
        "fit_count": 45,
        "model_epoch_count": 450,
        "phase0_fit_count": 0,
        "phase0_model_epoch_count": 0,
    }
    for key, expected in expected_budget.items():
        observed = budget[key]
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(f"legacy bridge budget {key} must be {expected!r}")

    profiles_raw = payload["profiles"]
    if isinstance(profiles_raw, (str, bytes, Mapping)) or not isinstance(
        profiles_raw, (list, tuple)
    ):
        raise TypeError("legacy_bridge.profiles must be a list")
    profiles: list[Mapping[str, Any]] = []
    for index, raw_profile in enumerate(profiles_raw):
        profile = _strict_mapping(
            raw_profile,
            label=f"legacy_bridge.profiles[{index}]",
            required=frozenset(
                {
                    "case_id",
                    "catalog_case_id",
                    "model_id",
                    "profile_id",
                    "predecessor_case_id",
                    "contract",
                    "interpretation",
                }
            ),
        )
        if not isinstance(profile["case_id"], str) or not profile["case_id"].strip():
            raise ValueError("legacy bridge profile case_id must be non-empty")
        _safe_identifier(
            profile["catalog_case_id"],
            label="legacy bridge profile catalog_case_id",
        )
        if profile["model_id"] not in {"CompactCNN1D", "InceptionTimeFull"}:
            raise ValueError("legacy bridge profile model_id is unsupported")
        if profile["profile_id"] not in {f"L{value}" for value in range(8)}:
            raise ValueError("legacy bridge profile_id must be L0..L7")
        predecessor = profile["predecessor_case_id"]
        if predecessor is not None and (
            not isinstance(predecessor, str) or not predecessor.strip()
        ):
            raise ValueError(
                "legacy bridge predecessor_case_id must be null or non-empty"
            )
        profile["contract"] = list(
            _string_tuple(
                profile["contract"],
                label=f"legacy_bridge.profiles[{index}].contract",
            )
        )
        if (
            not isinstance(profile["interpretation"], str)
            or not profile["interpretation"].strip()
        ):
            raise ValueError("legacy bridge profile interpretation must be non-empty")
        profiles.append(profile)

    aggregation_views = _string_tuple(
        payload["aggregation_views"],
        label="legacy_bridge.aggregation_views",
    )
    required_views = {
        "legacy_window_balanced_direct_participant_mean",
        "v2_role_balanced_window_file_role_family_participant_mean",
    }
    if not required_views <= set(aggregation_views):
        raise ValueError("legacy bridge requires both legacy and V2 aggregation views")
    primary_columns = _string_tuple(
        payload["primary_table_columns"],
        label="legacy_bridge.primary_table_columns",
    )
    required_columns = {
        "model",
        "profile",
        "BA_legacy_aggregation",
        "BA_v2_aggregation",
        "macroF1_legacy_aggregation",
        "macroF1_v2_aggregation",
        "worst_class_F1",
        "delta_from_previous_numeric_profile",
    }
    if not required_columns <= set(primary_columns):
        raise ValueError("legacy bridge primary table omits required columns")
    sampling_diagnostics = _string_tuple(
        payload["sampling_diagnostics"],
        label="legacy_bridge.sampling_diagnostics",
    )
    expected_sampling_diagnostics = (
        "dataset_row_count",
        "draw_count",
        "unique_row_draw_count",
        "duplicate_draw_fraction",
        "never_drawn_row_fraction",
        "draw_counts_by_participant",
        "draw_counts_by_class",
        "draw_counts_by_B_R_family",
        "draw_counts_by_file",
        "class_weight_vector",
        "sampler_identity",
    )
    if sampling_diagnostics != expected_sampling_diagnostics:
        raise ValueError("legacy bridge sampling diagnostic contract drifted")
    restrictions = _string_tuple(
        payload["restrictions"],
        label="legacy_bridge.restrictions",
    )
    required_restrictions = {
        "no_additional_cases_repeats_models_or_hyperparameter_search",
        "all_aggregation_views_from_same_OOF_window_probabilities",
        "sampling_diagnostics_saved_for_every_fold_and_epoch",
    }
    if not required_restrictions <= set(restrictions):
        raise ValueError("legacy bridge restrictions omit a mandatory guard")

    return LegacyBridgeSpec(
        schema_version=payload["schema_version"],
        protocol_id=payload["protocol_id"],
        source_specification=payload["source_specification"],
        source_specification_sha256=payload["source_specification_sha256"],
        runtime_status=payload["runtime_status"],
        phase0=copy.deepcopy(phase0),
        budget=copy.deepcopy(budget),
        execution_order=_string_tuple(
            payload["execution_order"],
            label="legacy_bridge.execution_order",
        ),
        numeric_profile_order=_string_tuple(
            payload["numeric_profile_order"],
            label="legacy_bridge.numeric_profile_order",
        ),
        adjacent_comparisons=_string_tuple(
            payload["adjacent_comparisons"],
            label="legacy_bridge.adjacent_comparisons",
        ),
        profiles=tuple(copy.deepcopy(profiles)),
        aggregation_views=aggregation_views,
        primary_table_columns=primary_columns,
        sampling_diagnostics=sampling_diagnostics,
        restrictions=restrictions,
        required_runtime_capabilities=_string_tuple(
            payload["required_runtime_capabilities"],
            label="legacy_bridge.required_runtime_capabilities",
        ),
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
