"""冻结十字段注册表与显式有效性模型编码 / Frozen registry and mask encoding.

English: The public builders implement a content-addressed FeatureVectorV1
allowlist composed from complete registered feature groups and the variable-K
146-channel window-matrix contract. Technical or unknown predictor fields
are rejected rather than silently appended.

中文：公共构建器以完整注册 feature groups 组合内容寻址的 FeatureVectorV1 allowlist，
并实现 146 通道、可变 K 的纯窗口特征矩阵；技术字段或未知 predictor
会被拒绝，而非静默追加。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from ..contracts import FeatureVectorV1, SignalRoute
from ..signal.morphology import MORPHOLOGY_NAMES
from ..signal.optical import OPTICAL_SCHEMA_VERSION
from ..signal.prv import SPECTRAL_METRICS, TIME_METRICS
from .engineering import (
    ENGINEERING_SCHEMA_VERSION,
    WELCH_MAX_SEGMENT_SAMPLES,
    WELCH_MIN_SEGMENT_SAMPLES,
    WELCH_SECONDS,
    WELCH_WINDOW,
    EngineeringExtraction,
    engineering_feature_names,
    validate_engineering_extraction,
)


DIRECT_ROUTES = (SignalRoute.DIRECT.value, SignalRoute.IDENTITY.value)
RATE_ROUTES = DIRECT_ROUTES + (SignalRoute.ARTIFACT_RATE_ONLY.value,)

# The ordered matrix is exactly the chronological 146-feature window sequence.
# File-level context and validity indicators remain provenance, not predictors.
FEATURE_GROUP_ORDER = (
    "ppi_basic_rate",
    "hrv_time_domain",
    "hrv_spectral",
    "hrv_nonlinear",
    "morphology",
    "dual_optical",
    "engineering_summary",
)
FORMAL_REGISTRY_VERSION = "feature_vector_282_v3"
ORDERED_MATRIX_SCHEMA_VERSION = "ordered_window_feature_matrix_d146_variable_k_v1"
ENGINEERING_FEATURE_COUNT = 115
WINDOW_FEATURE_COUNT = 146
MAX_ORDERED_MATRIX_K = None  # compatibility export: no dataset-wide K cap.
MISSING_POLICY = "NaN_internal/null_JSON_with_parallel_validity_false"

# These are migration documentation/profiles, not a second pair of executable
# switches.  Callers select the resulting groups through ``enabled_groups``.
LEGACY_FEATURE_GROUP_PROFILES = {
    "PPI": ("ppi_basic_rate",),
    "HRV": (
        "ppi_basic_rate",
        "hrv_time_domain",
        "hrv_spectral",
        "hrv_nonlinear",
    ),
    "morphology": ("morphology", "dual_optical"),
    "morphology_ppi_hrv": (
        "ppi_basic_rate",
        "hrv_time_domain",
        "hrv_spectral",
        "hrv_nonlinear",
        "morphology",
        "dual_optical",
    ),
}

def canonicalize_feature_groups(groups: Sequence[str] | None) -> tuple[str, ...]:
    """Validate and canonicalize a non-empty feature-group selection.

    Input ordering does not create a scientifically different schema: aliases,
    duplicates, and arbitrary ordering resolve to the one frozen group order.
    """

    if groups is None:
        return FEATURE_GROUP_ORDER
    if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
        raise ValueError("features.enabled_groups must be a non-empty sequence")
    aliases = {
        "ppi": "ppi_basic_rate",
        "basic_rate": "ppi_basic_rate",
        "hrv_time": "hrv_time_domain",
        "time_hrv": "hrv_time_domain",
        "spectral_hrv": "hrv_spectral",
        "nonlinear_hrv": "hrv_nonlinear",
        "optical": "dual_optical",
        "engineering": "engineering_summary",
    }
    selected: set[str] = set()
    for raw in groups:
        name = str(raw).strip().lower().replace("-", "_")
        name = aliases.get(name, name)
        if name not in FEATURE_GROUP_ORDER:
            raise ValueError(f"unknown feature group: {raw!r}")
        selected.add(name)
    if not selected:
        raise ValueError("features.enabled_groups must not be empty")
    return tuple(name for name in FEATURE_GROUP_ORDER if name in selected)

def legacy_feature_groups(profile: str) -> tuple[str, ...]:
    """Return the unified-group migration for one old classifier profile."""

    key = str(profile).strip()
    aliases = {
        "ppi": "PPI",
        "hrv": "HRV",
        "morphology": "morphology",
        "morphology_ppi_hrv": "morphology_ppi_hrv",
        "morphology_ppi_hrv_filelevel": "morphology_ppi_hrv",
    }
    key = aliases.get(key.lower(), key)
    if key not in LEGACY_FEATURE_GROUP_PROFILES:
        raise ValueError(f"unknown legacy feature profile: {profile!r}")
    return LEGACY_FEATURE_GROUP_PROFILES[key]

def _prv_group(name: str) -> str:
    """Assign each PRV field to exactly one independently selectable group."""

    if name in {
        "accepted_interval_count",
        "accepted_duration_s",
        "ppi_mean_s",
        "ppi_median_s",
        "ppi_sd_s",
        "ppi_iqr_s",
        "ppi_mad_s",
        "ppi_cv",
        "hr_mean_bpm",
        "hr_median_bpm",
        "hr_sd_bpm",
    }:
        return "ppi_basic_rate"
    if name in {"sdnn_s", "rmssd_s", "sdsd_s", "nn50_count", "pnn50"}:
        return "hrv_time_domain"
    if name in set(SPECTRAL_METRICS):
        return "hrv_spectral"
    if name in {"sd1_s", "sd2_s", "sd1_sd2_ratio", "sample_entropy"}:
        return "hrv_nonlinear"
    raise RuntimeError(f"PRV registry group is undefined for {name!r}")

@dataclass(frozen=True)
class FeatureDefinition:
    """规范要求的十字段加 route/group / Ten required fields plus route/group."""

    canonical_name: str
    formula_algorithm: str
    units: str
    source_signal_view: str
    endpoint_role_eligibility: str
    level: str
    aggregation_rule: str
    validity_rule: str
    missing_value_policy: str
    provenance_version: str
    eligible_routes: tuple[str, ...]
    group: str

    @property
    def name(self) -> str:
        """兼容只读名称访问 / Read-only compatibility alias."""

        return self.canonical_name

    def validate(self) -> None:
        """拒绝空字段及未注册 route / Reject empty fields and unknown routes."""

        required = (
            self.canonical_name,
            self.formula_algorithm,
            self.units,
            self.source_signal_view,
            self.endpoint_role_eligibility,
            self.level,
            self.aggregation_rule,
            self.validity_rule,
            self.missing_value_policy,
            self.provenance_version,
            self.group,
        )
        if any(not str(value).strip() for value in required):
            raise ValueError(f"feature definition contains an empty field: {self.canonical_name}")
        known = {item.value for item in SignalRoute}
        if not self.eligible_routes or not set(self.eligible_routes).issubset(known):
            raise ValueError(f"feature has invalid eligible_routes: {self.canonical_name}")

@dataclass(frozen=True)
class FeatureRegistry:
    """内容寻址的冻结注册表 / Content-addressed frozen feature registry."""

    definitions: tuple[FeatureDefinition, ...]
    schema_version: str
    sha256: str

    @property
    def names(self) -> tuple[str, ...]:
        """返回唯一有序 canonical names / Return ordered canonical names."""

        return tuple(item.canonical_name for item in self.definitions)

    def validate(self) -> None:
        """验证十字段、唯一顺序和内容哈希 / Validate fields, order, and hash."""

        if not self.schema_version or not self.definitions:
            raise ValueError("feature registry must be non-empty and versioned")
        for item in self.definitions:
            item.validate()
        if len(self.names) != len(set(self.names)):
            raise ValueError("feature registry contains duplicate canonical names")
        if self.sha256 != _hash_definitions(self.definitions):
            raise ValueError("feature registry SHA256 does not match its definitions")

def _hash_definitions(definitions: Sequence[FeatureDefinition]) -> str:
    """规范 JSON 哈希 / Hash definitions using canonical JSON."""

    payload = [item.__dict__ for item in definitions]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()

def _definition(
    name: str,
    *,
    formula: str,
    units: str,
    source: str,
    eligibility: str,
    level: str,
    aggregation: str,
    validity: str,
    routes: tuple[str, ...],
    group: str,
) -> FeatureDefinition:
    """统一构造完整定义 / Construct one complete definition."""

    return FeatureDefinition(
        canonical_name=name,
        formula_algorithm=formula,
        units=units,
        source_signal_view=source,
        endpoint_role_eligibility=eligibility,
        level=level,
        aggregation_rule=aggregation,
        validity_rule=validity,
        missing_value_policy=MISSING_POLICY,
        provenance_version="feature_registry_formula_v2",
        eligible_routes=routes,
        group=group,
    )

def _prv_unit(name: str) -> str:
    """PRV canonical unit / PRV 规范单位。"""

    if name.endswith("count") or name == "accepted_interval_count":
        return "count"
    if name in {"coverage", "pnn50", "ppi_cv", "sd1_sd2_ratio", "sample_entropy", "lf_normalized", "hf_normalized"}:
        return "dimensionless"
    if name.startswith("hr_"):
        return "beats_per_minute"
    if name.endswith("power_s2"):
        return "seconds_squared"
    if name == "lf_hf_ratio":
        return "dimensionless"
    return "seconds"

def default_registry() -> FeatureRegistry:
    """构建默认全量 282-field allowlist / Build the default complete allowlist."""

    definitions: list[FeatureDefinition] = []
    for name in TIME_METRICS + SPECTRAL_METRICS:
        if name == "coverage":
            continue
        definitions.append(
            _definition(
                f"prv.{name}",
                formula=f"compute_prv_v1:{name}",
                units=_prv_unit(name),
                source="peak_timestamps+interval_indices+x_filter_or_x_ar",
                eligibility="Q_rate; duration/coverage/role gates declared by metric",
                level="file",
                aggregation="accepted_interval_statistic",
                validity="metric-specific minimum count, continuity, duration and finite result",
                routes=RATE_ROUTES,
                group=_prv_group(name),
            )
        )
    for name in MORPHOLOGY_NAMES:
        for statistic in ("median", "mad"):
            definitions.append(
                _definition(
                    f"morphology.{name}_{statistic}",
                    formula=f"valley_to_valley_baseline_v1:{name}:{statistic}",
                    units="signal_unit_or_seconds_or_derived",
                    source="x_filter",
                    eligibility="Q_morph; direct or identity only",
                    level="file",
                    aggregation=f"valid_beat_{statistic}",
                    validity="at least three valid accepted beats and finite statistic",
                    routes=DIRECT_ROUTES,
                    group="morphology",
                )
            )
    optical_names = (
        "red_ac_median",
        "ir_ac_median",
        "red_dc_median",
        "ir_dc_median",
        "red_pi_median",
        "ir_pi_median",
        "red_ir_ac_ratio_median",
        "red_ir_dc_ratio_median",
        "ratio_of_ratios_median",
        "red_ir_zero_lag_correlation",
        "red_ir_max_xcorr",
        "red_ir_xcorr_lag_s",
    )
    optical_formulas = {
        "red_ac_median": "median(common_paired_valid_red_ac)",
        "ir_ac_median": "median(common_paired_valid_ir_ac)",
        "red_dc_median": "median(common_paired_valid_red_dc)",
        "ir_dc_median": "median(common_paired_valid_ir_dc)",
        "red_pi_median": "AC_RED/(abs(DC_RED)+1e-12)",
        "ir_pi_median": "AC_IR/(abs(DC_IR)+1e-12)",
        "red_ir_ac_ratio_median": "AC_RED/(AC_IR+1e-12)",
        "red_ir_dc_ratio_median": "abs(DC_RED)/(abs(DC_IR)+1e-12)",
        "ratio_of_ratios_median": "PI_RED/PI_IR",
        "red_ir_zero_lag_correlation": "pearson(population_z_RED,population_z_IR)",
        "red_ir_max_xcorr": "max_normalized_xcorr[-0.5s,+0.5s]_inclusive",
        "red_ir_xcorr_lag_s": "tau_star_of_max_normalized_xcorr_seconds",
    }
    for name in optical_names:
        definitions.append(
            _definition(
                f"optical.{name}",
                formula=f"{OPTICAL_SCHEMA_VERSION}:{optical_formulas[name]}",
                units="signal_unit_or_ratio_or_seconds",
                source="x_native+x_filter+independent_red_ir_pulses",
                eligibility="Q_morph; dual wavelength; direct or identity only",
                level="file",
                aggregation="common_paired_beat_ac_dc_medians_then_ratios_or_file_agreement",
                validity=(
                    "common paired-valid RED/IR beats, finite denominators, "
                    "minimum three-beat support, or finite standardized waveform agreement"
                ),
                routes=DIRECT_ROUTES,
                group="dual_optical",
            )
        )
    for name in engineering_feature_names():
        source = "x_filter" if name.startswith("ppg_") else "imu_processed"
        routes = DIRECT_ROUTES if name.startswith("ppg_") else RATE_ROUTES
        eligibility = (
            "direct or identity complete 10 s windows"
            if name.startswith("ppg_")
            else "all routes with valid processed IMU complete 10 s windows"
        )
        uses_welch = any(
            token in name
            for token in (
                ".total_power",
                ".normalized_spectral_entropy",
                ".dominant_frequency_hz",
                ".spectral_centroid_hz",
                ".bandpower_",
            )
        )
        welch_contract = (
            f":welch_window={WELCH_WINDOW}:"
            f"nperseg=min(N,max({WELCH_MIN_SEGMENT_SAMPLES},"
            f"min({WELCH_MAX_SEGMENT_SAMPLES},{WELCH_SECONDS:g}*fs))):"
            "noverlap=nperseg//2:return_onesided=true"
            if uses_welch
            else ""
        )
        for statistic in ("mean", "population_sd"):
            definitions.append(
                _definition(
                    f"engineering.{name}.{statistic}",
                    formula=(f"{ENGINEERING_SCHEMA_VERSION}:{name}:" f"across_window_{statistic}{welch_contract}"),
                    units="engineering_feature_native_unit",
                    source=source,
                    eligibility=eligibility,
                    level="file",
                    aggregation=f"valid_window_{statistic}",
                    validity="at least one valid finite engineering window value",
                    routes=routes,
                    group="engineering_summary",
                )
            )
    frozen = tuple(definitions)
    registry = FeatureRegistry(frozen, FORMAL_REGISTRY_VERSION, _hash_definitions(frozen))
    registry.validate()
    return registry

def registry_for_groups(groups: Sequence[str] | None = None) -> FeatureRegistry:
    """Return the content-addressed registry for selected complete groups."""

    selected_groups = canonicalize_feature_groups(groups)
    full = default_registry()
    if selected_groups == FEATURE_GROUP_ORDER:
        return full
    selected = tuple(definition for definition in full.definitions if definition.group in selected_groups)
    digest = _hash_definitions(selected)
    group_slug = "-".join(selected_groups)
    schema = f"feature_vector_{len(selected)}_{group_slug}_{digest[:12]}_v3"
    registry = FeatureRegistry(selected, schema, digest)
    registry.validate()
    return registry

def registry_for_feature_names(names: Sequence[str]) -> FeatureRegistry:
    """Resolve a legal whole-group registry from its exact ordered names."""

    observed = tuple(str(name) for name in names)
    full = default_registry()
    definitions = {item.canonical_name: item for item in full.definitions}
    if not observed or any(name not in definitions for name in observed):
        raise ValueError("feature names do not identify a registered feature-group selection")
    groups = canonicalize_feature_groups(tuple(definitions[name].group for name in observed))
    registry = registry_for_groups(groups)
    if observed != registry.names:
        raise ValueError("feature names must contain complete groups in canonical order")
    return registry

def summarize_engineering(
    extraction: EngineeringExtraction,
) -> tuple[dict[str, float], dict[str, bool]]:
    """按 §7.4 汇总 mean/population SD / Aggregate by mean/population SD."""

    validate_engineering_extraction(extraction, fold_transformed=False)
    values: dict[str, float] = {}
    validity: dict[str, bool] = {}
    matrix = np.asarray(extraction.sequence.values, dtype=np.float64)
    mask = np.asarray(extraction.value_validity, dtype=bool)
    for column, name in enumerate(extraction.sequence.channel_schema):
        selected = matrix[:, column][mask[:, column] & np.isfinite(matrix[:, column])]
        mean_key = f"engineering.{name}.mean"
        sd_key = f"engineering.{name}.population_sd"
        values[mean_key] = float(np.mean(selected)) if selected.size else float("nan")
        values[sd_key] = float(np.std(selected, ddof=0)) if selected.size else float("nan")
        validity[mean_key] = validity[sd_key] = bool(selected.size > 0)
    return values, validity

def build_feature_vector(
    feature_values: Mapping[str, float],
    *,
    feature_validity: Mapping[str, bool],
    provenance: Mapping[str, Any],
    registry: FeatureRegistry | None = None,
) -> FeatureVectorV1:
    """按 selected registry 顺序构建 predictor / Build the selected predictor.

    中文：缺失 slot 始终保留 NaN/false；未知 predictor（含技术字段）失败闭合。
    English: Missing slots remain NaN/false; unknown/technical predictors fail closed.
    """

    selected = registry or default_registry()
    selected.validate()
    expected = registry_for_feature_names(selected.names)
    if selected.schema_version != expected.schema_version or selected.sha256 != expected.sha256:
        raise ValueError("feature registry is not a registered whole-group selection")
    names = selected.names
    unknown_values = sorted(set(feature_values) - set(names))
    unknown_validity = sorted(set(feature_validity) - set(names))
    if unknown_values or unknown_validity:
        raise ValueError(f"unknown predictor fields: values={unknown_values}, validity={unknown_validity}")
    metadata = dict(provenance)
    if "route" not in metadata:
        raise ValueError("feature-vector provenance requires an explicit route")
    route = str(metadata["route"])
    if route not in {item.value for item in SignalRoute}:
        raise ValueError("feature-vector provenance contains an unknown route")
    definition_by_name = {item.canonical_name: item for item in selected.definitions}
    values = np.full(len(names), np.nan, dtype=np.float64)
    validity = np.zeros(len(names), dtype=bool)
    for index, name in enumerate(names):
        raw = feature_values.get(name, float("nan"))
        declared = bool(feature_validity.get(name, False))
        if route in definition_by_name[name].eligible_routes and declared and np.isfinite(raw):
            values[index] = float(raw)
            validity[index] = True
    metadata.update(
        {
            "registry_sha256": selected.sha256,
            "registry_schema_version": selected.schema_version,
            "enabled_groups": list(canonicalize_feature_groups(tuple(item.group for item in selected.definitions))),
            "feature_count": len(selected.names),
            "missing_value_contract": MISSING_POLICY,
            "technical_fields_excluded": True,
            "sqi_and_coverage_predictors_excluded": True,
        }
    )
    return FeatureVectorV1(values, validity, names, selected.schema_version, metadata)

def ordered_matrix_schema_version(
    k: int | None = None,
    registry: FeatureRegistry | None = None,
) -> str:
    """Return the K-independent 146-channel matrix identity.

    ``k`` remains only to fail clearly for callers that still declare the retired
    fixed matrix width; it never changes the variable-length schema.
    """

    if k is not None:
        raise ValueError("matrix_k is retired from the variable-length schema")
    if registry is not None:
        registry.validate()
    return ORDERED_MATRIX_SCHEMA_VERSION

def build_ordered_matrix(
    sequence: Any,
    *,
    provenance: Mapping[str, Any],
    k: int | None = None,
) -> Any:
    """Compatibility facade for the variable-K window-matrix builder."""

    if k is not None:
        raise ValueError("feature matrix no longer accepts a fixed matrix_k")
    from .window_matrix import build_ordered_window_matrix

    return build_ordered_window_matrix(sequence, provenance=provenance)
