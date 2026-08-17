"""冻结十字段注册表与显式有效性模型编码 / Frozen registry and mask encoding.

English: The public builders implement only the formal FeatureVectorV1 allowlist
and K=32 matrix contract. Experimental schemas require a separately named registry;
technical or unknown predictor fields are rejected rather than silently appended.

中文：公共构建器仅实现正式 FeatureVectorV1 allowlist 与 K=32 合同。实验 schema
必须使用单独命名的注册表；技术字段或未知 predictor 会被拒绝，而非静默追加。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from ..contracts import FeatureVectorV1, OrderedFeatureMatrixV1, SignalRoute
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
FORMAL_REGISTRY_VERSION = "feature_vector_thesis_115_v2"
ORDERED_MATRIX_SCHEMA_VERSION = "ordered_feature_matrix_thesis_115_d_by_32_v2"
MISSING_POLICY = "NaN_internal/null_JSON_with_parallel_validity_false"


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
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
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
    """构建 V1 完整固定 allowlist / Build the complete fixed V1 allowlist."""

    definitions: list[FeatureDefinition] = []
    for name in TIME_METRICS + SPECTRAL_METRICS:
        if name == "coverage":
            continue
        definitions.append(
            _definition(
                f"prv.{name}", formula=f"compute_prv_v1:{name}", units=_prv_unit(name),
                source="peak_timestamps+interval_indices+x_filter_or_x_ar",
                eligibility="Q_rate; duration/coverage/role gates declared by metric",
                level="file", aggregation="accepted_interval_statistic",
                validity="metric-specific minimum count, continuity, duration and finite result",
                routes=RATE_ROUTES, group="rate_prv",
            )
        )
    for name in MORPHOLOGY_NAMES:
        for statistic in ("median", "mad"):
            definitions.append(
                _definition(
                    f"morphology.{name}_{statistic}",
                    formula=f"valley_to_valley_baseline_v1:{name}:{statistic}",
                    units="signal_unit_or_seconds_or_derived", source="x_filter",
                    eligibility="Q_morph; direct or identity only", level="file",
                    aggregation=f"valid_beat_{statistic}",
                    validity="at least three valid accepted beats and finite statistic",
                    routes=DIRECT_ROUTES, group="morphology",
                )
            )
    optical_names = (
        "red_ac_median", "ir_ac_median", "red_dc_median", "ir_dc_median",
        "red_pi_median", "ir_pi_median", "red_ir_ac_ratio_median",
        "red_ir_dc_ratio_median", "ratio_of_ratios_median",
        "red_ir_zero_lag_correlation", "red_ir_max_xcorr", "red_ir_xcorr_lag_s",
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
                routes=DIRECT_ROUTES, group="dual_optical",
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
                    formula=(
                        f"{ENGINEERING_SCHEMA_VERSION}:{name}:"
                        f"across_window_{statistic}{welch_contract}"
                    ),
                    units="engineering_feature_native_unit", source=source,
                    eligibility=eligibility, level="file",
                    aggregation=f"valid_window_{statistic}",
                    validity="at least one valid finite engineering window value",
                    routes=routes, group="engineering_summary",
                )
            )
    frozen = tuple(definitions)
    registry = FeatureRegistry(frozen, FORMAL_REGISTRY_VERSION, _hash_definitions(frozen))
    registry.validate()
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
    """按完整冻结顺序构建 predictor / Build the complete ordered predictor.

    中文：缺失 slot 始终保留 NaN/false；未知 predictor（含技术字段）失败闭合。
    English: Missing slots remain NaN/false; unknown/technical predictors fail closed.
    """

    selected = registry or default_registry()
    selected.validate()
    if selected.schema_version != FORMAL_REGISTRY_VERSION and not selected.schema_version.startswith(
        "named_ablation_"
    ):
        raise ValueError("non-formal registry must use a named_ablation_ schema_version")
    names = selected.names
    unknown_values = sorted(set(feature_values) - set(names))
    unknown_validity = sorted(set(feature_validity) - set(names))
    if unknown_values or unknown_validity:
        raise ValueError(
            f"unknown predictor fields: values={unknown_values}, validity={unknown_validity}"
        )
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
            "missing_value_contract": MISSING_POLICY,
            "technical_fields_excluded": True,
            "sqi_and_coverage_predictors_excluded": True,
        }
    )
    return FeatureVectorV1(values, validity, names, selected.schema_version, metadata)


def _uniform_indices(count: int, target: int) -> np.ndarray:
    """按 recording progress 均匀抽取 K rows / Uniformly sample progress."""

    if count <= target:
        return np.arange(count, dtype=np.int64)
    indices = np.rint(np.linspace(0, count - 1, target)).astype(np.int64)
    if np.unique(indices).size != target:
        raise RuntimeError("uniform row sampler produced duplicate indices")
    return indices


def build_ordered_matrix(
    sequence: EngineeringExtraction,
    *,
    context: FeatureVectorV1,
    provenance: Mapping[str, Any],
    k: int = 32,
) -> OrderedFeatureMatrixV1:
    """构建带显式 validity channels 的唯一 D×32 合同 / Build formal D×32.

    English: Values and their 0/1 validity channels both enter the model tensor.
    Padding remains neutral zero with ``row_mask=false``. Mapping contexts and raw
    EngineeringFeatureSequence inputs are intentionally not canonical facades.

    中文：数值及其 0/1 validity channels 均进入模型 tensor。padding 保持零且
    ``row_mask=false``；Mapping context 与裸 EngineeringFeatureSequence 不属于正式入口。
    """

    if not isinstance(sequence, EngineeringExtraction):
        raise TypeError("canonical matrix builder requires EngineeringExtraction")
    if not isinstance(context, FeatureVectorV1):
        raise TypeError("canonical matrix builder requires a complete FeatureVectorV1 context")
    if k != 32:
        raise ValueError("formal OrderedFeatureMatrixV1 requires exactly K=32")
    base = sequence.sequence
    validate_engineering_extraction(sequence, fold_transformed=True)
    registry = default_registry()
    if tuple(context.feature_names) != registry.names:
        raise ValueError("context must be the complete formal FeatureVectorV1 registry order")
    if context.provenance.get("registry_sha256") != registry.sha256:
        raise ValueError("context registry hash differs from the frozen formal registry")
    if context.provenance.get("fold_standardized") is not True:
        raise ValueError("context must declare completed fold-local standardization")
    route = sequence.route.value
    context_route = str(context.provenance.get("route", ""))
    metadata = dict(provenance)
    if metadata.get("route") != route or context_route != route:
        raise ValueError("engineering, context, and matrix provenance routes must match")

    values = np.asarray(base.values, dtype=np.float64)
    physiological_validity = np.asarray(sequence.value_validity, dtype=bool)
    row_mask_input = np.asarray(base.valid_row_mask, dtype=bool)
    if (
        values.ndim != 2
        or physiological_validity.shape != values.shape
        or row_mask_input.shape != (values.shape[0],)
    ):
        raise ValueError("engineering sequence/value masks have incompatible shapes")
    valid_rows = np.flatnonzero(row_mask_input)
    if valid_rows.size == 0:
        raise ValueError("matrix requires at least one valid engineering row")
    selected_rows = valid_rows[_uniform_indices(valid_rows.size, k)]
    observed = min(k, selected_rows.size)
    base_count = values.shape[1]
    context_count = len(context.feature_names)
    physiological_count = base_count + context_count
    output = np.zeros((2 * physiological_count, k), dtype=np.float64)
    physical_mask = np.zeros((physiological_count, k), dtype=bool)
    row_mask = np.zeros(k, dtype=bool)

    selected_values = values[selected_rows[:observed]]
    selected_validity = (
        physiological_validity[selected_rows[:observed]] & np.isfinite(selected_values)
    )
    output[:base_count, :observed] = np.where(selected_validity, selected_values, 0.0).T
    output[base_count : 2 * base_count, :observed] = selected_validity.T.astype(np.float64)
    physical_mask[:base_count, :observed] = selected_validity.T
    context_value_offset = 2 * base_count
    context_validity_offset = context_value_offset + context_count
    for index, (value, valid) in enumerate(zip(context.values, context.validity)):
        is_valid = bool(valid and np.isfinite(value))
        if is_valid:
            output[context_value_offset + index, :observed] = float(value)
            physical_mask[base_count + index, :observed] = True
        output[context_validity_offset + index, :observed] = float(is_valid)
    row_mask[:observed] = True

    base_names = tuple(base.channel_schema)
    base_validity_names = tuple(f"{name}.validity" for name in base_names)
    context_names = tuple(context.feature_names)
    context_validity_names = tuple(f"{name}.validity" for name in context_names)
    channel_schema = base_names + base_validity_names + context_names + context_validity_names
    context_schema = context_names + context_validity_names
    schema_sha = hashlib.sha256("\n".join(channel_schema).encode("utf-8")).hexdigest()
    metadata.update(
        {
            "selected_source_rows": selected_rows[:observed].tolist(),
            "physiological_value_validity": physical_mask.tolist(),
            "validity_encoding": "paired_explicit_0_1_channels_v1",
            "validity_channel_map": {
                name: f"{name}.validity" for name in base_names + context_names
            },
            "zero_semantics": "fold_standardized_neutral_or_invalid_or_right_padding",
            "row_mask_policy": "right_padding_false",
            "context_schema_sha256": hashlib.sha256(
                "\n".join(context_schema).encode("utf-8")
            ).hexdigest(),
            "context_registry_sha256": registry.sha256,
            "matrix_channel_schema_sha256": schema_sha,
            "engineering_transform_version": base.schema_version,
        }
    )
    return OrderedFeatureMatrixV1(
        values=output,
        row_mask=row_mask,
        channel_schema=channel_schema,
        context_schema=context_schema,
        schema_version=ORDERED_MATRIX_SCHEMA_VERSION,
        provenance=metadata,
    )
