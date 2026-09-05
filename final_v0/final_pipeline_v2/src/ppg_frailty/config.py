"""V2 严格配置与决策档案合同 / Strict V2 config and decision profiles.

正式运行只接受 ``ppg_frailty.pipeline_config.v2``。复制进 V2 目录的 V1 YAML
只能通过显式 ``allow_legacy=True`` 读取作 provenance，不得进入正式 runner。

Formal execution accepts only ``ppg_frailty.pipeline_config.v2``. Copied V1 YAML
can be read only with explicit ``allow_legacy=True`` for provenance and can never
enter the formal runner by accident.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


TOP_LEVEL_KEYS = {
    "schema_version",
    "config_id",
    "manifest",
    "splits",
    "output",
    "representation_mode",
    "roles",
    "signal",
    "windows",
    "quality",
    "routing",
    "artifact",
    "features",
    "model",
    "training",
    "aggregation",
    "evaluation",
}

V2_SCHEMA_VERSION = "ppg_frailty.pipeline_config.v2"
LEGACY_SCHEMA_VERSION = "ppg_frailty.pipeline_config.v1"
V2_DECISION_PROFILE_SCHEMA = "ppg_frailty.v2_decision_profile.v3"
V2_FORMAL_CATALOG_SCHEMA = "ppg_frailty.formal_experiment_catalog.v2"
V2_FORMAL_ABLATION_PROFILES_SCHEMA = "ppg_frailty.formal_ablation_profiles.v2"
V2_SPLIT_SEEDS = (42, 10042, 20042, 30042, 40042)
FEATURE_REGISTRY_CONFIG_SCHEMA = "feature_vector_282_v3"
FEATURE_VECTOR_CONFIG_SCHEMA = "feature_vector_282_v3"
ENGINEERING_SEQUENCE_CONFIG_SCHEMA = "engineering_10s_hop2s_thesis_115_v3"
ORDERED_MATRIX_CONFIG_SCHEMA = (
    "ordered_window_feature_matrix_d146_variable_k_v1"
)
WINDOW_FEATURE_CONFIG_SCHEMA = "window_feature_set_d146_v1"
LEGACY_TOP_LEVEL_KEYS = TOP_LEVEL_KEYS - {"routing"}
def _strict_mapping(value: Any, name: str) -> dict[str, Any]:
    """验证对象类型 / Require a string-keyed mapping."""

    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return dict(value)


def _require_exact_keys(mapping: Mapping[str, Any], required: set[str], *, context: str) -> None:
    """拒绝缺字段和未知字段 / Reject missing and unknown fields."""

    observed = set(mapping)
    missing = sorted(required - observed)
    unknown = sorted(observed - required)
    if missing or unknown:
        raise ValueError(f"{context} key mismatch: missing={missing}, unknown={unknown}")


def canonical_json_bytes(value: Any) -> bytes:
    """稳定严格 JSON / Render canonical strict JSON bytes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class PipelineConfig:
    """规范实验配置 / Canonical experiment configuration."""

    payload: dict[str, Any]
    source_path: str
    sha256: str

    @property
    def config_id(self) -> str:
        """返回配置 ID / Return the configuration identity."""

        return str(self.payload["config_id"])

    @property
    def representation_mode(self) -> str:
        """返回表征模式 / Return the representation mode."""

        return str(self.payload["representation_mode"])

    @property
    def schema_version(self) -> str:
        """返回配置 schema / Return the explicit schema identity."""

        return str(self.payload["schema_version"])

    @property
    def is_legacy(self) -> bool:
        """V1 仅可作来源快照 / Whether this is a provenance-only V1 config."""

        return self.schema_version == LEGACY_SCHEMA_VERSION

    def section(self, name: str) -> dict[str, Any]:
        """读取一个显式 section / Return one explicit section."""

        if name not in TOP_LEVEL_KEYS:
            raise KeyError(name)
        return _strict_mapping(self.payload[name], name)

    def to_dict(self) -> dict[str, Any]:
        """复制可序列化配置 / Copy the serializable payload."""

        return json.loads(json.dumps(self.payload, allow_nan=False))


def _validate_common_payload(data: dict[str, Any]) -> None:
    """验证 V1/V2 共同结构 / Validate structure shared by V1 and V2."""

    expected_keys = (
        LEGACY_TOP_LEVEL_KEYS
        if data.get("schema_version") == LEGACY_SCHEMA_VERSION
        else TOP_LEVEL_KEYS
    )
    _require_exact_keys(data, expected_keys, context="config")
    if data["representation_mode"] not in {"raw", "feature_vector", "feature_matrix", "fusion"}:
        raise ValueError("unsupported representation_mode")
    from .data.schema import REGISTERED_ROLES

    roles = data["roles"]
    allowed_roles = set(REGISTERED_ROLES)
    if not isinstance(roles, list) or not roles or not all(role in allowed_roles for role in roles):
        raise ValueError("roles must be a non-empty registered role list")
    if len(set(roles)) != len(roles):
        raise ValueError("roles must not contain duplicate role IDs")
    for section in expected_keys - {"schema_version", "config_id", "representation_mode", "roles"}:
        _strict_mapping(data[section], section)
    training = _strict_mapping(data["training"], "training")
    if training.get("epoch_rule") not in {"fixed_epoch", "inner_grouped_selection"}:
        raise ValueError("training.epoch_rule must be explicit")
    if training.get("outer_labels_visible_to_trainer") is not False:
        raise ValueError("outer labels must be unavailable to the trainer")
    artifact = _strict_mapping(data["artifact"], "artifact")
    if artifact.get("selection_scope") != "run_before_evaluation":
        raise ValueError("artifact route must be selected before evaluation")


_QUALITY_DERIVED_FIELDS = frozenset(
    {"mode", "fit_scope", "components", "high_quality_rule", "failure_action"}
)


def _diagnostic_quality_runtime_mapping(config: Any) -> dict[str, Any]:
    """Serialize exactly the physical fields consumed by diagnostics-only."""

    config.validate()
    return {
        "cardiac_band_hz": [
            float(config.cardiac_low_hz),
            float(config.cardiac_high_hz),
        ],
        "peak_density_bpm_range": [
            float(config.peak_density_min_bpm),
            float(config.peak_density_max_bpm),
        ],
        "ppi_range_s": [float(config.ppi_min_s), float(config.ppi_max_s)],
        "long_gap_max_samples": int(config.long_gap_max_samples),
        "flatline_duration_s": float(config.flatline_duration_s),
        "spectral_analysis_band_hz": [
            float(config.spectral_analysis_low_hz),
            float(config.spectral_analysis_high_hz),
        ],
        "welch_max_nperseg": int(config.welch_max_nperseg),
        "template_min_peaks": int(config.template_min_peaks),
        "template_min_beats": int(config.template_min_beats),
        "template_resample_points": int(config.template_resample_points),
        "ppi_stability_min_intervals": int(
            config.ppi_stability_min_intervals
        ),
        "component_normalization": {
            "template_half_width_s": float(config.template_half_width_s),
        },
    }


def _validate_v2_quality(data: Mapping[str, Any]) -> None:
    """Validate only parameters consumed by the selected quality module."""

    quality = _strict_mapping(data["quality"], "quality")
    from .quality.window_selection import WindowSelectionConfig
    from .signal.sqi import SqiConfig, SqiDiagnosticConfig
    from .v2_contract import validate_quality_mode

    mode = validate_quality_mode(str(quality.get("mode")))
    window_selection = WindowSelectionConfig.from_mapping(
        quality.get("window_selection")
    )
    common = set(_QUALITY_DERIVED_FIELDS) | {
        "window_selection",
        "long_gap_max_samples",
        "flatline_duration_s",
    }
    artifact = _strict_mapping(data["artifact"], "artifact")
    denoiser_enabled = bool(
        artifact.get(
            "denoiser_enabled",
            str(artifact.get("reducer", "identity")) != "identity",
        )
    )
    if mode == "route":
        expected = common | set(SqiConfig().to_dict())
        _require_exact_keys(quality, expected, context="quality")
        sqi_quality = dict(quality)
        sqi_quality.pop("window_selection", None)
        SqiConfig.from_quality_mapping(sqi_quality)
    elif mode == "diagnostics_only":
        diagnostic = SqiDiagnosticConfig.from_resolved({"quality": quality})
        expected = common | set(_diagnostic_quality_runtime_mapping(diagnostic))
        if denoiser_enabled:
            expected |= set(SqiConfig().to_dict())
        _require_exact_keys(quality, expected, context="quality")
        if denoiser_enabled:
            recovery_mapping = dict(quality)
            recovery_mapping.pop("window_selection", None)
            recovery_sqi = SqiConfig.from_quality_mapping(recovery_mapping)
            if recovery_sqi.calibrator != "fixed_formula_thresholds_v1":
                raise ValueError(
                    "diagnostics-only denoiser recovery requires "
                    "fixed_formula_thresholds_v1"
                )
    else:
        expected = (
            common | set(SqiConfig().to_dict())
            if denoiser_enabled
            else common
        )
        _require_exact_keys(quality, expected, context="quality")
        if denoiser_enabled:
            sqi_quality = dict(quality)
            sqi_quality.pop("window_selection", None)
            recovery_sqi = SqiConfig.from_quality_mapping(sqi_quality)
            if recovery_sqi.calibrator != "fixed_formula_thresholds_v1":
                raise ValueError(
                    "SQI-off denoiser recovery requires fixed_formula_thresholds_v1"
                )
    flatline_duration_s = float(quality["flatline_duration_s"])
    if not math.isfinite(flatline_duration_s) or flatline_duration_s <= 0.0:
        raise ValueError("quality.flatline_duration_s must be positive and finite")
    if (
        window_selection.policy != "none"
        and str(data["representation_mode"]) not in {"raw", "fusion"}
    ):
        raise ValueError(
            "quality.window_selection is executable only for raw or fusion "
            "representations"
        )
    if (
        window_selection.policy != "none"
        and window_selection.application_scope == "legacy_train_and_aggregation"
        and str(data["representation_mode"]) != "raw"
    ):
        raise ValueError(
            "quality.window_selection.application_scope="
            "legacy_train_and_aggregation requires raw window-level OOF; "
            "file-level fusion cannot consume a held-out window selection view"
        )
    if quality.get("failure_action") != "fail_closed":
        raise ValueError("quality.failure_action must be fail_closed")
    gap_repair = _strict_mapping(
        _strict_mapping(data["signal"], "signal").get("gap_repair"),
        "signal.gap_repair",
    )
    if int(quality["long_gap_max_samples"]) != int(gap_repair.get("max_gap_samples", -1)):
        raise ValueError(
            "quality.long_gap_max_samples and signal.gap_repair.max_gap_samples "
            "describe one fused parameter and must match"
        )


def _materialize_quality_defaults(data: dict[str, Any]) -> None:
    """Persist only the runtime parameters consumed by the selected mode.

    ``off`` does not instantiate SQI, but still persists physical-admission
    parameters consumed by preprocessing. ``diagnostics_only`` persists all
    physical component parameters but removes endpoint thresholds, fusion
    weights and calibrator controls. ``route`` persists the complete endpoint
    policy. Consequently an inactive SQI field cannot change the effective
    config hash.
    """

    from .quality.window_selection import WindowSelectionConfig
    from .signal.sqi import SqiConfig, SqiDiagnosticConfig
    from .v2_contract import validate_quality_mode

    declared = _strict_mapping(data["quality"], "quality")
    if "supervised_route_ready" in declared:
        raise ValueError(
            "quality.supervised_route_ready is retired; remove it and select "
            "the executable module directly with quality.mode"
        )
    if "long_gap_max_samples" not in declared:
        gap_repair = _strict_mapping(
            _strict_mapping(data["signal"], "signal").get("gap_repair"),
            "signal.gap_repair",
        )
        declared["long_gap_max_samples"] = gap_repair.get("max_gap_samples", 100)
    # Flatline rejection belongs to the always-executed physical PPG admission
    # step in ``build_signal_views``.  It therefore remains an effective
    # runtime parameter even when endpoint SQI routing/diagnostics are off.
    declared.setdefault(
        "flatline_duration_s",
        SqiDiagnosticConfig().flatline_duration_s,
    )
    mode = validate_quality_mode(str(declared.get("mode", "off")))
    window_selection = WindowSelectionConfig.from_mapping(
        declared.pop("window_selection", None)
    )
    route_fields = set(SqiConfig().to_dict())
    allowed = (
        set(_QUALITY_DERIVED_FIELDS)
        | route_fields
    )
    unknown = sorted(set(declared) - allowed)
    if unknown:
        raise ValueError(f"quality contains unknown fields: {unknown}")
    normalization = declared.get("component_normalization")
    if normalization is not None:
        if not isinstance(normalization, Mapping):
            raise ValueError("quality.component_normalization must be a mapping")
        registered_normalization = set(
            SqiConfig().to_dict()["component_normalization"]
        )
        unknown_normalization = sorted(
            set(normalization) - registered_normalization
        )
        if unknown_normalization:
            raise ValueError(
                "quality.component_normalization has unknown fields: "
                f"{unknown_normalization}"
            )
    metadata_defaults = {
        "mode": mode,
        "fit_scope": (
            "outer_training_participants_only"
            if mode == "route"
            else "not_applied_" + mode
        ),
        "components": (
            []
            if mode == "off"
            else [
                "cardiac_concentration",
                "autocorrelation_periodicity",
                "normalized_spectral_entropy",
                "peak_density_bpm",
                "ppi_physiological_fraction",
                "ppi_stability",
                "red_ir_agreement",
                "motion_energy_rms",
                "nonflat_scale",
                "source_coverage",
                "flatline",
                "clipping",
                "saturation",
                "long_gap",
            ]
        ),
        "high_quality_rule": (
            "configured_endpoint_thresholds" if mode == "route" else "not_applied"
        ),
        "failure_action": "fail_closed",
        "window_selection": window_selection.to_mapping(),
    }
    if declared.get("failure_action", "fail_closed") != "fail_closed":
        raise ValueError("quality.failure_action must be fail_closed")
    artifact = _strict_mapping(data["artifact"], "artifact")
    denoiser_enabled = bool(
        artifact.get(
            "denoiser_enabled",
            str(artifact.get("reducer", "identity")) != "identity",
        )
    )
    if mode != "route" and denoiser_enabled:
        declared.setdefault("calibrator", "fixed_formula_thresholds_v1")
    if mode == "route":
        runtime = SqiConfig.from_quality_mapping(declared).to_dict()
    elif mode == "diagnostics_only":
        diagnostic_runtime = _diagnostic_quality_runtime_mapping(
            SqiDiagnosticConfig.from_resolved({"quality": declared})
        )
        if denoiser_enabled:
            recovery_sqi = SqiConfig.from_quality_mapping(declared)
            if recovery_sqi.calibrator != "fixed_formula_thresholds_v1":
                raise ValueError(
                    "diagnostics-only denoiser recovery requires "
                    "fixed_formula_thresholds_v1"
                )
            runtime = {**diagnostic_runtime, **recovery_sqi.to_dict()}
            metadata_defaults["high_quality_rule"] = (
                "direct_diagnostics_only_post_denoise_q_rate_"
                "fixed_formula_only"
            )
        else:
            runtime = diagnostic_runtime
    elif denoiser_enabled:
        # Direct SQI remains off.  A successful reducer still needs one
        # auditable fixed-formula Q_rate reassessment, so its active numerical
        # policy is materialized into the effective config and hash.
        recovery_sqi = SqiConfig.from_quality_mapping(declared)
        if recovery_sqi.calibrator != "fixed_formula_thresholds_v1":
            raise ValueError(
                "SQI-off denoiser recovery requires fixed_formula_thresholds_v1"
            )
        runtime = recovery_sqi.to_dict()
        metadata_defaults["high_quality_rule"] = (
            "direct_sqi_off_post_denoise_q_rate_fixed_formula_only"
        )
        metadata_defaults["components"] = []
    else:
        # Gap repair and physical flatline admission execute independently of
        # endpoint SQI, so both parameters remain visible while SQI is off.
        runtime = {
            "long_gap_max_samples": int(declared["long_gap_max_samples"]),
            "flatline_duration_s": float(declared["flatline_duration_s"]),
        }
    effective = {**metadata_defaults, **runtime}
    data["quality"] = effective


def _materialize_routing_defaults(data: dict[str, Any]) -> None:
    """Persist the common representation-independent 400 Hz evidence grid."""

    declared = data.get("routing", {})
    if declared is None:
        declared = {}
    if not isinstance(declared, Mapping):
        raise TypeError("routing must be a mapping")
    allowed = {"window_s", "hop_s", "fs_hz", "source_grid"}
    unknown = sorted(set(declared) - allowed)
    if unknown:
        raise ValueError(f"routing contains unknown fields: {unknown}")
    effective = {
        "window_s": float(declared.get("window_s", 8.0)),
        "hop_s": float(declared.get("hop_s", 2.0)),
        "fs_hz": float(declared.get("fs_hz", 400.0)),
        "source_grid": str(
            declared.get("source_grid", "canonical_acquisition_grid")
        ),
    }
    if effective != {
        "window_s": 8.0,
        "hop_s": 2.0,
        "fs_hz": 400.0,
        "source_grid": "canonical_acquisition_grid",
    }:
        raise ValueError("formal routing grid is fixed at canonical 400 Hz, 8 s/2 s")
    data["routing"] = effective


def _materialize_dl_resampling_defaults(data: dict[str, Any]) -> None:
    """Resolve the optional DL-only sampling module before config hashing."""

    signal = _strict_mapping(data["signal"], "signal")
    source_grid = float(signal.get("internal_fs_hz", 400.0))
    raw = signal.get("dl_resampling", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError("signal.dl_resampling must be a mapping")
    declared = dict(raw)
    enabled = declared.get("enabled", False)
    defaults = {
        "enabled": enabled,
        "target_fs_hz": (
            source_grid if not bool(enabled) else source_grid / 2.0
        ),
        "method": "polyphase_anti_alias",
        "preserve_feature_grid_hz": source_grid,
    }
    signal["dl_resampling"] = {**defaults, **declared}
    data["signal"] = signal


def _materialize_signal_normalization_defaults(data: dict[str, Any]) -> None:
    """Resolve strategy aliases and persist every raw normalization parameter."""

    from .normalization import RawNormalizationConfig

    signal = _strict_mapping(data["signal"], "signal")
    signal["normalization"] = RawNormalizationConfig.from_mapping(
        signal.get("normalization")
    ).to_mapping()
    data["signal"] = signal


def _materialize_signal_preprocessing_defaults(data: dict[str, Any]) -> None:
    """Canonicalize executable signal views and IMU profile controls."""

    from .signal.preprocess import materialize_signal_preprocessing_config

    data["signal"] = materialize_signal_preprocessing_config(
        _strict_mapping(data["signal"], "signal")
    )


def _materialize_peak_detector_defaults(data: dict[str, Any]) -> None:
    """Persist detector thresholds so overrides affect effective identity."""

    from .module_registry import resolve_peak_detector_config

    signal = _strict_mapping(data["signal"], "signal")
    signal["peak_detector"] = resolve_peak_detector_config(signal)
    data["signal"] = signal


def _materialize_artifact_defaults(data: dict[str, Any]) -> None:
    """Persist independent denoiser and frozen motion-inference controls."""

    from .quality.motion_bundle_adapter import (
        resolve_reused_motion_detector_config,
    )

    artifact = _strict_mapping(data["artifact"], "artifact")
    artifact.setdefault(
        "denoiser_enabled",
        str(artifact.get("reducer", "identity")) != "identity",
    )
    motion_defaults = resolve_reused_motion_detector_config().to_mapping(
        include_enabled=False
    )
    declared_motion = artifact.get("motion_detector", {})
    if declared_motion is None:
        declared_motion = {}
    if not isinstance(declared_motion, Mapping):
        raise TypeError("artifact.motion_detector must be a mapping")
    artifact["motion_detector"] = {
        **motion_defaults,
        **dict(declared_motion),
    }
    data["artifact"] = artifact


def _materialize_aggregation_defaults(data: dict[str, Any]) -> None:
    """Derive hierarchy plumbing from the one selected aggregation module.

    ``balance_line``, ``quality_weighting`` and ``quality_weight_source`` are
    the user-facing controls.
    The hierarchy/operator fields are persisted provenance, not five extra
    switches that callers must keep synchronized by hand.  A fully resolved
    config may still carry the derived values of the other registered line;
    that is normalized when changing a study axis.  Invented operators remain
    an error rather than being silently discarded.
    """

    aggregation = _strict_mapping(data["aggregation"], "aggregation")
    allowed = {
        "balance_line", "hierarchy", "window_to_file", "file_to_role",
        "role_to_participant", "missing_role_policy", "quality_weighting",
        "quality_weight_source", "quality_weight_levels",
        "direct_all_window_participant_mean",
    }
    unknown = sorted(set(aggregation) - allowed)
    if unknown:
        raise ValueError(f"aggregation contains unknown fields: {unknown}")
    line = str(aggregation.get("balance_line", "line_b_equal_role_families"))
    derived = {
        "line_a_equal_files": {
            "hierarchy": ["window", "file", "participant"],
            "file_to_role": "not_applicable",
            "role_to_participant": "not_applicable",
            "missing_role_policy": "not_applicable",
        },
        "line_b_equal_role_families": {
            "hierarchy": ["window", "file", "role", "participant"],
            "file_to_role": "ordinary_mean",
            "role_to_participant": "ordinary_mean",
            "missing_role_policy": "mean_available_roles",
        },
    }
    if line not in derived:
        raise ValueError("aggregation.balance_line must select registered Line A or Line B")
    for field in ("hierarchy", "missing_role_policy"):
        if field not in aggregation:
            continue
        recognized = {json.dumps(values[field], sort_keys=True) for values in derived.values()}
        if json.dumps(aggregation[field], sort_keys=True) not in recognized:
            raise ValueError(f"aggregation.{field} is not implemented by a registered line")
    recognized_operators = {
        "ordinary_mean",
        "quality_weighted_mean",
        "not_applicable",
    }
    for field in ("window_to_file", "file_to_role", "role_to_participant"):
        if aggregation.get(field, "ordinary_mean") not in recognized_operators:
            raise ValueError(f"unsupported aggregation.{field} operator")
    if aggregation.get("direct_all_window_participant_mean", False) is not False:
        raise ValueError(
            "direct-all-window aggregation is a reporting view, not a selected hierarchy"
        )
    quality_weighting = aggregation.get("quality_weighting", False)
    if not isinstance(quality_weighting, bool):
        raise ValueError("aggregation.quality_weighting must be boolean")
    declared_weight_source = aggregation.get("quality_weight_source")
    registered_weight_sources = {
        None,
        "none",
        "route_file_q_rate",
        "legacy_window_sqi",
    }
    if declared_weight_source not in registered_weight_sources:
        raise ValueError(
            "aggregation.quality_weight_source must be none, "
            "route_file_q_rate, or legacy_window_sqi"
        )
    weight_source = (
        "none"
        if not quality_weighting
        else "route_file_q_rate"
        if declared_weight_source in {None, "none"}
        else str(declared_weight_source)
    )
    quality_levels = {
        "none": [],
        "route_file_q_rate": (
            ["file_to_participant"]
            if line == "line_a_equal_files"
            else ["file_to_role", "role_to_participant"]
        ),
        "legacy_window_sqi": (
            ["window_to_file", "file_to_participant"]
            if line == "line_a_equal_files"
            else [
                "window_to_file",
                "file_to_role",
                "role_to_participant",
            ]
        ),
    }[weight_source]
    declared_levels = aggregation.get("quality_weight_levels")
    if declared_levels is not None:
        recognized_levels = {
            json.dumps([], sort_keys=True),
            json.dumps(["file_to_participant"], sort_keys=True),
            json.dumps(["file_to_role", "role_to_participant"], sort_keys=True),
            json.dumps(
                ["window_to_file", "file_to_participant"], sort_keys=True
            ),
            json.dumps(
                ["window_to_file", "file_to_role", "role_to_participant"],
                sort_keys=True,
            ),
        }
        if json.dumps(declared_levels, sort_keys=True) not in recognized_levels:
            raise ValueError(
                "aggregation.quality_weight_levels is derived from the selected "
                "source and contains an unsupported value"
            )
    effective = {
        "balance_line": line,
        **derived[line],
        "window_to_file": (
            "quality_weighted_mean"
            if weight_source == "legacy_window_sqi"
            else "ordinary_mean"
        ),
        "quality_weighting": quality_weighting,
        "quality_weight_source": weight_source,
        "quality_weight_levels": quality_levels,
        "direct_all_window_participant_mean": False,
    }
    if line == "line_b_equal_role_families" and weight_source != "none":
        effective["file_to_role"] = "quality_weighted_mean"
        effective["role_to_participant"] = "quality_weighted_mean"
    data["aggregation"] = effective


def _validate_v2_balance(data: Mapping[str, Any]) -> None:
    """Validate independently selectable training and reporting balance modules."""

    aggregation = _strict_mapping(data["aggregation"], "aggregation")
    required = {
        "balance_line", "hierarchy", "window_to_file", "file_to_role",
        "role_to_participant", "missing_role_policy", "quality_weighting",
        "quality_weight_source", "quality_weight_levels",
        "direct_all_window_participant_mean",
    }
    _require_exact_keys(aggregation, required, context="aggregation")
    line = str(aggregation.get("balance_line"))
    hierarchies = {
        "line_a_equal_files": ["window", "file", "participant"],
        "line_b_equal_role_families": [
            "window",
            "file",
            "role",
            "participant",
        ],
    }
    if line not in hierarchies:
        raise ValueError("aggregation.balance_line must select registered Line A or Line B")
    if aggregation.get("hierarchy") != hierarchies[line]:
        raise ValueError("aggregation hierarchy does not match the selected balance line")
    if not isinstance(aggregation.get("quality_weighting"), bool):
        raise ValueError("aggregation.quality_weighting must be boolean")
    weighting = bool(aggregation.get("quality_weighting"))
    weight_source = str(aggregation.get("quality_weight_source"))
    if weighting == (weight_source == "none"):
        raise ValueError(
            "aggregation quality_weighting and quality_weight_source disagree"
        )
    if weight_source not in {
        "none",
        "route_file_q_rate",
        "legacy_window_sqi",
    }:
        raise ValueError("unsupported aggregation.quality_weight_source")
    if weight_source == "route_file_q_rate" and (
        _strict_mapping(data["quality"], "quality").get("mode") != "route"
    ):
        raise ValueError(
            "aggregation.quality_weight_source=route_file_q_rate requires "
            "quality.mode=route"
        )
    if weight_source == "legacy_window_sqi":
        if str(data["representation_mode"]) != "raw":
            raise ValueError(
                "aggregation.quality_weight_source=legacy_window_sqi requires "
                "raw window-level predictions; fusion starts at file level"
            )
        window_selection = _strict_mapping(
            _strict_mapping(data["quality"], "quality").get("window_selection"),
            "quality.window_selection",
        )
        if window_selection.get("policy") != "legacy_per_file_top_fraction":
            raise ValueError(
                "aggregation.quality_weight_source=legacy_window_sqi requires "
                "quality.window_selection.policy=legacy_per_file_top_fraction"
            )
    expected_quality_levels = {
        "none": [],
        "route_file_q_rate": (
            ["file_to_participant"]
            if line == "line_a_equal_files"
            else ["file_to_role", "role_to_participant"]
        ),
        "legacy_window_sqi": (
            ["window_to_file", "file_to_participant"]
            if line == "line_a_equal_files"
            else [
                "window_to_file",
                "file_to_role",
                "role_to_participant",
            ]
        ),
    }[weight_source]
    if aggregation.get("quality_weight_levels") != expected_quality_levels:
        raise ValueError(
            "aggregation.quality_weight_levels does not match the selected "
            "quality weight source and balance line"
        )
    expected_window_operator = (
        "quality_weighted_mean"
        if weight_source == "legacy_window_sqi"
        else "ordinary_mean"
    )
    if aggregation.get("window_to_file") != expected_window_operator:
        raise ValueError(
            "aggregation.window_to_file does not match the selected quality source"
        )
    line_b = line == "line_b_equal_role_families"
    expected_upper = (
        (
            "quality_weighted_mean" if weight_source != "none" else "ordinary_mean",
            "quality_weighted_mean" if weight_source != "none" else "ordinary_mean",
            "mean_available_roles",
        )
        if line_b
        else ("not_applicable", "not_applicable", "not_applicable")
    )
    observed_upper = (
        aggregation.get("file_to_role"),
        aggregation.get("role_to_participant"),
        aggregation.get("missing_role_policy"),
    )
    if observed_upper != expected_upper:
        raise ValueError("aggregation upper hierarchy operators do not match the selected line")
    if aggregation.get("direct_all_window_participant_mean") is not False:
        raise ValueError(
            "direct-all-window aggregation is a separate reporting view, not a "
            "substitute for the selected participant hierarchy"
        )


def _validate_v2_signal_normalization(data: Mapping[str, Any]) -> None:
    """Validate implemented signal strategies and their numerical ranges."""

    signal = _strict_mapping(data["signal"], "signal")
    required_signal_keys = {
        "internal_fs_hz",
        "channel_order",
        "ppg_native_unit",
        "accelerometer_input_unit",
        "gyroscope_input_unit",
        "ppg_filter",
        "peak_detector",
        "analysis_view",
        "gap_repair",
        "imu",
        "dl_resampling",
        "normalization",
    }
    _require_exact_keys(signal, required_signal_keys, context="signal")
    from .signal.preprocess import materialize_signal_preprocessing_config

    if signal != materialize_signal_preprocessing_config(signal):
        raise ValueError("signal preprocessing config must be materialized before validation")
    normalization = _strict_mapping(
        signal.get("normalization"),
        "signal.normalization",
    )
    from .normalization import RawNormalizationConfig

    resolved_normalization = RawNormalizationConfig.from_mapping(normalization)
    if normalization != resolved_normalization.to_mapping():
        raise ValueError("signal.normalization must be materialized before validation")
    if data["representation_mode"] not in {"raw", "fusion"}:
        default_normalization = RawNormalizationConfig().to_mapping()
        if normalization != default_normalization:
            raise ValueError(
                "non-default signal.normalization requires representation_mode "
                "raw or fusion"
            )


def _materialize_feature_defaults(data: dict[str, Any]) -> None:
    """Resolve every executable feature parameter before hashing.

    The 115-column base engineering sequence remains intact for file-vector
    summaries. The matrix representation extends it to 146 pure window features
    and retains every complete chronological row with variable K. The deprecated
    ``time_prv_min_accepted_peaks`` input is translated to the unambiguous
    ``rate_prv_min_peaks`` effective field.
    """

    from .features.registry import (
        FEATURE_GROUP_ORDER,
        canonicalize_feature_groups,
        ordered_matrix_schema_version,
        registry_for_groups,
    )
    from .signal.prv import PrvConfig

    declared = _strict_mapping(data["features"], "features")
    metadata_defaults = {
        "prv_primary_backend": "local_manual",
        "prv_library_comparison_scope": "fixed_ppi_vectors_only_no_classifier",
        "engineering_sequence_schema": ENGINEERING_SEQUENCE_CONFIG_SCHEMA,
        "technical_metadata_allowed": False,
        "missing_physiology_encoding": "nan_and_validity_false",
        "file_aggregation": ["mean", "population_sd"],
        "window_feature_schema": WINDOW_FEATURE_CONFIG_SCHEMA,
        "matrix_length_policy": "all_complete_windows_variable_k",
        "enabled_groups": list(FEATURE_GROUP_ORDER),
    }
    prv_fields = {
        "rate_prv_min_duration_s", "rate_prv_min_peaks",
        "time_prv_min_duration_s", "time_prv_min_coverage",
        "time_prv_min_intervals", "spectral_prv_min_duration_s",
        "spectral_prv_min_coverage", "spectral_prv_min_intervals",
        "tachogram_fs_hz", "spectral_bands_hz", "sample_entropy",
        # Accepted only as a source compatibility alias; never persisted.
        "time_prv_min_accepted_peaks",
    }
    derived_fields = {"registry_id", "file_vector_schema", "matrix_schema"}
    # The retired fixed-width field is rejected rather than silently ignored.
    allowed = set(metadata_defaults) | prv_fields | derived_fields | {"matrix_k"}
    unknown = sorted(set(declared) - allowed)
    if unknown:
        raise ValueError(f"features contains unknown fields: {unknown}")
    if "matrix_k" in declared:
        raise ValueError("features.matrix_k is retired; remove the fixed-K value")
    enabled_groups = canonicalize_feature_groups(
        declared.get("enabled_groups", FEATURE_GROUP_ORDER)
    )
    registry = registry_for_groups(enabled_groups)
    matrix_schema = ordered_matrix_schema_version(None, registry)
    prv = PrvConfig.from_mapping(declared)
    effective = {
        **metadata_defaults,
        **{key: declared.get(key, value) for key, value in metadata_defaults.items()},
        **prv.to_dict(),
        "enabled_groups": list(enabled_groups),
        # These identities are outputs of the selected executable groups.  Old
        # YAML values are accepted as source compatibility fields but can never
        # override or forge the materialized registry/schema identity.
        "registry_id": registry.schema_version,
        "file_vector_schema": registry.schema_version,
        "matrix_schema": matrix_schema,
    }
    data["features"] = effective


def _validate_v2_feature_schemas(data: Mapping[str, Any]) -> None:
    """Validate the selected feature module and its resolved runtime controls."""

    from .features.registry import (
        FEATURE_GROUP_ORDER,
        ordered_matrix_schema_version,
        registry_for_groups,
    )
    from .signal.prv import PrvConfig

    features = _strict_mapping(data["features"], "features")
    registry = registry_for_groups(features.get("enabled_groups"))
    if features.get("registry_id") != registry.schema_version:
        raise ValueError("features.registry_id is not derived from enabled_groups")
    if features.get("prv_primary_backend") != "local_manual":
        raise ValueError("selected PRV backend is not available to classifier features")
    if (
        features.get("prv_library_comparison_scope")
        != "fixed_ppi_vectors_only_no_classifier"
    ):
        raise ValueError(
            "features.prv_library_comparison_scope is fixed comparison provenance, "
            "not a classifier parameter"
        )
    if features.get("file_vector_schema") != registry.schema_version:
        raise ValueError("features.file_vector_schema is not derived from enabled_groups")
    if features.get("engineering_sequence_schema") != ENGINEERING_SEQUENCE_CONFIG_SCHEMA:
        raise ValueError("features.engineering_sequence_schema is not registered")
    if features.get("window_feature_schema") != WINDOW_FEATURE_CONFIG_SCHEMA:
        raise ValueError("features.window_feature_schema is not registered")
    if features.get("matrix_length_policy") != "all_complete_windows_variable_k":
        raise ValueError("feature matrix must retain all complete variable-K rows")
    if "matrix_k" in features:
        raise ValueError("features.matrix_k must not persist in the variable-K contract")
    if features.get("matrix_schema") != ordered_matrix_schema_version(None, registry):
        raise ValueError("features.matrix_schema is not the registered variable-K schema")
    if features.get("technical_metadata_allowed") is not False:
        raise ValueError("technical metadata cannot enter physiology predictors")
    if features.get("missing_physiology_encoding") != "nan_and_validity_false":
        raise ValueError("unsupported missing physiology encoding")
    if features.get("file_aggregation") != ["mean", "population_sd"]:
        raise ValueError("unsupported file feature aggregation strategy")
    mode = str(data["representation_mode"])
    if mode == "raw" and tuple(features["enabled_groups"]) != FEATURE_GROUP_ORDER:
        raise ValueError(
            "features.enabled_groups is not consumed by raw representation; "
            "select feature_vector, feature_matrix, or fusion"
        )
    if mode == "feature_matrix" and tuple(features["enabled_groups"]) != (
        "engineering_summary",
    ):
        raise ValueError(
            "feature_matrix consumes the registered 146 window features; "
            "features.enabled_groups must be [engineering_summary]"
        )
    prv_payload = PrvConfig.from_mapping(features).to_dict()
    default_prv_payload = PrvConfig().validated().to_dict()
    # Cross-module composition contract: these sets mirror the independent
    # eligibility/computation blocks in ``signal.prv.compute_prv``.  A disabled
    # predictor group must not leave behind a parameter that changes only the
    # effective-config hash or diagnostic provenance.
    prv_parameter_consumers = {
        "rate_prv_min_duration_s": {"ppi_basic_rate"},
        "rate_prv_min_peaks": {"ppi_basic_rate"},
        "time_prv_min_duration_s": {
            "hrv_time_domain",
            "hrv_nonlinear",
        },
        "time_prv_min_coverage": {
            "hrv_time_domain",
            "hrv_nonlinear",
        },
        "time_prv_min_intervals": {
            "hrv_time_domain",
            "hrv_nonlinear",
        },
        "spectral_prv_min_duration_s": {"hrv_spectral"},
        "spectral_prv_min_coverage": {"hrv_spectral"},
        "spectral_prv_min_intervals": {"hrv_spectral"},
        "tachogram_fs_hz": {"hrv_spectral"},
        "spectral_bands_hz": {"hrv_spectral"},
        "sample_entropy": {"hrv_nonlinear"},
    }
    enabled_groups = set(features["enabled_groups"])
    changed_without_consumer = [
        f"features.{field}"
        for field, consumers in prv_parameter_consumers.items()
        if prv_payload[field] != default_prv_payload[field]
        and (mode in {"raw", "feature_matrix"} or enabled_groups.isdisjoint(consumers))
    ]
    if changed_without_consumer:
        scope = (
            f"{mode} representation"
            if mode in {"raw", "feature_matrix"}
            else "the enabled feature groups"
        )
        raise ValueError(
            "non-default PRV controls are not consumed by "
            f"{scope}: {changed_without_consumer}"
        )


def _materialize_evaluation_defaults(data: dict[str, Any]) -> None:
    """Resolve configurable reporting budgets while preserving safe defaults."""

    evaluation = _strict_mapping(data["evaluation"], "evaluation")
    statistics_defaults = {
        "cluster_unit": "participant_with_all_five_repeat_oof_predictions",
        "bootstrap_replicates": 10_000,
        "confidence_interval": "two_sided_95_percentile",
        "lcb95_percentile": 2.5,
        "lcb95_metrics": [
            "participant_level_mean_balanced_accuracy",
            "participant_level_mean_macro_f1",
        ],
        "paired_permutation_replicates": 100_000,
        "seed": 42,
        "paired_exchange_unit": "participant",
        "multiplicity_correction": "holm_within_comparison_family",
        "affects_automatic_selection": False,
    }
    ranking_defaults = {
        "sort_key": "participant_level_mean_balanced_accuracy",
        "max_qualified_per_comparison_group": 10,
        "automatic_final_selection": False,
        "manual_multiple_final_versions_allowed": True,
        "preserve_ablation_provenance": True,
    }
    evaluation_defaults = {
        "unit": "participant",
        "primary_metric": "balanced_accuracy",
        "metrics": [
            "balanced_accuracy", "macro_f1", "per_class_precision_recall_f1",
            "worst_class_recall", "worst_class_f1", "confusion_matrix",
            "coverage",
        ],
        "confidence_interval": "participant_cluster_bootstrap_two_sided_95",
        "paired_delta_key": ["repeat_index", "fold_index", "participant_id"],
        "rank_incomplete_configs": False,
        "independent_test_available": False,
        "metric_prefix": "oof_validation_",
        "calibration_metrics": ["multiclass_brier", "expected_calibration_error"],
    }
    raw_statistics = evaluation.get("statistics", {})
    raw_ranking = evaluation.get("ranking", {})
    if not isinstance(raw_statistics, Mapping) or not isinstance(raw_ranking, Mapping):
        raise TypeError("evaluation statistics and ranking must be mappings")
    evaluation["statistics"] = {**statistics_defaults, **dict(raw_statistics)}
    evaluation["ranking"] = {**ranking_defaults, **dict(raw_ranking)}
    for name, value in evaluation_defaults.items():
        evaluation.setdefault(name, value)
    data["evaluation"] = evaluation


def _validate_evaluation_config(data: Mapping[str, Any]) -> None:
    """Validate reporting parameters without freezing their numerical budgets."""

    evaluation = _strict_mapping(data["evaluation"], "evaluation")
    evaluation_keys = {
        "unit", "primary_metric", "metrics", "confidence_interval",
        "paired_delta_key", "rank_incomplete_configs",
        "independent_test_available", "metric_prefix", "calibration_metrics",
        "statistics", "ranking",
    }
    if set(evaluation) != evaluation_keys:
        raise ValueError("evaluation key mismatch")
    if evaluation["unit"] != "participant":
        raise ValueError("evaluation unit must remain participant")
    expected_evaluation_protocol = {
        "unit": "participant",
        "primary_metric": "balanced_accuracy",
        "metrics": [
            "balanced_accuracy", "macro_f1", "per_class_precision_recall_f1",
            "worst_class_recall", "worst_class_f1", "confusion_matrix",
            "coverage",
        ],
        "confidence_interval": "participant_cluster_bootstrap_two_sided_95",
        "paired_delta_key": ["repeat_index", "fold_index", "participant_id"],
        "metric_prefix": "oof_validation_",
        "calibration_metrics": ["multiclass_brier", "expected_calibration_error"],
    }
    for name, expected in expected_evaluation_protocol.items():
        if evaluation[name] != expected:
            raise ValueError(f"evaluation.{name} is not an implemented protocol")
    if not isinstance(evaluation["rank_incomplete_configs"], bool):
        raise ValueError("evaluation.rank_incomplete_configs must be boolean")
    if evaluation["rank_incomplete_configs"]:
        raise ValueError("incomplete configurations cannot enter ranking")
    statistics = _strict_mapping(evaluation["statistics"], "evaluation.statistics")
    statistics_keys = {
        "cluster_unit", "bootstrap_replicates", "confidence_interval",
        "lcb95_percentile", "lcb95_metrics", "paired_permutation_replicates",
        "paired_exchange_unit", "multiplicity_correction", "seed",
        "affects_automatic_selection",
    }
    if set(statistics) != statistics_keys:
        raise ValueError("evaluation.statistics key mismatch")
    if statistics["cluster_unit"] != "participant_with_all_five_repeat_oof_predictions":
        raise ValueError("statistics cluster unit must preserve participant repeats")
    if statistics["confidence_interval"] != "two_sided_95_percentile":
        raise ValueError("unsupported confidence interval module")
    for name in ("bootstrap_replicates", "paired_permutation_replicates"):
        value = statistics[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"evaluation.statistics.{name} must be a positive integer")
    seed = statistics["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFF_FFFF:
        raise ValueError("evaluation.statistics.seed must be in [0,2^32-1]")
    percentile = statistics["lcb95_percentile"]
    if isinstance(percentile, bool) or not isinstance(percentile, (int, float)):
        raise ValueError("evaluation.statistics.lcb95_percentile must be numeric")
    percentile = float(percentile)
    if percentile != 2.5:
        raise ValueError("two-sided 95% interval requires lower percentile 2.5")
    if statistics["lcb95_metrics"] != [
        "participant_level_mean_balanced_accuracy",
        "participant_level_mean_macro_f1",
    ]:
        raise ValueError("evaluation.statistics.lcb95_metrics protocol drift")
    if statistics["paired_exchange_unit"] != "participant":
        raise ValueError("paired exchange unit must remain participant")
    if statistics["multiplicity_correction"] != "holm_within_comparison_family":
        raise ValueError("unsupported multiplicity correction module")
    if statistics["affects_automatic_selection"] is not False:
        raise ValueError("statistics cannot silently trigger automatic selection")

    ranking = _strict_mapping(evaluation["ranking"], "evaluation.ranking")
    ranking_keys = {
        "sort_key", "max_qualified_per_comparison_group",
        "automatic_final_selection", "manual_multiple_final_versions_allowed",
        "preserve_ablation_provenance",
    }
    if set(ranking) != ranking_keys:
        raise ValueError("evaluation.ranking key mismatch")
    expected_ranking = {
        "sort_key": "participant_level_mean_balanced_accuracy",
        "max_qualified_per_comparison_group": 10,
        "automatic_final_selection": False,
        "manual_multiple_final_versions_allowed": True,
        "preserve_ablation_provenance": True,
    }
    if ranking != expected_ranking:
        raise ValueError("evaluation.ranking is declarative and cannot select automatically")
    if evaluation["independent_test_available"] is not False:
        raise ValueError("the current cohort is OOF validation, not an independent test")


def _validate_output_policy(data: Mapping[str, Any]) -> None:
    """Bind legacy output flags to the mandatory evidence writer behavior.

    The experiment writer always publishes strict JSON plus all hierarchy OOF
    Parquet contracts into a new directory.  These source-YAML fields are kept
    for compatibility, but they are invariants rather than switches until a
    corresponding writer branch exists.
    """

    output = _strict_mapping(data["output"], "output")
    required = {
        "root", "overwrite_existing", "strict_json", "write_parquet",
        "parquet_missing_dependency_action", "write_window_oof",
        "write_file_oof", "write_subject_oof", "write_member_oof",
    }
    optional = {"formal_ablation_materialization"}
    missing = sorted(required - set(output))
    unknown = sorted(set(output) - required - optional)
    if missing or unknown:
        raise ValueError(
            f"output policy key mismatch: missing={missing}, unknown={unknown}"
        )
    invariants = {
        "root": "artifacts/runs",
        "overwrite_existing": False,
        "strict_json": True,
        "write_parquet": True,
        "parquet_missing_dependency_action": "fail_closed",
        "write_window_oof": True,
        "write_file_oof": True,
        "write_subject_oof": True,
    }
    for name, expected in invariants.items():
        observed = output[name]
        matches = observed is expected if isinstance(expected, bool) else observed == expected
        if not matches:
            raise ValueError(
                f"output.{name} is a mandatory writer invariant ({expected!r}), "
                "not an implemented switch"
            )
    model = _strict_mapping(data["model"], "model")
    expected_member_oof = "member_seeds" in model
    if output["write_member_oof"] is not expected_member_oof:
        raise ValueError(
            "output.write_member_oof is derived from the model ensemble "
            f"capability and must be {expected_member_oof!r}"
        )


def _validate_v2_protocol(data: dict[str, Any]) -> None:
    """Validate data identity plus independently configured runtime modules."""

    splits = _strict_mapping(data["splits"], "splits")
    if (
        splits.get("n_splits") != 5
        or splits.get("n_repeats") != 5
        or tuple(splits.get("split_seeds", ())) != V2_SPLIT_SEEDS
        or splits.get("runtime_recompute") is not False
    ):
        raise ValueError("V2 formal configs require the frozen 5x5 participant registry")
    training = _strict_mapping(data["training"], "training")
    # One authority validates optimizer, sampler, class weighting, balance line,
    # and every numeric training range.  Config validation must never accept a
    # value that the runtime later ignores or dispatches differently.
    from .training.trainer import TrainingConfig

    resolved_training = TrainingConfig.from_mapping(training)
    data["training"] = resolved_training.to_mapping()
    from .training.aggregation import canonical_role_family

    selected_role_families = {
        canonical_role_family(role) for role in data["roles"]
    }
    configured_classifier_families = set(
        resolved_training.classifier_role_families
    )
    if not configured_classifier_families <= selected_role_families:
        missing = sorted(
            configured_classifier_families - selected_role_families
        )
        raise ValueError(
            "training.classifier_role_families must be represented by roles; "
            f"missing selectors for {missing}"
        )
    _validate_output_policy(data)
    _validate_evaluation_config(data)
    _validate_v2_signal_normalization(data)
    _validate_v2_feature_schemas(data)
    from .module_registry import (
        model_factory_contract,
        resolve_artifact_config,
        resolve_peak_detector_config,
        resolve_window_config,
        validate_model_config,
        validate_window_profiles_for_representation,
    )

    resolved_artifact = resolve_artifact_config(
        _strict_mapping(data["artifact"], "artifact")
    )
    if resolved_artifact["denoiser_enabled"]:
        representation_mode = str(data["representation_mode"])
        policy = str(resolved_artifact["degraded_policy"])
        if representation_mode == "feature_vector":
            if policy != "denoise_then_extract_rate_features":
                raise ValueError(
                    "feature-vector rate recovery requires "
                    "degraded_policy='denoise_then_extract_rate_features'"
                )
        elif policy != "denoise_then_compare_rate_exclude":
            raise ValueError(
                "raw, feature-matrix, and fusion denoiser execution is "
                "diagnostic-only and requires degraded_policy="
                "'denoise_then_compare_rate_exclude'"
            )
    resolve_peak_detector_config(_strict_mapping(data["signal"], "signal"))
    data["windows"] = validate_window_profiles_for_representation(
        _strict_mapping(data["windows"], "windows"),
        str(data["representation_mode"]),
        list(_strict_mapping(data["features"], "features")["enabled_groups"]),
    )
    resolve_window_config(_strict_mapping(data["windows"], "windows"))
    _validate_v2_dl_resampling(data)
    model = _strict_mapping(data["model"], "model")
    validate_model_config(
        model,
        str(data["representation_mode"]),
    )
    model_contract = model_factory_contract(str(model["model_id"]))
    resolved_training.validate_for_execution_backend(
        str(model_contract["execution_backend"])
    )
    _validate_model_window_contract(data)
    _validate_formal_ablation_materialization(data)


def _validate_v2_dl_resampling(data: Mapping[str, Any]) -> None:
    """Bind generic DL resampling to raw/fusion and named presets to raw only."""

    from .signal.resample import validate_dl_resampling_config

    signal = _strict_mapping(data["signal"], "signal")
    dl = validate_dl_resampling_config(signal.get("dl_resampling"))
    mode = str(data["representation_mode"])
    case_id = dl.get("case_id")
    if case_id is not None and mode != "raw":
        raise ValueError(
            "named fixed-kernel signal.dl_resampling case_id requires raw "
            "representation"
        )
    if bool(dl["enabled"]) and mode not in {"raw", "fusion"}:
        raise ValueError(
            "generic signal.dl_resampling is executable only for raw or fusion "
            "representations"
        )
    if not bool(dl["enabled"]) and case_id is None:
        return
    raw_window = _strict_mapping(
        _strict_mapping(data["windows"], "windows").get("raw_dl"),
        "windows.raw_dl",
    )
    if round(float(raw_window["length_s"]) * float(dl["target_fs_hz"])) < 2:
        raise ValueError(
            "DL target/window combination must contain at least two samples"
        )


def _validate_model_window_contract(data: Mapping[str, Any]) -> None:
    """Reject model/window combinations that cannot reach a valid forward pass.

    The window planner and DL resampler are independently configurable modules,
    but their resolved output length is an input-shape contract for temporal
    models.  Validate that contract before any fold data are loaded.
    """

    representation_mode = str(data["representation_mode"])
    if representation_mode not in {"raw", "fusion"}:
        return

    from .models.factory import (
        SHAPEFORMER_EXPERIMENTAL_NUMERIC_DEFAULTS,
        SHAPEFORMER_LEGACY_EFFECT_SIZE_DEFAULTS,
        SHAPEFORMER_REFERENCE_NUMERIC_DEFAULTS,
    )
    from .signal.resample import validate_dl_resampling_config

    model = _strict_mapping(data["model"], "model")
    windows = _strict_mapping(data["windows"], "windows")
    raw_window = _strict_mapping(windows.get("raw_dl"), "windows.raw_dl")
    signal = _strict_mapping(data["signal"], "signal")
    dl = validate_dl_resampling_config(signal.get("dl_resampling"))
    effective_fs_hz = float(dl["target_fs_hz"])
    exact_samples = float(raw_window["length_s"]) * effective_fs_hz
    sequence_samples = int(round(exact_samples))
    if sequence_samples < 1:
        raise ValueError(
            "windows.raw_dl.length_s and effective DL sampling rate yield no samples"
        )

    model_id = str(model["model_id"])
    model_options: Mapping[str, Any] = model
    model_path = "model"
    if model_id == "FileBagFusion":
        from .models.factory import normalize_fusion_signal_encoder_config

        model_options = normalize_fusion_signal_encoder_config(
            model.get("signal_encoder")
        )
        model_id = str(model_options["canonical_model_name"])
        model_path = "model.signal_encoder"

    if model_id in {"CompactCNN1D", "FileBagFusionCompact"}:
        if raw_window["padding"] != "none_complete_windows_only":
            raise ValueError(
                f"{model_path}={model_id} does not implement mask-aware "
                "padded-window pooling; "
                "windows.raw_dl.padding must be none_complete_windows_only"
            )
        if model_path == "model.signal_encoder":
            pool_field = "pool_sizes"
            pool_sizes = tuple(
                int(value) for value in model_options.get(pool_field, (4, 4))
            )
        else:
            pool_field = (
                "pool_sizes" if model_id == "CompactCNN1D" else "signal_pool_sizes"
            )
            pool_sizes = tuple(int(value) for value in model[pool_field])
        minimum_samples = math.prod(pool_sizes)
        if sequence_samples < minimum_samples:
            raise ValueError(
                f"{model_id} requires at least {minimum_samples} effective samples "
                f"for its configured {pool_field} chain; got {sequence_samples}"
            )

    shapeformer_ids = {
        "ShapeFormerChannelSpecificOSD",
        "ShapeFormerChannelSpecificScalarDistanceAblation",
        "ShapeFormerEffectSizeFixedV1",
        "ShapeFormerLegacyEffectSizePort",
    }
    if model_id not in shapeformer_ids:
        return
    configured_fs_hz = float(
        model_options.get(
            "input_fs_hz",
            (
                SHAPEFORMER_REFERENCE_NUMERIC_DEFAULTS["input_fs_hz"]
                if model_id == "ShapeFormerChannelSpecificOSD"
                else SHAPEFORMER_LEGACY_EFFECT_SIZE_DEFAULTS["input_fs_hz"]
                if model_id == "ShapeFormerLegacyEffectSizePort"
                else SHAPEFORMER_EXPERIMENTAL_NUMERIC_DEFAULTS["input_fs_hz"]
            ),
        )
    )
    if not math.isclose(
        configured_fs_hz, effective_fs_hz, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"{model_path}.input_fs_hz must match the effective "
            "signal.dl_resampling "
            "target_fs_hz"
        )

    if model_id in {
        "ShapeFormerChannelSpecificOSD",
        "ShapeFormerLegacyEffectSizePort",
    }:
        defaults = (
            SHAPEFORMER_REFERENCE_NUMERIC_DEFAULTS
            if model_id == "ShapeFormerChannelSpecificOSD"
            else SHAPEFORMER_LEGACY_EFFECT_SIZE_DEFAULTS
        )
        configured_sequence = int(
            model_options.get(
                "sequence_length_samples",
                defaults["sequence_length_samples"],
            )
        )
        if configured_sequence != sequence_samples:
            raise ValueError(
                f"{model_path}.sequence_length_samples must equal round("
                "windows.raw_dl.length_s * effective DL sampling rate)"
            )
        if configured_sequence < 3:
            raise ValueError(
                "ShapeFormer channel-specific OSD discovery requires at least "
                "three effective samples"
            )
        local_kernel = int(
            model_options.get(
                "local_kernel_width_samples",
                defaults["local_kernel_width_samples"],
            )
        )
        if local_kernel > configured_sequence:
            raise ValueError(
                f"{model_path}.local_kernel_width_samples cannot exceed "
                f"{model_path}.sequence_length_samples"
            )
        if model_id == "ShapeFormerLegacyEffectSizePort":
            if raw_window["padding"] != "none_complete_windows_only":
                raise ValueError(
                    "legacy effect-size ShapeFormer requires "
                    "windows.raw_dl.padding=none_complete_windows_only"
                )
            shapelet_length = int(
                model_options.get(
                    "shapelet_length_samples",
                    defaults["shapelet_length_samples"],
                )
            )
            if shapelet_length > configured_sequence:
                raise ValueError(
                    f"{model_path}.shapelet_length_samples cannot exceed "
                    f"{model_path}.sequence_length_samples"
                )
        return

    patch_size = int(
        model_options.get(
            "patch_size_samples",
            SHAPEFORMER_EXPERIMENTAL_NUMERIC_DEFAULTS["patch_size_samples"],
        )
    )
    if patch_size > sequence_samples:
        raise ValueError(
            f"{model_path}.patch_size_samples cannot exceed the effective raw-window "
            "sequence length"
        )
    if model_id == "ShapeFormerChannelSpecificScalarDistanceAblation":
        if sequence_samples < 3:
            raise ValueError(
                "ShapeFormer channel-specific OSD discovery requires at least "
                "three effective samples"
            )
        return
    shapelet_length = int(model_options.get("shapelet_length_samples", 128))
    if shapelet_length > sequence_samples:
        raise ValueError(
            f"{model_path}.shapelet_length_samples cannot exceed the effective "
            "raw-window sequence length"
        )


def _materialize_v2_defaults(data: dict[str, Any]) -> None:
    """Persist effective module defaults before hashing or runtime dispatch.

    Source YAML remains available through ``PipelineConfig.source_path``.  The
    payload and its SHA-256 deliberately describe the complete effective
    configuration so omitted defaults cannot disappear from ``section()``,
    ``to_dict()``, or downstream provenance.
    """

    from .data.schema import REGISTERED_ROLES
    from .module_registry import (
        derived_mask_aware_pooling,
        derived_model_ensemble_size,
        derived_model_variant,
        materialize_model_architecture,
        normalize_window_config,
        validate_legacy_ensemble_metadata,
    )
    from .training.trainer import TrainingConfig

    roles = data.get("roles")
    if (
        isinstance(roles, list)
        and roles
        and len(roles) == len(set(roles))
        and all(role in REGISTERED_ROLES for role in roles)
    ):
        # Role selectors are consumed as a set. Persist their registered order
        # so permutations cannot create hash-only experiment identities.
        data["roles"] = [role for role in REGISTERED_ROLES if role in roles]

    defaults = TrainingConfig().to_mapping()
    declared = _strict_mapping(data["training"], "training")
    legacy_weighting_aliases = {
        "outer_train_inverse_frequency": ("inverse_frequency", "participant"),
        "outer_train_window_inverse_frequency": ("inverse_frequency", "row"),
    }
    declared_weighting = declared.get("class_weighting")
    if declared_weighting in legacy_weighting_aliases:
        canonical_weighting, implied_basis = legacy_weighting_aliases[
            str(declared_weighting)
        ]
        explicit_basis = declared.get("class_count_basis")
        if explicit_basis is not None and explicit_basis != implied_basis:
            raise ValueError(
                f"training.class_weighting={declared_weighting} implies "
                f"class_count_basis={implied_basis}"
            )
        declared["class_weighting"] = canonical_weighting
        declared["class_count_basis"] = implied_basis
    if "optimizer_parameters" not in declared:
        # Resolve optimizer-specific defaults only after the selected optimizer
        # is known.  Carrying Adam's materialized mapping into an SGD/RMSprop
        # declaration would turn a default into an accidental compatibility gate.
        defaults["optimizer_parameters"] = {}
    data["training"] = {**defaults, **declared}
    model = _strict_mapping(data["model"], "model")
    validate_legacy_ensemble_metadata(model)
    model["ensemble_size"] = derived_model_ensemble_size(model)
    model["variant"] = derived_model_variant(model)
    mask_aware_pooling = derived_mask_aware_pooling(model)
    if mask_aware_pooling is not None:
        model["mask_aware_pooling"] = mask_aware_pooling
    model.pop("comparison_only", None)
    model.pop("member_seed_roster_id", None)
    model["architecture_parameters"] = materialize_model_architecture(
        model,
        str(data["representation_mode"]),
    )
    data["model"] = model
    _materialize_signal_preprocessing_defaults(data)
    _materialize_signal_normalization_defaults(data)
    _materialize_peak_detector_defaults(data)
    data["windows"] = normalize_window_config(
        _strict_mapping(data["windows"], "windows")
    )
    _materialize_dl_resampling_defaults(data)
    _materialize_feature_defaults(data)
    _materialize_routing_defaults(data)
    _materialize_quality_defaults(data)
    _materialize_artifact_defaults(data)
    _materialize_aggregation_defaults(data)
    _materialize_evaluation_defaults(data)


def _validate_formal_ablation_materialization(data: Mapping[str, Any]) -> None:
    """Validate one-factor provenance and reject hidden/cartesian profiles."""

    from .models import normalize_model_id
    from .models.time_scale import fixed_kernel_case
    from .signal.resample import validate_dl_resampling_config

    output = _strict_mapping(data["output"], "output")
    identity = output.get("formal_ablation_materialization")
    signal = _strict_mapping(data["signal"], "signal")
    training = _strict_mapping(data["training"], "training")
    model = _strict_mapping(data["model"], "model")
    dl = validate_dl_resampling_config(signal["dl_resampling"])
    raw_window = _strict_mapping(
        _strict_mapping(data["windows"], "windows").get("raw_dl"),
        "windows.raw_dl",
    )
    filter_pair = (
        float(signal["ppg_filter"]["low_hz"]),
        float(signal["ppg_filter"]["high_hz"]),
    )
    gravity = str(signal["imu"]["gravity_method"])
    detector_id = str(signal["peak_detector"]["detector_id"])
    balance_pair = (
        str(training["training_balance"]),
        str(data["aggregation"]["balance_line"]),
    )
    nonreference = {
        "epoch": (
            training["epoch_profile"], int(training["fixed_epochs"])
        ) != ("default_10", 10),
        "filter": filter_pair != (0.2, 8.0),
        "gravity": gravity != "profile_a_lowpass_0p3hz",
        "peak_detector": detector_id != "msptdfast_v2_3_python_port",
        "aggregation": balance_pair
        != ("equal_role_families", "line_b_equal_role_families"),
        "sampler": training["sampler"]
        != "exhaustive_shuffle_without_replacement",
        "class_count_basis": training["class_count_basis"] != "row",
        "fixed_kernel": (
            dl.get("case_id") is not None
            and not str(dl.get("case_id")).endswith("__reference")
        ),
    }
    if identity is None:
        # Arbitrary valid module combinations are ordinary V2 configurations.
        # ``formal_ablation_materialization`` is only a verifier for files
        # explicitly emitted from the historical named-preset catalogue; it is
        # not an authorization gate for non-default runtime parameters.
        return
    identity = _strict_mapping(identity, "formal_ablation_materialization")
    _require_exact_keys(
        identity,
        {
            "schema_version", "family", "profile_id", "catalog_role",
            "base_config_path", "base_config_sha256", "profile_catalog_sha256",
            "single_factor_only", "automatic_execution",
            "scientific_execution_completed",
        },
        context="formal_ablation_materialization",
    )
    if (
        identity["schema_version"]
        != "ppg_frailty.formal_ablation_materialization.v2"
        or identity["family"] not in {
            "deep_fixed_epoch", "direct_filter", "imu_gravity",
            "fixed_kernel_samples", "aggregation_balance", "peak_detector",
            "sampler", "class_count_basis",
        }
        or identity["single_factor_only"] is not True
        or identity["automatic_execution"] is not False
        or identity["scientific_execution_completed"] is not False
    ):
        raise ValueError("formal ablation materialization contract invalid")
    for name in ("base_config_sha256", "profile_catalog_sha256"):
        digest = str(identity[name])
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"formal ablation {name} must be SHA-256")
    family = str(identity["family"])
    active = {name for name, enabled in nonreference.items() if enabled}
    expected_active = {
        "deep_fixed_epoch": {"epoch"},
        "direct_filter": {"filter"},
        "imu_gravity": {"gravity"},
        "peak_detector": {"peak_detector"},
        "fixed_kernel_samples": {"fixed_kernel"},
        "aggregation_balance": {"aggregation"},
        "sampler": {"sampler"},
        "class_count_basis": {"class_count_basis"},
    }[family]
    if str(identity["catalog_role"]) == "reference":
        expected_active = set()
    if active != expected_active:
        raise ValueError("formal ablation config is not a single-factor profile")
    profile_id = str(identity["profile_id"])
    if family == "deep_fixed_epoch":
        expected = {
            "epoch_7_ablation": ("ablation_7", 7),
            "default_epoch_10": ("default_10", 10),
            "epoch_15_ablation": ("ablation_15", 15),
        }.get(profile_id)
        _canonical, machine = normalize_model_id(str(model["model_id"]))
        if expected is None or machine in {
            "logistic_regression", "rbf_svm", "extra_trees",
        } or (training["epoch_profile"], int(training["fixed_epochs"])) != expected:
            raise ValueError("deep epoch materialization identity drift")
    elif family == "direct_filter":
        expected = {
            "direct_filter_0p2_to_8hz": (0.2, 8.0),
            "direct_filter_0p5_to_5hz_ablation": (0.5, 5.0),
        }.get(profile_id)
        if expected is None or filter_pair != expected:
            raise ValueError("direct filter materialization identity drift")
    elif family == "imu_gravity":
        expected = {
            "profile_a_lowpass_0p3hz": "profile_a_lowpass_0p3hz",
            "calibrated_roll_pitch_ekf_ablation": "calibrated_roll_pitch_ekf",
            "sensor_filter_only_no_gravity_removal_ablation": (
                "sensor_filter_only_no_gravity_removal"
            ),
        }.get(profile_id)
        if expected is None or gravity != expected:
            raise ValueError("IMU gravity materialization identity drift")
    elif family == "sampler":
        expected = {
            "exhaustive_shuffle_without_replacement":
                "exhaustive_shuffle_without_replacement",
            "line_b_weighted_sampler_ablation": "balance_line_weighted_v2",
        }.get(profile_id)
        if expected is None or training["sampler"] != expected:
            raise ValueError("sampler materialization identity drift")
    elif family == "class_count_basis":
        expected = {
            "row_count_class_weights": "row",
            "participant_count_class_weights_ablation": "participant",
        }.get(profile_id)
        if (
            expected is None
            or training["class_weighting"] != "inverse_frequency"
            or training["class_count_basis"] != expected
        ):
            raise ValueError("class-count-basis materialization identity drift")
    elif family == "peak_detector":
        expected = {
            "aboy_project_v1": "aboy_project_v1",
            "aboy_project_v2": "aboy_project_v2",
            "dual_polarity_prominence_v1_ablation":
                "dual_polarity_prominence_v1_ablation",
            "msptdfast_v2_3_python_port": "msptdfast_v2_3_python_port",
        }.get(profile_id)
        if expected is None or detector_id != expected:
            raise ValueError("peak-detector materialization identity drift")
        if profile_id == "msptdfast_v2_3_python_port":
            from .peaks.msptdfast_v2 import DEFAULT_PARAMETERS

            if data["signal"]["peak_detector"].get("parameters") != DEFAULT_PARAMETERS:
                raise ValueError("MSPTDfast materialization parameter drift")
    elif family == "aggregation_balance":
        expected = {
            "role_aware_equal_roles": (
                "equal_role_families", "line_b_equal_role_families"
            ),
            "equal_files_line_a_ablation": (
                "equal_files", "line_a_equal_files"
            ),
        }.get(profile_id)
        if expected is None or balance_pair != expected:
            raise ValueError("aggregation-balance materialization identity drift")
    else:
        case = fixed_kernel_case(profile_id)
        _canonical, machine = normalize_model_id(str(model["model_id"]))
        expected_machine = (
            "compact_cnn" if case.model_name == "CompactCNN1D" else "inception_full"
        )
        if (
            data["representation_mode"] != "raw"
            or machine != expected_machine
            or dl.get("case_id") != case.case_id
            or float(data["windows"]["raw_dl"]["length_s"])
            != float(case.raw_window_seconds)
        ):
            raise ValueError("fixed-kernel materialization identity drift")


def validate_config_payload(payload: Mapping[str, Any], *, allow_legacy: bool = False) -> dict[str, Any]:
    """执行 fail-closed 配置验证 / Validate a formal V2 or explicit legacy config."""

    data = _strict_mapping(payload, "config")
    if data.get("schema_version") == V2_SCHEMA_VERSION:
        data.setdefault("routing", {})
    expected_keys = (
        LEGACY_TOP_LEVEL_KEYS
        if data.get("schema_version") == LEGACY_SCHEMA_VERSION
        else TOP_LEVEL_KEYS
    )
    _require_exact_keys(data, expected_keys, context="config")
    if data.get("schema_version") == V2_SCHEMA_VERSION:
        _materialize_v2_defaults(data)
    _validate_common_payload(data)
    schema = data["schema_version"]
    if schema == LEGACY_SCHEMA_VERSION:
        if not allow_legacy:
            raise ValueError("legacy V1 config is provenance-only; pass allow_legacy=True explicitly")
        aggregation = _strict_mapping(data["aggregation"], "aggregation")
        if aggregation.get("hierarchy") != ["window", "file", "role", "participant"]:
            raise ValueError("legacy V1 aggregation hierarchy drift")
        return data
    if schema != V2_SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    config_id = str(data["config_id"])
    if (
        not config_id.strip()
        or config_id != config_id.strip()
        or config_id in {".", ".."}
        or "\x00" in config_id
        or "/" in config_id
        or "\\" in config_id
    ):
        raise ValueError(
            "V2 config_id must be a non-empty path-safe identifier"
        )
    _validate_v2_quality(data)
    _validate_v2_balance(data)
    _validate_v2_protocol(data)
    return data


def load_config(path: str | Path, *, allow_legacy: bool = False) -> PipelineConfig:
    """加载正式 V2 或显式 legacy V1 / Load formal V2 or explicit legacy V1."""

    source = Path(path)
    source_text = source.read_text(encoding="utf-8")
    try:
        # JSON is a supported configuration serialization regardless of suffix.
        # Parsing it first preserves scientific-notation numbers such as 1e-08;
        # PyYAML otherwise treats that valid JSON token as a string.
        payload = json.loads(source_text)
    except json.JSONDecodeError:
        payload = yaml.safe_load(source_text)
    data = validate_config_payload(_strict_mapping(payload, "config"), allow_legacy=allow_legacy)
    digest = hashlib.sha256(canonical_json_bytes(data)).hexdigest()
    return PipelineConfig(data, source.as_posix(), digest)


def load_formal_experiment_catalog(path: str | Path) -> dict[str, Any]:
    """Load the declared active candidates and fixed comparison entries."""

    source = Path(path)
    payload = _strict_mapping(
        yaml.safe_load(source.read_text(encoding="utf-8")),
        "formal_experiment_catalog",
    )
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "catalog_id",
            "pipeline_generation",
            "execution_policy",
            "entries",
        },
        context="formal_experiment_catalog",
    )
    if payload["schema_version"] != V2_FORMAL_CATALOG_SCHEMA:
        raise ValueError("unsupported formal experiment catalog schema")
    if payload["pipeline_generation"] != "final_pipeline_v2":
        raise ValueError("formal catalog must be bound to final_pipeline_v2")
    policy = _strict_mapping(payload["execution_policy"], "execution_policy")
    if set(policy) != {
        "auto_run", "candidate_count", "matched_comparator_count",
        "ensemble_comparison_count", "default_balance_line",
        "selectable_balance_lines", "materialization_only",
    } or any(
        isinstance(policy[name], bool) or not isinstance(policy[name], int)
        or int(policy[name]) < 0
        for name in (
            "candidate_count", "matched_comparator_count",
            "ensemble_comparison_count",
        )
    ) or {
        "auto_run": policy["auto_run"],
        "default_balance_line": policy["default_balance_line"],
        "selectable_balance_lines": policy["selectable_balance_lines"],
        "materialization_only": policy["materialization_only"],
    } != {
        "auto_run": False,
        "default_balance_line": "line_b",
        "selectable_balance_lines": ["line_b", "line_a"],
        "materialization_only": True,
    }:
        raise ValueError("formal catalog execution policy drifted")
    raw_entries = payload["entries"]
    expected_total = sum(
        int(policy[name])
        for name in (
            "candidate_count", "matched_comparator_count",
            "ensemble_comparison_count",
        )
    )
    if not isinstance(raw_entries, list) or len(raw_entries) != expected_total:
        raise ValueError("formal catalog entry count differs from execution policy")
    from .module_registry import validate_model_config

    entries: list[dict[str, Any]] = []
    entry_ids: set[str] = set()
    stems: set[str] = set()
    for raw in raw_entries:
        entry = _strict_mapping(raw, "catalog_entry")
        _require_exact_keys(
            entry,
            {
                "entry_id",
                "config_stem",
                "representation_mode",
                "catalog_role",
                "model",
            },
            context="catalog_entry",
        )
        entry_id = str(entry["entry_id"])
        stem = str(entry["config_stem"])
        if (
            not entry_id
            or not stem
            or entry_id in entry_ids
            or stem in stems
        ):
            raise ValueError("catalog entry IDs/config stems must be non-empty and unique")
        entry_ids.add(entry_id)
        stems.add(stem)
        if entry["catalog_role"] not in {
            "reference_candidate",
            "ablation_candidate",
            "matched_comparator",
            "ensemble_comparison",
        }:
            raise ValueError("invalid formal catalog role")
        validate_model_config(
            _strict_mapping(entry["model"], f"{entry_id}.model"),
            str(entry["representation_mode"]),
        )
        entries.append(entry)
    ordinary_count = sum(
        entry["catalog_role"] in {"reference_candidate", "ablation_candidate"}
        for entry in entries
    )
    comparator_count = sum(
        entry["catalog_role"] == "matched_comparator" for entry in entries
    )
    ensemble_count = sum(
        entry["catalog_role"] == "ensemble_comparison" for entry in entries
    )
    if (ordinary_count, comparator_count, ensemble_count) != (
        int(policy["candidate_count"]),
        int(policy["matched_comparator_count"]),
        int(policy["ensemble_comparison_count"]),
    ):
        raise ValueError("formal catalogue count contract drifted")

    by_id = {str(entry["entry_id"]): entry for entry in entries}
    comparator_pairs = {
        "inception_full_member0_comparator": (
            "inception_full",
            "comparison_inception_full_member0_comparator",
            "raw",
        ),
    }
    for comparator_id, (
        ordinary_id,
        expected_stem,
        expected_mode,
    ) in comparator_pairs.items():
        comparator = by_id.get(comparator_id)
        ordinary = by_id.get(ordinary_id)
        if comparator is None or ordinary is None:
            raise ValueError("formal matched-comparator identity is missing")
        if (
            comparator["catalog_role"] != "matched_comparator"
            or comparator["config_stem"] != expected_stem
            or comparator["representation_mode"] != expected_mode
            or ordinary["representation_mode"] != expected_mode
        ):
            raise ValueError("formal matched-comparator identity drifted")
        ordinary_model = dict(ordinary["model"])
        comparator_model = dict(comparator["model"])
        if (
            ordinary_model.get("seed_policy")
            != "outer_cv_repeat_seed_equals_split_seed"
            or comparator_model.get("seed_policy")
            != "cv_fixed_member0_seed_50042_comparator"
        ):
            raise ValueError("formal matched-comparator seed policy drifted")
        comparator_model["seed_policy"] = ordinary_model["seed_policy"]
        if comparator_model != ordinary_model:
            raise ValueError(
                "formal matched comparator must share the ordinary architecture"
            )
    payload["entries"] = entries
    payload["catalog_sha256"] = hashlib.sha256(
        canonical_json_bytes(
            {key: value for key, value in payload.items() if key != "catalog_sha256"}
        )
    ).hexdigest()
    return payload


def load_formal_ablation_profiles(path: str | Path) -> dict[str, Any]:
    """Load the materialization-only single-factor V2 profile catalogue."""

    source = Path(path)
    payload = _strict_mapping(
        yaml.safe_load(source.read_text(encoding="utf-8")),
        "formal_ablation_profiles",
    )
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "catalog_id",
            "pipeline_generation",
            "execution_policy",
            "families",
        },
        context="formal_ablation_profiles",
    )
    if payload["schema_version"] != V2_FORMAL_ABLATION_PROFILES_SCHEMA:
        raise ValueError("unsupported formal ablation-profile schema")
    if payload["pipeline_generation"] != "final_pipeline_v2":
        raise ValueError("ablation profiles must be bound to final_pipeline_v2")
    if payload["execution_policy"] != {
        "auto_run": False,
        "materialization_only": True,
        "allow_cartesian_product": False,
        "safe_suite_execution": False,
    }:
        raise ValueError("formal ablation execution policy drifted")
    families = _strict_mapping(payload["families"], "ablation_profile_families")
    _require_exact_keys(
        families,
        {
            "aggregation_balance", "deep_fixed_epoch", "direct_filter",
            "imu_gravity", "fixed_kernel_samples", "peak_detector",
            "sampler", "class_count_basis",
        },
        context="ablation_profile_families",
    )
    expected_epoch = {
        "reference_profile_id": "default_epoch_10",
        "entries": [
            {"profile_id": "epoch_7_ablation", "fixed_epochs": 7, "catalog_role": "ablation", "auto_run": False},
            {"profile_id": "default_epoch_10", "fixed_epochs": 10, "catalog_role": "reference", "auto_run": False},
            {"profile_id": "epoch_15_ablation", "fixed_epochs": 15, "catalog_role": "ablation", "auto_run": False},
        ],
    }
    expected_aggregation = {
        "reference_profile_id": "role_aware_equal_roles",
        "entries": [
            {
                "profile_id": "role_aware_equal_roles",
                "training_balance": "equal_role_families",
                "balance_line": "line_b_equal_role_families",
                "hierarchy": ["window", "file", "role", "participant"],
                "catalog_role": "reference",
                "auto_run": False,
            },
            {
                "profile_id": "equal_files_line_a_ablation",
                "training_balance": "equal_files",
                "balance_line": "line_a_equal_files",
                "hierarchy": ["window", "file", "participant"],
                "catalog_role": "ablation",
                "auto_run": False,
            },
        ],
    }
    expected_filter = {
        "reference_profile_id": "direct_filter_0p2_to_8hz",
        "entries": [
            {"profile_id": "direct_filter_0p2_to_8hz", "low_hz": 0.2, "high_hz": 8.0, "catalog_role": "reference", "auto_run": False},
            {"profile_id": "direct_filter_0p5_to_5hz_ablation", "low_hz": 0.5, "high_hz": 5.0, "catalog_role": "ablation", "auto_run": False},
        ],
    }
    expected_sampler = {
        "reference_profile_id": "exhaustive_shuffle_without_replacement",
        "entries": [
            {
                "profile_id": "exhaustive_shuffle_without_replacement",
                "sampler": "exhaustive_shuffle_without_replacement",
                "catalog_role": "reference",
                "auto_run": False,
            },
            {
                "profile_id": "line_b_weighted_sampler_ablation",
                "sampler": "balance_line_weighted_v2",
                "catalog_role": "ablation",
                "auto_run": False,
            },
        ],
    }
    expected_class_count_basis = {
        "reference_profile_id": "row_count_class_weights",
        "entries": [
            {
                "profile_id": "row_count_class_weights",
                "class_weighting": "inverse_frequency",
                "class_count_basis": "row",
                "catalog_role": "reference",
                "auto_run": False,
            },
            {
                "profile_id": "participant_count_class_weights_ablation",
                "class_weighting": "inverse_frequency",
                "class_count_basis": "participant",
                "catalog_role": "ablation",
                "auto_run": False,
            },
        ],
    }
    expected_imu = {
        "reference_profile_id": "profile_a_lowpass_0p3hz",
        "silent_fallback_forbidden": True,
        "entries": [
            {"profile_id": "profile_a_lowpass_0p3hz", "method": "profile_a_lowpass_0p3hz", "catalog_role": "reference", "auto_run": False},
            {"profile_id": "calibrated_roll_pitch_ekf_ablation", "method": "calibrated_roll_pitch_ekf", "catalog_role": "ablation", "auto_run": False},
            {"profile_id": "sensor_filter_only_no_gravity_removal_ablation", "method": "sensor_filter_only_no_gravity_removal", "catalog_role": "ablation", "auto_run": False},
        ],
    }
    expected_peak_detector = {
        "reference_profile_id": "msptdfast_v2_3_python_port",
        "silent_fallback_forbidden": True,
        "entries": [
            {
                "profile_id": "msptdfast_v2_3_python_port",
                "detector_id": "msptdfast_v2_3_python_port",
                "parameters": {
                    "target_downsample_hz": 20.0,
                    "minimum_heart_rate_bpm": 30.0,
                    "window_s": 6.0,
                    "overlap_fraction": 0.2,
                },
                "catalog_role": "reference",
                "auto_run": False,
            },
            {
                "profile_id": "aboy_project_v2",
                "detector_id": "aboy_project_v2",
                "catalog_role": "ablation",
                "auto_run": False,
            },
            {
                "profile_id": "dual_polarity_prominence_v1_ablation",
                "detector_id": "dual_polarity_prominence_v1_ablation",
                "catalog_role": "ablation",
                "auto_run": False,
            },
        ],
    }
    if families["aggregation_balance"] != expected_aggregation:
        raise ValueError("aggregation-balance profiles drifted")
    if families["deep_fixed_epoch"] != expected_epoch:
        raise ValueError("fixed epoch profiles drifted")
    if families["direct_filter"] != expected_filter:
        raise ValueError("direct filter profiles drifted")
    if families["sampler"] != expected_sampler:
        raise ValueError("sampler profiles drifted")
    if families["class_count_basis"] != expected_class_count_basis:
        raise ValueError("class-count-basis profiles drifted")
    if families["imu_gravity"] != expected_imu:
        raise ValueError("IMU gravity profiles drifted")
    if families["peak_detector"] != expected_peak_detector:
        raise ValueError("peak detector profiles drifted")
    from .models.time_scale import build_fixed_kernel_resampling_cases

    expected_cases = [
        {
            "case_id": case.case_id,
            "model_name": case.model_name,
            "dl_fs_hz": case.dl_fs_hz,
            "raw_window_seconds": case.raw_window_seconds,
            "sequence_length_samples": case.sequence_length_samples,
            "kernel_samples": list(case.kernel_samples),
            "dilation": case.dilation,
            "catalog_role": (
                "reference" if case.case_id.endswith("__reference") else "ablation"
            ),
            "auto_run": False,
        }
        for case in build_fixed_kernel_resampling_cases()
    ]
    expected_fixed = {
        "family_id": "fixed_kernel_samples_resampling_ablation",
        "eligible_models": ["CompactCNN1D", "InceptionTimeFull"],
        "physical_time_matched_claim": "forbidden",
        "cases": expected_cases,
    }
    if families["fixed_kernel_samples"] != expected_fixed:
        raise ValueError("fixed-kernel 12-case profiles drifted")
    payload["catalog_sha256"] = hashlib.sha256(
        canonical_json_bytes(
            {key: value for key, value in payload.items() if key != "catalog_sha256"}
        )
    ).hexdigest()
    return payload


def materialize_formal_ablation_config(
    base_config_path: str | Path,
    *,
    family: str,
    profile_id: str,
    output_path: str | Path,
    profiles_path: str | Path,
) -> PipelineConfig:
    """Materialize exactly one registered comparison factor; never execute it."""

    base_path = Path(base_config_path).resolve()
    target = Path(output_path).resolve()
    pipeline_root = Path(profiles_path).resolve().parent.parent
    base_relative = base_path.relative_to(pipeline_root).as_posix()
    target.relative_to(pipeline_root)
    if target.exists():
        raise FileExistsError(f"ablation config overwrite forbidden: {target}")
    base = load_config(base_path)
    catalog = load_formal_ablation_profiles(profiles_path)
    if family not in {
        "deep_fixed_epoch", "direct_filter", "imu_gravity",
        "fixed_kernel_samples", "aggregation_balance", "peak_detector",
        "sampler", "class_count_basis",
    }:
        raise ValueError("unknown formal ablation family")
    payload = base.to_dict()
    from .models import normalize_model_id

    _canonical, machine_id = normalize_model_id(str(payload["model"]["model_id"]))
    estimator_ids = {
        "logistic_regression", "rbf_svm", "extra_trees",
    }
    selected: dict[str, Any]
    if family == "fixed_kernel_samples":
        from .models.time_scale import fixed_kernel_case

        case = fixed_kernel_case(profile_id)
        expected_machine = (
            "compact_cnn" if case.model_name == "CompactCNN1D" else "inception_full"
        )
        if payload["representation_mode"] != "raw" or machine_id != expected_machine:
            raise ValueError(
                "fixed-kernel case requires the matching raw CompactCNN/Inception config"
            )
        payload["windows"]["raw_dl"]["length_s"] = float(case.raw_window_seconds)
        resampling = payload["signal"]["dl_resampling"]
        resampling["case_id"] = case.case_id
        resampling["enabled"] = float(case.dl_fs_hz) != 400.0
        resampling["target_fs_hz"] = float(case.dl_fs_hz)
        if machine_id == "compact_cnn":
            dilations = [int(case.dilation)] * 3
            payload["model"]["dilations"] = dilations
            payload["model"]["architecture_parameters"]["dilations"] = dilations
        else:
            payload["model"]["dilation"] = int(case.dilation)
            payload["model"]["architecture_parameters"]["dilation"] = int(case.dilation)
        selected = {
            "profile_id": case.case_id,
            "catalog_role": (
                "reference" if case.case_id.endswith("__reference") else "ablation"
            ),
        }
    else:
        entries = catalog["families"][family]["entries"]
        matches = [dict(row) for row in entries if row["profile_id"] == profile_id]
        if len(matches) != 1:
            raise ValueError(f"unknown profile_id for {family}: {profile_id}")
        selected = matches[0]
        if selected.get("auto_run") is not False:
            raise ValueError("formal ablation profiles must never auto-run")
        if family == "deep_fixed_epoch":
            if machine_id in estimator_ids:
                raise ValueError("epoch profiles are deep-model-only")
            fixed = int(selected["fixed_epochs"])
            payload["training"]["fixed_epochs"] = fixed
            payload["training"]["epoch_profile"] = {
                7: "ablation_7", 10: "default_10", 15: "ablation_15"
            }[fixed]
        elif family == "direct_filter":
            low = float(selected["low_hz"])
            high = float(selected["high_hz"])
            payload["signal"]["ppg_filter"]["low_hz"] = low
            payload["signal"]["ppg_filter"]["high_hz"] = high
            payload["signal"]["analysis_view"].pop("direct_source", None)
        elif family == "imu_gravity":
            method = str(selected["method"])
            payload["signal"]["imu"]["gravity_method"] = method
            payload["signal"]["imu"]["comparison_method"] = {
                "calibrated_roll_pitch_ekf": "profile_a_lowpass_0p3hz",
                "profile_a_lowpass_0p3hz": "calibrated_roll_pitch_ekf",
                "sensor_filter_only_no_gravity_removal": (
                    "profile_a_lowpass_0p3hz"
                ),
            }[method]
        elif family == "sampler":
            payload["training"]["sampler"] = str(selected["sampler"])
        elif family == "class_count_basis":
            payload["training"]["class_weighting"] = str(
                selected["class_weighting"]
            )
            payload["training"]["class_count_basis"] = str(
                selected["class_count_basis"]
            )
        elif family == "peak_detector":
            payload["signal"]["peak_detector"]["detector_id"] = str(
                selected["detector_id"]
            )
            if "parameters" in selected:
                payload["signal"]["peak_detector"]["parameters"] = dict(
                    selected["parameters"]
                )
            else:
                payload["signal"]["peak_detector"].pop("parameters", None)
        else:
            is_line_b = selected["profile_id"] == "role_aware_equal_roles"
            payload["training"]["training_balance"] = str(
                selected["training_balance"]
            )
            payload["aggregation"].update(
                {
                    "balance_line": str(selected["balance_line"]),
                    "hierarchy": list(selected["hierarchy"]),
                    "window_to_file": "ordinary_mean",
                    "file_to_role": (
                        "ordinary_mean" if is_line_b else "not_applicable"
                    ),
                    "role_to_participant": (
                        "ordinary_mean" if is_line_b else "not_applicable"
                    ),
                    "missing_role_policy": (
                        "mean_available_roles" if is_line_b else "not_applicable"
                    ),
                    "quality_weighting": False,
                    "direct_all_window_participant_mean": False,
                }
            )

    payload["config_id"] = (
        base.config_id.removesuffix("_v2")
        + "__" + str(profile_id).replace("-", "_") + "_v2"
    )
    payload["output"]["formal_ablation_materialization"] = {
        "schema_version": "ppg_frailty.formal_ablation_materialization.v2",
        "family": family,
        "profile_id": str(profile_id),
        "catalog_role": str(selected["catalog_role"]),
        "base_config_path": base_relative,
        "base_config_sha256": base.sha256,
        "profile_catalog_sha256": catalog["catalog_sha256"],
        "single_factor_only": True,
        "automatic_execution": False,
        "scientific_execution_completed": False,
    }
    validated = validate_config_payload(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(validated, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return load_config(target)


def load_v2_decision_profile(path: str | Path) -> dict[str, Any]:
    """Load machine-auditable reference defaults and deferred evidence."""

    source = Path(path)
    data = _strict_mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "decision_profile")
    required = {
        "schema_version",
        "pipeline_generation",
        "profile_id",
        "authority",
        "confirmed_defaults",
        "comparison_profiles",
        "deferred_evidence",
    }
    _require_exact_keys(data, required, context="decision_profile")
    if data["schema_version"] != V2_DECISION_PROFILE_SCHEMA:
        raise ValueError("unsupported V2 decision profile schema")
    if data["pipeline_generation"] != "final_pipeline_v2":
        raise ValueError("decision profile is not bound to final_pipeline_v2")
    for key in (
        "authority",
        "confirmed_defaults",
        "comparison_profiles",
        "deferred_evidence",
    ):
        _strict_mapping(data[key], key)
    return data


_BASE_RUNTIME_MODULES = ("numpy", "scipy", "sklearn", "yaml", "pyarrow")


def required_runtime_modules(config: PipelineConfig) -> tuple[str, ...]:
    """Return import names needed for an ordinary run of this configuration."""

    from .module_registry import model_runtime_dependencies

    modules = list(_BASE_RUNTIME_MODULES)
    modules.extend(
        model_runtime_dependencies(str(config.section("model")["model_id"]))
    )
    return tuple(modules)


def dependency_availability_report(config: PipelineConfig) -> dict[str, Any]:
    """Report missing runtime imports without pinning versions or import origins."""

    import importlib.util

    modules = required_runtime_modules(config)
    rows = [
        {
            "module": module,
            "available": importlib.util.find_spec(module) is not None,
        }
        for module in modules
    ]
    missing = [row["module"] for row in rows if not row["available"]]
    return {
        "schema_version": "ppg_frailty.dependency_availability.v2",
        "pipeline_generation": "final_pipeline_v2",
        "config_id": config.config_id,
        "ready": not missing,
        "missing_modules": missing,
        "modules": rows,
        "policy": "ordinary_import_availability_no_version_or_origin_lock",
    }


def require_runtime_dependencies(config: PipelineConfig) -> dict[str, Any]:
    """Raise one actionable error when ordinary runtime imports are missing."""

    report = dependency_availability_report(config)
    if report["missing_modules"]:
        raise RuntimeError(
            "missing runtime dependencies: "
            + ", ".join(report["missing_modules"])
        )
    return report


__all__ = [
    "LEGACY_SCHEMA_VERSION",
    "PipelineConfig",
    "TOP_LEVEL_KEYS",
    "V2_DECISION_PROFILE_SCHEMA",
    "V2_FORMAL_CATALOG_SCHEMA",
    "V2_FORMAL_ABLATION_PROFILES_SCHEMA",
    "V2_SCHEMA_VERSION",
    "dependency_availability_report",
    "load_config",
    "load_formal_experiment_catalog",
    "materialize_formal_ablation_config",
    "load_formal_ablation_profiles",
    "load_v2_decision_profile",
    "require_runtime_dependencies",
    "required_runtime_modules",
    "validate_config_payload",
]
