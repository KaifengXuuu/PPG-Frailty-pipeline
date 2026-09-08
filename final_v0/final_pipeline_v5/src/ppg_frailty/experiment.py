"""Outer-fold training, OOF prediction, comparison and optional final refit."""
from __future__ import annotations
import gc
import gzip
import hashlib
import io
import json
import time
from dataclasses import asdict, dataclass, field as dataclass_field, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping
from .contracts import to_strict_json_value

@dataclass(frozen=True)
class ExperimentResult:
    """可严格序列化的实验结论 / Strictly serializable experiment outcome."""

    status: str
    scientific_scope: str
    config_id: str
    config_hash: str
    repeat_indices: tuple[int, ...]
    fold_indices: tuple[int, ...]
    output_dir: str | None
    cell_results: tuple[dict[str, Any], ...] = ()
    metrics: dict[str, Any] = dataclass_field(default_factory=dict)
    provenance: dict[str, Any] = dataclass_field(default_factory=dict)
    failure_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """转换为 strict JSON value / Convert to a strict-JSON value."""
        payload = asdict(self)
        payload["schema_version"] = "ppg_frailty.experiment_result.v2"
        payload["pipeline_generation"] = "final_pipeline_v2"
        return to_strict_json_value(payload)

class _RuntimeRecord:
    """单折内 recording 状态 / Per-record state inside one fold."""

    _NONE_FIELDS = (
        "views processed_views routing_timeline direct_quality final_quality route intended_route quality_tier reason "
        "direct_pulses_per_wavelength processed_pulses_per_wavelength vector engineering raw_windows "
        "transformed_vector transformed_engineering matrix fusion_features diagnostic_reason"
    ).split()
    _DICT_FIELDS = (
        "route_artifact route_window_sqi_evidence diagnostic_components physical_qc_evidence physical_qc_profile "
        "preprocessing_cache_refs"
    ).split()

    def __init__(self, row: Any, **values: Any) -> None:
        self.row, self.shape_features_eligible, self.retained = row, True, False
        self.route_status, self.artifact_name, self.artifact_version = "pending", "not_executed", "not_executed"
        self.__dict__.update({name: None for name in self._NONE_FIELDS})
        self.__dict__.update({name: {} for name in self._DICT_FIELDS})
        self.__dict__.update(values)

def _drop_after_routing(state: _RuntimeRecord, *, reason: str, route_status: str) -> None:
    """Keep final retention and persisted route evidence synchronized."""
    state.retained = False
    state.reason = reason
    state.route_status = route_status
    if state.route_artifact:
        state.route_artifact = {
            **state.route_artifact, "state": route_status, "abstained": True, "abstention_reason": reason,
        }

def _persisted_route_artifact_row(
    state: _RuntimeRecord, *, train_participant_ids: Iterable[str] = (), oof_participant_ids: Iterable[str] = ()
) -> dict[str, Any]:
    """Serialize final routing state even when no SQI diagnostics exist."""
    participant_id = str(state.row.participant_id)
    train = set(map(str, train_participant_ids))
    oof = set(map(str, oof_participant_ids))
    if participant_id in train and participant_id in oof:
        raise _ExperimentProtocolError("route_artifact_participant_in_train_and_oof")
    outer_partition = (
        "outer_oof" if participant_id in oof else "outer_train" if participant_id in train else "not_reported"
    )
    return {
        "record_id": state.row.record_id, "participant_id": state.row.participant_id,
        "outer_partition": outer_partition, "role": state.row.role, "retained": state.retained,
        "route_status": state.route_status, "signal_route": state.route.value if state.route is not None else None,
        "artifact_reducer_name": state.artifact_name, "artifact_reducer_version": state.artifact_version,
        "route_artifact": state.route_artifact,
    }

def _compact_sqi_result(value: Any) -> dict[str, Any] | None:
    """Keep scalar SQI evidence for legacy audit readers."""
    payload = value if isinstance(value, Mapping) else to_strict_json_value(value)
    if not isinstance(payload, Mapping):
        return None
    fields = ("state", "score", "coverage", "threshold", "reasons")
    output = {name: to_strict_json_value(payload[name]) for name in ("state", "reasons") if name in payload}
    for component in ("q_rate", "q_morph"):
        item = payload.get(component)
        if isinstance(item, Mapping):
            output[component] = {name: to_strict_json_value(item[name]) for name in fields if name in item}
    return output

def _compact_window_sqi_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(window): {stage: _compact_sqi_result(values.get(stage)) for stage in ("direct", "post_reduction")}
        for window, values in evidence.items()
        if isinstance(values, Mapping)
    }

class _ExperimentProtocolError(RuntimeError):
    """关闭失败异常 / Fail-closed protocol exception."""

_ESTIMATOR_NOT_APPLICABLE = "not_applicable_estimator_native"

@lru_cache(maxsize=None)
def _model_capability_contract(model_id: str) -> Mapping[str, Any]:
    """Resolve backend/ensemble behavior from the single model registry."""
    from .module_registry import model_factory_contract

    return model_factory_contract(str(model_id))

def _model_uses_estimator(model_id: str) -> bool:
    return _model_capability_contract(model_id)["execution_backend"] == "estimator"

def _epoch_override_for_backend(config: Any, requested_override: int | None) -> int | None:
    """Limit the Torch-only smoke epoch override to Torch backends."""
    if requested_override is None:
        return None
    model_id = str(config.section("model")["model_id"])
    return None if _model_uses_estimator(model_id) else int(requested_override)

def _model_is_ensemble(model_id: str) -> bool:
    return "member_seeds" in set(_model_capability_contract(model_id)["factory_fields"])

def _model_input_sampling_rate_hz(config: Any) -> float:
    """Return the actual model-input grid while preserving canonical 400-Hz views."""
    signal = config.section("signal")
    dl = signal["dl_resampling"]
    if not bool(dl["enabled"]) and dl.get("case_id") is None:
        return float(signal["internal_fs_hz"])
    if dl.get("case_id") is not None and config.representation_mode != "raw":
        raise _ExperimentProtocolError("named_fixed_kernel_dl_resampling_requires_raw_representation")
    if config.representation_mode not in {"raw", "fusion"}:
        raise _ExperimentProtocolError("generic_dl_resampling_requires_raw_or_fusion_representation")
    return float(dl["target_fs_hz"])

def _training_algorithm_provenance(
    model_id: str, training_section: Mapping[str, Any], model_section: Mapping[str, Any], *, fixed_epochs: int
) -> dict[str, Any]:
    """Describe only training controls that the selected implementation consumes."""
    class_weighting = str(training_section["class_weighting"])
    class_count_basis = str(training_section["class_count_basis"])
    class_weight_count_basis = class_count_basis if class_weighting != "none" else "not_applicable_uniform"
    weighting = {
        "strategy": class_weighting, "class_weight_beta": float(training_section.get("class_weight_beta", 0.999)),
        "count_basis": class_weight_count_basis,
    }
    if _model_uses_estimator(model_id):
        return {
            "loss": _ESTIMATOR_NOT_APPLICABLE, "class_weighting": weighting, "sampler": training_section["sampler"],
            "epoch_rule": {"rule": "not_applicable", "fixed_epochs": None}, "optimizer": _ESTIMATOR_NOT_APPLICABLE,
            "learning_rate": _ESTIMATOR_NOT_APPLICABLE, "weight_decay": _ESTIMATOR_NOT_APPLICABLE,
            "dropout": _ESTIMATOR_NOT_APPLICABLE, "label_smoothing": _ESTIMATOR_NOT_APPLICABLE,
            "gradient_clipping": {"enabled": False, "max_norm": None, "status": _ESTIMATOR_NOT_APPLICABLE},
        }
    gradient_clip_norm = training_section.get("gradient_clip_norm")
    return {
        "loss": {
            "strategy": training_section["loss"], "focal_gamma": float(training_section.get("focal_gamma", 2.0)),
            "balanced_softmax_count_basis": class_count_basis
            if training_section["loss"] == "balanced_softmax" else "not_applicable",
        }, "class_weighting": weighting, "sampler": training_section["sampler"], "epoch_rule": {
            "rule": training_section["epoch_rule"],
            "fixed_epochs": int(fixed_epochs) if training_section["epoch_rule"] == "fixed_epoch" else None,
        }, "optimizer": training_section["optimizer"], "learning_rate": float(training_section["learning_rate"]),
        "weight_decay": float(training_section["weight_decay"]),
        "dropout": float(model_section["dropout"]) if "dropout" in model_section else "not_applicable",
        "label_smoothing": float(training_section["label_smoothing"]),
        "gradient_clipping": {"enabled": gradient_clip_norm is not None, "max_norm": gradient_clip_norm},
    }

def _resolved_legacy_bridge_dropout_comparison(profile: Any, model_section: Mapping[str, Any]) -> dict[str, Any] | None:
    """Resolve the L7 dropout clause without inventing a parameter change."""
    if profile is None or profile.profile_id != "L7":
        return None
    architecture = model_section["architecture_parameters"]
    legacy_stage = (0.1, 0.15)
    legacy_head = 0.2
    current_stage = tuple((float(value) for value in architecture["stage_dropouts"]))
    current_head = float(architecture["classifier_dropout"])
    changed = current_stage != legacy_stage or abs(current_head - legacy_head) > 1e-12
    return {
        "legacy_resolved": {
            "cnn_dropout_input": -1, "encoder_stage_dropouts": list(legacy_stage),
            "classifier_head_dropout": legacy_head, "source": "frailty_3class_classifier.py:Cnn1DClassifier",
        }, "current_registered": {
            "encoder_stage_dropouts": list(current_stage), "classifier_head_dropout": current_head,
            "source": "catalog_model.architecture_parameters",
        }, "changed": changed,
        "interpretation": "resolved_values_changed" if changed else "no_change_resolved_values_identical",
    }

@dataclass(frozen=True)
class _CellResult:
    """单折摘要与 OOF / One cell summary and OOF tables."""

    summary: dict[str, Any]
    file_rows: tuple[Any, ...]
    subject_rows: tuple[Any, ...]
    window_rows: tuple[Any, ...] = ()
    role_rows: tuple[Any, ...] = ()
    member_rows: tuple[Any, ...] = ()
    checkpoint: Any = None

@dataclass(frozen=True)
class _LegacyBridgeExecution:
    """Hash-bound inputs for one isolated cumulative or centred-star cell."""

    profile: Any
    source_specification: str | None
    source_specification_sha256: str | None
    manifest_sha256: str
    split_sha256: str
    effective_config_hash: str
    protocol_design: str = "cumulative_chain_v1"
    profile_definition_sha256: str | None = None

@dataclass(frozen=True)
class _LegacyBridgePreparedFactory:
    """Expose actual bridge semantics while delegating model construction safely."""

    canonical_factory: Any
    provenance: Mapping[str, Any]

    def __call__(self) -> Any:
        return self.canonical_factory()

def _runtime_imports() -> dict[str, Any]:
    """延迟导入重型依赖 / Lazily import experiment dependencies."""
    from dataclasses import replace
    from importlib import import_module
    import numpy as np

    imports = {
        ".artifact": "run_artifact_route", ".contracts": "QualityState SignalRoute",
        ".data.schema": "canonicalize_role_family", ".data.windows": "WindowPlan",
        ".features.engineering": "extract_engineering_features", ".features.window_matrix": (
            "build_ordered_window_matrix build_route_eligible_rate_pulse extract_window_features "
            "fit_fold_window_feature_transform route_eligible_morphology_aggregates transform_window_features"
        ), ".features.registry": "build_feature_vector default_registry registry_for_groups summarize_engineering",
        ".features.vector_transform": "fit_fold_feature_vector_transform transform_feature_vector_batch",
        ".models": "ModelInputSpec normalize_model_id prepare_model_factory",
        ".pipeline": "PipelinePaths _load_record preflight_pipeline",
        ".module_registry": "materialize_model_architecture model_factory_contract resolve_peak_detector_config",
        ".peaks": "detect_pulses_per_wavelength select_reference_wavelength",
        ".provenance": "runtime_environment stable_payload_sha256", ".signal.morphology": "extract_morphology",
        ".signal.optical": "extract_dual_optical",
        ".signal.preprocess": "build_signal_views roll_pitch_ekf_config_from_resolved",
        ".signal.motion_imu": "fit_motion_imu_calibration", ".signal.prv": "PrvConfig compute_prv",
        ".signal.sqi": "SqiConfig SqiDiagnosticConfig evaluate_quality evaluate_quality_diagnostics "
        "fit_sqi_calibrator quality_component_scores",
        ".quality.routing": "QualityTier resolve_quality_mode route_module_switches_from_config route_quality_tier "
        "run_quality_mode",
        ".quality.motion_bundle_adapter": "infer_reused_motion_windows infer_reused_motion_recording "
        "load_reused_motion_detector motion_recording_from_signal_views resolve_reused_motion_detector_config",
        ".quality.routing_timeline": "RoutingEvidence build_routing_timeline build_routing_windows overlapping_cells "
        "resolve_routing_evidence",
        ".representations": "build_raw_windows fit_fold_imu_channel_transform transform_raw_windows_imu",
        ".training": "FeatureMatrixDataset FeatureVectorDataset FileBagDataset FrozenOuterSplit OofPredictionRow "
        "OofWriter RawWindowDataset SampleIdentity TrainingConfig UnifiedTrainer aggregate_hierarchy "
        "build_inner_grouped_split build_config_metrics_from_predictions_and_fold_summaries "
        "evaluate_predictions_with_abstentions measure_cpu_batch1_operational_metrics validate_expected_oof_roster",
        ".training.oof": "validate_role_level_oof",
    }
    api = {"np": np, "replace": replace}
    for module_name, names in imports.items():
        module = import_module(module_name, __package__)
        api.update({name: getattr(module, name) for name in names.split()})
    return api

def _choose_records(rows: Iterable[Any], participant_ids: Iterable[str], roles: Iterable[str],
                    cap: int | None) -> list[Any]:
    """逐 participant 选择最长文件 / Select longest files per participant."""
    participants = set((str(value) for value in participant_ids))
    role_set = set((str(value) for value in roles))
    output: list[Any] = []
    for participant in sorted(participants):
        candidates = sorted((
            row for row in rows if row.participant_id == participant and row.role in role_set and
            (row.qc_status in {"pass", "pass_with_warnings"})
        ), key=lambda row: (-float(row.duration_s), str(row.record_id)),
                            )
        output.extend(candidates if cap is None else candidates[:cap])
    if {row.participant_id for row in output} != participants:
        raise _ExperimentProtocolError("selected_records_do_not_cover_frozen_roster")
    return output

def _classifier_role_ids(config: Any) -> tuple[str, ...]:
    """Resolve concrete manifest roles from the configured classifier families."""
    from .training.aggregation import canonical_role_family

    families = set(config.section("training")["classifier_role_families"])
    roles = tuple((str(value) for value in config.to_dict()["roles"]))
    selected = tuple((role for role in roles if canonical_role_family(role) in families))
    if not selected or {canonical_role_family(role) for role in selected} != families:
        raise _ExperimentProtocolError("classifier_role_families_not_represented_by_concrete_roles")
    return selected

def _fit_imu_calibrations(
    states: list[_RuntimeRecord], config: Any, loader: Any, *, calibration_rows: Iterable[Any] | None = None,
    cache_session: Any = None,
) -> tuple[dict[str, Any], dict[str, str], str]:
    """Fit reusable participant-B calibration objects for canonical IMU profiles."""
    api = _runtime_imports()
    signal = config.section("signal")
    imu = signal["imu"]
    gravity_method = str(imu["gravity_method"])
    calibrated_methods = {
        "calibrated_roll_pitch_ekf", "profile_a_lowpass_0p3hz", "sensor_filter_only_no_gravity_removal",
    }
    calibrations: dict[str, Any] = {}
    calibration_errors: dict[str, str] = {}
    if gravity_method in calibrated_methods:
        ekf_config = api["roll_pitch_ekf_config_from_resolved"](imu)
        calibration_source_rows = tuple(
            calibration_rows if calibration_rows is not None else (state.row for state in states)
        )
        for participant_id in sorted({str(state.row.participant_id) for state in states}):
            candidates = sorted((
                row for row in calibration_source_rows if str(row.participant_id) == participant_id
                and str(row.role) == "B" and (str(row.qc_status) in {"pass", "pass_with_warnings"})
            ), key=lambda row: (-float(row.duration_s), str(row.record_id)),
                                )
            if not candidates:
                calibration_errors[participant_id] = "same_participant_role_B_calibration_record_missing"
                continue
            calibration_row = candidates[0]
            try:

                def build_calibration() -> Any:
                    loaded = loader(calibration_row, None)
                    return api["fit_motion_imu_calibration"](
                        api["np"].asarray(loaded["acc"], dtype=api["np"].float64),
                        api["np"].asarray(loaded["gyro"], dtype=api["np"].float64), participant_id=participant_id,
                        file_id=str(calibration_row.record_id), source_role="B", fs_hz=float(calibration_row.fs),
                        acceleration_unit=str(loaded["acc_unit"]), gyroscope_unit=str(loaded["gyro_unit"]),
                        config=ekf_config,
                    )

                calibrations[participant_id] = (
                    cache_session.calibration(calibration_row, imu, build_calibration)
                    if cache_session is not None and cache_session.enabled else build_calibration()
                )
            except Exception as exc:
                from .data.recording_cache import RecordingCacheError

                if isinstance(exc, RecordingCacheError):
                    raise
                calibration_errors[participant_id] = f"{type(exc).__name__}:{exc}"
    return (calibrations, calibration_errors, gravity_method)

def _preprocess_records(
    states: list[_RuntimeRecord], config: Any, maximum_seconds: float | None, loader: Any, *,
    calibration_rows: Iterable[Any] | None = None, cache_session: Any = None,
) -> None:
    """Build direct views with one explicit role-B IMU calibration per participant."""
    api = _runtime_imports()
    build_signal_views = api["build_signal_views"]
    calibrations, calibration_errors, gravity_method = _fit_imu_calibrations(
        states, config, loader, calibration_rows=calibration_rows, cache_session=cache_session
    )
    calibrated_methods = {
        "calibrated_roll_pitch_ekf", "profile_a_lowpass_0p3hz", "sensor_filter_only_no_gravity_removal",
    }
    for state in states:
        participant_id = str(state.row.participant_id)
        if participant_id in calibration_errors:
            state.reason = "imu_calibration_failed:" + calibration_errors[participant_id]
            state.route_status = "dropped_preprocess"
            continue
        maximum = (
            None if maximum_seconds is None else
            min(int(state.row.n_samples), int(round(maximum_seconds * float(state.row.fs))))
        )
        try:
            calibration = None
            if gravity_method in calibrated_methods:
                calibration = calibrations.get(participant_id)
                if calibration is None:
                    raise RuntimeError("same_participant_imu_calibration_unavailable")
                state.diagnostic_components["imu_calibration"] = {
                    "schema_version": calibration.schema_version, "participant_id": calibration.participant_id,
                    "file_id": calibration.file_id, "source_role": calibration.source_role,
                    "artifact_sha256": calibration.artifact_sha256, "gravity_method": gravity_method,
                    "fallback_used": False,
                }

            def build_views() -> tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
                loaded = dict(loader(state.row, maximum))
                loaded["participant_id"] = participant_id
                if calibration is not None:
                    loaded["imu_calibration"] = calibration
                qc = to_strict_json_value(
                    loaded.get(
                        "recording_qc",
                        {"status": "not_supplied_by_injected_test_loader", "record_id": state.row.record_id},
                    )
                )
                profile = to_strict_json_value(
                    loaded.get("recording_qc_profile", {"profile_id": "not_supplied_by_injected_test_loader"})
                )
                return (build_signal_views(loaded, config.to_dict()), qc, profile)

            if cache_session is not None and cache_session.enabled:
                quality = config.section("quality")
                (state.views, state.physical_qc_evidence, state.physical_qc_profile, view_key,
                 ) = cache_session.canonical_views(
                     state.row, maximum_samples=maximum, signal_config=config.section("signal"),
                     quality_preprocess_config={
                         "long_gap_max_samples": quality["long_gap_max_samples"],
                         "flatline_duration_s": quality["flatline_duration_s"],
                     }, calibration=calibration, builder=build_views,
                 )
                state.preprocessing_cache_refs["canonical_signal_views"] = view_key
            else:
                state.views, state.physical_qc_evidence, state.physical_qc_profile = build_views()
            state.route_status = "direct_preprocessed"
        except Exception as exc:
            from .data.recording_cache import RecordingCacheError

            if isinstance(exc, RecordingCacheError):
                raise
            state.reason = f"preprocess_failed:{type(exc).__name__}:{exc}"
            state.route_status = "dropped_preprocess"

def _preprocess_legacy_bridge_records(
    states: list[_RuntimeRecord], profile: Any, maximum_seconds: float | None, loader: Any, *,
    config: Any | None = None, calibration_rows: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Build field-driven bridge windows directly from freshly audited CSV bytes."""
    api = _runtime_imports()
    from .legacy_bridge import build_legacy_bridge_raw_windows

    calibrations: dict[str, Any] = {}
    calibration_errors: dict[str, str] = {}
    if profile.requires_calibrated_imu_views:
        if config is None:
            raise _ExperimentProtocolError("bridge_calibrated_imu_requires_resolved_config")
        calibrations, calibration_errors, gravity_method = _fit_imu_calibrations(
            states, config, loader, calibration_rows=calibration_rows
        )
        if gravity_method != "calibrated_roll_pitch_ekf":
            raise _ExperimentProtocolError("bridge_calibrated_imu_requires_roll_pitch_ekf_config")
    for state in states:
        participant_id = str(state.row.participant_id)
        if participant_id in calibration_errors:
            state.reason = "imu_calibration_failed:" + calibration_errors[participant_id]
            state.route_status = "dropped_legacy_bridge_preprocess"
            continue
        maximum = (
            None if maximum_seconds is None else
            min(int(state.row.n_samples), int(round(maximum_seconds * float(state.row.fs))))
        )
        try:
            loaded = dict(loader(state.row, maximum))
            calibrated_views = None
            if profile.requires_calibrated_imu_views:
                loaded["participant_id"] = participant_id
                calibration = calibrations.get(participant_id)
                if calibration is None:
                    raise RuntimeError("same_participant_imu_calibration_unavailable")
                loaded["imu_calibration"] = calibration
                calibrated_views = api["build_signal_views"](loaded, config.to_dict())
                state.views = calibrated_views
                state.diagnostic_components["imu_calibration"] = {
                    "schema_version": calibration.schema_version, "participant_id": calibration.participant_id,
                    "file_id": calibration.file_id, "source_role": calibration.source_role,
                    "artifact_sha256": calibration.artifact_sha256, "gravity_method": "calibrated_roll_pitch_ekf",
                    "fallback_used": False,
                }
            state.physical_qc_evidence = to_strict_json_value(
                loaded.get(
                    "recording_qc",
                    {"status": "not_supplied_by_injected_test_loader", "record_id": state.row.record_id}
                )
            )
            state.physical_qc_profile = to_strict_json_value(
                loaded.get("recording_qc_profile", {"profile_id": "not_supplied_by_injected_test_loader"})
            )
            state.raw_windows = build_legacy_bridge_raw_windows(loaded, profile, calibrated_views=calibrated_views)
            state.final_quality = None
            state.route = api["SignalRoute"].DIRECT
            state.intended_route = api["SignalRoute"].DIRECT
            state.retained = True
            state.route_status = "retained_legacy_bridge_fresh_raw_quality_off"
            state.artifact_name = "identity"
            state.artifact_version = "identity_v1"
            state.route_artifact = {
                "schema_version": "ppg_frailty.legacy_bridge_route.v1", "state": "direct_fresh_raw_csv",
                "quality_mode": "off", "classification_action": "keep_unchanged", "affects_retention": False,
                "affects_aggregation": False, "affects_prediction": False, "profile_id": profile.profile_id,
            }
        except Exception as exc:
            state.retained = False
            state.reason = f"legacy_bridge_preprocess_failed:{type(exc).__name__}:{exc}"
            state.route_status = "dropped_legacy_bridge_preprocess"
    return {
        "method": "legacy_bridge_quality_off_fresh_raw_csv"
        if profile.protocol_design == "cumulative_chain_v1" else "legacy_bridge_quality_off_fresh_raw_csv_field_driven",
        "fitted_on_participant_ids": tuple(sorted(calibrations)), "outer_oof_ids_absent": True,
        "classification_effect": "none", "historical_cache_used_for_training": False,
    }

def _extract_canonical_all_channel_bridge_raw(state: _RuntimeRecord, profile: Any) -> None:
    """Materialise canonical channels with declared legacy all-8 scaling."""
    if not state.retained:
        return
    from .legacy_bridge import build_v2_window_scaled_bridge_raw_windows

    try:
        state.raw_windows = build_v2_window_scaled_bridge_raw_windows(state.views, profile)
    except Exception as exc:
        state.retained = False
        if profile.protocol_design == "cumulative_chain_v1":
            state.reason = f"legacy_bridge_l3_windows_failed:{type(exc).__name__}:{exc}"
            state.route_status = "dropped_legacy_bridge_l3_window_failure"
        else:
            state.reason = f"legacy_bridge_canonical_windows_failed:{type(exc).__name__}:{exc}"
            state.route_status = "dropped_legacy_bridge_canonical_window_failure"

def _peak_detection_runtime_kwargs(detector: Mapping[str, Any]) -> dict[str, Any]:
    """Select the fully materialized detector arguments used at runtime."""
    resolved = {
        "detector_id": str(detector["detector_id"]), "min_observation_sec": float(detector["min_observation_sec"]),
        "min_peaks": int(detector["min_peaks"]),
    }
    if "parameters" in detector:
        resolved["detector_parameters"] = dict(detector["parameters"])
    return resolved

def _direct_pulses_for_state(state: _RuntimeRecord, api: Mapping[str, Any],
                             detector: Mapping[str, Any]) -> Mapping[str, Any]:
    """Detect a direct recording once and reuse the exact pulse evidence."""
    if state.direct_pulses_per_wavelength is None:
        state.direct_pulses_per_wavelength = api["detect_pulses_per_wavelength"](
            state.views, **_peak_detection_runtime_kwargs(detector)
        )
    return state.direct_pulses_per_wavelength

def _pulse_hr_audit(pulse: Any | None) -> dict[str, Any]:
    """Summarize detector PPIs for diagnostics, never as classifier input."""
    import numpy as np

    if pulse is None:
        return {"hr_bpm": None, "median_valid_ppi_s": None, "valid_ppi_count": 0, "peak_count": 0,
                "reference_wavelength": None, "estimator": "60_over_median_valid_ppi_s"}
    ppi = np.asarray(getattr(pulse, "ppi_s", ()), dtype=np.float64)
    mask = np.asarray(getattr(pulse, "valid_interval_mask", np.ones(ppi.shape, dtype=bool)), dtype=bool)
    if mask.shape != ppi.shape:
        raise _ExperimentProtocolError("pulse_hr_valid_interval_mask_misaligned")
    valid = ppi[mask & np.isfinite(ppi) & (ppi > 0.0)]
    median = float(np.median(valid)) if valid.size else None
    return {"hr_bpm": None if median is None else float(60.0 / median), "median_valid_ppi_s": median,
            "valid_ppi_count": int(valid.size), "peak_count": int(np.asarray(getattr(pulse, "peaks", ())).size),
            "reference_wavelength": str(pulse.wavelength) if getattr(pulse, "wavelength", None) is not None else None,
            "estimator": "60_over_median_valid_ppi_s"}

def _fit_quality_calibrator(
    states: list[_RuntimeRecord], config: Any, train_ids: tuple[str, ...], oof_ids: tuple[str, ...]
) -> tuple[Any, Any]:
    """Resolve fixed SQI or fit from participant-balanced train routing windows."""
    api = _runtime_imports()
    sqi_payload = config.to_dict()
    sqi_payload["quality"].pop("window_selection", None)
    formal = api["SqiConfig"].from_resolved(sqi_payload)
    if formal.calibrator == "fixed_formula_thresholds_v1":
        return (formal, None)
    detector = api["resolve_peak_detector_config"](config.section("signal"))
    base = api["replace"](formal, calibrator="fixed_formula_thresholds_v1")
    component_rows: list[dict[str, float]] = []
    participant_rows: list[str] = []
    routing = config.section("routing")
    training = set(map(str, train_ids))
    for state in states:
        participant = str(state.row.participant_id)
        if state.views is None or participant not in training:
            continue
        try:
            record_components: list[dict[str, float]] = []
            pulses = _direct_pulses_for_state(state, api, detector)
            pulse = pulses[api["select_reference_wavelength"](pulses)]
            windows = api["build_routing_windows"](
                str(state.row.record_id), int(state.views.x_filter.shape[0]), fs_hz=float(routing["fs_hz"]),
                window_s=float(routing["window_s"]), hop_s=float(routing["hop_s"]),
            )
            for window in windows:
                local_views = _slice_signal_views_for_routing(
                    state.views, window.start_sample_400, window.stop_sample_400
                )
                local_pulse = _slice_global_pulse_for_routing(pulse, window.start_sample_400, window.stop_sample_400)
                quality = api["evaluate_quality"](
                    local_views, config=base, pulse=local_pulse, **_peak_detection_runtime_kwargs(detector)
                )
                record_components.append(api["quality_component_scores"](quality))
            component_rows.extend(record_components)
            participant_rows.extend([participant] * len(record_components))
        except Exception as exc:
            state.views = None
            state.reason = f"direct_base_sqi_failed:{type(exc).__name__}:{exc}"
            state.route_status = "dropped_direct_base_sqi"
    fitted_ids = tuple(sorted({value for value in participant_rows if value in set(train_ids)}))
    if not fitted_ids:
        raise _ExperimentProtocolError("no_outer_train_direct_sqi_components")
    calibrator = api["fit_sqi_calibrator"](
        component_rows, participant_rows, fitted_on_participant_ids=fitted_ids, outer_train_participant_ids=train_ids,
        outer_oof_participant_ids=oof_ids, lower_quantile=formal.calibrator_lower_quantile,
        upper_quantile=formal.calibrator_upper_quantile,
    )
    if set(calibrator.fitted_on_participant_ids) & set(oof_ids):
        raise _ExperimentProtocolError("heldout_subject_in_sqi_calibrator")
    return (formal, calibrator)

def _quality_mode(config: Any) -> str:
    """Resolve the selected optional quality module without a readiness gate."""
    api = _runtime_imports()
    try:
        return api["resolve_quality_mode"](config.section("quality").get("mode", "off")).value
    except (TypeError, ValueError) as exc:
        raise _ExperimentProtocolError(f"unsupported_quality_mode:{exc}") from exc

def _retain_without_quality_routing(states: list[_RuntimeRecord], config: Any, *,
                                    diagnostics_only: bool) -> dict[str, Any]:
    """Retain hard-QC-valid records while SQI has no classification effect."""
    api = _runtime_imports()
    diagnostic_config = api["SqiDiagnosticConfig"].from_resolved(config.to_dict()) if diagnostics_only else None
    artifact = config.section("artifact")
    detector = api["resolve_peak_detector_config"](config.section("signal"))
    for state in states:
        if state.views is None:
            continue
        try:
            outcome = api["run_quality_mode"](
                state.views, mode="diagnostics_only" if diagnostics_only else "off",
                evaluator=api["evaluate_quality_diagnostics"],
                **{"config": diagnostic_config, **_peak_detection_runtime_kwargs(detector)}
                if diagnostic_config is not None else {},
            )
            state.direct_quality = outcome.result
            if outcome.result is not None:
                state.diagnostic_components = to_strict_json_value(asdict(outcome.result))
            state.route_artifact = {
                "schema_version": "reference_direct_quality_mode_v2", "segment_id": str(state.row.record_id),
                "start_sample": 0, "end_sample": int(state.views.x_filter.shape[0]),
                "state": "reference_direct_quality_nonrouting", "source_signal": "x_filter",
                "quality_mode": outcome.mode.value, "classification_action": outcome.classification_action,
                "affects_retention": outcome.affects_retention, "affects_aggregation": outcome.affects_aggregation,
                "affects_prediction": outcome.affects_prediction, "reasons": tuple(outcome.reasons),
            }
        except Exception as exc:
            if diagnostics_only:
                state.diagnostic_reason = f"diagnostics_only_failed:{type(exc).__name__}:{exc}"
                state.route_artifact = {
                    "schema_version": "reference_direct_quality_mode_v2", "segment_id": str(state.row.record_id),
                    "start_sample": 0, "end_sample": int(state.views.x_filter.shape[0]
                                                         ), "state": "reference_direct_diagnostics_unavailable",
                    "source_signal": "x_filter", "quality_mode": "diagnostics_only",
                    "classification_action": "keep_unchanged", "affects_retention": False, "affects_aggregation": False,
                    "affects_prediction": False, "reasons": (state.diagnostic_reason, ),
                }
            else:
                raise
        state.final_quality = None
        state.route = api["SignalRoute"].DIRECT
        state.intended_route = api["SignalRoute"].DIRECT
        state.retained = True
        state.route_status = "retained_direct_diagnostics_only" if diagnostics_only else "retained_direct_quality_off"
        state.artifact_name = "identity"
        state.artifact_version = "identity_v1"
    return {
        "method": "diagnostics_only_raw_components_no_weights_thresholds_or_fit"
        if diagnostics_only else "not_applicable_quality_off", "fitted_on_participant_ids": (),
        "outer_oof_ids_absent": True, "classification_effect": "none",
        "configured_reducer_not_executed": str(artifact["reducer"]), "runtime_parameters": asdict(diagnostic_config)
        if diagnostic_config is not None and is_dataclass(diagnostic_config) else
        vars(diagnostic_config) if diagnostic_config is not None and hasattr(diagnostic_config, "__dict__") else None,
    }

def _quality_route_provenance(sqi_config: Any, calibrator: Any | None, oof_ids: Iterable[str]) -> dict[str, Any]:
    """Describe the exact route policy and prove its fitted-state boundary."""
    fitted_ids = tuple(calibrator.fitted_on_participant_ids) if calibrator is not None else ()
    return {
        "method": calibrator.method if calibrator is not None else sqi_config.calibrator,
        "fitted_on_participant_ids": fitted_ids, "outer_oof_ids_absent": not bool(set(fitted_ids) & set(oof_ids)),
        "classification_effect": "routing", "runtime_parameters": sqi_config.to_dict(),
    }

def _apply_quality_motion_routing(
    states: list[_RuntimeRecord], config: Any, report: Any, paths: Any, *, train_ids: tuple[str, ...],
    oof_ids: tuple[str, ...], cache_session: Any = None,
) -> dict[str, Any]:
    """Execute one shared route orchestration for outer CV and final refit."""
    mode = _quality_mode(config)
    switches = _runtime_imports()["route_module_switches_from_config"](config.to_dict())
    if mode == "off" and (not switches.motion_detector_enabled) and (not switches.denoiser_enabled):
        provenance = _retain_without_quality_routing(states, config, diagnostics_only=False)
        provenance.update({
            "state_machine": "all_optional_route_modules_off_v1", "motion_detector": None, "denoiser": {
                "enabled": False, "configured_reducer": str(config.section("artifact")["reducer"]),
                "attempt_policy": "not_applicable", "successful_output_tier": "not_applicable",
            }, "acceptable_representation_contract": "pulse_derived_feature_vector_only",
            "diagnostics_only_provenance": None,
        })
        return to_strict_json_value(provenance)
    diagnostic_provenance = ({
        "method": "native_routing_windows_fixed_formula_diagnostics", "fitted_on_participant_ids": (),
        "outer_oof_ids_absent": True, "classification_effect": "none",
    } if mode == "diagnostics_only" else None)
    sqi_config = None
    calibrator = None
    if mode == "route":
        sqi_config, calibrator = _fit_quality_calibrator(states, config, train_ids, oof_ids)
    motion_detector = _load_reused_motion_detector_for_config(config, paths, train_ids=train_ids, oof_ids=oof_ids)
    _route_records_window_level(
        states, config, report, sqi_config, calibrator, motion_detector=motion_detector, cache_session=cache_session
    )
    if sqi_config is not None:
        provenance = _quality_route_provenance(sqi_config, calibrator, oof_ids)
    else:
        provenance = {
            "method": "not_applied_" + mode, "fitted_on_participant_ids": (), "outer_oof_ids_absent": True,
            "classification_effect": "sqi_disabled_tier_routing", "runtime_parameters": None,
        }
    artifact = config.section("artifact")
    provenance.update({
        "state_machine": "authoritative_sqi_motion_tiers_v1",
        "motion_detector": _compact_motion_provenance(motion_detector), "denoiser": {
            "enabled": bool(artifact["denoiser_enabled"]), "configured_reducer": str(artifact["reducer"]),
            "attempt_policy": "at_most_once_per_unfit_record", "successful_output_tier": "acceptable_rate_only",
        }, "acceptable_representation_contract": "pulse_derived_feature_vector_only",
        "diagnostics_only_provenance": diagnostic_provenance,
    })
    return to_strict_json_value(provenance)

def _reason_counts(rows: Iterable[Any]) -> dict[str, int]:
    """Count diagnostic reason codes without retaining one row per beat."""
    counts: dict[str, int] = {}
    for row in rows:
        reasons = row.get("reason_codes", ()) if isinstance(row, Mapping) else getattr(row, "reason_codes", ())
        for raw_reason in reasons:
            reason = str(raw_reason)
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))

def _compact_dual_optical_diagnostics(optical: Any) -> dict[str, Any]:
    """Persist bounded optical audit evidence, never full per-beat payloads."""
    policy = "per_beat_rows_omitted_from_default_artifacts"
    try:
        pairing = optical.pairing
        pairing_rows = pairing.rows
        beat_rows = optical.beat_audit
        valid_pair_count = sum((bool(row.pair_valid) for row in pairing_rows))
        optical_valid_count = sum((bool(row.optical_valid) for row in beat_rows))
        pairing_fields = (
            "schema_version detector_id reference_wavelength secondary_wavelength reference_score reference_coverage "
            "secondary_score secondary_coverage red_detection_run_id ir_detection_run_id red_detector_version "
            "ir_detector_version red_selected_polarity ir_selected_polarity red_block_hri_provenance_hash "
            "ir_block_hri_provenance_hash reference_selection_rule cycle_interval_policy ambiguity_tie_break"
        ).split()
        pairing_summary = {name: getattr(pairing, name) for name in pairing_fields}
        pairing_summary.update(
            row_count=len(pairing_rows), valid_pair_count=valid_pair_count,
            invalid_pair_count=len(pairing_rows) - valid_pair_count, reason_counts=_reason_counts(pairing_rows),
        )
        return to_strict_json_value({
            "schema_version": optical.schema_version, "detail_policy": policy, "status": "summary_available",
            "pairing_summary": pairing_summary, "beat_audit_summary": {
                "row_count": len(beat_rows), "optical_valid_count": optical_valid_count,
                "optical_invalid_count": len(beat_rows) - optical_valid_count,
                "reason_counts": _reason_counts(beat_rows),
            }, "aggregate_values": optical.aggregate_values, "aggregate_validity": optical.aggregate_validity,
            "diagnostics": optical.diagnostics, "reasons": tuple(getattr(optical, "reasons", ())),
        })
    except Exception as exc:
        return {
            "schema_version": str(getattr(optical, "schema_version", "dual_optical_unknown")), "detail_policy": policy,
            "status": "summary_unavailable_noncausal", "error_type": type(exc).__name__, "error": str(exc),
        }

def _load_reused_motion_detector_for_config(
    config: Any, paths: Any, *, train_ids: tuple[str, ...] = (), oof_ids: tuple[str, ...] = ()
) -> Any | None:
    """Load the configured immutable fold or all-29 motion detector."""
    api = _runtime_imports()
    artifact = config.section("artifact")
    if not bool(artifact["motion_detector_enabled"]):
        return None
    payload = dict(artifact["motion_detector"])
    declared_path = payload.get("evidence_path")
    if declared_path is None:
        raise _ExperimentProtocolError("enabled_motion_detector_requires_evidence_path")
    payload["evidence_path"] = paths.input_path(str(declared_path))
    payload["enabled"] = True
    try:
        detector_config = api["resolve_reused_motion_detector_config"](payload)
        return api["load_reused_motion_detector"](
            detector_config, outer_train_participant_ids=tuple(train_ids), outer_oof_participant_ids=tuple(oof_ids)
        )
    except Exception as exc:
        raise _ExperimentProtocolError(f"motion_detector_bundle_invalid:{type(exc).__name__}:{exc}") from exc

def _compact_motion_provenance(detector: Any | None) -> dict[str, Any]:
    if detector is None:
        return {"enabled": False, "training_scope": "not_applicable", "valid_outer_oof_claim": False}
    source = dict(detector.provenance)
    fields = (
        "execution training_scope reuse_scope frailty29_evaluation_relation valid_outer_oof_claim evidence_path "
        "evidence_sha256 model_artifact_sha256 ekf_config_sha256 frozen_bundle_threshold_sha256 threshold_source "
        "runtime_device window_probability_aggregation"
    ).split()
    return {"enabled": True, **{name: source[name] for name in fields}}

def _slice_signal_views_for_routing(views: Any, start: int, stop: int) -> Any:
    """Create a time-aligned view of one routing window without changing sources."""
    from dataclasses import replace
    import numpy as np

    total = int(views.x_filter.shape[0])
    if not 0 <= start < stop <= total:
        raise ValueError("routing slice lies outside canonical signal views")
    imu = {
        name: api_value[start:stop] if
        (api_value := np.asarray(value)).ndim >= 1 and api_value.shape[0] == total else api_value
        for name, value in views.imu_processed.items()
    }
    metadata = dict(views.metadata)
    metadata.update({"routing_window_start_sample_400": int(start), "routing_window_stop_sample_400": int(stop)})
    if "artifact_output_valid_mask" in metadata:
        metadata["artifact_output_valid_mask"] = np.asarray(metadata["artifact_output_valid_mask"],
                                                            dtype=bool)[start:stop]
    return replace(
        views, x_native=views.x_native[start:stop], x_filter=views.x_filter[start:stop],
        x_analysis_rate=views.x_analysis_rate[start:stop], imu_processed=imu, metadata=metadata,
        source_valid_mask=views.source_valid_mask[start:stop], repair_mask=views.repair_mask[start:stop],
        x_ar=None if views.x_ar is None else views.x_ar[start:stop],
    )

def _slice_global_pulse_for_routing(pulse: Any, start: int, stop: int, *, fs_hz: float = 400.0) -> Any:
    """Restrict one global detection run while retaining global peak ordinals."""
    from dataclasses import replace
    import numpy as np

    timestamps = np.asarray(pulse.peak_timestamps_s, dtype=np.float64)
    samples = np.rint(timestamps * fs_hz).astype(np.int64)
    peak_keep = (samples >= start) & (samples < stop)
    peak_indices = np.flatnonzero(peak_keep)
    index_map = {int(old): new for new, old in enumerate(peak_indices)}
    interval_starts = np.asarray(pulse.interval_start_peak_indices, dtype=np.int64)
    interval_stops = np.asarray(pulse.interval_stop_peak_indices, dtype=np.int64)
    interval_midpoints = 0.5 * (timestamps[interval_starts] + timestamps[interval_stops]) * fs_hz
    interval_keep = np.asarray([
        int(left) in index_map and int(right) in index_map and (start <= midpoint < stop)
        for left, right, midpoint in zip(interval_starts, interval_stops, interval_midpoints)
    ], dtype=bool,
                               )
    interval_indices = np.flatnonzero(interval_keep)
    original_ordinals = getattr(pulse, "peak_ordinals", None)
    if original_ordinals is None:
        original_ordinals = np.arange(samples.size, dtype=np.int64)
    rejection_reasons = tuple(getattr(pulse, "interval_rejection_reasons", ()))
    return replace(
        pulse, peaks=samples[peak_indices] - int(start), peak_timestamps_s=timestamps[peak_indices],
        accepted_peak_mask=np.asarray(pulse.accepted_peak_mask, dtype=bool)[peak_indices],
        interval_start_peak_indices=np.asarray([index_map[int(interval_starts[index])] for index in interval_indices],
                                               dtype=np.int64),
        interval_stop_peak_indices=np.asarray([index_map[int(interval_stops[index])] for index in interval_indices],
                                              dtype=np.int64), ppi_s=np.asarray(pulse.ppi_s,
                                                                                dtype=np.float64)[interval_indices],
        valid_interval_mask=np.asarray(pulse.valid_interval_mask, dtype=bool)[interval_indices],
        adjacency_mask=np.asarray(pulse.adjacency_mask,
                                  dtype=bool)[interval_indices], confidence=np.asarray(pulse.confidence)[peak_indices],
        interval_run_ids=np.asarray(pulse.interval_run_ids)[interval_indices], interval_rejection_reasons=tuple(
            (rejection_reasons[index] for index in interval_indices)
        ) if len(rejection_reasons) == len(interval_starts) else (),
        peak_ordinals=np.asarray(original_ordinals, dtype=np.int64)[peak_indices],
    )

def _route_records_window_level(
    states: list[_RuntimeRecord], config: Any, report: Any, sqi_config: Any | None, calibrator: Any | None, *,
    motion_detector: Any | None = None, cache_session: Any = None,
) -> None:
    """Run shared native-window evidence and build one RoutingTimeline per file."""
    api = _runtime_imports()
    artifact = config.section("artifact")
    mode = _quality_mode(config)
    switches = api["route_module_switches_from_config"](config.to_dict())
    detector_config = api["resolve_peak_detector_config"](config.section("signal"))
    routing = config.section("routing")
    representation_mode = str(config.representation_mode)
    allow_rate_feature_recovery_without_direct_sqi = bool(
        representation_mode == "feature_vector"
        and str(artifact["degraded_policy"]) == "denoise_then_extract_rate_features"
    )
    if switches.motion_detector_enabled != (motion_detector is not None):
        raise _ExperimentProtocolError("motion_detector_switch_and_loaded_runtime_disagree")
    if mode == "diagnostics_only":
        from .signal.sqi import SqiDiagnosticConfig

        diagnostic_config = SqiDiagnosticConfig.from_resolved(config.to_dict())
    else:
        diagnostic_config = None
    nonrouting_sqi_config = None
    if mode != "route":
        from .signal.sqi import SqiConfig

        fixed_payload = config.to_dict()
        fixed_payload["quality"].pop("window_selection", None)
        nonrouting_sqi_config = SqiConfig.from_resolved(fixed_payload)
        if nonrouting_sqi_config.calibrator != "fixed_formula_thresholds_v1":
            nonrouting_sqi_config = api["replace"](nonrouting_sqi_config, calibrator="fixed_formula_thresholds_v1")
    for state in states:
        state.retained = False
        state.route = api["SignalRoute"].DROPPED
        state.intended_route = api["SignalRoute"].DIRECT
        state.shape_features_eligible = False
        if state.views is None:
            continue
        record_id = str(state.row.record_id)
        participant_id = str(state.row.participant_id)
        role = str(state.row.role)
        native_windows = api["build_routing_windows"](
            record_id, int(state.views.x_filter.shape[0]), fs_hz=float(routing["fs_hz"]),
            window_s=float(routing["window_s"]), hop_s=float(routing["hop_s"]),
        )
        direct_pulse = None
        if native_windows and (mode != "off" or switches.denoiser_enabled):
            pulses = _direct_pulses_for_state(state, api, detector_config)
            direct_pulse = pulses[api["select_reference_wavelength"](pulses)]
        direct_hr, post_hr = _pulse_hr_audit(direct_pulse), _pulse_hr_audit(None)
        motion_series = None
        motion_by_id: dict[str, Any] = {}
        if motion_detector is not None and native_windows:
            try:
                recording = api["motion_recording_from_signal_views"](
                    state.views, detector=motion_detector, record_id=record_id, participant_id=participant_id, role=role
                )
            except ValueError as exc:
                if "gap-repaired native PPG" not in str(exc):
                    raise
                recording = None
            precomputed_motion_windows = None
            if recording is not None and cache_session is not None and cache_session.enabled:
                from .representations.motion import MOTION_REFERENCE_PROFILE_ID, build_motion_window_tensors

                upstream_key = state.preprocessing_cache_refs.get("canonical_signal_views")
                if upstream_key is None:
                    raise RuntimeError("motion_window_cache_requires_upstream_view_key")
                precomputed_motion_windows, motion_window_key = cache_session.motion_windows(
                    state.row, upstream_views_key=upstream_key, recording=recording,
                    profile_id=MOTION_REFERENCE_PROFILE_ID, builder=lambda: build_motion_window_tensors(
                        recording.ppg_red_ir, recording.motion_imu, record_id=recording.record_id, participant_id=
                        recording.participant_id, role_or_activity=recording.role_or_activity, dataset_id=recording.
                        dataset_id, fs_hz=recording.fs_hz, profile_id=MOTION_REFERENCE_PROFILE_ID,
                    ),
                )
                state.preprocessing_cache_refs["motion_windows"] = motion_window_key
            motion_series = api["infer_reused_motion_windows"](
                motion_detector, recording, precomputed_windows=precomputed_motion_windows
            )
            motion_by_id = {row.routing_window_id: row for row in motion_series.decisions}
        evidence_rows: list[Any] = []
        for window in native_windows:
            q_rate_score = q_rate_state = q_morph_score = q_morph_state = None
            if mode != "off":
                assert direct_pulse is not None
                local_views = _slice_signal_views_for_routing(
                    state.views, window.start_sample_400, window.stop_sample_400
                )
                local_pulse = _slice_global_pulse_for_routing(
                    direct_pulse, window.start_sample_400, window.stop_sample_400
                )
                if mode == "route":
                    quality = api["evaluate_quality"](
                        local_views, config=sqi_config, calibrator=calibrator, pulse=local_pulse,
                        **_peak_detection_runtime_kwargs(detector_config),
                    )
                    q_rate_score = quality.q_rate.score
                    q_rate_state = quality.q_rate.state.value
                    q_morph_score = quality.q_morph.score
                    q_morph_state = quality.q_morph.state.value
                else:
                    quality = api["evaluate_quality"](
                        local_views, config=nonrouting_sqi_config, pulse=local_pulse,
                        **_peak_detection_runtime_kwargs(detector_config),
                    )
                    q_rate_score = quality.q_rate.score
                    q_rate_state = quality.q_rate.state.value
                    q_morph_score = quality.q_morph.score
                    q_morph_state = quality.q_morph.state.value
                    diagnostics = api["evaluate_quality_diagnostics"](
                        local_views, config=diagnostic_config, pulse=local_pulse,
                        **_peak_detection_runtime_kwargs(detector_config),
                    )
                    state.diagnostic_components.setdefault("routing_window_sqi_diagnostics",
                                                           {})[window.routing_window_id
                                                               ] = to_strict_json_value(diagnostics)
            motion_row = motion_by_id.get(window.routing_window_id)
            motion_state = (
                "off" if motion_detector is None else "unavailable" if motion_row is None else motion_row.motion_state
            )
            unresolved = api["RoutingEvidence"](
                window=window, sqi_mode=mode, sqi_assessed=mode != "off", direct_q_rate_score=q_rate_score,
                direct_q_rate_state=q_rate_state, direct_q_morph_score=q_morph_score,
                direct_q_morph_state=q_morph_state, motion_detector_enabled=motion_detector is not None,
                motion_probability=None if motion_row is None else motion_row.probability,
                motion_threshold=None if motion_row is None else motion_row.threshold, motion_state=motion_state,
                denoiser_enabled=switches.denoiser_enabled,
            )
            evidence_rows.append(
                api["resolve_routing_evidence"](
                    unresolved, role=role,
                    allow_rate_feature_recovery_without_direct_sqi=allow_rate_feature_recovery_without_direct_sqi,
                )
            )
        requested = [row for row in evidence_rows if row.denoiser_requested]
        if requested:
            outcome = api["run_artifact_route"](
                state.views, report.artifact["runtime_reducer"], parameters=report.artifact["parameters"]
            )
            state.artifact_name = outcome.result.reducer_id
            state.artifact_version = outcome.result.reducer_version
            if outcome.views is not None and outcome.route is api["SignalRoute"].ARTIFACT_RATE_ONLY:
                state.processed_views = outcome.views
                state.processed_pulses_per_wavelength = api["detect_pulses_per_wavelength"](
                    outcome.views, **_peak_detection_runtime_kwargs(detector_config)
                )
                processed_pulse = state.processed_pulses_per_wavelength[api["select_reference_wavelength"](
                    state.processed_pulses_per_wavelength
                )]
                post_hr = _pulse_hr_audit(processed_pulse)
                updated_rows = []
                for row in evidence_rows:
                    if not row.denoiser_requested:
                        updated_rows.append(row)
                        continue
                    local_views = _slice_signal_views_for_routing(
                        outcome.views, row.window.start_sample_400, row.window.stop_sample_400
                    )
                    local_pulse = _slice_global_pulse_for_routing(
                        processed_pulse, row.window.start_sample_400, row.window.stop_sample_400
                    )
                    post = api["evaluate_quality"](
                        local_views, config=sqi_config if mode == "route" else nonrouting_sqi_config,
                        calibrator=calibrator, pulse=local_pulse, **_peak_detection_runtime_kwargs(detector_config),
                    )
                    if post.q_morph.state.value != "not_applicable":
                        raise _ExperimentProtocolError("processed_window_q_morph_must_be_not_applicable")
                    updated_rows.append(
                        api["resolve_routing_evidence"](
                            api["replace"](
                                row, denoiser_status="success", post_q_rate_score=post.q_rate.score,
                                post_q_rate_state=post.q_rate.state.value,
                            ), role=role, allow_rate_feature_recovery_without_direct_sqi=(
                                allow_rate_feature_recovery_without_direct_sqi
                            ),
                        )
                    )
                evidence_rows = updated_rows
            else:
                evidence_rows = [
                    api["resolve_routing_evidence"](
                        api["replace"](row, denoiser_status=outcome.result.status), role=role,
                        allow_rate_feature_recovery_without_direct_sqi=allow_rate_feature_recovery_without_direct_sqi,
                    ) if row.denoiser_requested else row for row in evidence_rows
                ]
        else:
            state.artifact_name = "identity"
            state.artifact_version = "identity_v1"
        motion_model_sha = None if motion_detector is None else str(motion_detector.provenance["model_artifact_sha256"])
        motion_schema_sha = (
            None if motion_detector is None else str(motion_detector.provenance["model_input_schema_sha256"])
        )
        sqi_hash = (
            None if mode == "off" else api["stable_payload_sha256"]({
                "config": None if sqi_config is None else sqi_config.to_dict(),
                "calibrator": to_strict_json_value(calibrator),
            })
        )
        state.routing_timeline = api["build_routing_timeline"](
            record_id=record_id, participant_id=participant_id, role=role, n_samples=int(state.views.x_filter.shape[0]),
            evidence=evidence_rows, config_sha256=str(config.sha256), sqi_calibrator_sha256=sqi_hash,
            motion_model_sha256=motion_model_sha, motion_input_schema_sha256=motion_schema_sha,
            reducer_sha256=None if not requested else api["stable_payload_sha256"]({
                "reducer": state.artifact_name, "version": state.artifact_version,
                "parameters": report.artifact["parameters"],
            }),
        )
        cells = state.routing_timeline.cells
        excellent = [cell for cell in cells if cell.final_tier == "excellent" and cell.source_route == "direct"]
        acceptable = [cell for cell in cells if cell.final_tier == "acceptable"]
        eligible = excellent if representation_mode in {"raw", "fusion"} else [*excellent, *acceptable]
        state.retained = bool(eligible)
        state.shape_features_eligible = bool(excellent)
        state.quality_tier = (
            "mixed"
            if excellent and acceptable else "excellent" if excellent else "acceptable" if acceptable else "excluded"
        )
        state.route = (
            api["SignalRoute"].DIRECT if excellent or any((cell.source_route == "direct" for cell in acceptable)) else
            api["SignalRoute"].ARTIFACT_RATE_ONLY if any((cell.source_route == "processed"
                                                          for cell in acceptable)) else api["SignalRoute"].DROPPED
        )
        state.route_status = (
            "retained_window_routing_timeline" if state.retained else "dropped_no_complete_8_second_routing_window"
            if not native_windows else "dropped_no_representation_eligible_routing_cell"
        )
        state.reason = None if state.retained else state.route_status
        state.route_artifact = to_strict_json_value({
            "schema_version": state.routing_timeline.schema_version, "state": state.route_status, "cells": cells,
            "motion_probability": None if motion_series is None else motion_series.file_median_probability_diagnostic,
            "denoiser_invoked": bool(requested), "denoiser_invocation_count": int(bool(requested)),
            "canonical_hybrid_waveform_created": False,
            "heart_rate_estimator": direct_hr["estimator"], "direct_hr_bpm": direct_hr["hr_bpm"],
            "direct_median_valid_ppi_s": direct_hr["median_valid_ppi_s"],
            "direct_valid_ppi_count": direct_hr["valid_ppi_count"], "direct_peak_count": direct_hr["peak_count"],
            "post_denoise_hr_bpm": post_hr["hr_bpm"],
            "post_denoise_median_valid_ppi_s": post_hr["median_valid_ppi_s"],
            "post_denoise_valid_ppi_count": post_hr["valid_ppi_count"],
            "post_denoise_peak_count": post_hr["peak_count"], "post_minus_direct_hr_bpm": None
            if direct_hr["hr_bpm"] is None or post_hr["hr_bpm"] is None
            else float(post_hr["hr_bpm"] - direct_hr["hr_bpm"]),
        })

def _route_records(
    states: list[_RuntimeRecord], config: Any, report: Any, sqi_config: Any | None, calibrator: Any | None, *,
    motion_detector: Any | None = None,
) -> None:
    """Apply the authoritative SQI/motion tiers and one optional recovery."""
    api = _runtime_imports()
    QualityState = api["QualityState"]
    QualityTier = api["QualityTier"]
    SignalRoute = api["SignalRoute"]
    artifact = config.section("artifact")
    denoiser_policy = str(artifact["degraded_policy"])
    switches = api["route_module_switches_from_config"](config.to_dict())
    detector = api["resolve_peak_detector_config"](config.section("signal"))
    representation_mode = str(getattr(config, "representation_mode", "feature_vector"))
    if switches.motion_detector_enabled != (motion_detector is not None):
        raise _ExperimentProtocolError("motion_detector_switch_and_loaded_runtime_disagree")
    if switches.denoiser_enabled and sqi_config is None:
        sqi_payload = config.to_dict()
        sqi_payload["quality"].pop("window_selection", None)
        sqi_config = api["SqiConfig"].from_resolved(sqi_payload)
        if sqi_config.calibrator != "fixed_formula_thresholds_v1":
            raise _ExperimentProtocolError("sqi_off_recovery_requires_fixed_formula_thresholds")
    for state in states:
        state.intended_route = SignalRoute.DIRECT
        state.retained = False
        state.shape_features_eligible = False
        if state.views is None:
            continue
        try:
            direct = None
            direct_pulse = None
            if switches.sqi_enabled or switches.denoiser_enabled:
                pulses = _direct_pulses_for_state(state, api, detector)
                direct_pulse = pulses[api["select_reference_wavelength"](pulses)]
            if switches.sqi_enabled:
                direct_outcome = api["run_quality_mode"](
                    state.views, mode="route", evaluator=api["evaluate_quality"], config=sqi_config,
                    calibrator=calibrator, pulse=direct_pulse, **_peak_detection_runtime_kwargs(detector),
                )
                direct = direct_outcome.result
                state.direct_quality = direct
            direct_hr = _pulse_hr_audit(direct_pulse)
            motion_decision = None
            motion_high: bool | None = None
            if motion_detector is not None:
                try:
                    recording = api["motion_recording_from_signal_views"](
                        state.views, detector=motion_detector, record_id=str(state.row.record_id),
                        participant_id=str(state.row.participant_id), role=str(state.row.role),
                    )
                except ValueError as exc:
                    if "gap-repaired native PPG" not in str(exc):
                        raise _ExperimentProtocolError("motion_recording_contract_failed:" + str(exc)) from exc
                    recording = None
                try:
                    motion_decision = api["infer_reused_motion_recording"](motion_detector, recording)
                except Exception as exc:
                    raise _ExperimentProtocolError(f"motion_inference_failed:{type(exc).__name__}:{exc}") from exc
                if motion_decision.motion_state == "low_motion":
                    motion_high = False
                elif motion_decision.motion_state == "high_motion":
                    motion_high = True
            tier = api["route_quality_tier"](
                sqi_enabled=switches.sqi_enabled, q_rate_state=None if direct is None else direct.q_rate.state,
                q_morph_state=None if direct is None else direct.q_morph.state,
                motion_enabled=switches.motion_detector_enabled, motion_high=motion_high,
            )
            state.quality_tier = tier.tier.value
            motion_state = "off" if motion_decision is None else motion_decision.motion_state
            route_artifact = {
                "schema_version": "ppg_frailty.sqi_motion_route.v2", "segment_id": str(state.row.record_id),
                "quality_mode": _quality_mode(config), "quality_tier": tier.tier.value, "tier_reasons": tuple(
                    tier.reasons
                ), "direct_q_rate_state": None if direct is None else direct.q_rate.state.value,
                "direct_q_rate_score": None if direct is None else direct.q_rate.score,
                "direct_q_morph_state": None if direct is None else direct.q_morph.state.value,
                "direct_q_morph_score": None if direct is None else direct.q_morph.score, "motion_state": motion_state,
                "motion_record_probability": None if motion_decision is None else motion_decision.record_probability,
                "motion_threshold": None if motion_decision is None else motion_decision.threshold, "abstained": False,
                "abstention_reason": None, "denoiser_attempted": False, "denoiser_id": str(artifact["reducer"]),
                "denoiser_status": "not_attempted", "denoiser_recovery_policy": denoiser_policy,
                "heart_rate_estimator": direct_hr["estimator"], "direct_hr_bpm": direct_hr["hr_bpm"],
                "direct_median_valid_ppi_s": direct_hr["median_valid_ppi_s"],
                "direct_valid_ppi_count": direct_hr["valid_ppi_count"], "direct_peak_count": direct_hr["peak_count"],
                "direct_reference_wavelength": direct_hr["reference_wavelength"], "post_denoise_hr_bpm": None,
                "post_denoise_median_valid_ppi_s": None, "post_denoise_valid_ppi_count": 0,
                "post_denoise_peak_count": 0, "post_denoise_reference_wavelength": None,
                "post_minus_direct_hr_bpm": None, "absolute_post_minus_direct_hr_bpm": None,
            }
            if tier.tier in {QualityTier.EXCELLENT, QualityTier.ACCEPTABLE}:
                excellent = tier.tier is QualityTier.EXCELLENT
                state.final_quality = direct
                state.route = SignalRoute.DIRECT
                state.shape_features_eligible = excellent
                state.artifact_name = "identity"
                state.artifact_version = "identity_v1"
                state.route_status = "full_direct" if excellent else "rate_only_direct"
                route_artifact["state"] = state.route_status
                route_artifact["configured_reducer_not_executed"] = str(artifact["reducer"])
                if excellent or representation_mode == "feature_vector":
                    state.retained = True
                else:
                    state.reason = "acceptable_tier_requires_feature_vector_representation"
                    state.route_status = "abstained_acceptable_unsupported_representation"
                    route_artifact["abstained"] = True
                    route_artifact["abstention_reason"] = state.reason
                    route_artifact["state"] = state.route_status
                state.route_artifact = to_strict_json_value(route_artifact)
                continue
            motion_unavailable = motion_decision is not None and motion_decision.motion_state == "unfit"
            if motion_unavailable or not switches.denoiser_enabled:
                state.route = SignalRoute.DROPPED
                state.reason = motion_decision.reason if motion_unavailable else ";".join(tier.reasons)
                state.route_status = "degraded_drop"
                route_artifact["state"] = state.route_status
                route_artifact["abstained"] = True
                route_artifact["abstention_reason"] = state.reason
                state.route_artifact = to_strict_json_value(route_artifact)
                continue
            state.intended_route = SignalRoute.ARTIFACT_RATE_ONLY
            route_artifact["denoiser_attempted"] = True
            outcome = api["run_artifact_route"](
                state.views, report.artifact["runtime_reducer"], parameters=report.artifact["parameters"]
            )
            state.artifact_name = outcome.result.reducer_id
            state.artifact_version = outcome.result.reducer_version
            route_artifact["denoiser_id"] = state.artifact_name
            route_artifact["denoiser_status"] = outcome.result.status
            if outcome.views is None or outcome.route is not SignalRoute.ARTIFACT_RATE_ONLY:
                state.route = SignalRoute.DROPPED
                state.reason = ";".join(tuple(tier.reasons) + tuple(outcome.result.reasons))
                state.route_status = "rejected_after_reduction"
                route_artifact["state"] = state.route_status
                route_artifact["abstained"] = True
                route_artifact["abstention_reason"] = state.reason
                state.route_artifact = to_strict_json_value(route_artifact)
                continue
            post_pulses = api["detect_pulses_per_wavelength"](outcome.views, **_peak_detection_runtime_kwargs(detector))
            post_pulse = post_pulses[api["select_reference_wavelength"](post_pulses)]
            post_outcome = api["run_quality_mode"](
                outcome.views, mode="route", evaluator=api["evaluate_quality"], config=sqi_config,
                calibrator=calibrator, pulse=post_pulse, **_peak_detection_runtime_kwargs(detector),
            )
            post = post_outcome.result
            if post.q_morph.state is not QualityState.NOT_APPLICABLE or post.q_morph.score is not None:
                raise _ExperimentProtocolError("nonidentity_post_q_morph_contract_failed")
            route_artifact["post_q_rate_state"] = post.q_rate.state.value
            route_artifact["post_q_rate_score"] = post.q_rate.score
            post_hr = _pulse_hr_audit(post_pulse)
            for key in ("hr_bpm", "median_valid_ppi_s", "valid_ppi_count", "peak_count", "reference_wavelength"):
                route_artifact[f"post_denoise_{key}"] = post_hr[key]
            if direct_hr["hr_bpm"] is not None and post_hr["hr_bpm"] is not None:
                route_artifact["post_minus_direct_hr_bpm"] = float(post_hr["hr_bpm"] - direct_hr["hr_bpm"])
                route_artifact["absolute_post_minus_direct_hr_bpm"] = abs(route_artifact["post_minus_direct_hr_bpm"])
            state.views = outcome.views
            state.final_quality = post
            state.route = SignalRoute.ARTIFACT_RATE_ONLY
            state.shape_features_eligible = False
            if post.q_rate.state is QualityState.PASS:
                state.quality_tier = QualityTier.ACCEPTABLE.value
                route_artifact["quality_tier"] = QualityTier.ACCEPTABLE.value
                if denoiser_policy == "denoise_then_extract_rate_features":
                    if representation_mode != "feature_vector":
                        raise _ExperimentProtocolError("rate_feature_recovery_policy_requires_feature_vector")
                    state.route_status = "rate_only_processed"
                    state.retained = True
                elif denoiser_policy == "denoise_then_compare_rate_exclude":
                    if representation_mode == "feature_vector":
                        raise _ExperimentProtocolError("rate_diagnostic_exclusion_policy_forbids_feature_vector")
                    state.route_status = "rate_only_diagnostic_excluded"
                    state.retained = False
                    state.reason = "post_denoise_rate_only_diagnostic_not_classifier_input"
                    route_artifact["abstained"] = True
                    route_artifact["abstention_reason"] = state.reason
                else:
                    raise _ExperimentProtocolError("unknown_denoiser_recovery_policy")
                route_artifact["state"] = state.route_status
            else:
                state.route = SignalRoute.DROPPED
                state.quality_tier = QualityTier.EXCLUDED.value
                state.reason = "post_denoise_q_rate_not_pass"
                state.route_status = "rejected_after_reduction"
                route_artifact["quality_tier"] = QualityTier.EXCLUDED.value
                route_artifact["state"] = state.route_status
                route_artifact["abstained"] = True
                route_artifact["abstention_reason"] = state.reason
            state.route_artifact = to_strict_json_value(route_artifact)
        except _ExperimentProtocolError:
            raise
        except Exception as exc:
            raise _ExperimentProtocolError(
                f"quality_route_execution_failed:{state.row.record_id}:{type(exc).__name__}:{exc}"
            ) from exc

def _extract_vector(state: _RuntimeRecord, report: Any, features_config: Mapping[str, Any] | None = None) -> None:
    """构建完整 FeatureVectorV1 / Build one complete FeatureVectorV1."""
    if not state.retained:
        return
    api = _runtime_imports()
    SignalRoute = api["SignalRoute"]
    QualityState = api["QualityState"]
    try:
        prv_config_type = api.get("PrvConfig")
        if prv_config_type is None:
            from .signal.prv import PrvConfig as prv_config_type
        prv_config = prv_config_type.from_mapping(features_config)
        if state.routing_timeline is not None:
            pulses_per_wavelength = _direct_pulses_for_state(state, api, report.peak_detector)
            direct_pulse = pulses_per_wavelength[api["select_reference_wavelength"](pulses_per_wavelength)]
            processed_pulse = None
            if state.processed_pulses_per_wavelength is not None:
                processed_pulse = state.processed_pulses_per_wavelength[api["select_reference_wavelength"](
                    state.processed_pulses_per_wavelength
                )]
            pulse = api["build_route_eligible_rate_pulse"](state.routing_timeline, direct_pulse, processed_pulse)
            prv_route = pulse.source_route
            q_rate_qualified = True
        else:
            pulses_per_wavelength = (
                state.direct_pulses_per_wavelength if state.route in {SignalRoute.DIRECT, SignalRoute.IDENTITY}
                and state.direct_pulses_per_wavelength is not None else api["detect_pulses_per_wavelength"]
                (state.views, **_peak_detection_runtime_kwargs(report.peak_detector))
            )
            pulse = pulses_per_wavelength[api["select_reference_wavelength"](pulses_per_wavelength)]
            direct_pulse = pulse
            prv_route = state.route
            q_rate_qualified = (
                True if state.final_quality is None else state.final_quality.q_rate.state is QualityState.PASS
            )
        prv = api["compute_prv"](
            pulse, observation_duration_s=state.views.x_filter.shape[0] / 400.0,
            role=api["canonicalize_role_family"](str(state.row.role)), route=prv_route,
            q_rate_qualified=q_rate_qualified, config=prv_config,
        )
        pulse_only = state.quality_tier == "acceptable"
        if pulse_only:
            values: dict[str, Any] = {}
            validity: dict[str, bool] = {}
            state.engineering = None
        else:
            plan = api["WindowPlan"](source_record_id=state.row.record_id, **report.window_profiles["engineering"])
            engineering = api["extract_engineering_features"](state.views, plan=plan)
            if state.routing_timeline is not None:
                from dataclasses import replace

                window_length = int(round(plan.window_seconds * 400.0))
                eligible_rows = api["np"].asarray([
                    bool((
                        cells :=
                        api["overlapping_cells"](state.routing_timeline, int(start), int(start) + window_length)
                    )) and all((cell.final_tier == "excellent" and cell.source_route == "direct" for cell in cells))
                    for start in engineering.sequence.start_samples
                ], dtype=bool,
                                                  )
                eligible_rows &= api["np"].asarray(engineering.sequence.valid_row_mask, dtype=bool)
                engineering = replace(
                    engineering, sequence=replace(engineering.sequence, valid_row_mask=eligible_rows),
                    value_validity=engineering.value_validity & eligible_rows[:, None],
                    reasons=tuple(dict.fromkeys((*engineering.reasons, "routing_excellent_direct_rows_only"))),
                )
            state.engineering = engineering
            values, validity = api["summarize_engineering"](engineering)
        if state.final_quality is not None:
            values["sqi.q_rate"] = float(state.final_quality.q_rate.score)
            validity["sqi.q_rate"] = True
            values["sqi.coverage"] = float(prv.values["coverage"])
            validity["sqi.coverage"] = True
            if state.final_quality.q_morph.score is not None:
                values["sqi.q_morph"] = float(state.final_quality.q_morph.score)
                validity["sqi.q_morph"] = True
        for name, value in prv.values.items():
            values[f"prv.{name}"] = value
            validity[f"prv.{name}"] = bool(prv.validity[name])
        if state.shape_features_eligible:
            morphology = api["extract_morphology"](state.views.x_filter, direct_pulse, route=SignalRoute.DIRECT)
            if state.routing_timeline is None:
                morphology_values = morphology.aggregate_values
                morphology_validity = morphology.aggregate_validity
            else:
                morphology_values, morphology_validity = api["route_eligible_morphology_aggregates"](
                    direct_pulse, morphology, state.routing_timeline
                )
            for name, value in morphology_values.items():
                values[f"morphology.{name}"] = value
                validity[f"morphology.{name}"] = bool(morphology_validity[name])
            optical_route_eligible = state.routing_timeline is None or all((
                cell.final_tier == "excellent" and cell.source_route == "direct"
                for cell in state.routing_timeline.cells
            ))
            if optical_route_eligible:
                optical = api["extract_dual_optical"](
                    state.views.x_native, state.views.x_filter, pulses_per_wavelength, route=SignalRoute.DIRECT
                )
                for name, value in optical.aggregate_values.items():
                    values[f"optical.{name}"] = value
                    validity[f"optical.{name}"] = bool(optical.aggregate_validity[name])
                state.diagnostic_components["dual_optical_pairing"] = _compact_dual_optical_diagnostics(optical)
            else:
                state.diagnostic_components["dual_optical_pairing"] = {
                    "status": "unavailable_mixed_or_excluded_routing_cells", "affects_prediction": True,
                }
        enabled_groups = None if features_config is None else features_config.get("enabled_groups")
        registry = api["registry_for_groups"](enabled_groups)
        complete_registry = api["default_registry"]()
        non_predictor_names = sorted(set(values) - set(complete_registry.names))
        disabled_registered_names = sorted(set(complete_registry.names) - set(registry.names))
        allowed_metadata_names = {"prv.coverage", "sqi.coverage", "sqi.q_morph", "sqi.q_rate"}
        unexpected = sorted(set(values) - set(complete_registry.names) - allowed_metadata_names)
        if unexpected:
            raise _ExperimentProtocolError("unregistered_feature_fields:" + ",".join(unexpected))
        state.diagnostic_components["non_predictor_features"] = {
            name: {"value": values.get(name), "valid": bool(validity.get(name, False))} for name in non_predictor_names
        }
        state.diagnostic_components["disabled_feature_groups"] = {
            "disabled_predictor_count": len(disabled_registered_names),
            "enabled_groups": list(dict.fromkeys((item.group for item in registry.definitions))),
        }
        predictor_values = {name: values[name] for name in registry.names if name in values}
        predictor_validity = {name: validity[name] for name in registry.names if name in validity}
        unavailable_predictors = tuple(
            (name for name in registry.names if not bool(predictor_validity.get(name, False)))
        )
        state.diagnostic_components["predictor_availability"] = {
            "schema_version": registry.schema_version, "registry_sha256": registry.sha256,
            "enabled_groups": list(dict.fromkeys((item.group for item in registry.definitions))
                                   ), "predictor_count": len(registry.names),
            "available_predictor_count": len(registry.names) - len(unavailable_predictors),
            "unavailable_predictor_count": len(unavailable_predictors),
            "unavailable_feature_names": unavailable_predictors,
        }
        state.vector = api["build_feature_vector"](
            predictor_values, feature_validity=predictor_validity, registry=registry, provenance={
                "route": state.route.value, "record_id": state.row.record_id,
                "non_predictor_metadata_fields": non_predictor_names, "sqi_and_coverage_predictors_excluded": True,
                "prv_config": prv_config.to_dict(), "quality_tier": state.quality_tier,
                "routing_interval_source_routes": [] if getattr(pulse, "interval_source_routes", None) is None else
                sorted(set(map(str, pulse.interval_source_routes)) - {"routing_boundary"}),
                "acceptable_contract": "record_specific_pulse_prv_only_nonpulse_slots_missing_outer_train_imputed"
                if pulse_only else "full_configured_feature_groups",
            },
        )
    except Exception as exc:
        _drop_after_routing(
            state, reason=f"feature_vector_failed:{type(exc).__name__}:{exc}",
            route_status="dropped_feature_vector_failure",
        )

def _dataset(states: Iterable[_RuntimeRecord]) -> Any:
    """物化 file-level vector dataset / Materialise a file-level vector dataset."""
    api = _runtime_imports()
    selected = sorted((state for state in states if state.retained and state.vector is not None),
                      key=lambda state: str(state.row.record_id),
                      )
    if not selected:
        return None
    names = tuple(selected[0].vector.feature_names)
    if any((tuple(state.vector.feature_names) != names for state in selected)):
        raise _ExperimentProtocolError("feature_vector_schema_drift")
    identities = tuple((
        api["SampleIdentity"](
            participant_id=state.row.participant_id, file_id=state.row.record_id,
            role=api["canonicalize_role_family"](str(state.row.role)), label=int(state.row.class_id),
            signal_route=state.route.value,
            quality_score=1.0 if state.final_quality is None else float(state.final_quality.q_rate.score),
        ) for state in selected
    ))
    return api["FeatureVectorDataset"](api["np"].stack([state.vector.values for state in selected]), names, identities)

def _extract_matrix_features(state: _RuntimeRecord, report: Any) -> None:
    """Extract the route-aware 146-feature variable-K window sequence."""
    if not state.retained:
        return
    api = _runtime_imports()
    try:
        if state.routing_timeline is None:
            raise RuntimeError("feature_matrix_requires_routing_timeline")
        plan = api["WindowPlan"](source_record_id=state.row.record_id, **report.window_profiles["engineering"])
        detector = report.peak_detector
        pulses = _direct_pulses_for_state(state, api, detector)
        direct_pulse = pulses[api["select_reference_wavelength"](pulses)]
        direct_morphology = api["extract_morphology"](
            state.views.x_filter, direct_pulse, route=api["SignalRoute"].DIRECT
        )
        processed_pulse = None
        if state.processed_pulses_per_wavelength is not None:
            processed_pulse = state.processed_pulses_per_wavelength[api["select_reference_wavelength"](
                state.processed_pulses_per_wavelength
            )]
        state.engineering = api["extract_window_features"](
            state.views, plan=plan, timeline=state.routing_timeline, direct_pulse=direct_pulse,
            direct_morphology=direct_morphology, processed_pulse=processed_pulse,
        )
    except Exception as exc:
        _drop_after_routing(
            state, reason=f"feature_matrix_engineering_failed:{type(exc).__name__}:{exc}",
            route_status="dropped_feature_matrix_engineering_failure",
        )

def _extract_raw(
    state: _RuntimeRecord, report: Any, signal_config: Mapping[str, Any] | None = None, cache_session: Any = None
) -> None:
    """Build configured mask-aware raw windows / 构建配置化 raw 窗口。"""
    if not state.retained:
        return
    api = _runtime_imports()
    try:
        plan = api["WindowPlan"](source_record_id=state.row.record_id, **report.window_profiles["raw_dl"])
        normalization = None
        if signal_config is not None:
            raw_normalization = signal_config.get("normalization", {})
            if not isinstance(raw_normalization, Mapping):
                raise TypeError("signal.normalization must be a mapping")
            normalization = dict(raw_normalization)
        if cache_session is not None and cache_session.enabled:
            upstream_key = state.preprocessing_cache_refs.get("canonical_signal_views")
            if upstream_key is None:
                raise RuntimeError("raw_window_cache_requires_upstream_view_key")
            materialized, raw_key = cache_session.raw_windows(
                state.row, upstream_views_key=upstream_key, plan=plan, normalization=normalization,
                builder=lambda: api["build_raw_windows"](state.views, plan, normalization=normalization),
            )
            state.preprocessing_cache_refs["raw_windows"] = raw_key
        else:
            materialized = api["build_raw_windows"](state.views, plan, normalization=normalization)
        if state.routing_timeline is not None:
            from dataclasses import replace

            keep = []
            for index, start_sample in enumerate(materialized.start_samples):
                stop_sample = int(start_sample) + int(materialized.values.shape[2])
                if stop_sample > state.routing_timeline.n_samples:
                    keep.append(False)
                    continue
                cells = api["overlapping_cells"](state.routing_timeline, int(start_sample), stop_sample)
                keep.append(
                    bool(cells)
                    and all((cell.final_tier == "excellent" and cell.source_route == "direct" for cell in cells))
                )
            selected = api["np"].asarray(keep, dtype=bool)
            materialized = replace(
                materialized, values=materialized.values[selected], valid_mask=materialized.valid_mask[selected],
                start_samples=materialized.start_samples[selected],
                dropped_invalid_count=int(materialized.dropped_invalid_count) +
                int(selected.size - api["np"].count_nonzero(selected)), provenance={
                    **dict(materialized.provenance), "routing_timeline_schema": state.routing_timeline.schema_version,
                    "routing_eligibility": "complete_support_excellent_direct",
                    "routing_excluded_window_count": int(selected.size - api["np"].count_nonzero(selected)),
                },
            )
        if materialized.values.shape[0] == 0:
            raise ValueError("no_complete_excellent_direct_raw_window")
        state.raw_windows = materialized
    except Exception as exc:
        from .data.recording_cache import RecordingCacheError

        if isinstance(exc, RecordingCacheError):
            raise
        _drop_after_routing(
            state, reason=f"raw_windows_failed:{type(exc).__name__}:{exc}", route_status="dropped_raw_window_failure"
        )

def _apply_window_quality_selection(
    states: list[_RuntimeRecord], config: Any, *, train_ids: tuple[str, ...], oof_ids: tuple[str, ...]
) -> dict[str, Any]:
    """Apply the configured label-free selector independently inside each file."""
    from .quality.window_selection import (
        WindowSelectionConfig, mark_raw_windows_for_aggregation, score_raw_windows, select_raw_windows,
    )

    selection = WindowSelectionConfig.from_mapping(config.section("quality").get("window_selection"))
    train = set(map(str, train_ids))
    oof = set(map(str, oof_ids))
    aggregation = config.section("aggregation")
    score_oof_for_aggregation = (
        bool(aggregation.get("quality_weighting", False))
        and aggregation.get("quality_weight_source") == "legacy_window_sqi"
    )
    summaries: list[dict[str, Any]] = []
    for state in states:
        if state.raw_windows is None:
            continue
        participant = str(state.row.participant_id)
        if participant not in train | oof:
            raise _ExperimentProtocolError("window_quality_selection_participant_outside_fold")
        apply_selection = selection.policy != "none" and (
            participant in train or selection.application_scope == "all_partitions"
        )
        apply_aggregation_selection = (
            selection.policy != "none" and participant in oof
            and (selection.application_scope == "legacy_train_and_aggregation")
        )
        if apply_selection:
            selected, summary = select_raw_windows(state.raw_windows, selection)
            state.raw_windows = selected
        elif apply_aggregation_selection:
            marked, summary = mark_raw_windows_for_aggregation(state.raw_windows, selection)
            state.raw_windows = marked
        elif selection.policy != "none" and score_oof_for_aggregation:
            scored, summary = score_raw_windows(state.raw_windows, selection)
            state.raw_windows = scored
        else:
            count = int(state.raw_windows.values.shape[0])
            summary = {
                "input_window_count": count, "retained_window_count": count, "aggregation_window_count": count,
                "score_vector_sha256": None, "aggregation_mask_sha256": None,
            }
        summaries.append({
            "record_id": str(state.row.record_id), "partition": "outer_train" if participant in train else "outer_oof",
            "input_window_count": int(summary["input_window_count"]),
            "retained_window_count": int(summary["retained_window_count"]), "selection_applied": bool(apply_selection),
            "aggregation_selection_applied": bool(apply_aggregation_selection),
            "aggregation_window_count": int(summary.get("aggregation_window_count", summary["retained_window_count"])
                                            ), "score_computed": summary.get("score_vector_sha256") is not None,
            "score_vector_sha256": summary.get("score_vector_sha256"),
            "aggregation_mask_sha256": summary.get("aggregation_mask_sha256"),
        })
    score_hash_rows = [{
        name: row[name] for name in ("record_id", "partition", "score_vector_sha256", "aggregation_mask_sha256")
    } for row in sorted(summaries, key=lambda item: item["record_id"]) if row["score_vector_sha256"] is not None]
    aggregate_fields = {
        "input_window_count": "input_window_count", "retained_window_count": "retained_window_count",
        "selection_applied_file_count": "selection_applied",
        "aggregation_selection_applied_file_count": "aggregation_selection_applied",
        "aggregation_window_count": "aggregation_window_count", "score_computed_file_count": "score_computed",
    }
    partition_fields = {
        "files": None, "input_windows": "input_window_count", "retained_windows": "retained_window_count",
        "selection_applied_files": "selection_applied",
        "aggregation_selection_applied_files": "aggregation_selection_applied",
        "aggregation_windows": "aggregation_window_count", "scored_files": "score_computed",
    }
    return {
        **selection.to_mapping(), "scope": "independent_within_each_file",
        "application_scope": selection.application_scope, "uses_labels": False, "fitted_on_participant_ids": [],
        "cross_file_statistics": False, "outer_oof_used_for_training_fit": False, "file_count": len(summaries), **{
            output: sum(row[source] for row in summaries) for output, source in aggregate_fields.items()
        }, "score_vector_bundle_sha256": hashlib.sha256(
            json.dumps(score_hash_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest() if score_hash_rows else None, "partition_counts": {
            partition: {
                output: sum((1 if source is None else row[source]) for row in summaries
                            if row["partition"] == partition) for output, source in partition_fields.items()
            } for partition in ("outer_train", "outer_oof")
        },
    }

def _retained_states(states: Iterable[_RuntimeRecord],
                     participant_ids: Iterable[str] | None = None) -> list[_RuntimeRecord]:
    """Return deterministic retained states / 返回确定排序的保留 recording。"""
    participants = None if participant_ids is None else set((str(value) for value in participant_ids))
    return sorted((
        state for state in states
        if state.retained and (participants is None or str(state.row.participant_id) in participants)
    ), key=lambda state: str(state.row.record_id),
                  )

def _assert_train_payload_roster(
    states: Iterable[_RuntimeRecord], train_ids: tuple[str, ...], *, required: tuple[str, ...]
) -> None:
    """Require every frozen train participant to reach the requested representation."""
    observed = {
        str(state.row.participant_id) for state in states
        if state.retained and all((getattr(state, name) is not None for name in required))
    }
    missing = sorted(set(train_ids) - observed)
    if missing:
        raise _ExperimentProtocolError("outer_train_subject_zero_representation_rows:" + ",".join(missing))

def _fit_representation_artifacts(
    states: list[_RuntimeRecord], mode: str, train_ids: tuple[str, ...], oof_ids: tuple[str, ...], *,
    fitted_objects: dict[str, Any] | None = None, raw_normalization_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit and apply only the transforms required by one representation mode."""
    api = _runtime_imports()
    provenance: dict[str, Any] = {}
    selected = _retained_states(states)
    if mode in {"raw", "fusion"}:
        _assert_train_payload_roster(states, train_ids, required=("raw_windows", ))
        raw_states = [state for state in selected if state.raw_windows is not None]
        normalization_payloads = [
            state.raw_windows.provenance.get("normalization_config") for state in raw_states
            if state.raw_windows.provenance.get("normalization_config") is not None
        ]
        normalization = (
            dict(raw_normalization_override) if raw_normalization_override is not None else
            dict(normalization_payloads[0]) if normalization_payloads else None
        )
        if any((dict(value) != normalization for value in normalization_payloads)):
            raise _ExperimentProtocolError("raw_normalization_config_differs_across_outer_fold_records")
        from .normalization import IMU_NONE, RawNormalizationConfig

        normalization_config = RawNormalizationConfig.from_mapping(normalization)
        if normalization_config.raw_imu == IMU_NONE:
            provenance["raw_imu"] = {
                "schema_version": "not_applicable_all8_window_normalized_v1", "artifact_sha256": None,
                "fitted_on_participant_ids": (), "channel_schema": ("A_dyn_x", "A_dyn_y", "A_dyn_z", "GX", "GY", "GZ"),
                "valid_count": None, "strategy": "none_after_all8_per_window_robust", "parameters": None,
            }
        else:
            raw_values = api["np"].concatenate([state.raw_windows.values for state in raw_states], axis=0)
            raw_masks = api["np"].concatenate([state.raw_windows.valid_mask for state in raw_states], axis=0)
            raw_participants = tuple((
                str(state.row.participant_id) for state in raw_states for _ in range(state.raw_windows.values.shape[0])
            ))
            imu_transform = api["fit_fold_imu_channel_transform"](
                raw_values, raw_participants, fitted_on_participant_ids=train_ids,
                outer_train_participant_ids=train_ids, outer_oof_participant_ids=oof_ids, valid_mask=raw_masks,
                normalization=normalization,
            )
            if fitted_objects is not None:
                fitted_objects["raw_imu"] = imu_transform
            for state in raw_states:
                state.raw_windows = api["transform_raw_windows_imu"](state.raw_windows, imu_transform)
            provenance["raw_imu"] = {
                "schema_version": imu_transform.schema_version, "artifact_sha256": imu_transform.artifact_sha256,
                "fitted_on_participant_ids": imu_transform.fitted_on_participant_ids,
                "channel_schema": imu_transform.channel_schema, "valid_count": imu_transform.valid_count.tolist(),
                "strategy": imu_transform.strategy, "parameters": {
                    "iqr_fallback": imu_transform.iqr_fallback, "robust_iqr_divisor": imu_transform.robust_iqr_divisor,
                    "mad_consistency_divisor": imu_transform.mad_consistency_divisor,
                    "scale_epsilon": imu_transform.scale_epsilon, "standard_ddof": imu_transform.standard_ddof,
                },
            }
    if mode == "fusion":
        _assert_train_payload_roster(states, train_ids, required=("vector", "engineering"))
        vector_states = [
            state for state in _retained_states(states) if state.vector is not None and state.engineering is not None
        ]
        vector_transform = api["fit_fold_feature_vector_transform"](
            [state.vector for state in vector_states], [state.row.participant_id for state in vector_states],
            fitted_on_participant_ids=train_ids, outer_train_participant_ids=train_ids,
            outer_oof_participant_ids=oof_ids,
        )
        if fitted_objects is not None:
            fitted_objects["feature_vector"] = vector_transform
        vector_batch = api["transform_feature_vector_batch"]([state.vector for state in vector_states],
                                                             vector_transform)
        for state, context, features in zip(vector_states, vector_batch.contexts, vector_batch.fusion_tensor):
            state.transformed_vector = context
            state.fusion_features = api["np"].asarray(features, dtype=api["np"].float32)
        provenance["feature_vector"] = {
            "schema_version": vector_transform.schema_version, "artifact_sha256": vector_transform.artifact_sha256,
            "registry_sha256": vector_transform.registry_sha256,
            "fitted_on_participant_ids": vector_transform.fitted_on_participant_ids,
            "valid_count": vector_transform.valid_count.tolist(), "fusion_tensor_schema": vector_batch.tensor_schema,
        }
    if mode == "feature_matrix":
        _assert_train_payload_roster(states, train_ids, required=("engineering", ))
        matrix_states = [state for state in _retained_states(states) if state.engineering is not None]
        train_engineering = [
            state.engineering for state in matrix_states if str(state.row.participant_id) in set(train_ids)
        ]
        engineering_transform = api["fit_fold_window_feature_transform"](
            train_engineering, fitted_on_participant_ids=train_ids, outer_train_participant_ids=train_ids,
            outer_oof_participant_ids=oof_ids,
        )
        if fitted_objects is not None:
            fitted_objects["engineering"] = engineering_transform
        engineering_payload = {
            "center": engineering_transform.center.tolist(), "scale": engineering_transform.scale.tolist(),
            "valid_count": engineering_transform.valid_count.tolist(),
            "feature_names": engineering_transform.feature_names,
            "fitted_on_participant_ids": engineering_transform.fitted_on_participant_ids,
        }
        engineering_hash = api["stable_payload_sha256"](engineering_payload)
        for state in matrix_states:
            try:
                state.transformed_engineering = api["transform_window_features"](
                    state.engineering, engineering_transform
                )
                state.matrix = api["build_ordered_window_matrix"](
                    state.transformed_engineering, provenance={
                        "route": state.route.value, "record_id": state.row.record_id,
                        "engineering_transform_sha256": engineering_hash,
                    },
                )
            except Exception as exc:
                _drop_after_routing(
                    state, reason=f"feature_matrix_failed:{type(exc).__name__}:{exc}",
                    route_status="dropped_feature_matrix_failure",
                )
        provenance["engineering"] = {
            **engineering_payload, "artifact_sha256": engineering_hash,
            "schema_version": "window_feature_set_d146_outer_train_robust_v1",
            "matrix_length_policy": "all_complete_windows_variable_k", "matrix_predictor_channels": 146,
            "batch_padding_policy": "batch_only_to_max_k_with_false_mask", "file_context_predictor_channels": 0,
            "validity_predictor_channels": 0,
        }
        _assert_train_payload_roster(states, train_ids, required=("matrix", ))
    if mode == "fusion":
        _assert_train_payload_roster(states, train_ids, required=("raw_windows", "fusion_features"))
    return provenance

def _legacy_bridge_representation_artifacts(
    states: list[_RuntimeRecord], profile: Any, train_ids: tuple[str, ...], oof_ids: tuple[str, ...]
) -> dict[str, Any]:
    """Apply only the normalization stage named by the reviewed bridge profile."""
    _assert_train_payload_roster(states, train_ids, required=("raw_windows", ))
    if profile.uses_fold_imu_transform:
        provenance = _fit_representation_artifacts(
            states, "raw", train_ids, oof_ids, raw_normalization_override={"raw_imu": "outer_train_robust"}
        )
        if profile.protocol_design != "cumulative_chain_v1":
            actual_imu_schema = tuple(profile.channel_schema[2:])
            transform_schema = tuple(provenance["raw_imu"]["channel_schema"])
            provenance["raw_imu"]["actual_ordered_source_channel_schema"] = actual_imu_schema
            provenance["raw_imu"]["registry_schema_is_positional_alias_only"] = actual_imu_schema != transform_schema
            provenance["raw_imu"]["tensor_channel_order_changed_by_alias"] = False
        provenance["legacy_bridge_normalization_stage"] = {
            "profile_id": profile.profile_id, "ppg": "legacy_profile_specific_window_scaling"
            if profile.protocol_design == "cumulative_chain_v1" else "legacy_profile_specific_window_scaling",
            "imu": "outer_train_axes6_fold_robust",
        }
        provenance["legacy_bridge_window_materialization"] = {
            "end_alignment": "include_right_aligned_if_distinct", "padding": "none_complete_windows_only",
            "source_rows_required_valid": True, "cap_per_file": profile.max_windows_per_file,
        }
        return provenance
    complete_windows_only = not profile.resolved_allow_short_record_padding
    window_materialization = {
        "end_alignment": "include_right_aligned_if_distinct", "padding": "none_complete_windows_only"
        if complete_windows_only else "zero_right_only_if_source_shorter_than_one_window",
        "source_rows_required_valid": True, "historical_retained_fraction": profile.historical_retained_fraction,
        "cap_per_file": profile.max_windows_per_file,
    }
    if not complete_windows_only:
        window_materialization.update({
            "padding_occurs_after_scaling_available_source_rows": True, "padded_values": 0.0
        })
    return {
        "legacy_bridge_normalization_stage": {
            "profile_id": profile.profile_id,
            "all_eight_channels": "per_window_median_iqr_over_1p349_sd_fallback_clip_-8_8",
            "outer_train_fitted_transform": False, "outer_oof_used_for_fitting": False,
        }, "legacy_bridge_window_materialization": window_materialization,
    }

def _sample_identity(
    state: _RuntimeRecord, *, window_id: str | None = None, quality_score: float | None = None,
    aggregation_retained: bool = True,
) -> Any:
    """Create one canonical physiological identity / 创建规范生理 role 身份。"""
    api = _runtime_imports()
    quality = quality_score
    if quality is None:
        quality = (
            1.0 if state.final_quality is None or state.final_quality.q_rate.score is None else
            float(state.final_quality.q_rate.score)
        )
    return api["SampleIdentity"](
        participant_id=str(state.row.participant_id), file_id=str(state.row.record_id),
        role=api["canonicalize_role_family"](str(state.row.role)), label=int(state.row.class_id),
        signal_route=state.route.value, quality_score=quality, window_id=window_id,
        aggregation_retained=bool(aggregation_retained),
    )

def _materialize_representation_dataset(
    states: Iterable[_RuntimeRecord], participant_ids: Iterable[str], mode: str, *, quality_weight_source: str = "none"
) -> Any:
    """Materialise exactly one typed dataset / 物化严格类型化的数据集。"""
    api = _runtime_imports()
    selected = _retained_states(states, participant_ids)
    if mode == "feature_vector":
        selected = [state for state in selected if state.vector is not None]
        if not selected:
            return None
        names = tuple(selected[0].vector.feature_names)
        if any((tuple(state.vector.feature_names) != names for state in selected)):
            raise _ExperimentProtocolError("feature_vector_schema_drift")
        return api["FeatureVectorDataset"](
            api["np"].stack([state.vector.values for state in selected]), names,
            tuple((_sample_identity(state) for state in selected)),
        )
    if mode == "raw":
        selected = [state for state in selected if state.raw_windows is not None]
        if not selected:
            return None
        values = []
        masks = []
        identities = []
        for state in selected:
            window_scores = state.raw_windows.window_quality_scores
            aggregation_mask = state.raw_windows.window_aggregation_mask
            if quality_weight_source == "legacy_window_sqi" and window_scores is None:
                raise _ExperimentProtocolError("legacy_window_sqi_scores_missing_from_raw_windows")
            for index, start_sample in enumerate(state.raw_windows.start_samples):
                values.append(state.raw_windows.values[index])
                masks.append(state.raw_windows.valid_mask[index])
                identities.append(
                    _sample_identity(
                        state, window_id=f"{state.row.record_id}::start_{int(start_sample):012d}",
                        quality_score=float(window_scores[index]) if window_scores is not None else None,
                        aggregation_retained=True if aggregation_mask is None else bool(aggregation_mask[index]),
                    )
                )
        return api["RawWindowDataset"](api["np"].stack(values), tuple(identities), api["np"].stack(masks))
    if mode == "feature_matrix":
        selected = [state for state in selected if state.matrix is not None]
        if not selected:
            return None
        return api["FeatureMatrixDataset"].from_contracts(
            tuple((state.matrix for state in selected)), tuple((_sample_identity(state) for state in selected))
        )
    if mode == "fusion":
        selected = [state for state in selected if state.raw_windows is not None and state.fusion_features is not None]
        if not selected:
            return None
        return api["FileBagDataset"](
            tuple((state.raw_windows.values for state in selected)),
            api["np"].stack([state.fusion_features for state in selected]),
            tuple((_sample_identity(state) for state in selected)),
            tuple((state.raw_windows.valid_mask for state in selected)),
        )
    raise _ExperimentProtocolError(f"unsupported_representation_mode:{mode}")

def _prepare_dl_input_dataset(dataset: Any, mode: str,
                              dl_config: Mapping[str, Any]) -> tuple[Any, str | None, dict[str, object] | None]:
    """Apply one configured model-input sampling transform to raw or fusion."""
    enabled = bool(dl_config["enabled"])
    case_id = dl_config.get("case_id")
    if not enabled and case_id is None:
        return (dataset, None, None)
    if case_id is not None and mode != "raw":
        raise _ExperimentProtocolError("named_fixed_kernel_dl_resampling_requires_raw_representation")
    if case_id is None and mode not in {"raw", "fusion"}:
        raise _ExperimentProtocolError("generic_dl_resampling_requires_raw_or_fusion_representation")
    source_fs_hz = float(dl_config["preserve_feature_grid_hz"])
    if case_id is not None:
        from .models.time_scale import prepare_fixed_kernel_dl_input

        values, mask, profile = prepare_fixed_kernel_dl_input(
            dataset.values, dataset.sample_mask, str(case_id), source_fs_hz=source_fs_hz
        )
        from .training.datasets import RawWindowDataset

        return (RawWindowDataset(values, dataset.identities, mask), "fixed_kernel_samples", profile)
    from .signal.resample import prepare_configured_dl_input

    target_fs_hz = float(dl_config["target_fs_hz"])
    if mode == "raw":
        values, mask, profile = prepare_configured_dl_input(
            dataset.values, dataset.sample_mask, target_fs_hz=target_fs_hz, source_fs_hz=source_fs_hz
        )
        from .training.datasets import RawWindowDataset

        return (RawWindowDataset(values, dataset.identities, mask), "dl_input_resampling", profile)
    import numpy as np

    counts = tuple((int(bag.shape[0]) for bag in dataset.window_bags))
    values = np.concatenate(dataset.window_bags, axis=0)
    masks = np.concatenate(dataset.sample_masks, axis=0)
    transformed, transformed_masks, profile = prepare_configured_dl_input(
        values, masks, target_fs_hz=target_fs_hz, source_fs_hz=source_fs_hz
    )
    offsets = np.cumsum((0, ) + counts)
    bags = tuple((transformed[offsets[index]:offsets[index + 1]] for index in range(len(counts))))
    bag_masks = tuple((transformed_masks[offsets[index]:offsets[index + 1]] for index in range(len(counts))))
    from .training.datasets import FileBagDataset

    return (FileBagDataset(bags, dataset.file_features, dataset.identities, bag_masks), "dl_input_resampling", profile)

_ENSEMBLE_BASE_MODEL_IDS = {
    "inception_full_five_member_ensemble": "inception_full",
    "inception_matrix_five_member_ensemble": "inception_matrix",
}

def _ensemble_member_seed_roster(model_section: Mapping[str, Any]) -> tuple[int, ...]:
    """Resolve the member roster as the single source of ensemble cardinality."""
    from .models import resolve_seed_policy

    policy = str(model_section.get("seed_policy", "member_roster"))
    return resolve_seed_policy(policy, member_seeds=model_section.get("member_seeds"))

def _resolved_model_config(config: Any, *, training_seed: int,
                           seed_scope: str = "outer_cv") -> tuple[dict[str, Any], str]:
    """Map archive metadata to strict factory options / 将归档字段映射为严格工厂参数。"""
    api = _runtime_imports()
    section = config.section("model")
    canonical_name, machine_id = api["normalize_model_id"](str(section["model_id"]))
    factory_contract = api["model_factory_contract"](canonical_name)
    factory_fields = set(factory_contract["factory_fields"])
    is_ensemble = "member_seeds" in factory_fields
    seed_policy = str(section.get("seed_policy", ""))
    if is_ensemble:
        member_seeds = _ensemble_member_seed_roster(section)
    resolved: dict[str, Any] = {"model_id": machine_id}
    for field in sorted(factory_fields):
        if field not in section and field != "seed":
            continue
        value = int(training_seed) if field == "seed" and (not is_ensemble) else section[field]
        if field in {
            "member_seeds", "kernel_sizes", "dilations", "pool_sizes", "stage_channels", "stage_dropouts",
            "signal_kernel_sizes", "signal_dilations", "signal_pool_sizes", "signal_stage_channels",
            "signal_stage_dropouts",
        }:
            value = tuple(value)
        resolved[field] = value
    if is_ensemble:
        resolved["member_seeds"] = member_seeds
        resolved["seed_policy"] = seed_policy or "member_roster"
        member_variant = section.get("member_variant")
        resolved["variant"] = str(member_variant if member_variant is not None else "full")
        resolved.pop("member_variant", None)
    else:
        resolved["seed_policy"] = "fixed_explicit" if seed_scope == "final_refit" else seed_policy
    resolved["architecture_parameters"] = api["materialize_model_architecture"](section)
    return (resolved, machine_id)

def _outer_cv_model_training_seed(config: Any, split: Mapping[str, Any]) -> int:
    """Resolve the configurable orchestration seed independently of member RNGs."""
    model = config.section("model")
    policy = str(model.get("seed_policy", ""))
    if policy == "cv_fixed_member0_seed_50042_comparator":
        return 50042
    if policy in {"outer_repeat", "outer_cv_repeat_seed_equals_split_seed"}:
        return int(split["training_seed"])
    if policy in {
        "member_roster", "cv_fixed_five_member_seed_roster", "fixed", "fixed_explicit", "final_refit_single_seed_42",
    }:
        return int(config.section("training")["seed"])
    raise _ExperimentProtocolError("unsupported_outer_cv_seed_policy")

_CANONICAL_RAW_CHANNEL_SCHEMA = ("RED", "IR", "A_dyn_x", "A_dyn_y", "A_dyn_z", "GX", "GY", "GZ")

def _bind_raw_dataset_for_model(dataset: Any, model_id: str, *,
                                declared_channel_order: Iterable[str] | None = None) -> tuple[Any, dict[str, Any]]:
    """Bind an explicit ordered DL-channel view from the canonical tensor."""
    api = _runtime_imports()
    source_schema = _CANONICAL_RAW_CHANNEL_SCHEMA
    values = api["np"].asarray(dataset.values)
    if values.ndim != 3 or values.shape[1] != len(source_schema):
        raise _ExperimentProtocolError(f"canonical_frailty_raw_tensor_must_be_8_channels:{values.shape}")
    declared = tuple((str(value) for value in declared_channel_order or ()))
    if (
        not declared or len(declared) != len(set(declared)) or any((value not in source_schema for value in declared))
        or (tuple((value for value in source_schema if value in declared)) != declared)
    ):
        raise _ExperimentProtocolError("frailty_model_input_channel_order_must_be_ordered_canonical_subset")
    source_indices = tuple((source_schema.index(value) for value in declared))
    if declared == source_schema:
        bound = dataset.__class__(values, dataset.identities, dataset.sample_mask, channel_schema=declared)
        status = "canonical_frailty_raw_8_identity"
    else:
        bound = dataset.__class__(
            values[:, source_indices, :], dataset.identities, dataset.sample_mask, channel_schema=declared
        )
        status = "explicit_frailty_raw_channel_subset"
    payload = {
        "status": status, "model_id": str(model_id), "source_channel_schema": source_schema,
        "target_channel_schema": declared, "source_indices": source_indices, "derived_motion_channels_present": False,
        "silent_channel_slicing": False, "physical_analysis_views_unchanged": True,
    }
    return (bound, {**payload, "binding_sha256": api["stable_payload_sha256"](payload)})

def _model_input_spec(dataset: Any, mode: str) -> Any:
    """Resolve dimensions from materialised train data / 从训练数据解析输入维度。"""
    api = _runtime_imports()
    if mode == "feature_vector":
        return api["ModelInputSpec"](mode, n_classes=3, feature_names=tuple(dataset.feature_names))
    if mode == "raw":
        channel_count = int(dataset.values.shape[1])
        channel_schema = tuple(getattr(dataset, "channel_schema", ()))
        if not channel_schema and channel_count == len(_CANONICAL_RAW_CHANNEL_SCHEMA):
            channel_schema = _CANONICAL_RAW_CHANNEL_SCHEMA
        if (
            len(channel_schema) != channel_count
            or any((value not in _CANONICAL_RAW_CHANNEL_SCHEMA for value in channel_schema))
            or tuple((value for value in _CANONICAL_RAW_CHANNEL_SCHEMA if value in channel_schema)) != channel_schema
        ):
            raise _ExperimentProtocolError(f"frailty_raw_channel_schema_invalid:{channel_count}:{channel_schema}")
        return api["ModelInputSpec"](mode, n_channels=channel_count, n_classes=3, channel_schema=channel_schema)
    if mode == "feature_matrix":
        return api["ModelInputSpec"](
            mode, n_channels=int(dataset.n_channels), n_classes=3, channel_schema=tuple(dataset.channel_schema)
        )
    if mode == "fusion":
        channel_count = int(dataset.window_bags[0].shape[1])
        if channel_count != len(_CANONICAL_RAW_CHANNEL_SCHEMA):
            raise _ExperimentProtocolError(f"fusion_requires_canonical_frailty_raw_8_channels:{channel_count}")
        return api["ModelInputSpec"](
            mode, n_channels=channel_count, n_classes=3, n_file_features=int(dataset.file_features.shape[1]),
            channel_schema=_CANONICAL_RAW_CHANNEL_SCHEMA,
        )
    raise _ExperimentProtocolError(f"unsupported_representation_mode:{mode}")

def _prepare_legacy_bridge_model_factory(
    api: Mapping[str, Any], model_config: Mapping[str, Any], input_spec: Any, train_dataset: Any, frozen: Any, *,
    profile: Any,
) -> _LegacyBridgePreparedFactory:
    """Use a transport-only canonical alias without falsifying bridge semantics."""
    if tuple(input_spec.channel_schema) != tuple(profile.channel_schema):
        raise _ExperimentProtocolError("legacy_bridge_input_schema_drift")
    transport_spec = api["ModelInputSpec"](
        "raw", n_channels=8, n_classes=3, channel_schema=_CANONICAL_RAW_CHANNEL_SCHEMA
    )
    canonical = api["prepare_model_factory"](model_config, transport_spec, train_dataset, frozen)
    provenance = {
        **dict(canonical.provenance), "legacy_bridge_profile_id": profile.profile_id,
        "actual_ordered_channel_schema": tuple(profile.channel_schema),
        "factory_transport_channel_schema": _CANONICAL_RAW_CHANNEL_SCHEMA,
        "factory_transport_alias_only": tuple(profile.channel_schema) != _CANONICAL_RAW_CHANNEL_SCHEMA,
        "tensor_channel_order_changed_by_alias": False, "canonical_model_validator_modified_or_relaxed": False,
    }
    return _LegacyBridgePreparedFactory(canonical, provenance)

def _legacy_bridge_allowed_model_ids(profile: Any) -> frozenset[str]:
    """Return the model family scope declared by the protocol design."""
    if profile.protocol_design != "cumulative_chain_v1":
        return frozenset({"compact_cnn", "inception_full"})
    if profile.profile_id == "L0":
        return frozenset({"compact_cnn", "inception_full"})
    return frozenset({"compact_cnn"})

@dataclass(frozen=True)
class _TrustedFull29Materialization:
    """Internally materialised all-29 representation and immutable evidence."""

    dataset: Any
    input_spec: Any
    fitted_objects: Mapping[str, Any]
    representation_provenance: Mapping[str, Any]
    quality_provenance: Mapping[str, Any]
    preprocessing_hash: str
    feature_hash: str
    feature_schema_id: str
    feature_contract: Mapping[str, Any]
    source_records: tuple[Mapping[str, Any], ...]
    source_records_hash: str
    dataset_hash: str
    golden_inputs: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_records", tuple(map(dict, self.source_records)))

def _golden_model_inputs(dataset: Any, mode: str) -> dict[str, Any]:
    """Build one deterministic already-transformed model-input golden case."""
    api = _runtime_imports()
    np = api["np"]
    if mode == "raw":
        return {
            "x": np.asarray(dataset.values[:1], dtype=np.float32),
            "mask": np.asarray(dataset.sample_mask[:1], dtype=bool),
        }
    if mode == "feature_vector":
        return {"x": np.asarray(dataset.values[:1], dtype=np.float32)}
    if mode == "feature_matrix":
        return {
            "x": np.asarray(dataset.values[:1], dtype=np.float32), "mask": np.asarray(dataset.row_mask[:1], dtype=bool),
        }
    if mode == "fusion":
        bag = np.asarray(dataset.window_bags[0], dtype=np.float32)
        return {
            "window_bag": bag[None, :, :, :], "window_mask": np.ones((1, bag.shape[0]), dtype=bool),
            "file_features": np.asarray(dataset.file_features[:1], dtype=np.float32),
            "sample_mask": np.asarray(dataset.sample_masks[0], dtype=bool)[None, :, :],
        }
    raise _ExperimentProtocolError(f"unsupported_representation_mode:{mode}")

def _materialize_trusted_full29(
    report: Any, config: Any, rows: Iterable[Any], paths: Any, preflight: Mapping[str, Any] | None = None,
) -> _TrustedFull29Materialization:
    """Fit preprocessing on the complete cohort selected by the resolved case."""
    from .training import dataset_binding_hash

    api = _runtime_imports()
    mode = str(config.representation_mode)
    row_values = tuple(rows)
    roles = set(_classifier_role_ids(config))
    participant_ids = tuple(
        sorted(
            map(str, preflight["participant_ids"]) if preflight is not None and "participant_ids" in preflight else {
                row.participant_id for row in row_values
                if row.role in roles and row.qc_status in {"pass", "pass_with_warnings"}
            }
        )
    )
    selected = _choose_records(row_values, participant_ids, roles, None)
    states = [_RuntimeRecord(row=row) for row in selected]
    _preprocess_records(
        states, config, None, lambda row, maximum: api["_load_record"](row, paths, max_samples=maximum),
        calibration_rows=row_values,
    )
    quality_mode = _quality_mode(config)
    quality_provenance = _apply_quality_motion_routing(
        states, config, report, paths, train_ids=participant_ids, oof_ids=()
    )
    for state in states:
        if mode in {"feature_vector", "fusion"}:
            _extract_vector(state, report, config.section("features"))
        elif mode == "feature_matrix":
            _extract_matrix_features(state, report)
        if mode in {"raw", "fusion"}:
            _extract_raw(state, report, config.section("signal"))
    window_selection_provenance = (
        _apply_window_quality_selection(states, config, train_ids=participant_ids, oof_ids=())
        if mode in {"raw", "fusion"} else None
    )
    if window_selection_provenance is not None and window_selection_provenance["policy"] != "none":
        quality_provenance = {
            **quality_provenance, "classification_effect": str(quality_provenance["classification_effect"]) +
            "+legacy_per_file_window_selection", "window_selection": window_selection_provenance,
        }
    required_by_mode = {
        "raw": ("raw_windows", ), "feature_vector": ("vector", ), "feature_matrix": ("engineering", ),
        "fusion": ("vector", "engineering", "raw_windows"),
    }
    if mode not in required_by_mode:
        raise _ExperimentProtocolError(f"unsupported_representation_mode:{mode}")
    _assert_train_payload_roster(states, participant_ids, required=required_by_mode[mode])
    fitted_objects: dict[str, Any] = {}
    representation_provenance = _fit_representation_artifacts(
        states, mode, participant_ids, (), fitted_objects=fitted_objects
    )
    if window_selection_provenance is not None:
        representation_provenance["window_quality_selection"] = window_selection_provenance
    if mode == "feature_vector":
        representation_provenance = {
            "feature_vector_estimator_pipeline": {
                "status": "fitted_inside_final_all29_estimator", "external_representation_transform": "not_applicable",
                "fitted_on_participant_ids": participant_ids,
            }
        }
    dataset = _materialize_representation_dataset(
        states, participant_ids, mode,
        quality_weight_source=config.section("aggregation").get("quality_weight_source", "none"),
    )
    if dataset is None:
        raise _ExperimentProtocolError("final_refit_all29_dataset_empty")
    dl_config = config.section("signal")["dl_resampling"]
    dataset, provenance_key, profile = _prepare_dl_input_dataset(dataset, mode, dl_config)
    if provenance_key is not None:
        if profile is None:
            raise _ExperimentProtocolError("dl_resampling_profile_missing")
        representation_provenance = {
            **dict(representation_provenance), provenance_key: {
                **profile, "application_scope": "dl_input_only_after_canonical_400hz_windows",
                "canonical_features_and_peaks_unchanged": True, "final_all29_same_frozen_transform_identity": True,
            },
        }
    if mode == "raw":
        model_section = config.section("model")
        _, model_id = api["normalize_model_id"](str(model_section["model_id"]))
        dataset, input_binding = _bind_raw_dataset_for_model(
            dataset, model_id, declared_channel_order=model_section.get("input_channel_order")
        )
        representation_provenance = {**dict(representation_provenance), "model_input_binding": input_binding}
    source_records_tuple = tuple(({
        **{
            name: str(getattr(state.row, name)) for name in (
                "record_id", "participant_id", "role", "source_path", "source_hash", "source_version"
            )
        }, "class_id": int(state.row.class_id), "fs_hz": float(state.row.fs), "n_samples": int(state.row.n_samples),
        "retained": bool(state.retained), "route_status": str(state.route_status),
        "route_artifact": dict(state.route_artifact), "rejection_reason": state.reason,
        "quality_diagnostics": dict(state.diagnostic_components), "quality_diagnostics_reason": state.diagnostic_reason,
        "artifact_reducer_name": state.artifact_name, "artifact_reducer_version": state.artifact_version,
        "physical_qc_source_byte_identity": dict(state.physical_qc_evidence.get("source_byte_identity", {})),
    }) for state in sorted(states, key=lambda value: str(value.row.record_id)))
    source_records_hash = api["stable_payload_sha256"](source_records_tuple)
    registry = api["registry_for_groups"](config.section("features")["enabled_groups"])
    input_spec = _model_input_spec(dataset, mode)
    if mode == "feature_vector":
        feature_schema_id = registry.schema_version
        feature_contract: dict[str, Any] = {
            "registry_hash": registry.sha256, "feature_names": tuple(dataset.feature_names),
            "estimator_transforms": "fit_inside_final_all29_estimator_pipeline",
        }
    elif mode == "raw":
        feature_schema_id = "raw_red_ir_imu_axes_8ch_all29_scaled_v2"
        feature_contract = {
            "channel_schema": tuple(input_spec.channel_schema),
            "raw_imu_transform": representation_provenance.get("raw_imu"),
            "model_input_binding": representation_provenance.get("model_input_binding"),
            "fixed_kernel_samples": representation_provenance.get("fixed_kernel_samples", {"status": "not_applied"}),
        }
    elif mode == "feature_matrix":
        first = _retained_states(states, participant_ids)[0]
        feature_schema_id = str(first.matrix.schema_version)
        feature_contract = {
            "channel_schema": tuple(dataset.channel_schema), "transforms": representation_provenance,
            "matrix_length_policy": "all_complete_windows_variable_k",
            "matrix_k_by_recording": list(dataset.sequence_lengths),
        }
    else:
        feature_schema_id = f"fusion_raw8_bag_plus_vector_{2 * len(registry.names)}_registry-{registry.sha256[:12]}_v3"
        feature_contract = {
            "channel_schema": tuple(input_spec.channel_schema), "file_feature_width": int(input_spec.n_file_features),
            "transforms": representation_provenance,
        }
    if mode != "raw":
        feature_contract["runtime_features_config"] = config.section("features")
    feature_hash = api["stable_payload_sha256"](feature_contract)
    preprocessing_hash = api["stable_payload_sha256"]({
        "signal": config.section("signal"), "quality": config.section("quality"),
        "artifact": config.section("artifact"), "quality_mode": quality_mode, "quality_provenance": quality_provenance,
        "representation_transforms": representation_provenance, "physical_recording_qc_profile": tuple(
            (state.physical_qc_profile for state in states)
        ), "source_records_hash": source_records_hash, "fit_scope": "verified_all29_participants",
    })
    dataset_hash = dataset_binding_hash(dataset)
    return _TrustedFull29Materialization(
        dataset=dataset, input_spec=input_spec, fitted_objects=fitted_objects,
        representation_provenance=representation_provenance, quality_provenance=quality_provenance,
        preprocessing_hash=preprocessing_hash, feature_hash=feature_hash, feature_schema_id=feature_schema_id,
        feature_contract=feature_contract, source_records=source_records_tuple, source_records_hash=source_records_hash,
        dataset_hash=dataset_hash, golden_inputs=_golden_model_inputs(dataset, mode),
    )

def _resolved_architecture_parameters(model_id: str, model_section: Mapping[str, Any]) -> dict[str, Any]:
    """Rematerialize architecture provenance from top-level factory inputs."""
    resolved = _runtime_imports()["materialize_model_architecture"](model_section)
    if str(resolved.get("model_id")) != model_id:
        raise _ExperimentProtocolError("architecture_parameters_model_id_mismatch")
    return resolved

_UNBOUND_CODE_VERSION = "not_git_bound"

def _code_version() -> str:
    """Return an ordinary provenance label without requiring a Git checkout."""
    return _UNBOUND_CODE_VERSION

@lru_cache(maxsize=1)
def _source_version() -> str:
    """Stable source label for output traceability; never used by computation."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def _make_oof(
    states: list[_RuntimeRecord], oof_ids: tuple[str, ...], prediction_identities: Iterable[Any], probabilities: Any,
    common: dict[str, Any], *, balance_line: str, quality_weighting: bool = False, quality_weight_source: str = "none",
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
    """Build window/file/participant OOF while retaining every rejected OOF file."""
    api = _runtime_imports()
    identities = tuple(prediction_identities)
    values = api["np"].asarray(probabilities, dtype=api["np"].float64)
    if values.shape != (len(identities), 3):
        raise _ExperimentProtocolError("oof_probability_shape_mismatch")
    heldout = set(oof_ids)
    state_by_file = {str(state.row.record_id): state for state in states if str(state.row.participant_id) in heldout}
    if not state_by_file:
        raise _ExperimentProtocolError("outer_oof_has_no_selected_records")
    prediction_rows = []
    hierarchy_prediction_rows = []
    for identity, probability in zip(identities, values):
        if str(identity.participant_id) not in heldout:
            raise _ExperimentProtocolError("non_oof_prediction_reached_oof_writer")
        state = state_by_file.get(str(identity.file_id))
        if state is None:
            raise _ExperimentProtocolError(f"prediction_file_not_in_selected_oof:{identity.file_id}")
        level = "window" if identity.window_id is not None else "file"
        prediction_row = api["OofPredictionRow"](
            participant_id=str(identity.participant_id), file_id=str(identity.file_id),
            role=api["canonicalize_role_family"](str(identity.role)), label=int(identity.label), probabilities=tuple(
                (float(value) for value in probability)
            ), signal_route=str(identity.signal_route), quality_score=float(identity.quality_score), retained=True,
            level=level, window_id=identity.window_id, artifact_reducer_name=state.artifact_name,
            artifact_reducer_version=state.artifact_version, route_status=state.route_status, rejection_reason=None,
            **common,
        )
        prediction_rows.append(prediction_row)
        if bool(getattr(identity, "aggregation_retained", True)):
            hierarchy_prediction_rows.append(prediction_row)
    source_levels = {row.level for row in prediction_rows}
    if len(source_levels) > 1:
        raise _ExperimentProtocolError("mixed_window_and_file_prediction_levels")
    window_mode = source_levels == {"window"}
    if not window_mode and len(hierarchy_prediction_rows) != len(prediction_rows):
        raise _ExperimentProtocolError("aggregation_window_selection_requires_window_predictions")
    predicted_files = {row.file_id for row in prediction_rows}
    dropped_file_rows = []
    for state in sorted(state_by_file.values(), key=lambda item: str(item.row.record_id)):
        if str(state.row.record_id) in predicted_files:
            continue
        route = state.route or state.intended_route or api["SignalRoute"].DROPPED
        dropped_common = dict(common)
        dropped_common["class_order"] = ()
        dropped_file_rows.append(
            api["OofPredictionRow"](
                participant_id=str(state.row.participant_id), file_id=str(state.row.record_id),
                role=api["canonicalize_role_family"](str(state.row.role)), label=int(state.row.class_id),
                probabilities=(), signal_route=route.value, quality_score=1.0 if state.final_quality is None
                or state.final_quality.q_rate.score is None else float(state.final_quality.q_rate.score),
                retained=False, level="file", artifact_reducer_name=state.artifact_name,
                artifact_reducer_version=state.artifact_version, route_status=state.route_status,
                rejection_reason=state.reason or "no_representation_prediction", **dropped_common,
            )
        )
    source_rows = tuple((*hierarchy_prediction_rows, *dropped_file_rows))
    hierarchy = api["aggregate_hierarchy"](
        source_rows, balance_line=balance_line, quality_weighted=quality_weighting,
        quality_weight_source=quality_weight_source,
    )
    if window_mode:
        retained_file_rows = {row.file_id: row for row in hierarchy.file_rows}
        dropped_by_file = {row.file_id: row for row in dropped_file_rows}
        file_rows = tuple(
            (retained_file_rows.get(record_id, dropped_by_file.get(record_id)) for record_id in sorted(state_by_file))
        )
        if any((row is None for row in file_rows)):
            raise _ExperimentProtocolError("raw_window_to_file_oof_incomplete")
        window_rows = tuple(
            sorted(prediction_rows, key=lambda row: (row.participant_id, row.file_id, str(row.window_id)))
        )
    else:
        duplicate_files = [row.file_id for row in prediction_rows]
        if len(duplicate_files) != len(set(duplicate_files)):
            raise _ExperimentProtocolError("duplicate_file_level_prediction")
        file_rows = tuple(sorted(source_rows, key=lambda row: (row.participant_id, row.file_id)))
        window_rows = ()
    role_rows = tuple(hierarchy.role_rows)
    if balance_line == "line_b_equal_role_families":
        api["validate_role_level_oof"](role_rows)
    elif role_rows:
        raise _ExperimentProtocolError("equal_files_ablation_must_not_emit_role_level_oof")
    subject_rows = list(hierarchy.participant_rows)
    observed = {row.participant_id for row in subject_rows}
    for participant in sorted(heldout - observed):
        candidates = [row for row in file_rows if row.participant_id == participant]
        if not candidates:
            raise _ExperimentProtocolError(f"heldout_subject_has_no_record:{participant}")
        routes = {row.signal_route for row in candidates}
        if len(routes) != 1:
            raise _ExperimentProtocolError(f"dropped_subject_mixed_routes:{participant}")
        subject_rows.append(
            api["replace"](
                candidates[0], file_id=f"participant::{participant}", role="participant", probabilities=(),
                retained=False, level="participant", class_order=(), window_id=None,
                route_status="dropped_no_participant_probability", rejection_reason="all_selected_files_dropped",
            )
        )
    return (window_rows, tuple(file_rows), role_rows, tuple(sorted(subject_rows, key=lambda row: row.participant_id)))

def _evaluate_subjects(subject_rows: tuple[Any, ...], total: int) -> dict[str, Any]:
    """Compute conditional and abstention-aware participant metrics."""
    api = _runtime_imports()
    retained = [row for row in subject_rows if row.retained]
    dropped = [row for row in subject_rows if not row.retained]
    if len(retained) + len(dropped) != total:
        raise _ExperimentProtocolError("participant_metric_roster_does_not_match_outer_fold")
    probability = (
        api["np"].asarray([row.probabilities for row in retained], dtype=api["np"].float64).reshape((len(retained), 3))
    )
    aware = api["evaluate_predictions_with_abstentions"](
        api["np"].asarray([row.label for row in retained]), probability,
        api["np"].asarray([row.label for row in dropped]), class_order=(0, 1, 2),
    )
    conditional = aware.conditional_metrics
    payload = ({
        "status": "unavailable_no_retained_participant", "n_rows": 0, "n_total": total, "n_retained": 0,
        "n_dropped": total, "coverage_rate": 0.0, "balanced_accuracy": None, "macro_f1": None,
        "multiclass_log_loss": None, "multiclass_brier": None, "expected_calibration_error": None,
        "confusion_matrix": ((0, 0, 0), (0, 0, 0), (0, 0, 0)), "class_order": (0, 1, 2), "per_class": (),
        "worst_class_label": None, "worst_class_precision": None, "worst_class_recall": None, "worst_class_f1": None,
    } if conditional is None else asdict(conditional))
    payload.update({
        "abstention_aware_balanced_accuracy": aware.balanced_accuracy,
        "abstention_aware_macro_precision": aware.macro_precision, "abstention_aware_macro_recall": aware.macro_recall,
        "abstention_aware_macro_f1": aware.macro_f1, "abstention_count": aware.n_abstained,
        "abstention_counts_by_class": aware.abstention_counts_by_class, "abstention_aware_per_class": aware.per_class,
        "abstention_probability_metrics_scope": aware.probability_metrics_scope,
    })
    return to_strict_json_value(payload)

def _batch1_operational_model_input(dataset: Any, *, mode: str, estimator: bool) -> Any:
    """Materialize one already-preprocessed model input without timing preprocessing."""
    if estimator:
        values = dataset.values[:1]
        if hasattr(dataset, "row_mask"):
            return {"x": values, "mask": dataset.row_mask[:1]}
        if hasattr(dataset, "sample_mask"):
            return {"x": values, "mask": dataset.sample_mask[:1]}
        return values
    if mode == "fusion":
        import torch

        bag = torch.from_numpy(dataset.window_bags[0]).unsqueeze(0)
        sample_mask = torch.from_numpy(dataset.sample_masks[0]).unsqueeze(0)
        return {
            "window_bag": bag, "window_mask": torch.ones((1, bag.shape[1]), dtype=torch.bool),
            "file_features": torch.from_numpy(dataset.file_features[:1]), "sample_mask": sample_mask,
        }
    sample = dataset[0]
    output = {"x": sample["x"].unsqueeze(0)}
    if "mask" in sample:
        output["mask"] = sample["mask"].unsqueeze(0)
    return output

def _assert_legacy_bridge_entrypoint_contract(
    config: Any, legacy_bridge: _LegacyBridgeExecution | None, *, dedicated_entrypoint: bool
) -> None:
    """Compatibility hook: resolved configs may use either CLI entry point."""
    del config, legacy_bridge, dedicated_entrypoint

def _cell_trace(**ctx: Any) -> tuple[dict[str, Any], dict[str, Any], str, str, Mapping[str, Any], Any, str, str]:
    """Build descriptive OOF trace fields after all numerical work is complete."""
    api, config, report = (ctx["api"], ctx["config"], ctx["report"])
    mode, bridge, legacy = (ctx["mode"], ctx["bridge_profile"], ctx["legacy_bridge"])
    representation, quality = (ctx["representation"], ctx["quality"])
    dataset, spec, training = (ctx["train_dataset"], ctx["input_spec"], ctx["training"])
    model_id, model, split, seed = (ctx["model_id"], ctx["model_section"], ctx["split"], ctx["seed"])
    members, ensemble, balance = (ctx["member_seeds"], ctx["ensemble"], ctx["balance_line"])
    registry = api["registry_for_groups"](config.section("features")["enabled_groups"])
    if mode == "feature_vector":
        schema = registry.schema_version
        feature_identity = {"registry_hash": registry.sha256, "feature_names": tuple(dataset.feature_names)}
    else:
        channels = tuple(dataset.channel_schema) if mode == "feature_matrix" else tuple(spec.channel_schema)
        schema = (
            str(_retained_states(ctx["states"], split["train_participant_ids"])[0].matrix.schema_version)
            if mode == "feature_matrix" else f"fusion_raw8_vector_registry-{registry.sha256[:12]}_v3"
            if mode == "fusion" else f"legacy_bridge_{bridge.profile_id.lower()}_raw8_v1"
            if bridge is not None else "raw_red_ir_imu_axes_8ch_outer_train_scaled_v2"
        )
        feature_identity = {"channel_schema": channels, "transforms": representation}
    preprocessing_hash = api["stable_payload_sha256"]({
        "signal": config.section("signal"), "quality": config.section("quality"),
        "artifact": config.section("artifact"), "quality_provenance": quality,
        "representation_transforms": representation,
    })
    feature_hash = api["stable_payload_sha256"](feature_identity)
    algorithm = ({
        "loss": "cross_entropy", "class_weighting": bridge.class_weighting, "sampler": bridge.sampler,
        "epoch_rule": {"rule": "fixed_epoch", "fixed_epochs": 10}, "optimizer": bridge.optimizer,
        "learning_rate": float(bridge.learning_rate), "weight_decay": float(bridge.weight_decay),
    } if bridge is not None else _training_algorithm_provenance(
        model_id, config.section("training"), model, fixed_epochs=int(ctx["training_config"].fixed_epochs)
    ))
    selected_window = ({
        "length_s": float(bridge.window_seconds), "hop_s": float(bridge.hop_seconds),
        "padding": "none_complete_windows_only" if not bridge.resolved_allow_short_record_padding else
        "zero_right_only_if_source_shorter_than_one_window", "min_valid_fraction": 1.0 if
        not bridge.resolved_allow_short_record_padding else "all_available_source_rows_valid_before_optional_padding",
    } if bridge is not None else config.section("windows")["engineering" if mode in
                                                           {"feature_vector", "feature_matrix"} else "raw_dl"])
    frozen = {
        "architecture_parameters": _resolved_architecture_parameters(model_id, model)
        if bridge is not None else dict(ctx["prepared"].resolved_model_config["architecture_parameters"]),
        "input_channels_order": tuple(dataset.feature_names)
        if mode == "feature_vector" else tuple(spec.channel_schema),
        "sampling_rate_hz": float(bridge.target_fs_hz) if bridge is not None else _model_input_sampling_rate_hz(config),
        "window_plan": {"representation_mode": mode, "selected": selected_window}, "feature_schema_hash": feature_hash,
        "training_algorithm": algorithm, "random_seeds": members if ensemble else (seed, ),
        "seed_policy": "legacy_bridge_fixed_training_seed_42" if bridge is not None else str(model["seed_policy"]),
        "fold_hash": report.fold_hash,
        "aggregation": {"balance_line": balance} if bridge is not None else config.section("aggregation"),
    }
    dropout = _resolved_legacy_bridge_dropout_comparison(bridge, model)
    if bridge is not None:
        frozen["legacy_bridge"] = {
            "profile": bridge.to_dict(), "manifest_sha256": legacy.manifest_sha256, "split_sha256": legacy.split_sha256,
            "effective_config_hash": legacy.effective_config_hash,
        }
    source = _source_version()
    common = {
        "repeat": int(split["repeat_index"]), "fold": int(split["fold_index"]), "split_seed": int(split["split_seed"]),
        "training_seed": seed, "config_hash": legacy.effective_config_hash if legacy else config.sha256,
        "manifest_hash": report.manifest_hash, "fold_hash": report.fold_hash, "preprocessing_hash": preprocessing_hash,
        "feature_hash": feature_hash, "model_hash": training.provenance.state_hash, "representation_mode": mode,
        "class_order": (0, 1, 2), "code_commit": _code_version(),
        "data_schema_id": str(config.section("manifest")["manifest_version"]
                              ), "feature_schema_id": schema, "model_version": str(model["variant"]),
        "aggregation_rule": balance, "environment_hash": api["stable_payload_sha256"](api["runtime_environment"]()),
        "manifest_version": str(config.section("manifest")["manifest_version"]),
        "fold_registry_version": str(config.section("splits")["registry_id"]), "source_snapshot_hash": source,
    }
    return (
        common, frozen, api["stable_payload_sha256"](frozen), source, config.section("evaluation"), dropout,
        preprocessing_hash, feature_hash,
    )

def _execute_cell_unchecked(
    report: Any, config: Any, rows: Iterable[Any], registry: Any, paths: Any, *, repeat_index: int, fold_index: int,
    maximum_seconds: float | None, record_cap: int | None, epoch_override: int | None,
    measure_operational_costs: bool = False, loader: Any = None, legacy_bridge: _LegacyBridgeExecution | None = None,
    preprocessing_cache: Mapping[str, Any] | None = None,
) -> _CellResult:
    """Execute one representation-aware frozen outer cell."""
    _assert_legacy_bridge_entrypoint_contract(config, legacy_bridge, dedicated_entrypoint=False)
    api = _runtime_imports()
    from .data.preprocessing_cache import PreprocessingCacheSession

    cache_session = PreprocessingCacheSession.from_mapping(preprocessing_cache, paths)
    if legacy_bridge is not None and cache_session.enabled:
        raise _ExperimentProtocolError("legacy_bridge_preprocessing_cache_must_be_disabled")
    started = time.perf_counter()
    mode = str(config.representation_mode)
    if mode not in {"raw", "feature_vector", "feature_matrix", "fusion"}:
        raise _ExperimentProtocolError(f"unsupported_representation_mode:{mode}")
    bridge_profile = None if legacy_bridge is None else legacy_bridge.profile
    if bridge_profile is not None and mode != "raw":
        raise _ExperimentProtocolError("legacy_bridge_requires_raw_representation")
    split = registry.get_split(repeat_index, fold_index)
    train_ids = tuple((str(value) for value in split["train_participant_ids"]))
    oof_ids = tuple((str(value) for value in split["oof_participant_ids"]))
    row_values = tuple(rows)
    selected = _choose_records(row_values, (*train_ids, *oof_ids), _classifier_role_ids(config), record_cap)
    states = [_RuntimeRecord(row=row) for row in selected]
    actual_loader = loader or (lambda row, maximum: api["_load_record"](row, paths, max_samples=maximum))
    if bridge_profile is not None:
        quality_mode = _quality_mode(config)
        if quality_mode != "off":
            raise _ExperimentProtocolError("legacy_bridge_requires_quality_mode_off")
        if bridge_profile.builds_windows_from_raw_record:
            quality_provenance = _preprocess_legacy_bridge_records(
                states, bridge_profile, maximum_seconds, actual_loader, config=config, calibration_rows=row_values
            )
        else:
            _preprocess_records(
                states, config, maximum_seconds, actual_loader, calibration_rows=row_values, cache_session=cache_session
            )
    else:
        _preprocess_records(
            states, config, maximum_seconds, actual_loader, calibration_rows=row_values, cache_session=cache_session
        )
        quality_mode = _quality_mode(config)
    if bridge_profile is not None and bridge_profile.builds_windows_from_raw_record:
        pass
    else:
        quality_provenance = _apply_quality_motion_routing(
            states, config, report, paths, train_ids=train_ids, oof_ids=oof_ids, cache_session=cache_session
        )
    for state in states:
        if mode in {"feature_vector", "fusion"}:
            _extract_vector(state, report, config.section("features"))
        elif mode == "feature_matrix":
            _extract_matrix_features(state, report)
        if mode in {"raw", "fusion"}:
            if bridge_profile is not None and bridge_profile.builds_windows_from_raw_record:
                continue
            if bridge_profile is not None and bridge_profile.uses_canonical_all_channel_window_scaling:
                _extract_canonical_all_channel_bridge_raw(state, bridge_profile)
            else:
                _extract_raw(state, report, config.section("signal"), cache_session=cache_session)
    window_selection_provenance = (
        _apply_window_quality_selection(states, config, train_ids=train_ids, oof_ids=oof_ids)
        if bridge_profile is None and mode in {"raw", "fusion"} else None
    )
    if window_selection_provenance is not None and window_selection_provenance["policy"] != "none":
        quality_provenance = {
            **quality_provenance, "classification_effect": str(quality_provenance["classification_effect"]) +
            "+legacy_per_file_window_selection", "window_selection": window_selection_provenance,
        }
    required_by_mode = {
        "raw": ("raw_windows", ), "feature_vector": ("vector", ), "feature_matrix": ("engineering", ),
        "fusion": ("vector", "engineering", "raw_windows"),
    }
    _assert_train_payload_roster(states, train_ids, required=required_by_mode[mode])
    representation_provenance = (
        _legacy_bridge_representation_artifacts(states, bridge_profile, train_ids, oof_ids)
        if bridge_profile is not None else _fit_representation_artifacts(states, mode, train_ids, oof_ids)
    )
    if window_selection_provenance is not None:
        representation_provenance["window_quality_selection"] = window_selection_provenance
    quality_weight_source = config.section("aggregation").get("quality_weight_source", "none")
    train_dataset = _materialize_representation_dataset(
        states, train_ids, mode, quality_weight_source=quality_weight_source
    )
    oof_dataset = _materialize_representation_dataset(
        states, oof_ids, mode, quality_weight_source=quality_weight_source
    )
    if train_dataset is None:
        raise _ExperimentProtocolError("outer_train_dataset_empty")
    dl_config = ({"enabled": False, "case_id": None, "target_fs_hz": 400.0}
                 if bridge_profile is not None else config.section("signal")["dl_resampling"])
    dl_case_id = dl_config.get("case_id")
    if bool(dl_config["enabled"]) or dl_case_id is not None:
        if oof_dataset is None:
            raise _ExperimentProtocolError("dl_resampling_requires_nonempty_train_and_oof_datasets")
        train_dataset, provenance_key, train_profile = _prepare_dl_input_dataset(train_dataset, mode, dl_config)
        oof_dataset, oof_provenance_key, oof_profile = _prepare_dl_input_dataset(oof_dataset, mode, dl_config)
        if provenance_key != oof_provenance_key:
            raise _ExperimentProtocolError("dl_resampling_train_oof_kind_drift")
        if train_profile != oof_profile:
            raise _ExperimentProtocolError("dl_resampling_train_oof_profile_drift")
        if provenance_key is None or train_profile is None:
            raise _ExperimentProtocolError("dl_resampling_profile_missing")
        representation_provenance[provenance_key] = {
            **train_profile, "application_scope": "dl_input_only_after_canonical_400hz_windows",
            "canonical_features_and_peaks_unchanged": True, "outer_train_and_oof_same_transform_identity": True,
        }
    observed_train = set((str(value) for value in train_dataset.participant_ids))
    if observed_train != set(train_ids):
        missing = sorted(set(train_ids) - observed_train)
        raise _ExperimentProtocolError("outer_train_dataset_roster_incomplete:" + ",".join(missing))
    observed_classes = set((int(value) for value in api["np"].asarray(train_dataset.labels)))
    if observed_classes != {0, 1, 2}:
        raise _ExperimentProtocolError(
            "outer_train_dataset_missing_class:" +
            ",".join((str(value) for value in sorted({0, 1, 2} - observed_classes)))
        )
    frozen = api["FrozenOuterSplit"](
        repeat=int(split["repeat_index"]), fold=int(split["fold_index"]), seed=int(split["split_seed"]),
        train_participant_ids=train_ids, oof_participant_ids=oof_ids,
        registry_hash=str(config.section("splits")["source_registry_payload_sha256"]), fold_hash=report.fold_hash,
    )
    effective_training_seed = 42 if bridge_profile is not None else _outer_cv_model_training_seed(config, split)
    if bridge_profile is not None:
        training_config = bridge_profile.training_config(device=str(config.section("training")["device"]))
    else:
        training_config = api["TrainingConfig"].from_mapping(config.section("training"))
        training_config = api["replace"](training_config, seed=effective_training_seed)
    if epoch_override is not None:
        if bridge_profile is not None:
            raise _ExperimentProtocolError("legacy_bridge_fixed10_does_not_accept_epoch_override")
        if training_config.epoch_rule != "fixed_epoch" or epoch_override <= 0:
            raise _ExperimentProtocolError("invalid_fixed_epoch_override")
        training_config = training_config._with_epoch_override(int(epoch_override))
    balance_line = (
        str(bridge_profile.expected_aggregation_rule)
        if bridge_profile is not None else str(config.section("aggregation")["balance_line"])
    )
    if bridge_profile is not None and training_config.expected_aggregation_rule != balance_line:
        raise _ExperimentProtocolError("legacy_bridge_training_and_aggregation_balance_line_mismatch")
    model_config, model_id = _resolved_model_config(config, training_seed=effective_training_seed)
    model_section = config.section("model")
    if bridge_profile is not None:
        allowed_bridge_models = _legacy_bridge_allowed_model_ids(bridge_profile)
        if model_id not in allowed_bridge_models:
            raise _ExperimentProtocolError(
                f"legacy_bridge_profile_model_mismatch:{bridge_profile.profile_id}:{model_id}"
            )
        input_binding = {
            "status": "legacy_bridge_ordered_raw_8_identity", "model_id": model_id,
            "source_channel_schema": tuple(bridge_profile.channel_schema),
            "target_channel_schema": tuple(bridge_profile.channel_schema), "source_indices": tuple(range(8)),
            "silent_channel_slicing": False,
        }
        input_binding["binding_sha256"] = api["stable_payload_sha256"](input_binding)
        representation_provenance["model_input_binding"] = input_binding
    elif mode == "raw":
        train_dataset, input_binding = _bind_raw_dataset_for_model(
            train_dataset, model_id, declared_channel_order=model_section.get("input_channel_order")
        )
        if oof_dataset is not None:
            oof_dataset, oof_binding = _bind_raw_dataset_for_model(
                oof_dataset, model_id, declared_channel_order=model_section.get("input_channel_order")
            )
            if oof_binding != input_binding:
                raise _ExperimentProtocolError("frailty_train_oof_input_binding_identity_drift")
        representation_provenance["model_input_binding"] = input_binding
    input_spec = (
        api["ModelInputSpec"](
            "raw", n_channels=int(train_dataset.values.shape[1]), n_classes=3,
            channel_schema=tuple(bridge_profile.channel_schema),
        ) if bridge_profile is not None else _model_input_spec(train_dataset, mode)
    )
    if int(model_section.get("n_classes", 3)) != 3:
        raise _ExperimentProtocolError("model_declared_class_count_must_be_three")
    declared_channels = int(model_section.get("input_channels", 0))
    if declared_channels > 0 and declared_channels != int(input_spec.n_channels):
        raise _ExperimentProtocolError(f"model_input_channel_mismatch:{declared_channels}:{int(input_spec.n_channels)}")
    is_ensemble = _model_is_ensemble(model_id)
    if bridge_profile is not None and is_ensemble:
        raise _ExperimentProtocolError("legacy_bridge_does_not_accept_ensemble_model")
    ensemble_member_seeds = _ensemble_member_seed_roster(model_section) if is_ensemble else ()
    prepared = (
        _prepare_legacy_bridge_model_factory(
            api, model_config, input_spec, train_dataset, frozen, profile=bridge_profile
        ) if bridge_profile is not None else
        api["prepare_model_factory"](model_config, input_spec, train_dataset, frozen)
    )
    if bridge_profile is not None:
        from .training.legacy_bridge import LegacyBridgeTrainer

        trainer = LegacyBridgeTrainer(training_config)
    else:
        trainer = api["UnifiedTrainer"](training_config)
    inner_split = None
    if bridge_profile is None and training_config.epoch_rule == "inner_grouped_selection":
        inner_split = api["build_inner_grouped_split"](
            train_dataset, frozen, n_folds=int(training_config.inner_grouped_folds), seed=int(training_config.seed)
        )
    uses_estimator = _model_uses_estimator(model_id)
    if uses_estimator:
        training = trainer.fit_estimator(prepared(), train_dataset, frozen)
        if oof_dataset is None:
            probabilities = api["np"].empty((0, 3), dtype=api["np"].float64)
            prediction_identities: tuple[Any, ...] = ()
        else:
            probabilities, _, prediction_identities = trainer.predict_estimator_probabilities(
                training.model, oof_dataset
            )
            classes = tuple((int(value) for value in training.model.classes_))
            if set(classes) != {0, 1, 2}:
                raise _ExperimentProtocolError(f"trained_model_missing_class:{classes}")
            if classes != (0, 1, 2):
                probabilities = probabilities[:, [classes.index(value) for value in (0, 1, 2)]]
    else:
        training = trainer.fit(prepared, train_dataset, frozen, inner_split=inner_split)
        if oof_dataset is None:
            probabilities = api["np"].empty((0, 3), dtype=api["np"].float64)
            member_probabilities = None
            prediction_identities = ()
        elif is_ensemble:
            member_probabilities, probabilities, _, prediction_identities = trainer.predict_ensemble_members(
                training.model, oof_dataset
            )
        else:
            member_probabilities = None
            probabilities, _, prediction_identities = trainer.predict_probabilities(training.model, oof_dataset)
    if training.provenance is None:
        raise _ExperimentProtocolError("training_provenance_missing")
    if is_ensemble:
        trained_roster = tuple((int(value) for value in getattr(training.model, "member_seeds", ())))
        provenance_roster = tuple(training.provenance.member_training_seeds)
        if trained_roster != ensemble_member_seeds or provenance_roster != trained_roster:
            raise _ExperimentProtocolError("ensemble_config_model_and_training_provenance_roster_drift")
        if member_probabilities is not None and int(member_probabilities.shape[0]) != len(ensemble_member_seeds):
            raise _ExperimentProtocolError("ensemble_member_probability_count_differs_from_roster")
    if measure_operational_costs:
        if oof_dataset is None or len(oof_dataset) == 0:
            raise _ExperimentProtocolError("operational_measurement_requires_nonempty_oof_model_input")
        model_input = _batch1_operational_model_input(oof_dataset, mode=mode, estimator=uses_estimator)
        operational_model = training.model
        if not uses_estimator:
            import torch

            operational_model = operational_model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        operational_metrics = {
            "status": "measured_explicit_cpu_batch1_request", **to_strict_json_value(
                asdict(api["measure_cpu_batch1_operational_metrics"](operational_model, model_input))
            ),
        }
    else:
        operational_metrics = {
            "status": "not_requested", "parameter_count": None,
            "inference_cost": {"cpu_batch1_model_only_p50_ms": None, "cpu_batch1_model_only_p95_ms": None},
        }
    (
        common, frozen_run_provenance, frozen_run_provenance_hash, source_version, evaluation_section,
        resolved_dropout_comparison, preprocessing_hash, feature_hash,
    ) = _cell_trace(
        api=api, config=config, report=report, states=states, mode=mode, bridge_profile=bridge_profile,
        legacy_bridge=legacy_bridge, representation=representation_provenance, quality=quality_provenance,
        prepared=prepared, train_dataset=train_dataset, input_spec=input_spec, training_config=training_config,
        training=training, model_id=model_id, model_section=model_section, split=split, seed=effective_training_seed,
        member_seeds=ensemble_member_seeds, ensemble=is_ensemble, balance_line=balance_line,
    )
    member_rows: tuple[Any, ...] = ()
    if is_ensemble:
        if member_probabilities is None:
            raise _ExperimentProtocolError("ensemble_member_probabilities_missing")
        member_seeds = ensemble_member_seeds
        base_model_id = _ENSEMBLE_BASE_MODEL_IDS[model_id]
        average_common = dict(common)
        average_common.update({
            "training_seed": None, "prediction_kind": "ensemble_average", "member_training_seeds": member_seeds,
            "ensemble_base_model_id": base_model_id,
        })
        window_rows, file_rows, role_rows, subject_rows = _make_oof(
            states, oof_ids, prediction_identities, probabilities, average_common, balance_line=balance_line,
            quality_weighting=bool(config.section("aggregation")["quality_weighting"]),
            quality_weight_source=str(config.section("aggregation")["quality_weight_source"]),
        )
        member_subject_rows: list[Any] = []
        member_hashes = tuple(training.provenance.member_state_hashes)
        if len(member_hashes) != len(member_seeds):
            raise _ExperimentProtocolError("ensemble_member_provenance_incomplete")
        for member_index, (member_seed, member_hash) in enumerate(zip(member_seeds, member_hashes)):
            member_common = dict(common)
            member_common.update({
                "training_seed": member_seed, "member_index": member_index, "prediction_kind": "ensemble_member",
                "ensemble_base_model_id": base_model_id, "model_hash": str(member_hash),
            })
            _, _, _, current_subject_rows = _make_oof(
                states, oof_ids, prediction_identities, member_probabilities[member_index], member_common,
                balance_line=balance_line, quality_weighting=bool(config.section("aggregation")["quality_weighting"]),
                quality_weight_source=str(config.section("aggregation")["quality_weight_source"]),
            )
            member_subject_rows.extend(current_subject_rows)
        member_rows = tuple(member_subject_rows)
    else:
        window_rows, file_rows, role_rows, subject_rows = _make_oof(
            states, oof_ids, prediction_identities, probabilities, common, balance_line=balance_line,
            quality_weighting=bool(config.section("aggregation")["quality_weighting"]),
            quality_weight_source=str(config.section("aggregation")["quality_weight_source"]),
        )
    bridge_window_probability_sha256 = (
        api["stable_payload_sha256"]([{
            "participant_id": row.participant_id, "file_id": row.file_id, "role": row.role, "label": int(row.label),
            "window_id": row.window_id, "probabilities": tuple((float(value) for value in row.probabilities)),
        } for row in sorted(
            window_rows, key=lambda value: (value.participant_id, value.file_id, value.role, str(value.window_id)),
        )]) if bridge_profile is not None else None
    )
    api["validate_expected_oof_roster"](
        (*member_rows, *subject_rows),
        {(int(split["repeat_index"]), int(split["fold_index"]), int(split["split_seed"])): oof_ids},
        expected_config_hashes=(common["config_hash"], ),
        expected_member_count=len(ensemble_member_seeds) if is_ensemble else 1, expect_ensemble=is_ensemble,
    )
    metrics = _evaluate_subjects(subject_rows, len(oof_ids))
    archived_training_history: list[dict[str, Any]] = []
    sampling_diagnostic_rows: list[dict[str, Any]] = []
    for raw_history_row in training.history:
        history_row = dict(raw_history_row)
        diagnostics = history_row.pop("sampling_diagnostics", None)
        if diagnostics is not None:
            sampling_diagnostic_rows.append({
                "epoch": int(history_row["epoch"]), "member": int(history_row.get("member", 0)),
                "training_seed": int(history_row.get("training_seed", effective_training_seed)), **dict(diagnostics),
            })
        archived_training_history.append(history_row)
    summary = {
        "schema_version": "ppg_frailty.experiment_cell.v2", "pipeline_generation": "final_pipeline_v2",
        "status": "passed", "runner_status": "outer_fold_execution_completed",
        "repeat_index": int(split["repeat_index"]), "fold_index": int(split["fold_index"]),
        "split_seed": int(split["split_seed"]), "training_seed": None if is_ensemble else effective_training_seed,
        "training_orchestration_seed": int(effective_training_seed), "config_id": str(config.config_id),
        "config_hash": common["config_hash"], "canonical_config_hash": config.sha256,
        "member_training_seeds": list(ensemble_member_seeds) if is_ensemble else [],
        "seed_policy": str(model_section.get("seed_policy", "member_roster")) if is_ensemble else
        "legacy_bridge_fixed_training_seed_42" if bridge_profile is not None else str(model_section["seed_policy"]),
        "representation_mode": mode, "model_id": str(model_section["model_id"]), "model_machine_id": model_id,
        "class_order": [0, 1, 2], "selected_record_count": len(states), "retained_train_record_count": sum(
            (state.retained and state.row.participant_id in set(train_ids) for state in states)
        ), "retained_oof_record_count": sum(
            (state.retained and state.row.participant_id in set(oof_ids) for state in states)
        ), "oof_window_prediction_count": len(window_rows), "dropped_records": [{
            "record_id": state.row.record_id, "participant_id": state.row.participant_id, "reason": state.reason,
            "route_status": state.route_status,
        } for state in states if not state.retained], "metrics": metrics, "operational_metrics": operational_metrics,
        "training_history": to_strict_json_value(archived_training_history), **({
            "sampling_diagnostics": to_strict_json_value(sampling_diagnostic_rows), "legacy_bridge": {
                "schema_version": "ppg_frailty.legacy_bridge_execution.v1"
                if legacy_bridge.protocol_design == "cumulative_chain_v1"
                else "ppg_frailty.legacy_bridge_execution.v2",
                "profile": bridge_profile.to_dict(), "source_specification": legacy_bridge.source_specification,
                "source_specification_sha256": legacy_bridge.source_specification_sha256,
                "manifest_sha256": legacy_bridge.manifest_sha256, "split_sha256": legacy_bridge.split_sha256,
                "canonical_config_hash": config.sha256, "effective_config_hash": legacy_bridge.effective_config_hash,
                "fresh_current_raw_csv_training_input": True, "historical_cache_used_for_training": False, **({
                    "protocol_design": legacy_bridge.protocol_design,
                    "profile_definition_sha256": legacy_bridge.profile_definition_sha256,
                    "training_identity_sha256": bridge_profile.training_identity_sha256,
                    "primary_report_aggregation_view": bridge_profile.resolved_primary_report_aggregation_view,
                    "train_dataset_binding_hash": training.provenance.dataset_binding_hash,
                    "oof_window_probability_sha256": bridge_window_probability_sha256,
                } if legacy_bridge.protocol_design != "cumulative_chain_v1" else {}),
                "resolved_dropout_comparison": resolved_dropout_comparison,
            },
        } if bridge_profile is not None else {}), "learning_curve_contract": {
            "status": "not_applicable_non_iterative_estimator"
            if not training.history and uses_estimator else "outer_train_loss_and_participant_ba_fixed_epoch" if any(
                ("training_participant_balanced_accuracy" in row for row in training.history)
            ) else "outer_train_loss_only_fixed_epoch", "training_data_scope": "full_outer_train_only",
            "outer_heldout_used_for_epoch_selection_or_curve": False,
            "training_metric": "training_participant_balanced_accuracy" if any(
                ("training_participant_balanced_accuracy" in row for row in training.history)
            ) else "not_available", "training_metric_unit": "participant",
            "training_metric_aggregation_rule": training.provenance.expected_aggregation_rule,
            "training_metric_used_for_epoch_selection_or_checkpoint": False,
            "validation_metric": "inner_participant_balanced_accuracy" if any(
                ("inner_participant_balanced_accuracy" in row for row in training.history)
            ) else "not_applicable_fixed_epoch_no_inner_validation", "selected_epoch": training.selected_epoch,
        }, "fitted_provenance": to_strict_json_value(asdict(training.provenance)),
        "model_factory_provenance": to_strict_json_value(dict(prepared.provenance)),
        "frozen_model_run_provenance": to_strict_json_value(frozen_run_provenance),
        "frozen_model_run_provenance_hash": frozen_run_provenance_hash,
        "representation_transform_provenance": to_strict_json_value(representation_provenance),
        "quality_mode": quality_mode, "evaluation_policy": to_strict_json_value(evaluation_section),
        "sqi_calibrator_provenance": quality_provenance, "quality_diagnostics": [
            {
                "record_id": state.row.record_id, "participant_id": state.row.participant_id, "role": state.row.role,
                "retained": state.retained, "route_status": state.route_status,
                "signal_route": state.route.value if state.route is not None else None,
                "artifact_reducer_name": state.artifact_name, "artifact_reducer_version": state.artifact_version,
                "components": state.diagnostic_components, "diagnostic_reason": state.diagnostic_reason,
                "rejection_reason": state.reason,
            } for state in states if state.diagnostic_components or state.diagnostic_reason is not None
        ], "route_artifacts": [
            _persisted_route_artifact_row(state, train_participant_ids=train_ids, oof_participant_ids=oof_ids)
            for state in states
        ], "route_window_sqi_evidence": [{
            "record_id": str(state.row.record_id), "participant_id": str(state.row.participant_id),
            "role": str(state.row.role), "windows": state.route_window_sqi_evidence,
        } for state in states if state.route_window_sqi_evidence], "physical_recording_qc": [{
            "record_id": state.row.record_id, "evidence": state.physical_qc_evidence,
            "profile": state.physical_qc_profile,
        } for state in states], "preprocessing_cache": cache_session.audit_payload(), "balance_line": balance_line,
        "preprocessing_hash": preprocessing_hash, "feature_hash": feature_hash,
        "model_hash": training.provenance.state_hash, "code_commit": common["code_commit"],
        "source_version": source_version, "elapsed_seconds": time.perf_counter() - started,
    }
    checkpoint_factory = getattr(prepared, "canonical_factory", prepared)
    from .v5.checkpoints import FoldCheckpointPayload

    checkpoint = FoldCheckpointPayload(
        model=training.model, model_config=dict(checkpoint_factory.resolved_model_config),
        input_spec=checkpoint_factory.input_spec, golden_inputs=_golden_model_inputs(train_dataset, mode),
        pipeline_config=config.to_dict(), cell_summary=summary,
    )
    return _CellResult(
        summary, file_rows, subject_rows, window_rows=window_rows, role_rows=role_rows, member_rows=member_rows,
        checkpoint=checkpoint,
    )

def _execute_cell(*args: Any, **kwargs: Any) -> _CellResult:
    """Fail closed with a representation-specific machine-readable reason."""
    config = args[1] if len(args) > 1 else kwargs.get("config")
    mode = str(getattr(config, "representation_mode", "unknown"))
    try:
        return _execute_cell_unchecked(*args, **kwargs)
    except _ExperimentProtocolError:
        raise
    except Exception as exc:
        raise _ExperimentProtocolError(f"{mode}_outer_cell_failed:{type(exc).__name__}:{exc}") from exc

def _execute_vector_cell(*args: Any, **kwargs: Any) -> _CellResult:
    """Backward-compatible strict feature-vector helper for existing callers."""
    config = args[1] if len(args) > 1 else kwargs.get("config")
    if str(getattr(config, "representation_mode", "")) != "feature_vector":
        raise _ExperimentProtocolError("execute_vector_cell_requires_feature_vector")
    return _execute_cell(*args, **kwargs)

def _strict_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write strict, deterministic JSON without silently accepting NaN/Infinity."""
    from .provenance import atomic_write_json

    atomic_write_json(path, dict(payload), root=path.parent)

def _write_empty_oof(path: Path, reason: str) -> None:
    """Materialize a schema-bearing empty parquet for a deliberately absent level."""
    try:
        from .training import write_empty_oof_parquet

        write_empty_oof_parquet(path, reason)
    except (ImportError, RuntimeError) as exc:
        if "pyarrow" in str(exc).lower():
            raise _ExperimentProtocolError("pyarrow_required_for_formal_oof") from exc
        raise

def _write_route_window_sqi_evidence(directory: Path, summary: Mapping[str, Any]) -> tuple[int, str]:
    """Stream complete component-level SQI evidence to deterministic gzip JSONL."""
    target = directory / "route_window_sqi_evidence.jsonl.gz"
    row_count = 0
    header = {
        "schema_version": "ppg_frailty.route_window_sqi_evidence.v1", "record_type": "header",
        "repeat_index": int(summary["repeat_index"]), "fold_index": int(summary["fold_index"]),
        "config_hash": str(summary.get("config_hash", "")),
        "payload_contract": "one_full_component_evidence_object_per_routing_window", "report_consumed": False,
    }
    with target.open("xb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
        stream = io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")
        json.dump(header, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        for record in summary.get("route_window_sqi_evidence", ()):
            windows = record.get("windows", {}) if isinstance(record, Mapping) else {}
            identity = {name: str(record.get(name, "")) for name in ("record_id", "participant_id", "role")}
            for window_id, evidence in windows.items():
                row = {
                    "schema_version": header["schema_version"], "record_type": "window_evidence",
                    **identity, "routing_window_id": str(window_id), "evidence": evidence,
                }
                json.dump(row, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
                stream.write("\n")
                row_count += 1
        stream.flush()
        stream.detach()
    return (row_count, _stream_file_sha256(target))

def _stream_file_sha256(path: str | Path) -> str:
    """Hash an artifact without materializing its complete byte payload."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _release_cell_memory() -> None:
    """Release host cycles and CUDA allocator blocks after one persisted fold."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

def _cell_for_root_aggregation(cell: _CellResult, *, artifact_prefix: str) -> _CellResult:
    """Drop high-cardinality cell payloads after their authoritative write."""
    compact_summary = _artifact_index_cell_summary(cell.summary, artifact_prefix=artifact_prefix)
    compact_summary["quality_diagnostics"] = list(cell.summary.get("quality_diagnostics", ()))
    return _CellResult(
        summary=compact_summary, file_rows=cell.file_rows, subject_rows=cell.subject_rows, window_rows=cell.window_rows,
        role_rows=cell.role_rows, member_rows=cell.member_rows,
    )

def _artifact_index_cell_summary(summary: Mapping[str, Any], *, artifact_prefix: str = "") -> dict[str, Any]:
    """Replace high-cardinality inline tables with their artifact pointers."""
    compact = dict(summary)
    prefix = str(artifact_prefix).strip("/")
    locate = lambda name: f"{prefix}/{name}" if prefix else name
    tables = {
        "quality_diagnostics": ("quality_diagnostics.json", "quality_diagnostic_row_count"),
        "training_history": ("training_history.json", "training_history_row_count"),
        "sampling_diagnostics": ("sampling_diagnostics.json", "sampling_diagnostics_row_count"),
        "physical_recording_qc": ("physical_recording_qc.json", "physical_recording_qc_row_count"),
        "route_artifacts": ("route_artifacts.json", "route_artifacts_row_count"),
    }
    for field, (filename, count_field) in tables.items():
        if field in compact:
            rows = compact.pop(field)
            compact[f"{field}_artifact"] = locate(filename)
            compact[count_field] = len(rows)
        elif compact.get(f"{field}_artifact") == filename:
            compact[f"{field}_artifact"] = locate(filename)
    if "route_window_sqi_evidence" in compact:
        records = compact.pop("route_window_sqi_evidence")
        compact.update({
            "route_window_sqi_evidence_artifact": locate("route_window_sqi_evidence.jsonl.gz"),
            "route_window_sqi_evidence_row_count": sum((len(row.get("windows", {})) for row in records)),
            "route_window_sqi_evidence_compression": "gzip_mtime0_jsonl",
            "route_window_sqi_evidence_report_consumed": False,
        })
    if "preprocessing_cache" in compact:
        cache = compact.pop("preprocessing_cache")
        compact["preprocessing_cache_artifact"] = locate("preprocessing_cache.json")
        compact["preprocessing_cache_summary"] = {
            "mode": cache.get("mode"), "counts": cache.get("counts", {}),
            "logical_array_bytes": cache.get("logical_array_bytes", 0), "affects_predictions": False,
        }
    return compact

def _write_cell_artifacts(directory: Path, cell: _CellResult) -> None:
    """Persist one fold's data tables and compact metadata; plotting is external."""
    writer = _runtime_imports()["OofWriter"]()
    directory.mkdir(parents=True, exist_ok=False)
    if cell.checkpoint is not None:
        from .v5.checkpoints import save_fold_checkpoint

        cell.summary["learned_model_checkpoint"] = save_fold_checkpoint(directory / "model_checkpoint", cell.checkpoint)
    _, cell.summary["route_window_sqi_evidence_sha256"] = _write_route_window_sqi_evidence(directory, cell.summary)
    levels = (("window", cell.window_rows,
               "representation_predictions_begin_at_file_level"), ("file", cell.file_rows, "no_file_predictions"),
              ("role", cell.role_rows, "equal_files_ablation_has_no_role_aggregation_level"),
              ("subject", cell.subject_rows, "no_subject_predictions"),
              ("member", cell.member_rows, "single_model_runner_ensemble_comparison_not_executed"),
              )
    for level, rows, reason in levels:
        path = directory / f"oof_{level}_predictions.parquet"
        writer.write(rows, path) if rows else _write_empty_oof(path, reason)
    repeat, fold = (cell.summary["repeat_index"], cell.summary["fold_index"])
    documents: dict[str, Mapping[str, Any]] = {
        "quality_diagnostics.json": {
            "schema_version": "ppg_frailty.quality_diagnostics.v2", "quality_mode": cell.summary["quality_mode"],
            "rows": cell.summary["quality_diagnostics"],
        }, "training_history.json": {
            "schema_version": "ppg_frailty.training_history.v2", "repeat_index": repeat, "fold_index": fold,
            "learning_curve_contract": cell.summary["learning_curve_contract"],
            "rows": [{"repeat": repeat, "fold": fold, **dict(row)} for row in cell.summary["training_history"]],
        }, "physical_recording_qc.json": {
            "schema_version": "ppg_frailty.physical_recording_qc.v2", "repeat_index": repeat, "fold_index": fold,
            "rows": cell.summary.get("physical_recording_qc", []),
        }, "route_artifacts.json": {
            "schema_version": "ppg_frailty.route_artifacts.v2", "repeat_index": repeat, "fold_index": fold,
            "rows": cell.summary.get("route_artifacts", []),
        }, "preprocessing_cache.json": cell.summary.get("preprocessing_cache", {"mode": "off", "counts": {}}),
        "metrics_per_fold_seed.json": {
            "schema_version": "ppg_frailty.metrics_per_fold_seed.v2", "pipeline_generation": "final_pipeline_v2",
            "cells": [_artifact_index_cell_summary(cell.summary)],
        }, "confusion_matrices.json": {
            "schema_version": "ppg_frailty.confusion_matrices.v2", "pipeline_generation": "final_pipeline_v2",
            "cells": [{
                "repeat_index": repeat, "fold_index": fold, "class_order": cell.summary["class_order"],
                "confusion_matrix": cell.summary["metrics"]["confusion_matrix"],
            }],
        },
    }
    if "sampling_diagnostics" in cell.summary:
        documents["sampling_diagnostics.json"] = {
            "schema_version": "ppg_frailty.legacy_bridge_sampling.v1", "repeat_index": repeat, "fold_index": fold,
            "profile_id": cell.summary["legacy_bridge"]["profile"]["profile_id"],
            "rows": cell.summary["sampling_diagnostics"],
        }
    for filename, payload in documents.items():
        _strict_json(directory / filename, payload)
    mandatory = [
        *documents, *(f"oof_{level}_predictions.parquet" for level, _, _ in levels),
        "route_window_sqi_evidence.jsonl.gz",
    ]
    if cell.checkpoint is not None:
        mandatory.append("model_checkpoint/manifest.json")
    _strict_json(
        directory / "run_manifest.json", {
            "schema_version": "ppg_frailty.run_manifest.v2", "pipeline_generation": "final_pipeline_v2",
            "status": "passed", "scientific_scope": cell.summary["scientific_scope"],
            "cell": _artifact_index_cell_summary(cell.summary
                                                 ), "mandatory_artifacts": ["run_manifest.json", *mandatory],
        },
    )

def _write_failed_artifacts(directory: Path, result: ExperimentResult) -> None:
    """Persist an explicit failure with typed empty prediction tables."""
    directory.mkdir(parents=True, exist_ok=False)
    reason = ";".join(result.failure_reasons) or "failed_closed"
    for level in ("window", "file", "role", "subject", "member"):
        _write_empty_oof(directory / f"oof_{level}_predictions.parquet", reason)
    empty = {"status": "failed_closed", "rows": []}
    for filename, schema in (("quality_diagnostics.json", "ppg_frailty.quality_diagnostics.v2"),
                             ("training_history.json", "ppg_frailty.training_history.v2"),
                             ("metrics_per_fold_seed.json", "ppg_frailty.metrics_per_fold_seed.v2"),
                             ("confusion_matrices.json", "ppg_frailty.confusion_matrices.v2"),
                             ):
        _strict_json(directory / filename, {"schema_version": schema, **empty})
    manifest = result.to_dict()
    manifest["schema_version"] = "ppg_frailty.run_manifest.v2"
    _strict_json(directory / "run_manifest.json", manifest)

def _resolve_output_directory(paths: Any, requested: Any, default_name: str) -> Path:
    """Resolve an explicit output target and reject overwrites."""
    raw = requested if requested is not None else Path("artifacts") / default_name
    candidate = Path(raw)
    target = candidate.expanduser().resolve() if candidate.is_absolute() else paths.output_path(candidate)
    if target.exists():
        raise FileExistsError(f"experiment_output_exists:{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target

def _commit_staging(staging: Path, target: Path) -> None:
    """Atomically publish a completed experiment directory on one filesystem."""
    staging.replace(target)

def _notify_progress(callback: Any, stage: str, **payload: Any) -> None:
    """Emit one small progress event without selecting a terminal UI library."""
    if callback is None:
        return
    if not callable(callback):
        raise TypeError("progress_callback must be callable")
    event = {"stage": str(stage), "event": str(stage), **payload}
    if "current_cell" in event:
        event["current"] = event["current_cell"]
    if "total_cells" in event:
        event["total"] = event["total_cells"]
    callback(event)

_LEGACY_BRIDGE_SOURCE_SPECIFICATION = (
    "AA_TODO/old_version_compare_V2/CODEX_LEGACY_V2_BRIDGE_REVISED_9_CASES_WITH_PHASE0.md"
)

def _resolve_legacy_bridge_execution(
    paths: Any, config: Any, *, profile_id: str, source_specification: str | None = None,
    source_specification_sha256: str | None = None, protocol_design: str = "cumulative_chain_v1",
    profile_definition: Mapping[str, Any] | None = None, profile_definition_sha256: str | None = None,
) -> _LegacyBridgeExecution:
    """Bind bridge execution only to algorithm inputs, never to audit status."""
    from .legacy_bridge import resolve_legacy_bridge_profile
    from .provenance import stable_payload_sha256

    design = str(protocol_design)
    del source_specification_sha256
    relative = (
        Path(source_specification or _LEGACY_BRIDGE_SOURCE_SPECIFICATION) if design == "cumulative_chain_v1" else None
    )
    source_path = None if relative is None else paths.repository_root / relative
    observed_sha256 = (
        hashlib.sha256(source_path.read_bytes()).hexdigest() if source_path and source_path.is_file() else None
    )
    manifest_path = paths.input_path(config.section("manifest")["path"])
    split_path = paths.input_path(config.section("splits")["path"])
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    split_sha256 = hashlib.sha256(split_path.read_bytes()).hexdigest()
    profile = resolve_legacy_bridge_profile(
        profile_id, protocol_design=design, profile_definition=profile_definition,
        profile_definition_sha256=profile_definition_sha256,
    )
    effective_payload = {
        "schema_version": f"ppg_frailty.legacy_bridge_effective_config.v{2 if relative else 3}",
        "canonical_config_hash": config.sha256, "profile": profile.to_dict(), "manifest_sha256": manifest_sha256,
        "split_sha256": split_sha256,
    }
    if relative:
        effective_payload.update(source_specification=relative.as_posix(), source_specification_sha256=observed_sha256)
    else:
        effective_payload.update(protocol_design=design, profile_definition_sha256=profile.profile_definition_sha256)
    effective_config_hash = stable_payload_sha256(effective_payload)
    return _LegacyBridgeExecution(
        profile=profile, source_specification=None if relative is None else relative.as_posix(),
        source_specification_sha256=observed_sha256, manifest_sha256=manifest_sha256, split_sha256=split_sha256,
        effective_config_hash=effective_config_hash, protocol_design=design,
        profile_definition_sha256=profile.profile_definition_sha256,
    )

def run_outer_cell(
    config_path: str | Path, repeat_index: int, fold_index: int, output_dir: str | Path, *,
    progress_callback: Any = None, measure_operational_costs: bool = False,
    preprocessing_cache: Mapping[str, Any] | None = None,
) -> ExperimentResult:
    """Run one complete outer cell with fold-local fitting and full records."""
    return _run_one_outer_cell(
        config_path, repeat_index=int(repeat_index), fold_index=int(fold_index), output_dir=output_dir,
        scope="selected_outer_cell", maximum_seconds=None, record_cap=None, epoch_override=None,
        progress_callback=progress_callback, measure_operational_costs=measure_operational_costs,
        preprocessing_cache=preprocessing_cache,
    )

def run_legacy_bridge_outer_cell(
    config_path: str | Path, repeat_index: int, fold_index: int, output_dir: str | Path, *, profile_id: str,
    source_specification: str | None = None, source_specification_sha256: str | None = None,
    protocol_design: str = "cumulative_chain_v1", profile_definition: Mapping[str, Any] | None = None,
    profile_definition_sha256: str | None = None, progress_callback: Any = None,
    measure_operational_costs: bool = False,
) -> ExperimentResult:
    """Run one isolated source-bound or field-driven cell from fresh raw bytes."""
    if str(protocol_design) == "cumulative_chain_v1" and int(repeat_index) != 0:
        raise ValueError("cumulative legacy bridge execution is frozen to repeat_index=0")
    return _run_one_outer_cell(
        config_path, repeat_index=int(repeat_index), fold_index=int(fold_index), output_dir=output_dir,
        scope="legacy_v2_bridge_selected_outer_cell", maximum_seconds=None, record_cap=None, epoch_override=None,
        progress_callback=progress_callback, measure_operational_costs=measure_operational_costs,
        legacy_bridge_profile_id=str(profile_id), legacy_bridge_source_specification=source_specification,
        legacy_bridge_source_specification_sha256=source_specification_sha256,
        legacy_bridge_protocol_design=str(protocol_design), legacy_bridge_profile_definition=profile_definition,
        legacy_bridge_profile_definition_sha256=profile_definition_sha256,
    )

def run_reduced_fold_experiment(
    config_path: str | Path, *, repeat_index: int = 0, fold_index: int = 0, max_seconds_per_record: float = 60.0,
    max_records_per_participant: int = 1, fixed_epochs_override: int = 1, output_dir: str | Path | None = None,
    progress_callback: Any = None,
) -> ExperimentResult:
    """Run one shortened diagnostic cell while preserving outer membership."""
    if max_seconds_per_record < 10.0:
        raise ValueError("reduced smoke requires at least ten seconds per record")
    if max_records_per_participant <= 0 or fixed_epochs_override <= 0:
        raise ValueError("record and epoch limits must be positive")
    resolved_output = (
        output_dir or Path("artifacts") / f"reduced_r{int(repeat_index)}_f{int(fold_index)}_{time.time_ns()}"
    )
    return _run_one_outer_cell(
        config_path, repeat_index=int(repeat_index), fold_index=int(fold_index), output_dir=resolved_output,
        scope="smoke_not_scientific_benchmark", maximum_seconds=float(max_seconds_per_record),
        record_cap=int(max_records_per_participant), epoch_override=int(fixed_epochs_override),
        progress_callback=progress_callback, measure_operational_costs=False,
    )

def _run_one_outer_cell(
    config_path: str | Path, *, repeat_index: int, fold_index: int, output_dir: str | Path, scope: str,
    maximum_seconds: float | None, record_cap: int | None, epoch_override: int | None, progress_callback: Any,
    measure_operational_costs: bool, preprocessing_cache: Mapping[str, Any] | None = None,
    legacy_bridge_profile_id: str | None = None, legacy_bridge_source_specification: str | None = None,
    legacy_bridge_source_specification_sha256: str | None = None,
    legacy_bridge_protocol_design: str = "cumulative_chain_v1",
    legacy_bridge_profile_definition: Mapping[str, Any] | None = None,
    legacy_bridge_profile_definition_sha256: str | None = None,
) -> ExperimentResult:
    """Shared single-cell runner for canonical, smoke and legacy entry points."""
    if repeat_index not in range(5) or fold_index not in range(5):
        raise ValueError("repeat_index and fold_index must lie in 0..4")
    api = _runtime_imports()
    paths = api["PipelinePaths"].discover()
    report, config, rows, registry = api["preflight_pipeline"](config_path, mode="full", paths=paths)
    bridge = (
        _resolve_legacy_bridge_execution(
            paths, config, profile_id=str(legacy_bridge_profile_id),
            source_specification=legacy_bridge_source_specification,
            source_specification_sha256=legacy_bridge_source_specification_sha256,
            protocol_design=str(legacy_bridge_protocol_design), profile_definition=legacy_bridge_profile_definition,
            profile_definition_sha256=legacy_bridge_profile_definition_sha256,
        ) if legacy_bridge_profile_id is not None else None
    )
    _assert_legacy_bridge_entrypoint_contract(config, bridge, dedicated_entrypoint=bridge is not None)
    target = _resolve_output_directory(paths, output_dir, "outer_cell")
    staging = target.with_name(f".{target.name}.staging.{time.time_ns()}")
    _notify_progress(progress_callback, "cell_start", repeat_index=repeat_index, fold_index=fold_index)
    config_hash = bridge.effective_config_hash if bridge is not None else config.sha256
    provenance = {
        "preflight_status": report.status, "manifest_hash": report.manifest_hash, "fold_hash": report.fold_hash,
        "frozen_outer_split": True, "record_seconds_cap": maximum_seconds, "record_cap_per_participant": record_cap,
        "code_version": _code_version(), "source_version": _source_version(),
    }
    try:
        try:
            cell = _execute_cell(
                report, config, rows, registry, paths, repeat_index=repeat_index, fold_index=fold_index,
                maximum_seconds=maximum_seconds, record_cap=record_cap,
                epoch_override=_epoch_override_for_backend(config, epoch_override),
                measure_operational_costs=bool(measure_operational_costs), legacy_bridge=bridge,
                preprocessing_cache=preprocessing_cache,
            )
            cell.summary["scientific_scope"] = scope
            result = ExperimentResult(
                "passed", scope, config.config_id, config_hash, (repeat_index, ), (fold_index, ), str(target),
                (_artifact_index_cell_summary(cell.summary), ), dict(cell.summary["metrics"]), provenance,
            )
            _write_cell_artifacts(staging, cell)
        except _ExperimentProtocolError as exc:
            result = ExperimentResult(
                "failed_closed", scope, config.config_id, config_hash, (repeat_index, ), (fold_index, ), str(target),
                provenance=provenance, failure_reasons=(str(exc), ),
            )
            _write_failed_artifacts(staging, result)
        _strict_json(staging / "experiment_result.json", result.to_dict())
        _commit_staging(staging, target)
        _notify_progress(progress_callback, "cell_complete", status=result.status, output_dir=str(target))
        return result
    except Exception:
        if staging.exists():
            import shutil

            shutil.rmtree(staging)
        raise

def _write_full_root_artifacts(
    directory: Path, cells: Iterable[_CellResult], result: ExperimentResult, *, cell_directory_layout: str = "flat"
) -> None:
    """Merge fold-level prediction tables and data-only summaries at run root."""
    values = tuple(cells)
    writer = _runtime_imports()["OofWriter"]()
    reasons = {
        "window": "no_window_level_predictions_in_successful_outer_cells", "file": "no_successful_outer_cells",
        "role": "no_role_level_predictions_or_equal_files_ablation", "subject": "no_successful_outer_cells",
        "member": "single_model_runner_ensemble_comparison_not_executed",
    }
    for level in reasons:
        rows = tuple((row for cell in values for row in getattr(cell, f"{level}_rows")))
        path = directory / f"oof_{level}_predictions.parquet"
        writer.write(rows, path) if rows else _write_empty_oof(path, reasons[level])
    prefix = (
        lambda cell: f"repeat_{int(cell.summary['repeat_index']):02d}/fold_{int(cell.summary['fold_index']):02d}"
        if cell_directory_layout == "nested" else
        f"repeat_{int(cell.summary['repeat_index']):02d}_fold_{int(cell.summary['fold_index']):02d}"
    )
    indexed = [_artifact_index_cell_summary(cell.summary, artifact_prefix=prefix(cell)) for cell in values]
    _strict_json(
        directory / "quality_diagnostics.json", {
            "schema_version": "ppg_frailty.quality_diagnostics.v2", "status": result.status, "cells": [{
                "repeat_index": cell.summary["repeat_index"], "fold_index": cell.summary["fold_index"],
                "quality_mode": cell.summary["quality_mode"], "rows": cell.summary["quality_diagnostics"],
            } for cell in values],
        },
    )
    _strict_json(
        directory / "metrics_per_fold_seed.json", {
            "schema_version": "ppg_frailty.metrics_per_fold_seed.v2", "pipeline_generation": "final_pipeline_v2",
            "status": result.status, "cells": indexed,
        },
    )
    _strict_json(
        directory / "confusion_matrices.json", {
            "schema_version": "ppg_frailty.confusion_matrices.v2", "pipeline_generation": "final_pipeline_v2",
            "status": result.status, "cells": [{
                "repeat_index": row["repeat_index"], "fold_index": row["fold_index"], "class_order": row["class_order"],
                "confusion_matrix": row["metrics"]["confusion_matrix"],
            } for row in indexed],
        },
    )
    policy = next((cell.summary.get("evaluation_policy", {}).get("statistics", {}) for cell in values), {})
    metrics = _trusted_config_metrics_payload(
        values, result, n_bootstrap_resamples=int(policy.get("bootstrap_replicates", 10000)),
        bootstrap_seed=int(policy.get("seed", 42)),
    )
    _strict_json(directory / "config_metrics_v2.json", metrics)
    manifest = result.to_dict()
    manifest.update({
        "schema_version": "ppg_frailty.run_manifest.v2", "pipeline_generation": "final_pipeline_v2",
        "mandatory_artifacts": [
            "run_manifest.json", "metrics_per_fold_seed.json", "confusion_matrices.json",
            "oof_window_predictions.parquet", "oof_file_predictions.parquet", "oof_role_predictions.parquet",
            "oof_subject_predictions.parquet", "oof_member_predictions.parquet", "quality_diagnostics.json",
            "config_metrics_v2.json",
        ],
    })
    _strict_json(directory / "run_manifest.json", manifest)

_FORMAL_SPLIT_SEEDS = (42, 10042, 20042, 30042, 40042)

def _registry_role_for_machine_id(machine_id: str) -> str:
    """Resolve provenance role from the complete model registry."""
    return str(_model_capability_contract(str(machine_id))["registry_role"])

def _participant_predictions_from_subject_rows(rows: Iterable[Any]) -> tuple[Any, ...]:
    """Return participant rows in deterministic participant/repeat order."""
    return tuple(sorted(rows, key=lambda item: (item.participant_id, item.repeat)))

def _fold_confusions_and_rosters_from_subject_rows(
    rows: Iterable[Any],
) -> tuple[dict[str, tuple[tuple[int, ...], ...]], dict[str, tuple[str, ...]]]:
    """Rebuild retained-only fold confusion and the complete held-out roster."""
    matrices = {f"r{repeat}f{fold}": [[0, 0, 0] for _ in range(3)] for repeat in range(5) for fold in range(5)}
    rosters = {key: [] for key in matrices}
    for row in rows:
        key = f"r{int(row.repeat)}f{int(row.fold)}"
        if row.retained:
            predicted = max(range(3), key=lambda index: float(row.probabilities[index]))
            matrices[key][int(row.label)][predicted] += 1
        rosters[key].append(str(row.participant_id))
    return ({key: tuple((tuple((int(value) for value in row)) for row in matrix))
             for key, matrix in matrices.items()}, {key: tuple(sorted(values)) for key, values in rosters.items()},
            )

def _operational_metrics_from_fold_summaries(summaries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate optional CPU batch-1 measurements without affecting eligibility math."""
    rows = tuple((row.get("operational_metrics", {}) for row in summaries))
    measured = all((
        row.get("status") == "measured_explicit_cpu_batch1_request" and row.get("parameter_count") is not None
        and (row.get("model_latency_p50_ms") is not None) and (row.get("model_latency_p95_ms") is not None)
        for row in rows
    ))
    if measured and rows:
        p50 = tuple((float(row["model_latency_p50_ms"]) for row in rows))
        p95 = tuple((float(row["model_latency_p95_ms"]) for row in rows))
        counts = tuple((int(row["parameter_count"]) for row in rows))
        return {
            "parameter_count": max(counts), "parameter_counts_by_cell": counts, "inference_cost": {
                "cpu_batch1_model_only_p50_ms_mean_across_25_outer_cells": sum(p50) / len(p50),
                "cpu_batch1_model_only_p95_ms_mean_across_25_outer_cells": sum(p95) / len(p95),
            }, "eligible": True, "exclusion_reason": "", "status": "measured_all_25_cells_explicit_request",
            "measurement_requested": True,
        }
    return {
        "parameter_count": None, "parameter_counts_by_cell": (),
        "inference_cost": {"participant_probability_latency_ms": None}, "eligible": False,
        "exclusion_reason": "parameter_count_and_inference_cost_not_measured",
        "status": "not_measured_in_current_runner", "measurement_requested": False,
    }

def _abstention_aware_root_metrics(
    rows: tuple[Any, ...], summaries: tuple[Mapping[str, Any], ...], *, config_id: str, registry_role: str,
    fold_confusions: Mapping[str, tuple[tuple[int, ...], ...]], operational: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute complete-roster repeat metrics without inventing an abstention class."""
    np = _runtime_imports()["np"]
    repeat_rows = [{
        "repeat": repeat, **_evaluate_subjects(
            tuple((row for row in rows if int(row.repeat) == repeat)), sum((int(row.repeat) == repeat for row in rows)),
        ),
    } for repeat in range(5)]
    pooled = _evaluate_subjects(rows, len(rows))
    confusion: dict[str, Any] = dict(fold_confusions)
    confusion.update({f"repeat_{row['repeat']}": row["confusion_matrix"] for row in repeat_rows})
    confusion["pooled_participant_repeat"] = pooled["confusion_matrix"]

    def metric_values(name: str) -> list[float]:
        return [float(row[name]) for row in repeat_rows if row.get(name) is not None]

    def metric_mean(name: str) -> float | None:
        values = [float(row[name]) for row in repeat_rows if row.get(name) is not None]
        return None if not values else float(np.mean(values))

    aware_ba = np.asarray(metric_values("abstention_aware_balanced_accuracy"), dtype=np.float64)
    aware_f1 = np.asarray(metric_values("abstention_aware_macro_f1"), dtype=np.float64)
    conditional_ba = metric_values("balanced_accuracy")
    conditional_f1 = metric_values("macro_f1")
    fold_aware_ba = [float(summary["metrics"]["abstention_aware_balanced_accuracy"]) for summary in summaries]
    conditional_per_class = pooled.get("per_class") or ()
    aware_per_class = pooled["abstention_aware_per_class"]
    return {
        "config_id": config_id, "registry_role": registry_role,
        "participant_mean_balanced_accuracy": None if not conditional_ba else float(np.mean(conditional_ba)),
        "participant_mean_macro_f1": None if not conditional_f1 else float(np.mean(conditional_f1)),
        "participant_mean_abstention_aware_balanced_accuracy": float(aware_ba.mean()),
        "participant_mean_abstention_aware_macro_precision": metric_mean("abstention_aware_macro_precision"),
        "participant_mean_abstention_aware_macro_recall": metric_mean("abstention_aware_macro_recall"),
        "participant_mean_abstention_aware_macro_f1": float(aware_f1.mean()),
        "participant_mean_coverage_rate": metric_mean("coverage_rate"), "abstention_count": pooled["abstention_count"],
        "abstention_counts_by_class": pooled["abstention_counts_by_class"], "worst_fold_balanced_accuracy": min((
            float(summary["metrics"]["balanced_accuracy"]) for summary in summaries
            if summary["metrics"].get("balanced_accuracy") is not None
        )) if any((summary["metrics"].get("balanced_accuracy") is not None for summary in summaries)) else None,
        "worst_fold_abstention_aware_balanced_accuracy": min(fold_aware_ba), "balanced_accuracy_lcb95": None,
        "macro_f1_lcb95": None, "abstention_aware_balanced_accuracy_lcb95": None,
        "abstention_aware_macro_f1_lcb95": None,
        "worst_class_recall": min((float(item["recall"])
                                   for item in conditional_per_class)) if conditional_per_class else None,
        "worst_class_f1": min((float(item["f1"]) for item in conditional_per_class)) if conditional_per_class else None,
        "abstention_aware_worst_class_recall": min(
            (float(item["recall"]) for item in aware_per_class)
        ), "abstention_aware_worst_class_f1": min((float(item["f1"]) for item in aware_per_class)
                                                  ), "expected_calibration_error": pooled["expected_calibration_error"],
        "probability_metrics_scope": "retained_only",
        "primary_ranking_metric": "participant_mean_abstention_aware_balanced_accuracy",
        "repeat_metrics": tuple(repeat_rows), "variability": {
            "repeat_balanced_accuracy_population_sd": 0.0
            if not conditional_ba else float(np.std(conditional_ba, ddof=0)),
            "repeat_macro_f1_population_sd": 0.0 if not conditional_f1 else float(np.std(conditional_f1, ddof=0)),
            "repeat_abstention_aware_balanced_accuracy_population_sd": float(aware_ba.std(ddof=0)),
            "repeat_abstention_aware_macro_f1_population_sd": float(aware_f1.std(ddof=0)),
            "fold_abstention_aware_balanced_accuracy_population_sd": float(np.std(fold_aware_ba, ddof=0)),
        }, "confusion_matrices": confusion, "confusion_matrix_scope": "retained_predictions_only",
        "inference_cost": operational["inference_cost"], "parameter_count": operational["parameter_count"],
        "eligible": operational["eligible"], "exclusion_reason": operational["exclusion_reason"],
    }

def _trusted_config_metrics_payload(
    cells: Iterable[_CellResult], result: ExperimentResult, *, n_bootstrap_resamples: int = 10000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Rebuild run-level metrics from typed OOF; never reuse displayed summaries."""
    values = tuple(cells)
    base = {
        "schema_version": "ppg_frailty.config_metrics.v2", "pipeline_generation": "final_pipeline_v2",
        "config_id": result.config_id, "config_hash": result.config_hash, "independent_test": False,
        "fold_protocol": "frozen_repeated_grouped_5x5", "automatic_selection": False,
        "training_executed_by_this_writer": False,
    }
    grid = {(int(cell.summary["repeat_index"]), int(cell.summary["fold_index"])) for cell in values}
    if (
        result.status != "passed" or len(values) != 25
        or grid != {(repeat, fold) for repeat in range(5) for fold in range(5)}
    ):
        return {**base, "status": "not_available_requires_complete_passed_5x5", "config_metrics": None}
    summaries = tuple((cell.summary for cell in values))
    machine_id = str(summaries[0]["model_machine_id"])
    subject_rows = _participant_predictions_from_subject_rows((row for cell in values for row in cell.subject_rows))
    participants = tuple(sorted({str(row.participant_id) for row in subject_rows}))
    fold_ba = {
        f"r{int(row['repeat_index'])}f{int(row['fold_index'])}": row["metrics"].get("balanced_accuracy")
        for row in summaries
    }
    confusions, rosters = _fold_confusions_and_rosters_from_subject_rows(subject_rows)
    operational = _operational_metrics_from_fold_summaries(summaries)
    role = _registry_role_for_machine_id(machine_id)
    training_seeds = (
        list(map(int, summaries[0].get("member_training_seeds", ())))
        if _model_is_ensemble(machine_id) else sorted({int(row["training_seed"]) for row in summaries})
    )
    common = {
        **base, "model_machine_id": machine_id, "registry_role": role, "seeds": list(_FORMAL_SPLIT_SEEDS),
        "training_seeds": training_seeds, "participant_oof_coverage": {
            "participant_count": len(participants), "repeat_count": 5, "participant_repeat_rows": len(subject_rows),
            "roster_complete": True,
        }, "fold_balanced_accuracies": fold_ba, "operational_measurement_status": operational["status"],
    }
    if any((not bool(row.retained) for row in subject_rows)):
        metrics = _abstention_aware_root_metrics(
            subject_rows, summaries, config_id=result.config_id, registry_role=role, fold_confusions=confusions,
            operational=operational,
        )
        return {
            **common, "status": "passed_trusted_abstention_aware_metrics_rebuilt_from_typed_oof",
            "config_metrics": to_strict_json_value(metrics), "bootstrap_results": [],
            "bootstrap_policy": {"status": "not_available_for_abstention_aware_endpoint"},
        }
    metrics, bootstrap = _runtime_imports()["build_config_metrics_from_predictions_and_fold_summaries"](
        config_id=result.config_id, registry_role=role, predictions=subject_rows,
        fold_balanced_accuracies={key: float(value) for key, value in fold_ba.items()
                                  }, fold_confusion_matrices=confusions, fold_participant_rosters=rosters,
        inference_cost=operational["inference_cost"], parameter_count=operational["parameter_count"],
        n_bootstrap_resamples=int(n_bootstrap_resamples), bootstrap_seed=int(bootstrap_seed),
        eligible=operational["eligible"], exclusion_reason=operational["exclusion_reason"],
    )
    return {
        **common, "status": "passed_trusted_metrics_rebuilt_from_typed_oof",
        "config_metrics": to_strict_json_value(asdict(metrics)),
        "bootstrap_results": to_strict_json_value([asdict(item) for item in bootstrap]), "bootstrap_policy": {
            "resamples": int(n_bootstrap_resamples), "seed": int(bootstrap_seed),
            "unit": "participant_with_all_repeats",
        },
    }

def run_full_experiment(
    config_path: str | Path, *, output_dir: str | Path, repeats: Iterable[int] = tuple(range(5)),
    folds: Iterable[int] = tuple(range(5)), measure_operational_costs: bool = False, progress_callback: Any = None,
    preprocessing_cache: Mapping[str, Any] | None = None, cell_directory_layout: str = "flat",
) -> ExperimentResult:
    """Execute the requested outer-fold grid without hidden training branches."""
    repeat_values, fold_values = (tuple(map(int, repeats)), tuple(map(int, folds)))
    if (
        not repeat_values or not fold_values or len(set(repeat_values)) != len(repeat_values)
        or (len(set(fold_values)) != len(fold_values)) or (not set(repeat_values) <= set(range(5)))
        or (not set(fold_values) <= set(range(5))) or (cell_directory_layout not in {"flat", "nested"})
    ):
        raise ValueError("repeats/folds must be unique subsets of 0..4 and layout flat/nested")
    complete = set(repeat_values) == set(range(5)) and set(fold_values) == set(range(5))
    scope = "frozen_5x5_scientific_benchmark" if complete else "selected_full_length_cells_not_complete_5x5"
    api = _runtime_imports()
    paths = api["PipelinePaths"].discover()
    report, config, rows, registry = api["preflight_pipeline"](config_path, mode="full", paths=paths)
    _assert_legacy_bridge_entrypoint_contract(config, None, dedicated_entrypoint=False)
    target = _resolve_output_directory(paths, output_dir, "full_experiment")
    staging = target.with_name(f".{target.name}.staging.{time.time_ns()}")
    staging.mkdir(parents=True, exist_ok=False)
    passed: list[_CellResult] = []
    summaries: list[dict[str, Any]] = []
    failures: list[str] = []
    started = time.perf_counter()
    grid = tuple(((repeat, fold) for repeat in repeat_values for fold in fold_values))
    _notify_progress(progress_callback, "run_start", config_id=config.config_id, total_cells=len(grid))
    try:
        for position, (repeat, fold) in enumerate(grid, 1):
            relative = (
                Path(f"repeat_{repeat:02d}") / f"fold_{fold:02d}"
                if cell_directory_layout == "nested" else Path(f"repeat_{repeat:02d}_fold_{fold:02d}")
            )
            _notify_progress(
                progress_callback, "cell_start", current_cell=position, repeat_index=repeat, fold_index=fold
            )
            try:
                cell = _execute_cell(
                    report, config, rows, registry, paths, repeat_index=repeat, fold_index=fold, maximum_seconds=None,
                    record_cap=None, epoch_override=None, measure_operational_costs=bool(measure_operational_costs),
                    preprocessing_cache=preprocessing_cache,
                )
                cell.summary["scientific_scope"] = scope
                _write_cell_artifacts(staging / relative, cell)
                summaries.append(_artifact_index_cell_summary(cell.summary, artifact_prefix=relative.as_posix()))
                passed.append(_cell_for_root_aggregation(cell, artifact_prefix=relative.as_posix()))
                status = "passed"
            except _ExperimentProtocolError as exc:
                failure = f"r{repeat}_f{fold}:{exc}"
                failures.append(failure)
                summaries.append({
                    "status": "failed_closed", "repeat_index": repeat, "fold_index": fold, "reason": str(exc)
                })
                _write_failed_artifacts(
                    staging / relative,
                    ExperimentResult(
                        "failed_closed", scope, config.config_id, config.sha256, (repeat, ), (fold, ),
                        str(staging / relative), failure_reasons=(failure, ),
                    ),
                )
                status = "failed_closed"
            _release_cell_memory()
            _notify_progress(progress_callback, "cell_complete", current_cell=position, status=status)
        result = ExperimentResult(
            "passed" if not failures else "failed_closed", scope, config.config_id, config.sha256, repeat_values,
            fold_values, str(target), tuple(summaries), {
                "requested_cell_count": len(grid), "passed_cell_count": len(passed), "failed_cell_count": len(failures),
                "elapsed_seconds": time.perf_counter() - started,
            }, {
                "preflight_status": report.status, "manifest_hash": report.manifest_hash, "fold_hash": report.fold_hash,
                "frozen_outer_split": True, "operational_measurement_requested": bool(measure_operational_costs),
            }, tuple(failures),
        )
        _write_full_root_artifacts(staging, passed, result, cell_directory_layout=cell_directory_layout)
        _strict_json(staging / "experiment_result.json", result.to_dict())
        _commit_staging(staging, target)
        _notify_progress(progress_callback, "run_complete", status=result.status, output_dir=str(target))
        return result
    except Exception:
        if staging.exists():
            import shutil

            shutil.rmtree(staging)
        raise

def _load_json(path: Path, root_type: type) -> Any:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, root_type):
        raise ValueError(f"json_root_must_be_{root_type.__name__}:{path}")
    return value

def _load_strict_json_object(path: Path) -> dict[str, Any]:
    return _load_json(path, dict)

def _fold_summaries_from_run(directory: Path) -> tuple[dict[str, Any], ...]:
    payload = _load_strict_json_object(directory / "metrics_per_fold_seed.json")
    return tuple(dict(row) for row in payload.get("cells", ()) if isinstance(row, Mapping))

def _read_comparison_run(
    requested_id: str, directory: str | Path, *, n_bootstrap_resamples: int | None, bootstrap_seed: int | None
) -> dict[str, Any]:
    """Read one run and rebuild decision metrics directly from typed OOF rows."""
    from .training import build_config_metrics_from_predictions_and_fold_summaries, read_oof_parquet

    root = Path(directory).resolve()
    manifest = _load_strict_json_object(root / "run_manifest.json")
    summaries = _fold_summaries_from_run(root)
    config_id = str(manifest.get("config_id", requested_id))
    config_hash = str(manifest.get("config_hash", summaries[0]["config_hash"]))
    rows = _participant_predictions_from_subject_rows(read_oof_parquet(root / "oof_subject_predictions.parquet"))
    predictions = tuple((row for row in rows if row.retained))
    participants = tuple(sorted({str(row.participant_id) for row in rows}))
    membership = {(str(row.participant_id), int(row.repeat)): (int(row.fold), int(row.split_seed), int(row.label))
                  for row in rows}
    policy = summaries[0].get("evaluation_policy", {}).get("statistics", {})
    resamples = int(n_bootstrap_resamples or policy.get("bootstrap_replicates", 10000))
    seed = int(bootstrap_seed if bootstrap_seed is not None else policy.get("seed", 42))
    fold_ba = {
        f"r{int(row['repeat_index'])}f{int(row['fold_index'])}": float(row["metrics"]["balanced_accuracy"])
        for row in summaries
    }
    confusions, rosters = _fold_confusions_and_rosters_from_subject_rows(rows)
    operational = _operational_metrics_from_fold_summaries(summaries)
    machine_id = str(summaries[0]["model_machine_id"])
    metrics, bootstrap = build_config_metrics_from_predictions_and_fold_summaries(
        config_id=config_id, registry_role=_registry_role_for_machine_id(machine_id), predictions=predictions,
        fold_balanced_accuracies=fold_ba, fold_confusion_matrices=confusions, fold_participant_rosters=rosters,
        inference_cost=operational["inference_cost"], parameter_count=operational["parameter_count"],
        n_bootstrap_resamples=resamples, bootstrap_seed=seed, eligible=operational["eligible"],
        exclusion_reason=operational["exclusion_reason"],
    )
    return {
        "config_id": config_id, "config_hash": config_hash, "machine_id": machine_id,
        "seed_policy": str(summaries[0].get("seed_policy", "")), "metrics": metrics, "bootstrap": bootstrap,
        "predictions": predictions, "membership": membership, "participants": participants, "run_directory": str(root),
        "statistics_policy": policy or None,
    }

def build_comparison_archive_from_run_directories(
    run_directories: Mapping[str, str | Path], *, reference_config_id: str, comparison_family: str, comparison_id: str,
    run_id: str, output_root: str | Path, n_bootstrap_resamples: int | None = None,
    n_permutation_resamples: int | None = None, statistics_seed: int | None = None,
    allowed_authority_differences: Iterable[str] = (),
) -> dict[str, Any]:
    """Build statistics from existing runs; this path never trains or refits."""
    from .training import ComparisonArchive, holm_adjust_by_family_metric, paired_participant_permutation
    from .training.statistics import _write_formal_comparison_archive

    directories = {str(key): value for key, value in run_directories.items()}
    if len(directories) < 2 or reference_config_id not in directories:
        raise ValueError("comparison requires at least two runs and a reference")
    loaded = {
        key:
        _read_comparison_run(key, path, n_bootstrap_resamples=n_bootstrap_resamples, bootstrap_seed=statistics_seed)
        for key, path in sorted(directories.items())
    }
    reference = loaded[reference_config_id]
    for key, current in loaded.items():
        if current["membership"] != reference["membership"]:
            raise ValueError(f"comparison_roster_or_fold_membership_differs:{key}")
    policy = reference["statistics_policy"] or {}
    permutations = int(n_permutation_resamples or policy.get("paired_permutation_replicates", 100000))
    seed = int(statistics_seed if statistics_seed is not None else policy.get("seed", 42))
    paired: dict[str, Any] = {}
    p_values: dict[tuple[str, str, str], float] = {}
    for key, current in loaded.items():
        if key == reference_config_id:
            continue
        comparison = f"{key}_vs_{reference_config_id}"
        for metric in ("balanced_accuracy", "macro_f1"):
            outcome = paired_participant_permutation(
                reference["predictions"], current["predictions"], metric=metric, n_resamples=permutations, seed=seed
            )
            paired[f"{comparison}__{metric}"] = outcome
            p_values[str(comparison_family), metric, comparison] = outcome.two_sided_p_value
    source_runs = {
        key: {
            "config_hash": value["config_hash"], "model_machine_id": value["machine_id"],
            "run_directory": value["run_directory"],
        } for key, value in loaded.items()
    }
    archive = ComparisonArchive(
        comparison_id=str(comparison_id), run_id=str(run_id),
        configs=tuple((value["metrics"] for value in loaded.values())),
        bootstrap_results={key: value["bootstrap"] for key, value in loaded.items()}, paired_permutation_results=paired,
        holm_results=holm_adjust_by_family_metric(p_values, alpha=0.05), selections=(), run_manifest={
            "schema_version": "ppg_frailty.comparison_run_manifest.v2", "pipeline_generation": "final_pipeline_v2",
            "status": "statistics_completed_no_automatic_selection", "independent_test": False,
            "fold_protocol": "frozen_repeated_grouped_5x5", "seeds": list(_FORMAL_SPLIT_SEEDS),
            "reference_config_id": reference_config_id, "comparison_family": str(comparison_family),
            "comparison_id": str(comparison_id), "run_id": str(run_id), "source_runs": source_runs,
            "allowed_authority_differences": sorted(set(map(str, allowed_authority_differences))),
            "automatic_selection": False, "training_executed": False, "refit_executed": False,
        },
    )
    target = _write_formal_comparison_archive(archive, output_root)
    return {
        "schema_version": "ppg_frailty.comparison_orchestrator_result.v2", "status": "passed",
        "comparison_id": str(comparison_id), "run_id": str(run_id), "output_directory": str(target),
        "config_ids": sorted(loaded), "reference_config_id": reference_config_id,
        "comparison_family": str(comparison_family), "automatic_selection": False, "training_executed": False,
        "refit_executed": False,
    }

def final_refit_preflight(config_path: str | Path, *, purpose: str = "configured_full_cohort_refit") -> dict[str, Any]:
    """Describe direct full-cohort refit inputs without comparison/selection authority."""
    api = _runtime_imports()
    paths = api["PipelinePaths"].discover()
    report, config, rows, _ = api["preflight_pipeline"](config_path, mode="full", paths=paths)
    roles = set(_classifier_role_ids(config))
    participants = tuple(
        sorted({
            str(row.participant_id) for row in rows
            if row.role in roles and row.qc_status in {"pass", "pass_with_warnings"}
        })
    )
    if len(participants) != 29:
        raise _ExperimentProtocolError("final_refit_materializer_requires_exact_29_roster")
    _, machine_id = api["normalize_model_id"](str(config.section("model")["model_id"]))
    evidence = api["stable_payload_sha256"]({
        "config_hash": config.sha256, "participant_ids": participants, "scope": "direct_full_cohort_refit"
    })
    return {
        "status": "ready_for_all29_refit", "purpose": str(purpose), "config_id": config.config_id,
        "config_hash": config.sha256, "model_machine_id": machine_id, "participant_ids": participants,
        "participant_count": len(participants), "manifest_hash": report.manifest_hash,
        "fold_registry_hash": report.fold_hash, "module_registry_hash": report.module_registry_hash,
        "oof_evidence_sha256": evidence, "manual_selection_sha256": evidence, "selection_record_file_sha256": evidence,
        "source_snapshot_sha256": _source_version(), "training_executed": False,
    }

def _final_refit_run_provenance(
    config: Any, resolved_model: Mapping[str, Any], input_spec: Any, feature_hash: str, training_config: Any,
    fold_hash: str, seeds: tuple[int, ...],
) -> dict[str, Any]:
    model = config.section("model")
    training = config.section("training")
    signal = config.section("signal")
    windows = config.section("windows")
    quality = config.section("quality")
    selected = windows["engineering" if config.representation_mode in
                       {"feature_vector", "feature_matrix"} else "raw_dl"]
    _, machine_id = _runtime_imports()["normalize_model_id"](str(model["model_id"]))
    channels = input_spec.feature_names if config.representation_mode == "feature_vector" else input_spec.channel_schema
    fs_hz = _model_input_sampling_rate_hz(config)
    return {
        "architecture_parameters": dict(resolved_model["architecture_parameters"]),
        "input_channels_order": tuple(map(str, channels)), "sampling_rate_hz": fs_hz, "window_plan": {
            "representation_mode": config.representation_mode, "selected": dict(selected),
            "shared_planner_version": windows["shared_planner_version"], "model_input_sampling_rate_hz": fs_hz,
            "canonical_signal_and_feature_sampling_rate_hz": float(signal["internal_fs_hz"]),
        }, "hop_plan": {"hop_s": float(selected["hop_s"]), "end_alignment": selected["end_alignment"]},
        "normalization": dict(signal["normalization"]), "padding_mask": {
            "padding": selected["padding"], "min_valid_fraction": selected.get("min_valid_fraction", 1.0),
            "mask_aware_pooling": model.get("mask_aware_pooling", "not_applicable"),
        }, "feature_schema_hash": feature_hash,
        "sqi_routing": {"mode": quality["mode"],
                        "failure_action": quality["failure_action"]}, **_training_algorithm_provenance(
                            machine_id, training, model, fixed_epochs=int(training_config.fixed_epochs)
                        ), "random_seeds": seeds,
        "seed_policy": "member_roster" if len(seeds) > 1 else "fixed_explicit", "fold_hash": fold_hash,
        "aggregation": dict(config.section("aggregation")), "calibration": {
            "metrics": tuple(config.section("evaluation")["calibration_metrics"]), "fit_scope": "all29_final_refit",
        },
    }

def execute_final_refit(
    config_path: str | Path, *, bundle_directory: str | Path, purpose: str = "configured_full_cohort_refit",
) -> Path:
    """Fit and publish a resolved case directly on its complete cohort."""
    from .training import (
        FinalRefitPlan, FullCohortRefitScope, TrainingConfig, UnifiedTrainer, execute_full_cohort_refit,
        materialize_final_refit_binding, save_final_refit_bundle,
    )

    api = _runtime_imports()
    paths = api["PipelinePaths"].discover()
    report, config, rows, _ = api["preflight_pipeline"](config_path, mode="full", paths=paths)
    evidence = final_refit_preflight(config_path, purpose=purpose)
    materialized = _materialize_trusted_full29(report, config, rows, paths, evidence)
    dataset = materialized.dataset
    seed = int(config.section("training")["seed"])
    model_config, machine_id = _resolved_model_config(config, training_seed=seed, seed_scope="final_refit")
    training_config = api["replace"](TrainingConfig.from_mapping(config.section("training")), seed=seed)
    input_spec = _model_input_spec(dataset, config.representation_mode)
    ensemble = _model_is_ensemble(machine_id)
    classical = _model_uses_estimator(machine_id)
    seeds = _ensemble_member_seed_roster(config.section("model")) if ensemble else (seed, )
    scope = FullCohortRefitScope(
        participant_ids=tuple(evidence["participant_ids"]), registry_hash=report.module_registry_hash,
        config_hash=config.sha256, oof_evidence_hash=evidence["oof_evidence_sha256"],
    ).bind_training_dataset(dataset)
    prepared = api["prepare_model_factory"](model_config, input_spec, dataset, scope)
    provenance = _final_refit_run_provenance(
        config, prepared.resolved_model_config, input_spec, materialized.feature_hash, training_config, scope.fold_hash,
        seeds,
    )
    binding = materialize_final_refit_binding(
        resolved_model_config=prepared.resolved_model_config, input_spec=input_spec, training_config=training_config,
        frozen_run_provenance=provenance, config_hash=config.sha256, registry_hash=report.module_registry_hash,
        source_snapshot_hash=evidence["source_snapshot_sha256"],
        manual_selection_hash=evidence["manual_selection_sha256"], oof_evidence_hash=evidence["oof_evidence_sha256"],
    )
    plan = FinalRefitPlan(
        purpose=str(purpose), config_hash=config.sha256, model_id=machine_id, participant_ids=tuple(
            evidence["participant_ids"]
        ), training_seeds=seeds, fixed_epochs=None if classical else int(training_config.fixed_epochs),
        epoch_rule="not_applicable" if classical else "fixed_epoch",
        model_family="classical_or_rocket" if classical else "deep", oof_evidence_hash=evidence["oof_evidence_sha256"],
        model_kind="ensemble" if ensemble else "single_model", registry_hash=report.module_registry_hash,
        source_snapshot_hash=evidence["source_snapshot_sha256"],
        manual_selection_hash=evidence["manual_selection_sha256"],
        resolved_model_config_hash=binding.resolved_model_config_hash,
        architecture_parameters_hash=binding.architecture_parameters_hash, input_schema_hash=binding.input_schema_hash,
        training_config_hash=binding.training_config_hash,
        frozen_run_provenance_hash=binding.frozen_run_provenance_hash, representation_mode=config.representation_mode,
    )
    execution = execute_full_cohort_refit(
        plan, UnifiedTrainer(training_config), dataset, registry_hash=report.module_registry_hash, binding=binding,
        model_factory=None if classical else prepared, estimator=prepared() if classical else None,
    )
    target = paths.output_path(bundle_directory).resolve()
    publication = _canonical_final_bundle_materialization(execution, materialized, config, report, evidence, paths)
    return save_final_refit_bundle(execution, target, materialization=publication).directory

def _canonical_final_bundle_materialization(
    execution: Any, materialized: _TrustedFull29Materialization, config: Any, report: Any, preflight: Mapping[str, Any],
    paths: Any,
) -> Any:
    """Build the reusable inference boundary for a completed full-cohort fit."""
    from .bundle import build_model_input_adapter
    from .training.bundle import (
        FrozenRepresentationTransformArchive, _bind_trusted_final_bundle_materialization, current_runtime_environment,
    )

    del paths
    payload = config.to_dict()
    mode = str(config.representation_mode)
    spec = materialized.input_spec
    channels = tuple(spec.feature_names) if mode == "feature_vector" else tuple(spec.channel_schema)
    model = dict(execution.binding.resolved_model_config)
    transforms = FrozenRepresentationTransformArchive(
        mode, execution.plan.input_schema_hash, execution.plan.participant_ids, materialized.fitted_objects,
        materialized.representation_provenance, materialized.source_records_hash, materialized.dataset_hash,
    )
    roles = tuple(map(str, payload["training"]["classifier_role_families"]))
    adapter = build_model_input_adapter(
        mode, input_schema_hash=execution.plan.input_schema_hash, allowed_role_families=roles
    )
    not_input = {"status": "not_model_input", "representation_mode": mode}
    window = payload["windows"]["engineering" if mode in {"feature_vector", "feature_matrix"} else "raw_dl"]
    identity = execution.plan.input_schema_hash
    metadata = {
        "model_identity": {
            "name": model["canonical_model_name"], "machine_id": model["model_id"],
            "version": payload["model"]["variant"]
        }, "representation_mode": mode, "signal_route": {"status": "configured", "artifact": payload["artifact"]},
        "class_order": list(payload["manifest"]["class_id_order"]), "channel_schema": list(channels), "preprocessing": {
            "signal": payload["signal"], "quality": payload["quality"],
            "source_records": list(materialized.source_records)
        }, "preprocessing_hash": materialized.preprocessing_hash, "resampling": {
            "canonical_fs_hz": payload["signal"]["internal_fs_hz"], "dl_input": payload["signal"]["dl_resampling"]
        }, "window_plan": {"selected": window, "shared_planner_version": payload["windows"]["shared_planner_version"]},
        "feature_registry": {
            "registry_id": payload["features"]["registry_id"], "schema": materialized.feature_schema_id
        }, "feature_hash": materialized.feature_hash,
        "feature_vector_schema": {"status": "model_input", "names": list(spec.feature_names)}
        if mode == "feature_vector" else not_input,
        "ordered_matrix_schema": {"status": "model_input", "channels": list(spec.channel_schema)}
        if mode == "feature_matrix" else not_input, "mask_semantics": {
            "boundary": "model_ready_input", "mask_required": mode != "feature_vector"
        }, "validity_policy": {"quality_mode": payload["quality"]["mode"], "classifier_role_families": list(roles)},
        "fitted_objects": [{"name": "final_model", "provenance": asdict(execution.result.provenance)}],
        "representation_state": {
            "dataset_hash": materialized.dataset_hash, "source_records_hash": materialized.source_records_hash,
            "input_schema_hash": identity
        }, "pooling_rule": {
            "model": payload["model"].get("pooling", "model_defined"),
            "window_to_file": payload["aggregation"]["window_to_file"]
        }, "aggregation_rule": payload["aggregation"]["balance_line"], "manifest_hash": report.manifest_hash,
        "fold_hash": execution.scope.fold_hash, "manifest_version": payload["manifest"]["manifest_version"],
        "fold_registry_version": payload["splits"]["registry_id"], "pipeline_generation": "final_pipeline_v2",
        "config_hash": config.sha256, "balance_hash": _runtime_imports()["stable_payload_sha256"]({
            "training": payload["training"]["training_balance"], "aggregation": payload["aggregation"]
        }), "run_hash": execution.execution_hash, "source_snapshot_hash": preflight["source_snapshot_sha256"],
        "code_version": _code_version(), "environment": current_runtime_environment(), "dependency_status": {
            "status": "captured_at_refit"
        }, "selection_record_file_sha256": preflight["selection_record_file_sha256"],
        "serialization_trust": {"trusted_local_only": True, "integrity": "reload_golden_parity"}, "golden_case": {
            "boundary": transforms.boundary, "source_identity": asdict(materialized.dataset.identities[0]),
            "expected_output": "three_class_probability"
        },
    }
    return _bind_trusted_final_bundle_materialization(
        execution, metadata=metadata, golden_inputs=materialized.golden_inputs, transforms=transforms,
        pipeline_adapter=adapter, source_records_hash=materialized.source_records_hash,
    )

DEFAULT_ENSEMBLE_MEMBER_SEEDS = (50042, 60042, 70042, 80042, 90042)
FINAL_ENSEMBLE_MEMBER_SEEDS = DEFAULT_ENSEMBLE_MEMBER_SEEDS

def final_refit_policy(config: Any) -> dict[str, Any]:
    """Return the frozen post-selection policy without fitting or exporting."""
    from .models import normalize_model_id

    model = config.section("model")
    model_id = str(model["model_id"])
    _, machine_id = normalize_model_id(model_id)
    orchestration_seed = int(config.section("training")["seed"])
    if not _model_is_ensemble(machine_id):
        refit = {"kind": "single_model", "model_seed": orchestration_seed, "member_seeds": [orchestration_seed]}
    else:
        declared = _ensemble_member_seed_roster(model)
        refit = {
            "kind": "probability_ensemble", "model_seed": None, "orchestration_seed": orchestration_seed,
            "member_seeds": list(declared),
        }
    return {
        "schema_version": "ppg_frailty.final_refit_policy.v2", "pipeline_generation": "final_pipeline_v2",
        "config_id": config.config_id, "config_hash": config.sha256,
        "trigger": "resolved_case_refit_enabled", "fit_scope": "complete_eligible_cohort_from_scratch",
        "performance_claim": "none_full_refit_uses_oof_as_only_internal_performance_evidence", "refit": refit,
        "training_executed": False,
    }

__all__ = [
    "build_comparison_archive_from_run_directories", "ExperimentResult", "DEFAULT_ENSEMBLE_MEMBER_SEEDS",
    "FINAL_ENSEMBLE_MEMBER_SEEDS", "execute_final_refit", "final_refit_preflight", "final_refit_policy",
    "run_full_experiment",
    "run_legacy_bridge_outer_cell", "run_outer_cell", "run_reduced_fold_experiment",
]
