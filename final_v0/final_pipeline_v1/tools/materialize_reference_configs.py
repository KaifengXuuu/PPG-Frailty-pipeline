#!/usr/bin/env python3
"""生成完全展开的参考配置 / Materialize fully resolved reference configs.

中文：本工具只负责生成声明式 YAML。运行时绝不从本工具读取默认值；每份输出配置都
包含完整字段，因此归档后的单个 YAML 足以复现实验语义。

English: This tool only materializes declarative YAML. Runtime code never imports
defaults from here; every emitted file is self-contained and fully resolved.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def _base_config() -> dict[str, Any]:
    """返回显式静态参考配置 / Return the explicit static reference payload."""

    return {
        "schema_version": "ppg_frailty.pipeline_config.v1",
        "config_id": "reference_static_raw_compactcnn_v1",
        "manifest": {
            "path": "manifests/internal_records_v1.csv",
            "manifest_version": "frailty3_internal_manifest_v1",
            "source_dataset_id": "frailty3_m2_20260815_a054800abda272f6",
            "source_manifest_sha256": "bd429ae9c56974ba9ffcb924dfbad0ed930f7d2d47418365754a1929ada06e90",
            "expected_record_count": 261,
            "expected_participant_count": 29,
            "class_id_order": [0, 1, 2],
            "class_name_order": ["Pre-Frail", "Robust/Non-Frail", "Young"],
            "channel_order": ["RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ"],
            "allow_qc_excluded_records": False,
        },
        "splits": {
            "path": "splits/sgkf5_repeats_v1.csv",
            "registry_id": "frailty3_future_corrected_sgkf5_v2",
            "source_registry_file_sha256": "c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c",
            "source_registry_payload_sha256": "0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46",
            "n_splits": 5,
            "n_repeats": 5,
            "split_seeds": [42, 10042, 20042, 30042, 40042],
            "runtime_recompute": False,
        },
        "output": {
            "root": "artifacts/runs",
            "overwrite_existing": False,
            "strict_json": True,
            "write_parquet": True,
            "parquet_missing_dependency_action": "fail_closed",
            "write_window_oof": True,
            "write_file_oof": True,
            "write_subject_oof": True,
            "write_member_oof": False,
        },
        "representation_mode": "raw",
        "roles": ["B", "R1", "R2", "R3", "R4"],
        "signal": {
            "internal_fs_hz": 400.0,
            "channel_order": ["RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ"],
            "ppg_native_unit": "raw_counts",
            "accelerometer_input_unit": "g",
            "gyroscope_input_unit": "deg/s",
            "ppg_filter": {
                "family": "butterworth_sos",
                "order": 3,
                "low_hz": 0.2,
                "high_hz": 8.0,
                "phase": "zero_phase",
                "short_signal_policy": "reject",
                "notch_enabled": False,
            },
            "analysis_view": {
                "direct_source": "x_filter_0p2_to_8hz",
                "non_identity_source": "aligned_x_ar",
                "non_identity_semantics": "rate_only",
                "additional_filter": "none",
            },
            "gap_repair": {
                "method": "linear_inside_only",
                "max_gap_samples": 100,
                "edge_extrapolation": False,
                "all_missing_channel_action": "reject_record",
            },
            "imu": {
                "gravity_method": "quaternion_error_state_ekf",
                "initialization": "online_no_precalibration",
                "comparison_method": "lowpass_0p3hz",
                "sensor_lowpass_acc_hz": 20.0,
                "sensor_lowpass_gyro_hz": 40.0,
                "gravity_lowpass_hz": 0.3,
                "output_units": {"acceleration": "m/s^2", "gyroscope": "rad/s", "jerk": "m/s^3"},
                "required_axes": 6,
                "failure_action": "fail_closed",
            },
            "dl_resampling": {
                "enabled": False,
                "target_fs_hz": 400.0,
                "method": "polyphase_anti_alias",
                "preserve_feature_grid_hz": 400.0,
            },
            "normalization": {
                "raw_ppg": "per_window_median_iqr",
                "raw_imu": "outer_train_fold_robust_scaler",
                "iqr_fallback": "median_absolute_deviation_then_one",
                "clip_after_scale": None,
            },
        },
        "windows": {
            "engineering": {
                "length_s": 10.0,
                "hop_s": 5.0,
                "end_alignment": "left_start_regular_grid",
                "padding": "none_complete_windows_only",
                "cap_per_file": None,
            },
            "raw_dl": {
                "length_s": 5.0,
                "hop_s": 2.5,
                "end_alignment": "include_right_aligned_if_distinct",
                "padding": "none_complete_windows_only",
                "min_valid_fraction": 1.0,
                "cap_per_file": 128,
            },
            "shared_planner_version": "window_plan_v1",
        },
        "quality": {
            "calibrator": "outer_train_empirical_quantiles_v1",
            "fit_scope": "outer_training_participants_only",
            "rate_threshold": 0.5,
            "morph_threshold": 0.65,
            "cardiac_band_hz": [0.5, 3.0],
            "peak_density_bpm_range": [30.0, 200.0],
            "ppi_range_s": [0.3, 2.0],
            "long_gap_max_samples": 100,
            "flatline_duration_s": 1.0,
            "components": [
                "cardiac_band_concentration",
                "periodicity_peak_density",
                "ppi_stability",
                "red_ir_agreement",
                "imu_motion_energy",
                "flatline_exclusion",
                "clipping_saturation_exclusion",
                "long_gap_exclusion",
            ],
            "high_quality_rule": "q_rate_pass_and_q_morph_pass",
            "failure_action": "fail_closed",
        },
        "artifact": {
            "reducer": "identity",
            "reducer_version": "identity_v1",
            "selection_scope": "run_before_evaluation",
            "degraded_policy": "drop",
            "motion_detector_enabled": False,
            "non_identity_output_contract": "rate_only",
            "failure_action": "no_result_no_fallback",
            "parameters": {},
        },
        "features": {
            "registry_id": "feature_registry_v1",
            "file_vector_schema": "FeatureVectorV1",
            "engineering_sequence_schema": "EngineeringFeatureSequenceV1",
            "matrix_schema": "OrderedFeatureMatrixV1",
            "matrix_k": 32,
            "technical_metadata_allowed": False,
            "missing_physiology_encoding": "nan_and_validity_false",
            "time_prv_min_duration_s": 60.0,
            "time_prv_min_accepted_peaks": 5,
            "spectral_prv_min_duration_s": 300.0,
            "spectral_prv_min_intervals": 200,
            "tachogram_fs_hz": 4.0,
            "spectral_bands_hz": {"vlf": [0.003, 0.04], "lf": [0.04, 0.15], "hf": [0.15, 0.4]},
            "sample_entropy": {"m": 2, "r_sd_fraction": 0.2, "min_intervals": 200},
            "file_aggregation": ["mean", "population_sd"],
        },
        "model": {
            "model_id": "CompactCNN1D",
            "variant": "legacy_reference_not_wang_fcn",
            "input_channels": 8,
            "input_channels_resolution": "fixed_config_value",
            "n_classes": 3,
            "ensemble_size": 1,
            "member_seeds": [42],
            "shapeformer_discovery_method": "not_applicable",
            "rocket_kernel_count": 0,
            "rocket_ridge_alpha": None,
            "mask_aware_pooling": True,
        },
        "training": {
            "epoch_rule": "fixed_epoch",
            "fixed_epochs": 50,
            "outer_labels_visible_to_trainer": False,
            "inner_grouped_folds": 0,
            "refit_on_all_outer_training": True,
            "batch_size": 64,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "class_weighting": "outer_train_inverse_frequency",
            "sampler": "participant_file_window_balanced_v1",
            "loss": "cross_entropy",
            "deterministic_algorithms": True,
            "num_workers": 0,
            "device": "cpu",
            "cache_policy": "content_addressed_strict",
        },
        "aggregation": {
            "hierarchy": ["window", "file", "role", "participant"],
            "window_to_file": "ordinary_mean",
            "file_to_role": "ordinary_mean",
            "role_to_participant": "equal_available_roles",
            "missing_role_policy": "renormalize_over_available_roles",
            "quality_weighting": False,
            "direct_all_window_participant_mean": False,
        },
        "evaluation": {
            "unit": "participant",
            "primary_metric": "balanced_accuracy",
            "metrics": [
                "balanced_accuracy",
                "macro_f1",
                "per_class_precision_recall_f1",
                "worst_class_recall",
                "worst_class_f1",
                "confusion_matrix",
                "coverage",
            ],
            "confidence_interval": "student_t_95_between_repeats",
            "paired_delta_key": ["repeat_index", "fold_index", "participant_id"],
            "rank_incomplete_configs": False,
            "independent_test_available": False,
            "metric_prefix": "oof_validation_",
            "calibration_metrics": ["multiclass_brier", "expected_calibration_error"],
        },
    }


def _resolved_variants() -> dict[str, dict[str, Any]]:
    """构造四份完全展开 payload / Build four fully expanded payloads."""

    static = _base_config()

    all_roles = copy.deepcopy(static)
    all_roles["config_id"] = "reference_all_roles_raw_inception_v1"
    all_roles["roles"] = ["B", "R1", "R2", "R3", "R4", "S1", "S2", "W1", "W2"]
    all_roles["model"].update({
        "model_id": "InceptionTimeFull",
        "variant": "single_network",
        "member_seeds": [42],
    })

    motion = copy.deepcopy(all_roles)
    motion["config_id"] = "motion_benchmark_spectral_rate_only_v1"
    motion["roles"] = ["S1", "S2", "W1", "W2"]
    motion["artifact"].update({
        "reducer": "spectral_mask",
        "reducer_version": "spectral_mask_v1",
        "degraded_policy": "denoise_then_extract_rate_features",
        "motion_detector_enabled": True,
        "parameters": {
            "stft_window_s": 4.0,
            "stft_hop_s": 1.0,
            "imu_mask_quantile": 0.75,
            "mask_strength": 0.8,
            "preserve_band_hz": [0.5, 3.0],
        },
    })
    motion["representation_mode"] = "feature_vector"
    motion["model"].update({
        "model_id": "LogisticRegressionL2",
        "variant": "rate_only_motion_file_vector",
        "input_channels": 0,
        "input_channels_resolution": "not_applicable_feature_vector",
        "member_seeds": [42],
    })

    matrix = copy.deepcopy(static)
    matrix["config_id"] = "feature_matrix_inception_single_v1"
    matrix["representation_mode"] = "feature_matrix"
    matrix["model"].update({
        "model_id": "InceptionTimeMatrix",
        "variant": "single_network_mask_aware",
        "input_channels": 0,
        "input_channels_resolution": "derive_exactly_from_frozen_matrix_channel_schema",
        "member_seeds": [42],
    })

    return {
        "reference_static_v1.yaml": static,
        "reference_all_roles_v1.yaml": all_roles,
        "motion_benchmark_v1.yaml": motion,
        "feature_matrix_v1.yaml": matrix,
    }


def main() -> int:
    """原子式写出稳定 YAML / Atomically write stable YAML files."""

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for filename, payload in _resolved_variants().items():
        destination = CONFIG_DIR / filename
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        temporary.write_text(text, encoding="utf-8", newline="\n")
        temporary.replace(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
