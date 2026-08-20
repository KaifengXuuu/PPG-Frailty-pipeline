"""V2 模块注册表与严格配置适配 / V2 module registry and strict adapters.

中文：本模块是 CLI、主流水线与对照测试共享的唯一模块目录。适配器只接受
逐字段声明且语义完全一致的配置；未知 reducer、未知参数或不兼容的
representation/model 组合均关闭失败，不做静默别名替换。

English: This is the sole module catalogue shared by the CLI, pipeline, and
comparisons.  Adapters accept only explicitly declared, semantically exact fields;
unknown reducers, parameters, and representation/model combinations fail closed.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ModuleDescriptor:
    """一个可审计模块条目 / One auditable module entry."""

    module_id: str
    family: str
    implementation: str
    representation_modes: tuple[str, ...]
    scientific_status: str
    quantitative_suite: str
    notes: str
    runtime_dependencies: tuple[str, ...] = ()


REPRESENTATION_MODULES = (
    ModuleDescriptor("raw", "representation", "ppg_frailty.representations.raw", ("raw",), "reference", "integration", "Line A window->file->participant; Line B window->file->role_family->participant"),
    ModuleDescriptor("feature_vector", "representation", "ppg_frailty.representations.feature_vector", ("feature_vector",), "reference", "integration", "one vector per recording"),
    ModuleDescriptor("feature_matrix", "representation", "ppg_frailty.representations.feature_matrix", ("feature_matrix",), "reference", "integration", "one registry-derived D-by-configured-K matrix per recording"),
    ModuleDescriptor("fusion", "representation", "ppg_frailty.representations.fusion", ("fusion",), "reference", "integration", "file-level window pooling then one vector concatenation"),
)

_ALL_REPRESENTATION_MODES = ("raw", "feature_vector", "feature_matrix", "fusion")

NORMALIZATION_MODULES = (
    ModuleDescriptor("ppg_per_window_robust", "normalization", "ppg_frailty.normalization.RawNormalizationConfig", ("raw", "fusion"), "runtime_selectable", "normalization", "runtime strategy per_window_robust dispatched by ppg_frailty.representations.raw; median/IQR with configurable fallback and scale constants"),
    ModuleDescriptor("ppg_per_window_standard_zscore", "normalization", "ppg_frailty.normalization.RawNormalizationConfig", ("raw", "fusion"), "runtime_selectable", "normalization", "runtime strategy per_window_standard_zscore dispatched by ppg_frailty.representations.raw with configurable ddof and epsilon"),
    ModuleDescriptor("ppg_none", "normalization", "ppg_frailty.normalization.RawNormalizationConfig", ("raw", "fusion"), "runtime_selectable_identity", "normalization", "runtime PPG strategy none dispatched by ppg_frailty.representations.raw; no optical scaling"),
    ModuleDescriptor("imu_outer_train_robust", "normalization", "ppg_frailty.normalization.RawNormalizationConfig", ("raw", "fusion"), "runtime_selectable_fold_local", "normalization", "runtime strategy outer_train_robust dispatched by ppg_frailty.representations.imu_transform; statistics fit only on outer-train rows"),
    ModuleDescriptor("imu_outer_train_mean_std", "normalization", "ppg_frailty.normalization.RawNormalizationConfig", ("raw", "fusion"), "runtime_selectable_fold_local", "normalization", "runtime strategy outer_train_mean_std dispatched by ppg_frailty.representations.imu_transform; statistics fit only on outer-train rows"),
    ModuleDescriptor("imu_none", "normalization", "ppg_frailty.normalization.RawNormalizationConfig", ("raw", "fusion"), "runtime_selectable_identity", "normalization", "runtime IMU strategy none dispatched by ppg_frailty.representations.imu_transform; no fold-local scaling"),
)

PPG_FILTER_MODULES = (
    ModuleDescriptor(
        "butterworth_sos",
        "ppg_filter",
        "ppg_frailty.signal.preprocess.materialize_signal_preprocessing_config",
        _ALL_REPRESENTATION_MODES,
        "runtime_parameterized_filter",
        "signal",
        "executable PPG filter family; order, passband, phase, notch and short-signal policy are validated config parameters rather than separate profile IDs",
    ),
)

GAP_REPAIR_MODULES = (
    ModuleDescriptor(
        "linear_inside_only",
        "gap_repair",
        "ppg_frailty.signal.preprocess.materialize_signal_preprocessing_config",
        _ALL_REPRESENTATION_MODES,
        "runtime_parameterized_internal_gap_repair",
        "signal",
        "executable finite internal-gap interpolation with configurable max_gap_samples; edge extrapolation remains an explicit data-integrity boundary",
    ),
)

IMU_GRAVITY_MODULES = tuple(
    ModuleDescriptor(
        module_id,
        "imu_gravity",
        "ppg_frailty.signal.preprocess.materialize_signal_preprocessing_config",
        _ALL_REPRESENTATION_MODES,
        (
            "runtime_selectable_legacy_parallel"
            if module_id in {"quaternion_error_state_ekf", "low_pass_0p3hz"}
            else "runtime_selectable_calibrated"
        ),
        "signal",
        "executable gravity-separation profile with profile-specific numerical parameters and no silent fallback",
    )
    for module_id in (
        "calibrated_roll_pitch_ekf",
        "profile_a_lowpass_0p3hz",
        "quaternion_error_state_ekf",
        "low_pass_0p3hz",
    )
)

DL_RESAMPLING_MODULES = (
    ModuleDescriptor(
        "off_identity_source_grid",
        "dl_resampling",
        "ppg_frailty.signal.resample.validate_dl_resampling_config",
        _ALL_REPRESENTATION_MODES,
        "runtime_selectable_identity",
        "signal",
        "disabled branch preserves the canonical source grid and does not create a second DL tensor",
    ),
    ModuleDescriptor(
        "polyphase_anti_alias",
        "dl_resampling",
        "ppg_frailty.signal.resample.prepare_configured_dl_input",
        ("raw", "fusion"),
        "runtime_parameterized_optional_dl_view",
        "signal",
        "enabled branch accepts any validated positive target no higher than the source grid; named fixed-kernel rates are catalog presets only",
    ),
)

WINDOW_PROFILE_MODULES = (
    ModuleDescriptor(
        "engineering",
        "window_profile",
        "ppg_frailty.module_registry.resolve_window_config",
        ("feature_vector", "feature_matrix", "fusion"),
        "runtime_parameterized_shared_planner_profile",
        "signal",
        "engineering window length, hop, alignment, padding, cap and valid-fraction controls share the registered planner",
    ),
    ModuleDescriptor(
        "raw_dl",
        "window_profile",
        "ppg_frailty.module_registry.resolve_window_config",
        ("raw", "fusion"),
        "runtime_parameterized_shared_planner_profile",
        "signal",
        "raw/fusion window length, hop, alignment, mask-aware padding, cap and valid-fraction controls share the registered planner",
    ),
)

OPTIMIZER_MODULES = tuple(
    ModuleDescriptor(
        module_id,
        "optimizer",
        "ppg_frailty.training.trainer.resolve_optimizer_parameters",
        ("raw", "feature_matrix", "fusion"),
        "runtime_selectable_torch_optimizer",
        "training",
        f"executable {module_id} strategy; optimizer-specific controls are materialized before UnifiedTrainer constructs torch.optim",
        ("torch",),
    )
    for module_id in ("adam", "adamw", "sgd", "rmsprop")
)

SAMPLER_MODULES = tuple(
    ModuleDescriptor(
        module_id,
        "sampler",
        "ppg_frailty.training.trainer.configured_row_sampling_weights",
        (
            ("raw", "feature_matrix", "fusion")
            if module_id == "uniform_replacement"
            else _ALL_REPRESENTATION_MODES
        ),
        (
            "runtime_selectable_torch_replacement_sampler"
            if module_id == "uniform_replacement"
            else "runtime_selectable"
        ),
        "training",
        (
            "executable replacement draw strategy for torch data loaders; estimators fail fast because uniform sample weights cannot encode replacement draws"
            if module_id == "uniform_replacement"
            else f"executable {module_id} row-distribution strategy shared by deep loaders and estimator sample weights"
        ),
        (("torch",) if module_id == "uniform_replacement" else ()),
    )
    for module_id in (
        "balance_line_weighted_v2",
        "uniform_replacement",
        "exhaustive_shuffle_without_replacement",
        "subject_balanced",
        "class_subject_balanced",
    )
)

LOSS_MODULES = tuple(
    ModuleDescriptor(
        module_id,
        "loss",
        "ppg_frailty.training.trainer.TrainingClassificationLoss",
        ("raw", "feature_matrix", "fusion"),
        "runtime_selectable_torch_loss",
        "training",
        f"executable {module_id} classification-loss strategy; weighted_ce is an input alias of cross_entropy, not a parallel module",
        ("torch",),
    )
    for module_id in ("cross_entropy", "balanced_softmax", "focal_loss")
)

CLASS_WEIGHTING_MODULES = tuple(
    ModuleDescriptor(
        module_id,
        "class_weighting",
        "ppg_frailty.training.trainer.configured_class_weight_vector",
        _ALL_REPRESENTATION_MODES,
        "runtime_selectable_single_weighting_entry",
        "training",
        f"executable {module_id} strategy; model.class_weight is not a second weighting capability",
    )
    for module_id in (
        "inverse_frequency",
        "effective_number",
        "none",
    )
)

CLASS_COUNT_BASIS_MODULES = tuple(
    ModuleDescriptor(
        module_id,
        "class_count_basis",
        "ppg_frailty.training.trainer.outer_train_class_counts",
        _ALL_REPRESENTATION_MODES,
        "runtime_selectable_outer_train_statistical_unit",
        "training",
        f"executable {module_id} count basis shared by inverse-frequency, effective-number, and balanced-softmax corrections",
    )
    for module_id in ("participant", "row")
)

TRAINING_BALANCE_MODULES = tuple(
    ModuleDescriptor(
        module_id,
        "training_balance",
        "ppg_frailty.training.aggregation.aggregation_rule_for_training_balance",
        _ALL_REPRESENTATION_MODES,
        "runtime_selectable_independent_of_reporting_aggregation",
        "training",
        (
            f"executable {module_id} hierarchy used by configured sampling and "
            "train/inner participant balanced-accuracy; reporting Line A/B is "
            "selected independently"
        ),
    )
    for module_id in ("equal_files", "equal_role_families")
)

EPOCH_SELECTION_MODULES = tuple(
    ModuleDescriptor(
        module_id,
        "epoch_selection",
        "ppg_frailty.training.trainer.UnifiedTrainer",
        ("raw", "feature_matrix", "fusion"),
        "runtime_selectable_deep_epoch_strategy",
        "training",
        f"executable {module_id} branch in UnifiedTrainer; inner selection remains outer-train-only",
        ("torch",),
    )
    for module_id in ("fixed_epoch", "inner_grouped_selection")
)

QUALITY_MODE_MODULES = tuple(
    ModuleDescriptor(
        module_id,
        "quality_mode",
        "ppg_frailty.quality.routing.run_quality_mode",
        _ALL_REPRESENTATION_MODES,
        "runtime_selectable_no_gate",
        "quality",
        f"executable quality mode {module_id}; no readiness or authorization gate",
    )
    for module_id in ("off", "diagnostics_only", "route")
)

WINDOW_QUALITY_SELECTION_MODULES = (
    ModuleDescriptor(
        "none",
        "window_quality_selection",
        "ppg_frailty.quality.window_selection.select_raw_windows",
        ("raw", "fusion"),
        "runtime_selectable_noop",
        "quality",
        "retains every raw window without computing SQI",
    ),
    ModuleDescriptor(
        "legacy_per_file_top_fraction",
        "window_quality_selection",
        "ppg_frailty.quality.window_selection.select_raw_windows",
        ("raw", "fusion"),
        "runtime_selectable_label_free_per_file_rank",
        "quality",
        "computes legacy_cardiac_motion_window_sqi_v1 independently per file and retains ceil(n*keep_fraction) windows; application_scope independently selects train-only, all-partitions, or legacy-style per-file heldout selection inside the V2 hierarchy",
    ),
)

SHAPEFORMER_DISCOVERY_BALANCE_MODULES = tuple(
    ModuleDescriptor(
        module_id,
        "shapeformer_discovery_balance",
        "ppg_frailty.models.pisd_port.discover_pisd_shapelets",
        ("raw",),
        "runtime_selectable_outer_train_discovery_sampler",
        "models",
        (
            "class/participant/file/window hierarchical discovery"
            if module_id == "participant_file_balanced"
            else "legacy class-balanced random-window discovery"
        ),
        ("torch",),
    )
    for module_id in ("participant_file_balanced", "class_window_balanced")
)

AGGREGATION_MODULES = tuple(
    ModuleDescriptor(
        module_id,
        "aggregation",
        "ppg_frailty.training.aggregation.aggregate_hierarchy",
        _ALL_REPRESENTATION_MODES,
        "runtime_selectable_parallel_reporting_line",
        "aggregation",
        f"executable {module_id} hierarchy ending in participant-balanced output",
    )
    for module_id in ("line_a_equal_files", "line_b_equal_role_families")
)

QUALITY_WEIGHT_SOURCE_MODULES = (
    ModuleDescriptor(
        "none",
        "quality_weight_source",
        "ppg_frailty.training.aggregation.aggregate_hierarchy",
        _ALL_REPRESENTATION_MODES,
        "runtime_selectable_unweighted_hierarchy",
        "aggregation",
        "ordinary means at every selected hierarchy edge",
    ),
    ModuleDescriptor(
        "route_file_q_rate",
        "quality_weight_source",
        "ppg_frailty.training.aggregation.aggregate_hierarchy",
        _ALL_REPRESENTATION_MODES,
        "runtime_selectable_file_level_endpoint_weight",
        "aggregation",
        "consumes route Q_rate from file-to-role/participant; raw window-to-file remains ordinary mean",
    ),
    ModuleDescriptor(
        "legacy_window_sqi",
        "quality_weight_source",
        "ppg_frailty.training.aggregation.aggregate_hierarchy",
        ("raw",),
        "runtime_selectable_row_aligned_legacy_weight",
        "aggregation",
        "consumes migrated per-window cardiac/motion scores at window-to-file and propagates file summaries upward",
    ),
)

FEATURE_GROUP_MODULES = tuple(
    ModuleDescriptor(
        module_id,
        "feature_group",
        "ppg_frailty.features.registry.registry_for_groups",
        ("feature_vector", "feature_matrix", "fusion"),
        "runtime_selectable_composable_group",
        "features",
        (
            "selected through features.enabled_groups; the same content-addressed "
            "registry drives extraction, fold transforms, fusion tensors, ordered "
            "matrix context channels, validation, experiments, and final refit"
        ),
    )
    for module_id in (
        "ppi_basic_rate",
        "hrv_time_domain",
        "hrv_spectral",
        "hrv_nonlinear",
        "morphology",
        "dual_optical",
        "engineering_summary",
    )
)

ARTIFACT_MODULES = (
    ModuleDescriptor("identity", "artifact", "ppg_frailty.artifact.identity.IdentityReducer", ("raw", "feature_vector", "feature_matrix", "fusion"), "direct_control", "artifacts", "exact no-op; morphology remains eligible"),
    ModuleDescriptor("nlms_imu_anc", "artifact", "ppg_frailty.artifact.nlms.NlmsReducer", ("feature_vector", "feature_matrix"), "comparison_rate_only", "artifacts", "ANC assumption may remove physiological response; rate-only endpoints enter vector or ordered-matrix validity channels"),
    ModuleDescriptor("ssa_decomposition", "artifact", "ppg_frailty.artifact.decomposition.SsaReducer", ("feature_vector", "feature_matrix"), "comparison_rate_only", "artifacts", "non-stationary decomposition comparator; rate-only endpoints enter vector or ordered-matrix validity channels"),
    ModuleDescriptor("spectral_mask", "artifact", "ppg_frailty.artifact.spectral.SpectralMaskReducer", ("feature_vector", "feature_matrix"), "comparison_rate_only", "artifacts", "formal STFT plus IMU soft mask; rate-only endpoints enter vector or ordered-matrix validity channels"),
    ModuleDescriptor("pca_bss", "artifact", "ppg_frailty.artifact.bss.PcaBssReducer", ("feature_vector", "feature_matrix"), "comparison_rate_only", "artifacts", "two-wavelength BSS comparator; rate-only endpoints enter vector or ordered-matrix validity channels"),
    ModuleDescriptor("fastica_bss", "artifact", "ppg_frailty.artifact.bss.FastIcaBssReducer", ("feature_vector", "feature_matrix"), "comparison_rate_only", "artifacts", "two-wavelength BSS comparator; rate-only endpoints enter vector or ordered-matrix validity channels"),
    ModuleDescriptor("nmf_bss", "artifact", "ppg_frailty.artifact.bss.NmfBssReducer", ("feature_vector", "feature_matrix"), "comparison_rate_only", "artifacts", "two-wavelength spectral BSS comparator; rate-only endpoints enter vector or ordered-matrix validity channels"),
    ModuleDescriptor("emd_sifting_rate_only", "artifact", "ppg_frailty.artifact.legacy.EmdSiftingRateOnlyReducer", ("feature_vector", "feature_matrix"), "comparison_rate_only", "artifacts", "named EMD sifting ablation; never morphology-preserving; rate-only endpoints enter vector or ordered-matrix validity channels"),
    ModuleDescriptor("ceemd_lite_nlms_legacy", "artifact", "ppg_frailty.artifact.legacy.CeemdLiteNlmsLegacyReducer", ("feature_vector", "feature_matrix"), "comparison_rate_only", "artifacts", "named CEEMD-lite plus NLMS legacy ablation; rate-only endpoints enter vector or ordered-matrix validity channels"),
    ModuleDescriptor("dwt_a2_legacy", "artifact", "ppg_frailty.artifact.legacy.DwtA2LegacyReducer", ("feature_vector", "feature_matrix"), "comparison_rate_only", "artifacts", "named DWT A2 legacy ablation; PyWavelets is optional; rate-only endpoints enter vector or ordered-matrix validity channels"),
)

PRV_BACKEND_MODULES = (
    ModuleDescriptor("local", "prv_backend", "ppg_frailty.features.prv_backend_compare.evaluate_prv_backend", ("feature_vector", "feature_matrix", "fusion"), "formal_primary", "prv_backend_output", "local manual PRV remains the only classifier-pipeline backend"),
    ModuleDescriptor("aura_hrv_analysis", "prv_backend", "ppg_frailty.features.prv_backend_compare.evaluate_prv_backend", (), "function_comparison_only", "prv_backend_output", "fixed PPI vectors only; never enters classifier training"),
    ModuleDescriptor("rhenan_hrv", "prv_backend", "ppg_frailty.features.prv_backend_compare.evaluate_prv_backend", (), "legacy_function_comparison_only", "prv_backend_output", "separate legacy requirements; never enters classifier training"),
)

PEAK_DETECTOR_MODULES = (
    ModuleDescriptor(
        "aboy_project_v1",
        "peak_detector",
        "ppg_frailty.peaks.aboy_project.detect_pulses_per_wavelength_aboy_project",
        ("raw", "feature_vector", "feature_matrix", "fusion"),
        "canonical_project_aboy_inspired",
        "signal",
        "400 Hz, complete non-overlapping 10 s blocks, HRI-adaptive second-order band-pass, dual polarity; configurable min_observation_sec=8.0 and min_peaks=5 defaults; not an exact upstream Aboy++ reproduction",
    ),
    ModuleDescriptor(
        "aboy_project_v2",
        "peak_detector",
        "ppg_frailty.peaks.aboy_project_v2.detect_pulses_per_wavelength_aboy_project_v2",
        ("raw", "feature_vector", "feature_matrix", "fusion"),
        "seven_step_authoritative_ablation",
        "signal",
        "authoritative project seven-step contract: owned 0.2-Hz high-pass, complete non-overlapping 10-s blocks, per-block dual-polarity selection, retained-Pd HRI, ratio peak removal before physiological/MAD cleaning",
    ),
    ModuleDescriptor(
        "dual_polarity_prominence_v1_ablation",
        "peak_detector",
        "ppg_frailty.signal.peaks._detect_pulses_dual_polarity_ablation",
        ("raw", "feature_vector", "feature_matrix", "fusion"),
        "explicit_legacy_ablation_only",
        "signal",
        "numerically preserved whole-record fixed distance/prominence detector; configurable min_observation_sec=8.0 and min_peaks=5 defaults; never a fallback",
    ),
    ModuleDescriptor(
        "msptdfast_v2_3_python_port",
        "peak_detector",
        "ppg_frailty.peaks.msptdfast_v2.detect_msptdfast_v2",
        ("raw", "feature_vector", "feature_matrix", "fusion"),
        "stage_ablation_01_equation_level_port",
        "signal",
        "single registered implementation shared by Stage-ablation-01 and the ordinary pipeline; bound to ppg-beats v2.3 source SHA-256; not bitwise MATLAB parity",
    ),
)

MOTION_EVIDENCE_MODULES = (
    ModuleDescriptor("sqi_only", "motion_evidence", "ppg_frailty.quality.motion.resolve_motion_option", (), "external_audit_control_not_classifier_runtime", "motion_contract", "V2-010 external evidence control; core quality behavior is configured through quality.mode"),
    ModuleDescriptor("sqi_plus_motion_override", "motion_evidence", "ppg_frailty.quality.motion.resolve_motion_option", (), "external_ptt_evidence_protocol_not_classifier_runtime", "motion_contract", "independent motion/PTT audit API only; never a core classifier-pipeline selector"),
    ModuleDescriptor("historical_light_cnn_backup", "motion_evidence", "ppg_frailty.quality.motion.HISTORICAL_LIGHT_CNN_EVIDENCE", (), "historical_frozen_evidence_not_classifier_runtime", "motion_contract", "archived SIM evidence only; never a V2 classifier or PTT result"),
)

COMPARISON_PROFILE_MODULES = (
    ModuleDescriptor(
        "line_b_equal_role_families",
        "comparison_profile",
        "ppg_frailty.v2_contract.resolve_balance_line",
        ("raw", "feature_vector", "feature_matrix", "fusion"),
        "registered_not_run",
        "formal_catalog",
        "matched training and aggregation Line B; never selected automatically",
    ),
    ModuleDescriptor(
        "epoch_7_ablation",
        "comparison_profile",
        "ppg_frailty.training.trainer.TrainingConfig",
        ("raw", "feature_matrix", "fusion"),
        "registered_not_run",
        "formal_catalog",
        "single factor from default fixed epoch 10",
    ),
    ModuleDescriptor(
        "epoch_15_ablation",
        "comparison_profile",
        "ppg_frailty.training.trainer.TrainingConfig",
        ("raw", "feature_matrix", "fusion"),
        "registered_not_run",
        "formal_catalog",
        "single factor from default fixed epoch 10",
    ),
    ModuleDescriptor(
        "direct_filter_0p5_to_5hz_ablation",
        "comparison_profile",
        "ppg_frailty.signal.ppg_preprocess.build_signal_views",
        ("raw", "feature_vector", "feature_matrix", "fusion"),
        "registered_not_run",
        "formal_catalog",
        "only named direct-filter ablation from 0.2 to 8 Hz reference",
    ),
    ModuleDescriptor(
        "imu_lpf_0p3hz_ablation",
        "comparison_profile",
        "ppg_frailty.signal.imu_preprocess.preprocess_imu",
        ("raw", "feature_vector", "feature_matrix", "fusion"),
        "registered_not_run",
        "formal_catalog",
        "independent gravity-separation comparison; never an EKF fallback",
    ),
    ModuleDescriptor(
        "fixed_kernel_samples_resampling_ablation",
        "comparison_profile",
        "ppg_frailty.models.time_scale.build_fixed_kernel_resampling_cases",
        ("raw",),
        "registered_not_run",
        "models",
        "V2-019: CompactCNN/Inception only; 100/160/200 Hz keep kernel sample counts fixed and are not physical-time matched",
    ),
    ModuleDescriptor(
        "fixed_kernel_samples_context_10s_400hz_ablation",
        "comparison_profile",
        "ppg_frailty.models.time_scale.build_fixed_kernel_context_cases",
        ("raw",),
        "registered_not_run",
        "models",
        "10-second input context at 400 Hz with unchanged convolution sample counts",
    ),
    ModuleDescriptor(
        "fixed_kernel_samples_dilation2_ablation",
        "comparison_profile",
        "ppg_frailty.models.time_scale.build_fixed_kernel_dilation_cases",
        ("raw",),
        "registered_not_run",
        "models",
        "dilation 2 with unchanged kernel sample counts",
    ),
    ModuleDescriptor(
        "quality_diagnostics_only",
        "comparison_profile",
        "ppg_frailty.experiment._retain_without_quality_routing",
        ("raw", "feature_vector", "feature_matrix", "fusion"),
        "registered_manual_not_run",
        "formal_catalog",
        "computes diagnostics only; retention, aggregation and prediction stay invariant",
    ),
)

MODEL_MODULES = (
    ModuleDescriptor("CompactCNN1D", "model", "ppg_frailty.models.compact_cnn.CompactCNN1D", ("raw",), "reference_not_wang_fcn", "models", "32/64/128 legacy-reference CNN", ("torch",)),
    ModuleDescriptor("InceptionTimeFull", "model", "ppg_frailty.models.inception_time_port.InceptionTimeSingleNetwork", ("raw",), "single_network", "models", "full single network, not five-member ensemble", ("torch",)),
    ModuleDescriptor("InceptionTimeSmall", "model", "ppg_frailty.models.inception_time_port.InceptionTimeSingleNetwork", ("raw",), "single_network", "models", "small single network", ("torch",)),
    ModuleDescriptor("InceptionTimeMatrix", "model", "ppg_frailty.models.inception_time_port.InceptionTimeSingleNetwork", ("feature_matrix",), "single_network", "models", "mask-aware registry-derived D-by-K input", ("torch",)),
    ModuleDescriptor("InceptionTimeFullFiveMemberEnsemble", "model", "ppg_frailty.models.inception_ensemble.InceptionTimeFiveMemberProbabilityEnsemble", ("raw",), "configurable_probability_ensemble_legacy_name", "models", "one or more independently seeded full raw models and exact probability mean; canonical name retained for compatibility", ("torch",)),
    ModuleDescriptor("InceptionTimeMatrixFiveMemberEnsemble", "model", "ppg_frailty.models.inception_ensemble.InceptionTimeFiveMemberProbabilityEnsemble", ("feature_matrix",), "configurable_probability_ensemble_legacy_name", "models", "one or more independently seeded full matrix models and exact probability mean; canonical name retained for compatibility", ("torch",)),
    ModuleDescriptor("ROCKET", "model", "ppg_frailty.models.rocket_ridge.RocketRidgeClassifier", ("feature_matrix",), "reference_10000_kernels", "models", "fold-local scaler, kernels and ridge"),
    ModuleDescriptor("MiniROCKET", "model", "ppg_frailty.models.rocket_ridge.MiniRocketAblation", ("feature_matrix",), "engineering_ablation_not_reference_port", "models", "cannot replace ROCKET silently"),
    ModuleDescriptor("LogisticRegressionL2", "model", "ppg_frailty.models.feature_models.FeatureVectorBaseline", ("feature_vector",), "reference", "models", "fold-local imputer and scaler"),
    ModuleDescriptor("RBFSVM", "model", "ppg_frailty.models.feature_models.FeatureVectorBaseline", ("feature_vector",), "reference", "models", "fold-local imputer and scaler"),
    ModuleDescriptor("ExtraTrees", "model", "ppg_frailty.models.feature_models.FeatureVectorBaseline", ("feature_vector",), "reference", "models", "fold-local imputer"),
    ModuleDescriptor("ShapeFormerChannelSpecificOSD", "model", "ppg_frailty.models.shapeformer_literature.LiteratureShapeFormerChannelSpecificOSD", ("raw",), "implemented_not_benchmarked_high_compute", "models", "faithful channel-specific OSD/PISD discovery plus ShapeBlock/IG route; never fixed-length and never fallback", ("torch",)),
    ModuleDescriptor("ShapeFormerChannelSpecificScalarDistanceAblation", "model", "ppg_frailty.models.shapeformer.ExperimentalShapeFormer", ("raw",), "optional_scalar_distance_ablation_not_literature_reference", "models", "reuses the fold-local channel-specific OSD bank with the separately selectable scalar-distance downstream module", ("torch",)),
    ModuleDescriptor("ShapeFormerEffectSizeFixedV1", "model", "ppg_frailty.models.shapeformer_port.ExperimentalShapeFormer", ("raw",), "parameterized_effect_size_ablation_legacy_name", "models", "effect-size discovery ablation with configurable positive length/stride; legacy name retains 128/64 defaults and is never labelled PISD", ("torch",)),
    ModuleDescriptor("ShapeFormerLegacyEffectSizePort", "model", "ppg_frailty.models.shapeformer_legacy.LegacyEffectSizeShapeFormer", ("raw",), "legacy_parallel_ablation_not_osd_parity", "models", "isolated historical channel-wise effect-size discovery plus the functional legacy local/shape-token downstream; configurable discovery caps and candidates are executed outer-train only", ("torch",)),
    ModuleDescriptor("FileBagFusionCompact", "model", "ppg_frailty.models.file_fusion.FileBagFusionClassifier", ("fusion",), "reference", "models", "file-level concatenation once", ("torch",)),
    ModuleDescriptor("FileBagFusionInception", "model", "ppg_frailty.models.file_fusion.FileBagFusionClassifier", ("fusion",), "reference", "models", "file-level concatenation once", ("torch",)),
    ModuleDescriptor("FileBagFusion", "model", "ppg_frailty.models.file_fusion.FileBagFusionClassifier", ("fusion",), "optional_composable_signal_encoder", "models", "file-level fusion with one registered raw forward_features encoder; fold-local ShapeFormer discovery remains outer-train only", ("torch",)),
)

ALL_MODULES = (
    REPRESENTATION_MODULES
    + NORMALIZATION_MODULES
    + PPG_FILTER_MODULES
    + GAP_REPAIR_MODULES
    + IMU_GRAVITY_MODULES
    + DL_RESAMPLING_MODULES
    + WINDOW_PROFILE_MODULES
    + OPTIMIZER_MODULES
    + SAMPLER_MODULES
    + LOSS_MODULES
    + CLASS_WEIGHTING_MODULES
    + CLASS_COUNT_BASIS_MODULES
    + TRAINING_BALANCE_MODULES
    + EPOCH_SELECTION_MODULES
    + QUALITY_MODE_MODULES
    + WINDOW_QUALITY_SELECTION_MODULES
    + SHAPEFORMER_DISCOVERY_BALANCE_MODULES
    + AGGREGATION_MODULES
    + QUALITY_WEIGHT_SOURCE_MODULES
    + FEATURE_GROUP_MODULES
    + ARTIFACT_MODULES
    + PRV_BACKEND_MODULES
    + PEAK_DETECTOR_MODULES
    + MOTION_EVIDENCE_MODULES
    + COMPARISON_PROFILE_MODULES
    + MODEL_MODULES
)


def list_modules(family: str = "all") -> list[dict[str, Any]]:
    """稳定排序导出模块 / Export modules in stable order."""

    allowed = {"all", *(item.family for item in ALL_MODULES)}
    if family not in allowed:
        raise ValueError(f"unknown module family: {family}")
    selected = ALL_MODULES if family == "all" else tuple(
        item for item in ALL_MODULES if item.family == family
    )
    return [asdict(item) for item in sorted(selected, key=lambda item: (item.family, item.module_id))]


def registry_sha256() -> str:
    """计算注册表身份 / Hash the complete registry identity."""

    encoded = json.dumps(list_modules(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_peak_detector_config(signal_section: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve one detector plus its public execution thresholds."""

    signal_data = dict(signal_section)
    raw = signal_data.get("peak_detector")
    if not isinstance(raw, Mapping):
        raise ValueError(
            "signal.peak_detector must persist detector_id and failure_action"
        )
    data = dict(raw)
    required = {"detector_id", "failure_action"}
    allowed = required | {"min_observation_sec", "min_peaks", "parameters"}
    if not required <= set(data) or not set(data) <= allowed:
        raise ValueError(
            "signal.peak_detector key mismatch: "
            f"missing={sorted(required-set(data))}, "
            f"unknown={sorted(set(data)-allowed)}"
        )
    from .peaks.resolver import (
        DEFAULT_MIN_OBSERVATION_SEC,
        DEFAULT_MIN_PEAKS,
        resolve_detector_id,
        resolve_detector_parameters,
        validate_peak_detection_parameters,
    )

    detector_id = resolve_detector_id(str(data["detector_id"]))
    min_observation_sec, min_peaks = validate_peak_detection_parameters(
        data.get("min_observation_sec", DEFAULT_MIN_OBSERVATION_SEC),
        data.get("min_peaks", DEFAULT_MIN_PEAKS),
    )
    parameters = resolve_detector_parameters(detector_id, data.get("parameters"))
    if data["failure_action"] != "fail_closed_no_fallback":
        raise ValueError(
            "signal.peak_detector.failure_action must be fail_closed_no_fallback"
        )
    resolved = {
        "detector_id": detector_id,
        "failure_action": "fail_closed_no_fallback",
        "min_observation_sec": min_observation_sec,
        "min_peaks": min_peaks,
    }
    if parameters:
        resolved["parameters"] = parameters
    return resolved


_ARTIFACT_CONFIG_TO_RUNTIME = {
    "identity": "identity",
    "nlms_imu_anc": "nlms_imu_anc",
    "ssa_decomposition": "ssa_decomposition",
    # 中文：formal motion YAML 与 runtime factory 保持逐字同名；
    # English: the formal motion YAML maps to the same exact runtime name.
    "spectral_mask": "spectral_mask",
    "pca_bss": "pca_bss",
    "fastica_bss": "fastica_bss",
    "nmf_bss": "nmf_bss",
    "emd_sifting_rate_only": "emd_sifting_rate_only",
    "ceemd_lite_nlms_legacy": "ceemd_lite_nlms_legacy",
    "dwt_a2_legacy": "dwt_a2_legacy",
}

_ARTIFACT_LEGACY_ALIASES = {
    "none": "identity",
    "direct": "identity",
    "nlms": "nlms_imu_anc",
    "ssa": "ssa_decomposition",
    "decomposition": "ssa_decomposition",
    "spectral": "spectral_mask",
    "stft": "spectral_mask",
    "stft_imu_mask": "spectral_mask",
    "pca": "pca_bss",
    "fastica": "fastica_bss",
    "ica": "fastica_bss",
    "nmf": "nmf_bss",
}


def resolve_artifact_module_id(value: str) -> dict[str, Any]:
    """解析 comparison ID 且显式标注旧别名 / Resolve IDs and label legacy aliases.

    中文：canonical ID 与 list-modules 完全一致；短名只为历史命令兼容，输出必须
    标注 legacy_alias_used，未知名称关闭失败。
    English: Canonical IDs exactly match list-modules. Legacy shorthands remain
    accepted only when the output explicitly records their alias status.
    """

    requested = str(value).strip().lower().replace("-", "_")
    if requested in _ARTIFACT_CONFIG_TO_RUNTIME:
        canonical = requested
        legacy = False
    elif requested in _ARTIFACT_LEGACY_ALIASES:
        canonical = _ARTIFACT_LEGACY_ALIASES[requested]
        legacy = True
    else:
        raise ValueError(f"artifact module ID is not registered: {value}")
    return {
        "requested_module_id": str(value),
        "canonical_module_id": canonical,
        "runtime_reducer": _ARTIFACT_CONFIG_TO_RUNTIME[canonical],
        "legacy_alias_used": legacy,
    }


def resolve_artifact_config(section: Mapping[str, Any]) -> dict[str, Any]:
    """验证并解析 artifact section / Validate and resolve an artifact section.

    中文：``spectral_mask`` 保持同名，并由真实 factory 严格验证
    `stft_window_s`、`stft_hop_s`、`imu_mask_quantile`、`mask_strength` 与
    `preserve_band_hz`；未知参数失败闭合。
    English: ``spectral_mask`` remains same-named. The real factory validates all
    five formal physical parameters and rejects every unknown key.
    """

    data = dict(section)
    required = {
        "reducer", "reducer_version", "selection_scope", "degraded_policy",
        "motion_detector_enabled", "non_identity_output_contract", "failure_action",
        "parameters",
    }
    if set(data) != required:
        raise ValueError(
            f"artifact key mismatch: missing={sorted(required-set(data))}, "
            f"unknown={sorted(set(data)-required)}"
        )
    declared = str(data["reducer"])
    if declared not in _ARTIFACT_CONFIG_TO_RUNTIME:
        raise ValueError(
            f"artifact.reducer={declared!r} has no exact V2 adapter; "
            "silent reducer/parameter translation is forbidden"
        )
    if data["selection_scope"] != "run_before_evaluation":
        raise ValueError("artifact route must be frozen before evaluation")
    if data["failure_action"] != "no_result_no_fallback":
        raise ValueError("artifact failures must not silently fall back")
    if not isinstance(data["motion_detector_enabled"], bool):
        raise ValueError("artifact.motion_detector_enabled must be boolean")
    policy = str(data["degraded_policy"])
    if policy not in {"drop", "denoise_then_extract_rate_features"}:
        raise ValueError(
            "artifact.degraded_policy must be drop or "
            "denoise_then_extract_rate_features"
        )
    expected_policy = (
        "drop" if declared == "identity" else "denoise_then_extract_rate_features"
    )
    if policy != expected_policy:
        raise ValueError(
            f"artifact reducer {declared!r} is executable only with "
            f"degraded_policy={expected_policy!r}"
        )
    if policy == "drop" and data["motion_detector_enabled"]:
        raise ValueError(
            "artifact.motion_detector_enabled requires an executable denoise recovery policy"
        )
    if data["non_identity_output_contract"] != "rate_only":
        raise ValueError("non-identity artifact output must be rate_only")
    parameters = data["parameters"]
    if not isinstance(parameters, Mapping):
        raise ValueError("artifact.parameters must be a mapping")
    # 中文：调用真实 factory 同时验证具体 dataclass 参数；English: the real
    # factory validates each concrete reducer dataclass and rejects unknown keys.
    from .artifact import get_reducer

    runtime_name = _ARTIFACT_CONFIG_TO_RUNTIME[declared]
    reducer = get_reducer(runtime_name, dict(parameters))
    declared_version = str(data["reducer_version"])
    runtime_version = str(reducer.reducer_version)
    # ``identity_v1`` is the source-YAML compatibility label predating the more
    # precise implementation version.  Every other declaration must be the
    # executable reducer version verbatim; arbitrary version metadata is never
    # allowed to masquerade as runtime provenance.
    compatible_versions = {runtime_version}
    if declared == "identity":
        compatible_versions.add("identity_v1")
    if declared_version not in compatible_versions:
        raise ValueError(
            f"artifact.reducer_version={declared_version!r} is not bound to "
            f"{declared!r} runtime version {runtime_version!r}"
        )
    return {
        "declared_reducer": declared,
        "runtime_reducer": runtime_name,
        "declared_version": declared_version,
        "runtime_version": runtime_version,
        "declared_version_is_compatibility_alias": (
            declared_version != runtime_version
        ),
        "is_identity": bool(reducer.is_identity),
        "parameters": dict(parameters),
    }


_MODEL_MODES = {
    "CompactCNN1D": {"raw"},
    "InceptionTimeFull": {"raw"},
    "InceptionTimeSmall": {"raw"},
    "InceptionTimeMatrix": {"feature_matrix"},
    "InceptionTimeFullFiveMemberEnsemble": {"raw"},
    "InceptionTimeMatrixFiveMemberEnsemble": {"feature_matrix"},
    "ROCKET": {"feature_matrix"},
    "MiniROCKET": {"feature_matrix"},
    "LogisticRegressionL2": {"feature_vector"},
    "RBFSVM": {"feature_vector"},
    "ExtraTrees": {"feature_vector"},
    "ShapeFormerChannelSpecificOSD": {"raw"},
    "ShapeFormerChannelSpecificScalarDistanceAblation": {"raw"},
    "ShapeFormerEffectSizeFixedV1": {"raw"},
    "ShapeFormerLegacyEffectSizePort": {"raw"},
    "FileBagFusionCompact": {"fusion"},
    "FileBagFusionInception": {"fusion"},
    "FileBagFusion": {"fusion"},
}


_MODEL_BASE_FIELDS = {
    "model_id",
    "variant",
    "input_channels",
    "input_channels_resolution",
    "n_classes",
    "ensemble_size",
    "architecture_parameters",
}

_MODEL_SPECIFIC_FIELDS = {
    "CompactCNN1D": {
        "seed_policy", "dropout", "kernel_sizes", "dilations", "pool_sizes",
        "stage_channels", "stage_dropouts", "mask_aware_pooling",
        "input_channel_order",
    },
    "InceptionTimeFull": {
        "seed_policy", "dropout", "kernel_sizes", "dilation", "mask_aware_pooling",
        "pool_size", "out_channels", "bottleneck_channels", "depth",
        "residual_interval", "input_channel_order",
    },
    "InceptionTimeSmall": {
        "seed_policy", "dropout", "kernel_sizes", "dilation", "mask_aware_pooling",
        "pool_size", "out_channels", "bottleneck_channels", "depth",
        "residual_interval", "input_channel_order",
    },
    "InceptionTimeMatrix": {
        "seed_policy", "dropout", "kernel_sizes", "dilation", "mask_aware_pooling",
        "pool_size", "out_channels", "bottleneck_channels", "depth",
        "residual_interval",
    },
    "InceptionTimeFullFiveMemberEnsemble": {
        "comparison_only", "member_seeds", "seed_policy",
        "member_seed_roster_id", "dropout", "kernel_sizes", "dilation",
        "pool_size", "out_channels", "bottleneck_channels", "depth",
        "residual_interval", "member_variant", "mask_aware_pooling",
        "input_channel_order",
    },
    "InceptionTimeMatrixFiveMemberEnsemble": {
        "comparison_only", "member_seeds", "seed_policy",
        "member_seed_roster_id", "dropout", "kernel_sizes", "dilation",
        "pool_size", "out_channels", "bottleneck_channels", "depth",
        "residual_interval", "member_variant", "mask_aware_pooling",
    },
    "ROCKET": {"seed_policy", "n_kernels", "alpha"},
    "MiniROCKET": {"seed_policy", "n_kernels", "alpha"},
    "LogisticRegressionL2": {
        "seed_policy", "class_weight", "logistic_c", "logistic_max_iter",
        "logistic_solver",
    },
    "RBFSVM": {
        "seed_policy", "class_weight", "svm_kernel", "svm_probability",
        "svm_c", "svm_gamma",
    },
    "ExtraTrees": {
        "seed_policy", "class_weight", "extra_trees_n_estimators",
        "extra_trees_n_jobs", "extra_trees_max_features",
        "extra_trees_min_samples_leaf",
    },
    "ShapeFormerChannelSpecificOSD": {
        "seed_policy", "input_channel_order", "discovery_method", "input_fs_hz", "num_pip_ratio",
        "shapelets_per_class", "max_discovery_windows", "discovery_balance",
        "position_search_neighbourhood_samples", "pip_rounding_rule",
        "pip_selection_rule", "candidate_generation_rule",
        "candidate_enumeration_rule", "candidate_ranking_rule",
        "selected_bank_order_rule",
        "discovery_position_search_boundary_rule",
        "information_gain_split_rule",
        "sequence_length_samples",
        "local_kernel_width_samples", "local_embedding_channels",
        "shape_embedding_channels", "attention_feedforward_channels",
        "attention_heads", "attention_query_chunk_size",
        "distance_position_chunk_size", "dropout", "complexity_norm",
        "max_complexity_ratio", "mask_aware_pooling",
    },
    "ShapeFormerChannelSpecificScalarDistanceAblation": {
        "seed_policy", "input_channel_order", "discovery_method", "input_fs_hz",
        "num_pip_ratio", "shapelets_per_class", "max_discovery_windows",
        "discovery_balance", "position_search_neighbourhood_samples",
        "pip_rounding_rule", "pip_selection_rule", "candidate_generation_rule",
        "candidate_enumeration_rule", "candidate_ranking_rule",
        "selected_bank_order_rule", "discovery_position_search_boundary_rule",
        "information_gain_split_rule", "hidden_channels", "dropout",
        "patch_size_samples", "attention_heads", "attention_layers",
        "attention_feedforward_channels", "distance_position_chunk_size",
        "mask_aware_pooling",
    },
    "ShapeFormerEffectSizeFixedV1": {
        "seed_policy", "input_channel_order", "discovery_method", "input_fs_hz",
        "shapelet_length_samples", "shapelets_per_class",
        "discovery_stride_samples", "max_candidates_per_class",
        "hidden_channels", "dropout", "patch_size_samples",
        "attention_heads", "attention_layers", "attention_feedforward_channels",
        "distance_position_chunk_size", "mask_aware_pooling",
    },
    "ShapeFormerLegacyEffectSizePort": {
        "seed_policy", "input_channel_order", "discovery_method",
        "discovery_balance", "input_fs_hz", "sequence_length_samples",
        "shapelet_length_samples", "discovery_stride_samples",
        "shapelets_per_class", "max_discovery_windows",
        "candidates_per_class_channel", "local_kernel_width_samples",
        "local_embedding_channels", "shape_embedding_channels",
        "attention_feedforward_channels", "attention_heads", "dropout",
        "shapelet_search_window_samples", "complexity_norm",
        "max_complexity_ratio", "mask_aware_pooling",
    },
    "FileBagFusionCompact": {
        "seed_policy", "signal_dropout", "signal_kernel_sizes",
        "signal_dilations", "signal_pool_sizes", "signal_stage_channels",
        "signal_stage_dropouts", "feature_hidden_dim", "fusion_hidden_dim",
        "pooling", "dropout", "mask_aware_pooling", "input_channel_order",
    },
    "FileBagFusionInception": {
        "seed_policy", "signal_variant", "signal_dropout",
        "signal_kernel_sizes", "signal_dilation", "signal_pool_size",
        "signal_out_channels", "signal_bottleneck_channels", "signal_depth",
        "signal_residual_interval", "feature_hidden_dim",
        "fusion_hidden_dim", "pooling", "dropout", "mask_aware_pooling",
        "input_channel_order",
    },
    "FileBagFusion": {
        "seed_policy", "signal_encoder", "feature_hidden_dim",
        "fusion_hidden_dim", "pooling", "dropout", "mask_aware_pooling",
        "input_channel_order",
    },
}


_MODEL_OPTIONAL_FIELDS = {
    "CompactCNN1D": {"stage_channels", "stage_dropouts"},
    "InceptionTimeFull": {
        "pool_size", "out_channels", "bottleneck_channels", "depth",
        "residual_interval",
    },
    "InceptionTimeSmall": {
        "pool_size", "out_channels", "bottleneck_channels", "depth",
        "residual_interval",
    },
    "InceptionTimeMatrix": {
        "pool_size", "out_channels", "bottleneck_channels", "depth",
        "residual_interval",
    },
    "InceptionTimeFullFiveMemberEnsemble": {
        "comparison_only", "member_seed_roster_id", "pool_size",
        "out_channels", "bottleneck_channels", "depth",
        "residual_interval", "member_variant",
    },
    "InceptionTimeMatrixFiveMemberEnsemble": {
        "comparison_only", "member_seed_roster_id", "pool_size",
        "out_channels", "bottleneck_channels", "depth",
        "residual_interval", "member_variant",
    },
    "LogisticRegressionL2": {
        "class_weight", "logistic_c", "logistic_max_iter", "logistic_solver",
    },
    "RBFSVM": {"class_weight"},
    "ExtraTrees": {
        "class_weight", "extra_trees_n_estimators", "extra_trees_n_jobs",
        "extra_trees_max_features", "extra_trees_min_samples_leaf",
    },
    "ShapeFormerChannelSpecificOSD": {
        "discovery_method", "input_fs_hz", "num_pip_ratio",
        "shapelets_per_class", "max_discovery_windows", "discovery_balance",
        "position_search_neighbourhood_samples", "pip_rounding_rule",
        "pip_selection_rule", "candidate_generation_rule",
        "candidate_enumeration_rule", "candidate_ranking_rule",
        "selected_bank_order_rule", "discovery_position_search_boundary_rule",
        "information_gain_split_rule", "sequence_length_samples",
        "local_kernel_width_samples", "local_embedding_channels",
        "shape_embedding_channels", "attention_feedforward_channels",
        "attention_heads", "attention_query_chunk_size",
        "distance_position_chunk_size", "dropout", "complexity_norm",
        "max_complexity_ratio",
    },
    "ShapeFormerChannelSpecificScalarDistanceAblation": {
        "discovery_method", "input_fs_hz", "num_pip_ratio",
        "shapelets_per_class", "max_discovery_windows", "discovery_balance",
        "position_search_neighbourhood_samples", "pip_rounding_rule",
        "pip_selection_rule", "candidate_generation_rule",
        "candidate_enumeration_rule", "candidate_ranking_rule",
        "selected_bank_order_rule", "discovery_position_search_boundary_rule",
        "information_gain_split_rule", "hidden_channels", "dropout",
        "patch_size_samples", "attention_heads", "attention_layers",
        "attention_feedforward_channels", "distance_position_chunk_size",
    },
    "ShapeFormerEffectSizeFixedV1": {
        "discovery_method", "input_fs_hz", "shapelet_length_samples",
        "shapelets_per_class", "discovery_stride_samples",
        "max_candidates_per_class", "hidden_channels", "dropout",
        "patch_size_samples", "attention_heads", "attention_layers",
        "attention_feedforward_channels",
        "distance_position_chunk_size",
    },
    "ShapeFormerLegacyEffectSizePort": {
        "discovery_method", "discovery_balance", "input_fs_hz",
        "sequence_length_samples", "shapelet_length_samples",
        "discovery_stride_samples", "shapelets_per_class",
        "max_discovery_windows", "candidates_per_class_channel",
        "local_kernel_width_samples", "local_embedding_channels",
        "shape_embedding_channels", "attention_feedforward_channels",
        "attention_heads", "dropout", "shapelet_search_window_samples",
        "complexity_norm", "max_complexity_ratio",
    },
    "FileBagFusionCompact": {"signal_stage_channels", "signal_stage_dropouts"},
    "FileBagFusionInception": {
        "signal_pool_size", "signal_out_channels", "signal_bottleneck_channels",
        "signal_depth", "signal_residual_interval",
    },
    "FileBagFusion": {
        "signal_encoder", "feature_hidden_dim", "fusion_hidden_dim", "pooling",
        "dropout",
    },
}

_MODEL_FACTORY_METADATA_FIELDS = {
    "input_channel_order",
    "mask_aware_pooling",
    "member_variant",
    "comparison_only",
    "member_seed_roster_id",
}
_MODEL_FACTORY_VARIANT_CONSUMERS = {"InceptionTimeMatrix"}
_MODEL_ENSEMBLE_NAMES = {
    "InceptionTimeFullFiveMemberEnsemble",
    "InceptionTimeMatrixFiveMemberEnsemble",
}
MODEL_REGISTRY_ROLES = frozenset(
    {"reference", "ablation", "comparison", "optional"}
)
_MODEL_VARIANT_LEGACY_ALIASES = {
    "CompactCNN1D": {"canonical_32_64_128", "legacy_reference_not_wang_fcn"},
    "InceptionTimeFull": {"full_single_network"},
    "InceptionTimeSmall": {"small_single_network"},
    "InceptionTimeFullFiveMemberEnsemble": {"full_five_independent_members"},
    "InceptionTimeMatrixFiveMemberEnsemble": {"full_five_independent_members"},
    "ROCKET": {"numpy_rocket_ridge"},
    "MiniROCKET": {"engineering_ablation"},
    "LogisticRegressionL2": {"l2_lbfgs", "reference_file_vector"},
    "RBFSVM": {"rbf_probability"},
    "ExtraTrees": {"500_trees"},
    "ShapeFormerChannelSpecificOSD": {"channel_specific_osd_reference"},
    "ShapeFormerChannelSpecificScalarDistanceAblation": {
        "channel_specific_scalar_distance_ablation"
    },
    "ShapeFormerEffectSizeFixedV1": {"effect_size_fixed_v1"},
    "ShapeFormerLegacyEffectSizePort": {
        "legacy_parallel_ablation_not_osd_parity"
    },
    "FileBagFusionCompact": {"compact_raw_encoder"},
    "FileBagFusionInception": {"small_raw_encoder"},
    "FileBagFusion": {"optional_composable_signal_encoder"},
}


def model_factory_contract(model_id_or_name: str) -> dict[str, Any]:
    """Return the registry-owned adapter contract for one model factory.

    The experiment runner consumes this contract instead of maintaining a
    second per-model option table. Runtime constructor validation remains in
    :mod:`ppg_frailty.models.factory`; this function owns only which validated
    model-section fields are forwarded and which may be omitted for defaults.
    """

    from .models.factory import normalize_model_id

    canonical, machine_id = normalize_model_id(str(model_id_or_name))
    descriptor = next(
        item for item in MODEL_MODULES if item.module_id == canonical
    )
    scientific_status = str(descriptor.scientific_status)
    if canonical in _MODEL_ENSEMBLE_NAMES:
        registry_role = "comparison"
    elif "ablation" in scientific_status:
        registry_role = "ablation"
    elif "optional" in scientific_status:
        registry_role = "optional"
    else:
        registry_role = "reference"
    factory_fields = (
        set(_MODEL_SPECIFIC_FIELDS[canonical]) - _MODEL_FACTORY_METADATA_FIELDS
    )
    if canonical in _MODEL_FACTORY_VARIANT_CONSUMERS:
        factory_fields.add("variant")
    if canonical not in _MODEL_ENSEMBLE_NAMES:
        factory_fields.add("seed")
    optional_fields = set(_MODEL_OPTIONAL_FIELDS.get(canonical, set())) & factory_fields
    return {
        "canonical_model_name": canonical,
        "machine_model_id": machine_id,
        "registry_role": registry_role,
        "scientific_status": scientific_status,
        "factory_fields": tuple(sorted(factory_fields)),
        "optional_factory_fields": tuple(sorted(optional_fields)),
        "derived_provenance_fields": tuple(
            sorted(
                {
                    "architecture_parameters",
                    "ensemble_size",
                    *(
                        ()
                        if canonical == "InceptionTimeMatrix"
                        else ("variant",)
                    ),
                    *(
                        ("mask_aware_pooling",)
                        if "mask_aware_pooling" in _MODEL_SPECIFIC_FIELDS[canonical]
                        else ()
                    ),
                    *(
                        ("comparison_only", "member_seed_roster_id")
                        if canonical in _MODEL_ENSEMBLE_NAMES
                        else ()
                    ),
                }
            )
        ),
        "representation_modes": descriptor.representation_modes,
        "runtime_dependencies": descriptor.runtime_dependencies,
        "execution_backend": (
            "torch" if "torch" in descriptor.runtime_dependencies else "estimator"
        ),
    }


def model_runtime_dependencies(model_id_or_name: str) -> tuple[str, ...]:
    """Return executable imports declared by the model's registry contract."""

    return tuple(model_factory_contract(model_id_or_name)["runtime_dependencies"])


def derived_model_ensemble_size(section: Mapping[str, Any]) -> int:
    """Derive member count from the executable roster, never a duplicate field."""

    if "ensemble_size" in section and (
        isinstance(section["ensemble_size"], bool)
        or not isinstance(section["ensemble_size"], int)
        or int(section["ensemble_size"]) <= 0
    ):
        raise ValueError(
            "legacy ensemble_size provenance must be a positive integer when supplied"
        )
    contract = model_factory_contract(str(section.get("model_id", "")))
    if contract["canonical_model_name"] in _MODEL_ENSEMBLE_NAMES:
        derived = len(
            _validated_ensemble_seed_roster(section.get("member_seeds"))
        )
    else:
        derived = 1
    if "ensemble_size" in section and int(section["ensemble_size"]) != derived:
        raise ValueError(
            "model.ensemble_size derived field mismatch; omit it or match the executable member roster"
        )
    return derived


def derived_model_variant(section: Mapping[str, Any]) -> str:
    """Return a semantic variant, deriving labels that are not factory inputs."""

    contract = model_factory_contract(str(section.get("model_id", "")))
    canonical = str(contract["canonical_model_name"])
    declared = section.get("variant")
    if canonical == "InceptionTimeMatrix":
        variant = str(section.get("variant", "full"))
        if variant not in {"full", "small"}:
            raise ValueError("InceptionTimeMatrix.variant must be full or small")
        return variant
    if canonical in _MODEL_ENSEMBLE_NAMES:
        member_variant = str(section.get("member_variant", "full"))
        if member_variant not in {"full", "small"}:
            raise ValueError("ensemble member_variant must be full or small")
        derived = f"{member_variant}_probability_ensemble"
    else:
        descriptor = next(
            item for item in MODEL_MODULES if item.module_id == canonical
        )
        derived = str(descriptor.scientific_status)
    if (
        declared is not None
        and str(declared) != derived
        and str(declared) not in _MODEL_VARIANT_LEGACY_ALIASES.get(canonical, set())
    ):
        raise ValueError(
            "model.variant derived field mismatch; omit it or use the registered semantic variant"
        )
    return derived


def derived_mask_aware_pooling(section: Mapping[str, Any]) -> bool | None:
    """Materialize mask handling only for models with that runtime capability."""

    contract = model_factory_contract(str(section.get("model_id", "")))
    canonical = str(contract["canonical_model_name"])
    if "mask_aware_pooling" not in _MODEL_SPECIFIC_FIELDS[canonical]:
        return None
    if section.get("mask_aware_pooling", True) is not True:
        raise ValueError(
            f"{canonical} does not implement a mask-unaware execution branch"
        )
    return True


def validate_legacy_ensemble_metadata(section: Mapping[str, Any]) -> None:
    """Accept only the two historical catalogue annotations, never new inputs."""

    contract = model_factory_contract(str(section.get("model_id", "")))
    canonical = str(contract["canonical_model_name"])
    if canonical not in _MODEL_ENSEMBLE_NAMES:
        unexpected = sorted(
            {"comparison_only", "member_seed_roster_id"} & set(section)
        )
        if unexpected:
            raise ValueError(
                f"{canonical} has no legacy ensemble metadata capability: "
                f"{unexpected}"
            )
        return
    if "comparison_only" in section and section["comparison_only"] is not True:
        raise ValueError(
            "model.comparison_only is legacy catalogue provenance and must be true when supplied"
        )
    if (
        "member_seed_roster_id" in section
        and section["member_seed_roster_id"]
        != "cv_fixed_five_member_seed_roster"
    ):
        raise ValueError(
            "model.member_seed_roster_id accepts only the historical catalogue annotation"
        )


def materialize_model_architecture(
    section: Mapping[str, Any],
    representation_mode: str | None = None,
) -> dict[str, Any]:
    """Derive complete architecture provenance from top-level factory inputs."""

    from .models.factory import (
        ModelInputSpec,
        materialize_architecture_parameters,
        validate_source_architecture_annotation,
    )

    data = dict(section)
    contract = model_factory_contract(str(data.get("model_id", "")))
    modes = tuple(str(value) for value in contract["representation_modes"])
    mode = modes[0] if representation_mode is None and len(modes) == 1 else str(
        representation_mode
    )
    if mode not in modes:
        raise ValueError(
            f"{contract['canonical_model_name']} is incompatible with {mode}"
        )
    factory_config: dict[str, Any] = {
        "model_id": contract["machine_model_id"],
    }
    for field in contract["factory_fields"]:
        if field in data:
            factory_config[field] = data[field]
    if contract["canonical_model_name"] in _MODEL_ENSEMBLE_NAMES:
        factory_config["variant"] = str(data.get("member_variant", "full"))
    channel_schema = tuple(str(value) for value in data.get("input_channel_order", ()))
    spec = ModelInputSpec(
        mode,
        n_channels=int(data.get("input_channels", 0)),
        n_classes=int(data.get("n_classes", 3)),
        channel_schema=channel_schema,
    )
    derived = materialize_architecture_parameters(factory_config, spec)
    validate_source_architecture_annotation(
        data.get("architecture_parameters"), derived
    )
    return derived


def _validated_ensemble_seed_roster(values: Any) -> tuple[int, ...]:
    """Return one explicit, non-empty, unique roster safe for OOF storage."""

    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ValueError("member_seeds must be a non-string list or tuple")
    seeds: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("member_seeds must contain integer values")
        seed = int(value)
        if seed < 0 or seed > 0xFFFF_FFFF:
            raise ValueError("member_seeds must be in the executable uint32 seed range")
        seeds.append(seed)
    roster = tuple(seeds)
    if not roster or len(roster) != len(set(roster)):
        raise ValueError("member_seeds must be non-empty and unique")
    return roster


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _nonzero_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value == 0:
        raise ValueError(f"{field} must be a non-zero integer")
    return int(value)


def _finite_positive(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite and positive")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return normalized


def _probability(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite in [0,1)")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized < 1.0:
        raise ValueError(f"{field} must be finite in [0,1)")
    return normalized


def _positive_integer_sequence(
    value: Any, *, field: str, length: int, odd: bool = False
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field} must contain exactly {length} integers")
    normalized = tuple(_positive_integer(item, field=field) for item in value)
    if odd and any(item % 2 == 0 for item in normalized):
        raise ValueError(f"{field} must contain positive odd integers")
    return normalized


def _probability_sequence(
    value: Any, *, field: str, length: int
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field} must contain exactly {length} probabilities")
    return tuple(_probability(item, field=field) for item in value)


def _validate_extra_trees_max_features(value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        if value not in {"sqrt", "log2"}:
            raise ValueError("extra_trees_max_features string must be 'sqrt' or 'log2'")
        return
    if isinstance(value, bool):
        raise ValueError("extra_trees_max_features cannot be boolean")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("integer extra_trees_max_features must be positive")
        return
    if isinstance(value, float) and math.isfinite(value) and 0.0 < value <= 1.0:
        return
    raise ValueError("float extra_trees_max_features must be finite in (0,1]")


def _validate_extra_trees_min_samples_leaf(value: Any) -> None:
    if isinstance(value, bool):
        raise ValueError("extra_trees_min_samples_leaf cannot be boolean")
    if isinstance(value, int) and value > 0:
        return
    if isinstance(value, float) and math.isfinite(value) and 0.0 < value <= 0.5:
        return
    raise ValueError(
        "extra_trees_min_samples_leaf must be a positive integer or fraction in (0,0.5]"
    )


def _validate_fusion_signal_encoder_declaration(value: Any) -> None:
    """Validate the nested public encoder mapping without accepting fit state."""

    from .models.factory import (
        FRAILTY_RAW_CHANNEL_SCHEMA,
        ModelInputSpec,
        materialize_architecture_parameters,
        normalize_fusion_signal_encoder_config,
    )

    source = {"model_id": "compact_cnn"} if value is None else value
    if not isinstance(source, Mapping):
        raise ValueError("model.signal_encoder must be a mapping")
    normalized = normalize_fusion_signal_encoder_config(source)
    contract = model_factory_contract(str(normalized["model_id"]))
    allowed = {
        "model_id",
        "model_name",
        *(set(contract["factory_fields"]) - {"seed", "seed_policy"}),
    }
    unknown = sorted(set(source) - allowed)
    if unknown:
        raise ValueError(
            "model.signal_encoder contains non-executable or fold-owned fields: "
            f"{unknown}"
        )
    if tuple(contract["representation_modes"]) != ("raw",):
        raise ValueError("model.signal_encoder must use a raw representation model")
    if contract["execution_backend"] != "torch":
        raise ValueError("model.signal_encoder must use a torch feature encoder")

    raw_spec = ModelInputSpec(
        "raw",
        n_channels=8,
        n_classes=3,
        channel_schema=FRAILTY_RAW_CHANNEL_SCHEMA,
    )
    architecture = materialize_architecture_parameters(normalized, raw_spec)
    nested = {
        key: item
        for key, item in source.items()
        if key not in {"model_id", "model_name"}
    }
    nested.update(
        {
            "model_id": str(contract["canonical_model_name"]),
            "input_channels": 8,
            "input_channels_resolution": "canonical_frailty_raw_8",
            "input_channel_order": list(FRAILTY_RAW_CHANNEL_SCHEMA),
            "n_classes": 3,
            "seed_policy": "fixed_explicit",
            "mask_aware_pooling": True,
        }
    )
    if contract["canonical_model_name"] == "CompactCNN1D":
        nested.setdefault("dropout", architecture["classifier_dropout"])
        for field in ("kernel_sizes", "dilations", "pool_sizes"):
            nested.setdefault(field, architecture[field])
    elif contract["canonical_model_name"] in {
        "InceptionTimeFull",
        "InceptionTimeSmall",
    }:
        nested.setdefault("dropout", architecture["classifier_dropout"])
        for field in ("kernel_sizes", "dilation"):
            nested.setdefault(field, architecture[field])
    validate_model_config(nested, "raw")


def validate_model_config(section: Mapping[str, Any], representation_mode: str) -> dict[str, str]:
    """检查模型元数据与 representation / Check model metadata against its mode."""

    data = dict(section)
    canonical = str(data.get("model_id", ""))
    if canonical not in _MODEL_MODES or canonical not in _MODEL_SPECIFIC_FIELDS:
        raise ValueError(f"model_id is not registered: {canonical}")
    allowed = _MODEL_BASE_FIELDS | _MODEL_SPECIFIC_FIELDS[canonical]
    optional = _MODEL_OPTIONAL_FIELDS.get(canonical, set()) | {
        "architecture_parameters",
        "ensemble_size",
        "variant",
        "mask_aware_pooling",
    }
    required = allowed - optional
    if not required <= set(data) or not set(data) <= allowed:
        raise ValueError(
            f"model key mismatch: missing={sorted(required-set(data))}, "
            f"unknown={sorted(set(data)-allowed)}"
        )
    if representation_mode not in _MODEL_MODES[canonical]:
        raise ValueError(f"{canonical} is incompatible with {representation_mode}")
    derived_model_ensemble_size(data)
    derived_model_variant(data)
    derived_mask_aware_pooling(data)
    validate_legacy_ensemble_metadata(data)
    if int(data["n_classes"]) != 3:
        raise ValueError("reference task requires exactly three classes")
    if "mask_aware_pooling" in data and not isinstance(data["mask_aware_pooling"], bool):
        raise ValueError("mask_aware_pooling must be explicit boolean")
    if "ensemble_size" in data and (
        isinstance(data["ensemble_size"], bool)
        or not isinstance(data["ensemble_size"], int)
        or int(data["ensemble_size"]) <= 0
    ):
        raise ValueError(
            "legacy ensemble_size provenance must be a positive integer when supplied"
        )
    ensemble_names = {
        "InceptionTimeFullFiveMemberEnsemble",
        "InceptionTimeMatrixFiveMemberEnsemble",
    }
    if canonical in ensemble_names:
        seeds = _validated_ensemble_seed_roster(data["member_seeds"])
        seed_policy = str(data["seed_policy"])
        if seed_policy not in {"member_roster", "cv_fixed_five_member_seed_roster"}:
            raise ValueError("ensemble seed_policy must resolve an explicit member roster")
        if seed_policy == "cv_fixed_five_member_seed_roster" and len(seeds) != 5:
            raise ValueError(
                "the legacy cv_fixed_five_member_seed_roster name denotes five members; "
                "use member_roster for any other roster length"
            )
        if "comparison_only" in data and not isinstance(data["comparison_only"], bool):
            raise ValueError("comparison_only is optional boolean metadata")
        if "member_seed_roster_id" in data and (
            not isinstance(data["member_seed_roster_id"], str)
            or not data["member_seed_roster_id"].strip()
        ):
            raise ValueError("member_seed_roster_id must be a non-empty string when supplied")
    if representation_mode in {"raw", "fusion"}:
        expected_channel_order = [
            "RED", "IR", "A_dyn_x", "A_dyn_y", "A_dyn_z", "GX", "GY", "GZ"
        ]
        if (
            int(data["input_channels"]) != 8
            or data["input_channels_resolution"]
            != "canonical_frailty_raw_8"
            or data["input_channel_order"] != expected_channel_order
        ):
            raise ValueError(
                "frailty raw/fusion models require the exact canonical 8-channel schema"
            )
    from .models import normalize_model_id

    normalized_name, machine_id = normalize_model_id(canonical)
    if canonical == "FileBagFusion":
        _validate_fusion_signal_encoder_declaration(data.get("signal_encoder"))
    if canonical in {"CompactCNN1D", "FileBagFusionCompact"}:
        prefix = "" if canonical == "CompactCNN1D" else "signal_"
        _positive_integer_sequence(
            data.get(f"{prefix}stage_channels", (32, 64, 128)),
            field=f"{prefix}stage_channels",
            length=3,
        )
        _probability_sequence(
            data.get(f"{prefix}stage_dropouts", (0.10, 0.15)),
            field=f"{prefix}stage_dropouts",
            length=2,
        )
        _probability(
            data["dropout" if canonical == "CompactCNN1D" else "signal_dropout"],
            field="dropout" if canonical == "CompactCNN1D" else "signal_dropout",
        )

    if canonical in {"LogisticRegressionL2", "RBFSVM", "ExtraTrees"}:
        if data.get("class_weight") is not None:
            raise ValueError(
                "model.class_weight is not an independent weighting capability; "
                "configure the single training.class_weighting strategy"
            )

    if canonical == "LogisticRegressionL2":
        _finite_positive(data.get("logistic_c", 1.0), field="logistic_c")
        _positive_integer(
            data.get("logistic_max_iter", 5000), field="logistic_max_iter"
        )
        solver = str(data.get("logistic_solver", "lbfgs"))
        supported_solvers = {
            "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga",
        }
        if solver not in supported_solvers:
            raise ValueError(
                f"logistic_solver must be one of {sorted(supported_solvers)}"
            )
    if canonical == "ExtraTrees":
        _positive_integer(
            data.get("extra_trees_n_estimators", 500),
            field="extra_trees_n_estimators",
        )
        _nonzero_integer(
            data.get("extra_trees_n_jobs", 1), field="extra_trees_n_jobs"
        )
        _validate_extra_trees_max_features(
            data.get("extra_trees_max_features", "sqrt")
        )
        _validate_extra_trees_min_samples_leaf(
            data.get("extra_trees_min_samples_leaf", 1)
        )

    channel_specific_shapeformers = {
        "ShapeFormerChannelSpecificOSD",
        "ShapeFormerChannelSpecificScalarDistanceAblation",
    }
    if canonical in channel_specific_shapeformers:
        from .models.factory import (
            SHAPEFORMER_EXPERIMENTAL_NUMERIC_DEFAULTS,
            SHAPEFORMER_REFERENCE_NUMERIC_DEFAULTS,
        )
        from .models.pisd_port import (
            CANDIDATE_ENUMERATION_RULE,
            CANDIDATE_GENERATION_RULE,
            CANDIDATE_RANKING_RULE,
            DISCOVERY_BALANCE,
            DISCOVERY_BALANCES,
            DISCOVERY_POSITION_SEARCH_BOUNDARY_RULE,
            INFORMATION_GAIN_SPLIT_RULE,
            PIP_ROUNDING_RULE,
            PIP_SELECTION_RULE,
            PISD_DISCOVERY_METHOD,
            SELECTED_BANK_ORDER_RULE,
        )

        expected_rules = {
            "discovery_method": PISD_DISCOVERY_METHOD,
            "pip_rounding_rule": PIP_ROUNDING_RULE,
            "pip_selection_rule": PIP_SELECTION_RULE,
            "candidate_generation_rule": CANDIDATE_GENERATION_RULE,
            "candidate_enumeration_rule": CANDIDATE_ENUMERATION_RULE,
            "candidate_ranking_rule": CANDIDATE_RANKING_RULE,
            "selected_bank_order_rule": SELECTED_BANK_ORDER_RULE,
            "discovery_position_search_boundary_rule": (
                DISCOVERY_POSITION_SEARCH_BOUNDARY_RULE
            ),
            "information_gain_split_rule": INFORMATION_GAIN_SPLIT_RULE,
        }
        for field, expected in expected_rules.items():
            if data.get(field, expected) != expected:
                raise ValueError(f"{canonical} requires {field}={expected}")
        discovery_balance = data.get("discovery_balance", DISCOVERY_BALANCE)
        if discovery_balance not in DISCOVERY_BALANCES:
            raise ValueError(
                f"{canonical} discovery_balance must be one of "
                f"{sorted(DISCOVERY_BALANCES)}"
            )
        input_fs_hz = _finite_positive(
            data.get(
                "input_fs_hz",
                SHAPEFORMER_REFERENCE_NUMERIC_DEFAULTS["input_fs_hz"],
            ),
            field="input_fs_hz",
        )
        del input_fs_hz
        ratio = _finite_positive(
            data.get(
                "num_pip_ratio",
                SHAPEFORMER_REFERENCE_NUMERIC_DEFAULTS["num_pip_ratio"],
            ),
            field="num_pip_ratio",
        )
        if ratio > 1.0:
            raise ValueError("num_pip_ratio must be in (0,1]")
        for field in (
            "shapelets_per_class",
            "max_discovery_windows",
            "position_search_neighbourhood_samples",
        ):
            _positive_integer(
                data.get(field, SHAPEFORMER_REFERENCE_NUMERIC_DEFAULTS[field]),
                field=field,
            )
        if canonical == "ShapeFormerChannelSpecificOSD":
            for field in (
                "sequence_length_samples",
                "local_kernel_width_samples",
                "local_embedding_channels",
                "shape_embedding_channels",
                "attention_feedforward_channels",
                "attention_heads",
                "attention_query_chunk_size",
                "distance_position_chunk_size",
            ):
                _positive_integer(
                    data.get(field, SHAPEFORMER_REFERENCE_NUMERIC_DEFAULTS[field]),
                    field=field,
                )
            heads = int(
                data.get(
                    "attention_heads",
                    SHAPEFORMER_REFERENCE_NUMERIC_DEFAULTS["attention_heads"],
                )
            )
            for field in ("local_embedding_channels", "shape_embedding_channels"):
                width = int(
                    data.get(field, SHAPEFORMER_REFERENCE_NUMERIC_DEFAULTS[field])
                )
                if width % heads:
                    raise ValueError(f"{field} must be divisible by attention_heads")
            _probability(
                data.get(
                    "dropout",
                    SHAPEFORMER_REFERENCE_NUMERIC_DEFAULTS["dropout"],
                ),
                field="dropout",
            )
            _finite_positive(
                data.get(
                    "complexity_norm",
                    SHAPEFORMER_REFERENCE_NUMERIC_DEFAULTS["complexity_norm"],
                ),
                field="complexity_norm",
            )
            max_ratio = _finite_positive(
                data.get(
                    "max_complexity_ratio",
                    SHAPEFORMER_REFERENCE_NUMERIC_DEFAULTS[
                        "max_complexity_ratio"
                    ],
                ),
                field="max_complexity_ratio",
            )
            if max_ratio < 1.0:
                raise ValueError("max_complexity_ratio must be at least one")
        else:
            for field in (
                "hidden_channels", "patch_size_samples", "attention_heads",
                "attention_layers", "attention_feedforward_channels",
                "distance_position_chunk_size",
            ):
                _positive_integer(
                    data.get(field, SHAPEFORMER_EXPERIMENTAL_NUMERIC_DEFAULTS[field]),
                    field=field,
                )
            if int(
                data.get(
                    "hidden_channels",
                    SHAPEFORMER_EXPERIMENTAL_NUMERIC_DEFAULTS["hidden_channels"],
                )
            ) % int(
                data.get(
                    "attention_heads",
                    SHAPEFORMER_EXPERIMENTAL_NUMERIC_DEFAULTS["attention_heads"],
                )
            ):
                raise ValueError("hidden_channels must be divisible by attention_heads")
            _probability(
                data.get(
                    "dropout",
                    SHAPEFORMER_EXPERIMENTAL_NUMERIC_DEFAULTS["dropout"],
                ),
                field="dropout",
            )
            if int(
                data.get(
                    "patch_size_samples",
                    SHAPEFORMER_EXPERIMENTAL_NUMERIC_DEFAULTS[
                        "patch_size_samples"
                    ],
                )
            ) < 2:
                raise ValueError(
                    "patch_size_samples must be at least 2; raw sample-token "
                    "attention is not an executable ShapeFormer capability"
                )

    if canonical == "ShapeFormerLegacyEffectSizePort":
        from .models.factory import SHAPEFORMER_LEGACY_EFFECT_SIZE_DEFAULTS
        from .models.shapeformer_legacy import (
            LEGACY_DISCOVERY_BALANCE,
            LEGACY_EFFECT_SIZE_DISCOVERY_METHOD,
        )

        if (
            data.get(
                "discovery_method", LEGACY_EFFECT_SIZE_DISCOVERY_METHOD
            )
            != LEGACY_EFFECT_SIZE_DISCOVERY_METHOD
        ):
            raise ValueError(
                "legacy effect-size ShapeFormer discovery method cannot drift"
            )
        if (
            data.get("discovery_balance", LEGACY_DISCOVERY_BALANCE)
            != LEGACY_DISCOVERY_BALANCE
        ):
            raise ValueError(
                "legacy effect-size ShapeFormer discovery balance cannot drift"
            )
        _finite_positive(
            data.get(
                "input_fs_hz",
                SHAPEFORMER_LEGACY_EFFECT_SIZE_DEFAULTS["input_fs_hz"],
            ),
            field="input_fs_hz",
        )
        integer_fields = (
            "sequence_length_samples",
            "shapelet_length_samples",
            "discovery_stride_samples",
            "shapelets_per_class",
            "max_discovery_windows",
            "candidates_per_class_channel",
            "local_kernel_width_samples",
            "local_embedding_channels",
            "shape_embedding_channels",
            "attention_feedforward_channels",
            "attention_heads",
            "shapelet_search_window_samples",
        )
        values = {
            field: _positive_integer(
                data.get(
                    field, SHAPEFORMER_LEGACY_EFFECT_SIZE_DEFAULTS[field]
                ),
                field=field,
            )
            for field in integer_fields
        }
        if values["max_discovery_windows"] < int(data["n_classes"]):
            raise ValueError(
                "max_discovery_windows must retain at least one window per class"
            )
        if values["shapelet_length_samples"] > values[
            "sequence_length_samples"
        ]:
            raise ValueError(
                "shapelet_length_samples cannot exceed sequence_length_samples"
            )
        if values["shapelet_length_samples"] < 4:
            raise ValueError(
                "shapelet_length_samples must be at least four effective samples"
            )
        if values["local_kernel_width_samples"] > values[
            "sequence_length_samples"
        ]:
            raise ValueError(
                "local_kernel_width_samples cannot exceed sequence_length_samples"
            )
        heads = values["attention_heads"]
        for field in ("local_embedding_channels", "shape_embedding_channels"):
            if values[field] % heads:
                raise ValueError(f"{field} must be divisible by attention_heads")
        _probability(
            data.get(
                "dropout",
                SHAPEFORMER_LEGACY_EFFECT_SIZE_DEFAULTS["dropout"],
            ),
            field="dropout",
        )
        _finite_positive(
            data.get(
                "complexity_norm",
                SHAPEFORMER_LEGACY_EFFECT_SIZE_DEFAULTS["complexity_norm"],
            ),
            field="complexity_norm",
        )
        max_ratio = _finite_positive(
            data.get(
                "max_complexity_ratio",
                SHAPEFORMER_LEGACY_EFFECT_SIZE_DEFAULTS[
                    "max_complexity_ratio"
                ],
            ),
            field="max_complexity_ratio",
        )
        if max_ratio < 1.0:
            raise ValueError("max_complexity_ratio must be at least one")

    if canonical == "ShapeFormerEffectSizeFixedV1":
        from .models.factory import SHAPEFORMER_EXPERIMENTAL_NUMERIC_DEFAULTS

        if data.get("discovery_method", "effect_size_fixed_v1") != "effect_size_fixed_v1":
            raise ValueError("effect_size_fixed_v1 discovery method cannot fall back")
        _positive_integer(
            data.get("shapelet_length_samples", 128),
            field="shapelet_length_samples",
        )
        _positive_integer(
            data.get("discovery_stride_samples", 64),
            field="discovery_stride_samples",
        )
        _finite_positive(
            data.get(
                "input_fs_hz",
                SHAPEFORMER_EXPERIMENTAL_NUMERIC_DEFAULTS["input_fs_hz"],
            ),
            field="input_fs_hz",
        )
        for field in (
            "shapelets_per_class", "max_candidates_per_class", "hidden_channels",
            "patch_size_samples", "attention_heads", "attention_layers",
            "attention_feedforward_channels", "distance_position_chunk_size",
        ):
            _positive_integer(
                data.get(field, SHAPEFORMER_EXPERIMENTAL_NUMERIC_DEFAULTS[field]),
                field=field,
            )
        if int(
            data.get(
                "hidden_channels",
                SHAPEFORMER_EXPERIMENTAL_NUMERIC_DEFAULTS["hidden_channels"],
            )
        ) % int(
            data.get(
                "attention_heads",
                SHAPEFORMER_EXPERIMENTAL_NUMERIC_DEFAULTS["attention_heads"],
            )
        ):
            raise ValueError("hidden_channels must be divisible by attention_heads")
        _probability(
            data.get(
                "dropout", SHAPEFORMER_EXPERIMENTAL_NUMERIC_DEFAULTS["dropout"]
            ),
            field="dropout",
        )
        if int(
            data.get(
                "patch_size_samples",
                SHAPEFORMER_EXPERIMENTAL_NUMERIC_DEFAULTS["patch_size_samples"],
            )
        ) < 2:
            raise ValueError(
                "patch_size_samples must be at least 2; raw sample-token "
                "attention is not an executable ShapeFormer capability"
            )
    # Exercise the single top-level input path during config validation. The
    # result is regenerated again with the fold's concrete input specification
    # before execution and persisted as derived provenance.
    materialize_model_architecture(data, representation_mode)
    return {"canonical_model_name": normalized_name, "machine_model_id": machine_id}


_WINDOW_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "engineering": {
        "length_s": 10.0,
        "hop_s": 5.0,
        "end_alignment": "left_start_regular_grid",
        "padding": "none_complete_windows_only",
        "cap_per_file": None,
        "cap_fraction_per_file": None,
        "min_valid_fraction": 1.0,
    },
    "raw_dl": {
        "length_s": 5.0,
        "hop_s": 2.5,
        "end_alignment": "include_right_aligned_if_distinct",
        "padding": "none_complete_windows_only",
        "cap_per_file": 128,
        "cap_fraction_per_file": None,
        "min_valid_fraction": 1.0,
    },
}


def normalize_window_config(section: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize the complete user-facing window configuration.

    The returned mapping uses YAML/config field names, not ``WindowPlan`` field
    names.  Persisting this mapping makes an omitted default and the runtime
    behavior independently auditable.
    """

    data = dict(section)
    allowed_section_keys = {"engineering", "raw_dl", "shared_planner_version"}
    unknown_section_keys = sorted(set(data) - allowed_section_keys)
    if unknown_section_keys:
        raise ValueError(
            f"windows section contains unknown fields: {unknown_section_keys}"
        )
    if data.get("shared_planner_version", "window_plan_v1") != "window_plan_v1":
        raise ValueError("unsupported shared window planner version")

    normalized: dict[str, Any] = {
        "shared_planner_version": "window_plan_v1",
    }
    for name, defaults in _WINDOW_PROFILE_DEFAULTS.items():
        candidate = data.get(name, {})
        if not isinstance(candidate, Mapping):
            raise ValueError(f"windows.{name} must be a mapping")
        unknown = sorted(set(candidate) - set(defaults))
        if unknown:
            raise ValueError(f"windows.{name} contains unknown fields: {unknown}")
        item = {**defaults, **dict(candidate)}
        if item["end_alignment"] not in {
            "left_start_regular_grid",
            "include_right_aligned_if_distinct",
        }:
            raise ValueError(f"unsupported windows.{name}.end_alignment")
        padding_modes = {
            "none_complete_windows_only",
            "right_zero_pad_short_records",
            "right_zero_pad_tail",
            "right_zero_pad_short_records_and_tail",
        }
        if item["padding"] not in padding_modes:
            raise ValueError(
                f"windows.{name}.padding must be one of {sorted(padding_modes)}"
            )
        # Engineering features have no mask-aware zero-padding semantics.  They
        # still accept arbitrary complete window/hop/alignment/cap parameters;
        # padding is available on the mask-aware raw/fusion representation.
        if name == "engineering" and item["padding"] != "none_complete_windows_only":
            raise ValueError("engineering windows do not support padded feature rows")
        for field_name in ("length_s", "hop_s"):
            raw_value = item[field_name]
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"windows.{name}.{field_name} must be numeric")
            value = float(raw_value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"windows.{name}.{field_name} must be finite and positive"
                )
            item[field_name] = value
        cap = item["cap_per_file"]
        if cap is not None and (
            isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0
        ):
            raise ValueError(f"windows.{name}.cap_per_file must be null or positive int")
        cap_fraction = item["cap_fraction_per_file"]
        if cap_fraction is not None and (
            isinstance(cap_fraction, bool)
            or not isinstance(cap_fraction, (int, float))
            or not math.isfinite(float(cap_fraction))
            or not 0.0 < float(cap_fraction) <= 1.0
        ):
            raise ValueError(
                f"windows.{name}.cap_fraction_per_file must be null or finite in (0,1]"
            )
        if cap is not None and cap_fraction is not None:
            raise ValueError(
                f"windows.{name}.cap_per_file and cap_fraction_per_file are mutually exclusive"
            )
        item["cap_fraction_per_file"] = (
            None if cap_fraction is None else float(cap_fraction)
        )
        min_valid = item["min_valid_fraction"]
        if isinstance(min_valid, bool) or not isinstance(min_valid, (int, float)):
            raise ValueError(f"windows.{name}.min_valid_fraction must be numeric")
        if not math.isfinite(float(min_valid)) or not 0.0 < float(min_valid) <= 1.0:
            raise ValueError(
                f"windows.{name}.min_valid_fraction must be finite in (0,1]"
            )
        if (
            item["padding"] == "none_complete_windows_only"
            and float(min_valid) != 1.0
        ):
            raise ValueError(
                f"windows.{name}.min_valid_fraction is only variable when padding is enabled"
            )
        item["min_valid_fraction"] = float(min_valid)
        normalized[name] = item
    return normalized


def validate_window_profiles_for_representation(
    section: Mapping[str, Any],
    representation_mode: str,
    enabled_feature_groups: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Reject non-default window controls with no runtime consumer.

    Raw models consume only ``raw_dl``. Feature matrices always consume the
    engineering sequence as matrix rows. Feature vectors and fusion consume the
    engineering profile only when ``engineering_summary`` is selected; fusion
    additionally always consumes ``raw_dl``. Default materialisation stays
    stable, while a non-default dormant profile is rejected instead of changing
    only the configuration hash.
    """

    normalized = normalize_window_config(section)
    mode = str(representation_mode)
    if mode not in {"raw", "feature_vector", "feature_matrix", "fusion"}:
        raise ValueError(f"unsupported representation_mode: {mode!r}")
    if not isinstance(enabled_feature_groups, (list, tuple)):
        raise ValueError("enabled_feature_groups must be a list or tuple")
    groups = tuple(str(value) for value in enabled_feature_groups)
    inactive_profiles: list[str] = []
    if mode == "raw":
        inactive_profiles.append("engineering")
    elif mode in {"feature_vector", "feature_matrix"}:
        inactive_profiles.append("raw_dl")
    if (
        mode in {"feature_vector", "fusion"}
        and "engineering_summary" not in groups
    ):
        inactive_profiles.append("engineering")
    for profile_name in inactive_profiles:
        defaults = _WINDOW_PROFILE_DEFAULTS[profile_name]
        observed = normalized[profile_name]
        changed = sorted(
            name
            for name, default in defaults.items()
            if observed[name] != default
        )
        if changed:
            raise ValueError(
                f"windows.{profile_name} is inactive for representation_mode="
                f"{mode!r}; non-default fields would have no runtime consumer: "
                f"{changed}"
            )
    return normalized


def resolve_window_config(section: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve configurable physical windows to the shared ``WindowPlan``.

    Omitted fields receive the preceding V2 defaults (engineering 10/5 seconds
    on a left-start grid without a cap; raw 5/2.5 seconds on the same start grid
    plus a distinct right-edge window, capped at 128). Explicit durations may be
    any finite positive values; the planner subsequently verifies that they map
    to whole samples at the selected input rate. Runtime supplies only
    ``source_record_id``.
    """

    data = normalize_window_config(section)

    def resolve(name: str) -> dict[str, Any]:
        item = dict(data[name])
        alignment_map = {
            "left_start_regular_grid": "start",
            "include_right_aligned_if_distinct": (
                "include_right_aligned_if_distinct"
            ),
        }
        padding_map = {
            "none_complete_windows_only": ("reject", False),
            "right_zero_pad_short_records": ("pad_right", False),
            "right_zero_pad_tail": ("reject", True),
            "right_zero_pad_short_records_and_tail": ("pad_right", True),
        }
        short_record_action, include_padded_tail = padding_map[str(item["padding"])]
        if include_padded_tail and alignment_map[str(item["end_alignment"])] != "start":
            raise ValueError(
                f"windows.{name} padded tails require left_start_regular_grid alignment"
            )
        cap = item["cap_per_file"]
        cap_fraction = item["cap_fraction_per_file"]
        has_cap = cap is not None or cap_fraction is not None
        return {
            "window_seconds": float(item["length_s"]),
            "hop_seconds": float(item["hop_s"]),
            "end_alignment": alignment_map[str(item["end_alignment"])],
            "short_record_action": short_record_action,
            "include_padded_tail": include_padded_tail,
            "max_windows": cap,
            "cap_policy": "uniform_progress" if has_cap else "not_applicable",
            "min_valid_fraction": float(item["min_valid_fraction"]),
            "max_window_fraction": cap_fraction,
        }

    return {"engineering": resolve("engineering"), "raw_dl": resolve("raw_dl")}


__all__ = [
    "AGGREGATION_MODULES", "ALL_MODULES", "ARTIFACT_MODULES",
    "CLASS_COUNT_BASIS_MODULES", "CLASS_WEIGHTING_MODULES",
    "COMPARISON_PROFILE_MODULES", "EPOCH_SELECTION_MODULES",
    "FEATURE_GROUP_MODULES", "LOSS_MODULES", "MODEL_MODULES",
    "MOTION_EVIDENCE_MODULES", "NORMALIZATION_MODULES", "OPTIMIZER_MODULES",
    "PRV_BACKEND_MODULES", "QUALITY_MODE_MODULES",
    "QUALITY_WEIGHT_SOURCE_MODULES", "REPRESENTATION_MODULES",
    "SAMPLER_MODULES", "SHAPEFORMER_DISCOVERY_BALANCE_MODULES",
    "TRAINING_BALANCE_MODULES", "WINDOW_QUALITY_SELECTION_MODULES",
    "ModuleDescriptor", "derived_mask_aware_pooling", "derived_model_ensemble_size",
    "derived_model_variant", "list_modules",
    "materialize_model_architecture", "model_factory_contract", "model_runtime_dependencies",
    "registry_sha256", "resolve_artifact_config",
    "normalize_window_config", "resolve_artifact_module_id", "resolve_window_config",
    "validate_window_profiles_for_representation",
    "validate_legacy_ensemble_metadata", "validate_model_config",
]
