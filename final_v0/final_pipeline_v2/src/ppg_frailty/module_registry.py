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


REPRESENTATION_MODULES = (
    ModuleDescriptor("raw", "representation", "ppg_frailty.representations.raw", ("raw",), "reference", "integration", "Line A window->file->participant; Line B window->file->role_family->participant"),
    ModuleDescriptor("feature_vector", "representation", "ppg_frailty.representations.feature_vector", ("feature_vector",), "reference", "integration", "one vector per recording"),
    ModuleDescriptor("feature_matrix", "representation", "ppg_frailty.representations.feature_matrix", ("feature_matrix",), "reference", "integration", "one D-by-32 matrix per recording"),
    ModuleDescriptor("fusion", "representation", "ppg_frailty.representations.fusion", ("fusion",), "reference", "integration", "file-level window pooling then one vector concatenation"),
)

ARTIFACT_MODULES = (
    ModuleDescriptor("identity", "artifact", "ppg_frailty.artifact.identity.IdentityReducer", ("raw", "feature_vector", "feature_matrix", "fusion"), "direct_control", "artifacts", "exact no-op; morphology remains eligible"),
    ModuleDescriptor("nlms_imu_anc", "artifact", "ppg_frailty.artifact.nlms.NlmsReducer", ("feature_vector",), "comparison_rate_only", "artifacts", "ANC assumption may remove physiological response; current rate-only endpoint integration is feature_vector only, not a claim that reducer mathematics is representation-specific"),
    ModuleDescriptor("ssa_decomposition", "artifact", "ppg_frailty.artifact.decomposition.SsaReducer", ("feature_vector",), "comparison_rate_only", "artifacts", "non-stationary decomposition comparator; current rate-only endpoint integration is feature_vector only, not a claim that reducer mathematics is representation-specific"),
    ModuleDescriptor("spectral_mask", "artifact", "ppg_frailty.artifact.spectral.SpectralMaskReducer", ("feature_vector",), "comparison_rate_only", "artifacts", "formal STFT plus IMU soft mask; current rate-only endpoint integration is feature_vector only, not a claim that reducer mathematics is representation-specific"),
    ModuleDescriptor("pca_bss", "artifact", "ppg_frailty.artifact.bss.PcaBssReducer", ("feature_vector",), "comparison_rate_only", "artifacts", "two-wavelength BSS comparator; current rate-only endpoint integration is feature_vector only, not a claim that reducer mathematics is representation-specific"),
    ModuleDescriptor("fastica_bss", "artifact", "ppg_frailty.artifact.bss.FastIcaBssReducer", ("feature_vector",), "comparison_rate_only", "artifacts", "two-wavelength BSS comparator; current rate-only endpoint integration is feature_vector only, not a claim that reducer mathematics is representation-specific"),
    ModuleDescriptor("nmf_bss", "artifact", "ppg_frailty.artifact.bss.NmfBssReducer", ("feature_vector",), "comparison_rate_only", "artifacts", "two-wavelength spectral BSS comparator; current rate-only endpoint integration is feature_vector only, not a claim that reducer mathematics is representation-specific"),
    ModuleDescriptor("emd_sifting_rate_only", "artifact", "ppg_frailty.artifact.legacy.EmdSiftingRateOnlyReducer", ("feature_vector",), "comparison_rate_only", "artifacts", "named EMD sifting ablation; never morphology-preserving; current rate-only endpoint integration is feature_vector only, not a claim that reducer mathematics is representation-specific"),
    ModuleDescriptor("ceemd_lite_nlms_legacy", "artifact", "ppg_frailty.artifact.legacy.CeemdLiteNlmsLegacyReducer", ("feature_vector",), "comparison_rate_only", "artifacts", "named CEEMD-lite plus NLMS legacy ablation; current rate-only endpoint integration is feature_vector only, not a claim that reducer mathematics is representation-specific"),
    ModuleDescriptor("dwt_a2_legacy", "artifact", "ppg_frailty.artifact.legacy.DwtA2LegacyReducer", ("feature_vector",), "comparison_rate_only", "artifacts", "named DWT A2 legacy ablation; PyWavelets is optional; current rate-only endpoint integration is feature_vector only, not a claim that reducer mathematics is representation-specific"),
)

PRV_BACKEND_MODULES = (
    ModuleDescriptor("local", "prv_backend", "ppg_frailty.features.prv_backend_compare.evaluate_prv_backend", ("feature_vector", "feature_matrix", "fusion"), "formal_primary", "prv_backend_output", "local manual PRV remains the only classifier-pipeline backend"),
    ModuleDescriptor("aura_hrv_analysis", "prv_backend", "ppg_frailty.features.prv_backend_compare.evaluate_prv_backend", (), "function_comparison_only", "prv_backend_output", "fixed PPI vectors only; never enters classifier training"),
    ModuleDescriptor("rhenan_hrv", "prv_backend", "ppg_frailty.features.prv_backend_compare.evaluate_prv_backend", (), "legacy_function_comparison_only", "prv_backend_output", "separate legacy requirements; never enters classifier training"),
)

MOTION_OPTION_MODULES = (
    ModuleDescriptor("sqi_only", "motion_option", "ppg_frailty.quality.motion.resolve_motion_option", (), "formal_default_motion_not_computed", "motion_contract", "V2-010 default; SQI remains independently off unless explicitly requested"),
    ModuleDescriptor("sqi_plus_motion_override", "motion_option", "ppg_frailty.quality.motion.resolve_motion_option", (), "registered_gate_closed_not_run", "motion_contract", "requires complete Frailty29 5x5 OOF evidence before PTT evaluation"),
    ModuleDescriptor("historical_light_cnn_backup", "motion_option", "ppg_frailty.quality.motion.HISTORICAL_LIGHT_CNN_EVIDENCE", (), "historical_frozen_backup_not_v2_run", "motion_contract", "archived SIM evidence only; never a V2 or PTT result"),
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
    ModuleDescriptor("CompactCNN1D", "model", "ppg_frailty.models.compact_cnn.CompactCNN1D", ("raw",), "reference_not_wang_fcn", "models", "32/64/128 legacy-reference CNN"),
    ModuleDescriptor("InceptionTimeFull", "model", "ppg_frailty.models.inception_time_port.InceptionTimeSingleNetwork", ("raw",), "single_network", "models", "full single network, not five-member ensemble"),
    ModuleDescriptor("InceptionTimeSmall", "model", "ppg_frailty.models.inception_time_port.InceptionTimeSingleNetwork", ("raw",), "single_network", "models", "small single network"),
    ModuleDescriptor("InceptionTimeMatrix", "model", "ppg_frailty.models.inception_time_port.InceptionTimeSingleNetwork", ("feature_matrix",), "single_network", "models", "mask-aware D-by-32 input"),
    ModuleDescriptor("InceptionTimeFullFiveMemberEnsemble", "model", "ppg_frailty.models.inception_ensemble.InceptionTimeFiveMemberProbabilityEnsemble", ("raw",), "comparison_only_five_member_ensemble", "models", "five independent full raw models and exact probability mean"),
    ModuleDescriptor("InceptionTimeMatrixFiveMemberEnsemble", "model", "ppg_frailty.models.inception_ensemble.InceptionTimeFiveMemberProbabilityEnsemble", ("feature_matrix",), "comparison_only_five_member_ensemble", "models", "five independent full matrix models and exact probability mean"),
    ModuleDescriptor("ROCKET", "model", "ppg_frailty.models.rocket_ridge.RocketRidgeClassifier", ("feature_matrix",), "reference_10000_kernels", "models", "fold-local scaler, kernels and ridge"),
    ModuleDescriptor("MiniROCKET", "model", "ppg_frailty.models.rocket_ridge.MiniRocketAblation", ("feature_matrix",), "engineering_ablation_not_reference_port", "models", "cannot replace ROCKET silently"),
    ModuleDescriptor("LogisticRegressionL2", "model", "ppg_frailty.models.feature_models.FeatureVectorBaseline", ("feature_vector",), "reference", "models", "fold-local imputer and scaler"),
    ModuleDescriptor("RBFSVM", "model", "ppg_frailty.models.feature_models.FeatureVectorBaseline", ("feature_vector",), "reference", "models", "fold-local imputer and scaler"),
    ModuleDescriptor("ExtraTrees", "model", "ppg_frailty.models.feature_models.FeatureVectorBaseline", ("feature_vector",), "reference", "models", "fold-local imputer"),
    ModuleDescriptor("ShapeFormerChannelSpecificOSD", "model", "ppg_frailty.models.shapeformer_literature.LiteratureShapeFormerChannelSpecificOSD", ("raw",), "implemented_not_benchmarked_high_compute", "models", "faithful channel-specific OSD/PISD discovery plus ShapeBlock/IG route; never fixed-length and never fallback"),
    ModuleDescriptor("ShapeFormerEffectSizeFixedV1", "model", "ppg_frailty.models.shapeformer_port.ExperimentalShapeFormer", ("raw",), "fixed_length_effect_size_ablation", "models", "128-sample stride-64 effect-size ablation, never labelled PISD"),
    ModuleDescriptor("FileBagFusionCompact", "model", "ppg_frailty.models.file_fusion.FileBagFusionClassifier", ("fusion",), "reference", "models", "file-level concatenation once"),
    ModuleDescriptor("FileBagFusionInception", "model", "ppg_frailty.models.file_fusion.FileBagFusionClassifier", ("fusion",), "reference", "models", "file-level concatenation once"),
)

ALL_MODULES = (
    REPRESENTATION_MODULES
    + ARTIFACT_MODULES
    + PRV_BACKEND_MODULES
    + MOTION_OPTION_MODULES
    + COMPARISON_PROFILE_MODULES
    + MODEL_MODULES
)


def list_modules(family: str = "all") -> list[dict[str, Any]]:
    """稳定排序导出模块 / Export modules in stable order."""

    allowed = {"all", "representation", "artifact", "prv_backend", "motion_option", "comparison_profile", "model"}
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
    if declared != "identity" and data["non_identity_output_contract"] != "rate_only":
        raise ValueError("non-identity artifact output must be rate_only")
    parameters = data["parameters"]
    if not isinstance(parameters, Mapping):
        raise ValueError("artifact.parameters must be a mapping")
    # 中文：调用真实 factory 同时验证具体 dataclass 参数；English: the real
    # factory validates each concrete reducer dataclass and rejects unknown keys.
    from .artifact import get_reducer

    runtime_name = _ARTIFACT_CONFIG_TO_RUNTIME[declared]
    reducer = get_reducer(runtime_name, dict(parameters))
    return {
        "declared_reducer": declared,
        "runtime_reducer": runtime_name,
        "declared_version": str(data["reducer_version"]),
        "runtime_version": str(reducer.reducer_version),
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
    "ShapeFormerEffectSizeFixedV1": {"raw"},
    "FileBagFusionCompact": {"fusion"},
    "FileBagFusionInception": {"fusion"},
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
        "mask_aware_pooling", "input_channel_order",
    },
    "InceptionTimeFull": {
        "seed_policy", "dropout", "kernel_sizes", "dilation", "mask_aware_pooling",
        "input_channel_order",
    },
    "InceptionTimeSmall": {
        "seed_policy", "dropout", "kernel_sizes", "dilation", "mask_aware_pooling",
        "input_channel_order",
    },
    "InceptionTimeMatrix": {
        "seed_policy", "dropout", "kernel_sizes", "dilation", "mask_aware_pooling",
    },
    "InceptionTimeFullFiveMemberEnsemble": {
        "comparison_only", "member_seeds", "seed_policy",
        "member_seed_roster_id", "dropout", "kernel_sizes", "dilation",
        "mask_aware_pooling", "input_channel_order",
    },
    "InceptionTimeMatrixFiveMemberEnsemble": {
        "comparison_only", "member_seeds", "seed_policy",
        "member_seed_roster_id", "dropout", "kernel_sizes", "dilation",
        "mask_aware_pooling",
    },
    "ROCKET": {"seed_policy", "n_kernels", "alpha"},
    "MiniROCKET": {"seed_policy", "n_kernels", "alpha"},
    "LogisticRegressionL2": {
        "seed_policy", "class_weight", "logistic_max_iter", "logistic_solver",
    },
    "RBFSVM": {
        "seed_policy", "class_weight", "svm_kernel", "svm_probability",
        "svm_c", "svm_gamma",
    },
    "ExtraTrees": {
        "seed_policy", "class_weight", "extra_trees_n_estimators",
        "extra_trees_n_jobs",
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
    "ShapeFormerEffectSizeFixedV1": {
        "seed_policy", "input_channel_order", "discovery_method", "input_fs_hz",
        "shapelet_length_samples", "shapelets_per_class",
        "discovery_stride_samples", "max_candidates_per_class",
        "hidden_channels", "dropout", "patch_size_samples",
        "attention_heads", "attention_layers",
        "distance_position_chunk_size", "mask_aware_pooling",
    },
    "FileBagFusionCompact": {
        "seed_policy", "signal_dropout", "signal_kernel_sizes",
        "signal_dilations", "signal_pool_sizes", "feature_hidden_dim",
        "fusion_hidden_dim", "pooling", "dropout", "mask_aware_pooling",
        "input_channel_order",
    },
    "FileBagFusionInception": {
        "seed_policy", "signal_variant", "signal_dropout",
        "signal_kernel_sizes", "signal_dilation", "feature_hidden_dim",
        "fusion_hidden_dim", "pooling", "dropout", "mask_aware_pooling",
        "input_channel_order",
    },
}


def validate_model_config(section: Mapping[str, Any], representation_mode: str) -> dict[str, str]:
    """检查模型元数据与 representation / Check model metadata against its mode."""

    data = dict(section)
    canonical = str(data.get("model_id", ""))
    if canonical not in _MODEL_MODES or canonical not in _MODEL_SPECIFIC_FIELDS:
        raise ValueError(f"model_id is not registered: {canonical}")
    required = _MODEL_BASE_FIELDS | _MODEL_SPECIFIC_FIELDS[canonical]
    if set(data) != required:
        raise ValueError(
            f"model key mismatch: missing={sorted(required-set(data))}, "
            f"unknown={sorted(set(data)-required)}"
        )
    if representation_mode not in _MODEL_MODES[canonical]:
        raise ValueError(f"{canonical} is incompatible with {representation_mode}")
    if int(data["n_classes"]) != 3:
        raise ValueError("reference task requires exactly three classes")
    if "mask_aware_pooling" in data and not isinstance(data["mask_aware_pooling"], bool):
        raise ValueError("mask_aware_pooling must be explicit boolean")
    ensemble_size = int(data["ensemble_size"])
    ensemble_names = {
        "InceptionTimeFullFiveMemberEnsemble",
        "InceptionTimeMatrixFiveMemberEnsemble",
    }
    expected = 5 if canonical in ensemble_names else 1
    if ensemble_size != expected:
        raise ValueError(f"{canonical} requires ensemble_size={expected}")
    if canonical in ensemble_names:
        seeds = data["member_seeds"]
        if (
            not isinstance(seeds, list)
            or not seeds
            or len(seeds) != len(set(seeds))
        ):
            raise ValueError("member_seeds must be a non-empty distinct list")
        if seeds != [50042, 60042, 70042, 80042, 90042]:
            raise ValueError("V2 ensemble member seeds must match the confirmed sequence")
        if data["comparison_only"] is not True:
            raise ValueError("five-member ensembles are explicit comparison-only routes")
        if data["member_seed_roster_id"] != "cv_fixed_five_member_seed_roster":
            raise ValueError("V2 ensemble comparison requires the named CV member roster")
        if data["seed_policy"] != "cv_fixed_five_member_seed_roster":
            raise ValueError("V2 ensemble CV requires the fixed five-member seed roster")
    elif (
        canonical in {"InceptionTimeFull", "InceptionTimeMatrix"}
        and data["seed_policy"] == "cv_fixed_member0_seed_50042_comparator"
    ):
        pass
    elif data["seed_policy"] != "outer_cv_repeat_seed_equals_split_seed":
        raise ValueError(
            "V2 outer-CV single models require "
            "outer_cv_repeat_seed_equals_split_seed"
        )
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
    if canonical == "ShapeFormerChannelSpecificOSD":
        if data["discovery_method"] != "channel_specific_osd":
            raise ValueError("ShapeFormer reference requires channel_specific_osd")
        if (
            float(data["input_fs_hz"]) != 400.0
            or float(data["num_pip_ratio"]) != 0.20
            or int(data["shapelets_per_class"]) != 3
            or int(data["max_discovery_windows"]) != 180
            or data["discovery_balance"] != "participant_file_balanced"
            or int(data["position_search_neighbourhood_samples"]) != 128
            or data["pip_rounding_rule"]
            != "floor_ratio_minimum_5_capped_at_actual_T"
            or data["pip_selection_rule"]
            != "upstream_zscored_time_index_perpendicular_distance_first_max"
            or data["candidate_generation_rule"]
            != "insertion_stage_three_consecutive_pips_half_open"
            or data["candidate_enumeration_rule"]
            != "upstream_class_channel_source_sample_insertion_order"
            or data["candidate_ranking_rule"]
            != "upstream_numpy_default_argsort_then_reverse"
            or data["selected_bank_order_rule"]
            != "upstream_per_class_start_sample_default_argsort"
            or data["discovery_position_search_boundary_rule"]
            != "upstream_pcs_start_minus_w_plus_1_end_plus_w_half_open"
            or data["information_gain_split_rule"]
            != "upstream_positive_recall_grid_0p2"
            or int(data["sequence_length_samples"]) != 2000
            or int(data["local_kernel_width_samples"]) != 8
            or int(data["local_embedding_channels"]) != 48
            or int(data["shape_embedding_channels"]) != 128
            or int(data["attention_feedforward_channels"]) != 256
            or int(data["attention_heads"]) != 4
            or int(data["attention_query_chunk_size"]) != 128
            or int(data["distance_position_chunk_size"]) != 256
            or float(data["dropout"]) != 0.30
            or float(data["complexity_norm"]) != 1000.0
            or float(data["max_complexity_ratio"]) != 3.0
        ):
            raise ValueError("channel_specific_osd discovery/architecture contract drifted")
    if canonical == "ShapeFormerEffectSizeFixedV1":
        if (
            data["discovery_method"] != "effect_size_fixed_v1"
            or int(data["shapelet_length_samples"]) != 128
            or int(data["discovery_stride_samples"]) != 64
        ):
            raise ValueError("effect_size_fixed_v1 must remain 128 samples / stride 64")
    from .models import normalize_model_id

    normalized_name, machine_id = normalize_model_id(canonical)
    return {"canonical_model_name": normalized_name, "machine_model_id": machine_id}


def resolve_window_config(section: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """将 YAML 窗口字段解析到唯一 data WindowPlan / Resolve the sole planner.

    中文：返回 ``data.windows.WindowPlan`` 的构造参数；运行时仅补充当前
    ``source_record_id``。English: return constructor fields for the sole
    ``data.windows.WindowPlan``; runtime supplies only ``source_record_id``.
    """

    data = dict(section)
    if set(data) != {"engineering", "raw_dl", "shared_planner_version"}:
        raise ValueError("windows section must explicitly contain engineering/raw_dl/shared_planner_version")
    if data["shared_planner_version"] != "window_plan_v1":
        raise ValueError("unsupported shared window planner version")

    def resolve(name: str, *, raw: bool) -> dict[str, Any]:
        candidate = data[name]
        if not isinstance(candidate, Mapping):
            raise ValueError(f"windows.{name} must be a mapping")
        item = dict(candidate)
        required = {"length_s", "hop_s", "end_alignment", "padding", "cap_per_file"}
        if raw:
            required.add("min_valid_fraction")
        if set(item) != required:
            raise ValueError(f"windows.{name} key mismatch")
        alignment_map = {
            "left_start_regular_grid": "start",
            "include_right_aligned_if_distinct": "end",
        }
        if item["end_alignment"] not in alignment_map:
            raise ValueError(f"unsupported windows.{name}.end_alignment")
        if item["padding"] != "none_complete_windows_only":
            raise ValueError(f"unsupported windows.{name}.padding")
        if raw and float(item["min_valid_fraction"]) != 1.0:
            raise ValueError("raw reference requires complete unpadded windows")
        cap = item["cap_per_file"]
        if cap is not None and (not isinstance(cap, int) or cap <= 0):
            raise ValueError(f"windows.{name}.cap_per_file must be null or positive int")
        return {
            "window_seconds": float(item["length_s"]),
            "hop_seconds": float(item["hop_s"]),
            "end_alignment": alignment_map[str(item["end_alignment"])],
            "short_record_action": "reject",
            "include_padded_tail": False,
            "max_windows": cap,
            "cap_policy": "uniform_progress" if cap is not None else "not_applicable",
        }

    return {"engineering": resolve("engineering", raw=False), "raw_dl": resolve("raw_dl", raw=True)}


__all__ = [
    "ALL_MODULES", "ARTIFACT_MODULES", "COMPARISON_PROFILE_MODULES", "MODEL_MODULES", "MOTION_OPTION_MODULES", "PRV_BACKEND_MODULES", "REPRESENTATION_MODULES",
    "ModuleDescriptor", "list_modules", "registry_sha256", "resolve_artifact_config",
    "resolve_artifact_module_id", "resolve_window_config", "validate_model_config",
]
