"""V1 模块注册表与严格配置适配 / V1 module registry and strict adapters.

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
    ModuleDescriptor("raw", "representation", "ppg_frailty.representations.raw", ("raw",), "reference", "integration", "window->file->role->participant"),
    ModuleDescriptor("feature_vector", "representation", "ppg_frailty.representations.feature_vector", ("feature_vector",), "reference", "integration", "one vector per recording"),
    ModuleDescriptor("feature_matrix", "representation", "ppg_frailty.representations.feature_matrix", ("feature_matrix",), "reference", "integration", "one D-by-32 matrix per recording"),
    ModuleDescriptor("fusion", "representation", "ppg_frailty.representations.fusion", ("fusion",), "reference", "integration", "file-level window pooling then one vector concatenation"),
)

ARTIFACT_MODULES = (
    ModuleDescriptor("identity", "artifact", "ppg_frailty.artifact.identity.IdentityReducer", ("raw", "feature_vector", "feature_matrix", "fusion"), "direct_control", "artifacts", "exact no-op; morphology remains eligible"),
    ModuleDescriptor("nlms_imu_anc", "artifact", "ppg_frailty.artifact.nlms.NlmsReducer", ("feature_vector",), "comparison_rate_only", "artifacts", "ANC assumption may remove physiological response"),
    ModuleDescriptor("ssa_decomposition", "artifact", "ppg_frailty.artifact.decomposition.SsaReducer", ("feature_vector",), "comparison_rate_only", "artifacts", "non-stationary decomposition comparator"),
    ModuleDescriptor("spectral_mask", "artifact", "ppg_frailty.artifact.spectral.SpectralMaskReducer", ("feature_vector",), "comparison_rate_only", "artifacts", "formal STFT plus IMU soft mask"),
    ModuleDescriptor("pca_bss", "artifact", "ppg_frailty.artifact.bss.PcaBssReducer", ("feature_vector",), "comparison_rate_only", "artifacts", "two-wavelength BSS comparator"),
    ModuleDescriptor("fastica_bss", "artifact", "ppg_frailty.artifact.bss.FastIcaBssReducer", ("feature_vector",), "comparison_rate_only", "artifacts", "two-wavelength BSS comparator"),
    ModuleDescriptor("nmf_bss", "artifact", "ppg_frailty.artifact.bss.NmfBssReducer", ("feature_vector",), "comparison_rate_only", "artifacts", "two-wavelength spectral BSS comparator"),
)

MODEL_MODULES = (
    ModuleDescriptor("CompactCNN1D", "model", "ppg_frailty.models.compact_cnn.CompactCNN1D", ("raw",), "reference_not_wang_fcn", "models", "32/64/128 legacy-reference CNN"),
    ModuleDescriptor("InceptionTimeFull", "model", "ppg_frailty.models.inception_time_port.InceptionTimeSingleNetwork", ("raw",), "single_network", "models", "full single network, not five-member ensemble"),
    ModuleDescriptor("InceptionTimeSmall", "model", "ppg_frailty.models.inception_time_port.InceptionTimeSingleNetwork", ("raw",), "single_network", "models", "small single network"),
    ModuleDescriptor("InceptionTimeMatrix", "model", "ppg_frailty.models.inception_time_port.InceptionTimeSingleNetwork", ("feature_matrix",), "single_network", "models", "mask-aware D-by-32 input"),
    ModuleDescriptor("InceptionTimeFiveMemberEnsemble", "model", "ppg_frailty.models.inception_ensemble.InceptionTimeFiveMemberProbabilityEnsemble", ("raw", "feature_matrix"), "optional_five_member_ensemble", "models", "exact arithmetic probability mean"),
    ModuleDescriptor("ROCKET", "model", "ppg_frailty.models.rocket_ridge.RocketRidgeClassifier", ("feature_matrix",), "reference_10000_kernels", "models", "fold-local scaler, kernels and ridge"),
    ModuleDescriptor("MiniROCKET", "model", "ppg_frailty.models.rocket_ridge.MiniRocketAblation", ("feature_matrix",), "engineering_ablation_not_reference_port", "models", "cannot replace ROCKET silently"),
    ModuleDescriptor("LogisticRegressionL2", "model", "ppg_frailty.models.feature_models.FeatureVectorBaseline", ("feature_vector",), "reference", "models", "fold-local imputer and scaler"),
    ModuleDescriptor("RBFSVM", "model", "ppg_frailty.models.feature_models.FeatureVectorBaseline", ("feature_vector",), "reference", "models", "fold-local imputer and scaler"),
    ModuleDescriptor("ExtraTrees", "model", "ppg_frailty.models.feature_models.FeatureVectorBaseline", ("feature_vector",), "reference", "models", "fold-local imputer"),
    ModuleDescriptor("ShapeFormerEffectSize", "model", "ppg_frailty.models.shapeformer_port.ExperimentalShapeFormer", ("raw", "feature_matrix"), "experimental_non_pisd", "models", "effect-size discovery must be named"),
    ModuleDescriptor("FileBagFusionCompact", "model", "ppg_frailty.models.file_fusion.FileBagFusionClassifier", ("fusion",), "reference", "models", "file-level concatenation once"),
    ModuleDescriptor("FileBagFusionInception", "model", "ppg_frailty.models.file_fusion.FileBagFusionClassifier", ("fusion",), "reference", "models", "file-level concatenation once"),
)

ALL_MODULES = REPRESENTATION_MODULES + ARTIFACT_MODULES + MODEL_MODULES


def list_modules(family: str = "all") -> list[dict[str, Any]]:
    """稳定排序导出模块 / Export modules in stable order."""

    allowed = {"all", "representation", "artifact", "model"}
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
            f"artifact.reducer={declared!r} has no exact V1 adapter; "
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
    "InceptionTimeFiveMemberEnsemble": {"raw", "feature_matrix"},
    "ROCKET": {"feature_matrix"},
    "MiniROCKET": {"feature_matrix"},
    "LogisticRegressionL2": {"feature_vector"},
    "RBFSVM": {"feature_vector"},
    "ExtraTrees": {"feature_vector"},
    "ShapeFormerEffectSize": {"raw", "feature_matrix"},
    "FileBagFusionCompact": {"fusion"},
    "FileBagFusionInception": {"fusion"},
}


def validate_model_config(section: Mapping[str, Any], representation_mode: str) -> dict[str, str]:
    """检查模型元数据与 representation / Check model metadata against its mode."""

    data = dict(section)
    required = {
        "model_id", "variant", "input_channels", "input_channels_resolution",
        "n_classes", "ensemble_size", "member_seeds", "shapeformer_discovery_method",
        "rocket_kernel_count", "rocket_ridge_alpha", "mask_aware_pooling",
    }
    if set(data) != required:
        raise ValueError(
            f"model key mismatch: missing={sorted(required-set(data))}, "
            f"unknown={sorted(set(data)-required)}"
        )
    canonical = str(data["model_id"])
    if canonical not in _MODEL_MODES:
        raise ValueError(f"model_id is not registered: {canonical}")
    if representation_mode not in _MODEL_MODES[canonical]:
        raise ValueError(f"{canonical} is incompatible with {representation_mode}")
    if int(data["n_classes"]) != 3:
        raise ValueError("reference task requires exactly three classes")
    seeds = data["member_seeds"]
    if not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("member_seeds must be a non-empty distinct list")
    ensemble_size = int(data["ensemble_size"])
    expected = 5 if canonical == "InceptionTimeFiveMemberEnsemble" else 1
    if ensemble_size != expected or len(seeds) != expected:
        raise ValueError(f"{canonical} requires ensemble_size/member_seeds={expected}")
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
    "ALL_MODULES", "ARTIFACT_MODULES", "MODEL_MODULES", "REPRESENTATION_MODULES",
    "ModuleDescriptor", "list_modules", "registry_sha256", "resolve_artifact_config",
    "resolve_artifact_module_id", "resolve_window_config", "validate_model_config",
]
