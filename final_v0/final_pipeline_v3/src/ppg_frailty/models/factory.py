"""Strict model factory spanning all four representation modes.

覆盖四种 representation mode 的严格模型工厂。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from ..contracts import RepresentationMode
from .feature_baselines import FeatureVectorBaseline
from .rocket import MiniRocketAblation, RocketRidgeClassifier


# English: The optional deep profile is the confirmed V2 dependency layer. Each
# execution environment must still pass a runtime probe before a deep run.
# 中文：可选 deep profile 已确认为 V2 依赖层；每个实际执行环境仍须先通过运行探测。
PYTORCH_DEPENDENCY_STATUS = "optional_deep_profile_confirmed_runtime_probe_required"

# Every formal config must materialise these fields before model construction;
# they are experiment identity, not documentation-only annotations.
FROZEN_MODEL_RUN_PROVENANCE_FIELDS = (
    "architecture_parameters",
    "input_channels_order",
    "sampling_rate_hz",
    "window_plan",
    "hop_plan",
    "normalization",
    "padding_mask",
    "feature_schema_hash",
    "sqi_routing",
    "loss",
    "class_weighting",
    "sampler",
    "epoch_rule",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "dropout",
    "label_smoothing",
    "gradient_clipping",
    "seed_policy",
    "random_seeds",
    "fold_hash",
    "aggregation",
    "calibration",
)


def validate_frozen_model_run_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless every V2-031 model/training identity is explicit."""

    payload = dict(value)
    missing = sorted(set(FROZEN_MODEL_RUN_PROVENANCE_FIELDS) - set(payload))
    if missing:
        raise ValueError(f"frozen model/run provenance is missing fields: {missing}")
    for name in FROZEN_MODEL_RUN_PROVENANCE_FIELDS:
        item = payload[name]
        if item is None or isinstance(item, str) and not item.strip():
            raise ValueError(f"frozen model/run provenance field {name!r} must be explicit")
        if isinstance(item, (tuple, list, dict)) and not item:
            raise ValueError(f"frozen model/run provenance field {name!r} must not be empty")
    channels = tuple(str(item) for item in payload["input_channels_order"])
    seeds = tuple(int(item) for item in payload["random_seeds"])
    if not channels or len(channels) != len(set(channels)):
        raise ValueError("input_channels_order must be non-empty and unique")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("random_seeds must be non-empty and unique")
    allowed_seed_policies = {
        "outer_cv_repeat_seed_equals_split_seed",
        "cv_fixed_five_member_seed_roster",
        "final_refit_single_seed_42",
        "final_refit_five_member_seeds",
    }
    if payload["seed_policy"] not in allowed_seed_policies:
        raise ValueError(
            f"seed_policy must be one of {sorted(allowed_seed_policies)}"
        )
    payload["input_channels_order"] = channels
    payload["random_seeds"] = seeds
    return payload


# English: Human specification names are mapped once to immutable machine IDs.
# Both are persisted in manifests; aliases outside this table are forbidden.
# 中文：规范中的人类可读名称只在此处映射一次到稳定 machine ID。manifest 同时
# 保存两者；本表之外的别名一律禁止。
CANONICAL_MODEL_REGISTRY: dict[str, str] = {
    "CompactCNN1D": "compact_cnn",
    "InceptionTimeFull": "inception_full",
    "InceptionTimeSmall": "inception_small",
    "InceptionTimeMatrix": "inception_matrix",
    "InceptionTimeFullFiveMemberEnsemble": "inception_full_five_member_ensemble",
    "InceptionTimeMatrixFiveMemberEnsemble": "inception_matrix_five_member_ensemble",
    "ROCKET": "rocket_numpy",
    "MiniROCKET": "minirocket_ablation",
    "LogisticRegressionL2": "logistic_regression",
    "RBFSVM": "rbf_svm",
    "ExtraTrees": "extra_trees",
    "ShapeFormerChannelSpecificOSD": "shapeformer_channel_specific_osd",
    "ShapeFormerChannelSpecificScalarDistanceAblation": (
        "shapeformer_channel_specific_scalar_distance_ablation"
    ),
    "ShapeFormerEffectSizeFixedV1": "shapeformer_effect_size_fixed_v1",
    "FileBagFusionCompact": "fusion_compact",
    "FileBagFusionInception": "fusion_inception",
}
_MACHINE_TO_CANONICAL = {machine: canonical for canonical, machine in CANONICAL_MODEL_REGISTRY.items()}


@dataclass(frozen=True)
class ModelCandidate:
    """One registered candidate identity; registration is not benchmark evidence."""

    canonical_name: str
    machine_id: str
    representation_mode: str
    registry_role: str
    comparison_family: str
    execution_status: str = "registered_runnable_contract_not_benchmarked"


NONENSEMBLE_MODEL_CANDIDATES: tuple[ModelCandidate, ...] = (
    ModelCandidate("CompactCNN1D", "compact_cnn", "raw", "reference", "compact"),
    ModelCandidate("InceptionTimeFull", "inception_full", "raw", "reference", "inception"),
    ModelCandidate("InceptionTimeSmall", "inception_small", "raw", "reference", "inception"),
    ModelCandidate("InceptionTimeMatrix", "inception_matrix", "feature_matrix", "reference", "inception"),
    ModelCandidate("ROCKET", "rocket_numpy", "feature_matrix", "reference", "rocket"),
    ModelCandidate("MiniROCKET", "minirocket_ablation", "feature_matrix", "ablation", "rocket"),
    ModelCandidate("LogisticRegressionL2", "logistic_regression", "feature_vector", "reference", "classical"),
    ModelCandidate("RBFSVM", "rbf_svm", "feature_vector", "reference", "classical"),
    ModelCandidate("ExtraTrees", "extra_trees", "feature_vector", "reference", "classical"),
    ModelCandidate("ShapeFormerChannelSpecificOSD", "shapeformer_channel_specific_osd", "raw", "reference", "shapeformer_discovery"),
    ModelCandidate("ShapeFormerEffectSizeFixedV1", "shapeformer_effect_size_fixed_v1", "raw", "ablation", "shapeformer_discovery"),
    ModelCandidate("FileBagFusionCompact", "fusion_compact", "fusion", "reference", "fusion"),
    ModelCandidate("FileBagFusionInception", "fusion_inception", "fusion", "reference", "fusion"),
)
RAW_FIVE_MEMBER_ENSEMBLE_COMPARISON = ModelCandidate(
    "InceptionTimeFullFiveMemberEnsemble",
    "inception_full_five_member_ensemble",
    "raw",
    "comparison",
    "ensemble",
    "registered_explicit_comparison_not_default_not_benchmarked",
)
MATRIX_FIVE_MEMBER_ENSEMBLE_COMPARISON = ModelCandidate(
    "InceptionTimeMatrixFiveMemberEnsemble",
    "inception_matrix_five_member_ensemble",
    "feature_matrix",
    "comparison",
    "ensemble",
    "registered_explicit_comparison_not_default_not_benchmarked",
)
# Backward import name now denotes both explicit comparisons, never an ambiguous model.
FIVE_MEMBER_ENSEMBLE_COMPARISONS = (
    RAW_FIVE_MEMBER_ENSEMBLE_COMPARISON,
    MATRIX_FIVE_MEMBER_ENSEMBLE_COMPARISON,
)
CHANNEL_SPECIFIC_SCALAR_DISTANCE_ABLATION = ModelCandidate(
    "ShapeFormerChannelSpecificScalarDistanceAblation",
    "shapeformer_channel_specific_scalar_distance_ablation",
    "raw",
    "ablation",
    "shapeformer_downstream",
    "named_optional_ablation_not_literature_shapeformer_not_default",
)
_CANDIDATE_BY_MACHINE_ID = {item.machine_id: item for item in NONENSEMBLE_MODEL_CANDIDATES}
if len(NONENSEMBLE_MODEL_CANDIDATES) != 13 or len(_CANDIDATE_BY_MACHINE_ID) != 13:
    raise RuntimeError("V2 requires exactly 13 unique non-ensemble candidate contracts")
if sum(item.registry_role == "reference" for item in NONENSEMBLE_MODEL_CANDIDATES) != 11:
    raise RuntimeError("V2 non-ensemble registry must contain 11 reference candidates")


def model_candidate(model_id_or_name: str) -> ModelCandidate:
    """Return one non-ensemble candidate record; the ensemble stays separate."""

    _, machine_id = normalize_model_id(model_id_or_name)
    try:
        return _CANDIDATE_BY_MACHINE_ID[machine_id]
    except KeyError as exc:
        raise ValueError(
            "ensemble/additional ablation entries are outside the 13 non-ensemble candidate set"
        ) from exc


def normalize_model_id(value: str) -> tuple[str, str]:
    """Return canonical name and machine ID, or fail closed.

    返回规范名称与 machine ID，未知名称关闭失败。
    """

    text = str(value)
    if text in CANONICAL_MODEL_REGISTRY:
        return text, CANONICAL_MODEL_REGISTRY[text]
    if text in _MACHINE_TO_CANONICAL:
        return _MACHINE_TO_CANONICAL[text], text
    raise ValueError(f"unsupported model_id/model_name: {text}")


def normalize_model_config(model_config: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalise the ID while preserving options / 规范化 ID 并保留选项。"""

    config = dict(model_config)
    supplied = config.get("model_id", config.get("model_name"))
    if supplied is None:
        raise ValueError("model_config requires model_id or model_name")
    if "model_id" in config and "model_name" in config:
        from_name = normalize_model_id(str(config["model_name"]))
        from_id = normalize_model_id(str(config["model_id"]))
        if from_name != from_id:
            raise ValueError("model_id and model_name resolve to different registry entries")
    canonical, machine = normalize_model_id(str(supplied))
    if machine in {
        "logistic_regression",
        "rbf_svm",
        "extra_trees",
        "rocket_numpy",
        "minirocket_ablation",
    }:
        requested = sorted(
            {"epochs", "fixed_epochs", "epoch_profile", "epoch_rule"} & set(config)
        )
        if requested:
            raise ValueError(
                "classical and ROCKET candidates do not accept epoch settings; "
                f"remove {requested}"
            )
    config.pop("model_name", None)
    config["model_id"] = machine
    config["canonical_model_name"] = canonical
    return config


@dataclass(frozen=True)
class ModelInputSpec:
    """Frozen dimensions and schema identifiers / 冻结的维度与 schema 标识。"""

    representation_mode: RepresentationMode | str
    n_channels: int = 0
    n_classes: int = 3
    n_file_features: int = 0
    feature_names: tuple[str, ...] = ()
    channel_schema: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate frozen dimensions / 校验冻结维度。"""

        RepresentationMode(self.representation_mode)
        if self.n_classes <= 1:
            raise ValueError("n_classes must exceed one")
        if self.n_channels < 0 or self.n_file_features < 0:
            raise ValueError("input dimensions cannot be negative")
        if self.feature_names and len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("feature_names must be unique")
        if self.channel_schema and len(self.channel_schema) != len(set(self.channel_schema)):
            raise ValueError("channel_schema must be unique")

    @classmethod
    def from_value(cls, value: "ModelInputSpec | Mapping[str, Any]") -> "ModelInputSpec":
        """Normalise a dataclass or mapping / 规范化数据类或映射。"""

        if isinstance(value, cls):
            return value
        return cls(**dict(value))

    @property
    def mode(self) -> RepresentationMode:
        """Return validated enum mode / 返回已校验枚举模式。"""

        return RepresentationMode(self.representation_mode)


def _normalise_architecture_value(value: Any) -> Any:
    """Convert one resolved architecture to a stable JSON-compatible value."""

    if isinstance(value, Mapping):
        return {
            str(key): _normalise_architecture_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_normalise_architecture_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("architecture parameters must be finite")
        return float(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported architecture parameter type: {type(value).__name__}")


def _compact_architecture(model: Any) -> dict[str, Any]:
    return {
        "stage_channels": tuple(int(value) for value in model.stage_channels),
        "kernel_sizes": tuple(int(value) for value in model.kernel_sizes),
        "dilations": tuple(int(value) for value in model.dilations),
        "pool_sizes": tuple(int(value) for value in model.pool_sizes),
        "stage_dropouts": tuple(float(value) for value in model.stage_dropouts),
        "global_pooling": "adaptive_average_1",
        "classifier_dropout": float(model.classifier_dropout),
    }


def _inception_architecture(model: Any) -> dict[str, Any]:
    return {
        "variant": str(model.variant),
        "out_channels": int(model.out_channels),
        "bottleneck_channels": int(model.bottleneck_channels),
        "depth": int(model.depth),
        "kernel_sizes": tuple(int(value) for value in model.kernel_sizes),
        "dilation": int(model.dilation),
        "branch_count": int(model.branch_count),
        "residual_interval": int(model.residual_interval),
        "global_pooling": "mask_aware_global_average",
        "classifier_dropout": float(model.classifier_dropout),
    }


def resolved_architecture_parameters(
    model: Any, input_spec: ModelInputSpec | Mapping[str, Any]
) -> dict[str, Any]:
    """Read the constructed object and return its complete V2 architecture identity."""

    spec = ModelInputSpec.from_value(input_spec)
    model_id = str(getattr(model, "model_id", ""))
    base: dict[str, Any] = {
        "model_id": model_id,
        "representation_mode": spec.mode.value,
        "n_classes": int(spec.n_classes),
    }

    if model_id == "compact_cnn":
        base.update(_compact_architecture(model))
    elif model_id in {"inception_full", "inception_small", "inception_matrix"}:
        base.update(_inception_architecture(model))
    elif model_id in {
        "inception_full_five_member_ensemble",
        "inception_matrix_five_member_ensemble",
    }:
        first = model.members[0]
        base.update({f"member_{key}": value for key, value in _inception_architecture(first).items()})
        base.update(
            {
                "member_count": len(model.members),
                "member_seeds": tuple(int(value) for value in model.member_seeds),
                "probability_aggregation": "arithmetic_mean",
            }
        )
    elif model_id in {"logistic_regression", "rbf_svm", "extra_trees"}:
        base.update(
            {
                "class_weight": model.class_weight,
            }
        )
        if model_id == "logistic_regression":
            base.update(
                {
                    "preprocessing": (
                        "median_imputer_keep_empty_features",
                        "standard_scaler",
                    ),
                    "estimator": "sklearn.linear_model.LogisticRegression",
                    "max_iter": int(model.logistic_max_iter),
                    "solver": str(model.logistic_solver),
                }
            )
        elif model_id == "rbf_svm":
            base.update(
                {
                    "preprocessing": (
                        "median_imputer_keep_empty_features",
                        "standard_scaler",
                    ),
                    "estimator": "sklearn.svm.SVC",
                    "kernel": str(model.svm_kernel),
                    "probability": bool(model.svm_probability),
                    "C": float(model.svm_c),
                    "gamma": model.svm_gamma,
                }
            )
        else:
            base.update(
                {
                    "preprocessing": ("median_imputer_keep_empty_features",),
                    "estimator": "sklearn.ensemble.ExtraTreesClassifier",
                    "n_estimators": int(model.extra_trees_n_estimators),
                    "n_jobs": int(model.extra_trees_n_jobs),
                }
            )
    elif model_id in {"rocket_numpy", "minirocket_ablation"}:
        base.update(
            {
                "implementation_algorithm_id": str(model.implementation_algorithm_id),
                "scientific_status": str(
                    getattr(model, "scientific_status", "reference_numpy_rocket")
                ),
                "robust_scaler": "outer_train_channel_median_iqr",
                "transformer": "numpy_random_convolution_max_ppv",
                "candidate_kernel_lengths": (7, 9, 11),
                "n_kernels": int(model.n_kernels),
                "features_per_kernel": 2,
                "classifier": "sklearn.linear_model.RidgeClassifier",
                "ridge_alpha": float(model.alpha),
            }
        )
    elif model_id in {
        "shapeformer_channel_specific_osd",
        "shapeformer_channel_specific_scalar_distance_ablation",
        "shapeformer_effect_size_fixed_v1",
    }:
        reference = model_id == "shapeformer_channel_specific_osd"
        channel_specific = model_id != "shapeformer_effect_size_fixed_v1"
        common = {
            "discovery_method": str(model.discovery_method),
            "shapelet_count": int(model.shapelet_count),
            "shapelet_count_per_class": int(model.shapelets_per_class),
            "shapelet_channel_policy": (
                "single_source_channel" if channel_specific else "joint_all_channels"
            ),
            "shapelet_length_policy": (
                "variable_insertion_stage_three_consecutive_pips"
                if channel_specific
                else "fixed_samples"
            ),
            "shapelet_length_samples": (
                None if channel_specific else int(model.shapelet_length_samples)
            ),
            "candidate_stride_samples": (
                None if channel_specific else int(model.discovery_stride_samples)
            ),
            "best_fit_search_stride_samples": 1,
            "input_fs_hz": float(model.input_fs_hz),
            "distance_position_chunk_size": int(
                model.distance_position_chunk_size
            ),
        }
        if reference:
            common.update(
                {
                    "implementation_status": str(model.implementation_status),
                    "sequence_length_samples": int(model.sequence_length),
                    "position_search_neighbourhood_samples": int(
                        model.position_search_neighbourhood_samples
                    ),
                    "num_pip_ratio": float(model.num_pip_ratio),
                    "pip_rounding_rule": str(model.pip_rounding_rule),
                    "pip_selection_rule": str(model.pip_selection_rule),
                    "candidate_generation_rule": str(
                        model.candidate_generation_rule
                    ),
                    "candidate_enumeration_rule": str(
                        model.candidate_enumeration_rule
                    ),
                    "candidate_ranking_rule": str(
                        model.candidate_ranking_rule
                    ),
                    "selected_bank_order_rule": str(
                        model.selected_bank_order_rule
                    ),
                    "discovery_position_search_boundary_rule": str(
                        model.discovery_position_search_boundary_rule
                    ),
                    "information_gain_split_rule": str(
                        model.information_gain_split_rule
                    ),
                    "max_discovery_windows": int(model.max_discovery_windows),
                    "discovery_balance": str(model.discovery_balance),
                    "local_kernel_width_samples": int(
                        model.local_kernel_width_samples
                    ),
                    "local_embedding_channels": int(
                        model.local_embedding_channels
                    ),
                    "shape_embedding_channels": int(
                        model.shape_embedding_channels
                    ),
                    "attention_heads": int(model.attention_heads),
                    "attention_query_chunk_size": int(
                        model.attention_query_chunk_size
                    ),
                    "attention_feedforward_channels": int(
                        model.attention_feedforward_channels
                    ),
                    "complexity_norm": float(model.complexity_norm),
                    "max_complexity_ratio": float(model.max_complexity_ratio),
                    "dropout": float(model.dropout_probability),
                    "shapelet_token_formula": (
                        "selected_projection(raw_best_segment)-"
                        "shapelet_projection(shapelet)"
                    ),
                    "shapelet_position_embeddings": (
                        "source_channel",
                        "source_start",
                        "source_end",
                    ),
                    "shapelet_weighting": (
                        "learnable_initialised_from_information_gain"
                    ),
                    "shape_position_embedding_width_policy": str(
                        model.shape_position_embedding_width_policy
                    ),
                    "shape_channel_position_width": int(
                        model.shape_channel_position_width
                    ),
                    "shape_start_position_width": int(
                        model.shape_start_position_width
                    ),
                    "shape_end_position_width": int(
                        model.shape_end_position_width
                    ),
                    "attention_probability_dropout_applied": bool(
                        model.attention_probability_dropout_applied
                    ),
                    "generic_branch_input": "full_multivariate_input",
                    "generic_branch_channel_count": int(model.n_channels),
                    "global_shape_pooling": "first_contextualised_shape_token",
                }
            )
        elif not channel_specific:
            common.update(
                {
                    "num_pip_ratio": None,
                    "max_discovery_windows": None,
                    "discovery_balance": None,
                    "hidden_channels": int(model.hidden_channels),
                    "patch_size_samples": int(model.patch_size_samples),
                    "patch_stride_samples": int(model.patch_size_samples),
                    "patch_bias": False,
                    "attention_heads": int(model.attention_heads),
                    "attention_layers": int(model.attention_layers),
                    "attention_feedforward_channels": int(
                        model.attention_feedforward_channels
                    ),
                    "attention_activation": "gelu",
                    "attention_norm_first": False,
                    "classifier_dropout": float(model.classifier_dropout),
                }
            )
        else:
            common.update(
                {
                    "downstream_status": (
                        "scalar_distance_ablation_not_literature_shapeformer"
                    ),
                    "num_pip_ratio": float(model.num_pip_ratio),
                    "max_discovery_windows": int(model.max_discovery_windows),
                    "discovery_balance": str(model.discovery_balance),
                    "position_search_neighbourhood_samples": int(
                        model.shapelet_candidate_records[0][
                            "position_search_neighbourhood_samples"
                        ]
                    ),
                    "information_gain_split_rule": str(
                        model.information_gain_split_rule
                    ),
                    "hidden_channels": int(model.hidden_channels),
                    "patch_size_samples": int(model.patch_size_samples),
                    "patch_stride_samples": int(model.patch_size_samples),
                    "patch_bias": False,
                    "attention_heads": int(model.attention_heads),
                    "attention_layers": int(model.attention_layers),
                    "attention_feedforward_channels": int(
                        model.attention_feedforward_channels
                    ),
                    "attention_activation": "gelu",
                    "attention_norm_first": False,
                    "classifier_dropout": float(model.classifier_dropout),
                }
            )
        base.update(common)
    elif model_id in {"fusion_compact", "fusion_inception"}:
        base.update(
            {
                "signal_feature_dim": int(model.signal_feature_dim),
                "feature_hidden_dim": int(model.feature_hidden_dim),
                "fusion_hidden_dim": int(model.fusion_hidden_dim),
                "pooling": str(model.pooling),
                "fusion_dropout": float(model.fusion_dropout),
            }
        )
        signal_parameters = (
            _compact_architecture(model.signal_encoder)
            if model_id == "fusion_compact"
            else _inception_architecture(model.signal_encoder)
        )
        base["signal_encoder"] = signal_parameters
    else:
        raise ValueError(f"cannot resolve architecture for unsupported model_id: {model_id}")
    return _normalise_architecture_value(base)


def resolved_architecture_hash(
    model: Any, input_spec: ModelInputSpec | Mapping[str, Any]
) -> str:
    """Hash the exact resolved architecture using canonical JSON."""

    payload = json.dumps(
        resolved_architecture_parameters(model, input_spec),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def materialize_architecture_parameters(
    model_config: Mapping[str, Any],
    input_spec: ModelInputSpec | Mapping[str, Any],
) -> dict[str, Any]:
    """Materialise the complete declaration that a formal config must persist.

    This helper is intentionally not called implicitly by :func:`create_model`.
    Formal YAML/run locks must store its complete result so review can see every
    constructor constant before execution.
    """

    spec = ModelInputSpec.from_value(input_spec)
    config = normalize_model_config(model_config)
    model_id = str(config.pop("model_id"))
    config.pop("canonical_model_name")
    config.pop("architecture_parameters", None)
    config.pop("seed", None)
    base: dict[str, Any] = {
        "model_id": model_id,
        "representation_mode": spec.mode.value,
        "n_classes": int(spec.n_classes),
    }

    if model_id == "compact_cnn":
        base.update(
            {
                "stage_channels": (32, 64, 128),
                "kernel_sizes": tuple(config["kernel_sizes"]),
                "dilations": tuple(config["dilations"]),
                "pool_sizes": tuple(config["pool_sizes"]),
                "stage_dropouts": (0.10, 0.15),
                "global_pooling": "adaptive_average_1",
                "classifier_dropout": float(config["dropout"]),
            }
        )
    elif model_id in {"inception_full", "inception_small", "inception_matrix"}:
        variant = (
            str(config["variant"])
            if model_id == "inception_matrix"
            else model_id.removeprefix("inception_")
        )
        capacity = {
            "full": (32, 32, 6),
            "small": (16, 16, 3),
        }[variant]
        base.update(
            {
                "variant": variant,
                "out_channels": capacity[0],
                "bottleneck_channels": capacity[1],
                "depth": capacity[2],
                "kernel_sizes": tuple(config["kernel_sizes"]),
                "dilation": int(config["dilation"]),
                "branch_count": 4,
                "residual_interval": 3,
                "global_pooling": "mask_aware_global_average",
                "classifier_dropout": float(config["dropout"]),
            }
        )
    elif model_id in {
        "inception_full_five_member_ensemble",
        "inception_matrix_five_member_ensemble",
    }:
        member = {
            "variant": "full",
            "out_channels": 32,
            "bottleneck_channels": 32,
            "depth": 6,
            "kernel_sizes": tuple(config["kernel_sizes"]),
            "dilation": int(config["dilation"]),
            "branch_count": 4,
            "residual_interval": 3,
            "global_pooling": "mask_aware_global_average",
            "classifier_dropout": float(config["dropout"]),
        }
        base.update({f"member_{key}": value for key, value in member.items()})
        base.update(
            {
                "member_count": 5,
                "member_seeds": tuple(int(value) for value in config["member_seeds"]),
                "probability_aggregation": "arithmetic_mean",
            }
        )
    elif model_id in {"logistic_regression", "rbf_svm", "extra_trees"}:
        base.update(
            {
                "class_weight": config["class_weight"],
            }
        )
        if model_id == "logistic_regression":
            base.update(
                {
                    "preprocessing": (
                        "median_imputer_keep_empty_features",
                        "standard_scaler",
                    ),
                    "estimator": "sklearn.linear_model.LogisticRegression",
                    "max_iter": int(config["logistic_max_iter"]),
                    "solver": str(config["logistic_solver"]),
                }
            )
        elif model_id == "rbf_svm":
            base.update(
                {
                    "preprocessing": (
                        "median_imputer_keep_empty_features",
                        "standard_scaler",
                    ),
                    "estimator": "sklearn.svm.SVC",
                    "kernel": str(config["svm_kernel"]),
                    "probability": bool(config["svm_probability"]),
                    "C": float(config["svm_c"]),
                    "gamma": config["svm_gamma"],
                }
            )
        else:
            base.update(
                {
                    "preprocessing": ("median_imputer_keep_empty_features",),
                    "estimator": "sklearn.ensemble.ExtraTreesClassifier",
                    "n_estimators": int(config["extra_trees_n_estimators"]),
                    "n_jobs": int(config["extra_trees_n_jobs"]),
                }
            )
    elif model_id in {"rocket_numpy", "minirocket_ablation"}:
        reference = model_id == "rocket_numpy"
        base.update(
            {
                "implementation_algorithm_id": (
                    "rocket_ridge_numpy"
                    if reference
                    else "minirocket_engineering_ablation"
                ),
                "scientific_status": (
                    "reference_numpy_rocket"
                    if reference
                    else "ablation_not_reference_implementation"
                ),
                "robust_scaler": "outer_train_channel_median_iqr",
                "transformer": "numpy_random_convolution_max_ppv",
                "candidate_kernel_lengths": (7, 9, 11),
                "n_kernels": int(config["n_kernels"]),
                "features_per_kernel": 2,
                "classifier": "sklearn.linear_model.RidgeClassifier",
                "ridge_alpha": float(config["alpha"]),
            }
        )
    elif model_id in {
        "shapeformer_channel_specific_osd",
        "shapeformer_channel_specific_scalar_distance_ablation",
        "shapeformer_effect_size_fixed_v1",
    }:
        reference = model_id == "shapeformer_channel_specific_osd"
        channel_specific = model_id != "shapeformer_effect_size_fixed_v1"
        bank = config.get("shapelets")
        if "shapelets_per_class" in config:
            per_class = int(config["shapelets_per_class"])
        elif bank is not None:
            classes = np.asarray(
                bank["source_classes"]
                if isinstance(bank, Mapping)
                else bank.source_classes
            )
            counts = np.unique(classes, return_counts=True)[1]
            if counts.size < 2 or not np.all(counts == counts[0]):
                raise ValueError("shapelet bank must have equal per-class capacity")
            per_class = int(counts[0])
        else:
            raise ValueError("ShapeFormer architecture requires shapelets_per_class or a bank")
        if bank is not None:
            count = (
                len(bank["values"])
                if isinstance(bank, Mapping)
                else len(bank.values)
            )
        else:
            count = int(spec.n_classes) * per_class
        def bank_or_config(name: str) -> Any:
            if name in config:
                return config[name]
            if bank is None:
                raise ValueError(f"ShapeFormer architecture is missing {name}")
            return bank[name] if isinstance(bank, Mapping) else getattr(bank, name)

        if reference:
            base.update(
                {
                    "discovery_method": str(config["discovery_method"]),
                    "shapelet_count": int(count),
                    "shapelet_count_per_class": per_class,
                    "shapelet_channel_policy": "single_source_channel",
                    "shapelet_length_policy": (
                        "variable_insertion_stage_three_consecutive_pips"
                    ),
                    "shapelet_length_samples": None,
                    "candidate_stride_samples": None,
                    "best_fit_search_stride_samples": 1,
                    "num_pip_ratio": float(bank_or_config("num_pip_ratio")),
                    "pip_rounding_rule": str(
                        bank_or_config("pip_rounding_rule")
                    ),
                    "pip_selection_rule": str(
                        bank_or_config("pip_selection_rule")
                    ),
                    "candidate_generation_rule": str(
                        bank_or_config("candidate_generation_rule")
                    ),
                    "candidate_enumeration_rule": str(
                        bank_or_config("candidate_enumeration_rule")
                    ),
                    "candidate_ranking_rule": str(
                        bank_or_config("candidate_ranking_rule")
                    ),
                    "selected_bank_order_rule": str(
                        bank_or_config("selected_bank_order_rule")
                    ),
                    "discovery_position_search_boundary_rule": str(
                        bank_or_config(
                            "discovery_position_search_boundary_rule"
                        )
                    ),
                    "information_gain_split_rule": str(
                        bank_or_config("information_gain_split_rule")
                    ),
                    "max_discovery_windows": int(
                        bank_or_config("max_discovery_windows")
                    ),
                    "discovery_balance": str(
                        bank_or_config("discovery_balance")
                    ),
                    "input_fs_hz": float(config["input_fs_hz"]),
                    "implementation_status": (
                        "implemented_not_benchmarked_high_compute"
                    ),
                    "sequence_length_samples": int(
                        config["sequence_length_samples"]
                    ),
                    "position_search_neighbourhood_samples": int(
                        bank_or_config(
                            "position_search_neighbourhood_samples"
                        )
                    ),
                    "local_kernel_width_samples": int(
                        config["local_kernel_width_samples"]
                    ),
                    "local_embedding_channels": int(
                        config["local_embedding_channels"]
                    ),
                    "shape_embedding_channels": int(
                        config["shape_embedding_channels"]
                    ),
                    "attention_heads": int(config["attention_heads"]),
                    "attention_query_chunk_size": int(
                        config["attention_query_chunk_size"]
                    ),
                    "attention_feedforward_channels": int(
                        config["attention_feedforward_channels"]
                    ),
                    "distance_position_chunk_size": int(
                        config["distance_position_chunk_size"]
                    ),
                    "complexity_norm": float(config["complexity_norm"]),
                    "max_complexity_ratio": float(
                        config["max_complexity_ratio"]
                    ),
                    "dropout": float(config["dropout"]),
                    "shapelet_token_formula": (
                        "selected_projection(raw_best_segment)-"
                        "shapelet_projection(shapelet)"
                    ),
                    "shapelet_position_embeddings": (
                        "source_channel",
                        "source_start",
                        "source_end",
                    ),
                    "shapelet_weighting": (
                        "learnable_initialised_from_information_gain"
                    ),
                    "shape_position_embedding_width_policy": (
                        "upstream_observed_max_plus_1_per_axis"
                    ),
                    "attention_probability_dropout_applied": False,
                    "generic_branch_input": "full_multivariate_input",
                    "generic_branch_channel_count": int(spec.n_channels),
                    "global_shape_pooling": (
                        "first_contextualised_shape_token"
                    ),
                }
            )
            if bank is not None:
                def bank_array(name: str) -> np.ndarray:
                    value = bank[name] if isinstance(bank, Mapping) else getattr(bank, name)
                    return np.asarray(value, dtype=np.int64)

                base.update(
                    {
                        "shape_channel_position_width": int(
                            bank_array("source_channels").max()
                        )
                        + 1,
                        "shape_start_position_width": int(
                            bank_array("source_starts").max()
                        )
                        + 1,
                        "shape_end_position_width": int(
                            bank_array("source_ends").max()
                        )
                        + 1,
                    }
                )
            return _normalise_architecture_value(base)

        if channel_specific:
            base.update(
                {
                    "discovery_method": str(config["discovery_method"]),
                    "shapelet_count": int(count),
                    "shapelet_count_per_class": per_class,
                    "shapelet_channel_policy": "single_source_channel",
                    "shapelet_length_policy": (
                        "variable_insertion_stage_three_consecutive_pips"
                    ),
                    "shapelet_length_samples": None,
                    "candidate_stride_samples": None,
                    "best_fit_search_stride_samples": 1,
                    "num_pip_ratio": float(bank_or_config("num_pip_ratio")),
                    "information_gain_split_rule": str(
                        bank_or_config("information_gain_split_rule")
                    ),
                    "max_discovery_windows": int(
                        bank_or_config("max_discovery_windows")
                    ),
                    "discovery_balance": str(
                        bank_or_config("discovery_balance")
                    ),
                    "position_search_neighbourhood_samples": int(
                        bank_or_config(
                            "position_search_neighbourhood_samples"
                        )
                    ),
                    "input_fs_hz": float(config["input_fs_hz"]),
                    "downstream_status": (
                        "scalar_distance_ablation_not_literature_shapeformer"
                    ),
                    "hidden_channels": int(config["hidden_channels"]),
                    "patch_size_samples": int(config["patch_size_samples"]),
                    "patch_stride_samples": int(config["patch_size_samples"]),
                    "patch_bias": False,
                    "attention_heads": int(config["attention_heads"]),
                    "attention_layers": int(config["attention_layers"]),
                    "attention_feedforward_channels": (
                        int(config["hidden_channels"]) * 2
                    ),
                    "attention_activation": "gelu",
                    "attention_norm_first": False,
                    "distance_position_chunk_size": int(
                        config["distance_position_chunk_size"]
                    ),
                    "classifier_dropout": float(config["dropout"]),
                }
            )
            return _normalise_architecture_value(base)

        base.update(
            {
                "discovery_method": str(config["discovery_method"]),
                "shapelet_count": int(count),
                "shapelet_count_per_class": per_class,
                "shapelet_channel_policy": (
                    "single_source_channel" if reference else "joint_all_channels"
                ),
                "shapelet_length_policy": (
                    "variable_three_consecutive_pips" if reference else "fixed_samples"
                ),
                "shapelet_length_samples": (
                    None
                    if reference
                    else int(bank_or_config("shapelet_length_samples"))
                ),
                "candidate_stride_samples": (
                    None
                    if reference
                    else int(bank_or_config("discovery_stride_samples"))
                ),
                "best_fit_search_stride_samples": 1,
                "num_pip_ratio": (
                    float(bank_or_config("num_pip_ratio")) if reference else None
                ),
                "max_discovery_windows": (
                    int(bank_or_config("max_discovery_windows")) if reference else None
                ),
                "discovery_balance": (
                    str(bank_or_config("discovery_balance")) if reference else None
                ),
                "input_fs_hz": float(config["input_fs_hz"]),
                "hidden_channels": int(config["hidden_channels"]),
                "patch_size_samples": int(config["patch_size_samples"]),
                "patch_stride_samples": int(config["patch_size_samples"]),
                "patch_bias": False,
                "attention_heads": int(config["attention_heads"]),
                "attention_layers": int(config["attention_layers"]),
                "attention_feedforward_channels": int(config["hidden_channels"]) * 2,
                "attention_activation": "gelu",
                "attention_norm_first": False,
                "distance_position_chunk_size": int(
                    config["distance_position_chunk_size"]
                ),
                "classifier_dropout": float(config["dropout"]),
            }
        )
    elif model_id in {"fusion_compact", "fusion_inception"}:
        base.update(
            {
                "signal_feature_dim": 128,
                "feature_hidden_dim": int(config["feature_hidden_dim"]),
                "fusion_hidden_dim": int(config["fusion_hidden_dim"]),
                "pooling": str(config["pooling"]),
                "fusion_dropout": float(config["dropout"]),
            }
        )
        if model_id == "fusion_compact":
            signal = {
                "stage_channels": (32, 64, 128),
                "kernel_sizes": tuple(config["signal_kernel_sizes"]),
                "dilations": tuple(config["signal_dilations"]),
                "pool_sizes": tuple(config["signal_pool_sizes"]),
                "stage_dropouts": (0.10, 0.15),
                "global_pooling": "adaptive_average_1",
                "classifier_dropout": float(config["signal_dropout"]),
            }
        else:
            variant = str(config["signal_variant"])
            capacity = {"full": (32, 32, 6), "small": (16, 16, 3)}[variant]
            base["signal_feature_dim"] = capacity[0] * 4
            signal = {
                "variant": variant,
                "out_channels": capacity[0],
                "bottleneck_channels": capacity[1],
                "depth": capacity[2],
                "kernel_sizes": tuple(config["signal_kernel_sizes"]),
                "dilation": int(config["signal_dilation"]),
                "branch_count": 4,
                "residual_interval": 3,
                "global_pooling": "mask_aware_global_average",
                "classifier_dropout": float(config["signal_dropout"]),
            }
        base["signal_encoder"] = signal
    else:
        raise ValueError(f"cannot materialise architecture for {model_id}")
    return _normalise_architecture_value(base)


def validate_resolved_architecture(
    model: Any,
    declared: Mapping[str, Any],
    input_spec: ModelInputSpec | Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed when a formal config differs from the constructed object."""

    if not isinstance(declared, Mapping) or not declared:
        raise ValueError("architecture_parameters must be an explicit non-empty mapping")
    actual = resolved_architecture_parameters(model, input_spec)
    expected = _normalise_architecture_value(declared)
    if expected != actual:
        keys = sorted(set(expected) | set(actual))
        differing = [key for key in keys if expected.get(key) != actual.get(key)]
        raise ValueError(
            "declared architecture_parameters differ from the constructed model; "
            f"differing keys={differing}"
        )
    return actual


def _finalize_model(
    model: Any,
    declared_architecture: Mapping[str, Any],
    input_spec: ModelInputSpec,
) -> Any:
    actual = validate_resolved_architecture(model, declared_architecture, input_spec)
    model.resolved_architecture_parameters = actual
    model.resolved_architecture_hash = resolved_architecture_hash(model, input_spec)
    return model


def _torch_seed(seed: int) -> None:
    """Seed PyTorch only when a deep model is requested / 仅在请求深度模型时设种子。"""

    import torch

    torch.manual_seed(int(seed))


@dataclass(frozen=True)
class PreparedModelFactory:
    """Fold-bound callable factory plus archive-safe provenance.

    绑定 outer-train fold 的可调用模型工厂，并携带可归档 provenance。ShapeFormer
    discovery 已在构造本对象前完成；重复调用只会从相同冻结配置创建新模型。
    """

    resolved_model_config: Mapping[str, Any]
    input_spec: ModelInputSpec
    provenance: Mapping[str, Any]

    def __call__(self) -> Any:
        """Construct a fresh model / 从冻结配置构造一个全新的模型。"""

        return create_model(copy.deepcopy(dict(self.resolved_model_config)), self.input_spec)

    @property
    def factory(self) -> Callable[[], Any]:
        """Expose the stable public factory handle / 公开稳定的 factory 句柄。"""

        return self


def _shapelet_bank_hash(bank: Any) -> str:
    """Hash learned bank arrays and metadata / 散列已学习的 bank 数组和元数据。"""

    digest = hashlib.sha256(type(bank).__name__.encode("utf-8"))

    def update(value: Any) -> None:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(b"ndarray\0")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        elif isinstance(value, (tuple, list)):
            digest.update(f"sequence:{len(value)}\0".encode("ascii"))
            for item in value:
                update(item)
        elif isinstance(value, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(value):
                digest.update(str(key).encode("utf-8"))
                update(value[key])
        else:
            digest.update(repr(value).encode("utf-8"))

    for name, value in sorted(vars(bank).items()):
        digest.update(name.encode("utf-8"))
        update(value)
    return digest.hexdigest()


def prepare_model_factory(
    model_config: Mapping[str, Any],
    input_spec: ModelInputSpec | Mapping[str, Any],
    outer_train_dataset: Any,
    frozen_split: Any,
) -> PreparedModelFactory:
    """Prepare a reproducible factory using outer-train data only.

    仅使用冻结 outer-train 数据准备可重复构造的模型工厂。PISDPort 与 EffectSize
    discovery 由 model_id 严格二选一；方法或依赖不匹配时关闭失败，不回退。
    """

    from ..training.datasets import RawWindowDataset
    from ..training.trainer import dataset_binding_hash

    spec = ModelInputSpec.from_value(input_spec)
    frozen_split.assert_training_dataset(outer_train_dataset, exact=True)
    config = normalize_model_config(model_config)
    model_id = str(config["model_id"])
    shapeformer_ids = {
        "shapeformer_channel_specific_osd",
        "shapeformer_channel_specific_scalar_distance_ablation",
        "shapeformer_effect_size_fixed_v1",
    }
    if model_id not in shapeformer_ids:
        return PreparedModelFactory(
            resolved_model_config=copy.deepcopy(config),
            input_spec=spec,
            provenance={
                "model_id": model_id,
                "fold_local_preparation": "not_required",
                "outer_repeat_index": int(frozen_split.repeat),
                "outer_fold_index": int(frozen_split.fold),
                "outer_train_dataset_hash": dataset_binding_hash(outer_train_dataset),
            },
        )

    if spec.mode is not RepresentationMode.RAW or not isinstance(
        outer_train_dataset, RawWindowDataset
    ):
        raise TypeError("ShapeFormer discovery requires a materialised RawWindowDataset")
    sample_mask = np.asarray(outer_train_dataset.sample_mask, dtype=bool)
    sequence_lengths = sample_mask.sum(axis=1).astype(np.int64)
    expected_prefix_mask = np.arange(sample_mask.shape[1])[None, :] < sequence_lengths[:, None]
    if not np.array_equal(sample_mask, expected_prefix_mask):
        raise ValueError("ShapeFormer discovery requires a contiguous valid prefix per window")
    forbidden = {
        "shapelets",
        "outer_repeat_index",
        "outer_fold_index",
        "outer_train_participant_hash",
    } & set(config)
    if forbidden:
        raise ValueError(
            "fold-local ShapeFormer preparation derives these fields and rejects supplied "
            f"values: {sorted(forbidden)}"
        )

    expected_method = (
        "channel_specific_osd"
        if model_id != "shapeformer_effect_size_fixed_v1"
        else "effect_size_fixed_v1"
    )
    discovery_method = str(config.pop("discovery_method", ""))
    if discovery_method != expected_method:
        raise ValueError(
            f"{model_id} requires explicit discovery_method={expected_method}; "
            "no discovery fallback is allowed"
        )
    if "input_fs_hz" not in config:
        raise ValueError("ShapeFormer preparation requires explicit input_fs_hz")
    input_fs_hz = float(config.pop("input_fs_hz"))
    if "seed" not in config:
        raise ValueError("ShapeFormer fold-local discovery requires explicit seed")
    seed = int(config["seed"])
    participants = tuple(
        str(identity.participant_id) for identity in outer_train_dataset.identities
    )
    files = tuple(str(identity.file_id) for identity in outer_train_dataset.identities)
    windows = tuple(str(identity.window_id) for identity in outer_train_dataset.identities)
    base_discovery = {
        "x_train": np.asarray(outer_train_dataset.values, dtype=np.float32),
        "y_train": np.asarray(outer_train_dataset.labels, dtype=np.int64),
        "participant_ids": participants,
        "discovery_method": discovery_method,
        "input_fs_hz": input_fs_hz,
        "outer_repeat_index": int(frozen_split.repeat),
        "outer_fold_index": int(frozen_split.fold),
        "seed": seed,
    }
    if model_id != "shapeformer_effect_size_fixed_v1":
        from .pisd_port import (
            CANDIDATE_ENUMERATION_RULE,
            CANDIDATE_GENERATION_RULE,
            CANDIDATE_RANKING_RULE,
            DISCOVERY_POSITION_SEARCH_BOUNDARY_RULE,
            INFORMATION_GAIN_SPLIT_RULE,
            PIP_ROUNDING_RULE,
            PIP_SELECTION_RULE,
            SELECTED_BANK_ORDER_RULE,
            discover_pisd_shapelets,
        )

        forbidden_reference = {
            "shapelet_length_samples",
            "shapelet_length",
            "candidate_stride",
            "discovery_stride",
            "discovery_stride_samples",
            "num_pips",
            "max_candidates_per_class",
        } & set(config)
        if forbidden_reference:
            raise ValueError(
                "channel_specific_osd has no fixed length/stride/count fallback controls: "
                f"{sorted(forbidden_reference)}"
            )
        if not spec.channel_schema or len(spec.channel_schema) != spec.n_channels:
            raise ValueError("channel_specific_osd requires the exact ordered channel_schema")
        if str(config.pop("pip_rounding_rule")) != PIP_ROUNDING_RULE:
            raise ValueError("channel_specific_osd pip_rounding_rule drifted")
        if str(config.pop("pip_selection_rule")) != PIP_SELECTION_RULE:
            raise ValueError("channel_specific_osd pip_selection_rule drifted")
        if str(config.pop("candidate_generation_rule")) != CANDIDATE_GENERATION_RULE:
            raise ValueError("channel_specific_osd candidate_generation_rule drifted")
        if (
            str(config.pop("candidate_enumeration_rule"))
            != CANDIDATE_ENUMERATION_RULE
        ):
            raise ValueError("channel_specific_osd candidate_enumeration_rule drifted")
        if str(config.pop("candidate_ranking_rule")) != CANDIDATE_RANKING_RULE:
            raise ValueError("channel_specific_osd candidate_ranking_rule drifted")
        if str(config.pop("selected_bank_order_rule")) != SELECTED_BANK_ORDER_RULE:
            raise ValueError("channel_specific_osd selected_bank_order_rule drifted")
        if (
            str(config.pop("discovery_position_search_boundary_rule"))
            != DISCOVERY_POSITION_SEARCH_BOUNDARY_RULE
        ):
            raise ValueError(
                "channel_specific_osd discovery_position_search_boundary_rule drifted"
            )
        if (
            str(config.pop("information_gain_split_rule"))
            != INFORMATION_GAIN_SPLIT_RULE
        ):
            raise ValueError("channel_specific_osd information_gain_split_rule drifted")
        bank = discover_pisd_shapelets(
            **base_discovery,
            file_ids=files,
            window_ids=windows,
            channel_schema=spec.channel_schema,
            sequence_lengths=sequence_lengths,
            num_pip_ratio=float(config.pop("num_pip_ratio")),
            shapelets_per_class=int(config.pop("shapelets_per_class")),
            max_discovery_windows=int(config.pop("max_discovery_windows")),
            discovery_balance=str(config.pop("discovery_balance")),
            position_search_neighbourhood_samples=int(
                config.pop("position_search_neighbourhood_samples")
            ),
            distance_position_chunk_size=int(config.get("distance_position_chunk_size", 256)),
        )
    else:
        from .shapeformer import discover_effect_size_shapelets

        bank = discover_effect_size_shapelets(
            **base_discovery,
            shapelet_length=int(config.pop("shapelet_length_samples")),
            shapelets_per_class=int(config.pop("shapelets_per_class")),
            stride=int(config.pop("discovery_stride_samples")),
            max_candidates_per_class=int(config.pop("max_candidates_per_class")),
        )

    resolved = copy.deepcopy(config)
    resolved.update(
        {
            "model_id": model_id,
            "shapelets": bank,
            "discovery_method": discovery_method,
            "input_fs_hz": input_fs_hz,
            "outer_repeat_index": int(frozen_split.repeat),
            "outer_fold_index": int(frozen_split.fold),
            "outer_train_participant_hash": bank.outer_train_participant_hash,
        }
    )
    if hasattr(bank, "position_search_neighbourhood_samples"):
        resolved["position_search_neighbourhood_samples"] = int(
            bank.position_search_neighbourhood_samples
        )
        resolved["information_gain_split_rule"] = str(
            bank.information_gain_split_rule
        )
        resolved["pip_rounding_rule"] = str(bank.pip_rounding_rule)
        resolved["pip_selection_rule"] = str(bank.pip_selection_rule)
        resolved["candidate_generation_rule"] = str(
            bank.candidate_generation_rule
        )
        resolved["candidate_enumeration_rule"] = str(
            bank.candidate_enumeration_rule
        )
        resolved["candidate_ranking_rule"] = str(bank.candidate_ranking_rule)
        resolved["selected_bank_order_rule"] = str(
            bank.selected_bank_order_rule
        )
        resolved["discovery_position_search_boundary_rule"] = str(
            bank.discovery_position_search_boundary_rule
        )
        resolved["architecture_parameters"] = materialize_architecture_parameters(
            resolved, spec
        )
    provenance = {
        "model_id": model_id,
        "canonical_model_name": str(config["canonical_model_name"]),
        "registry_role": "reference" if model_id == "shapeformer_channel_specific_osd" else "ablation",
        "discovery_method": discovery_method,
        "discovery_bank_hash": _shapelet_bank_hash(bank),
        "shapelet_count": int(bank.count if hasattr(bank, "count") else bank.values.shape[0]),
        "shapelet_length_samples": bank.shapelet_length_samples,
        "input_fs_hz": float(bank.input_fs_hz),
        "outer_repeat_index": int(bank.outer_repeat_index),
        "outer_fold_index": int(bank.outer_fold_index),
        "outer_train_participant_hash": bank.outer_train_participant_hash,
        "outer_train_dataset_hash": dataset_binding_hash(outer_train_dataset),
        "fallback_used": False,
    }
    if hasattr(bank, "candidate_records"):
        provenance.update(
            {
                "candidate_records": bank.candidate_records(),
                "num_pip_ratio": float(bank.num_pip_ratio),
                "pip_count_rounding": bank.pip_rounding_rule,
                "pip_selection_rule": bank.pip_selection_rule,
                "candidate_generation_rule": bank.candidate_generation_rule,
                "candidate_enumeration_rule": bank.candidate_enumeration_rule,
                "candidate_ranking_rule": bank.candidate_ranking_rule,
                "selected_bank_order_rule": bank.selected_bank_order_rule,
                "discovery_position_search_boundary_rule": (
                    bank.discovery_position_search_boundary_rule
                ),
                "information_gain_split_rule": bank.information_gain_split_rule,
                "candidate_stride": None,
                "best_fit_search_stride": 1,
                "position_search_neighbourhood_samples": int(
                    bank.position_search_neighbourhood_samples
                ),
                "discovery_window_count": int(bank.discovery_window_count),
                "discovery_selection_hash": bank.discovery_selection_hash,
                "discovery_balance": bank.discovery_balance,
            }
        )
    return PreparedModelFactory(
        resolved_model_config=resolved,
        input_spec=spec,
        provenance=provenance,
    )


def create_model(
    model_config: Mapping[str, Any], input_spec: ModelInputSpec | Mapping[str, Any]
) -> Any:
    """Create one unfitted model from strict configuration.

    从严格配置创建一个未拟合模型。``model_id`` 与 representation mode 必须匹配；
    任何未知键路线均显式失败，避免运行时悄悄退化为另一算法。
    """

    config = normalize_model_config(model_config)
    model_id = str(config.pop("model_id"))
    canonical_model_name = str(config.pop("canonical_model_name"))
    if "architecture_parameters" not in config:
        raise ValueError("formal V2 model_config requires architecture_parameters")
    declared_architecture = config.pop("architecture_parameters")
    ensemble_ids = {
        "inception_full_five_member_ensemble",
        "inception_matrix_five_member_ensemble",
    }
    if model_id in ensemble_ids:
        if "seed" in config:
            raise ValueError("ensemble has no single seed; use the exact member_seeds list")
        seed = 0
    else:
        if "seed" not in config:
            raise ValueError(f"{model_id} requires an explicit seed")
        seed = int(config.pop("seed"))
    spec = ModelInputSpec.from_value(input_spec)
    mode = spec.mode

    feature_ids = {"logistic_regression", "rbf_svm", "extra_trees"}
    if model_id in feature_ids:
        if mode is not RepresentationMode.FEATURE_VECTOR:
            raise ValueError("feature baselines require feature_vector representation")
        if "class_weight" not in config:
            raise ValueError(f"{model_id} requires explicit class_weight (null is allowed)")
        options: dict[str, Any] = {"class_weight": config.pop("class_weight")}
        required_by_model = {
            "logistic_regression": ("logistic_max_iter", "logistic_solver"),
            "rbf_svm": ("svm_kernel", "svm_probability", "svm_c", "svm_gamma"),
            "extra_trees": ("extra_trees_n_estimators", "extra_trees_n_jobs"),
        }
        missing_options = sorted(set(required_by_model[model_id]) - set(config))
        if missing_options:
            raise ValueError(f"{model_id} missing explicit options: {missing_options}")
        for name in required_by_model[model_id]:
            options[name] = config.pop(name)
        if config:
            raise ValueError(f"unknown feature baseline options: {sorted(config)}")
        model = FeatureVectorBaseline(model_id, spec.feature_names, seed=seed, **options)
        model.canonical_model_name = canonical_model_name
        return _finalize_model(model, declared_architecture, spec)

    if model_id in {"rocket_numpy", "minirocket_ablation"}:
        if mode is not RepresentationMode.FEATURE_MATRIX:
            raise ValueError("canonical ROCKET and MiniROCKET require feature_matrix representation")
        if spec.n_channels <= 0:
            raise ValueError("feature_matrix channel count D must be resolved before model creation")
        missing_options = sorted({"n_kernels", "alpha"} - set(config))
        if missing_options:
            raise ValueError(f"{model_id} missing explicit options: {missing_options}")
        n_kernels = int(config.pop("n_kernels"))
        alpha = float(config.pop("alpha"))
        if config:
            raise ValueError(f"unknown ROCKET options: {sorted(config)}")
        constructor = RocketRidgeClassifier if model_id == "rocket_numpy" else MiniRocketAblation
        model = constructor(n_kernels=n_kernels, alpha=alpha, seed=seed)
        model.implementation_algorithm_id = str(getattr(model, "model_id", type(model).__name__))
        model.model_id = model_id
        model.canonical_model_name = canonical_model_name
        return _finalize_model(model, declared_architecture, spec)

    if model_id == "compact_cnn":
        if mode is not RepresentationMode.RAW:
            raise ValueError("CompactCNN1D requires raw representation")
        if spec.n_channels <= 0:
            raise ValueError("raw input requires a positive n_channels")
        from .compact_cnn import CompactCNN1D

        _torch_seed(seed)
        required = {"dropout", "kernel_sizes", "dilations", "pool_sizes"}
        missing_options = sorted(required - set(config))
        if missing_options:
            raise ValueError(f"compact_cnn missing explicit options: {missing_options}")
        dropout = float(config.pop("dropout"))
        kernel_sizes = tuple(int(value) for value in config.pop("kernel_sizes"))
        dilations = tuple(int(value) for value in config.pop("dilations"))
        pool_sizes = tuple(int(value) for value in config.pop("pool_sizes"))
        if config:
            raise ValueError(f"unknown CompactCNN options: {sorted(config)}")
        model = CompactCNN1D(
            spec.n_channels,
            spec.n_classes,
            dropout=dropout,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            pool_sizes=pool_sizes,
        )
        model.model_id = model_id
        model.canonical_model_name = canonical_model_name
        return _finalize_model(model, declared_architecture, spec)

    if model_id in {"inception_full", "inception_small", "inception_matrix"}:
        if model_id in {"inception_full", "inception_small"} and mode is not RepresentationMode.RAW:
            raise ValueError("InceptionTimeFull/Small require raw representation")
        if model_id == "inception_matrix" and mode is not RepresentationMode.FEATURE_MATRIX:
            raise ValueError("InceptionTimeMatrix requires feature_matrix representation")
        if spec.n_channels <= 0:
            raise ValueError("InceptionTime input channel count must be resolved and positive")
        from .inception import InceptionTimeSingleNetwork

        _torch_seed(seed)
        required = {"dropout", "kernel_sizes", "dilation"}
        if model_id == "inception_matrix":
            required.add("variant")
        missing_options = sorted(required - set(config))
        if missing_options:
            raise ValueError(f"{model_id} missing explicit options: {missing_options}")
        variant = (
            str(config.pop("variant"))
            if model_id == "inception_matrix"
            else model_id.removeprefix("inception_")
        )
        dropout = float(config.pop("dropout"))
        kernel_sizes = tuple(int(value) for value in config.pop("kernel_sizes"))
        dilation = int(config.pop("dilation"))
        if config:
            raise ValueError(f"unknown InceptionTime options: {sorted(config)}")
        model = InceptionTimeSingleNetwork(
            spec.n_channels,
            spec.n_classes,
            variant=variant,
            dropout=dropout,
            kernel_sizes=kernel_sizes,
            dilation=dilation,
        )
        model.model_id = model_id
        model.canonical_model_name = canonical_model_name
        return _finalize_model(model, declared_architecture, spec)

    if model_id in {
        "inception_full_five_member_ensemble",
        "inception_matrix_five_member_ensemble",
    }:
        if config.pop("comparison_only", False) is not True:
            raise ValueError(
                "five-member ensemble requires comparison_only=true and is never the default"
            )
        expected_mode = (
            RepresentationMode.RAW
            if model_id == "inception_full_five_member_ensemble"
            else RepresentationMode.FEATURE_MATRIX
        )
        if mode is not expected_mode:
            raise ValueError(f"{model_id} requires {expected_mode.value} representation")
        if spec.n_channels <= 0:
            raise ValueError("ensemble input channel count must be resolved and positive")
        from .inception import (
            CANONICAL_ENSEMBLE_MEMBER_SEEDS,
            InceptionTimeFiveMemberProbabilityEnsemble,
            InceptionTimeSingleNetwork,
        )

        missing_options = sorted(
            {"dropout", "member_seeds", "kernel_sizes", "dilation"} - set(config)
        )
        if missing_options:
            raise ValueError(f"ensemble missing explicit options: {missing_options}")
        variant = "full"
        dropout = float(config.pop("dropout"))
        kernel_sizes = tuple(int(value) for value in config.pop("kernel_sizes"))
        dilation = int(config.pop("dilation"))
        member_seeds = tuple(int(value) for value in config.pop("member_seeds"))
        if member_seeds != CANONICAL_ENSEMBLE_MEMBER_SEEDS:
            raise ValueError(
                "V2 ensemble member_seeds must equal "
                f"{list(CANONICAL_ENSEMBLE_MEMBER_SEEDS)}"
            )
        if config:
            raise ValueError(f"unknown ensemble options: {sorted(config)}")
        members = []
        for member_seed in member_seeds:
            _torch_seed(member_seed)
            members.append(
                InceptionTimeSingleNetwork(
                    spec.n_channels,
                    spec.n_classes,
                    variant=variant,
                    dropout=dropout,
                    kernel_sizes=kernel_sizes,
                    dilation=dilation,
                )
            )
        model = InceptionTimeFiveMemberProbabilityEnsemble(members, member_seeds)
        model.model_id = model_id
        model.canonical_model_name = canonical_model_name
        model.base_model_id = (
            "inception_full"
            if model_id == "inception_full_five_member_ensemble"
            else "inception_matrix"
        )
        model.representation_mode = expected_mode.value
        return _finalize_model(model, declared_architecture, spec)

    if model_id in {
        "shapeformer_channel_specific_osd",
        "shapeformer_channel_specific_scalar_distance_ablation",
        "shapeformer_effect_size_fixed_v1",
    }:
        if mode is not RepresentationMode.RAW:
            raise ValueError("V2 ShapeFormer candidates require raw representation")
        if spec.n_channels <= 0:
            raise ValueError("ShapeFormer input channel count must be resolved and positive")
        from .pisd_port import (
            CANDIDATE_ENUMERATION_RULE,
            CANDIDATE_GENERATION_RULE,
            CANDIDATE_RANKING_RULE,
            DISCOVERY_POSITION_SEARCH_BOUNDARY_RULE,
            INFORMATION_GAIN_SPLIT_RULE,
            PISD_DISCOVERY_METHOD,
            PIP_ROUNDING_RULE,
            PIP_SELECTION_RULE,
            SELECTED_BANK_ORDER_RULE,
            PisdShapelets,
        )
        from .shapeformer import EffectSizeShapelets, ExperimentalShapeFormer

        required = {
            "shapelets",
            "discovery_method",
            "input_fs_hz",
            "outer_repeat_index",
            "outer_fold_index",
            "outer_train_participant_hash",
        }
        if model_id != "shapeformer_effect_size_fixed_v1":
            required.update(
                {
                    "pip_rounding_rule",
                    "pip_selection_rule",
                    "candidate_generation_rule",
                    "candidate_enumeration_rule",
                    "candidate_ranking_rule",
                    "selected_bank_order_rule",
                    "discovery_position_search_boundary_rule",
                    "information_gain_split_rule",
                }
            )
        missing = sorted(required - set(config))
        if missing:
            raise ValueError(f"{model_id} missing required options: {missing}")
        expected_method = (
            PISD_DISCOVERY_METHOD
            if model_id != "shapeformer_effect_size_fixed_v1"
            else "effect_size_fixed_v1"
        )
        discovery_method = str(config.pop("discovery_method"))
        if discovery_method != expected_method:
            raise ValueError(
                f"{model_id} requires discovery_method={expected_method}; never fall back"
            )
        input_fs_hz = float(config.pop("input_fs_hz"))
        outer_repeat_index = int(config.pop("outer_repeat_index"))
        outer_fold_index = int(config.pop("outer_fold_index"))
        outer_train_participant_hash = str(config.pop("outer_train_participant_hash"))
        shapelets_value = config.pop("shapelets")
        if isinstance(shapelets_value, Mapping):
            payload = dict(shapelets_value)
            if model_id != "shapeformer_effect_size_fixed_v1":
                array_fields = {
                    "source_classes",
                    "information_gains",
                    "source_sample_indices",
                    "source_channels",
                    "source_starts",
                    "source_ends",
                    "source_start_seconds",
                    "source_end_seconds",
                    "candidate_lengths",
                    "discovery_sequence_lengths",
                    "pip_counts",
                }
                tuple_fields = {
                    "source_channel_names",
                    "source_participant_ids",
                    "source_file_ids",
                    "source_window_ids",
                    "fitted_participant_ids",
                    "discovery_participant_ids",
                    "discovery_file_ids",
                    "discovery_window_ids",
                    "channel_schema",
                }
                payload["values"] = tuple(
                    np.asarray(value, dtype=np.float32) for value in payload["values"]
                )
                for name in array_fields:
                    payload[name] = np.asarray(payload[name])
                for name in tuple_fields:
                    payload[name] = tuple(payload[name])
                shapelets_value = PisdShapelets(**payload)
            else:
                payload["values"] = np.asarray(payload["values"], dtype=np.float32)
                payload["source_classes"] = np.asarray(payload["source_classes"])
                payload["effect_sizes"] = np.asarray(payload["effect_sizes"])
                payload["fitted_participant_ids"] = tuple(payload["fitted_participant_ids"])
                shapelets_value = EffectSizeShapelets(**payload)
        expected_type = (
            PisdShapelets
            if model_id != "shapeformer_effect_size_fixed_v1"
            else EffectSizeShapelets
        )
        if not isinstance(shapelets_value, expected_type):
            raise TypeError(f"{model_id} received the wrong discovery-bank type")
        if shapelets_value.discovery_method != discovery_method:
            raise ValueError("model discovery_method does not match fitted shapelet bank")
        if not np.isclose(shapelets_value.input_fs_hz, input_fs_hz, rtol=0.0, atol=1e-12):
            raise ValueError("model input_fs_hz does not match fitted shapelet bank")
        if (
            shapelets_value.outer_repeat_index != outer_repeat_index
            or shapelets_value.outer_fold_index != outer_fold_index
        ):
            raise ValueError("model outer repeat/fold does not match fitted shapelet bank")
        if shapelets_value.outer_train_participant_hash != outer_train_participant_hash:
            raise ValueError("model outer-train roster hash does not match fitted shapelet bank")
        if model_id != "shapeformer_effect_size_fixed_v1":
            pip_rounding_rule = str(config.pop("pip_rounding_rule"))
            pip_selection_rule = str(config.pop("pip_selection_rule"))
            candidate_generation_rule = str(
                config.pop("candidate_generation_rule")
            )
            candidate_enumeration_rule = str(
                config.pop("candidate_enumeration_rule")
            )
            candidate_ranking_rule = str(config.pop("candidate_ranking_rule"))
            selected_bank_order_rule = str(
                config.pop("selected_bank_order_rule")
            )
            discovery_position_search_boundary_rule = str(
                config.pop("discovery_position_search_boundary_rule")
            )
            information_gain_split_rule = str(
                config.pop("information_gain_split_rule")
            )
            if (
                pip_rounding_rule != PIP_ROUNDING_RULE
                or shapelets_value.pip_rounding_rule != PIP_ROUNDING_RULE
            ):
                raise ValueError(
                    "model pip_rounding_rule does not match the upstream PISD bank"
                )
            if (
                pip_selection_rule != PIP_SELECTION_RULE
                or shapelets_value.pip_selection_rule != PIP_SELECTION_RULE
            ):
                raise ValueError(
                    "model pip_selection_rule does not match the upstream PISD bank"
                )
            if (
                candidate_generation_rule != CANDIDATE_GENERATION_RULE
                or shapelets_value.candidate_generation_rule
                != CANDIDATE_GENERATION_RULE
            ):
                raise ValueError(
                    "model candidate_generation_rule does not match the upstream PISD bank"
                )
            if (
                candidate_enumeration_rule != CANDIDATE_ENUMERATION_RULE
                or shapelets_value.candidate_enumeration_rule
                != CANDIDATE_ENUMERATION_RULE
            ):
                raise ValueError(
                    "model candidate_enumeration_rule does not match the "
                    "upstream PISD bank"
                )
            if (
                candidate_ranking_rule != CANDIDATE_RANKING_RULE
                or shapelets_value.candidate_ranking_rule
                != CANDIDATE_RANKING_RULE
            ):
                raise ValueError(
                    "model candidate_ranking_rule does not match the upstream PISD bank"
                )
            if (
                selected_bank_order_rule != SELECTED_BANK_ORDER_RULE
                or shapelets_value.selected_bank_order_rule
                != SELECTED_BANK_ORDER_RULE
            ):
                raise ValueError(
                    "model selected_bank_order_rule does not match the upstream PISD bank"
                )
            if (
                discovery_position_search_boundary_rule
                != DISCOVERY_POSITION_SEARCH_BOUNDARY_RULE
                or shapelets_value.discovery_position_search_boundary_rule
                != DISCOVERY_POSITION_SEARCH_BOUNDARY_RULE
            ):
                raise ValueError(
                    "model discovery_position_search_boundary_rule does not "
                    "match the upstream PISD bank"
                )
            if (
                information_gain_split_rule != INFORMATION_GAIN_SPLIT_RULE
                or shapelets_value.information_gain_split_rule
                != INFORMATION_GAIN_SPLIT_RULE
            ):
                raise ValueError(
                    "model information_gain_split_rule does not match the "
                    "upstream PISD bank"
                )

        architecture_fields = (
            {
                "sequence_length_samples",
                "local_kernel_width_samples",
                "local_embedding_channels",
                "shape_embedding_channels",
                "attention_feedforward_channels",
                "attention_heads",
                "attention_query_chunk_size",
                "distance_position_chunk_size",
                "dropout",
                "complexity_norm",
                "max_complexity_ratio",
                "position_search_neighbourhood_samples",
            }
            if model_id == "shapeformer_channel_specific_osd"
            else {
                "hidden_channels",
                "dropout",
                "patch_size_samples",
                "attention_heads",
                "attention_layers",
                "distance_position_chunk_size",
            }
        )
        missing_architecture = sorted(architecture_fields - set(config))
        if missing_architecture:
            raise ValueError(f"ShapeFormer missing explicit architecture options: {missing_architecture}")
        _torch_seed(seed)
        if model_id == "shapeformer_channel_specific_osd":
            from .shapeformer_literature import (
                LiteratureShapeFormerChannelSpecificOSD,
            )

            model = LiteratureShapeFormerChannelSpecificOSD(
                n_channels=spec.n_channels,
                n_classes=spec.n_classes,
                sequence_length=int(config.pop("sequence_length_samples")),
                shapelets=shapelets_value,
                local_kernel_width_samples=int(
                    config.pop("local_kernel_width_samples")
                ),
                local_embedding_channels=int(
                    config.pop("local_embedding_channels")
                ),
                shape_embedding_channels=int(
                    config.pop("shape_embedding_channels")
                ),
                attention_feedforward_channels=int(
                    config.pop("attention_feedforward_channels")
                ),
                attention_heads=int(config.pop("attention_heads")),
                attention_query_chunk_size=int(
                    config.pop("attention_query_chunk_size")
                ),
                distance_position_chunk_size=int(
                    config.pop("distance_position_chunk_size")
                ),
                dropout=float(config.pop("dropout")),
                complexity_norm=float(config.pop("complexity_norm")),
                max_complexity_ratio=float(config.pop("max_complexity_ratio")),
                position_search_neighbourhood_samples=int(
                    config.pop("position_search_neighbourhood_samples")
                ),
                input_fs_hz=input_fs_hz,
            )
        else:
            model = ExperimentalShapeFormer(
                spec.n_channels,
                spec.n_classes,
                shapelets_value,
                hidden_channels=int(config.pop("hidden_channels")),
                dropout=float(config.pop("dropout")),
                patch_size_samples=int(config.pop("patch_size_samples")),
                attention_heads=int(config.pop("attention_heads")),
                attention_layers=int(config.pop("attention_layers")),
                distance_position_chunk_size=int(
                    config.pop("distance_position_chunk_size")
                ),
                input_fs_hz=input_fs_hz,
            )
        if config:
            raise ValueError(f"unknown ShapeFormer options: {sorted(config)}")
        model.model_id = model_id
        model.canonical_model_name = canonical_model_name
        model.registry_role = (
            "reference" if model_id == "shapeformer_channel_specific_osd" else "ablation"
        )
        return _finalize_model(model, declared_architecture, spec)

    if model_id in {"fusion_compact", "fusion_inception"}:
        if mode is not RepresentationMode.FUSION:
            raise ValueError("file-bag fusion requires fusion representation")
        if spec.n_channels <= 0 or spec.n_file_features <= 0:
            raise ValueError("fusion requires positive signal channels and file-feature width")
        from .compact_cnn import CompactCNN1D
        from .fusion import FileBagFusionClassifier
        from .inception import InceptionTimeSingleNetwork

        _torch_seed(seed)
        if model_id == "fusion_compact":
            required_signal = {
                "signal_dropout",
                "signal_kernel_sizes",
                "signal_dilations",
                "signal_pool_sizes",
            }
            missing_signal = sorted(required_signal - set(config))
            if missing_signal:
                raise ValueError(f"fusion_compact missing explicit options: {missing_signal}")
            signal = CompactCNN1D(
                spec.n_channels,
                spec.n_classes,
                dropout=float(config.pop("signal_dropout")),
                kernel_sizes=tuple(int(value) for value in config.pop("signal_kernel_sizes")),
                dilations=tuple(int(value) for value in config.pop("signal_dilations")),
                pool_sizes=tuple(int(value) for value in config.pop("signal_pool_sizes")),
            )
        else:
            required_signal = {
                "signal_variant",
                "signal_dropout",
                "signal_kernel_sizes",
                "signal_dilation",
            }
            missing_signal = sorted(required_signal - set(config))
            if missing_signal:
                raise ValueError(f"fusion_inception missing explicit options: {missing_signal}")
            signal = InceptionTimeSingleNetwork(
                spec.n_channels,
                spec.n_classes,
                variant=str(config.pop("signal_variant")),
                dropout=float(config.pop("signal_dropout")),
                kernel_sizes=tuple(int(value) for value in config.pop("signal_kernel_sizes")),
                dilation=int(config.pop("signal_dilation")),
            )
        required_fusion = {"feature_hidden_dim", "fusion_hidden_dim", "pooling", "dropout"}
        missing_fusion = sorted(required_fusion - set(config))
        if missing_fusion:
            raise ValueError(f"{model_id} missing explicit options: {missing_fusion}")
        feature_hidden_dim = int(config.pop("feature_hidden_dim"))
        fusion_hidden_dim = int(config.pop("fusion_hidden_dim"))
        pooling = str(config.pop("pooling"))
        dropout = float(config.pop("dropout"))
        if config:
            raise ValueError(f"unknown fusion options: {sorted(config)}")
        model = FileBagFusionClassifier(
            signal,
            int(signal.feature_dim),
            spec.n_file_features,
            spec.n_classes,
            feature_hidden_dim=feature_hidden_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            pooling=pooling,
            dropout=dropout,
        )
        model.model_id = model_id
        model.canonical_model_name = canonical_model_name
        return _finalize_model(model, declared_architecture, spec)

    raise ValueError(f"unsupported model_id: {model_id}")


# Backward-compatible descriptive alias used by internal configuration code.
# 内部配置代码使用的描述性兼容别名。
build_model = create_model
