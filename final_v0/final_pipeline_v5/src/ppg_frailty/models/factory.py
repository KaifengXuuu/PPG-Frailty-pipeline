"""Strict model factory spanning all four representation modes.

覆盖四种 representation mode 的严格模型工厂。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np

from ..contracts import RepresentationMode
from .feature_baselines import FeatureVectorBaseline


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


SHAPEFORMER_REFERENCE_NUMERIC_DEFAULTS: dict[str, int | float] = {
    "input_fs_hz": 400.0,
    "num_pip_ratio": 0.20,
    "shapelets_per_class": 3,
    "max_discovery_windows": 180,
    "position_search_neighbourhood_samples": 128,
    "sequence_length_samples": 2000,
    "local_kernel_width_samples": 8,
    "local_embedding_channels": 48,
    "shape_embedding_channels": 128,
    "attention_feedforward_channels": 256,
    "attention_heads": 4,
    "attention_query_chunk_size": 128,
    "distance_position_chunk_size": 256,
    "dropout": 0.30,
    "complexity_norm": 1000.0,
    "max_complexity_ratio": 3.0,
}

SHAPEFORMER_EXPERIMENTAL_NUMERIC_DEFAULTS: dict[str, int | float] = {
    "input_fs_hz": 400.0,
    "shapelets_per_class": 3,
    "max_candidates_per_class": 128,
    "hidden_channels": 64,
    "dropout": 0.20,
    "patch_size_samples": 16,
    "attention_heads": 4,
    "attention_layers": 1,
    "attention_feedforward_channels": 128,
    "distance_position_chunk_size": 256,
}

SHAPEFORMER_LEGACY_EFFECT_SIZE_DEFAULTS: dict[str, int | float] = {
    "input_fs_hz": 400.0,
    "sequence_length_samples": 2000,
    "shapelet_length_samples": 128,
    "discovery_stride_samples": 64,
    "shapelets_per_class": 3,
    "max_discovery_windows": 180,
    "candidates_per_class_channel": 8,
    # The historical functional local convolution used width 8.  Its exposed
    # len_w=64 bookkeeping was not consumed by forward and is not revived as a
    # fake option; the effective 64-sample control is the shapelet search span.
    "local_kernel_width_samples": 8,
    "local_embedding_channels": 48,
    "shape_embedding_channels": 128,
    "attention_feedforward_channels": 256,
    "attention_heads": 4,
    "dropout": 0.30,
    "shapelet_search_window_samples": 64,
    "complexity_norm": 1000.0,
    "max_complexity_ratio": 3.0,
}

SHAPEFORMER_CHANNEL_SPECIFIC_RULE_DEFAULTS: dict[str, str] = {
    "discovery_method": "channel_specific_osd",
    "discovery_balance": "participant_file_balanced",
    "pip_rounding_rule": "floor_ratio_minimum_5_capped_at_actual_T",
    "pip_selection_rule": ("upstream_zscored_time_index_perpendicular_distance_first_max"),
    "candidate_generation_rule": ("insertion_stage_three_consecutive_pips_half_open"),
    "candidate_enumeration_rule": ("upstream_class_channel_source_sample_insertion_order"),
    "candidate_ranking_rule": "upstream_numpy_default_argsort_then_reverse",
    "selected_bank_order_rule": "upstream_per_class_start_sample_default_argsort",
    "discovery_position_search_boundary_rule": ("upstream_pcs_start_minus_w_plus_1_end_plus_w_half_open"),
    "information_gain_split_rule": "upstream_positive_recall_grid_0p2",
}


_SEED_POLICY_ALIASES = {
    "outer_repeat": "outer_repeat",
    "outer_cv_repeat_seed_equals_split_seed": "outer_repeat",
    "fixed_explicit": "fixed_explicit",
    "fixed": "fixed_explicit",
    "cv_fixed_member0_seed_50042_comparator": "fixed_explicit",
    "final_refit_single_seed_42": "fixed_explicit",
    "member_roster": "member_roster",
    "cv_fixed_five_member_seed_roster": "member_roster",
    "final_refit_five_member_seeds": "member_roster",
}

def _normalise_seed(value: Any, *, field: str) -> int:
    """Return one exact integer seed without silently truncating floats."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must contain integer seeds, not booleans")
    try:
        seed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must contain integer seeds") from exc
    if isinstance(value, (float, np.floating)) and (not np.isfinite(value) or float(value) != float(seed)):
        raise ValueError(f"{field} must contain finite integer seeds")
    if seed < 0 or seed > 0xFFFF_FFFF:
        raise ValueError(f"{field} must contain executable uint32 seeds")
    return seed

def _normalise_seed_roster(values: Any, *, field: str) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field} must be an ordered list or tuple of integer seeds")
    seeds = tuple(_normalise_seed(value, field=field) for value in values)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError(f"{field} must be non-empty and unique")
    return seeds

def _normalise_positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must be a positive integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if isinstance(value, (float, np.floating)) and (not np.isfinite(value) or float(value) != float(normalized)):
        raise ValueError(f"{field} must be a finite positive integer")
    if normalized <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return normalized

def _normalise_positive_integer_sequence(
    values: Any,
    *,
    field: str,
    length: int | None = None,
    odd: bool = False,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field} must be a non-string integer sequence")
    try:
        normalized = tuple(_normalise_positive_integer(value, field=field) for value in values)
    except TypeError as exc:
        raise ValueError(f"{field} must be an integer sequence") from exc
    if not normalized or length is not None and len(normalized) != length:
        expected = "non-empty" if length is None else f"length {length}"
        raise ValueError(f"{field} must be a {expected} integer sequence")
    if odd and any(value % 2 == 0 for value in normalized):
        raise ValueError(f"{field} must contain positive odd integers")
    return normalized

def _normalise_dropout(value: Any, *, field: str = "dropout") -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a finite probability in [0, 1)") from exc
    if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
        raise ValueError(f"{field} must be a finite probability in [0, 1)")
    return probability

def _normalise_dropout_sequence(values: Any, *, field: str, length: int) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field} must be a non-string probability sequence")
    try:
        normalized = tuple(_normalise_dropout(value, field=field) for value in values)
    except TypeError as exc:
        raise ValueError(f"{field} must be a probability sequence") from exc
    if len(normalized) != length:
        raise ValueError(f"{field} must contain exactly {length} probabilities")
    return normalized

def _normalise_finite_positive(value: Any, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must be finite and positive")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be finite and positive") from exc
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return normalized

def _normalise_positive_fraction(value: Any, *, field: str) -> float:
    normalized = _normalise_finite_positive(value, field=field)
    if normalized > 1.0:
        raise ValueError(f"{field} must be finite in (0,1]")
    return normalized

def _normalise_nonzero_integer(value: Any, *, field: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be a non-zero integer")
    normalized = int(value)
    if normalized == 0:
        raise ValueError(f"{field} must be a non-zero integer")
    return normalized


_LOGISTIC_SOLVERS = {
    "lbfgs",
    "liblinear",
    "newton-cg",
    "newton-cholesky",
    "sag",
    "saga",
}

def _normalise_logistic_solver(value: Any) -> str:
    solver = str(value)
    if solver not in _LOGISTIC_SOLVERS:
        raise ValueError(f"logistic_solver must be one of {sorted(_LOGISTIC_SOLVERS)}")
    return solver

def _normalise_extra_trees_max_features(value: Any) -> str | int | float | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value not in {"sqrt", "log2"}:
            raise ValueError("extra_trees_max_features string must be 'sqrt' or 'log2'")
        return value
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("extra_trees_max_features cannot be boolean")
    if isinstance(value, (int, np.integer)):
        if int(value) <= 0:
            raise ValueError("integer extra_trees_max_features must be positive")
        return int(value)
    if isinstance(value, (float, np.floating)):
        normalized = float(value)
        if not math.isfinite(normalized) or not 0.0 < normalized <= 1.0:
            raise ValueError("float extra_trees_max_features must be in (0,1]")
        return normalized
    raise ValueError("unsupported extra_trees_max_features value")

def _normalise_extra_trees_min_samples_leaf(value: Any) -> int | float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("extra_trees_min_samples_leaf cannot be boolean")
    if isinstance(value, (int, np.integer)):
        if int(value) <= 0:
            raise ValueError("integer extra_trees_min_samples_leaf must be positive")
        return int(value)
    if isinstance(value, (float, np.floating)):
        normalized = float(value)
        if not math.isfinite(normalized) or not 0.0 < normalized <= 0.5:
            raise ValueError("float extra_trees_min_samples_leaf must be in (0,0.5]")
        return normalized
    raise ValueError("unsupported extra_trees_min_samples_leaf value")

def _normalise_training_owned_class_weight(value: Any) -> None:
    if value is not None:
        raise ValueError(
            "model.class_weight is not an independent weighting capability; "
            "configure the single training.class_weighting strategy"
        )
    return None

def resolve_seed_policy(
    seed_policy: str | None = None,
    *,
    seed: Any | None = None,
    outer_repeat_seed: Any | None = None,
    member_seeds: Any | None = None,
) -> tuple[int, ...]:
    """Resolve an optional seed strategy into the seeds actually executed.

    The three generic strategies are ``outer_repeat``, ``fixed_explicit`` and
    ``member_roster``. Historical policy names remain accepted as aliases, but
    names that encode a literal seed/member count retain that exact meaning.
    """

    if seed_policy is None or not str(seed_policy).strip():
        policy = (
            "member_roster"
            if member_seeds is not None
            else "outer_repeat"
            if outer_repeat_seed is not None
            else "fixed_explicit"
        )
        declared_policy = policy
    else:
        declared_policy = str(seed_policy).strip()
        try:
            policy = _SEED_POLICY_ALIASES[declared_policy]
        except KeyError as exc:
            raise ValueError(
                f"unsupported seed_policy {declared_policy!r}; expected one of " f"{sorted(_SEED_POLICY_ALIASES)}"
            ) from exc

    if policy == "member_roster":
        if seed is not None or outer_repeat_seed is not None:
            raise ValueError("member_roster cannot also declare a single/outer-repeat seed")
        seeds = _normalise_seed_roster(member_seeds, field="member_seeds")
        if declared_policy == "cv_fixed_five_member_seed_roster" and len(seeds) != 5:
            raise ValueError("cv_fixed_five_member_seed_roster declares exactly five member seeds")
        return seeds

    if member_seeds is not None:
        raise ValueError(f"{policy} cannot also declare member_seeds")
    if policy == "outer_repeat":
        if outer_repeat_seed is None:
            if seed is None:
                raise ValueError("outer_repeat requires outer_repeat_seed")
            outer_repeat_seed = seed
        resolved = _normalise_seed(outer_repeat_seed, field="outer_repeat_seed")
        if seed is not None and _normalise_seed(seed, field="seed") != resolved:
            raise ValueError("seed differs from the declared outer_repeat_seed")
        return (resolved,)

    if seed is None:
        raise ValueError("fixed_explicit requires seed")
    if outer_repeat_seed is not None:
        raise ValueError("fixed_explicit cannot also declare outer_repeat_seed")
    resolved = _normalise_seed(seed, field="seed")
    legacy_fixed = {
        "cv_fixed_member0_seed_50042_comparator": 50042,
        "final_refit_single_seed_42": 42,
    }
    if declared_policy in legacy_fixed and resolved != legacy_fixed[declared_policy]:
        raise ValueError(f"{declared_policy} declares seed {legacy_fixed[declared_policy]}, got {resolved}")
    return (resolved,)

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
    seeds = _normalise_seed_roster(payload["random_seeds"], field="random_seeds")
    if not channels or len(channels) != len(set(channels)):
        raise ValueError("input_channels_order must be non-empty and unique")
    policy = str(payload["seed_policy"])
    strategy = _SEED_POLICY_ALIASES.get(policy)
    if strategy == "member_roster":
        resolved_seeds = resolve_seed_policy(policy, member_seeds=seeds)
    else:
        if len(seeds) != 1:
            raise ValueError(f"seed_policy {policy!r} executes exactly one seed")
        resolved_seeds = resolve_seed_policy(
            policy,
            seed=seeds[0],
            outer_repeat_seed=(payload.get("outer_repeat_seed", seeds[0]) if strategy == "outer_repeat" else None),
        )
    if resolved_seeds != seeds:
        raise ValueError("seed_policy declaration differs from random_seeds execution roster")
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
    "LogisticRegressionL2": "logistic_regression",
    "RBFSVM": "rbf_svm",
    "ExtraTrees": "extra_trees",
    "ShapeFormerChannelSpecificOSD": "shapeformer_channel_specific_osd",
    "ShapeFormerChannelSpecificScalarDistanceAblation": ("shapeformer_channel_specific_scalar_distance_ablation"),
    "ShapeFormerEffectSizeFixedV1": "shapeformer_effect_size_fixed_v1",
    "ShapeFormerLegacyEffectSizePort": "shapeformer_legacy_effect_size_port",
    "FileBagFusionCompact": "fusion_compact",
    "FileBagFusionInception": "fusion_inception",
    "FileBagFusion": "file_bag_fusion",
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
    ModelCandidate("LogisticRegressionL2", "logistic_regression", "feature_vector", "reference", "classical"),
    ModelCandidate("RBFSVM", "rbf_svm", "feature_vector", "reference", "classical"),
    ModelCandidate("ExtraTrees", "extra_trees", "feature_vector", "reference", "classical"),
    ModelCandidate(
        "ShapeFormerChannelSpecificOSD", "shapeformer_channel_specific_osd", "raw", "reference", "shapeformer_discovery"
    ),
    ModelCandidate(
        "ShapeFormerEffectSizeFixedV1", "shapeformer_effect_size_fixed_v1", "raw", "ablation", "shapeformer_discovery"
    ),
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
_FORMAL_CATALOG_CANDIDATE_BY_MACHINE_ID = {item.machine_id: item for item in NONENSEMBLE_MODEL_CANDIDATES}

def model_candidate(model_id_or_name: str) -> ModelCandidate:
    """Return any registered non-ensemble model, not only the 13-case preset.

    ``NONENSEMBLE_MODEL_CANDIDATES`` remains the historical formal-catalog
    preset for compatibility. Runtime discovery comes from the complete module
    registry so optional and later parallel models do not require a second
    allow-list here.
    """

    from ..module_registry import model_factory_contract

    contract = model_factory_contract(model_id_or_name)
    if "member_seeds" in set(contract["factory_fields"]):
        raise ValueError(
            "model_candidate describes non-ensemble registered models; use the "
            "explicit ensemble comparison descriptors for ensemble identities"
        )
    modes = tuple(str(value) for value in contract["representation_modes"])
    if len(modes) != 1:
        raise ValueError("registered model candidate must have one representation mode")
    machine_id = str(contract["machine_model_id"])
    preset = _FORMAL_CATALOG_CANDIDATE_BY_MACHINE_ID.get(machine_id)
    if preset is not None:
        return preset
    return ModelCandidate(
        canonical_name=str(contract["canonical_model_name"]),
        machine_id=machine_id,
        representation_mode=modes[0],
        registry_role=str(contract["registry_role"]),
        comparison_family=str(contract["scientific_status"]),
        execution_status="registered_runnable_contract_not_benchmarked",
    )

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
    }:
        requested = sorted({"epochs", "fixed_epochs", "epoch_profile", "epoch_rule"} & set(config))
        if requested:
            raise ValueError("non-iterative estimator candidates do not accept epoch settings; " f"remove {requested}")
    config.pop("model_name", None)
    config["model_id"] = machine
    config["canonical_model_name"] = canonical
    return config


FUSION_SIGNAL_ENCODER_IDS = frozenset(
    {
        "compact_cnn",
        "inception_full",
        "inception_small",
        "shapeformer_channel_specific_osd",
        "shapeformer_channel_specific_scalar_distance_ablation",
        "shapeformer_effect_size_fixed_v1",
        "shapeformer_legacy_effect_size_port",
    }
)
FUSION_SHAPEFORMER_ENCODER_IDS = frozenset(
    {
        "shapeformer_channel_specific_osd",
        "shapeformer_channel_specific_scalar_distance_ablation",
        "shapeformer_effect_size_fixed_v1",
        "shapeformer_legacy_effect_size_port",
    }
)

def normalize_fusion_signal_encoder_config(value: Any) -> dict[str, Any]:
    """Normalize one raw signal encoder used by the generic file-bag composer.

    The nested mapping is a real factory input surface, not a label.  Only raw
    torch models that expose ``forward_features`` are composable.  Seed and
    fold-local discovery state are owned by the outer factory and are injected
    after the frozen outer-train dataset has been verified.
    """

    if value is None:
        value = {"model_id": "compact_cnn"}
    if not isinstance(value, Mapping):
        raise ValueError("signal_encoder must be a model configuration mapping")
    config = normalize_model_config(value)
    model_id = str(config["model_id"])
    if model_id not in FUSION_SIGNAL_ENCODER_IDS:
        raise ValueError(
            "file-bag signal_encoder must be one of the registered raw "
            f"feature encoders: {sorted(FUSION_SIGNAL_ENCODER_IDS)}"
        )
    return config


FRAILTY_RAW_CHANNEL_SCHEMA = (
    "RED",
    "IR",
    "A_dyn_x",
    "A_dyn_y",
    "A_dyn_z",
    "GX",
    "GY",
    "GZ",
)

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
        payload = dict(value)
        for key in ("feature_names", "channel_schema"):
            if key in payload:
                payload[key] = tuple(str(item) for item in payload[key])
        return cls(**payload)

    @property
    def mode(self) -> RepresentationMode:
        """Return validated enum mode / 返回已校验枚举模式。"""

        return RepresentationMode(self.representation_mode)

def _validate_frailty_factory_input(
    spec: ModelInputSpec,
    *,
    require_explicit_schema: bool,
) -> None:
    """Fail closed at model construction while keeping transport specs generic."""

    if spec.mode not in {RepresentationMode.RAW, RepresentationMode.FUSION}:
        return
    if require_explicit_schema and not spec.channel_schema:
        raise ValueError("formal frailty raw/fusion preparation requires explicit channel_schema")
    if spec.mode is RepresentationMode.FUSION:
        if spec.n_channels != len(FRAILTY_RAW_CHANNEL_SCHEMA) or spec.channel_schema != FRAILTY_RAW_CHANNEL_SCHEMA:
            raise ValueError("frailty fusion model factory requires the canonical ordered 8 channels")
        return
    schema = tuple(spec.channel_schema)
    if (
        spec.n_channels <= 0
        or len(schema) != spec.n_channels
        or len(schema) != len(set(schema))
        or any(value not in FRAILTY_RAW_CHANNEL_SCHEMA for value in schema)
        or tuple(value for value in FRAILTY_RAW_CHANNEL_SCHEMA if value in schema) != schema
    ):
        raise ValueError("frailty raw channel_schema must be a non-empty ordered subset of " "the canonical 8 channels")

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
        "pool_size": int(model.pool_size),
        "branch_count": int(model.branch_count),
        "residual_interval": int(model.residual_interval),
        "global_pooling": "mask_aware_global_average",
        "classifier_dropout": float(model.classifier_dropout),
    }

def _fusion_raw_input_spec(spec: ModelInputSpec) -> ModelInputSpec:
    """Return the exact raw-tensor view consumed by a fusion signal encoder."""

    return ModelInputSpec(
        RepresentationMode.RAW,
        n_channels=int(spec.n_channels),
        n_classes=int(spec.n_classes),
        channel_schema=tuple(spec.channel_schema),
    )

def _signal_feature_dim_from_architecture(
    architecture: Mapping[str, Any],
) -> int:
    """Derive the encoder output width from its complete raw architecture."""

    model_id = str(architecture.get("model_id", ""))
    if model_id == "compact_cnn":
        width = tuple(int(value) for value in architecture["stage_channels"])[-1]
    elif model_id in {"inception_full", "inception_small"}:
        width = int(architecture["out_channels"]) * int(architecture["branch_count"])
    elif model_id == "shapeformer_channel_specific_osd":
        width = int(architecture["local_embedding_channels"]) + int(architecture["shape_embedding_channels"])
    elif model_id in {
        "shapeformer_channel_specific_scalar_distance_ablation",
        "shapeformer_effect_size_fixed_v1",
    }:
        width = int(architecture["hidden_channels"]) + int(architecture["shapelet_count"])
    elif model_id == "shapeformer_legacy_effect_size_port":
        width = int(architecture["local_embedding_channels"]) + int(architecture["shape_embedding_channels"])
    else:  # pragma: no cover - guarded by normalize_fusion_signal_encoder_config
        raise ValueError(f"unsupported fusion signal encoder architecture: {model_id}")
    if width <= 0:
        raise ValueError("fusion signal encoder must expose a positive feature width")
    return width


_INCEPTION_CAPACITY = {"full": (32, 32, 6), "small": (16, 16, 3)}
_ENSEMBLE_IDS = {
    "inception_full_five_member_ensemble",
    "inception_matrix_five_member_ensemble",
}
_SHAPEFORMER_IDS = {
    "shapeformer_channel_specific_osd",
    "shapeformer_channel_specific_scalar_distance_ablation",
    "shapeformer_effect_size_fixed_v1",
    "shapeformer_legacy_effect_size_port",
}

def _bank_value(bank: Any, name: str) -> Any:
    return bank[name] if isinstance(bank, Mapping) else getattr(bank, name)

def _compact_options(config: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Resolve the one CompactCNN parameter surface used by raw and fusion."""

    get = lambda name, default: config.get(f"{prefix}{name}", default)
    return {
        "stage_channels": _normalise_positive_integer_sequence(
            get("stage_channels", (32, 64, 128)), field=f"{prefix}stage_channels", length=3
        ),
        "kernel_sizes": _normalise_positive_integer_sequence(
            get("kernel_sizes", (9, 9, 7)), field=f"{prefix}kernel_sizes", length=3, odd=True
        ),
        "dilations": _normalise_positive_integer_sequence(
            get("dilations", (1, 1, 1)), field=f"{prefix}dilations", length=3
        ),
        "pool_sizes": _normalise_positive_integer_sequence(
            get("pool_sizes", (4, 4)), field=f"{prefix}pool_sizes", length=2
        ),
        "stage_dropouts": _normalise_dropout_sequence(
            get("stage_dropouts", (0.10, 0.15)), field=f"{prefix}stage_dropouts", length=2
        ),
        "dropout": _normalise_dropout(get("dropout", 0.20), field=f"{prefix}dropout"),
    }

def _compact_parameters(options: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **{
            name: options[name]
            for name in ("stage_channels", "kernel_sizes", "dilations", "pool_sizes", "stage_dropouts")
        },
        "global_pooling": "adaptive_average_1",
        "classifier_dropout": options["dropout"],
    }

def _inception_options(config: Mapping[str, Any], variant: str, prefix: str = "") -> dict[str, Any]:
    """Resolve shared Inception parameters, including derived capacity."""

    capacity = _INCEPTION_CAPACITY[variant]
    get = lambda name, default: config.get(f"{prefix}{name}", default)
    kernels = _normalise_positive_integer_sequence(
        get("kernel_sizes", (39, 19, 9)), field=f"{prefix}kernel_sizes", odd=True
    )
    return {
        "variant": variant,
        "out_channels": _normalise_positive_integer(get("out_channels", capacity[0]), field=f"{prefix}out_channels"),
        "bottleneck_channels": _normalise_positive_integer(
            get("bottleneck_channels", capacity[1]), field=f"{prefix}bottleneck_channels"
        ),
        "depth": _normalise_positive_integer(get("depth", capacity[2]), field=f"{prefix}depth"),
        "kernel_sizes": kernels,
        "dilation": _normalise_positive_integer(get("dilation", 1), field=f"{prefix}dilation"),
        "pool_size": _normalise_positive_integer(get("pool_size", 3), field=f"{prefix}pool_size"),
        "residual_interval": _normalise_positive_integer(
            get("residual_interval", 3), field=f"{prefix}residual_interval"
        ),
        "dropout": _normalise_dropout(get("dropout", 0.20), field=f"{prefix}dropout"),
    }

def _inception_parameters(options: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **{
            name: options[name]
            for name in (
                "variant",
                "out_channels",
                "bottleneck_channels",
                "depth",
                "kernel_sizes",
                "dilation",
                "pool_size",
                "residual_interval",
            )
        },
        "branch_count": len(options["kernel_sizes"]) + 1,
        "global_pooling": "mask_aware_global_average",
        "classifier_dropout": options["dropout"],
    }


_BASELINE_DEFAULTS = {
    "logistic_regression": {"logistic_c": 1.0, "logistic_max_iter": 5000, "logistic_solver": "lbfgs"},
    "rbf_svm": {"svm_kernel": "rbf", "svm_probability": True, "svm_c": 1.0, "svm_gamma": "scale"},
    "extra_trees": {
        "extra_trees_n_estimators": 500,
        "extra_trees_n_jobs": 1,
        "extra_trees_max_features": "sqrt",
        "extra_trees_min_samples_leaf": 1,
    },
}

def _baseline_options(model_id: str, config: Mapping[str, Any]) -> dict[str, Any]:
    options = {name: config.get(name, default) for name, default in _BASELINE_DEFAULTS[model_id].items()}
    options["class_weight"] = _normalise_training_owned_class_weight(config.get("class_weight"))
    if model_id == "logistic_regression":
        options.update(
            logistic_c=_normalise_finite_positive(options["logistic_c"], field="logistic_c"),
            logistic_max_iter=_normalise_positive_integer(options["logistic_max_iter"], field="logistic_max_iter"),
            logistic_solver=_normalise_logistic_solver(options["logistic_solver"]),
        )
    elif model_id == "extra_trees":
        options.update(
            extra_trees_n_estimators=_normalise_positive_integer(
                options["extra_trees_n_estimators"], field="extra_trees_n_estimators"
            ),
            extra_trees_n_jobs=_normalise_nonzero_integer(options["extra_trees_n_jobs"], field="extra_trees_n_jobs"),
            extra_trees_max_features=_normalise_extra_trees_max_features(options["extra_trees_max_features"]),
            extra_trees_min_samples_leaf=_normalise_extra_trees_min_samples_leaf(
                options["extra_trees_min_samples_leaf"]
            ),
        )
    return options

def _baseline_parameters(model_id: str, options: Mapping[str, Any]) -> dict[str, Any]:
    if model_id == "logistic_regression":
        detail = {
            "preprocessing": ("median_imputer_keep_empty_features", "standard_scaler"),
            "estimator": "sklearn.linear_model.LogisticRegression",
            "C": options["logistic_c"],
            "max_iter": options["logistic_max_iter"],
            "solver": options["logistic_solver"],
        }
    elif model_id == "rbf_svm":
        detail = {
            "preprocessing": ("median_imputer_keep_empty_features", "standard_scaler"),
            "estimator": "sklearn.svm.SVC",
            "kernel": str(options["svm_kernel"]),
            "probability": bool(options["svm_probability"]),
            "C": float(options["svm_c"]),
            "gamma": options["svm_gamma"],
        }
    else:
        detail = {
            "preprocessing": ("median_imputer_keep_empty_features",),
            "estimator": "sklearn.ensemble.ExtraTreesClassifier",
            "n_estimators": options["extra_trees_n_estimators"],
            "n_jobs": options["extra_trees_n_jobs"],
            "max_features": options["extra_trees_max_features"],
            "min_samples_leaf": options["extra_trees_min_samples_leaf"],
        }
    return {"class_weight": options["class_weight"], **detail}

def _shapelet_count(config: Mapping[str, Any], spec: ModelInputSpec, *, channel_specific: bool) -> tuple[int, int]:
    bank = config.get("shapelets")
    if "shapelets_per_class" in config:
        per_class = _normalise_positive_integer(config["shapelets_per_class"], field="shapelets_per_class")
    elif bank is not None:
        counts = np.unique(np.asarray(_bank_value(bank, "source_classes")), return_counts=True)[1]
        if counts.size < 2 or not np.all(counts == counts[0]):
            raise ValueError("shapelet bank must have equal per-class capacity")
        per_class = int(counts[0])
    else:
        defaults = (
            SHAPEFORMER_REFERENCE_NUMERIC_DEFAULTS if channel_specific else SHAPEFORMER_EXPERIMENTAL_NUMERIC_DEFAULTS
        )
        per_class = int(defaults["shapelets_per_class"])
    return (len(_bank_value(bank, "values")) if bank is not None else spec.n_classes * per_class, per_class)

def _shape_value(config: Mapping[str, Any], name: str) -> Any:
    bank = config.get("shapelets")
    if name in config:
        return config[name]
    if bank is not None:
        return _bank_value(bank, name)
    if name in SHAPEFORMER_REFERENCE_NUMERIC_DEFAULTS:
        return SHAPEFORMER_REFERENCE_NUMERIC_DEFAULTS[name]
    if name in SHAPEFORMER_CHANNEL_SPECIFIC_RULE_DEFAULTS:
        return SHAPEFORMER_CHANNEL_SPECIFIC_RULE_DEFAULTS[name]
    raise ValueError(f"ShapeFormer architecture is missing {name}")

def _experimental_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    defaults = SHAPEFORMER_EXPERIMENTAL_NUMERIC_DEFAULTS
    hidden = _normalise_positive_integer(
        config.get("hidden_channels", defaults["hidden_channels"]), field="hidden_channels"
    )
    patch = _normalise_positive_integer(
        config.get("patch_size_samples", defaults["patch_size_samples"]), field="patch_size_samples"
    )
    return {
        "hidden_channels": hidden,
        "patch_size_samples": patch,
        "patch_stride_samples": patch,
        "patch_bias": False,
        "attention_heads": _normalise_positive_integer(
            config.get("attention_heads", defaults["attention_heads"]), field="attention_heads"
        ),
        "attention_layers": _normalise_positive_integer(
            config.get("attention_layers", defaults["attention_layers"]), field="attention_layers"
        ),
        "attention_feedforward_channels": _normalise_positive_integer(
            config.get("attention_feedforward_channels", hidden * 2), field="attention_feedforward_channels"
        ),
        "attention_activation": "gelu",
        "attention_norm_first": False,
        "distance_position_chunk_size": _normalise_positive_integer(
            config.get("distance_position_chunk_size", defaults["distance_position_chunk_size"]),
            field="distance_position_chunk_size",
        ),
        "classifier_dropout": _normalise_dropout(config.get("dropout", defaults["dropout"])),
    }

def _shapeformer_parameters(model_id: str, config: Mapping[str, Any], spec: ModelInputSpec) -> dict[str, Any]:
    reference = model_id == "shapeformer_channel_specific_osd"
    channel_specific = model_id != "shapeformer_effect_size_fixed_v1"
    count, per_class = _shapelet_count(config, spec, channel_specific=channel_specific)
    base = {
        "discovery_method": str(
            config.get(
                "discovery_method",
                "effect_size_fixed_v1" if not channel_specific else _shape_value(config, "discovery_method"),
            )
        ),
        "shapelet_count": int(count),
        "shapelet_count_per_class": per_class,
        "shapelet_channel_policy": "single_source_channel" if channel_specific else "joint_all_channels",
        "shapelet_length_policy": "variable_insertion_stage_three_consecutive_pips"
        if channel_specific
        else "fixed_samples",
        "shapelet_length_samples": None
        if channel_specific
        else _normalise_positive_integer(
            _bank_value(config["shapelets"], "shapelet_length_samples")
            if config.get("shapelets") is not None
            else config.get("shapelet_length_samples", 128),
            field="shapelet_length_samples",
        ),
        "candidate_stride_samples": None
        if channel_specific
        else _normalise_positive_integer(
            _bank_value(config["shapelets"], "discovery_stride_samples")
            if config.get("shapelets") is not None
            else config.get("discovery_stride_samples", 64),
            field="discovery_stride_samples",
        ),
        "best_fit_search_stride_samples": 1,
        "input_fs_hz": _normalise_finite_positive(
            config.get(
                "input_fs_hz",
                (
                    SHAPEFORMER_REFERENCE_NUMERIC_DEFAULTS
                    if channel_specific
                    else SHAPEFORMER_EXPERIMENTAL_NUMERIC_DEFAULTS
                )["input_fs_hz"],
            ),
            field="input_fs_hz",
        ),
    }
    if not reference:
        base.update(_experimental_parameters(config))
        if channel_specific:
            base.update(
                downstream_status="scalar_distance_ablation_not_literature_shapeformer",
                num_pip_ratio=_normalise_positive_fraction(
                    _shape_value(config, "num_pip_ratio"), field="num_pip_ratio"
                ),
                max_discovery_windows=_normalise_positive_integer(
                    _shape_value(config, "max_discovery_windows"), field="max_discovery_windows"
                ),
                discovery_balance=str(_shape_value(config, "discovery_balance")),
                position_search_neighbourhood_samples=_normalise_positive_integer(
                    _shape_value(config, "position_search_neighbourhood_samples"),
                    field="position_search_neighbourhood_samples",
                ),
                information_gain_split_rule=str(_shape_value(config, "information_gain_split_rule")),
            )
        else:
            base.update(num_pip_ratio=None, max_discovery_windows=None, discovery_balance=None)
        return base

    defaults = SHAPEFORMER_REFERENCE_NUMERIC_DEFAULTS
    for name in ("num_pip_ratio", "max_discovery_windows", "position_search_neighbourhood_samples"):
        normalizer = _normalise_positive_fraction if name == "num_pip_ratio" else _normalise_positive_integer
        base[name] = normalizer(_shape_value(config, name), field=name)
    for name in (
        "pip_rounding_rule",
        "pip_selection_rule",
        "candidate_generation_rule",
        "candidate_enumeration_rule",
        "candidate_ranking_rule",
        "selected_bank_order_rule",
        "discovery_position_search_boundary_rule",
        "information_gain_split_rule",
        "discovery_balance",
    ):
        base[name] = str(_shape_value(config, name))
    for name in (
        "sequence_length_samples",
        "local_kernel_width_samples",
        "local_embedding_channels",
        "shape_embedding_channels",
        "attention_heads",
        "attention_query_chunk_size",
        "attention_feedforward_channels",
        "distance_position_chunk_size",
    ):
        base[name] = _normalise_positive_integer(config.get(name, defaults[name]), field=name)
    base.update(
        implementation_status="implemented_not_benchmarked_high_compute",
        complexity_norm=_normalise_finite_positive(
            config.get("complexity_norm", defaults["complexity_norm"]), field="complexity_norm"
        ),
        max_complexity_ratio=_normalise_finite_positive(
            config.get("max_complexity_ratio", defaults["max_complexity_ratio"]), field="max_complexity_ratio"
        ),
        dropout=_normalise_dropout(config.get("dropout", defaults["dropout"])),
        shapelet_token_formula="selected_projection(raw_best_segment)-shapelet_projection(shapelet)",
        shapelet_position_embeddings=("source_channel", "source_start", "source_end"),
        shapelet_weighting="learnable_initialised_from_information_gain",
        shape_position_embedding_width_policy="upstream_observed_max_plus_1_per_axis",
        attention_probability_dropout_applied=False,
        generic_branch_input="full_multivariate_input",
        generic_branch_channel_count=int(spec.n_channels),
        global_shape_pooling="first_contextualised_shape_token",
    )
    if config.get("shapelets") is not None:
        for output, source in (
            ("shape_channel_position_width", "source_channels"),
            ("shape_start_position_width", "source_starts"),
            ("shape_end_position_width", "source_ends"),
        ):
            base[output] = int(np.asarray(_bank_value(config["shapelets"], source), dtype=np.int64).max()) + 1
    return base

def _legacy_parameters(config: Mapping[str, Any], spec: ModelInputSpec) -> dict[str, Any]:
    from .shapeformer_legacy import LEGACY_DISCOVERY_BALANCE, LEGACY_EFFECT_SIZE_DISCOVERY_METHOD

    bank = config.get("shapelets")
    value = (
        lambda name: config[name]
        if name in config
        else _bank_value(bank, name)
        if bank is not None
        else SHAPEFORMER_LEGACY_EFFECT_SIZE_DEFAULTS[name]
    )
    method = str(config.get("discovery_method", LEGACY_EFFECT_SIZE_DISCOVERY_METHOD))
    balance = str(
        config.get(
            "discovery_balance",
            _bank_value(bank, "discovery_balance") if bank is not None else LEGACY_DISCOVERY_BALANCE,
        )
    )
    if method != LEGACY_EFFECT_SIZE_DISCOVERY_METHOD or balance != LEGACY_DISCOVERY_BALANCE:
        raise ValueError("legacy effect-size ShapeFormer discovery method/balance drifted")
    per_class = _normalise_positive_integer(value("shapelets_per_class"), field="shapelets_per_class")
    result = {
        "scientific_status": "legacy_parallel_ablation_not_osd_parity",
        "discovery_method": method,
        "discovery_balance": balance,
        "shapelet_count": len(_bank_value(bank, "values")) if bank is not None else spec.n_classes * per_class,
        "shapelet_count_per_class": per_class,
        "shapelet_channel_policy": "single_source_channel",
        "shapelet_length_policy": "fixed_samples",
        "shapelet_length_samples": _normalise_positive_integer(
            value("shapelet_length_samples"), field="shapelet_length_samples"
        ),
        "candidate_stride_samples": _normalise_positive_integer(
            value("discovery_stride_samples"), field="discovery_stride_samples"
        ),
        "candidates_per_class_channel": _normalise_positive_integer(
            value("candidates_per_class_channel"), field="candidates_per_class_channel"
        ),
        "max_discovery_windows": _normalise_positive_integer(
            value("max_discovery_windows"), field="max_discovery_windows"
        ),
        "sequence_length_samples": _normalise_positive_integer(
            value("sequence_length_samples"), field="sequence_length_samples"
        ),
        "input_fs_hz": _normalise_finite_positive(value("input_fs_hz"), field="input_fs_hz"),
    }
    for name in (
        "local_kernel_width_samples",
        "local_embedding_channels",
        "shape_embedding_channels",
        "attention_feedforward_channels",
        "attention_heads",
        "shapelet_search_window_samples",
    ):
        result[name] = _normalise_positive_integer(
            config.get(name, SHAPEFORMER_LEGACY_EFFECT_SIZE_DEFAULTS[name]), field=name
        )
    result.update(
        dropout=_normalise_dropout(config.get("dropout", SHAPEFORMER_LEGACY_EFFECT_SIZE_DEFAULTS["dropout"])),
        complexity_norm=_normalise_finite_positive(
            config.get("complexity_norm", SHAPEFORMER_LEGACY_EFFECT_SIZE_DEFAULTS["complexity_norm"]),
            field="complexity_norm",
        ),
        max_complexity_ratio=_normalise_finite_positive(
            config.get("max_complexity_ratio", SHAPEFORMER_LEGACY_EFFECT_SIZE_DEFAULTS["max_complexity_ratio"]),
            field="max_complexity_ratio",
        ),
        shapelet_token_formula="selected_projection(raw_best_segment)-shapelet_projection(shapelet)",
        shapelet_weighting="learnable_initialised_from_effect_score",
        local_branch="legacy_two_conv_attention_feedforward",
        shape_branch="legacy_source_position_attention_first_token",
        complete_window_required=True,
    )
    if result["max_complexity_ratio"] < 1.0:
        raise ValueError("max_complexity_ratio must be at least 1")
    return result

def resolved_architecture_parameters(model: Any, input_spec: ModelInputSpec | Mapping[str, Any]) -> dict[str, Any]:
    """Return the architecture attached by the factory, with legacy fallback."""

    attached = getattr(model, "resolved_architecture_parameters", None)
    if isinstance(attached, Mapping):
        return _normalise_architecture_value(attached)
    spec = ModelInputSpec.from_value(input_spec)
    model_id = str(getattr(model, "model_id", ""))
    base = {"model_id": model_id, "representation_mode": spec.mode.value, "n_classes": spec.n_classes}
    if model_id == "compact_cnn":
        base.update(_compact_architecture(model))
    elif model_id in {"inception_full", "inception_small", "inception_matrix"}:
        base.update(_inception_architecture(model))
    elif model_id in _ENSEMBLE_IDS:
        base.update({f"member_{key}": value for key, value in _inception_architecture(model.members[0]).items()})
        base.update(
            member_count=len(model.members),
            member_seeds=tuple(model.member_seeds),
            probability_aggregation="arithmetic_mean",
        )
    elif model_id in _BASELINE_DEFAULTS:
        options = {name: getattr(model, name) for name in _BASELINE_DEFAULTS[model_id]}
        options["class_weight"] = model.class_weight
        base.update(_baseline_parameters(model_id, options))
    else:
        raise ValueError(f"cannot resolve architecture for unsupported model_id: {model_id}")
    return _normalise_architecture_value(base)

def resolved_architecture_hash(model: Any, input_spec: ModelInputSpec | Mapping[str, Any]) -> str:
    payload = json.dumps(
        resolved_architecture_parameters(model, input_spec), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()

def materialize_architecture_parameters(
    model_config: Mapping[str, Any], input_spec: ModelInputSpec | Mapping[str, Any]
) -> dict[str, Any]:
    """Derive the complete architecture from the single factory input surface."""

    spec = ModelInputSpec.from_value(input_spec)
    config = normalize_model_config(model_config)
    model_id = str(config.pop("model_id"))
    for name in (
        "canonical_model_name",
        "architecture_parameters",
        "seed",
        "seed_policy",
        "outer_repeat_seed",
        "member_seed_roster_id",
        "comparison_only",
    ):
        config.pop(name, None)
    base = {"model_id": model_id, "representation_mode": spec.mode.value, "n_classes": spec.n_classes}
    if model_id == "compact_cnn":
        base.update(_compact_parameters(_compact_options(config)))
    elif model_id in {"inception_full", "inception_small", "inception_matrix"}:
        variant = str(config["variant"]) if model_id == "inception_matrix" else model_id.removeprefix("inception_")
        base.update(_inception_parameters(_inception_options(config, variant)))
    elif model_id in _ENSEMBLE_IDS:
        member = _inception_parameters(_inception_options(config, str(config.get("variant", "full"))))
        seeds = _normalise_seed_roster(config["member_seeds"], field="member_seeds")
        base.update({f"member_{name}": value for name, value in member.items()})
        base.update(member_count=len(seeds), member_seeds=seeds, probability_aggregation="arithmetic_mean")
    elif model_id in _BASELINE_DEFAULTS:
        base.update(_baseline_parameters(model_id, _baseline_options(model_id, config)))
    elif model_id == "shapeformer_legacy_effect_size_port":
        base.update(_legacy_parameters(config, spec))
    elif model_id in _SHAPEFORMER_IDS:
        base.update(_shapeformer_parameters(model_id, config, spec))
    elif model_id == "file_bag_fusion":
        signal = materialize_architecture_parameters(
            normalize_fusion_signal_encoder_config(config.pop("signal_encoder", None)), _fusion_raw_input_spec(spec)
        )
        pooling = str(config.pop("pooling", "mean"))
        if pooling not in {"mean", "attention"}:
            raise ValueError("pooling must be 'mean' or 'attention'")
        base.update(
            signal_feature_dim=_signal_feature_dim_from_architecture(signal),
            feature_hidden_dim=_normalise_positive_integer(
                config.pop("feature_hidden_dim", 32), field="feature_hidden_dim"
            ),
            fusion_hidden_dim=_normalise_positive_integer(
                config.pop("fusion_hidden_dim", 64), field="fusion_hidden_dim"
            ),
            pooling=pooling,
            fusion_dropout=_normalise_dropout(config.pop("dropout", 0.20)),
            signal_encoder=signal,
        )
        if config:
            raise ValueError(f"unknown file-bag fusion options: {sorted(config)}")
    elif model_id in {"fusion_compact", "fusion_inception"}:
        base.update(
            feature_hidden_dim=int(config["feature_hidden_dim"]),
            fusion_hidden_dim=int(config["fusion_hidden_dim"]),
            pooling=str(config["pooling"]),
            fusion_dropout=float(config["dropout"]),
        )
        if model_id == "fusion_compact":
            options = _compact_options(config, "signal_")
            base.update(signal_feature_dim=options["stage_channels"][-1], signal_encoder=_compact_parameters(options))
        else:
            options = _inception_options(config, str(config["signal_variant"]), "signal_")
            signal = _inception_parameters(options)
            base.update(signal_feature_dim=signal["out_channels"] * signal["branch_count"], signal_encoder=signal)
    else:
        raise ValueError(f"cannot materialise architecture for {model_id}")
    return _normalise_architecture_value(base)

def validate_source_architecture_annotation(
    declared: Any,
    derived: Mapping[str, Any],
) -> None:
    """Validate a source-side legacy annotation as a subset of derived truth."""

    if declared is None:
        return
    if not isinstance(declared, Mapping) or not declared:
        raise ValueError("architecture_parameters must be omitted or a non-empty legacy provenance mapping")
    normalized = _normalise_architecture_value(declared)
    expected = _normalise_architecture_value(derived)
    legacy_source_only = {
        "shape_channel_position_width",
        "shape_start_position_width",
        "shape_end_position_width",
    }

    def compare(source: Mapping[str, Any], target: Mapping[str, Any], prefix: str) -> None:
        for field, value in source.items():
            path = f"{prefix}.{field}" if prefix else str(field)
            if field not in target:
                if path in legacy_source_only:
                    continue
                raise ValueError(f"legacy architecture_parameters contains unknown derived field: {path}")
            target_value = target[field]
            if isinstance(value, Mapping) and isinstance(target_value, Mapping):
                compare(value, target_value, path)
                continue
            if (
                path == "generic_branch_input"
                and value == "canonical_frailty_raw_8"
                and target_value == "full_multivariate_input"
            ):
                continue
            if value != target_value:
                raise ValueError("legacy architecture_parameters derived field mismatch: " f"{path}")

    compare(normalized, expected, "")

def _resolved_source_architecture_provenance(
    model_config: Mapping[str, Any],
    input_spec: ModelInputSpec | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a legacy annotation, then rematerialize all runtime values."""

    spec = ModelInputSpec.from_value(input_spec)
    config = normalize_model_config(model_config)
    derived = materialize_architecture_parameters(config, spec)
    validate_source_architecture_annotation(config.get("architecture_parameters"), derived)
    for field in ("model_id", "representation_mode", "n_classes"):
        if derived.get(field) is None:
            raise ValueError(f"derived architecture_parameters is missing identity field {field}")
    return derived

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
    # Architecture manifests written before pool_size became configurable did
    # not name the default Inception max-pool branch width. Preserve those
    # manifests only for the unchanged default; any non-default still has to be
    # declared explicitly and therefore fails this equality check.
    model_id = str(actual.get("model_id", ""))
    if model_id in {"inception_full", "inception_small", "inception_matrix"}:
        expected.setdefault("pool_size", 3)
    elif model_id in {
        "inception_full_five_member_ensemble",
        "inception_matrix_five_member_ensemble",
    }:
        expected.setdefault("member_pool_size", 3)
    elif model_id == "fusion_inception" and isinstance(expected.get("signal_encoder"), dict):
        expected["signal_encoder"].setdefault("pool_size", 3)
    if expected != actual:
        keys = sorted(set(expected) | set(actual))
        differing = [key for key in keys if expected.get(key) != actual.get(key)]
        raise ValueError(
            "declared architecture_parameters differ from the constructed model; " f"differing keys={differing}"
        )
    return actual

def _finalize_model(
    model: Any,
    declared_architecture: Mapping[str, Any],
    input_spec: ModelInputSpec,
) -> Any:
    # Construction and provenance share the same normalized option resolvers;
    # storing that single result avoids a second model-specific validation tree.
    actual = _normalise_architecture_value(declared_architecture)
    model.resolved_architecture_parameters = actual
    payload = json.dumps(actual, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    model.resolved_architecture_hash = hashlib.sha256(payload).hexdigest()
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

@dataclass(frozen=True)
class _DerivedDiscoveryScope:
    """Bind a deterministic raw-window view to an already verified file bag."""

    source_scope: Any
    derived_dataset_hash: str

    @property
    def repeat(self) -> int:
        return int(self.source_scope.repeat)

    @property
    def fold(self) -> int:
        return int(self.source_scope.fold)

    def assert_training_dataset(self, dataset: Any, *, exact: bool = True) -> None:
        from ..training.trainer import dataset_binding_hash

        self.source_scope.assert_train_only(dataset.participant_ids, exact=exact)
        if dataset_binding_hash(dataset) != self.derived_dataset_hash:
            raise ValueError("derived fusion discovery dataset content changed")

def _flatten_file_bag_for_discovery(dataset: Any) -> Any:
    """Expand verified file bags to raw windows without copying file features.

    This adapter is used only for fold-local shapelet discovery.  Its input is
    the exact outer-training ``FileBagDataset`` already checked by the source
    split; consequently no OOF row can enter the derived raw dataset.
    """

    from ..training.datasets import FileBagDataset, RawWindowDataset, SampleIdentity

    if not isinstance(dataset, FileBagDataset):
        raise TypeError("fusion ShapeFormer discovery requires a FileBagDataset")
    values: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    identities: list[SampleIdentity] = []
    for file_index, (bag, bag_mask, identity) in enumerate(
        zip(dataset.window_bags, dataset.sample_masks, dataset.identities)
    ):
        for window_index in range(int(bag.shape[0])):
            values.append(np.asarray(bag[window_index], dtype=np.float32))
            masks.append(np.asarray(bag_mask[window_index], dtype=bool))
            identities.append(
                SampleIdentity(
                    participant_id=str(identity.participant_id),
                    file_id=str(identity.file_id),
                    role=str(identity.role),
                    label=int(identity.label),
                    signal_route=str(identity.signal_route),
                    quality_score=float(identity.quality_score),
                    retained=bool(identity.retained),
                    aggregation_retained=bool(identity.aggregation_retained),
                    window_id=(f"fusion_file_{file_index:06d}_window_{window_index:06d}"),
                )
            )
    return RawWindowDataset(
        np.stack(values, axis=0),
        identities,
        np.stack(masks, axis=0),
    )

def _prepared(
    config: Mapping[str, Any],
    spec: ModelInputSpec,
    split: Any,
    dataset_hash: str,
    **provenance: Any,
) -> PreparedModelFactory:
    base = {
        "model_id": str(config["model_id"]),
        "fold_local_preparation": "not_required",
        "outer_repeat_index": int(split.repeat),
        "outer_fold_index": int(split.fold),
        "outer_train_dataset_hash": dataset_hash,
    }
    base.update(provenance)
    return PreparedModelFactory(copy.deepcopy(config), spec, base)

def _prepare_fusion_factory(
    config: dict[str, Any],
    spec: ModelInputSpec,
    dataset: Any,
    split: Any,
    dataset_hash: str,
) -> PreparedModelFactory:
    if spec.mode is not RepresentationMode.FUSION:
        raise TypeError("file_bag_fusion preparation requires fusion representation")
    signal_config = normalize_fusion_signal_encoder_config(config.get("signal_encoder"))
    signal_id = str(signal_config["model_id"])
    if signal_id not in FUSION_SHAPEFORMER_ENCODER_IDS:
        return _prepared(config, spec, split, dataset_hash, signal_encoder_model_id=signal_id)
    forbidden = {
        "seed",
        "seed_policy",
        "outer_repeat_seed",
        "shapelets",
        "outer_repeat_index",
        "outer_fold_index",
        "outer_train_participant_hash",
    } & set(signal_config)
    if forbidden:
        raise ValueError(
            "file_bag_fusion owns signal-encoder seed/discovery state; " f"remove nested fields {sorted(forbidden)}"
        )
    from ..training.trainer import dataset_binding_hash

    raw_dataset = _flatten_file_bag_for_discovery(dataset)
    raw_hash = dataset_binding_hash(raw_dataset)
    nested = copy.deepcopy(signal_config)
    nested.update(
        seed=_normalise_seed(config.get("seed", 42), field="seed"),
        seed_policy="fixed_explicit",
    )
    prepared_signal = prepare_model_factory(
        nested,
        _fusion_raw_input_spec(spec),
        raw_dataset,
        _DerivedDiscoveryScope(split, raw_hash),
    )
    resolved = copy.deepcopy(config)
    resolved["signal_encoder"] = copy.deepcopy(dict(prepared_signal.resolved_model_config))
    resolved["architecture_parameters"] = materialize_architecture_parameters(resolved, spec)
    return PreparedModelFactory(
        resolved,
        spec,
        {
            "model_id": "file_bag_fusion",
            "canonical_model_name": str(config["canonical_model_name"]),
            "registry_role": "optional_composable_fusion",
            "fold_local_preparation": "signal_encoder_shapelet_discovery",
            "signal_encoder_model_id": signal_id,
            "signal_encoder_provenance": dict(prepared_signal.provenance),
            "discovery_bank_hash": prepared_signal.provenance["discovery_bank_hash"],
            "outer_repeat_index": int(split.repeat),
            "outer_fold_index": int(split.fold),
            "outer_train_dataset_hash": dataset_hash,
            "derived_raw_discovery_dataset_hash": raw_hash,
            "file_features_used_for_discovery": False,
            "fallback_used": False,
        },
    )

def _discovery_rows(dataset: Any) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    identities = dataset.identities
    return (
        tuple(str(item.participant_id) for item in identities),
        tuple(str(item.file_id) for item in identities),
        tuple(str(item.window_id) for item in identities),
    )

def _prepare_legacy_discovery(
    config: dict[str, Any],
    spec: ModelInputSpec,
    dataset: Any,
    split: Any,
    dataset_hash: str,
    sample_mask: np.ndarray,
) -> PreparedModelFactory:
    from .shapeformer_legacy import (
        LEGACY_DISCOVERY_BALANCE,
        LEGACY_EFFECT_SIZE_DISCOVERY_METHOD,
        discover_legacy_effect_size_shapelets,
    )

    allowed = {
        "model_id",
        "canonical_model_name",
        "architecture_parameters",
        "seed",
        "seed_policy",
        "outer_repeat_seed",
        "discovery_method",
        "discovery_balance",
        "input_fs_hz",
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
        "dropout",
        "shapelet_search_window_samples",
        "complexity_norm",
        "max_complexity_ratio",
    }
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"unknown legacy effect-size ShapeFormer options: {unknown}")
    if not bool(sample_mask.all()):
        raise ValueError("legacy effect-size ShapeFormer requires complete, unpadded windows")
    method = str(config.pop("discovery_method", LEGACY_EFFECT_SIZE_DISCOVERY_METHOD))
    balance = str(config.pop("discovery_balance", LEGACY_DISCOVERY_BALANCE))
    if method != LEGACY_EFFECT_SIZE_DISCOVERY_METHOD:
        raise ValueError("legacy effect-size ShapeFormer discovery_method drifted")
    if balance != LEGACY_DISCOVERY_BALANCE:
        raise ValueError("legacy effect-size ShapeFormer discovery_balance drifted")
    values: dict[str, Any] = {}
    for name in (
        "sequence_length_samples",
        "shapelet_length_samples",
        "discovery_stride_samples",
        "shapelets_per_class",
        "max_discovery_windows",
        "candidates_per_class_channel",
    ):
        values[name] = _normalise_positive_integer(
            config.pop(name, SHAPEFORMER_LEGACY_EFFECT_SIZE_DEFAULTS[name]), field=name
        )
    if values["sequence_length_samples"] != int(dataset.values.shape[-1]):
        raise ValueError("legacy effect-size ShapeFormer sequence_length_samples does not match outer-train windows")
    input_fs_hz = _normalise_finite_positive(
        config.pop("input_fs_hz", SHAPEFORMER_LEGACY_EFFECT_SIZE_DEFAULTS["input_fs_hz"]),
        field="input_fs_hz",
    )
    seed = _normalise_seed(config.get("seed", 42), field="seed")
    config["seed"] = seed
    participants, files, windows = _discovery_rows(dataset)
    bank = discover_legacy_effect_size_shapelets(
        np.asarray(dataset.values, dtype=np.float32),
        np.asarray(dataset.labels, dtype=np.int64),
        participants,
        files,
        windows,
        discovery_method=method,
        input_fs_hz=input_fs_hz,
        **values,
        outer_repeat_index=int(split.repeat),
        outer_fold_index=int(split.fold),
        seed=seed,
    )
    resolved = copy.deepcopy(config)
    resolved.update(
        model_id="shapeformer_legacy_effect_size_port",
        shapelets=bank,
        discovery_method=method,
        discovery_balance=balance,
        input_fs_hz=input_fs_hz,
        **values,
        outer_repeat_index=int(split.repeat),
        outer_fold_index=int(split.fold),
        outer_train_participant_hash=bank.outer_train_participant_hash,
    )
    resolved["architecture_parameters"] = materialize_architecture_parameters(resolved, spec)
    provenance = {
        "model_id": resolved["model_id"],
        "canonical_model_name": str(config["canonical_model_name"]),
        "registry_role": "ablation",
        "scientific_status": "legacy_parallel_ablation_not_osd_parity",
        "discovery_method": method,
        "discovery_balance": balance,
        "discovery_bank_hash": _shapelet_bank_hash(bank),
        "shapelet_count": int(bank.count),
        **values,
        "discovery_window_count": int(bank.discovery_indices.size),
        "enumerated_candidate_count": int(bank.enumerated_candidate_count),
        "retained_candidate_count": int(bank.retained_candidate_count),
        "discovery_selection_hash": bank.discovery_selection_hash,
        "candidate_records": bank.candidate_records(),
        "input_fs_hz": input_fs_hz,
        "outer_repeat_index": int(split.repeat),
        "outer_fold_index": int(split.fold),
        "outer_train_participant_hash": bank.outer_train_participant_hash,
        "outer_train_dataset_hash": dataset_hash,
        "fallback_used": False,
    }
    return PreparedModelFactory(resolved, spec, provenance)

def _prepare_shapeformer_discovery(
    config: dict[str, Any],
    spec: ModelInputSpec,
    dataset: Any,
    split: Any,
    dataset_hash: str,
    sequence_lengths: np.ndarray,
) -> PreparedModelFactory:
    model_id = str(config["model_id"])
    channel_specific = model_id != "shapeformer_effect_size_fixed_v1"
    method = "channel_specific_osd" if channel_specific else "effect_size_fixed_v1"
    selected_method = str(config.pop("discovery_method", method))
    if selected_method != method:
        raise ValueError(f"{model_id} requires explicit discovery_method={method}; no discovery fallback is allowed")
    defaults = SHAPEFORMER_REFERENCE_NUMERIC_DEFAULTS if channel_specific else SHAPEFORMER_EXPERIMENTAL_NUMERIC_DEFAULTS
    input_fs_hz = _normalise_finite_positive(config.pop("input_fs_hz", defaults["input_fs_hz"]), field="input_fs_hz")
    seed = _normalise_seed(config.get("seed", 42), field="seed")
    config["seed"] = seed
    participants, files, windows = _discovery_rows(dataset)
    base = {
        "x_train": np.asarray(dataset.values, dtype=np.float32),
        "y_train": np.asarray(dataset.labels, dtype=np.int64),
        "participant_ids": participants,
        "discovery_method": method,
        "input_fs_hz": input_fs_hz,
        "outer_repeat_index": int(split.repeat),
        "outer_fold_index": int(split.fold),
        "seed": seed,
    }
    if channel_specific:
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

        forbidden = {
            "shapelet_length_samples",
            "shapelet_length",
            "candidate_stride",
            "discovery_stride",
            "discovery_stride_samples",
            "num_pips",
            "max_candidates_per_class",
        } & set(config)
        if forbidden:
            raise ValueError(
                "channel_specific_osd has no fixed length/stride/count fallback controls: " f"{sorted(forbidden)}"
            )
        if not spec.channel_schema or len(spec.channel_schema) != spec.n_channels:
            raise ValueError("channel_specific_osd requires the exact ordered channel_schema")
        rules = {
            "pip_rounding_rule": PIP_ROUNDING_RULE,
            "pip_selection_rule": PIP_SELECTION_RULE,
            "candidate_generation_rule": CANDIDATE_GENERATION_RULE,
            "candidate_enumeration_rule": CANDIDATE_ENUMERATION_RULE,
            "candidate_ranking_rule": CANDIDATE_RANKING_RULE,
            "selected_bank_order_rule": SELECTED_BANK_ORDER_RULE,
            "discovery_position_search_boundary_rule": DISCOVERY_POSITION_SEARCH_BOUNDARY_RULE,
            "information_gain_split_rule": INFORMATION_GAIN_SPLIT_RULE,
        }
        for name, expected in rules.items():
            if str(config.pop(name, expected)) != expected:
                raise ValueError(f"channel_specific_osd {name} drifted")
        bank = discover_pisd_shapelets(
            **base,
            file_ids=files,
            window_ids=windows,
            channel_schema=spec.channel_schema,
            sequence_lengths=sequence_lengths,
            num_pip_ratio=_normalise_positive_fraction(
                config.pop("num_pip_ratio", defaults["num_pip_ratio"]), field="num_pip_ratio"
            ),
            shapelets_per_class=_normalise_positive_integer(
                config.pop("shapelets_per_class", defaults["shapelets_per_class"]), field="shapelets_per_class"
            ),
            max_discovery_windows=_normalise_positive_integer(
                config.pop("max_discovery_windows", defaults["max_discovery_windows"]), field="max_discovery_windows"
            ),
            discovery_balance=str(config.pop("discovery_balance", "participant_file_balanced")),
            position_search_neighbourhood_samples=_normalise_positive_integer(
                config.pop("position_search_neighbourhood_samples", defaults["position_search_neighbourhood_samples"]),
                field="position_search_neighbourhood_samples",
            ),
            distance_position_chunk_size=_normalise_positive_integer(
                config.get("distance_position_chunk_size", defaults["distance_position_chunk_size"]),
                field="distance_position_chunk_size",
            ),
        )
    else:
        from .shapeformer import discover_effect_size_shapelets

        bank = discover_effect_size_shapelets(
            **base,
            shapelet_length=int(config.pop("shapelet_length_samples", 128)),
            shapelets_per_class=_normalise_positive_integer(
                config.pop("shapelets_per_class", defaults["shapelets_per_class"]), field="shapelets_per_class"
            ),
            stride=int(config.pop("discovery_stride_samples", 64)),
            max_candidates_per_class=_normalise_positive_integer(
                config.pop("max_candidates_per_class", defaults["max_candidates_per_class"]),
                field="max_candidates_per_class",
            ),
        )
    resolved = copy.deepcopy(config)
    resolved.update(
        model_id=model_id,
        shapelets=bank,
        discovery_method=method,
        input_fs_hz=input_fs_hz,
        outer_repeat_index=int(split.repeat),
        outer_fold_index=int(split.fold),
        outer_train_participant_hash=bank.outer_train_participant_hash,
    )
    if hasattr(bank, "position_search_neighbourhood_samples"):
        for name in (
            "position_search_neighbourhood_samples",
            "information_gain_split_rule",
            "pip_rounding_rule",
            "pip_selection_rule",
            "candidate_generation_rule",
            "candidate_enumeration_rule",
            "candidate_ranking_rule",
            "selected_bank_order_rule",
            "discovery_position_search_boundary_rule",
        ):
            resolved[name] = getattr(bank, name)
    resolved["architecture_parameters"] = materialize_architecture_parameters(resolved, spec)
    provenance = {
        "model_id": model_id,
        "canonical_model_name": str(config["canonical_model_name"]),
        "registry_role": "reference" if model_id == "shapeformer_channel_specific_osd" else "ablation",
        "discovery_method": method,
        "discovery_bank_hash": _shapelet_bank_hash(bank),
        "shapelet_count": int(bank.count if hasattr(bank, "count") else bank.values.shape[0]),
        "shapelet_length_samples": bank.shapelet_length_samples,
        "input_fs_hz": float(bank.input_fs_hz),
        "outer_repeat_index": int(bank.outer_repeat_index),
        "outer_fold_index": int(bank.outer_fold_index),
        "outer_train_participant_hash": bank.outer_train_participant_hash,
        "outer_train_dataset_hash": dataset_hash,
        "fallback_used": False,
    }
    if hasattr(bank, "candidate_records"):
        provenance.update(
            candidate_records=bank.candidate_records(),
            num_pip_ratio=float(bank.num_pip_ratio),
            pip_count_rounding=bank.pip_rounding_rule,
            pip_selection_rule=bank.pip_selection_rule,
            candidate_generation_rule=bank.candidate_generation_rule,
            candidate_enumeration_rule=bank.candidate_enumeration_rule,
            candidate_ranking_rule=bank.candidate_ranking_rule,
            selected_bank_order_rule=bank.selected_bank_order_rule,
            discovery_position_search_boundary_rule=bank.discovery_position_search_boundary_rule,
            information_gain_split_rule=bank.information_gain_split_rule,
            candidate_stride=None,
            best_fit_search_stride=1,
            position_search_neighbourhood_samples=int(bank.position_search_neighbourhood_samples),
            discovery_window_count=int(bank.discovery_window_count),
            discovery_selection_hash=bank.discovery_selection_hash,
            discovery_balance=bank.discovery_balance,
        )
    return PreparedModelFactory(resolved, spec, provenance)

def prepare_model_factory(
    model_config: Mapping[str, Any],
    input_spec: ModelInputSpec | Mapping[str, Any],
    outer_train_dataset: Any,
    frozen_split: Any,
) -> PreparedModelFactory:
    """Bind optional shapelet discovery to the verified outer-train fold."""

    from ..training.datasets import RawWindowDataset
    from ..training.trainer import dataset_binding_hash

    spec = ModelInputSpec.from_value(input_spec)
    _validate_frailty_factory_input(spec, require_explicit_schema=True)
    frozen_split.assert_training_dataset(outer_train_dataset, exact=True)
    dataset_hash = dataset_binding_hash(outer_train_dataset)
    config = normalize_model_config(model_config)
    config["architecture_parameters"] = _resolved_source_architecture_provenance(config, spec)
    model_id = str(config["model_id"])
    if model_id == "file_bag_fusion":
        return _prepare_fusion_factory(config, spec, outer_train_dataset, frozen_split, dataset_hash)
    if model_id not in _SHAPEFORMER_IDS:
        return _prepared(config, spec, frozen_split, dataset_hash)
    if spec.mode is not RepresentationMode.RAW or not isinstance(outer_train_dataset, RawWindowDataset):
        raise TypeError("ShapeFormer discovery requires a materialised RawWindowDataset")
    sample_mask = np.asarray(outer_train_dataset.sample_mask, dtype=bool)
    sequence_lengths = sample_mask.sum(axis=1).astype(np.int64)
    prefix_mask = np.arange(sample_mask.shape[1])[None, :] < sequence_lengths[:, None]
    if not np.array_equal(sample_mask, prefix_mask):
        raise ValueError("ShapeFormer discovery requires a contiguous valid prefix per window")
    forbidden = {"shapelets", "outer_repeat_index", "outer_fold_index", "outer_train_participant_hash"} & set(config)
    if forbidden:
        raise ValueError(
            "fold-local ShapeFormer preparation derives these fields and rejects supplied "
            f"values: {sorted(forbidden)}"
        )
    if model_id == "shapeformer_legacy_effect_size_port":
        return _prepare_legacy_discovery(config, spec, outer_train_dataset, frozen_split, dataset_hash, sample_mask)
    return _prepare_shapeformer_discovery(
        config, spec, outer_train_dataset, frozen_split, dataset_hash, sequence_lengths
    )

def _require(config: Mapping[str, Any], names: set[str], context: str) -> None:
    missing = sorted(names - set(config))
    if missing:
        raise ValueError(f"{context} missing required options: {missing}")

def _reject_unknown(config: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"unknown {context} options: {unknown}")

def _set_model_identity(
    model: Any,
    model_id: str,
    canonical_name: str,
    seed_policy: str | None,
    seeds: tuple[int, ...],
    architecture: Mapping[str, Any],
    spec: ModelInputSpec,
) -> Any:
    model.model_id = model_id
    model.canonical_model_name = canonical_name
    model.seed_policy = str(seed_policy or "fixed_explicit")
    model.training_seeds = seeds
    return _finalize_model(model, architecture, spec)

def _mapping_bank(value: Any, model_id: str) -> Any:
    """Restore a discovery bank serialized in a model bundle."""

    if not isinstance(value, Mapping):
        return value
    payload = dict(value)
    if model_id == "shapeformer_legacy_effect_size_port":
        from .shapeformer_legacy import LegacyEffectSizeShapelets

        payload["values"] = tuple(np.asarray(item, dtype=np.float32) for item in payload["values"])
        arrays = (
            "source_sample_indices",
            "source_starts",
            "source_ends",
            "source_scores",
            "source_weights",
            "source_classes",
            "source_channels",
            "discovery_indices",
        )
        tuples = (
            "source_participant_ids",
            "source_file_ids",
            "source_window_ids",
            "discovery_participant_ids",
            "discovery_file_ids",
            "discovery_window_ids",
            "fitted_participant_ids",
        )
        cls = LegacyEffectSizeShapelets
    elif model_id == "shapeformer_effect_size_fixed_v1":
        from .shapeformer import EffectSizeShapelets

        payload["values"] = np.asarray(payload["values"], dtype=np.float32)
        arrays, tuples, cls = (
            ("source_classes", "effect_sizes"),
            ("fitted_participant_ids",),
            EffectSizeShapelets,
        )
    else:
        from .pisd_port import PisdShapelets

        payload["values"] = tuple(np.asarray(item, dtype=np.float32) for item in payload["values"])
        arrays = (
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
        )
        tuples = (
            "source_channel_names",
            "source_participant_ids",
            "source_file_ids",
            "source_window_ids",
            "fitted_participant_ids",
            "discovery_participant_ids",
            "discovery_file_ids",
            "discovery_window_ids",
            "channel_schema",
        )
        cls = PisdShapelets
    for name in arrays:
        payload[name] = np.asarray(payload[name])
    for name in tuples:
        payload[name] = tuple(payload[name])
    return cls(**payload)

def _check_bank_binding(
    bank: Any,
    *,
    method: str,
    input_fs_hz: float,
    repeat: int,
    fold: int,
    participant_hash: str,
) -> None:
    if bank.discovery_method != method:
        raise ValueError("model discovery_method does not match fitted shapelet bank")
    if not np.isclose(bank.input_fs_hz, input_fs_hz, rtol=0.0, atol=1e-12):
        raise ValueError("model input_fs_hz does not match fitted shapelet bank")
    if bank.outer_repeat_index != repeat or bank.outer_fold_index != fold:
        raise ValueError("model outer repeat/fold does not match fitted shapelet bank")
    if bank.outer_train_participant_hash != participant_hash:
        raise ValueError("model outer-train roster hash does not match fitted shapelet bank")

def _create_shapeformer(
    model_id: str,
    config: Mapping[str, Any],
    spec: ModelInputSpec,
    seed: int,
) -> Any:
    from .pisd_port import PISD_DISCOVERY_METHOD, PisdShapelets
    from .shapeformer import EffectSizeShapelets, ExperimentalShapeFormer

    common = {
        "shapelets",
        "discovery_method",
        "input_fs_hz",
        "outer_repeat_index",
        "outer_fold_index",
        "outer_train_participant_hash",
    }
    rule_names = {
        "pip_rounding_rule",
        "pip_selection_rule",
        "candidate_generation_rule",
        "candidate_enumeration_rule",
        "candidate_ranking_rule",
        "selected_bank_order_rule",
        "discovery_position_search_boundary_rule",
        "information_gain_split_rule",
    }
    channel_specific = model_id != "shapeformer_effect_size_fixed_v1"
    _require(config, common | (rule_names if channel_specific else set()), model_id)
    method = str(config["discovery_method"])
    expected_method = PISD_DISCOVERY_METHOD if channel_specific else "effect_size_fixed_v1"
    if method != expected_method:
        raise ValueError(f"{model_id} requires discovery_method={expected_method}; never fall back")
    bank = _mapping_bank(config["shapelets"], model_id)
    expected_type = PisdShapelets if channel_specific else EffectSizeShapelets
    if not isinstance(bank, expected_type):
        raise TypeError(f"{model_id} received the wrong discovery-bank type")
    input_fs_hz = _normalise_finite_positive(config["input_fs_hz"], field="input_fs_hz")
    _check_bank_binding(
        bank,
        method=method,
        input_fs_hz=input_fs_hz,
        repeat=int(config["outer_repeat_index"]),
        fold=int(config["outer_fold_index"]),
        participant_hash=str(config["outer_train_participant_hash"]),
    )
    position = None
    if channel_specific:
        from .pisd_port import (
            CANDIDATE_ENUMERATION_RULE,
            CANDIDATE_GENERATION_RULE,
            CANDIDATE_RANKING_RULE,
            DISCOVERY_POSITION_SEARCH_BOUNDARY_RULE,
            INFORMATION_GAIN_SPLIT_RULE,
            PIP_ROUNDING_RULE,
            PIP_SELECTION_RULE,
            SELECTED_BANK_ORDER_RULE,
        )

        expected_rules = {
            "pip_rounding_rule": PIP_ROUNDING_RULE,
            "pip_selection_rule": PIP_SELECTION_RULE,
            "candidate_generation_rule": CANDIDATE_GENERATION_RULE,
            "candidate_enumeration_rule": CANDIDATE_ENUMERATION_RULE,
            "candidate_ranking_rule": CANDIDATE_RANKING_RULE,
            "selected_bank_order_rule": SELECTED_BANK_ORDER_RULE,
            "discovery_position_search_boundary_rule": DISCOVERY_POSITION_SEARCH_BOUNDARY_RULE,
            "information_gain_split_rule": INFORMATION_GAIN_SPLIT_RULE,
        }
        for name, expected in expected_rules.items():
            if str(config[name]) != expected or str(getattr(bank, name)) != expected:
                raise ValueError(f"model {name} does not match the upstream PISD bank")
        position = _normalise_positive_integer(
            config.get("position_search_neighbourhood_samples", bank.position_search_neighbourhood_samples),
            field="position_search_neighbourhood_samples",
        )
        if position != int(bank.position_search_neighbourhood_samples):
            raise ValueError("model position_search_neighbourhood_samples does not match the upstream PISD bank")

    _torch_seed(seed)
    architecture = _shapeformer_parameters(model_id, config, spec)
    if model_id == "shapeformer_channel_specific_osd":
        from .shapeformer_literature import LiteratureShapeFormerChannelSpecificOSD

        keys = (
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
        )
        model = LiteratureShapeFormerChannelSpecificOSD(
            n_channels=spec.n_channels,
            n_classes=spec.n_classes,
            sequence_length=architecture["sequence_length_samples"],
            shapelets=bank,
            **{name: architecture[name] for name in keys},
            position_search_neighbourhood_samples=position,
            input_fs_hz=input_fs_hz,
        )
    else:
        model = ExperimentalShapeFormer(
            spec.n_channels,
            spec.n_classes,
            bank,
            hidden_channels=architecture["hidden_channels"],
            dropout=architecture["classifier_dropout"],
            patch_size_samples=architecture["patch_size_samples"],
            attention_heads=architecture["attention_heads"],
            attention_layers=architecture["attention_layers"],
            attention_feedforward_channels=architecture["attention_feedforward_channels"],
            distance_position_chunk_size=architecture["distance_position_chunk_size"],
            input_fs_hz=input_fs_hz,
        )
    model.registry_role = "reference" if model_id == "shapeformer_channel_specific_osd" else "ablation"
    return model

def _create_legacy_shapeformer(config: Mapping[str, Any], spec: ModelInputSpec, seed: int) -> Any:
    from .shapeformer_legacy import (
        LEGACY_DISCOVERY_BALANCE,
        LEGACY_EFFECT_SIZE_DISCOVERY_METHOD,
        LegacyEffectSizeShapeFormer,
        LegacyEffectSizeShapelets,
    )

    required = {
        "shapelets",
        "discovery_method",
        "discovery_balance",
        "input_fs_hz",
        "sequence_length_samples",
        "shapelet_length_samples",
        "discovery_stride_samples",
        "shapelets_per_class",
        "max_discovery_windows",
        "candidates_per_class_channel",
        "outer_repeat_index",
        "outer_fold_index",
        "outer_train_participant_hash",
    }
    _require(config, required, "legacy effect-size ShapeFormer")
    bank = _mapping_bank(config["shapelets"], "shapeformer_legacy_effect_size_port")
    if not isinstance(bank, LegacyEffectSizeShapelets):
        raise TypeError("legacy effect-size ShapeFormer received the wrong discovery bank")
    method, balance = str(config["discovery_method"]), str(config["discovery_balance"])
    if method != LEGACY_EFFECT_SIZE_DISCOVERY_METHOD or bank.discovery_method != method:
        raise ValueError("legacy effect-size discovery method drifted")
    if balance != LEGACY_DISCOVERY_BALANCE or bank.discovery_balance != balance:
        raise ValueError("legacy effect-size discovery balance drifted")
    input_fs_hz = _normalise_finite_positive(config["input_fs_hz"], field="input_fs_hz")
    _check_bank_binding(
        bank,
        method=method,
        input_fs_hz=input_fs_hz,
        repeat=int(config["outer_repeat_index"]),
        fold=int(config["outer_fold_index"]),
        participant_hash=str(config["outer_train_participant_hash"]),
    )
    for name in (
        "sequence_length_samples",
        "shapelet_length_samples",
        "discovery_stride_samples",
        "shapelets_per_class",
        "max_discovery_windows",
        "candidates_per_class_channel",
    ):
        if _normalise_positive_integer(config[name], field=name) != int(getattr(bank, name)):
            raise ValueError(f"legacy effect-size {name} differs from its bank")
    architecture = _legacy_parameters(config, spec)
    _torch_seed(seed)
    keys = (
        "local_kernel_width_samples",
        "local_embedding_channels",
        "shape_embedding_channels",
        "attention_feedforward_channels",
        "attention_heads",
        "dropout",
        "shapelet_search_window_samples",
        "complexity_norm",
        "max_complexity_ratio",
    )
    model = LegacyEffectSizeShapeFormer(
        spec.n_channels,
        spec.n_classes,
        bank,
        sequence_length_samples=architecture["sequence_length_samples"],
        **{name: architecture[name] for name in keys},
        input_fs_hz=input_fs_hz,
    )
    model.registry_role = "ablation"
    model.scientific_status = "legacy_parallel_ablation_not_osd_parity"
    return model

def _fusion_model(signal: Any, spec: ModelInputSpec, options: Mapping[str, Any]) -> Any:
    from .fusion import FileBagFusionClassifier

    return FileBagFusionClassifier(
        signal,
        int(signal.feature_dim),
        spec.n_file_features,
        spec.n_classes,
        feature_hidden_dim=int(options["feature_hidden_dim"]),
        fusion_hidden_dim=int(options["fusion_hidden_dim"]),
        pooling=str(options["pooling"]),
        dropout=float(options["dropout"]),
    )

def create_model(model_config: Mapping[str, Any], input_spec: ModelInputSpec | Mapping[str, Any]) -> Any:
    """Create one unfitted model from normalized, shared parameter tables."""

    spec = ModelInputSpec.from_value(input_spec)
    _validate_frailty_factory_input(spec, require_explicit_schema=False)
    config = normalize_model_config(model_config)
    architecture = _resolved_source_architecture_provenance(config, spec)
    model_id = str(config.pop("model_id"))
    canonical = str(config.pop("canonical_model_name"))
    config.pop("architecture_parameters", None)
    seed_policy = config.pop("seed_policy", None)
    outer_seed = config.pop("outer_repeat_seed", None)
    config.pop("member_seed_roster_id", None)
    if model_id in _ENSEMBLE_IDS:
        if "seed" in config:
            raise ValueError("ensemble has no single seed; use member_seeds")
        seed = 0
    else:
        seed = resolve_seed_policy(
            seed_policy,
            seed=_normalise_seed(config.pop("seed", 42), field="seed"),
            outer_repeat_seed=outer_seed,
        )[0]

    if model_id in _BASELINE_DEFAULTS:
        if spec.mode is not RepresentationMode.FEATURE_VECTOR:
            raise ValueError("feature baselines require feature_vector representation")
        options = _baseline_options(model_id, config)
        _reject_unknown(config, set(options), "feature baseline")
        model = FeatureVectorBaseline(model_id, spec.feature_names, seed=seed, **options)
        return _set_model_identity(model, model_id, canonical, seed_policy, (seed,), architecture, spec)

    if model_id == "compact_cnn":
        if spec.mode is not RepresentationMode.RAW or spec.n_channels <= 0:
            raise ValueError("CompactCNN1D requires raw representation and positive n_channels")
        from .compact_cnn import CompactCNN1D

        options = _compact_options(config)
        _reject_unknown(config, set(options), "CompactCNN")
        _torch_seed(seed)
        model = CompactCNN1D(spec.n_channels, spec.n_classes, **options)
        return _set_model_identity(model, model_id, canonical, seed_policy, (seed,), architecture, spec)

    if model_id in {"inception_full", "inception_small", "inception_matrix"}:
        expected = RepresentationMode.FEATURE_MATRIX if model_id == "inception_matrix" else RepresentationMode.RAW
        if spec.mode is not expected or spec.n_channels <= 0:
            raise ValueError(f"{model_id} requires {expected.value} representation and positive channels")
        from .inception import InceptionTimeSingleNetwork

        variant = (
            str(config.get("variant", "full"))
            if model_id == "inception_matrix"
            else model_id.removeprefix("inception_")
        )
        options = _inception_options(config, variant)
        allowed = set(options) | ({"variant"} if model_id == "inception_matrix" else set())
        _reject_unknown(config, allowed, "InceptionTime")
        _torch_seed(seed)
        model = InceptionTimeSingleNetwork(spec.n_channels, spec.n_classes, **options)
        return _set_model_identity(model, model_id, canonical, seed_policy, (seed,), architecture, spec)

    if model_id in _ENSEMBLE_IDS:
        from .inception import InceptionTimeFiveMemberProbabilityEnsemble, InceptionTimeSingleNetwork

        comparison_only = config.pop("comparison_only", None)
        if comparison_only is not None and not isinstance(comparison_only, bool):
            raise ValueError("comparison_only is optional metadata and must be boolean")
        expected = (
            RepresentationMode.RAW if model_id.startswith("inception_full") else RepresentationMode.FEATURE_MATRIX
        )
        if spec.mode is not expected or spec.n_channels <= 0:
            raise ValueError(f"{model_id} requires {expected.value} representation and positive channels")
        _require(config, {"member_seeds"}, "ensemble")
        seeds = resolve_seed_policy(
            seed_policy or "member_roster",
            outer_repeat_seed=outer_seed,
            member_seeds=config["member_seeds"],
        )
        options = _inception_options(config, str(config.get("variant", "full")))
        _reject_unknown(config, set(options) | {"member_seeds"}, "ensemble")
        members = []
        for member_seed in seeds:
            _torch_seed(member_seed)
            members.append(InceptionTimeSingleNetwork(spec.n_channels, spec.n_classes, **options))
        model = InceptionTimeFiveMemberProbabilityEnsemble(members, seeds)
        model.base_model_id = "inception_full" if model_id.startswith("inception_full") else "inception_matrix"
        model.representation_mode = expected.value
        return _set_model_identity(
            model, model_id, canonical, seed_policy or "member_roster", seeds, architecture, spec
        )

    if model_id in _SHAPEFORMER_IDS:
        if spec.mode is not RepresentationMode.RAW or spec.n_channels <= 0:
            raise ValueError("ShapeFormer candidates require raw representation and positive channels")
        model = (
            _create_legacy_shapeformer(config, spec, seed)
            if model_id == "shapeformer_legacy_effect_size_port"
            else _create_shapeformer(model_id, config, spec, seed)
        )
        return _set_model_identity(model, model_id, canonical, seed_policy, (seed,), architecture, spec)

    if model_id == "file_bag_fusion":
        if spec.mode is not RepresentationMode.FUSION or min(spec.n_channels, spec.n_file_features) <= 0:
            raise ValueError("file_bag_fusion requires fusion representation and positive dimensions")
        signal_config = normalize_fusion_signal_encoder_config(config.get("signal_encoder"))
        nested_seed = signal_config.get("seed")
        if nested_seed is not None and _normalise_seed(nested_seed, field="signal_encoder.seed") != seed:
            raise ValueError("signal_encoder.seed must equal the outer file-bag fusion seed")
        if signal_config.get("seed_policy") not in {None, "fixed", "fixed_explicit"}:
            raise ValueError("signal_encoder.seed_policy must be fixed_explicit")
        signal_config.update(seed=seed, seed_policy="fixed_explicit")
        signal = create_model(signal_config, _fusion_raw_input_spec(spec))
        if not hasattr(signal, "feature_dim") or not hasattr(signal, "forward_features"):
            raise TypeError("configured file-bag signal encoder lacks forward_features/feature_dim")
        options = {
            "feature_hidden_dim": _normalise_positive_integer(
                config.get("feature_hidden_dim", 32), field="feature_hidden_dim"
            ),
            "fusion_hidden_dim": _normalise_positive_integer(
                config.get("fusion_hidden_dim", 64), field="fusion_hidden_dim"
            ),
            "pooling": str(config.get("pooling", "mean")),
            "dropout": _normalise_dropout(config.get("dropout", 0.20)),
        }
        _reject_unknown(config, set(options) | {"signal_encoder"}, "file-bag fusion")
        model = _fusion_model(signal, spec, options)
        return _set_model_identity(model, model_id, canonical, seed_policy, (seed,), architecture, spec)

    if model_id in {"fusion_compact", "fusion_inception"}:
        if spec.mode is not RepresentationMode.FUSION or min(spec.n_channels, spec.n_file_features) <= 0:
            raise ValueError("file-bag fusion requires fusion representation and positive dimensions")
        from .compact_cnn import CompactCNN1D
        from .inception import InceptionTimeSingleNetwork

        required = {"feature_hidden_dim", "fusion_hidden_dim", "pooling", "dropout"}
        if model_id == "fusion_compact":
            required |= {"signal_dropout", "signal_kernel_sizes", "signal_dilations", "signal_pool_sizes"}
            signal_options = _compact_options(config, "signal_")
            signal_type = CompactCNN1D
        else:
            required |= {"signal_variant", "signal_dropout", "signal_kernel_sizes", "signal_dilation"}
            signal_options = _inception_options(config, str(config.get("signal_variant", "full")), "signal_")
            signal_type = InceptionTimeSingleNetwork
        _require(config, required, model_id)
        allowed_signal = {f"signal_{name}" for name in signal_options}
        _reject_unknown(config, required | allowed_signal, "fusion")
        _torch_seed(seed)
        signal = signal_type(spec.n_channels, spec.n_classes, **signal_options)
        model = _fusion_model(signal, spec, config)
        return _set_model_identity(model, model_id, canonical, seed_policy, (seed,), architecture, spec)

    raise ValueError(f"unsupported model_id: {model_id}")


# Backward-compatible descriptive alias used by internal configuration code.
# 内部配置代码使用的描述性兼容别名。
build_model = create_model
