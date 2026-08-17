"""Strict model factory spanning all four representation modes.

覆盖四种 representation mode 的严格模型工厂。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..contracts import RepresentationMode
from .feature_baselines import FeatureVectorBaseline
from .rocket import MiniRocketAblation, RocketRidgeClassifier


# English: PyTorch was found locally, but V1 dependency approval remains a human
# decision. Runtime availability must not be confused with protocol approval.
# 中文：本地已发现 PyTorch，但 V1 依赖批准仍需人工决策；运行时可用不等于协议获批。
PYTORCH_DEPENDENCY_STATUS = "decision_pending"


# English: Human specification names are mapped once to immutable machine IDs.
# Both are persisted in manifests; aliases outside this table are forbidden.
# 中文：规范中的人类可读名称只在此处映射一次到稳定 machine ID。manifest 同时
# 保存两者；本表之外的别名一律禁止。
CANONICAL_MODEL_REGISTRY: dict[str, str] = {
    "CompactCNN1D": "compact_cnn",
    "InceptionTimeFull": "inception_full",
    "InceptionTimeSmall": "inception_small",
    "InceptionTimeMatrix": "inception_matrix",
    "InceptionTimeFiveMemberEnsemble": "inception_five_member_ensemble",
    "ROCKET": "rocket_numpy",
    "MiniROCKET": "minirocket_ablation",
    "LogisticRegressionL2": "logistic_regression",
    "RBFSVM": "rbf_svm",
    "ExtraTrees": "extra_trees",
    "ShapeFormerEffectSize": "shapeformer_effect_size",
    "FileBagFusionCompact": "fusion_compact",
    "FileBagFusionInception": "fusion_inception",
}
_MACHINE_TO_CANONICAL = {machine: canonical for canonical, machine in CANONICAL_MODEL_REGISTRY.items()}


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


def _torch_seed(seed: int) -> None:
    """Seed PyTorch only when a deep model is requested / 仅在请求深度模型时设种子。"""

    import torch

    torch.manual_seed(int(seed))


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
    seed = int(config.pop("seed", 42))
    spec = ModelInputSpec.from_value(input_spec)
    mode = spec.mode

    feature_ids = {"logistic_regression", "rbf_svm", "extra_trees"}
    if model_id in feature_ids:
        if mode is not RepresentationMode.FEATURE_VECTOR:
            raise ValueError("feature baselines require feature_vector representation")
        if config:
            raise ValueError(f"unknown feature baseline options: {sorted(config)}")
        model = FeatureVectorBaseline(model_id, spec.feature_names, seed=seed)
        model.canonical_model_name = canonical_model_name
        return model

    if model_id in {"rocket_numpy", "minirocket_ablation"}:
        if mode is not RepresentationMode.FEATURE_MATRIX:
            raise ValueError("canonical ROCKET and MiniROCKET require feature_matrix representation")
        if spec.n_channels <= 0:
            raise ValueError("feature_matrix channel count D must be resolved before model creation")
        n_kernels = int(config.pop("n_kernels", 10_000 if model_id == "rocket_numpy" else 1_000))
        alpha = float(config.pop("alpha", 1.0))
        if config:
            raise ValueError(f"unknown ROCKET options: {sorted(config)}")
        constructor = RocketRidgeClassifier if model_id == "rocket_numpy" else MiniRocketAblation
        model = constructor(n_kernels=n_kernels, alpha=alpha, seed=seed)
        model.canonical_model_name = canonical_model_name
        return model

    if model_id == "compact_cnn":
        if mode is not RepresentationMode.RAW:
            raise ValueError("CompactCNN1D requires raw representation")
        if spec.n_channels <= 0:
            raise ValueError("raw input requires a positive n_channels")
        from .compact_cnn import CompactCNN1D

        _torch_seed(seed)
        dropout = float(config.pop("dropout", 0.20))
        if config:
            raise ValueError(f"unknown CompactCNN options: {sorted(config)}")
        model = CompactCNN1D(spec.n_channels, spec.n_classes, dropout=dropout)
        model.model_id = model_id
        model.canonical_model_name = canonical_model_name
        return model

    if model_id in {"inception_full", "inception_small", "inception_matrix"}:
        if model_id in {"inception_full", "inception_small"} and mode is not RepresentationMode.RAW:
            raise ValueError("InceptionTimeFull/Small require raw representation")
        if model_id == "inception_matrix" and mode is not RepresentationMode.FEATURE_MATRIX:
            raise ValueError("InceptionTimeMatrix requires feature_matrix representation")
        if spec.n_channels <= 0:
            raise ValueError("InceptionTime input channel count must be resolved and positive")
        from .inception import InceptionTimeSingleNetwork

        _torch_seed(seed)
        variant = (
            str(config.pop("variant", "full"))
            if model_id == "inception_matrix"
            else model_id.removeprefix("inception_")
        )
        dropout = float(config.pop("dropout", 0.2))
        if config:
            raise ValueError(f"unknown InceptionTime options: {sorted(config)}")
        model = InceptionTimeSingleNetwork(
            spec.n_channels,
            spec.n_classes,
            variant=variant,
            dropout=dropout,
        )
        model.model_id = model_id
        model.canonical_model_name = canonical_model_name
        return model

    if model_id == "inception_five_member_ensemble":
        if mode not in {RepresentationMode.RAW, RepresentationMode.FEATURE_MATRIX}:
            raise ValueError("InceptionTime ensemble requires raw or feature_matrix representation")
        if spec.n_channels <= 0:
            raise ValueError("ensemble input channel count must be resolved and positive")
        from .inception import InceptionTimeFiveMemberProbabilityEnsemble, InceptionTimeSingleNetwork

        variant = str(config.pop("variant", "full"))
        dropout = float(config.pop("dropout", 0.2))
        member_seeds = tuple(int(value) for value in config.pop("member_seeds", (seed, seed + 1, seed + 2, seed + 3, seed + 4)))
        if len(member_seeds) != 5 or len(set(member_seeds)) != 5:
            raise ValueError("InceptionTime ensemble requires exactly five distinct member_seeds")
        if config:
            raise ValueError(f"unknown ensemble options: {sorted(config)}")
        members = []
        for member_seed in member_seeds:
            _torch_seed(member_seed)
            members.append(
                InceptionTimeSingleNetwork(
                    spec.n_channels, spec.n_classes, variant=variant, dropout=dropout
                )
            )
        model = InceptionTimeFiveMemberProbabilityEnsemble(members, member_seeds)
        model.model_id = model_id
        model.canonical_model_name = canonical_model_name
        return model

    if model_id == "shapeformer_effect_size":
        if mode not in {RepresentationMode.RAW, RepresentationMode.FEATURE_MATRIX}:
            raise ValueError("ShapeFormer requires raw or feature_matrix representation")
        if spec.n_channels <= 0:
            raise ValueError("ShapeFormer input channel count must be resolved and positive")
        from .shapeformer import EffectSizeShapelets, ExperimentalShapeFormer

        # English: The selected discovery identity and physical input scale are
        # mandatory model configuration, even when a fitted bank is supplied.
        # 中文：即使已经传入拟合库，发现方法和物理输入尺度仍是必填模型配置。
        required = {
            "shapelets",
            "discovery_method",
            "input_fs_hz",
            "outer_repeat_index",
            "outer_fold_index",
            "outer_train_participant_hash",
        }
        missing = sorted(required - set(config))
        if missing:
            raise ValueError(f"shapeformer_effect_size missing required options: {missing}")
        discovery_method = str(config.pop("discovery_method"))
        if discovery_method != "effect_size_shapelets_v1":
            raise ValueError(
                "ShapeFormer discovery_method must explicitly equal "
                "effect_size_shapelets_v1; unsupported/PISD requests never fall back"
            )
        input_fs_hz = float(config.pop("input_fs_hz"))
        outer_repeat_index = int(config.pop("outer_repeat_index"))
        outer_fold_index = int(config.pop("outer_fold_index"))
        outer_train_participant_hash = str(config.pop("outer_train_participant_hash"))
        shapelets_value = config.pop("shapelets")
        if isinstance(shapelets_value, Mapping):
            bank_required = {
                "values",
                "source_classes",
                "effect_sizes",
                "fitted_participant_ids",
                "discovery_method",
                "input_fs_hz",
                "shapelet_length_samples",
                "shapelet_length_seconds",
                "outer_repeat_index",
                "outer_fold_index",
                "outer_train_participant_hash",
            }
            bank_missing = sorted(bank_required - set(shapelets_value))
            if bank_missing:
                raise ValueError(f"shapelet bank missing provenance fields: {bank_missing}")
            shapelets_value = EffectSizeShapelets(
                values=np.asarray(shapelets_value["values"], dtype=np.float32),
                source_classes=np.asarray(shapelets_value["source_classes"]),
                effect_sizes=np.asarray(shapelets_value["effect_sizes"]),
                fitted_participant_ids=tuple(shapelets_value["fitted_participant_ids"]),
                discovery_method=str(shapelets_value["discovery_method"]),
                input_fs_hz=float(shapelets_value["input_fs_hz"]),
                shapelet_length_samples=int(shapelets_value["shapelet_length_samples"]),
                shapelet_length_seconds=float(shapelets_value["shapelet_length_seconds"]),
                outer_repeat_index=int(shapelets_value["outer_repeat_index"]),
                outer_fold_index=int(shapelets_value["outer_fold_index"]),
                outer_train_participant_hash=str(
                    shapelets_value["outer_train_participant_hash"]
                ),
            )
        if not isinstance(shapelets_value, EffectSizeShapelets):
            raise TypeError("shapelets must be EffectSizeShapelets or its complete mapping")
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

        hidden_channels = int(config.pop("hidden_channels", 64))
        dropout = float(config.pop("dropout", 0.2))
        patch_size_samples = int(config.pop("patch_size_samples", 16))
        attention_heads = int(config.pop("attention_heads", 4))
        attention_layers = int(config.pop("attention_layers", 1))
        if config:
            raise ValueError(f"unknown ShapeFormer options: {sorted(config)}")
        _torch_seed(seed)
        model = ExperimentalShapeFormer(
            spec.n_channels,
            spec.n_classes,
            shapelets_value,
            hidden_channels=hidden_channels,
            dropout=dropout,
            patch_size_samples=patch_size_samples,
            attention_heads=attention_heads,
            attention_layers=attention_layers,
            input_fs_hz=input_fs_hz,
        )
        model.model_id = model_id
        model.canonical_model_name = canonical_model_name
        return model

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
            signal = CompactCNN1D(spec.n_channels, spec.n_classes, dropout=0.0)
        else:
            signal = InceptionTimeSingleNetwork(
                spec.n_channels,
                spec.n_classes,
                variant=str(config.pop("variant", "small")),
                dropout=0.0,
            )
        feature_hidden_dim = int(config.pop("feature_hidden_dim", 32))
        fusion_hidden_dim = int(config.pop("fusion_hidden_dim", 64))
        pooling = str(config.pop("pooling", "mean"))
        dropout = float(config.pop("dropout", 0.2))
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
        return model

    raise ValueError(f"unsupported model_id: {model_id}")


# Backward-compatible descriptive alias used by internal configuration code.
# 内部配置代码使用的描述性兼容别名。
build_model = create_model
