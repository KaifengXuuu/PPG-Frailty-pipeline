"""Unified frozen-membership trainer with outer-label isolation.

统一的冻结成员训练器，并严格隔离 outer 标签。
"""

from __future__ import annotations

import hashlib
import inspect
import pickle
import random
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np
from sklearn.metrics import balanced_accuracy_score

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
except ImportError:  # pragma: no cover - covered by subprocess portability test
    torch = None
    nn = None
    DataLoader = None
    Dataset = Any
    Subset = None
    WeightedRandomSampler = None

from ..provenance import stable_payload_sha256
from .aggregation import (
    LINE_A_EQUAL_FILES,
    aggregation_rule_for_training_balance,
    canonical_role_family,
)
from .datasets import FeatureMatrixDataset, FileBagDataset, collate_samples


def _require_torch() -> None:
    """Fail only for deep operations / 仅深度操作时失败。"""

    if torch is None or nn is None or DataLoader is None:
        raise ImportError(
            "deep training requires optional dependency torch; "
            "UnifiedTrainer.fit_estimator remains available without it"
        )


@dataclass(frozen=True)
class FrozenOuterSplit:
    """Immutable subject membership for one repeat/fold / 单个 repeat/fold 的不可变成员。"""

    repeat: int
    fold: int
    seed: int
    train_participant_ids: tuple[str, ...]
    oof_participant_ids: tuple[str, ...]
    registry_hash: str
    fold_hash: str
    train_dataset_hash: str | None = None

    def __post_init__(self) -> None:
        train = tuple(sorted(set(str(value) for value in self.train_participant_ids)))
        oof = tuple(sorted(set(str(value) for value in self.oof_participant_ids)))
        if not train or not oof or set(train) & set(oof):
            raise ValueError("outer train and OOF participant sets must be non-empty and disjoint")
        object.__setattr__(self, "train_participant_ids", train)
        object.__setattr__(self, "oof_participant_ids", oof)

    def assert_train_only(
        self,
        participant_ids: list[str] | tuple[str, ...],
        *,
        exact: bool = False,
    ) -> None:
        """Fail on outer leakage and optionally require the complete train roster.

        若发生 outer 泄漏则关闭失败；最终 refit 时可进一步要求完整 train roster。
        """

        fitted = set(str(value) for value in participant_ids)
        if not fitted or not fitted <= set(self.train_participant_ids):
            raise ValueError("fitted rows must be a non-empty subset of frozen outer-train subjects")
        if fitted & set(self.oof_participant_ids):
            raise ValueError("outer-OOF participant reached a fitting operation")
        if exact and fitted != set(self.train_participant_ids):
            missing = sorted(set(self.train_participant_ids) - fitted)
            extra = sorted(fitted - set(self.train_participant_ids))
            raise ValueError(
                f"final fit must use the exact frozen outer-train roster; missing={missing}, extra={extra}"
            )

    def bind_training_dataset(self, dataset: Dataset) -> "FrozenOuterSplit":
        """Return a split bound to exact row identities and values.

        返回绑定到逐行身份与数值内容的 split。该哈希可识别把 held-out 数组伪装成
        train participant ID 的错误。
        """

        self.assert_train_only(dataset_participant_ids(dataset), exact=True)
        validate_dataset_identity_coherence(dataset)
        return replace(self, train_dataset_hash=dataset_binding_hash(dataset))

    def assert_training_dataset(self, dataset: Dataset, *, exact: bool = True) -> None:
        """Validate roster, row identity coherence and optional content binding.

        校验 roster、逐行身份一致性，以及可选的内容绑定哈希。
        """

        self.assert_train_only(dataset_participant_ids(dataset), exact=exact)
        validate_dataset_identity_coherence(dataset)
        if self.train_dataset_hash is not None:
            observed = dataset_binding_hash(dataset)
            if observed != self.train_dataset_hash:
                raise ValueError("training dataset identity/content hash does not match frozen split")

    @property
    def membership_hash(self) -> str:
        """Stable hash of exact membership / 精确成员关系的稳定哈希。"""

        return stable_payload_sha256(
            {
                "repeat": self.repeat,
                "fold": self.fold,
                "seed": self.seed,
                "train": self.train_participant_ids,
                "oof": self.oof_participant_ids,
                "registry_hash": self.registry_hash,
                "fold_hash": self.fold_hash,
                "train_dataset_hash": self.train_dataset_hash,
            }
        )


@dataclass(frozen=True)
class FullCohortRefitScope:
    """Exact all-29 final-refit membership guard with no pseudo holdout."""

    participant_ids: tuple[str, ...]
    registry_hash: str
    config_hash: str
    oof_evidence_hash: str
    train_dataset_hash: str | None = None
    scope_kind: str = "final_refit_all_29"

    def __post_init__(self) -> None:
        participants = tuple(sorted(set(map(str, self.participant_ids))))
        if len(participants) != 29:
            raise ValueError("full-cohort refit scope requires exactly 29 participants")
        if self.scope_kind != "final_refit_all_29":
            raise ValueError("full-cohort scope_kind is frozen")
        for name in ("registry_hash", "config_hash", "oof_evidence_hash"):
            digest = str(getattr(self, name))
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        object.__setattr__(self, "participant_ids", participants)

    @property
    def train_participant_ids(self) -> tuple[str, ...]:
        return self.participant_ids

    @property
    def oof_participant_ids(self) -> tuple[str, ...]:
        return ()

    @property
    def repeat(self) -> int:
        """Compatibility index for fold-local model preparation provenance."""

        return 0

    @property
    def fold(self) -> int:
        """Compatibility index; fold_hash remains explicitly full-cohort."""

        return 0

    @property
    def seed(self) -> int:
        return 42

    @property
    def fold_hash(self) -> str:
        return stable_payload_sha256(
            {
                "scope_kind": self.scope_kind,
                "participants": self.participant_ids,
                "registry_hash": self.registry_hash,
                "config_hash": self.config_hash,
                "oof_evidence_hash": self.oof_evidence_hash,
            }
        )

    @property
    def membership_hash(self) -> str:
        return stable_payload_sha256(
            {
                "scope_kind": self.scope_kind,
                "participants": self.participant_ids,
                "registry_hash": self.registry_hash,
                "config_hash": self.config_hash,
                "oof_evidence_hash": self.oof_evidence_hash,
                "train_dataset_hash": self.train_dataset_hash,
            }
        )

    def assert_train_only(
        self,
        participant_ids: list[str] | tuple[str, ...],
        *,
        exact: bool = False,
    ) -> None:
        fitted = set(map(str, participant_ids))
        expected = set(self.participant_ids)
        if not fitted or not fitted <= expected:
            raise ValueError("final-refit rows must be a non-empty subset of the all-29 roster")
        if exact and fitted != expected:
            raise ValueError("final refit must use the exact all-29 participant roster")

    def bind_training_dataset(self, dataset: Dataset) -> "FullCohortRefitScope":
        self.assert_train_only(dataset_participant_ids(dataset), exact=True)
        validate_dataset_identity_coherence(dataset)
        return replace(self, train_dataset_hash=dataset_binding_hash(dataset))

    def assert_training_dataset(self, dataset: Dataset, *, exact: bool = True) -> None:
        self.assert_train_only(dataset_participant_ids(dataset), exact=exact)
        validate_dataset_identity_coherence(dataset)
        if self.train_dataset_hash is not None:
            observed = dataset_binding_hash(dataset)
            if observed != self.train_dataset_hash:
                raise ValueError("final-refit dataset identity/content hash changed")


@dataclass(frozen=True)
class InnerGroupedSplit:
    """Optional inner train/validation membership / 可选 inner train/validation 成员。"""

    train_participant_ids: tuple[str, ...]
    validation_participant_ids: tuple[str, ...]

    def validate(self, outer: FrozenOuterSplit) -> None:
        """Require disjoint subsets of outer-train / 要求是 outer-train 的互斥子集。"""

        train = set(self.train_participant_ids)
        validation = set(self.validation_participant_ids)
        outer_train = set(outer.train_participant_ids)
        if not train or not validation or train & validation:
            raise ValueError("inner train/validation sets must be non-empty and disjoint")
        if train | validation != outer_train:
            raise ValueError("inner membership must exactly partition outer-train")


DEEP_EPOCH_PROFILES: dict[str, int] = {
    "default_10": 10,
    "ablation_7": 7,
    "ablation_15": 15,
}


@dataclass(frozen=True)
class TrainingConfig:
    """V2 fixed-epoch and balance-line controls.

    Formal deep runs use exactly one named 7/10/15 profile.  A one-epoch run is
    available only through the explicit smoke execution mode and is never a
    benchmark configuration. Inner-label epoch selection and early stopping are
    not executable in V2.
    """

    epoch_rule: str = "fixed_epoch"
    epoch_profile: str = "default_10"
    execution_mode: str = "formal"
    fixed_epochs: int = 10
    maximum_inner_epochs: int = 0
    inner_patience: int = 0
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    num_workers: int = 0
    seed: int = 42
    optimizer: str = "adam"
    class_weighting: str = "outer_train_inverse_frequency"
    training_balance: str = "equal_files"
    sampler: str = "balance_line_weighted_v2"
    classifier_role_families: tuple[str, ...] = ("B", "R")
    loss: str = "cross_entropy"
    label_smoothing: float = 0.0
    gradient_clip_norm: float | None = None
    deterministic_algorithms: bool = True
    cache_policy: str = "content_addressed_strict"
    outer_labels_visible_to_trainer: bool = False
    inner_grouped_folds: int = 0
    refit_on_all_outer_training: bool = True
    n_classes: int = 3
    legacy_epoch_rule_alias: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.epoch_rule == "fixed":
            object.__setattr__(self, "epoch_rule", "fixed_epoch")
            object.__setattr__(self, "legacy_epoch_rule_alias", "fixed")
        if self.epoch_rule != "fixed_epoch":
            raise ValueError("V2 permits fixed_epoch only; inner selection/early stop are disabled")
        if self.execution_mode not in {"formal", "smoke"}:
            raise ValueError("execution_mode must be formal or smoke")
        if self.execution_mode == "formal":
            expected = DEEP_EPOCH_PROFILES.get(self.epoch_profile)
            if expected is None or self.fixed_epochs != expected:
                raise ValueError(
                    "formal fixed_epochs must match default_10, ablation_7 or ablation_15"
                )
        elif self.epoch_profile != "smoke" or self.fixed_epochs <= 0:
            raise ValueError("smoke execution requires epoch_profile=smoke and positive epochs")
        if self.maximum_inner_epochs != 0 or self.inner_patience != 0:
            raise ValueError("V2 disables inner epochs, patience and early stopping")
        if self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers non-negative")
        if self.optimizer != "adam":
            raise ValueError("V2 trainer implements optimizer=adam exactly")
        if self.class_weighting != "outer_train_inverse_frequency":
            raise ValueError("unsupported class_weighting")
        if not np.isfinite(self.label_smoothing) or not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be finite in [0,1)")
        if self.gradient_clip_norm is not None and (
            not np.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0
        ):
            raise ValueError("gradient_clip_norm must be null or finite and positive")
        if self.training_balance not in {"equal_files", "equal_role_families"}:
            raise ValueError("unsupported training_balance")
        if self.sampler != "balance_line_weighted_v2":
            raise ValueError("V2 requires sampler=balance_line_weighted_v2")
        families = tuple(canonical_role_family(value) for value in self.classifier_role_families)
        if not families or len(families) != len(set(families)):
            raise ValueError("classifier_role_families must be non-empty and unique")
        object.__setattr__(self, "classifier_role_families", families)
        if self.loss != "cross_entropy":
            raise ValueError("unsupported loss")
        if self.cache_policy != "content_addressed_strict":
            raise ValueError("unsupported cache_policy")
        if self.outer_labels_visible_to_trainer:
            raise ValueError("outer_labels_visible_to_trainer must remain false")
        if not self.refit_on_all_outer_training:
            raise ValueError("refit_on_all_outer_training must remain true")
        if self.inner_grouped_folds != 0 or self.n_classes <= 1:
            raise ValueError("V2 requires inner_grouped_folds=0 and n_classes>1")

    @property
    def expected_aggregation_rule(self) -> str:
        """Return the aggregation line that must accompany this training balance."""

        return aggregation_rule_for_training_balance(self.training_balance)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "TrainingConfig":
        """Construct from a resolved YAML training block without ignored keys.

        从 resolved YAML 的 training 区块构造，禁止忽略任何未知字段。
        """

        accepted = {
            field_name
            for field_name, definition in cls.__dataclass_fields__.items()
            if definition.init
        }
        unknown = sorted(set(value) - accepted)
        if unknown:
            raise ValueError(f"unknown training configuration fields: {unknown}")
        return cls(**dict(value))


DEEP_EPOCH_CONFIG_IDS: dict[str, str] = {
    "fixed_epoch_10_reference": "default_10",
    "epoch_7_ablation": "ablation_7",
    "epoch_15_ablation": "ablation_15",
}
DEEP_MODEL_IDS = frozenset(
    {
        "compact_cnn",
        "inception_full",
        "inception_small",
        "inception_matrix",
        "inception_full_five_member_ensemble",
        "inception_matrix_five_member_ensemble",
        "shapeformer_channel_specific_osd",
        "shapeformer_channel_specific_scalar_distance_ablation",
        "shapeformer_effect_size_fixed_v1",
        "fusion_compact",
        "fusion_inception",
    }
)


def materialize_deep_epoch_config(
    model_id: str,
    config_id: str,
    *,
    base_config: TrainingConfig | None = None,
) -> TrainingConfig:
    """Materialize one exact 7/10/15 deep profile; reject non-deep models."""

    machine_id = str(model_id)
    if machine_id not in DEEP_MODEL_IDS:
        raise ValueError(
            "epoch 7/10/15 profiles are deep-only; classical and ROCKET models "
            "must use epoch_rule=not_applicable"
        )
    try:
        epoch_profile = DEEP_EPOCH_CONFIG_IDS[str(config_id)]
    except KeyError as exc:
        raise ValueError(f"unknown deep epoch config identity: {config_id}") from exc
    fixed_epochs = DEEP_EPOCH_PROFILES[epoch_profile]
    base = TrainingConfig() if base_config is None else base_config
    return replace(
        base,
        epoch_rule="fixed_epoch",
        epoch_profile=epoch_profile,
        execution_mode="formal",
        fixed_epochs=fixed_epochs,
        maximum_inner_epochs=0,
        inner_patience=0,
    )


def materialize_all_deep_epoch_configs(
    model_id: str,
    *,
    base_config: TrainingConfig | None = None,
) -> dict[str, TrainingConfig]:
    """Return the three named materializable identities without executing them."""

    return {
        config_id: materialize_deep_epoch_config(
            model_id, config_id, base_config=base_config
        )
        for config_id in DEEP_EPOCH_CONFIG_IDS
    }


@dataclass(frozen=True)
class FittedObjectProvenance:
    """Fold-local audit record for every fitted object / 每个拟合对象的 fold-local 审计记录。"""

    object_type: str
    fitted_participant_ids: tuple[str, ...]
    outer_membership_hash: str
    registry_hash: str
    fold_hash: str
    epoch_rule: str
    selected_epoch: int | None
    state_hash: str
    dataset_binding_hash: str = ""
    training_balance: str = ""
    expected_aggregation_rule: str = ""
    epoch_profile: str = ""
    execution_mode: str = ""
    training_seed: int = 42
    member_training_seeds: tuple[int, ...] = ()
    member_state_hashes: tuple[str, ...] = ()
    optimizer: str = ""
    learning_rate: float = 0.0
    weight_decay: float = 0.0
    loss: str = ""
    class_weighting: str = ""
    sampler: str = ""
    label_smoothing: float = 0.0
    gradient_clip_norm: float | None = None
    notes: tuple[str, ...] = ()


@dataclass
class TrainingResult:
    """Fitted model plus auditable history / 已拟合模型及可审计历史。"""

    model: Any
    selected_epoch: int | None
    history: list[dict[str, float | int | str]] = field(default_factory=list)
    provenance: FittedObjectProvenance | None = None


def dataset_participant_ids(dataset: Dataset) -> tuple[str, ...]:
    """Read row identities through nested ``Subset`` objects / 穿透 Subset 读取行身份。"""

    if Subset is not None and isinstance(dataset, Subset):
        parent = dataset_participant_ids(dataset.dataset)
        return tuple(parent[int(index)] for index in dataset.indices)
    if not hasattr(dataset, "participant_ids"):
        raise TypeError("dataset must expose row-aligned participant_ids")
    return tuple(str(value) for value in dataset.participant_ids)


def dataset_identities(dataset: Dataset) -> tuple[Any, ...]:
    """Return row identities through nested subsets / 穿透子集返回逐行身份。"""

    if Subset is not None and isinstance(dataset, Subset):
        parent = dataset_identities(dataset.dataset)
        return tuple(parent[int(index)] for index in dataset.indices)
    if not hasattr(dataset, "identities"):
        raise TypeError("dataset must expose row-aligned identities")
    return tuple(dataset.identities)


def validate_dataset_identity_coherence(dataset: Dataset) -> None:
    """Reject internally inconsistent participant/file/window identities.

    拒绝 participant、file、window 或标签彼此矛盾的数据身份。该校验不能由调用方
    单独提供 participant ID 绕过。
    """

    participant_ids = dataset_participant_ids(dataset)
    identities = dataset_identities(dataset)
    if len(participant_ids) != len(identities) or not identities:
        raise ValueError("dataset identities must be non-empty and row-aligned")
    labels_by_participant: dict[str, int] = {}
    files: dict[str, tuple[str, str, int]] = {}
    windows: dict[tuple[str, str], tuple[str, int]] = {}
    for declared, identity in zip(participant_ids, identities):
        participant = str(identity.participant_id)
        if declared != participant:
            raise ValueError("dataset participant_ids disagree with row identities")
        label = int(identity.label)
        previous_label = labels_by_participant.setdefault(participant, label)
        if previous_label != label:
            raise ValueError("one participant has inconsistent labels")
        file_signature = (participant, str(identity.role), label)
        previous_file = files.setdefault(str(identity.file_id), file_signature)
        if previous_file != file_signature:
            raise ValueError("one file_id has inconsistent participant, role or label")
        if identity.window_id is not None:
            window_key = (str(identity.file_id), str(identity.window_id))
            window_signature = (participant, label)
            previous_window = windows.setdefault(window_key, window_signature)
            if previous_window != window_signature:
                raise ValueError("one window identity has inconsistent participant or label")


def _hash_dataset_value(digest: "hashlib._Hash", name: str, value: Any) -> None:
    """Update one content digest deterministically / 确定性更新一个内容哈希。"""

    digest.update(name.encode("utf-8"))
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    elif isinstance(value, (list, tuple)):
        digest.update(str(len(value)).encode("ascii"))
        for index, item in enumerate(value):
            _hash_dataset_value(digest, f"{name}[{index}]", item)
    else:
        digest.update(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def dataset_binding_hash(dataset: Dataset) -> str:
    """Hash row identities together with all exposed model-input arrays.

    将逐行身份与数据集公开的全部模型输入数组共同哈希，绑定名称与真实数值。
    """

    validate_dataset_identity_coherence(dataset)
    digest = hashlib.sha256()
    identity_payload = tuple(
        (
            str(identity.participant_id),
            str(identity.file_id),
            str(identity.role),
            int(identity.label),
            str(identity.signal_route),
            None if identity.window_id is None else str(identity.window_id),
        )
        for identity in dataset_identities(dataset)
    )
    _hash_dataset_value(digest, "identities", identity_payload)
    found = False
    for name in (
        "values",
        "sample_mask",
        "row_mask",
        "window_bags",
        "window_masks",
        "file_features",
    ):
        if hasattr(dataset, name):
            _hash_dataset_value(digest, name, getattr(dataset, name))
            found = True
    if not found:
        raise TypeError("dataset binding requires exposed model-input arrays")
    return digest.hexdigest()


def participant_file_window_sampling_weights(
    dataset: Dataset,
    *,
    training_balance: str = "equal_files",
    allowed_role_families: tuple[str, ...] = ("B", "R"),
) -> np.ndarray:
    """Return Line A or Line B row weights with participant mass fixed first.

    Line A is participant→file→row. Line B is
    participant→available-role-family→file→row. Numeric role suffixes identify
    files only and are canonicalised before Line B weighting.
    """

    validate_dataset_identity_coherence(dataset)
    identities = dataset_identities(dataset)
    if training_balance not in {"equal_files", "equal_role_families"}:
        raise ValueError(f"unsupported training_balance: {training_balance!r}")
    allowed = tuple(canonical_role_family(value) for value in allowed_role_families)
    if not allowed or len(allowed) != len(set(allowed)):
        raise ValueError("allowed_role_families must be non-empty and unique")
    participants = sorted({str(identity.participant_id) for identity in identities})
    family_by_row = [canonical_role_family(identity.role) for identity in identities]
    outside = sorted(set(family_by_row) - set(allowed))
    if outside:
        raise ValueError(
            f"classifier rows contain role families outside the declared input set: {outside}"
        )
    files_by_participant: dict[str, set[str]] = {participant: set() for participant in participants}
    families_by_participant: dict[str, set[str]] = {
        participant: set() for participant in participants
    }
    files_by_participant_family: dict[tuple[str, str], set[str]] = {}
    row_counts: dict[tuple[str, str], int] = {}
    for identity, family in zip(identities, family_by_row):
        participant = str(identity.participant_id)
        file_id = str(identity.file_id)
        files_by_participant[participant].add(file_id)
        families_by_participant[participant].add(family)
        files_by_participant_family.setdefault((participant, family), set()).add(file_id)
        row_counts[(participant, file_id)] = row_counts.get((participant, file_id), 0) + 1

    values: list[float] = []
    for identity, family in zip(identities, family_by_row):
        participant = str(identity.participant_id)
        file_id = str(identity.file_id)
        if training_balance == "equal_files":
            denominator = (
                len(participants)
                * len(files_by_participant[participant])
                * row_counts[(participant, file_id)]
            )
        else:
            denominator = (
                len(participants)
                * len(families_by_participant[participant])
                * len(files_by_participant_family[(participant, family)])
                * row_counts[(participant, file_id)]
            )
        values.append(1.0 / denominator)
    weights = np.asarray(values, dtype=np.float64)
    if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("V2 hierarchy sampling weights do not sum to one")
    return weights

def outer_train_inverse_frequency_weights(dataset: Dataset, n_classes: int) -> np.ndarray:
    """Compute class weights from unique outer-training participants only.

    只根据 outer-training 的唯一 participant 计算类别权重。
    """

    labels_by_participant: dict[str, int] = {}
    for identity in dataset_identities(dataset):
        previous = labels_by_participant.setdefault(identity.participant_id, int(identity.label))
        if previous != int(identity.label):
            raise ValueError("one participant has inconsistent class labels")
    labels = np.asarray(list(labels_by_participant.values()), dtype=np.int64)
    if labels.size == 0 or np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError("outer-training participant labels are invalid")
    counts = np.bincount(labels, minlength=n_classes)
    present = counts > 0
    weights = np.zeros(n_classes, dtype=np.float32)
    weights[present] = labels.size / (int(present.sum()) * counts[present])
    return weights


def subset_by_participants(dataset: Dataset, participant_ids: set[str]) -> Subset:
    """Create a deterministic row subset / 创建确定性行子集。"""

    _require_torch()
    indices = [
        index
        for index, participant_id in enumerate(dataset_participant_ids(dataset))
        if participant_id in participant_ids
    ]
    if not indices:
        raise ValueError("participant subset contains no dataset rows")
    return Subset(dataset, indices)


def forward_batch(model: nn.Module, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Dispatch a typed batch without representation guessing / 分派类型化批次。"""

    _require_torch()
    if "window_bag" in batch:
        return model(
            batch["window_bag"].to(device),
            batch["window_mask"].to(device),
            batch["file_features"].to(device),
            batch["sample_mask"].to(device),
        )
    x = batch["x"].to(device)
    mask = batch.get("mask")
    return model(x, None if mask is None else mask.to(device))


def _state_hash(model: nn.Module) -> str:
    """Hash tensor names, dtypes, shapes and bytes / 哈希张量名称、类型、形状与字节。"""

    _require_torch()
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def model_member_state_hashes(model: nn.Module) -> tuple[str, ...]:
    """Return one state hash per independent member, or one for a single model."""

    members = tuple(model.members) if hasattr(model, "members") else (model,)
    return tuple(_state_hash(member) for member in members)


class UnifiedTrainer:
    """One trainer enforcing frozen membership and outer-label blindness.

    统一训练器，强制冻结成员且对 outer 标签保持不可见。公开 ``fit`` 签名没有
    ``outer_oof_dataset`` 或 ``outer_y`` 参数；epoch 只能固定，或由 outer-train 内部
    的 grouped validation 选择，随后在完整 outer-train 上从头重训。
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch.device(config.device) if torch is not None else None

    def _loader(
        self,
        dataset: Dataset,
        *,
        shuffle: bool,
        seed_offset: int = 0,
        absolute_seed: int | None = None,
    ) -> DataLoader:
        """Build a seeded deterministic loader / 构造带种子的确定性加载器。"""

        _require_torch()
        generator = torch.Generator()
        generator.manual_seed(
            self.config.seed + seed_offset if absolute_seed is None else int(absolute_seed)
        )
        sampler = None
        if shuffle and self.config.sampler == "balance_line_weighted_v2":
            sampler = WeightedRandomSampler(
                torch.as_tensor(
                    participant_file_window_sampling_weights(
                        dataset,
                        training_balance=self.config.training_balance,
                        allowed_role_families=self.config.classifier_role_families,
                    ),
                    dtype=torch.double,
                ),
                num_samples=len(dataset),
                replacement=True,
                generator=generator,
            )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=self.config.num_workers,
            collate_fn=collate_samples,
            generator=generator,
        )

    def _set_seed(self, offset: int = 0) -> None:
        """Set Python, NumPy and PyTorch seeds / 设置三类随机种子。"""

        self._set_absolute_seed(self.config.seed + offset)

    def _set_absolute_seed(self, seed: int) -> None:
        """Set all training RNGs to one explicitly archived seed."""

        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        _require_torch()
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(self.config.deterministic_algorithms)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = self.config.deterministic_algorithms
            torch.backends.cudnn.benchmark = False

    def _criterion(self, dataset: Dataset) -> nn.Module:
        """Build train-only weighted cross entropy / 构造仅训练集加权交叉熵。"""

        weights = outer_train_inverse_frequency_weights(dataset, self.config.n_classes)
        return nn.CrossEntropyLoss(
            weight=torch.as_tensor(weights, dtype=torch.float32, device=self.device),
            label_smoothing=float(self.config.label_smoothing),
        )

    def _optimizer(self, member: nn.Module) -> torch.optim.Optimizer:
        """Build the exactly declared Adam optimizer / 构造配置声明的 Adam。"""

        return torch.optim.Adam(
            member.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _train_member_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Train one independent member for one epoch / 独立训练一个成员一个 epoch。"""

        model.train()
        total_loss, total_rows = 0.0, 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = forward_batch(model, batch, self.device)
            labels = batch["y"].to(self.device)
            loss = criterion(logits, labels)
            loss.backward()
            if self.config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float(self.config.gradient_clip_norm)
                )
            optimizer.step()
            total_loss += float(loss.detach()) * labels.shape[0]
            total_rows += int(labels.shape[0])
        return total_loss / max(total_rows, 1)

    def _train_epochs(
        self, model: nn.Module, dataset: Dataset, epochs: int
    ) -> list[dict[str, float | int | str]]:
        """Train a network or five members independently / 训练单网络或五个独立成员。"""

        model.to(self.device)
        history: list[dict[str, float | int | str]] = []
        members = list(model.members) if hasattr(model, "members") else [model]
        member_seeds = tuple(
            int(value) for value in getattr(model, "member_seeds", (self.config.seed,))
        )
        if len(member_seeds) != len(members):
            raise ValueError("model member_seeds must align with independently trained members")
        optimizers = [self._optimizer(member) for member in members]
        criterion = self._criterion(dataset)
        for epoch in range(1, epochs + 1):
            for member_index, (member, optimizer) in enumerate(zip(members, optimizers)):
                member_seed = member_seeds[member_index]
                epoch_seed = member_seed + epoch * 1_000_000
                self._set_absolute_seed(epoch_seed)
                loader = self._loader(dataset, shuffle=True, absolute_seed=epoch_seed)
                loss = self._train_member_epoch(member, loader, optimizer, criterion)
                history.append(
                    {
                        "epoch": epoch,
                        "member": member_index,
                        "training_seed": member_seed,
                        "epoch_rng_seed": epoch_seed,
                        "training_loss": float(loss),
                    }
                )
        return history

    def _predict_probabilities_with_identities(
        self, model: nn.Module, dataset: Dataset
    ) -> tuple[np.ndarray, np.ndarray, tuple[Any, ...]]:
        """Predict inner probabilities with row-aligned immutable identities.

        预测 inner 概率并同时返回逐行不可变身份。
        """

        model.to(self.device)
        model.eval()
        probabilities, labels, identities = [], [], []
        with torch.no_grad():
            for batch in self._loader(dataset, shuffle=False):
                if hasattr(model, "predict_probabilities"):
                    if "window_bag" in batch:
                        raise TypeError("ensemble file-bag prediction is not defined")
                    x = batch["x"].to(self.device)
                    mask = batch.get("mask")
                    probability = model.predict_probabilities(
                        x, None if mask is None else mask.to(self.device)
                    )
                else:
                    probability = torch.softmax(forward_batch(model, batch, self.device), dim=-1)
                probabilities.append(probability.cpu().numpy())
                labels.append(batch["y"].numpy())
                identities.extend(batch["identities"])
        return np.concatenate(probabilities), np.concatenate(labels), tuple(identities)

    @staticmethod
    def _validate_row_aligned_probabilities(
        probability: np.ndarray,
        labels: np.ndarray,
        identities: tuple[Any, ...],
        *,
        n_classes: int,
    ) -> tuple[np.ndarray, np.ndarray, tuple[Any, ...]]:
        """Validate public prediction output / 校验公开预测结果的逐行对齐与概率语义。"""

        values = np.asarray(probability, dtype=np.float64)
        targets = np.asarray(labels, dtype=np.int64)
        if (
            values.ndim != 2
            or values.shape != (targets.size, int(n_classes))
            or targets.size != len(identities)
        ):
            raise ValueError("probabilities, labels and identities are not row-aligned")
        if (
            not np.isfinite(values).all()
            or np.any(values < 0.0)
            or not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)
        ):
            raise ValueError("predict_probabilities must return finite normalised probabilities")
        return values, targets, tuple(identities)

    def predict_probabilities(
        self, model: nn.Module, dataset: Dataset
    ) -> tuple[np.ndarray, np.ndarray, tuple[Any, ...]]:
        """Predict a deep/ensemble/FileBag dataset without fitting.

        对 deep、五成员 ensemble 或 FileBag 数据执行无拟合预测，返回与输入行严格
        对齐的 probabilities、labels 和 identities。
        """

        _require_torch()
        probability, labels, identities = self._predict_probabilities_with_identities(
            model, dataset
        )
        return self._validate_row_aligned_probabilities(
            probability,
            labels,
            identities,
            n_classes=self.config.n_classes,
        )

    def predict_ensemble_members(
        self, model: nn.Module, dataset: Dataset
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[Any, ...]]:
        """Return member and exact-average probabilities with row identities.

        The first result is ``[5,row,class]`` and the second is ``[row,class]``.
        This API lets the formal runner write five member OOF rows plus one
        independently checked ensemble row without retraining or re-predicting.
        """

        _require_torch()
        if not hasattr(model, "member_probabilities") or not hasattr(model, "member_seeds"):
            raise TypeError("predict_ensemble_members requires the canonical five-member model")
        model.to(self.device)
        model.eval()
        member_batches: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        identities: list[Any] = []
        with torch.no_grad():
            for batch in self._loader(dataset, shuffle=False):
                if "window_bag" in batch:
                    raise TypeError("Inception ensemble does not accept file-bag inputs")
                x = batch["x"].to(self.device)
                mask = batch.get("mask")
                members = model.member_probabilities(
                    x, None if mask is None else mask.to(self.device)
                )
                member_batches.append(members.cpu().numpy())
                labels.append(batch["y"].numpy())
                identities.extend(batch["identities"])
        member_probability = np.concatenate(member_batches, axis=1).astype(np.float64)
        averaged = member_probability.mean(axis=0)
        targets = np.concatenate(labels).astype(np.int64)
        self._validate_row_aligned_probabilities(
            averaged,
            targets,
            tuple(identities),
            n_classes=self.config.n_classes,
        )
        if member_probability.shape[0] != 5 or not np.allclose(
            member_probability.sum(axis=-1), 1.0, rtol=0.0, atol=1e-6
        ):
            raise RuntimeError("ensemble member probabilities violate the formal contract")
        return member_probability, averaged, targets, tuple(identities)

    def predict_estimator_probabilities(
        self, estimator: Any, dataset: Dataset
    ) -> tuple[np.ndarray, np.ndarray, tuple[Any, ...]]:
        """Predict sklearn/ROCKET probabilities with the dataset mask when supported.

        通过统一公开入口预测 sklearn/ROCKET；若 estimator 声明 mask 参数，则传入
        FeatureMatrix row mask 或 raw sample mask。该方法绝不执行 fit。
        """

        if Subset is not None and isinstance(dataset, Subset):
            raise TypeError("estimator prediction expects a materialised dataset")
        if not hasattr(dataset, "values") or not hasattr(estimator, "predict_proba"):
            raise TypeError("estimator and materialised values are required")
        keywords: dict[str, Any] = {}
        if "mask" in inspect.signature(estimator.predict_proba).parameters:
            if isinstance(dataset, FeatureMatrixDataset):
                keywords["mask"] = dataset.row_mask
            elif hasattr(dataset, "sample_mask"):
                keywords["mask"] = dataset.sample_mask
        probability = estimator.predict_proba(dataset.values, **keywords)
        labels = np.asarray(dataset.labels, dtype=np.int64)
        identities = dataset_identities(dataset)
        return self._validate_row_aligned_probabilities(
            probability,
            labels,
            identities,
            n_classes=self.config.n_classes,
        )

    def _predict_probabilities(
        self, model: nn.Module, dataset: Dataset
    ) -> tuple[np.ndarray, np.ndarray]:
        """Backward-compatible probability-only helper / 向后兼容的概率辅助函数。"""

        probability, labels, _ = self.predict_probabilities(model, dataset)
        return probability, labels

    @staticmethod
    def _participant_validation_predictions(
        probability: np.ndarray,
        labels: np.ndarray,
        identities: tuple[Any, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate window→file→role→participant for epoch selection.

        为 epoch 选择执行 window→file→role→participant 等权聚合。
        """

        if probability.shape[0] != labels.size or labels.size != len(identities):
            raise ValueError("inner predictions, labels and identities are not row-aligned")
        file_groups: dict[tuple[str, str, str], list[int]] = {}
        labels_by_participant: dict[str, int] = {}
        for index, identity in enumerate(identities):
            participant = str(identity.participant_id)
            label = int(labels[index])
            if labels_by_participant.setdefault(participant, label) != label:
                raise ValueError("inner participant has inconsistent labels")
            file_groups.setdefault(
                (participant, str(identity.role), str(identity.file_id)), []
            ).append(index)
        role_groups: dict[tuple[str, str], list[np.ndarray]] = {}
        for (participant, role, _), indices in file_groups.items():
            role_groups.setdefault((participant, role), []).append(
                probability[np.asarray(indices, dtype=np.int64)].mean(axis=0)
            )
        participant_groups: dict[str, list[np.ndarray]] = {}
        for (participant, _), file_probabilities in role_groups.items():
            participant_groups.setdefault(participant, []).append(
                np.asarray(file_probabilities, dtype=np.float64).mean(axis=0)
            )
        participants = sorted(participant_groups)
        aggregated = np.asarray(
            [
                np.asarray(participant_groups[participant], dtype=np.float64).mean(axis=0)
                for participant in participants
            ]
        )
        aggregated /= aggregated.sum(axis=1, keepdims=True)
        participant_labels = np.asarray(
            [labels_by_participant[participant] for participant in participants],
            dtype=np.int64,
        )
        return aggregated, participant_labels

    def _select_epoch(
        self, model: nn.Module, inner_train: Dataset, inner_validation: Dataset
    ) -> tuple[int, list[dict[str, float | int | str]]]:
        """Select an epoch using inner labels only / 仅使用 inner 标签选择 epoch。"""

        model.to(self.device)
        members = list(model.members) if hasattr(model, "members") else [model]
        optimizers = [self._optimizer(member) for member in members]
        criterion = self._criterion(inner_train)
        best_epoch, best_score, stale = 1, -np.inf, 0
        history: list[dict[str, float | int | str]] = []
        for epoch in range(1, self.config.maximum_inner_epochs + 1):
            for member_index, (member, optimizer) in enumerate(zip(members, optimizers)):
                loss = self._train_member_epoch(
                    member,
                    self._loader(inner_train, shuffle=True, seed_offset=epoch * 100 + member_index),
                    optimizer,
                    criterion,
                )
                history.append(
                    {"epoch": epoch, "member": member_index, "inner_training_loss": float(loss)}
                )
            probability, labels, identities = self._predict_probabilities_with_identities(
                model, inner_validation
            )
            participant_probability, participant_labels = (
                self._participant_validation_predictions(probability, labels, identities)
            )
            score = balanced_accuracy_score(
                participant_labels, participant_probability.argmax(axis=1)
            )
            history.append(
                {
                    "epoch": epoch,
                    "inner_balanced_accuracy": float(score),
                    "inner_participant_balanced_accuracy": float(score),
                    "inner_selection_unit": "participant",
                }
            )
            if score > best_score + 1e-12:
                best_epoch, best_score, stale = epoch, float(score), 0
            else:
                stale += 1
                if stale >= self.config.inner_patience:
                    break
        return best_epoch, history

    def fit(
        self,
        model_factory: Callable[[], nn.Module],
        outer_train_dataset: Dataset,
        frozen_split: FrozenOuterSplit | FullCohortRefitScope,
        *,
        inner_split: InnerGroupedSplit | None = None,
    ) -> TrainingResult:
        """Fit without accepting or reading outer-OOF labels.

        在不接受也不读取 outer-OOF 标签的前提下拟合。inner 路线选择 epoch 后必定
        重新创建模型并用全部 outer-train 从头训练，防止少用训练参与者。
        """

        _require_torch()
        fitted_ids = dataset_participant_ids(outer_train_dataset)
        frozen_split.assert_training_dataset(outer_train_dataset, exact=True)
        self._set_seed()
        if inner_split is not None:
            raise ValueError("V2 fixed-epoch training never accepts an inner split")
        selection_history: list[dict[str, float | int | str]] = []
        selected_epoch = self.config.fixed_epochs

        self._set_seed()
        model = model_factory()
        model.training_balance_ = self.config.training_balance
        model.expected_aggregation_rule_ = self.config.expected_aggregation_rule
        fit_history = self._train_epochs(model, outer_train_dataset, selected_epoch)
        member_training_seeds = tuple(
            int(value) for value in getattr(model, "member_seeds", ())
        )
        member_hashes = model_member_state_hashes(model)
        if member_training_seeds and len(member_training_seeds) != len(member_hashes):
            raise RuntimeError("ensemble seed/state provenance is not one-to-one")
        provenance = FittedObjectProvenance(
            object_type=type(model).__name__,
            fitted_participant_ids=tuple(sorted(set(fitted_ids))),
            outer_membership_hash=frozen_split.membership_hash,
            registry_hash=frozen_split.registry_hash,
            fold_hash=frozen_split.fold_hash,
            epoch_rule=self.config.epoch_rule,
            selected_epoch=selected_epoch,
            state_hash=_state_hash(model),
            dataset_binding_hash=dataset_binding_hash(outer_train_dataset),
            training_balance=self.config.training_balance,
            expected_aggregation_rule=self.config.expected_aggregation_rule,
            epoch_profile=self.config.epoch_profile,
            execution_mode=self.config.execution_mode,
            training_seed=int(self.config.seed),
            member_training_seeds=member_training_seeds,
            member_state_hashes=member_hashes,
            optimizer=self.config.optimizer,
            learning_rate=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
            loss=self.config.loss,
            class_weighting=self.config.class_weighting,
            sampler=self.config.sampler,
            label_smoothing=float(self.config.label_smoothing),
            gradient_clip_norm=self.config.gradient_clip_norm,
            notes=(
                "outer_labels_not_accepted_by_fit_api",
                "full_outer_train_refit",
                "training_dataset_identity_bound",
                f"membership_scope={getattr(frozen_split, 'scope_kind', 'frozen_outer_split')}",
                f"sampler={self.config.sampler}",
                f"training_balance={self.config.training_balance}",
                f"expected_aggregation_rule={self.config.expected_aggregation_rule}",
                f"class_weighting={self.config.class_weighting}",
                f"optimizer={self.config.optimizer}",
                f"loss={self.config.loss}",
                f"cache_policy={self.config.cache_policy}",
            )
            + (
                (
                    "training_seed_field_is_ensemble_orchestration_seed_only",
                    "member_training_seeds_are_authoritative_for_member_stochasticity",
                )
                if member_training_seeds
                else ()
            )
            + (
                (f"legacy_epoch_rule_alias={self.config.legacy_epoch_rule_alias}",)
                if self.config.legacy_epoch_rule_alias
                else ()
            ),
        )
        return TrainingResult(
            model=model,
            selected_epoch=selected_epoch,
            history=selection_history + fit_history,
            provenance=provenance,
        )

    def fit_estimator(
        self,
        estimator: Any,
        outer_train_dataset: Dataset,
        frozen_split: FrozenOuterSplit | FullCohortRefitScope,
    ) -> TrainingResult:
        """Fit sklearn/ROCKET estimators under the same membership guard.

        在相同成员守卫下拟合 sklearn/ROCKET estimator。
        """

        if self.config.epoch_profile in {"ablation_7", "ablation_15"}:
            raise ValueError(
                "epoch ablations apply only to deep models; classical/ROCKET fail closed"
            )
        fitted_ids = dataset_participant_ids(outer_train_dataset)
        frozen_split.assert_training_dataset(outer_train_dataset, exact=True)
        if Subset is not None and isinstance(outer_train_dataset, Subset):
            raise TypeError("fit_estimator expects a materialised dataset, not Subset")
        if not hasattr(outer_train_dataset, "values"):
            raise TypeError("estimator datasets must expose a values array")
        keywords: dict[str, Any] = {"participant_ids": fitted_ids}
        if "mask" in inspect.signature(estimator.fit).parameters:
            if isinstance(outer_train_dataset, FeatureMatrixDataset):
                keywords["mask"] = outer_train_dataset.row_mask
            elif hasattr(outer_train_dataset, "sample_mask"):
                keywords["mask"] = outer_train_dataset.sample_mask
        if "sample_weight" not in inspect.signature(estimator.fit).parameters:
            raise TypeError(
                "estimator must accept sample_weight for the declared balanced training protocol"
            )
        sampling_weight = participant_file_window_sampling_weights(
            outer_train_dataset,
            training_balance=self.config.training_balance,
            allowed_role_families=self.config.classifier_role_families,
        )
        class_weight = outer_train_inverse_frequency_weights(
            outer_train_dataset, self.config.n_classes
        )
        row_labels = np.asarray(outer_train_dataset.labels, dtype=np.int64)
        combined_weight = sampling_weight * class_weight[row_labels]
        if combined_weight.sum() <= 0:
            raise ValueError("combined estimator sample weights have zero mass")
        keywords["sample_weight"] = combined_weight / combined_weight.mean()
        estimator.fit(outer_train_dataset.values, outer_train_dataset.labels, **keywords)
        estimator.training_balance_ = self.config.training_balance
        estimator.expected_aggregation_rule_ = self.config.expected_aggregation_rule
        learned_ids = tuple(getattr(estimator, "fitted_participant_ids_", ()))
        if tuple(sorted(set(learned_ids))) != tuple(sorted(set(fitted_ids))):
            raise RuntimeError("fitted estimator did not preserve exact training provenance")
        expected_ids = tuple(sorted(set(fitted_ids)))
        nested = getattr(estimator, "fitted_object_provenance_", {})
        if not nested:
            raise RuntimeError("fitted estimator did not expose per-object provenance")
        for object_name, object_provenance in nested.items():
            observed_ids = tuple(
                sorted(set(str(value) for value in object_provenance["fitted_participant_ids"]))
            )
            if observed_ids != expected_ids:
                raise RuntimeError(
                    f"fitted object {object_name} has mismatched training provenance"
                )
        # English: Hash the learned estimator state, not merely hyperparameters.
        # 中文：哈希已经学习的 estimator 状态，而不只是超参数。
        state_hash = hashlib.sha256(
            pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)
        ).hexdigest()
        provenance = FittedObjectProvenance(
            object_type=type(estimator).__name__,
            fitted_participant_ids=tuple(sorted(set(fitted_ids))),
            outer_membership_hash=frozen_split.membership_hash,
            registry_hash=frozen_split.registry_hash,
            fold_hash=frozen_split.fold_hash,
            epoch_rule="not_applicable",
            selected_epoch=None,
            state_hash=state_hash,
            dataset_binding_hash=dataset_binding_hash(outer_train_dataset),
            training_balance=self.config.training_balance,
            expected_aggregation_rule=self.config.expected_aggregation_rule,
            epoch_profile="not_applicable",
            execution_mode=self.config.execution_mode,
            training_seed=int(self.config.seed),
            notes=(
                "all_stateful_transforms_fitted_on_outer_train",
                f"membership_scope={getattr(frozen_split, 'scope_kind', 'frozen_outer_split')}",
                f"training_balance={self.config.training_balance}",
                f"expected_aggregation_rule={self.config.expected_aggregation_rule}",
            ),
        )
        return TrainingResult(model=estimator, selected_epoch=None, provenance=provenance)
