"""Typed datasets for the four frozen representation modes.

四种冻结 representation mode 的类型化数据集。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..contracts import OrderedFeatureMatrixV1

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - covered by subprocess portability test
    torch = None

    class Dataset:  # type: ignore[no-redef]
        """English: Minimal non-deep base for estimator-only datasets.

        中文：仅 estimator 数据集使用的最小无深度依赖基类。
        """

        pass


def _require_torch() -> None:
    """Fail only when tensor materialisation is requested / 仅张量化时失败。"""

    if torch is None:
        raise ImportError("tensor dataset access requires optional dependency torch; "
                          "array-backed estimator fitting remains available")


@dataclass(frozen=True)
class SampleIdentity:
    """Immutable identity and grouping metadata / 不可变身份与分组元数据。"""

    participant_id: str
    file_id: str
    role: str
    label: int
    signal_route: str
    quality_score: float = 1.0
    retained: bool = True
    window_id: str | None = None
    aggregation_retained: bool = True

    def __post_init__(self) -> None:
        if not self.participant_id or not self.file_id or not self.role:
            raise ValueError("participant_id, file_id and role are required")
        if not np.isfinite(self.quality_score) or not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("quality_score must be finite in [0,1]")
        if not isinstance(self.retained, (bool, np.bool_)) or not isinstance(self.aggregation_retained,
                                                                             (bool, np.bool_)):
            raise ValueError("retained flags must be boolean")


class _IdentityDataset(Dataset):
    """Common identity helpers / 公共身份辅助方法。"""

    identities: tuple[SampleIdentity, ...]

    @property
    def participant_ids(self) -> tuple[str, ...]:
        """Return row-aligned participant ids / 返回与行对齐的参与者 ID。"""

        return tuple(identity.participant_id for identity in self.identities)

    @property
    def labels(self) -> np.ndarray:
        """Return row-aligned integer labels / 返回与行对齐的整数标签。"""

        return np.asarray([identity.label for identity in self.identities], dtype=np.int64)


def _validate_identities(identities: list[SampleIdentity] | tuple[SampleIdentity, ...],
                         size: int) -> tuple[SampleIdentity, ...]:
    """Freeze and validate identity rows / 冻结并校验身份行。"""

    frozen = tuple(identities)
    if len(frozen) != size:
        raise ValueError("one SampleIdentity is required per sample")
    return frozen


class RawWindowDataset(_IdentityDataset):
    """Raw fixed-length windows ``[N,C,T]`` / 原始定长窗口数据集。"""

    representation_mode = "raw"

    def __init__(
        self,
        values: np.ndarray,
        identities: list[SampleIdentity] | tuple[SampleIdentity, ...],
        sample_mask: np.ndarray | None = None,
        channel_schema: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 3 or not np.isfinite(array).all():
            raise ValueError("raw values must be finite [sample,channel,time]")
        self.values = array
        self.identities = _validate_identities(identities, array.shape[0])
        if channel_schema is None:
            self.channel_schema = ()
        else:
            schema = tuple(str(value) for value in channel_schema)
            if len(schema) != array.shape[1] or len(schema) != len(set(schema)):
                raise ValueError("channel_schema must uniquely name every raw input channel")
            self.channel_schema = schema
        if sample_mask is None:
            self.sample_mask = np.ones((array.shape[0], array.shape[2]), dtype=bool)
        else:
            mask = np.asarray(sample_mask, dtype=bool)
            if mask.shape != (array.shape[0], array.shape[2]):
                raise ValueError("sample_mask must be [sample,time]")
            self.sample_mask = mask

    def __len__(self) -> int:
        return self.values.shape[0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        _require_torch()
        return {
            "x": torch.from_numpy(self.values[index]),
            "mask": torch.from_numpy(self.sample_mask[index]),
            "y": torch.tensor(self.identities[index].label, dtype=torch.long),
            "identity": self.identities[index],
        }


class FeatureVectorDataset(_IdentityDataset):
    """One engineered feature vector per file / 每个文件一个工程特征向量。"""

    representation_mode = "feature_vector"

    def __init__(
        self,
        values: np.ndarray,
        feature_names: list[str] | tuple[str, ...],
        identities: list[SampleIdentity] | tuple[SampleIdentity, ...],
    ) -> None:
        array = np.asarray(values, dtype=np.float32)
        names = tuple(feature_names)
        if array.ndim != 2 or array.shape[1] != len(names):
            raise ValueError("feature values must be [file,feature] matching feature_names")
        if len(names) != len(set(names)):
            raise ValueError("feature_names must be unique")
        self.values = array
        self.feature_names = names
        self.identities = _validate_identities(identities, array.shape[0])

    def __len__(self) -> int:
        return self.values.shape[0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        _require_torch()
        return {
            "x": torch.from_numpy(self.values[index]),
            "y": torch.tensor(self.identities[index].label, dtype=torch.long),
            "identity": self.identities[index],
        }


class FeatureMatrixDataset(_IdentityDataset):
    """Ordered ``[N,D,K]`` matrix with a required column mask.

    带必需列掩码的有序 ``[N,D,K]`` 特征矩阵数据集。
    """

    representation_mode = "feature_matrix"

    @classmethod
    def from_contracts(
        cls,
        matrices: list[OrderedFeatureMatrixV1] | tuple[OrderedFeatureMatrixV1, ...],
        identities: list[SampleIdentity] | tuple[SampleIdentity, ...],
    ) -> "FeatureMatrixDataset":
        """Build only from schema-compatible OrderedFeatureMatrixV1 objects.

        只从 schema 完全兼容的 OrderedFeatureMatrixV1 对象构造批次。
        """

        frozen = tuple(matrices)
        if not frozen:
            raise ValueError("at least one OrderedFeatureMatrixV1 is required")
        from ..representations.feature_matrix import validate_feature_matrix

        for matrix in frozen:
            validate_feature_matrix(matrix)
        reference = frozen[0]
        for matrix in frozen:
            if (tuple(matrix.channel_schema) != tuple(reference.channel_schema)
                    or tuple(matrix.context_schema) != tuple(reference.context_schema)
                    or matrix.schema_version != reference.schema_version):
                raise ValueError("OrderedFeatureMatrixV1 schemas differ within a dataset")
        return cls(
            tuple(np.asarray(matrix.values, dtype=np.float32) for matrix in frozen),
            tuple(np.asarray(matrix.row_mask, dtype=bool) for matrix in frozen),
            identities,
            reference.channel_schema,
        )

    def __init__(
        self,
        values: np.ndarray | tuple[np.ndarray, ...] | list[np.ndarray],
        row_mask: np.ndarray | tuple[np.ndarray, ...] | list[np.ndarray],
        identities: list[SampleIdentity] | tuple[SampleIdentity, ...],
        channel_schema: list[str] | tuple[str, ...],
    ) -> None:
        if isinstance(values, np.ndarray):
            if values.ndim != 3:
                raise ValueError("matrix values must be [file,D,K] or variable [D,K_i]")
            matrices = tuple(np.asarray(value, dtype=np.float32) for value in values)
        else:
            matrices = tuple(np.asarray(value, dtype=np.float32) for value in values)
        if isinstance(row_mask, np.ndarray):
            if row_mask.ndim != 2:
                raise ValueError("matrix row_mask must be [file,K] or variable [K_i]")
            masks = tuple(np.asarray(value, dtype=bool) for value in row_mask)
        else:
            masks = tuple(np.asarray(value, dtype=bool) for value in row_mask)
        if not matrices or len(matrices) != len(masks):
            raise ValueError("matrix values and row masks must be non-empty and aligned")
        channel_count = matrices[0].shape[0] if matrices[0].ndim == 2 else -1
        if any(matrix.ndim != 2 or matrix.shape[0] != channel_count or mask.shape != (matrix.shape[1], )
               or not np.isfinite(matrix).all() or not np.any(mask) for matrix, mask in zip(matrices, masks)):
            raise ValueError("every matrix must be finite [D,K_i] with a valid column")
        schema = tuple(channel_schema)
        if len(schema) != channel_count or len(schema) != len(set(schema)):
            raise ValueError("channel_schema must uniquely name every matrix channel")
        self.values = matrices
        self.row_mask = masks
        self.n_channels = channel_count
        self.sequence_lengths = tuple(matrix.shape[1] for matrix in matrices)
        self.channel_schema = schema
        self.identities = _validate_identities(identities, len(matrices))

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> dict[str, Any]:
        _require_torch()
        return {
            "x": torch.from_numpy(self.values[index]),
            "mask": torch.from_numpy(self.row_mask[index]),
            "y": torch.tensor(self.identities[index].label, dtype=torch.long),
            "identity": self.identities[index],
        }


class FileBagDataset(_IdentityDataset):
    """Variable window bags plus exactly one feature vector per file.

    可变长度窗口袋，并且每个文件严格只有一个特征向量。文件特征存储结构不含窗口维。
    """

    representation_mode = "fusion"

    def __init__(
        self,
        window_bags: list[np.ndarray] | tuple[np.ndarray, ...],
        file_features: np.ndarray,
        identities: list[SampleIdentity] | tuple[SampleIdentity, ...],
        sample_masks: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
    ) -> None:
        bags = tuple(np.asarray(bag, dtype=np.float32) for bag in window_bags)
        features = np.asarray(file_features, dtype=np.float32)
        if not bags or any(bag.ndim != 3 or bag.shape[0] == 0 for bag in bags):
            raise ValueError("each bag must be finite [window,channel,time] with at least one window")
        if any(not np.isfinite(bag).all() for bag in bags):
            raise ValueError("window bags must be finite")
        shape = bags[0].shape[1:]
        if any(bag.shape[1:] != shape for bag in bags):
            raise ValueError("all windows must share channel and time dimensions")
        if features.ndim != 2 or features.shape[0] != len(bags):
            raise ValueError("file_features must be [file,feature] without a window dimension")
        self.window_bags = bags
        self.file_features = features
        self.identities = _validate_identities(identities, len(bags))
        if sample_masks is None:
            self.sample_masks = tuple(np.ones((bag.shape[0], bag.shape[2]), dtype=bool) for bag in bags)
        else:
            masks = tuple(np.asarray(mask, dtype=bool) for mask in sample_masks)
            if len(masks) != len(bags) or any(mask.shape != (bag.shape[0], bag.shape[2])
                                              for mask, bag in zip(masks, bags)):
                raise ValueError("each sample mask must be [window,time] for its bag")
            self.sample_masks = masks

    def __len__(self) -> int:
        return len(self.window_bags)

    def __getitem__(self, index: int) -> dict[str, Any]:
        _require_torch()
        return {
            "window_bag": torch.from_numpy(self.window_bags[index]),
            "sample_mask": torch.from_numpy(self.sample_masks[index]),
            "file_features": torch.from_numpy(self.file_features[index]),
            "y": torch.tensor(self.identities[index].label, dtype=torch.long),
            "identity": self.identities[index],
        }


def collate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate all four representations without losing identities.

    汇集四种 representation，同时保留身份元数据。
    """

    _require_torch()
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    identities = [sample["identity"] for sample in samples]
    labels = torch.stack([sample["y"] for sample in samples])
    if "window_bag" not in samples[0]:
        if ("mask" in samples[0] and samples[0]["x"].ndim == 2
                and any(sample["x"].shape[-1] != samples[0]["x"].shape[-1] for sample in samples)):
            channels = int(samples[0]["x"].shape[0])
            maximum_length = max(int(sample["x"].shape[-1]) for sample in samples)
            values = torch.zeros((len(samples), channels, maximum_length), dtype=torch.float32)
            mask = torch.zeros((len(samples), maximum_length), dtype=torch.bool)
            for index, sample in enumerate(samples):
                if int(sample["x"].shape[0]) != channels:
                    raise ValueError("feature-matrix channels differ within a batch")
                length = int(sample["x"].shape[-1])
                values[index, :, :length] = sample["x"]
                mask[index, :length] = sample["mask"]
            return {
                "x": values,
                "mask": mask,
                "y": labels,
                "identities": identities,
            }
        result: dict[str, Any] = {
            "x": torch.stack([sample["x"] for sample in samples]),
            "y": labels,
            "identities": identities,
        }
        if "mask" in samples[0]:
            result["mask"] = torch.stack([sample["mask"] for sample in samples])
        return result

    maximum_windows = max(sample["window_bag"].shape[0] for sample in samples)
    channels, time = samples[0]["window_bag"].shape[1:]
    window_bag = torch.zeros((len(samples), maximum_windows, channels, time), dtype=torch.float32)
    window_mask = torch.zeros((len(samples), maximum_windows), dtype=torch.bool)
    sample_mask = torch.zeros((len(samples), maximum_windows, time), dtype=torch.bool)
    for index, sample in enumerate(samples):
        count = sample["window_bag"].shape[0]
        window_bag[index, :count] = sample["window_bag"]
        window_mask[index, :count] = True
        sample_mask[index, :count] = sample["sample_mask"]
    return {
        "window_bag": window_bag,
        "window_mask": window_mask,
        "sample_mask": sample_mask,
        "file_features": torch.stack([sample["file_features"] for sample in samples]),
        "y": labels,
        "identities": identities,
    }
