"""Frozen model-input bundle adapters and participant aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from ..training.aggregation import (
    LINE_A_EQUAL_FILES,
    LINE_B_EQUAL_ROLE_FAMILIES,
    canonical_role_family,
)
from ..training.bundle import LoadedBundle, load_bundle, predict_bundle_raw


@runtime_checkable
class BundleModelInputAdapter(Protocol):
    """Serializable adapter at the already-frozen model-input boundary."""

    representation_mode: str
    input_schema_hash: str
    allowed_role_families: tuple[str, ...]
    boundary: str

    def transform_record(self, record: Any) -> Mapping[str, np.ndarray]:
        ...


@dataclass(frozen=True)
class FrozenModelInputAdapter:
    """Validate and batch one already-preprocessed file-level model input.

    This does not invent device preprocessing. Raw-device-to-model-input remains
    fail-closed until the deployment source format is frozen (V2-026).
    """

    representation_mode: str
    input_schema_hash: str
    allowed_role_families: tuple[str, ...] = ("B", "R")
    adapter_version: str = "frozen_model_input_adapter_v2"
    boundary: str = "already_preprocessed_file_record_to_model_input"

    def __post_init__(self) -> None:
        if self.representation_mode not in {
            "raw", "feature_vector", "feature_matrix", "fusion"
        }:
            raise ValueError("unsupported representation_mode")
        canonical_roles = tuple(
            canonical_role_family(value) for value in self.allowed_role_families
        )
        if not canonical_roles or len(canonical_roles) != len(set(canonical_roles)):
            raise ValueError(
                "adapter allowed_role_families must be non-empty and unique"
            )
        object.__setattr__(self, "allowed_role_families", canonical_roles)
        digest = str(self.input_schema_hash)
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("input_schema_hash must be a lowercase SHA-256 digest")

    @staticmethod
    def _finite(value: Any, name: str) -> np.ndarray:
        array = np.asarray(value)
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
        return array

    def transform_record(self, record: Any) -> dict[str, np.ndarray]:
        if not isinstance(record, Mapping):
            raise TypeError(
                "deployment source preprocessing is unresolved; provide an already-"
                "preprocessed model-input mapping"
            )
        values = dict(record)
        if self.representation_mode == "raw":
            allowed = {"x", "mask"}
            if set(values) - allowed or "x" not in values:
                raise ValueError("raw model input requires x and optional mask only")
            x = self._finite(values["x"], "x")
            if x.ndim != 3:
                raise ValueError("raw file input x must be [window,channel,time]")
            output = {"x": x.astype(np.float32, copy=False)}
            if "mask" in values:
                mask = np.asarray(values["mask"], dtype=bool)
                if mask.shape != (x.shape[0], x.shape[2]):
                    raise ValueError("raw mask must be [window,time]")
                output["mask"] = mask
            return output
        if self.representation_mode == "feature_vector":
            if set(values) != {"x"}:
                raise ValueError("feature_vector file input requires exactly x")
            x = np.asarray(values["x"])
            if x.ndim != 1:
                raise ValueError("feature_vector file input x must be [feature]")
            if np.isinf(x).any():
                raise ValueError(
                    "feature_vector x may use NaN for unavailable physiology but not infinity"
                )
            return {"x": x.astype(np.float64, copy=False)[None, :]}
        if self.representation_mode == "feature_matrix":
            if set(values) != {"x", "mask"}:
                raise ValueError("feature_matrix file input requires x and mask")
            x = self._finite(values["x"], "x")
            mask = np.asarray(values["mask"], dtype=bool)
            if x.ndim != 2 or mask.shape != (x.shape[1],):
                raise ValueError("matrix input requires x [channel,column], mask [column]")
            return {
                "x": x.astype(np.float32, copy=False)[None, :, :],
                "mask": mask[None, :],
            }
        required = {"window_bag", "window_mask", "file_features"}
        if not required <= set(values) or set(values) - (required | {"sample_mask"}):
            raise ValueError(
                "fusion file input requires window_bag, window_mask, file_features "
                "and optional sample_mask"
            )
        bag = self._finite(values["window_bag"], "window_bag")
        window_mask = np.asarray(values["window_mask"], dtype=bool)
        features = self._finite(values["file_features"], "file_features")
        if (
            bag.ndim != 3
            or window_mask.shape != (bag.shape[0],)
            or features.ndim != 1
        ):
            raise ValueError("fusion input shapes must be [window,C,T], [window], [feature]")
        output = {
            "window_bag": bag.astype(np.float32, copy=False)[None, :, :, :],
            "window_mask": window_mask[None, :],
            "file_features": features.astype(np.float32, copy=False)[None, :],
        }
        if "sample_mask" in values:
            sample_mask = np.asarray(values["sample_mask"], dtype=bool)
            if sample_mask.shape != (bag.shape[0], bag.shape[2]):
                raise ValueError("fusion sample_mask must be [window,time]")
            output["sample_mask"] = sample_mask[None, :, :]
        return output


def build_model_input_adapter(
    representation_mode: str,
    *,
    input_schema_hash: str,
    allowed_role_families: tuple[str, ...] = ("B", "R"),
) -> FrozenModelInputAdapter:
    """Construct the concrete serializable adapter for one frozen representation."""

    return FrozenModelInputAdapter(
        representation_mode=str(representation_mode),
        input_schema_hash=str(input_schema_hash),
        allowed_role_families=tuple(allowed_role_families),
    )


def infer_raw_record(
    bundle: LoadedBundle | str | Path,
    record: Any,
) -> dict[str, np.ndarray]:
    """Run bundled adapter -> model -> equal-window file probability."""

    loaded = load_bundle(bundle) if isinstance(bundle, (str, Path)) else bundle
    probabilities = np.asarray(predict_bundle_raw(loaded, record), dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise RuntimeError("bundle adapter must yield at least one [row,class] probability")
    file_probability = probabilities.mean(axis=0)
    total = float(file_probability.sum())
    if not np.isfinite(file_probability).all() or not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("bundle inference produced an invalid file probability")
    file_probability /= total
    return {"window_probabilities": probabilities, "file_probability": file_probability}


@dataclass(frozen=True)
class ParticipantFileInput:
    file_id: str
    role: str
    record: Any

    def __post_init__(self) -> None:
        if not self.file_id.strip() or not self.role.strip():
            raise ValueError("participant file_id and role are required")


def infer_participant(
    bundle: LoadedBundle | str | Path,
    files: tuple[ParticipantFileInput, ...] | list[ParticipantFileInput],
    *,
    balance_line: str | None = None,
) -> dict[str, Any]:
    """Apply archived Line A or Line B aggregation across deployment files."""

    loaded = load_bundle(bundle) if isinstance(bundle, (str, Path)) else bundle
    frozen = tuple(files)
    if not frozen or len({item.file_id for item in frozen}) != len(frozen):
        raise ValueError("participant inference requires non-empty unique files")
    archived = loaded.manifest.get("metadata", {}).get("aggregation_rule")
    selected_line = str(archived if balance_line is None else balance_line)
    if selected_line not in {LINE_A_EQUAL_FILES, LINE_B_EQUAL_ROLE_FAMILIES}:
        raise RuntimeError(
            "bundle aggregation_rule is not an executable V2 Line A/Line B identity"
        )
    if balance_line is not None and archived != selected_line:
        raise ValueError("requested balance_line differs from archived bundle aggregation")
    file_probability: dict[str, np.ndarray] = {}
    role_by_file: dict[str, str] = {}
    adapter = loaded.pipeline_adapter
    if adapter is None or not hasattr(adapter, "allowed_role_families"):
        raise RuntimeError("bundle adapter does not declare allowed_role_families")
    allowed_roles = tuple(str(value) for value in adapter.allowed_role_families)
    if not allowed_roles or len(allowed_roles) != len(set(allowed_roles)):
        raise RuntimeError(
            "bundle adapter role-family contract must be non-empty and unique"
        )
    for item in frozen:
        family = canonical_role_family(item.role)
        if family not in allowed_roles:
            raise ValueError(
                f"role family {family!r} is outside this bundle training scope "
                f"{','.join(allowed_roles)}"
            )
        prediction = infer_raw_record(loaded, item.record)
        file_probability[item.file_id] = prediction["file_probability"]
        role_by_file[item.file_id] = family
    matrix = np.asarray(list(file_probability.values()), dtype=np.float64)
    role_probability: dict[str, np.ndarray] = {}
    if selected_line == LINE_A_EQUAL_FILES:
        participant_probability = matrix.mean(axis=0)
    else:
        for family in sorted(set(role_by_file.values())):
            selected = [
                file_probability[file_id]
                for file_id, role in role_by_file.items()
                if role == family
            ]
            role_probability[family] = np.asarray(selected).mean(axis=0)
        participant_probability = np.asarray(list(role_probability.values())).mean(axis=0)
    participant_probability /= participant_probability.sum()
    return {
        "balance_line": selected_line,
        "file_probabilities": file_probability,
        "role_family_probabilities": role_probability,
        "participant_probability": participant_probability,
    }


__all__ = [
    "BundleModelInputAdapter",
    "FrozenModelInputAdapter",
    "ParticipantFileInput",
    "build_model_input_adapter",
    "infer_participant",
    "infer_raw_record",
]
