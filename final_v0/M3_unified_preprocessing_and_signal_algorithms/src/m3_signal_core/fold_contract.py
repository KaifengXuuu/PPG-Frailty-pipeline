"""M2 物化 fold 与拟合 artifact 绑定 / Bind fitted artifacts to M2 folds.

中文：调用方不能仅声明 fit_role=training；本模块读取 M2 corrected registry，
强制拟合数据的 subject 集合与指定训练折完全一致，并拒绝任何 OOF subject。

English: A caller declaration is insufficient. This module reads the materialized
M2 corrected registry, requires the exact training-subject roster, and rejects every
OOF subject before any statistic is fitted.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .registry import get_profile, load_registry, registry_sha256
from .scaling import FoldScaler


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
M2_ROOT = REPOSITORY_ROOT / "final_v0/M2_data_manifest_and_evaluation_protocol"
M2_FOLD_REGISTRY = M2_ROOT / "splits/frailty3_future_corrected_sgkf5_v2.json"
M2_FILE_MANIFEST = M2_ROOT / "manifests/frailty3_file_manifest.csv"
M2_PROTOCOL_REGISTRY = M2_ROOT / "registries/protocol_registry.json"


def _sha256_file(path: Path) -> str:
    """逐字节 hash / Hash every byte."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ids_hash(values: Sequence[str]) -> str:
    """稳定 subject hash / Stable subject hash."""

    payload = "\n".join(sorted({str(value) for value in values})).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_hash(value: Any) -> str:
    """哈希排序 strict-JSON payload / Hash a canonical strict-JSON payload."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_m2_fold(repeat_index: int, fold_index: int) -> dict[str, Any]:
    """解析唯一 corrected fold / Resolve one corrected fold."""

    registry = json.loads(M2_FOLD_REGISTRY.read_text(encoding="utf-8"))
    if registry["registry_id"] != "frailty3_future_corrected_sgkf5_v2":
        raise ValueError("unexpected_fold_registry")
    repeat = next(
        (item for item in registry["repeats"] if item["repeat_index"] == int(repeat_index)),
        None,
    )
    if repeat is None:
        raise KeyError(f"repeat_index_not_found:{repeat_index}")
    fold = next(
        (item for item in repeat["folds"] if item["fold_index"] == int(fold_index)),
        None,
    )
    if fold is None:
        raise KeyError(f"fold_index_not_found:{fold_index}")
    train = set(fold["train_subject_ids"])
    oof = set(fold["oof_validation_subject_ids"])
    # 中文：subject_input_order 是排序规则字符串，不是 roster；物化 union 应为 29。
    # English: subject_input_order names the sort rule; it is not the subject roster.
    if train & oof or len(train | oof) != int(registry["n_subjects"]):
        raise ValueError("materialized_fold_invariants_failed")
    return {
        "dataset_version_id": registry["dataset_version_id"],
        "fold_registry_id": registry["registry_id"],
        "fold_registry_payload_sha256": registry["payload_sha256"],
        "repeat_index": int(repeat_index),
        "split_seed": int(repeat["split_seed"]),
        "fold_index": int(fold_index),
        "training_seed": int(fold["training_seed"]),
        "train_subject_ids": sorted(train),
        "oof_validation_subject_ids": sorted(oof),
    }


def fit_fold_scaler(
    values: np.ndarray,
    sample_subject_ids: Sequence[str],
    feature_names: Sequence[str],
    *,
    repeat_index: int,
    fold_index: int,
    preprocessing_profile_ids: Sequence[str],
    method: str = "robust",
    clip: float | None = None,
) -> tuple[FoldScaler, dict[str, Any]]:
    """在精确 M2 训练 roster 上拟合 / Fit on the exact M2 training roster."""

    # 中文：D4 future-active hybrid view 固定 RobustScaler 且不裁剪。
    # English: D4 freezes an unclipped RobustScaler for the future-active hybrid view.
    if method != "robust" or clip is not None:
        raise ValueError("future_active_fold_scaler_requires_robust_no_clip")
    profile_ids = [str(value) for value in preprocessing_profile_ids]
    if not profile_ids or len(set(profile_ids)) != len(profile_ids):
        raise ValueError("preprocessing_profile_ids_must_be_nonempty_unique")
    resolved_profiles = [get_profile(profile_id) for profile_id in profile_ids]
    if any(
        not str(profile.get("status", "")).startswith("future_active")
        for profile in resolved_profiles
    ):
        raise ValueError("historical_or_deprecated_profile_cannot_fit_future_artifact")
    matrix = np.asarray(values, dtype=np.float64)
    subjects = [str(value) for value in sample_subject_ids]
    names = [str(value) for value in feature_names]
    if matrix.ndim != 2 or matrix.shape[0] != len(subjects):
        raise ValueError("sample_subject_ids_must_align_with_rows")
    if matrix.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError("feature_names_must_be_unique_and_match_columns")
    fold = resolve_m2_fold(repeat_index, fold_index)
    observed = set(subjects)
    train = set(fold["train_subject_ids"])
    oof = set(fold["oof_validation_subject_ids"])
    if observed & oof:
        raise ValueError("oof_subject_present_in_fit")
    if observed != train:
        missing = sorted(train - observed)
        extra = sorted(observed - train)
        raise ValueError(f"training_roster_mismatch:missing={missing},extra={extra}")
    scaler = FoldScaler(method=method, clip=clip).fit(
        matrix,
        fit_role="training",
        training_ids=fold["train_subject_ids"],
    )
    protocol_registry = json.loads(M2_PROTOCOL_REGISTRY.read_text(encoding="utf-8"))
    if protocol_registry["active_protocol_id"] != "frailty3_fixed_epoch_oof_v2_corrected_sgkf":
        raise ValueError("unexpected_active_m2_protocol")
    transformers = [
        {
            "transformer_id": "training_median_then_robust_iqr_v1",
            "method": "robust_median_iqr",
            "stage": "channel_sequence",
            "feature_names": names,
            "fit_dtype": "float64",
            "parameters": {
                "center": scaler.center.tolist(),
                "scale": scaler.scale.tolist(),
                "impute_value": scaler.impute_values.tolist(),
                "zero_scale_mask": scaler.zero_scale_mask.tolist(),
            },
        }
    ]
    registry = load_registry()
    artifact = {
        "schema_version": "m3.fold_fitted_artifact.v1",
        "artifact_id": (
            f"m3_scaler_r{repeat_index}_f{fold_index}_"
            f"{_ids_hash(fold['train_subject_ids'])[:12]}"
        ),
        "dataset_version_id": fold["dataset_version_id"],
        "dataset_manifest_sha256": _sha256_file(M2_FILE_MANIFEST),
        "fold_registry_id": fold["fold_registry_id"],
        "fold_registry_sha256": _sha256_file(M2_FOLD_REGISTRY),
        "fold_registry_payload_sha256": fold["fold_registry_payload_sha256"],
        "protocol_id": protocol_registry["active_protocol_id"],
        "status": "locked",
        "fit_scope": "training_subjects_only",
        "repeat_index": fold["repeat_index"],
        "split_seed": fold["split_seed"],
        "fold_index": fold["fold_index"],
        "training_seed": fold["training_seed"],
        "subject_partition": {
            "train_subject_ids": fold["train_subject_ids"],
            "oof_validation_subject_ids": fold["oof_validation_subject_ids"],
        },
        "feature_schema_version": "m3_raw8_dynamic_sequence.v1",
        "preprocessing_registry_id": registry["registry_id"],
        "preprocessing_profile_ids": profile_ids,
        "preprocessing_profile_payload_sha256": {
            profile["profile_id"]: _canonical_json_hash(profile)
            for profile in resolved_profiles
        },
        "transformers": transformers,
        "parameters_sha256": _canonical_json_hash(transformers),
        "provenance": {
            "producer_callable": "m3_signal_core.fit_fold_scaler",
            "producer_source_path": (
                "final_v0/M3_unified_preprocessing_and_signal_algorithms/"
                "src/m3_signal_core/fold_contract.py"
            ),
            "producer_source_sha256": _sha256_file(Path(__file__)),
            "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
        "preprocessing_registry_sha256": registry_sha256(),
    }
    return scaler, artifact


__all__ = ["fit_fold_scaler", "resolve_m2_fold"]
