"""Persist and select reloadable outer-fold model checkpoints.

The numerical pipeline hands this module an already fitted model and the exact
model-ready input boundary.  Saving is therefore an output side effect only:
it never refits a model, changes a split, or feeds held-out labels back into
training.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

import numpy as np

from ..models import ModelInputSpec, normalize_model_config
from ..provenance import sha256_file, stable_payload_sha256
from ..training.bundle import current_runtime_environment, save_bundle


FOLD_CHECKPOINT_SCHEMA = "ppg_frailty.v5_fold_checkpoint.v1"
MEDIAN_SELECTION_SCHEMA = "ppg_frailty.v5_median_fold_selection.v1"

@dataclass(frozen=True)
class FoldCheckpointPayload:
    """In-memory values needed to serialize one fitted outer-fold model."""

    model: Any
    model_config: Mapping[str, Any]
    input_spec: ModelInputSpec | Mapping[str, Any]
    golden_inputs: Mapping[str, Any]
    pipeline_config: Mapping[str, Any]
    cell_summary: Mapping[str, Any]

def _explicit(value: Any, *, reason: str) -> Any:
    """Use a structured not-applicable value where the bundle schema requires one."""

    return (
        value
        if value not in (None, {}, [], (), "")
        else {
            "status": "not_applicable",
            "reason": reason,
        }
    )

def _bundle_metadata(payload: FoldCheckpointPayload) -> dict[str, Any]:
    config = dict(payload.pipeline_config)
    summary = dict(payload.cell_summary)
    spec = ModelInputSpec.from_value(payload.input_spec)
    model_config = normalize_model_config(payload.model_config)
    model_section = dict(config.get("model", {}))
    training = dict(config.get("training", {}))
    aggregation = dict(config.get("aggregation", {}))
    signal = dict(config.get("signal", {}))
    manifest = dict(config.get("manifest", {}))
    splits = dict(config.get("splits", {}))
    features = dict(config.get("features", {}))
    windows = dict(config.get("windows", {}))
    quality = dict(config.get("quality", {}))
    artifact = dict(config.get("artifact", {}))
    fitted_provenance = dict(summary.get("fitted_provenance", {}))
    representation = str(summary["representation_mode"])
    channels = tuple(spec.channel_schema)
    if not channels:
        channels = tuple(str(value) for value in manifest.get("channel_order", ()))
    if not channels:
        channels = ("dense_model_input",)
    config_hash = str(summary["config_hash"])
    state_hash = str(summary["model_hash"])
    balance_hash = stable_payload_sha256(
        {
            "training_balance": training.get("training_balance"),
            "aggregation": aggregation,
            "classifier_role_families": training.get("classifier_role_families", ()),
        }
    )
    run_hash = stable_payload_sha256(
        {
            "config_hash": config_hash,
            "repeat": int(summary["repeat_index"]),
            "fold": int(summary["fold_index"]),
            "model_state_hash": state_hash,
        }
    )
    feature_names = tuple(spec.feature_names)
    representation_provenance = _explicit(
        summary.get("representation_transform_provenance"),
        reason="no external fold transform for this representation",
    )
    fitted = {
        "name": "trained_classifier",
        "kind": "outer_train_only_fitted_model",
        "provenance": summary.get("fitted_provenance", {}),
    }
    selected_window = (
        windows.get("engineering") if representation in {"feature_vector", "feature_matrix"} else windows.get("raw_dl")
    )
    return {
        "model_identity": {
            "name": str(model_config["canonical_model_name"]),
            "machine_id": str(model_config["model_id"]),
            "version": str(model_section.get("variant", "registered_v2")),
        },
        "representation_mode": representation,
        "signal_route": {
            "quality_mode": str(summary.get("quality_mode", quality.get("mode", "off"))),
            "artifact_reducer": str(artifact.get("reducer", "identity")),
            "boundary": "model_ready_input_after_fold_local_preprocessing",
        },
        "class_order": list(summary.get("class_order", (0, 1, 2))),
        "channel_schema": list(channels),
        "preprocessing": {
            "signal": signal,
            "quality": quality,
            "artifact": artifact,
            "representation_transform_provenance": representation_provenance,
            "fit_scope": "outer_train_participants_only",
        },
        "preprocessing_hash": str(summary["preprocessing_hash"]),
        "resampling": {
            "canonical_fs_hz": signal.get("internal_fs_hz"),
            "dl_input": signal.get("dl_resampling", {"enabled": False}),
        },
        "window_plan": {
            "selected": _explicit(selected_window, reason="model has no window input"),
            "shared_planner_version": windows.get("shared_planner_version", "not_declared"),
        },
        "feature_registry": {
            "registry_id": features.get("registry_id", "not_model_input"),
            "enabled_groups": features.get("enabled_groups", []),
        },
        "feature_hash": str(summary["feature_hash"]),
        "feature_vector_schema": (
            {"status": "model_input", "feature_names": list(feature_names)}
            if representation == "feature_vector"
            else {"status": "not_model_input", "reason": f"representation={representation}"}
        ),
        "ordered_matrix_schema": (
            {"status": "model_input", "channel_schema": list(channels)}
            if representation == "feature_matrix"
            else {"status": "not_model_input", "reason": f"representation={representation}"}
        ),
        "mask_semantics": {
            "boundary": "model_ready_input",
            "mask_required": representation in {"raw", "feature_matrix", "fusion"},
        },
        "validity_policy": {
            "quality_mode": quality.get("mode", "off"),
            "failure_action": quality.get("failure_action", "fail_closed"),
        },
        "fitted_objects": [fitted],
        "representation_state": {
            "provenance": representation_provenance,
            "input_boundary": "already_preprocessed_fold_model_input",
        },
        "pooling_rule": {
            "model_pooling": model_section.get("pooling", model_section.get("mask_aware_pooling", "model_defined")),
            "window_to_file": aggregation.get("window_to_file", "ordinary_mean"),
        },
        "aggregation_rule": str(summary.get("balance_line", aggregation.get("balance_line"))),
        "manifest_hash": str(summary.get("manifest_hash", manifest.get("source_manifest_sha256", "not_declared"))),
        "fold_hash": str(
            summary.get(
                "fold_hash",
                fitted_provenance.get("fold_hash", splits.get("source_registry_payload_sha256", "not_declared")),
            )
        ),
        "manifest_version": str(manifest.get("manifest_version", "not_declared")),
        "fold_registry_version": str(splits.get("registry_id", "not_declared")),
        "pipeline_generation": "final_pipeline_v2",
        "config_hash": config_hash,
        "balance_hash": balance_hash,
        "run_hash": run_hash,
        "source_snapshot_hash": str(summary.get("source_snapshot_hash", summary["source_version"])),
        "code_version": str(summary.get("code_commit", "not_git_bound")),
        "environment": current_runtime_environment(),
        "dependency_status": {
            "status": "captured_at_checkpoint_write",
            "exact_environment_gate": "recorded_separately_by_v5_entrypoint",
        },
        "serialization_trust": {
            "trusted_local_only": True,
            "authenticated_signature": False,
            "integrity": "hashes_plus_reload_golden_prediction_parity",
        },
        "golden_case": {
            "boundary": "already_preprocessed_fold_model_input",
            "source_identity": {
                "repeat": int(summary["repeat_index"]),
                "fold": int(summary["fold_index"]),
                "scope": "first_outer_train_model_input",
            },
            "expected_output": "three_class_probability",
        },
    }

def save_fold_checkpoint(directory: str | Path, payload: FoldCheckpointPayload) -> dict[str, Any]:
    """Save, reload, and parity-check one model-ready outer-fold bundle."""

    target = Path(directory)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng: Any = None
    cuda_rng: Any = None
    try:
        try:
            import torch

            torch_rng = torch.random.get_rng_state()
            if torch.cuda.is_available():
                cuda_rng = torch.cuda.get_rng_state_all()
        except ImportError:  # pragma: no cover - estimator-only environment.
            torch = None  # type: ignore[assignment]
        save_bundle(
            payload.model,
            target,
            model_config=payload.model_config,
            input_spec=payload.input_spec,
            metadata=_bundle_metadata(payload),
            golden_inputs=payload.golden_inputs,
        )
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        if torch_rng is not None:
            torch.random.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state_file = str(manifest["state_file"])
    return {
        "schema_version": FOLD_CHECKPOINT_SCHEMA,
        "purpose": "research_outer_fold_model_for_replay_and_dashboard_trial",
        "deployment_status": "not_final_refit_outer_train_subset_only",
        "selection_metric_use": "outer_oof_balanced_accuracy_used_only_after_all_folds_finish",
        "model_input_boundary": "already_preprocessed_fold_model_input",
        "manifest_path": "model_checkpoint/manifest.json",
        "manifest_sha256": sha256_file(manifest_path),
        "state_file": f"model_checkpoint/{state_file}",
        "state_sha256": str(manifest["file_hashes"][state_file]),
        "golden_parity_atol": float(manifest["golden_parity_atol"]),
    }

def select_median_fold(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Choose the lower middle ordered fold; ties use repeat then fold."""

    eligible: list[tuple[float, int, int, Mapping[str, Any]]] = []
    for row in rows:
        try:
            score = float(row["balanced_accuracy"])
            repeat = int(row["repeat"])
            fold = int(row["fold"])
        except (KeyError, TypeError, ValueError):
            continue
        checkpoint = str(row.get("learned_weight_checkpoint", ""))
        if (
            str(row.get("status", "")) != "passed"
            or not math.isfinite(score)
            or not checkpoint
            or checkpoint.startswith("not_")
        ):
            continue
        eligible.append((score, repeat, fold, row))
    if not eligible:
        return None
    ordered = sorted(eligible, key=lambda value: (value[0], value[1], value[2]))
    rank = (len(ordered) - 1) // 2
    score, repeat, fold, row = ordered[rank]
    return {
        "schema_version": MEDIAN_SELECTION_SCHEMA,
        "selection_role": "research_outer_fold_median_for_dashboard_trial",
        "deployment_status": "not_unbiased_deployment_model_outer_train_subset_only",
        "metric": "balanced_accuracy",
        "ordering": "ascending_metric_then_repeat_then_fold",
        "even_count_policy": "lower_middle",
        "eligible_fold_count": len(ordered),
        "selected_rank_zero_based": rank,
        "balanced_accuracy": score,
        "repeat": repeat,
        "fold": fold,
        "checkpoint_manifest": str(row["learned_weight_checkpoint"]),
        "checkpoint_manifest_sha256": str(row.get("checkpoint_manifest_sha256", "")),
        "config_hash": str(row.get("config_hash", "")),
        "model_id": str(row.get("model_id", "")),
    }


__all__ = [
    "FOLD_CHECKPOINT_SCHEMA",
    "FoldCheckpointPayload",
    "MEDIAN_SELECTION_SCHEMA",
    "save_fold_checkpoint",
    "select_median_fold",
]
