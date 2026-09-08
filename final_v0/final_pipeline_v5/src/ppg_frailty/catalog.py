"""Resolve the registered V2 model catalog into canonical pipeline configs.

This module is deliberately materialization-only. It combines a registered
model entry with the matching representation base and the selected Line A/B
aggregation contract, validates the result, and never starts training.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import (
    canonical_json_bytes,
    load_formal_ablation_profiles,
    load_formal_experiment_catalog,
    validate_config_payload,
)

BASE_CONFIG_FILENAMES = {
    "raw": "reference_static_role_aware_v2.yaml",
    "feature_vector": "reference_static_feature_vector_v2.yaml",
    "feature_matrix": "reference_static_feature_matrix_v2.yaml",
    "fusion": "reference_static_fusion_v2.yaml",
}


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return dict(value)


def _load_yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), str(path))


def _line_sections(*, pipeline_root: Path, line: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if line not in {"line_a", "line_b"}:
        raise ValueError("line must be line_a or line_b")
    source = _load_yaml(pipeline_root / "configs" / "reference_static_feature_vector_v2.yaml")
    aggregation = copy.deepcopy(source["aggregation"])
    training = copy.deepcopy(source["training"])
    if line == "line_a":
        training["training_balance"] = "equal_files"
        aggregation.update({
            "balance_line": "line_a_equal_files",
            "hierarchy": ["window", "file", "participant"],
            "window_to_file": "ordinary_mean",
            "file_to_role": "not_applicable",
            "role_to_participant": "not_applicable",
            "missing_role_policy": "not_applicable",
            "quality_weighting": False,
            "direct_all_window_participant_mean": False,
        })
    return training, aggregation


def resolved_catalog_payloads(
    *,
    pipeline_root: str | Path,
    line: str,
    catalog_path: str | Path | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return every active fully resolved and validated catalog configuration."""

    root = Path(pipeline_root).resolve()
    source = (root / "configs" /
              "formal_experiment_catalog_v2.yaml" if catalog_path is None else Path(catalog_path).resolve())
    catalog = load_formal_experiment_catalog(source)
    ablation_catalog = (load_formal_ablation_profiles(root / "configs" /
                                                      "formal_ablation_profiles_v2.yaml") if line == "line_a" else None)
    line_training, line_aggregation = _line_sections(
        pipeline_root=root,
        line=line,
    )
    output: list[dict[str, Any]] = []
    for entry in catalog["entries"]:
        mode = str(entry["representation_mode"])
        base_path = root / "configs" / BASE_CONFIG_FILENAMES[mode]
        base = _load_yaml(base_path)
        payload = copy.deepcopy(base)
        payload["config_id"] = f"{entry['config_stem']}_{line}_v2"
        payload["representation_mode"] = mode
        payload["model"] = copy.deepcopy(entry["model"])
        payload["training"] = copy.deepcopy(line_training)
        payload["aggregation"] = copy.deepcopy(line_aggregation)
        is_ensemble = entry["catalog_role"] == "ensemble_comparison"
        payload["output"]["write_member_oof"] = is_ensemble
        if line == "line_a":
            assert ablation_catalog is not None
            payload["output"]["formal_ablation_materialization"] = {
                "schema_version": "ppg_frailty.formal_ablation_materialization.v2",
                "family": "aggregation_balance",
                "profile_id": "equal_files_line_a_ablation",
                "catalog_role": "ablation",
                "base_config_path": base_path.relative_to(root).as_posix(),
                "base_config_sha256": hashlib.sha256(canonical_json_bytes(base)).hexdigest(),
                "profile_catalog_sha256": ablation_catalog["catalog_sha256"],
                "single_factor_only": True,
                "automatic_execution": False,
                "scientific_execution_completed": False,
            }
        output.append(validate_config_payload(payload))
    expected = len(catalog["entries"])
    if len(output) != expected or len({row["config_id"] for row in output}) != expected:
        raise RuntimeError("formal catalog did not resolve to unique active configs")
    return tuple(output)


__all__ = ["BASE_CONFIG_FILENAMES", "resolved_catalog_payloads"]
