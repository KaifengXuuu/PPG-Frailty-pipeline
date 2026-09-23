from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


pytest.importorskip("pyarrow")
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_v2_v5_outputs", ROOT / "tools/compare_v2_v5_outputs.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _synthetic_cell() -> dict[str, object]:
    fold_hash = _digest("fold")
    model_hash = _digest("model-state")
    fitted = {
        "fold_hash": fold_hash,
        "state_hash": model_hash,
        "training_seed": 42,
        "fitted_participant_ids": ["P01", "P02"],
    }
    return {
        "schema_version": "ppg_frailty.fold_cell.v2",
        "pipeline_generation": "final_pipeline_v2",
        "status": "passed",
        "runner_status": "passed",
        "scientific_scope": "synthetic_numeric_contract",
        "config_id": "synthetic_finalcase",
        "config_hash": _digest("config"),
        "canonical_config_hash": _digest("canonical-config"),
        "repeat_index": 0,
        "fold_index": 0,
        "split_seed": 42,
        "training_seed": 42,
        "training_orchestration_seed": 42,
        "seed_policy": "outer_cv_repeat_seed_equals_split_seed",
        "class_order": [0, 1, 2],
        "representation_mode": "raw",
        "quality_mode": "off",
        "balance_line": "line_b_equal_role_families",
        "preprocessing_hash": _digest("preprocessing"),
        "feature_hash": _digest("features"),
        "model_hash": model_hash,
        "model_id": "SyntheticModel",
        "model_machine_id": "synthetic_model",
        "fitted_provenance": fitted,
        "frozen_model_run_provenance": {
            "fold_hash": fold_hash,
            "architecture_parameters": {"width": 8},
            "random_seeds": [42],
        },
        "frozen_model_run_provenance_hash": _digest("run-provenance"),
        "metrics": {
            "balanced_accuracy": 0.5,
            "macro_f1": 0.4,
            "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
        "dropped_records": [],
        "elapsed_seconds": 10.0,
        "code_commit": "v2-commit",
        "source_version": _digest("v2-source"),
        "operational_metrics": {
            "status": "measured",
            "parameter_count": 123,
            "inference_cost": {
                "cpu_batch1_model_only_p50_ms": 1.0,
                "cpu_batch1_model_only_p95_ms": 2.0,
            },
        },
        "preprocessing_cache_summary": {
            "mode": "read_write",
            "counts": {"raw_windows": {"written": 2}},
            "logical_array_bytes": 2048,
            "elapsed_seconds": 3.0,
            "affects_predictions": False,
        },
    }


def _write_synthetic_metrics_pair(
    root: Path,
) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    v2_fold = root / "v2" / "repeat_00_fold_00"
    v5_fold = root / "v5" / "repeat_00" / "fold_00"
    v2_fold.mkdir(parents=True)
    v5_fold.mkdir(parents=True)
    v2_cell = _synthetic_cell()
    v5_cell = copy.deepcopy(v2_cell)
    v5_cell.update(
        {
            "elapsed_seconds": 99.0,
            "code_commit": "v5-commit",
            "source_version": _digest("v5-source"),
            "operational_metrics": {
                "status": "measured",
                "parameter_count": 123,
                "inference_cost": {
                    "cpu_batch1_model_only_p50_ms": 7.0,
                    "cpu_batch1_model_only_p95_ms": 8.0,
                },
            },
            "preprocessing_cache_summary": {
                "mode": "read_write",
                "counts": {"raw_windows": {"hit": 1}},
                "logical_array_bytes": 1024,
                "elapsed_seconds": 0.2,
                "affects_predictions": False,
            },
        }
    )
    v5_cell["learned_model_checkpoint"] = {
        "manifest_path": "model_checkpoint/manifest.json",
    }
    v2_payload = {
        "schema_version": MODULE._METRICS_SCHEMA,
        "pipeline_generation": "final_pipeline_v2",
        "cells": [v2_cell],
    }
    v5_payload = {
        "schema_version": MODULE._METRICS_SCHEMA,
        "pipeline_generation": "final_pipeline_v2",
        "cells": [v5_cell],
    }
    v2_path = v2_fold / "metrics_per_fold_seed.json"
    v5_path = v5_fold / "metrics_per_fold_seed.json"
    v2_path.write_text(json.dumps(v2_payload), encoding="utf-8")
    v5_path.write_text(json.dumps(v5_payload), encoding="utf-8")
    return v2_path, v5_path, v2_payload, v5_payload


def _write_fold(
    root: Path,
    probability: float,
    *,
    config_hash: str = "same-config",
    confusion: tuple[tuple[int, ...], ...] = ((1, 0), (0, 1)),
) -> None:
    fold = root / "repeat_00_fold_00"
    fold.mkdir(parents=True)
    rows = {
        "participant_id": ["P01"],
        "file_id": ["P01_B"],
        "role": ["B"],
        "label": [1],
        "probabilities": [[0.1, probability, 0.9 - probability]],
        "repeat": [0],
        "fold": [0],
        "split_seed": [42],
        "training_seed": [42],
        "config_hash": [config_hash],
        "retained": [True],
        "level": ["participant"],
        "window_id": [None],
        "member_index": [None],
        "prediction_kind": ["declared"],
        "quality_score": [1.0],
        "aggregation_rule": ["line_b_equal_role_families"],
        "rejection_reason": [None],
    }
    table = pa.Table.from_pydict(rows)
    for name in (
        "oof_window_predictions.parquet",
        "oof_file_predictions.parquet",
        "oof_role_predictions.parquet",
        "oof_subject_predictions.parquet",
        "oof_member_predictions.parquet",
    ):
        pq.write_table(table, fold / name)
    (fold / "run_manifest.json").write_text(
        '{"cell":{"metrics":{"balanced_accuracy":0.5,"confusion_matrix":'
        + str([list(row) for row in confusion]).replace(" ", "")
        + '}}}\n',
        encoding="utf-8",
    )


def test_numeric_comparison_accepts_agreed_tolerance(tmp_path: Path) -> None:
    v2, v5 = tmp_path / "v2", tmp_path / "v5"
    _write_fold(v2, 0.4)
    _write_fold(v5, 0.4000005)
    result = MODULE.compare_outputs(v2, v5, atol=1.0e-6, expected_folds=1)
    assert result["status"] == "passed"
    assert 0.0 < result["maximum_absolute_difference"] <= 1.0e-6


def test_numeric_comparison_rejects_larger_drift(tmp_path: Path) -> None:
    v2, v5 = tmp_path / "v2", tmp_path / "v5"
    _write_fold(v2, 0.4)
    _write_fold(v5, 0.40001)
    result = MODULE.compare_outputs(v2, v5, atol=1.0e-6)
    assert result["status"] == "failed"


def test_numeric_comparison_rejects_semantic_identity_tampering(tmp_path: Path) -> None:
    v2, v5 = tmp_path / "v2", tmp_path / "v5"
    _write_fold(v2, 0.4, config_hash="expected")
    _write_fold(v5, 0.4, config_hash="tampered")
    result = MODULE.compare_outputs(v2, v5, atol=1.0e-6)
    assert result["status"] == "failed"
    assert any(
        "column_differs:config_hash" in row["failures"] for row in result["tables"]
    )


def test_numeric_comparison_rejects_argmax_flip_within_float_tolerance(
    tmp_path: Path,
) -> None:
    v2, v5 = tmp_path / "v2", tmp_path / "v5"
    _write_fold(v2, 0.4500002)
    _write_fold(v5, 0.4499998)
    result = MODULE.compare_outputs(v2, v5, atol=1.0e-6)
    assert result["status"] == "failed"
    assert any("predicted_class_differs" in row["failures"] for row in result["tables"])


def test_numeric_comparison_rejects_confusion_matrix_count_change(
    tmp_path: Path,
) -> None:
    v2, v5 = tmp_path / "v2", tmp_path / "v5"
    _write_fold(v2, 0.4)
    _write_fold(v5, 0.4, confusion=((0, 1), (0, 1)))
    result = MODULE.compare_outputs(v2, v5, atol=1.0e-6)
    assert result["status"] == "failed"
    assert result["metric_failures"][0]["reason"] == "metrics_differ"


def test_numeric_comparison_checks_fold_learning_and_qc_artifacts(
    tmp_path: Path,
) -> None:
    v2, v5 = tmp_path / "v2", tmp_path / "v5"
    _write_fold(v2, 0.4)
    _write_fold(v5, 0.4)
    for root, loss in ((v2, 0.5), (v5, 0.50001)):
        fold = root / "repeat_00_fold_00"
        (fold / "training_history.json").write_text(
            '{"schema_version":"history.v1","rows":[{"epoch":1,'
            f'"training_loss":{loss}}}]}}\n',
            encoding="utf-8",
        )
        (fold / "quality_diagnostics.json").write_text(
            '{"schema_version":"quality.v1","rows":[]}\n', encoding="utf-8"
        )
    result = MODULE.compare_outputs(v2, v5, atol=1.0e-6)
    assert result["status"] == "failed"
    assert any(
        row["artifact"] == "training_history.json"
        and row["status"] == "failed"
        for row in result["structured_artifacts"]
    )


def test_metrics_schema_excludes_runtime_and_checkpoint_differences(
    tmp_path: Path,
) -> None:
    v2_path, v5_path, _, _ = _write_synthetic_metrics_pair(tmp_path)

    result = MODULE._compare_structured(v2_path, v5_path, atol=1.0e-6)

    assert result["status"] == "passed"
    evidence = result["excluded_non_scientific_runtime"]
    assert "preprocessing_cache_summary" in evidence["v2"][0]["excluded_fields"]
    assert "learned_model_checkpoint" in evidence["v5"][0]["excluded_fields"]
    assert "code_commit" in evidence["v2"][0]["excluded_fields"]
    assert "elapsed_seconds" in evidence["v5"][0]["excluded_fields"]


@pytest.mark.parametrize(
    "mutation",
    ("config", "model", "split", "seed", "metrics"),
)
def test_metrics_schema_rejects_every_scientific_identity_or_value_change(
    tmp_path: Path,
    mutation: str,
) -> None:
    v2_path, v5_path, _, v5_payload = _write_synthetic_metrics_pair(tmp_path)
    cell = v5_payload["cells"][0]
    if mutation == "config":
        cell["config_hash"] = _digest("changed-config")
    elif mutation == "model":
        cell["model_hash"] = _digest("changed-model")
    elif mutation == "split":
        cell["split_seed"] = 43
    elif mutation == "seed":
        cell["training_seed"] = 43
    else:
        cell["metrics"]["balanced_accuracy"] = 0.6
    v5_path.write_text(json.dumps(v5_payload), encoding="utf-8")

    result = MODULE._compare_structured(v2_path, v5_path, atol=1.0e-6)

    assert result["status"] == "failed"


def test_metrics_comparison_ignores_checkpoint_payload_bytes(tmp_path: Path) -> None:
    v2_path, v5_path, _, _ = _write_synthetic_metrics_pair(tmp_path)
    checkpoint = v5_path.parent / "model_checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.joblib").write_bytes(b"tampered")

    result = MODULE._compare_structured(v2_path, v5_path, atol=1.0e-6)

    assert result["status"] == "passed"


def test_metrics_comparison_does_not_follow_checkpoint_symlinks(tmp_path: Path) -> None:
    v2_path, v5_path, _, _ = _write_synthetic_metrics_pair(tmp_path)
    checkpoint = v5_path.parent / "model_checkpoint"
    checkpoint.mkdir()
    outside = tmp_path / "outside.joblib"
    outside.write_bytes(b"outside")
    (checkpoint / "model.joblib").symlink_to(outside)

    result = MODULE._compare_structured(v2_path, v5_path, atol=1.0e-6)

    assert result["status"] == "passed"


def test_metrics_comparison_excludes_cache_metadata(
    tmp_path: Path,
) -> None:
    v2_path, v5_path, _, v5_payload = _write_synthetic_metrics_pair(tmp_path)
    v5_payload["cells"][0]["preprocessing_cache_summary"][
        "affects_predictions"
    ] = True
    v5_path.write_text(json.dumps(v5_payload), encoding="utf-8")

    result = MODULE._compare_structured(v2_path, v5_path, atol=1.0e-6)

    assert result["status"] == "passed"


def test_metrics_comparison_excludes_source_version_format(tmp_path: Path) -> None:
    v2_path, v5_path, v2_payload, _ = _write_synthetic_metrics_pair(tmp_path)
    v2_payload["cells"][0]["source_version"] = "descriptive-version"
    v2_path.write_text(json.dumps(v2_payload), encoding="utf-8")

    result = MODULE._compare_structured(v2_path, v5_path, atol=1.0e-6)

    assert result["status"] == "passed"


def test_non_metrics_qc_artifact_remains_strict(tmp_path: Path) -> None:
    v2 = tmp_path / "v2" / "repeat_00_fold_00" / "quality_diagnostics.json"
    v5 = tmp_path / "v5" / "repeat_00" / "fold_00" / "quality_diagnostics.json"
    v2.parent.mkdir(parents=True)
    v5.parent.mkdir(parents=True)
    payload = {
        "schema_version": "ppg_frailty.quality_diagnostics.v2",
        "quality_mode": "off",
        "classification_effect": "none",
        "rows": [{"file_id": "P01_B", "quality_score": 0.9}],
    }
    v2.write_text(json.dumps(payload), encoding="utf-8")
    payload["rows"][0]["quality_score"] = 0.8
    v5.write_text(json.dumps(payload), encoding="utf-8")

    result = MODULE._compare_structured(v2, v5, atol=1.0e-6)

    assert result["status"] == "failed"
    assert result["failures"] == ["structured_content_differs"]


@pytest.mark.parametrize("atol", (-1.0, 1.000001e-6, float("inf"), float("nan")))
def test_public_api_rejects_tolerance_outside_frozen_contract(
    tmp_path: Path,
    atol: float,
) -> None:
    with pytest.raises(ValueError, match="frozen range"):
        MODULE.compare_outputs(tmp_path / "v2", tmp_path / "v5", atol=atol)


@pytest.mark.parametrize("expected_folds", (0, -1))
def test_public_api_rejects_nonpositive_expected_fold_count(
    tmp_path: Path,
    expected_folds: int,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        MODULE.compare_outputs(
            tmp_path / "v2",
            tmp_path / "v5",
            expected_folds=expected_folds,
        )


def test_cli_rejects_wider_tolerance(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        MODULE.main(
            [
                "--v2-output",
                str(tmp_path / "v2"),
                "--v5-output",
                str(tmp_path / "v5"),
                "--atol",
                "1.1e-6",
            ]
        )
    assert raised.value.code == 2
