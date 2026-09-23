from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from ppg_frailty.v5.results import build_study_data_index


PREDICTION_FILES = (
    "oof_window_predictions.parquet",
    "oof_file_predictions.parquet",
    "oof_role_predictions.parquet",
    "oof_subject_predictions.parquet",
    "oof_member_predictions.parquet",
)


def _write_prediction(path: Path, rows: int, *, empty_reason: str = "") -> None:
    schema = pa.schema(
        [pa.field("value", pa.int64())],
        metadata={
            b"schema_version": b"ppg_frailty_oof_v2",
            b"artifact_state": b"empty" if rows == 0 else b"populated",
            b"empty_reason": empty_reason.encode("utf-8"),
        },
    )
    table = pa.Table.from_arrays([pa.array(range(rows), type=pa.int64())], schema=schema)
    pq.write_table(table, path)


def test_index_keeps_per_fold_files_without_copying_rows(tmp_path: Path) -> None:
    study = tmp_path / "study"
    case = study / "cases/reference"
    cell = case / "attempts/attempt_001/experiment/repeat_00_fold_00"
    cell.mkdir(parents=True)
    config = {
        "schema_version": "example",
        "config_id": "example_config",
        "model": {"model_id": "Example"},
    }
    (case / "resolved_config.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    for index, filename in enumerate(PREDICTION_FILES):
        _write_prediction(
            cell / filename,
            0 if filename == "oof_member_predictions.parquet" else index + 1,
            empty_reason=(
                "single_model_has_no_member_rows"
                if filename == "oof_member_predictions.parquet"
                else ""
            ),
        )
    run_manifest = {
        "status": "passed",
        "cell": {
            "status": "passed",
            "repeat_index": 0,
            "fold_index": 0,
            "config_id": "example_config",
            "config_hash": "c" * 64,
            "split_seed": 42,
            "training_seed": 42,
            "model_id": "Example",
            "model_machine_id": "example",
            "representation_mode": "raw",
            "model_hash": "m" * 64,
            "preprocessing_hash": "p" * 64,
            "feature_hash": "f" * 64,
            "balance_line": "line_b_equal_role_families",
            "metrics": {"balanced_accuracy": 0.5, "macro_f1": 0.4},
            "fitted_provenance": {"state_hash": "s" * 64},
            "model_factory_provenance": {"parameter_count": 123},
            "frozen_model_run_provenance": {"dropout": 0.2},
        },
    }
    (cell / "run_manifest.json").write_text(
        json.dumps(run_manifest), encoding="utf-8"
    )
    (case / "case_result.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "artifact_root": "attempts/attempt_001/experiment",
            }
        ),
        encoding="utf-8",
    )
    (study / "study_manifest.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "cases": [
                    {
                        "case_id": "reference",
                        "case_directory": "cases/reference",
                        "resolved_config_path": "cases/reference/resolved_config.yaml",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = build_study_data_index(study)

    assert result["status"] == "complete"
    assert result["fold_count"] == 1
    assert result["prediction_artifact_count"] == 5
    assert result["prediction_row_count"] == 1 + 2 + 3 + 4
    assert (study / "tables/v5_fold_predictions.csv").is_file()
    assert (study / "tables/v5_fold_models.csv").is_file()
    assert (study / "tables/v5_config_parameters.csv").is_file()
    assert not (study / "predictions_copy").exists()


def test_index_accepts_nested_cells_and_publishes_median_checkpoint(tmp_path: Path) -> None:
    study = tmp_path / "study"
    cell = study / "reference/repeat_00/fold_00"
    checkpoint = cell / "model_checkpoint/manifest.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("{}\n", encoding="utf-8")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config = {"schema_version": "example", "config_id": "nested"}
    (study / "configs").mkdir()
    (study / "configs/reference.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    for filename in PREDICTION_FILES:
        _write_prediction(cell / filename, 1)
    (cell / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "cell": {
                    "status": "passed",
                    "repeat_index": 0,
                    "fold_index": 0,
                    "config_id": "nested",
                    "config_hash": "c" * 64,
                    "model_id": "Example",
                    "metrics": {"balanced_accuracy": 0.6},
                    "learned_model_checkpoint": {
                        "schema_version": "ppg_frailty.v5_fold_checkpoint.v1",
                        "manifest_path": "model_checkpoint/manifest.json",
                        "manifest_sha256": checkpoint_hash,
                        "state_sha256": "s" * 64,
                        "deployment_status": "research_only",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (study / "reference/case_result.json").write_text(
        json.dumps({"status": "passed", "artifact_root": "."}), encoding="utf-8"
    )
    (study / "study_manifest.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "reference",
                        "case_directory": "reference",
                        "resolved_config_path": "configs/reference.yaml",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = build_study_data_index(study)

    assert result["fold_count"] == 1
    assert result["published_models"][0]["repeat"] == 0
    assert result["published_models"][0]["fold"] == 0
    selection = json.loads(
        (study / "models/reference/median_fold/selection.json").read_text(
            encoding="utf-8"
        )
    )
    assert selection["complete_5x5"] is False
    assert selection["checkpoint_manifest"] == (
        "reference/repeat_00/fold_00/model_checkpoint/manifest.json"
    )
