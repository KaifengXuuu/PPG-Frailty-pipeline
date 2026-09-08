from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from ppg_frailty.training import OofPredictionRow, aggregate_hierarchy
from ppg_frailty.v5.inference_service import (
    _assert_source_contract,
    _assert_supported_raw_contract,
    _manifest_rows,
    _preprocess_live_scope,
    _validated_role_scope,
)


ROOT = Path(__file__).resolve().parents[1]


def _csv(path: Path) -> None:
    path.write_text(
        "RED,IR,AX,AY,AZ,GX,GY,GZ\n1,2,3,4,5,6,7,8\n",
        encoding="utf-8",
    )


def _source_contract() -> dict[str, object]:
    return {
        "provenance": "user_declared",
        "sampling_rate_hz": 400.0,
        "channel_order": ["RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ"],
        "accelerometer_unit": "g",
        "gyroscope_unit": "deg/s",
        "synchrony": "row_aligned_eight_channel_fixed_grid_no_timestamp",
    }


def test_finalcase_live_inference_contract_is_supported() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/presets/finalcase.yaml").read_text(encoding="utf-8")
    )
    _assert_supported_raw_contract(config)


def test_fold_fitted_raw_transform_fails_closed() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/presets/finalcase.yaml").read_text(encoding="utf-8")
    )
    changed = copy.deepcopy(config)
    changed["signal"]["normalization"]["raw_imu"] = "outer_train_robust"
    with pytest.raises(RuntimeError, match="fold-fitted raw IMU transform"):
        _assert_supported_raw_contract(changed)


def test_dynamic_live_input_requires_static_b(tmp_path: Path) -> None:
    _csv(tmp_path / "r.csv")
    payload = {
        "participant_id": "live-01",
        "source_contract": _source_contract(),
        "files": [{"file_id": "r1", "role": "R1", "path": "r.csv"}],
    }
    with pytest.raises(ValueError, match="missing_static_b_calibration"):
        _manifest_rows(
            payload,
            repository_root=tmp_path,
            class_names=("Pre-Frail", "Robust/Non-Frail", "Young"),
            configured_roles=("B", "R1"),
            classifier_role_families=("R",),
        )


def test_live_rows_share_participant_label_and_source_identity(tmp_path: Path) -> None:
    _csv(tmp_path / "b.csv")
    _csv(tmp_path / "r.csv")
    payload = {
        "participant_id": "live-01",
        "source_contract": _source_contract(),
        "files": [
            {"file_id": "b", "role": "B", "path": "b.csv", "label": "Young"},
            {"file_id": "r1", "role": "R1", "path": "r.csv"},
        ],
    }
    participant, calibration_rows, classifier_rows, label = _manifest_rows(
        payload,
        repository_root=tmp_path,
        class_names=("Pre-Frail", "Robust/Non-Frail", "Young"),
        configured_roles=("B", "R1"),
        classifier_role_families=("R",),
    )
    assert participant == "live-01"
    assert label == 2
    assert [row.class_id for row in calibration_rows] == [2, 2]
    assert [row.role for row in calibration_rows] == ["B", "R1"]
    assert [row.role for row in classifier_rows] == ["R1"]
    assert all(len(row.source_hash) == 64 for row in calibration_rows)
    assert all("user_declared" in row.synchrony_status for row in calibration_rows)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("provenance", "inferred", "provenance"),
        ("sampling_rate_hz", 64.0, "sampling_rate_hz"),
        (
            "channel_order",
            ["IR", "RED", "AX", "AY", "AZ", "GX", "GY", "GZ"],
            "channel_order",
        ),
        ("accelerometer_unit", "m/s^2", "accelerometer_unit"),
        ("gyroscope_unit", "rad/s", "gyroscope_unit"),
        ("synchrony", "timestamp_interpolated", "synchrony"),
    ],
)
def test_source_contract_values_must_match_exactly(
    field: str, value: object, message: str
) -> None:
    payload = {"source_contract": _source_contract()}
    payload["source_contract"][field] = value  # type: ignore[index]
    with pytest.raises(ValueError, match=message):
        _assert_source_contract(payload)


def test_source_contract_cannot_be_omitted_or_partially_declared() -> None:
    with pytest.raises(TypeError, match="source_contract"):
        _assert_source_contract({})
    contract = _source_contract()
    contract.pop("gyroscope_unit")
    with pytest.raises(ValueError, match="missing required fields"):
        _assert_source_contract({"source_contract": contract})

    contract = _source_contract()
    contract["timestamp_column"] = "time"
    with pytest.raises(ValueError, match="unsupported fields"):
        _assert_source_contract({"source_contract": contract})


def test_role_scope_preserves_concrete_roles_and_rejects_adapter_drift() -> None:
    config = {
        "roles": ["B", "R1", "R2", "S1"],
        "training": {"classifier_role_families": ["R", "S"]},
    }
    roles, families = _validated_role_scope(config, None)
    assert roles == ("B", "R1", "R2", "S1")
    assert families == ("R", "S")

    class Adapter:
        allowed_role_families = ("B", "R")

    with pytest.raises(RuntimeError, match="adapter allowed_role_families"):
        _validated_role_scope(config, Adapter())


def test_input_role_must_be_concrete_config_role(tmp_path: Path) -> None:
    _csv(tmp_path / "b.csv")
    _csv(tmp_path / "r.csv")
    payload = {
        "participant_id": "live-01",
        "source_contract": _source_contract(),
        "files": [
            {"file_id": "b", "role": "B", "path": "b.csv"},
            {"file_id": "r", "role": "R", "path": "r.csv"},
        ],
    }
    with pytest.raises(ValueError, match="exact member of config.roles"):
        _manifest_rows(
            payload,
            repository_root=tmp_path,
            class_names=("Pre-Frail", "Robust/Non-Frail", "Young"),
            configured_roles=("B", "R1"),
            classifier_role_families=("R",),
        )


def test_non_b_role_outside_classifier_scope_is_rejected(tmp_path: Path) -> None:
    _csv(tmp_path / "s.csv")
    payload = {
        "participant_id": "live-01",
        "source_contract": _source_contract(),
        "files": [{"file_id": "s", "role": "S1", "path": "s.csv"}],
    }
    with pytest.raises(ValueError, match="only B may be calibration-only"):
        _manifest_rows(
            payload,
            repository_root=tmp_path,
            class_names=("Pre-Frail", "Robust/Non-Frail", "Young"),
            configured_roles=("B", "R1", "S1"),
            classifier_role_families=("R",),
        )


def test_unlabelled_calibration_b_is_not_a_classifier_row(tmp_path: Path) -> None:
    _csv(tmp_path / "b.csv")
    _csv(tmp_path / "r.csv")
    payload = {
        "participant_id": "live-01",
        "source_contract": _source_contract(),
        "files": [
            {"file_id": "b", "role": "B", "path": "b.csv"},
            {"file_id": "r1", "role": "R1", "path": "r.csv"},
        ],
    }
    _, calibration_rows, classifier_rows, label = _manifest_rows(
        payload,
        repository_root=tmp_path,
        class_names=("Pre-Frail", "Robust/Non-Frail", "Young"),
        configured_roles=("B", "R1"),
        classifier_role_families=("R",),
    )
    assert label is None
    assert [(row.role, row.class_id) for row in calibration_rows] == [
        ("B", -1),
        ("R1", -1),
    ]
    assert [(row.role, row.class_id) for row in classifier_rows] == [("R1", -1)]

    observed: dict[str, object] = {}

    class Core:
        @staticmethod
        def _RuntimeRecord(*, row: object) -> object:
            return row

        @staticmethod
        def _preprocess_records(
            states: list[object],
            config: object,
            split: object,
            loader: object,
            *,
            calibration_rows: tuple[object, ...],
            cache_session: object,
        ) -> None:
            observed.update(
                states=states,
                config=config,
                split=split,
                loader=loader,
                calibration_rows=calibration_rows,
                cache_session=cache_session,
            )

    loader = object()
    states = _preprocess_live_scope(
        Core,
        classifier_rows=classifier_rows,
        calibration_rows=calibration_rows,
        config="resolved-config",
        loader=loader,
    )
    assert [row.role for row in states] == ["R1"]
    supplied_calibration = observed["calibration_rows"]
    assert isinstance(supplied_calibration, tuple)
    assert [row.role for row in supplied_calibration] == ["B", "R1"]
    assert observed["config"] == "resolved-config"
    assert observed["loader"] is loader


def test_inference_module_has_no_exported_environment_revalidation_chain() -> None:
    """Environment checking belongs to the inference entry point, not artifacts."""

    import ppg_frailty.v5.inference_service as service

    assert not hasattr(service, "_assert_exact_environment_evidence")
    assert not hasattr(service, "_assert_exported_training_environment")


def test_live_contract_is_independent_of_training_request_metadata() -> None:
    payload = {"source_contract": _source_contract(), "training_request": {"ignored": True}}
    assert _assert_source_contract(payload) == _source_contract()


def test_unlabelled_minus_one_is_metadata_only_during_probability_aggregation() -> None:
    common = {
        "participant_id": "live-01",
        "role": "R",
        "label": -1,
        "repeat": 0,
        "fold": 0,
        "split_seed": 0,
        "training_seed": 42,
        "config_hash": "config",
        "manifest_hash": "manifest",
        "fold_hash": "fold",
        "preprocessing_hash": "preprocessing",
        "feature_hash": "feature",
        "model_hash": "model",
        "representation_mode": "raw",
        "signal_route": "direct",
        "quality_score": 1.0,
        "retained": True,
        "level": "file",
        "class_order": (),
        "aggregation_rule": "line_b_equal_role_families",
    }
    rows = (
        OofPredictionRow(
            file_id="r1",
            probabilities=(0.8, 0.1, 0.1),
            **common,
        ),
        OofPredictionRow(
            file_id="r2",
            probabilities=(0.2, 0.3, 0.5),
            **common,
        ),
    )
    aggregated = aggregate_hierarchy(
        rows,
        balance_line="line_b_equal_role_families",
        quality_weighted=False,
        quality_weight_source="none",
    )
    participant = aggregated.participant_rows[0]
    assert participant.label == -1
    assert participant.class_order == ()
    assert participant.probabilities == pytest.approx((0.5, 0.2, 0.3))
