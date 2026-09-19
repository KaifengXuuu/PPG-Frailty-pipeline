"""Stage chaining, control invalidation, and unchanged numerical kernels."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ppg_frailty.config import load_config, validate_config_payload
from ppg_frailty.dashboard.workflow_service import STAGES, WorkflowService


ROOT = Path(__file__).resolve().parents[1]


def _numerical_cells(timeline):
    """Compare every route/score/boundary, excluding UI-vs-run provenance hashes."""
    return [{key: value for key, value in asdict(cell).items() if not key.endswith("sha256")}
            for cell in timeline.cells]


@pytest.fixture
def config():
    payload = load_config(ROOT / "configs/presets/finalcase.yaml").to_dict()
    payload["signal"]["imu"] = {"gravity_method": "low_pass_0p3hz"}
    return validate_config_payload(payload)


@pytest.fixture
def source():
    time = np.arange(8000) / 400.0
    ppg = np.column_stack((100 + np.sin(2 * np.pi * 1.2 * time), 120 + 2 * np.sin(2 * np.pi * 1.2 * time + 0.1)))
    acc = np.column_stack((0.01 * np.sin(time), 0.015 * np.sin(0.3 * time), 1 + 0.01 * np.sin(0.2 * time)))
    gyro = np.column_stack((np.sin(time), np.cos(time), np.sin(time / 2)))
    row = SimpleNamespace(record_id="fixture", participant_id="person", role="B", class_id=1,
                          class_name="Robust", fs=400.0, n_samples=len(time), duration_s=20.0,
                          source_path="fixture.csv", qc_status="pass")
    loaded = dict(record_id="fixture", participant_id="person", fs_hz=400.0, ppg=ppg, acc=acc,
                  gyro=gyro, acc_unit="g", gyro_unit="deg/s")
    return {"row": row, "rows": [row], "loaded": loaded,
            "signals": {"RED": ppg[:, 0], "IR": ppg[:, 1]}, "metadata": {"sample_count": len(time)}}


@pytest.fixture
def service(monkeypatch, source):
    service = WorkflowService(ROOT)
    monkeypatch.setattr(service, "_input", lambda *args: source)
    return service


def test_upstream_is_automatic_and_reused_for_display_only(service, config):
    first = service.analyse("imu", config, "fixture", session_id="A")
    assert first["computed_stages"] == ["input", "ppg", "imu"]
    second = service.analyse("imu", config, "fixture", session_id="A", start_s=2, duration_s=3)
    assert second["computed_stages"] == []
    assert second["reused_stages"] == ["input", "ppg", "imu"]
    assert second["previews"]["ppg"]["traces"]["filtered_RED"]["x"][0] == 2.0
    other_session = service.analyse("imu", config, "fixture", session_id="B")
    assert other_session["computed_stages"] == ["input", "ppg", "imu"]


def test_parameter_edit_invalidates_only_affected_descendants(service, config):
    service.analyse("quality", config, "fixture")
    updated = deepcopy(config)
    updated["signal"]["imu"]["sensor_lowpass_acc_hz"] = 12.0
    changed = service.analyse("quality", updated, "fixture")
    assert changed["reused_stages"] == ["input", "ppg"]
    assert changed["computed_stages"] == ["imu", "motion", "quality"]
    updated["signal"]["ppg_filter"]["high_hz"] = 4.0
    changed = service.analyse("ppg", updated, "fixture")
    assert changed["reused_stages"] == ["input"]
    assert changed["computed_stages"] == ["ppg"]
    assert set(service._sessions["default"]["fixture"]) == {"input", "ppg"}


def test_reusing_ppg_is_exactly_equal_to_original_combined_call(config, source):
    from ppg_frailty.signal.preprocess import build_signal_views, preprocess_ppg_pair
    original = build_signal_views(source["loaded"], config)
    signal = config["signal"]
    pair = preprocess_ppg_pair(source["loaded"]["ppg"],
        max_gap_samples=signal["gap_repair"]["max_gap_samples"],
        flatline_sec=config["quality"]["flatline_duration_s"],
        filter_low_hz=signal["ppg_filter"]["low_hz"], filter_high_hz=signal["ppg_filter"]["high_hz"],
        filter_order=signal["ppg_filter"]["order"])
    reused = build_signal_views(source["loaded"], config, ppg_result=pair)
    for field in ("x_native", "x_filter", "x_analysis_rate", "source_valid_mask", "repair_mask"):
        np.testing.assert_array_equal(getattr(reused, field), getattr(original, field))
    for key in original.imu_processed:
        np.testing.assert_array_equal(reused.imu_processed[key], original.imu_processed[key])


def test_ppg_control_changes_real_full_record_filter_output(service, config):
    first = service.analyse("ppg", config, "fixture")
    edited = deepcopy(config)
    edited["signal"]["ppg_filter"]["high_hz"] = 1.0
    second = service.analyse("ppg", edited, "fixture")
    a = first["previews"]["ppg"]["traces"]["filtered_RED"]["y"]
    b = second["previews"]["ppg"]["traces"]["filtered_RED"]["y"]
    assert not np.allclose(a, b)


def test_end_to_end_representation_uses_existing_algorithms_without_fitting(service, config, monkeypatch):
    from ppg_frailty.signal import sqi
    from ppg_frailty.training.trainer import UnifiedTrainer
    def forbidden(*args, **kwargs):
        raise AssertionError("Analyse must never fit")
    monkeypatch.setattr(sqi, "fit_sqi_calibrator", forbidden)
    monkeypatch.setattr(UnifiedTrainer, "fit", forbidden, raising=False)
    result = service.analyse("representation", config, "fixture")
    assert result["computed_stages"] == list(STAGES[:8])
    assert result["previews"]["features"]["metadata"]["feature_count"] > 100
    assert result["previews"]["representation"]["metadata"]["shape"][1:] == [8, 320]
    assert result["previews"]["representation"]["metadata"]["retained"]
    assert all(not preview["metadata"]["model_training_executed"] for preview in result["previews"].values())


def test_sqi_empirical_requires_existing_artifact_without_fitting(service, config):
    from ppg_frailty.signal.sqi import SqiConfig
    edited = deepcopy(config)
    edited["quality"].update(SqiConfig(calibrator="outer_train_empirical_quantiles_v1").to_dict(), mode="route")
    with pytest.raises(ValueError, match="existing fitted SQI"):
        service.analyse("quality", edited, "fixture")


def test_motion_requires_existing_weights_and_never_trains(service, config):
    edited = deepcopy(config)
    edited["artifact"]["motion_detector_enabled"] = True
    edited["artifact"]["motion_detector"]["evidence_path"] = None
    with pytest.raises(ValueError, match="existing motion"):
        service.analyse("motion", edited, "fixture")


def test_failed_stage_keeps_valid_upstream_previews_and_removes_old_descendants(service, config):
    service.analyse("representation", config, "fixture", session_id="browser")
    edited = deepcopy(config)
    edited["artifact"]["motion_detector_enabled"] = True
    edited["artifact"]["motion_detector"]["evidence_path"] = None
    with pytest.raises(ValueError, match="existing motion"):
        service.analyse("motion", edited, "fixture", session_id="browser")
    previews = service.cached_previews("browser", "fixture", start_s=2.0, duration_s=3.0)
    assert list(previews) == ["input", "ppg", "imu"]
    assert previews["ppg"]["traces"]["filtered_RED"]["x"][0] == 2.0
    assert service.cached_previews("another-browser", "fixture") == {}
    assert service.cached_previews("browser", "unknown-record") == {}


def test_model_analyse_requires_weights_not_training(service, config):
    with pytest.raises(ValueError, match="pretrained model weights"):
        service.analyse("model", config, "fixture")


def test_clear_releases_only_requested_session(service, config):
    service.analyse("ppg", config, "fixture", session_id="one")
    service.analyse("ppg", config, "fixture", session_id="two")
    service.clear("one")
    assert list(service._sessions) == ["two"]


@pytest.mark.parametrize("mode", ["diagnostics_only", "route"])
def test_stage_routing_matches_production_window_route(service, config, mode):
    from ppg_frailty.config import PipelineConfig
    from ppg_frailty.experiment import _RuntimeRecord, _route_records_window_level
    from ppg_frailty.signal.sqi import SqiConfig
    edited = deepcopy(config)
    edited["quality"].update(SqiConfig().to_dict(), mode=mode)
    edited = validate_config_payload(edited)
    service.analyse("denoiser", edited, "fixture")
    context = service._sessions["default"]["fixture"]
    staged = context["denoiser"]["value"]["state"]
    reference = _RuntimeRecord(staged.row, views=staged.views)
    proxy = PipelineConfig(edited, "fixture", "same-source")
    report = SimpleNamespace(artifact={"runtime_reducer": "identity", "parameters": {}})
    sqi = context["quality"]["value"]["sqi"]
    _route_records_window_level([reference], proxy, report, sqi, None)
    assert _numerical_cells(staged.routing_timeline) == _numerical_cells(reference.routing_timeline)
    assert staged.retained == reference.retained
    assert staged.route == reference.route


def test_all_optional_modules_off_preserves_production_direct_route(service, config):
    from ppg_frailty.config import PipelineConfig
    from ppg_frailty.experiment import _RuntimeRecord, _retain_without_quality_routing, _extract_raw
    service.analyse("representation", config, "fixture")
    context = service._sessions["default"]["fixture"]
    staged = context["representation"]["value"]["state"]
    reference = _RuntimeRecord(staged.row, views=staged.views)
    _retain_without_quality_routing([reference], PipelineConfig(config, "fixture", "source"), diagnostics_only=False)
    _extract_raw(reference, service._report(config), config["signal"])
    assert staged.routing_timeline is None
    assert staged.route_status == reference.route_status
    np.testing.assert_array_equal(staged.raw_windows.values, reference.raw_windows.values)
    np.testing.assert_array_equal(staged.raw_windows.start_samples, reference.raw_windows.start_samples)


def test_no_extra_eight_second_gate_on_direct_raw_analysis(service, config, source):
    source["row"].n_samples, source["row"].duration_s = 2800, 7.0
    for key in ("ppg", "acc", "gyro"):
        source["loaded"][key] = source["loaded"][key][:2800]
    source["signals"] = {key: value[:2800] for key, value in source["signals"].items()}
    result = service.analyse("representation", config, "fixture")
    assert result["previews"]["representation"]["metadata"]["retained"]
    assert result["previews"]["representation"]["metadata"]["shape"] == [1, 8, 320]


def test_denoiser_matches_production_and_preserves_native_views(service, config):
    from ppg_frailty.config import PipelineConfig
    from ppg_frailty.experiment import _RuntimeRecord, _route_records_window_level
    from ppg_frailty.signal.sqi import SqiConfig
    edited = deepcopy(config)
    edited["quality"].update(SqiConfig(q_rate_threshold=1.0, q_morph_threshold=1.0).to_dict(), mode="route")
    edited["artifact"].update(denoiser_enabled=True, reducer="dwt_a2_legacy", reducer_version="dwt_a2_legacy_v1",
                                degraded_policy="denoise_then_extract_rate_features", parameters={})
    # The stage takes current controls; formal config materialization owns reducer metadata.
    service.analyse("denoiser", edited, "fixture")
    context = service._sessions["default"]["fixture"]
    staged = context["denoiser"]["value"]["state"]
    assert context["denoiser"]["value"]["metadata"]["invoked"]
    reference = _RuntimeRecord(staged.row, views=staged.views)
    proxy = PipelineConfig(edited, "fixture", "same-source")
    report = SimpleNamespace(artifact={"runtime_reducer": "dwt_a2_legacy", "parameters": {}})
    _route_records_window_level([reference], proxy, report, context["quality"]["value"]["sqi"], None)
    assert _numerical_cells(staged.routing_timeline) == _numerical_cells(reference.routing_timeline)
    np.testing.assert_array_equal(staged.processed_views.x_ar, reference.processed_views.x_ar)
    assert staged.processed_views.x_filter is staged.views.x_filter
    assert staged.processed_views.x_native is staged.views.x_native


@pytest.mark.parametrize("mode", ["feature_vector", "feature_matrix", "fusion"])
def test_all_representation_branches_are_materialized_without_fitting(service, config, mode):
    from ppg_frailty.signal.sqi import SqiConfig
    edited = deepcopy(config)
    edited["representation_mode"] = mode
    edited["quality"].update(SqiConfig().to_dict(), mode="diagnostics_only")
    result = service.analyse("representation", edited, "fixture")
    preview = result["previews"]["representation"]
    assert preview["metadata"]["retained"]
    if mode == "feature_matrix":
        assert preview["metadata"]["shape"][0] == 146
        assert preview["tables"]["matrix_windows"]
    else:
        assert preview["metadata"]["feature_vector_shape"] == [282]
        assert len(preview["tables"]["feature_vector"]) == 282
    if mode == "fusion":
        assert preview["metadata"]["shape"][1:] == [8, 320]


def test_matrix_does_not_inherit_failure_of_nonessential_file_vector_preview(service, config, monkeypatch):
    from ppg_frailty import experiment
    from ppg_frailty.signal.sqi import SqiConfig
    edited = deepcopy(config)
    edited["representation_mode"] = "feature_matrix"
    edited["quality"].update(SqiConfig().to_dict(), mode="diagnostics_only")
    def fail_vector_preview(state, *args):
        state.retained, state.reason = False, "optional_vector_preview_failed"
    monkeypatch.setattr(experiment, "_extract_vector", fail_vector_preview)
    result = service.analyse("representation", edited, "fixture")
    assert result["previews"]["features"]["metadata"]["feature_vector_preview_reason"] == "optional_vector_preview_failed"
    assert result["previews"]["representation"]["metadata"]["retained"]
    assert result["previews"]["representation"]["metadata"]["shape"][0] == 146


def test_feature_group_edit_reuses_all_signal_and_quality_stages(service, config):
    service.analyse("representation", config, "fixture")
    edited = deepcopy(config)
    edited["features"]["enabled_groups"] = ["ppi_basic_rate"]
    result = service.analyse("representation", edited, "fixture")
    assert result["reused_stages"] == list(STAGES[:6])
    assert result["computed_stages"] == ["features", "representation"]
    assert 0 < result["previews"]["features"]["metadata"]["feature_count"] < 282


def test_existing_sqi_calibrator_loaded_without_fit(service, config, tmp_path):
    from ppg_frailty.signal.sqi import SqiConfig
    import json
    artifact = tmp_path / "calibrator.json"
    artifact.write_text(json.dumps({"bounds": {"cardiac_concentration": [0.0, 1.0]},
        "fitted_on_participant_ids": ["training-person"], "method": "outer_train_empirical_quantiles_v1"}))
    edited = deepcopy(config)
    edited["quality"].update(SqiConfig(calibrator="outer_train_empirical_quantiles_v1").to_dict(), mode="route")
    result = service.analyse("quality", edited, "fixture", selections={"sqi_artifact": str(artifact)})
    assert result["previews"]["quality"]["tables"]["windows"]
    calibrator = service._sessions["default"]["fixture"]["quality"]["value"]["calibrator"]
    assert calibrator.fitted_on_participant_ids == ("training-person",)


def _mock_model(monkeypatch, tmp_path):
    from ppg_frailty.training.datasets import SampleIdentity
    from ppg_frailty.training import bundle
    from ppg_frailty.v5 import inference_service
    monkeypatch.setattr(bundle, "load_bundle", lambda path: object())
    monkeypatch.setattr(inference_service, "resolve_workflow_bundle", lambda *args, **kwargs: tmp_path)
    calls = []
    def predict(weights, **kwargs):
        calls.append(kwargs["record_id"])
        return {"probabilities": np.array([[0.1, 0.6, 0.3]]), "identities": (SampleIdentity(
            participant_id=kwargs["participant_id"], file_id=kwargs["record_id"], role=kwargs["role"],
            label=-1 if kwargs["label"] is None else kwargs["label"], signal_route=kwargs["route"], window_id="window-0"),)}
    monkeypatch.setattr(inference_service, "predict_workflow_record", predict)
    (tmp_path / "manifest.json").write_text('{}')
    return calls


@pytest.mark.parametrize("record_ids", [["one", "two"], [f"file-{i}" for i in range(9)]])
def test_multiple_record_model_analysis_reuses_each_representation_and_aggregates(service, config, source, monkeypatch, tmp_path, record_ids):
    from copy import copy
    def input_for_record(config, selected, context, record_id, session_id):
        result = dict(source)
        result["row"] = copy(source["row"])
        result["row"].record_id = record_id
        result["loaded"] = {**source["loaded"], "record_id": record_id}
        return result
    monkeypatch.setattr(service, "_input", input_for_record)
    calls = _mock_model(monkeypatch, tmp_path)
    selected = {"record_ids": record_ids, "model_export": str(tmp_path)}
    result = service.analyse("aggregation", config, record_ids[0], selections=selected)
    assert calls == record_ids
    assert len(result["previews"]["aggregation"]["tables"]["recordings"]) == len(record_ids)
    assert len(result["previews"]["aggregation"]["tables"]["participants"]) == 1
    second = service.analyse("aggregation", config, record_ids[0], selections=selected)
    assert second["computed_stages"] == []
    assert calls == record_ids


def test_balance_line_edit_reuses_predictions_while_role_scope_edit_repredicts(service, config, monkeypatch, tmp_path):
    calls = _mock_model(monkeypatch, tmp_path)
    selected = {"model_export": str(tmp_path)}
    service.analyse("aggregation", config, "fixture", selections=selected)
    edited = deepcopy(config)
    edited["aggregation"]["balance_line"] = "line_a_equal_files"
    result = service.analyse("aggregation", edited, "fixture", selections=selected)
    assert result["computed_stages"] == ["aggregation"]
    assert calls == ["fixture"]
    edited["roles"] = ["B"]
    result = service.analyse("aggregation", edited, "fixture", selections=selected)
    assert result["computed_stages"] == ["model", "aggregation"]
    assert calls == ["fixture", "fixture"]


def test_prediction_previews_name_the_highest_probability_without_changing_values(service, config, monkeypatch, tmp_path):
    from ppg_frailty.dashboard.workflow_service import _prediction_fields
    _mock_model(monkeypatch, tmp_path)
    result = service.analyse("aggregation", config, "fixture", selections={"model_export": str(tmp_path)})
    expected_name = config["manifest"]["class_name_order"][1]
    expected_id = config["manifest"]["class_id_order"][1]
    for stage in ("model", "aggregation"):
        assert result["previews"][stage]["metadata"]["class_names"] == config["manifest"]["class_name_order"]
    window = result["previews"]["model"]["tables"]["predictions"][0]
    assert window["predicted_class_name"] == expected_name
    assert window["predicted_class_id"] == expected_id
    assert [window[f"p_{index}"] for index in range(3)] == [0.1, 0.6, 0.3]
    for rows in result["previews"]["aggregation"]["tables"].values():
        for row in rows:
            assert row["predicted_class_name"] == expected_name
            assert row["predicted_class_id"] == expected_id
            np.testing.assert_allclose(row["probabilities"], [0.1, 0.6, 0.3], atol=0, rtol=0)
    assert _prediction_fields(config, []) == {}


@pytest.mark.parametrize("labels", [(2, None), (None, 2), (None, None), (0, 2)])
def test_custom_files_share_participant_labels_before_aggregation(service, config, source, monkeypatch, tmp_path, labels):
    from ppg_frailty.data.manifest import load_internal_manifest
    from ppg_frailty import pipeline
    template = load_internal_manifest(ROOT / "manifests/internal_records_v2.csv")[0]
    def csv_row(item, participant_id, *, class_names):
        label = -1 if item["label"] is None else item["label"]
        return replace(template, record_id=item["file_id"], participant_id=participant_id, role=item["role"],
                       class_id=label, class_name="unlabelled" if label < 0 else class_names[label],
                       n_samples=8000, duration_s=20.0)
    monkeypatch.setattr(service, "_csv_row", csv_row)
    monkeypatch.setattr(service, "_input", WorkflowService._input.__get__(service))
    monkeypatch.setattr(pipeline, "_load_record", lambda row, *args, **kwargs: {**source["loaded"], "record_id": row.record_id})
    _mock_model(monkeypatch, tmp_path)
    files = [{"file_id": "b", "role": "B", "path": "b.csv", "label": labels[0]},
             {"file_id": "r", "role": "R1", "path": "r.csv", "label": labels[1]}]
    selected = {"files": files, "participant_id": "one-person", "model_export": str(tmp_path)}
    if len({label for label in labels if label is not None}) > 1:
        with pytest.raises(ValueError, match="share one label"):
            service.analyse("aggregation", config, "b", selections=selected)
        return
    result = service.analyse("aggregation", config, "b", selections=selected)
    expected = next((label for label in labels if label is not None), -1)
    rows = result["previews"]["aggregation"]["tables"]["recordings"]
    assert len(rows) == 2
    assert {row["label"] for row in rows} == {expected}
    assert result["previews"]["aggregation"]["tables"]["participants"][0]["label"] == expected
