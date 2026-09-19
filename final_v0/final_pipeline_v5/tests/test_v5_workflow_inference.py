"""No-fit prediction from current notebook controls and cached stage outputs."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from ppg_frailty.models import normalize_model_id
from ppg_frailty.representations.raw import RawWindows
from ppg_frailty.training.bundle import LoadedBundle
from ppg_frailty.v5.inference_service import (
    _assert_loaded_bundle_contract, _assert_model_compatible,
    predict_model_ready, predict_workflow_record,
)

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ("RED", "IR", "A_dyn_x", "A_dyn_y", "A_dyn_z", "GX", "GY", "GZ")


class _PredictOnlyEstimator:
    classes_ = np.arange(3)

    def fit(self, *_: object, **__: object) -> None:
        raise AssertionError("Analyse must never fit")

    def predict_proba(self, values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        self.last_values = np.asarray(values).copy()
        self.last_mask = mask
        score = np.tanh(np.nanmean(values.reshape(len(values), -1), axis=1))
        return np.column_stack((0.3 + 0.1 * score, 0.3 - 0.1 * score, np.full(len(values), 0.4)))


def _config(preset: str = "finalcase") -> dict:
    payload = yaml.safe_load((ROOT / f"configs/presets/{preset}.yaml").read_text())
    payload["training"]["device"] = "cpu"
    return payload


def _bundle(config: dict) -> LoadedBundle:
    mode = config["representation_mode"]
    return LoadedBundle(_PredictOnlyEstimator(), None, {
        "kind": "estimator", "machine_model_id": normalize_model_id(config["model"]["model_id"])[1],
        "config_hash": "saved_training_config", "metadata": {"class_order": [0, 1, 2]},
        "input_spec": {"representation_mode": mode, "n_classes": 3,
                       "n_channels": 0 if mode == "feature_vector" else 8,
                       "feature_names": ["a", "b"] if mode == "feature_vector" else [],
                       "channel_schema": [] if mode == "feature_vector" else list(CHANNELS)},
    }, ROOT)


def _windows() -> RawWindows:
    values = np.random.default_rng(7).normal(size=(2, 8, 2000)).astype(np.float32)
    return RawWindows(values, np.ones((2, 2000), dtype=bool), np.array([0, 1000]), 2, 0)


def test_preprocessing_hash_change_does_not_reject_same_architecture() -> None:
    config = _config()
    loaded = _bundle(config)
    config["signal"]["ppg_filter"]["high_hz"] = 7.0
    spec, _, _ = _assert_loaded_bundle_contract(loaded, config, config_hash="current_changed_config")
    assert spec.n_channels == 8


def test_raw_cached_output_and_current_resampling_reach_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    from ppg_frailty import experiment

    def no_fit(*_: object, **__: object) -> None:
        raise AssertionError("cached model Analyse must not fit or preprocess again")

    monkeypatch.setattr(experiment, "_fit_representation_artifacts", no_fit)
    monkeypatch.setattr(experiment, "_preprocess_records", no_fit)
    config = _config()
    loaded, windows = _bundle(config), _windows()
    original = windows.values.copy()
    first = predict_workflow_record(loaded, config_payload=config, participant_id="p", record_id="r", role="B",
                                    raw_windows=windows)
    assert first["input_shape"] == [2, 8, 320]
    config["signal"]["dl_resampling"]["target_fs_hz"] = 128.0
    second = predict_workflow_record(loaded, config_payload=config, participant_id="p", record_id="r", role="B",
                                     raw_windows=windows)
    assert second["input_shape"] == [2, 8, 640]
    assert loaded.model.last_values.shape == (2, 8, 640)
    assert second["identities"][0].participant_id == "p"
    assert second["identities"][0].label == -1
    assert second["window_ids"] == ["r::start_000000000000", "r::start_000000001000"]
    assert second["training_performed"] is False
    np.testing.assert_array_equal(original, windows.values)


def test_feature_vector_uses_selected_estimator_without_refitting() -> None:
    config = _config("feature_vector")
    loaded = _bundle(config)
    vector = SimpleNamespace(values=np.array([1.0, 2.0]), feature_names=("a", "b"))
    result = predict_workflow_record(loaded, config_payload=config, participant_id="p", record_id="r", role="B",
                                     feature_vector=vector, label=2, quality_score=0.6)
    assert result["probabilities"].shape == (1, 3)
    assert result["identities"][0].label == 2
    assert result["identities"][0].quality_score == 0.6
    np.testing.assert_array_equal(loaded.model.last_values, [[1.0, 2.0]])
    vector.feature_names = ("b", "a")
    with pytest.raises(ValueError, match="feature names"):
        predict_workflow_record(loaded, config_payload=config, participant_id="p", record_id="r", role="B",
                                 feature_vector=vector)


def test_missing_fitted_state_is_not_estimated_from_current_participant() -> None:
    config = _config()
    loaded = _bundle(config)
    config["signal"]["normalization"]["raw_imu"] = "outer_train_robust"
    with pytest.raises(ValueError, match="fitted raw_imu"):
        predict_workflow_record(loaded, config_payload=config, participant_id="p", record_id="r", role="B",
                                 raw_windows=_windows())


def test_architecture_change_requires_compatible_weights() -> None:
    from ppg_frailty.module_registry import materialize_model_architecture

    config = _config()
    loaded = _bundle(config)
    loaded.manifest["model_config"] = {"architecture_parameters": materialize_model_architecture(config["model"], "raw")}
    assert _assert_model_compatible(loaded, config).n_channels == 8
    changed = copy.deepcopy(config)
    changed["model"]["depth"] = int(changed["model"].get("depth", 3)) + 1
    changed["model"].pop("architecture_parameters", None)
    with pytest.raises(ValueError, match="architecture"):
        _assert_model_compatible(loaded, changed)


def test_model_ready_bundle_path_and_dimensions() -> None:
    config = _config()
    loaded = _bundle(config)
    inputs = {"x": np.ones((2, 8, 64), dtype=np.float32), "mask": np.ones((2, 64), dtype=bool)}
    probabilities = predict_model_ready(loaded, inputs, config_payload=config)
    assert probabilities.shape == (2, 3)
    with pytest.raises(ValueError, match="channel count"):
        predict_model_ready(loaded, {**inputs, "x": np.ones((2, 2, 64))}, config_payload=config)


def test_infer_cli_accepts_optional_current_config() -> None:
    from ppg_frailty.v5.cli import build_parser

    base = ["infer", "--model-config", "model_config/demo", "--input-manifest", "input.yaml"]
    assert build_parser().parse_args(base).config is None
    assert build_parser().parse_args([*base, "--config", "current.yaml"]).config == "current.yaml"


def test_live_inference_reads_current_yaml_not_saved_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ppg_frailty.v5 import inference_service as service

    saved = _config()
    current = copy.deepcopy(saved)
    current["signal"]["ppg_filter"]["high_hz"] = 7.0
    old_path, current_path = tmp_path / "saved.yaml", tmp_path / "current.yaml"
    old_path.write_text(yaml.safe_dump(saved))
    current_path.write_text(yaml.safe_dump(current))
    input_path = tmp_path / "input.yaml"
    input_path.write_text("participant_id: p")
    monkeypatch.setattr(service, "_resolve_export", lambda *_: (old_path, tmp_path, {"model_role": "fold"}))

    class ConfigurationReached(Exception):
        pass

    def check(payload: dict) -> None:
        assert payload["signal"]["ppg_filter"]["high_hz"] == 7.0
        raise ConfigurationReached

    monkeypatch.setattr(service, "_assert_supported_raw_contract", check)
    with pytest.raises(ConfigurationReached):
        service.infer_from_manifest(model_config_directory=tmp_path, input_manifest=input_path,
                                    config_path=current_path, pipeline_root=tmp_path)
