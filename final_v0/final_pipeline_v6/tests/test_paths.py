"""Root-layout portability without loading recordings or fitting any model."""

from pathlib import Path

import pytest

from ppg_frailty import paths
from ppg_frailty.dashboard.control_service import V5ControlService
from ppg_frailty.dashboard.preview_service import PipelinePreviewService
from ppg_frailty.dashboard.workflow_service import WorkflowService
from ppg_frailty.pipeline import PipelinePaths


@pytest.mark.parametrize("nested", [False, True])
def test_data_root_supports_standalone_and_historical_layout(tmp_path, monkeypatch, nested):
    monkeypatch.delenv("PPG_FRAILTY_DATA_ROOT", raising=False)
    distribution = tmp_path / "final_v0/final_pipeline_v6" if nested else tmp_path
    assert paths.resolve_repository_root(distribution) == tmp_path


def test_explicit_data_root_takes_precedence_and_expands_home(tmp_path, monkeypatch):
    monkeypatch.setenv("PPG_FRAILTY_DATA_ROOT", "~/recordings")
    assert paths.resolve_repository_root(tmp_path) == (Path.home() / "recordings").resolve()


def test_shipped_resources_never_follow_external_data_root(tmp_path, monkeypatch):
    distribution = tmp_path / "release"
    monkeypatch.setattr(paths, "__file__", str(distribution / "src/ppg_frailty/paths.py"))
    monkeypatch.setenv("PPG_FRAILTY_DATA_ROOT", str(tmp_path / "external_data"))
    assert paths.pipeline_resource("splits/sgkf5_seed42_v2.csv") == (
        distribution / "splits/sgkf5_seed42_v2.csv"
    )
    assert paths.resolve_repository_root() == tmp_path / "external_data"
    monkeypatch.delenv("PPG_FRAILTY_DATA_ROOT")
    assert paths.resolve_repository_root() == distribution


def test_pipeline_discovery_keeps_packaged_inputs_with_external_data(tmp_path, monkeypatch):
    monkeypatch.setenv("PPG_FRAILTY_DATA_ROOT", str(tmp_path))
    discovered = PipelinePaths.discover()
    distribution = Path(__file__).resolve().parents[1]
    assert discovered.pipeline_root == distribution
    assert discovered.repository_root == tmp_path
    assert discovered.input_path("configs/presets/finalcase.yaml") == (
        distribution / "configs/presets/finalcase.yaml"
    )


@pytest.mark.parametrize("service", [WorkflowService, PipelinePreviewService, V5ControlService])
@pytest.mark.parametrize("layout", ["standalone", "nested", "external_data"])
def test_dash_services_share_root_resolution(tmp_path, monkeypatch, service, layout):
    monkeypatch.delenv("PPG_FRAILTY_DATA_ROOT", raising=False)
    distribution = tmp_path / "release"
    expected_data_root = distribution
    if layout == "nested":
        distribution = tmp_path / "final_v0/final_pipeline_v6"
        expected_data_root = tmp_path
    elif layout == "external_data":
        expected_data_root = tmp_path / "external_data"
        monkeypatch.setenv("PPG_FRAILTY_DATA_ROOT", str(expected_data_root))
    instance = service(distribution)
    assert instance.pipeline_root == distribution
    assert instance.repository_root == expected_data_root


def test_dash_accepts_packaged_config_and_external_recording(tmp_path, monkeypatch):
    distribution, data_root = tmp_path / "release", tmp_path / "external_data"
    monkeypatch.setenv("PPG_FRAILTY_DATA_ROOT", str(data_root))
    service = V5ControlService(distribution)
    config = distribution / "configs/presets/finalcase.yaml"
    recording = data_root / "PPG_Testing_05_01_2026/participant_B.csv"
    assert service.safe_input(config, label="config", must_exist=False) == config
    assert service.safe_input(recording, label="recording", must_exist=False) == recording
    assert service.cli_input_path(config) == "configs/presets/finalcase.yaml"
    assert service.cli_input_path(recording) == str(recording)
