"""Stage CLI loads concise presets and the same exploratory snapshots as Dash."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from ppg_frailty.dashboard.workflow_controls import default_configuration
from ppg_frailty.dashboard.workflow_service import WorkflowService
from ppg_frailty.features.registry import FEATURE_GROUP_ORDER

ROOT = Path(__file__).resolve().parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location('stage_analyse_for_test', ROOT / 'stage_analyse.py')
stage_analyse = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(stage_analyse)


@pytest.mark.parametrize('preset', ['baseline', 'finalcase'])
def test_stage_cli_materializes_missing_defaults_and_passes_current_overrides(monkeypatch, capsys, preset):
    observed = {}

    def analyse(self, stage, config, record_id, **kwargs):
        observed.update(config=config, stage=stage, record_id=record_id, kwargs=kwargs)
        return {'status': 'ok', 'stage': stage}

    monkeypatch.setattr(WorkflowService, 'analyse', analyse)
    assert stage_analyse.main(['--config', str(ROOT / f'configs/presets/{preset}.yaml'), '--stage', 'quality',
        '--record-id', 'record_1', '--set', 'signal.ppg_filter.high_hz=7.5']) == 0
    config = observed['config']
    assert 'flatline_duration_s' in config['quality']
    assert config['artifact']['denoiser_enabled'] is False
    assert config['routing']
    assert config['signal']['ppg_filter']['high_hz'] == 7.5
    assert observed['stage'] == 'quality' and observed['record_id'] == 'record_1'
    assert json.loads(capsys.readouterr().out)['status'] == 'ok'


def test_stage_cli_accepts_raw_feature_exploration_without_training_contract(monkeypatch, tmp_path):
    config = default_configuration(ROOT)
    config['features']['enabled_groups'] = [FEATURE_GROUP_ORDER[0]]
    source = tmp_path / 'exploration.yaml'
    source.write_text(yaml.safe_dump(config), encoding='utf-8')
    seen = []
    monkeypatch.setattr(WorkflowService, 'analyse', lambda self, stage, current, record, **kwargs:
                        seen.append(current) or {'status': 'ok'})
    assert stage_analyse.main(['--config', str(source), '--stage', 'features']) == 0
    assert seen[0]['features']['enabled_groups'] == [FEATURE_GROUP_ORDER[0]]


def test_stage_cli_invalid_configuration_is_not_replaced_with_defaults(monkeypatch, tmp_path, capsys):
    config = default_configuration(ROOT)
    config['signal']['ppg_filter']['low_hz'] = 9.0
    source = tmp_path / 'invalid.yaml'
    source.write_text(yaml.safe_dump(config), encoding='utf-8')
    def unexpected(*args, **kwargs):
        pytest.fail('invalid filter settings must not silently reach the stage with defaults')
    monkeypatch.setattr(WorkflowService, 'analyse', unexpected)
    assert stage_analyse.main(['--config', str(source)]) == 2
    assert json.loads(capsys.readouterr().err)['status'] == 'error'
