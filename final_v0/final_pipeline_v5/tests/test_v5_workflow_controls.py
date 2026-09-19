"""The notebook controls adapt existing algorithms; they do not implement them."""
from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest

from ppg_frailty.config import validate_config_payload
from ppg_frailty.dashboard.workflow_controls import (
    STAGES, apply_control_values, default_configuration, grouped_parameter_specs,
)
from ppg_frailty.module_registry import list_modules


def test_defaults_are_function_values_without_loading_a_yaml(monkeypatch):
    original = Path.read_text

    def read_text(path, *args, **kwargs):
        if path.suffix in {'.yaml', '.yml'}:
            raise AssertionError('default_configuration must not load a YAML preset')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read_text)
    from ppg_frailty.models.compact_cnn import CompactCNN1D
    from ppg_frailty.normalization import RawNormalizationConfig
    from ppg_frailty.training.trainer import TrainingConfig

    config = default_configuration()
    assert validate_config_payload(copy.deepcopy(config)) == config
    assert config['training'] == TrainingConfig().to_mapping()
    assert config['signal']['normalization'] == RawNormalizationConfig().to_mapping()
    defaults = inspect.signature(CompactCNN1D).parameters
    assert config['model']['dropout'] == defaults['dropout'].default
    assert config['model']['kernel_sizes'] == list(defaults['kernel_sizes'].default)
    assert config['signal']['ppg_filter']['low_hz'] == 0.2
    assert config['signal']['ppg_filter']['high_hz'] == 8.0
    assert config['signal']['dl_resampling']['enabled'] is False


def test_specs_are_workflow_ordered_unique_and_have_real_controls():
    config = default_configuration()
    rows = grouped_parameter_specs(config)
    paths = [row['path'] for row in rows]
    assert len(paths) == len(set(paths))
    assert [STAGES.index(row['stage']) for row in rows] == sorted(STAGES.index(row['stage']) for row in rows)
    assert {'signal.ppg_filter.low_hz', 'signal.ppg_filter.high_hz', 'signal.ppg_filter.order',
            'signal.imu.gravity_method', 'artifact.motion_detector_enabled', 'quality.mode',
            'artifact.reducer', 'features.enabled_groups', 'representation_mode', 'model.model_id'} <= set(paths)
    assert {'signal.ppg_filter.notch_enabled', 'signal.internal_fs_hz',
            'training.outer_labels_visible_to_trainer', 'training.refit_on_all_outer_training'} .isdisjoint(paths)
    assert next(row for row in rows if row['path'] == 'artifact.motion_detector_enabled')['stage'] == 'motion'
    assert not any(path.startswith('output.') for path in paths)
    for row in rows:
        assert {'path', 'value', 'kind', 'choices', 'min', 'max', 'step', 'group', 'stage'} <= row.keys()
        if row['kind'] in {'number', 'integer'}:
            assert row['step'] > 0
            assert row['min'] < row['max']
        if row['kind'] == 'select':
            assert len(row['choices']) > 1


def test_unchanged_full_widget_mapping_does_not_reset_loaded_values():
    config = apply_control_values(default_configuration(), {
        'signal.ppg_filter.high_hz': 7.5, 'training.fixed_epochs': 17,
        'signal.imu.gravity_method': 'sensor_filter_only_no_gravity_removal',
    }, validate=True)
    before = copy.deepcopy(config)
    controls = {row['path']: row['value'] for row in grouped_parameter_specs(config)}
    assert apply_control_values(config, controls, validate=True) == config
    assert config == before


@pytest.mark.parametrize('descriptor', list_modules('artifact'), ids=lambda row: row['module_id'])
def test_every_reducer_selector_materializes_its_own_parameters(descriptor):
    from ppg_frailty.artifacts import get_reducer
    from ppg_frailty.artifacts.base import parameters_dict

    name = descriptor['module_id']
    config = apply_control_values(default_configuration(), {'artifact.reducer': name}, validate=True)
    reducer = get_reducer(name)
    assert config['artifact']['parameters'] == parameters_dict(getattr(reducer, 'config', {}))
    assert config['artifact']['denoiser_enabled'] is (name != 'identity')
    if name == 'dwt_a2_legacy':
        assert not any(row['path'].startswith('artifact.parameters.') for row in grouped_parameter_specs(config))


@pytest.mark.parametrize('descriptor', list_modules('model'), ids=lambda row: row['module_id'])
def test_every_registered_model_has_a_working_selector(descriptor):
    mode = descriptor['representation_modes'][0]
    config = apply_control_values(default_configuration(), {
        'representation_mode': mode, 'model.model_id': descriptor['module_id'],
    }, validate=True)
    assert config['representation_mode'] == mode
    assert config['model']['model_id'] == descriptor['module_id']
    specs = grouped_parameter_specs(config)
    assert specs
    replay = apply_control_values(config, {row['path']: row['value'] for row in specs}, validate=True)
    assert replay['model']['architecture_parameters'] == config['model']['architecture_parameters']


@pytest.mark.parametrize('descriptor', list_modules('imu_gravity'), ids=lambda row: row['module_id'])
def test_imu_switch_replaces_only_its_own_parameter_surface(descriptor):
    source = default_configuration()
    config = apply_control_values(source, {'signal.imu.gravity_method': descriptor['module_id']}, validate=True)
    assert config['signal']['imu']['gravity_method'] == descriptor['module_id']
    assert config['signal']['ppg_filter'] == source['signal']['ppg_filter']
    assert config['model'] == source['model']


def test_vector_elements_and_ordinary_numeric_inputs_use_the_same_configuration():
    source = apply_control_values(default_configuration(), {'artifact.reducer': 'nlms_imu_anc'}, validate=True)
    config = apply_control_values(source, {'model.kernel_sizes.0': 11, 'artifact.parameters.delay_taps.0': 1}, validate=True)
    assert config['model']['kernel_sizes'][0] == 11
    assert config['artifact']['parameters']['delay_taps'][0] == 1
    assert source['model']['kernel_sizes'][0] == 9


def test_numeric_entry_is_not_limited_by_slider_display_bounds():
    config = default_configuration()
    row = next(row for row in grouped_parameter_specs(config) if row['path'] == 'training.learning_rate')
    typed = row['max'] * 2
    result = apply_control_values(config, {'training.learning_rate': typed}, validate=True)
    assert result['training']['learning_rate'] == typed


def test_incomplete_edit_is_allowed_but_execution_validates_canonical_math():
    config = default_configuration()
    draft = apply_control_values(config, {'signal.ppg_filter.low_hz': 9.0})
    assert draft['signal']['ppg_filter']['low_hz'] == 9.0
    with pytest.raises(ValueError, match='band edges'):
        apply_control_values(draft, {}, validate=True)
    motion = apply_control_values(config, {'artifact.motion_detector_enabled': True})
    assert motion['artifact']['motion_detector_enabled'] is True
    with pytest.raises(ValueError, match='evidence_path'):
        apply_control_values(motion, {}, validate=True)


def test_selector_defaults_do_not_override_explicit_new_values():
    config = apply_control_values(default_configuration(), {
        'training.optimizer': 'sgd', 'training.optimizer_parameters.momentum': 0.8,
        'signal.imu.gravity_method': 'calibrated_roll_pitch_ekf',
        'signal.imu.process_covariance_diagonal_per_second.0': 6.0,
    }, validate=True)
    assert config['training']['optimizer_parameters']['momentum'] == 0.8
    assert 'betas' not in config['training']['optimizer_parameters']
    assert config['signal']['imu']['process_covariance_diagonal_per_second'][0] == 6.0


def test_denoiser_off_selects_identity_without_retaining_unused_parameters():
    config = apply_control_values(default_configuration(), {'artifact.reducer': 'nmf_bss'}, validate=True)
    result = apply_control_values(config, {'artifact.denoiser_enabled': False}, validate=True)
    assert result['artifact']['reducer'] == 'identity'
    assert result['artifact']['parameters'] == {}


def test_feature_multiselect_and_raw_channel_subset_roundtrip():
    config = apply_control_values(default_configuration(), {'representation_mode': 'feature_vector'}, validate=True)
    config = apply_control_values(config, {'features.enabled_groups': ['ppi_basic_rate']}, validate=True)
    assert config['features']['enabled_groups'] == ['ppi_basic_rate']
    config = apply_control_values(default_configuration(), {'model.input_channel_order': ['IR', 'RED']}, validate=True)
    assert config['model']['input_channel_order'] == ['RED', 'IR']
    assert config['model']['input_channels'] == 2


def test_omitted_reducer_parameters_are_shown_without_mutating_yaml_values():
    source = apply_control_values(default_configuration(), {'artifact.reducer': 'nlms_imu_anc'}, validate=True)
    source['artifact']['parameters'] = {}
    before = copy.deepcopy(source)
    paths = {row['path'] for row in grouped_parameter_specs(source)}
    assert 'artifact.parameters.step_size' in paths
    assert 'artifact.parameters.delay_taps.0' in paths
    assert source == before


def test_window_selection_off_restores_its_neutral_parameters():
    source = apply_control_values(default_configuration(), {
        'quality.window_selection.policy': 'legacy_per_file_top_fraction',
        'quality.window_selection.keep_fraction': 0.5,
    }, validate=True)
    result = apply_control_values(source, {'quality.window_selection.policy': 'none'}, validate=True)
    assert result['quality']['window_selection']['keep_fraction'] == 1.0


@pytest.mark.parametrize('descriptor', list_modules('peak_detector'), ids=lambda row: row['module_id'])
def test_every_peak_algorithm_replaces_incompatible_parameters(descriptor):
    result = apply_control_values(default_configuration(), {'signal.peak_detector.detector_id': descriptor['module_id']}, validate=True)
    assert result['signal']['peak_detector']['detector_id'] == descriptor['module_id']


def test_vector_parent_can_remove_clipping_without_stale_children():
    config = default_configuration()
    controls = {row['path']: row['value'] for row in grouped_parameter_specs(config)}
    controls['signal.normalization.clip_after_scale'] = None
    result = apply_control_values(config, controls, validate=True)
    assert result['signal']['normalization']['clip_after_scale'] is None
    assert any(row['path'] == 'signal.normalization.clip_after_scale' for row in grouped_parameter_specs(result))
