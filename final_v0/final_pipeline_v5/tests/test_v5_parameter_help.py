"""Parameter explanations and folding must not change executable controls."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re

import pytest
import yaml

pytest.importorskip('dash')
from ppg_frailty.dashboard import create_app
from ppg_frailty.dashboard.app import TOOLS, _parameter_control, _tool_parser
from ppg_frailty.dashboard.control_service import V5ControlService
from ppg_frailty.dashboard.tool_controls import command_parameter_specs, plan_parameter_specs
from ppg_frailty.dashboard.workflow_controls import (
    apply_control_values, default_configuration, grouped_parameter_specs,
)
from ppg_frailty.module_registry import list_modules
from ppg_frailty.v5.configuration import PRESETS, _DIRECT_MODULE_PATHS

ROOT = Path(__file__).resolve().parents[1]
FOLDED_STAGES = {'imu', 'quality', 'denoiser', 'features', 'representation', 'model'}
MAIN_SELECTORS = {
    'imu': {'signal.imu.gravity_method'},
    'quality': {'quality.mode', 'quality.calibrator'},
    'denoiser': {'artifact.reducer', 'artifact.denoiser_enabled'},
    'features': set(),
    'representation': {'representation_mode'},
    'model': {'model.model_id'},
}


def components(component):
    return [component, *component._traverse()]


def control_paths(component):
    return {identity['path'] for item in components(component)
            if isinstance(identity := getattr(item, 'id', None), dict)
            and identity.get('type') in {'param', 'param-number', 'param-slide'}}


@pytest.fixture(scope='module')
def all_core_specs():
    """Visit actual registered alternatives, including nested fusion encoders."""
    from ppg_frailty.models.factory import FUSION_SIGNAL_ENCODER_IDS

    base = default_configuration(ROOT)
    configurations = [base]
    service = V5ControlService(ROOT)
    configurations.extend(service.load_yaml(preset.relative_path)[0] for preset in PRESETS.values())
    for family, path in [('artifact', 'artifact.reducer'), ('peak_detector', 'signal.peak_detector.detector_id'),
                         *_DIRECT_MODULE_PATHS.items()]:
        configurations.extend(apply_control_values(base, {path: module['module_id']}, pipeline_root=ROOT)
                              for module in list_modules(family))
    configurations.extend(apply_control_values(base, {
        'representation_mode': module['representation_modes'][0], 'model.model_id': module['module_id'],
    }, pipeline_root=ROOT) for module in list_modules('model'))
    fusion = apply_control_values(base, {'representation_mode': 'fusion'}, pipeline_root=ROOT)
    configurations.extend(apply_control_values(fusion, {'model.signal_encoder.model_id': model}, pipeline_root=ROOT)
                          for model in sorted(FUSION_SIGNAL_ENCODER_IDS))
    return {row['path']: row for config in configurations
            for row in grouped_parameter_specs(config, pipeline_root=ROOT)}


def test_all_registered_core_control_paths_have_visible_chinese_help(all_core_specs):
    from ppg_frailty.dashboard.parameter_help import parameter_help

    # Includes algorithms absent from finalcase, not only the initial layout.
    assert {'artifact.parameters.step_size', 'artifact.parameters.nmf_rank',
            'signal.imu.process_covariance_diagonal_per_second.0',
            'training.optimizer_parameters.betas.1', 'model.signal_encoder.model_id',
            'quality.morph_component_weights.template_correlation'} <= all_core_specs.keys()
    before = copy.deepcopy(all_core_specs)
    for path, spec in all_core_specs.items():
        assert spec.get('description'), path
        help_text = spec.get('description') or parameter_help(spec)
        assert isinstance(help_text, str) and re.search(r'[\u4e00-\u9fff]', help_text), path
        assert not help_text.startswith(('计划字段 ', 'CLI 参数 ')), path
        widget = _parameter_control(spec)
        explanations = [item for item in components(widget)
                        if 'parameter-help' in (getattr(item, 'className', '') or '').split()]
        assert len(explanations) == 1, path
        assert explanations[0].children == help_text, path
        assert getattr(explanations[0], 'hidden', False) is not True, path
        assert getattr(explanations[0], 'style', {}).get('display') != 'none', path
    assert all_core_specs == before


def test_help_preserves_control_identity_and_value_for_every_kind(all_core_specs):
    for path, spec in all_core_specs.items():
        widget = _parameter_control(spec)
        controls = {identity['type']: item for item in components(widget)
                    if isinstance(identity := getattr(item, 'id', None), dict)}
        value = spec['value']
        if spec['kind'] in {'number', 'integer'}:
            assert set(controls) == {'param-slide', 'param-number'}, path
            assert controls['param-slide'].value == controls['param-number'].value == value, path
            assert controls['param-slide'].step == controls['param-number'].step == spec['step'], path
        else:
            assert set(controls) == {'param'}, path
            actual = controls['param'].value
            if spec['kind'] == 'boolean':
                assert actual == ([True] if value else []), path
            elif spec['kind'] in {'select', 'multi'}:
                expected = '__inherit__' if value is None and None in spec['choices'] else value
                assert actual == expected, path
            else:
                assert (actual if isinstance(value, str) else yaml.safe_load(actual)) == value, path
        assert all(item.id['path'] == path for item in controls.values())


def test_explicit_description_is_displayed_for_additional_parameters():
    spec = {'path': 'extension.depth', 'kind': 'integer', 'value': 3,
            'min': 0, 'max': 20, 'step': 1, 'description': '控制扩展模块重复处理的次数。'}
    widget = _parameter_control(spec)
    explanation = next(item for item in components(widget)
                       if getattr(item, 'className', None) == 'parameter-help')
    assert explanation.children == spec['description']


def test_index_and_fusion_specific_help_is_not_replaced_by_parent_fallback():
    from ppg_frailty.dashboard.parameter_help import parameter_help
    from ppg_frailty.dashboard.parameter_help_model import MODEL_HELP
    from ppg_frailty.dashboard.parameter_help_signal import ARTIFACT_HELP, SIGNAL_HELP

    paths = ['training.optimizer_parameters.betas.0', 'training.optimizer_parameters.betas.1',
             'model.signal_encoder.dropout', 'quality.calibrator_quantiles.0',
             'quality.calibrator_quantiles.1', 'quality.cardiac_band_hz.0',
             'signal.imu.process_covariance_diagonal_per_second.0',
             'signal.imu.process_covariance_diagonal_per_second.4']
    exact_help = {**MODEL_HELP, **SIGNAL_HELP}
    for path in paths:
        assert exact_help[path] in parameter_help({'path': path})
        plan_path = 'cases.0.overrides.' + path.replace('.', '~1')
        assert exact_help[path] in parameter_help({'path': plan_path})
    for endpoint in (0, 1):
        key = f'preserve_band_hz.{endpoint}'
        for context in (None, {'artifact': {'reducer': 'spectral_mask'}}):
            assert ARTIFACT_HELP['spectral_mask'][key] in parameter_help(
                {'path': 'artifact.parameters.' + key}, context)


def test_advanced_cli_and_plan_controls_also_display_parameter_help():
    specifications = []
    for _, script, command, _ in TOOLS.values():
        specifications.extend((spec, 'tool-param') for spec in command_parameter_specs(_tool_parser(script), command))
    plan = {'study': {'name': 'example'}, 'cases': [{'overrides': {'training.batch_size': 16}}]}
    specifications.extend((spec, 'plan-param') for spec in plan_parameter_specs(plan))
    assert any(spec['path'].endswith('training~1batch_size') for spec, _ in specifications)
    for spec, namespace in specifications:
        widget = _parameter_control(spec, namespace)
        explanation = next(item for item in components(widget)
                           if getattr(item, 'className', None) == 'parameter-help')
        assert re.search(r'[\u4e00-\u9fff]', explanation.children), (namespace, spec['path'])
        assert all(identity['type'].startswith(namespace) for item in components(widget)
                   if isinstance(identity := getattr(item, 'id', None), dict))


def test_only_six_parameter_sections_start_folded_and_selectors_stay_visible(tmp_path):
    app = create_app(tmp_path)
    layout = app.layout()
    sections = {item.id: item for item in components(layout)
                if isinstance(getattr(item, 'id', None), str) and item.id.startswith('stage-')}
    folded = {item.id['stage']: item for item in components(layout)
              if isinstance(getattr(item, 'id', None), dict) and item.id.get('type') == 'stage-parameters'}
    assert set(folded) == FOLDED_STAGES
    specs = grouped_parameter_specs(default_configuration(tmp_path), pipeline_root=tmp_path)
    for stage, details in folded.items():
        assert details.__class__.__name__ == 'Details'
        assert details.open is False
        assert details.children[0].__class__.__name__ == 'Summary'
        assert details.children[0].children
        assert sum(item.__class__.__name__ == 'Details'
                   for item in components(sections['stage-' + stage])) == 1
        inside = control_paths(details)
        all_paths = control_paths(sections['stage-' + stage])
        expected = {spec['path'] for spec in specs if spec['stage'] == stage}
        assert all_paths == expected
        assert MAIN_SELECTORS[stage].isdisjoint(inside)
        assert MAIN_SELECTORS[stage] & expected <= all_paths - inside
        assert expected - MAIN_SELECTORS[stage] <= inside
        inner_ids = [getattr(item, 'id', None) for item in components(details)]
        assert {'type': 'stage-controls', 'stage': stage} in inner_ids
        assert not any(isinstance(identity, dict) and identity.get('type') == 'analyse-stage'
                       for identity in inner_ids)
        assert not {'train', 'stop-train', 'calibration-path', 'sqi-artifact', 'model-export',
                    'model-case', 'model-bundle', 'model-mode'} & {
                        identity for identity in inner_ids if isinstance(identity, str)}


def test_stage_preview_does_not_add_another_collapse_control():
    from ppg_frailty.dashboard.app import _render_preview

    preview = _render_preview({'metadata': {'stage': 'imu'}})
    assert not any(item.__class__.__name__ == 'Details'
                   for block in preview for item in components(block))


def test_callback_updates_cannot_recreate_static_parameter_folds(tmp_path):
    app = create_app(tmp_path)
    for definition in app.callback_map.values():
        outputs = definition['output']
        for output in outputs if isinstance(outputs, list) else [outputs]:
            identity = output.component_id
            if isinstance(identity, dict):
                assert identity.get('type') != 'stage-parameters'
            else:
                assert identity not in {'stage-' + stage for stage in FOLDED_STAGES}
    assert app.server.test_client().get('/_dash-layout').status_code == 200
    assert app.server.test_client().get('/_dash-dependencies').status_code == 200


def test_algorithm_and_yaml_changes_only_replace_children_of_open_folds():
    from dash._callback_context import context_value
    from dash._utils import AttributeDict

    app = create_app(ROOT)
    layout = app.layout()
    folds = [item for item in components(layout) if isinstance(getattr(item, 'id', None), dict)
             and item.id.get('type') == 'stage-parameters']
    for fold in folds:
        fold.open = True
    stages = [item.id for item in components(layout) if isinstance(getattr(item, 'id', None), dict)
              and item.id.get('type') == 'stage-controls']
    selectors = [item.id for item in components(layout) if isinstance(getattr(item, 'id', None), dict)
                 and item.id.get('type') == 'stage-selectors']
    update = next(value['callback'].__wrapped__ for key, value in app.callback_map.items() if 'config-state.data' in key)
    config = default_configuration(ROOT)
    selector = {'type': 'param', 'path': 'artifact.reducer'}
    cases = [(json.dumps(selector, sort_keys=True, separators=(',', ':')), None, [selector], ['nlms_imu_anc']),
             ('training-yaml', 'configs/presets/finalcase.yaml', [], [])]
    for trigger, yaml_path, ids, values in cases:
        token = context_value.set(AttributeDict(triggered_inputs=[{'prop_id': trigger + '.value', 'value': 1}]))
        try:
            config, status, panels, visible_selectors = update(
                yaml_path, None, values, [], ids, [], config, stages, selectors)
        finally:
            context_value.reset(token)
        assert 'Error' not in status
        assert all(fold.open is True for fold in folds)
        for panel in [*panels, *visible_selectors]:
            assert isinstance(panel, list)
            assert not any(getattr(item, 'id', None) == fold.id for child in panel
                           for item in components(child) for fold in folds)
    assert config['model']['model_id'] == 'InceptionTimeSmall'
