"""Public command widgets round-trip through the original argparse parsers."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pytest
import yaml

from ppg_frailty.dashboard.tool_controls import (
    apply_plan_values, arguments_from_values, command_parameter_specs, plan_parameter_specs,
)
from ppg_frailty.v5 import cli, specialized, sweep
from ppg_frailty.v5_reporting import cli as report_cli


@pytest.mark.parametrize('factory', [cli.build_parser, sweep.build_parser, specialized.build_parser, report_cli.build_parser])
def test_every_public_command_argument_is_described(factory):
    parser = factory()
    commands = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    for name, command in commands.choices.items():
        specs = command_parameter_specs(parser, name)
        expected = {action.dest for action in command._actions if action.dest != 'help' and action.help != argparse.SUPPRESS}
        assert {row['path'] for row in specs} == expected
        assert len(specs) == len(expected)
        json.dumps(specs, allow_nan=False)
        for row in specs:
            assert {'path', 'value', 'kind', 'choices', 'min', 'max', 'step', 'group', 'stage',
                    'arg', 'type', 'required', 'default', 'array'} <= row.keys()
            if row['kind'] in {'number', 'integer'}:
                assert row['step'] > 0


def test_pipeline_arguments_preserve_inherited_boolean_and_append_values():
    parser = cli.build_parser()
    specs = command_parameter_specs(parser, 'run')
    by_path = {row['path']: row for row in specs}
    assert by_path['continue_on_error']['choices'] == [None, True, False]
    assert by_path['manual']['exclusive_required'] is True
    values = {'manual': True, 'assignments': ['training.fixed_epochs=2', 'roles=[B, R1]'],
              'module': ['artifact=nlms_imu_anc'], 'jobs': 2.0, 'repeats': '0,1', 'refit': True}
    argv = arguments_from_values(specs, values)
    result = parser.parse_args(['run', *argv])
    assert result.manual is True
    assert result.assignments == values['assignments']
    assert result.module == values['module']
    assert result.jobs == 2 and result.repeats == (0, 1) and result.refit is True
    assert result.continue_on_error is None
    assert '--continue-on-error' not in argv and '--no-continue-on-error' not in argv
    assert parser.parse_args(['run', *arguments_from_values(specs, {**values, 'continue_on_error': False})]).continue_on_error is False


def test_report_aliases_paths_and_negative_boolean_round_trip():
    parser = report_cli.build_parser()
    specs = command_parameter_specs(parser, 'run')
    values = {'input': '/tmp/run with spaces', 'module': ['roc', 'confusion'], 'figure': ['roc_curve'],
              'bootstrap_resamples': 7, 'v2_compatibility': False, 'output_name': 'review'}
    argv = arguments_from_values(specs, values)
    result = parser.parse_args(['run', *argv])
    assert result.input == Path(values['input'])
    assert result.module == values['module']
    assert result.figure == ['roc_curve'] and result.bootstrap_resamples == 7
    assert result.v2_compatibility is False
    assert '--module' in argv and '--modules' not in argv


def test_nargs_and_repeated_array_actions_use_original_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('--numbers', type=int, nargs='+')
    parser.add_argument('--pair', type=float, nargs=2, action='append')
    parser.add_argument('--quiet', dest='verbose', action='store_false', default=True)
    parser.add_argument('--level', action='count', default=0)
    specs = command_parameter_specs(parser, 'plain')
    argv = arguments_from_values(specs, {'source': 'record.csv', 'numbers': [0, -2, 5],
                                       'pair': [[0.2, 4.0], [0.5, 8.0]], 'verbose': False, 'level': 2})
    args = parser.parse_args(argv)
    assert args.source == 'record.csv' and args.numbers == [0, -2, 5]
    assert args.pair == [[0.2, 4.0], [0.5, 8.0]] and args.verbose is False and args.level == 2


def test_serialization_does_not_validate_or_silently_repair_invalid_commands():
    parser = cli.build_parser()
    specs = command_parameter_specs(parser, 'run')
    argv = arguments_from_values(specs, {'manual': True, 'preset': 'finalcase', 'jobs': 2.5})
    assert '2.5' in argv
    with pytest.raises(SystemExit):
        parser.parse_args(['run', *argv])


def test_plan_controls_cover_nested_cases_without_inventing_enums():
    plan = {'kind': 'motion_training', 'cases': [{'name': 'cnn', 'options': {'epochs': 4, 'lr': 0.001,
             'normalize': True, 'kernels': [3, 5], 'device': 'cuda:0', 'threshold': None}}], 'empty': {}}
    before = copy.deepcopy(plan)
    specs = plan_parameter_specs(plan)
    rows = {row['path']: row for row in specs}
    assert rows['cases.0.options.epochs']['kind'] == 'integer'
    assert rows['cases.0.options.lr']['kind'] == 'number'
    assert rows['cases.0.options.normalize']['kind'] == 'boolean'
    assert rows['cases.0.options.kernels']['kind'] == 'list'
    assert rows['kind']['kind'] == 'text' and rows['kind']['choices'] == []
    assert rows['cases.0.options.threshold']['value'] is None
    edits = {row['path']: row['value'] for row in specs}
    edits.update({'cases.0.options.epochs': 10000, 'cases.0.options.normalize': False,
                  'cases.0.options.kernels': [3, 7, 11]})
    result = apply_plan_values(plan, edits)
    assert result['cases'][0]['options']['epochs'] == 10000
    assert result['cases'][0]['options']['normalize'] is False
    assert result['cases'][0]['options']['kernels'] == [3, 7, 11]
    assert plan == before
    assert plan_parameter_specs({}) == []


def test_flat_override_keys_and_numeric_mapping_keys_are_preserved():
    plan = {'overrides': {'training.learning_rate': 0.001, 'a~b': True}, 'labels': {0: 'static', 1: 'motion'}}
    specs = plan_parameter_specs(plan)
    values = {row['path']: row['value'] for row in specs}
    assert apply_plan_values(plan, values) == plan
    values['overrides.training~1learning_rate'] = 0.01
    values['labels.0'] = 'rest'
    result = apply_plan_values(plan, values)
    assert result['overrides'] == {'training.learning_rate': 0.01, 'a~b': True}
    assert result['labels'] == {0: 'rest', 1: 'motion'}


def test_every_shipped_yaml_plan_is_lossless_through_controls():
    root = Path(__file__).resolve().parents[1]
    for path in (root / 'configs').rglob('*.yaml'):
        plan = yaml.safe_load(path.read_text(encoding='utf-8'))
        if not isinstance(plan, dict):
            continue
        specs = plan_parameter_specs(plan)
        paths = [row['path'] for row in specs]
        assert len(paths) == len(set(paths)), path
        assert apply_plan_values(plan, {row['path']: row['value'] for row in specs}) == plan, path
