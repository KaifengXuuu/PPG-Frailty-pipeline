"""Advanced command/plan widgets use the current values, never a stale request."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import shlex

import pytest
import yaml

pytest.importorskip('dash')
from ppg_frailty.dashboard import create_app
from ppg_frailty.dashboard.app import TOOLS, _tool_parser
from ppg_frailty.dashboard.tool_controls import command_parameter_specs
from ppg_frailty.dashboard.workflow_controls import default_configuration

ROOT = Path(__file__).resolve().parents[1]
EXECUTION = ('', 'all', 'all', 1, 'off', 'cache/preprocessing', [], '')


def callback(app, output):
    return next(row['callback'].__wrapped__ for key, row in app.callback_map.items() if output in key)


@contextmanager
def triggered(identity):
    from dash._callback_context import context_value
    from dash._utils import AttributeDict
    text = json.dumps(identity, sort_keys=True, separators=(',', ':')) if isinstance(identity, dict) else identity
    token = context_value.set(AttributeDict(triggered_inputs=[{'prop_id': text+'.n_clicks', 'value': 1}]))
    try:
        yield
    finally:
        context_value.reset(token)


class Jobs:
    def __init__(self):
        self.started, self.stopped = [], []
    def start_request(self, request, *, kind):
        self.started.append((request, kind))
        return 'job-1'
    def terminate(self, job):
        self.stopped.append(job)
    def status(self, job):
        return {'state': 'running', 'elapsed_s': 0, 'log_tail': []}


def widget_args(operation, *, ordinary=None, numbers=None, plan=None, plan_values=None, plan_numbers=None):
    def items(namespace, mapping):
        return ([{'type': namespace, 'path': key} for key in (mapping or {})], list((mapping or {}).values()))
    return (operation, '--jobs 999 --plan stale.yaml', *items('tool-param', ordinary),
            *items('tool-param-number', numbers), plan, *items('plan-param', plan_values),
            *items('plan-param-number', plan_numbers))


def all_components(children):
    for child in children:
        yield child
        yield from child._traverse()


def test_every_public_tool_parameter_is_rendered_with_its_real_widget(tmp_path):
    app = create_app(tmp_path)
    render = callback(app, 'tool-controls.children')
    for operation, (_, script, command, _) in TOOLS.items():
        children, style = render(operation)
        specs = command_parameter_specs(_tool_parser(script), command)
        identities = [getattr(component, 'id', None) for component in all_components(children)]
        ids = {(identity['type'], identity['path']) for identity in identities if isinstance(identity, dict)}
        for spec in specs:
            path, kind = spec['path'], spec['kind']
            if path == 'plan':
                assert style == {}
            elif kind in {'number', 'integer'}:
                assert {('tool-param-slide', path), ('tool-param-number', path)} <= ids
            else:
                assert ('tool-param', path) in ids
    components = {component.id: component for component in app.layout()._traverse()
                  if isinstance(getattr(component, 'id', None), str)}
    assert components['tool-arguments'].readOnly is True


def test_tool_analyse_reads_live_widgets_and_ignores_stale_request_and_cli_text(tmp_path):
    jobs = Jobs()
    app = create_app(tmp_path, job_manager=jobs)
    old, *_ = callback(app, 'tool-request.data')('pipeline_index', '--study-dir pipeline_output/old')
    current = widget_args('pipeline_index', ordinary={'study_dir': 'pipeline_output/current', 'hash_predictions': [True]})
    with triggered('analyse-tool'):
        job, status = callback(app, 'active-tool-job.data')(1, 0, current[0], old, None, *current[1:])
    assert job == 'job-1', status
    request, kind = jobs.started[0]
    assert kind == 'tool'
    assert request.arguments == ('index', '--study-dir', 'pipeline_output/current', '--hash-predictions')
    assert '999' not in request.arguments


@pytest.mark.parametrize('name', ['yes', 'on', '2026-01-01'])
def test_optional_cli_text_preserves_literal_strings_instead_of_yaml_coercion(tmp_path, name):
    app = create_app(tmp_path)
    current = widget_args('execution_audit', ordinary={'input': 'pipeline_output/run', 'output_name': name})
    request, command, *_ = callback(app, 'tool-request.data')(*current)
    assert request is not None, command
    parsed = _tool_parser(request['script']).parse_args(request['arguments'])
    assert parsed.output_name == name


def test_plan_source_read_controls_download_and_execution_have_one_snapshot(tmp_path):
    configs = tmp_path/'configs'
    configs.mkdir()
    source = configs/'study.yaml'
    original = {'schema_version': 'example', 'candidates': [{'overrides': {'training.learning_rate': .001}}],
                'execution': {'jobs': 1, 'continue_on_error': False}}
    original_text = yaml.safe_dump(original)
    source.write_text(original_text)
    jobs = Jobs()
    app = create_app(tmp_path, job_manager=jobs)
    state, rendered, status = callback(app, 'tool-plan-state.data')('configs/study.yaml', '')
    assert state['plan'] == original and rendered
    current = widget_args('specialized_pipeline_validate', plan=state,
                         plan_numbers={'candidates.0.overrides.training~1learning_rate': .03, 'execution.jobs': 3},
                         plan_values={'execution.continue_on_error': [True]})
    request, command, resolved, _, _ = callback(app, 'tool-request.data')(*current)
    assert request is not None, command
    snapshot = yaml.safe_load(resolved)
    assert snapshot['candidates'][0]['overrides']['training.learning_rate'] == .03
    assert snapshot['execution'] == {'jobs': 3, 'continue_on_error': True}
    filename = Path(request['arguments'][request['arguments'].index('--plan')+1])
    assert filename.parent.as_posix() == 'pipeline_output/.dashboard_requests/plans'
    assert not (tmp_path/filename).exists()
    cli = callback(app, 'download-tool-cli.data')(1, None, *current)
    downloaded = callback(app, 'download-tool-yaml.data')(1, None, *current)
    assert yaml.safe_load(downloaded['content']) == snapshot
    assert downloaded['filename'] == filename.name
    lines = cli['content'].splitlines()
    assert shlex.split(lines[0])[:2] == ['python', '-c']
    assert resolved in shlex.split(lines[0])[2].replace('\\n', '\n') or 'write_text' in lines[0]
    assert lines[-1] == command
    assert source.read_text() == original_text and not (tmp_path/'pipeline_output').exists()
    with triggered('analyse-tool'):
        job, status = callback(app, 'active-tool-job.data')(1, 0, current[0], None, None, *current[1:])
    assert job == 'job-1', status
    assert (tmp_path/filename).read_text() == resolved
    assert source.read_text() == original_text
    assert jobs.started[0][0].arguments == tuple(request['arguments'])


def test_training_tool_uses_selected_plan_yaml_without_unrelated_pipeline_yaml(tmp_path):
    source = tmp_path/'motion.yaml'
    source.write_text('motion_detector:\n  training_device: cpu\n')
    jobs = Jobs()
    app = create_app(tmp_path, job_manager=jobs)
    state, _, _ = callback(app, 'tool-plan-state.data')(str(source), '')
    current = widget_args('specialized_pipeline_run', ordinary={'run_name': 'current'},
                         numbers={'jobs': 2}, plan=state, plan_values={'motion_detector.training_device': 'cuda'})
    with triggered('analyse-tool'):
        job, status = callback(app, 'active-tool-job.data')(1, 0, current[0], None, None, *current[1:])
    assert job is None and 'model Train mode' in status and not jobs.started
    assert not (tmp_path/'pipeline_output').exists()
    with triggered('train'):
        job, status = callback(app, 'active-train-job.data')(
            1, 0, 'Train', None, 'tool', None, [], None, None, {}, [], [], [], [], *EXECUTION, *current)
    assert job == 'job-1', status
    request, kind = jobs.started[0]
    parsed = _tool_parser(request.script).parse_args(request.arguments)
    assert parsed.jobs == 2 and parsed.run_name == 'current' and kind == 'pipeline'
    assert yaml.safe_load((tmp_path/parsed.plan).read_text())['motion_detector']['training_device'] == 'cuda'
    with triggered('stop-train'):
        callback(app, 'active-train-job.data')(
            0, 1, 'Analyse', None, 'tool', None, [], job, None, {}, [], [], [], [], *EXECUTION, *current)
    assert jobs.stopped == ['job-1']


@pytest.mark.parametrize('namespace', ['tool-param', 'plan-param'])
def test_tool_numeric_input_extends_slider_instead_of_clipping(tmp_path, namespace):
    app = create_app(tmp_path)
    with triggered({'type': namespace+'-number', 'path': 'jobs'}):
        assert callback(app, f'"type":"{namespace}-slide"')(1, 200, 0, 20) == (200, 200, 0, 200)


def test_bad_plan_is_reported_and_does_not_reuse_previous_state(tmp_path):
    source = tmp_path/'bad.yaml'
    source.write_text('- not\n- a mapping\n')
    app = create_app(tmp_path)
    state, rendered, status = callback(app, 'tool-plan-state.data')(str(source), '')
    assert state is None and rendered == [] and 'mapping' in status


@pytest.mark.parametrize('duration', [0, -1])
def test_invalid_preview_duration_is_not_replaced_or_allowed_to_break_error_display(tmp_path, duration):
    seen = []
    class Workflow:
        def analyse(self, **request):
            seen.append(request['duration_s'])
            raise ValueError('duration must be positive')
        def cached_previews(self, *args, **kwargs):
            raise ValueError('duration must be positive')
    app = create_app(tmp_path, workflow_service=Workflow())
    config = default_configuration(tmp_path)
    selection = ('person', '', None, '', None, '', None, None, None, '')
    with triggered({'type': 'analyse-stage', 'stage': 'ppg'}):
        result, stage = callback(app, 'preview-store.data')(
            [1], config, [], [], [], [], 'record', 'session', 0, duration, [], [], *selection)
    assert seen == [float(duration)]
    assert result['error'] == 'ValueError: duration must be positive' and result['previews'] == {}
    command = callback(app, 'command-view.children')(config, stage, 'record', 0, duration, [], [], *selection)
    arguments = shlex.split(command)
    assert arguments[arguments.index('--duration-s')+1] == str(duration)
