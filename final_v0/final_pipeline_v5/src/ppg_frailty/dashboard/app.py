"""Notebook-style control panel. Widgets own the current configuration."""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shlex
from typing import Any, Mapping
import uuid
import yaml

from .control_service import (CommandRequest, V5ControlService,
                              comparison_sequence_cli, comparison_sequence_export_yaml)
from .job_manager import DashboardJobManager
from .preview_service import PipelinePreviewService

STAGES = (
    ('input', 'Input recordings'), ('ppg', 'PPG preprocessing'),
    ('imu', 'IMU preprocessing'), ('motion', 'Motion detector'),
    ('quality', 'Signal quality'), ('denoiser', 'Motion denoiser'),
    ('features', 'Feature engineering'), ('representation', 'Representation'),
    ('model', 'Machine learning model'), ('aggregation', 'Aggregation'),
)
PARAMETER_PANELS = {
    'imu': ('gravity_method 参数', {'signal.imu.gravity_method'}),
    'quality': ('calibrator / SQI 参数', {'quality.mode', 'quality.calibrator'}),
    'denoiser': ('Reducer 参数', {'artifact.denoiser_enabled', 'artifact.reducer'}),
    'features': ('Feature 参数', set()),
    'representation': ('Representation 参数', {'representation_mode'}),
    'model': ('Model / Train 参数', {'model.model_id'}),
}
GRID = {'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit,minmax(min(100%,280px),1fr))', 'gap': '16px'}
TOOLS = {
    'pipeline_validate': ('Pipeline validate', 'pipeline.py', 'validate', False),
    'show_config': ('Show config', 'pipeline.py', 'show-config', False),
    'sweep_validate': ('Sweep validate', 'sweep.py', 'validate', False),
    'pipeline_index': ('Rebuild index', 'pipeline.py', 'index', False),
    'model_export': ('Export model config', 'pipeline.py', 'export-model-config', False),
    'pipeline_excel': ('Pipeline Excel', 'sweep.py', 'export-excel', False),
    'report_excel': ('Report Excel', 'analyse_report.py', 'export-excel', False),
    'execution_audit': ('Execution audit', 'analyse_report.py', 'execution-audit', False),
    'specialized_validate': ('Special validate', 'analyse_report.py', 'specialized-validate', False),
    'specialized_run': ('Special analyse', 'analyse_report.py', 'specialized-run', False),
    'specialized_report': ('Special report', 'analyse_report.py', 'specialized-report', False),
    'specialized_pipeline_validate': ('Special pipe check', 'specialized_pipeline.py', 'validate', False),
    'specialized_pipeline_run': ('Special pipe train', 'specialized_pipeline.py', 'run', True),
    'specialized_pipeline_complete': ('Special CV complete', 'specialized_pipeline.py', 'complete', True),
}


def _tool_parser(script: str) -> Any:
    from ..v5 import cli, sweep, specialized
    from ..v5_reporting import cli as reporting
    return {'pipeline.py': cli, 'sweep.py': sweep, 'specialized_pipeline.py': specialized,
            'analyse_report.py': reporting}[script].build_parser()


def _tool_request(operation: str, arguments: str) -> CommandRequest:
    """Use the actual public parser, not another dashboard option validator."""
    import contextlib
    import io
    _, script, command, _ = TOOLS[operation]
    argv = [command, *shlex.split(arguments or '')]
    stream = io.StringIO()
    try:
        with contextlib.redirect_stderr(stream), contextlib.redirect_stdout(stream):
            _tool_parser(script).parse_args(argv)
    except SystemExit as error:
        raise ValueError(stream.getvalue().strip()) from error
    return V5ControlService._command_request(script, argv)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _options(values: Any) -> list:
    return [{'label': str(v), 'value': v} for v in values]


def _button(label: str, identity: Any, **kwargs: Any) -> Any:
    from dash import html
    return html.Button(label, id=identity, n_clicks=0, **kwargs)


def _field(label: str, component: Any, description: str | None = None) -> Any:
    from dash import html
    from .parameter_help import field_help
    description = description or field_help(getattr(component, 'id', None))
    heading = [html.Span(label, className='parameter-name')]
    if description:
        heading.append(html.Span(description, className='parameter-help'))
    return html.Div([html.Label(heading, className='parameter-label'), component], className='field')


def _table(rows: list[dict]) -> Any:
    from dash import dash_table
    keys = list(dict.fromkeys(k for r in rows for k in r))
    rows = [{k: _json(v) if isinstance(v, (dict, list, tuple)) else v for k, v in r.items()} for r in rows]
    return dash_table.DataTable(
        data=rows, columns=[{'name': k, 'id': k} for k in keys],
        page_size=12, sort_action='native', filter_action='native',
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '7px', 'maxWidth': '450px',
                    'whiteSpace': 'normal', 'fontFamily': 'inherit', 'fontSize': 12})


def _parameter_control(spec: Mapping[str, Any], namespace: str = 'param') -> Any:
    from dash import dcc, html
    from .parameter_help import parameter_help
    path, value, kind = spec['path'], spec['value'], spec['kind']
    identity = {'type': namespace, 'path': path}
    if kind in {'number', 'integer'}:
        component = html.Div([
            dcc.Slider(id={'type': namespace+'-slide', 'path': path}, value=value,
                       min=spec['min'], max=spec['max'], step=spec['step'], marks=None,
                       tooltip={'placement': 'bottom', 'always_visible': False}),
            dcc.Input(id={'type': namespace+'-number', 'path': path}, value=value, type='number', step=spec['step'], debounce=True),
        ], className='numeric-control')
    elif kind == 'boolean':
        component = dcc.Checklist(id=identity, options=[{'label': 'Enabled', 'value': True}], value=[True] if value else [])
    elif kind in {'select', 'multi'}:
        choices = [{'label': 'Inherit / unset' if item is None else str(item),
                    'value': '__inherit__' if item is None else item} for item in spec['choices']]
        component = (dcc.Checklist(id=identity, options=choices, value=value or []) if kind == 'multi'
                     else dcc.Dropdown(id=identity, options=choices,
                         value='__inherit__' if value is None and None in spec['choices'] else value,
                         clearable=bool(spec.get('nullable'))))
    else:
        text = value if isinstance(value, str) else yaml.safe_dump(value, default_flow_style=True).strip().removesuffix('...').strip()
        component = dcc.Input(id=identity, value=text, type='text', debounce=True)
    label = path.replace('~1', '.').replace('~0', '~') if namespace == 'plan-param' else path
    return html.Div([html.Label([html.Span(label.rsplit('.', 1)[-1], className='parameter-name'),
                                html.Span(parameter_help(spec), className='parameter-help')],
                               title=label, className='parameter-label'), component,
                     html.Small(label, title=str(spec.get('range', '')))], className='parameter')


def _widget_values(specifications: list, identities: list | None, values: list | None) -> dict:
    specs = {row['path']: row for row in specifications}
    output = {}
    for identity, value in zip(identities or [], values or []):
        path = identity['path']
        if path not in specs:
            continue  # A replaced algorithm's widgets can still be in flight.
        kind = specs[path]['kind']
        if value == '__inherit__' and None in specs[path]['choices']:
            value = None
        elif kind == 'boolean':
            value = bool(value)
        elif kind == 'text' and 'action' in specs[path]:
            if specs[path]['value'] is None and (value is None or str(value).strip() in {'', 'null'}):
                value = None
        elif kind == 'list' or (kind == 'text' and not isinstance(specs[path]['value'], str)):
            value = yaml.safe_load(value) if isinstance(value, str) else value
        output[path] = value
    return output


def _control_values(config: Mapping, identities: list, values: list, root: Path) -> dict:
    from .workflow_controls import grouped_parameter_specs
    return _widget_values(grouped_parameter_specs(config, pipeline_root=root), identities, values)


def _tool_download_command(request: Mapping) -> str:
    """A downloaded plan command recreates its snapshot, without a UI-side write."""
    prefix = ''
    if request.get('config_sha256'):
        path = f"pipeline_output/.dashboard_requests/plans/{request['config_sha256']}.yaml"
        code = (f'from pathlib import Path; p=Path({path!r}); p.parent.mkdir(parents=True, exist_ok=True); '
                f"p.write_text({request['resolved_yaml']!r}, encoding='utf-8')")
        prefix = shlex.join(['python', '-c', code])+'\n'
    return prefix+request['display']+'\n'


def _render_preview(preview: Mapping | None) -> list:
    from dash import dcc, html
    import plotly.graph_objects as go
    if not preview:
        return [html.Div('Analyse computes missing upstream stages automatically.', className='empty-preview')]
    output = []
    if preview.get('traces'):
        from plotly.subplots import make_subplots
        def group(name):
            name = name.removeprefix('first_window_')
            if name.startswith('native_') or name in {'RED', 'IR'}:
                return 'PPG windows' if preview.get('metadata', {}).get('stage') == 'representation' else 'Native PPG'
            if name.startswith(('filtered_', 'direct_', 'reduced_')) or name.endswith('_peaks'):
                return 'Filtered / recovered PPG'
            if name in {'AX', 'AY', 'AZ'} or name.startswith(('acc_', 'gravity_', 'dynamic_')):
                return 'Acceleration'
            if name in {'GX', 'GY', 'GZ'} or name.startswith('gyro_'):
                return 'Angular velocity'
            if name.startswith('jerk_'):
                return 'Acceleration change'
            if name in {'roll_rad', 'pitch_rad'}:
                return 'Orientation'
            return 'Scores / state'
        groups = list(dict.fromkeys(group(name) for name in preview['traces']))
        figure = make_subplots(rows=len(groups), cols=1, shared_xaxes=True, subplot_titles=groups,
                               vertical_spacing=min(.1, .3 / len(groups)))
        for name, points in preview['traces'].items():
            figure.add_trace(go.Scattergl(x=points['x'], y=points['y'],
                                         mode='markers' if name.endswith('_peaks') else 'lines', name=name),
                             row=groups.index(group(name)) + 1, col=1)
        figure.update_layout(template='plotly_white', height=max(350, 220 * len(groups)),
                             margin=dict(l=50, r=20, t=35, b=40), legend=dict(orientation='h'), uirevision='signal')
        figure.update_xaxes(title_text='Time / s', row=len(groups), col=1)
        output.append(dcc.Graph(figure=figure, config={'displaylogo': False}))
    for title, rows in preview.get('tables', {}).items():
        if rows:
            output.extend([html.H4(title.replace('_', ' ')), _table(list(rows))])
            if title in {'recordings', 'participants'} and any(row.get('probabilities') for row in rows):
                chart = go.Figure()
                for row in rows:
                    if row.get('probabilities'):
                        names = preview.get('metadata', {}).get('class_names') or [f'class {i}' for i in range(len(row['probabilities']))]
                        chart.add_bar(x=names, y=row['probabilities'],
                                      name=str(row.get('participant_id') if title == 'participants' else row.get('file_id')))
                chart.update_layout(template='plotly_white', height=300, yaxis_title='Probability', yaxis_range=[0, 1])
                output.append(dcc.Graph(figure=chart, config={'displaylogo': False}))
    if preview.get('metadata'):
        output.append(html.Div([html.H4('Stage details'), html.Pre(_json(preview['metadata']))]))
    return output


def _asset_choices(root: Path) -> dict:
    found = {'sqi': [], 'motion': [], 'model': []}
    patterns = {'sqi': ('*sqi*calibration*.json', '*component*calibration*.json'),
                'motion': ('*motion*evidence*.json', '*internal*evidence*.json'),
                'model': ('manifest.json',)}
    for directory in ('pipeline_output', 'model_config', 'artifacts'):
        base = root / directory
        if not base.is_dir():
            continue
        for family, globs in patterns.items():
            for pattern in globs:
                for path in base.rglob(pattern):
                    if family == 'model' and not ('model' in str(path.parent) or 'bundle' in str(path.parent)):
                        continue
                    found[family].append(str((path.parent if family == 'model' else path).relative_to(root)))
    return {k: sorted(set(v)) for k, v in found.items()}


def _training_request(control: V5ControlService, config: Mapping, options: Mapping,
                      source_yaml: str | None = None) -> CommandRequest:
    """Replace entire sections, avoiding stale fields after algorithm switches."""
    source = source_yaml if source_yaml in control.yaml_paths() else 'configs/presets/baseline.yaml'
    rows = [{'path': key, 'value_yaml': yaml.safe_dump(value, default_flow_style=True).strip(),
             'original_yaml': '__dashboard_snapshot__'}
            for key, value in config.items() if key not in {'schema_version', 'config_id'}]
    return control.build_train_request(
        config_path=source, parameter_rows=rows, config_id=str(config['config_id']),
        repeats=options.get('repeats') or 'all', folds=options.get('folds') or 'all',
        jobs=int(options.get('jobs') or 1), device=str(config['training'].get('device', 'cpu')),
        cache_mode=options.get('cache_mode') or 'off', cache_root=options.get('cache_root') or 'cache/preprocessing',
        run_name=options.get('run_name') or None, refit=bool(options.get('refit')),
        dry_run=bool(options.get('dry_run')), resume=options.get('resume') or None)


def create_app(pipeline_root: str | Path | None = None, *, control_service=None,
               preview_service=None, job_manager=None, workflow_service=None) -> Any:
    from dash import ALL, MATCH, Dash, Input, Output, State, ctx, dcc, html, dash_table, no_update
    from .workflow_controls import default_configuration, grouped_parameter_specs, apply_control_values
    from .tool_controls import (command_parameter_specs, arguments_from_values,
                                plan_parameter_specs, apply_plan_values)
    from .workflow_service import WorkflowService
    from ..v5_reporting.registry import KNOWN_FIGURES, KNOWN_TABLES, MODULES, PRESETS
    from ..v5_reporting.contracts import REPORT_MODES

    control = control_service or V5ControlService(pipeline_root)
    root = control.pipeline_root
    browser = preview_service or PipelinePreviewService(root)
    jobs = job_manager or DashboardJobManager(root)
    workflow = workflow_service or WorkflowService(root)
    defaults = default_configuration(root)
    try:
        records = browser.records()
    except (OSError, ValueError):
        records = ()
    assets = _asset_choices(root)
    app = Dash(__name__, title='PPG Frailty · Workflow', suppress_callback_exceptions=True)
    app.index_string = '''<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
    <style>body{margin:0;background:#f6f7f9;color:#202b38;font-family:Arial,"Microsoft YaHei",sans-serif}
    *{box-sizing:border-box}h1{font-size:27px}h2{font-size:21px}h3{font-size:16px}h4{font-size:14px}
    .stage,.top-panel{background:white;border:1px solid #dde2e8;border-radius:8px;padding:22px;margin:18px 0}
    .stage-body{display:grid;grid-template-columns:minmax(280px,32%) minmax(0,1fr);gap:24px}
    .parameter{padding:10px 0;border-bottom:1px solid #edf0f3}.parameter label,.field>label{display:block;font-size:13px;font-weight:600;margin:0 0 8px}
    .parameter small{display:block;color:#77828e;font-size:10px;overflow-wrap:anywhere;margin-top:4px}
    .parameter label.parameter-label,.field>label.parameter-label{display:flex;flex-wrap:wrap;align-items:baseline;gap:5px 10px}
    .parameter-name{overflow-wrap:anywhere}.parameter-help{flex:1 1 220px;color:#526577;font-size:12px;font-weight:400;line-height:1.65}
    .parameter-panel{border:1px solid #dde2e8;border-radius:6px;padding:0 12px;margin:12px 0;background:#fafbfd}
    .parameter-panel>summary{padding:12px 0;color:#2165a6}.parameter-panel[open]>summary{border-bottom:1px solid #dde2e8}
    .numeric-control{display:grid;grid-template-columns:minmax(110px,1fr) 100px;align-items:center;gap:8px}
    input,textarea{max-width:100%;padding:7px;border:1px solid #cbd2da;border-radius:4px}input[type=text]{width:100%}
    button{background:#2165a6;color:white;border:0;border-radius:4px;padding:9px 16px;margin:5px 6px 5px 0;cursor:pointer;font-weight:600}
    button:disabled{opacity:.45;cursor:default}.stop{background:#ad3636}.actions{margin-top:16px}
    .empty-preview{min-height:180px;background:#fafbfd;border:1px dashed #d9e0e7;padding:22px;color:#7b8793}
    .notice{color:#637384;font-size:13px;line-height:1.6}.status{white-space:pre-wrap;color:#526577;line-height:1.5}
    pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:360px;overflow:auto}
    summary{cursor:pointer;padding:8px 0;font-weight:600}.field{margin:10px 0}.nav a{margin-right:14px;color:#2165a6}
    @media(max-width:900px){.stage-body{grid-template-columns:1fr}.stage{padding:14px}}
    </style></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>'''

    def path_picker(label, identity, choices):
        from .parameter_help import field_help
        return _field(label, html.Div([dcc.Dropdown(id=identity, options=_options(choices), value=None,
                                                    placeholder='Select existing artifact'),
                                      dcc.Input(id=identity+'-path', placeholder='Or enter a path', debounce=True)]),
                      field_help(identity))

    def controls_for(config, stage, *, selectors=False):
        output, previous = [], None
        visible = PARAMETER_PANELS.get(stage, ('', set()))[1]
        for spec in grouped_parameter_specs(config, pipeline_root=root):
            if spec['stage'] == stage and (spec['path'] in visible) == selectors:
                if spec['group'] != previous:
                    output.append(html.H3(str(spec['group']).replace('_', ' ')))
                    previous = spec['group']
                output.append(_parameter_control(spec))
        return output

    def tool_specs(operation):
        _, script, command, _ = TOOLS[operation]
        return command_parameter_specs(_tool_parser(script), command)

    def parameter_panel(specs, namespace):
        children, previous = [], None
        for spec in specs:
            if spec['group'] != previous:
                children.append(html.H3(str(spec['group']).replace('_', ' ')))
                previous = spec['group']
            children.append(_parameter_control(spec, namespace))
        return children

    def tool_snapshot(operation, arguments=None, identities=None, values=None, number_ids=None, numbers=None,
                      plan_state=None, plan_ids=None, plan_values=None, plan_number_ids=None, plan_numbers=None,
                      *, materialize=False):
        # Direct callers may still build an old CLI request. The browser always
        # supplies widget IDs (including an empty list); only widgets govern it.
        if identities is None:
            return _tool_request(operation, arguments)
        specs = tool_specs(operation)
        current_values = _widget_values(specs, identities + (number_ids or []), (values or []) + (numbers or []))
        plan_text, digest = '', ''
        if plan_state and any(spec['path'] == 'plan' for spec in specs):
            plan = plan_state['plan']
            edits = _widget_values(plan_parameter_specs(plan), (plan_ids or []) + (plan_number_ids or []),
                                   (plan_values or []) + (plan_numbers or []))
            plan = apply_plan_values(plan, edits)
            plan_text = yaml.safe_dump(plan, sort_keys=False, allow_unicode=True)
            digest = hashlib.sha256(plan_text.encode('utf-8')).hexdigest()
            current_values['plan'] = f'pipeline_output/.dashboard_requests/plans/{digest}.yaml'
        request = _tool_request(operation, shlex.join(arguments_from_values(specs, current_values)))
        if plan_text:
            request = replace(request, resolved_yaml=plan_text, config_sha256=digest)
            if materialize:
                destination = root / current_values['plan']
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(plan_text, encoding='utf-8')
        return request

    def stage_panel(stage, title, number):
        extra = []
        if stage == 'input':
            extra = [
                _field('Manifest recordings · one or more', dcc.Dropdown(id='record-ids', multi=True, value=[],
                    options=[{'label': f'{r.participant_id} · {r.role} · {r.record_id}', 'value': r.record_id} for r in records])),
                _field('Recording shown in stage plots', dcc.Dropdown(id='preview-record', options=[], value=None)),
                _field('Participant ID · custom CSV input', dcc.Input(id='participant-id', value='live_participant')),
                html.Details([html.Summary('Custom CSV files / labels'),
                    dash_table.DataTable(id='input-files', data=[], row_deletable=True, editable=True,
                        columns=[{'name': k, 'id': k} for k in ('file_id', 'role', 'path', 'label')],
                        style_table={'overflowX': 'auto'}, style_cell={'minWidth': '80px', 'textAlign': 'left'}),
                    _button('Add file', 'add-file'),
                    html.Div('A nonempty CSV table replaces the manifest selection. RED, IR, AX, AY, AZ, GX, GY, GZ; '
                             '400 Hz; g and deg/s. Dynamic R/S/W uses the same participant’s B.', className='notice')]),
                _field('Preview start / s', dcc.Input(id='preview-start', type='number', value=0, min=0)),
                _field('Preview duration / s', dcc.Input(id='preview-duration', type='number', value=20, min=.1)),
                html.Div('Full-record processing; the interval only controls the plot.', className='notice')]
        elif stage == 'imu':
            extra = [_field('Custom B calibration CSV · optional', dcc.Input(id='calibration-path', value='', debounce=True))]
        elif stage == 'motion':
            extra = [path_picker('Fitted motion evidence / weights', 'motion-bundle', assets['motion'])]
        elif stage == 'quality':
            extra = [path_picker('Fitted SQI calibration', 'sqi-artifact', assets['sqi'])]
        elif stage == 'model':
            extra = [
                _field('Mode', dcc.RadioItems(id='model-mode', options=_options(['Analyse', 'Train']), value='Analyse', inline=True)),
                _field('model_config weights', dcc.Dropdown(id='model-export', options=_options(control.model_exports()), value=None)),
                _field('Model case', dcc.Dropdown(id='model-case', options=[], value=None)),
                path_picker('Or a learned model bundle', 'model-bundle', assets['model']),
                html.Div('Weights do not replace current controls. Choose the exported YAML above only to load its parameter values.', className='notice'),
                html.Div([
                    _field('Training target', dcc.Dropdown(id='train-target', value='current', clearable=False, options=[
                        {'label': 'Current controls', 'value': 'current'}, {'label': 'Comparison queue', 'value': 'comparison'},
                        {'label': 'Selected study YAML', 'value': 'sweep'}, {'label': 'Advanced tool request', 'value': 'tool'}])),
                    _field('Run name', dcc.Input(id='run-name', value='', debounce=True)),
                    _field('Repeat indices', dcc.Input(id='repeat-indices', value='all', debounce=True)),
                    _field('Fold indices', dcc.Input(id='fold-indices', value='all', debounce=True)),
                    _field('Parallel jobs', dcc.Input(id='job-count', type='number', value=1, min=1, step=1)),
                    _field('Preprocessing cache', dcc.Dropdown(id='cache-mode', options=_options(['off', 'read_only', 'read_write']), value='off', clearable=False)),
                    _field('Cache directory', dcc.Input(id='cache-root', value='cache/preprocessing', debounce=True)),
                    _field('Resume directory', dcc.Input(id='resume-path', value='', debounce=True)),
                    _field('Training options', dcc.Checklist(id='train-flags', options=[{'label': 'Refit', 'value': 'refit'}, {'label': 'Dry run', 'value': 'dry_run'}], value=[])),
                ], id='train-options', style={'display': 'none'})]
        action = [_button('Analyse', {'type': 'analyse-stage', 'stage': stage})]
        if stage == 'model':
            action += [_button('Run', 'train', style={'display': 'none'}), _button('Stop', 'stop-train', className='stop'),
                       html.Pre(id='train-status', className='status')]
        parameters = html.Div(controls_for(defaults, stage), id={'type': 'stage-controls', 'stage': stage})
        if stage in PARAMETER_PANELS:
            # Keep this Details outside callback outputs: changing algorithms or
            # loading YAML replaces controls, not the user's expanded state.
            parameters = html.Div([
                html.Div(controls_for(defaults, stage, selectors=True), id={'type': 'stage-selectors', 'stage': stage}),
                html.Details([html.Summary(PARAMETER_PANELS[stage][0]), parameters],
                             id={'type': 'stage-parameters', 'stage': stage}, open=False, className='parameter-panel'),
            ])
        return html.Section([html.H2(f'{number:02d} · {title}'), html.Div([
            html.Div([*extra, parameters,
                      html.Div(action, className='actions')]),
            dcc.Loading(html.Div(_render_preview(None), id={'type': 'stage-output', 'stage': stage})),
        ], className='stage-body')], id='stage-'+stage, className='stage')

    def make_layout():
        return html.Main([
            dcc.Store(id='session-id', data=uuid.uuid4().hex, storage_type='session'),
            dcc.Store(id='config-state', data=defaults), dcc.Store(id='preview-store', data={}),
            dcc.Store(id='report-preview-job', data=None),
            dcc.Store(id='comparison-store', data=[]), dcc.Store(id='active-train-job'), dcc.Store(id='active-report-job'),
            dcc.Store(id='last-stage', data='ppg'), dcc.Store(id='train-request'), dcc.Store(id='analysis-request'),
            dcc.Store(id='tool-request'), dcc.Store(id='active-tool-job'), dcc.Store(id='tool-plan-state'),
            *[dcc.Download(id=name) for name in ('download-cli', 'download-yaml', 'download-sequence-cli', 'download-sequence-yaml',
                                                 'download-tool-cli', 'download-tool-yaml', 'download-report-cli')],
            dcc.Interval(id='job-poll', interval=1500),
            html.H1('PPG Frailty · Workflow'),
            html.Div('Choose inputs → adjust a module → Analyse → inspect the output.', className='notice'),
            html.Nav([html.A(title, href='#stage-'+stage) for stage, title in STAGES], className='nav'),
            html.Section([
                html.H2('Configuration'),
                _field('Optional YAML · empty means function defaults', dcc.Dropdown(id='training-yaml',
                    options=_options(sorted(set(control.yaml_paths()) | set(control.study_plan_paths()))),
                    value=None, placeholder='No YAML · function defaults', clearable=True)),
                _field('Study case', dcc.Dropdown(id='yaml-case', options=[], value=None)),
                _button('Refresh', 'refresh-configs'), html.Div(id='config-status', className='status'),
                html.Div('YAML fills controls once. Analyse always uses current values. Analysis never starts training.', className='notice'),
            ], className='top-panel'),
            *[stage_panel(stage, title, i) for i, (stage, title) in enumerate(STAGES, 1)],
            html.Section([
                html.H2('Comparison'), _field('Unit name', dcc.Input(id='comparison-name', value='case_01')),
                _button('Add', 'add-comparison'), _button('Remove last', 'remove-comparison'), _button('Clear', 'clear-comparison'),
                html.Div(id='comparison-table'), html.Div(id='comparison-status', className='status'),
                _button('Download CLI', 'export-sequence-cli'), _button('Download YAML', 'export-sequence-yaml'),
                html.Details([html.Summary('Comparison CLI / YAML'), html.Pre(id='comparison-cli-view'), html.Pre(id='comparison-yaml-view')]),
                html.Div('Use the model Train mode and select Comparison queue to execute.', className='notice'),
            ], className='stage'),
            html.Section([
                html.H2('11 · Analyse report'), html.Div([
                    html.Div([
                        _field('Pipeline output(s)', dcc.Dropdown(id='analysis-runs', options=_options(control.study_outputs()), multi=True, value=[])),
                        _field('Mode', dcc.Dropdown(id='analysis-mode', options=_options(sorted(REPORT_MODES)), value='single', clearable=False)),
                        _field('Report preset', dcc.Dropdown(id='analysis-preset', options=_options(sorted(PRESETS)), value='full', clearable=True)),
                        _field('Report modules', dcc.Checklist(id='analysis-modules', options=_options([m.name for m in MODULES]), value=[])),
                        _field('Figures · empty uses preset', dcc.Dropdown(id='analysis-figures', options=_options(sorted(KNOWN_FIGURES)), multi=True, value=[])),
                        _field('Tables · empty uses preset', dcc.Dropdown(id='analysis-tables', options=_options(sorted(KNOWN_TABLES)), multi=True, value=[])),
                        *[_field(label, dcc.Input(id=name, value='')) for name, label in (
                            ('reference-case', 'Reference case'), ('factor-paths', 'Factor paths · comma separated'),
                            ('include-cases', 'Include cases · comma separated'), ('exclude-cases', 'Exclude cases · comma separated'),
                            ('report-name', 'Output name · optional'))],
                        html.Div(controls_for(defaults, 'report'), id={'type': 'stage-controls', 'stage': 'report'}),
                        *[_field(label, dcc.Input(id=name, type='number', value=value)) for name, label, value in (
                            ('statistics-alpha', 'Alpha', .05), ('calibration-bins', 'Calibration bins', 10))],
                        _button('Analyse', 'analyse-report'), _button('Stop', 'stop-report', className='stop'),
                        _button('Download CLI', 'save-report-cli'),
                        html.Pre(id='analysis-command'), html.Pre(id='analysis-status', className='status')],
                    ),
                    html.Div([
                        _field('Report output preview', dcc.Dropdown(id='report-output', options=_options(control.report_outputs()), value=None)),
                        _field('Figure', dcc.Dropdown(id='report-figure', options=[], value=None)),
                        html.Img(id='report-image', style={'maxWidth': '100%'}),
                        html.Div(id='report-document'),
                        _field('Data table', dcc.Dropdown(id='report-table', options=[], value=None)),
                        html.Div(id='report-table-preview')]),
                ], className='stage-body'),
            ], className='stage'),
            html.Details([
                html.Summary('Advanced tools · export, audits and specialized studies'),
                _field('Operation', dcc.Dropdown(id='tool-operation', value='pipeline_validate', clearable=False,
                    options=[{'label': row[0], 'value': key} for key, row in TOOLS.items()])),
                html.Div(parameter_panel([spec for spec in tool_specs('pipeline_validate') if spec['path'] != 'plan'],
                                         'tool-param'), id='tool-controls'),
                html.Div([
                    _field('Plan YAML · fills the plan controls', dcc.Dropdown(id='tool-plan-yaml', value=None,
                        options=_options(sorted(str(path.relative_to(root)) for path in (root/'configs').rglob('*.yaml'))))),
                    _field('Or enter a plan YAML path', dcc.Input(id='tool-plan-path', value='', debounce=True)),
                    html.Div(id='tool-plan-status', className='status'),
                    html.Div(id='tool-plan-controls'),
                    html.Div('The source YAML is not overwritten. Current plan controls are saved only when you execute. '
                             'The downloaded CLI recreates the same snapshot.', className='notice'),
                ], id='tool-plan-picker', style={'display': 'none'}),
                html.Details([html.Summary('Equivalent CLI arguments'), dcc.Textarea(id='tool-arguments',
                    value='', readOnly=True, style={'width': '100%', 'minHeight': '80px'})]),
                html.Details([html.Summary('Command help'), html.Pre(id='tool-help')]),
                _button('Analyse', 'analyse-tool'), _button('Stop', 'stop-tool', className='stop'),
                _button('Download CLI', 'save-tool-cli'), _button('Download YAML', 'save-tool-yaml'),
                html.Div('Training tools execute only through the model Train → Advanced tool request → Run.', className='notice'),
                html.Pre(id='tool-command'), html.Pre(id='tool-yaml'), html.Pre(id='tool-status', className='status'),
            ], className='stage'),
            html.Section([
                html.H2('CLI / resolved YAML'), _button('Download CLI', 'save-cli'), _button('Download YAML', 'save-yaml'),
                html.Div('Save workflow_config.yaml beside the downloaded stage command; run from the V5 directory.', className='notice'),
                html.Details([html.Summary('Current Analyse CLI'), html.Pre(id='command-view')], open=True),
                html.Details([html.Summary('Training CLI · does not run automatically'), html.Pre(id='train-command')]),
                html.Details([html.Summary('Current YAML'), html.Pre(id='yaml-view')]),
            ], className='stage'),
        ], style={'maxWidth': '1600px', 'margin': 'auto', 'padding': 'clamp(12px,2vw,28px)'})

    app.layout = make_layout

    def current(config, identities, values):
        return apply_control_values(config, _control_values(config, identities, values, root), pipeline_root=root)

    @app.callback(Output('training-yaml', 'options'), Output('model-export', 'options'),
                  Output('analysis-runs', 'options'), Output('report-output', 'options'),
                  Output('sqi-artifact', 'options'), Output('motion-bundle', 'options'), Output('model-bundle', 'options'),
                  Input('refresh-configs', 'n_clicks'))
    def refresh(_):
        available = _asset_choices(root)
        return (_options(sorted(set(control.yaml_paths()) | set(control.study_plan_paths()))), _options(control.model_exports()),
                _options(control.study_outputs()), _options(control.report_outputs()),
                *[_options(available[family]) for family in ('sqi', 'motion', 'model')])

    @app.callback(Output('record-ids', 'options'), Output('record-ids', 'value'),
                  Input('config-state', 'data'), State('record-ids', 'value'))
    def manifest_records(config, selected):
        try:
            choices = browser.records(config['manifest']['path'])
        except (OSError, ValueError):
            return [], []
        ids = {record.record_id for record in choices}
        return ([{'label': f'{r.participant_id} · {r.role} · {r.record_id}', 'value': r.record_id} for r in choices],
                [record for record in selected or [] if record in ids])

    @app.callback(Output('yaml-case', 'options'), Output('yaml-case', 'value'), Input('training-yaml', 'value'))
    def yaml_cases(path):
        if not path or path not in control.study_plan_paths():
            return [], None
        from ..study import expand_study
        plan, _ = control.load_study_plan(path)
        cases = expand_study(plan, pipeline_root=root).cases
        return _options([c.case_id for c in cases]), cases[0].case_id if cases else None

    @app.callback(Output('config-state', 'data'), Output('config-status', 'children'),
                  Output({'type': 'stage-controls', 'stage': ALL}, 'children'),
                  Output({'type': 'stage-selectors', 'stage': ALL}, 'children'),
                  Input('training-yaml', 'value'), Input('yaml-case', 'value'), Input({'type': 'param', 'path': ALL}, 'value'),
                  Input({'type': 'param-number', 'path': ALL}, 'value'),
                  State({'type': 'param', 'path': ALL}, 'id'), State({'type': 'param-number', 'path': ALL}, 'id'),
                  State('config-state', 'data'), State({'type': 'stage-controls', 'stage': ALL}, 'id'),
                  State({'type': 'stage-selectors', 'stage': ALL}, 'id'))
    def update_config(path, case, values, numbers, identities, number_ids, config, stages, selector_stages):
        trigger = ctx.triggered_id
        try:
            if trigger in ('training-yaml', 'yaml-case') or trigger is None:
                if not path:
                    result = default_configuration(root)
                elif path in control.study_plan_paths():
                    from ..study import expand_study
                    plan, _ = control.load_study_plan(path)
                    cases = expand_study(plan, pipeline_root=root).cases
                    result = copy.deepcopy(next((c.config for c in cases if c.case_id == case), cases[0].config))
                else:
                    result, _ = control.load_yaml(path)
                message = 'Loaded YAML values.' if path else 'Function defaults. No YAML selected.'
            else:
                changes = _control_values(config, identities + number_ids, values + numbers, root)
                path_changed = trigger.get('path') if isinstance(trigger, dict) else None
                result = apply_control_values(config, {path_changed: changes[path_changed]} if path_changed in changes else changes,
                                              pipeline_root=root)
                message = 'Controls updated. Analyse reuses unchanged stages and recomputes affected outputs.'
            if result == config:
                return no_update, no_update, [no_update] * len(stages), [no_update] * len(selector_stages)
            old_specs = grouped_parameter_specs(config, pipeline_root=root)
            new_specs = grouped_parameter_specs(result, pipeline_root=root)
            # Only selector/schema changes replace widgets. Ordinary dragging
            # leaves focus and the slider itself intact.
            signature = lambda specs: [(s['path'], s['kind'], s['choices']) for s in specs]
            previous = {s['path']: s['value'] for s in old_specs}
            changed = {s['path'] for s in new_specs if previous.get(s['path']) != s['value']}
            trigger_path = trigger.get('path') if isinstance(trigger, dict) else None
            reload = (trigger in ('training-yaml', 'yaml-case') or signature(old_specs) != signature(new_specs)
                      or bool(changed - {trigger_path}))
            rendered = [controls_for(result, i['stage']) for i in stages] if reload else [no_update] * len(stages)
            rendered_selectors = ([controls_for(result, i['stage'], selectors=True) for i in selector_stages]
                                  if reload else [no_update] * len(selector_stages))
            return result, message, rendered, rendered_selectors
        except Exception as error:
            return no_update, f'{type(error).__name__}: {error}', [no_update] * len(stages), [no_update] * len(selector_stages)

    for namespace in ('param', 'tool-param', 'plan-param'):
        @app.callback(Output({'type': namespace+'-slide', 'path': MATCH}, 'value'), Output({'type': namespace+'-number', 'path': MATCH}, 'value'),
                      Output({'type': namespace+'-slide', 'path': MATCH}, 'min'), Output({'type': namespace+'-slide', 'path': MATCH}, 'max'),
                      Input({'type': namespace+'-slide', 'path': MATCH}, 'value'), Input({'type': namespace+'-number', 'path': MATCH}, 'value'),
                      State({'type': namespace+'-slide', 'path': MATCH}, 'min'), State({'type': namespace+'-slide', 'path': MATCH}, 'max'),
                      prevent_initial_call=True)
        def sync_numeric(slider, number, lower, upper):
            value = slider if ctx.triggered_id['type'].endswith('-slide') else number
            return (no_update, no_update, no_update, no_update) if value is None else (value, value, min(lower, value), max(upper, value))

    @app.callback(Output('input-files', 'data'), Input('add-file', 'n_clicks'), State('input-files', 'data'), prevent_initial_call=True)
    def add_file(_, rows):
        rows = list(rows or [])
        return rows+[dict(file_id=f'file_{len(rows)+1}', role='B', path='', label='')]

    @app.callback(Output('preview-record', 'options'), Output('preview-record', 'value'),
                  Input('record-ids', 'value'), Input('input-files', 'data'), State('preview-record', 'value'))
    def preview_records(ids, files, selected):
        choices = [r['file_id'] for r in files or [] if r.get('file_id') and r.get('path')] or list(ids or [])
        return _options(choices), selected if selected in choices else choices[0] if choices else None

    @app.callback(Output('record-ids', 'disabled'), Input('input-files', 'data'))
    def custom_inputs(files):
        return any(row.get('path') for row in files or [])

    @app.callback(Output('model-case', 'options'), Output('model-case', 'value'), Input('model-export', 'value'))
    def model_cases(export):
        cases = list(control.model_cases(export)) if export else []
        return _options(cases), cases[0] if cases else None

    @app.callback(Output('train-options', 'style'), Output('train', 'style'),
                  Output({'type': 'analyse-stage', 'stage': 'model'}, 'style'), Input('model-mode', 'value'))
    def training_mode(mode):
        return ({}, {}, {'display': 'none'}) if mode == 'Train' else ({'display': 'none'}, {'display': 'none'}, {})

    selection_names = ('participant-id', 'calibration-path', 'motion-bundle', 'motion-bundle-path', 'sqi-artifact',
                       'sqi-artifact-path', 'model-export', 'model-case', 'model-bundle', 'model-bundle-path')

    def selections(files, ids, args):
        participant, calibration, motion, motion_path, sqi, sqi_path, export, case, bundle, bundle_path = args
        supplied = [r for r in files or [] if r.get('path')]
        return dict(files=supplied, record_ids=[] if supplied else ids or [], participant_id=participant,
                    calibration_path=calibration or None, motion_bundle=motion_path or motion, sqi_artifact=sqi_path or sqi,
                    model_export=export, model_case=case, model_bundle=bundle_path or bundle)

    @app.callback(Output('preview-store', 'data'), Output('last-stage', 'data'),
                  Input({'type': 'analyse-stage', 'stage': ALL}, 'n_clicks'), State('config-state', 'data'),
                  State({'type': 'param', 'path': ALL}, 'id'), State({'type': 'param', 'path': ALL}, 'value'),
                  State({'type': 'param-number', 'path': ALL}, 'id'), State({'type': 'param-number', 'path': ALL}, 'value'),
                  State('preview-record', 'value'), State('session-id', 'data'), State('preview-start', 'value'),
                  State('preview-duration', 'value'), State('input-files', 'data'), State('record-ids', 'value'),
                  *[State(n, 'value') for n in selection_names], prevent_initial_call=True)
    def analyse(clicks, config, identities, values, number_ids, numbers, record, session, start, duration, files, ids, *args):
        if not any(clicks or []):
            return no_update, no_update
        stage = ctx.triggered_id['stage']
        try:
            payload = current(config, identities + number_ids, values + numbers)
            result = workflow.analyse(stage=stage, config_payload=payload, record_id=record or '', session_id=session,
                selections=selections(files, ids, args), start_s=float(0 if start is None else start),
                duration_s=float(20 if duration is None else duration))
            result['configuration'] = payload
            result['input_selection'] = selections(files, ids, args)
            result['display_interval'] = [start, duration]
            return result, stage
        except Exception as error:
            try:
                cached = getattr(workflow, 'cached_previews', lambda *a, **k: {})(session, record or '',
                    start_s=float(0 if start is None else start), duration_s=float(20 if duration is None else duration))
            except ValueError:
                cached = {}
            return dict(requested_stage=stage, error=f'{type(error).__name__}: {error}', previews=cached), stage

    @app.callback(Output({'type': 'stage-output', 'stage': ALL}, 'children'), Input('preview-store', 'data'),
                  Input('config-state', 'data'), State({'type': 'stage-output', 'stage': ALL}, 'id'),
                  Input('preview-record', 'value'), Input('input-files', 'data'), Input('record-ids', 'value'),
                  Input('preview-start', 'value'), Input('preview-duration', 'value'),
                  *[Input(n, 'value') for n in selection_names])
    def show_previews(result, config, identities, record, files, ids, start, duration, *args):
        output = []
        for identity in identities:
            stage = identity['stage']
            preview = (result or {}).get('previews', {}).get(stage)
            children = _render_preview(preview)
            if (result or {}).get('error') and stage == result.get('requested_stage'):
                children = [html.Pre(result['error'], className='status')]
            elif preview:
                state = 'Reused' if stage in result.get('reused_stages', []) else 'Computed'
                if result.get('configuration') != config:
                    state = 'Settings changed · Analyse checks dependencies before reusing this output.'
                elif (result.get('record_id') != record or result.get('input_selection') != selections(files, ids, args)
                      or result.get('display_interval') != [start, duration]):
                    state = 'Input / preview selection changed · press Analyse to refresh this output.'
                children.insert(0, html.Div(state, className='notice'))
            output.append(children)
        return output

    execution_names = ('run-name', 'repeat-indices', 'fold-indices', 'job-count',
                       'cache-mode', 'cache-root', 'train-flags', 'resume-path')

    def tool_dependencies(value_dependency):
        return [value_dependency('tool-operation', 'value'), State('tool-arguments', 'value'),
                State({'type': 'tool-param', 'path': ALL}, 'id'), value_dependency({'type': 'tool-param', 'path': ALL}, 'value'),
                State({'type': 'tool-param-number', 'path': ALL}, 'id'), value_dependency({'type': 'tool-param-number', 'path': ALL}, 'value'),
                value_dependency('tool-plan-state', 'data'),
                State({'type': 'plan-param', 'path': ALL}, 'id'), value_dependency({'type': 'plan-param', 'path': ALL}, 'value'),
                State({'type': 'plan-param-number', 'path': ALL}, 'id'), value_dependency({'type': 'plan-param-number', 'path': ALL}, 'value')]

    def execution(args):
        name, repeats, folds, count, cache_mode, cache_root, flags, resume = args
        return dict(run_name=name, repeats=repeats, folds=folds, jobs=count, cache_mode=cache_mode, cache_root=cache_root,
                    refit='refit' in (flags or []), dry_run='dry_run' in (flags or []), resume=resume)

    @app.callback(Output('train-request', 'data'), Output('train-command', 'children'), Output('yaml-view', 'children'),
                  Input('config-state', 'data'), Input('training-yaml', 'value'), *[Input(n, 'value') for n in execution_names])
    def build_training(config, path, *args):
        text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        try:
            request = _training_request(control, config, execution(args), path)
            return request.to_dict(), request.display, text
        except Exception as error:
            return None, f'{type(error).__name__}: {error}', text

    @app.callback(Output('command-view', 'children'), Input('config-state', 'data'), Input('last-stage', 'data'),
                  Input('preview-record', 'value'), Input('preview-start', 'value'), Input('preview-duration', 'value'),
                  Input('input-files', 'data'), Input('record-ids', 'value'), *[Input(n, 'value') for n in selection_names])
    def stage_command(config, stage, record, start, duration, files, ids, *args):
        argv = ['python', 'stage_analyse.py', '--config', 'workflow_config.yaml', '--stage', stage,
                '--record-id', record or '', '--start-s', str(0 if start is None else start),
                '--duration-s', str(20 if duration is None else duration),
                '--selections', json.dumps(selections(files, ids, args), ensure_ascii=False)]
        return shlex.join(argv)

    @app.callback(Output('active-train-job', 'data'), Output('train-status', 'children', allow_duplicate=True),
                  Input('train', 'n_clicks'), Input('stop-train', 'n_clicks'), State('model-mode', 'value'),
                  State('training-yaml', 'value'), State('train-target', 'value'), State('train-request', 'data'),
                  State('comparison-store', 'data'), State('active-train-job', 'data'),
                  State('tool-request', 'data'),
                  State('config-state', 'data'), State({'type': 'param', 'path': ALL}, 'id'),
                  State({'type': 'param', 'path': ALL}, 'value'), State({'type': 'param-number', 'path': ALL}, 'id'),
                  State({'type': 'param-number', 'path': ALL}, 'value'),
                  *[State(n, 'value') for n in execution_names], *tool_dependencies(State), prevent_initial_call=True)
    def training(_, __, mode, path, target, request, queue, job, tool, config, identities, values, number_ids, numbers, *args):
        try:
            if ctx.triggered_id == 'stop-train':
                if job:
                    jobs.terminate(job)
                return job, 'Stopped.' if job else 'No active training job.'
            tool_args = args[len(execution_names):]
            selected_tool = tool_snapshot(*tool_args) if target == 'tool' and tool_args else None
            tool_plan = selected_tool
            if target == 'tool' and not tool_args and tool:
                tool_plan = CommandRequest(**{**tool, 'arguments': tuple(tool['arguments'])})
            if mode != 'Train' or not (path or tool_plan and '--plan' in tool_plan.arguments):
                raise ValueError('Select a YAML and Train mode before Run.')
            if job and jobs.status(job)['state'] == 'running':
                return job, 'Training is already running. Stop it before starting another job.'
            if target == 'comparison':
                selected, _ = control.build_comparison_execution_request(queue or [])
            elif target == 'tool':
                if not tool_plan:
                    raise ValueError('Configure the advanced tool request first.')
                selected = tool_snapshot(*tool_args, materialize=True) if tool_args else tool_plan
            elif target == 'sweep':
                opts = execution(args[:len(execution_names)])
                selected = control.build_train_request(config_path=None, plan_path=path, operation='sweep',
                    run_name=opts['run_name'] or None, refit=opts['refit'], dry_run=opts['dry_run'], resume=opts['resume'] or None)
            else:
                payload = current(config, identities + number_ids, values + numbers)
                selected = _training_request(control, payload, execution(args[:len(execution_names)]), path)
            new_job = jobs.start_request(selected, kind='pipeline')
            return new_job, 'Training started. Stop remains available.\n'+selected.display
        except Exception as error:
            return job, f'{type(error).__name__}: {error}'

    @app.callback(Output('train-status', 'children'), Output('analysis-status', 'children'),
                  Input('job-poll', 'n_intervals'), State('active-train-job', 'data'), State('active-report-job', 'data'))
    def poll_jobs(_, train, report):
        def status(job):
            if not job:
                return no_update
            data = jobs.status(job)
            return f"{data['state']} · {data['elapsed_s']:.1f}s\n"+'\n'.join(data.get('log_tail', [])[-12:])
        return status(train), status(report)

    @app.callback(Output('comparison-store', 'data'), Output('comparison-status', 'children'),
                  Input('add-comparison', 'n_clicks'), Input('remove-comparison', 'n_clicks'), Input('clear-comparison', 'n_clicks'),
                  State('comparison-name', 'value'), State('train-request', 'data'), State('comparison-store', 'data'),
                  State('training-yaml', 'value'), State('config-state', 'data'), State({'type': 'param', 'path': ALL}, 'id'),
                  State({'type': 'param', 'path': ALL}, 'value'), State({'type': 'param-number', 'path': ALL}, 'id'),
                  State({'type': 'param-number', 'path': ALL}, 'value'),
                  *[State(n, 'value') for n in execution_names], prevent_initial_call=True)
    def comparison(_, __, ___, name, request, queue, path, config, identities, values, number_ids, numbers, *args):
        queue = list(queue or [])
        try:
            if ctx.triggered_id == 'clear-comparison':
                return [], 'Queue cleared.'
            if ctx.triggered_id == 'remove-comparison':
                return queue[:-1], 'Last unit removed.'
            payload = current(config, identities + number_ids, values + numbers)
            request = _training_request(control, payload, execution(args), path)
            candidate = dict(request.to_dict(), name=name)
            candidate['arguments'] = list(candidate['arguments'])
            comparison_sequence_export_yaml(queue+[candidate], pipeline_root=root)
            return queue+[candidate], f'Added {name}; {len(queue)+1} units.'
        except Exception as error:
            return queue, f'{type(error).__name__}: {error}'

    @app.callback(Output('comparison-table', 'children'), Output('comparison-cli-view', 'children'),
                  Output('comparison-yaml-view', 'children'), Input('comparison-store', 'data'))
    def show_comparison(queue):
        if not queue:
            return html.Div('No comparison units yet.', className='notice'), '', ''
        return (_table([{'order': n, 'name': r['name'], 'config_id': yaml.safe_load(r['resolved_yaml'])['config_id']}
                        for n, r in enumerate(queue, 1)]), comparison_sequence_cli(queue),
                comparison_sequence_export_yaml(queue, pipeline_root=root))

    report_fields = ('analysis-runs', 'analysis-mode', 'analysis-preset', 'analysis-modules', 'analysis-figures', 'analysis-tables',
                     'reference-case', 'factor-paths', 'include-cases', 'exclude-cases', 'report-name',
                     'statistics-alpha', 'calibration-bins')

    @app.callback(Output('analysis-request', 'data'), Output('analysis-command', 'children'),
                  *[Input(n, 'value') for n in report_fields], Input('config-state', 'data'))
    def build_report(runs, mode, preset, modules, figures, tables, reference, factors, include, exclude,
                     name, alpha, bins, config):
        split = lambda text: [x.strip() for x in (text or '').split(',') if x.strip()]
        if not runs:
            return None, 'Select pipeline output(s).'
        try:
            statistics = config['evaluation']['statistics']
            request = control.build_analysis_request(run_paths=runs, mode=mode, preset=preset or '', modules=modules or [],
                figures=figures or None, tables=tables or None, reference_case=reference or None, factor_paths=split(factors),
                include_cases=split(include), exclude_cases=split(exclude), output_name=name or None,
                bootstrap_resamples=statistics['bootstrap_replicates'],
                permutation_resamples=statistics['paired_permutation_replicates'], statistics_seed=statistics['seed'],
                alpha=alpha, calibration_bins=bins)
            return request.to_dict(), request.display
        except Exception as error:
            return None, f'{type(error).__name__}: {error}'

    @app.callback(Output('active-report-job', 'data'), Output('analysis-status', 'children', allow_duplicate=True),
                  Input('analyse-report', 'n_clicks'), Input('stop-report', 'n_clicks'),
                  State('analysis-request', 'data'), State('active-report-job', 'data'),
                  State('config-state', 'data'), State({'type': 'param', 'path': ALL}, 'id'),
                  State({'type': 'param', 'path': ALL}, 'value'), State({'type': 'param-number', 'path': ALL}, 'id'),
                  State({'type': 'param-number', 'path': ALL}, 'value'),
                  *[State(n, 'value') for n in report_fields], prevent_initial_call=True)
    def report(_, __, request, job, config, identities, values, number_ids, numbers, *args):
        try:
            if ctx.triggered_id == 'stop-report':
                if job:
                    jobs.terminate(job)
                return job, 'Report stopped.'
            payload = current(config, identities + number_ids, values + numbers)
            request, message = build_report(*args, payload)
            if not request:
                raise ValueError(message)
            if job and jobs.status(job)['state'] == 'running':
                return job, 'Report is already running.'
            selected = CommandRequest(**{**request, 'arguments': tuple(request['arguments'])})
            return jobs.start_request(selected, kind='report'), 'Report analysis started.'
        except Exception as error:
            return job, f'{type(error).__name__}: {error}'

    @app.callback(Output('report-figure', 'options'), Output('report-figure', 'value'),
                  Output('report-table', 'options'), Output('report-table', 'value'), Input('report-output', 'value'))
    def report_artifacts(path):
        figures = list(browser.study_figure_paths(path)) if path else []
        tables = list(browser.study_table_paths(path)) if path else []
        if path:
            directory = browser._study_dir(path)
            figures += [str(item.relative_to(directory)) for item in sorted(directory.rglob('*.html'))]
        return _options(figures), figures[0] if figures else None, _options(tables), tables[0] if tables else None

    @app.callback(Output('report-image', 'src'), Output('report-document', 'children'),
                  Input('report-figure', 'value'), State('report-output', 'value'))
    def figure(path, root_path):
        if not path or not root_path:
            return None, None
        if str(path).endswith('.html'):
            base = browser._study_dir(root_path)
            document = (base / path).resolve()
            document.relative_to(base)
            return None, html.Iframe(srcDoc=document.read_text(encoding='utf-8'), sandbox='allow-scripts',
                                     style={'width': '100%', 'height': '650px', 'border': 'none'})
        return browser.study_figure_data_uri(root_path, path), None

    @app.callback(Output('report-output', 'options', allow_duplicate=True), Output('report-output', 'value'),
                  Output('report-preview-job', 'data'),
                  Input('job-poll', 'n_intervals'), State('active-report-job', 'data'), State('report-output', 'value'),
                  State('report-preview-job', 'data'),
                  prevent_initial_call=True)
    def finished_report(_, job, selected, displayed=None):
        if not job or job == displayed:
            return no_update, no_update, no_update
        status = jobs.status(job)
        if status['state'] != 'passed':
            return no_update, no_update, no_update
        paths = list(control.report_outputs())
        command = status.get('command', [])
        target = None
        if command:
            option = lambda name: DashboardJobManager._argument_value(command, name)
            name = option('--output-name')
            if not name:
                source = option('--input') or option('--run').split('=', 1)[1]
                path = Path(source)
                relative = (path if path.is_absolute() else root / path).resolve().relative_to(root / 'pipeline_output')
                name = relative.parts[0]
            target = f'report_output/{name}'
        return (_options(paths), target if target in paths else selected if selected in paths else paths[0] if paths else None,
                job)

    @app.callback(Output('report-table-preview', 'children'), Input('report-table', 'value'), State('report-output', 'value'))
    def report_table(path, root_path):
        return _table(browser.study_table(root_path, path)[0]) if path and root_path else None

    @app.callback(Output('tool-controls', 'children'), Output('tool-plan-picker', 'style'),
                  Input('tool-operation', 'value'))
    def tool_parameters(operation):
        specs = tool_specs(operation)
        return (parameter_panel([spec for spec in specs if spec['path'] != 'plan'], 'tool-param'),
                {} if any(spec['path'] == 'plan' for spec in specs) else {'display': 'none'})

    @app.callback(Output('tool-plan-state', 'data'), Output('tool-plan-controls', 'children'),
                  Output('tool-plan-status', 'children'), Input('tool-plan-yaml', 'value'), Input('tool-plan-path', 'value'))
    def load_tool_plan(selected, entered):
        path = entered or selected
        if not path:
            return None, [], 'Select a plan YAML to show its current parameters.'
        try:
            source = Path(path).expanduser()
            source = source if source.is_absolute() else root / source
            plan = yaml.safe_load(source.read_text(encoding='utf-8'))
            if not isinstance(plan, Mapping):
                raise TypeError('A plan YAML must contain a mapping.')
            return ({'source': str(source), 'plan': plan}, parameter_panel(plan_parameter_specs(plan), 'plan-param'),
                    'Loaded plan values. Edits affect the snapshot, not the source file.')
        except Exception as error:
            return None, [], f'{type(error).__name__}: {error}'

    @app.callback(Output('tool-request', 'data'), Output('tool-command', 'children'), Output('tool-yaml', 'children'),
                  Output('tool-help', 'children'), Output('analyse-tool', 'disabled'),
                  *tool_dependencies(Input))
    def build_tool(operation, arguments, *args):
        _, script, command, trains = TOOLS[operation]
        import argparse
        parser = _tool_parser(script)
        subcommands = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        help_text = subcommands.choices[command].format_help()
        try:
            request = tool_snapshot(operation, arguments, *args)
            return request.to_dict(), request.display, request.resolved_yaml, help_text, trains
        except Exception as error:
            return None, str(error), '', help_text, trains

    @app.callback(Output('tool-arguments', 'value'), Input('tool-request', 'data'))
    def tool_arguments(request):
        return shlex.join(request['arguments'][1:]) if request else ''

    @app.callback(Output('active-tool-job', 'data'), Output('tool-status', 'children', allow_duplicate=True),
                  Input('analyse-tool', 'n_clicks'), Input('stop-tool', 'n_clicks'), State('tool-operation', 'value'),
                  State('tool-request', 'data'), State('active-tool-job', 'data'),
                  *tool_dependencies(State)[1:], prevent_initial_call=True)
    def tool_action(_, __, operation, request, job, *args):
        try:
            if ctx.triggered_id == 'stop-tool':
                if job:
                    jobs.terminate(job)
                return job, 'Stopped.'
            if TOOLS[operation][3]:
                return job, 'Use the model Train mode to start this training operation.'
            if job and jobs.status(job)['state'] == 'running':
                return job, 'This tool is already running.'
            if not args and not request:
                raise ValueError('Set the required command controls shown in Command help.')
            selected = (tool_snapshot(operation, *args, materialize=True) if args else
                        CommandRequest(**{**request, 'arguments': tuple(request['arguments'])}))
            return jobs.start_request(selected, kind='tool'), 'Started.\n'+selected.display
        except Exception as error:
            return job, f'{type(error).__name__}: {error}'

    @app.callback(Output('tool-status', 'children'), Input('job-poll', 'n_intervals'), State('active-tool-job', 'data'))
    def poll_tool(_, job):
        if not job:
            return no_update
        data = jobs.status(job)
        return f"{data['state']} · {data['elapsed_s']:.1f}s\n"+'\n'.join(data.get('log_tail', [])[-16:])

    @app.callback(Output('download-tool-cli', 'data'), Input('save-tool-cli', 'n_clicks'), State('tool-request', 'data'),
                  *tool_dependencies(State), prevent_initial_call=True)
    def tool_cli(_, request, *args):
        if args:
            try:
                request = tool_snapshot(*args).to_dict()
            except (ValueError, TypeError, yaml.YAMLError):
                return no_update  # The command preview displays the parser error.
        return dcc.send_string(_tool_download_command(request), 'v5_tool.sh') if request else no_update

    @app.callback(Output('download-tool-yaml', 'data'), Input('save-tool-yaml', 'n_clicks'), State('tool-request', 'data'),
                  *tool_dependencies(State), prevent_initial_call=True)
    def tool_yaml(_, request, *args):
        if args:
            try:
                request = tool_snapshot(*args).to_dict()
            except (ValueError, TypeError, yaml.YAMLError):
                return no_update
        name = f"{request['config_sha256']}.yaml" if request and request.get('config_sha256') else 'v5_tool_request.yaml'
        return dcc.send_string(request['resolved_yaml'], name) if request else no_update

    @app.callback(Output('download-report-cli', 'data'), Input('save-report-cli', 'n_clicks'), State('analysis-request', 'data'), prevent_initial_call=True)
    def report_cli(_, request):
        return dcc.send_string(request['display']+'\n', 'v5_report.sh') if request else no_update

    @app.callback(Output('download-cli', 'data'), Input('save-cli', 'n_clicks'), State('command-view', 'children'), prevent_initial_call=True)
    def download_cli(_, command):
        return dcc.send_string(command+'\n', 'workflow_analyse.sh')

    @app.callback(Output('download-yaml', 'data'), Input('save-yaml', 'n_clicks'), State('config-state', 'data'), prevent_initial_call=True)
    def download_yaml(_, config):
        return dcc.send_string(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), 'workflow_config.yaml')

    @app.callback(Output('download-sequence-cli', 'data'), Input('export-sequence-cli', 'n_clicks'), State('comparison-store', 'data'), prevent_initial_call=True)
    def sequence_cli(_, queue):
        return dcc.send_string(comparison_sequence_cli(queue), 'comparison_sequence.sh') if queue else no_update

    @app.callback(Output('download-sequence-yaml', 'data'), Input('export-sequence-yaml', 'n_clicks'), State('comparison-store', 'data'), prevent_initial_call=True)
    def sequence_yaml(_, queue):
        return dcc.send_string(comparison_sequence_export_yaml(queue, pipeline_root=root), 'comparison_sequence.yaml') if queue else no_update

    return app
