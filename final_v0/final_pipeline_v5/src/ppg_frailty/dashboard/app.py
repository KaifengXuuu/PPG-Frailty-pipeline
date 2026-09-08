"""Dash presentation adapter for the complete V5 control surface."""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import yaml
from .control_service import (
    CommandRequest, INFERENCE_SOURCE_CONFIRMATION, INHERIT_MODULE, MISSING_B_TODO,
    SINGLE_PARTICIPANT_NOTICE, V5ControlService, WORKFLOW_STAGES,
    comparison_sequence_cli, comparison_sequence_export_yaml, flatten_parameters,
)
from .job_manager import DashboardJobManager
from .preview_service import PipelinePreviewService

COLORS = {
    'background': '#f3f6f8',
    'panel': '#ffffff',
    'ink': '#17212b',
    'muted': '#65717f',
    'accent': '#176b68',
    'accent_soft': '#e4f2f0',
    'border': '#d8e0e5',
    'warning': '#9a5b00',
    'danger': '#a33a32'
}
TOOL_OPTIONS: tuple[tuple[str, str], ...] = (
    ('Pipeline validate', 'pipeline_validate'), ('Show config', 'show_config'),
    ('Sweep validate', 'sweep_validate'), ('Rebuild index', 'pipeline_index'),
    ('Export model config', 'model_export'), ('Pipeline Excel', 'pipeline_excel'),
    ('Report Excel', 'report_excel'), ('Special validate', 'specialized_validate'),
    ('Special analyse', 'specialized_run'), ('Special report', 'specialized_report'),
    ('Special pipe check', 'specialized_pipeline_validate'),
    ('Special pipe run', 'specialized_pipeline_run'),
    ('Special CV complete', 'specialized_pipeline_complete'),
)
EQUIVALENT_SURFACE_NOTICE = (
    'Every displayed execution request uses a public parser-backed CLI, although '
    'not every CLI subcommand has a same-named button. Ablation/grid use the '
    'comparison queue and run-plan uses Sweep. Read-only catalogs use '
    'Configure/Analyse; Preview and Stop are local UI controls.'
)
_TOOL_SUBCOMMANDS = {
    'pipeline.py': frozenset({'validate', 'show-config', 'index'}),
    'sweep.py': frozenset({'validate', 'export-excel'}),
    'analyse_report.py': frozenset({'export-excel', 'specialized-validate', 'specialized-run', 'specialized-report'}),
    'specialized_pipeline.py': frozenset({'validate', 'run', 'complete'})
}


def _options(values: Sequence[str]) -> list[dict[str, str]]:
    return [{'label': value, 'value': value} for value in values]


def _panel(children: Any, *, style: Mapping[str, Any] | None = None) -> Any:
    from dash import html
    merged = {'background': COLORS['panel'], 'border': f"1px solid {COLORS['border']}", 'borderRadius': '10px', 'padding': '14px'}
    merged.update(dict(style or {}))
    return html.Div(children, style=merged)


def _button(label: str, component_id: str, *, danger: bool = False, disabled: bool = False) -> Any:
    from dash import html
    if len(label) > 20:
        raise ValueError('dashboard button labels must not exceed 20 characters')
    return html.Button(label,
                       id=component_id,
                       n_clicks=0,
                       disabled=disabled,
                       style={
                           'border': 0,
                           'borderRadius': '7px',
                           'padding': '9px 15px',
                           'fontWeight': 700,
                           'cursor': 'pointer',
                           'background': COLORS['danger'] if danger else COLORS['accent'],
                           'color': 'white'
                       })


def _error(error: BaseException) -> str:
    return f'{type(error).__name__}: {error}'


def _module_controls(catalog: Mapping[str, Sequence[Mapping[str, Any]]], defaults: Mapping[str, Any],
                     features: Sequence[str]) -> list[Any]:
    from dash import dcc, html
    controls = []
    for family, rows in catalog.items():
        values = [str(row['module_id']) for row in rows]
        is_feature = family == 'feature_group'
        value: Any = list(features) if is_feature else defaults.get(family, INHERIT_MODULE)
        choices = [{'label': value_id, 'value': value_id, 'title': str(row.get('notes', ''))} for value_id, row in zip(values, rows)]
        if not is_feature:
            choices.insert(
                0, {
                    'label': 'YAML value / inactive',
                    'value': INHERIT_MODULE,
                    'title': 'Keep the complete resolved YAML value; no module override.'
                })
        controls.append(
            html.Div([
                html.Label(family.replace('_', ' '),
                           title='Feature groups are composable; other families are mutually exclusive.',
                           style={
                               'fontSize': '12px',
                               'fontWeight': 700
                           }),
                dcc.Dropdown(id={
                    'type': 'module-select',
                    'family': family
                },
                             options=choices,
                             value=value,
                             multi=is_feature,
                             clearable=True,
                             placeholder='Select modules' if is_feature else 'Use YAML value')
            ],
                     style={'minWidth': '230px'}))
    return controls


def _as_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _lines(value: str | None) -> list[str]:
    """Parse repeatable CLI values without comma/path ambiguity."""
    return [line.strip() for line in str(value or '').splitlines() if line.strip()]


_CLOSED_NUMERIC_RANGE = re.compile(
    '^(?:finite )?(?:float|integer) in \\[\\s*(-?(?:\\d+(?:\\.\\d*)?|\\.\\d+))\\s*,\\s*(-?(?:\\d+(?:\\.\\d*)?|\\.\\d+))\\s*\\](?:;.*)?$'
)


def _slider_bounds(row: Mapping[str, Any], value: int | float) -> tuple[int | float, int | float] | None:
    """Read only closed finite bounds declared by the live parameter catalog."""
    if str(row.get('control', '')) != 'parameter':
        return None
    description = str(row.get('range', '')).strip()
    if description == 'integer 0 or 1':
        return (0, 1)
    match = _CLOSED_NUMERIC_RANGE.fullmatch(description)
    if match is None:
        return None
    lower, upper = (float(match.group(index)) for index in (1, 2))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        return None
    if float(value) < lower or float(value) > upper:
        return None
    if isinstance(value, int):
        if not lower.is_integer() or not upper.is_integer():
            return None
        if upper - lower > 10000:
            return None
        return (int(lower), int(upper))
    return (lower, upper)


def _numeric_sliders(rows: Sequence[Mapping[str, Any]]) -> list[Any]:
    """Build catalog-bound quick picks; never infer a validation range."""
    from dash import dcc, html
    controls = []
    for row in rows:
        try:
            value = yaml.safe_load(str(row['value_yaml']))
        except yaml.YAMLError:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if not np.isfinite(numeric):
            continue
        bounds = _slider_bounds(row, value)
        if bounds is None:
            continue
        minimum, maximum = bounds
        marks = {choice: f'{choice:g}' if isinstance(choice, float) else str(choice) for choice in sorted({minimum, value, maximum})}
        controls.append(
            html.Div([
                html.Label(
                    f"{row['path']} · {row['range']}",
                    title=
                    'Quick picks use only closed bounds from the live pipeline.py parameters catalog. Edit the table for other valid values.',
                    style={
                        'fontSize': '11px',
                        'fontFamily': 'monospace'
                    }),
                dcc.Slider(id={
                    'type': 'parameter-slider',
                    'path': str(row['path'])
                },
                           min=minimum,
                           max=maximum,
                           step=None,
                           marks=marks,
                           value=value,
                           tooltip={
                               'placement': 'bottom',
                               'always_visible': False
                           })
            ],
                     style={'padding': '4px 6px'}))
    return controls


def create_app(pipeline_root: str | Path | None = None,
               *,
               control_service: V5ControlService | None = None,
               preview_service: PipelinePreviewService | None = None,
               job_manager: DashboardJobManager | None = None) -> Any:
    """Create, but do not start, the local V5 control panel."""
    from dash import ALL, Dash, Input, Output, State, callback_context, dash_table, dcc, html, no_update
    import plotly.graph_objects as go
    control = control_service or V5ControlService(pipeline_root)
    preview = preview_service or PipelinePreviewService(control.pipeline_root)
    jobs = job_manager or DashboardJobManager(control.pipeline_root)
    try:
        records = preview.records()
    except Exception:
        records = ()
    participants = sorted({row.participant_id for row in records})
    roles = sorted({row.role for row in records})
    from ..v5_reporting.registry import KNOWN_FIGURES, KNOWN_TABLES, MODULES as REPORT_MODULES, PRESETS as REPORT_PRESETS
    yaml_paths = list(control.yaml_paths())
    study_plan_paths = list(control.study_plan_paths())
    sweep_capabilities = control.sweep_capabilities()
    model_exports = list(control.model_exports())
    study_outputs = list(control.study_outputs())
    report_outputs = list(control.report_outputs())
    report_module_ids = sorted((module.name for module in REPORT_MODULES))
    try:
        parameter_contract = control.parameter_contract()
        parameter_contract_notice = 'Only closed finite ranges from the live parameter catalog appear here. The YAML table and canonical resolver remain authoritative.'
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as error:
        parameter_contract = {}
        parameter_contract_notice = f'Live parameter guidance is unavailable; edit the YAML table and rely on the canonical resolver. {_error(error)}'
    app = Dash(__name__, title='PPG Frailty V5', suppress_callback_exceptions=True)
    app.layout = html.Div(
        [
            dcc.Store(id='config-state'),
            dcc.Store(id='train-request'),
            dcc.Store(id='inference-request'),
            dcc.Store(id='analysis-request'),
            dcc.Store(id='active-train-job'),
            dcc.Store(id='active-report-job'),
            dcc.Store(id='tool-request'),
            dcc.Store(id='active-tool-job'),
            dcc.Store(id='preview-store'),
            dcc.Store(id='comparison-store', data=[]),
            dcc.Store(id='comparison-execution-request'),
            dcc.Download(id='download-cli'),
            dcc.Download(id='download-yaml'),
            dcc.Download(id='download-sequence-cli'),
            dcc.Download(id='download-sequence-yaml'),
            dcc.Download(id='download-comparison-run-cli'),
            dcc.Download(id='download-comparison-run-yaml'),
            dcc.Download(id='download-inference-cli'),
            dcc.Download(id='download-inference-yaml'),
            dcc.Download(id='download-analysis-cli'),
            dcc.Download(id='download-analysis-yaml'),
            dcc.Download(id='download-tool-cli'),
            dcc.Download(id='download-tool-yaml'),
            dcc.Interval(id='job-poll', interval=1500, n_intervals=0),
            html.Div(
                [
                    html.Div([
                        html.H1('PPG Frailty V5', style={'margin': 0}),
                        html.Div('One validated control plane for CLI and Dash', style={'color': COLORS['muted']})
                    ]),
                    html.Div('LOCAL · HASH-BOUND · FAIL-CLOSED', style={
                        'fontWeight': 700,
                        'color': COLORS['accent']
                    })
                ],
                style={
                    'display': 'flex',
                    'flexWrap': 'wrap',
                    'gap': '10px',
                    'justifyContent': 'space-between',
                    'alignItems': 'center',
                    'marginBottom': '14px'
                }),
            html.Div(
                [html.Div(SINGLE_PARTICIPANT_NOTICE),
                 html.Div(MISSING_B_TODO, style={'marginTop': '5px'})],
                style={
                    'background': '#fff6e6',
                    'border': '1px solid #eed29c',
                    'color': COLORS['warning'],
                    'padding': '10px 13px',
                    'borderRadius': '8px',
                    'fontSize': '13px',
                    'marginBottom': '14px'
                }),
            dcc.Tabs(
                value='configure',
                children=[
                    dcc.Tab(label='Configure',
                            value='configure',
                            children=html.Div([
                                _panel([
                                    html.H3('Configuration source', style={'marginTop': 0}),
                                    dcc.RadioItems(id='config-source',
                                                   options=[{
                                                       'label': 'Selected YAML',
                                                       'value': 'yaml'
                                                   }, {
                                                       'label': 'model_config defaults',
                                                       'value': 'model_config'
                                                   }],
                                                   value='yaml',
                                                   inline=True),
                                    html.Div(
                                        [
                                            html.Div([
                                                html.Label('Training YAML'),
                                                dcc.Dropdown(id='training-yaml',
                                                             options=_options(yaml_paths),
                                                             value=None,
                                                             placeholder='Required before Train')
                                            ]),
                                            html.Div([
                                                html.Label('model_config export'),
                                                dcc.Dropdown(id='model-export',
                                                             options=_options(model_exports),
                                                             value=model_exports[0] if model_exports else None,
                                                             placeholder='No export available')
                                            ]),
                                            html.Div([
                                                html.Label('Export case'),
                                                dcc.Dropdown(id='model-case', placeholder='Select case')
                                            ])
                                        ],
                                        style={
                                            'display': 'grid',
                                            'gridTemplateColumns': 'repeat(auto-fit,minmax(min(100%,230px),1fr))',
                                            'gap': '12px',
                                            'marginTop': '12px'
                                        }),
                                    html.Div([
                                        _button('Load', 'load-config'),
                                        _button('Refresh', 'refresh-configs'),
                                        html.Span(id='config-status')
                                    ],
                                             style={
                                                 'display': 'flex',
                                                 'gap': '12px',
                                                 'alignItems': 'center',
                                                 'marginTop': '12px'
                                             })
                                ]),
                                _panel([
                                    html.H3('Modules', style={'marginTop': 0}),
                                    html.Div('Each family is a mutually exclusive dropdown; feature_group is composable.',
                                             style={
                                                 'color': COLORS['muted'],
                                                 'fontSize': '13px'
                                             }),
                                    html.Div(id='module-controls',
                                             style={
                                                 'display': 'grid',
                                                 'gridTemplateColumns': 'repeat(auto-fit,minmax(min(100%,230px),1fr))',
                                                 'gap': '10px',
                                                 'marginTop': '12px'
                                             })
                                ],
                                       style={'marginTop': '12px'}),
                                _panel([
                                    html.H3('All parameters', style={'marginTop': 0}),
                                    html.Details([
                                        html.Summary('Contract-bound numeric shortcuts'),
                                        html.Div(parameter_contract_notice,
                                                 style={
                                                     'color': COLORS['muted'],
                                                     'fontSize': '12px',
                                                     'marginTop': '6px'
                                                 }),
                                        html.Div(id='parameter-sliders',
                                                 style={
                                                     'display': 'grid',
                                                     'gridTemplateColumns': 'repeat(auto-fit,minmax(min(100%,260px),1fr))',
                                                     'gap': '5px',
                                                     'margin': '10px 0'
                                                 })
                                    ]),
                                    dash_table.DataTable(id='parameter-table',
                                                         columns=[{
                                                             'name': 'path',
                                                             'id': 'path',
                                                             'editable': False
                                                         }, {
                                                             'name': 'value (YAML)',
                                                             'id': 'value_yaml',
                                                             'editable': True
                                                         }, {
                                                             'name': 'type',
                                                             'id': 'type',
                                                             'editable': False
                                                         }, {
                                                             'name': 'control',
                                                             'id': 'control',
                                                             'editable': False
                                                         }, {
                                                             'name': 'accepted range',
                                                             'id': 'range',
                                                             'editable': False
                                                         }, {
                                                             'name': 'CLI input',
                                                             'id': 'input',
                                                             'editable': False
                                                         }, {
                                                             'name': 'original',
                                                             'id': 'original_yaml'
                                                         }],
                                                         hidden_columns=['original_yaml'],
                                                         data=[],
                                                         editable=True,
                                                         filter_action='native',
                                                         sort_action='native',
                                                         page_size=18,
                                                         style_table={'overflowX': 'auto'},
                                                         style_cell={
                                                             'fontFamily': 'monospace',
                                                             'fontSize': 11,
                                                             'padding': '6px'
                                                         },
                                                         style_header={
                                                             'fontWeight': 700,
                                                             'background': '#edf3f4'
                                                         })
                                ],
                                       style={'marginTop': '12px'})
                            ],
                                              style={'paddingTop': '14px'})),
                    dcc.Tab(label='Workflow',
                            value='workflow',
                            children=html.Div([
                                _panel([
                                    html.H3('Executable workflow', style={'marginTop': 0}),
                                    dash_table.DataTable(data=list(WORKFLOW_STAGES),
                                                         columns=[{
                                                             'name': key,
                                                             'id': key
                                                         } for key in ('stage', 'families', 'preview')],
                                                         style_cell={
                                                             'textAlign': 'left',
                                                             'whiteSpace': 'normal',
                                                             'height': 'auto',
                                                             'padding': '8px'
                                                         },
                                                         style_header={
                                                             'fontWeight': 700,
                                                             'background': '#edf3f4'
                                                         })
                                ]),
                                html.Div(
                                    [
                                        _panel([
                                            html.H3('Stage preview', style={'marginTop': 0}),
                                            html.Label('Participant'),
                                            dcc.Dropdown(id='preview-participant',
                                                         options=_options(participants),
                                                         value=participants[0] if participants else None),
                                            html.Label('Role', style={
                                                'display': 'block',
                                                'marginTop': '9px'
                                            }),
                                            dcc.Dropdown(
                                                id='preview-role', options=_options(roles), value=None, placeholder='All roles'),
                                            html.Label('Recording', style={
                                                'display': 'block',
                                                'marginTop': '9px'
                                            }),
                                            dcc.Dropdown(id='preview-record'),
                                            html.Div([
                                                dcc.Input(
                                                    id='preview-start', type='number', value=0, min=0, step=1, placeholder='Start s'),
                                                dcc.Input(id='preview-duration',
                                                          type='number',
                                                          value=20,
                                                          min=1,
                                                          max=120,
                                                          step=1,
                                                          placeholder='Duration s')
                                            ],
                                                     style={
                                                         'display': 'flex',
                                                         'flexWrap': 'wrap',
                                                         'gap': '8px',
                                                         'marginTop': '9px'
                                                     }),
                                            dcc.Checklist(id='preview-traces',
                                                          options=_options(list(preview.DEFAULT_TRACES)),
                                                          value=list(preview.DEFAULT_TRACES),
                                                          style={'marginTop': '9px'}),
                                            dcc.Dropdown(id='preview-stages',
                                                         options=_options([
                                                             'source', 'preprocess', 'quality_route', 'pulse_ppi', 'dual_optical',
                                                             'morphology', 'engineering', 'representation_model', 'aggregation'
                                                         ]),
                                                         value=[
                                                             'source', 'preprocess', 'quality_route', 'pulse_ppi', 'dual_optical',
                                                             'morphology', 'engineering', 'representation_model', 'aggregation'
                                                         ],
                                                         multi=True),
                                            html.Label('Completed training artifact', style={
                                                'display': 'block',
                                                'marginTop': '9px'
                                            }),
                                            dcc.Dropdown(id='preview-artifact-run',
                                                         options=_options(study_outputs),
                                                         value=study_outputs[0] if study_outputs else None,
                                                         placeholder='N/A until a completed pipeline run is selected'),
                                            html.Div([_button('Preview', 'preview')], style={'marginTop': '10px'}),
                                            html.Div(id='preview-status', style={'marginTop': '8px'})
                                        ]),
                                        _panel([
                                            dcc.Graph(id='time-graph', figure=go.Figure(), config={'displaylogo': False}),
                                            dcc.Graph(id='spectrum-graph', figure=go.Figure(), config={'displaylogo': False}),
                                            dash_table.DataTable(id='stage-table',
                                                                 page_size=20,
                                                                 filter_action='native',
                                                                 style_cell={
                                                                     'fontFamily': 'monospace',
                                                                     'fontSize': 11,
                                                                     'whiteSpace': 'normal',
                                                                     'height': 'auto'
                                                                 })
                                        ])
                                    ],
                                    style={
                                        'display': 'grid',
                                        'gridTemplateColumns': 'repeat(auto-fit,minmax(min(100%,320px),1fr))',
                                        'gap': '12px',
                                        'marginTop': '12px'
                                    })
                            ],
                                              style={'paddingTop': '14px'})),
                    dcc.Tab(
                        label='Run',
                        value='run',
                        children=html.Div([
                            _panel([
                                html.H3('Execution', style={'marginTop': 0}),
                                html.Div([
                                    dcc.Dropdown(
                                        id='train-operation',
                                        options=_options(['run', 'sweep']), value='run', clearable=False),
                                    dcc.Dropdown(
                                        id='sweep-plan', options=_options(study_plan_paths), value=None,
                                        placeholder='Study-plan YAML'),
                                    dcc.Input(id='run-name', value='', placeholder='Optional run name'),
                                    dcc.Input(id='study-id', value='v5_dashboard', placeholder='Study ID'),
                                    dcc.Input(id='study-purpose', value='V5 dashboard run', placeholder='Purpose'),
                                    dcc.Input(id='config-id', value='', placeholder='Optional config ID'),
                                    dcc.Dropdown(id='repeats',
                                                 options=_options(['0', '1', '2', '3', '4']),
                                                 value=['0', '1', '2', '3', '4'],
                                                 multi=True,
                                                 placeholder='Repeat subset 0..4'),
                                    dcc.Dropdown(id='folds',
                                                 options=_options(['0', '1', '2', '3', '4']),
                                                 value=['0', '1', '2', '3', '4'],
                                                 multi=True,
                                                 placeholder='Fold subset 0..4'),
                                    dcc.Input(id='jobs', type='number', value=1, min=1, step=1),
                                    dcc.Dropdown(id='device', options=_options(['cuda', 'cpu']), value='cuda', clearable=False),
                                    dcc.Dropdown(id='cache-mode',
                                                 options=_options(['read_write', 'read_only', 'off']),
                                                 value='read_write',
                                                 clearable=False)
                                ],
                                         style={
                                             'display': 'grid',
                                             'gridTemplateColumns': 'repeat(auto-fit,minmax(min(100%,140px),1fr))',
                                             'gap': '8px'
                                         }),
                                html.Div(
                                    'Run uses Configure controls. Sweep executes the selected validated study plan exactly; its cases, resources, and cache settings remain authoritative.',
                                    style={
                                        'color': COLORS['muted'],
                                        'fontSize': '12px',
                                        'marginTop': '8px'
                                    }),
                                html.Details([
                                    html.Summary('Advanced execution'),
                                    html.Div(
                                        [
                                            dcc.Input(id='cache-root', value='', placeholder='Cache root'),
                                            dcc.Input(id='cache-namespaces', value='', placeholder='cache namespaces'),
                                            dcc.Textarea(
                                                id='unset-paths', value='', placeholder='Optional --unset dotted paths, one per line'),
                                            dcc.Input(id='resume-directory', value='', placeholder='Resume path'),
                                            dcc.Input(id='environment-lock',
                                                      value='requirements/environment-finalcase-lock.yaml',
                                                      placeholder='Environment lock'),
                                            dcc.Dropdown(id='environment-policy',
                                                         options=_options(['exact', 'record']),
                                                         value='exact',
                                                         clearable=False),
                                            dcc.Checklist(id='execution-flags',
                                                          options=[{
                                                              'label': 'Continue on error',
                                                              'value': 'continue'
                                                          }, {
                                                              'label': 'Measure costs',
                                                              'value': 'costs'
                                                          }, {
                                                              'label': 'Hash predictions',
                                                              'value': 'hash'
                                                          }, {
                                                              'label': 'Dry run',
                                                              'value': 'dry'
                                                          }],
                                                          value=[],
                                                          inline=True),
                                            dcc.Checklist(
                                                id='refit-enabled', options=[{
                                                    'label': 'Final refit',
                                                    'value': 'refit'
                                                }], value=[])
                                        ],
                                        style={
                                            'display': 'grid',
                                            'gridTemplateColumns': 'repeat(auto-fit,minmax(min(100%,180px),1fr))',
                                            'gap': '8px',
                                            'marginTop': '8px'
                                        }),
                                    html.Div('Refit is available and defaults to off.'
                                             if sweep_capabilities.get('refit') else 'Current sweep CLI has no --refit flag.',
                                             style={
                                                 'color': COLORS['muted'],
                                                 'fontSize': '12px',
                                                 'marginTop': '6px'
                                             })
                                ],
                                             style={'marginTop': '9px'}),
                                html.Div(id='request-status', style={'marginTop': '8px'}),
                                html.Label('Equivalent CLI', style={
                                    'display': 'block',
                                    'marginTop': '10px',
                                    'fontWeight': 700
                                }),
                                html.Pre(id='command-view',
                                         style={
                                             'whiteSpace': 'pre-wrap',
                                             'background': '#f5f8f9',
                                             'padding': '10px'
                                         }),
                                html.Label('Resolved YAML', style={
                                    'display': 'block',
                                    'fontWeight': 700
                                }),
                                html.Pre(id='resolved-view',
                                         style={
                                             'maxHeight': '320px',
                                             'overflow': 'auto',
                                             'background': '#f5f8f9',
                                             'padding': '10px'
                                         }),
                                html.Div([
                                    _button('Train', 'train', disabled=True),
                                    _button('Stop', 'stop-train', danger=True),
                                    _button('Download CLI', 'save-cli'),
                                    _button('Download YAML', 'save-yaml')
                                ],
                                         style={
                                             'display': 'flex',
                                             'flexWrap': 'wrap',
                                             'gap': '8px'
                                         }),
                                html.Div(id='train-status', style={
                                    'marginTop': '10px',
                                    'whiteSpace': 'pre-wrap'
                                })
                            ]),
                            html.Div(
                                [
                                    _panel([
                                        html.H3('Participant inference', style={'marginTop': 0}),
                                        html.Label('Participant ID'),
                                        dcc.Input(id='inference-participant-id',
                                                  type='text',
                                                  placeholder='Required participant_id',
                                                  style={'width': '100%'},
                                                  debounce=True),
                                        html.Label('Optional existing manifest to import',
                                                   style={
                                                       'display': 'block',
                                                       'marginTop': '8px'
                                                   }),
                                        dcc.Input(id='inference-manifest',
                                                  type='text',
                                                  placeholder='Local YAML/JSON manifest path',
                                                  style={'width': '100%'},
                                                  debounce=True),
                                        html.Div(id='input-contract-status', style={
                                            'fontSize': '12px',
                                            'marginTop': '7px'
                                        }),
                                        dash_table.DataTable(
                                            id='inference-file-table',
                                            columns=[{
                                                'name': 'file_id',
                                                'id': 'file_id',
                                                'editable': True
                                            }, {
                                                'name': 'path',
                                                'id': 'path',
                                                'editable': True
                                            }, {
                                                'name': 'role',
                                                'id': 'role',
                                                'editable': True,
                                                'presentation': 'dropdown'
                                            }, {
                                                'name': 'label',
                                                'id': 'label',
                                                'editable': True
                                            }],
                                            data=[{
                                                'file_id': '',
                                                'path': '',
                                                'role': 'B',
                                                'label': ''
                                            }],
                                            editable=True,
                                            row_deletable=True,
                                            dropdown={
                                                'role': {
                                                    'options': _options(['B', 'R1', 'R2', 'R3', 'R4', 'S1', 'S2', 'W1', 'W2']),
                                                    'clearable': False
                                                }
                                            },
                                            page_size=8,
                                            style_cell={
                                                'fontFamily': 'monospace',
                                                'fontSize': 10,
                                                'textAlign': 'left'
                                            }),
                                        html.Div([_button('Add file', 'add-inference-file')], style={'marginTop': '8px'}),
                                        dcc.Checklist(id='inference-source-contract',
                                                      options=[{
                                                          'label': INFERENCE_SOURCE_CONFIRMATION,
                                                          'value': 'confirmed'
                                                      }],
                                                      value=[],
                                                      style={
                                                          'fontSize': '12px',
                                                          'marginTop': '8px'
                                                      }),
                                        html.Div([
                                            _button('Infer', 'infer'),
                                            _button('Infer CLI', 'save-inference-cli'),
                                            _button('Infer YAML', 'save-inference-yaml')
                                        ],
                                                 style={
                                                     'display': 'flex',
                                                     'flexWrap': 'wrap',
                                                     'gap': '8px',
                                                     'marginTop': '8px'
                                                 }),
                                        html.Label('Equivalent CLI', style={
                                            'display': 'block',
                                            'fontWeight': 700,
                                            'marginTop': '9px'
                                        }),
                                        html.Pre(id='inference-command-view',
                                                 style={
                                                     'whiteSpace': 'pre-wrap',
                                                     'background': '#f5f8f9',
                                                     'padding': '8px'
                                                 }),
                                        html.Label('Resolved YAML', style={
                                            'display': 'block',
                                            'fontWeight': 700,
                                            'marginTop': '9px'
                                        }),
                                        html.Pre(id='inference-yaml-view',
                                                 style={
                                                     'whiteSpace': 'pre-wrap',
                                                     'maxHeight': '260px',
                                                     'overflow': 'auto',
                                                     'background': '#f5f8f9',
                                                     'padding': '8px'
                                                 }),
                                        html.Div(SINGLE_PARTICIPANT_NOTICE,
                                                 style={
                                                     'fontSize': '12px',
                                                     'color': COLORS['muted'],
                                                     'marginTop': '8px'
                                                 }),
                                        html.Pre(id='infer-result',
                                                 style={
                                                     'whiteSpace': 'pre-wrap',
                                                     'maxHeight': '270px',
                                                     'overflow': 'auto'
                                                 })
                                    ]),
                                    _panel([
                                        html.H3('Comparison queue', style={'marginTop': 0}),
                                        html.Div([
                                            dcc.Input(id='comparison-name', placeholder='Case name'),
                                            _button('Add', 'add-comparison'),
                                            _button('Run queue', 'run-comparison'),
                                            _button('Export CLI', 'export-sequence-cli'),
                                            _button('Export YAML', 'export-sequence-yaml'),
                                            _button('Run CLI', 'save-comparison-run-cli'),
                                            _button('Run YAML', 'save-comparison-run-yaml')
                                        ],
                                                 style={
                                                     'display': 'flex',
                                                     'gap': '8px',
                                                     'flexWrap': 'wrap'
                                                 }),
                                        html.Div(id='comparison-status', style={'marginTop': '8px'}),
                                        html.Div(
                                            'Run queue first materializes a parser-backed YAML. Use the Training Stop button above to terminate the whole submitted process group.',
                                            style={
                                                'color': COLORS['muted'],
                                                'fontSize': '12px',
                                                'marginTop': '6px'
                                            }),
                                        dash_table.DataTable(id='comparison-table',
                                                             columns=[{
                                                                 'name': 'order',
                                                                 'id': 'order'
                                                             }, {
                                                                 'name': 'name',
                                                                 'id': 'name'
                                                             }, {
                                                                 'name': 'config SHA',
                                                                 'id': 'config_sha256'
                                                             }, {
                                                                 'name': 'CLI',
                                                                 'id': 'display'
                                                             }],
                                                             data=[],
                                                             style_cell={
                                                                 'fontFamily': 'monospace',
                                                                 'fontSize': 10,
                                                                 'whiteSpace': 'normal',
                                                                 'height': 'auto',
                                                                 'textAlign': 'left'
                                                             }),
                                        html.Label('Sequential CLI', style={
                                            'display': 'block',
                                            'fontWeight': 700,
                                            'marginTop': '9px'
                                        }),
                                        html.Pre(id='comparison-cli-view',
                                                 style={
                                                     'whiteSpace': 'pre-wrap',
                                                     'maxHeight': '220px',
                                                     'overflow': 'auto',
                                                     'background': '#f5f8f9',
                                                     'padding': '8px'
                                                 }),
                                        html.Label('Executable plan or sequential request YAML',
                                                   style={
                                                       'display': 'block',
                                                       'fontWeight': 700,
                                                       'marginTop': '9px'
                                                   }),
                                        html.Pre(id='comparison-yaml-view',
                                                 style={
                                                     'whiteSpace': 'pre-wrap',
                                                     'maxHeight': '260px',
                                                     'overflow': 'auto',
                                                     'background': '#f5f8f9',
                                                     'padding': '8px'
                                                 }),
                                        html.Label('Submitted execution CLI',
                                                   style={
                                                       'display': 'block',
                                                       'fontWeight': 700,
                                                       'marginTop': '9px'
                                                   }),
                                        html.Pre(id='comparison-run-cli-view',
                                                 style={
                                                     'whiteSpace': 'pre-wrap',
                                                     'maxHeight': '160px',
                                                     'overflow': 'auto',
                                                     'background': '#e4f2f0',
                                                     'padding': '8px'
                                                 }),
                                        html.Label('Materialized execution YAML',
                                                   style={
                                                       'display': 'block',
                                                       'fontWeight': 700,
                                                       'marginTop': '9px'
                                                   }),
                                        html.Pre(id='comparison-run-yaml-view',
                                                 style={
                                                     'whiteSpace': 'pre-wrap',
                                                     'maxHeight': '260px',
                                                     'overflow': 'auto',
                                                     'background': '#e4f2f0',
                                                     'padding': '8px'
                                                 })
                                    ])
                                ],
                                style={
                                    'display': 'grid',
                                    'gridTemplateColumns': 'repeat(auto-fit,minmax(min(100%,320px),1fr))',
                                    'gap': '12px',
                                    'marginTop': '12px'
                                })
                        ],
                                          style={'paddingTop': '14px'})),
                    dcc.Tab(
                        label='Analyse',
                        value='analyse',
                        children=html.Div([
                            _panel([
                                html.H3('Analysis request', style={'marginTop': 0}),
                                html.Div([
                                    dcc.Dropdown(id='analysis-run',
                                                 options=_options(study_outputs),
                                                 value=[study_outputs[0]] if study_outputs else [],
                                                 multi=True,
                                                 placeholder='pipeline_output runs'),
                                    dcc.Dropdown(id='analysis-mode',
                                                 options=_options(['single', 'comparison', 'ablation', 'test']),
                                                 value='single',
                                                 clearable=False),
                                    dcc.Dropdown(id='analysis-preset',
                                                 options=_options(sorted(REPORT_PRESETS)),
                                                 value='classification',
                                                 clearable=False),
                                    dcc.Input(id='analysis-reference', placeholder='Reference case'),
                                    dcc.Input(id='analysis-factors', placeholder='factor.path,other.path'),
                                    dcc.Input(id='report-name', placeholder='Optional report name')
                                ],
                                         style={
                                             'display': 'grid',
                                             'gridTemplateColumns': 'repeat(auto-fit,minmax(min(100%,150px),1fr))',
                                             'gap': '8px'
                                         }),
                                html.Details([
                                    html.Summary('Analysis contract'),
                                    html.Div(
                                        [
                                            dcc.Input(id='include-cases', placeholder='include cases,comma'),
                                            dcc.Input(id='exclude-cases', placeholder='exclude cases,comma'),
                                            dcc.Input(
                                                id='comparison-family', value='declared_comparison', placeholder='comparison family'),
                                            dcc.Dropdown(id='validation-depth',
                                                         options=_options(['full', 'selected']),
                                                         value='full',
                                                         clearable=False),
                                            dcc.Dropdown(id='on-missing',
                                                         options=_options(['na', 'error', 'skip']),
                                                         value='na',
                                                         clearable=False),
                                            dcc.Checklist(id='analysis-flags',
                                                          options=[{
                                                              'label': 'V2 compatibility',
                                                              'value': 'v2'
                                                          }],
                                                          value=['v2'])
                                        ],
                                        style={
                                            'display': 'grid',
                                            'gridTemplateColumns': 'repeat(auto-fit,minmax(min(100%,150px),1fr))',
                                            'gap': '8px',
                                            'marginTop': '8px'
                                        })
                                ],
                                             style={'marginTop': '9px'}),
                                html.Label('Analysis modules', style={
                                    'display': 'block',
                                    'marginTop': '9px'
                                }),
                                dcc.Dropdown(id='analysis-modules', options=_options(report_module_ids), value=[], multi=True),
                                html.Div(
                                    [
                                        dcc.Dropdown(id='analysis-figures',
                                                     options=_options(sorted(KNOWN_FIGURES)),
                                                     value=None,
                                                     multi=True,
                                                     placeholder='Preset figures'),
                                        dcc.Dropdown(id='analysis-tables',
                                                     options=_options(sorted(KNOWN_TABLES)),
                                                     value=None,
                                                     multi=True,
                                                     placeholder='Preset tables'),
                                        dcc.Input(id='bootstrap', type='number', value=10000, min=1),
                                        dcc.Input(id='permutation', type='number', value=100000, min=1),
                                        dcc.Input(id='statistics-seed', type='number', value=42),
                                        dcc.Input(id='alpha', type='number', value=0.05, min=0.0001, max=1, step=0.01),
                                        dcc.Input(id='calibration-bins', type='number', value=10, min=2)
                                    ],
                                    style={
                                        'display': 'grid',
                                        'gridTemplateColumns': 'repeat(auto-fit,minmax(min(100%,150px),1fr))',
                                        'gap': '8px',
                                        'marginTop': '9px'
                                    }),
                                html.Div([
                                    _button('Analyse', 'analyse'),
                                    _button('Validate', 'validate-report'),
                                    _button('Stop', 'stop-report', danger=True),
                                    _button('Report CLI', 'save-analysis-cli'),
                                    _button('Report YAML', 'save-analysis-yaml'),
                                    _button('Refresh', 'refresh-outputs')
                                ],
                                         style={
                                             'display': 'flex',
                                             'gap': '8px',
                                             'marginTop': '10px'
                                         }),
                                html.Pre(id='analysis-command',
                                         style={
                                             'whiteSpace': 'pre-wrap',
                                             'background': '#f5f8f9',
                                             'padding': '10px'
                                         }),
                                html.Pre(id='analysis-yaml',
                                         style={
                                             'whiteSpace': 'pre-wrap',
                                             'maxHeight': '260px',
                                             'overflow': 'auto',
                                             'background': '#f5f8f9',
                                             'padding': '10px'
                                         }),
                                html.Div(id='analysis-status', style={'whiteSpace': 'pre-wrap'})
                            ]),
                            html.Div(
                                [
                                    _panel([
                                        html.H3('Pipeline data', style={'marginTop': 0}),
                                        dcc.Dropdown(id='pipeline-table-select', placeholder='Select data table'),
                                        dash_table.DataTable(id='pipeline-table',
                                                             page_size=12,
                                                             filter_action='native',
                                                             sort_action='native',
                                                             style_table={'overflowX': 'auto'},
                                                             style_cell={
                                                                 'fontFamily': 'monospace',
                                                                 'fontSize': 10,
                                                                 'maxWidth': 260,
                                                                 'overflow': 'hidden',
                                                                 'textOverflow': 'ellipsis'
                                                             })
                                    ]),
                                    _panel([
                                        html.H3('Report preview', style={'marginTop': 0}),
                                        dcc.Dropdown(id='report-output',
                                                     options=_options(report_outputs),
                                                     value=report_outputs[0] if report_outputs else None,
                                                     placeholder='report_output run'),
                                        dcc.Dropdown(id='report-table-select', placeholder='Report table', style={'marginTop': '8px'}),
                                        dash_table.DataTable(id='report-table',
                                                             page_size=10,
                                                             style_table={'overflowX': 'auto'},
                                                             style_cell={
                                                                 'fontFamily': 'monospace',
                                                                 'fontSize': 10
                                                             }),
                                        dcc.Dropdown(
                                            id='report-figure-select', placeholder='Report figure', style={'marginTop': '8px'}),
                                        html.Div(id='report-figure', style={'marginTop': '8px'})
                                    ])
                                ],
                                style={
                                    'display': 'grid',
                                    'gridTemplateColumns': 'repeat(auto-fit,minmax(min(100%,320px),1fr))',
                                    'gap': '12px',
                                    'marginTop': '12px'
                                })
                        ],
                                          style={'paddingTop': '14px'})),
                    dcc.Tab(
                        label='Tools',
                        value='tools',
                        children=html.Div([
                            _panel([
                                html.H3('Validated maintenance tools', style={'marginTop': 0}),
                                dcc.Dropdown(id='tool-operation',
                                             options=[{
                                                 'label': label,
                                                 'value': value
                                             } for label, value in TOOL_OPTIONS],
                                             value='pipeline_validate',
                                             clearable=False),
                                html.Div(
                                    'Pipeline validate/show-config reuse the current server-built Run request. Analyse > Validate reuses every current Analyse selector.',
                                    style={
                                        'color': COLORS['muted'],
                                        'fontSize': '12px',
                                        'marginTop': '7px'
                                    }),
                                html.Details([
                                    html.Summary('Common paths and policy'),
                                    html.Div(
                                        [
                                            dcc.Input(
                                                id='tool-pipeline-path', placeholder='pipeline_output/<run>', style={'width': '100%'}),
                                            dcc.Input(
                                                id='tool-report-path', placeholder='report_output/<run>', style={'width': '100%'}),
                                            dcc.Input(id='tool-plan-path', placeholder='Study-plan YAML', style={'width': '100%'}),
                                            dcc.Dropdown(id='tool-validation-mode',
                                                         options=_options(['config', 'smoke', 'full']),
                                                         value='smoke',
                                                         clearable=False),
                                            dcc.Checklist(id='tool-flags',
                                                          options=[{
                                                              'label': 'Hash predictions',
                                                              'value': 'hash'
                                                          }, {
                                                              'label': 'Replace workbook',
                                                              'value': 'replace'
                                                          }],
                                                          value=[],
                                                          inline=True)
                                        ],
                                        style={
                                            'display': 'grid',
                                            'gridTemplateColumns': 'repeat(auto-fit,minmax(min(100%,190px),1fr))',
                                            'gap': '8px',
                                            'marginTop': '8px'
                                        })
                                ],
                                             style={'marginTop': '9px'}),
                                html.Details([
                                    html.Summary('Specialized artifact workflows'),
                                    html.Div(
                                        [
                                            dcc.Input(id='tool-special-plan', placeholder='Specialized YAML'),
                                            dcc.Input(id='tool-source-root', value='.', placeholder='V5 source root'),
                                            dcc.Input(id='tool-special-output', placeholder='Optional report output name'),
                                            dcc.Input(id='tool-special-study', placeholder='Optional source study'),
                                            dcc.Input(id='tool-special-run-name', placeholder='Optional computation run name'),
                                            dcc.Input(id='tool-special-resume', placeholder='Optional computation resume'),
                                            dcc.Input(id='tool-special-upstream', placeholder='Optional upstream study'),
                                            dcc.Input(id='tool-special-case', placeholder='Optional case ID'),
                                            dcc.Input(id='tool-prediction-file', placeholder='Optional prediction file'),
                                            dcc.Input(id='tool-step', type='number', min=1e-12, placeholder='Optional step'),
                                            dcc.Input(id='tool-special-report-input', placeholder='Special report input'),
                                            dcc.Checklist(id='tool-special-flags',
                                                          options=[{
                                                              'label': 'Exclude denoiser',
                                                              'value': 'no_denoiser'
                                                          }, {
                                                              'label': 'Completion dry run',
                                                              'value': 'dry'
                                                          }],
                                                          value=[],
                                                          inline=True)
                                        ],
                                        style={
                                            'display': 'grid',
                                            'gridTemplateColumns': 'repeat(auto-fit,minmax(min(100%,190px),1fr))',
                                            'gap': '8px',
                                            'marginTop': '8px'
                                        })
                                ],
                                             style={'marginTop': '9px'}),
                                html.Div([
                                    _button('Build CLI', 'build-tool'),
                                    _button('Run tool', 'run-tool'),
                                    _button('Stop tool', 'stop-tool', danger=True),
                                    _button('Download CLI', 'save-tool-cli'),
                                    _button('Download YAML', 'save-tool-yaml')
                                ],
                                         style={
                                             'display': 'flex',
                                             'flexWrap': 'wrap',
                                             'gap': '8px',
                                             'marginTop': '12px'
                                         }),
                                html.Label('Equivalent CLI', style={
                                    'display': 'block',
                                    'fontWeight': 700,
                                    'marginTop': '10px'
                                }),
                                html.Pre(id='tool-command',
                                         style={
                                             'whiteSpace': 'pre-wrap',
                                             'background': '#f5f8f9',
                                             'padding': '10px'
                                         }),
                                html.Label('Request YAML', style={
                                    'display': 'block',
                                    'fontWeight': 700
                                }),
                                html.Pre(id='tool-yaml',
                                         style={
                                             'whiteSpace': 'pre-wrap',
                                             'maxHeight': '260px',
                                             'overflow': 'auto',
                                             'background': '#f5f8f9',
                                             'padding': '10px'
                                         }),
                                html.Pre(id='tool-status', style={
                                    'whiteSpace': 'pre-wrap',
                                    'maxHeight': '300px',
                                    'overflow': 'auto'
                                })
                            ]),
                            _panel([
                                html.H3('Equivalent read-only surfaces', style={'marginTop': 0}),
                                html.Div(EQUIVALENT_SURFACE_NOTICE,
                                         id='equivalent-tools',
                                         style={
                                             'color': COLORS['muted'],
                                             'fontSize': '13px'
                                         }),
                                html.Pre(
                                    'Configure = pipeline.py modules/presets/parameters/manual-cli\nAnalyse selectors = analyse_report.py list',
                                    style={
                                        'whiteSpace': 'pre-wrap',
                                        'background': '#fff6e6',
                                        'padding': '10px'
                                    })
                            ],
                                   style={'marginTop': '12px'})
                        ],
                                          style={'paddingTop': '14px'}))
                ])
        ],
        style={
            'background': COLORS['background'],
            'color': COLORS['ink'],
            'fontFamily': 'Inter, Segoe UI, sans-serif',
            'minHeight': '100vh',
            'padding': 'clamp(10px,2.5vw,26px) clamp(10px,3vw,26px) 40px'
        })

    @app.callback(Output('training-yaml', 'options'),
                  Output('model-export', 'options'),
                  Output('sweep-plan', 'options'),
                  Input('refresh-configs', 'n_clicks'),
                  prevent_initial_call=True)
    def refresh_configuration_sources(_: int) -> tuple[Any, Any, Any]:
        return (_options(list(control.yaml_paths())), _options(list(control.model_exports())),
                _options(list(control.study_plan_paths())))

    @app.callback(Output('model-case', 'options'), Output('model-case', 'value'), Input('model-export', 'value'))
    def model_case_options(export: str | None) -> tuple[Any, Any]:
        if not export:
            return ([], None)
        try:
            values = list(control.model_cases(export))
        except Exception:
            return ([], None)
        return (_options(values), values[0] if values else None)

    @app.callback(Output('config-state', 'data'),
                  Output('module-controls', 'children'),
                  Output('parameter-table', 'data'),
                  Output('parameter-sliders', 'children'),
                  Output('training-yaml', 'value'),
                  Output('config-status', 'children'),
                  Input('load-config', 'n_clicks'),
                  State('config-source', 'value'),
                  State('training-yaml', 'value'),
                  State('model-export', 'value'),
                  State('model-case', 'value'),
                  prevent_initial_call=True)
    def load_configuration(_: int, source: str, yaml_path: str | None, export: str | None,
                           case_id: str | None) -> tuple[Any, Any, Any, Any, Any, str]:
        try:
            if source == 'model_config':
                if not export:
                    raise ValueError('select a model_config export')
                loaded = control.load_model_defaults(export, case_id)
                config = loaded.config
                defaults = loaded.module_defaults
                features = loaded.feature_defaults
                selected_yaml = loaded.config_path
                capability = loaded.inference_capability
            else:
                if not yaml_path:
                    raise ValueError('select a training YAML')
                config, _ = control.load_yaml(yaml_path)
                defaults = control.module_defaults_from_config(config)
                feature_section = config.get('features', {})
                raw_features = feature_section.get('enabled_groups', []) if isinstance(feature_section, Mapping) else []
                features = tuple((str(value) for value in raw_features))
                selected_yaml = yaml_path
                capability = {'available': False, 'reason': 'YAML_has_no_learned_weights'}
            catalog = control.module_catalog(export if source == 'model_config' else None)
            rows = flatten_parameters(config, parameter_contract=parameter_contract)
            state = {
                'config_path': selected_yaml,
                'default_modules': defaults,
                'default_features': list(features),
                'model_export': export if source == 'model_config' else None,
                'model_case': case_id if source == 'model_config' else None,
                'inference_capability': capability
            }
            status = f"Loaded {selected_yaml}. Registry families: {len(catalog)}. Inference bundle: {('available' if capability.get('available') else 'unavailable')}; adapter: {capability.get('adapter_source', 'none')}."
            return (state, _module_controls(catalog, defaults, features), rows, _numeric_sliders(rows), selected_yaml, status)
        except Exception as error:
            return (no_update, no_update, no_update, no_update, no_update, _error(error))

    @app.callback(Output('parameter-table', 'data', allow_duplicate=True),
                  Input({
                      'type': 'parameter-slider',
                      'path': ALL
                  }, 'value'),
                  State({
                      'type': 'parameter-slider',
                      'path': ALL
                  }, 'id'),
                  State('parameter-table', 'data'),
                  prevent_initial_call=True)
    def apply_numeric_sliders(values: Sequence[int | float], identities: Sequence[Mapping[str, Any]],
                              rows: Sequence[Mapping[str, Any]]) -> Any:
        if not rows:
            return no_update
        by_path = {str(identity['path']): value for identity, value in zip(identities, values)}
        output = [dict(row) for row in rows]
        for row in output:
            path = str(row.get('path', ''))
            if path not in by_path:
                continue
            original = yaml.safe_load(str(row.get('original_yaml', 'null')))
            value: int | float = by_path[path]
            if isinstance(original, int) and (not isinstance(original, bool)):
                value = int(round(float(value)))
            row['value_yaml'] = yaml.safe_dump(value, default_flow_style=True, sort_keys=False).strip().removesuffix('...').rstrip()
        return output

    @app.callback(Output('train-request', 'data'), Output('command-view', 'children'), Output('resolved-view', 'children'),
                  Output('request-status', 'children'), Output('train', 'disabled'), Input('config-state', 'data'),
                  Input({
                      'type': 'module-select',
                      'family': ALL
                  }, 'value'), Input('parameter-table', 'data'), Input('train-operation', 'value'), Input('sweep-plan', 'value'),
                  Input('run-name', 'value'), Input('study-id', 'value'), Input('study-purpose', 'value'), Input('config-id', 'value'),
                  Input('unset-paths', 'value'), Input('repeats', 'value'), Input('folds', 'value'), Input('jobs', 'value'),
                  Input('device', 'value'), Input('cache-mode', 'value'), Input('cache-root', 'value'),
                  Input('cache-namespaces', 'value'), Input('resume-directory', 'value'), Input('environment-lock', 'value'),
                  Input('environment-policy', 'value'), Input('execution-flags', 'value'), Input('refit-enabled', 'value'),
                  State({
                      'type': 'module-select',
                      'family': ALL
                  }, 'id'))
    def build_request(state: Mapping[str, Any] | None, module_values: Sequence[Any], parameter_rows: Sequence[Mapping[str, Any]],
                      operation: str, sweep_plan: str | None, run_name: str | None, study_id: str | None, purpose: str | None,
                      config_id: str | None, unset_text: str | None, repeats: str | Sequence[int | str],
                      folds: str | Sequence[int | str], job_count: int, device: str, cache_mode: str, cache_root: str | None,
                      cache_namespaces: str | None, resume_directory: str | None, environment_lock: str | None,
                      environment_policy: str, execution_flags: Sequence[str], refit_enabled: Sequence[str],
                      module_ids: Sequence[Mapping[str, Any]]) -> tuple[Any, str, str, str, bool]:
        if not state and operation == 'run':
            return (None, 'Select and load a YAML first.', '', 'Not ready', True)
        try:
            state = state or {}
            selected: dict[str, Any] = {}
            features: list[str] = []
            for identity, value in zip(module_ids, module_values):
                family = str(identity['family'])
                if family == 'feature_group':
                    features = list(value or [])
                else:
                    selected[family] = value
            request = control.build_train_request(
                config_path=state.get('config_path'),
                plan_path=sweep_plan or None,
                operation=operation,
                selected_modules=selected,
                default_modules=state.get('default_modules', {}),
                feature_groups=features,
                default_feature_groups=state.get('default_features', []),
                parameter_rows=parameter_rows or [],
                unset_paths=[value.strip() for value in str(unset_text or '').splitlines() if value.strip()],
                config_id=config_id or None,
                study_id=study_id or None,
                purpose=purpose or None,
                repeats=repeats,
                folds=folds,
                jobs=int(job_count or 1),
                device=device,
                cache_mode=cache_mode,
                cache_root=cache_root or None,
                cache_namespaces=cache_namespaces or None,
                continue_on_error='continue' in (execution_flags or []),
                measure_operational_costs='costs' in (execution_flags or []),
                hash_predictions='hash' in (execution_flags or []),
                dry_run='dry' in (execution_flags or []),
                resume=resume_directory or None,
                run_name=run_name or None,
                environment_lock=environment_lock or None,
                environment_policy=environment_policy,
                refit='refit' in (refit_enabled or []))
            identity = 'plan' if operation == 'sweep' else 'config'
            return (request.to_dict(), request.display, request.resolved_yaml,
                    f'Ready · {identity} {request.config_sha256[:12]}', False)
        except Exception as error:
            return (None, '', '', _error(error), True)

    @app.callback(Output('active-train-job', 'data'),
                  Output('train-status', 'children', allow_duplicate=True),
                  Input('train', 'n_clicks'),
                  Input('stop-train', 'n_clicks'),
                  State('train-request', 'data'),
                  State('active-train-job', 'data'),
                  prevent_initial_call=True)
    def control_training(_: int, __: int, request_data: Mapping[str, Any] | None, job_id: str | None) -> tuple[Any, str]:
        trigger = callback_context.triggered_id
        try:
            if trigger == 'stop-train':
                if not job_id:
                    return (None, 'No active training job.')
                jobs.terminate(job_id)
                return (job_id, f'Stop requested for {job_id}.')
            if not request_data:
                raise ValueError('Train requires a selected YAML and a valid request')
            if job_id:
                try:
                    if jobs.status(job_id)['state'] == 'running':
                        raise RuntimeError('stop the active training job first')
                except KeyError:
                    pass
            from .control_service import CommandRequest
            request = CommandRequest(script=str(request_data['script']),
                                     arguments=tuple(request_data['arguments']),
                                     display=str(request_data['display']),
                                     resolved_yaml=str(request_data.get('resolved_yaml', '')),
                                     config_sha256=str(request_data.get('config_sha256', '')))
            new_id = jobs.start_request(request, kind='pipeline')
            return (new_id, f'Training job {new_id} started.')
        except Exception as error:
            return (job_id, _error(error))

    @app.callback(Output('train-status', 'children'),
                  Input('job-poll', 'n_intervals'),
                  State('active-train-job', 'data'),
                  prevent_initial_call=True)
    def poll_training(_: int, job_id: str | None) -> Any:
        if not job_id:
            return no_update
        try:
            payload = jobs.status(job_id)
        except Exception as error:
            return _error(error)
        progress = payload.get('progress') or {}
        return f"{payload['state']} · {payload['elapsed_s']:.1f}s\n{(_as_json(progress) if progress else '')}\n" + '\n'.join(
            payload.get('log_tail', [])[-12:])

    @app.callback(Output('inference-participant-id', 'value'), Output('inference-file-table', 'data'),
                  Output('input-contract-status', 'children'), Input('inference-manifest', 'value'))
    def inspect_inference_input(manifest: str | None) -> tuple[Any, Any, str]:
        if not manifest:
            return (no_update, no_update, 'Enter a participant and files, or import a local YAML/JSON manifest.')
        try:
            resolved = control.read_inference_manifest(manifest)
            rows = [{'file_id': row.file_id, 'role': row.role, 'label': row.label, 'path': row.path} for row in resolved.files]
            return (
                resolved.participant_id, rows,
                f'participant={resolved.participant_id}; files={len(rows)}; labelled participants={resolved.labelled_participant_count}. '
                + SINGLE_PARTICIPANT_NOTICE)
        except Exception as error:
            return (no_update, no_update, _error(error))

    @app.callback(Output('inference-file-table', 'data', allow_duplicate=True),
                  Input('add-inference-file', 'n_clicks'),
                  State('inference-file-table', 'data'),
                  prevent_initial_call=True)
    def add_inference_file(_: int, rows: Sequence[Mapping[str, Any]] | None) -> Any:
        return [*(dict(row) for row in rows or []), {'file_id': '', 'path': '', 'role': 'B', 'label': ''}]

    @app.callback(Output('infer-result', 'children'),
                  Output('inference-request', 'data'),
                  Output('inference-command-view', 'children'),
                  Output('inference-yaml-view', 'children'),
                  Output('inference-manifest', 'value'),
                  Input('infer', 'n_clicks'),
                  State('config-state', 'data'),
                  State('inference-participant-id', 'value'),
                  State('inference-file-table', 'data'),
                  State('inference-source-contract', 'value'),
                  prevent_initial_call=True)
    def run_inference(_: int, state: Mapping[str, Any] | None, participant_id: str | None, files: Sequence[Mapping[str, Any]] | None,
                      source_confirmation: Sequence[str] | None) -> tuple[str, Any, Any, Any, Any]:
        request = None
        manifest_path = None
        try:
            if not state or not state.get('model_export'):
                raise ValueError('Infer requires model_config defaults with a deployable bundle')
            manifest_path = control.materialize_inference_manifest(participant_id=str(participant_id or ''),
                                                                   files=files or (),
                                                                   source_contract_confirmed='confirmed' in (source_confirmation
                                                                                                             or ()))
            request = control.build_inference_request(model_export=str(state['model_export']),
                                                      case_id=state.get('model_case'),
                                                      input_manifest=manifest_path)
            result = control.infer(model_export=str(state['model_export']),
                                   case_id=state.get('model_case'),
                                   input_manifest=manifest_path)
            return (_as_json({
                'analysis_limit': SINGLE_PARTICIPANT_NOTICE,
                'result': result
            }), request.to_dict(), request.display, request.resolved_yaml, control.cli_input_path(manifest_path))
        except Exception as error:
            return (_error(error), request.to_dict() if request is not None else no_update,
                    request.display if request is not None else '', request.resolved_yaml if request is not None else '',
                    control.cli_input_path(manifest_path) if manifest_path is not None else no_update)

    @app.callback(Output('download-inference-cli', 'data'),
                  Input('save-inference-cli', 'n_clicks'),
                  State('inference-request', 'data'),
                  prevent_initial_call=True)
    def download_inference_cli(_: int, request: Mapping[str, Any] | None) -> Any:
        if not request:
            return no_update
        return dcc.send_string(str(request.get('display', '')) + '\n', 'inference_command.sh')

    @app.callback(Output('download-inference-yaml', 'data'),
                  Input('save-inference-yaml', 'n_clicks'),
                  State('inference-request', 'data'),
                  prevent_initial_call=True)
    def download_inference_yaml(_: int, request: Mapping[str, Any] | None) -> Any:
        if not request:
            return no_update
        return dcc.send_string(str(request.get('resolved_yaml', '')), 'inference_request.yaml')

    @app.callback(Output('download-cli', 'data'),
                  Input('save-cli', 'n_clicks'),
                  State('train-request', 'data'),
                  prevent_initial_call=True)
    def download_cli(_: int, request: Mapping[str, Any] | None) -> Any:
        if not request:
            return no_update
        return dcc.send_string(str(request['display']) + '\n', 'pipeline_command.sh')

    @app.callback(Output('download-yaml', 'data'),
                  Input('save-yaml', 'n_clicks'),
                  State('train-request', 'data'),
                  prevent_initial_call=True)
    def download_yaml(_: int, request: Mapping[str, Any] | None) -> Any:
        if not request:
            return no_update
        filename = 'resolved_study_plan.yaml' if request.get('script') == 'sweep.py' else 'resolved_pipeline.yaml'
        return dcc.send_string(str(request.get('resolved_yaml', '')), filename)

    @app.callback(Output('comparison-store', 'data'),
                  Output('comparison-table', 'data'),
                  Output('comparison-cli-view', 'children'),
                  Output('comparison-yaml-view', 'children'),
                  Output('comparison-execution-request', 'data'),
                  Output('comparison-run-cli-view', 'children'),
                  Output('comparison-run-yaml-view', 'children'),
                  Output('comparison-status', 'children'),
                  Input('add-comparison', 'n_clicks'),
                  State('comparison-name', 'value'),
                  State('train-request', 'data'),
                  State('comparison-store', 'data'),
                  State('comparison-execution-request', 'data'),
                  prevent_initial_call=True)
    def add_comparison(_: int, name: str | None, request: Mapping[str, Any] | None, stored: Sequence[Mapping[str, Any]] | None,
                       execution_request: Mapping[str, Any] | None) -> tuple[Any, Any, str, str, Any, Any, Any, str]:
        current = [dict(row) for row in stored or []]
        try:
            if not request:
                raise ValueError('build a valid request before Add')
            candidate = {
                'name': str(name or '').strip(),
                'script': str(request['script']),
                'arguments': list(request['arguments']),
                'display': str(request['display']),
                'config_sha256': str(request.get('config_sha256', '')),
                'resolved_yaml': str(request.get('resolved_yaml', ''))
            }
            trial = [*current, candidate]
            sequential_cli = comparison_sequence_cli(trial)
            export_yaml = comparison_sequence_export_yaml(trial, pipeline_root=control.pipeline_root)
            table = [{'order': i + 1, **row} for i, row in enumerate(trial)]
            return (trial, table, sequential_cli, export_yaml, None, '', '',
                    f'Cached {len(trial)} sequential cases. Executable sequence YAML ready.')
        except Exception as error:
            try:
                current_cli = comparison_sequence_cli(current)
                current_yaml = comparison_sequence_export_yaml(current, pipeline_root=control.pipeline_root)
            except Exception:
                current_cli = ''
                current_yaml = ''
            return (current, [{
                'order': i + 1,
                **row
            } for i, row in enumerate(current)], current_cli, current_yaml, execution_request, no_update, no_update, _error(error))

    @app.callback(Output('comparison-execution-request', 'data', allow_duplicate=True),
                  Output('active-train-job', 'data', allow_duplicate=True),
                  Output('comparison-run-cli-view', 'children', allow_duplicate=True),
                  Output('comparison-run-yaml-view', 'children', allow_duplicate=True),
                  Output('comparison-status', 'children', allow_duplicate=True),
                  Input('run-comparison', 'n_clicks'),
                  State('comparison-store', 'data'),
                  State('active-train-job', 'data'),
                  prevent_initial_call=True)
    def run_comparison_queue(_: int, stored: Sequence[Mapping[str, Any]] | None, job_id: str | None) -> tuple[Any, Any, Any, Any, str]:
        request: CommandRequest | None = None
        target: Path | None = None
        try:
            if job_id:
                try:
                    if jobs.status(job_id)['state'] == 'running':
                        raise RuntimeError('stop the active training job first')
                except KeyError:
                    pass
            request, target = control.build_comparison_execution_request(stored or [])
            new_id = jobs.start_request(request, kind='pipeline')
            payload = yaml.safe_load(request.resolved_yaml)
            schema = str(payload.get('schema_version', 'unknown'))
            return (request.to_dict(), new_id, request.display, request.resolved_yaml,
                    f'Training job {new_id} started from {schema} at {control.relative(target)}.')
        except Exception as error:
            return (request.to_dict() if request is not None else no_update, job_id,
                    request.display if request is not None else no_update, request.resolved_yaml if request is not None else no_update,
                    _error(error))

    @app.callback(Output('download-sequence-cli', 'data'),
                  Input('export-sequence-cli', 'n_clicks'),
                  State('comparison-store', 'data'),
                  prevent_initial_call=True)
    def export_sequence_cli(_: int, stored: Sequence[Mapping[str, Any]]) -> Any:
        try:
            return dcc.send_string(comparison_sequence_cli(stored or []), 'comparison_sequence.sh')
        except Exception:
            return no_update

    @app.callback(Output('download-sequence-yaml', 'data'),
                  Input('export-sequence-yaml', 'n_clicks'),
                  State('comparison-store', 'data'),
                  prevent_initial_call=True)
    def export_sequence_yaml(_: int, stored: Sequence[Mapping[str, Any]]) -> Any:
        try:
            return dcc.send_string(comparison_sequence_export_yaml(stored or [], pipeline_root=control.pipeline_root),
                                   'comparison_sequence.yaml')
        except Exception:
            return no_update

    @app.callback(Output('download-comparison-run-cli', 'data'),
                  Input('save-comparison-run-cli', 'n_clicks'),
                  State('comparison-execution-request', 'data'),
                  prevent_initial_call=True)
    def download_comparison_run_cli(_: int, request: Mapping[str, Any] | None) -> Any:
        if not request:
            return no_update
        return dcc.send_string(str(request.get('display', '')) + '\n', 'comparison_execution.sh')

    @app.callback(Output('download-comparison-run-yaml', 'data'),
                  Input('save-comparison-run-yaml', 'n_clicks'),
                  State('comparison-execution-request', 'data'),
                  prevent_initial_call=True)
    def download_comparison_run_yaml(_: int, request: Mapping[str, Any] | None) -> Any:
        if not request:
            return no_update
        return dcc.send_string(str(request.get('resolved_yaml', '')), 'comparison_execution.yaml')

    @app.callback(Output('preview-record', 'options'), Output('preview-record', 'value'), Input('preview-participant', 'value'),
                  Input('preview-role', 'value'))
    def preview_records(participant: str | None, role: str | None) -> tuple[Any, Any]:
        selected = [
            row for row in records if (participant is None or row.participant_id == participant) and (role is None or row.role == role)
        ]
        options = [{'label': f'{row.role} · {row.record_id} · {row.duration_s:.1f}s', 'value': row.record_id} for row in selected]
        return (options, selected[0].record_id if selected else None)

    @app.callback(Output('preview-store', 'data'),
                  Output('time-graph', 'figure'),
                  Output('spectrum-graph', 'figure'),
                  Output('stage-table', 'data'),
                  Output('stage-table', 'columns'),
                  Output('preview-status', 'children'),
                  Input('preview', 'n_clicks'),
                  State('config-state', 'data'),
                  State('train-request', 'data'),
                  State('preview-record', 'value'),
                  State('preview-start', 'value'),
                  State('preview-duration', 'value'),
                  State('preview-traces', 'value'),
                  State('preview-stages', 'value'),
                  State('preview-artifact-run', 'value'),
                  prevent_initial_call=True)
    def build_preview(_: int, state: Mapping[str, Any] | None, train_request: Mapping[str, Any] | None, record_id: str | None,
                      start: float, duration: float, traces: Sequence[str], stages: Sequence[str],
                      artifact_run: str | None) -> tuple[Any, Any, Any, Any, Any, str]:
        empty = go.Figure()
        try:
            if not state or not record_id:
                raise ValueError('load a config and select a recording')
            resolved_payload = None
            if train_request and train_request.get('script') == 'pipeline.py':
                candidate = yaml.safe_load(str(train_request.get('resolved_yaml', '')))
                if isinstance(candidate, Mapping) and candidate.get('schema_version') == 'ppg_frailty.pipeline_config.v2':
                    resolved_payload = candidate
            result = preview.preview(config_path=str(state['config_path']) if resolved_payload is None else None,
                                     config_payload=resolved_payload,
                                     record_id=record_id,
                                     start_s=float(start or 0),
                                     duration_s=float(duration or 20),
                                     trace_names=traces,
                                     stage_names=stages)
            time_figure = go.Figure()
            spectrum_figure = go.Figure()
            time_s = result.time_s
            for name, values in result.traces.items():
                time_figure.add_scatter(x=time_s, y=np.asarray(values), mode='lines', name=name)
            for name, (frequency, power) in result.spectra.items():
                spectrum_figure.add_scatter(x=frequency, y=power, mode='lines', name=name)
            time_figure.update_layout(template='plotly_white', title='Canonical stage traces', xaxis_title='Time (s)')
            spectrum_figure.update_layout(template='plotly_white', title='Welch spectra', xaxis_title='Hz', yaxis_type='log')
            artifact_stages = {'representation_model', 'aggregation'} & set(stages or ())
            rows = [dict(row) for row in result.stage_rows if str(row.get('stage')) not in artifact_stages]
            if artifact_stages:
                try:
                    completed_rows = control.completed_workflow_stage_rows(artifact_run,
                                                                           record_id=result.record_id,
                                                                           participant_id=result.participant_id)
                    rows.extend((dict(row) for row in completed_rows if str(row.get('stage')) in artifact_stages))
                except Exception as artifact_error:
                    rows.extend(({
                        'stage': stage,
                        'metric': 'completed_artifact',
                        'value': _error(artifact_error),
                        'status': 'failed'
                    } for stage in sorted(artifact_stages)))
            columns = [{'name': key, 'id': key} for key in (rows[0].keys() if rows else ())]
            stored = {
                'record_id': result.record_id,
                'metadata': dict(result.stage_metadata),
                'stage_rows': rows,
                'completed_artifact': artifact_run
            }
            if any((row.get('status') == 'failed' for row in rows)):
                artifact_status = 'model/aggregation artifact failed validation'
            elif any((row.get('status') == 'artifact' for row in rows)):
                artifact_status = 'completed OOF artifact loaded'
            else:
                artifact_status = 'model/aggregation artifact N/A'
            return (stored, time_figure, spectrum_figure, rows, columns,
                    f'Previewed {record_id}; no model fit executed; {artifact_status}.')
        except Exception as error:
            return (None, empty, empty, [], [], _error(error))

    @app.callback(Output('active-report-job', 'data'),
                  Output('analysis-request', 'data'),
                  Output('analysis-command', 'children'),
                  Output('analysis-yaml', 'children'),
                  Output('analysis-status', 'children', allow_duplicate=True),
                  Input('analyse', 'n_clicks'),
                  Input('validate-report', 'n_clicks'),
                  Input('stop-report', 'n_clicks'),
                  State('analysis-run', 'value'),
                  State('analysis-mode', 'value'),
                  State('analysis-preset', 'value'),
                  State('analysis-modules', 'value'),
                  State('analysis-figures', 'value'),
                  State('analysis-tables', 'value'),
                  State('analysis-reference', 'value'),
                  State('analysis-factors', 'value'),
                  State('report-name', 'value'),
                  State('include-cases', 'value'),
                  State('exclude-cases', 'value'),
                  State('comparison-family', 'value'),
                  State('validation-depth', 'value'),
                  State('on-missing', 'value'),
                  State('analysis-flags', 'value'),
                  State('bootstrap', 'value'),
                  State('permutation', 'value'),
                  State('statistics-seed', 'value'),
                  State('alpha', 'value'),
                  State('calibration-bins', 'value'),
                  State('active-report-job', 'data'),
                  prevent_initial_call=True)
    def control_analysis(_: int, __: int, ___: int, run: Sequence[str] | None, mode: str, preset: str, modules: Sequence[str],
                         figures: Sequence[str] | None, tables: Sequence[str] | None, reference: str | None, factors: str | None,
                         report_name: str | None, include_cases: str | None, exclude_cases: str | None, comparison_family: str,
                         validation_depth: str, on_missing: str, analysis_flags: Sequence[str], bootstrap: int, permutation: int,
                         seed: int, alpha: float, bins: int, job_id: str | None) -> tuple[Any, Any, Any, Any, str]:
        trigger = callback_context.triggered_id
        try:
            if trigger == 'stop-report':
                if not job_id:
                    return (None, no_update, no_update, no_update, 'No active analysis job.')
                jobs.terminate(job_id)
                return (job_id, no_update, no_update, no_update, f'Stop requested for {job_id}.')
            if not run:
                raise ValueError('select a pipeline_output run')
            validation_only = trigger == 'validate-report'
            request = control.build_analysis_request(
                run_paths=list(run),
                mode=mode,
                preset=preset,
                modules=modules or [],
                figures=figures,
                tables=tables,
                reference_case=reference or None,
                factor_paths=[value.strip() for value in (factors or '').split(',') if value.strip()],
                bootstrap_resamples=int(bootstrap),
                permutation_resamples=int(permutation),
                statistics_seed=int(seed),
                alpha=float(alpha),
                calibration_bins=int(bins),
                output_name=None if validation_only else report_name or None,
                include_cases=[value.strip() for value in (include_cases or '').split(',') if value.strip()],
                exclude_cases=[value.strip() for value in (exclude_cases or '').split(',') if value.strip()],
                comparison_family=comparison_family,
                validation_depth=validation_depth,
                on_missing=on_missing,
                allow_v2_compatibility='v2' in (analysis_flags or []),
                command='validate' if validation_only else 'run')
            if job_id:
                try:
                    if jobs.status(job_id)['state'] == 'running':
                        raise RuntimeError('stop the active analysis job first')
                except KeyError:
                    pass
            new_id = jobs.start_request(request, kind='analysis')
            action = 'validation' if validation_only else 'analysis'
            return (new_id, request.to_dict(), request.display, request.resolved_yaml, f'Report {action} job {new_id} started.')
        except Exception as error:
            return (job_id, no_update, no_update, no_update, _error(error))

    @app.callback(Output('download-analysis-cli', 'data'),
                  Input('save-analysis-cli', 'n_clicks'),
                  State('analysis-request', 'data'),
                  prevent_initial_call=True)
    def download_analysis_cli(_: int, request: Mapping[str, Any] | None) -> Any:
        if not request:
            return no_update
        return dcc.send_string(str(request.get('display', '')) + '\n', 'analysis_command.sh')

    @app.callback(Output('download-analysis-yaml', 'data'),
                  Input('save-analysis-yaml', 'n_clicks'),
                  State('analysis-request', 'data'),
                  prevent_initial_call=True)
    def download_analysis_yaml(_: int, request: Mapping[str, Any] | None) -> Any:
        if not request:
            return no_update
        return dcc.send_string(str(request.get('resolved_yaml', '')), 'analysis_request.yaml')

    @app.callback(Output('analysis-status', 'children'),
                  Input('job-poll', 'n_intervals'),
                  State('active-report-job', 'data'),
                  prevent_initial_call=True)
    def poll_analysis(_: int, job_id: str | None) -> Any:
        if not job_id:
            return no_update
        try:
            payload = jobs.status(job_id)
        except Exception as error:
            return _error(error)
        return f"{payload['state']} · {payload['elapsed_s']:.1f}s\n" + '\n'.join(payload.get('log_tail', [])[-12:])

    @app.callback(Output('tool-request', 'data'),
                  Output('tool-command', 'children'),
                  Output('tool-yaml', 'children'),
                  Output('tool-status', 'children', allow_duplicate=True),
                  Input('build-tool', 'n_clicks'),
                  State('tool-operation', 'value'),
                  State('train-request', 'data'),
                  State('environment-policy', 'value'),
                  State('environment-lock', 'value'),
                  State('device', 'value'),
                  State('jobs', 'value'),
                  State('tool-validation-mode', 'value'),
                  State('tool-pipeline-path', 'value'),
                  State('tool-report-path', 'value'),
                  State('tool-plan-path', 'value'),
                  State('tool-flags', 'value'),
                  State('tool-special-plan', 'value'),
                  State('tool-source-root', 'value'),
                  State('tool-special-output', 'value'),
                  State('tool-special-study', 'value'),
                  State('tool-special-run-name', 'value'),
                  State('tool-special-resume', 'value'),
                  State('tool-special-upstream', 'value'),
                  State('tool-special-case', 'value'),
                  State('tool-prediction-file', 'value'),
                  State('tool-step', 'value'),
                  State('tool-special-report-input', 'value'),
                  State('tool-special-flags', 'value'),
                  prevent_initial_call=True)
    def build_tool_request(_: int, operation: str, train_request: Mapping[str, Any] | None, environment_policy: str,
                           environment_lock: str | None, device: str | None, job_count: int | None, validation_mode: str,
                           pipeline_path: str | None, report_path: str | None, plan_path: str | None, tool_flags: Sequence[str] | None,
                           specialized_plan: str | None, source_root: str | None, specialized_output: str | None,
                           specialized_study: str | None, specialized_run_name: str | None, specialized_resume: str | None,
                           specialized_upstream: str | None, specialized_case: str | None, prediction_file: str | None,
                           step: float | None, specialized_report_input: str | None,
                           specialized_flags: Sequence[str] | None) -> tuple[Any, str, str, str]:
        def required(value: Any, label: str) -> str:
            text = str(value or '').strip()
            if not text:
                raise ValueError(f'{label} is required')
            return text

        try:
            flags = set(tool_flags or ())
            if operation in {'pipeline_validate', 'show_config'}:
                request = control.build_pipeline_config_tool_request(
                    operation='validate' if operation == 'pipeline_validate' else 'show-config',
                    run_request=train_request,
                    validation_mode=validation_mode,
                    environment_policy=environment_policy,
                    environment_lock=environment_lock)
            elif operation == 'sweep_validate':
                request = control.build_sweep_validate_request(plan_path=required(plan_path, 'study-plan YAML'),
                                                               environment_policy=environment_policy,
                                                               environment_lock=environment_lock)
            elif operation == 'pipeline_index':
                request = control.build_pipeline_index_request(study_directory=required(pipeline_path, 'pipeline output'),
                                                               hash_predictions='hash' in flags)
            elif operation == 'model_export':
                request = control.build_model_export_request(pipeline_output=required(pipeline_path, 'pipeline output'))
            elif operation == 'pipeline_excel':
                request = control.build_pipeline_excel_request(pipeline_output=required(pipeline_path, 'pipeline output'),
                                                               replace='replace' in flags)
            elif operation == 'report_excel':
                request = control.build_report_excel_request(report_output=required(report_path, 'report output'),
                                                             replace='replace' in flags)
            elif operation in {'specialized_pipeline_validate', 'specialized_pipeline_run', 'specialized_pipeline_complete'}:
                pipeline_operation = {
                    'specialized_pipeline_validate': 'validate',
                    'specialized_pipeline_run': 'run',
                    'specialized_pipeline_complete': 'complete'
                }[operation]
                special_flags = set(specialized_flags or ())
                request = control.build_specialized_pipeline_request(
                    operation=pipeline_operation,
                    plan_path=required(specialized_plan, 'specialized plan') if pipeline_operation != 'complete' else None,
                    study_directory=required(specialized_study, 'specialized completion study')
                    if pipeline_operation == 'complete' else specialized_study or None,
                    run_name=specialized_run_name or None,
                    resume=specialized_resume or None,
                    source_root=source_root or '.',
                    upstream_study=specialized_upstream or None,
                    device=device,
                    jobs=job_count,
                    include_denoiser='no_denoiser' not in special_flags,
                    dry_run='dry' in special_flags,
                    environment_policy=environment_policy,
                    environment_lock=environment_lock)
            elif operation in {'specialized_validate', 'specialized_run'}:
                request = control.build_specialized_request(
                    operation='specialized-validate' if operation == 'specialized_validate' else 'specialized-run',
                    plan_path=required(specialized_plan, 'specialized plan'),
                    source_root=source_root or '.',
                    output_name=specialized_output or None,
                    study_directory=specialized_study or None,
                    case_id=specialized_case or None,
                    prediction_file=prediction_file or None,
                    step=step)
            elif operation == 'specialized_report':
                request = control.build_specialized_request(operation='specialized-report',
                                                            report_input=required(specialized_report_input,
                                                                                  'specialized report input'),
                                                            output_name=specialized_output or None)
            else:
                raise ValueError(f'unsupported Dashboard tool: {operation}')
            return (request.to_dict(), request.display, request.resolved_yaml,
                    'Ready; review the exact CLI and request YAML, then Run tool.')
        except Exception as error:
            return (None, '', '', _error(error))

    @app.callback(Output('active-tool-job', 'data'),
                  Output('tool-status', 'children', allow_duplicate=True),
                  Input('run-tool', 'n_clicks'),
                  Input('stop-tool', 'n_clicks'),
                  State('tool-request', 'data'),
                  State('active-tool-job', 'data'),
                  prevent_initial_call=True)
    def control_tool(_: int, __: int, raw_request: Mapping[str, Any] | None, job_id: str | None) -> tuple[Any, str]:
        trigger = callback_context.triggered_id
        try:
            if trigger == 'stop-tool':
                if not job_id:
                    return (None, 'No active tool job.')
                jobs.terminate(job_id)
                return (job_id, f'Stop requested for {job_id}.')
            if not raw_request:
                raise ValueError('Build CLI before running a tool')
            if job_id:
                try:
                    if jobs.status(job_id)['state'] == 'running':
                        raise RuntimeError('stop the active tool job first')
                except KeyError:
                    pass
            request = CommandRequest(script=str(raw_request.get('script', '')),
                                     arguments=tuple((str(value) for value in raw_request.get('arguments', ()))),
                                     display=str(raw_request.get('display', '')),
                                     resolved_yaml=str(raw_request.get('resolved_yaml', '')),
                                     config_sha256=str(raw_request.get('config_sha256', '')))
            if request.script == 'export_model_config.py':
                if len(request.arguments) != 2 or request.arguments[0] != '--pipeline-output':
                    raise ValueError('invalid model export request')
                result = control.execute_model_export(pipeline_output=request.arguments[1])
                return (None, _as_json(result))
            allowed = _TOOL_SUBCOMMANDS.get(request.script, frozenset())
            if not request.arguments or request.arguments[0] not in allowed:
                raise ValueError('request is outside the Dashboard maintenance allowlist')
            new_id = jobs.start_request(request, kind='tool')
            return (new_id, f'Tool job {new_id} started.')
        except Exception as error:
            return (job_id, _error(error))

    @app.callback(Output('tool-status', 'children'),
                  Input('job-poll', 'n_intervals'),
                  State('active-tool-job', 'data'),
                  prevent_initial_call=True)
    def poll_tool(_: int, job_id: str | None) -> Any:
        if not job_id:
            return no_update
        try:
            payload = jobs.status(job_id)
        except Exception as error:
            return _error(error)
        output = '\n'.join(payload.get('log_tail', [])[-12:])
        if payload.get('state') != 'running' and payload.get('log_path'):
            try:
                log_path = control.safe_pipeline_input(str(payload['log_path']), label='tool job log')
                output = log_path.read_text(encoding='utf-8', errors='replace')
                if len(output) > 200000:
                    output = '[output truncated to final 200000 characters]\n' + output[-200000:]
            except Exception as error:
                output = _error(error)
        return f"{payload['state']} · {payload['elapsed_s']:.1f}s\n" + output

    @app.callback(Output('download-tool-cli', 'data'),
                  Input('save-tool-cli', 'n_clicks'),
                  State('tool-request', 'data'),
                  prevent_initial_call=True)
    def download_tool_cli(_: int, request: Mapping[str, Any] | None) -> Any:
        if not request:
            return no_update
        return dcc.send_string(str(request.get('display', '')) + '\n', 'v5_tool_command.sh')

    @app.callback(Output('download-tool-yaml', 'data'),
                  Input('save-tool-yaml', 'n_clicks'),
                  State('tool-request', 'data'),
                  prevent_initial_call=True)
    def download_tool_yaml(_: int, request: Mapping[str, Any] | None) -> Any:
        if not request:
            return no_update
        return dcc.send_string(str(request.get('resolved_yaml', '')), 'v5_tool_request.yaml')

    @app.callback(Output('analysis-run', 'options'),
                  Output('report-output', 'options'),
                  Output('preview-artifact-run', 'options'),
                  Input('refresh-outputs', 'n_clicks'),
                  prevent_initial_call=True)
    def refresh_outputs(_: int) -> tuple[Any, Any, Any]:
        runs = _options(list(control.study_outputs()))
        return (runs, _options(list(control.report_outputs())), runs)

    @app.callback(Output('pipeline-table-select', 'options'), Output('pipeline-table-select', 'value'), Input('analysis-run', 'value'))
    def pipeline_table_options(run: Sequence[str] | None) -> tuple[Any, Any]:
        if not run:
            return ([], None)
        selected_run = str(run[0])
        try:
            values = list(preview.study_table_paths(selected_run))
        except Exception:
            return ([], None)
        preferred = next((value for value in values if value.endswith('v5_fold_predictions.csv')), values[0] if values else None)
        return (_options(values), preferred)

    @app.callback(Output('pipeline-table', 'data'), Output('pipeline-table', 'columns'), Input('pipeline-table-select', 'value'),
                  State('analysis-run', 'value'))
    def pipeline_table(path: str | None, run: Sequence[str] | None) -> tuple[Any, Any]:
        if not path or not run:
            return ([], [])
        try:
            data, columns = preview.study_table(str(run[0]), path)
            return (data, [{'name': value, 'id': value} for value in columns])
        except Exception as error:
            return ([{'error': _error(error)}], [{'name': 'error', 'id': 'error'}])

    @app.callback(Output('report-table-select', 'options'), Output('report-table-select', 'value'),
                  Output('report-figure-select', 'options'), Output('report-figure-select', 'value'), Input('report-output', 'value'))
    def report_artifact_options(report: str | None) -> tuple[Any, Any, Any, Any]:
        if not report:
            return ([], None, [], None)
        try:
            tables = list(preview.study_table_paths(report))
            figures = list(preview.study_figure_paths(report))
        except Exception:
            return ([], None, [], None)
        return (_options(tables), tables[0] if tables else None, _options(figures), figures[0] if figures else None)

    @app.callback(Output('report-table', 'data'), Output('report-table', 'columns'), Input('report-table-select', 'value'),
                  State('report-output', 'value'))
    def report_table(path: str | None, report: str | None) -> tuple[Any, Any]:
        if not path or not report:
            return ([], [])
        try:
            data, columns = preview.study_table(report, path)
            return (data, [{'name': value, 'id': value} for value in columns])
        except Exception as error:
            return ([{'error': _error(error)}], [{'name': 'error', 'id': 'error'}])

    @app.callback(Output('report-figure', 'children'), Input('report-figure-select', 'value'), State('report-output', 'value'))
    def report_figure(path: str | None, report: str | None) -> Any:
        if not path or not report:
            return None
        try:
            return html.Img(src=preview.study_figure_data_uri(report, path),
                            style={
                                'width': '100%',
                                'maxHeight': '480px',
                                'objectFit': 'contain'
                            })
        except Exception as error:
            return html.Pre(_error(error))

    return app


__all__ = ['create_app']
