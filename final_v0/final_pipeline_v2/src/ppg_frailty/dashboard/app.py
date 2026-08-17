"""Dash UI for visual inspection, study monitoring, and artifact download."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

import numpy as np

from .downloads import preview_csv_bytes, preview_metadata_bytes, study_zip_bytes
from .job_manager import StudyJobManager
from .preview_service import PipelinePreviewService


COLORS = {
    "background": "#f4f7fb",
    "panel": "#ffffff",
    "ink": "#172033",
    "muted": "#5f6b7a",
    "accent": "#2457a7",
    "border": "#d9e2ef",
    "success": "#147d64",
    "warning": "#b66a00",
}


def _panel(children: Any, *, style: dict[str, Any] | None = None) -> Any:
    from dash import html

    base = {
        "background": COLORS["panel"],
        "border": f"1px solid {COLORS['border']}",
        "borderRadius": "12px",
        "padding": "16px",
        "boxShadow": "0 4px 18px rgba(29, 51, 84, 0.06)",
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)


def _options(values: list[str] | tuple[str, ...]) -> list[dict[str, str]]:
    return [{"label": value, "value": value} for value in values]


def _control_study_job(
    jobs: StudyJobManager,
    *,
    trigger: str | None,
    study_plan: str | None,
    jobs_value: int | None,
    resume_directory: str | None,
    arguments: str | None,
    job_id: str | None,
) -> tuple[Any, bool, str]:
    """Apply one explicit Start/Stop action without ambiguous fall-through."""

    try:
        if trigger == "stop-job-button":
            if not job_id:
                return None, True, "No active study job to stop."
            jobs.terminate(job_id)
            return job_id, True, f"Stopped study job {job_id}."
        if trigger != "start-job-button":
            return job_id, True, "No study job action requested."
        if job_id:
            try:
                existing = jobs.status(job_id)
            except KeyError:
                existing = None
            if existing is not None and existing.get("state") == "running":
                return (
                    job_id,
                    False,
                    f"Study job {job_id} is already running; stop it before starting another.",
                )
        if (arguments or "").strip():
            parsed = shlex.split(arguments or "")
        else:
            if not study_plan:
                raise ValueError("select a study plan or enter advanced arguments")
            parsed = [
                "run",
                "--plan",
                str(study_plan),
                "--jobs",
                str(int(jobs_value or 1)),
            ]
            if (resume_directory or "").strip():
                parsed.extend(["--resume", str(resume_directory).strip()])
        new_job = jobs.start(parsed)
        return new_job, False, f"Started study job {new_job}."
    except Exception as exc:
        return job_id, True, f"{type(exc).__name__}: {exc}"


def create_app(
    pipeline_root: str | Path | None = None,
    *,
    preview_service: PipelinePreviewService | None = None,
    job_manager: StudyJobManager | None = None,
) -> Any:
    """Create, but do not start, the local inspection application."""

    from dash import Dash, Input, Output, State, callback_context, dash_table, dcc, html, no_update
    import plotly.graph_objects as go

    service = preview_service or PipelinePreviewService(pipeline_root)
    jobs = job_manager or StudyJobManager(service.pipeline_root)
    records = service.records()
    participants = sorted({item.participant_id for item in records})
    roles = sorted({item.role for item in records})
    configs = [path.relative_to(service.pipeline_root).as_posix() for path in service.config_paths()]
    studies = [path.relative_to(service.pipeline_root).as_posix() for path in service.study_directories()]
    study_plans = list(service.study_plan_paths())
    default_study_plan = next(
        (
            value for value in study_plans
            if value.endswith("single_config_v2.yaml")
        ),
        study_plans[0] if study_plans else None,
    )

    app = Dash(__name__, title="PPG Frailty V2 Pipeline Inspector", suppress_callback_exceptions=True)
    app.layout = html.Div(
        [
            dcc.Store(id="preview-store"),
            dcc.Store(id="active-job-store"),
            dcc.Download(id="preview-download"),
            dcc.Download(id="metadata-download"),
            dcc.Download(id="study-download"),
            html.Div(
                [
                    html.Div(
                        [
                            html.H1("PPG Frailty V2 Pipeline Inspector", style={"margin": 0, "fontSize": "28px"}),
                            html.Div(
                                "Inspect canonical stage outputs, compare parallel lines, monitor studies, and download evidence.",
                                style={"color": COLORS["muted"], "marginTop": "6px"},
                            ),
                        ]
                    ),
                    html.Div(
                        "LOCAL · READABLE · CONFIG-DRIVEN",
                        style={
                            "color": COLORS["accent"],
                            "fontWeight": 700,
                            "fontSize": "12px",
                            "letterSpacing": "0.08em",
                        },
                    ),
                ],
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "18px"},
            ),
            dcc.Tabs(
                id="main-tabs",
                value="inspect",
                children=[
                    dcc.Tab(label="Pipeline inspection", value="inspect"),
                    dcc.Tab(label="Completed studies", value="studies"),
                    dcc.Tab(label="Run study", value="run"),
                ],
            ),
            html.Div(id="tab-content", style={"marginTop": "16px"}),
            dcc.Interval(id="job-poll", interval=1500, n_intervals=0, disabled=True),
        ],
        style={
            "background": COLORS["background"],
            "color": COLORS["ink"],
            "fontFamily": "Inter, Segoe UI, sans-serif",
            "minHeight": "100vh",
            "padding": "22px 28px 42px",
        },
    )

    def inspection_layout() -> Any:
        initial_participant = participants[0] if participants else None
        participant_records = [item for item in records if item.participant_id == initial_participant]
        return html.Div(
            [
                html.Div(
                    [
                        _panel(
                            [
                                html.H3("Data and configuration", style={"marginTop": 0}),
                                html.Label("Pipeline config"),
                                dcc.Dropdown(id="config-select", options=_options(configs), value=configs[0] if configs else None),
                                html.Label("Participant", style={"marginTop": "12px", "display": "block"}),
                                dcc.Dropdown(id="participant-select", options=_options(participants), value=initial_participant),
                                html.Label("Role", style={"marginTop": "12px", "display": "block"}),
                                dcc.Dropdown(id="role-select", options=_options(roles), value=None, placeholder="All roles"),
                                html.Label("Record", style={"marginTop": "12px", "display": "block"}),
                                dcc.Dropdown(
                                    id="record-select",
                                    options=[
                                        {
                                            "label": f"{item.role} · {item.record_id} · {item.duration_s:.1f}s",
                                            "value": item.record_id,
                                        }
                                        for item in participant_records
                                    ],
                                    value=participant_records[0].record_id if participant_records else None,
                                ),
                                html.Div(
                                    [
                                        html.Div([html.Label("Start (s)"), dcc.Input(id="start-seconds", type="number", value=0, min=0, step=1, style={"width": "100%"})]),
                                        html.Div([html.Label("Duration (s)"), dcc.Input(id="duration-seconds", type="number", value=20, min=1, max=120, step=1, style={"width": "100%"})]),
                                    ],
                                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px", "marginTop": "12px"},
                                ),
                                html.Label("Signal traces", style={"marginTop": "12px", "display": "block"}),
                                dcc.Checklist(
                                    id="trace-select",
                                    options=[
                                        {"label": value.replace("_", " "), "value": value}
                                        for value in service.DEFAULT_TRACES
                                    ],
                                    value=list(service.DEFAULT_TRACES),
                                    inputStyle={"marginRight": "6px"},
                                    labelStyle={"display": "block", "marginBottom": "5px"},
                                ),
                                html.Label(
                                    "Pipeline stage outputs",
                                    style={"marginTop": "12px", "display": "block"},
                                ),
                                dcc.Checklist(
                                    id="stage-select",
                                    options=[
                                        {"label": value.replace("_", " "), "value": value}
                                        for value in (
                                            "source",
                                            "preprocess",
                                            "quality_route",
                                            "pulse_ppi",
                                            "morphology",
                                            "engineering",
                                            "representation_model",
                                            "aggregation",
                                        )
                                    ],
                                    value=[
                                        "source",
                                        "preprocess",
                                        "quality_route",
                                        "pulse_ppi",
                                        "morphology",
                                        "engineering",
                                        "representation_model",
                                        "aggregation",
                                    ],
                                    inputStyle={"marginRight": "6px"},
                                    labelStyle={"display": "block", "marginBottom": "5px"},
                                ),
                                html.Button(
                                    "Build preview",
                                    id="preview-button",
                                    n_clicks=0,
                                    style={
                                        "width": "100%",
                                        "marginTop": "14px",
                                        "padding": "10px",
                                        "border": 0,
                                        "borderRadius": "8px",
                                        "background": COLORS["accent"],
                                        "color": "white",
                                        "fontWeight": 700,
                                    },
                                ),
                                html.Div(id="preview-status", style={"marginTop": "10px", "fontSize": "13px", "color": COLORS["muted"]}),
                            ]
                        )
                    ],
                    style={"minWidth": "285px"},
                ),
                html.Div(
                    [
                        _panel(
                            [
                                html.Div(
                                    [
                                        html.H3("Stage outputs", style={"margin": 0}),
                                        html.Div(
                                            [
                                                html.Button("Download CSV", id="preview-download-button", n_clicks=0),
                                                html.Button("Download metadata", id="metadata-download-button", n_clicks=0),
                                            ],
                                            style={"display": "flex", "gap": "8px"},
                                        ),
                                    ],
                                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                                ),
                                dcc.Graph(id="time-graph", figure=go.Figure(), config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}),
                                dcc.Graph(id="spectrum-graph", figure=go.Figure(), config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}),
                            ]
                        ),
                        _panel(
                            [
                                html.H3("Stage statistics", style={"marginTop": 0}),
                                dash_table.DataTable(
                                    id="statistics-table",
                                    page_size=12,
                                    style_table={"overflowX": "auto"},
                                    style_cell={"fontFamily": "monospace", "fontSize": 12, "padding": "7px"},
                                    style_header={"fontWeight": 700, "background": "#eef3fa"},
                                ),
                                html.H3("Resolved preview metadata", style={"marginTop": "18px"}),
                                html.Pre(id="metadata-view", style={"whiteSpace": "pre-wrap", "fontSize": "12px", "background": "#f7f9fc", "padding": "12px", "borderRadius": "8px"}),
                                html.H3(
                                    "Module stage outputs (full-record scope)",
                                    style={"marginTop": "18px"},
                                ),
                                dash_table.DataTable(
                                    id="module-stage-table",
                                    page_size=20,
                                    filter_action="native",
                                    sort_action="native",
                                    style_table={"overflowX": "auto"},
                                    style_cell={"fontFamily": "monospace", "fontSize": 11, "padding": "6px", "whiteSpace": "normal", "height": "auto"},
                                    style_header={"fontWeight": 700, "background": "#eef3fa"},
                                ),
                            ],
                            style={"marginTop": "14px"},
                        ),
                    ]
                ),
            ],
            style={"display": "grid", "gridTemplateColumns": "310px minmax(0, 1fr)", "gap": "16px"},
        )

    def studies_layout() -> Any:
        initial = studies[0] if studies else None
        tables = list(service.study_table_paths(initial)) if initial else []
        initial_table = (
            "tables/predictive_leaderboard.csv"
            if "tables/predictive_leaderboard.csv" in tables
            else (tables[0] if tables else None)
        )
        study_figures = list(service.study_figure_paths(initial)) if initial else []
        return html.Div(
            [
                _panel(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Study directory"),
                                        dcc.Dropdown(id="study-select", options=_options(studies), value=initial),
                                    ],
                                    style={"flex": 2},
                                ),
                                html.Div(
                                    [
                                        html.Label("Table"),
                                        dcc.Dropdown(id="study-table-select", options=_options(tables), value=initial_table),
                                    ],
                                    style={"flex": 3},
                                ),
                                html.Button("Download full study ZIP", id="study-download-button", n_clicks=0, style={"alignSelf": "end", "height": "38px"}),
                            ],
                            style={"display": "flex", "gap": "12px", "alignItems": "end"},
                        ),
                        html.Div(id="study-status", style={"marginTop": "10px", "color": COLORS["muted"]}),
                    ]
                ),
                _panel(
                    [
                        dash_table.DataTable(
                            id="study-table",
                            page_size=20,
                            filter_action="native",
                            sort_action="native",
                            style_table={"overflowX": "auto"},
                            style_cell={"fontFamily": "monospace", "fontSize": 11, "padding": "6px", "maxWidth": "260px", "overflow": "hidden", "textOverflow": "ellipsis"},
                            style_header={"fontWeight": 700, "background": "#eef3fa"},
                        )
                    ],
                    style={"marginTop": "14px"},
                ),
                _panel(
                    [
                        html.H3("Paper figures and visual comparison", style={"marginTop": 0}),
                        dcc.Dropdown(
                            id="study-figure-select",
                            options=_options(study_figures),
                            value=study_figures[:4],
                            multi=True,
                            placeholder="Select completed-study figures",
                        ),
                        html.Div(
                            id="study-figure-gallery",
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(auto-fit, minmax(360px, 1fr))",
                                "gap": "14px",
                                "marginTop": "14px",
                            },
                        ),
                    ],
                    style={"marginTop": "14px"},
                ),
            ]
        )

    def run_layout() -> Any:
        return html.Div(
            [
                _panel(
                    [
                        html.H3("Launch dedicated study CLI", style={"marginTop": 0}),
                        html.P(
                            "The Dash callback only starts the external CLI. Training and reporting run in that independent process.",
                            style={"color": COLORS["muted"]},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Study plan"),
                                        dcc.Dropdown(
                                            id="study-plan-select",
                                            options=_options(study_plans),
                                            value=default_study_plan,
                                            placeholder="Select a reusable study plan",
                                        ),
                                    ],
                                    style={"flex": 4},
                                ),
                                html.Div(
                                    [
                                        html.Label("Parallel cases"),
                                        dcc.Input(
                                            id="study-job-count",
                                            type="number",
                                            value=1,
                                            min=1,
                                            step=1,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    style={"flex": 1},
                                ),
                            ],
                            style={"display": "flex", "gap": "12px"},
                        ),
                        html.Label(
                            "Resume existing study directory (optional)",
                            style={"marginTop": "12px", "display": "block"},
                        ),
                        dcc.Input(
                            id="study-resume-directory",
                            type="text",
                            value="",
                            placeholder="/path/to/existing/study",
                            style={"width": "100%"},
                        ),
                        html.Label(
                            "Advanced command arguments (optional; overrides the menu)",
                            style={"marginTop": "12px", "display": "block"},
                        ),
                        dcc.Textarea(
                            id="study-arguments",
                            value="",
                            placeholder=(
                                "Example: ablation --base-config ... "
                                "--factor training.fixed_epochs --values 7 10 15 ..."
                            ),
                            style={"width": "100%", "height": "92px", "fontFamily": "monospace"},
                        ),
                        html.Div(
                            [
                                html.Button("Start", id="start-job-button", n_clicks=0),
                                html.Button("Stop", id="stop-job-button", n_clicks=0),
                            ],
                            style={"display": "flex", "gap": "8px", "marginTop": "10px"},
                        ),
                        html.Pre(id="job-status", style={"marginTop": "14px", "whiteSpace": "pre-wrap", "background": "#111827", "color": "#d1fae5", "padding": "14px", "borderRadius": "8px", "minHeight": "180px"}),
                    ]
                )
            ]
        )

    @app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
    def render_tab(value: str) -> Any:
        if value == "studies":
            return studies_layout()
        if value == "run":
            return run_layout()
        return inspection_layout()

    @app.callback(
        Output("record-select", "options"),
        Output("record-select", "value"),
        Input("participant-select", "value"),
        Input("role-select", "value"),
        prevent_initial_call=True,
    )
    def update_records(participant: str | None, role: str | None) -> tuple[Any, Any]:
        selected = [
            item for item in records
            if (participant is None or item.participant_id == participant)
            and (role is None or item.role == role)
        ]
        options = [
            {"label": f"{item.role} · {item.record_id} · {item.duration_s:.1f}s", "value": item.record_id}
            for item in selected
        ]
        return options, (selected[0].record_id if selected else None)

    @app.callback(
        Output("preview-store", "data"),
        Output("time-graph", "figure"),
        Output("spectrum-graph", "figure"),
        Output("statistics-table", "data"),
        Output("statistics-table", "columns"),
        Output("module-stage-table", "data"),
        Output("module-stage-table", "columns"),
        Output("metadata-view", "children"),
        Output("preview-status", "children"),
        Input("preview-button", "n_clicks"),
        State("config-select", "value"),
        State("record-select", "value"),
        State("start-seconds", "value"),
        State("duration-seconds", "value"),
        State("trace-select", "value"),
        State("stage-select", "value"),
        prevent_initial_call=True,
    )
    def build_preview(
        _: int,
        config_path: str | None,
        record_id: str | None,
        start_s: float | None,
        duration_s: float | None,
        trace_names: list[str] | None,
        stage_names: list[str] | None,
    ) -> tuple[Any, ...]:
        if not config_path or not record_id:
            return (
                no_update,
                go.Figure(),
                go.Figure(),
                [],
                [],
                [],
                [],
                "",
                "Select a config and record.",
            )
        try:
            result = service.preview(
                config_path=config_path,
                record_id=record_id,
                start_s=float(start_s or 0.0),
                duration_s=float(duration_s or 20.0),
                trace_names=trace_names,
                stage_names=stage_names,
            )
        except Exception as exc:  # UI boundary: show the exact algorithm error.
            return (
                no_update,
                go.Figure(),
                go.Figure(),
                [],
                [],
                [],
                [],
                "",
                f"{type(exc).__name__}: {exc}",
            )
        time_figure = go.Figure()
        for name, values in result.traces.items():
            time_figure.add_trace(go.Scattergl(x=result.time_s, y=values, mode="lines", name=name))
        time_figure.update_layout(template="plotly_white", title="Synchronized stage traces", xaxis_title="Time (s)", yaxis_title="Value", legend={"orientation": "h"})
        spectrum_figure = go.Figure()
        for name, (frequency, power) in result.spectra.items():
            spectrum_figure.add_trace(go.Scatter(x=frequency, y=power, mode="lines", name=name))
        spectrum_figure.update_layout(template="plotly_white", title="Welch preview spectra", xaxis_title="Frequency (Hz)", yaxis_title="PSD", yaxis_type="log", legend={"orientation": "h"})
        statistics = [dict(value) for value in result.statistics]
        columns = [{"name": key, "id": key} for key in statistics[0]] if statistics else []
        stage_rows = [dict(value) for value in result.stage_rows]
        stage_columns = (
            [{"name": key, "id": key} for key in stage_rows[0]]
            if stage_rows
            else []
        )
        metadata_payload = {
            "record_id": result.record_id,
            "participant_id": result.participant_id,
            "role": result.role,
            "class_name": result.class_name,
            "fs_hz": result.fs_hz,
            "start_s": result.start_s,
            "duration_s": result.duration_s,
            "trace_names": list(result.traces),
            "stage_metadata": dict(result.stage_metadata),
            "stage_outputs": stage_rows,
        }
        stored = {
            "record_id": result.record_id,
            "time_s": result.time_s.tolist(),
            "traces": {key: value.tolist() for key, value in result.traces.items()},
            "metadata": metadata_payload,
        }
        return (
            stored,
            time_figure,
            spectrum_figure,
            statistics,
            columns,
            stage_rows,
            stage_columns,
            json.dumps(metadata_payload, ensure_ascii=False, indent=2, sort_keys=True),
            f"Preview ready: {result.record_id}, {result.duration_s:.1f}s, no training executed.",
        )

    @app.callback(
        Output("preview-download", "data"),
        Input("preview-download-button", "n_clicks"),
        State("preview-store", "data"),
        prevent_initial_call=True,
    )
    def download_preview(_: int, stored: dict[str, Any] | None) -> Any:
        if not stored:
            return no_update
        return dcc.send_bytes(
            preview_csv_bytes(stored["time_s"], stored["traces"]),
            f"{stored['record_id'].replace(':', '_')}_pipeline_preview.csv",
        )

    @app.callback(
        Output("metadata-download", "data"),
        Input("metadata-download-button", "n_clicks"),
        State("preview-store", "data"),
        prevent_initial_call=True,
    )
    def download_metadata(_: int, stored: dict[str, Any] | None) -> Any:
        if not stored:
            return no_update
        return dcc.send_bytes(
            preview_metadata_bytes(stored["metadata"]),
            f"{stored['record_id'].replace(':', '_')}_pipeline_preview_metadata.json",
        )

    @app.callback(
        Output("study-table-select", "options"),
        Output("study-table-select", "value"),
        Output("study-figure-select", "options"),
        Output("study-figure-select", "value"),
        Input("study-select", "value"),
        prevent_initial_call=True,
    )
    def update_study_tables(study_dir: str | None) -> tuple[Any, Any, Any, Any]:
        if not study_dir:
            return [], None, [], []
        values = list(service.study_table_paths(study_dir))
        preferred = (
            "tables/predictive_leaderboard.csv"
            if "tables/predictive_leaderboard.csv" in values
            else (values[0] if values else None)
        )
        figures = list(service.study_figure_paths(study_dir))
        return _options(values), preferred, _options(figures), figures[:4]

    @app.callback(
        Output("study-figure-gallery", "children"),
        Input("study-figure-select", "value"),
        State("study-select", "value"),
        prevent_initial_call=True,
    )
    def load_study_figures(
        relative_paths: list[str] | None,
        study_dir: str | None,
    ) -> Any:
        if not relative_paths or not study_dir:
            return html.Em("Select one or more generated figures.")
        children = []
        for relative_path in relative_paths:
            try:
                source = service.study_figure_data_uri(study_dir, relative_path)
            except Exception as exc:
                children.append(
                    html.Div(f"{relative_path}: {type(exc).__name__}: {exc}")
                )
                continue
            children.append(
                html.Div(
                    [
                        html.Div(
                            relative_path,
                            style={"fontWeight": 700, "marginBottom": "6px"},
                        ),
                        html.Img(
                            src=source,
                            style={"width": "100%", "height": "auto"},
                        ),
                    ],
                    style={
                        "border": f"1px solid {COLORS['border']}",
                        "borderRadius": "8px",
                        "padding": "10px",
                    },
                )
            )
        return children

    @app.callback(
        Output("study-table", "data"),
        Output("study-table", "columns"),
        Output("study-status", "children"),
        Input("study-table-select", "value"),
        State("study-select", "value"),
        prevent_initial_call=True,
    )
    def load_study_table(relative_path: str | None, study_dir: str | None) -> tuple[Any, Any, str]:
        if not relative_path or not study_dir:
            return [], [], "Select a completed study table."
        try:
            data, columns = service.study_table(study_dir, relative_path)
        except Exception as exc:
            return [], [], f"{type(exc).__name__}: {exc}"
        return data, [{"name": value, "id": value} for value in columns], f"Loaded first {len(data)} rows from {relative_path}."

    @app.callback(
        Output("study-download", "data"),
        Input("study-download-button", "n_clicks"),
        State("study-select", "value"),
        prevent_initial_call=True,
    )
    def download_study(_: int, study_dir: str | None) -> Any:
        if not study_dir:
            return no_update
        target = (service.pipeline_root / study_dir).resolve()
        data = study_zip_bytes(target, studies_root=service.pipeline_root / "artifacts" / "studies")
        return dcc.send_bytes(data, f"{target.name}.zip")

    @app.callback(
        Output("active-job-store", "data"),
        Output("job-poll", "disabled"),
        Output("job-status", "children"),
        Input("start-job-button", "n_clicks"),
        Input("stop-job-button", "n_clicks"),
        State("study-plan-select", "value"),
        State("study-job-count", "value"),
        State("study-resume-directory", "value"),
        State("study-arguments", "value"),
        State("active-job-store", "data"),
        prevent_initial_call=True,
    )
    def control_job(
        _: int,
        __: int,
        study_plan: str | None,
        jobs_value: int | None,
        resume_directory: str | None,
        arguments: str | None,
        job_id: str | None,
    ) -> tuple[Any, bool, str]:
        return _control_study_job(
            jobs,
            trigger=callback_context.triggered_id,
            study_plan=study_plan,
            jobs_value=jobs_value,
            resume_directory=resume_directory,
            arguments=arguments,
            job_id=job_id,
        )

    @app.callback(
        Output("job-status", "children", allow_duplicate=True),
        Output("job-poll", "disabled", allow_duplicate=True),
        Input("job-poll", "n_intervals"),
        State("active-job-store", "data"),
        prevent_initial_call=True,
    )
    def poll_job(_: int, job_id: str | None) -> tuple[str, bool]:
        if not job_id:
            return "No active study job.", True
        try:
            payload = jobs.status(job_id)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}", True
        text = json.dumps(
            {key: value for key, value in payload.items() if key != "log_tail"},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        if payload["log_tail"]:
            text += "\n\n--- log tail ---\n" + "\n".join(payload["log_tail"])
        return text, payload["state"] != "running"

    return app
