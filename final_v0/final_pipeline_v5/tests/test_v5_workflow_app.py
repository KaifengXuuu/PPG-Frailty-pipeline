"""Exercise notebook callbacks through Dash's HTTP protocol without training."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import importlib.util
import json
from pathlib import Path
import shlex

import pytest
import yaml

pytest.importorskip("dash")
from ppg_frailty.dashboard import create_app
from ppg_frailty.dashboard.control_service import CommandRequest, V5ControlService

ROOT = Path(__file__).resolve().parents[1]
EXECUTION = ("", "all", "all", 1, "off", "cache/preprocessing", [], "")


def callback(app, output):
    return next(v["callback"].__wrapped__ for k, v in app.callback_map.items() if output in k)


@contextmanager
def triggered(identity, prop="n_clicks"):
    from dash._callback_context import context_value
    from dash._utils import AttributeDict
    value = json.dumps(identity, sort_keys=True, separators=(",", ":")) if isinstance(identity, dict) else identity
    token = context_value.set(AttributeDict(triggered_inputs=[{"prop_id": value + "." + prop, "value": 1}]))
    try:
        yield
    finally:
        context_value.reset(token)


def post_callback(app, output, values, changed):
    """Build the same callback envelope sent by Dash's browser renderer."""
    key, definition = next((k, v) for k, v in app.callback_map.items() if output in k)
    outputs = definition["output"]
    def fields(group):
        rows = []
        for item in definition[group]:
            identity = item["id"]
            lookup = json.loads(identity)["type"] if identity.startswith("{") else identity
            rows.append({**item, "value": values.get((lookup, item["property"]))})
        return rows
    encoded = [{"id": out.component_id, "property": out.component_property} for out in outputs]
    response = app.server.test_client().post("/_dash-update-component", json={
        "output": key, "outputs": encoded, "inputs": fields("inputs"), "state": fields("state"),
        "changedPropIds": [changed],
    })
    assert response.status_code == 200, response.get_data(as_text=True)
    return response.get_json()["response"]


class FakeJobs:
    def __init__(self):
        self.started, self.stopped = [], []
    def start_request(self, request, *, kind):
        self.started.append((request, kind))
        return "job-1"
    def terminate(self, job):
        self.stopped.append(job)
    def status(self, job):
        return {"state": "running", "elapsed_s": 3, "log_tail": []}


def test_http_analyse_uses_live_controls_and_downloaded_cli_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observed = {}
    class Workflow:
        def analyse(self, **request):
            observed.update(request)
            return {"requested_stage": request["stage"], "computed_stages": ["input", "ppg"],
                    "reused_stages": [], "previews": {"ppg": {"metadata": {"samples": 40}}}}

    jobs = FakeJobs()
    app = create_app(ROOT, workflow_service=Workflow(), job_manager=jobs)
    config, _ = V5ControlService(ROOT).load_yaml("configs/presets/finalcase.yaml")
    numbers = [{"type": "param-number", "path": "signal.ppg_filter.high_hz"}]
    files = [{"file_id": "B static", "role": "B", "path": "data with spaces/static.csv", "label": "Young"}]
    selected = ("p-new", "data/b.csv", None, "models/motion.json", None, "models/sqi.json",
                "model_config/example", "case", None, "models/direct bundle")
    selection_names = ("participant-id", "calibration-path", "motion-bundle", "motion-bundle-path", "sqi-artifact",
                       "sqi-artifact-path", "model-export", "model-case", "model-bundle", "model-bundle-path")
    values = {
        ("analyse-stage", "n_clicks"): [0, 1], ("config-state", "data"): config,
        ("param", "id"): [], ("param", "value"): [],
        ("param-number", "id"): numbers, ("param-number", "value"): [7.25],
        ("preview-record", "value"): "B static", ("session-id", "data"): "session-test",
        ("preview-start", "value"): 1.5, ("preview-duration", "value"): 8,
        ("input-files", "data"): files, ("record-ids", "value"): [],
        **{(name, "value"): value for name, value in zip(selection_names, selected)},
    }
    response = post_callback(app, "preview-store.data", values,
                             '{"stage":"ppg","type":"analyse-stage"}.n_clicks')
    result = response["preview-store"]["data"]
    assert observed["config_payload"]["signal"]["ppg_filter"]["high_hz"] == 7.25
    assert config["signal"]["ppg_filter"]["high_hz"] == 8
    assert observed["record_id"] == "B static" and observed["session_id"] == "session-test"
    assert result["computed_stages"] == ["input", "ppg"] and jobs.started == []
    command = callback(app, "command-view.children")(
        result["configuration"], "ppg", "B static", 1.5, 8, files, [], *selected)
    download = callback(app, "download-yaml.data")(1, result["configuration"])
    assert yaml.safe_load(download["content"]) == observed["config_payload"]
    assert callback(app, "download-cli.data")(1, command)["content"].strip() == command
    module_spec = importlib.util.spec_from_file_location("stage_cli_test", ROOT / "stage_analyse.py")
    cli = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(cli)
    parsed = cli.build_parser().parse_args(shlex.split(command)[2:])
    assert parsed.stage == observed["stage"] == "ppg"
    assert parsed.record_id == observed["record_id"]
    assert parsed.start_s == observed["start_s"] and parsed.duration_s == observed["duration_s"]
    assert json.loads(parsed.selections) == observed["selections"]
    replay = {}
    class ReplayWorkflow:
        def __init__(self, root):
            pass
        def analyse(self, stage, config, record, **options):
            replay.update(stage=stage, config=config, record=record, **options)
            return {"status": "complete"}
    from ppg_frailty.dashboard import workflow_service
    monkeypatch.setattr(workflow_service, "WorkflowService", ReplayWorkflow)
    config_path = tmp_path / "workflow_config.yaml"
    config_path.write_text(download["content"])
    arguments = shlex.split(command)[2:]
    arguments[arguments.index("--config") + 1] = str(config_path)
    assert cli.main(arguments) == 0
    from ppg_frailty.config import _materialize_v2_defaults
    expected = copy.deepcopy(observed["config_payload"])
    _materialize_v2_defaults(expected)
    assert replay["config"] == expected
    assert replay["selections"] == observed["selections"]


@pytest.mark.parametrize(("mode", "yaml_path"), [("Analyse", "configs/presets/finalcase.yaml"), ("Train", None)])
def test_run_requires_train_mode_and_selected_yaml(mode, yaml_path) -> None:
    jobs = FakeJobs()
    app = create_app(ROOT, job_manager=jobs)
    request = CommandRequest("pipeline.py", ("run",), "python pipeline.py run", "{}").to_dict()
    with triggered("train"):
        job, status = callback(app, "active-train-job.data")(
            1, 0, mode, yaml_path, "current", request, [], None, None, {}, [], [], [], [], *EXECUTION)
    assert job is None and "Select a YAML and Train mode" in status
    assert jobs.started == []


def test_stop_is_effective_without_yaml_or_train_mode() -> None:
    jobs = FakeJobs()
    app = create_app(ROOT, job_manager=jobs)
    with triggered("stop-train"):
        job, status = callback(app, "active-train-job.data")(
            0, 1, "Analyse", None, "current", None, [], "active-job", None, {}, [], [], [], [], *EXECUTION)
    assert job == "active-job" and jobs.stopped == ["active-job"]
    assert jobs.started == [] and status == "Stopped."
    mode = callback(app, "train-options.style")
    assert mode("Analyse") == ({"display": "none"}, {"display": "none"}, {})
    assert mode("Train") == ({}, {}, {"display": "none"})


def test_run_submits_only_the_explicit_request_and_rejects_duplicate_job() -> None:
    jobs = FakeJobs()
    app = create_app(ROOT, job_manager=jobs)
    config, _ = V5ControlService(ROOT).load_yaml("configs/presets/finalcase.yaml")
    request, _, _ = callback(app, "train-request.data")(config, "configs/presets/finalcase.yaml", *EXECUTION)
    assert request is not None and jobs.started == []
    action = callback(app, "active-train-job.data")
    with triggered("train"):
        job, _ = action(1, 0, "Train", "configs/presets/finalcase.yaml", "current",
                        request, [], None, None, config, [], [],
                        [{"type": "param-number", "path": "training.batch_size"}], [32], *EXECUTION)
        _, status = action(2, 0, "Train", "configs/presets/finalcase.yaml", "current",
                           request, [], job, None, config, [], [], [], [], *EXECUTION)
    assert len(jobs.started) == 1
    assert yaml.safe_load(jobs.started[0][0].resolved_yaml)["training"]["batch_size"] == 32
    assert yaml.safe_load(request["resolved_yaml"])["training"]["batch_size"] != 32
    assert "already running" in status


def test_training_tool_cannot_start_from_analyse_but_uses_model_run() -> None:
    jobs = FakeJobs()
    app = create_app(ROOT, job_manager=jobs)
    request, _, _, _, disabled = callback(app, "tool-request.data")(
        "specialized_pipeline_run", "--plan configs/studies/specialized.yaml")
    assert disabled is True and request is not None
    with triggered("analyse-tool"):
        job, status = callback(app, "active-tool-job.data")(1, 0, "specialized_pipeline_run", request, None)
    assert job is None and "model Train mode" in status and jobs.started == []
    with triggered("train"):
        job, _ = callback(app, "active-train-job.data")(
            1, 0, "Train", "configs/studies/specialized.yaml", "tool", None, [], None, request,
            {}, [], [], [], [], *EXECUTION)
    assert job == "job-1" and jobs.started[0][0].script == "specialized_pipeline.py"


def test_numeric_editor_extends_slider_without_clipping_current_value() -> None:
    app = create_app(ROOT)
    sync = callback(app, '"type":"param-slide"')
    with triggered({"type": "param-number", "path": "training.learning_rate"}, "value"):
        values = sync(0.001, 3.5, 0.0, 0.01)
    assert values == (3.5, 3.5, 0.0, 3.5)
    with triggered({"type": "param-slide", "path": "training.learning_rate"}, "value"):
        assert sync(.005, 3.5, 0.0, 3.5)[:2] == (.005, .005)


def test_parameter_edit_updates_config_without_recreating_numeric_widget() -> None:
    from dash import no_update
    app = create_app(ROOT)
    config, _ = V5ControlService(ROOT).load_yaml("configs/presets/finalcase.yaml")
    numbers = [{"type": "param-number", "path": "signal.ppg_filter.high_hz"}]
    with triggered(numbers[0], "value"):
        updated, status, panels, selectors = callback(app, "config-state.data")(
            "configs/presets/finalcase.yaml", None, [], [7.0], [], numbers, config,
            [{"type": "stage-controls", "stage": "ppg"}], [])
    assert updated["signal"]["ppg_filter"]["high_hz"] == 7
    assert "Controls updated" in status and panels == [no_update]
    assert selectors == []
    assert config["signal"]["ppg_filter"]["high_hz"] == 8


def test_vector_element_edit_refreshes_parent_without_stale_parent_replay() -> None:
    from dash import no_update
    app = create_app(ROOT)
    config, _ = V5ControlService(ROOT).load_yaml("configs/presets/finalcase.yaml")
    parent = {"type": "param", "path": "signal.normalization.clip_after_scale"}
    children = [{"type": "param-number", "path": parent["path"] + f".{i}"} for i in (0, 1)]
    with triggered(children[0], "value"):
        updated, _, rendered, selectors = callback(app, "config-state.data")(
            "configs/presets/finalcase.yaml", None, ["[-8, 8]"], [-4.0, 8.0], [parent], children, config,
            [{"type": "stage-controls", "stage": "representation"}],
            [{"type": "stage-selectors", "stage": "representation"}])
    assert updated["signal"]["normalization"]["clip_after_scale"] == [-4.0, 8.0]
    assert rendered[0] is not no_update
    assert selectors[0] is not no_update
    parent_editor = next(c for block in rendered[0] for c in block._traverse()
                         if getattr(c, "id", None) == parent)
    assert yaml.safe_load(parent_editor.value) == [-4.0, 8.0]
    assert config["signal"]["normalization"]["clip_after_scale"] == [-8.0, 8.0]


def test_coupled_control_change_ignores_stale_other_widget() -> None:
    from dash import no_update
    app = create_app(ROOT)
    config, _ = V5ControlService(ROOT).load_yaml("configs/presets/finalcase.yaml")
    controls = [{"type": "param", "path": path} for path in
                ("aggregation.balance_line", "training.training_balance")]
    with triggered(controls[0], "value"):
        updated, _, rendered, selectors = callback(app, "config-state.data")(
            "configs/presets/finalcase.yaml", None, ["line_a_equal_files", "equal_role_families"], [],
            controls, [], config, [{"type": "stage-controls", "stage": "model"}],
            [{"type": "stage-selectors", "stage": "model"}])
    assert updated["aggregation"]["balance_line"] == "line_a_equal_files"
    assert updated["training"]["training_balance"] == "equal_files"
    assert rendered[0] is not no_update
    assert selectors[0] is not no_update


def test_report_uses_current_statistics_and_public_cli(tmp_path: Path) -> None:
    from ppg_frailty.v5_reporting.cli import build_parser

    root = tmp_path / "repo" / "final_v0" / "final_pipeline_v5"
    run = root / "pipeline_output" / "run_a"
    run.mkdir(parents=True)
    (run / "study_manifest.json").write_text("{}")
    jobs = FakeJobs()
    app = create_app(root, job_manager=jobs)
    config, _ = V5ControlService(ROOT).load_yaml("configs/presets/finalcase.yaml")
    options = (["pipeline_output/run_a"], "single", "full", ["predictions"], [], ["file_predictions"],
               "", "", "", "", "", .1, 15)
    cached, command = callback(app, "analysis-request.data")(*options, config)
    assert cached is not None, command
    old = build_parser().parse_args(list(cached["arguments"]))
    assert old.bootstrap_resamples == 10000
    controls = [{"type": "param-number", "path": f"evaluation.statistics.{key}"} for key in
                ("bootstrap_replicates", "paired_permutation_replicates", "seed")]
    with triggered("analyse-report"):
        job, status = callback(app, "active-report-job.data")(
            1, 0, cached, None, config, [], [], controls, [17, 19, 73], *options)
    assert job == "job-1", status
    request, kind = jobs.started[0]
    parsed = build_parser().parse_args(list(request.arguments))
    assert kind == "report" and request.script == "analyse_report.py"
    assert (parsed.bootstrap_resamples, parsed.permutation_resamples, parsed.statistics_seed) == (17, 19, 73)
    assert parsed.alpha == .1 and parsed.calibration_bins == 15
    assert parsed.mode == "single" and parsed.table == ["file_predictions"]
    assert callback(app, "download-report-cli.data")(1, request.to_dict())["content"] == request.display + "\n"
    components = {c.id: c for c in app.layout()._traverse() if isinstance(getattr(c, "id", None), str)}
    assert not {"bootstrap-count", "permutation-count", "statistics-seed"} & set(components)
    with triggered("stop-report"):
        callback(app, "active-report-job.data")(0, 1, None, job, None, [], [], [], [], *options)
    assert jobs.stopped == [job]


def test_signal_preview_separates_physical_axes_and_marks_peaks() -> None:
    from ppg_frailty.dashboard.app import _render_preview

    traces = {name: {"x": [0, 1], "y": [1, 2]} for name in
              ("native_RED", "native_IR", "direct_RED", "filtered_IR", "RED_peaks",
               "acc_x", "AX", "gyro_x", "GX", "jerk_magnitude", "roll_rad")}
    preview = _render_preview({"traces": traces})
    figure = next(component.figure for component in preview if component.__class__.__name__ == "Graph")
    plotted = {trace.name: trace for trace in figure.data}
    assert plotted["native_RED"].yaxis == plotted["native_IR"].yaxis
    assert plotted["direct_RED"].yaxis == plotted["filtered_IR"].yaxis == plotted["RED_peaks"].yaxis
    assert plotted["native_RED"].yaxis != plotted["direct_RED"].yaxis
    assert plotted["acc_x"].yaxis == plotted["AX"].yaxis
    assert plotted["gyro_x"].yaxis == plotted["GX"].yaxis
    assert len({plotted[name].yaxis for name in
                ("native_RED", "direct_RED", "acc_x", "gyro_x", "jerk_magnitude", "roll_rad")}) == 6
    assert plotted["RED_peaks"].mode == "markers"
    assert all(trace.mode == "lines" for name, trace in plotted.items() if name != "RED_peaks")
    assert list(plotted["RED_peaks"].x) == traces["RED_peaks"]["x"]
    assert list(plotted["RED_peaks"].y) == traces["RED_peaks"]["y"]
    representation = _render_preview({"metadata": {"stage": "representation"},
                                      "traces": {"first_window_RED": traces["native_RED"]}})
    figure = next(component.figure for component in representation if component.__class__.__name__ == "Graph")
    assert figure.layout.annotations[0].text == "PPG windows"


@pytest.mark.parametrize("level", ["recordings", "participants"])
def test_probability_preview_preserves_bundle_class_names(level: str) -> None:
    from ppg_frailty.dashboard.app import _render_preview

    names = ["Pre-frail", "Robust non-frail", "Young"]
    preview = _render_preview({"metadata": {"class_names": names}, "tables": {
        level: [{"participant_id": "p-1", "file_id": "record-1", "probabilities": [.6, .3, .1]}]}})
    figure = next(component.figure for component in preview if component.__class__.__name__ == "Graph")
    assert list(figure.data[0].x) == names
    assert list(figure.data[0].y) == [.6, .3, .1]
    assert figure.data[0].name == ("p-1" if level == "participants" else "record-1")
    assert tuple(figure.layout.yaxis.range) == (0, 1)


def test_refresh_discovers_new_sqi_motion_and_model_assets(tmp_path: Path) -> None:
    app = create_app(tmp_path)
    refresh = callback(app, "training-yaml.options")
    assert refresh(0)[4:] == ([], [], [])
    paths = ("pipeline_output/run/sqi_calibration.json", "artifacts/component_calibration.json",
             "pipeline_output/run/motion_evidence.json", "artifacts/internal_evidence.json",
             "model_config/run/cases/model/manifest.json")
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    refreshed = refresh(1)
    assert len(refreshed) == 7
    assert {row["value"] for row in refreshed[4]} == set(paths[:2])
    assert {row["value"] for row in refreshed[5]} == set(paths[2:4])
    assert {row["value"] for row in refreshed[6]} == {str(Path(paths[4]).parent)}


def test_manifest_change_refreshes_choices_and_drops_only_missing_selections(tmp_path: Path, monkeypatch) -> None:
    from ppg_frailty.dashboard.preview_service import PipelinePreviewService, RecordChoice

    observed = []
    records = {"first.csv": (RecordChoice("shared", "p1", "B", "Young", 60),
                              RecordChoice("old", "p1", "R", "Young", 60)),
               "second.csv": (RecordChoice("new", "p2", "B", "Young", 60),
                               RecordChoice("shared", "p1", "B", "Young", 60))}
    def read_records(self, path=None):
        observed.append(path)
        if path == "missing.csv":
            raise FileNotFoundError(path)
        return records.get(path, ())
    monkeypatch.setattr(PipelinePreviewService, "records", read_records)
    app = create_app(tmp_path)
    choices = callback(app, "record-ids.options")
    first, selected = choices({"manifest": {"path": "first.csv"}}, ["old", "shared", "unknown"])
    assert [option["value"] for option in first] == ["shared", "old"]
    assert selected == ["old", "shared"]
    second, selected = choices({"manifest": {"path": "second.csv"}}, selected)
    assert [option["value"] for option in second] == ["new", "shared"]
    assert second[0]["label"] == "p2 · B · new"
    assert selected == ["shared"]
    assert observed[-2:] == ["first.csv", "second.csv"]
    assert choices({"manifest": {"path": "missing.csv"}}, selected) == ([], [])


@pytest.mark.parametrize("source", ["output-name", "relative-input", "absolute-input", "run"])
def test_finished_report_selects_actual_job_output_once(tmp_path: Path, source: str) -> None:
    from dash import no_update

    class CompletedJobs(FakeJobs):
        def __init__(self):
            super().__init__()
            self.state, self.polled, self.command = "running", [], []
        def status(self, job):
            self.polled.append(job)
            return {"state": self.state, "command": self.command}
    jobs = CompletedJobs()
    app = create_app(tmp_path, job_manager=jobs)
    for name in ("a-target", "z-unrelated"):
        report = tmp_path / "report_output" / name
        report.mkdir(parents=True)
        (report / "analysis_manifest.json").write_text("{}", encoding="utf-8")
    original = "report_output/z-unrelated"
    target = "report_output/a-target"
    data = "pipeline_output/a-target/comparison/repeat_01"
    options = {
        "output-name": ["--input", "pipeline_output/other", "--output-name", "a-target"],
        "relative-input": ["--input", data],
        "absolute-input": ["--input", str(tmp_path / data)],
        "run": ["--run", f"not-the-output-name={data}", "--run", f"second={data}/fold_02"],
    }
    jobs.command = ["python", "analyse_report.py", "run", *options[source]]
    finished = callback(app, "report-preview-job.data")
    untouched = (no_update, no_update, no_update)
    assert finished(0, None, original, None) == untouched
    assert jobs.polled == []
    for state in ("running", "failed"):
        jobs.state = state
        assert finished(1, "job-1", original, None) == untouched
    jobs.state = "passed"
    choices, selected, displayed = finished(2, "job-1", original, None)
    assert {row["value"] for row in choices} == {original, target}
    assert selected == target and displayed == "job-1"
    polled = list(jobs.polled)
    assert finished(3, "job-1", original, displayed) == untouched
    assert jobs.polled == polled  # A later user selection is not overridden on the next poll.
    assert finished(4, "job-2", original, displayed)[1:] == (target, "job-2")
    components = {component.id: component for component in app.layout()._traverse()
                  if isinstance(getattr(component, "id", None), str)}
    assert "report-preview-job" in components
    assert jobs.started == []
