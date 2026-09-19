from __future__ import annotations

import hashlib
import json
import shutil
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

from ppg_frailty.config import canonical_json_bytes
from ppg_frailty.dashboard.control_service import (
    INFERENCE_SOURCE_CONTRACT,
    CommandRequest,
    MISSING_B_TODO,
    ModelDefaults,
    V5ControlService,
    changed_assignments,
    comparison_sequence_cli,
    comparison_sequence_export_yaml,
    comparison_sequence_yaml,
    flatten_parameters,
    validate_comparison_sequence_payload,
)
from ppg_frailty.dashboard.job_manager import DashboardJobManager
from ppg_frailty.dashboard.sequence_cli import (
    build_parser as build_sequence_parser,
    load_sequence_request,
    run_sequence_request,
)
from ppg_frailty.training import OofPredictionRow, write_oof_parquet
from ppg_frailty.v5.cli import build_parser as build_pipeline_parser
from ppg_frailty.v5.model_config_export import build_parser as build_model_export_parser
from ppg_frailty.v5.specialized import build_parser as build_specialized_parser
from ppg_frailty.v5.sweep import build_parser as build_sweep_parser
from ppg_frailty.v5_reporting.cli import build_parser as build_report_parser


PIPELINE_ROOT = Path(__file__).resolve().parents[1]


def _callback(app: object, output: str):
    return next(value["callback"].__wrapped__ for key, value in app.callback_map.items() if output in key)


def _layout(app: object):
    return app.layout() if callable(app.layout) else app.layout


@contextmanager
def _trigger(component: object, prop: str = "n_clicks"):
    from dash._callback_context import context_value
    from dash._utils import AttributeDict

    identity = json.dumps(component, separators=(",", ":"), sort_keys=True) if isinstance(component, dict) else str(component)
    token = context_value.set(AttributeDict(triggered_inputs=[{"prop_id": f"{identity}.{prop}", "value": 1}]))
    try:
        yield
    finally:
        context_value.reset(token)


def _comparison_case(name: str, request: CommandRequest) -> dict[str, object]:
    payload = request.to_dict()
    payload["name"] = name
    payload["arguments"] = list(request.arguments)
    return payload


def test_paths_are_confined_and_output_roots_are_fixed(tmp_path: Path) -> None:
    root = tmp_path / "repo" / "final_v0" / "final_pipeline_v5"
    root.mkdir(parents=True)
    service = V5ControlService(root)

    assert service.output_directory("pipeline_output", "finalcase").parent.name == "pipeline_output"
    assert service.output_directory("report_output", "finalcase").parent.name == "report_output"
    with pytest.raises(ValueError, match="output name"):
        service.output_directory("pipeline_output", "../escape")
    with pytest.raises(ValueError, match="unsupported V5 output root"):
        service.output_directory("artifacts", "run")
    with pytest.raises(ValueError, match="must remain"):
        service.safe_input(tmp_path.parent / "outside.csv", label="input", must_exist=False)


def _write_manifest(root: Path, files: list[dict[str, object]]) -> Path:
    for row in files:
        path = root / str(row["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("RED,IR,AX,AY,AZ,GX,GY,GZ\n", encoding="utf-8")
    manifest = root / "participant.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "participant_id": "new-001",
                "source_contract": INFERENCE_SOURCE_CONTRACT,
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_dynamic_inference_without_b_calibration_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "repo" / "final_v0" / "final_pipeline_v5"
    root.mkdir(parents=True)
    service = V5ControlService(root)
    manifest = _write_manifest(
        root,
        [{"file_id": "r1", "role": "R1", "path": "input/r1.csv"}],
    )

    with pytest.raises(ValueError, match="missing_static_b_calibration") as error:
        service.read_inference_manifest(manifest)
    assert MISSING_B_TODO in str(error.value)


def test_static_b_plus_dynamic_file_is_a_valid_input_contract(tmp_path: Path) -> None:
    root = tmp_path / "repo" / "final_v0" / "final_pipeline_v5"
    root.mkdir(parents=True)
    service = V5ControlService(root)
    manifest = _write_manifest(
        root,
        [
            {"file_id": "b", "role": "B", "path": "input/b.csv"},
            {"file_id": "r1", "role": "R1", "path": "input/r1.csv", "label": 0},
        ],
    )

    resolved = service.read_inference_manifest(manifest)
    assert resolved.participant_id == "new-001"
    assert [row.role for row in resolved.files] == ["B", "R1"]
    assert resolved.labelled_participant_count == 1


def test_dashboard_rows_are_safely_materialized_as_a_hash_bound_manifest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo" / "final_v0" / "final_pipeline_v5"
    source = root / "input"
    source.mkdir(parents=True)
    for name in ("b.csv", "r1.csv"):
        (source / name).write_text(
            "RED,IR,AX,AY,AZ,GX,GY,GZ\n", encoding="utf-8"
        )
    service = V5ControlService(root)
    rows = [
        {"file_id": "b", "path": "input/b.csv", "role": "B", "label": ""},
        {"file_id": "r1", "path": "input/r1.csv", "role": "R1", "label": 0},
    ]

    with pytest.raises(ValueError, match="confirm the exact 400 Hz"):
        service.materialize_inference_manifest(
            participant_id="new-001",
            files=rows,
            source_contract_confirmed=False,
        )

    first = service.materialize_inference_manifest(
        participant_id="new-001",
        files=rows,
        source_contract_confirmed=True,
    )
    second = service.materialize_inference_manifest(
        participant_id="new-001",
        files=rows,
        source_contract_confirmed=True,
    )
    payload = yaml.safe_load(first.read_text(encoding="utf-8"))

    assert first == second
    assert first.parent.relative_to(root).as_posix() == (
        "pipeline_output/.dashboard_requests/inference"
    )
    assert first.name.startswith("request_") and first.suffix == ".yaml"
    assert payload["source_contract"] == INFERENCE_SOURCE_CONTRACT
    assert [row["path"] for row in payload["files"]] == [
        "input/b.csv",
        "input/r1.csv",
    ]
    resolved = service.read_inference_manifest(first)
    assert resolved.participant_id == "new-001"
    assert [row.role for row in resolved.files] == ["B", "R1"]
    assert resolved.files[0].label is None


def test_dashboard_manifest_materialization_fails_closed_for_bad_rows(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo" / "final_v0" / "final_pipeline_v5"
    source = root / "input"
    source.mkdir(parents=True)
    (source / "b.csv").write_text(
        "RED,IR,AX,AY,AZ,GX,GY,GZ\n", encoding="utf-8"
    )
    service = V5ControlService(root)

    with pytest.raises(ValueError, match="filesystem-safe"):
        service.materialize_inference_manifest(
            participant_id="../escape",
            files=[
                {"file_id": "b", "path": "input/b.csv", "role": "B", "label": 0}
            ],
            source_contract_confirmed=True,
        )
    with pytest.raises(ValueError, match="share one label"):
        service.materialize_inference_manifest(
            participant_id="new-001",
            files=[
                {"file_id": "b", "path": "input/b.csv", "role": "B", "label": 0},
                {"file_id": "b2", "path": "input/b.csv", "role": "B", "label": 1},
            ],
            source_contract_confirmed=True,
        )


def test_inference_request_round_trips_through_public_cli_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo" / "final_v0" / "final_pipeline_v5"
    source = root / "input"
    source.mkdir(parents=True)
    (root / "model_config" / "run_a").mkdir(parents=True)
    (root / "model_config" / "run_a" / "config.yaml").write_text(
        "schema_version: test_fixture\n", encoding="utf-8"
    )
    (source / "b.csv").write_text(
        "RED,IR,AX,AY,AZ,GX,GY,GZ\n", encoding="utf-8"
    )
    service = V5ControlService(root)
    manifest = service.materialize_inference_manifest(
        participant_id="new-001",
        files=[
            {"file_id": "b", "path": "input/b.csv", "role": "B", "label": None}
        ],
        source_contract_confirmed=True,
    )
    monkeypatch.setattr(
        service,
        "load_model_defaults",
        lambda *_: ModelDefaults(
            export_directory="model_config/run_a",
            case_id="finalcase",
            config_path="model_config/run_a/config.yaml",
            config={},
            module_defaults={},
            feature_defaults=(),
            inference_capability={"available": True},
        ),
    )

    request = service.build_inference_request(
        model_export="model_config/run_a",
        case_id="finalcase",
        input_manifest=manifest,
    )
    parsed = build_pipeline_parser().parse_args(list(request.arguments))

    assert request.script == "pipeline.py"
    assert parsed.command == "infer"
    assert parsed.model_config == "model_config/run_a"
    assert parsed.case_id == "finalcase"
    assert parsed.input_manifest == service.relative(manifest)
    assert request.display == (
        "python pipeline.py infer --model-config model_config/run_a "
        f"--case-id finalcase --input-manifest {service.relative(manifest)}"
    )


def test_parameter_table_emits_only_changed_values() -> None:
    rows = flatten_parameters(
        {"training": {"batch_size": 16, "enabled": True}, "roles": ["B", "R1"]}
    )
    assert changed_assignments(rows) == ()
    selected = next(row for row in rows if row["path"] == "training.batch_size")
    selected["value_yaml"] = "32"
    assert changed_assignments(rows) == ("training.batch_size=32",)


def test_parameter_table_and_shortcuts_use_the_live_cli_contract() -> None:
    pytest.importorskip("dash")
    from ppg_frailty.dashboard.app import _parameter_control
    from ppg_frailty.dashboard.workflow_controls import default_configuration, grouped_parameter_specs

    service = V5ControlService(PIPELINE_ROOT)
    contract = service.parameter_contract()
    assert contract["evaluation.statistics.lcb95_percentile"]["range"] == "finite float in [0,100]"
    specs = grouped_parameter_specs(default_configuration(PIPELINE_ROOT), pipeline_root=PIPELINE_ROOT)
    numeric = {s["path"]: s for s in specs if s["kind"] in {"integer", "number"}}
    assert {"signal.ppg_filter.low_hz", "signal.ppg_filter.high_hz", "signal.ppg_filter.order",
            "signal.imu.sensor_lowpass_acc_hz", "training.learning_rate"} <= set(numeric)
    for path, spec in numeric.items():
        component = _parameter_control(spec)
        slider = next(c for c in component._traverse() if c.__class__.__name__ == "Slider")
        editor = next(c for c in component._traverse() if c.__class__.__name__ == "Input")
        assert slider.id["path"] == editor.id["path"] == path
        assert slider.step is not None and slider.step > 0
        assert slider.value == editor.value == spec["value"]
        assert editor.type == "number"
    assert next(s for s in specs if s["path"] == "signal.normalization.standard_ddof")["choices"] == [0, 1]


def test_train_request_uses_same_resolver_and_fixed_output_root() -> None:
    service = V5ControlService(PIPELINE_ROOT)
    config_path = "configs/presets/finalcase.yaml"
    config, _ = service.load_yaml(config_path)
    rows = flatten_parameters(config)
    next(row for row in rows if row["path"] == "training.batch_size")[
        "value_yaml"
    ] = "32"
    defaults = service.module_defaults_from_config(config)

    request = service.build_train_request(
        config_path=config_path,
        selected_modules=defaults,
        default_modules=defaults,
        feature_groups=config["features"]["enabled_groups"],
        default_feature_groups=config["features"]["enabled_groups"],
        parameter_rows=rows,
        study_id="dash_finalcase",
        jobs=1,
    )

    assert request.script == "pipeline.py"
    assert request.arguments[:3] == ("run", "--config", config_path)
    assert ("--set", "training.batch_size=32") == tuple(
        request.arguments[index : index + 2]
        for index, value in enumerate(request.arguments)
        if value == "--set"
    )[0]
    output_index = request.arguments.index("--output-root")
    assert request.arguments[output_index + 1] == "pipeline_output"
    assert yaml.safe_load(request.resolved_yaml)["training"]["device"] == "cuda"


def test_train_request_preserves_index_subsets_and_repeated_unsets() -> None:
    service = V5ControlService(PIPELINE_ROOT)
    paths = [
        "training.gradient_clip_norm",
        "training.samples_per_epoch",
    ]
    request = service.build_train_request(
        config_path="configs/presets/finalcase.yaml",
        repeats=[4, 1, 3],
        folds=["0", "2"],
        unset_paths=paths,
    )
    parsed = build_pipeline_parser().parse_args(list(request.arguments))

    assert parsed.repeats == (4, 1, 3)
    assert parsed.folds == (0, 2)
    assert parsed.unset == paths
    assert request.arguments.count("--unset") == 2

    for invalid in ([], [0, 0], [0, 5]):
        with pytest.raises(ValueError, match="unique subset"):
            service.build_train_request(
                config_path="configs/presets/finalcase.yaml",
                repeats=invalid,
            )


def test_sweep_rejects_unset_instead_of_approximating_study_plan() -> None:
    service = V5ControlService(PIPELINE_ROOT)
    with pytest.raises(ValueError, match="StudyPlan cannot encode arbitrary --unset"):
        service.build_train_request(
            config_path=None,
            plan_path="configs/studies/finalcase.yaml",
            operation="sweep",
            unset_paths=["training.gradient_clip_norm"],
        )


def test_train_request_device_round_trip_without_environment_gate() -> None:
    service = V5ControlService(PIPELINE_ROOT)
    request = service.build_train_request(
        config_path="configs/presets/finalcase.yaml",
        device="cpu",
    )

    parsed = build_pipeline_parser().parse_args(list(request.arguments))
    resolved = yaml.safe_load(request.resolved_yaml)
    assert parsed.device == resolved["training"]["device"] == "cpu"
    assert not {"--environment-policy", "--environment-lock"} & set(request.arguments)
    assert request.config_sha256 == hashlib.sha256(
        canonical_json_bytes(resolved)
    ).hexdigest()


def test_train_request_rejects_resume_with_run_name_before_path_resolution() -> None:
    service = V5ControlService(PIPELINE_ROOT)
    with pytest.raises(ValueError, match="--run-name cannot be combined with --resume"):
        service.build_train_request(
            config_path="configs/presets/finalcase.yaml",
            resume="pipeline_output/does_not_need_to_exist",
            run_name="new_run",
        )


def test_config_tools_reuse_exact_configure_selectors_and_public_parser() -> None:
    service = V5ControlService(PIPELINE_ROOT)
    config, _ = service.load_yaml("configs/presets/finalcase.yaml")
    rows = flatten_parameters(config)
    next(row for row in rows if row["path"] == "training.batch_size")[
        "value_yaml"
    ] = "32"
    run_request = service.build_train_request(
        config_path="configs/presets/finalcase.yaml",
        parameter_rows=rows,
        unset_paths=["training.gradient_clip_norm"],
        config_id="dashboard_review",
        repeats=[2],
        folds=[3],
        jobs=2,
    )

    show = service.build_pipeline_config_tool_request(
        operation="show-config",
        run_request=run_request,
    )
    validate = service.build_pipeline_config_tool_request(
        operation="validate",
        run_request=run_request.to_dict(),
        validation_mode="config",
    )
    parsed_show = build_pipeline_parser().parse_args(list(show.arguments))
    parsed_validate = build_pipeline_parser().parse_args(list(validate.arguments))

    assert parsed_show.command == "show-config"
    assert parsed_show.assignments == ["training.batch_size=32"]
    assert parsed_show.unset == ["training.gradient_clip_norm"]
    assert parsed_show.config_id == "dashboard_review"
    assert "--repeats" not in show.arguments
    assert "--jobs" not in show.arguments
    assert parsed_validate.command == "validate"
    assert parsed_validate.mode == "config"

    sweep = CommandRequest(
        script="sweep.py",
        arguments=("run", "--plan", "configs/studies/finalcase.yaml"),
        display="",
    )
    with pytest.raises(ValueError, match="not a sweep"):
        service.build_pipeline_config_tool_request(
            operation="validate",
            run_request=sweep,
        )


def test_index_excel_and_model_export_commands_round_trip_public_parsers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo" / "final_v0" / "final_pipeline_v5"
    pipeline = root / "pipeline_output" / "run_a"
    report = root / "report_output" / "run_a"
    pipeline.mkdir(parents=True)
    report.mkdir(parents=True)
    service = V5ControlService(root)

    index = service.build_pipeline_index_request(
        study_directory="pipeline_output/run_a",
        hash_predictions=True,
    )
    pipeline_excel = service.build_pipeline_excel_request(
        pipeline_output="pipeline_output/run_a",
        replace=True,
    )
    report_excel = service.build_report_excel_request(
        report_output="report_output/run_a",
        replace=True,
    )
    execution_audit = service.build_execution_audit_request(
        pipeline_output="pipeline_output/run_a",
        output_name="run_a_failure",
    )
    model_export = service.build_model_export_request(
        pipeline_output="pipeline_output/run_a"
    )

    assert build_pipeline_parser().parse_args(list(index.arguments)).hash_predictions
    assert build_sweep_parser().parse_args(list(pipeline_excel.arguments)).replace
    assert build_report_parser().parse_args(list(report_excel.arguments)).replace
    parsed_audit = build_report_parser().parse_args(list(execution_audit.arguments))
    assert parsed_audit.command == "execution-audit"
    assert parsed_audit.input == "pipeline_output/run_a"
    assert parsed_audit.output_name == "run_a_failure"
    assert build_model_export_parser().parse_args(
        list(model_export.arguments)
    ).pipeline_output == "pipeline_output/run_a"
    assert model_export.display == (
        "python export_model_config.py --pipeline-output pipeline_output/run_a"
    )
    with pytest.raises(ValueError, match="inside pipeline_output"):
        service.build_pipeline_index_request(study_directory="report_output/run_a")


def test_model_export_tool_calls_existing_atomic_exporter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo" / "final_v0" / "final_pipeline_v5"
    source = root / "pipeline_output" / "run_a"
    source.mkdir(parents=True)
    service = V5ControlService(root)
    observed: dict[str, object] = {}

    def fake_export(value: object, *, pipeline_root: object) -> dict[str, object]:
        observed.update({"value": value, "pipeline_root": pipeline_root})
        return {"status": "complete", "output_directory": "model_config/run_a"}

    monkeypatch.setattr(
        "ppg_frailty.v5.model_config_export.export_model_config",
        fake_export,
    )
    result = service.execute_model_export(
        pipeline_output="pipeline_output/run_a"
    )

    assert observed == {
        "value": "pipeline_output/run_a",
        "pipeline_root": root.resolve(),
    }
    assert result["output_directory"] == "model_config/run_a"


def test_tools_callback_builds_the_same_index_cli(tmp_path: Path) -> None:
    pytest.importorskip("dash")
    from ppg_frailty.dashboard import create_app
    from ppg_frailty.dashboard.app import TOOLS

    root = tmp_path / "repo" / "final_v0" / "final_pipeline_v5"
    (root / "pipeline_output" / "run_a").mkdir(parents=True)
    app = create_app(root)
    callback = _callback(app, "tool-request.data")
    request, command, request_yaml, help_text, disabled = callback(
        "pipeline_index", "--study-dir pipeline_output/run_a --hash-predictions"
    )
    assert request["script"] == "pipeline.py"
    assert request["arguments"] == ("index", "--study-dir", "pipeline_output/run_a", "--hash-predictions")
    assert command == "python pipeline.py index --study-dir pipeline_output/run_a --hash-predictions"
    assert yaml.safe_load(request_yaml)["script"] == "pipeline.py"
    assert "--hash-predictions" in help_text and disabled is False
    request, _, request_yaml, help_text, disabled = callback(
        "specialized_pipeline_run", "--plan configs/specialized.yaml --source-root . "
        "--run-name preserved_run --device cuda --jobs 1 --no-denoiser"
    )
    parsed = build_specialized_parser().parse_args(list(request["arguments"]))
    assert request["script"] == "specialized_pipeline.py"
    assert parsed.command == "run" and parsed.run_name == "preserved_run"
    assert parsed.no_denoiser is True and disabled is True
    assert "--plan" in help_text
    download = _callback(app, "download-tool-yaml.data")(1, request)
    assert download["filename"] == "v5_tool_request.yaml"
    assert yaml.safe_load(download["content"])["script"] == "specialized_pipeline.py"
    assert _callback(app, "download-tool-cli.data")(1, request)["content"] == request["display"] + "\n"
    assert set(TOOLS) == {
        "pipeline_validate", "show_config", "sweep_validate", "pipeline_index", "model_export",
        "pipeline_excel", "report_excel", "execution_audit", "specialized_validate", "specialized_run",
        "specialized_report", "specialized_pipeline_validate", "specialized_pipeline_run",
        "specialized_pipeline_complete",
    }


def test_refit_is_one_optional_flag_without_selection_gates() -> None:
    service = V5ControlService(PIPELINE_ROOT)
    config, _ = service.load_yaml("configs/presets/finalcase.yaml")
    common = {
        "config_path": "configs/presets/finalcase.yaml",
        "parameter_rows": flatten_parameters(config),
    }

    ordinary = service.build_train_request(**common)
    refit = service.build_train_request(**common, refit=True)
    parsed = build_pipeline_parser().parse_args(list(refit.arguments))

    assert "--refit" not in ordinary.arguments
    assert refit.arguments.count("--refit") == 1
    assert parsed.refit is True
    assert not {"--selection-record", "--comparison-archive", "--confirm-refit-config-sha256"} & set(refit.arguments)


def test_sweep_and_specialized_maintenance_commands_use_public_wrappers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo" / "final_v0" / "final_pipeline_v5"
    plan = root / "configs" / "study.yaml"
    specialized = root / "configs" / "specialized.yaml"
    source_study = root / "pipeline_output" / "source"
    prediction = source_study / "predictions.parquet"
    report_source = root / "pipeline_output" / "specialized_source"
    plan.parent.mkdir(parents=True)
    source_study.mkdir(parents=True)
    report_source.mkdir(parents=True)
    plan.write_text("study: test\n", encoding="utf-8")
    specialized.write_text("schema_version: preserved\n", encoding="utf-8")
    prediction.write_bytes(b"PAR1")
    service = V5ControlService(root)
    monkeypatch.setattr(service, "load_study_plan", lambda _: (object(), "study: test\n"))

    sweep = service.build_sweep_validate_request(
        plan_path="configs/study.yaml",
    )
    validate = service.build_specialized_request(
        operation="specialized-validate",
        plan_path="configs/specialized.yaml",
        source_root=".",
    )
    run = service.build_specialized_request(
        operation="specialized-run",
        plan_path="configs/specialized.yaml",
        source_root=".",
        output_name="special_analysis",
        study_directory="pipeline_output/source",
        case_id="candidate_a",
        prediction_file="pipeline_output/source/predictions.parquet",
        step=0.25,
    )
    report = service.build_specialized_request(
        operation="specialized-report",
        report_input="pipeline_output/specialized_source",
        output_name="special_report",
    )
    computation_validate = service.build_specialized_pipeline_request(
        operation="validate",
        plan_path="configs/specialized.yaml",
        source_root=".",
    )
    computation_run = service.build_specialized_pipeline_request(
        operation="run",
        plan_path="configs/specialized.yaml",
        run_name="preserved_run",
        source_root=".",
        upstream_study="pipeline_output/source",
        device="cuda",
        jobs=2,
        include_denoiser=False,
    )
    computation_complete = service.build_specialized_pipeline_request(
        operation="complete",
        study_directory="pipeline_output/source",
        device="cuda:0",
        jobs=1,
        dry_run=True,
    )

    assert build_sweep_parser().parse_args(list(sweep.arguments)).command == "validate"
    assert build_report_parser().parse_args(list(validate.arguments)).command == (
        "specialized-validate"
    )
    parsed_run = build_report_parser().parse_args(list(run.arguments))
    assert parsed_run.command == "specialized-run"
    assert parsed_run.step == 0.25
    assert build_report_parser().parse_args(list(report.arguments)).command == (
        "specialized-report"
    )
    assert build_specialized_parser().parse_args(
        list(computation_validate.arguments)
    ).command == "validate"
    parsed_computation = build_specialized_parser().parse_args(
        list(computation_run.arguments)
    )
    assert parsed_computation.command == "run"
    assert parsed_computation.run_name == "preserved_run"
    assert parsed_computation.no_denoiser is True
    assert parsed_computation.jobs == 2
    parsed_completion = build_specialized_parser().parse_args(
        list(computation_complete.arguments)
    )
    assert parsed_completion.command == "complete"
    assert parsed_completion.dry_run is True
    assert parsed_completion.device == "cuda:0"

    with pytest.raises(ValueError, match="cannot be combined"):
        service.build_specialized_pipeline_request(
            operation="run",
            plan_path="configs/specialized.yaml",
            run_name="new",
            resume="pipeline_output/source",
        )


def test_dash_run_callback_forwards_execution_controls() -> None:
    pytest.importorskip("dash")
    from ppg_frailty.dashboard import create_app

    service = V5ControlService(PIPELINE_ROOT)
    config, _ = service.load_yaml("configs/presets/finalcase.yaml")
    config["training"]["gradient_clip_norm"] = None
    app = create_app(PIPELINE_ROOT)
    callback = _callback(app, "train-request.data")
    request, command, resolved = callback(
        config, "configs/presets/finalcase.yaml", "", "0,2,4", "1,3", 1,
        "off", "cache/preprocessing", ["refit"], "",
    )
    assert request is not None, command
    parsed = build_pipeline_parser().parse_args(list(request["arguments"]))
    assert parsed.repeats == (0, 2, 4)
    assert parsed.folds == (1, 3)
    assert parsed.refit is True
    assert parsed.preprocessing_cache_mode == "off"
    assert not {"--environment-policy", "--environment-lock"} & set(request["arguments"])
    assert yaml.safe_load(resolved)["training"]["gradient_clip_norm"] is None
    assert yaml.safe_load(request["resolved_yaml"])["training"]["gradient_clip_norm"] is None
    assert request["display"] == command


def test_dash_inference_table_materializes_then_calls_shared_service() -> None:
    pytest.importorskip("dash")
    from ppg_frailty.dashboard import create_app

    observed = {}
    class Workflow:
        def analyse(self, **kwargs):
            observed.update(kwargs)
            return {"requested_stage": kwargs["stage"], "previews": {"model": {"tables": {"prediction": [{"class": 1}]}}}}

    class NoJobs:
        def start_request(self, *args, **kwargs):
            pytest.fail("Analyse must not submit a training job")

    app = create_app(PIPELINE_ROOT, workflow_service=Workflow(), job_manager=NoJobs())
    components = {c.id: c for c in _layout(app)._traverse() if isinstance(getattr(c, "id", None), str)}
    table = components["input-files"]
    assert table.editable is True and table.row_deletable is True
    assert [c["id"] for c in table.columns] == ["file_id", "role", "path", "label"]
    assert components["record-ids"].multi is True
    assert components["training-yaml"].value is None
    config, _ = V5ControlService(PIPELINE_ROOT).load_yaml("configs/presets/finalcase.yaml")
    files = [{"file_id": "b", "path": "input/b.csv", "role": "B", "label": "Young"},
             {"file_id": "r", "path": "input/r.csv", "role": "R1", "label": ""}]
    with _trigger({"type": "analyse-stage", "stage": "model"}):
        result, stage = _callback(app, "preview-store.data")(
            [1], config, [], [], [{"type": "param-number", "path": "signal.ppg_filter.high_hz"}], [7.0],
            "r", "browser-session", 3, 15, files, [],
            "new-001", None, None, None, None, None, "model_config/run_a", "finalcase", None, None,
        )
    assert stage == observed["stage"] == "model"
    assert observed["config_payload"]["signal"]["ppg_filter"]["high_hz"] == 7.0
    assert config["signal"]["ppg_filter"]["high_hz"] == 8.0
    assert observed["selections"]["files"] == files
    assert observed["selections"]["model_export"] == "model_config/run_a"
    assert observed["selections"]["model_case"] == "finalcase"
    assert observed["selections"]["participant_id"] == "new-001"
    assert observed["session_id"] == "browser-session"
    assert observed["start_s"] == 3 and observed["duration_s"] == 15
    assert result["previews"]["model"]["tables"]["prediction"] == [{"class": 1}]


def test_dash_preview_replaces_model_placeholders_with_completed_artifacts() -> None:
    pytest.importorskip("dash")
    from ppg_frailty.dashboard import create_app

    app = create_app(PIPELINE_ROOT)
    callback = _callback(app, '"type":"stage-output"')
    config = {"config_id": "current"}
    rows = [{"file_id": "r", "probabilities": [0.7, 0.2, 0.1]}]
    payload = {"configuration": config, "requested_stage": "model",
               "previews": {"model": {"tables": {"predictions": rows}}},
               "computed_stages": ["model"], "reused_stages": [],
               "record_id": "r", "display_interval": [0, 20],
               "input_selection": {"files": [], "record_ids": [], "participant_id": "p",
                                   "calibration_path": None, "motion_bundle": None, "sqi_artifact": None,
                                   "model_export": None, "model_case": None, "model_bundle": None}}
    identities = [{"type": "stage-output", "stage": "model"}]
    selection = ("r", [], [], 0, 20, "p", None, None, None, None, None, None, None, None, None)
    rendered = callback(payload, config, identities, *selection)[0]
    table = next(c for c in rendered if c.__class__.__name__ == "DataTable")
    assert table.data[0]["file_id"] == "r"
    assert json.loads(table.data[0]["probabilities"]) == [0.7, 0.2, 0.1]
    assert rendered[0].children == "Computed"
    payload["reused_stages"] = ["model"]
    assert callback(payload, config, identities, *selection)[0][0].children == "Reused"
    assert "Settings changed" in callback(payload, {"config_id": "changed"}, identities, *selection)[0][0].children
    assert "selection changed" in callback(payload, config, identities, "other-record", *selection[1:])[0][0].children
    assert "selection changed" in callback(payload, config, identities, *selection[:3], 5, 20, *selection[5:])[0][0].children
    failed = {"requested_stage": "model", "error": "Missing fitted transform", "previews": {}}
    assert callback(failed, config, identities, *selection)[0][0].children == "Missing fitted transform"


def test_train_requires_explicit_yaml() -> None:
    service = V5ControlService(PIPELINE_ROOT)
    with pytest.raises(ValueError, match="explicitly selected YAML"):
        service.build_train_request(config_path=None)


def test_comparison_queue_exports_ordered_cli_and_yaml() -> None:
    base = ["run", "--config", "configs/presets/finalcase.yaml"]
    cases = [
        {
            "name": "case_a",
            "script": "pipeline.py",
            "arguments": [*base, "--set", "training.batch_size=16"],
            "config_sha256": "a" * 64,
        },
        {
            "name": "case_b",
            "script": "pipeline.py",
            "arguments": [*base, "--set", "training.batch_size=32"],
            "config_sha256": "b" * 64,
        },
    ]
    cli = comparison_sequence_cli(cases)
    payload = yaml.safe_load(
        comparison_sequence_yaml(cases, pipeline_root=PIPELINE_ROOT)
    )

    assert cli.splitlines()[0].endswith("training.batch_size=16")
    assert cli.splitlines()[1].endswith("training.batch_size=32")
    assert payload["schema_version"] == "ppg_frailty.dashboard_comparison_sequence.v2"
    assert payload["study"]["kind"] == "comparison_sequence"
    assert set(payload["launch"]) == {"run_name", "hash_predictions", "dry_run", "refit"}
    assert payload["study_plan_v2"] is False
    assert [row["order"] for row in payload["cases"]] == [1, 2]
    assert [row["pipeline_request"]["command"] for row in payload["cases"]] == cli.splitlines()
    with pytest.raises(ValueError, match="unique safe name"):
        comparison_sequence_yaml([cases[0], cases[0]], pipeline_root=PIPELINE_ROOT)
    unrelated = [
        cases[0],
        {
            **cases[1],
            "arguments": ["run", "--config", "configs/presets/baseline.yaml"],
        },
    ]
    unrelated_payload = yaml.safe_load(comparison_sequence_yaml(unrelated, pipeline_root=PIPELINE_ROOT))
    assert unrelated_payload["cases"][1]["pipeline_request"]["arguments"][2] == "configs/presets/baseline.yaml"


def test_comparison_callback_displays_the_same_cli_and_yaml_it_exports() -> None:
    pytest.importorskip("dash")
    from ppg_frailty.dashboard import create_app

    service = V5ControlService(PIPELINE_ROOT)
    config, _ = service.load_yaml("configs/presets/finalcase.yaml")
    first = service.build_train_request(config_path="configs/presets/finalcase.yaml")
    app = create_app(PIPELINE_ROOT, control_service=service)
    add = _callback(app, "comparison-store.data")
    snapshot = ("configs/presets/finalcase.yaml", config, [], [], [], [], "", "all", "all", 1,
                "off", "cache/preprocessing", [], "")
    with _trigger("add-comparison"):
        stored, _ = add(1, 0, 0, "reference", first.to_dict(), [], *snapshot)
        # The cached request deliberately remains old: Add must snapshot live controls.
        stored, status = add(2, 0, 0, "batch_32", first.to_dict(), stored,
                             "configs/presets/finalcase.yaml", config, [], [],
                             [{"type": "param-number", "path": "training.batch_size"}], [32],
                             "", "all", "all", 1, "off", "cache/preprocessing", [], "")
    _, cli, exported = _callback(app, "comparison-cli-view.children")(stored)
    assert cli == comparison_sequence_cli(stored)
    assert exported == comparison_sequence_export_yaml(stored, pipeline_root=PIPELINE_ROOT)
    assert yaml.safe_load(exported)["schema_version"] == "ppg_frailty.dashboard_comparison_sequence.v2"
    assert "2 units" in status
    assert yaml.safe_load(stored[1]["resolved_yaml"])["training"]["batch_size"] == 32
    assert _callback(app, "download-sequence-cli.data")(1, stored)["content"] == cli
    assert _callback(app, "download-sequence-yaml.data")(1, stored)["content"] == exported
    assert len(stored) == 2
    with _trigger("remove-comparison"):
        reduced, _ = add(2, 1, 0, "", None, stored, *snapshot)
    assert [row["name"] for row in reduced] == ["reference"]
    assert len(stored) == 2
    with _trigger("clear-comparison"):
        assert add(2, 1, 1, "", None, stored, *snapshot)[0] == []


def test_comparison_run_callback_submits_through_the_training_job(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("dash")
    from ppg_frailty.dashboard import create_app

    service = V5ControlService(PIPELINE_ROOT)
    execution = CommandRequest(
        script="comparison_sequence.py", arguments=("run", "--request", "sequence.yaml"),
        display="python comparison_sequence.py run --request sequence.yaml",
        resolved_yaml="schema_version: ppg_frailty.dashboard_comparison_sequence.v2\\n",
        config_sha256="a" * 64,
    )
    def fake_execution(cases):
        assert cases == [{"name": "cached"}]
        return execution, PIPELINE_ROOT / "sequence.yaml"

    class FakeJobs:
        def __init__(self):
            self.requests = []
        def start_request(self, request, *, kind):
            self.requests.append((request, kind))
            return "queue-job-01"
        def status(self, _):
            return {"state": "passed"}

    monkeypatch.setattr(service, "build_comparison_execution_request", fake_execution)
    jobs = FakeJobs()
    app = create_app(PIPELINE_ROOT, control_service=service, job_manager=jobs)
    callback = _callback(app, "active-train-job.data")
    with _trigger("train"):
        job, status = callback(1, 0, "Train", "configs/presets/finalcase.yaml", "comparison",
                               None, [{"name": "cached"}], None, None, {}, [], [], [], [], "", "all", "all", 1,
                               "off", "cache/preprocessing", [], "")
    assert job == "queue-job-01"
    assert jobs.requests == [(execution, "pipeline")]
    assert execution.display in status


def test_sparse_comparison_exports_exact_sequence_without_claiming_study_plan() -> None:
    service = V5ControlService(PIPELINE_ROOT)
    config, _ = service.load_yaml("configs/presets/finalcase.yaml")
    cases: list[dict[str, object]] = []
    for name, batch_size, weight_decay in (
        ("low_no_decay", 16, 0.0),
        ("high_with_decay", 32, 0.001),
    ):
        rows = flatten_parameters(config)
        next(row for row in rows if row["path"] == "training.batch_size")[
            "value_yaml"
        ] = str(batch_size)
        next(row for row in rows if row["path"] == "training.weight_decay")[
            "value_yaml"
        ] = str(weight_decay)
        cases.append(
            _comparison_case(
                name,
                service.build_train_request(
                    config_path="configs/presets/finalcase.yaml",
                    parameter_rows=rows,
                    repeats="4,1",
                    folds="3",
                ),
            )
        )

    assert comparison_sequence_yaml(cases, pipeline_root=PIPELINE_ROOT) == comparison_sequence_export_yaml(
        cases, pipeline_root=PIPELINE_ROOT
    )
    cli = comparison_sequence_cli(cases)
    payload = yaml.safe_load(
        comparison_sequence_export_yaml(cases, pipeline_root=PIPELINE_ROOT)
    )

    assert payload["schema_version"] == (
        "ppg_frailty.dashboard_comparison_sequence.v2"
    )
    assert payload["study"]["kind"] == "comparison_sequence"
    assert payload["execution"]["repeats"] == [4, 1]
    assert payload["execution"]["folds"] == [3]
    assert payload["study_plan_v2"] is False
    assert [row["pipeline_request"]["command"] for row in payload["cases"]] == (
        cli.splitlines()
    )
    assert [row["resolved_config"]["training"]["batch_size"] for row in payload["cases"]] == [
        16,
        32,
    ]


def test_cartesian_queue_uses_the_same_sequence_executor(tmp_path: Path) -> None:
    root = tmp_path / "repo" / "final_v0" / "final_pipeline_v5"
    shutil.copytree(PIPELINE_ROOT / "configs", root / "configs")
    service = V5ControlService(root)
    config, _ = service.load_yaml("configs/presets/finalcase.yaml")
    cases: list[dict[str, object]] = []
    for batch_size in (16, 32):
        rows = flatten_parameters(config)
        next(row for row in rows if row["path"] == "training.batch_size")[
            "value_yaml"
        ] = str(batch_size)
        cases.append(
            _comparison_case(
                f"batch_{batch_size}",
                service.build_train_request(
                    config_path="configs/presets/finalcase.yaml",
                    parameter_rows=rows,
                ),
            )
        )

    request, target = service.build_comparison_execution_request(cases)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))

    assert request.script == "comparison_sequence.py"
    assert request.arguments == (
        "run",
        "--request",
        service.relative(target),
    )
    assert request.resolved_yaml == target.read_text(encoding="utf-8")
    assert payload["schema_version"] == "ppg_frailty.dashboard_comparison_sequence.v2"
    assert payload["study_plan_v2"] is False
    assert target.parent == (
        root / "pipeline_output" / ".dashboard_requests" / "comparison"
    )
    assert target.name == f"sequence_{request.config_sha256}.yaml"
    assert build_sequence_parser().parse_args(list(request.arguments)).command == "run"


def test_sparse_queue_materializes_its_distinct_executable_schema(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo" / "final_v0" / "final_pipeline_v5"
    shutil.copytree(PIPELINE_ROOT / "configs", root / "configs")
    service = V5ControlService(root)
    config, _ = service.load_yaml("configs/presets/finalcase.yaml")
    cases: list[dict[str, object]] = []
    for name, batch_size, weight_decay in (
        ("low_no_decay", 16, 0.0),
        ("high_with_decay", 32, 0.001),
    ):
        rows = flatten_parameters(config)
        next(row for row in rows if row["path"] == "training.batch_size")[
            "value_yaml"
        ] = str(batch_size)
        next(row for row in rows if row["path"] == "training.weight_decay")[
            "value_yaml"
        ] = str(weight_decay)
        cases.append(
            _comparison_case(
                name,
                service.build_train_request(
                    config_path="configs/presets/finalcase.yaml",
                    parameter_rows=rows,
                ),
            )
        )

    request, target = service.build_comparison_execution_request(cases)
    persisted, payload, normalized, digest = load_sequence_request(
        target, pipeline_root=root
    )

    assert persisted == target
    assert request.script == "comparison_sequence.py"
    assert request.arguments == ("run", "--request", service.relative(target))
    assert build_sequence_parser().parse_args(list(request.arguments)).command == "run"
    assert payload["schema_version"] == (
        "ppg_frailty.dashboard_comparison_sequence.v2"
    )
    assert payload["study_plan_v2"] is False
    assert [row["name"] for row in normalized] == [
        "low_no_decay",
        "high_with_decay",
    ]
    assert digest == request.config_sha256
    assert validate_comparison_sequence_payload(
        payload, pipeline_root=root
    ) == normalized

    target.write_text(target.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="filename is not bound"):
        load_sequence_request(target, pipeline_root=root)


def test_correlated_sequence_runs_under_one_anchored_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ppg_frailty.dashboard.sequence_cli as sequence_module
    import ppg_frailty.v5.run as run_module
    import ppg_frailty.v5.sweep as sweep_module

    root = tmp_path / "repo" / "final_v0" / "final_pipeline_v5"
    shutil.copytree(PIPELINE_ROOT / "configs", root / "configs")
    service = V5ControlService(root)
    config, _ = service.load_yaml("configs/presets/finalcase.yaml")
    queue: list[dict[str, object]] = []
    for name, batch_size, weight_decay in (
        ("low_no_decay", 16, 0.0),
        ("high_with_decay", 32, 0.001),
    ):
        rows = flatten_parameters(config)
        next(row for row in rows if row["path"] == "training.batch_size")[
            "value_yaml"
        ] = str(batch_size)
        next(row for row in rows if row["path"] == "training.weight_decay")[
            "value_yaml"
        ] = str(weight_decay)
        queue.append(
            _comparison_case(
                name,
                service.build_train_request(
                    config_path="configs/presets/finalcase.yaml",
                    parameter_rows=rows,
                    repeats="0",
                    folds="0",
                    device="cuda",
                    run_name="correlated_run",
                ),
            )
        )
    _, request_path = service.build_comparison_execution_request(queue)

    monkeypatch.setattr(sequence_module, "PIPELINE_ROOT", root)
    monkeypatch.setattr(sweep_module, "V5_ROOT", root)
    monkeypatch.setattr(sweep_module, "PIPELINE_OUTPUT_ROOT", root / "pipeline_output")
    observed: list[tuple[str, str]] = []

    def fake_executor(case: object, config_path: Path, case_directory: Path, *_: object) -> dict[str, object]:
        case_directory.mkdir(parents=True, exist_ok=False)
        observed.append((case.case_id, config_path.name))
        return {"status": "passed", "output_dir": str(case_directory)}

    monkeypatch.setattr(sequence_module, "v5_experiment_executor", fake_executor)
    finalize_calls: list[Path] = []

    def fake_finalize(study_directory: Path, **_: object) -> dict[str, object]:
        study = Path(study_directory)
        finalize_calls.append(study)
        export = root / "model_config" / study.name
        export.mkdir(parents=True)
        return {
            "data_manifest": {
                "status": "complete",
                "fold_count": 2,
                "prediction_artifact_count": 0,
            },
            "model_config_export": str(export),
            "refit": None,
        }

    monkeypatch.setattr(run_module, "post_run_finalize", fake_finalize)
    monkeypatch.setattr(
        run_module,
        "try_export_pipeline_excel",
        lambda *_args, **_kwargs: {"status": "complete"},
    )

    assert run_sequence_request(request_path, pipeline_root=root) == 0

    run = root / "pipeline_output" / "correlated_run"
    assert [item[0] for item in observed] == ["low_no_decay", "high_with_decay"]
    assert finalize_calls == [run]
    assert (root / "model_config" / "correlated_run").is_dir()
    assert (run / "low_no_decay" / "case_result.json").is_file()
    assert (run / "high_with_decay" / "case_result.json").is_file()
    assert (run / "configs" / "low_no_decay.yaml").is_file()
    assert (run / "configs" / "high_with_decay.yaml").is_file()
    manifest = json.loads((run / "study_manifest.json").read_text(encoding="utf-8"))
    assert [case["case_directory"] for case in manifest["cases"]] == [
        "low_no_decay",
        "high_with_decay",
    ]
    request = json.loads((run / "v5_run_request.json").read_text(encoding="utf-8"))
    assert request["schema_version"] == (
        "ppg_frailty.v5_comparison_sequence_request.v1"
    )
    assert [
        row["case_id"]
        for row in request["comparison_sequence"]["ordered_case_bindings"]
    ] == ["low_no_decay", "high_with_decay"]


def test_comparison_sequence_rejects_damaged_cached_cli() -> None:
    cases = [
        {
            "name": "damaged",
            "script": "pipeline.py",
            "arguments": [
                "run",
                "--config",
                "configs/presets/finalcase.yaml",
                "--not-a-public-option",
            ],
        }
    ]
    with pytest.raises(ValueError, match="unsupported cached pipeline option"):
        comparison_sequence_cli(cases)


def test_comparison_yaml_preserves_common_fixed_set() -> None:
    base = ["run", "--config", "configs/presets/finalcase.yaml"]
    cases = [
        {
            "name": "case_a",
            "script": "pipeline.py",
            "arguments": [
                *base,
                "--set",
                "model.dropout=0.4",
                "--set",
                "training.batch_size=16",
            ],
        },
        {
            "name": "case_b",
            "script": "pipeline.py",
            "arguments": [
                *base,
                "--set",
                "model.dropout=0.4",
                "--set",
                "training.batch_size=32",
            ],
        },
    ]

    payload = yaml.safe_load(comparison_sequence_yaml(cases, pipeline_root=PIPELINE_ROOT))
    commands = [row["pipeline_request"]["arguments"] for row in payload["cases"]]
    assert all("model.dropout=0.4" in arguments for arguments in commands)


@pytest.mark.parametrize(
    "extra",
    [
        ["--module", "model=InceptionTimeSmall"],
        ["--unset", "model.gradient_clip_norm"],
        ["--config-id", "custom_case"],
    ],
)
def test_comparison_yaml_preserves_per_case_cli_controls(
    extra: list[str],
) -> None:
    base = ["run", "--config", "configs/presets/finalcase.yaml"]
    cases = [
        {
            "name": "case_a",
            "script": "pipeline.py",
            "arguments": [*base, *extra, "--set", "training.batch_size=16"],
        },
        {
            "name": "case_b",
            "script": "pipeline.py",
            "arguments": [*base, *extra, "--set", "training.batch_size=32"],
        },
    ]

    payload = yaml.safe_load(comparison_sequence_yaml(cases, pipeline_root=PIPELINE_ROOT))
    for row in payload["cases"]:
        arguments = row["pipeline_request"]["arguments"]
        start = arguments.index(extra[0])
        assert arguments[start : start + len(extra)] == extra


def test_comparison_yaml_preserves_representable_global_controls() -> None:
    base = [
        "run",
        "--config",
        "configs/presets/finalcase.yaml",
        "--study-id",
        "dash_study",
        "--purpose",
        "Reviewed dashboard comparison",
        "--device",
        "cuda",
        "--no-continue-on-error",
    ]
    cases = [
        {"name": "case_a", "arguments": [*base, "--set", "training.batch_size=16"]},
        {"name": "case_b", "arguments": [*base, "--set", "training.batch_size=32"]},
    ]

    payload = yaml.safe_load(
        comparison_sequence_yaml(cases, pipeline_root=PIPELINE_ROOT)
    )

    assert payload["study"]["study_id"] == "dash_study"
    assert payload["study"]["purpose"] == "Reviewed dashboard comparison"
    assert payload["execution"]["device"] == "cuda"
    assert payload["execution"]["continue_on_error"] is False


def test_sweep_request_uses_real_sweep_cli_and_validated_plan() -> None:
    service = V5ControlService(PIPELINE_ROOT)
    request = service.build_train_request(
        config_path=None,
        plan_path="configs/studies/finalcase.yaml",
        operation="sweep",
        run_name="finalcase_dash",
        dry_run=True,
    )

    assert request.script == "sweep.py"
    assert request.arguments[:3] == (
        "run",
        "--plan",
        "configs/studies/finalcase.yaml",
    )
    assert "--config" not in request.arguments
    assert request.arguments[-3:] == ("--run-name", "finalcase_dash", "--dry-run")
    assert yaml.safe_load(request.resolved_yaml)["study"]["study_id"] == "finalcase"


def test_analysis_automatic_name_is_owned_by_report_cli(tmp_path: Path) -> None:
    root = tmp_path / "repo" / "final_v0" / "final_pipeline_v5"
    common = root / "pipeline_output" / "run_a"
    first = common / "case_a"
    second = common / "case_b"
    first.mkdir(parents=True)
    second.mkdir()
    (common / "study_manifest.json").write_text("{}\n", encoding="utf-8")
    service = V5ControlService(root)

    request = service.build_analysis_request(
        run_paths=[first, second],
        mode="comparison",
        preset="classification",
        modules=[],
        figures=None,
        tables=None,
    )

    assert "--output-dir" not in request.arguments
    assert "--output-name" not in request.arguments
    assert request.arguments.count("--run") == 2


def test_report_validation_reuses_full_analysis_surface_without_output(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo" / "final_v0" / "final_pipeline_v5"
    source = root / "pipeline_output" / "legacy_case"
    source.mkdir(parents=True)
    service = V5ControlService(root)

    request = service.build_analysis_request(
        run_path="pipeline_output/legacy_case",
        mode="test",
        preset="classification",
        modules=["confusion"],
        figures=["roc_auc"],
        tables=[],
        validation_depth="selected",
        on_missing="error",
        command="validate",
    )
    parsed = build_report_parser().parse_args(list(request.arguments))

    assert parsed.command == "validate"
    assert parsed.mode == "test"
    assert parsed.module == ["confusion"]
    assert parsed.figure == ["roc_auc"]
    assert parsed.table == ["none"]
    assert not hasattr(parsed, "output_name")


def test_completed_workflow_stages_read_exact_model_and_oof_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo" / "final_v0" / "final_pipeline_v5"
    study = root / "pipeline_output" / "run_a"
    case = study / "reference"
    cell = case / "repeat_00" / "fold_02"
    cell.mkdir(parents=True)
    (study / "study_manifest.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "cases": [
                    {
                        "case_id": "reference",
                        "case_directory": "reference",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (case / "case_result.json").write_text(
        json.dumps({"status": "passed", "artifact_root": "."}),
        encoding="utf-8",
    )
    (cell / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "ppg_frailty.run_manifest.v2",
                "status": "passed",
                "cell": {
                    "status": "passed",
                    "repeat_index": 0,
                    "fold_index": 2,
                    "model_id": "InceptionTimeSmall",
                    "model_machine_id": "inception_small",
                    "representation_mode": "raw",
                    "model_hash": "model-hash",
                    "balance_line": "line_b_equal_role_families",
                    "metrics": {"balanced_accuracy": 0.75, "macro_f1": 0.7},
                    "model_factory_provenance": {"parameter_count": 12345},
                    "fitted_provenance": {"state_hash": "state-hash"},
                    "learned_model_checkpoint": {
                        "deployment_status": "research_only"
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    common = {
        "participant_id": "P01",
        "label": 0,
        "probabilities": (0.7, 0.2, 0.1),
        "repeat": 0,
        "fold": 2,
        "split_seed": 42,
        "training_seed": 42,
        "config_hash": "config",
        "manifest_hash": "manifest",
        "fold_hash": "fold",
        "preprocessing_hash": "preprocess",
        "feature_hash": "feature",
        "model_hash": "model-hash",
        "representation_mode": "raw",
        "signal_route": "identity_direct",
        "quality_score": 1.0,
        "retained": True,
        "class_order": (0, 1, 2),
        "aggregation_rule": "line_b_equal_role_families",
        "route_status": "retained",
    }
    write_oof_parquet(
        [OofPredictionRow(file_id="rec-1", role="B", level="file", **common)],
        cell / "oof_file_predictions.parquet",
    )
    write_oof_parquet(
        [
            OofPredictionRow(
                file_id="role::P01::B", role="B", level="role", **common
            )
        ],
        cell / "oof_role_predictions.parquet",
    )
    write_oof_parquet(
        [
            OofPredictionRow(
                file_id="participant::P01",
                role="participant",
                level="participant",
                **common,
            )
        ],
        cell / "oof_subject_predictions.parquet",
    )

    service = V5ControlService(root)
    rows = service.completed_workflow_stage_rows(
        "pipeline_output/run_a",
        record_id="rec-1",
        participant_id="P01",
    )
    by_metric = {row["metric"]: row for row in rows}

    assert {row["status"] for row in rows} == {"artifact"}
    assert by_metric[
        "reference.repeat_00.fold_02.model_id"
    ]["value"] == "InceptionTimeSmall"
    assert by_metric[
        "reference.repeat_00.fold_02.parameter_count"
    ]["value"] == 12345
    assert by_metric[
        "reference.repeat_00.fold_02.file.B.0.probabilities"
    ]["value"] == "[0.7,0.2,0.1]"
    assert by_metric[
        "reference.repeat_00.fold_02.participant.participant.0.aggregation_rule"
    ]["value"] == "line_b_equal_role_families"

    unavailable = service.completed_workflow_stage_rows(
        None,
        record_id="rec-1",
        participant_id="P01",
    )
    assert {row["stage"] for row in unavailable} == {
        "representation_model",
        "aggregation",
    }
    assert {row["status"] for row in unavailable} == {"N/A"}


def test_model_config_defaults_and_bundle_capability_are_discovered(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo" / "final_v0" / "final_pipeline_v5"
    case = root / "model_config" / "run_a" / "cases" / "finalcase"
    bundle = root / "model_config" / "run_a" / "bundle"
    case.mkdir(parents=True)
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "pipeline_adapter_boundary": (
                    "serialized_raw_record_to_model_input_mapping"
                )
            }
        ),
        encoding="utf-8",
    )
    (case / "resolved_pipeline_config.yaml").write_text(
        yaml.safe_dump(
            {
                "config_id": "finalcase",
                "features": {"enabled_groups": ["morphology", "dual_optical"]},
            }
        ),
        encoding="utf-8",
    )
    (case / "pipeline_module_defaults.yaml").write_text(
        yaml.safe_dump(
            {
                "derived_module_defaults": [
                    {"family": "model", "module_id": "inception_small"}
                ]
            }
        ),
        encoding="utf-8",
    )
    export = root / "model_config" / "run_a"
    (export / "export_manifest.json").write_text(
        json.dumps(
            {
                "capabilities": {
                    "new_participant_inference_available": True,
                    "new_participant_inference_status": (
                        "available_through_serialized_raw_input_adapter"
                    ),
                },
                "cases": [
                    {
                        "case_id": "finalcase",
                        "directory": "cases/finalcase",
                        "resolved_config": "cases/finalcase/resolved_pipeline_config.yaml",
                        "bundle_path": "bundle",
                        "new_participant_inference": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    service = V5ControlService(root)

    defaults = service.load_model_defaults("model_config/run_a", "finalcase")

    assert defaults.module_defaults == {"model": "inception_small"}
    assert defaults.feature_defaults == ("morphology", "dual_optical")
    assert defaults.inference_capability["available"] is True
    assert (
        defaults.inference_capability["adapter_source"]
        == "v5_frozen_raw_workflow_service"
    )


def test_model_checkpoint_without_raw_adapter_stays_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "repo" / "final_v0" / "final_pipeline_v5"
    case = root / "model_config" / "run_a" / "cases" / "finalcase"
    bundle = case / "learned_model"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(
        json.dumps({"pipeline_adapter_boundary": "model_ready_tensor_only"}),
        encoding="utf-8",
    )
    (case / "resolved_pipeline_config.yaml").write_text(
        yaml.safe_dump({"config_id": "finalcase", "features": {"enabled_groups": []}}),
        encoding="utf-8",
    )
    export = root / "model_config" / "run_a"
    (export / "export_manifest.json").write_text(
        json.dumps(
            {
                "capabilities": {
                    "new_participant_inference_available": False,
                    "new_participant_inference_status": (
                        "model_ready_input_only_raw_adapter_not_bundled"
                    ),
                },
                "cases": [
                    {
                        "case_id": "finalcase",
                        "directory": "cases/finalcase",
                        "resolved_config": "cases/finalcase/resolved_pipeline_config.yaml",
                        "bundle_path": "cases/finalcase/learned_model",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    defaults = V5ControlService(root).load_model_defaults(
        "model_config/run_a", "finalcase"
    )

    assert defaults.inference_capability["available"] is False
    assert "model_ready_input_only" in defaults.inference_capability["reason"]


def test_job_manager_never_uses_a_shell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo" / "final_v0" / "final_pipeline_v5"
    root.mkdir(parents=True)
    (root / "pipeline.py").write_text("pass\n", encoding="utf-8")
    (root / "specialized_pipeline.py").write_text("pass\n", encoding="utf-8")
    (root / "comparison_sequence.py").write_text("pass\n", encoding="utf-8")
    observed: dict[str, object] = {}

    class FakeProcess:
        pid = 321

        def poll(self) -> None:
            return None

        def wait(self, timeout: float) -> int:
            assert timeout == 5.0
            return 0

    def fake_popen(command: object, **kwargs: object) -> FakeProcess:
        observed["command"] = command
        observed.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr("ppg_frailty.dashboard.job_manager.subprocess.Popen", fake_popen)
    manager = DashboardJobManager(root)
    job_id = manager.start("pipeline.py", ["run", "--config", "configs/a.yaml"])

    assert job_id
    assert observed["shell"] is False
    assert isinstance(observed["command"], tuple)

    specialized_id = manager.start(
        "specialized_pipeline.py",
        ["complete", "--study-dir", "pipeline_output/preserved"],
        kind="tool",
    )
    assert specialized_id
    command = observed["command"]
    assert isinstance(command, tuple)
    assert Path(command[1]).name == "specialized_pipeline.py"
    assert command[2:] == (
        "complete",
        "--study-dir",
        "pipeline_output/preserved",
    )

    sequence_id = manager.start(
        "comparison_sequence.py",
        [
            "run",
            "--request",
            "pipeline_output/.dashboard_requests/comparison/sequence_a.yaml",
        ],
        kind="pipeline",
    )
    command = observed["command"]
    assert isinstance(command, tuple)
    assert Path(command[1]).name == "comparison_sequence.py"
    assert command[2] == "run"

    terminated: list[tuple[int, object]] = []
    monkeypatch.setattr("ppg_frailty.dashboard.job_manager.os.getpgid", lambda pid: pid)
    monkeypatch.setattr(
        "ppg_frailty.dashboard.job_manager.os.killpg",
        lambda group, sent_signal: terminated.append((group, sent_signal)),
    )
    manager.terminate(sequence_id)
    assert terminated and terminated[0][0] == 321


def test_dash_layout_smoke_and_button_labels() -> None:
    pytest.importorskip("dash")
    from ppg_frailty.dashboard import create_app
    from ppg_frailty.dashboard.app import STAGES

    app = create_app(PIPELINE_ROOT)
    client = app.server.test_client()
    for endpoint in ("/", "/_dash-layout", "/_dash-dependencies"):
        assert client.get(endpoint).status_code == 200
    assert len(app.callback_map) >= 20
    layout = _layout(app)
    all_components = list(layout._traverse())
    buttons = [c for c in all_components if c.__class__.__name__ == "Button"]
    labels = [str(c.children) for c in buttons]
    assert {"Analyse", "Run", "Stop", "Add", "Download CLI", "Download YAML"} <= set(labels)
    assert all(len(label) <= 20 for label in labels)
    assert labels.count("Run") == 1
    assert not {"Infer", "Run queue", "Run CLI", "Run YAML", "Run tool"} & set(labels)
    stages = [c.id["stage"] for c in buttons if isinstance(getattr(c, "id", None), dict)
              and c.id.get("type") == "analyse-stage"]
    assert stages == [stage for stage, _ in STAGES]
    components = {c.id: c for c in all_components if isinstance(getattr(c, "id", None), str)}
    assert components["model-mode"].value == "Analyse"
    assert components["train"].style == {"display": "none"}
    assert not getattr(components["stop-train"], "disabled", False)
    assert components["training-yaml"].value is None
    assert {"comparison-cli-view", "comparison-yaml-view", "command-view", "train-command", "yaml-view"} <= set(components)
    assert "download-sequence-cli.data" in app.callback_map
    assert "download-sequence-yaml.data" in app.callback_map
    assert str(layout.style["padding"]).startswith("clamp(")
    assert "grid-template-columns:1fr" in app.index_string
