from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import yaml

from ppg_frailty.quality.stage5_pre import run_motion_peak_study
from ppg_frailty.study.hyperparameter import (
    _phase_resume_directory,
    _phase_run_name,
    complete_successive_halving_study,
    run_hyperparameter_study,
)
import ppg_frailty.v5.specialized as specialized
import ppg_frailty.v5.specialized_outputs as specialized_outputs
import ppg_frailty.study.hyperparameter as hyperparameter
from ppg_frailty.v5_reporting.cli import build_parser as build_report_parser


ROOT = Path(__file__).resolve().parents[1]
SPECIALIZED_PLANS = tuple(
    path
    for path in sorted((ROOT / "configs/studies").rglob("*.yaml"))
    if yaml.safe_load(path.read_text(encoding="utf-8")).get("schema_version")
    in specialized.SUPPORTED_SCHEMAS
)


class _Sink:
    def close(self) -> None:
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _motion_export_fixture(
    tmp_path: Path,
) -> Path:
    import torch

    def write_checkpoint(
        path: Path,
        *,
        dataset: str,
        marker: int,
        schema_sha256: str,
        roster: list[str],
        inference_cost: dict[str, float],
    ) -> None:
        torch.save(
            {
                "schema_version": (
                    "ppg_frailty.formal_motion_model_artifact."
                    "imu_iqr_over_1p349.v3"
                ),
                "model_id": "formal_motion_cnn_v1",
                "model_input_schema_sha256": schema_sha256,
                "trainer_config": {"device": "cuda:0", "epochs": 1},
                "training_participant_ids": roster,
                "final_training_loss": 0.1,
                "state_dict": {"weight": torch.tensor([marker])},
                "imu_transform": {"dataset": dataset},
                "inference_cost": inference_cost,
            },
            path,
        )

    output = tmp_path / "pipeline_output" / "motion-run"
    output.mkdir(parents=True)
    plan = {
        "schema_version": specialized.STAGE5_SCHEMA,
        "study_type": "stage5_pre_motion_ptt",
        "study_id": "fixture",
        "motion_detector": {"training_device": "cuda:0"},
    }
    (output / "resolved_plan.yaml").write_text(
        yaml.safe_dump(plan, sort_keys=False), encoding="utf-8"
    )
    stages = {}
    specifications = (
        (
            "frailty29",
            "internal_motion_oof",
            "motion_internal",
            "motion_internal_evidence.json",
            "final_threshold",
            "final_threshold_artifact_sha256",
            "final_all_internal",
        ),
        (
            "ptt22",
            "ptt_motion_training_ablation",
            "motion_ptt_training",
            "motion_ptt_training_evidence.json",
            "deployment_threshold",
            "deployment_threshold_artifact_sha256",
            "final_all_ptt",
        ),
    )
    for (
        dataset,
        stage_name,
        directory_name,
        evidence_name,
        final_threshold_name,
        final_threshold_hash_name,
        final_directory_name,
    ) in specifications:
        directory = output / directory_name
        directory.mkdir()
        schema = directory / "formal_motion_input_schema.json"
        _write_json(schema, {"dataset": dataset, "channels": ["imu_norm"]})
        semantic_schema_sha256 = hashlib.sha256(
            f"semantic:{dataset}".encode("utf-8")
        ).hexdigest()
        cells = []
        for fold in range(5):
            model_directory = directory / "repeat_0" / f"fold_{fold}"
            model_directory.mkdir(parents=True)
            model = model_directory / "formal_motion_model.pt"
            training_roster = [f"{dataset}-train-0", f"{dataset}-train-1"]
            inference_cost = {"median_batch_latency_ms": 1.0 + fold}
            write_checkpoint(
                model,
                dataset=dataset,
                marker=fold,
                schema_sha256=semantic_schema_sha256,
                roster=training_roster,
                inference_cost=inference_cost,
            )
            _write_json(
                model_directory / "motion_training_history.json",
                {"dataset": dataset, "repeat": 0, "fold": fold},
            )
            threshold = {
                "threshold": 0.2 + fold / 100.0,
                "training_participant_ids": [f"{dataset}-train"],
            }
            cells.append(
                {
                    "repeat_index": 0,
                    "fold_index": fold,
                    "model_artifact_path": str(model),
                    "model_artifact_sha256": _sha256(model),
                    "model_input_schema_sha256": semantic_schema_sha256,
                    "parameter_count": 10 + fold,
                    "training_participant_ids": training_roster,
                    "training_participant_count": len(training_roster),
                    "inference_cost": inference_cost,
                    "threshold": threshold,
                    "threshold_artifact_sha256": specialized_outputs._payload_sha256(
                        threshold
                    ),
                }
            )
        final_directory = directory / final_directory_name
        final_directory.mkdir()
        final_model = final_directory / "formal_motion_model.pt"
        final_roster = [f"{dataset}-all"]
        final_inference_cost = {"median_batch_latency_ms": 2.0}
        write_checkpoint(
            final_model,
            dataset=dataset,
            marker=99,
            schema_sha256=semantic_schema_sha256,
            roster=final_roster,
            inference_cost=final_inference_cost,
        )
        _write_json(
            final_directory / "motion_training_history.json",
            {"dataset": dataset, "final_fit": True},
        )
        final_threshold = {
            "threshold": 0.4,
            "training_participant_ids": [f"{dataset}-all"],
        }
        evidence = {
            "schema_version": f"fixture.{dataset}.v1",
            "model_id": "formal_motion_cnn_v1",
            "model_input_schema_path": str(schema),
            "model_input_schema_file_sha256": _sha256(schema),
            "model_input_schema_sha256": semantic_schema_sha256,
            "cell_evidence": cells,
            final_threshold_name: final_threshold,
            final_threshold_hash_name: specialized_outputs._payload_sha256(
                final_threshold
            ),
            "final_model": {
                "artifact_path": str(final_model),
                "artifact_sha256": _sha256(final_model),
                "model_input_schema_sha256": semantic_schema_sha256,
                "parameter_count": 20,
                "training_participant_ids": final_roster,
                "inference_cost": final_inference_cost,
            },
        }
        evidence_path = directory / evidence_name
        _write_json(evidence_path, evidence)
        stages[stage_name] = {
            "status": "passed",
            "artifact_dir": directory.relative_to(output).as_posix(),
            "evidence_sha256": _sha256(evidence_path),
        }
    _write_json(
        output / "study_manifest.json",
        {
            "schema_version": specialized.MOTION_PEAK_RESULT_SCHEMA,
            "study_type": "stage5_pre_motion_ptt",
            "status": "passed",
            "training_device": "cuda:0",
            "denoiser_enabled": True,
            "stages": stages,
        },
    )
    return output



def test_all_ten_preserved_specialized_plans_use_their_native_loader() -> None:
    assert len(SPECIALIZED_PLANS) == 10
    descriptions = [specialized.validate_specialized_plan(path) for path in SPECIALIZED_PLANS]
    assert {row["schema_version"] for row in descriptions} == specialized.SUPPORTED_SCHEMAS
    assert sum(row["workflow_kind"] == "analysis_only" for row in descriptions) == 5
    assert sum(row["workflow_kind"] == "computation" for row in descriptions) == 5


def test_training_entry_points_default_to_data_only() -> None:
    assert "generate_report" not in inspect.signature(run_motion_peak_study).parameters
    assert "generate_reports" not in inspect.signature(run_hyperparameter_study).parameters
    assert (
        "generate_reports"
        not in inspect.signature(complete_successive_halving_study).parameters
    )


def test_motion_adapter_calls_native_runner_without_report_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline_output = tmp_path / "pipeline_output"
    pipeline_output.mkdir()
    monkeypatch.setattr(specialized, "PIPELINE_OUTPUT_ROOT", pipeline_output)
    monkeypatch.setattr(specialized, "TerminalProgressSink", _Sink)
    calls = {}

    def fake_run(plan, **kwargs):
        calls.update(kwargs)
        return Path(kwargs["resume"])

    monkeypatch.setattr(specialized, "run_motion_peak_study", fake_run)
    monkeypatch.setattr(
        specialized, "_publish_specialized_artifact_contract",
        lambda *args, **kwargs: {},
    )
    plan = next(
        path for path in SPECIALIZED_PLANS
        if yaml.safe_load(path.read_text())["schema_version"] == specialized.PEAK_ABLATION_SCHEMA
    )
    output = specialized.run_specialized_computation(plan, run_name="static-peaks")
    assert output == pipeline_output / "static-peaks"
    assert "generate_report" not in calls
    assert calls["include_denoiser"] is True


def test_stage5_dispatch_forwards_scientific_options_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline_output = tmp_path / "pipeline_output"
    pipeline_output.mkdir()
    monkeypatch.setattr(specialized, "PIPELINE_OUTPUT_ROOT", pipeline_output)
    monkeypatch.setattr(specialized, "TerminalProgressSink", _Sink)
    observed = {}

    def fake_run(plan, **kwargs):
        observed.update(kwargs)
        return Path(kwargs["resume"])

    monkeypatch.setattr(specialized, "run_motion_peak_study", fake_run)
    monkeypatch.setattr(
        specialized, "_publish_specialized_artifact_contract",
        lambda output, schema, **kwargs: observed.update(final=(output, schema)),
    )
    plan = next(
        path for path in SPECIALIZED_PLANS
        if yaml.safe_load(path.read_text())["schema_version"] == specialized.STAGE5_SCHEMA
    )
    output = specialized.run_specialized_computation(
        plan, run_name="stage5", device="cuda:0", include_denoiser=False
    )
    assert output == pipeline_output / "stage5"
    assert observed["device"] == "cuda:0"
    assert observed["include_denoiser"] is False
    assert observed["final"] == (output, specialized.STAGE5_SCHEMA)


def test_stage5_motion_export_copies_all_fold_and_final_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline_output = tmp_path / "pipeline_output"
    model_config = tmp_path / "model_config"
    monkeypatch.setattr(specialized_outputs, "V5_ROOT", tmp_path)
    monkeypatch.setattr(specialized_outputs, "PIPELINE_OUTPUT_ROOT", pipeline_output)
    monkeypatch.setattr(specialized_outputs, "MODEL_CONFIG_ROOT", model_config)
    output = _motion_export_fixture(tmp_path)
    result = specialized_outputs.export_motion_model_config(output)
    target = tmp_path / str(result["output_directory"])
    manifest = json.loads((target / "export_manifest.json").read_text())
    assert result["model_count"] == 12
    assert manifest["outer_fold_model_count"] == 10
    assert manifest["final_model_count"] == 2
    assert len(list((target / "learned_models").rglob("formal_motion_model.pt"))) == 12
    assert len(list((target / "input_schemas").glob("*.json"))) == 2
    reuse = yaml.safe_load((target / "model_reuse_parameters.yaml").read_text())
    assert reuse["model_count"] == 12
    assert all(row["exported_input_schema"] for row in reuse["models"])
    assert all(row["loader_metadata"]["training_participant_ids"] for row in reuse["models"])


def test_motion_export_rejects_changed_learned_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(specialized_outputs, "V5_ROOT", tmp_path)
    monkeypatch.setattr(
        specialized_outputs, "PIPELINE_OUTPUT_ROOT", tmp_path / "pipeline_output"
    )
    monkeypatch.setattr(
        specialized_outputs, "MODEL_CONFIG_ROOT", tmp_path / "model_config"
    )
    output = _motion_export_fixture(tmp_path)
    model = next(output.rglob("formal_motion_model.pt"))
    model.write_bytes(model.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        specialized_outputs.export_motion_model_config(output)


def test_pipeline_adapter_does_not_scan_or_reject_presentation_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline_output = tmp_path / "pipeline_output"
    pipeline_output.mkdir()
    monkeypatch.setattr(specialized, "PIPELINE_OUTPUT_ROOT", pipeline_output)
    monkeypatch.setattr(specialized, "TerminalProgressSink", _Sink)

    def fake_run(plan, **kwargs):
        output = Path(kwargs["resume"])
        (output / "historical-note.md").write_text("kept", encoding="utf-8")
        return output

    monkeypatch.setattr(specialized, "run_motion_peak_study", fake_run)
    monkeypatch.setattr(
        specialized, "_publish_specialized_artifact_contract",
        lambda *args, **kwargs: {},
    )
    plan = next(
        path for path in SPECIALIZED_PLANS
        if yaml.safe_load(path.read_text())["schema_version"] == specialized.PEAK_ABLATION_SCHEMA
    )
    output = specialized.run_specialized_computation(plan, run_name="no-scan")
    assert (output / "historical-note.md").read_text() == "kept"


def test_hyperparameter_contract_finalizes_every_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    output = root / "pipeline_output" / "hyper-run"
    phase = output / "phases" / "full"
    phase.mkdir(parents=True)
    _write_json(
        output / "study_manifest.json",
        {
            "schema_version": "ppg_frailty.hyperparameter_study_manifest.v1",
            "phase_directories": {"full_cv": "phases/full"},
        },
    )
    monkeypatch.setattr(specialized, "V5_ROOT", root)
    monkeypatch.setattr(
        specialized, "export_specialized_data_excel",
        lambda *args, **kwargs: {"workbook": "tables/pipeline_data.xlsx"},
    )
    finalized = []
    monkeypatch.setattr(
        specialized, "post_run_finalize",
        lambda path, **kwargs: finalized.append(Path(path))
        or {"model_config_export": {"output_directory": "model_config/full"}},
    )
    monkeypatch.setattr(
        specialized, "try_export_pipeline_excel",
        lambda *args, **kwargs: {"workbook": "tables/pipeline_data.xlsx"},
    )
    result = specialized._publish_specialized_artifact_contract(
        output, specialized.HYPERPARAMETER_SCHEMA
    )
    assert finalized == [phase]
    assert result["public_phase_runs"][0]["phase"] == "full_cv"


def test_hyperparameter_contract_has_no_layout_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "pipeline_output" / "hyper-run"
    phase = output / "legacy-phase"
    phase.mkdir(parents=True)
    _write_json(
        output / "study_manifest.json",
        {
            "schema_version": "ppg_frailty.hyperparameter_study_manifest.v1",
            "phase_directories": {"full_cv": "legacy-phase"},
        },
    )
    monkeypatch.setattr(specialized, "V5_ROOT", tmp_path)
    monkeypatch.setattr(
        specialized, "export_specialized_data_excel",
        lambda *args, **kwargs: {"workbook": "tables/pipeline_data.xlsx"},
    )
    monkeypatch.setattr(
        specialized, "post_run_finalize",
        lambda *args, **kwargs: {"model_config_export": {}},
    )
    monkeypatch.setattr(
        specialized, "try_export_pipeline_excel",
        lambda *args, **kwargs: {"workbook": "tables/pipeline_data.xlsx"},
    )
    result = specialized._publish_specialized_artifact_contract(
        output, specialized.HYPERPARAMETER_SCHEMA
    )
    assert result["status"] == "complete"


def test_hyperparameter_phase_resume_resolution_and_name_are_deterministic(
    tmp_path: Path,
) -> None:
    output = tmp_path / "outer-run"
    nested = output / "phases/screen/one-public-run"
    nested.mkdir(parents=True)
    (nested / "study_plan.yaml").write_text("schema_version: test\n", encoding="utf-8")
    assert _phase_resume_directory(output, "screen") == nested.resolve()
    first = _phase_run_name(output, "screen")
    assert first == _phase_run_name(output, "screen")
    assert first.startswith("outer-run__screen__")


def test_specialized_phase_factory_uses_ordinary_v5_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class Runner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(specialized, "StudyRunner", Runner)
    factory = specialized._specialized_phase_runner_factory(
        source_yaml=ROOT / "configs/studies/example.yaml", sink=_Sink()
    )
    result = factory(object(), "screen", None)
    assert isinstance(result, Runner)
    assert captured["output_layout"] == "v5"


def test_stage_report_reads_source_without_mutating_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ppg_frailty.reporting.specialized as reporter

    report_output = tmp_path / "report_output"
    source = tmp_path / "pipeline_output" / "stage5-run"
    source.mkdir(parents=True)
    _write_json(
        source / "study_manifest.json",
        {
            "schema_version": specialized.MOTION_PEAK_RESULT_SCHEMA,
            "status": "passed",
            "study_id": "stage5-run",
        },
    )
    monkeypatch.setattr(specialized, "REPORT_OUTPUT_ROOT", report_output)
    before = (source / "study_manifest.json").read_bytes()

    def fake_report(study, *, output_dir):
        target = Path(output_dir)
        target.mkdir()
        (target / "STUDY_SUMMARY.md").write_text("summary\n", encoding="utf-8")
        return {"status": "complete"}

    monkeypatch.setattr(reporter, "generate_motion_peak_report", fake_report)
    output = specialized.rebuild_specialized_report(source)
    assert (source / "study_manifest.json").read_bytes() == before
    assert (output / "STUDY_SUMMARY.md").is_file()


def test_static_peak_report_builds_tables_figures_and_excel(tmp_path: Path) -> None:
    from ppg_frailty.reporting.specialized import generate_motion_peak_report

    source = tmp_path / "static"
    stage = source / "static_peak_ablation"
    stage.mkdir(parents=True)
    (source / "resolved_plan.yaml").write_text("report: {}\n", encoding="utf-8")
    _write_json(
        source / "study_manifest.json",
        {
            "study_id": "static",
            "study_type": "stage_ablation_01_static_peak_detectors",
            "stages": {"static_peak_ablation": {"artifact_dir": stage.name}},
        },
    )
    rows = [
        {
            "algorithm_or_reducer": algorithm,
            "f1_percent": 90 + index,
            "sensitivity_percent": 89 + index,
            "positive_predictive_value_percent": 91 + index,
            "ibi_ppi_rmse_ms": 12 - index,
            "execution_time_percent": 1 + index,
        }
        for index, algorithm in enumerate(("aboy_project", "msptdfast_v2"))
    ]
    _write_json(
        stage / "static_peak_ablation.json",
        {"rows": rows, "summary_rows": rows, "statistical_comparisons": []},
    )
    output = tmp_path / "report"
    manifest = generate_motion_peak_report(source, output_dir=output)
    assert {"rows", "summary_rows", "statistical_comparisons"} <= set(
        manifest["tables"]
    )
    assert (output / "tables/report_tables.xlsx").is_file()
    assert (output / "figures/static_peak_f1_percent.png").is_file()
    canonical = {
        "static_peak_detector_f1",
        "static_peak_detector_sensitivity",
        "static_peak_detector_ppv",
        "static_peak_detector_interval_rmse",
        "static_peak_detector_runtime",
    }
    assert canonical <= {path.stem for path in (output / "figures").glob("*.png")}


def test_hyperparameter_report_uses_persisted_rankings(tmp_path: Path) -> None:
    from ppg_frailty.reporting.specialized import generate_hyperparameter_report

    source = tmp_path / "hyper"
    (source / "tables").mkdir(parents=True)
    (source / "study_plan.yaml").write_text(
        "study:\n  study_id: hyper\nresource:\n  ranking_metric: balanced_accuracy\n",
        encoding="utf-8",
    )
    _write_json(
        source / "study_manifest.json",
        {"ranking_tables": ["full_cv_ranking"]},
    )
    ranking = [
        {
            "rank": 1,
            "case_id": "candidate",
            "balanced_accuracy_mean": 0.8,
            "balanced_accuracy_sd": 0.02,
        }
    ]
    _write_json(source / "tables/full_cv_ranking.json", ranking)
    output = tmp_path / "report"
    result = generate_hyperparameter_report(source, output_dir=output)
    assert "full_cv_ranking" in result["tables"]
    assert "v2_v5_specialized_inventory" in result["tables"]
    assert (output / "figures/full_cv_ranking.png").is_file()
    assert (output / "STUDY_SUMMARY.md").is_file()


def test_specialized_data_excel_contains_inventory_and_csv_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline_output = tmp_path / "pipeline_output"
    output = pipeline_output / "static-run"
    output.mkdir(parents=True)
    (output / "values.csv").write_text("record,value\na,1\nb,2\n", encoding="utf-8")
    _write_json(output / "metadata.json", {"status": "passed"})
    monkeypatch.setattr(specialized_outputs, "PIPELINE_OUTPUT_ROOT", pipeline_output)
    result = specialized_outputs.export_specialized_data_excel(output)
    workbook = output / str(result["workbook"])
    assert result["status"] == "complete"
    assert result["included_csv_tables"] == [{"path": "values.csv", "rows": 2}]
    with zipfile.ZipFile(workbook) as archive:
        assert archive.testzip() is None
        assert "xl/workbook.xml" in archive.namelist()
    assert not any(
        path.suffix.lower() in {".png", ".svg", ".html", ".md"}
        for path in output.rglob("*")
        if path.is_file()
    )


def test_hyperparameter_adapter_disables_nested_and_root_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline_output = tmp_path / "pipeline_output"
    pipeline_output.mkdir()
    monkeypatch.setattr(specialized, "PIPELINE_OUTPUT_ROOT", pipeline_output)
    monkeypatch.setattr(specialized, "TerminalProgressSink", _Sink)
    calls = {}
    contract_calls = []

    def fake_run(plan, **kwargs):
        calls.update(kwargs)
        output = Path(kwargs["output_root"]) / str(kwargs["run_name"])
        output.mkdir()
        return output

    monkeypatch.setattr(specialized, "run_hyperparameter_study", fake_run)
    monkeypatch.setattr(
        specialized,
        "_publish_specialized_artifact_contract",
        lambda output, schema, **kwargs: contract_calls.append((output, schema)),
    )
    plan = next(
        path
        for path in SPECIALIZED_PLANS
        if path.name == "stage6_batch_LR_search.yaml"
    )
    output = specialized.run_specialized_computation(
        plan,
        run_name="halving",
        environment_evidence={"status": "passed"},
    )
    assert output == pipeline_output / "halving"
    assert "generate_reports" not in calls
    assert calls["run_name"] == "halving"
    assert calls["resume"] is None
    assert callable(calls["phase_runner_factory"])
    assert contract_calls == [(output, specialized.HYPERPARAMETER_SCHEMA)]


def test_hyperparameter_adapter_uses_yaml_stem_utc_default_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline_output = tmp_path / "pipeline_output"
    pipeline_output.mkdir()
    monkeypatch.setattr(specialized, "PIPELINE_OUTPUT_ROOT", pipeline_output)
    monkeypatch.setattr(specialized, "TerminalProgressSink", _Sink)
    monkeypatch.setattr(
        specialized,
        "automatic_run_name",
        lambda source: f"{Path(source).stem}_20990101_000000Z",
    )
    captured = {}

    def fake_run(plan, **kwargs):
        captured.update(kwargs)
        output = Path(kwargs["output_root"]) / str(kwargs["run_name"])
        output.mkdir()
        return output

    monkeypatch.setattr(specialized, "run_hyperparameter_study", fake_run)
    monkeypatch.setattr(
        specialized,
        "_publish_specialized_artifact_contract",
        lambda *args, **kwargs: {},
    )
    plan = next(
        path
        for path in SPECIALIZED_PLANS
        if path.name == "stage6_batch_LR_search.yaml"
    )
    result = specialized.run_specialized_computation(
        plan, environment_evidence={"status": "passed"}
    )
    assert captured["run_name"] == "stage6_batch_LR_search_20990101_000000Z"
    assert result.name == captured["run_name"]


def test_hyperparameter_adapter_forwards_resume_to_native_orchestrator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline_output = tmp_path / "pipeline_output"
    resume = pipeline_output / "halving"
    resume.mkdir(parents=True)
    monkeypatch.setattr(specialized, "PIPELINE_OUTPUT_ROOT", pipeline_output)
    monkeypatch.setattr(specialized, "TerminalProgressSink", _Sink)
    calls = {}

    def fake_run(plan, **kwargs):
        calls.update(kwargs)
        return Path(kwargs["resume"])

    monkeypatch.setattr(specialized, "run_hyperparameter_study", fake_run)
    monkeypatch.setattr(
        specialized,
        "_publish_specialized_artifact_contract",
        lambda *args, **kwargs: {},
    )
    plan = next(
        path
        for path in SPECIALIZED_PLANS
        if path.name == "stage6_batch_LR_search.yaml"
    )
    output = specialized.run_specialized_computation(
        plan,
        resume=resume,
        environment_evidence={"status": "passed"},
    )
    assert output == resume
    assert calls["resume"] == resume
    assert calls["run_name"] is None


def test_schema_specific_options_fail_closed_instead_of_being_ignored() -> None:
    hyper = next(
        path for path in SPECIALIZED_PLANS if path.name == "stage6_batch_LR_search.yaml"
    )
    static_peak = next(
        path
        for path in SPECIALIZED_PLANS
        if path.name == "stage_ablation_01_static_peak_detectors.yaml"
    )
    with pytest.raises(ValueError, match="no-denoiser"):
        specialized.run_specialized_computation(hyper, include_denoiser=False)
    with pytest.raises(ValueError, match="Stage5-pre"):
        specialized.run_specialized_computation(static_peak, device="cuda")


def test_hyperparameter_meta_run_resumes_each_existing_v5_phase_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_phase(plan, **kwargs):
        phase = (
            Path(kwargs["resume_directory"])
            if kwargs["resume_directory"] is not None
            else Path(kwargs["output"])
            / "phases"
            / kwargs["phase_id"]
            / kwargs["run_name"]
        )
        phase.mkdir(parents=True, exist_ok=True)
        (phase / "study_plan.yaml").write_text(
            "schema_version: ppg_frailty.study_plan.v2\n", encoding="utf-8"
        )
        calls.append(
            {
                "phase_id": kwargs["phase_id"],
                "resume": kwargs["resume_directory"],
                "run_name": kwargs["run_name"],
            }
        )
        metric = str(plan["resource"]["ranking_metric"])
        tie_break = str(plan["resource"]["tie_break_metric"])
        ranking = [
            {
                "case_id": str(candidate["case_id"]),
                "cell_count": len(kwargs["repeats"]) * len(kwargs["folds"]),
                f"{metric}_mean": 1.0 - index / 100.0,
                f"{metric}_sd": 0.0,
                f"{metric}_percent_mean_sd": "100.0 ± 0.0",
                f"{tie_break}_mean": 1.0 - index / 100.0,
                f"{tie_break}_sd": 0.0,
                f"{tie_break}_percent_mean_sd": "100.0 ± 0.0",
                "rank": index + 1,
            }
            for index, candidate in enumerate(kwargs["candidates"])
        ]
        return SimpleNamespace(status="passed"), phase, ranking

    def fake_resolved(phase, case_id):
        path = Path(phase) / f"{case_id}.yaml"
        path.write_text("config_id: test\n", encoding="utf-8")
        return path, {"config_id": "test"}

    monkeypatch.setattr(hyperparameter, "_run_phase", fake_phase)
    monkeypatch.setattr(hyperparameter, "_resolved_config", fake_resolved)
    plan = next(
        path
        for path in SPECIALIZED_PLANS
        if path.name == "stage6_batch_LR_search.yaml"
    )
    factory = lambda *args: None
    output = hyperparameter.run_hyperparameter_study(
        plan,
        pipeline_root=ROOT,
        output_root=tmp_path,
        run_name="resume-contract",
        phase_runner_factory=factory,
    )
    assert all(row["resume"] is None for row in calls)
    calls.clear()
    resumed = hyperparameter.run_hyperparameter_study(
        plan,
        pipeline_root=ROOT,
        output_root=tmp_path,
        resume=output,
        phase_runner_factory=factory,
    )
    assert resumed == output
    assert calls
    assert all(row["resume"] is not None for row in calls)
    assert all(row["run_name"] is None for row in calls)


def test_completed_halving_command_only_recovers_public_products(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline_output = tmp_path / "pipeline_output"
    output = pipeline_output / "halving"
    output.mkdir(parents=True)
    monkeypatch.setattr(specialized, "PIPELINE_OUTPUT_ROOT", pipeline_output)
    monkeypatch.setattr(specialized, "TerminalProgressSink", _Sink)
    monkeypatch.setattr(
        specialized,
        "inspect_successive_halving_completion",
        lambda path: {"status": "already_complete"},
    )
    monkeypatch.setattr(
        specialized,
        "complete_successive_halving_study",
        lambda *args, **kwargs: pytest.fail("completed study must not retrain"),
    )
    published = []
    monkeypatch.setattr(
        specialized,
        "_publish_specialized_artifact_contract",
        lambda path, schema, **kwargs: published.append((path, schema)),
    )
    result = specialized.complete_specialized_halving(
        output,
        environment_evidence={"status": "passed"},
    )
    assert result == output
    assert published == [(output, specialized.HYPERPARAMETER_SCHEMA)]


def test_analysis_adapter_atomically_publishes_below_report_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_output = tmp_path / "report_output"
    monkeypatch.setattr(specialized, "REPORT_OUTPUT_ROOT", report_output)

    def fake_oracle(plan, **kwargs):
        generated = Path(kwargs["output_root"]) / "timestamped-v2-name"
        generated.mkdir()
        (generated / "STUDY_SUMMARY.md").write_text("result\n", encoding="utf-8")
        return generated

    monkeypatch.setattr(specialized, "run_decision_bias_oracle", fake_oracle)
    plan = next(path for path in SPECIALIZED_PLANS if path.name == "stage0_decision_bias_oracle.yaml")
    output = specialized.run_specialized_analysis(
        plan,
        source_root=ROOT,
        output_name="oracle-report",
    )
    assert output == report_output / "oracle-report"
    assert (output / "STUDY_SUMMARY.md").read_text() == "result\n"
    assert not any(path.name.startswith(".specialized-analysis-") for path in report_output.iterdir())


def test_report_cli_exposes_all_specialized_routes() -> None:
    parser = build_report_parser()
    assert parser.parse_args(
        ["specialized-validate", "--plan", "plan.yaml"]
    ).command == "specialized-validate"
    assert parser.parse_args(
        ["specialized-run", "--plan", "plan.yaml"]
    ).command == "specialized-run"
    assert parser.parse_args(
        ["specialized-report", "--input", "pipeline_output/run"]
    ).command == "specialized-report"
