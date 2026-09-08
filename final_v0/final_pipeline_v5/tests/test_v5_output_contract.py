from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from ppg_frailty.study import StudyRunner, load_study_plan
from ppg_frailty.audit import legacy_v2_bridge
from ppg_frailty.v5.output_contract import export_pipeline_excel
from ppg_frailty.v5 import sweep as sweep_module


def _fake_executor(
    case: Any,
    config_path: Path,
    output: Path,
    plan: Any,
    progress: Any,
) -> Mapping[str, Any]:
    cells = []
    for repeat in plan.execution.repeats:
        for fold in plan.execution.folds:
            directory = output / f"repeat_{repeat:02d}" / f"fold_{fold:02d}"
            directory.mkdir(parents=True, exist_ok=False)
            (directory / "marker.json").write_text("{}\n", encoding="utf-8")
            cells.append(
                {
                    "status": "passed",
                    "repeat_index": repeat,
                    "fold_index": fold,
                }
            )
    return {
        "status": "passed",
        "config_id": case.config["config_id"],
        "output_dir": str(output),
        "cell_results": cells,
    }


def test_v5_runner_publishes_comparison_repeat_fold_layout(tmp_path: Path) -> None:
    plan = load_study_plan(ROOT / "configs/studies/single_config_v2.yaml")
    plan = replace(
        plan,
        execution=replace(
            plan.execution,
            repeats=(0,),
            folds=(0,),
            jobs=1,
            measure_operational_costs=False,
        ),
    )
    result = StudyRunner(
        pipeline_root=ROOT,
        executor=_fake_executor,
        output_layout="v5",
    ).run(plan, output_root=tmp_path, run_name="reviewable_run")

    manifest = json.loads(
        (result.output_directory / "study_manifest.json").read_text(encoding="utf-8")
    )
    case = manifest["cases"][0]
    comparison = result.output_directory / case["case_directory"]
    assert manifest["output_layout"] == "comparison/repeat/fold"
    assert (comparison / "repeat_00/fold_00/marker.json").is_file()
    assert (comparison / "case_result.json").is_file()
    assert not (comparison / "attempts").exists()
    assert (result.output_directory / case["resolved_config_path"]).is_file()
    assert (result.output_directory / ".runner_state").is_dir()


def _write_csv(path: Path, fields: tuple[str, ...], rows: tuple[tuple[Any, ...], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def test_pipeline_excel_is_recoverable_postprocessing(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    run = tmp_path / "run"
    tables = run / "tables"
    tables.mkdir(parents=True)
    (run / "v5_data_manifest.json").write_text("{}\n", encoding="utf-8")
    _write_csv(
        tables / "v5_fold_predictions.csv",
        ("level", "artifact_state", "path"),
        (),
    )
    _write_csv(tables / "v5_fold_models.csv", ("case_id", "model_id"), (("a", "m"),))
    _write_csv(
        tables / "v5_config_parameters.csv",
        ("case_id", "parameter_path", "value_json"),
        (("a", "training.fixed_epochs", "10"),),
    )

    first = export_pipeline_excel(run, allow_legacy_location=True)
    assert first["status"] == "complete"
    assert (tables / "pipeline_data.xlsx").is_file()
    second = export_pipeline_excel(
        run,
        allow_legacy_location=True,
        replace=True,
    )
    assert second["status"] == "complete"


def test_sweep_uses_the_shared_environment_evaluator(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    events: list[str] = []

    def _reject_environment(*_: Any, **__: Any) -> Any:
        events.append("environment_checked")
        raise RuntimeError("locked environment mismatch")

    monkeypatch.setattr(sweep_module, "evaluate_environment", _reject_environment)
    args = sweep_module.build_parser().parse_args(
        ["validate", "--plan", str(ROOT / "configs/studies/single_config_v2.yaml")]
    )
    plan = load_study_plan(ROOT / "configs/studies/single_config_v2.yaml")
    with pytest.raises(RuntimeError, match="locked environment mismatch"):
        sweep_module._environment_check(args, plan)
    assert events == ["environment_checked"]


def test_v5_phase0_runner_binds_data_only_outputs_to_the_run(tmp_path: Path) -> None:
    plan = load_study_plan(
        ROOT / "configs/studies/static_line_b_staged_v2/stage3_alter.yaml"
    )
    run = tmp_path / "pipeline_output" / "legacy_bridge"
    run.mkdir(parents=True)
    captured: dict[str, Any] = {}

    def phase0_runner(**kwargs: Any) -> Mapping[str, Any]:
        captured.update(kwargs)
        return {"decision": "PASS", "outputs": {}}

    result = StudyRunner(
        pipeline_root=ROOT,
        phase0_runner=phase0_runner,
        output_layout="v5",
    )._run_phase0_audit(plan, output_directory=run)

    assert result is not None
    assert result["audit_status"] == "completed"
    assert Path(captured["artifact_root"]) == run
    assert captured["generate_report"] is False


def test_v5_phase0_data_gate_never_writes_markdown(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    repository = tmp_path / "repository"
    pipeline = repository / "final_v0" / "final_pipeline_v5"
    (pipeline / "manifests").mkdir(parents=True)
    (pipeline / "splits").mkdir(parents=True)
    (pipeline / "manifests/internal_records_v2.csv").write_text(
        "fixture\n", encoding="utf-8"
    )
    (pipeline / "splits/sgkf5_repeated_grouped_5x5_v2.csv").write_text(
        "fixture\n", encoding="utf-8"
    )
    specification = repository / "spec.md"
    specification.write_text("registered phase zero\n", encoding="utf-8")
    source_sha = hashlib.sha256(specification.read_bytes()).hexdigest()
    phase0 = {
        "manifest_path": "manifests/internal_records_v2.csv",
        "manifest_expected_rows": 0,
        "split_path": "splits/sgkf5_repeated_grouped_5x5_v2.csv",
        "required_channel_order": [
            "RED",
            "IR",
            "AX",
            "AY",
            "AZ",
            "GX",
            "GY",
            "GZ",
        ],
        "audit_outputs": list(legacy_v2_bridge._EXPECTED_OUTPUTS),
    }
    cache_payload = {
        "schema_version": "fixture.cache.v1",
        "status": "historical_cache_not_available",
        "cache_files": [],
    }
    split_payload = {"schema_version": "fixture.split.v1", "status": "PASS"}
    monkeypatch.setattr(legacy_v2_bridge, "load_internal_manifest", lambda _: [])
    monkeypatch.setattr(legacy_v2_bridge, "_read_label_table", lambda _: ({}, []))
    monkeypatch.setattr(
        legacy_v2_bridge, "_discover_static", lambda *_: ([], [], [])
    )
    monkeypatch.setattr(
        legacy_v2_bridge, "_audit_sources", lambda *_: ([], [], [], [])
    )
    monkeypatch.setattr(legacy_v2_bridge, "_manifest_diff", lambda *_: ([], []))
    monkeypatch.setattr(legacy_v2_bridge, "_audit_imu", lambda *_: ([], [], []))
    monkeypatch.setattr(
        legacy_v2_bridge,
        "_audit_cache",
        lambda *_: (cache_payload, ["historical_cache_not_available"]),
    )
    monkeypatch.setattr(
        legacy_v2_bridge, "_audit_split", lambda *_: (split_payload, [])
    )
    run = pipeline / "pipeline_output" / "legacy_bridge"
    result = legacy_v2_bridge.run_legacy_v2_phase0(
        repository,
        pipeline_root=pipeline,
        artifact_root=run,
        phase0_spec=phase0,
        source_specification="spec.md",
        source_specification_sha256=source_sha,
        generate_report=False,
    )

    assert len(result.outputs) == 8
    assert all(Path(path).is_file() for path in result.outputs.values())
    assert all(run in Path(path).parents for path in result.outputs.values())
    assert not any(path.suffix.lower() == ".md" for path in run.rglob("*"))
