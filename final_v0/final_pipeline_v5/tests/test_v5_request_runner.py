from __future__ import annotations

import json
from pathlib import Path

import pytest

from ppg_frailty.study import StudyRunner, load_study_plan
from ppg_frailty.v5.request_runner import (
    RequestRecordingStudyRunner,
    execution_binding,
    exclusive_resume_lock,
    read_anchored_request,
    validate_resume_environment,
    write_request_status,
)


ROOT = Path(__file__).resolve().parents[1]


def test_execution_binding_uses_resolved_config_identity() -> None:
    plan = load_study_plan(ROOT / "configs/studies/finalcase.yaml")
    expansion = StudyRunner(pipeline_root=ROOT, output_layout="v5").expand(plan)

    binding = execution_binding(plan, expansion)

    assert binding["cases"] == [
        {
            "case_id": expansion.cases[0].case_id,
            "config_id": expansion.cases[0].config["config_id"],
            "config_sha256": expansion.cases[0].config_sha256,
        }
    ]
    assert len(binding["binding_sha256"]) == 64


def test_pre_run_artifact_is_published_once_as_plain_json(
    tmp_path: Path,
) -> None:
    runner = RequestRecordingStudyRunner(
        pipeline_root=tmp_path,
        pre_run_artifacts={"request_history/request.json": {"ready": True}},
    )
    output = tmp_path / "output"
    output.mkdir()
    runner._publish_pre_run_artifacts(output)
    assert json.loads(
        (output / "request_history/request.json").read_text(encoding="utf-8")
    ) == {"ready": True}
    assert not (output / "request_history/request.anchor.json").exists()
    assert read_anchored_request(output, "request_history/request.json")[0] == {
        "ready": True
    }
    with pytest.raises(FileExistsError, match="already exists"):
        runner._publish_pre_run_artifacts(output)


@pytest.mark.parametrize("name", ["../escape.json", "/tmp/escape.json"])
def test_pre_run_artifact_path_cannot_escape(tmp_path: Path, name: str) -> None:
    runner = RequestRecordingStudyRunner(
        pipeline_root=tmp_path,
        pre_run_artifacts={name: {"ready": True}},
    )
    with pytest.raises(ValueError, match="safe relative"):
        runner._publish_pre_run_artifacts(tmp_path)


def test_resume_environment_compatibility_hook_adds_no_second_gate(
    tmp_path: Path,
) -> None:
    initial = {
        "environment_policy": "exact",
        "environment_lock_sha256": "a" * 64,
        "environment_check": {"status": "passed", "lock_id": "locked"},
        "execution_binding": {"binding_sha256": "b" * 64},
    }
    runner = RequestRecordingStudyRunner(
        pipeline_root=tmp_path,
        pre_run_artifacts={"v5_run_request.json": initial},
    )
    runner._publish_pre_run_artifacts(tmp_path)
    validate_resume_environment(tmp_path, initial)
    validate_resume_environment(tmp_path, {**initial, "environment_policy": "record"})


def test_plain_request_reader_reports_current_payload_and_digest(tmp_path: Path) -> None:
    runner = RequestRecordingStudyRunner(
        pipeline_root=tmp_path,
        pre_run_artifacts={"v5_run_request.json": {"ready": True}},
    )
    runner._publish_pre_run_artifacts(tmp_path)
    before = read_anchored_request(tmp_path, "v5_run_request.json")[1]
    (tmp_path / "v5_run_request.json").write_text(
        json.dumps({"ready": False}), encoding="utf-8"
    )
    payload, after, anchor = read_anchored_request(tmp_path, "v5_run_request.json")
    assert payload == {"ready": False}
    assert after != before
    assert anchor == {}


def test_run_scoped_lock_rejects_concurrent_owner(tmp_path: Path) -> None:
    target = tmp_path / "future_run"
    with exclusive_resume_lock(target):
        with pytest.raises(RuntimeError, match="another V5 process"):
            with exclusive_resume_lock(target):
                pass


def test_failed_resume_status_points_to_immutable_attempt(tmp_path: Path) -> None:
    relative = "request_history/resume.json"
    payload = {
        "schema_version": "ppg_frailty.v5_resume_request.v1",
        "resumed": True,
        "environment_lock_sha256": "a" * 64,
    }
    runner = RequestRecordingStudyRunner(
        pipeline_root=tmp_path,
        pre_run_artifacts={relative: payload},
    )
    runner._publish_pre_run_artifacts(tmp_path)
    immutable_before = (tmp_path / relative).read_bytes()

    write_request_status(
        tmp_path,
        relative,
        status="runner_failed",
        error=ValueError("test failure"),
    )

    status = json.loads((tmp_path / "v5_resume_request.json").read_text())
    assert status["latest_immutable_request"] == relative
    assert status["attempt_status"] == "runner_failed"
    assert status["attempt_error"] == {
        "type": "ValueError",
        "message": "test failure",
    }
    assert (tmp_path / relative).read_bytes() == immutable_before


def test_request_exists_before_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = load_study_plan(ROOT / "configs/studies/finalcase.yaml")
    expansion = StudyRunner(pipeline_root=ROOT, output_layout="v5").expand(plan)
    request = {
        "environment_lock_sha256": "a" * 64,
        "resumed": False,
        "refit_requested": False,
    }

    def stop_before_science(
        _runner: object, _expansion: object, output: Path, *, resumed: bool
    ) -> None:
        assert resumed is False
        assert read_anchored_request(output, "v5_run_request.json")[0] == request
        raise RuntimeError("stop before scientific execution")

    monkeypatch.setattr(StudyRunner, "_materialize", stop_before_science)
    runner = RequestRecordingStudyRunner(
        pipeline_root=ROOT,
        output_layout="v5",
        pre_run_artifacts={"v5_run_request.json": request},
        precomputed_expansion=expansion,
    )

    with pytest.raises(RuntimeError, match="stop before scientific execution"):
        runner.run(plan, output_root=tmp_path, run_name="anchored")
    status = json.loads((tmp_path / "anchored/v5_run_status.json").read_text())
    assert status["attempt_status"] == "runner_failed"
