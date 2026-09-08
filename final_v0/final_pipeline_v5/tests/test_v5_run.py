from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping

import ppg_frailty.v5.run as run_module
from ppg_frailty.study import load_study_plan
from ppg_frailty.v5.service import RefitOptions


ROOT = Path(__file__).resolve().parents[1]


def _executor(
    case: Any,
    _config: Path,
    output: Path,
    plan: Any,
    _progress: Any,
) -> Mapping[str, Any]:
    cells = []
    for repeat in plan.execution.repeats:
        for fold in plan.execution.folds:
            directory = output / f"repeat_{repeat:02d}" / f"fold_{fold:02d}"
            directory.mkdir(parents=True)
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


def test_shared_run_service_keeps_multi_case_layout_and_resumes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = ROOT / "configs/studies/ablation_fixed_epochs_v2.yaml"
    original = load_study_plan(source)
    plan = replace(
        original,
        execution=replace(
            original.execution,
            repeats=(0,),
            folds=(0,),
            jobs=1,
            device="cuda",
        ),
    )
    finalized: list[tuple[Path, bool]] = []

    def finalize(study: Path, **kwargs: Any) -> dict[str, Any]:
        finalized.append((Path(study), kwargs["refit"].enabled))
        return {
            "data_manifest": {"status": "complete"},
            "model_config_export": str(tmp_path / "model_config" / Path(study).name),
            "refit": None,
        }

    monkeypatch.setattr(run_module, "post_run_finalize", finalize)
    monkeypatch.setattr(
        run_module,
        "try_export_pipeline_excel",
        lambda *_args, **_kwargs: {"status": "complete"},
    )
    common = {
        "pipeline_root": ROOT,
        "source": source,
        "output_root": tmp_path / "pipeline_output",
        "environment_policy": "record",
        "environment_lock": ROOT / "requirements/environment-finalcase-lock.yaml",
        "environment_hook": lambda _plan: {"status": "recorded"},
        "runner_executor": _executor,
        "refit": RefitOptions(),
    }

    first = run_module.run_study(plan, run_name="multi_case", **common)
    run = Path(first["pipeline_output"])
    manifest = json.loads((run / "study_manifest.json").read_text(encoding="utf-8"))
    case_directories = [run / row["case_directory"] for row in manifest["cases"]]

    assert first["status"] == "passed"
    assert len(case_directories) == 3
    assert all((case / "repeat_00/fold_00/marker.json").is_file() for case in case_directories)
    request = json.loads((run / "v5_run_request.json").read_text(encoding="utf-8"))
    assert request["data_only"] is True
    assert request["plots_generated"] is False
    assert request["refit_requested"] is False

    resumed = run_module.run_study(plan, resume=run, **common)

    assert Path(resumed["pipeline_output"]) == run
    assert len(list((run / "request_history").glob("v5_resume_request_*.json"))) == 1
    assert len(finalized) == 2
    assert all(study == run and enabled is False for study, enabled in finalized)
