"""The single training path shared by CLI, sweep, and Dashboard callers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
from typing import Any, Callable, Mapping

from ..module_registry import model_factory_contract
from ..study import (
    NullProgressSink,
    OutputSpec,
    ReportSpec,
    StudyPlan,
    StudyRunner,
    TerminalProgressSink,
    validate_canonical_expansion,
)
from .environment import DEFAULT_LOCK, EnvironmentCheck, evaluate_environment
from .io import file_sha256
from .output_contract import automatic_run_name, try_export_pipeline_excel
from .request_runner import (
    RequestRecordingStudyRunner,
    exclusive_resume_lock,
    execution_binding,
    write_request_status,
)
from .service import RefitOptions, post_run_finalize, preflight_refit_request


EnvironmentHook = Callable[[StudyPlan], EnvironmentCheck | Mapping[str, Any]]

def data_only_plan(plan: StudyPlan, output_root: str | Path | None = None) -> StudyPlan:
    """Return the same study with presentation disabled and no scientific edits."""

    return replace(
        plan,
        output=plan.output if output_root is None else OutputSpec(root=str(output_root)),
        report=ReportSpec(
            write_html=False,
            write_static_figures=False,
            write_excel_workbook=False,
            figure_modules=("all",),
        ),
    )

def _torch_devices(expansion: Any) -> set[str]:
    return {
        str(case.config["training"]["device"])
        for case in expansion.cases
        if model_factory_contract(str(case.config["model"]["model_id"]))["execution_backend"] == "torch"
    }

def _expanded(
    plan: StudyPlan,
    runner: StudyRunner,
    prepared: Any | None,
) -> tuple[StudyPlan, Any]:
    expansion = prepared
    if plan.execution.device is None:
        if expansion is not None:
            raise ValueError("prepared expansion requires an explicit execution device")
        preliminary = validate_canonical_expansion(runner.expand(plan))
        devices = _torch_devices(preliminary)
        if len(devices) > 1 or "" in devices:
            raise ValueError("all Torch cases must use one training.device")
        plan = replace(
            plan,
            execution=replace(plan.execution, device=next(iter(devices), "cpu")),
        )
        expansion = replace(preliminary, plan=plan)
    elif expansion is None:
        expansion = validate_canonical_expansion(runner.expand(plan))
    else:
        expansion = replace(expansion, plan=plan)
    devices = _torch_devices(expansion)
    if devices and devices != {str(plan.execution.device)}:
        raise ValueError("execution device differs from resolved training.device")
    return plan, expansion

def _environment_payload(
    plan: StudyPlan,
    policy: str,
    lock: Path,
    hook: EnvironmentHook | None,
) -> dict[str, Any]:
    checked = (
        hook(plan)
        if hook is not None
        else evaluate_environment(policy, device=str(plan.execution.device), lock_path=lock)
    )
    return checked.to_dict() if isinstance(checked, EnvironmentCheck) else dict(checked)

def run_study(
    plan: StudyPlan,
    *,
    pipeline_root: str | Path,
    source: str | Path | None = None,
    output_root: str | Path | None = None,
    run_name: str | None = None,
    resume: str | Path | None = None,
    environment_policy: str = "exact",
    environment_lock: str | Path = DEFAULT_LOCK,
    hash_predictions: bool = False,
    refit: RefitOptions | None = None,
    dry_run: bool = False,
    request_schema: str = "ppg_frailty.v5_run_request.v1",
    request_metadata: Mapping[str, Any] | None = None,
    prepared_expansion: Any | None = None,
    runner_executor: Any | None = None,
    environment_hook: EnvironmentHook | None = None,
    progress_sink: Any | None = None,
) -> dict[str, Any]:
    """Expand, check, train, index, export, and write Excel exactly once."""

    root = Path(pipeline_root).resolve()
    output = Path(output_root or root / "pipeline_output").resolve()
    plan = data_only_plan(plan, output)
    sink = progress_sink or (NullProgressSink() if dry_run else TerminalProgressSink())
    base_runner = StudyRunner(
        pipeline_root=root,
        output_layout="v5",
        **({} if runner_executor is None else {"executor": runner_executor}),
    )
    try:
        plan, expansion = _expanded(plan, base_runner, prepared_expansion)
        lock = Path(environment_lock).resolve()
        environment = _environment_payload(plan, environment_policy, lock, environment_hook)
        resumed = None if resume is None else Path(resume).resolve()
        if resumed is not None and run_name is not None:
            raise ValueError("--run-name cannot be combined with --resume")
        options = refit or RefitOptions()
        refit_preflight = preflight_refit_request(
            options,
            pipeline_root=root,
            cases=tuple(case.to_dict() for case in expansion.cases),
            repeats=tuple(plan.execution.repeats),
            folds=tuple(plan.execution.folds),
            resume_directory=resumed,
        )
        preview = {
            "data_only": True,
            "environment_check": environment,
            "refit_preflight": refit_preflight,
            "study": plan.to_dict(),
            "reference_case_id": expansion.reference_case_id,
            "cases": [case.to_dict() for case in expansion.cases],
            "varied_parameters": list(expansion.varied_parameters),
            "controlled_parameter_count": len(expansion.controlled_parameters),
        }
        if dry_run:
            return {"status": "valid", "dry_run": True, **preview}

        source_path = None if source is None else Path(source).resolve()
        name = None if resumed is not None else (run_name or automatic_run_name(source_path or "manual.yaml"))
        request = {
            **dict(request_metadata or {}),
            "schema_version": "ppg_frailty.v5_resume_request.v1" if resumed else request_schema,
            "source_yaml": None if source_path is None else str(source_path),
            "source_yaml_sha256": None if source_path is None else file_sha256(source_path),
            "run_name": resumed.name if resumed else name,
            "data_only": True,
            "plots_generated": False,
            "resumed": resumed is not None,
            "refit_requested": options.enabled,
            "environment_policy": environment_policy,
            "environment_lock": str(lock),
            "environment_lock_sha256": file_sha256(lock),
            "environment_check": environment,
            "execution_binding": execution_binding(plan, expansion),
            "refit_preflight": refit_preflight,
        }
        relative = (
            Path("v5_run_request.json")
            if resumed is None
            else Path("request_history") / f"v5_resume_request_{time.time_ns()}.json"
        )
        target = resumed or output / str(name)
        with exclusive_resume_lock(target):
            runner = RequestRecordingStudyRunner(
                pipeline_root=root,
                progress_sink=sink,
                output_layout="v5",
                pre_run_artifacts={relative.as_posix(): request},
                precomputed_expansion=expansion,
                **({} if runner_executor is None else {"executor": runner_executor}),
            )
            result = runner.run(
                plan,
                output_root=output,
                resume_directory=resumed,
                run_name=name,
            )
            try:
                finalized = post_run_finalize(
                    result.output_directory,
                    pipeline_root=root,
                    hash_prediction_files=hash_predictions,
                    export_configuration=True,
                    refit=options,
                )
                index = finalized["data_manifest"]
                excel = (
                    try_export_pipeline_excel(result.output_directory, replace=resumed is not None)
                    if index["status"] == "complete"
                    else {"status": "not_run_requires_complete_index", "pipeline_data_preserved": True}
                )
            except BaseException as error:
                write_request_status(
                    result.output_directory,
                    relative,
                    status="finalize_failed",
                    error=error,
                )
                raise
            write_request_status(result.output_directory, relative, status="complete")
        return {
            "status": result.status,
            "pipeline_output": str(result.output_directory),
            "index": index,
            "model_config_export": finalized["model_config_export"],
            "refit": finalized["refit"],
            "excel": excel,
            "exit_code": 0 if result.status == "passed" and index["status"] == "complete" else 2,
        }
    finally:
        sink.close()


__all__ = ["data_only_plan", "run_study"]
