"""Case-level study runner with resume, failure records, and delayed adapters."""

from __future__ import annotations

import csv
import inspect
import json
import os
import time
import traceback
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from .expand import ResolvedCase, StudyExpansion, expand_study
from .progress import (
    CompositeProgressSink,
    JsonlProgressSink,
    NullProgressSink,
    ProgressEvent,
    ProgressSink,
)
from .schema import StudyPlan


CaseExecutor = Callable[
    [ResolvedCase, Path, Path, StudyPlan, ProgressSink], Mapping[str, Any] | Any
]

_DEEP_MODEL_TOKENS = (
    "compactcnn",
    "compact_cnn",
    "inception",
    "shapeformer",
    "fusioncompact",
    "fusioninception",
    "fusion_compact",
    "fusion_inception",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"value is not JSON compatible: {type(value)!r}")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{time.time_ns()}")
    try:
        temporary.write_text(
            json.dumps(
                _jsonable(value),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_csv(path: Path, rows: tuple[Mapping[str, Any], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _resume_contract(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    execution = payload.get("execution")
    execution = execution if isinstance(execution, Mapping) else {}
    return {
        "schema_version": payload.get("schema_version"),
        "study": payload.get("study"),
        "base_config": payload.get("base_config"),
        "axes": payload.get("axes"),
        "repeats": execution.get("repeats"),
        "folds": execution.get("folds"),
    }


def _study_object(plan: StudyPlan) -> str:
    if plan.study.kind == "ablation":
        return plan.axes[0].path.replace(".", "-")
    return plan.study.study_id


def _safe_slug(value: str) -> str:
    return "".join(
        character.lower() if character.isalnum() else "-" for character in value
    ).strip("-") or "study"


def _contains_deep_case(expansion: StudyExpansion) -> bool:
    for case in expansion.cases:
        model = case.config.get("model", {})
        if isinstance(model, Mapping):
            identity = " ".join(
                str(model.get(key, "")) for key in ("model_id", "variant")
            ).lower()
            if any(token in identity for token in _DEEP_MODEL_TOKENS):
                return True
    return False


def _reject_unfrozen_outer_cv_ensembles(expansion: StudyExpansion) -> None:
    unresolved = []
    for case in expansion.cases:
        model = case.config.get("model", {})
        if isinstance(model, Mapping) and int(model.get("ensemble_size", 1)) > 1:
            unresolved.append(case.case_id)
    if unresolved:
        raise RuntimeError(
            "outer-CV ensemble execution is fail-closed until the repeat-by-member "
            "seed matrix is manually frozen; affected cases: "
            + ", ".join(unresolved)
        )


def _executor_progress_adapter(
    sink: ProgressSink, case_id: str
) -> Callable[[Any], None]:
    def emit(value: Any) -> None:
        event = ProgressEvent.from_value(value)
        if event.case_id is None:
            event = ProgressEvent(
                event=event.event,
                current=event.current,
                total=event.total,
                case_id=case_id,
                repeat=event.repeat,
                fold=event.fold,
                epoch=event.epoch,
                message=event.message,
                timestamp_utc=event.timestamp_utc,
            )
        sink(event)

    return emit


def _invoke_with_supported_kwargs(function: Callable[..., Any], **kwargs: Any) -> Any:
    parameters = inspect.signature(function).parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    selected = kwargs if accepts_kwargs else {
        key: value for key, value in kwargs.items() if key in parameters
    }
    return function(**selected)


def default_experiment_executor(
    case: ResolvedCase,
    config_path: Path,
    case_directory: Path,
    plan: StudyPlan,
    progress_sink: ProgressSink,
) -> Mapping[str, Any]:
    """Delayed adapter to the canonical full runner; importing does not train."""

    from ppg_frailty import experiment as experiment_module

    emit = _executor_progress_adapter(progress_sink, case.case_id)
    experiment_output = case_directory / "experiment"
    full_runner = getattr(experiment_module, "run_full_experiment", None)
    cell_runner = getattr(experiment_module, "run_outer_cell", None)
    complete_5x5 = (
        tuple(plan.execution.repeats) == tuple(range(5))
        and tuple(plan.execution.folds) == tuple(range(5))
    )
    if callable(full_runner) and complete_5x5:
        result = _invoke_with_supported_kwargs(
            full_runner,
            config_path=config_path,
            output_dir=experiment_output,
            repeats=plan.execution.repeats,
            folds=plan.execution.folds,
            progress_callback=emit,
        )
        return _jsonable(result)
    if not callable(cell_runner):
        if callable(full_runner) and not complete_5x5:
            raise RuntimeError(
                "partial repeat/fold selection requires run_outer_cell; refusing "
                "to delegate it to a full-only runner"
            )
        raise RuntimeError(
            "canonical experiment adapter exposes neither run_full_experiment "
            "nor run_outer_cell"
        )
    cells: list[Any] = []
    failed_cells: list[str] = []
    for repeat in plan.execution.repeats:
        for fold in plan.execution.folds:
            cell_result = _jsonable(
                _invoke_with_supported_kwargs(
                    cell_runner,
                    config_path=config_path,
                    output_dir=experiment_output / f"repeat_{repeat:02d}_fold_{fold:02d}",
                    repeat_index=repeat,
                    fold_index=fold,
                    progress_callback=emit,
                )
            )
            nested_cells = (
                cell_result.get("cell_results")
                if isinstance(cell_result, Mapping)
                else None
            )
            if isinstance(nested_cells, list):
                cells.extend(nested_cells)
            else:
                cells.append(cell_result)
            if (
                isinstance(cell_result, Mapping)
                and cell_result.get("status") != "passed"
            ):
                failed_cells.append(
                    f"r{repeat}_f{fold}:{cell_result.get('status')}"
                )
    return {
        "status": "failed_closed" if failed_cells else "passed",
        "config_id": case.config.get("config_id"),
        "cell_results": cells,
        "output_dir": str(experiment_output),
        "failure_reasons": failed_cells,
    }


def _process_default_case(request: Mapping[str, Any]) -> Mapping[str, Any]:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    case = ResolvedCase(
        case_id=str(request["case_id"]),
        config=dict(request["config"]),
        changed_values=dict(request["changed_values"]),
        config_sha256=str(request["config_sha256"]),
        is_reference=bool(request["is_reference"]),
    )
    from .expand import parse_study_plan

    plan = parse_study_plan(dict(request["plan"]))
    attempt_directory = Path(str(request["attempt_directory"]))
    child_sink = JsonlProgressSink(attempt_directory / "executor_events.jsonl")
    return default_experiment_executor(
        case,
        Path(str(request["config_path"])),
        attempt_directory,
        plan,
        child_sink,
    )


@dataclass(frozen=True)
class StudyRunResult:
    status: str
    output_directory: Path
    planned_case_count: int
    passed_case_count: int
    failed_case_count: int
    not_run_case_count: int
    planned_cell_count: int
    reported_cell_count: int
    passed_cell_count: int
    failed_cell_count: int
    not_run_cell_count: int
    resumed_case_count: int
    effective_jobs: int
    case_records: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "ppg_frailty.study_run_result.v2",
            "status": self.status,
            "output_directory": str(self.output_directory),
            "planned_case_count": self.planned_case_count,
            "passed_case_count": self.passed_case_count,
            "failed_case_count": self.failed_case_count,
            "not_run_case_count": self.not_run_case_count,
            "planned_cell_count": self.planned_cell_count,
            "reported_cell_count": self.reported_cell_count,
            "passed_cell_count": self.passed_cell_count,
            "failed_cell_count": self.failed_cell_count,
            "not_run_cell_count": self.not_run_cell_count,
            "resumed_case_count": self.resumed_case_count,
            "effective_jobs": self.effective_jobs,
            "case_records": list(self.case_records),
        }


class StudyRunner:
    """Materialize cases and execute each through one canonical adapter."""

    def __init__(
        self,
        *,
        pipeline_root: str | Path,
        executor: CaseExecutor | None = None,
        progress_sink: ProgressSink | None = None,
    ) -> None:
        self.pipeline_root = Path(pipeline_root).resolve()
        self.executor = executor
        self.progress_sink = progress_sink or NullProgressSink()

    def expand(self, plan: StudyPlan) -> StudyExpansion:
        return expand_study(plan, pipeline_root=self.pipeline_root)

    def _new_output(self, plan: StudyPlan, output_root: str | Path | None) -> Path:
        raw = Path(output_root or plan.output.root)
        root = raw.resolve() if raw.is_absolute() else (self.pipeline_root / raw).resolve()
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = root / (
            f"{stamp}_{plan.study.kind}_{_safe_slug(_study_object(plan))}"
        )
        for index in range(1, 1000):
            candidate = base if index == 1 else base.with_name(f"{base.name}_{index:02d}")
            try:
                candidate.mkdir(parents=False, exist_ok=False)
                return candidate
            except FileExistsError:
                continue
        raise RuntimeError(f"cannot create unique study directory below {root}")

    def _materialize(
        self, expansion: StudyExpansion, output: Path, *, resumed: bool
    ) -> dict[str, Path]:
        output.mkdir(parents=True, exist_ok=True)
        configs = output / "resolved_configs"
        cases_root = output / "cases"
        tables = output / "tables"
        configs.mkdir(exist_ok=True)
        cases_root.mkdir(exist_ok=True)
        tables.mkdir(exist_ok=True)
        plan_path = output / "study_plan.yaml"
        if resumed:
            if not plan_path.is_file():
                raise FileNotFoundError("resume directory has no study_plan.yaml")
            existing_plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
            if not isinstance(existing_plan, Mapping) or _resume_contract(
                existing_plan
            ) != _resume_contract(expansion.plan.to_dict()):
                raise ValueError(
                    "resume study-plan drift: study/base/axes/repeats/folds must "
                    "match; jobs and report presentation may be changed"
                )
        else:
            plan_path.write_text(
                yaml.safe_dump(
                    expansion.plan.to_dict(), sort_keys=False, allow_unicode=True
                ),
                encoding="utf-8",
            )
        case_rows: list[Mapping[str, Any]] = []
        for case in expansion.cases:
            target = configs / f"{case.case_id}.yaml"
            encoded = yaml.safe_dump(
                dict(case.config), sort_keys=False, allow_unicode=True
            )
            if target.exists() and target.read_text(encoding="utf-8") != encoded:
                raise ValueError(f"resume config drift for case {case.case_id}")
            target.write_text(encoded, encoding="utf-8")
            case_rows.append(case.to_dict())
            (cases_root / case.case_id).mkdir(exist_ok=True)
        _write_csv(tables / "resolved_cases.csv", tuple(case_rows))
        _write_csv(tables / "varied_parameters.csv", expansion.varied_parameters)
        _write_csv(
            tables / "controlled_parameters.csv", expansion.controlled_parameters
        )
        return {"configs": configs, "cases": cases_root, "tables": tables}

    def _attempt_number(self, case_directory: Path) -> int:
        attempts = case_directory / "attempts"
        attempts.mkdir(exist_ok=True)
        numbers: list[int] = []
        for path in attempts.glob("attempt_*"):
            try:
                numbers.append(int(path.stem.split("_")[-1]))
            except ValueError:
                continue
        return max(numbers, default=0) + 1

    def _create_attempt_directory(
        self,
        case_directory: Path,
    ) -> tuple[int, Path]:
        attempt = self._attempt_number(case_directory)
        target = case_directory / "attempts" / f"attempt_{attempt:03d}"
        target.mkdir(parents=False, exist_ok=False)
        return attempt, target

    @staticmethod
    def _artifact_root(
        normalized_result: Any,
        *,
        case_directory: Path,
        attempt_directory: Path,
    ) -> str:
        declared = (
            normalized_result.get("output_dir")
            if isinstance(normalized_result, Mapping)
            else None
        )
        if declared:
            raw = Path(str(declared))
            target = raw.resolve() if raw.is_absolute() else (attempt_directory / raw).resolve()
        else:
            target = attempt_directory.resolve()
        target.relative_to(attempt_directory.resolve())
        return target.relative_to(case_directory.resolve()).as_posix()

    def _existing_pass(self, case: ResolvedCase, case_directory: Path) -> Mapping[str, Any] | None:
        path = case_directory / "case_result.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("config_sha256") != case.config_sha256:
            raise ValueError(f"resume result config drift for {case.case_id}")
        return payload if payload.get("status") == "passed" else None

    def _run_one(
        self,
        case: ResolvedCase,
        config_path: Path,
        case_directory: Path,
        plan: StudyPlan,
        executor: CaseExecutor,
    ) -> Mapping[str, Any]:
        attempt, attempt_directory = self._create_attempt_directory(case_directory)
        started = time.perf_counter()
        started_utc = _utc_now()
        try:
            result = executor(
                case, config_path, attempt_directory, plan, self.progress_sink
            )
            normalized_result = _jsonable(result)
            reported_status = (
                str(normalized_result.get("status", "passed"))
                if isinstance(normalized_result, Mapping)
                else "passed"
            )
            passed = reported_status in {"passed", "success", "complete", "completed"}
            payload: dict[str, Any] = {
                "schema_version": "ppg_frailty.study_case_result.v2",
                "case_id": case.case_id,
                "config_sha256": case.config_sha256,
                "status": "passed" if passed else "failed",
                "attempt": attempt,
                "attempt_directory": attempt_directory.relative_to(
                    case_directory
                ).as_posix(),
                "artifact_root": self._artifact_root(
                    normalized_result,
                    case_directory=case_directory,
                    attempt_directory=attempt_directory,
                ),
                "started_utc": started_utc,
                "finished_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "result": normalized_result,
            }
            if not passed:
                payload["error_type"] = "CanonicalExperimentFailedClosed"
                payload["error"] = f"executor reported status={reported_status}"
        except Exception as error:  # noqa: BLE001 - failure is a study artifact.
            payload = {
                "schema_version": "ppg_frailty.study_case_result.v2",
                "case_id": case.case_id,
                "config_sha256": case.config_sha256,
                "status": "failed",
                "attempt": attempt,
                "attempt_directory": attempt_directory.relative_to(
                    case_directory
                ).as_posix(),
                "artifact_root": attempt_directory.relative_to(
                    case_directory
                ).as_posix(),
                "started_utc": started_utc,
                "finished_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        attempt_path = attempt_directory / "attempt_result.json"
        _atomic_json(attempt_path, payload)
        _atomic_json(case_directory / "case_result.json", payload)
        return payload

    def run(
        self,
        plan: StudyPlan,
        *,
        output_root: str | Path | None = None,
        resume_directory: str | Path | None = None,
    ) -> StudyRunResult:
        expansion = self.expand(plan)
        _reject_unfrozen_outer_cv_ensembles(expansion)
        resumed_run = resume_directory is not None
        output = (
            Path(resume_directory).resolve()
            if resumed_run
            else self._new_output(plan, output_root)
        )
        if resumed_run and not output.is_dir():
            raise FileNotFoundError(output)
        paths = self._materialize(expansion, output, resumed=resumed_run)
        jsonl = JsonlProgressSink(output / "progress_events.jsonl")
        original_sink = self.progress_sink
        self.progress_sink = CompositeProgressSink((original_sink, jsonl))
        deep = _contains_deep_case(expansion)
        requested_jobs = int(plan.execution.jobs)
        effective_jobs = requested_jobs
        job_message = f"jobs={effective_jobs}"
        if deep and requested_jobs > 1 and not plan.execution.allow_parallel_deep:
            effective_jobs = 1
            job_message = f"deep model detected; jobs reduced {requested_jobs}->1"
        if not plan.execution.continue_on_error and effective_jobs > 1:
            effective_jobs = 1
            job_message = (
                "continue_on_error=false requires deterministic fail-fast order; "
                f"jobs reduced {requested_jobs}->1"
            )
        total = len(expansion.cases)
        records: list[Mapping[str, Any]] = []
        pending: list[tuple[ResolvedCase, Path, Path]] = []
        resumed_count = 0
        self.progress_sink(
            ProgressEvent(
                event="study_started",
                current=0,
                total=total,
                message=job_message,
            )
        )
        for case in expansion.cases:
            case_directory = paths["cases"] / case.case_id
            existing = self._existing_pass(case, case_directory) if resumed_run else None
            if existing is not None:
                resumed_count += 1
                records.append(existing)
                self.progress_sink(
                    ProgressEvent(
                        event="case_resumed",
                        current=len(records),
                        total=total,
                        case_id=case.case_id,
                    )
                )
            else:
                pending.append(
                    (case, paths["configs"] / f"{case.case_id}.yaml", case_directory)
                )
        completed = len(records)
        chosen_executor = self.executor or default_experiment_executor
        if effective_jobs == 1:
            for case, config_path, case_directory in pending:
                self.progress_sink(
                    ProgressEvent(
                        event="case_started",
                        current=completed,
                        total=total,
                        case_id=case.case_id,
                    )
                )
                payload = self._run_one(
                    case, config_path, case_directory, plan, chosen_executor
                )
                records.append(payload)
                completed += 1
                self.progress_sink(
                    ProgressEvent(
                        event="case_finished",
                        current=completed,
                        total=total,
                        case_id=case.case_id,
                        message=str(payload["status"]),
                    )
                )
                if payload["status"] != "passed" and not plan.execution.continue_on_error:
                    break
        else:
            pool_type = ProcessPoolExecutor if self.executor is None else ThreadPoolExecutor
            futures: dict[
                Future[Any],
                tuple[ResolvedCase, Path, bool, float, str, int | None, Path | None],
            ] = {}
            with pool_type(max_workers=effective_jobs) as pool:
                for case, config_path, case_directory in pending:
                    submitted = time.perf_counter()
                    submitted_utc = _utc_now()
                    self.progress_sink(
                        ProgressEvent(
                            event="case_started",
                            current=completed,
                            total=total,
                            case_id=case.case_id,
                        )
                    )
                    if self.executor is None:
                        attempt, attempt_directory = self._create_attempt_directory(
                            case_directory
                        )
                        request = {
                            **case.to_dict(),
                            "config": dict(case.config),
                            "plan": plan.to_dict(),
                            "config_path": str(config_path),
                            "attempt_directory": str(attempt_directory),
                        }
                        future = pool.submit(_process_default_case, request)
                        needs_parent_record = True
                    else:
                        attempt = None
                        attempt_directory = None
                        future = pool.submit(
                            self._run_one,
                            case,
                            config_path,
                            case_directory,
                            plan,
                            chosen_executor,
                        )
                        needs_parent_record = False
                    futures[future] = (
                        case,
                        case_directory,
                        needs_parent_record,
                        submitted,
                        submitted_utc,
                        attempt,
                        attempt_directory,
                    )
                for future in as_completed(futures):
                    (
                        case,
                        case_directory,
                        needs_parent_record,
                        submitted,
                        submitted_utc,
                        attempt,
                        attempt_directory,
                    ) = futures[future]
                    payload = (
                        self._wrap_process_result(
                            future,
                            case,
                            case_directory,
                            attempt=attempt,
                            attempt_directory=attempt_directory,
                            started=submitted,
                            started_utc=submitted_utc,
                        )
                        if needs_parent_record
                        else future.result()
                    )
                    records.append(payload)
                    completed += 1
                    self.progress_sink(
                        ProgressEvent(
                            event="case_finished",
                            current=completed,
                            total=total,
                            case_id=case.case_id,
                            message=str(payload["status"]),
                        )
                    )
        by_case = {str(record["case_id"]): record for record in records}
        ordered = tuple(
            by_case[case.case_id]
            for case in expansion.cases
            if case.case_id in by_case
        )
        passed = sum(record.get("status") == "passed" for record in ordered)
        failed = sum(record.get("status") == "failed" for record in ordered)
        not_run = total - passed - failed
        expected_cells_per_case = (
            len(plan.execution.repeats) * len(plan.execution.folds)
        )
        planned_cells = total * expected_cells_per_case
        reported_cells = 0
        passed_cells = 0
        failed_cells = 0
        for record in ordered:
            result_payload = (
                record.get("result")
                if isinstance(record.get("result"), Mapping)
                else {}
            )
            cells = (
                result_payload.get("cell_results")
                if isinstance(result_payload.get("cell_results"), list)
                else []
            )
            for cell in cells:
                if not isinstance(cell, Mapping):
                    continue
                reported_cells += 1
                if str(cell.get("status")) in {
                    "passed",
                    "success",
                    "complete",
                    "completed",
                }:
                    passed_cells += 1
                else:
                    failed_cells += 1
        not_run_cells = planned_cells - reported_cells
        if not_run_cells < 0:
            raise RuntimeError("reported cell count exceeds the declared study plan")
        status = "passed" if passed == total else ("failed" if passed == 0 else "partial")
        result = StudyRunResult(
            status=status,
            output_directory=output,
            planned_case_count=total,
            passed_case_count=passed,
            failed_case_count=failed,
            not_run_case_count=not_run,
            planned_cell_count=planned_cells,
            reported_cell_count=reported_cells,
            passed_cell_count=passed_cells,
            failed_cell_count=failed_cells,
            not_run_cell_count=not_run_cells,
            resumed_case_count=resumed_count,
            effective_jobs=effective_jobs,
            case_records=ordered,
        )
        manifest = {
            "schema_version": "ppg_frailty.study_manifest.v2",
            "status": status,
            "created_or_resumed_utc": _utc_now(),
            "study": plan.to_dict()["study"],
            "base_config": str(expansion.base_config_path),
            "reference_case_id": expansion.reference_case_id,
            "execution": plan.to_dict()["execution"],
            "effective_jobs": effective_jobs,
            "planned_case_count": total,
            "passed_case_count": passed,
            "failed_case_count": failed,
            "not_run_case_count": not_run,
            "planned_cell_count": planned_cells,
            "reported_cell_count": reported_cells,
            "passed_cell_count": passed_cells,
            "failed_cell_count": failed_cells,
            "not_run_cell_count": not_run_cells,
            "resumed_case_count": resumed_count,
            "cases": [case.to_dict() for case in expansion.cases],
        }
        _atomic_json(output / "study_manifest.json", manifest)
        _atomic_json(output / "study_run_result.json", result.to_dict())
        self.progress_sink(
            ProgressEvent(
                event="study_finished",
                current=passed + failed,
                total=total,
                message=status,
            )
        )
        close = getattr(self.progress_sink, "close", None)
        if callable(close):
            close()
        self.progress_sink = original_sink
        return result

    def _wrap_process_result(
        self,
        child_future: Future[Any],
        case: ResolvedCase,
        case_directory: Path,
        *,
        attempt: int | None,
        attempt_directory: Path | None,
        started: float,
        started_utc: str,
    ) -> Mapping[str, Any]:
        if attempt is None or attempt_directory is None:
            raise RuntimeError("process result lost its allocated attempt directory")
        try:
            result = child_future.result()
            normalized_result = _jsonable(result)
            reported_status = (
                str(normalized_result.get("status", "passed"))
                if isinstance(normalized_result, Mapping)
                else "passed"
            )
            passed = reported_status in {"passed", "success", "complete", "completed"}
            payload = {
                "schema_version": "ppg_frailty.study_case_result.v2",
                "case_id": case.case_id,
                "config_sha256": case.config_sha256,
                "status": "passed" if passed else "failed",
                "attempt": attempt,
                "attempt_directory": attempt_directory.relative_to(
                    case_directory
                ).as_posix(),
                "artifact_root": self._artifact_root(
                    normalized_result,
                    case_directory=case_directory,
                    attempt_directory=attempt_directory,
                ),
                "started_utc": started_utc,
                "finished_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "result": normalized_result,
            }
            if not passed:
                payload["error_type"] = "CanonicalExperimentFailedClosed"
                payload["error"] = f"executor reported status={reported_status}"
        except Exception as error:  # noqa: BLE001
            payload = {
                "schema_version": "ppg_frailty.study_case_result.v2",
                "case_id": case.case_id,
                "config_sha256": case.config_sha256,
                "status": "failed",
                "attempt": attempt,
                "attempt_directory": attempt_directory.relative_to(
                    case_directory
                ).as_posix(),
                "artifact_root": attempt_directory.relative_to(
                    case_directory
                ).as_posix(),
                "started_utc": started_utc,
                "finished_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        _atomic_json(
            attempt_directory / "attempt_result.json", payload
        )
        _atomic_json(case_directory / "case_result.json", payload)
        return payload
