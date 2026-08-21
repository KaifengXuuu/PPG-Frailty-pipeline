"""Case-level study runner with resume, failure records, and delayed adapters."""

from __future__ import annotations

import copy
import csv
import gc
import inspect
import json
import os
import time
import traceback
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import asdict, dataclass, is_dataclass, replace
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
Phase0Runner = Callable[..., Mapping[str, Any] | Any]

_OUTPUT_GROUPS = frozenset(
    {"raw", "fusion", "feature_vector", "feature_matrix"}
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


_COMPACT_CELL_FIELDS = (
    "status",
    "repeat_index",
    "fold_index",
    "split_seed",
    "training_seed",
    "config_hash",
    "preprocessing_hash",
    "code_commit",
    "source_version",
    "model_machine_id",
    "model_id",
    "representation_mode",
    "elapsed_seconds",
    "retained_train_record_count",
    "retained_oof_record_count",
    "selected_record_count",
    "oof_window_prediction_count",
    "class_order",
    "metrics",
    "operational_metrics",
)


def _compact_experiment_result(value: Any) -> dict[str, Any]:
    """Return a small study index while full experiment details remain on disk."""

    def field(name: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    compact_cells: list[dict[str, Any]] = []
    for raw in field("cell_results", ()) or ():
        if not isinstance(raw, Mapping):
            continue
        compact_cells.append(
            {
                key: _jsonable(raw[key])
                for key in _COMPACT_CELL_FIELDS
                if key in raw
            }
        )
    return {
        "schema_version": "ppg_frailty.study_executor_result.v3",
        "status": str(field("status", "passed")),
        "scientific_scope": field("scientific_scope"),
        "config_id": field("config_id"),
        "config_hash": field("config_hash"),
        "repeat_indices": _jsonable(field("repeat_indices", ())),
        "fold_indices": _jsonable(field("fold_indices", ())),
        "output_dir": (
            None if field("output_dir") is None else str(field("output_dir"))
        ),
        "cell_results": compact_cells,
        "metrics": _jsonable(field("metrics", {})),
        "failure_reasons": _jsonable(field("failure_reasons", ())),
        "detail_source": "persisted_experiment_artifacts",
    }


def _compact_case_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep case status/pointers while removing legacy nested experiment payloads."""

    compact = dict(record)
    result = compact.get("result")
    if isinstance(result, Mapping):
        compact["result"] = _compact_experiment_result(result)
    return compact


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
    contract = {
        str(key): value
        for key, value in payload.items()
        if key not in {"execution", "output", "report"}
    }
    contract["execution"] = {
        "repeats": execution.get("repeats"),
        "folds": execution.get("folds"),
        "device": execution.get("device"),
        "measure_operational_costs": execution.get(
            "measure_operational_costs", False
        ),
    }
    return contract


def _study_object(plan: StudyPlan) -> str:
    if plan.study.kind == "ablation":
        return plan.axes[0].path.replace(".", "-")
    return plan.study.study_id


def _safe_slug(value: str) -> str:
    return "".join(
        character.lower() if character.isalnum() else "-" for character in value
    ).strip("-") or "study"


def _contains_deep_case(expansion: StudyExpansion) -> bool:
    from ..module_registry import model_factory_contract

    for case in expansion.cases:
        model = case.config.get("model", {})
        if isinstance(model, Mapping):
            model_id = model.get("model_id")
            if model_id is None:
                raise ValueError(
                    f"study case {case.case_id} has no model.model_id"
                )
            if model_factory_contract(str(model_id))["execution_backend"] == "torch":
                return True
    return False


def _executor_progress_adapter(
    sink: ProgressSink,
    case_id: str,
    *,
    repeats: tuple[int, ...],
    folds: tuple[int, ...],
) -> Callable[[Any], None]:
    repeat_positions = {value: index + 1 for index, value in enumerate(repeats)}
    fold_positions = {value: index + 1 for index, value in enumerate(folds)}
    completed_repeats = 0

    def emit(value: Any) -> None:
        nonlocal completed_repeats
        event = ProgressEvent.from_value(value)
        unit_current = event.unit_current
        unit_total = event.unit_total
        detail_current = event.detail_current
        detail_total = event.detail_total
        detail_label = event.detail_label
        if event.event in {"module_start", "run_start"}:
            unit_current = completed_repeats
            unit_total = len(repeats)
            detail_current = 0
            detail_total = len(folds)
            detail_label = (
                "loading pipeline / preflight"
                if event.event == "module_start"
                else "CV setup"
            )
        elif event.repeat in repeat_positions and event.fold in fold_positions:
            repeat_position = repeat_positions[int(event.repeat)]
            fold_position = fold_positions[int(event.fold)]
            if event.event == "cell_complete":
                detail_current = fold_position
                if fold_position == len(folds):
                    completed_repeats = max(completed_repeats, repeat_position)
            else:
                detail_current = max(0, fold_position - 1)
            unit_current = completed_repeats
            unit_total = len(repeats)
            detail_total = len(folds)
            detail_label = (
                f"CV repeat {repeat_position}/{len(repeats)} · "
                f"fold {fold_position}/{len(folds)}"
            )
        elif event.event == "run_complete":
            completed_repeats = len(repeats)
            unit_current = completed_repeats
            unit_total = len(repeats)
            detail_current = len(folds)
            detail_total = len(folds)
            detail_label = "CV complete"
        elif event.event == "run_error":
            unit_current = completed_repeats
            unit_total = len(repeats)
            detail_current = 0
            detail_total = 0
            detail_label = "CV failed"
        event = replace(
            event,
            case_id=event.case_id or case_id,
            unit_current=unit_current,
            unit_total=unit_total,
            detail_current=detail_current,
            detail_total=detail_total,
            detail_label=detail_label,
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
    """Delayed adapter to the canonical or isolated Legacy Bridge runner."""

    emit = _executor_progress_adapter(
        progress_sink,
        case.case_id,
        repeats=tuple(plan.execution.repeats),
        folds=tuple(plan.execution.folds),
    )
    emit(
        {
            "event": "module_start",
            "message": "loading canonical pipeline and preflight",
        }
    )
    from ppg_frailty import experiment as experiment_module

    experiment_output = case_directory / "experiment"
    bridge = plan.legacy_bridge
    full_runner = getattr(experiment_module, "run_full_experiment", None)
    cell_runner = getattr(
        experiment_module,
        "run_legacy_bridge_outer_cell" if bridge is not None else "run_outer_cell",
        None,
    )
    complete_5x5 = (
        tuple(plan.execution.repeats) == tuple(range(5))
        and tuple(plan.execution.folds) == tuple(range(5))
    )
    if bridge is None and callable(full_runner) and complete_5x5:
        result = _invoke_with_supported_kwargs(
            full_runner,
            config_path=config_path,
            output_dir=experiment_output,
            repeats=plan.execution.repeats,
            folds=plan.execution.folds,
            progress_callback=emit,
            measure_operational_costs=plan.execution.measure_operational_costs,
        )
        compact = _compact_experiment_result(result)
        del result
        gc.collect()
        return compact
    if not callable(cell_runner):
        if bridge is None and callable(full_runner) and not complete_5x5:
            raise RuntimeError(
                "partial repeat/fold selection requires run_outer_cell; refusing "
                "to delegate it to a full-only runner"
            )
        required = (
            "run_legacy_bridge_outer_cell"
            if bridge is not None
            else "run_full_experiment or run_outer_cell"
        )
        raise RuntimeError(f"experiment adapter does not expose {required}")
    cells: list[Any] = []
    failed_cells: list[str] = []
    bridge_profile_id = (
        None
        if bridge is None
        else str(case.changed_values.get("study.legacy_bridge_profile", ""))
    )
    bridge_profile_definition: Mapping[str, Any] | None = None
    if bridge is not None:
        allowed_profiles = {
            str(profile["profile_id"]) for profile in bridge.profiles
        }
        if bridge_profile_id not in allowed_profiles:
            raise ValueError(
                f"{case.case_id} lacks one frozen Legacy Bridge profile for "
                f"design={bridge.design}"
            )
        matches = tuple(
            profile
            for profile in bridge.profiles
            if str(profile["catalog_case_id"]) == case.case_id
        )
        if len(matches) != 1:
            raise ValueError(
                f"{case.case_id} must bind exactly one Legacy Bridge profile"
            )
        bridge_profile_definition = matches[0]
    for repeat in plan.execution.repeats:
        for fold in plan.execution.folds:
            call_kwargs: dict[str, Any] = {
                "config_path": config_path,
                "output_dir": (
                    experiment_output / f"repeat_{repeat:02d}_fold_{fold:02d}"
                ),
                "repeat_index": repeat,
                "fold_index": fold,
                "progress_callback": emit,
                "measure_operational_costs": (
                    plan.execution.measure_operational_costs
                ),
            }
            if bridge is not None:
                call_kwargs["profile_id"] = bridge_profile_id
                if bridge.uses_inline_profiles:
                    assert bridge_profile_definition is not None
                    call_kwargs.update(
                        {
                            "protocol_design": bridge.design,
                            "profile_definition": copy.deepcopy(
                                dict(bridge_profile_definition)
                            ),
                            "profile_definition_sha256": (
                                bridge.controls_sha256(
                                    bridge_profile_definition
                                )
                            ),
                        }
                    )
                else:
                    call_kwargs.update(
                        {
                            "source_specification": bridge.source_specification,
                            "source_specification_sha256": (
                                bridge.source_specification_sha256
                            ),
                        }
                    )
            raw_result = _invoke_with_supported_kwargs(cell_runner, **call_kwargs)
            cell_result = _compact_experiment_result(raw_result)
            del raw_result
            gc.collect()
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
        output_group=(
            None
            if request.get("output_group") is None
            else str(request["output_group"])
        ),
        catalog_entry=(
            None
            if request.get("catalog_entry") is None
            else str(request["catalog_entry"])
        ),
        screen_profile_id=(
            None
            if request.get("screen_profile_id") is None
            else str(request["screen_profile_id"])
        ),
        rationale=(
            None if request.get("rationale") is None else str(request["rationale"])
        ),
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


class _ExecutorEventRelay:
    """Relay complete child JSONL rows to the parent terminal without new IPC."""

    def __init__(self, sink: ProgressSink) -> None:
        self._sink = sink
        self._case_by_path: dict[Path, str] = {}
        self._offsets: dict[Path, int] = {}
        self._buffers: dict[Path, bytes] = {}

    def register(self, case_id: str, path: Path) -> None:
        self._case_by_path[path] = case_id
        self._offsets[path] = 0
        self._buffers[path] = b""

    def unregister(self, path: Path) -> None:
        self._case_by_path.pop(path, None)
        self._offsets.pop(path, None)
        self._buffers.pop(path, None)

    def drain(self) -> int:
        relayed = 0
        for path, case_id in tuple(self._case_by_path.items()):
            if not path.is_file():
                continue
            with path.open("rb") as stream:
                stream.seek(self._offsets[path])
                chunk = stream.read()
            if not chunk:
                continue
            self._offsets[path] += len(chunk)
            rows = (self._buffers[path] + chunk).split(b"\n")
            self._buffers[path] = rows.pop()
            for encoded in rows:
                if not encoded.strip():
                    continue
                try:
                    event = ProgressEvent.from_value(
                        json.loads(encoded.decode("utf-8"))
                    )
                except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
                    continue
                if event.case_id is None:
                    event = replace(event, case_id=case_id)
                self._sink(event)
                relayed += 1
        return relayed


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
        phase0_runner: Phase0Runner | None = None,
        progress_sink: ProgressSink | None = None,
    ) -> None:
        self.pipeline_root = Path(pipeline_root).resolve()
        self.executor = executor
        self.phase0_runner = phase0_runner
        self.progress_sink = progress_sink or NullProgressSink()

    def expand(self, plan: StudyPlan) -> StudyExpansion:
        return expand_study(plan, pipeline_root=self.pipeline_root)

    def _run_phase0_audit(self, plan: StudyPlan) -> Mapping[str, Any] | None:
        """Run Phase 0 as advisory evidence; never authorize or block training."""

        bridge = plan.legacy_bridge
        if bridge is None or not bool(bridge.phase0.get("enabled", False)):
            return None
        base: dict[str, Any] = {
            "schema_version": "ppg_frailty.legacy_v2_phase0_advisory.v1",
            "advisory_only": True,
            "affects_training_execution": False,
            "training_blocked": False,
            "recorded_utc": _utc_now(),
            "source_specification": bridge.source_specification,
            "declared_source_specification_sha256": (
                bridge.source_specification_sha256
            ),
        }
        try:
            repository_root = self.pipeline_root.parents[1]
            source = (repository_root / bridge.source_specification).resolve()
            source.relative_to(repository_root)
            if not source.is_file():
                raise FileNotFoundError(source)
            phase0_runner = self.phase0_runner
            if phase0_runner is None:
                from ppg_frailty.audit.legacy_v2_bridge import (
                    run_legacy_v2_phase0,
                )

                phase0_runner = run_legacy_v2_phase0
            result = _invoke_with_supported_kwargs(
                phase0_runner,
                repository_root=repository_root,
                pipeline_root=self.pipeline_root,
                phase0_spec=bridge.phase0,
                source_specification=bridge.source_specification,
                source_specification_sha256=bridge.source_specification_sha256,
            )
            payload = _jsonable(result)
            if not isinstance(payload, Mapping):
                raise TypeError("legacy bridge Phase 0 result must be a mapping")
            json.dumps(payload, allow_nan=False)
            return {
                **base,
                "audit_status": "completed",
                "audit_decision": payload.get("decision"),
                "audit_result": dict(payload),
            }
        except Exception as error:  # noqa: BLE001 - advisory evidence only.
            return {
                **base,
                "audit_status": "error",
                "audit_decision": None,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }

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

    @staticmethod
    def _output_group(case: ResolvedCase) -> str | None:
        raw = getattr(case, "output_group", None)
        if raw is None:
            return None
        group = str(raw).strip()
        if group not in _OUTPUT_GROUPS:
            raise ValueError(
                f"case {case.case_id} output_group must be one of "
                f"{sorted(_OUTPUT_GROUPS)}"
            )
        return group

    def _case_paths(self, output: Path, case: ResolvedCase) -> tuple[Path, Path]:
        group = self._output_group(case)
        if group is None:
            case_directory = output / "cases" / case.case_id
            config_path = output / "resolved_configs" / f"{case.case_id}.yaml"
        else:
            case_directory = output / group / case.case_id
            config_path = case_directory / "resolved_config.yaml"
        output_resolved = output.resolve()
        case_directory.resolve().relative_to(output_resolved)
        config_path.resolve().relative_to(output_resolved)
        return config_path, case_directory

    def _case_manifest_row(
        self,
        output: Path,
        case: ResolvedCase,
    ) -> dict[str, Any]:
        config_path, case_directory = self._case_paths(output, case)
        return {
            **case.to_dict(),
            "output_group": self._output_group(case),
            "case_directory": case_directory.relative_to(output).as_posix(),
            "resolved_config_path": config_path.relative_to(output).as_posix(),
        }

    def _materialize(
        self, expansion: StudyExpansion, output: Path, *, resumed: bool
    ) -> None:
        output.mkdir(parents=True, exist_ok=True)
        tables = output / "tables"
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
                    "resume study-plan drift: scientific case definitions and "
                    "repeats/folds/device must match; jobs, output root, and "
                    "report presentation may be changed"
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
            target, case_directory = self._case_paths(output, case)
            target.parent.mkdir(parents=True, exist_ok=True)
            encoded = yaml.safe_dump(
                dict(case.config), sort_keys=False, allow_unicode=True
            )
            if target.exists():
                if target.read_text(encoding="utf-8") != encoded:
                    raise ValueError(f"resume config drift for case {case.case_id}")
            else:
                target.write_text(encoded, encoding="utf-8")
            case_rows.append(self._case_manifest_row(output, case))
            case_directory.mkdir(parents=True, exist_ok=True)
        _write_csv(tables / "resolved_cases.csv", tuple(case_rows))
        _write_csv(tables / "varied_parameters.csv", expansion.varied_parameters)
        _write_csv(
            tables / "controlled_parameters.csv", expansion.controlled_parameters
        )

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
        return _compact_case_record(payload) if payload.get("status") == "passed" else None

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
        self.progress_sink(
            ProgressEvent(
                event="study_preparing",
                message="resolving cases and materializing configs",
            )
        )
        expansion = self.expand(plan)
        resumed_run = resume_directory is not None
        output = (
            Path(resume_directory).resolve()
            if resumed_run
            else self._new_output(plan, output_root)
        )
        if resumed_run and not output.is_dir():
            raise FileNotFoundError(output)
        phase0_audit = self._run_phase0_audit(plan)
        if phase0_audit is not None:
            _atomic_json(output / "phase0_audit.json", phase0_audit)
        self._materialize(expansion, output, resumed=resumed_run)
        jsonl = JsonlProgressSink(output / "progress_events.jsonl")
        original_sink = self.progress_sink
        self.progress_sink = CompositeProgressSink(
            (original_sink, jsonl),
            close_sinks=(jsonl,),
        )
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
        repeats_per_case = len(plan.execution.repeats)
        records: list[Mapping[str, Any]] = []
        pending: list[tuple[ResolvedCase, Path, Path]] = []
        resumed_count = 0
        self.progress_sink(
            ProgressEvent(
                event="study_started",
                current=0,
                total=total,
                unit_current=0,
                unit_total=repeats_per_case,
                message=job_message,
            )
        )
        for case in expansion.cases:
            config_path, case_directory = self._case_paths(output, case)
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
                        unit_current=repeats_per_case,
                        unit_total=repeats_per_case,
                    )
                )
            else:
                pending.append((case, config_path, case_directory))
        completed = len(records)
        chosen_executor = self.executor or default_experiment_executor
        if effective_jobs == 1:
            self.progress_sink(
                ProgressEvent(event="study_running", message="running cases")
            )
            for case, config_path, case_directory in pending:
                self.progress_sink(
                    ProgressEvent(
                        event="case_started",
                        current=completed,
                        total=total,
                        case_id=case.case_id,
                        unit_current=0,
                        unit_total=repeats_per_case,
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
                        unit_current=(
                            repeats_per_case
                        ),
                        unit_total=repeats_per_case,
                        message=str(payload["status"]),
                    )
                )
                if payload["status"] != "passed" and not plan.execution.continue_on_error:
                    break
        else:
            pool_type = ProcessPoolExecutor if self.executor is None else ThreadPoolExecutor
            if pool_type is ProcessPoolExecutor:
                suspend_refresh = getattr(original_sink, "suspend_refresh", None)
                if callable(suspend_refresh):
                    suspend_refresh()
            futures: dict[
                Future[Any],
                tuple[ResolvedCase, Path, bool, float, str, int | None, Path | None],
            ] = {}
            relay = _ExecutorEventRelay(self.progress_sink)
            with pool_type(max_workers=effective_jobs) as pool:
                for case, config_path, case_directory in pending:
                    submitted = time.perf_counter()
                    submitted_utc = _utc_now()
                    self.progress_sink(
                        ProgressEvent(
                            event="case_queued",
                            current=completed,
                            total=total,
                            case_id=case.case_id,
                            unit_current=0,
                            unit_total=repeats_per_case,
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
                        relay.register(
                            case.case_id,
                            attempt_directory / "executor_events.jsonl",
                        )
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
                self.progress_sink(
                    ProgressEvent(
                        event="study_running",
                        message=f"running {effective_jobs} cases in parallel",
                    )
                )
                remaining = set(futures)
                while remaining:
                    done, _ = wait(
                        remaining,
                        timeout=0.5,
                        return_when=FIRST_COMPLETED,
                    )
                    relay.drain()
                    for future in done:
                        remaining.remove(future)
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
                        if attempt_directory is not None:
                            relay.unregister(
                                attempt_directory / "executor_events.jsonl"
                            )
                        records.append(payload)
                        completed += 1
                        self.progress_sink(
                            ProgressEvent(
                                event="case_finished",
                                current=completed,
                                total=total,
                                case_id=case.case_id,
                                unit_current=(
                                    repeats_per_case
                                ),
                                unit_total=repeats_per_case,
                                message=str(payload["status"]),
                            )
                        )
                relay.drain()
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
            "cases": [
                self._case_manifest_row(output, case) for case in expansion.cases
            ],
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
