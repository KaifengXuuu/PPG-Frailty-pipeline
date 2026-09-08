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
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from .expand import ResolvedCase, StudyExpansion, expand_study
from .progress import CompositeProgressSink, JsonlProgressSink, NullProgressSink, ProgressEvent, ProgressSink
from .schema import StudyPlan

CaseExecutor = Callable[[ResolvedCase, Path, Path, StudyPlan, ProgressSink], Mapping[str, Any] | Any]
Phase0Runner = Callable[..., Mapping[str, Any] | Any]
PendingCase = tuple[ResolvedCase, Path, Path]
ProcessState = tuple[ResolvedCase, Path, bool, float, str, int | None, Path | None]
_OUTPUT_GROUPS = frozenset({'raw', 'fusion', 'feature_vector', 'feature_matrix'})
_PASS_STATUSES = frozenset({'passed', 'success', 'complete', 'completed'})
_COMPACT_CELL_FIELDS = ('status', 'repeat_index', 'fold_index', 'split_seed', 'training_seed', 'config_hash',
                        'preprocessing_hash', 'code_commit', 'source_version', 'model_machine_id', 'model_id',
                        'representation_mode', 'elapsed_seconds', 'retained_train_record_count',
                        'retained_oof_record_count', 'selected_record_count', 'oof_window_prediction_count',
                        'class_order', 'metrics', 'operational_metrics', 'preprocessing_cache_artifact',
                        'preprocessing_cache_summary')

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')

def _jsonable(value: Any) -> Any:
    if hasattr(value, 'to_dict') and callable(value.to_dict):
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
    if hasattr(value, 'item'):
        return _jsonable(value.item())
    raise TypeError(f'value is not JSON compatible: {type(value)!r}')

def _compact_experiment_result(value: Any) -> dict[str, Any]:
    """Keep the study index small; detailed experiment artifacts remain on disk."""
    def field(name: str, default: Any = None) -> Any:
        return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)

    cells = [{key: _jsonable(raw[key])
              for key in _COMPACT_CELL_FIELDS if key in raw} for raw in field('cell_results', ()) or ()
             if isinstance(raw, Mapping)]
    output_dir = field('output_dir')
    return {
        'schema_version': 'ppg_frailty.study_executor_result.v3', 'status': str(field('status', 'passed')),
        'scientific_scope': field('scientific_scope'), 'config_id': field('config_id'),
        'config_hash': field('config_hash'), 'repeat_indices': _jsonable(field('repeat_indices', ())),
        'fold_indices': _jsonable(field('fold_indices',
                                        ())), 'output_dir': None if output_dir is None else str(output_dir),
        'cell_results': cells, 'metrics': _jsonable(field('metrics', {})),
        'failure_reasons': _jsonable(field('failure_reasons', ())), 'detail_source': 'persisted_experiment_artifacts'
    }

def _compact_case_record(record: Mapping[str, Any]) -> dict[str, Any]:
    compact = dict(record)
    if isinstance(compact.get('result'), Mapping):
        compact['result'] = _compact_experiment_result(compact['result'])
    return compact

def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp-{time.time_ns()}')
    try:
        serialized = json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        temporary.write_text(serialized + '\n', encoding='utf-8')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

def _write_csv(path: Path, rows: tuple[Mapping[str, Any], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('\n', encoding='utf-8')
        return
    fields = list(dict.fromkeys((key for row in rows for key in row)))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(({
            key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value,
                                                                                     (dict, list, tuple)) else value
            for key, value in row.items()
        } for row in rows))

def _resume_contract(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    execution = payload.get('execution')
    execution = execution if isinstance(execution, Mapping) else {}
    contract = {str(key): value for key, value in payload.items() if key not in {'execution', 'output', 'report'}}
    contract['execution'] = {
        key: execution.get(key, False) if key == 'measure_operational_costs' else execution.get(key)
        for key in ('repeats', 'folds', 'device', 'measure_operational_costs')
    }
    return contract

def _study_object(plan: StudyPlan) -> str:
    return plan.axes[0].path.replace('.', '-') if plan.study.kind == 'ablation' else plan.study.study_id

def _safe_slug(value: str) -> str:
    return ''.join((character.lower() if character.isalnum() else '-' for character in value)).strip('-') or 'study'

def _contains_deep_case(expansion: StudyExpansion) -> bool:
    from ..module_registry import model_factory_contract
    for case in expansion.cases:
        model = case.config.get('model', {})
        if not isinstance(model, Mapping):
            continue
        model_id = model.get('model_id')
        if model_id is None:
            raise ValueError(f'study case {case.case_id} has no model.model_id')
        if model_factory_contract(str(model_id))['execution_backend'] == 'torch':
            return True
    return False

def _invoke_with_supported_kwargs(function: Callable[..., Any], **kwargs: Any) -> Any:
    parameters = inspect.signature(function).parameters
    accepts_kwargs = any((parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()))
    selected = kwargs if accepts_kwargs else {key: value for key, value in kwargs.items() if key in parameters}
    return function(**selected)

def _executor_progress_adapter(sink: ProgressSink, case_id: str, *, repeats: tuple[int, ...],
                               folds: tuple[int, ...]) -> Callable[[Any], None]:
    repeat_positions = {value: index + 1 for index, value in enumerate(repeats)}
    fold_positions = {value: index + 1 for index, value in enumerate(folds)}
    completed_repeats = 0

    def emit(value: Any) -> None:
        nonlocal completed_repeats
        event = ProgressEvent.from_value(value)
        updates: dict[str, Any] = {'case_id': event.case_id or case_id}
        if event.event in {'module_start', 'run_start'}:
            updates.update(unit_current=completed_repeats, unit_total=len(repeats), detail_current=0,
                           detail_total=len(folds),
                           detail_label='loading pipeline / preflight' if event.event == 'module_start' else 'CV setup')
        elif event.repeat in repeat_positions and event.fold in fold_positions:
            repeat_position = repeat_positions[int(event.repeat)]
            fold_position = fold_positions[int(event.fold)]
            detail_current = fold_position if event.event == 'cell_complete' else max(0, fold_position - 1)
            if event.event == 'cell_complete' and fold_position == len(folds):
                completed_repeats = max(completed_repeats, repeat_position)
            updates.update(
                unit_current=completed_repeats, unit_total=len(repeats), detail_current=detail_current,
                detail_total=len(folds),
                detail_label=f'CV repeat {repeat_position}/{len(repeats)} · fold {fold_position}/{len(folds)}')
        elif event.event == 'run_complete':
            completed_repeats = len(repeats)
            updates.update(unit_current=completed_repeats, unit_total=len(repeats), detail_current=len(folds),
                           detail_total=len(folds), detail_label='CV complete')
        elif event.event == 'run_error':
            updates.update(unit_current=completed_repeats, unit_total=len(repeats), detail_current=0, detail_total=0,
                           detail_label='CV failed')
        sink(replace(event, **updates))

    return emit

def _bridge_binding(case: ResolvedCase, bridge: Any) -> tuple[str, Mapping[str, Any]]:
    profile_id = str(case.changed_values.get('study.legacy_bridge_profile', ''))
    allowed = {str(profile['profile_id']) for profile in bridge.profiles}
    if profile_id not in allowed:
        raise ValueError(f'{case.case_id} lacks one frozen Legacy Bridge profile for design={bridge.design}')
    matches = tuple((profile for profile in bridge.profiles if str(profile['catalog_case_id']) == case.case_id))
    if len(matches) != 1:
        raise ValueError(f'{case.case_id} must bind exactly one Legacy Bridge profile')
    return (profile_id, matches[0])

def _execute_experiment(case: ResolvedCase, config_path: Path, case_directory: Path, plan: StudyPlan,
                        progress_sink: ProgressSink, *, v5_output_layout: bool) -> Mapping[str, Any]:
    """Delayed adapter to the canonical or isolated Legacy Bridge runner."""
    emit = _executor_progress_adapter(progress_sink, case.case_id, repeats=tuple(plan.execution.repeats),
                                      folds=tuple(plan.execution.folds))
    emit({'event': 'module_start', 'message': 'loading canonical pipeline and preflight'})
    from ppg_frailty import experiment as experiment_module
    output = case_directory if v5_output_layout else case_directory / 'experiment'
    bridge = plan.legacy_bridge
    full_runner = getattr(experiment_module, 'run_full_experiment', None)
    cell_name = 'run_legacy_bridge_outer_cell' if bridge is not None else 'run_outer_cell'
    cell_runner = getattr(experiment_module, cell_name, None)
    complete_5x5 = tuple(plan.execution.repeats) == tuple(range(5)) and tuple(plan.execution.folds) == tuple(range(5))
    if bridge is None and callable(full_runner) and complete_5x5:
        result = _invoke_with_supported_kwargs(full_runner, config_path=config_path, output_dir=output,
                                               repeats=plan.execution.repeats, folds=plan.execution.folds,
                                               progress_callback=emit,
                                               measure_operational_costs=plan.execution.measure_operational_costs,
                                               preprocessing_cache=plan.execution.preprocessing_cache.to_dict(),
                                               cell_directory_layout='nested' if v5_output_layout else 'flat')
        compact = _compact_experiment_result(result)
        del result
        gc.collect()
        return compact
    if not callable(cell_runner):
        if bridge is None and callable(full_runner) and (not complete_5x5):
            raise RuntimeError(
                'partial repeat/fold selection requires run_outer_cell; '
                'refusing to delegate it to a full-only runner'
            )
        required = cell_name if bridge is not None else 'run_full_experiment or run_outer_cell'
        raise RuntimeError(f'experiment adapter does not expose {required}')
    profile_id, profile = _bridge_binding(case, bridge) if bridge is not None else (None, None)
    cells: list[Any] = []
    failed_cells: list[str] = []
    for repeat in plan.execution.repeats:
        for fold in plan.execution.folds:
            cell_output = (
                output / f'repeat_{repeat:02d}' / f'fold_{fold:02d}' if v5_output_layout
                else output / f'repeat_{repeat:02d}_fold_{fold:02d}'
            )
            call: dict[str, Any] = {
                'config_path': config_path, 'output_dir': cell_output, 'repeat_index': repeat, 'fold_index': fold,
                'progress_callback': emit, 'measure_operational_costs': plan.execution.measure_operational_costs
            }
            if bridge is None:
                call['preprocessing_cache'] = plan.execution.preprocessing_cache.to_dict()
            elif bridge.uses_inline_profiles:
                assert profile is not None
                call.update(profile_id=profile_id, protocol_design=bridge.design,
                            profile_definition=copy.deepcopy(dict(profile)),
                            profile_definition_sha256=bridge.controls_sha256(profile))
            else:
                call.update(profile_id=profile_id, source_specification=bridge.source_specification,
                            source_specification_sha256=bridge.source_specification_sha256)
            raw = _invoke_with_supported_kwargs(cell_runner, **call)
            result = _compact_experiment_result(raw)
            del raw
            gc.collect()
            nested = result.get('cell_results') if isinstance(result, Mapping) else None
            cells.extend(nested if isinstance(nested, list) else [result])
            if isinstance(result, Mapping) and result.get('status') != 'passed':
                failed_cells.append(f"r{repeat}_f{fold}:{result.get('status')}")
    return {
        'status': 'failed_closed' if failed_cells else 'passed', 'config_id': case.config.get('config_id'),
        'cell_results': cells, 'output_dir': str(output), 'failure_reasons': failed_cells
    }

def default_experiment_executor(case: ResolvedCase, config_path: Path, case_directory: Path, plan: StudyPlan,
                                progress_sink: ProgressSink) -> Mapping[str, Any]:
    """Execute with the historical attempt/experiment artifact layout."""
    return _execute_experiment(case, config_path, case_directory, plan, progress_sink, v5_output_layout=False)

def v5_experiment_executor(case: ResolvedCase, config_path: Path, comparison_directory: Path, plan: StudyPlan,
                           progress_sink: ProgressSink) -> Mapping[str, Any]:
    """Execute unchanged numerical cells into ``comparison/repeat/fold``."""
    return _execute_experiment(case, config_path, comparison_directory, plan, progress_sink, v5_output_layout=True)

def _process_default_case(request: Mapping[str, Any]) -> Mapping[str, Any]:
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'

    def optional(name: str) -> str | None:
        return None if request.get(name) is None else str(request[name])

    case = ResolvedCase(case_id=str(request['case_id']), config=dict(request['config']),
                        changed_values=dict(request['changed_values']), config_sha256=str(request['config_sha256']),
                        is_reference=bool(request['is_reference']), output_group=optional('output_group'),
                        catalog_entry=optional('catalog_entry'), screen_profile_id=optional('screen_profile_id'),
                        rationale=optional('rationale'))
    from .expand import parse_study_plan
    plan = parse_study_plan(dict(request['plan']))
    attempt = Path(str(request['attempt_directory']))
    executor = v5_experiment_executor if request.get('output_layout', 'legacy') == 'v5' else default_experiment_executor
    execution = Path(str(request.get('execution_directory', attempt)))
    sink = JsonlProgressSink(attempt / 'executor_events.jsonl')
    return executor(case, Path(str(request['config_path'])), execution, plan, sink)

class _ExecutorEventRelay:
    """Relay complete child JSONL rows to the parent without new IPC."""
    def __init__(self, sink: ProgressSink) -> None:
        self._sink = sink
        self._case_by_path: dict[Path, str] = {}
        self._offsets: dict[Path, int] = {}
        self._buffers: dict[Path, bytes] = {}

    def register(self, case_id: str, path: Path) -> None:
        self._case_by_path[path], self._offsets[path], self._buffers[path] = (case_id, 0, b'')

    def unregister(self, path: Path) -> None:
        self._case_by_path.pop(path, None)
        self._offsets.pop(path, None)
        self._buffers.pop(path, None)

    def drain(self) -> int:
        relayed = 0
        for path, case_id in tuple(self._case_by_path.items()):
            if not path.is_file():
                continue
            with path.open('rb') as stream:
                stream.seek(self._offsets[path])
                chunk = stream.read()
            if not chunk:
                continue
            self._offsets[path] += len(chunk)
            rows = (self._buffers[path] + chunk).split(b'\n')
            self._buffers[path] = rows.pop()
            for encoded in filter(bytes.strip, rows):
                try:
                    event = ProgressEvent.from_value(json.loads(encoded.decode('utf-8')))
                except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
                    continue
                self._sink(event if event.case_id is not None else replace(event, case_id=case_id))
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
        return {'schema_version': 'ppg_frailty.study_run_result.v2', **_jsonable(asdict(self))}

class StudyRunner:
    """Materialize cases and execute each through one canonical adapter."""
    def __init__(self, *, pipeline_root: str | Path, executor: CaseExecutor | None = None,
                 phase0_runner: Phase0Runner | None = None, progress_sink: ProgressSink | None = None,
                 output_layout: str = 'legacy') -> None:
        if output_layout not in {'legacy', 'v5'}:
            raise ValueError('output_layout must be legacy or v5')
        self.pipeline_root = Path(pipeline_root).resolve()
        self.executor = executor
        self.phase0_runner = phase0_runner
        self.progress_sink = progress_sink or NullProgressSink()
        self.output_layout = output_layout

    def expand(self, plan: StudyPlan) -> StudyExpansion:
        return expand_study(plan, pipeline_root=self.pipeline_root)

    def _run_phase0_audit(self, plan: StudyPlan, *, output_directory: Path) -> Mapping[str, Any] | None:
        """Run Phase 0 as advisory evidence; never authorize or block training."""
        bridge = plan.legacy_bridge
        if bridge is None or not bool(bridge.phase0.get('enabled', False)):
            return None
        base = {
            'schema_version': 'ppg_frailty.legacy_v2_phase0_advisory.v1', 'advisory_only': True,
            'affects_training_execution': False, 'training_blocked': False, 'recorded_utc': _utc_now(),
            'source_specification': bridge.source_specification,
            'declared_source_specification_sha256': bridge.source_specification_sha256
        }
        try:
            repository_root = self.pipeline_root.parents[1]
            source = (repository_root / bridge.source_specification).resolve()
            source.relative_to(repository_root)
            if not source.is_file():
                raise FileNotFoundError(source)
            if self.phase0_runner is None:
                from ppg_frailty.audit.legacy_v2_bridge import run_legacy_v2_phase0
                runner = run_legacy_v2_phase0
            else:
                runner = self.phase0_runner
            arguments: dict[str, Any] = {
                'repository_root': repository_root, 'pipeline_root': self.pipeline_root, 'phase0_spec': bridge.phase0,
                'source_specification': bridge.source_specification,
                'source_specification_sha256': bridge.source_specification_sha256
            }
            if self.output_layout == 'v5':
                arguments.update(artifact_root=output_directory, generate_report=False)
            payload = _jsonable(_invoke_with_supported_kwargs(runner, **arguments))
            if not isinstance(payload, Mapping):
                raise TypeError('legacy bridge Phase 0 result must be a mapping')
            json.dumps(payload, allow_nan=False)
            return {
                **base, 'audit_status': 'completed', 'audit_decision': payload.get('decision'),
                'audit_result': dict(payload)
            }
        except Exception as error:
            return {
                **base, 'audit_status': 'error', 'audit_decision': None, 'error_type': type(error).__name__,
                'error': str(error), 'traceback': traceback.format_exc()
            }

    def _new_output(self, plan: StudyPlan, output_root: str | Path | None, *, run_name: str | None = None) -> Path:
        raw = Path(output_root or plan.output.root)
        root = raw.resolve() if raw.is_absolute() else (self.pipeline_root / raw).resolve()
        root.mkdir(parents=True, exist_ok=True)
        if run_name is not None:
            name = str(run_name).strip()
            valid = name and name not in {'.', '..'} and (Path(name).name == name)
            if not valid or any((not (value.isalnum() or value in '._-') for value in name)):
                raise ValueError('run_name must be one portable directory name')
            candidate = root / name
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base = root / f'{stamp}_{plan.study.kind}_{_safe_slug(_study_object(plan))}'
        for index in range(1, 1000):
            candidate = base if index == 1 else base.with_name(f'{base.name}_{index:02d}')
            try:
                candidate.mkdir(parents=False, exist_ok=False)
                return candidate
            except FileExistsError:
                pass
        raise RuntimeError(f'cannot create unique study directory below {root}')

    @staticmethod
    def _output_group(case: ResolvedCase) -> str | None:
        raw = getattr(case, 'output_group', None)
        if raw is None:
            return None
        group = str(raw).strip()
        if group not in _OUTPUT_GROUPS:
            raise ValueError(f'case {case.case_id} output_group must be one of {sorted(_OUTPUT_GROUPS)}')
        return group

    def _case_paths(self, output: Path, case: ResolvedCase) -> tuple[Path, Path]:
        if self.output_layout == 'v5':
            config = output / 'configs' / f'{case.case_id}.yaml'
            state = output / '.runner_state' / case.case_id
        else:
            group = self._output_group(case)
            state = output / group / case.case_id if group else output / 'cases' / case.case_id
            config = state / 'resolved_config.yaml' if group else output / 'resolved_configs' / f'{case.case_id}.yaml'
        resolved = output.resolve()
        config.resolve().relative_to(resolved)
        state.resolve().relative_to(resolved)
        return (config, state)

    def _published_case_directory(self, output: Path, case: ResolvedCase) -> Path:
        target = output / case.case_id if self.output_layout == 'v5' else self._case_paths(output, case)[1]
        target.resolve().relative_to(output.resolve())
        return target

    def _published_case_from_state(self, case: ResolvedCase, state_directory: Path) -> Path:
        if self.output_layout == 'v5':
            return self._published_case_directory(state_directory.parent.parent, case)
        return state_directory

    def _case_manifest_row(self, output: Path, case: ResolvedCase) -> dict[str, Any]:
        config, _ = self._case_paths(output, case)
        published = self._published_case_directory(output, case)
        return {
            **case.to_dict(), 'output_group': None if self.output_layout == 'v5' else self._output_group(case),
            'case_directory': published.relative_to(output).as_posix(),
            'resolved_config_path': config.relative_to(output).as_posix()
        }

    def _materialize(self, expansion: StudyExpansion, output: Path, *, resumed: bool) -> None:
        output.mkdir(parents=True, exist_ok=True)
        tables = output / 'tables'
        tables.mkdir(exist_ok=True)
        plan_path = output / 'study_plan.yaml'
        if resumed:
            if not plan_path.is_file():
                raise FileNotFoundError('resume directory has no study_plan.yaml')
            existing = yaml.safe_load(plan_path.read_text(encoding='utf-8'))
            if not isinstance(existing,
                              Mapping) or _resume_contract(existing) != _resume_contract(expansion.plan.to_dict()):
                raise ValueError(
                    'resume study-plan drift: scientific case definitions and repeats/folds/device must match; '
                    'jobs, output root, and report presentation may be changed'
                )
        else:
            encoded_plan = yaml.safe_dump(expansion.plan.to_dict(), sort_keys=False, allow_unicode=True)
            plan_path.write_text(encoded_plan, encoding='utf-8')
        rows: list[Mapping[str, Any]] = []
        for case in expansion.cases:
            config, state = self._case_paths(output, case)
            config.parent.mkdir(parents=True, exist_ok=True)
            encoded = yaml.safe_dump(dict(case.config), sort_keys=False, allow_unicode=True)
            if config.exists() and config.read_text(encoding='utf-8') != encoded:
                raise ValueError(f'resume config drift for case {case.case_id}')
            if not config.exists():
                config.write_text(encoded, encoding='utf-8')
            rows.append(self._case_manifest_row(output, case))
            state.mkdir(parents=True, exist_ok=True)
        _write_csv(tables / 'resolved_cases.csv', tuple(rows))
        _write_csv(tables / 'varied_parameters.csv', expansion.varied_parameters)
        _write_csv(tables / 'controlled_parameters.csv', expansion.controlled_parameters)

    def _attempt_number(self, case_directory: Path) -> int:
        attempts = case_directory / 'attempts'
        attempts.mkdir(exist_ok=True)
        numbers: list[int] = []
        for path in attempts.glob('attempt_*'):
            try:
                numbers.append(int(path.stem.rsplit('_', 1)[-1]))
            except ValueError:
                pass
        return max(numbers, default=0) + 1

    def _create_attempt_directory(self, case_directory: Path) -> tuple[int, Path]:
        attempt = self._attempt_number(case_directory)
        directory = case_directory / 'attempts' / f'attempt_{attempt:03d}'
        directory.mkdir(parents=False, exist_ok=False)
        return (attempt, directory)

    def _artifact_root(self, normalized_result: Any, *, case_directory: Path, attempt_directory: Path) -> str:
        declared = normalized_result.get('output_dir') if isinstance(normalized_result, Mapping) else None
        if declared:
            raw = Path(str(declared))
            target = raw.resolve() if raw.is_absolute() else (attempt_directory / raw).resolve()
        else:
            target = attempt_directory.resolve()
        allowed = case_directory.resolve() if self.output_layout == 'v5' else attempt_directory.resolve()
        target.relative_to(allowed)
        return target.relative_to(case_directory.resolve()).as_posix()

    def _archive_failed_v5_output(self, published_directory: Path, attempt_directory: Path) -> None:
        if self.output_layout != 'v5' or not published_directory.exists():
            return
        destination = attempt_directory / 'failed_output'
        if destination.exists():
            raise FileExistsError(destination)
        os.replace(published_directory, destination)

    def _existing_pass(self, case: ResolvedCase, case_directory: Path) -> Mapping[str, Any] | None:
        path = case_directory / 'case_result.json'
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload.get('config_sha256') != case.config_sha256:
            raise ValueError(f'resume result config drift for {case.case_id}')
        return _compact_case_record(payload) if payload.get('status') == 'passed' else None

    @staticmethod
    def _incomplete_attempts(state: Path) -> list[tuple[int, Path]]:
        attempts = state / 'attempts'
        if not attempts.is_dir():
            return []
        found: list[tuple[int, Path]] = []
        for path in attempts.glob('attempt_[0-9][0-9][0-9]'):
            if (path / 'attempt_result.json').exists():
                continue
            try:
                found.append((int(path.name.removeprefix('attempt_')), path))
            except ValueError:
                pass
        return found

    def _publish_recovery(self, case: ResolvedCase, state: Path, attempt: int, attempt_directory: Path, published: Path,
                          result: Any, *, artifact_root: str, started: float, started_utc: str, index_only: bool,
                          interrupted: str | None) -> Mapping[str, Any]:
        payload = {
            'schema_version': 'ppg_frailty.study_case_result.v2', 'case_id': case.case_id,
            'config_sha256': case.config_sha256, 'status': 'passed', 'attempt': attempt,
            'attempt_directory': attempt_directory.relative_to(state).as_posix(), 'artifact_root': artifact_root,
            'started_utc': started_utc, 'finished_utc': _utc_now(), 'elapsed_seconds': time.perf_counter() - started,
            'result': _compact_experiment_result(result), 'recovered_from_complete_interrupted_staging': True,
            'recovery_index_only': index_only, 'interrupted_staging_preserved': interrupted
        }
        _atomic_json(attempt_directory / 'attempt_result.json', payload)
        _atomic_json(published / 'case_result.json', payload)
        return _compact_case_record(payload)

    def _recover_complete_interrupted_pass(self, case: ResolvedCase, config_path: Path, case_directory: Path,
                                           plan: StudyPlan) -> Mapping[str, Any] | None:
        """Reuse only a complete canonical 5x5 run; partial fitted state is never resumed."""
        complete_5x5 = tuple(plan.execution.repeats) == tuple(range(5)) and tuple(plan.execution.folds) == tuple(
            range(5))
        if self.executor is not None or plan.legacy_bridge is not None or (not complete_5x5):
            return None
        if self.output_layout == 'v5':
            return self._recover_complete_interrupted_v5_pass(case, config_path, case_directory, plan)
        candidates: list[tuple[int, Path, Path | None, bool]] = []
        for attempt, directory in self._incomplete_attempts(case_directory):
            staging = tuple((path for path in directory.glob('.experiment.staging.*') if path.is_dir()))
            published = (directory / 'experiment').is_dir()
            if published or len(staging) == 1:
                candidates.append((attempt, directory, staging[0] if len(staging) == 1 else None, published))
        if not candidates:
            return None
        attempt, directory, interrupted, published = max(candidates, key=lambda item: item[0])
        from .recovery import recover_completed_full_experiment_staging, validate_published_recovered_experiment
        started, started_utc = (time.perf_counter(), _utc_now())
        emit = _executor_progress_adapter(self.progress_sink, case.case_id, repeats=tuple(plan.execution.repeats),
                                          folds=tuple(plan.execution.folds))
        if published:
            result = validate_published_recovered_experiment(config_path, output_dir=directory / 'experiment',
                                                             repeats=plan.execution.repeats, folds=plan.execution.folds)
        elif interrupted is not None:
            result = recover_completed_full_experiment_staging(
                config_path, interrupted_staging=interrupted, output_dir=directory / 'experiment',
                repeats=plan.execution.repeats, folds=plan.execution.folds,
                measure_operational_costs=plan.execution.measure_operational_costs, progress_callback=emit)
        else:
            result = None
        if result is None:
            return None
        preserved = None if interrupted is None else interrupted.relative_to(case_directory).as_posix()
        artifact = (directory / 'experiment').relative_to(case_directory).as_posix()
        return self._publish_recovery(case, case_directory, attempt, directory, case_directory, result,
                                      artifact_root=artifact,
                                      started=started, started_utc=started_utc, index_only=published,
                                      interrupted=preserved)

    def _recover_complete_interrupted_v5_pass(self, case: ResolvedCase, config_path: Path, state_directory: Path,
                                              plan: StudyPlan) -> Mapping[str, Any] | None:
        attempts = self._incomplete_attempts(state_directory)
        if not attempts:
            return None
        attempt, directory = max(attempts, key=lambda item: item[0])
        output = state_directory.parent.parent
        published = self._published_case_directory(output, case)
        staging = tuple(
            sorted((path for path in output.glob(f'.{case.case_id}.staging.*') if path.is_dir()),
                   key=lambda path: path.name))
        if not published.exists() and len(staging) > 1:
            raise RuntimeError(
                f'multiple interrupted staging trees exist for {case.case_id}; refusing to guess or retrain')
        from .recovery import recover_completed_full_experiment_staging, validate_published_complete_experiment
        started, started_utc = (time.perf_counter(), _utc_now())
        emit = _executor_progress_adapter(self.progress_sink, case.case_id, repeats=tuple(plan.execution.repeats),
                                          folds=tuple(plan.execution.folds))
        interrupted = staging[0] if len(staging) == 1 else None
        if published.exists():
            result = validate_published_complete_experiment(config_path, output_dir=published,
                                                            repeats=plan.execution.repeats, folds=plan.execution.folds,
                                                            cell_directory_layout='nested')
            if result is None:
                raise RuntimeError(
                    f'published V5 case {published} is not a complete validated 5x5 result; '
                    'refusing to overwrite it or retrain'
                )
            index_only = True
        elif interrupted is not None:
            result = recover_completed_full_experiment_staging(
                config_path, interrupted_staging=interrupted, output_dir=published, repeats=plan.execution.repeats,
                folds=plan.execution.folds, measure_operational_costs=plan.execution.measure_operational_costs,
                progress_callback=emit, cell_directory_layout='nested')
            index_only = False
        else:
            return None
        if result is None:
            assert interrupted is not None
            archived = directory / 'interrupted_partial_staging'
            if archived.exists():
                raise FileExistsError(archived)
            os.replace(interrupted, archived)
            payload = {
                'schema_version': 'ppg_frailty.study_case_result.v2', 'case_id': case.case_id,
                'config_sha256': case.config_sha256, 'status': 'interrupted_incomplete', 'attempt': attempt,
                'attempt_directory': directory.relative_to(state_directory).as_posix(),
                'artifact_root': archived.relative_to(state_directory).as_posix(), 'started_utc': None,
                'finished_utc': _utc_now(), 'elapsed_seconds': None, 'model_fits_reused': False,
                'reason': 'partial V2 full-run state is not mathematically resumable'
            }
            _atomic_json(directory / 'attempt_result.json', payload)
            return None
        preserved = None if interrupted is None else interrupted.relative_to(output).as_posix()
        return self._publish_recovery(case, state_directory, attempt, directory, published, result, artifact_root='.',
                                      started=started, started_utc=started_utc, index_only=index_only,
                                      interrupted=preserved)

    def _run_one(self, case: ResolvedCase, config_path: Path, case_directory: Path, plan: StudyPlan,
                 executor: CaseExecutor) -> Mapping[str, Any]:
        attempt, directory = self._create_attempt_directory(case_directory)
        published = self._published_case_from_state(case, case_directory)
        execution = published if self.output_layout == 'v5' else directory
        return self._finish_attempt(case, case_directory, attempt, directory,
                                    lambda: executor(case, config_path, execution, plan, self.progress_sink))

    def _finish_attempt(self, case: ResolvedCase, case_directory: Path, attempt: int,
                        attempt_directory: Path, execute: Callable[[], Any], *,
                        started: float | None = None, started_utc: str | None = None) -> Mapping[str, Any]:
        """Run or collect one executor and persist its single case record."""
        started = time.perf_counter() if started is None else started
        started_utc = _utc_now() if started_utc is None else started_utc
        published = self._published_case_from_state(case, case_directory)
        common = {
            'schema_version': 'ppg_frailty.study_case_result.v2', 'case_id': case.case_id,
            'config_sha256': case.config_sha256, 'attempt': attempt,
            'attempt_directory': attempt_directory.relative_to(case_directory).as_posix(), 'started_utc': started_utc
        }
        try:
            normalized = _jsonable(execute())
            reported = str(normalized.get('status', 'passed')) if isinstance(normalized, Mapping) else 'passed'
            passed = reported in _PASS_STATUSES
            if not passed:
                self._archive_failed_v5_output(published, attempt_directory)
            payload = {
                **common, 'status': 'passed' if passed else 'failed',
                'artifact_root': self._artifact_root(normalized, case_directory=published,
                                                     attempt_directory=attempt_directory), 'result': normalized
            }
            if not passed:
                payload.update(error_type='CanonicalExperimentFailedClosed',
                               error=f'executor reported status={reported}')
        except Exception as error:
            self._archive_failed_v5_output(published, attempt_directory)
            payload = {
                **common, 'status': 'failed',
                'artifact_root': attempt_directory.relative_to(case_directory).as_posix(),
                'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc()
            }
        payload.update(finished_utc=_utc_now(), elapsed_seconds=time.perf_counter() - started)
        _atomic_json(attempt_directory / 'attempt_result.json', payload)
        destination = published if payload['status'] == 'passed' else case_directory
        _atomic_json(destination / 'case_result.json', payload)
        return payload

    @staticmethod
    def _case_event(sink: ProgressSink, event: str, case: ResolvedCase, completed: int, total: int, repeats: int, *,
                    finished: bool = False, message: str | None = None) -> None:
        sink(
            ProgressEvent(event=event, current=completed, total=total, case_id=case.case_id,
                          unit_current=repeats if finished else 0, unit_total=repeats, message=message or ''))

    def _pending_cases(self, expansion: StudyExpansion, output: Path, plan: StudyPlan, resumed: bool,
                       repeats: int) -> tuple[list[Mapping[str, Any]], list[PendingCase], int]:
        records: list[Mapping[str, Any]] = []
        pending: list[PendingCase] = []
        for case in expansion.cases:
            config, state = self._case_paths(output, case)
            existing = self._existing_pass(case, self._published_case_directory(output, case)) if resumed else None
            if existing is None and resumed:
                existing = self._recover_complete_interrupted_pass(case, config, state, plan)
            if existing is None:
                pending.append((case, config, state))
            else:
                records.append(existing)
                self._case_event(self.progress_sink, 'case_resumed', case, len(records), len(expansion.cases), repeats,
                                 finished=True)
        return (records, pending, len(records))

    def _run_serial(self, pending: list[PendingCase], plan: StudyPlan, executor: CaseExecutor,
                    records: list[Mapping[str, Any]], total: int, repeats: int) -> None:
        self.progress_sink(ProgressEvent(event='study_running', message='running cases'))
        completed = len(records)
        for case, config, state in pending:
            self._case_event(self.progress_sink, 'case_started', case, completed, total, repeats)
            payload = self._run_one(case, config, state, plan, executor)
            records.append(payload)
            completed += 1
            self._case_event(self.progress_sink, 'case_finished', case, completed, total, repeats, finished=True,
                             message=str(payload['status']))
            if payload['status'] != 'passed' and (not plan.execution.continue_on_error):
                break

    def _process_request(self, case: ResolvedCase, config: Path, state: Path, plan: StudyPlan,
                         attempt: Path) -> dict[str, Any]:
        execution = self._published_case_from_state(case, state) if self.output_layout == 'v5' else attempt
        return {
            **case.to_dict(), 'config': dict(case.config), 'plan': plan.to_dict(), 'config_path': str(config),
            'attempt_directory': str(attempt), 'execution_directory': str(execution),
            'output_layout': self.output_layout
        }

    def _run_parallel(self, pending: list[PendingCase], plan: StudyPlan, executor: CaseExecutor,
                      records: list[Mapping[str, Any]], total: int, repeats: int, jobs: int,
                      original_sink: ProgressSink) -> None:
        pool_type = ProcessPoolExecutor if self.executor is None else ThreadPoolExecutor
        if pool_type is ProcessPoolExecutor:
            suspend = getattr(original_sink, 'suspend_refresh', None)
            if callable(suspend):
                suspend()
        futures: dict[Future[Any], ProcessState] = {}
        relay = _ExecutorEventRelay(self.progress_sink)
        with pool_type(max_workers=jobs) as pool:
            for case, config, state in pending:
                submitted, submitted_utc = (time.perf_counter(), _utc_now())
                self._case_event(self.progress_sink, 'case_queued', case, len(records), total, repeats)
                if self.executor is None:
                    attempt, directory = self._create_attempt_directory(state)
                    future = pool.submit(_process_default_case,
                                         self._process_request(case, config, state, plan, directory))
                    relay.register(case.case_id, directory / 'executor_events.jsonl')
                    wrap = True
                else:
                    attempt, directory = (None, None)
                    future = pool.submit(self._run_one, case, config, state, plan, executor)
                    wrap = False
                futures[future] = (case, state, wrap, submitted, submitted_utc, attempt, directory)
            self.progress_sink(ProgressEvent(event='study_running', message=f'running {jobs} cases in parallel'))
            remaining = set(futures)
            while remaining:
                done, _ = wait(remaining, timeout=0.5, return_when=FIRST_COMPLETED)
                relay.drain()
                for future in done:
                    remaining.remove(future)
                    case, state, wrap, started, started_utc, attempt, directory = futures[future]
                    if wrap:
                        payload = self._wrap_process_result(future, case, state, attempt=attempt,
                                                            attempt_directory=directory, started=started,
                                                            started_utc=started_utc)
                    else:
                        payload = future.result()
                    if directory is not None:
                        relay.unregister(directory / 'executor_events.jsonl')
                    records.append(payload)
                    self._case_event(self.progress_sink, 'case_finished', case, len(records), total, repeats,
                                     finished=True, message=str(payload['status']))
            relay.drain()

    @staticmethod
    def _counts(records: tuple[Mapping[str, Any], ...], total: int, cells_per_case: int) -> dict[str, int]:
        passed_cases = sum((record.get('status') == 'passed' for record in records))
        failed_cases = sum((record.get('status') == 'failed' for record in records))
        reported_cells = passed_cells = failed_cells = 0
        for record in records:
            result = record.get('result') if isinstance(record.get('result'), Mapping) else {}
            cells = result.get('cell_results') if isinstance(result.get('cell_results'), list) else []
            for cell in (value for value in cells if isinstance(value, Mapping)):
                reported_cells += 1
                if str(cell.get('status')) in _PASS_STATUSES:
                    passed_cells += 1
                else:
                    failed_cells += 1
        planned_cells = total * cells_per_case
        if reported_cells > planned_cells:
            raise RuntimeError('reported cell count exceeds the declared study plan')
        return {
            'planned_case_count': total, 'passed_case_count': passed_cases, 'failed_case_count': failed_cases,
            'not_run_case_count': total - passed_cases - failed_cases, 'planned_cell_count': planned_cells,
            'reported_cell_count': reported_cells, 'passed_cell_count': passed_cells, 'failed_cell_count': failed_cells,
            'not_run_cell_count': planned_cells - reported_cells
        }

    def run(self, plan: StudyPlan, *, output_root: str | Path | None = None, resume_directory: str | Path | None = None,
            run_name: str | None = None) -> StudyRunResult:
        self.progress_sink(ProgressEvent(event='study_preparing', message='resolving cases and materializing configs'))
        expansion = self.expand(plan)
        resumed = resume_directory is not None
        if resumed and run_name is not None:
            raise ValueError('run_name cannot be combined with resume_directory')
        output = Path(resume_directory).resolve() if resumed else self._new_output(plan, output_root, run_name=run_name)
        if resumed and (not output.is_dir()):
            raise FileNotFoundError(output)
        phase0 = self._run_phase0_audit(plan, output_directory=output)
        if phase0 is not None:
            _atomic_json(output / 'phase0_audit.json', phase0)
        self._materialize(expansion, output, resumed=resumed)
        jsonl = JsonlProgressSink(output / 'progress_events.jsonl')
        original_sink = self.progress_sink
        self.progress_sink = CompositeProgressSink((original_sink, jsonl), close_sinks=(jsonl, ))
        requested_jobs = int(plan.execution.jobs)
        jobs, message = (requested_jobs, f'jobs={requested_jobs}')
        if _contains_deep_case(expansion) and requested_jobs > 1 and (not plan.execution.allow_parallel_deep):
            jobs, message = (1, f'deep model detected; jobs reduced {requested_jobs}->1')
        if not plan.execution.continue_on_error and jobs > 1:
            jobs = 1
            message = (
                'continue_on_error=false requires deterministic fail-fast order; '
                f'jobs reduced {requested_jobs}->1'
            )
        total, repeats = (len(expansion.cases), len(plan.execution.repeats))
        self.progress_sink(
            ProgressEvent(event='study_started', current=0, total=total, unit_current=0, unit_total=repeats,
                          message=message))
        records, pending, resumed_count = self._pending_cases(expansion, output, plan, resumed, repeats)
        executor = self.executor or (v5_experiment_executor
                                     if self.output_layout == 'v5' else default_experiment_executor)
        if jobs == 1:
            self._run_serial(pending, plan, executor, records, total, repeats)
        else:
            self._run_parallel(pending, plan, executor, records, total, repeats, jobs, original_sink)
        by_case = {str(record['case_id']): record for record in records}
        ordered = tuple((by_case[case.case_id] for case in expansion.cases if case.case_id in by_case))
        cells_per_case = len(plan.execution.repeats) * len(plan.execution.folds)
        counts = self._counts(ordered, total, cells_per_case)
        passed = counts['passed_case_count']
        status = 'passed' if passed == total else 'failed' if passed == 0 else 'partial'
        result = StudyRunResult(status=status, output_directory=output, resumed_case_count=resumed_count,
                                effective_jobs=jobs, case_records=ordered, **counts)
        plan_payload = plan.to_dict()
        manifest = {
            'schema_version': 'ppg_frailty.study_manifest.v2',
            'output_layout': 'comparison/repeat/fold' if self.output_layout == 'v5' else 'legacy', 'status': status,
            'created_or_resumed_utc': _utc_now(), 'study': plan_payload['study'],
            'base_config': str(expansion.base_config_path), 'reference_case_id': expansion.reference_case_id,
            'execution': plan_payload['execution'], 'effective_jobs': jobs,
            **counts, 'resumed_case_count': resumed_count,
            'cases': [self._case_manifest_row(output, case) for case in expansion.cases]
        }
        _atomic_json(output / 'study_manifest.json', manifest)
        _atomic_json(output / 'study_run_result.json', result.to_dict())
        finished = counts['passed_case_count'] + counts['failed_case_count']
        self.progress_sink(ProgressEvent(event='study_finished', current=finished, total=total, message=status))
        close = getattr(self.progress_sink, 'close', None)
        if callable(close):
            close()
        self.progress_sink = original_sink
        return result

    def _wrap_process_result(self, child_future: Future[Any], case: ResolvedCase, case_directory: Path, *,
                             attempt: int | None,
                             attempt_directory: Path | None, started: float, started_utc: str) -> Mapping[str, Any]:
        if attempt is None or attempt_directory is None:
            raise RuntimeError('process result lost its allocated attempt directory')
        return self._finish_attempt(case, case_directory, attempt, attempt_directory, child_future.result,
                                    started=started, started_utc=started_utc)
