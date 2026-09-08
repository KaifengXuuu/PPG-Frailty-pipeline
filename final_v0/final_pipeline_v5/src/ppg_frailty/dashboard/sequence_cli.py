"""Run a sparse Dashboard comparison as one canonical V5 study.

The public ``study_plan.v2`` schema represents Cartesian grids and its explicit
cases are reserved for the formal model catalogue. This module owns a separate
strict schema for arbitrary ordered, fully resolved cases. It reuses the same V5
study runner/finalizer as ``sweep.py`` so one sequence produces exactly one
``run/comparison/repeat/fold`` tree; it never stitches independent runs together.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import yaml
from ..study import (
    ExecutionSpec, OutputSpec, PreprocessingCacheSpec, ReportSpec, ResolvedCase,
    StudyExpansion, flatten_mapping, validate_canonical_expansion,
)
from ..study.runner import v5_experiment_executor
from ..v5.sweep import run_prepared_study
from .control_service import COMPARISON_SEQUENCE_SCHEMA, PIPELINE_OUTPUT, validate_comparison_sequence_payload

PIPELINE_ROOT = Path(__file__).resolve().parents[3]
INITIAL_REQUEST_SCHEMA = 'ppg_frailty.v5_comparison_sequence_request.v1'


@dataclass(frozen=True)
class _SequenceStudy:
    study_id: str
    kind: str
    purpose: str
    flow_position: str
    decision_role: str
    reference_case_id: str | None
    thesis_sections: tuple[str, ...]


@dataclass(frozen=True)
class _SequencePlan:
    """Small plan protocol consumed by the unchanged canonical StudyRunner."""
    schema_version: str
    study: _SequenceStudy
    execution: ExecutionSpec
    output: OutputSpec
    report: ReportSpec
    document: Mapping[str, Any]
    axes: tuple[Any, ...] = ()
    legacy_bridge: None = None

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.document))


def _request_path(value: str | Path, *, pipeline_root: Path) -> Path:
    raw = Path(value)
    target = raw.resolve() if raw.is_absolute() else (pipeline_root / raw).resolve()
    try:
        target.relative_to(pipeline_root)
    except ValueError as error:
        raise ValueError('comparison request must remain inside V5') from error
    if not target.is_file() or target.is_symlink():
        raise ValueError('comparison request must be an existing non-symlink YAML file')
    if target.suffix.lower() not in {'.yaml', '.yml'}:
        raise ValueError('comparison request must be YAML')
    return target


def load_sequence_request(
        value: str | Path,
        *,
        pipeline_root: str | Path = PIPELINE_ROOT) -> tuple[Path, Mapping[str, Any], tuple[dict[str, Any], ...], str]:
    """Read and fully bind one executable explicit-case request."""
    root = Path(pipeline_root).resolve()
    path = _request_path(value, pipeline_root=root)
    encoded = path.read_bytes()
    try:
        payload = yaml.safe_load(encoded)
    except yaml.YAMLError as error:
        raise ValueError('comparison request contains invalid YAML') from error
    if not isinstance(payload, Mapping):
        raise TypeError('comparison request must be a YAML mapping')
    cases = validate_comparison_sequence_payload(payload, pipeline_root=root, require_resolved_evidence=True)
    digest = hashlib.sha256(encoded).hexdigest()
    relative = path.relative_to(root)
    dashboard_cache = Path(PIPELINE_OUTPUT) / '.dashboard_requests' / 'comparison'
    if relative.parent == dashboard_cache and path.stem != f'sequence_{digest}':
        raise ValueError('comparison request filename is not bound to its SHA-256')
    return (path, dict(payload), cases, digest)


def _execution(payload: Mapping[str, Any]) -> ExecutionSpec:
    raw = dict(payload['execution'])
    cache = dict(raw['preprocessing_cache'])
    return ExecutionSpec(repeats=tuple((int(value) for value in raw['repeats'])),
                         folds=tuple((int(value) for value in raw['folds'])),
                         jobs=int(raw['jobs']),
                         device=str(raw['device']),
                         parallel_level=str(raw['parallel_level']),
                         continue_on_error=raw['continue_on_error'],
                         allow_parallel_deep=raw['allow_parallel_deep'],
                         measure_operational_costs=raw['measure_operational_costs'],
                         preprocessing_cache=PreprocessingCacheSpec(mode=str(cache['mode']),
                                                                    root=str(cache['root']),
                                                                    namespaces=tuple((str(value) for value in cache['namespaces'])),
                                                                    verify_source_sha256=cache['verify_source_sha256']))


def _report(payload: Mapping[str, Any]) -> ReportSpec:
    raw = dict(payload['report'])
    return ReportSpec(top_k=int(raw['top_k']),
                      detailed_configuration_top_k=int(raw['detailed_configuration_top_k']),
                      write_html=raw['write_html'],
                      write_static_figures=raw['write_static_figures'],
                      calibration_bins=int(raw['calibration_bins']),
                      figure_modules=tuple((str(value) for value in raw['figure_modules'])),
                      compact_mean_sd=raw['compact_mean_sd'],
                      write_excel_workbook=raw['write_excel_workbook'],
                      classification_tsne_random_state=int(raw['classification_tsne_random_state']),
                      classification_tsne_perplexity=float(raw['classification_tsne_perplexity']),
                      classification_tsne_max_samples=int(raw['classification_tsne_max_samples']),
                      classification_roc_macro_grid_points=int(raw['classification_roc_macro_grid_points']),
                      classification_score_histogram_bins=int(raw['classification_score_histogram_bins']))


def _parameter_rows(
        case_ids: Sequence[str],
        configs: Sequence[Mapping[str,
                                  Any]]) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    flattened = [flatten_mapping(config) for config in configs]
    paths = sorted(set().union(*(set(row) for row in flattened)) - {'config_id'})
    varied: list[dict[str, Any]] = []
    controlled: list[dict[str, Any]] = []
    changes = [dict() for _ in configs]
    for path in paths:
        values = [row.get(path, 'not_applicable') for row in flattened]
        keys = {json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False) for value in values}
        if len(keys) == 1:
            controlled.append({'parameter_path': path, 'value': values[0]})
            continue
        varied.append({'parameter_path': path, 'case_values': dict(zip(case_ids, copy.deepcopy(values)))})
        for changed, value in zip(changes, values):
            changed[path] = copy.deepcopy(value)
    return (tuple(changes), tuple(varied), tuple(controlled))


def prepare_sequence_study(path: Path, payload: Mapping[str, Any]) -> tuple[_SequencePlan, StudyExpansion]:
    """Create an exact prevalidated expansion without changing any configuration."""
    study_raw = dict(payload['study'])
    execution = _execution(payload)
    output = OutputSpec(root=str(dict(payload['output'])['root']))
    report = _report(payload)
    plan = _SequencePlan(schema_version=str(payload['schema_version']),
                         study=_SequenceStudy(
                             study_id=str(study_raw['study_id']),
                             kind=str(study_raw['kind']),
                             purpose=str(study_raw['purpose']),
                             flow_position=str(study_raw['flow_position']),
                             decision_role=str(study_raw['decision_role']),
                             reference_case_id=None if study_raw['reference_case_id'] is None else str(study_raw['reference_case_id']),
                             thesis_sections=tuple((str(value) for value in study_raw['thesis_sections']))),
                         execution=execution,
                         output=output,
                         report=report,
                         document=payload)
    raw_cases = [dict(value) for value in payload['cases']]
    case_ids = [str(value['case_id']) for value in raw_cases]
    configs = [copy.deepcopy(dict(value['resolved_config'])) for value in raw_cases]
    changes, varied, controlled = _parameter_rows(case_ids, configs)
    cases = tuple((ResolvedCase(case_id=case_id,
                                config=config,
                                changed_values=changed,
                                config_sha256=str(raw['config_sha256']),
                                is_reference=case_id == plan.study.reference_case_id)
                   for raw, case_id, config, changed in zip(raw_cases, case_ids, configs, changes)))
    expansion = StudyExpansion(plan=plan,
                               base_config_path=path,
                               base_config={
                                   'schema_version': COMPARISON_SEQUENCE_SCHEMA,
                                   'source_request_sha256': hashlib.sha256(path.read_bytes()).hexdigest()
                               },
                               cases=cases,
                               varied_parameters=varied,
                               controlled_parameters=controlled,
                               reference_case_id=plan.study.reference_case_id)
    return (plan, validate_canonical_expansion(expansion))


def run_sequence_request(value: str | Path,
                         *,
                         pipeline_root: str | Path = PIPELINE_ROOT,
                         run_name: str | None = None,
                         resume: str | None = None,
                         hash_predictions: bool | None = None,
                         dry_run: bool | None = None,
                         environment_policy: str | None = None,
                         environment_lock: str | None = None) -> int:
    """Run all exact cases inside one canonical V5 output root."""
    root = Path(pipeline_root).resolve()
    path, payload, _, request_sha256 = load_sequence_request(value, pipeline_root=root)
    if root != PIPELINE_ROOT.resolve():
        raise ValueError('sequence execution pipeline_root must be the installed V5 root')
    plan, expansion = prepare_sequence_study(path, payload)
    launch = dict(payload['launch'])
    selected_resume = resume
    selected_run_name = None if selected_resume is not None else run_name if run_name is not None else launch['run_name']
    args = argparse.Namespace(resume=selected_resume,
                              run_name=selected_run_name,
                              hash_predictions=launch['hash_predictions'] if hash_predictions is None else bool(hash_predictions),
                              dry_run=launch['dry_run'] if dry_run is None else bool(dry_run),
                              environment_policy=environment_policy or launch['environment_policy'],
                              environment_lock=environment_lock or launch['environment_lock'],
                              refit=bool(launch['refit']))
    bindings = [{
        'order': int(case['order']),
        'case_id': str(case['case_id']),
        'config_sha256': str(case['config_sha256']),
        'pipeline_command': str(dict(case['pipeline_request'])['command']),
        'pipeline_command_sha256': hashlib.sha256(str(dict(case['pipeline_request'])['command']).encode('utf-8')).hexdigest()
    } for case in payload['cases']]
    return run_prepared_study(args,
                              source=path,
                              plan=plan,
                              prepared_expansion=expansion,
                              initial_request_schema=INITIAL_REQUEST_SCHEMA,
                              request_metadata={
                                  'comparison_sequence': {
                                      'schema_version': COMPARISON_SEQUENCE_SCHEMA,
                                      'source_request_sha256': request_sha256,
                                      'ordered_case_bindings': bindings
                                  }
                              },
                              runner_executor=v5_experiment_executor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Validate or run one ordered Dashboard comparison as a single V5 run.'
    )
    commands = parser.add_subparsers(dest='command', required=True)
    validate = commands.add_parser('validate')
    validate.add_argument('--request', required=True)
    run = commands.add_parser('run')
    run.add_argument('--request', required=True)
    run.add_argument('--run-name')
    run.add_argument('--resume')
    run.add_argument('--hash-predictions', action=argparse.BooleanOptionalAction, default=None)
    run.add_argument('--dry-run', action=argparse.BooleanOptionalAction, default=None)
    run.add_argument('--environment-policy', choices=('exact', 'record'))
    run.add_argument('--environment-lock')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path, payload, cases, digest = load_sequence_request(args.request)
    if args.command == 'validate':
        print(
            json.dumps(
                {
                    'status': 'valid',
                    'request': path.relative_to(PIPELINE_ROOT).as_posix(),
                    'request_sha256': digest,
                    'case_count': len(cases),
                    'execution': 'one_run_with_ordered_comparison_cases',
                    'output_layout': 'run/comparison/repeat/fold'
                },
                ensure_ascii=False,
                indent=2))
        return 0
    return run_sequence_request(path,
                                run_name=args.run_name,
                                resume=args.resume,
                                hash_predictions=args.hash_predictions,
                                dry_run=args.dry_run,
                                environment_policy=args.environment_policy,
                                environment_lock=args.environment_lock)


__all__ = ['INITIAL_REQUEST_SCHEMA', 'build_parser', 'load_sequence_request', 'main', 'prepare_sequence_study', 'run_sequence_request']
