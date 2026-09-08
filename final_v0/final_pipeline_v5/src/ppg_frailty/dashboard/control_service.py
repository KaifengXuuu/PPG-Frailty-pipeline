"""V5 dashboard control-plane services.

The dashboard is intentionally a thin presentation adapter.  Configuration is
resolved by :mod:`ppg_frailty.v5.configuration`, training and reporting are
launched through the same root CLI scripts, and inference is delegated to the
optional V5 inference service.  No numerical pipeline algorithm lives here.
"""
from __future__ import annotations
import copy
import hashlib
import importlib
import inspect
import json
import math
import re
import shlex
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import yaml
from ..module_registry import list_modules, model_factory_contract, registry_sha256
from ..v5.configuration import resolve_configuration

PIPELINE_OUTPUT = 'pipeline_output'
REPORT_OUTPUT = 'report_output'
MODEL_CONFIG = 'model_config'
COMPARISON_SEQUENCE_SCHEMA = 'ppg_frailty.dashboard_comparison_sequence.v2'
INHERIT_MODULE = '__yaml__'
SINGLE_PARTICIPANT_NOTICE = (
    'One participant supports QC and file/role/participant probabilities only. '
    'ROC/AUC, cohort confusion matrices, and significance tests require a labelled '
    'multi-participant dataset with adequate class coverage.'
)
MISSING_B_TODO = (
    'V5 TODO only: a missing-B calibration ablation is not implemented. Dynamic '
    'R/S/W inference remains fail-closed without a same-participant B recording.'
)
INFERENCE_SOURCE_CONTRACT: dict[str, Any] = {
    'provenance': 'user_declared',
    'sampling_rate_hz': 400,
    'channel_order': ['RED', 'IR', 'AX', 'AY', 'AZ', 'GX', 'GY', 'GZ'],
    'accelerometer_unit': 'g',
    'gyroscope_unit': 'deg/s',
    'synchrony': 'row_aligned_eight_channel_fixed_grid_no_timestamp'
}
INFERENCE_SOURCE_CONFIRMATION = (
    'I confirm that every selected CSV is row-aligned at exactly 400 Hz, has the '
    'ordered header RED,IR,AX,AY,AZ,GX,GY,GZ, uses acceleration in g and '
    'gyroscope in deg/s, and contains no timestamp column.'
)
_SAFE_NAME = re.compile('^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$')
_DYNAMIC_ROLE_FAMILIES = frozenset({'R', 'S', 'W'})
_COMPLETE_STATUSES = frozenset({'passed', 'success', 'complete', 'completed'})


@dataclass(frozen=True)
class CommandRequest:
    """A shell-free CLI request and its resolved configuration evidence."""
    script: str
    arguments: tuple[str, ...]
    display: str
    resolved_yaml: str = ''
    config_sha256: str = ''

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelDefaults:
    """Configuration and selectors loaded from one model_config export case."""
    export_directory: str
    case_id: str
    config_path: str
    config: dict[str, Any]
    module_defaults: dict[str, Any]
    feature_defaults: tuple[str, ...]
    inference_capability: dict[str, Any]


@dataclass(frozen=True)
class InferenceFile:
    file_id: str
    role: str
    path: str
    label: str | int | None = None


@dataclass(frozen=True)
class InferenceInput:
    participant_id: str
    files: tuple[InferenceFile, ...]
    labelled_participant_count: int


WORKFLOW_STAGES: tuple[dict[str, str], ...] = ({
    'stage': '1 · Input contract',
    'families': 'manifest, roles, frozen participant folds',
    'preview': 'record identity, class/role, raw traces'
}, {
    'stage': '2 · Signal integrity',
    'families': 'gap_repair, ppg_filter, imu_gravity, dl_resampling',
    'preview': 'repaired/filter/IMU traces and spectra'
}, {
    'stage': '3 · Windows',
    'families': 'window_profile, normalization',
    'preview': 'window counts, masks, retained coverage'
}, {
    'stage': '4 · Quality/artifact',
    'families': 'quality_mode, artifact, detector/denoiser switches',
    'preview': 'routing state and module-local evidence'
}, {
    'stage': '5 · Physiology',
    'families': 'peak_detector, prv_backend, feature_group',
    'preview': 'peaks, PPI/PRV, morphology, dual optical'
}, {
    'stage': '6 · Representation',
    'families': 'representation, model',
    'preview': 'raw/vector/matrix/fusion model-input schema'
}, {
    'stage': '7 · Fold-local fit',
    'families': 'optimizer, loss, sampler, class weighting, epoch selection',
    'preview': 'training history and fitted provenance'
}, {
    'stage': '8 · Prediction',
    'families': 'window → file → role → participant',
    'preview': 'OOF probabilities at every retained level'
}, {
    'stage': '9 · Aggregation',
    'families': 'aggregation, quality_weight_source',
    'preview': 'Line A/Line B file-role-participant views'
}, {
    'stage': '10 · Data products',
    'families': 'Parquet/CSV/JSON/XLSX contracts',
    'preview': 'pipeline_output; plots remain in analyse_report'
})


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f'{label} must be a mapping')
    return dict(value)


def _yaml_mapping(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding='utf-8')), label=str(path))


def _json_mapping(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding='utf-8')), label=str(path))


def _role_family(value: Any) -> str:
    text = _concrete_role(value)
    if text == 'B':
        return 'B'
    return text[0]


def _concrete_role(value: Any) -> str:
    """Validate one concrete V2 role without collapsing R1/S2/W1 identity."""
    text = str(value).strip().upper()
    if text == 'B' or (text[:1] in _DYNAMIC_ROLE_FAMILIES and (len(text) == 1 or text[1:].isdigit())):
        return text
    raise ValueError(f'unsupported role {value!r}; expected B/R*/S*/W*')


def _inference_label(value: Any) -> str | int | None:
    """Normalize the editable label cell without guessing class semantics."""
    if value is None or (isinstance(value, str) and (not value.strip())):
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError('inference label must be blank, an integer, or a class name')
    return value.strip() if isinstance(value, str) else value


def _assert_inference_source_contract(payload: Mapping[str, Any]) -> None:
    supplied = _mapping(payload.get('source_contract'), label='source_contract')
    required = set(INFERENCE_SOURCE_CONTRACT)
    missing = sorted(required - set(supplied))
    if missing:
        raise ValueError(f'source_contract is missing required fields: {missing}')
    sampling_rate = supplied.get('sampling_rate_hz')
    sampling_rate_valid = not isinstance(sampling_rate, bool) and isinstance(sampling_rate,
                                                                             (int, float)) and (float(sampling_rate) == 400.0)
    fixed_fields_match = all((supplied.get(field) == INFERENCE_SOURCE_CONTRACT[field]
                              for field in ('provenance', 'channel_order', 'accelerometer_unit', 'gyroscope_unit', 'synchrony')))
    if not sampling_rate_valid or not fixed_fields_match:
        raise ValueError('inference source_contract must exactly declare the Dashboard 400 Hz eight-channel raw CSV contract')


def _workflow_artifact_na(reason: str) -> tuple[dict[str, Any], ...]:
    """Make absence explicit for stages which cannot be computed in preview mode."""
    return tuple(({
        'stage': stage,
        'metric': 'completed_artifact',
        'value': str(reason),
        'status': 'N/A'
    } for stage in ('representation_model', 'aggregation')))


def _yaml_scalar(value: Any) -> str:
    encoded = yaml.safe_dump(value, default_flow_style=True, sort_keys=False).strip()
    return encoded.removesuffix('...').rstrip()


def flatten_parameters(config: Mapping[str, Any],
                       *,
                       parameter_contract: Mapping[str, Mapping[str, Any]] | None = None) -> list[dict[str, str]]:
    """Expose every leaf as editable YAML with optional live CLI metadata.

    ``parameter_contract`` is presentation evidence from
    :func:`ppg_frailty.v5.configuration.parameter_rows`.  It never validates or
    rewrites a value: the editable YAML table and the canonical resolver remain
    authoritative.
    """
    rows: list[dict[str, str]] = []
    contracts = dict(parameter_contract or {})

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(item, f'{path}.{key}' if path else str(key))
            return
        encoded = _yaml_scalar(value)
        contract = contracts.get(path, {})
        rows.append({
            'path': path,
            'value_yaml': encoded,
            'original_yaml': encoded,
            'type': type(value).__name__,
            'control': str(contract.get('control', '')),
            'input': str(contract.get('input', '')),
            'range': str(contract.get('range', ''))
        })

    visit(config, '')
    return rows


def changed_assignments(rows: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    """Convert edited parameter-table rows to strict CLI ``--set`` values."""
    output: list[str] = []
    seen: set[str] = set()
    for raw in rows:
        path = str(raw.get('path', '')).strip()
        value = str(raw.get('value_yaml', '')).strip()
        original = str(raw.get('original_yaml', '')).strip()
        if not path or path in seen:
            raise ValueError('parameter rows require unique non-empty dotted paths')
        seen.add(path)
        if value != original:
            if path in {'schema_version', 'config_id'}:
                raise ValueError(f'{path} is controlled outside --set')
            yaml.safe_load(value)
            output.append(f'{path}={value}')
    return tuple(output)


def _normalized_comparison_cases(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate cached commands without pretending they form a study plan."""
    if not cases:
        raise ValueError('comparison queue is empty')
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, item in enumerate(cases):
        name = str(item.get('name', '')).strip()
        if not _SAFE_NAME.fullmatch(name) or name in names:
            raise ValueError(f'comparison case {index + 1} needs a unique safe name')
        names.add(name)
        arguments = item.get('arguments')
        if not isinstance(arguments, list) or not all((isinstance(value, str) for value in arguments)):
            raise TypeError('comparison arguments must be a string list')
        script = str(item.get('script', 'pipeline.py'))
        if script != 'pipeline.py' or not arguments or arguments[0] != 'run':
            raise ValueError("comparison queue accepts only executable 'pipeline.py run' requests")
        _argument_map(arguments)
        normalized.append({
            'order': index + 1,
            'name': name,
            'script': script,
            'arguments': list(arguments),
            'display': shlex.join(['python', script, *arguments]),
            'config_sha256': str(item.get('config_sha256', '')),
            'resolved_yaml': str(item.get('resolved_yaml', ''))
        })
    return normalized


def _argument_map(arguments: Sequence[str]) -> tuple[dict[str, list[str]], set[str]]:
    """Parse the small, already validated ``pipeline.py run`` argv surface."""
    values: dict[str, list[str]] = {}
    switches: set[str] = set()
    value_options = {
        '--config', '--module', '--set', '--unset', '--config-id', '--study-id', '--purpose', '--repeats', '--folds', '--jobs',
        '--device', '--preprocessing-cache-mode', '--preprocessing-cache-root', '--preprocessing-cache-namespaces', '--output-root',
        '--run-name', '--resume', '--environment-lock', '--environment-policy'
    }
    switch_options = {
        '--continue-on-error', '--no-continue-on-error', '--measure-operational-costs', '--no-measure-operational-costs',
        '--hash-predictions', '--dry-run', '--refit'
    }
    index = 1
    while index < len(arguments):
        option = arguments[index]
        if option in value_options:
            if index + 1 >= len(arguments):
                raise ValueError(f'{option} requires a value')
            values.setdefault(option, []).append(arguments[index + 1])
            index += 2
            continue
        if option in switch_options:
            switches.add(option)
            index += 1
            continue
        raise ValueError(f'unsupported cached pipeline option: {option}')
    return (values, switches)


def _one(values: Mapping[str, list[str]], option: str, default: str | None = None) -> str | None:
    selected = values.get(option, [])
    if len(selected) > 1:
        raise ValueError(f'{option} may occur only once in a comparison case')
    return selected[0] if selected else default


def _boolean_switch(switches: set[str], *, positive: str, negative: str, default: bool) -> bool:
    """Resolve one argparse BooleanOptionalAction without losing intent."""
    if positive in switches and negative in switches:
        raise ValueError(f'comparison case cannot combine {positive} and {negative}')
    if positive in switches:
        return True
    if negative in switches:
        return False
    return default


def _indices_for_plan(value: str | None) -> list[int]:
    if value in (None, 'all'):
        return list(range(5))
    result = [int(item.strip()) for item in str(value).split(',') if item.strip()]
    if not result or len(result) != len(set(result)) or (not set(result) <= set(range(5))):
        raise ValueError("repeats/folds must be 'all' or unique indices 0..4")
    return result


def _index_expression(value: str | Sequence[int | str], *, label: str) -> str:
    """Canonicalize the Dash multi-select to the public CLI ``_indices`` syntax."""
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, Sequence) and (not isinstance(value, (bytes, bytearray))):
        text = ','.join((str(item) for item in value))
    else:
        raise ValueError(f"{label} must be 'all' or a unique subset of 0..4")
    try:
        selected = _indices_for_plan(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be 'all' or a comma-separated unique subset of 0..4") from error
    if text.lower() == 'all' or selected == list(range(5)):
        return 'all'
    return ','.join((str(index) for index in selected))


def _normalized_unsets(values: Iterable[str]) -> tuple[str, ...]:
    """Validate repeated ``--unset`` paths without altering their order."""
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        path = str(value).strip()
        if not path or path in seen:
            raise ValueError('unset paths must be unique non-empty dotted paths')
        if any((not part for part in path.split('.'))):
            raise ValueError(f'invalid unset dotted path: {path!r}')
        output.append(path)
        seen.add(path)
    return tuple(output)


def _value_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _effective_config_for_device(config: Mapping[str, Any], requested_device: str) -> tuple[dict[str, Any], str]:
    """Mirror the study runner's execution-device materialization for previews."""
    effective = copy.deepcopy(dict(config))
    model = _mapping(effective.get('model'), label='resolved model')
    backend = str(model_factory_contract(str(model.get('model_id', '')))['execution_backend'])
    device = requested_device if backend == 'torch' else 'cpu'
    if requested_device != device:
        raise ValueError(f"device must be cpu for estimator model {model.get('model_id')!r}")
    training = _mapping(effective.get('training'), label='resolved training')
    training['device'] = device
    effective['training'] = training
    from ..config import validate_config_payload
    validated = validate_config_payload(effective)
    digest = hashlib.sha256(_value_key(validated).encode('utf-8')).hexdigest()
    return (validated, digest)


_SEQUENCE_VALUE_OPTIONS = ('--study-id', '--purpose', '--repeats', '--folds', '--jobs', '--device', '--preprocessing-cache-mode',
                           '--preprocessing-cache-root', '--preprocessing-cache-namespaces', '--output-root', '--run-name',
                           '--environment-lock', '--environment-policy')
_SEQUENCE_SWITCH_OPTIONS = ('--continue-on-error', '--no-continue-on-error', '--measure-operational-costs',
                            '--no-measure-operational-costs', '--hash-predictions', '--dry-run', '--refit')


def _sequence_global_contract(normalized: Sequence[Mapping[str, Any]], *, pipeline_root: Path) -> dict[str, Any]:
    """Resolve controls shared by every queued training request."""
    parsed = [_argument_map(row['arguments']) for row in normalized]
    signatures = [
        tuple(((option, tuple(values.get(option, ())))
               for option in _SEQUENCE_VALUE_OPTIONS)) + tuple(((option, option in switches) for option in _SEQUENCE_SWITCH_OPTIONS))
        for values, switches in parsed
    ]
    if len(set(signatures)) != 1:
        raise ValueError('comparison cases must share execution, output, environment, and refit controls')
    values, switches = parsed[0]
    if _one(values, '--output-root', PIPELINE_OUTPUT) != PIPELINE_OUTPUT:
        raise ValueError('comparison output root must be pipeline_output')
    device = _one(values, '--device')
    policy = str(_one(values, '--environment-policy', 'exact'))
    if device not in {None, 'cpu', 'cuda'} or policy not in {'exact', 'record'}:
        raise ValueError('comparison device/policy is invalid')
    run_name = _one(values, '--run-name')
    study_id = _one(values, '--study-id', 'dashboard_comparison')
    if run_name is not None and (not _SAFE_NAME.fullmatch(run_name)):
        raise ValueError('comparison run name is not filesystem-safe')
    if study_id is None or not _SAFE_NAME.fullmatch(study_id):
        raise ValueError('comparison study ID is not filesystem-safe')
    purpose = _one(values, '--purpose', 'Dashboard-authored ordered comparison sequence.')
    if purpose is None or not purpose.strip():
        raise ValueError('comparison purpose must be non-empty')
    namespaces = tuple((item.strip() for item in str(
        _one(values, '--preprocessing-cache-namespaces', 'imu_calibration,canonical_signal_views,motion_windows,raw_windows')).split(
            ',') if item.strip()))
    from ..study import ExecutionSpec, PreprocessingCacheSpec
    execution = ExecutionSpec(
        repeats=tuple(_indices_for_plan(_one(values, '--repeats', 'all'))),
        folds=tuple(_indices_for_plan(_one(values, '--folds', 'all'))),
        jobs=int(str(_one(values, '--jobs', '1'))), device=device,
        continue_on_error=_boolean_switch(
            switches, positive='--continue-on-error',
            negative='--no-continue-on-error', default=True,
        ),
        allow_parallel_deep=False,
        measure_operational_costs=_boolean_switch(
            switches, positive='--measure-operational-costs',
            negative='--no-measure-operational-costs', default=False,
        ),
        preprocessing_cache=PreprocessingCacheSpec(
            mode=str(_one(values, '--preprocessing-cache-mode', 'off')),
            root=str(_one(values, '--preprocessing-cache-root', 'artifacts/studies/cache')),
            namespaces=namespaces, verify_source_sha256=True,
        ),
    )
    lock_value = str(_one(values, '--environment-lock', 'requirements/environment-finalcase-lock.yaml'))
    lock = Path(lock_value)
    lock = lock.resolve() if lock.is_absolute() else (pipeline_root / lock).resolve()
    try:
        lock = lock.relative_to(pipeline_root)
    except ValueError as error:
        raise ValueError('comparison environment lock must remain inside V5') from error
    return {
        'study': {
            'study_id': study_id,
            'kind': 'comparison_sequence',
            'purpose': purpose,
            'flow_position': 'User-controlled V5 comparison queue.',
            'decision_role': 'single_run' if len(normalized) == 1 else 'screening',
            'reference_case_id': str(normalized[0]['name']),
            'thesis_sections': []
        },
        'execution': asdict(execution),
        'launch': {
            'run_name': run_name,
            'hash_predictions': '--hash-predictions' in switches,
            'dry_run': '--dry-run' in switches,
            'environment_policy': policy,
            'environment_lock': lock.as_posix(),
            'refit': '--refit' in switches
        },
        'output': {
            'root': PIPELINE_OUTPUT
        },
        'report': {
            'top_k': max(1, len(normalized)),
            'detailed_configuration_top_k': 0,
            'write_html': False,
            'write_static_figures': False,
            'calibration_bins': 10,
            'figure_modules': ['all'],
            'compact_mean_sd': True,
            'write_excel_workbook': False,
            'classification_tsne_random_state': 42,
            'classification_tsne_perplexity': 30.0,
            'classification_tsne_max_samples': 5000,
            'classification_roc_macro_grid_points': 201,
            'classification_score_histogram_bins': 40
        }
    }


def comparison_sequence_export_yaml(cases: Sequence[Mapping[str, Any]], *, pipeline_root: str | Path | None = None) -> str:
    """Export the ordered comparison queue as its executable sequential schema."""
    normalized = _normalized_comparison_cases(cases)
    root = Path(pipeline_root or Path(__file__).resolve().parents[3]).resolve()
    payload = {
        'schema_version':
        COMPARISON_SEQUENCE_SCHEMA,
        **_sequence_global_contract(normalized, pipeline_root=root), 'study_plan_v2':
        False,
        'study_plan_v2_unavailable_reason':
        'ordered queue uses the comparison-sequence executor',
        'cases': [{
            'order': index,
            'case_id': str(row['name']),
            'config_sha256': str(row.get('config_sha256', '')),
            'pipeline_request': {
                'script': str(row['script']),
                'arguments': list(row['arguments']),
                'command': str(row['display'])
            },
            'resolved_config': yaml.safe_load(str(row.get('resolved_yaml', '')))
        } for index, row in enumerate(normalized, start=1)]
    }
    require_evidence = all(row.get('config_sha256') and row.get('resolved_yaml') for row in normalized)
    validate_comparison_sequence_payload(payload, pipeline_root=root, require_resolved_evidence=require_evidence)
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def comparison_sequence_yaml(cases: Sequence[Mapping[str, Any]], *, pipeline_root: str | Path | None = None) -> str:
    """Compatibility name for the executable ordered YAML export."""
    return comparison_sequence_export_yaml(cases, pipeline_root=pipeline_root)


def comparison_sequence_cli(cases: Sequence[Mapping[str, Any]]) -> str:
    """Return one auditable command per queued case."""
    normalized = _normalized_comparison_cases(cases)
    return '\n'.join((row['display'] for row in normalized)) + '\n'


def validate_comparison_sequence_payload(payload: Mapping[str, Any],
                                         *,
                                         pipeline_root: str | Path,
                                         require_resolved_evidence: bool = True) -> tuple[dict[str, Any], ...]:
    """Validate queue structure, exact commands, and optionally resolved-config evidence."""
    data = _mapping(payload, label='comparison sequence')
    required = {
        'schema_version', 'study', 'execution', 'launch', 'output', 'report', 'study_plan_v2', 'study_plan_v2_unavailable_reason',
        'cases'
    }
    if set(data) != required or data.get('schema_version') != COMPARISON_SEQUENCE_SCHEMA:
        raise ValueError('comparison sequence differs from the executable schema')
    if data['study_plan_v2'] is not False or not str(data['study_plan_v2_unavailable_reason']).strip():
        raise ValueError('comparison sequence must identify its sequential schema')
    raw_cases = data['cases']
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError('comparison sequence cases must be a non-empty list')
    root = Path(pipeline_root).resolve()
    reconstructed: list[dict[str, Any]] = []
    expected_case_keys = {'order', 'case_id', 'config_sha256', 'pipeline_request', 'resolved_config'}
    for index, raw in enumerate(raw_cases, start=1):
        case = _mapping(raw, label=f'comparison sequence case {index}')
        request = _mapping(case.get('pipeline_request'), label=f'comparison sequence case {index} request')
        if set(case) != expected_case_keys or set(request) != {'script', 'arguments', 'command'}:
            raise ValueError(f'comparison sequence case {index} differs from the schema')
        if isinstance(case['order'], bool) or case['order'] != index:
            raise ValueError('comparison sequence order must be contiguous and one-based')
        reconstructed.append({
            'name': case['case_id'],
            'script': request['script'],
            'arguments': request['arguments'],
            'display': request['command'],
            'config_sha256': case['config_sha256'],
            'resolved_yaml': yaml.safe_dump(case['resolved_config'], sort_keys=False, allow_unicode=True)
        })
    normalized = _normalized_comparison_cases(reconstructed)
    expected = _sequence_global_contract(normalized, pipeline_root=root)
    for key in ('study', 'execution', 'launch', 'output', 'report'):
        if _value_key(data[key]) != _value_key(expected[key]):
            raise ValueError(f'comparison sequence {key} differs from its exact CLI')
    for index, (case, row) in enumerate(zip(raw_cases, normalized), start=1):
        request = _mapping(case['pipeline_request'], label=f'comparison sequence case {index} request')
        if request['command'] != row['display']:
            raise ValueError(f'comparison sequence case {index} command differs from its argv')
        if not require_resolved_evidence:
            continue
        digest = str(case['config_sha256'])
        stored = _mapping(case['resolved_config'], label=f'comparison sequence case {index} resolved config')
        if not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError(f'comparison sequence case {index} requires a config SHA-256')
        if hashlib.sha256(_value_key(stored).encode('utf-8')).hexdigest() != digest:
            raise ValueError(f'comparison sequence case {index} resolved config hash mismatch')
        values, _ = _argument_map(row['arguments'])
        config_value = _one(values, '--config')
        if config_value is None:
            raise ValueError(f'comparison sequence case {index} requires exactly one --config')
        config_path = Path(config_value)
        config_path = config_path.resolve() if config_path.is_absolute() else (root / config_path).resolve()
        try:
            config_path.relative_to(root)
        except ValueError as error:
            raise ValueError('comparison config must remain inside V5') from error
        resolved, _ = resolve_configuration(pipeline_root=root,
                                            config_path=config_path,
                                            assignments=tuple(values.get('--set', ())),
                                            unsets=tuple(values.get('--unset', ())),
                                            modules=tuple(values.get('--module', ())),
                                            config_id=_one(values, '--config-id'))
        device = _one(values, '--device') or str(_mapping(resolved['training'], label='training').get('device', 'cpu'))
        effective, effective_digest = _effective_config_for_device(resolved, device)
        if effective_digest != digest or _value_key(effective) != _value_key(stored):
            raise ValueError(f'comparison sequence case {index} CLI no longer resolves to its stored configuration')
    return tuple(normalized)


class V5ControlService:
    """Safe, testable adapter shared by the Dash callbacks."""
    def __init__(self, pipeline_root: str | Path | None = None) -> None:
        inferred = Path(__file__).resolve().parents[3]
        self.pipeline_root = Path(pipeline_root or inferred).resolve()
        self.repository_root = self.pipeline_root.parents[1]

    def relative(self, path: str | Path) -> str:
        return Path(path).resolve().relative_to(self.pipeline_root).as_posix()

    def cli_input_path(self, path: str | Path) -> str:
        """Prefer portable V5-relative paths; retain absolute repo evidence paths."""
        target = Path(path).resolve()
        try:
            return target.relative_to(self.pipeline_root).as_posix()
        except ValueError:
            target.relative_to(self.repository_root)
            return str(target)

    def safe_input(self, value: str | Path, *, label: str, must_exist: bool = True, suffixes: Iterable[str] | None = None) -> Path:
        raw = Path(value)
        target = raw.resolve() if raw.is_absolute() else (self.pipeline_root / raw).resolve()
        try:
            target.relative_to(self.repository_root)
        except ValueError as error:
            raise ValueError(f'{label} must remain inside {self.repository_root}') from error
        if must_exist and (not target.exists()):
            raise FileNotFoundError(target)
        allowed = None if suffixes is None else {str(item).lower() for item in suffixes}
        if allowed is not None and target.suffix.lower() not in allowed:
            raise ValueError(f'{label} must use one of {sorted(allowed)}')
        return target

    def safe_pipeline_input(self, value: str | Path, *, label: str, must_exist: bool = True) -> Path:
        target = self.safe_input(value, label=label, must_exist=must_exist)
        try:
            target.relative_to(self.pipeline_root)
        except ValueError as error:
            raise ValueError(f'{label} must remain inside {self.pipeline_root}') from error
        return target

    def output_directory(self, root_name: str, leaf: str) -> Path:
        if root_name not in {PIPELINE_OUTPUT, REPORT_OUTPUT, MODEL_CONFIG}:
            raise ValueError(f'unsupported V5 output root: {root_name}')
        name = str(leaf).strip()
        if not _SAFE_NAME.fullmatch(name):
            raise ValueError("output name must contain only letters, digits, '.', '_' or '-'")
        target = (self.pipeline_root / root_name / name).resolve()
        target.relative_to((self.pipeline_root / root_name).resolve())
        return target

    def _named_output_input(self, value: str | Path, *, root_name: str, label: str, must_exist: bool = True) -> Path:
        """Resolve one browser-supplied path below a fixed V5 output root."""
        if root_name not in {PIPELINE_OUTPUT, REPORT_OUTPUT, MODEL_CONFIG}:
            raise ValueError(f'unsupported V5 output root: {root_name}')
        target = self.safe_pipeline_input(value, label=label, must_exist=must_exist)
        try:
            target.relative_to((self.pipeline_root / root_name).resolve())
        except ValueError as error:
            raise ValueError(f'{label} must remain inside {root_name}') from error
        return target

    @staticmethod
    def _command_request(script: str, arguments: Sequence[str]) -> CommandRequest:
        values = tuple((str(value) for value in arguments))
        evidence = {'schema_version': 'ppg_frailty.dashboard_command.v1', 'script': str(script), 'arguments': list(values)}
        return CommandRequest(script=script,
                              arguments=values,
                              display=shlex.join(['python', script, *values]),
                              resolved_yaml=yaml.safe_dump(evidence, sort_keys=False, allow_unicode=True))

    def yaml_paths(self) -> tuple[str, ...]:
        candidates = list((self.pipeline_root / 'configs').rglob('*.yaml'))
        model_root = self.pipeline_root / MODEL_CONFIG
        if model_root.is_dir():
            candidates.extend(model_root.rglob('resolved_pipeline_config.yaml'))
        accepted: list[str] = []
        for path in candidates:
            if not path.is_file() or path.is_symlink():
                continue
            try:
                payload = yaml.safe_load(path.read_text(encoding='utf-8'))
            except (OSError, yaml.YAMLError):
                continue
            if isinstance(payload, Mapping) and payload.get('schema_version') == 'ppg_frailty.pipeline_config.v2':
                accepted.append(self.relative(path))
        return tuple(sorted(set(accepted)))

    def study_plan_paths(self) -> tuple[str, ...]:
        """List only YAML files accepted by the real study-plan parser."""
        from ..study import load_study_plan
        candidates = sorted((self.pipeline_root / 'configs').rglob('*.yaml'))
        accepted: list[str] = []
        for path in candidates:
            if not path.is_file() or path.is_symlink():
                continue
            try:
                plan = load_study_plan(path)
            except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError):
                continue
            if plan.schema_version == 'ppg_frailty.study_plan.v2':
                accepted.append(self.relative(path))
        return tuple(accepted)

    def parameter_contract(self) -> dict[str, dict[str, str]]:
        """Return stable metadata from the same live catalog as ``--help``.

        Preset-specific coupled/derived descriptions contain the current value,
        so they are deliberately left to the canonical parameter table. Only
        invariant catalog rows are attached as UI guidance; disagreement across
        presets fails closed rather than choosing an arbitrary range.
        """
        from ..v5.configuration import PRESETS, parameter_rows
        candidates: dict[str, list[dict[str, str]]] = {}
        for preset in PRESETS:
            for raw in parameter_rows(self.pipeline_root, source_preset=preset):
                if str(raw.get('control', '')) in {'coupled', 'derived'}:
                    continue
                row = {'control': str(raw.get('control', '')), 'input': str(raw.get('input', '')), 'range': str(raw.get('range', ''))}
                candidates.setdefault(str(raw['path']), []).append(row)
        output: dict[str, dict[str, str]] = {}
        for path, rows in candidates.items():
            first = rows[0]
            if any((row != first for row in rows[1:])):
                raise RuntimeError(f'live parameter contract differs across presets for {path}')
            output[path] = first
        return output

    def build_comparison_execution_request(self, cases: Sequence[Mapping[str, Any]]) -> tuple[CommandRequest, Path]:
        """Atomically materialize a queue for the comparison-sequence runner."""
        encoded = comparison_sequence_export_yaml(cases, pipeline_root=self.pipeline_root)
        payload = _mapping(yaml.safe_load(encoded), label='materialized comparison request')
        validate_comparison_sequence_payload(payload, pipeline_root=self.pipeline_root, require_resolved_evidence=True)
        digest = hashlib.sha256(encoded.encode('utf-8')).hexdigest()
        request_root = (self.pipeline_root / PIPELINE_OUTPUT / '.dashboard_requests' / 'comparison').resolve()
        request_root.relative_to(self.pipeline_root)
        request_root.mkdir(parents=True, exist_ok=True)
        target = request_root / f'sequence_{digest}.yaml'
        if target.is_symlink():
            raise ValueError('comparison request target must not be a symlink')
        if target.exists():
            if not target.is_file():
                raise ValueError('comparison request target must be a regular file')
            if target.read_text(encoding='utf-8') != encoded:
                raise RuntimeError(f'comparison request hash collision: {target}')
        else:
            temporary = target.with_name(f'.{target.name}.tmp-{uuid.uuid4().hex}')
            try:
                temporary.write_text(encoded, encoding='utf-8')
                temporary.replace(target)
            finally:
                if temporary.exists():
                    temporary.unlink()
        persisted = _mapping(yaml.safe_load(target.read_text(encoding='utf-8')), label='persisted comparison sequence')
        validate_comparison_sequence_payload(persisted, pipeline_root=self.pipeline_root, require_resolved_evidence=True)
        arguments = ('run', '--request', self.relative(target))
        return (CommandRequest(script='comparison_sequence.py',
                               arguments=arguments,
                               display=shlex.join(['python', 'comparison_sequence.py', *arguments]),
                               resolved_yaml=encoded,
                               config_sha256=digest), target)

    @staticmethod
    def _command_options(parser: Any, command: str) -> frozenset[str]:
        """Return one argparse subcommand's flags for forward-compatible gates."""
        for action in getattr(parser, '_actions', ()):
            choices = getattr(action, 'choices', None)
            if isinstance(choices, Mapping) and command in choices:
                selected = choices[command]
                return frozenset(
                    (option for item in getattr(selected, '_actions', ()) for option in getattr(item, 'option_strings', ())))
        return frozenset()

    def sweep_capabilities(self) -> dict[str, Any]:
        """Probe the installed sweep CLI instead of assuming future refit flags."""
        script = self.pipeline_root / 'sweep.py'
        if not script.is_file():
            return {'available': False, 'run_options': (), 'reason': 'sweep.py missing'}
        try:
            from ..v5.sweep import build_parser
            options = self._command_options(build_parser(), 'run')
        except Exception as error:
            return {'available': False, 'run_options': (), 'reason': f'{type(error).__name__}: {error}'}
        required = {'--plan'}
        return {
            'available': required <= options,
            'run_options': tuple(sorted(options)),
            'refit': '--refit' in options,
            'reason': '' if required <= options else 'sweep run --plan unavailable'
        }

    def load_study_plan(self, plan_path: str | Path) -> tuple[Any, str]:
        """Parse and fully expand a sweep plan without fitting or writing."""
        from ..study import StudyRunner, load_study_plan, validate_canonical_expansion
        target = self.safe_pipeline_input(plan_path, label='study plan')
        if target.suffix.lower() not in {'.yaml', '.yml'} or not target.is_file():
            raise ValueError('study plan must be an existing .yaml/.yml file')
        plan = load_study_plan(target)
        validate_canonical_expansion(StudyRunner(pipeline_root=self.pipeline_root, output_layout='v5').expand(plan))
        return (plan, yaml.safe_dump(plan.to_dict(), sort_keys=False, allow_unicode=True))

    def model_exports(self) -> tuple[str, ...]:
        root = self.pipeline_root / MODEL_CONFIG
        if not root.is_dir():
            return ()
        return tuple(
            sorted((self.relative(path.parent) for path in root.glob('*/export_manifest.json')
                    if path.is_file() and (not path.is_symlink()))))

    def study_outputs(self) -> tuple[str, ...]:
        root = self.pipeline_root / PIPELINE_OUTPUT
        if not root.is_dir():
            return ()
        return tuple(
            sorted(
                {
                    self.relative(path.parent)
                    for name in ('study_manifest.json', 'v5_data_manifest.json') for path in root.rglob(name) if path.is_file()
                },
                reverse=True))

    def completed_workflow_stage_rows(self, study_directory: str | Path | None, *, record_id: str,
                                      participant_id: str) -> tuple[dict[str, Any], ...]:
        """Read model/aggregation preview rows only from completed V2 artifacts.

        No prediction is recomputed here. Exact persisted OOF tables are read
        through their canonical validator and filtered to the selected record.
        """
        if not study_directory:
            return _workflow_artifact_na('no completed pipeline artifact selected')
        study = self.safe_pipeline_input(study_directory, label='completed pipeline artifact')
        output_root = (self.pipeline_root / PIPELINE_OUTPUT).resolve()
        try:
            study.relative_to(output_root)
        except ValueError as error:
            raise ValueError('completed workflow artifact must remain inside pipeline_output') from error
        if not study.is_dir():
            raise NotADirectoryError(study)
        manifest_path = study / 'study_manifest.json'
        manifest = _json_mapping(manifest_path)
        status = str(manifest.get('status', '')).lower()
        if status not in _COMPLETE_STATUSES:
            return _workflow_artifact_na(f"selected study is not complete (status={status or 'missing'})")
        raw_cases = manifest.get('cases')
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError('completed study manifest has no cases')
        from ..training import read_oof_parquet
        output: list[dict[str, Any]] = []

        def add(stage: str, metric: str, value: Any) -> None:
            if isinstance(value, (Mapping, list, tuple)):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False, default=str)
            output.append({'stage': stage, 'metric': metric, 'value': value, 'status': 'artifact'})

        for raw_case in raw_cases:
            case = _mapping(raw_case, label='completed study case')
            case_id = str(case.get('case_id', '')).strip()
            case_directory = str(case.get('case_directory', '')).strip()
            if not case_id or not case_directory:
                raise ValueError('completed study case lacks identity or directory')
            case_root = (study / case_directory).resolve()
            case_root.relative_to(study)
            result_path = case_root / 'case_result.json'
            if not result_path.is_file():
                continue
            result = _json_mapping(result_path)
            if str(result.get('status', '')).lower() not in _COMPLETE_STATUSES:
                continue
            artifact_value = str(result.get('artifact_root', '')).strip()
            if not artifact_value:
                continue
            artifact_root = (case_root / artifact_value).resolve()
            artifact_root.relative_to(case_root)
            if not artifact_root.is_dir():
                continue
            for cell_path in sorted(artifact_root.rglob('run_manifest.json')):
                cell_manifest = _json_mapping(cell_path)
                cell_raw = cell_manifest.get('cell')
                if not isinstance(cell_raw, Mapping):
                    continue
                cell = dict(cell_raw)
                cell_status = str(cell.get('status', cell_manifest.get('status', ''))).lower()
                if cell_status not in _COMPLETE_STATUSES:
                    continue
                repeat = int(cell['repeat_index'])
                fold = int(cell['fold_index'])
                file_path = cell_path.parent / 'oof_file_predictions.parquet'
                if not file_path.is_file():
                    continue
                all_file_rows = read_oof_parquet(file_path)
                record_rows = [row for row in all_file_rows if str(row.file_id) == str(record_id)]
                if record_rows and any((str(row.participant_id) != str(participant_id) for row in record_rows)):
                    raise ValueError('completed artifact record_id maps to a different participant')
                record_rows = [row for row in record_rows if str(row.participant_id) == str(participant_id)]
                if not record_rows:
                    continue
                identity = f'{case_id}.repeat_{repeat:02d}.fold_{fold:02d}'
                metrics = cell.get('metrics')
                metrics = metrics if isinstance(metrics, Mapping) else {}
                factory = cell.get('model_factory_provenance')
                factory = factory if isinstance(factory, Mapping) else {}
                fitted = cell.get('fitted_provenance')
                fitted = fitted if isinstance(fitted, Mapping) else {}
                checkpoint = cell.get('learned_model_checkpoint')
                checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
                model_values = {
                    'artifact': cell_path.relative_to(study).as_posix(),
                    'model_id': cell.get('model_id'),
                    'model_machine_id': cell.get('model_machine_id'),
                    'representation_mode': cell.get('representation_mode'),
                    'parameter_count': factory.get('parameter_count'),
                    'model_hash': cell.get('model_hash'),
                    'state_hash': fitted.get('state_hash'),
                    'checkpoint_status': checkpoint.get('deployment_status'),
                    'balanced_accuracy': metrics.get('balanced_accuracy'),
                    'macro_f1': metrics.get('macro_f1')
                }
                for name, value in model_values.items():
                    add('representation_model', f'{identity}.{name}', value)
                add('aggregation', f'{identity}.balance_line', cell.get('balance_line'))
                for level, filename in (('file', 'oof_file_predictions.parquet'), ('role', 'oof_role_predictions.parquet'),
                                        ('participant', 'oof_subject_predictions.parquet')):
                    prediction_path = cell_path.parent / filename
                    if not prediction_path.is_file():
                        raise FileNotFoundError(f'completed cell lacks mandatory {filename}: {cell_path.parent}')
                    level_rows = record_rows if level == 'file' else [
                        row for row in read_oof_parquet(prediction_path) if str(row.participant_id) == str(participant_id)
                    ]
                    add('aggregation', f'{identity}.{level}.artifact', prediction_path.relative_to(study).as_posix())
                    for index, row in enumerate(level_rows):
                        row_id = f'{row.role}.{index}'
                        prefix = f'{identity}.{level}.{row_id}'
                        add('aggregation', f'{prefix}.file_id', row.file_id)
                        add('aggregation', f'{prefix}.retained', row.retained)
                        add('aggregation', f'{prefix}.label', row.label)
                        add('aggregation', f'{prefix}.class_order', row.class_order)
                        add('aggregation', f'{prefix}.probabilities', row.probabilities)
                        add('aggregation', f'{prefix}.aggregation_rule', row.aggregation_rule)
                        add('aggregation', f'{prefix}.route_status', row.route_status)
        if not output:
            return _workflow_artifact_na('selected record has no completed OOF artifact in this study')
        return tuple(output)

    def report_outputs(self) -> tuple[str, ...]:
        root = self.pipeline_root / REPORT_OUTPUT
        if not root.is_dir():
            return ()
        return tuple(
            sorted(
                {
                    self.relative(path.parent)
                    for name in ('analysis_manifest.json', 'report_manifest.json') for path in root.rglob(name) if path.is_file()
                },
                reverse=True))

    def module_catalog(self, export_directory: str | Path | None = None) -> dict[str, list[dict[str, Any]]]:
        rows: list[dict[str, Any]]
        if export_directory:
            export = self.safe_pipeline_input(export_directory, label='model config export')
            candidate = export / 'available_modules.json'
            if candidate.is_file():
                payload = _json_mapping(candidate)
                raw_rows = payload.get('modules')
                if not isinstance(raw_rows, list):
                    raise ValueError('available_modules.json lacks modules')
                rows = [_mapping(row, label='module row') for row in raw_rows]
            else:
                rows = list_modules()
        else:
            rows = list_modules()
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row['family']), []).append(dict(row))
        return {family: sorted(values, key=lambda item: str(item['module_id'])) for family, values in sorted(grouped.items())}

    @staticmethod
    def module_defaults_from_config(config: Mapping[str, Any]) -> dict[str, str]:
        """Reuse the model-config exporter's conservative exact derivation."""
        from ..v5.model_config_export import _derived_module_defaults
        return {
            str(row['family']): str(row['module_id'])
            for row in _derived_module_defaults(config) if str(row['family']) != 'feature_group'
        }

    def model_cases(self, export_directory: str | Path) -> tuple[str, ...]:
        export = self.safe_pipeline_input(export_directory, label='model config export')
        manifest = _json_mapping(export / 'export_manifest.json')
        cases = manifest.get('cases')
        if not isinstance(cases, list):
            raise ValueError('model config export manifest lacks cases')
        return tuple((str(_mapping(row, label='export case').get('case_id', '')) for row in cases))

    def _inference_capability(self, export: Path, manifest: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
        capabilities = manifest.get('capabilities')
        raw_adapter_declared = bool(
            isinstance(capabilities, Mapping) and capabilities.get('new_participant_inference_available') is True
            and (case.get('new_participant_inference') is True))
        candidates: list[tuple[str, Any]] = []
        for owner_name, owner in (('case', case), ('export', manifest)):
            if not isinstance(owner, Mapping):
                continue
            for key in ('bundle_path', 'checkpoint_path', 'learned_weights'):
                if key in owner:
                    candidates.append((f'{owner_name}.{key}', owner[key]))
            deployment = owner.get('deployment')
            if isinstance(deployment, Mapping):
                for key in ('bundle_path', 'checkpoint_path', 'learned_weights'):
                    if key in deployment:
                        candidates.append((f'{owner_name}.deployment.{key}', deployment[key]))
        for source, raw in candidates:
            if not isinstance(raw, str) or not raw.strip():
                continue
            value = Path(raw)
            candidate_paths = (value.resolve(), ) if value.is_absolute() else ((export / value).resolve(),
                                                                               (self.pipeline_root / value).resolve())
            for path in candidate_paths:
                try:
                    path.relative_to(self.pipeline_root)
                except ValueError:
                    continue
                if not path.exists():
                    continue
                bundle_manifest = path / 'manifest.json' if path.is_dir() else path
                try:
                    bundle_payload = _json_mapping(bundle_manifest) if bundle_manifest.is_file() else {}
                except (json.JSONDecodeError, OSError, TypeError, ValueError):
                    bundle_payload = {}
                boundary = str(bundle_payload.get('pipeline_adapter_boundary', ''))
                if raw_adapter_declared and source.endswith('bundle_path') and path.is_dir() and (path / 'manifest.json').is_file():
                    return {
                        'available': True,
                        'source': source,
                        'path': path.as_posix(),
                        'model_bundle_boundary': boundary or 'model_ready_input',
                        'adapter_source': 'v5_frozen_raw_workflow_service'
                    }
        return {
            'available':
            False,
            'reason':
            str(capabilities.get('new_participant_inference_status'))
            if isinstance(capabilities, Mapping) else 'no_raw_input_adapter_declared'
        }

    def load_model_defaults(self, export_directory: str | Path, case_id: str | None = None) -> ModelDefaults:
        export = self.safe_pipeline_input(export_directory, label='model config export')
        manifest = _json_mapping(export / 'export_manifest.json')
        raw_cases = manifest.get('cases')
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError('model config export has no cases')
        cases = [_mapping(row, label='model config case') for row in raw_cases]
        selected = next((row for row in cases if str(row.get('case_id')) == str(case_id)), cases[0] if case_id is None else None)
        if selected is None:
            raise ValueError(f'unknown model config case: {case_id}')
        directory = str(selected.get('directory', ''))
        case_root = (export / directory).resolve()
        case_root.relative_to(export)
        config_name = str(selected.get('resolved_config', ''))
        config_path = (export / config_name).resolve() if config_name else case_root / 'resolved_pipeline_config.yaml'
        config_path.relative_to(export)
        config = _yaml_mapping(config_path)
        defaults_path = case_root / 'pipeline_module_defaults.yaml'
        defaults = _yaml_mapping(defaults_path) if defaults_path.is_file() else {}
        modules: dict[str, Any] = {}
        for key in ('derived_module_defaults', 'requested_module_selections'):
            values = defaults.get(key, ())
            if isinstance(values, list):
                for row in values:
                    if not isinstance(row, Mapping):
                        continue
                    family = str(row.get('family', ''))
                    module_id = str(row.get('module_id', ''))
                    if family and module_id and (family != 'feature_group'):
                        modules[family] = module_id
        features = config.get('features', {})
        enabled = features.get('enabled_groups', ()) if isinstance(features, Mapping) else ()
        feature_defaults = tuple((str(value) for value in enabled)) if isinstance(enabled, list) else ()
        return ModelDefaults(export_directory=self.relative(export),
                             case_id=str(selected.get('case_id', '')),
                             config_path=self.relative(config_path),
                             config=config,
                             module_defaults=modules,
                             feature_defaults=feature_defaults,
                             inference_capability=self._inference_capability(export, manifest, selected))

    def load_yaml(self, config_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
        target = self.safe_pipeline_input(config_path, label='training YAML')
        if target.suffix.lower() not in {'.yaml', '.yml'} or not target.is_file():
            raise ValueError('training YAML must be an existing .yaml/.yml file')
        return resolve_configuration(pipeline_root=self.pipeline_root, config_path=target)

    def _refit_arguments(self, *, enabled: bool, supported_options: Iterable[str]) -> list[str]:
        """Expose refit as the same optional, default-off CLI module."""
        if not enabled:
            return []
        supported = set(supported_options)
        if '--refit' not in supported:
            raise RuntimeError('the selected training CLI does not support --refit')
        return ['--refit']

    def _build_sweep_request(self, *, plan_path: str | Path | None, selected_modules: Mapping[str, Any] | None,
                             default_modules: Mapping[str, Any] | None, feature_groups: Sequence[str],
                             default_feature_groups: Sequence[str], parameter_rows: Sequence[Mapping[str, Any]], run_name: str | None,
                             hash_predictions: bool, dry_run: bool, resume: str | Path | None, environment_lock: str | Path | None,
                             environment_policy: str, refit: bool) -> CommandRequest:
        """Create exactly ``sweep.py run --plan``; the plan owns its cases."""
        if not plan_path:
            raise ValueError('Train sweep requires an explicitly selected study-plan YAML')
        capabilities = self.sweep_capabilities()
        if not capabilities.get('available'):
            raise RuntimeError(f"sweep unavailable: {capabilities.get('reason', 'unknown')}")
        options = set(capabilities.get('run_options', ()))
        selected = dict(selected_modules or {})
        defaults = dict(default_modules or {})
        module_edits = {
            family: value
            for family, value in selected.items() if value not in (None, '', INHERIT_MODULE) and value != defaults.get(family)
        }
        if module_edits or tuple(feature_groups) != tuple(default_feature_groups) or changed_assignments(parameter_rows):
            raise ValueError(
                'sweep cases are owned by the selected study plan; Configure edits cannot be silently applied. Export a validator-backed comparison plan first'
            )
        plan_file = self.safe_pipeline_input(plan_path, label='study plan')
        plan, resolved_yaml = self.load_study_plan(plan_file)
        arguments = ['run', '--plan', self.relative(plan_file)]
        if run_name:
            name = str(run_name).strip()
            if not _SAFE_NAME.fullmatch(name):
                raise ValueError('run name is not filesystem-safe')
            if '--run-name' not in options:
                raise RuntimeError('the installed sweep CLI does not support --run-name')
            arguments.extend(['--run-name', name])
        if resume:
            if '--resume' not in options:
                raise RuntimeError('the installed sweep CLI does not support --resume')
            resume_path = self.safe_pipeline_input(resume, label='resume directory')
            resume_path.relative_to((self.pipeline_root / PIPELINE_OUTPUT).resolve())
            if not resume_path.is_dir():
                raise NotADirectoryError(resume_path)
            arguments.extend(['--resume', self.relative(resume_path)])
        if hash_predictions:
            if '--hash-predictions' not in options:
                raise RuntimeError('the installed sweep CLI does not support prediction hashing')
            arguments.append('--hash-predictions')
        if dry_run:
            if '--dry-run' not in options:
                raise RuntimeError('the installed sweep CLI does not support --dry-run')
            arguments.append('--dry-run')
        if environment_lock:
            if '--environment-lock' not in options:
                raise RuntimeError('the installed sweep CLI does not support --environment-lock')
            lock = self.safe_pipeline_input(environment_lock, label='environment lock')
            if lock.suffix.lower() not in {'.yaml', '.yml'} or not lock.is_file():
                raise ValueError('environment lock must be an existing YAML file')
            arguments.extend(['--environment-lock', self.relative(lock)])
        if environment_policy not in {'exact', 'record'}:
            raise ValueError('environment policy must be exact or record')
        if environment_policy != 'exact':
            if '--environment-policy' not in options:
                raise RuntimeError('the installed sweep CLI does not support environment policy')
            arguments.extend(['--environment-policy', environment_policy])
        arguments.extend(self._refit_arguments(enabled=refit, supported_options=options))
        encoded_plan = yaml.safe_dump(plan.to_dict(), sort_keys=True, allow_unicode=True)
        plan_sha256 = hashlib.sha256(encoded_plan.encode('utf-8')).hexdigest()
        return CommandRequest(script='sweep.py',
                              arguments=tuple(arguments),
                              display=shlex.join(['python', 'sweep.py', *arguments]),
                              resolved_yaml=resolved_yaml,
                              config_sha256=plan_sha256)

    def build_train_request(self,
                            *,
                            config_path: str | Path | None,
                            plan_path: str | Path | None = None,
                            operation: str = 'run',
                            selected_modules: Mapping[str, Any] | None = None,
                            default_modules: Mapping[str, Any] | None = None,
                            feature_groups: Sequence[str] = (),
                            default_feature_groups: Sequence[str] = (),
                            parameter_rows: Sequence[Mapping[str, Any]] = (),
                            unset_paths: Sequence[str] = (),
                            config_id: str | None = None,
                            study_id: str | None = None,
                            purpose: str | None = None,
                            repeats: str | Sequence[int | str] = 'all',
                            folds: str | Sequence[int | str] = 'all',
                            jobs: int = 1,
                            device: str = 'cuda',
                            cache_mode: str = 'read_write',
                            cache_root: str | None = None,
                            cache_namespaces: str | None = None,
                            continue_on_error: bool | None = None,
                            measure_operational_costs: bool | None = None,
                            hash_predictions: bool = False,
                            dry_run: bool = False,
                            resume: str | Path | None = None,
                            run_name: str | None = None,
                            environment_lock: str | Path | None = None,
                            environment_policy: str = 'exact',
                            refit: bool = False) -> CommandRequest:
        if operation not in {'run', 'sweep'}:
            raise ValueError('training operation must be run or sweep')
        if resume and run_name:
            raise ValueError('--run-name cannot be combined with --resume')
        unsets = _normalized_unsets(unset_paths)
        if operation == 'sweep':
            if unsets:
                raise ValueError(
                    'canonical StudyPlan cannot encode arbitrary --unset controls; export or run the ordered pipeline.py CLI sequence instead'
                )
            return self._build_sweep_request(plan_path=plan_path or config_path,
                                             selected_modules=selected_modules,
                                             default_modules=default_modules,
                                             feature_groups=feature_groups,
                                             default_feature_groups=default_feature_groups,
                                             parameter_rows=parameter_rows,
                                             run_name=run_name,
                                             hash_predictions=hash_predictions,
                                             dry_run=dry_run,
                                             resume=resume,
                                             environment_lock=environment_lock,
                                             environment_policy=environment_policy,
                                             refit=refit)
        if not config_path:
            raise ValueError('Train requires an explicitly selected YAML')
        config = self.safe_pipeline_input(config_path, label='training YAML')
        if config.suffix.lower() not in {'.yaml', '.yml'} or not config.is_file():
            raise ValueError('Train requires an explicitly selected YAML')
        if int(jobs) <= 0:
            raise ValueError('jobs must be positive')
        if device not in {'cpu', 'cuda'}:
            raise ValueError('device must be cpu or cuda')
        if cache_mode not in {'off', 'read_only', 'read_write'}:
            raise ValueError('invalid preprocessing cache mode')
        if environment_policy not in {'exact', 'record'}:
            raise ValueError('environment policy must be exact or record')
        if environment_policy == 'exact' and device != 'cuda':
            raise ValueError('exact environment policy requires the locked CUDA device')
        repeat_expression = _index_expression(repeats, label='repeats')
        fold_expression = _index_expression(folds, label='folds')
        config_relative = self.relative(config)
        arguments: list[str] = [operation, '--config', config_relative]
        selected = dict(selected_modules or {})
        defaults = dict(default_modules or {})
        for family in sorted(selected):
            module_id = selected[family]
            if module_id in (None, '', INHERIT_MODULE) or module_id == defaults.get(family):
                continue
            arguments.extend(['--module', f'{family}={module_id}'])
        features_changed = tuple(feature_groups) != tuple(default_feature_groups)
        if features_changed:
            for module_id in feature_groups:
                arguments.extend(['--module', f'feature_group={module_id}'])
        assignments = list(changed_assignments(parameter_rows))
        if features_changed and any((value.startswith('features.enabled_groups=') for value in assignments)):
            raise ValueError('edit feature groups through either the multi-select or parameter table, not both')
        if features_changed and (not feature_groups):
            assignments.append('features.enabled_groups=[]')
        for assignment in assignments:
            arguments.extend(['--set', assignment])
        for path in unsets:
            arguments.extend(['--unset', path])
        if config_id:
            if not _SAFE_NAME.fullmatch(str(config_id)):
                raise ValueError('config ID is not safe')
            arguments.extend(['--config-id', str(config_id)])
        if study_id:
            if not _SAFE_NAME.fullmatch(str(study_id)):
                raise ValueError('study ID is not safe')
            arguments.extend(['--study-id', str(study_id)])
        if purpose and operation == 'run':
            arguments.extend(['--purpose', str(purpose)])
        arguments.extend([
            '--repeats', repeat_expression, '--folds', fold_expression, '--jobs',
            str(int(jobs)), '--device', device, '--preprocessing-cache-mode', cache_mode, '--output-root', PIPELINE_OUTPUT
        ])
        if cache_root:
            cache_path = self.safe_pipeline_input(cache_root, label='preprocessing cache root', must_exist=False)
            arguments.extend(['--preprocessing-cache-root', self.relative(cache_path)])
        if cache_namespaces:
            namespaces = tuple((value.strip() for value in str(cache_namespaces).split(',') if value.strip()))
            if not namespaces:
                raise ValueError('preprocessing cache namespaces cannot be empty')
            arguments.extend(['--preprocessing-cache-namespaces', ','.join(namespaces)])
        if continue_on_error is not None:
            arguments.append('--continue-on-error' if continue_on_error else '--no-continue-on-error')
        if measure_operational_costs is not None:
            arguments.append('--measure-operational-costs' if measure_operational_costs else '--no-measure-operational-costs')
        if hash_predictions:
            arguments.append('--hash-predictions')
        if dry_run:
            arguments.append('--dry-run')
        if environment_lock:
            lock = self.safe_pipeline_input(environment_lock, label='environment lock')
            if lock.suffix.lower() not in {'.yaml', '.yml'} or not lock.is_file():
                raise ValueError('environment lock must be an existing YAML file')
            arguments.extend(['--environment-lock', self.relative(lock)])
        arguments.extend(['--environment-policy', environment_policy])
        if resume:
            resume_path = self.safe_pipeline_input(resume, label='resume directory')
            resume_path.relative_to((self.pipeline_root / PIPELINE_OUTPUT).resolve())
            if not resume_path.is_dir():
                raise NotADirectoryError(resume_path)
            arguments.extend(['--resume', self.relative(resume_path)])
        if run_name:
            name = str(run_name).strip()
            if not _SAFE_NAME.fullmatch(name):
                raise ValueError('run name is not filesystem-safe')
            arguments.extend(['--run-name', name])
        from ..v5.cli import build_parser as build_pipeline_parser
        arguments.extend(self._refit_arguments(enabled=refit, supported_options=self._command_options(build_pipeline_parser(), 'run')))
        resolved, _ = resolve_configuration(
            pipeline_root=self.pipeline_root,
            config_path=config,
            assignments=tuple(assignments),
            unsets=unsets,
            modules=(f'{family}={module_id}' for family, module_id in sorted(selected.items())
                     if module_id not in (None, '', INHERIT_MODULE) and module_id != defaults.get(family)),
            config_id=config_id)
        if tuple(feature_groups) != tuple(default_feature_groups):
            resolved, _ = resolve_configuration(
                pipeline_root=self.pipeline_root,
                config_path=config,
                assignments=tuple(assignments),
                unsets=unsets,
                modules=[
                    *(f'{family}={module_id}' for family, module_id in sorted(selected.items())
                      if module_id not in (None, '', INHERIT_MODULE) and module_id != defaults.get(family)),
                    *(f'feature_group={value}' for value in feature_groups)
                ],
                config_id=config_id)
        resolved, effective_sha256 = _effective_config_for_device(resolved, device)
        resolved_yaml = yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True)
        display = shlex.join(['python', 'pipeline.py', *arguments])
        return CommandRequest(script='pipeline.py',
                              arguments=tuple(arguments),
                              display=display,
                              resolved_yaml=resolved_yaml,
                              config_sha256=effective_sha256)

    @staticmethod
    def _config_arguments_from_run_request(request: CommandRequest | Mapping[str, Any] | None) -> tuple[str, ...]:
        """Copy only public configuration selectors from a validated Run request.

        Configure already owns the detailed module/parameter controls.  Maintenance
        commands consume its server-built request instead of creating a second,
        potentially divergent configuration editor.
        """
        if request is None:
            raise ValueError('build a valid Run request in Configure before this tool')
        payload = request.to_dict() if isinstance(request, CommandRequest) else dict(request)
        if payload.get('script') != 'pipeline.py':
            raise ValueError('this tool requires a pipeline.py Run request, not a sweep')
        raw = payload.get('arguments')
        if not isinstance(raw, (list, tuple)) or not raw or raw[0] != 'run':
            raise ValueError('configuration source is not a pipeline.py run request')
        arguments = tuple((str(value) for value in raw))
        valued = {'--preset', '--config', '--module', '--set', '--unset', '--config-id'}
        copied: list[str] = []
        source_count = 0
        index = 1
        while index < len(arguments):
            option = arguments[index]
            if option == '--manual':
                copied.append(option)
                source_count += 1
                index += 1
                continue
            if option in valued:
                if index + 1 >= len(arguments):
                    raise ValueError(f'configuration option lacks a value: {option}')
                copied.extend((option, arguments[index + 1]))
                if option in {'--preset', '--config'}:
                    source_count += 1
                index += 2
                continue
            index += 1
        if source_count != 1:
            raise ValueError('configuration source must contain exactly one source selector')
        return tuple(copied)

    def build_pipeline_config_tool_request(self,
                                           *,
                                           operation: str,
                                           run_request: CommandRequest | Mapping[str, Any] | None,
                                           validation_mode: str = 'smoke',
                                           environment_policy: str = 'exact',
                                           environment_lock: str | Path | None = None) -> CommandRequest:
        """Build ``pipeline.py validate`` or ``show-config`` from Configure state."""
        if operation not in {'validate', 'show-config'}:
            raise ValueError('pipeline config tool must be validate or show-config')
        config_arguments = list(self._config_arguments_from_run_request(run_request))
        if '--config' in config_arguments:
            position = config_arguments.index('--config') + 1
            config = self.safe_pipeline_input(config_arguments[position], label='configuration YAML')
            if config.suffix.lower() not in {'.yaml', '.yml'} or not config.is_file():
                raise ValueError('configuration must be an existing YAML file')
            config_arguments[position] = self.relative(config)
        arguments = [operation, *config_arguments]
        if operation == 'validate':
            if validation_mode not in {'config', 'smoke', 'full'}:
                raise ValueError('validation mode must be config, smoke, or full')
            if environment_policy not in {'exact', 'record'}:
                raise ValueError('environment policy must be exact or record')
            arguments.extend(['--mode', validation_mode, '--environment-policy', environment_policy])
            if environment_lock:
                lock = self.safe_pipeline_input(environment_lock, label='environment lock')
                if lock.suffix.lower() not in {'.yaml', '.yml'} or not lock.is_file():
                    raise ValueError('environment lock must be an existing YAML file')
                arguments.extend(['--environment-lock', self.relative(lock)])
        return self._command_request('pipeline.py', arguments)

    def build_pipeline_index_request(self, *, study_directory: str | Path, hash_predictions: bool = False) -> CommandRequest:
        study = self._named_output_input(study_directory, root_name=PIPELINE_OUTPUT, label='pipeline study')
        if not study.is_dir():
            raise NotADirectoryError(study)
        arguments = ['index', '--study-dir', self.relative(study)]
        if hash_predictions:
            arguments.append('--hash-predictions')
        return self._command_request('pipeline.py', arguments)

    def build_model_export_request(self, *, pipeline_output: str | Path) -> CommandRequest:
        source = self._named_output_input(pipeline_output, root_name=PIPELINE_OUTPUT, label='model export pipeline output')
        if not source.is_dir():
            raise NotADirectoryError(source)
        return self._command_request('export_model_config.py', ('--pipeline-output', self.relative(source)))

    def execute_model_export(self, *, pipeline_output: str | Path) -> Mapping[str, Any]:
        """Run the existing short, atomic exporter without a second algorithm."""
        request = self.build_model_export_request(pipeline_output=pipeline_output)
        from ..v5.model_config_export import export_model_config
        result = export_model_config(request.arguments[-1], pipeline_root=self.pipeline_root)
        if not isinstance(result, Mapping):
            raise TypeError('model config exporter must return a mapping')
        return dict(result)

    def build_pipeline_excel_request(self, *, pipeline_output: str | Path, replace: bool = False) -> CommandRequest:
        source = self._named_output_input(pipeline_output, root_name=PIPELINE_OUTPUT, label='pipeline Excel source')
        if not source.is_dir():
            raise NotADirectoryError(source)
        arguments = ['export-excel', '--pipeline-output', self.relative(source)]
        if replace:
            arguments.append('--replace')
        return self._command_request('sweep.py', arguments)

    def build_report_excel_request(self, *, report_output: str | Path, replace: bool = False) -> CommandRequest:
        source = self._named_output_input(report_output, root_name=REPORT_OUTPUT, label='report Excel source')
        if not source.is_dir():
            raise NotADirectoryError(source)
        arguments = ['export-excel', '--report-output', self.relative(source)]
        if replace:
            arguments.append('--replace')
        return self._command_request('analyse_report.py', arguments)

    def build_sweep_validate_request(self,
                                     *,
                                     plan_path: str | Path,
                                     environment_policy: str = 'exact',
                                     environment_lock: str | Path | None = None) -> CommandRequest:
        plan = self.safe_pipeline_input(plan_path, label='study plan')
        if plan.suffix.lower() not in {'.yaml', '.yml'} or not plan.is_file():
            raise ValueError('study plan must be an existing YAML file')
        self.load_study_plan(plan)
        if environment_policy not in {'exact', 'record'}:
            raise ValueError('environment policy must be exact or record')
        arguments = ['validate', '--plan', self.relative(plan), '--environment-policy', environment_policy]
        if environment_lock:
            lock = self.safe_pipeline_input(environment_lock, label='environment lock')
            if lock.suffix.lower() not in {'.yaml', '.yml'} or not lock.is_file():
                raise ValueError('environment lock must be an existing YAML file')
            arguments.extend(['--environment-lock', self.relative(lock)])
        return self._command_request('sweep.py', arguments)

    def build_specialized_pipeline_request(self,
                                           *,
                                           operation: str,
                                           plan_path: str | Path | None = None,
                                           study_directory: str | Path | None = None,
                                           run_name: str | None = None,
                                           resume: str | Path | None = None,
                                           source_root: str | Path | None = None,
                                           upstream_study: str | Path | None = None,
                                           device: str | None = None,
                                           jobs: int | None = None,
                                           include_denoiser: bool = True,
                                           dry_run: bool = False,
                                           environment_policy: str = 'exact',
                                           environment_lock: str | Path | None = None) -> CommandRequest:
        """Build a stop-safe request for the preserved computation CLI."""
        if operation not in {'validate', 'run', 'complete'}:
            raise ValueError('specialized pipeline operation is unsupported')
        if operation == 'run' and resume and run_name:
            raise ValueError('specialized --run-name cannot be combined with --resume')
        if device is not None and (not re.fullmatch('(?:cpu|cuda(?::\\d+)?)', str(device))):
            raise ValueError('specialized device must be cpu, cuda, or cuda:<index>')
        if jobs is not None:
            worker_count = int(jobs)
            if isinstance(jobs, bool) or float(jobs) != worker_count or worker_count <= 0:
                raise ValueError('specialized jobs must be a positive integer')
        else:
            worker_count = None
        if operation in {'validate', 'run'}:
            if not plan_path:
                raise ValueError('specialized pipeline requires a plan YAML')
            plan = self.safe_pipeline_input(plan_path, label='specialized plan')
            if plan.suffix.lower() not in {'.yaml', '.yml'} or not plan.is_file():
                raise ValueError('specialized plan must be an existing YAML file')
            root = self.safe_pipeline_input(source_root or self.pipeline_root, label='specialized source root')
            if not root.is_dir():
                raise NotADirectoryError(root)
            arguments = [operation, '--plan', self.relative(plan), '--source-root', self.relative(root) or '.']
        else:
            if not study_directory:
                raise ValueError('specialized completion requires a study directory')
            study = self._named_output_input(study_directory, root_name=PIPELINE_OUTPUT, label='specialized completion study')
            if not study.is_dir():
                raise NotADirectoryError(study)
            arguments = ['complete', '--study-dir', self.relative(study)]
        if operation == 'validate':
            return self._command_request('specialized_pipeline.py', arguments)
        if operation == 'run':
            if run_name:
                name = str(run_name).strip()
                if not _SAFE_NAME.fullmatch(name):
                    raise ValueError('specialized run name must be filesystem-safe')
                arguments.extend(['--run-name', name])
            if resume:
                resume_path = self._named_output_input(resume, root_name=PIPELINE_OUTPUT, label='specialized resume directory')
                if not resume_path.is_dir():
                    raise NotADirectoryError(resume_path)
                arguments.extend(['--resume', self.relative(resume_path)])
            if upstream_study:
                upstream = self.safe_pipeline_input(upstream_study, label='specialized upstream study')
                if not upstream.is_dir():
                    raise NotADirectoryError(upstream)
                arguments.extend(['--upstream-study', self.relative(upstream)])
        if device is not None:
            arguments.extend(['--device', str(device)])
        if worker_count is not None:
            arguments.extend(['--jobs', str(worker_count)])
        if operation == 'run' and (not include_denoiser):
            arguments.append('--no-denoiser')
        if operation == 'complete' and dry_run:
            arguments.append('--dry-run')
        return self._command_request('specialized_pipeline.py', arguments)

    def build_specialized_request(self,
                                  *,
                                  operation: str,
                                  plan_path: str | Path | None = None,
                                  source_root: str | Path | None = None,
                                  output_name: str | None = None,
                                  study_directory: str | Path | None = None,
                                  case_id: str | None = None,
                                  prediction_file: str | Path | None = None,
                                  step: float | None = None,
                                  report_input: str | Path | None = None) -> CommandRequest:
        """Build the public artifact-only specialized report wrappers."""
        if operation == 'specialized-report':
            if not report_input:
                raise ValueError('specialized report requires an input directory')
            source = self.safe_pipeline_input(report_input, label='specialized report input')
            if not source.is_dir():
                raise NotADirectoryError(source)
            arguments = [operation, '--input', self.relative(source)]
            if output_name:
                output = self.output_directory(REPORT_OUTPUT, output_name)
                arguments.extend(['--output-name', output.name])
            return self._command_request('analyse_report.py', arguments)
        if operation not in {'specialized-validate', 'specialized-run'}:
            raise ValueError('unsupported specialized report operation')
        if not plan_path:
            raise ValueError('specialized operation requires a plan YAML')
        plan = self.safe_pipeline_input(plan_path, label='specialized plan')
        if plan.suffix.lower() not in {'.yaml', '.yml'} or not plan.is_file():
            raise ValueError('specialized plan must be an existing YAML file')
        root = self.safe_pipeline_input(source_root or self.pipeline_root, label='specialized source root')
        if not root.is_dir():
            raise NotADirectoryError(root)
        arguments = [operation, '--plan', self.relative(plan), '--source-root', self.relative(root) or '.']
        if operation == 'specialized-run':
            if output_name:
                output = self.output_directory(REPORT_OUTPUT, output_name)
                arguments.extend(['--output-name', output.name])
            if study_directory:
                study = self.safe_pipeline_input(study_directory, label='specialized source study')
                if not study.is_dir():
                    raise NotADirectoryError(study)
                arguments.extend(['--study-dir', self.relative(study)])
            if case_id:
                selected_case = str(case_id).strip()
                if not _SAFE_NAME.fullmatch(selected_case):
                    raise ValueError('specialized case ID must be filesystem-safe')
                arguments.extend(['--case-id', selected_case])
            if prediction_file:
                prediction = self.safe_pipeline_input(prediction_file, label='specialized prediction file')
                if not prediction.is_file():
                    raise FileNotFoundError(prediction)
                arguments.extend(['--prediction-file', self.relative(prediction)])
            if step is not None:
                numeric_step = float(step)
                if not math.isfinite(numeric_step) or not numeric_step > 0:
                    raise ValueError('specialized step must be positive')
                arguments.extend(['--step', str(numeric_step)])
        return self._command_request('analyse_report.py', arguments)

    def build_analysis_request(self,
                               *,
                               run_path: str | Path | None = None,
                               run_paths: Sequence[str | Path] = (),
                               mode: str,
                               preset: str,
                               modules: Sequence[str],
                               figures: Sequence[str] | None,
                               tables: Sequence[str] | None,
                               reference_case: str | None = None,
                               factor_paths: Sequence[str] = (),
                               bootstrap_resamples: int = 10000,
                               permutation_resamples: int = 100000,
                               statistics_seed: int = 42,
                               alpha: float = 0.05,
                               calibration_bins: int = 10,
                               output_name: str | None = None,
                               include_cases: Sequence[str] = (),
                               exclude_cases: Sequence[str] = (),
                               comparison_family: str = 'declared_comparison',
                               validation_depth: str = 'full',
                               on_missing: str = 'na',
                               allow_v2_compatibility: bool = True,
                               command: str = 'run') -> CommandRequest:
        if command not in {'run', 'validate'}:
            raise ValueError('analysis command must be run or validate')
        if command == 'validate' and output_name:
            raise ValueError('analysis validation does not accept an output name')
        requested_runs = [*run_paths]
        if run_path is not None:
            requested_runs.insert(0, run_path)
        if not requested_runs:
            raise ValueError('analysis requires at least one pipeline output')
        runs = [self.safe_pipeline_input(value, label='pipeline output') for value in requested_runs]
        for run in runs:
            run.relative_to((self.pipeline_root / PIPELINE_OUTPUT).resolve())
            if not run.is_dir():
                raise NotADirectoryError(run)
        names = [run.name for run in runs]
        if len(names) != len(set(names)):
            raise ValueError('analysis input directory names must be unique')
        arguments = [command, '--mode', str(mode)]
        if len(runs) == 1:
            arguments.extend(['--input', self.relative(runs[0])])
        else:
            for run in runs:
                arguments.extend(['--run', f'{run.name}={self.relative(run)}'])
        if output_name:
            output = self.output_directory(REPORT_OUTPUT, output_name)
            arguments.extend(['--output-name', output.name])
        elif command == 'run':
            canonical_names: set[str] = set()
            for run in runs:
                relative = run.relative_to((self.pipeline_root / PIPELINE_OUTPUT).resolve())
                if not relative.parts:
                    raise ValueError('analysis input must be below a pipeline run')
                run_root = self.pipeline_root / PIPELINE_OUTPUT / relative.parts[0]
                if not (run_root / 'study_manifest.json').is_file():
                    raise ValueError(
                        'automatic report naming requires canonical pipeline_output runs; enter an explicit report name for legacy inputs'
                    )
                canonical_names.add(relative.parts[0])
            if len(canonical_names) != 1:
                raise ValueError(
                    'automatic report naming requires all inputs to share one pipeline_output/<run>; enter an explicit report name')
        for case in include_cases:
            arguments.extend(['--include-case', str(case)])
        for case in exclude_cases:
            arguments.extend(['--exclude-case', str(case)])
        arguments.extend(['--comparison-family', str(comparison_family)])
        if preset:
            arguments.extend(['--preset', preset])
        for module in modules:
            arguments.extend(['--module', str(module)])
        if figures is not None:
            arguments.extend(['--figures', ','.join(figures) if figures else 'none'])
        if tables is not None:
            arguments.extend(['--tables', ','.join(tables) if tables else 'none'])
        if reference_case:
            arguments.extend(['--reference-case', str(reference_case)])
        for path in factor_paths:
            arguments.extend(['--factor-path', str(path)])
        arguments.extend([
            '--bootstrap-resamples',
            str(int(bootstrap_resamples)), '--permutation-resamples',
            str(int(permutation_resamples)), '--statistics-seed',
            str(int(statistics_seed)), '--alpha',
            str(float(alpha)), '--calibration-bins',
            str(int(calibration_bins)), '--validation-depth',
            str(validation_depth), '--on-missing',
            str(on_missing)
        ])
        arguments.append('--v2-compatibility' if allow_v2_compatibility else '--no-v2-compatibility')
        return self._command_request('analyse_report.py', arguments)

    def _validated_inference_payload(self, payload: Mapping[str, Any]) -> InferenceInput:
        """Validate the UI/CLI manifest boundary without running preprocessing."""
        _assert_inference_source_contract(payload)
        participant_id = str(payload.get('participant_id', '')).strip()
        if not _SAFE_NAME.fullmatch(participant_id):
            raise ValueError('inference participant_id must be a non-empty filesystem-safe ID')
        raw_files = payload.get('files')
        if not isinstance(raw_files, list) or not raw_files:
            raise ValueError('inference manifest requires a non-empty files list')
        files: list[InferenceFile] = []
        ids: set[str] = set()
        roles: set[str] = set()
        labels: set[str] = set()
        for index, raw in enumerate(raw_files):
            row = _mapping(raw, label=f'inference file {index + 1}')
            file_id = str(row.get('file_id', '')).strip()
            role = _concrete_role(row.get('role'))
            family = _role_family(role)
            if not _SAFE_NAME.fullmatch(file_id) or file_id in ids:
                raise ValueError('inference file_id values must be unique filesystem-safe IDs')
            ids.add(file_id)
            roles.add(family)
            source = self.safe_input(str(row.get('path', '')), label=f'inference file {file_id}', suffixes={'.csv'})
            if not source.is_file():
                raise ValueError(f'inference source must be a CSV file: {source}')
            label = _inference_label(row.get('label'))
            if label is not None:
                labels.add(_value_key(label))
            files.append(InferenceFile(file_id=file_id, role=role, path=source.as_posix(), label=label))
        if roles & _DYNAMIC_ROLE_FAMILIES and 'B' not in roles:
            raise ValueError(f'missing_static_b_calibration: {MISSING_B_TODO}')
        if len(labels) > 1:
            raise ValueError('all labelled files for one participant must share one label')
        return InferenceInput(participant_id=participant_id, files=tuple(files), labelled_participant_count=1 if labels else 0)

    def read_inference_manifest(self, manifest_path: str | Path) -> InferenceInput:
        path = self.safe_input(manifest_path, label='inference manifest', suffixes={'.yaml', '.yml', '.json'})
        payload = _json_mapping(path) if path.suffix.lower() == '.json' else _yaml_mapping(path)
        return self._validated_inference_payload(payload)

    def materialize_inference_manifest(self, *, participant_id: str, files: Sequence[Mapping[str, Any]],
                                       source_contract_confirmed: bool) -> Path:
        """Atomically persist one hash-addressed Dashboard inference request.

        The browser never controls the destination. Source CSVs remain read-only;
        only a compact manifest is written below ``pipeline_output``.
        """
        if source_contract_confirmed is not True:
            raise ValueError('confirm the exact 400 Hz eight-channel raw CSV source contract')
        editable_rows = [_mapping(row, label=f'inference table row {index + 1}') for index, row in enumerate(files)]
        payload = {
            'participant_id':
            str(participant_id).strip(),
            'source_contract':
            copy.deepcopy(INFERENCE_SOURCE_CONTRACT),
            'files': [{
                'file_id': row.get('file_id'),
                'path': row.get('path'),
                'role': row.get('role'),
                'label': row.get('label')
            } for row in editable_rows]
        }
        validated = self._validated_inference_payload(payload)
        normalized = {
            'participant_id':
            validated.participant_id,
            'source_contract':
            copy.deepcopy(INFERENCE_SOURCE_CONTRACT),
            'files': [{
                'file_id': row.file_id,
                'path': self.cli_input_path(row.path),
                'role': row.role,
                'label': row.label
            } for row in validated.files]
        }
        encoded = yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True)
        digest = hashlib.sha256(encoded.encode('utf-8')).hexdigest()
        request_root = (self.pipeline_root / PIPELINE_OUTPUT / '.dashboard_requests' / 'inference').resolve()
        request_root.relative_to(self.pipeline_root)
        request_root.mkdir(parents=True, exist_ok=True)
        target = request_root / f'request_{digest}.yaml'
        if target.exists():
            if target.read_text(encoding='utf-8') != encoded:
                raise RuntimeError(f'inference request hash collision: {target}')
        else:
            temporary = target.with_name(f'.{target.name}.tmp-{uuid.uuid4().hex}')
            try:
                temporary.write_text(encoded, encoding='utf-8')
                temporary.replace(target)
            finally:
                if temporary.exists():
                    temporary.unlink()
        self.read_inference_manifest(target)
        return target

    def build_inference_request(self, *, model_export: str | Path, case_id: str | None, input_manifest: str | Path) -> CommandRequest:
        """Build the exact public CLI corresponding to Dashboard Infer."""
        defaults = self.load_model_defaults(model_export, case_id)
        if not defaults.inference_capability.get('available'):
            raise RuntimeError('selected model_config has no deployable learned-weight bundle')
        manifest = self.safe_input(input_manifest, label='inference manifest', suffixes={'.yaml', '.yml', '.json'})
        self.read_inference_manifest(manifest)
        if not _SAFE_NAME.fullmatch(defaults.case_id):
            raise ValueError('model export case ID is not filesystem-safe')
        arguments = ('infer', '--model-config', defaults.export_directory, '--case-id', defaults.case_id, '--input-manifest',
                     self.cli_input_path(manifest))
        request = self._command_request('pipeline.py', arguments)
        defaults_config_path = self.safe_pipeline_input(defaults.config_path, label='resolved model configuration')
        defaults_payload = yaml.safe_load(defaults_config_path.read_text(encoding='utf-8'))
        evidence = {
            'schema_version': 'ppg_frailty.dashboard_inference_request.v1',
            'command': {
                'script': request.script,
                'arguments': list(request.arguments)
            },
            'model_export': defaults.export_directory,
            'case_id': defaults.case_id,
            'resolved_pipeline_config': defaults_payload,
            'input_manifest': yaml.safe_load(manifest.read_text(encoding='utf-8'))
        }
        return CommandRequest(script=request.script,
                              arguments=request.arguments,
                              display=request.display,
                              resolved_yaml=yaml.safe_dump(evidence, sort_keys=False, allow_unicode=True))

    def infer(self, *, model_export: str | Path, case_id: str | None, input_manifest: str | Path) -> Mapping[str, Any]:
        validated_input = self.read_inference_manifest(input_manifest)
        defaults = self.load_model_defaults(model_export, case_id)
        if not defaults.inference_capability.get('available'):
            raise RuntimeError(
                'selected model_config has no deployable learned-weight bundle; configuration provenance alone cannot run Infer')
        try:
            module = importlib.import_module('ppg_frailty.v5.inference_service')
        except ModuleNotFoundError as error:
            raise RuntimeError('V5 inference service is unavailable; raw-device preprocessing remains fail-closed') from error
        function = next(
            (getattr(module, name)
             for name in ('infer_from_manifest', 'run_inference', 'infer_participant') if callable(getattr(module, name, None))), None)
        if function is None:
            raise RuntimeError('V5 inference service exposes no supported inference API')
        available = {
            'model_config_directory': self.safe_pipeline_input(model_export, label='model config export'),
            'model_export': self.safe_pipeline_input(model_export, label='model config export'),
            'case_id': defaults.case_id,
            'input_manifest': self.safe_input(input_manifest, label='inference manifest'),
            'validated_input': validated_input,
            'pipeline_root': self.pipeline_root
        }
        signature = inspect.signature(function)
        accepts_kwargs = any((value.kind == inspect.Parameter.VAR_KEYWORD for value in signature.parameters.values()))
        kwargs = available if accepts_kwargs else {name: value for name, value in available.items() if name in signature.parameters}
        result = function(**kwargs)
        if isinstance(result, Mapping):
            return dict(result)
        if hasattr(result, 'to_dict'):
            return dict(result.to_dict())
        raise TypeError('V5 inference service must return a mapping')


__all__ = [
    'COMPARISON_SEQUENCE_SCHEMA', 'CommandRequest', 'INFERENCE_SOURCE_CONFIRMATION', 'INFERENCE_SOURCE_CONTRACT', 'InferenceInput',
    'INHERIT_MODULE', 'MISSING_B_TODO', 'MODEL_CONFIG', 'ModelDefaults', 'PIPELINE_OUTPUT', 'REPORT_OUTPUT',
    'SINGLE_PARTICIPANT_NOTICE', 'V5ControlService', 'WORKFLOW_STAGES', 'changed_assignments', 'comparison_sequence_cli',
    'comparison_sequence_export_yaml', 'comparison_sequence_yaml', 'flatten_parameters', 'registry_sha256',
    'validate_comparison_sequence_payload'
]
