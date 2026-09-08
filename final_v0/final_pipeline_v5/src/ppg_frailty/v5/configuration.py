"""Strict preset and CLI-override handling for the V5 application surface."""
from __future__ import annotations
import copy
from functools import lru_cache
import hashlib
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping
import yaml
from ..config import canonical_json_bytes, load_formal_ablation_profiles, load_formal_experiment_catalog, validate_config_payload
from ..module_registry import list_modules, model_factory_contract

_SAFE_ID = re.compile('^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$')

@dataclass(frozen=True)
class Preset:
    """One reviewed base configuration exposed by the V5 CLI."""
    name: str
    relative_path: str
    purpose: str
    source_identity: str
    is_default: bool = False
    case_id: str | None = None

PRESETS: Mapping[str, Preset] = {
    'baseline': Preset(name='baseline', relative_path='configs/presets/baseline.yaml', purpose='Unchanged V2 reference baseline and V5 default.',
                       source_identity='V2 configs/reference_static_role_aware_v2.yaml', is_default=True),
    'finalcase': Preset(name='finalcase', relative_path='configs/presets/finalcase.yaml', purpose='User-selected merged-study Rank 2: tuned_all_roles_small_no_gravity.',
                        source_identity='V2 case small_no_gravity__raw__tuned_all_roles__inception_small_no_gravity',
                        case_id='tuned_all_roles__inception_small_no_gravity'),
    'feature_vector': Preset(name='feature_vector', relative_path='configs/presets/feature_vector.yaml', purpose='V2 feature-vector reference workflow.',
                             source_identity='V2 configs/reference_static_feature_vector_v2.yaml'),
    'feature_matrix': Preset(name='feature_matrix', relative_path='configs/presets/feature_matrix.yaml', purpose='V2 ordered feature-matrix reference workflow.',
                             source_identity='V2 configs/reference_static_feature_matrix_v2.yaml'), 'fusion': Preset(name='fusion', relative_path='configs/presets/fusion.yaml',
                                                                                                                     purpose='V2 signal/feature fusion reference workflow.',
                                                                                                                     source_identity='V2 configs/reference_static_fusion_v2.yaml')
}

# One-to-one module fields, switches, and help ownership share these catalogs.
_DIRECT_MODULE_PATHS: Mapping[str, str] = {
    'class_weighting': 'training.class_weighting', 'epoch_selection': 'training.epoch_rule', 'gap_repair': 'signal.gap_repair.method', 'imu_gravity': 'signal.imu.gravity_method',
    'loss': 'training.loss', 'optimizer': 'training.optimizer', 'ppg_filter': 'signal.ppg_filter.family', 'quality_mode': 'quality.mode', 'representation': 'representation_mode',
    'sampler': 'training.sampler', 'training_balance': 'training.training_balance', 'window_quality_selection': 'quality.window_selection.policy'
}
_SWITCH_MODULE_PATHS: Mapping[str, str] = {'motion_detector_switch': 'artifact.motion_detector_enabled', 'denoiser_switch': 'artifact.denoiser_enabled'}
_PARAMETER_MODULE_PATHS: Mapping[str, str] = {
    **{path: family
       for family, path in _DIRECT_MODULE_PATHS.items()}, 'aggregation.balance_line': 'aggregation', 'aggregation.quality_weight_source': 'quality_weight_source',
    'artifact.reducer': 'artifact', 'features.enabled_groups': 'feature_group', 'model.model_id': 'model', 'signal.dl_resampling.method': 'dl_resampling',
    'signal.normalization.raw_imu': 'normalization', 'signal.normalization.raw_ppg': 'normalization', 'signal.peak_detector.detector_id': 'peak_detector',
    'training.class_count_basis': 'class_count_basis'
}
_ARTIFACT_VERSIONS: Mapping[str, str] = {
    'identity': 'identity_v1', 'nlms_imu_anc': 'nlms_delay_taps_v1', 'ssa_decomposition': 'ssa_hankel_cardiac_select_v1', 'spectral_mask': 'spectral_mask_v1',
    'pca_bss': 'pca_component_select_v2', 'fastica_bss': 'fastica_component_select_v2', 'nmf_bss': 'nmf_shared_spectral_basis_v1',
    'emd_sifting_rate_only': 'historical_derived_funcs_emd_v2', 'ceemd_lite_nlms_legacy': 'historical_funcs_ceemd_nlms_v2', 'dwt_a2_legacy': 'historical_pttppg_v7_dwt_a2_v2'
}

# CLI parsing keeps YAML values and registry module identities explicit.
def parse_yaml_value(text: str) -> Any:
    """Parse one CLI value using the same scalar/list syntax as config YAML."""
    if text.strip().lower() in {'on', 'off', 'yes', 'no'}:
        return text.strip()
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ValueError(f'invalid YAML value {text!r}: {error}') from error

def parse_assignment(text: str) -> tuple[str, Any]:
    """Parse ``DOTTED.PATH=YAML_VALUE`` without guessing string types."""
    if '=' not in text:
        raise ValueError('expected DOTTED.PATH=YAML_VALUE')
    path, raw = text.split('=', 1)
    path = path.strip()
    if not path or any((not part for part in path.split('.'))):
        raise ValueError('configuration path must be a non-empty dotted path')
    return (path, parse_yaml_value(raw))

def parse_module_assignment(text: str) -> tuple[str, str]:
    """Parse and validate one ``FAMILY=MODULE_ID`` selection."""
    if '=' not in text:
        raise ValueError('expected FAMILY=MODULE_ID')
    family, module_id = (part.strip() for part in text.split('=', 1))
    if not family or not module_id:
        raise ValueError('module family and module ID must be non-empty')
    registered = {(str(row['family']), str(row['module_id'])) for row in list_modules()}
    if (family, module_id) not in registered:
        raise ValueError(f'unregistered module selection: {family}={module_id}')
    return (family, module_id)

def _mapping_at(payload: MutableMapping[str, Any], path: str, *, create_leaf: bool) -> tuple[MutableMapping[str, Any], str]:
    parts = path.split('.')
    current: MutableMapping[str, Any] = payload
    for part in parts[:-1]:
        value = current.get(part)
        if not isinstance(value, MutableMapping):
            raise KeyError(f'configuration parent does not exist: {path}')
        current = value
    leaf = parts[-1]
    if not create_leaf and leaf not in current:
        raise KeyError(f'configuration field does not exist: {path}')
    return (current, leaf)

def set_dotted(payload: MutableMapping[str, Any], path: str, value: Any) -> None:
    """Set an existing section's leaf; schema validation rejects unknown keys."""
    parent, leaf = _mapping_at(payload, path, create_leaf=True)
    parent[leaf] = copy.deepcopy(value)

def unset_dotted(payload: MutableMapping[str, Any], path: str) -> None:
    """Remove an explicitly named optional field."""
    parent, leaf = _mapping_at(payload, path, create_leaf=False)
    del parent[leaf]

def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(value, Mapping):
        raise TypeError(f'configuration root must be a mapping: {path}')
    return copy.deepcopy(dict(value))

def _preset_entry(pipeline_root: Path, preset: Preset) -> Mapping[str, Any]:
    registry_path = pipeline_root / 'configs/presets/registry.yaml'
    registry = _load_yaml_mapping(registry_path)
    entries = registry.get('presets')
    entry = entries.get(preset.name, {}) if isinstance(entries, Mapping) else {}
    return dict(entry) if isinstance(entry, Mapping) else {}

def _drop_model_derived_provenance(payload: MutableMapping[str, Any]) -> None:
    model = payload['model']
    contract = model_factory_contract(str(model['model_id']))
    for field in contract['derived_provenance_fields']:
        model.pop(str(field), None)

def _formal_ablation_entry(pipeline_root: Path, family: str, profile_id: str) -> dict[str, Any]:
    catalog = load_formal_ablation_profiles(pipeline_root / 'configs/formal_ablation_profiles_v2.yaml')
    profile_family = catalog['families'][family]
    matches = [dict(row) for row in profile_family['entries'] if str(row['profile_id']) == profile_id]
    if len(matches) != 1:
        raise ValueError(f'no unique V2 profile {family}={profile_id}')
    return matches[0]

def _apply_aggregation_line(payload: MutableMapping[str, Any], module_id: str, pipeline_root: Path) -> None:
    profile_id = {'line_a_equal_files': 'equal_files_line_a_ablation', 'line_b_equal_role_families': 'role_aware_equal_roles'}[module_id]
    profile = _formal_ablation_entry(pipeline_root, 'aggregation_balance', profile_id)
    line_b = module_id == 'line_b_equal_role_families'
    set_dotted(payload, 'training.training_balance', str(profile['training_balance']))
    payload['aggregation'].update({
        'balance_line': str(profile['balance_line']), 'hierarchy': list(profile['hierarchy']), 'window_to_file': 'ordinary_mean',
        'file_to_role': 'ordinary_mean' if line_b else 'not_applicable', 'role_to_participant': 'ordinary_mean' if line_b else 'not_applicable',
        'missing_role_policy': 'mean_available_roles' if line_b else 'not_applicable', 'quality_weighting': False, 'quality_weight_source': 'none', 'quality_weight_levels': [],
        'direct_all_window_participant_mean': False
    })

def _quality_diagnostics_only_controls_are_neutral(payload: Mapping[str, Any]) -> bool:
    artifact = payload['artifact']
    motion_enabled = bool(artifact.get('motion_detector_enabled', False))
    denoiser_enabled = bool(artifact.get('denoiser_enabled', str(artifact.get('reducer', 'identity')) != 'identity'))
    return not motion_enabled and (not denoiser_enabled)

def _apply_comparison_profile(payload: MutableMapping[str, Any], module_id: str, pipeline_root: Path) -> None:
    if module_id in {'epoch_7_ablation', 'epoch_15_ablation'}:
        if str(payload['training'].get('epoch_rule', 'fixed_epoch')) != 'fixed_epoch':
            raise ValueError(f'comparison_profile={module_id} requires fixed_epoch')
        backend = model_factory_contract(str(payload['model']['model_id']))['execution_backend']
        if backend != 'torch':
            raise ValueError(f'comparison_profile={module_id} is deep-model-only')
        profile = _formal_ablation_entry(pipeline_root, 'deep_fixed_epoch', module_id)
        set_dotted(payload, 'training.fixed_epochs', int(profile['fixed_epochs']))
        return
    if module_id == 'direct_filter_0p5_to_5hz_ablation':
        profile = _formal_ablation_entry(pipeline_root, 'direct_filter', module_id)
        set_dotted(payload, 'signal.ppg_filter.low_hz', float(profile['low_hz']))
        set_dotted(payload, 'signal.ppg_filter.high_hz', float(profile['high_hz']))
        payload['signal']['analysis_view'].pop('direct_source', None)
        return
    if module_id == 'line_b_equal_role_families':
        _apply_aggregation_line(payload, module_id, pipeline_root)
        return
    if module_id == 'quality_diagnostics_only':
        if not _quality_diagnostics_only_controls_are_neutral(payload):
            raise ValueError('comparison_profile=quality_diagnostics_only requires motion and denoiser modules to be disabled')
        set_dotted(payload, 'quality.mode', 'diagnostics_only')
        return
    unsafe_reason = {
        'imu_lpf_0p3hz_ablation': 'the old ablation name now aliases the V2 reference IMU profile',
        'fixed_kernel_samples_resampling_ablation': 'it names a twelve-case family, not one configuration',
        'fixed_kernel_samples_context_10s_400hz_ablation': 'its registered materializer binding does not exist',
        'fixed_kernel_samples_dilation2_ablation': 'its registered materializer binding does not exist'
    }.get(module_id, 'it has no exact V2 in-memory materializer')
    raise ValueError(f'comparison_profile={module_id} cannot be selected safely: {unsafe_reason}; use a reviewed --config/--plan')

def _apply_feature_groups(payload: MutableMapping[str, Any], module_ids: Iterable[str]) -> None:
    from ..features.registry import canonicalize_feature_groups
    mode = str(payload.get('representation_mode'))
    if mode not in {'feature_vector', 'feature_matrix', 'fusion'}:
        raise ValueError('feature_group selections require feature_vector, feature_matrix, or fusion')
    selected = list(canonicalize_feature_groups(tuple(module_ids)))
    set_dotted(payload, 'features.enabled_groups', selected)

def _replace_model_from_catalog(payload: MutableMapping[str, Any], module_id: str, pipeline_root: Path) -> None:
    current = payload.get('model')
    if isinstance(current, Mapping) and str(current.get('model_id')) == module_id:
        return
    catalog_path = pipeline_root / 'configs/formal_experiment_catalog_v2.yaml'
    catalog = load_formal_experiment_catalog(catalog_path)
    matches = [row for row in catalog.get('entries', ()) if isinstance(row, Mapping) and str(row.get('model', {}).get('model_id', '')) == module_id]
    if len(matches) != 1:
        raise ValueError(f'model={module_id} has {len(matches)} complete catalog definitions; use --config plus --set for a non-unique or historical model')
    entry = matches[0]
    representation = str(entry['representation_mode'])
    if str(payload.get('representation_mode')) != representation:
        raise ValueError(f'model={module_id} requires representation={representation}; select the matching representation preset/config first')
    selected_model = copy.deepcopy(dict(entry['model']))
    payload['model'] = selected_model
    set_dotted(payload, 'output.write_member_oof', 'member_seeds' in selected_model)

# Correlated families retain dedicated materializers; simple families use the tables above.
def _apply_module(payload: MutableMapping[str, Any], family: str, module_id: str, pipeline_root: Path) -> None:
    routed = {'model': _replace_model_from_catalog, 'aggregation': _apply_aggregation_line, 'comparison_profile': _apply_comparison_profile}.get(family)
    if routed is not None:
        routed(payload, module_id, pipeline_root)
        return
    if family == 'feature_group':
        raise RuntimeError('feature_group selections must be applied as one batch')
    if family == 'shapeformer_discovery_balance':
        contract = model_factory_contract(str(payload['model']['model_id']))
        supported_models = {'ShapeFormerChannelSpecificOSD', 'ShapeFormerChannelSpecificScalarDistanceAblation'}
        if str(contract['canonical_model_name']) not in supported_models:
            raise ValueError('shapeformer_discovery_balance requires a channel-specific ShapeFormer model')
        set_dotted(payload, 'model.discovery_balance', module_id)
        _drop_model_derived_provenance(payload)
        return
    if family == 'class_count_basis':
        set_dotted(payload, 'training.class_count_basis', module_id)
        if module_id == 'participant':
            set_dotted(payload, 'training.class_weighting', 'inverse_frequency')
        return
    if family == 'prv_backend':
        if module_id != 'local':
            raise ValueError(f'prv_backend={module_id} is comparison-only and cannot enter classifier features')
        set_dotted(payload, 'features.prv_primary_backend', 'local_manual')
        return
    if family == 'motion_evidence':
        if module_id != 'reused_frailty29_all29_bundle':
            raise ValueError(f'motion_evidence={module_id} is external or historical audit evidence and cannot enter the classifier pipeline')
        from ..quality.motion_bundle_adapter import resolve_reused_motion_detector_config
        artifact = payload['artifact']
        declared_motion = artifact.get('motion_detector')
        if declared_motion is None:
            declared_motion = {}
        if not isinstance(declared_motion, Mapping):
            raise TypeError('artifact.motion_detector must be a mapping')
        defaults = resolve_reused_motion_detector_config().to_mapping(include_enabled=False)
        artifact['motion_detector'] = {**defaults, **dict(declared_motion)}
        set_dotted(payload, 'artifact.motion_detector_enabled', True)
        return
    if family == 'imu_gravity':
        payload['signal']['imu'] = {'gravity_method': module_id}
        return
    if family == 'normalization':
        if module_id.startswith('ppg_'):
            set_dotted(payload, 'signal.normalization.raw_ppg', module_id.removeprefix('ppg_'))
        elif module_id.startswith('imu_'):
            set_dotted(payload, 'signal.normalization.raw_imu', module_id.removeprefix('imu_'))
        return
    if family == 'peak_detector':
        set_dotted(payload, 'signal.peak_detector.detector_id', module_id)
        peak = payload['signal']['peak_detector']
        peak.pop('parameters', None)
        return
    if family == 'dl_resampling':
        enabled = module_id != 'off_identity_source_grid'
        set_dotted(payload, 'signal.dl_resampling.enabled', enabled)
        if enabled:
            set_dotted(payload, 'signal.dl_resampling.method', module_id)
        return
    if family in _SWITCH_MODULE_PATHS:
        set_dotted(payload, _SWITCH_MODULE_PATHS[family], module_id == 'enabled')
        return
    if family == 'artifact':
        set_dotted(payload, 'artifact.reducer', module_id)
        set_dotted(payload, 'artifact.reducer_version', _ARTIFACT_VERSIONS[module_id])
        set_dotted(payload, 'artifact.parameters', {})
        enabled = module_id != 'identity'
        set_dotted(payload, 'artifact.denoiser_enabled', enabled)
        representation = str(payload.get('representation_mode'))
        degraded_policy = 'drop' if not enabled else 'denoise_then_extract_rate_features' if representation == 'feature_vector' else 'denoise_then_compare_rate_exclude'
        set_dotted(payload, 'artifact.degraded_policy', degraded_policy)
        return
    if family == 'quality_weight_source':
        set_dotted(payload, 'aggregation.quality_weight_source', module_id)
        set_dotted(payload, 'aggregation.quality_weighting', module_id != 'none')
        return
    path = _DIRECT_MODULE_PATHS.get(family)
    if path is None:
        raise ValueError(f'module family {family!r} needs correlated scientific parameters; select it through --config/--plan or explicit --set values')
    set_dotted(payload, path, module_id)

# All entry modes converge here before the unchanged V2 schema validator and canonical hash.
def resolve_configuration(*, pipeline_root: str | Path, preset: str = 'baseline', config_path: str | Path | None = None, assignments: Iterable[str] = (),
                          unsets: Iterable[str] = (), modules: Iterable[str] = (), config_id: str | None = None, manual: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve one complete config and return it with a provenance record."""
    root = Path(pipeline_root).resolve()
    assignments = tuple(assignments)
    unsets = tuple(unsets)
    modules = tuple(modules)
    if manual and config_path is not None:
        raise ValueError('manual configuration cannot also use a config file')
    if manual and preset != 'baseline':
        raise ValueError('manual configuration cannot also select a named preset')
    if manual and config_id is None:
        raise ValueError('--manual requires an explicit --config-id')
    if manual and (not assignments) and (not modules) and (not unsets):
        raise ValueError('--manual requires explicit module or parameter values')
    if config_path is None:
        try:
            selected = PRESETS[preset]
        except KeyError as error:
            raise ValueError(f'unknown preset: {preset}') from error
        source = (root / selected.relative_path).resolve()
        source_identity = selected.source_identity
    else:
        source = Path(config_path).resolve()
        source_identity = f'explicit:{source}'
    if not source.is_file():
        raise FileNotFoundError(source)
    if manual:
        source_identity = 'manual_schema_defaults_from_baseline'
    payload = _load_yaml_mapping(source)
    if manual:
        payload = validate_config_payload(payload)
    parsed_modules = [parse_module_assignment(raw) for raw in modules]
    requested_modules = [{'family': family, 'module_id': module_id} for family, module_id in parsed_modules]
    for family, module_id in parsed_modules:
        if family != 'feature_group':
            _apply_module(payload, family, module_id, root)
    feature_groups = [module_id for family, module_id in parsed_modules if family == 'feature_group']
    if feature_groups:
        _apply_feature_groups(payload, feature_groups)
    requested_assignments: list[dict[str, Any]] = []
    for raw in assignments:
        path, value = parse_assignment(raw)
        set_dotted(payload, path, value)
        requested_assignments.append({'path': path, 'value': value})
    requested_unsets = tuple((str(path) for path in unsets))
    for path in requested_unsets:
        unset_dotted(payload, path)
    if any((family == 'comparison_profile' and module_id == 'quality_diagnostics_only'
            for family, module_id in parsed_modules)) and (not _quality_diagnostics_only_controls_are_neutral(payload)):
        raise ValueError('comparison_profile=quality_diagnostics_only requires motion and denoiser modules to be disabled')
    customized = bool(requested_modules or requested_assignments or requested_unsets or config_id)
    if config_id is not None:
        if not _SAFE_ID.fullmatch(config_id):
            raise ValueError('--config-id is not a safe identifier')
        payload['config_id'] = config_id
    elif customized:
        before_id = copy.deepcopy(payload)
        before_id['config_id'] = 'v5_cli_resolution'
        candidate = validate_config_payload(before_id)
        digest = hashlib.sha256(canonical_json_bytes(candidate)).hexdigest()[:12]
        payload['config_id'] = f'v5_{preset}_{digest}'
    validated = validate_config_payload(payload)
    canonical_sha256 = hashlib.sha256(canonical_json_bytes(validated)).hexdigest()
    provenance = {
        'schema_version': 'ppg_frailty.v5_configuration_resolution.v1', 'preset': preset if config_path is None and (not manual) else None, 'manual': bool(manual),
        'source_path': str(source), 'source_identity': source_identity, 'source_file_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'resolved_config_id': validated['config_id'], 'resolved_config_sha256': canonical_sha256, 'module_selections': requested_modules, 'assignments': requested_assignments,
        'unsets': list(requested_unsets), 'numerical_engine': 'copied_v2_contract'
    }
    json.dumps(provenance, allow_nan=False)
    return (validated, provenance)

def _leaf_values(value: Any, prefix: str = '') -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping) and value:
        for key in sorted(value):
            child = f'{prefix}.{key}' if prefix else str(key)
            yield from _leaf_values(value[key], child)
        return
    yield (prefix, copy.deepcopy(value))

def _missing_paths(base: Mapping[str, Any], target: Mapping[str, Any], prefix: str = '') -> list[str]:
    missing: list[str] = []
    for key, value in base.items():
        path = f'{prefix}.{key}' if prefix else str(key)
        if key not in target:
            missing.append(path)
        elif isinstance(value, Mapping) and isinstance(target[key], Mapping):
            missing.extend(_missing_paths(value, target[key], path))
    return missing

def _mapping_scaffolds(base: MutableMapping[str, Any], target: Mapping[str, Any], prefix: str = '') -> list[str]:
    paths: list[str] = []
    for key in sorted(target):
        value = target[key]
        if not isinstance(value, Mapping):
            continue
        path = f'{prefix}.{key}' if prefix else str(key)
        current = base.get(key)
        if not isinstance(current, MutableMapping):
            base[key] = {}
            current = base[key]
            paths.append(path)
        paths.extend(_mapping_scaffolds(current, value, path))
    return paths

def _cli_value(value: Any) -> str:
    if isinstance(value, float):
        text = repr(value)
        if 'e' in text.lower():
            mantissa, exponent = re.split('[eE]', text, maxsplit=1)
            if '.' not in mantissa:
                mantissa += '.0'
            return f'{mantissa}e{exponent}'
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)

# Manual CLI export round-trips every leaf and verifies the resolved hash immediately.
def manual_cli_tokens(pipeline_root: str | Path, *, source_preset: str = 'finalcase') -> tuple[str, ...]:
    """Return a preset-free CLI spelling with every target leaf explicit."""
    root = Path(pipeline_root).resolve()
    if source_preset not in PRESETS:
        raise ValueError(f'unknown source preset: {source_preset}')
    target, _ = resolve_configuration(pipeline_root=root, preset=source_preset)
    base, _ = resolve_configuration(pipeline_root=root, preset='baseline')
    tokens = ['--manual', '--config-id', str(target['config_id'])]
    for path in _mapping_scaffolds(copy.deepcopy(base), target):
        tokens.extend(('--set', f'{path}={{}}'))
    for path, value in _leaf_values(target):
        if path == 'config_id':
            continue
        tokens.extend(('--set', f'{path}={_cli_value(value)}'))
    for path in sorted(_missing_paths(base, target)):
        tokens.extend(('--unset', path))
    resolved, provenance = resolve_configuration(pipeline_root=root, assignments=tuple(
        (tokens[index + 1] for index, value in enumerate(tokens[:-1]) if value == '--set')), unsets=tuple(
            (tokens[index + 1] for index, value in enumerate(tokens[:-1]) if value == '--unset')), config_id=str(target['config_id']), manual=True)
    expected_hash = hashlib.sha256(canonical_json_bytes(target)).hexdigest()
    if provenance['resolved_config_sha256'] != expected_hash or resolved != target:
        raise RuntimeError('generated manual CLI does not reproduce its source preset')
    return tuple(tokens)

def manual_cli_command(pipeline_root: str | Path, *, source_preset: str = 'finalcase', executable: str = 'python pipeline.py run') -> str:
    """Return a shell-safe, directly runnable manual command."""
    return executable + ' ' + shlex.join(manual_cli_tokens(pipeline_root, source_preset=source_preset))

# Help rows and the cross-preset/study/module union use one leaf-contract projection.
def _parameter_rows_for_config(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    known_ranges = {
        'config_id': 'portable ID: [A-Za-z0-9][A-Za-z0-9_.-]{0,159}', 'schema_version': 'fixed enum: ppg_frailty.pipeline_config.v2', 'training.batch_size': 'integer >= 1',
        'training.fixed_epochs': 'integer >= 1 for fixed_epoch', 'training.learning_rate': 'finite float > 0', 'training.weight_decay': 'finite float >= 0',
        'training.class_weight_beta': 'finite float in [0,1); effective_number only', 'training.focal_gamma': 'finite float >= 0; focal_loss only',
        'training.label_smoothing': 'finite float in [0,1)', 'training.gradient_clip_norm': 'null or finite float > 0',
        'training.samples_per_epoch': 'null or integer >= 1; replacement samplers only',
        'training.participant_window_quota': 'all | positive integer | fraction in (0,1] | percentage string in (0,100]%',
        'training.maximum_inner_epochs': '0 for fixed_epoch; integer >= 1 for inner selection', 'training.inner_patience': '0 for fixed_epoch; integer >= 1 for inner selection',
        'training.inner_grouped_folds': '0 for fixed_epoch; integer >= 2 for inner selection', 'training.num_workers': 'integer >= 0', 'training.seed': 'integer in [0,4294967295]',
        'training.optimizer_parameters.betas': 'two finite floats, each in [0,1)', 'training.optimizer_parameters.eps': 'finite float > 0',
        'signal.internal_fs_hz': 'finite float > 0', 'signal.dl_resampling.target_fs_hz': 'finite float > 0', 'signal.dl_resampling.preserve_feature_grid_hz': 'finite float > 0',
        'signal.ppg_filter.low_hz': 'finite float > 0 and < high_hz', 'signal.ppg_filter.high_hz': 'finite float > low_hz and below Nyquist',
        'signal.ppg_filter.order': 'integer >= 1', 'signal.gap_repair.max_gap_samples': 'integer >= 0', 'signal.peak_detector.min_observation_sec': 'finite float > 0',
        'signal.peak_detector.min_peaks': 'integer >= 1', 'signal.peak_detector.parameters.minimum_heart_rate_bpm': 'finite float > 0',
        'signal.peak_detector.parameters.overlap_fraction': 'finite float in [0,1)', 'signal.peak_detector.parameters.target_downsample_hz': 'finite float > 0',
        'signal.peak_detector.parameters.window_s': 'finite float > 0', 'signal.imu.calibration_start_s': 'finite float >= 0 and < calibration_stop_s',
        'signal.imu.calibration_stop_s': 'finite float > calibration_start_s', 'signal.imu.gravity_mps2': 'finite float > 0', 'signal.imu.sensor_filter_order': 'integer >= 1',
        'signal.imu.sensor_lowpass_acc_hz': 'finite float > 0 and below Nyquist', 'signal.imu.sensor_lowpass_gyro_hz': 'finite float > 0 and below Nyquist',
        'signal.normalization.clip_after_scale': 'two finite floats [low,high] with low < high', 'signal.normalization.scale_epsilon': 'finite float > 0',
        'signal.normalization.standard_ddof': 'integer 0 or 1', 'quality.window_selection.keep_fraction': 'float in (0,1]', 'quality.long_gap_max_samples': 'integer >= 0',
        'quality.flatline_duration_s': 'finite float > 0', 'model.dropout': 'float in [0,1)', 'model.architecture_parameters.classifier_dropout': 'float in [0,1)',
        'model.kernel_sizes': 'non-empty YAML list of positive odd integers',
        'model.architecture_parameters.kernel_sizes': 'derived non-empty list of positive odd integers; must match model',
        'evaluation.statistics.bootstrap_replicates': 'integer >= 1', 'evaluation.statistics.paired_permutation_replicates': 'integer >= 1',
        'evaluation.statistics.seed': 'integer in [0,4294967295]', 'evaluation.statistics.lcb95_percentile': 'finite float in [0,100]',
        'features.rate_prv_min_duration_s': 'finite float > 0', 'features.rate_prv_min_peaks': 'integer >= 1', 'features.time_prv_min_duration_s': 'finite float > 0',
        'features.time_prv_min_coverage': 'finite float in (0,1]', 'features.time_prv_min_intervals': 'integer >= 1', 'features.spectral_prv_min_duration_s': 'finite float > 0',
        'features.spectral_prv_min_coverage': 'finite float in (0,1]', 'features.spectral_prv_min_intervals': 'integer >= 1', 'features.sample_entropy.m': 'integer >= 1',
        'features.sample_entropy.min_intervals': 'integer >= 1', 'features.sample_entropy.r_sd_fraction': 'finite float > 0', 'features.tachogram_fs_hz': 'finite float > 0',
        'features.spectral_bands_hz.vlf': 'two floats [low,high] with 0 <= low < high', 'features.spectral_bands_hz.lf': 'two floats [low,high] with 0 <= low < high',
        'features.spectral_bands_hz.hf': 'two floats [low,high] with 0 <= low < high', 'windows.engineering.length_s': 'finite float > 0',
        'windows.engineering.hop_s': 'finite float > 0', 'windows.engineering.min_valid_fraction': 'finite float in (0,1]',
        'windows.engineering.cap_per_file': 'null or integer >= 1', 'windows.engineering.cap_fraction_per_file': 'null or finite float in (0,1]',
        'windows.raw_dl.length_s': 'finite float > 0', 'windows.raw_dl.hop_s': 'finite float > 0', 'windows.raw_dl.min_valid_fraction': 'finite float in (0,1]',
        'windows.raw_dl.cap_per_file': 'null or integer >= 1', 'windows.raw_dl.cap_fraction_per_file': 'null or finite float in (0,1]',
        'manifest.expected_participant_count': 'integer >= 1; must match the manifest', 'manifest.expected_record_count': 'integer >= 1; must match the manifest',
        'manifest.class_id_order': 'unique integer class IDs; model/training class count must match', 'manifest.channel_order': 'exact ordered acquisition-channel YAML list',
        'signal.channel_order': 'exact ordered acquisition-channel YAML list', 'roles': 'non-empty unique YAML list of B/R*/S*/W* role IDs present in the manifest',
        'training.classifier_role_families': 'non-empty unique subset of [B,R,S,W]', 'splits.split_seeds': 'five unique integers in [0,4294967295]; registry must match',
        'aggregation.hierarchy': 'fixed ordered list [window,file,role,participant]', 'routing.fs_hz': 'fixed finalcase value 400 Hz',
        'routing.window_s': 'fixed finalcase value 8 s', 'routing.hop_s': 'fixed finalcase value 2 s'
    }
    catalogs: dict[str, tuple[str, ...]] = {}
    for family in set(_PARAMETER_MODULE_PATHS.values()):
        values = tuple((str(row['module_id']) for row in list_modules(family)))
        if family == 'normalization':
            catalogs[f'{family}:ppg'] = tuple((value.removeprefix('ppg_') for value in values if value.startswith('ppg_')))
            catalogs[f'{family}:imu'] = tuple((value.removeprefix('imu_') for value in values if value.startswith('imu_')))
        catalogs[family] = values

    def contract(path: str, value: Any, value_type: str) -> tuple[str, str, str]:
        if path == 'config_id':
            return (known_ranges[path], '--config-id ID', 'application_id')
        if path == 'training.device':
            return ('cpu | cuda (numeric-equivalence lock requires cuda)', '--device DEVICE', 'execution')
        if path == 'output.root':
            return ('application-owned: a directory below pipeline_output', '--output-root PATH', 'execution')
        if path in _PARAMETER_MODULE_PATHS:
            family = _PARAMETER_MODULE_PATHS[path]
            key = f"normalization:{('ppg' if path.endswith('raw_ppg') else 'imu')}" if family == 'normalization' else family
            values = catalogs[key]
            return ('enum: ' + ' | '.join(values), f'--module {family}=MODULE_ID (or --set {path}=YAML_VALUE)', 'module')
        if path.startswith('model.architecture_parameters.'):
            return (known_ranges.get(path, f'derived exact value for selected model: {value!r}'), 'select with --module model=MODULE_ID; direct --set must match derived contract',
                    'derived')
        if path.endswith('sha256'):
            return ('null or exactly 64 lowercase hexadecimal characters; referenced bytes must match', f'--set {path}=YAML_VALUE', 'authority')
        if path.endswith('.path') or path in {'manifest.path', 'splits.path'}:
            return ('existing repository-confined path; companion identity/hash fields must match', f'--set {path}=YAML_STRING', 'authority')
        if path in known_ranges:
            return (known_ranges[path], f'--set {path}=YAML_VALUE', 'parameter')
        if value_type == 'boolean':
            return ('boolean: true | false; coupled module invariants still apply', f'--set {path}=true|false', 'switch')
        if value_type == 'mapping':
            return ("YAML mapping accepted only by the active module's exact parameter schema", f"--set {path}='{{key: value}}'", 'module_parameter')
        if value_type == 'null':
            return ('null in the active module; a non-null value is allowed only where its module contract declares it', f'--set {path}=null', 'conditional')
        if value_type == 'list':
            return (f'active-contract ordered YAML list; current exact value: {json.dumps(value, ensure_ascii=False)}', f"--set {path}='[...]'", 'coupled')
        if value_type in {'integer', 'float'}:
            return (f'active-contract exact value {value!r}; no independent range is declared', f'--set {path}=YAML_NUMBER', 'coupled')
        return (f'active-contract enum singleton: {value!r}; change through its owning module/config', f'--set {path}=YAML_STRING', 'coupled')

    rows: list[dict[str, Any]] = []
    for path, value in _leaf_values(config):
        value_type = 'null' if value is None else 'boolean' if isinstance(value, bool) else 'integer' if isinstance(
            value, int) else 'float' if isinstance(value, float) else 'list' if isinstance(value, list) else 'mapping' if isinstance(value, Mapping) else 'string'
        allowed, input_form, control = contract(path, value, value_type)
        rows.append({'path': path, 'type': value_type, 'input': input_form, 'default': value, 'range': allowed, 'control': control})
    return rows

def _catalog_value_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def _unique_catalog_values(values: Iterable[Any]) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = _catalog_value_key(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(copy.deepcopy(value))
    return output

def _changed_leaf_paths(before: Mapping[str, Any], after: Mapping[str, Any]) -> set[str]:
    old = {path: value for path, value in _leaf_values(before)}
    return {path for path, value in _leaf_values(after) if path not in old or _catalog_value_key(old[path]) != _catalog_value_key(value)}

_MODULE_PARAMETER_PREFIXES: Mapping[str, tuple[str, ...]] = {
    'aggregation': ('aggregation.', ), 'artifact': ('artifact.', ), 'class_count_basis': ('training.class_count_basis', 'training.class_weighting'),
    'class_weighting': ('training.class_weighting', 'training.class_weight_beta'), 'denoiser_switch': ('artifact.denoiser_enabled', ), 'dl_resampling': ('signal.dl_resampling.', ),
    'epoch_selection': ('training.epoch_rule', 'training.fixed_epochs', 'training.maximum_inner_epochs',
                        'training.inner_'), 'feature_group': ('features.', ), 'gap_repair': ('signal.gap_repair.', ), 'imu_gravity': ('signal.imu.', ),
    'loss': ('training.loss', 'training.focal_gamma', 'training.label_smoothing'), 'model': ('model.', ), 'motion_detector_switch': ('artifact.motion_detector', ),
    'normalization': ('signal.normalization.', ), 'optimizer': ('training.optimizer', 'training.learning_rate', 'training.weight_decay', 'training.optimizer_parameters.'),
    'peak_detector': ('signal.peak_detector.', ), 'ppg_filter': ('signal.ppg_filter.', ), 'prv_backend': ('features.prv_primary_backend', ), 'quality_mode': ('quality.', ),
    'quality_weight_source': ('aggregation.quality_weight', ), 'representation': ('representation_mode', ), 'sampler': ('training.sampler', 'training.samples_per_epoch',
                                                                                                                        'training.participant_window_quota'),
    'shapeformer_discovery_balance': ('model.', ), 'training_balance': ('training.training_balance', ), 'window_quality_selection': ('quality.window_selection.', )
}

def _module_parameter_paths(family: str, before: Mapping[str, Any], after: Mapping[str, Any]) -> set[str]:
    paths = _changed_leaf_paths(before, after)
    prefixes = _MODULE_PARAMETER_PREFIXES.get(family, ())
    paths.update((path for path, _value in _leaf_values(after) if any((path == prefix or path.startswith(prefix) for prefix in prefixes))))
    return paths

@lru_cache(maxsize=4)
def _all_parameter_rows_cached(pipeline_root: str) -> tuple[dict[str, Any], ...]:
    root = Path(pipeline_root).resolve()
    entries: dict[str, dict[str, Any]] = {}

    def entry_for(row: Mapping[str, Any]) -> dict[str, Any]:
        path = str(row['path'])
        entry = entries.setdefault(path, {'variants': [], 'preset_defaults': {}, 'module_defaults': {}, 'study_defaults': {}})
        entry['variants'].append(dict(row))
        return entry

    preset_configs: dict[str, dict[str, Any]] = {}
    for preset_name in PRESETS:
        config, _ = resolve_configuration(pipeline_root=root, preset=preset_name)
        preset_configs[preset_name] = config
        for row in _parameter_rows_for_config(config):
            entry_for(row)['preset_defaults'][preset_name] = copy.deepcopy(row['default'])
    from ..study import load_study_plan
    from ..study.expand import expand_study
    studies_root = root / 'configs' / 'studies'
    for source in sorted(studies_root.rglob('*.yaml'), key=lambda path: str(path)):
        declared = _load_yaml_mapping(source)
        if declared.get('schema_version') != 'ppg_frailty.study_plan.v2':
            continue
        plan_name = source.relative_to(root).as_posix()
        expansion = expand_study(load_study_plan(source), pipeline_root=root)
        for case in expansion.cases:
            for row in _parameter_rows_for_config(case.config):
                entry = entry_for(row)
                entry['study_defaults'].setdefault(plan_name, []).append(copy.deepcopy(row['default']))
    for descriptor in list_modules():
        family = str(descriptor['family'])
        module_id = str(descriptor['module_id'])
        selection = f'{family}={module_id}'
        for preset_name, base in preset_configs.items():
            try:
                resolved, _ = resolve_configuration(pipeline_root=root, preset=preset_name, modules=(selection, ))
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                continue
            owned_paths = _module_parameter_paths(family, base, resolved)
            for row in _parameter_rows_for_config(resolved):
                entry = entry_for(row)
                if row['path'] in owned_paths:
                    entry['module_defaults'].setdefault(selection, {})[preset_name] = copy.deepcopy(row['default'])
    output: list[dict[str, Any]] = []
    preset_order = {name: index for index, name in enumerate(PRESETS)}
    for path, entry in sorted(entries.items()):
        variants = entry['variants']

        def distinct(field: str) -> list[Any]:
            return _unique_catalog_values((row[field] for row in variants))

        types = distinct('type')
        inputs = distinct('input')
        ranges = distinct('range')
        controls = distinct('control')
        preset_defaults = {name: copy.deepcopy(value) for name, value in sorted(entry['preset_defaults'].items(), key=lambda item: preset_order[item[0]])}
        module_defaults = {
            name: {preset: copy.deepcopy(value)
                   for preset, value in sorted(values.items(), key=lambda item: preset_order[item[0]])}
            for name, values in sorted(entry['module_defaults'].items())
        }
        study_defaults = {name: _unique_catalog_values(values) for name, values in sorted(entry['study_defaults'].items())}
        observed = _unique_catalog_values([
            *preset_defaults.values(), *(value for values in module_defaults.values() for value in values.values()),
            *(value for values in study_defaults.values() for value in values)
        ])
        default_source = 'preset:baseline' if 'baseline' in preset_defaults else f'preset:{next(iter(preset_defaults))}' if preset_defaults else 'catalog:first_observed'
        default = preset_defaults.get('baseline') if 'baseline' in preset_defaults else next(iter(preset_defaults.values())) if preset_defaults else observed[0]
        output.append({
            'path': path, 'type': types[0] if len(types) == 1 else 'context-dependent', 'types': types, 'input': inputs[0] if len(inputs) == 1 else 'context-dependent; see inputs',
            'inputs': inputs, 'default': copy.deepcopy(default), 'default_source': default_source, 'range': ranges[0] if len(ranges) == 1 else 'context-dependent; see ranges',
            'ranges': ranges, 'control': controls[0] if len(controls) == 1 else 'context-dependent', 'controls': controls, 'applicable_presets': list(preset_defaults),
            'defaults_by_preset': preset_defaults, 'applicable_modules': list(module_defaults), 'defaults_by_module': module_defaults,
            'applicable_study_plans': list(study_defaults), 'observed_catalog_defaults': observed
        })
    return tuple(output)

def _all_parameter_rows(pipeline_root: str | Path) -> list[dict[str, Any]]:
    root = str(Path(pipeline_root).resolve())
    return copy.deepcopy(list(_all_parameter_rows_cached(root)))

def parameter_rows(pipeline_root: str | Path, *, source_preset: str = 'baseline') -> list[dict[str, Any]]:
    """Describe one preset or the complete reviewed dotted CLI parameter union."""
    if source_preset == 'all':
        return _all_parameter_rows(pipeline_root)
    if source_preset not in PRESETS:
        raise ValueError(f'unknown source preset: {source_preset}')
    config, _ = resolve_configuration(pipeline_root=pipeline_root, preset=source_preset)
    return _parameter_rows_for_config(config)

def preset_rows(pipeline_root: str | Path) -> list[dict[str, Any]]:
    """Return stable, hash-bound preset metadata for CLI display and audits."""
    root = Path(pipeline_root).resolve()
    rows: list[dict[str, Any]] = []
    for name, preset in PRESETS.items():
        path = (root / preset.relative_path).resolve()
        entry = _preset_entry(root, preset)
        rows.append({
            'name': name, 'default': preset.is_default, 'path': str(path), 'purpose': preset.purpose, 'source_identity': preset.source_identity,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'registry_semantics': str(entry.get('semantics', ''))
        })
    return rows

__all__ = [
    'PRESETS', 'Preset', 'parse_assignment', 'parse_module_assignment', 'parse_yaml_value', 'manual_cli_command', 'manual_cli_tokens', 'parameter_rows', 'preset_rows',
    'resolve_configuration', 'set_dotted', 'unset_dotted'
]
