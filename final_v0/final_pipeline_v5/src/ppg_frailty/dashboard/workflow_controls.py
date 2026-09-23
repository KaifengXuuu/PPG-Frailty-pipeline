"""Notebook-style controls backed by the existing numerical configuration.

This module contains no signal processing. Slider bounds are display suggestions,
not execution limits: typed values go to the same validator as the CLI. An edit
may temporarily be incomplete; validation belongs to Analyse/Run, not each drag.
"""
from __future__ import annotations

import copy
import inspect
import math
from pathlib import Path
from typing import Any, Mapping

from ..config import V2_SCHEMA_VERSION, validate_config_payload
from ..module_registry import list_modules, model_factory_contract
from ..v5.configuration import (
    _PARAMETER_MODULE_PATHS, _apply_module, _parameter_rows_for_config,
)

STAGES = ('input', 'ppg', 'imu', 'motion', 'quality', 'denoiser', 'features',
          'representation', 'model', 'aggregation', 'report')


def _root(value: str | Path | None) -> Path:
    return Path(value).resolve() if value is not None else Path(__file__).resolve().parents[3]


def default_configuration(pipeline_root: str | Path | None = None) -> dict[str, Any]:
    """Assemble function/dataclass defaults without loading a preset YAML.

    Paths and dataset identities are the existing project input contract, not
    tunable algorithm defaults. Only the initial representation/model choice
    (raw/CompactCNN1D) is UI policy; their numerical values come from the code.
    """
    from ..data.folds import (M2_SEEDS, M2_SPLIT_FILE_SHA256,
                              M2_SPLIT_PAYLOAD_SHA256, M2_SPLIT_REGISTRY_ID)
    from ..data.manifest import M2_DATASET_VERSION_ID, M2_FILE_MANIFEST_SHA256
    from ..data.schema import CANONICAL_CHANNEL_SCHEMA, CANONICAL_CLASS_NAMES, REGISTERED_ROLES
    from ..models.compact_cnn import CompactCNN1D
    from ..peaks.resolver import CANONICAL_DETECTOR_ID
    from ..training.trainer import TrainingConfig

    training = TrainingConfig().to_mapping()
    model_defaults = {
        name: list(parameter.default) if isinstance(parameter.default, tuple) else parameter.default
        for name, parameter in inspect.signature(CompactCNN1D).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    channels = ['RED', 'IR', 'A_dyn_x', 'A_dyn_y', 'A_dyn_z', 'GX', 'GY', 'GZ']
    payload = {
        'schema_version': V2_SCHEMA_VERSION, 'config_id': 'dashboard_function_defaults',
        'manifest': {
            'path': 'manifests/internal_records_v2.csv', 'manifest_version': 'internal_records_v2',
            'source_dataset_id': M2_DATASET_VERSION_ID, 'source_manifest_sha256': M2_FILE_MANIFEST_SHA256,
            'expected_record_count': 261, 'expected_participant_count': 29,
            'class_id_order': list(CANONICAL_CLASS_NAMES),
            'class_name_order': list(CANONICAL_CLASS_NAMES.values()),
            'channel_order': list(CANONICAL_CHANNEL_SCHEMA), 'allow_qc_excluded_records': False,
        },
        'splits': {
            'path': 'splits/sgkf5_repeated_grouped_5x5_v2.csv', 'registry_id': M2_SPLIT_REGISTRY_ID,
            'source_registry_file_sha256': M2_SPLIT_FILE_SHA256,
            'source_registry_payload_sha256': M2_SPLIT_PAYLOAD_SHA256,
            'n_splits': 5, 'n_repeats': len(M2_SEEDS), 'split_seeds': list(M2_SEEDS),
            'runtime_recompute': False,
        },
        'output': {'root': 'pipeline_output', 'overwrite_existing': False, 'strict_json': True,
                   'write_parquet': True, 'write_window_oof': True, 'write_file_oof': True,
                   'write_subject_oof': True, 'write_member_oof': False},
        'representation_mode': 'raw',
        'roles': [role for role in REGISTERED_ROLES if role[0] in training['classifier_role_families']],
        'signal': {'peak_detector': {'detector_id': CANONICAL_DETECTOR_ID,
                                     'failure_action': 'fail_closed_no_fallback'}},
        'windows': {}, 'quality': {}, 'routing': {},
        'artifact': {'reducer': 'identity', 'reducer_version': 'identity_v1',
                     'selection_scope': 'run_before_evaluation', 'degraded_policy': 'drop',
                     'motion_detector_enabled': False, 'non_identity_output_contract': 'rate_only',
                     'failure_action': 'no_result_no_fallback', 'parameters': {}},
        'features': {},
        'model': {'model_id': 'CompactCNN1D', 'input_channels': len(channels),
                  'input_channels_resolution': 'canonical_frailty_raw_8', 'input_channel_order': channels,
                  'n_classes': len(CANONICAL_CLASS_NAMES), 'seed_policy': 'outer_repeat', **model_defaults},
        'training': training, 'aggregation': {}, 'evaluation': {},
    }
    return validate_config_payload(payload)


def _get(config: Mapping[str, Any], path: str, default: Any = None) -> Any:
    value: Any = config
    try:
        for key in path.split('.'):
            value = value[int(key)] if isinstance(value, (list, tuple)) else value[key]
        return value
    except (KeyError, IndexError, TypeError, ValueError):
        return default


def _set(config: dict[str, Any], path: str, value: Any) -> None:
    """Set a leaf or a numerical vector element without aliasing UI state."""
    parts = path.split('.')
    current: Any = config
    for key in parts[:-1]:
        index = int(key) if isinstance(current, list) else key
        if isinstance(current, dict):
            current.setdefault(index, {})
        if isinstance(current[index], tuple):
            current[index] = list(current[index])
        current = current[index]
    key = int(parts[-1]) if isinstance(current, list) else parts[-1]
    current[key] = copy.deepcopy(value)


def _section(path: str) -> tuple[str, str]:
    owners = (
        ('signal.gap_repair', 'ppg', 'Missing samples'),
        ('signal.ppg_filter', 'ppg', 'PPG bandpass'),
        ('signal.imu', 'imu', 'IMU preprocessing'),
        ('artifact.motion_detector_enabled', 'motion', 'Motion detector'),
        ('artifact.motion_detector', 'motion', 'Motion detector'),
        ('quality.window_selection', 'quality', 'Window selection'),
        ('quality', 'quality', 'Signal quality'),
        ('artifact', 'denoiser', 'Motion denoiser'),
        ('signal.peak_detector', 'features', 'Peak detector'),
        ('features.sample_entropy', 'features', 'Sample entropy'),
        ('features.spectral_bands_hz', 'features', 'PRV frequency bands'),
        ('features', 'features', 'Feature engineering'),
        ('windows.engineering', 'features', 'Feature windows'),
        ('windows.raw_dl', 'representation', 'Raw windows'),
        ('signal.dl_resampling', 'representation', 'DL resampling'),
        ('signal.normalization', 'representation', 'Normalization'),
        ('representation_mode', 'representation', 'Representation'),
        ('model', 'model', 'Model architecture'),
        ('training.optimizer_parameters', 'model', 'Optimizer'),
        ('training.optimizer', 'model', 'Optimizer'),
        ('training', 'model', 'Training'),
        ('aggregation', 'aggregation', 'Aggregation'),
        ('evaluation', 'report', 'Report statistics'),
    )
    return next(((stage, group) for prefix, stage, group in owners
                 if path == prefix or path.startswith(prefix + '.')), ('input', 'Input'))


# Only genuinely executable alternatives appear. Singleton scientific metadata,
# derived hashes, acquisition-grid constants and unsupported toggles stay out.
_CHOICES = {
    'signal.normalization.iqr_fallback': ['standard_deviation_then_finite_one',
                                        'median_absolute_deviation_then_finite_one', 'finite_one'],
    'signal.normalization.standard_ddof': [0, 1],
    'quality.window_selection.application_scope': ['outer_train_only', 'all_partitions',
                                                  'legacy_train_and_aggregation'],
    'quality.calibrator': ['fixed_formula_thresholds_v1', 'outer_train_empirical_quantiles_v1'],
    'artifact.parameters.imu_reference_profile': ['imu_axes6_reference_v2',
                                                  'imu_axes6_plus_derived3_augmentation_ablation_v2'],
    'windows.raw_dl.padding': ['none_complete_windows_only', 'right_zero_pad_short_records',
                              'right_zero_pad_tail', 'right_zero_pad_short_records_and_tail'],
    'windows.engineering.end_alignment': ['left_start_regular_grid', 'include_right_aligned_if_distinct'],
    'windows.raw_dl.end_alignment': ['left_start_regular_grid', 'include_right_aligned_if_distinct'],
    'model.seed_policy': ['outer_repeat', 'fixed_explicit', 'member_roster'],
    'model.pooling': ['mean', 'attention'],
    'artifact.motion_detector.reuse_scope': ['all29_smoke_or_final_only', 'all29_frozen_in_sample_auxiliary',
                                            'matching_outer_fold_or_all29_final'],
}
_FIXED = {
    'signal.internal_fs_hz', 'signal.ppg_filter.notch_enabled', 'signal.imu.required_axes',
    'signal.dl_resampling.preserve_feature_grid_hz', 'model.input_channels', 'model.ensemble_size',
    'model.n_classes', 'model.mask_aware_pooling', 'training.n_classes', 'training.cache_policy',
    'training.refit_on_all_outer_training', 'training.outer_labels_visible_to_trainer',
    'features.technical_metadata_allowed', 'evaluation.rank_incomplete_configs',
    'evaluation.independent_test_available', 'aggregation.direct_all_window_participant_mean',
    'signal.gap_repair.edge_extrapolation', 'output.write_member_oof', 'quality.long_gap_max_samples',
    'model.svm_probability',
}
_OPTIONAL_NUMBERS = {'training.gradient_clip_norm', 'training.samples_per_epoch',
                     'windows.raw_dl.cap_per_file', 'windows.raw_dl.cap_fraction_per_file',
                     'windows.engineering.cap_per_file', 'windows.engineering.cap_fraction_per_file'}
_TEXT = {'manifest.path', 'splits.path', 'training.participant_window_quota',
         'artifact.motion_detector.evidence_path', 'model.svm_gamma', 'model.extra_trees_max_features',
         'model.logistic_solver', 'config_id', 'training.device', 'artifact.motion_detector.device'}


def _choices(path: str, config: Mapping[str, Any]) -> list[Any]:
    if path == 'model.logistic_solver':
        from ..models.factory import _LOGISTIC_SOLVERS
        return sorted(_LOGISTIC_SOLVERS)
    if path in {'model.discovery_balance', 'model.signal_encoder.discovery_balance'}:
        return [row['module_id'] for row in list_modules('shapeformer_discovery_balance')]
    if path == 'model.seed_policy' and not _get(config, 'model.member_seeds'):
        return ['outer_repeat', 'fixed_explicit']
    if path == 'model.variant':
        return ['full', 'small'] if _get(config, 'model.model_id') == 'InceptionTimeMatrix' else []
    if path == 'model.signal_encoder.model_id':
        from ..models.factory import FUSION_SIGNAL_ENCODER_IDS
        return sorted(FUSION_SIGNAL_ENCODER_IDS)
    if path in _CHOICES:
        return list(_CHOICES[path])
    family = _PARAMETER_MODULE_PATHS.get(path)
    if family is None:
        return []
    rows = list_modules(family)
    if family == 'model':
        rows = [row for row in rows if config['representation_mode'] in row['representation_modes']]
    choices = [row['module_id'] for row in rows]
    if family == 'normalization':
        prefix = 'ppg_' if path.endswith('raw_ppg') else 'imu_'
        choices = [str(value).removeprefix(prefix) for value in choices if str(value).startswith(prefix)]
    if family == 'dl_resampling':
        choices = [value for value in choices if value != 'off_identity_source_grid']
    return choices


def _bounds(value: Any, path: str, integer: bool) -> tuple[float | int, float | int, float | int]:
    """Generous slider display extents; a typed value is never clipped here."""
    number = float(value or 0)
    if integer:
        return (min(0, int(number)), max(20, int(abs(number) * 2)), 1)
    if 'dropout' in path or 'fraction' in path or path.endswith(('smoothing', 'class_weight_beta')):
        return (min(0.0, number), max(1.0, number), 0.01)
    scale = 10 ** math.floor(math.log10(abs(number))) if number else 1.0
    return (min(0.0, number * 2), max(scale * 10, abs(number) * 2), scale / 100)


def grouped_parameter_specs(config: Mapping[str, Any], *,
                            pipeline_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Return workflow-ordered controls for the currently selected algorithms.

    Numerical vectors use dotted index paths, e.g. ``model.kernel_sizes.0``;
    selections with varying length (roles/features) remain multi-selects. Fields
    fixed by the numerical implementation do not acquire fictitious switches.
    """
    from ..data.schema import ROLE_FAMILIES, canonicalize_role_family
    from ..artifacts import get_reducer
    from ..artifacts.base import parameters_dict

    config = copy.deepcopy(dict(config))
    # A concise YAML may omit a reducer's parameters. Show its actual dataclass
    # defaults without validating partially edited widget values or running it.
    reducer = get_reducer(config['artifact']['reducer'])
    config['artifact']['parameters'] = {
        **parameters_dict(getattr(reducer, 'config', {})), **config['artifact'].get('parameters', {})}
    model = config['model']
    architecture = model.get('architecture_parameters', {})
    from ..models.factory import _BASELINE_DEFAULTS
    contract = model_factory_contract(model['model_id'])
    for key, value in _BASELINE_DEFAULTS.get(contract['machine_model_id'], {}).items():
        model.setdefault(key, value)
    for key in contract['factory_fields']:
        detail = architecture.get('signal_encoder', {}) if key.startswith('signal_') else architecture
        source_key = key.removeprefix('signal_') if key.startswith('signal_') else key
        if source_key == 'dropout':
            source_key = 'classifier_dropout' if 'classifier_dropout' in detail else 'fusion_dropout'
        if key not in model and source_key in detail:
            model[key] = copy.deepcopy(detail[source_key])
    if model['model_id'] == 'FileBagFusion':
        from ..models.factory import normalize_fusion_signal_encoder_config
        nested = model.setdefault('signal_encoder', normalize_fusion_signal_encoder_config(None))
        nested_contract = model_factory_contract(nested['model_id'])
        detail = architecture.get('signal_encoder', {})
        for key in nested_contract['factory_fields']:
            source = 'classifier_dropout' if key == 'dropout' else key
            if key not in nested and source in detail:
                nested[key] = copy.deepcopy(detail[source])

    result: list[dict[str, Any]] = []
    for row in _parameter_rows_for_config(config):
        path, value = row['path'], row['default']
        if path.startswith('artifact.parameters.') and _get(config, 'artifact.reducer') == 'dwt_a2_legacy':
            continue
        if path in _FIXED or path.startswith(('output.', 'routing.', 'model.architecture_parameters.', 'splits.')) and path != 'splits.path':
            continue
        if row['control'] in {'derived', 'authority'} and path not in _TEXT:
            continue
        if path.startswith(('manifest.', 'signal.analysis_view.', 'evaluation.ranking.')) and path not in _TEXT:
            continue
        if path.startswith('evaluation.') and not path.startswith('evaluation.statistics.'):
            continue
        if path.startswith('evaluation.statistics.') and path.rsplit('.', 1)[-1] not in {
                'bootstrap_replicates', 'paired_permutation_replicates', 'seed', 'lcb95_percentile'}:
            continue
        if path == 'training.focal_gamma' and _get(config, 'training.loss') != 'focal_loss':
            continue
        if path == 'training.class_weight_beta' and _get(config, 'training.class_weighting') != 'effective_number':
            continue
        choices = _choices(path, config)
        if choices and not isinstance(value, (list, tuple, dict)) and value is not None and value not in choices:
            choices.append(value)  # Preserve a valid legacy spelling loaded from YAML.
        stage, group = _section(path)
        multi = {'roles': list(ROLE_FAMILIES), 'training.classifier_role_families': list(ROLE_FAMILIES),
                 'features.enabled_groups': choices}
        if path == 'model.input_channel_order' and config['representation_mode'] == 'raw':
            from ..models.factory import FRAILTY_RAW_CHANNEL_SCHEMA
            multi[path] = list(FRAILTY_RAW_CHANNEL_SCHEMA)
        if path in multi:
            kind, choices = 'multi', multi[path]
        elif len(choices) > 1:
            kind = 'select'
        elif isinstance(value, bool):
            kind = 'boolean'
        elif isinstance(value, (int, float)) or path in _OPTIONAL_NUMBERS or value is None and row['range'].startswith(('integer', 'finite float')):
            kind = 'integer' if isinstance(value, int) or row['range'].startswith('integer') or path.endswith(('cap_per_file', 'samples_per_epoch')) else 'number'
        elif isinstance(value, (tuple, list)) and value and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            result.append({'path': path, 'value': list(value), 'kind': 'list', 'choices': [],
                           'min': None, 'max': None, 'step': 1, 'stage': stage, 'group': group,
                           'range': row['range'], 'nullable': path == 'signal.normalization.clip_after_scale'})
            for index, item in enumerate(value):
                integer = isinstance(item, int)
                low, high, step = _bounds(item, path, integer)
                result.append({'path': f'{path}.{index}', 'value': item, 'kind': 'integer' if integer else 'number',
                               'choices': [], 'min': low, 'max': high, 'step': step,
                               'stage': stage, 'group': group, 'range': row['range'], 'nullable': False})
            continue
        elif path == 'signal.normalization.clip_after_scale':
            kind = 'list'
        elif path in _TEXT:
            kind = 'text'
        else:
            continue
        low, high, step = _bounds(value, path, kind == 'integer') if kind in {'number', 'integer'} else (None, None, 1)
        result.append({'path': path, 'value': copy.deepcopy(value), 'kind': kind, 'choices': choices,
                       'min': low, 'max': high, 'step': step, 'stage': stage, 'group': group,
                       'range': row['range'], 'nullable': path in _OPTIONAL_NUMBERS or value is None})
        if path == 'roles':
            result[-1]['resolved_roles'] = list(value)
            families = {canonicalize_role_family(role) for role in value}
            result[-1]['value'] = [family for family in ROLE_FAMILIES if family in families]
    from .parameter_help import parameter_help
    for spec in result:
        spec['description'] = parameter_help(spec, config)
    return sorted(result, key=lambda row: STAGES.index(row['stage']))


def _model_defaults(config: dict[str, Any], model_id: str, root: Path) -> None:
    """Use the existing catalog's complete module definition, not a run preset."""
    from ..models.factory import (FRAILTY_RAW_CHANNEL_SCHEMA, SHAPEFORMER_REFERENCE_NUMERIC_DEFAULTS,
                                  SHAPEFORMER_EXPERIMENTAL_NUMERIC_DEFAULTS,
                                  SHAPEFORMER_LEGACY_EFFECT_SIZE_DEFAULTS,
                                  SHAPEFORMER_CHANNEL_SPECIFIC_RULE_DEFAULTS, _compact_options, _inception_options)
    from ..module_registry import materialize_model_architecture

    contract = model_factory_contract(model_id)
    if config['representation_mode'] not in contract['representation_modes']:
        raise ValueError(f"{model_id} requires representation {contract['representation_modes']}")
    factory_only = {'FileBagFusion', 'InceptionTimeFull', 'InceptionTimeMatrixFiveMemberEnsemble',
                    'ShapeFormerChannelSpecificScalarDistanceAblation', 'ShapeFormerLegacyEffectSizePort'}
    if model_id not in factory_only:
        _apply_module(config, 'model', model_id, root)
    elif model_id == 'InceptionTimeMatrixFiveMemberEnsemble':
        donor = copy.deepcopy(config)
        donor['representation_mode'] = 'raw'
        _apply_module(donor, 'model', 'InceptionTimeFullFiveMemberEnsemble', root)
        model = donor['model']
        model.update(model_id=model_id, input_channels=146,
                     input_channels_resolution='exact_window_feature_schema_d146')
        for key in ('input_channel_order', 'architecture_parameters', 'variant'):
            model.pop(key, None)
        config['model'] = model
    else:
        config['model'] = {'model_id': model_id, 'input_channels': 8,
                           'input_channels_resolution': 'canonical_frailty_raw_8',
                           'input_channel_order': list(FRAILTY_RAW_CHANNEL_SCHEMA),
                           'n_classes': 3, 'seed_policy': 'outer_repeat'}
        if model_id == 'FileBagFusion':
            config['model'].update(signal_encoder={'model_id': 'compact_cnn', **_compact_options({})},
                                   feature_hidden_dim=32, fusion_hidden_dim=64, pooling='mean', dropout=0.20)
        elif model_id == 'InceptionTimeFull':
            config['model'].update({key: value for key, value in _inception_options({}, 'full').items() if key != 'variant'})
        else:
            defaults = (SHAPEFORMER_LEGACY_EFFECT_SIZE_DEFAULTS if model_id == 'ShapeFormerLegacyEffectSizePort'
                        else {**SHAPEFORMER_REFERENCE_NUMERIC_DEFAULTS, **SHAPEFORMER_EXPERIMENTAL_NUMERIC_DEFAULTS,
                              **SHAPEFORMER_CHANNEL_SPECIFIC_RULE_DEFAULTS})
            config['model'].update({key: value for key, value in defaults.items() if key in contract['factory_fields']})
    model = config['model']
    architecture = materialize_model_architecture(model, config['representation_mode'])
    for key in contract['factory_fields']:
        if key in architecture and key not in model:
            model[key] = copy.deepcopy(architecture[key])
    config['output']['write_member_oof'] = 'member_seeds' in model


def apply_control_values(config: Mapping[str, Any], mapping: Mapping[str, Any], *,
                         pipeline_root: str | Path | None = None, validate: bool = False) -> dict[str, Any]:
    """Apply current widgets, rebuilding only changed algorithm-owned defaults.

    The default accepts incomplete editing state. ``validate=True`` is the
    execution boundary and invokes the unmodified canonical configuration code.
    Changing a selector resets its old algorithm parameters; explicitly supplied
    changed parameters win. Existing YAML values never silently override widgets.
    """
    from ..artifacts import get_reducer
    from ..artifacts.base import parameters_dict
    from ..config import (_materialize_feature_defaults, _materialize_quality_defaults,
                          _materialize_aggregation_defaults)
    from ..data.schema import REGISTERED_ROLES, ROLE_FAMILIES, canonicalize_role_family
    from ..features.registry import FEATURE_GROUP_ORDER
    from ..module_registry import normalize_window_config
    from ..peaks.resolver import resolve_detector_parameters
    from ..quality.motion_bundle_adapter import resolve_reused_motion_detector_config
    from ..signal.preprocess import _materialize_imu_profile
    from ..training.trainer import OPTIMIZER_PARAMETER_DEFAULTS, derived_epoch_profile

    payload = copy.deepcopy(dict(config))
    requested_roles = mapping.get('roles')
    if isinstance(requested_roles, (list, tuple)) and all(role in ROLE_FAMILIES for role in requested_roles):
        # The UI groups roles, but loaded YAML may intentionally select only R1.
        # Keep existing concrete selections; expand only newly enabled families.
        current_roles = set(config['roles'])
        current_families = {canonicalize_role_family(role) for role in current_roles}
        mapping = {**mapping, 'roles': [role for role in REGISTERED_ROLES
                   if canonicalize_role_family(role) in requested_roles
                   and (role in current_roles or canonicalize_role_family(role) not in current_families)]}
    changes = {path: copy.deepcopy(value) for path, value in mapping.items() if value != _get(config, path)}
    if not changes:
        return validate_config_payload(payload) if validate else payload
    root = _root(pipeline_root)
    representation = changes.get('representation_mode')
    if representation is not None:
        payload['representation_mode'] = representation
        groups = ['engineering_summary'] if representation == 'feature_matrix' else list(FEATURE_GROUP_ORDER)
        payload['features']['enabled_groups'] = groups
        if representation not in {'raw', 'fusion'}:
            payload['signal']['dl_resampling']['enabled'] = False
            payload['signal']['dl_resampling']['target_fs_hz'] = payload['signal']['internal_fs_hz']
            payload['quality']['window_selection']['policy'] = 'none'
        selected = changes.get('model.model_id') or {
            'raw': 'CompactCNN1D', 'feature_vector': 'LogisticRegressionL2',
            'feature_matrix': 'InceptionTimeMatrix', 'fusion': 'FileBagFusion',
        }[representation]
        _model_defaults(payload, selected, root)
        payload['windows'] = normalize_window_config(payload['windows'])
    elif 'model.model_id' in changes:
        _model_defaults(payload, str(changes['model.model_id']), root)
    if 'signal.imu.gravity_method' in changes:
        payload['signal']['imu'] = _materialize_imu_profile(
            {'gravity_method': changes['signal.imu.gravity_method']}, fs_hz=400.0)
    if 'signal.peak_detector.detector_id' in changes:
        detector = str(changes['signal.peak_detector.detector_id'])
        payload['signal']['peak_detector']['detector_id'] = detector
        payload['signal']['peak_detector']['parameters'] = resolve_detector_parameters(detector)
    if 'model.signal_encoder.model_id' in changes:
        from ..models.factory import normalize_model_id
        nested = copy.deepcopy(payload)
        nested['representation_mode'] = 'raw'
        canonical, machine_id = normalize_model_id(str(changes['model.signal_encoder.model_id']))
        _model_defaults(nested, canonical, root)
        fields = model_factory_contract(canonical)['factory_fields']
        payload['model']['signal_encoder'] = {'model_id': machine_id,
                                              **{key: value for key, value in nested['model'].items() if key in fields}}
    if 'training.optimizer' in changes:
        optimizer = str(changes['training.optimizer'])
        payload['training']['optimizer_parameters'] = copy.deepcopy(OPTIMIZER_PARAMETER_DEFAULTS[optimizer])
    if changes.get('artifact.denoiser_enabled') is False:
        changes['artifact.reducer'] = 'identity'
    elif changes.get('artifact.denoiser_enabled') is True and _get(payload, 'artifact.reducer') == 'identity':
        changes.setdefault('artifact.reducer', 'nlms_imu_anc')
    if 'artifact.reducer' in changes:
        _apply_module(payload, 'artifact', str(changes['artifact.reducer']), root)
        reducer = get_reducer(str(changes['artifact.reducer']))
        payload['artifact']['parameters'] = parameters_dict(getattr(reducer, 'config', {}))
    if changes.get('artifact.motion_detector_enabled') is False:
        payload['artifact']['motion_detector'] = resolve_reused_motion_detector_config().to_mapping(include_enabled=False)
    for path, value in changes.items():
        _set(payload, path, value)
    # Synchronize existing coupled configuration metadata, never scientific values.
    payload['signal'].get('analysis_view', {}).pop('direct_source', None)
    payload['quality']['long_gap_max_samples'] = payload['signal']['gap_repair']['max_gap_samples']
    if 'aggregation.balance_line' in changes:
        payload['training']['training_balance'] = ('equal_files' if changes['aggregation.balance_line'] == 'line_a_equal_files' else 'equal_role_families')
    if 'training.training_balance' in changes:
        payload['aggregation']['balance_line'] = ('line_a_equal_files' if changes['training.training_balance'] == 'equal_files' else 'line_b_equal_role_families')
    if 'aggregation.quality_weight_source' in changes:
        payload['aggregation']['quality_weighting'] = changes['aggregation.quality_weight_source'] != 'none'
    if 'signal.dl_resampling.enabled' in changes and not changes['signal.dl_resampling.enabled']:
        payload['signal']['dl_resampling']['target_fs_hz'] = payload['signal']['internal_fs_hz']
    if changes.get('quality.window_selection.policy') == 'none':
        from ..quality.window_selection import WindowSelectionConfig
        payload['quality']['window_selection'] = WindowSelectionConfig().to_mapping()
    if 'training.loss' in changes and changes['training.loss'] != 'focal_loss':
        payload['training']['focal_gamma'] = 2.0
    if 'training.class_weighting' in changes and changes['training.class_weighting'] != 'effective_number':
        payload['training']['class_weight_beta'] = 0.999
    if 'training.sampler' in changes:
        if changes['training.sampler'] not in {'balance_line_weighted_v2', 'uniform_replacement'}:
            payload['training']['samples_per_epoch'] = None
        if changes['training.sampler'] not in {'subject_balanced', 'class_subject_balanced'}:
            payload['training']['participant_window_quota'] = 'all'
    if 'model.input_channel_order' in changes:
        from ..models.factory import FRAILTY_RAW_CHANNEL_SCHEMA
        channels = [name for name in FRAILTY_RAW_CHANNEL_SCHEMA if name in changes['model.input_channel_order']]
        payload['model'].update(input_channel_order=channels, input_channels=len(channels),
                                input_channels_resolution=('canonical_frailty_raw_8' if channels == list(FRAILTY_RAW_CHANNEL_SCHEMA)
                                                           else 'explicit_frailty_raw_channel_subset'))
    if 'training.epoch_rule' in changes:
        inner = changes['training.epoch_rule'] == 'inner_grouped_selection'
        payload['training'].update(inner_grouped_folds=2 if inner else 0,
                                   maximum_inner_epochs=10 if inner else 0, inner_patience=3 if inner else 0)
        for key in ('inner_grouped_folds', 'maximum_inner_epochs', 'inner_patience'):
            if 'training.' + key in changes:
                payload['training'][key] = changes['training.' + key]
    if payload['training']['fixed_epochs'] is not None:
        payload['training']['epoch_profile'] = derived_epoch_profile(payload['training']['epoch_rule'], payload['training']['fixed_epochs'])
    if payload['artifact']['denoiser_enabled']:
        payload['artifact']['degraded_policy'] = ('denoise_then_extract_rate_features' if payload['representation_mode'] == 'feature_vector' else 'denoise_then_compare_rate_exclude')
    if any(path.startswith('model.') for path in changes) or representation is not None:
        for key in model_factory_contract(str(payload['model']['model_id']))['derived_provenance_fields']:
            payload['model'].pop(key, None)
    for materializer in (_materialize_feature_defaults, _materialize_quality_defaults, _materialize_aggregation_defaults):
        try:
            materializer(payload)
        except (ValueError, TypeError):
            # Two linked widgets can be temporarily inconsistent while editing.
            # Analyse/Run will report the original canonical validation error.
            if validate:
                raise
    return validate_config_payload(payload) if validate else payload
