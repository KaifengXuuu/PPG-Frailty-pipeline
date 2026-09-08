"""V2 严格配置与决策档案合同 / Strict V2 config and decision profiles."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

TOP_LEVEL_KEYS = {
    'schema_version', 'config_id', 'manifest', 'splits', 'output', 'representation_mode', 'roles', 'signal', 'windows',
    'quality', 'routing', 'artifact', 'features', 'model', 'training', 'aggregation', 'evaluation'
}
V2_SCHEMA_VERSION = 'ppg_frailty.pipeline_config.v2'
LEGACY_SCHEMA_VERSION = 'ppg_frailty.pipeline_config.v1'
V2_DECISION_PROFILE_SCHEMA = 'ppg_frailty.v2_decision_profile.v3'
V2_FORMAL_CATALOG_SCHEMA = 'ppg_frailty.formal_experiment_catalog.v2'
V2_FORMAL_ABLATION_PROFILES_SCHEMA = 'ppg_frailty.formal_ablation_profiles.v2'
V2_SPLIT_SEEDS = (42, 10042, 20042, 30042, 40042)
FEATURE_REGISTRY_CONFIG_SCHEMA = 'feature_vector_282_v3'
FEATURE_VECTOR_CONFIG_SCHEMA = 'feature_vector_282_v3'
ENGINEERING_SEQUENCE_CONFIG_SCHEMA = 'engineering_10s_hop2s_thesis_115_v3'
ORDERED_MATRIX_CONFIG_SCHEMA = 'ordered_window_feature_matrix_d146_variable_k_v1'
WINDOW_FEATURE_CONFIG_SCHEMA = 'window_feature_set_d146_v1'
LEGACY_TOP_LEVEL_KEYS = TOP_LEVEL_KEYS - {'routing'}

def _strict_mapping(value: Any, name: str) -> dict[str, Any]:
    """验证对象类型 / Require a string-keyed mapping."""
    if not isinstance(value, Mapping) or not all((isinstance(key, str) for key in value)):
        raise ValueError(f'{name} must be a string-keyed mapping')
    return dict(value)

def _require_exact_keys(mapping: Mapping[str, Any], required: set[str], *, context: str) -> None:
    """拒绝缺字段和未知字段 / Reject missing and unknown fields."""
    observed = set(mapping)
    missing = sorted(required - observed)
    unknown = sorted(observed - required)
    if missing or unknown:
        raise ValueError(f'{context} key mismatch: missing={missing}, unknown={unknown}')

def canonical_json_bytes(value: Any) -> bytes:
    """稳定严格 JSON / Render canonical strict JSON bytes."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')

@dataclass(frozen=True)
class PipelineConfig:
    """规范实验配置 / Canonical experiment configuration."""
    payload: dict[str, Any]
    source_path: str
    sha256: str

    @property
    def config_id(self) -> str:
        """返回配置 ID / Return the configuration identity."""
        return str(self.payload['config_id'])

    @property
    def representation_mode(self) -> str:
        """返回表征模式 / Return the representation mode."""
        return str(self.payload['representation_mode'])

    @property
    def schema_version(self) -> str:
        """返回配置 schema / Return the explicit schema identity."""
        return str(self.payload['schema_version'])

    @property
    def is_legacy(self) -> bool:
        """V1 仅可作来源快照 / Whether this is a provenance-only V1 config."""
        return self.schema_version == LEGACY_SCHEMA_VERSION

    def section(self, name: str) -> dict[str, Any]:
        """读取一个显式 section / Return one explicit section."""
        if name not in TOP_LEVEL_KEYS:
            raise KeyError(name)
        return _strict_mapping(self.payload[name], name)

    def to_dict(self) -> dict[str, Any]:
        """复制可序列化配置 / Copy the serializable payload."""
        return json.loads(json.dumps(self.payload, allow_nan=False))

def _validate_common_payload(data: dict[str, Any]) -> None:
    """验证 V1/V2 共同结构 / Validate structure shared by V1 and V2."""
    expected_keys = LEGACY_TOP_LEVEL_KEYS if data.get('schema_version') == LEGACY_SCHEMA_VERSION else TOP_LEVEL_KEYS
    _require_exact_keys(data, expected_keys, context='config')
    if data['representation_mode'] not in {'raw', 'feature_vector', 'feature_matrix', 'fusion'}:
        raise ValueError('unsupported representation_mode')
    from .data.schema import REGISTERED_ROLES
    roles = data['roles']
    allowed_roles = set(REGISTERED_ROLES)
    if not isinstance(roles, list) or not roles or (not all((role in allowed_roles for role in roles))):
        raise ValueError('roles must be a non-empty registered role list')
    if len(set(roles)) != len(roles):
        raise ValueError('roles must not contain duplicate role IDs')
    for section in expected_keys - {'schema_version', 'config_id', 'representation_mode', 'roles'}:
        _strict_mapping(data[section], section)
    training = _strict_mapping(data['training'], 'training')
    if training.get('epoch_rule') not in {'fixed_epoch', 'inner_grouped_selection'}:
        raise ValueError('training.epoch_rule must be explicit')
    if training.get('outer_labels_visible_to_trainer') is not False:
        raise ValueError('outer labels must be unavailable to the trainer')
    artifact = _strict_mapping(data['artifact'], 'artifact')
    if artifact.get('selection_scope') != 'run_before_evaluation':
        raise ValueError('artifact route must be selected before evaluation')


_QUALITY_DERIVED_FIELDS = frozenset({'mode', 'fit_scope', 'components', 'high_quality_rule', 'failure_action'})

def _diagnostic_quality_runtime_mapping(config: Any) -> dict[str, Any]:
    """Serialize exactly the physical fields consumed by diagnostics-only."""
    config.validate()
    return {
        'cardiac_band_hz': [float(config.cardiac_low_hz), float(config.cardiac_high_hz)],
        'peak_density_bpm_range': [float(config.peak_density_min_bpm),
                                   float(config.peak_density_max_bpm)],
        'ppi_range_s': [float(config.ppi_min_s), float(config.ppi_max_s)],
        'long_gap_max_samples': int(config.long_gap_max_samples),
        'flatline_duration_s': float(config.flatline_duration_s),
        'spectral_analysis_band_hz': [float(config.spectral_analysis_low_hz),
                                      float(config.spectral_analysis_high_hz)],
        'welch_max_nperseg': int(config.welch_max_nperseg),
        'template_min_peaks': int(config.template_min_peaks),
        'template_min_beats': int(config.template_min_beats),
        'template_resample_points': int(config.template_resample_points),
        'ppi_stability_min_intervals': int(config.ppi_stability_min_intervals),
        'component_normalization': {
            'template_half_width_s': float(config.template_half_width_s)
        }
    }

def _validate_v2_quality(data: Mapping[str, Any]) -> None:
    """Validate only parameters consumed by the selected quality module."""
    quality = _strict_mapping(data['quality'], 'quality')
    from .quality.window_selection import WindowSelectionConfig
    from .signal.sqi import SqiConfig, SqiDiagnosticConfig
    from .v2_contract import validate_quality_mode
    mode = validate_quality_mode(str(quality.get('mode')))
    window_selection = WindowSelectionConfig.from_mapping(quality.get('window_selection'))
    common = set(_QUALITY_DERIVED_FIELDS) | {'window_selection', 'long_gap_max_samples', 'flatline_duration_s'}
    artifact = _strict_mapping(data['artifact'], 'artifact')
    denoiser_enabled = bool(artifact.get('denoiser_enabled', str(artifact.get('reducer', 'identity')) != 'identity'))
    if mode == 'route':
        expected = common | set(SqiConfig().to_dict())
        _require_exact_keys(quality, expected, context='quality')
        sqi_quality = dict(quality)
        sqi_quality.pop('window_selection', None)
        SqiConfig.from_quality_mapping(sqi_quality)
    elif mode == 'diagnostics_only':
        diagnostic = SqiDiagnosticConfig.from_resolved({'quality': quality})
        expected = common | set(_diagnostic_quality_runtime_mapping(diagnostic))
        if denoiser_enabled:
            expected |= set(SqiConfig().to_dict())
        _require_exact_keys(quality, expected, context='quality')
        if denoiser_enabled:
            recovery_mapping = dict(quality)
            recovery_mapping.pop('window_selection', None)
            recovery_sqi = SqiConfig.from_quality_mapping(recovery_mapping)
            if recovery_sqi.calibrator != 'fixed_formula_thresholds_v1':
                raise ValueError('diagnostics-only denoiser recovery requires fixed_formula_thresholds_v1')
    else:
        expected = common | set(SqiConfig().to_dict()) if denoiser_enabled else common
        _require_exact_keys(quality, expected, context='quality')
        if denoiser_enabled:
            sqi_quality = dict(quality)
            sqi_quality.pop('window_selection', None)
            recovery_sqi = SqiConfig.from_quality_mapping(sqi_quality)
            if recovery_sqi.calibrator != 'fixed_formula_thresholds_v1':
                raise ValueError('SQI-off denoiser recovery requires fixed_formula_thresholds_v1')
    flatline_duration_s = float(quality['flatline_duration_s'])
    if not math.isfinite(flatline_duration_s) or flatline_duration_s <= 0.0:
        raise ValueError('quality.flatline_duration_s must be positive and finite')
    if window_selection.policy != 'none' and str(data['representation_mode']) not in {'raw', 'fusion'}:
        raise ValueError('quality.window_selection is executable only for raw or fusion representations')
    if window_selection.policy != 'none' and window_selection.application_scope == 'legacy_train_and_aggregation' and (
            str(data['representation_mode']) != 'raw'):
        raise ValueError(
            'quality.window_selection.application_scope=legacy_train_and_aggregation requires raw window-level OOF; file-level fusion cannot consume a held-out window selection view'
        )
    if quality.get('failure_action') != 'fail_closed':
        raise ValueError('quality.failure_action must be fail_closed')
    gap_repair = _strict_mapping(_strict_mapping(data['signal'], 'signal').get('gap_repair'), 'signal.gap_repair')
    if int(quality['long_gap_max_samples']) != int(gap_repair.get('max_gap_samples', -1)):
        raise ValueError(
            'quality.long_gap_max_samples and signal.gap_repair.max_gap_samples describe one fused parameter and must match'
        )

def _materialize_quality_defaults(data: dict[str, Any]) -> None:
    """Persist only the runtime parameters consumed by the selected mode."""
    from .quality.window_selection import WindowSelectionConfig
    from .signal.sqi import SqiConfig, SqiDiagnosticConfig
    from .v2_contract import validate_quality_mode
    declared = _strict_mapping(data['quality'], 'quality')
    if 'supervised_route_ready' in declared:
        raise ValueError(
            'quality.supervised_route_ready is retired; remove it and select the executable module directly with quality.mode'
        )
    if 'long_gap_max_samples' not in declared:
        gap_repair = _strict_mapping(_strict_mapping(data['signal'], 'signal').get('gap_repair'), 'signal.gap_repair')
        declared['long_gap_max_samples'] = gap_repair.get('max_gap_samples', 100)
    declared.setdefault('flatline_duration_s', SqiDiagnosticConfig().flatline_duration_s)
    mode = validate_quality_mode(str(declared.get('mode', 'off')))
    window_selection = WindowSelectionConfig.from_mapping(declared.pop('window_selection', None))
    route_fields = set(SqiConfig().to_dict())
    allowed = set(_QUALITY_DERIVED_FIELDS) | route_fields
    unknown = sorted(set(declared) - allowed)
    if unknown:
        raise ValueError(f'quality contains unknown fields: {unknown}')
    normalization = declared.get('component_normalization')
    if normalization is not None:
        if not isinstance(normalization, Mapping):
            raise ValueError('quality.component_normalization must be a mapping')
        registered_normalization = set(SqiConfig().to_dict()['component_normalization'])
        unknown_normalization = sorted(set(normalization) - registered_normalization)
        if unknown_normalization:
            raise ValueError(f'quality.component_normalization has unknown fields: {unknown_normalization}')
    metadata_defaults = {
        'mode':
        mode,
        'fit_scope':
        'outer_training_participants_only' if mode == 'route' else 'not_applied_' + mode,
        'components': [] if mode == 'off' else [
            'cardiac_concentration', 'autocorrelation_periodicity', 'normalized_spectral_entropy', 'peak_density_bpm',
            'ppi_physiological_fraction', 'ppi_stability', 'red_ir_agreement', 'motion_energy_rms', 'nonflat_scale',
            'source_coverage', 'flatline', 'clipping', 'saturation', 'long_gap'
        ],
        'high_quality_rule':
        'configured_endpoint_thresholds' if mode == 'route' else 'not_applied',
        'failure_action':
        'fail_closed',
        'window_selection':
        window_selection.to_mapping()
    }
    if declared.get('failure_action', 'fail_closed') != 'fail_closed':
        raise ValueError('quality.failure_action must be fail_closed')
    artifact = _strict_mapping(data['artifact'], 'artifact')
    denoiser_enabled = bool(artifact.get('denoiser_enabled', str(artifact.get('reducer', 'identity')) != 'identity'))
    if mode != 'route' and denoiser_enabled:
        declared.setdefault('calibrator', 'fixed_formula_thresholds_v1')
    if mode == 'route':
        runtime = SqiConfig.from_quality_mapping(declared).to_dict()
    elif mode == 'diagnostics_only':
        diagnostic_runtime = _diagnostic_quality_runtime_mapping(
            SqiDiagnosticConfig.from_resolved({'quality': declared}))
        if denoiser_enabled:
            recovery_sqi = SqiConfig.from_quality_mapping(declared)
            if recovery_sqi.calibrator != 'fixed_formula_thresholds_v1':
                raise ValueError('diagnostics-only denoiser recovery requires fixed_formula_thresholds_v1')
            runtime = {**diagnostic_runtime, **recovery_sqi.to_dict()}
            metadata_defaults['high_quality_rule'] = 'direct_diagnostics_only_post_denoise_q_rate_fixed_formula_only'
        else:
            runtime = diagnostic_runtime
    elif denoiser_enabled:
        recovery_sqi = SqiConfig.from_quality_mapping(declared)
        if recovery_sqi.calibrator != 'fixed_formula_thresholds_v1':
            raise ValueError('SQI-off denoiser recovery requires fixed_formula_thresholds_v1')
        runtime = recovery_sqi.to_dict()
        metadata_defaults['high_quality_rule'] = 'direct_sqi_off_post_denoise_q_rate_fixed_formula_only'
        metadata_defaults['components'] = []
    else:
        runtime = {
            'long_gap_max_samples': int(declared['long_gap_max_samples']),
            'flatline_duration_s': float(declared['flatline_duration_s'])
        }
    effective = {**metadata_defaults, **runtime}
    data['quality'] = effective

def _materialize_routing_defaults(data: dict[str, Any]) -> None:
    """Persist the common representation-independent 400 Hz evidence grid."""
    declared = data.get('routing', {})
    if declared is None:
        declared = {}
    if not isinstance(declared, Mapping):
        raise TypeError('routing must be a mapping')
    allowed = {'window_s', 'hop_s', 'fs_hz', 'source_grid'}
    unknown = sorted(set(declared) - allowed)
    if unknown:
        raise ValueError(f'routing contains unknown fields: {unknown}')
    effective = {
        'window_s': float(declared.get('window_s', 8.0)),
        'hop_s': float(declared.get('hop_s', 2.0)),
        'fs_hz': float(declared.get('fs_hz', 400.0)),
        'source_grid': str(declared.get('source_grid', 'canonical_acquisition_grid'))
    }
    if effective != {'window_s': 8.0, 'hop_s': 2.0, 'fs_hz': 400.0, 'source_grid': 'canonical_acquisition_grid'}:
        raise ValueError('formal routing grid is fixed at canonical 400 Hz, 8 s/2 s')
    data['routing'] = effective

def _materialize_dl_resampling_defaults(data: dict[str, Any]) -> None:
    """Resolve the optional DL-only sampling module before config hashing."""
    signal = _strict_mapping(data['signal'], 'signal')
    source_grid = float(signal.get('internal_fs_hz', 400.0))
    raw = signal.get('dl_resampling', {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError('signal.dl_resampling must be a mapping')
    declared = dict(raw)
    enabled = declared.get('enabled', False)
    defaults = {
        'enabled': enabled,
        'target_fs_hz': source_grid if not bool(enabled) else source_grid / 2.0,
        'method': 'polyphase_anti_alias',
        'preserve_feature_grid_hz': source_grid
    }
    signal['dl_resampling'] = {**defaults, **declared}
    data['signal'] = signal

def _materialize_signal_normalization_defaults(data: dict[str, Any]) -> None:
    """Resolve strategy aliases and persist every raw normalization parameter."""
    from .normalization import RawNormalizationConfig
    signal = _strict_mapping(data['signal'], 'signal')
    signal['normalization'] = RawNormalizationConfig.from_mapping(signal.get('normalization')).to_mapping()
    data['signal'] = signal

def _materialize_signal_preprocessing_defaults(data: dict[str, Any]) -> None:
    """Canonicalize executable signal views and IMU profile controls."""
    from .signal.preprocess import materialize_signal_preprocessing_config
    data['signal'] = materialize_signal_preprocessing_config(_strict_mapping(data['signal'], 'signal'))

def _materialize_peak_detector_defaults(data: dict[str, Any]) -> None:
    """Persist detector thresholds so overrides affect effective identity."""
    from .module_registry import resolve_peak_detector_config
    signal = _strict_mapping(data['signal'], 'signal')
    signal['peak_detector'] = resolve_peak_detector_config(signal)
    data['signal'] = signal

def _materialize_artifact_defaults(data: dict[str, Any]) -> None:
    """Persist independent denoiser and frozen motion-inference controls."""
    from .quality.motion_bundle_adapter import resolve_reused_motion_detector_config
    artifact = _strict_mapping(data['artifact'], 'artifact')
    artifact.setdefault('denoiser_enabled', str(artifact.get('reducer', 'identity')) != 'identity')
    motion_defaults = resolve_reused_motion_detector_config().to_mapping(include_enabled=False)
    declared_motion = artifact.get('motion_detector', {})
    if declared_motion is None:
        declared_motion = {}
    if not isinstance(declared_motion, Mapping):
        raise TypeError('artifact.motion_detector must be a mapping')
    artifact['motion_detector'] = {**motion_defaults, **dict(declared_motion)}
    data['artifact'] = artifact

def _materialize_aggregation_defaults(data: dict[str, Any]) -> None:
    """Derive hierarchy plumbing from the one selected aggregation module."""
    aggregation = _strict_mapping(data['aggregation'], 'aggregation')
    allowed = {
        'balance_line', 'hierarchy', 'window_to_file', 'file_to_role', 'role_to_participant', 'missing_role_policy',
        'quality_weighting', 'quality_weight_source', 'quality_weight_levels', 'direct_all_window_participant_mean'
    }
    unknown = sorted(set(aggregation) - allowed)
    if unknown:
        raise ValueError(f'aggregation contains unknown fields: {unknown}')
    line = str(aggregation.get('balance_line', 'line_b_equal_role_families'))
    derived = {
        'line_a_equal_files': {
            'hierarchy': ['window', 'file', 'participant'],
            'file_to_role': 'not_applicable',
            'role_to_participant': 'not_applicable',
            'missing_role_policy': 'not_applicable'
        },
        'line_b_equal_role_families': {
            'hierarchy': ['window', 'file', 'role', 'participant'],
            'file_to_role': 'ordinary_mean',
            'role_to_participant': 'ordinary_mean',
            'missing_role_policy': 'mean_available_roles'
        }
    }
    if line not in derived:
        raise ValueError('aggregation.balance_line must select registered Line A or Line B')
    for field in ('hierarchy', 'missing_role_policy'):
        if field not in aggregation:
            continue
        recognized = {json.dumps(values[field], sort_keys=True) for values in derived.values()}
        if json.dumps(aggregation[field], sort_keys=True) not in recognized:
            raise ValueError(f'aggregation.{field} is not implemented by a registered line')
    recognized_operators = {'ordinary_mean', 'quality_weighted_mean', 'not_applicable'}
    for field in ('window_to_file', 'file_to_role', 'role_to_participant'):
        if aggregation.get(field, 'ordinary_mean') not in recognized_operators:
            raise ValueError(f'unsupported aggregation.{field} operator')
    if aggregation.get('direct_all_window_participant_mean', False) is not False:
        raise ValueError('direct-all-window aggregation is a reporting view, not a selected hierarchy')
    quality_weighting = aggregation.get('quality_weighting', False)
    if not isinstance(quality_weighting, bool):
        raise ValueError('aggregation.quality_weighting must be boolean')
    declared_weight_source = aggregation.get('quality_weight_source')
    registered_weight_sources = {None, 'none', 'route_file_q_rate', 'legacy_window_sqi'}
    if declared_weight_source not in registered_weight_sources:
        raise ValueError('aggregation.quality_weight_source must be none, route_file_q_rate, or legacy_window_sqi')
    weight_source = 'none' if not quality_weighting else 'route_file_q_rate' if declared_weight_source in {
        None, 'none'
    } else str(declared_weight_source)
    quality_levels = {
        'none': [],
        'route_file_q_rate':
        ['file_to_participant'] if line == 'line_a_equal_files' else ['file_to_role', 'role_to_participant'],
        'legacy_window_sqi': ['window_to_file', 'file_to_participant']
        if line == 'line_a_equal_files' else ['window_to_file', 'file_to_role', 'role_to_participant']
    }[weight_source]
    declared_levels = aggregation.get('quality_weight_levels')
    if declared_levels is not None:
        recognized_levels = {
            json.dumps([], sort_keys=True),
            json.dumps(['file_to_participant'], sort_keys=True),
            json.dumps(['file_to_role', 'role_to_participant'], sort_keys=True),
            json.dumps(['window_to_file', 'file_to_participant'], sort_keys=True),
            json.dumps(['window_to_file', 'file_to_role', 'role_to_participant'], sort_keys=True)
        }
        if json.dumps(declared_levels, sort_keys=True) not in recognized_levels:
            raise ValueError(
                'aggregation.quality_weight_levels is derived from the selected source and contains an unsupported value'
            )
    effective = {
        'balance_line': line,
        **derived[line], 'window_to_file':
        'quality_weighted_mean' if weight_source == 'legacy_window_sqi' else 'ordinary_mean',
        'quality_weighting': quality_weighting,
        'quality_weight_source': weight_source,
        'quality_weight_levels': quality_levels,
        'direct_all_window_participant_mean': False
    }
    if line == 'line_b_equal_role_families' and weight_source != 'none':
        effective['file_to_role'] = 'quality_weighted_mean'
        effective['role_to_participant'] = 'quality_weighted_mean'
    data['aggregation'] = effective

def _validate_v2_balance(data: Mapping[str, Any]) -> None:
    """Validate the selected aggregation algorithm and quality-weight source."""
    aggregation = _strict_mapping(data['aggregation'], 'aggregation')
    line = str(aggregation.get('balance_line'))
    hierarchy = {
        'line_a_equal_files': ['window', 'file', 'participant'],
        'line_b_equal_role_families': ['window', 'file', 'role', 'participant']
    }
    if line not in hierarchy or aggregation.get('hierarchy') != hierarchy[line]:
        raise ValueError('aggregation hierarchy does not match its registered balance line')
    source = str(aggregation.get('quality_weight_source'))
    if source not in {'none', 'route_file_q_rate', 'legacy_window_sqi'}:
        raise ValueError('unsupported aggregation.quality_weight_source')
    if bool(aggregation.get('quality_weighting')) != (source != 'none'):
        raise ValueError('aggregation quality weighting and source disagree')
    quality = _strict_mapping(data['quality'], 'quality')
    if source == 'route_file_q_rate' and quality.get('mode') != 'route':
        raise ValueError('route_file_q_rate requires quality.mode=route')
    if source == 'legacy_window_sqi' and (data['representation_mode'] != 'raw' or _strict_mapping(
            quality.get('window_selection'), 'quality.window_selection').get('policy') != 'legacy_per_file_top_fraction'
                                          ):
        raise ValueError('legacy_window_sqi requires raw legacy window selection')

def _validate_v2_signal_normalization(data: Mapping[str, Any]) -> None:
    """Re-run the executable preprocessing and normalization parsers once."""
    from .normalization import RawNormalizationConfig
    from .signal.preprocess import materialize_signal_preprocessing_config
    signal = _strict_mapping(data['signal'], 'signal')
    if signal != materialize_signal_preprocessing_config(signal):
        raise ValueError('signal preprocessing must be fully materialized')
    normalized = RawNormalizationConfig.from_mapping(signal.get('normalization')).to_mapping()
    if signal.get('normalization') != normalized:
        raise ValueError('signal normalization must be fully materialized')

def _materialize_feature_defaults(data: dict[str, Any]) -> None:
    """Resolve every executable feature parameter before hashing."""
    from .features.registry import FEATURE_GROUP_ORDER, canonicalize_feature_groups, ordered_matrix_schema_version, registry_for_groups
    from .signal.prv import PrvConfig
    declared = _strict_mapping(data['features'], 'features')
    metadata_defaults = {
        'prv_primary_backend': 'local_manual',
        'prv_library_comparison_scope': 'fixed_ppi_vectors_only_no_classifier',
        'engineering_sequence_schema': ENGINEERING_SEQUENCE_CONFIG_SCHEMA,
        'technical_metadata_allowed': False,
        'missing_physiology_encoding': 'nan_and_validity_false',
        'file_aggregation': ['mean', 'population_sd'],
        'window_feature_schema': WINDOW_FEATURE_CONFIG_SCHEMA,
        'matrix_length_policy': 'all_complete_windows_variable_k',
        'enabled_groups': list(FEATURE_GROUP_ORDER)
    }
    prv_fields = {
        'rate_prv_min_duration_s', 'rate_prv_min_peaks', 'time_prv_min_duration_s', 'time_prv_min_coverage',
        'time_prv_min_intervals', 'spectral_prv_min_duration_s', 'spectral_prv_min_coverage',
        'spectral_prv_min_intervals', 'tachogram_fs_hz', 'spectral_bands_hz', 'sample_entropy',
        'time_prv_min_accepted_peaks'
    }
    derived_fields = {'registry_id', 'file_vector_schema', 'matrix_schema'}
    allowed = set(metadata_defaults) | prv_fields | derived_fields | {'matrix_k'}
    unknown = sorted(set(declared) - allowed)
    if unknown:
        raise ValueError(f'features contains unknown fields: {unknown}')
    if 'matrix_k' in declared:
        raise ValueError('features.matrix_k is retired; remove the fixed-K value')
    enabled_groups = canonicalize_feature_groups(declared.get('enabled_groups', FEATURE_GROUP_ORDER))
    registry = registry_for_groups(enabled_groups)
    matrix_schema = ordered_matrix_schema_version(None, registry)
    prv = PrvConfig.from_mapping(declared)
    effective = {
        **metadata_defaults,
        **{key: declared.get(key, value)
           for key, value in metadata_defaults.items()},
        **prv.to_dict(), 'enabled_groups': list(enabled_groups),
        'registry_id': registry.schema_version,
        'file_vector_schema': registry.schema_version,
        'matrix_schema': matrix_schema
    }
    data['features'] = effective

def _validate_v2_feature_schemas(data: Mapping[str, Any]) -> None:
    """Validate feature identities derived by the executable registry."""
    from .features.registry import FEATURE_GROUP_ORDER, ordered_matrix_schema_version, registry_for_groups
    from .signal.prv import PrvConfig
    features = _strict_mapping(data['features'], 'features')
    groups = tuple(features.get('enabled_groups', ()))
    registry = registry_for_groups(groups)
    expected = {
        'registry_id': registry.schema_version,
        'file_vector_schema': registry.schema_version,
        'matrix_schema': ordered_matrix_schema_version(None, registry)
    }
    if any((features.get(field) != value for field, value in expected.items())):
        raise ValueError('feature schema identities differ from enabled_groups')
    mode = str(data['representation_mode'])
    if mode == 'raw' and groups != FEATURE_GROUP_ORDER:
        raise ValueError('raw representation does not consume feature-group selection')
    if mode == 'feature_matrix' and groups != ('engineering_summary', ):
        raise ValueError('feature_matrix requires engineering_summary')
    PrvConfig.from_mapping(features).validated()

def _materialize_evaluation_defaults(data: dict[str, Any]) -> None:
    """Resolve configurable reporting budgets while preserving safe defaults."""
    evaluation = _strict_mapping(data['evaluation'], 'evaluation')
    statistics_defaults = {
        'cluster_unit': 'participant_with_all_five_repeat_oof_predictions',
        'bootstrap_replicates': 10000,
        'confidence_interval': 'two_sided_95_percentile',
        'lcb95_percentile': 2.5,
        'lcb95_metrics': ['participant_level_mean_balanced_accuracy', 'participant_level_mean_macro_f1'],
        'paired_permutation_replicates': 100000,
        'seed': 42,
        'paired_exchange_unit': 'participant',
        'multiplicity_correction': 'holm_within_comparison_family',
        'affects_automatic_selection': False
    }
    ranking_defaults = {
        'sort_key': 'participant_level_mean_balanced_accuracy',
        'max_qualified_per_comparison_group': 10,
        'automatic_final_selection': False,
        'manual_multiple_final_versions_allowed': True,
        'preserve_ablation_provenance': True
    }
    evaluation_defaults = {
        'unit':
        'participant',
        'primary_metric':
        'balanced_accuracy',
        'metrics': [
            'balanced_accuracy', 'macro_f1', 'per_class_precision_recall_f1', 'worst_class_recall', 'worst_class_f1',
            'confusion_matrix', 'coverage'
        ],
        'confidence_interval':
        'participant_cluster_bootstrap_two_sided_95',
        'paired_delta_key': ['repeat_index', 'fold_index', 'participant_id'],
        'rank_incomplete_configs':
        False,
        'independent_test_available':
        False,
        'metric_prefix':
        'oof_validation_',
        'calibration_metrics': ['multiclass_brier', 'expected_calibration_error']
    }
    raw_statistics = evaluation.get('statistics', {})
    raw_ranking = evaluation.get('ranking', {})
    if not isinstance(raw_statistics, Mapping) or not isinstance(raw_ranking, Mapping):
        raise TypeError('evaluation statistics and ranking must be mappings')
    evaluation['statistics'] = {**statistics_defaults, **dict(raw_statistics)}
    evaluation['ranking'] = {**ranking_defaults, **dict(raw_ranking)}
    for name, value in evaluation_defaults.items():
        evaluation.setdefault(name, value)
    data['evaluation'] = evaluation

def _validate_evaluation_config(data: Mapping[str, Any]) -> None:
    """Validate only values consumed by statistical/reporting algorithms."""
    evaluation = _strict_mapping(data['evaluation'], 'evaluation')
    if evaluation.get('unit') != 'participant':
        raise ValueError('evaluation.unit must be participant')
    statistics = _strict_mapping(evaluation.get('statistics'), 'evaluation.statistics')
    for field in ('bootstrap_replicates', 'paired_permutation_replicates'):
        value = statistics.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'evaluation.statistics.{field} must be a positive integer')
    seed = statistics.get('seed')
    if isinstance(seed, bool) or not isinstance(seed, int) or (not 0 <= seed <= 4294967295):
        raise ValueError('evaluation.statistics.seed must be in [0,2^32-1]')
    percentile = float(statistics.get('lcb95_percentile', 2.5))
    if not 0.0 < percentile < 50.0:
        raise ValueError('evaluation.statistics.lcb95_percentile must be in (0,50)')
    _strict_mapping(evaluation.get('ranking'), 'evaluation.ranking')

def _validate_output_policy(data: Mapping[str, Any]) -> None:
    """Validate serialization switches; V5 paths are owned by its output contract."""
    output = _strict_mapping(data['output'], 'output')
    for field in ('overwrite_existing', 'strict_json', 'write_parquet', 'write_window_oof', 'write_file_oof',
                  'write_subject_oof', 'write_member_oof'):
        if field in output and (not isinstance(output[field], bool)):
            raise ValueError(f'output.{field} must be boolean')

def _validate_v2_protocol(data: dict[str, Any]) -> None:
    """Validate data identity plus independently configured runtime modules."""
    splits = _strict_mapping(data['splits'], 'splits')
    if splits.get('n_splits') != 5 or splits.get('n_repeats') != 5 or tuple(splits.get(
            'split_seeds', ())) != V2_SPLIT_SEEDS or (splits.get('runtime_recompute') is not False):
        raise ValueError('V2 formal configs require the frozen 5x5 participant registry')
    training = _strict_mapping(data['training'], 'training')
    from .training.trainer import TrainingConfig
    resolved_training = TrainingConfig.from_mapping(training)
    data['training'] = resolved_training.to_mapping()
    from .training.aggregation import canonical_role_family
    selected_role_families = {canonical_role_family(role) for role in data['roles']}
    configured_classifier_families = set(resolved_training.classifier_role_families)
    if not configured_classifier_families <= selected_role_families:
        missing = sorted(configured_classifier_families - selected_role_families)
        raise ValueError(
            f'training.classifier_role_families must be represented by roles; missing selectors for {missing}')
    _validate_output_policy(data)
    _validate_evaluation_config(data)
    _validate_v2_signal_normalization(data)
    _validate_v2_feature_schemas(data)
    from .module_registry import model_factory_contract, resolve_artifact_config, resolve_peak_detector_config, resolve_window_config, validate_model_config, validate_window_profiles_for_representation
    resolved_artifact = resolve_artifact_config(_strict_mapping(data['artifact'], 'artifact'))
    if resolved_artifact['denoiser_enabled']:
        representation_mode = str(data['representation_mode'])
        policy = str(resolved_artifact['degraded_policy'])
        if representation_mode == 'feature_vector':
            if policy != 'denoise_then_extract_rate_features':
                raise ValueError(
                    "feature-vector rate recovery requires degraded_policy='denoise_then_extract_rate_features'")
        elif policy != 'denoise_then_compare_rate_exclude':
            raise ValueError(
                "raw, feature-matrix, and fusion denoiser execution is diagnostic-only and requires degraded_policy='denoise_then_compare_rate_exclude'"
            )
    resolve_peak_detector_config(_strict_mapping(data['signal'], 'signal'))
    data['windows'] = validate_window_profiles_for_representation(
        _strict_mapping(data['windows'], 'windows'), str(data['representation_mode']),
        list(_strict_mapping(data['features'], 'features')['enabled_groups']))
    resolve_window_config(_strict_mapping(data['windows'], 'windows'))
    _validate_v2_dl_resampling(data)
    model = _strict_mapping(data['model'], 'model')
    validate_model_config(model, str(data['representation_mode']))
    model_contract = model_factory_contract(str(model['model_id']))
    resolved_training.validate_for_execution_backend(str(model_contract['execution_backend']))
    _validate_model_window_contract(data)
    _validate_formal_ablation_materialization(data)

def _validate_v2_dl_resampling(data: Mapping[str, Any]) -> None:
    """Bind generic DL resampling to raw/fusion and named presets to raw only."""
    from .signal.resample import validate_dl_resampling_config
    signal = _strict_mapping(data['signal'], 'signal')
    dl = validate_dl_resampling_config(signal.get('dl_resampling'))
    mode = str(data['representation_mode'])
    case_id = dl.get('case_id')
    if case_id is not None and mode != 'raw':
        raise ValueError('named fixed-kernel signal.dl_resampling case_id requires raw representation')
    if bool(dl['enabled']) and mode not in {'raw', 'fusion'}:
        raise ValueError('generic signal.dl_resampling is executable only for raw or fusion representations')
    if not bool(dl['enabled']) and case_id is None:
        return
    raw_window = _strict_mapping(_strict_mapping(data['windows'], 'windows').get('raw_dl'), 'windows.raw_dl')
    if round(float(raw_window['length_s']) * float(dl['target_fs_hz'])) < 2:
        raise ValueError('DL target/window combination must contain at least two samples')

def _validate_model_window_contract(data: Mapping[str, Any]) -> None:
    """Validate temporal input sizes against the selected executable model."""
    mode = str(data['representation_mode'])
    if mode not in {'raw', 'fusion'}:
        return
    from .models.factory import SHAPEFORMER_EXPERIMENTAL_NUMERIC_DEFAULTS as experimental, SHAPEFORMER_LEGACY_EFFECT_SIZE_DEFAULTS as legacy, SHAPEFORMER_REFERENCE_NUMERIC_DEFAULTS as reference, normalize_fusion_signal_encoder_config
    from .signal.resample import validate_dl_resampling_config
    model = _strict_mapping(data['model'], 'model')
    window = _strict_mapping(_strict_mapping(data['windows'], 'windows')['raw_dl'], 'windows.raw_dl')
    dl = validate_dl_resampling_config(_strict_mapping(data['signal'], 'signal').get('dl_resampling'))
    samples = int(round(float(window['length_s']) * float(dl['target_fs_hz'])))
    if samples < 1:
        raise ValueError('raw window and sampling rate yield no samples')
    options: Mapping[str, Any] = model
    model_id = str(model['model_id'])
    nested = model_id == 'FileBagFusion'
    if nested:
        options = normalize_fusion_signal_encoder_config(model.get('signal_encoder'))
        model_id = str(options['canonical_model_name'])
    if model_id in {'CompactCNN1D', 'FileBagFusionCompact'} or (nested and model_id == 'CompactCNN1D'):
        if window['padding'] != 'none_complete_windows_only':
            raise ValueError('CompactCNN input does not support padded windows')
        field = 'pool_sizes' if model_id == 'CompactCNN1D' else 'signal_pool_sizes'
        pools = tuple((int(value) for value in options.get(field, (4, 4))))
        if samples < math.prod(pools):
            raise ValueError(f'{model_id} input is shorter than its pooling chain')
    shape_defaults = {
        'ShapeFormerChannelSpecificOSD': reference,
        'ShapeFormerChannelSpecificScalarDistanceAblation': experimental,
        'ShapeFormerEffectSizeFixedV1': experimental,
        'ShapeFormerLegacyEffectSizePort': legacy
    }
    if model_id not in shape_defaults:
        return
    defaults = shape_defaults[model_id]
    configured_fs = float(options.get('input_fs_hz', defaults['input_fs_hz']))
    if not math.isclose(configured_fs, float(dl['target_fs_hz']), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError('ShapeFormer input_fs_hz differs from effective sampling rate')
    if model_id in {'ShapeFormerChannelSpecificOSD', 'ShapeFormerLegacyEffectSizePort'}:
        sequence = int(options.get('sequence_length_samples', defaults['sequence_length_samples']))
        if sequence != samples or sequence < 3:
            raise ValueError('ShapeFormer sequence length differs from the planned window')
        if int(options.get('local_kernel_width_samples', defaults['local_kernel_width_samples'])) > sequence:
            raise ValueError('ShapeFormer local kernel exceeds its sequence')
        if model_id == 'ShapeFormerLegacyEffectSizePort' and (window['padding'] != 'none_complete_windows_only' or int(
                options.get('shapelet_length_samples', defaults['shapelet_length_samples'])) > sequence):
            raise ValueError('legacy ShapeFormer window/shapelet contract is invalid')
        return
    patch = int(options.get('patch_size_samples', experimental['patch_size_samples']))
    if patch > samples:
        raise ValueError('ShapeFormer patch exceeds its sequence')
    if model_id == 'ShapeFormerChannelSpecificScalarDistanceAblation' and samples < 3:
        raise ValueError('ShapeFormer discovery requires at least three samples')
    if model_id == 'ShapeFormerEffectSizeFixedV1' and int(options.get('shapelet_length_samples', 128)) > samples:
        raise ValueError('ShapeFormer shapelet exceeds its sequence')

def _materialize_v2_defaults(data: dict[str, Any]) -> None:
    """Persist effective module defaults before hashing or runtime dispatch."""
    from .data.schema import REGISTERED_ROLES
    from .module_registry import derived_mask_aware_pooling, derived_model_ensemble_size, derived_model_variant, materialize_model_architecture, normalize_window_config, validate_legacy_ensemble_metadata
    from .training.trainer import TrainingConfig
    roles = data.get('roles')
    if isinstance(roles, list) and roles and (len(roles) == len(set(roles))) and all(
        (role in REGISTERED_ROLES for role in roles)):
        data['roles'] = [role for role in REGISTERED_ROLES if role in roles]
    defaults = TrainingConfig().to_mapping()
    declared = _strict_mapping(data['training'], 'training')
    legacy_weighting_aliases = {
        'outer_train_inverse_frequency': ('inverse_frequency', 'participant'),
        'outer_train_window_inverse_frequency': ('inverse_frequency', 'row')
    }
    declared_weighting = declared.get('class_weighting')
    if declared_weighting in legacy_weighting_aliases:
        canonical_weighting, implied_basis = legacy_weighting_aliases[str(declared_weighting)]
        explicit_basis = declared.get('class_count_basis')
        if explicit_basis is not None and explicit_basis != implied_basis:
            raise ValueError(f'training.class_weighting={declared_weighting} implies class_count_basis={implied_basis}')
        declared['class_weighting'] = canonical_weighting
        declared['class_count_basis'] = implied_basis
    if 'optimizer_parameters' not in declared:
        defaults['optimizer_parameters'] = {}
    data['training'] = {**defaults, **declared}
    model = _strict_mapping(data['model'], 'model')
    validate_legacy_ensemble_metadata(model)
    model['ensemble_size'] = derived_model_ensemble_size(model)
    model['variant'] = derived_model_variant(model)
    mask_aware_pooling = derived_mask_aware_pooling(model)
    if mask_aware_pooling is not None:
        model['mask_aware_pooling'] = mask_aware_pooling
    model.pop('comparison_only', None)
    model.pop('member_seed_roster_id', None)
    model['architecture_parameters'] = materialize_model_architecture(model, str(data['representation_mode']))
    data['model'] = model
    _materialize_signal_preprocessing_defaults(data)
    _materialize_signal_normalization_defaults(data)
    _materialize_peak_detector_defaults(data)
    data['windows'] = normalize_window_config(_strict_mapping(data['windows'], 'windows'))
    _materialize_dl_resampling_defaults(data)
    _materialize_feature_defaults(data)
    _materialize_routing_defaults(data)
    _materialize_quality_defaults(data)
    _materialize_artifact_defaults(data)
    _materialize_aggregation_defaults(data)
    _materialize_evaluation_defaults(data)

def _validate_formal_ablation_materialization(data: Mapping[str, Any]) -> None:
    """Validate optional historical labels without constraining composition."""
    identity = _strict_mapping(data['output'], 'output').get('formal_ablation_materialization')
    if identity is None:
        return
    identity = _strict_mapping(identity, 'formal_ablation_materialization')
    required = {'schema_version', 'family', 'profile_id'}
    missing = required - set(identity)
    if missing:
        raise ValueError(f'formal ablation metadata missing {sorted(missing)}')
    if identity['schema_version'] != 'ppg_frailty.formal_ablation_materialization.v2':
        raise ValueError('unsupported formal ablation metadata schema')
    if not str(identity['family']).strip() or not str(identity['profile_id']).strip():
        raise ValueError('formal ablation family/profile_id must be non-empty')

def validate_config_payload(payload: Mapping[str, Any], *, allow_legacy: bool = False) -> dict[str, Any]:
    """执行 fail-closed 配置验证 / Validate a formal V2 or explicit legacy config."""
    data = _strict_mapping(payload, 'config')
    if data.get('schema_version') == V2_SCHEMA_VERSION:
        data.setdefault('routing', {})
    expected_keys = LEGACY_TOP_LEVEL_KEYS if data.get('schema_version') == LEGACY_SCHEMA_VERSION else TOP_LEVEL_KEYS
    _require_exact_keys(data, expected_keys, context='config')
    if data.get('schema_version') == V2_SCHEMA_VERSION:
        _materialize_v2_defaults(data)
    _validate_common_payload(data)
    schema = data['schema_version']
    if schema == LEGACY_SCHEMA_VERSION:
        if not allow_legacy:
            raise ValueError('legacy V1 config is provenance-only; pass allow_legacy=True explicitly')
        aggregation = _strict_mapping(data['aggregation'], 'aggregation')
        if aggregation.get('hierarchy') != ['window', 'file', 'role', 'participant']:
            raise ValueError('legacy V1 aggregation hierarchy drift')
        return data
    if schema != V2_SCHEMA_VERSION:
        raise ValueError('unsupported schema_version')
    config_id = str(data['config_id'])
    if not config_id.strip() or config_id != config_id.strip() or config_id in {
            '.', '..'
    } or ('\x00' in config_id) or ('/' in config_id) or ('\\' in config_id):
        raise ValueError('V2 config_id must be a non-empty path-safe identifier')
    _validate_v2_quality(data)
    _validate_v2_balance(data)
    _validate_v2_protocol(data)
    return data

def load_config(path: str | Path, *, allow_legacy: bool = False) -> PipelineConfig:
    """加载正式 V2 或显式 legacy V1 / Load formal V2 or explicit legacy V1."""
    source = Path(path)
    source_text = source.read_text(encoding='utf-8')
    try:
        payload = json.loads(source_text)
    except json.JSONDecodeError:
        payload = yaml.safe_load(source_text)
    data = validate_config_payload(_strict_mapping(payload, 'config'), allow_legacy=allow_legacy)
    digest = hashlib.sha256(canonical_json_bytes(data)).hexdigest()
    return PipelineConfig(data, source.as_posix(), digest)

def load_formal_experiment_catalog(path: str | Path) -> dict[str, Any]:
    """Load catalog entries and validate each executable model once."""
    payload = _strict_mapping(yaml.safe_load(Path(path).read_text(encoding='utf-8')), 'formal_experiment_catalog')
    if payload.get('schema_version') != V2_FORMAL_CATALOG_SCHEMA:
        raise ValueError('unsupported formal experiment catalog schema')
    policy = _strict_mapping(payload.get('execution_policy'), 'execution_policy')
    entries_value = payload.get('entries')
    if not isinstance(entries_value, list) or not entries_value:
        raise ValueError('formal catalog entries must be a non-empty list')
    from .module_registry import validate_model_config
    entries, identities, stems = ([], set(), set())
    for raw in entries_value:
        entry = _strict_mapping(raw, 'catalog_entry')
        identity, stem = (str(entry.get('entry_id', '')), str(entry.get('config_stem', '')))
        if not identity or not stem or identity in identities or (stem in stems):
            raise ValueError('catalog entry IDs/config stems must be non-empty and unique')
        identities.add(identity)
        stems.add(stem)
        validate_model_config(_strict_mapping(entry.get('model'), f'{identity}.model'),
                              str(entry.get('representation_mode')))
        entries.append(entry)
    declared = sum((int(policy.get(field, 0))
                    for field in ('candidate_count', 'matched_comparator_count', 'ensemble_comparison_count')))
    if declared and declared != len(entries):
        raise ValueError('formal catalog entry count differs from execution policy')
    payload['entries'] = entries
    payload['catalog_sha256'] = hashlib.sha256(
        canonical_json_bytes({key: value
                              for key, value in payload.items() if key != 'catalog_sha256'})).hexdigest()
    return payload

def load_formal_ablation_profiles(path: str | Path) -> dict[str, Any]:
    """Load historical profile data without using it as an execution gate."""
    payload = _strict_mapping(yaml.safe_load(Path(path).read_text(encoding='utf-8')), 'formal_ablation_profiles')
    if payload.get('schema_version') != V2_FORMAL_ABLATION_PROFILES_SCHEMA:
        raise ValueError('unsupported formal ablation-profile schema')
    families = _strict_mapping(payload.get('families'), 'ablation_profile_families')
    for family, value in families.items():
        section = _strict_mapping(value, f'ablation family {family}')
        rows = section.get('entries', section.get('cases'))
        if not isinstance(rows, list) or not rows:
            raise ValueError(f'ablation family {family} requires entries/cases')
        key = 'profile_id' if 'entries' in section else 'case_id'
        identities = [str(row.get(key, '')) for row in rows if isinstance(row, Mapping)]
        if len(identities) != len(rows) or any(
            (not value for value in identities)) or len(identities) != len(set(identities)):
            raise ValueError(f'ablation family {family} has invalid/duplicate IDs')
    payload['catalog_sha256'] = hashlib.sha256(
        canonical_json_bytes({key: value
                              for key, value in payload.items() if key != 'catalog_sha256'})).hexdigest()
    return payload

def materialize_formal_ablation_config(base_config_path: str | Path, *, family: str, profile_id: str,
                                       output_path: str | Path, profiles_path: str | Path) -> PipelineConfig:
    """Materialize exactly one registered comparison factor; never execute it."""
    base_path = Path(base_config_path).resolve()
    target = Path(output_path).resolve()
    pipeline_root = Path(profiles_path).resolve().parent.parent
    base_relative = base_path.relative_to(pipeline_root).as_posix()
    target.relative_to(pipeline_root)
    if target.exists():
        raise FileExistsError(f'ablation config overwrite forbidden: {target}')
    base = load_config(base_path)
    catalog = load_formal_ablation_profiles(profiles_path)
    if family not in {
            'deep_fixed_epoch', 'direct_filter', 'imu_gravity', 'fixed_kernel_samples', 'aggregation_balance',
            'peak_detector', 'sampler', 'class_count_basis'
    }:
        raise ValueError('unknown formal ablation family')
    payload = base.to_dict()
    from .models import normalize_model_id
    _canonical, machine_id = normalize_model_id(str(payload['model']['model_id']))
    estimator_ids = {'logistic_regression', 'rbf_svm', 'extra_trees'}
    selected: dict[str, Any]
    if family == 'fixed_kernel_samples':
        from .models.time_scale import fixed_kernel_case
        case = fixed_kernel_case(profile_id)
        expected_machine = 'compact_cnn' if case.model_name == 'CompactCNN1D' else 'inception_full'
        if payload['representation_mode'] != 'raw' or machine_id != expected_machine:
            raise ValueError('fixed-kernel case requires the matching raw CompactCNN/Inception config')
        payload['windows']['raw_dl']['length_s'] = float(case.raw_window_seconds)
        resampling = payload['signal']['dl_resampling']
        resampling['case_id'] = case.case_id
        resampling['enabled'] = float(case.dl_fs_hz) != 400.0
        resampling['target_fs_hz'] = float(case.dl_fs_hz)
        if machine_id == 'compact_cnn':
            dilations = [int(case.dilation)] * 3
            payload['model']['dilations'] = dilations
            payload['model']['architecture_parameters']['dilations'] = dilations
        else:
            payload['model']['dilation'] = int(case.dilation)
            payload['model']['architecture_parameters']['dilation'] = int(case.dilation)
        selected = {
            'profile_id': case.case_id,
            'catalog_role': 'reference' if case.case_id.endswith('__reference') else 'ablation'
        }
    else:
        entries = catalog['families'][family]['entries']
        matches = [dict(row) for row in entries if row['profile_id'] == profile_id]
        if len(matches) != 1:
            raise ValueError(f'unknown profile_id for {family}: {profile_id}')
        selected = matches[0]
        if selected.get('auto_run') is not False:
            raise ValueError('formal ablation profiles must never auto-run')
        if family == 'deep_fixed_epoch':
            if machine_id in estimator_ids:
                raise ValueError('epoch profiles are deep-model-only')
            fixed = int(selected['fixed_epochs'])
            payload['training']['fixed_epochs'] = fixed
            payload['training']['epoch_profile'] = {7: 'ablation_7', 10: 'default_10', 15: 'ablation_15'}[fixed]
        elif family == 'direct_filter':
            low = float(selected['low_hz'])
            high = float(selected['high_hz'])
            payload['signal']['ppg_filter']['low_hz'] = low
            payload['signal']['ppg_filter']['high_hz'] = high
            payload['signal']['analysis_view'].pop('direct_source', None)
        elif family == 'imu_gravity':
            method = str(selected['method'])
            payload['signal']['imu']['gravity_method'] = method
            payload['signal']['imu']['comparison_method'] = {
                'calibrated_roll_pitch_ekf': 'profile_a_lowpass_0p3hz',
                'profile_a_lowpass_0p3hz': 'calibrated_roll_pitch_ekf',
                'sensor_filter_only_no_gravity_removal': 'profile_a_lowpass_0p3hz'
            }[method]
        elif family == 'sampler':
            payload['training']['sampler'] = str(selected['sampler'])
        elif family == 'class_count_basis':
            payload['training']['class_weighting'] = str(selected['class_weighting'])
            payload['training']['class_count_basis'] = str(selected['class_count_basis'])
        elif family == 'peak_detector':
            payload['signal']['peak_detector']['detector_id'] = str(selected['detector_id'])
            if 'parameters' in selected:
                payload['signal']['peak_detector']['parameters'] = dict(selected['parameters'])
            else:
                payload['signal']['peak_detector'].pop('parameters', None)
        else:
            is_line_b = selected['profile_id'] == 'role_aware_equal_roles'
            payload['training']['training_balance'] = str(selected['training_balance'])
            payload['aggregation'].update({
                'balance_line': str(selected['balance_line']),
                'hierarchy': list(selected['hierarchy']),
                'window_to_file': 'ordinary_mean',
                'file_to_role': 'ordinary_mean' if is_line_b else 'not_applicable',
                'role_to_participant': 'ordinary_mean' if is_line_b else 'not_applicable',
                'missing_role_policy': 'mean_available_roles' if is_line_b else 'not_applicable',
                'quality_weighting': False,
                'direct_all_window_participant_mean': False
            })
    payload['config_id'] = base.config_id.removesuffix('_v2') + '__' + str(profile_id).replace('-', '_') + '_v2'
    payload['output']['formal_ablation_materialization'] = {
        'schema_version': 'ppg_frailty.formal_ablation_materialization.v2',
        'family': family,
        'profile_id': str(profile_id),
        'catalog_role': str(selected['catalog_role']),
        'base_config_path': base_relative,
        'base_config_sha256': base.sha256,
        'profile_catalog_sha256': catalog['catalog_sha256'],
        'single_factor_only': True,
        'automatic_execution': False,
        'scientific_execution_completed': False
    }
    validated = validate_config_payload(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.tmp')
    try:
        temporary.write_text(yaml.safe_dump(validated, sort_keys=False, allow_unicode=True), encoding='utf-8')
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return load_config(target)

def load_v2_decision_profile(path: str | Path) -> dict[str, Any]:
    """Load machine-auditable reference defaults and deferred evidence."""
    source = Path(path)
    data = _strict_mapping(yaml.safe_load(source.read_text(encoding='utf-8')), 'decision_profile')
    required = {
        'schema_version', 'pipeline_generation', 'profile_id', 'authority', 'confirmed_defaults', 'comparison_profiles',
        'deferred_evidence'
    }
    _require_exact_keys(data, required, context='decision_profile')
    if data['schema_version'] != V2_DECISION_PROFILE_SCHEMA:
        raise ValueError('unsupported V2 decision profile schema')
    if data['pipeline_generation'] != 'final_pipeline_v2':
        raise ValueError('decision profile is not bound to final_pipeline_v2')
    for key in ('authority', 'confirmed_defaults', 'comparison_profiles', 'deferred_evidence'):
        _strict_mapping(data[key], key)
    return data


_BASE_RUNTIME_MODULES = ('numpy', 'scipy', 'sklearn', 'yaml', 'pyarrow')

def required_runtime_modules(config: PipelineConfig) -> tuple[str, ...]:
    """Return import names needed for an ordinary run of this configuration."""
    from .module_registry import model_runtime_dependencies
    modules = list(_BASE_RUNTIME_MODULES)
    modules.extend(model_runtime_dependencies(str(config.section('model')['model_id'])))
    return tuple(modules)

def dependency_availability_report(config: PipelineConfig) -> dict[str, Any]:
    """Report missing runtime imports without pinning versions or import origins."""
    import importlib.util
    modules = required_runtime_modules(config)
    rows = [{'module': module, 'available': importlib.util.find_spec(module) is not None} for module in modules]
    missing = [row['module'] for row in rows if not row['available']]
    return {
        'schema_version': 'ppg_frailty.dependency_availability.v2',
        'pipeline_generation': 'final_pipeline_v2',
        'config_id': config.config_id,
        'ready': not missing,
        'missing_modules': missing,
        'modules': rows,
        'policy': 'ordinary_import_availability_no_version_or_origin_lock'
    }

def require_runtime_dependencies(config: PipelineConfig) -> dict[str, Any]:
    """Raise one actionable error when ordinary runtime imports are missing."""
    report = dependency_availability_report(config)
    if report['missing_modules']:
        raise RuntimeError('missing runtime dependencies: ' + ', '.join(report['missing_modules']))
    return report


__all__ = [
    'LEGACY_SCHEMA_VERSION', 'PipelineConfig', 'TOP_LEVEL_KEYS', 'V2_DECISION_PROFILE_SCHEMA',
    'V2_FORMAL_CATALOG_SCHEMA', 'V2_FORMAL_ABLATION_PROFILES_SCHEMA', 'V2_SCHEMA_VERSION',
    'dependency_availability_report', 'load_config', 'load_formal_experiment_catalog',
    'materialize_formal_ablation_config', 'load_formal_ablation_profiles', 'load_v2_decision_profile',
    'require_runtime_dependencies', 'required_runtime_modules', 'validate_config_payload'
]
