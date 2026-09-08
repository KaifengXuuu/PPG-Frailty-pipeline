"""V2 模块注册表与严格配置适配 / V2 module registry and strict adapters."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

@dataclass(frozen=True)
class ModuleDescriptor:
    """一个可审计模块条目 / One auditable module entry."""
    module_id: str
    family: str
    implementation: str
    representation_modes: tuple[str, ...]
    scientific_status: str
    quantitative_suite: str
    notes: str
    runtime_dependencies: tuple[str, ...] = ()

@dataclass(frozen=True)
class ModuleReporterBinding:
    """Presentation-only reporter metadata owned by the module registry."""
    reporter_extension_id: str
    binding_kind: str
    algorithm_summary: str
    references: tuple[str, ...]

def _modules(family: str, suite: str, *rows: tuple[Any, ...]) -> tuple[ModuleDescriptor, ...]:
    """Build a family table; rows are id, implementation, modes, status, notes[, dependencies]."""
    return tuple(
        ModuleDescriptor(row[0], family, row[1], row[2], row[3], suite, row[4], row[5] if len(row) == 6 else ())
        for row in rows
    )


REPRESENTATION_MODULES = _modules(
    'representation', 'integration',
    ('raw', 'ppg_frailty.representations.raw',
     ('raw', ), 'reference', 'Line A window->file->participant; Line B window->file->role_family->participant'),
    ('feature_vector', 'ppg_frailty.representations.feature_vector',
     ('feature_vector', ), 'reference', 'one vector per recording'),
    ('feature_matrix', 'ppg_frailty.representations.feature_matrix', ('feature_matrix', ), 'reference',
     'one chronological 146-by-variable-K window matrix per recording; batch-only padding'),
    ('fusion', 'ppg_frailty.representations.fusion',
     ('fusion', ), 'reference', 'file-level window pooling then one vector concatenation'))
_ALL_REPRESENTATION_MODES = ('raw', 'feature_vector', 'feature_matrix', 'fusion')
NORMALIZATION_MODULES = _modules(
    'normalization', 'normalization',
    ('ppg_per_window_robust', 'ppg_frailty.normalization.RawNormalizationConfig',
     ('raw', 'fusion'), 'runtime_selectable',
     'runtime strategy per_window_robust dispatched by ppg_frailty.representations.raw; median/IQR with configurable fallback and scale constants'
     ),
    ('ppg_per_window_standard_zscore', 'ppg_frailty.normalization.RawNormalizationConfig',
     ('raw', 'fusion'), 'runtime_selectable',
     'runtime strategy per_window_standard_zscore dispatched by ppg_frailty.representations.raw with configurable ddof and epsilon'
     ),
    ('ppg_none', 'ppg_frailty.normalization.RawNormalizationConfig', ('raw', 'fusion'), 'runtime_selectable_identity',
     'runtime PPG strategy none dispatched by ppg_frailty.representations.raw; no optical scaling'),
    ('imu_outer_train_robust', 'ppg_frailty.normalization.RawNormalizationConfig',
     ('raw', 'fusion'), 'runtime_selectable_fold_local',
     'runtime strategy outer_train_robust dispatched by ppg_frailty.representations.imu_transform; statistics fit only on outer-train rows'
     ),
    ('imu_outer_train_mean_std', 'ppg_frailty.normalization.RawNormalizationConfig',
     ('raw', 'fusion'), 'runtime_selectable_fold_local',
     'runtime strategy outer_train_mean_std dispatched by ppg_frailty.representations.imu_transform; statistics fit only on outer-train rows'
     ), ('imu_none', 'ppg_frailty.normalization.RawNormalizationConfig',
         ('raw', 'fusion'), 'runtime_selectable_identity',
         'runtime IMU strategy none dispatched by ppg_frailty.representations.imu_transform; no fold-local scaling'))
PPG_FILTER_MODULES = _modules('ppg_filter', 'signal', (
    'butterworth_sos', 'ppg_frailty.signal.preprocess.materialize_signal_preprocessing_config',
    _ALL_REPRESENTATION_MODES, 'runtime_parameterized_filter',
    'executable PPG filter family; order, passband, phase, notch and short-signal policy are validated config parameters rather than separate profile IDs'
))
GAP_REPAIR_MODULES = _modules('gap_repair', 'signal', (
    'linear_inside_only', 'ppg_frailty.signal.preprocess.materialize_signal_preprocessing_config',
    _ALL_REPRESENTATION_MODES, 'runtime_parameterized_internal_gap_repair',
    'executable finite internal-gap interpolation with configurable max_gap_samples; edge extrapolation remains an explicit data-integrity boundary'
))
IMU_GRAVITY_MODULES = tuple(
    (ModuleDescriptor(
        module_id, 'imu_gravity', 'ppg_frailty.signal.preprocess.materialize_signal_preprocessing_config',
        _ALL_REPRESENTATION_MODES, {
            'profile_a_lowpass_0p3hz': 'runtime_selectable_reference',
            'calibrated_roll_pitch_ekf': 'runtime_selectable_ablation',
            'sensor_filter_only_no_gravity_removal': 'runtime_selectable_ablation',
            'quaternion_error_state_ekf': 'runtime_selectable_legacy_ablation',
            'low_pass_0p3hz': 'runtime_selectable_legacy_parallel'
        }[module_id], 'signal',
        'executable gravity-separation profile with profile-specific numerical parameters and no silent fallback')
     for module_id in ('calibrated_roll_pitch_ekf', 'sensor_filter_only_no_gravity_removal', 'profile_a_lowpass_0p3hz',
                       'quaternion_error_state_ekf', 'low_pass_0p3hz')))
DL_RESAMPLING_MODULES = _modules('dl_resampling', 'signal', (
    'off_identity_source_grid', 'ppg_frailty.signal.resample.validate_dl_resampling_config', _ALL_REPRESENTATION_MODES,
    'runtime_selectable_identity',
    'disabled branch preserves the canonical source grid and does not create a second DL tensor'
), ('polyphase_anti_alias', 'ppg_frailty.signal.resample.prepare_configured_dl_input',
    ('raw', 'fusion'), 'runtime_parameterized_optional_dl_view',
    'enabled branch accepts any validated positive target no higher than the source grid; named fixed-kernel rates are catalog presets only'
    ))
WINDOW_PROFILE_MODULES = _modules('window_profile', 'signal', (
    'engineering', 'ppg_frailty.module_registry.resolve_window_config',
    ('feature_vector', 'feature_matrix', 'fusion'), 'runtime_parameterized_shared_planner_profile',
    'engineering window length, hop, alignment, padding, cap and valid-fraction controls share the registered planner'
), ('raw_dl', 'ppg_frailty.module_registry.resolve_window_config',
    ('raw', 'fusion'), 'runtime_parameterized_shared_planner_profile',
    'raw/fusion window length, hop, alignment, mask-aware padding, cap and valid-fraction controls share the registered planner'
    ))
OPTIMIZER_MODULES = tuple((ModuleDescriptor(
    module_id, 'optimizer', 'ppg_frailty.training.trainer.resolve_optimizer_parameters',
    ('raw', 'feature_matrix', 'fusion'), 'runtime_selectable_torch_optimizer', 'training',
    f'executable {module_id} strategy; optimizer-specific controls are materialized before UnifiedTrainer constructs torch.optim',
    ('torch', )) for module_id in ('adam', 'adamw', 'sgd', 'rmsprop')))
SAMPLER_MODULES = tuple((ModuleDescriptor(
    module_id, 'sampler', 'ppg_frailty.training.trainer.configured_row_sampling_weights',
    ('raw', 'feature_matrix', 'fusion') if module_id == 'uniform_replacement' else _ALL_REPRESENTATION_MODES,
    'runtime_selectable_torch_replacement_sampler' if module_id == 'uniform_replacement' else
    'runtime_selectable_reference' if module_id == 'exhaustive_shuffle_without_replacement' else
    'runtime_selectable_ablation' if module_id == 'balance_line_weighted_v2' else 'runtime_selectable', 'training',
    'executable replacement draw strategy for torch data loaders; estimators fail fast because uniform sample weights cannot encode replacement draws'
    if module_id == 'uniform_replacement' else
    f'executable {module_id} row-distribution strategy shared by deep loaders and estimator sample weights',
    ('torch', ) if module_id == 'uniform_replacement' else ())
                         for module_id in ('balance_line_weighted_v2', 'uniform_replacement',
                                           'exhaustive_shuffle_without_replacement', 'subject_balanced',
                                           'class_subject_balanced')))
LOSS_MODULES = tuple((ModuleDescriptor(
    module_id, 'loss', 'ppg_frailty.training.trainer.TrainingClassificationLoss', ('raw', 'feature_matrix', 'fusion'),
    'runtime_selectable_torch_loss', 'training',
    f'executable {module_id} classification-loss strategy; weighted_ce is an input alias of cross_entropy, not a parallel module',
    ('torch', )) for module_id in ('cross_entropy', 'balanced_softmax', 'focal_loss')))
CLASS_WEIGHTING_MODULES = tuple(
    (ModuleDescriptor(module_id, 'class_weighting', 'ppg_frailty.training.trainer.configured_class_weight_vector',
                      _ALL_REPRESENTATION_MODES, 'runtime_selectable_single_weighting_entry', 'training',
                      f'executable {module_id} strategy; model.class_weight is not a second weighting capability')
     for module_id in ('inverse_frequency', 'effective_number', 'none')))
CLASS_COUNT_BASIS_MODULES = tuple((ModuleDescriptor(
    module_id, 'class_count_basis', 'ppg_frailty.training.trainer.outer_train_class_counts', _ALL_REPRESENTATION_MODES,
    'runtime_selectable_reference' if module_id == 'row' else 'runtime_selectable_ablation', 'training',
    f'executable {module_id} count basis shared by inverse-frequency, effective-number, and balanced-softmax corrections'
) for module_id in ('participant', 'row')))
TRAINING_BALANCE_MODULES = tuple((ModuleDescriptor(
    module_id, 'training_balance', 'ppg_frailty.training.aggregation.aggregation_rule_for_training_balance',
    _ALL_REPRESENTATION_MODES, 'runtime_selectable_independent_of_reporting_aggregation', 'training',
    f'executable {module_id} hierarchy used by configured sampling and train/inner participant balanced-accuracy; reporting Line A/B is selected independently'
) for module_id in ('equal_files', 'equal_role_families')))
EPOCH_SELECTION_MODULES = tuple(
    (ModuleDescriptor(module_id, 'epoch_selection', 'ppg_frailty.training.trainer.UnifiedTrainer',
                      ('raw', 'feature_matrix', 'fusion'), 'runtime_selectable_deep_epoch_strategy', 'training',
                      f'executable {module_id} branch in UnifiedTrainer; inner selection remains outer-train-only',
                      ('torch', )) for module_id in ('fixed_epoch', 'inner_grouped_selection')))
QUALITY_MODE_MODULES = tuple(
    (ModuleDescriptor(module_id, 'quality_mode', 'ppg_frailty.quality.routing.run_quality_mode',
                      _ALL_REPRESENTATION_MODES, 'runtime_selectable_no_gate', 'quality',
                      f'executable quality mode {module_id}; no readiness or authorization gate')
     for module_id in ('off', 'diagnostics_only', 'route')))
MOTION_DETECTOR_SWITCH_MODULES = tuple(
    (ModuleDescriptor(module_id, 'motion_detector_switch',
                      'ppg_frailty.quality.routing.route_module_switches_from_config', _ALL_REPRESENTATION_MODES,
                      'runtime_selectable_independent_switch', 'quality',
                      'artifact.motion_detector_enabled is independent of SQI and denoiser selection')
     for module_id in ('disabled', 'enabled')))
DENOISER_SWITCH_MODULES = (
    ModuleDescriptor('disabled', 'denoiser_switch', 'ppg_frailty.quality.routing.route_module_switches_from_config',
                     _ALL_REPRESENTATION_MODES, 'runtime_selectable_independent_switch', 'quality',
                     'artifact.denoiser_enabled is independent of SQI and motion detection'),
    ModuleDescriptor(
        'enabled', 'denoiser_switch', 'ppg_frailty.quality.routing.route_module_switches_from_config',
        _ALL_REPRESENTATION_MODES, 'runtime_selectable_independent_switch', 'quality',
        'one configured reducer runs only after Unfit; recovered Acceptable evidence enters feature-vector pulse features or remains HR-only diagnostic evidence for other representations'
    ))
WINDOW_QUALITY_SELECTION_MODULES = (
    ModuleDescriptor('none', 'window_quality_selection', 'ppg_frailty.quality.window_selection.select_raw_windows',
                     ('raw', 'fusion'), 'runtime_selectable_noop', 'quality',
                     'retains every raw window without computing SQI'),
    ModuleDescriptor(
        'legacy_per_file_top_fraction', 'window_quality_selection',
        'ppg_frailty.quality.window_selection.select_raw_windows', ('raw', 'fusion'),
        'runtime_selectable_label_free_per_file_rank', 'quality',
        'computes legacy_cardiac_motion_window_sqi_v1 independently per file and retains ceil(n*keep_fraction) windows; application_scope independently selects train-only, all-partitions, or legacy-style per-file heldout selection inside the V2 hierarchy'
    ))
SHAPEFORMER_DISCOVERY_BALANCE_MODULES = tuple((ModuleDescriptor(
    module_id, 'shapeformer_discovery_balance', 'ppg_frailty.models.pisd_port.discover_pisd_shapelets', ('raw', ),
    'runtime_selectable_outer_train_discovery_sampler', 'models', 'class/participant/file/window hierarchical discovery'
    if module_id == 'participant_file_balanced' else 'legacy class-balanced random-window discovery', ('torch', ))
                                               for module_id in ('participant_file_balanced', 'class_window_balanced')))
AGGREGATION_MODULES = tuple(
    (ModuleDescriptor(module_id, 'aggregation', 'ppg_frailty.training.aggregation.aggregate_hierarchy',
                      _ALL_REPRESENTATION_MODES, 'runtime_selectable_parallel_reporting_line', 'aggregation',
                      f'executable {module_id} hierarchy ending in participant-balanced output')
     for module_id in ('line_a_equal_files', 'line_b_equal_role_families')))
QUALITY_WEIGHT_SOURCE_MODULES = (
    ModuleDescriptor('none', 'quality_weight_source', 'ppg_frailty.training.aggregation.aggregate_hierarchy',
                     _ALL_REPRESENTATION_MODES, 'runtime_selectable_unweighted_hierarchy', 'aggregation',
                     'ordinary means at every selected hierarchy edge'),
    ModuleDescriptor('route_file_q_rate', 'quality_weight_source',
                     'ppg_frailty.training.aggregation.aggregate_hierarchy', _ALL_REPRESENTATION_MODES,
                     'runtime_selectable_file_level_endpoint_weight', 'aggregation',
                     'consumes route Q_rate from file-to-role/participant; raw window-to-file remains ordinary mean'),
    ModuleDescriptor(
        'legacy_window_sqi', 'quality_weight_source', 'ppg_frailty.training.aggregation.aggregate_hierarchy', ('raw', ),
        'runtime_selectable_row_aligned_legacy_weight', 'aggregation',
        'consumes migrated per-window cardiac/motion scores at window-to-file and propagates file summaries upward'))
FEATURE_GROUP_MODULES = tuple((ModuleDescriptor(
    module_id, 'feature_group', 'ppg_frailty.features.registry.registry_for_groups',
    ('feature_vector', 'feature_matrix', 'fusion'), 'runtime_selectable_composable_group', 'features',
    'selected through features.enabled_groups; the same content-addressed registry drives extraction, fold transforms, fusion tensors, ordered matrix context channels, validation, experiments, and final refit'
) for module_id in ('ppi_basic_rate', 'hrv_time_domain', 'hrv_spectral', 'hrv_nonlinear', 'morphology', 'dual_optical',
                    'engineering_summary')))
ARTIFACT_MODULES = _modules(
    'artifact', 'artifacts',
    ('identity', 'ppg_frailty.artifact.identity.IdentityReducer', ('raw', 'feature_vector', 'feature_matrix', 'fusion'),
     'direct_control', 'exact no-op; morphology remains eligible', ()),
    ('nlms_imu_anc', 'ppg_frailty.artifact.nlms.NlmsReducer',
     ('raw', 'feature_vector', 'feature_matrix', 'fusion'), 'comparison_rate_only',
     'ANC assumption may remove physiological response; rate-only output enters vectors or an explicit HR-only diagnostic exclusion route',
     ()),
    ('ssa_decomposition', 'ppg_frailty.artifact.decomposition.SsaReducer',
     ('raw', 'feature_vector', 'feature_matrix',
      'fusion'), 'comparison_rate_only',
     'non-stationary decomposition comparator; rate-only output enters vectors or an explicit HR-only diagnostic exclusion route',
     ()),
    ('spectral_mask', 'ppg_frailty.artifact.spectral.SpectralMaskReducer',
     ('raw', 'feature_vector', 'feature_matrix', 'fusion'), 'comparison_rate_only',
     'formal STFT plus IMU soft mask; rate-only output enters vectors or an explicit HR-only diagnostic exclusion route',
     ()),
    ('pca_bss', 'ppg_frailty.artifact.bss.PcaBssReducer',
     ('raw', 'feature_vector', 'feature_matrix', 'fusion'), 'preferred_rate_recovery',
     'preferred two-wavelength PCA BSS reducer; rate-only output enters vectors or the Stage05 CNN HR-only diagnostic exclusion route',
     ()),
    ('fastica_bss', 'ppg_frailty.artifact.bss.FastIcaBssReducer',
     ('raw', 'feature_vector', 'feature_matrix', 'fusion'), 'parallel_rate_recovery_ablation',
     'parallel FastICA BSS ablation; rate-only output enters vectors or the Stage05 CNN HR-only diagnostic exclusion route',
     ()),
    ('nmf_bss', 'ppg_frailty.artifact.bss.NmfBssReducer',
     ('raw', 'feature_vector', 'feature_matrix', 'fusion'), 'comparison_rate_only',
     'two-wavelength spectral BSS comparator; rate-only output enters vectors or an explicit HR-only diagnostic exclusion route',
     ()),
    ('emd_sifting_rate_only', 'ppg_frailty.artifact.legacy.EmdSiftingRateOnlyReducer',
     ('raw', 'feature_vector', 'feature_matrix', 'fusion'), 'comparison_rate_only',
     'named EMD sifting ablation; rate-only output enters vectors or an explicit HR-only diagnostic exclusion route',
     ()),
    ('ceemd_lite_nlms_legacy', 'ppg_frailty.artifact.legacy.CeemdLiteNlmsLegacyReducer',
     ('raw', 'feature_vector', 'feature_matrix', 'fusion'), 'comparison_rate_only',
     'named CEEMD-lite plus NLMS legacy ablation; rate-only output enters vectors or an explicit HR-only diagnostic exclusion route',
     ()),
    ('dwt_a2_legacy', 'ppg_frailty.artifact.legacy.DwtA2LegacyReducer',
     ('raw', 'feature_vector', 'feature_matrix', 'fusion'), 'comparison_rate_only',
     'named DWT A2 legacy ablation; rate-only output enters vectors or an explicit HR-only diagnostic exclusion route',
     ()))
PRV_BACKEND_MODULES = _modules(
    'prv_backend', 'prv_backend_output', ('local', 'ppg_frailty.features.prv_backend_compare.evaluate_prv_backend',
                                          ('feature_vector', 'feature_matrix', 'fusion'), 'formal_primary',
                                          'local manual PRV remains the only classifier-pipeline backend', ()),
    ('aura_hrv_analysis', 'ppg_frailty.features.prv_backend_compare.evaluate_prv_backend',
     (), 'function_comparison_only', 'fixed PPI vectors only; never enters classifier training', ()),
    ('rhenan_hrv', 'ppg_frailty.features.prv_backend_compare.evaluate_prv_backend',
     (), 'legacy_function_comparison_only', 'separate legacy requirements; never enters classifier training', ()))
PEAK_DETECTOR_MODULES = _modules(
    'peak_detector', 'signal',
    ('aboy_project_v1', 'ppg_frailty.peaks.aboy_project.detect_pulses_per_wavelength_aboy_project',
     ('raw', 'feature_vector', 'feature_matrix', 'fusion'), 'implicit_fallback_explicit_selection_only',
     'historical project detector retained for explicit recovery/audit selection only; never invoked automatically and excluded from Stage-ablation-01',
     ()),
    ('aboy_project_v2', 'ppg_frailty.peaks.aboy_project_v2.detect_pulses_per_wavelength_aboy_project_v2',
     ('raw', 'feature_vector', 'feature_matrix', 'fusion'), 'explicit_seven_step_ablation',
     'authoritative project seven-step contract: owned 0.2-Hz high-pass, complete non-overlapping 10-s blocks, per-block dual-polarity selection, retained-Pd HRI, ratio peak removal before physiological/MAD cleaning',
     ()),
    ('dual_polarity_prominence_v1_ablation', 'ppg_frailty.signal.peaks._detect_pulses_dual_polarity_ablation',
     ('raw', 'feature_vector', 'feature_matrix',
      'fusion'), 'explicit_legacy_ablation_only',
     'numerically preserved whole-record fixed distance/prominence detector; configurable min_observation_sec=8.0 and min_peaks=5 defaults; never a fallback',
     ()),
    ('msptdfast_v2_3_python_port', 'ppg_frailty.peaks.msptdfast_v2.detect_msptdfast_v2',
     ('raw', 'feature_vector', 'feature_matrix', 'fusion'), 'canonical_default_and_stage_ablation_reference',
     'default registered implementation shared by Stage-ablation-01 and the ordinary pipeline; bound to ppg-beats v2.3 source SHA-256; not bitwise MATLAB parity',
     ()))
MOTION_EVIDENCE_MODULES = _modules('motion_evidence', 'quality', (
    'reused_frailty29_all29_bundle', 'ppg_frailty.quality.motion_bundle_adapter.load_reused_motion_detector',
    ('raw', 'feature_vector', 'feature_matrix', 'fusion'), 'runtime_selectable_inference_only',
    'frozen Stage5 matching-fold model/threshold for formal OOF; all-29 is in-sample auxiliary evidence for final/smoke only; native-window routing, no refit or recalibration',
    ())) + _modules('motion_evidence', 'motion_contract',
                    ('sqi_only', 'ppg_frailty.quality.motion.resolve_motion_option',
                     (), 'external_audit_control_not_classifier_runtime',
                     'V2-010 external evidence control; core quality behavior is configured through quality.mode', ()),
                    ('sqi_plus_motion_override', 'ppg_frailty.quality.motion.resolve_motion_option',
                     (), 'external_ptt_evidence_protocol_not_classifier_runtime',
                     'independent motion/PTT audit API only; never a core classifier-pipeline selector', ()),
                    ('historical_light_cnn_backup', 'ppg_frailty.quality.motion.HISTORICAL_LIGHT_CNN_EVIDENCE',
                     (), 'historical_frozen_evidence_not_classifier_runtime',
                     'archived SIM evidence only; never a V2 classifier or PTT result', ()))
COMPARISON_PROFILE_MODULES = _modules(
    'comparison_profile', 'formal_catalog',
    ('line_b_equal_role_families', 'ppg_frailty.v2_contract.resolve_balance_line',
     ('raw', 'feature_vector', 'feature_matrix',
      'fusion'), 'registered_not_run', 'matched training and aggregation Line B; never selected automatically',
     ()), ('epoch_7_ablation', 'ppg_frailty.training.trainer.TrainingConfig',
           ('raw', 'feature_matrix', 'fusion'), 'registered_not_run', 'single factor from default fixed epoch 10', ()),
    ('epoch_15_ablation', 'ppg_frailty.training.trainer.TrainingConfig',
     ('raw', 'feature_matrix', 'fusion'), 'registered_not_run', 'single factor from default fixed epoch 10', ()),
    ('direct_filter_0p5_to_5hz_ablation', 'ppg_frailty.signal.ppg_preprocess.build_signal_views',
     ('raw', 'feature_vector', 'feature_matrix',
      'fusion'), 'registered_not_run', 'only named direct-filter ablation from 0.2 to 8 Hz reference', ()),
    ('imu_lpf_0p3hz_ablation', 'ppg_frailty.signal.imu_preprocess.preprocess_imu',
     ('raw', 'feature_vector', 'feature_matrix',
      'fusion'), 'registered_not_run', 'independent gravity-separation comparison; never an EKF fallback', ())
) + _modules('comparison_profile', 'models', (
    'fixed_kernel_samples_resampling_ablation',
    'ppg_frailty.models.time_scale.build_fixed_kernel_resampling_cases', ('raw',
                                                                          ), 'registered_not_run',
    'V2-019: CompactCNN/Inception only; 100/160/200 Hz keep kernel sample counts fixed and are not physical-time matched',
    ()
), ('fixed_kernel_samples_context_10s_400hz_ablation', 'ppg_frailty.models.time_scale.build_fixed_kernel_context_cases',
    ('raw', ), 'registered_not_run', '10-second input context at 400 Hz with unchanged convolution sample counts',
    ()), ('fixed_kernel_samples_dilation2_ablation', 'ppg_frailty.models.time_scale.build_fixed_kernel_dilation_cases',
          ('raw', ), 'registered_not_run', 'dilation 2 with unchanged kernel sample counts',
          ())) + _modules('comparison_profile', 'formal_catalog',
                          ('quality_diagnostics_only', 'ppg_frailty.experiment._retain_without_quality_routing',
                           ('raw', 'feature_vector', 'feature_matrix', 'fusion'), 'registered_manual_not_run',
                           'computes diagnostics only; retention, aggregation and prediction stay invariant', ()))
MODEL_MODULES = _modules(
    'model', 'models', ('CompactCNN1D', 'ppg_frailty.models.compact_cnn.CompactCNN1D',
                        ('raw', ), 'reference_not_wang_fcn', '32/64/128 legacy-reference CNN', ('torch', )),
    ('InceptionTimeFull', 'ppg_frailty.models.inception_time_port.InceptionTimeSingleNetwork',
     ('raw', ), 'single_network', 'full single network, not five-member ensemble', ('torch', )),
    ('InceptionTimeSmall', 'ppg_frailty.models.inception_time_port.InceptionTimeSingleNetwork',
     ('raw', ), 'single_network', 'small single network', ('torch', )),
    ('InceptionTimeMatrix', 'ppg_frailty.models.inception_time_port.InceptionTimeSingleNetwork',
     ('feature_matrix', ), 'small_single_network', 'mask-aware 146-by-variable-K input; batch-only padding',
     ('torch', )),
    ('InceptionTimeFullFiveMemberEnsemble',
     'ppg_frailty.models.inception_ensemble.InceptionTimeFiveMemberProbabilityEnsemble',
     ('raw', ), 'configurable_probability_ensemble_legacy_name',
     'one or more independently seeded full raw models and exact probability mean; canonical name retained for compatibility',
     ('torch', )),
    ('InceptionTimeMatrixFiveMemberEnsemble',
     'ppg_frailty.models.inception_ensemble.InceptionTimeFiveMemberProbabilityEnsemble',
     ('feature_matrix', ), 'configurable_probability_ensemble_legacy_name',
     'one or more independently seeded full matrix models and exact probability mean; canonical name retained for compatibility',
     ('torch', )), ('LogisticRegressionL2', 'ppg_frailty.models.feature_models.FeatureVectorBaseline',
                    ('feature_vector', ), 'reference', 'fold-local imputer and scaler',
                    ()), ('RBFSVM', 'ppg_frailty.models.feature_models.FeatureVectorBaseline',
                          ('feature_vector', ), 'reference', 'fold-local imputer and scaler',
                          ()), ('ExtraTrees', 'ppg_frailty.models.feature_models.FeatureVectorBaseline',
                                ('feature_vector', ), 'reference', 'fold-local imputer', ()),
    ('ShapeFormerChannelSpecificOSD',
     'ppg_frailty.models.shapeformer_literature.LiteratureShapeFormerChannelSpecificOSD',
     ('raw', ), 'implemented_not_benchmarked_high_compute',
     'faithful channel-specific OSD/PISD discovery plus ShapeBlock/IG route; never fixed-length and never fallback',
     ('torch', )),
    ('ShapeFormerChannelSpecificScalarDistanceAblation', 'ppg_frailty.models.shapeformer.ExperimentalShapeFormer',
     ('raw', ), 'optional_scalar_distance_ablation_not_literature_reference',
     'reuses the fold-local channel-specific OSD bank with the separately selectable scalar-distance downstream module',
     ('torch', )),
    ('ShapeFormerEffectSizeFixedV1', 'ppg_frailty.models.shapeformer_port.ExperimentalShapeFormer',
     ('raw', ), 'parameterized_effect_size_ablation_legacy_name',
     'effect-size discovery ablation with configurable positive length/stride; legacy name retains 128/64 defaults and is never labelled PISD',
     ('torch', )),
    ('ShapeFormerLegacyEffectSizePort', 'ppg_frailty.models.shapeformer_legacy.LegacyEffectSizeShapeFormer',
     ('raw', ), 'legacy_parallel_ablation_not_osd_parity',
     'isolated historical channel-wise effect-size discovery plus the functional legacy local/shape-token downstream; configurable discovery caps and candidates are executed outer-train only',
     ('torch', )), ('FileBagFusionCompact', 'ppg_frailty.models.file_fusion.FileBagFusionClassifier',
                    ('fusion', ), 'reference', 'file-level concatenation once',
                    ('torch', )), ('FileBagFusionInception', 'ppg_frailty.models.file_fusion.FileBagFusionClassifier',
                                   ('fusion', ), 'reference', 'file-level concatenation once', ('torch', )),
    ('FileBagFusion', 'ppg_frailty.models.file_fusion.FileBagFusionClassifier',
     ('fusion', ), 'optional_composable_signal_encoder',
     'file-level fusion with one registered raw forward_features encoder; fold-local ShapeFormer discovery remains outer-train only',
     ('torch', )))
ALL_MODULES = REPRESENTATION_MODULES + NORMALIZATION_MODULES + PPG_FILTER_MODULES + GAP_REPAIR_MODULES + IMU_GRAVITY_MODULES + DL_RESAMPLING_MODULES + WINDOW_PROFILE_MODULES + OPTIMIZER_MODULES + SAMPLER_MODULES + LOSS_MODULES + CLASS_WEIGHTING_MODULES + CLASS_COUNT_BASIS_MODULES + TRAINING_BALANCE_MODULES + EPOCH_SELECTION_MODULES + QUALITY_MODE_MODULES + MOTION_DETECTOR_SWITCH_MODULES + DENOISER_SWITCH_MODULES + WINDOW_QUALITY_SELECTION_MODULES + SHAPEFORMER_DISCOVERY_BALANCE_MODULES + AGGREGATION_MODULES + QUALITY_WEIGHT_SOURCE_MODULES + FEATURE_GROUP_MODULES + ARTIFACT_MODULES + PRV_BACKEND_MODULES + PEAK_DETECTOR_MODULES + MOTION_EVIDENCE_MODULES + COMPARISON_PROFILE_MODULES + MODEL_MODULES

def _reporter_binding(extension_id: str,
                      summary: str,
                      *references: str,
                      binding_kind: str = 'extension') -> ModuleReporterBinding:
    return ModuleReporterBinding(reporter_extension_id=extension_id,
                                 binding_kind=binding_kind,
                                 algorithm_summary=summary,
                                 references=tuple(references))


_REPORTER_ROWS = (
    (('model', 'CompactCNN1D'), 'compactcnn_model_v1', 'extension',
     'Project CompactCNN1D: three temporal convolution stages with a pooled classification head.',
     'Project implementation: src/ppg_frailty/models/compact_cnn.py'),
    (('model', 'InceptionTimeFull'), 'inceptiontime_single_network_model_v1', 'extension',
     "Six-block, single-network InceptionTime adaptation with bottleneck and parallel fixed-sample kernels; not the paper's five-member ensemble.",
     'Fawaz et al. (2020), InceptionTime, DOI:10.1007/s10618-020-00710-y'),
    (('model', 'InceptionTimeSmall'), 'inceptiontime_single_network_model_v1', 'extension',
     'Three-block project-small single-network InceptionTime adaptation.',
     'Architecture family: Fawaz et al. (2020), DOI:10.1007/s10618-020-00710-y'),
    (('model', 'InceptionTimeMatrix'), 'inceptiontime_matrix_model_v1', 'extension',
     'Project InceptionTime adaptation over the ordered feature-matrix time axis.',
     'Architecture family: Fawaz et al. (2020), DOI:10.1007/s10618-020-00710-y'),
    (('model', 'InceptionTimeFullFiveMemberEnsemble'), 'inceptiontime_probability_ensemble_model_v1', 'extension',
     'Independently seeded raw InceptionTime members combined by an exact arithmetic mean of class probabilities.',
     'Fawaz et al. (2020), InceptionTime, DOI:10.1007/s10618-020-00710-y'),
    (('model', 'InceptionTimeMatrixFiveMemberEnsemble'), 'inceptiontime_probability_ensemble_model_v1', 'extension',
     'Independently seeded matrix InceptionTime members combined by an exact arithmetic mean of class probabilities.',
     'Architecture family: Fawaz et al. (2020), DOI:10.1007/s10618-020-00710-y'),
    (('model', 'LogisticRegressionL2'), 'logistic_l2_model_v1', 'extension',
     'L2-regularized multinomial logistic-regression feature-vector baseline.',
     'Hastie, Tibshirani & Friedman (2009), DOI:10.1007/978-0-387-84858-7'),
    (('model', 'RBFSVM'), 'rbf_svm_model_v1', 'extension',
     'RBF-kernel support-vector classifier with fold-local feature preprocessing and probability output.',
     'Cortes & Vapnik (1995), Support-vector networks, DOI:10.1007/BF00994018'),
    (('model', 'ExtraTrees'), 'extra_trees_model_v1', 'extension',
     'Extremely randomized tree ensemble with fold-local feature imputation.',
     'Geurts, Ernst & Wehenkel (2006), Extremely randomized trees, DOI:10.1007/s10994-006-6226-1'),
    (('model', 'ShapeFormerChannelSpecificOSD'), 'shapeformer_model_v1', 'extension',
     'Channel-specific OSD/PISD discovery plus the project ShapeBlock/information-gain route.',
     'Project implementation/parity contract: src/ppg_frailty/models/shapeformer_literature.py and pisd_port.py'),
    (('model', 'ShapeFormerChannelSpecificScalarDistanceAblation'), 'shapeformer_model_v1', 'extension',
     'Project scalar-distance downstream ablation using a fold-local channel-specific OSD bank.',
     'Project implementation: src/ppg_frailty/models/shapeformer.py; no literature-parity claim'),
    (('model', 'ShapeFormerEffectSizeFixedV1'), 'shapeformer_model_v1', 'extension',
     'Project effect-size shapelet-discovery ablation with configurable length and stride.',
     'Project implementation: src/ppg_frailty/models/shapeformer_port.py; no PISD-equivalence claim'),
    (('model', 'ShapeFormerLegacyEffectSizePort'), 'shapeformer_model_v1', 'extension',
     'Isolated legacy channel-wise effect-size discovery and local/shape-token downstream ablation.',
     'Project legacy implementation: src/ppg_frailty/models/shapeformer_legacy.py; no OSD-parity claim'),
    (('model', 'FileBagFusionCompact'), 'file_bag_fusion_model_v1', 'extension',
     'File-level compact raw-window encoding concatenated once with engineered file features.',
     'Project implementation: src/ppg_frailty/models/file_fusion.py'),
    (('model', 'FileBagFusionInception'), 'file_bag_fusion_model_v1', 'extension',
     'File-level Inception raw-window encoding concatenated once with engineered file features.',
     'Project implementation: src/ppg_frailty/models/file_fusion.py'),
    (('model', 'FileBagFusion'), 'file_bag_fusion_model_v1', 'extension',
     'Composable registered raw encoder with one file-level engineered-feature fusion head.',
     'Project implementation: src/ppg_frailty/models/file_fusion.py'),
    (('peak_detector', 'msptdfast_v2_3_python_port'), 'audit_provenance_v1', 'audit_only',
     'Equation-level Python port of MSPTDfast (v.2); no bitwise MATLAB-parity claim.',
     'Charlton et al. (2025), MSPTDfast (v.2), DOI:10.1088/1361-6579/adb89e'),
    (('peak_detector', 'aboy_project_v2'), 'audit_provenance_v1', 'audit_only',
     'Project-owned seven-step adaptive dual-polarity beat detector.',
     'Project seven-step adaptation; historical family: Aboy et al. (2005), DOI:10.1109/TBME.2005.855725'),
    (('peak_detector', 'aboy_project_v1'), 'audit_provenance_v1', 'audit_only',
     'Historical shared-preprocessing Aboy-family project detector.',
     'Historical project adaptation; algorithm family: Aboy et al. (2005), DOI:10.1109/TBME.2005.855725'),
    (('artifact', 'pca_bss'), 'audit_provenance_v1', 'audit_only',
     'Project PCA-BSS motion-artifact reducer using principal-subspace separation.',
     'PCA basis: Jolliffe (2002), DOI:10.1007/b98835; project BSS reducer implementation is source-local'),
    (('artifact', 'fastica_bss'), 'audit_provenance_v1', 'audit_only',
     'Project FastICA-BSS motion-artifact reducer using fixed-point independent components.',
     'FastICA basis: Hyvärinen (1999), DOI:10.1109/72.761722; project BSS reducer implementation is source-local'),
    (('imu_gravity', 'calibrated_roll_pitch_ekf'), 'audit_provenance_v1', 'audit_only',
     'Calibrated roll-pitch EKF gravity compensation retaining physical-unit dynamic acceleration and gyroscope views.',
     'Roll/pitch EKF context: Sabatini (2011), DOI:10.3390/s110201482; project equations/noise are persisted in resolved config'
     ),
    (('imu_gravity', 'sensor_filter_only_no_gravity_removal'), 'audit_provenance_v1', 'audit_only',
     'Calibrated sensor-filter-only IMU ablation that retains gravity in the three acceleration input channels.',
     'Project ablation implementation: src/ppg_frailty/signal/motion_imu.py; no gravity vector is estimated or subtracted'
     ), (('optimizer', 'adamw'), 'audit_provenance_v1', 'audit_only',
         'AdamW optimizer with decoupled weight decay and persisted hyperparameters.',
         'Loshchilov & Hutter (2019), DOI:10.48550/arXiv.1711.05101'),
    (('optimizer', 'adam'), 'audit_provenance_v1', 'audit_only',
     'Adam optimizer with persisted learning-rate, beta and epsilon parameters.',
     'Kingma & Ba (2015), Adam, DOI:10.48550/arXiv.1412.6980'))
_EXPLICIT_MODULE_REPORTER_BINDINGS: Mapping[tuple[str, str], ModuleReporterBinding] = {
    key: ModuleReporterBinding(extension, kind, summary, (reference, ))
    for key, extension, kind, summary, reference in _REPORTER_ROWS
}
_REGISTERED_MODULES_BY_KEY: Mapping[tuple[str, str], ModuleDescriptor] = {(item.family, item.module_id): item
                                                                          for item in ALL_MODULES}
if len(_REGISTERED_MODULES_BY_KEY) != len(ALL_MODULES):
    raise RuntimeError('module registry contains duplicate family/module identities')
if not set(_EXPLICIT_MODULE_REPORTER_BINDINGS) <= set(_REGISTERED_MODULES_BY_KEY):
    raise RuntimeError('reporter binding names an unregistered module')
MODULE_REPORTER_BINDINGS: Mapping[tuple[str, str], ModuleReporterBinding] = {
    key: _EXPLICIT_MODULE_REPORTER_BINDINGS.get(
        key,
        ModuleReporterBinding(
            reporter_extension_id='audit_provenance_v1',
            binding_kind='audit_only',
            algorithm_summary=descriptor.notes,
            references=
            (f'Project registry implementation: {descriptor.implementation}; no separate external literature source claimed',
             )))
    for key, descriptor in _REGISTERED_MODULES_BY_KEY.items()
}
_REPORTER_MODULE_ALIASES: Mapping[tuple[str, str], str] = {
    ('peak_detector', 'aboy_project'): 'aboy_project_v2',
    ('quality_mode', 'quality_off'): 'off',
    ('quality_mode', 'quality_diagnostics_only'): 'diagnostics_only',
    ('quality_mode', 'quality_route'): 'route',
    ('trainer_model', 'sklearn.linear_model.LogisticRegression'): 'LogisticRegressionL2',
    ('trainer_model', 'sklearn.svm.SVC'): 'RBFSVM',
    ('trainer_model', 'sklearn.ensemble.ExtraTreesClassifier'): 'ExtraTrees'
}
_COMPONENT_ROLE_MODULE_FAMILIES: Mapping[str, tuple[str, ...]] = {
    'classifier': ('model', ),
    'classifier_tuning_candidate': ('model', ),
    'ppg_preprocessing': ('ppg_filter', ),
    'imu_preprocessing': ('imu_gravity', ),
    'peak_detector': ('peak_detector', ),
    'motion_detector': ('motion_evidence', ),
    'motion_detector_reverse_ablation': ('motion_evidence', ),
    'denoiser': ('artifact', ),
    'representation': ('representation', ),
    'aggregation': ('aggregation', ),
    'sqi': ('quality_mode', ),
    'trainer': ('optimizer', 'trainer_model')
}
_AUDIT_ONLY_COMPONENT_ROLES = frozenset({
    'dataset_adapter', 'split_registry', 'legacy_bridge_effective_profile', 'signal_views_and_scaling',
    'window_planner', 'motion_threshold', 'feature_extractor', 'evaluation', 'peak_validation'
})
_EXPLICIT_COMPONENT_REPORTER_BINDINGS: Mapping[tuple[str, str], ModuleReporterBinding] = {
    (role, 'formal_local_supervised_motion_detector_v2'): _reporter_binding(
        'audit_provenance_v1',
        'Project LightCNN binary motion detector over RED/IR plus six processed physical IMU channels.',
        'Project implementation: src/ppg_frailty/models/motion.py; no external architecture-equivalence claim',
        binding_kind='audit_only')
    for role in ('motion_detector', 'motion_detector_reverse_ablation')
}
_AUDIT_ONLY_COMPONENT_IDENTITIES = frozenset({('ppg_preprocessing', 'legacy_detrend_bandpass_0p2_8'),
                                              ('imu_preprocessing', 'legacy_filtered_axes'),
                                              ('imu_preprocessing', 'calibrated_ekf_adyn'),
                                              ('aggregation', 'window_balanced_to_participant')})

def module_reporter_binding(module_id: str, *, family: str | None = None) -> dict[str, Any]:
    """Resolve registry-owned reporter metadata or fail closed."""
    text = str(module_id).strip()
    if not text:
        raise ValueError('active component requires a non-empty module_id')
    requested_family = None if family is None else str(family).strip()
    if requested_family == 'trainer_model':
        requested_family = 'model'
        text = _REPORTER_MODULE_ALIASES.get(('trainer_model', text), text)
    if requested_family == 'model' or requested_family is None:
        try:
            from .models.factory import normalize_model_id
            canonical, _ = normalize_model_id(text)
        except ValueError:
            if requested_family == 'model':
                raise ValueError(f'unknown active model reporter binding: {text}') from None
        else:
            text = canonical
            if requested_family is None:
                requested_family = 'model'
    if requested_family is not None:
        text = _REPORTER_MODULE_ALIASES.get((requested_family, text), text)
        key = (requested_family, text)
        descriptor = _REGISTERED_MODULES_BY_KEY.get(key)
        if descriptor is None:
            raise ValueError(f'unknown active module reporter binding: family={requested_family}, module_id={text}')
    else:
        matches = [(key, descriptor) for key, descriptor in _REGISTERED_MODULES_BY_KEY.items() if key[1] == text]
        if len(matches) != 1:
            qualifier = 'ambiguous' if matches else 'unknown'
            raise ValueError(f'{qualifier} active module reporter binding: {text}')
        key, descriptor = matches[0]
    binding = MODULE_REPORTER_BINDINGS[key]
    return {
        'registered_module_id': descriptor.module_id,
        'registered_module_family': descriptor.family,
        'reporter_extension_id': binding.reporter_extension_id,
        'reporter_binding_kind': binding.binding_kind,
        'algorithm_summary': binding.algorithm_summary,
        'references': binding.references,
        'reporter_binding_source': 'module_registry'
    }

def component_reporter_binding(component_role: str, module_id: str, *, active: bool = True) -> dict[str, Any]:
    """Bind one report component to a registered module or named audit role."""
    role = str(component_role).strip().lower()
    text = str(module_id).strip()
    if not active:
        return {
            'registered_module_id': 'not_applicable',
            'registered_module_family': 'not_applicable',
            'reporter_extension_id': 'not_applicable',
            'reporter_binding_kind': 'inactive_not_applicable',
            'algorithm_summary': '',
            'references': (),
            'reporter_binding_source': 'inactive_component'
        }
    explicit_component_binding = _EXPLICIT_COMPONENT_REPORTER_BINDINGS.get((role, text))
    if explicit_component_binding is not None:
        return {
            'registered_module_id': text,
            'registered_module_family': 'component_contract',
            'reporter_extension_id': explicit_component_binding.reporter_extension_id,
            'reporter_binding_kind': explicit_component_binding.binding_kind,
            'algorithm_summary': explicit_component_binding.algorithm_summary,
            'references': explicit_component_binding.references,
            'reporter_binding_source': 'module_registry:component_identity'
        }
    if (role, text) in _AUDIT_ONLY_COMPONENT_IDENTITIES:
        return {
            'registered_module_id': text,
            'registered_module_family': 'legacy_bridge_audit_identity',
            'reporter_extension_id': 'audit_provenance_v1',
            'reporter_binding_kind': 'audit_only',
            'algorithm_summary': f'Explicit historical bridge identity for {role}; no V2 module-equivalence claim.',
            'references': ('Project legacy bridge execution contract: src/ppg_frailty/legacy_bridge.py', ),
            'reporter_binding_source': 'module_registry:legacy_bridge_identity'
        }
    families = _COMPONENT_ROLE_MODULE_FAMILIES.get(role)
    if families is not None:
        errors: list[str] = []
        for family in families:
            try:
                return module_reporter_binding(text, family=family)
            except ValueError as exc:
                errors.append(str(exc))
        raise ValueError(f'unknown active component reporter binding: role={role}, module_id={text}; ' +
                         ' | '.join(errors))
    if role in _AUDIT_ONLY_COMPONENT_ROLES:
        if not text:
            raise ValueError(f'active {role} component requires a non-empty identity')
        return {
            'registered_module_id': text,
            'registered_module_family': 'component_contract',
            'reporter_extension_id': 'audit_provenance_v1',
            'reporter_binding_kind': 'audit_only',
            'algorithm_summary':
            f'Persisted {role} contract; detailed values remain in the component input and fixed-parameter fields.',
            'references':
            (f'Project component-role audit binding: {role}; no separate external literature source claimed', ),
            'reporter_binding_source': 'module_registry:component_role'
        }
    raise ValueError(f"unknown active component_role reporter binding: {role or '<empty>'}")

def list_modules(family: str = 'all') -> list[dict[str, Any]]:
    """稳定排序导出模块 / Export modules in stable order."""
    allowed = {'all', *(item.family for item in ALL_MODULES)}
    if family not in allowed:
        raise ValueError(f'unknown module family: {family}')
    selected = ALL_MODULES if family == 'all' else tuple((item for item in ALL_MODULES if item.family == family))
    return [asdict(item) for item in sorted(selected, key=lambda item: (item.family, item.module_id))]

def registry_sha256() -> str:
    """计算注册表身份 / Hash the complete registry identity."""
    encoded = json.dumps(list_modules(), sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()

def resolve_peak_detector_config(signal_section: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve one detector plus its public execution thresholds."""
    signal_data = dict(signal_section)
    raw = signal_data.get('peak_detector')
    if not isinstance(raw, Mapping):
        raise ValueError('signal.peak_detector must persist detector_id and failure_action')
    data = dict(raw)
    required = {'detector_id', 'failure_action'}
    allowed = required | {'min_observation_sec', 'min_peaks', 'parameters'}
    if not required <= set(data) or not set(data) <= allowed:
        raise ValueError(
            f'signal.peak_detector key mismatch: missing={sorted(required - set(data))}, unknown={sorted(set(data) - allowed)}'
        )
    from .peaks.resolver import DEFAULT_MIN_OBSERVATION_SEC, DEFAULT_MIN_PEAKS, resolve_detector_id, resolve_detector_parameters, validate_peak_detection_parameters
    detector_id = resolve_detector_id(str(data['detector_id']))
    min_observation_sec, min_peaks = validate_peak_detection_parameters(
        data.get('min_observation_sec', DEFAULT_MIN_OBSERVATION_SEC), data.get('min_peaks', DEFAULT_MIN_PEAKS))
    parameters = resolve_detector_parameters(detector_id, data.get('parameters'))
    if data['failure_action'] != 'fail_closed_no_fallback':
        raise ValueError('signal.peak_detector.failure_action must be fail_closed_no_fallback')
    resolved = {
        'detector_id': detector_id,
        'failure_action': 'fail_closed_no_fallback',
        'min_observation_sec': min_observation_sec,
        'min_peaks': min_peaks
    }
    if parameters:
        resolved['parameters'] = parameters
    return resolved


_ARTIFACT_CONFIG_TO_RUNTIME = {
    'identity': 'identity',
    'nlms_imu_anc': 'nlms_imu_anc',
    'ssa_decomposition': 'ssa_decomposition',
    'spectral_mask': 'spectral_mask',
    'pca_bss': 'pca_bss',
    'fastica_bss': 'fastica_bss',
    'nmf_bss': 'nmf_bss',
    'emd_sifting_rate_only': 'emd_sifting_rate_only',
    'ceemd_lite_nlms_legacy': 'ceemd_lite_nlms_legacy',
    'dwt_a2_legacy': 'dwt_a2_legacy'
}
_ARTIFACT_LEGACY_ALIASES = {
    'none': 'identity',
    'direct': 'identity',
    'nlms': 'nlms_imu_anc',
    'ssa': 'ssa_decomposition',
    'decomposition': 'ssa_decomposition',
    'spectral': 'spectral_mask',
    'stft': 'spectral_mask',
    'stft_imu_mask': 'spectral_mask',
    'pca': 'pca_bss',
    'fastica': 'fastica_bss',
    'ica': 'fastica_bss',
    'nmf': 'nmf_bss'
}

def resolve_artifact_module_id(value: str) -> dict[str, Any]:
    """解析 comparison ID 且显式标注旧别名 / Resolve IDs and label legacy aliases."""
    requested = str(value).strip().lower().replace('-', '_')
    if requested in _ARTIFACT_CONFIG_TO_RUNTIME:
        canonical = requested
        legacy = False
    elif requested in _ARTIFACT_LEGACY_ALIASES:
        canonical = _ARTIFACT_LEGACY_ALIASES[requested]
        legacy = True
    else:
        raise ValueError(f'artifact module ID is not registered: {value}')
    return {
        'requested_module_id': str(value),
        'canonical_module_id': canonical,
        'runtime_reducer': _ARTIFACT_CONFIG_TO_RUNTIME[canonical],
        'legacy_alias_used': legacy
    }

def resolve_artifact_config(section: Mapping[str, Any]) -> dict[str, Any]:
    """验证并解析 artifact section / Validate and resolve an artifact section."""
    data = dict(section)
    declared = str(data.get('reducer', 'identity'))
    data.setdefault('denoiser_enabled', declared != 'identity')
    from .quality.motion_bundle_adapter import resolve_reused_motion_detector_config
    neutral_motion = resolve_reused_motion_detector_config()
    data.setdefault('motion_detector', neutral_motion.to_mapping(include_enabled=False))
    required = {
        'reducer', 'reducer_version', 'selection_scope', 'degraded_policy', 'motion_detector_enabled',
        'denoiser_enabled', 'motion_detector', 'non_identity_output_contract', 'failure_action', 'parameters'
    }
    if set(data) != required:
        raise ValueError(
            f'artifact key mismatch: missing={sorted(required - set(data))}, unknown={sorted(set(data) - required)}')
    declared = str(data['reducer'])
    if declared not in _ARTIFACT_CONFIG_TO_RUNTIME:
        raise ValueError(
            f'artifact.reducer={declared!r} has no exact V2 adapter; silent reducer/parameter translation is forbidden')
    if data['selection_scope'] != 'run_before_evaluation':
        raise ValueError('artifact route must be frozen before evaluation')
    if data['failure_action'] != 'no_result_no_fallback':
        raise ValueError('artifact failures must not silently fall back')
    if not isinstance(data['motion_detector_enabled'], bool):
        raise ValueError('artifact.motion_detector_enabled must be boolean')
    if not isinstance(data['denoiser_enabled'], bool):
        raise ValueError('artifact.denoiser_enabled must be boolean')
    motion_mapping = data['motion_detector']
    if not isinstance(motion_mapping, Mapping):
        raise ValueError('artifact.motion_detector must be a mapping')
    if 'enabled' in motion_mapping:
        raise ValueError(
            'artifact.motion_detector.enabled is a duplicate switch; use only artifact.motion_detector_enabled')
    resolved_motion = resolve_reused_motion_detector_config({
        **dict(motion_mapping), 'enabled':
        bool(data['motion_detector_enabled'])
    })
    if not bool(data['motion_detector_enabled']) and resolved_motion != neutral_motion:
        raise ValueError('disabled artifact.motion_detector cannot carry active overrides')
    policy = str(data['degraded_policy'])
    denoiser_policies = {'denoise_then_extract_rate_features', 'denoise_then_compare_rate_exclude'}
    if policy not in {'drop', *denoiser_policies}:
        raise ValueError(
            'artifact.degraded_policy must be drop, denoise_then_extract_rate_features, or denoise_then_compare_rate_exclude'
        )
    denoiser_enabled = bool(data['denoiser_enabled'])
    if denoiser_enabled and declared == 'identity':
        raise ValueError('artifact.denoiser_enabled=true requires a registered non-identity reducer')
    if not denoiser_enabled and declared != 'identity':
        raise ValueError(
            "artifact.denoiser_enabled=false requires reducer='identity'; inactive non-identity parameters may not alter runtime identity"
        )
    if not denoiser_enabled and policy != 'drop' or (denoiser_enabled and policy not in denoiser_policies):
        raise ValueError(
            f'artifact.denoiser_enabled={denoiser_enabled!r} is incompatible with degraded_policy={policy!r}')
    if data['non_identity_output_contract'] != 'rate_only':
        raise ValueError('non-identity artifact output must be rate_only')
    parameters = data['parameters']
    if not isinstance(parameters, Mapping):
        raise ValueError('artifact.parameters must be a mapping')
    from .artifact import get_reducer
    runtime_name = _ARTIFACT_CONFIG_TO_RUNTIME[declared]
    reducer = get_reducer(runtime_name, dict(parameters))
    declared_version = str(data['reducer_version'])
    runtime_version = str(reducer.reducer_version)
    compatible_versions = {runtime_version}
    if declared == 'identity':
        compatible_versions.add('identity_v1')
    if declared_version not in compatible_versions:
        raise ValueError(
            f'artifact.reducer_version={declared_version!r} is not bound to {declared!r} runtime version {runtime_version!r}'
        )
    return {
        'declared_reducer': declared,
        'runtime_reducer': runtime_name,
        'declared_version': declared_version,
        'runtime_version': runtime_version,
        'declared_version_is_compatibility_alias': declared_version != runtime_version,
        'is_identity': bool(reducer.is_identity),
        'motion_detector_enabled': bool(data['motion_detector_enabled']),
        'motion_detector': resolved_motion.to_mapping(include_enabled=False),
        'denoiser_enabled': denoiser_enabled,
        'degraded_policy': policy,
        'parameters': dict(parameters)
    }


_MODEL_MODES = {item.module_id: set(item.representation_modes) for item in MODEL_MODULES}
_MODEL_BASE_FIELDS = set(
    'model_id variant input_channels input_channels_resolution n_classes ensemble_size architecture_parameters'.split())
_MODEL_COMMON = set('seed_policy mask_aware_pooling input_channel_order'.split())
_INCEPTION_FIELDS = _MODEL_COMMON | set(
    'dropout kernel_sizes dilation pool_size out_channels bottleneck_channels depth residual_interval'.split())
_ENSEMBLE_FIELDS = set('comparison_only member_seeds member_seed_roster_id member_variant'.split())
_PISD_DISCOVERY_FIELDS = set(
    'discovery_method input_fs_hz num_pip_ratio shapelets_per_class max_discovery_windows discovery_balance position_search_neighbourhood_samples pip_rounding_rule pip_selection_rule candidate_generation_rule candidate_enumeration_rule candidate_ranking_rule selected_bank_order_rule discovery_position_search_boundary_rule information_gain_split_rule'
    .split())
_SHAPE_DOWNSTREAM_FIELDS = set(
    'hidden_channels dropout patch_size_samples attention_heads attention_layers attention_feedforward_channels distance_position_chunk_size'
    .split())
_MODEL_SPECIFIC_FIELDS = {
    'CompactCNN1D':
    _MODEL_COMMON | set('dropout kernel_sizes dilations pool_sizes stage_channels stage_dropouts'.split()),
    'InceptionTimeFull':
    _INCEPTION_FIELDS,
    'InceptionTimeSmall':
    _INCEPTION_FIELDS,
    'InceptionTimeMatrix':
    _INCEPTION_FIELDS - {'input_channel_order'},
    'InceptionTimeFullFiveMemberEnsemble':
    _INCEPTION_FIELDS | _ENSEMBLE_FIELDS,
    'InceptionTimeMatrixFiveMemberEnsemble':
    _INCEPTION_FIELDS - {'input_channel_order'} | _ENSEMBLE_FIELDS,
    'LogisticRegressionL2':
    set('seed_policy class_weight logistic_c logistic_max_iter logistic_solver'.split()),
    'RBFSVM':
    set('seed_policy class_weight svm_kernel svm_probability svm_c svm_gamma'.split()),
    'ExtraTrees':
    set('seed_policy class_weight extra_trees_n_estimators extra_trees_n_jobs extra_trees_max_features extra_trees_min_samples_leaf'
        .split()),
    'ShapeFormerChannelSpecificOSD':
    _MODEL_COMMON | _PISD_DISCOVERY_FIELDS |
    set('sequence_length_samples local_kernel_width_samples local_embedding_channels shape_embedding_channels attention_feedforward_channels attention_heads attention_query_chunk_size distance_position_chunk_size dropout complexity_norm max_complexity_ratio'
        .split()),
    'ShapeFormerChannelSpecificScalarDistanceAblation':
    _MODEL_COMMON | _PISD_DISCOVERY_FIELDS | _SHAPE_DOWNSTREAM_FIELDS,
    'ShapeFormerEffectSizeFixedV1':
    _MODEL_COMMON | _SHAPE_DOWNSTREAM_FIELDS | set(
        'discovery_method input_fs_hz shapelet_length_samples shapelets_per_class discovery_stride_samples max_candidates_per_class'
        .split()),
    'ShapeFormerLegacyEffectSizePort':
    _MODEL_COMMON | set(
        'discovery_method discovery_balance input_fs_hz sequence_length_samples shapelet_length_samples discovery_stride_samples shapelets_per_class max_discovery_windows candidates_per_class_channel local_kernel_width_samples local_embedding_channels shape_embedding_channels attention_feedforward_channels attention_heads dropout shapelet_search_window_samples complexity_norm max_complexity_ratio'
        .split()),
    'FileBagFusionCompact':
    _MODEL_COMMON | set(
        'signal_dropout signal_kernel_sizes signal_dilations signal_pool_sizes signal_stage_channels signal_stage_dropouts feature_hidden_dim fusion_hidden_dim pooling dropout'
        .split()),
    'FileBagFusionInception':
    _MODEL_COMMON | set(
        'signal_variant signal_dropout signal_kernel_sizes signal_dilation signal_pool_size signal_out_channels signal_bottleneck_channels signal_depth signal_residual_interval feature_hidden_dim fusion_hidden_dim pooling dropout'
        .split()),
    'FileBagFusion':
    _MODEL_COMMON | set('signal_encoder feature_hidden_dim fusion_hidden_dim pooling dropout'.split())
}
_INCEPTION_OPTIONAL = set('pool_size out_channels bottleneck_channels depth residual_interval'.split())
_SHAPE_REQUIRED = {'seed_policy', 'input_channel_order', 'mask_aware_pooling'}
_MODEL_OPTIONAL_FIELDS = {
    'CompactCNN1D': {'stage_channels', 'stage_dropouts'},
    'InceptionTimeFull':
    _INCEPTION_OPTIONAL,
    'InceptionTimeSmall':
    _INCEPTION_OPTIONAL,
    'InceptionTimeMatrix':
    _INCEPTION_OPTIONAL,
    'InceptionTimeFullFiveMemberEnsemble':
    _INCEPTION_OPTIONAL | {'comparison_only', 'member_seed_roster_id', 'member_variant'},
    'InceptionTimeMatrixFiveMemberEnsemble':
    _INCEPTION_OPTIONAL | {'comparison_only', 'member_seed_roster_id', 'member_variant'},
    'LogisticRegressionL2':
    set('class_weight logistic_c logistic_max_iter logistic_solver'.split()),
    'RBFSVM': {'class_weight'},
    'ExtraTrees':
    set('class_weight extra_trees_n_estimators extra_trees_n_jobs extra_trees_max_features extra_trees_min_samples_leaf'
        .split()),
    **{
        name: _MODEL_SPECIFIC_FIELDS[name] - _SHAPE_REQUIRED
        for name in ('ShapeFormerChannelSpecificOSD', 'ShapeFormerChannelSpecificScalarDistanceAblation', 'ShapeFormerEffectSizeFixedV1', 'ShapeFormerLegacyEffectSizePort')
    }, 'FileBagFusionCompact': {'signal_stage_channels', 'signal_stage_dropouts'},
    'FileBagFusionInception':
    set('signal_pool_size signal_out_channels signal_bottleneck_channels signal_depth signal_residual_interval'.split()
        ),
    'FileBagFusion':
    set('signal_encoder feature_hidden_dim fusion_hidden_dim pooling dropout'.split())
}
_MODEL_FACTORY_METADATA_FIELDS = {
    'input_channel_order', 'mask_aware_pooling', 'member_variant', 'comparison_only', 'member_seed_roster_id'
}
_MODEL_FACTORY_VARIANT_CONSUMERS = {'InceptionTimeMatrix'}
_MODEL_ENSEMBLE_NAMES = {'InceptionTimeFullFiveMemberEnsemble', 'InceptionTimeMatrixFiveMemberEnsemble'}
MODEL_REGISTRY_ROLES = frozenset({'reference', 'ablation', 'comparison', 'optional'})
_MODEL_VARIANT_LEGACY_ALIASES = {
    'CompactCNN1D': {'canonical_32_64_128', 'legacy_reference_not_wang_fcn'},
    'InceptionTimeFull': {'full_single_network'},
    'InceptionTimeSmall': {'small_single_network'},
    'InceptionTimeFullFiveMemberEnsemble': {'full_five_independent_members'},
    'InceptionTimeMatrixFiveMemberEnsemble': {'full_five_independent_members'},
    'LogisticRegressionL2': {'l2_lbfgs', 'reference_file_vector'},
    'RBFSVM': {'rbf_probability'},
    'ExtraTrees': {'500_trees'},
    'ShapeFormerChannelSpecificOSD': {'channel_specific_osd_reference'},
    'ShapeFormerChannelSpecificScalarDistanceAblation': {'channel_specific_scalar_distance_ablation'},
    'ShapeFormerEffectSizeFixedV1': {'effect_size_fixed_v1'},
    'ShapeFormerLegacyEffectSizePort': {'legacy_parallel_ablation_not_osd_parity'},
    'FileBagFusionCompact': {'compact_raw_encoder'},
    'FileBagFusionInception': {'small_raw_encoder'},
    'FileBagFusion': {'optional_composable_signal_encoder'}
}

def model_factory_contract(model_id_or_name: str) -> dict[str, Any]:
    """Return the registry-owned adapter contract for one model factory."""
    from .models.factory import normalize_model_id
    canonical, machine_id = normalize_model_id(str(model_id_or_name))
    descriptor = next((item for item in MODEL_MODULES if item.module_id == canonical))
    scientific_status = str(descriptor.scientific_status)
    if canonical in _MODEL_ENSEMBLE_NAMES:
        registry_role = 'comparison'
    elif 'ablation' in scientific_status:
        registry_role = 'ablation'
    elif 'optional' in scientific_status:
        registry_role = 'optional'
    else:
        registry_role = 'reference'
    factory_fields = set(_MODEL_SPECIFIC_FIELDS[canonical]) - _MODEL_FACTORY_METADATA_FIELDS
    if canonical in _MODEL_FACTORY_VARIANT_CONSUMERS:
        factory_fields.add('variant')
    if canonical not in _MODEL_ENSEMBLE_NAMES:
        factory_fields.add('seed')
    optional_fields = set(_MODEL_OPTIONAL_FIELDS.get(canonical, set())) & factory_fields
    return {
        'canonical_model_name':
        canonical,
        'machine_model_id':
        machine_id,
        'registry_role':
        registry_role,
        'scientific_status':
        scientific_status,
        'factory_fields':
        tuple(sorted(factory_fields)),
        'optional_factory_fields':
        tuple(sorted(optional_fields)),
        'derived_provenance_fields':
        tuple(
            sorted({
                'architecture_parameters', 'ensemble_size', *(() if canonical == 'InceptionTimeMatrix' else
                                                              ('variant', )),
                *(('mask_aware_pooling', ) if 'mask_aware_pooling' in _MODEL_SPECIFIC_FIELDS[canonical] else
                  ()), *(('comparison_only', 'member_seed_roster_id') if canonical in _MODEL_ENSEMBLE_NAMES else ())
            })),
        'representation_modes':
        descriptor.representation_modes,
        'runtime_dependencies':
        descriptor.runtime_dependencies,
        'execution_backend':
        'torch' if 'torch' in descriptor.runtime_dependencies else 'estimator'
    }

def model_runtime_dependencies(model_id_or_name: str) -> tuple[str, ...]:
    """Return executable imports declared by the model's registry contract."""
    return tuple(model_factory_contract(model_id_or_name)['runtime_dependencies'])

def derived_model_ensemble_size(section: Mapping[str, Any]) -> int:
    """Derive member count from the executable roster, never a duplicate field."""
    if 'ensemble_size' in section and (isinstance(section['ensemble_size'], bool)
                                       or not isinstance(section['ensemble_size'], int)
                                       or int(section['ensemble_size']) <= 0):
        raise ValueError('legacy ensemble_size provenance must be a positive integer when supplied')
    contract = model_factory_contract(str(section.get('model_id', '')))
    if contract['canonical_model_name'] in _MODEL_ENSEMBLE_NAMES:
        derived = len(_validated_ensemble_seed_roster(section.get('member_seeds')))
    else:
        derived = 1
    if 'ensemble_size' in section and int(section['ensemble_size']) != derived:
        raise ValueError('model.ensemble_size derived field mismatch; omit it or match the executable member roster')
    return derived

def derived_model_variant(section: Mapping[str, Any]) -> str:
    """Return a semantic variant, deriving labels that are not factory inputs."""
    contract = model_factory_contract(str(section.get('model_id', '')))
    canonical = str(contract['canonical_model_name'])
    declared = section.get('variant')
    if canonical == 'InceptionTimeMatrix':
        variant = str(section.get('variant', 'full'))
        if variant not in {'full', 'small'}:
            raise ValueError('InceptionTimeMatrix.variant must be full or small')
        return variant
    if canonical in _MODEL_ENSEMBLE_NAMES:
        member_variant = str(section.get('member_variant', 'full'))
        if member_variant not in {'full', 'small'}:
            raise ValueError('ensemble member_variant must be full or small')
        derived = f'{member_variant}_probability_ensemble'
    else:
        descriptor = next((item for item in MODEL_MODULES if item.module_id == canonical))
        derived = str(descriptor.scientific_status)
    if declared is not None and str(declared) != derived and (str(declared) not in _MODEL_VARIANT_LEGACY_ALIASES.get(
            canonical, set())):
        raise ValueError('model.variant derived field mismatch; omit it or use the registered semantic variant')
    return derived

def derived_mask_aware_pooling(section: Mapping[str, Any]) -> bool | None:
    """Materialize mask handling only for models with that runtime capability."""
    contract = model_factory_contract(str(section.get('model_id', '')))
    canonical = str(contract['canonical_model_name'])
    if 'mask_aware_pooling' not in _MODEL_SPECIFIC_FIELDS[canonical]:
        return None
    if section.get('mask_aware_pooling', True) is not True:
        raise ValueError(f'{canonical} does not implement a mask-unaware execution branch')
    return True

def validate_legacy_ensemble_metadata(section: Mapping[str, Any]) -> None:
    """Accept only the two historical catalogue annotations, never new inputs."""
    contract = model_factory_contract(str(section.get('model_id', '')))
    canonical = str(contract['canonical_model_name'])
    if canonical not in _MODEL_ENSEMBLE_NAMES:
        unexpected = sorted({'comparison_only', 'member_seed_roster_id'} & set(section))
        if unexpected:
            raise ValueError(f'{canonical} has no legacy ensemble metadata capability: {unexpected}')
        return
    if 'comparison_only' in section and section['comparison_only'] is not True:
        raise ValueError('model.comparison_only is legacy catalogue provenance and must be true when supplied')
    if 'member_seed_roster_id' in section and section['member_seed_roster_id'] != 'cv_fixed_five_member_seed_roster':
        raise ValueError('model.member_seed_roster_id accepts only the historical catalogue annotation')

def materialize_model_architecture(section: Mapping[str, Any],
                                   representation_mode: str | None = None) -> dict[str, Any]:
    """Derive complete architecture provenance from top-level factory inputs."""
    from .models.factory import ModelInputSpec, materialize_architecture_parameters, validate_source_architecture_annotation
    data = dict(section)
    contract = model_factory_contract(str(data.get('model_id', '')))
    modes = tuple((str(value) for value in contract['representation_modes']))
    mode = modes[0] if representation_mode is None and len(modes) == 1 else str(representation_mode)
    if mode not in modes:
        raise ValueError(f"{contract['canonical_model_name']} is incompatible with {mode}")
    factory_config: dict[str, Any] = {'model_id': contract['machine_model_id']}
    for field in contract['factory_fields']:
        if field in data:
            factory_config[field] = data[field]
    if contract['canonical_model_name'] in _MODEL_ENSEMBLE_NAMES:
        factory_config['variant'] = str(data.get('member_variant', 'full'))
    channel_schema = tuple((str(value) for value in data.get('input_channel_order', ())))
    spec = ModelInputSpec(mode,
                          n_channels=int(data.get('input_channels', 0)),
                          n_classes=int(data.get('n_classes', 3)),
                          channel_schema=channel_schema)
    derived = materialize_architecture_parameters(factory_config, spec)
    validate_source_architecture_annotation(data.get('architecture_parameters'), derived)
    return derived

def _validated_ensemble_seed_roster(values: Any) -> tuple[int, ...]:
    """Return one explicit, non-empty, unique roster safe for OOF storage."""
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ValueError('member_seeds must be a non-string list or tuple')
    seeds: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError('member_seeds must contain integer values')
        seed = int(value)
        if seed < 0 or seed > 4294967295:
            raise ValueError('member_seeds must be in the executable uint32 seed range')
        seeds.append(seed)
    roster = tuple(seeds)
    if not roster or len(roster) != len(set(roster)):
        raise ValueError('member_seeds must be non-empty and unique')
    return roster

def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{field} must be a positive integer')
    return int(value)

def _nonzero_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value == 0:
        raise ValueError(f'{field} must be a non-zero integer')
    return int(value)

def _finite_positive(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{field} must be finite and positive')
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f'{field} must be finite and positive')
    return normalized

def _probability(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{field} must be finite in [0,1)')
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized < 1.0:
        raise ValueError(f'{field} must be finite in [0,1)')
    return normalized

def _positive_integer_sequence(value: Any, *, field: str, length: int, odd: bool = False) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f'{field} must contain exactly {length} integers')
    normalized = tuple((_positive_integer(item, field=field) for item in value))
    if odd and any((item % 2 == 0 for item in normalized)):
        raise ValueError(f'{field} must contain positive odd integers')
    return normalized

def _probability_sequence(value: Any, *, field: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f'{field} must contain exactly {length} probabilities')
    return tuple((_probability(item, field=field) for item in value))

def _validate_extra_trees_max_features(value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        if value not in {'sqrt', 'log2'}:
            raise ValueError("extra_trees_max_features string must be 'sqrt' or 'log2'")
        return
    if isinstance(value, bool):
        raise ValueError('extra_trees_max_features cannot be boolean')
    if isinstance(value, int):
        if value <= 0:
            raise ValueError('integer extra_trees_max_features must be positive')
        return
    if isinstance(value, float) and math.isfinite(value) and (0.0 < value <= 1.0):
        return
    raise ValueError('float extra_trees_max_features must be finite in (0,1]')

def _validate_extra_trees_min_samples_leaf(value: Any) -> None:
    if isinstance(value, bool):
        raise ValueError('extra_trees_min_samples_leaf cannot be boolean')
    if isinstance(value, int) and value > 0:
        return
    if isinstance(value, float) and math.isfinite(value) and (0.0 < value <= 0.5):
        return
    raise ValueError('extra_trees_min_samples_leaf must be a positive integer or fraction in (0,0.5]')

def _validate_fusion_signal_encoder_declaration(value: Any) -> None:
    """Validate the nested public encoder mapping without accepting fit state."""
    from .models.factory import FRAILTY_RAW_CHANNEL_SCHEMA, ModelInputSpec, materialize_architecture_parameters, normalize_fusion_signal_encoder_config
    source = {'model_id': 'compact_cnn'} if value is None else value
    if not isinstance(source, Mapping):
        raise ValueError('model.signal_encoder must be a mapping')
    normalized = normalize_fusion_signal_encoder_config(source)
    contract = model_factory_contract(str(normalized['model_id']))
    allowed = {'model_id', 'model_name', *set(contract['factory_fields']) - {'seed', 'seed_policy'}}
    unknown = sorted(set(source) - allowed)
    if unknown:
        raise ValueError(f'model.signal_encoder contains non-executable or fold-owned fields: {unknown}')
    if tuple(contract['representation_modes']) != ('raw', ):
        raise ValueError('model.signal_encoder must use a raw representation model')
    if contract['execution_backend'] != 'torch':
        raise ValueError('model.signal_encoder must use a torch feature encoder')
    raw_spec = ModelInputSpec('raw', n_channels=8, n_classes=3, channel_schema=FRAILTY_RAW_CHANNEL_SCHEMA)
    architecture = materialize_architecture_parameters(normalized, raw_spec)
    nested = {key: item for key, item in source.items() if key not in {'model_id', 'model_name'}}
    nested.update({
        'model_id': str(contract['canonical_model_name']),
        'input_channels': 8,
        'input_channels_resolution': 'canonical_frailty_raw_8',
        'input_channel_order': list(FRAILTY_RAW_CHANNEL_SCHEMA),
        'n_classes': 3,
        'seed_policy': 'fixed_explicit',
        'mask_aware_pooling': True
    })
    if contract['canonical_model_name'] == 'CompactCNN1D':
        nested.setdefault('dropout', architecture['classifier_dropout'])
        for field in ('kernel_sizes', 'dilations', 'pool_sizes'):
            nested.setdefault(field, architecture[field])
    elif contract['canonical_model_name'] in {'InceptionTimeFull', 'InceptionTimeSmall'}:
        nested.setdefault('dropout', architecture['classifier_dropout'])
        for field in ('kernel_sizes', 'dilation'):
            nested.setdefault(field, architecture[field])
    validate_model_config(nested, 'raw')

def validate_model_config(section: Mapping[str, Any], representation_mode: str) -> dict[str, str]:
    """Validate structure here and numerical options through the model factory."""
    data = dict(section)
    canonical = str(data.get('model_id', ''))
    if canonical not in _MODEL_MODES:
        raise ValueError(f'model_id is not registered: {canonical}')
    allowed = _MODEL_BASE_FIELDS | _MODEL_SPECIFIC_FIELDS[canonical]
    optional = _MODEL_OPTIONAL_FIELDS.get(
        canonical, set()) | {'architecture_parameters', 'ensemble_size', 'variant', 'mask_aware_pooling'}
    missing, unknown = (allowed - optional - set(data), set(data) - allowed)
    if missing or unknown:
        raise ValueError(f'model key mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}')
    if representation_mode not in _MODEL_MODES[canonical]:
        raise ValueError(f'{canonical} is incompatible with {representation_mode}')
    if int(data['n_classes']) != 3:
        raise ValueError('reference task requires exactly three classes')
    derived_model_ensemble_size(data)
    derived_model_variant(data)
    derived_mask_aware_pooling(data)
    validate_legacy_ensemble_metadata(data)
    if representation_mode in {'raw', 'fusion'}:
        canonical_channels = ('RED', 'IR', 'A_dyn_x', 'A_dyn_y', 'A_dyn_z', 'GX', 'GY', 'GZ')
        channels = tuple(data.get('input_channel_order', ()))
        resolution = data.get('input_channels_resolution')
        valid_subset = tuple((x for x in canonical_channels if x in channels)) == channels
        valid = channels and len(channels) == len(set(channels)) == int(data['input_channels']) and valid_subset and (
            representation_mode == 'fusion' and channels == canonical_channels and
            (resolution == 'canonical_frailty_raw_8') or
            (representation_mode == 'raw' and resolution ==
             ('canonical_frailty_raw_8' if channels == canonical_channels else 'explicit_frailty_raw_channel_subset')))
        if not valid:
            raise ValueError('raw/fusion model channel schema is incompatible with the canonical frailty input')
    if canonical in {'LogisticRegressionL2', 'RBFSVM', 'ExtraTrees'} and data.get('class_weight') is not None:
        raise ValueError('model.class_weight is not independent; use training.class_weighting')
    materialize_model_architecture(data, representation_mode)
    from .models import normalize_model_id
    normalized_name, machine_id = normalize_model_id(canonical)
    return {'canonical_model_name': normalized_name, 'machine_model_id': machine_id}


_WINDOW_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    'engineering': {
        'length_s': 10.0,
        'hop_s': 2.0,
        'end_alignment': 'left_start_regular_grid',
        'padding': 'none_complete_windows_only',
        'cap_per_file': None,
        'cap_fraction_per_file': None,
        'min_valid_fraction': 1.0
    },
    'raw_dl': {
        'length_s': 5.0,
        'hop_s': 2.5,
        'end_alignment': 'include_right_aligned_if_distinct',
        'padding': 'none_complete_windows_only',
        'cap_per_file': 128,
        'cap_fraction_per_file': None,
        'min_valid_fraction': 1.0
    }
}

def normalize_window_config(section: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize the complete user-facing window configuration."""
    data = dict(section)
    allowed_section_keys = {'engineering', 'raw_dl', 'shared_planner_version'}
    unknown_section_keys = sorted(set(data) - allowed_section_keys)
    if unknown_section_keys:
        raise ValueError(f'windows section contains unknown fields: {unknown_section_keys}')
    if data.get('shared_planner_version', 'window_plan_v1') != 'window_plan_v1':
        raise ValueError('unsupported shared window planner version')
    normalized: dict[str, Any] = {'shared_planner_version': 'window_plan_v1'}
    for name, defaults in _WINDOW_PROFILE_DEFAULTS.items():
        candidate = data.get(name, {})
        if not isinstance(candidate, Mapping):
            raise ValueError(f'windows.{name} must be a mapping')
        unknown = sorted(set(candidate) - set(defaults))
        if unknown:
            raise ValueError(f'windows.{name} contains unknown fields: {unknown}')
        item = {**defaults, **dict(candidate)}
        if item['end_alignment'] not in {'left_start_regular_grid', 'include_right_aligned_if_distinct'}:
            raise ValueError(f'unsupported windows.{name}.end_alignment')
        padding_modes = {
            'none_complete_windows_only', 'right_zero_pad_short_records', 'right_zero_pad_tail',
            'right_zero_pad_short_records_and_tail'
        }
        if item['padding'] not in padding_modes:
            raise ValueError(f'windows.{name}.padding must be one of {sorted(padding_modes)}')
        if name == 'engineering' and item['padding'] != 'none_complete_windows_only':
            raise ValueError('engineering windows do not support padded feature rows')
        for field_name in ('length_s', 'hop_s'):
            raw_value = item[field_name]
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f'windows.{name}.{field_name} must be numeric')
            value = float(raw_value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'windows.{name}.{field_name} must be finite and positive')
            item[field_name] = value
        cap = item['cap_per_file']
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0):
            raise ValueError(f'windows.{name}.cap_per_file must be null or positive int')
        cap_fraction = item['cap_fraction_per_file']
        if cap_fraction is not None and (isinstance(cap_fraction, bool) or not isinstance(cap_fraction, (int, float)) or
                                         (not math.isfinite(float(cap_fraction))) or
                                         (not 0.0 < float(cap_fraction) <= 1.0)):
            raise ValueError(f'windows.{name}.cap_fraction_per_file must be null or finite in (0,1]')
        if cap is not None and cap_fraction is not None:
            raise ValueError(f'windows.{name}.cap_per_file and cap_fraction_per_file are mutually exclusive')
        item['cap_fraction_per_file'] = None if cap_fraction is None else float(cap_fraction)
        min_valid = item['min_valid_fraction']
        if isinstance(min_valid, bool) or not isinstance(min_valid, (int, float)):
            raise ValueError(f'windows.{name}.min_valid_fraction must be numeric')
        if not math.isfinite(float(min_valid)) or not 0.0 < float(min_valid) <= 1.0:
            raise ValueError(f'windows.{name}.min_valid_fraction must be finite in (0,1]')
        if item['padding'] == 'none_complete_windows_only' and float(min_valid) != 1.0:
            raise ValueError(f'windows.{name}.min_valid_fraction is only variable when padding is enabled')
        item['min_valid_fraction'] = float(min_valid)
        normalized[name] = item
    return normalized

def validate_window_profiles_for_representation(section: Mapping[str, Any], representation_mode: str,
                                                enabled_feature_groups: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Reject non-default window controls with no runtime consumer."""
    normalized = normalize_window_config(section)
    mode = str(representation_mode)
    if mode not in {'raw', 'feature_vector', 'feature_matrix', 'fusion'}:
        raise ValueError(f'unsupported representation_mode: {mode!r}')
    if not isinstance(enabled_feature_groups, (list, tuple)):
        raise ValueError('enabled_feature_groups must be a list or tuple')
    groups = tuple((str(value) for value in enabled_feature_groups))
    inactive_profiles: list[str] = []
    if mode == 'raw':
        inactive_profiles.append('engineering')
    elif mode in {'feature_vector', 'feature_matrix'}:
        inactive_profiles.append('raw_dl')
    if mode in {'feature_vector', 'fusion'} and 'engineering_summary' not in groups:
        inactive_profiles.append('engineering')
    for profile_name in inactive_profiles:
        defaults = _WINDOW_PROFILE_DEFAULTS[profile_name]
        observed = normalized[profile_name]
        changed = sorted((name for name, default in defaults.items() if observed[name] != default))
        if changed:
            raise ValueError(
                f'windows.{profile_name} is inactive for representation_mode={mode!r}; non-default fields would have no runtime consumer: {changed}'
            )
    return normalized

def resolve_window_config(section: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve configurable physical windows to the shared ``WindowPlan``."""
    data = normalize_window_config(section)

    def resolve(name: str) -> dict[str, Any]:
        item = dict(data[name])
        alignment_map = {
            'left_start_regular_grid': 'start',
            'include_right_aligned_if_distinct': 'include_right_aligned_if_distinct'
        }
        padding_map = {
            'none_complete_windows_only': ('reject', False),
            'right_zero_pad_short_records': ('pad_right', False),
            'right_zero_pad_tail': ('reject', True),
            'right_zero_pad_short_records_and_tail': ('pad_right', True)
        }
        short_record_action, include_padded_tail = padding_map[str(item['padding'])]
        if include_padded_tail and alignment_map[str(item['end_alignment'])] != 'start':
            raise ValueError(f'windows.{name} padded tails require left_start_regular_grid alignment')
        cap = item['cap_per_file']
        cap_fraction = item['cap_fraction_per_file']
        has_cap = cap is not None or cap_fraction is not None
        return {
            'window_seconds': float(item['length_s']),
            'hop_seconds': float(item['hop_s']),
            'end_alignment': alignment_map[str(item['end_alignment'])],
            'short_record_action': short_record_action,
            'include_padded_tail': include_padded_tail,
            'max_windows': cap,
            'cap_policy': 'uniform_progress' if has_cap else 'not_applicable',
            'min_valid_fraction': float(item['min_valid_fraction']),
            'max_window_fraction': cap_fraction
        }

    return {'engineering': resolve('engineering'), 'raw_dl': resolve('raw_dl')}


__all__ = [
    'AGGREGATION_MODULES', 'ALL_MODULES', 'ARTIFACT_MODULES', 'CLASS_COUNT_BASIS_MODULES', 'CLASS_WEIGHTING_MODULES',
    'COMPARISON_PROFILE_MODULES', 'EPOCH_SELECTION_MODULES', 'DENOISER_SWITCH_MODULES', 'FEATURE_GROUP_MODULES',
    'LOSS_MODULES', 'MODEL_MODULES', 'MOTION_DETECTOR_SWITCH_MODULES', 'MOTION_EVIDENCE_MODULES',
    'NORMALIZATION_MODULES', 'OPTIMIZER_MODULES', 'PRV_BACKEND_MODULES', 'QUALITY_MODE_MODULES',
    'QUALITY_WEIGHT_SOURCE_MODULES', 'REPRESENTATION_MODULES', 'SAMPLER_MODULES',
    'SHAPEFORMER_DISCOVERY_BALANCE_MODULES', 'TRAINING_BALANCE_MODULES', 'WINDOW_QUALITY_SELECTION_MODULES',
    'MODULE_REPORTER_BINDINGS', 'ModuleDescriptor', 'ModuleReporterBinding', 'component_reporter_binding',
    'derived_mask_aware_pooling', 'derived_model_ensemble_size', 'derived_model_variant', 'list_modules',
    'materialize_model_architecture', 'model_factory_contract', 'module_reporter_binding', 'model_runtime_dependencies',
    'registry_sha256', 'resolve_artifact_config', 'normalize_window_config', 'resolve_artifact_module_id',
    'resolve_window_config', 'validate_window_profiles_for_representation', 'validate_legacy_ensemble_metadata',
    'validate_model_config'
]
