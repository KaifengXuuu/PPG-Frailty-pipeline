"""冻结 outer-fold 实验执行入口 / Frozen outer-fold experiment entry points.

中文：任何尚不能满足完整科学合同的执行都会返回结构化 ``failed_closed``，
绝不缩写 roster、放宽 SQI 或输出伪造指标。English: Any execution that cannot
meet the complete scientific contract returns structured ``failed_closed`` without
shortening the roster, relaxing SQI, or emitting fabricated metrics.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import to_strict_json_value


@dataclass(frozen=True)
class ExperimentResult:
    """可严格序列化的实验结论 / Strictly serializable experiment outcome."""

    status: str
    scientific_scope: str
    config_id: str
    config_hash: str
    repeat_indices: tuple[int, ...]
    fold_indices: tuple[int, ...]
    output_dir: str | None
    cell_results: tuple[dict[str, Any], ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    failure_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """转换为 strict JSON value / Convert to a strict-JSON value."""

        payload = asdict(self)
        payload["schema_version"] = "ppg_frailty.experiment_result.v2"
        payload["pipeline_generation"] = "final_pipeline_v2"
        return to_strict_json_value(payload)


@dataclass
class _RuntimeRecord:
    '''单折内 recording 状态 / Per-record state inside one fold.'''

    row: Any
    views: Any = None
    direct_quality: Any = None
    final_quality: Any = None
    route: Any = None
    intended_route: Any = None
    retained: bool = False
    reason: str | None = None
    route_status: str = 'pending'
    artifact_name: str = 'not_executed'
    artifact_version: str = 'not_executed'
    vector: Any = None
    engineering: Any = None
    raw_windows: Any = None
    transformed_vector: Any = None
    transformed_engineering: Any = None
    matrix: Any = None
    fusion_features: Any = None
    diagnostic_components: dict[str, Any] = field(default_factory=dict)
    diagnostic_reason: str | None = None
    physical_qc_evidence: dict[str, Any] = field(default_factory=dict)
    physical_qc_profile: dict[str, Any] = field(default_factory=dict)


class _ExperimentProtocolError(RuntimeError):
    '''关闭失败异常 / Fail-closed protocol exception.'''


_ESTIMATOR_MODEL_IDS = frozenset(
    {
        'logistic_regression',
        'rbf_svm',
        'extra_trees',
        'rocket_numpy',
        'minirocket_ablation',
    }
)
_ESTIMATOR_NOT_APPLICABLE = 'not_applicable_estimator_native'


def _model_input_sampling_rate_hz(config: Any) -> float:
    '''Return the actual model-input grid while preserving canonical 400-Hz views.'''

    signal = config.section('signal')
    dl = signal['dl_resampling']
    case_id = dl.get('case_id')
    if case_id is None:
        return float(signal['internal_fs_hz'])
    if config.representation_mode != 'raw':
        raise _ExperimentProtocolError(
            'fixed_kernel_samples_sampling_rate_requires_raw_representation'
        )
    target = float(dl['target_fs_hz'])
    if target not in {100.0, 160.0, 200.0, 400.0}:
        raise _ExperimentProtocolError('fixed_kernel_samples_sampling_rate_invalid')
    return target


def _training_algorithm_provenance(
    model_id: str,
    training_section: Mapping[str, Any],
    model_section: Mapping[str, Any],
    *,
    fixed_epochs: int,
) -> dict[str, Any]:
    '''Describe only training controls that the selected implementation consumes.'''

    if model_id in _ESTIMATOR_MODEL_IDS:
        return {
            'loss': _ESTIMATOR_NOT_APPLICABLE,
            'class_weighting': training_section['class_weighting'],
            'sampler': training_section['sampler'],
            'epoch_rule': {'rule': 'not_applicable', 'fixed_epochs': None},
            'optimizer': _ESTIMATOR_NOT_APPLICABLE,
            'learning_rate': _ESTIMATOR_NOT_APPLICABLE,
            'weight_decay': _ESTIMATOR_NOT_APPLICABLE,
            'dropout': _ESTIMATOR_NOT_APPLICABLE,
            'label_smoothing': _ESTIMATOR_NOT_APPLICABLE,
            'gradient_clipping': {
                'enabled': False,
                'max_norm': None,
                'status': _ESTIMATOR_NOT_APPLICABLE,
            },
        }
    gradient_clip_norm = training_section.get('gradient_clip_norm')
    return {
        'loss': training_section['loss'],
        'class_weighting': training_section['class_weighting'],
        'sampler': training_section['sampler'],
        'epoch_rule': {
            'rule': training_section['epoch_rule'],
            'profile': training_section['epoch_profile'],
            'fixed_epochs': int(fixed_epochs),
        },
        'optimizer': training_section['optimizer'],
        'learning_rate': float(training_section['learning_rate']),
        'weight_decay': float(training_section['weight_decay']),
        'dropout': (
            float(model_section['dropout'])
            if 'dropout' in model_section
            else 'not_applicable'
        ),
        'label_smoothing': float(training_section['label_smoothing']),
        'gradient_clipping': {
            'enabled': gradient_clip_norm is not None,
            'max_norm': gradient_clip_norm,
        },
    }


@dataclass(frozen=True)
class _CellResult:
    '''单折摘要与 OOF / One cell summary and OOF tables.'''

    summary: dict[str, Any]
    file_rows: tuple[Any, ...]
    subject_rows: tuple[Any, ...]
    window_rows: tuple[Any, ...] = ()
    member_rows: tuple[Any, ...] = ()


def _runtime_imports() -> dict[str, Any]:
    '''延迟导入重型依赖 / Lazily import experiment dependencies.'''

    import numpy as np
    from dataclasses import replace
    from ppg_frailty.artifact import run_artifact_route
    from ppg_frailty.contracts import QualityState, SignalRoute
    from ppg_frailty.data.schema import canonicalize_role_family
    from ppg_frailty.data.windows import WindowPlan
    from ppg_frailty.features.engineering import (
        extract_engineering_features,
        fit_fold_feature_transform,
        transform_engineering,
    )
    from ppg_frailty.features.registry import (
        build_feature_vector,
        build_ordered_matrix,
        default_registry,
        summarize_engineering,
    )
    from ppg_frailty.features.vector_transform import (
        fit_fold_feature_vector_transform,
        transform_feature_vector_batch,
    )
    from ppg_frailty.models import (
        ModelInputSpec,
        normalize_model_id,
        prepare_model_factory,
        validate_frozen_model_run_provenance,
    )
    from ppg_frailty.pipeline import PipelinePaths, _load_record, preflight_pipeline
    from ppg_frailty.provenance import runtime_environment, stable_payload_sha256
    from ppg_frailty.signal.morphology import extract_morphology
    from ppg_frailty.signal.optical import extract_dual_optical
    from ppg_frailty.signal.peaks import detect_pulses
    from ppg_frailty.signal.preprocess import build_signal_views
    from ppg_frailty.signal.prv import compute_prv
    from ppg_frailty.signal.sqi import (
        SqiConfig,
        evaluate_quality,
        evaluate_quality_diagnostics,
        fit_sqi_calibrator,
        quality_component_scores,
    )
    from ppg_frailty.representations import (
        build_raw_windows,
        fit_fold_imu_channel_transform,
        transform_raw_windows_imu,
    )
    from ppg_frailty.training import (
        ParticipantPrediction,
        FeatureMatrixDataset,
        FeatureVectorDataset,
        FileBagDataset,
        FrozenOuterSplit,
        OofPredictionRow,
        OofWriter,
        RawWindowDataset,
        build_config_metrics_from_predictions_and_fold_summaries,
        measure_cpu_batch1_operational_metrics,
    )
    from ppg_frailty.training import SampleIdentity, TrainingConfig, UnifiedTrainer
    from ppg_frailty.training import aggregate_hierarchy, evaluate_predictions, validate_expected_oof_roster
    return locals()


def _choose_records(rows: Iterable[Any], participant_ids: Iterable[str], roles: Iterable[str], cap: int | None) -> list[Any]:
    '''逐 participant 选择最长文件 / Select longest files per participant.'''

    participants = set(str(value) for value in participant_ids)
    role_set = set(str(value) for value in roles)
    output: list[Any] = []
    for participant in sorted(participants):
        candidates = sorted(
            (
                row for row in rows
                if row.participant_id == participant
                and row.role in role_set
                and row.qc_status in {'pass', 'pass_with_warnings'}
            ),
            key=lambda row: (-float(row.duration_s), str(row.record_id)),
        )
        output.extend(candidates if cap is None else candidates[:cap])
    if {row.participant_id for row in output} != participants:
        raise _ExperimentProtocolError('selected_records_do_not_cover_frozen_roster')
    return output


def _preprocess_records(states: list[_RuntimeRecord], config: Any, maximum_seconds: float | None, loader: Any) -> None:
    '''读取并构建 direct views / Read and build direct signal views.'''

    build_signal_views = _runtime_imports()['build_signal_views']
    for state in states:
        maximum = None if maximum_seconds is None else min(
            int(state.row.n_samples), int(round(maximum_seconds * float(state.row.fs)))
        )
        try:
            loaded = loader(state.row, maximum)
            state.physical_qc_evidence = to_strict_json_value(
                loaded.get(
                    'recording_qc',
                    {
                        'status': 'not_supplied_by_injected_test_loader',
                        'record_id': state.row.record_id,
                    },
                )
            )
            state.physical_qc_profile = to_strict_json_value(
                loaded.get(
                    'recording_qc_profile',
                    {'profile_id': 'not_supplied_by_injected_test_loader'},
                )
            )
            state.views = build_signal_views(loaded, config.to_dict())
            state.route_status = 'direct_preprocessed'
        except Exception as exc:
            state.reason = f'preprocess_failed:{type(exc).__name__}:{exc}'
            state.route_status = 'dropped_preprocess'


def _fit_quality_calibrator(states: list[_RuntimeRecord], config: Any, train_ids: tuple[str, ...], oof_ids: tuple[str, ...]) -> tuple[Any, Any]:
    '''先 direct fixed-formula，再仅 train 拟合 empirical SQI / Fit SQI train-only.'''

    api = _runtime_imports()
    formal = api['SqiConfig'].from_resolved(config.to_dict())
    base = api['replace'](formal, calibrator='fixed_formula_thresholds_v1')
    component_rows: list[dict[str, float]] = []
    participant_rows: list[str] = []
    for state in states:
        if state.views is None:
            continue
        try:
            quality = api['evaluate_quality'](state.views, config=base)
            component_rows.append(api['quality_component_scores'](quality))
            participant_rows.append(str(state.row.participant_id))
        except Exception as exc:
            state.views = None
            state.reason = f'direct_base_sqi_failed:{type(exc).__name__}:{exc}'
            state.route_status = 'dropped_direct_base_sqi'
    fitted_ids = tuple(sorted({value for value in participant_rows if value in set(train_ids)}))
    if not fitted_ids:
        raise _ExperimentProtocolError('no_outer_train_direct_sqi_components')
    calibrator = api['fit_sqi_calibrator'](
        component_rows,
        participant_rows,
        fitted_on_participant_ids=fitted_ids,
        outer_train_participant_ids=train_ids,
        outer_oof_participant_ids=oof_ids,
    )
    if set(calibrator.fitted_on_participant_ids) & set(oof_ids):
        raise _ExperimentProtocolError('heldout_subject_in_sqi_calibrator')
    return formal, calibrator


def _quality_mode(config: Any) -> str:
    '''Resolve explicit V2 quality mode; route remains disabled / V2 quality gate.'''

    mode = str(config.section('quality').get('mode', 'off'))
    if mode not in {'off', 'diagnostics_only', 'route'}:
        raise _ExperimentProtocolError(f'unsupported_quality_mode:{mode}')
    # A user-editable readiness boolean is not scientific evidence. V2 has no
    # frozen supervised SQI artifact ID/hash, so YAML alone cannot enable routing.
    if mode == 'route':
        raise _ExperimentProtocolError(
            'quality_route_disabled_no_frozen_supervised_artifact'
        )
    return mode


def _retain_without_quality_routing(
    states: list[_RuntimeRecord],
    config: Any,
    *,
    diagnostics_only: bool,
) -> dict[str, Any]:
    '''Retain hard-QC-valid records while SQI has no classification effect.

    Off never evaluates SQI. Diagnostics-only records fixed-formula observations,
    but diagnostic failures cannot drop a record or alter routing, weights,
    aggregation, features, or predictions.
    '''

    api = _runtime_imports()
    diagnostic_config = None
    if diagnostics_only:
        diagnostic_config = api['SqiConfig'].from_resolved(config.to_dict())
    artifact = config.section('artifact')
    for state in states:
        if state.views is None:
            continue
        if diagnostic_config is not None:
            try:
                observed = api['evaluate_quality_diagnostics'](
                    state.views,
                    config=diagnostic_config,
                )
                state.direct_quality = observed
                state.diagnostic_components = to_strict_json_value(
                    asdict(observed)
                )
            except Exception as exc:
                state.diagnostic_reason = (
                    f'diagnostics_only_failed:{type(exc).__name__}:{exc}'
                )
        state.final_quality = None
        state.route = api['SignalRoute'].DIRECT
        state.intended_route = api['SignalRoute'].DIRECT
        state.retained = True
        state.route_status = (
            'retained_direct_diagnostics_only'
            if diagnostics_only
            else 'retained_direct_quality_off'
        )
        state.artifact_name = 'identity'
        state.artifact_version = 'identity_v1'
    return {
        'method': (
            'diagnostics_only_raw_components_no_weights_thresholds_or_fit'
            if diagnostics_only
            else 'not_applicable_quality_off'
        ),
        'fitted_on_participant_ids': (),
        'outer_oof_ids_absent': True,
        'classification_effect': 'none',
        'configured_reducer_not_executed': str(artifact['reducer']),
    }


def _route_records(states: list[_RuntimeRecord], config: Any, report: Any, sqi_config: Any, calibrator: Any) -> None:
    '''direct SQI 后执行 motion override 与 drop XOR reducer / Apply locked routing.'''

    api = _runtime_imports()
    QualityState = api['QualityState']
    SignalRoute = api['SignalRoute']
    artifact = config.section('artifact')
    policy = str(artifact['degraded_policy'])
    if policy not in {'drop', 'denoise_then_extract_rate_features'}:
        raise _ExperimentProtocolError(f'unsupported_degraded_policy:{policy}')
    for state in states:
        motion_override = bool(
            artifact['motion_detector_enabled']
            and str(state.row.role).startswith(('S', 'W'))
        )
        state.intended_route = SignalRoute.ARTIFACT_RATE_ONLY if (
            motion_override or policy == 'denoise_then_extract_rate_features'
        ) else SignalRoute.DIRECT
        if state.views is None:
            continue
        try:
            direct = api['evaluate_quality'](state.views, config=sqi_config, calibrator=calibrator)
            state.direct_quality = direct
            direct_high = direct.q_rate.state is QualityState.PASS and direct.q_morph.state is QualityState.PASS
            if direct_high and not motion_override:
                state.final_quality = direct
                state.route = SignalRoute.DIRECT
                state.intended_route = SignalRoute.DIRECT
                state.retained = True
                state.route_status = 'retained_direct_high_quality'
                state.artifact_name = str(artifact['reducer'])
                state.artifact_version = str(artifact['reducer_version'])
                continue
            if policy == 'drop':
                state.reason = 'motion_override_drop' if motion_override else 'direct_quality_below_threshold_drop'
                state.route_status = 'dropped_by_run_locked_policy'
                continue
            outcome = api['run_artifact_route'](
                state.views,
                report.artifact['runtime_reducer'],
                parameters=report.artifact['parameters'],
            )
            state.artifact_name = outcome.result.reducer_id
            state.artifact_version = outcome.result.reducer_version
            if outcome.views is None or outcome.route is not SignalRoute.ARTIFACT_RATE_ONLY:
                state.reason = 'artifact_no_result:' + ';'.join(outcome.result.reasons)
                state.route_status = 'dropped_artifact_no_result'
                continue
            post = api['evaluate_quality'](outcome.views, config=sqi_config, calibrator=calibrator)
            if post.q_morph.state is not QualityState.NOT_APPLICABLE or post.q_morph.score is not None:
                raise _ExperimentProtocolError('nonidentity_post_q_morph_contract_failed')
            state.views = outcome.views
            state.final_quality = post
            state.route = SignalRoute.ARTIFACT_RATE_ONLY
            if post.q_rate.state is QualityState.PASS:
                state.retained = True
                state.route_status = 'retained_artifact_rate_only'
            else:
                state.reason = 'post_q_rate_below_threshold'
                state.route_status = 'dropped_post_q_rate'
        except _ExperimentProtocolError:
            raise
        except Exception as exc:
            state.reason = f'quality_route_failed:{type(exc).__name__}:{exc}'
            state.route_status = 'dropped_quality_route_failure'


def _extract_vector(state: _RuntimeRecord, report: Any) -> None:
    '''构建完整 FeatureVectorV1 / Build one complete FeatureVectorV1.'''

    if not state.retained:
        return
    api = _runtime_imports()
    SignalRoute = api['SignalRoute']
    QualityState = api['QualityState']
    try:
        pulse = api['detect_pulses'](state.views)
        prv = api['compute_prv'](
            pulse,
            observation_duration_s=state.views.x_filter.shape[0] / 400.0,
            role=api['canonicalize_role_family'](str(state.row.role)),
            route=state.route,
            q_rate_qualified=(
                True
                if state.final_quality is None
                else state.final_quality.q_rate.state is QualityState.PASS
            ),
        )
        plan = api['WindowPlan'](
            source_record_id=state.row.record_id,
            **report.window_profiles['engineering'],
        )
        engineering = api['extract_engineering_features'](state.views, plan=plan)
        state.engineering = engineering
        values, validity = api['summarize_engineering'](engineering)
        if state.final_quality is not None:
            values['sqi.q_rate'] = float(state.final_quality.q_rate.score)
            validity['sqi.q_rate'] = True
            values['sqi.coverage'] = float(prv.values['coverage'])
            validity['sqi.coverage'] = True
            if state.final_quality.q_morph.score is not None:
                values['sqi.q_morph'] = float(state.final_quality.q_morph.score)
                validity['sqi.q_morph'] = True
        for name, value in prv.values.items():
            values[f'prv.{name}'] = value
            validity[f'prv.{name}'] = bool(prv.validity[name])
        if state.route in {SignalRoute.DIRECT, SignalRoute.IDENTITY}:
            morphology = api['extract_morphology'](state.views.x_filter, pulse, route=state.route)
            for name, value in morphology.aggregate_values.items():
                values[f'morphology.{name}'] = value
                validity[f'morphology.{name}'] = bool(morphology.aggregate_validity[name])
            optical = api['extract_dual_optical'](
                state.views.x_native,
                state.views.x_filter,
                pulse,
                route=state.route,
            )
            for name, value in optical.aggregate_values.items():
                values[f'optical.{name}'] = value
                validity[f'optical.{name}'] = bool(optical.aggregate_validity[name])
        state.vector = api['build_feature_vector'](
            values,
            feature_validity=validity,
            provenance={'route': state.route.value, 'record_id': state.row.record_id},
        )
    except Exception as exc:
        state.retained = False
        state.reason = f'feature_vector_failed:{type(exc).__name__}:{exc}'
        state.route_status = 'dropped_feature_vector_failure'


def _dataset(states: Iterable[_RuntimeRecord]) -> Any:
    '''物化 file-level vector dataset / Materialise a file-level vector dataset.'''

    api = _runtime_imports()
    selected = sorted(
        (state for state in states if state.retained and state.vector is not None),
        key=lambda state: str(state.row.record_id),
    )
    if not selected:
        return None
    names = tuple(selected[0].vector.feature_names)
    if any(tuple(state.vector.feature_names) != names for state in selected):
        raise _ExperimentProtocolError('feature_vector_schema_drift')
    identities = tuple(
        api['SampleIdentity'](
            participant_id=state.row.participant_id,
            file_id=state.row.record_id,
            role=api['canonicalize_role_family'](str(state.row.role)),
            label=int(state.row.class_id),
            signal_route=state.route.value,
            quality_score=(
                1.0
                if state.final_quality is None
                else float(state.final_quality.q_rate.score)
            ),
        )
        for state in selected
    )
    return api['FeatureVectorDataset'](
        api['np'].stack([state.vector.values for state in selected]),
        names,
        identities,
    )


def _extract_raw(state: _RuntimeRecord, report: Any) -> None:
    '''Build the sole fixed raw-window contract / 构建唯一的定长 raw 窗口合同。'''

    if not state.retained:
        return
    api = _runtime_imports()
    try:
        plan = api['WindowPlan'](
            source_record_id=state.row.record_id,
            **report.window_profiles['raw_dl'],
        )
        state.raw_windows = api['build_raw_windows'](state.views, plan)
    except Exception as exc:
        state.retained = False
        state.reason = f'raw_windows_failed:{type(exc).__name__}:{exc}'
        state.route_status = 'dropped_raw_window_failure'


def _retained_states(
    states: Iterable[_RuntimeRecord],
    participant_ids: Iterable[str] | None = None,
) -> list[_RuntimeRecord]:
    '''Return deterministic retained states / 返回确定排序的保留 recording。'''

    participants = None if participant_ids is None else set(str(value) for value in participant_ids)
    return sorted(
        (
            state
            for state in states
            if state.retained
            and (participants is None or str(state.row.participant_id) in participants)
        ),
        key=lambda state: str(state.row.record_id),
    )


def _assert_train_payload_roster(
    states: Iterable[_RuntimeRecord],
    train_ids: tuple[str, ...],
    *,
    required: tuple[str, ...],
) -> None:
    '''Require every frozen train participant to reach the requested representation.'''

    observed = {
        str(state.row.participant_id)
        for state in states
        if state.retained and all(getattr(state, name) is not None for name in required)
    }
    missing = sorted(set(train_ids) - observed)
    if missing:
        raise _ExperimentProtocolError(
            'outer_train_subject_zero_representation_rows:' + ','.join(missing)
        )


def _fit_representation_artifacts(
    states: list[_RuntimeRecord],
    mode: str,
    train_ids: tuple[str, ...],
    oof_ids: tuple[str, ...],
    *,
    fitted_objects: dict[str, Any] | None = None,
) -> dict[str, Any]:
    '''Fit and apply only the transforms required by one representation mode.

    raw/fusion share one IMU artifact; matrix/fusion share one complete-vector
    artifact. Matrix additionally fits engineering rows only on outer-train.
    '''

    api = _runtime_imports()
    provenance: dict[str, Any] = {}
    selected = _retained_states(states)

    if mode in {'raw', 'fusion'}:
        _assert_train_payload_roster(states, train_ids, required=('raw_windows',))
        raw_states = [state for state in selected if state.raw_windows is not None]
        raw_values = api['np'].concatenate(
            [state.raw_windows.values for state in raw_states],
            axis=0,
        )
        raw_masks = api['np'].concatenate(
            [state.raw_windows.valid_mask for state in raw_states],
            axis=0,
        )
        raw_participants = tuple(
            str(state.row.participant_id)
            for state in raw_states
            for _ in range(state.raw_windows.values.shape[0])
        )
        imu_transform = api['fit_fold_imu_channel_transform'](
            raw_values,
            raw_participants,
            fitted_on_participant_ids=train_ids,
            outer_train_participant_ids=train_ids,
            outer_oof_participant_ids=oof_ids,
            valid_mask=raw_masks,
        )
        if fitted_objects is not None:
            fitted_objects['raw_imu'] = imu_transform
        for state in raw_states:
            state.raw_windows = api['transform_raw_windows_imu'](
                state.raw_windows,
                imu_transform,
            )
        provenance['raw_imu'] = {
            'schema_version': imu_transform.schema_version,
            'artifact_sha256': imu_transform.artifact_sha256,
            'fitted_on_participant_ids': imu_transform.fitted_on_participant_ids,
            'channel_schema': imu_transform.channel_schema,
            'valid_count': imu_transform.valid_count.tolist(),
        }

    if mode in {'feature_matrix', 'fusion'}:
        _assert_train_payload_roster(
            states,
            train_ids,
            required=('vector', 'engineering'),
        )
        vector_states = [
            state
            for state in _retained_states(states)
            if state.vector is not None and state.engineering is not None
        ]
        vector_transform = api['fit_fold_feature_vector_transform'](
            [state.vector for state in vector_states],
            [state.row.participant_id for state in vector_states],
            fitted_on_participant_ids=train_ids,
            outer_train_participant_ids=train_ids,
            outer_oof_participant_ids=oof_ids,
        )
        if fitted_objects is not None:
            fitted_objects['feature_vector'] = vector_transform
        vector_batch = api['transform_feature_vector_batch'](
            [state.vector for state in vector_states],
            vector_transform,
        )
        for state, context, features in zip(
            vector_states,
            vector_batch.contexts,
            vector_batch.fusion_tensor,
        ):
            state.transformed_vector = context
            state.fusion_features = api['np'].asarray(features, dtype=api['np'].float32)
        provenance['feature_vector'] = {
            'schema_version': vector_transform.schema_version,
            'artifact_sha256': vector_transform.artifact_sha256,
            'registry_sha256': vector_transform.registry_sha256,
            'fitted_on_participant_ids': vector_transform.fitted_on_participant_ids,
            'valid_count': vector_transform.valid_count.tolist(),
            'fusion_tensor_schema': vector_batch.tensor_schema,
        }

        if mode == 'feature_matrix':
            train_engineering = [
                state.engineering
                for state in vector_states
                if str(state.row.participant_id) in set(train_ids)
            ]
            engineering_transform = api['fit_fold_feature_transform'](
                train_engineering,
                fitted_on_participant_ids=train_ids,
                outer_train_participant_ids=train_ids,
                outer_oof_participant_ids=oof_ids,
            )
            if fitted_objects is not None:
                fitted_objects['engineering'] = engineering_transform
            engineering_payload = {
                'center': engineering_transform.center.tolist(),
                'scale': engineering_transform.scale.tolist(),
                'feature_names': engineering_transform.feature_names,
                'fitted_on_participant_ids': engineering_transform.fitted_on_participant_ids,
            }
            engineering_hash = api['stable_payload_sha256'](engineering_payload)
            for state in vector_states:
                try:
                    state.transformed_engineering = api['transform_engineering'](
                        state.engineering,
                        engineering_transform,
                    )
                    state.matrix = api['build_ordered_matrix'](
                        state.transformed_engineering,
                        context=state.transformed_vector,
                        provenance={
                            'route': state.route.value,
                            'record_id': state.row.record_id,
                            'engineering_transform_sha256': engineering_hash,
                            'feature_vector_transform_sha256': vector_transform.artifact_sha256,
                        },
                        k=32,
                    )
                except Exception as exc:
                    state.retained = False
                    state.reason = f'feature_matrix_failed:{type(exc).__name__}:{exc}'
                    state.route_status = 'dropped_feature_matrix_failure'
            provenance['engineering'] = {
                **engineering_payload,
                'artifact_sha256': engineering_hash,
                'schema_version': 'engineering_outer_train_robust_v2',
            }
            _assert_train_payload_roster(states, train_ids, required=('matrix',))

    if mode == 'fusion':
        _assert_train_payload_roster(
            states,
            train_ids,
            required=('raw_windows', 'fusion_features'),
        )
    return provenance


def _sample_identity(state: _RuntimeRecord, *, window_id: str | None = None) -> Any:
    '''Create one canonical physiological identity / 创建规范生理 role 身份。'''

    api = _runtime_imports()
    quality = (
        1.0
        if state.final_quality is None or state.final_quality.q_rate.score is None
        else float(state.final_quality.q_rate.score)
    )
    return api['SampleIdentity'](
        participant_id=str(state.row.participant_id),
        file_id=str(state.row.record_id),
        role=api['canonicalize_role_family'](str(state.row.role)),
        label=int(state.row.class_id),
        signal_route=state.route.value,
        quality_score=quality,
        window_id=window_id,
    )


def _materialize_representation_dataset(
    states: Iterable[_RuntimeRecord],
    participant_ids: Iterable[str],
    mode: str,
) -> Any:
    '''Materialise exactly one typed dataset / 物化严格类型化的数据集。'''

    api = _runtime_imports()
    selected = _retained_states(states, participant_ids)
    if mode == 'feature_vector':
        selected = [state for state in selected if state.vector is not None]
        if not selected:
            return None
        names = tuple(selected[0].vector.feature_names)
        if any(tuple(state.vector.feature_names) != names for state in selected):
            raise _ExperimentProtocolError('feature_vector_schema_drift')
        return api['FeatureVectorDataset'](
            api['np'].stack([state.vector.values for state in selected]),
            names,
            tuple(_sample_identity(state) for state in selected),
        )
    if mode == 'raw':
        selected = [state for state in selected if state.raw_windows is not None]
        if not selected:
            return None
        values = []
        masks = []
        identities = []
        for state in selected:
            for index, start_sample in enumerate(state.raw_windows.start_samples):
                values.append(state.raw_windows.values[index])
                masks.append(state.raw_windows.valid_mask[index])
                identities.append(
                    _sample_identity(
                        state,
                        window_id=(
                            f'{state.row.record_id}::start_{int(start_sample):012d}'
                        ),
                    )
                )
        return api['RawWindowDataset'](
            api['np'].stack(values),
            tuple(identities),
            api['np'].stack(masks),
        )
    if mode == 'feature_matrix':
        selected = [state for state in selected if state.matrix is not None]
        if not selected:
            return None
        return api['FeatureMatrixDataset'].from_contracts(
            tuple(state.matrix for state in selected),
            tuple(_sample_identity(state) for state in selected),
        )
    if mode == 'fusion':
        selected = [
            state
            for state in selected
            if state.raw_windows is not None and state.fusion_features is not None
        ]
        if not selected:
            return None
        return api['FileBagDataset'](
            tuple(state.raw_windows.values for state in selected),
            api['np'].stack([state.fusion_features for state in selected]),
            tuple(_sample_identity(state) for state in selected),
            tuple(state.raw_windows.valid_mask for state in selected),
        )
    raise _ExperimentProtocolError(f'unsupported_representation_mode:{mode}')


def _resolved_model_config(
    config: Any,
    *,
    training_seed: int,
) -> tuple[dict[str, Any], str]:
    '''Map archive metadata to strict factory options / 将归档字段映射为严格工厂参数。'''

    api = _runtime_imports()
    section = config.section('model')
    _, machine_id = api['normalize_model_id'](str(section['model_id']))
    ensemble_ids = {
        'inception_full_five_member_ensemble',
        'inception_matrix_five_member_ensemble',
    }
    factory_fields = {
        'compact_cnn': {
            'dropout', 'kernel_sizes', 'dilations', 'pool_sizes', 'seed',
        },
        'inception_full': {
            'dropout', 'kernel_sizes', 'dilation', 'seed',
        },
        'inception_small': {
            'dropout', 'kernel_sizes', 'dilation', 'seed',
        },
        'inception_matrix': {
            'variant', 'dropout', 'kernel_sizes', 'dilation', 'seed',
        },
        'inception_full_five_member_ensemble': {
            'comparison_only', 'member_seeds', 'dropout', 'kernel_sizes',
            'dilation',
        },
        'inception_matrix_five_member_ensemble': {
            'comparison_only', 'member_seeds', 'dropout', 'kernel_sizes',
            'dilation',
        },
        'rocket_numpy': {'n_kernels', 'alpha', 'seed'},
        'minirocket_ablation': {'n_kernels', 'alpha', 'seed'},
        'logistic_regression': {
            'class_weight', 'logistic_max_iter', 'logistic_solver', 'seed',
        },
        'rbf_svm': {
            'class_weight', 'svm_kernel', 'svm_probability', 'svm_c',
            'svm_gamma', 'seed',
        },
        'extra_trees': {
            'class_weight', 'extra_trees_n_estimators',
            'extra_trees_n_jobs', 'seed',
        },
        'fusion_compact': {
            'signal_dropout', 'signal_kernel_sizes', 'signal_dilations',
            'signal_pool_sizes', 'feature_hidden_dim', 'fusion_hidden_dim',
            'pooling', 'dropout', 'seed',
        },
        'fusion_inception': {
            'signal_variant', 'signal_dropout', 'signal_kernel_sizes',
            'signal_dilation', 'feature_hidden_dim', 'fusion_hidden_dim',
            'pooling', 'dropout', 'seed',
        },
        'shapeformer_channel_specific_osd': {
            'discovery_method', 'input_fs_hz', 'num_pip_ratio',
            'shapelets_per_class', 'max_discovery_windows',
            'discovery_balance', 'position_search_neighbourhood_samples',
            'pip_rounding_rule', 'pip_selection_rule',
            'candidate_generation_rule', 'candidate_enumeration_rule',
            'candidate_ranking_rule', 'selected_bank_order_rule',
            'discovery_position_search_boundary_rule',
            'information_gain_split_rule', 'sequence_length_samples',
            'local_kernel_width_samples', 'local_embedding_channels',
            'shape_embedding_channels', 'attention_feedforward_channels',
            'attention_heads', 'attention_query_chunk_size',
            'distance_position_chunk_size', 'dropout', 'complexity_norm',
            'max_complexity_ratio', 'seed',
        },
        'shapeformer_effect_size_fixed_v1': {
            'discovery_method', 'input_fs_hz', 'shapelet_length_samples',
            'shapelets_per_class', 'discovery_stride_samples',
            'max_candidates_per_class', 'hidden_channels', 'dropout',
            'patch_size_samples', 'attention_heads', 'attention_layers',
            'distance_position_chunk_size', 'seed',
        },
    }
    if machine_id not in factory_fields:
        raise _ExperimentProtocolError(
            f'formal_runner_has_no_explicit_model_mapping:{machine_id}'
        )
    declared_fields = set(factory_fields[machine_id])
    if machine_id not in ensemble_ids:
        declared_fields.remove('seed')
    missing = sorted(declared_fields - set(section))
    if missing:
        raise _ExperimentProtocolError(
            'model_config_missing_explicit_factory_fields:' + ','.join(missing)
        )
    if (
        machine_id not in ensemble_ids
        and section.get('seed_policy')
        != 'outer_cv_repeat_seed_equals_split_seed'
    ):
        raise _ExperimentProtocolError(
            'single_model_seed_policy_must_equal_repeat_split_seed'
        )
    if machine_id not in ensemble_ids and int(training_seed) not in {
        42, 10042, 20042, 30042, 40042
    }:
        raise _ExperimentProtocolError(
            'single_model_training_seed_not_in_frozen_repeat_seed_registry'
        )
    if machine_id in ensemble_ids:
        if (
            section.get('seed_policy')
            != 'pending_cv_repeat_member_seed_matrix_decision'
            or section.get('member_seed_roster_id')
            != 'cv_fixed_five_member_seed_roster'
        ):
            raise _ExperimentProtocolError(
                'ensemble_cv_seed_policy_or_roster_identity_invalid'
            )
        raise _ExperimentProtocolError(
            'ensemble_cv_repeat_member_seed_matrix_decision_pending'
        )
    resolved: dict[str, Any] = {'model_id': machine_id}
    for field in sorted(factory_fields[machine_id]):
        value = (
            int(training_seed)
            if field == 'seed' and machine_id not in ensemble_ids
            else section[field]
        )
        if field in {'member_seeds', 'kernel_sizes', 'dilations', 'pool_sizes',
                     'signal_kernel_sizes', 'signal_dilations',
                     'signal_pool_sizes'}:
            value = tuple(value)
        resolved[field] = value
    architecture = section.get('architecture_parameters')
    if not isinstance(architecture, Mapping) or not architecture:
        raise _ExperimentProtocolError(
            'model_config_missing_explicit_architecture_parameters'
        )
    resolved['architecture_parameters'] = dict(architecture)
    return resolved, machine_id


def _model_input_spec(dataset: Any, mode: str) -> Any:
    '''Resolve dimensions from materialised train data / 从训练数据解析输入维度。'''

    api = _runtime_imports()
    if mode == 'feature_vector':
        return api['ModelInputSpec'](
            mode,
            n_classes=3,
            feature_names=tuple(dataset.feature_names),
        )
    if mode == 'raw':
        return api['ModelInputSpec'](
            mode,
            n_channels=int(dataset.values.shape[1]),
            n_classes=3,
            channel_schema=('RED', 'IR', 'AX', 'AY', 'AZ', 'GX', 'GY', 'GZ'),
        )
    if mode == 'feature_matrix':
        return api['ModelInputSpec'](
            mode,
            n_channels=int(dataset.values.shape[1]),
            n_classes=3,
            channel_schema=tuple(dataset.channel_schema),
        )
    if mode == 'fusion':
        return api['ModelInputSpec'](
            mode,
            n_channels=int(dataset.window_bags[0].shape[1]),
            n_classes=3,
            n_file_features=int(dataset.file_features.shape[1]),
            channel_schema=('RED', 'IR', 'AX', 'AY', 'AZ', 'GX', 'GY', 'GZ'),
        )
    raise _ExperimentProtocolError(f'unsupported_representation_mode:{mode}')


@dataclass(frozen=True)
class _TrustedFull29Materialization:
    '''Internally materialised all-29 representation and immutable evidence.'''

    dataset: Any
    input_spec: Any
    fitted_objects: Mapping[str, Any]
    representation_provenance: Mapping[str, Any]
    quality_provenance: Mapping[str, Any]
    preprocessing_hash: str
    feature_hash: str
    feature_schema_id: str
    feature_contract: Mapping[str, Any]
    source_records: tuple[Mapping[str, Any], ...]
    source_records_hash: str
    dataset_hash: str
    golden_inputs: Mapping[str, Any]

    def __post_init__(self) -> None:
        from .provenance import stable_payload_sha256
        from .training import canonical_input_spec_payload, dataset_binding_hash

        participants = tuple(sorted(set(map(str, self.dataset.participant_ids))))
        if len(participants) != 29:
            raise ValueError('trusted full29 materialization requires exact 29 roster')
        source_records = tuple(dict(value) for value in self.source_records)
        if not source_records or {
            str(value.get('participant_id', '')) for value in source_records
        } != set(participants):
            raise ValueError('trusted source records must cover the exact 29 roster')
        if stable_payload_sha256(source_records) != self.source_records_hash:
            raise ValueError('trusted source-record roster hash mismatch')
        if dataset_binding_hash(self.dataset) != self.dataset_hash:
            raise ValueError('trusted all29 dataset hash mismatch')
        expected_spec = _model_input_spec(
            self.dataset,
            str(self.dataset.representation_mode),
        )
        if canonical_input_spec_payload(expected_spec) != canonical_input_spec_payload(
            self.input_spec
        ):
            raise ValueError('trusted all29 input spec differs from dataset')
        for name in (
            'preprocessing_hash', 'feature_hash', 'source_records_hash', 'dataset_hash'
        ):
            digest = str(getattr(self, name))
            if len(digest) != 64 or any(
                character not in '0123456789abcdef' for character in digest
            ):
                raise ValueError(f'{name} must be a lowercase SHA-256 digest')
        if not isinstance(self.golden_inputs, Mapping) or not self.golden_inputs:
            raise ValueError('trusted all29 materialization requires golden inputs')
        object.__setattr__(self, 'source_records', source_records)


def _golden_model_inputs(dataset: Any, mode: str) -> dict[str, Any]:
    '''Build one deterministic already-transformed model-input golden case.'''

    api = _runtime_imports()
    np = api['np']
    if mode == 'raw':
        return {
            'x': np.asarray(dataset.values[:1], dtype=np.float32),
            'mask': np.asarray(dataset.sample_mask[:1], dtype=bool),
        }
    if mode == 'feature_vector':
        return {'x': np.asarray(dataset.values[:1], dtype=np.float32)}
    if mode == 'feature_matrix':
        return {
            'x': np.asarray(dataset.values[:1], dtype=np.float32),
            'mask': np.asarray(dataset.row_mask[:1], dtype=bool),
        }
    if mode == 'fusion':
        bag = np.asarray(dataset.window_bags[0], dtype=np.float32)
        return {
            'window_bag': bag[None, :, :, :],
            'window_mask': np.ones((1, bag.shape[0]), dtype=bool),
            'file_features': np.asarray(dataset.file_features[:1], dtype=np.float32),
            'sample_mask': np.asarray(dataset.sample_masks[0], dtype=bool)[None, :, :],
        }
    raise _ExperimentProtocolError(f'unsupported_representation_mode:{mode}')


def _materialize_trusted_full29(
    report: Any,
    config: Any,
    rows: Iterable[Any],
    paths: Any,
    preflight: Mapping[str, Any],
) -> _TrustedFull29Materialization:
    '''Materialise all data only from preflight-verified manifest/source identities.'''

    from .training import dataset_binding_hash

    api = _runtime_imports()
    np = api['np']
    mode = str(config.representation_mode)
    participant_ids = tuple(sorted(map(str, preflight['participant_ids'])))
    if len(participant_ids) != 29:
        raise _ExperimentProtocolError('final_refit_materializer_requires_exact_29_roster')
    expected_source_snapshot = str(preflight['source_snapshot_sha256'])
    if _source_snapshot_sha256(paths) != expected_source_snapshot:
        raise _ExperimentProtocolError('final_refit_source_snapshot_drift_before_materialization')
    selected = _choose_records(
        rows,
        participant_ids,
        config.to_dict()['roles'],
        None,
    )
    if any(
        api['canonicalize_role_family'](str(row.role)) not in {'B', 'R'}
        for row in selected
    ):
        raise _ExperimentProtocolError('final_refit_classifier_scope_must_be_b_r_only')
    states = [_RuntimeRecord(row=row) for row in selected]
    _preprocess_records(
        states,
        config,
        None,
        lambda row, maximum: api['_load_record'](row, paths, max_samples=maximum),
    )
    quality_mode = _quality_mode(config)
    if quality_mode == 'route':
        raise _ExperimentProtocolError('final_refit_sqi_route_is_not_frozen')
    quality_provenance = _retain_without_quality_routing(
        states,
        config,
        diagnostics_only=quality_mode == 'diagnostics_only',
    )
    for state in states:
        if mode in {'feature_vector', 'feature_matrix', 'fusion'}:
            _extract_vector(state, report)
        if mode in {'raw', 'fusion'}:
            _extract_raw(state, report)
    required_by_mode = {
        'raw': ('raw_windows',),
        'feature_vector': ('vector',),
        'feature_matrix': ('vector', 'engineering'),
        'fusion': ('vector', 'engineering', 'raw_windows'),
    }
    if mode not in required_by_mode:
        raise _ExperimentProtocolError(f'unsupported_representation_mode:{mode}')
    _assert_train_payload_roster(
        states,
        participant_ids,
        required=required_by_mode[mode],
    )
    fitted_objects: dict[str, Any] = {}
    representation_provenance = _fit_representation_artifacts(
        states,
        mode,
        participant_ids,
        (),
        fitted_objects=fitted_objects,
    )
    if mode == 'feature_vector':
        representation_provenance = {
            'feature_vector_estimator_pipeline': {
                'status': 'fitted_inside_final_all29_estimator',
                'external_representation_transform': 'not_applicable',
                'fitted_on_participant_ids': participant_ids,
            }
        }
    dataset = _materialize_representation_dataset(states, participant_ids, mode)
    if dataset is None:
        raise _ExperimentProtocolError('final_refit_all29_dataset_empty')
    dl_case_id = config.section('signal')['dl_resampling'].get('case_id')
    if dl_case_id is not None:
        if mode != 'raw':
            raise _ExperimentProtocolError(
                'fixed_kernel_samples_profile_requires_raw_representation'
            )
        from .models.time_scale import prepare_fixed_kernel_dl_input

        values, mask, profile = prepare_fixed_kernel_dl_input(
            dataset.values,
            dataset.sample_mask,
            str(dl_case_id),
            source_fs_hz=400.0,
        )
        dataset = api['RawWindowDataset'](values, dataset.identities, mask)
        representation_provenance = {
            **dict(representation_provenance),
            'fixed_kernel_samples': {
                **profile,
                'application_scope': 'dl_input_only_after_canonical_400hz_windows',
                'canonical_features_and_peaks_unchanged': True,
                'final_all29_same_frozen_transform_identity': True,
            },
        }
    observed_participants = set(map(str, dataset.participant_ids))
    if observed_participants != set(participant_ids):
        missing = sorted(set(participant_ids) - observed_participants)
        raise _ExperimentProtocolError(
            'final_refit_all29_dataset_roster_incomplete:' + ','.join(missing)
        )
    if set(map(int, np.asarray(dataset.labels))) != {0, 1, 2}:
        raise _ExperimentProtocolError('final_refit_all29_dataset_missing_class')

    source_records: list[dict[str, Any]] = []
    for state in sorted(states, key=lambda value: str(value.row.record_id)):
        source_records.append(
            {
                'record_id': str(state.row.record_id),
                'participant_id': str(state.row.participant_id),
                'role': str(state.row.role),
                'class_id': int(state.row.class_id),
                'source_path': str(state.row.source_path),
                'source_hash': str(state.row.source_hash),
                'source_version': str(state.row.source_version),
                'fs_hz': float(state.row.fs),
                'n_samples': int(state.row.n_samples),
                'retained': bool(state.retained),
                'route_status': str(state.route_status),
                'rejection_reason': state.reason,
                'physical_qc_source_byte_identity': dict(
                    state.physical_qc_evidence.get('source_byte_identity', {})
                ),
            }
        )
    source_records_tuple = tuple(source_records)
    source_records_hash = api['stable_payload_sha256'](source_records_tuple)
    if _source_snapshot_sha256(paths) != expected_source_snapshot:
        raise _ExperimentProtocolError('final_refit_source_snapshot_drift_after_materialization')

    registry = api['default_registry']()
    input_spec = _model_input_spec(dataset, mode)
    if mode == 'feature_vector':
        feature_schema_id = registry.schema_version
        feature_contract: dict[str, Any] = {
            'registry_hash': registry.sha256,
            'feature_names': tuple(dataset.feature_names),
            'estimator_transforms': 'fit_inside_final_all29_estimator_pipeline',
        }
    elif mode == 'raw':
        feature_schema_id = 'raw_red_ir_imu_8ch_all29_scaled_v2'
        feature_contract = {
            'channel_schema': tuple(input_spec.channel_schema),
            'raw_imu_transform': representation_provenance.get('raw_imu'),
            'fixed_kernel_samples': representation_provenance.get(
                'fixed_kernel_samples', {'status': 'not_applied'}
            ),
        }
    elif mode == 'feature_matrix':
        first = _retained_states(states, participant_ids)[0]
        feature_schema_id = str(first.matrix.schema_version)
        feature_contract = {
            'channel_schema': tuple(dataset.channel_schema),
            'transforms': representation_provenance,
            'matrix_k': int(dataset.values.shape[2]),
        }
    else:
        feature_schema_id = 'fusion_raw_bag_plus_vector_validity_v2'
        feature_contract = {
            'channel_schema': tuple(input_spec.channel_schema),
            'file_feature_width': int(input_spec.n_file_features),
            'transforms': representation_provenance,
        }
    feature_hash = api['stable_payload_sha256'](feature_contract)
    preprocessing_hash = api['stable_payload_sha256'](
        {
            'signal': config.section('signal'),
            'quality': config.section('quality'),
            'artifact': config.section('artifact'),
            'quality_mode': quality_mode,
            'quality_provenance': quality_provenance,
            'representation_transforms': representation_provenance,
            'physical_recording_qc_profile': tuple(
                state.physical_qc_profile for state in states
            ),
            'source_records_hash': source_records_hash,
            'fit_scope': 'verified_all29_participants',
        }
    )
    dataset_hash = dataset_binding_hash(dataset)
    return _TrustedFull29Materialization(
        dataset=dataset,
        input_spec=input_spec,
        fitted_objects=fitted_objects,
        representation_provenance=representation_provenance,
        quality_provenance=quality_provenance,
        preprocessing_hash=preprocessing_hash,
        feature_hash=feature_hash,
        feature_schema_id=feature_schema_id,
        feature_contract=feature_contract,
        source_records=source_records_tuple,
        source_records_hash=source_records_hash,
        dataset_hash=dataset_hash,
        golden_inputs=_golden_model_inputs(dataset, mode),
    )


def _resolved_architecture_parameters(
    model_id: str,
    model_section: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact declaration already checked by the strict model factory."""

    declared = model_section.get('architecture_parameters')
    if not isinstance(declared, Mapping) or not declared:
        raise _ExperimentProtocolError(
            'model_config_missing_explicit_architecture_parameters'
        )
    if str(declared.get('model_id')) != model_id:
        raise _ExperimentProtocolError(
            'architecture_parameters_model_id_mismatch'
        )
    return dict(declared)


def _code_commit(paths: Any) -> str:
    '''读取 git commit / Read the source commit.'''

    import subprocess
    completed = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=paths.repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if completed.returncode != 0 or len(value) != 40:
        raise _ExperimentProtocolError('code_commit_unavailable')
    return value


def _git_source_state(paths: Any) -> dict[str, Any]:
    """Describe whether the active V2 bytes are completely tracked and clean."""

    from .scientific_gate import ScientificGateError, git_source_state

    try:
        return git_source_state(paths.repository_root, paths.pipeline_root)
    except ScientificGateError as exc:
        raise _ExperimentProtocolError(str(exc)) from exc


def _dependency_gate(
    config: Any,
    paths: Any,
    *,
    operation: str,
    require_exact_lock: bool,
) -> dict[str, Any]:
    """Resolve dependency eligibility from this config, never all optional lines."""

    from .config import dependency_gate_report

    return dependency_gate_report(
        config,
        operation=operation,
        profiles_path=paths.pipeline_root / "requirements/profiles.json",
        lock_path=paths.pipeline_root / "locks/profiles.lock.json",
        require_exact_lock=require_exact_lock,
    )


def _capture_formal_execution_gate(
    config: Any,
    paths: Any,
    *,
    operation: str,
) -> dict[str, Any]:
    """Capture one source/dependency authority without an internal TOCTOU gap."""

    from .provenance import stable_payload_sha256
    from .config import load_config

    config_source = Path(config.source_path).resolve()
    configs_root = (paths.pipeline_root / 'configs').resolve()
    try:
        config_relative = config_source.relative_to(configs_root)
    except ValueError as exc:
        raise _ExperimentProtocolError(
            'formal_config_must_resolve_inside_tracked_configs_directory'
        ) from exc
    if not config_source.is_file():
        raise _ExperimentProtocolError('formal_config_source_missing')
    config_source_bytes = config_source.read_bytes()
    config_source_sha256 = __import__('hashlib').sha256(
        config_source_bytes
    ).hexdigest()
    if load_config(config_source).sha256 != config.sha256:
        raise _ExperimentProtocolError(
            'formal_loaded_config_differs_from_canonical_source'
        )
    source_state_before = _require_scientific_source_gate(paths)
    source_snapshot_before = _source_snapshot_sha256(paths)
    dependency_gate = _dependency_gate(
        config,
        paths,
        operation=operation,
        require_exact_lock=True,
    )
    source_state_after = _require_scientific_source_gate(paths)
    source_snapshot_after = _source_snapshot_sha256(paths)
    if (
        stable_payload_sha256(source_state_before)
        != stable_payload_sha256(source_state_after)
        or source_snapshot_before != source_snapshot_after
        or config_source.read_bytes() != config_source_bytes
        or load_config(config_source).sha256 != config.sha256
    ):
        raise _ExperimentProtocolError(
            'formal_execution_source_changed_during_gate_capture'
        )
    return {
        'schema_version': 'ppg_frailty.formal_execution_gate_capture.v2',
        'pipeline_generation': 'final_pipeline_v2',
        'operation': str(operation),
        'config_id': str(config.config_id),
        'config_hash': str(config.sha256),
        'config_source_relative_path':
            (Path('configs') / config_relative).as_posix(),
        'config_source_file_sha256': config_source_sha256,
        'source_tree_state': source_state_before,
        'source_snapshot_sha256': source_snapshot_before,
        'dependency_gate': dependency_gate,
        'dependency_gate_sha256': stable_payload_sha256(dependency_gate),
    }


def _formal_gate_checkpoint_record(
    phase: str,
    capture: Mapping[str, Any],
) -> dict[str, Any]:
    from .provenance import stable_payload_sha256

    record = {
        'phase': str(phase),
        'capture_sha256': stable_payload_sha256(capture),
        'source_snapshot_sha256': str(capture['source_snapshot_sha256']),
        'dependency_gate_sha256': str(capture['dependency_gate_sha256']),
    }
    return {**record, 'checkpoint_sha256': stable_payload_sha256(record)}


@dataclass
class _FormalExecutionGateSession:
    """TOCTOU-resistant baseline used by every formal V2 producer."""

    config: Any
    paths: Any
    operation: str
    baseline: dict[str, Any]
    baseline_sha256: str
    checkpoints: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def establish(
        cls,
        config: Any,
        paths: Any,
        *,
        operation: str,
    ) -> '_FormalExecutionGateSession':
        from .provenance import stable_payload_sha256

        baseline = _capture_formal_execution_gate(
            config,
            paths,
            operation=operation,
        )
        session = cls(
            config=config,
            paths=paths,
            operation=str(operation),
            baseline=baseline,
            baseline_sha256=stable_payload_sha256(baseline),
        )
        session.checkpoints.append(
            _formal_gate_checkpoint_record('entry_preflight', baseline)
        )
        return session

    def checkpoint(self, phase: str) -> dict[str, Any]:
        from .provenance import stable_payload_sha256

        normalized = str(phase).strip()
        if not normalized or any(
            row['phase'] == normalized for row in self.checkpoints
        ):
            raise _ExperimentProtocolError(
                'formal_execution_gate_checkpoint_invalid_or_duplicate'
            )
        current = _capture_formal_execution_gate(
            self.config,
            self.paths,
            operation=self.operation,
        )
        if stable_payload_sha256(current) != self.baseline_sha256:
            raise _ExperimentProtocolError(
                f'formal_execution_gate_changed_at_{normalized}'
            )
        record = _formal_gate_checkpoint_record(normalized, current)
        self.checkpoints.append(record)
        return dict(record)

    def evidence(self) -> dict[str, Any]:
        from .provenance import stable_payload_sha256

        payload = {
            'schema_version': 'ppg_frailty.formal_execution_gate_evidence.v2',
            'pipeline_generation': 'final_pipeline_v2',
            'baseline': dict(self.baseline),
            'baseline_sha256': self.baseline_sha256,
            'checkpoints': [dict(row) for row in self.checkpoints],
        }
        return {
            **payload,
            'formal_execution_gate_evidence_sha256':
                stable_payload_sha256(payload),
        }

    def assert_current(self, phase: str) -> dict[str, Any]:
        """Recheck immediately before an atomic publish without mutating evidence."""

        from .provenance import stable_payload_sha256

        current = _capture_formal_execution_gate(
            self.config,
            self.paths,
            operation=self.operation,
        )
        if stable_payload_sha256(current) != self.baseline_sha256:
            raise _ExperimentProtocolError(
                f'formal_execution_gate_changed_at_{str(phase).strip()}'
            )
        return _formal_gate_checkpoint_record(str(phase), current)


def _require_scientific_source_gate(paths: Any) -> dict[str, Any]:
    """Forbid scientific full runs from an untracked or dirty V2 tree."""

    from .scientific_gate import (
        ScientificGateError,
        require_clean_tracked_source_state,
    )

    state = _git_source_state(paths)
    try:
        return require_clean_tracked_source_state(state)
    except ScientificGateError as exc:
        raise _ExperimentProtocolError(str(exc)) from exc


def _source_snapshot_sha256(paths: Any) -> str:
    '''Hash active V2 implementation inputs / 哈希当前 V2 实现输入。'''

    from .scientific_gate import ScientificGateError, source_snapshot_sha256

    try:
        return source_snapshot_sha256(paths.pipeline_root)
    except ScientificGateError as exc:
        raise _ExperimentProtocolError(str(exc)) from exc


def _make_oof(
    states: list[_RuntimeRecord],
    oof_ids: tuple[str, ...],
    prediction_identities: Iterable[Any],
    probabilities: Any,
    common: dict[str, Any],
    *,
    balance_line: str,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
    '''Build window/file/participant OOF while retaining every rejected OOF file.'''

    api = _runtime_imports()
    identities = tuple(prediction_identities)
    values = api['np'].asarray(probabilities, dtype=api['np'].float64)
    if values.shape != (len(identities), 3):
        raise _ExperimentProtocolError('oof_probability_shape_mismatch')
    heldout = set(oof_ids)
    state_by_file = {
        str(state.row.record_id): state
        for state in states
        if str(state.row.participant_id) in heldout
    }
    if not state_by_file:
        raise _ExperimentProtocolError('outer_oof_has_no_selected_records')

    prediction_rows = []
    for identity, probability in zip(identities, values):
        if str(identity.participant_id) not in heldout:
            raise _ExperimentProtocolError('non_oof_prediction_reached_oof_writer')
        state = state_by_file.get(str(identity.file_id))
        if state is None:
            raise _ExperimentProtocolError(
                f'prediction_file_not_in_selected_oof:{identity.file_id}'
            )
        level = 'window' if identity.window_id is not None else 'file'
        prediction_rows.append(
            api['OofPredictionRow'](
                participant_id=str(identity.participant_id),
                file_id=str(identity.file_id),
                role=api['canonicalize_role_family'](str(identity.role)),
                label=int(identity.label),
                probabilities=tuple(float(value) for value in probability),
                signal_route=str(identity.signal_route),
                quality_score=float(identity.quality_score),
                retained=True,
                level=level,
                window_id=identity.window_id,
                artifact_reducer_name=state.artifact_name,
                artifact_reducer_version=state.artifact_version,
                route_status=state.route_status,
                rejection_reason=None,
                **common,
            )
        )
    source_levels = {row.level for row in prediction_rows}
    if len(source_levels) > 1:
        raise _ExperimentProtocolError('mixed_window_and_file_prediction_levels')
    window_mode = source_levels == {'window'}

    predicted_files = {row.file_id for row in prediction_rows}
    dropped_file_rows = []
    for state in sorted(state_by_file.values(), key=lambda item: str(item.row.record_id)):
        if str(state.row.record_id) in predicted_files:
            continue
        route = state.route or state.intended_route or api['SignalRoute'].DROPPED
        dropped_common = dict(common)
        dropped_common['class_order'] = ()
        dropped_file_rows.append(
            api['OofPredictionRow'](
                participant_id=str(state.row.participant_id),
                file_id=str(state.row.record_id),
                role=api['canonicalize_role_family'](str(state.row.role)),
                label=int(state.row.class_id),
                probabilities=(),
                signal_route=route.value,
                quality_score=(
                    1.0
                    if state.final_quality is None
                    or state.final_quality.q_rate.score is None
                    else float(state.final_quality.q_rate.score)
                ),
                retained=False,
                level='file',
                artifact_reducer_name=state.artifact_name,
                artifact_reducer_version=state.artifact_version,
                route_status=state.route_status,
                rejection_reason=state.reason or 'no_representation_prediction',
                **dropped_common,
            )
        )

    source_rows = tuple((*prediction_rows, *dropped_file_rows))
    hierarchy = api['aggregate_hierarchy'](
        source_rows,
        balance_line=balance_line,
    )
    if window_mode:
        retained_file_rows = {row.file_id: row for row in hierarchy.file_rows}
        dropped_by_file = {row.file_id: row for row in dropped_file_rows}
        file_rows = tuple(
            retained_file_rows.get(record_id, dropped_by_file.get(record_id))
            for record_id in sorted(state_by_file)
        )
        if any(row is None for row in file_rows):
            raise _ExperimentProtocolError('raw_window_to_file_oof_incomplete')
        window_rows = tuple(
            sorted(
                prediction_rows,
                key=lambda row: (
                    row.participant_id,
                    row.file_id,
                    str(row.window_id),
                ),
            )
        )
    else:
        duplicate_files = [row.file_id for row in prediction_rows]
        if len(duplicate_files) != len(set(duplicate_files)):
            raise _ExperimentProtocolError('duplicate_file_level_prediction')
        file_rows = tuple(
            sorted(
                source_rows,
                key=lambda row: (row.participant_id, row.file_id),
            )
        )
        window_rows = ()

    subject_rows = list(hierarchy.participant_rows)
    observed = {row.participant_id for row in subject_rows}
    for participant in sorted(heldout - observed):
        candidates = [row for row in file_rows if row.participant_id == participant]
        if not candidates:
            raise _ExperimentProtocolError(f'heldout_subject_has_no_record:{participant}')
        routes = {row.signal_route for row in candidates}
        if len(routes) != 1:
            raise _ExperimentProtocolError(f'dropped_subject_mixed_routes:{participant}')
        subject_rows.append(
            api['replace'](
                candidates[0],
                file_id=f'participant::{participant}',
                role='participant',
                probabilities=(),
                retained=False,
                level='participant',
                class_order=(),
                window_id=None,
                route_status='dropped_no_participant_probability',
                rejection_reason='all_selected_files_dropped',
            )
        )
    return (
        window_rows,
        tuple(file_rows),
        tuple(sorted(subject_rows, key=lambda row: row.participant_id)),
    )


def _require_retained_oof(subject_rows: Iterable[Any]) -> None:
    '''Reject a cell with no usable participant prediction / 拒绝零可用参与者预测。'''

    if not any(bool(row.retained) for row in subject_rows):
        raise _ExperimentProtocolError('outer_oof_zero_retained_predictions')


def _evaluate_subjects(subject_rows: tuple[Any, ...], total: int) -> dict[str, Any]:
    '''计算 participant metrics / Compute participant-unit metrics.'''

    api = _runtime_imports()
    retained = [row for row in subject_rows if row.retained]
    if not retained:
        return {
            'status': 'unavailable_no_retained_participant',
            'n_total': total,
            'n_retained': 0,
            'coverage_rate': 0.0,
        }
    metrics = api['evaluate_predictions'](
        api['np'].asarray([row.label for row in retained]),
        api['np'].asarray([row.probabilities for row in retained]),
        class_order=(0, 1, 2),
        n_total=total,
    )
    return to_strict_json_value(asdict(metrics))


def _batch1_operational_model_input(
    dataset: Any,
    *,
    mode: str,
    estimator: bool,
) -> Any:
    """Materialize one already-preprocessed model input without timing preprocessing."""

    if estimator:
        values = dataset.values[:1]
        if hasattr(dataset, 'row_mask'):
            return {'x': values, 'mask': dataset.row_mask[:1]}
        if hasattr(dataset, 'sample_mask'):
            return {'x': values, 'mask': dataset.sample_mask[:1]}
        return values
    if mode == 'fusion':
        import torch

        bag = torch.from_numpy(dataset.window_bags[0]).unsqueeze(0)
        sample_mask = torch.from_numpy(dataset.sample_masks[0]).unsqueeze(0)
        return {
            'window_bag': bag,
            'window_mask': torch.ones((1, bag.shape[1]), dtype=torch.bool),
            'file_features': torch.from_numpy(dataset.file_features[:1]),
            'sample_mask': sample_mask,
        }
    sample = dataset[0]
    output = {'x': sample['x'].unsqueeze(0)}
    if 'mask' in sample:
        output['mask'] = sample['mask'].unsqueeze(0)
    return output


def _execute_cell_unchecked(
    report: Any,
    config: Any,
    rows: Iterable[Any],
    registry: Any,
    paths: Any,
    *,
    repeat_index: int,
    fold_index: int,
    maximum_seconds: float | None,
    record_cap: int | None,
    epoch_override: int | None,
    measure_operational_costs: bool = False,
    loader: Any = None,
) -> _CellResult:
    '''Execute one representation-aware frozen outer cell.'''

    api = _runtime_imports()
    started = time.perf_counter()
    mode = str(config.representation_mode)
    if mode not in {'raw', 'feature_vector', 'feature_matrix', 'fusion'}:
        raise _ExperimentProtocolError(f'unsupported_representation_mode:{mode}')
    split = registry.get_split(repeat_index, fold_index)
    train_ids = tuple(str(value) for value in split['train_participant_ids'])
    oof_ids = tuple(str(value) for value in split['oof_participant_ids'])
    selected = _choose_records(
        rows,
        (*train_ids, *oof_ids),
        config.to_dict()['roles'],
        record_cap,
    )
    states = [_RuntimeRecord(row=row) for row in selected]
    actual_loader = loader or (
        lambda row, maximum: api['_load_record'](row, paths, max_samples=maximum)
    )
    _preprocess_records(states, config, maximum_seconds, actual_loader)
    quality_mode = _quality_mode(config)
    if quality_mode == 'route':
        sqi_config, calibrator = _fit_quality_calibrator(
            states,
            config,
            train_ids,
            oof_ids,
        )
        _route_records(states, config, report, sqi_config, calibrator)
        quality_provenance = {
            'method': calibrator.method,
            'fitted_on_participant_ids': calibrator.fitted_on_participant_ids,
            'outer_oof_ids_absent': not bool(
                set(calibrator.fitted_on_participant_ids) & set(oof_ids)
            ),
            'classification_effect': 'routing',
        }
    else:
        quality_provenance = _retain_without_quality_routing(
            states,
            config,
            diagnostics_only=quality_mode == 'diagnostics_only',
        )

    for state in states:
        if mode in {'feature_vector', 'feature_matrix', 'fusion'}:
            _extract_vector(state, report)
        if mode in {'raw', 'fusion'}:
            _extract_raw(state, report)

    required_by_mode = {
        'raw': ('raw_windows',),
        'feature_vector': ('vector',),
        'feature_matrix': ('vector', 'engineering'),
        'fusion': ('vector', 'engineering', 'raw_windows'),
    }
    _assert_train_payload_roster(
        states,
        train_ids,
        required=required_by_mode[mode],
    )
    representation_provenance = _fit_representation_artifacts(
        states,
        mode,
        train_ids,
        oof_ids,
    )
    train_dataset = _materialize_representation_dataset(states, train_ids, mode)
    oof_dataset = _materialize_representation_dataset(states, oof_ids, mode)
    if train_dataset is None:
        raise _ExperimentProtocolError('outer_train_dataset_empty')
    dl_case_id = config.section('signal')['dl_resampling'].get('case_id')
    if dl_case_id is not None:
        if mode != 'raw' or oof_dataset is None:
            raise _ExperimentProtocolError(
                'fixed_kernel_samples_profile_requires_nonempty_raw_train_and_oof'
            )
        from .models.time_scale import prepare_fixed_kernel_dl_input

        train_values, train_mask, train_profile = prepare_fixed_kernel_dl_input(
            train_dataset.values,
            train_dataset.sample_mask,
            str(dl_case_id),
            source_fs_hz=400.0,
        )
        oof_values, oof_mask, oof_profile = prepare_fixed_kernel_dl_input(
            oof_dataset.values,
            oof_dataset.sample_mask,
            str(dl_case_id),
            source_fs_hz=400.0,
        )
        if train_profile != oof_profile:
            raise _ExperimentProtocolError('fixed_kernel_train_oof_profile_drift')
        train_dataset = api['RawWindowDataset'](
            train_values,
            train_dataset.identities,
            train_mask,
        )
        oof_dataset = api['RawWindowDataset'](
            oof_values,
            oof_dataset.identities,
            oof_mask,
        )
        representation_provenance['fixed_kernel_samples'] = {
            **train_profile,
            'application_scope': 'dl_input_only_after_canonical_400hz_windows',
            'canonical_features_and_peaks_unchanged': True,
            'outer_train_and_oof_same_transform_identity': True,
        }
    observed_train = set(str(value) for value in train_dataset.participant_ids)
    if observed_train != set(train_ids):
        missing = sorted(set(train_ids) - observed_train)
        raise _ExperimentProtocolError(
            'outer_train_dataset_roster_incomplete:' + ','.join(missing)
        )
    observed_classes = set(
        int(value) for value in api['np'].asarray(train_dataset.labels)
    )
    if observed_classes != {0, 1, 2}:
        raise _ExperimentProtocolError(
            'outer_train_dataset_missing_class:'
            + ','.join(str(value) for value in sorted({0, 1, 2} - observed_classes))
        )

    frozen = api['FrozenOuterSplit'](
        repeat=int(split['repeat_index']),
        fold=int(split['fold_index']),
        seed=int(split['split_seed']),
        train_participant_ids=train_ids,
        oof_participant_ids=oof_ids,
        registry_hash=str(
            config.section('splits')['source_registry_payload_sha256']
        ),
        fold_hash=report.fold_hash,
    )
    training_config = api['TrainingConfig'].from_mapping(
        config.section('training')
    )
    training_config = api['replace'](
        training_config,
        seed=int(split['training_seed']),
    )
    if epoch_override is not None:
        if training_config.epoch_rule != 'fixed_epoch' or epoch_override <= 0:
            raise _ExperimentProtocolError('invalid_fixed_epoch_override')
        training_config = api['replace'](
            training_config,
            execution_mode='smoke',
            epoch_profile='smoke',
            fixed_epochs=int(epoch_override),
        )
    balance_line = str(
        config.section('aggregation').get(
            'balance_line',
            'line_a_equal_files',
        )
    )
    if training_config.expected_aggregation_rule != balance_line:
        raise _ExperimentProtocolError(
            'training_and_aggregation_balance_line_mismatch'
        )

    model_config, model_id = _resolved_model_config(
        config,
        training_seed=int(split['training_seed']),
    )
    input_spec = _model_input_spec(train_dataset, mode)
    model_section = config.section('model')
    if int(model_section.get('n_classes', 3)) != 3:
        raise _ExperimentProtocolError('model_declared_class_count_must_be_three')
    declared_channels = int(model_section.get('input_channels', 0))
    if declared_channels > 0 and declared_channels != int(input_spec.n_channels):
        raise _ExperimentProtocolError(
            f'model_input_channel_mismatch:{declared_channels}:'
            f'{int(input_spec.n_channels)}'
        )
    ensemble_ids = {
        'inception_full_five_member_ensemble',
        'inception_matrix_five_member_ensemble',
    }
    is_ensemble = model_id in ensemble_ids
    expected_ensemble_size = 5 if is_ensemble else 1
    if int(model_section.get('ensemble_size', 1)) != expected_ensemble_size:
        raise _ExperimentProtocolError(
            f'model_ensemble_size_mismatch:{expected_ensemble_size}'
        )
    prepared = api['prepare_model_factory'](
        model_config,
        input_spec,
        train_dataset,
        frozen,
    )
    trainer = api['UnifiedTrainer'](training_config)
    estimator_ids = _ESTIMATOR_MODEL_IDS
    if model_id in estimator_ids:
        training = trainer.fit_estimator(
            prepared(),
            train_dataset,
            frozen,
        )
        if oof_dataset is None:
            probabilities = api['np'].empty((0, 3), dtype=api['np'].float64)
            prediction_identities: tuple[Any, ...] = ()
        else:
            probabilities, _, prediction_identities = (
                trainer.predict_estimator_probabilities(
                    training.model,
                    oof_dataset,
                )
            )
            classes = tuple(int(value) for value in training.model.classes_)
            if set(classes) != {0, 1, 2}:
                raise _ExperimentProtocolError(
                    f'trained_model_missing_class:{classes}'
                )
            if classes != (0, 1, 2):
                probabilities = probabilities[
                    :,
                    [classes.index(value) for value in (0, 1, 2)],
                ]
    else:
        training = trainer.fit(
            prepared,
            train_dataset,
            frozen,
        )
        if oof_dataset is None:
            probabilities = api['np'].empty((0, 3), dtype=api['np'].float64)
            member_probabilities = None
            prediction_identities = ()
        elif is_ensemble:
            (
                member_probabilities,
                probabilities,
                _,
                prediction_identities,
            ) = trainer.predict_ensemble_members(
                training.model,
                oof_dataset,
            )
        else:
            member_probabilities = None
            probabilities, _, prediction_identities = (
                trainer.predict_probabilities(
                    training.model,
                    oof_dataset,
                )
            )
    if training.provenance is None:
        raise _ExperimentProtocolError('training_provenance_missing')

    if measure_operational_costs:
        if oof_dataset is None or len(oof_dataset) == 0:
            raise _ExperimentProtocolError(
                'operational_measurement_requires_nonempty_oof_model_input'
            )
        model_input = _batch1_operational_model_input(
            oof_dataset,
            mode=mode,
            estimator=model_id in estimator_ids,
        )
        operational_metrics = {
            'status': 'measured_explicit_cpu_batch1_request',
            **to_strict_json_value(
                asdict(
                    api['measure_cpu_batch1_operational_metrics'](
                        training.model,
                        model_input,
                    )
                )
            ),
        }
    else:
        operational_metrics = {
            'status': 'not_requested',
            'parameter_count': None,
            'inference_cost': {
                'cpu_batch1_model_only_p50_ms': None,
                'cpu_batch1_model_only_p95_ms': None,
            },
        }

    feature_registry = api['default_registry']()
    if mode == 'feature_vector':
        feature_schema_id = feature_registry.schema_version
        feature_contract = {
            'registry_hash': feature_registry.sha256,
            'feature_names': tuple(train_dataset.feature_names),
            'estimator_transforms': 'fit_inside_outer_train_pipeline',
        }
    elif mode == 'raw':
        feature_schema_id = 'raw_red_ir_imu_8ch_outer_train_scaled_v2'
        feature_contract = {
            'channel_schema': tuple(input_spec.channel_schema),
            'raw_imu_transform': representation_provenance.get('raw_imu'),
        }
    elif mode == 'feature_matrix':
        feature_schema_id = str(
            _retained_states(states, train_ids)[0].matrix.schema_version
        )
        feature_contract = {
            'channel_schema': tuple(train_dataset.channel_schema),
            'transforms': representation_provenance,
            'matrix_k': int(train_dataset.values.shape[2]),
        }
    else:
        feature_schema_id = 'fusion_raw_bag_plus_vector_validity_v2'
        feature_contract = {
            'channel_schema': tuple(input_spec.channel_schema),
            'file_feature_width': int(input_spec.n_file_features),
            'transforms': representation_provenance,
        }

    preprocessing_hash = api['stable_payload_sha256'](
        {
            'signal': config.section('signal'),
            'quality': config.section('quality'),
            'artifact': config.section('artifact'),
            'quality_mode': quality_mode,
            'quality_provenance': quality_provenance,
            'representation_transforms': representation_provenance,
            'physical_recording_qc_profile': tuple(
                state.physical_qc_profile for state in states
            ),
        }
    )
    feature_hash = api['stable_payload_sha256'](feature_contract)
    training_section = config.section('training')
    window_section = config.section('windows')
    signal_section = config.section('signal')
    quality_section = config.section('quality')
    aggregation_section = config.section('aggregation')
    evaluation_section = config.section('evaluation')
    if mode == 'feature_vector':
        input_channels_order = tuple(str(value) for value in train_dataset.feature_names)
    else:
        input_channels_order = tuple(str(value) for value in input_spec.channel_schema)
    selected_window = (
        window_section['engineering']
        if mode in {'feature_vector', 'feature_matrix'}
        else window_section['raw_dl']
    )
    model_input_fs_hz = _model_input_sampling_rate_hz(config)
    canonical_signal_fs_hz = float(signal_section['internal_fs_hz'])
    if canonical_signal_fs_hz != 400.0:
        raise _ExperimentProtocolError('canonical_signal_grid_must_remain_400hz')
    training_algorithm = _training_algorithm_provenance(
        model_id,
        training_section,
        model_section,
        fixed_epochs=int(training_config.fixed_epochs),
    )
    frozen_run_provenance = api['validate_frozen_model_run_provenance'](
        {
            'architecture_parameters': _resolved_architecture_parameters(
                model_id,
                model_section,
            ),
            'input_channels_order': input_channels_order,
            'sampling_rate_hz': model_input_fs_hz,
            'window_plan': {
                'representation_mode': mode,
                'selected': dict(selected_window),
                'shared_planner_version': window_section['shared_planner_version'],
                'model_input_sampling_rate_hz': model_input_fs_hz,
                'canonical_signal_and_feature_sampling_rate_hz':
                    canonical_signal_fs_hz,
            },
            'hop_plan': {
                'hop_s': float(selected_window['hop_s']),
                'end_alignment': selected_window['end_alignment'],
            },
            'normalization': dict(signal_section['normalization']),
            'padding_mask': {
                'padding': selected_window['padding'],
                'min_valid_fraction': selected_window.get('min_valid_fraction', 1.0),
                'mask_aware_pooling': model_section.get(
                    'mask_aware_pooling',
                    'not_applicable',
                ),
            },
            'feature_schema_hash': feature_hash,
            'sqi_routing': {
                'mode': quality_section['mode'],
                'supervised_route_ready': quality_section['supervised_route_ready'],
                'failure_action': quality_section['failure_action'],
            },
            **training_algorithm,
            'random_seeds': (
                tuple(int(value) for value in model_section['member_seeds'])
                if is_ensemble
                else (int(split['training_seed']),)
            ),
            'seed_policy': (
                'cv_fixed_five_member_seed_roster'
                if is_ensemble
                else 'outer_cv_repeat_seed_equals_split_seed'
            ),
            'fold_hash': report.fold_hash,
            'aggregation': dict(aggregation_section),
            'calibration': {
                'metrics': tuple(evaluation_section['calibration_metrics']),
                'fit_scope': 'outer_training_participants_only',
            },
        }
    )
    frozen_run_provenance_hash = api['stable_payload_sha256'](
        frozen_run_provenance
    )
    environment = api['runtime_environment']()
    source_snapshot_sha256 = _source_snapshot_sha256(paths)
    common = {
        'repeat': int(split['repeat_index']),
        'fold': int(split['fold_index']),
        'split_seed': int(split['split_seed']),
        'training_seed': int(split['training_seed']),
        'config_hash': config.sha256,
        'manifest_hash': report.manifest_hash,
        'fold_hash': report.fold_hash,
        'preprocessing_hash': preprocessing_hash,
        'feature_hash': feature_hash,
        'model_hash': training.provenance.state_hash,
        'representation_mode': mode,
        'class_order': (0, 1, 2),
        'code_commit': _code_commit(paths),
        'data_schema_id': str(
            getattr(
                states[0].row,
                'manifest_version',
                config.section('manifest')['manifest_version'],
            )
        ),
        'feature_schema_id': feature_schema_id,
        'model_version': str(model_section['variant']),
        'aggregation_rule': balance_line,
        'environment_hash': api['stable_payload_sha256'](environment),
        'manifest_version': str(config.section('manifest')['manifest_version']),
        'fold_registry_version': str(config.section('splits')['registry_id']),
        'source_snapshot_hash': source_snapshot_sha256,
    }
    member_rows: tuple[Any, ...] = ()
    if is_ensemble:
        if member_probabilities is None:
            raise _ExperimentProtocolError('ensemble_member_probabilities_missing')
        member_seeds = tuple(int(value) for value in model_section['member_seeds'])
        base_model_id = (
            'inception_full'
            if model_id == 'inception_full_five_member_ensemble'
            else 'inception_matrix'
        )
        average_common = dict(common)
        average_common.update(
            {
                'training_seed': None,
                'prediction_kind': 'ensemble_average',
                'member_training_seeds': member_seeds,
                'ensemble_base_model_id': base_model_id,
            }
        )
        window_rows, file_rows, subject_rows = _make_oof(
            states,
            oof_ids,
            prediction_identities,
            probabilities,
            average_common,
            balance_line=balance_line,
        )
        member_subject_rows: list[Any] = []
        member_hashes = tuple(training.provenance.member_state_hashes)
        if len(member_hashes) != 5 or len(member_seeds) != 5:
            raise _ExperimentProtocolError('ensemble_member_provenance_incomplete')
        for member_index, (member_seed, member_hash) in enumerate(
            zip(member_seeds, member_hashes)
        ):
            member_common = dict(common)
            member_common.update(
                {
                    'training_seed': member_seed,
                    'member_index': member_index,
                    'prediction_kind': 'ensemble_member',
                    'ensemble_base_model_id': base_model_id,
                    'model_hash': str(member_hash),
                }
            )
            _, _, current_subject_rows = _make_oof(
                states,
                oof_ids,
                prediction_identities,
                member_probabilities[member_index],
                member_common,
                balance_line=balance_line,
            )
            member_subject_rows.extend(current_subject_rows)
        member_rows = tuple(member_subject_rows)
    else:
        window_rows, file_rows, subject_rows = _make_oof(
            states,
            oof_ids,
            prediction_identities,
            probabilities,
            common,
            balance_line=balance_line,
        )
    api['validate_expected_oof_roster'](
        (*member_rows, *subject_rows),
        {
            (
                int(split['repeat_index']),
                int(split['fold_index']),
                int(split['split_seed']),
            ): oof_ids
        },
        expected_config_hashes=(config.sha256,),
        expected_member_count=5 if is_ensemble else 1,
    )
    _require_retained_oof(subject_rows)
    metrics = _evaluate_subjects(subject_rows, len(oof_ids))
    summary = {
        'schema_version': 'ppg_frailty.experiment_cell.v2',
        'pipeline_generation': 'final_pipeline_v2',
        'status': 'passed',
        'runner_status': 'outer_fold_execution_completed',
        'repeat_index': int(split['repeat_index']),
        'fold_index': int(split['fold_index']),
        'split_seed': int(split['split_seed']),
        'training_seed': (
            None if is_ensemble else int(split['training_seed'])
        ),
        'member_training_seeds': (
            list(int(value) for value in model_section['member_seeds'])
            if is_ensemble
            else []
        ),
        'seed_policy': (
            'cv_fixed_five_member_seed_roster'
            if is_ensemble
            else 'outer_cv_repeat_seed_equals_split_seed'
        ),
        'representation_mode': mode,
        'model_id': str(model_section['model_id']),
        'model_machine_id': model_id,
        'class_order': [0, 1, 2],
        'selected_record_count': len(states),
        'retained_train_record_count': sum(
            state.retained and state.row.participant_id in set(train_ids)
            for state in states
        ),
        'retained_oof_record_count': sum(
            state.retained and state.row.participant_id in set(oof_ids)
            for state in states
        ),
        'oof_window_prediction_count': len(window_rows),
        'dropped_records': [
            {
                'record_id': state.row.record_id,
                'participant_id': state.row.participant_id,
                'reason': state.reason,
                'route_status': state.route_status,
            }
            for state in states
            if not state.retained
        ],
        'metrics': metrics,
        'operational_metrics': operational_metrics,
        'fitted_provenance': to_strict_json_value(asdict(training.provenance)),
        'model_factory_provenance': to_strict_json_value(
            dict(prepared.provenance)
        ),
        'frozen_model_run_provenance': to_strict_json_value(
            frozen_run_provenance
        ),
        'frozen_model_run_provenance_hash': frozen_run_provenance_hash,
        'representation_transform_provenance': to_strict_json_value(
            representation_provenance
        ),
        'quality_mode': quality_mode,
        'sqi_calibrator_provenance': quality_provenance,
        'quality_diagnostics': [
            {
                'record_id': state.row.record_id,
                'components': state.diagnostic_components,
                'reason': state.diagnostic_reason,
            }
            for state in states
            if state.diagnostic_components or state.diagnostic_reason is not None
        ],
        'physical_recording_qc': [
            {
                'record_id': state.row.record_id,
                'evidence': state.physical_qc_evidence,
                'profile': state.physical_qc_profile,
            }
            for state in states
        ],
        'balance_line': balance_line,
        'preprocessing_hash': preprocessing_hash,
        'feature_hash': feature_hash,
        'model_hash': training.provenance.state_hash,
        'code_commit': common['code_commit'],
        'source_snapshot_sha256': source_snapshot_sha256,
        'elapsed_seconds': time.perf_counter() - started,
    }
    return _CellResult(
        summary,
        file_rows,
        subject_rows,
        window_rows=window_rows,
        member_rows=member_rows,
    )


def _execute_cell(*args: Any, **kwargs: Any) -> _CellResult:
    '''Fail closed with a representation-specific machine-readable reason.'''

    config = args[1] if len(args) > 1 else kwargs.get('config')
    mode = str(getattr(config, 'representation_mode', 'unknown'))
    try:
        return _execute_cell_unchecked(*args, **kwargs)
    except _ExperimentProtocolError:
        raise
    except Exception as exc:
        raise _ExperimentProtocolError(
            f'{mode}_outer_cell_failed:{type(exc).__name__}:{exc}'
        ) from exc


def _execute_vector_cell(*args: Any, **kwargs: Any) -> _CellResult:
    '''Backward-compatible strict feature-vector helper for existing callers.'''

    config = args[1] if len(args) > 1 else kwargs.get('config')
    if str(getattr(config, 'representation_mode', '')) != 'feature_vector':
        raise _ExperimentProtocolError(
            'execute_vector_cell_requires_feature_vector'
        )
    return _execute_cell(*args, **kwargs)


def _strict_json(path: Path, payload: Mapping[str, Any]) -> None:
    '''Write strict, deterministic JSON without silently accepting NaN/Infinity.

    严格、确定性地写入 JSON；拒绝 NaN/Infinity，避免下游审计得到非标准文件。
    '''
    from .provenance import atomic_write_json

    if path.exists():
        raise FileExistsError(f'artifact_overwrite_forbidden:{path}')
    atomic_write_json(path, dict(payload), root=path.parent)


def _write_empty_oof(path: Path, reason: str) -> None:
    '''Materialize a schema-bearing empty parquet for a deliberately absent level.

    为当前表示层级不产生的 OOF 写入带模式的空 parquet，而不是伪造预测。
    '''
    if path.exists():
        raise FileExistsError(f'artifact_overwrite_forbidden:{path}')
    try:
        from .training import write_empty_oof_parquet

        write_empty_oof_parquet(path, reason)
    except (ImportError, RuntimeError) as exc:  # pragma: no cover - dependency guard
        if 'pyarrow' in str(exc).lower():
            raise _ExperimentProtocolError('pyarrow_required_for_formal_oof') from exc
        raise


def _write_artifact_index(directory: Path) -> None:
    """Hash every completed artifact recursively without indexing itself."""

    from .provenance import sha256_file, stable_payload_sha256

    target = directory / 'artifact_index.json'
    if target.exists():
        raise FileExistsError(f'artifact_overwrite_forbidden:{target}')
    rows = [
        {
            'path': path.relative_to(directory).as_posix(),
            'bytes': path.stat().st_size,
            'sha256': sha256_file(path),
        }
        for path in sorted(directory.rglob('*'))
        if path.is_file() and path != target
    ]
    _strict_json(
        target,
        {
            'schema_version': 'ppg_frailty.artifact_index.v2',
            'pipeline_generation': 'final_pipeline_v2',
            'artifact_count': len(rows),
            'artifacts': rows,
            'index_payload_sha256': stable_payload_sha256(rows),
            'overwrite_existing': False,
        },
    )


def _write_cell_artifacts(directory: Path, cell: _CellResult) -> None:
    '''Write the six mandatory, non-overwriting artifacts for one outer cell.

    为单个 outer cell 写入六个强制产物；目录必须预先不存在，从而禁止覆盖。
    '''
    imports = _runtime_imports()
    writer = imports['OofWriter']()
    directory.mkdir(parents=True, exist_ok=False)
    if cell.window_rows:
        writer.write(cell.window_rows, directory / 'oof_window_predictions.parquet')
    else:
        _write_empty_oof(
            directory / 'oof_window_predictions.parquet',
            'representation_predictions_begin_at_file_level',
        )
    writer.write(cell.file_rows, directory / 'oof_file_predictions.parquet')
    writer.write(cell.subject_rows, directory / 'oof_subject_predictions.parquet')
    if cell.member_rows:
        writer.write(cell.member_rows, directory / 'oof_member_predictions.parquet')
    else:
        _write_empty_oof(
            directory / 'oof_member_predictions.parquet',
            'single_model_runner_ensemble_comparison_not_executed',
        )
    _strict_json(
        directory / 'quality_diagnostics.json',
        {
            'schema_version': 'ppg_frailty.quality_diagnostics.v2',
            'quality_mode': cell.summary['quality_mode'],
            'classification_effect': (
                'routing' if cell.summary['quality_mode'] == 'route' else 'none'
            ),
            'rows': cell.summary['quality_diagnostics'],
        },
    )
    _strict_json(
        directory / 'metrics_per_fold_seed.json',
        {
            'schema_version': 'ppg_frailty.metrics_per_fold_seed.v2',
            'pipeline_generation': 'final_pipeline_v2',
            'cells': [dict(cell.summary)],
        },
    )
    _strict_json(
        directory / 'confusion_matrices.json',
        {
            'schema_version': 'ppg_frailty.confusion_matrices.v2',
            'pipeline_generation': 'final_pipeline_v2',
            'cells': [{
                'repeat_index': cell.summary['repeat_index'],
                'fold_index': cell.summary['fold_index'],
                'class_order': cell.summary['class_order'],
                'confusion_matrix': cell.summary['metrics']['confusion_matrix'],
            }],
        },
    )
    _strict_json(
        directory / 'run_manifest.json',
        {
            'schema_version': 'ppg_frailty.run_manifest.v2',
            'pipeline_generation': 'final_pipeline_v2',
            'status': 'passed',
            'scientific_scope': cell.summary['scientific_scope'],
            'cell': dict(cell.summary),
            'mandatory_artifacts': [
                'run_manifest.json',
                'metrics_per_fold_seed.json',
                'confusion_matrices.json',
                'oof_window_predictions.parquet',
                'oof_file_predictions.parquet',
                'oof_subject_predictions.parquet',
                'oof_member_predictions.parquet',
                'quality_diagnostics.json',
                'artifact_index.json',
            ],
        },
    )


def _write_failed_artifacts(directory: Path, result: ExperimentResult) -> None:
    '''Persist an explicit fail-closed result with empty, never-fabricated OOF.

    失败时仍落盘完整契约，但 OOF 保持为空，绝不伪造预测或静默回退。
    '''
    directory.mkdir(parents=True, exist_ok=False)
    reason = ';'.join(result.failure_reasons) or 'failed_closed'
    _write_empty_oof(directory / 'oof_window_predictions.parquet', reason)
    _write_empty_oof(directory / 'oof_file_predictions.parquet', reason)
    _write_empty_oof(directory / 'oof_subject_predictions.parquet', reason)
    _write_empty_oof(directory / 'oof_member_predictions.parquet', reason)
    _strict_json(
        directory / 'quality_diagnostics.json',
        {
            'schema_version': 'ppg_frailty.quality_diagnostics.v2',
            'status': 'failed_closed',
            'classification_effect': 'none',
            'rows': [],
        },
    )
    _strict_json(
        directory / 'metrics_per_fold_seed.json',
        {
            'schema_version': 'ppg_frailty.metrics_per_fold_seed.v2',
            'pipeline_generation': 'final_pipeline_v2',
            'status': 'failed_closed',
            'cells': [],
        },
    )
    _strict_json(
        directory / 'confusion_matrices.json',
        {
            'schema_version': 'ppg_frailty.confusion_matrices.v2',
            'pipeline_generation': 'final_pipeline_v2',
            'status': 'failed_closed',
            'cells': [],
        },
    )
    failed_manifest = result.to_dict()
    failed_manifest['schema_version'] = 'ppg_frailty.run_manifest.v2'
    failed_manifest['pipeline_generation'] = 'final_pipeline_v2'
    _strict_json(directory / 'run_manifest.json', failed_manifest)


def _resolve_output_directory(paths: Any, requested: Any, default_name: str) -> Path:
    '''Resolve a write target through PipelinePaths and reject overwrites.

    通过 PipelinePaths 解析写入位置，并以 fail-closed 方式拒绝覆盖已有目录。
    '''
    raw = requested if requested is not None else Path('artifacts') / default_name
    target = paths.output_path(raw)
    if target.exists():
        raise FileExistsError(f'experiment_output_exists:{target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _require_controlled_formal_artifact_target(paths: Any, target: Path) -> Path:
    """Limit in-tree formal outputs to the one source-gate-excluded namespace."""

    resolved = target.resolve()
    pipeline_root = paths.pipeline_root.resolve()
    try:
        relative = resolved.relative_to(pipeline_root)
    except ValueError:
        return resolved
    if not relative.parts or relative.parts[0] != 'artifacts':
        raise ValueError(
            'formal_output_inside_pipeline_must_be_under_artifacts'
        )
    return resolved


def _commit_staging(staging: Path, target: Path) -> None:
    '''Atomically publish a completed experiment directory on one filesystem.

    在同一文件系统内原子发布完整实验目录，避免消费者看到半成品。
    '''
    staging.replace(target)


def run_reduced_fold_experiment(
    config_path: str | Path,
    *,
    repeat_index: int = 0,
    fold_index: int = 0,
    max_seconds_per_record: float = 60.0,
    max_records_per_participant: int = 1,
    fixed_epochs_override: int = 1,
    output_dir: str | Path | None = None,
) -> ExperimentResult:
    '''Execute one real frozen outer fold as a non-scientific smoke run.

    中文：完整保留冻结 participant roster，仅允许缩短每条记录、每人文件数和
    epoch；所有 SQI 拟合和特征变换仍严格局限在 outer-train。
    '''
    if max_seconds_per_record < 10.0:
        raise ValueError('reduced smoke requires at least ten seconds per record')
    if max_records_per_participant <= 0 or fixed_epochs_override <= 0:
        raise ValueError('record and epoch limits must be positive')
    if repeat_index not in range(5) or fold_index not in range(5):
        raise ValueError('repeat_index and fold_index must lie in 0..4')

    api = _runtime_imports()
    paths = api['PipelinePaths'].discover()
    report, config, rows, registry = api['preflight_pipeline'](
        config_path,
        mode='full',
        paths=paths,
    )
    dependency_gate = _dependency_gate(
        config,
        paths,
        operation='reduced_smoke',
        require_exact_lock=False,
    )
    source_tree_state = _git_source_state(paths)
    scope = 'smoke_not_scientific_benchmark'
    default_name = (
        f'reduced_r{int(repeat_index)}_f{int(fold_index)}_{time.time_ns()}'
    )
    target = _resolve_output_directory(paths, output_dir, default_name)
    staging = target.with_name(f'.{target.name}.staging.{time.time_ns()}')
    try:
        try:
            cell = _execute_cell(
                report,
                config,
                rows,
                registry,
                paths,
                repeat_index=int(repeat_index),
                fold_index=int(fold_index),
                maximum_seconds=float(max_seconds_per_record),
                record_cap=int(max_records_per_participant),
                epoch_override=int(fixed_epochs_override),
            )
            cell.summary['scientific_scope'] = scope
            result = ExperimentResult(
                status='passed',
                scientific_scope=scope,
                config_id=config.config_id,
                config_hash=config.sha256,
                repeat_indices=(int(repeat_index),),
                fold_indices=(int(fold_index),),
                output_dir=str(target),
                cell_results=(dict(cell.summary),),
                metrics=dict(cell.summary['metrics']),
                provenance={
                    'preflight_status': report.status,
                    'manifest_hash': report.manifest_hash,
                    'fold_hash': report.fold_hash,
                    'quality_mode': str(config.section('quality').get('mode', 'off')),
                    'outer_train_only_calibrator': (
                        str(config.section('quality').get('mode', 'off')) == 'route'
                    ),
                    'frozen_outer_split': True,
                    'record_seconds_cap': float(max_seconds_per_record),
                    'record_cap_per_participant': int(max_records_per_participant),
                    'fixed_epochs_override': int(fixed_epochs_override),
                    'source_snapshot_sha256': cell.summary['source_snapshot_sha256'],
                    'source_tree_state': source_tree_state,
                    'dependency_gate': dependency_gate,
                },
            )
            _write_cell_artifacts(staging, cell)
        except _ExperimentProtocolError as exc:
            result = ExperimentResult(
                status='failed_closed',
                scientific_scope=scope,
                config_id=config.config_id,
                config_hash=config.sha256,
                repeat_indices=(int(repeat_index),),
                fold_indices=(int(fold_index),),
                output_dir=str(target),
                provenance={
                    'preflight_status': report.status,
                    'manifest_hash': report.manifest_hash,
                    'fold_hash': report.fold_hash,
                    'frozen_outer_split': True,
                    'source_snapshot_sha256': _source_snapshot_sha256(paths),
                    'source_tree_state': source_tree_state,
                    'dependency_gate': dependency_gate,
                },
                failure_reasons=(str(exc),),
            )
            _write_failed_artifacts(staging, result)
        _strict_json(staging / 'experiment_result.json', result.to_dict())
        _write_artifact_index(staging)
        _commit_staging(staging, target)
        return result
    except Exception:
        # Only the uniquely named staging directory is recoverable scratch.
        # 仅清理本次唯一命名的暂存目录，不触碰任何既有实验结果。
        if staging.exists():
            import shutil
            shutil.rmtree(staging)
        raise


def _write_full_root_artifacts(
    directory: Path,
    cells: Iterable[_CellResult],
    result: ExperimentResult,
) -> None:
    '''Write cross-cell formal artifacts at the experiment root.

    中文：合并各 cell 的 OOF，但保留 repeat/fold/seed 键；失败 cell 不会被
    合成为假预测，根 manifest 明确保持 failed_closed。
    '''
    api = _runtime_imports()
    writer = api['OofWriter']()
    cell_values = tuple(cells)
    window_rows = tuple(row for cell in cell_values for row in cell.window_rows)
    file_rows = tuple(row for cell in cell_values for row in cell.file_rows)
    subject_rows = tuple(row for cell in cell_values for row in cell.subject_rows)
    member_rows = tuple(row for cell in cell_values for row in cell.member_rows)
    if window_rows:
        writer.write(window_rows, directory / 'oof_window_predictions.parquet')
    else:
        _write_empty_oof(
            directory / 'oof_window_predictions.parquet',
            'no_window_level_predictions_in_successful_outer_cells',
        )
    if file_rows:
        writer.write(file_rows, directory / 'oof_file_predictions.parquet')
    else:
        _write_empty_oof(
            directory / 'oof_file_predictions.parquet',
            'no_successful_outer_cells',
        )
    if subject_rows:
        writer.write(subject_rows, directory / 'oof_subject_predictions.parquet')
    else:
        _write_empty_oof(
            directory / 'oof_subject_predictions.parquet',
            'no_successful_outer_cells',
        )
    if member_rows:
        writer.write(member_rows, directory / 'oof_member_predictions.parquet')
    else:
        _write_empty_oof(
            directory / 'oof_member_predictions.parquet',
            'single_model_runner_ensemble_comparison_not_executed',
        )
    _strict_json(
        directory / 'quality_diagnostics.json',
        {
            'schema_version': 'ppg_frailty.quality_diagnostics.v2',
            'status': result.status,
            'cells': [
                {
                    'repeat_index': cell.summary['repeat_index'],
                    'fold_index': cell.summary['fold_index'],
                    'quality_mode': cell.summary['quality_mode'],
                    'rows': cell.summary['quality_diagnostics'],
                }
                for cell in cell_values
            ],
        },
    )
    _strict_json(
        directory / 'metrics_per_fold_seed.json',
        {
            'schema_version': 'ppg_frailty.metrics_per_fold_seed.v2',
            'pipeline_generation': 'final_pipeline_v2',
            'status': result.status,
            'cells': [dict(cell.summary) for cell in cell_values],
        },
    )
    _strict_json(
        directory / 'confusion_matrices.json',
        {
            'schema_version': 'ppg_frailty.confusion_matrices.v2',
            'pipeline_generation': 'final_pipeline_v2',
            'status': result.status,
            'cells': [
                {
                    'repeat_index': cell.summary['repeat_index'],
                    'fold_index': cell.summary['fold_index'],
                    'class_order': cell.summary['class_order'],
                    'confusion_matrix': cell.summary['metrics']['confusion_matrix'],
                }
                for cell in cell_values
            ],
        },
    )
    _strict_json(
        directory / 'config_metrics_v2.json',
        _trusted_config_metrics_payload(cell_values, result),
    )
    manifest = result.to_dict()
    manifest['schema_version'] = 'ppg_frailty.run_manifest.v2'
    manifest['pipeline_generation'] = 'final_pipeline_v2'
    manifest['mandatory_artifacts'] = [
        'run_manifest.json',
        'metrics_per_fold_seed.json',
        'confusion_matrices.json',
        'oof_window_predictions.parquet',
        'oof_file_predictions.parquet',
        'oof_subject_predictions.parquet',
        'oof_member_predictions.parquet',
        'quality_diagnostics.json',
        'config_metrics_v2.json',
        'artifact_index.json',
    ]
    _strict_json(directory / 'run_manifest.json', manifest)


_FORMAL_SPLIT_SEEDS = (42, 10042, 20042, 30042, 40042)
_FORMAL_ABLATION_MACHINE_IDS = frozenset(
    {'minirocket_ablation', 'shapeformer_effect_size_fixed_v1'}
)
_FORMAL_COMPARISON_MACHINE_IDS = frozenset(
    {
        'inception_full_five_member_ensemble',
        'inception_matrix_five_member_ensemble',
    }
)
_FORMAL_REFERENCE_MACHINE_IDS = frozenset(
    {
        'compact_cnn',
        'inception_full',
        'inception_small',
        'inception_matrix',
        'rocket_numpy',
        'logistic_regression',
        'rbf_svm',
        'extra_trees',
        'shapeformer_channel_specific_osd',
        'fusion_compact',
        'fusion_inception',
    }
)


def _registry_role_for_machine_id(machine_id: str) -> str:
    """Map all 15 formal catalog identities to immutable provenance roles."""

    value = str(machine_id)
    if value in _FORMAL_REFERENCE_MACHINE_IDS:
        return 'reference'
    if value in _FORMAL_ABLATION_MACHINE_IDS:
        return 'ablation'
    if value in _FORMAL_COMPARISON_MACHINE_IDS:
        return 'comparison'
    raise _ExperimentProtocolError(f'model_not_in_formal_catalog:{value}')


def _participant_predictions_from_subject_rows(
    rows: Iterable[Any],
) -> tuple[Any, ...]:
    """Convert exact retained participant OOF rows to statistics records."""

    ParticipantPrediction = _runtime_imports()['ParticipantPrediction']
    output: list[Any] = []
    keys: set[tuple[str, int]] = set()
    for row in rows:
        if row.level != 'participant':
            raise _ExperimentProtocolError('root_subject_oof_contains_nonparticipant_row')
        if not row.retained:
            raise _ExperimentProtocolError(
                f'complete_5x5_metrics_require_retained_subject:{row.participant_id}'
            )
        if row.prediction_kind not in {'single_model', 'ensemble_average'}:
            raise _ExperimentProtocolError(
                'root_subject_oof_contains_ensemble_member_or_unknown_kind'
            )
        if tuple(row.class_order) != (0, 1, 2):
            raise _ExperimentProtocolError('root_subject_oof_class_order_drift')
        key = (str(row.participant_id), int(row.repeat))
        if key in keys:
            raise _ExperimentProtocolError(
                f'duplicate_participant_repeat_oof:{key[0]}:{key[1]}'
            )
        keys.add(key)
        output.append(
            ParticipantPrediction(
                participant_id=key[0],
                label=int(row.label),
                repeat=key[1],
                probabilities=tuple(float(value) for value in row.probabilities),
            )
        )
    return tuple(sorted(output, key=lambda item: (item.participant_id, item.repeat)))


def _fold_confusions_and_rosters_from_subject_rows(
    rows: Iterable[Any],
) -> tuple[dict[str, tuple[tuple[int, ...], ...]], dict[str, tuple[str, ...]]]:
    """Rebuild exact 25-fold confusion/coverage from trusted subject OOF."""

    matrices = {
        f"r{repeat}f{fold}": [[0, 0, 0] for _ in range(3)]
        for repeat in range(5)
        for fold in range(5)
    }
    rosters = {key: [] for key in matrices}
    seen: set[tuple[str, int]] = set()
    for row in rows:
        key = f"r{int(row.repeat)}f{int(row.fold)}"
        identity = (str(row.participant_id), int(row.repeat))
        if key not in matrices or identity in seen:
            raise _ExperimentProtocolError("subject_oof_fold_identity_invalid_or_duplicate")
        if (
            not row.retained
            or tuple(row.class_order) != (0, 1, 2)
            or len(row.probabilities) != 3
        ):
            raise _ExperimentProtocolError("trusted_fold_metrics_require_retained_three_class_oof")
        seen.add(identity)
        predicted = max(range(3), key=lambda index: float(row.probabilities[index]))
        matrices[key][int(row.label)][predicted] += 1
        rosters[key].append(str(row.participant_id))
    return (
        {
            key: tuple(tuple(int(value) for value in row) for row in matrix)
            for key, matrix in matrices.items()
        },
        {key: tuple(sorted(values)) for key, values in rosters.items()},
    )


def _operational_metrics_from_fold_summaries(
    summaries: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Rebuild eligibility and CPU batch-1 cost from all 25 cell summaries."""

    import math

    frozen = tuple(summaries)
    if len(frozen) != 25:
        raise _ExperimentProtocolError(
            'operational_metrics_require_exact_25_cell_summaries'
        )
    rows = tuple(summary.get('operational_metrics', {}) for summary in frozen)
    measured_status = 'measured_explicit_cpu_batch1_request'
    statuses = tuple(
        row.get('status') if isinstance(row, Mapping) else None for row in rows
    )
    any_measured = any(status == measured_status for status in statuses)
    operational_ready = all(
        isinstance(row, Mapping)
        and row.get('status') == measured_status
        and row.get('parameter_count') is not None
        and row.get('model_latency_p50_ms') is not None
        and row.get('model_latency_p95_ms') is not None
        for row in rows
    )
    if any_measured and not operational_ready:
        raise _ExperimentProtocolError(
            'operational_metrics_partial_or_incomplete_across_25_cells'
        )
    if operational_ready:
        parameter_counts = tuple(int(row['parameter_count']) for row in rows)
        p50_values = tuple(float(row['model_latency_p50_ms']) for row in rows)
        p95_values = tuple(float(row['model_latency_p95_ms']) for row in rows)
        if (
            any(value < 0 for value in parameter_counts)
            or any(not math.isfinite(value) or value < 0.0 for value in p50_values)
            or any(not math.isfinite(value) or value < 0.0 for value in p95_values)
            or any(p50 > p95 for p50, p95 in zip(p50_values, p95_values))
        ):
            raise _ExperimentProtocolError('operational_metrics_invalid')
        return {
            'parameter_count': max(parameter_counts),
            'parameter_counts_by_cell': parameter_counts,
            'inference_cost': {
                'cpu_batch1_model_only_p50_ms_mean_across_25_outer_cells':
                    sum(p50_values) / len(p50_values),
                'cpu_batch1_model_only_p95_ms_mean_across_25_outer_cells':
                    sum(p95_values) / len(p95_values),
            },
            'eligible': True,
            'exclusion_reason': '',
            'status': 'measured_all_25_cells_explicit_request',
            'measurement_requested': True,
        }
    if any(status not in {'not_requested', None} for status in statuses):
        raise _ExperimentProtocolError('operational_metrics_unknown_cell_status')
    return {
        'parameter_count': None,
        'parameter_counts_by_cell': (),
        'inference_cost': {'participant_probability_latency_ms': None},
        'eligible': False,
        'exclusion_reason': 'parameter_count_and_inference_cost_not_measured',
        'status': 'not_measured_in_current_runner',
        'measurement_requested': False,
    }


def _trusted_config_metrics_payload(
    cells: Iterable[_CellResult],
    result: ExperimentResult,
    *,
    n_bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Build root metrics only from typed OOF and completed fold summaries.

    Operational measurements are intentionally missing until the runner measures
    them. This keeps the configuration visible for review while making automatic
    eligibility fail closed.
    """

    cell_values = tuple(cells)
    base = {
        'schema_version': 'ppg_frailty.config_metrics.v2',
        'pipeline_generation': 'final_pipeline_v2',
        'config_id': result.config_id,
        'config_hash': result.config_hash,
        'independent_test': False,
        'fold_protocol': 'frozen_repeated_grouped_5x5',
        'automatic_selection': False,
    }
    complete = (
        result.status == 'passed'
        and result.scientific_scope == 'frozen_5x5_scientific_benchmark'
        and len(cell_values) == 25
        and {(int(cell.summary['repeat_index']), int(cell.summary['fold_index']))
             for cell in cell_values} == {(repeat, fold) for repeat in range(5) for fold in range(5)}
    )
    if not complete:
        return {
            **base,
            'status': 'not_available_requires_complete_passed_5x5',
            'config_metrics': None,
            'bootstrap_results': [],
            'training_executed_by_this_writer': False,
        }

    summaries = tuple(cell.summary for cell in cell_values)
    if any(summary.get('status') != 'passed' for summary in summaries):
        raise _ExperimentProtocolError('complete_metrics_received_nonpassed_cell')
    machine_ids = {str(summary['model_machine_id']) for summary in summaries}
    if len(machine_ids) != 1:
        raise _ExperimentProtocolError('root_cells_mix_model_machine_ids')
    machine_id = next(iter(machine_ids))
    split_seed_by_repeat: dict[int, int] = {}
    for summary in summaries:
        repeat = int(summary['repeat_index'])
        split_seed = int(summary['split_seed'])
        previous = split_seed_by_repeat.setdefault(repeat, split_seed)
        if previous != split_seed:
            raise _ExperimentProtocolError('split_seed_varies_within_repeat')
    if tuple(split_seed_by_repeat[index] for index in range(5)) != _FORMAL_SPLIT_SEEDS:
        raise _ExperimentProtocolError('formal_split_seed_sequence_drift')

    predictions = _participant_predictions_from_subject_rows(
        row for cell in cell_values for row in cell.subject_rows
    )
    participants = tuple(sorted({row.participant_id for row in predictions}))
    if len(participants) != 29 or len(predictions) != 29 * 5:
        raise _ExperimentProtocolError(
            f'complete_metrics_require_29x5_participant_oof:{len(participants)}:{len(predictions)}'
        )
    if {row.repeat for row in predictions} != set(range(5)):
        raise _ExperimentProtocolError('complete_metrics_repeat_roster_drift')

    fold_ba = {
        f"r{int(summary['repeat_index'])}f{int(summary['fold_index'])}":
            float(summary['metrics']['balanced_accuracy'])
        for summary in summaries
    }
    fold_confusions, fold_rosters = _fold_confusions_and_rosters_from_subject_rows(
        row for cell in cell_values for row in cell.subject_rows
    )
    for summary in summaries:
        key = f"r{int(summary['repeat_index'])}f{int(summary['fold_index'])}"
        if to_strict_json_value(summary['metrics']['confusion_matrix']) != (
            to_strict_json_value(fold_confusions[key])
        ):
            raise _ExperimentProtocolError(f'cell_confusion_matrix_oof_drift:{key}')
    operational = _operational_metrics_from_fold_summaries(summaries)

    metrics, bootstrap = _runtime_imports()[
        'build_config_metrics_from_predictions_and_fold_summaries'
    ](
        config_id=result.config_id,
        registry_role=_registry_role_for_machine_id(machine_id),
        predictions=predictions,
        fold_balanced_accuracies=fold_ba,
        fold_confusion_matrices=fold_confusions,
        fold_participant_rosters=fold_rosters,
        inference_cost=operational['inference_cost'],
        parameter_count=operational['parameter_count'],
        n_bootstrap_resamples=int(n_bootstrap_resamples),
        bootstrap_seed=int(bootstrap_seed),
        eligible=operational['eligible'],
        exclusion_reason=operational['exclusion_reason'],
    )
    return {
        **base,
        'status': 'passed_trusted_metrics_rebuilt_from_typed_oof',
        'model_machine_id': machine_id,
        'registry_role': metrics.registry_role,
        'seeds': list(_FORMAL_SPLIT_SEEDS),
        'training_seeds': (
            [42, 10042, 20042, 30042, 40042]
            if machine_id in _FORMAL_COMPARISON_MACHINE_IDS
            else list(_FORMAL_SPLIT_SEEDS)
        ),
        'participant_oof_coverage': {
            'participant_count': len(participants),
            'repeat_count': 5,
            'participant_repeat_rows': len(predictions),
            'expected_participant_repeat_rows': 145,
            'roster_complete': True,
        },
        'fold_balanced_accuracies': fold_ba,
        'operational_measurement_status': operational['status'],
        'operational_aggregation': {
            'parameter_count_rule': (
                'maximum_across_25_outer_fitted_models'
                if operational['measurement_requested'] else 'not_available'
            ),
            'parameter_counts_by_cell': list(
                operational['parameter_counts_by_cell']
            ),
            'latency_rule': (
                'arithmetic_mean_of_per_cell_fixed_cpu_batch1_p50_and_p95'
                if operational['measurement_requested'] else 'not_available'
            ),
            'measurement_requested': operational['measurement_requested'],
        },
        'config_metrics': to_strict_json_value(asdict(metrics)),
        'bootstrap_results': to_strict_json_value([asdict(value) for value in bootstrap]),
        'bootstrap_policy': {
            'resamples': int(n_bootstrap_resamples),
            'seed': int(bootstrap_seed),
            'unit': 'participant_with_all_repeats',
        },
        'training_executed_by_this_writer': False,
    }


def run_full_experiment(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    repeats: Iterable[int] = tuple(range(5)),
    folds: Iterable[int] = tuple(range(5)),
    measure_operational_costs: bool = False,
    confirm_scientific_execution: bool = False,
) -> ExperimentResult:
    '''Execute unshortened frozen outer cells, including the complete 5x5 grid.

    中文：本入口禁止记录截短、每人文件数裁剪和 epoch override；每个 cell
    独立训练，任何 fail-closed cell 都会使根结果失败，同时继续留下可审计证据。
    '''
    if not confirm_scientific_execution:
        raise PermissionError(
            'full experiment requires explicit scientific execution confirmation'
        )
    repeat_values = tuple(int(value) for value in repeats)
    fold_values = tuple(int(value) for value in folds)
    if (
        not repeat_values
        or not fold_values
        or len(set(repeat_values)) != len(repeat_values)
        or len(set(fold_values)) != len(fold_values)
        or not set(repeat_values) <= set(range(5))
        or not set(fold_values) <= set(range(5))
    ):
        raise ValueError('repeats and folds must be unique non-empty subsets of 0..4')
    scope = (
        'frozen_5x5_scientific_benchmark'
        if set(repeat_values) == set(range(5))
        and set(fold_values) == set(range(5))
        else 'selected_full_length_cells_not_complete_5x5'
    )
    api = _runtime_imports()
    paths = api['PipelinePaths'].discover()
    report, config, rows, registry = api['preflight_pipeline'](
        config_path,
        mode='full',
        paths=paths,
    )
    gate_session = _FormalExecutionGateSession.establish(
        config,
        paths,
        operation='formal_benchmark',
    )
    target = _require_controlled_formal_artifact_target(
        paths,
        _resolve_output_directory(paths, output_dir, 'full_experiment'),
    )
    staging = target.with_name(f'.{target.name}.staging.{time.time_ns()}')
    staging.mkdir(parents=True, exist_ok=False)
    passed_cells: list[_CellResult] = []
    summaries: list[dict[str, Any]] = []
    failures: list[str] = []
    started = time.perf_counter()
    try:
        for repeat_index in repeat_values:
            for fold_index in fold_values:
                cell_directory = staging / (
                    f'repeat_{repeat_index:02d}_fold_{fold_index:02d}'
                )
                phase = f'repeat_{repeat_index:02d}_fold_{fold_index:02d}'
                gate_pre = gate_session.checkpoint(f'{phase}_pre')
                try:
                    cell = _execute_cell(
                        report,
                        config,
                        rows,
                        registry,
                        paths,
                        repeat_index=repeat_index,
                        fold_index=fold_index,
                        maximum_seconds=None,
                        record_cap=None,
                        epoch_override=None,
                        measure_operational_costs=bool(measure_operational_costs),
                    )
                    gate_post = gate_session.checkpoint(f'{phase}_post')
                    cell.summary['scientific_scope'] = scope
                    cell.summary['formal_execution_gate_checkpoints'] = {
                        'pre': gate_pre,
                        'post': gate_post,
                    }
                    passed_cells.append(cell)
                    summaries.append(dict(cell.summary))
                    _write_cell_artifacts(cell_directory, cell)
                    _write_artifact_index(cell_directory)
                except _ExperimentProtocolError as exc:
                    gate_post = gate_session.checkpoint(f'{phase}_post')
                    failure = f'r{repeat_index}_f{fold_index}:{exc}'
                    failures.append(failure)
                    cell_failure = ExperimentResult(
                        status='failed_closed',
                        scientific_scope=scope,
                        config_id=config.config_id,
                        config_hash=config.sha256,
                        repeat_indices=(repeat_index,),
                        fold_indices=(fold_index,),
                        output_dir=str(cell_directory),
                        failure_reasons=(failure,),
                    )
                    summaries.append({
                        'status': 'failed_closed',
                        'repeat_index': repeat_index,
                        'fold_index': fold_index,
                        'scientific_scope': scope,
                        'reason': str(exc),
                        'formal_execution_gate_checkpoints': {
                            'pre': gate_pre,
                            'post': gate_post,
                        },
                    })
                    _write_failed_artifacts(cell_directory, cell_failure)
                    _write_artifact_index(cell_directory)
        root_artifacts_pre = gate_session.checkpoint('root_artifacts_pre')
        status = 'passed' if not failures else 'failed_closed'
        result = ExperimentResult(
            status=status,
            scientific_scope=scope,
            config_id=config.config_id,
            config_hash=config.sha256,
            repeat_indices=repeat_values,
            fold_indices=fold_values,
            output_dir=str(target),
            cell_results=tuple(summaries),
            metrics={
                'requested_cell_count': len(repeat_values) * len(fold_values),
                'passed_cell_count': len(passed_cells),
                'failed_cell_count': len(failures),
                'elapsed_seconds': time.perf_counter() - started,
            },
            provenance={
                'preflight_status': report.status,
                'manifest_hash': report.manifest_hash,
                'fold_hash': report.fold_hash,
                'frozen_outer_split': True,
                'data_shortening': False,
                'record_cap': None,
                'epoch_override': None,
                'operational_measurement_requested': bool(measure_operational_costs),
                'source_snapshot_sha256':
                    gate_session.baseline['source_snapshot_sha256'],
                'source_tree_state':
                    gate_session.baseline['source_tree_state'],
                'dependency_gate': gate_session.baseline['dependency_gate'],
                'formal_execution_gate_baseline_sha256':
                    gate_session.baseline_sha256,
                'root_artifacts_pre_checkpoint': root_artifacts_pre,
            },
            failure_reasons=tuple(failures),
        )
        _write_full_root_artifacts(staging, passed_cells, result)
        _strict_json(staging / 'experiment_result.json', result.to_dict())
        gate_session.checkpoint('root_commit_pre')
        _strict_json(
            staging / 'formal_execution_gate_evidence.json',
            gate_session.evidence(),
        )
        _write_artifact_index(staging)
        gate_session.assert_current('root_atomic_publish')
        _commit_staging(staging, target)
        return result
    except Exception:
        if staging.exists():
            import shutil
            shutil.rmtree(staging)
        raise


def _load_strict_json_object(path: Path) -> dict[str, Any]:
    """Read one strict-JSON mapping and reject duplicate keys/NaN constants."""

    import json

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in values:
            if key in output:
                raise ValueError(f'duplicate_json_key:{path}:{key}')
            output[key] = value
        return output

    def invalid_constant(value: str) -> None:
        raise ValueError(f'nonfinite_json_constant:{path}:{value}')

    payload = json.loads(
        path.read_text(encoding='utf-8'),
        object_pairs_hook=pairs,
        parse_constant=invalid_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f'json_root_must_be_mapping:{path}')
    return payload


def _verify_run_artifact_index(directory: Path) -> dict[str, Any]:
    """Verify every byte/size row and forbid files absent from the root index."""

    from .provenance import sha256_file, stable_payload_sha256

    root = directory.resolve()
    index_path = root / 'artifact_index.json'
    if not index_path.is_file():
        raise FileNotFoundError(f'missing_run_artifact_index:{index_path}')
    index = _load_strict_json_object(index_path)
    if (
        index.get('schema_version') != 'ppg_frailty.artifact_index.v2'
        or index.get('pipeline_generation') != 'final_pipeline_v2'
        or index.get('overwrite_existing') is not False
    ):
        raise ValueError(f'invalid_run_artifact_index_contract:{index_path}')
    rows = index.get('artifacts')
    if not isinstance(rows, list) or int(index.get('artifact_count', -1)) != len(rows):
        raise ValueError(f'invalid_run_artifact_index_rows:{index_path}')
    indexed: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError(f'invalid_run_artifact_index_row:{index_path}')
        relative = str(raw.get('path', ''))
        target = (root / relative).resolve()
        target.relative_to(root)
        if not relative or relative in indexed or not target.is_file():
            raise ValueError(f'indexed_artifact_missing_or_duplicate:{relative}')
        observed_bytes = target.stat().st_size
        observed_sha = sha256_file(target)
        if int(raw.get('bytes', -1)) != observed_bytes or raw.get('sha256') != observed_sha:
            raise ValueError(f'indexed_artifact_hash_or_size_drift:{relative}')
        indexed[relative] = dict(raw)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob('*')
        if path.is_file() and path != index_path
    }
    if actual != set(indexed):
        raise ValueError(
            'run_artifact_index_file_set_drift:'
            f'missing={sorted(set(indexed) - actual)}:'
            f'unindexed={sorted(actual - set(indexed))}'
        )
    if index.get('index_payload_sha256') != stable_payload_sha256(rows):
        raise ValueError(f'run_artifact_index_payload_hash_drift:{index_path}')
    return {
        'artifact_index_sha256': sha256_file(index_path),
        'artifact_count': len(rows),
        'artifacts': indexed,
    }


def _fold_summaries_from_run(directory: Path) -> tuple[dict[str, Any], ...]:
    payload = _load_strict_json_object(directory / 'metrics_per_fold_seed.json')
    if (
        payload.get('schema_version') != 'ppg_frailty.metrics_per_fold_seed.v2'
        or payload.get('pipeline_generation') != 'final_pipeline_v2'
        or payload.get('status') != 'passed'
    ):
        raise ValueError('comparison_input_metrics_contract_or_status_invalid')
    rows = payload.get('cells')
    if not isinstance(rows, list) or len(rows) != 25:
        raise ValueError('comparison_input_requires_exact_25_cell_summaries')
    summaries = tuple(dict(row) for row in rows if isinstance(row, Mapping))
    keys = {
        (int(row['repeat_index']), int(row['fold_index'])) for row in summaries
    }
    if len(summaries) != 25 or keys != {
        (repeat, fold) for repeat in range(5) for fold in range(5)
    }:
        raise ValueError('comparison_input_cell_grid_drift')
    if any(row.get('status') != 'passed' for row in summaries):
        raise ValueError('comparison_input_contains_nonpassed_cell')
    split_seeds = {
        repeat: {int(row['split_seed']) for row in summaries if int(row['repeat_index']) == repeat}
        for repeat in range(5)
    }
    observed = tuple(next(iter(split_seeds[index])) for index in range(5))
    if any(len(values) != 1 for values in split_seeds.values()) or observed != _FORMAL_SPLIT_SEEDS:
        raise ValueError('comparison_input_split_seed_registry_drift')
    machine_ids = {str(row.get('model_machine_id', '')) for row in summaries}
    if len(machine_ids) != 1:
        raise ValueError('comparison_input_cell_model_identity_drift')
    ensemble = next(iter(machine_ids)) in _FORMAL_COMPARISON_MACHINE_IDS
    expected_members = [42, 10042, 20042, 30042, 40042]
    for row in summaries:
        if ensemble:
            if (
                row.get('training_seed') is not None
                or row.get('member_training_seeds') != expected_members
                or row.get('seed_policy') != 'cv_fixed_five_member_seed_roster'
            ):
                raise ValueError('comparison_input_ensemble_seed_provenance_drift')
        elif (
            int(row.get('training_seed', -1)) != int(row['split_seed'])
            or row.get('member_training_seeds') != []
            or row.get('seed_policy')
            != 'outer_cv_repeat_seed_equals_split_seed'
        ):
            raise ValueError('comparison_input_single_repeat_seed_provenance_drift')
    return summaries


_COMPARISON_OOF_AUTHORITY_FIELDS = (
    'file_id', 'role', 'label', 'training_seed', 'manifest_hash', 'fold_hash',
    'preprocessing_hash', 'feature_hash', 'model_hash', 'representation_mode',
    'signal_route', 'quality_score', 'retained', 'prediction_kind',
    'member_training_seeds', 'ensemble_base_model_id', 'class_order',
    'code_commit', 'data_schema_id', 'feature_schema_id', 'model_version',
    'aggregation_rule', 'environment_hash', 'manifest_version',
    'fold_registry_version', 'artifact_reducer_name',
    'artifact_reducer_version', 'route_status', 'source_snapshot_hash',
    'rejection_reason',
)
_COMPARISON_IMMUTABLE_OOF_AUTHORITY_FIELDS = frozenset(
    {
        'file_id', 'role', 'label', 'manifest_hash', 'fold_hash', 'retained',
        'class_order', 'code_commit', 'data_schema_id', 'environment_hash',
        'manifest_version', 'fold_registry_version', 'source_snapshot_hash',
        'rejection_reason',
    }
)
_COMPARISON_SUMMARY_AUTHORITY_FIELDS = (
    'quality_mode', 'balance_line', 'representation_transform_provenance',
    'sqi_calibrator_provenance', 'physical_recording_qc',
)


def _comparison_authority_identity(
    rows: Iterable[Any],
    summaries: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    """Hash every cross-run authority/base factor on its exact cell roster."""

    from .models.factory import FROZEN_MODEL_RUN_PROVENANCE_FIELDS
    from .provenance import stable_payload_sha256

    frozen_rows = tuple(
        sorted(rows, key=lambda row: (row.participant_id, row.repeat, row.fold))
    )
    frozen_summaries = tuple(
        sorted(
            summaries,
            key=lambda row: (int(row['repeat_index']), int(row['fold_index'])),
        )
    )
    if len(frozen_rows) != 145 or len(frozen_summaries) != 25:
        raise ValueError('comparison_authority_requires_complete_145_row_25_cell_run')
    identity: dict[str, str] = {}
    for field_name in _COMPARISON_OOF_AUTHORITY_FIELDS:
        identity[f'oof.{field_name}'] = stable_payload_sha256(
            [
                {
                    'participant_id': row.participant_id,
                    'repeat': int(row.repeat),
                    'value': to_strict_json_value(getattr(row, field_name)),
                }
                for row in frozen_rows
            ]
        )
    expected_frozen_fields = set(FROZEN_MODEL_RUN_PROVENANCE_FIELDS)
    for field_name in FROZEN_MODEL_RUN_PROVENANCE_FIELDS:
        values: list[dict[str, Any]] = []
        for summary in frozen_summaries:
            provenance = summary.get('frozen_model_run_provenance')
            if (
                not isinstance(provenance, Mapping)
                or set(provenance) != expected_frozen_fields
            ):
                raise ValueError(
                    'comparison_cell_frozen_run_provenance_contract_drift:'
                    f"r{summary.get('repeat_index')}f{summary.get('fold_index')}"
                )
            values.append(
                {
                    'repeat': int(summary['repeat_index']),
                    'fold': int(summary['fold_index']),
                    'value': to_strict_json_value(provenance[field_name]),
                }
            )
        identity[f'frozen.{field_name}'] = stable_payload_sha256(values)
    for field_name in _COMPARISON_SUMMARY_AUTHORITY_FIELDS:
        if any(field_name not in summary for summary in frozen_summaries):
            raise ValueError(
                f'comparison_cell_summary_authority_field_missing:{field_name}'
            )
        identity[f'summary.{field_name}'] = stable_payload_sha256(
            [
                {
                    'repeat': int(summary['repeat_index']),
                    'fold': int(summary['fold_index']),
                    'value': to_strict_json_value(summary[field_name]),
                }
                for summary in frozen_summaries
            ]
        )
    provenance = manifest.get('provenance')
    if not isinstance(provenance, Mapping):
        raise ValueError('comparison_root_manifest_provenance_missing')
    root_bindings = {
        'manifest_hash': 'manifest_hash',
        'fold_hash': 'fold_hash',
        'source_snapshot_sha256': 'source_snapshot_hash',
    }
    for root_name, row_name in root_bindings.items():
        value = str(provenance.get(root_name, ''))
        observed = {str(getattr(row, row_name)) for row in frozen_rows}
        if len(value) != 64 or observed != {value}:
            raise ValueError(f'comparison_root_oof_authority_drift:{root_name}')
        identity[f'root.{root_name}'] = stable_payload_sha256(value)
    return identity


def _allowed_comparison_authority_differences() -> frozenset[str]:
    """Return named factors that may differ only when explicitly declared."""

    from .models.factory import FROZEN_MODEL_RUN_PROVENANCE_FIELDS

    mutable_oof = set(_COMPARISON_OOF_AUTHORITY_FIELDS) - set(
        _COMPARISON_IMMUTABLE_OOF_AUTHORITY_FIELDS
    )
    return frozenset(
        {f'oof.{name}' for name in mutable_oof}
        | {
            f'frozen.{name}'
            for name in FROZEN_MODEL_RUN_PROVENANCE_FIELDS
            if name != 'fold_hash'
        }
        | {f'summary.{name}' for name in _COMPARISON_SUMMARY_AUTHORITY_FIELDS}
    )


def _validate_complete_formal_gate_evidence(
    root: Path,
    manifest: Mapping[str, Any],
    summaries: Iterable[Mapping[str, Any]],
    *,
    config_id: str,
    config_hash: str,
    machine_id: str,
) -> dict[str, Any]:
    """Verify the sealed entry/cell/root gate evidence of one complete 5x5 run."""

    from .provenance import stable_payload_sha256

    evidence = _load_strict_json_object(
        root / 'formal_execution_gate_evidence.json'
    )
    evidence_fields = {
        'schema_version', 'pipeline_generation', 'baseline',
        'baseline_sha256', 'checkpoints',
        'formal_execution_gate_evidence_sha256',
    }
    if (
        set(evidence) != evidence_fields
        or evidence.get('schema_version')
        != 'ppg_frailty.formal_execution_gate_evidence.v2'
        or evidence.get('pipeline_generation') != 'final_pipeline_v2'
    ):
        raise ValueError('comparison_run_formal_gate_evidence_schema_drift')
    sealed = {
        key: value
        for key, value in evidence.items()
        if key != 'formal_execution_gate_evidence_sha256'
    }
    if (
        evidence['formal_execution_gate_evidence_sha256']
        != stable_payload_sha256(sealed)
    ):
        raise ValueError('comparison_run_formal_gate_evidence_seal_drift')
    baseline = evidence.get('baseline')
    baseline_fields = {
        'schema_version', 'pipeline_generation', 'operation', 'config_id',
        'config_hash', 'config_source_relative_path',
        'config_source_file_sha256', 'source_tree_state',
        'source_snapshot_sha256',
        'dependency_gate', 'dependency_gate_sha256',
    }
    if (
        not isinstance(baseline, Mapping)
        or set(baseline) != baseline_fields
        or baseline.get('schema_version')
        != 'ppg_frailty.formal_execution_gate_capture.v2'
        or baseline.get('pipeline_generation') != 'final_pipeline_v2'
        or baseline.get('operation') != 'formal_benchmark'
        or baseline.get('config_id') != config_id
        or baseline.get('config_hash') != config_hash
        or not str(baseline.get('config_source_relative_path', '')).startswith(
            'configs/'
        )
        or len(str(baseline.get('config_source_file_sha256', ''))) != 64
        or evidence.get('baseline_sha256')
        != stable_payload_sha256(baseline)
    ):
        raise ValueError('comparison_run_formal_gate_baseline_drift')
    source_state = baseline.get('source_tree_state')
    dependency_gate = baseline.get('dependency_gate')
    source_state_fields = {
        'schema_version', 'v2_path', 'git_head_commit',
        'v2_pyproject_tracked', 'status_entry_count', 'status_sha256',
        'status_preview', 'tracked_and_clean',
    }
    required_profiles = (
        tuple(dependency_gate.get('required_profile_ids', ()))
        if isinstance(dependency_gate, Mapping) else ()
    )
    dependency_rows = (
        dependency_gate.get('profiles')
        if isinstance(dependency_gate, Mapping) else None
    )
    dependency_row_fields = {
        'profile_id', 'requirements_file', 'lock_status', 'python_version',
        'requirements_sha256', 'resolved_packages', 'live_runtime',
        'exact_lock_ready',
    }
    deep_machine_ids = {
        'compact_cnn', 'inception_full', 'inception_small',
        'inception_matrix', 'inception_full_five_member_ensemble',
        'inception_matrix_five_member_ensemble', 'fusion_compact',
        'fusion_inception', 'shapeformer_channel_specific_osd',
        'shapeformer_channel_specific_scalar_distance_ablation',
        'shapeformer_effect_size_fixed_v1',
    }
    expected_profiles = (
        ('core', 'deep', 'formal_benchmark')
        if machine_id in deep_machine_ids
        else ('core', 'formal_benchmark')
    )
    if (
        not isinstance(source_state, Mapping)
        or set(source_state) != source_state_fields
        or source_state.get('schema_version')
        != 'ppg_frailty.scientific_source_gate.v1'
        or source_state.get('v2_path') != 'final_v0/final_pipeline_v2'
        or len(str(source_state.get('git_head_commit', ''))) != 40
        or len(str(source_state.get('status_sha256', ''))) != 64
        or source_state.get('tracked_and_clean') is not True
        or source_state.get('v2_pyproject_tracked') is not True
        or source_state.get('status_entry_count') != 0
        or source_state.get('status_preview') != []
        or not isinstance(dependency_gate, Mapping)
        or dependency_gate.get('schema_version')
        != 'ppg_frailty.dependency_gate.v2'
        or dependency_gate.get('operation') != 'formal_benchmark'
        or dependency_gate.get('config_id') != config_id
        or dependency_gate.get('require_exact_lock') is not True
        or dependency_gate.get('all_required_exact_locks_ready') is not True
        or not required_profiles
        or len(required_profiles) != len(set(required_profiles))
        or required_profiles != expected_profiles
        or not isinstance(dependency_rows, list)
        or len(dependency_rows) != len(required_profiles)
        or any(not isinstance(row, Mapping) for row in dependency_rows)
        or tuple(row.get('profile_id') for row in dependency_rows)
        != required_profiles
        or any(set(row) != dependency_row_fields for row in dependency_rows)
        or any(
            row.get('lock_status') != 'validated_exact_lock'
            or row.get('exact_lock_ready') is not True
            or not isinstance(row.get('live_runtime'), Mapping)
            or row['live_runtime'].get('live_exact_match') is not True
            for row in dependency_rows
        )
        or baseline.get('dependency_gate_sha256')
        != stable_payload_sha256(dependency_gate)
    ):
        raise ValueError('comparison_run_formal_gate_authority_not_exact')
    expected_phases = ['entry_preflight']
    for repeat in range(5):
        for fold in range(5):
            prefix = f'repeat_{repeat:02d}_fold_{fold:02d}'
            expected_phases.extend((f'{prefix}_pre', f'{prefix}_post'))
    expected_phases.extend(('root_artifacts_pre', 'root_commit_pre'))
    checkpoints = evidence.get('checkpoints')
    if (
        not isinstance(checkpoints, list)
        or len(checkpoints) != 53
        or [row.get('phase') for row in checkpoints] != expected_phases
    ):
        raise ValueError('comparison_run_formal_gate_checkpoint_phase_drift')
    checkpoint_by_phase: dict[str, Mapping[str, Any]] = {}
    for row in checkpoints:
        if not isinstance(row, Mapping) or set(row) != {
            'phase', 'capture_sha256', 'source_snapshot_sha256',
            'dependency_gate_sha256', 'checkpoint_sha256',
        }:
            raise ValueError('comparison_run_formal_gate_checkpoint_schema_drift')
        payload = {
            key: value for key, value in row.items()
            if key != 'checkpoint_sha256'
        }
        if (
            row['capture_sha256'] != evidence['baseline_sha256']
            or row['source_snapshot_sha256']
            != baseline['source_snapshot_sha256']
            or row['dependency_gate_sha256']
            != baseline['dependency_gate_sha256']
            or row['checkpoint_sha256'] != stable_payload_sha256(payload)
        ):
            raise ValueError('comparison_run_formal_gate_checkpoint_seal_drift')
        checkpoint_by_phase[str(row['phase'])] = row
    frozen_summaries = tuple(summaries)
    if len(frozen_summaries) != 25:
        raise ValueError('comparison_run_formal_gate_requires_25_cell_summaries')
    for summary in frozen_summaries:
        prefix = (
            f"repeat_{int(summary['repeat_index']):02d}_"
            f"fold_{int(summary['fold_index']):02d}"
        )
        archived = summary.get('formal_execution_gate_checkpoints')
        if (
            not isinstance(archived, Mapping)
            or to_strict_json_value(archived.get('pre'))
            != to_strict_json_value(checkpoint_by_phase[f'{prefix}_pre'])
            or to_strict_json_value(archived.get('post'))
            != to_strict_json_value(checkpoint_by_phase[f'{prefix}_post'])
        ):
            raise ValueError('comparison_run_cell_formal_gate_checkpoint_drift')
    provenance = manifest.get('provenance')
    if (
        not isinstance(provenance, Mapping)
        or provenance.get('formal_execution_gate_baseline_sha256')
        != evidence['baseline_sha256']
        or provenance.get('source_snapshot_sha256')
        != baseline['source_snapshot_sha256']
        or to_strict_json_value(provenance.get('source_tree_state'))
        != to_strict_json_value(source_state)
        or to_strict_json_value(provenance.get('dependency_gate'))
        != to_strict_json_value(dependency_gate)
        or to_strict_json_value(provenance.get('root_artifacts_pre_checkpoint'))
        != to_strict_json_value(checkpoint_by_phase['root_artifacts_pre'])
    ):
        raise ValueError('comparison_run_manifest_formal_gate_binding_drift')
    return dict(evidence)


def _read_trusted_comparison_run(
    config_id: str,
    directory: str | Path,
    *,
    n_bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Read one complete 5x5 run and rebuild every predictive metric."""

    from .training import (
        build_config_metrics_from_predictions_and_fold_summaries,
        read_oof_parquet,
        read_oof_parquet_metadata,
        validate_expected_oof_roster,
    )

    root = Path(directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'comparison_run_directory_missing:{root}')
    index_evidence = _verify_run_artifact_index(root)
    mandatory = {
        'artifact_index.json',
        'run_manifest.json',
        'formal_execution_gate_evidence.json',
        'metrics_per_fold_seed.json',
        'config_metrics_v2.json',
        'oof_subject_predictions.parquet',
        'oof_member_predictions.parquet',
    }
    if not mandatory <= ({'artifact_index.json'} | set(index_evidence['artifacts'])):
        raise ValueError('comparison_run_missing_mandatory_indexed_artifact')
    cell_required = {
        'artifact_index.json',
        'run_manifest.json',
        'metrics_per_fold_seed.json',
        'confusion_matrices.json',
        'quality_diagnostics.json',
        'oof_window_predictions.parquet',
        'oof_file_predictions.parquet',
        'oof_subject_predictions.parquet',
        'oof_member_predictions.parquet',
    }
    cell_manifests: dict[tuple[int, int], dict[str, Any]] = {}
    for repeat in range(5):
        for fold in range(5):
            relative_root = f'repeat_{repeat:02d}_fold_{fold:02d}'
            expected = {f'{relative_root}/{name}' for name in cell_required}
            if not expected <= set(index_evidence['artifacts']):
                raise ValueError(
                    f'comparison_run_missing_cell_artifacts:{relative_root}'
                )
            _verify_run_artifact_index(root / relative_root)
            cell_manifest = _load_strict_json_object(
                root / relative_root / 'run_manifest.json'
            )
            if (
                cell_manifest.get('schema_version') != 'ppg_frailty.run_manifest.v2'
                or cell_manifest.get('pipeline_generation') != 'final_pipeline_v2'
                or cell_manifest.get('status') != 'passed'
                or cell_manifest.get('scientific_scope')
                != 'frozen_5x5_scientific_benchmark'
            ):
                raise ValueError(
                    f'comparison_run_cell_manifest_invalid:{relative_root}'
                )
            cell_manifests[(repeat, fold)] = cell_manifest
    manifest = _load_strict_json_object(root / 'run_manifest.json')
    if (
        manifest.get('schema_version') != 'ppg_frailty.run_manifest.v2'
        or manifest.get('pipeline_generation') != 'final_pipeline_v2'
        or manifest.get('status') != 'passed'
        or manifest.get('scientific_scope') != 'frozen_5x5_scientific_benchmark'
        or manifest.get('config_id') != config_id
        or tuple(manifest.get('repeat_indices', ())) != tuple(range(5))
        or tuple(manifest.get('fold_indices', ())) != tuple(range(5))
    ):
        raise ValueError(f'comparison_run_manifest_not_complete_passed_5x5:{config_id}')
    config_hash = str(manifest.get('config_hash', ''))
    if len(config_hash) != 64:
        raise ValueError(f'comparison_run_config_hash_invalid:{config_id}')
    summaries = _fold_summaries_from_run(root)
    summary_by_cell = {
        (int(row['repeat_index']), int(row['fold_index'])): row
        for row in summaries
    }
    for key, cell_manifest in cell_manifests.items():
        if to_strict_json_value(cell_manifest.get('cell')) != to_strict_json_value(
            summary_by_cell[key]
        ):
            raise ValueError(
                f'comparison_run_cell_root_summary_drift:r{key[0]}f{key[1]}'
            )
    machine_ids = {str(row['model_machine_id']) for row in summaries}
    if len(machine_ids) != 1:
        raise ValueError(f'comparison_run_mixed_model_ids:{config_id}')
    machine_id = next(iter(machine_ids))
    if machine_id in {
        'inception_full_five_member_ensemble',
        'inception_matrix_five_member_ensemble',
    }:
        raise ValueError(
            'comparison_ensemble_cv_seed_matrix_pending_human_decision'
        )
    registry_role = _registry_role_for_machine_id(machine_id)
    if any(str(row.get('model_machine_id')) != machine_id for row in summaries):
        raise ValueError(f'comparison_run_model_identity_drift:{config_id}')

    _validate_complete_formal_gate_evidence(
        root,
        manifest,
        summaries,
        config_id=config_id,
        config_hash=config_hash,
        machine_id=machine_id,
    )
    oof_rows = read_oof_parquet(root / 'oof_subject_predictions.parquet')
    if len(oof_rows) != 145:
        raise ValueError(f'comparison_run_requires_145_subject_rows:{config_id}')
    if any(row.config_hash != config_hash for row in oof_rows):
        raise ValueError(f'comparison_run_oof_config_hash_drift:{config_id}')
    ensemble_base_by_id = {
        'inception_full_five_member_ensemble': 'inception_full',
        'inception_matrix_five_member_ensemble': 'inception_matrix',
    }
    if machine_id in ensemble_base_by_id:
        member_relative = 'oof_member_predictions.parquet'
        if member_relative not in index_evidence['artifacts']:
            raise ValueError(f'comparison_run_missing_root_member_oof:{config_id}')
        member_rows = read_oof_parquet(root / member_relative)
        if len(member_rows) != 725:
            raise ValueError(
                f'comparison_run_requires_725_ensemble_member_rows:{config_id}'
            )
        base_model_id = ensemble_base_by_id[machine_id]
        expected_member_seeds = (42, 10042, 20042, 30042, 40042)
        for row in member_rows:
            if (
                row.prediction_kind != 'ensemble_member'
                or row.member_index not in range(5)
                or row.training_seed != expected_member_seeds[int(row.member_index)]
                or row.ensemble_base_model_id != base_model_id
                or row.config_hash != config_hash
            ):
                raise ValueError(
                    f'comparison_run_ensemble_member_identity_drift:{config_id}'
                )
        for row in oof_rows:
            if (
                row.prediction_kind != 'ensemble_average'
                or tuple(row.member_training_seeds) != expected_member_seeds
                or row.ensemble_base_model_id != base_model_id
            ):
                raise ValueError(
                    f'comparison_run_ensemble_average_identity_drift:{config_id}'
                )
        expected_roster: dict[tuple[int, int, int], list[str]] = {}
        for row in oof_rows:
            expected_roster.setdefault(
                (int(row.repeat), int(row.fold), int(row.split_seed)),
                [],
            ).append(str(row.participant_id))
        validate_expected_oof_roster(
            (*member_rows, *oof_rows),
            expected_roster,
            expected_config_hashes=(config_hash,),
            expected_member_count=5,
            require_trace=True,
        )
    else:
        member_rows = ()
        for row in oof_rows:
            if (
                row.prediction_kind != 'single_model'
                or row.training_seed != _FORMAL_SPLIT_SEEDS[int(row.repeat)]
                or row.training_seed != row.split_seed
                or row.member_training_seeds
            ):
                raise ValueError(
                    f'comparison_run_single_repeat_seed_identity_drift:{config_id}'
                )
        member_path = root / 'oof_member_predictions.parquet'
        if read_oof_parquet(member_path):
            raise ValueError(f'comparison_run_single_model_has_member_rows:{config_id}')
        member_metadata = read_oof_parquet_metadata(member_path)
        if (
            member_metadata.get('artifact_state') != 'empty'
            or member_metadata.get('empty_reason')
            != 'single_model_runner_ensemble_comparison_not_executed'
        ):
            raise ValueError(
                f'comparison_run_single_model_member_artifact_not_typed_empty:{config_id}'
            )
    from collections import Counter

    root_subject_by_cell: dict[tuple[int, int], tuple[Any, ...]] = {}
    root_member_by_cell: dict[tuple[int, int], tuple[Any, ...]] = {}
    for repeat in range(5):
        for fold in range(5):
            key = (repeat, fold)
            cell_root = root / f'repeat_{repeat:02d}_fold_{fold:02d}'
            cell_subjects = read_oof_parquet(
                cell_root / 'oof_subject_predictions.parquet'
            )
            expected_subjects = tuple(
                row for row in oof_rows if (int(row.repeat), int(row.fold)) == key
            )
            if Counter(cell_subjects) != Counter(expected_subjects):
                raise ValueError(
                    f'comparison_run_cell_root_subject_oof_drift:r{repeat}f{fold}'
                )
            root_subject_by_cell[key] = cell_subjects
            expected_roster = {
                (repeat, fold, int(_FORMAL_SPLIT_SEEDS[repeat])):
                    tuple(row.participant_id for row in expected_subjects)
            }
            member_path = cell_root / 'oof_member_predictions.parquet'
            if machine_id in ensemble_base_by_id:
                cell_members = read_oof_parquet(member_path)
                expected_members = tuple(
                    row
                    for row in member_rows
                    if (int(row.repeat), int(row.fold)) == key
                )
                if Counter(cell_members) != Counter(expected_members):
                    raise ValueError(
                        f'comparison_run_cell_root_member_oof_drift:r{repeat}f{fold}'
                    )
                validate_expected_oof_roster(
                    (*cell_members, *cell_subjects),
                    expected_roster,
                    expected_config_hashes=(config_hash,),
                    expected_member_count=5,
                    require_trace=True,
                )
                root_member_by_cell[key] = cell_members
            else:
                if read_oof_parquet(member_path):
                    raise ValueError(
                        f'comparison_run_single_cell_has_member_rows:r{repeat}f{fold}'
                    )
                metadata = read_oof_parquet_metadata(member_path)
                if (
                    metadata.get('artifact_state') != 'empty'
                    or metadata.get('empty_reason')
                    != 'single_model_runner_ensemble_comparison_not_executed'
                ):
                    raise ValueError(
                        'comparison_run_single_cell_member_artifact_not_typed_empty:'
                        f'r{repeat}f{fold}'
                    )
                validate_expected_oof_roster(
                    cell_subjects,
                    expected_roster,
                    expected_config_hashes=(config_hash,),
                    expected_member_count=1,
                    require_trace=True,
                )
    predictions = _participant_predictions_from_subject_rows(oof_rows)
    participants = tuple(sorted({row.participant_id for row in predictions}))
    if len(participants) != 29 or len(predictions) != 145:
        raise ValueError(f'comparison_run_participant_repeat_roster_incomplete:{config_id}')
    membership = {
        (row.participant_id, int(row.repeat)):
            (int(row.fold), int(row.split_seed), int(row.label))
        for row in oof_rows
    }
    for repeat in range(5):
        counts = sorted(
            sum(row.repeat == repeat and row.fold == fold for row in oof_rows)
            for fold in range(5)
        )
        if counts != [5, 6, 6, 6, 6]:
            raise ValueError(f'comparison_run_fold_size_drift:{config_id}:repeat={repeat}')

    stored_payload = _load_strict_json_object(root / 'config_metrics_v2.json')
    stored = stored_payload.get('config_metrics')
    if (
        stored_payload.get('status') != 'passed_trusted_metrics_rebuilt_from_typed_oof'
        or stored_payload.get('config_id') != config_id
        or stored_payload.get('config_hash') != config_hash
        or stored_payload.get('independent_test') is not False
        or stored_payload.get('fold_protocol') != 'frozen_repeated_grouped_5x5'
        or stored_payload.get('seeds') != list(_FORMAL_SPLIT_SEEDS)
        or not isinstance(stored, Mapping)
    ):
        raise ValueError(f'comparison_run_trusted_metrics_contract_invalid:{config_id}')
    if stored.get('registry_role') != registry_role:
        raise ValueError(f'comparison_run_registry_role_drift:{config_id}')
    fold_ba = {
        f"r{int(row['repeat_index'])}f{int(row['fold_index'])}":
            float(row['metrics']['balanced_accuracy'])
        for row in summaries
    }
    fold_confusions, fold_rosters = _fold_confusions_and_rosters_from_subject_rows(
        oof_rows
    )
    for summary in summaries:
        key = f"r{int(summary['repeat_index'])}f{int(summary['fold_index'])}"
        if to_strict_json_value(summary['metrics']['confusion_matrix']) != (
            to_strict_json_value(fold_confusions[key])
        ):
            raise ValueError(
                f'comparison_run_cell_confusion_oof_drift:{config_id}:{key}'
            )
    operational = _operational_metrics_from_fold_summaries(summaries)
    metrics, bootstrap = build_config_metrics_from_predictions_and_fold_summaries(
        config_id=config_id,
        registry_role=registry_role,
        predictions=predictions,
        fold_balanced_accuracies=fold_ba,
        fold_confusion_matrices=fold_confusions,
        fold_participant_rosters=fold_rosters,
        inference_cost=operational['inference_cost'],
        parameter_count=operational['parameter_count'],
        n_bootstrap_resamples=int(n_bootstrap_resamples),
        bootstrap_seed=int(bootstrap_seed),
        eligible=operational['eligible'],
        exclusion_reason=operational['exclusion_reason'],
    )
    scalar_fields = (
        'participant_mean_balanced_accuracy',
        'participant_mean_macro_f1',
        'worst_fold_balanced_accuracy',
        'worst_class_recall',
        'worst_class_f1',
        'expected_calibration_error',
    )
    for field_name in scalar_fields:
        if abs(float(stored[field_name]) - float(getattr(metrics, field_name))) > 1e-12:
            raise ValueError(f'comparison_run_stored_metric_drift:{config_id}:{field_name}')
    if to_strict_json_value(stored['variability']) != to_strict_json_value(metrics.variability):
        raise ValueError(f'comparison_run_variability_drift:{config_id}')
    if to_strict_json_value(stored['confusion_matrices']) != to_strict_json_value(metrics.confusion_matrices):
        raise ValueError(f'comparison_run_confusion_matrix_drift:{config_id}')
    for field_name in ('inference_cost', 'parameter_count', 'eligible', 'exclusion_reason'):
        if to_strict_json_value(stored[field_name]) != to_strict_json_value(
            getattr(metrics, field_name)
        ):
            raise ValueError(
                f'comparison_run_operational_metric_drift:{config_id}:{field_name}'
            )
    if stored_payload.get('operational_measurement_status') != operational['status']:
        raise ValueError(
            f'comparison_run_operational_status_drift:{config_id}'
        )
    authority_identity = _comparison_authority_identity(
        oof_rows,
        summaries,
        manifest,
    )
    return {
        'config_id': config_id,
        'config_hash': config_hash,
        'machine_id': machine_id,
        'metrics': metrics,
        'bootstrap': bootstrap,
        'predictions': predictions,
        'membership': membership,
        'participants': participants,
        'artifact_index_sha256': index_evidence['artifact_index_sha256'],
        'artifact_count': index_evidence['artifact_count'],
        'authority_identity': authority_identity,
        'source_snapshot_sha256': manifest.get('provenance', {}).get(
            'source_snapshot_sha256'
        ),
    }


def build_comparison_archive_from_run_directories(
    run_directories: Mapping[str, str | Path],
    *,
    reference_config_id: str,
    comparison_family: str,
    comparison_id: str,
    run_id: str,
    output_root: str | Path,
    n_bootstrap_resamples: int = 10_000,
    n_permutation_resamples: int = 100_000,
    statistics_seed: int = 42,
    allowed_authority_differences: Iterable[str] = (),
) -> dict[str, Any]:
    """Build one explicit statistics archive from complete indexed run roots.

    This function never trains, refits, exports, or selects a winner. Predictive
    metrics and both LCB columns are rebuilt from typed participant OOF. Paired
    inference fails closed unless every config has the identical participant,
    repeat, fold, split-seed, label and immutable data/code authority. Every
    other changed base factor must be explicitly named as the comparison axis.
    """

    from .training import (
        ComparisonArchive,
        holm_adjust_by_family_metric,
        paired_participant_permutation,
    )
    from .training.statistics import _write_formal_comparison_archive
    from .config import load_config
    from .pipeline import PipelinePaths
    from .provenance import stable_payload_sha256

    directories = {str(key): value for key, value in run_directories.items()}
    if len(directories) < 2 or any(not key.strip() for key in directories):
        raise ValueError('comparison archive requires at least two named run directories')
    if reference_config_id not in directories:
        raise ValueError('reference_config_id must occur in run_directories')
    if not str(comparison_family).strip():
        raise ValueError('comparison_family must be explicit')
    if int(n_bootstrap_resamples) <= 0 or int(n_permutation_resamples) <= 0:
        raise ValueError('statistics resample counts must be positive')
    declared_differences = tuple(
        sorted(set(str(value).strip() for value in allowed_authority_differences))
    )
    allowed_names = _allowed_comparison_authority_differences()
    unknown_differences = sorted(set(declared_differences) - set(allowed_names))
    if '' in declared_differences or unknown_differences:
        raise ValueError(
            'comparison_allowed_authority_difference_invalid:'
            + ','.join(unknown_differences or ['empty'])
        )

    paths = PipelinePaths.discover()
    authority_config = load_config(
        paths.pipeline_root / 'configs/reference_static_feature_vector_v2.yaml'
    )
    gate_session = _FormalExecutionGateSession.establish(
        authority_config,
        paths,
        operation='formal_benchmark',
    )
    gate_session.checkpoint('statistics_inputs_pre')
    loaded = {
        config_id: _read_trusted_comparison_run(
            config_id,
            directory,
            n_bootstrap_resamples=int(n_bootstrap_resamples),
            bootstrap_seed=int(statistics_seed),
        )
        for config_id, directory in sorted(directories.items())
    }
    reference = loaded[reference_config_id]
    reference_membership = reference['membership']
    reference_participants = reference['participants']
    reference_authority = reference['authority_identity']
    observed_differences: set[str] = set()
    for config_id, current in loaded.items():
        if current['membership'] != reference_membership:
            raise ValueError(
                f'comparison_roster_or_fold_membership_differs:{config_id}'
            )
        if current['participants'] != reference_participants:
            raise ValueError(f'comparison_participant_coverage_differs:{config_id}')
        if set(current['authority_identity']) != set(reference_authority):
            raise ValueError(f'comparison_authority_field_set_drift:{config_id}')
        for field_name, expected_hash in reference_authority.items():
            if current['authority_identity'][field_name] == expected_hash:
                continue
            if field_name not in declared_differences:
                raise ValueError(
                    f'comparison_undeclared_authority_difference:'
                    f'{config_id}:{field_name}'
                )
            observed_differences.add(field_name)
    unused_differences = sorted(set(declared_differences) - observed_differences)
    if unused_differences:
        raise ValueError(
            'comparison_declared_authority_difference_not_observed:'
            + ','.join(unused_differences)
        )

    paired: dict[str, Any] = {}
    raw_p_values: dict[tuple[str, str, str], float] = {}
    for config_id, current in loaded.items():
        if config_id == reference_config_id:
            continue
        comparison_key = f'{config_id}_vs_{reference_config_id}'
        for metric in ('balanced_accuracy', 'macro_f1'):
            result = paired_participant_permutation(
                reference['predictions'],
                current['predictions'],
                metric=metric,
                n_resamples=int(n_permutation_resamples),
                seed=int(statistics_seed),
            )
            result_key = f'{comparison_key}__{metric}'
            paired[result_key] = result
            raw_p_values[(str(comparison_family), metric, comparison_key)] = (
                result.two_sided_p_value
            )
    holm = holm_adjust_by_family_metric(raw_p_values, alpha=0.05)
    gate_session.checkpoint('statistics_completed')
    gate_session.checkpoint('archive_publish_pre')
    gate_evidence = gate_session.evidence()
    authority_unsigned = {
        'schema_version':
            'ppg_frailty.formal_comparison_producer_authority.v2',
        'pipeline_generation': 'final_pipeline_v2',
        'status': 'formal_clean_source_exact_dependency_toc_guarded',
        'operation': 'formal_benchmark',
        'authority_config_id': authority_config.config_id,
        'authority_config_hash': authority_config.sha256,
        'formal_execution_gate': gate_evidence,
        'formal_execution_gate_sha256':
            gate_evidence['formal_execution_gate_evidence_sha256'],
    }
    producer_authority = {
        **authority_unsigned,
        'producer_authority_sha256':
            stable_payload_sha256(authority_unsigned),
    }
    archive = ComparisonArchive(
        comparison_id=str(comparison_id),
        run_id=str(run_id),
        configs=tuple(current['metrics'] for current in loaded.values()),
        bootstrap_results={
            config_id: current['bootstrap'] for config_id, current in loaded.items()
        },
        paired_permutation_results=paired,
        holm_results=holm,
        selections=(),
        run_manifest={
            'schema_version': 'ppg_frailty.comparison_run_manifest.v2',
            'pipeline_generation': 'final_pipeline_v2',
            'status': 'statistics_completed_no_automatic_selection',
            'independent_test': False,
            'fold_protocol': 'frozen_repeated_grouped_5x5',
            'seeds': list(_FORMAL_SPLIT_SEEDS),
            'reference_config_id': reference_config_id,
            'comparison_family': str(comparison_family),
            'strict_roster_match': True,
            'authority_identity_policy': (
                'immutable_data_code_fold_authority_exact;'
                'all_other_differences_must_be_named_and_observed'
            ),
            'allowed_authority_differences': list(declared_differences),
            'observed_authority_differences': sorted(observed_differences),
            'participant_count': len(reference_participants),
            'participant_repeat_count': len(reference_membership),
            'source_runs': {
                config_id: {
                    'config_hash': current['config_hash'],
                    'model_machine_id': current['machine_id'],
                    'artifact_index_sha256': current['artifact_index_sha256'],
                    'artifact_count': current['artifact_count'],
                    'source_snapshot_sha256': current['source_snapshot_sha256'],
                    'authority_identity_sha256': _runtime_imports()[
                        'stable_payload_sha256'
                    ](current['authority_identity']),
                }
                for config_id, current in loaded.items()
            },
            'bootstrap_policy': {
                'resamples': int(n_bootstrap_resamples),
                'seed': int(statistics_seed),
                'metrics': ['balanced_accuracy', 'macro_f1'],
            },
            'paired_permutation_policy': {
                'resamples': int(n_permutation_resamples),
                'seed': int(statistics_seed),
                'exchange_unit': 'participant_with_all_repeats',
                'coverage_policy': 'exact_roster_required_no_intersection',
            },
            'multiplicity_policy': 'Holm_within_comparison_family_x_metric',
            'automatic_selection': False,
            'producer_authority': producer_authority,
            'training_executed': False,
            'refit_executed': False,
            'onnx_export_executed': False,
        },
    )
    requested_target = (
        Path(output_root).resolve() / str(comparison_id) / str(run_id)
    )
    _require_controlled_formal_artifact_target(paths, requested_target)
    target = _write_formal_comparison_archive(
        archive,
        output_root,
        prepublish_guard=lambda: gate_session.assert_current(
            'archive_atomic_publish'
        ),
    )
    return {
        'schema_version': 'ppg_frailty.comparison_orchestrator_result.v2',
        'pipeline_generation': 'final_pipeline_v2',
        'status': 'passed',
        'comparison_id': str(comparison_id),
        'run_id': str(run_id),
        'output_directory': str(target),
        'config_ids': sorted(loaded),
        'reference_config_id': reference_config_id,
        'comparison_family': str(comparison_family),
        'automatic_selection': False,
        'training_executed': False,
        'refit_executed': False,
        'onnx_export_executed': False,
    }


def _load_strict_json_array(path: Path) -> list[Any]:
    """Read one strict JSON array, rejecting duplicate keys and NaN/Infinity."""

    import json

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in values:
            if key in output:
                raise ValueError(f'duplicate_json_key:{path}:{key}')
            output[key] = value
        return output

    def invalid_constant(value: str) -> None:
        raise ValueError(f'nonfinite_json_constant:{path}:{value}')

    payload = json.loads(
        path.read_text(encoding='utf-8'),
        object_pairs_hook=pairs,
        parse_constant=invalid_constant,
    )
    if not isinstance(payload, list):
        raise ValueError(f'json_root_must_be_array:{path}')
    return payload


_MANUAL_SELECTION_FIELDS = {
    'schema_version',
    'pipeline_generation',
    'selection_authority',
    'purpose',
    'config_id',
    'config_hash',
    'registry_role',
    'model_machine_id',
    'human_rationale',
    'comparison_id',
    'comparison_run_id',
    'comparison_artifact_index_sha256',
    'selected_run_artifact_index_sha256',
    'automatic_selection',
    'manual_selection_sha256',
}


def verify_manual_selection_record(
    path: str | Path,
    *,
    expected_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify a separate purpose-specific human selection record."""

    from .provenance import sha256_file, stable_payload_sha256

    target = Path(path).resolve()
    if not target.is_file():
        raise FileNotFoundError(f'manual_selection_record_missing:{target}')
    observed_file_sha = sha256_file(target)
    if (
        expected_file_sha256 is not None
        and observed_file_sha != str(expected_file_sha256)
    ):
        raise ValueError('manual_selection_record_file_sha256_mismatch')
    payload = _load_strict_json_object(target)
    if set(payload) != _MANUAL_SELECTION_FIELDS:
        raise ValueError('manual_selection_record_field_schema_drift')
    selection_hash = str(payload.pop('manual_selection_sha256', ''))
    if stable_payload_sha256(payload) != selection_hash:
        raise ValueError('manual_selection_record_payload_hash_mismatch')
    payload['manual_selection_sha256'] = selection_hash
    if (
        payload.get('schema_version')
        != 'ppg_frailty.manual_final_selection.v2'
        or payload.get('pipeline_generation') != 'final_pipeline_v2'
        or payload.get('selection_authority')
        != 'explicit_human_purpose_specific'
        or payload.get('automatic_selection') is not False
        or not str(payload.get('purpose', '')).strip()
        or not str(payload.get('human_rationale', '')).strip()
        or payload.get('registry_role')
        not in {'reference', 'ablation', 'comparison'}
    ):
        raise ValueError('manual_selection_record_contract_invalid')
    for name in (
        'config_hash',
        'comparison_artifact_index_sha256',
        'selected_run_artifact_index_sha256',
        'manual_selection_sha256',
    ):
        value = str(payload.get(name, ''))
        if len(value) != 64 or any(character not in '0123456789abcdef' for character in value):
            raise ValueError(f'manual_selection_record_invalid_sha256:{name}')
    payload['selection_record_file_sha256'] = observed_file_sha
    return payload


def _validate_selection_against_formal_archive(
    archive: Path,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild the archive/eligible-top10/source-run binding at every consumer."""

    import csv

    from .provenance import sha256_file, stable_payload_sha256
    from .training import verify_comparison_archive

    verify_comparison_archive(archive)
    if (
        sha256_file(archive / 'artifact_index.json')
        != selection.get('comparison_artifact_index_sha256')
    ):
        raise ValueError('manual_selection_comparison_archive_hash_mismatch')
    if _load_strict_json_array(archive / 'selection_record.json') != []:
        raise ValueError('comparison_archive_selection_record_must_remain_empty')
    manifest = _load_strict_json_object(archive / 'run_manifest.json')
    attestation = _load_strict_json_object(
        archive / 'formal_producer_attestation.json'
    )
    authority = manifest.get('producer_authority')
    if (
        not isinstance(authority, Mapping)
        or set(authority) != {
            'schema_version', 'pipeline_generation', 'status', 'operation',
            'authority_config_id', 'authority_config_hash',
            'formal_execution_gate', 'formal_execution_gate_sha256',
            'producer_authority_sha256',
        }
        or authority.get('schema_version')
        != 'ppg_frailty.formal_comparison_producer_authority.v2'
        or authority.get('status')
        != 'formal_clean_source_exact_dependency_toc_guarded'
        or to_strict_json_value(attestation)
        != to_strict_json_value(authority)
    ):
        raise ValueError('manual_selection_requires_formal_producer_authority')
    unsigned = {
        key: value for key, value in authority.items()
        if key != 'producer_authority_sha256'
    }
    if (
        authority['producer_authority_sha256']
        != stable_payload_sha256(unsigned)
        or authority.get('formal_execution_gate_sha256')
        != authority.get(
            'formal_execution_gate', {}
        ).get('formal_execution_gate_evidence_sha256')
    ):
        raise ValueError('manual_selection_formal_producer_authority_seal_drift')
    if (
        manifest.get('comparison_id') != selection.get('comparison_id')
        or manifest.get('run_id') != selection.get('comparison_run_id')
        or manifest.get('automatic_selection') is not False
    ):
        raise ValueError('manual_selection_comparison_identity_mismatch')
    metrics = _load_strict_json_array(archive / 'metrics_all_configs.json')
    matches = [
        dict(row) for row in metrics
        if isinstance(row, Mapping)
        and row.get('config_id') == selection.get('config_id')
    ]
    with (archive / 'ranking_top10.csv').open(
        'r', encoding='utf-8', newline=''
    ) as handle:
        ranking = tuple(csv.DictReader(handle))
    ranked_ids = tuple(str(row.get('config_id', '')) for row in ranking)
    expected_ranking = tuple(
        sorted(
            (
                dict(row) for row in metrics
                if isinstance(row, Mapping) and row.get('eligible') is True
            ),
            key=lambda row: (
                -float(row['participant_mean_balanced_accuracy']),
                str(row['config_id']),
            ),
        )[:10]
    )
    ranking_semantics_match = (
        ranked_ids
        == tuple(str(row['config_id']) for row in expected_ranking)
        and all(
            str(observed.get('rank')) == str(index)
            and observed.get('registry_role') == expected.get('registry_role')
            and float(observed.get('participant_mean_balanced_accuracy', 'nan'))
            == float(expected['participant_mean_balanced_accuracy'])
            and float(observed.get('participant_mean_macro_f1', 'nan'))
            == float(expected['participant_mean_macro_f1'])
            and float(observed.get('balanced_accuracy_lcb95', 'nan'))
            == float(expected['balanced_accuracy_lcb95'])
            and float(observed.get('macro_f1_lcb95', 'nan'))
            == float(expected['macro_f1_lcb95'])
            for index, (observed, expected) in enumerate(
                zip(ranking, expected_ranking),
                start=1,
            )
        )
    )
    if (
        len(matches) != 1
        or matches[0].get('eligible') is not True
        or str(matches[0].get('exclusion_reason', '')) != ''
        or selection.get('config_id') not in ranked_ids
        or len(ranked_ids) > 10
        or matches[0].get('registry_role') != selection.get('registry_role')
        or not ranking_semantics_match
    ):
        raise ValueError(
            'manual_selection_requires_eligible_archived_top10_configuration'
        )
    source_runs = manifest.get('source_runs')
    source_run = (
        source_runs.get(selection.get('config_id'))
        if isinstance(source_runs, Mapping) else None
    )
    if (
        not isinstance(source_run, Mapping)
        or source_run.get('config_hash') != selection.get('config_hash')
        or source_run.get('model_machine_id')
        != selection.get('model_machine_id')
        or source_run.get('artifact_index_sha256')
        != selection.get('selected_run_artifact_index_sha256')
    ):
        raise ValueError('manual_selection_source_run_identity_mismatch')
    return dict(source_run)


def write_manual_selection_record(
    comparison_archive: str | Path,
    *,
    config_id: str,
    purpose: str,
    human_rationale: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Write one immutable manual selection outside the comparison archive."""

    from .provenance import sha256_file, stable_payload_sha256
    from .training import verify_comparison_archive

    archive = Path(comparison_archive).resolve()
    verify_comparison_archive(archive)
    if _load_strict_json_array(archive / 'selection_record.json') != []:
        raise ValueError('comparison_archive_selection_record_must_remain_empty')
    manifest = _load_strict_json_object(archive / 'run_manifest.json')
    metrics = _load_strict_json_array(archive / 'metrics_all_configs.json')
    matches = [
        dict(row)
        for row in metrics
        if isinstance(row, Mapping) and row.get('config_id') == str(config_id)
    ]
    if len(matches) != 1:
        raise ValueError('manual_selection_config_not_unique_in_comparison_archive')
    selected = matches[0]
    with (archive / 'ranking_top10.csv').open(
        'r', encoding='utf-8', newline=''
    ) as handle:
        ranking_rows = tuple(csv.DictReader(handle))
    ranked_ids = tuple(str(row.get('config_id', '')) for row in ranking_rows)
    if (
        selected.get('eligible') is not True
        or str(selected.get('exclusion_reason', '')) != ''
        or config_id not in ranked_ids
        or len(ranked_ids) > 10
    ):
        raise ValueError(
            'manual_selection_requires_eligible_archived_top10_configuration'
        )
    source_runs = manifest.get('source_runs')
    if not isinstance(source_runs, Mapping) or config_id not in source_runs:
        raise ValueError('manual_selection_source_run_identity_missing')
    source_run = source_runs[config_id]
    if (
        not isinstance(source_run, Mapping)
        or selected.get('registry_role') not in {'reference', 'ablation', 'comparison'}
        or not str(purpose).strip()
        or not str(human_rationale).strip()
    ):
        raise ValueError('manual_selection_input_invalid')
    payload: dict[str, Any] = {
        'schema_version': 'ppg_frailty.manual_final_selection.v2',
        'pipeline_generation': 'final_pipeline_v2',
        'selection_authority': 'explicit_human_purpose_specific',
        'purpose': str(purpose).strip(),
        'config_id': str(config_id),
        'config_hash': str(source_run.get('config_hash', '')),
        'registry_role': str(selected['registry_role']),
        'model_machine_id': str(source_run.get('model_machine_id', '')),
        'human_rationale': str(human_rationale).strip(),
        'comparison_id': str(manifest.get('comparison_id', '')),
        'comparison_run_id': str(manifest.get('run_id', '')),
        'comparison_artifact_index_sha256': sha256_file(
            archive / 'artifact_index.json'
        ),
        'selected_run_artifact_index_sha256': str(
            source_run.get('artifact_index_sha256', '')
        ),
        'automatic_selection': False,
    }
    _validate_selection_against_formal_archive(archive, payload)
    payload['manual_selection_sha256'] = stable_payload_sha256(payload)
    target = Path(output_path).resolve()
    _strict_json(target, payload)
    return verify_manual_selection_record(target)


def final_refit_preflight_from_verified_artifacts(
    run_directory: str | Path,
    selection_record: str | Path,
    *,
    expected_selection_file_sha256: str,
    comparison_archive: str | Path,
    config_path: str | Path,
    confirm_scientific_execution: bool,
) -> dict[str, Any]:
    """Verify the only permitted trust path into the all-29 refit executor.

    This interface performs no fit. It proves that a later caller can only build
    the strict FinalRefitPlan after verified 5x5 OOF, a separately hashed manual
    purpose-specific selection, the current frozen manifest/fold registry, and
    the exact source snapshot all agree.
    """

    from .config import dependency_gate_report
    from .provenance import sha256_file
    from .training import read_oof_parquet, verify_comparison_archive

    if not confirm_scientific_execution:
        raise PermissionError(
            'final refit requires --confirm-scientific-execution'
        )
    selection = verify_manual_selection_record(
        selection_record,
        expected_file_sha256=expected_selection_file_sha256,
    )
    archive = Path(comparison_archive).resolve()
    verify_comparison_archive(archive)
    if (
        sha256_file(archive / 'artifact_index.json')
        != selection['comparison_artifact_index_sha256']
    ):
        raise ValueError('manual_selection_comparison_archive_hash_mismatch')
    selected_source_run = _validate_selection_against_formal_archive(
        archive,
        selection,
    )

    run_root = Path(run_directory).resolve()
    trusted = _read_trusted_comparison_run(
        str(selection['config_id']),
        run_root,
        n_bootstrap_resamples=1,
        bootstrap_seed=42,
    )
    if (
        trusted['config_hash'] != selection['config_hash']
        or trusted['machine_id'] != selection['model_machine_id']
        or trusted['metrics'].registry_role != selection['registry_role']
        or trusted['artifact_index_sha256']
        != selection['selected_run_artifact_index_sha256']
        or trusted['config_hash'] != selected_source_run['config_hash']
        or trusted['machine_id'] != selected_source_run['model_machine_id']
        or trusted['artifact_index_sha256']
        != selected_source_run['artifact_index_sha256']
    ):
        raise ValueError('manual_selection_selected_run_identity_mismatch')

    api = _runtime_imports()
    paths = api['PipelinePaths'].discover()
    source_tree_gate = _require_scientific_source_gate(paths)
    report, config, _rows, _registry = api['preflight_pipeline'](
        config_path,
        mode='full',
        paths=paths,
    )
    dependency_gate = dependency_gate_report(
        config,
        operation='formal_benchmark',
        profiles_path=paths.pipeline_root / 'requirements/profiles.json',
        lock_path=paths.pipeline_root / 'locks/profiles.lock.json',
        require_exact_lock=True,
    )
    if config.config_id != selection['config_id'] or config.sha256 != selection['config_hash']:
        raise ValueError('final_refit_config_identity_differs_from_selection')
    oof_path = run_root / 'oof_subject_predictions.parquet'
    oof_rows = read_oof_parquet(oof_path)
    manifest_hashes = {str(row.manifest_hash) for row in oof_rows}
    fold_hashes = {str(row.fold_hash) for row in oof_rows}
    source_hashes = {str(row.source_snapshot_hash) for row in oof_rows}
    participants = tuple(sorted({str(row.participant_id) for row in oof_rows}))
    if (
        len(oof_rows) != 145
        or len(participants) != 29
        or manifest_hashes != {report.manifest_hash}
        or fold_hashes != {report.fold_hash}
        or len(source_hashes) != 1
        or trusted['source_snapshot_sha256'] not in source_hashes
        or _source_snapshot_sha256(paths) not in source_hashes
    ):
        raise ValueError('final_refit_manifest_fold_oof_or_source_snapshot_drift')
    return {
        'schema_version': 'ppg_frailty.final_refit_verified_preflight.v2',
        'pipeline_generation': 'final_pipeline_v2',
        'status': 'ready_for_all29_materialization_and_strict_executor',
        'purpose': selection['purpose'],
        'config_id': selection['config_id'],
        'config_hash': selection['config_hash'],
        'registry_role': selection['registry_role'],
        'model_machine_id': selection['model_machine_id'],
        'participant_ids': list(participants),
        'participant_count': len(participants),
        'manifest_hash': report.manifest_hash,
        'fold_registry_hash': report.fold_hash,
        'module_registry_hash': report.module_registry_hash,
        'source_snapshot_sha256': next(iter(source_hashes)),
        'scientific_source_gate': source_tree_gate,
        'scientific_source_gate_sha256':
            stable_payload_sha256(source_tree_gate),
        'oof_evidence_sha256': sha256_file(oof_path),
        'manual_selection_sha256': selection['manual_selection_sha256'],
        'selection_record_file_sha256':
            selection['selection_record_file_sha256'],
        'next_executable_api': 'execute_final_refit_from_verified_artifacts',
        'execution_status':
            'ready_internal_verified_source_all29_materializer',
        'plan_hashes_caller_authored': False,
        'live_dependency_gate': dependency_gate,
        'training_executed': False,
        'onnx_export_executed': False,
    }


def _canonical_final_bundle_materialization(
    execution: Any,
    materialized: _TrustedFull29Materialization,
    config: Any,
    report: Any,
    preflight: Mapping[str, Any],
    paths: Any,
    formal_execution_gate: Mapping[str, Any],
) -> Any:
    '''Build metadata, fitted-transform archive and adapter from internal evidence.'''

    from .bundle import build_model_input_adapter
    from .training.bundle import (
        FrozenRepresentationTransformArchive,
        _bind_trusted_final_bundle_materialization,
        current_runtime_environment,
    )

    api = _runtime_imports()
    mode = str(config.representation_mode)
    manifest = config.section('manifest')
    signal = config.section('signal')
    windows = config.section('windows')
    quality = config.section('quality')
    artifact = config.section('artifact')
    features = config.section('features')
    model = config.section('model')
    training = config.section('training')
    aggregation = config.section('aggregation')
    selected_window = (
        windows['engineering']
        if mode in {'feature_vector', 'feature_matrix'}
        else windows['raw_dl']
    )
    spec = materialized.input_spec
    channel_schema = (
        tuple(spec.feature_names)
        if mode == 'feature_vector'
        else tuple(spec.channel_schema)
    )
    if not channel_schema:
        raise _ExperimentProtocolError('final_bundle_channel_schema_empty')
    normalized_model = dict(execution.binding.resolved_model_config)
    canonical_name = str(normalized_model['canonical_model_name'])
    machine_id = str(normalized_model['model_id'])
    fitted_objects: list[dict[str, Any]] = [
        {
            'name': 'final_model',
            'object_type': str(execution.result.provenance.object_type),
            'state_hash': str(execution.result.provenance.state_hash),
            'dataset_binding_hash': str(
                execution.result.provenance.dataset_binding_hash
            ),
            'fitted_on_participant_ids': list(
                execution.result.provenance.fitted_participant_ids
            ),
            'training_seed': int(execution.result.provenance.training_seed),
            'member_training_seeds': list(
                execution.result.provenance.member_training_seeds
            ),
        }
    ]
    for name, provenance in sorted(materialized.representation_provenance.items()):
        fitted_objects.append(
            {
                'name': str(name),
                'kind': 'representation_transform_provenance',
                'provenance': provenance,
            }
        )
    transform_archive = FrozenRepresentationTransformArchive(
        representation_mode=mode,
        input_schema_hash=execution.plan.input_schema_hash,
        fitted_on_participant_ids=execution.plan.participant_ids,
        fitted_artifacts=materialized.fitted_objects,
        provenance=materialized.representation_provenance,
        source_records_hash=materialized.source_records_hash,
        dataset_hash=materialized.dataset_hash,
    )
    adapter = build_model_input_adapter(
        mode,
        input_schema_hash=execution.plan.input_schema_hash,
        allowed_role_families=('B', 'R'),
    )
    feature_vector_schema: dict[str, Any]
    if mode == 'feature_vector':
        feature_vector_schema = {
            'status': 'model_input',
            'feature_names': list(spec.feature_names),
            'width': len(spec.feature_names),
            'missing_value_handling':
                'fitted_inside_final_all29_estimator_pipeline',
        }
    elif mode == 'fusion':
        vector_provenance = materialized.representation_provenance['feature_vector']
        feature_vector_schema = {
            'status': 'fusion_file_feature_input',
            'width': int(spec.n_file_features),
            'tensor_schema': list(vector_provenance['fusion_tensor_schema']),
        }
    else:
        feature_vector_schema = {
            'status': 'not_model_input',
            'reason': f'representation_mode={mode}',
        }
    if mode == 'feature_matrix':
        ordered_matrix_schema = {
            'status': 'model_input',
            'schema_version': materialized.feature_schema_id,
            'channel_schema': list(spec.channel_schema),
            'columns': int(materialized.dataset.values.shape[2]),
            'mask': 'row_mask_true_for_observed_ordered_columns',
        }
    else:
        ordered_matrix_schema = {
            'status': 'not_model_input',
            'reason': f'representation_mode={mode}',
        }
    if mode == 'raw':
        mask_semantics = {
            'model_input_mask': 'sample_validity_[batch,time]',
            'file_input_shape': '[window,channel,time]',
        }
    elif mode == 'feature_matrix':
        mask_semantics = {
            'model_input_mask': 'ordered_column_validity_[batch,column]',
            'file_input_shape': '[channel,column]',
        }
    elif mode == 'fusion':
        mask_semantics = {
            'window_mask': 'real_window_membership_[batch,window]',
            'sample_mask': 'sample_validity_[batch,window,time]',
        }
    else:
        mask_semantics = {
            'status': 'not_applicable_dense_feature_vector',
            'missingness': 'handled_by_fitted_estimator_pipeline',
        }
    balance_hash = api['stable_payload_sha256'](
        {
            'training_balance': training['training_balance'],
            'aggregation': aggregation,
            'classifier_role_families': ('B', 'R'),
        }
    )
    metadata = {
        'model_identity': {
            'name': canonical_name,
            'machine_id': machine_id,
            'version': str(model['variant']),
        },
        'representation_mode': mode,
        'signal_route': {
            'artifact_reducer': artifact['reducer'],
            'artifact_reducer_version': artifact['reducer_version'],
            'observed_route_statuses': sorted(
                {str(row['route_status']) for row in materialized.source_records}
            ),
        },
        'class_order': list(manifest['class_id_order']),
        'channel_schema': list(channel_schema),
        'preprocessing': {
            'signal': signal,
            'quality': quality,
            'artifact': artifact,
            'representation_transform_boundary': transform_archive.boundary,
            'source_records': list(materialized.source_records),
            'source_records_hash': materialized.source_records_hash,
            'fit_scope': 'verified_all29_participants_only',
        },
        'preprocessing_hash': materialized.preprocessing_hash,
        'resampling': {
            'canonical_signal_and_feature_fs_hz': float(signal['internal_fs_hz']),
            'dl_input': signal['dl_resampling'],
            'model_input_fs_hz': _model_input_sampling_rate_hz(config),
        },
        'window_plan': {
            'selected': selected_window,
            'shared_planner_version': windows['shared_planner_version'],
            'representation_mode': mode,
        },
        'feature_registry': {
            'registry_id': features['registry_id'],
            'feature_schema_id': materialized.feature_schema_id,
            'contract': materialized.feature_contract,
        },
        'feature_hash': materialized.feature_hash,
        'feature_vector_schema': feature_vector_schema,
        'ordered_matrix_schema': ordered_matrix_schema,
        'mask_semantics': mask_semantics,
        'validity_policy': {
            'quality_mode': quality['mode'],
            'quality_affects_classification': False,
            'hard_recording_qc': 'required_before_materialization',
            'classifier_role_families': ['B', 'R'],
        },
        'fitted_objects': fitted_objects,
        'representation_state': {
            'dataset_hash': materialized.dataset_hash,
            'source_records_hash': materialized.source_records_hash,
            'input_schema_hash': execution.plan.input_schema_hash,
            'transform_provenance': materialized.representation_provenance,
        },
        'pooling_rule': {
            'model_pooling': model.get('pooling', model.get('mask_aware_pooling', 'model_defined')),
            'window_to_file': aggregation['window_to_file'],
            'participant_inference': aggregation['balance_line'],
        },
        'aggregation_rule': aggregation['balance_line'],
        'manifest_hash': report.manifest_hash,
        'fold_hash': execution.scope.fold_hash,
        'manifest_version': manifest['manifest_version'],
        'fold_registry_version': config.section('splits')['registry_id'],
        'pipeline_generation': 'final_pipeline_v2',
        'config_hash': config.sha256,
        'balance_hash': balance_hash,
        'run_hash': execution.execution_hash,
        'source_snapshot_hash': preflight['source_snapshot_sha256'],
        'scientific_source_gate': preflight['scientific_source_gate'],
        'scientific_source_gate_sha256':
            preflight['scientific_source_gate_sha256'],
        'selection_record_file_sha256':
            preflight['selection_record_file_sha256'],
        'code_version': _code_commit(paths),
        'environment': current_runtime_environment(),
        'dependency_status': preflight['live_dependency_gate'],
        'formal_execution_gate': dict(formal_execution_gate),
        'serialization_trust': {
            'trusted_local_only': True,
            'authenticated_signature': False,
            'integrity': 'sha256_bound_files_plus_golden_reload_parity',
        },
        'golden_case': {
            'boundary': transform_archive.boundary,
            'source_identity': {
                'participant_id': materialized.dataset.identities[0].participant_id,
                'file_id': materialized.dataset.identities[0].file_id,
                'window_id': materialized.dataset.identities[0].window_id,
            },
            'expected_output': 'three_class_probability',
        },
    }
    return _bind_trusted_final_bundle_materialization(
        execution,
        metadata=metadata,
        golden_inputs=materialized.golden_inputs,
        transforms=transform_archive,
        pipeline_adapter=adapter,
        source_records_hash=materialized.source_records_hash,
    )


def execute_final_refit_from_verified_artifacts(
    run_directory: str | Path,
    selection_record: str | Path,
    *,
    expected_selection_file_sha256: str,
    comparison_archive: str | Path,
    config_path: str | Path,
    bundle_directory: str | Path,
    confirm_scientific_execution: bool,
) -> Any:
    """Materialise, refit and publish solely from verified internal artifacts.

    No caller dataset, trainer, binding, factory, estimator, metadata, transform
    or adapter is accepted.  The selected config and exact source bytes are
    reverified before materialisation, before fitting and before immutable bundle
    publication.  Calling this function performs the requested scientific refit;
    preflight alone remains available through the separate read-only API.
    """

    from .training import (
        FinalRefitPlan,
        FullCohortRefitScope,
        UnifiedTrainer,
        canonical_input_spec_payload,
        materialize_final_refit_binding,
    )
    from .training.bundle import (
        _execute_prepared_full_cohort_refit,
        _save_trusted_final_refit_bundle,
    )

    if not confirm_scientific_execution:
        raise PermissionError(
            'final refit requires --confirm-scientific-execution'
        )
    preflight = final_refit_preflight_from_verified_artifacts(
        run_directory,
        selection_record,
        expected_selection_file_sha256=expected_selection_file_sha256,
        comparison_archive=comparison_archive,
        config_path=config_path,
        confirm_scientific_execution=True,
    )
    api = _runtime_imports()
    paths = api['PipelinePaths'].discover()
    if (
        stable_payload_sha256(_require_scientific_source_gate(paths))
        != preflight['scientific_source_gate_sha256']
    ):
        raise RuntimeError('final_refit_scientific_source_gate_changed')
    target = _require_controlled_formal_artifact_target(
        paths,
        paths.output_path(bundle_directory),
    )
    if target.exists():
        raise FileExistsError(f'final refit bundle target already exists: {target}')
    report, config, rows, _registry = api['preflight_pipeline'](
        config_path,
        mode='full',
        paths=paths,
    )
    if (
        config.config_id != preflight['config_id']
        or config.sha256 != preflight['config_hash']
        or report.module_registry_hash != preflight['module_registry_hash']
    ):
        raise ValueError('final_refit_current_config_or_registry_drift')
    gate_session = _FormalExecutionGateSession.establish(
        config,
        paths,
        operation='formal_benchmark',
    )
    if (
        gate_session.baseline['source_snapshot_sha256']
        != preflight['source_snapshot_sha256']
        or stable_payload_sha256(
            gate_session.baseline['source_tree_state']
        ) != preflight['scientific_source_gate_sha256']
        or to_strict_json_value(gate_session.baseline['dependency_gate'])
        != to_strict_json_value(preflight['live_dependency_gate'])
    ):
        raise RuntimeError('final_refit_entry_formal_gate_differs_from_preflight')

    gate_session.checkpoint('all29_materialization_pre')
    materialized = _materialize_trusted_full29(
        report,
        config,
        rows,
        paths,
        preflight,
    )
    gate_session.checkpoint('all29_materialization_post')
    full_dataset = materialized.dataset

    factory_model_config, machine_id = _resolved_model_config(
        config,
        training_seed=42,
    )
    expected_training = api['replace'](
        api['TrainingConfig'].from_mapping(config.section('training')),
        seed=42,
    )
    expected_spec = _model_input_spec(full_dataset, config.representation_mode)
    if canonical_input_spec_payload(expected_spec) != canonical_input_spec_payload(
        materialized.input_spec
    ):
        raise ValueError('final_refit_internal_input_spec_drift')

    participant_ids = tuple(str(value) for value in preflight['participant_ids'])
    scope = FullCohortRefitScope(
        participant_ids=participant_ids,
        registry_hash=report.module_registry_hash,
        config_hash=config.sha256,
        oof_evidence_hash=preflight['oof_evidence_sha256'],
    ).bind_training_dataset(full_dataset)
    prepared = api['prepare_model_factory'](
        factory_model_config,
        expected_spec,
        full_dataset,
        scope,
    )
    model_config_for_binding: Mapping[str, Any] = prepared.resolved_model_config
    model_section = config.section('model')
    training_section = config.section('training')
    windows = config.section('windows')
    signal = config.section('signal')
    quality = config.section('quality')
    aggregation = config.section('aggregation')
    evaluation = config.section('evaluation')
    selected_window = (
        windows['engineering']
        if config.representation_mode in {'feature_vector', 'feature_matrix'}
        else windows['raw_dl']
    )
    classical = machine_id in _ESTIMATOR_MODEL_IDS
    ensemble = machine_id in {
        'inception_full_five_member_ensemble',
        'inception_matrix_five_member_ensemble',
    }
    seeds = FINAL_ENSEMBLE_MEMBER_SEEDS if ensemble else (42,)
    model_input_fs_hz = _model_input_sampling_rate_hz(config)
    input_channels_order = (
        tuple(str(value) for value in expected_spec.feature_names)
        if config.representation_mode == 'feature_vector'
        else tuple(str(value) for value in expected_spec.channel_schema)
    )
    base_provenance = api['validate_frozen_model_run_provenance']({
        'architecture_parameters': _resolved_architecture_parameters(
            machine_id,
            model_section,
        ),
        'input_channels_order': input_channels_order,
        'sampling_rate_hz': model_input_fs_hz,
        'window_plan': {
            'representation_mode': config.representation_mode,
            'selected': dict(selected_window),
            'shared_planner_version': windows['shared_planner_version'],
            'model_input_sampling_rate_hz': model_input_fs_hz,
            'canonical_signal_and_feature_sampling_rate_hz':
                float(signal['internal_fs_hz']),
        },
        'hop_plan': {
            'hop_s': float(selected_window['hop_s']),
            'end_alignment': selected_window['end_alignment'],
        },
        'normalization': dict(signal['normalization']),
        'padding_mask': {
            'padding': selected_window['padding'],
            'min_valid_fraction': selected_window.get('min_valid_fraction', 1.0),
            'mask_aware_pooling': model_section.get(
                'mask_aware_pooling',
                'not_applicable',
            ),
        },
        'feature_schema_hash': materialized.feature_hash,
        'sqi_routing': {
            'mode': quality['mode'],
            'supervised_route_ready': quality['supervised_route_ready'],
            'failure_action': quality['failure_action'],
        },
        **_training_algorithm_provenance(
            machine_id,
            training_section,
            model_section,
            fixed_epochs=int(expected_training.fixed_epochs),
        ),
        'random_seeds': seeds,
        'seed_policy': (
            'final_refit_five_member_seeds'
            if ensemble else 'final_refit_single_seed_42'
        ),
        'fold_hash': scope.fold_hash,
        'aggregation': dict(aggregation),
        'calibration': {
            'metrics': tuple(evaluation['calibration_metrics']),
            'fit_scope': 'all29_final_refit',
        },
    })
    expected_binding = materialize_final_refit_binding(
        resolved_model_config=model_config_for_binding,
        input_spec=expected_spec,
        training_config=expected_training,
        frozen_run_provenance=base_provenance,
        config_hash=config.sha256,
        registry_hash=report.module_registry_hash,
        source_snapshot_hash=preflight['source_snapshot_sha256'],
        manual_selection_hash=preflight['manual_selection_sha256'],
        oof_evidence_hash=preflight['oof_evidence_sha256'],
    )

    plan = FinalRefitPlan(
        purpose=str(preflight['purpose']),
        config_hash=config.sha256,
        model_id=machine_id,
        participant_ids=participant_ids,
        training_seeds=seeds,
        fixed_epochs=None if classical else int(expected_training.fixed_epochs),
        epoch_rule='not_applicable' if classical else 'fixed_epoch',
        model_family='classical_or_rocket' if classical else 'deep',
        oof_evidence_hash=preflight['oof_evidence_sha256'],
        model_kind='five_member_ensemble' if ensemble else 'single_model',
        registry_hash=report.module_registry_hash,
        source_snapshot_hash=preflight['source_snapshot_sha256'],
        manual_selection_hash=preflight['manual_selection_sha256'],
        resolved_model_config_hash=expected_binding.resolved_model_config_hash,
        architecture_parameters_hash=expected_binding.architecture_parameters_hash,
        input_schema_hash=expected_binding.input_schema_hash,
        training_config_hash=expected_binding.training_config_hash,
        frozen_run_provenance_hash=expected_binding.frozen_run_provenance_hash,
        representation_mode=config.representation_mode,
    )
    immediately_reverified = final_refit_preflight_from_verified_artifacts(
        run_directory,
        selection_record,
        expected_selection_file_sha256=expected_selection_file_sha256,
        comparison_archive=comparison_archive,
        config_path=config_path,
        confirm_scientific_execution=True,
    )
    if to_strict_json_value(immediately_reverified) != to_strict_json_value(preflight):
        raise RuntimeError('final_refit_artifacts_changed_before_execution')
    if _source_snapshot_sha256(paths) != preflight['source_snapshot_sha256']:
        raise RuntimeError('final_refit_source_snapshot_changed_before_execution')
    if (
        stable_payload_sha256(_require_scientific_source_gate(paths))
        != preflight['scientific_source_gate_sha256']
    ):
        raise RuntimeError('final_refit_scientific_source_gate_changed_before_execution')
    gate_session.checkpoint('all29_fit_pre')
    execution = _execute_prepared_full_cohort_refit(
        plan,
        UnifiedTrainer(expected_training),
        full_dataset,
        registry_hash=report.module_registry_hash,
        binding=expected_binding,
        model_factory=None if classical else prepared,
        estimator=prepared() if classical else None,
    )
    gate_session.checkpoint('all29_fit_post')
    if execution.dataset_hash != materialized.dataset_hash:
        raise RuntimeError('final_refit_execution_dataset_hash_drift')
    postfit_reverified = final_refit_preflight_from_verified_artifacts(
        run_directory,
        selection_record,
        expected_selection_file_sha256=expected_selection_file_sha256,
        comparison_archive=comparison_archive,
        config_path=config_path,
        confirm_scientific_execution=True,
    )
    if to_strict_json_value(postfit_reverified) != to_strict_json_value(preflight):
        raise RuntimeError('final_refit_artifacts_changed_during_execution')
    if _source_snapshot_sha256(paths) != preflight['source_snapshot_sha256']:
        raise RuntimeError('final_refit_source_snapshot_changed_during_execution')
    if (
        stable_payload_sha256(_require_scientific_source_gate(paths))
        != preflight['scientific_source_gate_sha256']
    ):
        raise RuntimeError('final_refit_scientific_source_gate_changed_during_execution')
    gate_session.checkpoint('bundle_materialization_pre')
    gate_session.checkpoint('bundle_publish_pre')
    publication = _canonical_final_bundle_materialization(
        execution,
        materialized,
        config,
        report,
        preflight,
        paths,
        gate_session.evidence(),
    )
    return _save_trusted_final_refit_bundle(
        execution,
        target,
        materialization=publication,
        prepublish_guard=lambda: gate_session.assert_current(
            'bundle_atomic_publish'
        ),
    )


FINAL_ENSEMBLE_MEMBER_SEEDS = (42, 10042, 20042, 30042, 40042)


def final_refit_policy(config: Any) -> dict[str, Any]:
    """Return the frozen post-selection policy without fitting or exporting."""

    model = config.section("model")
    model_id = str(model["model_id"])
    ensemble_size = int(model.get("ensemble_size", 1))
    if ensemble_size == 1:
        refit = {
            "kind": "single_model",
            "model_seed": 42,
            "member_seeds": [42],
        }
    elif model_id in {
        "InceptionTimeFullFiveMemberEnsemble",
        "InceptionTimeMatrixFiveMemberEnsemble",
    }:
        declared = tuple(int(value) for value in model.get("member_seeds", ()))
        if declared != FINAL_ENSEMBLE_MEMBER_SEEDS:
            raise _ExperimentProtocolError(
                "final_ensemble_member_seeds_must_match_confirmed_v2_sequence"
            )
        refit = {
            "kind": "five_member_probability_ensemble",
            "model_seed": None,
            "member_seeds": list(FINAL_ENSEMBLE_MEMBER_SEEDS),
        }
    else:
        raise _ExperimentProtocolError("unsupported_final_refit_ensemble_identity")
    return {
        "schema_version": "ppg_frailty.final_refit_policy.v2",
        "pipeline_generation": "final_pipeline_v2",
        "config_id": config.config_id,
        "config_hash": config.sha256,
        "selection_authority": "manual_purpose_specific_final_only",
        "fit_scope": "all_29_participants_from_scratch_after_manual_selection",
        "performance_claim": "none_full_refit_uses_oof_as_only_internal_performance_evidence",
        "refit": refit,
        "onnx_gate": {
            "required_for_winner": True,
            "boundary": "model_input_tensor_to_class_probabilities",
            "python_bundle_retains_preprocessing": True,
            "end_to_end_hardware_export": "deferred_v2_026",
            "executed": False,
        },
        "training_executed": False,
        "onnx_export_executed": False,
    }


_ONNX_WINNER_CERTIFICATE_FIELDS = {
    'schema_version', 'pipeline_generation', 'status', 'producer_entry_id',
    'selection_record_file_sha256', 'manual_selection_sha256', 'config_id',
    'config_hash', 'model_machine_id', 'final_refit_execution_hash',
    'bundle_manifest_sha256', 'bundle_manifest_bytes', 'bundle_run_hash',
    'input_spec_hash', 'class_order', 'boundary', 'opset_version',
    'onnx_model_sha256', 'onnx_model_bytes', 'export_executed',
    'onnxruntime_readback_executed', 'onnxruntime_version', 'case_count',
    'python_probabilities_sha256', 'onnx_probabilities_sha256',
    'absolute_tolerance', 'relative_tolerance', 'maximum_absolute_error',
    'maximum_relative_error', 'mean_absolute_error',
    'argmax_class_order_match', 'parity_passed',
}
_ONNX_WINNER_SOURCE_BOUND_PRODUCER_IMPLEMENTED = True


def _validate_onnx_certificate_parity_metrics(
    certificate: Mapping[str, Any],
) -> None:
    """Require both frozen error bounds and class-order argmax agreement."""

    from .onnx_winner import validate_certificate_parity_metrics

    validate_certificate_parity_metrics(certificate)


def winner_release_preflight(
    *,
    bundle_directory: str | Path,
    onnx_model_path: str | Path,
    certificate_path: str | Path,
    expected_certificate_file_sha256: str,
    selection_record: str | Path,
    expected_selection_file_sha256: str,
    expected_final_bundle_manifest_sha256: str,
    final_refit_attestation_path: str | Path,
    expected_final_refit_attestation_sha256: str,
    config_path: str | Path,
) -> dict[str, Any]:
    """Re-run the source-bound ONNX/ORT evidence; never export or release."""

    from .onnx_winner import winner_release_preflight as verified_preflight

    return verified_preflight(
        bundle_directory=bundle_directory,
        onnx_model_path=onnx_model_path,
        certificate_path=certificate_path,
        expected_certificate_file_sha256=expected_certificate_file_sha256,
        selection_record=selection_record,
        expected_selection_file_sha256=expected_selection_file_sha256,
        expected_final_bundle_manifest_sha256=
            expected_final_bundle_manifest_sha256,
        final_refit_attestation_path=final_refit_attestation_path,
        expected_final_refit_attestation_sha256=
            expected_final_refit_attestation_sha256,
        config_path=config_path,
    )


__all__ = [
    "build_comparison_archive_from_run_directories",
    "ExperimentResult",
    "FINAL_ENSEMBLE_MEMBER_SEEDS",
    "execute_final_refit_from_verified_artifacts",
    "final_refit_preflight_from_verified_artifacts",
    "final_refit_policy",
    "run_full_experiment",
    "run_reduced_fold_experiment",
    "verify_manual_selection_record",
    "winner_release_preflight",
    "write_manual_selection_record",
]
