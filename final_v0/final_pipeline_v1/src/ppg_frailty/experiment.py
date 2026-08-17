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

        return to_strict_json_value(asdict(self))


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


class _ExperimentProtocolError(RuntimeError):
    '''关闭失败异常 / Fail-closed protocol exception.'''


@dataclass(frozen=True)
class _CellResult:
    '''单折摘要与 OOF / One cell summary and OOF tables.'''

    summary: dict[str, Any]
    file_rows: tuple[Any, ...]
    subject_rows: tuple[Any, ...]


def _runtime_imports() -> dict[str, Any]:
    '''延迟导入重型依赖 / Lazily import experiment dependencies.'''

    import numpy as np
    from dataclasses import replace
    from ppg_frailty.artifact import run_artifact_route
    from ppg_frailty.contracts import QualityState, SignalRoute
    from ppg_frailty.data.windows import WindowPlan
    from ppg_frailty.features.engineering import extract_engineering_features
    from ppg_frailty.features.registry import build_feature_vector, default_registry, summarize_engineering
    from ppg_frailty.models import ModelInputSpec, create_model
    from ppg_frailty.pipeline import PipelinePaths, _load_record, preflight_pipeline
    from ppg_frailty.provenance import runtime_environment, stable_payload_sha256
    from ppg_frailty.signal.morphology import extract_morphology
    from ppg_frailty.signal.optical import extract_dual_optical
    from ppg_frailty.signal.peaks import detect_pulses
    from ppg_frailty.signal.preprocess import build_signal_views
    from ppg_frailty.signal.prv import compute_prv
    from ppg_frailty.signal.sqi import SqiConfig, evaluate_quality, fit_sqi_calibrator, quality_component_scores
    from ppg_frailty.training import FeatureVectorDataset, FrozenOuterSplit, OofPredictionRow, OofWriter
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
            state.views = build_signal_views(loader(state.row, maximum), config.to_dict())
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
            role=str(state.row.role),
            route=state.route,
            q_rate_qualified=state.final_quality.q_rate.state is QualityState.PASS,
        )
        plan = api['WindowPlan'](
            source_record_id=state.row.record_id,
            **report.window_profiles['engineering'],
        )
        engineering = api['extract_engineering_features'](state.views, plan=plan)
        values, validity = api['summarize_engineering'](engineering)
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
            role=state.row.role,
            label=int(state.row.class_id),
            signal_route=state.route.value,
            quality_score=float(state.final_quality.q_rate.score),
        )
        for state in selected
    )
    return api['FeatureVectorDataset'](
        api['np'].stack([state.vector.values for state in selected]),
        names,
        identities,
    )


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


def _make_oof(states: list[_RuntimeRecord], oof_ids: tuple[str, ...], dataset: Any, probabilities: Any, common: dict[str, Any]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    '''生成 file 与 participant OOF，保留 drop trace / Build complete OOF traces.'''

    api = _runtime_imports()
    probability_by_file = {}
    if dataset is not None:
        if probabilities.shape != (len(dataset), 3):
            raise _ExperimentProtocolError('oof_probability_shape_mismatch')
        probability_by_file = {
            identity.file_id: tuple(float(value) for value in probability)
            for identity, probability in zip(dataset.identities, probabilities)
        }
    file_rows = []
    for state in sorted(
        (item for item in states if item.row.participant_id in set(oof_ids)),
        key=lambda item: str(item.row.record_id),
    ):
        probability = probability_by_file.get(state.row.record_id, ())
        retained = bool(probability) and state.retained
        route = state.route if retained else state.intended_route
        if route is None:
            route = api['SignalRoute'].DROPPED
        quality = 0.0 if state.final_quality is None or state.final_quality.q_rate.score is None else float(state.final_quality.q_rate.score)
        row_common = dict(common)
        # A dropped row has no probability vector, therefore the OOF contract
        # requires an empty class order. 丢弃行无概率向量，类别顺序必须为空。
        if not retained:
            row_common['class_order'] = ()
        file_rows.append(api['OofPredictionRow'](
            participant_id=state.row.participant_id,
            file_id=state.row.record_id,
            role=state.row.role,
            label=int(state.row.class_id),
            probabilities=probability,
            signal_route=route.value,
            quality_score=quality,
            retained=retained,
            level='file',
            artifact_reducer_name=state.artifact_name,
            artifact_reducer_version=state.artifact_version,
            route_status=state.route_status,
            rejection_reason=None if retained else (state.reason or 'no_result'),
            **row_common,
        ))
    hierarchy = api['aggregate_hierarchy'](file_rows)
    subject_rows = list(hierarchy.participant_rows)
    observed = {row.participant_id for row in subject_rows}
    for participant in sorted(set(oof_ids) - observed):
        candidates = [row for row in file_rows if row.participant_id == participant]
        if not candidates:
            raise _ExperimentProtocolError(f'heldout_subject_has_no_record:{participant}')
        routes = {row.signal_route for row in candidates}
        if len(routes) != 1:
            raise _ExperimentProtocolError(f'dropped_subject_mixed_routes:{participant}')
        subject_rows.append(api['replace'](
            candidates[0],
            file_id=f'participant::{participant}',
            role='participant',
            probabilities=(),
            retained=False,
            level='participant',
            class_order=(),
            route_status='dropped_no_participant_probability',
            rejection_reason='all_selected_files_dropped',
        ))
    return tuple(file_rows), tuple(sorted(subject_rows, key=lambda row: row.participant_id))


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


def _execute_vector_cell(report: Any, config: Any, rows: Iterable[Any], registry: Any, paths: Any, *, repeat_index: int, fold_index: int, maximum_seconds: float | None, record_cap: int | None, epoch_override: int | None, loader: Any = None) -> _CellResult:
    '''执行一个 feature-vector outer cell / Execute one feature-vector outer cell.'''

    api = _runtime_imports()
    started = time.perf_counter()
    if config.representation_mode != 'feature_vector':
        raise _ExperimentProtocolError(
            f'unsupported_representation_for_current_runner:{config.representation_mode}'
        )
    split = registry.get_split(repeat_index, fold_index)
    train_ids = tuple(split['train_participant_ids'])
    oof_ids = tuple(split['oof_participant_ids'])
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
    sqi_config, calibrator = _fit_quality_calibrator(states, config, train_ids, oof_ids)
    _route_records(states, config, report, sqi_config, calibrator)
    for state in states:
        _extract_vector(state, report)
    retained_train_ids = {
        state.row.participant_id
        for state in states
        if state.retained
        and state.vector is not None
        and state.row.participant_id in set(train_ids)
    }
    missing_train = sorted(set(train_ids) - retained_train_ids)
    if missing_train:
        raise _ExperimentProtocolError(
            'outer_train_subject_zero_retained_files:' + ','.join(missing_train)
        )
    train_dataset = _dataset(
        state for state in states if state.row.participant_id in set(train_ids)
    )
    oof_dataset = _dataset(
        state for state in states if state.row.participant_id in set(oof_ids)
    )
    if train_dataset is None:
        raise _ExperimentProtocolError('outer_train_dataset_empty')
    frozen = api['FrozenOuterSplit'](
        repeat=int(split['repeat_index']),
        fold=int(split['fold_index']),
        seed=int(split['split_seed']),
        train_participant_ids=train_ids,
        oof_participant_ids=oof_ids,
        registry_hash=str(config.section('splits')['source_registry_payload_sha256']),
        fold_hash=report.fold_hash,
    )
    training_config = api['TrainingConfig'].from_mapping(config.section('training'))
    training_config = api['replace'](
        training_config,
        seed=int(split['training_seed']),
    )
    if epoch_override is not None:
        if training_config.epoch_rule != 'fixed_epoch' or epoch_override <= 0:
            raise _ExperimentProtocolError('invalid_fixed_epoch_override')
        training_config = api['replace'](
            training_config,
            fixed_epochs=int(epoch_override),
        )
    model_section = config.section('model')
    model_config = {
        'model_id': str(model_section['model_id']),
        'seed': int(split['training_seed']),
    }
    input_spec = api['ModelInputSpec'](
        'feature_vector',
        n_classes=3,
        feature_names=tuple(train_dataset.feature_names),
    )
    model = api['create_model'](model_config, input_spec)
    training = api['UnifiedTrainer'](training_config).fit_estimator(
        model,
        train_dataset,
        frozen,
    )
    probabilities = (
        api['np'].empty((0, 3))
        if oof_dataset is None
        else api['np'].asarray(training.model.predict_proba(oof_dataset.values))
    )
    classes = tuple(int(value) for value in training.model.classes_)
    if set(classes) != {0, 1, 2}:
        raise _ExperimentProtocolError(f'trained_model_missing_class:{classes}')
    if classes != (0, 1, 2):
        probabilities = probabilities[:, [classes.index(value) for value in (0, 1, 2)]]

    preprocessing_hash = api['stable_payload_sha256']({
        'signal': config.section('signal'),
        'quality': config.section('quality'),
        'artifact': config.section('artifact'),
        'calibrator_method': calibrator.method,
        'calibrator_bounds': calibrator.bounds,
        'calibrator_fitted_ids': calibrator.fitted_on_participant_ids,
    })
    feature_registry = api['default_registry']()
    feature_hash = api['stable_payload_sha256']({
        'registry_hash': feature_registry.sha256,
        'feature_names': train_dataset.feature_names,
        'estimator_transforms': 'fit_inside_outer_train_pipeline',
    })
    environment = api['runtime_environment']()
    common = {
        'repeat': int(split['repeat_index']),
        'fold': int(split['fold_index']),
        'seed': int(split['split_seed']),
        'config_hash': config.sha256,
        'manifest_hash': report.manifest_hash,
        'fold_hash': report.fold_hash,
        'preprocessing_hash': preprocessing_hash,
        'feature_hash': feature_hash,
        'model_hash': training.provenance.state_hash,
        'representation_mode': config.representation_mode,
        'class_order': (0, 1, 2),
        'code_commit': _code_commit(paths),
        'data_schema_id': 'internal_records_v1',
        'feature_schema_id': feature_registry.schema_version,
        'model_version': str(model_section['variant']),
        'aggregation_rule': 'window_file_role_participant_equal',
        'environment_hash': api['stable_payload_sha256'](environment),
        'manifest_version': str(config.section('manifest')['manifest_version']),
        'fold_registry_version': str(config.section('splits')['registry_id']),
    }
    file_rows, subject_rows = _make_oof(
        states,
        oof_ids,
        oof_dataset,
        probabilities,
        common,
    )
    api['validate_expected_oof_roster'](
        subject_rows,
        {
            (
                int(split['repeat_index']),
                int(split['fold_index']),
                int(split['split_seed']),
            ): oof_ids
        },
        expected_config_hashes=(config.sha256,),
    )
    metrics = _evaluate_subjects(subject_rows, len(oof_ids))
    summary = {
        'schema_version': 'ppg_frailty.experiment_cell.v1',
        'status': 'passed',
        'repeat_index': int(split['repeat_index']),
        'fold_index': int(split['fold_index']),
        'split_seed': int(split['split_seed']),
        'training_seed': int(split['training_seed']),
        'representation_mode': config.representation_mode,
        'model_id': str(model_section['model_id']),
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
        'fitted_provenance': to_strict_json_value(asdict(training.provenance)),
        'sqi_calibrator_provenance': {
            'method': calibrator.method,
            'fitted_on_participant_ids': calibrator.fitted_on_participant_ids,
            'outer_oof_ids_absent': not bool(
                set(calibrator.fitted_on_participant_ids) & set(oof_ids)
            ),
        },
        'preprocessing_hash': preprocessing_hash,
        'feature_hash': feature_hash,
        'model_hash': training.provenance.state_hash,
        'elapsed_seconds': time.perf_counter() - started,
    }
    return _CellResult(summary, file_rows, subject_rows)


def _strict_json(path: Path, payload: Mapping[str, Any]) -> None:
    '''Write strict, deterministic JSON without silently accepting NaN/Infinity.

    严格、确定性地写入 JSON；拒绝 NaN/Infinity，避免下游审计得到非标准文件。
    '''
    from .provenance import atomic_write_json

    atomic_write_json(path, dict(payload), root=path.parent)


def _write_empty_oof(path: Path, reason: str) -> None:
    '''Materialize a schema-bearing empty parquet for a deliberately absent level.

    为当前表示层级不产生的 OOF 写入带模式的空 parquet，而不是伪造预测。
    '''
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment contract guard
        raise _ExperimentProtocolError('pyarrow_required_for_formal_oof') from exc
    table = pa.table(
        {
            'record_id': pa.array([], type=pa.string()),
            'empty_reason': pa.array([], type=pa.string()),
        }
    )
    metadata = dict(table.schema.metadata or {})
    metadata[b'scientific_empty_reason'] = str(reason).encode('utf-8')
    pq.write_table(table.replace_schema_metadata(metadata), path)


def _write_cell_artifacts(directory: Path, cell: _CellResult) -> None:
    '''Write the six mandatory, non-overwriting artifacts for one outer cell.

    为单个 outer cell 写入六个强制产物；目录必须预先不存在，从而禁止覆盖。
    '''
    imports = _runtime_imports()
    writer = imports['OofWriter']()
    directory.mkdir(parents=True, exist_ok=False)
    _write_empty_oof(
        directory / 'oof_window_predictions.parquet',
        'feature_vector_predictions_begin_at_file_level',
    )
    writer.write(cell.file_rows, directory / 'oof_file_predictions.parquet')
    writer.write(cell.subject_rows, directory / 'oof_subject_predictions.parquet')
    _write_empty_oof(
        directory / 'oof_member_predictions.parquet',
        'not_an_ensemble_model',
    )
    _strict_json(
        directory / 'metrics_per_fold_seed.json',
        {'schema_version': 1, 'cells': [dict(cell.summary)]},
    )
    _strict_json(
        directory / 'confusion_matrices.json',
        {
            'schema_version': 1,
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
            'schema_version': 1,
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
        directory / 'metrics_per_fold_seed.json',
        {'schema_version': 1, 'status': 'failed_closed', 'cells': []},
    )
    _strict_json(
        directory / 'confusion_matrices.json',
        {'schema_version': 1, 'status': 'failed_closed', 'cells': []},
    )
    _strict_json(directory / 'run_manifest.json', result.to_dict())


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
    scope = 'smoke_not_scientific_benchmark'
    default_name = (
        f'reduced_r{int(repeat_index)}_f{int(fold_index)}_{time.time_ns()}'
    )
    target = _resolve_output_directory(paths, output_dir, default_name)
    staging = target.with_name(f'.{target.name}.staging.{time.time_ns()}')
    try:
        try:
            cell = _execute_vector_cell(
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
                    'outer_train_only_calibrator': True,
                    'frozen_outer_split': True,
                    'record_seconds_cap': float(max_seconds_per_record),
                    'record_cap_per_participant': int(max_records_per_participant),
                    'fixed_epochs_override': int(fixed_epochs_override),
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
                },
                failure_reasons=(str(exc),),
            )
            _write_failed_artifacts(staging, result)
        _strict_json(staging / 'experiment_result.json', result.to_dict())
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
    file_rows = tuple(row for cell in cell_values for row in cell.file_rows)
    subject_rows = tuple(row for cell in cell_values for row in cell.subject_rows)
    _write_empty_oof(
        directory / 'oof_window_predictions.parquet',
        'feature_vector_predictions_begin_at_file_level',
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
    _write_empty_oof(
        directory / 'oof_member_predictions.parquet',
        'not_an_ensemble_model',
    )
    _strict_json(
        directory / 'metrics_per_fold_seed.json',
        {
            'schema_version': 1,
            'status': result.status,
            'cells': [dict(cell.summary) for cell in cell_values],
        },
    )
    _strict_json(
        directory / 'confusion_matrices.json',
        {
            'schema_version': 1,
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
    manifest = result.to_dict()
    manifest['schema_version'] = 1
    manifest['mandatory_artifacts'] = [
        'run_manifest.json',
        'metrics_per_fold_seed.json',
        'confusion_matrices.json',
        'oof_window_predictions.parquet',
        'oof_file_predictions.parquet',
        'oof_subject_predictions.parquet',
        'oof_member_predictions.parquet',
    ]
    _strict_json(directory / 'run_manifest.json', manifest)


def run_full_experiment(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    repeats: Iterable[int] = tuple(range(5)),
    folds: Iterable[int] = tuple(range(5)),
) -> ExperimentResult:
    '''Execute unshortened frozen outer cells, including the complete 5x5 grid.

    中文：本入口禁止记录截短、每人文件数裁剪和 epoch override；每个 cell
    独立训练，任何 fail-closed cell 都会使根结果失败，同时继续留下可审计证据。
    '''
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
    target = _resolve_output_directory(paths, output_dir, 'full_experiment')
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
                try:
                    cell = _execute_vector_cell(
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
                    )
                    cell.summary['scientific_scope'] = scope
                    passed_cells.append(cell)
                    summaries.append(dict(cell.summary))
                    _write_cell_artifacts(cell_directory, cell)
                except _ExperimentProtocolError as exc:
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
                    })
                    _write_failed_artifacts(cell_directory, cell_failure)
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
            },
            failure_reasons=tuple(failures),
        )
        _write_full_root_artifacts(staging, passed_cells, result)
        _strict_json(staging / 'experiment_result.json', result.to_dict())
        _commit_staging(staging, target)
        return result
    except Exception:
        if staging.exists():
            import shutil
            shutil.rmtree(staging)
        raise


__all__ = ["ExperimentResult", "run_full_experiment", "run_reduced_fold_experiment"]
