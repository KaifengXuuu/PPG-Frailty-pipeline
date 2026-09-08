"""Executable PTT motion/denoiser study and static peak-detector ablation.

This module deliberately composes the existing motion and artifact APIs.  It
does not duplicate the frailty experiment runner and never imports PTT labels
into the 29-participant motion-model fit.
"""
from __future__ import annotations
import copy
import csv
import json
import math
import platform
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import numpy as np
import yaml
from ..artifact import get_reducer
from ..data.external_manifest import M2_EXTERNAL_RELATIVE_PATH, PTT_DATASET_ID, ExternalRecord, adapt_ptt_synchronized_channels, load_m2_external_manifest
from ..peaks.pairing import match_events
from ..peaks.aboy_project_v2 import DETECTOR_ID as ABOY_V2_ID, IMPLEMENTATION_PATH as ABOY_V2_IMPLEMENTATION_PATH
from ..peaks.msptdfast_v2 import AUTHOR_SOURCE_SHA256 as MSPTDFAST_AUTHOR_SOURCE_SHA256, DETECTOR_ID as MSPTDFAST_V2_ID, IMPLEMENTATION_PATH as MSPTDFAST_IMPLEMENTATION_PATH, resolve_parameters as resolve_msptdfast_parameters
from ..peaks.resolver import CANONICAL_DETECTOR_ID, detect_pulses, resolve_detector_id, resolve_detector_parameters
from ..provenance import atomic_write_json, sha256_file, stable_payload_sha256
from ..signal.motion_imu import PTT_STATIC_CALIBRATION_ROLE, RollPitchEkfConfig, fit_motion_imu_calibration, preprocess_motion_imu_calibrated_ekf
from ..signal.preprocess import preprocess_ppg_pair
from ..signal.views import CanonicalSignalViews
from ..study.progress import NullProgressSink, ProgressEvent, ProgressSink
from .motion_reference import (
    PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH, PTT_IMU_UNIT_EVIDENCE_SHA256, PTT_SOURCE_ROOT,
    load_ptt_imu_unit_evidence, run_formal_internal_motion_reference, run_formal_internal_reverse_evaluation,
    run_formal_ptt_motion_training_ablation, run_formal_ptt_motion_reference,
)
from .motion_adapters import FormalMotionTrainerConfig, validate_formal_motion_cuda_device
from .motion import MOTION_DEPLOYMENT_THRESHOLD_FIT_SCOPE, MOTION_DEPLOYMENT_THRESHOLD_SCORE_ORIGIN
from .motion_runner import _deployment_threshold_from_oof

STAGE5_SCHEMA = 'ppg_frailty.stage5_pre_motion_ptt.v1'
PEAK_ABLATION_SCHEMA = 'ppg_frailty.stage_ablation_01_static_peaks.v3'
RESULT_SCHEMA = 'ppg_frailty.motion_peak_study_result.v1'
_STATIC_PEAK_STATISTICAL_METRICS: Mapping[str, tuple[str, str, str, str]] = {
    'recording_f1_percent': ('f1_percent', 'Beat-detection F1', '%', 'higher'),
    'recording_sensitivity_percent': ('sensitivity_percent', 'Beat-detection sensitivity', '%', 'higher'),
    'recording_positive_predictive_value_percent': ('positive_predictive_value_percent', 'Beat-detection positive predictive value', '%', 'higher'),
    'recording_ibi_ppi_rmse_ms': ('ibi_ppi_rmse_ms', 'IBI–PPI RMSE', 'ms', 'lower'),
    'execution_time_percent_of_ppg_signal_duration': ('execution_time_percent', 'Execution time relative to PPG duration', '% of signal duration', 'lower')
}
_STATIC_PEAK_STATISTICAL_METRIC_IDS = tuple(_STATIC_PEAK_STATISTICAL_METRICS)
_STATIC_PEAK_HOLM_SIDAK_FAMILY = 'all_selected_metrics_channels_and_reference_comparators'
MOTION_MODEL_COMPARISON_SCHEMA = 'ppg_frailty.motion_model_comparison_package.v1'
_PTT_COLUMNS = ('peaks', 'pleth_1', 'pleth_2', 'a_x', 'a_y', 'a_z', 'g_x', 'g_y', 'g_z')
_CHANNELS = ('RED', 'IR')

@dataclass(frozen=True)
class StudyPlan:
    path: Path
    payload: Mapping[str, Any]
    schema_version: str
    study_type: str
    study_id: str

# Plan validation fixes source, split, sampling, and comparison semantics before execution.
def _stage_progress(sink: ProgressSink, overall_current: int, overall_total: int, stage_label: str) -> Callable[[int, int, str], None]:
    def emit(current: int, total: int, detail: str) -> None:
        sink(
            ProgressEvent(event='motion_peak_subtask', current=overall_current, total=overall_total, case_id=stage_label, detail_current=current, detail_total=total,
                          detail_label=detail, message=f'running {stage_label}'))

    return emit

def _strict_json(path: Path, payload: object) -> None:
    atomic_write_json(path, payload, root=path.parent)

def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be numeric')
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f'{name} must be finite' + (' and positive' if positive else ''))
    return result

def load_motion_peak_plan(path: str | Path) -> StudyPlan:
    """Load either plan with registered-source and scientific-scope checks."""
    source = Path(path).resolve()
    data = yaml.safe_load(source.read_text(encoding='utf-8'))
    if not isinstance(data, Mapping):
        raise ValueError('motion/peak study plan must be a mapping')
    schema = str(data.get('schema_version', ''))
    study_type = str(data.get('study_type', ''))
    study_id = str(data.get('study_id', ''))
    if schema not in {STAGE5_SCHEMA, PEAK_ABLATION_SCHEMA} or not study_id:
        raise ValueError('unknown motion/peak study schema or empty study_id')
    expected_type = 'stage5_pre_motion_ptt' if schema == STAGE5_SCHEMA else 'stage_ablation_01_static_peak_detectors'
    if study_type != expected_type:
        raise ValueError('study_type disagrees with schema_version')
    dataset = data.get('ptt_dataset')
    if not isinstance(dataset, Mapping):
        raise ValueError('ptt_dataset must be a mapping')
    if dataset.get('dataset_id') != PTT_DATASET_ID or dataset.get('root') != PTT_SOURCE_ROOT.as_posix() or int(dataset.get('participant_count', -1)) != 22 or (int(
            dataset.get('record_count', -1)) != 66):
        raise ValueError('PTT plan must declare the registered 22-participant/66-record source')
    if schema == STAGE5_SCHEMA:
        motion_detector = data.get('motion_detector')
        if not isinstance(motion_detector, Mapping):
            raise ValueError('Stage5-pre requires motion_detector settings')
        training_device = motion_detector.get('training_device')
        if training_device is None:
            raise ValueError('Stage5-pre motion_detector.training_device is required and must be CUDA')
        trainer_config = FormalMotionTrainerConfig(device=str(training_device))
        trainer_config.validate()
        validate_formal_motion_cuda_device(trainer_config.device)
        reverse = motion_detector.get('reverse_ablation')
        if not isinstance(reverse, Mapping):
            raise ValueError('Stage5-pre requires motion_detector.reverse_ablation')
        if reverse.get('enabled') is not True or reverse.get('training_dataset') != PTT_DATASET_ID or reverse.get('evaluation_dataset') != 'frailty29' or (
                reverse.get('split_registry') != 'splits/ptt_formal_repeated_grouped_5x5_v2.csv') or (reverse.get('repeat_indices') != [0]) or (
                    reverse.get('folds') != [0, 1, 2, 3, 4]) or (reverse.get('split_seed') != 42) or (reverse.get('evaluation_fit_or_recalibration') is not False):
            raise ValueError('Stage5 reverse ablation must be PTT repeat-0 grouped five-fold training followed by frozen-model Frailty29 evaluation')
        comparison = data.get('motion_model_comparison')
        if not isinstance(comparison, Mapping):
            raise ValueError('Stage5-pre requires motion_model_comparison')
        legacy_study = Path(str(comparison.get('legacy_frailty29_stage5_study', '')))
        if legacy_study.is_absolute() or '..' in legacy_study.parts or comparison.get('candidates') != ['frailty29_trained_legacy_reference', 'ptt22_trained_reverse_ablation'] or (
                comparison.get('downstream_role') != 'comparison_only_single_factor_motion_detector_training_dataset') or (comparison.get('candidate_bound_preprocessing_required')
                                                                                                                           is not True):
            raise ValueError('Stage5 motion-model comparison declaration drift')
        benchmark = data.get('denoiser_benchmark')
        if not isinstance(benchmark, Mapping):
            raise ValueError('Stage5-pre requires denoiser_benchmark')
        reducers = benchmark.get('reducers')
        if not isinstance(reducers, list) or not reducers:
            raise ValueError('Stage5-pre requires one or more reducer IDs')
        for reducer_id in reducers:
            get_reducer(str(reducer_id))
        activities = benchmark.get('activities')
        if activities != ['sit', 'walk', 'run']:
            raise ValueError('Stage5-pre denoiser benchmark must separate sit/walk/run')
        _finite(benchmark.get('segment_s'), 'denoiser_benchmark.segment_s', positive=True)
        validation = benchmark.get('validation')
        if not isinstance(validation, Mapping):
            raise ValueError('Stage5-pre denoiser benchmark requires validation settings')
        for key in ('max_lag_s', 'lag_step_s', 'beat_tolerance_s'):
            _finite(validation.get(key), f'denoiser_benchmark.validation.{key}', positive=True)
        scoring_detector = resolve_detector_id(str(benchmark.get('scoring_peak_detector', CANONICAL_DETECTOR_ID)))
        resolve_detector_parameters(scoring_detector, benchmark.get('scoring_peak_detector_parameters'))
        if float(dataset.get('source_fs_hz', -1.0)) != 500.0 or float(dataset.get('pipeline_fs_hz', -1.0)) != 400.0:
            raise ValueError('Stage5-pre uses the registered PTT 500-to-400-Hz adapter')
    else:
        if data.get('activities') != ['sit']:
            raise ValueError('Stage-ablation-01 is a pure-static sit-only experiment')
        algorithms = data.get('algorithms')
        declared = {str(item.get('algorithm_id')): item for item in algorithms or () if isinstance(item, Mapping)}
        if set(declared) != {'aboy_project', MSPTDFAST_V2_ID}:
            raise ValueError('static peak ablation must compare only aboy_project (the registered v2 module) and default MSPTDfast v2')
        aboy = declared['aboy_project']
        if aboy.get('module_id') != ABOY_V2_ID or aboy.get('implementation') != ABOY_V2_IMPLEMENTATION_PATH:
            raise ValueError('Stage-ablation-01 aboy_project must reference the registered authoritative Aboy v2 module')
        if data.get('detector_input') != 'repaired_native_ppg_each_registered_module_owns_preprocessing':
            raise ValueError('Stage-ablation-01 detector_input must preserve module-owned preprocessing')
        paper_source = data.get('paper_source')
        if not isinstance(paper_source, Mapping) or paper_source.get('msptdfast_source_sha256') != MSPTDFAST_AUTHOR_SOURCE_SHA256:
            raise ValueError('static peak ablation must bind the reviewed author source')
        comparator = declared[MSPTDFAST_V2_ID]
        if comparator.get('module_id') != MSPTDFAST_V2_ID or comparator.get('implementation') != MSPTDFAST_IMPLEMENTATION_PATH or CANONICAL_DETECTOR_ID != MSPTDFAST_V2_ID:
            raise ValueError('Stage-ablation-01 must use the registered default MSPTDfast module')
        resolve_msptdfast_parameters(comparator.get('parameters'))
        validation = data.get('validation')
        if not isinstance(validation, Mapping):
            raise ValueError('static peak ablation requires validation settings')
        _finite(validation.get('beat_tolerance_s'), 'beat_tolerance_s', positive=True)
        _finite(validation.get('max_lag_s'), 'max_lag_s', positive=True)
        _finite(validation.get('lag_step_s'), 'lag_step_s', positive=True)
        _finite(validation.get('lag_window_s'), 'lag_window_s', positive=True)
        if float(validation['lag_window_s']) != 300.0 or float(
                validation['beat_tolerance_s']) != 0.15 or validation.get('evaluation_unit') != 'subject_recording_and_wavelength' or (
                    validation.get('aggregation') != 'median_and_interquartile_range_across_subject_recordings') or (validation.get('boxplot_whisker_percentiles') != [10, 90]) or (
                        validation.get('lag_tie_break') != 'maximum_ncorrect_then_smallest_absolute_lag_then_positive_lag'):
            raise ValueError('static peak assessment must use 300-s lag updates, +/-150-ms one-to-one matching, and recording-level median/IQR reporting')
        statistical = validation.get('statistical_comparison')
        if not isinstance(statistical, Mapping):
            raise ValueError('static peak statistical-comparison contract drifted')
        selected_metrics = statistical.get('metrics')
        if selected_metrics is None:
            selected_metrics = [statistical.get('metric')]
        if statistical.get('reference_algorithm_id') != MSPTDFAST_V2_ID or statistical.get('test') != 'wilcoxon_rank_sum_two_sided' or statistical.get(
                'multiple_comparison_correction') != 'holm_sidak_step_down' or (float(statistical.get('alpha', -1.0)) != 0.05) or (not isinstance(selected_metrics, list)) or (
                    not selected_metrics) or (len(selected_metrics) != len(set(selected_metrics))) or (not set(selected_metrics) <= set(_STATIC_PEAK_STATISTICAL_METRIC_IDS)) or (
                        statistical.get('family_definition', _STATIC_PEAK_HOLM_SIDAK_FAMILY) != _STATIC_PEAK_HOLM_SIDAK_FAMILY):
            raise ValueError('static peak statistical-comparison contract drifted')
    return StudyPlan(source, dict(data), schema, study_type, study_id)

# Beat alignment and lag fitting are pure numerical kernels shared by both studies.
def _window_starts(length: int, window: int, hop: int) -> tuple[int, ...]:
    if length <= 0 or window <= 0 or hop <= 0:
        raise ValueError('window inputs must be positive')
    if length <= window:
        return (0, )
    starts = list(range(0, length - window + 1, hop))
    right = length - window
    if starts[-1] != right:
        starts.append(right)
    return tuple(starts)

def _matched_pairs(reference_s: np.ndarray, predicted_s: np.ndarray, *, tolerance_s: float) -> tuple[np.ndarray, np.ndarray]:
    raw_reference = np.asarray(reference_s, dtype=np.float64)
    raw_predicted = np.asarray(predicted_s, dtype=np.float64)
    reference_order = np.argsort(raw_reference, kind='stable')
    predicted_order = np.argsort(raw_predicted, kind='stable')
    reference = raw_reference[reference_order]
    predicted = raw_predicted[predicted_order]
    used = np.zeros(predicted.size, dtype=bool)
    pairs: list[tuple[int, int]] = []
    for reference_index, event in enumerate(reference):
        candidates = np.flatnonzero(~used & (np.abs(predicted - event) <= tolerance_s))
        if candidates.size:
            predicted_index = int(candidates[np.argmin(np.abs(predicted[candidates] - event))])
            used[predicted_index] = True
            pairs.append((int(reference_order[reference_index]), int(predicted_order[predicted_index])))
    if not pairs:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    return tuple(np.asarray(pairs, dtype=np.int64).T)

def _best_lag_grid(reference_s: np.ndarray, predicted_s: np.ndarray, *, max_lag_s: float, lag_step_s: float, tolerance_s: float) -> float:
    reference = np.sort(np.asarray(reference_s, dtype=np.float64))
    predicted = np.sort(np.asarray(predicted_s, dtype=np.float64))
    if reference.size == 0 or predicted.size == 0:
        return 0.0
    lags = np.arange(-max_lag_s, max_lag_s + lag_step_s * 0.5, lag_step_s, dtype=np.float64)
    support = np.zeros(lags.size, dtype=np.int64)
    grid_start = float(lags[0])
    for event in reference:
        deltas = predicted - float(event)
        deltas = deltas[(deltas >= -max_lag_s - tolerance_s) & (deltas <= max_lag_s + tolerance_s)]
        if deltas.size == 0:
            continue
        difference = np.zeros(lags.size + 1, dtype=np.int64)
        lower = np.ceil((deltas - tolerance_s - grid_start) / lag_step_s - 1e-12).astype(np.int64)
        upper = np.floor((deltas + tolerance_s - grid_start) / lag_step_s + 1e-12).astype(np.int64)
        lower = np.clip(lower, 0, lags.size - 1)
        upper = np.clip(upper, 0, lags.size - 1)
        valid = lower <= upper
        np.add.at(difference, lower[valid], 1)
        np.add.at(difference, upper[valid] + 1, -1)
        support += np.cumsum(difference[:-1]) > 0
    order = sorted(range(lags.size), key=lambda index: (-int(support[index]), abs(float(lags[index])), -float(lags[index])))
    best_key = (-1, float('-inf'), float('-inf'))
    best_lag = 0.0
    for index in order:
        if int(support[index]) < best_key[0]:
            break
        lag = float(lags[index])
        metrics = match_events(reference + lag, predicted, tolerance_s=tolerance_s)
        key = (int(metrics.true_positive), -abs(lag), lag)
        if key > best_key:
            best_key = key
            best_lag = lag
    return best_lag

def _piecewise_shift_reference(reference_s: np.ndarray, predicted_s: np.ndarray, *, max_lag_s: float, lag_step_s: float, tolerance_s: float, lag_window_s: float | None,
                               recording_duration_s: float | None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    reference = np.sort(np.asarray(reference_s, dtype=np.float64))
    predicted = np.sort(np.asarray(predicted_s, dtype=np.float64))
    if lag_window_s is None:
        starts = (0.0, )
        base_duration = max(float(recording_duration_s or 0.0), float(reference[-1]) if reference.size else 0.0, float(predicted[-1]) if predicted.size else 0.0)
        duration = np.nextafter(base_duration, float('inf'))
        window_length = max(duration, np.finfo(np.float64).eps)
    else:
        if not math.isfinite(lag_window_s) or lag_window_s <= 0.0:
            raise ValueError('lag_window_s must be finite and positive')
        base_duration = max(float(recording_duration_s or 0.0), float(reference[-1]) if reference.size else 0.0, float(predicted[-1]) if predicted.size else 0.0)
        duration = np.nextafter(base_duration, float('inf'))
        window_count = max(1, int(math.ceil(base_duration / lag_window_s)))
        starts = tuple((float(index) * lag_window_s for index in range(window_count)))
        window_length = float(lag_window_s)
    shifted = reference.copy()
    audit_rows: list[dict[str, Any]] = []
    for window_index, start in enumerate(starts):
        stop = min(float(start) + window_length, duration)
        reference_mask = (reference >= start) & (reference < stop)
        current_reference = reference[reference_mask]
        predicted_mask = (predicted >= start - max_lag_s - tolerance_s) & (predicted < stop + max_lag_s + tolerance_s)
        current_predicted = predicted[predicted_mask]
        lag = _best_lag_grid(current_reference, current_predicted, max_lag_s=max_lag_s, lag_step_s=lag_step_s, tolerance_s=tolerance_s)
        shifted[reference_mask] += lag
        preliminary = match_events(current_reference + lag, current_predicted, tolerance_s=tolerance_s)
        audit_rows.append({
            'lag_window_index': int(window_index), 'lag_window_start_s': float(start), 'lag_window_stop_s': float(stop), 'lag_s': float(lag),
            'reference_beat_count': int(current_reference.size), 'candidate_ppg_beat_count': int(current_predicted.size),
            'preliminary_correct_count': int(preliminary.true_positive)
        })
    return (shifted, audit_rows)

def align_and_score_beats(reference_s: np.ndarray, predicted_s: np.ndarray, *, max_lag_s: float = 10.0, lag_step_s: float = 0.02, tolerance_s: float = 0.2,
                          lag_window_s: float | None = None, recording_duration_s: float | None = None) -> dict[str, Any]:
    """One-to-one beat scoring with optional within-record drifting lag."""
    reference = np.sort(np.asarray(reference_s, dtype=np.float64))
    predicted = np.sort(np.asarray(predicted_s, dtype=np.float64))
    shifted, lag_windows = _piecewise_shift_reference(reference, predicted, max_lag_s=max_lag_s, lag_step_s=lag_step_s, tolerance_s=tolerance_s, lag_window_s=lag_window_s,
                                                      recording_duration_s=recording_duration_s)
    metrics = match_events(shifted, predicted, tolerance_s=tolerance_s)
    reference_indices, predicted_indices = _matched_pairs(shifted, predicted, tolerance_s=tolerance_s)
    chronological = np.argsort(reference_indices, kind='stable')
    reference_indices = reference_indices[chronological]
    predicted_indices = predicted_indices[chronological]
    interval_errors: list[float] = []
    for index in range(1, reference_indices.size):
        if reference_indices[index] == reference_indices[index - 1] + 1 and predicted_indices[index] == predicted_indices[index - 1] + 1:
            ibi = reference[reference_indices[index]] - reference[reference_indices[index - 1]]
            ppi = predicted[predicted_indices[index]] - predicted[predicted_indices[index - 1]]
            interval_errors.append(float(ppi - ibi))
    errors = np.asarray(interval_errors, dtype=np.float64)
    lag_values = [float(row['lag_s']) for row in lag_windows]
    return {
        'lag_s': float(np.median(lag_values)) if lag_values else 0.0, 'lag_windows': lag_windows, 'nref': int(reference.size), 'nppg': int(predicted.size),
        'ncorrect': int(metrics.true_positive), 'true_positives': metrics.true_positive, 'false_positives': metrics.false_positive, 'false_negatives': metrics.false_negative,
        'sensitivity': metrics.recall, 'positive_predictive_value': metrics.precision, 'f1': metrics.f1, 'sensitivity_percent': 100.0 * metrics.recall,
        'positive_predictive_value_percent': 100.0 * metrics.precision, 'f1_percent': 100.0 * metrics.f1, 'beat_tolerance_s': float(tolerance_s),
        'timing_mae_s': metrics.timing_mae_s, 'matched_interval_count': int(errors.size),
        'ibi_ppi_rmse_ms': float(np.sqrt(np.mean(np.square(errors))) * 1000.0) if errors.size else None,
        'ibi_ppi_mae_ms': float(np.mean(np.abs(errors)) * 1000.0) if errors.size else None
    }

def _ptt_records(repository_root: Path) -> tuple[ExternalRecord, ...]:
    records = tuple((row for row in load_m2_external_manifest(repository_root / M2_EXTERNAL_RELATIVE_PATH) if row.dataset_id == PTT_DATASET_ID))
    if len(records) != 66 or len({row.subject_id for row in records}) != 22:
        raise ValueError('PTT source roster is not 66 records / 22 participants')
    return records

def _load_record(repository_root: Path, row: ExternalRecord) -> tuple[Any, np.ndarray]:
    path = (repository_root / PTT_SOURCE_ROOT / row.canonical_representation).resolve()
    if sha256_file(path) != row.checksum_sha256:
        raise ValueError(f'PTT source hash mismatch: {row.record_id}')
    with path.open('r', encoding='utf-8', newline='') as handle:
        header = tuple(next(csv.reader(handle)))
    indices = tuple((header.index(name) for name in _PTT_COLUMNS))
    values = np.loadtxt(path, delimiter=',', skiprows=1, usecols=indices, ndmin=2)
    if values.shape[1] != len(_PTT_COLUMNS) or not np.isfinite(values).all():
        raise ValueError(f'PTT numeric source invalid: {row.record_id}')
    columns = {name: values[:, index] for index, name in enumerate(_PTT_COLUMNS)}
    adapted = adapt_ptt_synchronized_channels(
        {
            'pleth_1': columns['pleth_1'], 'pleth_2': columns['pleth_2'], 'AX': columns['a_x'], 'AY': columns['a_y'], 'AZ': columns['a_z'], 'GX': columns['g_x'],
            'GY': columns['g_y'], 'GZ': columns['g_z']
        }, external_record=row, observed_source_file_sha256=row.checksum_sha256, additional_channel_order=('AX', 'AY', 'AZ', 'GX', 'GY', 'GZ'))
    reference_times = np.flatnonzero(columns['peaks'] > 0.5).astype(np.float64) / 500.0
    return (adapted, reference_times)

def _slice_processed(processed: Mapping[str, Any], start: int, stop: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, value in processed.items():
        array = np.asarray(value)
        if array.ndim >= 1 and array.shape[0] >= stop:
            output[name] = array[start:stop]
    return output

def _motion_processed(motion: Any) -> dict[str, np.ndarray]:
    motion.validate()
    values = np.asarray(motion.values, dtype=np.float64)
    return {
        'dynamic_acc_mps2': values[:, 0:3], 'gyro_rads': values[:, 3:6], 'dynamic_magnitude': values[:, 6], 'gyro_magnitude': values[:, 7], 'jerk_magnitude': values[:, 8],
        'imu_valid_mask': np.asarray(motion.valid_mask, dtype=bool)
    }

def _score_segment(values: np.ndarray | CanonicalSignalViews, reference_s: np.ndarray, *, algorithm_id: str, algorithm_parameters: Mapping[str, Any] | None = None, fs_hz: float,
                   validation: Mapping[str,
                                       Any], wavelength: str = 'RED', lag_window_s: float | None = None, recording_duration_s: float | None = None) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    pulse = detect_pulses(values, detector_id=algorithm_id, detector_parameters=algorithm_parameters, fs_hz=fs_hz, wavelength=wavelength, min_observation_sec=6.0, min_peaks=3)
    peaks_s = np.asarray(pulse.peak_timestamps_s, dtype=np.float64)
    elapsed = time.perf_counter() - started
    scored = align_and_score_beats(reference_s, peaks_s, max_lag_s=float(validation['max_lag_s']), lag_step_s=float(validation['lag_step_s']),
                                   tolerance_s=float(validation['beat_tolerance_s']), lag_window_s=lag_window_s, recording_duration_s=recording_duration_s)
    return (scored, elapsed)

def _nonnegative_integer(value: Any, field: str) -> int:
    if value is None or isinstance(value, bool):
        raise ValueError(f'{field} must be an integer count')
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or (not numeric.is_integer()):
        raise ValueError(f'{field} must be an integer count')
    return int(numeric)

def _passed_segment_counts(row: Mapping[str, Any]) -> tuple[int, int, int]:
    tp_source = row.get('true_positives') if row.get('true_positives') is not None else row.get('ncorrect')
    true_positive = _nonnegative_integer(tp_source, 'true_positives/ncorrect')
    if row.get('false_positives') is not None:
        false_positive = _nonnegative_integer(row['false_positives'], 'false_positives')
    elif row.get('nppg') is not None:
        false_positive = _nonnegative_integer(row['nppg'], 'nppg') - true_positive
    else:
        raise ValueError('passed denoiser row lacks false_positives/nppg')
    if row.get('false_negatives') is not None:
        false_negative = _nonnegative_integer(row['false_negatives'], 'false_negatives')
    elif row.get('nref') is not None:
        false_negative = _nonnegative_integer(row['nref'], 'nref') - true_positive
    else:
        raise ValueError('passed denoiser row lacks false_negatives/nref')
    if false_positive < 0 or false_negative < 0:
        raise ValueError('denoiser beat totals cannot be smaller than true positives')
    redundant_totals = (('ncorrect', true_positive), ('nppg', true_positive + false_positive), ('nref', true_positive + false_negative))
    for field, expected in redundant_totals:
        if row.get(field) is not None and _nonnegative_integer(row[field], field) != expected:
            raise ValueError(f'{field} disagrees with TP/FP/FN counts')
    return (true_positive, false_positive, false_negative)

def _runtime_seconds(row: Mapping[str, Any]) -> float:
    runtime = _finite(row.get('runtime_s'), 'runtime_s')
    if runtime < 0.0:
        raise ValueError('runtime_s must be non-negative')
    return runtime

# Aggregators preserve recording-level evidence; presentation remains in reporting.specialized.
def _aggregate_benchmark(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(str(row['algorithm_or_reducer']), str(row['activity_group']), str(row['channel'])) for row in rows})
    output: list[dict[str, Any]] = []
    for algorithm, activity_group, channel in keys:
        attempted = [row for row in rows if (str(row['algorithm_or_reducer']), str(row['activity_group']), str(row['channel'])) == (algorithm, activity_group, channel)]
        if not attempted:
            continue
        if any((not str(row.get('participant_id', '')).strip() for row in attempted)):
            raise ValueError('denoiser benchmark rows require participant_id')
        if any((not str(row.get('status', '')).strip() for row in attempted)):
            raise ValueError('denoiser benchmark rows require status')
        passed = [row for row in attempted if row.get('status') == 'passed']
        failed = [row for row in attempted if row.get('status') != 'passed']
        attempted_participants = sorted({str(row['participant_id']) for row in attempted})
        passed_participants = {str(row['participant_id']) for row in passed}
        failed_participants = {str(row['participant_id']) for row in failed}
        all_failed_participants = set(attempted_participants) - passed_participants
        participant_rows: list[dict[str, float | None]] = []
        total_matched_interval_count = 0
        total_rmse_evaluable_segment_count = 0
        for participant in sorted(passed_participants):
            current = [row for row in passed if str(row['participant_id']) == participant]
            true_positive = 0
            false_positive = 0
            false_negative = 0
            interval_count = 0
            interval_sse_ms2 = 0.0
            rmse_evaluable_segment_count = 0
            for row in current:
                row_tp, row_fp, row_fn = _passed_segment_counts(row)
                true_positive += row_tp
                false_positive += row_fp
                false_negative += row_fn
                row_interval_count = _nonnegative_integer(row.get('matched_interval_count'), 'matched_interval_count')
                row_rmse = row.get('ibi_ppi_rmse_ms')
                if row_interval_count == 0:
                    if row_rmse is not None:
                        raise ValueError('ibi_ppi_rmse_ms requires matched_interval_count > 0')
                else:
                    rmse = _finite(row_rmse, 'ibi_ppi_rmse_ms')
                    if rmse < 0.0:
                        raise ValueError('ibi_ppi_rmse_ms must be non-negative')
                    interval_sse_ms2 += rmse * rmse * row_interval_count
                    interval_count += row_interval_count
                    rmse_evaluable_segment_count += 1
            sensitivity_denominator = true_positive + false_negative
            precision_denominator = true_positive + false_positive
            f1_denominator = 2 * true_positive + false_positive + false_negative
            total_matched_interval_count += interval_count
            total_rmse_evaluable_segment_count += rmse_evaluable_segment_count
            participant_rows.append({
                'f1': 2.0 * true_positive / f1_denominator if f1_denominator else 0.0, 'sensitivity': true_positive / sensitivity_denominator if sensitivity_denominator else 0.0,
                'positive_predictive_value': true_positive / precision_denominator if precision_denominator else 0.0,
                'rmse': math.sqrt(interval_sse_ms2 / interval_count) if interval_count else None
            })
        participant_f1 = [float(row['f1']) for row in participant_rows]
        participant_sensitivity = [float(row['sensitivity']) for row in participant_rows]
        participant_ppv = [float(row['positive_predictive_value']) for row in participant_rows]
        participant_rmse = [float(row['rmse']) for row in participant_rows if row['rmse'] is not None]
        passed_runtime = math.fsum((_runtime_seconds(row) for row in passed))
        failed_runtime = math.fsum((_runtime_seconds(row) for row in failed))
        output.append({
            'algorithm_or_reducer': algorithm, 'activity_group': activity_group, 'channel': channel,
            'endpoint_aggregation': 'pool_tp_fp_fn_and_interval_sse_within_participant_then_macro', 'attempted_participant_count': len(attempted_participants),
            'passed_participant_count': len(passed_participants), 'failed_participant_count': len(failed_participants),
            'all_failed_participant_count': len(all_failed_participants), 'partially_failed_participant_count': len(failed_participants & passed_participants),
            'participant_coverage_rate': len(passed_participants) / len(attempted_participants), 'attempted_segment_count': len(attempted), 'passed_segment_count': len(passed),
            'failed_segment_count': len(failed), 'segment_coverage_rate': len(passed) / len(attempted), 'rmse_evaluable_participant_count': len(participant_rmse),
            'rmse_evaluable_segment_count': total_rmse_evaluable_segment_count, 'matched_interval_count': total_matched_interval_count, 'participant_count': len(participant_rows),
            'segment_count': len(passed), 'participant_macro_f1': float(np.mean(participant_f1)) if participant_f1 else None,
            'participant_macro_f1_sd': float(np.std(participant_f1, ddof=1)) if len(participant_f1) >= 2 else None,
            'participant_macro_sensitivity': float(np.mean(participant_sensitivity)) if participant_sensitivity else None,
            'participant_macro_sensitivity_sd': float(np.std(participant_sensitivity, ddof=1)) if len(participant_sensitivity) >= 2 else None,
            'participant_macro_positive_predictive_value': float(np.mean(participant_ppv)) if participant_ppv else None,
            'participant_macro_positive_predictive_value_sd': float(np.std(participant_ppv, ddof=1)) if len(participant_ppv) >= 2 else None,
            'participant_macro_ibi_ppi_rmse_ms': float(np.mean(participant_rmse)) if participant_rmse else None,
            'participant_macro_ibi_ppi_rmse_ms_sd': float(np.std(participant_rmse, ddof=1)) if len(participant_rmse) >= 2 else None, 'passed_runtime_s': passed_runtime,
            'failed_runtime_s': failed_runtime, 'total_runtime_s': passed_runtime + failed_runtime
        })
    return output

def _distribution_summary(values: Sequence[float]) -> dict[str, float | None]:
    finite = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if finite.size == 0:
        return {'median': None, 'q1': None, 'q3': None, 'iqr': None, 'p10': None, 'p90': None}
    q10, q25, q50, q75, q90 = np.percentile(finite, [10, 25, 50, 75, 90])
    return {'median': float(q50), 'q1': float(q25), 'q3': float(q75), 'iqr': float(q75 - q25), 'p10': float(q10), 'p90': float(q90)}

def _aggregate_static_peak_benchmark(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    metrics = {
        'recording_f1_percent': 'f1_percent', 'recording_sensitivity_percent': 'sensitivity_percent',
        'recording_positive_predictive_value_percent': 'positive_predictive_value_percent', 'recording_ibi_ppi_rmse_ms': 'ibi_ppi_rmse_ms',
        'execution_time_percent': 'execution_time_percent'
    }
    keys = sorted({(str(row['algorithm_or_reducer']), str(row['channel'])) for row in rows if row.get('status') == 'passed'})
    output: list[dict[str, Any]] = []
    for algorithm, channel in keys:
        selected = [row for row in rows if row.get('status') == 'passed' and str(row['algorithm_or_reducer']) == algorithm and (str(row['channel']) == channel)]
        summary: dict[str, Any] = {
            'algorithm_or_reducer': algorithm, 'activity_group': 'static', 'channel': channel, 'participant_count': len({str(row['participant_id'])
                                                                                                                         for row in selected}),
            'subject_recording_count': len({(str(row['participant_id']), str(row['record_id']))
                                            for row in selected}), 'aggregation': 'median_iqr_across_subject_recordings', 'boxplot_whiskers': 'p10_p90'
        }
        for output_name, source_name in metrics.items():
            distribution = _distribution_summary([float(row[source_name]) for row in selected if row.get(source_name) is not None])
            for statistic, value in distribution.items():
                summary[f'{output_name}_{statistic}'] = value
        output.append(summary)
    return output

def _holm_sidak_step_down(p_values: Sequence[float], *, alpha: float) -> tuple[list[float], list[bool], list[int]]:
    count = len(p_values)
    adjusted = [1.0] * count
    rejected = [False] * count
    ranks = [0] * count
    order = sorted(range(count), key=lambda index: (float(p_values[index]), index))
    running_adjusted = 0.0
    rejection_open = True
    for position, index in enumerate(order):
        remaining = count - position
        raw = min(max(float(p_values[index]), 0.0), 1.0)
        candidate = 1.0 - (1.0 - raw)**remaining
        running_adjusted = max(running_adjusted, candidate)
        adjusted[index] = min(running_adjusted, 1.0)
        critical = 1.0 - (1.0 - float(alpha))**(1.0 / remaining)
        rejected[index] = bool(rejection_open and raw <= critical)
        if not rejected[index]:
            rejection_open = False
        ranks[index] = position + 1
    return (adjusted, rejected, ranks)

def _static_peak_rank_sum_comparisons(rows: Sequence[Mapping[str, Any]], *, reference_algorithm_id: str, alpha: float,
                                      metric_ids: Sequence[str] = _STATIC_PEAK_STATISTICAL_METRIC_IDS, registered_metric_ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
    from scipy.stats import ranksums
    selected_metric_ids = tuple((str(value) for value in metric_ids))
    if not selected_metric_ids or len(selected_metric_ids) != len(set(selected_metric_ids)) or (not set(selected_metric_ids) <= set(_STATIC_PEAK_STATISTICAL_METRICS)):
        raise ValueError('static peak rank-sum metric roster is invalid')
    registered = {str(value) for value in (selected_metric_ids if registered_metric_ids is None else registered_metric_ids)}
    if not registered <= set(selected_metric_ids):
        raise ValueError('registered metrics must be a subset of selected metrics')
    passed = [row for row in rows if row.get('status') == 'passed']
    algorithms = sorted({str(row['algorithm_or_reducer']) for row in passed if str(row['algorithm_or_reducer']) != reference_algorithm_id})
    output: list[dict[str, Any]] = []
    for metric_id in selected_metric_ids:
        row_field, metric_label, metric_unit, better_direction = _STATIC_PEAK_STATISTICAL_METRICS[metric_id]
        for channel in _CHANNELS:
            reference_by_record = {(str(row['participant_id']), str(row['record_id'])): float(row[row_field])
                                   for row in passed if str(row['algorithm_or_reducer']) == reference_algorithm_id and str(row['channel']) == channel and (
                                       row.get(row_field) is not None) and math.isfinite(float(row[row_field]))}
            for comparator in algorithms:
                comparator_by_record = {(str(row['participant_id']), str(row['record_id'])): float(row[row_field])
                                        for row in passed if str(row['algorithm_or_reducer']) == comparator and str(row['channel']) == channel and (
                                            row.get(row_field) is not None) and math.isfinite(float(row[row_field]))}
                common = sorted(set(reference_by_record) & set(comparator_by_record))
                reference_values = [reference_by_record[key] for key in common]
                comparator_values = [comparator_by_record[key] for key in common]
                if reference_values and comparator_values:
                    statistic, p_value = ranksums(reference_values, comparator_values, alternative='two-sided')
                    reference_median = float(np.median(reference_values))
                    comparator_median = float(np.median(comparator_values))
                    reference_advantage = reference_median - comparator_median if better_direction == 'higher' else comparator_median - reference_median
                else:
                    statistic, p_value = (float('nan'), float('nan'))
                    reference_median = comparator_median = None
                    reference_advantage = None
                output.append({
                    'reference_algorithm': reference_algorithm_id, 'comparator_algorithm': comparator, 'channel': channel, 'metric': metric_id, 'metric_label': metric_label,
                    'metric_unit': metric_unit, 'better_direction': better_direction,
                    'analysis_registration': 'prespecified_in_resolved_plan' if metric_id in registered else 'retrospective_supplement_requested_2026-08-24',
                    'test': 'wilcoxon_rank_sum_two_sided', 'scipy_function': 'scipy.stats.ranksums', 'alternative': 'two-sided', 'common_subject_recordings': len(common),
                    'identical_record_roster_enforced': True, 'pairing_used_by_test': False,
                    'ties_present': len(set(reference_values + comparator_values)) < len(reference_values + comparator_values), 'reference_median': reference_median,
                    'comparator_median': comparator_median, 'reference_advantage': reference_advantage,
                    'reference_advantage_definition': 'reference_minus_comparator' if better_direction == 'higher' else 'comparator_minus_reference',
                    'rank_sum_z': float(statistic) if math.isfinite(float(statistic)) else None, 'p_value': float(p_value) if math.isfinite(float(p_value)) else None
                })
    valid_indices = [index for index, row in enumerate(output) if row['p_value'] is not None]
    adjusted, rejected, ranks = _holm_sidak_step_down([float(output[index]['p_value']) for index in valid_indices], alpha=alpha)
    for local_index, output_index in enumerate(valid_indices):
        output[output_index].update({
            'holm_sidak_family_size': len(valid_indices), 'holm_sidak_family_definition': _STATIC_PEAK_HOLM_SIDAK_FAMILY, 'holm_sidak_rank': ranks[local_index],
            'holm_sidak_adjusted_p': adjusted[local_index], 'reject_at_alpha': rejected[local_index], 'alpha': float(alpha)
        })
    registered_indices = [index for index in valid_indices if str(output[index]['metric']) in registered]
    if registered_indices:
        registered_adjusted, registered_rejected, registered_ranks = _holm_sidak_step_down([float(output[index]['p_value']) for index in registered_indices], alpha=alpha)
        for local_index, output_index in enumerate(registered_indices):
            output[output_index].update({
                'registered_family_size': len(registered_indices), 'registered_family_holm_sidak_rank': registered_ranks[local_index],
                'registered_family_holm_sidak_adjusted_p': registered_adjusted[local_index], 'registered_family_reject_at_alpha': registered_rejected[local_index]
            })
    return output

def run_ptt_denoiser_benchmark(repository_root: str | Path, *, reducer_ids: Sequence[str], segment_s: float, validation: Mapping[str, Any],
                               activities: Sequence[str] = ('sit', 'walk', 'run'), scoring_peak_detector: str = CANONICAL_DETECTOR_ID,
                               scoring_peak_detector_parameters: Mapping[str, Any] | None = None,
                               progress_callback: Callable[[int, int, str], None] | None = None) -> dict[str, Any]:
    """Run implemented reducers on all declared PTT records, sequentially."""
    repository = Path(repository_root).resolve()
    detector_id = resolve_detector_id(scoring_peak_detector)
    detector_parameters = resolve_detector_parameters(detector_id, scoring_peak_detector_parameters)
    records = _ptt_records(repository)
    unit_evidence = load_ptt_imu_unit_evidence(repository / PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH, expected_sha256=PTT_IMU_UNIT_EVIDENCE_SHA256, expected_records=records)
    by_subject: dict[str, dict[str, ExternalRecord]] = {}
    for row in records:
        by_subject.setdefault(row.subject_id, {})[row.activity_raw.lower()] = row
    fs_hz = 400.0
    segment_samples = int(round(float(segment_s) * fs_hz))
    rows: list[dict[str, Any]] = []
    subjects = sorted(by_subject.items())
    for subject_index, (subject_id, activity_records) in enumerate(subjects):
        if progress_callback is not None:
            progress_callback(subject_index, len(subjects), f'benchmark PTT participant {subject_id}')
        loaded = {activity: _load_record(repository, activity_records[activity]) for activity in ('sit', 'walk', 'run')}
        sit = loaded['sit'][0]
        calibration = fit_motion_imu_calibration(sit.values[:, 2:5], sit.values[:, 5:8], participant_id=subject_id, file_id=activity_records['sit'].record_id,
                                                 source_role=PTT_STATIC_CALIBRATION_ROLE, fs_hz=fs_hz, acceleration_unit=unit_evidence.acceleration_unit,
                                                 gyroscope_unit=unit_evidence.gyroscope_unit, config=RollPitchEkfConfig())
        for activity in activities:
            adapted, reference_times = loaded[str(activity)]
            motion = preprocess_motion_imu_calibrated_ekf(adapted.values[:, 2:5], adapted.values[:, 5:8], fs_hz=fs_hz, acceleration_unit=unit_evidence.acceleration_unit,
                                                          gyroscope_unit=unit_evidence.gyroscope_unit, participant_id=subject_id, calibration=calibration,
                                                          config=RollPitchEkfConfig())
            _, filtered, _ = preprocess_ppg_pair(adapted.ppg_red_ir, fs_hz=fs_hz, timestamps_s=adapted.timestamps_s)
            for start in _window_starts(filtered.shape[0], segment_samples, segment_samples):
                stop = min(start + segment_samples, filtered.shape[0])
                if stop - start < int(round(8.0 * fs_hz)):
                    continue
                reference = reference_times[(reference_times >= start / fs_hz) & (reference_times < stop / fs_hz)] - start / fs_hz
                if reference.size < 3:
                    continue
                ppg = filtered[start:stop]
                imu = _slice_processed(_motion_processed(motion), start, stop)
                for reducer_id in reducer_ids:
                    started = time.perf_counter()
                    result = get_reducer(str(reducer_id)).reduce(ppg, imu, fs_hz=fs_hz)
                    reduction_s = time.perf_counter() - started
                    if result.status != 'success' or result.x_ar is None:
                        for channel in _CHANNELS:
                            rows.append({
                                'participant_id': subject_id, 'record_id': activity_records[str(activity)].record_id, 'activity': activity,
                                'activity_group': 'static' if activity == 'sit' else 'dynamic', 'channel': channel, 'algorithm_or_reducer': str(reducer_id),
                                'segment_start_s': start / fs_hz, 'status': result.status, 'failure_reasons': list(result.reasons), 'runtime_s': reduction_s
                            })
                        continue
                    for channel_index, channel in enumerate(_CHANNELS):
                        scored, detector_s = _score_segment(
                            np.asarray(result.x_ar)[:, channel_index], reference, algorithm_id=detector_id, algorithm_parameters=detector_parameters, fs_hz=fs_hz,
                            validation=validation)
                        rows.append({
                            'participant_id': subject_id, 'record_id': activity_records[str(activity)].record_id, 'activity': activity,
                            'activity_group': 'static' if activity == 'sit' else 'dynamic', 'channel': channel, 'algorithm_or_reducer': str(reducer_id),
                            'segment_start_s': start / fs_hz, 'status': 'passed', 'runtime_s': reduction_s + detector_s, **scored
                        })
    if progress_callback is not None:
        progress_callback(len(subjects), len(subjects), 'completed PTT benchmark')
    return {
        'schema_version': 'ppg_frailty.stage5_pre_denoiser_benchmark.v2', 'status': 'passed', 'participant_count': len(by_subject), 'record_count': len(records),
        'activities': list(activities), 'segment_s': float(segment_s), 'reducers': list(reducer_ids), 'scoring_peak_detector': detector_id,
        'scoring_peak_detector_parameters': detector_parameters, 'validation': dict(validation),
        'summary_aggregation': 'pool_tp_fp_fn_and_interval_sse_within_participant_then_macro', 'rows': rows, 'summary_rows': _aggregate_benchmark(rows)
    }

def run_static_peak_ablation(repository_root: str | Path, plan: StudyPlan, progress_callback: Callable[[int, int, str], None] | None = None) -> dict[str, Any]:
    """Run two registered detectors once per complete PTT sit recording."""
    if plan.schema_version != PEAK_ABLATION_SCHEMA:
        raise ValueError('static peak runner received a non-ablation plan')
    repository = Path(repository_root).resolve()
    records = [row for row in _ptt_records(repository) if row.activity_raw.lower() == 'sit']
    validation = dict(plan.payload['validation'])
    algorithm_specs = [(str(item['algorithm_id']), str(item['module_id']), dict(item.get('parameters', {}))) for item in plan.payload['algorithms']]
    rows: list[dict[str, Any]] = []
    lag_window_s = float(validation['lag_window_s'])
    for record_index, row in enumerate(records):
        if progress_callback is not None:
            progress_callback(record_index, len(records), f'score static PTT participant {row.subject_id}')
        adapted, reference = _load_record(repository, row)
        native, filtered, qc = preprocess_ppg_pair(adapted.ppg_red_ir, fs_hz=400.0, timestamps_s=adapted.timestamps_s)
        views = CanonicalSignalViews(x_native=native, x_filter=filtered, x_analysis_rate=filtered.copy(), imu_processed={}, metadata={'fs_hz': 400.0, 'record_id': row.record_id},
                                     source_valid_mask=qc.source_valid_mask, repair_mask=qc.repair_mask)
        views.validate()
        recording_duration_s = native.shape[0] / 400.0
        for algorithm_id, module_id, parameters in algorithm_specs:
            for channel in _CHANNELS:
                scored, elapsed = _score_segment(views, reference, algorithm_id=module_id, algorithm_parameters=parameters, fs_hz=400.0, validation=validation, wavelength=channel,
                                                 lag_window_s=lag_window_s, recording_duration_s=recording_duration_s)
                rows.append({
                    'participant_id': row.subject_id, 'record_id': row.record_id, 'activity': 'sit', 'activity_group': 'static', 'channel': channel,
                    'algorithm_or_reducer': algorithm_id, 'detector_module_id': module_id, 'evaluation_unit': 'subject_recording_and_wavelength',
                    'recording_duration_s': recording_duration_s, 'status': 'passed', 'runtime_s': elapsed, 'runtime_fraction_of_signal': elapsed / recording_duration_s,
                    'execution_time_percent': 100.0 * elapsed / recording_duration_s, **scored
                })
    if progress_callback is not None:
        progress_callback(len(records), len(records), 'completed static peak ablation')
    summary_rows = _aggregate_static_peak_benchmark(rows)
    statistical = dict(validation['statistical_comparison'])
    configured_metric_ids = statistical.get('metrics')
    if configured_metric_ids is None:
        configured_metric_ids = [statistical['metric']]
    comparisons = _static_peak_rank_sum_comparisons(rows, reference_algorithm_id=str(statistical['reference_algorithm_id']), alpha=float(statistical['alpha']),
                                                    metric_ids=[str(value) for value in configured_metric_ids],
                                                    registered_metric_ids=[str(value) for value in configured_metric_ids])
    return {
        'schema_version': 'ppg_frailty.stage_ablation_01_static_peak_result.v3', 'status': 'passed', 'participant_count': len({row.subject_id
                                                                                                                               for row in records}), 'record_count': len(records),
        'activities': ['sit'], 'rows': rows, 'summary_rows': summary_rows, 'statistical_comparisons': comparisons, 'paper_source': dict(plan.payload['paper_source']),
        'validation': validation, 'execution_environment': {
            'platform': platform.platform(), 'processor': platform.processor() or 'not_reported_by_platform', 'python_version': platform.python_version(),
            'timer': 'time.perf_counter_wall_time', 'parallelization': 'runner_sequential_no_explicit_parallelism', 'paper_hardware_claim_applied_to_this_run': False
        }
    }

def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError('motion report requires pyarrow') from exc
    return [dict(row) for row in pq.read_table(path).to_pylist()]

def _copy_bound_file(source: Path, target: Path, expected_sha256: str) -> str:
    source = source.resolve()
    if sha256_file(source) != expected_sha256:
        raise ValueError(f'comparison source SHA-256 mismatch: {source}')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != expected_sha256:
            raise ValueError(f'comparison target already exists with different bytes: {target}')
    else:
        temporary = target.with_suffix(target.suffix + '.tmp')
        shutil.copy2(source, temporary)
        temporary.replace(target)
    return sha256_file(target)

# Package frozen candidates without selecting or retraining a downstream frailty model.
def _write_motion_model_comparison_package(*, pipeline_root: Path, output_dir: Path, legacy_study_relative: str, ptt_training_evidence_path: Path) -> Path:
    legacy_study = (pipeline_root / legacy_study_relative).resolve()
    if not legacy_study.is_relative_to(pipeline_root):
        raise ValueError('legacy Stage5 comparison source escapes pipeline root')
    legacy_evidence_path = legacy_study / 'motion_internal/motion_internal_evidence.json'
    legacy_evidence = json.loads(legacy_evidence_path.read_text(encoding='utf-8'))
    ptt_evidence = json.loads(ptt_training_evidence_path.read_text(encoding='utf-8'))
    legacy_oof_path = Path(str(legacy_evidence.get('window_oof_parquet_path', '')))
    if not legacy_oof_path.is_file() or sha256_file(legacy_oof_path) != legacy_evidence.get('window_oof_parquet_sha256'):
        raise ValueError('legacy Frailty29 OOF evidence is missing or hash-mismatched')
    legacy_threshold, legacy_threshold_hash = _deployment_threshold_from_oof(_read_parquet(legacy_oof_path))
    candidates: list[dict[str, Any]] = []
    definitions = (('frailty29_trained_legacy_reference', 'frailty29', legacy_evidence_path, legacy_evidence, 'final_model', legacy_threshold, legacy_threshold_hash),
                   ('ptt22_trained_reverse_ablation', PTT_DATASET_ID, ptt_training_evidence_path, ptt_evidence, 'final_model', ptt_evidence.get('deployment_threshold'),
                    ptt_evidence.get('deployment_threshold_artifact_sha256')))
    for candidate_id, training_dataset, evidence_path, evidence, model_key, threshold, threshold_hash in definitions:
        model = evidence.get(model_key)
        if not isinstance(model, Mapping) or not isinstance(threshold, Mapping) or threshold.get('score_origin') != MOTION_DEPLOYMENT_THRESHOLD_SCORE_ORIGIN or (
                threshold.get('fit_scope') != MOTION_DEPLOYMENT_THRESHOLD_FIT_SCOPE):
            raise ValueError(f'comparison evidence incomplete for {candidate_id}')
        model_hash = str(model.get('artifact_sha256', ''))
        candidate_dir = output_dir / candidate_id
        copied_model = candidate_dir / 'formal_motion_model.pt'
        copied_hash = _copy_bound_file(Path(str(model.get('artifact_path', ''))), copied_model, model_hash)
        threshold_path = candidate_dir / 'deployment_threshold.json'
        _strict_json(threshold_path, dict(threshold))
        threshold_hash = str(threshold_hash)
        if stable_payload_sha256(dict(threshold)) != threshold_hash:
            raise ValueError(f'comparison threshold hash mismatch for {candidate_id}')
        source_evidence = evidence.get('formal_source_evidence', {})
        candidates.append({
            'candidate_id': candidate_id, 'training_dataset': training_dataset, 'model_path': copied_model.relative_to(output_dir).as_posix(), 'model_sha256': copied_hash,
            'threshold_path': threshold_path.relative_to(output_dir).as_posix(), 'threshold_sha256': threshold_hash,
            'threshold_derivation': 'strict_training_dataset_outer_oof_only', 'source_evidence_path': str(evidence_path.resolve()),
            'source_evidence_sha256': sha256_file(evidence_path),
            'preprocessing_ekf_config_sha256': source_evidence.get('ekf_config_sha256') if isinstance(source_evidence, Mapping) else None,
            'training_participant_count': len(model.get('training_participant_ids', ()))
        })
    manifest = {
        'schema_version': MOTION_MODEL_COMPARISON_SCHEMA, 'status': 'ready_for_downstream_single_factor_comparison', 'comparison_factor': 'motion_detector_training_dataset',
        'candidate_count': 2, 'candidates': candidates, 'candidate_bound_preprocessing_required': True, 'downstream_execution_performed_by_stage5_pre': False,
        'downstream_role': 'comparison_only_single_factor_motion_detector_training_dataset',
        'interpretation': 'Stage5-pre packages both frozen parameter sets and thresholds. The eventual selected frailty classifier must evaluate the two candidates as a paired single-factor comparison; this preparatory study does not choose or train that downstream classifier.'
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / 'motion_model_comparison_manifest.json'
    _strict_json(manifest_path, manifest)
    return manifest_path

def _new_study_dir(output_root: Path, study_id: str) -> Path:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = output_root / f'{timestamp}_{study_id}'
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = output_root / f'{base.name}_{suffix:02d}'
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate.resolve()

def _stage_directory(root: Path, manifest: Mapping[str, Any], stage_name: str, default_name: str, required_name: str) -> tuple[Path, bool]:
    stages = manifest.get('stages', {})
    stage = stages.get(stage_name, {}) if isinstance(stages, Mapping) else {}
    declared = stage.get('artifact_dir') if isinstance(stage, Mapping) else None
    if declared:
        candidate = (root / str(declared)).resolve()
        if candidate.is_relative_to(root) and (candidate / required_name).is_file():
            return (candidate, True)
    default = root / default_name
    if (default / required_name).is_file():
        return (default, True)
    attempt = 2
    while True:
        candidate = root / f'{default_name}_attempt_{attempt:03d}'
        if (candidate / required_name).is_file():
            return (candidate, True)
        if not candidate.exists() or not any(candidate.iterdir()):
            break
        attempt += 1
    return (_fresh_stage_attempt(root, default_name), False)

def _fresh_stage_attempt(root: Path, default_name: str) -> Path:
    default = root / default_name
    if not default.exists() or not any(default.iterdir()):
        return default
    attempt = 2
    while True:
        candidate = root / f'{default_name}_attempt_{attempt:03d}'
        if not candidate.exists() or not any(candidate.iterdir()):
            return candidate
        attempt += 1

def run_motion_peak_study(plan_path: str | Path, *, pipeline_root: str | Path, output_root: str | Path, resume: str | Path | None = None, progress_sink: ProgressSink | None = None,
                          device: str | None = None, include_denoiser: bool = True) -> Path:
    """Execute a Stage5-pre or static peak-ablation plan."""
    plan = load_motion_peak_plan(plan_path)
    if plan.schema_version != STAGE5_SCHEMA and (not include_denoiser):
        raise ValueError('--no-denoiser applies only to Stage5-pre')
    if device is not None:
        if plan.schema_version != STAGE5_SCHEMA:
            raise ValueError('--device applies only to the Stage5-pre training plan')
        requested = str(device)
        trainer_config = FormalMotionTrainerConfig(device=requested)
        trainer_config.validate()
        validate_formal_motion_cuda_device(trainer_config.device)
        payload = copy.deepcopy(dict(plan.payload))
        payload['motion_detector']['training_device'] = requested
        plan = StudyPlan(path=plan.path, payload=payload, schema_version=plan.schema_version, study_type=plan.study_type, study_id=plan.study_id)
    progress = progress_sink or NullProgressSink()
    progress_total = (6 if include_denoiser else 5) if plan.schema_version == STAGE5_SCHEMA else 1
    progress_current = 0
    progress(
        ProgressEvent(event='motion_peak_study_started', current=0, total=progress_total, detail_current=0, detail_total=1, detail_label='prepare study directory',
                      message=plan.study_id))
    pipeline = Path(pipeline_root).resolve()
    repository = pipeline.parents[1]
    root = Path(resume).resolve() if resume else _new_study_dir(Path(output_root).resolve(), plan.study_id)
    root.mkdir(parents=True, exist_ok=True)
    resolved_plan = root / 'resolved_plan.yaml'
    if resolved_plan.exists():
        try:
            existing = load_motion_peak_plan(resolved_plan)
        except ValueError as exc:
            contract = 'Stage5 CUDA contract' if plan.schema_version == STAGE5_SCHEMA else 'current study contract'
            raise ValueError(f'resume resolved plan is incompatible with the current {contract}: {exc}') from exc
        if existing.payload != plan.payload:
            raise ValueError('resume plan differs from the persisted resolved plan')
    else:
        resolved_plan.write_text(yaml.safe_dump(dict(plan.payload), sort_keys=False), encoding='utf-8')
    manifest_path = root / 'study_manifest.json'
    manifest: dict[str, Any] = {
        'schema_version': RESULT_SCHEMA, 'study_id': plan.study_id, 'study_type': plan.study_type, 'status': 'running',
        'scientific_scope': 'paired motion-detector training-dataset ablation: Frailty29 grouped OOF/final to PTT and PTT repeat-0 grouped OOF/final to Frailty29; optional PTT denoiser comparison; PTT is not claimed independent'
        if plan.schema_version == STAGE5_SCHEMA else
        'PTT sit-only subject-recording detector comparison: default MSPTDfast versus explicit aboy_project ablation; 300-s drifting lag and +/-150-ms one-to-one beat assessment; no motion segments and no denoiser selection',
        'plan_sha256': sha256_file(resolved_plan), 'training_device': plan.payload['motion_detector']['training_device'] if plan.schema_version == STAGE5_SCHEMA else None,
        'denoiser_enabled': bool(include_denoiser) if plan.schema_version == STAGE5_SCHEMA else None, 'stages': {}
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if plan.schema_version == STAGE5_SCHEMA and bool(old.get('denoiser_enabled', True)) != bool(include_denoiser):
            raise ValueError('resume denoiser execution option differs from the persisted study')
        manifest['stages'] = dict(old.get('stages', {}))
    _strict_json(manifest_path, manifest)
    try:
        if plan.schema_version == STAGE5_SCHEMA:
            internal_dir, internal_complete = _stage_directory(root, manifest, 'internal_motion_oof', 'motion_internal', 'motion_internal_evidence.json')
            if internal_complete and (not (internal_dir / 'motion_window_oof.parquet').is_file() or not any(internal_dir.rglob('motion_training_history.json'))):
                internal_dir = _fresh_stage_attempt(root, 'motion_internal')
                internal_complete = False
            if not internal_complete:
                subprogress = _stage_progress(progress, progress_current, progress_total, 'internal motion OOF')
                subprogress(0, 1, 'load source manifest')
                result = run_formal_internal_motion_reference(repository, output_dir=internal_dir, progress_callback=subprogress,
                                                              training_device=str(plan.payload['motion_detector']['training_device']))
                manifest['stages']['internal_motion_oof'] = {
                    'status': 'passed', 'evidence_sha256': result.evidence_sha256, 'artifact_dir': internal_dir.relative_to(root).as_posix()
                }
                _strict_json(manifest_path, manifest)
            else:
                manifest['stages']['internal_motion_oof'] = {
                    'status': 'passed', 'evidence_sha256': sha256_file(internal_dir / 'motion_internal_evidence.json'), 'artifact_dir': internal_dir.relative_to(root).as_posix()
                }
            progress_current += 1
            _stage_progress(progress, progress_current, progress_total, 'internal motion OOF')(1, 1, 'completed or resumed')
            evidence_path = internal_dir / 'motion_internal_evidence.json'
            evidence_sha = sha256_file(evidence_path)
            external_dir, external_complete = _stage_directory(root, manifest, 'ptt_motion_external', 'motion_external', 'motion_ptt_external_report.json')
            if external_complete and (not (external_dir / 'motion_ptt_window_predictions.parquet').is_file()):
                external_dir = _fresh_stage_attempt(root, 'motion_external')
                external_complete = False
            if external_complete:
                external_payload = json.loads((external_dir / 'motion_ptt_external_report.json').read_text(encoding='utf-8'))
                if external_payload.get('internal_evidence_sha256') != evidence_sha:
                    external_dir = _fresh_stage_attempt(root, 'motion_external')
                    external_complete = False
            if not external_complete:
                subprogress = _stage_progress(progress, progress_current, progress_total, 'PTT motion evaluation')
                subprogress(0, 1, 'load PTT source manifest')
                external = run_formal_ptt_motion_reference(repository, internal_evidence_path=evidence_path, expected_internal_evidence_sha256=evidence_sha,
                                                           output_dir=external_dir, unit_evidence_path=repository / PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH,
                                                           expected_unit_evidence_sha256=PTT_IMU_UNIT_EVIDENCE_SHA256, progress_callback=subprogress)
                manifest['stages']['ptt_motion_external'] = {'status': 'passed', 'report_sha256': external.report_sha256, 'artifact_dir': external_dir.relative_to(root).as_posix()}
                _strict_json(manifest_path, manifest)
            else:
                manifest['stages']['ptt_motion_external'] = {
                    'status': 'passed', 'report_sha256': sha256_file(external_dir / 'motion_ptt_external_report.json'), 'artifact_dir': external_dir.relative_to(root).as_posix()
                }
            progress_current += 1
            _stage_progress(progress, progress_current, progress_total, 'PTT motion evaluation')(1, 1, 'completed or resumed')
            ptt_training_dir, ptt_training_complete = _stage_directory(root, manifest, 'ptt_motion_training_ablation', 'motion_ptt_training', 'motion_ptt_training_evidence.json')
            if ptt_training_complete and (not (ptt_training_dir / 'motion_ptt_training_oof.parquet').is_file()
                                          or not (ptt_training_dir / 'final_all_ptt/formal_motion_model.pt').is_file() or
                                          (not any(ptt_training_dir.rglob('motion_training_history.json')))):
                ptt_training_dir = _fresh_stage_attempt(root, 'motion_ptt_training')
                ptt_training_complete = False
            if not ptt_training_complete:
                subprogress = _stage_progress(progress, progress_current, progress_total, 'PTT motion training ablation')
                subprogress(0, 1, 'load PTT source manifest')
                ptt_training = run_formal_ptt_motion_training_ablation(repository, output_dir=ptt_training_dir, unit_evidence_path=repository / PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH,
                                                                       expected_unit_evidence_sha256=PTT_IMU_UNIT_EVIDENCE_SHA256, progress_callback=subprogress,
                                                                       training_device=str(plan.payload['motion_detector']['training_device']))
                ptt_training_evidence_sha = ptt_training.evidence_sha256
            else:
                ptt_training_evidence_sha = sha256_file(ptt_training_dir / 'motion_ptt_training_evidence.json')
            manifest['stages']['ptt_motion_training_ablation'] = {
                'status': 'passed', 'evidence_sha256': ptt_training_evidence_sha, 'artifact_dir': ptt_training_dir.relative_to(root).as_posix()
            }
            _strict_json(manifest_path, manifest)
            progress_current += 1
            _stage_progress(progress, progress_current, progress_total, 'PTT motion training ablation')(1, 1, 'completed or resumed')
            ptt_training_evidence_path = ptt_training_dir / 'motion_ptt_training_evidence.json'
            reverse_dir, reverse_complete = _stage_directory(root, manifest, 'frailty29_reverse_evaluation', 'motion_internal_reverse',
                                                             'motion_internal_reverse_evaluation_report.json')
            if reverse_complete and (not (reverse_dir / 'motion_internal_reverse_predictions.parquet').is_file()):
                reverse_dir = _fresh_stage_attempt(root, 'motion_internal_reverse')
                reverse_complete = False
            if reverse_complete:
                reverse_payload = json.loads((reverse_dir / 'motion_internal_reverse_evaluation_report.json').read_text(encoding='utf-8'))
                if reverse_payload.get('ptt_training_evidence_sha256') != ptt_training_evidence_sha:
                    reverse_dir = _fresh_stage_attempt(root, 'motion_internal_reverse')
                    reverse_complete = False
            if not reverse_complete:
                subprogress = _stage_progress(progress, progress_current, progress_total, 'Frailty29 reverse evaluation')
                subprogress(0, 1, 'load Frailty29 source manifest')
                reverse_result = run_formal_internal_reverse_evaluation(repository, ptt_training_evidence_path=ptt_training_evidence_path,
                                                                        expected_ptt_training_evidence_sha256=ptt_training_evidence_sha, output_dir=reverse_dir,
                                                                        progress_callback=subprogress, runtime_device=str(plan.payload['motion_detector']['training_device']))
                reverse_report_sha = reverse_result.report_sha256
            else:
                reverse_report_sha = sha256_file(reverse_dir / 'motion_internal_reverse_evaluation_report.json')
            manifest['stages']['frailty29_reverse_evaluation'] = {'status': 'passed', 'report_sha256': reverse_report_sha, 'artifact_dir': reverse_dir.relative_to(root).as_posix()}
            _strict_json(manifest_path, manifest)
            progress_current += 1
            _stage_progress(progress, progress_current, progress_total, 'Frailty29 reverse evaluation')(1, 1, 'completed or resumed')
            comparison_dir, _ = _stage_directory(root, manifest, 'motion_model_comparison_package', 'motion_model_comparison', 'motion_model_comparison_manifest.json')
            comparison_progress = _stage_progress(progress, progress_current, progress_total, 'motion-model comparison package')
            comparison_progress(0, 1, 'verify and copy frozen parameter sets')
            comparison_manifest = _write_motion_model_comparison_package(pipeline_root=pipeline, output_dir=comparison_dir,
                                                                         legacy_study_relative=str(plan.payload['motion_model_comparison']['legacy_frailty29_stage5_study']),
                                                                         ptt_training_evidence_path=ptt_training_evidence_path)
            manifest['stages']['motion_model_comparison_package'] = {
                'status': 'passed', 'manifest_sha256': sha256_file(comparison_manifest), 'artifact_dir': comparison_dir.relative_to(root).as_posix()
            }
            _strict_json(manifest_path, manifest)
            progress_current += 1
            _stage_progress(progress, progress_current, progress_total, 'motion-model comparison package')(1, 1, 'completed or resumed')
            if include_denoiser:
                denoiser_dir, denoiser_complete = _stage_directory(root, manifest, 'ptt_denoiser_benchmark', 'denoiser', 'denoiser_benchmark.json')
                denoiser_path = denoiser_dir / 'denoiser_benchmark.json'
                if not denoiser_complete:
                    benchmark = dict(plan.payload['denoiser_benchmark'])
                    subprogress = _stage_progress(progress, progress_current, progress_total, 'PTT denoiser benchmark')
                    subprogress(0, 1, 'prepare benchmark')
                    result = run_ptt_denoiser_benchmark(repository, reducer_ids=[str(value) for value in benchmark['reducers']], segment_s=float(benchmark['segment_s']),
                                                        validation=dict(benchmark['validation']), activities=[str(value) for value in benchmark['activities']],
                                                        scoring_peak_detector=str(benchmark.get('scoring_peak_detector', CANONICAL_DETECTOR_ID)),
                                                        scoring_peak_detector_parameters=benchmark.get('scoring_peak_detector_parameters'), progress_callback=subprogress)
                    _strict_json(denoiser_path, result)
                manifest['stages']['ptt_denoiser_benchmark'] = {
                    'status': 'passed', 'result_sha256': sha256_file(denoiser_path), 'artifact_dir': denoiser_dir.relative_to(root).as_posix()
                }
                progress_current += 1
                _stage_progress(progress, progress_current, progress_total, 'PTT denoiser benchmark')(1, 1, 'completed or resumed')
            else:
                manifest['stages']['ptt_denoiser_benchmark'] = {'status': 'skipped_by_cli', 'reason': '--no-denoiser'}
                _strict_json(manifest_path, manifest)
        else:
            ablation_dir, ablation_complete = _stage_directory(root, manifest, 'static_peak_ablation', 'static_peak_ablation', 'static_peak_ablation.json')
            result_path = ablation_dir / 'static_peak_ablation.json'
            if not ablation_complete:
                subprogress = _stage_progress(progress, progress_current, progress_total, 'static peak ablation')
                subprogress(0, 1, 'prepare PTT static records')
                _strict_json(result_path, run_static_peak_ablation(repository, plan, progress_callback=subprogress))
            manifest['stages']['static_peak_ablation'] = {'status': 'passed', 'result_sha256': sha256_file(result_path), 'artifact_dir': ablation_dir.relative_to(root).as_posix()}
            progress_current += 1
            _stage_progress(progress, progress_current, progress_total, 'static peak ablation')(1, 1, 'completed or resumed')
        manifest['status'] = 'passed'
        _strict_json(manifest_path, manifest)
        progress(
            ProgressEvent(event='motion_peak_study_finished', current=progress_total, total=progress_total, detail_current=1, detail_total=1, detail_label='study complete',
                          message='passed'))
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['failure_reason'] = f'{type(exc).__name__}: {exc}'
        _strict_json(manifest_path, manifest)
        progress(
            ProgressEvent(event='motion_peak_study_failed', current=progress_current, total=progress_total, detail_current=0, detail_total=1,
                          detail_label=f'failed: {type(exc).__name__}', message='failed'))
        raise
    return root

def generate_motion_peak_report(study_dir: str | Path) -> dict[str, Any]:
    """Compatibility entry point; presentation lives in the reporting package."""
    from ..reporting.specialized import generate_motion_peak_report as generate
    return generate(study_dir)

__all__ = [
    'MSPTDFAST_V2_ID', 'StudyPlan', 'align_and_score_beats', 'generate_motion_peak_report', 'load_motion_peak_plan', 'run_motion_peak_study', 'run_ptt_denoiser_benchmark',
    'run_static_peak_ablation'
]
