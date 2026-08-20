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
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import yaml

from ..artifacts.router import get_reducer
from ..data.external_manifest import (
    M2_EXTERNAL_RELATIVE_PATH,
    PTT_DATASET_ID,
    ExternalRecord,
    adapt_ptt_synchronized_channels,
    load_m2_external_manifest,
)
from ..peaks.pairing import match_events
from ..peaks.aboy_project import IMPLEMENTATION_PATH as ABOY_IMPLEMENTATION_PATH
from ..peaks.aboy_project_v2 import (
    DETECTOR_ID as ABOY_V2_ID,
    IMPLEMENTATION_PATH as ABOY_V2_IMPLEMENTATION_PATH,
)
from ..peaks.msptdfast_v2 import (
    AUTHOR_SOURCE_SHA256 as MSPTDFAST_AUTHOR_SOURCE_SHA256,
    DETECTOR_ID as MSPTDFAST_V2_ID,
    IMPLEMENTATION_PATH as MSPTDFAST_IMPLEMENTATION_PATH,
    resolve_parameters as resolve_msptdfast_parameters,
)
from ..peaks.resolver import CANONICAL_DETECTOR_ID, detect_pulses
from ..provenance import sha256_file, stable_payload_sha256
from ..signal.motion_imu import (
    PTT_STATIC_CALIBRATION_ROLE,
    RollPitchEkfConfig,
    fit_motion_imu_calibration,
    preprocess_motion_imu_calibrated_ekf,
)
from ..signal.preprocess import preprocess_ppg_pair
from ..signal.views import CanonicalSignalViews
from ..study.progress import NullProgressSink, ProgressEvent, ProgressSink
from .motion_reference import (
    PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH,
    PTT_IMU_UNIT_EVIDENCE_SHA256,
    PTT_SOURCE_ROOT,
    load_ptt_imu_unit_evidence,
    run_formal_internal_motion_reference,
    run_formal_internal_reverse_evaluation,
    run_formal_ptt_motion_training_ablation,
    run_formal_ptt_motion_reference,
)
from .motion_adapters import (
    FormalMotionTrainerConfig,
    validate_formal_motion_cuda_device,
)
from .motion import (
    MOTION_DEPLOYMENT_THRESHOLD_FIT_SCOPE,
    MOTION_DEPLOYMENT_THRESHOLD_SCORE_ORIGIN,
)
from .motion_runner import (
    _deployment_threshold_from_oof,
    participant_macro_motion_metrics,
)


STAGE5_SCHEMA = "ppg_frailty.stage5_pre_motion_ptt.v1"
PEAK_ABLATION_SCHEMA = "ppg_frailty.stage_ablation_01_static_peaks.v2"
RESULT_SCHEMA = "ppg_frailty.motion_peak_study_result.v1"
MOTION_MODEL_COMPARISON_SCHEMA = "ppg_frailty.motion_model_comparison_package.v1"
_PTT_COLUMNS = (
    "peaks", "pleth_1", "pleth_2", "a_x", "a_y", "a_z", "g_x", "g_y", "g_z"
)
_CHANNELS = ("RED", "IR")


@dataclass(frozen=True)
class StudyPlan:
    path: Path
    payload: Mapping[str, Any]
    schema_version: str
    study_type: str
    study_id: str


def _stage_progress(
    sink: ProgressSink,
    overall_current: int,
    overall_total: int,
    stage_label: str,
) -> Callable[[int, int, str], None]:
    def emit(current: int, total: int, detail: str) -> None:
        sink(
            ProgressEvent(
                event="motion_peak_subtask",
                current=overall_current,
                total=overall_total,
                case_id=stage_label,
                detail_current=current,
                detail_total=total,
                detail_label=detail,
                message=f"running {stage_label}",
            )
        )

    return emit


def _strict_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def load_motion_peak_plan(path: str | Path) -> StudyPlan:
    """Load either plan with registered-source and scientific-scope checks."""

    source = Path(path).resolve()
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("motion/peak study plan must be a mapping")
    schema = str(data.get("schema_version", ""))
    study_type = str(data.get("study_type", ""))
    study_id = str(data.get("study_id", ""))
    if schema not in {STAGE5_SCHEMA, PEAK_ABLATION_SCHEMA} or not study_id:
        raise ValueError("unknown motion/peak study schema or empty study_id")
    expected_type = (
        "stage5_pre_motion_ptt" if schema == STAGE5_SCHEMA
        else "stage_ablation_01_static_peak_detectors"
    )
    if study_type != expected_type:
        raise ValueError("study_type disagrees with schema_version")
    dataset = data.get("ptt_dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("ptt_dataset must be a mapping")
    if (
        dataset.get("dataset_id") != PTT_DATASET_ID
        or dataset.get("root") != PTT_SOURCE_ROOT.as_posix()
        or int(dataset.get("participant_count", -1)) != 22
        or int(dataset.get("record_count", -1)) != 66
    ):
        raise ValueError("PTT plan must declare the registered 22-participant/66-record source")
    if schema == STAGE5_SCHEMA:
        motion_detector = data.get("motion_detector")
        if not isinstance(motion_detector, Mapping):
            raise ValueError("Stage5-pre requires motion_detector settings")
        training_device = motion_detector.get("training_device")
        if training_device is None:
            raise ValueError(
                "Stage5-pre motion_detector.training_device is required and must be CUDA"
            )
        trainer_config = FormalMotionTrainerConfig(device=str(training_device))
        trainer_config.validate()
        validate_formal_motion_cuda_device(trainer_config.device)
        reverse = motion_detector.get("reverse_ablation")
        if not isinstance(reverse, Mapping):
            raise ValueError("Stage5-pre requires motion_detector.reverse_ablation")
        if (
            reverse.get("enabled") is not True
            or reverse.get("training_dataset") != PTT_DATASET_ID
            or reverse.get("evaluation_dataset") != "frailty29"
            or reverse.get("split_registry")
            != "splits/ptt_formal_repeated_grouped_5x5_v2.csv"
            or reverse.get("repeat_indices") != [0]
            or reverse.get("folds") != [0, 1, 2, 3, 4]
            or reverse.get("split_seed") != 42
            or reverse.get("evaluation_fit_or_recalibration") is not False
        ):
            raise ValueError(
                "Stage5 reverse ablation must be PTT repeat-0 grouped five-fold "
                "training followed by frozen-model Frailty29 evaluation"
            )
        comparison = data.get("motion_model_comparison")
        if not isinstance(comparison, Mapping):
            raise ValueError("Stage5-pre requires motion_model_comparison")
        legacy_study = Path(str(comparison.get("legacy_frailty29_stage5_study", "")))
        if (
            legacy_study.is_absolute()
            or ".." in legacy_study.parts
            or comparison.get("candidates")
            != [
                "frailty29_trained_legacy_reference",
                "ptt22_trained_reverse_ablation",
            ]
            or comparison.get("downstream_role")
            != "comparison_only_single_factor_motion_detector_training_dataset"
            or comparison.get("candidate_bound_preprocessing_required") is not True
        ):
            raise ValueError("Stage5 motion-model comparison declaration drift")
        benchmark = data.get("denoiser_benchmark")
        if not isinstance(benchmark, Mapping):
            raise ValueError("Stage5-pre requires denoiser_benchmark")
        reducers = benchmark.get("reducers")
        if not isinstance(reducers, list) or not reducers:
            raise ValueError("Stage5-pre requires one or more reducer IDs")
        for reducer_id in reducers:
            get_reducer(str(reducer_id))
        activities = benchmark.get("activities")
        if activities != ["sit", "walk", "run"]:
            raise ValueError("Stage5-pre denoiser benchmark must separate sit/walk/run")
        _finite(benchmark.get("segment_s"), "denoiser_benchmark.segment_s", positive=True)
        validation = benchmark.get("validation")
        if not isinstance(validation, Mapping):
            raise ValueError("Stage5-pre denoiser benchmark requires validation settings")
        for key in ("max_lag_s", "lag_step_s", "beat_tolerance_s"):
            _finite(validation.get(key), f"denoiser_benchmark.validation.{key}", positive=True)
        if float(dataset.get("source_fs_hz", -1.0)) != 500.0 or float(
            dataset.get("pipeline_fs_hz", -1.0)
        ) != 400.0:
            raise ValueError("Stage5-pre uses the registered PTT 500-to-400-Hz adapter")
        report = data.get("report")
        if (
            not isinstance(report, Mapping)
            or report.get("required_detector_figures")
            != [
                "motion_internal_confusion_matrix",
                "motion_internal_subject_confusion_matrix",
                "motion_ptt_confusion_matrix",
                "motion_ptt_subject_confusion_matrix",
                "motion_ptt_training_oof_confusion_matrix",
                "motion_ptt_training_oof_subject_confusion_matrix",
                "motion_internal_reverse_confusion_matrix",
                "motion_internal_reverse_subject_confusion_matrix",
                "motion_training_learning_curves",
            ]
            or report.get("denoiser_figures_when_enabled")
            != [
                "denoiser_interval_rmse",
                "denoiser_beat_f1",
                "denoiser_runtime",
            ]
        ):
            raise ValueError("Stage5 report figure contract drift")
    else:
        if data.get("activities") != ["sit"]:
            raise ValueError("Stage-ablation-01 is a pure-static sit-only experiment")
        algorithms = data.get("algorithms")
        ids = {
            str(item.get("algorithm_id"))
            for item in algorithms or ()
            if isinstance(item, Mapping)
        }
        if ids != {CANONICAL_DETECTOR_ID, ABOY_V2_ID, MSPTDFAST_V2_ID}:
            raise ValueError(
                "static peak ablation must compare Aboy v1, authoritative "
                "Aboy v2, and MSPTDfast v2"
            )
        implementation_by_id = {
            str(item.get("algorithm_id")): str(item.get("implementation"))
            for item in algorithms or ()
            if isinstance(item, Mapping)
        }
        if implementation_by_id.get(CANONICAL_DETECTOR_ID) != ABOY_IMPLEMENTATION_PATH:
            raise ValueError(
                "Stage-ablation-01 must reference the registered Aboy module"
            )
        if implementation_by_id.get(ABOY_V2_ID) != ABOY_V2_IMPLEMENTATION_PATH:
            raise ValueError(
                "Stage-ablation-01 must reference the registered Aboy v2 module"
            )
        if data.get("detector_input") != (
            "repaired_native_ppg_each_registered_module_owns_preprocessing"
        ):
            raise ValueError(
                "Stage-ablation-01 detector_input must preserve module-owned preprocessing"
            )
        paper_source = data.get("paper_source")
        if (
            not isinstance(paper_source, Mapping)
            or paper_source.get("msptdfast_source_sha256")
            != MSPTDFAST_AUTHOR_SOURCE_SHA256
        ):
            raise ValueError("static peak ablation must bind the reviewed author source")
        comparator = next(
            item for item in algorithms
            if isinstance(item, Mapping) and item.get("algorithm_id") == MSPTDFAST_V2_ID
        )
        if comparator.get("implementation") != MSPTDFAST_IMPLEMENTATION_PATH:
            raise ValueError(
                "Stage-ablation-01 must reference the registered MSPTDfast module"
            )
        resolve_msptdfast_parameters(comparator.get("parameters"))
        validation = data.get("validation")
        if not isinstance(validation, Mapping):
            raise ValueError("static peak ablation requires validation settings")
        _finite(validation.get("beat_tolerance_s"), "beat_tolerance_s", positive=True)
        _finite(validation.get("max_lag_s"), "max_lag_s", positive=True)
        _finite(validation.get("lag_step_s"), "lag_step_s", positive=True)
        _finite(validation.get("lag_window_s"), "lag_window_s", positive=True)
    return StudyPlan(source, dict(data), schema, study_type, study_id)


def _window_starts(length: int, window: int, hop: int) -> tuple[int, ...]:
    if length <= 0 or window <= 0 or hop <= 0:
        raise ValueError("window inputs must be positive")
    if length <= window:
        return (0,)
    starts = list(range(0, length - window + 1, hop))
    right = length - window
    if starts[-1] != right:
        starts.append(right)
    return tuple(starts)


def _matched_pairs(
    reference_s: np.ndarray,
    predicted_s: np.ndarray,
    *,
    tolerance_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.sort(np.asarray(reference_s, dtype=np.float64))
    predicted = np.sort(np.asarray(predicted_s, dtype=np.float64))
    used = np.zeros(predicted.size, dtype=bool)
    pairs: list[tuple[int, int]] = []
    for reference_index, event in enumerate(reference):
        candidates = np.flatnonzero(
            (~used) & (np.abs(predicted - event) <= tolerance_s)
        )
        if candidates.size:
            predicted_index = int(
                candidates[np.argmin(np.abs(predicted[candidates] - event))]
            )
            used[predicted_index] = True
            pairs.append((reference_index, predicted_index))
    if not pairs:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return tuple(np.asarray(pairs, dtype=np.int64).T)  # type: ignore[return-value]


def align_and_score_beats(
    reference_s: np.ndarray,
    predicted_s: np.ndarray,
    *,
    max_lag_s: float = 10.0,
    lag_step_s: float = 0.02,
    tolerance_s: float = 0.2,
) -> dict[str, Any]:
    """Paper-style lag search plus delay-invariant matched IBI/PPI errors."""

    reference = np.sort(np.asarray(reference_s, dtype=np.float64))
    predicted = np.sort(np.asarray(predicted_s, dtype=np.float64))
    lags = np.arange(-max_lag_s, max_lag_s + lag_step_s * 0.5, lag_step_s)
    candidates: list[tuple[int, float, float]] = []
    for lag in lags:
        metrics = match_events(reference + lag, predicted, tolerance_s=tolerance_s)
        candidates.append((metrics.true_positive, -abs(float(lag)), float(lag)))
    lag = max(candidates)[2] if candidates else 0.0
    shifted = reference + lag
    metrics = match_events(shifted, predicted, tolerance_s=tolerance_s)
    reference_indices, predicted_indices = _matched_pairs(
        shifted, predicted, tolerance_s=tolerance_s
    )
    interval_errors: list[float] = []
    for index in range(1, reference_indices.size):
        if (
            reference_indices[index] == reference_indices[index - 1] + 1
            and predicted_indices[index] == predicted_indices[index - 1] + 1
        ):
            ibi = reference[reference_indices[index]] - reference[reference_indices[index - 1]]
            ppi = predicted[predicted_indices[index]] - predicted[predicted_indices[index - 1]]
            interval_errors.append(float(ppi - ibi))
    errors = np.asarray(interval_errors, dtype=np.float64)
    return {
        "lag_s": lag,
        "true_positives": metrics.true_positive,
        "false_positives": metrics.false_positive,
        "false_negatives": metrics.false_negative,
        "sensitivity": metrics.recall,
        "positive_predictive_value": metrics.precision,
        "f1": metrics.f1,
        "timing_mae_s": metrics.timing_mae_s,
        "matched_interval_count": int(errors.size),
        "ibi_ppi_rmse_ms": (
            float(np.sqrt(np.mean(np.square(errors))) * 1000.0) if errors.size else None
        ),
        "ibi_ppi_mae_ms": (
            float(np.mean(np.abs(errors)) * 1000.0) if errors.size else None
        ),
    }


def _ptt_records(repository_root: Path) -> tuple[ExternalRecord, ...]:
    records = tuple(
        row
        for row in load_m2_external_manifest(repository_root / M2_EXTERNAL_RELATIVE_PATH)
        if row.dataset_id == PTT_DATASET_ID
    )
    if len(records) != 66 or len({row.subject_id for row in records}) != 22:
        raise ValueError("PTT source roster is not 66 records / 22 participants")
    return records


def _load_record(
    repository_root: Path, row: ExternalRecord
) -> tuple[Any, np.ndarray]:
    path = (repository_root / PTT_SOURCE_ROOT / row.canonical_representation).resolve()
    if sha256_file(path) != row.checksum_sha256:
        raise ValueError(f"PTT source hash mismatch: {row.record_id}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = tuple(next(csv.reader(handle)))
    indices = tuple(header.index(name) for name in _PTT_COLUMNS)
    values = np.loadtxt(path, delimiter=",", skiprows=1, usecols=indices, ndmin=2)
    if values.shape[1] != len(_PTT_COLUMNS) or not np.isfinite(values).all():
        raise ValueError(f"PTT numeric source invalid: {row.record_id}")
    columns = {name: values[:, index] for index, name in enumerate(_PTT_COLUMNS)}
    adapted = adapt_ptt_synchronized_channels(
        {
            "pleth_1": columns["pleth_1"], "pleth_2": columns["pleth_2"],
            "AX": columns["a_x"], "AY": columns["a_y"], "AZ": columns["a_z"],
            "GX": columns["g_x"], "GY": columns["g_y"], "GZ": columns["g_z"],
        },
        external_record=row,
        observed_source_file_sha256=row.checksum_sha256,
        additional_channel_order=("AX", "AY", "AZ", "GX", "GY", "GZ"),
    )
    reference_times = np.flatnonzero(columns["peaks"] > 0.5).astype(np.float64) / 500.0
    return adapted, reference_times


def _slice_processed(processed: Mapping[str, Any], start: int, stop: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, value in processed.items():
        array = np.asarray(value)
        if array.ndim >= 1 and array.shape[0] >= stop:
            output[name] = array[start:stop]
    return output


def _motion_processed(motion: Any) -> dict[str, np.ndarray]:
    """Expose the existing nine-channel motion result through reducer field names."""

    motion.validate()
    values = np.asarray(motion.values, dtype=np.float64)
    return {
        "dynamic_acc_mps2": values[:, 0:3],
        "gyro_rads": values[:, 3:6],
        "dynamic_magnitude": values[:, 6],
        "gyro_magnitude": values[:, 7],
        "jerk_magnitude": values[:, 8],
        "imu_valid_mask": np.asarray(motion.valid_mask, dtype=bool),
    }


def _score_segment(
    values: np.ndarray | CanonicalSignalViews,
    reference_s: np.ndarray,
    *,
    algorithm_id: str,
    algorithm_parameters: Mapping[str, Any] | None = None,
    fs_hz: float,
    validation: Mapping[str, Any],
    wavelength: str = "RED",
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    pulse = detect_pulses(
        values,
        detector_id=algorithm_id,
        detector_parameters=algorithm_parameters,
        fs_hz=fs_hz,
        wavelength=wavelength,
        min_observation_sec=6.0,
        min_peaks=3,
    )
    peaks_s = np.asarray(pulse.peak_timestamps_s, dtype=np.float64)
    elapsed = time.perf_counter() - started
    scored = align_and_score_beats(
        reference_s,
        peaks_s,
        max_lag_s=float(validation["max_lag_s"]),
        lag_step_s=float(validation["lag_step_s"]),
        tolerance_s=float(validation["beat_tolerance_s"]),
    )
    return scored, elapsed


def _aggregate_benchmark(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted(
        {
            (str(row["algorithm_or_reducer"]), str(row["activity_group"]), str(row["channel"]))
            for row in rows
        }
    )
    output: list[dict[str, Any]] = []
    for algorithm, activity_group, channel in keys:
        selected = [
            row for row in rows
            if (row["algorithm_or_reducer"], row["activity_group"], row["channel"])
            == (algorithm, activity_group, channel)
            and row.get("status") == "passed"
        ]
        participant_rows: list[dict[str, float]] = []
        for participant in sorted({str(row["participant_id"]) for row in selected}):
            current = [row for row in selected if str(row["participant_id"]) == participant]
            if not current:
                continue
            available_rmse = [
                float(row["ibi_ppi_rmse_ms"])
                for row in current
                if row.get("ibi_ppi_rmse_ms") is not None
            ]
            participant_rows.append(
                {
                    "f1": float(np.mean([float(row["f1"]) for row in current])),
                    "rmse": (
                        float(np.mean(available_rmse))
                        if available_rmse
                        else float("nan")
                    ),
                    "runtime": float(np.sum([float(row["runtime_s"]) for row in current])),
                }
            )
        participant_f1 = [row["f1"] for row in participant_rows]
        participant_rmse = [row["rmse"] for row in participant_rows]
        output.append(
            {
                "algorithm_or_reducer": algorithm,
                "activity_group": activity_group,
                "channel": channel,
                "participant_count": len(participant_rows),
                "segment_count": len(selected),
                "participant_macro_f1": (
                    float(np.mean(participant_f1)) if participant_f1 else None
                ),
                "participant_macro_ibi_ppi_rmse_ms": (
                    float(np.nanmean(participant_rmse))
                    if participant_rmse and np.any(np.isfinite(participant_rmse))
                    else None
                ),
                "total_runtime_s": float(np.sum([row["runtime"] for row in participant_rows])),
            }
        )
    return output


def run_ptt_denoiser_benchmark(
    repository_root: str | Path,
    *,
    reducer_ids: Sequence[str],
    segment_s: float,
    validation: Mapping[str, Any],
    activities: Sequence[str] = ("sit", "walk", "run"),
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Run implemented reducers on all declared PTT records, sequentially."""

    repository = Path(repository_root).resolve()
    records = _ptt_records(repository)
    unit_evidence = load_ptt_imu_unit_evidence(
        repository / PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH,
        expected_sha256=PTT_IMU_UNIT_EVIDENCE_SHA256,
        expected_records=records,
    )
    by_subject: dict[str, dict[str, ExternalRecord]] = {}
    for row in records:
        by_subject.setdefault(row.subject_id, {})[row.activity_raw.lower()] = row
    fs_hz = 400.0
    segment_samples = int(round(float(segment_s) * fs_hz))
    rows: list[dict[str, Any]] = []
    subjects = sorted(by_subject.items())
    for subject_index, (subject_id, activity_records) in enumerate(subjects):
        if progress_callback is not None:
            progress_callback(
                subject_index,
                len(subjects),
                f"benchmark PTT participant {subject_id}",
            )
        loaded = {
            activity: _load_record(repository, activity_records[activity])
            for activity in ("sit", "walk", "run")
        }
        sit = loaded["sit"][0]
        calibration = fit_motion_imu_calibration(
            sit.values[:, 2:5], sit.values[:, 5:8],
            participant_id=subject_id,
            file_id=activity_records["sit"].record_id,
            source_role=PTT_STATIC_CALIBRATION_ROLE,
            fs_hz=fs_hz,
            acceleration_unit=unit_evidence.acceleration_unit,
            gyroscope_unit=unit_evidence.gyroscope_unit,
            config=RollPitchEkfConfig(),
        )
        for activity in activities:
            adapted, reference_times = loaded[str(activity)]
            motion = preprocess_motion_imu_calibrated_ekf(
                adapted.values[:, 2:5], adapted.values[:, 5:8],
                fs_hz=fs_hz,
                acceleration_unit=unit_evidence.acceleration_unit,
                gyroscope_unit=unit_evidence.gyroscope_unit,
                participant_id=subject_id,
                calibration=calibration,
                config=RollPitchEkfConfig(),
            )
            _, filtered, _ = preprocess_ppg_pair(
                adapted.ppg_red_ir, fs_hz=fs_hz, timestamps_s=adapted.timestamps_s
            )
            for start in _window_starts(filtered.shape[0], segment_samples, segment_samples):
                stop = min(start + segment_samples, filtered.shape[0])
                if stop - start < int(round(8.0 * fs_hz)):
                    continue
                reference = reference_times[
                    (reference_times >= start / fs_hz) & (reference_times < stop / fs_hz)
                ] - start / fs_hz
                if reference.size < 3:
                    continue
                ppg = filtered[start:stop]
                imu = _slice_processed(_motion_processed(motion), start, stop)
                for reducer_id in reducer_ids:
                    started = time.perf_counter()
                    result = get_reducer(str(reducer_id)).reduce(ppg, imu, fs_hz=fs_hz)
                    reduction_s = time.perf_counter() - started
                    if result.status != "success" or result.x_ar is None:
                        for channel in _CHANNELS:
                            rows.append({
                                "participant_id": subject_id,
                                "record_id": activity_records[str(activity)].record_id,
                                "activity": activity,
                                "activity_group": "static" if activity == "sit" else "dynamic",
                                "channel": channel,
                                "algorithm_or_reducer": str(reducer_id),
                                "segment_start_s": start / fs_hz,
                                "status": result.status,
                                "failure_reasons": list(result.reasons),
                                "runtime_s": reduction_s,
                            })
                        continue
                    for channel_index, channel in enumerate(_CHANNELS):
                        scored, detector_s = _score_segment(
                            np.asarray(result.x_ar)[:, channel_index], reference,
                            algorithm_id=CANONICAL_DETECTOR_ID,
                            fs_hz=fs_hz,
                            validation=validation,
                        )
                        rows.append({
                            "participant_id": subject_id,
                            "record_id": activity_records[str(activity)].record_id,
                            "activity": activity,
                            "activity_group": "static" if activity == "sit" else "dynamic",
                            "channel": channel,
                            "algorithm_or_reducer": str(reducer_id),
                            "segment_start_s": start / fs_hz,
                            "status": "passed",
                            "runtime_s": reduction_s + detector_s,
                            **scored,
                        })
    if progress_callback is not None:
        progress_callback(len(subjects), len(subjects), "completed PTT benchmark")
    return {
        "schema_version": "ppg_frailty.stage5_pre_denoiser_benchmark.v1",
        "status": "passed",
        "participant_count": len(by_subject),
        "record_count": len(records),
        "activities": list(activities),
        "segment_s": float(segment_s),
        "reducers": list(reducer_ids),
        "validation": dict(validation),
        "rows": rows,
        "summary_rows": _aggregate_benchmark(rows),
    }


def run_static_peak_ablation(
    repository_root: str | Path,
    plan: StudyPlan,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Run the three registered detectors on complete PTT sit records only."""

    if plan.schema_version != PEAK_ABLATION_SCHEMA:
        raise ValueError("static peak runner received a non-ablation plan")
    repository = Path(repository_root).resolve()
    records = [row for row in _ptt_records(repository) if row.activity_raw.lower() == "sit"]
    validation = dict(plan.payload["validation"])
    algorithm_parameters = {
        str(item["algorithm_id"]): dict(item.get("parameters", {}))
        for item in plan.payload["algorithms"]
    }
    rows: list[dict[str, Any]] = []
    lag_window_s = float(validation["lag_window_s"])
    window_samples = int(round(lag_window_s * 400.0))
    for record_index, row in enumerate(records):
        if progress_callback is not None:
            progress_callback(
                record_index,
                len(records),
                f"score static PTT participant {row.subject_id}",
            )
        adapted, reference = _load_record(repository, row)
        native, filtered, qc = preprocess_ppg_pair(
            adapted.ppg_red_ir, fs_hz=400.0, timestamps_s=adapted.timestamps_s
        )
        views = CanonicalSignalViews(
            x_native=native,
            x_filter=filtered,
            x_analysis_rate=filtered.copy(),
            imu_processed={},
            metadata={"fs_hz": 400.0, "record_id": row.record_id},
            source_valid_mask=qc.source_valid_mask,
            repair_mask=qc.repair_mask,
        )
        views.validate()
        for start in _window_starts(native.shape[0], window_samples, window_samples):
            stop = min(start + window_samples, native.shape[0])
            if stop - start < int(round(8.0 * 400.0)):
                continue
            current_reference = reference[
                (reference >= start / 400.0) & (reference < stop / 400.0)
            ] - start / 400.0
            if current_reference.size < 3:
                continue
            window_views = CanonicalSignalViews(
                x_native=views.x_native[start:stop],
                x_filter=views.x_filter[start:stop],
                x_analysis_rate=views.x_analysis_rate[start:stop],
                imu_processed={},
                metadata=dict(views.metadata),
                source_valid_mask=views.source_valid_mask[start:stop],
                repair_mask=views.repair_mask[start:stop],
            )
            window_views.validate()
            for algorithm_id, parameters in algorithm_parameters.items():
                for channel in _CHANNELS:
                    scored, elapsed = _score_segment(
                        window_views,
                        current_reference,
                        algorithm_id=algorithm_id,
                        algorithm_parameters=parameters,
                        fs_hz=400.0,
                        validation=validation,
                        wavelength=channel,
                    )
                    rows.append({
                        "participant_id": row.subject_id,
                        "record_id": row.record_id,
                        "activity": "sit",
                        "activity_group": "static",
                        "channel": channel,
                        "algorithm_or_reducer": algorithm_id,
                        "lag_window_start_s": start / 400.0,
                        "status": "passed",
                        "runtime_s": elapsed,
                        "runtime_fraction_of_signal": elapsed / ((stop - start) / 400.0),
                        **scored,
                    })
    if progress_callback is not None:
        progress_callback(len(records), len(records), "completed static peak ablation")
    return {
        "schema_version": "ppg_frailty.stage_ablation_01_static_peak_result.v2",
        "status": "passed",
        "participant_count": len({row.subject_id for row in records}),
        "record_count": len(records),
        "activities": ["sit"],
        "rows": rows,
        "summary_rows": _aggregate_benchmark(rows),
        "paper_source": dict(plan.payload["paper_source"]),
        "validation": validation,
    }


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("motion report requires pyarrow") from exc
    return [dict(row) for row in pq.read_table(path).to_pylist()]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "N/A"
    fields = sorted({str(key) for row in rows for key in row})
    body = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    body.extend(
        "| " + " | ".join(str(row.get(field, "")) for field in fields) + " |"
        for row in rows
    )
    return "\n".join(body)


def _html_table(rows: Sequence[Mapping[str, Any]]) -> str:
    import html

    if not rows:
        return "<p>N/A</p>"
    fields = sorted({str(key) for row in rows for key in row})
    heading = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body = "".join(
        "<tr>" + "".join(
            f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields
        ) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table>"


def _detector_report_rows(
    datasets: Sequence[
        tuple[str, Sequence[Mapping[str, Any]], float]
    ],
) -> list[dict[str, Any]]:
    """Recompute all requested detector metrics from persisted predictions."""

    output: list[dict[str, Any]] = []
    for dataset, rows, worst_fold in datasets:
        metrics = participant_macro_motion_metrics(rows)
        output.append(
            {
                "dataset": dataset,
                "participant_macro_balanced_accuracy": metrics["balanced_accuracy"],
                "participant_macro_f1": metrics["macro_f1"],
                "participant_macro_sensitivity": metrics["sensitivity"],
                "participant_macro_specificity": metrics["specificity"],
                "participant_macro_roc_auc": metrics["roc_auc"],
                "participant_macro_pr_auc": metrics["pr_auc"],
                "worst_fold_balanced_accuracy": float(worst_fold),
            }
        )
    return output


def _rank_and_mark_denoiser_rows(
    rows: Sequence[Mapping[str, Any]], activity_group: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return numeric rows plus report-only best-cell decorations.

    Subject-macro PPI--RR RMSE is the primary ascending sort key; F1 is only
    the descending tie-breaker.  Machine-readable CSV/JSON remains numeric;
    only the Markdown/HTML display rows receive ``*`` markers.
    """

    numeric = [
        dict(row) for row in rows if str(row.get("activity_group")) == activity_group
    ]
    numeric.sort(
        key=lambda row: (
            (
                float(row["participant_macro_ibi_ppi_rmse_ms"])
                if row.get("participant_macro_ibi_ppi_rmse_ms") is not None
                and math.isfinite(
                    float(row["participant_macro_ibi_ppi_rmse_ms"])
                )
                else math.inf
            ),
            -float(row["participant_macro_f1"]),
            str(row.get("algorithm_or_reducer", "")),
            str(row.get("channel", "")),
        )
    )
    if not numeric:
        return [], []
    best_f1 = max(float(row["participant_macro_f1"]) for row in numeric)
    finite_rmse = [
        float(row["participant_macro_ibi_ppi_rmse_ms"])
        for row in numeric
        if row.get("participant_macro_ibi_ppi_rmse_ms") is not None
        and math.isfinite(float(row["participant_macro_ibi_ppi_rmse_ms"]))
    ]
    best_rmse = min(finite_rmse) if finite_rmse else None
    display: list[dict[str, Any]] = []
    for row in numeric:
        marked = dict(row)
        f1_best = math.isclose(
            float(row["participant_macro_f1"]), best_f1, rel_tol=0.0, abs_tol=1e-15
        )
        rmse_best = best_rmse is not None and math.isclose(
            float(row["participant_macro_ibi_ppi_rmse_ms"]),
            best_rmse,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        if f1_best:
            marked["participant_macro_f1"] = f"{row['participant_macro_f1']}*"
        if rmse_best:
            marked["participant_macro_ibi_ppi_rmse_ms"] = (
                f"{row['participant_macro_ibi_ppi_rmse_ms']}*"
            )
        if f1_best or rmse_best:
            marker = "**" if f1_best and rmse_best else "*"
            marked["activity_group"] = f"{row['activity_group']}{marker}"
        display.append(marked)
    return numeric, display


def _subject_activity_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse windows to one score per participant and activity class.

    Every Stage5 dataset contains both static and motion observations for each
    participant, so a scientifically meaningful subject-level confusion matrix
    uses participant-by-activity-class units rather than assigning one label to
    an entire participant. The unit score is the median window probability.
    """

    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        participant_id = str(row.get("participant_id", ""))
        label = int(row.get("activity_label", -1))
        if not participant_id or label not in (0, 1):
            raise ValueError(
                "subject-level motion confusion requires valid participant and label"
            )
        grouped.setdefault((participant_id, label), []).append(row)
    if not grouped:
        raise ValueError("subject-level motion confusion requires prediction rows")

    output: list[dict[str, Any]] = []
    for (participant_id, label), selected in sorted(grouped.items()):
        probabilities = np.asarray(
            [float(row["p_active"]) for row in selected], dtype=np.float64
        )
        thresholds = {float(row["threshold"]) for row in selected}
        if (
            not np.all(np.isfinite(probabilities))
            or len(thresholds) != 1
            or not math.isfinite(next(iter(thresholds)))
        ):
            raise ValueError(
                "subject-level motion confusion requires finite scores and one "
                "frozen threshold per participant"
            )
        probability = float(np.median(probabilities))
        threshold = thresholds.pop()
        output.append(
            {
                "participant_id": participant_id,
                "activity_label": label,
                "window_count": len(selected),
                "median_p_active": probability,
                "threshold": threshold,
                "predicted_activity": int(probability >= threshold),
            }
        )
    return output


def _subject_confusion_report_row(
    dataset: str, rows: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], np.ndarray]:
    subject_rows = _subject_activity_prediction_rows(rows)
    matrix = _confusion(subject_rows)
    return (
        {
            "dataset": dataset,
            "aggregation_level": "participant_by_activity_class_median_probability",
            "participant_count": len(
                {str(row["participant_id"]) for row in subject_rows}
            ),
            "participant_activity_class_count": len(subject_rows),
            "true_static_predicted_static": int(matrix[0, 0]),
            "true_static_predicted_motion": int(matrix[0, 1]),
            "true_motion_predicted_static": int(matrix[1, 0]),
            "true_motion_predicted_motion": int(matrix[1, 1]),
        },
        matrix,
    )


def _confusion(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    matrix = np.zeros((2, 2), dtype=np.int64)
    for row in rows:
        matrix[int(row["activity_label"]), int(row["predicted_activity"])] += 1
    return matrix


def _plot_confusion(path: Path, matrix: np.ndarray, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.4, 4.6))
    image = axis.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axis.text(j, i, str(int(matrix[i, j])), ha="center", va="center")
    axis.set_xticks((0, 1), ("static", "motion"))
    axis.set_yticks((0, 1), ("static", "motion"))
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Reference")
    axis.set_title(title)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_summary(path: Path, rows: Sequence[Mapping[str, Any]], metric: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row.get(metric) is not None]
    labels = [
        f"{row['algorithm_or_reducer']}\n{row['activity_group']} · {row['channel']}"
        for row in selected
    ]
    values = [float(row[metric]) for row in selected]
    width = max(7.0, 0.55 * len(values))
    figure, axis = plt.subplots(figsize=(width, 4.8))
    x = np.arange(len(values))
    axis.bar(x, values)
    axis.set_xticks(x, labels, rotation=35, ha="right")
    axis.set_title(title)
    axis.set_ylabel(metric)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_motion_learning_curves(path: Path, history_paths: Sequence[Path]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), sharex=True)
    for history_path in sorted(history_paths):
        payload = json.loads(history_path.read_text(encoding="utf-8"))
        rows = payload.get("rows", [])
        if not rows:
            continue
        ptt_training = "motion_ptt_training" in history_path.parts
        if payload.get("final_fit"):
            label = "final-all22-PTT" if ptt_training else "final-all29"
        else:
            prefix = "PTT-fold" if ptt_training else "Frailty29-fold"
            label = f"{prefix}-{int(payload['fold_index']) + 1}"
        epochs = [int(row["epoch"]) for row in rows]
        axes[0].plot(epochs, [float(row["training_loss"]) for row in rows], label=label)
        axes[1].plot(
            epochs,
            [float(row["training_balanced_accuracy"]) for row in rows],
            label=label,
        )
    axes[0].set_title("Outer-train loss")
    axes[1].set_title("Outer-train balanced accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Balanced accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(loc="best", fontsize=8)
    figure.suptitle("Motion detector learning curves (diagnostic only; no held-out data)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _copy_bound_file(source: Path, target: Path, expected_sha256: str) -> str:
    source = source.resolve()
    if sha256_file(source) != expected_sha256:
        raise ValueError(f"comparison source SHA-256 mismatch: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != expected_sha256:
            raise ValueError(f"comparison target already exists with different bytes: {target}")
    else:
        temporary = target.with_suffix(target.suffix + ".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(target)
    return sha256_file(target)


def _write_motion_model_comparison_package(
    *,
    pipeline_root: Path,
    output_dir: Path,
    legacy_study_relative: str,
    ptt_training_evidence_path: Path,
) -> Path:
    """Copy the two small final model assets into one hash-bound handoff."""

    legacy_study = (pipeline_root / legacy_study_relative).resolve()
    if not legacy_study.is_relative_to(pipeline_root):
        raise ValueError("legacy Stage5 comparison source escapes pipeline root")
    legacy_evidence_path = legacy_study / "motion_internal/motion_internal_evidence.json"
    legacy_evidence = json.loads(legacy_evidence_path.read_text(encoding="utf-8"))
    ptt_evidence = json.loads(ptt_training_evidence_path.read_text(encoding="utf-8"))
    legacy_oof_path = Path(str(legacy_evidence.get("window_oof_parquet_path", "")))
    if (
        not legacy_oof_path.is_file()
        or sha256_file(legacy_oof_path)
        != legacy_evidence.get("window_oof_parquet_sha256")
    ):
        raise ValueError("legacy Frailty29 OOF evidence is missing or hash-mismatched")
    legacy_threshold, legacy_threshold_hash = _deployment_threshold_from_oof(
        _read_parquet(legacy_oof_path)
    )
    candidates: list[dict[str, Any]] = []
    definitions = (
        (
            "frailty29_trained_legacy_reference",
            "frailty29",
            legacy_evidence_path,
            legacy_evidence,
            "final_model",
            legacy_threshold,
            legacy_threshold_hash,
        ),
        (
            "ptt22_trained_reverse_ablation",
            PTT_DATASET_ID,
            ptt_training_evidence_path,
            ptt_evidence,
            "final_model",
            ptt_evidence.get("deployment_threshold"),
            ptt_evidence.get("deployment_threshold_artifact_sha256"),
        ),
    )
    for (
        candidate_id,
        training_dataset,
        evidence_path,
        evidence,
        model_key,
        threshold,
        threshold_hash,
    ) in definitions:
        model = evidence.get(model_key)
        if (
            not isinstance(model, Mapping)
            or not isinstance(threshold, Mapping)
            or threshold.get("score_origin")
            != MOTION_DEPLOYMENT_THRESHOLD_SCORE_ORIGIN
            or threshold.get("fit_scope") != MOTION_DEPLOYMENT_THRESHOLD_FIT_SCOPE
        ):
            raise ValueError(f"comparison evidence incomplete for {candidate_id}")
        model_hash = str(model.get("artifact_sha256", ""))
        candidate_dir = output_dir / candidate_id
        copied_model = candidate_dir / "formal_motion_model.pt"
        copied_hash = _copy_bound_file(
            Path(str(model.get("artifact_path", ""))), copied_model, model_hash
        )
        threshold_path = candidate_dir / "deployment_threshold.json"
        _strict_json(threshold_path, dict(threshold))
        threshold_hash = str(threshold_hash)
        if stable_payload_sha256(dict(threshold)) != threshold_hash:
            raise ValueError(f"comparison threshold hash mismatch for {candidate_id}")
        source_evidence = evidence.get("formal_source_evidence", {})
        candidates.append(
            {
                "candidate_id": candidate_id,
                "training_dataset": training_dataset,
                "model_path": copied_model.relative_to(output_dir).as_posix(),
                "model_sha256": copied_hash,
                "threshold_path": threshold_path.relative_to(output_dir).as_posix(),
                "threshold_sha256": threshold_hash,
                "threshold_derivation": "strict_training_dataset_outer_oof_only",
                "source_evidence_path": str(evidence_path.resolve()),
                "source_evidence_sha256": sha256_file(evidence_path),
                "preprocessing_ekf_config_sha256": (
                    source_evidence.get("ekf_config_sha256")
                    if isinstance(source_evidence, Mapping)
                    else None
                ),
                "training_participant_count": len(
                    model.get("training_participant_ids", ())
                ),
            }
        )
    manifest = {
        "schema_version": MOTION_MODEL_COMPARISON_SCHEMA,
        "status": "ready_for_downstream_single_factor_comparison",
        "comparison_factor": "motion_detector_training_dataset",
        "candidate_count": 2,
        "candidates": candidates,
        "candidate_bound_preprocessing_required": True,
        "downstream_execution_performed_by_stage5_pre": False,
        "downstream_role": "comparison_only_single_factor_motion_detector_training_dataset",
        "interpretation": (
            "Stage5-pre packages both frozen parameter sets and thresholds. "
            "The eventual selected frailty classifier must evaluate the two candidates "
            "as a paired single-factor comparison; this preparatory study does not choose "
            "or train that downstream classifier."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "motion_model_comparison_manifest.json"
    _strict_json(manifest_path, manifest)
    return manifest_path


def _result_backup(study_dir: Path, files: Iterable[Path]) -> Path:
    backup = study_dir / "result_backup"
    backup.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for source in sorted({path.resolve() for path in files if path.is_file()}):
        relative = source.relative_to(study_dir.resolve())
        target = backup / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.resolve() == source:
            continue
        shutil.copy2(source, target)
        entries.append({
            "source": relative.as_posix(),
            "backup": target.relative_to(study_dir).as_posix(),
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
        })
    _strict_json(backup / "backup_manifest.json", {
        "schema_version": "ppg_frailty.small_result_backup.v1",
        "policy": (
            "reports_tables_plans_and_hash_manifests_only_"
            "no_source_data_or_model_duplication"
        ),
        "entries": entries,
    })
    return backup


def generate_motion_peak_report(study_dir: str | Path) -> dict[str, Any]:
    """Generate Markdown/HTML, numerical tables, plots, index and backup."""

    root = Path(study_dir).resolve()
    manifest_path = root / "study_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tables = root / "tables"
    figures = root / "figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    images: list[Path] = []
    summary_rows: list[Mapping[str, Any]] = []
    headline_metric_rows: list[Mapping[str, Any]] = []
    detector_internal_rows: list[Mapping[str, Any]] = []
    detector_transfer_rows: list[Mapping[str, Any]] = []
    detector_subject_confusion_rows: list[Mapping[str, Any]] = []
    denoiser_display_tables: dict[str, list[dict[str, Any]]] = {}
    comparison_rows: list[dict[str, Any]] = []
    comparison_payload: Mapping[str, Any] | None = None
    denoiser_enabled = False
    if manifest["study_type"] == "stage5_pre_motion_ptt":
        internal_dir = _report_stage_dir(root, manifest, "internal_motion_oof", "motion_internal")
        external_dir = _report_stage_dir(root, manifest, "ptt_motion_external", "motion_external")
        internal = json.loads(
            (internal_dir / "motion_internal_evidence.json").read_text(encoding="utf-8")
        )
        external = json.loads(
            (external_dir / "motion_ptt_external_report.json").read_text(
                encoding="utf-8"
            )
        )
        internal_rows = _read_parquet(internal_dir / "motion_window_oof.parquet")
        external_rows = _read_parquet(external_dir / "motion_ptt_window_predictions.parquet")
        detector_datasets: list[
            tuple[str, Sequence[Mapping[str, Any]], float]
        ] = [
            (
                "frailty29_outer_oof",
                internal_rows,
                float(internal["major_metrics"]["worst_fold_balanced_accuracy"]),
            ),
            (
                "frailty29_trained_to_ptt22",
                external_rows,
                float(external["major_metrics"]["worst_fold_balanced_accuracy"]),
            ),
        ]
        stages = manifest.get("stages", {})
        reverse_training_stage = (
            stages.get("ptt_motion_training_ablation", {})
            if isinstance(stages, Mapping)
            else {}
        )
        reverse_evaluation_stage = (
            stages.get("frailty29_reverse_evaluation", {})
            if isinstance(stages, Mapping)
            else {}
        )
        reverse_available = (
            isinstance(reverse_training_stage, Mapping)
            and reverse_training_stage.get("status") == "passed"
            and isinstance(reverse_evaluation_stage, Mapping)
            and reverse_evaluation_stage.get("status") == "passed"
        )
        ptt_training: Mapping[str, Any] | None = None
        internal_reverse: Mapping[str, Any] | None = None
        reverse_training_rows: list[dict[str, Any]] = []
        internal_reverse_rows: list[dict[str, Any]] = []
        ptt_training_dir: Path | None = None
        if reverse_available:
            ptt_training_dir = _report_stage_dir(
                root, manifest, "ptt_motion_training_ablation", "motion_ptt_training"
            )
            internal_reverse_dir = _report_stage_dir(
                root, manifest, "frailty29_reverse_evaluation", "motion_internal_reverse"
            )
            ptt_training = json.loads(
                (ptt_training_dir / "motion_ptt_training_evidence.json").read_text(
                    encoding="utf-8"
                )
            )
            internal_reverse = json.loads(
                (
                    internal_reverse_dir
                    / "motion_internal_reverse_evaluation_report.json"
                ).read_text(encoding="utf-8")
            )
            reverse_training_rows = _read_parquet(
                ptt_training_dir / "motion_ptt_training_oof.parquet"
            )
            internal_reverse_rows = _read_parquet(
                internal_reverse_dir / "motion_internal_reverse_predictions.parquet"
            )
            detector_datasets.extend(
                [
                    (
                        "ptt22_outer_oof",
                        reverse_training_rows,
                        float(
                            ptt_training["major_metrics"][
                                "worst_fold_balanced_accuracy"
                            ]
                        ),
                    ),
                    (
                        "ptt22_trained_to_frailty29",
                        internal_reverse_rows,
                        float(
                            internal_reverse["major_metrics"][
                                "worst_fold_balanced_accuracy"
                            ]
                        ),
                    ),
                ]
            )
        headline_metric_rows = _detector_report_rows(detector_datasets)
        _write_csv(tables / "motion_detector_metrics.csv", headline_metric_rows)
        _strict_json(tables / "motion_detector_metrics.json", headline_metric_rows)
        detector_internal_rows = [
            row for row in headline_metric_rows
            if str(row["dataset"]).endswith("outer_oof")
        ]
        detector_transfer_rows = [
            row for row in headline_metric_rows
            if not str(row["dataset"]).endswith("outer_oof")
        ]
        _write_csv(
            tables / "motion_detector_internal_evaluation.csv",
            detector_internal_rows,
        )
        _strict_json(
            tables / "motion_detector_internal_evaluation.json",
            detector_internal_rows,
        )
        _write_csv(
            tables / "motion_detector_cross_dataset_evaluation.csv",
            detector_transfer_rows,
        )
        _strict_json(
            tables / "motion_detector_cross_dataset_evaluation.json",
            detector_transfer_rows,
        )
        for dataset, name, subject_name, rows, title in (
            (
                "frailty29_outer_oof",
                "motion_internal_confusion_matrix.png",
                "motion_internal_subject_confusion_matrix.png",
                internal_rows,
                "Internal 29-person OOF",
            ),
            (
                "frailty29_trained_to_ptt22",
                "motion_ptt_confusion_matrix.png",
                "motion_ptt_subject_confusion_matrix.png",
                external_rows,
                "PTT external evaluation",
            ),
            *(
                (
                    (
                        "ptt22_outer_oof",
                        "motion_ptt_training_oof_confusion_matrix.png",
                        "motion_ptt_training_oof_subject_confusion_matrix.png",
                        reverse_training_rows,
                        "PTT 22-person training OOF",
                    ),
                    (
                        "ptt22_trained_to_frailty29",
                        "motion_internal_reverse_confusion_matrix.png",
                        "motion_internal_reverse_subject_confusion_matrix.png",
                        internal_reverse_rows,
                        "PTT-trained frozen model on Frailty29",
                    ),
                )
                if reverse_available
                else ()
            ),
        ):
            path = figures / name
            _plot_confusion(path, _confusion(rows), title)
            images.append(path)
            subject_row, subject_matrix = _subject_confusion_report_row(dataset, rows)
            detector_subject_confusion_rows.append(subject_row)
            subject_path = figures / subject_name
            _plot_confusion(
                subject_path,
                subject_matrix,
                f"{title} · subject×activity-class",
            )
            images.append(subject_path)
        _write_csv(
            tables / "motion_detector_subject_confusion.csv",
            detector_subject_confusion_rows,
        )
        _strict_json(
            tables / "motion_detector_subject_confusion.json",
            detector_subject_confusion_rows,
        )
        learning_path = figures / "motion_training_learning_curves.png"
        history_paths = tuple(internal_dir.rglob("motion_training_history.json"))
        if ptt_training_dir is not None:
            history_paths += tuple(
                ptt_training_dir.rglob("motion_training_history.json")
            )
        if not history_paths:
            raise ValueError("completed Stage5-pre report requires motion training histories")
        _plot_motion_learning_curves(learning_path, history_paths)
        images.append(learning_path)
        denoiser_stage = manifest.get("stages", {}).get("ptt_denoiser_benchmark", {})
        denoiser_enabled = (
            isinstance(denoiser_stage, Mapping)
            and denoiser_stage.get("status") == "passed"
        )
        if denoiser_enabled:
            denoiser_dir = _report_stage_dir(
                root, manifest, "ptt_denoiser_benchmark", "denoiser"
            )
            benchmark = json.loads(
                (denoiser_dir / "denoiser_benchmark.json").read_text(encoding="utf-8")
            )
            summary_rows = benchmark["summary_rows"]
            _write_csv(tables / "denoiser_summary.csv", summary_rows)
            _strict_json(tables / "denoiser_summary.json", summary_rows)
            for activity_group in ("static", "dynamic"):
                numeric, display = _rank_and_mark_denoiser_rows(
                    summary_rows, activity_group
                )
                denoiser_display_tables[activity_group] = display
                _write_csv(tables / f"denoiser_{activity_group}.csv", numeric)
                _strict_json(tables / f"denoiser_{activity_group}.json", numeric)
            for metric, name, title in (
                (
                    "participant_macro_ibi_ppi_rmse_ms",
                    "denoiser_interval_rmse.png",
                    "IBI–PPI RMSE (lower is better)",
                ),
                ("participant_macro_f1", "denoiser_beat_f1.png", "Delay-aligned beat F1"),
                ("total_runtime_s", "denoiser_runtime.png", "Reducer + detector runtime"),
            ):
                path = figures / name
                _plot_summary(path, summary_rows, metric, title)
                images.append(path)
        comparison_stage = manifest.get("stages", {}).get(
            "motion_model_comparison_package", {}
        )
        if (
            isinstance(comparison_stage, Mapping)
            and comparison_stage.get("status") == "passed"
        ):
            comparison_dir = _report_stage_dir(
                root,
                manifest,
                "motion_model_comparison_package",
                "motion_model_comparison",
            )
            comparison_payload = json.loads(
                (
                    comparison_dir / "motion_model_comparison_manifest.json"
                ).read_text(encoding="utf-8")
            )
            comparison_rows = [
                dict(row) for row in comparison_payload.get("candidates", ())
            ]
            _write_csv(tables / "motion_model_comparison_candidates.csv", comparison_rows)
            _strict_json(
                tables / "motion_model_comparison_candidates.json", comparison_rows
            )
        headline = {
            "internal_major_metrics": internal["major_metrics"],
            "ptt_major_metrics": external["major_metrics"],
            "ptt_training_oof_major_metrics": (
                ptt_training["major_metrics"] if ptt_training is not None else None
            ),
            "frailty29_reverse_major_metrics": (
                internal_reverse["major_metrics"]
                if internal_reverse is not None
                else None
            ),
            "detector_report_rows": headline_metric_rows,
            "detector_subject_confusion_rows": detector_subject_confusion_rows,
            "reverse_ablation_status": "passed" if reverse_available else "not_available",
            "motion_model_comparison": comparison_payload,
            "denoiser_status": "passed" if denoiser_enabled else "skipped_by_execution_option",
            "denoiser_summary_rows": summary_rows,
        }
    else:
        ablation_dir = _report_stage_dir(root, manifest, "static_peak_ablation", ".")
        result = json.loads(
            (ablation_dir / "static_peak_ablation.json").read_text(encoding="utf-8")
        )
        summary_rows = result["summary_rows"]
        _write_csv(tables / "static_peak_detector_summary.csv", summary_rows)
        _strict_json(tables / "static_peak_detector_summary.json", summary_rows)
        for metric, name, title in (
            ("participant_macro_f1", "static_peak_detector_f1.png", "Static PTT beat F1"),
            (
                "participant_macro_ibi_ppi_rmse_ms",
                "static_peak_detector_interval_rmse.png",
                "Static IBI–PPI RMSE",
            ),
            ("total_runtime_s", "static_peak_detector_runtime.png", "Static detector runtime"),
        ):
            path = figures / name
            _plot_summary(path, summary_rows, metric, title)
            images.append(path)
        headline = {"static_peak_detector_summary_rows": summary_rows}
    _strict_json(root / "study_summary.json", {
        "schema_version": RESULT_SCHEMA,
        "study_id": manifest["study_id"],
        "status": manifest["status"],
        **headline,
    })
    numerical_sections = (
        [
            "### Detector internal grouped-OOF evaluation",
            "",
            _markdown_table(detector_internal_rows),
            "",
            "### Detector frozen cross-dataset evaluation",
            "",
            _markdown_table(detector_transfer_rows),
            "",
            "### Detector subject-level confusion counts",
            "",
            _markdown_table(detector_subject_confusion_rows),
            "",
            "Participant-macro metrics first score every participant from all of "
            "that participant's windows, then average participants equally. The "
            "subject-level confusion matrix uses one median probability for each "
            "participant×activity-class unit because every participant has both "
            "static and motion observations.",
            "",
        ]
        if manifest["study_type"] == "stage5_pre_motion_ptt"
        else [_markdown_table(headline_metric_rows), ""]
    )
    if manifest["study_type"] == "stage5_pre_motion_ptt":
        if comparison_rows:
            numerical_sections.extend(
                [
                    "### Frozen motion-model comparison candidates",
                    "",
                    _markdown_table(comparison_rows),
                    "",
                    "These two parameter sets are packaged for a later paired "
                    "single-factor comparison in the selected final frailty "
                    "classifier; Stage5-pre does not train that downstream classifier.",
                    "",
                ]
            )
        if denoiser_enabled:
            numerical_sections.extend(
                [
                    "### Static denoiser results", "",
                    _markdown_table(denoiser_display_tables["static"]), "",
                    "### Dynamic denoiser results", "",
                    _markdown_table(denoiser_display_tables["dynamic"]), "",
                    "Rows are sorted primarily by subject-macro PPI–RR RMSE "
                    "ascending; beat F1 descending is only the tie-breaker. The "
                    "stored `participant_macro_ibi_ppi_rmse_ms` field compares "
                    "ECG RR/IBI intervals with PPG PPI intervals.", "",
                    "Denoiser F1 is beat-event matching F1 after constant-lag "
                    "alignment to ECG annotations, not motion-classification F1. "
                    "It guards against a deceptively low interval RMSE obtained "
                    "from only a small, easy subset of matched beats.", "",
                    "`*` marks the best value in a metric column (F1 maximum; "
                    "RMSE minimum). `**` in the first column marks a row that "
                    "is best for both metrics.", "",
                ]
            )
        else:
            numerical_sections.extend(
                ["### Denoiser benchmark", "", "Skipped by execution option.", ""]
            )
    else:
        numerical_sections.extend([_markdown_table(summary_rows), ""])
    lines = [
        f"# {manifest['study_id']}", "", f"Status: **{manifest['status']}**", "",
        "## Scientific scope", "",
        str(manifest["scientific_scope"]), "",
        "## Figures", "",
        *[f"![{path.stem}](figures/{path.name})" for path in images], "",
        "## Numerical outputs", "",
        *numerical_sections,
        "Machine-readable values are in `study_summary.json` and `tables/`.", "",
    ]
    markdown = "\n".join(lines)
    (root / "STUDY_SUMMARY.md").write_text(markdown, encoding="utf-8")
    html_images = "\n".join(
        f'<figure><img src="figures/{path.name}" alt="{path.stem}">'
        f"<figcaption>{path.stem}</figcaption></figure>"
        for path in images
    )
    html_numerical = (
        "<h3>Detector internal grouped-OOF evaluation</h3>"
        + _html_table(detector_internal_rows)
        + "<h3>Detector frozen cross-dataset evaluation</h3>"
        + _html_table(detector_transfer_rows)
        + "<h3>Detector subject-level confusion counts</h3>"
        + _html_table(detector_subject_confusion_rows)
        + "<p>Participant-macro metrics score each participant first and then "
        "average participants equally. Subject-level confusion uses one median "
        "probability for each participant-by-activity-class unit because every "
        "participant has both static and motion observations.</p>"
        if manifest["study_type"] == "stage5_pre_motion_ptt"
        else _html_table(headline_metric_rows)
    )
    if manifest["study_type"] == "stage5_pre_motion_ptt":
        if comparison_rows:
            html_numerical += (
                "<h3>Frozen motion-model comparison candidates</h3>"
                + _html_table(comparison_rows)
                + "<p>Packaged for a later paired single-factor comparison in the "
                "selected final frailty classifier; Stage5-pre does not train that "
                "downstream classifier.</p>"
            )
        if denoiser_enabled:
            html_numerical += (
                "<h3>Static denoiser results</h3>"
                + _html_table(denoiser_display_tables["static"])
                + "<h3>Dynamic denoiser results</h3>"
                + _html_table(denoiser_display_tables["dynamic"])
                + "<p>Rows are sorted primarily by subject-macro PPI–RR RMSE "
                "ascending; beat F1 descending is only the tie-breaker. The stored "
                "participant_macro_ibi_ppi_rmse_ms field compares ECG RR/IBI with "
                "PPG PPI intervals.</p>"
                + "<p>Denoiser F1 is beat-event matching F1 after constant-lag "
                "alignment to ECG annotations, not motion-classification F1. It "
                "guards against a deceptively low interval RMSE obtained from only "
                "a small, easy subset of matched beats.</p>"
                + "<p>* marks the best metric value (F1 maximum; RMSE minimum); "
                "** marks a row best on both.</p>"
            )
        else:
            html_numerical += "<h3>Denoiser benchmark</h3><p>Skipped by execution option.</p>"
    else:
        html_numerical += _html_table(summary_rows)
    (root / "STUDY_SUMMARY.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>"
        + manifest["study_id"] + "</title><h1>" + manifest["study_id"]
        + "</h1><p>Status: " + manifest["status"] + "</p>"
        + "<h2>Numerical outputs</h2>" + html_numerical
        + "<h2>Figures</h2>" + html_images,
        encoding="utf-8",
    )
    indexed = [
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and "result_backup" not in path.parts
            and path.name != "outputs_index.json"
        )
    ]
    _strict_json(root / "outputs_index.json", {
        "schema_version": "ppg_frailty.motion_peak_outputs_index.v1",
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(indexed)
        ],
    })
    backup = _result_backup(
        root,
        [
            root / "resolved_plan.yaml", root / "study_manifest.json",
            root / "study_summary.json", root / "STUDY_SUMMARY.md",
            root / "STUDY_SUMMARY.html", root / "outputs_index.json",
            *tables.rglob("*"), *figures.rglob("*"),
            *root.glob("motion_model_comparison*/**/*.json"),
        ],
    )
    return {"study_dir": str(root), "figure_count": len(images), "backup_dir": str(backup)}


def _new_study_dir(output_root: Path, study_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_root / f"{timestamp}_{study_id}"
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = output_root / f"{base.name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate.resolve()


def _stage_directory(
    root: Path,
    manifest: Mapping[str, Any],
    stage_name: str,
    default_name: str,
    required_name: str,
) -> tuple[Path, bool]:
    """Resolve one completed stage or allocate a non-destructive new attempt."""

    stages = manifest.get("stages", {})
    stage = stages.get(stage_name, {}) if isinstance(stages, Mapping) else {}
    declared = stage.get("artifact_dir") if isinstance(stage, Mapping) else None
    if declared:
        candidate = (root / str(declared)).resolve()
        if candidate.is_relative_to(root) and (candidate / required_name).is_file():
            return candidate, True
    default = root / default_name
    if (default / required_name).is_file():
        return default, True
    attempt = 2
    while True:
        candidate = root / f"{default_name}_attempt_{attempt:03d}"
        if (candidate / required_name).is_file():
            return candidate, True
        if not candidate.exists() or not any(candidate.iterdir()):
            break
        attempt += 1
    return _fresh_stage_attempt(root, default_name), False


def _fresh_stage_attempt(root: Path, default_name: str) -> Path:
    default = root / default_name
    if not default.exists() or not any(default.iterdir()):
        return default
    attempt = 2
    while True:
        candidate = root / f"{default_name}_attempt_{attempt:03d}"
        if not candidate.exists() or not any(candidate.iterdir()):
            return candidate
        attempt += 1


def _report_stage_dir(
    root: Path,
    manifest: Mapping[str, Any],
    stage_name: str,
    fallback: str,
) -> Path:
    stages = manifest.get("stages", {})
    stage = stages.get(stage_name, {}) if isinstance(stages, Mapping) else {}
    relative = stage.get("artifact_dir", fallback) if isinstance(stage, Mapping) else fallback
    path = (root / str(relative)).resolve()
    if not path.is_relative_to(root):
        raise ValueError("study manifest stage artifact directory escapes study root")
    return path


def run_motion_peak_study(
    plan_path: str | Path,
    *,
    pipeline_root: str | Path,
    output_root: str | Path,
    resume: str | Path | None = None,
    progress_sink: ProgressSink | None = None,
    device: str | None = None,
    include_denoiser: bool = True,
) -> Path:
    """Execute a Stage5-pre or static peak-ablation plan."""

    plan = load_motion_peak_plan(plan_path)
    if plan.schema_version != STAGE5_SCHEMA and not include_denoiser:
        raise ValueError("--no-denoiser applies only to Stage5-pre")
    if device is not None:
        if plan.schema_version != STAGE5_SCHEMA:
            raise ValueError("--device applies only to the Stage5-pre training plan")
        requested = str(device)
        trainer_config = FormalMotionTrainerConfig(device=requested)
        trainer_config.validate()
        validate_formal_motion_cuda_device(trainer_config.device)
        payload = copy.deepcopy(dict(plan.payload))
        payload["motion_detector"]["training_device"] = requested
        plan = StudyPlan(
            path=plan.path,
            payload=payload,
            schema_version=plan.schema_version,
            study_type=plan.study_type,
            study_id=plan.study_id,
        )
    progress = progress_sink or NullProgressSink()
    progress_total = (
        (7 if include_denoiser else 6)
        if plan.schema_version == STAGE5_SCHEMA
        else 2
    )
    progress_current = 0
    progress(ProgressEvent(
        event="motion_peak_study_started",
        current=0,
        total=progress_total,
        detail_current=0,
        detail_total=1,
        detail_label="prepare study directory",
        message=plan.study_id,
    ))
    pipeline = Path(pipeline_root).resolve()
    repository = pipeline.parents[1]
    root = (
        Path(resume).resolve()
        if resume
        else _new_study_dir(Path(output_root).resolve(), plan.study_id)
    )
    root.mkdir(parents=True, exist_ok=True)
    resolved_plan = root / "resolved_plan.yaml"
    if resolved_plan.exists():
        try:
            existing = load_motion_peak_plan(resolved_plan)
        except ValueError as exc:
            contract = (
                "Stage5 CUDA contract"
                if plan.schema_version == STAGE5_SCHEMA
                else "current study contract"
            )
            raise ValueError(
                f"resume resolved plan is incompatible with the current {contract}: {exc}"
            ) from exc
        if existing.payload != plan.payload:
            raise ValueError("resume plan differs from the persisted resolved plan")
    else:
        resolved_plan.write_text(
            yaml.safe_dump(dict(plan.payload), sort_keys=False),
            encoding="utf-8",
        )
    manifest_path = root / "study_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "study_id": plan.study_id,
        "study_type": plan.study_type,
        "status": "running",
        "scientific_scope": (
            "paired motion-detector training-dataset ablation: Frailty29 grouped "
            "OOF/final to PTT and PTT repeat-0 grouped OOF/final to Frailty29; "
            "optional PTT denoiser comparison; PTT is not claimed independent"
            if plan.schema_version == STAGE5_SCHEMA
            else "PTT sit-only detector comparison; no motion segments and no denoiser selection"
        ),
        "plan_sha256": sha256_file(resolved_plan),
        "training_device": (
            plan.payload["motion_detector"]["training_device"]
            if plan.schema_version == STAGE5_SCHEMA
            else None
        ),
        "denoiser_enabled": (
            bool(include_denoiser) if plan.schema_version == STAGE5_SCHEMA else None
        ),
        "stages": {},
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if plan.schema_version == STAGE5_SCHEMA and bool(
            old.get("denoiser_enabled", True)
        ) != bool(include_denoiser):
            raise ValueError(
                "resume denoiser execution option differs from the persisted study"
            )
        manifest["stages"] = dict(old.get("stages", {}))
    _strict_json(manifest_path, manifest)
    try:
        if plan.schema_version == STAGE5_SCHEMA:
            internal_dir, internal_complete = _stage_directory(
                root, manifest, "internal_motion_oof", "motion_internal",
                "motion_internal_evidence.json",
            )
            if internal_complete and (
                not (internal_dir / "motion_window_oof.parquet").is_file()
                or not any(internal_dir.rglob("motion_training_history.json"))
            ):
                internal_dir = _fresh_stage_attempt(root, "motion_internal")
                internal_complete = False
            if not internal_complete:
                subprogress = _stage_progress(
                    progress, progress_current, progress_total, "internal motion OOF"
                )
                subprogress(0, 1, "load source manifest")
                result = run_formal_internal_motion_reference(
                    repository,
                    output_dir=internal_dir,
                    progress_callback=subprogress,
                    training_device=str(
                        plan.payload["motion_detector"]["training_device"]
                    ),
                )
                manifest["stages"]["internal_motion_oof"] = {
                    "status": "passed", "evidence_sha256": result.evidence_sha256,
                    "artifact_dir": internal_dir.relative_to(root).as_posix(),
                }
                _strict_json(manifest_path, manifest)
            else:
                manifest["stages"]["internal_motion_oof"] = {
                    "status": "passed",
                    "evidence_sha256": sha256_file(internal_dir / "motion_internal_evidence.json"),
                    "artifact_dir": internal_dir.relative_to(root).as_posix(),
                }
            progress_current += 1
            _stage_progress(
                progress, progress_current, progress_total, "internal motion OOF"
            )(1, 1, "completed or resumed")
            evidence_path = internal_dir / "motion_internal_evidence.json"
            evidence_sha = sha256_file(evidence_path)
            external_dir, external_complete = _stage_directory(
                root, manifest, "ptt_motion_external", "motion_external",
                "motion_ptt_external_report.json",
            )
            if external_complete and not (
                external_dir / "motion_ptt_window_predictions.parquet"
            ).is_file():
                external_dir = _fresh_stage_attempt(root, "motion_external")
                external_complete = False
            if external_complete:
                external_payload = json.loads(
                    (external_dir / "motion_ptt_external_report.json").read_text(
                        encoding="utf-8"
                    )
                )
                if external_payload.get("internal_evidence_sha256") != evidence_sha:
                    external_dir = _fresh_stage_attempt(root, "motion_external")
                    external_complete = False
            if not external_complete:
                subprogress = _stage_progress(
                    progress, progress_current, progress_total, "PTT motion evaluation"
                )
                subprogress(0, 1, "load PTT source manifest")
                external = run_formal_ptt_motion_reference(
                    repository,
                    internal_evidence_path=evidence_path,
                    expected_internal_evidence_sha256=evidence_sha,
                    output_dir=external_dir,
                    unit_evidence_path=repository / PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH,
                    expected_unit_evidence_sha256=PTT_IMU_UNIT_EVIDENCE_SHA256,
                    progress_callback=subprogress,
                )
                manifest["stages"]["ptt_motion_external"] = {
                    "status": "passed", "report_sha256": external.report_sha256,
                    "artifact_dir": external_dir.relative_to(root).as_posix(),
                }
                _strict_json(manifest_path, manifest)
            else:
                manifest["stages"]["ptt_motion_external"] = {
                    "status": "passed",
                    "report_sha256": sha256_file(external_dir / "motion_ptt_external_report.json"),
                    "artifact_dir": external_dir.relative_to(root).as_posix(),
                }
            progress_current += 1
            _stage_progress(
                progress, progress_current, progress_total, "PTT motion evaluation"
            )(1, 1, "completed or resumed")
            ptt_training_dir, ptt_training_complete = _stage_directory(
                root,
                manifest,
                "ptt_motion_training_ablation",
                "motion_ptt_training",
                "motion_ptt_training_evidence.json",
            )
            if ptt_training_complete and (
                not (ptt_training_dir / "motion_ptt_training_oof.parquet").is_file()
                or not (
                    ptt_training_dir
                    / "final_all_ptt/formal_motion_model.pt"
                ).is_file()
                or not any(ptt_training_dir.rglob("motion_training_history.json"))
            ):
                ptt_training_dir = _fresh_stage_attempt(root, "motion_ptt_training")
                ptt_training_complete = False
            if not ptt_training_complete:
                subprogress = _stage_progress(
                    progress,
                    progress_current,
                    progress_total,
                    "PTT motion training ablation",
                )
                subprogress(0, 1, "load PTT source manifest")
                ptt_training = run_formal_ptt_motion_training_ablation(
                    repository,
                    output_dir=ptt_training_dir,
                    unit_evidence_path=repository / PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH,
                    expected_unit_evidence_sha256=PTT_IMU_UNIT_EVIDENCE_SHA256,
                    progress_callback=subprogress,
                    training_device=str(
                        plan.payload["motion_detector"]["training_device"]
                    ),
                )
                ptt_training_evidence_sha = ptt_training.evidence_sha256
            else:
                ptt_training_evidence_sha = sha256_file(
                    ptt_training_dir / "motion_ptt_training_evidence.json"
                )
            manifest["stages"]["ptt_motion_training_ablation"] = {
                "status": "passed",
                "evidence_sha256": ptt_training_evidence_sha,
                "artifact_dir": ptt_training_dir.relative_to(root).as_posix(),
            }
            _strict_json(manifest_path, manifest)
            progress_current += 1
            _stage_progress(
                progress,
                progress_current,
                progress_total,
                "PTT motion training ablation",
            )(1, 1, "completed or resumed")

            ptt_training_evidence_path = (
                ptt_training_dir / "motion_ptt_training_evidence.json"
            )
            reverse_dir, reverse_complete = _stage_directory(
                root,
                manifest,
                "frailty29_reverse_evaluation",
                "motion_internal_reverse",
                "motion_internal_reverse_evaluation_report.json",
            )
            if reverse_complete and not (
                reverse_dir / "motion_internal_reverse_predictions.parquet"
            ).is_file():
                reverse_dir = _fresh_stage_attempt(root, "motion_internal_reverse")
                reverse_complete = False
            if reverse_complete:
                reverse_payload = json.loads(
                    (
                        reverse_dir / "motion_internal_reverse_evaluation_report.json"
                    ).read_text(encoding="utf-8")
                )
                if (
                    reverse_payload.get("ptt_training_evidence_sha256")
                    != ptt_training_evidence_sha
                ):
                    reverse_dir = _fresh_stage_attempt(root, "motion_internal_reverse")
                    reverse_complete = False
            if not reverse_complete:
                subprogress = _stage_progress(
                    progress,
                    progress_current,
                    progress_total,
                    "Frailty29 reverse evaluation",
                )
                subprogress(0, 1, "load Frailty29 source manifest")
                reverse_result = run_formal_internal_reverse_evaluation(
                    repository,
                    ptt_training_evidence_path=ptt_training_evidence_path,
                    expected_ptt_training_evidence_sha256=ptt_training_evidence_sha,
                    output_dir=reverse_dir,
                    progress_callback=subprogress,
                    runtime_device=str(
                        plan.payload["motion_detector"]["training_device"]
                    ),
                )
                reverse_report_sha = reverse_result.report_sha256
            else:
                reverse_report_sha = sha256_file(
                    reverse_dir / "motion_internal_reverse_evaluation_report.json"
                )
            manifest["stages"]["frailty29_reverse_evaluation"] = {
                "status": "passed",
                "report_sha256": reverse_report_sha,
                "artifact_dir": reverse_dir.relative_to(root).as_posix(),
            }
            _strict_json(manifest_path, manifest)
            progress_current += 1
            _stage_progress(
                progress,
                progress_current,
                progress_total,
                "Frailty29 reverse evaluation",
            )(1, 1, "completed or resumed")

            comparison_dir, _ = _stage_directory(
                root,
                manifest,
                "motion_model_comparison_package",
                "motion_model_comparison",
                "motion_model_comparison_manifest.json",
            )
            comparison_progress = _stage_progress(
                progress,
                progress_current,
                progress_total,
                "motion-model comparison package",
            )
            comparison_progress(0, 1, "verify and copy frozen parameter sets")
            comparison_manifest = _write_motion_model_comparison_package(
                pipeline_root=pipeline,
                output_dir=comparison_dir,
                legacy_study_relative=str(
                    plan.payload["motion_model_comparison"][
                        "legacy_frailty29_stage5_study"
                    ]
                ),
                ptt_training_evidence_path=ptt_training_evidence_path,
            )
            manifest["stages"]["motion_model_comparison_package"] = {
                "status": "passed",
                "manifest_sha256": sha256_file(comparison_manifest),
                "artifact_dir": comparison_dir.relative_to(root).as_posix(),
            }
            _strict_json(manifest_path, manifest)
            progress_current += 1
            _stage_progress(
                progress,
                progress_current,
                progress_total,
                "motion-model comparison package",
            )(1, 1, "completed or resumed")
            if include_denoiser:
                denoiser_dir, denoiser_complete = _stage_directory(
                    root, manifest, "ptt_denoiser_benchmark", "denoiser",
                    "denoiser_benchmark.json",
                )
                denoiser_path = denoiser_dir / "denoiser_benchmark.json"
                if not denoiser_complete:
                    benchmark = dict(plan.payload["denoiser_benchmark"])
                    subprogress = _stage_progress(
                        progress, progress_current, progress_total, "PTT denoiser benchmark"
                    )
                    subprogress(0, 1, "prepare benchmark")
                    result = run_ptt_denoiser_benchmark(
                        repository,
                        reducer_ids=[str(value) for value in benchmark["reducers"]],
                        segment_s=float(benchmark["segment_s"]),
                        validation=dict(benchmark["validation"]),
                        activities=[str(value) for value in benchmark["activities"]],
                        progress_callback=subprogress,
                    )
                    _strict_json(denoiser_path, result)
                manifest["stages"]["ptt_denoiser_benchmark"] = {
                    "status": "passed",
                    "result_sha256": sha256_file(denoiser_path),
                    "artifact_dir": denoiser_dir.relative_to(root).as_posix(),
                }
                progress_current += 1
                _stage_progress(
                    progress, progress_current, progress_total, "PTT denoiser benchmark"
                )(1, 1, "completed or resumed")
            else:
                manifest["stages"]["ptt_denoiser_benchmark"] = {
                    "status": "skipped_by_cli",
                    "reason": "--no-denoiser",
                }
                _strict_json(manifest_path, manifest)
        else:
            ablation_dir, ablation_complete = _stage_directory(
                root, manifest, "static_peak_ablation", "static_peak_ablation",
                "static_peak_ablation.json",
            )
            result_path = ablation_dir / "static_peak_ablation.json"
            if not ablation_complete:
                subprogress = _stage_progress(
                    progress, progress_current, progress_total, "static peak ablation"
                )
                subprogress(0, 1, "prepare PTT static records")
                _strict_json(
                    result_path,
                    run_static_peak_ablation(
                        repository, plan, progress_callback=subprogress
                    ),
                )
            manifest["stages"]["static_peak_ablation"] = {
                "status": "passed", "result_sha256": sha256_file(result_path),
                "artifact_dir": ablation_dir.relative_to(root).as_posix(),
            }
            progress_current += 1
            _stage_progress(
                progress, progress_current, progress_total, "static peak ablation"
            )(1, 1, "completed or resumed")
        manifest["status"] = "passed"
        _strict_json(manifest_path, manifest)
        report_progress = _stage_progress(
            progress, progress_current, progress_total, "report and result backup"
        )
        report_progress(0, 1, "generate report")
        generate_motion_peak_report(root)
        progress_current += 1
        report_progress = _stage_progress(
            progress, progress_current, progress_total, "report and result backup"
        )
        report_progress(1, 1, "completed")
        progress(ProgressEvent(
            event="motion_peak_study_finished",
            current=progress_total,
            total=progress_total,
            detail_current=1,
            detail_total=1,
            detail_label="study complete",
            message="passed",
        ))
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure_reason"] = f"{type(exc).__name__}: {exc}"
        _strict_json(manifest_path, manifest)
        progress(ProgressEvent(
            event="motion_peak_study_failed",
            current=progress_current,
            total=progress_total,
            detail_current=0,
            detail_total=1,
            detail_label=f"failed: {type(exc).__name__}",
            message="failed",
        ))
        raise
    return root


__all__ = [
    "MSPTDFAST_V2_ID",
    "StudyPlan",
    "align_and_score_beats",
    "generate_motion_peak_report",
    "load_motion_peak_plan",
    "run_motion_peak_study",
    "run_ptt_denoiser_benchmark",
    "run_static_peak_ablation",
]
