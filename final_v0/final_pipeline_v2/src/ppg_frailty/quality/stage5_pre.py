"""Executable PTT motion/denoiser study and static peak-detector ablation.

This module deliberately composes the existing motion and artifact APIs.  It
does not duplicate the frailty experiment runner and never imports PTT labels
into the 29-participant motion-model fit.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml
from scipy import signal

from ..artifacts.router import get_reducer
from ..data.external_manifest import (
    M2_EXTERNAL_RELATIVE_PATH,
    PTT_DATASET_ID,
    ExternalRecord,
    adapt_ptt_synchronized_channels,
    load_m2_external_manifest,
)
from ..peaks.pairing import match_events
from ..peaks.resolver import CANONICAL_DETECTOR_ID, detect_pulses
from ..provenance import sha256_file
from ..signal.motion_imu import (
    PTT_STATIC_CALIBRATION_ROLE,
    RollPitchEkfConfig,
    fit_motion_imu_calibration,
    preprocess_motion_imu_calibrated_ekf,
)
from ..signal.preprocess import preprocess_ppg_pair
from .motion_reference import (
    PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH,
    PTT_IMU_UNIT_EVIDENCE_SHA256,
    PTT_SOURCE_ROOT,
    load_ptt_imu_unit_evidence,
    run_formal_internal_motion_reference,
    run_formal_ptt_motion_reference,
)


STAGE5_SCHEMA = "ppg_frailty.stage5_pre_motion_ptt.v1"
PEAK_ABLATION_SCHEMA = "ppg_frailty.stage_ablation_01_static_peaks.v1"
RESULT_SCHEMA = "ppg_frailty.motion_peak_study_result.v1"
MSPTDFAST_V2_ID = "msptdfast_v2_3_python_port"
MSPTDFAST_AUTHOR_SOURCE_SHA256 = (
    "39f5010f1d485f2dc180bffbbe662dc9bd16e8116bd1ae630761f1e2b58bcabd"
)
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
    else:
        if data.get("activities") != ["sit"]:
            raise ValueError("Stage-ablation-01 is a pure-static sit-only experiment")
        algorithms = data.get("algorithms")
        ids = {
            str(item.get("algorithm_id"))
            for item in algorithms or ()
            if isinstance(item, Mapping)
        }
        if ids != {CANONICAL_DETECTOR_ID, MSPTDFAST_V2_ID}:
            raise ValueError("static peak ablation must compare Aboy++ and MSPTDfast v2")
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
        parameters = comparator.get("parameters")
        expected_parameter_keys = {
            "target_downsample_hz", "minimum_heart_rate_bpm", "window_s",
            "overlap_fraction",
        }
        if not isinstance(parameters, Mapping) or set(parameters) != expected_parameter_keys:
            raise ValueError("MSPTDfast parameters must be complete and contain no unknown keys")
        for key in ("target_downsample_hz", "minimum_heart_rate_bpm", "window_s"):
            _finite(parameters.get(key), f"MSPTDfast.{key}", positive=True)
        overlap = _finite(parameters.get("overlap_fraction"), "MSPTDfast.overlap_fraction")
        if not 0.0 <= overlap < 1.0:
            raise ValueError("MSPTDfast.overlap_fraction must be in [0, 1)")
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


def _msptd_window_peaks(
    values: np.ndarray,
    fs_hz: float,
    *,
    minimum_heart_rate_bpm: float,
) -> np.ndarray:
    """Equation-level Python port of the author's MSPTD core for one window."""

    x = signal.detrend(np.asarray(values, dtype=np.float64), type="linear")
    n = x.size
    if n < 5:
        return np.empty(0, dtype=np.int64)
    largest_scale = int(math.ceil(n / 2.0) - 1)
    duration = n / float(fs_hz)
    scales = np.arange(1, largest_scale + 1, dtype=np.int64)
    scale_frequencies = (largest_scale / scales) / duration
    included = scales[scale_frequencies >= minimum_heart_rate_bpm / 60.0]
    if included.size == 0:
        return np.empty(0, dtype=np.int64)
    max_scale = int(included[-1])
    local_max = np.zeros((max_scale, n), dtype=bool)
    for scale_index, k in enumerate(range(1, max_scale + 1)):
        centers = np.arange(k, n - k)
        local_max[scale_index, centers] = (
            (x[centers] > x[centers - k]) & (x[centers] > x[centers + k])
        )
    lambda_max = int(np.argmax(np.sum(local_max, axis=1))) + 1
    return np.flatnonzero(np.all(local_max[:lambda_max], axis=0)).astype(np.int64)


def detect_msptdfast_v2(
    values: np.ndarray,
    fs_hz: float,
    *,
    target_downsample_hz: float = 20.0,
    minimum_heart_rate_bpm: float = 30.0,
    window_s: float = 6.0,
    overlap_fraction: float = 0.2,
) -> np.ndarray:
    """Detect pulse peaks using the published MSPTDfast v2 configuration.

    The port preserves the author's 6-s/20%-overlap windows, integer-factor
    downsampling to at least 20 Hz, >30-bpm scale truncation and full-rate peak
    refinement.  It is not labelled bitwise MATLAB parity.
    """

    source = np.asarray(values, dtype=np.float64).reshape(-1)
    if source.size < 3 or not np.isfinite(source).all() or fs_hz <= 0.0:
        raise ValueError("MSPTDfast requires one finite non-empty PPG signal and fs")
    target_downsample_hz = _finite(
        target_downsample_hz, "target_downsample_hz", positive=True
    )
    minimum_heart_rate_bpm = _finite(
        minimum_heart_rate_bpm, "minimum_heart_rate_bpm", positive=True
    )
    window_s = _finite(window_s, "window_s", positive=True)
    overlap_fraction = _finite(overlap_fraction, "overlap_fraction")
    if not 0.0 <= overlap_fraction < 1.0:
        raise ValueError("overlap_fraction must be in [0, 1)")
    nominal = int(round(window_s * fs_hz))
    hop = int(round(nominal * (1.0 - overlap_fraction)))
    # MATLAB reference uses inclusive win_ends=start+6*fs.
    window = nominal + 1
    detected: list[int] = []
    factor = max(1, int(math.floor(fs_hz / target_downsample_hz)))
    relative_fs = fs_hz / factor
    tolerance_s = 0.2 if relative_fs < 10.0 else 0.1 if relative_fs < 20.0 else 0.05
    tolerance = int(math.ceil(fs_hz * tolerance_s))
    for start in _window_starts(source.size, min(window, source.size), max(hop, 1)):
        segment = source[start : start + window]
        downsampled = segment[::factor]
        for peak in _msptd_window_peaks(
            downsampled,
            relative_fs,
            minimum_heart_rate_bpm=minimum_heart_rate_bpm,
        ):
            # Translate the author's one-based p*factor before local refinement.
            approximate = int((int(peak) + 1) * factor - 1)
            left = max(0, approximate - tolerance)
            right = min(segment.size, approximate + tolerance + 1)
            if right > left:
                detected.append(start + left + int(np.argmax(segment[left:right])))
    return np.unique(np.asarray(detected, dtype=np.int64))


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


def _detect_current(values: np.ndarray, fs_hz: float) -> np.ndarray:
    result = detect_pulses(
        values,
        detector_id=CANONICAL_DETECTOR_ID,
        fs_hz=fs_hz,
        min_observation_sec=6.0,
        min_peaks=3,
    )
    return np.asarray(result.peak_timestamps_s, dtype=np.float64)


def _score_segment(
    values: np.ndarray,
    reference_s: np.ndarray,
    *,
    algorithm_id: str,
    algorithm_parameters: Mapping[str, Any] | None = None,
    fs_hz: float,
    validation: Mapping[str, Any],
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    if algorithm_id == CANONICAL_DETECTOR_ID:
        if algorithm_parameters:
            raise ValueError("Aboy detector received unsupported study parameters")
        peaks_s = _detect_current(values, fs_hz)
    elif algorithm_id == MSPTDFAST_V2_ID:
        peaks_s = detect_msptdfast_v2(
            values,
            fs_hz,
            **dict(algorithm_parameters or {}),
        ).astype(np.float64) / fs_hz
    else:
        raise ValueError(f"unknown peak detector in study: {algorithm_id}")
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
    for subject_id, activity_records in sorted(by_subject.items()):
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
) -> dict[str, Any]:
    """Run the two detectors on complete PTT sit records only."""

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
    for row in records:
        adapted, reference = _load_record(repository, row)
        _, filtered, _ = preprocess_ppg_pair(
            adapted.ppg_red_ir, fs_hz=400.0, timestamps_s=adapted.timestamps_s
        )
        for start in _window_starts(filtered.shape[0], window_samples, window_samples):
            stop = min(start + window_samples, filtered.shape[0])
            if stop - start < int(round(8.0 * 400.0)):
                continue
            current_reference = reference[
                (reference >= start / 400.0) & (reference < stop / 400.0)
            ] - start / 400.0
            if current_reference.size < 3:
                continue
            for algorithm_id, parameters in algorithm_parameters.items():
                for channel_index, channel in enumerate(_CHANNELS):
                    scored, elapsed = _score_segment(
                        filtered[start:stop, channel_index], current_reference,
                        algorithm_id=algorithm_id,
                        algorithm_parameters=parameters,
                        fs_hz=400.0,
                        validation=validation,
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
    return {
        "schema_version": "ppg_frailty.stage_ablation_01_static_peak_result.v1",
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
        label = "final-all29" if payload.get("final_fit") else (
            f"fold-{int(payload['fold_index']) + 1}"
        )
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
    if manifest["study_type"] == "stage5_pre_motion_ptt":
        internal_dir = _report_stage_dir(root, manifest, "internal_motion_oof", "motion_internal")
        external_dir = _report_stage_dir(root, manifest, "ptt_motion_external", "motion_external")
        denoiser_dir = _report_stage_dir(root, manifest, "ptt_denoiser_benchmark", "denoiser")
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
        for name, rows, title in (
            ("motion_internal_confusion_matrix.png", internal_rows, "Internal 29-person OOF"),
            ("motion_ptt_confusion_matrix.png", external_rows, "PTT external evaluation"),
        ):
            path = figures / name
            _plot_confusion(path, _confusion(rows), title)
            images.append(path)
        learning_path = figures / "motion_training_learning_curves.png"
        history_paths = tuple(internal_dir.rglob("motion_training_history.json"))
        if not history_paths:
            raise ValueError("completed Stage5-pre report requires motion training histories")
        _plot_motion_learning_curves(learning_path, history_paths)
        images.append(learning_path)
        benchmark = json.loads(
            (denoiser_dir / "denoiser_benchmark.json").read_text(encoding="utf-8")
        )
        summary_rows = benchmark["summary_rows"]
        _write_csv(tables / "denoiser_summary.csv", summary_rows)
        _strict_json(tables / "denoiser_summary.json", summary_rows)
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
        headline = {
            "internal_major_metrics": internal["major_metrics"],
            "ptt_major_metrics": external["major_metrics"],
            "denoiser_summary_rows": summary_rows,
        }
        headline_metric_rows = [
            {"dataset": dataset, "metric": metric, "value": value}
            for dataset, metrics in (
                ("internal_oof", internal["major_metrics"]),
                ("ptt_external", external["major_metrics"]),
            )
            for metric, value in metrics.items()
            if not isinstance(value, Mapping)
        ]
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
    lines = [
        f"# {manifest['study_id']}", "", f"Status: **{manifest['status']}**", "",
        "## Scientific scope", "",
        str(manifest["scientific_scope"]), "",
        "## Figures", "",
        *[f"![{path.stem}](figures/{path.name})" for path in images], "",
        "## Numerical outputs", "",
        _markdown_table(headline_metric_rows), "",
        _markdown_table(summary_rows), "",
        "Machine-readable values are in `study_summary.json` and `tables/`.", "",
    ]
    markdown = "\n".join(lines)
    (root / "STUDY_SUMMARY.md").write_text(markdown, encoding="utf-8")
    html_images = "\n".join(
        f'<figure><img src="figures/{path.name}" alt="{path.stem}">'
        f"<figcaption>{path.stem}</figcaption></figure>"
        for path in images
    )
    (root / "STUDY_SUMMARY.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>"
        + manifest["study_id"] + "</title><h1>" + manifest["study_id"]
        + "</h1><p>Status: " + manifest["status"] + "</p>"
        + "<h2>Numerical outputs</h2>" + _html_table(headline_metric_rows)
        + _html_table(summary_rows) + "<h2>Figures</h2>" + html_images,
        encoding="utf-8",
    )
    indexed = [
        path
        for path in root.rglob("*")
        if path.is_file() and "result_backup" not in path.parts
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
) -> Path:
    """Execute a Stage5-pre or static peak-ablation plan."""

    plan = load_motion_peak_plan(plan_path)
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
        existing = load_motion_peak_plan(resolved_plan)
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
            "29-participant grouped OOF motion detector plus frozen-model PTT "
            "evaluation and PTT denoiser comparison; PTT is not claimed independent"
            if plan.schema_version == STAGE5_SCHEMA
            else "PTT sit-only detector comparison; no motion segments and no denoiser selection"
        ),
        "plan_sha256": sha256_file(resolved_plan),
        "stages": {},
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
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
                result = run_formal_internal_motion_reference(repository, output_dir=internal_dir)
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
                external = run_formal_ptt_motion_reference(
                    repository,
                    internal_evidence_path=evidence_path,
                    expected_internal_evidence_sha256=evidence_sha,
                    output_dir=external_dir,
                    unit_evidence_path=repository / PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH,
                    expected_unit_evidence_sha256=PTT_IMU_UNIT_EVIDENCE_SHA256,
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
            denoiser_dir, denoiser_complete = _stage_directory(
                root, manifest, "ptt_denoiser_benchmark", "denoiser",
                "denoiser_benchmark.json",
            )
            denoiser_path = denoiser_dir / "denoiser_benchmark.json"
            if not denoiser_complete:
                benchmark = dict(plan.payload["denoiser_benchmark"])
                result = run_ptt_denoiser_benchmark(
                    repository,
                    reducer_ids=[str(value) for value in benchmark["reducers"]],
                    segment_s=float(benchmark["segment_s"]),
                    validation=dict(benchmark["validation"]),
                    activities=[str(value) for value in benchmark["activities"]],
                )
                _strict_json(denoiser_path, result)
                manifest["stages"]["ptt_denoiser_benchmark"] = {
                    "status": "passed", "result_sha256": sha256_file(denoiser_path),
                    "artifact_dir": denoiser_dir.relative_to(root).as_posix(),
                }
            else:
                manifest["stages"]["ptt_denoiser_benchmark"] = {
                    "status": "passed",
                    "result_sha256": sha256_file(denoiser_path),
                    "artifact_dir": denoiser_dir.relative_to(root).as_posix(),
                }
        else:
            ablation_dir, ablation_complete = _stage_directory(
                root, manifest, "static_peak_ablation", "static_peak_ablation",
                "static_peak_ablation.json",
            )
            result_path = ablation_dir / "static_peak_ablation.json"
            if not ablation_complete:
                _strict_json(result_path, run_static_peak_ablation(repository, plan))
            manifest["stages"]["static_peak_ablation"] = {
                "status": "passed", "result_sha256": sha256_file(result_path),
                "artifact_dir": ablation_dir.relative_to(root).as_posix(),
            }
        manifest["status"] = "passed"
        _strict_json(manifest_path, manifest)
        generate_motion_peak_report(root)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure_reason"] = f"{type(exc).__name__}: {exc}"
        _strict_json(manifest_path, manifest)
        raise
    return root


__all__ = [
    "MSPTDFAST_V2_ID",
    "StudyPlan",
    "align_and_score_beats",
    "detect_msptdfast_v2",
    "generate_motion_peak_report",
    "load_motion_peak_plan",
    "run_motion_peak_study",
    "run_ptt_denoiser_benchmark",
    "run_static_peak_ablation",
]
