"""PTT ECG 监督的延迟拟合与评价 / PTT ECG-supervised delay evaluation.

中文：PPG 相对 ECG R peak 有生理 transit delay。延迟只允许在 training subjects
上拟合；评价 subject 与训练 roster 必须不相交。报告同时保留未校正与校正后的
timing/F1、PPI/HR 误差和 coverage，不把 detector 验证误写成 PPG peak 成绩。

English: PPG pulses lag ECG R peaks physiologically. Delay is fitted only on training
subjects and evaluation subjects must be disjoint. Raw and corrected timing/F1,
PPI/HR errors, and coverage are all retained.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def _subject_hash(subjects: Sequence[str]) -> str:
    """稳定 subject hash / Stable subject hash."""

    payload = "\n".join(sorted({str(subject) for subject in subjects})).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _match_pairs(
    reference: np.ndarray,
    candidate: np.ndarray,
    tolerance_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """单调一对一匹配并保留索引 / Monotonic one-to-one indexed matching."""

    left = np.asarray(reference, dtype=np.int64)
    right = np.asarray(candidate, dtype=np.int64)
    left_indices = []
    right_indices = []
    i = 0
    j = 0
    while i < left.size and j < right.size:
        delta = int(right[j] - left[i])
        if abs(delta) <= int(tolerance_samples):
            left_indices.append(i)
            right_indices.append(j)
            i += 1
            j += 1
        elif right[j] < left[i]:
            j += 1
        else:
            i += 1
    return np.asarray(left_indices, dtype=np.int64), np.asarray(right_indices, dtype=np.int64)


def _following_delays(
    ecg_peaks: np.ndarray,
    ppg_peaks: np.ndarray,
    minimum_samples: int,
    maximum_samples: int,
) -> np.ndarray:
    """匹配 ECG 后首个合法 PPG pulse / Match the first valid following pulse."""

    ecg = np.asarray(ecg_peaks, dtype=np.int64)
    ppg = np.asarray(ppg_peaks, dtype=np.int64)
    delays = []
    ppg_index = 0
    for ecg_peak in ecg:
        while ppg_index < ppg.size and ppg[ppg_index] - ecg_peak < minimum_samples:
            ppg_index += 1
        if ppg_index >= ppg.size:
            break
        delay = int(ppg[ppg_index] - ecg_peak)
        if delay <= maximum_samples:
            delays.append(delay)
            ppg_index += 1
    return np.asarray(delays, dtype=np.int64)


@dataclass(frozen=True)
class TransitDelayArtifact:
    """训练折 PTT delay artifact / Training-only PTT delay artifact."""

    schema_version: str
    artifact_id: str
    sampling_rate_hz: float
    delay_samples: int
    delay_sec: float
    fit_subject_ids: tuple[str, ...]
    fit_subject_ids_sha256: str
    matched_delay_count: int
    minimum_delay_sec: float
    maximum_delay_sec: float
    fit_role: str
    source_reference: str
    dataset_id: str = "ptt_ppg_1_1_0_local"
    training_split_id: str = ""
    preprocessing_profile_id: str = "external_ppg_to_400_polyphase_v1"
    algorithm_id: str = "m3_ptt_median_transit_delay_v1"

    def to_dict(self) -> dict[str, object]:
        """序列化 artifact / Serialize the artifact."""

        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "sampling_rate_hz": self.sampling_rate_hz,
            "delay_samples": self.delay_samples,
            "delay_sec": self.delay_sec,
            "fit_subject_ids": list(self.fit_subject_ids),
            "fit_subject_ids_sha256": self.fit_subject_ids_sha256,
            "matched_delay_count": self.matched_delay_count,
            "minimum_delay_sec": self.minimum_delay_sec,
            "maximum_delay_sec": self.maximum_delay_sec,
            "fit_role": self.fit_role,
            "source_reference": self.source_reference,
            "dataset_id": self.dataset_id,
            "training_split_id": self.training_split_id,
            "preprocessing_profile_id": self.preprocessing_profile_id,
            "algorithm_id": self.algorithm_id,
        }


def fit_transit_delay(
    subject_peak_pairs: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    training_subject_ids: Sequence[str],
    fs_hz: float,
    fit_role: str,
    training_split_id: str,
    minimum_delay_sec: float = 0.05,
    maximum_delay_sec: float = 0.60,
    dataset_id: str = "ptt_ppg_1_1_0_local",
    preprocessing_profile_id: str = "external_ppg_to_400_polyphase_v1",
) -> TransitDelayArtifact:
    """仅训练 subjects 拟合 median transit delay / Fit delay on training only."""

    if fit_role != "training":
        raise ValueError("transit_delay_fit_requires_training_role")
    if not str(training_split_id).strip():
        raise ValueError("transit_delay_training_split_id_required")
    expected = {str(subject) for subject in training_subject_ids}
    observed = {str(subject) for subject in subject_peak_pairs}
    if not expected or observed != expected:
        raise ValueError("transit_delay_training_roster_mismatch")
    minimum = int(round(float(minimum_delay_sec) * float(fs_hz)))
    maximum = int(round(float(maximum_delay_sec) * float(fs_hz)))
    all_delays = []
    for subject in sorted(expected):
        ecg, ppg = subject_peak_pairs[subject]
        delays = _following_delays(ecg, ppg, minimum, maximum)
        if delays.size:
            all_delays.append(delays)
    if not all_delays:
        raise ValueError("no_valid_training_transit_delays")
    delays = np.concatenate(all_delays)
    delay_samples = int(np.rint(np.median(delays)))
    subject_hash = _subject_hash(sorted(expected))
    return TransitDelayArtifact(
        schema_version="m3.transit_delay_artifact.v1",
        artifact_id=f"ptt_delay_{subject_hash[:12]}_{delay_samples}samples",
        sampling_rate_hz=float(fs_hz),
        delay_samples=delay_samples,
        delay_sec=float(delay_samples / float(fs_hz)),
        fit_subject_ids=tuple(sorted(expected)),
        fit_subject_ids_sha256=subject_hash,
        matched_delay_count=int(delays.size),
        minimum_delay_sec=float(minimum_delay_sec),
        maximum_delay_sec=float(maximum_delay_sec),
        fit_role=fit_role,
        source_reference="PTT ECG peaks manually verified per M2 external manifest",
        dataset_id=str(dataset_id),
        training_split_id=str(training_split_id),
        preprocessing_profile_id=str(preprocessing_profile_id),
    )


def _score(reference: np.ndarray, candidate: np.ndarray, tolerance: int) -> dict[str, float | int]:
    """计算 one-to-one event score / Compute one-to-one event scores."""

    left, right = _match_pairs(reference, candidate, tolerance)
    matched = int(left.size)
    precision = matched / len(candidate) if len(candidate) else 0.0
    recall = matched / len(reference) if len(reference) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    timing = (
        (np.asarray(candidate)[right] - np.asarray(reference)[left]).astype(np.float64)
        if matched
        else np.empty(0)
    )
    return {
        "matched": matched,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "timing_error_samples_median": float(np.median(timing)) if timing.size else None,
        "timing_error_samples_mae": float(np.mean(np.abs(timing))) if timing.size else None,
    }


def _complete_scorecard(
    reference: np.ndarray,
    candidate: np.ndarray,
    tolerance: int,
    fs_hz: float,
) -> dict[str, float | int | str | None]:
    """统一 event、PPI、HR、coverage 口径 / Build one symmetric scorecard."""

    score = _score(reference, candidate, tolerance)
    left, right = _match_pairs(reference, candidate, tolerance)
    ppi_errors_ms = []
    for index in range(1, left.size):
        if left[index] - left[index - 1] == 1 and right[index] - right[index - 1] == 1:
            reference_ppi = (
                reference[left[index]] - reference[left[index - 1]]
            ) / float(fs_hz)
            candidate_ppi = (
                candidate[right[index]] - candidate[right[index - 1]]
            ) / float(fs_hz)
            ppi_errors_ms.append((candidate_ppi - reference_ppi) * 1000.0)
    reference_ppi = np.diff(reference) / float(fs_hz)
    candidate_ppi = np.diff(candidate) / float(fs_hz)
    hr_error = None
    if reference_ppi.size >= 4 and candidate_ppi.size >= 4:
        hr_error = float(
            60.0 / np.median(candidate_ppi) - 60.0 / np.median(reference_ppi)
        )
    score.update(
        {
            "timing_error_ms_median": (
                None
                if score["timing_error_samples_median"] is None
                else float(score["timing_error_samples_median"] * 1000.0 / fs_hz)
            ),
            "timing_error_ms_mae": (
                None
                if score["timing_error_samples_mae"] is None
                else float(score["timing_error_samples_mae"] * 1000.0 / fs_hz)
            ),
            "ppi_error_ms_mae": (
                float(np.mean(np.abs(ppi_errors_ms))) if ppi_errors_ms else None
            ),
            "hr_error_bpm": hr_error,
            "coverage_fraction": float(left.size / len(reference)) if len(reference) else 0.0,
            "failure_reason": None if left.size else "NO_MATCHED_PEAKS",
        }
    )
    return score


def evaluate_ppg_against_ecg(
    ecg_peaks: np.ndarray,
    ppg_peaks: np.ndarray,
    *,
    evaluation_subject_id: str,
    fs_hz: float,
    delay_artifact: TransitDelayArtifact,
    tolerance_ms: float = 50.0,
) -> dict[str, object]:
    """输出未校正/校正完整 scorecard / Emit raw and corrected scorecards."""

    if evaluation_subject_id in set(delay_artifact.fit_subject_ids):
        raise ValueError("evaluation_subject_present_in_delay_training")
    if not np.isclose(float(fs_hz), delay_artifact.sampling_rate_hz, atol=1e-12, rtol=0.0):
        raise ValueError("transit_delay_sampling_rate_mismatch")
    ecg = np.asarray(ecg_peaks, dtype=np.int64)
    ppg = np.asarray(ppg_peaks, dtype=np.int64)
    corrected = ppg - int(delay_artifact.delay_samples)
    tolerance = int(round(float(tolerance_ms) * float(fs_hz) / 1000.0))
    raw_score = _complete_scorecard(ecg, ppg, tolerance, fs_hz)
    corrected_score = _complete_scorecard(ecg, corrected, tolerance, fs_hz)
    return {
        "schema_version": "m3.ptt_reference_evaluation.v1",
        "evaluation_subject_id": str(evaluation_subject_id),
        "delay_artifact_id": delay_artifact.artifact_id,
        "dataset_id": delay_artifact.dataset_id,
        "training_split_id": delay_artifact.training_split_id,
        "preprocessing_profile_id": delay_artifact.preprocessing_profile_id,
        "algorithm_id": "m3_ptt_ecg_reference_evaluation_v1",
        "delay_training_subjects_disjoint": True,
        "tolerance_ms": float(tolerance_ms),
        "raw": raw_score,
        "delay_corrected": corrected_score,
        # 中文：保留旧顶层 alias，只指向 corrected 分支，便于历史调用迁移。
        # English: Keep top-level aliases explicitly mapped to the corrected branch.
        "ppi_error_ms_mae": corrected_score["ppi_error_ms_mae"],
        "hr_error_bpm": corrected_score["hr_error_bpm"],
        "coverage_fraction": corrected_score["coverage_fraction"],
        "failure_reason": corrected_score["failure_reason"],
    }


__all__ = [
    "TransitDelayArtifact",
    "evaluate_ppg_against_ecg",
    "fit_transit_delay",
]
