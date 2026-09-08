"""Optional legacy-compatible per-file raw-window SQI selection."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Any, Mapping

import numpy as np
from scipy import signal

WINDOW_SELECTION_POLICIES = frozenset({"none", "legacy_per_file_top_fraction"})
WINDOW_SELECTION_APPLICATION_SCOPES = frozenset({
    "outer_train_only",
    "all_partitions",
    "legacy_train_and_aggregation",
})
LEGACY_WINDOW_SCORE_ALGORITHM = "legacy_cardiac_motion_window_sqi_v1"


@dataclass(frozen=True)
class WindowSelectionConfig:
    """One independently selectable window-retention strategy."""

    policy: str = "none"
    keep_fraction: float = 1.0
    score_algorithm: str = LEGACY_WINDOW_SCORE_ALGORITHM
    application_scope: str = "outer_train_only"

    def __post_init__(self) -> None:
        if self.policy not in WINDOW_SELECTION_POLICIES:
            raise ValueError("quality.window_selection.policy must be one of " f"{sorted(WINDOW_SELECTION_POLICIES)}")
        if (isinstance(self.keep_fraction, bool) or not isinstance(self.keep_fraction,
                                                                   (int, float, np.integer, np.floating))
                or not np.isfinite(self.keep_fraction) or not 0.0 < float(self.keep_fraction) <= 1.0):
            raise ValueError("quality.window_selection.keep_fraction must be finite in (0,1]")
        if self.score_algorithm != LEGACY_WINDOW_SCORE_ALGORITHM:
            raise ValueError("quality.window_selection.score_algorithm must be " f"{LEGACY_WINDOW_SCORE_ALGORITHM}")
        if self.application_scope not in WINDOW_SELECTION_APPLICATION_SCOPES:
            raise ValueError("quality.window_selection.application_scope must be one of "
                             f"{sorted(WINDOW_SELECTION_APPLICATION_SCOPES)}")
        if self.policy == "none" and float(self.keep_fraction) != 1.0:
            raise ValueError("quality.window_selection.keep_fraction must be 1 when policy=none")
        if self.policy == "none" and self.application_scope != "outer_train_only":
            raise ValueError("quality.window_selection.application_scope must remain "
                             "outer_train_only when policy=none")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "WindowSelectionConfig":
        if value is None:
            return cls()
        if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
            raise ValueError("quality.window_selection must be a string-keyed mapping")
        allowed = {"policy", "keep_fraction", "score_algorithm", "application_scope"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"quality.window_selection contains unknown fields: {unknown}")
        return cls(**dict(value))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "keep_fraction": float(self.keep_fraction),
            "score_algorithm": self.score_algorithm,
            "application_scope": self.application_scope,
        }


def legacy_window_sqi_scores(
    values: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    fs_hz: float = 400.0,
) -> np.ndarray:
    """Port the historical cardiac/motion window score without labels.

    The historical 5th/95th percentile display scaling is retained per file.
    Selection is rank based, so no statistic is shared between files or folds.
    """

    windows = np.asarray(values, dtype=np.float32)
    if windows.ndim != 3 or windows.shape[1] < 2:
        raise ValueError("window SQI values must be [window,channel,time]")
    if not np.isfinite(windows).all():
        raise ValueError("window SQI values must be finite")
    if not np.isfinite(fs_hz) or float(fs_hz) <= 0.0:
        raise ValueError("window SQI fs_hz must be finite and positive")
    if valid_mask is None:
        masks = np.ones((windows.shape[0], windows.shape[2]), dtype=bool)
    else:
        masks = np.asarray(valid_mask, dtype=bool)
        if masks.shape != (windows.shape[0], windows.shape[2]):
            raise ValueError("window SQI valid_mask must be [window,time]")

    scores: list[float] = []
    for window, mask in zip(windows, masks):
        valid_length = int(np.flatnonzero(mask)[-1] + 1) if np.any(mask) else 0
        if valid_length < 16:
            scores.append(0.0)
            continue
        # The legacy scorer received the old classifier's all-channel,
        # per-window robust-standardized tensor. Reproduce that scorer-local
        # view without changing the independently configured V2 model tensor.
        segment = np.asarray(window[:, :valid_length].T, dtype=np.float64)
        median = np.median(segment, axis=0, keepdims=True)
        q25, q75 = np.percentile(segment, (25.0, 75.0), axis=0, keepdims=True)
        robust_scale = (q75 - q25) / 1.349
        standard_scale = np.std(segment, axis=0, keepdims=True)
        scale = np.where(robust_scale > 1e-6, robust_scale, standard_scale)
        legacy_window = np.clip(
            (segment - median) / (scale + 1e-6),
            -8.0,
            8.0,
        ).T
        red = np.asarray(legacy_window[0], dtype=np.float64)
        ir = np.asarray(legacy_window[1], dtype=np.float64)
        ppg = ir if np.std(ir) >= np.std(red) else red
        acc = (np.linalg.norm(legacy_window[2:5].astype(np.float64), axis=0)
               if window.shape[0] >= 5 else np.zeros(valid_length, dtype=np.float64))
        gyro = (np.linalg.norm(legacy_window[5:8].astype(np.float64), axis=0)
                if window.shape[0] >= 8 else np.zeros(valid_length, dtype=np.float64))
        ppg_std = float(np.std(ppg))
        if ppg_std < 1e-8:
            scores.append(0.0)
            continue
        frequencies, psd = signal.welch(
            ppg,
            fs=float(fs_hz),
            nperseg=min(512, ppg.size),
        )
        total = float(np.trapezoid(psd, frequencies)) + 1e-12
        cardiac = (frequencies >= 0.5) & (frequencies <= 3.0)
        spectral_ratio = float(np.trapezoid(psd[cardiac], frequencies[cardiac]) / total) if np.any(cardiac) else 0.0
        peaks, _ = signal.find_peaks(
            (ppg - np.median(ppg)) / (ppg_std + 1e-8),
            distance=max(1, int(round(0.28 * float(fs_hz)))),
            prominence=0.3,
        )
        if peaks.size >= 3:
            intervals = np.diff(peaks).astype(np.float64)
            ppi_stability = 1.0 / (1.0 + float(np.std(intervals) / (np.mean(intervals) + 1e-8)))
            peak_density = min(
                1.0,
                float(peaks.size) / max(2.0, ppg.size / float(fs_hz) * 3.0),
            )
        else:
            ppi_stability = 0.0
            peak_density = 0.0
        motion = float(np.sqrt(np.mean(np.square(acc))) + 0.25 * np.sqrt(np.mean(np.square(gyro))))
        motion_penalty = 1.0 / (1.0 + max(0.0, motion - 1.0))
        score = 0.40 * spectral_ratio + 0.35 * ppi_stability + 0.15 * peak_density + 0.10 * motion_penalty
        scores.append(float(score) if np.isfinite(score) else 0.0)

    result = np.asarray(scores, dtype=np.float32)
    if result.size and float(np.max(result)) > float(np.min(result)):
        lower = float(np.percentile(result, 5.0))
        upper = float(np.percentile(result, 95.0))
        result = np.clip(
            (result - lower) / (upper - lower + 1e-8),
            0.0,
            1.0,
        ).astype(np.float32)
    return np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)


def legacy_per_file_top_fraction_mask(
    scores: np.ndarray,
    *,
    keep_fraction: float,
) -> np.ndarray:
    """Retain ``ceil(n*fraction)`` highest-scoring windows for one file."""

    quality = np.asarray(scores, dtype=np.float64)
    config = WindowSelectionConfig(
        policy="legacy_per_file_top_fraction",
        keep_fraction=keep_fraction,
    )
    if quality.ndim != 1 or not np.isfinite(quality).all():
        raise ValueError("window SQI scores must be a finite vector")
    keep = np.zeros(quality.size, dtype=bool)
    if quality.size == 0:
        return keep
    n_keep = max(1, int(math.ceil(quality.size * config.keep_fraction)))
    order = np.argsort(quality)[::-1]
    keep[order[:n_keep]] = True
    return keep


def score_raw_windows(
    raw_windows: Any,
    config: WindowSelectionConfig,
) -> tuple[Any, dict[str, Any]]:
    """Attach the selected file-local score without changing window retention.

    The historical classifier computed SQI for held-out windows even when its
    training retention policy was train-only.  Keeping scoring separate from
    selection preserves that behavior and lets OOF aggregation consume the
    real row-aligned scores without leaking labels or cross-file statistics.
    """

    count = int(raw_windows.values.shape[0])
    if config.policy == "none":
        return raw_windows, {
            **config.to_mapping(),
            "input_window_count": count,
            "retained_window_count": count,
            "aggregation_window_count": count,
            "fitted_on_participant_ids": [],
            "uses_labels": False,
            "cross_file_statistics": False,
            "score_vector_sha256": None,
        }
    scores = legacy_window_sqi_scores(
        raw_windows.values,
        raw_windows.valid_mask,
    )
    score_hash = hashlib.sha256(np.asarray(scores, dtype="<f4").tobytes()).hexdigest()
    scored = replace(
        raw_windows,
        window_quality_scores=scores,
        provenance={
            **dict(raw_windows.provenance),
            "window_quality_scoring": {
                "score_algorithm": config.score_algorithm,
                "window_count": count,
                "score_vector_sha256": score_hash,
                "uses_labels": False,
                "cross_file_statistics": False,
            },
        },
    )
    return scored, {
        **config.to_mapping(),
        "input_window_count": count,
        "retained_window_count": count,
        "aggregation_window_count": count,
        "fitted_on_participant_ids": [],
        "uses_labels": False,
        "cross_file_statistics": False,
        "score_vector_sha256": score_hash,
    }


def select_raw_windows(raw_windows: Any, config: WindowSelectionConfig) -> tuple[Any, dict[str, Any]]:
    """Apply one file-local retention strategy to a scored ``RawWindows``."""

    scored, score_summary = score_raw_windows(raw_windows, config)
    count = int(scored.values.shape[0])
    if config.policy == "none":
        return scored, score_summary
    scores = np.asarray(scored.window_quality_scores, dtype=np.float32)
    keep = legacy_per_file_top_fraction_mask(
        scores,
        keep_fraction=config.keep_fraction,
    )
    selected = replace(
        scored,
        values=scored.values[keep],
        valid_mask=scored.valid_mask[keep],
        start_samples=scored.start_samples[keep],
        window_quality_scores=scores[keep],
        window_aggregation_mask=np.ones(int(np.count_nonzero(keep)), dtype=bool),
        provenance={
            **dict(scored.provenance),
            "window_quality_selection": {
                **config.to_mapping(),
                "input_window_count":
                count,
                "retained_window_count":
                int(np.count_nonzero(keep)),
                "score_vector_sha256":
                score_summary["score_vector_sha256"],
                "retained_score_vector_sha256":
                hashlib.sha256(np.asarray(scores[keep], dtype="<f4").tobytes()).hexdigest(),
                "retained_start_samples_sha256":
                hashlib.sha256(np.asarray(scored.start_samples[keep], dtype="<i8").tobytes()).hexdigest(),
            },
        },
    )
    return selected, {
        **config.to_mapping(),
        "input_window_count": count,
        "retained_window_count": int(np.count_nonzero(keep)),
        "aggregation_window_count": int(np.count_nonzero(keep)),
        "fitted_on_participant_ids": [],
        "uses_labels": False,
        "cross_file_statistics": False,
        "score_vector_sha256": score_summary["score_vector_sha256"],
        "retained_score_vector_sha256": hashlib.sha256(np.asarray(scores[keep], dtype="<f4").tobytes()).hexdigest(),
    }


def mark_raw_windows_for_aggregation(
    raw_windows: Any,
    config: WindowSelectionConfig,
) -> tuple[Any, dict[str, Any]]:
    """Keep every prediction row but mark the historical aggregation subset.

    This reproduces the old classifier's held-out *file* aggregation semantics:
    window metrics see every prediction, while the V2 hierarchy consumes the
    configured top fraction independently inside each file. Continuous SQI
    weighting remains a separate aggregation switch. The old script's direct
    subject-key aggregation ranked every subject window together; that is not
    equivalent to V2's mandatory window→file→(role)→participant hierarchy and
    is intentionally treated as a separate reporting sensitivity view.
    """

    scored, score_summary = score_raw_windows(raw_windows, config)
    count = int(scored.values.shape[0])
    if config.policy == "none":
        return scored, score_summary
    scores = np.asarray(scored.window_quality_scores, dtype=np.float32)
    keep = legacy_per_file_top_fraction_mask(
        scores,
        keep_fraction=config.keep_fraction,
    )
    mask_hash = hashlib.sha256(np.asarray(keep, dtype=np.uint8).tobytes()).hexdigest()
    marked = replace(
        scored,
        window_aggregation_mask=keep,
        provenance={
            **dict(scored.provenance),
            "window_quality_aggregation_selection": {
                **config.to_mapping(),
                "input_window_count": count,
                "aggregation_window_count": int(np.count_nonzero(keep)),
                "score_vector_sha256": score_summary["score_vector_sha256"],
                "aggregation_mask_sha256": mask_hash,
            },
        },
    )
    return marked, {
        **config.to_mapping(),
        "input_window_count": count,
        "retained_window_count": count,
        "aggregation_window_count": int(np.count_nonzero(keep)),
        "fitted_on_participant_ids": [],
        "uses_labels": False,
        "cross_file_statistics": False,
        "score_vector_sha256": score_summary["score_vector_sha256"],
        "aggregation_mask_sha256": mask_hash,
    }


__all__ = [
    "LEGACY_WINDOW_SCORE_ALGORITHM",
    "WINDOW_SELECTION_POLICIES",
    "WINDOW_SELECTION_APPLICATION_SCOPES",
    "WindowSelectionConfig",
    "legacy_per_file_top_fraction_mask",
    "legacy_window_sqi_scores",
    "mark_raw_windows_for_aggregation",
    "score_raw_windows",
    "select_raw_windows",
]
