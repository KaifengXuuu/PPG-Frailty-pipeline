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
from html import escape as html_escape
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import yaml
from scipy.stats import t as student_t

from ..artifact import get_reducer
from ..data.external_manifest import (
    M2_EXTERNAL_RELATIVE_PATH,
    PTT_DATASET_ID,
    ExternalRecord,
    adapt_ptt_synchronized_channels,
    load_m2_external_manifest,
)
from ..peaks.pairing import match_events
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
from ..peaks.resolver import (
    CANONICAL_DETECTOR_ID,
    detect_pulses,
    resolve_detector_id,
    resolve_detector_parameters,
)
from ..provenance import sha256_file, stable_payload_sha256
from ..reporting.tabular import (
    ReportTable,
    compact_rows,
    format_interval,
    format_mean_sd,
    html_column_definitions_block,
    markdown_column_definitions_block,
    write_csv,
    write_excel_workbook,
    write_table_column_definitions,
)
from ..reporting.components import (
    TEST_COMPONENT_VIEW_SCHEMAS,
    build_motion_peak_test_component_rows,
    markdown_test_component_table,
    write_test_component_markdown,
)
from ..reporting.profiles import (
    REPORTER_PROFILE_VIEW_SCHEMAS,
    markdown_reporter_profile_tables,
    reporter_profile_rows,
    write_reporter_methods,
)
from ..reporting.classification_diagnostics import (
    ClassificationDiagnosticConfig,
    classification_diagnostic_status_rows,
    classification_per_class_metric_rows,
    classification_roc_curve_rows,
    classification_tsne_rows,
    normalize_classification_rows,
)
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
    _pr_auc_average_precision,
    _roc_auc,
)


STAGE5_SCHEMA = "ppg_frailty.stage5_pre_motion_ptt.v1"
PEAK_ABLATION_SCHEMA = "ppg_frailty.stage_ablation_01_static_peaks.v3"
RESULT_SCHEMA = "ppg_frailty.motion_peak_study_result.v1"

_STAGE5_DETECTOR_FIGURE_MODULES = {
    "motion_detector_metrics",
    "motion_internal_confusion_matrix",
    "motion_internal_file_confusion_matrix",
    "motion_ptt_confusion_matrix",
    "motion_ptt_file_confusion_matrix",
    "motion_ptt_training_oof_confusion_matrix",
    "motion_ptt_training_oof_file_confusion_matrix",
    "motion_internal_reverse_confusion_matrix",
    "motion_internal_reverse_file_confusion_matrix",
    "frailty29_trained_window_score_distribution",
    "frailty29_trained_file_score_distribution",
    "ptt22_trained_window_score_distribution",
    "ptt22_trained_file_score_distribution",
    "frailty29_trained_window_prediction_tsne",
    "frailty29_trained_file_prediction_tsne",
    "ptt22_trained_window_prediction_tsne",
    "ptt22_trained_file_prediction_tsne",
    "frailty29_trained_window_roc_auc_curve",
    "frailty29_trained_file_roc_auc_curve",
    "ptt22_trained_window_roc_auc_curve",
    "ptt22_trained_file_roc_auc_curve",
    "motion_training_learning_curves",
}
_OBSOLETE_STAGE5_REPORT_RELATIVE_PATHS = (
    "tables/motion_detector_subject_confusion.csv",
    "tables/motion_detector_subject_confusion.json",
    "tables/denoiser_beat_f1_red.csv",
    "tables/denoiser_beat_f1_red.json",
    "tables/denoiser_beat_f1_ir.csv",
    "tables/denoiser_beat_f1_ir.json",
    "tables/denoiser_sensitivity_red.csv",
    "tables/denoiser_sensitivity_red.json",
    "tables/denoiser_sensitivity_ir.csv",
    "tables/denoiser_sensitivity_ir.json",
    "tables/denoiser_ppv_red.csv",
    "tables/denoiser_ppv_red.json",
    "tables/denoiser_ppv_ir.csv",
    "tables/denoiser_ppv_ir.json",
    "tables/denoiser_ibi_ppi_rmse_red.csv",
    "tables/denoiser_ibi_ppi_rmse_red.json",
    "tables/denoiser_ibi_ppi_rmse_ir.csv",
    "tables/denoiser_ibi_ppi_rmse_ir.json",
    "figures/motion_internal_subject_confusion_matrix.png",
    "figures/motion_ptt_subject_confusion_matrix.png",
    "figures/motion_ptt_training_oof_subject_confusion_matrix.png",
    "figures/motion_internal_reverse_subject_confusion_matrix.png",
)
_OBSOLETE_STAGE5_SUBJECT_CONFUSION_REPLACEMENTS = {
    "motion_internal_subject_confusion_matrix": (
        "motion_internal_confusion_matrix",
        "motion_internal_file_confusion_matrix",
        "frailty29_outer_oof",
    ),
    "motion_ptt_subject_confusion_matrix": (
        "motion_ptt_confusion_matrix",
        "motion_ptt_file_confusion_matrix",
        "frailty29_trained_to_ptt22",
    ),
    "motion_ptt_training_oof_subject_confusion_matrix": (
        "motion_ptt_training_oof_confusion_matrix",
        "motion_ptt_training_oof_file_confusion_matrix",
        "ptt22_outer_oof",
    ),
    "motion_internal_reverse_subject_confusion_matrix": (
        "motion_internal_reverse_confusion_matrix",
        "motion_internal_reverse_file_confusion_matrix",
        "ptt22_trained_to_frailty29",
    ),
}
_DENOISER_FIGURE_MODULES = {
    "denoiser_interval_rmse",
    "denoiser_beat_f1",
    "denoiser_beat_sensitivity",
    "denoiser_beat_ppv",
    "denoiser_runtime",
}
_PEAK_FIGURE_MODULES = {
    "static_peak_detector_f1",
    "static_peak_detector_sensitivity",
    "static_peak_detector_ppv",
    "static_peak_detector_interval_rmse",
    "static_peak_detector_runtime",
}

# Report-level endpoint IDs map to the recording-row fields produced by the
# detector benchmark.  Keeping this registry beside the reporter makes the
# inferential roster auditable and lets a plan select any supported subset.
_STATIC_PEAK_STATISTICAL_METRICS: Mapping[
    str, tuple[str, str, str, str]
] = {
    "recording_f1_percent": (
        "f1_percent",
        "Beat-detection F1",
        "%",
        "higher",
    ),
    "recording_sensitivity_percent": (
        "sensitivity_percent",
        "Beat-detection sensitivity",
        "%",
        "higher",
    ),
    "recording_positive_predictive_value_percent": (
        "positive_predictive_value_percent",
        "Beat-detection positive predictive value",
        "%",
        "higher",
    ),
    "recording_ibi_ppi_rmse_ms": (
        "ibi_ppi_rmse_ms",
        "IBI–PPI RMSE",
        "ms",
        "lower",
    ),
    "execution_time_percent_of_ppg_signal_duration": (
        "execution_time_percent",
        "Execution time relative to PPG duration",
        "% of signal duration",
        "lower",
    ),
}
_STATIC_PEAK_STATISTICAL_METRIC_IDS = tuple(
    _STATIC_PEAK_STATISTICAL_METRICS
)
_STATIC_PEAK_HOLM_SIDAK_FAMILY = (
    "all_selected_metrics_channels_and_reference_comparators"
)

_MOTION_DETECTOR_METRICS = (
    "balanced_accuracy",
    "macro_f1",
    "sensitivity",
    "specificity",
    "roc_auc",
    "pr_auc",
)
_MOTION_CLUSTER_CI_METRICS = (
    "balanced_accuracy",
    "macro_f1",
    "sensitivity",
    "specificity",
    "roc_auc",
)
_MOTION_CLUSTER_BOOTSTRAP_RESAMPLES = 10_000
_MOTION_CLUSTER_BOOTSTRAP_SEED = 42
_STAGE5_PAIRED_PERMUTATION_RESAMPLES = 100_000
_STAGE5_PAIRED_PERMUTATION_SEED = 42
_STAGE5_INFERENCE_ALPHA = 0.05
_MOTION_DETECTOR_METRIC_LABELS: Mapping[str, str] = {
    "balanced_accuracy": "Balanced accuracy",
    "macro_f1": "Macro-F1",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}
_MOTION_DETECTOR_METRIC_TABLE_IDS: Mapping[str, str] = {
    "balanced_accuracy": "motion_detector_balanced_accuracy",
    "macro_f1": "motion_detector_macro_f1",
    "sensitivity": "motion_detector_sensitivity",
    "specificity": "motion_detector_specificity",
    "roc_auc": "motion_detector_roc_auc",
    "pr_auc": "motion_detector_pr_auc",
}
_MOTION_DETECTOR_RESULT_FIELDS = (
    "model_id",
    "evaluation",
    "level",
    "participant_macro_mean_sd",
    "participant_bootstrap_ci95",
    "holm_p_vs_reference",
)
_DENOISER_PRIMARY_METRICS: Mapping[str, tuple[str, bool, str]] = {
    "participant_macro_f1": ("Beat F1", True, "higher"),
    "participant_macro_sensitivity": ("Sensitivity", True, "higher"),
    "participant_macro_positive_predictive_value": (
        "Positive predictive value",
        True,
        "higher",
    ),
    "participant_macro_ibi_ppi_rmse_ms": (
        "IBI-PPI RMSE",
        False,
        "lower",
    ),
}
_DENOISER_ACTIVITY_RESULT_FIELDS = (
    "denoiser",
    "IR/RED",
    "RMSE ± SD (ms)",
    "F1 ± SD (%)",
    "RMSE P versus identity",
)
_DEFAULT_MOTION_DETECTOR_REFERENCE_ID = "frailty29_trained_motion_detector"
_DEFAULT_MOTION_DETECTOR_CANDIDATE_ID = "ptt22_trained_motion_detector"
_DEFAULT_DENOISER_REFERENCE_ID = "identity"
_MOTION_TARGET_DATASET_BY_EVALUATION_ID = {
    "frailty29_outer_oof": "frailty29",
    "frailty29_trained_to_ptt22": "ptt22",
    "ptt22_outer_oof": "ptt22",
    "ptt22_trained_to_frailty29": "frailty29",
}
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
        scoring_detector = resolve_detector_id(
            str(benchmark.get("scoring_peak_detector", CANONICAL_DETECTOR_ID))
        )
        resolve_detector_parameters(
            scoring_detector,
            benchmark.get("scoring_peak_detector_parameters"),
        )
        if float(dataset.get("source_fs_hz", -1.0)) != 500.0 or float(
            dataset.get("pipeline_fs_hz", -1.0)
        ) != 400.0:
            raise ValueError("Stage5-pre uses the registered PTT 500-to-400-Hz adapter")
        report = data.get("report")
        if not isinstance(report, Mapping):
            raise ValueError("Stage5 report settings must be a mapping")
        if report.get("file_score_aggregation", "median") not in {
            "median",
            "mean",
            "maximum",
        }:
            raise ValueError(
                "Stage5 report file_score_aggregation must be median, mean, or maximum"
            )
        ClassificationDiagnosticConfig(
            tsne_random_state=int(
                report.get("classification_tsne_random_state", 42)
            ),
            tsne_perplexity=float(
                report.get("classification_tsne_perplexity", 30.0)
            ),
            tsne_max_samples=int(
                report.get("classification_tsne_max_samples", 2000)
            ),
            roc_macro_grid_points=int(
                report.get("classification_roc_macro_grid_points", 201)
            ),
            score_histogram_bins=int(
                report.get("classification_score_histogram_bins", 40)
            ),
        )
        for field, supported in (
            ("required_detector_figures", _STAGE5_DETECTOR_FIGURE_MODULES),
            ("denoiser_figures_when_enabled", _DENOISER_FIGURE_MODULES),
        ):
            configured = report.get(field)
            if (
                not isinstance(configured, list)
                or not configured
                or len(configured) != len(set(configured))
                or not set(configured) <= supported
            ):
                raise ValueError(
                    f"Stage5 report {field} must select unique registered modules"
                )
    else:
        if data.get("activities") != ["sit"]:
            raise ValueError("Stage-ablation-01 is a pure-static sit-only experiment")
        algorithms = data.get("algorithms")
        declared = {
            str(item.get("algorithm_id")): item
            for item in algorithms or ()
            if isinstance(item, Mapping)
        }
        if set(declared) != {"aboy_project", MSPTDFAST_V2_ID}:
            raise ValueError(
                "static peak ablation must compare only aboy_project (the "
                "registered v2 module) and default MSPTDfast v2"
            )
        aboy = declared["aboy_project"]
        if (
            aboy.get("module_id") != ABOY_V2_ID
            or aboy.get("implementation") != ABOY_V2_IMPLEMENTATION_PATH
        ):
            raise ValueError(
                "Stage-ablation-01 aboy_project must reference the registered "
                "authoritative Aboy v2 module"
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
        comparator = declared[MSPTDFAST_V2_ID]
        if (
            comparator.get("module_id") != MSPTDFAST_V2_ID
            or comparator.get("implementation") != MSPTDFAST_IMPLEMENTATION_PATH
            or CANONICAL_DETECTOR_ID != MSPTDFAST_V2_ID
        ):
            raise ValueError(
                "Stage-ablation-01 must use the registered default MSPTDfast module"
            )
        resolve_msptdfast_parameters(comparator.get("parameters"))
        validation = data.get("validation")
        if not isinstance(validation, Mapping):
            raise ValueError("static peak ablation requires validation settings")
        _finite(validation.get("beat_tolerance_s"), "beat_tolerance_s", positive=True)
        _finite(validation.get("max_lag_s"), "max_lag_s", positive=True)
        _finite(validation.get("lag_step_s"), "lag_step_s", positive=True)
        _finite(validation.get("lag_window_s"), "lag_window_s", positive=True)
        if (
            float(validation["lag_window_s"]) != 300.0
            or float(validation["beat_tolerance_s"]) != 0.15
            or validation.get("evaluation_unit")
            != "subject_recording_and_wavelength"
            or validation.get("aggregation")
            != "median_and_interquartile_range_across_subject_recordings"
            or validation.get("boxplot_whisker_percentiles") != [10, 90]
            or validation.get("lag_tie_break")
            != "maximum_ncorrect_then_smallest_absolute_lag_then_positive_lag"
        ):
            raise ValueError(
                "static peak assessment must use 300-s lag updates, +/-150-ms "
                "one-to-one matching, and recording-level median/IQR reporting"
            )
        statistical = validation.get("statistical_comparison")
        if not isinstance(statistical, Mapping):
            raise ValueError("static peak statistical-comparison contract drifted")
        selected_metrics = statistical.get("metrics")
        if selected_metrics is None:
            # Backward-compatible reader for the originally persisted v3 plan,
            # which pre-registered only recording F1.
            selected_metrics = [statistical.get("metric")]
        if (
            statistical.get("reference_algorithm_id") != MSPTDFAST_V2_ID
            or statistical.get("test") != "wilcoxon_rank_sum_two_sided"
            or statistical.get("multiple_comparison_correction")
            != "holm_sidak_step_down"
            or float(statistical.get("alpha", -1.0)) != 0.05
            or not isinstance(selected_metrics, list)
            or not selected_metrics
            or len(selected_metrics) != len(set(selected_metrics))
            or not set(selected_metrics) <= set(_STATIC_PEAK_STATISTICAL_METRIC_IDS)
            or statistical.get(
                "family_definition", _STATIC_PEAK_HOLM_SIDAK_FAMILY
            )
            != _STATIC_PEAK_HOLM_SIDAK_FAMILY
        ):
            raise ValueError("static peak statistical-comparison contract drifted")
        report = data.get("report")
        configured = (
            report.get("required_figures") if isinstance(report, Mapping) else None
        )
        if (
            not isinstance(configured, list)
            or not configured
            or len(configured) != len(set(configured))
            or not set(configured) <= _PEAK_FIGURE_MODULES
        ):
            raise ValueError(
                "static peak report required_figures must select unique registered modules"
            )
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
    raw_reference = np.asarray(reference_s, dtype=np.float64)
    raw_predicted = np.asarray(predicted_s, dtype=np.float64)
    reference_order = np.argsort(raw_reference, kind="stable")
    predicted_order = np.argsort(raw_predicted, kind="stable")
    reference = raw_reference[reference_order]
    predicted = raw_predicted[predicted_order]
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
            pairs.append(
                (
                    int(reference_order[reference_index]),
                    int(predicted_order[predicted_index]),
                )
            )
    if not pairs:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return tuple(np.asarray(pairs, dtype=np.int64).T)  # type: ignore[return-value]


def _best_lag_grid(
    reference_s: np.ndarray,
    predicted_s: np.ndarray,
    *,
    max_lag_s: float,
    lag_step_s: float,
    tolerance_s: float,
) -> float:
    """Find the exact best grid lag after a fast per-reference upper bound."""

    reference = np.sort(np.asarray(reference_s, dtype=np.float64))
    predicted = np.sort(np.asarray(predicted_s, dtype=np.float64))
    if reference.size == 0 or predicted.size == 0:
        return 0.0
    lags = np.arange(
        -max_lag_s,
        max_lag_s + lag_step_s * 0.5,
        lag_step_s,
        dtype=np.float64,
    )
    support = np.zeros(lags.size, dtype=np.int64)
    grid_start = float(lags[0])
    for event in reference:
        deltas = predicted - float(event)
        deltas = deltas[
            (deltas >= -max_lag_s - tolerance_s)
            & (deltas <= max_lag_s + tolerance_s)
        ]
        if deltas.size == 0:
            continue
        difference = np.zeros(lags.size + 1, dtype=np.int64)
        lower = np.ceil(
            (deltas - tolerance_s - grid_start) / lag_step_s - 1e-12
        ).astype(np.int64)
        upper = np.floor(
            (deltas + tolerance_s - grid_start) / lag_step_s + 1e-12
        ).astype(np.int64)
        lower = np.clip(lower, 0, lags.size - 1)
        upper = np.clip(upper, 0, lags.size - 1)
        valid = lower <= upper
        np.add.at(difference, lower[valid], 1)
        np.add.at(difference, upper[valid] + 1, -1)
        support += np.cumsum(difference[:-1]) > 0

    order = sorted(
        range(lags.size),
        key=lambda index: (
            -int(support[index]),
            abs(float(lags[index])),
            -float(lags[index]),
        ),
    )
    best_key = (-1, float("-inf"), float("-inf"))
    best_lag = 0.0
    for index in order:
        if int(support[index]) < best_key[0]:
            break
        lag = float(lags[index])
        metrics = match_events(reference + lag, predicted, tolerance_s=tolerance_s)
        key = (
            int(metrics.true_positive),
            -abs(lag),
            lag,
        )
        if key > best_key:
            best_key = key
            best_lag = lag
    return best_lag


def _piecewise_shift_reference(
    reference_s: np.ndarray,
    predicted_s: np.ndarray,
    *,
    max_lag_s: float,
    lag_step_s: float,
    tolerance_s: float,
    lag_window_s: float | None,
    recording_duration_s: float | None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Re-estimate ECG-to-PPG lag in consecutive recording-time windows."""

    reference = np.sort(np.asarray(reference_s, dtype=np.float64))
    predicted = np.sort(np.asarray(predicted_s, dtype=np.float64))
    if lag_window_s is None:
        starts = (0.0,)
        base_duration = max(
            float(recording_duration_s or 0.0),
            float(reference[-1]) if reference.size else 0.0,
            float(predicted[-1]) if predicted.size else 0.0,
        )
        duration = np.nextafter(base_duration, float("inf"))
        window_length = max(duration, np.finfo(np.float64).eps)
    else:
        if not math.isfinite(lag_window_s) or lag_window_s <= 0.0:
            raise ValueError("lag_window_s must be finite and positive")
        base_duration = max(
            float(recording_duration_s or 0.0),
            float(reference[-1]) if reference.size else 0.0,
            float(predicted[-1]) if predicted.size else 0.0,
        )
        duration = np.nextafter(base_duration, float("inf"))
        window_count = max(1, int(math.ceil(base_duration / lag_window_s)))
        starts = tuple(float(index) * lag_window_s for index in range(window_count))
        window_length = float(lag_window_s)

    shifted = reference.copy()
    audit_rows: list[dict[str, Any]] = []
    for window_index, start in enumerate(starts):
        stop = min(float(start) + window_length, duration)
        reference_mask = (reference >= start) & (reference < stop)
        current_reference = reference[reference_mask]
        predicted_mask = (
            (predicted >= start - max_lag_s - tolerance_s)
            & (predicted < stop + max_lag_s + tolerance_s)
        )
        current_predicted = predicted[predicted_mask]
        lag = _best_lag_grid(
            current_reference,
            current_predicted,
            max_lag_s=max_lag_s,
            lag_step_s=lag_step_s,
            tolerance_s=tolerance_s,
        )
        shifted[reference_mask] += lag
        preliminary = match_events(
            current_reference + lag,
            current_predicted,
            tolerance_s=tolerance_s,
        )
        audit_rows.append(
            {
                "lag_window_index": int(window_index),
                "lag_window_start_s": float(start),
                "lag_window_stop_s": float(stop),
                "lag_s": float(lag),
                "reference_beat_count": int(current_reference.size),
                "candidate_ppg_beat_count": int(current_predicted.size),
                "preliminary_correct_count": int(preliminary.true_positive),
            }
        )
    return shifted, audit_rows


def align_and_score_beats(
    reference_s: np.ndarray,
    predicted_s: np.ndarray,
    *,
    max_lag_s: float = 10.0,
    lag_step_s: float = 0.02,
    tolerance_s: float = 0.2,
    lag_window_s: float | None = None,
    recording_duration_s: float | None = None,
) -> dict[str, Any]:
    """One-to-one beat scoring with optional within-record drifting lag."""

    reference = np.sort(np.asarray(reference_s, dtype=np.float64))
    predicted = np.sort(np.asarray(predicted_s, dtype=np.float64))
    shifted, lag_windows = _piecewise_shift_reference(
        reference,
        predicted,
        max_lag_s=max_lag_s,
        lag_step_s=lag_step_s,
        tolerance_s=tolerance_s,
        lag_window_s=lag_window_s,
        recording_duration_s=recording_duration_s,
    )
    metrics = match_events(shifted, predicted, tolerance_s=tolerance_s)
    reference_indices, predicted_indices = _matched_pairs(
        shifted, predicted, tolerance_s=tolerance_s
    )
    chronological = np.argsort(reference_indices, kind="stable")
    reference_indices = reference_indices[chronological]
    predicted_indices = predicted_indices[chronological]
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
    lag_values = [float(row["lag_s"]) for row in lag_windows]
    return {
        "lag_s": float(np.median(lag_values)) if lag_values else 0.0,
        "lag_windows": lag_windows,
        "nref": int(reference.size),
        "nppg": int(predicted.size),
        "ncorrect": int(metrics.true_positive),
        "true_positives": metrics.true_positive,
        "false_positives": metrics.false_positive,
        "false_negatives": metrics.false_negative,
        "sensitivity": metrics.recall,
        "positive_predictive_value": metrics.precision,
        "f1": metrics.f1,
        "sensitivity_percent": 100.0 * metrics.recall,
        "positive_predictive_value_percent": 100.0 * metrics.precision,
        "f1_percent": 100.0 * metrics.f1,
        "beat_tolerance_s": float(tolerance_s),
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
    lag_window_s: float | None = None,
    recording_duration_s: float | None = None,
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
        lag_window_s=lag_window_s,
        recording_duration_s=recording_duration_s,
    )
    return scored, elapsed


def _nonnegative_integer(value: Any, field: str) -> int:
    """Return one exact non-negative count or fail closed."""

    if value is None or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer count")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
        raise ValueError(f"{field} must be an integer count")
    return int(numeric)


def _passed_segment_counts(row: Mapping[str, Any]) -> tuple[int, int, int]:
    """Resolve TP/FP/FN from one passed row and audit redundant totals."""

    tp_source = (
        row.get("true_positives")
        if row.get("true_positives") is not None
        else row.get("ncorrect")
    )
    true_positive = _nonnegative_integer(tp_source, "true_positives/ncorrect")
    if row.get("false_positives") is not None:
        false_positive = _nonnegative_integer(
            row["false_positives"], "false_positives"
        )
    elif row.get("nppg") is not None:
        false_positive = (
            _nonnegative_integer(row["nppg"], "nppg") - true_positive
        )
    else:
        raise ValueError("passed denoiser row lacks false_positives/nppg")
    if row.get("false_negatives") is not None:
        false_negative = _nonnegative_integer(
            row["false_negatives"], "false_negatives"
        )
    elif row.get("nref") is not None:
        false_negative = (
            _nonnegative_integer(row["nref"], "nref") - true_positive
        )
    else:
        raise ValueError("passed denoiser row lacks false_negatives/nref")
    if false_positive < 0 or false_negative < 0:
        raise ValueError("denoiser beat totals cannot be smaller than true positives")
    redundant_totals = (
        ("ncorrect", true_positive),
        ("nppg", true_positive + false_positive),
        ("nref", true_positive + false_negative),
    )
    for field, expected in redundant_totals:
        if row.get(field) is not None and _nonnegative_integer(
            row[field], field
        ) != expected:
            raise ValueError(f"{field} disagrees with TP/FP/FN counts")
    return true_positive, false_positive, false_negative


def _runtime_seconds(row: Mapping[str, Any]) -> float:
    runtime = _finite(row.get("runtime_s"), "runtime_s")
    if runtime < 0.0:
        raise ValueError("runtime_s must be non-negative")
    return runtime


def _aggregate_benchmark(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pool beat counts and interval SSE within subject, then macro-average.

    ``participant_count`` and ``segment_count`` remain compatibility aliases for
    endpoint-evaluable participants and passed segments. Explicit attempted,
    passed, failed and coverage fields make every excluded attempt auditable.
    """

    keys = sorted(
        {
            (
                str(row["algorithm_or_reducer"]),
                str(row["activity_group"]),
                str(row["channel"]),
            )
            for row in rows
        }
    )
    output: list[dict[str, Any]] = []
    for algorithm, activity_group, channel in keys:
        attempted = [
            row for row in rows
            if (
                str(row["algorithm_or_reducer"]),
                str(row["activity_group"]),
                str(row["channel"]),
            )
            == (algorithm, activity_group, channel)
        ]
        if not attempted:
            continue
        if any(
            not str(row.get("participant_id", "")).strip()
            for row in attempted
        ):
            raise ValueError("denoiser benchmark rows require participant_id")
        if any(not str(row.get("status", "")).strip() for row in attempted):
            raise ValueError("denoiser benchmark rows require status")
        passed = [row for row in attempted if row.get("status") == "passed"]
        failed = [row for row in attempted if row.get("status") != "passed"]
        attempted_participants = sorted(
            {str(row["participant_id"]) for row in attempted}
        )
        passed_participants = {
            str(row["participant_id"]) for row in passed
        }
        failed_participants = {
            str(row["participant_id"]) for row in failed
        }
        all_failed_participants = set(attempted_participants) - passed_participants
        participant_rows: list[dict[str, float | None]] = []
        total_matched_interval_count = 0
        total_rmse_evaluable_segment_count = 0
        for participant in sorted(passed_participants):
            current = [
                row
                for row in passed
                if str(row["participant_id"]) == participant
            ]
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
                row_interval_count = _nonnegative_integer(
                    row.get("matched_interval_count"), "matched_interval_count"
                )
                row_rmse = row.get("ibi_ppi_rmse_ms")
                if row_interval_count == 0:
                    if row_rmse is not None:
                        raise ValueError(
                            "ibi_ppi_rmse_ms requires matched_interval_count > 0"
                        )
                else:
                    rmse = _finite(row_rmse, "ibi_ppi_rmse_ms")
                    if rmse < 0.0:
                        raise ValueError("ibi_ppi_rmse_ms must be non-negative")
                    interval_sse_ms2 += rmse * rmse * row_interval_count
                    interval_count += row_interval_count
                    rmse_evaluable_segment_count += 1
            sensitivity_denominator = true_positive + false_negative
            precision_denominator = true_positive + false_positive
            f1_denominator = 2 * true_positive + false_positive + false_negative
            total_matched_interval_count += interval_count
            total_rmse_evaluable_segment_count += rmse_evaluable_segment_count
            participant_rows.append(
                {
                    "f1": (
                        2.0 * true_positive / f1_denominator
                        if f1_denominator
                        else 0.0
                    ),
                    "sensitivity": (
                        true_positive / sensitivity_denominator
                        if sensitivity_denominator
                        else 0.0
                    ),
                    "positive_predictive_value": (
                        true_positive / precision_denominator
                        if precision_denominator
                        else 0.0
                    ),
                    "rmse": (
                        math.sqrt(interval_sse_ms2 / interval_count)
                        if interval_count
                        else None
                    ),
                }
            )
        participant_f1 = [float(row["f1"]) for row in participant_rows]
        participant_sensitivity = [
            float(row["sensitivity"]) for row in participant_rows
        ]
        participant_ppv = [
            float(row["positive_predictive_value"])
            for row in participant_rows
        ]
        participant_rmse = [
            float(row["rmse"])
            for row in participant_rows
            if row["rmse"] is not None
        ]
        passed_runtime = math.fsum(_runtime_seconds(row) for row in passed)
        failed_runtime = math.fsum(_runtime_seconds(row) for row in failed)
        output.append(
            {
                "algorithm_or_reducer": algorithm,
                "activity_group": activity_group,
                "channel": channel,
                "endpoint_aggregation": (
                    "pool_tp_fp_fn_and_interval_sse_within_participant_then_macro"
                ),
                "attempted_participant_count": len(attempted_participants),
                "passed_participant_count": len(passed_participants),
                "failed_participant_count": len(failed_participants),
                "all_failed_participant_count": len(all_failed_participants),
                "partially_failed_participant_count": len(
                    failed_participants & passed_participants
                ),
                "participant_coverage_rate": (
                    len(passed_participants) / len(attempted_participants)
                ),
                "attempted_segment_count": len(attempted),
                "passed_segment_count": len(passed),
                "failed_segment_count": len(failed),
                "segment_coverage_rate": len(passed) / len(attempted),
                "rmse_evaluable_participant_count": len(participant_rmse),
                "rmse_evaluable_segment_count": total_rmse_evaluable_segment_count,
                "matched_interval_count": total_matched_interval_count,
                # Compatibility aliases used by existing report consumers.
                "participant_count": len(participant_rows),
                "segment_count": len(passed),
                "participant_macro_f1": (
                    float(np.mean(participant_f1)) if participant_f1 else None
                ),
                "participant_macro_f1_sd": (
                    float(np.std(participant_f1, ddof=1))
                    if len(participant_f1) >= 2
                    else None
                ),
                "participant_macro_sensitivity": (
                    float(np.mean(participant_sensitivity))
                    if participant_sensitivity
                    else None
                ),
                "participant_macro_sensitivity_sd": (
                    float(np.std(participant_sensitivity, ddof=1))
                    if len(participant_sensitivity) >= 2
                    else None
                ),
                "participant_macro_positive_predictive_value": (
                    float(np.mean(participant_ppv)) if participant_ppv else None
                ),
                "participant_macro_positive_predictive_value_sd": (
                    float(np.std(participant_ppv, ddof=1))
                    if len(participant_ppv) >= 2
                    else None
                ),
                "participant_macro_ibi_ppi_rmse_ms": (
                    float(np.mean(participant_rmse))
                    if participant_rmse
                    else None
                ),
                "participant_macro_ibi_ppi_rmse_ms_sd": (
                    float(np.std(participant_rmse, ddof=1))
                    if len(participant_rmse) >= 2
                    else None
                ),
                "passed_runtime_s": passed_runtime,
                "failed_runtime_s": failed_runtime,
                "total_runtime_s": passed_runtime + failed_runtime,
            }
        )
    return output


def _mean_sample_sd(values: Sequence[float]) -> tuple[float, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    if finite.ndim != 1 or finite.size == 0 or not np.all(np.isfinite(finite)):
        raise ValueError("participant endpoint values must be a finite vector")
    return (
        float(np.mean(finite)),
        float(np.std(finite, ddof=1)) if finite.size >= 2 else None,
    )


def _participant_mean_percentile_ci(
    values: Sequence[float],
    *,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap whole participant endpoint values and return a mean CI95."""

    finite = np.asarray(values, dtype=np.float64)
    if finite.ndim != 1 or finite.size < 2 or not np.all(np.isfinite(finite)):
        raise ValueError("participant bootstrap requires at least two finite values")
    if n_resamples <= 0 or seed < 0:
        raise ValueError("participant bootstrap controls are invalid")
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        int(finite.size),
        np.full(int(finite.size), 1.0 / float(finite.size)),
        size=int(n_resamples),
    )
    draws = (counts @ finite) / float(finite.size)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(low), float(high)


def _paired_participant_sign_flip_p(
    differences: Sequence[float],
    *,
    n_resamples: int,
    seed: int,
) -> float:
    """Two-sided Monte-Carlo sign-flip P with participant as exchange unit."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("paired permutation requires at least two finite differences")
    if n_resamples <= 0 or seed < 0:
        raise ValueError("paired permutation controls are invalid")
    observed = abs(float(np.mean(values)))
    if np.allclose(values, 0.0, rtol=0.0, atol=1e-15):
        return 1.0
    rng = np.random.default_rng(seed)
    extreme = 0
    processed = 0
    while processed < n_resamples:
        current = min(10_000, n_resamples - processed)
        signs = rng.integers(
            0,
            2,
            size=(current, int(values.size)),
            dtype=np.int8,
        ).astype(np.float64)
        signs = 2.0 * signs - 1.0
        draws = np.mean(signs * values[None, :], axis=1)
        extreme += int(np.count_nonzero(np.abs(draws) >= observed - 1e-15))
        processed += current
    return float((extreme + 1) / (n_resamples + 1))


def _holm_adjusted_p_values(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return standard Holm (1979) adjusted P values for one declared family."""

    if not p_values:
        return {}
    ordered = sorted((float(value), str(key)) for key, value in p_values.items())
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value, _ in ordered):
        raise ValueError("Holm input P values must be finite in [0, 1]")
    family_size = len(ordered)
    running = 0.0
    output: dict[str, float] = {}
    for rank, (raw, key) in enumerate(ordered, start=1):
        running = max(running, (family_size - rank + 1) * raw)
        output[key] = float(min(1.0, running))
    return output


def _format_probability(value: float | None) -> str:
    if value is None:
        return "N/A"
    if value < 0.0001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _distribution_summary(values: Sequence[float]) -> dict[str, float | None]:
    finite = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {
            "median": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "p10": None,
            "p90": None,
        }
    q10, q25, q50, q75, q90 = np.percentile(finite, [10, 25, 50, 75, 90])
    return {
        "median": float(q50),
        "q1": float(q25),
        "q3": float(q75),
        "iqr": float(q75 - q25),
        "p10": float(q10),
        "p90": float(q90),
    }


def _aggregate_static_peak_benchmark(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize subject recordings by median/IQR with 10th/90th whiskers."""

    metrics = {
        "recording_f1_percent": "f1_percent",
        "recording_sensitivity_percent": "sensitivity_percent",
        "recording_positive_predictive_value_percent":
            "positive_predictive_value_percent",
        "recording_ibi_ppi_rmse_ms": "ibi_ppi_rmse_ms",
        "execution_time_percent": "execution_time_percent",
    }
    keys = sorted(
        {
            (str(row["algorithm_or_reducer"]), str(row["channel"]))
            for row in rows
            if row.get("status") == "passed"
        }
    )
    output: list[dict[str, Any]] = []
    for algorithm, channel in keys:
        selected = [
            row
            for row in rows
            if row.get("status") == "passed"
            and str(row["algorithm_or_reducer"]) == algorithm
            and str(row["channel"]) == channel
        ]
        summary: dict[str, Any] = {
            "algorithm_or_reducer": algorithm,
            "activity_group": "static",
            "channel": channel,
            "participant_count": len(
                {str(row["participant_id"]) for row in selected}
            ),
            "subject_recording_count": len(
                {
                    (str(row["participant_id"]), str(row["record_id"]))
                    for row in selected
                }
            ),
            "aggregation": "median_iqr_across_subject_recordings",
            "boxplot_whiskers": "p10_p90",
        }
        for output_name, source_name in metrics.items():
            distribution = _distribution_summary(
                [
                    float(row[source_name])
                    for row in selected
                    if row.get(source_name) is not None
                ]
            )
            for statistic, value in distribution.items():
                summary[f"{output_name}_{statistic}"] = value
        output.append(summary)
    return output


def _median_iqr_text(
    row: Mapping[str, Any], metric: str, *, decimals: int = 1
) -> str:
    median = row.get(f"{metric}_median")
    q1 = row.get(f"{metric}_q1")
    q3 = row.get(f"{metric}_q3")
    if median is None or q1 is None or q3 is None:
        return "N/A"
    return (
        f"{float(median):.{decimals}f} "
        f"[{float(q1):.{decimals}f}, {float(q3):.{decimals}f}]"
    )


def _static_peak_display_rows(
    summary_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the report table compact; raw CSV retains every recording value."""

    return [
        {
            "algorithm": row["algorithm_or_reducer"],
            "channel": row["channel"],
            "subject_recordings": row["subject_recording_count"],
            "F1 %, median [Q1, Q3]": _median_iqr_text(
                row, "recording_f1_percent"
            ),
            "Sensitivity %, median [Q1, Q3]": _median_iqr_text(
                row, "recording_sensitivity_percent"
            ),
            "PPV %, median [Q1, Q3]": _median_iqr_text(
                row, "recording_positive_predictive_value_percent"
            ),
            "IBI-PPI RMSE ms, median [Q1, Q3]": _median_iqr_text(
                row, "recording_ibi_ppi_rmse_ms"
            ),
            "Execution time %, median [Q1, Q3]": _median_iqr_text(
                row, "execution_time_percent", decimals=3
            ),
        }
        for row in summary_rows
    ]


def _static_peak_statistical_display_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project the lossless inferential table into a compact report table."""

    def number(value: object, *, decimals: int = 3) -> str:
        if value is None or value == "":
            return "N/A"
        return f"{float(value):.{decimals}f}"

    def probability(value: object) -> str:
        if value is None or value == "":
            return "N/A"
        return f"{float(value):.6g}"

    output: list[dict[str, Any]] = []
    for row in rows:
        decimals = (
            6
            if row.get("metric")
            == "execution_time_percent_of_ppg_signal_duration"
            else 3
        )
        output.append(
            {
                "endpoint": row.get("metric_label", row.get("metric")),
                "channel": row.get("channel"),
                "registration": row.get("analysis_registration"),
                "common subject-recordings": row.get(
                    "common_subject_recordings"
                ),
                "MSPTDfast median": number(
                    row.get("reference_median"), decimals=decimals
                ),
                "aboy_project median": number(
                    row.get("comparator_median"), decimals=decimals
                ),
                "unit": row.get("metric_unit"),
                "MSPTDfast advantage": number(
                    row.get("reference_advantage"), decimals=decimals
                ),
                "rank-sum z": number(row.get("rank_sum_z"), decimals=3),
                "raw p": probability(row.get("p_value")),
                "Holm–Sidak adjusted p (global family)": probability(
                    row.get("holm_sidak_adjusted_p")
                ),
                "reject at alpha=0.05": row.get("reject_at_alpha"),
            }
        )
    return output


def _holm_sidak_step_down(
    p_values: Sequence[float], *, alpha: float
) -> tuple[list[float], list[bool], list[int]]:
    """Return monotone Holm-Sidak adjusted p-values and step-down decisions."""

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
        candidate = 1.0 - (1.0 - raw) ** remaining
        running_adjusted = max(running_adjusted, candidate)
        adjusted[index] = min(running_adjusted, 1.0)
        critical = 1.0 - (1.0 - float(alpha)) ** (1.0 / remaining)
        rejected[index] = bool(rejection_open and raw <= critical)
        if not rejected[index]:
            rejection_open = False
        ranks[index] = position + 1
    return adjusted, rejected, ranks


def _static_peak_rank_sum_comparisons(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_algorithm_id: str,
    alpha: float,
    metric_ids: Sequence[str] = _STATIC_PEAK_STATISTICAL_METRIC_IDS,
    registered_metric_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Compare recording endpoints and apply one declared Holm-Sidak family.

    The historical benchmark contract calls for the two-sample Wilcoxon
    rank-sum test.  Rows are intersected by subject-recording to ensure the two
    detector samples cover identical records, but ``scipy.stats.ranksums`` does
    not use that pairing.  The report records this limitation explicitly.
    """

    from scipy.stats import ranksums

    selected_metric_ids = tuple(str(value) for value in metric_ids)
    if (
        not selected_metric_ids
        or len(selected_metric_ids) != len(set(selected_metric_ids))
        or not set(selected_metric_ids) <= set(_STATIC_PEAK_STATISTICAL_METRICS)
    ):
        raise ValueError("static peak rank-sum metric roster is invalid")
    registered = {
        str(value)
        for value in (
            selected_metric_ids
            if registered_metric_ids is None
            else registered_metric_ids
        )
    }
    if not registered <= set(selected_metric_ids):
        raise ValueError("registered metrics must be a subset of selected metrics")
    passed = [row for row in rows if row.get("status") == "passed"]
    algorithms = sorted(
        {
            str(row["algorithm_or_reducer"])
            for row in passed
            if str(row["algorithm_or_reducer"]) != reference_algorithm_id
        }
    )
    output: list[dict[str, Any]] = []
    for metric_id in selected_metric_ids:
        row_field, metric_label, metric_unit, better_direction = (
            _STATIC_PEAK_STATISTICAL_METRICS[metric_id]
        )
        for channel in _CHANNELS:
            reference_by_record = {
                (str(row["participant_id"]), str(row["record_id"])): float(
                    row[row_field]
                )
                for row in passed
                if str(row["algorithm_or_reducer"]) == reference_algorithm_id
                and str(row["channel"]) == channel
                and row.get(row_field) is not None
                and math.isfinite(float(row[row_field]))
            }
            for comparator in algorithms:
                comparator_by_record = {
                    (str(row["participant_id"]), str(row["record_id"])): float(
                        row[row_field]
                    )
                    for row in passed
                    if str(row["algorithm_or_reducer"]) == comparator
                    and str(row["channel"]) == channel
                    and row.get(row_field) is not None
                    and math.isfinite(float(row[row_field]))
                }
                common = sorted(
                    set(reference_by_record) & set(comparator_by_record)
                )
                reference_values = [reference_by_record[key] for key in common]
                comparator_values = [comparator_by_record[key] for key in common]
                if reference_values and comparator_values:
                    statistic, p_value = ranksums(
                        reference_values,
                        comparator_values,
                        alternative="two-sided",
                    )
                    reference_median = float(np.median(reference_values))
                    comparator_median = float(np.median(comparator_values))
                    reference_advantage = (
                        reference_median - comparator_median
                        if better_direction == "higher"
                        else comparator_median - reference_median
                    )
                else:
                    statistic, p_value = float("nan"), float("nan")
                    reference_median = comparator_median = None
                    reference_advantage = None
                output.append(
                    {
                        "reference_algorithm": reference_algorithm_id,
                        "comparator_algorithm": comparator,
                        "channel": channel,
                        "metric": metric_id,
                        "metric_label": metric_label,
                        "metric_unit": metric_unit,
                        "better_direction": better_direction,
                        "analysis_registration": (
                            "prespecified_in_resolved_plan"
                            if metric_id in registered
                            else "retrospective_supplement_requested_2026-08-24"
                        ),
                        "test": "wilcoxon_rank_sum_two_sided",
                        "scipy_function": "scipy.stats.ranksums",
                        "alternative": "two-sided",
                        "common_subject_recordings": len(common),
                        "identical_record_roster_enforced": True,
                        "pairing_used_by_test": False,
                        "ties_present": len(
                            set(reference_values + comparator_values)
                        ) < len(reference_values + comparator_values),
                        "reference_median": reference_median,
                        "comparator_median": comparator_median,
                        "reference_advantage": reference_advantage,
                        "reference_advantage_definition": (
                            "reference_minus_comparator"
                            if better_direction == "higher"
                            else "comparator_minus_reference"
                        ),
                        "rank_sum_z": (
                            float(statistic)
                            if math.isfinite(float(statistic))
                            else None
                        ),
                        "p_value": (
                            float(p_value)
                            if math.isfinite(float(p_value))
                            else None
                        ),
                    }
                )
    valid_indices = [
        index for index, row in enumerate(output) if row["p_value"] is not None
    ]
    adjusted, rejected, ranks = _holm_sidak_step_down(
        [float(output[index]["p_value"]) for index in valid_indices],
        alpha=alpha,
    )
    for local_index, output_index in enumerate(valid_indices):
        output[output_index].update(
            {
                "holm_sidak_family_size": len(valid_indices),
                "holm_sidak_family_definition": _STATIC_PEAK_HOLM_SIDAK_FAMILY,
                "holm_sidak_rank": ranks[local_index],
                "holm_sidak_adjusted_p": adjusted[local_index],
                "reject_at_alpha": rejected[local_index],
                "alpha": float(alpha),
            }
        )

    # Preserve the originally registered family result as an audit field when
    # a historical plan registered fewer endpoints than the unified reanalysis.
    registered_indices = [
        index
        for index in valid_indices
        if str(output[index]["metric"]) in registered
    ]
    if registered_indices:
        registered_adjusted, registered_rejected, registered_ranks = (
            _holm_sidak_step_down(
                [float(output[index]["p_value"]) for index in registered_indices],
                alpha=alpha,
            )
        )
        for local_index, output_index in enumerate(registered_indices):
            output[output_index].update(
                {
                    "registered_family_size": len(registered_indices),
                    "registered_family_holm_sidak_rank": registered_ranks[
                        local_index
                    ],
                    "registered_family_holm_sidak_adjusted_p": (
                        registered_adjusted[local_index]
                    ),
                    "registered_family_reject_at_alpha": registered_rejected[
                        local_index
                    ],
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
    scoring_peak_detector: str = CANONICAL_DETECTOR_ID,
    scoring_peak_detector_parameters: Mapping[str, Any] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Run implemented reducers on all declared PTT records, sequentially."""

    repository = Path(repository_root).resolve()
    detector_id = resolve_detector_id(scoring_peak_detector)
    detector_parameters = resolve_detector_parameters(
        detector_id, scoring_peak_detector_parameters
    )
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
                            algorithm_id=detector_id,
                            algorithm_parameters=detector_parameters,
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
        "schema_version": "ppg_frailty.stage5_pre_denoiser_benchmark.v2",
        "status": "passed",
        "participant_count": len(by_subject),
        "record_count": len(records),
        "activities": list(activities),
        "segment_s": float(segment_s),
        "reducers": list(reducer_ids),
        "scoring_peak_detector": detector_id,
        "scoring_peak_detector_parameters": detector_parameters,
        "validation": dict(validation),
        "summary_aggregation": (
            "pool_tp_fp_fn_and_interval_sse_within_participant_then_macro"
        ),
        "rows": rows,
        "summary_rows": _aggregate_benchmark(rows),
    }


def run_static_peak_ablation(
    repository_root: str | Path,
    plan: StudyPlan,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Run two registered detectors once per complete PTT sit recording."""

    if plan.schema_version != PEAK_ABLATION_SCHEMA:
        raise ValueError("static peak runner received a non-ablation plan")
    repository = Path(repository_root).resolve()
    records = [row for row in _ptt_records(repository) if row.activity_raw.lower() == "sit"]
    validation = dict(plan.payload["validation"])
    algorithm_specs = [
        (
            str(item["algorithm_id"]),
            str(item["module_id"]),
            dict(item.get("parameters", {})),
        )
        for item in plan.payload["algorithms"]
    ]
    rows: list[dict[str, Any]] = []
    lag_window_s = float(validation["lag_window_s"])
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
        recording_duration_s = native.shape[0] / 400.0
        for algorithm_id, module_id, parameters in algorithm_specs:
            for channel in _CHANNELS:
                scored, elapsed = _score_segment(
                    views,
                    reference,
                    algorithm_id=module_id,
                    algorithm_parameters=parameters,
                    fs_hz=400.0,
                    validation=validation,
                    wavelength=channel,
                    lag_window_s=lag_window_s,
                    recording_duration_s=recording_duration_s,
                )
                rows.append({
                    "participant_id": row.subject_id,
                    "record_id": row.record_id,
                    "activity": "sit",
                    "activity_group": "static",
                    "channel": channel,
                    "algorithm_or_reducer": algorithm_id,
                    "detector_module_id": module_id,
                    "evaluation_unit": "subject_recording_and_wavelength",
                    "recording_duration_s": recording_duration_s,
                    "status": "passed",
                    "runtime_s": elapsed,
                    "runtime_fraction_of_signal": elapsed / recording_duration_s,
                    "execution_time_percent": 100.0 * elapsed / recording_duration_s,
                    **scored,
                })
    if progress_callback is not None:
        progress_callback(len(records), len(records), "completed static peak ablation")
    summary_rows = _aggregate_static_peak_benchmark(rows)
    statistical = dict(validation["statistical_comparison"])
    configured_metric_ids = statistical.get("metrics")
    if configured_metric_ids is None:
        configured_metric_ids = [statistical["metric"]]
    comparisons = _static_peak_rank_sum_comparisons(
        rows,
        reference_algorithm_id=str(statistical["reference_algorithm_id"]),
        alpha=float(statistical["alpha"]),
        metric_ids=[str(value) for value in configured_metric_ids],
        registered_metric_ids=[str(value) for value in configured_metric_ids],
    )
    return {
        "schema_version": "ppg_frailty.stage_ablation_01_static_peak_result.v3",
        "status": "passed",
        "participant_count": len({row.subject_id for row in records}),
        "record_count": len(records),
        "activities": ["sit"],
        "rows": rows,
        "summary_rows": summary_rows,
        "statistical_comparisons": comparisons,
        "paper_source": dict(plan.payload["paper_source"]),
        "validation": validation,
        "execution_environment": {
            "platform": platform.platform(),
            "processor": platform.processor() or "not_reported_by_platform",
            "python_version": platform.python_version(),
            "timer": "time.perf_counter_wall_time",
            "parallelization": "runner_sequential_no_explicit_parallelism",
            "paper_hardware_claim_applied_to_this_run": False,
        },
    }


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("motion report requires pyarrow") from exc
    return [dict(row) for row in pq.read_table(path).to_pylist()]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_csv(path, compact_rows(rows))


_REPORT_MODEL_FIELD_PRIORITY = (
    "model_name",
    "model_id",
    "module_id",
    "algorithm_or_reducer",
    "classifier_id",
    "candidate_id",
    "reference_algorithm",
    "leading_or_selected_case",
)


def _ordered_report_fields(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Keep source order but force an available model identity into column 1."""

    fields = list(dict.fromkeys(str(key) for row in rows for key in row))
    model_field = next(
        (field for field in _REPORT_MODEL_FIELD_PRIORITY if field in fields),
        None,
    )
    if model_field is None:
        return fields
    return [model_field, *(field for field in fields if field != model_field)]


def _report_cell(value: Any) -> str:
    """Render unavailable report values explicitly instead of showing ``None``."""

    return "N/A" if value is None else str(value)


def _markdown_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str] | None = None,
) -> str:
    if not rows:
        return "N/A"
    displayed = (
        [dict(row) for row in rows]
        if any(
            isinstance(value, str) and "*" in value
            for row in rows
            for value in row.values()
        )
        else compact_rows(rows)
    )
    selected_fields = (
        list(fields) if fields is not None else _ordered_report_fields(displayed)
    )
    if not selected_fields:
        raise ValueError("report table requires at least one column")
    if len(selected_fields) > 8:
        raise ValueError(
            "human-facing Stage5 report tables may contain at most eight columns; "
            "split the evidence into semantic subtables"
        )
    body = [
        "| " + " | ".join(selected_fields) + " |",
        "| " + " | ".join("---" for _ in selected_fields) + " |",
    ]
    body.extend(
        "| "
        + " | ".join(
            _report_cell(row.get(field, "")) for field in selected_fields
        )
        + " |"
        for row in displayed
    )
    body.extend(("", markdown_column_definitions_block(selected_fields)))
    return "\n".join(body)


def _html_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str] | None = None,
) -> str:
    import html

    if not rows:
        return "<p>N/A</p>"
    displayed = (
        [dict(row) for row in rows]
        if any(
            isinstance(value, str) and "*" in value
            for row in rows
            for value in row.values()
        )
        else compact_rows(rows)
    )
    selected_fields = (
        list(fields) if fields is not None else _ordered_report_fields(displayed)
    )
    if not selected_fields:
        raise ValueError("report table requires at least one column")
    if len(selected_fields) > 8:
        raise ValueError(
            "human-facing Stage5 report tables may contain at most eight columns; "
            "split the evidence into semantic subtables"
        )
    heading = "".join(
        f"<th>{html.escape(field)}</th>" for field in selected_fields
    )
    body = "".join(
        "<tr>" + "".join(
            f"<td>{html.escape(_report_cell(row.get(field, '')))}</td>"
            for field in selected_fields
        ) + "</tr>"
        for row in displayed
    )
    return (
        f"<table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table>"
        + html_column_definitions_block(selected_fields)
    )


def _motion_target_dataset(evaluation_id: str) -> str:
    """Resolve the target dataset without conflating it with training origin."""

    value = str(evaluation_id).strip()
    if not value:
        raise ValueError("motion detector evaluation id must be non-empty")
    return _MOTION_TARGET_DATASET_BY_EVALUATION_ID.get(value, value)


def _motion_repeat_uncertainty(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return repeat SD/t-CI or an explicit N/A contract for one endpoint."""

    output: dict[str, Any] = {
        "repeat_count": None,
        "repeat_uncertainty_applicability": "N/A",
        "repeat_uncertainty_reason": "not_computed",
        "repeat_sd_estimator": "sample_sd_ddof1",
        "repeat_ci95_method": "two_sided_student_t_0.95",
    }
    for metric in _MOTION_DETECTOR_METRICS:
        output[f"{metric}_repeat_sample_sd"] = None
        output[f"{metric}_repeat_t_ci95_low"] = None
        output[f"{metric}_repeat_t_ci95_high"] = None

    persisted = [row.get("repeat_index") for row in rows]
    if all(value is None for value in persisted):
        output["repeat_count"] = 1
        output["repeat_uncertainty_reason"] = (
            "single_frozen_target_evaluation_without_repeat_axis"
        )
        return output
    if any(value is None for value in persisted):
        output["repeat_uncertainty_reason"] = (
            "mixed_repeat_provenance_not_safe_for_repeat_uncertainty"
        )
        return output

    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["repeat_index"]), []).append(row)
    output["repeat_count"] = len(grouped)
    if len(grouped) < 2:
        output["repeat_uncertainty_reason"] = (
            "single_participant_grouped_cv_repeat"
        )
        return output
    if any(
        {int(row["activity_label"]) for row in selected} != {0, 1}
        for selected in grouped.values()
    ):
        output["repeat_uncertainty_reason"] = (
            "not_computed_because_at_least_one_repeat_lacks_a_protocol_class"
        )
        return output

    repeat_metrics = [
        _detector_level_metrics(grouped[index]) for index in sorted(grouped)
    ]
    output["repeat_uncertainty_applicability"] = "available"
    output["repeat_uncertainty_reason"] = ""
    for metric in _MOTION_DETECTOR_METRICS:
        values = np.asarray(
            [float(row[metric]) for row in repeat_metrics], dtype=np.float64
        )
        sample_sd = float(np.std(values, ddof=1))
        margin = float(
            student_t.ppf(0.975, df=values.size - 1)
            * sample_sd
            / math.sqrt(values.size)
        )
        mean = float(np.mean(values))
        output[f"{metric}_repeat_sample_sd"] = sample_sd
        output[f"{metric}_repeat_t_ci95_low"] = mean - margin
        output[f"{metric}_repeat_t_ci95_high"] = mean + margin
    return output


def _motion_participant_cluster_uncertainty(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_resamples: int = _MOTION_CLUSTER_BOOTSTRAP_RESAMPLES,
    seed: int = _MOTION_CLUSTER_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap participants while carrying every window/file in each cluster.

    Thresholded metrics and ROC-AUC use the same participant multiplicities.
    ROC-AUC is evaluated as the weighted Mann-Whitney statistic, including
    half credit for tied scores. PR-AUC remains point-estimate only because a
    weighted participant-cluster AP interval is not registered here.
    """

    output: dict[str, Any] = {
        "participant_cluster_ci95_applicability": "N/A",
        "participant_cluster_ci95_reason": "not_computed",
        "participant_cluster_ci95_method": (
            "participant_cluster_percentile_bootstrap_two_sided_95"
        ),
        "participant_cluster_ci95_cluster_unit": (
            "participant_with_all_rows_in_exact_target_scope_level"
        ),
        "participant_cluster_bootstrap_resamples": int(n_resamples),
        "participant_cluster_bootstrap_seed": int(seed),
        "participant_cluster_count": 0,
        "participant_cluster_ci95_metrics": ",".join(
            _MOTION_CLUSTER_CI_METRICS
        ),
        "roc_pr_participant_cluster_ci95_applicability": "N/A",
        "roc_pr_participant_cluster_ci95_reason": (
            "combined_ROC_PR_interval_not_reported_ROC_has_separate_cluster_"
            "CI_and_weighted_cluster_PR_AUC_is_not_registered"
        ),
        "roc_auc_participant_cluster_ci95_applicability": "N/A",
        "roc_auc_participant_cluster_ci95_reason": "not_computed",
        "pr_auc_participant_cluster_ci95_applicability": "N/A",
        "pr_auc_participant_cluster_ci95_reason": (
            "weighted_participant_cluster_average_precision_interval_not_"
            "registered"
        ),
    }
    for metric in _MOTION_CLUSTER_CI_METRICS:
        output[f"{metric}_participant_cluster_ci95_low"] = None
        output[f"{metric}_participant_cluster_ci95_high"] = None

    if n_resamples <= 0:
        raise ValueError("participant-cluster bootstrap resamples must be positive")
    participant_ids = [str(row.get("participant_id", "")).strip() for row in rows]
    if not participant_ids or any(not value for value in participant_ids):
        output["participant_cluster_ci95_reason"] = (
            "not_computed_because_participant_id_is_missing"
        )
        return output
    participants = sorted(set(participant_ids))
    output["participant_cluster_count"] = len(participants)
    if len(participants) < 2:
        output["participant_cluster_ci95_reason"] = (
            "not_computed_because_fewer_than_two_participant_clusters"
        )
        return output

    participant_index = {value: index for index, value in enumerate(participants)}
    confusion_by_participant = np.zeros((len(participants), 4), dtype=np.int64)
    row_participant_index = np.asarray(
        [participant_index[value] for value in participant_ids], dtype=np.int64
    )
    row_labels = np.asarray(
        [int(row["activity_label"]) for row in rows], dtype=np.int64
    )
    row_scores = np.asarray(
        [float(row["p_active"]) for row in rows], dtype=np.float64
    )
    if not np.all(np.isfinite(row_scores)):
        output["participant_cluster_ci95_reason"] = (
            "not_computed_because_nonfinite_probability_was_persisted"
        )
        return output
    labels_by_participant: dict[str, set[int]] = {
        value: set() for value in participants
    }
    for row, participant_id in zip(rows, participant_ids, strict=True):
        label = int(row["activity_label"])
        predicted = int(row["predicted_activity"])
        if label not in {0, 1} or predicted not in {0, 1}:
            output["participant_cluster_ci95_reason"] = (
                "not_computed_because_nonbinary_prediction_was_persisted"
            )
            return output
        labels_by_participant[participant_id].add(label)
        column = (
            0 if (label, predicted) == (0, 0)
            else 1 if (label, predicted) == (0, 1)
            else 2 if (label, predicted) == (1, 0)
            else 3
        )
        confusion_by_participant[participant_index[participant_id], column] += 1
    if any(labels != {0, 1} for labels in labels_by_participant.values()):
        output["participant_cluster_ci95_reason"] = (
            "not_computed_because_at_least_one_participant_cluster_lacks_"
            "a_protocol_class"
        )
        return output

    rng = np.random.default_rng(seed)
    multiplicities = rng.multinomial(
        len(participants),
        np.full(len(participants), 1.0 / len(participants), dtype=np.float64),
        size=int(n_resamples),
    )
    confusion = multiplicities @ confusion_by_participant
    true_static, false_motion, false_static, true_motion = (
        confusion[:, index].astype(np.float64) for index in range(4)
    )
    specificity = true_static / (true_static + false_motion)
    sensitivity = true_motion / (true_motion + false_static)
    balanced_accuracy = 0.5 * (specificity + sensitivity)
    static_precision = np.divide(
        true_static,
        true_static + false_static,
        out=np.zeros_like(true_static),
        where=(true_static + false_static) > 0.0,
    )
    motion_precision = np.divide(
        true_motion,
        true_motion + false_motion,
        out=np.zeros_like(true_motion),
        where=(true_motion + false_motion) > 0.0,
    )
    static_f1 = np.divide(
        2.0 * static_precision * specificity,
        static_precision + specificity,
        out=np.zeros_like(static_precision),
        where=(static_precision + specificity) > 0.0,
    )
    motion_f1 = np.divide(
        2.0 * motion_precision * sensitivity,
        motion_precision + sensitivity,
        out=np.zeros_like(motion_precision),
        where=(motion_precision + sensitivity) > 0.0,
    )
    macro_f1 = 0.5 * (static_f1 + motion_f1)

    # For fixed scores, binary ROC-AUC is the Mann-Whitney probability that a
    # positive score exceeds a negative score (ties contribute 0.5). Cluster
    # multiplicities make its numerator m.T @ pair_credit @ m and retain exact
    # participant resampling without materialising every duplicated row.
    positive_counts = np.bincount(
        row_participant_index[row_labels == 1], minlength=len(participants)
    ).astype(np.float64)
    negative_counts = np.bincount(
        row_participant_index[row_labels == 0], minlength=len(participants)
    ).astype(np.float64)
    pair_credit = np.zeros(
        (len(participants), len(participants)), dtype=np.float64
    )
    negative_scores_by_participant = [
        np.sort(
            row_scores[
                (row_labels == 0) & (row_participant_index == participant_position)
            ]
        )
        for participant_position in range(len(participants))
    ]
    for positive_participant in range(len(participants)):
        positive_scores = row_scores[
            (row_labels == 1)
            & (row_participant_index == positive_participant)
        ]
        for negative_participant, negative_scores in enumerate(
            negative_scores_by_participant
        ):
            below = np.searchsorted(
                negative_scores, positive_scores, side="left"
            )
            at_or_below = np.searchsorted(
                negative_scores, positive_scores, side="right"
            )
            pair_credit[positive_participant, negative_participant] = float(
                np.sum(below + 0.5 * (at_or_below - below))
            )
    roc_numerator = np.einsum(
        "bi,ij,bj->b",
        multiplicities,
        pair_credit,
        multiplicities,
        optimize=True,
    )
    roc_denominator = (
        multiplicities @ positive_counts
    ) * (multiplicities @ negative_counts)
    roc_auc = np.divide(
        roc_numerator,
        roc_denominator,
        out=np.full_like(roc_numerator, np.nan, dtype=np.float64),
        where=roc_denominator > 0.0,
    )
    draws = {
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "roc_auc": roc_auc,
    }
    if any(not np.all(np.isfinite(values)) for values in draws.values()):
        output["participant_cluster_ci95_reason"] = (
            "not_computed_because_bootstrap_produced_nonfinite_metrics"
        )
        return output
    for metric, values in draws.items():
        low, high = np.quantile(values, (0.025, 0.975))
        output[f"{metric}_participant_cluster_ci95_low"] = float(low)
        output[f"{metric}_participant_cluster_ci95_high"] = float(high)
    output["participant_cluster_ci95_applicability"] = "available"
    output["participant_cluster_ci95_reason"] = ""
    output["roc_auc_participant_cluster_ci95_applicability"] = "available"
    output["roc_auc_participant_cluster_ci95_reason"] = ""
    return output


def _detector_report_rows(
    datasets: Sequence[
        tuple[str, str, str, Sequence[Mapping[str, Any]]]
    ],
    *,
    file_score_aggregation: str = "median",
    participant_cluster_bootstrap_resamples: int = (
        _MOTION_CLUSTER_BOOTSTRAP_RESAMPLES
    ),
    participant_cluster_bootstrap_seed: int = _MOTION_CLUSTER_BOOTSTRAP_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recompute window- and file-level metrics from persisted predictions."""

    metrics_output: list[dict[str, Any]] = []
    file_predictions: list[dict[str, Any]] = []
    for model_id, dataset, evaluation_scope, window_rows in datasets:
        dataset_file_rows = _file_prediction_rows(
            window_rows, score_aggregation=file_score_aggregation
        )
        file_predictions.extend(
            {
                "model_id": model_id,
                "dataset": dataset,
                "evaluation_scope": evaluation_scope,
                **row,
            }
            for row in dataset_file_rows
        )
        for aggregation_level, selected in (
            ("window", window_rows),
            ("file", dataset_file_rows),
        ):
            metrics = _detector_level_metrics(selected)
            worst_fold = _worst_fold_balanced_accuracy(selected)
            metrics_output.append(
                {
                    "model_id": model_id,
                    "dataset": dataset,
                    "target_dataset": _motion_target_dataset(dataset),
                    "evaluation_scope": evaluation_scope,
                    "aggregation_level": aggregation_level,
                    "file_score_aggregation": (
                        file_score_aggregation
                        if aggregation_level == "file"
                        else "not_applicable"
                    ),
                    "observation_count": len(selected),
                    "participant_count": len(
                        {str(row["participant_id"]) for row in selected}
                    ),
                    "file_count": len(
                        {
                            (str(row["participant_id"]), str(row["file_id"]))
                            for row in selected
                        }
                    ),
                    "window_count": (
                        len(selected)
                        if aggregation_level == "window"
                        else sum(int(row["window_count"]) for row in selected)
                    ),
                    **metrics,
                    "metric_applicability": (
                        "available_two_class_protocol_activity_state_endpoint"
                    ),
                    "worst_fold_balanced_accuracy": worst_fold,
                    "worst_fold_balanced_accuracy_applicability": (
                        "available" if worst_fold is not None else "N/A"
                    ),
                    "worst_fold_balanced_accuracy_reason": (
                        ""
                        if worst_fold is not None
                        else "frozen_target_evaluation_has_no_training_fold_axis"
                    ),
                    **_motion_repeat_uncertainty(selected),
                    **_motion_participant_cluster_uncertainty(
                        selected,
                        n_resamples=participant_cluster_bootstrap_resamples,
                        seed=participant_cluster_bootstrap_seed,
                    ),
                    "paired_inference_applicability": "N/A",
                    "paired_inference_reason": (
                        "no_declared_matched_candidate_family_within_exact_"
                        "target_dataset_evaluation_scope_aggregation_level"
                    ),
                    "paired_comparison_family": None,
                    "balanced_accuracy_p_value": None,
                    "macro_f1_p_value": None,
                }
            )
    return metrics_output, file_predictions


def _detector_level_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Binary metrics where every supplied row has exactly equal weight."""

    if not rows:
        raise ValueError("motion detector metrics require prediction rows")
    labels = np.asarray([int(row["activity_label"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [float(row["p_active"]) for row in rows], dtype=np.float64
    )
    predicted = np.asarray(
        [int(row["predicted_activity"]) for row in rows], dtype=np.int64
    )
    if (
        set(labels.tolist()) != {0, 1}
        or not set(predicted.tolist()) <= {0, 1}
        or not np.all(np.isfinite(probabilities))
    ):
        raise ValueError("motion detector metrics require finite binary-class rows")
    recalls: list[float] = []
    f1s: list[float] = []
    for class_id in (0, 1):
        true_positive = int(np.sum((labels == class_id) & (predicted == class_id)))
        false_negative = int(np.sum((labels == class_id) & (predicted != class_id)))
        false_positive = int(np.sum((labels != class_id) & (predicted == class_id)))
        recall = true_positive / (true_positive + false_negative)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        recalls.append(float(recall))
        f1s.append(float(f1))
    return {
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "sensitivity": recalls[1],
        "specificity": recalls[0],
        "roc_auc": _roc_auc(labels, probabilities),
        "pr_auc": _pr_auc_average_precision(labels, probabilities),
    }


def _worst_fold_balanced_accuracy(
    rows: Sequence[Mapping[str, Any]],
) -> float | None:
    """Return worst grouped-OOF fold BA; final-model transfers have no fold."""

    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        repeat = row.get("repeat_index")
        fold = row.get("fold_index")
        if repeat is None or fold is None:
            return None
        grouped.setdefault((int(repeat), int(fold)), []).append(row)
    scores = [
        _detector_level_metrics(selected)["balanced_accuracy"]
        for selected in grouped.values()
        if {int(row["activity_label"]) for row in selected} == {0, 1}
    ]
    return float(min(scores)) if scores else None


def _motion_participant_metric_rows(
    datasets: Sequence[
        tuple[str, str, str, Sequence[Mapping[str, Any]]]
    ],
    *,
    file_score_aggregation: str,
) -> list[dict[str, Any]]:
    """Materialize participant endpoints without conflating window/file units."""

    output: list[dict[str, Any]] = []
    for model_id, dataset, evaluation_scope, window_rows in datasets:
        for aggregation_level, selected in (
            ("window", list(window_rows)),
            (
                "file",
                _file_prediction_rows(
                    window_rows,
                    score_aggregation=file_score_aggregation,
                ),
            ),
        ):
            grouped: dict[str, list[Mapping[str, Any]]] = {}
            for row in selected:
                grouped.setdefault(str(row["participant_id"]), []).append(row)
            for participant_id, participant_rows in sorted(grouped.items()):
                output.append(
                    {
                        "model_id": model_id,
                        "dataset": dataset,
                        "target_dataset": _motion_target_dataset(dataset),
                        "evaluation_scope": evaluation_scope,
                        "aggregation_level": aggregation_level,
                        "participant_id": participant_id,
                        "observation_count": len(participant_rows),
                        **_detector_level_metrics(participant_rows),
                    }
                )
    return output


def _motion_training_source_inference_rows(
    datasets: Sequence[
        tuple[str, str, str, Sequence[Mapping[str, Any]]]
    ],
    participant_rows: Sequence[Mapping[str, Any]],
    *,
    file_score_aggregation: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    permutation_resamples: int,
    permutation_seed: int,
    reference_model_id: str,
    candidate_model_id: str,
) -> list[dict[str, Any]]:
    """Retrospective paired training-source ablation for exact shared targets.

    This is intentionally separate from the ordinary within-scope detector
    leaderboard: one route is grouped OOF on the target and the other is a
    frozen cross-dataset model.  It is therefore exploratory, asymmetric, and
    conditional on the persisted predictions.
    """

    reference_id = str(reference_model_id)
    candidate_id = str(candidate_model_id)
    if not reference_id or not candidate_id or reference_id == candidate_id:
        raise ValueError(
            "detector inference requires distinct non-empty reference/candidate ids"
        )
    source: dict[tuple[str, str], Sequence[Mapping[str, Any]]] = {}
    for model_id, dataset, _scope, rows in datasets:
        source[(model_id, _motion_target_dataset(dataset))] = rows
    targets = sorted(
        target
        for target in {target for _model, target in source}
        if (reference_id, target) in source and (candidate_id, target) in source
    )
    output: list[dict[str, Any]] = []
    for target in targets:
        for aggregation_level in ("window", "file"):
            selected_by_model: dict[str, list[Mapping[str, Any]]] = {}
            for model_id in (reference_id, candidate_id):
                selected = list(source[(model_id, target)])
                if aggregation_level == "file":
                    selected = _file_prediction_rows(
                        selected,
                        score_aggregation=file_score_aggregation,
                    )
                selected_by_model[model_id] = selected

            def unit_key(row: Mapping[str, Any]) -> tuple[str, ...]:
                base = (str(row["participant_id"]), str(row["file_id"]))
                return (
                    (*base, str(row["window_id"]))
                    if aggregation_level == "window"
                    else base
                )

            reference_by_unit = {
                unit_key(row): row for row in selected_by_model[reference_id]
            }
            candidate_by_unit = {
                unit_key(row): row for row in selected_by_model[candidate_id]
            }
            if set(reference_by_unit) != set(candidate_by_unit):
                raise ValueError(
                    "training-source detector comparison requires identical "
                    f"{target}/{aggregation_level} prediction units"
                )
            if any(
                int(reference_by_unit[key]["activity_label"])
                != int(candidate_by_unit[key]["activity_label"])
                for key in reference_by_unit
            ):
                raise ValueError(
                    "training-source detector comparison labels disagree"
                )

            endpoint_rows = [
                row
                for row in participant_rows
                if row["target_dataset"] == target
                and row["aggregation_level"] == aggregation_level
                and row["model_id"] in {reference_id, candidate_id}
            ]
            values_by_model_metric: dict[
                tuple[str, str], dict[str, float]
            ] = {}
            for row in endpoint_rows:
                for metric in _MOTION_DETECTOR_METRICS:
                    values_by_model_metric.setdefault(
                        (str(row["model_id"]), metric), {}
                    )[str(row["participant_id"])] = float(row[metric])
            family_rows: list[dict[str, Any]] = []
            raw_p: dict[str, float] = {}
            for metric in _MOTION_DETECTOR_METRICS:
                reference = values_by_model_metric[(reference_id, metric)]
                candidate = values_by_model_metric[(candidate_id, metric)]
                if set(reference) != set(candidate):
                    raise ValueError(
                        "training-source detector participant rosters disagree"
                    )
                participants = sorted(reference)
                differences = [
                    candidate[participant] - reference[participant]
                    for participant in participants
                ]
                ci_low, ci_high = _participant_mean_percentile_ci(
                    differences,
                    n_resamples=bootstrap_resamples,
                    seed=bootstrap_seed,
                )
                p_value = _paired_participant_sign_flip_p(
                    differences,
                    n_resamples=permutation_resamples,
                    seed=permutation_seed,
                )
                raw_p[metric] = p_value
                family_rows.append(
                    {
                        "reference_model_id": reference_id,
                        "candidate_model_id": candidate_id,
                        "target_dataset": target,
                        "aggregation_level": aggregation_level,
                        "metric": metric,
                        "candidate_minus_reference": float(np.mean(differences)),
                        "paired_participant_ci95_low": ci_low,
                        "paired_participant_ci95_high": ci_high,
                        "raw_p_value": p_value,
                        "paired_participant_count": len(participants),
                        "bootstrap_resamples": bootstrap_resamples,
                        "bootstrap_seed": bootstrap_seed,
                        "permutation_resamples": permutation_resamples,
                        "permutation_seed": permutation_seed,
                        "analysis_registration": (
                            "retrospective_exploratory_requested_2026-08-24"
                        ),
                        "comparison_design": (
                            "asymmetric_target_grouped_oof_vs_frozen_cross_dataset"
                        ),
                    }
                )
            adjusted = _holm_adjusted_p_values(raw_p)
            family_id = (
                f"detector_training_source::{target}::{aggregation_level}::"
                "registered_detector_endpoints"
            )
            for row in family_rows:
                row.update(
                    {
                        "holm_family": family_id,
                        "holm_family_size": len(family_rows),
                        "holm_adjusted_p_value": adjusted[str(row["metric"])],
                        "alpha": _STAGE5_INFERENCE_ALPHA,
                        "reject_after_holm": bool(
                            adjusted[str(row["metric"])]
                            <= _STAGE5_INFERENCE_ALPHA
                        ),
                    }
                )
            output.extend(family_rows)
    return output


def _motion_compact_metric_rows(
    participant_rows: Sequence[Mapping[str, Any]],
    inference_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a seven-column metric-long display plus lossless numeric audit."""

    grouped: dict[
        tuple[str, str, str, str], list[Mapping[str, Any]]
    ] = {}
    for row in participant_rows:
        key = (
            str(row["model_id"]),
            str(row["dataset"]),
            str(row["evaluation_scope"]),
            str(row["aggregation_level"]),
        )
        grouped.setdefault(key, []).append(row)
    inference_by_key = {
        (
            str(row["candidate_model_id"]),
            str(row["target_dataset"]),
            str(row["aggregation_level"]),
            str(row["metric"]),
        ): row
        for row in inference_rows
    }
    reference_keys = {
        (
            str(row["reference_model_id"]),
            str(row["target_dataset"]),
            str(row["aggregation_level"]),
            str(row["metric"]),
        )
        for row in inference_rows
    }
    display: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for (model_id, dataset, scope, level), selected in sorted(grouped.items()):
        target = _motion_target_dataset(dataset)
        for metric in _MOTION_DETECTOR_METRICS:
            values = [float(row[metric]) for row in selected]
            mean, sample_sd = _mean_sample_sd(values)
            ci_low, ci_high = _participant_mean_percentile_ci(
                values,
                n_resamples=bootstrap_resamples,
                seed=bootstrap_seed,
            )
            inference = inference_by_key.get((model_id, target, level, metric))
            if inference is not None:
                p_display = _format_probability(
                    float(inference["holm_adjusted_p_value"])
                )
                p_applicability = "available_retrospective_exploratory"
            elif (model_id, target, level, metric) in reference_keys:
                p_display = "Reference"
                p_applicability = "reference"
            else:
                p_display = "N/A"
                p_applicability = "N/A_no_matched_detector_reference"
            display.append(
                {
                    "model_id": model_id,
                    "evaluation": f"{dataset} ({scope})",
                    "level": level,
                    "metric": _MOTION_DETECTOR_METRIC_LABELS[metric],
                    "participant_macro_mean_sd": format_mean_sd(
                        mean,
                        sample_sd,
                        percent=True,
                    ),
                    "participant_bootstrap_ci95": format_interval(
                        ci_low,
                        ci_high,
                        percent=True,
                    ),
                    "holm_p_vs_reference": p_display,
                }
            )
            audit.append(
                {
                    "model_id": model_id,
                    "dataset": dataset,
                    "target_dataset": target,
                    "evaluation_scope": scope,
                    "aggregation_level": level,
                    "metric": metric,
                    "participant_count": len(values),
                    "participant_macro_mean": mean,
                    "between_participant_sample_sd": sample_sd,
                    "participant_bootstrap_ci95_low": ci_low,
                    "participant_bootstrap_ci95_high": ci_high,
                    "participant_bootstrap_resamples": bootstrap_resamples,
                    "participant_bootstrap_seed": bootstrap_seed,
                    "p_value_applicability": p_applicability,
                    "holm_adjusted_p_value": (
                        None
                        if inference is None
                        else float(inference["holm_adjusted_p_value"])
                    ),
                }
            )
    return display, audit


def _motion_worst_fold_rows(
    lossless_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "model_id": row["model_id"],
            "evaluation": (
                f"{row['dataset']} ({row['evaluation_scope']})"
            ),
            "level": row["aggregation_level"],
            "worst_fold_balanced_accuracy": format_mean_sd(
                row.get("worst_fold_balanced_accuracy"),
                None,
                percent=True,
            ),
        }
        for row in lossless_rows
        if row.get("worst_fold_balanced_accuracy") is not None
    ]


def _motion_per_class_table_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split per-class evidence into performance and discrimination tables."""

    performance: list[dict[str, Any]] = []
    discrimination: list[dict[str, Any]] = []
    for row in rows:
        identity = {
            "model_id": row["classifier_id"],
            "evaluation": row["evaluation_id"],
            "level": row["aggregation_level"],
            "activity": row["class_name"],
        }
        performance.append(
            {
                **identity,
                "sensitivity": format_mean_sd(
                    row.get("sensitivity"), None, percent=True
                ),
                "specificity": format_mean_sd(
                    row.get("specificity"), None, percent=True
                ),
                "balanced_accuracy_ovr": format_mean_sd(
                    row.get("balanced_accuracy_ovr"), None, percent=True
                ),
                "f1": format_mean_sd(row.get("f1"), None, percent=True),
            }
        )
        discrimination.append(
            {
                **identity,
                "precision": format_mean_sd(
                    row.get("precision"), None, percent=True
                ),
                "roc_auc_ovr": format_mean_sd(
                    row.get("roc_auc_ovr"), None, percent=True
                ),
                "pr_auc_ovr": format_mean_sd(
                    row.get("pr_auc_ovr"), None, percent=True
                ),
            }
        )
    return performance, discrimination


def _motion_detector_conclusion_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    denoiser_enabled: bool,
) -> list[dict[str, Any]]:
    """Conclude only inside exact target/scope/level detector strata."""

    grouped: dict[
        tuple[str, str, str], list[Mapping[str, Any]]
    ] = {}
    for row in rows:
        key = (
            str(row["target_dataset"]),
            str(row["evaluation_scope"]),
            str(row["aggregation_level"]),
        )
        grouped.setdefault(key, []).append(row)
    bootstrap_seeds = sorted(
        {
            int(row["participant_cluster_bootstrap_seed"])
            for row in rows
            if row.get("participant_cluster_bootstrap_seed") is not None
        }
    )
    bootstrap_resamples = sorted(
        {
            int(row["participant_cluster_bootstrap_resamples"])
            for row in rows
            if row.get("participant_cluster_bootstrap_resamples") is not None
        }
    )
    bootstrap_seed_display: Any = (
        bootstrap_seeds[0] if len(bootstrap_seeds) == 1 else bootstrap_seeds or "N/A"
    )
    bootstrap_resample_display: Any = (
        bootstrap_resamples[0]
        if len(bootstrap_resamples) == 1
        else bootstrap_resamples or "N/A"
    )

    conclusions: list[dict[str, Any]] = []
    for (target_dataset, evaluation_scope, aggregation_level), selected in sorted(
        grouped.items()
    ):
        leader = max(
            selected,
            key=lambda row: float(row.get("balanced_accuracy", -math.inf)),
        )
        candidate_count = len(selected)
        prefix = (
            "Within-stratum BA leader"
            if candidate_count > 1
            else "Reported endpoint; no within-stratum candidate family"
        )
        conclusions.append(
            {
                "angle": (
                    "motion_detector_endpoint::"
                    f"{target_dataset}::{evaluation_scope}::{aggregation_level}"
                ),
                "target_dataset": target_dataset,
                "evaluation_scope": evaluation_scope,
                "aggregation_level": aggregation_level,
                "within_stratum_candidate_count": candidate_count,
                "leading_or_selected_case": leader.get("model_id"),
                "finding": (
                    f"{prefix}: model={leader.get('model_id')}, "
                    f"evaluation_id={leader.get('dataset')}, "
                    f"BA={100.0*float(leader['balanced_accuracy']):.1f}%, "
                    f"macro-F1={100.0*float(leader['macro_f1']):.1f}%, "
                    f"ROC-AUC={100.0*float(leader['roc_auc']):.1f}%. "
                    "No comparison is made across target datasets, evaluation "
                    "scopes, or aggregation levels."
                ),
                "confidence": (
                    "grouped_oof_descriptive"
                    if evaluation_scope == "source_grouped_oof"
                    else "cross_dataset_benchmark_not_independent_untouched_test"
                ),
                "selection_effect": "none_automatic",
            }
        )
    conclusions.extend(
        [
            {
                "angle": "uncertainty_and_inference",
                "target_dataset": "all_reported_endpoints",
                "evaluation_scope": "endpoint_specific",
                "aggregation_level": "window_and_file_kept_separate",
                "within_stratum_candidate_count": None,
                "leading_or_selected_case": None,
                "finding": (
                    "Repeat sample SD and Student-t CI are N/A for a one-repeat "
                    "grouped OOF endpoint or a single frozen transfer endpoint and "
                    "are explicitly marked not computed. A configured-and-persisted "
                    "participant-cluster percentile bootstrap provides BA, macro-F1, "
                    "sensitivity, specificity and ROC-AUC intervals "
                    f"(seed={bootstrap_seed_display}, "
                    f"resamples={bootstrap_resample_display}) whenever "
                    "every participant cluster safely carries both protocol classes. "
                    "Paired P values are N/A because no declared matched candidate "
                    "family exists inside an exact target/scope/level stratum; any "
                    "separately labeled retrospective training-source analysis is "
                    "not this within-stratum estimand."
                ),
                "confidence": "descriptive_endpoint_specific_uncertainty",
                "selection_effect": "none_automatic",
            },
            {
                "angle": "denoiser",
                "target_dataset": "ptt22",
                "evaluation_scope": "endpoint_benchmark",
                "aggregation_level": "subject_macro_by_activity_and_channel",
                "within_stratum_candidate_count": None,
                "leading_or_selected_case": None,
                "finding": (
                    "Denoiser evidence is reported separately by static/dynamic "
                    "activity and ordered by subject-macro PPI–RR RMSE. Its SD "
                    "columns are between-subject sample SD, not repeat-training "
                    "uncertainty. Any reducer-reference P values are a separately "
                    "labeled retrospective exploratory supplement."
                    if denoiser_enabled
                    else "Denoiser benchmark was skipped; outputs are explicitly "
                    "N/A with the execution-option reason."
                ),
                "confidence": (
                    "endpoint_benchmark" if denoiser_enabled else "not_available"
                ),
                "selection_effect": "none_automatic",
            },
        ]
    )
    return conclusions


def _confusion_table_has_endpoint(
    path: Path,
    *,
    dataset: str,
    aggregation_level: str,
) -> bool:
    """Verify that a persisted confusion table contains the exact endpoint."""

    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"confusion table must contain a row list: {path}")
    return any(
        isinstance(row, Mapping)
        and str(row.get("dataset")) == dataset
        and (
            str(row.get("aggregation_level")) == "window"
            if aggregation_level == "window"
            else str(row.get("aggregation_level", "")).startswith("file_")
        )
        for row in payload
    )


def _stage5_reporter_output_status(
    *,
    profile_rows: Sequence[Mapping[str, Any]],
    report_config: Mapping[str, Any],
    tables: Path,
    figures: Path,
    reverse_available: bool,
    denoiser_enabled: bool,
) -> list[dict[str, Any]]:
    """Audit every Stage5 profile/config output as generated or reasoned N/A."""

    requirements: dict[tuple[str, str], set[str]] = {}
    for profile in profile_rows:
        profile_id = str(profile["profile_id"])
        for output_type, field in (
            ("table", "required_tables"),
            ("figure", "required_figures"),
        ):
            for output_id in profile.get(field, ()):
                key = (output_type, str(output_id))
                requirements.setdefault(key, set()).add(
                    f"reporter_profile:{profile_id}"
                )
    for output_id in report_config.get("required_detector_figures", ()):
        requirements.setdefault(("figure", str(output_id)), set()).add(
            "resolved_plan.report.required_detector_figures"
        )
    for output_id in report_config.get("denoiser_figures_when_enabled", ()):
        requirements.setdefault(("figure", str(output_id)), set()).add(
            "resolved_plan.report.denoiser_figures_when_enabled"
        )

    reverse_prefixes = (
        "motion_ptt_training_oof",
        "motion_internal_reverse",
        "ptt22_trained",
    )
    denoiser_prefixes = ("denoiser_",)
    output: list[dict[str, Any]] = []
    missing_active: list[str] = []
    for (output_type, output_id), sources in sorted(requirements.items()):
        if output_type == "table":
            paths = (tables / f"{output_id}.csv", tables / f"{output_id}.json")
        else:
            paths = (figures / f"{output_id}.png",)
        obsolete_replacements = (
            _OBSOLETE_STAGE5_SUBJECT_CONFUSION_REPLACEMENTS.get(output_id)
            if output_type == "figure"
            else None
        )
        replacement_paths: tuple[Path, ...] = ()
        replacement_status = "not_applicable"
        generated = all(path.is_file() for path in paths)
        status = "generated" if generated else "N/A"
        reason = ""
        if obsolete_replacements is not None:
            replacement_figure_ids = obsolete_replacements[:2]
            replacement_dataset = obsolete_replacements[2]
            replacement_paths = (
                tables / "motion_detector_window_confusion.csv",
                tables / "motion_detector_window_confusion.json",
                tables / "motion_detector_file_confusion.csv",
                tables / "motion_detector_file_confusion.json",
                *(figures / f"{value}.png" for value in replacement_figure_ids),
            )
            status = "N/A"
            reason = "superseded_by_window_and_file_level_contract"
            if output_id.startswith(reverse_prefixes) and not reverse_available:
                replacement_status = "N/A_reverse_motion_ablation_not_executed"
            elif (
                all(path.is_file() for path in replacement_paths)
                and _confusion_table_has_endpoint(
                    tables / "motion_detector_window_confusion.json",
                    dataset=replacement_dataset,
                    aggregation_level="window",
                )
                and _confusion_table_has_endpoint(
                    tables / "motion_detector_file_confusion.json",
                    dataset=replacement_dataset,
                    aggregation_level="file",
                )
            ):
                replacement_status = "generated"
            else:
                missing_replacements = [
                    path.relative_to(tables.parent).as_posix()
                    for path in replacement_paths
                    if not path.is_file()
                ]
                for table_name, aggregation_level in (
                    ("motion_detector_window_confusion", "window"),
                    ("motion_detector_file_confusion", "file"),
                ):
                    table_path = tables / f"{table_name}.json"
                    if table_path.is_file() and not _confusion_table_has_endpoint(
                        table_path,
                        dataset=replacement_dataset,
                        aggregation_level=aggregation_level,
                    ):
                        missing_replacements.append(
                            f"tables/{table_name}.json#dataset={replacement_dataset}"
                        )
                replacement_status = "required_replacement_output_missing"
                reason = (
                    "superseded_subject_output_has_incomplete_window_file_"
                    "replacement_contract"
                )
                missing_active.append(
                    f"{output_type}:{output_id} replacements="
                    + ",".join(missing_replacements)
                )
        elif not generated:
            if output_id.startswith(reverse_prefixes) and not reverse_available:
                reason = "reverse_motion_ablation_not_executed"
            elif output_id.startswith(denoiser_prefixes) and not denoiser_enabled:
                reason = "denoiser_disabled_by_execution_option"
            else:
                reason = "required_active_output_missing"
                missing_active.append(f"{output_type}:{output_id}")
        output.append(
            {
                "output_type": output_type,
                "output_id": output_id,
                "required_by": ";".join(sorted(sources)),
                "status": status,
                "reason": reason,
                "paths": ";".join(
                    path.relative_to(tables.parent).as_posix() for path in paths
                ),
                "replacement_status": replacement_status,
                "replacement_paths": ";".join(
                    path.relative_to(tables.parent).as_posix()
                    for path in replacement_paths
                ),
            }
        )
    if missing_active:
        raise ValueError(
            "Stage5 reporter profile outputs are missing without a valid N/A "
            "condition: " + ", ".join(sorted(missing_active))
        )
    return output


def _rank_and_mark_denoiser_rows(
    rows: Sequence[Mapping[str, Any]], activity_group: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return numeric rows plus report-only best-cell decorations.

    Subject-macro PPI--RR RMSE is the primary ascending sort key; F1 is only
    the descending tie-breaker. Lossless JSON remains numeric. The compact
    static/dynamic CSV, XLSX, Markdown, and HTML result tables use the marked
    display projection.
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
            (
                -float(row["participant_macro_f1"])
                if row.get("participant_macro_f1") is not None
                and math.isfinite(float(row["participant_macro_f1"]))
                else math.inf
            ),
            str(row.get("algorithm_or_reducer", "")),
            str(row.get("channel", "")),
        )
    )
    if not numeric:
        return [], []
    finite_f1 = [
        float(row["participant_macro_f1"])
        for row in numeric
        if row.get("participant_macro_f1") is not None
        and math.isfinite(float(row["participant_macro_f1"]))
    ]
    best_f1 = max(finite_f1) if finite_f1 else None
    finite_rmse = [
        float(row["participant_macro_ibi_ppi_rmse_ms"])
        for row in numeric
        if row.get("participant_macro_ibi_ppi_rmse_ms") is not None
        and math.isfinite(float(row["participant_macro_ibi_ppi_rmse_ms"]))
    ]
    best_rmse = min(finite_rmse) if finite_rmse else None
    display: list[dict[str, Any]] = []
    for row, compact in zip(numeric, compact_rows(numeric), strict=True):
        marked = dict(compact)
        f1_best = (
            best_f1 is not None
            and row.get("participant_macro_f1") is not None
            and math.isfinite(float(row["participant_macro_f1"]))
            and math.isclose(
                float(row["participant_macro_f1"]),
                best_f1,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )
        rmse_best = (
            best_rmse is not None
            and row.get("participant_macro_ibi_ppi_rmse_ms") is not None
            and math.isfinite(float(row["participant_macro_ibi_ppi_rmse_ms"]))
            and math.isclose(
                float(row["participant_macro_ibi_ppi_rmse_ms"]),
                best_rmse,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        if f1_best:
            field = (
                "participant_macro_f1_mean_sd"
                if "participant_macro_f1_mean_sd" in marked
                else "participant_macro_f1"
            )
            marked[field] = f"{marked[field]}*"
        if rmse_best:
            field = (
                "participant_macro_ibi_ppi_rmse_ms_mean_sd"
                if "participant_macro_ibi_ppi_rmse_ms_mean_sd" in marked
                else "participant_macro_ibi_ppi_rmse_ms"
            )
            marked[field] = f"{marked[field]}*"
        if f1_best or rmse_best:
            marker = "**" if f1_best and rmse_best else "*"
            marked["algorithm_or_reducer"] = (
                f"{row['algorithm_or_reducer']}{marker}"
            )
        display.append(marked)
    return numeric, display


def _denoiser_activity_result_rows(
    summary_rows: Sequence[Mapping[str, Any]],
    inference_rows: Sequence[Mapping[str, Any]],
    *,
    activity_group: str,
    reference_id: str,
) -> list[dict[str, Any]]:
    """Project one activity into the frozen five-column denoiser table.

    Rows are sorted by participant-macro IBI--PPI RMSE ascending across both
    optical channels. A star decorates the best RMSE and F1 cells and the
    denoiser identifier; a double star on the identifier means the same row is
    best for both endpoints. Only the RMSE Holm-adjusted P value versus the
    configured identity reference is exposed in this human-facing table.
    """

    if activity_group not in {"static", "dynamic"}:
        raise ValueError("denoiser activity table must be static or dynamic")
    numeric, marked = _rank_and_mark_denoiser_rows(
        summary_rows,
        activity_group,
    )
    rmse_inference = {
        (str(row["candidate_denoiser"]), str(row["channel"])): row
        for row in inference_rows
        if str(row.get("activity_group")) == activity_group
        and str(row.get("metric"))
        == "participant_macro_ibi_ppi_rmse_ms"
    }
    output: list[dict[str, Any]] = []
    for source, display in zip(numeric, marked, strict=True):
        algorithm = str(source["algorithm_or_reducer"])
        rmse_value = display.get(
            "participant_macro_ibi_ppi_rmse_ms_mean_sd",
            display.get("participant_macro_ibi_ppi_rmse_ms", "N/A"),
        )
        f1_value = display.get(
            "participant_macro_f1_mean_sd",
            display.get("participant_macro_f1", "N/A"),
        )
        if algorithm == reference_id:
            p_value = "Reference"
        else:
            inference = rmse_inference.get(
                (algorithm, str(source["channel"]))
            )
            p_value = _format_probability(
                None
                if inference is None
                else float(inference["holm_adjusted_p_value"])
            )
        output.append(
            {
                "denoiser": display["algorithm_or_reducer"],
                "IR/RED": source["channel"],
                "RMSE ± SD (ms)": rmse_value,
                "F1 ± SD (%)": f1_value,
                "RMSE P versus identity": p_value,
            }
        )
    return output


def _annotate_denoiser_uncertainty(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Label denoiser SDs as between-subject spread, never repeat uncertainty."""

    metric_fields = (
        "participant_macro_f1",
        "participant_macro_sensitivity",
        "participant_macro_positive_predictive_value",
        "participant_macro_ibi_ppi_rmse_ms",
    )
    output: list[dict[str, Any]] = []
    for row in rows:
        annotated = dict(row)
        for metric in metric_fields:
            annotated.setdefault(f"{metric}_sd", None)
        has_subject_sd = any(
            annotated.get(f"{metric}_sd") is not None for metric in metric_fields
        )
        annotated.update(
            {
                "denoiser_sd_unit": "subject",
                "denoiser_sd_estimator": (
                    "between_subject_sample_sd_ddof1"
                    if has_subject_sd
                    else "N/A"
                ),
                "denoiser_sd_applicability": (
                    "available" if has_subject_sd else "N/A"
                ),
                "denoiser_sd_reason": (
                    ""
                    if has_subject_sd
                    else "between_subject_sd_not_present_in_persisted_summary"
                ),
                "denoiser_sd_interpretation": (
                    "descriptive_between_subject_variability_not_repeat_"
                    "training_uncertainty"
                ),
                "repeat_uncertainty_applicability": "N/A",
                "repeat_uncertainty_reason": (
                    "denoiser_endpoint_has_no_repeated_training_evaluation_axis"
                ),
            }
        )
        output.append(annotated)
    return output


def _denoiser_participant_endpoint_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Preserve the participant endpoints discarded by the legacy aggregate."""

    grouped: dict[
        tuple[str, str, str, str], list[Mapping[str, Any]]
    ] = {}
    for row in rows:
        key = (
            str(row["algorithm_or_reducer"]),
            str(row["activity_group"]),
            str(row["channel"]),
            str(row["participant_id"]),
        )
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (algorithm, activity, channel, participant), selected in sorted(
        grouped.items()
    ):
        summary = _aggregate_benchmark(selected)
        if len(summary) != 1:
            raise ValueError("participant denoiser aggregation must yield one row")
        endpoint = summary[0]
        output.append(
            {
                "algorithm_or_reducer": algorithm,
                "activity_group": activity,
                "channel": channel,
                "participant_id": participant,
                "passed_segment_count": endpoint["passed_segment_count"],
                "failed_segment_count": endpoint["failed_segment_count"],
                "participant_macro_f1": endpoint["participant_macro_f1"],
                "participant_macro_sensitivity": endpoint[
                    "participant_macro_sensitivity"
                ],
                "participant_macro_positive_predictive_value": endpoint[
                    "participant_macro_positive_predictive_value"
                ],
                "participant_macro_ibi_ppi_rmse_ms": endpoint[
                    "participant_macro_ibi_ppi_rmse_ms"
                ],
            }
        )
    return output


def _denoiser_common_segment_participant_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_id: str,
    candidate_id: str,
    activity_group: str,
    channel: str,
    metric: str,
) -> tuple[dict[str, tuple[float, float]], int]:
    """Return identity/candidate endpoints on an identical successful roster."""

    selected = [
        row
        for row in rows
        if str(row["activity_group"]) == activity_group
        and str(row["channel"]) == channel
        and str(row["algorithm_or_reducer"])
        in {reference_id, candidate_id}
    ]

    def key(row: Mapping[str, Any]) -> tuple[str, str, float]:
        return (
            str(row["participant_id"]),
            str(row["record_id"]),
            float(row["segment_start_s"]),
        )

    by_algorithm: dict[str, dict[tuple[str, str, float], Mapping[str, Any]]] = {}
    for row in selected:
        algorithm = str(row["algorithm_or_reducer"])
        row_key = key(row)
        if row_key in by_algorithm.setdefault(algorithm, {}):
            raise ValueError("denoiser comparison has duplicate segment keys")
        by_algorithm[algorithm][row_key] = row
    reference = by_algorithm.get(reference_id, {})
    candidate = by_algorithm.get(candidate_id, {})
    common_keys = sorted(set(reference) & set(candidate))
    usable: list[tuple[str, str, float]] = []
    is_rmse = metric == "participant_macro_ibi_ppi_rmse_ms"
    for row_key in common_keys:
        reference_row = reference[row_key]
        candidate_row = candidate[row_key]
        if (
            reference_row.get("status") != "passed"
            or candidate_row.get("status") != "passed"
        ):
            continue
        if is_rmse and any(
            _nonnegative_integer(row.get("matched_interval_count"), "matched_interval_count")
            <= 0
            or row.get("ibi_ppi_rmse_ms") is None
            or not math.isfinite(float(row["ibi_ppi_rmse_ms"]))
            for row in (reference_row, candidate_row)
        ):
            continue
        usable.append(row_key)
    by_participant: dict[str, list[tuple[str, str, float]]] = {}
    for row_key in usable:
        by_participant.setdefault(row_key[0], []).append(row_key)
    output: dict[str, tuple[float, float]] = {}
    for participant, participant_keys in sorted(by_participant.items()):
        reference_summary = _aggregate_benchmark(
            [reference[row_key] for row_key in participant_keys]
        )[0]
        candidate_summary = _aggregate_benchmark(
            [candidate[row_key] for row_key in participant_keys]
        )[0]
        reference_value = reference_summary.get(metric)
        candidate_value = candidate_summary.get(metric)
        if reference_value is None or candidate_value is None:
            continue
        output[participant] = (
            float(reference_value),
            float(candidate_value),
        )
    return output, len(usable)


def _denoiser_inference_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    permutation_resamples: int,
    permutation_seed: int,
    reference_id: str,
) -> list[dict[str, Any]]:
    """Retrospective identity-controlled participant-paired denoiser tests."""

    algorithms = sorted(
        {
            str(row["algorithm_or_reducer"])
            for row in rows
            if str(row["algorithm_or_reducer"]) != reference_id
        }
    )
    if not reference_id or not any(
        str(row["algorithm_or_reducer"]) == reference_id for row in rows
    ):
        raise ValueError(
            "denoiser inference reference must name one persisted reducer"
        )
    output: list[dict[str, Any]] = []
    for activity_group in ("static", "dynamic"):
        for channel in _CHANNELS:
            for metric in _DENOISER_PRIMARY_METRICS:
                family: list[dict[str, Any]] = []
                raw_p: dict[str, float] = {}
                for candidate_id in algorithms:
                    paired, common_segment_count = (
                        _denoiser_common_segment_participant_values(
                            rows,
                            reference_id=reference_id,
                            candidate_id=candidate_id,
                            activity_group=activity_group,
                            channel=channel,
                            metric=metric,
                        )
                    )
                    participants = sorted(paired)
                    if len(participants) < 2:
                        continue
                    differences = [
                        paired[participant][1] - paired[participant][0]
                        for participant in participants
                    ]
                    ci_low, ci_high = _participant_mean_percentile_ci(
                        differences,
                        n_resamples=bootstrap_resamples,
                        seed=bootstrap_seed,
                    )
                    p_value = _paired_participant_sign_flip_p(
                        differences,
                        n_resamples=permutation_resamples,
                        seed=permutation_seed,
                    )
                    raw_p[candidate_id] = p_value
                    family.append(
                        {
                            "reference_denoiser": reference_id,
                            "candidate_denoiser": candidate_id,
                            "activity_group": activity_group,
                            "channel": channel,
                            "metric": metric,
                            "candidate_minus_reference": float(
                                np.mean(differences)
                            ),
                            "paired_participant_ci95_low": ci_low,
                            "paired_participant_ci95_high": ci_high,
                            "raw_p_value": p_value,
                            "paired_participant_count": len(participants),
                            "endpoint_common_segment_count": common_segment_count,
                            "bootstrap_resamples": bootstrap_resamples,
                            "bootstrap_seed": bootstrap_seed,
                            "permutation_resamples": permutation_resamples,
                            "permutation_seed": permutation_seed,
                            "analysis_registration": (
                                "retrospective_exploratory_requested_2026-08-24"
                            ),
                            "common_segment_rule": (
                                "same_participant_record_segment_passed_in_identity_"
                                "and_candidate_endpoint_specific"
                            ),
                        }
                    )
                adjusted = _holm_adjusted_p_values(raw_p)
                family_id = (
                    f"denoiser::{activity_group}::{channel}::{metric}::"
                    "candidate_reducers_vs_reference"
                )
                for row in family:
                    row.update(
                        {
                            "holm_family": family_id,
                            "holm_family_size": len(family),
                            "holm_adjusted_p_value": adjusted[
                                str(row["candidate_denoiser"])
                            ],
                            "alpha": _STAGE5_INFERENCE_ALPHA,
                            "reject_after_holm": bool(
                                adjusted[str(row["candidate_denoiser"])]
                                <= _STAGE5_INFERENCE_ALPHA
                            ),
                        }
                    )
                output.extend(family)
    return output


def _denoiser_compact_metric_rows(
    participant_rows: Sequence[Mapping[str, Any]],
    inference_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    reference_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build metric-long denoiser result rows with at most seven columns."""

    inference_by_key = {
        (
            str(row["candidate_denoiser"]),
            str(row["activity_group"]),
            str(row["channel"]),
            str(row["metric"]),
        ): row
        for row in inference_rows
    }
    grouped: dict[
        tuple[str, str, str], list[Mapping[str, Any]]
    ] = {}
    for row in participant_rows:
        grouped.setdefault(
            (
                str(row["algorithm_or_reducer"]),
                str(row["activity_group"]),
                str(row["channel"]),
            ),
            [],
        ).append(row)
    audit: list[dict[str, Any]] = []
    for (algorithm, activity, channel), selected in sorted(grouped.items()):
        for metric, (_label, percent, _direction) in _DENOISER_PRIMARY_METRICS.items():
            values = [
                float(row[metric])
                for row in selected
                if row.get(metric) is not None
                and math.isfinite(float(row[metric]))
            ]
            if not values:
                continue
            mean, sample_sd = _mean_sample_sd(values)
            ci_low, ci_high = _participant_mean_percentile_ci(
                values,
                n_resamples=bootstrap_resamples,
                seed=bootstrap_seed,
            )
            inference = inference_by_key.get(
                (algorithm, activity, channel, metric)
            )
            audit.append(
                {
                    "algorithm_or_reducer": algorithm,
                    "activity_group": activity,
                    "channel": channel,
                    "metric": metric,
                    "participant_count": len(values),
                    "participant_macro_mean": mean,
                    "between_participant_sample_sd": sample_sd,
                    "participant_bootstrap_ci95_low": ci_low,
                    "participant_bootstrap_ci95_high": ci_high,
                    "participant_bootstrap_resamples": bootstrap_resamples,
                    "participant_bootstrap_seed": bootstrap_seed,
                    "raw_p_value": (
                        None
                        if inference is None
                        else float(inference["raw_p_value"])
                    ),
                    "holm_adjusted_p_value": (
                        None
                        if inference is None
                        else float(inference["holm_adjusted_p_value"])
                    ),
                    "p_value_role": (
                        "reference"
                        if algorithm == reference_id
                        else "retrospective_exploratory_paired_vs_reference"
                    ),
                    "percent_metric": percent,
                }
            )

    best_keys: set[tuple[str, str, str, str]] = set()
    for activity in ("static", "dynamic"):
        for channel in _CHANNELS:
            for metric, (_label, _percent, direction) in _DENOISER_PRIMARY_METRICS.items():
                selected = [
                    row
                    for row in audit
                    if row["activity_group"] == activity
                    and row["channel"] == channel
                    and row["metric"] == metric
                ]
                if not selected:
                    continue
                best = (
                    min(selected, key=lambda row: float(row["participant_macro_mean"]))
                    if direction == "lower"
                    else max(selected, key=lambda row: float(row["participant_macro_mean"]))
                )
                best_keys.add((
                    str(best["algorithm_or_reducer"]),
                    activity,
                    channel,
                    metric,
                ))

    display: list[dict[str, Any]] = []
    for row in audit:
        metric = str(row["metric"])
        label, percent, _direction = _DENOISER_PRIMARY_METRICS[metric]
        algorithm = str(row["algorithm_or_reducer"])
        marked_algorithm = (
            f"{algorithm}*"
            if (
                algorithm,
                str(row["activity_group"]),
                str(row["channel"]),
                metric,
            ) in best_keys
            else algorithm
        )
        if algorithm == reference_id:
            p_display = "Reference"
        else:
            p_display = _format_probability(
                None
                if row["holm_adjusted_p_value"] is None
                else float(row["holm_adjusted_p_value"])
            )
        display.append(
            {
                "denoiser": marked_algorithm,
                "activity": row["activity_group"],
                "channel": row["channel"],
                "metric": label,
                "mean_sd": format_mean_sd(
                    row["participant_macro_mean"],
                    row["between_participant_sample_sd"],
                    percent=percent,
                ),
                "participant_bootstrap_ci95": format_interval(
                    row["participant_bootstrap_ci95_low"],
                    row["participant_bootstrap_ci95_high"],
                    percent=percent,
                ),
                "holm_p_vs_reference": p_display,
            }
        )
    numeric_by_key = {
        (
            str(row["algorithm_or_reducer"]),
            str(row["activity_group"]),
            str(row["channel"]),
            str(row["metric"]),
        ): float(row["participant_macro_mean"])
        for row in audit
    }
    metric_by_label = {
        label: (metric, direction)
        for metric, (label, _percent, direction) in _DENOISER_PRIMARY_METRICS.items()
    }

    def display_order(row: Mapping[str, Any]) -> tuple[Any, ...]:
        metric, direction = metric_by_label[str(row["metric"])]
        algorithm = str(row["denoiser"]).rstrip("*")
        numeric = numeric_by_key[
            (
                algorithm,
                str(row["activity"]),
                str(row["channel"]),
                metric,
            )
        ]
        return (
            tuple(_DENOISER_PRIMARY_METRICS).index(metric),
            str(row["channel"]),
            ("static", "dynamic").index(str(row["activity"])),
            numeric if direction == "lower" else -numeric,
            algorithm,
        )

    display.sort(key=display_order)
    return display, audit


def _denoiser_coverage_rows(
    summary_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "denoiser": row["algorithm_or_reducer"],
            "activity": row["activity_group"],
            "channel": row["channel"],
            "participant_coverage_percent": format_mean_sd(
                row.get("participant_coverage_rate"), None, percent=True
            ),
            "segment_coverage_percent": format_mean_sd(
                row.get("segment_coverage_rate"), None, percent=True
            ),
            "failed_segments": row.get("failed_segment_count"),
        }
        for row in summary_rows
    ]


def _file_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_aggregation: str = "median",
) -> list[dict[str, Any]]:
    """Collapse windows to one configured score per physical recording file."""

    aggregators: Mapping[str, Callable[[np.ndarray], float]] = {
        "median": lambda values: float(np.median(values)),
        "mean": lambda values: float(np.mean(values)),
        "maximum": lambda values: float(np.max(values)),
    }
    if score_aggregation not in aggregators:
        raise ValueError("file score aggregation must be median, mean, or maximum")

    grouped: dict[tuple[int | None, int | None, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        participant_id = str(row.get("participant_id", ""))
        file_id = str(row.get("file_id", ""))
        if not participant_id or not file_id:
            raise ValueError("file-level motion reporting requires participant_id and file_id")
        repeat = (
            int(row["repeat_index"]) if row.get("repeat_index") is not None else None
        )
        fold = int(row["fold_index"]) if row.get("fold_index") is not None else None
        grouped.setdefault((repeat, fold, participant_id, file_id), []).append(row)
    if not grouped:
        raise ValueError("file-level motion reporting requires prediction rows")

    output: list[dict[str, Any]] = []
    for (repeat, fold, participant_id, file_id), selected in sorted(
        grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        probabilities = np.asarray(
            [float(row["p_active"]) for row in selected], dtype=np.float64
        )
        thresholds = {float(row["threshold"]) for row in selected}
        labels = {int(row["activity_label"]) for row in selected}
        if (
            not np.all(np.isfinite(probabilities))
            or len(thresholds) != 1
            or len(labels) != 1
            or not math.isfinite(next(iter(thresholds)))
        ):
            raise ValueError(
                "file-level motion reporting requires one label, one frozen "
                "threshold, and finite scores per file"
            )
        probability = aggregators[score_aggregation](probabilities)
        threshold = thresholds.pop()
        output.append(
            {
                "participant_id": participant_id,
                "file_id": file_id,
                "activity": selected[0].get(
                    "activity", selected[0].get("role_family")
                ),
                "activity_label": labels.pop(),
                "window_count": len(selected),
                "p_active": probability,
                "score_aggregation": score_aggregation,
                "threshold": threshold,
                "predicted_activity": int(probability >= threshold),
                "repeat_index": repeat,
                "fold_index": fold,
            }
        )
    return output


def _confusion(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    matrix = np.zeros((2, 2), dtype=np.int64)
    for row in rows:
        matrix[int(row["activity_label"]), int(row["predicted_activity"])] += 1
    return matrix


def _confusion_report_row(
    model_id: str,
    dataset: str,
    matrix: np.ndarray,
    *,
    aggregation_level: str,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "dataset": dataset,
        "aggregation_level": aggregation_level,
        "true_static_predicted_static": int(matrix[0, 0]),
        "true_static_predicted_motion": int(matrix[0, 1]),
        "true_motion_predicted_static": int(matrix[1, 0]),
        "true_motion_predicted_motion": int(matrix[1, 1]),
    }


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


def _plot_static_peak_boxplot(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    title: str,
) -> None:
    """Plot recording distributions with paper-specified 10th/90th whiskers."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = sorted(
        {
            (str(row["algorithm_or_reducer"]), str(row["channel"]))
            for row in rows
            if row.get("status") == "passed" and row.get(metric) is not None
        }
    )
    values = [
        [
            float(row[metric])
            for row in rows
            if row.get("status") == "passed"
            and (str(row["algorithm_or_reducer"]), str(row["channel"])) == group
            and row.get(metric) is not None
        ]
        for group in groups
    ]
    labels = [f"{algorithm}\n{channel}" for algorithm, channel in groups]
    width = max(7.0, 1.1 * len(groups))
    figure, axis = plt.subplots(figsize=(width, 4.8))
    if values:
        axis.boxplot(values, tick_labels=labels, whis=(10, 90), showfliers=True)
    axis.set_title(title)
    axis.set_ylabel(metric)
    axis.tick_params(axis="x", labelrotation=25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _score_distribution_rows(
    datasets: Sequence[
        tuple[str, str, str, Sequence[Mapping[str, Any]]]
    ],
    *,
    aggregation_level: str,
    file_score_aggregation: str = "median",
) -> list[dict[str, Any]]:
    """Materialize the numerical table paired with detector score plots."""

    output: list[dict[str, Any]] = []
    for model_id, dataset, evaluation_scope, window_rows in datasets:
        rows = (
            window_rows
            if aggregation_level == "window"
            else _file_prediction_rows(
                window_rows, score_aggregation=file_score_aggregation
            )
        )
        thresholds = np.asarray(
            [float(row["threshold"]) for row in rows], dtype=np.float64
        )
        for class_id, class_name in ((0, "static"), (1, "motion")):
            scores = np.asarray(
                [
                    float(row["p_active"])
                    for row in rows
                    if int(row["activity_label"]) == class_id
                ],
                dtype=np.float64,
            )
            if not scores.size or not np.all(np.isfinite(scores)):
                raise ValueError("score distribution requires finite rows from both classes")
            quantiles = np.quantile(scores, (0.05, 0.25, 0.5, 0.75, 0.95))
            output.append(
                {
                    "model_id": model_id,
                    "dataset": dataset,
                    "evaluation_scope": evaluation_scope,
                    "aggregation_level": aggregation_level,
                    "activity_class": class_name,
                    "observation_count": int(scores.size),
                    "score_mean": float(np.mean(scores)),
                    "score_sd": (
                        float(np.std(scores, ddof=1)) if scores.size >= 2 else None
                    ),
                    "score_q05": float(quantiles[0]),
                    "score_q25": float(quantiles[1]),
                    "score_median": float(quantiles[2]),
                    "score_q75": float(quantiles[3]),
                    "score_q95": float(quantiles[4]),
                    "threshold_min": float(np.min(thresholds)),
                    "threshold_median": float(np.median(thresholds)),
                    "threshold_max": float(np.max(thresholds)),
                }
            )
    return output


def _plot_score_distribution(
    path: Path,
    datasets: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    *,
    aggregation_level: str,
    file_score_aggregation: str = "median",
    score_histogram_bins: int = 40,
    title: str,
) -> None:
    """Plot class-conditional scores and every frozen threshold location."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(max(6.4, 6.0 * len(datasets)), 4.6),
        squeeze=False,
        sharex=True,
    )
    bins = np.linspace(0.0, 1.0, int(score_histogram_bins) + 1)
    for axis, (dataset, window_rows) in zip(axes[0], datasets, strict=True):
        rows = (
            window_rows
            if aggregation_level == "window"
            else _file_prediction_rows(
                window_rows, score_aggregation=file_score_aggregation
            )
        )
        for class_id, class_name, color in (
            (0, "static", "tab:blue"),
            (1, "motion", "tab:orange"),
        ):
            scores = [
                float(row["p_active"])
                for row in rows
                if int(row["activity_label"]) == class_id
            ]
            axis.hist(
                scores,
                bins=bins,
                alpha=0.58,
                color=color,
                label=f"{class_name} (n={len(scores)})",
            )
        thresholds = np.unique(
            np.asarray([float(row["threshold"]) for row in rows], dtype=np.float64)
        )
        threshold_median = float(np.median(thresholds))
        if thresholds.size > 1:
            axis.axvspan(
                float(np.min(thresholds)),
                float(np.max(thresholds)),
                color="black",
                alpha=0.10,
                label="fold-threshold range",
            )
        axis.axvline(
            threshold_median,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"threshold median={threshold_median:.6f}",
        )
        axis.set_title(dataset)
        axis.set_xlabel("Predicted motion probability")
        axis.set_ylabel("Count")
        axis.set_xlim(0.0, 1.0)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_motion_prediction_tsne(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str,
) -> None:
    """Plot probability-space t-SNE for each evaluation dataset."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["evaluation_id"]), []).append(row)
    if not grouped:
        raise ValueError("motion prediction t-SNE requires coordinate rows")
    figure, axes = plt.subplots(
        1,
        len(grouped),
        figsize=(max(6.2, 5.8 * len(grouped)), 4.8),
        squeeze=False,
    )
    for axis, (dataset, selected) in zip(
        axes[0], sorted(grouped.items()), strict=True
    ):
        for label, class_name, color in (
            (0, "static", "tab:blue"),
            (1, "motion", "tab:orange"),
        ):
            class_rows = [
                row for row in selected if int(row["true_label"]) == label
            ]
            axis.scatter(
                [float(row["tsne_x"]) for row in class_rows],
                [float(row["tsne_y"]) for row in class_rows],
                s=18,
                alpha=0.65,
                color=color,
                label=f"{class_name} (n={len(class_rows)})",
            )
        incorrect = [
            row for row in selected if not bool(row["prediction_correct"])
        ]
        if incorrect:
            axis.scatter(
                [float(row["tsne_x"]) for row in incorrect],
                [float(row["tsne_y"]) for row in incorrect],
                s=30,
                marker="x",
                linewidths=0.9,
                color="black",
                label="misclassified",
            )
        axis.set_title(dataset)
        axis.set_xlabel("t-SNE 1")
        axis.set_ylabel("t-SNE 2")
        axis.grid(alpha=0.15)
        axis.legend(fontsize=8)
    figure.suptitle(
        f"{title}\nPersisted prediction probabilities; not hidden features"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_motion_roc_auc_curve(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str,
) -> None:
    """Plot the empirical motion-class ROC curve and annotate its AUC."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in rows
        if row.get("curve") == "one_vs_rest"
        and str(row.get("class_label")) == "1"
    ]
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in selected:
        grouped.setdefault(str(row["evaluation_id"]), []).append(row)
    if not grouped:
        raise ValueError("motion ROC-AUC plot requires both reference classes")
    figure, axes = plt.subplots(
        1,
        len(grouped),
        figsize=(max(6.2, 5.5 * len(grouped)), 4.8),
        squeeze=False,
    )
    for axis, (dataset, points) in zip(
        axes[0], sorted(grouped.items()), strict=True
    ):
        points = sorted(points, key=lambda row: int(row["point_index"]))
        curve_auc = float(points[0]["roc_auc"])
        axis.plot(
            [float(row["false_positive_rate"]) for row in points],
            [float(row["true_positive_rate"]) for row in points],
            linewidth=2.0,
            color="tab:blue",
            label=f"motion ROC (AUC={curve_auc:.3f})",
        )
        axis.plot([0, 1], [0, 1], color="0.45", linestyle=":", linewidth=1.0)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("False-positive rate")
        axis.set_ylabel("True-positive rate")
        axis.set_title(dataset)
        axis.grid(alpha=0.2)
        axis.legend(loc="lower right", fontsize=8)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_detector_metrics(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    """Pair the detector score table with one compact multi-metric figure."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        ("balanced_accuracy", "BA"),
        ("macro_f1", "Macro-F1"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
        ("roc_auc", "ROC AUC"),
        ("pr_auc", "PR AUC"),
        ("worst_fold_balanced_accuracy", "Worst-fold BA"),
    )
    labels = [f"{row['dataset']}\n{row['aggregation_level']}" for row in rows]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.8 / len(metrics)
    figure, axis = plt.subplots(figsize=(max(9.0, len(labels) * 2.2), 5.0))
    for index, (field, label) in enumerate(metrics):
        axis.bar(
            x + (index - (len(metrics) - 1) / 2.0) * width,
            [
                float(row[field]) if row.get(field) is not None else np.nan
                for row in rows
            ],
            width=width,
            label=label,
        )
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Score")
    axis.set_title("Motion detector window- and file-level metrics")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
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


def _motion_learning_curve_rows(
    history_paths: Sequence[Path], root: Path
) -> list[dict[str, Any]]:
    """Materialize the exact rows paired with the learning-curve figure."""

    output: list[dict[str, Any]] = []
    for history_path in sorted(history_paths):
        payload = json.loads(history_path.read_text(encoding="utf-8"))
        for row in payload.get("rows", ()):
            output.append(
                {
                    "history_path": history_path.relative_to(root).as_posix(),
                    "fold_index": payload.get("fold_index"),
                    "final_fit": payload.get("final_fit", False),
                    **dict(row),
                }
            )
    return output


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
    resolved_plan_path = root / "resolved_plan.yaml"
    resolved_plan = yaml.safe_load(resolved_plan_path.read_text(encoding="utf-8"))
    peak_validation = (
        dict(resolved_plan.get("validation", {}))
        if isinstance(resolved_plan, Mapping)
        else {}
    )
    peak_lag_window_s = float(peak_validation.get("lag_window_s", 0.0) or 0.0)
    peak_tolerance_s = float(
        peak_validation.get("beat_tolerance_s", 0.0) or 0.0
    )
    peak_current_contract = (
        peak_lag_window_s == 300.0 and peak_tolerance_s == 0.15
    )
    peak_alignment_id = str(
        peak_validation.get("alignment", "not_persisted")
    )
    peak_matching_id = str(
        peak_validation.get("matching", "not_separately_persisted")
    )
    peak_aggregation_id = str(
        peak_validation.get("aggregation", "not_persisted")
    )
    peak_statistical_contract = peak_validation.get("statistical_comparison")
    peak_registered_statistical_metric_ids: tuple[str, ...] = ()
    if isinstance(peak_statistical_contract, Mapping):
        registered_values = peak_statistical_contract.get("metrics")
        if registered_values is None:
            registered_values = [peak_statistical_contract.get("metric")]
        if isinstance(registered_values, list):
            peak_registered_statistical_metric_ids = tuple(
                str(value)
                for value in registered_values
                if str(value) in _STATIC_PEAK_STATISTICAL_METRICS
            )
    report_config = (
        resolved_plan.get("report", {})
        if isinstance(resolved_plan, Mapping)
        else {}
    )
    file_score_aggregation = str(
        report_config.get("file_score_aggregation", "median")
    )
    participant_cluster_bootstrap_resamples = int(
        report_config.get(
            "participant_cluster_bootstrap_resamples",
            _MOTION_CLUSTER_BOOTSTRAP_RESAMPLES,
        )
    )
    participant_cluster_bootstrap_seed = int(
        report_config.get(
            "participant_cluster_bootstrap_seed",
            _MOTION_CLUSTER_BOOTSTRAP_SEED,
        )
    )
    paired_permutation_resamples = int(
        report_config.get(
            "participant_paired_permutation_resamples",
            _STAGE5_PAIRED_PERMUTATION_RESAMPLES,
        )
    )
    paired_permutation_seed = int(
        report_config.get(
            "participant_paired_permutation_seed",
            _STAGE5_PAIRED_PERMUTATION_SEED,
        )
    )
    detector_inference_reference_model_id = str(
        report_config.get(
            "detector_inference_reference_model_id",
            _DEFAULT_MOTION_DETECTOR_REFERENCE_ID,
        )
    )
    detector_inference_candidate_model_id = str(
        report_config.get(
            "detector_inference_candidate_model_id",
            _DEFAULT_MOTION_DETECTOR_CANDIDATE_ID,
        )
    )
    denoiser_inference_reference_id = str(
        report_config.get(
            "denoiser_inference_reference_id",
            _DEFAULT_DENOISER_REFERENCE_ID,
        )
    )
    if participant_cluster_bootstrap_resamples <= 0:
        raise ValueError(
            "report.participant_cluster_bootstrap_resamples must be positive"
        )
    if participant_cluster_bootstrap_seed < 0:
        raise ValueError(
            "report.participant_cluster_bootstrap_seed must be non-negative"
        )
    if paired_permutation_resamples <= 0 or paired_permutation_seed < 0:
        raise ValueError(
            "report participant-paired permutation controls are invalid"
        )
    if (
        not detector_inference_reference_model_id
        or not detector_inference_candidate_model_id
        or detector_inference_reference_model_id
        == detector_inference_candidate_model_id
        or not denoiser_inference_reference_id
    ):
        raise ValueError("report inference reference/candidate ids are invalid")
    diagnostic_config = ClassificationDiagnosticConfig(
        tsne_random_state=int(
            report_config.get("classification_tsne_random_state", 42)
        ),
        tsne_perplexity=float(
            report_config.get("classification_tsne_perplexity", 30.0)
        ),
        tsne_max_samples=int(
            report_config.get("classification_tsne_max_samples", 2000)
        ),
        roc_macro_grid_points=int(
            report_config.get("classification_roc_macro_grid_points", 201)
        ),
        score_histogram_bins=int(
            report_config.get("classification_score_histogram_bins", 40)
        ),
    )
    configured_figures = set(
        report_config.get(
            "required_detector_figures"
            if manifest["study_type"] == "stage5_pre_motion_ptt"
            else "required_figures",
            (),
        )
    )
    if manifest["study_type"] == "stage5_pre_motion_ptt":
        configured_figures.update(
            report_config.get("denoiser_figures_when_enabled", ())
        )
    if "xlsx" not in set(report_config.get("formats", ())):
        # Completed pre-XLSX studies receive the new report-only modules when
        # their report is rebuilt; training evidence and the resolved plan stay
        # untouched.
        configured_figures.update(
            _STAGE5_DETECTOR_FIGURE_MODULES | _DENOISER_FIGURE_MODULES
            if manifest["study_type"] == "stage5_pre_motion_ptt"
            else _PEAK_FIGURE_MODULES
        )
    if manifest["study_type"] == "stage5_pre_motion_ptt":
        # Report-only migration: completed studies predate the corrected
        # window/file requirement, so rebuilding them must add the current
        # detector modules without altering any training evidence.
        configured_figures.update(_STAGE5_DETECTOR_FIGURE_MODULES)
    tables = root / "tables"
    figures = root / "figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    if manifest["study_type"] == "stage5_pre_motion_ptt":
        for relative in _OBSOLETE_STAGE5_REPORT_RELATIVE_PATHS:
            (root / relative).unlink(missing_ok=True)
            (root / "result_backup" / relative).unlink(missing_ok=True)
    test_component_rows = build_motion_peak_test_component_rows(
        resolved_plan,
        manifest,
        study_root=root,
    )
    _write_csv(tables / "test_components.csv", test_component_rows)
    _strict_json(tables / "test_components.json", test_component_rows)
    profile_rows = reporter_profile_rows(test_component_rows)
    _write_csv(tables / "reporter_profiles.csv", profile_rows)
    _strict_json(tables / "reporter_profiles.json", profile_rows)
    methods_markdown = write_reporter_methods(root, test_component_rows)
    denoiser_algorithm_rows = [
        dict(row)
        for row in test_component_rows
        if row.get("component_role") == "denoiser"
    ]
    if denoiser_algorithm_rows:
        _write_csv(tables / "denoiser_algorithms.csv", denoiser_algorithm_rows)
        _strict_json(tables / "denoiser_algorithms.json", denoiser_algorithm_rows)
    component_markdown = write_test_component_markdown(root, test_component_rows)
    images: list[Path] = []
    summary_rows: list[Mapping[str, Any]] = []
    headline_metric_rows: list[Mapping[str, Any]] = []
    detector_participant_rows: list[dict[str, Any]] = []
    detector_compact_rows: list[dict[str, Any]] = []
    detector_compact_audit_rows: list[dict[str, Any]] = []
    detector_metric_tables: dict[str, list[dict[str, Any]]] = {}
    detector_training_source_inference_rows: list[dict[str, Any]] = []
    detector_worst_fold_rows: list[dict[str, Any]] = []
    detector_internal_rows: list[Mapping[str, Any]] = []
    detector_transfer_rows: list[Mapping[str, Any]] = []
    detector_window_confusion_rows: list[Mapping[str, Any]] = []
    detector_file_confusion_rows: list[Mapping[str, Any]] = []
    detector_file_prediction_rows: list[Mapping[str, Any]] = []
    detector_score_distribution_rows: list[Mapping[str, Any]] = []
    detector_prediction_rows: list[Mapping[str, Any]] = []
    detector_roc_curve_rows: list[Mapping[str, Any]] = []
    detector_tsne_rows: list[Mapping[str, Any]] = []
    detector_diagnostic_status_rows: list[Mapping[str, Any]] = []
    detector_per_class_rows: list[Mapping[str, Any]] = []
    detector_per_class_performance_rows: list[dict[str, Any]] = []
    detector_per_class_discrimination_rows: list[dict[str, Any]] = []
    reporter_output_status_rows: list[Mapping[str, Any]] = []
    denoiser_activity_tables: dict[str, list[dict[str, Any]]] = {}
    denoiser_compact_rows: list[dict[str, Any]] = []
    denoiser_compact_audit_rows: list[dict[str, Any]] = []
    denoiser_inference_rows: list[dict[str, Any]] = []
    denoiser_coverage_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    static_peak_statistical_rows: list[dict[str, Any]] = []
    static_peak_statistical_display_rows: list[dict[str, Any]] = []
    static_peak_effect_display_rows: list[dict[str, Any]] = []
    static_peak_inference_display_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
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
            tuple[str, str, str, Sequence[Mapping[str, Any]]]
        ] = [
            (
                "frailty29_trained_motion_detector",
                "frailty29_outer_oof",
                "source_grouped_oof",
                internal_rows,
            ),
            (
                "frailty29_trained_motion_detector",
                "frailty29_trained_to_ptt22",
                "frozen_cross_dataset",
                external_rows,
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
        motion_plan = dict(resolved_plan.get("motion_detector", {}))
        internal_split = dict(motion_plan.get("split", {}))
        reverse_plan = dict(motion_plan.get("reverse_ablation", {}))
        reproducibility_rows = [
            {
                "evidence_scope": "frailty29_source_grouped_oof",
                "split_method": internal_split.get("method", "N/A"),
                "group_unit": internal_split.get("groups", "N/A"),
                "fold_count": internal_split.get("folds"),
                "repeat_count": internal_split.get("repeats"),
                "resample_count": None,
                "split_seed": internal_split.get("seed"),
                "resampling_seed": None,
                "fit_or_recalibration_on_target": False,
                "status": "generated_from_resolved_plan",
            },
            {
                "evidence_scope": "frailty29_final_to_ptt22_frozen_transfer",
                "split_method": "N/A",
                "group_unit": "participant_id",
                "fold_count": None,
                "repeat_count": 1,
                "resample_count": None,
                "split_seed": None,
                "resampling_seed": None,
                "fit_or_recalibration_on_target": bool(
                    motion_plan.get("external_fit_or_recalibration", False)
                ),
                "status": "generated_from_resolved_plan",
            },
            {
                "evidence_scope": "ptt22_reverse_ablation",
                "split_method": reverse_plan.get("split_registry", "N/A"),
                "group_unit": "participant_id",
                "fold_count": len(reverse_plan.get("folds", ())),
                "repeat_count": len(reverse_plan.get("repeat_indices", ())),
                "resample_count": None,
                "split_seed": reverse_plan.get("split_seed"),
                "resampling_seed": None,
                "fit_or_recalibration_on_target": bool(
                    reverse_plan.get("evaluation_fit_or_recalibration", False)
                ),
                "status": (
                    "executed" if reverse_available else "N/A"
                ),
                "status_reason": (
                    "" if reverse_available else "reverse_motion_ablation_not_executed"
                ),
            },
            {
                "evidence_scope": "participant_cluster_bootstrap_report",
                "split_method": (
                    "participant_cluster_percentile_bootstrap_two_sided_95"
                ),
                "group_unit": "participant_id",
                "fold_count": None,
                "repeat_count": None,
                "resample_count": participant_cluster_bootstrap_resamples,
                "split_seed": None,
                "resampling_seed": participant_cluster_bootstrap_seed,
                "fit_or_recalibration_on_target": False,
                "status": "generated_from_resolved_report_config",
            },
            {
                "evidence_scope": "participant_paired_sign_flip_report",
                "split_method": (
                    "two_sided_participant_paired_monte_carlo_sign_flip_plus_one"
                ),
                "group_unit": "participant_id",
                "fold_count": None,
                "repeat_count": None,
                "resample_count": paired_permutation_resamples,
                "split_seed": None,
                "resampling_seed": paired_permutation_seed,
                "fit_or_recalibration_on_target": False,
                "status": (
                    "generated_from_resolved_report_config"
                    if {
                        "participant_paired_permutation_resamples",
                        "participant_paired_permutation_seed",
                    }.issubset(report_config)
                    else "generated_from_versioned_reporter_default"
                ),
            },
        ]
        _write_csv(
            tables / "reproducibility_summary.csv", reproducibility_rows
        )
        _strict_json(
            tables / "reproducibility_summary.json", reproducibility_rows
        )
        inference_configuration_rows = [
            {
                "analysis": "motion_detector_training_source",
                "reference": detector_inference_reference_model_id,
                "candidate_or_family": detector_inference_candidate_model_id,
                "paired_resamples": paired_permutation_resamples,
                "seed": paired_permutation_seed,
                "multiplicity": (
                    "Holm_within_target_x_level_across_registered_endpoints"
                ),
            },
            {
                "analysis": "denoiser_reducer_comparison",
                "reference": denoiser_inference_reference_id,
                "candidate_or_family": "all_non_reference_reducers",
                "paired_resamples": paired_permutation_resamples,
                "seed": paired_permutation_seed,
                "multiplicity": (
                    "Holm_within_activity_x_channel_x_endpoint_across_reducers"
                ),
            },
        ]
        _write_csv(
            tables / "inference_configuration.csv",
            inference_configuration_rows,
        )
        _strict_json(
            tables / "inference_configuration.json",
            inference_configuration_rows,
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
                        "ptt22_trained_motion_detector",
                        "ptt22_outer_oof",
                        "source_grouped_oof",
                        reverse_training_rows,
                    ),
                    (
                        "ptt22_trained_motion_detector",
                        "ptt22_trained_to_frailty29",
                        "frozen_cross_dataset",
                        internal_reverse_rows,
                    ),
                ]
            )
        headline_metric_rows, detector_file_prediction_rows = _detector_report_rows(
            detector_datasets,
            file_score_aggregation=file_score_aggregation,
            participant_cluster_bootstrap_resamples=(
                participant_cluster_bootstrap_resamples
            ),
            participant_cluster_bootstrap_seed=participant_cluster_bootstrap_seed,
        )
        detector_participant_rows = _motion_participant_metric_rows(
            detector_datasets,
            file_score_aggregation=file_score_aggregation,
        )
        detector_training_source_inference_rows = (
            _motion_training_source_inference_rows(
                detector_datasets,
                detector_participant_rows,
                file_score_aggregation=file_score_aggregation,
                bootstrap_resamples=participant_cluster_bootstrap_resamples,
                bootstrap_seed=participant_cluster_bootstrap_seed,
                permutation_resamples=paired_permutation_resamples,
                permutation_seed=paired_permutation_seed,
                reference_model_id=detector_inference_reference_model_id,
                candidate_model_id=detector_inference_candidate_model_id,
            )
            if reverse_available
            else []
        )
        detector_compact_rows, detector_compact_audit_rows = (
            _motion_compact_metric_rows(
                detector_participant_rows,
                detector_training_source_inference_rows,
                bootstrap_resamples=participant_cluster_bootstrap_resamples,
                bootstrap_seed=participant_cluster_bootstrap_seed,
            )
        )
        detector_worst_fold_rows = _motion_worst_fold_rows(
            headline_metric_rows
        )
        _write_csv(tables / "motion_detector_metrics.csv", detector_compact_rows)
        _strict_json(tables / "motion_detector_metrics.json", headline_metric_rows)
        _strict_json(
            tables / "motion_detector_participant_macro_statistics.json",
            detector_compact_audit_rows,
        )
        _strict_json(
            tables / "motion_detector_training_source_inference.json",
            detector_training_source_inference_rows,
        )
        detector_inference_display_rows = [
            {
                "reference_model": row["reference_model_id"],
                "candidate_model": row["candidate_model_id"],
                "target": row["target_dataset"],
                "level": row["aggregation_level"],
                "metric": _MOTION_DETECTOR_METRIC_LABELS[str(row["metric"])],
                "delta_ci95": format_interval(
                    row["paired_participant_ci95_low"],
                    row["paired_participant_ci95_high"],
                    percent=True,
                ),
                "raw_p": _format_probability(float(row["raw_p_value"])),
                "holm_p": _format_probability(
                    float(row["holm_adjusted_p_value"])
                ),
            }
            for row in detector_training_source_inference_rows
        ]
        if not detector_inference_display_rows:
            detector_inference_display_rows = [
                {
                    "reference_model": detector_inference_reference_model_id,
                    "candidate_model": detector_inference_candidate_model_id,
                    "target": "N/A",
                    "level": "N/A",
                    "metric": "N/A_no_identical_target_model_pair",
                    "delta_ci95": "N/A",
                    "raw_p": "N/A",
                    "holm_p": "N/A",
                }
            ]
        _write_csv(
            tables / "motion_detector_training_source_inference.csv",
            detector_inference_display_rows,
        )
        _write_csv(
            tables / "motion_detector_worst_fold_ba.csv",
            detector_worst_fold_rows,
        )
        _strict_json(
            tables / "motion_detector_worst_fold_ba.json",
            detector_worst_fold_rows,
        )
        detector_participant_long_rows = [
            {
                "model_id": row["model_id"],
                "dataset": row["dataset"],
                "scope": row["evaluation_scope"],
                "level": row["aggregation_level"],
                "participant_id": row["participant_id"],
                "metric": metric,
                "value": row[metric],
            }
            for row in detector_participant_rows
            for metric in _MOTION_DETECTOR_METRICS
        ]
        _write_csv(
            tables / "motion_detector_participant_metrics_raw.csv",
            detector_participant_long_rows,
        )
        _strict_json(
            tables / "motion_detector_participant_metrics_raw.json",
            detector_participant_long_rows,
        )
        for metric in _MOTION_DETECTOR_METRICS:
            metric_rows = [
                {
                    key: value
                    for key, value in row.items()
                    if key != "metric"
                }
                for row in detector_compact_rows
                if row["metric"] == _MOTION_DETECTOR_METRIC_LABELS[metric]
            ]
            table_id = _MOTION_DETECTOR_METRIC_TABLE_IDS[metric]
            detector_metric_tables[metric] = metric_rows
            _write_csv(tables / f"{table_id}.csv", metric_rows)
            _strict_json(tables / f"{table_id}.json", metric_rows)
        _write_csv(
            tables / "motion_detector_file_predictions.csv",
            detector_file_prediction_rows,
        )
        _strict_json(
            tables / "motion_detector_file_predictions.json",
            detector_file_prediction_rows,
        )
        detector_metric_figure = figures / "motion_detector_metrics.png"
        if detector_metric_figure.stem in configured_figures:
            _plot_detector_metrics(detector_metric_figure, headline_metric_rows)
            images.append(detector_metric_figure)
        detector_internal_rows = [
            row for row in headline_metric_rows
            if row["evaluation_scope"] == "source_grouped_oof"
        ]
        detector_transfer_rows = [
            row for row in headline_metric_rows
            if row["evaluation_scope"] == "frozen_cross_dataset"
        ]
        _write_csv(
            tables / "motion_detector_internal_evaluation.csv",
            [
                row
                for row in detector_compact_rows
                if "(source_grouped_oof)" in str(row["evaluation"])
            ],
        )
        _strict_json(
            tables / "motion_detector_internal_evaluation.json",
            detector_internal_rows,
        )
        _write_csv(
            tables / "motion_detector_cross_dataset_evaluation.csv",
            [
                row
                for row in detector_compact_rows
                if "(frozen_cross_dataset)" in str(row["evaluation"])
            ],
        )
        _strict_json(
            tables / "motion_detector_cross_dataset_evaluation.json",
            detector_transfer_rows,
        )
        for aggregation_level in ("window", "file"):
            detector_score_distribution_rows.extend(
                _score_distribution_rows(
                    detector_datasets,
                    aggregation_level=aggregation_level,
                    file_score_aggregation=file_score_aggregation,
                )
            )
        _write_csv(
            tables / "motion_detector_score_distributions.csv",
            detector_score_distribution_rows,
        )
        _strict_json(
            tables / "motion_detector_score_distributions.json",
            detector_score_distribution_rows,
        )
        for model_id, dataset, _scope, window_rows in detector_datasets:
            for aggregation_level in ("window", "file"):
                level_rows = (
                    list(window_rows)
                    if aggregation_level == "window"
                    else _file_prediction_rows(
                        window_rows,
                        score_aggregation=file_score_aggregation,
                    )
                )
                detector_prediction_rows.extend(
                    normalize_classification_rows(
                        level_rows,
                        classifier_id=model_id,
                        evaluation_id=dataset,
                        aggregation_level=aggregation_level,
                        label_field="activity_label",
                    )
                )
        detector_per_class_rows = list(
            classification_per_class_metric_rows(
                detector_prediction_rows,
                class_names={0: "static", 1: "motion"},
            )
        )
        (
            detector_per_class_performance_rows,
            detector_per_class_discrimination_rows,
        ) = _motion_per_class_table_rows(detector_per_class_rows)
        detector_roc_curve_rows = list(
            classification_roc_curve_rows(
                detector_prediction_rows,
                macro_grid_points=diagnostic_config.roc_macro_grid_points,
            )
        )
        detector_tsne_rows = list(
            classification_tsne_rows(
                detector_prediction_rows,
                random_state=diagnostic_config.tsne_random_state,
                perplexity=diagnostic_config.tsne_perplexity,
                max_samples=diagnostic_config.tsne_max_samples,
            )
        )
        expected_detector_ids = [
            "frailty29_trained_motion_detector",
            *(
                ["ptt22_trained_motion_detector"]
                if reverse_available
                else []
            ),
        ]
        detector_diagnostic_status_rows = list(
            classification_diagnostic_status_rows(
                expected_detector_ids,
                detector_prediction_rows,
                detector_roc_curve_rows,
                detector_tsne_rows,
            )
        )
        for table_name, table_rows in (
            ("motion_detector_prediction_scores", detector_prediction_rows),
            ("motion_detector_per_class_results", detector_per_class_rows),
            (
                "motion_detector_per_class_performance",
                detector_per_class_performance_rows,
            ),
            (
                "motion_detector_per_class_discrimination",
                detector_per_class_discrimination_rows,
            ),
            ("motion_detector_roc_curves", detector_roc_curve_rows),
            ("motion_detector_prediction_tsne", detector_tsne_rows),
            (
                "motion_detector_diagnostic_status",
                detector_diagnostic_status_rows,
            ),
        ):
            csv_rows = (
                detector_per_class_performance_rows
                if table_name == "motion_detector_per_class_results"
                else table_rows
            )
            _write_csv(tables / f"{table_name}.csv", csv_rows)
            _strict_json(tables / f"{table_name}.json", table_rows)
        score_plot_specs = [
            (
                "frailty29_trained_motion_detector",
                "frailty29_trained",
                "Frailty29-trained motion detector",
            )
        ]
        if reverse_available:
            score_plot_specs.append(
                (
                    "ptt22_trained_motion_detector",
                    "ptt22_trained",
                    "PTT22-trained motion detector",
                )
            )
        for model_id, file_prefix, plot_title in score_plot_specs:
            model_datasets = [
                (dataset, rows)
                for candidate, dataset, _scope, rows in detector_datasets
                if candidate == model_id
            ]
            for aggregation_level in ("window", "file"):
                path = figures / (
                    f"{file_prefix}_{aggregation_level}_score_distribution.png"
                )
                if path.stem in configured_figures:
                    _plot_score_distribution(
                        path,
                        model_datasets,
                        aggregation_level=aggregation_level,
                        file_score_aggregation=file_score_aggregation,
                        score_histogram_bins=(
                            diagnostic_config.score_histogram_bins
                        ),
                        title=f"{plot_title} · {aggregation_level}-level scores",
                    )
                    images.append(path)
                tsne_path = figures / (
                    f"{file_prefix}_{aggregation_level}_prediction_tsne.png"
                )
                if tsne_path.stem in configured_figures:
                    selected_tsne = [
                        row
                        for row in detector_tsne_rows
                        if row["classifier_id"] == model_id
                        and row["aggregation_level"] == aggregation_level
                    ]
                    _plot_motion_prediction_tsne(
                        tsne_path,
                        selected_tsne,
                        title=(
                            f"{plot_title} · {aggregation_level}-level "
                            "prediction-space t-SNE"
                        ),
                    )
                    images.append(tsne_path)
                roc_path = figures / (
                    f"{file_prefix}_{aggregation_level}_roc_auc_curve.png"
                )
                if roc_path.stem in configured_figures:
                    selected_roc = [
                        row
                        for row in detector_roc_curve_rows
                        if row["classifier_id"] == model_id
                        and row["aggregation_level"] == aggregation_level
                    ]
                    _plot_motion_roc_auc_curve(
                        roc_path,
                        selected_roc,
                        title=(
                            f"{plot_title} · {aggregation_level}-level ROC–AUC"
                        ),
                    )
                    images.append(roc_path)
        for model_id, dataset, name, file_name, rows, title in (
            (
                "frailty29_trained_motion_detector",
                "frailty29_outer_oof",
                "motion_internal_confusion_matrix.png",
                "motion_internal_file_confusion_matrix.png",
                internal_rows,
                "Internal 29-person OOF",
            ),
            (
                "frailty29_trained_motion_detector",
                "frailty29_trained_to_ptt22",
                "motion_ptt_confusion_matrix.png",
                "motion_ptt_file_confusion_matrix.png",
                external_rows,
                "PTT external evaluation",
            ),
            *(
                (
                    (
                        "ptt22_trained_motion_detector",
                        "ptt22_outer_oof",
                        "motion_ptt_training_oof_confusion_matrix.png",
                        "motion_ptt_training_oof_file_confusion_matrix.png",
                        reverse_training_rows,
                        "PTT 22-person training OOF",
                    ),
                    (
                        "ptt22_trained_motion_detector",
                        "ptt22_trained_to_frailty29",
                        "motion_internal_reverse_confusion_matrix.png",
                        "motion_internal_reverse_file_confusion_matrix.png",
                        internal_reverse_rows,
                        "PTT-trained frozen model on Frailty29",
                    ),
                )
                if reverse_available
                else ()
            ),
        ):
            path = figures / name
            window_matrix = _confusion(rows)
            detector_window_confusion_rows.append(
                _confusion_report_row(
                    model_id,
                    dataset,
                    window_matrix,
                    aggregation_level="window",
                )
            )
            if path.stem in configured_figures:
                _plot_confusion(path, window_matrix, title)
                images.append(path)
            file_rows = _file_prediction_rows(
                rows, score_aggregation=file_score_aggregation
            )
            file_matrix = _confusion(file_rows)
            detector_file_confusion_rows.append(
                _confusion_report_row(
                    model_id,
                    dataset,
                    file_matrix,
                    aggregation_level=(
                        f"file_{file_score_aggregation}_window_probability"
                    ),
                )
            )
            file_path = figures / file_name
            if file_path.stem in configured_figures:
                _plot_confusion(
                    file_path,
                    file_matrix,
                    f"{title} · file median probability",
                )
                images.append(file_path)
        _write_csv(
            tables / "motion_detector_window_confusion.csv",
            detector_window_confusion_rows,
        )
        _strict_json(
            tables / "motion_detector_window_confusion.json",
            detector_window_confusion_rows,
        )
        _write_csv(
            tables / "motion_detector_file_confusion.csv",
            detector_file_confusion_rows,
        )
        _strict_json(
            tables / "motion_detector_file_confusion.json",
            detector_file_confusion_rows,
        )
        learning_path = figures / "motion_training_learning_curves.png"
        history_paths = tuple(internal_dir.rglob("motion_training_history.json"))
        if ptt_training_dir is not None:
            history_paths += tuple(
                ptt_training_dir.rglob("motion_training_history.json")
            )
        if not history_paths:
            raise ValueError("completed Stage5-pre report requires motion training histories")
        if learning_path.stem in configured_figures:
            _plot_motion_learning_curves(learning_path, history_paths)
            images.append(learning_path)
        motion_history_rows = _motion_learning_curve_rows(history_paths, root)
        _write_csv(tables / "motion_training_history.csv", motion_history_rows)
        _strict_json(tables / "motion_training_history.json", motion_history_rows)
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
            summary_rows = (
                _aggregate_benchmark(benchmark["rows"])
                if benchmark.get("rows")
                else benchmark["summary_rows"]
            )
            summary_rows = _annotate_denoiser_uncertainty(summary_rows)
            _strict_json(tables / "denoiser_summary.json", summary_rows)
            if benchmark.get("rows"):
                denoiser_participant_rows = (
                    _denoiser_participant_endpoint_rows(benchmark["rows"])
                )
                denoiser_inference_rows = _denoiser_inference_rows(
                    benchmark["rows"],
                    bootstrap_resamples=participant_cluster_bootstrap_resamples,
                    bootstrap_seed=participant_cluster_bootstrap_seed,
                    permutation_resamples=paired_permutation_resamples,
                    permutation_seed=paired_permutation_seed,
                    reference_id=denoiser_inference_reference_id,
                )
                denoiser_compact_rows, denoiser_compact_audit_rows = (
                    _denoiser_compact_metric_rows(
                        denoiser_participant_rows,
                        denoiser_inference_rows,
                        bootstrap_resamples=(
                            participant_cluster_bootstrap_resamples
                        ),
                        bootstrap_seed=participant_cluster_bootstrap_seed,
                        reference_id=denoiser_inference_reference_id,
                    )
                )
            else:
                for row in summary_rows:
                    for metric, (
                        label,
                        percent,
                        _direction,
                    ) in _DENOISER_PRIMARY_METRICS.items():
                        denoiser_compact_rows.append(
                            {
                                "denoiser": row["algorithm_or_reducer"],
                                "activity": row["activity_group"],
                                "channel": row["channel"],
                                "metric": label,
                                "mean_sd": format_mean_sd(
                                    row.get(metric),
                                    row.get(f"{metric}_sd"),
                                    percent=percent,
                                ),
                                "participant_bootstrap_ci95": "N/A",
                                "holm_p_vs_reference": (
                                    "Reference"
                                    if row["algorithm_or_reducer"]
                                    == denoiser_inference_reference_id
                                    else "N/A"
                                ),
                            }
                        )
            denoiser_coverage_rows = _denoiser_coverage_rows(summary_rows)
            _write_csv(tables / "denoiser_summary.csv", denoiser_compact_rows)
            _strict_json(
                tables / "denoiser_compact_statistics.json",
                denoiser_compact_audit_rows,
            )
            _strict_json(
                tables / "denoiser_paired_inference.json",
                denoiser_inference_rows,
            )
            denoiser_inference_display = [
                {
                    "reference": row["reference_denoiser"],
                    "candidate": row["candidate_denoiser"],
                    "activity": row["activity_group"],
                    "channel": row["channel"],
                    "metric": _DENOISER_PRIMARY_METRICS[str(row["metric"])][0],
                    "delta_ci95": format_interval(
                        row["paired_participant_ci95_low"],
                        row["paired_participant_ci95_high"],
                        percent=_DENOISER_PRIMARY_METRICS[
                            str(row["metric"])
                        ][1],
                    ),
                    "raw_p": _format_probability(float(row["raw_p_value"])),
                    "holm_p": _format_probability(
                        float(row["holm_adjusted_p_value"])
                    ),
                }
                for row in denoiser_inference_rows
            ]
            if not denoiser_inference_display:
                denoiser_inference_display = [
                    {
                        "reference": denoiser_inference_reference_id,
                        "candidate": "N/A",
                        "activity": "N/A",
                        "channel": "N/A",
                        "metric": "N/A_no_raw_matched_segment_evidence",
                        "delta_ci95": "N/A",
                        "raw_p": "N/A",
                        "holm_p": "N/A",
                    }
                ]
            _write_csv(
                tables / "denoiser_paired_inference.csv",
                denoiser_inference_display,
            )
            _write_csv(
                tables / "denoiser_coverage.csv",
                denoiser_coverage_rows,
            )
            _strict_json(
                tables / "denoiser_coverage.json",
                denoiser_coverage_rows,
            )
            for activity_group in ("static", "dynamic"):
                selected = _denoiser_activity_result_rows(
                    summary_rows,
                    denoiser_inference_rows,
                    activity_group=activity_group,
                    reference_id=denoiser_inference_reference_id,
                )
                denoiser_activity_tables[activity_group] = selected
                _write_csv(
                    tables / f"denoiser_{activity_group}.csv", selected
                )
                _strict_json(
                    tables / f"denoiser_{activity_group}.json", selected
                )
            for metric, name, title in (
                (
                    "participant_macro_ibi_ppi_rmse_ms",
                    "denoiser_interval_rmse.png",
                    "IBI–PPI RMSE (lower is better)",
                ),
                ("participant_macro_f1", "denoiser_beat_f1.png", "Delay-aligned beat F1"),
                (
                    "participant_macro_sensitivity",
                    "denoiser_beat_sensitivity.png",
                    "Delay-aligned beat sensitivity",
                ),
                (
                    "participant_macro_positive_predictive_value",
                    "denoiser_beat_ppv.png",
                    "Delay-aligned beat positive predictive value",
                ),
                ("total_runtime_s", "denoiser_runtime.png", "Reducer + detector runtime"),
            ):
                path = figures / name
                if path.stem in configured_figures:
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
            "detector_report_rows": headline_metric_rows,
            "detector_window_confusion_rows": detector_window_confusion_rows,
            "detector_file_confusion_rows": detector_file_confusion_rows,
            "detector_score_distribution_rows": detector_score_distribution_rows,
            "detector_diagnostic_status_rows": detector_diagnostic_status_rows,
            "detector_per_class_rows": detector_per_class_rows,
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
        recording_rows = (
            [dict(row) for row in result.get("rows", ())]
            if result.get("schema_version")
            == "ppg_frailty.stage_ablation_01_static_peak_result.v3"
            else []
        )
        if recording_rows:
            distribution_rows = _aggregate_static_peak_benchmark(recording_rows)
            summary_rows = _static_peak_display_rows(distribution_rows)
            if isinstance(peak_statistical_contract, Mapping):
                statistical = dict(
                    resolved_plan["validation"]["statistical_comparison"]
                )
                static_peak_statistical_rows = _static_peak_rank_sum_comparisons(
                    recording_rows,
                    reference_algorithm_id=str(
                        statistical["reference_algorithm_id"]
                    ),
                    alpha=float(statistical["alpha"]),
                    metric_ids=_STATIC_PEAK_STATISTICAL_METRIC_IDS,
                    registered_metric_ids=peak_registered_statistical_metric_ids,
                )
            else:
                static_peak_statistical_rows = [
                    dict(row)
                    for row in result.get("statistical_comparisons", ())
                ]
            static_peak_statistical_display_rows = (
                _static_peak_statistical_display_rows(
                    static_peak_statistical_rows
                )
            )
            static_peak_effect_display_rows = [
                {
                    key: row.get(key)
                    for key in (
                        "endpoint",
                        "channel",
                        "registration",
                        "common subject-recordings",
                        "MSPTDfast median",
                        "aboy_project median",
                        "unit",
                        "MSPTDfast advantage",
                    )
                }
                for row in static_peak_statistical_display_rows
            ]
            static_peak_inference_display_rows = [
                {
                    key: row.get(key)
                    for key in (
                        "endpoint",
                        "channel",
                        "registration",
                        "rank-sum z",
                        "raw p",
                        "Holm–Sidak adjusted p (global family)",
                        "reject at alpha=0.05",
                    )
                }
                for row in static_peak_statistical_display_rows
            ]
            _write_csv(
                tables / "static_peak_detector_recording_metrics.csv",
                recording_rows,
            )
            _strict_json(
                tables / "static_peak_detector_recording_metrics.json",
                recording_rows,
            )
            _write_csv(
                tables / "static_peak_detector_distribution_statistics.csv",
                distribution_rows,
            )
            _strict_json(
                tables / "static_peak_detector_distribution_statistics.json",
                distribution_rows,
            )
            _write_csv(
                tables / "static_peak_detector_rank_sum_holm_sidak.csv",
                static_peak_statistical_rows,
            )
            _strict_json(
                tables / "static_peak_detector_rank_sum_holm_sidak.json",
                static_peak_statistical_rows,
            )
            _write_csv(
                tables / "static_peak_detector_significance_summary.csv",
                static_peak_inference_display_rows,
            )
            _strict_json(
                tables / "static_peak_detector_significance_summary.json",
                static_peak_statistical_display_rows,
            )
            _write_csv(
                tables / "static_peak_detector_endpoint_effects.csv",
                static_peak_effect_display_rows,
            )
            _strict_json(
                tables / "static_peak_detector_endpoint_effects.json",
                static_peak_effect_display_rows,
            )
        else:
            summary_rows = result["summary_rows"]
        _write_csv(tables / "static_peak_detector_summary.csv", summary_rows)
        _strict_json(tables / "static_peak_detector_summary.json", summary_rows)
        plot_metrics = (
            (
                "f1_percent",
                "static_peak_detector_f1.png",
                "Static PTT recording beat F1 (%)",
            ),
            (
                "sensitivity_percent",
                "static_peak_detector_sensitivity.png",
                "Static PTT recording beat sensitivity (%)",
            ),
            (
                "positive_predictive_value_percent",
                "static_peak_detector_ppv.png",
                "Static PTT recording beat positive predictive value (%)",
            ),
            (
                "ibi_ppi_rmse_ms",
                "static_peak_detector_interval_rmse.png",
                "Static recording IBI–PPI RMSE (ms)",
            ),
            (
                "execution_time_percent",
                "static_peak_detector_runtime.png",
                "Execution time as percentage of PPG duration",
            ),
        ) if recording_rows else (
            ("participant_macro_f1", "static_peak_detector_f1.png", "Static PTT beat F1"),
            ("participant_macro_sensitivity", "static_peak_detector_sensitivity.png", "Static PTT beat sensitivity"),
            ("participant_macro_positive_predictive_value", "static_peak_detector_ppv.png", "Static PTT beat positive predictive value"),
            ("participant_macro_ibi_ppi_rmse_ms", "static_peak_detector_interval_rmse.png", "Static IBI-PPI RMSE"),
            ("total_runtime_s", "static_peak_detector_runtime.png", "Static detector runtime"),
        )
        for metric, name, title in plot_metrics:
            path = figures / name
            if path.stem in configured_figures:
                if recording_rows:
                    _plot_static_peak_boxplot(
                        path, recording_rows, metric, title
                    )
                else:
                    _plot_summary(path, summary_rows, metric, title)
                images.append(path)
        headline = {
            "static_peak_detector_summary_rows": summary_rows,
            "static_peak_detector_statistical_comparisons":
                static_peak_statistical_rows,
        }
    if manifest["study_type"] == "stage5_pre_motion_ptt":
        interpretation_comparison_audit_rows = [
            {"evidence_type": "motion_detector", **dict(row)}
            for row in headline_metric_rows
        ] + [
            {"evidence_type": "denoiser", **dict(row)} for row in summary_rows
        ]
        interpretation_comparison_rows = [
            {
                "evidence_type": "motion_detector",
                "model_or_module": row["model_id"],
                "evaluation": row["evaluation"],
                "stratum": row["level"],
                "metric": row["metric"],
                "mean_sd": row["participant_macro_mean_sd"],
                "ci95": row["participant_bootstrap_ci95"],
                "p": row["holm_p_vs_reference"],
            }
            for row in detector_compact_rows
        ] + [
            {
                "evidence_type": "denoiser",
                "model_or_module": row["denoiser"],
                "evaluation": row["activity"],
                "stratum": row["channel"],
                "metric": row["metric"],
                "mean_sd": row["mean_sd"],
                "ci95": row["participant_bootstrap_ci95"],
                "p": row["holm_p_vs_reference"],
            }
            for row in denoiser_compact_rows
        ]
        interpretation_conclusion_rows = _motion_detector_conclusion_rows(
            headline_metric_rows,
            denoiser_enabled=denoiser_enabled,
        )
        if detector_training_source_inference_rows:
            detector_rejections = sum(
                bool(row["reject_after_holm"])
                for row in detector_training_source_inference_rows
            )
            interpretation_conclusion_rows.append(
                {
                    "angle": "motion_detector_training_source_inference",
                    "target_dataset": "frailty29_and_ptt22",
                    "evaluation_scope": (
                        "asymmetric_grouped_oof_vs_frozen_cross_dataset"
                    ),
                    "aggregation_level": "window_and_file_kept_separate",
                    "within_stratum_candidate_count": 2,
                    "leading_or_selected_case": (
                        detector_inference_candidate_model_id
                    ),
                    "finding": (
                        f"{detector_rejections}/"
                        f"{len(detector_training_source_inference_rows)} "
                        "endpoint comparisons reject after Holm correction. "
                        "The analysis is participant-paired on identical target "
                        "units but retrospective and asymmetric; it does not "
                        "select a deployment model automatically."
                    ),
                    "confidence": "retrospective_exploratory",
                    "selection_effect": "none_automatic",
                }
            )
        if denoiser_inference_rows:
            denoiser_rejections = sum(
                bool(row["reject_after_holm"])
                for row in denoiser_inference_rows
            )
            interpretation_conclusion_rows.append(
                {
                    "angle": "denoiser_reference_inference",
                    "target_dataset": "ptt22",
                    "evaluation_scope": "common_successful_segments",
                    "aggregation_level": "participant_paired",
                    "within_stratum_candidate_count": None,
                    "leading_or_selected_case": None,
                    "finding": (
                        f"{denoiser_rejections}/"
                        f"{len(denoiser_inference_rows)} reducer-endpoint "
                        "comparisons reject after Holm correction versus "
                        f"{denoiser_inference_reference_id}. Reference choice and "
                        "tests are retrospective exploratory supplements."
                    ),
                    "confidence": "retrospective_exploratory",
                    "selection_effect": "none_automatic",
                }
            )
    else:
        interpretation_comparison_audit_rows = [
            {"evidence_type": "peak_detector", **dict(row)}
            for row in (distribution_rows or summary_rows)
        ] + [
            {"evidence_type": "recording_rank_sum_endpoint_test", **dict(row)}
            for row in static_peak_statistical_rows
        ]
        static_peak_summary_metrics = (
            ("Beat F1", "F1 %, median [Q1, Q3]"),
            ("Sensitivity", "Sensitivity %, median [Q1, Q3]"),
            ("Positive predictive value", "PPV %, median [Q1, Q3]"),
            ("IBI-PPI RMSE", "IBI-PPI RMSE ms, median [Q1, Q3]"),
            ("Execution time", "Execution time %, median [Q1, Q3]"),
        )
        interpretation_comparison_rows = [
            {
                "evidence_type": "peak_detector",
                "model_or_module": row.get(
                    "algorithm", row.get("algorithm_or_reducer")
                ),
                "evaluation": "PTT sit subject-recordings",
                "stratum": row.get("channel"),
                "metric": metric_label,
                "estimate": row.get(field, "N/A"),
                "ci95": "N/A — distribution reported as median [Q1, Q3]",
                "p": "see static_peak_detector_significance_summary",
            }
            for row in summary_rows
            for metric_label, field in static_peak_summary_metrics
        ]
        if distribution_rows:
            top_peak = max(
                distribution_rows,
                key=lambda row: float(row.get("recording_f1_percent_median", -math.inf)),
            )
            top_f1 = float(top_peak["recording_f1_percent_median"])
            metric_contract = (
                "recording median F1 under the persisted "
                f"{peak_lag_window_s:g} s/±{1000.0*peak_tolerance_s:g} ms contract"
            )
        else:
            top_peak = max(
                summary_rows,
                key=lambda row: float(row.get("participant_macro_f1", -math.inf)),
            )
            top_f1 = 100.0 * float(top_peak["participant_macro_f1"])
            metric_contract = "historical participant-macro F1 under persisted legacy validation"
        significant = sum(
            bool(row.get("reject_at_alpha")) for row in static_peak_statistical_rows
        )
        endpoint_rejections = {
            metric_id: (
                sum(
                    bool(row.get("reject_at_alpha"))
                    for row in static_peak_statistical_rows
                    if row.get("metric") == metric_id
                ),
                sum(
                    1
                    for row in static_peak_statistical_rows
                    if row.get("metric") == metric_id
                ),
            )
            for metric_id in _STATIC_PEAK_STATISTICAL_METRIC_IDS
        }
        endpoint_rejection_text = "; ".join(
            f"{metric_id} {rejected}/{total}"
            for metric_id, (rejected, total) in endpoint_rejections.items()
            if total
        )
        interpretation_conclusion_rows = [
            {
                "angle": "beat_detection_accuracy",
                "leading_or_selected_case": top_peak.get(
                    "algorithm_or_reducer", top_peak.get("algorithm")
                ),
                "finding": (
                    f"Highest {metric_contract}: {top_f1:.1f}% on channel "
                    f"{top_peak.get('channel')}."
                ),
                "confidence": (
                    "current_contract_recording_distribution"
                    if distribution_rows
                    else "historical_contract_not_poolable_with_v3"
                ),
                "selection_effect": "manual_default_review_only",
            },
            {
                "angle": "statistical_comparison",
                "leading_or_selected_case": (
                    static_peak_statistical_rows[0].get("reference_algorithm")
                    if static_peak_statistical_rows
                    else None
                ),
                "finding": (
                    f"{significant}/{len(static_peak_statistical_rows)} "
                    "recording-endpoint comparisons reject in the unified "
                    f"Holm-Sidak family ({endpoint_rejection_text})."
                    if static_peak_statistical_rows
                    else "No current-contract recording-level corrected P-value family is available."
                ),
                "confidence": (
                    "f1_prespecified_other_endpoints_retrospective_supplement"
                    if static_peak_statistical_rows
                    else "not_available"
                ),
                "selection_effect": "none_automatic",
            },
        ]
    _write_csv(tables / "result_comparison.csv", interpretation_comparison_rows)
    _strict_json(
        tables / "result_comparison.json",
        interpretation_comparison_audit_rows or interpretation_comparison_rows,
    )
    _strict_json(
        tables / "result_comparison_compact.json",
        interpretation_comparison_rows,
    )
    (tables / "result_comparison_audit.json").unlink(missing_ok=True)
    (
        root / "result_backup/tables/result_comparison_audit.json"
    ).unlink(missing_ok=True)
    interpretation_conclusion_display_rows = [
        {
            "angle": row.get("angle"),
            "leading_or_selected_case": row.get("leading_or_selected_case"),
            "finding": row.get("finding"),
            "confidence": row.get("confidence"),
            "selection_effect": row.get("selection_effect"),
        }
        for row in interpretation_conclusion_rows
    ]
    _write_csv(
        tables / "result_conclusions.csv",
        interpretation_conclusion_display_rows,
    )
    _strict_json(tables / "result_conclusions.json", interpretation_conclusion_rows)
    if manifest["study_type"] == "stage5_pre_motion_ptt":
        reporter_output_status_rows = _stage5_reporter_output_status(
            profile_rows=profile_rows,
            report_config=report_config,
            tables=tables,
            figures=figures,
            reverse_available=reverse_available,
            denoiser_enabled=denoiser_enabled,
        )
        _write_csv(
            tables / "reporter_output_status.csv", reporter_output_status_rows
        )
        _strict_json(
            tables / "reporter_output_status.json", reporter_output_status_rows
        )
        headline["reporter_output_status_rows"] = reporter_output_status_rows
    if manifest["study_type"] == "stage5_pre_motion_ptt":
        interpretation_markdown = root / "RESULT_INTERPRETATION.md"
        interpretation_markdown.write_text(
            "\n".join(
                [
                    "# Result interpretation and selection confidence",
                    "",
                    "Stage5-pre outcome evidence is metric-long below. The "
                    "lossless source fields remain in `tables/result_comparison."
                    "json`; detector and denoiser endpoints are never "
                    "merged into one wide result row.",
                    "",
                    "## Compact comparison index",
                    "",
                    _markdown_table(
                        interpretation_comparison_rows,
                        fields=(
                            "evidence_type",
                            "model_or_module",
                            "evaluation",
                            "stratum",
                            "metric",
                            "mean_sd",
                            "ci95",
                            "p",
                        ),
                    ),
                    "",
                    "## Conclusions by evidence angle",
                    "",
                    _markdown_table(
                        interpretation_conclusion_display_rows,
                        fields=(
                            "angle",
                            "leading_or_selected_case",
                            "finding",
                            "confidence",
                            "selection_effect",
                        ),
                    ),
                    "",
                ]
            ),
            encoding="utf-8",
        )
    else:
        interpretation_markdown = root / "RESULT_INTERPRETATION.md"
        interpretation_markdown.write_text(
            "\n".join(
                [
                    "# Result interpretation and selection confidence",
                    "",
                    "Peak-detector outcome rows are metric-long. The lossless "
                    "distribution and test fields remain in "
                    "`tables/result_comparison.json`.",
                    "",
                    "## Compact comparison index",
                    "",
                    _markdown_table(interpretation_comparison_rows),
                    "",
                    "## Conclusions by evidence angle",
                    "",
                    _markdown_table(
                        interpretation_conclusion_display_rows,
                        fields=(
                            "angle",
                            "leading_or_selected_case",
                            "finding",
                            "confidence",
                            "selection_effect",
                        ),
                    ),
                    "",
                ]
            ),
            encoding="utf-8",
        )
    figure_table_sources: Mapping[str, tuple[str, ...]] = {
        "motion_detector_metrics": ("motion_detector_metrics",),
        "motion_internal_confusion_matrix": ("motion_detector_window_confusion",),
        "motion_ptt_confusion_matrix": ("motion_detector_window_confusion",),
        "motion_ptt_training_oof_confusion_matrix": (
            "motion_detector_window_confusion",
        ),
        "motion_internal_reverse_confusion_matrix": (
            "motion_detector_window_confusion",
        ),
        "motion_internal_file_confusion_matrix": (
            "motion_detector_file_confusion",
        ),
        "motion_ptt_file_confusion_matrix": (
            "motion_detector_file_confusion",
        ),
        "motion_ptt_training_oof_file_confusion_matrix": (
            "motion_detector_file_confusion",
        ),
        "motion_internal_reverse_file_confusion_matrix": (
            "motion_detector_file_confusion",
        ),
        "frailty29_trained_window_score_distribution": (
            "motion_detector_score_distributions",
        ),
        "frailty29_trained_file_score_distribution": (
            "motion_detector_score_distributions",
        ),
        "ptt22_trained_window_score_distribution": (
            "motion_detector_score_distributions",
        ),
        "ptt22_trained_file_score_distribution": (
            "motion_detector_score_distributions",
        ),
        "frailty29_trained_window_prediction_tsne": (
            "motion_detector_prediction_tsne",
        ),
        "frailty29_trained_file_prediction_tsne": (
            "motion_detector_prediction_tsne",
        ),
        "ptt22_trained_window_prediction_tsne": (
            "motion_detector_prediction_tsne",
        ),
        "ptt22_trained_file_prediction_tsne": (
            "motion_detector_prediction_tsne",
        ),
        "frailty29_trained_window_roc_auc_curve": (
            "motion_detector_roc_curves",
        ),
        "frailty29_trained_file_roc_auc_curve": (
            "motion_detector_roc_curves",
        ),
        "ptt22_trained_window_roc_auc_curve": (
            "motion_detector_roc_curves",
        ),
        "ptt22_trained_file_roc_auc_curve": (
            "motion_detector_roc_curves",
        ),
        "motion_training_learning_curves": ("motion_training_history",),
        "denoiser_interval_rmse": ("denoiser_static", "denoiser_dynamic"),
        "denoiser_beat_f1": ("denoiser_static", "denoiser_dynamic"),
        "denoiser_beat_sensitivity": ("denoiser_summary",),
        "denoiser_beat_ppv": ("denoiser_summary",),
        "denoiser_runtime": ("denoiser_summary",),
        "static_peak_detector_f1": (
            "static_peak_detector_summary",
            "static_peak_detector_recording_metrics",
        ),
        "static_peak_detector_sensitivity": (
            "static_peak_detector_summary",
            "static_peak_detector_recording_metrics",
        ),
        "static_peak_detector_ppv": (
            "static_peak_detector_summary",
            "static_peak_detector_recording_metrics",
        ),
        "static_peak_detector_interval_rmse": (
            "static_peak_detector_summary",
            "static_peak_detector_recording_metrics",
        ),
        "static_peak_detector_runtime": (
            "static_peak_detector_summary",
            "static_peak_detector_recording_metrics",
        ),
    }
    table_figure_pairs = [
        {
            "table": table_name,
            "table_status": (
                "available"
                if (tables / f"{table_name}.csv").is_file()
                else "not_registered"
            ),
            "figure": image.stem,
            "figure_status": "generated",
            "figure_path": image.relative_to(root).as_posix(),
        }
        for image in images
        for table_name in figure_table_sources.get(image.stem, ())
    ]
    _write_csv(tables / "table_figure_pairs.csv", table_figure_pairs)
    _strict_json(tables / "table_figure_pairs.json", table_figure_pairs)
    write_table_column_definitions(tables, csv_directory=tables)
    workbook_tables: list[ReportTable] = []
    for csv_path in sorted(tables.glob("*.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        workbook_tables.append(
            ReportTable(
                name=csv_path.stem,
                rows=rows,
                description="Stage motion/peak report table",
            )
        )
    write_excel_workbook(tables / "report_tables.xlsx", workbook_tables)
    _strict_json(root / "study_summary.json", {
        "schema_version": RESULT_SCHEMA,
        "study_id": manifest["study_id"],
        "status": manifest["status"],
        "table_figure_pairs": table_figure_pairs,
        "test_components": test_component_rows,
        "reporter_profiles": profile_rows,
        "result_comparison": interpretation_comparison_rows,
        "result_conclusions": interpretation_conclusion_rows,
        **headline,
    })
    numerical_sections: list[str] = []
    if manifest["study_type"] == "stage5_pre_motion_ptt":
        numerical_sections.extend(
            [
                "### Detector results: one endpoint per table",
                "",
                "Each table is intentionally narrow. `mean ± SD` is the "
                "arithmetic mean and between-participant sample SD of the named "
                "window- or file-level participant endpoint; CI95 is a percentile "
                "bootstrap of participants. Complete numerical and applicability "
                "fields remain in `motion_detector_participant_macro_statistics.json`.",
                "",
            ]
        )
        for metric in _MOTION_DETECTOR_METRICS:
            numerical_sections.extend(
                [
                    f"#### Detector — {_MOTION_DETECTOR_METRIC_LABELS[metric]}",
                    "",
                    _markdown_table(
                        detector_metric_tables.get(metric, []),
                        fields=_MOTION_DETECTOR_RESULT_FIELDS,
                    ),
                    "",
                ]
            )
        numerical_sections.extend(
            [
                "#### Detector — worst-fold balanced accuracy",
                "",
                _markdown_table(
                    detector_worst_fold_rows,
                    fields=(
                        "model_id",
                        "evaluation",
                        "level",
                        "worst_fold_balanced_accuracy",
                    ),
                ),
                "",
                "Worst-fold BA is a separate robustness endpoint and applies "
                "only to grouped-OOF evaluations; frozen transfer has no training "
                "fold axis.",
                "",
                "#### Detector — per-class performance",
                "",
                _markdown_table(
                    detector_per_class_performance_rows,
                    fields=(
                        "model_id",
                        "evaluation",
                        "level",
                        "activity",
                        "sensitivity",
                        "specificity",
                        "balanced_accuracy_ovr",
                        "f1",
                    ),
                ),
                "",
                "#### Detector — per-class discrimination",
                "",
                _markdown_table(
                    detector_per_class_discrimination_rows,
                    fields=(
                        "model_id",
                        "evaluation",
                        "level",
                        "activity",
                        "precision",
                        "roc_auc_ovr",
                        "pr_auc_ovr",
                    ),
                ),
                "",
                "#### Detector — window-level confusion counts",
                "",
                _markdown_table(
                    detector_window_confusion_rows,
                    fields=(
                        "model_id",
                        "dataset",
                        "aggregation_level",
                        "true_static_predicted_static",
                        "true_static_predicted_motion",
                        "true_motion_predicted_static",
                        "true_motion_predicted_motion",
                    ),
                ),
                "",
                "#### Detector — file-level confusion counts",
                "",
                _markdown_table(
                    detector_file_confusion_rows,
                    fields=(
                        "model_id",
                        "dataset",
                        "aggregation_level",
                        "true_static_predicted_static",
                        "true_static_predicted_motion",
                        "true_motion_predicted_static",
                        "true_motion_predicted_motion",
                    ),
                ),
                "",
                "Window-level metrics retain every persisted 8 s window. "
                f"File-level metrics first take the {file_score_aggregation} "
                "probability within one physical file and apply its frozen "
                "threshold once.",
                "",
                (
                    "Detector P is the Holm-adjusted, two-sided participant-paired "
                    "Monte-Carlo sign-flip P value for the retrospective training-"
                    f"source comparison (`{detector_inference_candidate_model_id}` "
                    f"minus `{detector_inference_reference_model_id}`) on an "
                    "identical target/level roster. Holm correction covers all "
                    "detector endpoints within each target × level family. This "
                    "asymmetric grouped-OOF-versus-frozen comparison is exploratory."
                    if detector_training_source_inference_rows
                    else f"Detector P is N/A because `{detector_inference_candidate_model_id}` "
                    f"and `{detector_inference_reference_model_id}` do not both "
                    "predict the same target units in this artifact. Cross-dataset "
                    "rows and window/file rows are not treated as paired model "
                    "comparisons."
                ),
                "",
                "The detailed classification-diagnostic and reporter-applicability "
                "audits are separate files: `motion_detector_diagnostic_status.*` "
                "and `reporter_output_status.*`; they are not result columns.",
                "",
                "ROC figures are empirical ROC curves with AUC annotated. t-SNE "
                "embeds persisted prediction-probability vectors, not hidden "
                "features.",
                "",
            ]
        )
    if manifest["study_type"] == "stage5_pre_motion_ptt":
        if comparison_rows:
            numerical_sections.extend(
                [
                    "### Frozen motion-model comparison candidates",
                    "",
                    _markdown_table(
                        comparison_rows,
                        fields=(
                            "candidate_id",
                            "training_dataset",
                            "training_participant_count",
                            "model_path",
                            "threshold_path",
                        ),
                    ),
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
                    "### Denoiser results: static",
                    "",
                    _markdown_table(
                        denoiser_activity_tables.get("static", []),
                        fields=_DENOISER_ACTIVITY_RESULT_FIELDS,
                    ),
                    "",
                    "### Denoiser results: dynamic",
                    "",
                    _markdown_table(
                        denoiser_activity_tables.get("dynamic", []),
                        fields=_DENOISER_ACTIVITY_RESULT_FIELDS,
                    ),
                    "",
                    "Both tables are sorted by participant-macro IBI–PPI RMSE "
                    "ascending across RED and IR rows. `*` marks a best metric "
                    "value (minimum RMSE or maximum F1); `**` on the denoiser "
                    "name marks a row best on both. SD uses ddof=1 and measures "
                    "between-subject dispersion, not repeat-training uncertainty. "
                    "Best-value marks use the unrounded participant mean. "
                    "Coverage/failure counts and the full endpoint audit remain "
                    "in `denoiser_coverage.*`, `denoiser_summary.json`, and "
                    "`denoiser_compact_statistics.json` rather than widening "
                    "these two result tables.",
                    "",
                    "The displayed RMSE P is the retrospective exploratory two-sided "
                    "participant-paired Monte-Carlo sign-flip test versus "
                    f"`{denoiser_inference_reference_id}`, restricted "
                    "to identical successfully processed segment keys; RMSE also "
                    "requires matched intervals on both sides. Holm correction is "
                    "applied across all non-reference reducers separately within "
                    "each activity × channel × endpoint family. The raw and adjusted "
                    "numeric audit is in `denoiser_paired_inference.*`.",
                    "",
                    "Denoiser F1 is beat-event matching F1 after lag alignment to "
                    "ECG annotations, not motion-classification F1. It guards "
                    "against a deceptively low interval RMSE computed from only a "
                    "small easy subset of beats.",
                    "",
                ]
            )
        else:
            numerical_sections.extend(
                ["### Denoiser benchmark", "", "Skipped by execution option.", ""]
            )
    else:
        if peak_current_contract:
            peak_method_text = (
                "Values are median [Q1, Q3] across subject recordings. Each "
                "detector is called once on each complete recording/channel; "
                f"ECG-to-PPG lag is re-estimated in consecutive {peak_lag_window_s:g} s "
                "recording windows without restarting the detector. A reference "
                f"beat is correct only under `{peak_matching_id}` with a "
                f"±{1000.0*peak_tolerance_s:g} ms tolerance. Sensitivity = "
                "100*ncorrect/nref, PPV = 100*ncorrect/nPPG, and F1 is their "
                "harmonic mean. Boxplot whiskers are the 10th and 90th percentiles."
            )
        else:
            peak_method_text = (
                "This historical report preserves its resolved validation contract: "
                f"alignment=`{peak_alignment_id}`, lag window={peak_lag_window_s:g} s, "
                f"beat tolerance=±{1000.0*peak_tolerance_s:g} ms, and "
                f"aggregation=`{peak_aggregation_id}`. The later 300 s/±150 ms "
                "recording-level contract is not back-applied."
            )
        if isinstance(peak_statistical_contract, Mapping):
            peak_statistical_text = (
                "The same two-sided Wilcoxon rank-sum procedure compares "
                "MSPTDfast with each comparator for recording F1, sensitivity, "
                "PPV, IBI–PPI RMSE, and execution-time percentage. All selected "
                "metrics, channels, and reference-comparator contrasts form one "
                "Holm–Sidak step-down family (10 hypotheses here; alpha=0.05). "
                "The MSPTDfast-advantage column is reference minus comparator "
                "for higher-is-better endpoints and comparator minus reference "
                "for lower-is-better endpoints, so positive always favors "
                "MSPTDfast. "
                "For ordered raw p-values p_(i), m_i=m-i+1, the local critical "
                "value is 1-(1-alpha)^(1/m_i), and adjusted p_(i) is the running "
                "maximum of 1-(1-p_(i))^m_i, capped at 1. F1 was the only "
                "endpoint pre-specified in this historical resolved plan; the "
                "other four endpoints are retrospective supplements "
                "requested on 2026-08-24. The table retains the original F1-only "
                "family adjustment for audit, but inferential decisions in this "
                "updated report use the unified 10-hypothesis adjustment. "
                "The samples are restricted to the same subject-recordings, but "
                "scipy.stats.ranksums is an unpaired rank-sum test and does not "
                "use within-record pairing or correct ties; this source-faithful "
                "choice is a limitation, not a paired signed-rank claim. Runtime "
                "is one sequential local wall-time observation per detector, "
                "recording, and channel, so it is exploratory hardware/load-"
                "specific efficiency evidence. Sources: Charlton et al. (2025), "
                "DOI 10.1088/1361-6579/adb89e; Wilcoxon (1945), DOI "
                "10.2307/3001968; Holm (1979), DOI 10.2307/4615733; Šidák "
                "(1967), DOI 10.1080/01621459.1967.10482935."
            )
        else:
            peak_statistical_text = (
                "This historical resolved plan did not register the later "
                "recording-level Wilcoxon/Holm-Sidak comparison, so no such "
                "inferential claim is added during report regeneration."
            )
        numerical_sections.extend(
            [
                "### Subject-recording performance",
                "",
                _markdown_table(summary_rows),
                "",
                peak_method_text,
                "",
                "### MSPTDfast endpoint effects",
                "",
                _markdown_table(
                    static_peak_effect_display_rows,
                    fields=(
                        "endpoint",
                        "channel",
                        "registration",
                        "common subject-recordings",
                        "MSPTDfast median",
                        "aboy_project median",
                        "unit",
                        "MSPTDfast advantage",
                    ),
                ),
                "",
                "### MSPTDfast endpoint inference",
                "",
                _markdown_table(
                    static_peak_inference_display_rows,
                    fields=(
                        "endpoint",
                        "channel",
                        "registration",
                        "rank-sum z",
                        "raw p",
                        "Holm–Sidak adjusted p (global family)",
                        "reject at alpha=0.05",
                    ),
                ),
                "",
                peak_statistical_text,
                "",
            ]
        )
    lines = [
        f"# {manifest['study_id']}", "", f"Status: **{manifest['status']}**", "",
        "## Scientific scope", "",
        str(manifest["scientific_scope"]), "",
        "## Test models, modules, inputs, and fixed parameters", "",
        "The identical standalone table is in `TEST_COMPONENTS.md`; "
        "machine-readable copies are `tables/test_components.csv` and `.json`. "
        "Input data are named directly rather than represented by hashes.", "",
        markdown_test_component_table(test_component_rows), "",
        "## Model/module-owned reporter methods and literature", "",
        "Reporter profiles are selected from persisted component identities and "
        "change presentation only. The complete method/source record is in "
        "`REPORT_METHODS.md`; machine-readable rows are in "
        "`tables/reporter_profiles.csv`.", "",
        markdown_reporter_profile_tables(profile_rows), "",
        "## Confidence-qualified result interpretation", "",
        "P values are null-hypothesis tail probabilities, not posterior confidence. "
        "The standalone detailed table is in `RESULT_INTERPRETATION.md`.", "",
        _markdown_table(
            interpretation_conclusion_display_rows,
            fields=(
                "angle",
                "leading_or_selected_case",
                "finding",
                "confidence",
                "selection_effect",
            ),
        ), "",
        "## Figures", "",
        *[f"![{path.stem}](figures/{path.name})" for path in images], "",
        "## Numerical outputs", "",
        *numerical_sections,
        "Machine-readable values are in `study_summary.json` and `tables/`. "
        "Each report table has an individual CSV; `tables/report_tables.xlsx` "
        "contains one table per worksheet, and `tables/table_figure_pairs.csv` "
        "records every analytical figure/table pair.", "",
    ]
    markdown = "\n".join(lines)
    (root / "STUDY_SUMMARY.md").write_text(markdown, encoding="utf-8")
    html_images = "\n".join(
        f'<figure><img src="figures/{path.name}" alt="{path.stem}">'
        f"<figcaption>{path.stem}</figcaption></figure>"
        for path in images
    )
    html_numerical = ""
    if manifest["study_type"] == "stage5_pre_motion_ptt":
        html_numerical = (
            "<h3>Detector results: one endpoint per table</h3>"
            "<p>Mean ± SD is the participant-macro mean and between-participant "
            "sample SD. CI95 is a participant percentile bootstrap.</p>"
        )
        for metric in _MOTION_DETECTOR_METRICS:
            html_numerical += (
                f"<h4>Detector — {_MOTION_DETECTOR_METRIC_LABELS[metric]}</h4>"
                + _html_table(
                    detector_metric_tables.get(metric, []),
                    fields=_MOTION_DETECTOR_RESULT_FIELDS,
                )
            )
        html_numerical += (
            "<h4>Detector — worst-fold balanced accuracy</h4>"
            + _html_table(
                detector_worst_fold_rows,
                fields=(
                    "model_id",
                    "evaluation",
                    "level",
                    "worst_fold_balanced_accuracy",
                ),
            )
            + "<h4>Detector — per-class performance</h4>"
            + _html_table(
                detector_per_class_performance_rows,
                fields=(
                    "model_id",
                    "evaluation",
                    "level",
                    "activity",
                    "sensitivity",
                    "specificity",
                    "balanced_accuracy_ovr",
                    "f1",
                ),
            )
            + "<h4>Detector — per-class discrimination</h4>"
            + _html_table(
                detector_per_class_discrimination_rows,
                fields=(
                    "model_id",
                    "evaluation",
                    "level",
                    "activity",
                    "precision",
                    "roc_auc_ovr",
                    "pr_auc_ovr",
                ),
            )
            + "<h4>Detector — window-level confusion counts</h4>"
            + _html_table(
                detector_window_confusion_rows,
                fields=(
                    "model_id",
                    "dataset",
                    "aggregation_level",
                    "true_static_predicted_static",
                    "true_static_predicted_motion",
                    "true_motion_predicted_static",
                    "true_motion_predicted_motion",
                ),
            )
            + "<h4>Detector — file-level confusion counts</h4>"
            + _html_table(
                detector_file_confusion_rows,
                fields=(
                    "model_id",
                    "dataset",
                    "aggregation_level",
                    "true_static_predicted_static",
                    "true_static_predicted_motion",
                    "true_motion_predicted_static",
                    "true_motion_predicted_motion",
                ),
            )
            + "<p>Window-level metrics retain every persisted 8 s window; file-"
            f"level metrics aggregate by {file_score_aggregation} probability "
            "before one frozen-threshold decision. Worst-fold BA applies only "
            "to grouped OOF.</p>"
            + (
                "<p>Detector P is the retrospective, exploratory participant-"
                f"paired sign-flip P for {html_escape(detector_inference_candidate_model_id)} "
                f"versus {html_escape(detector_inference_reference_model_id)}, "
                "Holm-adjusted across registered endpoints per target and level.</p>"
                if detector_training_source_inference_rows
                else "<p>Detector P is N/A: no second model predicts the same "
                "target units in this artifact.</p>"
            )
            + "<p>Diagnostic availability and output applicability are separate "
            "audit files, not result columns. ROC plots are empirical curves; "
            "t-SNE embeds prediction probabilities only.</p>"
        )
    if manifest["study_type"] == "stage5_pre_motion_ptt":
        if comparison_rows:
            html_numerical += (
                "<h3>Frozen motion-model comparison candidates</h3>"
                + _html_table(
                    comparison_rows,
                    fields=(
                        "candidate_id",
                        "training_dataset",
                        "training_participant_count",
                        "model_path",
                        "threshold_path",
                    ),
                )
                + "<p>Packaged for a later paired single-factor comparison in the "
                "selected final frailty classifier; Stage5-pre does not train that "
                "downstream classifier.</p>"
            )
        if denoiser_enabled:
            html_numerical += (
                "<h3>Denoiser results: static</h3>"
                + _html_table(
                    denoiser_activity_tables.get("static", []),
                    fields=_DENOISER_ACTIVITY_RESULT_FIELDS,
                )
                + "<h3>Denoiser results: dynamic</h3>"
                + _html_table(
                    denoiser_activity_tables.get("dynamic", []),
                    fields=_DENOISER_ACTIVITY_RESULT_FIELDS,
                )
                + "<p>Both tables sort IBI–PPI RMSE ascending across RED and IR. "
                "* marks minimum RMSE or maximum F1; ** on the denoiser name "
                "marks a row best on both. SD is between-subject sample SD "
                "(ddof=1); marks use the unrounded participant mean. Coverage "
                "and full endpoint evidence remain in the "
                "audit files rather than widening these tables.</p>"
                + "<p>The displayed RMSE P is a retrospective exploratory "
                "participant-paired sign-flip test versus "
                f"{html_escape(denoiser_inference_reference_id)} on common successful "
                "segments, Holm-adjusted across reducers within each activity × "
                "channel × RMSE family.</p>"
                + "<p>Denoiser F1 is ECG-aligned beat-event matching F1, not "
                "motion-classification F1.</p>"
            )
        else:
            html_numerical += "<h3>Denoiser benchmark</h3><p>Skipped by execution option.</p>"
    else:
        html_numerical += (
            "<h3>Subject-recording performance</h3>"
            + _html_table(summary_rows)
            + "<p>" + html_escape(peak_method_text) + "</p>"
            + "<h3>MSPTDfast endpoint effects</h3>"
            + _html_table(
                static_peak_effect_display_rows,
                fields=(
                    "endpoint",
                    "channel",
                    "registration",
                    "common subject-recordings",
                    "MSPTDfast median",
                    "aboy_project median",
                    "unit",
                    "MSPTDfast advantage",
                ),
            )
            + "<h3>MSPTDfast endpoint inference</h3>"
            + _html_table(
                static_peak_inference_display_rows,
                fields=(
                    "endpoint",
                    "channel",
                    "registration",
                    "rank-sum z",
                    "raw p",
                    "Holm–Sidak adjusted p (global family)",
                    "reject at alpha=0.05",
                ),
            )
            + "<p>" + html_escape(peak_statistical_text) + "</p>"
        )
    html_component_tables = "".join(
        f"<h3>{html_escape(title)}</h3>"
        + _html_table(
            test_component_rows,
            fields=tuple(field for field, _label in schema),
        )
        for title, schema in TEST_COMPONENT_VIEW_SCHEMAS
    )
    html_profile_tables = "".join(
        f"<h3>{html_escape(title)}</h3>"
        + _html_table(
            profile_rows,
            fields=tuple(field for field, _label in schema),
        )
        for title, schema in REPORTER_PROFILE_VIEW_SCHEMAS
    )
    (root / "STUDY_SUMMARY.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>"
        + manifest["study_id"] + "</title><h1>" + manifest["study_id"]
        + "</h1><p>Status: " + manifest["status"] + "</p>"
        + "<h2>Test models, modules, inputs, and fixed parameters</h2>"
        + "<p>The identical Markdown table is in TEST_COMPONENTS.md. Input data "
        + "are named directly rather than represented by hashes.</p>"
        + html_component_tables
        + "<h2>Model/module-owned reporter methods and literature</h2>"
        + "<p>See REPORT_METHODS.md. Profiles are presentation-only.</p>"
        + html_profile_tables
        + "<h2>Confidence-qualified result interpretation</h2>"
        + "<p>P values are null-hypothesis tail probabilities, not posterior confidence. "
        + "See RESULT_INTERPRETATION.md.</p>"
        + _html_table(
            interpretation_conclusion_display_rows,
            fields=(
                "angle",
                "leading_or_selected_case",
                "finding",
                "confidence",
                "selection_effect",
            ),
        )
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
            root / "STUDY_SUMMARY.html", component_markdown,
            methods_markdown, interpretation_markdown,
            root / "outputs_index.json",
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
            else (
                "PTT sit-only subject-recording detector comparison: default "
                "MSPTDfast versus explicit aboy_project ablation; 300-s drifting "
                "lag and +/-150-ms one-to-one beat assessment; no motion segments "
                "and no denoiser selection"
            )
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
                        scoring_peak_detector=str(
                            benchmark.get(
                                "scoring_peak_detector", CANONICAL_DETECTOR_ID
                            )
                        ),
                        scoring_peak_detector_parameters=benchmark.get(
                            "scoring_peak_detector_parameters"
                        ),
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
