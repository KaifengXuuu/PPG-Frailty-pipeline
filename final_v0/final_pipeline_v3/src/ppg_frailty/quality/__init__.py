"""规范 SQI 门面 / Canonical endpoint-SQI facade.

中文：稳定导出 direct Q_rate/Q_morph 与 non-identity rate-only 路由检查。
English: Stable exports for direct endpoint SQI and rate-only route enforcement.
"""

from .components import QualityComponent, QualityEndpoint, QualityResult, QualityState, component_rows
from .endpoint_sqi import (
    SqiCalibrator,
    SqiConfig,
    SqiDiagnosticComponent,
    SqiDiagnosticConfig,
    SqiDiagnostics,
    evaluate_quality,
    evaluate_quality_diagnostics,
    fit_sqi_calibrator,
    quality_component_scores,
)
from .motion import (
    HISTORICAL_LIGHT_CNN_EVIDENCE,
    MOTION_MAJOR_METRIC_FIELDS,
    MOTION_OPTIONS,
    MidpointThresholdArtifact,
    MotionFoldJob,
    MotionOptionDescriptor,
    MotionOptionId,
    fit_train_only_midpoint_threshold,
    load_motion_fold_jobs,
    motion_activity_label,
    motion_contract_payload,
    resolve_motion_option,
    validate_motion_major_metrics,
)
from .motion_runner import (
    FormalMotionEntryRequiredError,
    MotionFitContext,
    MotionFittedArtifact,
    MotionPredictionInput,
    MotionWindowExample,
    evaluate_ptt_external_gate,
    load_motion_internal_evidence,
    run_internal_motion_oof,
    run_ptt_external_evaluation,
)
from .motion_reference import (
    PttImuUnitEvidence,
    PttImuUnitEvidenceRequired,
    load_ptt_imu_unit_evidence,
    run_formal_internal_motion_reference,
    run_formal_ptt_motion_reference,
)
from .motion_adapters import (
    FormalMotionRuntime,
    FormalMotionTrainerConfig,
    MotionRecordingInput,
    fit_formal_motion_model,
    load_formal_motion_model,
    materialize_motion_window_examples,
    predict_formal_motion_probability,
    write_formal_motion_input_schema,
)
from .routing import (
    QualityMode,
    QualityModeOutcome,
    QualityRoutingDisabledError,
    assert_quality_route,
    quality_mode_from_config,
    resolve_quality_mode,
    run_quality_mode,
)

__all__ = [
    "QualityComponent", "QualityEndpoint", "QualityResult", "QualityState",
    "QualityMode", "QualityModeOutcome", "QualityRoutingDisabledError",
    "SqiCalibrator", "SqiConfig", "SqiDiagnosticComponent", "SqiDiagnosticConfig",
    "SqiDiagnostics", "assert_quality_route", "component_rows",
    "evaluate_quality", "evaluate_quality_diagnostics", "fit_sqi_calibrator",
    "quality_component_scores",
    "quality_mode_from_config", "resolve_quality_mode", "run_quality_mode",
    "HISTORICAL_LIGHT_CNN_EVIDENCE", "MOTION_MAJOR_METRIC_FIELDS", "MOTION_OPTIONS",
    "MidpointThresholdArtifact", "MotionFoldJob", "MotionOptionDescriptor", "MotionOptionId",
    "fit_train_only_midpoint_threshold", "load_motion_fold_jobs", "motion_activity_label",
    "motion_contract_payload", "resolve_motion_option", "validate_motion_major_metrics",
    "FormalMotionEntryRequiredError", "MotionFitContext", "MotionFittedArtifact",
    "MotionPredictionInput", "MotionWindowExample",
    "evaluate_ptt_external_gate", "load_motion_internal_evidence",
    "run_internal_motion_oof", "run_ptt_external_evaluation",
    "PttImuUnitEvidence", "PttImuUnitEvidenceRequired",
    "load_ptt_imu_unit_evidence", "run_formal_internal_motion_reference",
    "run_formal_ptt_motion_reference",
    "FormalMotionRuntime", "FormalMotionTrainerConfig", "MotionRecordingInput",
    "fit_formal_motion_model", "load_formal_motion_model",
    "materialize_motion_window_examples", "predict_formal_motion_probability",
    "write_formal_motion_input_schema",
]
