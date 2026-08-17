"""M3 唯一未来活动信号算法入口 / Sole future-active M3 signal API."""

from .contracts import (
    ExternalResampleResult,
    HrvResult,
    ImuPreprocessResult,
    PeakResult,
    PpgPreprocessResult,
    ProcessingStatus,
    QualityAssessment,
    QualityIssue,
    map_processing_status_to_m1,
    to_serializable,
)
from .ppg import (
    design_ppg_sos,
    dual_ppg_raw_metrics,
    normalized_spectral_entropy,
    ppg_statistical_features,
    preprocess_ppg,
    raw_ppg_metrics,
    resample_external_ppg_to_400,
)
from .imu import (
    EskfConfiguration,
    NoPrecalibrationEskf,
    STANDARD_GRAVITY_MPS2,
    convert_imu_to_si,
    vector_jerk,
)
from .imu_runtime import CausalImuProcessor, preprocess_imu
from .physiology import (
    PPI_MAX_SEC,
    PPI_MIN_SEC,
    autocorrelation_periodicity,
    choose_primary_channel,
    compute_prv,
    derive_ppi,
    detect_peaks_corrected,
    dual_channel_agreement,
    template_correlation,
)
from .quality import (
    inspect_and_repair_signal,
    validate_timestamp_grid,
    validate_channel_contract,
    with_contract_issues,
)
from .registry import get_profile, load_registry, registry_sha256
from .fold_contract import fit_fold_scaler, resolve_m2_fold
from .reference_evaluation import (
    TransitDelayArtifact,
    evaluate_ppg_against_ecg,
    fit_transit_delay,
)
from .scaling import (
    FoldAmplitudeRiskModel,
    FoldScaler,
    build_raw8_model_view,
    robust_window_scale,
)

__all__ = [
    "FoldScaler",
    "FoldAmplitudeRiskModel",
    "ExternalResampleResult",
    "TransitDelayArtifact",
    "EskfConfiguration",
    "HrvResult",
    "ImuPreprocessResult",
    "PeakResult",
    "PpgPreprocessResult",
    "ProcessingStatus",
    "QualityAssessment",
    "QualityIssue",
    "map_processing_status_to_m1",
    "NoPrecalibrationEskf",
    "CausalImuProcessor",
    "PPI_MAX_SEC",
    "PPI_MIN_SEC",
    "STANDARD_GRAVITY_MPS2",
    "autocorrelation_periodicity",
    "build_raw8_model_view",
    "convert_imu_to_si",
    "choose_primary_channel",
    "compute_prv",
    "derive_ppi",
    "detect_peaks_corrected",
    "dual_channel_agreement",
    "dual_ppg_raw_metrics",
    "design_ppg_sos",
    "get_profile",
    "fit_fold_scaler",
    "fit_transit_delay",
    "inspect_and_repair_signal",
    "load_registry",
    "normalized_spectral_entropy",
    "ppg_statistical_features",
    "preprocess_ppg",
    "preprocess_imu",
    "raw_ppg_metrics",
    "registry_sha256",
    "resolve_m2_fold",
    "resample_external_ppg_to_400",
    "robust_window_scale",
    "to_serializable",
    "template_correlation",
    "validate_channel_contract",
    "validate_timestamp_grid",
    "vector_jerk",
    "with_contract_issues",
    "evaluate_ppg_against_ecg",
]
