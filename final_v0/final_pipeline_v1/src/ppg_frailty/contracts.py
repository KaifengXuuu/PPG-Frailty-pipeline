"""跨模块类型与科学不变量 / Cross-module types and scientific invariants."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class RepresentationMode(str, Enum):
    """四种规范表征 / Four canonical representations."""

    RAW = "raw"
    FEATURE_VECTOR = "feature_vector"
    FEATURE_MATRIX = "feature_matrix"
    FUSION = "fusion"


class SignalRoute(str, Enum):
    """信号来源路线 / Signal-source route."""

    DIRECT = "direct_x_filter"
    IDENTITY = "identity_direct"
    ARTIFACT_RATE_ONLY = "non_identity_x_ar_rate_only"
    DROPPED = "dropped"


class QualityState(str, Enum):
    """Endpoint 质量状态；not_applicable 不等于 pass/fail。"""

    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ManifestRow:
    """一个 recording 的冻结描述 / Frozen description of one recording."""

    record_id: str
    participant_id: str
    class_id: int
    class_name: str
    role: str
    source_path: str
    source_hash: str
    source_version: str
    fs: float
    n_samples: int
    duration_s: float
    channel_schema: tuple[str, ...]
    channel_units: dict[str, str]
    synchrony_status: str
    reference_available: bool
    qc_status: str
    qc_reasons: tuple[str, ...]
    manifest_version: str


@dataclass
class SignalViews:
    """保幅 direct 与 rate-only analysis 视图 / Direct and analysis signal views."""

    x_native: np.ndarray
    x_filter: np.ndarray
    x_analysis: np.ndarray
    imu_processed: dict[str, np.ndarray]
    metadata: dict[str, Any]

    def validate(self) -> None:
        """验证时间对齐和 route 语义 / Validate alignment and route semantics."""

        native = np.asarray(self.x_native)
        filtered = np.asarray(self.x_filter)
        analysis = np.asarray(self.x_analysis)
        if native.ndim != 2 or native.shape[1] != 2:
            raise ValueError("x_native must be samples×[RED,IR]")
        if filtered.shape != native.shape or analysis.shape != native.shape:
            raise ValueError("all PPG views must share the original time grid")
        if not np.isfinite(filtered).all() or not np.isfinite(analysis).all():
            raise ValueError("filtered/analysis views must be finite")
        if float(self.metadata.get("fs_hz", 0.0)) != 400.0:
            raise ValueError("canonical signal views must remain at 400 Hz")
        non_identity = bool(self.metadata.get("non_identity_artifact_reduction", False))
        if non_identity and not bool(self.metadata.get("rate_only", False)):
            raise ValueError("non-identity x_ar must be rate-only")


@dataclass(frozen=True)
class QualityComponent:
    """一个可审计 SQI component / One auditable SQI component."""

    raw_value: float | None
    normalized_value: float | None
    state: QualityState
    reason: str


@dataclass(frozen=True)
class QualityEndpoint:
    """Q_rate 或 Q_morph / One endpoint-specific quality result."""

    score: float | None
    state: QualityState
    threshold: float | None
    components: dict[str, QualityComponent]
    reasons: tuple[str, ...]
    coverage: float


@dataclass(frozen=True)
class QualityResult:
    """分离 rate 与 morphology 的质量合同 / Separate endpoint SQI contract."""

    q_rate: QualityEndpoint
    q_morph: QualityEndpoint
    state: str
    components: dict[str, QualityComponent]
    reasons: tuple[str, ...]
    coverage: float
    fitted_on_participant_ids: tuple[str, ...] = ()

    def validate_for_route(self, route: SignalRoute) -> None:
        """强制非恒等 x_ar 的 morphology 不适用 / Enforce rate-only semantics."""

        if route is SignalRoute.ARTIFACT_RATE_ONLY:
            if self.q_morph.state is not QualityState.NOT_APPLICABLE:
                raise ValueError("non-identity x_ar requires q_morph=not_applicable")
            if self.q_morph.score is not None:
                raise ValueError("not_applicable q_morph cannot carry a score")


@dataclass(frozen=True)
class PulseResult:
    """峰、间期和时间邻接合同 / Event, interval, and adjacency contract."""

    peaks: np.ndarray
    peak_timestamps_s: np.ndarray
    accepted_peak_mask: np.ndarray
    interval_start_peak_indices: np.ndarray
    interval_stop_peak_indices: np.ndarray
    ppi_s: np.ndarray
    valid_interval_mask: np.ndarray
    adjacency_mask: np.ndarray
    wavelength: str
    detector_version: str
    confidence: np.ndarray


@dataclass(frozen=True)
class FeatureVectorV1:
    """完整有序 file predictor / Complete ordered file-level predictor."""

    values: np.ndarray
    validity: np.ndarray
    feature_names: tuple[str, ...]
    schema_version: str
    provenance: dict[str, Any]


@dataclass(frozen=True)
class EngineeringFeatureSequence:
    """按时间排列的 engineering rows / Chronological engineering descriptors."""

    values: np.ndarray
    start_samples: np.ndarray
    valid_row_mask: np.ndarray
    channel_schema: tuple[str, ...]
    schema_version: str


@dataclass(frozen=True)
class OrderedFeatureMatrixV1:
    """每 recording 一个 D×32 矩阵 / One D-by-32 matrix per recording."""

    values: np.ndarray
    row_mask: np.ndarray
    channel_schema: tuple[str, ...]
    context_schema: tuple[str, ...]
    schema_version: str
    provenance: dict[str, Any]


@dataclass(frozen=True)
class ArtifactReductionResult:
    """ArtifactReducer 公共返回 / Common artifact-reducer result."""

    x_ar: np.ndarray | None
    reducer_id: str
    reducer_version: str
    is_identity: bool
    status: str
    confidence: float
    diagnostics: dict[str, Any]
    parameters: dict[str, Any]
    channel_available: tuple[bool, bool]
    alignment: dict[str, Any]
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class PredictionBundle:
    """分层预测与 coverage / Hierarchical predictions with coverage."""

    file_probabilities: dict[str, np.ndarray]
    participant_probabilities: dict[str, np.ndarray]
    coverage: dict[str, float]
    route: str
    model_version: str
    provenance: dict[str, Any]
    role_probabilities: dict[str, np.ndarray] = field(default_factory=dict)


def to_strict_json_value(value: Any) -> Any:
    """递归转 strict JSON；非有限值保留为 null / Convert recursively."""

    if isinstance(value, np.ndarray):
        return to_strict_json_value(value.tolist())
    if isinstance(value, np.generic):
        return to_strict_json_value(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return to_strict_json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_strict_json_value(item) for item in value]
    return value

