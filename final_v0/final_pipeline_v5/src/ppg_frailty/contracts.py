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
    class_name_provenance_alias: str
    class_source: str
    label_record_id: str
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
    q_shape: QualityEndpoint | None = None

    @property
    def shape_endpoint(self) -> QualityEndpoint:
        """Return audited Q_shape, accepting legacy Q_morph-only artifacts."""

        return self.q_shape if self.q_shape is not None else self.q_morph

    def validate_for_route(self, route: SignalRoute) -> None:
        """强制非恒等 x_ar 的 morphology 不适用 / Enforce rate-only semantics."""

        if route is SignalRoute.ARTIFACT_RATE_ONLY:
            if self.q_morph.state is not QualityState.NOT_APPLICABLE:
                raise ValueError("non-identity x_ar requires q_morph=not_applicable")
            if self.q_morph.score is not None:
                raise ValueError("not_applicable q_morph cannot carry a score")
            if self.q_shape is not None and self.q_shape.state is not QualityState.NOT_APPLICABLE:
                raise ValueError("non-identity x_ar requires q_shape=not_applicable")
            if self.q_shape is not None and self.q_shape.score is not None:
                raise ValueError("not_applicable q_shape cannot carry a score")

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
    source_route: SignalRoute
    detection_run_id: str
    interval_run_ids: np.ndarray
    detector_id: str = ""
    selected_polarity: int = 0
    block_hri_provenance_hash: str = ""
    block_provenance: tuple[dict[str, Any], ...] = ()
    interval_rejection_reasons: tuple[str, ...] = ()
    peak_ordinals: np.ndarray | None = None
    detector_score: float = 0.0
    detector_coverage: float = float("nan")
    interval_source_routes: np.ndarray | None = None

    def validate_identity(self) -> None:
        """Prevent PPI intervals from crossing detector runs or signal routes."""

        route = self.source_route
        if not isinstance(route, SignalRoute):
            try:
                route = SignalRoute(str(route))
            except ValueError as exc:
                raise ValueError("PulseResult source_route is invalid") from exc
        if route is SignalRoute.DROPPED:
            raise ValueError("PulseResult cannot originate from a dropped route")
        run_id = str(self.detection_run_id).strip()
        if not run_id:
            raise ValueError("PulseResult requires a non-empty detection_run_id")
        interval_ids = np.asarray(self.interval_run_ids).astype(str)
        intervals = np.asarray(self.ppi_s)
        if interval_ids.shape != intervals.shape:
            raise ValueError("interval_run_ids must align one-to-one with PPI intervals")
        source_routes = self.interval_source_routes
        if source_routes is None:
            if interval_ids.size and np.any(interval_ids != run_id):
                raise ValueError("PPI intervals cannot cross pulse-detection runs")
        else:
            routes = np.asarray(source_routes).astype(str)
            if routes.shape != intervals.shape or np.any(interval_ids == ""):
                raise ValueError("composite PPI route/run provenance must align one-to-one")
            valid_routes = {
                SignalRoute.DIRECT.value,
                SignalRoute.IDENTITY.value,
                SignalRoute.ARTIFACT_RATE_ONLY.value,
                "routing_boundary",
            }
            if any(value not in valid_routes for value in routes):
                raise ValueError("composite PPI contains an unknown source route")
            boundary = routes == "routing_boundary"
            valid_intervals = np.asarray(self.valid_interval_mask, dtype=bool)
            if np.any(boundary & valid_intervals):
                raise ValueError("routing-boundary PPI separators must be invalid")
        if self.detector_id:
            if self.selected_polarity not in {-1, 1}:
                raise ValueError("detector PulseResult requires polarity -1 or +1")
            digest = str(self.block_hri_provenance_hash)
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("detector PulseResult requires a SHA-256 block provenance hash")
            if len(self.interval_rejection_reasons) != intervals.size:
                raise ValueError("interval rejection reasons must align with PPI intervals")
            ordinals = np.asarray(self.peak_ordinals)
            peaks = np.asarray(self.peaks)
            if (
                ordinals.shape != peaks.shape
                or not np.issubdtype(ordinals.dtype, np.integer)
                or np.any(ordinals < 0)
                or (ordinals.size > 1 and np.any(np.diff(ordinals) <= 0))
            ):
                raise ValueError("peak_ordinals must preserve unique increasing global identities")

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
    """每 recording 一个 D×K 矩阵 / One D-by-K matrix per recording."""

    values: np.ndarray
    row_mask: np.ndarray
    channel_schema: tuple[str, ...]
    context_schema: tuple[str, ...]
    schema_version: str
    provenance: dict[str, Any]

@dataclass(frozen=True)
class RoutingWindow:
    """One common 8 s/2 s evidence window on the canonical 400 Hz grid."""

    record_id: str
    routing_window_id: str
    start_s: float
    stop_s: float
    centre_s: float
    start_sample_400: int
    stop_sample_400: int

@dataclass(frozen=True)
class RoutingCell:
    """One non-overlapping ownership cell and its complete route provenance."""

    record_id: str
    participant_id: str
    role: str
    routing_window_id: str
    cell_id: str
    cell_start_s: float
    cell_stop_s: float
    start_sample_400: int
    stop_sample_400: int
    sqi_mode: str
    sqi_assessed: bool
    direct_q_rate_score: float | None
    direct_q_rate_state: str | None
    direct_q_morph_score: float | None
    direct_q_morph_state: str | None
    motion_detector_enabled: bool
    motion_probability: float | None
    motion_threshold: float | None
    motion_state: str
    pre_route_tier: str
    denoiser_enabled: bool
    denoiser_requested: bool
    denoiser_status: str
    post_q_rate_score: float | None
    post_q_rate_state: str | None
    final_tier: str
    source_route: str
    source_view: str
    reason_codes: tuple[str, ...]
    config_sha256: str
    sqi_calibrator_sha256: str | None
    motion_model_sha256: str | None
    motion_input_schema_sha256: str | None
    reducer_sha256: str | None

@dataclass(frozen=True)
class RoutingTimeline:
    """Canonical chronological route ownership for one complete recording."""

    record_id: str
    participant_id: str
    role: str
    fs_hz: float
    n_samples: int
    windows: tuple[RoutingWindow, ...]
    cells: tuple[RoutingCell, ...]
    schema_version: str = "ppg_frailty.routing_timeline.v1"

    def validate(self) -> None:
        if not self.record_id or not self.participant_id or not self.role:
            raise ValueError("routing timeline identity fields must be non-empty")
        if float(self.fs_hz) != 400.0 or self.n_samples <= 0:
            raise ValueError("routing timeline requires a positive canonical 400 Hz grid")
        if not self.cells:
            raise ValueError("routing timeline requires at least one ownership cell")
        previous_stop = 0
        for index, cell in enumerate(self.cells):
            if cell.record_id != self.record_id:
                raise ValueError("routing cell record identity drift")
            if not 0 <= cell.start_sample_400 < cell.stop_sample_400 <= self.n_samples:
                raise ValueError("routing cell bounds are outside the recording")
            if cell.start_sample_400 != previous_stop:
                raise ValueError("routing cells must be contiguous and non-overlapping")
            if cell.cell_id != f"{self.record_id}::cell_{index:06d}":
                raise ValueError("routing cell identity/order drift")
            previous_stop = cell.stop_sample_400
        if previous_stop != self.n_samples:
            raise ValueError("routing cells must explicitly account for every sample")

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
