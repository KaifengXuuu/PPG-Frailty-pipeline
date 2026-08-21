"""Read-only reuse adapter for a frozen Stage5 Frailty29 motion bundle.

The adapter deliberately performs no fitting, calibration, threshold search,
or cross-validation.  It reuses the all-participant model and the deployment
threshold already stored by Stage5, applies the canonical 8 s / 2 s motion
materializer, and reduces window probabilities to one recording-level median.

Because the reused model was fitted on all 29 Frailty participants, every
result carries an explicit ``in_sample_for_frailty29`` provenance warning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..data.manifest import M2_DATASET_VERSION_ID
from ..provenance import stable_payload_sha256
from ..representations.motion import (
    MOTION_NETWORK_SCHEMA_SHA256,
    MOTION_WINDOW_SAMPLES,
    build_motion_window_tensors,
)
from ..signal.motion_imu import (
    MOTION_IMU_CHANNEL_SCHEMA,
    MOTION_IMU_CHANNEL_UNITS,
    MotionImuResult,
)
from ..signal.views import CANONICAL_FS_HZ, CanonicalSignalViews
from .motion import (
    MOTION_DEPLOYMENT_THRESHOLD_FIT_SCOPE,
    MOTION_DEPLOYMENT_THRESHOLD_SCHEMA,
    MOTION_DEPLOYMENT_THRESHOLD_SCORE_ORIGIN,
    MOTION_INTERNAL_EVIDENCE_SCHEMA,
    MOTION_PARTICIPANT_COUNT,
)
from .motion_adapters import (
    FormalMotionRuntime,
    MotionRecordingInput,
    load_formal_motion_model,
    predict_formal_motion_probability,
)
from .motion_runner import MotionPredictionInput, load_motion_internal_evidence


MOTION_BUNDLE_REUSE_SCHEMA = "ppg_frailty.motion_bundle_reuse.v1"
_CUDA_DEVICE = re.compile(r"^cuda(?::[0-9]+)?$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ReusedMotionDetectorConfig:
    """Minimal independently switchable motion-detector configuration."""

    enabled: bool = False
    evidence_path: Path | None = None
    expected_evidence_sha256: str | None = None
    device: str = "cuda"
    batch_size: int = 64
    window_probability_aggregation: str = "median"
    threshold_source: str = "bundle_frozen"

    def to_mapping(self, *, include_enabled: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "evidence_path": (
                None if self.evidence_path is None else str(self.evidence_path)
            ),
            "expected_evidence_sha256": self.expected_evidence_sha256,
            "device": self.device,
            "batch_size": self.batch_size,
            "window_probability_aggregation": self.window_probability_aggregation,
            "threshold_source": self.threshold_source,
        }
        if include_enabled:
            payload = {"enabled": self.enabled, **payload}
        return payload

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("motion detector enabled must be bool")
        if self.enabled and self.evidence_path is None:
            raise ValueError("enabled reused motion detector requires evidence_path")
        if self.evidence_path is not None and not isinstance(self.evidence_path, Path):
            raise TypeError(
                "motion detector evidence_path must resolve to pathlib.Path"
            )
        if self.enabled and self.expected_evidence_sha256 is None:
            raise ValueError(
                "enabled reused motion detector requires expected_evidence_sha256"
            )
        if self.expected_evidence_sha256 is not None and (
            not isinstance(self.expected_evidence_sha256, str)
            or not _SHA256.fullmatch(self.expected_evidence_sha256)
        ):
            raise ValueError(
                "motion detector expected_evidence_sha256 must be lowercase SHA-256"
            )
        if not isinstance(self.device, str) or (
            self.device != "cpu" and not _CUDA_DEVICE.fullmatch(self.device)
        ):
            raise ValueError("motion detector device must be cpu, cuda, or cuda:N")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise TypeError("motion detector batch_size must be int")
        if self.batch_size < 1:
            raise ValueError("motion detector batch_size must be positive")
        if self.window_probability_aggregation != "median":
            raise ValueError(
                "reused motion detector currently supports recording median only"
            )
        if not isinstance(self.threshold_source, str) or (
            self.threshold_source != "bundle_frozen"
        ):
            raise ValueError("reused motion detector requires bundle_frozen threshold")


def resolve_reused_motion_detector_config(
    payload: Mapping[str, Any] | None = None,
) -> ReusedMotionDetectorConfig:
    """Resolve a typed config without normalizing legacy artifact fields."""

    values = {} if payload is None else dict(payload)
    allowed = {
        "enabled",
        "evidence_path",
        "expected_evidence_sha256",
        "device",
        "batch_size",
        "window_probability_aggregation",
        "threshold_source",
    }
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(
            "unknown reused motion detector configuration fields: "
            + ", ".join(unknown)
        )
    if "evidence_path" in values and values["evidence_path"] is not None:
        values["evidence_path"] = Path(str(values["evidence_path"])).expanduser()
    config = ReusedMotionDetectorConfig(**values)
    config.validate()
    return config


@dataclass(frozen=True)
class LoadedReusedMotionDetector:
    """Strict Stage5 runtime plus its immutable recording decision contract."""

    runtime: FormalMotionRuntime
    threshold: float
    ekf_config_sha256: str
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class MotionRecordDecision:
    """Recording-level low/high result, or fail-closed signal-level Unfit."""

    motion_state: str
    record_probability: float | None
    threshold: float
    window_count: int
    reason: str

    def validate(self) -> None:
        if self.motion_state not in {"low_motion", "high_motion", "unfit"}:
            raise ValueError("unknown recording motion state")
        if self.window_count < 0:
            raise ValueError("motion decision window count cannot be negative")
        if not np.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("motion decision threshold must lie in [0, 1]")
        if self.motion_state == "unfit":
            if self.record_probability is not None or self.window_count != 0:
                raise ValueError("unfit motion decision cannot carry a usable score")
        elif (
            self.record_probability is None
            or not np.isfinite(float(self.record_probability))
            or not 0.0 <= float(self.record_probability) <= 1.0
            or self.window_count < 1
        ):
            raise ValueError("usable motion decision requires probability and windows")


def _resolve_model_path(evidence_path: Path, value: object) -> Path:
    declared = Path(str(value)).expanduser()
    candidate = declared if declared.is_absolute() else evidence_path.parent / declared
    if candidate.is_file():
        return candidate.resolve()
    # Stage5 evidence may be copied as a self-contained archive while retaining
    # the original absolute path.  Resolve the canonical adjacent bundle member
    # without depending on a study timestamp or model identifier.
    adjacent = evidence_path.parent / "final_all_internal" / declared.name
    if adjacent.is_file():
        return adjacent.resolve()
    raise FileNotFoundError(f"motion model artifact not found: {candidate.resolve()}")


def _validated_frozen_threshold(
    evidence: Mapping[str, Any],
    training_participant_ids: tuple[str, ...],
) -> tuple[float, str]:
    payload = evidence.get("final_threshold")
    if not isinstance(payload, Mapping):
        raise ValueError("motion bundle final_threshold is missing")
    if (
        payload.get("schema_version") != MOTION_DEPLOYMENT_THRESHOLD_SCHEMA
        or payload.get("score_origin")
        != MOTION_DEPLOYMENT_THRESHOLD_SCORE_ORIGIN
        or payload.get("fit_scope") != MOTION_DEPLOYMENT_THRESHOLD_FIT_SCOPE
    ):
        raise ValueError("motion bundle threshold is not frozen from internal OOF")
    participant_ids = tuple(
        sorted(str(value) for value in payload.get("participant_ids", ()))
    )
    if participant_ids != training_participant_ids:
        raise ValueError("motion bundle model and threshold participant rosters differ")
    try:
        threshold = float(payload["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("motion bundle threshold is missing or non-numeric") from exc
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("motion bundle threshold must lie in [0, 1]")
    threshold_sha256 = stable_payload_sha256(payload)
    if evidence.get("final_threshold_artifact_sha256") != threshold_sha256:
        raise ValueError("motion bundle frozen threshold SHA-256 mismatch")
    return threshold, threshold_sha256


def load_reused_motion_detector(
    config: ReusedMotionDetectorConfig,
) -> LoadedReusedMotionDetector:
    """Load one frozen all-29 model; never fit it inside frailty CV."""

    config.validate()
    if (
        not config.enabled
        or config.evidence_path is None
        or config.expected_evidence_sha256 is None
    ):
        raise ValueError("cannot load a disabled reused motion detector")
    evidence_path = config.evidence_path.expanduser().resolve()
    evidence, evidence_sha256 = load_motion_internal_evidence(
        evidence_path,
        expected_sha256=config.expected_evidence_sha256,
    )
    if evidence.get("schema_version") != MOTION_INTERNAL_EVIDENCE_SCHEMA:
        raise ValueError("motion bundle is not Stage5 internal evidence")
    if (
        evidence.get("execution_status") != "completed_formal_not_smoke"
        or evidence.get("scientific_scope") != "frailty29_single_sgkf5_oof"
        or evidence.get("participant_count") != MOTION_PARTICIPANT_COUNT
    ):
        raise ValueError("motion bundle is not a completed formal Frailty29 run")
    source_evidence = evidence.get("formal_source_evidence")
    if not isinstance(source_evidence, Mapping):
        raise ValueError("motion bundle formal_source_evidence is missing")
    trained_ekf_config_sha256 = source_evidence.get("ekf_config_sha256")
    if not isinstance(trained_ekf_config_sha256, str) or not _SHA256.fullmatch(
        trained_ekf_config_sha256
    ):
        raise ValueError("motion bundle EKF configuration SHA-256 is missing")
    model = evidence.get("final_model")
    if not isinstance(model, Mapping):
        raise ValueError("motion bundle final_model is missing")
    roster = tuple(
        sorted(str(value) for value in model.get("training_participant_ids", ()))
    )
    if len(roster) != MOTION_PARTICIPANT_COUNT or len(set(roster)) != len(roster):
        raise ValueError("reused motion model must contain the exact 29-person roster")
    if model.get("model_input_schema_sha256") != MOTION_NETWORK_SCHEMA_SHA256:
        raise ValueError("reused motion model input schema drift")
    frozen_threshold, threshold_sha256 = _validated_frozen_threshold(
        evidence, roster
    )
    model_path = _resolve_model_path(evidence_path, model.get("artifact_path"))
    metadata = {
        "artifact_sha256": model.get("artifact_sha256"),
        "training_participant_ids": list(roster),
        "parameter_count": model.get("parameter_count"),
        "inference_cost": model.get("inference_cost"),
        "model_input_schema_sha256": model.get("model_input_schema_sha256"),
    }
    runtime = load_formal_motion_model(
        model_path,
        metadata,
        runtime_device=config.device,
    )
    runtime.batch_size = config.batch_size
    selected_threshold = frozen_threshold
    provenance = {
        "schema_version": MOTION_BUNDLE_REUSE_SCHEMA,
        "execution": "inference_only_no_fit_no_recalibration",
        "training_scope": "frailty29_all_participants",
        "reuse_scope": "all29_reused",
        "frailty29_evaluation_relation": "in_sample_for_frailty29",
        "valid_outer_oof_claim": False,
        "model_id": evidence.get("model_id"),
        "evidence_path": str(evidence_path),
        "expected_evidence_sha256": config.expected_evidence_sha256,
        "evidence_sha256": evidence_sha256,
        "model_artifact_path": str(model_path),
        "model_artifact_sha256": model.get("artifact_sha256"),
        "model_input_schema_sha256": model.get("model_input_schema_sha256"),
        "training_participant_ids": list(roster),
        "ekf_config_sha256": trained_ekf_config_sha256,
        "frozen_bundle_threshold": frozen_threshold,
        "frozen_bundle_threshold_sha256": threshold_sha256,
        "threshold_source": config.threshold_source,
        "selected_threshold": selected_threshold,
        "runtime_device": config.device,
        "inference_batch_size": config.batch_size,
        "window_probability_aggregation": config.window_probability_aggregation,
        "expected_ppg_input_source": "CanonicalSignalViews.x_native",
        "stage5_training_ppg_source": "manifest_numeric_values_columns_RED_IR",
        "ppg_source_equivalence_requirement": (
            "x_native_on_same_source_with_no_gap_repair"
        ),
        "source_qc_difference": (
            "canonical_ppg_qc_precedes_inference_whereas_stage5_used_"
            "physical_recording_qc_without_signal_repair"
        ),
        "window_seconds": 8.0,
        "hop_seconds": 2.0,
    }
    return LoadedReusedMotionDetector(
        runtime=runtime,
        threshold=selected_threshold,
        ekf_config_sha256=trained_ekf_config_sha256,
        provenance=provenance,
    )


def _fail_closed_unfit(
    detector: LoadedReusedMotionDetector,
    *,
    reason: str,
) -> MotionRecordDecision:
    result = MotionRecordDecision(
        motion_state="unfit",
        record_probability=None,
        threshold=detector.threshold,
        window_count=0,
        reason=reason,
    )
    result.validate()
    return result


def motion_recording_from_signal_views(
    views: CanonicalSignalViews,
    *,
    detector: LoadedReusedMotionDetector,
    record_id: str,
    participant_id: str,
    role: str,
) -> MotionRecordingInput:
    """Rebuild the formal input from native PPG and real EKF audit arrays.

    Stage5 trained on the unfiltered RED/IR source, so this adapter uses
    ``x_native`` rather than ``x_filter``.  A repaired native source is rejected:
    the frozen Stage5 materializer admitted finite physical source rows but did
    not perform the canonical gap-repair mutation.
    """

    views.validate()
    if not record_id or not participant_id or not role:
        raise ValueError("motion recording identity fields must be non-empty")
    metadata_record_id = views.metadata.get("record_id")
    if metadata_record_id is not None and str(metadata_record_id) != record_id:
        raise ValueError("motion record_id differs from CanonicalSignalViews metadata")
    if np.any(np.asarray(views.repair_mask, dtype=bool)):
        raise ValueError(
            "reused Stage5 motion detector forbids gap-repaired native PPG"
        )
    required = {
        "dynamic_acc_mps2",
        "gyro_rads",
        "dynamic_magnitude",
        "gyro_magnitude",
        "jerk_magnitude",
        "roll_rad",
        "pitch_rad",
        "gravity_mps2",
        "imu_valid_mask",
    }
    missing = sorted(required - set(views.imu_processed))
    if missing:
        raise ValueError(
            "CanonicalSignalViews lacks formal motion IMU fields: "
            + ", ".join(missing)
        )
    dynamic = np.asarray(views.imu_processed["dynamic_acc_mps2"], dtype=np.float64)
    gyro = np.asarray(views.imu_processed["gyro_rads"], dtype=np.float64)
    values = np.column_stack(
        (
            dynamic,
            gyro,
            np.asarray(views.imu_processed["dynamic_magnitude"], dtype=np.float64),
            np.asarray(views.imu_processed["gyro_magnitude"], dtype=np.float64),
            np.asarray(views.imu_processed["jerk_magnitude"], dtype=np.float64),
        )
    )
    diagnostics = views.metadata.get("imu_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("CanonicalSignalViews lacks formal IMU diagnostics")
    runtime_ekf_config_sha256 = diagnostics.get("ekf_config_sha256")
    if not isinstance(runtime_ekf_config_sha256, str) or not _SHA256.fullmatch(
        runtime_ekf_config_sha256
    ):
        raise ValueError("CanonicalSignalViews EKF configuration SHA-256 is missing")
    if runtime_ekf_config_sha256 != detector.ekf_config_sha256:
        raise ValueError("reused motion bundle/runtime EKF configuration mismatch")
    profile_id = str(views.metadata.get("gravity_method", ""))
    motion_imu = MotionImuResult(
        values=values,
        channel_schema=MOTION_IMU_CHANNEL_SCHEMA,
        channel_units=MOTION_IMU_CHANNEL_UNITS,
        roll_rad=np.asarray(views.imu_processed["roll_rad"], dtype=np.float64),
        pitch_rad=np.asarray(views.imu_processed["pitch_rad"], dtype=np.float64),
        gravity_mps2=np.asarray(
            views.imu_processed["gravity_mps2"], dtype=np.float64
        ),
        valid_mask=np.asarray(views.imu_processed["imu_valid_mask"], dtype=bool),
        profile_id=profile_id,
        diagnostics=dict(diagnostics),
    )
    result = MotionRecordingInput(
        ppg_red_ir=np.asarray(views.x_native, dtype=np.float64),
        motion_imu=motion_imu,
        record_id=record_id,
        participant_id=participant_id,
        role_or_activity=role,
        dataset_id=M2_DATASET_VERSION_ID,
        fs_hz=CANONICAL_FS_HZ,
    )
    result.validate()
    return result


def infer_reused_motion_recording(
    detector: LoadedReusedMotionDetector,
    recording: MotionRecordingInput | None,
) -> MotionRecordDecision:
    """Infer one record with median aggregation and fail-closed missing data."""

    if recording is None:
        return _fail_closed_unfit(
            detector,
            reason="motion_signal_missing",
        )
    motion_imu = getattr(recording, "motion_imu", None)
    if motion_imu is None or getattr(motion_imu, "values", None) is None:
        return _fail_closed_unfit(
            detector,
            reason="motion_signal_missing",
        )
    ppg = np.asarray(recording.ppg_red_ir)
    imu_values = np.asarray(motion_imu.values)
    if (
        ppg.ndim != 2
        or ppg.shape[1] != 2
        or imu_values.ndim != 2
        or imu_values.shape[1] < 6
    ):
        return _fail_closed_unfit(
            detector,
            reason="motion_signal_missing",
        )
    if min(ppg.shape[0], imu_values.shape[0]) < MOTION_WINDOW_SAMPLES:
        return _fail_closed_unfit(
            detector,
            reason="no_complete_8_second_motion_window",
        )

    recording.validate()
    windows = build_motion_window_tensors(
        recording.ppg_red_ir,
        recording.motion_imu,
        record_id=recording.record_id,
        participant_id=recording.participant_id,
        role_or_activity=recording.role_or_activity,
        dataset_id=recording.dataset_id,
        fs_hz=recording.fs_hz,
    )
    rows = tuple(MotionPredictionInput(value) for value in windows.values)
    probabilities = predict_formal_motion_probability(detector.runtime, rows)
    if (
        probabilities.shape != (len(rows),)
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0.0) | (probabilities > 1.0))
    ):
        return _fail_closed_unfit(
            detector,
            reason="motion_probability_unavailable",
        )
    record_probability = float(np.median(probabilities))
    low_motion = record_probability < detector.threshold
    result = MotionRecordDecision(
        motion_state="low_motion" if low_motion else "high_motion",
        record_probability=record_probability,
        threshold=detector.threshold,
        window_count=len(rows),
        reason="record_median_below_frozen_threshold" if low_motion else (
            "record_median_at_or_above_frozen_threshold"
        ),
    )
    result.validate()
    return result


__all__ = [
    "LoadedReusedMotionDetector",
    "MOTION_BUNDLE_REUSE_SCHEMA",
    "MotionRecordDecision",
    "ReusedMotionDetectorConfig",
    "infer_reused_motion_recording",
    "load_reused_motion_detector",
    "motion_recording_from_signal_views",
    "resolve_reused_motion_detector_config",
]
