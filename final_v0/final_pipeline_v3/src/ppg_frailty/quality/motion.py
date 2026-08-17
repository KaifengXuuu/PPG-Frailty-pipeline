"""Fail-closed V2 activity/motion supervision and evaluation contract.

This module registers contracts; importing it never trains a model, runs cross
validation, evaluates PTT, or executes an ablation. The formal local detector
must be trained on all 29 internal participants through one materialized
participant-grouped SGKF5 split with split seed 42. PTT is a subsequent cross-dataset
evaluation and cannot open until complete internal OOF evidence is archived.

The supervised target is protocol activity state, not optical-artifact truth:
canonical B/R are static (0), while S/W are motion (1).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..data.external_folds import (
    PTT_FORMAL_FOLD_SIZES,
    PTT_FORMAL_REGISTRY_ID,
    PTT_FORMAL_REPEAT_SEEDS,
)
from ..data.external_manifest import (
    PTT_ADOPTED_ACCELERATION_CONVERSION,
    PTT_ADOPTED_ACCELERATION_UNIT,
    PTT_ADOPTED_GYROSCOPE_CONVERSION,
    PTT_ADOPTED_GYROSCOPE_UNIT,
    PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH,
    PTT_IMU_UNIT_EVIDENCE_SHA256,
)
from ..data.folds import (
    M2_SPLIT_FILE_SHA256,
    M2_SPLIT_PAYLOAD_SHA256,
    FrozenFoldRegistry,
)
from ..data.manifest import M2_DATASET_VERSION_ID
from ..data.schema import canonicalize_role_family
from ..motion_ids import FORMAL_MOTION_MODEL_ID, HISTORICAL_LIGHT_CNN_MODEL_ID
from ..representations.motion import (
    MOTION_HOP_SAMPLES,
    MOTION_HOP_SECONDS,
    MOTION_NETWORK_CHANNEL_SCHEMA,
    MOTION_NETWORK_CHANNEL_UNITS,
    MOTION_NETWORK_SCHEMA_SHA256,
    MOTION_WINDOW_SAMPLES,
    MOTION_WINDOW_SECONDS,
)
from ..scientific_gate import (
    MOTION_INTERNAL_GATE_PHASES,
    MOTION_REQUIRED_DEPENDENCY_PROFILES,
    MOTION_SCIENTIFIC_GATE_CONFIG_SCHEMA,
    SCIENTIFIC_SOURCE_SNAPSHOT_ROOTS,
    verify_scientific_gate_evidence,
)

HISTORICAL_LIGHT_CNN_CHANNELS = (
    "ppg_single_historical_loader_pleth_2_first",
    "acc_dynamic_x",
    "acc_dynamic_y",
    "acc_dynamic_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "acc_magnitude",
    "gyro_magnitude",
    "jerk_magnitude",
)


MOTION_CONTRACT_SCHEMA = "ppg_frailty.motion_detector_contract.v2"
MOTION_INTERNAL_EVIDENCE_SCHEMA = "ppg_frailty.motion_internal_evidence.v2"
MOTION_MIDPOINT_THRESHOLD_SCHEMA = "ppg_frailty.motion_midpoint_threshold.v2"
MOTION_SPLIT_REGISTRY_ID = "frailty29_single_sgkf5_seed42_v2"
MOTION_SOURCE_SPLIT_REGISTRY_ID = "frailty3_future_corrected_sgkf5_v2"
MOTION_SPLIT_CSV_SHA256 = (
    "130b2887eb29a5a534397b4ce4dc7032f9de30ae46533fa0b2c41559ff4a1284"
)
PTT_SPLIT_CSV_SHA256 = (
    "138f999da334e2280c9e7df3304f9f017496f0ed6b60562a8d33e3c69c502dec"
)
MOTION_PARTICIPANT_COUNT = 29
MOTION_REPEAT_COUNT = 1
MOTION_FOLD_COUNT = 5
MOTION_COMPLETED_CELL_COUNT = MOTION_REPEAT_COUNT * MOTION_FOLD_COUNT
MOTION_OOF_PARTICIPANT_REPEAT_ROWS = MOTION_PARTICIPANT_COUNT * MOTION_REPEAT_COUNT
MOTION_SPLIT_SEED = 42
MOTION_TRAINING_SEED = 42
MOTION_THRESHOLD_RULE_ID = "participant_balanced_class_median_midpoint_train_only_v1"
MOTION_THRESHOLD_SCORE_ORIGIN = "outer_training_fit_predictions_only"
MOTION_THRESHOLD_FIT_SCOPE = "outer_training_participants_only_no_oof_no_ptt"
FORMAL_MOTION_TRAINING_CONFIG = {
    "fixed_epochs": 10,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "loss": "binary_cross_entropy_with_logits",
    "class_weighting": "none_balancing_is_sampler_only",
    "sampler": "historical_dataset_class_inverse_x_participant_sqrt",
    "dropout": 0.0,
    "label_smoothing": 0.0,
    "gradient_clip_norm": 1.0,
    "seed": 42,
    "num_workers": 0,
    "augmentation": "none",
    "device": "cpu",
    "inference_warmup_iterations": 10,
    "inference_timed_iterations": 50,
    "schema_version": "ppg_frailty.formal_motion_trainer.v2",
}

FORMAL_RAW_INPUT_SOURCES = (
    "RED_optical_raw_source",
    "IR_optical_raw_source",
    "AX_accelerometer_raw_source",
    "AY_accelerometer_raw_source",
    "AZ_accelerometer_raw_source",
    "GX_gyroscope_raw_source",
    "GY_gyroscope_raw_source",
    "GZ_gyroscope_raw_source",
)
FORMAL_NETWORK_TENSOR_STATUS = "frozen_before_training"

MOTION_MAJOR_METRIC_FIELDS = (
    "participant_macro_balanced_accuracy",
    "worst_fold_balanced_accuracy",
    "participant_macro_f1",
    "ece",
    "parameter_count",
    "inference_cost",
)
MOTION_INFERENCE_COST_FIELDS = (
    "device",
    "batch_size",
    "window_samples",
    "warmup_iterations",
    "timed_iterations",
    "latency_ms_per_window_p50",
    "latency_ms_per_window_p95",
    "throughput_windows_per_second",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class MotionOptionId(str, Enum):
    """The three mutually exclusive V2-010 options."""

    SQI_ONLY = "sqi_only"
    SQI_PLUS_MOTION_OVERRIDE = "sqi_plus_motion_override"
    HISTORICAL_LIGHT_CNN_BACKUP = "historical_light_cnn_backup"


@dataclass(frozen=True)
class MotionOptionDescriptor:
    option_id: MotionOptionId
    formal_default: bool
    execution_status: str
    classifier_effect: str
    supervised_training: str
    external_evaluation: str
    notes: tuple[str, ...]


MOTION_OPTIONS = (
    MotionOptionDescriptor(
        option_id=MotionOptionId.SQI_ONLY,
        formal_default=True,
        execution_status="implemented_default",
        classifier_effect="none_motion_not_computed",
        supervised_training="not_applicable",
        external_evaluation="not_applicable",
        notes=(
            "V2-010 default; SQI itself is currently off by V2-009d unless separately requested",
            "does not retain, drop, aggregate, route, or predict from motion scores",
        ),
    ),
    MotionOptionDescriptor(
        option_id=MotionOptionId.SQI_PLUS_MOTION_OVERRIDE,
        formal_default=False,
        execution_status="implemented_contract_registered_not_run_gate_closed",
        classifier_effect="may_override_high_sqi_only_after_frozen_supervised_gate",
        supervised_training="frailty29_single_sgkf5_seed42_required",
        external_evaluation="ptt_cross_dataset_after_complete_internal_oof_only",
        notes=(
            "activity/motion supervision is B/R=static and S/W=motion",
            "network tensor schema and its SHA-256 must be frozen before training",
            "no YAML boolean can bypass the evidence gate",
        ),
    ),
    MotionOptionDescriptor(
        option_id=MotionOptionId.HISTORICAL_LIGHT_CNN_BACKUP,
        formal_default=False,
        execution_status="historical_frozen_backup_not_v2_run",
        classifier_effect="disabled_unless_explicit_historical_backup_run",
        supervised_training="historical_assets_only_no_v2_retraining_claim",
        external_evaluation="historical_sim_evidence_only_not_v2_ptt_test",
        notes=(
            "not the removed V2 IR-only 10-channel ablation",
            "architecture and archived asset hashes are reproducibility evidence only",
        ),
    ),
)


@dataclass(frozen=True)
class HistoricalLightCnnEvidence:
    """Immutable description of the audited 2026-04-27 backup evidence."""

    status: str
    source_run: str
    model_id: str
    source_detector_config_base_channels: int
    resolved_network_base_channels: int
    input_channels: tuple[str, ...]
    fs_hz: float
    window_s: float
    hop_s: float
    threshold: float
    external_dataset: str
    external_window_count: int
    external_balanced_accuracy: float
    external_f1: float
    external_roc_auc: float
    external_pr_auc: float
    pytorch_sha256: str
    onnx_sha256: str
    onnx_data_sha256: str
    metadata_sha256: str
    benchmark_summary_sha256: str
    parameter_count: int
    claim_boundary: str


HISTORICAL_LIGHT_CNN_EVIDENCE = HistoricalLightCnnEvidence(
    status="frozen_historical_evidence_only_not_v2_execution",
    source_run=".CNN_results/20260427-01_peak_hr_gate_balanced_v2",
    model_id=HISTORICAL_LIGHT_CNN_MODEL_ID,
    source_detector_config_base_channels=24,
    resolved_network_base_channels=12,
    input_channels=HISTORICAL_LIGHT_CNN_CHANNELS,
    fs_hz=256.0,
    window_s=8.0,
    hop_s=2.0,
    threshold=0.05,
    external_dataset="simultaneous_measurements",
    external_window_count=12032,
    external_balanced_accuracy=0.7802173951757236,
    external_f1=0.7633982422884715,
    external_roc_auc=0.8642102117410313,
    external_pr_auc=0.9054938134772867,
    pytorch_sha256="4bd4d8302dbfabd099a91e5478a274da5553b71737a41c48bf513034a66356e0",
    onnx_sha256="4280cd9fe1246f6a7cf32d76c2781a768d04608bb9b235f475a9eda5b08356c9",
    onnx_data_sha256="b1ef787dd416d35f24d5b1cd8f78697016dc218e0469d81ead0d3b481ebfb7f8",
    metadata_sha256="f189e8edbea7985a4c76e7c3908e5885d261ae93778a8330b8e228c193f845cb",
    benchmark_summary_sha256="0a34ae2656aac00a3cdeb789e582981745c5322ba90d3cf36d108ae925c8f7a0",
    parameter_count=6181,
    claim_boundary=(
        "pooled overlapping-window historical external-SIM result; no detector "
        "5x5 subject OOF, no participant bootstrap CI, and never a V2/PTT result"
    ),
)


@dataclass(frozen=True)
class MotionFoldJob:
    """One immutable outer evaluation cell from the corrected registry."""

    repeat_index: int
    fold_index: int
    split_seed: int
    training_seed: int
    train_participant_ids: tuple[str, ...]
    oof_participant_ids: tuple[str, ...]
    registry_id: str = MOTION_SPLIT_REGISTRY_ID
    registry_csv_sha256: str = MOTION_SPLIT_CSV_SHA256
    grouping_variable: str = "participant_id"
    runtime_split_recomputation_allowed: bool = False
    threshold_score_origin: str = MOTION_THRESHOLD_SCORE_ORIGIN


@dataclass(frozen=True)
class MidpointThresholdArtifact:
    """Train-only, participant-balanced midpoint threshold artifact."""

    schema_version: str
    threshold_rule_id: str
    score_origin: str
    fit_scope: str
    participant_ids: tuple[str, ...]
    participant_roster_sha256: str
    static_center: float
    motion_center: float
    threshold: float
    center_statistic: str
    participant_weighting: str
    score_space: str
    class_ids: tuple[int, int]
    observed_row_count: int

    def validate(self) -> None:
        if self.schema_version != MOTION_MIDPOINT_THRESHOLD_SCHEMA:
            raise ValueError("motion threshold schema drift")
        if self.threshold_rule_id != MOTION_THRESHOLD_RULE_ID:
            raise ValueError("motion threshold rule drift")
        if self.score_origin != MOTION_THRESHOLD_SCORE_ORIGIN:
            raise ValueError("motion threshold must use train-fit predictions only")
        if self.fit_scope != MOTION_THRESHOLD_FIT_SCOPE:
            raise ValueError("motion threshold fit scope drift")
        if not self.participant_ids:
            raise ValueError("motion threshold requires training participants")
        if not _SHA256.fullmatch(self.participant_roster_sha256):
            raise ValueError("motion threshold participant roster hash is invalid")
        expected = (self.static_center + self.motion_center) / 2.0
        if not math.isclose(self.threshold, expected, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("motion threshold is not the exact class-center midpoint")
        if not 0.0 <= self.static_center < self.motion_center <= 1.0:
            raise ValueError("motion class score centers are unordered or outside [0,1]")
        if not self.static_center < self.threshold < self.motion_center:
            raise ValueError("motion threshold does not lie strictly between classes")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class PttExternalGateDecision:
    """Fail-closed decision for the post-internal PTT evaluation stage."""

    allowed: bool
    reasons: tuple[str, ...]
    external_registry_id: str = PTT_FORMAL_REGISTRY_ID
    external_registry_csv_sha256: str = PTT_SPLIT_CSV_SHA256
    external_repeat_seeds: tuple[int, ...] = PTT_FORMAL_REPEAT_SEEDS
    external_fold_sizes: tuple[int, ...] = PTT_FORMAL_FOLD_SIZES
    independence_claim: str = "none_not_an_independent_external_test"
    action: str = "evaluation_only_never_fit_or_recalibrate"


def resolve_motion_option(value: MotionOptionId | str | None) -> MotionOptionDescriptor:
    """Resolve an explicit option; omission is the frozen SQI-only default."""

    option_id = MotionOptionId.SQI_ONLY if value is None else MotionOptionId(value)
    return next(item for item in MOTION_OPTIONS if item.option_id is option_id)


def motion_activity_label(role: str) -> int:
    """Map B/R to static=0 and S/W to motion=1 via the canonical role family."""

    family = canonicalize_role_family(role)
    return 0 if family in {"B", "R"} else 1


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_motion_fold_jobs(split_csv: str | Path) -> tuple[MotionFoldJob, ...]:
    """Load the five seed-42 jobs and bind them to the exact Frailty29 CSV.

    This loader does not invoke ``StratifiedGroupKFold`` at runtime.  It accepts
    only the reviewed, materialized registry and verifies its file identity,
    upstream registry identity, seeds, participant coverage, and disjointness.
    """

    source = Path(split_csv)
    if _sha256_file(source) != MOTION_SPLIT_CSV_SHA256:
        raise ValueError("motion split CSV SHA-256 drift")
    registry = FrozenFoldRegistry.from_csv(source)
    rows = registry.assignments
    expected_identity = {
        (row.source_registry_id, row.source_registry_file_sha256,
         row.source_registry_payload_sha256, row.dataset_version_id)
        for row in rows
    }
    if expected_identity != {
        (
            MOTION_SOURCE_SPLIT_REGISTRY_ID,
            M2_SPLIT_FILE_SHA256,
            M2_SPLIT_PAYLOAD_SHA256,
            M2_DATASET_VERSION_ID,
        )
    }:
        raise ValueError("motion split registry provenance drift")
    if registry.participant_ids and len(registry.participant_ids) != MOTION_PARTICIPANT_COUNT:
        raise ValueError("motion split registry must contain exactly 29 participants")

    jobs: list[MotionFoldJob] = []
    for repeat_index, expected_seed in enumerate((MOTION_SPLIT_SEED,)):
        repeat_rows = [row for row in rows if row.repeat_index == repeat_index]
        if {row.split_seed for row in repeat_rows} != {expected_seed}:
            raise ValueError("motion split seed drift")
        for fold_index in range(MOTION_FOLD_COUNT):
            split = registry.get_split(repeat_index, fold_index)
            train = tuple(split["train_participant_ids"])
            oof = tuple(split["oof_participant_ids"])
            if set(train) & set(oof):
                raise ValueError("motion fold train/OOF overlap")
            if set(train) | set(oof) != set(registry.participant_ids):
                raise ValueError("motion fold is not a complete participant partition")
            jobs.append(
                MotionFoldJob(
                    repeat_index=repeat_index,
                    fold_index=fold_index,
                    split_seed=expected_seed,
                    training_seed=MOTION_TRAINING_SEED,
                    train_participant_ids=train,
                    oof_participant_ids=oof,
                )
            )
    if len(jobs) != MOTION_COMPLETED_CELL_COUNT:
        raise ValueError("motion split registry did not resolve the five SGKF cells")
    return tuple(jobs)


def _roster_sha256(participant_ids: Sequence[str]) -> str:
    encoded = json.dumps(
        sorted(set(participant_ids)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fit_train_only_midpoint_threshold(
    scores: Sequence[float],
    labels: Sequence[int],
    participant_ids: Sequence[str],
    *,
    training_participant_ids: Iterable[str],
    forbidden_oof_participant_ids: Iterable[str] = (),
    forbidden_ptt_participant_ids: Iterable[str] = (),
    score_origin: str = MOTION_THRESHOLD_SCORE_ORIGIN,
) -> MidpointThresholdArtifact:
    """Fit the exact midpoint between participant-balanced class medians.

    First, each training participant contributes one median score per class.
    Second, the cohort class center is the median across participants.  The
    threshold is the exact arithmetic midpoint of those two centers.  This
    prevents B/R's greater window count from dominating the threshold.

    Scores must come from the model fitted on this outer-training partition.
    Outer OOF and PTT participants are explicitly forbidden, and no BA/F1
    optimization is performed.
    """

    if score_origin != MOTION_THRESHOLD_SCORE_ORIGIN:
        raise ValueError("midpoint threshold may not use OOF, PTT, or tuned score origins")
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    id_array = np.asarray([str(value) for value in participant_ids], dtype=object)
    if score_array.ndim != 1 or label_array.ndim != 1 or id_array.ndim != 1:
        raise ValueError("motion threshold inputs must be one-dimensional")
    if not (score_array.size == label_array.size == id_array.size) or score_array.size == 0:
        raise ValueError("motion threshold inputs must be non-empty and length-aligned")
    if not np.all(np.isfinite(score_array)) or np.any((score_array < 0.0) | (score_array > 1.0)):
        raise ValueError("motion scores must be finite probabilities in [0,1]")
    if set(label_array.tolist()) != {0, 1}:
        raise ValueError("motion midpoint threshold requires exactly static=0 and motion=1")

    training = {str(value) for value in training_participant_ids}
    observed = set(id_array.tolist())
    if not training or observed != training:
        raise ValueError("threshold rows must cover exactly the declared training roster")
    forbidden = {
        str(value)
        for value in (*tuple(forbidden_oof_participant_ids), *tuple(forbidden_ptt_participant_ids))
    }
    if observed & forbidden:
        raise ValueError("OOF/PTT participant leaked into motion threshold fitting")

    participant_centers = {0: [], 1: []}
    for participant in sorted(training):
        participant_mask = id_array == participant
        for class_id in (0, 1):
            values = score_array[participant_mask & (label_array == class_id)]
            if values.size == 0:
                raise ValueError(
                    f"training participant {participant!r} lacks motion class {class_id}"
                )
            participant_centers[class_id].append(float(np.median(values)))
    static_center = float(np.median(participant_centers[0]))
    motion_center = float(np.median(participant_centers[1]))
    artifact = MidpointThresholdArtifact(
        schema_version=MOTION_MIDPOINT_THRESHOLD_SCHEMA,
        threshold_rule_id=MOTION_THRESHOLD_RULE_ID,
        score_origin=MOTION_THRESHOLD_SCORE_ORIGIN,
        fit_scope=MOTION_THRESHOLD_FIT_SCOPE,
        participant_ids=tuple(sorted(training)),
        participant_roster_sha256=_roster_sha256(tuple(training)),
        static_center=static_center,
        motion_center=motion_center,
        threshold=(static_center + motion_center) / 2.0,
        center_statistic="median_of_participant_class_medians",
        participant_weighting="each_training_participant_equal_within_each_class",
        score_space="calibrated_p_active_probability",
        class_ids=(0, 1),
        observed_row_count=int(score_array.size),
    )
    artifact.validate()
    return artifact


def validate_motion_major_metrics(payload: Mapping[str, Any]) -> None:
    """Validate the user-confirmed compact comparison report schema."""

    missing = set(MOTION_MAJOR_METRIC_FIELDS) - set(payload)
    if missing:
        raise ValueError(f"motion major metrics missing fields: {sorted(missing)}")
    for name in MOTION_MAJOR_METRIC_FIELDS[:4]:
        value = float(payload[name])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"motion metric {name} must be finite in [0,1]")
    parameter_count = payload["parameter_count"]
    if not isinstance(parameter_count, int) or isinstance(parameter_count, bool) or parameter_count <= 0:
        raise ValueError("motion parameter_count must be a positive integer")
    inference = payload["inference_cost"]
    if not isinstance(inference, Mapping):
        raise ValueError("motion inference_cost must be a mapping")
    missing_cost = set(MOTION_INFERENCE_COST_FIELDS) - set(inference)
    if missing_cost:
        raise ValueError(f"motion inference cost missing fields: {sorted(missing_cost)}")
    if not str(inference["device"]).strip():
        raise ValueError("motion inference device must be explicit")
    for name in ("batch_size", "window_samples", "warmup_iterations", "timed_iterations"):
        value = inference[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"motion inference {name} must be a positive integer")
    for name in (
        "latency_ms_per_window_p50",
        "latency_ms_per_window_p95",
        "throughput_windows_per_second",
    ):
        value = float(inference[name])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"motion inference {name} must be finite and positive")
    if float(inference["latency_ms_per_window_p95"]) < float(
        inference["latency_ms_per_window_p50"]
    ):
        raise ValueError("motion inference p95 latency cannot be below p50")


def _evaluate_ptt_external_gate_payload(
    internal_evidence: Mapping[str, Any],
) -> PttExternalGateDecision:
    """Open PTT evaluation only after complete, frozen internal formal evidence.

    The returned action is evaluation-only.  PTT values may not fit the model,
    calibrator, tensor schema, or midpoint threshold.  The PTT registry itself
    carries ``none_not_an_independent_external_test`` and this function preserves
    that claim boundary.
    """

    reasons: list[str] = []
    evidence = dict(internal_evidence)
    if (
        evidence.get("formal_entry_id")
        != "formal_internal_motion_reference_source_bound_v2"
    ):
        reasons.append("canonical_internal_motion_entry_missing")
    source_evidence = evidence.get("formal_source_evidence")
    if (
        not isinstance(source_evidence, Mapping)
        or source_evidence.get("formal_entry_id")
        != "formal_internal_motion_reference_source_bound_v2"
    ):
        reasons.append("canonical_internal_source_evidence_missing")
    if evidence.get("schema_version") != MOTION_INTERNAL_EVIDENCE_SCHEMA:
        reasons.append("missing_or_wrong_internal_evidence_schema")
    if evidence.get("execution_status") != "completed_formal_not_smoke":
        reasons.append("internal_sgkf5_not_completed_formally")
    if evidence.get("scientific_scope") != "frailty29_single_sgkf5_oof":
        reasons.append("internal_evidence_scope_is_not_strict_oof")
    scientific_gate_evidence = evidence.get("scientific_execution_gate")
    reasons.extend(
        verify_scientific_gate_evidence(
            scientific_gate_evidence,
            expected_phases=MOTION_INTERNAL_GATE_PHASES,
        )
    )
    scientific_checkpoints = (
        {
            str(item["phase"]): str(item["checkpoint_sha256"])
            for item in scientific_gate_evidence.get("checkpoints", ())
            if isinstance(item, Mapping)
            and "phase" in item
            and "checkpoint_sha256" in item
        }
        if isinstance(scientific_gate_evidence, Mapping)
        else {}
    )
    if evidence.get("model_id") != FORMAL_MOTION_MODEL_ID:
        reasons.append("formal_local_motion_model_id_missing")
    if evidence.get("split_registry_id") != MOTION_SPLIT_REGISTRY_ID:
        reasons.append("internal_split_registry_id_drift")
    if evidence.get("source_split_registry_id") != MOTION_SOURCE_SPLIT_REGISTRY_ID:
        reasons.append("internal_source_split_registry_id_drift")
    if evidence.get("split_registry_csv_sha256") != MOTION_SPLIT_CSV_SHA256:
        reasons.append("internal_split_registry_hash_drift")
    if evidence.get("upstream_split_registry_file_sha256") != M2_SPLIT_FILE_SHA256:
        reasons.append("internal_upstream_split_file_hash_drift")
    if (
        evidence.get("upstream_split_registry_payload_sha256")
        != M2_SPLIT_PAYLOAD_SHA256
    ):
        reasons.append("internal_upstream_split_payload_hash_drift")
    if evidence.get("participant_count") != MOTION_PARTICIPANT_COUNT:
        reasons.append("internal_participant_count_not_29")
    if evidence.get("oof_participant_repeat_rows") != MOTION_OOF_PARTICIPANT_REPEAT_ROWS:
        reasons.append("internal_oof_repeat_coverage_incomplete")
    if evidence.get("model_input_schema_status") != "frozen_before_training":
        reasons.append("model_input_tensor_schema_not_frozen_before_training")
    if evidence.get("model_input_schema_sha256") != MOTION_NETWORK_SCHEMA_SHA256:
        reasons.append("model_input_tensor_semantic_schema_hash_drift")
    if not _SHA256.fullmatch(
        str(evidence.get("model_input_schema_file_sha256", ""))
    ):
        reasons.append("model_input_tensor_schema_file_hash_missing")
    if evidence.get("threshold_rule_id") != MOTION_THRESHOLD_RULE_ID:
        reasons.append("midpoint_threshold_rule_drift")
    if evidence.get("threshold_score_origin") != MOTION_THRESHOLD_SCORE_ORIGIN:
        reasons.append("threshold_used_oof_ptt_or_unapproved_scores")

    cell_evidence = evidence.get("cell_evidence", [])
    observed_cells: set[tuple[int, int]] = set()
    if not isinstance(cell_evidence, list):
        reasons.append("cell_evidence_not_a_list")
    else:
        for row in cell_evidence:
            if not isinstance(row, Mapping):
                reasons.append("invalid_cell_evidence_row")
                continue
            try:
                cell = (int(row["repeat_index"]), int(row["fold_index"]))
            except (KeyError, TypeError, ValueError):
                reasons.append("invalid_cell_identity")
                continue
            if cell in observed_cells:
                reasons.append("duplicate_internal_cell_evidence")
            observed_cells.add(cell)
            if row.get("threshold_fit_scope") != MOTION_THRESHOLD_FIT_SCOPE:
                reasons.append("cell_threshold_scope_drift")
            for field in ("model_artifact_sha256", "threshold_artifact_sha256"):
                if not _SHA256.fullmatch(str(row.get(field, ""))):
                    reasons.append(f"cell_{field}_missing")
            if (
                row.get("scientific_gate_fit_pre_checkpoint_sha256")
                != scientific_checkpoints.get(
                    f"fold_{cell[0]}_{cell[1]}_fit_pre"
                )
                or row.get("scientific_gate_fit_post_checkpoint_sha256")
                != scientific_checkpoints.get(
                    f"fold_{cell[0]}_{cell[1]}_fit_post"
                )
            ):
                reasons.append("cell_scientific_gate_checkpoint_binding_drift")
        expected_cells = {
            (repeat_index, fold_index)
            for repeat_index in range(MOTION_REPEAT_COUNT)
            for fold_index in range(MOTION_FOLD_COUNT)
        }
        if observed_cells != expected_cells:
            reasons.append("internal_5_cell_evidence_incomplete")

    metric_names = evidence.get("major_metric_names", [])
    if not isinstance(metric_names, list) or not set(MOTION_MAJOR_METRIC_FIELDS).issubset(
        {str(name) for name in metric_names}
    ):
        reasons.append("internal_major_metric_schema_incomplete")
    if evidence.get("model_and_threshold_frozen_before_ptt") is not True:
        reasons.append("model_or_threshold_not_frozen_before_ptt")
    final_model = evidence.get("final_model")
    if (
        not isinstance(final_model, Mapping)
        or final_model.get("scientific_gate_fit_pre_checkpoint_sha256")
        != scientific_checkpoints.get("final_fit_pre")
        or final_model.get("scientific_gate_fit_post_checkpoint_sha256")
        != scientific_checkpoints.get("final_fit_post")
    ):
        reasons.append("final_model_scientific_gate_checkpoint_binding_drift")

    return PttExternalGateDecision(
        allowed=not reasons,
        reasons=tuple(dict.fromkeys(reasons)) if reasons else ("all_internal_gates_passed",),
    )


def motion_contract_payload() -> dict[str, Any]:
    """Return a strict-JSON-ready registry payload without executing science."""

    return {
        "schema_version": MOTION_CONTRACT_SCHEMA,
        "execution_status": "implemented_contract_registered_not_run",
        "scientific_execution_gate": {
            "schema_version": MOTION_SCIENTIFIC_GATE_CONFIG_SCHEMA,
            "operation": "motion_formal_reference",
            "source_tree_policy":
                "tracked_clean_final_pipeline_v2_no_override",
            "source_snapshot_roots": list(SCIENTIFIC_SOURCE_SNAPSHOT_ROOTS),
            "required_dependency_profile_ids":
                list(MOTION_REQUIRED_DEPENDENCY_PROFILES),
            "require_exact_locks": True,
            "internal_checkpoints": [
                "entry_preflight",
                "every_fit_pre",
                "every_fit_post",
            ],
            "ptt_checkpoints": [
                "entry_preflight",
                "predict_pre",
                "predict_post",
            ],
        },
        "default_option": MotionOptionId.SQI_ONLY.value,
        "options": [
            {**asdict(option), "option_id": option.option_id.value}
            for option in MOTION_OPTIONS
        ],
        "supervision": {
            "target": "protocol_activity_motion_state_not_optical_artifact_truth",
            "class_map": {"B": 0, "R": 0, "S": 1, "W": 1},
            "grouping_variable": "participant_id",
        },
        "formal_input": {
            "physical_sources": list(FORMAL_RAW_INPUT_SOURCES),
            "network_tensor_status": FORMAL_NETWORK_TENSOR_STATUS,
            "network_channel_schema": list(MOTION_NETWORK_CHANNEL_SCHEMA),
            "network_channel_units": list(MOTION_NETWORK_CHANNEL_UNITS),
            "network_tensor_schema_sha256": MOTION_NETWORK_SCHEMA_SHA256,
            "fs_hz": 400.0,
            "window_s": MOTION_WINDOW_SECONDS,
            "hop_s": MOTION_HOP_SECONDS,
            "window_samples": MOTION_WINDOW_SAMPLES,
            "hop_samples": MOTION_HOP_SAMPLES,
            "imu_scaling": "all_nine_channels_outer_training_participant_only",
            "imu_per_window_scaling": False,
            "internal_source_units": {
                "acceleration": "g",
                "gyroscope": "deg/s",
                "acceleration_conversion": "g_to_m_per_s2",
                "gyroscope_conversion": "degrees_per_second_to_radians_per_second",
            },
            "ptt_source_units_v2_036": {
                "acceleration": PTT_ADOPTED_ACCELERATION_UNIT,
                "gyroscope": PTT_ADOPTED_GYROSCOPE_UNIT,
                "acceleration_conversion": PTT_ADOPTED_ACCELERATION_CONVERSION,
                "gyroscope_conversion": PTT_ADOPTED_GYROSCOPE_CONVERSION,
                "acceleration_scale_factor": 1.0,
                "acceleration_multiply_9p80665": False,
            },
            "internal_calibration": "same_participant_B_file_only",
            "ptt_calibration": (
                "same_participant_PTT_SIT_STATIC_CALIBRATION_explicit_external_preprocessing"
            ),
            "cross_participant_calibration": "forbidden",
            "ekf_failure_fallback": "forbidden_fail_closed",
            "silent_channel_derivation": "forbidden",
        },
        "internal_protocol": {
            "canonical_entry_id": "formal_internal_motion_reference_source_bound_v2",
            "injected_examples_or_callbacks_allowed": False,
            "dataset_version_id": M2_DATASET_VERSION_ID,
            "participant_count": MOTION_PARTICIPANT_COUNT,
            "registry_id": MOTION_SPLIT_REGISTRY_ID,
            "source_registry_id": MOTION_SOURCE_SPLIT_REGISTRY_ID,
            "registry_csv_sha256": MOTION_SPLIT_CSV_SHA256,
            "upstream_registry_file_sha256": M2_SPLIT_FILE_SHA256,
            "upstream_registry_payload_sha256": M2_SPLIT_PAYLOAD_SHA256,
            "repeat_seeds": [MOTION_SPLIT_SEED],
            "n_repeats": MOTION_REPEAT_COUNT,
            "n_folds": MOTION_FOLD_COUNT,
            "training_seed": MOTION_TRAINING_SEED,
            "runtime_split_recomputation_allowed": False,
            "formal_training": dict(FORMAL_MOTION_TRAINING_CONFIG),
        },
        "threshold": {
            "rule_id": MOTION_THRESHOLD_RULE_ID,
            "score_origin": MOTION_THRESHOLD_SCORE_ORIGIN,
            "fit_scope": MOTION_THRESHOLD_FIT_SCOPE,
            "center_statistic": "median_of_participant_class_medians",
            "outer_oof_or_ptt_scores_allowed": False,
        },
        "external_ptt_gate": {
            "status": (
                "requires_complete_internal_formal_evidence_and_exact_"
                "v2_036_unit_artifact"
            ),
            "canonical_entry_id": "formal_ptt_motion_reference_source_bound_v2",
            "injected_examples_or_callbacks_allowed": False,
            "imu_unit_gate": (
                "exact_v2_036_path_sha_schema_and_66_source_hashes_"
                "before_conversion"
            ),
            "unit_evidence_schema": "ppg_frailty.ptt_imu_unit_evidence.v3",
            "unit_evidence_relative_path": (
                PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH.as_posix()
            ),
            "unit_evidence_sha256": PTT_IMU_UNIT_EVIDENCE_SHA256,
            "acceleration_unit": PTT_ADOPTED_ACCELERATION_UNIT,
            "acceleration_conversion": PTT_ADOPTED_ACCELERATION_CONVERSION,
            "gyroscope_unit": PTT_ADOPTED_GYROSCOPE_UNIT,
            "gyroscope_conversion": PTT_ADOPTED_GYROSCOPE_CONVERSION,
            "registry_id": PTT_FORMAL_REGISTRY_ID,
            "registry_csv_sha256": PTT_SPLIT_CSV_SHA256,
            "repeat_seeds": list(PTT_FORMAL_REPEAT_SEEDS),
            "fold_sizes": list(PTT_FORMAL_FOLD_SIZES),
            "action": "evaluation_only_never_fit_or_recalibrate",
            "independence_claim": "none_not_an_independent_external_test",
        },
        "major_metrics": list(MOTION_MAJOR_METRIC_FIELDS),
        "historical_light_cnn": asdict(HISTORICAL_LIGHT_CNN_EVIDENCE),
    }


__all__ = [
    "FORMAL_NETWORK_TENSOR_STATUS",
    "FORMAL_MOTION_TRAINING_CONFIG",
    "FORMAL_RAW_INPUT_SOURCES",
    "HISTORICAL_LIGHT_CNN_EVIDENCE",
    "MOTION_COMPLETED_CELL_COUNT",
    "MOTION_CONTRACT_SCHEMA",
    "MOTION_INFERENCE_COST_FIELDS",
    "MOTION_INTERNAL_EVIDENCE_SCHEMA",
    "MOTION_MAJOR_METRIC_FIELDS",
    "MOTION_MIDPOINT_THRESHOLD_SCHEMA",
    "MOTION_OOF_PARTICIPANT_REPEAT_ROWS",
    "MOTION_OPTIONS",
    "MOTION_SPLIT_CSV_SHA256",
    "MOTION_SPLIT_REGISTRY_ID",
    "MOTION_SOURCE_SPLIT_REGISTRY_ID",
    "MOTION_SPLIT_SEED",
    "MOTION_TRAINING_SEED",
    "MOTION_THRESHOLD_FIT_SCOPE",
    "MOTION_THRESHOLD_RULE_ID",
    "MOTION_THRESHOLD_SCORE_ORIGIN",
    "PTT_SPLIT_CSV_SHA256",
    "HistoricalLightCnnEvidence",
    "MidpointThresholdArtifact",
    "MotionFoldJob",
    "MotionOptionDescriptor",
    "MotionOptionId",
    "PttExternalGateDecision",
    "fit_train_only_midpoint_threshold",
    "load_motion_fold_jobs",
    "motion_activity_label",
    "motion_contract_payload",
    "resolve_motion_option",
    "validate_motion_major_metrics",
]
