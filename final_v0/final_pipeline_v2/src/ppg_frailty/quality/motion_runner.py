"""Injected smoke/internal motion runners and private canonical execution cores.

Nothing in this module runs at import time. Public injected runners can never
emit formal evidence or reach PTT. Formal internal/PTT execution is exposed only
by the no-callback canonical entry points in quality.motion_reference, which
load hash-bound source files and call the private cores with fixed adapters.

The small ``smoke`` mode exists solely for interface/bug tests.  Its archives
are permanently ineligible for the PTT gate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ..representations.motion import (
    MOTION_NETWORK_SCHEMA_SHA256,
    motion_network_schema_payload,
)
from ..data.external_folds import (
    PTT_FORMAL_FOLD_SIZES,
    PTT_FORMAL_REGISTRY_ID,
    PTT_FORMAL_REPEAT_SEEDS,
    load_formal_ptt_repeated_folds,
)
from ..data.external_manifest import M2_EXTERNAL_MANIFEST_SHA256
from ..data.folds import M2_SPLIT_FILE_SHA256, M2_SPLIT_PAYLOAD_SHA256
from ..motion_ids import FORMAL_MOTION_MODEL_ID
from .motion import (
    MOTION_FOLD_COUNT,
    MOTION_INTERNAL_EVIDENCE_SCHEMA,
    MOTION_MAJOR_METRIC_FIELDS,
    MOTION_MIDPOINT_THRESHOLD_SCHEMA,
    MOTION_OOF_PARTICIPANT_REPEAT_ROWS,
    MOTION_PARTICIPANT_COUNT,
    MOTION_REPEAT_COUNT,
    MOTION_SPLIT_CSV_SHA256,
    MOTION_SPLIT_REGISTRY_ID,
    MOTION_SOURCE_SPLIT_REGISTRY_ID,
    MOTION_SPLIT_SEED,
    MOTION_TRAINING_SEED,
    MOTION_THRESHOLD_FIT_SCOPE,
    MOTION_THRESHOLD_RULE_ID,
    MOTION_THRESHOLD_SCORE_ORIGIN,
    PTT_SPLIT_CSV_SHA256,
    MotionFoldJob,
    PttExternalGateDecision,
    _evaluate_ptt_external_gate_payload,
    fit_train_only_midpoint_threshold,
    load_motion_fold_jobs,
    motion_activity_label,
    validate_motion_major_metrics,
)


MOTION_WINDOW_OOF_SCHEMA = (
    "ppg_frailty.motion_window_oof.imu_iqr_over_1p349.v3"
)
MOTION_EXTERNAL_REPORT_SCHEMA = (
    "ppg_frailty.motion_ptt_external_report.imu_iqr_over_1p349.v3"
)
MOTION_INPUT_SCHEMA_STATUS = "frozen_before_training"
_SHA256_LENGTH = 64
_FORMAL_CANONICAL_ENTRY_TOKEN = object()


@dataclass(frozen=True)
class _FormalMotionRunAuthorization:
    """Private canonical-entry marker; it performs no Git/environment checks."""

    entry_id: str

    def __post_init__(self) -> None:
        if self.entry_id not in {
            "formal_internal_motion_reference_source_bound_v2",
            "formal_ptt_motion_reference_source_bound_v2",
        }:
            raise ValueError("unknown canonical motion entry")


class FormalMotionEntryRequiredError(RuntimeError):
    """Raised when injected arrays/callbacks attempt to enter a formal route."""


@dataclass(frozen=True)
class MotionWindowExample:
    """One pre-materialized supervised window.

    ``values`` is deliberately model-agnostic.  The ordered network tensor
    schema is stored in a separate hash-bound file, not inferred from its shape.
    """

    window_id: str
    participant_id: str
    file_id: str
    role_or_activity: str
    activity_label: int
    values: Any
    dataset_id: str

    def validate_common(self) -> None:
        if not self.window_id or not self.participant_id or not self.file_id:
            raise ValueError("motion window identity fields must be non-empty")
        if self.activity_label not in {0, 1}:
            raise ValueError("motion activity_label must be static=0 or motion=1")
        if not self.dataset_id:
            raise ValueError("motion dataset_id must be explicit")

    def validate_internal(self) -> None:
        self.validate_common()
        if motion_activity_label(self.role_or_activity) != self.activity_label:
            raise ValueError("internal motion label disagrees with canonical B/R/S/W role")

    def validate_ptt(self) -> None:
        self.validate_common()
        activity = self.role_or_activity.strip().lower()
        expected = 0 if activity == "sit" else 1 if activity in {"walk", "run"} else None
        if expected is None or expected != self.activity_label:
            raise ValueError("PTT activity must map sit=0 and walk/run=1")


@dataclass(frozen=True)
class MotionPredictionInput:
    """Label- and identity-free view passed to every prediction callback.

    The runner preserves row association by positional ordering.  Source IDs,
    file/role metadata, and labels are deliberately withheld so an injected
    predictor cannot use protocol metadata as a shortcut.
    """

    values: Any


@dataclass(frozen=True)
class MotionFitContext:
    """Immutable context passed to the injected fit function."""

    execution_mode: str
    repeat_index: int
    fold_index: int
    split_seed: int
    training_seed: int
    final_fit: bool
    training_participant_ids: tuple[str, ...]
    held_out_participant_ids: tuple[str, ...]
    model_input_schema_sha256: str
    artifact_directory: Path


@dataclass(frozen=True)
class MotionFittedArtifact:
    """Return type required from an injected model fit function."""

    runtime_model: Any
    model_id: str
    artifact_path: str
    artifact_sha256: str
    model_input_schema_sha256: str
    training_participant_ids: tuple[str, ...]
    parameter_count: int
    inference_cost: Mapping[str, Any]


@dataclass(frozen=True)
class MotionInternalRunResult:
    evidence: Mapping[str, Any]
    window_oof_rows: tuple[Mapping[str, Any], ...]
    evidence_path: str | None
    evidence_sha256: str | None
    window_oof_path: str | None
    window_oof_sha256: str | None


@dataclass(frozen=True)
class MotionExternalRunResult:
    report: Mapping[str, Any]
    prediction_rows: tuple[Mapping[str, Any], ...]
    report_path: str
    report_sha256: str
    prediction_path: str
    prediction_sha256: str


FitModel = Callable[[Sequence[MotionWindowExample], MotionFitContext], MotionFittedArtifact]
PredictProbability = Callable[[Any, Sequence[MotionPredictionInput]], Sequence[float]]
LoadFrozenModel = Callable[[Path, Mapping[str, Any]], Any]


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == _SHA256_LENGTH and all(character in "0123456789abcdef" for character in text)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json_bytes(payload: object, *, pretty: bool = True) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            allow_nan=False,
        )
        + ("\n" if pretty else "")
    ).encode("utf-8")


def _payload_sha256(payload: object) -> str:
    return hashlib.sha256(_strict_json_bytes(payload, pretty=False)).hexdigest()


def _write_strict_json(path: Path, payload: object) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing motion artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_strict_json_bytes(payload))
    temporary.replace(path)
    return _sha256_file(path)


def _validate_formal_motion_schema_file(path: Path) -> str:
    """Require exact semantic content, not merely caller-supplied file bytes."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("formal motion input schema is invalid JSON") from exc
    expected = {
        **motion_network_schema_payload(),
        "semantic_sha256": MOTION_NETWORK_SCHEMA_SHA256,
    }
    if payload != expected:
        raise ValueError("formal motion input schema semantic content drift")
    return MOTION_NETWORK_SCHEMA_SHA256


def _motion_arrow_schema(schema_version: str, pa: Any) -> Any:
    if schema_version == MOTION_WINDOW_OOF_SCHEMA:
        return pa.schema(
            [
                ("schema_version", pa.string(), False),
                ("repeat_index", pa.int16(), False),
                ("fold_index", pa.int16(), False),
                ("split_seed", pa.int64(), False),
                ("training_seed", pa.int64(), False),
                ("window_id", pa.string(), False),
                ("participant_id", pa.string(), False),
                ("file_id", pa.string(), False),
                ("role_family", pa.string(), False),
                ("activity_label", pa.int8(), False),
                ("p_active", pa.float64(), False),
                ("threshold", pa.float64(), False),
                ("predicted_activity", pa.int8(), False),
                ("score_origin", pa.string(), False),
                ("threshold_score_origin", pa.string(), False),
                ("model_artifact_sha256", pa.string(), False),
            ]
        )
    if schema_version == MOTION_EXTERNAL_REPORT_SCHEMA:
        return pa.schema(
            [
                ("schema_version", pa.string(), False),
                ("dataset_id", pa.string(), False),
                ("window_id", pa.string(), False),
                ("participant_id", pa.string(), False),
                ("file_id", pa.string(), False),
                ("activity", pa.string(), False),
                ("activity_label", pa.int8(), False),
                ("p_active", pa.float64(), False),
                ("threshold", pa.float64(), False),
                ("predicted_activity", pa.int8(), False),
                ("model_artifact_sha256", pa.string(), False),
                ("threshold_artifact_sha256", pa.string(), False),
                ("action", pa.string(), False),
            ]
        )
    raise ValueError("unregistered motion Parquet schema_version")


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    """Write typed Parquet, read it back, and reject semantic byte substitutes."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - formal optional-profile gate.
        raise RuntimeError(
            "formal motion OOF/external output requires the pyarrow benchmark profile"
        ) from exc
    if not rows:
        raise ValueError("motion Parquet output requires at least one row")
    versions = {str(row.get("schema_version", "")) for row in rows}
    if len(versions) != 1:
        raise ValueError("motion Parquet rows mix schema versions")
    schema = _motion_arrow_schema(versions.pop(), pa)
    expected_names = set(schema.names)
    if any(set(row) != expected_names for row in rows):
        raise ValueError("motion Parquet row keys differ from the typed schema")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing motion artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical_rows = [dict(row) for row in rows]
    table = pa.Table.from_pylist(canonical_rows, schema=schema)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary)
    observed = pq.read_table(temporary)
    if not observed.schema.equals(schema, check_metadata=False):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("motion Parquet readback schema drift")
    if observed.to_pylist() != canonical_rows:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("motion Parquet readback value drift")
    temporary.replace(path)
    return _sha256_file(path)


def _read_motion_parquet(path: Path, schema_version: str) -> list[dict[str, Any]]:
    """Read one typed motion table and reject envelope-only mock files."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("motion Parquet validation requires pyarrow") from exc
    expected = _motion_arrow_schema(schema_version, pa)
    table = pq.read_table(path)
    if not table.schema.equals(expected, check_metadata=False):
        raise ValueError("motion Parquet typed schema drift")
    rows = table.to_pylist()
    if not rows:
        raise ValueError("motion Parquet contains no rows")
    return rows


def _probabilities(
    predictor: PredictProbability,
    model: Any,
    rows: Sequence[MotionWindowExample],
) -> np.ndarray:
    prediction_inputs = tuple(MotionPredictionInput(values=row.values) for row in rows)
    values = np.asarray(predictor(model, prediction_inputs), dtype=np.float64)
    if values.ndim != 1 or values.size != len(rows):
        raise ValueError("motion predictor must return one probability per input window")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("motion predictor returned non-finite or out-of-range probabilities")
    return values


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    if labels.size == 0 or set(labels.tolist()) != {0, 1}:
        raise ValueError("motion metrics require both static and motion classes")
    predicted = (probabilities >= threshold).astype(np.int64)
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
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(float(recall))
        f1s.append(float(f1))
    return {
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
    }


def _ece(labels: np.ndarray, probabilities: np.ndarray, *, bins: int = 10) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = float(labels.size)
    value = 0.0
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if np.any(mask):
            value += float(np.sum(mask)) / total * abs(
                float(np.mean(probabilities[mask])) - float(np.mean(labels[mask]))
            )
    return float(value)


def _participant_macro_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    participant_metrics: list[dict[str, float]] = []
    ece_values: list[float] = []
    participant_ids = sorted({str(row["participant_id"]) for row in rows})
    for participant_id in participant_ids:
        selected = [row for row in rows if str(row["participant_id"]) == participant_id]
        labels = np.asarray([int(row["activity_label"]) for row in selected], dtype=np.int64)
        probabilities = np.asarray([float(row["p_active"]) for row in selected], dtype=np.float64)
        thresholds = {float(row["threshold"]) for row in selected}
        if len(thresholds) != 1:
            raise ValueError("participant metric rows mix motion thresholds")
        participant_metrics.append(_binary_metrics(labels, probabilities, thresholds.pop()))
        ece_values.append(_ece(labels, probabilities))
    if not participant_metrics:
        raise ValueError("participant metrics require OOF rows")
    return {
        "balanced_accuracy": float(
            np.mean([item["balanced_accuracy"] for item in participant_metrics])
        ),
        "macro_f1": float(np.mean([item["macro_f1"] for item in participant_metrics])),
        "ece": float(np.mean(ece_values)),
    }


def _validate_inference_cost(inference_cost: Mapping[str, Any], parameter_count: int) -> None:
    validate_motion_major_metrics(
        {
            "participant_macro_balanced_accuracy": 0.5,
            "worst_fold_balanced_accuracy": 0.5,
            "participant_macro_f1": 0.5,
            "ece": 0.5,
            "parameter_count": parameter_count,
            "inference_cost": dict(inference_cost),
        }
    )


def _validate_fitted_artifact(
    fitted: MotionFittedArtifact,
    *,
    context: MotionFitContext,
    output_root: Path,
) -> Path:
    if fitted.model_id != FORMAL_MOTION_MODEL_ID:
        raise ValueError("motion fit callback returned an unregistered formal model ID")
    if fitted.model_input_schema_sha256 != context.model_input_schema_sha256:
        raise ValueError("motion fitted artifact input schema hash drift")
    if tuple(sorted(fitted.training_participant_ids)) != tuple(
        sorted(context.training_participant_ids)
    ):
        raise ValueError("motion fitted artifact training roster drift")
    if not _is_sha256(fitted.artifact_sha256):
        raise ValueError("motion fitted artifact SHA-256 is invalid")
    artifact_path = Path(fitted.artifact_path).resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"motion fitted artifact missing: {artifact_path}")
    if context.execution_mode == "formal":
        try:
            artifact_path.relative_to(output_root.resolve())
        except ValueError as exc:
            raise ValueError("formal motion model artifact must stay inside output_dir") from exc
    if _sha256_file(artifact_path) != fitted.artifact_sha256:
        raise ValueError("motion fitted artifact file SHA-256 mismatch")
    _validate_inference_cost(fitted.inference_cost, fitted.parameter_count)
    return artifact_path


def _validate_examples(examples: Sequence[MotionWindowExample], *, internal: bool) -> None:
    if not examples:
        raise ValueError("motion runner requires materialized windows")
    ids = [item.window_id for item in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("motion window_id values must be unique")
    for item in examples:
        item.validate_internal() if internal else item.validate_ptt()


def _validate_jobs(
    jobs: Sequence[MotionFoldJob],
    participant_ids: set[str],
    *,
    execution_mode: str,
) -> None:
    if execution_mode not in {"formal", "internal", "smoke"}:
        raise ValueError("motion execution_mode must be formal, internal, or smoke")
    if not jobs:
        raise ValueError("motion runner requires at least one frozen fold job")
    cells = [(job.repeat_index, job.fold_index) for job in jobs]
    if len(set(cells)) != len(cells):
        raise ValueError("motion fold jobs contain duplicate cells")
    for job in jobs:
        if job.repeat_index < 0 or job.repeat_index >= MOTION_REPEAT_COUNT:
            raise ValueError("motion job repeat_index is outside the frozen protocol")
        if job.fold_index < 0 or job.fold_index >= MOTION_FOLD_COUNT:
            raise ValueError("motion job fold_index is outside the frozen protocol")
        if job.split_seed != MOTION_SPLIT_SEED:
            raise ValueError("motion job split seed drift")
        if job.training_seed != MOTION_TRAINING_SEED:
            raise ValueError("motion job training seed drift")
        if job.registry_id != MOTION_SPLIT_REGISTRY_ID:
            raise ValueError("motion job registry ID drift")
        if job.registry_csv_sha256 != MOTION_SPLIT_CSV_SHA256:
            raise ValueError("motion job registry hash drift")
        if job.runtime_split_recomputation_allowed:
            raise ValueError("runtime motion split recomputation is forbidden")
        train = set(job.train_participant_ids)
        oof = set(job.oof_participant_ids)
        if train & oof or train | oof != participant_ids:
            raise ValueError("motion job is not a disjoint complete participant partition")
    if execution_mode == "formal":
        expected_cells = {
            (repeat_index, fold_index)
            for repeat_index in range(MOTION_REPEAT_COUNT)
            for fold_index in range(MOTION_FOLD_COUNT)
        }
        if participant_ids and len(participant_ids) != MOTION_PARTICIPANT_COUNT:
            raise ValueError("formal motion run requires exactly 29 participants")
        if set(cells) != expected_cells:
            raise ValueError("formal motion run requires the five seed-42 SGKF cells")
        for repeat_index in range(MOTION_REPEAT_COUNT):
            repeat_oof = [
                participant
                for job in jobs
                if job.repeat_index == repeat_index
                for participant in job.oof_participant_ids
            ]
            if len(repeat_oof) != len(participant_ids) or set(repeat_oof) != participant_ids:
                raise ValueError("formal motion repeat is not an exact OOF participant partition")


def _validate_internal_oof_rows(
    rows: Sequence[Mapping[str, Any]],
    jobs: Sequence[MotionFoldJob],
) -> None:
    """Validate row-level split, label, score, threshold, and hash semantics."""

    if not rows:
        raise ValueError("motion OOF table contains no rows")
    job_by_cell = {(job.repeat_index, job.fold_index): job for job in jobs}
    seen: set[tuple[int, str]] = set()
    participants_by_cell: dict[tuple[int, int], set[str]] = {
        cell: set() for cell in job_by_cell
    }
    for row in rows:
        try:
            repeat = int(row["repeat_index"])
            fold = int(row["fold_index"])
            label = int(row["activity_label"])
            probability = float(row["p_active"])
            threshold = float(row["threshold"])
            predicted = int(row["predicted_activity"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("motion OOF row has invalid typed values") from exc
        cell = (repeat, fold)
        job = job_by_cell.get(cell)
        if job is None:
            raise ValueError("motion OOF row is not bound to a frozen cell")
        window_key = (repeat, str(row["window_id"]))
        if window_key in seen:
            raise ValueError("motion OOF table contains duplicate repeat/window rows")
        seen.add(window_key)
        participant = str(row["participant_id"])
        participants_by_cell[cell].add(participant)
        if participant not in job.oof_participant_ids:
            raise ValueError("motion OOF row participant is not held out in its cell")
        if (
            row.get("schema_version") != MOTION_WINDOW_OOF_SCHEMA
            or int(row["split_seed"]) != job.split_seed
            or int(row["training_seed"]) != job.training_seed
            or row.get("score_origin") != "strict_outer_oof_model_prediction"
            or row.get("threshold_score_origin") != MOTION_THRESHOLD_SCORE_ORIGIN
        ):
            raise ValueError("motion OOF row protocol provenance drift")
        if label != motion_activity_label(str(row["role_family"])):
            raise ValueError("motion OOF row label disagrees with role family")
        if (
            not np.isfinite(probability)
            or not 0.0 <= probability <= 1.0
            or not np.isfinite(threshold)
            or not 0.0 < threshold < 1.0
            or predicted != int(probability >= threshold)
            or not _is_sha256(row.get("model_artifact_sha256"))
        ):
            raise ValueError("motion OOF row score/prediction/hash semantics drift")
    for cell, job in job_by_cell.items():
        if participants_by_cell[cell] != set(job.oof_participant_ids):
            raise ValueError("motion OOF cell participant coverage is incomplete")


def _validate_internal_evidence_semantics(
    rows: Sequence[Mapping[str, Any]],
    jobs: Sequence[MotionFoldJob],
    evidence: Mapping[str, Any],
) -> tuple[str, ...]:
    """Recompute row-linked metrics and compare every cell artifact binding."""

    try:
        _validate_internal_oof_rows(rows, jobs)
    except ValueError:
        return ("frozen_window_oof_semantic_validation_failed",)
    reasons: list[str] = []
    if int(evidence.get("window_oof_row_count", -1)) != len(rows):
        reasons.append("window_oof_row_count_drift")
    cell_evidence_value = evidence.get("cell_evidence")
    cell_metrics_value = evidence.get("cell_metrics")
    if not isinstance(cell_evidence_value, list) or not isinstance(cell_metrics_value, list):
        return ("cell_evidence_or_metrics_missing",)
    evidence_by_cell = {
        (int(item["repeat_index"]), int(item["fold_index"])): item
        for item in cell_evidence_value
        if isinstance(item, Mapping)
        and "repeat_index" in item
        and "fold_index" in item
    }
    metrics_by_cell = {
        (int(item["repeat_index"]), int(item["fold_index"])): item
        for item in cell_metrics_value
        if isinstance(item, Mapping)
        and "repeat_index" in item
        and "fold_index" in item
    }
    expected_cells = {(job.repeat_index, job.fold_index) for job in jobs}
    if set(evidence_by_cell) != expected_cells or len(cell_evidence_value) != len(expected_cells):
        reasons.append("cell_evidence_identity_drift")
    if set(metrics_by_cell) != expected_cells or len(cell_metrics_value) != len(expected_cells):
        reasons.append("cell_metrics_identity_drift")
    recomputed_cells: list[dict[str, float]] = []
    for cell in sorted(expected_cells):
        selected = [
            row
            for row in rows
            if (int(row["repeat_index"]), int(row["fold_index"])) == cell
        ]
        if not selected:
            reasons.append("cell_oof_rows_missing")
            continue
        evidence_row = evidence_by_cell.get(cell, {})
        thresholds = {float(row["threshold"]) for row in selected}
        hashes = {str(row["model_artifact_sha256"]) for row in selected}
        threshold_payload = evidence_row.get("threshold")
        if (
            len(thresholds) != 1
            or not isinstance(threshold_payload, Mapping)
            or thresholds.pop() != float(threshold_payload.get("threshold", float("nan")))
        ):
            reasons.append("cell_threshold_not_bound_to_oof_rows")
        if hashes != {str(evidence_row.get("model_artifact_sha256", ""))}:
            reasons.append("cell_model_not_bound_to_oof_rows")
        if int(evidence_row.get("oof_window_count", -1)) != len(selected):
            reasons.append("cell_oof_window_count_drift")
        recomputed = _participant_macro_metrics(selected)
        recomputed_cells.append(recomputed)
        declared = metrics_by_cell.get(cell, {})
        for name in ("balanced_accuracy", "macro_f1", "ece"):
            if not np.isclose(
                recomputed[name],
                float(declared.get(name, float("nan"))),
                rtol=0.0,
                atol=1e-12,
            ):
                reasons.append(f"cell_{name}_metric_drift")
    overall = _participant_macro_metrics(rows)
    declared_major = evidence.get("major_metrics")
    if not isinstance(declared_major, Mapping):
        reasons.append("major_metrics_missing")
    else:
        expected_major = {
            "participant_macro_balanced_accuracy": overall["balanced_accuracy"],
            "participant_macro_f1": overall["macro_f1"],
            "ece": overall["ece"],
            "worst_fold_balanced_accuracy": min(
                item["balanced_accuracy"] for item in recomputed_cells
            ),
        }
        for name, expected in expected_major.items():
            if not np.isclose(
                expected,
                float(declared_major.get(name, float("nan"))),
                rtol=0.0,
                atol=1e-12,
            ):
                reasons.append(f"major_{name}_drift")
    return tuple(dict.fromkeys(reasons))


def _validate_external_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_participant_ids: set[str],
) -> None:
    if not rows or len({str(row["window_id"]) for row in rows}) != len(rows):
        raise ValueError("PTT motion prediction rows are empty or duplicate")
    observed = {str(row["participant_id"]) for row in rows}
    if observed != expected_participant_ids:
        raise ValueError("PTT motion prediction participant coverage drift")
    for row in rows:
        probability = float(row["p_active"])
        threshold = float(row["threshold"])
        activity = str(row["activity"]).strip().lower()
        expected_label = 0 if activity == "sit" else 1 if activity in {"walk", "run"} else None
        if (
            row.get("schema_version") != MOTION_EXTERNAL_REPORT_SCHEMA
            or expected_label is None
            or int(row["activity_label"]) != expected_label
            or not np.isfinite(probability)
            or not 0.0 <= probability <= 1.0
            or not np.isfinite(threshold)
            or int(row["predicted_activity"]) != int(probability >= threshold)
            or not _is_sha256(row.get("model_artifact_sha256"))
            or not _is_sha256(row.get("threshold_artifact_sha256"))
            or row.get("action") != "evaluation_only_never_fit_or_recalibrate"
        ):
            raise ValueError("PTT motion prediction row semantic drift")


def _threshold_and_hash(
    fitted: MotionFittedArtifact,
    train_rows: Sequence[MotionWindowExample],
    oof_participant_ids: Iterable[str],
    predictor: PredictProbability,
) -> tuple[Mapping[str, Any], str]:
    train_probabilities = _probabilities(predictor, fitted.runtime_model, train_rows)
    threshold = fit_train_only_midpoint_threshold(
        train_probabilities,
        [row.activity_label for row in train_rows],
        [row.participant_id for row in train_rows],
        training_participant_ids=fitted.training_participant_ids,
        forbidden_oof_participant_ids=oof_participant_ids,
        forbidden_ptt_participant_ids=(),
        score_origin=MOTION_THRESHOLD_SCORE_ORIGIN,
    ).as_dict()
    return threshold, _payload_sha256(threshold)


def _run_internal_motion_oof_impl(
    examples: Sequence[MotionWindowExample],
    fold_jobs: Sequence[MotionFoldJob],
    *,
    fit_model: FitModel,
    predict_probability: PredictProbability,
    model_input_schema_path: str | Path,
    expected_model_input_schema_sha256: str,
    output_dir: str | Path,
    motion_split_csv_path: str | Path | None = None,
    execution_mode: str,
    write_artifacts: bool = True,
    formal_source_evidence: Mapping[str, Any] | None = None,
    formal_run_authorization: _FormalMotionRunAuthorization | None = None,
    _canonical_entry_token: object | None = None,
) -> MotionInternalRunResult:
    """Train/evaluate the formal five-cell protocol without hidden splitting.

    Formal mode always fits thresholds from current outer-training scores only,
    emits held-out window probabilities, writes real Parquet, and fits a final
    all-29-participant model only after OOF completion.  No metric-based model or
    threshold selection occurs in this runner.
    """

    if execution_mode == "formal":
        if _canonical_entry_token is not _FORMAL_CANONICAL_ENTRY_TOKEN:
            raise FormalMotionEntryRequiredError(
                "formal internal motion is available only through the canonical source loader"
            )
        if not isinstance(formal_source_evidence, Mapping):
            raise ValueError("formal internal motion source evidence is required")
        if (
            not isinstance(
                formal_run_authorization,
                _FormalMotionRunAuthorization,
            )
            or formal_run_authorization.entry_id
            != "formal_internal_motion_reference_source_bound_v2"
        ):
            raise ValueError(
                "formal internal motion canonical authorization is required"
            )
        from .motion_reference import verify_formal_internal_source_evidence

        source_reasons = verify_formal_internal_source_evidence(formal_source_evidence)
        if source_reasons:
            raise ValueError(
                "formal internal motion source evidence rejected: "
                + ";".join(source_reasons)
            )
    elif (
        formal_source_evidence is not None
        or formal_run_authorization is not None
        or _canonical_entry_token is not None
    ):
        raise ValueError("non-formal injected runner may not carry formal source authority")

    records = tuple(examples)
    jobs = tuple(fold_jobs)
    _validate_examples(records, internal=True)
    participant_ids = {item.participant_id for item in records}
    _validate_jobs(jobs, participant_ids, execution_mode=execution_mode)
    verified_split_path: Path | None = None
    if execution_mode == "formal":
        if motion_split_csv_path is None:
            raise ValueError("formal motion run requires the real hash-bound split CSV path")
        verified_split_path = Path(motion_split_csv_path).resolve()
        authoritative_jobs = load_motion_fold_jobs(verified_split_path)
        ordered_jobs = tuple(sorted(jobs, key=lambda item: (item.repeat_index, item.fold_index)))
        if ordered_jobs != authoritative_jobs:
            raise ValueError(
                "caller motion jobs differ from the real hash-bound corrected SGKF assignments"
            )
    if execution_mode == "formal" and not write_artifacts:
        raise ValueError("formal motion run may not suppress frozen artifacts")

    schema_path = Path(model_input_schema_path).resolve()
    if not schema_path.is_file():
        raise FileNotFoundError(f"motion input schema file not found: {schema_path}")
    if not _is_sha256(expected_model_input_schema_sha256):
        raise ValueError("expected motion input schema SHA-256 is invalid")
    if _sha256_file(schema_path) != expected_model_input_schema_sha256:
        raise ValueError("motion input schema file SHA-256 mismatch")
    semantic_schema_sha256 = (
        _validate_formal_motion_schema_file(schema_path)
        if execution_mode == "formal"
        else expected_model_input_schema_sha256
    )

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    window_oof_rows: list[dict[str, Any]] = []
    cell_evidence: list[dict[str, Any]] = []
    cell_metrics: list[dict[str, Any]] = []
    for job in sorted(jobs, key=lambda item: (item.repeat_index, item.fold_index)):
        train_ids = set(job.train_participant_ids)
        oof_ids = set(job.oof_participant_ids)
        train_rows = tuple(row for row in records if row.participant_id in train_ids)
        oof_rows = tuple(row for row in records if row.participant_id in oof_ids)
        if not train_rows or not oof_rows:
            raise ValueError("motion fold contains an empty train or OOF window set")
        cell_dir = root / f"repeat_{job.repeat_index}" / f"fold_{job.fold_index}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        context = MotionFitContext(
            execution_mode=execution_mode,
            repeat_index=job.repeat_index,
            fold_index=job.fold_index,
            split_seed=job.split_seed,
            training_seed=job.training_seed,
            final_fit=False,
            training_participant_ids=tuple(sorted(train_ids)),
            held_out_participant_ids=tuple(sorted(oof_ids)),
            model_input_schema_sha256=semantic_schema_sha256,
            artifact_directory=cell_dir,
        )
        fitted = fit_model(train_rows, context)
        model_path = _validate_fitted_artifact(fitted, context=context, output_root=root)
        threshold, threshold_sha256 = _threshold_and_hash(
            fitted,
            train_rows,
            oof_ids,
            predict_probability,
        )
        probabilities = _probabilities(predict_probability, fitted.runtime_model, oof_rows)
        threshold_value = float(threshold["threshold"])
        current_rows: list[dict[str, Any]] = []
        for example, probability in zip(oof_rows, probabilities, strict=True):
            row = {
                "schema_version": MOTION_WINDOW_OOF_SCHEMA,
                "repeat_index": job.repeat_index,
                "fold_index": job.fold_index,
                "split_seed": job.split_seed,
                "training_seed": job.training_seed,
                "window_id": example.window_id,
                "participant_id": example.participant_id,
                "file_id": example.file_id,
                "role_family": example.role_or_activity,
                "activity_label": example.activity_label,
                "p_active": float(probability),
                "threshold": threshold_value,
                "predicted_activity": int(probability >= threshold_value),
                "score_origin": "strict_outer_oof_model_prediction",
                "threshold_score_origin": MOTION_THRESHOLD_SCORE_ORIGIN,
                "model_artifact_sha256": fitted.artifact_sha256,
            }
            current_rows.append(row)
            window_oof_rows.append(row)
        metrics = _participant_macro_metrics(current_rows)
        cell_metrics.append(
            {
                "repeat_index": job.repeat_index,
                "fold_index": job.fold_index,
                **metrics,
            }
        )
        cell_row: dict[str, Any] = {
            "repeat_index": job.repeat_index,
            "fold_index": job.fold_index,
            "training_participant_count": len(train_ids),
            "oof_participant_count": len(oof_ids),
            "model_artifact_path": str(model_path),
            "model_artifact_sha256": fitted.artifact_sha256,
            "model_input_schema_sha256": semantic_schema_sha256,
            "parameter_count": fitted.parameter_count,
            "inference_cost": dict(fitted.inference_cost),
            "threshold_artifact_sha256": threshold_sha256,
            "threshold_fit_scope": MOTION_THRESHOLD_FIT_SCOPE,
            "threshold": threshold,
            "oof_window_count": len(current_rows),
        }
        cell_evidence.append(cell_row)

    _validate_internal_oof_rows(window_oof_rows, jobs)
    repeat_participant_metrics: list[dict[str, float]] = []
    for repeat_index in sorted({job.repeat_index for job in jobs}):
        selected = [row for row in window_oof_rows if row["repeat_index"] == repeat_index]
        repeat_participant_metrics.append(_participant_macro_metrics(selected))

    # Final model is fit only after all requested OOF cells complete.  In formal
    # mode this is the deployable artifact that the PTT runner may load.
    final_dir = root / "final_all_internal"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_context = MotionFitContext(
        execution_mode=execution_mode,
        repeat_index=-1,
        fold_index=-1,
        split_seed=MOTION_SPLIT_SEED,
        training_seed=MOTION_TRAINING_SEED,
        final_fit=True,
        training_participant_ids=tuple(sorted(participant_ids)),
        held_out_participant_ids=(),
        model_input_schema_sha256=semantic_schema_sha256,
        artifact_directory=final_dir,
    )
    final_fitted = fit_model(records, final_context)
    final_model_path = _validate_fitted_artifact(
        final_fitted,
        context=final_context,
        output_root=root,
    )
    final_threshold, final_threshold_sha256 = _threshold_and_hash(
        final_fitted,
        records,
        (),
        predict_probability,
    )

    major_metrics = {
        "participant_macro_balanced_accuracy": float(
            np.mean([item["balanced_accuracy"] for item in repeat_participant_metrics])
        ),
        "worst_fold_balanced_accuracy": float(
            min(item["balanced_accuracy"] for item in cell_metrics)
        ),
        "participant_macro_f1": float(
            np.mean([item["macro_f1"] for item in repeat_participant_metrics])
        ),
        "ece": float(np.mean([item["ece"] for item in repeat_participant_metrics])),
        "parameter_count": final_fitted.parameter_count,
        "inference_cost": dict(final_fitted.inference_cost),
    }
    validate_motion_major_metrics(major_metrics)

    formal = execution_mode == "formal"
    evidence: dict[str, Any] = {
        "schema_version": MOTION_INTERNAL_EVIDENCE_SCHEMA,
        "execution_status": (
            "completed_formal_not_smoke"
            if formal
            else "completed_internal_injected_not_formal"
            if execution_mode == "internal"
            else "completed_smoke_not_formal"
        ),
        "scientific_scope": (
            "frailty29_single_sgkf5_oof"
            if formal
            else "injected_internal_not_gate_eligible"
            if execution_mode == "internal"
            else "synthetic_or_reduced_smoke"
        ),
        "model_id": FORMAL_MOTION_MODEL_ID,
        "split_registry_id": MOTION_SPLIT_REGISTRY_ID,
        "source_split_registry_id": MOTION_SOURCE_SPLIT_REGISTRY_ID,
        "split_registry_csv_sha256": MOTION_SPLIT_CSV_SHA256,
        "upstream_split_registry_file_sha256": M2_SPLIT_FILE_SHA256,
        "upstream_split_registry_payload_sha256": M2_SPLIT_PAYLOAD_SHA256,
        "split_registry_csv_path": (
            str(verified_split_path) if verified_split_path is not None else "smoke_not_bound"
        ),
        "participant_count": len(participant_ids),
        "oof_participant_repeat_rows": (
            MOTION_OOF_PARTICIPANT_REPEAT_ROWS if formal else len(participant_ids) * len({j.repeat_index for j in jobs})
        ),
        "model_input_schema_status": MOTION_INPUT_SCHEMA_STATUS,
        "model_input_schema_path": str(schema_path),
        "model_input_schema_file_sha256": expected_model_input_schema_sha256,
        "model_input_schema_sha256": semantic_schema_sha256,
        "threshold_rule_id": MOTION_THRESHOLD_RULE_ID,
        "threshold_score_origin": MOTION_THRESHOLD_SCORE_ORIGIN,
        "major_metric_names": list(MOTION_MAJOR_METRIC_FIELDS),
        "major_metrics": major_metrics,
        "cell_metrics": cell_metrics,
        "cell_evidence": cell_evidence,
        "window_oof_row_count": len(window_oof_rows),
        "model_and_threshold_frozen_before_ptt": formal,
        "final_model": {
            "artifact_path": str(final_model_path),
            "artifact_sha256": final_fitted.artifact_sha256,
            "model_input_schema_sha256": semantic_schema_sha256,
            "training_participant_ids": list(final_fitted.training_participant_ids),
            "parameter_count": final_fitted.parameter_count,
            "inference_cost": dict(final_fitted.inference_cost),
        },
        "final_threshold": final_threshold,
        "final_threshold_artifact_sha256": final_threshold_sha256,
        "ablation_executed": False,
    }
    if formal:
        evidence["formal_entry_id"] = str(
            formal_source_evidence["formal_entry_id"]
        )
        evidence["formal_source_evidence"] = dict(formal_source_evidence)

    evidence_path: str | None = None
    evidence_sha256: str | None = None
    oof_path: str | None = None
    oof_sha256: str | None = None
    if write_artifacts:
        if formal:
            output_oof = root / "motion_window_oof.parquet"
            oof_sha256 = _write_parquet(output_oof, window_oof_rows)
            oof_path = str(output_oof)
            evidence["window_oof_parquet_path"] = oof_path
            evidence["window_oof_parquet_sha256"] = oof_sha256
        else:
            output_oof = root / "motion_window_oof_smoke.json"
            oof_sha256 = _write_strict_json(output_oof, window_oof_rows)
            oof_path = str(output_oof)
            evidence["smoke_window_oof_json_path"] = oof_path
            evidence["smoke_window_oof_json_sha256"] = oof_sha256
        output_evidence = root / "motion_internal_evidence.json"
        evidence_sha256 = _write_strict_json(output_evidence, evidence)
        evidence_path = str(output_evidence)

    return MotionInternalRunResult(
        evidence=evidence,
        window_oof_rows=tuple(window_oof_rows),
        evidence_path=evidence_path,
        evidence_sha256=evidence_sha256,
        window_oof_path=oof_path,
        window_oof_sha256=oof_sha256,
    )


def load_motion_internal_evidence(
    evidence_path: str | Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Load a real on-disk archive and require its trusted out-of-band digest."""

    if not _is_sha256(expected_sha256):
        raise ValueError("expected internal evidence SHA-256 is invalid")
    source = Path(evidence_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"internal motion evidence archive not found: {source}")
    observed = _sha256_file(source)
    if observed != expected_sha256:
        raise ValueError("internal motion evidence archive SHA-256 mismatch")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("internal motion evidence archive is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("internal motion evidence archive must contain one JSON object")
    return payload, observed


def _threshold_payload_is_bound_to_participants(
    payload: object,
    expected_participant_ids: Iterable[str],
) -> bool:
    """Validate a serialized midpoint artifact and its exact training roster."""

    if not isinstance(payload, Mapping):
        return False
    expected = tuple(sorted(str(value) for value in expected_participant_ids))
    observed_value = payload.get("participant_ids")
    if not isinstance(observed_value, (list, tuple)):
        return False
    observed = tuple(sorted(str(value) for value in observed_value))
    if len(observed) != len(set(observed)) or observed != expected:
        return False
    roster_bytes = json.dumps(
        sorted(set(observed)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    try:
        static_center = float(payload["static_center"])
        motion_center = float(payload["motion_center"])
        threshold = float(payload["threshold"])
    except (KeyError, TypeError, ValueError):
        return False
    midpoint = (static_center + motion_center) / 2.0
    return bool(
        payload.get("schema_version") == MOTION_MIDPOINT_THRESHOLD_SCHEMA
        and payload.get("threshold_rule_id") == MOTION_THRESHOLD_RULE_ID
        and payload.get("score_origin") == MOTION_THRESHOLD_SCORE_ORIGIN
        and payload.get("fit_scope") == MOTION_THRESHOLD_FIT_SCOPE
        and payload.get("participant_roster_sha256")
        == hashlib.sha256(roster_bytes).hexdigest()
        and tuple(payload.get("class_ids", ())) == (0, 1)
        and np.isfinite(static_center)
        and np.isfinite(motion_center)
        and np.isfinite(threshold)
        and 0.0 <= static_center < threshold < motion_center <= 1.0
        and abs(threshold - midpoint) <= 1e-15
    )


def _frozen_file_matches(
    path_value: object,
    sha256_value: object,
    *,
    required_root: Path | None = None,
    parquet_envelope: bool = False,
) -> bool:
    """Check a real immutable file, optionally constrained to the run archive."""

    if not _is_sha256(sha256_value):
        return False
    candidate = Path(str(path_value)).resolve()
    if not candidate.is_file():
        return False
    if required_root is not None:
        try:
            candidate.relative_to(required_root.resolve())
        except ValueError:
            return False
    if _sha256_file(candidate) != str(sha256_value):
        return False
    if parquet_envelope:
        size = candidate.stat().st_size
        if size < 12:
            return False
        with candidate.open("rb") as handle:
            leading = handle.read(4)
            handle.seek(-4, 2)
            trailing = handle.read(4)
        if leading != b"PAR1" or trailing != b"PAR1":
            return False
    return True


def _verify_frozen_internal_assets(
    evidence: Mapping[str, Any],
    *,
    archive_root: Path,
) -> tuple[str, ...]:
    """Bind the formal archive to real splits, models, thresholds, and OOF data."""

    reasons: list[str] = []
    source_evidence = evidence.get("formal_source_evidence")
    if not isinstance(source_evidence, Mapping):
        reasons.append("formal_source_evidence_missing")
    else:
        try:
            from .motion_reference import verify_formal_internal_source_evidence

            reasons.extend(verify_formal_internal_source_evidence(source_evidence))
        except (ImportError, OSError, TypeError, ValueError):
            reasons.append("formal_source_evidence_semantic_reload_failed")
    split_path = Path(str(evidence.get("split_registry_csv_path", ""))).resolve()
    try:
        authoritative_jobs = load_motion_fold_jobs(split_path)
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        authoritative_jobs = ()
        reasons.append("frozen_internal_split_csv_missing_or_hash_mismatch")
    job_by_cell = {
        (job.repeat_index, job.fold_index): job for job in authoritative_jobs
    }

    schema_matches = _frozen_file_matches(
        evidence.get("model_input_schema_path"),
        evidence.get("model_input_schema_file_sha256"),
    )
    if not schema_matches:
        reasons.append("frozen_model_input_schema_missing_or_hash_mismatch")
    else:
        try:
            semantic_hash = _validate_formal_motion_schema_file(
                Path(str(evidence["model_input_schema_path"])).resolve()
            )
        except (KeyError, OSError, TypeError, ValueError):
            reasons.append("frozen_model_input_schema_semantic_drift")
        else:
            if evidence.get("model_input_schema_sha256") != semantic_hash:
                reasons.append("frozen_model_input_schema_semantic_hash_drift")
    parquet_matches = _frozen_file_matches(
        evidence.get("window_oof_parquet_path"),
        evidence.get("window_oof_parquet_sha256"),
        required_root=archive_root,
        parquet_envelope=True,
    )
    if not parquet_matches:
        reasons.append("frozen_window_oof_parquet_missing_or_hash_mismatch")
    elif authoritative_jobs:
        try:
            oof_rows = _read_motion_parquet(
                Path(str(evidence["window_oof_parquet_path"])).resolve(),
                MOTION_WINDOW_OOF_SCHEMA,
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            reasons.append("frozen_window_oof_parquet_typed_readback_failed")
        else:
            try:
                semantic_reasons = _validate_internal_evidence_semantics(
                    oof_rows, authoritative_jobs, evidence
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                reasons.append("frozen_window_oof_evidence_semantic_validation_failed")
            else:
                reasons.extend(semantic_reasons)

    cell_rows = evidence.get("cell_evidence")
    if isinstance(cell_rows, list):
        for row in cell_rows:
            if not isinstance(row, Mapping):
                continue
            try:
                cell = (int(row["repeat_index"]), int(row["fold_index"]))
            except (KeyError, TypeError, ValueError):
                continue
            job = job_by_cell.get(cell)
            if job is None:
                reasons.append("cell_not_bound_to_authoritative_split")
                continue
            if row.get("training_participant_count") != len(job.train_participant_ids):
                reasons.append("cell_training_participant_count_drift")
            if row.get("oof_participant_count") != len(job.oof_participant_ids):
                reasons.append("cell_oof_participant_count_drift")
            if (
                row.get("model_input_schema_sha256")
                != evidence.get("model_input_schema_sha256")
            ):
                reasons.append("cell_model_input_schema_hash_drift")
            model_matches = _frozen_file_matches(
                row.get("model_artifact_path"),
                row.get("model_artifact_sha256"),
                required_root=archive_root,
            )
            if not model_matches:
                reasons.append("cell_model_artifact_missing_or_hash_mismatch")
            else:
                try:
                    from .motion_adapters import load_formal_motion_model

                    load_formal_motion_model(
                        Path(str(row["model_artifact_path"])).resolve(),
                        {
                            "artifact_sha256": row["model_artifact_sha256"],
                            "training_participant_ids": list(job.train_participant_ids),
                            "parameter_count": row["parameter_count"],
                            "inference_cost": row["inference_cost"],
                            "model_input_schema_sha256": row[
                                "model_input_schema_sha256"
                            ],
                        },
                    )
                except (KeyError, OSError, RuntimeError, TypeError, ValueError):
                    reasons.append("cell_model_artifact_semantic_reload_failed")
            threshold = row.get("threshold")
            if (
                not _threshold_payload_is_bound_to_participants(
                    threshold,
                    job.train_participant_ids,
                )
                or _payload_sha256(threshold) != row.get("threshold_artifact_sha256")
            ):
                reasons.append("cell_threshold_artifact_missing_or_hash_mismatch")

    all_participants = tuple(
        sorted(
            {
                participant_id
                for job in authoritative_jobs
                for participant_id in (
                    *job.train_participant_ids,
                    *job.oof_participant_ids,
                )
            }
        )
    )
    final_model = evidence.get("final_model")
    if not isinstance(final_model, Mapping):
        reasons.append("frozen_final_model_missing")
    else:
        final_model_matches = _frozen_file_matches(
            final_model.get("artifact_path"),
            final_model.get("artifact_sha256"),
            required_root=archive_root,
        )
        if not final_model_matches:
            reasons.append("frozen_final_model_missing_or_hash_mismatch")
        roster_value = final_model.get("training_participant_ids")
        if not isinstance(roster_value, (list, tuple)) or tuple(
            sorted(str(value) for value in roster_value)
        ) != all_participants:
            reasons.append("frozen_final_model_training_roster_drift")
        if (
            final_model.get("model_input_schema_sha256")
            != evidence.get("model_input_schema_sha256")
        ):
            reasons.append("frozen_final_model_input_schema_hash_drift")
        if final_model_matches:
            try:
                from .motion_adapters import load_formal_motion_model

                load_formal_motion_model(
                    Path(str(final_model["artifact_path"])).resolve(),
                    {
                        "artifact_sha256": final_model["artifact_sha256"],
                        "training_participant_ids": list(all_participants),
                        "parameter_count": final_model["parameter_count"],
                        "inference_cost": final_model["inference_cost"],
                        "model_input_schema_sha256": final_model[
                            "model_input_schema_sha256"
                        ],
                    },
                )
            except (KeyError, OSError, RuntimeError, TypeError, ValueError):
                reasons.append("frozen_final_model_semantic_reload_failed")

    final_threshold = evidence.get("final_threshold")
    if (
        not _threshold_payload_is_bound_to_participants(
            final_threshold,
            all_participants,
        )
        or _payload_sha256(final_threshold)
        != evidence.get("final_threshold_artifact_sha256")
    ):
        reasons.append("frozen_final_threshold_missing_or_hash_mismatch")
    return tuple(dict.fromkeys(reasons))


def evaluate_ptt_external_gate(
    evidence_path: str | Path,
    *,
    expected_sha256: str,
) -> PttExternalGateDecision:
    """Open PTT only for a complete hash-bound archive and real frozen assets."""

    evidence, _ = load_motion_internal_evidence(
        evidence_path,
        expected_sha256=expected_sha256,
    )
    decision = _evaluate_ptt_external_gate_payload(evidence)
    reasons = [] if decision.allowed else list(decision.reasons)
    if decision.allowed:
        reasons.extend(
            _verify_frozen_internal_assets(
                evidence,
                archive_root=Path(evidence_path).resolve().parent,
            )
        )
    return PttExternalGateDecision(
        allowed=not reasons and decision.allowed,
        reasons=tuple(dict.fromkeys(reasons)) if reasons else decision.reasons,
    )


def _run_ptt_external_evaluation_impl(
    examples: Sequence[MotionWindowExample],
    *,
    internal_evidence_path: str | Path,
    expected_internal_evidence_sha256: str,
    ptt_split_csv: str | Path,
    load_frozen_model: LoadFrozenModel,
    predict_probability: PredictProbability,
    output_dir: str | Path,
    formal_source_evidence: Mapping[str, Any],
    formal_run_authorization: _FormalMotionRunAuthorization,
    _canonical_entry_token: object | None = None,
) -> MotionExternalRunResult:
    """Evaluate the frozen final model on PTT; fitting/recalibration is impossible.

    There is intentionally no trainer, threshold fitter, or model-selector
    argument.  The PTT labels influence metrics only after predictions are made.
    """

    if _canonical_entry_token is not _FORMAL_CANONICAL_ENTRY_TOKEN:
        raise FormalMotionEntryRequiredError(
            "formal PTT motion is available only through the canonical source loader"
        )
    if (
        not isinstance(formal_run_authorization, _FormalMotionRunAuthorization)
        or formal_run_authorization.entry_id
        != "formal_ptt_motion_reference_source_bound_v2"
    ):
        raise ValueError("formal PTT canonical authorization is required")
    if (
        formal_source_evidence.get("formal_entry_id")
        != "formal_ptt_motion_reference_source_bound_v2"
        or formal_source_evidence.get("source_manifest_sha256")
        != M2_EXTERNAL_MANIFEST_SHA256
        or formal_source_evidence.get("record_count") != 66
        or formal_source_evidence.get("participant_count") != 22
        or formal_source_evidence.get("tensor_schema_sha256")
        != MOTION_NETWORK_SCHEMA_SHA256
    ):
        raise ValueError("formal PTT source evidence identity/roster drift")
    from .motion_reference import verify_formal_ptt_source_evidence

    ptt_source_reasons = verify_formal_ptt_source_evidence(formal_source_evidence)
    if ptt_source_reasons:
        raise ValueError(
            "formal PTT source evidence rejected: " + ";".join(ptt_source_reasons)
        )

    gate = evaluate_ptt_external_gate(
        internal_evidence_path,
        expected_sha256=expected_internal_evidence_sha256,
    )
    if not gate.allowed:
        raise RuntimeError("PTT external motion gate is closed: " + ";".join(gate.reasons))
    evidence, evidence_sha256 = load_motion_internal_evidence(
        internal_evidence_path,
        expected_sha256=expected_internal_evidence_sha256,
    )
    ptt_csv = Path(ptt_split_csv).resolve()
    if _sha256_file(ptt_csv) != PTT_SPLIT_CSV_SHA256:
        raise ValueError("PTT formal split CSV SHA-256 drift")
    split_rows = load_formal_ptt_repeated_folds(ptt_csv)

    records = tuple(examples)
    _validate_examples(records, internal=False)
    expected_participants = {str(row["subject_id"]) for row in split_rows}
    observed_participants = {row.participant_id for row in records}
    if observed_participants != expected_participants:
        raise ValueError("PTT window participant roster differs from the frozen registry")

    final_model = evidence["final_model"]
    model_path = Path(str(final_model["artifact_path"]))
    runtime_model = load_frozen_model(model_path, final_model)
    probabilities = _probabilities(predict_probability, runtime_model, records)
    threshold = float(evidence["final_threshold"]["threshold"])
    prediction_rows = [
        {
            "schema_version": MOTION_EXTERNAL_REPORT_SCHEMA,
            "dataset_id": example.dataset_id,
            "window_id": example.window_id,
            "participant_id": example.participant_id,
            "file_id": example.file_id,
            "activity": example.role_or_activity,
            "activity_label": example.activity_label,
            "p_active": float(probability),
            "threshold": threshold,
            "predicted_activity": int(probability >= threshold),
            "model_artifact_sha256": final_model["artifact_sha256"],
            "threshold_artifact_sha256": evidence["final_threshold_artifact_sha256"],
            "action": "evaluation_only_never_fit_or_recalibrate",
        }
        for example, probability in zip(records, probabilities, strict=True)
    ]
    _validate_external_prediction_rows(
        prediction_rows,
        expected_participant_ids=expected_participants,
    )

    fold_metrics: list[dict[str, Any]] = []
    for repeat_index, seed in enumerate(PTT_FORMAL_REPEAT_SEEDS):
        repeat_rows = [row for row in split_rows if int(row["repeat_index"]) == repeat_index]
        for fold_index in range(len(PTT_FORMAL_FOLD_SIZES)):
            participants = {
                str(row["subject_id"])
                for row in repeat_rows
                if int(row["fold_index"]) == fold_index
            }
            selected = [
                row for row in prediction_rows if str(row["participant_id"]) in participants
            ]
            fold_metrics.append(
                {
                    "repeat_index": repeat_index,
                    "fold_index": fold_index,
                    "split_seed": seed,
                    **_participant_macro_metrics(selected),
                }
            )
    overall = _participant_macro_metrics(prediction_rows)
    major_metrics = {
        "participant_macro_balanced_accuracy": overall["balanced_accuracy"],
        "worst_fold_balanced_accuracy": min(
            item["balanced_accuracy"] for item in fold_metrics
        ),
        "participant_macro_f1": overall["macro_f1"],
        "ece": overall["ece"],
        "parameter_count": int(final_model["parameter_count"]),
        "inference_cost": dict(final_model["inference_cost"]),
    }
    validate_motion_major_metrics(major_metrics)

    report = {
        "schema_version": MOTION_EXTERNAL_REPORT_SCHEMA,
        "execution_status": "completed_external_evaluation_only",
        "internal_evidence_path": str(Path(internal_evidence_path).resolve()),
        "internal_evidence_sha256": evidence_sha256,
        "model_artifact_sha256": final_model["artifact_sha256"],
        "threshold_artifact_sha256": evidence["final_threshold_artifact_sha256"],
        "threshold_fit_scope": MOTION_THRESHOLD_FIT_SCOPE,
        "threshold_score_origin": MOTION_THRESHOLD_SCORE_ORIGIN,
        "ptt_registry_id": PTT_FORMAL_REGISTRY_ID,
        "ptt_registry_csv_sha256": PTT_SPLIT_CSV_SHA256,
        "ptt_repeat_seeds": list(PTT_FORMAL_REPEAT_SEEDS),
        "ptt_fold_sizes": list(PTT_FORMAL_FOLD_SIZES),
        "independence_claim": "none_not_an_independent_external_test",
        "fit_or_recalibration_performed": False,
        "prediction_row_count": len(prediction_rows),
        "major_metrics": major_metrics,
        "fold_metrics": fold_metrics,
        "ablation_executed": False,
        "formal_entry_id": formal_source_evidence["formal_entry_id"],
        "formal_source_evidence": dict(formal_source_evidence),
    }
    root = Path(output_dir).resolve()
    prediction_path = root / "motion_ptt_window_predictions.parquet"
    prediction_sha256 = _write_parquet(prediction_path, prediction_rows)
    report["prediction_parquet_path"] = str(prediction_path)
    report["prediction_parquet_sha256"] = prediction_sha256
    report_path = root / "motion_ptt_external_report.json"
    report_sha256 = _write_strict_json(report_path, report)
    return MotionExternalRunResult(
        report=report,
        prediction_rows=tuple(prediction_rows),
        report_path=str(report_path),
        report_sha256=report_sha256,
        prediction_path=str(prediction_path),
        prediction_sha256=prediction_sha256,
    )


def run_internal_motion_oof(
    examples: Sequence[MotionWindowExample],
    fold_jobs: Sequence[MotionFoldJob],
    *,
    fit_model: FitModel,
    predict_probability: PredictProbability,
    model_input_schema_path: str | Path,
    expected_model_input_schema_sha256: str,
    output_dir: str | Path,
    motion_split_csv_path: str | Path | None = None,
    execution_mode: str = "smoke",
    write_artifacts: bool = True,
) -> MotionInternalRunResult:
    """Run injected smoke/internal checks; formal evidence is impossible here."""

    if execution_mode == "formal":
        raise FormalMotionEntryRequiredError(
            "injected examples/callbacks cannot enter formal motion; use "
            "run_formal_internal_motion_reference"
        )
    if execution_mode not in {"smoke", "internal"}:
        raise ValueError("injected motion execution_mode must be smoke or internal")
    return _run_internal_motion_oof_impl(
        examples,
        fold_jobs,
        fit_model=fit_model,
        predict_probability=predict_probability,
        model_input_schema_path=model_input_schema_path,
        expected_model_input_schema_sha256=expected_model_input_schema_sha256,
        output_dir=output_dir,
        motion_split_csv_path=motion_split_csv_path,
        execution_mode=execution_mode,
        write_artifacts=write_artifacts,
    )


def run_ptt_external_evaluation(
    examples: Sequence[MotionWindowExample],
    *,
    internal_evidence_path: str | Path,
    expected_internal_evidence_sha256: str,
    ptt_split_csv: str | Path,
    load_frozen_model: LoadFrozenModel,
    predict_probability: PredictProbability,
    output_dir: str | Path,
) -> MotionExternalRunResult:
    """Reject all injected PTT arrays/callbacks at the public boundary."""

    del (
        examples,
        internal_evidence_path,
        expected_internal_evidence_sha256,
        ptt_split_csv,
        load_frozen_model,
        predict_probability,
        output_dir,
    )
    raise FormalMotionEntryRequiredError(
        "injected PTT examples/callbacks are forbidden; use "
        "run_formal_ptt_motion_reference"
    )


__all__ = [
    "FitModel",
    "FormalMotionEntryRequiredError",
    "LoadFrozenModel",
    "MOTION_EXTERNAL_REPORT_SCHEMA",
    "MOTION_INPUT_SCHEMA_STATUS",
    "MOTION_WINDOW_OOF_SCHEMA",
    "MotionExternalRunResult",
    "MotionFitContext",
    "MotionFittedArtifact",
    "MotionInternalRunResult",
    "MotionPredictionInput",
    "MotionWindowExample",
    "PredictProbability",
    "evaluate_ptt_external_gate",
    "load_motion_internal_evidence",
    "run_internal_motion_oof",
    "run_ptt_external_evaluation",
]
