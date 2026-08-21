"""Directly usable materialize/fit/predict/load adapters for V2 motion.

Importing this module performs no materialization, training, evaluation, or
benchmark. The public callbacks match motion_runner exactly and keep the
frozen split authority in that runner.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..models.motion import (
    build_formal_motion_cnn,
    count_trainable_parameters,
)
from ..motion_ids import FORMAL_MOTION_MODEL_ID
from ..representations.motion import (
    MOTION_NETWORK_CHANNEL_SCHEMA,
    MOTION_NETWORK_SCHEMA_SHA256,
    MOTION_REFERENCE_PROFILE_ID,
    MOTION_WINDOW_SAMPLES,
    MotionFoldImuTransform,
    apply_motion_fold_imu_transform,
    build_motion_window_tensors,
    fit_motion_fold_imu_transform,
    motion_network_schema_payload,
)
from ..signal.motion_imu import (
    CALIBRATED_ROLL_PITCH_EKF_PROFILE_ID,
    PTT_STATIC_CALIBRATION_ROLE,
    MotionImuResult,
)
from .motion import FORMAL_MOTION_TRAINING_CONFIG, motion_activity_label
from .motion_runner import (
    MotionFitContext,
    MotionFittedArtifact,
    MotionPredictionInput,
    MotionWindowExample,
)


FORMAL_MOTION_ARTIFACT_SCHEMA = (
    "ppg_frailty.formal_motion_model_artifact.imu_iqr_over_1p349.v3"
)
FORMAL_MOTION_TRAINER_SCHEMA = "ppg_frailty.formal_motion_trainer.v2"
_CUDA_DEVICE = re.compile(r"^cuda(?::([0-9]+))?$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class MotionRecordingInput:
    """One already synchronized, SI-unit, EKF-processed recording."""

    ppg_red_ir: np.ndarray
    motion_imu: MotionImuResult
    record_id: str
    participant_id: str
    role_or_activity: str
    dataset_id: str
    fs_hz: float = 400.0

    def validate(self) -> None:
        if not self.record_id or not self.participant_id or not self.dataset_id:
            raise ValueError("motion recording identity fields must be non-empty")
        if not self.role_or_activity:
            raise ValueError("motion recording role/activity must be explicit")
        if float(self.fs_hz) != 400.0:
            raise ValueError("formal motion materialization requires exactly 400 Hz")
        self.motion_imu.validate()
        if self.motion_imu.profile_id != CALIBRATED_ROLL_PITCH_EKF_PROFILE_ID:
            raise ValueError("formal motion materializer requires calibrated EKF reference")
        if self.motion_imu.diagnostics.get("calibration_participant_id") != self.participant_id:
            raise ValueError("formal motion materializer calibration participant drift")


def materialize_motion_window_examples(
    recordings: Sequence[MotionRecordingInput],
    *,
    dataset_kind: str,
) -> tuple[MotionWindowExample, ...]:
    """Materialize exact 8-s/2-s examples for internal or PTT records."""

    kind = str(dataset_kind).strip().lower()
    if kind not in {"internal", "ptt"}:
        raise ValueError("motion dataset_kind must be internal or ptt")
    examples: list[MotionWindowExample] = []
    for recording in recordings:
        recording.validate()
        if kind == "internal":
            if recording.motion_imu.diagnostics.get("calibration_source_role") != "B":
                raise ValueError("internal motion recording requires same-participant B calibration")
            label = motion_activity_label(recording.role_or_activity)
        else:
            if (
                recording.motion_imu.diagnostics.get("calibration_source_role")
                != PTT_STATIC_CALIBRATION_ROLE
            ):
                raise ValueError(
                    "PTT motion recording requires declared same-participant sit-static calibration"
                )
            activity = recording.role_or_activity.strip().lower()
            if activity == "sit":
                label = 0
            elif activity in {"walk", "run"}:
                label = 1
            else:
                raise ValueError("PTT motion activity must be sit, walk, or run")
        windows = build_motion_window_tensors(
            recording.ppg_red_ir,
            recording.motion_imu,
            record_id=recording.record_id,
            participant_id=recording.participant_id,
            role_or_activity=recording.role_or_activity,
            dataset_id=recording.dataset_id,
            fs_hz=recording.fs_hz,
        )
        for index, start in enumerate(windows.start_samples.tolist()):
            examples.append(
                MotionWindowExample(
                    window_id=f"{recording.record_id}:{int(start):010d}",
                    participant_id=recording.participant_id,
                    file_id=recording.record_id,
                    role_or_activity=recording.role_or_activity,
                    activity_label=label,
                    values=np.asarray(windows.values[index], dtype=np.float32),
                    dataset_id=recording.dataset_id,
                )
            )
    if not examples:
        raise ValueError("motion materializer requires at least one recording")
    ids = [row.window_id for row in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("motion materializer produced duplicate window IDs")
    return tuple(examples)


def write_formal_motion_input_schema(path: str | Path) -> tuple[str, str]:
    """Write the canonical semantic tensor schema once and return file identity."""

    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite motion schema: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **motion_network_schema_payload(),
        "semantic_sha256": MOTION_NETWORK_SCHEMA_SHA256,
    }
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(target)
    return str(target), _sha256_file(target)


@dataclass(frozen=True)
class FormalMotionTrainerConfig:
    """Frozen V2 formal optimizer/epoch/sampler identity."""

    fixed_epochs: int = 10
    batch_size: int = 16
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    loss: str = "binary_cross_entropy_with_logits"
    class_weighting: str = "none_balancing_is_sampler_only"
    sampler: str = "historical_dataset_class_inverse_x_participant_sqrt"
    dropout: float = 0.0
    label_smoothing: float = 0.0
    gradient_clip_norm: float = 1.0
    seed: int = 42
    num_workers: int = 0
    augmentation: str = "none"
    device: str = "cuda"
    inference_warmup_iterations: int = 10
    inference_timed_iterations: int = 50
    schema_version: str = FORMAL_MOTION_TRAINER_SCHEMA

    def validate(self) -> None:
        observed = asdict(self)
        if set(observed) != set(FORMAL_MOTION_TRAINING_CONFIG) or any(
            type(observed[name]) is not type(expected) or observed[name] != expected
            for name, expected in FORMAL_MOTION_TRAINING_CONFIG.items()
            if name != "device"
        ):
            raise ValueError("formal motion trainer configuration drift")
        if not isinstance(self.device, str) or (
            self.device != "cpu" and not _CUDA_DEVICE.fullmatch(self.device)
        ):
            raise ValueError(
                "formal motion artifact device must be cpu or explicit CUDA (cuda or cuda:N)"
            )


@dataclass
class FormalMotionRuntime:
    """Runtime model plus the fold-local six-axis reference IMU transform."""

    model: Any
    imu_transform: MotionFoldImuTransform
    device: str
    batch_size: int = 16


def _training_arrays(
    examples: Sequence[MotionWindowExample],
    context: MotionFitContext,
) -> tuple[np.ndarray, np.ndarray, MotionFoldImuTransform]:
    if not examples:
        raise ValueError("formal motion fit requires training windows")
    values = np.stack([np.asarray(row.values, dtype=np.float32) for row in examples])
    labels = np.asarray([row.activity_label for row in examples], dtype=np.float32)
    participant_ids = tuple(row.participant_id for row in examples)
    if values.shape[1:] != (
        len(MOTION_NETWORK_CHANNEL_SCHEMA),
        MOTION_WINDOW_SAMPLES,
    ):
        raise ValueError("formal motion fit input must use canonical 8-channel windows")
    if set(labels.tolist()) != {0.0, 1.0}:
        raise ValueError("formal motion fit requires both activity classes")
    transform = fit_motion_fold_imu_transform(
        values,
        participant_ids,
        fitted_on_participant_ids=context.training_participant_ids,
        outer_train_participant_ids=context.training_participant_ids,
        outer_oof_participant_ids=context.held_out_participant_ids,
    )
    return apply_motion_fold_imu_transform(values, transform), labels, transform


def _balanced_sampler_weights(
    examples: Sequence[MotionWindowExample],
) -> np.ndarray:
    """Port the archived detector's dataset/class and participant balancing."""

    class_counts: dict[tuple[str, int], int] = {}
    participant_counts: dict[str, int] = {}
    keys: list[tuple[str, int, str]] = []
    for row in examples:
        key = (row.dataset_id, int(row.activity_label))
        keys.append((key[0], key[1], row.participant_id))
        class_counts[key] = class_counts.get(key, 0) + 1
        participant_counts[row.participant_id] = participant_counts.get(row.participant_id, 0) + 1
    weights = np.asarray(
        [
            1.0 / class_counts[(dataset, label)]
            / np.sqrt(participant_counts[participant])
            for dataset, label, participant in keys
        ],
        dtype=np.float64,
    )
    return weights / float(np.mean(weights))


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("formal motion adapters require the deep PyTorch profile") from exc
    return torch


def validate_formal_motion_cuda_device(device: str) -> str:
    """Validate a CUDA device name without importing/probing PyTorch."""

    if not isinstance(device, str) or not _CUDA_DEVICE.fullmatch(device):
        raise ValueError(
            "formal motion training requires explicit CUDA (cuda or cuda:N); "
            "CPU fallback is forbidden"
        )
    return device


def require_formal_motion_cuda(
    config: FormalMotionTrainerConfig = FormalMotionTrainerConfig(),
) -> Any:
    """Fail before materialization when the requested CUDA device is unusable."""

    config.validate()
    validate_formal_motion_cuda_device(config.device)
    expected_workspace = ":4096:8"
    deterministic_workspaces = {":16:8", expected_workspace}
    observed_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if (
        observed_workspace is not None
        and observed_workspace not in deterministic_workspaces
    ):
        raise RuntimeError(
            "Stage5 deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG "
            f"to be :4096:8 or :16:8; observed "
            f"{observed_workspace!r} before CUDA initialization"
        )
    # PyTorch requires this variable to be present before the first cuBLAS
    # operation when deterministic algorithms are enabled.
    if observed_workspace is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected_workspace
    torch = _require_torch()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Stage5 formal motion runtime requires CUDA, but "
            "torch.cuda.is_available() is false"
        )
    requested_index = torch.device(config.device).index
    if requested_index is not None and requested_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Stage5 motion training requested {config.device}, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are visible"
        )
    try:
        probe = torch.empty(1, dtype=torch.float32, device=config.device)
        torch.cuda.synchronize(config.device)
        del probe
    except Exception as exc:
        raise RuntimeError(
            f"Stage5 motion training cannot initialize requested device {config.device}"
        ) from exc
    return torch


def _benchmark_inference(
    model: Any,
    sample: np.ndarray,
    config: FormalMotionTrainerConfig,
) -> dict[str, Any]:
    torch = _require_torch()
    tensor = torch.as_tensor(sample[:1], dtype=torch.float32, device=config.device)
    model.eval()
    with torch.no_grad():
        for _ in range(config.inference_warmup_iterations):
            model(tensor)
        torch.cuda.synchronize(config.device)
        durations_ms: list[float] = []
        for _ in range(config.inference_timed_iterations):
            torch.cuda.synchronize(config.device)
            started = time.perf_counter()
            model(tensor)
            torch.cuda.synchronize(config.device)
            durations_ms.append((time.perf_counter() - started) * 1000.0)
    p50 = float(np.percentile(durations_ms, 50.0))
    p95 = float(np.percentile(durations_ms, 95.0))
    return {
        "device": config.device,
        "batch_size": 1,
        "window_samples": MOTION_WINDOW_SAMPLES,
        "warmup_iterations": config.inference_warmup_iterations,
        "timed_iterations": config.inference_timed_iterations,
        "latency_ms_per_window_p50": p50,
        "latency_ms_per_window_p95": p95,
        "throughput_windows_per_second": 1000.0 / float(np.mean(durations_ms)),
    }


def fit_formal_motion_model(
    examples: Sequence[MotionWindowExample],
    context: MotionFitContext,
    *,
    config: FormalMotionTrainerConfig = FormalMotionTrainerConfig(),
) -> MotionFittedArtifact:
    """Fit one fixed-epoch model and persist its fold-local scaler with weights.

    This adapter is selected only by the no-callback canonical formal entry.
    It performs no split construction, threshold selection, OOF access, PTT
    access, early stopping, augmentation, or model selection. The returned
    runtime is reloaded from the just-written artifact, never the in-memory
    training object.
    """

    config.validate()
    torch = require_formal_motion_cuda(config)
    if context.training_seed != config.seed:
        raise ValueError("formal motion fit context training seed must remain 42")
    if context.model_input_schema_sha256 != MOTION_NETWORK_SCHEMA_SHA256:
        raise ValueError("formal motion fit requires the canonical semantic schema hash")
    for row in examples:
        if context.training_dataset_kind == "frailty29":
            row.validate_internal()
        elif context.training_dataset_kind == "ptt":
            row.validate_ptt()
        else:
            raise ValueError("formal motion training dataset kind is unregistered")
    scaled, labels, transform = _training_arrays(examples, context)
    weights = _balanced_sampler_weights(examples)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(examples),
        replacement=True,
        generator=generator,
    )
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(scaled, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        drop_last=False,
    )
    model = build_formal_motion_cnn().to(config.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    final_loss = float("nan")
    training_history: list[dict[str, Any]] = []
    evaluation_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )
    for epoch_index in range(config.fixed_epochs):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch_values, batch_labels in loader:
            batch_values = batch_values.to(config.device, non_blocking=True)
            batch_labels = batch_labels.to(config.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_values)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                batch_labels,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(batch_values.shape[0])
            total_rows += int(batch_values.shape[0])
        final_loss = total_loss / max(total_rows, 1)
        model.eval()
        observed_labels: list[np.ndarray] = []
        observed_predictions: list[np.ndarray] = []
        with torch.no_grad():
            for batch_values, batch_labels in evaluation_loader:
                logits = model(batch_values.to(config.device, non_blocking=True))
                observed_predictions.append(
                    (torch.sigmoid(logits) >= 0.5).to(torch.int64).cpu().numpy()
                )
                observed_labels.append(batch_labels.to(torch.int64).cpu().numpy())
        truth = np.concatenate(observed_labels)
        predicted = np.concatenate(observed_predictions)
        recalls = [
            float(np.mean(predicted[truth == label] == label))
            for label in (0, 1)
            if np.any(truth == label)
        ]
        training_history.append(
            {
                "epoch": epoch_index + 1,
                "training_loss": float(final_loss),
                "training_balanced_accuracy": float(np.mean(recalls)),
                "data_scope": (
                    f"all_{len(context.training_participant_ids)}_"
                    f"{context.training_dataset_kind}_participants"
                    if context.final_fit
                    else f"outer_training_{context.training_dataset_kind}_participants_only"
                ),
                "outer_heldout_used": False,
                "used_for_epoch_selection_or_checkpoint": False,
            }
        )
    if not np.isfinite(final_loss):
        raise RuntimeError("formal motion training produced a non-finite loss")

    history_path = context.artifact_directory / "motion_training_history.json"
    if history_path.exists():
        raise FileExistsError(f"refusing to overwrite motion history: {history_path}")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_payload = {
        "schema_version": "ppg_frailty.motion_training_history.v1",
        "repeat_index": context.repeat_index,
        "fold_index": context.fold_index,
        "final_fit": context.final_fit,
        "selection_rule": "fixed_epoch_history_is_diagnostic_only",
        "rows": training_history,
    }
    history_temporary = history_path.with_suffix(history_path.suffix + ".tmp")
    history_temporary.write_text(
        json.dumps(history_payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    history_temporary.replace(history_path)

    inference_cost = _benchmark_inference(model, scaled, config)
    artifact_path = context.artifact_directory / "formal_motion_model.pt"
    if artifact_path.exists():
        raise FileExistsError(f"refusing to overwrite motion model: {artifact_path}")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    parameter_count = count_trainable_parameters(model)
    payload = {
        "schema_version": FORMAL_MOTION_ARTIFACT_SCHEMA,
        "model_id": FORMAL_MOTION_MODEL_ID,
        "model_input_schema_sha256": MOTION_NETWORK_SCHEMA_SHA256,
        "trainer_config": asdict(config),
        "training_participant_ids": list(context.training_participant_ids),
        "final_training_loss": final_loss,
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "imu_transform": {
            "center": transform.center.tolist(),
            "scale": transform.scale.tolist(),
            "valid_count": transform.valid_count.tolist(),
            "fitted_on_participant_ids": list(transform.fitted_on_participant_ids),
            "artifact_sha256": transform.artifact_sha256,
            "profile_id": transform.profile_id,
            "schema_version": transform.schema_version,
            "channel_schema": list(transform.channel_schema),
        },
        "inference_cost": inference_cost,
    }
    temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(artifact_path)
    artifact_sha256 = _sha256_file(artifact_path)
    metadata = {
        "artifact_sha256": artifact_sha256,
        "training_participant_ids": list(context.training_participant_ids),
        "parameter_count": parameter_count,
        "inference_cost": inference_cost,
        "model_input_schema_sha256": MOTION_NETWORK_SCHEMA_SHA256,
    }
    # The full all-participant CPU tensor is no longer needed after the
    # artifact is written.  Release every owner before loading the small
    # deployment runtime, otherwise Python/PyTorch can retain multiple GiB
    # until the next fit has already allocated its own window bank.
    del evaluation_loader, loader, dataset, sampler, generator, optimizer
    del batch_values, batch_labels, logits, loss
    del scaled, labels, weights, model
    gc.collect()
    torch.cuda.empty_cache()
    runtime = load_formal_motion_model(
        artifact_path,
        metadata,
        runtime_device=config.device,
    )
    return MotionFittedArtifact(
        runtime_model=runtime,
        model_id=FORMAL_MOTION_MODEL_ID,
        artifact_path=str(artifact_path),
        artifact_sha256=artifact_sha256,
        model_input_schema_sha256=MOTION_NETWORK_SCHEMA_SHA256,
        training_participant_ids=tuple(context.training_participant_ids),
        parameter_count=int(metadata["parameter_count"]),
        inference_cost=inference_cost,
    )


def predict_formal_motion_probability(
    runtime_model: FormalMotionRuntime,
    rows: Sequence[MotionPredictionInput],
) -> np.ndarray:
    """Apply the stored fold scaler and return one label-free probability/row."""

    if not isinstance(runtime_model, FormalMotionRuntime):
        raise TypeError("formal motion predictor requires FormalMotionRuntime")
    if not rows:
        return np.empty(0, dtype=np.float64)
    torch = _require_torch()
    runtime_model.model.eval()
    outputs: list[np.ndarray] = []
    # Threshold fitting and OOF scoring can each contain tens of thousands of
    # windows.  Keep both host and device residency bounded to one inference
    # batch rather than stacking the complete fold/all-29 bank again.
    with torch.inference_mode():
        for start in range(0, len(rows), runtime_model.batch_size):
            batch = rows[start : start + runtime_model.batch_size]
            values = np.stack(
                [np.asarray(row.values, dtype=np.float32) for row in batch]
            )
            scaled = apply_motion_fold_imu_transform(
                values,
                runtime_model.imu_transform,
            )
            logits = runtime_model.model(
                torch.as_tensor(
                    scaled,
                    dtype=torch.float32,
                    device=runtime_model.device,
                )
            )
            outputs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.asarray(np.concatenate(outputs), dtype=np.float64)


def load_formal_motion_model(
    artifact_path: Path,
    metadata: Mapping[str, Any],
    *,
    runtime_device: str | None = None,
) -> FormalMotionRuntime:
    """Load only the strict V2 state/scaler payload used by the PTT runner."""

    required_metadata = {
        "artifact_sha256",
        "training_participant_ids",
        "parameter_count",
        "inference_cost",
        "model_input_schema_sha256",
    }
    if not required_metadata.issubset(metadata):
        raise ValueError("formal motion load metadata is incomplete")
    if metadata.get("artifact_sha256") != _sha256_file(artifact_path):
        raise ValueError("formal motion load artifact SHA-256 mismatch")
    if metadata.get("model_input_schema_sha256") != MOTION_NETWORK_SCHEMA_SHA256:
        raise ValueError("formal motion load metadata tensor schema drift")
    torch = _require_torch()
    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("formal motion artifact payload must be a mapping")
    expected_payload_fields = {
        "schema_version",
        "model_id",
        "model_input_schema_sha256",
        "trainer_config",
        "training_participant_ids",
        "final_training_loss",
        "state_dict",
        "imu_transform",
        "inference_cost",
    }
    if set(payload) != expected_payload_fields:
        raise ValueError("formal motion artifact field schema drift")
    if (
        payload.get("schema_version") != FORMAL_MOTION_ARTIFACT_SCHEMA
        or payload.get("model_id") != FORMAL_MOTION_MODEL_ID
        or payload.get("model_input_schema_sha256") != MOTION_NETWORK_SCHEMA_SHA256
    ):
        raise ValueError("formal motion artifact semantic identity drift")
    config = FormalMotionTrainerConfig(**dict(payload["trainer_config"]))
    config.validate()
    selected_device = config.device if runtime_device is None else str(runtime_device)
    if selected_device != "cpu":
        require_formal_motion_cuda(
            FormalMotionTrainerConfig(**{**asdict(config), "device": selected_device})
        )
    expected_roster = tuple(
        sorted(str(value) for value in metadata["training_participant_ids"])
    )
    payload_roster = tuple(
        sorted(str(value) for value in payload["training_participant_ids"])
    )
    if (
        len(payload_roster) != len(set(payload_roster))
        or payload_roster != expected_roster
        or not np.isfinite(float(payload["final_training_loss"]))
    ):
        raise ValueError("formal motion artifact training roster/loss drift")
    scaler = payload["imu_transform"]
    expected_scaler_fields = {
        "center",
        "scale",
        "valid_count",
        "fitted_on_participant_ids",
        "artifact_sha256",
        "profile_id",
        "schema_version",
        "channel_schema",
    }
    if not isinstance(scaler, Mapping) or set(scaler) != expected_scaler_fields:
        raise ValueError("formal motion artifact IMU scaler field schema drift")
    transform = MotionFoldImuTransform(
        center=np.asarray(scaler["center"], dtype=np.float64),
        scale=np.asarray(scaler["scale"], dtype=np.float64),
        valid_count=np.asarray(scaler["valid_count"], dtype=np.int64),
        fitted_on_participant_ids=tuple(scaler["fitted_on_participant_ids"]),
        artifact_sha256=str(scaler["artifact_sha256"]),
        profile_id=str(scaler["profile_id"]),
        schema_version=str(scaler["schema_version"]),
        channel_schema=tuple(scaler["channel_schema"]),
    )
    transform.validate()
    if transform.profile_id != MOTION_REFERENCE_PROFILE_ID:
        raise ValueError("formal motion artifact must use the 8-channel reference profile")
    if tuple(sorted(transform.fitted_on_participant_ids)) != expected_roster:
        raise ValueError("formal motion artifact IMU scaler participant roster drift")
    model = build_formal_motion_cnn()
    state_dict = payload["state_dict"]
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("formal motion artifact state_dict is empty or malformed")
    model.load_state_dict(state_dict, strict=True)
    if count_trainable_parameters(model) != int(metadata["parameter_count"]):
        raise ValueError("formal motion artifact parameter-count drift")
    if dict(payload["inference_cost"]) != dict(metadata["inference_cost"]):
        raise ValueError("formal motion artifact inference-cost metadata drift")
    model.to(selected_device)
    model.eval()
    return FormalMotionRuntime(
        model=model,
        imu_transform=transform,
        device=selected_device,
        batch_size=config.batch_size,
    )


__all__ = [
    "FORMAL_MOTION_ARTIFACT_SCHEMA",
    "FORMAL_MOTION_TRAINER_SCHEMA",
    "FormalMotionRuntime",
    "FormalMotionTrainerConfig",
    "MotionRecordingInput",
    "fit_formal_motion_model",
    "load_formal_motion_model",
    "materialize_motion_window_examples",
    "predict_formal_motion_probability",
    "require_formal_motion_cuda",
    "validate_formal_motion_cuda_device",
    "write_formal_motion_input_schema",
]
