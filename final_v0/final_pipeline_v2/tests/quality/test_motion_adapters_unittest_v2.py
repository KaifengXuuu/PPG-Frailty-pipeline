"""Non-scientific formal motion adapter and typed-evidence smoke tests."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ppg_frailty.quality.motion import (
    MOTION_THRESHOLD_SCORE_ORIGIN,
    MotionFoldJob,
)
from ppg_frailty.quality.motion_adapters import (
    FORMAL_MOTION_ARTIFACT_SCHEMA,
    FormalMotionRuntime,
    FormalMotionTrainerConfig,
    MotionRecordingInput,
    fit_formal_motion_model,
    load_formal_motion_model,
    materialize_motion_window_examples,
    predict_formal_motion_probability,
    require_formal_motion_cuda,
    write_formal_motion_input_schema,
)
from ppg_frailty.quality.motion_runner import (
    MOTION_WINDOW_OOF_SCHEMA,
    MotionPredictionInput,
    _read_motion_parquet,
    _validate_internal_oof_rows,
    _write_parquet,
)
from ppg_frailty.models.motion import (
    build_formal_motion_cnn,
    count_trainable_parameters,
)
from ppg_frailty.motion_ids import FORMAL_MOTION_MODEL_ID
from ppg_frailty.representations.motion import (
    MOTION_NETWORK_CHANNEL_SCHEMA,
    MOTION_NETWORK_SCHEMA_SHA256,
    apply_motion_fold_imu_transform,
    fit_motion_fold_imu_transform,
)
from ppg_frailty.signal.motion_imu import (
    MotionImuResult,
    RollPitchEkfConfig,
    fit_motion_imu_calibration,
    preprocess_motion_imu_calibrated_ekf,
)


def _imu(samples: int) -> MotionImuResult:
    acceleration = np.tile([0.0, 0.0, 1.0], (samples, 1))
    gyroscope = np.zeros((samples, 3), dtype=np.float64)
    config = RollPitchEkfConfig(
        calibration_start_s=0.2,
        calibration_stop_s=1.2,
    )
    calibration = fit_motion_imu_calibration(
        acceleration,
        gyroscope,
        participant_id="p1",
        file_id="p1_B",
        source_role="B",
        fs_hz=400.0,
        acceleration_unit="g",
        gyroscope_unit="deg/s",
        config=config,
    )
    result = preprocess_motion_imu_calibrated_ekf(
        acceleration,
        gyroscope,
        fs_hz=400.0,
        acceleration_unit="g",
        gyroscope_unit="deg/s",
        participant_id="p1",
        calibration=calibration,
        config=config,
    )
    result.validate()
    return result


class MotionAdaptersTest(unittest.TestCase):
    def test_materializer_and_schema_writer_are_directly_usable(self) -> None:
        samples = 4000
        time = np.arange(samples, dtype=np.float64) / 400.0
        recording = MotionRecordingInput(
            ppg_red_ir=np.column_stack(
                (
                    np.sin(2.0 * np.pi * time),
                    np.cos(2.0 * np.pi * time),
                )
            ),
            motion_imu=_imu(samples),
            record_id="p1_B",
            participant_id="p1",
            role_or_activity="B",
            dataset_id="frailty29",
        )
        examples = materialize_motion_window_examples(
            [recording],
            dataset_kind="internal",
        )
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].values.shape, (8, 3200))
        self.assertEqual(examples[0].activity_label, 0)
        with tempfile.TemporaryDirectory() as directory:
            schema_path, file_sha = write_formal_motion_input_schema(
                Path(directory) / "motion_schema.json"
            )
            payload = json.loads(Path(schema_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["semantic_sha256"], MOTION_NETWORK_SCHEMA_SHA256)
            self.assertEqual(len(file_sha), 64)

    def test_trainer_configuration_is_frozen_without_training(self) -> None:
        FormalMotionTrainerConfig().validate()
        self.assertEqual(FormalMotionTrainerConfig().device, "cuda")
        with self.assertRaisesRegex(ValueError, "configuration drift"):
            replace(FormalMotionTrainerConfig(), fixed_epochs=7).validate()
        historical_cpu = replace(FormalMotionTrainerConfig(), device="cpu")
        historical_cpu.validate()
        with self.assertRaisesRegex(ValueError, "CPU fallback is forbidden"):
            require_formal_motion_cuda(historical_cpu)
        with self.assertRaisesRegex(ValueError, "CPU fallback is forbidden"):
            fit_formal_motion_model((), SimpleNamespace(), config=historical_cpu)

    def test_cuda_preflight_fails_instead_of_silently_using_cpu(self) -> None:
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        with patch.dict(
            os.environ,
            {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
        ), patch(
            "ppg_frailty.quality.motion_adapters._require_torch",
            return_value=fake_torch,
        ):
            with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                require_formal_motion_cuda()

    def test_motion_transform_and_prediction_keep_float32_and_batch_bound(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        values = np.zeros((5, 8, 3200), dtype=np.float32)
        for index in range(values.shape[0]):
            values[index, 2:, :] = np.float32(index + 1)
        transform = fit_motion_fold_imu_transform(
            values,
            ("p1",) * len(values),
            fitted_on_participant_ids=("p1",),
            outer_train_participant_ids=("p1",),
            outer_oof_participant_ids=(),
        )
        scaled = apply_motion_fold_imu_transform(values, transform)
        self.assertEqual(scaled.dtype, np.float32)
        legacy_formula = values.astype(np.float64)
        legacy_formula[:, 2:, :] = (
            legacy_formula[:, 2:, :]
            - transform.center.reshape(1, 6, 1)
        ) / transform.scale.reshape(1, 6, 1)
        np.testing.assert_array_equal(scaled, legacy_formula.astype(np.float32))

        class BatchRecorder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.maximum_batch = 0

            def forward(self, tensor):
                self.maximum_batch = max(self.maximum_batch, int(tensor.shape[0]))
                return torch.zeros(tensor.shape[0], device=tensor.device)

        model = BatchRecorder()
        runtime = FormalMotionRuntime(
            model=model,
            imu_transform=transform,
            device="cpu",
            batch_size=2,
        )
        probabilities = predict_formal_motion_probability(
            runtime,
            tuple(MotionPredictionInput(row) for row in values),
        )
        self.assertEqual(probabilities.shape, (5,))
        self.assertEqual(model.maximum_batch, 2)

    def test_strict_model_loader_rejects_mock_or_field_drift_without_training(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        values = np.zeros((4, 8, 3200), dtype=np.float32)
        values[:, 2:, :] = np.arange(4, dtype=np.float32)[:, None, None]
        transform = fit_motion_fold_imu_transform(
            values,
            ("p1", "p1", "p2", "p2"),
            fitted_on_participant_ids=("p1", "p2"),
            outer_train_participant_ids=("p1", "p2"),
            outer_oof_participant_ids=(),
        )
        model = build_formal_motion_cnn()
        inference_cost = {
            "device": "cpu",
            "batch_size": 1,
            "window_samples": 3200,
            "warmup_iterations": 1,
            "timed_iterations": 1,
            "latency_ms_per_window_p50": 1.0,
            "latency_ms_per_window_p95": 1.0,
            "throughput_windows_per_second": 1.0,
        }
        payload = {
            "schema_version": FORMAL_MOTION_ARTIFACT_SCHEMA,
            "model_id": FORMAL_MOTION_MODEL_ID,
            "model_input_schema_sha256": MOTION_NETWORK_SCHEMA_SHA256,
            "trainer_config": asdict(
                replace(FormalMotionTrainerConfig(), device="cpu")
            ),
            "training_participant_ids": ["p1", "p2"],
            "final_training_loss": 0.5,
            "state_dict": model.state_dict(),
            "imu_transform": {
                "center": transform.center.tolist(),
                "scale": transform.scale.tolist(),
                "valid_count": transform.valid_count.tolist(),
                "fitted_on_participant_ids": list(
                    transform.fitted_on_participant_ids
                ),
                "artifact_sha256": transform.artifact_sha256,
                "profile_id": transform.profile_id,
                "schema_version": transform.schema_version,
                "channel_schema": list(MOTION_NETWORK_CHANNEL_SCHEMA[2:]),
            },
            "inference_cost": inference_cost,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "formal.pt"
            torch.save(payload, artifact)
            metadata = {
                "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "training_participant_ids": ["p1", "p2"],
                "parameter_count": count_trainable_parameters(model),
                "inference_cost": inference_cost,
                "model_input_schema_sha256": MOTION_NETWORK_SCHEMA_SHA256,
            }
            loaded = load_formal_motion_model(artifact, metadata)
            self.assertEqual(loaded.device, "cpu")
            self.assertEqual(
                tuple(loaded.imu_transform.fitted_on_participant_ids),
                ("p1", "p2"),
            )
            malformed = dict(payload)
            del malformed["state_dict"]
            malformed_path = root / "malformed.pt"
            torch.save(malformed, malformed_path)
            malformed_metadata = {
                **metadata,
                "artifact_sha256": hashlib.sha256(
                    malformed_path.read_bytes()
                ).hexdigest(),
            }
            with self.assertRaisesRegex(ValueError, "field schema"):
                load_formal_motion_model(malformed_path, malformed_metadata)

    def test_typed_oof_rows_reject_prediction_tampering(self) -> None:
        job = MotionFoldJob(
            repeat_index=0,
            fold_index=0,
            split_seed=42,
            training_seed=42,
            train_participant_ids=("train",),
            oof_participant_ids=("held",),
        )
        model_sha = "a" * 64
        rows = [
            {
                "schema_version": MOTION_WINDOW_OOF_SCHEMA,
                "repeat_index": 0,
                "fold_index": 0,
                "split_seed": 42,
                "training_seed": 42,
                "window_id": f"held_{role}",
                "participant_id": "held",
                "file_id": f"held_{role}",
                "role_family": role,
                "activity_label": label,
                "p_active": probability,
                "threshold": 0.5,
                "predicted_activity": label,
                "score_origin": "strict_outer_oof_model_prediction",
                "threshold_score_origin": MOTION_THRESHOLD_SCORE_ORIGIN,
                "model_artifact_sha256": model_sha,
            }
            for role, label, probability in (("B", 0, 0.2), ("S1", 1, 0.8))
        ]
        _validate_internal_oof_rows(rows, [job])
        tampered = [dict(row) for row in rows]
        tampered[0]["predicted_activity"] = 1
        with self.assertRaisesRegex(ValueError, "score/prediction"):
            _validate_internal_oof_rows(tampered, [job])
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow benchmark profile is not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oof.parquet"
            digest = _write_parquet(path, rows)
            self.assertEqual(len(digest), 64)
            self.assertEqual(_read_motion_parquet(path, MOTION_WINDOW_OOF_SCHEMA), rows)


if __name__ == "__main__":
    unittest.main()
