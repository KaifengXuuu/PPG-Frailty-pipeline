"""Calibrated roll-pitch EKF motion preprocessing with no silent fallback."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from scipy import signal

from ..provenance import stable_payload_sha256
from .imu import convert_gyro
from .views import CANONICAL_FS_HZ


MOTION_IMU_CHANNEL_SCHEMA = (
    "A_dyn_x",
    "A_dyn_y",
    "A_dyn_z",
    "GX",
    "GY",
    "GZ",
    "A_mag",
    "Omega_mag",
    "J_mag",
)
MOTION_IMU_CHANNEL_UNITS = (
    "m/s^2",
    "m/s^2",
    "m/s^2",
    "rad/s",
    "rad/s",
    "rad/s",
    "m/s^2",
    "rad/s",
    "m/s^3",
)
CALIBRATED_ROLL_PITCH_EKF_PROFILE_ID = (
    "calibrated_roll_pitch_ekf_sensor_lpf_order3_v4_reference"
)
PROFILE_A_LPF_ID = (
    "profile_a_sensor_lpf_order3_gravity_0p3hz_v4_ablation"
)
PTT_STATIC_CALIBRATION_ROLE = "PTT_SIT_STATIC_CALIBRATION"
FORMAL_STATIC_CALIBRATION_ROLES = ("B", PTT_STATIC_CALIBRATION_ROLE)
MOTION_IMU_CALIBRATION_SCHEMA = (
    "ppg_frailty.motion_imu_calibration.sensor_lpf_order3.v4"
)
MOTION_IMU_LINEAGE_SCHEMA = (
    "ppg_frailty.motion_imu_lineage.sensor_lpf_order3.v4"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _array_lineage_sha256(values: np.ndarray, semantic: str) -> str:
    array = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "semantic": str(semantic),
                "dtype": "<f8",
                "shape": list(array.shape),
                "order": "C",
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class RollPitchEkfConfig:
    """All numerical EKF and calibration settings; persisted with every result."""

    process_covariance_diagonal_per_second: tuple[float, ...] = (
        5.0,
        5.0,
        0.05,
        0.05,
        0.05,
    )
    observation_covariance_diagonal_rad2: tuple[float, float] = (0.5, 0.5)
    initial_covariance_diagonal: tuple[float, ...] = (1.0, 1.0, 0.5, 0.5, 0.5)
    dynamic_observation_scale: float = 3.0
    gravity_mps2: float = 9.81
    accelerometer_lowpass_hz: float = 20.0
    gyroscope_lowpass_hz: float = 40.0
    sensor_filter_order: int = 3
    calibration_start_s: float = 5.0
    calibration_stop_s: float = 100.0
    gravity_lowpass_hz: float = 0.3
    gravity_filter_order: int = 4
    source_algorithm: str = (
        "authoritative_calibrated_roll_pitch_bias_ekf_"
        "one_sided_dynamic_R_sensor_lpf_order3_v4"
    )

    def validate(self, fs_hz: float = CANONICAL_FS_HZ) -> None:
        q = np.asarray(self.process_covariance_diagonal_per_second, dtype=np.float64)
        r = np.asarray(self.observation_covariance_diagonal_rad2, dtype=np.float64)
        p = np.asarray(self.initial_covariance_diagonal, dtype=np.float64)
        if q.shape != (5,) or r.shape != (2,) or p.shape != (5,):
            raise ValueError("roll-pitch EKF covariance diagonals have wrong shape")
        if not np.isfinite(q).all() or not np.isfinite(r).all() or not np.isfinite(p).all():
            raise ValueError("roll-pitch EKF covariances must be finite")
        if np.any(q <= 0.0) or np.any(r <= 0.0) or np.any(p <= 0.0):
            raise ValueError("roll-pitch EKF covariance diagonals must be positive")
        if (
            not np.isfinite(self.dynamic_observation_scale)
            or self.dynamic_observation_scale < 0.0
        ):
            raise ValueError("dynamic observation scale cannot be negative")
        if not np.isfinite(self.gravity_mps2) or self.gravity_mps2 <= 0.0:
            raise ValueError("roll-pitch EKF gravity_mps2 must be finite and positive")
        if not np.isfinite(fs_hz) or fs_hz <= 0.0:
            raise ValueError("motion IMU sampling frequency must be finite and positive")
        for name, order in (
            ("sensor_filter_order", self.sensor_filter_order),
            ("gravity_filter_order", self.gravity_filter_order),
        ):
            if (
                isinstance(order, bool)
                or not isinstance(order, (int, np.integer))
                or not 1 <= int(order) <= 20
            ):
                raise ValueError(f"{name} must be an integer in [1,20]")
        if any(
            not np.isfinite(value) or not 0.0 < float(value) < fs_hz / 2.0
            for value in (
                self.accelerometer_lowpass_hz,
                self.gyroscope_lowpass_hz,
                self.gravity_lowpass_hz,
            )
        ):
            raise ValueError("motion IMU filter configuration is invalid")
        if (
            not np.isfinite([self.calibration_start_s, self.calibration_stop_s]).all()
            or not 0.0 <= self.calibration_start_s < self.calibration_stop_s
        ):
            raise ValueError("motion calibration interval is invalid")


@dataclass(frozen=True)
class MotionImuCalibration:
    """Static-segment sensor calibration bound to its participant and file."""

    participant_id: str
    file_id: str
    source_role: str
    acceleration_bias_mps2: np.ndarray
    gyroscope_bias_rads: np.ndarray
    initial_roll_rad: float
    initial_pitch_rad: float
    calibration_start_sample: int
    calibration_stop_sample: int
    calibration_quality: dict[str, Any]
    config: RollPitchEkfConfig
    artifact_sha256: str
    schema_version: str = MOTION_IMU_CALIBRATION_SCHEMA

    def validate(self) -> None:
        acc_bias = np.asarray(self.acceleration_bias_mps2, dtype=np.float64)
        gyro_bias = np.asarray(self.gyroscope_bias_rads, dtype=np.float64)
        if not self.participant_id or not self.file_id:
            raise ValueError("motion IMU calibration identity is incomplete")
        if self.source_role not in FORMAL_STATIC_CALIBRATION_ROLES:
            raise ValueError(
                "formal motion IMU calibration must come from same-participant B "
                "or explicitly declared PTT sit-static calibration"
            )
        if acc_bias.shape != (3,) or gyro_bias.shape != (3,):
            raise ValueError("motion IMU calibration bias shape drift")
        if not np.isfinite(acc_bias).all() or not np.isfinite(gyro_bias).all():
            raise ValueError("motion IMU calibration biases must be finite")
        if self.calibration_stop_sample <= self.calibration_start_sample:
            raise ValueError("motion IMU calibration sample range is empty")
        if self.schema_version != MOTION_IMU_CALIBRATION_SCHEMA:
            raise ValueError("motion IMU calibration schema drift")
        self.config.validate()
        expected = _calibration_hash(
            self.participant_id,
            self.file_id,
            self.source_role,
            acc_bias,
            gyro_bias,
            self.initial_roll_rad,
            self.initial_pitch_rad,
            self.calibration_start_sample,
            self.calibration_stop_sample,
            self.config,
        )
        if self.artifact_sha256 != expected:
            raise ValueError("motion IMU calibration artifact identity drift")


@dataclass(frozen=True)
class MotionImuResult:
    """Nine SI-unit motion channels plus audit arrays and EKF provenance."""

    values: np.ndarray
    channel_schema: tuple[str, ...]
    channel_units: tuple[str, ...]
    roll_rad: np.ndarray
    pitch_rad: np.ndarray
    gravity_mps2: np.ndarray
    valid_mask: np.ndarray
    profile_id: str
    diagnostics: dict[str, Any]

    def validate(self) -> None:
        matrix = np.asarray(self.values)
        samples = matrix.shape[0] if matrix.ndim == 2 else -1
        if matrix.ndim != 2 or matrix.shape[1] != 9:
            raise ValueError("motion IMU result must have shape samples-by-9")
        if self.channel_schema != MOTION_IMU_CHANNEL_SCHEMA:
            raise ValueError("motion IMU channel schema drift")
        if self.channel_units != MOTION_IMU_CHANNEL_UNITS:
            raise ValueError("motion IMU unit schema drift")
        if (
            np.asarray(self.roll_rad).shape != (samples,)
            or np.asarray(self.pitch_rad).shape != (samples,)
            or np.asarray(self.gravity_mps2).shape != (samples, 3)
            or np.asarray(self.valid_mask).shape != (samples,)
        ):
            raise ValueError("motion IMU audit arrays lost sample alignment")
        if (
            not np.isfinite(matrix).all()
            or not np.isfinite(self.roll_rad).all()
            or not np.isfinite(self.pitch_rad).all()
            or not np.isfinite(self.gravity_mps2).all()
            or np.asarray(self.valid_mask).dtype != np.dtype(bool)
            or not np.all(self.valid_mask)
        ):
            raise ValueError("formal motion IMU result must be fully finite")
        if self.diagnostics.get("silent_fallback") is not False:
            raise ValueError("motion IMU result may not hide a fallback")
        known_profiles = {
            CALIBRATED_ROLL_PITCH_EKF_PROFILE_ID,
            PROFILE_A_LPF_ID,
        }
        if self.profile_id not in known_profiles:
            raise ValueError("motion IMU result profile identity is stale or unknown")
        if self.diagnostics.get("profile_id") != self.profile_id:
            raise ValueError("motion IMU result/diagnostics profile identity drift")
        if self.profile_id == CALIBRATED_ROLL_PITCH_EKF_PROFILE_ID:
            required_hashes = {
                "source_acceleration_sha256",
                "source_gyroscope_sha256",
                "filtered_si_acceleration_sha256",
                "filtered_si_gyroscope_sha256",
                "ekf_config_sha256",
                "calibration_artifact_sha256",
                "output_values_sha256",
                "gravity_sha256",
                "roll_pitch_sha256",
                "lineage_sha256",
            }
            if self.diagnostics.get("lineage_schema") != MOTION_IMU_LINEAGE_SCHEMA:
                raise ValueError("formal motion IMU lineage schema drift")
            if (
                self.diagnostics.get("profile_id")
                != CALIBRATED_ROLL_PITCH_EKF_PROFILE_ID
                or not str(self.diagnostics.get("calibration_file_id", "")).strip()
                or self.diagnostics.get("calibration_source_role")
                not in FORMAL_STATIC_CALIBRATION_ROLES
                or self.diagnostics.get("calibration_participant_id")
                != self.diagnostics.get("runtime_participant_id")
            ):
                raise ValueError("formal motion IMU calibration/runtime lineage drift")
            unit_conversion = self.diagnostics.get("unit_conversion")
            if (
                not isinstance(unit_conversion, Mapping)
                or not str(unit_conversion.get("acceleration", "")).endswith(
                    "->m/s^2"
                )
                or not str(unit_conversion.get("gyroscope", "")).endswith(
                    "->rad/s"
                )
            ):
                raise ValueError("formal motion IMU unit-conversion lineage drift")
            ekf_config = self.diagnostics.get("ekf_config")
            if (
                not isinstance(ekf_config, Mapping)
                or self.diagnostics.get("ekf_config_sha256")
                != stable_payload_sha256(dict(ekf_config))
            ):
                raise ValueError("formal motion IMU EKF configuration lineage drift")
            if any(
                not _SHA256_PATTERN.fullmatch(str(self.diagnostics.get(name, "")))
                for name in required_hashes
            ):
                raise ValueError("formal motion IMU lineage hash missing")
            expected_arrays = {
                "output_values_sha256": _array_lineage_sha256(
                    matrix, "motion_imu_output_values"
                ),
                "gravity_sha256": _array_lineage_sha256(
                    self.gravity_mps2, "motion_imu_gravity"
                ),
                "roll_pitch_sha256": _array_lineage_sha256(
                    np.column_stack((self.roll_rad, self.pitch_rad)),
                    "motion_imu_roll_pitch",
                ),
            }
            if any(
                self.diagnostics.get(name) != expected
                for name, expected in expected_arrays.items()
            ):
                raise ValueError("formal motion IMU recomputable lineage drift")
            lineage_payload = {
                name: self.diagnostics[name]
                for name in sorted(required_hashes - {"lineage_sha256"})
            }
            lineage_payload["lineage_schema"] = MOTION_IMU_LINEAGE_SCHEMA
            if self.diagnostics["lineage_sha256"] != stable_payload_sha256(
                lineage_payload
            ):
                raise ValueError("formal motion IMU lineage aggregate hash drift")


def _zero_phase_lowpass(values: np.ndarray, cutoff_hz: float, fs_hz: float, order: int) -> np.ndarray:
    sos = signal.butter(order, cutoff_hz, btype="lowpass", fs=fs_hz, output="sos")
    try:
        output = signal.sosfiltfilt(sos, np.asarray(values, dtype=np.float64), axis=0)
    except ValueError as exc:
        raise ValueError(f"motion_imu_zero_phase_filter_failed:{exc}") from exc
    if not np.isfinite(output).all():
        raise FloatingPointError("motion_imu_filter_returned_nonfinite")
    return np.asarray(output, dtype=np.float64)


def _robust_mean(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    median = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - median), axis=0)
    output = np.empty(matrix.shape[1], dtype=np.float64)
    for axis in range(matrix.shape[1]):
        if mad[axis] <= 1e-12:
            output[axis] = float(np.mean(matrix[:, axis]))
        else:
            keep = np.abs(matrix[:, axis] - median[axis]) / mad[axis] < 3.5
            output[axis] = float(np.mean(matrix[keep, axis]))
    return output


def _roll_pitch_from_acc(acceleration: np.ndarray) -> tuple[float, float]:
    ax, ay, az = np.asarray(acceleration, dtype=np.float64)
    return (
        float(np.arctan2(ay, np.sqrt(ax * ax + az * az))),
        float(np.arctan2(-ax, np.sqrt(ay * ay + az * az))),
    )


def _gravity_from_roll_pitch(
    roll: np.ndarray,
    pitch: np.ndarray,
    *,
    gravity_mps2: float,
) -> np.ndarray:
    """Compute ``(Rx(phi) Ry(theta)).T @ [0, 0, g]`` sample-wise."""

    r = np.asarray(roll, dtype=np.float64)
    p = np.asarray(pitch, dtype=np.float64)
    return float(gravity_mps2) * np.column_stack(
        (-np.sin(p) * np.cos(r), np.sin(r), np.cos(p) * np.cos(r))
    )


def _convert_profile_acceleration(
    values: np.ndarray,
    unit: str,
    *,
    gravity_mps2: float,
) -> np.ndarray:
    """Apply the Profile-B 9.81 conversion while retaining explicit SI input."""

    normalized = str(unit).strip().lower().replace("²", "2")
    factors = {
        "g": float(gravity_mps2),
        "mg": float(gravity_mps2) / 1000.0,
        "m/s2": 1.0,
        "m/s^2": 1.0,
    }
    if normalized not in factors:
        raise ValueError(f"unit_unknown:acceleration_unit={unit}")
    return np.asarray(values, dtype=np.float64) * factors[normalized]


def _calibration_hash(
    participant_id: str,
    file_id: str,
    source_role: str,
    acc_bias: np.ndarray,
    gyro_bias: np.ndarray,
    roll: float,
    pitch: float,
    start: int,
    stop: int,
    config: RollPitchEkfConfig,
) -> str:
    return stable_payload_sha256(
        {
            "schema_version": MOTION_IMU_CALIBRATION_SCHEMA,
            "participant_id": participant_id,
            "file_id": file_id,
            "source_role": source_role,
            "acceleration_bias_mps2": np.asarray(acc_bias).tolist(),
            "gyroscope_bias_rads": np.asarray(gyro_bias).tolist(),
            "initial_roll_rad": float(roll),
            "initial_pitch_rad": float(pitch),
            "calibration_start_sample": int(start),
            "calibration_stop_sample": int(stop),
            "config": asdict(config),
        }
    )


def fit_motion_imu_calibration(
    acceleration: np.ndarray,
    gyroscope: np.ndarray,
    *,
    participant_id: str,
    file_id: str,
    source_role: str,
    fs_hz: float,
    acceleration_unit: str,
    gyroscope_unit: str,
    config: RollPitchEkfConfig,
) -> MotionImuCalibration:
    """Fit explicit static sensor biases; no label or held-out scaler is used."""

    if float(fs_hz) != CANONICAL_FS_HZ:
        raise ValueError("formal motion calibration requires exactly 400 Hz")
    if str(source_role) not in FORMAL_STATIC_CALIBRATION_ROLES:
        raise ValueError(
            "formal motion calibration source must be same-participant B or "
            "explicit PTT sit-static calibration"
        )
    config.validate(fs_hz)
    acc = _convert_profile_acceleration(
        acceleration,
        acceleration_unit,
        gravity_mps2=config.gravity_mps2,
    )
    gyro = convert_gyro(gyroscope, gyroscope_unit)
    if acc.ndim != 2 or gyro.shape != acc.shape or acc.shape[1] != 3:
        raise ValueError("motion calibration requires aligned samples-by-3 ACC/Gyro")
    if not np.isfinite(acc).all() or not np.isfinite(gyro).all():
        raise ValueError("motion calibration input must be finite")
    start = int(round(config.calibration_start_s * fs_hz))
    stop = int(round(config.calibration_stop_s * fs_hz))
    if start < 0 or stop > acc.shape[0] or stop - start < 16:
        raise ValueError("declared motion calibration interval is unavailable")
    acc_filtered = _zero_phase_lowpass(
        acc[start:stop], config.accelerometer_lowpass_hz, fs_hz, config.sensor_filter_order
    )
    gyro_filtered = _zero_phase_lowpass(
        gyro[start:stop], config.gyroscope_lowpass_hz, fs_hz, config.sensor_filter_order
    )
    acc_mean = _robust_mean(acc_filtered)
    gyro_mean = _robust_mean(gyro_filtered)
    roll, pitch = _roll_pitch_from_acc(acc_mean)
    gravity = _gravity_from_roll_pitch(
        np.asarray([roll]),
        np.asarray([pitch]),
        gravity_mps2=config.gravity_mps2,
    )[0]
    acc_bias = acc_mean - gravity
    quality = {
        "acceleration_gravity_norm_error_mps2": float(
            abs(np.linalg.norm(acc_mean - acc_bias) - config.gravity_mps2)
        ),
        "gyroscope_rms_rads_by_axis": np.sqrt(np.mean(np.square(gyro_filtered), axis=0)).tolist(),
        "calibration_sample_count": int(stop - start),
        "quality_threshold_applied": False,
    }
    digest = _calibration_hash(
        str(participant_id),
        str(file_id),
        str(source_role),
        acc_bias,
        gyro_mean,
        roll,
        pitch,
        start,
        stop,
        config,
    )
    result = MotionImuCalibration(
        participant_id=str(participant_id),
        file_id=str(file_id),
        source_role=str(source_role),
        acceleration_bias_mps2=acc_bias,
        gyroscope_bias_rads=gyro_mean,
        initial_roll_rad=roll,
        initial_pitch_rad=pitch,
        calibration_start_sample=start,
        calibration_stop_sample=stop,
        calibration_quality=quality,
        config=config,
        artifact_sha256=digest,
    )
    result.validate()
    return result


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def _run_roll_pitch_ekf(
    acceleration_mps2: np.ndarray,
    gyroscope_rads: np.ndarray,
    *,
    fs_hz: float,
    initial_roll_rad: float,
    initial_pitch_rad: float,
    config: RollPitchEkfConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Corrected historical five-state EKF using the calibrated initial state."""

    acc = np.asarray(acceleration_mps2, dtype=np.float64)
    gyro = np.asarray(gyroscope_rads, dtype=np.float64)
    count = acc.shape[0]
    roll = np.empty(count, dtype=np.float64)
    pitch = np.empty(count, dtype=np.float64)
    bias = np.zeros((count, 3), dtype=np.float64)
    roll[0], pitch[0] = float(initial_roll_rad), float(initial_pitch_rad)
    covariance = np.diag(np.asarray(config.initial_covariance_diagonal, dtype=np.float64))
    process = np.diag(
        np.asarray(config.process_covariance_diagonal_per_second, dtype=np.float64)
    ) / float(fs_hz)
    base_observation = np.asarray(
        config.observation_covariance_diagonal_rad2, dtype=np.float64
    )
    observation_scale_min = float("inf")
    observation_scale_max = 0.0
    dt = 1.0 / float(fs_hz)
    identity = np.eye(5, dtype=np.float64)
    measurement_matrix = np.array(
        [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    for index in range(1, count):
        previous_roll = roll[index - 1]
        previous_pitch = pitch[index - 1]
        cosine_pitch = float(np.cos(previous_pitch))
        if abs(cosine_pitch) < 1e-6:
            raise FloatingPointError("roll_pitch_ekf_gimbal_singularity")
        gx, gy, gz = gyro[index] - bias[index - 1]
        sin_roll, cos_roll = np.sin(previous_roll), np.cos(previous_roll)
        tan_pitch = np.tan(previous_pitch)
        secant_squared = 1.0 / (cosine_pitch * cosine_pitch)
        roll_rate = gx + gy * sin_roll * tan_pitch + gz * cos_roll * tan_pitch
        pitch_rate = gy * cos_roll - gz * sin_roll
        predicted = np.array(
            [
                previous_roll + dt * roll_rate,
                previous_pitch + dt * pitch_rate,
                *bias[index - 1],
            ],
            dtype=np.float64,
        )
        transition = np.array(
            [
                [
                    1.0 + dt * (gy * cos_roll * tan_pitch - gz * sin_roll * tan_pitch),
                    dt * (gy * sin_roll * secant_squared + gz * cos_roll * secant_squared),
                    -dt,
                    -dt * sin_roll * tan_pitch,
                    -dt * cos_roll * tan_pitch,
                ],
                [
                    dt * (-gy * sin_roll - gz * cos_roll),
                    1.0,
                    0.0,
                    -dt * cos_roll,
                    dt * sin_roll,
                ],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        covariance = transition @ covariance @ transition.T + process
        measured_roll, measured_pitch = _roll_pitch_from_acc(acc[index])
        measurement = np.asarray([measured_roll, measured_pitch], dtype=np.float64)
        deviation = max(
            0.0,
            float(np.linalg.norm(acc[index])) - config.gravity_mps2,
        ) / config.gravity_mps2
        scale = 1.0 + config.dynamic_observation_scale * deviation
        observation_scale_min = min(observation_scale_min, scale)
        observation_scale_max = max(observation_scale_max, scale)
        observation_covariance = np.diag(base_observation * scale)
        innovation = _wrap_angle(measurement - measurement_matrix @ predicted)
        innovation_covariance = (
            measurement_matrix @ covariance @ measurement_matrix.T
            + observation_covariance
        )
        gain = np.linalg.solve(
            innovation_covariance,
            measurement_matrix @ covariance,
        ).T
        updated = predicted + gain @ innovation
        updated[:2] = _wrap_angle(updated[:2])
        kh = gain @ measurement_matrix
        covariance = (
            (identity - kh) @ covariance @ (identity - kh).T
            + gain @ observation_covariance @ gain.T
        )
        covariance = 0.5 * (covariance + covariance.T)
        if not np.isfinite(updated).all() or not np.isfinite(covariance).all():
            raise FloatingPointError("roll_pitch_ekf_nonfinite_state")
        if float(np.min(np.linalg.eigvalsh(covariance))) < -1e-9:
            raise FloatingPointError("roll_pitch_ekf_covariance_not_psd")
        roll[index], pitch[index] = updated[0], updated[1]
        bias[index] = updated[2:]
    diagnostics = {
        "process_covariance_diagonal_per_second": list(
            config.process_covariance_diagonal_per_second
        ),
        "observation_covariance_diagonal_rad2": list(
            config.observation_covariance_diagonal_rad2
        ),
        "initial_covariance_diagonal": list(config.initial_covariance_diagonal),
        "dynamic_observation_scale": config.dynamic_observation_scale,
        "gravity_mps2": config.gravity_mps2,
        "observation_scale_equation": (
            "1+alpha_R*max(0,norm_acc-g)/g"
        ),
        "observed_measurement_scale_min": (
            observation_scale_min if np.isfinite(observation_scale_min) else 1.0
        ),
        "observed_measurement_scale_max": observation_scale_max,
        "final_residual_gyroscope_bias_rads": bias[-1].tolist(),
        "covariance_update": "joseph_form",
        "angle_innovation": "wrapped_to_minus_pi_pi",
    }
    return roll, pitch, bias, diagnostics


def _prepare_si_inputs(
    acceleration: np.ndarray,
    gyroscope: np.ndarray,
    *,
    fs_hz: float,
    acceleration_unit: str,
    gyroscope_unit: str,
    participant_id: str,
    calibration: MotionImuCalibration,
    config: RollPitchEkfConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if float(fs_hz) != CANONICAL_FS_HZ:
        raise ValueError("formal motion IMU preprocessing requires exactly 400 Hz")
    config.validate(fs_hz)
    calibration.validate()
    if str(participant_id) != calibration.participant_id:
        raise ValueError("cross-participant motion IMU calibration is forbidden")
    if calibration.config != config:
        raise ValueError("motion calibration and runtime EKF config differ")
    source_acc = np.asarray(acceleration, dtype=np.float64)
    source_gyro = np.asarray(gyroscope, dtype=np.float64)
    acc = _convert_profile_acceleration(
        source_acc,
        acceleration_unit,
        gravity_mps2=config.gravity_mps2,
    )
    gyro = convert_gyro(source_gyro, gyroscope_unit)
    if acc.ndim != 2 or gyro.shape != acc.shape or acc.shape[1] != 3 or acc.shape[0] < 16:
        raise ValueError("motion IMU input must be aligned samples-by-3")
    if not np.isfinite(acc).all() or not np.isfinite(gyro).all():
        raise ValueError("motion IMU input must be finite")
    acc = acc - np.asarray(calibration.acceleration_bias_mps2)
    gyro = gyro - np.asarray(calibration.gyroscope_bias_rads)
    acc_filtered = _zero_phase_lowpass(
        acc, config.accelerometer_lowpass_hz, fs_hz, config.sensor_filter_order
    )
    gyro_filtered = _zero_phase_lowpass(
        gyro, config.gyroscope_lowpass_hz, fs_hz, config.sensor_filter_order
    )
    return (
        acc_filtered,
        gyro_filtered,
        {
            "source_acceleration_sha256": _array_lineage_sha256(
                source_acc, f"source_acceleration:{acceleration_unit}"
            ),
            "source_gyroscope_sha256": _array_lineage_sha256(
                source_gyro, f"source_gyroscope:{gyroscope_unit}"
            ),
            "filtered_si_acceleration_sha256": _array_lineage_sha256(
                acc_filtered, "filtered_calibrated_acceleration_mps2"
            ),
            "filtered_si_gyroscope_sha256": _array_lineage_sha256(
                gyro_filtered, "filtered_calibrated_gyroscope_rads"
            ),
            "ekf_config_sha256": stable_payload_sha256(asdict(config)),
            "ekf_config": asdict(config),
        },
    )


def _motion_result(
    acc_filtered: np.ndarray,
    gyro_filtered: np.ndarray,
    gravity: np.ndarray,
    roll: np.ndarray,
    pitch: np.ndarray,
    *,
    fs_hz: float,
    profile_id: str,
    diagnostics: dict[str, Any],
) -> MotionImuResult:
    dynamic = acc_filtered - gravity
    jerk = np.diff(dynamic, axis=0, prepend=dynamic[:1]) * float(fs_hz)
    values = np.column_stack(
        (
            dynamic,
            gyro_filtered,
            np.linalg.norm(dynamic, axis=1),
            np.linalg.norm(gyro_filtered, axis=1),
            np.linalg.norm(jerk, axis=1),
        )
    )
    lineage = {
        "lineage_schema": MOTION_IMU_LINEAGE_SCHEMA,
        **diagnostics,
        "output_values_sha256": _array_lineage_sha256(
            values, "motion_imu_output_values"
        ),
        "gravity_sha256": _array_lineage_sha256(
            gravity, "motion_imu_gravity"
        ),
        "roll_pitch_sha256": _array_lineage_sha256(
            np.column_stack((roll, pitch)), "motion_imu_roll_pitch"
        ),
    }
    lineage_payload = {
        name: lineage[name]
        for name in sorted(
            {
                "source_acceleration_sha256",
                "source_gyroscope_sha256",
                "filtered_si_acceleration_sha256",
                "filtered_si_gyroscope_sha256",
                "ekf_config_sha256",
                "calibration_artifact_sha256",
                "output_values_sha256",
                "gravity_sha256",
                "roll_pitch_sha256",
            }
        )
    }
    lineage_payload["lineage_schema"] = MOTION_IMU_LINEAGE_SCHEMA
    lineage["lineage_sha256"] = stable_payload_sha256(lineage_payload)
    result = MotionImuResult(
        values=values,
        channel_schema=MOTION_IMU_CHANNEL_SCHEMA,
        channel_units=MOTION_IMU_CHANNEL_UNITS,
        roll_rad=roll,
        pitch_rad=pitch,
        gravity_mps2=gravity,
        valid_mask=np.isfinite(values).all(axis=1),
        profile_id=profile_id,
        diagnostics=lineage,
    )
    result.validate()
    return result


def preprocess_motion_imu_calibrated_ekf(
    acceleration: np.ndarray,
    gyroscope: np.ndarray,
    *,
    fs_hz: float,
    acceleration_unit: str,
    gyroscope_unit: str,
    participant_id: str,
    calibration: MotionImuCalibration,
    config: RollPitchEkfConfig,
) -> MotionImuResult:
    """Build A_dyn, Omega and J using the calibrated EKF reference only."""

    acc_filtered, gyro_filtered, input_lineage = _prepare_si_inputs(
        acceleration,
        gyroscope,
        fs_hz=fs_hz,
        acceleration_unit=acceleration_unit,
        gyroscope_unit=gyroscope_unit,
        participant_id=participant_id,
        calibration=calibration,
        config=config,
    )
    roll, pitch, _bias, ekf_diagnostics = _run_roll_pitch_ekf(
        acc_filtered,
        gyro_filtered,
        fs_hz=fs_hz,
        initial_roll_rad=calibration.initial_roll_rad,
        initial_pitch_rad=calibration.initial_pitch_rad,
        config=config,
    )
    gravity = _gravity_from_roll_pitch(
        roll,
        pitch,
        gravity_mps2=config.gravity_mps2,
    )
    diagnostics = {
        "profile_id": CALIBRATED_ROLL_PITCH_EKF_PROFILE_ID,
        "unit_conversion": {
            "acceleration": f"{acceleration_unit}->m/s^2",
            "gyroscope": f"{gyroscope_unit}->rad/s",
            "g_to_mps2_factor": config.gravity_mps2,
        },
        "calibration_artifact_sha256": calibration.artifact_sha256,
        "calibration_participant_id": calibration.participant_id,
        "calibration_file_id": calibration.file_id,
        "calibration_source_role": calibration.source_role,
        "runtime_participant_id": str(participant_id),
        "sensor_filters": {
            "phase": "zero_phase",
            "order": config.sensor_filter_order,
            "acceleration_lowpass_hz": config.accelerometer_lowpass_hz,
            "gyroscope_lowpass_hz": config.gyroscope_lowpass_hz,
        },
        "silent_fallback": False,
        "fallback_profile": None,
        "yaw_correction": "not_available_no_magnetometer_roll_pitch_only",
        "gravity_rotation": "R_x_roll_then_R_y_pitch_transpose_times_0_0_g",
        "profile_consistency_scope": (
            "artifact_reduction_imu_features_time_series_input"
        ),
        **input_lineage,
        **ekf_diagnostics,
    }
    return _motion_result(
        acc_filtered,
        gyro_filtered,
        gravity,
        roll,
        pitch,
        fs_hz=fs_hz,
        profile_id=CALIBRATED_ROLL_PITCH_EKF_PROFILE_ID,
        diagnostics=diagnostics,
    )


def preprocess_motion_imu_profile_a_lpf(
    acceleration: np.ndarray,
    gyroscope: np.ndarray,
    *,
    fs_hz: float,
    acceleration_unit: str,
    gyroscope_unit: str,
    participant_id: str,
    calibration: MotionImuCalibration,
    config: RollPitchEkfConfig,
) -> MotionImuResult:
    """Run the explicit Profile-A LPF; never use it as an EKF fallback."""

    acc_filtered, gyro_filtered, input_lineage = _prepare_si_inputs(
        acceleration,
        gyroscope,
        fs_hz=fs_hz,
        acceleration_unit=acceleration_unit,
        gyroscope_unit=gyroscope_unit,
        participant_id=participant_id,
        calibration=calibration,
        config=config,
    )
    gravity = _zero_phase_lowpass(
        acc_filtered,
        config.gravity_lowpass_hz,
        fs_hz,
        config.gravity_filter_order,
    )
    roll = np.empty(acc_filtered.shape[0], dtype=np.float64)
    pitch = np.empty(acc_filtered.shape[0], dtype=np.float64)
    for index, gravity_sample in enumerate(gravity):
        roll[index], pitch[index] = _roll_pitch_from_acc(gravity_sample)
    diagnostics = {
        "profile_id": PROFILE_A_LPF_ID,
        "unit_conversion": {
            "acceleration": f"{acceleration_unit}->m/s^2",
            "gyroscope": f"{gyroscope_unit}->rad/s",
            "g_to_mps2_factor": config.gravity_mps2,
        },
        "calibration_artifact_sha256": calibration.artifact_sha256,
        "calibration_source_role": calibration.source_role,
        "runtime_participant_id": str(participant_id),
        "sensor_filters": {
            "phase": "zero_phase",
            "order": config.sensor_filter_order,
            "acceleration_lowpass_hz": config.accelerometer_lowpass_hz,
            "gyroscope_lowpass_hz": config.gyroscope_lowpass_hz,
        },
        "gravity_lowpass_hz": config.gravity_lowpass_hz,
        "gravity_filter_order": config.gravity_filter_order,
        "phase": "zero_phase",
        "silent_fallback": False,
        "executed_as": "named_reference_profile",
        **input_lineage,
    }
    return _motion_result(
        acc_filtered,
        gyro_filtered,
        gravity,
        roll,
        pitch,
        fs_hz=fs_hz,
        profile_id=PROFILE_A_LPF_ID,
        diagnostics=diagnostics,
    )


# Backward-compatible import name. The runtime identity and persisted profile
# ID are unchanged; only its catalog role was changed from ablation to reference.
preprocess_motion_imu_lpf_ablation = preprocess_motion_imu_profile_a_lpf


__all__ = [
    "CALIBRATED_ROLL_PITCH_EKF_PROFILE_ID",
    "MOTION_IMU_CALIBRATION_SCHEMA",
    "MOTION_IMU_CHANNEL_SCHEMA",
    "MOTION_IMU_CHANNEL_UNITS",
    "FORMAL_STATIC_CALIBRATION_ROLES",
    "PROFILE_A_LPF_ID",
    "PTT_STATIC_CALIBRATION_ROLE",
    "MotionImuCalibration",
    "MotionImuResult",
    "RollPitchEkfConfig",
    "fit_motion_imu_calibration",
    "preprocess_motion_imu_calibrated_ekf",
    "preprocess_motion_imu_profile_a_lpf",
    "preprocess_motion_imu_lpf_ablation",
]
