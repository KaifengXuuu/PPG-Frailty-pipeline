"""ESKF 数学与共享多轴滤波 / ESKF mathematics and shared axis filtering.

中文：本模块实现 scalar-first body-to-navigation quaternion、六维右乘误差状态、
切平面 accelerometer 更新、Joseph covariance update 和无预校准状态机。

English: This module implements scalar-first body-to-navigation quaternions, a
six-dimensional right-multiplicative error state, tangent-plane accelerometer updates,
Joseph covariance updates, and the no-precalibration state machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal


STANDARD_GRAVITY_MPS2 = 9.80665


def convert_imu_to_si(
    acceleration: np.ndarray,
    gyroscope: np.ndarray,
    *,
    acceleration_unit: str,
    gyroscope_unit: str,
) -> tuple[np.ndarray, np.ndarray]:
    """只按显式 metadata 转换 SI / Convert to SI from explicit metadata only."""

    acc_factor = {
        "g": STANDARD_GRAVITY_MPS2,
        "m/s^2": 1.0,
        "m/s2": 1.0,
        "mg": STANDARD_GRAVITY_MPS2 / 1000.0,
    }
    gyro_factor = {"deg/s": np.pi / 180.0, "rad/s": 1.0}
    if acceleration_unit not in acc_factor:
        raise ValueError(f"unit_unknown: acceleration_unit={acceleration_unit}")
    if gyroscope_unit not in gyro_factor:
        raise ValueError(f"unit_unknown: gyroscope_unit={gyroscope_unit}")
    return (
        np.asarray(acceleration, dtype=np.float64) * acc_factor[acceleration_unit],
        np.asarray(gyroscope, dtype=np.float64) * gyro_factor[gyroscope_unit],
    )


def skew(vector: np.ndarray) -> np.ndarray:
    """构造 cross-product matrix / Build a cross-product matrix."""

    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def quat_normalize(quaternion: np.ndarray) -> np.ndarray:
    """归一化 quaternion / Normalize a scalar-first quaternion."""

    q = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-15:
        raise FloatingPointError("quaternion_norm_invalid")
    return q / norm


def quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton 乘积 / Hamilton product."""

    w1, x1, y1, z1 = np.asarray(left, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(right, dtype=np.float64)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_exp(rotation_vector: np.ndarray) -> np.ndarray:
    """旋转向量 exponential map / Rotation-vector exponential map."""

    vector = np.asarray(rotation_vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return quat_normalize(np.concatenate(([1.0], 0.5 * vector)))
    axis = vector / angle
    half = 0.5 * angle
    return np.concatenate(([np.cos(half)], axis * np.sin(half)))


def quat_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    """把 q_NB 转为 R_NB / Convert q_NB into body-to-navigation R_NB."""

    w, x, y, z = quat_normalize(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_from_two_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """稳定最短弧 quaternion / Stable shortest-arc quaternion."""

    source_unit = np.asarray(source, dtype=np.float64)
    target_unit = np.asarray(target, dtype=np.float64)
    source_unit /= np.linalg.norm(source_unit)
    target_unit /= np.linalg.norm(target_unit)
    dot = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))
    if dot < -1.0 + 1e-10:
        basis = np.eye(3)[int(np.argmin(np.abs(source_unit)))]
        axis = np.cross(source_unit, basis)
        axis /= np.linalg.norm(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])
    return quat_normalize(
        np.concatenate(([1.0 + dot], np.cross(source_unit, target_unit)))
    )


def tangent_basis(direction: np.ndarray) -> np.ndarray:
    """构造确定性二维切平面 / Build a deterministic 2-D tangent plane."""

    h = np.asarray(direction, dtype=np.float64)
    h /= np.linalg.norm(h)
    reference = np.eye(3)[int(np.argmin(np.abs(h)))]
    first = np.cross(h, reference)
    first /= np.linalg.norm(first)
    second = np.cross(h, first)
    second /= np.linalg.norm(second)
    return np.vstack([first, second])


def filter_axes(
    values: np.ndarray,
    fs_hz: float,
    cutoff_hz: float,
    *,
    order: int,
    phase_mode: str,
    initial_state: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """共享 SOS 低通并显式返回 causal state / Shared SOS filter with state."""

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("values must be non-empty samples×axes")
    sos = signal.butter(
        int(order), float(cutoff_hz), btype="lowpass", fs=float(fs_hz), output="sos"
    )
    if phase_mode == "offline_zero_phase":
        try:
            return np.asarray(signal.sosfiltfilt(sos, x, axis=0)), None
        except ValueError as exc:
            raise ValueError(f"insufficient_length:{exc}") from exc
    if phase_mode != "causal_stateful":
        raise ValueError(f"unsupported_phase_mode:{phase_mode}")
    state = initial_state
    if state is None:
        state = signal.sosfilt_zi(sos)[:, :, None] * x[0][None, None, :]
    filtered, final_state = signal.sosfilt(sos, x, axis=0, zi=state)
    return np.asarray(filtered), np.asarray(final_state)


@dataclass
class EskfConfiguration:
    """冻结的 ESKF 工程参数 / Frozen ESKF engineering parameters."""

    gyro_noise_density: float = 2e-3
    gyro_bias_random_walk: float = 2e-4
    base_accel_angle_noise_rad: float = np.deg2rad(5.0)
    nis_gate: float = 13.8155
    update_decimation: int = 4
    min_tracking_sec: float = 0.5
    min_accepted_updates: int = 20
    tracking_tilt_sigma_deg: float = 10.0
    max_prediction_only_sec: float = 2.0
    max_tracking_tilt_sigma_deg: float = 20.0
    max_bias_rad_s: float = 0.35


class NoPrecalibrationEskf:
    """无静态预校准 quaternion MEKF / Quaternion MEKF without precalibration."""

    def __init__(self, fs_hz: float, config: EskfConfiguration | None = None) -> None:
        """创建 initialization_pending 状态 / Create pending online state."""

        self.fs_hz = float(fs_hz)
        self.dt = 1.0 / self.fs_hz
        self.config = config or EskfConfiguration()
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.bias = np.zeros(3)
        self.p = np.eye(6)
        self.initialized = False
        self.tracking_reached = False
        self.samples_since_initialization = 0
        self.accepted_updates = 0
        self.samples_since_update = 0
        self.last_acc: np.ndarray | None = None
        self.no_estimate_latched = False
        self.covariance_floor_used = False

    def _initialize(self, acceleration: np.ndarray) -> None:
        """从首个可接受向量初始化 tilt、yaw=0 / Initialize tilt online."""

        direction = acceleration / np.linalg.norm(acceleration)
        self.q = quat_from_two_vectors(direction, np.array([0.0, 0.0, 1.0]))
        tilt = np.deg2rad(20.0)
        yaw = np.deg2rad(180.0)
        bias = np.deg2rad(5.0)
        self.p = np.zeros((6, 6))
        self.p[:3, :3] = (
            tilt**2 * (np.eye(3) - np.outer(direction, direction))
            + yaw**2 * np.outer(direction, direction)
        )
        self.p[3:, 3:] = np.eye(3) * bias**2
        self.bias[:] = 0.0
        self.initialized = True
        self.last_acc = acceleration.copy()
        self.tracking_reached = False
        self.samples_since_initialization = 0
        self.accepted_updates = 0
        self.samples_since_update = 0
        self.no_estimate_latched = False

    def _tilt_sigma_deg(self) -> float:
        """计算重力切平面 tilt uncertainty / Compute tangent tilt uncertainty."""

        h = quat_to_rotation(self.q).T @ np.array([0.0, 0.0, 1.0])
        tangent = tangent_basis(h)
        covariance = tangent @ self.p[:3, :3] @ tangent.T
        return float(np.rad2deg(np.sqrt(max(np.max(np.linalg.eigvalsh(covariance)), 0.0))))

    def step(
        self, acceleration: np.ndarray, gyroscope: np.ndarray, sample_index: int
    ) -> dict[str, Any]:
        """预测并按门控更新一个样本 / Predict and conditionally update one sample."""

        acc = np.asarray(acceleration, dtype=np.float64)
        gyro = np.asarray(gyroscope, dtype=np.float64)
        if acc.shape != (3,) or gyro.shape != (3,) or not np.isfinite(acc).all() or not np.isfinite(gyro).all():
            return {
                "state": "invalid",
                "valid": False,
                "gravity": np.full(3, np.nan),
                "reason": "nonfinite_or_invalid_imu_sample",
            }
        if self.no_estimate_latched:
            return {
                "state": "no_estimate",
                "valid": False,
                "gravity": np.full(3, np.nan),
                "reason": "explicit_session_reset_required",
            }
        acc_norm = float(np.linalg.norm(acc))
        initialized_now = False
        if not self.initialized:
            if 0.5 * STANDARD_GRAVITY_MPS2 <= acc_norm <= 1.5 * STANDARD_GRAVITY_MPS2:
                self._initialize(acc)
                initialized_now = True
            else:
                return {
                    "state": "initialization_pending",
                    "valid": False,
                    "gravity": np.full(3, np.nan),
                }

        rate = gyro - self.bias
        if not initialized_now:
            self.q = quat_normalize(quat_multiply(self.q, quat_exp(rate * self.dt)))
            f_matrix = np.zeros((6, 6))
            f_matrix[:3, :3] = -skew(rate)
            f_matrix[:3, 3:] = -np.eye(3)
            f_dt = f_matrix * self.dt
            transition = np.eye(6) + f_dt + 0.5 * (f_dt @ f_dt)
            # 中文：bias random walk 对 attitude 的同一步积分与交叉协方差不可省略。
            # English: Include within-step bias-walk attitude and cross covariance.
            gyro_variance = self.config.gyro_noise_density**2
            bias_variance = self.config.gyro_bias_random_walk**2
            process = np.zeros((6, 6))
            process[:3, :3] = np.eye(3) * (
                gyro_variance * self.dt + bias_variance * self.dt**3 / 3.0
            )
            process[:3, 3:] = -np.eye(3) * bias_variance * self.dt**2 / 2.0
            process[3:, :3] = process[:3, 3:].T
            process[3:, 3:] = np.eye(3) * bias_variance * self.dt
            self.p = transition @ self.p @ transition.T + process
        self.samples_since_initialization += 1
        self.samples_since_update += 1

        accepted = False
        nis = np.nan
        downweighted = False
        if sample_index % self.config.update_decimation == 0:
            predicted = quat_to_rotation(self.q).T @ np.array([0.0, 0.0, 1.0])
            observed = acc / max(acc_norm, 1e-15)
            rho = abs(acc_norm / STANDARD_GRAVITY_MPS2 - 1.0)
            eta = (
                0.0
                if self.last_acc is None
                else float(
                    np.linalg.norm(acc - self.last_acc)
                    / (STANDARD_GRAVITY_MPS2 * self.dt)
                )
            )
            scale_r = float(
                np.clip(1.0 + (rho / 0.05) ** 2 + (eta / 2.0) ** 2, 1.0, 100.0)
            )
            downweighted = scale_r >= 25.0
            tangent = tangent_basis(predicted)
            residual = tangent @ (observed - predicted)
            h_matrix = np.zeros((2, 6))
            h_matrix[:, :3] = tangent @ skew(predicted)
            measurement = (
                self.config.base_accel_angle_noise_rad**2 * scale_r * np.eye(2)
            )
            innovation_covariance = h_matrix @ self.p @ h_matrix.T + measurement
            nis = float(residual @ np.linalg.solve(innovation_covariance, residual))
            if (
                0.5 * STANDARD_GRAVITY_MPS2 <= acc_norm <= 1.5 * STANDARD_GRAVITY_MPS2
                and nis <= self.config.nis_gate
            ):
                gain = np.linalg.solve(innovation_covariance, h_matrix @ self.p).T
                correction = gain @ residual
                self.q = quat_normalize(
                    quat_multiply(self.q, quat_exp(correction[:3]))
                )
                self.bias += correction[3:]
                identity = np.eye(6)
                kh = gain @ h_matrix
                self.p = (
                    (identity - kh) @ self.p @ (identity - kh).T
                    + gain @ measurement @ gain.T
                )
                reset = np.eye(6)
                reset[:3, :3] -= 0.5 * skew(correction[:3])
                self.p = reset @ self.p @ reset.T
                accepted = True
                self.accepted_updates += 1
                self.samples_since_update = 0
        self.last_acc = acc.copy()
        self.p = 0.5 * (self.p + self.p.T)
        eigenvalues, eigenvectors = np.linalg.eigh(self.p)
        if float(np.min(eigenvalues)) < -1e-10:
            raise FloatingPointError("covariance_not_positive_semidefinite")
        if float(np.min(eigenvalues)) < 1e-15:
            self.p = eigenvectors @ np.diag(np.maximum(eigenvalues, 1e-15)) @ eigenvectors.T
            self.covariance_floor_used = True

        tilt_sigma = self._tilt_sigma_deg()
        ready = (
            self.samples_since_initialization / self.fs_hz >= self.config.min_tracking_sec
            and self.accepted_updates >= self.config.min_accepted_updates
            and tilt_sigma <= self.config.tracking_tilt_sigma_deg
        )
        self.tracking_reached = self.tracking_reached or ready
        prediction_sec = self.samples_since_update / self.fs_hz
        divergent = (
            not np.isfinite(self.q).all()
            or not np.isfinite(self.p).all()
            or np.any(np.abs(self.bias) > self.config.max_bias_rad_s)
        )
        if divergent:
            state, valid = "no_estimate", False
        elif not self.tracking_reached:
            state, valid = "initialization_pending", False
        elif (
            prediction_sec > self.config.max_prediction_only_sec
            or tilt_sigma > self.config.max_tracking_tilt_sigma_deg
        ):
            state, valid = "no_estimate", False
        elif not accepted and self.samples_since_update > self.config.update_decimation:
            state, valid = "prediction_only", True
        else:
            state, valid = "tracking", True
        if state == "no_estimate":
            # 中文：终止状态锁存；只有显式新建/重置 session 才能恢复。
            # English: Latch terminal failure until an explicit session reset.
            self.no_estimate_latched = True
        gravity = quat_to_rotation(self.q).T @ np.array(
            [0.0, 0.0, STANDARD_GRAVITY_MPS2]
        )
        return {
            "state": state,
            "valid": valid,
            "gravity": gravity,
            "quaternion": self.q.copy(),
            "bias": self.bias.copy(),
            "nis": nis,
            "accepted": accepted,
            "downweighted": downweighted,
            "tilt_sigma_deg": tilt_sigma,
            "covariance_min_eigenvalue": float(np.min(np.linalg.eigvalsh(self.p))),
            "covariance_floor_used": self.covariance_floor_used,
            "physical_observability_status": "unverified_no_static_precalibration",
            "accelerometer_bias_status": "not_estimated",
        }


def run_ekf(
    acceleration: np.ndarray,
    gyroscope: np.ndarray,
    fs_hz: float,
    *,
    config: EskfConfiguration | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """运行整段 ESKF 并保存逐样本诊断 / Run ESKF with sample diagnostics."""

    estimator = NoPrecalibrationEskf(fs_hz, config=config)
    count = acceleration.shape[0]
    gravity = np.full((count, 3), np.nan)
    valid = np.zeros(count, dtype=bool)
    quaternions = np.full((count, 4), np.nan)
    biases = np.full((count, 3), np.nan)
    nis = np.full(count, np.nan)
    tilt = np.full(count, np.nan)
    accepted = np.zeros(count, dtype=bool)
    downweighted = np.zeros(count, dtype=bool)
    states: list[str] = []
    for index in range(count):
        result = estimator.step(acceleration[index], gyroscope[index], index)
        states.append(str(result["state"]))
        valid[index] = bool(result["valid"])
        gravity[index] = result["gravity"]
        if "quaternion" in result:
            quaternions[index] = result["quaternion"]
            biases[index] = result["bias"]
            nis[index] = result["nis"]
            tilt[index] = result["tilt_sigma_deg"]
            accepted[index] = result["accepted"]
            downweighted[index] = result["downweighted"]
    return gravity, valid, {
        "state_per_sample": states,
        "quaternion_wxyz": quaternions,
        "gyro_bias_rad_s": biases,
        "nis": nis,
        "tilt_sigma_deg": tilt,
        "accepted_gravity_update": accepted,
        "dynamic_accel_downweighted": downweighted,
        "no_static_precalibration": True,
        "bias_observability": "partial_full_unverified",
        "yaw_reference": "unobservable_relative_only",
    }
