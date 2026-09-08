"""Stateful M3 quaternion-MEKF and causal LPF IMU preprocessing."""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal

from .views import CANONICAL_FS_HZ

STANDARD_GRAVITY = 9.80665
M3_IMU_SOURCE_SHA256 = "296e24b76efee4f3417a2932d818c2eea12020658b59b2bef36cdeab2421f787"
M3_IMU_MATH_SOURCE_SHA256 = "e5e1e4690ab04a184ee9838b856e6495a46f894b0d28b747d4f5e868ab8e9da9"
M3_IMU_RUNTIME_SOURCE_SHA256 = "f4d214dbe0a1b52fab9c98765de7cd13ee1a920518961332713522881ead3f95"

@dataclass(frozen=True)
class EskfConfiguration:
    """冻结 MEKF 工程参数 / Frozen MEKF engineering parameters."""
    gyro_noise_density: float = 2e-3
    gyro_bias_random_walk: float = 2e-4
    base_accel_angle_noise_rad: float = float(np.deg2rad(5.0))
    nis_gate: float = 13.8155
    update_decimation: int = 4
    min_tracking_sec: float = 0.5
    min_accepted_updates: int = 20
    tracking_tilt_sigma_deg: float = 10.0
    max_prediction_only_sec: float = 2.0
    max_tracking_tilt_sigma_deg: float = 20.0
    max_bias_rad_s: float = 0.35

@dataclass(frozen=True)
class ImuProfile:
    """显式、版本化 runtime profile / Explicit versioned runtime profile."""
    gravity_method: str
    acceleration_lowpass_hz: float = 20.0
    gyroscope_lowpass_hz: float = 40.0
    sensor_filter_order: int = 3
    gravity_lowpass_hz: float = 0.3
    gravity_filter_order: int = 2
    eskf: EskfConfiguration = EskfConfiguration()

    def validate(self, fs_hz: float = CANONICAL_FS_HZ) -> None:
        """拒绝未实现算法及越界参数 / Reject unimplemented methods and bad ranges."""
        if self.gravity_method not in {"no_precalibration_ekf", "lpf_0p3"}:
            raise ValueError("gravity_method must be no_precalibration_ekf or lpf_0p3")
        if not np.isfinite(fs_hz) or fs_hz <= 0.0:
            raise ValueError("IMU sampling frequency must be finite and positive")
        for name, order in (
            ("sensor_filter_order", self.sensor_filter_order),
            ("gravity_filter_order", self.gravity_filter_order),
        ):
            if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or not 1 <= int(order) <= 20:
                raise ValueError(f"{name} must be an integer in [1,20]")
        for name, cutoff_hz in (
            ("acceleration_lowpass_hz", self.acceleration_lowpass_hz),
            ("gyroscope_lowpass_hz", self.gyroscope_lowpass_hz),
            ("gravity_lowpass_hz", self.gravity_lowpass_hz),
        ):
            if not np.isfinite(cutoff_hz) or not 0.0 < float(cutoff_hz) < fs_hz / 2.0:
                raise ValueError(f"{name} must be finite and within Nyquist")

@dataclass(frozen=True)
class ImuPreprocessResult:
    """同步结果、掩码与诊断 / Synchronized result, masks, and diagnostics."""
    processed: dict[str, np.ndarray]
    status: str
    reasons: tuple[str, ...]
    gravity_method: str
    diagnostics: dict[str, Any]
    valid_mask: np.ndarray

def convert_acceleration(values: np.ndarray, unit: str) -> np.ndarray:
    """仅按显式 metadata 转换 ACC / Convert ACC only from explicit metadata."""
    factors = {
        "g": STANDARD_GRAVITY,
        "m/s2": 1.0,
        "m/s^2": 1.0,
        "mg": STANDARD_GRAVITY / 1000.0,
    }
    normalized = unit.strip().lower().replace("²", "2")
    if normalized not in factors:
        raise ValueError(f"unit_unknown:acceleration_unit={unit}")
    return np.asarray(values, dtype=np.float64) * factors[normalized]

def convert_gyro(values: np.ndarray, unit: str) -> np.ndarray:
    """仅按显式 metadata 转换 gyro / Convert gyro only from explicit metadata."""
    factors = {"deg/s": np.pi / 180.0, "rad/s": 1.0}
    normalized = unit.strip().lower()
    if normalized not in factors:
        raise ValueError(f"unit_unknown:gyroscope_unit={unit}")
    return np.asarray(values, dtype=np.float64) * factors[normalized]

def skew(vector: np.ndarray) -> np.ndarray:
    """构造叉乘矩阵 / Build a cross-product matrix."""
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)

def quat_normalize(quaternion: np.ndarray) -> np.ndarray:
    """归一化 scalar-first quaternion / Normalize a scalar-first quaternion."""
    value = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1e-15:
        raise FloatingPointError("quaternion_norm_invalid")
    return value / norm

def quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton quaternion product / Hamilton 四元数乘积。"""
    w1, x1, y1, z1 = np.asarray(left, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(right, dtype=np.float64)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
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
    """把 q_NB 转为 R_NB / Convert q_NB to body-to-navigation R_NB."""
    w, x, y, z = quat_normalize(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
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
        return np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float64)
    return quat_normalize(np.concatenate(([1.0 + dot], np.cross(source_unit, target_unit))))

def tangent_basis(direction: np.ndarray) -> np.ndarray:
    """构造确定性二维切平面 / Build a deterministic two-dimensional tangent plane."""
    value = np.asarray(direction, dtype=np.float64)
    value /= np.linalg.norm(value)
    reference = np.eye(3)[int(np.argmin(np.abs(value)))]
    first = np.cross(value, reference)
    first /= np.linalg.norm(first)
    second = np.cross(value, first)
    second /= np.linalg.norm(second)
    return np.vstack((first, second))

def _causal_filter_axes(
    values: np.ndarray,
    fs_hz: float,
    cutoff_hz: float,
    *,
    order: int,
    initial_state: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """共享 causal SOS 并返回持久 state / Shared causal SOS with persistent state."""
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 2 or source.shape[0] == 0:
        raise ValueError("filter input must be non-empty samples-by-axes")
    if not np.isfinite([fs_hz, cutoff_hz]).all() or fs_hz <= 0.0 or not 0.0 < cutoff_hz < fs_hz / 2.0:
        raise ValueError("filter cutoff must be finite and within Nyquist")
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or not 1 <= int(order) <= 20:
        raise ValueError("filter order must be an integer in [1,20]")
    sos = signal.butter(order, cutoff_hz, btype="lowpass", fs=fs_hz, output="sos")
    state = initial_state
    if state is None:
        state = signal.sosfilt_zi(sos)[:, :, None] * source[0][None, None, :]
    filtered, final_state = signal.sosfilt(sos, source, axis=0, zi=state)
    return np.asarray(filtered, dtype=np.float64), np.asarray(final_state, dtype=np.float64)

class NoPrecalibrationEskf:
    """无静态预校准 quaternion MEKF / Quaternion MEKF without precalibration."""
    def __init__(self, fs_hz: float, config: EskfConfiguration | None = None) -> None:
        """创建 initialization_pending 状态 / Create initialization-pending state."""
        self.fs_hz = float(fs_hz)
        self.dt = 1.0 / self.fs_hz
        self.config = config or EskfConfiguration()
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.bias = np.zeros(3, dtype=np.float64)
        self.p = np.eye(6, dtype=np.float64)
        self.initialized = False
        self.tracking_reached = False
        self.samples_since_initialization = 0
        self.accepted_updates = 0
        self.samples_since_update = 0
        self.last_acc: np.ndarray | None = None
        self.no_estimate_latched = False
        self.covariance_floor_used = False

    def _initialize(self, acceleration: np.ndarray) -> None:
        """首个合格 ACC 初始化 tilt、yaw=0 / Initialize tilt online with yaw zero."""
        direction = acceleration / np.linalg.norm(acceleration)
        self.q = quat_from_two_vectors(direction, np.array([0.0, 0.0, 1.0]))
        tilt, yaw, bias = np.deg2rad(20.0), np.deg2rad(180.0), np.deg2rad(5.0)
        self.p = np.zeros((6, 6), dtype=np.float64)
        self.p[:3, :3] = tilt**2 * (np.eye(3) - np.outer(direction, direction)) + yaw**2 * np.outer(
            direction, direction
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

    def reset_for_new_session(self) -> None:
        """显式恢复 pending；终止 latch 不能自动清除 / Explicitly reset the latch."""
        self.__init__(self.fs_hz, self.config)

    def _tilt_sigma_deg(self) -> float:
        """计算切平面 tilt uncertainty / Compute tangent-plane tilt uncertainty."""
        predicted = quat_to_rotation(self.q).T @ np.array([0.0, 0.0, 1.0])
        tangent = tangent_basis(predicted)
        covariance = tangent @ self.p[:3, :3] @ tangent.T
        return float(np.rad2deg(np.sqrt(max(np.max(np.linalg.eigvalsh(covariance)), 0.0))))

    def step(self, acceleration: np.ndarray, gyroscope: np.ndarray, sample_index: int) -> dict[str, Any]:
        """预测并按自适应 NIS 门控更新一个样本 / Predict and gate one update."""
        acc = np.asarray(acceleration, dtype=np.float64)
        gyro = np.asarray(gyroscope, dtype=np.float64)
        invalid = acc.shape != (3,) or gyro.shape != (3,) or not np.isfinite(acc).all() or not np.isfinite(gyro).all()
        if invalid:
            return {
                "state": "invalid",
                "valid": False,
                "gravity": np.full(3, np.nan),
                "reason": "invalid_imu_sample",
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
            if 0.5 * STANDARD_GRAVITY <= acc_norm <= 1.5 * STANDARD_GRAVITY:
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
            f_matrix = np.zeros((6, 6), dtype=np.float64)
            f_matrix[:3, :3] = -skew(rate)
            f_matrix[:3, 3:] = -np.eye(3)
            f_dt = f_matrix * self.dt
            transition = np.eye(6) + f_dt + 0.5 * (f_dt @ f_dt)
            gyro_variance = self.config.gyro_noise_density**2
            bias_variance = self.config.gyro_bias_random_walk**2
            process = np.zeros((6, 6), dtype=np.float64)
            # 中文：bias random walk 的 attitude 积分和交叉协方差不可省略。
            # English: Include bias-walk attitude integration and cross covariance.
            process[:3, :3] = np.eye(3) * (gyro_variance * self.dt + bias_variance * self.dt**3 / 3.0)
            process[:3, 3:] = -np.eye(3) * bias_variance * self.dt**2 / 2.0
            process[3:, :3] = process[:3, 3:].T
            process[3:, 3:] = np.eye(3) * bias_variance * self.dt
            self.p = transition @ self.p @ transition.T + process
        self.samples_since_initialization += 1
        self.samples_since_update += 1

        accepted, downweighted, nis = False, False, float("nan")
        if sample_index % self.config.update_decimation == 0:
            predicted = quat_to_rotation(self.q).T @ np.array([0.0, 0.0, 1.0])
            observed = acc / max(acc_norm, 1e-15)
            rho = abs(acc_norm / STANDARD_GRAVITY - 1.0)
            eta = (
                0.0
                if self.last_acc is None
                else float(np.linalg.norm(acc - self.last_acc) / (STANDARD_GRAVITY * self.dt))
            )
            scale_r = float(
                np.clip(
                    1.0 + (rho / 0.05) ** 2 + (eta / 2.0) ** 2,
                    1.0,
                    100.0,
                )
            )
            downweighted = scale_r >= 25.0
            tangent = tangent_basis(predicted)
            residual = tangent @ (observed - predicted)
            h_matrix = np.zeros((2, 6), dtype=np.float64)
            h_matrix[:, :3] = tangent @ skew(predicted)
            measurement = self.config.base_accel_angle_noise_rad**2 * scale_r * np.eye(2)
            innovation_covariance = h_matrix @ self.p @ h_matrix.T + measurement
            nis = float(residual @ np.linalg.solve(innovation_covariance, residual))
            if 0.5 * STANDARD_GRAVITY <= acc_norm <= 1.5 * STANDARD_GRAVITY and nis <= self.config.nis_gate:
                gain = np.linalg.solve(innovation_covariance, h_matrix @ self.p).T
                correction = gain @ residual
                self.q = quat_normalize(quat_multiply(self.q, quat_exp(correction[:3])))
                self.bias += correction[3:]
                identity = np.eye(6)
                kh = gain @ h_matrix
                self.p = (identity - kh) @ self.p @ (identity - kh).T + gain @ measurement @ gain.T
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
            prediction_sec > self.config.max_prediction_only_sec or tilt_sigma > self.config.max_tracking_tilt_sigma_deg
        ):
            state, valid = "no_estimate", False
        elif not accepted and self.samples_since_update > self.config.update_decimation:
            state, valid = "prediction_only", True
        else:
            state, valid = "tracking", True
        if state == "no_estimate":
            # 中文：终止状态锁存，只有显式 session reset 可以恢复。
            # English: Terminal failure latches until an explicit session reset.
            self.no_estimate_latched = True
        gravity = quat_to_rotation(self.q).T @ np.array([0.0, 0.0, STANDARD_GRAVITY])
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

def _timestamp_ok(
    timestamps_s: np.ndarray | None,
    n_samples: int,
    fs_hz: float,
    previous_timestamp_s: float | None,
) -> tuple[bool, str | None]:
    """验证 chunk 内及边界时间网格 / Validate chunk and boundary time grids."""
    if timestamps_s is None:
        return True, None
    time = np.asarray(timestamps_s, dtype=np.float64).ravel()
    if time.size != n_samples or not np.isfinite(time).all():
        return False, "timestamp_length_or_nonfinite"
    expected = 1.0 / fs_hz
    if time.size > 1:
        intervals = np.diff(time)
        relative = np.abs(intervals - expected) / expected
        if np.any(intervals <= 0.0) or float(np.percentile(relative, 99)) > 0.05:
            return False, "timestamp_order_or_jitter"
    if previous_timestamp_s is not None and time.size:
        boundary = time[0] - previous_timestamp_s
        if boundary <= 0.0 or abs(boundary - expected) / expected > 0.05:
            return False, "timestamp_chunk_boundary"
    return True, None

def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """返回布尔真值右开区间 / Return right-open runs of true values."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    changes = np.diff(padded.astype(np.int8))
    return list(
        zip(
            np.flatnonzero(changes == 1).tolist(),
            np.flatnonzero(changes == -1).tolist(),
        )
    )

def _repair_chunk(values: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """只修复 <=0.25 s 内部短缺口 / Repair only bounded internal short gaps."""
    source = np.asarray(values, dtype=np.float64)
    repaired = source.copy()
    repair_mask = np.zeros_like(source, dtype=bool)
    reasons: list[str] = []
    maximum_gap = int(round(0.25 * fs_hz))
    for column in range(source.shape[1]):
        finite = np.isfinite(source[:, column])
        if not finite.any() or np.count_nonzero(~finite) / finite.size > 0.01:
            raise ValueError(f"channel_{column}:excessive_nonfinite")
        for start, stop in _true_runs(~finite):
            length = stop - start
            if start == 0 or stop == source.shape[0] or length > maximum_gap:
                raise ValueError(f"channel_{column}:unrepairable_gap:{length}")
            repaired[start:stop, column] = np.linspace(
                repaired[start - 1, column],
                repaired[stop, column],
                length + 2,
            )[1:-1]
            repair_mask[start:stop, column] = True
            reasons.append(f"channel_{column}:short_gap_interpolated:{length}")
    return repaired, repair_mask, tuple(reasons)

def _jerk(
    dynamic: np.ndarray,
    route_valid: np.ndarray,
    fs_hz: float,
    previous_dynamic: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """跨 chunk 连续 jerk 与公共 mask / Continuous jerk and common mask."""
    jerk = np.full_like(dynamic, np.nan)
    if dynamic.shape[0] == 0:
        return jerk, np.zeros(0, dtype=bool)
    if previous_dynamic is not None and route_valid[0] and np.isfinite(previous_dynamic).all():
        jerk[0] = (dynamic[0] - previous_dynamic) * fs_hz
    if dynamic.shape[0] > 1:
        adjacent = route_valid[1:] & route_valid[:-1]
        differences = np.diff(dynamic, axis=0) * fs_hz
        jerk[1:][adjacent] = differences[adjacent]
    valid = route_valid & np.isfinite(dynamic).all(axis=1) & np.isfinite(jerk).all(axis=1)
    return jerk, valid

class CausalImuProcessor:
    """持久 causal IMU processor / Persistent causal IMU processor."""
    def __init__(
        self,
        *,
        fs_hz: float,
        acceleration_unit: str,
        gyroscope_unit: str,
        profile: ImuProfile,
    ) -> None:
        """绑定所有显式参数并创建空 state / Bind explicit profile and state."""
        if float(fs_hz) != CANONICAL_FS_HZ:
            raise ValueError("V1 causal IMU processor requires exactly 400 Hz")
        profile.validate(float(fs_hz))
        # 中文：构造时先校验单位，禁止首 chunk 后才发现 metadata 错误。
        # English: Validate unit names before a session mutates any state.
        convert_acceleration(np.zeros((1, 3)), acceleration_unit)
        convert_gyro(np.zeros((1, 3)), gyroscope_unit)
        self.fs_hz = float(fs_hz)
        self.acceleration_unit = acceleration_unit
        self.gyroscope_unit = gyroscope_unit
        self.profile = profile
        self.acc_filter_state: np.ndarray | None = None
        self.gyro_filter_state: np.ndarray | None = None
        self.gravity_filter_state: np.ndarray | None = None
        self.last_dynamic: np.ndarray | None = None
        self.previous_timestamp_s: float | None = None
        self.global_sample_index = 0
        self.estimator = (
            NoPrecalibrationEskf(self.fs_hz, profile.eskf)
            if profile.gravity_method == "no_precalibration_ekf"
            else None
        )

    def reset_for_new_session(self) -> None:
        """显式清除滤波、MEKF、jerk 和时间状态 / Reset complete state."""
        self.__init__(
            fs_hz=self.fs_hz,
            acceleration_unit=self.acceleration_unit,
            gyroscope_unit=self.gyroscope_unit,
            profile=self.profile,
        )

    def process_chunk(
        self,
        acceleration: np.ndarray,
        gyroscope: np.ndarray,
        *,
        timestamps_s: np.ndarray | None = None,
    ) -> ImuPreprocessResult:
        """处理连续 chunk；失败不提交 state / Process one transactional chunk."""
        acc_raw = np.asarray(acceleration, dtype=np.float64)
        gyro_raw = np.asarray(gyroscope, dtype=np.float64)
        if (
            acc_raw.ndim != 2
            or gyro_raw.ndim != 2
            or acc_raw.shape != gyro_raw.shape
            or acc_raw.shape[1] != 3
            or acc_raw.shape[0] == 0
        ):
            count = acc_raw.shape[0] if acc_raw.ndim == 2 else 0
            return ImuPreprocessResult(
                {},
                "failed",
                ("invalid_imu_shape",),
                self.profile.gravity_method,
                {"silent_fallback": False},
                np.zeros(count, dtype=bool),
            )
        time_ok, time_reason = _timestamp_ok(
            timestamps_s,
            acc_raw.shape[0],
            self.fs_hz,
            self.previous_timestamp_s,
        )
        if not time_ok:
            return ImuPreprocessResult(
                {},
                "failed",
                (str(time_reason),),
                self.profile.gravity_method,
                {"silent_fallback": False},
                np.zeros(acc_raw.shape[0], dtype=bool),
            )
        try:
            repaired, repair_mask, repair_reasons = _repair_chunk(np.column_stack((acc_raw, gyro_raw)), self.fs_hz)
            acc_si = convert_acceleration(repaired[:, :3], self.acceleration_unit)
            gyro_si = convert_gyro(repaired[:, 3:], self.gyroscope_unit)
            acc_filtered, next_acc_state = _causal_filter_axes(
                acc_si,
                self.fs_hz,
                self.profile.acceleration_lowpass_hz,
                order=self.profile.sensor_filter_order,
                initial_state=self.acc_filter_state,
            )
            gyro_filtered, next_gyro_state = _causal_filter_axes(
                gyro_si,
                self.fs_hz,
                self.profile.gyroscope_lowpass_hz,
                order=self.profile.sensor_filter_order,
                initial_state=self.gyro_filter_state,
            )
            count = acc_raw.shape[0]
            state_names: list[str] = []
            route_diagnostics: dict[str, Any] = {}
            next_gravity_state = self.gravity_filter_state
            # 中文：在副本上推进；任何异常都不污染正式 session。
            # English: Advance a copy so exceptions cannot contaminate the session.
            next_estimator = copy.deepcopy(self.estimator)
            if self.profile.gravity_method == "no_precalibration_ekf":
                assert next_estimator is not None
                gravity = np.full((count, 3), np.nan)
                route_valid = np.zeros(count, dtype=bool)
                quaternions = np.full((count, 4), np.nan)
                biases = np.full((count, 3), np.nan)
                tilt = np.full(count, np.nan)
                nis = np.full(count, np.nan)
                accepted = np.zeros(count, dtype=bool)
                downweighted = np.zeros(count, dtype=bool)
                covariance_min = np.full(count, np.nan)
                for local_index in range(count):
                    sample = next_estimator.step(
                        acc_filtered[local_index],
                        gyro_filtered[local_index],
                        self.global_sample_index + local_index,
                    )
                    state_names.append(str(sample["state"]))
                    route_valid[local_index] = bool(sample["valid"])
                    gravity[local_index] = sample["gravity"]
                    if "quaternion" in sample:
                        quaternions[local_index] = sample["quaternion"]
                        biases[local_index] = sample["bias"]
                        tilt[local_index] = sample["tilt_sigma_deg"]
                        nis[local_index] = sample["nis"]
                        accepted[local_index] = sample["accepted"]
                        downweighted[local_index] = sample["downweighted"]
                        covariance_min[local_index] = sample["covariance_min_eigenvalue"]
                confidence = np.where(
                    route_valid & np.isfinite(tilt),
                    np.exp(-tilt / 10.0),
                    0.0,
                )
                route_diagnostics.update(
                    {
                        "quaternion_wxyz": quaternions,
                        "gyro_bias_rad_s": biases,
                        "statistical_tilt_sigma_deg": tilt,
                        "nis": nis,
                        "accepted_gravity_update": accepted,
                        "dynamic_accel_downweighted": downweighted,
                        "covariance_min_eigenvalue": covariance_min,
                        "no_static_precalibration": True,
                        "physical_observability_status": ("unverified_no_static_precalibration"),
                        "bias_observability": "partial_full_unverified",
                        "yaw_reference": "unobservable_relative_only",
                        "accelerometer_bias_status": "not_estimated",
                    }
                )
            else:
                gravity, next_gravity_state = _causal_filter_axes(
                    acc_filtered,
                    self.fs_hz,
                    self.profile.gravity_lowpass_hz,
                    order=self.profile.gravity_filter_order,
                    initial_state=self.gravity_filter_state,
                )
                route_valid = np.isfinite(gravity).all(axis=1)
                state_names = ["tracking" if valid else "no_estimate" for valid in route_valid]
                confidence = route_valid.astype(np.float64)

            dynamic = acc_filtered - gravity
            dynamic[~route_valid] = np.nan
            jerk, common_valid = _jerk(dynamic, route_valid, self.fs_hz, self.last_dynamic)
            terminal_state = state_names[-1] if state_names else "initialization_pending"
            if terminal_state == "no_estimate":
                status = "no_estimate"
            elif not np.any(common_valid):
                status = "initialization_pending"
            elif not np.all(common_valid):
                status = "partial"
            else:
                status = "success"
            processed = {
                "acc_mps2": acc_filtered,
                "gyro_rads": gyro_filtered,
                "gravity_mps2": gravity,
                "dynamic_acc_mps2": dynamic,
                "acc_magnitude": np.linalg.norm(acc_filtered, axis=1),
                "dynamic_magnitude": np.linalg.norm(dynamic, axis=1),
                "gyro_magnitude": np.linalg.norm(gyro_filtered, axis=1),
                "jerk_mps3": jerk,
                "jerk_magnitude": np.linalg.norm(jerk, axis=1),
                "imu_valid_mask": common_valid,
                "gravity_valid_mask": route_valid,
                "gravity_confidence": confidence,
                "repair_mask": repair_mask,
            }
            diagnostics = {
                "gravity_method": self.profile.gravity_method,
                "phase_mode": "causal_stateful",
                "acceleration_lowpass_hz": (self.profile.acceleration_lowpass_hz),
                "gyroscope_lowpass_hz": (self.profile.gyroscope_lowpass_hz),
                "sensor_filter_order": self.profile.sensor_filter_order,
                "gravity_lowpass_hz": (
                    self.profile.gravity_lowpass_hz if self.profile.gravity_method == "lpf_0p3" else None
                ),
                "gravity_filter_order": (
                    self.profile.gravity_filter_order if self.profile.gravity_method == "lpf_0p3" else None
                ),
                "state_per_sample": tuple(state_names),
                "state_counts": dict(sorted(Counter(state_names).items())),
                "terminal_state": terminal_state,
                "valid_fraction": float(np.mean(common_valid)),
                "gravity_norm_mean_mps2": (
                    float(np.mean(np.linalg.norm(gravity[route_valid], axis=1)))
                    if np.any(route_valid)
                    else float("nan")
                ),
                "gravity_norm_rmse_mps2": (
                    float(np.sqrt(np.mean(np.square(np.linalg.norm(gravity[route_valid], axis=1) - STANDARD_GRAVITY))))
                    if np.any(route_valid)
                    else float("nan")
                ),
                "first_valid_index": (int(np.flatnonzero(common_valid)[0]) if np.any(common_valid) else None),
                "last_valid_index": (int(np.flatnonzero(common_valid)[-1]) if np.any(common_valid) else None),
                "silent_fallback": False,
                "source_hashes": {
                    "m3_imu_sha256": M3_IMU_SOURCE_SHA256,
                    "m3_imu_math_sha256": M3_IMU_MATH_SOURCE_SHA256,
                    "m3_imu_runtime_sha256": M3_IMU_RUNTIME_SOURCE_SHA256,
                },
                **route_diagnostics,
            }
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            return ImuPreprocessResult(
                {},
                "failed",
                (str(exc),),
                self.profile.gravity_method,
                {"silent_fallback": False},
                np.zeros(acc_raw.shape[0], dtype=bool),
            )

        # 中文：仅全部成功后原子提交 state；失败 chunk 不改变下一 chunk。
        # English: Commit runtime state only after the whole chunk succeeds.
        self.acc_filter_state = next_acc_state
        self.gyro_filter_state = next_gyro_state
        self.gravity_filter_state = next_gravity_state
        self.estimator = next_estimator
        self.last_dynamic = dynamic[-1].copy() if route_valid[-1] else None
        self.global_sample_index += acc_raw.shape[0]
        if timestamps_s is not None:
            self.previous_timestamp_s = float(np.asarray(timestamps_s).ravel()[-1])
        return ImuPreprocessResult(
            processed,
            status,
            repair_reasons,
            self.profile.gravity_method,
            diagnostics,
            common_valid,
        )

def preprocess_imu(
    acc: np.ndarray,
    gyro: np.ndarray,
    *,
    fs_hz: float,
    acc_unit: str,
    gyro_unit: str,
    gravity_method: str,
    timestamps_s: np.ndarray | None = None,
    eskf_config: EskfConfiguration | None = None,
    acceleration_lowpass_hz: float = 20.0,
    gyroscope_lowpass_hz: float = 40.0,
    sensor_filter_order: int = 3,
    gravity_lowpass_hz: float = 0.3,
    gravity_filter_order: int = 2,
) -> ImuPreprocessResult:
    """所有正式参数显式的一次入口 / One-shot facade with explicit parameters."""
    profile = ImuProfile(
        gravity_method=gravity_method,
        acceleration_lowpass_hz=acceleration_lowpass_hz,
        gyroscope_lowpass_hz=gyroscope_lowpass_hz,
        sensor_filter_order=sensor_filter_order,
        gravity_lowpass_hz=gravity_lowpass_hz,
        gravity_filter_order=gravity_filter_order,
        eskf=eskf_config or EskfConfiguration(),
    )
    processor = CausalImuProcessor(
        fs_hz=fs_hz,
        acceleration_unit=acc_unit,
        gyroscope_unit=gyro_unit,
        profile=profile,
    )
    return processor.process_chunk(acc, gyro, timestamps_s=timestamps_s)

def estimate_gravity_no_precalibration_ekf(
    acc_mps2: np.ndarray,
    gyro_rads: np.ndarray,
    *,
    fs_hz: float = CANONICAL_FS_HZ,
    config: EskfConfiguration | None = None,
    acceleration_lowpass_hz: float = 20.0,
    gyroscope_lowpass_hz: float = 40.0,
    sensor_filter_order: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """One-shot MEKF 兼容包装 / Compatibility wrapper around stateful MEKF."""
    result = preprocess_imu(
        acc_mps2,
        gyro_rads,
        fs_hz=fs_hz,
        acc_unit="m/s2",
        gyro_unit="rad/s",
        gravity_method="no_precalibration_ekf",
        eskf_config=config,
        acceleration_lowpass_hz=acceleration_lowpass_hz,
        gyroscope_lowpass_hz=gyroscope_lowpass_hz,
        sensor_filter_order=sensor_filter_order,
    )
    if result.status in {"failed", "no_estimate"}:
        raise ValueError("MEKF failed: " + ";".join(result.reasons))
    return result.processed["gravity_mps2"], result.diagnostics

def estimate_gravity_lpf(
    acc_mps2: np.ndarray,
    *,
    fs_hz: float = CANONICAL_FS_HZ,
    gyro_rads: np.ndarray | None = None,
    acceleration_lowpass_hz: float = 20.0,
    gyroscope_lowpass_hz: float = 40.0,
    sensor_filter_order: int = 3,
    gravity_lowpass_hz: float = 0.3,
    gravity_filter_order: int = 2,
) -> np.ndarray:
    """共享 ACC20 后 causal order2 0.3 Hz comparator / Frozen LPF route."""
    acc = np.asarray(acc_mps2, dtype=np.float64)
    gyro = np.zeros_like(acc) if gyro_rads is None else np.asarray(gyro_rads, dtype=np.float64)
    result = preprocess_imu(
        acc,
        gyro,
        fs_hz=fs_hz,
        acc_unit="m/s2",
        gyro_unit="rad/s",
        gravity_method="lpf_0p3",
        acceleration_lowpass_hz=acceleration_lowpass_hz,
        gyroscope_lowpass_hz=gyroscope_lowpass_hz,
        sensor_filter_order=sensor_filter_order,
        gravity_lowpass_hz=gravity_lowpass_hz,
        gravity_filter_order=gravity_filter_order,
    )
    if result.status == "failed":
        raise ValueError("LPF comparator failed: " + ";".join(result.reasons))
    return result.processed["gravity_mps2"]

@dataclass(frozen=True)
class GravityComparisonResult:
    """Same-input EKF-primary versus LPF-comparator evidence."""
    ekf_gravity_mps2: np.ndarray
    lpf_gravity_mps2: np.ndarray
    common_valid_mask: np.ndarray
    metrics: dict[str, Any]
    primary_method: str = "no_precalibration_ekf"
    comparator_method: str = "lpf_0p3"

def compare_ekf_lpf_gravity(
    acc_mps2: np.ndarray,
    gyro_rads: np.ndarray,
    *,
    fs_hz: float = CANONICAL_FS_HZ,
    config: EskfConfiguration | None = None,
    acceleration_lowpass_hz: float = 20.0,
    gyroscope_lowpass_hz: float = 40.0,
    sensor_filter_order: int = 3,
    gravity_lowpass_hz: float = 0.3,
    gravity_filter_order: int = 2,
) -> GravityComparisonResult:
    """Compute paired gravity estimates and descriptive differences only."""
    ekf, ekf_diagnostics = estimate_gravity_no_precalibration_ekf(
        acc_mps2,
        gyro_rads,
        fs_hz=fs_hz,
        config=config,
        acceleration_lowpass_hz=acceleration_lowpass_hz,
        gyroscope_lowpass_hz=gyroscope_lowpass_hz,
        sensor_filter_order=sensor_filter_order,
    )
    lpf = estimate_gravity_lpf(
        acc_mps2,
        fs_hz=fs_hz,
        gyro_rads=gyro_rads,
        acceleration_lowpass_hz=acceleration_lowpass_hz,
        gyroscope_lowpass_hz=gyroscope_lowpass_hz,
        sensor_filter_order=sensor_filter_order,
        gravity_lowpass_hz=gravity_lowpass_hz,
        gravity_filter_order=gravity_filter_order,
    )
    if ekf.shape != lpf.shape:
        raise ValueError("EKF and LPF gravity outputs lost alignment")
    common = np.isfinite(ekf).all(axis=1) & np.isfinite(lpf).all(axis=1)
    if not np.any(common):
        raise ValueError("EKF/LPF gravity comparison has no common valid samples")
    difference = ekf[common] - lpf[common]
    metrics: dict[str, Any] = {
        "common_valid_fraction": float(np.mean(common)),
        "rmse_mps2": float(np.sqrt(np.mean(np.square(difference)))),
        "mae_mps2": float(np.mean(np.abs(difference))),
        "rmse_by_axis_mps2": np.sqrt(np.mean(np.square(difference), axis=0)),
        "ekf_gravity_norm_mean_mps2": float(np.mean(np.linalg.norm(ekf[common], axis=1))),
        "lpf_gravity_norm_mean_mps2": float(np.mean(np.linalg.norm(lpf[common], axis=1))),
        "ekf_no_static_precalibration": bool(ekf_diagnostics.get("no_static_precalibration", False)),
        "selection_performed": False,
    }
    return GravityComparisonResult(ekf, lpf, common, metrics)

__all__ = [
    "STANDARD_GRAVITY", "EskfConfiguration", "ImuProfile", "ImuPreprocessResult", "GravityComparisonResult",
    "NoPrecalibrationEskf", "CausalImuProcessor", "convert_acceleration", "convert_gyro", "preprocess_imu",
    "compare_ekf_lpf_gravity", "estimate_gravity_no_precalibration_ekf", "estimate_gravity_lpf",
]
