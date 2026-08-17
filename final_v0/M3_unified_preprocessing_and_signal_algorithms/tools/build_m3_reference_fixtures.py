#!/usr/bin/env python3
"""构建确定性 M3 合成 reference fixtures / Build deterministic M3 fixtures.

中文：只写 M3 包内 fixtures，使用固定 seed 和稳定 NPY 格式。NPY 不含运行时间戳，
因此相同 NumPy dtype/shape/bytes 会得到相同 SHA-256。

English: Write only package-local fixtures using a fixed seed and stable NPY format.
NPY carries no runtime timestamp, so identical dtype, shape, and bytes produce the same
SHA-256 digest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = PACKAGE_ROOT / "fixtures"
SEED = 20260815
FS_HZ = 400.0


def sha256_file(path: Path) -> str:
    """计算文件哈希 / Compute a file digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_npy(path: Path, values: np.ndarray) -> None:
    """原子写稳定 NPY / Atomically write a stable NPY file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values), allow_pickle=False)
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    """原子写 strict JSON / Atomically write strict JSON."""

    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)


def build_ppg(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """生成带 DC、漂移、双相脉搏和噪声的 PPG / Generate synthetic PPG."""

    duration_sec = 30.0
    time = np.arange(int(round(duration_sec * FS_HZ))) / FS_HZ
    event_times = np.arange(0.8, duration_sec - 0.2, 0.8)
    raw = 100000.0 + 1200.0 * np.sin(2.0 * np.pi * 0.08 * time)
    for event_time in event_times:
        raw += 2500.0 * np.exp(-0.5 * ((time - event_time) / 0.045) ** 2)
        raw -= 450.0 * np.exp(-0.5 * ((time - event_time - 0.16) / 0.070) ** 2)
    raw += rng.normal(0.0, 35.0, size=time.size)
    expected_peaks = np.rint(event_times * FS_HZ).astype(np.int64)
    return raw.astype(np.float64), expected_peaks


def build_imu(rng: np.random.Generator) -> np.ndarray:
    """生成已知 roll、重力和动态加速度 / Generate IMU with known truth."""

    duration_sec = 12.0
    time = np.arange(int(round(duration_sec * FS_HZ))) / FS_HZ
    gravity = 9.80665
    roll = 0.25 * np.sin(2.0 * np.pi * 0.35 * time)
    roll_rate = 0.25 * 2.0 * np.pi * 0.35 * np.cos(2.0 * np.pi * 0.35 * time)
    gravity_body = np.column_stack(
        [
            np.zeros_like(time),
            gravity * np.sin(roll),
            gravity * np.cos(roll),
        ]
    )
    dynamic = np.column_stack(
        [
            0.8 * np.sin(2.0 * np.pi * 1.2 * time),
            0.15 * np.sin(2.0 * np.pi * 0.7 * time),
            np.zeros_like(time),
        ]
    )
    gyro_bias = np.array([0.012, -0.006, 0.003])
    gyroscope = np.column_stack([roll_rate, np.zeros_like(time), np.zeros_like(time)])
    gyroscope += gyro_bias
    acceleration = gravity_body + dynamic
    acceleration += rng.normal(0.0, 0.003, size=acceleration.shape)
    gyroscope += rng.normal(0.0, 2e-4, size=gyroscope.shape)
    return np.column_stack(
        [acceleration, gyroscope, gravity_body, dynamic]
    ).astype(np.float64)


def main() -> None:
    """构建全部 fixtures 与 hash manifest / Build all fixtures and manifest."""

    rng = np.random.default_rng(SEED)
    ppg, peaks = build_ppg(rng)
    imu = build_imu(rng)
    files = {
        "ppg_reference_v1.npy": ppg,
        "ppg_expected_peaks_v1.npy": peaks,
        "imu_reference_v1.npy": imu,
    }
    # 中文：语义与列顺序属于 fixture 合同，不能仅靠测试代码中的切片位置猜测。
    # English: Semantics and column order are contract data, not implicit test slices.
    semantics = {
        "ppg_reference_v1.npy": (
            "30 s synthetic raw PPG with DC, baseline drift, positive systolic pulses, "
            "negative secondary lobes, and fixed-seed Gaussian noise / 30 秒含 DC、基线漂移、"
            "正向收缩峰、负向次级波及固定种子高斯噪声的合成原始 PPG"
        ),
        "ppg_expected_peaks_v1.npy": (
            "Ground-truth systolic-event sample indices for ppg_reference_v1.npy / "
            "ppg_reference_v1.npy 的收缩事件真值采样点索引"
        ),
        "imu_reference_v1.npy": (
            "12 s synthetic six-axis IMU plus gravity and dynamic-acceleration truth / "
            "12 秒合成六轴 IMU 及重力、动态加速度真值"
        ),
    }
    columns = {
        "imu_reference_v1.npy": [
            "AX_m_s2",
            "AY_m_s2",
            "AZ_m_s2",
            "GX_rad_s",
            "GY_rad_s",
            "GZ_rad_s",
            "gravity_X_truth_m_s2",
            "gravity_Y_truth_m_s2",
            "gravity_Z_truth_m_s2",
            "dynamic_X_truth_m_s2",
            "dynamic_Y_truth_m_s2",
            "dynamic_Z_truth_m_s2",
        ]
    }
    entries: list[dict[str, Any]] = []
    for name, values in files.items():
        path = FIXTURE_ROOT / name
        write_npy(path, values)
        entry: dict[str, Any] = {
            "file": name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "semantics": semantics[name],
        }
        if name in columns:
            entry["columns"] = columns[name]
        entries.append(entry)
    generator_path = Path(__file__).resolve()
    manifest = {
        "schema_version": "m3.reference_fixture_manifest.v1",
        "fixture_manifest_id": "m3_reference_fixtures_v1",
        "status": "deterministic_synthetic_truth",
        "seed": SEED,
        "sampling_rate_hz": FS_HZ,
        "generator": "tools/build_m3_reference_fixtures.py",
        "generator_sha256": sha256_file(generator_path),
        "files": entries,
        "limitations": [
            "Synthetic truth is an engineering fixture, not clinical validation.",
            "Frailty3 has no gravity or PPG peak ground truth.",
        ],
    }
    write_json(FIXTURE_ROOT / "reference_fixture_manifest.json", manifest)


if __name__ == "__main__":
    main()
