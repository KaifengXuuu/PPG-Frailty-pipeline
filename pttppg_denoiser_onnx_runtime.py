from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal

try:
    import onnxruntime as ort
except Exception as exc:  # pragma: no cover
    ort = None
    ORT_IMPORT_ERROR = exc
else:
    ORT_IMPORT_ERROR = None

EPS = 1e-6
DEFAULT_LAGS_MS = (-200, -120, -80, -40, 0, 40, 80, 120, 200)
INPUT_MODE_RAW_IMU = "raw_imu"
INPUT_MODE_RAW_IMU_BASELINE = "raw_imu_baseline"

CANDIDATE_COLUMNS = {
    "time": ["time", "Time", "t"],
    "ecg": ["ecg", "ECG"],
    "peaks": ["peaks", "Rpeaks", "r_peaks"],
    "pleth_1": ["pleth_1", "RED", "red", "Red", "PPG_Wrist_R"],
    "pleth_2": ["pleth_2", "IR", "ir", "Ir", "PPG_Wrist_IR"],
    "a_x": ["a_x", "AX", "accX"],
    "a_y": ["a_y", "AY", "accY"],
    "a_z": ["a_z", "AZ", "accZ"],
    "g_x": ["g_x", "GX", "gyroX"],
    "g_y": ["g_y", "GY", "gyroY"],
    "g_z": ["g_z", "GZ", "gyroZ"],
}


@dataclass
class WindowConfig:
    fs: float = 500.0
    win_sec: float = 6.0
    hop_sec: float = 1.0


@dataclass
class BaselineConfig:
    band_low: float = 0.5
    band_high: float = 8.0
    band_order: int = 3
    gravity_cutoff_hz: float = 0.3
    ridge_alpha: float = 8.0
    lags_ms: Tuple[int, ...] = DEFAULT_LAGS_MS

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaselineConfig":
        data = dict(data)
        if "lags_ms" in data:
            data["lags_ms"] = tuple(int(v) for v in data["lags_ms"])
        return cls(**data)


@dataclass
class ModelConfig:
    input_mode: str = INPUT_MODE_RAW_IMU_BASELINE
    in_channels: int = 15
    base_channels: int = 32

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        data = dict(data)
        return cls(**data)


def ort_available() -> bool:
    return ort is not None


def ort_status_message() -> str:
    if ort is not None:
        return "onnxruntime available"
    return f"onnxruntime unavailable: {ORT_IMPORT_ERROR}"


def _pick(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def safe_float_col(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32)


def bandpass_filter(x: np.ndarray, fs: float, lowcut: float, highcut: float, order: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("bandpass_filter expects a 1D array")
    if len(x) < max(15, 3 * (order + 1)):
        return x.copy()
    wn = [lowcut / (fs / 2.0), highcut / (fs / 2.0)]
    b, a = signal.butter(order, wn, btype="band")
    return signal.filtfilt(b, a, x).astype(np.float32)


def lowpass_filter(x: np.ndarray, fs: float, cutoff_hz: float, order: int = 2, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape[axis] < max(15, 3 * (order + 1)):
        return x.copy()
    b, a = signal.butter(order, cutoff_hz / (fs / 2.0), btype="low")
    return signal.filtfilt(b, a, x, axis=axis).astype(np.float32)


def split_windows(n: int, fs: float, win_sec: float, hop_sec: float) -> Iterable[Tuple[int, int]]:
    win = int(round(win_sec * fs))
    hop = int(round(hop_sec * fs))
    if win <= 0 or hop <= 0 or n < win:
        return
    for start in range(0, n - win + 1, hop):
        yield start, start + win


def deg2rad(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32) * (np.pi / 180.0)


def acc_g_to_ms2(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32) * 9.80665


def linear_acceleration(acc_g: np.ndarray, fs: float, cutoff_hz: float = 0.3) -> np.ndarray:
    acc_ms2 = acc_g_to_ms2(acc_g)
    gravity = lowpass_filter(acc_ms2, fs=fs, cutoff_hz=cutoff_hz, axis=0)
    return (acc_ms2 - gravity).astype(np.float32)


def magn(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt(np.sum(np.asarray(x, dtype=np.float32) ** 2, axis=axis))


def jerk_mag(acc_dyn: np.ndarray, fs: float) -> np.ndarray:
    grad = np.gradient(np.asarray(acc_dyn, dtype=np.float32), axis=0) * float(fs)
    return magn(grad, axis=1).astype(np.float32)


def standardize_channels(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = np.mean(x, axis=1, keepdims=True)
    sd = np.std(x, axis=1, keepdims=True) + EPS
    return (x - mu) / sd


def normalize_pair(raw_pair: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_pair = np.asarray(raw_pair, dtype=np.float32)
    center = np.mean(raw_pair, axis=1, keepdims=True).astype(np.float32)
    scale = (np.std(raw_pair, axis=1, keepdims=True) + EPS).astype(np.float32)
    raw_norm = ((raw_pair - center) / scale).astype(np.float32)
    return raw_norm, center, scale


def normalize_with_reference(raw_pair: np.ndarray, ref_pair: np.ndarray) -> np.ndarray:
    raw_pair = np.asarray(raw_pair, dtype=np.float32)
    ref_pair = np.asarray(ref_pair, dtype=np.float32)
    center = np.mean(raw_pair, axis=1, keepdims=True).astype(np.float32)
    scale = (np.std(raw_pair, axis=1, keepdims=True) + EPS).astype(np.float32)
    return ((ref_pair - center) / scale).astype(np.float32)


def build_reference_channels(acc_dyn: np.ndarray, gyro_rad: np.ndarray, fs: float) -> np.ndarray:
    acc_dyn = np.asarray(acc_dyn, dtype=np.float32)
    gyro_rad = np.asarray(gyro_rad, dtype=np.float32)
    acc_mag = magn(acc_dyn, axis=1).astype(np.float32)
    gyro_mag = magn(gyro_rad, axis=1).astype(np.float32)
    jmag = jerk_mag(acc_dyn, fs=fs)
    ref = np.vstack(
        [
            acc_dyn.T,
            gyro_rad.T,
            acc_mag[None, :],
            gyro_mag[None, :],
            jmag[None, :],
        ]
    )
    return ref.astype(np.float32)


def lag_samples_from_ms(fs: float, lags_ms: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(round(float(ms) * float(fs) / 1000.0)) for ms in lags_ms)


def shift_2d(x: np.ndarray, lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    out = np.zeros_like(x)
    if lag == 0:
        out[:] = x
    elif lag > 0:
        out[:, lag:] = x[:, :-lag]
    else:
        kk = -lag
        out[:, :-kk] = x[:, kk:]
    return out


def build_lagged_bank(ref_channels: np.ndarray, lags_samples: Sequence[int]) -> np.ndarray:
    bank = [shift_2d(ref_channels, lag) for lag in lags_samples]
    return np.concatenate(bank, axis=0).astype(np.float32)


def ridge_artifact_estimate(ppg: np.ndarray, ref_bank: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    ppg = np.asarray(ppg, dtype=np.float32)
    x = np.asarray(ref_bank, dtype=np.float32).T
    x_mu = np.mean(x, axis=0, keepdims=True)
    x_sd = np.std(x, axis=0, keepdims=True) + EPS
    xz = (x - x_mu) / x_sd
    y = ppg - np.mean(ppg)
    xtx = xz.T @ xz
    reg = alpha * np.eye(xtx.shape[0], dtype=np.float32)
    w = np.linalg.solve(xtx + reg, xz.T @ y)
    artifact = xz @ w
    clean = ppg - artifact
    return artifact.astype(np.float32), clean.astype(np.float32)


def build_record_from_dataframe(
    df: pd.DataFrame,
    name: str,
    fs: float,
    baseline_cfg: BaselineConfig,
) -> Dict[str, Any]:
    cols = {key: _pick(df, names) for key, names in CANDIDATE_COLUMNS.items()}
    required = ["pleth_1", "pleth_2", "a_x", "a_y", "a_z", "g_x", "g_y", "g_z"]
    missing = [key for key in required if cols.get(key) is None]
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}")

    rec: Dict[str, Any] = {}
    if cols.get("time"):
        time_col = cols["time"]
        time_vals = pd.to_datetime(df[time_col], errors="coerce")
        if time_vals.notna().all():
            rec["time"] = (time_vals - time_vals.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float32)
        else:
            rec["time"] = np.arange(len(df), dtype=np.float32) / float(fs)
    else:
        rec["time"] = np.arange(len(df), dtype=np.float32) / float(fs)

    rec["pleth_1"] = bandpass_filter(
        safe_float_col(df[cols["pleth_1"]]),
        fs=fs,
        lowcut=baseline_cfg.band_low,
        highcut=baseline_cfg.band_high,
        order=baseline_cfg.band_order,
    )
    rec["pleth_2"] = bandpass_filter(
        safe_float_col(df[cols["pleth_2"]]),
        fs=fs,
        lowcut=baseline_cfg.band_low,
        highcut=baseline_cfg.band_high,
        order=baseline_cfg.band_order,
    )

    acc_g = np.column_stack(
        [
            safe_float_col(df[cols["a_x"]]),
            safe_float_col(df[cols["a_y"]]),
            safe_float_col(df[cols["a_z"]]),
        ]
    ).astype(np.float32)
    gyro_deg = np.column_stack(
        [
            safe_float_col(df[cols["g_x"]]),
            safe_float_col(df[cols["g_y"]]),
            safe_float_col(df[cols["g_z"]]),
        ]
    ).astype(np.float32)

    acc_dyn = linear_acceleration(acc_g, fs=fs, cutoff_hz=baseline_cfg.gravity_cutoff_hz)
    gyro_rad = deg2rad(gyro_deg)
    rec["acc_dyn"] = acc_dyn
    rec["gyro_rad"] = gyro_rad
    rec["ref_channels"] = build_reference_channels(acc_dyn, gyro_rad, fs=fs)
    return rec


def compute_record_linear_baseline(
    rec: Dict[str, Any],
    fs: float,
    baseline_cfg: BaselineConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    ref_bank = build_lagged_bank(rec["ref_channels"], lag_samples_from_ms(fs, baseline_cfg.lags_ms))
    art_1, clean_1 = ridge_artifact_estimate(rec["pleth_1"], ref_bank, alpha=baseline_cfg.ridge_alpha)
    art_2, clean_2 = ridge_artifact_estimate(rec["pleth_2"], ref_bank, alpha=baseline_cfg.ridge_alpha)
    base_art = np.stack([art_1, art_2], axis=0).astype(np.float32)
    base_clean = np.stack([clean_1, clean_2], axis=0).astype(np.float32)
    return base_art, base_clean


def prepare_window_sample_for_onnx(
    rec: Dict[str, Any],
    start: int,
    end: int,
    fs: float,
    baseline_cfg: BaselineConfig,
    model_cfg: ModelConfig,
) -> Dict[str, np.ndarray]:
    raw_pair = np.stack(
        [
            rec["pleth_1"][start:end].astype(np.float32),
            rec["pleth_2"][start:end].astype(np.float32),
        ],
        axis=0,
    )
    ref_slice = rec["ref_channels"][:, start:end].astype(np.float32)
    ref_summary = standardize_channels(ref_slice)
    if "base_art_full" in rec and "base_clean_full" in rec:
        base_art = rec["base_art_full"][:, start:end].astype(np.float32)
        base_clean = rec["base_clean_full"][:, start:end].astype(np.float32)
    else:
        lags = lag_samples_from_ms(fs, baseline_cfg.lags_ms)
        ref_bank = build_lagged_bank(ref_slice, lags)
        art_1, clean_1 = ridge_artifact_estimate(raw_pair[0], ref_bank, alpha=baseline_cfg.ridge_alpha)
        art_2, clean_2 = ridge_artifact_estimate(raw_pair[1], ref_bank, alpha=baseline_cfg.ridge_alpha)
        base_art = np.stack([art_1, art_2], axis=0).astype(np.float32)
        base_clean = np.stack([clean_1, clean_2], axis=0).astype(np.float32)

    raw_norm, center, scale = normalize_pair(raw_pair)
    base_clean_norm = normalize_with_reference(raw_pair, base_clean)
    base_art_norm = (base_art / scale).astype(np.float32)

    if model_cfg.input_mode == INPUT_MODE_RAW_IMU:
        model_input = np.concatenate([raw_norm, ref_summary], axis=0).astype(np.float32)
    elif model_cfg.input_mode == INPUT_MODE_RAW_IMU_BASELINE:
        model_input = np.concatenate([raw_norm, base_clean_norm, base_art_norm, ref_summary], axis=0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported input_mode: {model_cfg.input_mode}")

    return {
        "model_input": model_input,
        "raw_pair": raw_pair.astype(np.float32),
        "base_clean": base_clean.astype(np.float32),
        "center": center.astype(np.float32),
        "scale": scale.astype(np.float32),
    }


def _resolve_meta_path(onnx_path: Path) -> Path:
    candidate = onnx_path.with_suffix(".json")
    if candidate.exists():
        return candidate
    candidate = onnx_path.parent / "hybrid_denoiser_meta.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find metadata json next to {onnx_path}")


def load_onnx_bundle(onnx_path: Path | str) -> Dict[str, Any]:
    if ort is None:
        raise RuntimeError(f"onnxruntime is not available: {ORT_IMPORT_ERROR}")

    onnx_path = Path(onnx_path)
    meta_path = _resolve_meta_path(onnx_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model_cfg = ModelConfig.from_dict(meta["model_cfg"])
    window_cfg = WindowConfig(**meta["window_cfg"])
    baseline_cfg = BaselineConfig.from_dict(meta["baseline_cfg"])

    return {
        "session": session,
        "onnx_path": onnx_path,
        "meta_path": meta_path,
        "window_cfg": window_cfg,
        "baseline_cfg": baseline_cfg,
        "model_cfg": model_cfg,
        "input_name": session.get_inputs()[0].name,
        "output_name": session.get_outputs()[0].name,
        "meta": meta,
    }


def denoise_record_onnx(
    rec: Dict[str, Any],
    bundle: Dict[str, Any],
    motion_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    session = bundle["session"]
    window_cfg: WindowConfig = bundle["window_cfg"]
    baseline_cfg: BaselineConfig = bundle["baseline_cfg"]
    model_cfg: ModelConfig = bundle["model_cfg"]
    input_name: str = bundle["input_name"]

    raw_full = np.stack([rec["pleth_1"], rec["pleth_2"]], axis=0).astype(np.float32)
    if "base_art_full" not in rec or "base_clean_full" not in rec:
        base_art_full, base_clean_full = compute_record_linear_baseline(rec, fs=window_cfg.fs, baseline_cfg=baseline_cfg)
        rec["base_art_full"] = base_art_full.astype(np.float32)
        rec["base_clean_full"] = base_clean_full.astype(np.float32)

    den_num = np.zeros_like(raw_full, dtype=np.float32)
    base_num = np.zeros_like(raw_full, dtype=np.float32)
    den_w = np.zeros(raw_full.shape[1], dtype=np.float32)

    win = int(round(window_cfg.win_sec * window_cfg.fs))
    taper = np.hanning(win).astype(np.float32)
    if np.allclose(taper.sum(), 0.0):
        taper = np.ones(win, dtype=np.float32)

    windows = list(split_windows(raw_full.shape[1], window_cfg.fs, window_cfg.win_sec, window_cfg.hop_sec))
    if not windows:
        return {"raw": raw_full, "baseline": raw_full.copy(), "denoised": raw_full.copy(), "windows": []}

    for start, end in windows:
        sample = prepare_window_sample_for_onnx(
            rec,
            start,
            end,
            fs=window_cfg.fs,
            baseline_cfg=baseline_cfg,
            model_cfg=model_cfg,
        )
        model_input = sample["model_input"][None, ...].astype(np.float32)
        artifact_hat = session.run(None, {input_name: model_input})[0]
        artifact_hat = np.asarray(artifact_hat, dtype=np.float32).squeeze(0)

        raw_pair = sample["raw_pair"]
        center = sample["center"]
        scale = sample["scale"]
        raw_norm = ((raw_pair - center) / scale).astype(np.float32)
        clean_norm = raw_norm - artifact_hat
        clean_win = clean_norm * scale + center
        base_win = sample["base_clean"]

        den_num[:, start:end] += clean_win * taper[None, :]
        base_num[:, start:end] += base_win * taper[None, :]
        den_w[start:end] += taper

    den_w = np.maximum(den_w, EPS)
    denoised = den_num / den_w[None, :]
    baseline = base_num / den_w[None, :]

    if motion_mask is not None:
        motion_mask = np.asarray(motion_mask, dtype=bool).reshape(-1)
        if motion_mask.shape[0] != raw_full.shape[1]:
            raise ValueError("motion_mask length must match signal length")
        denoised = np.where(motion_mask[None, :], denoised, raw_full)
        baseline = np.where(motion_mask[None, :], baseline, raw_full)

    return {
        "raw": raw_full.astype(np.float32),
        "baseline": baseline.astype(np.float32),
        "denoised": denoised.astype(np.float32),
        "windows": windows,
    }
