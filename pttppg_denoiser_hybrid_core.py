from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

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

REF_CHANNEL_NAMES = [
    "acc_dyn_x",
    "acc_dyn_y",
    "acc_dyn_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "acc_mag",
    "gyro_mag",
    "jerk_mag",
]


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
        if "input_mode" not in data:
            data["input_mode"] = infer_input_mode(data.get("in_channels"))
        data["in_channels"] = model_input_channels(data["input_mode"])
        return cls(**data)


@dataclass
class PriorConfig:
    target_lowpass_hz: float = 6.0
    target_band_low: float = 0.5
    target_band_high: float = 8.0
    target_band_order: int = 3
    target_lowpass_order: int = 2
    template_len: int = 256
    library_bins: int = 5
    min_ibi_sec: float = 0.35
    max_ibi_sec: float = 1.60
    base_channels: int = 16


@dataclass
class LossConfig:
    artifact: float = 1.0
    clean: float = 0.35
    sit: float = 0.75
    peak: float = 0.2
    decorr: float = 0.12
    slope: float = 0.18
    anchor: float = 0.12


def loss_config_from_dict(data: Dict[str, Any]) -> LossConfig:
    data = dict(data)
    if "teacher" in data and "artifact" not in data:
        data["artifact"] = data.pop("teacher")
    if "prior" in data and "artifact" not in data:
        data["artifact"] = data.pop("prior")
    if "beat" in data and "slope" not in data:
        data["slope"] = data.pop("beat")
    if "identity" in data and "sit" not in data:
        data["sit"] = data.pop("identity")
    if "delta" in data and "anchor" not in data:
        data["anchor"] = data.pop("delta")
    data.setdefault("artifact", LossConfig.artifact)
    data.setdefault("clean", LossConfig.clean)
    data.setdefault("sit", LossConfig.sit)
    data.setdefault("peak", LossConfig.peak)
    data.setdefault("decorr", LossConfig.decorr)
    data.setdefault("slope", LossConfig.slope)
    data.setdefault("anchor", LossConfig.anchor)
    return LossConfig(**data)


def model_input_channels(input_mode: str) -> int:
    if input_mode == INPUT_MODE_RAW_IMU:
        return 11
    if input_mode == INPUT_MODE_RAW_IMU_BASELINE:
        return 15
    raise ValueError(f"Unsupported input_mode: {input_mode}")


def infer_input_mode(in_channels: Optional[int]) -> str:
    if int(in_channels or 15) == 11:
        return INPUT_MODE_RAW_IMU
    return INPUT_MODE_RAW_IMU_BASELINE


def _pick(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def safe_float_col(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32)


def activity_from_filename(stem: str) -> str:
    parts = stem.split("_")
    return parts[-1].lower() if parts else stem.lower()


def subject_from_filename(stem: str) -> str:
    parts = stem.split("_")
    return parts[0].lower() if parts else stem.lower()


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


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - np.mean(x)) / (np.std(x) + EPS)


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


def smooth_target_pair(raw_pair: np.ndarray, fs: float, prior_cfg: PriorConfig) -> np.ndarray:
    raw_pair = np.asarray(raw_pair, dtype=np.float32)
    out = []
    for ch in range(raw_pair.shape[0]):
        x = lowpass_filter(
            raw_pair[ch],
            fs=fs,
            cutoff_hz=prior_cfg.target_lowpass_hz,
            order=prior_cfg.target_lowpass_order,
        )
        x = bandpass_filter(
            x,
            fs=fs,
            lowcut=prior_cfg.target_band_low,
            highcut=prior_cfg.target_band_high,
            order=prior_cfg.target_band_order,
        )
        out.append(x.astype(np.float32))
    return np.stack(out, axis=0).astype(np.float32)


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
        k = -lag
        out[:, :-k] = x[:, k:]
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

    pleth_1_raw = safe_float_col(df[cols["pleth_1"]])
    pleth_2_raw = safe_float_col(df[cols["pleth_2"]])
    rec["pleth_1"] = bandpass_filter(
        pleth_1_raw,
        fs=fs,
        lowcut=baseline_cfg.band_low,
        highcut=baseline_cfg.band_high,
        order=baseline_cfg.band_order,
    )
    rec["pleth_2"] = bandpass_filter(
        pleth_2_raw,
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

    acc_mag = magn(acc_dyn, axis=1)
    gyro_mag = magn(gyro_rad, axis=1)
    rec["motion_proxy"] = (zscore_1d(acc_mag) + 0.35 * zscore_1d(gyro_mag)).astype(np.float32)

    peaks_col = cols.get("peaks")
    if peaks_col is None:
        rec["peaks"] = np.zeros(len(df), dtype=np.float32)
    else:
        rec["peaks"] = (safe_float_col(df[peaks_col]) > 0.5).astype(np.float32)

    stem = Path(name).stem
    rec["_file"] = name
    rec["_subject"] = subject_from_filename(stem)
    rec["_activity"] = activity_from_filename(stem)
    return rec


def load_physionet_csv(root: Path | str, fs: float, baseline_cfg: BaselineConfig) -> Dict[str, List[Dict[str, Any]]]:
    root = Path(root)
    csv_dir = root / "csv"
    if not csv_dir.exists():
        csv_dir = root / "files" / "pulse-transit-time-ppg" / "1.1.0" / "csv"
    if not csv_dir.exists():
        raise FileNotFoundError(f"Could not find csv directory under {root}")

    sub2recs: Dict[str, List[Dict[str, Any]]] = {}
    paths = sorted(p for p in csv_dir.glob("s*_*.csv") if p.name != "subjects_info.csv")
    for path in paths:
        rec = build_record_from_dataframe(pd.read_csv(path), path.name, fs=fs, baseline_cfg=baseline_cfg)
        sub2recs.setdefault(rec["_subject"], []).append(rec)
    return sub2recs


def load_single_record(csv_path: Path | str, fs: float, baseline_cfg: BaselineConfig) -> Dict[str, Any]:
    csv_path = Path(csv_path)
    return build_record_from_dataframe(pd.read_csv(csv_path), csv_path.name, fs=fs, baseline_cfg=baseline_cfg)


def _estimate_ppg_peaks(ppg: np.ndarray, fs: float) -> np.ndarray:
    ppg = np.asarray(ppg, dtype=np.float32)
    distance = max(1, int(round(0.35 * fs)))
    prominence = 0.15 * float(np.std(ppg) + EPS)
    peaks, _ = signal.find_peaks(ppg, distance=distance, prominence=prominence)
    return peaks.astype(int)


def estimate_subject_ppg_delay(records: Sequence[Dict[str, Any]], fs: float) -> float:
    delays: List[float] = []
    for rec in records:
        if rec.get("_activity") != "sit":
            continue
        ecg_peaks = np.where(np.asarray(rec.get("peaks", [])) > 0.5)[0]
        ppg_peaks = _estimate_ppg_peaks(rec["pleth_2"], fs=fs)
        if len(ecg_peaks) < 8 or len(ppg_peaks) < 8:
            continue
        for rp in ecg_peaks:
            j = np.searchsorted(ppg_peaks, rp)
            if j >= len(ppg_peaks):
                continue
            delay = ppg_peaks[j] - rp
            if 0.08 * fs <= delay <= 0.45 * fs:
                delays.append(float(delay))
    if not delays:
        return float(round(0.2 * fs))
    return float(np.median(np.asarray(delays, dtype=np.float32)))


def estimate_delay_table(records: Sequence[Dict[str, Any]], fs: float) -> Tuple[Dict[str, float], float]:
    by_subject: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        by_subject.setdefault(rec["_subject"], []).append(rec)
    delay_by_subject = {sid: estimate_subject_ppg_delay(sub_records, fs=fs) for sid, sub_records in by_subject.items()}
    if delay_by_subject:
        default_delay = float(np.median(np.asarray(list(delay_by_subject.values()), dtype=np.float32)))
    else:
        default_delay = float(round(0.2 * fs))
    return delay_by_subject, default_delay


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


def _filter_peak_indices(
    peaks: np.ndarray,
    fs: float,
    min_ibi_sec: float,
    max_ibi_sec: float,
    n: int,
) -> np.ndarray:
    peaks = np.asarray(peaks, dtype=int)
    peaks = peaks[(peaks > 1) & (peaks < max(2, n - 2))]
    if peaks.size == 0:
        return peaks
    min_gap = max(1, int(round(min_ibi_sec * fs)))
    max_gap = max(min_gap + 1, int(round(max_ibi_sec * fs)))
    keep = [int(peaks[0])]
    for peak in peaks[1:]:
        gap = int(peak) - int(keep[-1])
        if gap < min_gap:
            continue
        if gap > max_gap:
            keep.append(int(peak))
        else:
            keep.append(int(peak))
    keep = np.asarray(sorted(set(keep)), dtype=int)
    valid = [keep[0]]
    for peak in keep[1:]:
        gap = int(peak) - int(valid[-1])
        if gap >= min_gap:
            valid.append(int(peak))
    valid = np.asarray(valid, dtype=int)
    if valid.size >= 2:
        gaps = np.diff(valid)
        good = np.ones(valid.size, dtype=bool)
        good[1:] &= gaps <= max_gap
        good[:-1] |= gaps <= max_gap
        valid = valid[good]
    return valid.astype(int)


def _ppg_peaks_with_prominence(ppg: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    ppg = np.asarray(ppg, dtype=np.float32)
    distance = max(1, int(round(0.35 * fs)))
    prominence = 0.15 * float(np.std(ppg) + EPS)
    peaks, props = signal.find_peaks(ppg, distance=distance, prominence=prominence)
    prom = props.get("prominences")
    if prom is None:
        prom = np.ones(len(peaks), dtype=np.float32)
    return peaks.astype(int), np.asarray(prom, dtype=np.float32)


def get_peak_indices_for_record(
    rec: Dict[str, Any],
    fs: float,
    delay_by_subject: Dict[str, float],
    default_delay: float,
    prior_cfg: PriorConfig,
    prefer_ecg: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(rec["pleth_2"])
    if prefer_ecg and np.any(np.asarray(rec.get("peaks", [])) > 0.5):
        delay = int(round(delay_by_subject.get(rec["_subject"], default_delay)))
        ecg_peaks = np.where(np.asarray(rec.get("peaks", [])) > 0.5)[0] + delay
        peaks = _filter_peak_indices(ecg_peaks, fs, prior_cfg.min_ibi_sec, prior_cfg.max_ibi_sec, n=n)
        conf = np.ones(peaks.size, dtype=np.float32)
        if peaks.size >= 2:
            return peaks, conf
    peaks, prom = _ppg_peaks_with_prominence(rec["pleth_2"], fs=fs)
    peaks = _filter_peak_indices(peaks, fs, prior_cfg.min_ibi_sec, prior_cfg.max_ibi_sec, n=n)
    if peaks.size == 0:
        return peaks, np.zeros(0, dtype=np.float32)
    if prom.size >= peaks.size:
        prom = prom[: peaks.size]
    else:
        prom = np.pad(prom, (0, peaks.size - prom.size), constant_values=float(np.mean(prom) if prom.size else 1.0))
    prom = prom / (float(np.median(prom) + EPS))
    conf = np.clip(prom, 0.15, 1.5).astype(np.float32)
    return peaks, conf


def _normalize_beat_template(seg: np.ndarray) -> np.ndarray:
    seg = np.asarray(seg, dtype=np.float32)
    seg = seg - np.mean(seg, axis=1, keepdims=True)
    seg = seg / (np.std(seg, axis=1, keepdims=True) + EPS)
    return seg.astype(np.float32)


def build_sit_beat_library(
    records: Sequence[Dict[str, Any]],
    fs: float,
    prior_cfg: PriorConfig,
    delay_by_subject: Dict[str, float],
    default_delay: float,
) -> Dict[str, Any]:
    beats: List[np.ndarray] = []
    ibis: List[int] = []

    for rec in records:
        if rec.get("_activity") != "sit":
            continue
        raw_pair = np.stack([rec["pleth_1"], rec["pleth_2"]], axis=0).astype(np.float32)
        target_pair = smooth_target_pair(raw_pair, fs=fs, prior_cfg=prior_cfg)
        peaks, _ = get_peak_indices_for_record(
            rec,
            fs=fs,
            delay_by_subject=delay_by_subject,
            default_delay=default_delay,
            prior_cfg=prior_cfg,
            prefer_ecg=True,
        )
        if peaks.size < 2:
            continue
        for p0, p1 in zip(peaks[:-1], peaks[1:]):
            ibi = int(p1 - p0)
            if ibi < int(round(prior_cfg.min_ibi_sec * fs)) or ibi > int(round(prior_cfg.max_ibi_sec * fs)):
                continue
            seg = target_pair[:, p0 : p1 + 1]
            if seg.shape[1] < 4:
                continue
            seg_rs = signal.resample(seg, prior_cfg.template_len, axis=1).astype(np.float32)
            beats.append(_normalize_beat_template(seg_rs))
            ibis.append(ibi)

    if not beats:
        raise RuntimeError("Could not build sit beat library: no valid sit beats found")

    beats_np = np.stack(beats, axis=0).astype(np.float32)
    ibis_np = np.asarray(ibis, dtype=np.float32)
    n_bins = max(1, min(int(prior_cfg.library_bins), beats_np.shape[0]))
    if n_bins == 1:
        bin_ids = np.zeros(beats_np.shape[0], dtype=int)
        centers = np.asarray([float(np.median(ibis_np))], dtype=np.float32)
    else:
        edges = np.quantile(ibis_np, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0] -= 1.0
        edges[-1] += 1.0
        bin_ids = np.clip(np.digitize(ibis_np, edges[1:-1], right=False), 0, n_bins - 1).astype(int)
        centers = np.asarray([float(np.median(ibis_np[bin_ids == idx])) for idx in range(n_bins)], dtype=np.float32)

    templates = []
    counts = []
    for idx in range(n_bins):
        sel = beats_np[bin_ids == idx]
        if sel.size == 0:
            sel = beats_np
        templates.append(np.median(sel, axis=0).astype(np.float32))
        counts.append(int(sel.shape[0]))

    return {
        "templates": np.stack(templates, axis=0).astype(np.float32),
        "ibi_centers": centers.astype(np.float32),
        "template_len": int(prior_cfg.template_len),
        "library_bins": int(n_bins),
        "counts": np.asarray(counts, dtype=np.int32),
    }


def _choose_template_for_ibi(ibi: int, beat_library: Dict[str, Any]) -> np.ndarray:
    centers = np.asarray(beat_library["ibi_centers"], dtype=np.float32)
    idx = int(np.argmin(np.abs(centers - float(ibi))))
    return np.asarray(beat_library["templates"][idx], dtype=np.float32)


def synthesize_pseudo_clean(
    anchor_pair: np.ndarray,
    peak_idx: np.ndarray,
    peak_conf: np.ndarray,
    beat_library: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchor_pair = np.asarray(anchor_pair, dtype=np.float32)
    n = anchor_pair.shape[1]
    pseudo = np.zeros_like(anchor_pair, dtype=np.float32)
    weight = np.zeros(n, dtype=np.float32)
    peak_mask = np.zeros(n, dtype=np.float32)

    if peak_idx.size < 2:
        return anchor_pair.copy(), np.ones(n, dtype=np.float32), peak_mask

    for idx, (p0, p1) in enumerate(zip(peak_idx[:-1], peak_idx[1:])):
        p0 = int(max(0, p0))
        p1 = int(min(n - 1, p1))
        if p1 <= p0 + 2:
            continue
        ibi = p1 - p0
        templ = _choose_template_for_ibi(ibi, beat_library)
        templ_rs = signal.resample(templ, ibi + 1, axis=1).astype(np.float32)
        templ_rs = _normalize_beat_template(templ_rs)

        seg = anchor_pair[:, p0 : p1 + 1]
        seg_center = np.median(seg, axis=1, keepdims=True).astype(np.float32)
        seg_scale = (
            np.percentile(seg, 90, axis=1, keepdims=True) - np.percentile(seg, 10, axis=1, keepdims=True)
        ).astype(np.float32)
        seg_scale = np.maximum(seg_scale, np.std(seg, axis=1, keepdims=True).astype(np.float32) + EPS)
        seg_syn = seg_center + 0.5 * seg_scale * templ_rs

        conf = float(peak_conf[min(idx, peak_conf.size - 1)]) if peak_conf.size > 0 else 1.0
        pseudo[:, p0 : p1 + 1] += conf * seg_syn
        weight[p0 : p1 + 1] += conf
        peak_mask[p0] = max(peak_mask[p0], conf)
        peak_mask[p1] = max(peak_mask[p1], conf)

    uncovered = weight <= 0.0
    if np.any(uncovered):
        pseudo[:, uncovered] = anchor_pair[:, uncovered]
        weight[uncovered] = 1.0

    pseudo = pseudo / np.maximum(weight[None, :], EPS)
    return pseudo.astype(np.float32), weight.astype(np.float32), peak_mask.astype(np.float32)


def attach_record_targets(
    records: Sequence[Dict[str, Any]],
    fs: float,
    baseline_cfg: BaselineConfig,
    prior_cfg: PriorConfig,
    beat_library: Dict[str, Any],
    delay_by_subject: Dict[str, float],
    default_delay: float,
    prefer_ecg: bool,
) -> None:
    for rec in records:
        raw_pair = np.stack([rec["pleth_1"], rec["pleth_2"]], axis=0).astype(np.float32)
        rec["sit_target_full"] = smooth_target_pair(raw_pair, fs=fs, prior_cfg=prior_cfg)

        base_art_full, base_clean_full = compute_record_linear_baseline(rec, fs=fs, baseline_cfg=baseline_cfg)
        rec["base_art_full"] = base_art_full.astype(np.float32)
        rec["base_clean_full"] = base_clean_full.astype(np.float32)

        peaks, peak_conf = get_peak_indices_for_record(
            rec,
            fs=fs,
            delay_by_subject=delay_by_subject,
            default_delay=default_delay,
            prior_cfg=prior_cfg,
            prefer_ecg=prefer_ecg,
        )
        rec["prior_peak_idx"] = peaks.astype(int)
        rec["prior_peak_conf"] = peak_conf.astype(np.float32)

        anchor_pair = rec["sit_target_full"] if rec.get("_activity") == "sit" else base_clean_full
        pseudo_clean, pseudo_weight, pseudo_peak_mask = synthesize_pseudo_clean(
            anchor_pair=anchor_pair,
            peak_idx=peaks,
            peak_conf=peak_conf,
            beat_library=beat_library,
        )
        rec["pseudo_clean_full"] = pseudo_clean.astype(np.float32)
        rec["pseudo_weight_full"] = pseudo_weight.astype(np.float32)
        rec["pseudo_peak_mask_full"] = pseudo_peak_mask.astype(np.float32)


def prepare_window_sample(
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

    sit_target = rec.get("sit_target_full")
    if sit_target is None:
        sit_target = raw_pair
    else:
        sit_target = sit_target[:, start:end].astype(np.float32)
    sit_target_norm = normalize_with_reference(raw_pair, sit_target)

    pseudo_clean = rec.get("pseudo_clean_full")
    if pseudo_clean is None:
        pseudo_clean = base_clean
    else:
        pseudo_clean = pseudo_clean[:, start:end].astype(np.float32)
    pseudo_clean_norm = normalize_with_reference(raw_pair, pseudo_clean)
    pseudo_art_norm = (raw_norm - pseudo_clean_norm).astype(np.float32)

    pseudo_weight = rec.get("pseudo_weight_full")
    if pseudo_weight is None:
        pseudo_weight = np.ones(end - start, dtype=np.float32)
    else:
        pseudo_weight = pseudo_weight[start:end].astype(np.float32)

    peaks = rec.get("pseudo_peak_mask_full")
    if peaks is None:
        peaks = rec["peaks"][start:end].astype(np.float32)
    else:
        peaks = peaks[start:end].astype(np.float32)
    motion_proxy = zscore_1d(rec["motion_proxy"][start:end]).astype(np.float32)

    if model_cfg.input_mode == INPUT_MODE_RAW_IMU:
        model_input = np.concatenate([raw_norm, ref_summary], axis=0).astype(np.float32)
    elif model_cfg.input_mode == INPUT_MODE_RAW_IMU_BASELINE:
        model_input = np.concatenate([raw_norm, base_clean_norm, base_art_norm, ref_summary], axis=0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported input_mode: {model_cfg.input_mode}")

    return {
        "model_input": model_input,
        "raw_norm": raw_norm.astype(np.float32),
        "base_clean_norm": base_clean_norm.astype(np.float32),
        "base_art_norm": base_art_norm.astype(np.float32),
        "pseudo_clean_norm": pseudo_clean_norm.astype(np.float32),
        "pseudo_art_norm": pseudo_art_norm.astype(np.float32),
        "sit_target_norm": sit_target_norm.astype(np.float32),
        "pseudo_weight": pseudo_weight.astype(np.float32),
        "peaks": peaks.astype(np.float32),
        "motion_proxy": motion_proxy.astype(np.float32),
        "raw_pair": raw_pair.astype(np.float32),
        "base_clean": base_clean.astype(np.float32),
        "pseudo_clean": pseudo_clean.astype(np.float32),
        "sit_target": sit_target.astype(np.float32),
        "center": center.astype(np.float32),
        "scale": scale.astype(np.float32),
        "is_motion": np.asarray(float(rec["_activity"] != "sit"), dtype=np.float32),
    }


class HybridWindowDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        window_cfg: WindowConfig,
        baseline_cfg: BaselineConfig,
        model_cfg: ModelConfig,
        max_windows_per_record: Optional[int] = None,
    ):
        self.records = list(records)
        self.window_cfg = window_cfg
        self.baseline_cfg = baseline_cfg
        self.model_cfg = model_cfg
        self.index: List[Tuple[int, int, int]] = []
        for rid, rec in enumerate(self.records):
            windows = list(split_windows(len(rec["pleth_2"]), window_cfg.fs, window_cfg.win_sec, window_cfg.hop_sec))
            if max_windows_per_record is not None:
                windows = windows[: max_windows_per_record]
            for start, end in windows:
                self.index.append((rid, start, end))
        self.in_channels = model_cfg.in_channels

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        rid, start, end = self.index[idx]
        rec = self.records[rid]
        sample = prepare_window_sample(
            rec,
            start,
            end,
            fs=self.window_cfg.fs,
            baseline_cfg=self.baseline_cfg,
            model_cfg=self.model_cfg,
        )
        return (
            torch.from_numpy(sample["model_input"]),
            torch.from_numpy(sample["raw_norm"]),
            torch.from_numpy(sample["base_clean_norm"]),
            torch.from_numpy(sample["base_art_norm"]),
            torch.from_numpy(sample["pseudo_clean_norm"]),
            torch.from_numpy(sample["pseudo_art_norm"]),
            torch.from_numpy(sample["sit_target_norm"]),
            torch.from_numpy(sample["pseudo_weight"][None, :]),
            torch.from_numpy(sample["peaks"]),
            torch.from_numpy(sample["motion_proxy"][None, :]),
            torch.tensor(sample["is_motion"], dtype=torch.float32),
            rec["_subject"],
        )


class CleanPriorWindowDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        window_cfg: WindowConfig,
        max_windows_per_record: Optional[int] = None,
    ):
        self.records = [rec for rec in records if rec.get("_activity") == "sit"]
        self.window_cfg = window_cfg
        self.index: List[Tuple[int, int, int]] = []
        for rid, rec in enumerate(self.records):
            windows = list(split_windows(len(rec["pleth_2"]), window_cfg.fs, window_cfg.win_sec, window_cfg.hop_sec))
            if max_windows_per_record is not None:
                windows = windows[: max_windows_per_record]
            for start, end in windows:
                self.index.append((rid, start, end))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        rid, start, end = self.index[idx]
        rec = self.records[rid]
        raw_pair = np.stack(
            [
                rec["pleth_1"][start:end].astype(np.float32),
                rec["pleth_2"][start:end].astype(np.float32),
            ],
            axis=0,
        )
        raw_norm, _, _ = normalize_pair(raw_pair)
        return torch.from_numpy(raw_norm)


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
            nn.GroupNorm(1, out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2),
            nn.GroupNorm(1, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridArtifactRefiner(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()
        b = base_channels
        self.enc1 = ConvBlock1D(in_channels, b)
        self.down1 = nn.Sequential(nn.AvgPool1d(2), ConvBlock1D(b, 2 * b))
        self.down2 = nn.Sequential(nn.AvgPool1d(2), ConvBlock1D(2 * b, 4 * b))
        self.mid = ConvBlock1D(4 * b, 4 * b)
        self.up1 = ConvBlock1D(4 * b + 2 * b, 2 * b)
        self.up2 = ConvBlock1D(2 * b + b, b)
        self.out = nn.Conv1d(b, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        mid = self.mid(e3)

        u1 = F.interpolate(mid, size=e2.shape[-1], mode="linear", align_corners=False)
        u1 = self.up1(torch.cat([u1, e2], dim=1))

        u2 = F.interpolate(u1, size=e1.shape[-1], mode="linear", align_corners=False)
        u2 = self.up2(torch.cat([u2, e1], dim=1))
        return self.out(u2)


class CleanPriorAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 2, base_channels: int = 16):
        super().__init__()
        b = base_channels
        self.enc1 = ConvBlock1D(in_channels, b)
        self.down1 = nn.Sequential(nn.AvgPool1d(2), ConvBlock1D(b, 2 * b))
        self.down2 = nn.Sequential(nn.AvgPool1d(2), ConvBlock1D(2 * b, 4 * b))
        self.mid = ConvBlock1D(4 * b, 4 * b)
        self.up1 = ConvBlock1D(4 * b + 2 * b, 2 * b)
        self.up2 = ConvBlock1D(2 * b + b, b)
        self.out = nn.Conv1d(b, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        mid = self.mid(e3)

        u1 = F.interpolate(mid, size=e2.shape[-1], mode="linear", align_corners=False)
        u1 = self.up1(torch.cat([u1, e2], dim=1))

        u2 = F.interpolate(u1, size=e1.shape[-1], mode="linear", align_corners=False)
        u2 = self.up2(torch.cat([u2, e1], dim=1))
        return self.out(u2)


def shift_peaks(peaks: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(peaks)
    batch, length = peaks.shape
    for idx in range(batch):
        shift = int(shifts[idx].item())
        if shift == 0:
            out[idx] = peaks[idx]
        elif shift > 0:
            out[idx, shift:] = peaks[idx, : length - shift]
        else:
            kk = -shift
            out[idx, : length - kk] = peaks[idx, kk:]
    return out


def soft_peak_map(y: torch.Tensor, fs: float, gain: float = 10.0) -> torch.Tensor:
    y = y.squeeze(1)
    win = max(5, int(round(0.15 * fs)))
    if win % 2 == 0:
        win += 1
    pad = win // 2
    local = F.avg_pool1d(y.unsqueeze(1), kernel_size=win, stride=1, padding=pad).squeeze(1)
    high = y - local
    d1 = y[:, 1:] - y[:, :-1]
    d2 = d1[:, 1:] - d1[:, :-1]
    d2 = F.pad(d2, (1, 1), value=0.0)
    peakness = high * F.relu(-d2)
    peakness = peakness / (peakness.abs().amax(dim=1, keepdim=True) + EPS)
    return torch.sigmoid(gain * peakness)


def masked_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    time_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    sample_weights = weights.view(-1, 1, 1)
    if time_weights is None:
        full_weights = sample_weights.expand_as(pred)
    else:
        if time_weights.ndim == 2:
            time_weights = time_weights[:, None, :]
        full_weights = (sample_weights * time_weights).expand_as(pred)
    denom = full_weights.sum() + EPS
    return (full_weights * (pred - target).abs()).sum() / denom


def masked_bce(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.view(-1, 1)
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    denom = weights.sum() * pred.shape[1] + EPS
    return (weights * bce).sum() / denom


def batch_abs_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    num = (x * y).mean(dim=-1)
    den = torch.sqrt((x.square().mean(dim=-1) * y.square().mean(dim=-1)).clamp_min(EPS))
    return (num / den).abs()


def masked_abs_corr(x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    corr = batch_abs_corr(x, y)
    return (corr * weights).sum() / (weights.sum() + EPS)


def derivative_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    time_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    pred_d = pred[:, :, 1:] - pred[:, :, :-1]
    target_d = target[:, :, 1:] - target[:, :, :-1]
    tw = None
    if time_weights is not None:
        if time_weights.ndim == 3:
            time_weights = time_weights[:, 0, :]
        tw = 0.5 * (time_weights[:, 1:] + time_weights[:, :-1])
    return masked_l1(pred_d, target_d, weights, tw)


def forward_clean(
    model: nn.Module,
    model_input: torch.Tensor,
    raw_norm: torch.Tensor,
    base_art_norm: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    artifact_hat = model(model_input)
    clean = raw_norm - artifact_hat
    return clean, artifact_hat


def _loss_dict_to_float(losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {key: float(value.detach().cpu().item()) for key, value in losses.items()}


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _set_module_frozen(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def _run_prior_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    desc: str,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    agg: Dict[str, float] = {"count": 0.0}
    context = torch.enable_grad() if train_mode else torch.no_grad()
    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)

    with context:
        for batch_idx, raw_norm in enumerate(progress, start=1):
            raw_norm = raw_norm.to(device)
            recon = model(raw_norm)
            l1 = F.l1_loss(recon, raw_norm)
            l2 = F.mse_loss(recon, raw_norm)
            total = l1 + 0.5 * l2

            if train_mode:
                optimizer.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            batch_size = float(raw_norm.size(0))
            agg["count"] += batch_size
            agg["recon_l1"] = agg.get("recon_l1", 0.0) + batch_size * float(l1.detach().cpu().item())
            agg["recon_l2"] = agg.get("recon_l2", 0.0) + batch_size * float(l2.detach().cpu().item())
            agg["total"] = agg.get("total", 0.0) + batch_size * float(total.detach().cpu().item())
            avg_total = agg["total"] / max(1.0, agg["count"])
            progress.set_postfix(batch=batch_idx, avg_total=f"{avg_total:.4f}")

    denom = max(1.0, agg.pop("count"))
    return {key: value / denom for key, value in agg.items()}


def train_clean_prior(
    train_records: Sequence[Dict[str, Any]],
    val_records: Sequence[Dict[str, Any]],
    window_cfg: WindowConfig,
    prior_cfg: PriorConfig,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-3,
    patience: int = 3,
    device: Optional[str] = None,
    max_windows_per_record: Optional[int] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    train_ds = CleanPriorWindowDataset(
        records=train_records,
        window_cfg=window_cfg,
        max_windows_per_record=max_windows_per_record,
    )
    val_ds = CleanPriorWindowDataset(
        records=val_records,
        window_cfg=window_cfg,
        max_windows_per_record=max_windows_per_record,
    )
    if len(train_ds) == 0:
        raise RuntimeError("Need sit windows to train the clean prior")

    dev = torch.device(device or "cpu")
    model = CleanPriorAutoencoder(in_channels=2, base_channels=prior_cfg.base_channels).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = None
    if len(val_ds) > 0:
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    best = {"val_total": float("inf"), "state_dict": None, "epoch": 0}
    history: List[Dict[str, float]] = []
    bad_epochs = 0
    prior_start = time.time()

    pbar = tqdm(range(1, epochs + 1), desc="Clean prior epochs", leave=True)
    for epoch in pbar:
        epoch_start = time.time()
        tr_metrics = _run_prior_epoch(model, train_loader, optimizer, dev, desc=f"prior train {epoch}/{epochs}")
        va_metrics = (
            _run_prior_epoch(model, val_loader, None, dev, desc=f"prior valid {epoch}/{epochs}")
            if val_loader is not None
            else dict(tr_metrics)
        )
        row = {
            "epoch": float(epoch),
            **{f"train_{k}": float(v) for k, v in tr_metrics.items()},
            **{f"val_{k}": float(v) for k, v in va_metrics.items()},
        }
        history.append(row)

        elapsed = time.time() - prior_start
        epoch_elapsed = time.time() - epoch_start
        avg_epoch = elapsed / max(1, epoch)
        eta = avg_epoch * max(0, epochs - epoch)
        pbar.set_postfix(
            train_total=f"{tr_metrics['total']:.4f}",
            val_total=f"{va_metrics['total']:.4f}",
            epoch_time=_format_seconds(epoch_elapsed),
            elapsed=_format_seconds(elapsed),
            eta=_format_seconds(eta),
        )

        if va_metrics["total"] < best["val_total"]:
            best = {"val_total": float(va_metrics["total"]), "state_dict": _cpu_state_dict(model), "epoch": epoch}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])
    _set_module_frozen(model)

    info = {
        "history": history,
        "best_epoch": int(best["epoch"]),
        "best_val_total": float(best["val_total"]),
        "num_train_windows": len(train_ds),
        "num_val_windows": len(val_ds),
    }
    return model, info


def compute_losses(
    clean_hat: torch.Tensor,
    artifact_hat: torch.Tensor,
    raw_norm: torch.Tensor,
    base_clean_norm: torch.Tensor,
    pseudo_clean_norm: torch.Tensor,
    pseudo_art_norm: torch.Tensor,
    sit_target_norm: torch.Tensor,
    pseudo_weight: torch.Tensor,
    peaks: torch.Tensor,
    motion_proxy: torch.Tensor,
    is_motion: torch.Tensor,
    fs: float,
    loss_cfg: LossConfig,
) -> Dict[str, torch.Tensor]:
    is_motion = is_motion.view(-1)
    is_sit = 1.0 - is_motion
    peaks_shifted = peaks
    peak_map = soft_peak_map(clean_hat[:, 1:2, :], fs=fs)
    target_clean = pseudo_clean_norm * is_motion[:, None, None] + sit_target_norm * is_sit[:, None, None]

    losses = {
        "artifact": masked_l1(artifact_hat, pseudo_art_norm, is_motion, pseudo_weight),
        "clean": masked_l1(clean_hat, pseudo_clean_norm, is_motion, pseudo_weight),
        "sit": masked_l1(clean_hat, sit_target_norm, is_sit),
        "peak": masked_bce(peak_map, peaks_shifted, 0.75 * is_motion + 0.25 * is_sit),
        "decorr": masked_abs_corr(clean_hat[:, 1, :], motion_proxy[:, 0, :], is_motion),
        "slope": derivative_l1(clean_hat, target_clean, is_motion + is_sit, pseudo_weight),
        "anchor": masked_l1(clean_hat, base_clean_norm, is_motion, pseudo_weight),
    }
    losses["total"] = (
        loss_cfg.artifact * losses["artifact"]
        + loss_cfg.clean * losses["clean"]
        + loss_cfg.sit * losses["sit"]
        + loss_cfg.peak * losses["peak"]
        + loss_cfg.decorr * losses["decorr"]
        + loss_cfg.slope * losses["slope"]
        + loss_cfg.anchor * losses["anchor"]
    )
    return losses


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    fs: float,
    loss_cfg: LossConfig,
    device: torch.device,
    desc: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    agg: Dict[str, float] = {"count": 0.0}
    context = torch.enable_grad() if train_mode else torch.no_grad()
    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)

    with context:
        for batch_idx, (
            model_input,
            raw_norm,
            base_clean_norm,
            base_art_norm,
            pseudo_clean_norm,
            pseudo_art_norm,
            sit_target_norm,
            pseudo_weight,
            peaks,
            motion_proxy,
            is_motion,
            _sids,
        ) in enumerate(progress, start=1):
            model_input = model_input.to(device)
            raw_norm = raw_norm.to(device)
            base_clean_norm = base_clean_norm.to(device)
            base_art_norm = base_art_norm.to(device)
            pseudo_clean_norm = pseudo_clean_norm.to(device)
            pseudo_art_norm = pseudo_art_norm.to(device)
            sit_target_norm = sit_target_norm.to(device)
            pseudo_weight = pseudo_weight.to(device)
            peaks = peaks.to(device)
            motion_proxy = motion_proxy.to(device)
            is_motion = is_motion.to(device)

            clean_hat, artifact_hat = forward_clean(model, model_input, raw_norm, base_art_norm)
            losses = compute_losses(
                clean_hat=clean_hat,
                artifact_hat=artifact_hat,
                raw_norm=raw_norm,
                base_clean_norm=base_clean_norm,
                pseudo_clean_norm=pseudo_clean_norm,
                pseudo_art_norm=pseudo_art_norm,
                sit_target_norm=sit_target_norm,
                pseudo_weight=pseudo_weight,
                peaks=peaks,
                motion_proxy=motion_proxy,
                is_motion=is_motion,
                fs=fs,
                loss_cfg=loss_cfg,
            )

            if train_mode:
                optimizer.zero_grad()
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            batch_size = float(model_input.size(0))
            agg["count"] += batch_size
            for key, value in _loss_dict_to_float(losses).items():
                agg[key] = agg.get(key, 0.0) + batch_size * value
            avg_total = agg.get("total", 0.0) / max(1.0, agg["count"])
            progress.set_postfix(batch=batch_idx, avg_total=f"{avg_total:.4f}")

    denom = max(1.0, agg.pop("count"))
    return {key: value / denom for key, value in agg.items()}


def train_hybrid_model(
    train_records: Sequence[Dict[str, Any]],
    val_records: Sequence[Dict[str, Any]],
    window_cfg: WindowConfig,
    baseline_cfg: BaselineConfig,
    model_cfg: ModelConfig,
    prior_cfg: PriorConfig,
    loss_cfg: LossConfig,
    epochs: int = 12,
    batch_size: int = 16,
    lr: float = 1e-3,
    patience: int = 4,
    prior_epochs: int = 8,
    prior_lr: float = 1e-3,
    prior_patience: int = 3,
    device: Optional[str] = None,
    max_windows_per_record: Optional[int] = None,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    del prior_epochs, prior_lr, prior_patience
    train_delay, train_default = estimate_delay_table(train_records, fs=window_cfg.fs)
    val_delay, val_default = estimate_delay_table(val_records, fs=window_cfg.fs)
    beat_library = build_sit_beat_library(
        records=train_records,
        fs=window_cfg.fs,
        prior_cfg=prior_cfg,
        delay_by_subject=train_delay,
        default_delay=train_default,
    )

    attach_record_targets(
        records=train_records,
        fs=window_cfg.fs,
        baseline_cfg=baseline_cfg,
        prior_cfg=prior_cfg,
        beat_library=beat_library,
        delay_by_subject=train_delay,
        default_delay=train_default,
        prefer_ecg=True,
    )
    attach_record_targets(
        records=val_records,
        fs=window_cfg.fs,
        baseline_cfg=baseline_cfg,
        prior_cfg=prior_cfg,
        beat_library=beat_library,
        delay_by_subject=val_delay,
        default_delay=val_default,
        prefer_ecg=True,
    )

    train_ds = HybridWindowDataset(
        records=train_records,
        window_cfg=window_cfg,
        baseline_cfg=baseline_cfg,
        model_cfg=model_cfg,
        max_windows_per_record=max_windows_per_record,
    )
    val_ds = HybridWindowDataset(
        records=val_records,
        window_cfg=window_cfg,
        baseline_cfg=baseline_cfg,
        model_cfg=model_cfg,
        max_windows_per_record=max_windows_per_record,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Need non-empty train and val datasets")

    dev = torch.device(device or "cpu")
    model = HybridArtifactRefiner(
        in_channels=model_cfg.in_channels,
        base_channels=model_cfg.base_channels,
    ).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    best = {"val_total": float("inf"), "state_dict": None, "epoch": 0}
    history: List[Dict[str, float]] = []
    bad_epochs = 0
    train_start = time.time()

    pbar = tqdm(range(1, epochs + 1), desc="Hybrid denoiser epochs", leave=True)
    for epoch in pbar:
        epoch_start = time.time()
        tr_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            fs=window_cfg.fs,
            loss_cfg=loss_cfg,
            device=dev,
            desc=f"train {epoch}/{epochs}",
            optimizer=optimizer,
        )
        va_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            fs=window_cfg.fs,
            loss_cfg=loss_cfg,
            device=dev,
            desc=f"valid {epoch}/{epochs}",
            optimizer=None,
        )
        row = {
            "epoch": float(epoch),
            **{f"train_{k}": float(v) for k, v in tr_metrics.items()},
            **{f"val_{k}": float(v) for k, v in va_metrics.items()},
        }
        history.append(row)

        elapsed = time.time() - train_start
        epoch_elapsed = time.time() - epoch_start
        avg_epoch = elapsed / max(1, epoch)
        eta = avg_epoch * max(0, epochs - epoch)
        pbar.set_postfix(
            train_total=f"{tr_metrics['total']:.4f}",
            val_total=f"{va_metrics['total']:.4f}",
            epoch_time=_format_seconds(epoch_elapsed),
            elapsed=_format_seconds(elapsed),
            eta=_format_seconds(eta),
        )

        if va_metrics["total"] < best["val_total"]:
            best = {"val_total": float(va_metrics["total"]), "state_dict": _cpu_state_dict(model), "epoch": epoch}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])

    info = {
        "history": history,
        "best_epoch": int(best["epoch"]),
        "best_val_total": float(best["val_total"]),
        "train_delay_by_subject": train_delay,
        "train_default_delay": float(train_default),
        "val_delay_by_subject": val_delay,
        "val_default_delay": float(val_default),
        "num_train_windows": len(train_ds),
        "num_val_windows": len(val_ds),
        "library_bins": int(beat_library["library_bins"]),
        "library_template_len": int(beat_library["template_len"]),
        "library_counts": [int(x) for x in np.asarray(beat_library["counts"]).tolist()],
    }
    return model, beat_library, info


def save_bundle(
    outdir: Path | str,
    model: nn.Module,
    beat_library: Dict[str, Any],
    window_cfg: WindowConfig,
    baseline_cfg: BaselineConfig,
    model_cfg: ModelConfig,
    prior_cfg: Optional[PriorConfig],
    loss_cfg: LossConfig,
    train_info: Dict[str, Any],
    splits: Dict[str, Any],
) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / "hybrid_denoiser.pt"

    payload = {
        "state_dict": _cpu_state_dict(model),
        "window_cfg": asdict(window_cfg),
        "baseline_cfg": {**asdict(baseline_cfg), "lags_ms": list(baseline_cfg.lags_ms)},
        "model_cfg": asdict(model_cfg),
        "prior_cfg": asdict(prior_cfg) if prior_cfg is not None else None,
        "beat_library": beat_library,
        "loss_cfg": asdict(loss_cfg),
        "train_info": train_info,
        "splits": splits,
        "ref_channel_names": REF_CHANNEL_NAMES,
    }
    torch.save(payload, model_path)

    meta = {
        "model_path": model_path.name,
        "window_cfg": payload["window_cfg"],
        "baseline_cfg": payload["baseline_cfg"],
        "model_cfg": payload["model_cfg"],
        "prior_cfg": payload["prior_cfg"],
        "loss_cfg": payload["loss_cfg"],
        "best_epoch": train_info["best_epoch"],
        "best_val_total": train_info["best_val_total"],
        "num_train_windows": train_info["num_train_windows"],
        "num_val_windows": train_info["num_val_windows"],
        "library_bins": train_info.get("library_bins"),
        "library_template_len": train_info.get("library_template_len"),
        "train_subjects": splits.get("train_subjects", []),
        "val_subjects": splits.get("val_subjects", []),
        "holdout_subjects": splits.get("holdout_subjects", []),
    }
    (outdir / "hybrid_denoiser_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (outdir / "train_history.json").write_text(json.dumps(train_info["history"], indent=2), encoding="utf-8")
    (outdir / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    (outdir / "delay_train_subjects.json").write_text(
        json.dumps(
            {
                "by_subject": train_info["train_delay_by_subject"],
                "default_delay": train_info["train_default_delay"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model_path


def export_bundle_to_onnx(
    model_path: Path | str,
    onnx_path: Optional[Path | str] = None,
    device: Optional[str] = None,
    opset_version: int = 17,
) -> Path:
    model_path = Path(model_path)
    bundle = load_bundle(model_path, device=device or "cpu")
    model: nn.Module = bundle["model"]
    model_cfg: ModelConfig = bundle["model_cfg"]
    window_cfg: WindowConfig = bundle["window_cfg"]

    win = int(round(window_cfg.fs * window_cfg.win_sec))
    dummy = torch.zeros(1, model_cfg.in_channels, win, dtype=torch.float32, device=bundle["device"])
    target_path = Path(onnx_path) if onnx_path is not None else (model_path.parent / "hybrid_denoiser.onnx")

    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(target_path),
            export_params=True,
            opset_version=int(opset_version),
            do_constant_folding=True,
            input_names=["model_input"],
            output_names=["artifact_hat"],
            dynamic_axes={
                "model_input": {0: "batch"},
                "artifact_hat": {0: "batch"},
            },
        )

    meta_path = model_path.parent / "hybrid_denoiser_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["onnx_path"] = target_path.name
        meta["onnx_opset_version"] = int(opset_version)
        meta["onnx_input_name"] = "model_input"
        meta["onnx_output_name"] = "artifact_hat"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    target_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "onnx_path": target_path.name,
                "pt_path": model_path.name,
                "window_cfg": asdict(window_cfg),
                "baseline_cfg": {**asdict(bundle["baseline_cfg"]), "lags_ms": list(bundle["baseline_cfg"].lags_ms)},
                "model_cfg": asdict(model_cfg),
                "opset_version": int(opset_version),
                "input_name": "model_input",
                "output_name": "artifact_hat",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return target_path


def load_bundle(model_path: Path | str, device: Optional[str] = None) -> Dict[str, Any]:
    model_path = Path(model_path)
    payload = torch.load(model_path, map_location=device or "cpu", weights_only=False)
    window_cfg = WindowConfig(**payload["window_cfg"])
    baseline_cfg = BaselineConfig.from_dict(payload["baseline_cfg"])
    model_cfg = ModelConfig.from_dict(payload["model_cfg"])
    model = HybridArtifactRefiner(
        in_channels=model_cfg.in_channels,
        base_channels=model_cfg.base_channels,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    dev = torch.device(device or "cpu")
    model.to(dev)
    return {
        "model": model,
        "window_cfg": window_cfg,
        "baseline_cfg": baseline_cfg,
        "model_cfg": model_cfg,
        "prior_cfg": PriorConfig(**payload["prior_cfg"]) if payload.get("prior_cfg") else None,
        "beat_library": payload.get("beat_library"),
        "loss_cfg": loss_config_from_dict(payload["loss_cfg"]),
        "train_info": payload["train_info"],
        "splits": payload["splits"],
        "device": dev,
        "ref_channel_names": payload.get("ref_channel_names", REF_CHANNEL_NAMES),
    }


def denoise_record(
    rec: Dict[str, Any],
    bundle: Dict[str, Any],
    motion_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    model = bundle["model"]
    window_cfg: WindowConfig = bundle["window_cfg"]
    baseline_cfg: BaselineConfig = bundle["baseline_cfg"]
    model_cfg: ModelConfig = bundle["model_cfg"]
    device: torch.device = bundle["device"]

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
        return {
            "raw": raw_full,
            "baseline": raw_full.copy(),
            "denoised": raw_full.copy(),
            "windows": [],
        }

    for start, end in windows:
        sample = prepare_window_sample(
            rec,
            start,
            end,
            fs=window_cfg.fs,
            baseline_cfg=baseline_cfg,
            model_cfg=model_cfg,
        )
        model_input = torch.from_numpy(sample["model_input"][None, ...]).to(device)
        raw_norm = torch.from_numpy(sample["raw_norm"][None, ...]).to(device)
        base_art_norm = torch.from_numpy(sample["base_art_norm"][None, ...]).to(device)
        with torch.no_grad():
            clean_norm, _ = forward_clean(model, model_input, raw_norm, base_art_norm)
        clean_norm_np = clean_norm.squeeze(0).detach().cpu().numpy().astype(np.float32)
        center = sample["center"]
        scale = sample["scale"]
        clean_win = clean_norm_np * scale + center
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
        "raw": raw_full,
        "baseline": baseline.astype(np.float32),
        "denoised": denoised.astype(np.float32),
        "windows": windows,
    }
