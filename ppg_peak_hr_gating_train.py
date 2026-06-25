from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-codex"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

EPS = 1e-6
PTT_NAME = "pulse_transit_time_ppg"
SIM_NAME = "simultaneous_measurements"
IAM_NAME = "iamwell"
MIMIC_NAME = "mimic_perform"
VITALDB_NAME = "vitaldb_open"
DEFAULT_RESULTS_ROOT = ".CNN_results"
MIMIC_DEFAULT_SUBSETS = ("mimic_perform_train_all_csv", "mimic_perform_test_all_csv")
MIMIC_EXTRA_HOLDOUT_SUBSETS = ("mimic_perform_af_csv", "mimic_perform_non_af_csv")
MIMIC_EXTRA_HOLDOUT_MAT_FILES = ("MIMIC_PERform_1_min_noisy.mat", "MIMIC_PERform_1_min_neonate.mat")
PPG_EVENT_TOLERANCE_SECS = (0.010, 0.020, 0.030, 0.040)
PPG_MAIN_EVENT_TOLERANCE_SEC = 0.020


@dataclass
class WindowConfig:
    fs: float = 256.0
    win_sec: float = 8.0
    hop_sec: float = 2.0


@dataclass
class LossConfig:
    peak: float = 1.0
    beat: float = 0.10
    ibi: float = 0.35
    gate: float = 0.25
    domain: float = 0.0
    worst_domain: float = 0.25
    fs: float = 256.0
    ibi_huber_delta: float = 0.08
    peak_timing_radius_sec: float = 0.120


@dataclass
class ModelConfig:
    in_channels: int = 1
    base_channels: int = 32
    norm_type: str = "instance"
    ibi_min_sec: float = 0.30
    ibi_max_sec: float = 2.00
    num_domains: int = 5


@dataclass
class AugmentConfig:
    enabled: bool = True
    amplitude_min: float = 0.70
    amplitude_max: float = 1.30
    noise_std: float = 0.025
    drift_std: float = 0.050
    dropout_prob: float = 0.15
    dropout_max_frac: float = 0.08
    respiration_mod_prob: float = 0.35
    respiration_mod_depth: float = 0.12
    motion_burst_prob: float = 0.25
    motion_burst_std: float = 0.18
    motion_burst_max_frac: float = 0.20
    clip_prob: float = 0.12
    clip_quantile: float = 0.96
    lowpass_prob: float = 0.20
    lowpass_min_hz: float = 3.5
    lowpass_max_hz: float = 8.0
    polarity_flip_prob: float = 0.05
    time_warp_prob: float = 0.15
    time_warp_max_frac: float = 0.04
    target_jitter_sec: float = 0.020


@dataclass
class DetectorConfig:
    in_channels: int = 10
    base_channels: int = 24
    fs: float = 256.0
    win_sec: float = 8.0
    hop_sec: float = 2.0


DOMAIN_TO_IDX = {PTT_NAME: 0, SIM_NAME: 1, IAM_NAME: 2, MIMIC_NAME: 3, VITALDB_NAME: 4}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hh, rem = divmod(seconds, 3600)
    mm, ss = divmod(rem, 60)
    if hh:
        return f"{hh:d}:{mm:02d}:{ss:02d}"
    return f"{mm:02d}:{ss:02d}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_timestamped_run_dir(results_root: Path, run_name: str) -> Tuple[Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d-%H")
    suffix = "" if run_name in ("", "auto") else f"_{run_name}"
    base_name = f"{timestamp}{suffix}"
    candidate = results_root / base_name
    if not candidate.exists():
        return ensure_dir(candidate), base_name
    idx = 2
    while True:
        alt_name = f"{base_name}_{idx:02d}"
        candidate = results_root / alt_name
        if not candidate.exists():
            return ensure_dir(candidate), alt_name
        idx += 1


def resample_signal(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if abs(float(fs_in) - float(fs_out)) < 1e-6:
        return x.astype(np.float32, copy=True)
    frac = Fraction(str(float(fs_out) / float(fs_in))).limit_denominator(1000)
    y = signal.resample_poly(x, frac.numerator, frac.denominator)
    return y.astype(np.float32)


def resample_peak_indices(peaks: np.ndarray, fs_in: float, fs_out: float, n_out: int) -> np.ndarray:
    peaks = np.asarray(peaks, dtype=np.int64)
    if peaks.size == 0:
        return peaks.astype(np.int32)
    scaled = np.round(peaks.astype(np.float64) * float(fs_out) / float(fs_in)).astype(np.int64)
    scaled = scaled[(scaled >= 0) & (scaled < int(n_out))]
    return np.unique(scaled).astype(np.int32)


def bandpass_filter(x: np.ndarray, fs: float, low: float, high: float, order: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1 or len(x) < max(15, 3 * (order + 1)):
        return x.astype(np.float32, copy=True)
    hi = min(float(high), 0.45 * float(fs))
    lo = max(0.05, float(low))
    if hi <= lo:
        return x.astype(np.float32, copy=True)
    b, a = signal.butter(order, [lo / (fs / 2.0), hi / (fs / 2.0)], btype="band")
    return signal.filtfilt(b, a, x).astype(np.float32)


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return ((x - np.mean(x)) / (np.std(x) + EPS)).astype(np.float32)


def standardize_channels(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = np.mean(x, axis=1, keepdims=True)
    sd = np.std(x, axis=1, keepdims=True) + EPS
    return ((x - mu) / sd).astype(np.float32)


def lowpass_filter_nd(x: np.ndarray, fs: float, cutoff_hz: float, order: int = 2, axis: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape[axis] < max(15, 3 * (order + 1)):
        return x.astype(np.float32, copy=True)
    hi = min(float(cutoff_hz), 0.45 * float(fs))
    if hi <= 0:
        return x.astype(np.float32, copy=True)
    b, a = signal.butter(order, hi / (float(fs) / 2.0), btype="low")
    return signal.filtfilt(b, a, x, axis=axis).astype(np.float32)


def vector_magnitude(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt(np.sum(np.asarray(x, dtype=np.float32) ** 2, axis=axis)).astype(np.float32)


def remove_gravity(acc: np.ndarray, fs: float, cutoff_hz: float = 0.3) -> np.ndarray:
    acc = np.asarray(acc, dtype=np.float32)
    gravity = lowpass_filter_nd(acc, fs=fs, cutoff_hz=cutoff_hz, axis=0)
    return (acc - gravity).astype(np.float32)


def infer_acc_to_ms2(acc: np.ndarray) -> Tuple[np.ndarray, str]:
    acc = np.asarray(acc, dtype=np.float32)
    mag = vector_magnitude(acc, axis=1)
    med = float(np.nanmedian(np.abs(mag))) if mag.size else 0.0
    if 6.0 <= med <= 14.0:
        return acc.astype(np.float32), "m/s^2 inferred from median magnitude near gravity"
    if 0.5 <= med <= 2.5:
        return (acc * 9.80665).astype(np.float32), "g converted to m/s^2"
    if 500.0 <= med <= 1500.0:
        return (acc / 1000.0 * 9.80665).astype(np.float32), "milli-g converted to m/s^2"
    return acc.astype(np.float32), "raw source scale retained"


def maybe_gyro_to_rad(gyro: np.ndarray) -> Tuple[np.ndarray, str]:
    gyro = np.asarray(gyro, dtype=np.float32)
    med = float(np.nanmedian(np.abs(gyro))) if gyro.size else 0.0
    if med > 6.5:
        return (gyro * np.pi / 180.0).astype(np.float32), "deg/s converted to rad/s"
    return gyro.astype(np.float32), "raw or rad/s scale retained"


def split_windows(n: int, fs: float, win_sec: float, hop_sec: float) -> Iterable[Tuple[int, int]]:
    win = int(round(float(win_sec) * float(fs)))
    hop = int(round(float(hop_sec) * float(fs)))
    if win <= 0 or hop <= 0 or n < win:
        return
    for start in range(0, n - win + 1, hop):
        yield start, start + win


def pick_first_available(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    colset = set(columns)
    for name in candidates:
        if name in colset:
            return name
    return None


def detect_ecg_rpeaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    ecg = np.asarray(ecg, dtype=np.float32)
    if ecg.size < int(max(128, fs)):
        return np.zeros(0, dtype=np.int32)
    y = bandpass_filter(ecg, fs=fs, low=5.0, high=min(25.0, 0.45 * fs), order=2)
    d = np.diff(y, prepend=y[:1])
    energy = d * d
    win = max(3, int(round(0.120 * fs)))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    env = np.convolve(energy, kernel, mode="same")
    distance = max(1, int(round(0.30 * fs)))
    prominence = 0.30 * float(np.std(env) + EPS)
    peaks, props = signal.find_peaks(env, distance=distance, prominence=prominence)
    if peaks.size == 0:
        return np.zeros(0, dtype=np.int32)

    radius = max(1, int(round(0.060 * fs)))
    refined_with_scores: Dict[int, float] = {}
    prominences = props.get("prominences", np.ones(len(peaks), dtype=np.float32))
    radius = max(1, int(round(0.060 * fs)))
    for peak, score in zip(peaks, prominences):
        lo = max(0, int(peak) - radius)
        hi = min(len(y), int(peak) + radius + 1)
        if hi <= lo:
            continue
        # Energy-envelope peaks are robust for candidate detection, but exact
        # timing is better anchored on the original ECG R deflection.
        idx = int(lo + np.argmax(ecg[lo:hi]))
        refined_with_scores[idx] = max(float(score), refined_with_scores.get(idx, 0.0))
    if not refined_with_scores:
        return np.zeros(0, dtype=np.int32)

    refined = sorted(refined_with_scores)
    scores = [refined_with_scores[idx] for idx in refined]
    if len(refined) >= 3:
        rr = np.diff(np.asarray(refined, dtype=np.float32)) / float(fs)
        median_rr = float(np.median(rr))
        min_sep_sec = 0.36 if median_rr >= 0.55 else max(0.24, 0.65 * median_rr)
        min_sep = max(1, int(round(min_sep_sec * float(fs))))
        changed = True
        while changed and len(refined) >= 2:
            changed = False
            i = 0
            while i < len(refined) - 1:
                if refined[i + 1] - refined[i] < min_sep:
                    remove_i = i if scores[i] < scores[i + 1] else i + 1
                    del refined[remove_i]
                    del scores[remove_i]
                    changed = True
                    i = max(0, remove_i - 1)
                    continue
                i += 1
    return np.asarray(refined, dtype=np.int32)


def build_peak_target(n: int, peak_idx: np.ndarray, sigma_samples: float) -> np.ndarray:
    target = np.zeros(n, dtype=np.float32)
    peak_idx = np.asarray(peak_idx, dtype=np.int32)
    if peak_idx.size == 0:
        return target
    radius = max(1, int(round(3.0 * sigma_samples)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / max(sigma_samples, 1.0)) ** 2).astype(np.float32)
    kernel /= float(np.max(kernel) + EPS)
    for peak in peak_idx:
        lo = max(0, int(peak) - radius)
        hi = min(n, int(peak) + radius + 1)
        klo = radius - (int(peak) - lo)
        khi = radius + (hi - int(peak))
        target[lo:hi] = np.maximum(target[lo:hi], kernel[klo:khi])
    return target.astype(np.float32)


def build_rr_track(n: int, peak_idx: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    track = np.zeros(n, dtype=np.float32)
    mask = np.zeros(n, dtype=np.float32)
    peak_idx = np.asarray(peak_idx, dtype=np.int32)
    if peak_idx.size < 2:
        return track, mask
    for p0, p1 in zip(peak_idx[:-1], peak_idx[1:]):
        if p1 <= p0:
            continue
        rr = float(p1 - p0) / float(fs)
        track[p0:p1] = rr
        mask[p0:p1] = 1.0
    first_rr = float(peak_idx[1] - peak_idx[0]) / float(fs)
    last_rr = float(peak_idx[-1] - peak_idx[-2]) / float(fs)
    track[: peak_idx[0]] = first_rr
    mask[: peak_idx[0]] = 1.0
    track[peak_idx[-1] :] = last_rr
    mask[peak_idx[-1] :] = 1.0
    return track.astype(np.float32), mask.astype(np.float32)


def read_wfdb_format16(record_path_no_ext: Path) -> Tuple[np.ndarray, List[str], float]:
    hea_path = record_path_no_ext.with_suffix(".hea")
    dat_path = record_path_no_ext.with_suffix(".dat")
    lines = hea_path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = lines[0].split()
    nsig = int(header[1])
    fs = float(header[2])
    channel_defs: List[Tuple[str, float, float]] = []
    for line in lines[1 : 1 + nsig]:
        parts = line.split()
        desc = " ".join(parts[8:]) if len(parts) > 8 else parts[-1]
        gain_base = parts[2]
        if "(" in gain_base:
            gain = float(gain_base.split("(")[0])
            base = float(gain_base.split("(")[1].split(")")[0])
        else:
            gain = float(gain_base)
            base = 0.0
        channel_defs.append((desc, gain, base))
    raw = np.fromfile(dat_path, dtype="<i2")
    raw = raw.reshape(-1, nsig)
    out = np.zeros_like(raw, dtype=np.float32)
    for idx, (_, gain, base) in enumerate(channel_defs):
        denom = gain if abs(gain) > EPS else 1.0
        out[:, idx] = ((raw[:, idx].astype(np.float32) - float(base)) / float(denom)).astype(np.float32)
    return out, [x[0] for x in channel_defs], fs


def _wfdb_skip_interval(raw4: bytes) -> int:
    # WFDB stores SKIP intervals in a byte order that is different from normal
    # little-endian integers. This matches the generated_data/*.aux markers.
    val = (raw4[0] << 16) | (raw4[1] << 24) | raw4[2] | (raw4[3] << 8)
    if val >= 2**31:
        val -= 2**32
    return int(val)


def read_wfdb_annotations(path: Path) -> List[Dict[str, Any]]:
    data = path.read_bytes()
    out: List[Dict[str, Any]] = []
    pos = 0
    sample = 0
    while pos + 2 <= len(data):
        b0 = data[pos]
        b1 = data[pos + 1]
        pos += 2
        ann_code = b1 >> 2
        interval = b0 + 256 * (b1 & 0x03)
        if ann_code == 0 and interval == 0:
            break
        if ann_code == 59 and pos + 4 <= len(data):
            sample += _wfdb_skip_interval(data[pos : pos + 4])
            pos += 4
            continue
        sample += interval
        aux = None
        if ann_code == 63:
            n = int(interval)
            aux = data[pos : pos + n].split(b"\0")[0].decode("latin1", errors="replace")
            pos += n + (n % 2)
        out.append({"sample": int(sample), "code": int(ann_code), "aux": aux})
    return out


def read_wfdb_beat_peaks(atr_path: Path, n_samples: int) -> np.ndarray:
    peaks = [
        int(item["sample"])
        for item in read_wfdb_annotations(atr_path)
        if int(item["code"]) == 1 and 0 <= int(item["sample"]) < int(n_samples)
    ]
    return np.asarray(sorted(set(peaks)), dtype=np.int32)


def activity_label_to_gate(label: str) -> Optional[float]:
    text = label.lower()
    if "walking" in text or "running" in text or "run" in text:
        return 1.0
    if "rest" in text or "2-back" in text or "2 back" in text or "standing" in text:
        return 0.0
    return None


def gate_track_from_aux_markers(aux_path: Path, n_samples: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    gate = np.full(int(n_samples), -1.0, dtype=np.float32)
    markers: List[Dict[str, Any]] = []
    for item in read_wfdb_annotations(aux_path):
        text = item.get("aux")
        if not text or str(text).startswith("##"):
            continue
        value = activity_label_to_gate(str(text))
        if value is None:
            continue
        sample = max(0, min(int(item["sample"]), int(n_samples)))
        markers.append({"sample": sample, "label": str(text), "gate": float(value)})
    markers = sorted(markers, key=lambda x: int(x["sample"]))
    for idx, marker in enumerate(markers):
        start = int(marker["sample"])
        end = int(markers[idx + 1]["sample"]) if idx + 1 < len(markers) else int(n_samples)
        if end > start:
            gate[start:end] = float(marker["gate"])
    return gate, markers


def inspect_new_datasets(data_root: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}

    ptt_csv = next((data_root / "pulse-transit-time-ppg" / "1.1.0" / "csv").glob("s*_*.csv"), None)
    if ptt_csv is not None and ptt_csv.exists():
        df = pd.read_csv(ptt_csv, nrows=3)
        summary[PTT_NAME] = {
            "sample_record": ptt_csv.name,
            "fs": 500.0,
            "columns": list(df.columns),
            "first_3_rows": df.to_dict(orient="records"),
            "files": [str(ptt_csv)],
        }

    sim_hea = data_root / "simultaneous-measurements" / "1.0.0" / "generated_data" / "x001.hea"
    sim_vals, sim_cols, sim_fs = read_wfdb_format16(sim_hea.with_suffix(""))
    summary[SIM_NAME] = {
        "sample_record": "x001",
        "fs": sim_fs,
        "columns": sim_cols,
        "first_3_rows": [
            {sim_cols[j]: float(sim_vals[i, j]) for j in range(sim_vals.shape[1])}
            for i in range(min(3, sim_vals.shape[0]))
        ],
        "files": [
            str(sim_hea.with_suffix(".hea")),
            str(sim_hea.with_suffix(".dat")),
            str(sim_hea.with_suffix(".atr")),
            str(sim_hea.with_suffix(".aux")),
        ],
    }

    iam_mat = data_root / "iAMwell Dataset - Intelligent Athlete Monitoring for Cardiovascular Wellness" / "A07.mat"
    iam = loadmat(iam_mat)
    labels = [str(x).strip() for x in iam["labels"].reshape(-1)]
    data = np.asarray(iam["data"], dtype=np.float32)
    summary[IAM_NAME] = {
        "sample_record": "A07",
        "fs": float(1000.0 / float(iam["isi"][0, 0])),
        "columns": labels,
        "first_3_rows": [
            {labels[j]: float(data[i, j]) for j in range(data.shape[1])}
            for i in range(min(3, data.shape[0]))
        ],
        "files": [str(iam_mat), str(iam_mat.with_suffix(".acq"))],
    }

    mimic_csv = data_root / "MIMIC" / "mimic_perform_train_all_csv" / "mimic_perform_train_all_001_data.csv"
    mimic_fix = mimic_csv.with_name(mimic_csv.name.replace("_data.csv", "_fix.txt"))
    if mimic_csv.exists():
        df = pd.read_csv(mimic_csv, nrows=3)
        summary[MIMIC_NAME] = {
            "sample_record": mimic_csv.stem.replace("_data", ""),
            "fs": 125.0,
            "columns": list(df.columns),
            "first_3_rows": df.to_dict(orient="records"),
            "metadata": parse_mimic_fix_metadata(mimic_fix) if mimic_fix.exists() else {},
            "files": [str(mimic_csv), str(mimic_fix)],
        }

    vitaldb_dir = data_root / "VitalDB"
    vitaldb_docs = sorted(vitaldb_dir.glob("*.docx"))
    if vitaldb_docs:
        summary[VITALDB_NAME] = {
            "sample_record": "open_dataset_api",
            "fs": 500.0,
            "columns": ["SNUADC/PLETH", "SNUADC/ECG_II"],
            "first_3_rows": [],
            "metadata": {
                "docx_files": [p.name for p in vitaldb_docs],
                "api_tracks": ["PLETH", "ECG_II"],
                "device_tracks": ["SNUADC/PLETH", "SNUADC/ECG_II"],
                "python_package": "vitaldb",
                "notes": "Use vitaldb.find_cases(['PLETH', 'ECG_II']) and load via VitalFile/load_case.",
            },
            "files": [str(p) for p in vitaldb_docs],
        }
    return summary


def print_dataset_inspection(summary: Dict[str, Any]) -> None:
    for name, info in summary.items():
        print(f"\n[{name}] sample={info['sample_record']} fs={info['fs']}")
        print("columns:")
        print(info["columns"])
        print("first_3_rows:")
        for row in info["first_3_rows"]:
            print(row)


def constant_gate_track(n: int, value: float) -> np.ndarray:
    return np.full(n, float(value), dtype=np.float32)


def unknown_gate_track(n: int) -> np.ndarray:
    return np.full(n, -1.0, dtype=np.float32)


def approximate_simultaneous_gate_track(n: int, fs: float) -> np.ndarray:
    gate = np.full(n, -1.0, dtype=np.float32)
    seg = int(round(5.0 * 60.0 * float(fs)))
    ranges = [
        (0, seg, 0.0),
        (seg, 2 * seg, 1.0),
        (2 * seg, 3 * seg, 0.0),
        (3 * seg, 4 * seg, 1.0),
    ]
    for start, end, value in ranges:
        if start >= n:
            break
        gate[start : min(end, n)] = float(value)
    return gate


def load_pulse_transit_time_ppg(data_root: Path, target_fs: float) -> List[Dict[str, Any]]:
    csv_dir = data_root / "pulse-transit-time-ppg" / "1.1.0" / "csv"
    records: List[Dict[str, Any]] = []
    for path in sorted(csv_dir.glob("s*_*.csv")):
        df = pd.read_csv(path)
        ppg_col = pick_first_available(df.columns, ("pleth_2", "pleth_1", "IR", "PPG"))
        ecg_col = pick_first_available(df.columns, ("ecg", "ECG"))
        peaks_col = pick_first_available(df.columns, ("peaks", "peak", "rpeaks"))
        if ppg_col is None or ecg_col is None or peaks_col is None:
            continue
        ppg = pd.to_numeric(df[ppg_col], errors="coerce").to_numpy(dtype=np.float32)
        ecg = pd.to_numeric(df[ecg_col], errors="coerce").to_numpy(dtype=np.float32)
        peaks = np.where(pd.to_numeric(df[peaks_col], errors="coerce").fillna(0).to_numpy(dtype=np.float32) > 0.5)[0]
        fs_in = 500.0
        ppg_rs = resample_signal(ppg, fs_in=fs_in, fs_out=target_fs)
        ecg_rs = resample_signal(ecg, fs_in=fs_in, fs_out=target_fs)
        peaks_rs = resample_peak_indices(peaks, fs_in=fs_in, fs_out=target_fs, n_out=len(ppg_rs))
        rr_track, rr_mask = build_rr_track(len(ppg_rs), peaks_rs, fs=target_fs)
        peak_target = build_peak_target(len(ppg_rs), peaks_rs, sigma_samples=0.030 * target_fs)
        subject, activity = path.stem.split("_", 1)
        records.append(
            {
                "dataset": PTT_NAME,
                "subject": f"{PTT_NAME}:{subject}",
                "record": path.name,
                "activity": activity,
                "gate_track": constant_gate_track(len(ppg_rs), 0.0 if activity == "sit" else 1.0),
                "gate_label_source": "explicit_activity",
                "ppg": zscore_1d(ppg_rs),
                "ecg": zscore_1d(ecg_rs),
                "peak_idx": peaks_rs.astype(np.int32),
                "peak_target": peak_target,
                "rr_track": rr_track,
                "rr_mask": rr_mask,
                "fs": float(target_fs),
            }
        )
    return records


def load_simultaneous_measurements(data_root: Path, target_fs: float) -> List[Dict[str, Any]]:
    base = data_root / "simultaneous-measurements" / "1.0.0" / "generated_data"
    records: List[Dict[str, Any]] = []
    for hea_path in sorted(base.glob("x*.hea")):
        values, columns, fs_in = read_wfdb_format16(hea_path.with_suffix(""))
        ppg_idx = next((i for i, name in enumerate(columns) if name == "SOT/Pleth"), None)
        ecg_idx = next((i for i, name in enumerate(columns) if name == "SOT/EKG_filtered"), None)
        if ppg_idx is None or ecg_idx is None:
            continue
        ppg = values[:, ppg_idx]
        ecg = values[:, ecg_idx]
        ppg_rs = resample_signal(ppg, fs_in=fs_in, fs_out=target_fs)
        ecg_rs = resample_signal(ecg, fs_in=fs_in, fs_out=target_fs)
        atr_path = hea_path.with_suffix(".atr")
        aux_path = hea_path.with_suffix(".aux")
        if atr_path.exists():
            peaks = read_wfdb_beat_peaks(atr_path, n_samples=len(ppg))
            peaks_rs = resample_peak_indices(peaks, fs_in=fs_in, fs_out=target_fs, n_out=len(ppg_rs))
            peak_source = "atr_consensus_annotations"
        else:
            peaks_rs = detect_ecg_rpeaks(ecg_rs, fs=target_fs)
            peak_source = "ecg_rpeak_detector"
        rr_track, rr_mask = build_rr_track(len(ppg_rs), peaks_rs, fs=target_fs)
        peak_target = build_peak_target(len(ppg_rs), peaks_rs, sigma_samples=0.030 * target_fs)
        if aux_path.exists():
            gate_src, aux_markers = gate_track_from_aux_markers(aux_path, n_samples=len(ppg))
            gate_track = resample_signal(gate_src, fs_in=fs_in, fs_out=target_fs)
            gate_track = np.where(gate_track >= 0.5, 1.0, np.where(gate_track >= 0.0, 0.0, -1.0)).astype(np.float32)
            gate_label_source = "aux_annotations"
        else:
            gate_track = approximate_simultaneous_gate_track(len(ppg_rs), fs=target_fs)
            aux_markers = []
            gate_label_source = "protocol_approx_5min_blocks"
        records.append(
            {
                "dataset": SIM_NAME,
                "subject": f"{SIM_NAME}:{hea_path.stem}",
                "record": hea_path.stem,
                "activity": "protocol_mixed",
                "gate_track": gate_track,
                "gate_label_source": gate_label_source,
                "aux_markers": aux_markers,
                "ppg": zscore_1d(ppg_rs),
                "ecg": zscore_1d(ecg_rs),
                "peak_idx": peaks_rs.astype(np.int32),
                "peak_source": peak_source,
                "peak_target": peak_target,
                "rr_track": rr_track,
                "rr_mask": rr_mask,
                "fs": float(target_fs),
            }
        )
    return records


def load_iamwell(data_root: Path, target_fs: float) -> List[Dict[str, Any]]:
    base = data_root / "iAMwell Dataset - Intelligent Athlete Monitoring for Cardiovascular Wellness"
    records: List[Dict[str, Any]] = []
    for mat_path in sorted(base.glob("*.mat")):
        d = loadmat(mat_path)
        labels = [str(x).strip() for x in d["labels"].reshape(-1)]
        data = np.asarray(d["data"], dtype=np.float32)
        fs_in = 1000.0 / float(d["isi"][0, 0])
        ppg_idx = next((i for i, label in enumerate(labels) if label.startswith("PPG")), None)
        ecg_idx = next((i for i, label in enumerate(labels) if label.startswith("ECG")), None)
        if ppg_idx is None or ecg_idx is None:
            continue
        ppg = data[:, ppg_idx]
        ecg = data[:, ecg_idx]
        ppg_rs = resample_signal(ppg, fs_in=fs_in, fs_out=target_fs)
        ecg_rs = resample_signal(ecg, fs_in=fs_in, fs_out=target_fs)
        peaks_rs = detect_ecg_rpeaks(ecg_rs, fs=target_fs)
        rr_track, rr_mask = build_rr_track(len(ppg_rs), peaks_rs, fs=target_fs)
        peak_target = build_peak_target(len(ppg_rs), peaks_rs, sigma_samples=0.030 * target_fs)
        records.append(
            {
                "dataset": IAM_NAME,
                "subject": f"{IAM_NAME}:{mat_path.stem}",
                "record": mat_path.name,
                "activity": "protocol_mixed",
                "gate_track": unknown_gate_track(len(ppg_rs)),
                "gate_label_source": "unlabeled",
                "ppg": zscore_1d(ppg_rs),
                "ecg": zscore_1d(ecg_rs),
                "peak_idx": peaks_rs.astype(np.int32),
                "peak_target": peak_target,
                "rr_track": rr_track,
                "rr_mask": rr_mask,
                "fs": float(target_fs),
            }
        )
    return records


def parse_mimic_fix_metadata(path: Path) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    if not path.exists():
        return meta
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if lines:
        meta["record"] = lines[0].strip()
    for line in lines[1:]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip().lower().replace(" ", "_")] = value.strip()
    return meta


def _mat_scalar_text(x: Any) -> str:
    arr = np.asarray(x)
    while arr.dtype == object and arr.size == 1:
        arr = np.asarray(arr.reshape(-1)[0])
    if arr.size == 0:
        return ""
    if arr.dtype.kind in "US":
        return str(arr.reshape(-1)[0]).strip()
    return str(arr.reshape(-1)[0]).strip()


def _mat_wave_vector_and_fs(record: Any, field_name: str) -> Tuple[np.ndarray, float]:
    obj = record[field_name]
    if not isinstance(obj, np.ndarray) or obj.size == 0 or obj.dtype.names is None:
        return np.zeros(0, dtype=np.float32), 0.0
    item = obj.reshape(-1)[0]
    vals = np.asarray(item["v"], dtype=np.float32).reshape(-1)
    fs = float(np.asarray(item["fs"]).reshape(-1)[0])
    return vals, fs


def _mimic_mat_metadata(record: Any, source_file: str) -> Dict[str, str]:
    meta: Dict[str, str] = {"source_file": source_file}
    if "fix" not in record.dtype.names:
        return meta
    fix = record["fix"]
    if not isinstance(fix, np.ndarray) or fix.size == 0 or fix.dtype.names is None:
        return meta
    item = fix.reshape(-1)[0]
    for field_name in item.dtype.names or ():
        meta[field_name] = _mat_scalar_text(item[field_name])
    return meta


def make_pseudo_labeled_ppg_ecg_record(
    dataset_name: str,
    subject: str,
    record_name: str,
    activity: str,
    ppg: np.ndarray,
    ecg: np.ndarray,
    fs_in: float,
    target_fs: float,
    subset: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    ppg = np.asarray(ppg, dtype=np.float32)
    ecg = np.asarray(ecg, dtype=np.float32)
    if ppg.size == 0 or ecg.size == 0:
        return None
    n = min(ppg.size, ecg.size)
    ppg = ppg[:n]
    ecg = ecg[:n]
    valid = np.isfinite(ppg) & np.isfinite(ecg)
    if not np.any(valid):
        return None
    if not np.all(valid):
        ppg = pd.Series(ppg).interpolate(limit_direction="both").to_numpy(dtype=np.float32)
        ecg = pd.Series(ecg).interpolate(limit_direction="both").to_numpy(dtype=np.float32)
    ppg_rs = resample_signal(ppg, fs_in=fs_in, fs_out=target_fs)
    ecg_rs = resample_signal(ecg, fs_in=fs_in, fs_out=target_fs)
    peaks_rs = detect_ecg_rpeaks(ecg_rs, fs=target_fs)
    if peaks_rs.size < 3:
        return None
    rr_track, rr_mask = build_rr_track(len(ppg_rs), peaks_rs, fs=target_fs)
    peak_target = build_peak_target(len(ppg_rs), peaks_rs, sigma_samples=0.030 * target_fs)
    return {
        "dataset": dataset_name,
        "subject": subject,
        "record": record_name,
        "activity": activity,
        "subset": subset,
        "gate_track": unknown_gate_track(len(ppg_rs)),
        "gate_label_source": "unlabeled",
        "ppg": zscore_1d(ppg_rs),
        "ecg": zscore_1d(ecg_rs),
        "peak_idx": peaks_rs.astype(np.int32),
        "peak_source": "ecg_rpeak_detector_pseudo",
        "peak_target": peak_target,
        "rr_track": rr_track,
        "rr_mask": rr_mask,
        "fs": float(target_fs),
        "metadata": metadata or {},
    }


def load_mimic_perform_csv(
    data_root: Path,
    target_fs: float,
    subsets: Sequence[str] = MIMIC_DEFAULT_SUBSETS,
    max_records: int = 0,
) -> List[Dict[str, Any]]:
    base = data_root / "MIMIC"
    records: List[Dict[str, Any]] = []
    loaded = 0
    for subset in subsets:
        subset_dir = base / str(subset)
        if not subset_dir.exists():
            continue
        for csv_path in sorted(subset_dir.glob("*_data.csv")):
            if max_records > 0 and loaded >= int(max_records):
                return records
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if not {"Time", "PPG", "ECG"}.issubset(df.columns):
                continue
            ppg = pd.to_numeric(df["PPG"], errors="coerce").to_numpy(dtype=np.float32)
            ecg = pd.to_numeric(df["ECG"], errors="coerce").to_numpy(dtype=np.float32)
            time_vals = pd.to_numeric(df["Time"], errors="coerce").to_numpy(dtype=np.float64)
            if len(time_vals) > 2 and np.all(np.isfinite(time_vals[: min(32, len(time_vals))])):
                dt = float(np.nanmedian(np.diff(time_vals[: min(256, len(time_vals))])))
                fs_in = 1.0 / dt if dt > EPS else 125.0
            else:
                fs_in = 125.0
            fix_path = csv_path.with_name(csv_path.name.replace("_data.csv", "_fix.txt"))
            meta = parse_mimic_fix_metadata(fix_path)
            subject_id = meta.get("original_subject_id") or meta.get("record") or csv_path.stem.replace("_data", "")
            group = meta.get("group", "").strip()
            record = make_pseudo_labeled_ppg_ecg_record(
                dataset_name=MIMIC_NAME,
                subject=f"{MIMIC_NAME}:{subject_id}",
                record_name=csv_path.name,
                activity=f"clinical_{group}" if group else "clinical_unknown",
                ppg=ppg,
                ecg=ecg,
                fs_in=fs_in,
                target_fs=target_fs,
                subset=str(subset),
                metadata=meta,
            )
            if record is None:
                continue
            loaded += 1
            records.append(record)
    return records


def load_mimic_perform_mat_records(
    data_root: Path,
    target_fs: float,
    mat_files: Sequence[str] = MIMIC_EXTRA_HOLDOUT_MAT_FILES,
) -> List[Dict[str, Any]]:
    base = data_root / "MIMIC"
    records: List[Dict[str, Any]] = []
    for filename in mat_files:
        path = base / str(filename)
        if not path.exists():
            continue
        try:
            data = loadmat(path)["data"].reshape(-1)
        except Exception:
            continue
        for rec_idx, item in enumerate(data):
            if item.dtype.names is None or "ppg" not in item.dtype.names or "ekg" not in item.dtype.names:
                continue
            ppg, ppg_fs = _mat_wave_vector_and_fs(item, "ppg")
            ecg, ecg_fs = _mat_wave_vector_and_fs(item, "ekg")
            if ppg_fs <= 0 or ecg_fs <= 0:
                continue
            if abs(ppg_fs - ecg_fs) > 1e-6:
                ecg = resample_signal(ecg, fs_in=ecg_fs, fs_out=ppg_fs)
            meta = _mimic_mat_metadata(item, source_file=path.name)
            subject_id = meta.get("subj_id") or meta.get("rec_id") or f"{path.stem}_{rec_idx + 1}"
            group = meta.get("group") or meta.get("af_status") or "special"
            record = make_pseudo_labeled_ppg_ecg_record(
                dataset_name=MIMIC_NAME,
                subject=f"{MIMIC_NAME}:extra:{subject_id}",
                record_name=f"{path.name}:{rec_idx + 1}",
                activity=f"clinical_extra_{group}",
                ppg=ppg,
                ecg=ecg,
                fs_in=ppg_fs,
                target_fs=target_fs,
                subset=path.stem,
                metadata=meta,
            )
            if record is not None:
                records.append(record)
    return records


def load_mimic_special_extra_holdout(
    data_root: Path,
    target_fs: float,
    subsets: Sequence[str] = MIMIC_EXTRA_HOLDOUT_SUBSETS,
    mat_files: Sequence[str] = MIMIC_EXTRA_HOLDOUT_MAT_FILES,
    max_records: int = 0,
) -> List[Dict[str, Any]]:
    csv_records = load_mimic_perform_csv(data_root, target_fs=target_fs, subsets=subsets, max_records=max_records)
    mat_records = load_mimic_perform_mat_records(data_root, target_fs=target_fs, mat_files=mat_files)
    out = csv_records + mat_records
    if max_records > 0:
        out = out[: int(max_records)]
    return out


def _interpolate_nan_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    valid = np.isfinite(x)
    if not np.any(valid):
        return x
    if np.all(valid):
        return x.astype(np.float32, copy=True)
    idx = np.arange(x.size)
    return np.interp(idx, idx[valid], x[valid]).astype(np.float32)


def load_vitaldb_open_dataset(
    target_fs: float,
    max_cases: int = 20,
    maxlen_sec: float = 600.0,
) -> List[Dict[str, Any]]:
    try:
        import vitaldb  # type: ignore
    except Exception as exc:
        print(f"[vitaldb] skipped: python package is unavailable ({exc})")
        return []

    short_tracks = ["PLETH", "ECG_II"]
    device_tracks = ["SNUADC/PLETH", "SNUADC/ECG_II"]
    try:
        caseids = list(vitaldb.find_cases(short_tracks))
    except Exception as exc:
        print(f"[vitaldb] skipped: find_cases failed ({exc})")
        return []
    if max_cases > 0:
        caseids = caseids[: int(max_cases)]

    records: List[Dict[str, Any]] = []
    interval = 1.0 / float(target_fs)
    for caseid in tqdm(caseids, desc="load VitalDB", leave=False, dynamic_ncols=True):
        vals: Optional[np.ndarray] = None
        try:
            vf = vitaldb.VitalFile(int(caseid), device_tracks, maxlen=float(maxlen_sec), interval=interval)
            vals = np.asarray(vf.to_numpy(device_tracks, interval), dtype=np.float32)
            if maxlen_sec > 0:
                vals = vals[: int(round(float(maxlen_sec) * float(target_fs)))]
        except Exception:
            try:
                vals = np.asarray(vitaldb.load_case(int(caseid), short_tracks, interval), dtype=np.float32)
                if maxlen_sec > 0:
                    vals = vals[: int(round(float(maxlen_sec) * float(target_fs)))]
            except Exception:
                vals = None
        if vals is None or vals.ndim != 2 or vals.shape[1] < 2 or vals.shape[0] < int(10 * target_fs):
            continue
        ppg = _interpolate_nan_1d(vals[:, 0])
        ecg = _interpolate_nan_1d(vals[:, 1])
        if not np.any(np.isfinite(ppg)) or not np.any(np.isfinite(ecg)):
            continue
        peaks = detect_ecg_rpeaks(ecg, fs=target_fs)
        if peaks.size < 3:
            continue
        rr_track, rr_mask = build_rr_track(len(ppg), peaks, fs=target_fs)
        peak_target = build_peak_target(len(ppg), peaks, sigma_samples=0.030 * target_fs)
        records.append(
            {
                "dataset": VITALDB_NAME,
                "subject": f"{VITALDB_NAME}:{caseid}",
                "record": f"case_{caseid}",
                "activity": "surgical_open_dataset",
                "gate_track": unknown_gate_track(len(ppg)),
                "gate_label_source": "unlabeled",
                "ppg": zscore_1d(ppg),
                "ecg": zscore_1d(ecg),
                "peak_idx": peaks.astype(np.int32),
                "peak_source": "ecg_rpeak_detector_pseudo",
                "peak_target": peak_target,
                "rr_track": rr_track,
                "rr_mask": rr_mask,
                "fs": float(target_fs),
                "metadata": {"caseid": int(caseid), "tracks": device_tracks, "maxlen_sec": float(maxlen_sec)},
            }
        )
    return records


def load_all_datasets(
    data_root: Path,
    target_fs: float,
    mimic_subsets: Sequence[str] = MIMIC_DEFAULT_SUBSETS,
    mimic_max_records: int = 0,
    enable_vitaldb: bool = False,
    vitaldb_max_cases: int = 20,
    vitaldb_maxlen_sec: float = 600.0,
) -> Dict[str, List[Dict[str, Any]]]:
    vitaldb_records = (
        load_vitaldb_open_dataset(target_fs=target_fs, max_cases=vitaldb_max_cases, maxlen_sec=vitaldb_maxlen_sec)
        if enable_vitaldb
        else []
    )
    return {
        PTT_NAME: load_pulse_transit_time_ppg(data_root, target_fs=target_fs),
        SIM_NAME: load_simultaneous_measurements(data_root, target_fs=target_fs),
        IAM_NAME: load_iamwell(data_root, target_fs=target_fs),
        MIMIC_NAME: load_mimic_perform_csv(
            data_root,
            target_fs=target_fs,
            subsets=mimic_subsets,
            max_records=mimic_max_records,
        ),
        VITALDB_NAME: vitaldb_records,
    }


def make_detector_feature_matrix(
    ppg: np.ndarray,
    acc: np.ndarray,
    gyro: Optional[np.ndarray],
    fs: float,
) -> Tuple[np.ndarray, Dict[str, str]]:
    ppg = bandpass_filter(np.asarray(ppg, dtype=np.float32), fs=fs, low=0.5, high=min(8.0, 0.45 * fs), order=2)
    acc_ms2, acc_unit = infer_acc_to_ms2(np.asarray(acc, dtype=np.float32))
    acc_dyn = remove_gravity(acc_ms2, fs=fs, cutoff_hz=0.3)
    if gyro is None:
        gyro_rad = np.zeros_like(acc_dyn, dtype=np.float32)
        gyro_unit = "missing; zero-filled"
    else:
        gyro_rad, gyro_unit = maybe_gyro_to_rad(np.asarray(gyro, dtype=np.float32))
        if gyro_rad.shape[0] != acc_dyn.shape[0]:
            n = min(gyro_rad.shape[0], acc_dyn.shape[0])
            gyro_rad = gyro_rad[:n]
            acc_dyn = acc_dyn[:n]
            ppg = ppg[:n]
    n = min(len(ppg), acc_dyn.shape[0], gyro_rad.shape[0])
    ppg = ppg[:n]
    acc_dyn = acc_dyn[:n]
    gyro_rad = gyro_rad[:n]
    acc_mag = vector_magnitude(acc_dyn, axis=1)
    gyro_mag = vector_magnitude(gyro_rad, axis=1)
    jerk = np.gradient(acc_dyn, axis=0) * float(fs)
    jerk_mag = vector_magnitude(jerk, axis=1)
    features = np.vstack(
        [
            zscore_1d(ppg)[None, :],
            acc_dyn.T,
            gyro_rad.T,
            acc_mag[None, :],
            gyro_mag[None, :],
            jerk_mag[None, :],
        ]
    ).astype(np.float32)
    return standardize_channels(features), {"acc_unit": acc_unit, "gyro_unit": gyro_unit}


def load_ptt_detector_records(data_root: Path, target_fs: float) -> List[Dict[str, Any]]:
    csv_dir = data_root / "pulse-transit-time-ppg" / "1.1.0" / "csv"
    records: List[Dict[str, Any]] = []
    for path in sorted(csv_dir.glob("s*_*.csv")):
        df = pd.read_csv(path)
        required = {"a_x", "a_y", "a_z", "g_x", "g_y", "g_z"}
        if not required.issubset(df.columns):
            continue
        ppg_col = pick_first_available(df.columns, ("pleth_2", "pleth_1", "IR", "PPG"))
        if ppg_col is None:
            continue
        fs_in = 500.0
        ppg = resample_signal(pd.to_numeric(df[ppg_col], errors="coerce").to_numpy(dtype=np.float32), fs_in=fs_in, fs_out=target_fs)
        acc = np.column_stack(
            [
                resample_signal(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32), fs_in=fs_in, fs_out=target_fs)
                for col in ("a_x", "a_y", "a_z")
            ]
        )
        gyro = np.column_stack(
            [
                resample_signal(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32), fs_in=fs_in, fs_out=target_fs)
                for col in ("g_x", "g_y", "g_z")
            ]
        )
        n = min(len(ppg), acc.shape[0], gyro.shape[0])
        features, unit_info = make_detector_feature_matrix(ppg[:n], acc[:n], gyro[:n], fs=target_fs)
        subject, activity = path.stem.split("_", 1)
        label = 0.0 if activity == "sit" else 1.0
        records.append(
            {
                "dataset": PTT_NAME,
                "subject": f"{PTT_NAME}:{subject}",
                "record": path.name,
                "activity": activity,
                "features": features,
                "gate_track": constant_gate_track(features.shape[1], label),
                "label_source": "explicit_activity",
                "unit_info": unit_info,
            }
        )
    return records


def _pick_sim_acc(values: np.ndarray, columns: Sequence[str]) -> Tuple[Optional[np.ndarray], Dict[str, str]]:
    faros = ["FAROS/Accelerometer_X", "FAROS/Accelerometer_Y", "FAROS/Accelerometer_Z"]
    hexo = ["HEXOSKIN/acceleration_X", "HEXOSKIN/acceleration_Y", "HEXOSKIN/acceleration_Z"]
    col_to_idx = {name: idx for idx, name in enumerate(columns)}
    if all(name in col_to_idx for name in faros):
        acc = np.column_stack([values[:, col_to_idx[name]] for name in faros]).astype(np.float32)
        return acc, {"source": "FAROS/Accelerometer_*"}
    if all(name in col_to_idx for name in hexo):
        acc = np.column_stack([values[:, col_to_idx[name]] for name in hexo]).astype(np.float32)
        return acc, {"source": "HEXOSKIN/acceleration_*"}
    return None, {"source": "missing"}


def load_sim_detector_records(data_root: Path, target_fs: float) -> List[Dict[str, Any]]:
    base = data_root / "simultaneous-measurements" / "1.0.0" / "generated_data"
    records: List[Dict[str, Any]] = []
    for hea_path in sorted(base.glob("x*.hea")):
        values, columns, fs_in = read_wfdb_format16(hea_path.with_suffix(""))
        ppg_idx = next((i for i, name in enumerate(columns) if name == "SOT/Pleth"), None)
        if ppg_idx is None:
            continue
        acc, acc_meta = _pick_sim_acc(values, columns)
        if acc is None:
            continue
        ppg = resample_signal(values[:, ppg_idx], fs_in=fs_in, fs_out=target_fs)
        acc_rs = np.column_stack([resample_signal(acc[:, i], fs_in=fs_in, fs_out=target_fs) for i in range(3)])
        n = min(len(ppg), acc_rs.shape[0])
        features, unit_info = make_detector_feature_matrix(ppg[:n], acc_rs[:n], gyro=None, fs=target_fs)
        aux_path = hea_path.with_suffix(".aux")
        if aux_path.exists():
            gate_src, aux_markers = gate_track_from_aux_markers(aux_path, n_samples=len(values))
            gate_track = resample_signal(gate_src, fs_in=fs_in, fs_out=target_fs)
            gate_track = np.where(gate_track >= 0.5, 1.0, np.where(gate_track >= 0.0, 0.0, -1.0)).astype(np.float32)[: features.shape[1]]
            label_source = "aux_annotations"
        else:
            gate_track = approximate_simultaneous_gate_track(features.shape[1], fs=target_fs)
            aux_markers = []
            label_source = "protocol_approx_5min_blocks"
        unit_info.update(acc_meta)
        records.append(
            {
                "dataset": SIM_NAME,
                "subject": f"{SIM_NAME}:{hea_path.stem}",
                "record": hea_path.stem,
                "activity": "protocol_mixed",
                "features": features,
                "gate_track": gate_track,
                "label_source": label_source,
                "aux_markers": aux_markers,
                "unit_info": unit_info,
            }
        )
    return records


def load_motion_detector_records(data_root: Path, target_fs: float) -> Dict[str, List[Dict[str, Any]]]:
    return {
        PTT_NAME: load_ptt_detector_records(data_root, target_fs=target_fs),
        SIM_NAME: load_sim_detector_records(data_root, target_fs=target_fs),
    }


def build_loaded_dataset_summary(dataset_records: Dict[str, List[Dict[str, Any]]], inspect_summary: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"inspection": inspect_summary, "datasets": {}}
    for name, recs in dataset_records.items():
        out["datasets"][name] = {
            "record_count": int(len(recs)),
            "subject_count": int(len({rec["subject"] for rec in recs})),
            "gate_labeled_records": int(sum(np.any(rec["gate_track"] >= 0.0) for rec in recs)),
            "gate_label_sources": sorted({rec["gate_label_source"] for rec in recs}),
        }
    return out


def _looks_like_imu_column(name: str) -> bool:
    text = str(name).lower()
    compact = text.replace(" ", "").replace("-", "_")
    if compact in {"a_x", "a_y", "a_z", "g_x", "g_y", "g_z", "ax", "ay", "az", "gx", "gy", "gz"}:
        return True
    return any(token in text for token in ("acc", "accelerometer", "acceleration", "gyro", "imu"))


def infer_imu_availability(inspect_summary: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for dataset, info in inspect_summary.items():
        columns = [str(x) for x in info.get("columns", [])]
        imu_cols = [col for col in columns if _looks_like_imu_column(col)]
        unit_note = "not_available"
        gravity_note = "not_applicable"
        if imu_cols:
            joined = " ".join(imu_cols).lower()
            if any(col in {"a_x", "a_y", "a_z"} for col in imu_cols):
                unit_note = "PTT a_x/a_y/a_z appear near m/s^2 scale; raw gravity component is present unless removed upstream"
            elif any(col in {"g_x", "g_y", "g_z"} for col in imu_cols):
                unit_note = "PTT g_x/g_y/g_z gyroscope scale is source-specific; not used by this script"
            elif "hexoskin" in joined:
                unit_note = "likely g for HEXOSKIN acceleration channels"
            elif "faros" in joined:
                unit_note = "unknown FAROS accelerometer scale from WFDB header conversion"
            else:
                unit_note = "unknown; raw source-specific acceleration unit"
            gravity_note = "not standardized in this PPG-only training script"
        out[dataset] = {
            "raw_imu_columns_available": bool(imu_cols),
            "imu_like_columns": imu_cols,
            "in_loaded_model_record": False,
            "unit_status": unit_note,
            "gravity_removal_status": gravity_note,
        }
    return out


def build_gate_input_audit(
    model_cfg: ModelConfig,
    inspect_summary: Dict[str, Any],
    train_records: Sequence[Dict[str, Any]],
    holdout_records: Sequence[Dict[str, Any]],
    extra_holdout_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    def split_sources(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "records": int(len(records)),
            "datasets": sorted({str(rec.get("dataset", "unknown")) for rec in records}),
            "gate_label_sources": sorted({str(rec.get("gate_label_source", "unknown")) for rec in records}),
            "records_with_loaded_imu_key": int(sum("imu" in rec or "acc" in rec or "accelerometer" in rec for rec in records)),
        }

    return {
        "model_input_channels": int(model_cfg.in_channels),
        "model_input_names": ["ppg"],
        "gate_head_input": "shared PPG encoder mid-level features",
        "uses_imu_in_gate_head": False,
        "uses_imu_in_peak_or_ibi_heads": False,
        "important_note": "This script trains a PPG-only gate. It should not be interpreted as the IMU-based static/dynamic detector.",
        "raw_imu_availability": infer_imu_availability(inspect_summary),
        "split_input_audit": {
            "train": split_sources(train_records),
            "holdout": split_sources(holdout_records),
            "extra_holdout": split_sources(extra_holdout_records),
        },
    }


def split_subjects(subjects: Sequence[str], train_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    subjects = sorted(set(subjects))
    if len(subjects) < 2:
        return list(subjects), []
    rng = np.random.RandomState(seed)
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    n_train = int(round(len(shuffled) * train_ratio))
    n_train = max(1, min(len(shuffled) - 1, n_train))
    return sorted(shuffled[:n_train]), sorted(shuffled[n_train:])


def group_kfold_subject_splits(subjects: Sequence[str], n_splits: int, seed: int) -> List[Tuple[List[str], List[str]]]:
    subjects = sorted(set(subjects))
    if len(subjects) < 2:
        return [(subjects, [])]
    n_splits = max(2, min(int(n_splits), len(subjects)))
    rng = np.random.RandomState(seed)
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    fold_bins: List[List[str]] = [[] for _ in range(n_splits)]
    for idx, subject in enumerate(shuffled):
        fold_bins[idx % n_splits].append(subject)
    splits: List[Tuple[List[str], List[str]]] = []
    for fold_idx in range(n_splits):
        val_subjects = sorted(fold_bins[fold_idx])
        train_subjects = sorted([sid for j, bucket in enumerate(fold_bins) if j != fold_idx for sid in bucket])
        if train_subjects and val_subjects:
            splits.append((train_subjects, val_subjects))
    return splits


def _shift_1d_zero_fill(x: np.ndarray, shift: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if shift == 0 or x.size == 0:
        return x.astype(np.float32, copy=True)
    out = np.zeros_like(x, dtype=np.float32)
    if shift > 0:
        out[shift:] = x[:-shift]
    else:
        out[:shift] = x[-shift:]
    return out


def augment_ppg_window(ppg: np.ndarray, cfg: AugmentConfig, fs: float) -> np.ndarray:
    x = np.asarray(ppg, dtype=np.float32).copy()
    if x.size == 0:
        return x
    scale = np.random.uniform(cfg.amplitude_min, cfg.amplitude_max)
    x *= float(scale)
    if np.random.rand() < cfg.polarity_flip_prob:
        x *= -1.0
    if np.random.rand() < cfg.time_warp_prob and cfg.time_warp_max_frac > 0:
        warp = float(np.random.uniform(1.0 - cfg.time_warp_max_frac, 1.0 + cfg.time_warp_max_frac))
        n_warp = max(8, int(round(x.size * warp)))
        warped = signal.resample(x, n_warp).astype(np.float32)
        x = signal.resample(warped, x.size).astype(np.float32)
    if np.random.rand() < cfg.respiration_mod_prob and cfg.respiration_mod_depth > 0:
        phase = np.random.uniform(0.0, 2.0 * np.pi)
        resp_hz = np.random.uniform(0.12, 0.35)
        t = np.arange(x.size, dtype=np.float32) / max(float(fs), EPS)
        mod = 1.0 + np.random.uniform(0.0, cfg.respiration_mod_depth) * np.sin(2.0 * np.pi * resp_hz * t + phase)
        x *= mod.astype(np.float32)
    if cfg.drift_std > 0:
        phase = np.random.uniform(0.0, 2.0 * np.pi)
        cycles = np.random.uniform(0.15, 0.80)
        drift = np.sin(np.linspace(phase, phase + 2.0 * np.pi * cycles, x.size, dtype=np.float32))
        x += np.float32(np.random.normal(0.0, cfg.drift_std)) * drift
    if np.random.rand() < cfg.motion_burst_prob and cfg.motion_burst_std > 0:
        width = max(2, int(round(np.random.uniform(0.03, cfg.motion_burst_max_frac) * x.size)))
        start = np.random.randint(0, max(1, x.size - width + 1))
        burst = np.random.normal(0.0, cfg.motion_burst_std, size=width).astype(np.float32)
        kernel = max(3, int(round(0.040 * float(fs))))
        if kernel % 2 == 0:
            kernel += 1
        burst = signal.savgol_filter(burst, window_length=min(kernel, width if width % 2 else width - 1), polyorder=1).astype(np.float32) if width >= 5 else burst
        x[start : start + width] += burst
    if cfg.noise_std > 0:
        x += np.random.normal(0.0, cfg.noise_std, size=x.shape).astype(np.float32)
    if np.random.rand() < cfg.dropout_prob and cfg.dropout_max_frac > 0:
        width = max(1, int(round(np.random.uniform(0.01, cfg.dropout_max_frac) * x.size)))
        start = np.random.randint(0, max(1, x.size - width + 1))
        fill = float(np.median(x[max(0, start - 8) : start])) if start > 0 else float(np.median(x))
        x[start : start + width] = fill
    if np.random.rand() < cfg.clip_prob:
        q = min(0.999, max(0.50, float(cfg.clip_quantile)))
        lo = float(np.quantile(x, 1.0 - q))
        hi = float(np.quantile(x, q))
        if hi > lo:
            x = np.clip(x, lo, hi).astype(np.float32)
    if np.random.rand() < cfg.lowpass_prob:
        cutoff = float(np.random.uniform(cfg.lowpass_min_hz, cfg.lowpass_max_hz))
        hi = min(cutoff, 0.45 * float(fs))
        if hi > 0.5 and x.size > int(3 * fs / hi):
            b, a = signal.butter(2, hi / (float(fs) / 2.0), btype="low")
            x = signal.filtfilt(b, a, x).astype(np.float32)
    return x.astype(np.float32)


class MultiDatasetPeakWindowDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, Any]], window_cfg: WindowConfig, augment_cfg: Optional[AugmentConfig] = None):
        self.records = list(records)
        self.window_cfg = window_cfg
        self.augment_cfg = augment_cfg if augment_cfg and augment_cfg.enabled else None
        self.index: List[Tuple[int, int, int]] = []
        for rid, rec in enumerate(self.records):
            for start, end in split_windows(len(rec["ppg"]), window_cfg.fs, window_cfg.win_sec, window_cfg.hop_sec):
                self.index.append((rid, start, end))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        rid, start, end = self.index[idx]
        rec = self.records[rid]
        ppg = rec["ppg"][start:end]
        if self.augment_cfg is not None:
            ppg = augment_ppg_window(ppg, self.augment_cfg, fs=float(self.window_cfg.fs))
        peak_target = rec["peak_target"][start:end]
        if self.augment_cfg is not None and self.augment_cfg.target_jitter_sec > 0:
            max_shift = int(round(float(self.augment_cfg.target_jitter_sec) * float(self.window_cfg.fs)))
            if max_shift > 0:
                peak_target = _shift_1d_zero_fill(peak_target, int(np.random.randint(-max_shift, max_shift + 1)))
        rr_track = rec["rr_track"][start:end]
        rr_mask = rec["rr_mask"][start:end]
        gate_window = rec["gate_track"][start:end]
        valid = gate_window >= 0.0
        if np.any(valid):
            mean_gate = float(np.mean(gate_window[valid]))
            if mean_gate <= 0.05:
                gate_target = 0.0
            elif mean_gate >= 0.95:
                gate_target = 1.0
            else:
                gate_target = -1.0
        else:
            gate_target = -1.0
        return (
            torch.from_numpy(ppg[None, :].astype(np.float32)),
            torch.from_numpy(peak_target.astype(np.float32)),
            torch.from_numpy(rr_track.astype(np.float32)),
            torch.from_numpy(rr_mask.astype(np.float32)),
            torch.tensor(gate_target, dtype=torch.float32),
            torch.tensor(DOMAIN_TO_IDX.get(rec["dataset"], 0), dtype=torch.long),
            rec["subject"],
            rec["dataset"],
            rec.get("activity", "unknown"),
            rec.get("record", "unknown"),
            rec.get("subset", rec["dataset"]),
        )


def make_norm_1d(channels: int, norm_type: str) -> nn.Module:
    if norm_type == "instance":
        return nn.InstanceNorm1d(channels, affine=True, eps=1e-5)
    return nn.GroupNorm(1, channels)


class GradientReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradientReverseFn.apply(x, float(lambd))


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm_type: str = "instance"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
            make_norm_1d(out_ch, norm_type),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2),
            make_norm_1d(out_ch, norm_type),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PeakIntervalGateNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        norm_type: str = "instance",
        ibi_min_sec: float = 0.30,
        ibi_max_sec: float = 2.00,
        num_domains: int = 3,
    ):
        super().__init__()
        b = base_channels
        self.ibi_min_sec = float(ibi_min_sec)
        self.ibi_max_sec = float(ibi_max_sec)
        self.grl_lambda = 0.0
        self.enc1 = ConvBlock1D(in_channels, b, norm_type=norm_type)
        self.down1 = nn.Sequential(nn.AvgPool1d(2), ConvBlock1D(b, 2 * b, norm_type=norm_type))
        self.down2 = nn.Sequential(nn.AvgPool1d(2), ConvBlock1D(2 * b, 4 * b, norm_type=norm_type))
        self.mid = ConvBlock1D(4 * b, 4 * b, norm_type=norm_type)
        self.up1 = ConvBlock1D(4 * b + 2 * b, 2 * b, norm_type=norm_type)
        self.up2 = ConvBlock1D(2 * b + b, b, norm_type=norm_type)
        self.peak_head = nn.Conv1d(b, 1, kernel_size=1)
        self.ibi_head = nn.Conv1d(b, 1, kernel_size=1)
        self.gate_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(4 * b, 2 * b),
            nn.GELU(),
            nn.Linear(2 * b, 1),
        )
        self.domain_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(4 * b, 2 * b),
            nn.GELU(),
            nn.Linear(2 * b, int(num_domains)),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        mid = self.mid(e3)
        gate_logit = self.gate_head(mid)
        domain_logit = self.domain_head(grad_reverse(mid, self.grl_lambda))

        u1 = F.interpolate(mid, size=e2.shape[-1], mode="linear", align_corners=False)
        u1 = self.up1(torch.cat([u1, e2], dim=1))
        u2 = F.interpolate(u1, size=e1.shape[-1], mode="linear", align_corners=False)
        u2 = self.up2(torch.cat([u2, e1], dim=1))
        ibi_raw = self.ibi_head(u2).squeeze(1)
        ibi_pred = self.ibi_min_sec + (self.ibi_max_sec - self.ibi_min_sec) * torch.sigmoid(ibi_raw)
        return {
            "peak_logit": self.peak_head(u2).squeeze(1),
            "ibi_pred": ibi_pred,
            "gate_logit": gate_logit.squeeze(1),
            "domain_logit": domain_logit,
        }


class PeakIntervalGateOnnxWrapper(nn.Module):
    def __init__(self, model: PeakIntervalGateNet):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = self.model(x)
        return pred["peak_logit"], pred["ibi_pred"], pred["gate_logit"]


class MotionDetectorWindowDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, Any]], detector_cfg: DetectorConfig, augment_cfg: Optional[AugmentConfig] = None):
        self.records = list(records)
        self.detector_cfg = detector_cfg
        self.augment_cfg = augment_cfg if augment_cfg and augment_cfg.enabled else None
        self.index: List[Tuple[int, int, int, float]] = []
        for rid, rec in enumerate(self.records):
            n = int(rec["features"].shape[1])
            for start, end in split_windows(n, detector_cfg.fs, detector_cfg.win_sec, detector_cfg.hop_sec):
                gate_window = np.asarray(rec["gate_track"][start:end], dtype=np.float32)
                valid = gate_window >= 0.0
                if not np.any(valid):
                    continue
                mean_gate = float(np.mean(gate_window[valid]))
                if mean_gate <= 0.05:
                    label = 0.0
                elif mean_gate >= 0.95:
                    label = 1.0
                else:
                    continue
                self.index.append((rid, start, end, label))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        rid, start, end, label = self.index[idx]
        rec = self.records[rid]
        x = np.asarray(rec["features"][:, start:end], dtype=np.float32).copy()
        if self.augment_cfg is not None:
            x[0] = augment_ppg_window(x[0], self.augment_cfg, fs=float(self.detector_cfg.fs))
        return (
            torch.from_numpy(standardize_channels(x).astype(np.float32)),
            torch.tensor(float(label), dtype=torch.float32),
            str(rec.get("dataset", "unknown")),
            str(rec.get("subject", "unknown")),
            str(rec.get("activity", "unknown")),
            str(rec.get("record", "unknown")),
        )


def make_detector_balanced_sampler(ds: MotionDetectorWindowDataset) -> WeightedRandomSampler:
    counts: Dict[Tuple[str, int], int] = {}
    subject_counts: Dict[str, int] = {}
    keys: List[Tuple[str, int, str]] = []
    for rid, _, _, label in ds.index:
        rec = ds.records[rid]
        dkey = str(rec.get("dataset", "unknown"))
        y = int(label >= 0.5)
        skey = str(rec.get("subject", "unknown"))
        keys.append((dkey, y, skey))
        counts[(dkey, y)] = counts.get((dkey, y), 0) + 1
        subject_counts[skey] = subject_counts.get(skey, 0) + 1
    weights = [
        1.0 / max(1, counts[(dkey, y)]) * 1.0 / np.sqrt(max(1, subject_counts[skey]))
        for dkey, y, skey in keys
    ]
    weights_np = np.asarray(weights, dtype=np.float64)
    weights_np = weights_np / max(float(np.mean(weights_np)), EPS)
    return WeightedRandomSampler(torch.as_tensor(weights_np, dtype=torch.double), num_samples=len(ds), replacement=True)


def build_detector_loader(
    records: Sequence[Dict[str, Any]],
    detector_cfg: DetectorConfig,
    batch_size: int,
    shuffle: bool,
    augment_cfg: Optional[AugmentConfig] = None,
    balanced_sampling: bool = True,
) -> Tuple[MotionDetectorWindowDataset, DataLoader]:
    ds = MotionDetectorWindowDataset(records, detector_cfg=detector_cfg, augment_cfg=augment_cfg)
    sampler = make_detector_balanced_sampler(ds) if balanced_sampling and len(ds) > 0 else None
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle if sampler is None else False, sampler=sampler, drop_last=False)
    return ds, loader


class DenoiserEncoderMotionDetector(nn.Module):
    def __init__(self, in_channels: int = 10, base_channels: int = 24):
        super().__init__()
        b = int(base_channels)
        self.enc1 = ConvBlock1D(in_channels, b)
        self.down1 = nn.Sequential(nn.AvgPool1d(2), ConvBlock1D(b, 2 * b))
        self.down2 = nn.Sequential(nn.AvgPool1d(2), ConvBlock1D(2 * b, 4 * b))
        self.mid = ConvBlock1D(4 * b, 4 * b)
        self.motion_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(4 * b, 2 * b),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(2 * b, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        mid = self.mid(e3)
        return self.motion_head(mid).squeeze(1)


class LightCnnMotionDetector(nn.Module):
    def __init__(self, in_channels: int = 10, base_channels: int = 16):
        super().__init__()
        b = int(base_channels)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, b, kernel_size=9, padding=4),
            nn.GroupNorm(1, b),
            nn.GELU(),
            nn.AvgPool1d(2),
            nn.Conv1d(b, 2 * b, kernel_size=7, padding=3),
            nn.GroupNorm(1, 2 * b),
            nn.GELU(),
            nn.AvgPool1d(2),
            nn.Conv1d(2 * b, 2 * b, kernel_size=5, padding=2),
            nn.GroupNorm(1, 2 * b),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(2 * b, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def soft_peak_timing_loss(logits: torch.Tensor, target: torch.Tensor, fs: float, radius_sec: float) -> torch.Tensor:
    if logits.numel() == 0:
        return torch.zeros((), device=logits.device)
    radius = max(1, int(round(float(radius_sec) * float(fs))))
    losses: List[torch.Tensor] = []
    positions = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
    with torch.no_grad():
        core = target[:, 1:-1]
        is_peak = (core >= target[:, :-2]) & (core >= target[:, 2:]) & (core > 0.98)
        peak_batches, peak_pos = torch.where(is_peak)
        peak_pos = peak_pos + 1
    for batch_idx, center in zip(peak_batches.tolist(), peak_pos.tolist()):
        lo = max(0, int(center) - radius)
        hi = min(logits.shape[-1], int(center) + radius + 1)
        local_logits = logits[batch_idx, lo:hi]
        if local_logits.numel() < 2:
            continue
        weights = torch.softmax(local_logits * 4.0, dim=0)
        pred_center = torch.sum(weights * positions[lo:hi])
        losses.append(torch.abs(pred_center - float(center)) / float(fs))
    if not losses:
        return torch.zeros((), device=logits.device)
    return torch.stack(losses).mean()


def compute_losses(
    pred: Dict[str, torch.Tensor],
    peak_target: torch.Tensor,
    rr_track: torch.Tensor,
    rr_mask: torch.Tensor,
    gate_target: torch.Tensor,
    domain_target: Optional[torch.Tensor],
    loss_cfg: LossConfig,
) -> Dict[str, torch.Tensor]:
    pos_weight = torch.tensor(8.0, device=peak_target.device)
    peak_loss = F.binary_cross_entropy_with_logits(pred["peak_logit"], peak_target, pos_weight=pos_weight)
    beat_loss = soft_peak_timing_loss(
        pred["peak_logit"],
        peak_target,
        fs=loss_cfg.fs,
        radius_sec=loss_cfg.peak_timing_radius_sec,
    )
    rr_valid = rr_mask > 0.5
    if torch.any(rr_valid):
        ibi_loss = F.huber_loss(
            pred["ibi_pred"][rr_valid],
            rr_track[rr_valid],
            reduction="mean",
            delta=float(loss_cfg.ibi_huber_delta),
        )
    else:
        ibi_loss = torch.zeros((), device=peak_target.device)

    gate_valid = gate_target >= 0.0
    if torch.any(gate_valid):
        gate_loss = F.binary_cross_entropy_with_logits(pred["gate_logit"][gate_valid], gate_target[gate_valid])
    else:
        gate_loss = torch.zeros((), device=peak_target.device)

    if domain_target is not None and loss_cfg.domain > 0.0 and "domain_logit" in pred:
        domain_loss = F.cross_entropy(pred["domain_logit"], domain_target)
    else:
        domain_loss = torch.zeros((), device=peak_target.device)

    total = (
        loss_cfg.peak * peak_loss
        + loss_cfg.beat * beat_loss
        + loss_cfg.ibi * ibi_loss
        + loss_cfg.gate * gate_loss
        + loss_cfg.domain * domain_loss
    )
    return {"total": total, "peak": peak_loss, "beat": beat_loss, "ibi": ibi_loss, "gate": gate_loss, "domain": domain_loss}


def compute_worst_domain_total(
    pred: Dict[str, torch.Tensor],
    peak_target: torch.Tensor,
    rr_track: torch.Tensor,
    rr_mask: torch.Tensor,
    gate_target: torch.Tensor,
    domain_target: torch.Tensor,
    loss_cfg: LossConfig,
) -> torch.Tensor:
    if domain_target is None or domain_target.numel() == 0:
        return compute_losses(pred, peak_target, rr_track, rr_mask, gate_target, domain_target, loss_cfg)["total"]
    domain_losses: List[torch.Tensor] = []
    for domain_id in torch.unique(domain_target.detach()):
        rows = domain_target == domain_id
        if not torch.any(rows):
            continue
        pred_rows = {key: value[rows] if torch.is_tensor(value) and value.shape[0] == rows.shape[0] else value for key, value in pred.items()}
        losses = compute_losses(
            pred_rows,
            peak_target[rows],
            rr_track[rows],
            rr_mask[rows],
            gate_target[rows],
            domain_target[rows],
            loss_cfg=loss_cfg,
        )
        domain_losses.append(losses["total"])
    if not domain_losses:
        return compute_losses(pred, peak_target, rr_track, rr_mask, gate_target, domain_target, loss_cfg)["total"]
    return torch.stack(domain_losses).max()


def _loss_dict_to_float(losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {k: float(v.detach().cpu().item()) for k, v in losses.items()}


def _gate_accuracy(pred_logit: torch.Tensor, target: torch.Tensor) -> Tuple[float, int]:
    valid = target >= 0.0
    if not torch.any(valid):
        return 0.0, 0
    pred = (torch.sigmoid(pred_logit[valid]) >= 0.5).float()
    target_bin = (target[valid] >= 0.5).float()
    acc = float((pred == target_bin).float().mean().detach().cpu().item())
    return acc, int(valid.sum().detach().cpu().item())


def _peak_f1_from_batch(pred_logit: torch.Tensor, target: torch.Tensor) -> float:
    pred_bin = (torch.sigmoid(pred_logit) >= 0.5).float()
    targ_bin = (target >= 0.5).float()
    tp = float((pred_bin * targ_bin).sum().detach().cpu().item())
    fp = float((pred_bin * (1.0 - targ_bin)).sum().detach().cpu().item())
    fn = float(((1.0 - pred_bin) * targ_bin).sum().detach().cpu().item())
    precision = tp / max(tp + fp, EPS)
    recall = tp / max(tp + fn, EPS)
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _window_gate_bucket(rec: Dict[str, Any], start: int, end: int) -> str:
    gate_window = rec["gate_track"][start:end]
    valid = gate_window >= 0.0
    if not np.any(valid):
        return "unknown"
    mean_gate = float(np.mean(gate_window[valid]))
    if mean_gate <= 0.05:
        return "rest"
    if mean_gate >= 0.95:
        return "motion"
    return "mixed"


def make_balanced_sampler(ds: MultiDatasetPeakWindowDataset) -> WeightedRandomSampler:
    dataset_counts: Dict[str, int] = {}
    cell_counts: Dict[Tuple[str, str], int] = {}
    subject_counts: Dict[str, int] = {}
    keys: List[Tuple[str, str, str]] = []
    for rid, start, end in ds.index:
        rec = ds.records[rid]
        dkey = str(rec["dataset"])
        akey = _window_gate_bucket(rec, start, end)
        skey = str(rec["subject"])
        keys.append((dkey, akey, skey))
        dataset_counts[dkey] = dataset_counts.get(dkey, 0) + 1
        cell_counts[(dkey, akey)] = cell_counts.get((dkey, akey), 0) + 1
        subject_counts[skey] = subject_counts.get(skey, 0) + 1
    weights = []
    for dkey, akey, skey in keys:
        # Equalize dataset/activity cells first, then gently reduce domination
        # by very long subjects without exploding rare-subject weights.
        cell_w = 1.0 / max(1, cell_counts[(dkey, akey)])
        subject_w = 1.0 / np.sqrt(max(1, subject_counts[skey]))
        dataset_w = 1.0 / np.sqrt(max(1, dataset_counts[dkey]))
        weights.append(cell_w * subject_w * dataset_w)
    weights_np = np.asarray(weights, dtype=np.float64)
    weights_np = weights_np / max(float(np.mean(weights_np)), EPS)
    return WeightedRandomSampler(weights=torch.as_tensor(weights_np, dtype=torch.double), num_samples=len(ds), replacement=True)


def build_loader(
    records: Sequence[Dict[str, Any]],
    window_cfg: WindowConfig,
    batch_size: int,
    shuffle: bool,
    augment_cfg: Optional[AugmentConfig] = None,
    balanced_sampling: bool = False,
) -> Tuple[MultiDatasetPeakWindowDataset, DataLoader]:
    ds = MultiDatasetPeakWindowDataset(records, window_cfg, augment_cfg=augment_cfg)
    sampler = make_balanced_sampler(ds) if balanced_sampling and len(ds) > 0 else None
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=False,
    )
    return ds, loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_cfg: LossConfig,
    device: torch.device,
    desc: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    domain_lambda: float = 0.0,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    if hasattr(model, "grl_lambda"):
        model.grl_lambda = float(domain_lambda if train_mode else 0.0)
    agg: Dict[str, float] = {"count": 0.0, "gate_acc_sum": 0.0, "gate_count": 0.0, "peak_f1_sum": 0.0}
    context = torch.enable_grad() if train_mode else torch.no_grad()
    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)

    with context:
        for batch_idx, (ppg, peak_target, rr_track, rr_mask, gate_target, domain_target, *_) in enumerate(progress, start=1):
            ppg = ppg.to(device)
            peak_target = peak_target.to(device)
            rr_track = rr_track.to(device)
            rr_mask = rr_mask.to(device)
            gate_target = gate_target.to(device)
            domain_target = domain_target.to(device)

            pred = model(ppg)
            losses = compute_losses(pred, peak_target, rr_track, rr_mask, gate_target, domain_target, loss_cfg=loss_cfg)
            if train_mode and float(loss_cfg.worst_domain) > 0.0:
                worst_total = compute_worst_domain_total(
                    pred,
                    peak_target,
                    rr_track,
                    rr_mask,
                    gate_target,
                    domain_target,
                    loss_cfg=loss_cfg,
                )
                wd = min(1.0, max(0.0, float(loss_cfg.worst_domain)))
                losses["total"] = (1.0 - wd) * losses["total"] + wd * worst_total
                losses["worst_domain"] = worst_total

            if train_mode:
                optimizer.zero_grad()
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            batch_size_f = float(ppg.size(0))
            agg["count"] += batch_size_f
            for key, value in _loss_dict_to_float(losses).items():
                agg[key] = agg.get(key, 0.0) + batch_size_f * value
            gate_acc, gate_count = _gate_accuracy(pred["gate_logit"], gate_target)
            agg["gate_acc_sum"] += gate_acc * float(gate_count)
            agg["gate_count"] += float(gate_count)
            agg["peak_f1_sum"] += batch_size_f * _peak_f1_from_batch(pred["peak_logit"], peak_target)
            avg_total = agg["total"] / max(1.0, agg["count"])
            progress.set_postfix(batch=batch_idx, avg_total=f"{avg_total:.4f}")

    denom = max(1.0, agg.pop("count"))
    out = {k: v / denom for k, v in agg.items() if k not in ("gate_acc_sum", "gate_count", "peak_f1_sum")}
    out["peak_f1"] = agg["peak_f1_sum"] / denom
    out["gate_acc"] = agg["gate_acc_sum"] / agg["gate_count"] if agg["gate_count"] > 0 else float("nan")
    return out


def fit_model_with_validation(
    train_records: Sequence[Dict[str, Any]],
    val_records: Sequence[Dict[str, Any]],
    window_cfg: WindowConfig,
    model_cfg: ModelConfig,
    loss_cfg: LossConfig,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    device: str,
    title: str,
    augment_cfg: Optional[AugmentConfig],
    balanced_sampling: bool,
    domain_lambda: float,
) -> Tuple[nn.Module, Dict[str, Any]]:
    train_ds, train_loader = build_loader(
        train_records,
        window_cfg,
        batch_size=batch_size,
        shuffle=True,
        augment_cfg=augment_cfg,
        balanced_sampling=balanced_sampling,
    )
    val_ds, val_loader = build_loader(val_records, window_cfg, batch_size=batch_size, shuffle=False)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Train/validation dataset is empty")

    dev = torch.device(device)
    model = PeakIntervalGateNet(**asdict(model_cfg)).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best = {"val_total": float("inf"), "epoch": 0, "state_dict": None}
    history: List[Dict[str, float]] = []
    bad_epochs = 0
    t0 = time.time()
    pbar = tqdm(range(1, epochs + 1), desc=title, leave=True)

    for epoch in pbar:
        epoch_t0 = time.time()
        tr = run_epoch(
            model,
            train_loader,
            loss_cfg,
            dev,
            desc=f"{title} train {epoch}/{epochs}",
            optimizer=optimizer,
            domain_lambda=domain_lambda,
        )
        va = run_epoch(model, val_loader, loss_cfg, dev, desc=f"{title} valid {epoch}/{epochs}", optimizer=None)
        row = {
            "epoch": float(epoch),
            **{f"train_{k}": float(v) for k, v in tr.items()},
            **{f"val_{k}": float(v) for k, v in va.items()},
        }
        history.append(row)
        elapsed = time.time() - t0
        eta = (elapsed / max(epoch, 1)) * max(0, epochs - epoch)
        pbar.set_postfix(
            train_total=f"{tr['total']:.4f}",
            val_total=f"{va['total']:.4f}",
            val_peak_f1=f"{va['peak_f1']:.3f}",
            elapsed=_fmt_seconds(elapsed),
            eta=_fmt_seconds(eta),
            epoch_time=_fmt_seconds(time.time() - epoch_t0),
        )

        if va["total"] < best["val_total"]:
            best = {
                "val_total": float(va["total"]),
                "epoch": int(epoch),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])
    return model, {
        "history": history,
        "best_epoch": int(best["epoch"]),
        "best_val_total": float(best["val_total"]),
        "num_train_windows": int(len(train_ds)),
        "num_val_windows": int(len(val_ds)),
    }


def fit_model_fixed_epochs(
    train_records: Sequence[Dict[str, Any]],
    window_cfg: WindowConfig,
    model_cfg: ModelConfig,
    loss_cfg: LossConfig,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    title: str,
    augment_cfg: Optional[AugmentConfig],
    balanced_sampling: bool,
    domain_lambda: float,
) -> Tuple[nn.Module, Dict[str, Any]]:
    train_ds, train_loader = build_loader(
        train_records,
        window_cfg,
        batch_size=batch_size,
        shuffle=True,
        augment_cfg=augment_cfg,
        balanced_sampling=balanced_sampling,
    )
    if len(train_ds) == 0:
        raise RuntimeError("Final training dataset is empty")
    dev = torch.device(device)
    model = PeakIntervalGateNet(**asdict(model_cfg)).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: List[Dict[str, float]] = []
    t0 = time.time()
    pbar = tqdm(range(1, epochs + 1), desc=title, leave=True)
    for epoch in pbar:
        epoch_t0 = time.time()
        tr = run_epoch(
            model,
            train_loader,
            loss_cfg,
            dev,
            desc=f"{title} train {epoch}/{epochs}",
            optimizer=optimizer,
            domain_lambda=domain_lambda,
        )
        history.append({"epoch": float(epoch), **{f"train_{k}": float(v) for k, v in tr.items()}})
        elapsed = time.time() - t0
        eta = (elapsed / max(epoch, 1)) * max(0, epochs - epoch)
        pbar.set_postfix(
            train_total=f"{tr['total']:.4f}",
            train_peak_f1=f"{tr['peak_f1']:.3f}",
            elapsed=_fmt_seconds(elapsed),
            eta=_fmt_seconds(eta),
            epoch_time=_fmt_seconds(time.time() - epoch_t0),
        )
    return model, {"history": history, "trained_epochs": int(epochs), "num_train_windows": int(len(train_ds))}


def collect_outputs(
    model: nn.Module,
    records: Sequence[Dict[str, Any]],
    window_cfg: WindowConfig,
    loss_cfg: LossConfig,
    batch_size: int,
    device: str,
    desc: str,
) -> Dict[str, Any]:
    ds, loader = build_loader(records, window_cfg, batch_size=batch_size, shuffle=False)
    dev = torch.device(device)
    model.eval()
    agg: Dict[str, float] = {"count": 0.0}
    peak_prob_parts: List[np.ndarray] = []
    peak_true_parts: List[np.ndarray] = []
    rr_pred_parts: List[np.ndarray] = []
    rr_true_parts: List[np.ndarray] = []
    gate_prob_parts: List[np.ndarray] = []
    gate_true_parts: List[np.ndarray] = []
    window_meta: List[Dict[str, str]] = []
    rr_offsets: List[Tuple[int, int]] = []
    gate_window_indices: List[int] = []
    rr_offset = 0
    window_idx = 0

    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress, start=1):
            (
                ppg,
                peak_target,
                rr_track,
                rr_mask,
                gate_target,
                domain_target,
                subjects,
                datasets,
                activities,
                records,
                subsets,
            ) = batch
            ppg = ppg.to(dev)
            peak_target = peak_target.to(dev)
            rr_track = rr_track.to(dev)
            rr_mask = rr_mask.to(dev)
            gate_target = gate_target.to(dev)
            domain_target = domain_target.to(dev)

            pred = model(ppg)
            losses = compute_losses(pred, peak_target, rr_track, rr_mask, gate_target, domain_target, loss_cfg=loss_cfg)
            batch_size_f = float(ppg.size(0))
            agg["count"] += batch_size_f
            for key, value in _loss_dict_to_float(losses).items():
                agg[key] = agg.get(key, 0.0) + batch_size_f * value
            progress.set_postfix(batch=batch_idx, avg_total=f"{agg['total'] / max(1.0, agg['count']):.4f}")

            peak_prob = torch.sigmoid(pred["peak_logit"]).detach().cpu().numpy().reshape(-1).astype(np.float16)
            peak_true = (peak_target.detach().cpu().numpy().reshape(-1) >= 0.5).astype(np.uint8)
            peak_prob_parts.append(peak_prob)
            peak_true_parts.append(peak_true)

            rr_mask_np = rr_mask.detach().cpu().numpy().reshape(-1) > 0.5
            if np.any(rr_mask_np):
                rr_pred = pred["ibi_pred"].detach().cpu().numpy().reshape(-1)[rr_mask_np].astype(np.float32)
                rr_true = rr_track.detach().cpu().numpy().reshape(-1)[rr_mask_np].astype(np.float32)
                rr_pred_parts.append(rr_pred)
                rr_true_parts.append(rr_true)
            rr_mask_2d = rr_mask.detach().cpu().numpy() > 0.5

            gate_target_np = gate_target.detach().cpu().numpy().astype(np.float32)
            gate_valid = gate_target_np >= 0.0
            if np.any(gate_valid):
                gate_prob = torch.sigmoid(pred["gate_logit"]).detach().cpu().numpy().astype(np.float32)[gate_valid]
                gate_true = (gate_target_np[gate_valid] >= 0.5).astype(np.uint8)
                gate_prob_parts.append(gate_prob)
                gate_true_parts.append(gate_true)
            for i in range(int(ppg.shape[0])):
                meta = {
                    "subject": str(subjects[i]),
                    "dataset": str(datasets[i]),
                    "activity": str(activities[i]),
                    "record": str(records[i]),
                    "subset": str(subsets[i]),
                    "gate_bucket": "unknown",
                }
                gval = float(gate_target_np[i])
                if gval >= 0.0:
                    meta["gate_bucket"] = "motion" if gval >= 0.5 else "rest"
                    gate_window_indices.append(window_idx)
                cnt = int(np.sum(rr_mask_2d[i]))
                rr_offsets.append((rr_offset, rr_offset + cnt))
                rr_offset += cnt
                window_meta.append(meta)
                window_idx += 1

    denom = max(1.0, agg.pop("count", 0.0))
    losses_out = {f"loss_{k}": float(v / denom) for k, v in agg.items()}
    return {
        "n_windows": int(len(ds)),
        "losses": losses_out,
        "peak_prob": np.concatenate(peak_prob_parts).astype(np.float32) if peak_prob_parts else np.zeros(0, dtype=np.float32),
        "peak_true": np.concatenate(peak_true_parts).astype(np.uint8) if peak_true_parts else np.zeros(0, dtype=np.uint8),
        "rr_pred": np.concatenate(rr_pred_parts).astype(np.float32) if rr_pred_parts else np.zeros(0, dtype=np.float32),
        "rr_true": np.concatenate(rr_true_parts).astype(np.float32) if rr_true_parts else np.zeros(0, dtype=np.float32),
        "gate_prob": np.concatenate(gate_prob_parts).astype(np.float32) if gate_prob_parts else np.zeros(0, dtype=np.float32),
        "gate_true": np.concatenate(gate_true_parts).astype(np.uint8) if gate_true_parts else np.zeros(0, dtype=np.uint8),
        "window_samples": int(round(window_cfg.win_sec * window_cfg.fs)),
        "fs": float(window_cfg.fs),
        "window_meta": window_meta,
        "rr_offsets": rr_offsets,
        "gate_window_indices": gate_window_indices,
    }


def merge_output_dicts(outputs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "n_windows": int(sum(out.get("n_windows", 0) for out in outputs)),
        "window_samples": int(next((out.get("window_samples") for out in outputs if out.get("window_samples")), 0)),
        "fs": float(next((out.get("fs") for out in outputs if out.get("fs")), 0.0)),
        "losses": {},
        "peak_prob": np.concatenate([out["peak_prob"] for out in outputs if len(out["peak_prob"])], axis=0) if any(len(out["peak_prob"]) for out in outputs) else np.zeros(0, dtype=np.float32),
        "peak_true": np.concatenate([out["peak_true"] for out in outputs if len(out["peak_true"])], axis=0) if any(len(out["peak_true"]) for out in outputs) else np.zeros(0, dtype=np.uint8),
        "rr_pred": np.concatenate([out["rr_pred"] for out in outputs if len(out["rr_pred"])], axis=0) if any(len(out["rr_pred"]) for out in outputs) else np.zeros(0, dtype=np.float32),
        "rr_true": np.concatenate([out["rr_true"] for out in outputs if len(out["rr_true"])], axis=0) if any(len(out["rr_true"]) for out in outputs) else np.zeros(0, dtype=np.float32),
        "gate_prob": np.concatenate([out["gate_prob"] for out in outputs if len(out["gate_prob"])], axis=0) if any(len(out["gate_prob"]) for out in outputs) else np.zeros(0, dtype=np.float32),
        "gate_true": np.concatenate([out["gate_true"] for out in outputs if len(out["gate_true"])], axis=0) if any(len(out["gate_true"]) for out in outputs) else np.zeros(0, dtype=np.uint8),
        "window_meta": [],
        "rr_offsets": [],
        "gate_window_indices": [],
    }
    window_base = 0
    rr_base = 0
    for out in outputs:
        merged["window_meta"].extend(out.get("window_meta", []))
        for lo, hi in out.get("rr_offsets", []):
            merged["rr_offsets"].append((int(lo) + rr_base, int(hi) + rr_base))
        merged["gate_window_indices"].extend([int(i) + window_base for i in out.get("gate_window_indices", [])])
        window_base += int(out.get("n_windows", 0))
        rr_base += int(len(out.get("rr_true", [])))
    loss_keys = sorted({key for out in outputs for key in out.get("losses", {}).keys()})
    total_windows = max(1, merged["n_windows"])
    for key in loss_keys:
        weighted = 0.0
        for out in outputs:
            weighted += float(out.get("losses", {}).get(key, 0.0)) * float(out.get("n_windows", 0))
        merged["losses"][key] = weighted / float(total_windows)
    return merged


def select_binary_threshold(y_true: np.ndarray, y_prob: np.ndarray, objective: str) -> float:
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    best_thr = 0.5
    best_score = -np.inf
    for thr in np.linspace(0.05, 0.95, 37):
        y_pred = (y_prob >= thr).astype(np.uint8)
        if objective == "bal_acc":
            score = float(metrics.balanced_accuracy_score(y_true, y_pred))
        else:
            score = float(metrics.f1_score(y_true, y_pred, zero_division=0))
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    out: Dict[str, Any] = {
        "n": int(y_true.size),
        "threshold": float(threshold),
        "positive_rate": float(np.mean(y_true)) if y_true.size else float("nan"),
    }
    if y_true.size == 0:
        out.update(
            {
                "accuracy": float("nan"),
                "balanced_accuracy": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "roc_auc": float("nan"),
                "pr_auc": float("nan"),
                "confusion_matrix": [[0, 0], [0, 0]],
            }
        )
        return out
    y_pred = (y_prob >= threshold).astype(np.uint8)
    cm = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int)
    out.update(
        {
            "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(metrics.balanced_accuracy_score(y_true, y_pred)),
            "precision": float(metrics.precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(metrics.recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(metrics.f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": cm.tolist(),
        }
    )
    if len(np.unique(y_true)) >= 2:
        out["roc_auc"] = float(metrics.roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(metrics.average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def compute_gate_diagnostics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    out: Dict[str, Any] = {
        "n": int(y_true.size),
        "threshold": float(threshold),
        "true_motion_rate": float(np.mean(y_true)) if y_true.size else float("nan"),
        "pred_motion_rate": float(np.mean(y_prob >= threshold)) if y_prob.size else float("nan"),
        "near_threshold_frac_0p05": float(np.mean(np.abs(y_prob - float(threshold)) <= 0.05)) if y_prob.size else float("nan"),
    }
    if y_prob.size == 0:
        return out
    quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    out["score_quantiles"] = {f"q{int(q * 100):02d}": float(np.quantile(y_prob, q)) for q in quantiles}
    out["threshold_percentile"] = float(np.mean(y_prob <= float(threshold)))
    for cls, name in ((0, "rest"), (1, "motion")):
        vals = y_prob[y_true == cls]
        if vals.size:
            out[f"{name}_score_mean"] = float(np.mean(vals))
            out[f"{name}_score_std"] = float(np.std(vals))
            out[f"{name}_score_q25"] = float(np.quantile(vals, 0.25))
            out[f"{name}_score_median"] = float(np.quantile(vals, 0.50))
            out[f"{name}_score_q75"] = float(np.quantile(vals, 0.75))
        else:
            out[f"{name}_score_mean"] = float("nan")
            out[f"{name}_score_std"] = float("nan")
            out[f"{name}_score_q25"] = float("nan")
            out[f"{name}_score_median"] = float("nan")
            out[f"{name}_score_q75"] = float("nan")
    rest_vals = y_prob[y_true == 0]
    motion_vals = y_prob[y_true == 1]
    if rest_vals.size and motion_vals.size:
        pooled = np.sqrt(0.5 * (float(np.var(rest_vals)) + float(np.var(motion_vals))) + EPS)
        out["class_score_gap"] = float(np.mean(motion_vals) - np.mean(rest_vals))
        out["class_separation_cohen_d"] = float((np.mean(motion_vals) - np.mean(rest_vals)) / pooled)
    else:
        out["class_score_gap"] = float("nan")
        out["class_separation_cohen_d"] = float("nan")
    cm = metrics.confusion_matrix(y_true, (y_prob >= threshold).astype(np.uint8), labels=[0, 1]).astype(int)
    tn, fp, fn, tp = cm.ravel()
    out["rest_false_motion_rate"] = float(fp / max(fp + tn, 1))
    out["motion_false_rest_rate"] = float(fn / max(fn + tp, 1))
    return out


def _output_subset_by_windows(outputs: Dict[str, Any], window_indices: Sequence[int]) -> Dict[str, Any]:
    win = int(outputs.get("window_samples", 0))
    fs = float(outputs.get("fs", 0.0))
    idx = [int(i) for i in window_indices if 0 <= int(i) < int(outputs.get("n_windows", 0))]
    if not idx or win <= 0:
        return {
            "n_windows": 0,
            "losses": {},
            "peak_prob": np.zeros(0, dtype=np.float32),
            "peak_true": np.zeros(0, dtype=np.uint8),
            "rr_pred": np.zeros(0, dtype=np.float32),
            "rr_true": np.zeros(0, dtype=np.float32),
            "gate_prob": np.zeros(0, dtype=np.float32),
            "gate_true": np.zeros(0, dtype=np.uint8),
            "window_samples": win,
            "fs": fs,
        }
    peak_prob = np.concatenate([outputs["peak_prob"][i * win : (i + 1) * win] for i in idx]).astype(np.float32)
    peak_true = np.concatenate([outputs["peak_true"][i * win : (i + 1) * win] for i in idx]).astype(np.uint8)
    rr_chunks_pred: List[np.ndarray] = []
    rr_chunks_true: List[np.ndarray] = []
    for i in idx:
        if i < len(outputs.get("rr_offsets", [])):
            lo, hi = outputs["rr_offsets"][i]
            rr_chunks_pred.append(outputs["rr_pred"][lo:hi])
            rr_chunks_true.append(outputs["rr_true"][lo:hi])
    idx_set = set(idx)
    gate_positions = [j for j, widx in enumerate(outputs.get("gate_window_indices", [])) if int(widx) in idx_set]
    return {
        "n_windows": int(len(idx)),
        "losses": {},
        "peak_prob": peak_prob,
        "peak_true": peak_true,
        "rr_pred": np.concatenate(rr_chunks_pred).astype(np.float32) if rr_chunks_pred else np.zeros(0, dtype=np.float32),
        "rr_true": np.concatenate(rr_chunks_true).astype(np.float32) if rr_chunks_true else np.zeros(0, dtype=np.float32),
        "gate_prob": outputs["gate_prob"][gate_positions].astype(np.float32) if gate_positions else np.zeros(0, dtype=np.float32),
        "gate_true": outputs["gate_true"][gate_positions].astype(np.uint8) if gate_positions else np.zeros(0, dtype=np.uint8),
        "window_samples": win,
        "fs": fs,
    }


def summarize_output_groups(
    outputs: Dict[str, Any],
    peak_threshold: float,
    gate_threshold: float,
    group_key: str,
    min_windows: int = 1,
) -> Dict[str, Any]:
    groups: Dict[str, List[int]] = {}
    for idx, meta in enumerate(outputs.get("window_meta", [])):
        if group_key == "dataset_activity":
            key = f"{meta.get('dataset', 'unknown')}::{meta.get('activity', 'unknown')}"
        elif group_key == "dataset_subset":
            key = f"{meta.get('dataset', 'unknown')}::{meta.get('subset', 'unknown')}"
        else:
            key = str(meta.get(group_key, "unknown"))
        groups.setdefault(key, []).append(idx)
    out: Dict[str, Any] = {}
    for key, indices in sorted(groups.items()):
        if len(indices) < int(min_windows):
            continue
        out[key] = summarize_outputs(
            _output_subset_by_windows(outputs, indices),
            peak_threshold=peak_threshold,
            gate_threshold=gate_threshold,
        )
    return out


def build_group_scorecard_bundle(outputs: Dict[str, Any], peak_threshold: float, gate_threshold: float) -> Dict[str, Any]:
    return {
        "by_dataset": summarize_output_groups(outputs, peak_threshold, gate_threshold, group_key="dataset"),
        "by_subject": summarize_output_groups(outputs, peak_threshold, gate_threshold, group_key="subject"),
        "by_activity": summarize_output_groups(outputs, peak_threshold, gate_threshold, group_key="activity"),
        "by_dataset_activity": summarize_output_groups(outputs, peak_threshold, gate_threshold, group_key="dataset_activity"),
        "by_dataset_subset": summarize_output_groups(outputs, peak_threshold, gate_threshold, group_key="dataset_subset"),
    }


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    out: Dict[str, Any] = {"n": int(y_true.size)}
    if y_true.size == 0:
        out.update(
            {
                "mae": float("nan"),
                "rmse": float("nan"),
                "median_ae": float("nan"),
                "bias": float("nan"),
                "pearson_r": float("nan"),
                "r2": float("nan"),
            }
        )
        return out
    diff = y_pred - y_true
    out.update(
        {
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff * diff))),
            "median_ae": float(np.median(np.abs(diff))),
            "bias": float(np.mean(diff)),
        }
    )
    if y_true.size > 1 and float(np.std(y_true)) > EPS and float(np.std(y_pred)) > EPS:
        out["pearson_r"] = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        out["pearson_r"] = float("nan")
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum(diff ** 2))
    out["r2"] = float(1.0 - ss_res / (ss_tot + EPS))
    return out


def _target_peak_centers(target_binary: np.ndarray) -> np.ndarray:
    y = np.asarray(target_binary, dtype=np.uint8)
    if y.size == 0 or not np.any(y):
        return np.zeros(0, dtype=np.int32)
    padded = np.pad(y, (1, 1), constant_values=0)
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    centers = np.round((starts + ends - 1) * 0.5).astype(np.int32)
    return centers


def _predict_peak_centers(prob: np.ndarray, threshold: float, fs: float) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float32)
    if prob.size == 0:
        return np.zeros(0, dtype=np.int32)
    distance = max(1, int(round(0.30 * float(fs))))
    peaks, _ = signal.find_peaks(prob, height=float(threshold), distance=distance)
    return peaks.astype(np.int32)


def _match_peak_events(pred_idx: np.ndarray, true_idx: np.ndarray, tolerance_samples: int) -> Tuple[List[Tuple[int, int]], int, int, int]:
    pred = list(map(int, pred_idx))
    true = list(map(int, true_idx))
    used_pred: set[int] = set()
    matches: List[Tuple[int, int]] = []
    for t in true:
        best_j = None
        best_abs = tolerance_samples + 1
        for j, p in enumerate(pred):
            if j in used_pred:
                continue
            err = abs(p - t)
            if err <= tolerance_samples and err < best_abs:
                best_j = j
                best_abs = err
        if best_j is not None:
            used_pred.add(best_j)
            matches.append((pred[best_j], t))
    tp = len(matches)
    fp = len(pred) - tp
    fn = len(true) - tp
    return matches, tp, fp, fn


def compute_event_error_arrays(
    outputs: Dict[str, Any],
    peak_threshold: float,
    tolerance_sec: float = PPG_MAIN_EVENT_TOLERANCE_SEC,
) -> Dict[str, Any]:
    n_windows = int(outputs.get("n_windows", 0))
    win = int(outputs.get("window_samples", 0))
    fs = float(outputs.get("fs", 0.0))
    if n_windows <= 0 or win <= 0 or fs <= 0 or len(outputs["peak_prob"]) < n_windows * win:
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tolerance_sec": float(tolerance_sec),
            "tolerance_ms": float(1000.0 * tolerance_sec),
            "peak_timing_errors": np.zeros(0, dtype=np.float32),
            "rri_errors": np.zeros(0, dtype=np.float32),
        }
    prob_windows = outputs["peak_prob"][: n_windows * win].reshape(n_windows, win)
    true_windows = outputs["peak_true"][: n_windows * win].reshape(n_windows, win)
    tolerance_samples = max(1, int(round(float(tolerance_sec) * fs)))
    tp = fp = fn = 0
    timing_errors: List[float] = []
    rri_errors: List[float] = []
    for prob, true_bin in zip(prob_windows, true_windows):
        pred_idx = _predict_peak_centers(prob, threshold=peak_threshold, fs=fs)
        true_idx = _target_peak_centers(true_bin)
        matches, w_tp, w_fp, w_fn = _match_peak_events(pred_idx, true_idx, tolerance_samples=tolerance_samples)
        tp += w_tp
        fp += w_fp
        fn += w_fn
        if matches:
            matches_sorted = sorted(matches, key=lambda x: x[1])
            timing_errors.extend([(p - t) / fs for p, t in matches_sorted])
            if len(matches_sorted) >= 2:
                pred_seq = np.asarray([p for p, _ in matches_sorted], dtype=np.float32)
                true_seq = np.asarray([t for _, t in matches_sorted], dtype=np.float32)
                rri_errors.extend(((np.diff(pred_seq) - np.diff(true_seq)) / fs).astype(np.float32).tolist())
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tolerance_sec": float(tolerance_sec),
        "tolerance_ms": float(1000.0 * tolerance_sec),
        "peak_timing_errors": np.asarray(timing_errors, dtype=np.float32),
        "rri_errors": np.asarray(rri_errors, dtype=np.float32),
    }


def compute_error_distribution_metrics(errors: np.ndarray) -> Dict[str, Any]:
    errors = np.asarray(errors, dtype=np.float32)
    out: Dict[str, Any] = {"n": int(errors.size)}
    if errors.size == 0:
        out.update({"bias": float("nan"), "mae": float("nan"), "median_ae": float("nan"), "std": float("nan"), "variance": float("nan")})
        return out
    out.update(
        {
            "bias": float(np.mean(errors)),
            "mae": float(np.mean(np.abs(errors))),
            "median_ae": float(np.median(np.abs(errors))),
            "std": float(np.std(errors)),
            "variance": float(np.var(errors)),
        }
    )
    return out


def _summarize_event_arrays(arr: Dict[str, Any]) -> Dict[str, Any]:
    tp, fp, fn = int(arr["tp"]), int(arr["fp"]), int(arr["fn"])
    precision = tp / max(tp + fp, EPS)
    recall = tp / max(tp + fn, EPS)
    f1 = 2.0 * precision * recall / max(precision + recall, EPS)
    return {
        "tolerance_sec": float(arr.get("tolerance_sec", PPG_MAIN_EVENT_TOLERANCE_SEC)),
        "tolerance_ms": float(arr.get("tolerance_ms", 1000.0 * PPG_MAIN_EVENT_TOLERANCE_SEC)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        **{f"timing_{k}": v for k, v in compute_error_distribution_metrics(arr["peak_timing_errors"]).items()},
    }


def compute_event_metrics(outputs: Dict[str, Any], peak_threshold: float) -> Dict[str, Any]:
    layered_peak_events: Dict[str, Any] = {}
    layered_rri_events: Dict[str, Any] = {}
    main_key = f"{int(round(PPG_MAIN_EVENT_TOLERANCE_SEC * 1000.0))}ms"
    for tolerance_sec in PPG_EVENT_TOLERANCE_SECS:
        key = f"{int(round(float(tolerance_sec) * 1000.0))}ms"
        arr = compute_event_error_arrays(outputs, peak_threshold=peak_threshold, tolerance_sec=float(tolerance_sec))
        layered_peak_events[key] = _summarize_event_arrays(arr)
        rri_metrics = compute_error_distribution_metrics(arr["rri_errors"])
        rri_metrics["tolerance_sec"] = float(tolerance_sec)
        rri_metrics["tolerance_ms"] = float(1000.0 * tolerance_sec)
        layered_rri_events[key] = rri_metrics
    return {
        "peak_events": layered_peak_events[main_key],
        "peak_events_by_tolerance": layered_peak_events,
        "rri_event_sequence": layered_rri_events[main_key],
        "rri_event_sequence_by_tolerance": layered_rri_events,
    }


def evaluate_ecg_detector_on_ptt(
    data_root: Path,
    tolerance_sec: float = 0.004,
    ibi_abs_tolerance_sec: float = 0.050,
    ibi_rel_tolerance: float = 0.10,
) -> Dict[str, Any]:
    csv_dir = data_root / "pulse-transit-time-ppg" / "1.1.0" / "csv"
    tp = fp = fn = 0
    timing_errors: List[float] = []
    ibi_ok: List[bool] = []
    ibi_abs_errors: List[float] = []
    per_record: List[Dict[str, Any]] = []
    fs = 500.0
    tolerance_samples = max(1, int(round(float(tolerance_sec) * fs)))

    for path in sorted(csv_dir.glob("s*_*.csv")):
        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            continue
        if not {"ecg", "peaks"}.issubset(header.columns):
            continue
        df = pd.read_csv(path, usecols=["ecg", "peaks"])
        ecg = pd.to_numeric(df["ecg"], errors="coerce").to_numpy(dtype=np.float32)
        true_idx = np.where(pd.to_numeric(df["peaks"], errors="coerce").fillna(0).to_numpy(dtype=np.float32) > 0.5)[0].astype(np.int32)
        pred_idx = detect_ecg_rpeaks(ecg, fs=fs)
        matches, rec_tp, rec_fp, rec_fn = _match_peak_events(pred_idx, true_idx, tolerance_samples=tolerance_samples)
        tp += rec_tp
        fp += rec_fp
        fn += rec_fn
        rec_precision = rec_tp / max(rec_tp + rec_fp, EPS)
        rec_recall = rec_tp / max(rec_tp + rec_fn, EPS)
        rec_f1 = 2.0 * rec_precision * rec_recall / max(rec_precision + rec_recall, EPS)
        rec_timing = [(p - t) / fs for p, t in matches]
        timing_errors.extend(rec_timing)
        rec_ibi_ok: List[bool] = []
        rec_ibi_abs_errors: List[float] = []
        matches_sorted = sorted(matches, key=lambda x: x[1])
        if len(matches_sorted) >= 2:
            pred_ibi = np.diff([p for p, _ in matches_sorted]).astype(np.float32) / fs
            true_ibi = np.diff([t for _, t in matches_sorted]).astype(np.float32) / fs
            ibi_err = np.abs(pred_ibi - true_ibi)
            rec_ibi_abs_errors = ibi_err.astype(np.float32).tolist()
            ibi_abs_errors.extend(rec_ibi_abs_errors)
            rec_ibi_ok = (ibi_err <= np.maximum(float(ibi_abs_tolerance_sec), float(ibi_rel_tolerance) * true_ibi)).tolist()
            ibi_ok.extend(rec_ibi_ok)
        per_record.append(
            {
                "record": path.name,
                "true_peaks": int(true_idx.size),
                "pred_peaks": int(pred_idx.size),
                "tp": int(rec_tp),
                "fp": int(rec_fp),
                "fn": int(rec_fn),
                "peak_precision": float(rec_precision),
                "peak_recall": float(rec_recall),
                "peak_f1": float(rec_f1),
                "peak_timing_bias_sec": float(np.mean(rec_timing)) if rec_timing else float("nan"),
                "peak_timing_mae_sec": float(np.mean(np.abs(rec_timing))) if rec_timing else float("nan"),
                "peak_timing_std_sec": float(np.std(rec_timing)) if rec_timing else float("nan"),
                "peak_timing_median_ae_sec": float(np.median(np.abs(rec_timing))) if rec_timing else float("nan"),
                "ibi_accuracy": float(np.mean(rec_ibi_ok)) if rec_ibi_ok else float("nan"),
                "ibi_pairs": int(len(rec_ibi_ok)),
                "ibi_mae_sec": float(np.mean(rec_ibi_abs_errors)) if rec_ibi_abs_errors else float("nan"),
                "ibi_median_ae_sec": float(np.median(rec_ibi_abs_errors)) if rec_ibi_abs_errors else float("nan"),
            }
        )

    precision = tp / max(tp + fp, EPS)
    recall = tp / max(tp + fn, EPS)
    f1 = 2.0 * precision * recall / max(precision + recall, EPS)
    timing_arr = np.asarray(timing_errors, dtype=np.float32)
    ibi_arr = np.asarray(ibi_ok, dtype=np.float32)
    ibi_err_arr = np.asarray(ibi_abs_errors, dtype=np.float32)
    return {
        "dataset": PTT_NAME,
        "records": int(len(per_record)),
        "tolerance_sec": float(tolerance_sec),
        "ibi_abs_tolerance_sec": float(ibi_abs_tolerance_sec),
        "ibi_rel_tolerance": float(ibi_rel_tolerance),
        "peak_precision": float(precision),
        "peak_recall": float(recall),
        "peak_f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "peak_timing_bias_sec": float(np.mean(timing_arr)) if timing_arr.size else float("nan"),
        "peak_timing_mae_sec": float(np.mean(np.abs(timing_arr))) if timing_arr.size else float("nan"),
        "peak_timing_median_ae_sec": float(np.median(np.abs(timing_arr))) if timing_arr.size else float("nan"),
        "peak_timing_std_sec": float(np.std(timing_arr)) if timing_arr.size else float("nan"),
        "peak_timing_variance_sec2": float(np.var(timing_arr)) if timing_arr.size else float("nan"),
        "ibi_accuracy": float(np.mean(ibi_arr)) if ibi_arr.size else float("nan"),
        "ibi_pairs": int(ibi_arr.size),
        "ibi_mae_sec": float(np.mean(ibi_err_arr)) if ibi_err_arr.size else float("nan"),
        "ibi_median_ae_sec": float(np.median(ibi_err_arr)) if ibi_err_arr.size else float("nan"),
        "ibi_std_sec": float(np.std(ibi_err_arr)) if ibi_err_arr.size else float("nan"),
        "ibi_variance_sec2": float(np.var(ibi_err_arr)) if ibi_err_arr.size else float("nan"),
        "low_peak_f1_records": [row for row in per_record if row["peak_f1"] < 0.95],
        "low_ibi_accuracy_records": [row for row in per_record if not np.isnan(row["ibi_accuracy"]) and row["ibi_accuracy"] < 0.95],
        "per_record": per_record,
    }


def run_ptt_ecg_detector_preflight(
    data_root: Path,
    outdir: Path,
    min_peak_f1: float,
    min_ibi_accuracy: float,
    tolerance_sec: float,
) -> Dict[str, Any]:
    result = evaluate_ecg_detector_on_ptt(data_root=data_root, tolerance_sec=tolerance_sec)
    result["min_peak_f1"] = float(min_peak_f1)
    result["min_ibi_accuracy"] = float(min_ibi_accuracy)
    result["low_peak_f1_records"] = [
        row for row in result.get("per_record", []) if row.get("peak_f1", float("nan")) < float(min_peak_f1)
    ]
    result["low_ibi_accuracy_records"] = [
        row
        for row in result.get("per_record", [])
        if not np.isnan(float(row.get("ibi_accuracy", float("nan")))) and float(row.get("ibi_accuracy", float("nan"))) < float(min_ibi_accuracy)
    ]
    result["worst_peak_f1_records"] = sorted(
        result.get("per_record", []),
        key=lambda row: float(row.get("peak_f1", float("nan"))),
    )[:10]
    result["worst_ibi_accuracy_records"] = sorted(
        [row for row in result.get("per_record", []) if not np.isnan(float(row.get("ibi_accuracy", float("nan"))))],
        key=lambda row: float(row.get("ibi_accuracy", float("nan"))),
    )[:10]
    result["passed"] = bool(result["peak_f1"] >= float(min_peak_f1) and result["ibi_accuracy"] >= float(min_ibi_accuracy))
    save_json(outdir / "ecg_detector_preflight.json", result)
    print(
        "[ecg-preflight] "
        f"PTT peak_f1={result['peak_f1']:.4f} "
        f"ibi_accuracy={result['ibi_accuracy']:.4f} "
        f"thresholds=({min_peak_f1:.4f}, {min_ibi_accuracy:.4f})"
    )
    if not result["passed"]:
        raise RuntimeError(
            "ECG detector preflight failed: "
            f"peak_f1={result['peak_f1']:.4f}, ibi_accuracy={result['ibi_accuracy']:.4f}. "
            f"Required >= {min_peak_f1:.4f} and >= {min_ibi_accuracy:.4f}."
        )
    return result


def _contiguous_true_segments(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.pad(mask.astype(np.int8), (1, 1), constant_values=0)
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if int(e - s) >= int(min_len)]


def detect_ppg_pulse_peaks(ppg: np.ndarray, fs: float, polarity: int = 1) -> np.ndarray:
    x = np.asarray(ppg, dtype=np.float32)
    if x.size < int(max(128, fs)):
        return np.zeros(0, dtype=np.int32)
    y = bandpass_filter(x, fs=fs, low=0.4, high=min(8.0, 0.45 * fs), order=2)
    y = y if int(polarity) >= 0 else -y
    distance = max(1, int(round(0.30 * float(fs))))
    prominence = 0.25 * float(np.std(y) + EPS)
    peaks, _ = signal.find_peaks(y, distance=distance, prominence=prominence)
    return peaks.astype(np.int32)


def _match_forward_delays(ecg_idx: np.ndarray, ppg_idx: np.ndarray, fs: float, min_delay_sec: float, max_delay_sec: float) -> np.ndarray:
    ecg_idx = np.asarray(ecg_idx, dtype=np.int32)
    ppg_idx = np.asarray(ppg_idx, dtype=np.int32)
    if ecg_idx.size == 0 or ppg_idx.size == 0:
        return np.zeros(0, dtype=np.float32)
    delays: List[float] = []
    lo_s = int(round(float(min_delay_sec) * float(fs)))
    hi_s = int(round(float(max_delay_sec) * float(fs)))
    p = 0
    for r in ecg_idx:
        while p < len(ppg_idx) and int(ppg_idx[p]) < int(r) + lo_s:
            p += 1
        if p < len(ppg_idx) and int(ppg_idx[p]) <= int(r) + hi_s:
            delays.append((int(ppg_idx[p]) - int(r)) / float(fs))
    return np.asarray(delays, dtype=np.float32)


def analyze_ppg_ecg_delay(
    records: Sequence[Dict[str, Any]],
    fs: float,
    min_segment_sec: float = 10.0,
    min_delay_sec: float = 0.05,
    max_delay_sec: float = 0.60,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    min_len = int(round(float(min_segment_sec) * float(fs)))
    for rec in records:
        ppg = np.asarray(rec.get("ppg", []), dtype=np.float32)
        ecg_peaks = np.asarray(rec.get("peak_idx", []), dtype=np.int32)
        if ppg.size == 0 or ecg_peaks.size < 3:
            continue
        gate = np.asarray(rec.get("gate_track", []), dtype=np.float32)
        if gate.size == ppg.size and np.any(gate >= 0.0):
            rest_mask = gate < 0.5
            segment_source = "gate_rest"
        else:
            rest_mask = np.ones(ppg.size, dtype=bool)
            segment_source = "full_unlabeled_or_unavailable"
        for seg_start, seg_end in _contiguous_true_segments(rest_mask, min_len=min_len):
            seg_peak_mask = (ecg_peaks >= seg_start) & (ecg_peaks < seg_end)
            ecg_seg = ecg_peaks[seg_peak_mask] - seg_start
            if ecg_seg.size < 3:
                continue
            ppg_seg = ppg[seg_start:seg_end]
            best: Optional[Dict[str, Any]] = None
            for polarity in (1, -1):
                ppg_peaks = detect_ppg_pulse_peaks(ppg_seg, fs=fs, polarity=polarity)
                delays = _match_forward_delays(
                    ecg_seg,
                    ppg_peaks,
                    fs=fs,
                    min_delay_sec=min_delay_sec,
                    max_delay_sec=max_delay_sec,
                )
                coverage = float(delays.size / max(ecg_seg.size, 1))
                score = coverage - 0.20 * float(np.std(delays) if delays.size else 1.0)
                candidate = {
                    "polarity": int(polarity),
                    "delays": delays,
                    "coverage": coverage,
                    "score": score,
                    "ppg_peak_count": int(ppg_peaks.size),
                }
                if best is None or candidate["score"] > best["score"]:
                    best = candidate
            if best is None or best["delays"].size < 3:
                continue
            delays = np.asarray(best["delays"], dtype=np.float32)
            rows.append(
                {
                    "dataset": str(rec.get("dataset", "unknown")),
                    "subject": str(rec.get("subject", "unknown")),
                    "record": str(rec.get("record", "unknown")),
                    "activity": str(rec.get("activity", "unknown")),
                    "subset": str(rec.get("subset", rec.get("dataset", "unknown"))),
                    "segment_source": segment_source,
                    "segment_start_sec": float(seg_start / float(fs)),
                    "segment_end_sec": float(seg_end / float(fs)),
                    "ecg_peak_count": int(ecg_seg.size),
                    "ppg_peak_count": int(best["ppg_peak_count"]),
                    "matched_count": int(delays.size),
                    "coverage": float(best["coverage"]),
                    "polarity": int(best["polarity"]),
                    "delay_mean_sec": float(np.mean(delays)),
                    "delay_median_sec": float(np.median(delays)),
                    "delay_std_sec": float(np.std(delays)),
                    "delay_iqr_sec": float(np.quantile(delays, 0.75) - np.quantile(delays, 0.25)),
                    "delay_p05_sec": float(np.quantile(delays, 0.05)),
                    "delay_p95_sec": float(np.quantile(delays, 0.95)),
                }
            )

    def summarize_rows(group_key: str) -> Dict[str, Any]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row[group_key]), []).append(row)
        out: Dict[str, Any] = {}
        for key, vals in sorted(grouped.items()):
            weights = np.asarray([row["matched_count"] for row in vals], dtype=np.float32)
            medians = np.asarray([row["delay_median_sec"] for row in vals], dtype=np.float32)
            coverages = np.asarray([row["coverage"] for row in vals], dtype=np.float32)
            wsum = float(np.sum(weights))
            out[key] = {
                "segments": int(len(vals)),
                "matched_count": int(wsum),
                "median_delay_sec": float(np.average(medians, weights=weights)) if wsum > 0 else float("nan"),
                "median_delay_iqr_across_segments_sec": float(np.quantile(medians, 0.75) - np.quantile(medians, 0.25)) if medians.size else float("nan"),
                "mean_coverage": float(np.mean(coverages)) if coverages.size else float("nan"),
            }
        return out

    return {
        "method": {
            "ppg_peak_detector": "bandpass 0.4-8Hz + find_peaks, positive/negative polarity selected per segment",
            "match_rule": f"first PPG peak after ECG peak within {min_delay_sec:.3f}-{max_delay_sec:.3f}s",
            "segment_rule": "gate rest segments when labels exist, otherwise full unlabeled record segments",
            "min_segment_sec": float(min_segment_sec),
        },
        "record_segments": rows,
        "by_dataset": summarize_rows("dataset"),
        "by_subject": summarize_rows("subject"),
        "by_activity": summarize_rows("activity"),
        "by_subset": summarize_rows("subset"),
    }


def save_delay_analysis_plots(delay_analysis: Dict[str, Any], outdir: Path) -> None:
    rows = delay_analysis.get("record_segments", [])
    if not rows:
        return
    delays = np.asarray([row["delay_median_sec"] for row in rows], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(delays, bins=60, alpha=0.85)
    ax.set_xlabel("Median ECG-to-PPG delay per segment (s)")
    ax.set_ylabel("Segments")
    ax.set_title("PPG-ECG Delay Distribution")
    fig.tight_layout()
    fig.savefig(outdir / "ppg_ecg_delay_hist.png", dpi=160)
    plt.close(fig)

    grouped = delay_analysis.get("by_dataset", {})
    if grouped:
        names = list(grouped.keys())
        vals = [grouped[name].get("median_delay_sec", float("nan")) for name in names]
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(names)), 4))
        ax.bar(names, vals)
        ax.set_ylabel("Weighted median delay (s)")
        ax.set_title("PPG-ECG Delay by Dataset")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(outdir / "ppg_ecg_delay_by_dataset.png", dpi=160)
        plt.close(fig)


def summarize_outputs(outputs: Dict[str, Any], peak_threshold: float, gate_threshold: float) -> Dict[str, Any]:
    event_metrics = compute_event_metrics(outputs, peak_threshold=peak_threshold)
    scorecard = {
        "counts": {
            "windows": int(outputs["n_windows"]),
            "peak_samples": int(len(outputs["peak_true"])),
            "rr_samples": int(len(outputs["rr_true"])),
            "gate_windows_labeled": int(len(outputs["gate_true"])),
        },
        "losses": outputs["losses"],
        "peak_sequence": compute_binary_metrics(outputs["peak_true"], outputs["peak_prob"], peak_threshold),
        "peak_events": event_metrics["peak_events"],
        "peak_events_by_tolerance": event_metrics["peak_events_by_tolerance"],
        "hr_interval_sequence": compute_regression_metrics(outputs["rr_true"], outputs["rr_pred"]),
        "rri_event_sequence": event_metrics["rri_event_sequence"],
        "rri_event_sequence_by_tolerance": event_metrics["rri_event_sequence_by_tolerance"],
        "gate_logit": compute_binary_metrics(outputs["gate_true"], outputs["gate_prob"], gate_threshold),
        "gate_diagnostics": compute_gate_diagnostics(outputs["gate_true"], outputs["gate_prob"], gate_threshold),
    }
    return scorecard


def make_motion_detector(model_name: str, detector_cfg: DetectorConfig) -> nn.Module:
    if model_name == "A_denoiser_encoder":
        return DenoiserEncoderMotionDetector(in_channels=detector_cfg.in_channels, base_channels=detector_cfg.base_channels)
    if model_name == "B_light_cnn":
        return LightCnnMotionDetector(in_channels=detector_cfg.in_channels, base_channels=max(8, detector_cfg.base_channels // 2))
    raise ValueError(f"Unknown detector model: {model_name}")


def run_detector_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    agg = {"count": 0.0, "loss": 0.0, "acc_sum": 0.0}
    context = torch.enable_grad() if train_mode else torch.no_grad()
    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    with context:
        for batch_idx, (x, y, *_) in enumerate(progress, start=1):
            x = x.to(device)
            y = y.to(device)
            logit = model(x)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            prob = torch.sigmoid(logit)
            pred = (prob >= 0.5).float()
            bs = float(x.shape[0])
            agg["count"] += bs
            agg["loss"] += bs * float(loss.detach().cpu().item())
            agg["acc_sum"] += bs * float((pred == y).float().mean().detach().cpu().item())
            progress.set_postfix(batch=batch_idx, avg_loss=f"{agg['loss'] / max(1.0, agg['count']):.4f}")
    denom = max(1.0, agg["count"])
    return {"loss": agg["loss"] / denom, "accuracy": agg["acc_sum"] / denom}


def fit_motion_detector(
    model_name: str,
    train_records: Sequence[Dict[str, Any]],
    val_records: Sequence[Dict[str, Any]],
    detector_cfg: DetectorConfig,
    augment_cfg: Optional[AugmentConfig],
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    device: str,
    balanced_sampling: bool,
) -> Tuple[nn.Module, Dict[str, Any]]:
    train_ds, train_loader = build_detector_loader(
        train_records,
        detector_cfg=detector_cfg,
        batch_size=batch_size,
        shuffle=True,
        augment_cfg=augment_cfg,
        balanced_sampling=balanced_sampling,
    )
    val_ds, val_loader = build_detector_loader(val_records, detector_cfg=detector_cfg, batch_size=batch_size, shuffle=False, balanced_sampling=False)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"Detector {model_name}: empty train or validation windows")
    dev = torch.device(device)
    model = make_motion_detector(model_name, detector_cfg).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best = {"val_loss": float("inf"), "epoch": 0, "state_dict": None}
    history: List[Dict[str, float]] = []
    bad_epochs = 0
    pbar = tqdm(range(1, int(epochs) + 1), desc=f"Detector {model_name}", leave=True)
    for epoch in pbar:
        tr = run_detector_epoch(model, train_loader, dev, desc=f"{model_name} train {epoch}/{epochs}", optimizer=optimizer)
        va = run_detector_epoch(model, val_loader, dev, desc=f"{model_name} valid {epoch}/{epochs}", optimizer=None)
        row = {"epoch": float(epoch), **{f"train_{k}": float(v) for k, v in tr.items()}, **{f"val_{k}": float(v) for k, v in va.items()}}
        history.append(row)
        pbar.set_postfix(train_loss=f"{tr['loss']:.4f}", val_loss=f"{va['loss']:.4f}", val_acc=f"{va['accuracy']:.3f}")
        if va["loss"] < best["val_loss"]:
            best = {"val_loss": float(va["loss"]), "epoch": int(epoch), "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                break
    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])
    return model, {
        "history": history,
        "best_epoch": int(best["epoch"]),
        "best_val_loss": float(best["val_loss"]),
        "num_train_windows": int(len(train_ds)),
        "num_val_windows": int(len(val_ds)),
    }


def collect_motion_detector_outputs(
    model: nn.Module,
    records: Sequence[Dict[str, Any]],
    detector_cfg: DetectorConfig,
    batch_size: int,
    device: str,
    desc: str,
) -> Dict[str, Any]:
    ds, loader = build_detector_loader(records, detector_cfg=detector_cfg, batch_size=batch_size, shuffle=False, balanced_sampling=False)
    dev = torch.device(device)
    model.eval()
    probs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    meta: List[Dict[str, str]] = []
    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for x, y, datasets, subjects, activities, records_ in progress:
            x = x.to(dev)
            logit = model(x)
            probs.append(torch.sigmoid(logit).detach().cpu().numpy().astype(np.float32))
            labels.append(y.detach().cpu().numpy().astype(np.uint8))
            for i in range(len(datasets)):
                meta.append(
                    {
                        "dataset": str(datasets[i]),
                        "subject": str(subjects[i]),
                        "activity": str(activities[i]),
                        "record": str(records_[i]),
                    }
                )
    return {
        "n_windows": int(len(ds)),
        "prob": np.concatenate(probs).astype(np.float32) if probs else np.zeros(0, dtype=np.float32),
        "true": np.concatenate(labels).astype(np.uint8) if labels else np.zeros(0, dtype=np.uint8),
        "window_meta": meta,
    }


def summarize_motion_detector_outputs(outputs: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    y_true = outputs.get("true", np.zeros(0, dtype=np.uint8))
    y_prob = outputs.get("prob", np.zeros(0, dtype=np.float32))
    return {
        "counts": {"windows": int(outputs.get("n_windows", 0))},
        "motion_logit": compute_binary_metrics(y_true, y_prob, threshold),
        "motion_diagnostics": compute_gate_diagnostics(y_true, y_prob, threshold),
    }


def summarize_motion_groups(outputs: Dict[str, Any], threshold: float, group_key: str) -> Dict[str, Any]:
    groups: Dict[str, List[int]] = {}
    for idx, meta in enumerate(outputs.get("window_meta", [])):
        groups.setdefault(str(meta.get(group_key, "unknown")), []).append(idx)
    out: Dict[str, Any] = {}
    for key, indices in sorted(groups.items()):
        idx = np.asarray(indices, dtype=np.int64)
        sub = {
            "n_windows": int(idx.size),
            "prob": outputs["prob"][idx],
            "true": outputs["true"][idx],
            "window_meta": [outputs["window_meta"][int(i)] for i in idx],
        }
        out[key] = summarize_motion_detector_outputs(sub, threshold)
    return out


def save_motion_detector_curves(history: Sequence[Dict[str, float]], outdir: Path, prefix: str) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    for metric_key, title in (("loss", "Loss"), ("accuracy", "Accuracy")):
        fig, ax = plt.subplots(figsize=(6, 4))
        for split in ("train", "val"):
            vals = [row.get(f"{split}_{metric_key}", float("nan")) for row in history]
            if np.any(np.isfinite(vals)):
                ax.plot(epochs, vals, label=split)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"{prefix} {title}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"{prefix}_{metric_key}.png", dpi=160)
        plt.close(fig)


def save_detector_comparison_plot(summary: Dict[str, Any], outdir: Path) -> None:
    rows = []
    for model_name, model_block in summary.get("models", {}).items():
        for split_name in ("validation", "holdout", "extra_holdout"):
            metrics_block = model_block.get(split_name, {}).get("motion_logit", {})
            rows.append((model_name, split_name, metrics_block.get("f1", float("nan")), metrics_block.get("roc_auc", float("nan"))))
    if not rows:
        return
    models = sorted({r[0] for r in rows})
    splits = ["validation", "holdout", "extra_holdout"]
    width = 0.35
    x = np.arange(len(splits))
    fig, ax = plt.subplots(figsize=(7, 4))
    for mi, model_name in enumerate(models):
        vals = []
        for split in splits:
            found = [r for r in rows if r[0] == model_name and r[1] == split]
            vals.append(found[0][2] if found else float("nan"))
        ax.bar(x + (mi - 0.5) * width, vals, width=width, label=model_name)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("Motion F1")
    ax.set_title("Motion Detector A/B Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "detector_ab_f1_comparison.png", dpi=160)
    plt.close(fig)


def run_motion_detector_benchmark(
    detector_records_by_dataset: Dict[str, List[Dict[str, Any]]],
    train_subjects: Sequence[str],
    holdout_subjects: Sequence[str],
    external_holdout_dataset: str,
    detector_cfg: DetectorConfig,
    augment_cfg: Optional[AugmentConfig],
    outdir: Path,
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    device: str,
    balanced_sampling: bool,
) -> Dict[str, Any]:
    ensure_dir(outdir)
    all_records = [rec for recs in detector_records_by_dataset.values() for rec in recs]
    train_subject_set = set(train_subjects)
    holdout_subject_set = set(holdout_subjects)
    train_pool = [rec for rec in all_records if rec["subject"] in train_subject_set and rec["dataset"] != external_holdout_dataset]
    holdout_records = [rec for rec in all_records if rec["subject"] in holdout_subject_set and rec["dataset"] != external_holdout_dataset]
    extra_records = [rec for rec in all_records if rec["dataset"] == external_holdout_dataset]
    train_pool_subjects = sorted({rec["subject"] for rec in train_pool})
    if len(train_pool_subjects) >= 2:
        det_train_subjects, det_val_subjects = split_subjects(train_pool_subjects, train_ratio=0.8, seed=42)
    else:
        det_train_subjects, det_val_subjects = train_pool_subjects, train_pool_subjects
    det_train_records = [rec for rec in train_pool if rec["subject"] in set(det_train_subjects)]
    det_val_records = [rec for rec in train_pool if rec["subject"] in set(det_val_subjects)]
    summary: Dict[str, Any] = {
        "status": "skipped",
        "reason": None,
        "detector_input": {
            "channels": [
                "ppg",
                "acc_dyn_x",
                "acc_dyn_y",
                "acc_dyn_z",
                "gyro_x",
                "gyro_y",
                "gyro_z",
                "acc_mag",
                "gyro_mag",
                "jerk_mag",
            ],
            "uses_imu": True,
            "unit_handling": "acceleration inferred per record and gravity removed; missing gyro zero-filled",
        },
        "split_counts": {
            "train_records": int(len(det_train_records)),
            "validation_records": int(len(det_val_records)),
            "holdout_records": int(len(holdout_records)),
            "extra_holdout_records": int(len(extra_records)),
        },
        "models": {},
    }
    if not det_train_records or not det_val_records:
        summary["reason"] = "No labeled detector train/validation records with IMU were available."
        save_json(outdir / "detector_benchmark_summary.json", summary)
        return summary
    summary["status"] = "completed"
    for model_name in ("A_denoiser_encoder", "B_light_cnn"):
        model_dir = ensure_dir(outdir / model_name)
        model, fit_info = fit_motion_detector(
            model_name=model_name,
            train_records=det_train_records,
            val_records=det_val_records,
            detector_cfg=detector_cfg,
            augment_cfg=augment_cfg,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            lr=lr,
            device=device,
            balanced_sampling=balanced_sampling,
        )
        val_outputs = collect_motion_detector_outputs(model, det_val_records, detector_cfg, batch_size, device, desc=f"collect {model_name} validation")
        threshold = select_binary_threshold(val_outputs["true"], val_outputs["prob"], objective="bal_acc")
        holdout_outputs = collect_motion_detector_outputs(model, holdout_records, detector_cfg, batch_size, device, desc=f"collect {model_name} holdout") if holdout_records else {"n_windows": 0, "prob": np.zeros(0), "true": np.zeros(0), "window_meta": []}
        extra_outputs = collect_motion_detector_outputs(model, extra_records, detector_cfg, batch_size, device, desc=f"collect {model_name} extra") if extra_records else {"n_windows": 0, "prob": np.zeros(0), "true": np.zeros(0), "window_meta": []}
        save_binary_eval_plots(val_outputs["true"], val_outputs["prob"], threshold, model_dir, prefix="validation_motion", title_prefix=f"{model_name} Validation Motion")
        if holdout_outputs["n_windows"]:
            save_binary_eval_plots(holdout_outputs["true"], holdout_outputs["prob"], threshold, model_dir, prefix="holdout_motion", title_prefix=f"{model_name} Holdout Motion")
        if extra_outputs["n_windows"]:
            save_binary_eval_plots(extra_outputs["true"], extra_outputs["prob"], threshold, model_dir, prefix="extra_holdout_motion", title_prefix=f"{model_name} Extra Holdout Motion")
        save_motion_detector_curves(fit_info["history"], model_dir, prefix=model_name)
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "model_name": model_name,
                "detector_cfg": asdict(detector_cfg),
                "threshold": float(threshold),
                "fit_info": fit_info,
            },
            model_dir / f"{model_name}.pt",
        )
        model.eval()
        export_status: Dict[str, Any] = {
            "model_name": model_name,
            "success": False,
            "onnx_path": str(model_dir / f"{model_name}.onnx"),
            "threshold": float(threshold),
            "detector_cfg": asdict(detector_cfg),
        }
        try:
            dummy = torch.zeros(
                1,
                detector_cfg.in_channels,
                int(round(float(detector_cfg.win_sec) * float(detector_cfg.fs))),
                dtype=torch.float32,
                device=next(model.parameters()).device,
            )
            torch.onnx.export(
                model,
                dummy,
                model_dir / f"{model_name}.onnx",
                input_names=["features"],
                output_names=["motion_logit"],
                dynamic_axes={"features": {0: "batch"}, "motion_logit": {0: "batch"}},
                opset_version=18,
            )
            export_status["success"] = True
        except Exception as exc:
            export_status["error"] = str(exc)
        save_json(model_dir / "detector_export_status.json", export_status)
        save_json(
            model_dir / "detector_metadata.json",
            {
                "model_name": model_name,
                "detector_cfg": asdict(detector_cfg),
                "threshold": float(threshold),
                "input_channels": summary["detector_input"]["channels"],
                "preprocessing": summary["detector_input"]["unit_handling"],
                "output": "motion_logit; apply sigmoid and compare with threshold",
                "onnx_export": export_status,
            },
        )
        summary["models"][model_name] = {
            "fit_info": fit_info,
            "threshold": float(threshold),
            "validation": summarize_motion_detector_outputs(val_outputs, threshold),
            "holdout": summarize_motion_detector_outputs(holdout_outputs, threshold),
            "extra_holdout": summarize_motion_detector_outputs(extra_outputs, threshold),
            "validation_by_dataset": summarize_motion_groups(val_outputs, threshold, group_key="dataset"),
            "holdout_by_dataset": summarize_motion_groups(holdout_outputs, threshold, group_key="dataset"),
            "extra_holdout_by_dataset": summarize_motion_groups(extra_outputs, threshold, group_key="dataset"),
        }
    save_detector_comparison_plot(summary, outdir)
    save_json(outdir / "detector_benchmark_summary.json", summary)
    return summary


def _maybe_downsample(x: np.ndarray, y: np.ndarray, max_points: int = 20000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(x), size=max_points, replace=False)
    return x[idx], y[idx]


def save_binary_eval_plots(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, outdir: Path, prefix: str, title_prefix: str) -> None:
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    if y_true.size == 0:
        return
    y_pred = (y_prob >= threshold).astype(np.uint8)
    cm = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
    disp.plot(ax=ax, values_format="d")
    ax.set_title(f"{title_prefix} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_confusion_matrix.png", dpi=160)
    plt.close(fig)

    if len(np.unique(y_true)) >= 2:
        fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
        roc_auc = float(metrics.roc_auc_score(y_true, y_prob))
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{title_prefix} ROC")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(outdir / f"{prefix}_roc.png", dpi=160)
        plt.close(fig)

        prec, rec, _ = metrics.precision_recall_curve(y_true, y_prob)
        pr_auc = float(metrics.average_precision_score(y_true, y_prob))
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(rec, prec, label=f"AP={pr_auc:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{title_prefix} PR")
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(outdir / f"{prefix}_pr.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(y_prob[y_true == 0], bins=60, alpha=0.5, density=True, label="class 0")
    ax.hist(y_prob[y_true == 1], bins=60, alpha=0.5, density=True, label="class 1")
    ax.axvline(threshold, color="black", linestyle="--", label=f"thr={threshold:.2f}")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.set_title(f"{title_prefix} Score Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_score_hist.png", dpi=160)
    plt.close(fig)


def save_regression_plots(y_true: np.ndarray, y_pred: np.ndarray, outdir: Path, prefix: str, title_prefix: str) -> None:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if y_true.size == 0:
        return
    xs, ys = _maybe_downsample(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(xs, ys, s=6, alpha=0.25)
    lo = float(min(np.min(xs), np.min(ys)))
    hi = float(max(np.max(xs), np.max(ys)))
    ax.plot([lo, hi], [lo, hi], "--", color="gray")
    ax.set_xlabel("Target IBI (s)")
    ax.set_ylabel("Predicted IBI (s)")
    ax.set_title(f"{title_prefix} Pred vs True")
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_scatter.png", dpi=160)
    plt.close(fig)

    diff = y_pred - y_true
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(diff, bins=80, alpha=0.8)
    ax.set_xlabel("Prediction error (s)")
    ax.set_ylabel("Count")
    ax.set_title(f"{title_prefix} Error Histogram")
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_error_hist.png", dpi=160)
    plt.close(fig)

    mean_xy = (y_pred + y_true) * 0.5
    mx, md = _maybe_downsample(mean_xy, diff)
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(mx, md, s=6, alpha=0.25)
    ax.axhline(mean_diff, color="black", linestyle="--", label="mean")
    ax.axhline(mean_diff + 1.96 * std_diff, color="red", linestyle=":")
    ax.axhline(mean_diff - 1.96 * std_diff, color="red", linestyle=":")
    ax.set_xlabel("Mean of pred/true IBI (s)")
    ax.set_ylabel("Pred - True (s)")
    ax.set_title(f"{title_prefix} Bland-Altman")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_bland_altman.png", dpi=160)
    plt.close(fig)


def save_event_error_plots(
    outputs: Dict[str, Any],
    peak_threshold: float,
    outdir: Path,
    prefix: str,
    title_prefix: str,
    tolerance_sec: float = PPG_MAIN_EVENT_TOLERANCE_SEC,
) -> None:
    arrays = compute_event_error_arrays(outputs, peak_threshold=peak_threshold, tolerance_sec=tolerance_sec)
    plot_specs = [
        ("peak_timing_errors", "Peak timing error (s)", f"{prefix}_peak_timing_error_hist.png"),
        ("rri_errors", "RRi sequence error (s)", f"{prefix}_rri_error_hist.png"),
    ]
    for key, xlabel, filename in plot_specs:
        errors = np.asarray(arrays[key], dtype=np.float32)
        if errors.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(errors, bins=80, alpha=0.85)
        ax.axvline(float(np.mean(errors)), color="black", linestyle="--", label=f"bias={np.mean(errors):.3f}s")
        ax.axvline(0.0, color="gray", linestyle=":")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(f"{title_prefix} {xlabel} (+/-{int(round(tolerance_sec * 1000.0))} ms)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / filename, dpi=160)
        plt.close(fig)


def save_cv_training_curves(fold_histories: Sequence[Dict[str, Any]], outdir: Path) -> None:
    metric_pairs = [
        ("total", "Total Loss"),
        ("peak", "Peak Loss"),
        ("beat", "Beat Timing Loss"),
        ("ibi", "IBI Loss"),
        ("gate", "Gate Loss"),
        ("domain", "Domain Loss"),
        ("peak_f1", "Peak F1"),
        ("gate_acc", "Gate Accuracy"),
    ]
    for metric_key, title in metric_pairs:
        fig, ax = plt.subplots(figsize=(6, 4))
        plotted = False
        for fold_idx, fold_info in enumerate(fold_histories, start=1):
            hist = fold_info["history"]
            epochs = [row["epoch"] for row in hist]
            train_key = f"train_{metric_key}"
            val_key = f"val_{metric_key}"
            train_vals = [row.get(train_key, float("nan")) for row in hist]
            val_vals = [row.get(val_key, float("nan")) for row in hist]
            if np.any(np.isfinite(train_vals)):
                ax.plot(epochs, train_vals, alpha=0.65, label=f"fold{fold_idx} train" if fold_idx <= 3 else None)
                plotted = True
            if np.any(np.isfinite(val_vals)):
                ax.plot(epochs, val_vals, linestyle="--", alpha=0.65, label=f"fold{fold_idx} val" if fold_idx <= 3 else None)
                plotted = True
        if plotted:
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.set_title(f"Cross-Validation {title}")
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(outdir / f"cv_{metric_key}_curves.png", dpi=160)
        plt.close(fig)


def save_final_training_curves(final_history: Sequence[Dict[str, Any]], outdir: Path) -> None:
    if not final_history:
        return
    metric_pairs = [
        ("total", "Total Loss"),
        ("peak", "Peak Loss"),
        ("beat", "Beat Timing Loss"),
        ("ibi", "IBI Loss"),
        ("gate", "Gate Loss"),
        ("domain", "Domain Loss"),
    ]
    epochs = [row["epoch"] for row in final_history]
    for metric_key, title in metric_pairs:
        values = [row.get(f"train_{metric_key}", float("nan")) for row in final_history]
        if not np.any(np.isfinite(values)):
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, values)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"Final Train {title}")
        fig.tight_layout()
        fig.savefig(outdir / f"final_train_{metric_key}.png", dpi=160)
        plt.close(fig)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_deploy_bundle(
    model: PeakIntervalGateNet,
    run_dir: Path,
    model_cfg: ModelConfig,
    window_cfg: WindowConfig,
    thresholds: Dict[str, float],
    split_info: Dict[str, Any],
) -> Dict[str, Any]:
    deploy_dir = ensure_dir(run_dir / "deploy_bundle")
    export_status: Dict[str, Any] = {
        "success": False,
        "format": "onnx",
        "model_path": str(deploy_dir / "peak_hr_gate_model.onnx"),
        "config_path": str(deploy_dir / "model_reuse_params.json"),
        "error": None,
    }
    reuse_payload = {
        "model_family": "PeakIntervalGateNet",
        "cpu_only_reuse": True,
        "model_cfg": asdict(model_cfg),
        "window_cfg": asdict(window_cfg),
        "thresholds": thresholds,
        "input_spec": {
            "name": "ppg_input",
            "shape": [
                "batch",
                int(model_cfg.in_channels),
                int(round(window_cfg.win_sec * window_cfg.fs)),
            ],
            "dtype": "float32",
            "preprocess": {
                "ppg_filter": "bandpass already applied in training loader",
                "normalization": "per-record zscore_1d",
            },
        },
        "output_spec": {
            "peak_logit": {"shape": ["batch", int(round(window_cfg.win_sec * window_cfg.fs))], "postprocess": "sigmoid then threshold with peak_threshold"},
            "ibi_pred": {"shape": ["batch", int(round(window_cfg.win_sec * window_cfg.fs))], "unit": "seconds"},
            "gate_logit": {"shape": ["batch"], "postprocess": "sigmoid then threshold with gate_threshold"},
        },
        "split_summary": {
            "external_holdout_dataset": split_info["external_holdout_dataset"],
            "train_subject_count": len(split_info["train_subjects"]),
            "holdout_subject_count": len(split_info["holdout_subjects"]),
            "extra_holdout_subject_count": len(split_info["extra_holdout_subjects"]),
        },
    }
    save_json(deploy_dir / "model_reuse_params.json", reuse_payload)
    readme = "\n".join(
        [
            "CPU-only deploy bundle",
            "",
            "Files:",
            "- peak_hr_gate_model.onnx : model weights for ONNX Runtime inference",
            "- model_reuse_params.json : windowing, thresholds, I/O specification, and deployment metadata",
            "",
            "Inference contract:",
            "- input: float32 normalized PPG window shaped [batch, 1, window_samples]",
            "- outputs: peak_logit, ibi_pred, gate_logit",
            "- postprocess: sigmoid on peak_logit and gate_logit, then apply exported thresholds",
        ]
    )
    (deploy_dir / "README.txt").write_text(readme, encoding="utf-8")

    try:
        onnx_path = deploy_dir / "peak_hr_gate_model.onnx"
        wrapper = PeakIntervalGateOnnxWrapper(model.cpu().eval())
        dummy = torch.zeros(
            1,
            int(model_cfg.in_channels),
            int(round(window_cfg.win_sec * window_cfg.fs)),
            dtype=torch.float32,
        )
        torch.onnx.export(
            wrapper,
            dummy,
            onnx_path,
            input_names=["ppg_input"],
            output_names=["peak_logit", "ibi_pred", "gate_logit"],
            opset_version=18,
        )
        export_status["success"] = True
    except Exception as exc:
        export_status["error"] = str(exc)
    save_json(deploy_dir / "export_status.json", export_status)
    return export_status


def write_scorecard_markdown(
    path: Path,
    config_summary: Dict[str, Any],
    split_info: Dict[str, Any],
    thresholds: Dict[str, float],
    cv_summary: Dict[str, Any],
    holdout_summary: Dict[str, Any],
    extra_summary: Dict[str, Any],
    group_scorecards: Optional[Dict[str, Any]] = None,
    delay_analysis: Optional[Dict[str, Any]] = None,
    lodo_summary: Optional[Dict[str, Any]] = None,
    detector_benchmark_summary: Optional[Dict[str, Any]] = None,
) -> None:
    def fmt_num(v: Any) -> str:
        if v is None:
            return "NA"
        if isinstance(v, float):
            if np.isnan(v) or np.isinf(v):
                return "NA"
            return f"{v:.4f}"
        return str(v)

    def head_section(name: str, section: Dict[str, Any]) -> List[str]:
        lines = [f"### {name}"]
        counts = section.get("counts", {})
        if counts:
            lines.append(f"- windows: {counts.get('windows', 'NA')}")
            lines.append(f"- peak_samples: {counts.get('peak_samples', 'NA')}")
            lines.append(f"- rr_samples: {counts.get('rr_samples', 'NA')}")
            lines.append(f"- gate_windows_labeled: {counts.get('gate_windows_labeled', 'NA')}")
        for head_name in ("peak_sequence", "peak_events", "hr_interval_sequence", "rri_event_sequence", "gate_logit"):
            metrics_block = section.get(head_name, {})
            lines.append(f"- {head_name}:")
            for key, value in metrics_block.items():
                if key == "confusion_matrix":
                    lines.append(f"  confusion_matrix: {value}")
                else:
                    lines.append(f"  {key}: {fmt_num(value)}")
        peak_layers = section.get("peak_events_by_tolerance", {})
        if peak_layers:
            lines.append("- peak_events_by_tolerance:")
            lines.append("  | tolerance | TP | FP | FN | precision | recall | F1 | timing_MAE_s | timing_median_AE_s |")
            lines.append("  |---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for key in sorted(peak_layers, key=lambda x: int(str(x).replace("ms", ""))):
                block = peak_layers[key]
                lines.append(
                    "  | "
                    f"{key} | "
                    f"{block.get('tp', 'NA')} | "
                    f"{block.get('fp', 'NA')} | "
                    f"{block.get('fn', 'NA')} | "
                    f"{fmt_num(block.get('precision'))} | "
                    f"{fmt_num(block.get('recall'))} | "
                    f"{fmt_num(block.get('f1'))} | "
                    f"{fmt_num(block.get('timing_mae'))} | "
                    f"{fmt_num(block.get('timing_median_ae'))} |"
                )
        rri_layers = section.get("rri_event_sequence_by_tolerance", {})
        if rri_layers:
            lines.append("- rri_event_sequence_by_tolerance:")
            lines.append("  | tolerance | n | bias_s | MAE_s | median_AE_s | std_s | variance_s2 |")
            lines.append("  |---|---:|---:|---:|---:|---:|---:|")
            for key in sorted(rri_layers, key=lambda x: int(str(x).replace("ms", ""))):
                block = rri_layers[key]
                lines.append(
                    "  | "
                    f"{key} | "
                    f"{block.get('n', 'NA')} | "
                    f"{fmt_num(block.get('bias'))} | "
                    f"{fmt_num(block.get('mae'))} | "
                    f"{fmt_num(block.get('median_ae'))} | "
                    f"{fmt_num(block.get('std'))} | "
                    f"{fmt_num(block.get('variance'))} |"
                )
        return lines

    def ecg_detector_section(result: Optional[Dict[str, Any]]) -> List[str]:
        lines = ["## ECG Peak Detector Preflight"]
        if not result:
            lines.append("- status: disabled")
            return lines
        lines.extend(
            [
                f"- status: {'passed' if result.get('passed') else 'failed'}",
                f"- dataset: {result.get('dataset', 'NA')}",
                f"- records: {result.get('records', 'NA')}",
                f"- match_tolerance_sec: {fmt_num(result.get('tolerance_sec'))}",
                f"- required_peak_f1: {fmt_num(result.get('min_peak_f1'))}",
                f"- required_ibi_accuracy: {fmt_num(result.get('min_ibi_accuracy'))}",
                f"- peak_tp: {result.get('tp', 'NA')}",
                f"- peak_fp: {result.get('fp', 'NA')}",
                f"- peak_fn: {result.get('fn', 'NA')}",
                f"- peak_precision: {fmt_num(result.get('peak_precision'))}",
                f"- peak_recall: {fmt_num(result.get('peak_recall'))}",
                f"- peak_f1: {fmt_num(result.get('peak_f1'))}",
                f"- peak_timing_bias_sec: {fmt_num(result.get('peak_timing_bias_sec'))}",
                f"- peak_timing_mae_sec: {fmt_num(result.get('peak_timing_mae_sec'))}",
                f"- peak_timing_median_ae_sec: {fmt_num(result.get('peak_timing_median_ae_sec'))}",
                f"- peak_timing_std_sec: {fmt_num(result.get('peak_timing_std_sec'))}",
                f"- peak_timing_variance_sec2: {fmt_num(result.get('peak_timing_variance_sec2'))}",
                f"- ibi_pairs: {result.get('ibi_pairs', 'NA')}",
                f"- ibi_accuracy: {fmt_num(result.get('ibi_accuracy'))}",
                f"- ibi_mae_sec: {fmt_num(result.get('ibi_mae_sec'))}",
                f"- ibi_median_ae_sec: {fmt_num(result.get('ibi_median_ae_sec'))}",
                f"- ibi_std_sec: {fmt_num(result.get('ibi_std_sec'))}",
                f"- ibi_variance_sec2: {fmt_num(result.get('ibi_variance_sec2'))}",
                f"- low_peak_f1_record_count: {len(result.get('low_peak_f1_records', []))}",
                f"- low_ibi_accuracy_record_count: {len(result.get('low_ibi_accuracy_records', []))}",
            ]
        )
        worst = result.get("worst_peak_f1_records", [])[:10]
        if worst:
            lines.extend(["", "### Worst ECG detector records by peak F1"])
            lines.append("| record | true_peaks | pred_peaks | F1 | timing_MAE_s | IBI_acc |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for row in worst:
                lines.append(
                    "| "
                    f"{row.get('record', 'NA')} | "
                    f"{row.get('true_peaks', 'NA')} | "
                    f"{row.get('pred_peaks', 'NA')} | "
                    f"{fmt_num(row.get('peak_f1'))} | "
                    f"{fmt_num(row.get('peak_timing_mae_sec'))} | "
                    f"{fmt_num(row.get('ibi_accuracy'))} |"
                )
        return lines

    def gate_diagnostic_lines(name: str, section: Dict[str, Any]) -> List[str]:
        diag = section.get("gate_diagnostics", {})
        if not diag:
            return []
        keys = [
            "true_motion_rate",
            "pred_motion_rate",
            "near_threshold_frac_0p05",
            "threshold_percentile",
            "rest_score_mean",
            "motion_score_mean",
            "class_score_gap",
            "class_separation_cohen_d",
            "rest_false_motion_rate",
            "motion_false_rest_rate",
        ]
        lines = [f"### {name} gate diagnostics"]
        for key in keys:
            lines.append(f"- {key}: {fmt_num(diag.get(key))}")
        return lines

    def group_table_lines(title: str, bundle: Dict[str, Any], group_name: str = "by_dataset") -> List[str]:
        groups = bundle.get(group_name, {}) if bundle else {}
        if not groups:
            return [f"### {title}", "- no grouped metrics available"]
        lines = [f"### {title}"]
        lines.append("| group | windows | peak_event_F1_20ms | peak_seq_F1 | IBI_MAE_s | gate_F1 | gate_AUC | gate_pred_motion_rate |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for key, section in sorted(groups.items()):
            lines.append(
                "| "
                f"{key} | "
                f"{section.get('counts', {}).get('windows', 'NA')} | "
                f"{fmt_num(section.get('peak_events', {}).get('f1'))} | "
                f"{fmt_num(section.get('peak_sequence', {}).get('f1'))} | "
                f"{fmt_num(section.get('hr_interval_sequence', {}).get('mae'))} | "
                f"{fmt_num(section.get('gate_logit', {}).get('f1'))} | "
                f"{fmt_num(section.get('gate_logit', {}).get('roc_auc'))} | "
                f"{fmt_num(section.get('gate_diagnostics', {}).get('pred_motion_rate'))} |"
            )
        return lines

    def delay_lines(result: Optional[Dict[str, Any]]) -> List[str]:
        lines = ["## PPG-ECG Delay Analysis"]
        if not result:
            lines.append("- status: disabled or unavailable")
            return lines
        lines.append(f"- analyzed_segments: {len(result.get('record_segments', []))}")
        by_dataset = result.get("by_dataset", {})
        if by_dataset:
            lines.append("| dataset | segments | matched_count | median_delay_s | delay_IQR_across_segments_s | mean_coverage |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for key, row in sorted(by_dataset.items()):
                lines.append(
                    "| "
                    f"{key} | "
                    f"{row.get('segments', 'NA')} | "
                    f"{row.get('matched_count', 'NA')} | "
                    f"{fmt_num(row.get('median_delay_sec'))} | "
                    f"{fmt_num(row.get('median_delay_iqr_across_segments_sec'))} | "
                    f"{fmt_num(row.get('mean_coverage'))} |"
                )
        return lines

    def lodo_lines(result: Optional[Dict[str, Any]]) -> List[str]:
        lines = ["## Leave-One-Dataset-Out Validation"]
        if not result or not result.get("datasets"):
            lines.append("- status: disabled or unavailable")
            return lines
        lines.append("| heldout_dataset | records | peak_event_F1_20ms | peak_seq_F1 | IBI_MAE_s | gate_F1 | gate_AUC |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for key, row in sorted(result.get("datasets", {}).items()):
            section = row.get("summary", {})
            lines.append(
                "| "
                f"{key} | "
                f"{row.get('heldout_record_count', 'NA')} | "
                f"{fmt_num(section.get('peak_events', {}).get('f1'))} | "
                f"{fmt_num(section.get('peak_sequence', {}).get('f1'))} | "
                f"{fmt_num(section.get('hr_interval_sequence', {}).get('mae'))} | "
                f"{fmt_num(section.get('gate_logit', {}).get('f1'))} | "
                f"{fmt_num(section.get('gate_logit', {}).get('roc_auc'))} |"
            )
        return lines

    def detector_benchmark_lines(result: Optional[Dict[str, Any]]) -> List[str]:
        lines = ["## Motion Detector Benchmark"]
        if not result:
            lines.append("- status: disabled")
            return lines
        lines.extend(
            [
                f"- status: {result.get('status', 'NA')}",
                f"- reason: {result.get('reason', 'NA')}",
                f"- uses_imu: {(result.get('detector_input') or {}).get('uses_imu', 'NA')}",
                f"- input_channels: {(result.get('detector_input') or {}).get('channels', 'NA')}",
                f"- unit_handling: {(result.get('detector_input') or {}).get('unit_handling', 'NA')}",
                f"- detector_A: denoiser-style encoder reused as a classifier; artifact decoder removed; motion head added",
                f"- detector_B: directly trained lightweight CNN motion detector",
            ]
        )
        split_counts = result.get("split_counts", {})
        if split_counts:
            lines.append(
                "- split_counts: "
                f"train_records={split_counts.get('train_records', 'NA')}, "
                f"validation_records={split_counts.get('validation_records', 'NA')}, "
                f"holdout_records={split_counts.get('holdout_records', 'NA')}, "
                f"extra_holdout_records={split_counts.get('extra_holdout_records', 'NA')}"
            )
        models = result.get("models", {})
        if not models:
            return lines
        lines.append("| model | split | windows | threshold | precision | recall | F1 | balanced_acc | ROC-AUC | PR-AUC | confusion_matrix |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for model_name, model_block in sorted(models.items()):
            for split_name in ("validation", "holdout", "extra_holdout"):
                block = model_block.get(split_name, {})
                metric_block = block.get("motion_logit", {})
                lines.append(
                    "| "
                    f"{model_name} | "
                    f"{split_name} | "
                    f"{block.get('counts', {}).get('windows', metric_block.get('n', 'NA'))} | "
                    f"{fmt_num(metric_block.get('threshold'))} | "
                    f"{fmt_num(metric_block.get('precision'))} | "
                    f"{fmt_num(metric_block.get('recall'))} | "
                    f"{fmt_num(metric_block.get('f1'))} | "
                    f"{fmt_num(metric_block.get('balanced_accuracy'))} | "
                    f"{fmt_num(metric_block.get('roc_auc'))} | "
                    f"{fmt_num(metric_block.get('pr_auc'))} | "
                    f"{metric_block.get('confusion_matrix', 'NA')} |"
                )
        lines.append("")
        lines.append("### Detector grouped by dataset")
        for model_name, model_block in sorted(models.items()):
            lines.append(f"#### {model_name}")
            lines.append("| split | dataset | windows | F1 | balanced_acc | ROC-AUC | pred_motion_rate | motion_false_rest_rate |")
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
            for split_name, key in (
                ("validation", "validation_by_dataset"),
                ("holdout", "holdout_by_dataset"),
                ("extra_holdout", "extra_holdout_by_dataset"),
            ):
                for dataset_name, section in sorted(model_block.get(key, {}).items()):
                    metric_block = section.get("motion_logit", {})
                    diag = section.get("motion_diagnostics", {})
                    lines.append(
                        "| "
                        f"{split_name} | "
                        f"{dataset_name} | "
                        f"{section.get('counts', {}).get('windows', metric_block.get('n', 'NA'))} | "
                        f"{fmt_num(metric_block.get('f1'))} | "
                        f"{fmt_num(metric_block.get('balanced_accuracy'))} | "
                        f"{fmt_num(metric_block.get('roc_auc'))} | "
                        f"{fmt_num(diag.get('pred_motion_rate'))} | "
                        f"{fmt_num(diag.get('motion_false_rest_rate'))} |"
                    )
        return lines

    lines: List[str] = [
        "# Peak / HR Interval / Gate Scorecard",
        "",
        "## Configuration",
        f"- results_root: {config_summary['results_root']}",
        f"- run_name: {config_summary['run_name']}",
        f"- external_holdout_dataset: {split_info['external_holdout_dataset']}",
        f"- internal_train_subject_count: {len(split_info['train_subjects'])}",
        f"- internal_holdout_subject_count: {len(split_info['holdout_subjects'])}",
        f"- extra_holdout_subject_count: {len(split_info['extra_holdout_subjects'])}",
        f"- cv_folds: {config_summary['cv_folds']}",
        f"- final_train_epochs: {config_summary['final_train_epochs']}",
        f"- norm_type: {config_summary.get('norm_type', 'NA')}",
        f"- balanced_sampling: {config_summary.get('balanced_sampling', 'NA')}",
        f"- augmentation_enabled: {config_summary.get('augmentation_enabled', 'NA')}",
        f"- domain_adversarial_lambda: {fmt_num(config_summary.get('domain_adversarial_lambda'))}",
        f"- worst_domain_weight: {fmt_num(config_summary.get('worst_domain_weight'))}",
        f"- group_scorecards: {config_summary.get('group_scorecards', 'NA')}",
        f"- delay_analysis: {config_summary.get('delay_analysis', 'NA')}",
        f"- lodo_validation: {config_summary.get('lodo_validation', 'NA')}",
        f"- detector_benchmark: {config_summary.get('detector_benchmark', 'NA')}",
        f"- detector_epochs: {config_summary.get('detector_epochs', 'NA')}",
        f"- detector_model_base_channels: {config_summary.get('detector_base_channels', 'NA')}",
        f"- gate_input: {(config_summary.get('gate_input_audit') or {}).get('gate_head_input', 'NA')}",
        f"- gate_uses_imu: {(config_summary.get('gate_input_audit') or {}).get('uses_imu_in_gate_head', 'NA')}",
        f"- mimic_subsets: {config_summary.get('mimic_subsets', 'NA')}",
        f"- mimic_max_records: {config_summary.get('mimic_max_records', 'NA')}",
        f"- include_mimic_special_extra_holdout: {config_summary.get('include_mimic_special_extra_holdout', 'NA')}",
        f"- mimic_extra_holdout_subsets: {config_summary.get('mimic_extra_holdout_subsets', 'NA')}",
        f"- mimic_extra_holdout_mat_files: {config_summary.get('mimic_extra_holdout_mat_files', 'NA')}",
        f"- enable_vitaldb: {config_summary.get('enable_vitaldb', 'NA')}",
        f"- vitaldb_max_cases: {config_summary.get('vitaldb_max_cases', 'NA')}",
        f"- ecg_detector_preflight: {config_summary.get('ecg_detector_preflight', 'NA')}",
        f"- ecg_preflight_peak_f1: {fmt_num((config_summary.get('ecg_preflight_result') or {}).get('peak_f1'))}",
        f"- ecg_preflight_ibi_accuracy: {fmt_num((config_summary.get('ecg_preflight_result') or {}).get('ibi_accuracy'))}",
        "",
        "## Thresholds",
        f"- peak_threshold: {fmt_num(thresholds['peak_threshold'])}",
        f"- gate_threshold: {fmt_num(thresholds['gate_threshold'])}",
        f"- ppg_main_event_tolerance_ms: {int(round(PPG_MAIN_EVENT_TOLERANCE_SEC * 1000.0))}",
        f"- ppg_layered_event_tolerances_ms: {[int(round(x * 1000.0)) for x in PPG_EVENT_TOLERANCE_SECS]}",
        "",
    ]
    lines.extend(ecg_detector_section(config_summary.get("ecg_preflight_result")))
    lines.extend(
        [
            "",
            "## Cross-Validation Aggregate",
        ]
    )
    lines.extend(head_section("cross_validation", cv_summary["aggregate"]))
    lines.extend(
        [
            "",
            "## Holdout",
        ]
    )
    lines.extend(head_section("holdout", holdout_summary))
    lines.extend(
        [
            "",
            "## Extra Holdout",
        ]
    )
    lines.extend(head_section("extra_holdout", extra_summary))
    lines.extend(["", "## Gate Diagnostics"])
    lines.extend(gate_diagnostic_lines("cross_validation", cv_summary["aggregate"]))
    lines.extend(gate_diagnostic_lines("holdout", holdout_summary))
    lines.extend(gate_diagnostic_lines("extra_holdout", extra_summary))
    lines.extend(["", "## Grouped Scorecards"])
    group_scorecards = group_scorecards or {}
    lines.extend(group_table_lines("Cross-validation by dataset", group_scorecards.get("cross_validation", {}), "by_dataset"))
    lines.extend(group_table_lines("Holdout by dataset", group_scorecards.get("holdout", {}), "by_dataset"))
    lines.extend(group_table_lines("Extra holdout by dataset/subset", group_scorecards.get("extra_holdout", {}), "by_dataset_subset"))
    lines.extend(["", "## Gate Input / IMU Audit"])
    audit = config_summary.get("gate_input_audit") or {}
    lines.extend(
        [
            f"- model_input_channels: {audit.get('model_input_channels', 'NA')}",
            f"- model_input_names: {audit.get('model_input_names', 'NA')}",
            f"- gate_head_input: {audit.get('gate_head_input', 'NA')}",
            f"- uses_imu_in_gate_head: {audit.get('uses_imu_in_gate_head', 'NA')}",
            f"- note: {audit.get('important_note', 'NA')}",
        ]
    )
    raw_imu = audit.get("raw_imu_availability", {})
    if raw_imu:
        lines.append("| dataset | raw_IMU_columns_available | loaded_into_model | unit_status | gravity_removal_status |")
        lines.append("|---|---:|---:|---|---|")
        for key, row in sorted(raw_imu.items()):
            lines.append(
                "| "
                f"{key} | "
                f"{row.get('raw_imu_columns_available', 'NA')} | "
                f"{row.get('in_loaded_model_record', 'NA')} | "
                f"{row.get('unit_status', 'NA')} | "
                f"{row.get('gravity_removal_status', 'NA')} |"
            )
    lines.extend([""])
    lines.extend(delay_lines(delay_analysis))
    lines.extend([""])
    lines.extend(lodo_lines(lodo_summary))
    lines.extend([""])
    lines.extend(detector_benchmark_lines(detector_benchmark_summary))
    lines.extend(["", "## Cross-Validation Fold Summary"])
    for fold in cv_summary["folds"]:
        lines.append(f"### fold_{fold['fold']}")
        lines.append(f"- best_epoch: {fold['best_epoch']}")
        lines.append(f"- train_subject_count: {len(fold['train_subjects'])}")
        lines.append(f"- val_subject_count: {len(fold['val_subjects'])}")
        for head_name in ("peak_sequence", "peak_events", "hr_interval_sequence", "rri_event_sequence", "gate_logit"):
            block = fold["summary"][head_name]
            lines.append(f"- {head_name}:")
            for key, value in block.items():
                if key == "confusion_matrix":
                    lines.append(f"  confusion_matrix: {value}")
                else:
                    lines.append(f"  {key}: {fmt_num(value)}")
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "- simultaneous_measurements uses .atr consensus beat annotations when available and .aux phase markers for gate supervision.",
            "- iamwell currently contributes peak/IBI supervision only; its gate labels remain unavailable in this script.",
            "- mimic_perform contributes PPG/ECG-derived pseudo peak/IBI supervision from local CSV files; CSV and WFDB mirrors should not be loaded together.",
            "- mimic_perform_af_csv, mimic_perform_non_af_csv, MIMIC_PERform_1_min_noisy.mat, and MIMIC_PERform_1_min_neonate.mat are reserved for extra-holdout by default.",
            "- vitaldb_open is optional and uses the vitaldb Python API with SNUADC/PLETH and SNUADC/ECG_II when --enable_vitaldb is set.",
            "- ECG pseudo-label training is blocked unless the PTT ECG detector preflight passes aggregate peak F1 and IBI accuracy thresholds.",
            "- peak_sequence is point-wise over the dense peak target; peak_events is the main beat-level metric at +/-20 ms.",
            "- peak_events_by_tolerance reports layered PPG beat matching at +/-10, +/-20, +/-30, and +/-40 ms.",
            "- rri_event_sequence is the main matched-beat interval metric at +/-20 ms; rri_event_sequence_by_tolerance reports the same layers.",
            "- gate_logit remains an auxiliary PPG-only state head inside the peak/IBI model; the Motion Detector Benchmark is the dedicated IMU-aware rest/motion detector module.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_cross_validation(
    train_records: Sequence[Dict[str, Any]],
    window_cfg: WindowConfig,
    model_cfg: ModelConfig,
    loss_cfg: LossConfig,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    cv_folds: int,
    device: str,
    seed: int,
    augment_cfg: Optional[AugmentConfig],
    balanced_sampling: bool,
    domain_lambda: float,
) -> Dict[str, Any]:
    subjects = sorted({rec["subject"] for rec in train_records})
    splits = group_kfold_subject_splits(subjects, n_splits=cv_folds, seed=seed)
    fold_histories: List[Dict[str, Any]] = []
    fold_outputs: List[Dict[str, Any]] = []

    for fold_idx, (fold_train_subjects, fold_val_subjects) in enumerate(splits, start=1):
        fold_train_records = [rec for rec in train_records if rec["subject"] in set(fold_train_subjects)]
        fold_val_records = [rec for rec in train_records if rec["subject"] in set(fold_val_subjects)]
        model, fit_info = fit_model_with_validation(
            train_records=fold_train_records,
            val_records=fold_val_records,
            window_cfg=window_cfg,
            model_cfg=model_cfg,
            loss_cfg=loss_cfg,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            device=device,
            title=f"CV fold {fold_idx}",
            augment_cfg=augment_cfg,
            balanced_sampling=balanced_sampling,
            domain_lambda=domain_lambda,
        )
        outputs = collect_outputs(
            model=model,
            records=fold_val_records,
            window_cfg=window_cfg,
            loss_cfg=loss_cfg,
            batch_size=batch_size,
            device=device,
            desc=f"collect cv fold {fold_idx}",
        )
        fold_histories.append(
            {
                "fold": fold_idx,
                "history": fit_info["history"],
                "best_epoch": fit_info["best_epoch"],
                "train_subjects": fold_train_subjects,
                "val_subjects": fold_val_subjects,
            }
        )
        fold_outputs.append(
            {
                "fold": fold_idx,
                "outputs": outputs,
                "best_epoch": fit_info["best_epoch"],
                "train_subjects": fold_train_subjects,
                "val_subjects": fold_val_subjects,
            }
        )

    merged = merge_output_dicts([item["outputs"] for item in fold_outputs])
    peak_threshold = select_binary_threshold(merged["peak_true"], merged["peak_prob"], objective="f1")
    gate_threshold = select_binary_threshold(merged["gate_true"], merged["gate_prob"], objective="bal_acc")

    fold_summaries: List[Dict[str, Any]] = []
    for item in fold_outputs:
        fold_summaries.append(
            {
                "fold": int(item["fold"]),
                "best_epoch": int(item["best_epoch"]),
                "train_subjects": item["train_subjects"],
                "val_subjects": item["val_subjects"],
                "summary": summarize_outputs(item["outputs"], peak_threshold=peak_threshold, gate_threshold=gate_threshold),
            }
        )

    aggregate_summary = summarize_outputs(merged, peak_threshold=peak_threshold, gate_threshold=gate_threshold)
    median_best_epoch = max(1, int(round(float(np.median([item["best_epoch"] for item in fold_outputs])))))
    return {
        "fold_histories": fold_histories,
        "folds": fold_summaries,
        "aggregate_outputs": merged,
        "aggregate": aggregate_summary,
        "thresholds": {"peak_threshold": float(peak_threshold), "gate_threshold": float(gate_threshold)},
        "final_train_epochs": int(median_best_epoch),
    }


def run_leave_one_dataset_out(
    records: Sequence[Dict[str, Any]],
    window_cfg: WindowConfig,
    model_cfg: ModelConfig,
    loss_cfg: LossConfig,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    device: str,
    seed: int,
    augment_cfg: Optional[AugmentConfig],
    balanced_sampling: bool,
    domain_lambda: float,
    max_datasets: int = 0,
) -> Dict[str, Any]:
    datasets = sorted({str(rec.get("dataset", "unknown")) for rec in records})
    if max_datasets > 0:
        datasets = datasets[: int(max_datasets)]
    out: Dict[str, Any] = {"datasets": {}, "notes": "Each row trains on all other datasets and evaluates on the left-out dataset."}
    for held_dataset in datasets:
        held_records = [rec for rec in records if str(rec.get("dataset", "unknown")) == held_dataset]
        train_pool = [rec for rec in records if str(rec.get("dataset", "unknown")) != held_dataset]
        if not held_records or len({rec["subject"] for rec in train_pool}) < 2:
            continue
        lodo_train_subjects, lodo_val_subjects = split_subjects(
            [rec["subject"] for rec in train_pool],
            train_ratio=0.8,
            seed=seed,
        )
        lodo_train_records = [rec for rec in train_pool if rec["subject"] in set(lodo_train_subjects)]
        lodo_val_records = [rec for rec in train_pool if rec["subject"] in set(lodo_val_subjects)]
        if not lodo_train_records or not lodo_val_records:
            continue
        model, fit_info = fit_model_with_validation(
            train_records=lodo_train_records,
            val_records=lodo_val_records,
            window_cfg=window_cfg,
            model_cfg=model_cfg,
            loss_cfg=loss_cfg,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            device=device,
            title=f"LODO {held_dataset}",
            augment_cfg=augment_cfg,
            balanced_sampling=balanced_sampling,
            domain_lambda=domain_lambda,
        )
        val_outputs = collect_outputs(
            model=model,
            records=lodo_val_records,
            window_cfg=window_cfg,
            loss_cfg=loss_cfg,
            batch_size=batch_size,
            device=device,
            desc=f"collect LODO {held_dataset} val",
        )
        peak_threshold = select_binary_threshold(val_outputs["peak_true"], val_outputs["peak_prob"], objective="f1")
        gate_threshold = select_binary_threshold(val_outputs["gate_true"], val_outputs["gate_prob"], objective="bal_acc")
        held_outputs = collect_outputs(
            model=model,
            records=held_records,
            window_cfg=window_cfg,
            loss_cfg=loss_cfg,
            batch_size=batch_size,
            device=device,
            desc=f"collect LODO {held_dataset}",
        )
        out["datasets"][held_dataset] = {
            "heldout_record_count": int(len(held_records)),
            "train_record_count": int(len(lodo_train_records)),
            "val_record_count": int(len(lodo_val_records)),
            "best_epoch": int(fit_info["best_epoch"]),
            "thresholds": {"peak_threshold": float(peak_threshold), "gate_threshold": float(gate_threshold)},
            "summary": summarize_outputs(held_outputs, peak_threshold=peak_threshold, gate_threshold=gate_threshold),
        }
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train a raw-PPG -> peak sequence + HR interval + gate model with ECG supervision.")
    ap.add_argument("--data_root", default="physionet.org/files", type=str)
    ap.add_argument("--results_root", default=DEFAULT_RESULTS_ROOT, type=str)
    ap.add_argument("--run_name", default="auto", type=str)
    ap.add_argument("--external_holdout", default=SIM_NAME, choices=(PTT_NAME, SIM_NAME, IAM_NAME, MIMIC_NAME, VITALDB_NAME), type=str)
    ap.add_argument("--internal_train_ratio", default=0.8, type=float)
    ap.add_argument("--cv_folds", default=5, type=int)
    ap.add_argument("--target_fs", default=256.0, type=float)
    ap.add_argument("--win_sec", default=8.0, type=float)
    ap.add_argument("--hop_sec", default=2.0, type=float)
    ap.add_argument("--epochs", default=12, type=int)
    ap.add_argument("--batch_size", default=16, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--patience", default=4, type=int)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--base_channels", default=32, type=int)
    ap.add_argument("--norm_type", default="instance", choices=("instance", "group"), type=str)
    ap.add_argument("--ibi_min_sec", default=0.30, type=float)
    ap.add_argument("--ibi_max_sec", default=2.00, type=float)
    ap.add_argument("--ibi_huber_delta", default=0.08, type=float)
    ap.add_argument("--lam_peak", default=1.0, type=float)
    ap.add_argument("--lam_beat", default=0.10, type=float)
    ap.add_argument("--lam_ibi", default=0.35, type=float)
    ap.add_argument("--lam_gate", default=0.25, type=float)
    ap.add_argument("--lam_domain", default=0.0, type=float)
    ap.add_argument("--worst_domain_weight", default=0.25, type=float)
    ap.add_argument("--domain_adv_lambda", default=1.0, type=float)
    ap.add_argument("--balanced_sampling", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--aug_noise_std", default=0.025, type=float)
    ap.add_argument("--aug_drift_std", default=0.050, type=float)
    ap.add_argument("--aug_dropout_prob", default=0.15, type=float)
    ap.add_argument("--aug_respiration_mod_prob", default=0.35, type=float)
    ap.add_argument("--aug_motion_burst_prob", default=0.25, type=float)
    ap.add_argument("--aug_clip_prob", default=0.12, type=float)
    ap.add_argument("--aug_lowpass_prob", default=0.20, type=float)
    ap.add_argument("--aug_polarity_flip_prob", default=0.05, type=float)
    ap.add_argument("--aug_time_warp_prob", default=0.15, type=float)
    ap.add_argument("--aug_target_jitter_sec", default=0.020, type=float)
    ap.add_argument("--group_scorecards", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--delay_analysis", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--lodo_validation", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--lodo_max_datasets", default=0, type=int)
    ap.add_argument("--detector_benchmark", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--detector_epochs", default=8, type=int)
    ap.add_argument("--detector_patience", default=3, type=int)
    ap.add_argument("--detector_batch_size", default=0, type=int)
    ap.add_argument("--detector_lr", default=1e-3, type=float)
    ap.add_argument("--detector_base_channels", default=24, type=int)
    ap.add_argument("--mimic_subsets", default=",".join(MIMIC_DEFAULT_SUBSETS), type=str)
    ap.add_argument("--mimic_max_records", default=0, type=int)
    ap.add_argument("--include_mimic_special_extra_holdout", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mimic_extra_holdout_subsets", default=",".join(MIMIC_EXTRA_HOLDOUT_SUBSETS), type=str)
    ap.add_argument("--mimic_extra_holdout_mat_files", default=",".join(MIMIC_EXTRA_HOLDOUT_MAT_FILES), type=str)
    ap.add_argument("--mimic_extra_holdout_max_records", default=0, type=int)
    ap.add_argument("--enable_vitaldb", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--vitaldb_max_cases", default=20, type=int)
    ap.add_argument("--vitaldb_maxlen_sec", default=600.0, type=float)
    ap.add_argument("--ecg_detector_preflight", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ecg_preflight_min_peak_f1", default=0.95, type=float)
    ap.add_argument("--ecg_preflight_min_ibi_accuracy", default=0.95, type=float)
    ap.add_argument("--ecg_preflight_tolerance_sec", default=0.004, type=float)
    ap.add_argument("--device", default="cpu", type=str)
    ap.add_argument("--inspect_only", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_root = Path(args.data_root)
    results_root = ensure_dir(Path(args.results_root))
    run_dir, resolved_run_name = resolve_timestamped_run_dir(results_root, args.run_name)
    cv_dir = ensure_dir(run_dir / "cross_validation")
    holdout_dir = ensure_dir(run_dir / "holdout")
    extra_dir = ensure_dir(run_dir / "extra_holdout")

    inspect_summary = inspect_new_datasets(data_root)
    print_dataset_inspection(inspect_summary)
    if args.inspect_only:
        return

    preflight_result: Optional[Dict[str, Any]] = None
    if args.ecg_detector_preflight:
        preflight_result = run_ptt_ecg_detector_preflight(
            data_root=data_root,
            outdir=run_dir,
            min_peak_f1=args.ecg_preflight_min_peak_f1,
            min_ibi_accuracy=args.ecg_preflight_min_ibi_accuracy,
            tolerance_sec=args.ecg_preflight_tolerance_sec,
        )

    mimic_subsets = tuple(x.strip() for x in str(args.mimic_subsets).split(",") if x.strip())
    mimic_extra_holdout_subsets = tuple(x.strip() for x in str(args.mimic_extra_holdout_subsets).split(",") if x.strip())
    mimic_extra_holdout_mat_files = tuple(x.strip() for x in str(args.mimic_extra_holdout_mat_files).split(",") if x.strip())
    dataset_records = load_all_datasets(
        data_root,
        target_fs=args.target_fs,
        mimic_subsets=mimic_subsets,
        mimic_max_records=args.mimic_max_records,
        enable_vitaldb=bool(args.enable_vitaldb),
        vitaldb_max_cases=args.vitaldb_max_cases,
        vitaldb_maxlen_sec=args.vitaldb_maxlen_sec,
    )
    mimic_special_extra_records = (
        load_mimic_special_extra_holdout(
            data_root,
            target_fs=args.target_fs,
            subsets=mimic_extra_holdout_subsets,
            mat_files=mimic_extra_holdout_mat_files,
            max_records=args.mimic_extra_holdout_max_records,
        )
        if args.include_mimic_special_extra_holdout
        else []
    )
    loaded_summary = build_loaded_dataset_summary(dataset_records, inspect_summary)
    loaded_summary["special_extra_holdout"] = {
        "mimic_special_record_count": int(len(mimic_special_extra_records)),
        "mimic_special_subject_count": int(len({rec["subject"] for rec in mimic_special_extra_records})),
        "mimic_extra_holdout_subsets": list(mimic_extra_holdout_subsets),
        "mimic_extra_holdout_mat_files": list(mimic_extra_holdout_mat_files),
    }
    usable = {name: recs for name, recs in dataset_records.items() if len(recs) > 0}
    if args.external_holdout not in usable:
        raise RuntimeError(f"External holdout dataset {args.external_holdout} is not usable. Usable datasets: {sorted(usable)}")

    external_extra_records = list(usable[args.external_holdout])
    extra_holdout_records = external_extra_records + list(mimic_special_extra_records)
    modeling_pool = [rec for name, recs in usable.items() if name != args.external_holdout for rec in recs]
    train_subjects, holdout_subjects = split_subjects(
        [rec["subject"] for rec in modeling_pool],
        train_ratio=args.internal_train_ratio,
        seed=args.seed,
    )
    train_records = [rec for rec in modeling_pool if rec["subject"] in set(train_subjects)]
    holdout_records = [rec for rec in modeling_pool if rec["subject"] in set(holdout_subjects)]

    window_cfg = WindowConfig(fs=args.target_fs, win_sec=args.win_sec, hop_sec=args.hop_sec)
    model_cfg = ModelConfig(
        in_channels=1,
        base_channels=args.base_channels,
        norm_type=args.norm_type,
        ibi_min_sec=args.ibi_min_sec,
        ibi_max_sec=args.ibi_max_sec,
        num_domains=len(DOMAIN_TO_IDX),
    )
    gate_input_audit = build_gate_input_audit(
        model_cfg=model_cfg,
        inspect_summary=inspect_summary,
        train_records=train_records,
        holdout_records=holdout_records,
        extra_holdout_records=extra_holdout_records,
    )
    loss_cfg = LossConfig(
        peak=args.lam_peak,
        beat=args.lam_beat,
        ibi=args.lam_ibi,
        gate=args.lam_gate,
        domain=args.lam_domain,
        worst_domain=args.worst_domain_weight,
        fs=args.target_fs,
        ibi_huber_delta=args.ibi_huber_delta,
    )
    augment_cfg = AugmentConfig(
        enabled=bool(args.augment),
        noise_std=args.aug_noise_std,
        drift_std=args.aug_drift_std,
        dropout_prob=args.aug_dropout_prob,
        respiration_mod_prob=args.aug_respiration_mod_prob,
        motion_burst_prob=args.aug_motion_burst_prob,
        clip_prob=args.aug_clip_prob,
        lowpass_prob=args.aug_lowpass_prob,
        polarity_flip_prob=args.aug_polarity_flip_prob,
        time_warp_prob=args.aug_time_warp_prob,
        target_jitter_sec=args.aug_target_jitter_sec,
    )
    detector_cfg = DetectorConfig(
        in_channels=10,
        base_channels=int(args.detector_base_channels),
        fs=float(args.target_fs),
        win_sec=float(args.win_sec),
        hop_sec=float(args.hop_sec),
    )

    delay_analysis: Dict[str, Any] = {}
    if args.delay_analysis:
        delay_analysis = analyze_ppg_ecg_delay(
            records=list(modeling_pool) + list(extra_holdout_records),
            fs=float(args.target_fs),
        )
        save_json(run_dir / "ppg_ecg_delay_analysis.json", delay_analysis)
        save_delay_analysis_plots(delay_analysis, run_dir)

    cv_info = run_cross_validation(
        train_records=train_records,
        window_cfg=window_cfg,
        model_cfg=model_cfg,
        loss_cfg=loss_cfg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        cv_folds=args.cv_folds,
        device=args.device,
        seed=args.seed,
        augment_cfg=augment_cfg,
        balanced_sampling=bool(args.balanced_sampling),
        domain_lambda=args.domain_adv_lambda,
    )
    save_cv_training_curves(cv_info["fold_histories"], cv_dir)
    save_binary_eval_plots(
        cv_info["aggregate_outputs"]["peak_true"],
        cv_info["aggregate_outputs"]["peak_prob"],
        cv_info["thresholds"]["peak_threshold"],
        cv_dir,
        prefix="peak_sequence",
        title_prefix="Cross-Validation Peak Sequence",
    )
    save_regression_plots(
        cv_info["aggregate_outputs"]["rr_true"],
        cv_info["aggregate_outputs"]["rr_pred"],
        cv_dir,
        prefix="hr_interval_sequence",
        title_prefix="Cross-Validation HR Interval",
    )
    save_binary_eval_plots(
        cv_info["aggregate_outputs"]["gate_true"],
        cv_info["aggregate_outputs"]["gate_prob"],
        cv_info["thresholds"]["gate_threshold"],
        cv_dir,
        prefix="gate_logit",
        title_prefix="Cross-Validation Gate Logit",
    )
    save_event_error_plots(
        cv_info["aggregate_outputs"],
        cv_info["thresholds"]["peak_threshold"],
        cv_dir,
        prefix="event",
        title_prefix="Cross-Validation",
    )

    lodo_summary: Dict[str, Any] = {}
    if args.lodo_validation:
        lodo_summary = run_leave_one_dataset_out(
            records=modeling_pool,
            window_cfg=window_cfg,
            model_cfg=model_cfg,
            loss_cfg=loss_cfg,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            device=args.device,
            seed=args.seed,
            augment_cfg=augment_cfg,
            balanced_sampling=bool(args.balanced_sampling),
            domain_lambda=args.domain_adv_lambda,
            max_datasets=args.lodo_max_datasets,
        )
        save_json(run_dir / "leave_one_dataset_out_summary.json", lodo_summary)

    final_model, final_train_info = fit_model_fixed_epochs(
        train_records=train_records,
        window_cfg=window_cfg,
        model_cfg=model_cfg,
        loss_cfg=loss_cfg,
        epochs=cv_info["final_train_epochs"],
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        title="Final model",
        augment_cfg=augment_cfg,
        balanced_sampling=bool(args.balanced_sampling),
        domain_lambda=args.domain_adv_lambda,
    )
    save_final_training_curves(final_train_info["history"], run_dir)

    holdout_outputs = collect_outputs(
        model=final_model,
        records=holdout_records,
        window_cfg=window_cfg,
        loss_cfg=loss_cfg,
        batch_size=args.batch_size,
        device=args.device,
        desc="collect holdout",
    )
    extra_outputs = collect_outputs(
        model=final_model,
        records=extra_holdout_records,
        window_cfg=window_cfg,
        loss_cfg=loss_cfg,
        batch_size=args.batch_size,
        device=args.device,
        desc="collect extra_holdout",
    )
    holdout_summary = summarize_outputs(
        holdout_outputs,
        peak_threshold=cv_info["thresholds"]["peak_threshold"],
        gate_threshold=cv_info["thresholds"]["gate_threshold"],
    )
    extra_summary = summarize_outputs(
        extra_outputs,
        peak_threshold=cv_info["thresholds"]["peak_threshold"],
        gate_threshold=cv_info["thresholds"]["gate_threshold"],
    )
    group_scorecards: Dict[str, Any] = {}
    if args.group_scorecards:
        group_scorecards = {
            "cross_validation": build_group_scorecard_bundle(
                cv_info["aggregate_outputs"],
                peak_threshold=cv_info["thresholds"]["peak_threshold"],
                gate_threshold=cv_info["thresholds"]["gate_threshold"],
            ),
            "holdout": build_group_scorecard_bundle(
                holdout_outputs,
                peak_threshold=cv_info["thresholds"]["peak_threshold"],
                gate_threshold=cv_info["thresholds"]["gate_threshold"],
            ),
            "extra_holdout": build_group_scorecard_bundle(
                extra_outputs,
                peak_threshold=cv_info["thresholds"]["peak_threshold"],
                gate_threshold=cv_info["thresholds"]["gate_threshold"],
            ),
        }

    save_binary_eval_plots(
        holdout_outputs["peak_true"],
        holdout_outputs["peak_prob"],
        cv_info["thresholds"]["peak_threshold"],
        holdout_dir,
        prefix="peak_sequence",
        title_prefix="Holdout Peak Sequence",
    )
    save_regression_plots(
        holdout_outputs["rr_true"],
        holdout_outputs["rr_pred"],
        holdout_dir,
        prefix="hr_interval_sequence",
        title_prefix="Holdout HR Interval",
    )
    save_binary_eval_plots(
        holdout_outputs["gate_true"],
        holdout_outputs["gate_prob"],
        cv_info["thresholds"]["gate_threshold"],
        holdout_dir,
        prefix="gate_logit",
        title_prefix="Holdout Gate Logit",
    )
    save_event_error_plots(
        holdout_outputs,
        cv_info["thresholds"]["peak_threshold"],
        holdout_dir,
        prefix="event",
        title_prefix="Holdout",
    )
    save_binary_eval_plots(
        extra_outputs["peak_true"],
        extra_outputs["peak_prob"],
        cv_info["thresholds"]["peak_threshold"],
        extra_dir,
        prefix="peak_sequence",
        title_prefix="Extra Holdout Peak Sequence",
    )
    save_regression_plots(
        extra_outputs["rr_true"],
        extra_outputs["rr_pred"],
        extra_dir,
        prefix="hr_interval_sequence",
        title_prefix="Extra Holdout HR Interval",
    )
    save_binary_eval_plots(
        extra_outputs["gate_true"],
        extra_outputs["gate_prob"],
        cv_info["thresholds"]["gate_threshold"],
        extra_dir,
        prefix="gate_logit",
        title_prefix="Extra Holdout Gate Logit",
    )
    save_event_error_plots(
        extra_outputs,
        cv_info["thresholds"]["peak_threshold"],
        extra_dir,
        prefix="event",
        title_prefix="Extra Holdout",
    )

    detector_benchmark_summary: Dict[str, Any] = {}
    if args.detector_benchmark:
        detector_dir = ensure_dir(run_dir / "detector_benchmark")
        try:
            detector_records_by_dataset = load_motion_detector_records(data_root, target_fs=args.target_fs)
            loaded_summary["motion_detector_records"] = {
                name: {
                    "record_count": int(len(recs)),
                    "subject_count": int(len({rec["subject"] for rec in recs})),
                    "label_sources": sorted({str(rec.get("label_source", "unknown")) for rec in recs}),
                    "unit_info_examples": [rec.get("unit_info", {}) for rec in recs[:3]],
                }
                for name, recs in detector_records_by_dataset.items()
            }
            detector_benchmark_summary = run_motion_detector_benchmark(
                detector_records_by_dataset=detector_records_by_dataset,
                train_subjects=train_subjects,
                holdout_subjects=holdout_subjects,
                external_holdout_dataset=args.external_holdout,
                detector_cfg=detector_cfg,
                augment_cfg=augment_cfg,
                outdir=detector_dir,
                batch_size=int(args.detector_batch_size or args.batch_size),
                epochs=int(args.detector_epochs),
                patience=int(args.detector_patience),
                lr=float(args.detector_lr),
                device=args.device,
                balanced_sampling=bool(args.balanced_sampling),
            )
        except Exception as exc:
            detector_benchmark_summary = {
                "status": "failed",
                "reason": str(exc),
                "models": {},
                "detector_input": {
                    "uses_imu": True,
                    "channels": [
                        "ppg",
                        "acc_dyn_x",
                        "acc_dyn_y",
                        "acc_dyn_z",
                        "gyro_x",
                        "gyro_y",
                        "gyro_z",
                        "acc_mag",
                        "gyro_mag",
                        "jerk_mag",
                    ],
                    "unit_handling": "acceleration inferred per record and gravity removed; missing gyro zero-filled",
                },
            }
            save_json(detector_dir / "detector_benchmark_summary.json", detector_benchmark_summary)
        save_json(run_dir / "detector_benchmark_summary.json", detector_benchmark_summary)

    split_info = {
        "external_holdout_dataset": args.external_holdout,
        "train_subjects": train_subjects,
        "holdout_subjects": holdout_subjects,
        "extra_holdout_subjects": sorted({rec["subject"] for rec in extra_holdout_records}),
        "dataset_counts": {name: len(recs) for name, recs in usable.items()},
        "extra_holdout_sources": {
            "external_records": int(len(external_extra_records)),
            "mimic_special_records": int(len(mimic_special_extra_records)),
            "mimic_extra_holdout_subsets": list(mimic_extra_holdout_subsets),
            "mimic_extra_holdout_mat_files": list(mimic_extra_holdout_mat_files),
        },
        "train_record_count": len(train_records),
        "holdout_record_count": len(holdout_records),
        "extra_holdout_record_count": len(extra_holdout_records),
    }
    config_summary = {
        "results_root": str(results_root),
        "run_name": resolved_run_name,
        "cv_folds": int(args.cv_folds),
        "final_train_epochs": int(cv_info["final_train_epochs"]),
        "target_fs": float(args.target_fs),
        "win_sec": float(args.win_sec),
        "hop_sec": float(args.hop_sec),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "base_channels": int(args.base_channels),
        "norm_type": str(args.norm_type),
        "ibi_min_sec": float(args.ibi_min_sec),
        "ibi_max_sec": float(args.ibi_max_sec),
        "ibi_huber_delta": float(args.ibi_huber_delta),
        "lam_peak": float(args.lam_peak),
        "lam_beat": float(args.lam_beat),
        "lam_ibi": float(args.lam_ibi),
        "lam_gate": float(args.lam_gate),
        "lam_domain": float(args.lam_domain),
        "worst_domain_weight": float(args.worst_domain_weight),
        "domain_adversarial_lambda": float(args.domain_adv_lambda),
        "balanced_sampling": bool(args.balanced_sampling),
        "augmentation_enabled": bool(args.augment),
        "augmentation": asdict(augment_cfg),
        "group_scorecards": bool(args.group_scorecards),
        "delay_analysis": bool(args.delay_analysis),
        "lodo_validation": bool(args.lodo_validation),
        "lodo_max_datasets": int(args.lodo_max_datasets),
        "detector_benchmark": bool(args.detector_benchmark),
        "detector_epochs": int(args.detector_epochs),
        "detector_patience": int(args.detector_patience),
        "detector_batch_size": int(args.detector_batch_size or args.batch_size),
        "detector_lr": float(args.detector_lr),
        "detector_base_channels": int(args.detector_base_channels),
        "detector_cfg": asdict(detector_cfg),
        "detector_benchmark_status": detector_benchmark_summary.get("status", "disabled") if detector_benchmark_summary else "disabled",
        "mimic_subsets": list(mimic_subsets),
        "mimic_max_records": int(args.mimic_max_records),
        "include_mimic_special_extra_holdout": bool(args.include_mimic_special_extra_holdout),
        "mimic_extra_holdout_subsets": list(mimic_extra_holdout_subsets),
        "mimic_extra_holdout_mat_files": list(mimic_extra_holdout_mat_files),
        "mimic_extra_holdout_max_records": int(args.mimic_extra_holdout_max_records),
        "enable_vitaldb": bool(args.enable_vitaldb),
        "vitaldb_max_cases": int(args.vitaldb_max_cases),
        "vitaldb_maxlen_sec": float(args.vitaldb_maxlen_sec),
        "ecg_detector_preflight": bool(args.ecg_detector_preflight),
        "ecg_preflight_min_peak_f1": float(args.ecg_preflight_min_peak_f1),
        "ecg_preflight_min_ibi_accuracy": float(args.ecg_preflight_min_ibi_accuracy),
        "ecg_preflight_tolerance_sec": float(args.ecg_preflight_tolerance_sec),
        "ecg_preflight_result": preflight_result,
        "gate_input_audit": gate_input_audit,
        "ppg_main_event_tolerance_sec": float(PPG_MAIN_EVENT_TOLERANCE_SEC),
        "ppg_layered_event_tolerance_secs": [float(x) for x in PPG_EVENT_TOLERANCE_SECS],
    }

    deploy_export = export_deploy_bundle(
        model=final_model,
        run_dir=run_dir,
        model_cfg=model_cfg,
        window_cfg=window_cfg,
        thresholds=cv_info["thresholds"],
        split_info=split_info,
    )

    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in final_model.state_dict().items()},
            "model_cfg": asdict(model_cfg),
            "window_cfg": asdict(window_cfg),
            "loss_cfg": asdict(loss_cfg),
            "config_summary": config_summary,
            "split_info": split_info,
            "thresholds": cv_info["thresholds"],
            "cv_summary": cv_info["aggregate"],
            "holdout_summary": holdout_summary,
            "extra_holdout_summary": extra_summary,
            "group_scorecards": group_scorecards,
            "delay_analysis": delay_analysis,
            "lodo_summary": lodo_summary,
            "detector_benchmark_summary": detector_benchmark_summary,
            "gate_input_audit": gate_input_audit,
            "ecg_preflight_result": preflight_result,
        },
        run_dir / "peak_hr_gate_model.pt",
    )

    save_json(run_dir / "dataset_summary.json", loaded_summary)
    save_json(run_dir / "split_info.json", split_info)
    save_json(run_dir / "config_summary.json", config_summary)
    save_json(run_dir / "cv_summary.json", {"aggregate": cv_info["aggregate"], "folds": cv_info["folds"], "thresholds": cv_info["thresholds"]})
    save_json(run_dir / "holdout_summary.json", holdout_summary)
    save_json(run_dir / "extra_holdout_summary.json", extra_summary)
    save_json(run_dir / "group_scorecards.json", group_scorecards)
    save_json(run_dir / "gate_input_audit.json", gate_input_audit)
    save_json(run_dir / "detector_benchmark_summary.json", detector_benchmark_summary)
    if lodo_summary:
        save_json(run_dir / "leave_one_dataset_out_summary.json", lodo_summary)
    save_json(run_dir / "final_train_history.json", final_train_info["history"])
    save_json(run_dir / "deploy_export.json", deploy_export)
    write_scorecard_markdown(
        run_dir / "scorecard.md",
        config_summary=config_summary,
        split_info=split_info,
        thresholds=cv_info["thresholds"],
        cv_summary={"aggregate": cv_info["aggregate"], "folds": cv_info["folds"]},
        holdout_summary=holdout_summary,
        extra_summary=extra_summary,
        group_scorecards=group_scorecards,
        delay_analysis=delay_analysis,
        lodo_summary=lodo_summary,
        detector_benchmark_summary=detector_benchmark_summary,
    )

    print(f"\nSaved results to: {run_dir}")
    if deploy_export["success"]:
        print(f"Deploy bundle exported to: {run_dir / 'deploy_bundle'}")
    else:
        print(f"Deploy bundle export failed: {deploy_export['error']}")
    print(f"CV peak threshold: {cv_info['thresholds']['peak_threshold']:.3f}")
    print(f"CV gate threshold: {cv_info['thresholds']['gate_threshold']:.3f}")
    print(f"Final train epochs: {cv_info['final_train_epochs']}")
    if args.detector_benchmark:
        print(f"Detector benchmark status: {detector_benchmark_summary.get('status', 'unknown')}")


if __name__ == "__main__":
    main()
