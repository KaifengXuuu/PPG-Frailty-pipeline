from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    Dataset = object

try:
    from shapeformer_port import PortedShapeFormer, ShapeletBundle, discover_shapelets, discover_shapelets_pisd
except Exception:
    PortedShapeFormer = None
    ShapeletBundle = object
    discover_shapelets = None
    discover_shapelets_pisd = None

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


REQUIRED_COLUMNS = ("RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ")
ROLE_SUFFIXES = {"B", "R1", "R2", "R3", "R4", "S1", "S2", "W1", "W2"}
INCLUDED_ROLE_SUFFIXES = ("B", "R1", "R2", "R3", "R4")
ROLE_CACHE_TAG = "B_R1_R2_R3_R4"
ROLE_MODE_CHOICES = ("static_only", "all_roles")
CLASS_NAMES = ("pre_frail", "robust_non_frail", "young")
STATUS_TO_CLASS = {2: "pre_frail", 3: "robust_non_frail"}
CLASS_TO_LABEL = {name: idx for idx, name in enumerate(CLASS_NAMES)}
MODEL_CHOICES = (
    "logreg_l2",
    "rbf_svm",
    "extra_trees",
    "cnn",
    "cnn1d",
    "inceptiontime",
    "inception_time",
    "shapeformer_pisd",
    "shapeformer",
)
SWEEP_MODEL_CHOICES = ("cnn", "inceptiontime", "shapeformer_pisd", "shapeformer")
EXTRA_INPUT_CHOICES = ("0", "PPI", "HRV")
META_COLS = {"path", "dataset", "subject", "role", "class_name", "label"}
PPI_FEATURE_SUFFIXES = (
    "peak_count",
    "ppi_count",
    "ppi_coverage",
    "ppi_mean_ms",
    "ppi_median_ms",
    "ppi_std_ms",
    "ppi_iqr_ms",
    "hr_mean_bpm",
    "hr_std_bpm",
)
HRV_FEATURE_SUFFIXES = (
    "sdnn_ms",
    "rmssd_ms",
    "sdsd_ms",
    "nn50",
    "pnn50",
    "sd1_ms",
    "sd2_ms",
    "lf_power",
    "hf_power",
    "lf_hf",
    "total_power",
)


@dataclass
class RunConfig:
    data_root: str = "PPG_Testing_05_01_2026"
    label_csv: str = "StudyData_frailtyScored/StudyData_V7_standard.csv"
    study_dir: str = "StudyData"
    young_dir: str = "TestDataYoungers"
    fs: float = 400.0
    win_sec: float = 10.0
    hop_sec: float = 5.0
    folds: int = 5
    seed: int = 42
    cnn_target_fs: float = 64.0
    cnn_seq_sec: float = 30.0
    cnn_hop_sec: float = 30.0
    cnn_max_windows_per_file: int = 6
    cnn_max_windows_fraction: float = 0.0
    cnn_epochs: int = 8
    cnn_batch_size: int = 32
    cnn_lr: float = 1e-3
    cnn_patience: int = 3
    cnn_num_workers: int = 0
    cnn_weight_decay: float = 1e-4
    cnn_dropout: float = -1.0
    cnn_label_smoothing: float = 0.0
    cnn_select_best_epoch: bool = True
    role_mode: str = "static_only"
    extra_input: str = "0"
    shapeformer_num_shapelets: int = 3
    shapeformer_shapelet_len: int = 128
    shapeformer_shapelet_stride: int = 64
    shapeformer_local_window: int = 64
    shapeformer_local_embed_dim: int = 48
    shapeformer_shape_embed_dim: int = 128
    shapeformer_dim_ff: int = 256
    shapeformer_heads: int = 4
    shapeformer_dropout: float = 0.30
    shapeformer_discovery_method: str = "effect_size"
    shapeformer_discovery_windows: int = 180
    shapeformer_candidates_per_class_channel: int = 8
    shapeformer_num_pip: float = 0.2
    shapeformer_processes: int = 1
    shapeformer_pisd_verbose: bool = False


def finite_float(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else 0.0


def read_numeric_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=list(REQUIRED_COLUMNS))
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def interp_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return x
    mask = np.isfinite(x)
    if mask.all():
        return x
    if not mask.any():
        return np.zeros_like(x)
    idx = np.arange(x.size)
    return np.interp(idx, idx[mask], x[mask])


def bandpass_ppg(x: np.ndarray, fs: float) -> np.ndarray:
    x = interp_nan(x)
    if x.size < 16:
        return x - np.nanmedian(x) if x.size else x
    y = signal.detrend(x, type="linear")
    nyq = 0.5 * fs
    high = min(8.0, nyq * 0.9)
    low = min(0.2, high * 0.5)
    sos = signal.butter(3, [low, high], btype="bandpass", fs=fs, output="sos")
    try:
        return signal.sosfiltfilt(sos, y)
    except ValueError:
        return signal.sosfilt(sos, y)


def lowpass_imu(x: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 16:
        return np.nan_to_num(x, nan=0.0)
    x = np.vstack([interp_nan(x[:, i]) for i in range(x.shape[1])]).T
    cutoff = min(cutoff, 0.45 * fs)
    sos = signal.butter(3, cutoff, btype="lowpass", fs=fs, output="sos")
    try:
        return signal.sosfiltfilt(sos, x, axis=0)
    except ValueError:
        return signal.sosfilt(sos, x, axis=0)


def time_features(x: np.ndarray, prefix: str) -> Dict[str, float]:
    x = interp_nan(x)
    if x.size == 0:
        return {f"{prefix}_{name}": 0.0 for name in ("mean", "std", "rms", "iqr", "mad", "skew", "kurt")}
    med = float(np.median(x))
    centered = x - med
    q75, q25 = np.percentile(x, [75, 25])
    return {
        f"{prefix}_mean": finite_float(np.mean(x)),
        f"{prefix}_std": finite_float(np.std(x)),
        f"{prefix}_rms": finite_float(np.sqrt(np.mean(np.square(x)))),
        f"{prefix}_iqr": finite_float(q75 - q25),
        f"{prefix}_mad": finite_float(np.median(np.abs(centered))),
        f"{prefix}_skew": finite_float(stats.skew(x, bias=False)) if x.size > 2 else 0.0,
        f"{prefix}_kurt": finite_float(stats.kurtosis(x, fisher=False, bias=False)) if x.size > 3 else 0.0,
    }


def spectral_features(x: np.ndarray, fs: float, prefix: str) -> Dict[str, float]:
    x = interp_nan(x)
    if x.size < 8 or np.allclose(np.std(x), 0.0):
        keys = ["total", "entropy", "peak_hz", "centroid_hz", "bp_0p1_0p5", "bp_0p5_3", "bp_3_8", "bp_8_20"]
        return {f"{prefix}_{key}": 0.0 for key in keys}

    nperseg = int(min(x.size, max(64, min(2048, round(4 * fs)))))
    freqs, psd = signal.welch(x, fs=fs, window="hann", nperseg=nperseg)
    psd = np.maximum(psd, 0.0)
    total = finite_float(np.trapezoid(psd, freqs))
    p = psd + 1e-18
    p = p / np.sum(p)

    def bp(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return finite_float(np.trapezoid(psd[mask], freqs[mask])) if np.any(mask) else 0.0

    hr_mask = (freqs >= 0.5) & (freqs <= 3.0)
    peak_hz = float(freqs[hr_mask][np.argmax(psd[hr_mask])]) if np.any(hr_mask) else 0.0
    centroid = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-18))
    return {
        f"{prefix}_total": total,
        f"{prefix}_entropy": finite_float(-np.sum(p * np.log(p))),
        f"{prefix}_peak_hz": finite_float(peak_hz),
        f"{prefix}_centroid_hz": finite_float(centroid),
        f"{prefix}_bp_0p1_0p5": bp(0.1, 0.5),
        f"{prefix}_bp_0p5_3": bp(0.5, 3.0),
        f"{prefix}_bp_3_8": bp(3.0, 8.0),
        f"{prefix}_bp_8_20": bp(8.0, 20.0),
    }


def empty_ppi_hrv_features(prefix: str) -> Dict[str, float]:
    keys = (
        "peak_count",
        "ppi_count",
        "ppi_coverage",
        "ppi_mean_ms",
        "ppi_median_ms",
        "ppi_std_ms",
        "ppi_iqr_ms",
        "hr_mean_bpm",
        "hr_std_bpm",
        "sdnn_ms",
        "rmssd_ms",
        "sdsd_ms",
        "nn50",
        "pnn50",
        "sd1_ms",
        "sd2_ms",
        "lf_power",
        "hf_power",
        "lf_hf",
        "total_power",
    )
    return {f"{prefix}_{key}": 0.0 for key in keys}


def clean_pp_intervals(peaks: np.ndarray, fs: float, min_bpm: float = 35.0, max_bpm: float = 210.0) -> np.ndarray:
    peaks = np.asarray(peaks, dtype=np.int64)
    if peaks.size < 2:
        return np.empty(0, dtype=np.float64)
    ppi = np.diff(peaks).astype(np.float64) / float(fs)
    lo = 60.0 / float(max_bpm)
    hi = 60.0 / float(min_bpm)
    ppi = ppi[(ppi >= lo) & (ppi <= hi)]
    if ppi.size >= 5:
        med = float(np.median(ppi))
        mad = float(np.median(np.abs(ppi - med)))
        if mad > 1e-6:
            robust_sigma = 1.4826 * mad
            ppi = ppi[np.abs(ppi - med) <= 4.0 * robust_sigma]
    return ppi


def detect_ppg_peaks(ppg: np.ndarray, fs: float, min_bpm: float = 35.0, max_bpm: float = 210.0) -> np.ndarray:
    x = interp_nan(ppg)
    if x.size < int(3 * fs):
        return np.empty(0, dtype=np.int64)
    scale = float(np.std(x))
    if scale < 1e-8:
        return np.empty(0, dtype=np.int64)
    z = (x - float(np.median(x))) / (scale + 1e-8)
    distance = max(1, int(round(fs * 60.0 / max_bpm)))
    candidates: List[Tuple[float, np.ndarray]] = []
    for polarity in (1.0, -1.0):
        peaks, _props = signal.find_peaks(polarity * z, distance=distance, prominence=0.30)
        ppi = clean_pp_intervals(peaks, fs, min_bpm=min_bpm, max_bpm=max_bpm)
        if ppi.size == 0:
            score = float(peaks.size) * 0.01
        else:
            cv = float(np.std(ppi) / (np.mean(ppi) + 1e-8))
            score = float(ppi.size) - min(cv, 1.0)
        candidates.append((score, peaks.astype(np.int64)))
    return max(candidates, key=lambda item: item[0])[1]


def ppi_hrv_features(ppg: np.ndarray, fs: float, prefix: str, duration_sec: float) -> Dict[str, float]:
    feats = empty_ppi_hrv_features(prefix)
    peaks = detect_ppg_peaks(ppg, fs)
    ppi_sec = clean_pp_intervals(peaks, fs)
    feats[f"{prefix}_peak_count"] = float(peaks.size)
    feats[f"{prefix}_ppi_count"] = float(ppi_sec.size)
    feats[f"{prefix}_ppi_coverage"] = finite_float(np.sum(ppi_sec) / max(duration_sec, 1e-8))
    if ppi_sec.size < 2:
        return feats

    ppi_ms = ppi_sec * 1000.0
    hr_bpm = 60.0 / ppi_sec
    diff_ms = np.diff(ppi_ms)
    feats.update(
        {
            f"{prefix}_ppi_mean_ms": finite_float(np.mean(ppi_ms)),
            f"{prefix}_ppi_median_ms": finite_float(np.median(ppi_ms)),
            f"{prefix}_ppi_std_ms": finite_float(np.std(ppi_ms)),
            f"{prefix}_ppi_iqr_ms": finite_float(np.percentile(ppi_ms, 75) - np.percentile(ppi_ms, 25)),
            f"{prefix}_hr_mean_bpm": finite_float(np.mean(hr_bpm)),
            f"{prefix}_hr_std_bpm": finite_float(np.std(hr_bpm)),
            f"{prefix}_sdnn_ms": finite_float(np.std(ppi_ms, ddof=1)) if ppi_ms.size > 1 else 0.0,
            f"{prefix}_rmssd_ms": finite_float(np.sqrt(np.mean(np.square(diff_ms)))) if diff_ms.size else 0.0,
            f"{prefix}_sdsd_ms": finite_float(np.std(diff_ms, ddof=1)) if diff_ms.size > 1 else 0.0,
            f"{prefix}_nn50": finite_float(np.sum(np.abs(diff_ms) > 50.0)) if diff_ms.size else 0.0,
            f"{prefix}_pnn50": finite_float(np.mean(np.abs(diff_ms) > 50.0)) if diff_ms.size else 0.0,
        }
    )
    feats[f"{prefix}_sd1_ms"] = finite_float(np.sqrt(0.5) * feats[f"{prefix}_sdsd_ms"])
    sdnn_sq = feats[f"{prefix}_sdnn_ms"] ** 2
    sd1_sq = feats[f"{prefix}_sd1_ms"] ** 2
    feats[f"{prefix}_sd2_ms"] = finite_float(np.sqrt(max(0.0, 2.0 * sdnn_sq - sd1_sq)))

    if ppi_sec.size >= 10 and np.sum(ppi_sec) >= 60.0:
        beat_t = np.cumsum(ppi_sec)
        interp_fs = 4.0
        grid = np.arange(float(beat_t[0]), float(beat_t[-1]), 1.0 / interp_fs)
        if grid.size >= 16:
            tach = np.interp(grid, beat_t, ppi_ms)
            tach = signal.detrend(tach)
            nperseg = min(256, len(tach))
            freqs, psd = signal.welch(tach, fs=interp_fs, window="hann", nperseg=nperseg)

            def band(lo: float, hi: float) -> float:
                mask = (freqs >= lo) & (freqs < hi)
                return finite_float(np.trapezoid(psd[mask], freqs[mask])) if np.any(mask) else 0.0

            lf = band(0.04, 0.15)
            hf = band(0.15, 0.40)
            total = band(0.003, 0.40)
            feats[f"{prefix}_lf_power"] = lf
            feats[f"{prefix}_hf_power"] = hf
            feats[f"{prefix}_lf_hf"] = finite_float(lf / (hf + 1e-8))
            feats[f"{prefix}_total_power"] = total
    return feats


def iter_windows(n: int, fs: float, win_sec: float, hop_sec: float) -> Iterable[Tuple[int, int]]:
    win = max(1, int(round(win_sec * fs)))
    hop = max(1, int(round(hop_sec * fs)))
    if n <= win:
        yield 0, n
        return
    for start in range(0, n - win + 1, hop):
        yield start, start + win


def per_window_features(
    red: np.ndarray,
    ir: np.ndarray,
    acc: np.ndarray,
    gyro: np.ndarray,
    fs: float,
) -> Dict[str, float]:
    feats: Dict[str, float] = {}

    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    jerk_mag = np.linalg.norm(np.diff(acc, axis=0, prepend=acc[:1]) * fs, axis=1)

    for name, sig in (
        ("red", red),
        ("ir", ir),
        ("acc_mag", acc_mag),
        ("gyro_mag", gyro_mag),
        ("jerk_mag", jerk_mag),
    ):
        feats.update(time_features(sig, name))
        feats.update(spectral_features(sig, fs, f"{name}_spec"))

    for idx, axis in enumerate(("ax", "ay", "az")):
        feats.update(time_features(acc[:, idx], axis))
    for idx, axis in enumerate(("gx", "gy", "gz")):
        feats.update(time_features(gyro[:, idx], axis))

    red_std = float(np.std(red))
    ir_std = float(np.std(ir))
    feats["red_ir_corr"] = finite_float(np.corrcoef(red, ir)[0, 1]) if red.size > 2 and red_std > 0 and ir_std > 0 else 0.0
    feats["red_ir_std_ratio"] = finite_float(red_std / (ir_std + 1e-8))
    feats["red_ir_iqr_ratio"] = finite_float(
        (np.percentile(red, 75) - np.percentile(red, 25))
        / ((np.percentile(ir, 75) - np.percentile(ir, 25)) + 1e-8)
    )
    return feats


def extract_file_features(path: Path, fs: float, win_sec: float, hop_sec: float) -> Dict[str, float]:
    df = read_numeric_csv(path)
    red = bandpass_ppg(df["RED"].to_numpy(), fs)
    ir = bandpass_ppg(df["IR"].to_numpy(), fs)
    acc = lowpass_imu(df[["AX", "AY", "AZ"]].to_numpy(), fs, cutoff=20.0)
    gyro = lowpass_imu(df[["GX", "GY", "GZ"]].to_numpy(), fs, cutoff=40.0)

    rows = []
    for start, stop in iter_windows(len(df), fs, win_sec, hop_sec):
        rows.append(per_window_features(red[start:stop], ir[start:stop], acc[start:stop], gyro[start:stop], fs))

    feat_df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out: Dict[str, float] = {
        "n_rows": float(len(df)),
        "duration_sec": float(len(df) / fs),
        "n_windows": float(len(feat_df)),
    }
    out.update(ppi_hrv_features(red, fs, "red_ppg", duration_sec=out["duration_sec"]))
    out.update(ppi_hrv_features(ir, fs, "ir_ppg", duration_sec=out["duration_sec"]))
    for col in feat_df.columns:
        vals = feat_df[col].to_numpy(dtype=float)
        out[f"{col}__mean"] = finite_float(np.mean(vals))
        out[f"{col}__std"] = finite_float(np.std(vals))
    return out


def extract_file_ppi_hrv_only(path: Path, fs: float) -> Dict[str, float]:
    df = read_numeric_csv(path)
    red = bandpass_ppg(df["RED"].to_numpy(), fs)
    ir = bandpass_ppg(df["IR"].to_numpy(), fs)
    duration_sec = float(len(df) / fs)
    out: Dict[str, float] = {
        "n_rows": float(len(df)),
        "duration_sec": duration_sec,
    }
    out.update(ppi_hrv_features(red, fs, "red_ppg", duration_sec=duration_sec))
    out.update(ppi_hrv_features(ir, fs, "ir_ppg", duration_sec=duration_sec))
    return out


def study_subject_id(path: Path) -> str:
    return path.stem.split("_")[0]


def young_subject_id(path: Path) -> str:
    parts = path.stem.split("_")
    if parts[-1] in ROLE_SUFFIXES and len(parts) >= 3:
        return "_".join(parts[:-1])
    return parts[0]


def file_role(path: Path) -> str:
    suffix = path.stem.split("_")[-1]
    return suffix if suffix in ROLE_SUFFIXES else "UNK"


def normalize_role_mode(value: str) -> str:
    text = str(value).strip().lower().replace("-", "_")
    aliases = {
        "static": "static_only",
        "static_only": "static_only",
        "included": "static_only",
        "included_roles": "static_only",
        "all": "all_roles",
        "all_roles": "all_roles",
        "dynamic": "all_roles",
        "with_dynamic": "all_roles",
    }
    if text not in aliases:
        raise ValueError(f"Unknown role mode: {value}. Use one of {ROLE_MODE_CHOICES}.")
    return aliases[text]


def role_suffixes_for_config(config: RunConfig) -> Optional[Tuple[str, ...]]:
    mode = normalize_role_mode(getattr(config, "role_mode", "static_only"))
    if mode == "static_only":
        return INCLUDED_ROLE_SUFFIXES
    if mode == "all_roles":
        return None
    raise ValueError(f"Unknown role mode: {mode}")


def role_cache_tag(config: RunConfig) -> str:
    mode = normalize_role_mode(getattr(config, "role_mode", "static_only"))
    if mode == "static_only":
        return ROLE_CACHE_TAG
    return "all_roles"


def load_label_map(label_path: Path) -> Dict[str, str]:
    labels = pd.read_csv(label_path, encoding="utf-8-sig")
    labels = labels.dropna(subset=["ID", "FRAILTY-STATUS"])
    out: Dict[str, str] = {}
    for _, row in labels.iterrows():
        try:
            status = int(row["FRAILTY-STATUS"])
        except (TypeError, ValueError):
            continue
        if status in STATUS_TO_CLASS:
            out[str(row["ID"]).strip()] = STATUS_TO_CLASS[status]
    return out


def build_manifest(config: RunConfig) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    root = Path(config.data_root)
    label_map = load_label_map(root / config.label_csv)
    rows: List[Dict[str, object]] = []
    skipped: Dict[str, List[str]] = {"study_unmatched": [], "excluded_role": [], "bad_columns": []}
    allowed_roles = role_suffixes_for_config(config)

    for path in sorted((root / config.study_dir).glob("*.csv")):
        role = file_role(path)
        if allowed_roles is not None and role not in allowed_roles:
            skipped["excluded_role"].append(path.name)
            continue
        subject = study_subject_id(path)
        class_name = label_map.get(subject)
        if class_name is None:
            skipped["study_unmatched"].append(path.name)
            continue
        rows.append(
            {
                "path": str(path),
                "dataset": "study",
                "subject": subject,
                "role": role,
                "class_name": class_name,
                "label": CLASS_TO_LABEL[class_name],
            }
        )

    for path in sorted((root / config.young_dir).glob("*.csv")):
        role = file_role(path)
        if allowed_roles is not None and role not in allowed_roles:
            skipped["excluded_role"].append(path.name)
            continue
        rows.append(
            {
                "path": str(path),
                "dataset": "young",
                "subject": young_subject_id(path),
                "role": role,
                "class_name": "young",
                "label": CLASS_TO_LABEL["young"],
            }
        )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("No usable files found.")
    return manifest, skipped


def feature_cache_name(config: RunConfig) -> Path:
    root = Path(config.data_root)
    out_dir = Path("datasets")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"frailty3_features_ppi_hrv_{role_cache_tag(config)}_fs{int(config.fs)}_w{config.win_sec:g}_h{config.hop_sec:g}.csv"


def ppi_hrv_cache_name(config: RunConfig) -> Path:
    out_dir = Path("datasets")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"frailty3_ppi_hrv_{role_cache_tag(config)}_fs{int(config.fs)}.csv"


def build_feature_table(config: RunConfig, refresh: bool = False) -> Tuple[pd.DataFrame, Dict[str, List[str]], Path]:
    cache_path = feature_cache_name(config)
    manifest, skipped = build_manifest(config)
    if cache_path.exists() and not refresh:
        features = pd.read_csv(cache_path)
        return features, skipped, cache_path

    rows: List[Dict[str, object]] = []
    for idx, row in manifest.iterrows():
        path = Path(str(row["path"]))
        print(f"[{idx + 1:03d}/{len(manifest):03d}] extracting {path.name}", flush=True)
        try:
            feats = extract_file_features(path, fs=config.fs, win_sec=config.win_sec, hop_sec=config.hop_sec)
        except Exception as exc:
            skipped["bad_columns"].append(f"{path.name}: {exc}")
            continue
        rows.append({**row.to_dict(), **feats})

    features = pd.DataFrame(rows)
    features.to_csv(cache_path, index=False)
    return features, skipped, cache_path


def build_ppi_hrv_feature_table(config: RunConfig, refresh: bool = False) -> Tuple[pd.DataFrame, Dict[str, List[str]], Path]:
    cache_path = ppi_hrv_cache_name(config)
    manifest, skipped = build_manifest(config)
    if cache_path.exists() and not refresh:
        features = pd.read_csv(cache_path)
        return features, skipped, cache_path

    rows: List[Dict[str, object]] = []
    for idx, row in manifest.iterrows():
        path = Path(str(row["path"]))
        print(f"[ppi/hrv {idx + 1:03d}/{len(manifest):03d}] extracting {path.name}", flush=True)
        try:
            feats = extract_file_ppi_hrv_only(path, fs=config.fs)
        except Exception as exc:
            skipped["bad_columns"].append(f"{path.name}: {exc}")
            continue
        rows.append({**row.to_dict(), **feats})

    features = pd.DataFrame(rows)
    features.to_csv(cache_path, index=False)
    return features, skipped, cache_path


def normalize_extra_input(value: str) -> str:
    value = str(value).strip()
    if value in {"", "none", "None", "no", "0"}:
        return "0"
    upper = value.upper()
    if upper in {"PPI", "HRV"}:
        return upper
    raise ValueError(f"Unknown extra input mode: {value}. Use one of {EXTRA_INPUT_CHOICES}.")


def select_extra_feature_columns(features: pd.DataFrame, extra_input: str) -> List[str]:
    mode = normalize_extra_input(extra_input)
    if mode == "0":
        return []
    suffixes = PPI_FEATURE_SUFFIXES if mode == "PPI" else PPI_FEATURE_SUFFIXES + HRV_FEATURE_SUFFIXES
    cols: List[str] = []
    for col in features.columns:
        if col in META_COLS:
            continue
        if any(col.endswith(f"_{suffix}") for suffix in suffixes):
            cols.append(col)
    if not cols:
        raise RuntimeError(f"No {mode} feature columns were found in the feature table.")
    return cols


def scaled_file_features_for_fold(
    features: pd.DataFrame,
    cols: Sequence[str],
    train_file_idx: np.ndarray,
) -> np.ndarray:
    raw = features[list(cols)].to_numpy(dtype=np.float32)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_imputed = imputer.fit_transform(raw[train_file_idx])
    scaler.fit(train_imputed)
    return scaler.transform(imputer.transform(raw)).astype(np.float32)


def set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    if torch is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))


def ensure_torch() -> None:
    if torch is None or nn is None or DataLoader is None:
        raise RuntimeError("The cnn1d model requires PyTorch, but torch could not be imported.")


if nn is not None:

    class Cnn1DClassifier(nn.Module):
        def __init__(self, n_channels: int, n_classes: int, dropout: float = -1.0) -> None:
            super().__init__()
            if float(dropout) >= 0.0:
                d1 = d2 = d3 = float(dropout)
            else:
                d1, d2, d3 = 0.10, 0.15, 0.20
            self.encoder = nn.Sequential(
                nn.Conv1d(n_channels, 32, kernel_size=9, padding=4),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Dropout(d1),
                nn.Conv1d(32, 64, kernel_size=9, padding=4),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Dropout(d2),
                nn.Conv1d(64, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(d3),
                nn.Linear(128, n_classes),
            )
            self.feature_dim = 128

        def forward_features(self, x: "torch.Tensor") -> "torch.Tensor":
            return torch.flatten(self.encoder(x), start_dim=1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            features = self.forward_features(x)
            return self.head[1:](features)


    class InceptionModule(nn.Module):
        def __init__(
            self,
            in_channels: int,
            n_filters: int = 32,
            kernel_sizes: Sequence[int] = (39, 19, 9),
            bottleneck_channels: int = 32,
        ) -> None:
            super().__init__()
            self.use_bottleneck = in_channels > 1 and bottleneck_channels > 0
            conv_in = bottleneck_channels if self.use_bottleneck else in_channels
            self.bottleneck = (
                nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
                if self.use_bottleneck
                else nn.Identity()
            )
            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(conv_in, n_filters, kernel_size=int(k), padding=int(k) // 2, bias=False)
                    for k in kernel_sizes
                ]
            )
            self.pool_branch = nn.Sequential(
                nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False),
            )
            out_channels = n_filters * (len(kernel_sizes) + 1)
            self.bn = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            z = self.bottleneck(x)
            branches = [conv(z) for conv in self.convs]
            branches.append(self.pool_branch(x))
            return self.relu(self.bn(torch.cat(branches, dim=1)))


    class InceptionBlock(nn.Module):
        def __init__(
            self,
            in_channels: int,
            depth: int = 6,
            n_filters: int = 32,
            bottleneck_channels: int = 32,
        ) -> None:
            super().__init__()
            self.blocks = nn.ModuleList()
            self.shortcuts = nn.ModuleDict()
            channels = in_channels
            residual_channels = in_channels
            out_channels = n_filters * 4
            for idx in range(depth):
                self.blocks.append(
                    InceptionModule(
                        channels,
                        n_filters=n_filters,
                        bottleneck_channels=bottleneck_channels,
                    )
                )
                channels = out_channels
                if idx % 3 == 2:
                    if residual_channels == channels:
                        self.shortcuts[str(idx)] = nn.BatchNorm1d(channels)
                    else:
                        self.shortcuts[str(idx)] = nn.Sequential(
                            nn.Conv1d(residual_channels, channels, kernel_size=1, bias=False),
                            nn.BatchNorm1d(channels),
                        )
                    residual_channels = channels
            self.out_channels = channels
            self.relu = nn.ReLU()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            residual = x
            for idx, block in enumerate(self.blocks):
                x = block(x)
                key = str(idx)
                if key in self.shortcuts:
                    x = self.relu(x + self.shortcuts[key](residual))
                    residual = x
            return x


    class InceptionTimeClassifier(nn.Module):
        def __init__(self, n_channels: int, n_classes: int, dropout: float = 0.0) -> None:
            super().__init__()
            dropout = max(0.0, float(dropout))
            self.encoder = InceptionBlock(
                in_channels=n_channels,
                depth=6,
                n_filters=32,
                bottleneck_channels=32,
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(self.encoder.out_channels, n_classes),
            )
            self.feature_dim = int(self.encoder.out_channels)

        def forward_features(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.head[:2](self.encoder(x))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            features = self.forward_features(x)
            features = self.head[2](features)
            return self.head[3](features)

else:

    class Cnn1DClassifier:  # type: ignore[no-redef]
        pass


    class InceptionTimeClassifier:  # type: ignore[no-redef]
        pass


class CnnWindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx], dtype=torch.long)


class CnnWindowFeatureDataset(Dataset):
    def __init__(self, x: np.ndarray, extra: np.ndarray, y: np.ndarray) -> None:
        self.x = np.asarray(x, dtype=np.float32)
        self.extra = np.asarray(extra, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        return (
            torch.from_numpy(self.x[idx]),
            torch.from_numpy(self.extra[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


if nn is not None:

    class FeatureFusionClassifier(nn.Module):
        def __init__(
            self,
            signal_model: "nn.Module",
            signal_feature_dim: int,
            n_extra_features: int,
            n_classes: int,
            dropout: float = 0.25,
        ) -> None:
            super().__init__()
            self.signal_model = signal_model
            self.feature_encoder = nn.Sequential(
                nn.Linear(n_extra_features, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.LayerNorm(32),
                nn.ReLU(),
            )
            self.classifier = nn.Sequential(
                nn.Linear(int(signal_feature_dim) + 32, 96),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(96, n_classes),
            )

        def forward(self, x: "torch.Tensor", extra: "torch.Tensor") -> "torch.Tensor":
            signal_repr = self.signal_model.forward_features(x)
            extra_repr = self.feature_encoder(extra)
            return self.classifier(torch.cat([signal_repr, extra_repr], dim=1))

else:

    class FeatureFusionClassifier:  # type: ignore[no-redef]
        pass


def cnn_cache_name(config: RunConfig) -> Path:
    out_dir = Path("datasets")
    out_dir.mkdir(parents=True, exist_ok=True)
    if config.cnn_max_windows_fraction > 0:
        max_tag = f"mf{int(round(config.cnn_max_windows_fraction * 100)):03d}"
    else:
        max_tag = f"m{config.cnn_max_windows_per_file}"
    return out_dir / (
        f"frailty3_cnn_windows_{role_cache_tag(config)}_fs{config.cnn_target_fs:g}_"
        f"s{config.cnn_seq_sec:g}_h{config.cnn_hop_sec:g}_{max_tag}.npz"
    )


def resample_multichannel(x: np.ndarray, src_fs: float, dst_fs: float) -> np.ndarray:
    if abs(src_fs - dst_fs) < 1e-6:
        return np.asarray(x, dtype=np.float32)
    ratio = Fraction(str(dst_fs)).limit_denominator(1000) / Fraction(str(src_fs)).limit_denominator(1000)
    return signal.resample_poly(x, ratio.numerator, ratio.denominator, axis=0).astype(np.float32)


def standardize_segment(seg: np.ndarray) -> np.ndarray:
    seg = np.asarray(seg, dtype=np.float32)
    med = np.median(seg, axis=0, keepdims=True)
    q75 = np.percentile(seg, 75, axis=0, keepdims=True)
    q25 = np.percentile(seg, 25, axis=0, keepdims=True)
    robust_scale = (q75 - q25) / 1.349
    std_scale = np.std(seg, axis=0, keepdims=True)
    scale = np.where(robust_scale > 1e-6, robust_scale, std_scale)
    seg = (seg - med) / (scale + 1e-6)
    return np.clip(seg, -8.0, 8.0).astype(np.float32)


def select_window_starts(
    n_samples: int,
    seq_samples: int,
    hop_samples: int,
    max_windows: int,
    max_windows_fraction: float = 0.0,
) -> np.ndarray:
    if n_samples <= seq_samples:
        return np.array([0], dtype=np.int64)
    starts = np.arange(0, n_samples - seq_samples + 1, max(1, hop_samples), dtype=np.int64)
    if starts.size == 0 or starts[-1] != n_samples - seq_samples:
        starts = np.append(starts, n_samples - seq_samples)
    cap = int(max_windows)
    if max_windows_fraction > 0:
        fraction = min(1.0, max(0.0, float(max_windows_fraction)))
        cap = max(1, int(math.ceil(starts.size * fraction)))
    if cap > 0 and starts.size > cap:
        keep = np.linspace(0, starts.size - 1, cap).round().astype(np.int64)
        starts = np.unique(starts[keep])
    return starts.astype(np.int64)


def extract_cnn_windows_from_file(path: Path, config: RunConfig) -> np.ndarray:
    df = read_numeric_csv(path)
    red = bandpass_ppg(df["RED"].to_numpy(), config.fs)
    ir = bandpass_ppg(df["IR"].to_numpy(), config.fs)
    acc = lowpass_imu(df[["AX", "AY", "AZ"]].to_numpy(), config.fs, cutoff=20.0)
    gyro = lowpass_imu(df[["GX", "GY", "GZ"]].to_numpy(), config.fs, cutoff=40.0)
    data = np.column_stack([red, ir, acc, gyro]).astype(np.float32)
    data = resample_multichannel(data, src_fs=config.fs, dst_fs=config.cnn_target_fs)

    seq_samples = max(8, int(round(config.cnn_seq_sec * config.cnn_target_fs)))
    hop_samples = max(1, int(round(config.cnn_hop_sec * config.cnn_target_fs)))
    windows: List[np.ndarray] = []
    for start in select_window_starts(
        len(data),
        seq_samples,
        hop_samples,
        config.cnn_max_windows_per_file,
        max_windows_fraction=config.cnn_max_windows_fraction,
    ):
        stop = int(start + seq_samples)
        seg = data[int(start) : min(stop, len(data))]
        seg = standardize_segment(seg)
        if len(seg) < seq_samples:
            seg = np.pad(seg, ((0, seq_samples - len(seg)), (0, 0)), mode="constant")
        windows.append(seg.T.astype(np.float32))
    return np.stack(windows, axis=0) if windows else np.empty((0, 8, seq_samples), dtype=np.float32)


def build_cnn_window_table(
    features: pd.DataFrame,
    config: RunConfig,
    refresh: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path]:
    features = features.reset_index(drop=True)
    cache_path = cnn_cache_name(config)
    expected_paths = features["path"].astype(str).to_numpy(dtype=str)
    if cache_path.exists() and not refresh:
        cached = np.load(cache_path, allow_pickle=False)
        cached_paths = cached["paths"].astype(str)
        if cached_paths.shape == expected_paths.shape and np.all(cached_paths == expected_paths):
            return (
                cached["x"].astype(np.float32),
                cached["y"].astype(np.int64),
                cached["subjects"].astype(str),
                cached["file_index"].astype(np.int64),
                cache_path,
            )

    x_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    subject_rows: List[np.ndarray] = []
    file_rows: List[np.ndarray] = []
    for file_idx, row in features.iterrows():
        path = Path(str(row["path"]))
        windows = extract_cnn_windows_from_file(path, config)
        if windows.size == 0:
            continue
        n = windows.shape[0]
        x_rows.append(windows)
        y_rows.append(np.full(n, int(row["label"]), dtype=np.int64))
        subject_rows.append(np.full(n, str(row["subject"]), dtype="<U64"))
        file_rows.append(np.full(n, int(file_idx), dtype=np.int64))

    if not x_rows:
        raise RuntimeError("No CNN windows could be extracted.")
    x = np.concatenate(x_rows, axis=0).astype(np.float32)
    y = np.concatenate(y_rows, axis=0).astype(np.int64)
    subjects = np.concatenate(subject_rows, axis=0).astype(str)
    file_index = np.concatenate(file_rows, axis=0).astype(np.int64)
    np.savez(
        cache_path,
        x=x,
        y=y,
        subjects=subjects,
        file_index=file_index,
        paths=expected_paths.astype("<U512"),
    )
    return x, y, subjects, file_index, cache_path


def cnn_device() -> "torch.device":
    ensure_torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def forward_window_model(
    model: "nn.Module",
    xb: "torch.Tensor",
    extra: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    if extra is None:
        return model(xb)
    return model(xb, extra)


def cnn_predict_proba(
    model: "nn.Module",
    x: np.ndarray,
    batch_size: int,
    device: "torch.device",
    extra: Optional[np.ndarray] = None,
) -> np.ndarray:
    ensure_torch()
    model.eval()
    probs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size].astype(np.float32)).to(device)
            eb = (
                torch.from_numpy(extra[start : start + batch_size].astype(np.float32)).to(device)
                if extra is not None
                else None
            )
            prob = torch.softmax(forward_window_model(model, xb, eb), dim=1).detach().cpu().numpy()
            probs.append(prob)
    return np.concatenate(probs, axis=0) if probs else np.empty((0, len(CLASS_NAMES)), dtype=np.float32)


def torch_window_eval_metrics(
    model: "nn.Module",
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: "torch.device",
    criterion: Optional["nn.Module"] = None,
    extra: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    ensure_torch()
    if len(y) == 0:
        return {"loss": 0.0, "balanced_accuracy": 0.0}
    model.eval()
    preds: List[int] = []
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size].astype(np.float32)).to(device)
            yb = torch.from_numpy(y[start : start + batch_size].astype(np.int64)).to(device)
            eb = (
                torch.from_numpy(extra[start : start + batch_size].astype(np.float32)).to(device)
                if extra is not None
                else None
            )
            logits = forward_window_model(model, xb, eb)
            if criterion is not None:
                loss = criterion(logits, yb)
                total_loss += float(loss.detach().cpu()) * int(len(yb))
            total_n += int(len(yb))
            preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int).tolist())
    return {
        "loss": finite_float(total_loss / max(1, total_n)),
        "balanced_accuracy": finite_float(balanced_accuracy_score(y, preds)),
    }


def make_torch_window_model(model_name: str, n_channels: int, n_classes: int, dropout: float = -1.0) -> "nn.Module":
    ensure_torch()
    if model_name == "cnn1d":
        return Cnn1DClassifier(n_channels=n_channels, n_classes=n_classes, dropout=dropout)
    if model_name == "inception_time":
        inception_dropout = 0.0 if float(dropout) < 0.0 else float(dropout)
        return InceptionTimeClassifier(n_channels=n_channels, n_classes=n_classes, dropout=inception_dropout)
    raise ValueError(f"Unknown torch window model: {model_name}")


def aggregate_by_key(probs: np.ndarray, labels: np.ndarray, keys: Sequence[object]) -> Tuple[List[int], List[int]]:
    grouped: Dict[object, List[int]] = {}
    for idx, key in enumerate(keys):
        grouped.setdefault(key, []).append(idx)
    true: List[int] = []
    pred: List[int] = []
    for key, idxs in grouped.items():
        idx_arr = np.asarray(idxs, dtype=np.int64)
        avg_probs = np.mean(probs[idx_arr], axis=0)
        true.append(int(labels[idx_arr[0]]))
        pred.append(int(np.argmax(avg_probs)))
    return true, pred


def train_cnn_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: RunConfig,
    seed: int,
    model_name: str = "cnn1d",
    x_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    extra_train: Optional[np.ndarray] = None,
    extra_val: Optional[np.ndarray] = None,
) -> Tuple["nn.Module", Dict[str, object]]:
    ensure_torch()
    set_all_seeds(seed)
    device = cnn_device()
    base_model = make_torch_window_model(
        model_name,
        n_channels=x_train.shape[1],
        n_classes=len(CLASS_NAMES),
        dropout=float(config.cnn_dropout),
    )
    if extra_train is not None:
        fusion_dropout = 0.25 if float(config.cnn_dropout) < 0.0 else float(config.cnn_dropout)
        model = FeatureFusionClassifier(
            base_model,
            signal_feature_dim=int(base_model.feature_dim),
            n_extra_features=int(extra_train.shape[1]),
            n_classes=len(CLASS_NAMES),
            dropout=fusion_dropout,
        ).to(device)
    else:
        model = base_model.to(device)
    counts = np.bincount(y_train, minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = counts.sum() / (len(CLASS_NAMES) * np.maximum(counts, 1.0))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    try:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=max(0.0, float(config.cnn_label_smoothing)),
        )
    except TypeError:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.cnn_lr, weight_decay=float(config.cnn_weight_decay))
    dataset = (
        CnnWindowFeatureDataset(x_train, extra_train, y_train)
        if extra_train is not None
        else CnnWindowDataset(x_train, y_train)
    )
    train_loader = DataLoader(
        dataset,
        batch_size=config.cnn_batch_size,
        shuffle=True,
        num_workers=config.cnn_num_workers,
    )

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: List[Dict[str, float]] = []
    for epoch in range(1, config.cnn_epochs + 1):
        model.train()
        losses: List[float] = []
        train_true: List[int] = []
        train_pred: List[int] = []
        for batch in train_loader:
            if extra_train is not None:
                xb, eb, yb = batch
                eb = eb.to(device)
            else:
                xb, yb = batch
                eb = None
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = forward_window_model(model, xb, eb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            train_true.extend(yb.detach().cpu().numpy().astype(int).tolist())
            train_pred.extend(torch.argmax(logits.detach(), dim=1).cpu().numpy().astype(int).tolist())

        mean_loss = float(np.mean(losses)) if losses else 0.0
        train_score = finite_float(balanced_accuracy_score(train_true, train_pred)) if train_true else 0.0
        val_loss: Optional[float] = None
        if x_val is not None and y_val is not None and len(y_val):
            val_metrics = torch_window_eval_metrics(
                model,
                x_val,
                y_val,
                config.cnn_batch_size,
                device,
                criterion,
                extra=extra_val,
            )
            val_loss = val_metrics["loss"]
            val_score = val_metrics["balanced_accuracy"]
        else:
            val_score = -mean_loss
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": mean_loss,
                "train_balanced_accuracy": train_score,
                "val_loss": val_loss,
                "val_balanced_accuracy": val_score,
            }
        )

        if not bool(getattr(config, "cnn_select_best_epoch", True)):
            best_score = val_score
            best_epoch = epoch
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            continue

        if val_score > best_score + 1e-6:
            best_score = val_score
            best_epoch = epoch
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1
            if x_val is not None and config.cnn_patience > 0 and stale_epochs >= config.cnn_patience:
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, {"best_epoch": int(best_epoch), "best_window_balanced_accuracy": finite_float(best_score), "history": history}


def evaluate_cnn(
    features: pd.DataFrame,
    config: RunConfig,
    model_name: str = "cnn1d",
    refresh_cnn: bool = False,
) -> Dict[str, object]:
    ensure_torch()
    features = features.reset_index(drop=True)
    x_win, y_win, subject_win, file_win, cnn_cache_path = build_cnn_window_table(features, config, refresh=refresh_cnn)
    y_file = features["label"].to_numpy(dtype=int)
    groups = features["subject"].to_numpy()
    extra_cols = select_extra_feature_columns(features, config.extra_input)

    subject_labels = features.groupby("subject")["label"].first()
    min_class_subjects = int(subject_labels.value_counts().min())
    n_splits = max(2, min(config.folds, min_class_subjects))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.seed)

    window_true: List[int] = []
    window_pred: List[int] = []
    file_true: List[int] = []
    file_pred: List[int] = []
    subject_true: List[int] = []
    subject_pred: List[int] = []
    fold_summaries: List[Dict[str, object]] = []
    device = cnn_device()

    for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(features)), y_file, groups), start=1):
        train_mask = np.isin(file_win, train_idx)
        test_mask = np.isin(file_win, test_idx)
        window_extra: Optional[np.ndarray] = None
        if extra_cols:
            scaled_file_features = scaled_file_features_for_fold(features, extra_cols, train_idx)
            window_extra = scaled_file_features[file_win]
        model, fold_info = train_cnn_model(
            x_win[train_mask],
            y_win[train_mask],
            config,
            seed=config.seed + fold,
            model_name=model_name,
            x_val=x_win[test_mask],
            y_val=y_win[test_mask],
            extra_train=window_extra[train_mask] if window_extra is not None else None,
            extra_val=window_extra[test_mask] if window_extra is not None else None,
        )
        probs = cnn_predict_proba(
            model,
            x_win[test_mask],
            config.cnn_batch_size,
            device,
            extra=window_extra[test_mask] if window_extra is not None else None,
        )
        preds = np.argmax(probs, axis=1)
        y_test = y_win[test_mask]
        window_true.extend(y_test.tolist())
        window_pred.extend(preds.tolist())

        fold_file_true, fold_file_pred = aggregate_by_key(probs, y_test, file_win[test_mask].tolist())
        fold_subject_true, fold_subject_pred = aggregate_by_key(probs, y_test, subject_win[test_mask].tolist())
        file_true.extend(fold_file_true)
        file_pred.extend(fold_file_pred)
        subject_true.extend(fold_subject_true)
        subject_pred.extend(fold_subject_pred)

        fold_summaries.append(
            {
                "fold": fold,
                "n_train_files": int(len(train_idx)),
                "n_test_files": int(len(test_idx)),
                "n_train_windows": int(np.sum(train_mask)),
                "n_test_windows": int(np.sum(test_mask)),
                "test_subjects": sorted(map(str, np.unique(groups[test_idx]))),
                "best_epoch": fold_info["best_epoch"],
                "best_window_balanced_accuracy": fold_info["best_window_balanced_accuracy"],
                "history": fold_info["history"],
                "file_balanced_accuracy": finite_float(balanced_accuracy_score(fold_file_true, fold_file_pred)),
                "subject_balanced_accuracy": finite_float(balanced_accuracy_score(fold_subject_true, fold_subject_pred)),
            }
        )

    return {
        "model": model_name,
        "n_files": int(len(features)),
        "n_subjects": int(features["subject"].nunique()),
        "n_windows": int(len(y_win)),
        "extra_input": normalize_extra_input(config.extra_input),
        "n_extra_features": int(len(extra_cols)),
        "n_splits": int(n_splits),
        "cnn_cache": str(cnn_cache_path),
        "window_balanced_accuracy": finite_float(balanced_accuracy_score(window_true, window_pred)),
        "window_macro_f1": finite_float(f1_score(window_true, window_pred, average="macro")),
        "file_balanced_accuracy": finite_float(balanced_accuracy_score(file_true, file_pred)),
        "file_macro_f1": finite_float(f1_score(file_true, file_pred, average="macro")),
        "subject_balanced_accuracy": finite_float(balanced_accuracy_score(subject_true, subject_pred)),
        "subject_macro_f1": finite_float(f1_score(subject_true, subject_pred, average="macro")),
        "window_confusion_matrix": confusion_matrix(window_true, window_pred, labels=[0, 1, 2]).tolist(),
        "file_confusion_matrix": confusion_matrix(file_true, file_pred, labels=[0, 1, 2]).tolist(),
        "subject_confusion_matrix": confusion_matrix(subject_true, subject_pred, labels=[0, 1, 2]).tolist(),
        "file_classification_report": classification_report(
            file_true,
            file_pred,
            labels=[0, 1, 2],
            target_names=list(CLASS_NAMES),
            zero_division=0,
            output_dict=True,
        ),
        "subject_classification_report": classification_report(
            subject_true,
            subject_pred,
            labels=[0, 1, 2],
            target_names=list(CLASS_NAMES),
            zero_division=0,
            output_dict=True,
        ),
        "folds": fold_summaries,
        "feature_columns": extra_cols,
    }


def ensure_shapeformer() -> None:
    ensure_torch()
    if PortedShapeFormer is None or discover_shapelets is None:
        raise RuntimeError("The shapeformer model could not be imported from shapeformer_port.py.")


def shapelet_count_summary(info: np.ndarray) -> Dict[str, Dict[str, int]]:
    info = np.asarray(info)
    by_class: Dict[str, int] = {}
    by_channel: Dict[str, int] = {}
    if info.size == 0:
        return {"by_class": by_class, "by_channel": by_channel}
    for cls in info[:, 4].astype(int):
        name = CLASS_NAMES[int(cls)] if 0 <= int(cls) < len(CLASS_NAMES) else str(int(cls))
        by_class[name] = by_class.get(name, 0) + 1
    for channel in info[:, 5].astype(int):
        name = REQUIRED_COLUMNS[int(channel)] if 0 <= int(channel) < len(REQUIRED_COLUMNS) else str(int(channel))
        by_channel[name] = by_channel.get(name, 0) + 1
    return {"by_class": by_class, "by_channel": by_channel}


def train_shapeformer_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: RunConfig,
    seed: int,
    x_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    extra_train: Optional[np.ndarray] = None,
    extra_val: Optional[np.ndarray] = None,
) -> Tuple["nn.Module", Dict[str, object]]:
    ensure_shapeformer()
    set_all_seeds(seed)
    device = cnn_device()
    stride = config.shapeformer_shapelet_stride if config.shapeformer_shapelet_stride > 0 else None
    if config.shapeformer_discovery_method == "pisd":
        if discover_shapelets_pisd is None:
            raise RuntimeError("Original PISD ShapeFormer discovery could not be imported.")
        bundle = discover_shapelets_pisd(
            x_train,
            y_train,
            n_shapelets_per_class=config.shapeformer_num_shapelets,
            shapelet_len=config.shapeformer_shapelet_len,
            max_discovery_windows=config.shapeformer_discovery_windows,
            num_pip=config.shapeformer_num_pip,
            processes=config.shapeformer_processes,
            seed=seed,
            verbose=config.shapeformer_pisd_verbose,
        )
    else:
        bundle = discover_shapelets(
            x_train,
            y_train,
            n_shapelets_per_class=config.shapeformer_num_shapelets,
            shapelet_len=config.shapeformer_shapelet_len,
            stride=stride,
            max_discovery_windows=config.shapeformer_discovery_windows,
            candidates_per_class_channel=config.shapeformer_candidates_per_class_channel,
            seed=seed,
        )
    base_model = PortedShapeFormer(
        n_channels=x_train.shape[1],
        seq_len=x_train.shape[2],
        n_classes=len(CLASS_NAMES),
        shapelets_info=bundle.info,
        shapelets=bundle.shapelets,
        len_w=config.shapeformer_local_window,
        local_embed_dim=config.shapeformer_local_embed_dim,
        shape_embed_dim=config.shapeformer_shape_embed_dim,
        dim_ff=config.shapeformer_dim_ff,
        num_heads=config.shapeformer_heads,
        dropout=config.shapeformer_dropout,
        shapelet_search_window=max(1, config.shapeformer_shapelet_len // 2),
    )
    if extra_train is not None:
        model = FeatureFusionClassifier(
            base_model,
            signal_feature_dim=int(base_model.feature_dim),
            n_extra_features=int(extra_train.shape[1]),
            n_classes=len(CLASS_NAMES),
        ).to(device)
    else:
        model = base_model.to(device)

    counts = np.bincount(y_train, minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = counts.sum() / (len(CLASS_NAMES) * np.maximum(counts, 1.0))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    try:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=max(0.0, float(config.cnn_label_smoothing)),
        )
    except TypeError:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.cnn_lr, weight_decay=float(config.cnn_weight_decay))
    dataset = (
        CnnWindowFeatureDataset(x_train, extra_train, y_train)
        if extra_train is not None
        else CnnWindowDataset(x_train, y_train)
    )
    train_loader = DataLoader(
        dataset,
        batch_size=config.cnn_batch_size,
        shuffle=True,
        num_workers=config.cnn_num_workers,
    )

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: List[Dict[str, float]] = []
    for epoch in range(1, config.cnn_epochs + 1):
        model.train()
        losses: List[float] = []
        train_true: List[int] = []
        train_pred: List[int] = []
        for batch in train_loader:
            if extra_train is not None:
                xb, eb, yb = batch
                eb = eb.to(device)
            else:
                xb, yb = batch
                eb = None
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = forward_window_model(model, xb, eb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            train_true.extend(yb.detach().cpu().numpy().astype(int).tolist())
            train_pred.extend(torch.argmax(logits.detach(), dim=1).cpu().numpy().astype(int).tolist())

        mean_loss = float(np.mean(losses)) if losses else 0.0
        train_score = finite_float(balanced_accuracy_score(train_true, train_pred)) if train_true else 0.0
        val_loss: Optional[float] = None
        if x_val is not None and y_val is not None and len(y_val):
            val_metrics = torch_window_eval_metrics(
                model,
                x_val,
                y_val,
                config.cnn_batch_size,
                device,
                criterion,
                extra=extra_val,
            )
            val_loss = val_metrics["loss"]
            val_score = val_metrics["balanced_accuracy"]
        else:
            val_score = -mean_loss
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": mean_loss,
                "train_balanced_accuracy": train_score,
                "val_loss": val_loss,
                "val_balanced_accuracy": val_score,
            }
        )

        if val_score > best_score + 1e-6:
            best_score = val_score
            best_epoch = epoch
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1
            if x_val is not None and config.cnn_patience > 0 and stale_epochs >= config.cnn_patience:
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    summary = shapelet_count_summary(bundle.info)
    return model, {
        "best_epoch": int(best_epoch),
        "best_window_balanced_accuracy": finite_float(best_score),
        "history": history,
        "n_shapelets": int(len(bundle.shapelets)),
        "shapelet_counts": summary,
        "discovery_windows": int(len(bundle.discovery_indices)),
        "discovery_method": config.shapeformer_discovery_method,
        "shapelets_info": bundle.info.tolist(),
        "shapelets": [np.asarray(shapelet, dtype=np.float32).tolist() for shapelet in bundle.shapelets],
    }


def evaluate_shapeformer(
    features: pd.DataFrame,
    config: RunConfig,
    refresh_cnn: bool = False,
) -> Dict[str, object]:
    ensure_shapeformer()
    features = features.reset_index(drop=True)
    x_win, y_win, subject_win, file_win, cnn_cache_path = build_cnn_window_table(features, config, refresh=refresh_cnn)
    y_file = features["label"].to_numpy(dtype=int)
    groups = features["subject"].to_numpy()
    extra_cols = select_extra_feature_columns(features, config.extra_input)

    subject_labels = features.groupby("subject")["label"].first()
    min_class_subjects = int(subject_labels.value_counts().min())
    n_splits = max(2, min(config.folds, min_class_subjects))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.seed)

    window_true: List[int] = []
    window_pred: List[int] = []
    file_true: List[int] = []
    file_pred: List[int] = []
    subject_true: List[int] = []
    subject_pred: List[int] = []
    fold_summaries: List[Dict[str, object]] = []
    device = cnn_device()

    for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(features)), y_file, groups), start=1):
        train_mask = np.isin(file_win, train_idx)
        test_mask = np.isin(file_win, test_idx)
        window_extra: Optional[np.ndarray] = None
        if extra_cols:
            scaled_file_features = scaled_file_features_for_fold(features, extra_cols, train_idx)
            window_extra = scaled_file_features[file_win]
        model, fold_info = train_shapeformer_model(
            x_win[train_mask],
            y_win[train_mask],
            config,
            seed=config.seed + fold,
            x_val=x_win[test_mask],
            y_val=y_win[test_mask],
            extra_train=window_extra[train_mask] if window_extra is not None else None,
            extra_val=window_extra[test_mask] if window_extra is not None else None,
        )
        probs = cnn_predict_proba(
            model,
            x_win[test_mask],
            config.cnn_batch_size,
            device,
            extra=window_extra[test_mask] if window_extra is not None else None,
        )
        preds = np.argmax(probs, axis=1)
        y_test = y_win[test_mask]
        window_true.extend(y_test.tolist())
        window_pred.extend(preds.tolist())

        fold_file_true, fold_file_pred = aggregate_by_key(probs, y_test, file_win[test_mask].tolist())
        fold_subject_true, fold_subject_pred = aggregate_by_key(probs, y_test, subject_win[test_mask].tolist())
        file_true.extend(fold_file_true)
        file_pred.extend(fold_file_pred)
        subject_true.extend(fold_subject_true)
        subject_pred.extend(fold_subject_pred)

        fold_summaries.append(
            {
                "fold": fold,
                "n_train_files": int(len(train_idx)),
                "n_test_files": int(len(test_idx)),
                "n_train_windows": int(np.sum(train_mask)),
                "n_test_windows": int(np.sum(test_mask)),
                "test_subjects": sorted(map(str, np.unique(groups[test_idx]))),
                "best_epoch": fold_info["best_epoch"],
                "best_window_balanced_accuracy": fold_info["best_window_balanced_accuracy"],
                "history": fold_info["history"],
                "n_shapelets": fold_info["n_shapelets"],
                "shapelet_counts": fold_info["shapelet_counts"],
                "discovery_windows": fold_info["discovery_windows"],
                "discovery_method": fold_info["discovery_method"],
                "shapelets_info": fold_info["shapelets_info"],
                "file_balanced_accuracy": finite_float(balanced_accuracy_score(fold_file_true, fold_file_pred)),
                "subject_balanced_accuracy": finite_float(balanced_accuracy_score(fold_subject_true, fold_subject_pred)),
            }
        )

    return {
        "model": "shapeformer",
        "shapeformer_source": "/home/trinker/Code/github/multivariate-time-series-analysis/ShapeFormer",
        "n_files": int(len(features)),
        "n_subjects": int(features["subject"].nunique()),
        "n_windows": int(len(y_win)),
        "extra_input": normalize_extra_input(config.extra_input),
        "n_extra_features": int(len(extra_cols)),
        "n_splits": int(n_splits),
        "cnn_cache": str(cnn_cache_path),
        "window_balanced_accuracy": finite_float(balanced_accuracy_score(window_true, window_pred)),
        "window_macro_f1": finite_float(f1_score(window_true, window_pred, average="macro")),
        "file_balanced_accuracy": finite_float(balanced_accuracy_score(file_true, file_pred)),
        "file_macro_f1": finite_float(f1_score(file_true, file_pred, average="macro")),
        "subject_balanced_accuracy": finite_float(balanced_accuracy_score(subject_true, subject_pred)),
        "subject_macro_f1": finite_float(f1_score(subject_true, subject_pred, average="macro")),
        "window_confusion_matrix": confusion_matrix(window_true, window_pred, labels=[0, 1, 2]).tolist(),
        "file_confusion_matrix": confusion_matrix(file_true, file_pred, labels=[0, 1, 2]).tolist(),
        "subject_confusion_matrix": confusion_matrix(subject_true, subject_pred, labels=[0, 1, 2]).tolist(),
        "file_classification_report": classification_report(
            file_true,
            file_pred,
            labels=[0, 1, 2],
            target_names=list(CLASS_NAMES),
            zero_division=0,
            output_dict=True,
        ),
        "subject_classification_report": classification_report(
            subject_true,
            subject_pred,
            labels=[0, 1, 2],
            target_names=list(CLASS_NAMES),
            zero_division=0,
            output_dict=True,
        ),
        "folds": fold_summaries,
        "feature_columns": extra_cols,
    }


def make_model(name: str, seed: int) -> Pipeline:
    if name == "logreg_l2":
        clf = LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=3000,
            random_state=seed,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )
    if name == "rbf_svm":
        clf = SVC(C=1.0, gamma="scale", kernel="rbf", class_weight="balanced", probability=True, random_state=seed)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )
    if name == "extra_trees":
        clf = ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", clf),
            ]
        )
    raise ValueError(f"Unknown model: {name}")


def predict_scores(model: Pipeline, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)
    if hasattr(model[-1], "predict_proba"):
        return model.predict_proba(x)
    decision = model.decision_function(x)
    decision = np.atleast_2d(decision)
    decision = decision - np.max(decision, axis=1, keepdims=True)
    exp = np.exp(decision)
    return exp / np.sum(exp, axis=1, keepdims=True)


def evaluate(
    features: pd.DataFrame,
    model_name: str,
    config: RunConfig,
) -> Dict[str, object]:
    meta_cols = {"path", "dataset", "subject", "role", "class_name", "label"}
    feature_cols = [c for c in features.columns if c not in meta_cols]
    x = features[feature_cols].to_numpy(dtype=float)
    y = features["label"].to_numpy(dtype=int)
    groups = features["subject"].to_numpy()

    subject_labels = features.groupby("subject")["label"].first()
    min_class_subjects = int(subject_labels.value_counts().min())
    n_splits = max(2, min(config.folds, min_class_subjects))

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
    file_true: List[int] = []
    file_pred: List[int] = []
    subject_true: List[int] = []
    subject_pred: List[int] = []
    fold_summaries: List[Dict[str, object]] = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(x, y, groups), start=1):
        model = make_model(model_name, config.seed + fold)
        model.fit(x[train_idx], y[train_idx])

        pred = model.predict(x[test_idx])
        scores = predict_scores(model, x[test_idx])
        file_true.extend(y[test_idx].tolist())
        file_pred.extend(pred.tolist())

        fold_frame = features.iloc[test_idx][["subject", "label"]].copy()
        fold_frame["row_in_fold"] = np.arange(len(test_idx))
        for subject, sub in fold_frame.groupby("subject"):
            idx = sub["row_in_fold"].to_numpy(dtype=int)
            avg_scores = np.mean(scores[idx], axis=0)
            subject_true.append(int(sub["label"].iloc[0]))
            subject_pred.append(int(np.argmax(avg_scores)))

        fold_summaries.append(
            {
                "fold": fold,
                "n_train_files": int(len(train_idx)),
                "n_test_files": int(len(test_idx)),
                "test_subjects": sorted(map(str, np.unique(groups[test_idx]))),
                "file_balanced_accuracy": finite_float(balanced_accuracy_score(y[test_idx], pred)),
            }
        )

    report = {
        "model": model_name,
        "n_files": int(len(features)),
        "n_subjects": int(features["subject"].nunique()),
        "n_splits": int(n_splits),
        "file_balanced_accuracy": finite_float(balanced_accuracy_score(file_true, file_pred)),
        "file_macro_f1": finite_float(f1_score(file_true, file_pred, average="macro")),
        "subject_balanced_accuracy": finite_float(balanced_accuracy_score(subject_true, subject_pred)),
        "subject_macro_f1": finite_float(f1_score(subject_true, subject_pred, average="macro")),
        "file_confusion_matrix": confusion_matrix(file_true, file_pred, labels=[0, 1, 2]).tolist(),
        "subject_confusion_matrix": confusion_matrix(subject_true, subject_pred, labels=[0, 1, 2]).tolist(),
        "file_classification_report": classification_report(
            file_true,
            file_pred,
            labels=[0, 1, 2],
            target_names=list(CLASS_NAMES),
            zero_division=0,
            output_dict=True,
        ),
        "subject_classification_report": classification_report(
            subject_true,
            subject_pred,
            labels=[0, 1, 2],
            target_names=list(CLASS_NAMES),
            zero_division=0,
            output_dict=True,
        ),
        "folds": fold_summaries,
        "feature_columns": feature_cols,
    }
    return report


def save_final_model(features: pd.DataFrame, model_name: str, config: RunConfig, report: Dict[str, object]) -> Path:
    meta_cols = {"path", "dataset", "subject", "role", "class_name", "label"}
    feature_cols = [c for c in features.columns if c not in meta_cols]
    x = features[feature_cols].to_numpy(dtype=float)
    y = features["label"].to_numpy(dtype=int)
    model = make_model(model_name, config.seed)
    model.fit(x, y)

    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"frailty3_{model_name}.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_cols,
            "class_names": CLASS_NAMES,
            "config": asdict(config),
            "cv_report": report,
        },
        out_path,
    )
    return out_path


def save_final_cnn_model(
    features: pd.DataFrame,
    config: RunConfig,
    report: Dict[str, object],
    model_name: str = "cnn1d",
    refresh_cnn: bool = False,
) -> Path:
    ensure_torch()
    features = features.reset_index(drop=True)
    x_win, y_win, _, _, cnn_cache_path = build_cnn_window_table(features, config, refresh=refresh_cnn)
    extra_cols = select_extra_feature_columns(features, config.extra_input)
    extra_win: Optional[np.ndarray] = None
    if extra_cols:
        all_file_idx = np.arange(len(features), dtype=np.int64)
        scaled_file_features = scaled_file_features_for_fold(features, extra_cols, all_file_idx)
        _, _, _, file_win, _ = build_cnn_window_table(features, config, refresh=False)
        extra_win = scaled_file_features[file_win]
    model, train_info = train_cnn_model(
        x_win,
        y_win,
        config,
        seed=config.seed,
        model_name=model_name,
        extra_train=extra_win,
    )
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"frailty3_{model_name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "class_names": CLASS_NAMES,
            "config": asdict(config),
            "cv_report": report,
            "train_info": train_info,
            "cnn_cache": str(cnn_cache_path),
            "n_channels": int(x_win.shape[1]),
            "extra_input": normalize_extra_input(config.extra_input),
            "extra_feature_columns": extra_cols,
        },
        out_path,
    )
    return out_path


def save_final_shapeformer_model(
    features: pd.DataFrame,
    config: RunConfig,
    report: Dict[str, object],
    refresh_cnn: bool = False,
) -> Path:
    ensure_shapeformer()
    features = features.reset_index(drop=True)
    x_win, y_win, _, file_win, cnn_cache_path = build_cnn_window_table(features, config, refresh=refresh_cnn)
    extra_cols = select_extra_feature_columns(features, config.extra_input)
    extra_win: Optional[np.ndarray] = None
    if extra_cols:
        all_file_idx = np.arange(len(features), dtype=np.int64)
        scaled_file_features = scaled_file_features_for_fold(features, extra_cols, all_file_idx)
        extra_win = scaled_file_features[file_win]
    model, train_info = train_shapeformer_model(x_win, y_win, config, seed=config.seed, extra_train=extra_win)
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "frailty3_shapeformer.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": "shapeformer",
            "class_names": CLASS_NAMES,
            "config": asdict(config),
            "cv_report": report,
            "train_info": train_info,
            "cnn_cache": str(cnn_cache_path),
            "n_channels": int(x_win.shape[1]),
            "seq_len": int(x_win.shape[2]),
            "extra_input": normalize_extra_input(config.extra_input),
            "extra_feature_columns": extra_cols,
        },
        out_path,
    )
    return out_path


def save_learning_curve_artifacts(
    report: Dict[str, object],
    out_dir: Optional[Path] = None,
    filename_prefix: Optional[str] = None,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for fold in report.get("folds", []):
        fold_id = int(fold.get("fold", len(rows) + 1))
        for item in fold.get("history", []):
            rows.append(
                {
                    "model": report.get("model"),
                    "fold": fold_id,
                    "epoch": int(item.get("epoch", 0)),
                    "train_loss": item.get("train_loss"),
                    "train_balanced_accuracy": item.get("train_balanced_accuracy"),
                    "val_loss": item.get("val_loss"),
                    "val_balanced_accuracy": item.get("val_balanced_accuracy"),
                }
            )
    if not rows:
        return {}

    out_dir = out_dir if out_dir is not None else Path("results_frailty3") / "learning_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = str(report.get("model", "model"))
    file_stem = filename_prefix if filename_prefix is not None else model_name
    csv_path = out_dir / f"{file_stem}_learning_curve.csv"
    png_path = out_dir / f"{file_stem}_learning_curve.png"
    curve_df = pd.DataFrame(rows)
    curve_df.to_csv(csv_path, index=False)

    best_loss: Dict[str, object] = {}
    loss_df = curve_df.dropna(subset=["val_loss"])
    if not loss_df.empty:
        row = loss_df.loc[loss_df["val_loss"].astype(float).idxmin()]
        best_loss = {
            "fold": int(row["fold"]),
            "epoch": int(row["epoch"]),
            "val_loss": finite_float(row["val_loss"]),
        }

    best_accuracy: Dict[str, object] = {}
    acc_df = curve_df.dropna(subset=["val_balanced_accuracy"])
    if not acc_df.empty:
        row = acc_df.loc[acc_df["val_balanced_accuracy"].astype(float).idxmax()]
        best_accuracy = {
            "fold": int(row["fold"]),
            "epoch": int(row["epoch"]),
            "val_balanced_accuracy": finite_float(row["val_balanced_accuracy"]),
        }

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for fold_id, sub in curve_df.groupby("fold"):
            axes[0].plot(sub["epoch"], sub["train_loss"], color="#7aa6c2", alpha=0.35, linewidth=1)
            if sub["val_loss"].notna().any():
                axes[0].plot(sub["epoch"], sub["val_loss"], color="#d9796c", alpha=0.45, linewidth=1)
            if sub["train_balanced_accuracy"].notna().any():
                axes[1].plot(sub["epoch"], sub["train_balanced_accuracy"], color="#7aa6c2", alpha=0.35, linewidth=1)
            if sub["val_balanced_accuracy"].notna().any():
                axes[1].plot(sub["epoch"], sub["val_balanced_accuracy"], color="#d9796c", alpha=0.45, linewidth=1)

        mean_df = curve_df.groupby("epoch", as_index=False).mean(numeric_only=True)
        axes[0].plot(mean_df["epoch"], mean_df["train_loss"], color="#1f5f86", linewidth=2, label="train loss")
        if mean_df["val_loss"].notna().any():
            axes[0].plot(mean_df["epoch"], mean_df["val_loss"], color="#b43c30", linewidth=2, label="val loss")
        if "val_loss" in mean_df and mean_df["val_loss"].notna().any():
            best_mean = mean_df.loc[mean_df["val_loss"].idxmin()]
            axes[0].scatter([best_mean["epoch"]], [best_mean["val_loss"]], color="#b43c30", s=45, zorder=5)
        axes[0].set_title(f"{model_name} loss")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend()

        if mean_df["train_balanced_accuracy"].notna().any():
            axes[1].plot(
                mean_df["epoch"],
                mean_df["train_balanced_accuracy"],
                color="#1f5f86",
                linewidth=2,
                label="train balanced acc",
            )
        if mean_df["val_balanced_accuracy"].notna().any():
            axes[1].plot(
                mean_df["epoch"],
                mean_df["val_balanced_accuracy"],
                color="#b43c30",
                linewidth=2,
                label="val balanced acc",
            )
            best_mean = mean_df.loc[mean_df["val_balanced_accuracy"].idxmax()]
            axes[1].scatter([best_mean["epoch"]], [best_mean["val_balanced_accuracy"]], color="#b43c30", s=45, zorder=5)
        axes[1].set_title(f"{model_name} balanced accuracy")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("balanced accuracy")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(png_path, dpi=180)
        plt.close(fig)
        plot_path: Optional[str] = str(png_path)
    except Exception as exc:
        plot_path = None
        best_loss["plot_error"] = str(exc)

    return {
        "learning_curve_csv": str(csv_path),
        "learning_curve_png": plot_path,
        "best_validation_loss": best_loss,
        "best_validation_accuracy": best_accuracy,
    }


def write_report(
    report: Dict[str, object],
    config: RunConfig,
    cache_path: Path,
    skipped: Dict[str, List[str]],
    out_dir: Optional[Path] = None,
    filename_stem: Optional[str] = None,
) -> Path:
    out_dir = out_dir if out_dir is not None else Path("results_frailty3")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = filename_stem if filename_stem is not None else str(report["model"])
    out_path = out_dir / f"{stem}_report.json"
    payload = {
        "config": asdict(config),
        "feature_cache": str(cache_path),
        "class_names": CLASS_NAMES,
        "skipped": skipped,
        **report,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def print_dataset_summary(features: pd.DataFrame, skipped: Dict[str, List[str]]) -> None:
    subject_counts = features.drop_duplicates("subject")["class_name"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    file_counts = features["class_name"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    print("\nDataset summary")
    print("subjects by class:")
    print(subject_counts.to_string())
    print("files by class:")
    print(file_counts.to_string())
    if skipped["study_unmatched"]:
        print("\nStudyData files skipped because subject ID has no matching label:")
        print(", ".join(skipped["study_unmatched"][:12]) + (" ..." if len(skipped["study_unmatched"]) > 12 else ""))
    if skipped.get("excluded_role"):
        print("\nFiles skipped because suffix is not in B/R1/R2/R3/R4:")
        print(", ".join(skipped["excluded_role"][:12]) + (" ..." if len(skipped["excluded_role"]) > 12 else ""))
    if skipped["bad_columns"]:
        print("\nFiles skipped because they could not be parsed:")
        print("\n".join(skipped["bad_columns"]))


def parse_csv_values(raw: str, cast=str) -> List[object]:
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if item:
            values.append(cast(item))
    return values


def parse_percent_values(raw: str) -> List[float]:
    values: List[float] = []
    for item in str(raw).split(","):
        text = item.strip().replace("%", "")
        if not text:
            continue
        value = float(text)
        values.append(value / 100.0 if value > 1.0 else value)
    return values


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{sec:02d}s"
    return f"{minutes:d}m{sec:02d}s"


def progress_message(done: int, total: int, start_time: float, label: str) -> None:
    elapsed = time.perf_counter() - start_time
    rate = elapsed / max(1, done)
    eta = rate * max(0, total - done)
    width = 28
    filled = int(round(width * done / max(1, total)))
    bar = "#" * filled + "-" * (width - filled)
    pct = 100.0 * done / max(1, total)
    print(
        f"[{bar}] {done}/{total} {pct:5.1f}% | elapsed {format_duration(elapsed)} | "
        f"ETA {format_duration(eta)} | {label}",
        flush=True,
    )


class SweepProgress:
    def __init__(self, total_runs: int) -> None:
        self.total_runs = int(total_runs)
        self.completed = 0
        self.start_time = time.perf_counter()
        self.total_bar = None
        self.group_bar = None
        self.group_label = ""
        self.use_tqdm = tqdm is not None and sys.stderr.isatty()
        self.log_every = max(1, self.total_runs // 50)
        if self.use_tqdm:
            self.total_bar = tqdm(
                total=self.total_runs,
                desc="Total sweep",
                position=0,
                leave=True,
                dynamic_ncols=True,
                unit="run",
            )

    def start_group(self, label: str, total_repeats: int) -> None:
        self.close_group()
        self.group_label = str(label)
        if self.use_tqdm:
            self.group_bar = tqdm(
                total=max(1, int(total_repeats)),
                desc="Current group",
                position=1,
                leave=False,
                dynamic_ncols=True,
                unit="rep",
            )
            self.group_bar.set_postfix_str(self.group_label, refresh=True)

    def set_current_run(self, label: str) -> None:
        if self.group_bar is not None:
            self.group_bar.set_postfix_str(str(label), refresh=True)
        elif not self.use_tqdm:
            self.group_label = str(label)

    def finish_run(self, label: str) -> None:
        self.completed += 1
        if self.group_bar is not None:
            self.group_bar.update(1)
            self.group_bar.set_postfix_str(str(label), refresh=True)
        if self.total_bar is not None:
            self.total_bar.update(1)
            self.total_bar.set_postfix_str(str(label), refresh=True)
        elif not self.use_tqdm and (self.completed == self.total_runs or self.completed % self.log_every == 0):
            self.group_label = str(label)
            self._fallback_refresh("done")

    def close_group(self) -> None:
        if self.group_bar is not None:
            self.group_bar.close()
            self.group_bar = None

    def close(self) -> None:
        self.close_group()
        if self.total_bar is not None:
            self.total_bar.close()
            self.total_bar = None

    def write(self, message: str) -> None:
        if self.use_tqdm:
            tqdm.write(str(message))
        else:
            sys.stdout.write(f"\n{message}\n")
            sys.stdout.flush()

    def _fallback_refresh(self, state: str) -> None:
        elapsed = time.perf_counter() - self.start_time
        rate = elapsed / max(1, self.completed)
        eta = rate * max(0, self.total_runs - self.completed)
        pct = 100.0 * self.completed / max(1, self.total_runs)
        message = (
            f"Total sweep {self.completed}/{self.total_runs} {pct:5.1f}% | "
            f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta)} | "
            f"{state}: {self.group_label[:120]}"
        )
        sys.stdout.write(f"{message}\n")
        sys.stdout.flush()


def unique_minute_run_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base = datetime.now().strftime("%Y%m%d_%H%M")
    candidate = root / base
    if not candidate.exists():
        candidate.mkdir(parents=True)
        return candidate
    for idx in range(2, 1000):
        candidate = root / f"{base}_{idx:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError(f"Could not create a unique run directory under {root}")


def sanitize_tag(text: str) -> str:
    out = []
    for ch in str(text):
        out.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "-")
    return "".join(out).strip("-")


def resolve_model_alias(model_name: str) -> Tuple[str, str]:
    name = str(model_name).strip()
    aliases = {
        "cnn": ("cnn1d", "cnn"),
        "cnn1d": ("cnn1d", "cnn1d"),
        "inceptiontime": ("inception_time", "inceptiontime"),
        "inception_time": ("inception_time", "inception_time"),
        "shapeformer": ("shapeformer", "shapeformer"),
        "shapeformer_pisd": ("shapeformer", "shapeformer_pisd"),
    }
    if name in aliases:
        return aliases[name]
    return name, name


def config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        data_root=args.data_root,
        fs=args.fs,
        win_sec=args.win_sec,
        hop_sec=args.hop_sec,
        folds=args.folds,
        seed=args.seed,
        cnn_target_fs=args.cnn_target_fs,
        cnn_seq_sec=args.cnn_seq_sec,
        cnn_hop_sec=args.cnn_hop_sec,
        cnn_max_windows_per_file=args.cnn_max_windows_per_file,
        cnn_max_windows_fraction=args.cnn_max_windows_fraction,
        cnn_epochs=args.cnn_epochs,
        cnn_batch_size=args.cnn_batch_size,
        cnn_lr=args.cnn_lr,
        cnn_patience=args.cnn_patience,
        cnn_num_workers=args.cnn_num_workers,
        cnn_weight_decay=args.cnn_weight_decay,
        cnn_dropout=args.cnn_dropout,
        cnn_label_smoothing=args.cnn_label_smoothing,
        cnn_select_best_epoch=not args.cnn_use_final_epoch,
        role_mode=normalize_role_mode(args.role_mode),
        extra_input=normalize_extra_input(args.extra_input),
        shapeformer_num_shapelets=args.shapeformer_num_shapelets,
        shapeformer_shapelet_len=args.shapeformer_shapelet_len,
        shapeformer_shapelet_stride=args.shapeformer_shapelet_stride,
        shapeformer_local_window=args.shapeformer_local_window,
        shapeformer_local_embed_dim=args.shapeformer_local_embed_dim,
        shapeformer_shape_embed_dim=args.shapeformer_shape_embed_dim,
        shapeformer_dim_ff=args.shapeformer_dim_ff,
        shapeformer_heads=args.shapeformer_heads,
        shapeformer_dropout=args.shapeformer_dropout,
        shapeformer_discovery_method=args.shapeformer_discovery_method,
        shapeformer_discovery_windows=args.shapeformer_discovery_windows,
        shapeformer_candidates_per_class_channel=args.shapeformer_candidates_per_class_channel,
        shapeformer_num_pip=args.shapeformer_num_pip,
        shapeformer_processes=args.shapeformer_processes,
        shapeformer_pisd_verbose=args.shapeformer_pisd_verbose,
    )


def evaluate_deep_model(
    features: pd.DataFrame,
    config: RunConfig,
    model_arg: str,
    refresh_cnn: bool = False,
) -> Dict[str, object]:
    resolved_model, display_model = resolve_model_alias(model_arg)
    if display_model == "shapeformer_pisd":
        config.shapeformer_discovery_method = "pisd"
    elif display_model == "shapeformer":
        config.shapeformer_discovery_method = "effect_size"

    if resolved_model in {"cnn1d", "inception_time"}:
        report = evaluate_cnn(features, config, model_name=resolved_model, refresh_cnn=refresh_cnn)
    elif resolved_model == "shapeformer":
        report = evaluate_shapeformer(features, config, refresh_cnn=refresh_cnn)
    else:
        report = evaluate(features, resolved_model, config)
    report["model"] = display_model
    report["resolved_model"] = resolved_model
    report["extra_input"] = normalize_extra_input(config.extra_input)
    return report


def single_run_result_row(
    report: Dict[str, object],
    config: RunConfig,
    group_id: int,
    repeat: int,
    seed: int,
    report_path: Path,
    duration_sec: float,
    status: str = "ok",
    error: str = "",
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "error": error,
        "group_id": group_id,
        "repeat": repeat,
        "seed": seed,
        "model": report.get("model"),
        "resolved_model": report.get("resolved_model"),
        "extra_input": normalize_extra_input(config.extra_input),
        "cnn_epochs": config.cnn_epochs,
        "cnn_patience": config.cnn_patience,
        "window_sec": config.cnn_seq_sec,
        "hop_sec": config.cnn_hop_sec,
        "overlap_pct": finite_float(100.0 * (1.0 - config.cnn_hop_sec / max(config.cnn_seq_sec, 1e-8))),
        "max_windows_fraction": finite_float(config.cnn_max_windows_fraction),
        "n_windows": report.get("n_windows"),
        "n_extra_features": report.get("n_extra_features", 0),
        "window_balanced_accuracy": report.get("window_balanced_accuracy"),
        "window_macro_f1": report.get("window_macro_f1"),
        "file_balanced_accuracy": report.get("file_balanced_accuracy"),
        "file_macro_f1": report.get("file_macro_f1"),
        "subject_balanced_accuracy": report.get("subject_balanced_accuracy"),
        "subject_macro_f1": report.get("subject_macro_f1"),
        "best_val_loss": report.get("best_validation_loss", {}).get("val_loss"),
        "best_val_loss_epoch": report.get("best_validation_loss", {}).get("epoch"),
        "best_val_accuracy": report.get("best_validation_accuracy", {}).get("val_balanced_accuracy"),
        "best_val_accuracy_epoch": report.get("best_validation_accuracy", {}).get("epoch"),
        "duration_sec": finite_float(duration_sec),
        "report_path": str(report_path),
        "learning_curve_png": report.get("learning_curve_png"),
        "learning_curve_csv": report.get("learning_curve_csv"),
    }
    return row


def build_sweep_summary(run_rows: List[Dict[str, object]]) -> pd.DataFrame:
    if not run_rows:
        return pd.DataFrame()
    df = pd.DataFrame(run_rows)
    group_cols = [
        "group_id",
        "model",
        "resolved_model",
        "extra_input",
        "cnn_epochs",
        "cnn_patience",
        "window_sec",
        "hop_sec",
        "overlap_pct",
        "max_windows_fraction",
    ]
    metric_cols = [
        "window_balanced_accuracy",
        "window_macro_f1",
        "file_balanced_accuracy",
        "file_macro_f1",
        "subject_balanced_accuracy",
        "subject_macro_f1",
        "best_val_loss",
        "best_val_accuracy",
        "duration_sec",
    ]
    rows: List[Dict[str, object]] = []
    for key, sub in df.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, key))
        row["n_completed"] = int(len(sub))
        row["n_failed"] = int((sub.get("status", "") == "failed").sum()) if "status" in sub else 0
        score = pd.to_numeric(sub["subject_macro_f1"], errors="coerce")
        valid = sub[score.notna()]
        if valid.empty:
            best = sub.iloc[0]
            row["best_single_repeat"] = int(best["repeat"])
            row["best_single_subject_macro_f1"] = None
            row["best_single_subject_balanced_accuracy"] = None
            row["best_single_file_macro_f1"] = None
            row["best_single_file_balanced_accuracy"] = None
            row["best_single_report_path"] = best["report_path"]
        else:
            best_idx = pd.to_numeric(valid["subject_macro_f1"], errors="coerce").idxmax()
            best = sub.loc[best_idx]
            row["best_single_repeat"] = int(best["repeat"])
            row["best_single_subject_macro_f1"] = finite_float(best["subject_macro_f1"])
            row["best_single_subject_balanced_accuracy"] = finite_float(best["subject_balanced_accuracy"])
            row["best_single_file_macro_f1"] = finite_float(best["file_macro_f1"])
            row["best_single_file_balanced_accuracy"] = finite_float(best["file_balanced_accuracy"])
            row["best_single_report_path"] = best["report_path"]
        for metric in metric_cols:
            vals = pd.to_numeric(sub[metric], errors="coerce")
            row[f"{metric}_mean"] = finite_float(vals.mean()) if vals.notna().any() else None
            row[f"{metric}_std"] = finite_float(vals.std(ddof=0)) if vals.notna().any() else None
            row[f"{metric}_best"] = finite_float(vals.max()) if vals.notna().any() else None
        rows.append(row)
    out = pd.DataFrame(rows)
    if "subject_macro_f1_mean" in out.columns:
        out = out.sort_values(["subject_macro_f1_mean", "subject_balanced_accuracy_mean"], ascending=False)
    return out


def write_sweep_csvs(run_rows: List[Dict[str, object]], run_dir: Path) -> None:
    pd.DataFrame(run_rows).to_csv(run_dir / "sweep_runs.csv", index=False)
    build_sweep_summary(run_rows).to_csv(run_dir / "sweep_summary.csv", index=False)


def run_auto_sweep(args: argparse.Namespace) -> None:
    base_config = config_from_args(args)
    models = [str(v) for v in parse_csv_values(args.sweep_models, str)]
    extras = [normalize_extra_input(str(v)) for v in parse_csv_values(args.sweep_extra_inputs, str)]
    epochs_list = [int(v) for v in parse_csv_values(args.sweep_epochs, int)]
    patience_list = [int(v) for v in parse_csv_values(args.sweep_patiences, int)]
    window_secs = [float(v) for v in parse_csv_values(args.sweep_window_sec, float)]
    overlap_fracs = parse_percent_values(args.sweep_overlap_pct)
    max_window_fracs = parse_percent_values(args.sweep_max_window_frac)
    repeats = max(1, int(args.sweep_repeats))

    run_dir = unique_minute_run_dir(Path(args.sweep_output_root))
    report_dir = run_dir / "reports"
    curve_dir = run_dir / "learning_curves"
    study_curve_dir = run_dir / "study_curve"
    for path in (report_dir, curve_dir, study_curve_dir):
        path.mkdir(parents=True, exist_ok=True)

    features, skipped, cache_path = build_ppi_hrv_feature_table(base_config, refresh=args.refresh_features)
    print_dataset_summary(features, skipped)

    groups: List[Dict[str, object]] = []
    group_id = 0
    for window_sec in window_secs:
        for overlap in overlap_fracs:
            hop_sec = max(1.0 / base_config.cnn_target_fs, float(window_sec) * (1.0 - float(overlap)))
            for max_frac in max_window_fracs:
                for model in models:
                    if model not in SWEEP_MODEL_CHOICES:
                        raise ValueError(f"Sweep model must be one of {SWEEP_MODEL_CHOICES}: {model}")
                    for extra_input in extras:
                        for epochs in epochs_list:
                            for patience in patience_list:
                                group_id += 1
                                groups.append(
                                    {
                                        "group_id": group_id,
                                        "window_sec": float(window_sec),
                                        "overlap": float(overlap),
                                        "hop_sec": float(hop_sec),
                                        "max_frac": float(max_frac),
                                        "model": model,
                                        "extra_input": extra_input,
                                        "epochs": int(epochs),
                                        "patience": int(patience),
                                    }
                                )

    total_runs = len(groups) * repeats
    if args.sweep_max_runs > 0:
        total_runs = min(total_runs, int(args.sweep_max_runs))
    manifest_path = run_dir / "sweep_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "total_groups": len(groups),
                "total_requested_runs": len(groups) * repeats,
                "total_planned_runs_this_invocation": total_runs,
                "base_config": asdict(base_config),
                "groups": groups,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSweep output directory: {run_dir}")
    print(f"Planned parameter groups: {len(groups)}; planned single runs: {total_runs}")
    if args.sweep_dry_run:
        print("Dry run only. No model training was started.")
        return

    run_rows: List[Dict[str, object]] = []
    completed = 0
    stop_now = False
    warmed_window_keys = set()
    progress = SweepProgress(total_runs)
    try:
        for group in groups:
            remaining_capacity = total_runs - completed
            if remaining_capacity <= 0:
                break
            _, group_display_model = resolve_model_alias(str(group["model"]))
            group_label = sanitize_tag(
                f"g{int(group['group_id']):04d}_{group_display_model}_extra-{group['extra_input']}_"
                f"ep{int(group['epochs'])}_seq{float(group['window_sec']):g}_"
                f"ov{int(round(float(group['overlap']) * 100)):03d}_"
                f"mw{int(round(float(group['max_frac']) * 100)):03d}"
            )
            progress.start_group(group_label, min(repeats, remaining_capacity))

            warm_config = RunConfig(**asdict(base_config))
            warm_config.win_sec = float(group["window_sec"])
            warm_config.hop_sec = float(group["hop_sec"])
            warm_config.cnn_seq_sec = float(group["window_sec"])
            warm_config.cnn_hop_sec = float(group["hop_sec"])
            warm_config.cnn_max_windows_fraction = float(group["max_frac"])
            warm_config.cnn_max_windows_per_file = 0
            window_key = (
                round(warm_config.cnn_seq_sec, 6),
                round(warm_config.cnn_hop_sec, 6),
                round(warm_config.cnn_max_windows_fraction, 6),
            )
            if window_key not in warmed_window_keys:
                build_cnn_window_table(features, warm_config, refresh=args.refresh_cnn_windows)
                warmed_window_keys.add(window_key)

            for repeat in range(1, repeats + 1):
                if args.sweep_max_runs > 0 and completed >= args.sweep_max_runs:
                    stop_now = True
                    break
                config = RunConfig(**asdict(warm_config))
                config.cnn_epochs = int(group["epochs"])
                config.cnn_patience = int(group["patience"])
                config.extra_input = str(group["extra_input"])
                config.seed = int(base_config.seed + (repeat - 1) * 10000)
                resolved, display_model = resolve_model_alias(str(group["model"]))
                config.shapeformer_discovery_method = "pisd" if display_model == "shapeformer_pisd" else "effect_size"
                tag = sanitize_tag(
                    f"g{int(group['group_id']):04d}_r{repeat}_{display_model}_extra-{config.extra_input}_"
                    f"ep{config.cnn_epochs}_seq{config.cnn_seq_sec:g}_ov{int(round(float(group['overlap']) * 100)):03d}_"
                    f"mw{int(round(config.cnn_max_windows_fraction * 100)):03d}_seed{config.seed}"
                )
                progress.set_current_run(f"{tag} ({completed + 1}/{total_runs})")
                run_start = time.perf_counter()
                try:
                    report = evaluate_deep_model(features, config, str(group["model"]), refresh_cnn=False)
                    report["sweep_group"] = group
                    report["sweep_repeat"] = repeat
                    report["sweep_seed"] = config.seed
                    report.update(save_learning_curve_artifacts(report, out_dir=curve_dir, filename_prefix=tag))
                    report_path = write_report(report, config, cache_path, skipped, out_dir=report_dir, filename_stem=tag)
                    curve_png = report.get("learning_curve_png")
                    if curve_png:
                        dest = study_curve_dir / f"{tag}_learning_curve.png"
                        shutil.copy2(str(curve_png), dest)
                        report["study_curve_png"] = str(dest)
                        report_path = write_report(report, config, cache_path, skipped, out_dir=report_dir, filename_stem=tag)
                    duration_sec = time.perf_counter() - run_start
                    run_rows.append(
                        single_run_result_row(
                            report,
                            config,
                            group_id=int(group["group_id"]),
                            repeat=repeat,
                            seed=config.seed,
                            report_path=report_path,
                            duration_sec=duration_sec,
                        )
                    )
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    duration_sec = time.perf_counter() - run_start
                    error_report = {
                        "model": display_model,
                        "resolved_model": resolved,
                        "extra_input": normalize_extra_input(config.extra_input),
                        "sweep_group": group,
                        "sweep_repeat": repeat,
                        "sweep_seed": config.seed,
                        "error": str(exc),
                    }
                    report_path = write_report(
                        error_report,
                        config,
                        cache_path,
                        skipped,
                        out_dir=report_dir,
                        filename_stem=f"{tag}_FAILED",
                    )
                    run_rows.append(
                        single_run_result_row(
                            error_report,
                            config,
                            group_id=int(group["group_id"]),
                            repeat=repeat,
                            seed=config.seed,
                            report_path=report_path,
                            duration_sec=duration_sec,
                            status="failed",
                            error=str(exc),
                        )
                    )
                    progress.write(f"[sweep warning] failed {tag}: {exc}")
                write_sweep_csvs(run_rows, run_dir)
                completed += 1
                progress.finish_run(tag)
            progress.close_group()
            if stop_now:
                break
    finally:
        progress.close()

    print(f"\nSweep finished. Completed runs: {completed}/{total_runs}")
    print(f"Run details CSV: {run_dir / 'sweep_runs.csv'}")
    print(f"Summary CSV: {run_dir / 'sweep_summary.csv'}")
    print(f"Learning curve copies: {study_curve_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate a 3-class frailty-status classifier from PPG+IMU CSV files.")
    parser.add_argument("--data-root", default="PPG_Testing_05_01_2026")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="logreg_l2")
    parser.add_argument("--fs", type=float, default=400.0)
    parser.add_argument("--win-sec", type=float, default=10.0)
    parser.add_argument("--hop-sec", type=float, default=5.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cnn-target-fs", type=float, default=64.0)
    parser.add_argument("--cnn-seq-sec", type=float, default=30.0)
    parser.add_argument("--cnn-hop-sec", type=float, default=30.0)
    parser.add_argument("--cnn-max-windows-per-file", type=int, default=6)
    parser.add_argument("--cnn-max-windows-fraction", type=float, default=0.0)
    parser.add_argument("--cnn-epochs", type=int, default=8)
    parser.add_argument("--cnn-batch-size", type=int, default=32)
    parser.add_argument("--cnn-lr", type=float, default=1e-3)
    parser.add_argument("--cnn-patience", type=int, default=3)
    parser.add_argument("--cnn-num-workers", type=int, default=0)
    parser.add_argument("--cnn-weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--cnn-dropout",
        type=float,
        default=-1.0,
        help="Deep-model dropout override. -1 keeps each architecture's historical default.",
    )
    parser.add_argument("--cnn-label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--cnn-use-final-epoch",
        action="store_true",
        help="Train for the full epoch limit and keep final-epoch weights instead of validation-best weights.",
    )
    parser.add_argument("--role-mode", choices=ROLE_MODE_CHOICES, default="static_only")
    parser.add_argument("--extra-input", choices=EXTRA_INPUT_CHOICES, default="0")
    parser.add_argument("--shapeformer-num-shapelets", type=int, default=3)
    parser.add_argument("--shapeformer-shapelet-len", type=int, default=128)
    parser.add_argument("--shapeformer-shapelet-stride", type=int, default=64)
    parser.add_argument("--shapeformer-local-window", type=int, default=64)
    parser.add_argument("--shapeformer-local-embed-dim", type=int, default=48)
    parser.add_argument("--shapeformer-shape-embed-dim", type=int, default=128)
    parser.add_argument("--shapeformer-dim-ff", type=int, default=256)
    parser.add_argument("--shapeformer-heads", type=int, default=4)
    parser.add_argument("--shapeformer-dropout", type=float, default=0.30)
    parser.add_argument("--shapeformer-discovery-method", choices=("effect_size", "pisd"), default="effect_size")
    parser.add_argument("--shapeformer-discovery-windows", type=int, default=180)
    parser.add_argument("--shapeformer-candidates-per-class-channel", type=int, default=8)
    parser.add_argument("--shapeformer-num-pip", type=float, default=0.2)
    parser.add_argument("--shapeformer-processes", type=int, default=1)
    parser.add_argument(
        "--shapeformer-pisd-verbose",
        action="store_true",
        help="Show original PISD extract/discovery debug prints instead of suppressing them.",
    )
    parser.add_argument("--refresh-features", action="store_true")
    parser.add_argument("--refresh-cnn-windows", action="store_true")
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--auto-sweep", action="store_true", help="Run the built-in hyperparameter sweep and write incremental reports/CSVs.")
    parser.add_argument("--sweep-models", default="cnn,inceptiontime,shapeformer_pisd,shapeformer")
    parser.add_argument("--sweep-extra-inputs", default="0,PPI,HRV")
    parser.add_argument("--sweep-epochs", default="20,50")
    parser.add_argument("--sweep-patiences", default="10")
    parser.add_argument("--sweep-window-sec", default="15,10,5,2")
    parser.add_argument("--sweep-overlap-pct", default="80,50,30,0")
    parser.add_argument("--sweep-max-window-frac", default="90,50")
    parser.add_argument("--sweep-repeats", type=int, default=3)
    parser.add_argument("--sweep-output-root", default="results_frailty3")
    parser.add_argument("--sweep-max-runs", type=int, default=0, help="Optional cap for debugging; 0 means run the full sweep.")
    parser.add_argument("--sweep-dry-run", action="store_true", help="Create the timestamped sweep manifest without training models.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.auto_sweep:
        run_auto_sweep(args)
        return

    config = config_from_args(args)
    resolved_model, display_model = resolve_model_alias(args.model)
    if display_model == "shapeformer_pisd":
        config.shapeformer_discovery_method = "pisd"
    features, skipped, cache_path = build_feature_table(config, refresh=args.refresh_features)
    print_dataset_summary(features, skipped)

    if resolved_model in {"cnn1d", "inception_time"}:
        report = evaluate_cnn(features, config, model_name=resolved_model, refresh_cnn=args.refresh_cnn_windows)
    elif resolved_model == "shapeformer":
        report = evaluate_shapeformer(features, config, refresh_cnn=args.refresh_cnn_windows)
    else:
        report = evaluate(features, resolved_model, config)
    report["model"] = display_model
    report["resolved_model"] = resolved_model
    report.update(save_learning_curve_artifacts(report))
    report_path = write_report(report, config, cache_path, skipped)
    print("\nCross-validation summary")
    print(f"model: {display_model}")
    if "window_balanced_accuracy" in report:
        print(f"window balanced accuracy: {report['window_balanced_accuracy']:.3f}")
        print(f"window macro F1: {report['window_macro_f1']:.3f}")
    print(f"file balanced accuracy: {report['file_balanced_accuracy']:.3f}")
    print(f"file macro F1: {report['file_macro_f1']:.3f}")
    print(f"subject balanced accuracy: {report['subject_balanced_accuracy']:.3f}")
    print(f"subject macro F1: {report['subject_macro_f1']:.3f}")
    print("subject confusion matrix rows=true cols=pred, order=pre_frail, robust_non_frail, young")
    print(np.array(report["subject_confusion_matrix"]))
    print(f"report: {report_path}")

    if not args.no_save_model:
        if resolved_model in {"cnn1d", "inception_time"}:
            model_path = save_final_cnn_model(features, config, report, model_name=resolved_model, refresh_cnn=False)
        elif resolved_model == "shapeformer":
            model_path = save_final_shapeformer_model(features, config, report, refresh_cnn=False)
        else:
            model_path = save_final_model(features, resolved_model, config, report)
        print(f"saved model: {model_path}")


if __name__ == "__main__":
    main()
