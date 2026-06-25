
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pttppg_pipeline_v7_3_noleak_viz.py
============================================================

v7.3 (NO-LEAK + VIZ + FEATURE-THRESHOLD DETECTOR + STFT-MASK DENOISER + ONNX EXPORT)

This script implements the user's revised specification:

A) Detector (motion segment detection)
-------------------------------------
- Target label: activity-based binary classification (sit=0, walk/run=1).
- Inputs: pleth_1 (red), pleth_2 (IR), plus IMU-derived features.
- Procedure:
  1) Compute per-window PPG feature matrix F_ppg and IMU feature matrix F_imu.
  2) Learn per-feature thresholds τ_ppg[j] on F_ppg to predict activity (T1: maximize balanced accuracy).
  3) Find the best lag n ∈ [-5s, +5s] that maximizes cross-modal association between F_ppg(t) and F_imu(t+n)
     using mean |Spearman ρ| across matched feature indices.
  4) Learn per-feature thresholds τ_imu[j] on lag-aligned IMU features to predict activity (T1).
  5) Inference rule: motion = (IMU_feature > τ_imu) OR (PPG_feature > τ_ppg) (default OR; AND also reported).
- Strict no-leak:
  - Holdout subjects are split first and never used for threshold/lag selection.
  - CV is performed only within training subjects.

B) Denoiser (walk/run separate, STFT magnitude mask, subject-specific phase compensation)
---------------------------------------------------------------------------------------
- Inputs (raw time-series channels): 8ch
    1–2: pleth_1, pleth_2
    3–5: a_dyn_x, a_dyn_y, a_dyn_z  (linear acceleration in m/s^2 after gravity removal)
    6–8: gyro_x, gyro_y, gyro_z    (rad/s; source gyro is deg/s, converted)
- Additional per-window features: Time stats + Bandpowers + Spectral shape computed from the above;
  they are broadcast (repeated) over the STFT time-frequency grid as extra channels.
- Network outputs an STFT magnitude mask M ∈ (0,1). Clean spectrum keeps the *original noisy phase*:
    Y_clean = (M ⊙ |Y_noisy|) * exp(j * angle(Y_noisy))
- Supervision:
  - ECG peaks come from CSV column "peaks". ECG is NOT an input.
  - PPG peaks are defined as systolic maxima (highest peak per beat).
  - Subject-specific median delay Δ0 (PPG vs ECG) is estimated from SIT segments.
  - For each subject and activity (walk/run), a learnable scalar a ∈ (0.5, 1.5) scales Δ0.
    ECG peaks are shifted by round(a * Δ0) samples; PPG peaks are not moved.
  - Primary loss: differentiable soft-peak alignment between denoised PPG and shifted ECG peaks.
  - Secondary loss: shape similarity to a SIT-derived smoothed template (lower weight).

Artifacts & CPU-only reuse:
---------------------------
- Detector artifact: JSON with feature schema, τ_ppg, τ_imu, lag n, normalization stats.
- Denoiser artifact: ONNX model (mask net), plus JSON with STFT params, feature schema,
  normalization stats, Δ0 table, and subject-wise a table for walk/run.
  Inference can be run CPU-only with onnxruntime + numpy/scipy STFT/ISTFT.

Notes:
- This script depends on numpy, pandas, scipy, sklearn, torch, matplotlib, tqdm.
- For IMU gravity removal we use a pragmatic approach:
  *If a sensor fusion filter is available in funcs.py, you can wire it in.*
  By default we use a low-pass estimate of gravity on accelerometer and subtract it.
  This is deterministic and export-friendly.

Author: generated/modified for Kaifeng Xu, Jan 2026
"""
from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from tqdm import tqdm

# ML
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# SciPy signal/FFT
from scipy import signal
from scipy.stats import spearmanr

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Plotting
import matplotlib.pyplot as plt


# ---------------------------
# Optional project utilities
# ---------------------------
# funcs.py is expected to contain signal preprocessing and peak detection utilities,
# including Aboy++ style peak detection (aboypp_peak_hr) and helpers.
# We import defensively to keep this script runnable.
try:
    import funcs as project_funcs  # type: ignore
except Exception:
    project_funcs = None


# ===========================
# Configuration data classes
# ===========================
@dataclass
class STFTConfig:
    n_fft: int = 256
    hop_length: int = 64
    win_length: int = 256
    window: str = "hann"

@dataclass
class WinConfig:
    fs: float = 500.0
    win_sec: float = 6.0
    hop_sec: float = 1.0
    lag_min_sec: float = -5.0
    lag_max_sec: float = 5.0


# ===========================
# Basic helpers
# ===========================
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def zscore_np(x: np.ndarray) -> np.ndarray:
    m = float(np.mean(x))
    s = float(np.std(x) + 1e-8)
    return (x - m) / s

def split_windows(N: int, fs: float, win_sec: float, hop_sec: float):
    w = int(round(win_sec * fs))
    h = int(round(hop_sec * fs))
    if w <= 0:
        raise ValueError("win_sec too small")
    if h <= 0:
        raise ValueError("hop_sec too small")
    for s in range(0, max(1, N - w + 1), h):
        yield s, s + w

def activity_from_filename(stem: str) -> str:
    # e.g., s1_walk -> walk
    parts = stem.split("_")
    return parts[-1].lower() if parts else ""

def subject_from_filename(stem: str) -> str:
    # e.g., s1_walk -> s1
    parts = stem.split("_")
    return parts[0].lower() if parts else stem.lower()

def safe_float_col(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32)

def deg2rad(x: np.ndarray) -> np.ndarray:
    return x * (np.pi / 180.0)

def acc_g_to_ms2(x: np.ndarray) -> np.ndarray:
    return x * 9.80665

def lowpass_gravity(acc_ms2: np.ndarray, fs: float, cutoff_hz: float = 0.3, order: int = 2) -> np.ndarray:
    """Estimate gravity vector by low-pass filtering each axis."""
    b, a = signal.butter(order, cutoff_hz / (fs / 2), btype="low")
    g = signal.filtfilt(b, a, acc_ms2, axis=0)
    return g

def linear_acceleration(acc_g: np.ndarray, fs: float) -> np.ndarray:
    """Remove gravity component from accelerometer (input in g). Output m/s^2."""
    acc_ms2 = acc_g_to_ms2(acc_g)
    gvec = lowpass_gravity(acc_ms2, fs=fs)
    return acc_ms2 - gvec

def magn(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt(np.sum(x * x, axis=axis))

def jerk_mag(acc_dyn: np.ndarray, fs: float) -> np.ndarray:
    # derivative of dynamic acceleration magnitude
    j = np.gradient(acc_dyn, axis=0) * fs
    return magn(j, axis=1)

def robust_iqr(x: np.ndarray) -> float:
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)

def spectral_entropy(psd: np.ndarray) -> float:
    p = psd.astype(np.float64)
    p = p / (np.sum(p) + 1e-12)
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))

def dominant_freq(f: np.ndarray, psd: np.ndarray, fmin: float = 0.0, fmax: float = 8.0) -> float:
    m = (f >= fmin) & (f <= fmax)
    if not np.any(m):
        return float("nan")
    i = int(np.argmax(psd[m]))
    return float(f[m][i])

def bandpowers_welch(x: np.ndarray, fs: float, bands=((0.1,0.5),(0.5,3.0),(3.0,8.0))) -> List[float]:
    f, pxx = signal.welch(x, fs=fs, nperseg=min(len(x), int(fs*2)))
    out = []
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        out.append(float(np.trapz(pxx[m], f[m])) if np.any(m) else 0.0)
    return out

def time_stats(x: np.ndarray) -> List[float]:
    return [
        float(np.mean(x)),
        float(np.std(x) + 1e-12),
        float(np.median(x)),
        robust_iqr(x),
        float(np.sqrt(np.mean(x*x) + 1e-12)),
    ]

# ===========================
# Feature extraction
# ===========================
PPG_FEAT_GROUPS = ["ppg"]
IMU_FEAT_GROUPS = ["accmag", "gyromag", "jerkmag"]

def ppg_features(ppg: np.ndarray, fs: float) -> Tuple[np.ndarray, List[str]]:
    """Compute PPG features for one window; ppg is 1D array (pleth_2 recommended)."""
    feats: List[float] = []
    names: List[str] = []
    # Time stats
    ts = time_stats(ppg)
    feats += ts
    names += [f"PPG_ts_{k}" for k in ["mean","std","med","iqr","rms"]]
    # Bandpowers
    bp = bandpowers_welch(ppg, fs=fs)
    feats += bp
    names += [f"PPG_bp_{b}" for b in ["0.1_0.5","0.5_3","3_8"]]
    # Spectral shape
    f, pxx = signal.welch(ppg, fs=fs, nperseg=min(len(ppg), int(fs*2)))
    feats += [spectral_entropy(pxx), dominant_freq(f, pxx, 0.1, 8.0)]
    names += ["PPG_spec_entropy", "PPG_dom_freq"]
    return np.asarray(feats, dtype=np.float32), names

def imu_features(acc_dyn_xyz: np.ndarray, gyro_rads_xyz: np.ndarray, fs: float) -> Tuple[np.ndarray, List[str]]:
    """
    Compute IMU feature vector from dynamic acceleration and gyro (windowed).
    acc_dyn_xyz: (T,3) m/s^2
    gyro_rads_xyz: (T,3) rad/s
    """
    feats: List[float] = []
    names: List[str] = []

    accmag = magn(acc_dyn_xyz, axis=1)
    gyromag = magn(gyro_rads_xyz, axis=1)
    jmag = jerk_mag(acc_dyn_xyz, fs=fs)

    # Time stats
    for sig, prefix in [(accmag,"AccMag"), (gyromag,"GyroMag"), (jmag,"JerkMag")]:
        ts = time_stats(sig)
        feats += ts
        names += [f"{prefix}_ts_{k}" for k in ["mean","std","med","iqr","rms"]]

    # Bandpowers: AccMag, GyroMag only (per your spec)
    for sig, prefix in [(accmag,"AccMag"), (gyromag,"GyroMag")]:
        bp = bandpowers_welch(sig, fs=fs)
        feats += bp
        names += [f"{prefix}_bp_{b}" for b in ["0.1_0.5","0.5_3","3_8"]]

    # Spectral shape
    for sig, prefix in [(accmag,"AccMag"), (gyromag,"GyroMag"), (jmag,"JerkMag")]:
        f, pxx = signal.welch(sig, fs=fs, nperseg=min(len(sig), int(fs*2)))
        feats += [spectral_entropy(pxx), dominant_freq(f, pxx, 0.1, 8.0)]
        names += [f"{prefix}_spec_entropy", f"{prefix}_dom_freq"]

    return np.asarray(feats, dtype=np.float32), names

def window_feature_pack(
    pleth1: np.ndarray,
    pleth2: np.ndarray,
    acc_dyn_xyz: np.ndarray,
    gyro_rads_xyz: np.ndarray,
    fs: float
) -> Tuple[np.ndarray, List[str]]:
    """Features used for both detector & as auxiliary channels for denoiser."""
    # PPG features computed on pleth2 (IR) by default; pleth1 can be added later.
    f_ppg, n_ppg = ppg_features(pleth2, fs=fs)
    f_imu, n_imu = imu_features(acc_dyn_xyz, gyro_rads_xyz, fs=fs)
    feats = np.concatenate([f_ppg, f_imu], axis=0)
    names = n_ppg + n_imu
    return feats.astype(np.float32), names


# ===========================
# CSV loading
# ===========================
CAND = {
    "time": ["time", "Time", "t"],
    "ecg": ["ecg", "ECG"],
    "peaks": ["peaks", "Rpeaks", "r_peaks"],
    "pleth_1": ["pleth_1"],
    "pleth_2": ["pleth_2"],
    "a_x": ["a_x", "AX", "accX"],
    "a_y": ["a_y", "AY", "accY"],
    "a_z": ["a_z", "AZ", "accZ"],
    "g_x": ["g_x", "GX", "gyroX"],
    "g_y": ["g_y", "GY", "gyroY"],
    "g_z": ["g_z", "GZ", "gyroZ"],
}
def _pick(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None

def load_physionet_csv(root: Path, fs: float) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load PhysioNet PTT-PPG CSV format.

    Returns:
        sub2recs: dict(subject_id -> list of records)
    Each record contains numpy arrays for channels + metadata:
        rec["_file"], rec["_activity"], rec["_subject"]
    """
    csv_dir = root / "csv"
    if not csv_dir.exists():
        csv_dir = root / "files" / "pulse-transit-time-ppg" / "1.1.0" / "csv"
    paths = sorted([p for p in csv_dir.glob("s*_*.csv") if p.name != "subjects_info.csv"])
    sub2recs: Dict[str, List[Dict[str, Any]]] = {}
    for p in paths:
        df = pd.read_csv(p)
        m = {k: _pick(df, v) for k, v in CAND.items()}
        rec: Dict[str, Any] = {}
        for k in CAND.keys():
            col = m.get(k)
            if col is None:
                continue
            if k == "time":
                t = pd.to_datetime(df[col], errors="coerce")
                if t.notna().all():
                    rec["time"] = (t - t.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float32)
                else:
                    rec["time"] = (np.arange(len(df), dtype=np.float32) / float(fs))
            else:
                rec[k] = safe_float_col(df[col])
        if "time" not in rec:
            rec["time"] = (np.arange(len(df), dtype=np.float32) / float(fs))

        stem = p.stem
        sid = subject_from_filename(stem)
        act = activity_from_filename(stem)
        rec["_file"] = p.name
        rec["_subject"] = sid
        rec["_activity"] = act
        sub2recs.setdefault(sid, []).append(rec)
    return sub2recs


# ===========================
# Subject-wise split (NO-LEAK)
# ===========================
def split_subjects(sub_ids: List[str], train_size: float, seed: int = 42) -> Tuple[List[str], List[str]]:
    rng = np.random.RandomState(seed)
    sub_ids = sorted(sub_ids)
    rng.shuffle(sub_ids)
    n_train = int(round(len(sub_ids) * train_size))
    train_sub = sub_ids[:n_train]
    hold_sub = sub_ids[n_train:]
    return train_sub, hold_sub

def build_group_kfold(train_records: List[Dict[str,Any]], n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray([r["_subject"] for r in train_records])
    idx = np.arange(len(train_records))
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    return list(gkf.split(idx, groups=groups))


# ===========================
# Detector threshold learning
# ===========================
def best_threshold_per_feature(x: np.ndarray, y: np.ndarray) -> Tuple[float, int, float]:
    """
    For a single feature x and binary label y (0/1),
    find threshold τ and direction d in {+1,-1} maximizing balanced accuracy.

    Returns:
        (tau, direction, best_bal_acc)
    direction=+1 means predict 1 if x > tau
    direction=-1 means predict 1 if x < tau
    """
    # candidate thresholds: percentiles
    qs = np.unique(np.quantile(x[~np.isnan(x)], np.linspace(0.01, 0.99, 99))) if np.any(~np.isnan(x)) else np.array([0.0])
    best_tau, best_dir, best_ba = float(qs[0]), 1, -1.0
    for tau in qs:
        for d in (1, -1):
            pred = (x > tau).astype(int) if d == 1 else (x < tau).astype(int)
            ba = metrics.balanced_accuracy_score(y, pred)
            if ba > best_ba:
                best_tau, best_dir, best_ba = float(tau), int(d), float(ba)
    return best_tau, best_dir, best_ba

def apply_threshold_matrix(X: np.ndarray, taus: np.ndarray, dirs: np.ndarray, rule: str = "any") -> np.ndarray:
    """
    Apply per-feature thresholds.
    - rule="any": motion if ANY feature votes 1
    - rule="all": motion if ALL features vote 1
    """
    votes = []
    for j in range(X.shape[1]):
        x = X[:, j]
        tau = taus[j]
        d = dirs[j]
        v = (x > tau) if d == 1 else (x < tau)
        votes.append(v.astype(bool))
    V = np.stack(votes, axis=1)
    if rule == "all":
        return V.all(axis=1).astype(int)
    return V.any(axis=1).astype(int)

def mean_abs_spearman(A: np.ndarray, B: np.ndarray) -> float:
    """
    Mean absolute Spearman correlation across matched columns (min dim).
    """
    m = min(A.shape[1], B.shape[1])
    vals = []
    for j in range(m):
        a = A[:, j]
        b = B[:, j]
        mask = ~np.isnan(a) & ~np.isnan(b)
        if mask.sum() < 10:
            continue
        rho, _ = spearmanr(a[mask], b[mask])
        if np.isfinite(rho):
            vals.append(abs(float(rho)))
    return float(np.mean(vals)) if vals else float("nan")

def lag_search_ppg_imu(
    F_ppg: np.ndarray,
    F_imu: np.ndarray,
    lags_samples: List[int]
) -> Tuple[int, Dict[int, float]]:
    """
    Search best lag (in samples of hop, not signal samples) maximizing association.
    Here lags_samples is in window-steps (i.e., hop units), not raw samples.
    """
    scores: Dict[int, float] = {}
    T = min(len(F_ppg), len(F_imu))
    for lag in lags_samples:
        if lag >= 0:
            p = F_ppg[:T - lag]
            i = F_imu[lag:T]
        else:
            k = -lag
            p = F_ppg[k:T]
            i = F_imu[:T - k]
        if len(p) < 20:
            scores[lag] = float("nan")
            continue
        scores[lag] = mean_abs_spearman(p, i)
    # pick max score, tie -> smallest abs lag
    valid = [(lag, sc) for lag, sc in scores.items() if np.isfinite(sc)]
    if not valid:
        return 0, scores
    valid.sort(key=lambda t: (-(t[1]), abs(t[0])))
    return int(valid[0][0]), scores


# ===========================
# Denoiser: mask network (STFT domain)
# ===========================
class MaskNet(nn.Module):
    """
    A compact 2D-conv mask estimator on magnitude spectrogram.

    Inputs:
        X: (B, C, F, TT) float
    Output:
        M: (B, 1, F, TT) in (0,1)
    """
    def __init__(self, in_ch: int, base: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(base, base*2, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base*2, base*2, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(base*2, base*4, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base*4, base*2, 3, padding=1), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base*2, base, 3, padding=1), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

def freq_smooth_reg(mask: torch.Tensor) -> torch.Tensor:
    """
    Frequency-direction smoothness regularizer for mask.
    mask: (B,1,F,T)
    """
    df = mask[:, :, 1:, :] - mask[:, :, :-1, :]
    return (df * df).mean()

def soft_peak_map(y: torch.Tensor, fs: float, k: float = 10.0) -> torch.Tensor:
    """
    Create a differentiable proxy for peak positions.
    Approach: emphasize positive curvature + amplitude above local mean.
    Returns a [0,1] map of shape (B, T).
    """
    # y: (B,1,T)
    y1 = y.squeeze(1)
    # local mean via avgpool
    win = max(5, int(round(0.15 * fs)))  # ~150ms
    if win % 2 == 0:
        win += 1
    pad = win // 2
    local = F.avg_pool1d(y1.unsqueeze(1), kernel_size=win, stride=1, padding=pad).squeeze(1)
    high = y1 - local
    # curvature (second derivative)
    d1 = y1[:, 1:] - y1[:, :-1]
    d2 = d1[:, 1:] - d1[:, :-1]
    d2 = F.pad(d2, (1, 1), value=0.0)
    peakness = high * F.relu(d2)
    # normalize per-sample
    peakness = peakness / (peakness.abs().amax(dim=1, keepdim=True) + 1e-6)
    return torch.sigmoid(k * peakness)


# ===========================
# Datasets
# ===========================
class WindowIndex:
    __slots__ = ("rid", "s", "e")
    def __init__(self, rid: int, s: int, e: int):
        self.rid, self.s, self.e = rid, s, e

class DenoiseDataset(Dataset):
    """
    Provides:
      - noisy time-series: pleth1, pleth2, acc_dyn_xyz, gyro_rads_xyz  -> (8,T)
      - auxiliary window features -> (D,)
      - ECG peaks vector -> (T,) float 0/1
      - subject id -> string
      - activity -> string
    """
    def __init__(self, records: List[Dict[str,Any]], win_cfg: WinConfig):
        self.records = records
        self.cfg = win_cfg
        self.index: List[WindowIndex] = []
        for rid, r in enumerate(records):
            N = len(r["pleth_2"])
            for s, e in split_windows(N, win_cfg.fs, win_cfg.win_sec, win_cfg.hop_sec):
                self.index.append(WindowIndex(rid, s, e))

        # Feature schema built on first sample
        self._feat_names: Optional[List[str]] = None

    def __len__(self): return len(self.index)

    @property
    def feat_names(self) -> List[str]:
        if self._feat_names is None:
            # build once
            wi = self.index[0]
            r = self.records[wi.rid]
            s, e = wi.s, wi.e
            pleth1 = r["pleth_1"][s:e]
            pleth2 = r["pleth_2"][s:e]
            acc_g = np.column_stack([r["a_x"][s:e], r["a_y"][s:e], r["a_z"][s:e]]).astype(np.float32)
            gyro_deg = np.column_stack([r["g_x"][s:e], r["g_y"][s:e], r["g_z"][s:e]]).astype(np.float32)
            acc_dyn = linear_acceleration(acc_g, fs=self.cfg.fs).astype(np.float32)
            gyro_rad = deg2rad(gyro_deg).astype(np.float32)
            feats, names = window_feature_pack(pleth1, pleth2, acc_dyn, gyro_rad, fs=self.cfg.fs)
            self._feat_names = names
        return self._feat_names

    def __getitem__(self, idx: int):
        wi = self.index[idx]
        r = self.records[wi.rid]
        s, e = wi.s, wi.e

        pleth1 = r["pleth_1"][s:e].astype(np.float32)
        pleth2 = r["pleth_2"][s:e].astype(np.float32)

        acc_g = np.column_stack([r["a_x"][s:e], r["a_y"][s:e], r["a_z"][s:e]]).astype(np.float32)
        gyro_deg = np.column_stack([r["g_x"][s:e], r["g_y"][s:e], r["g_z"][s:e]]).astype(np.float32)

        acc_dyn = linear_acceleration(acc_g, fs=self.cfg.fs).astype(np.float32)
        gyro_rad = deg2rad(gyro_deg).astype(np.float32)

        # raw 8ch
        x_raw = np.concatenate([
            pleth1[None, :], pleth2[None, :],
            acc_dyn.T,
            gyro_rad.T
        ], axis=0).astype(np.float32)

        # auxiliary features
        feats, names = window_feature_pack(pleth1, pleth2, acc_dyn, gyro_rad, fs=self.cfg.fs)
        if self._feat_names is None:
            self._feat_names = names

        # ECG peaks (label only)
        peaks = r.get("peaks", np.zeros_like(pleth2, dtype=np.float32))[s:e].astype(np.float32)
        # make sure it's 0/1
        peaks = (peaks > 0.5).astype(np.float32)

        sid = r["_subject"]
        act = r["_activity"]
        return torch.from_numpy(x_raw), torch.from_numpy(feats), torch.from_numpy(peaks), sid, act


# ===========================
# SIT delay estimation (Δ0)
# ===========================
def estimate_subject_delay_sit(records: List[Dict[str,Any]], fs: float) -> float:
    """
    Estimate subject-specific median delay Δ0 in samples between ECG peaks and PPG peaks on SIT records.
    Δ0 = median( t_ppg_peak - t_ecg_peak ) in samples (positive means PPG peak occurs after ECG peak).
    """
    delays = []
    for r in records:
        if r["_activity"] != "sit":
            continue
        if "peaks" not in r:
            continue
        ppg = r["pleth_2"].astype(np.float32)
        ecg_peaks = np.where(r["peaks"] > 0.5)[0].astype(int)

        if project_funcs is not None and hasattr(project_funcs, "aboypp_peak_hr"):
            try:
                out = project_funcs.aboypp_peak_hr(ppg, fs=fs)  # type: ignore
                ppg_peaks = np.asarray(out.get("peaks_all", []), dtype=int)
            except Exception:
                ppg_peaks = np.array([], dtype=int)
        else:
            # fallback: simple find_peaks
            ppg_f = signal.filtfilt(*signal.butter(2, [0.5/(fs/2), 8/(fs/2)], btype="band"), ppg)
            ppg_peaks, _ = signal.find_peaks(ppg_f, distance=int(fs*0.3), prominence=np.std(ppg_f)*0.3)

        if len(ppg_peaks) < 10 or len(ecg_peaks) < 10:
            continue

        # match each ECG peak to nearest following PPG peak
        for rp in ecg_peaks:
            j = np.searchsorted(ppg_peaks, rp)
            if j < len(ppg_peaks):
                delays.append(ppg_peaks[j] - rp)

    if not delays:
        return float(round(0.2 * fs))  # fallback 200ms
    return float(np.median(np.asarray(delays, dtype=np.float32)))


# ===========================
# Denoiser training
# ===========================
class SubjectA(nn.Module):
    """
    Learnable per-subject scalar a in (0.5, 1.5), per activity (walk/run).
    Implemented as two embeddings.
    """
    def __init__(self, subject_ids: List[str]):
        super().__init__()
        self.sub_ids = list(subject_ids)
        self.id2idx = {sid: i for i, sid in enumerate(self.sub_ids)}
        n = len(self.sub_ids)
        self.emb_walk = nn.Embedding(n, 1)
        self.emb_run = nn.Embedding(n, 1)
        nn.init.zeros_(self.emb_walk.weight)
        nn.init.zeros_(self.emb_run.weight)

    def forward(self, sid_idx: torch.Tensor, activity: str) -> torch.Tensor:
        raw = self.emb_walk(sid_idx) if activity == "walk" else self.emb_run(sid_idx)
        # map to (0.5,1.5)
        return 0.5 + torch.sigmoid(raw) * 1.0

    def sid_to_index(self, sids: List[str], device: torch.device) -> torch.Tensor:
        idx = [self.id2idx[s] for s in sids]
        return torch.tensor(idx, dtype=torch.long, device=device)

def torch_stft(x: torch.Tensor, cfg: STFTConfig) -> torch.Tensor:
    """x: (B,T) -> (B,F,Tt) complex"""
    win = torch.hann_window(cfg.win_length, device=x.device)
    return torch.stft(
        x,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=win,
        return_complex=True,
        center=True,
    )

def torch_istft(X: torch.Tensor, cfg: STFTConfig, length: int) -> torch.Tensor:
    """X: (B,F,Tt) complex -> (B,T)"""
    win = torch.hann_window(cfg.win_length, device=X.device)
    return torch.istft(
        X,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=win,
        length=length,
        center=True,
    )

def build_aux_feature_maps(feats: torch.Tensor, F: int, TT: int) -> torch.Tensor:
    """
    Broadcast per-window feature vector feats (B,D) to (B,D,F,TT).
    """
    return feats[:, :, None, None].repeat(1, 1, F, TT)

def shift_peaks(peaks: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """
    Shift a batch of peak vectors by integer shifts (per-sample).
    peaks: (B,T) in {0,1}
    shift: (B,) integer (can be negative)
    """
    B, T = peaks.shape
    out = torch.zeros_like(peaks)
    for b in range(B):
        k = int(shift[b].item())
        if k == 0:
            out[b] = peaks[b]
        elif k > 0:
            out[b, k:] = peaks[b, :T-k]
        else:
            kk = -k
            out[b, :T-kk] = peaks[b, kk:]
    return out

def bce_peak_loss(soft_map: torch.Tensor, peaks: torch.Tensor) -> torch.Tensor:
    """soft_map: (B,T) in (0,1); peaks: (B,T) in {0,1}"""
    return F.binary_cross_entropy(soft_map, peaks)

def make_sit_template(records: List[Dict[str,Any]], fs: float, win_cfg: WinConfig) -> np.ndarray:
    """
    Build a simple SIT template by band-pass filtering and taking a median window.
    Returns a 1D template of length win.
    """
    w = int(round(win_cfg.win_sec * fs))
    segs = []
    for r in records:
        if r["_activity"] != "sit":
            continue
        p = r["pleth_2"].astype(np.float32)
        if len(p) < w:
            continue
        # pick a mid segment
        s = max(0, (len(p) - w)//2)
        seg = p[s:s+w]
        b, a = signal.butter(2, [0.5/(fs/2), 8/(fs/2)], btype="band")
        seg = signal.filtfilt(b, a, seg)
        seg = zscore_np(seg)
        segs.append(seg)
    if not segs:
        return np.zeros(w, dtype=np.float32)
    return np.median(np.stack(segs, axis=0), axis=0).astype(np.float32)

def train_denoiser_one_activity(
    activity: str,
    train_records: List[Dict[str,Any]],
    val_records: List[Dict[str,Any]],
    subject_ids: List[str],
    delta0_by_subject: Dict[str, float],
    win_cfg: WinConfig,
    stft_cfg: STFTConfig,
    outdir: Path,
    epochs: int = 10,
    lr: float = 1e-3,
    lam_shape: float = 0.05,
    lam_smooth: float = 0.05,
    batch_size: int = 16,
    device: Optional[str] = None,
) -> Tuple[MaskNet, SubjectA, Dict[str,Any]]:
    """
    Train a denoiser for a single activity (walk or run).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)

    # Dataset filtered by activity
    tr = [r for r in train_records if r["_activity"] == activity]
    va = [r for r in val_records if r["_activity"] == activity]
    if not tr or not va:
        raise RuntimeError(f"Not enough {activity} records for denoiser training.")

    ds_tr = DenoiseDataset(tr, win_cfg)
    ds_va = DenoiseDataset(va, win_cfg)

    feat_names = ds_tr.feat_names
    D = len(feat_names)

    # Build a SIT template from training SIT only (no-leak)
    sit_template = make_sit_template(train_records, fs=win_cfg.fs, win_cfg=win_cfg)
    sit_template_t = torch.from_numpy(sit_template[None, None, :]).to(dev)

    # Determine spectrogram dimensions from one sample
    x_raw0, feats0, peaks0, sid0, act0 = ds_tr[0]
    T = x_raw0.shape[1]
    Y0 = torch_stft(x_raw0[1].unsqueeze(0).to(dev), stft_cfg)  # pleth_2
    F_bins, TT_bins = Y0.shape[1], Y0.shape[2]

    # Model input channels:
    #  - mag pleth_2 (1)
    #  - mag pleth_1 (1)
    #  - aux features broadcasted (D)
    in_ch = 2 + D
    model = MaskNet(in_ch=in_ch, base=32).to(dev)
    subj_a = SubjectA(subject_ids).to(dev)

    opt = torch.optim.Adam(list(model.parameters()) + list(subj_a.parameters()), lr=lr)

    Ltr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True)
    Lva = DataLoader(ds_va, batch_size=batch_size, shuffle=False)

    best = {"val": float("inf"), "state": None, "a_state": None}

    pbar = tqdm(range(1, epochs + 1), desc=f"Denoiser[{activity}] epochs", leave=False)
    for ep in pbar:
        model.train()
        subj_a.train()
        tr_loss = 0.0
        for x_raw, feats, ecg_peaks, sids, acts in Ltr:
            # filter: all acts should be activity by dataset design, but keep safe
            x_raw = x_raw.to(dev)                 # (B,8,T)
            feats = feats.to(dev)                 # (B,D)
            ecg_peaks = ecg_peaks.to(dev)         # (B,T)

            pleth1 = x_raw[:, 0, :]               # (B,T)
            pleth2 = x_raw[:, 1, :]               # (B,T)

            # STFT of pleth_2 (complex)
            Y_noisy = torch_stft(pleth2, stft_cfg)            # (B,F,TT) complex
            mag_noisy = torch.abs(Y_noisy).unsqueeze(1)       # (B,1,F,TT)
            phase = torch.angle(Y_noisy)                      # (B,F,TT)

            # Add pleth_1 magnitude as context channel
            Y1 = torch_stft(pleth1, stft_cfg)
            mag1 = torch.abs(Y1).unsqueeze(1)

            aux = build_aux_feature_maps(feats, F_bins, TT_bins)  # (B,D,F,TT)
            X = torch.cat([mag_noisy, mag1, aux], dim=1)          # (B,2+D,F,TT)

            M = model(X)                                          # (B,1,F,TT)
            # Ensure mask has the same (F,TT) as the input magnitude (pool/upsample can cause off-by-one)
            if (M.shape[-2] != mag_noisy.shape[-2]) or (M.shape[-1] != mag_noisy.shape[-1]):
                M = F.interpolate(M, size=mag_noisy.shape[-2:], mode="bilinear", align_corners=False)
            # Reconstruct clean with original phase
            mag_clean = (M * mag_noisy).squeeze(1)               # (B,F,TT)
            Y_clean = torch.polar(mag_clean, phase)               # (B,F,TT) complex
            y_hat = torch_istft(Y_clean, stft_cfg, length=T).unsqueeze(1)  # (B,1,T)

            # Build shifted ECG peaks as supervision
            sid_idx = subj_a.sid_to_index(list(sids), dev)        # (B,)
            a_val = subj_a(sid_idx, activity).squeeze(1)          # (B,)
            # Δ0 table in samples
            delta0 = torch.tensor([delta0_by_subject[str(s)] for s in sids], device=dev, dtype=torch.float32)
            shift_samples = torch.round(a_val * delta0).to(torch.long)
            ecg_shifted = shift_peaks(ecg_peaks, shift_samples)

            # Primary: soft peak alignment (peak map vs shifted ECG peaks)
            pm = soft_peak_map(y_hat, fs=win_cfg.fs, k=10.0)      # (B,T)
            l_peak = bce_peak_loss(pm, ecg_shifted)

            # Secondary: shape similarity to SIT template (very low weight)
            # Align lengths
            l_shape = F.l1_loss(y_hat, sit_template_t)

            # Smoothness regularizer on mask (frequency)
            l_smooth = freq_smooth_reg(M)

            loss = l_peak + lam_shape * l_shape + lam_smooth * l_smooth

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(subj_a.parameters()), 1.0)
            opt.step()

            tr_loss += float(loss.item()) * x_raw.size(0)

        tr_loss /= max(1, len(Ltr.dataset))

        # validation
        model.eval()
        subj_a.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x_raw, feats, ecg_peaks, sids, acts in Lva:
                x_raw = x_raw.to(dev)
                feats = feats.to(dev)
                ecg_peaks = ecg_peaks.to(dev)
                pleth1 = x_raw[:, 0, :]
                pleth2 = x_raw[:, 1, :]

                Y_noisy = torch_stft(pleth2, stft_cfg)
                mag_noisy = torch.abs(Y_noisy).unsqueeze(1)
                phase = torch.angle(Y_noisy)

                Y1 = torch_stft(pleth1, stft_cfg)
                mag1 = torch.abs(Y1).unsqueeze(1)

                aux = build_aux_feature_maps(feats, F_bins, TT_bins)
                X = torch.cat([mag_noisy, mag1, aux], dim=1)
                M = model(X)
                # Ensure mask has the same (F,TT) as the input magnitude (pool/upsample can cause off-by-one)
                if (M.shape[-2] != mag_noisy.shape[-2]) or (M.shape[-1] != mag_noisy.shape[-1]):
                    M = F.interpolate(M, size=mag_noisy.shape[-2:], mode="bilinear", align_corners=False)
                mag_clean = (M * mag_noisy).squeeze(1)
                Y_clean = torch.polar(mag_clean, phase)
                y_hat = torch_istft(Y_clean, stft_cfg, length=T).unsqueeze(1)

                sid_idx = subj_a.sid_to_index(list(sids), dev)
                a_val = subj_a(sid_idx, activity).squeeze(1)
                delta0 = torch.tensor([delta0_by_subject[str(s)] for s in sids], device=dev, dtype=torch.float32)
                shift_samples = torch.round(a_val * delta0).to(torch.long)
                ecg_shifted = shift_peaks(ecg_peaks, shift_samples)

                pm = soft_peak_map(y_hat, fs=win_cfg.fs, k=10.0)
                l_peak = bce_peak_loss(pm, ecg_shifted)
                l_shape = F.l1_loss(y_hat, sit_template_t)
                l_smooth = freq_smooth_reg(M)
                loss = l_peak + lam_shape * l_shape + lam_smooth * l_smooth
                va_loss += float(loss.item()) * x_raw.size(0)

        va_loss /= max(1, len(Lva.dataset))
        pbar.set_postfix(train=f"{tr_loss:.4f}", val=f"{va_loss:.4f}")

        if va_loss < best["val"]:
            best["val"] = va_loss
            best["state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best["a_state"] = {k: v.detach().cpu() for k, v in subj_a.state_dict().items()}

    # restore best
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    if best["a_state"] is not None:
        subj_a.load_state_dict(best["a_state"])

    # export torch weights and ONNX
    ensure_dir(outdir)
    torch.save(model.state_dict(), outdir / f"masknet_{activity}.pt")
    torch.save(subj_a.state_dict(), outdir / f"subject_a_{activity}.pt")

    # ONNX export: forward on magnitude+aux; runtime will do STFT/ISTFT outside torch
    # Create dummy input for ONNX: (1,2+D,F,TT)
    dummy = torch.randn(1, in_ch, F_bins, TT_bins, device=dev)
    onnx_path = outdir / f"masknet_{activity}.onnx"
    try:
        torch.onnx.export(
            model,
            dummy,
            onnx_path.as_posix(),
            input_names=["X"],
            output_names=["M"],
            opset_version=18,
            do_constant_folding=True,
            dynamic_axes={"X": {0: "B"}, "M": {0: "B"}},
        )
    except Exception as e:
        print(f"[WARN] ONNX export failed for {activity}: {e}")

    meta = {
        "activity": activity,
        "stft": stft_cfg.__dict__,
        "win": win_cfg.__dict__,
        "feature_names": feat_names,
        "in_channels": int(in_ch),
        "model": "MaskNet",
        "lam_shape": lam_shape,
        "lam_smooth": lam_smooth,
        "best_val_loss": float(best["val"]),
    }
    (outdir / f"denoiser_{activity}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # also write a table for a values (CPU inference lookup)
    with torch.no_grad():
        sid_tensor = subj_a.sid_to_index(subject_ids, dev)
        a_vals = subj_a(sid_tensor, activity).squeeze(1).detach().cpu().numpy().astype(float).tolist()
    a_table = {sid: float(a_vals[i]) for i, sid in enumerate(subject_ids)}
    (outdir / f"subject_a_{activity}_table.json").write_text(json.dumps(a_table, indent=2), encoding="utf-8")

    return model, subj_a, meta


# ===========================
# Detector evaluation + plots
# ===========================
def plot_confusion(cm: np.ndarray, out: Path, title: str):
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["sit", "motion"])
    plt.yticks(ticks, ["sit", "motion"])
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

def plot_curve(x: np.ndarray, y: np.ndarray, out: Path, title: str, xlabel: str, ylabel: str):
    plt.figure(figsize=(5,4))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

def detector_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray] = None) -> Dict[str,float]:
    out = {
        "bal_acc": float(metrics.balanced_accuracy_score(y_true, y_pred)),
        "f1": float(metrics.f1_score(y_true, y_pred)),
        "precision": float(metrics.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(metrics.recall_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(metrics.roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(metrics.average_precision_score(y_true, y_score))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


# ===========================
# Pipeline: build window-level matrices
# ===========================
def build_window_matrices(
    records: List[Dict[str,Any]],
    win_cfg: WinConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[Tuple[str,str,str,int]]]:
    """
    Build:
      - F_ppg: (Nw, Dp) features (from pleth_2)
      - F_imu: (Nw, Di) features (from IMU)
      - y:     (Nw,) activity label (sit=0, walk/run=1)
      - names_ppg, names_imu
      - meta: list of (subject, activity, filename, window_index) for traceability
    """
    Fp, Fi, Y = [], [], []
    names_ppg: Optional[List[str]] = None
    names_imu: Optional[List[str]] = None
    meta: List[Tuple[str,str,str,int]] = []

    for r in tqdm(records, desc="Feature extraction", leave=False):
        N = len(r["pleth_2"])
        act = r["_activity"]
        y = 0 if act == "sit" else 1
        for wi, (s,e) in enumerate(split_windows(N, win_cfg.fs, win_cfg.win_sec, win_cfg.hop_sec)):
            pleth1 = r["pleth_1"][s:e].astype(np.float32)
            pleth2 = r["pleth_2"][s:e].astype(np.float32)

            acc_g = np.column_stack([r["a_x"][s:e], r["a_y"][s:e], r["a_z"][s:e]]).astype(np.float32)
            gyro_deg = np.column_stack([r["g_x"][s:e], r["g_y"][s:e], r["g_z"][s:e]]).astype(np.float32)

            acc_dyn = linear_acceleration(acc_g, fs=win_cfg.fs).astype(np.float32)
            gyro_rad = deg2rad(gyro_deg).astype(np.float32)

            f_ppg, n_ppg = ppg_features(pleth2, fs=win_cfg.fs)
            f_imu, n_imu = imu_features(acc_dyn, gyro_rad, fs=win_cfg.fs)

            if names_ppg is None:
                names_ppg = n_ppg
            if names_imu is None:
                names_imu = n_imu

            Fp.append(f_ppg)
            Fi.append(f_imu)
            Y.append(y)
            meta.append((r["_subject"], act, r["_file"], wi))

    F_ppg = np.stack(Fp, axis=0) if Fp else np.empty((0,0), dtype=np.float32)
    F_imu = np.stack(Fi, axis=0) if Fi else np.empty((0,0), dtype=np.float32)
    y = np.asarray(Y, dtype=int)

    return F_ppg, F_imu, y, (names_ppg or []), (names_imu or []), meta


# ===========================
# End-to-end run
# ===========================
def run_pipeline(
    data_root: str,
    outdir: str,
    fs: float = 500.0,
    win_sec: float = 6.0,
    hop_sec: float = 1.0,
    train_size: float = 0.8,
    n_splits: int = 5,
    seed: int = 42,
    # denoiser
    epochs_denoise: int = 10,
    lr: float = 1e-3,
    lam_shape: float = 0.05,
    lam_smooth: float = 0.05,
    stft_n_fft: int = 256,
    stft_hop: int = 64,
    stft_win: int = 256,
):
    root = Path(data_root)
    out = ensure_dir(Path(outdir))

    win_cfg = WinConfig(fs=fs, win_sec=win_sec, hop_sec=hop_sec, lag_min_sec=-5.0, lag_max_sec=5.0)
    stft_cfg = STFTConfig(n_fft=stft_n_fft, hop_length=stft_hop, win_length=stft_win)

    sub2recs = load_physionet_csv(root, fs=fs)
    subjects = sorted(list(sub2recs.keys()))
    train_sub, hold_sub = split_subjects(subjects, train_size=train_size, seed=seed)

    # flatten records
    train_records = [r for sid in train_sub for r in sub2recs[sid]]
    hold_records = [r for sid in hold_sub for r in sub2recs[sid]]

    # save split for audit
    split_info = {
        "train_subjects": train_sub,
        "holdout_subjects": hold_sub,
        "seed": seed,
        "train_size": train_size,
    }
    (out / "splits.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    # ---------------- Detector: CV within train subjects ----------------
    folds = build_group_kfold(train_records, n_splits=n_splits)
    lag_steps = list(range(int(math.floor(win_cfg.lag_min_sec / win_cfg.hop_sec)),
                           int(math.ceil(win_cfg.lag_max_sec / win_cfg.hop_sec)) + 1))

    det_cv_results = []
    det_cv_lags = []

    det_dir = ensure_dir(out / "detector")
    for k, (tr_idx, va_idx) in enumerate(tqdm(folds, desc="Detector CV folds", leave=False), start=1):
        tr_recs = [train_records[i] for i in tr_idx]
        va_recs = [train_records[i] for i in va_idx]

        # build matrices
        Fp_tr, Fi_tr, y_tr, names_ppg, names_imu, meta_tr = build_window_matrices(tr_recs, win_cfg)
        Fp_va, Fi_va, y_va, _, _, meta_va = build_window_matrices(va_recs, win_cfg)

        # per-feature τ_ppg on train
        taus_ppg = np.zeros(Fp_tr.shape[1], dtype=np.float32)
        dirs_ppg = np.ones(Fp_tr.shape[1], dtype=np.int32)
        for j in range(Fp_tr.shape[1]):
            taus_ppg[j], dirs_ppg[j], _ = best_threshold_per_feature(Fp_tr[:, j], y_tr)

        # lag search using PPG vs IMU train matrices
        lag_best, lag_scores = lag_search_ppg_imu(Fp_tr, Fi_tr, lag_steps)
        det_cv_lags.append(lag_best)

        # align IMU by lag on train and fit τ_imu
        def align(Fp: np.ndarray, Fi: np.ndarray, y: np.ndarray, lag: int):
            T = min(len(Fp), len(Fi))
            if lag >= 0:
                return Fp[:T-lag], Fi[lag:T], y[:T-lag]
            kk = -lag
            return Fp[kk:T], Fi[:T-kk], y[kk:T]

        Fp_tr2, Fi_tr2, y_tr2 = align(Fp_tr, Fi_tr, y_tr, lag_best)
        Fp_va2, Fi_va2, y_va2 = align(Fp_va, Fi_va, y_va, lag_best)

        taus_imu = np.zeros(Fi_tr2.shape[1], dtype=np.float32)
        dirs_imu = np.ones(Fi_tr2.shape[1], dtype=np.int32)
        for j in range(Fi_tr2.shape[1]):
            taus_imu[j], dirs_imu[j], _ = best_threshold_per_feature(Fi_tr2[:, j], y_tr2)

        # predictions
        pred_imu_or = apply_threshold_matrix(Fi_va2, taus_imu, dirs_imu, rule="any")
        pred_ppg_or = apply_threshold_matrix(Fp_va2, taus_ppg, dirs_ppg, rule="any")
        pred_or = ((pred_imu_or > 0) | (pred_ppg_or > 0)).astype(int)
        pred_and = ((pred_imu_or > 0) & (pred_ppg_or > 0)).astype(int)

        m_or = detector_metrics(y_va2, pred_or)
        m_and = detector_metrics(y_va2, pred_and)

        cm_or = metrics.confusion_matrix(y_va2, pred_or)
        cm_and = metrics.confusion_matrix(y_va2, pred_and)

        fold_out = det_dir / f"cv_fold{k}"
        ensure_dir(fold_out)
        plot_confusion(cm_or, fold_out / "confusion_or.png", f"Detector CV fold{k} (OR)")
        plot_confusion(cm_and, fold_out / "confusion_and.png", f"Detector CV fold{k} (AND)")
        # lag curve
        xs = np.array(sorted(lag_scores.keys()), dtype=float) * win_cfg.hop_sec
        ys = np.array([lag_scores[int(l)] for l in sorted(lag_scores.keys())], dtype=float)
        plot_curve(xs, ys, fold_out / "lag_assoc.png", f"Lag association fold{k}", "lag (s)", "mean |Spearman|")

        det_cv_results.append({
            "fold": k,
            "lag_best_steps": int(lag_best),
            "lag_best_sec": float(lag_best * win_cfg.hop_sec),
            "metrics_or": m_or,
            "metrics_and": m_and,
            "n_val": int(len(y_va2)),
            "pos_ratio": float(np.mean(y_va2)) if len(y_va2) else float("nan"),
        })

    # lag stability
    lags_sec = [l * win_cfg.hop_sec for l in det_cv_lags]
    det_summary = {
        "cv_folds": det_cv_results,
        "lag_median_sec": float(np.median(lags_sec)) if lags_sec else 0.0,
        "lag_iqr_sec": float(np.percentile(lags_sec, 75) - np.percentile(lags_sec, 25)) if lags_sec else 0.0,
        "ppg_feature_names": names_ppg,
        "imu_feature_names": names_imu,
        "win_cfg": win_cfg.__dict__,
    }
    (det_dir / "cv_summary.json").write_text(json.dumps(det_summary, indent=2), encoding="utf-8")

    # ---------------- Detector: Holdout ----------------
    # Fit thresholds and lag on ALL train subjects (no holdout usage), evaluate on holdout.
    Fp_tr, Fi_tr, y_tr, names_ppg, names_imu, _ = build_window_matrices(train_records, win_cfg)
    Fp_ho, Fi_ho, y_ho, _, _, _ = build_window_matrices(hold_records, win_cfg)

    taus_ppg = np.zeros(Fp_tr.shape[1], dtype=np.float32)
    dirs_ppg = np.ones(Fp_tr.shape[1], dtype=np.int32)
    for j in range(Fp_tr.shape[1]):
        taus_ppg[j], dirs_ppg[j], _ = best_threshold_per_feature(Fp_tr[:, j], y_tr)

    lag_best, lag_scores = lag_search_ppg_imu(Fp_tr, Fi_tr, lag_steps)

    def align2(Fp: np.ndarray, Fi: np.ndarray, y: np.ndarray, lag: int):
        T = min(len(Fp), len(Fi))
        if lag >= 0:
            return Fp[:T-lag], Fi[lag:T], y[:T-lag]
        kk = -lag
        return Fp[kk:T], Fi[:T-kk], y[kk:T]

    Fp_tr2, Fi_tr2, y_tr2 = align2(Fp_tr, Fi_tr, y_tr, lag_best)
    Fp_ho2, Fi_ho2, y_ho2 = align2(Fp_ho, Fi_ho, y_ho, lag_best)

    taus_imu = np.zeros(Fi_tr2.shape[1], dtype=np.float32)
    dirs_imu = np.ones(Fi_tr2.shape[1], dtype=np.int32)
    for j in range(Fi_tr2.shape[1]):
        taus_imu[j], dirs_imu[j], _ = best_threshold_per_feature(Fi_tr2[:, j], y_tr2)

    pred_imu_or = apply_threshold_matrix(Fi_ho2, taus_imu, dirs_imu, rule="any")
    pred_ppg_or = apply_threshold_matrix(Fp_ho2, taus_ppg, dirs_ppg, rule="any")
    pred_or = ((pred_imu_or > 0) | (pred_ppg_or > 0)).astype(int)
    pred_and = ((pred_imu_or > 0) & (pred_ppg_or > 0)).astype(int)

    m_or = detector_metrics(y_ho2, pred_or)
    m_and = detector_metrics(y_ho2, pred_and)

    cm_or = metrics.confusion_matrix(y_ho2, pred_or)
    cm_and = metrics.confusion_matrix(y_ho2, pred_and)

    hold_dir = ensure_dir(det_dir / "holdout")
    plot_confusion(cm_or, hold_dir / "confusion_or.png", "Detector HOLDOUT (OR)")
    plot_confusion(cm_and, hold_dir / "confusion_and.png", "Detector HOLDOUT (AND)")
    xs = np.array(sorted(lag_scores.keys()), dtype=float) * win_cfg.hop_sec
    ys = np.array([lag_scores[int(l)] for l in sorted(lag_scores.keys())], dtype=float)
    plot_curve(xs, ys, hold_dir / "lag_assoc.png", "Lag association (train fit)", "lag (s)", "mean |Spearman|")

    detector_artifact = {
        "ppg_feature_names": names_ppg,
        "imu_feature_names": names_imu,
        "taus_ppg": taus_ppg.tolist(),
        "dirs_ppg": dirs_ppg.tolist(),
        "taus_imu": taus_imu.tolist(),
        "dirs_imu": dirs_imu.tolist(),
        "lag_best_steps": int(lag_best),
        "lag_best_sec": float(lag_best * win_cfg.hop_sec),
        "metrics_holdout_or": m_or,
        "metrics_holdout_and": m_and,
        "win_cfg": win_cfg.__dict__,
        "rule": {"primary": "OR", "also_report": ["AND"]},
        "version": "v7.3",
    }
    (det_dir / "detector_artifact.json").write_text(json.dumps(detector_artifact, indent=2), encoding="utf-8")

    # ---------------- Denoiser: walk & run (using train/holdout split discipline) ----------------
    den_dir = ensure_dir(out / "denoiser")
    # subject delay Δ0 from train only, per subject
    delta0 = {sid: estimate_subject_delay_sit(sub2recs[sid], fs=fs) for sid in train_sub}
    (den_dir / "delta0_train_subjects.json").write_text(json.dumps({k: float(v) for k,v in delta0.items()}, indent=2), encoding="utf-8")

    # inner split within train subjects for early stopping (GroupShuffleSplit on records)
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=seed)
    idx = np.arange(len(train_records))
    groups = np.asarray([r["_subject"] for r in train_records])
    tr_idx, va_idx = next(gss.split(idx, groups=groups))
    inner_tr = [train_records[i] for i in tr_idx]
    inner_va = [train_records[i] for i in va_idx]

    # Train separate models
    for act in ["walk", "run"]:
        train_denoiser_one_activity(
            activity=act,
            train_records=inner_tr,
            val_records=inner_va,
            subject_ids=train_sub,
            delta0_by_subject=delta0,
            win_cfg=win_cfg,
            stft_cfg=stft_cfg,
            outdir=ensure_dir(den_dir / act),
            epochs=epochs_denoise,
            lr=lr,
            lam_shape=lam_shape,
            lam_smooth=lam_smooth,
            batch_size=16,
        )

    # Save an inference helper stub (CPU-only) for convenience
    helper = {
        "note": "Use onnxruntime to run masknet_*.onnx on magnitude spectrogram + broadcasted features. "
                "Compute STFT/ISTFT with scipy.signal.stft/istft and keep noisy phase."
    }
    (out / "README_inference.json").write_text(json.dumps(helper, indent=2), encoding="utf-8")

    print(f"Done. Outputs saved to: {out.resolve()}")


# ===========================
# CLI
# ===========================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Root folder containing csv/ (PhysioNet PTT-PPG CSV)")
    ap.add_argument("--outdir", type=str, default="results_v7_3", help="Output directory")

    ap.add_argument("--fs", type=float, default=500.0)
    ap.add_argument("--win", type=float, default=6.0)
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--train_size", type=float, default=0.8, help="Subject-level train fraction (rest is holdout)")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--epochs_denoise", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam_shape", type=float, default=0.05)
    ap.add_argument("--lam_smooth", type=float, default=0.05)

    ap.add_argument("--stft_n_fft", type=int, default=256)
    ap.add_argument("--stft_hop", type=int, default=64)
    ap.add_argument("--stft_win", type=int, default=256)

    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        data_root=args.data_root,
        outdir=args.outdir,
        fs=args.fs,
        win_sec=args.win,
        hop_sec=args.hop,
        train_size=args.train_size,
        n_splits=args.n_splits,
        seed=args.seed,
        epochs_denoise=args.epochs_denoise,
        lr=args.lr,
        lam_shape=args.lam_shape,
        lam_smooth=args.lam_smooth,
        stft_n_fft=args.stft_n_fft,
        stft_hop=args.stft_hop,
        stft_win=args.stft_win,
    )
