# ========== JSON helpers ==========
def _json_sanitize(obj):
    """Recursively convert objects to JSON-serializable Python types.

    - numpy arrays -> lists
    - numpy scalars -> Python scalars
    - NaN/Inf -> None
    - pathlib.Path -> str
    """
    import math
    try:
        import numpy as _np
    except Exception:
        _np = None

    if obj is None:
        return None

    # Path-like
    try:
        from pathlib import Path as _Path
        if isinstance(obj, _Path):
            return str(obj)
    except Exception:
        pass

    # numpy types
    if _np is not None:
        if isinstance(obj, _np.ndarray):
            return _json_sanitize(obj.tolist())
        if isinstance(obj, (_np.floating,)):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.bool_,)):
            return bool(obj)

    # torch tensors (avoid hard dependency)
    try:
        import torch as _torch
        if isinstance(obj, _torch.Tensor):
            return _json_sanitize(obj.detach().cpu().numpy())
    except Exception:
        pass

    # plain floats
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj

    # mappings / sequences
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]

    # fallback: try to cast
    if isinstance(obj, (int, bool, str)):
        return obj

    return str(obj)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pttppg_detector_v8_scores.py

Detector (motion/activity) training script for the PhysioNet "Pulse Transit Time PPG" CSV export.

Design goals
------------
1) Subject-level no-leak splitting:
   - A subject-level holdout split is created FIRST and never touched during training or CV.
   - CV (GroupKFold) is done ONLY inside the training subjects.

2) "Score model" instead of OR/AND threshold rules:
   - Compute window-level feature vectors for PPG and IMU.
   - Use AE as a SIT clean-anchor selector (not as the final detector).
   - Estimate robust mean/covariance on SIT clean anchors.
   - Define PPG score and IMU score as whitened L2 / Mahalanobis distance to the SIT anchor distribution.
   - Learn lag (±lag_sec) by maximizing correlation between IMU score and PPG score.

3) Final classifier:
   - Logistic regression on [PPG_score, IMU_score_shifted].
   - Targets: activity (sit=0, walk/run=1) inferred from filename.

Outputs
-------
- JSON summary with CV + holdout metrics
- Confusion matrices (CV aggregate + holdout) as PNG
- ROC/PR curve (when defined) as PNG
- Saved detector bundle (npz): mu/cov for PPG+IMU, lag, logistic coefficients
"""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    balanced_accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay
)

import matplotlib

# Headless-safe backend (WSL / servers without DISPLAY)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Optional torch (AE clean anchor selector) ----
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception:
    torch = None

# ---- Import project functions (funcs.py) ----
try:
    import funcs
except Exception:
    funcs = None

# ----------------------------
# Filtering utilities
# ----------------------------
def bandpass_ppg(x: np.ndarray, fs: float, low: float = 0.5, high: float = 8.0) -> np.ndarray:
    """Bandpass for PPG prior to windowing."""
    x = np.asarray(x, dtype=np.float32)
    if funcs is not None and hasattr(funcs, "bandpass_filter"):
        # funcs.bandpass_filter(sig, lowcut=..., highcut=..., fs=..., order=...)
        return funcs.bandpass_filter(x, lowcut=low, highcut=high, fs=fs, order=3).astype(np.float32)
    try:
        from scipy.signal import butter, filtfilt
        nyq = 0.5 * fs
        b, a = butter(3, [low/nyq, high/nyq], btype="band")
        return filtfilt(b, a, x).astype(np.float32)
    except Exception:
        return x

def lowpass(x: np.ndarray, fs: float, cutoff: float = 0.3) -> np.ndarray:
    """Low-pass used to estimate gravity component from accelerometer."""
    x = np.asarray(x, dtype=np.float32)
    try:
        from scipy.signal import butter, filtfilt
        nyq = 0.5 * fs
        b, a = butter(2, cutoff/nyq, btype="low")
        return filtfilt(b, a, x).astype(np.float32)
    except Exception:
        w = max(3, int(fs * 1.0))
        k = np.ones(w, dtype=np.float32) / w
        return np.convolve(x, k, mode="same").astype(np.float32)

def zscore_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = float(np.nanmean(x))
    s = float(np.nanstd(x) + 1e-8)
    return (x - m) / s

def window_slices(n: int, fs: float, win_sec: float, hop_sec: float):
    w = int(round(win_sec * fs))
    h = int(round(hop_sec * fs))
    w = max(2, w); h = max(1, h)
    for s in range(0, max(1, n - w + 1), h):
        yield s, s + w

# ----------------------------
# Feature extraction
# ----------------------------
def spectral_entropy_and_domfreq(x: np.ndarray, fs: float, fmin: float = 0.1, fmax: float = 8.0) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float32)
    try:
        from scipy.signal import welch
        f, p = welch(x, fs=fs, nperseg=min(len(x), 512))
        m = (f >= fmin) & (f <= fmax)
        f = f[m]; p = p[m]
        p = np.maximum(p, 1e-12)
        p = p / np.sum(p)
        ent = float(-np.sum(p * np.log(p)))
        dom = float(f[int(np.argmax(p))]) if len(p) else float("nan")
        return ent, dom
    except Exception:
        return float("nan"), float("nan")

def bandpowers(x: np.ndarray, fs: float, bands=((0.1,0.5),(0.5,3.0),(3.0,8.0))) -> List[float]:
    x = np.asarray(x, dtype=np.float32)
    try:
        from scipy.signal import welch
        f, p = welch(x, fs=fs, nperseg=min(len(x), 512))
        out = []
        for lo, hi in bands:
            m = (f >= lo) & (f < hi)
            out.append(float(np.trapezoid(p[m], f[m])) if np.any(m) else 0.0)
        return out
    except Exception:
        return [0.0]*len(bands)

def time_stats(x: np.ndarray) -> List[float]:
    x = np.asarray(x, dtype=np.float32)
    med = float(np.nanmedian(x))
    q1 = float(np.nanpercentile(x, 25))
    q3 = float(np.nanpercentile(x, 75))
    iqr = q3 - q1
    rms = float(np.sqrt(np.nanmean(x*x)))
    return [float(np.nanmean(x)), float(np.nanstd(x)), med, float(iqr), rms]

def extract_window_features(ppg: np.ndarray,
                            acc_dyn_xyz_mps2: np.ndarray,
                            gyro_xyz_rads: np.ndarray,
                            fs: float) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    """Returns (ppg_feat, ppg_names, imu_feat, imu_names)."""
    ppg_feat = []
    ppg_names = []

    ppg_feat.extend(time_stats(ppg))
    ppg_names += ["PPG_ts_mean","PPG_ts_std","PPG_ts_med","PPG_ts_iqr","PPG_ts_rms"]
    bp = bandpowers(ppg, fs)
    ppg_feat.extend(bp)
    ppg_names += ["PPG_bp_0.1_0.5","PPG_bp_0.5_3","PPG_bp_3_8"]
    ent, dom = spectral_entropy_and_domfreq(ppg, fs)
    ppg_feat += [ent, dom]
    ppg_names += ["PPG_spec_entropy","PPG_dom_freq"]

    acc_mag = np.linalg.norm(acc_dyn_xyz_mps2, axis=1)
    gyro_mag = np.linalg.norm(gyro_xyz_rads, axis=1)
    jerk = np.diff(acc_mag, prepend=acc_mag[:1]) * fs

    imu_feat = []
    imu_names = []

    for name, sig in [("AccMag", acc_mag), ("GyroMag", gyro_mag), ("JerkMag", jerk)]:
        imu_feat.extend(time_stats(sig))
        imu_names += [f"{name}_ts_mean",f"{name}_ts_std",f"{name}_ts_med",f"{name}_ts_iqr",f"{name}_ts_rms"]

    for name, sig in [("AccMag", acc_mag), ("GyroMag", gyro_mag)]:
        imu_feat.extend(bandpowers(sig, fs))
        imu_names += [f"{name}_bp_0.1_0.5",f"{name}_bp_0.5_3",f"{name}_bp_3_8"]
        ent, dom = spectral_entropy_and_domfreq(sig, fs)
        imu_feat += [ent, dom]
        imu_names += [f"{name}_spec_entropy",f"{name}_dom_freq"]

    ent, dom = spectral_entropy_and_domfreq(jerk, fs)
    imu_feat += [ent, dom]
    imu_names += ["JerkMag_spec_entropy","JerkMag_dom_freq"]

    return np.asarray(ppg_feat, dtype=np.float32), ppg_names, np.asarray(imu_feat, dtype=np.float32), imu_names

# ----------------------------
# Data loading
# ----------------------------
CAND = {
    "time": ["time","Time","t"],
    "pleth_1": ["pleth_1"],
    "pleth_2": ["pleth_2"],
    "a_x": ["a_x"], "a_y": ["a_y"], "a_z": ["a_z"],
    "g_x": ["g_x"], "g_y": ["g_y"], "g_z": ["g_z"],
}

def _pick(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def parse_activity_from_name(stem: str) -> str:
    parts = stem.split("_")
    return parts[1] if len(parts) >= 2 else "unknown"

def load_physionet_csv(data_root: Path, fs: float) -> Dict[str, List[dict]]:
    csv_dir = data_root / "csv"
    if not csv_dir.exists():
        csv_dir = data_root / "files" / "pulse-transit-time-ppg" / "1.1.0" / "csv"
    paths = sorted([p for p in csv_dir.glob("s*_*.csv") if p.name != "subjects_info.csv"])
    out: Dict[str, List[dict]] = {}
    for p in tqdm(paths, desc="CSV load", leave=False):
        df = pd.read_csv(p)
        m = {k:_pick(df,v) for k,v in CAND.items()}
        rec = {"_path": str(p), "_name": p.stem, "_activity": parse_activity_from_name(p.stem)}

        if m["time"] is not None:
            t = pd.to_datetime(df[m["time"]], errors="coerce")
            if t.notna().all():
                rec["time"] = (t - t.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float32)
            else:
                rec["time"] = (np.arange(len(df), dtype=np.float32) / float(fs))
        else:
            rec["time"] = (np.arange(len(df), dtype=np.float32) / float(fs))

        for k in ["pleth_1","pleth_2","a_x","a_y","a_z","g_x","g_y","g_z"]:
            if m[k] is None:
                continue
            rec[k] = pd.to_numeric(df[m[k]], errors="coerce").to_numpy(dtype=np.float32)

        sid = p.stem.split("_")[0]
        out.setdefault(sid, []).append(rec)
    return out

# ----------------------------
# AE clean-anchor selector
# ----------------------------
class _AEDataset(Dataset):
    def __init__(self, recs: List[dict], fs: float, win_sec: float, hop_sec: float):
        self.fs=fs; self.win_sec=win_sec; self.hop_sec=hop_sec
        self.idx=[]
        for ri, r in enumerate(recs):
            if r.get("_activity") != "sit":
                continue
            if "pleth_1" not in r or "pleth_2" not in r:
                continue
            n = len(r["pleth_2"])
            for s,e in window_slices(n, fs, win_sec, hop_sec):
                self.idx.append((ri,s,e))
        self.recs=recs

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        ri,s,e = self.idx[i]
        r = self.recs[ri]
        p1 = bandpass_ppg(r["pleth_1"][s:e], self.fs)
        p2 = bandpass_ppg(r["pleth_2"][s:e], self.fs)
        x = np.stack([zscore_np(p1), zscore_np(p2)], axis=0).astype(np.float32)
        return torch.from_numpy(x)

class _SmallConvAE(nn.Module):
    def __init__(self, in_ch=2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(in_ch, 32, 5, padding=2), nn.ReLU(True),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(True),
            nn.MaxPool1d(4),
        )
        self.dec = nn.Sequential(
            nn.Conv1d(64, 32, 3, padding=1), nn.ReLU(True),
            nn.Upsample(scale_factor=4, mode="linear", align_corners=False),
            nn.Conv1d(32, 16, 3, padding=1), nn.ReLU(True),
            nn.Upsample(scale_factor=4, mode="linear", align_corners=False),
            nn.Conv1d(16, in_ch, 1),
        )

    def forward(self, x):
        z = self.enc(x)
        y = self.dec(z)
        if y.shape[-1] != x.shape[-1]:
            y = F.interpolate(y, size=x.shape[-1], mode="linear", align_corners=False)
        return y

def ae_select_clean_sit_windows(sit_recs: List[dict],
                               fs: float,
                               win_sec: float,
                               hop_sec: float,
                               q: float = 0.2,
                               epochs: int = 10,
                               lr: float = 1e-3,
                               batch_size: int = 128,
                               disable: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Return (recon_errs, clean_mask)."""
    if disable or torch is None:
        errs = []
        for r in sit_recs:
            if r.get("_activity") != "sit" or "pleth_2" not in r:
                continue
            x = bandpass_ppg(r["pleth_2"], fs)
            for s,e in window_slices(len(x), fs, win_sec, hop_sec):
                seg = x[s:e]
                errs.append(float(np.sqrt(np.mean(seg*seg))))
        errs = np.asarray(errs, dtype=np.float32)
        thr = np.quantile(errs, q)
        return errs, (errs <= thr)

    ds = _AEDataset(sit_recs, fs, win_sec, hop_sec)
    if len(ds) < 50:
        return ae_select_clean_sit_windows(sit_recs, fs, win_sec, hop_sec, q=q, disable=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = _SmallConvAE().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    for _ in tqdm(range(epochs), desc="AE epochs (sit anchors)", leave=False):
        model.train()
        for x in dl:
            x = x.to(dev)
            y = model(x)
            loss = F.mse_loss(y, x)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    dl2 = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    errs = []
    with torch.no_grad():
        for x in dl2:
            x = x.to(dev)
            y = model(x)
            e = F.mse_loss(y, x, reduction="none").mean(dim=(1,2)).detach().cpu().numpy()
            errs.append(e)
    errs = np.concatenate(errs, axis=0).astype(np.float32)
    thr = float(np.quantile(errs, q))
    return errs, (errs <= thr)

# ----------------------------
# Robust covariance / scoring
# ----------------------------
def shrinkage_cov(X: np.ndarray, shrink: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """mu=median, cov=empirical then shrink toward diagonal."""
    X = np.asarray(X, dtype=np.float64)
    mu = np.median(X, axis=0)
    Xm = X - mu
    cov = (Xm.T @ Xm) / max(1, (len(X)-1))
    diag = np.diag(np.diag(cov))
    cov_sh = (1.0 - shrink) * cov + shrink * diag
    return mu.astype(np.float32), cov_sh.astype(np.float32)

def whitened_l2_scores(X: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=np.float32) + np.eye(cov.shape[0], dtype=np.float32)*1e-6
    L = np.linalg.cholesky(cov).astype(np.float32)
    Xm = (np.asarray(X, dtype=np.float32) - np.asarray(mu, dtype=np.float32))
    z = np.linalg.solve(L, Xm.T).T
    return np.sqrt(np.sum(z*z, axis=1)).astype(np.float32)

# ----------------------------
# Window table + lag
# ----------------------------
@dataclass
class WinCfg:
    fs: float
    win_sec: float
    hop_sec: float
    lag_min_sec: float
    lag_max_sec: float

def build_window_table(sub2recs: Dict[str, List[dict]], cfg: WinCfg) -> pd.DataFrame:
    rows = []
    for sid, recs in sub2recs.items():
        for r in recs:
            if "pleth_1" not in r or "pleth_2" not in r:
                continue
            if not all(k in r for k in ["a_x","a_y","a_z","g_x","g_y","g_z"]):
                continue

            p1 = bandpass_ppg(r["pleth_1"], cfg.fs)
            p2 = bandpass_ppg(r["pleth_2"], cfg.fs)

            ax = r["a_x"] * 9.81; ay = r["a_y"] * 9.81; az = r["a_z"] * 9.81
            gx = np.deg2rad(r["g_x"]); gy = np.deg2rad(r["g_y"]); gz = np.deg2rad(r["g_z"])
            n = min(len(p2), len(ax), len(gx))
            p2=p2[:n]; ax=ax[:n]; ay=ay[:n]; az=az[:n]; gx=gx[:n]; gy=gy[:n]; gz=gz[:n]

            gax = lowpass(ax, cfg.fs); gay = lowpass(ay, cfg.fs); gaz = lowpass(az, cfg.fs)
            acc_dyn = np.column_stack([ax-gax, ay-gay, az-gaz]).astype(np.float32)
            gyro = np.column_stack([gx, gy, gz]).astype(np.float32)

            y = 0 if r.get("_activity") == "sit" else 1

            for s,e in window_slices(n, cfg.fs, cfg.win_sec, cfg.hop_sec):
                ppg_win = p2[s:e]
                ppg_feat, ppg_names, imu_feat, imu_names = extract_window_features(ppg_win, acc_dyn[s:e], gyro[s:e], cfg.fs)
                rows.append({
                    "sid": sid,
                    "activity": r.get("_activity","unknown"),
                    "y": y,
                    "t0": float(s / cfg.fs),
                    "ppg_feat": ppg_feat,
                    "imu_feat": imu_feat,
                })
    df = pd.DataFrame(rows)
    df.attrs["ppg_feature_names"] = ppg_names
    df.attrs["imu_feature_names"] = imu_names
    return df

def best_lag_from_scores(ppg_score: np.ndarray, imu_score: np.ndarray, cfg: WinCfg) -> Tuple[int, float, np.ndarray]:
    """Search lag in steps of hop; maximize Pearson correlation."""
    max_steps = int(round(cfg.lag_max_sec / cfg.hop_sec))
    min_steps = int(round(cfg.lag_min_sec / cfg.hop_sec))
    lags = list(range(min_steps, max_steps+1))
    corrs = []
    for k in lags:
        if k >= 0:
            a = imu_score[:-k] if k>0 else imu_score
            b = ppg_score[k:] if k>0 else ppg_score
        else:
            kk = -k
            a = imu_score[kk:]
            b = ppg_score[:-kk]
        if len(a) < 10:
            corrs.append(-np.inf)
            continue
        c = np.corrcoef(a, b)[0,1]
        corrs.append(float(c) if np.isfinite(c) else -np.inf)
    corrs = np.asarray(corrs, dtype=np.float32)
    best_i = int(np.argmax(corrs))
    best_steps = int(lags[best_i])
    best_sec = float(best_steps * cfg.hop_sec)
    return best_steps, best_sec, corrs

def shift_by_steps(x: np.ndarray, steps: int) -> np.ndarray:
    if steps == 0:
        return x
    out = np.empty_like(x)
    out[:] = np.nan
    if steps > 0:
        out[steps:] = x[:-steps]
    else:
        s = -steps
        out[:-s] = x[s:]
    return out

# ----------------------------
# Metrics & plots
# ----------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray]) -> dict:
    out = {
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            out["roc_auc"] = float("nan")
        try:
            out["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            out["pr_auc"] = float("nan")
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out

def save_confmat_png(y_true, y_pred, path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["sit","motion"])
    fig, ax = plt.subplots(figsize=(4,4))
    disp.plot(ax=ax, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def save_roc_pr_png(y_true, y_prob, outdir: Path, prefix: str):
    try:
        fig, ax = plt.subplots(figsize=(5,4))
        RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
        ax.set_title(f"{prefix} ROC")
        fig.tight_layout()
        fig.savefig(outdir / f"{prefix}_roc.png", dpi=160)
        plt.close(fig)
    except Exception:
        pass
    try:
        fig, ax = plt.subplots(figsize=(5,4))
        PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
        ax.set_title(f"{prefix} PR")
        fig.tight_layout()
        fig.savefig(outdir / f"{prefix}_pr.png", dpi=160)
        plt.close(fig)
    except Exception:
        pass

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="results_detector_v8")
    ap.add_argument("--fs", type=float, default=500.0)
    ap.add_argument("--win", type=float, default=6.0)
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--train_size", type=float, default=0.8)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--lag_min_sec", type=float, default=-5.0)
    ap.add_argument("--lag_max_sec", type=float, default=5.0)
    ap.add_argument("--ae_q", type=float, default=0.2)
    ap.add_argument("--ae_epochs", type=int, default=10)
    ap.add_argument("--ae_disable", action="store_true")
    ap.add_argument("--cov_shrink", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    np.random.seed(args.seed)
    random.seed(args.seed)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cfg = WinCfg(fs=args.fs, win_sec=args.win, hop_sec=args.hop,
                 lag_min_sec=args.lag_min_sec, lag_max_sec=args.lag_max_sec)

    sub2recs = load_physionet_csv(Path(args.data_root), fs=args.fs)
    subjects = np.array(sorted(sub2recs.keys()))
    if len(subjects) < 3:
        raise RuntimeError("Not enough subjects found.")

    # Holdout FIRST (subject-level no-leak)
    gss = GroupShuffleSplit(n_splits=1, train_size=args.train_size, random_state=args.seed)
    idx = np.arange(len(subjects))
    tr_sub_idx, ho_sub_idx = next(gss.split(idx, groups=subjects))
    tr_subjects = subjects[tr_sub_idx].tolist()
    ho_subjects = subjects[ho_sub_idx].tolist()
    tr_sub2recs = {sid: sub2recs[sid] for sid in tr_subjects}
    ho_sub2recs = {sid: sub2recs[sid] for sid in ho_subjects}

    # Features
    df_tr = build_window_table(tr_sub2recs, cfg)
    df_ho = build_window_table(ho_sub2recs, cfg)

    ppg_names = df_tr.attrs["ppg_feature_names"]
    imu_names = df_tr.attrs["imu_feature_names"]

    # AE anchors from SIT in training subjects
    sit_recs = [r for sid in tr_subjects for r in sub2recs[sid] if r.get("_activity") == "sit"]
    ae_errs, ae_mask = ae_select_clean_sit_windows(
        sit_recs, fs=args.fs, win_sec=args.win, hop_sec=args.hop,
        q=args.ae_q, epochs=args.ae_epochs, disable=args.ae_disable
    )

    sit_df = df_tr[df_tr["activity"] == "sit"].copy()
    if len(sit_df) < 10:
        raise RuntimeError("No SIT windows in training set after feature extraction.")

    if len(ae_errs) == len(sit_df):
        sit_df["ae_err"] = ae_errs
        thr = float(np.quantile(ae_errs, args.ae_q))
        anchor_df = sit_df[sit_df["ae_err"] <= thr]
    else:
        ppg_rms = np.stack(sit_df["ppg_feat"].to_list())[:, 4]
        thr = float(np.quantile(ppg_rms, args.ae_q))
        anchor_df = sit_df[ppg_rms <= thr]

    Xp = np.stack(anchor_df["ppg_feat"].to_list())
    Xi = np.stack(anchor_df["imu_feat"].to_list())
    mu_p, cov_p = shrinkage_cov(Xp, shrink=args.cov_shrink)
    mu_i, cov_i = shrinkage_cov(Xi, shrink=args.cov_shrink)

    def add_scores(df: pd.DataFrame) -> pd.DataFrame:
        Xp = np.stack(df["ppg_feat"].to_list())
        Xi = np.stack(df["imu_feat"].to_list())
        df = df.copy()
        df["ppg_score"] = whitened_l2_scores(Xp, mu_p, cov_p)
        df["imu_score"] = whitened_l2_scores(Xi, mu_i, cov_i)
        return df

    df_tr = add_scores(df_tr)
    df_ho = add_scores(df_ho)

    
    # ----------------------------
    # Lag search + CV (no-leak for lag: lag is learned on each fold's train split)
    # ----------------------------

    # Global lag from ALL training subjects (used for final holdout model)
    best_steps, best_sec, corr_by_lag = best_lag_from_scores(
        df_tr["ppg_score"].to_numpy(np.float32),
        df_tr["imu_score"].to_numpy(np.float32),
        cfg
    )
    df_tr["imu_score_shift"] = shift_by_steps(df_tr["imu_score"].to_numpy(np.float32), best_steps)
    df_ho["imu_score_shift"] = shift_by_steps(df_ho["imu_score"].to_numpy(np.float32), best_steps)
    df_tr2 = df_tr.dropna(subset=["imu_score_shift"]).copy()
    df_ho2 = df_ho.dropna(subset=["imu_score_shift"]).copy()

    # CV inside training subjects (fold-specific lag)
    gkf = GroupKFold(n_splits=min(args.n_splits, len(tr_subjects)))
    X_raw = df_tr2[["ppg_score","imu_score"]].to_numpy(np.float32)   # NOTE: unshifted IMU for fold-wise lag
    y_raw = df_tr2["y"].to_numpy(int)
    groups = df_tr2["sid"].to_numpy(str)

    cv_metrics = []
    cv_lags = []
    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    for fold, (tri, vai) in enumerate(gkf.split(X_raw, y_raw, groups=groups), start=1):
        # Learn lag ONLY on this fold's train split (audit point F)
        ppg_tr = X_raw[tri, 0]
        imu_tr = X_raw[tri, 1]
        lag_steps_f, lag_sec_f, _corr = best_lag_from_scores(ppg_tr, imu_tr, cfg)
        cv_lags.append({"fold": fold, "lag_steps": int(lag_steps_f), "lag_sec": float(lag_sec_f), "corr_by_lag": [float(x) for x in _corr]})

        # Build train/val matrices with the fold-specific shift
        imu_tr_shift = shift_by_steps(imu_tr, lag_steps_f)
        ppg_va = X_raw[vai, 0]
        imu_va = X_raw[vai, 1]
        imu_va_shift = shift_by_steps(imu_va, lag_steps_f)

        # Drop NaNs introduced by shifting (keep aligned)
        tr_mask = np.isfinite(imu_tr_shift)
        va_mask = np.isfinite(imu_va_shift)
        X_tr = np.stack([ppg_tr[tr_mask], imu_tr_shift[tr_mask]], axis=1).astype(np.float32)
        y_tr = y_raw[tri][tr_mask].astype(int)
        X_va = np.stack([ppg_va[va_mask], imu_va_shift[va_mask]], axis=1).astype(np.float32)
        y_va = y_raw[vai][va_mask].astype(int)

        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_va)[:, 1]
        pred = (prob >= 0.5).astype(int)

        m = compute_metrics(y_va, pred, prob)
        m.update({"fold": fold, "n_val": int(len(y_va)), "pos_ratio": float(np.mean(y_va))})
        cv_metrics.append(m)

        y_true_all.append(y_va)
        y_pred_all.append(pred)
        y_prob_all.append(prob)

    y_true_all = np.concatenate(y_true_all) if len(y_true_all) else np.array([], dtype=int)
    y_pred_all = np.concatenate(y_pred_all) if len(y_pred_all) else np.array([], dtype=int)
    cm_cv = confusion_matrix(y_true_all, y_pred_all, labels=[0,1]).astype(int) if y_true_all.size else None
    y_prob_all = np.concatenate(y_prob_all) if len(y_prob_all) else np.array([], dtype=float)

    # Confusion matrix over pooled CV validation predictions
    if y_prob_all.size:
        try:
            _plot_roc_pr(y_true_all, y_prob_all, 'CV_pooled_fused', audit_dir)
        except Exception:
            pass


    # ----------------------------
    # Final detector (global lag on training subjects only) + Holdout evaluation
    # ----------------------------
    clf_final = LogisticRegression(max_iter=2000, class_weight="balanced")
    X_train_final = df_tr2[["ppg_score", "imu_score_shift"]].to_numpy(np.float32)
    y_train_final = df_tr2["y"].to_numpy(int)
    clf_final.fit(X_train_final, y_train_final)

    prob_ho = clf_final.predict_proba(df_ho2[["ppg_score", "imu_score_shift"]].to_numpy(np.float32))[:, 1]
    pred_ho = (prob_ho >= 0.5).astype(int)
    ho_metrics = compute_metrics(df_ho2["y"].to_numpy(int), pred_ho, prob_ho)
    ho_metrics.update({"n": int(len(df_ho2)), "pos_ratio": float(df_ho2["y"].mean())})

    # ----------------------------
    # Audits (A-F)
    # ----------------------------
    audit_dir = Path(args.outdir) / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    def _plot_hist_kde(scores, y, title, out_png, vlines=None):
        """Plot class-wise histogram + optional KDE for a 1D score.

        Parameters
        ----------
        scores : array-like
            1D score array.
        y : array-like
            Binary labels (0=sit, 1=walk/run).
        vlines : list[tuple[float,str]] or None
            Optional vertical lines: [(x, label), ...].
        """
        scores = np.asarray(scores, dtype=float)
        y = np.asarray(y, dtype=int)
        plt.figure()
        bins = 50
        plt.hist(scores[y==0], bins=bins, density=True, alpha=0.5, label="sit (0)")
        plt.hist(scores[y==1], bins=bins, density=True, alpha=0.5, label="walk/run (1)")
        # kde if available
        try:
            from scipy.stats import gaussian_kde
            xs = np.linspace(np.nanmin(scores), np.nanmax(scores), 200)
            if np.sum(y==0) > 5:
                plt.plot(xs, gaussian_kde(scores[y==0][np.isfinite(scores[y==0])])(xs))
            if np.sum(y==1) > 5:
                plt.plot(xs, gaussian_kde(scores[y==1][np.isfinite(scores[y==1])])(xs))
        except Exception:
            pass

        if vlines:
            for x, lab in vlines:
                try:
                    plt.axvline(float(x), linestyle="--", linewidth=1.5, label=lab)
                except Exception:
                    continue

        plt.title(title)
        plt.xlabel("score")
        plt.ylabel("density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()



    def _plot_roc_pr(y_true, y_prob, title_prefix, out_dir: Path):
        """Save ROC and PR curves (if both classes present)."""
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        if y_true.size == 0:
            return
        # ROC
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
            plt.plot([0,1],[0,1], linestyle="--")
            plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"{title_prefix} ROC")
            plt.legend(); plt.tight_layout()
            plt.savefig(out_dir/f"{title_prefix.replace(' ','_')}_ROC.png", dpi=150)
            plt.close()
        except Exception:
            pass
        # PR
        try:
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            plt.figure()
            plt.plot(rec, prec, label=f"AP={ap:.3f}")
            plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"{title_prefix} PR")
            plt.legend(); plt.tight_layout()
            plt.savefig(out_dir/f"{title_prefix.replace(' ','_')}_PR.png", dpi=150)
            plt.close()
        except Exception:
            pass

    # Pre-compute IMU-only threshold on training (BA-opt). Used in (A) and (D).
    imu_tr = df_tr2["imu_score_shift"].to_numpy(np.float32)
    y_tr = y_train_final
    finite_mask = np.isfinite(imu_tr)
    if np.any(finite_mask):
        thr_grid = np.quantile(imu_tr[finite_mask], np.linspace(0.01, 0.99, 99))
        best_thr = float(thr_grid[0]); best_ba = -1.0
        for thr in thr_grid:
            pred = (imu_tr >= thr).astype(int)
            ba = balanced_accuracy_score(y_tr, pred)
            if ba > best_ba:
                best_ba = float(ba); best_thr = float(thr)
    else:
        best_thr = float("nan"); best_ba = float("nan")

    # (A) Distributions on holdout
    _plot_hist_kde(df_ho2["imu_score_shift"].to_numpy(), df_ho2["y"].to_numpy(int),
                   "Holdout IMU_score (shifted) | sit vs walk/run", audit_dir/"A_holdout_imu_score_hist.png",
                   vlines=[(best_thr, "IMU thr (train BA-opt)")])
    _plot_hist_kde(df_ho2["ppg_score"].to_numpy(), df_ho2["y"].to_numpy(int),
                   "Holdout PPG_score | sit vs walk/run", audit_dir/"A_holdout_ppg_score_hist.png")

    # Confusion matrices (CV pooled + holdout)
    cm_ho = confusion_matrix(df_ho2["y"].to_numpy(int), pred_ho, labels=[0,1]).astype(int)
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_ho, display_labels=["sit","walk/run"])
    disp.plot(values_format="d")
    plt.title("Holdout confusion matrix (LogReg fused)")
    plt.tight_layout()
    plt.savefig(audit_dir/"CM_holdout_fused.png", dpi=150)
    plt.close()

    # ROC/PR curves on holdout (where probabilities are available)
    try:
        _plot_roc_pr(df_ho2['y'].to_numpy(int), prob_ho, 'Holdout_fused', audit_dir)
    except Exception:
        pass
    try:
        _plot_roc_pr(df_ho2['y'].to_numpy(int), prob_ho_ppg, 'Holdout_ppg_only', audit_dir)
    except Exception:
        pass

    if (y_true_all.size and (cm_cv is not None)):
        plt.figure()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_cv, display_labels=["sit","walk/run"])
        disp.plot(values_format="d")
        plt.title("CV pooled confusion matrix (LogReg fused)")
        plt.tight_layout()
        plt.savefig(audit_dir/"CM_cv_fused.png", dpi=150)
        plt.close()

    # (B) Feature leakage scan: ensure forbidden fields are not in feature vectors
    forbidden_cols = {"activity","sid","time","timestamp","datetime","peaks","ecg"}
    used_cols_ppg = list(ppg_names)
    used_cols_imu = list(imu_names)
    leakage_findings = {
        "forbidden_cols": sorted(list(forbidden_cols)),
        "ppg_used_cols": used_cols_ppg,
        "imu_used_cols": used_cols_imu,
        "forbidden_in_ppg": sorted([c for c in used_cols_ppg if c in forbidden_cols]),
        "forbidden_in_imu": sorted([c for c in used_cols_imu if c in forbidden_cols]),
    }

    # (C) PPG-only LogReg
    clf_ppg = LogisticRegression(max_iter=2000, class_weight="balanced")
    Xtr_ppg = df_tr2[["ppg_score"]].to_numpy(np.float32)
    clf_ppg.fit(Xtr_ppg, y_train_final)
    prob_ho_ppg = clf_ppg.predict_proba(df_ho2[["ppg_score"]].to_numpy(np.float32))[:,1]
    pred_ho_ppg = (prob_ho_ppg>=0.5).astype(int)
    metrics_ho_ppg = compute_metrics(df_ho2["y"].to_numpy(int), pred_ho_ppg, prob_ho_ppg)

    # (D) IMU-threshold only (no LogReg)
    # Use BA-opt threshold learned on the training split above (best_thr).
    pred_ho_imu = (df_ho2["imu_score_shift"].to_numpy(np.float32) >= best_thr).astype(int)
    metrics_ho_imu_thr = compute_metrics(df_ho2["y"].to_numpy(int), pred_ho_imu, None)
    metrics_ho_imu_thr.update({"thr": float(best_thr), "train_bal_acc_at_thr": float(best_ba)})

    # (E) walk vs run (exclude sit), fused LogReg on scores
    df_tr_wr = df_tr2[df_tr2["activity"].isin(["walk","run"])].copy()
    df_ho_wr = df_ho2[df_ho2["activity"].isin(["walk","run"])].copy()
    metrics_wr = None
    if len(df_tr_wr) and len(df_ho_wr):
        y_tr_wr = (df_tr_wr["activity"]=="run").astype(int).to_numpy()
        X_tr_wr = df_tr_wr[["ppg_score","imu_score_shift"]].to_numpy(np.float32)
        y_ho_wr = (df_ho_wr["activity"]=="run").astype(int).to_numpy()
        X_ho_wr = df_ho_wr[["ppg_score","imu_score_shift"]].to_numpy(np.float32)
        clf_wr = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf_wr.fit(X_tr_wr, y_tr_wr)
        prob_wr = clf_wr.predict_proba(X_ho_wr)[:,1]
        pred_wr = (prob_wr>=0.5).astype(int)
        metrics_wr = compute_metrics(y_ho_wr, pred_wr, prob_wr)
        # confusion
        cm_wr = confusion_matrix(y_ho_wr, pred_wr, labels=[0,1]).astype(int)
        plt.figure()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_wr, display_labels=["walk","run"])
        disp.plot(values_format="d")
        plt.title("Holdout walk vs run confusion matrix (scores)")
        plt.tight_layout()
        plt.savefig(audit_dir/"E_holdout_walk_vs_run_cm.png", dpi=150)
        plt.close()

    # (F) Lag audit: global lag vs fold-specific lags
    lag_audit = {
        "global": {"lag_steps": int(best_steps), "lag_sec": float(best_sec), "corr_by_lag": [float(x) for x in corr_by_lag]},
        "per_fold": cv_lags,
        "note": "CV lags are learned on each fold's train split; global lag is learned on all training subjects (excluding holdout) for the final model."
    }

    # Save audit summary JSON
    audit_summary = {
        "A": {
            "holdout_hist_imu_png": str((audit_dir/"A_holdout_imu_score_hist.png").as_posix()),
            "holdout_hist_ppg_png": str((audit_dir/"A_holdout_ppg_score_hist.png").as_posix()),
        },
        "B": leakage_findings,
        "C": {"holdout_metrics_ppg_only": metrics_ho_ppg},
        "D": {"holdout_metrics_imu_threshold": metrics_ho_imu_thr},
        "E": {"holdout_metrics_walk_vs_run": metrics_wr},
        "F": lag_audit,
        "cm_cv": cm_cv.tolist() if isinstance(cm_cv, np.ndarray) else cm_cv,
        "cm_holdout": cm_ho.tolist() if isinstance(cm_ho, np.ndarray) else cm_ho,
    }
    (audit_dir/"audit_summary.json").write_text(json.dumps(_json_sanitize(audit_summary), ensure_ascii=False, indent=2), encoding="utf-8")

    # ----------------------------
    # Final bundle (detector)
    # ----------------------------
    bundle = {
        "version": "v8_scores_audit",
        "win_cfg": dict(cfg),
        "lag_best_steps": int(best_steps),
        "lag_best_sec": float(best_sec),
        "corr_by_lag": [float(x) for x in corr_by_lag],
        "logreg_coef": clf_final.coef_.tolist(),
        "logreg_intercept": clf_final.intercept_.tolist(),
        "logreg_threshold": 0.5,
        "cv_metrics": cv_metrics,
        "cv_lags": cv_lags,
        "cm_cv": cm_cv.tolist() if isinstance(cm_cv, np.ndarray) else cm_cv,
        "holdout_metrics": ho_metrics,
        "cm_holdout": cm_ho.tolist() if isinstance(cm_ho, np.ndarray) else cm_ho,
        "holdout_subjects": ho_subjects,
        "train_subjects": tr_subjects,
        # Persist the exact feature names used to compute window-level scores.
        "ppg_feature_names": list(ppg_names),
        "imu_feature_names": list(imu_names),
        "audit_dir": str(audit_dir.as_posix()),
    }
    (outdir / "detector_v8_summary.json").write_text(json.dumps(_json_sanitize(bundle), ensure_ascii=False, indent=2), encoding="utf-8")

    np.savez(
        outdir / "detector_v8_bundle.npz",
        mu_ppg=mu_p, cov_ppg=cov_p,
        mu_imu=mu_i, cov_imu=cov_i,
        lag_best_steps=np.asarray([best_steps], dtype=np.int32),
        coef=clf_final.coef_.astype(np.float32),
        intercept=clf_final.intercept_.astype(np.float32),
        ppg_feature_names=np.asarray(ppg_names, dtype=object),
        imu_feature_names=np.asarray(imu_names, dtype=object),
    )

    dt = time.time() - t0
    print(f"[Done] Detector v8 finished in {dt/60:.1f} min. Outputs: {outdir}")

if __name__ == "__main__":
    main()