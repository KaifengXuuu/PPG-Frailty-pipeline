"""pttppg_stage1_detector.py

Stage 1: Subject-level no-leak motion detector for the PhysioNet Pulse Transit Time PPG dataset.

Design goals (per user spec)
---------------------------
* Binary activity classifier: sit=0 vs walk/run=1.
* Use unified inputs (same 8 raw channels available for Stage 2):
  - pleth_1 (red distal), pleth_2 (IR distal)
  - a_dyn_x, a_dyn_y, a_dyn_z (dynamic accel after gravity removal, m/s^2)
  - gyro_x, gyro_y, gyro_z (rad/s)
* Build per-window feature vectors for PPG and IMU, then compute two independent scores:
  - PPG_score  = w_ppg^T * z(ppg_features)
  - IMU_score  = w_imu^T * z(imu_features)
  where weights are learned from subject-level training folds.
* Find lag (in [-5s, +5s]) maximizing association between PPG_score and IMU_score on training folds.
* Final detector uses BOTH scores with learned thresholds:
  motion = (IMU_score_lagged > tau_imu) OR/AND (PPG_score > tau_ppg)
  (both are reported).
* Optional AE fusion is NOT in stage-1; stage-2 can add AE gating.

Outputs
-------
* JSON with learned weights, thresholds, best lag, and metrics for CV and holdout.
* PNG confusion matrices for CV folds and holdout (OR and AND rules).

Run example
-----------
python pttppg_stage1_detector.py \
  --data_root ./physionet.org/files/pulse-transit-time-ppg/1.1.0 \
  --outdir results_stage1 \
  --fs 500 --win 6 --hop 1 --train_size 0.8 --n_splits 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def _try_import_scipy():
    try:
        from scipy import signal  # type: ignore
        return signal
    except Exception as e:
        raise RuntimeError("scipy is required for this script") from e


signal = _try_import_scipy()


def zscore_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = float(np.mean(x))
    s = float(np.std(x)) + 1e-8
    return (x - m) / s


def bandpass_ppg(x: np.ndarray, fs: float, lowcut: float = 0.5, highcut: float = 8.0, order: int = 3) -> np.ndarray:
    """Band-pass for PPG before windowing (user requirement)."""
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return signal.filtfilt(b, a, x).astype(np.float32)


def gyro_deg2rad(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32) * (math.pi / 180.0)).astype(np.float32)


def split_windows(n: int, fs: float, win_sec: float, hop_sec: float):
    w = int(round(win_sec * fs))
    h = int(round(hop_sec * fs))
    if w <= 0 or h <= 0:
        raise ValueError("win/hop must be positive")
    for s in range(0, max(1, n - w + 1), h):
        e = s + w
        if e <= n:
            yield s, e


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def spectral_entropy(psd: np.ndarray) -> float:
    psd = np.asarray(psd, dtype=np.float64)
    psd = np.maximum(psd, 1e-12)
    p = psd / np.sum(psd)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def bandpowers(x: np.ndarray, fs: float, bands=((0.1, 0.5), (0.5, 3.0), (3.0, 8.0))) -> List[float]:
    f, pxx = signal.welch(x, fs=fs, nperseg=min(len(x), int(fs * 2)))
    out = []
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        out.append(float(np.trapz(pxx[m], f[m])) if np.any(m) else 0.0)
    return out


def dom_freq(x: np.ndarray, fs: float, lo: float = 0.1, hi: float = 8.0) -> float:
    f, pxx = signal.welch(x, fs=fs, nperseg=min(len(x), int(fs * 2)))
    m = (f >= lo) & (f <= hi)
    if not np.any(m):
        return float("nan")
    fi = f[m]
    pi = pxx[m]
    return float(fi[int(np.argmax(pi))])


def time_stats(x: np.ndarray) -> Tuple[float, float, float, float, float]:
    x = np.asarray(x, dtype=np.float32)
    med = float(np.median(x))
    q1 = float(np.percentile(x, 25))
    q3 = float(np.percentile(x, 75))
    return (
        float(np.mean(x)),
        float(np.std(x) + 1e-8),
        med,
        float(q3 - q1),
        rms(x),
    )


PPG_FEATURE_NAMES = [
    "PPG_ts_mean",
    "PPG_ts_std",
    "PPG_ts_med",
    "PPG_ts_iqr",
    "PPG_ts_rms",
    "PPG_bp_0.1_0.5",
    "PPG_bp_0.5_3",
    "PPG_bp_3_8",
    "PPG_spec_entropy",
    "PPG_dom_freq",
]


IMU_FEATURE_NAMES = [
    # Time stats
    "AccMag_ts_mean",
    "AccMag_ts_std",
    "AccMag_ts_med",
    "AccMag_ts_iqr",
    "AccMag_ts_rms",
    "GyroMag_ts_mean",
    "GyroMag_ts_std",
    "GyroMag_ts_med",
    "GyroMag_ts_iqr",
    "GyroMag_ts_rms",
    "JerkMag_ts_mean",
    "JerkMag_ts_std",
    "JerkMag_ts_med",
    "JerkMag_ts_iqr",
    "JerkMag_ts_rms",
    # Bandpowers
    "AccMag_bp_0.1_0.5",
    "AccMag_bp_0.5_3",
    "AccMag_bp_3_8",
    "GyroMag_bp_0.1_0.5",
    "GyroMag_bp_0.5_3",
    "GyroMag_bp_3_8",
    # Spectral shape
    "AccMag_spec_entropy",
    "AccMag_dom_freq",
    "GyroMag_spec_entropy",
    "GyroMag_dom_freq",
    "JerkMag_spec_entropy",
    "JerkMag_dom_freq",
]


def compute_window_features(
    p1: np.ndarray,
    p2: np.ndarray,
    a_dyn: np.ndarray,  # (T,3)
    gyro: np.ndarray,   # (T,3)
    fs: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (ppg_feat[10], imu_feat[27])."""

    # PPG: use pleth_2 (IR) as primary; pleth_1 is auxiliary in Stage 2.
    ppg = p2
    ts = time_stats(ppg)
    bp = bandpowers(ppg, fs)
    f, pxx = signal.welch(ppg, fs=fs, nperseg=min(len(ppg), int(fs * 2)))
    m = (f >= 0.1) & (f <= 8.0)
    se = spectral_entropy(pxx[m] if np.any(m) else pxx)
    df = dom_freq(ppg, fs)
    ppg_feat = np.array([*ts, *bp, se, df], dtype=np.float32)

    acc_mag = np.sqrt(np.sum(a_dyn * a_dyn, axis=1))
    gyro_mag = np.sqrt(np.sum(gyro * gyro, axis=1))
    # jerk magnitude from dynamic accel
    jerk = np.diff(a_dyn, axis=0) * fs
    jerk_mag = np.sqrt(np.sum(jerk * jerk, axis=1)) if len(jerk) else np.zeros_like(acc_mag)

    acc_ts = time_stats(acc_mag)
    gyro_ts = time_stats(gyro_mag)
    jerk_ts = time_stats(jerk_mag)

    acc_bp = bandpowers(acc_mag, fs)
    gyro_bp = bandpowers(gyro_mag, fs)

    f1, p1x = signal.welch(acc_mag, fs=fs, nperseg=min(len(acc_mag), int(fs * 2)))
    m1 = (f1 >= 0.1) & (f1 <= 8.0)
    acc_se = spectral_entropy(p1x[m1] if np.any(m1) else p1x)
    acc_df = dom_freq(acc_mag, fs)

    f2, p2x = signal.welch(gyro_mag, fs=fs, nperseg=min(len(gyro_mag), int(fs * 2)))
    m2 = (f2 >= 0.1) & (f2 <= 8.0)
    gyro_se = spectral_entropy(p2x[m2] if np.any(m2) else p2x)
    gyro_df = dom_freq(gyro_mag, fs)

    f3, p3x = signal.welch(jerk_mag, fs=fs, nperseg=min(len(jerk_mag), int(fs * 2)))
    m3 = (f3 >= 0.1) & (f3 <= 8.0)
    jerk_se = spectral_entropy(p3x[m3] if np.any(m3) else p3x)
    jerk_df = dom_freq(jerk_mag, fs)

    imu_feat = np.array(
        [*acc_ts, *gyro_ts, *jerk_ts, *acc_bp, *gyro_bp, acc_se, acc_df, gyro_se, gyro_df, jerk_se, jerk_df],
        dtype=np.float32,
    )
    return ppg_feat, imu_feat


def parse_activity_from_filename(stem: str) -> str:
    stem = stem.lower()
    if "_sit" in stem:
        return "sit"
    if "_walk" in stem:
        return "walk"
    if "_run" in stem:
        return "run"
    raise ValueError(f"Cannot remind activity from filename: {stem}")


def load_physionet_csv(data_root: Path, fs: float) -> Dict[str, List[dict]]:
    """Load CSVs; perform band-pass on pleth_1/2 before windowing."""
    csv_dir = data_root / "csv"
    if not csv_dir.exists():
        csv_dir = data_root / "files" / "pulse-transit-time-ppg" / "1.1.0" / "csv"
    paths = sorted([p for p in csv_dir.glob("s*_*.csv") if p.name != "subjects_info.csv"])
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {csv_dir}")

    def pick(df: pd.DataFrame, candidates: List[str]) -> str:
        for c in candidates:
            if c in df.columns:
                return c
        raise KeyError(f"Missing columns among {candidates}")

    out: Dict[str, List[dict]] = {}
    for p in tqdm(paths, desc="Load CSV", leave=True):
        df = pd.read_csv(p)
        rec = {}
        rec["subject"] = p.stem.split("_")[0]
        rec["activity"] = parse_activity_from_filename(p.stem)

        # time (shifted datetime or numeric)
        if "time" in df.columns:
            t = pd.to_datetime(df["time"], errors="coerce")
            if t.notna().all():
                rec["time"] = (t - t.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float32)
            else:
                rec["time"] = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=np.float32)
        else:
            rec["time"] = (np.arange(len(df), dtype=np.float32) / float(fs))

        # channels
        rec["pleth_1"] = pd.to_numeric(df[pick(df, ["pleth_1"])], errors="coerce").to_numpy(np.float32)
        rec["pleth_2"] = pd.to_numeric(df[pick(df, ["pleth_2"])], errors="coerce").to_numpy(np.float32)

        # IMU
        rec["a_x"] = pd.to_numeric(df[pick(df, ["a_x"])], errors="coerce").to_numpy(np.float32)
        rec["a_y"] = pd.to_numeric(df[pick(df, ["a_y"])], errors="coerce").to_numpy(np.float32)
        rec["a_z"] = pd.to_numeric(df[pick(df, ["a_z"])], errors="coerce").to_numpy(np.float32)
        rec["g_x"] = pd.to_numeric(df[pick(df, ["g_x"])], errors="coerce").to_numpy(np.float32)
        rec["g_y"] = pd.to_numeric(df[pick(df, ["g_y"])], errors="coerce").to_numpy(np.float32)
        rec["g_z"] = pd.to_numeric(df[pick(df, ["g_z"])], errors="coerce").to_numpy(np.float32)

        # Pre-filter PPG
        rec["pleth_1"] = bandpass_ppg(rec["pleth_1"], fs=fs)
        rec["pleth_2"] = bandpass_ppg(rec["pleth_2"], fs=fs)

        # Gyro deg/s -> rad/s
        rec["g_x"] = gyro_deg2rad(rec["g_x"])
        rec["g_y"] = gyro_deg2rad(rec["g_y"])
        rec["g_z"] = gyro_deg2rad(rec["g_z"])

        out.setdefault(rec["subject"], []).append(rec)
    return out


def remove_gravity_simple(acc_g: np.ndarray, fs: float, cutoff_hz: float = 0.3) -> np.ndarray:
    """Remove gravity component using a low-pass estimate; return dynamic accel in m/s^2."""
    # acc_g is in g; convert to m/s^2
    acc_ms2 = np.asarray(acc_g, dtype=np.float32) * 9.81
    nyq = 0.5 * fs
    b, a = signal.butter(2, cutoff_hz / nyq, btype="low")
    g_est = signal.filtfilt(b, a, acc_ms2, axis=0).astype(np.float32)
    return (acc_ms2 - g_est).astype(np.float32)


def build_windows_for_subjects(
    sub2recs: Dict[str, List[dict]],
    fs: float,
    win_sec: float,
    hop_sec: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ppgX, imuX, y) aligned per-window for a set of subjects.

    y: sit=0, walk/run=1 (label from filename only).
    """
    ppg_feats = []
    imu_feats = []
    y = []

    for sid, recs in sub2recs.items():
        for r in recs:
            p1 = r["pleth_1"]
            p2 = r["pleth_2"]
            acc = np.column_stack([r["a_x"], r["a_y"], r["a_z"]]).astype(np.float32)
            gyro = np.column_stack([r["g_x"], r["g_y"], r["g_z"]]).astype(np.float32)
            a_dyn = remove_gravity_simple(acc, fs)

            n = min(len(p2), len(a_dyn), len(gyro))
            p1, p2 = p1[:n], p2[:n]
            a_dyn, gyro = a_dyn[:n], gyro[:n]

            lab = 0 if r["activity"] == "sit" else 1

            for s, e in split_windows(n, fs, win_sec, hop_sec):
                ppg_f, imu_f = compute_window_features(p1[s:e], p2[s:e], a_dyn[s:e], gyro[s:e], fs)
                ppg_feats.append(ppg_f)
                imu_feats.append(imu_f)
                y.append(lab)

    return (
        np.asarray(ppg_feats, dtype=np.float32),
        np.asarray(imu_feats, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
    )


def learn_score_model(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, StandardScaler, LogisticRegression]:
    """Fit logistic regression on standardized features; return weights in original feature space."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=200, class_weight="balanced", solver="lbfgs")
    clf.fit(Xs, y)
    # Convert to weights on z-scored features (we keep scaler+clf for reuse).
    w = clf.coef_.reshape(-1).astype(np.float32)
    return w, scaler, clf


def score_windows(X: np.ndarray, scaler: StandardScaler, clf: LogisticRegression) -> np.ndarray:
    Xs = scaler.transform(X)
    # probability of class 1
    p = clf.predict_proba(Xs)[:, 1]
    return p.astype(np.float32)


def find_best_lag(ppg_score: np.ndarray, imu_score: np.ndarray, fs_hop: float, lag_min_sec: float, lag_max_sec: float) -> Tuple[int, float]:
    """Maximize absolute Pearson correlation between ppg_score and time-shifted imu_score."""
    lag_min_steps = int(round(lag_min_sec * fs_hop))
    lag_max_steps = int(round(lag_max_sec * fs_hop))
    best_lag = 0
    best_r = -1.0
    for lag in range(lag_min_steps, lag_max_steps + 1):
        if lag < 0:
            a = ppg_score[:lag]
            b = imu_score[-lag:]
        elif lag > 0:
            a = ppg_score[lag:]
            b = imu_score[:-lag]
        else:
            a = ppg_score
            b = imu_score
        if len(a) < 10:
            continue
        ra = a - float(np.mean(a))
        rb = b - float(np.mean(b))
        denom = float(np.sqrt(np.sum(ra * ra) * np.sum(rb * rb)) + 1e-12)
        r = float(np.sum(ra * rb) / denom)
        if abs(r) > best_r:
            best_r = abs(r)
            best_lag = lag
    return best_lag, best_lag / fs_hop


def apply_lag(x: np.ndarray, lag_steps: int) -> np.ndarray:
    if lag_steps == 0:
        return x
    if lag_steps > 0:
        return np.concatenate([np.full(lag_steps, x[0], dtype=x.dtype), x[:-lag_steps]])
    lag = -lag_steps
    return np.concatenate([x[lag:], np.full(lag, x[-1], dtype=x.dtype)])


def pick_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """Pick threshold maximizing balanced accuracy."""
    # Search on quantiles for robustness
    qs = np.linspace(0.05, 0.95, 37)
    cand = np.quantile(scores, qs)
    best_t = float(cand[len(cand) // 2])
    best_ba = -1.0
    for t in cand:
        pred = (scores >= t).astype(int)
        ba = balanced_accuracy_score(y, pred)
        if ba > best_ba:
            best_ba = float(ba)
            best_t = float(t)
    return best_t


def metrics_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def save_confusion_png(cm: np.ndarray, labels: List[str], title: str, path: Path):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4.2, 3.6), dpi=150)
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_ylabel("True")
    ax.set_xlabel("Pred")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(int(v)), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def split_subjects(sub_ids: List[str], train_size: float, seed: int = 42) -> Tuple[List[str], List[str]]:
    rng = np.random.default_rng(seed)
    sub_ids = sorted(sub_ids)
    rng.shuffle(sub_ids)
    n_tr = max(1, int(round(len(sub_ids) * train_size)))
    return sub_ids[:n_tr], sub_ids[n_tr:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="results_stage1")
    ap.add_argument("--fs", type=float, default=500.0)
    ap.add_argument("--win", type=float, default=6.0)
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--split_mode", type=str, default="both", choices=["both", "kfold", "holdout"])
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--train_size", type=float, default=0.8)
    ap.add_argument("--lag_min_sec", type=float, default=-5.0)
    ap.add_argument("--lag_max_sec", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sub2recs = load_physionet_csv(Path(args.data_root), fs=args.fs)
    subjects = sorted(sub2recs.keys())
    if len(subjects) < 4:
        raise RuntimeError("Need at least 4 subjects for subject-level split")

    # Holdout subjects are selected BEFORE any feature extraction to avoid leakage.
    tr_subs, ho_subs = split_subjects(subjects, train_size=args.train_size, seed=args.seed)
    sub2recs_tr = {s: sub2recs[s] for s in tr_subs}
    sub2recs_ho = {s: sub2recs[s] for s in ho_subs}

    win_cfg = {
        "fs": float(args.fs),
        "win_sec": float(args.win),
        "hop_sec": float(args.hop),
        "lag_min_sec": float(args.lag_min_sec),
        "lag_max_sec": float(args.lag_max_sec),
    }

    # Build window-level datasets for train subjects and holdout subjects.
    ppgX_tr, imuX_tr, y_tr = build_windows_for_subjects(sub2recs_tr, args.fs, args.win, args.hop)
    ppgX_ho, imuX_ho, y_ho = build_windows_for_subjects(sub2recs_ho, args.fs, args.win, args.hop)

    # Group labels at window-level (subject id repeated per record windows) are not trivial here.
    # For Stage 1, we do GroupKFold at the SUBJECT level by folding subjects directly.
    folds = []
    if args.split_mode in ("both", "kfold"):
        gkf = GroupKFold(n_splits=min(args.n_splits, len(tr_subs)))
        # build per-subject window index ranges
        # We reconstruct windows per subject to create groups exactly.
        subj_windows = []
        subj_y = []
        subj_ppg = []
        subj_imu = []
        groups = []
        for sid in tr_subs:
            ppgX_s, imuX_s, y_s = build_windows_for_subjects({sid: sub2recs[sid]}, args.fs, args.win, args.hop)
            subj_ppg.append(ppgX_s)
            subj_imu.append(imuX_s)
            subj_y.append(y_s)
            groups.append(np.full(len(y_s), sid, dtype=object))
        ppgX = np.concatenate(subj_ppg, axis=0)
        imuX = np.concatenate(subj_imu, axis=0)
        y = np.concatenate(subj_y, axis=0)
        g = np.concatenate(groups, axis=0)
        idx = np.arange(len(y))
        for tr_idx, va_idx in gkf.split(idx, y=y, groups=g):
            folds.append((tr_idx, va_idx, ppgX, imuX, y, g))
    else:
        folds = []

    results = {
        "ppg_feature_names": PPG_FEATURE_NAMES,
        "imu_feature_names": IMU_FEATURE_NAMES,
        "win_cfg": win_cfg,
        "cv_folds": [],
        "holdout": None,
    }

    # --- CV ---
    if folds:
        for k, (tr_idx, va_idx, ppgX, imuX, y, g) in enumerate(tqdm(folds, desc="Detector CV folds", leave=True), start=1):
            # Train score models
            _, ppg_scaler, ppg_clf = learn_score_model(ppgX[tr_idx], y[tr_idx])
            _, imu_scaler, imu_clf = learn_score_model(imuX[tr_idx], y[tr_idx])

            ppg_s_tr = score_windows(ppgX[tr_idx], ppg_scaler, ppg_clf)
            imu_s_tr = score_windows(imuX[tr_idx], imu_scaler, imu_clf)
            hop_rate = 1.0 / float(args.hop)
            lag_steps, lag_sec = find_best_lag(ppg_s_tr, imu_s_tr, hop_rate, args.lag_min_sec, args.lag_max_sec)

            # thresholds fitted on training split
            tau_ppg = pick_threshold(ppg_s_tr, y[tr_idx])
            tau_imu = pick_threshold(apply_lag(imu_s_tr, lag_steps), y[tr_idx])

            # Evaluate on validation
            ppg_s_va = score_windows(ppgX[va_idx], ppg_scaler, ppg_clf)
            imu_s_va = apply_lag(score_windows(imuX[va_idx], imu_scaler, imu_clf), lag_steps)
            y_va = y[va_idx]

            pred_or = ((ppg_s_va >= tau_ppg) | (imu_s_va >= tau_imu)).astype(int)
            pred_and = ((ppg_s_va >= tau_ppg) & (imu_s_va >= tau_imu)).astype(int)

            cm_or = confusion_matrix(y_va, pred_or)
            cm_and = confusion_matrix(y_va, pred_and)
            save_confusion_png(cm_or, ["sit", "motion"], f"CV fold {k} | OR", outdir / f"cm_cv_fold{k:02d}_or.png")
            save_confusion_png(cm_and, ["sit", "motion"], f"CV fold {k} | AND", outdir / f"cm_cv_fold{k:02d}_and.png")

            fold_res = {
                "fold": k,
                "lag_best_steps": int(lag_steps),
                "lag_best_sec": float(lag_sec),
                "tau_ppg": float(tau_ppg),
                "tau_imu": float(tau_imu),
                "metrics_or": metrics_binary(y_va, pred_or),
                "metrics_and": metrics_binary(y_va, pred_and),
                "n_val": int(len(y_va)),
                "pos_ratio": float(np.mean(y_va)),
            }
            results["cv_folds"].append(fold_res)

        # lag summary
        lags = np.array([f["lag_best_sec"] for f in results["cv_folds"]], dtype=float)
        results["lag_median_sec"] = float(np.median(lags))
        results["lag_iqr_sec"] = float(np.percentile(lags, 75) - np.percentile(lags, 25))

    # --- Holdout: train on ALL train subjects, test on holdout subjects ---
    if args.split_mode in ("both", "holdout"):
        _, ppg_scaler, ppg_clf = learn_score_model(ppgX_tr, y_tr)
        _, imu_scaler, imu_clf = learn_score_model(imuX_tr, y_tr)

        ppg_s_tr = score_windows(ppgX_tr, ppg_scaler, ppg_clf)
        imu_s_tr = score_windows(imuX_tr, imu_scaler, imu_clf)
        hop_rate = 1.0 / float(args.hop)
        lag_steps, lag_sec = find_best_lag(ppg_s_tr, imu_s_tr, hop_rate, args.lag_min_sec, args.lag_max_sec)
        tau_ppg = pick_threshold(ppg_s_tr, y_tr)
        tau_imu = pick_threshold(apply_lag(imu_s_tr, lag_steps), y_tr)

        ppg_s_ho = score_windows(ppgX_ho, ppg_scaler, ppg_clf)
        imu_s_ho = apply_lag(score_windows(imuX_ho, imu_scaler, imu_clf), lag_steps)
        pred_or = ((ppg_s_ho >= tau_ppg) | (imu_s_ho >= tau_imu)).astype(int)
        pred_and = ((ppg_s_ho >= tau_ppg) & (imu_s_ho >= tau_imu)).astype(int)

        cm_or = confusion_matrix(y_ho, pred_or)
        cm_and = confusion_matrix(y_ho, pred_and)
        save_confusion_png(cm_or, ["sit", "motion"], "Holdout | OR", outdir / "cm_holdout_or.png")
        save_confusion_png(cm_and, ["sit", "motion"], "Holdout | AND", outdir / "cm_holdout_and.png")

        results["holdout"] = {
            "subjects_train": tr_subs,
            "subjects_holdout": ho_subs,
            "lag_best_steps": int(lag_steps),
            "lag_best_sec": float(lag_sec),
            "tau_ppg": float(tau_ppg),
            "tau_imu": float(tau_imu),
            "metrics_or": metrics_binary(y_ho, pred_or),
            "metrics_and": metrics_binary(y_ho, pred_and),
            "n_holdout": int(len(y_ho)),
            "pos_ratio": float(np.mean(y_ho)),
            # Persist the models for reuse (sklearn objects are not JSON-serializable)
            "artifacts": {
                "ppg_scaler_npz": "ppg_scaler.npz",
                "imu_scaler_npz": "imu_scaler.npz",
                "ppg_clf_npz": "ppg_clf.npz",
                "imu_clf_npz": "imu_clf.npz",
            },
        }

        # Save lightweight artifacts for CPU-only reuse (no pickle): scaler params + LR weights.
        def save_scaler_npz(path: Path, scaler: StandardScaler):
            np.savez(path, mean_=scaler.mean_.astype(np.float32), scale_=scaler.scale_.astype(np.float32))

        def save_lr_npz(path: Path, clf: LogisticRegression):
            np.savez(path, coef_=clf.coef_.astype(np.float32), intercept_=clf.intercept_.astype(np.float32))

        save_scaler_npz(outdir / "ppg_scaler.npz", ppg_scaler)
        save_scaler_npz(outdir / "imu_scaler.npz", imu_scaler)
        save_lr_npz(outdir / "ppg_clf.npz", ppg_clf)
        save_lr_npz(outdir / "imu_clf.npz", imu_clf)

    (outdir / "detector_stage1.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    dt = time.time() - t0
    print(f"[Stage1 Detector] Done in {dt/60:.1f} min. Results at: {outdir}")


if __name__ == "__main__":
    main()
