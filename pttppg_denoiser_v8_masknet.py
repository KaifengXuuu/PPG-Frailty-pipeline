#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pttppg_denoiser_v8_masknet.py

Denoiser training script (walk/run separately) with additional evaluation metrics:
- Peak timing MAE vs ECG peaks (with subject-specific phase compensation factor a)
- Peak F1 (matching PPG peaks to ECG peaks within tolerance)
- SNR improvement (time-domain)

Key points
----------
- Subject-specific 'a' is estimated by GRID SEARCH on TRAINING records only (no-leak).
  We store an "a_table" mapping {subject -> {walk/run -> a}} for inference-time lookup.
- STFT magnitude mask acts on pleth_2 (IR). Phase is preserved from pleth_2.
- Inputs: 8 raw time channels (pleth_1, pleth_2, a_dyn_xyz[m/s^2], gyro_xyz[rad/s])
  + window features appended as constant channels over time (repeat across T).
"""

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit, GroupKFold

try:
    import funcs
except Exception:
    funcs = None

# ----------------------------
# Basic signal utilities
# ----------------------------
def bandpass_ppg(x: np.ndarray, fs: float, low: float = 0.5, high: float = 8.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if funcs is not None and hasattr(funcs, "bandpass_filter"):
        return funcs.bandpass_filter(x, lowcut=low, highcut=high, fs=fs, order=3).astype(np.float32)
    try:
        from scipy.signal import butter, filtfilt
        nyq = 0.5*fs
        b,a = butter(3, [low/nyq, high/nyq], btype="band")
        return filtfilt(b,a,x).astype(np.float32)
    except Exception:
        return x

def lowpass(x: np.ndarray, fs: float, cutoff: float = 0.3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    try:
        from scipy.signal import butter, filtfilt
        nyq = 0.5*fs
        b,a = butter(2, cutoff/nyq, btype="low")
        return filtfilt(b,a,x).astype(np.float32)
    except Exception:
        w = max(3, int(fs*1.0))
        k = np.ones(w, dtype=np.float32)/w
        return np.convolve(x,k,mode="same").astype(np.float32)

def zscore_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = float(np.nanmean(x))
    s = float(np.nanstd(x) + 1e-8)
    return (x - m) / s

def window_slices(n: int, fs: float, win_sec: float, hop_sec: float):
    w = int(round(win_sec*fs))
    h = int(round(hop_sec*fs))
    w = max(2,w); h = max(1,h)
    for s in range(0, max(1, n-w+1), h):
        yield s, s+w

# ----------------------------
# Feature extraction (same dimensionality as detector)
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
            out.append(float(np.trapz(p[m], f[m])) if np.any(m) else 0.0)
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

def extract_window_features(ppg_ir: np.ndarray,
                            acc_dyn_xyz_mps2: np.ndarray,
                            gyro_xyz_rads: np.ndarray,
                            fs: float) -> Tuple[np.ndarray, List[str]]:
    feat = []
    names = []

    feat.extend(time_stats(ppg_ir))
    names += ["PPG_ts_mean","PPG_ts_std","PPG_ts_med","PPG_ts_iqr","PPG_ts_rms"]
    feat.extend(bandpowers(ppg_ir, fs))
    names += ["PPG_bp_0.1_0.5","PPG_bp_0.5_3","PPG_bp_3_8"]
    ent, dom = spectral_entropy_and_domfreq(ppg_ir, fs)
    feat += [ent, dom]
    names += ["PPG_spec_entropy","PPG_dom_freq"]

    acc_mag = np.linalg.norm(acc_dyn_xyz_mps2, axis=1)
    gyro_mag = np.linalg.norm(gyro_xyz_rads, axis=1)
    jerk = np.diff(acc_mag, prepend=acc_mag[:1]) * fs

    for name, sig in [("AccMag", acc_mag), ("GyroMag", gyro_mag), ("JerkMag", jerk)]:
        feat.extend(time_stats(sig))
        names += [f"{name}_ts_mean",f"{name}_ts_std",f"{name}_ts_med",f"{name}_ts_iqr",f"{name}_ts_rms"]

    for name, sig in [("AccMag", acc_mag), ("GyroMag", gyro_mag)]:
        feat.extend(bandpowers(sig, fs))
        names += [f"{name}_bp_0.1_0.5",f"{name}_bp_0.5_3",f"{name}_bp_3_8"]
        ent, dom = spectral_entropy_and_domfreq(sig, fs)
        feat += [ent, dom]
        names += [f"{name}_spec_entropy",f"{name}_dom_freq"]

    ent, dom = spectral_entropy_and_domfreq(jerk, fs)
    feat += [ent, dom]
    names += ["JerkMag_spec_entropy","JerkMag_dom_freq"]

    return np.asarray(feat, dtype=np.float32), names

# ----------------------------
# Data loading
# ----------------------------
CAND = {
    "time": ["time","Time","t"],
    "ecg": ["ecg","ECG"],
    "peaks": ["peaks","Rpeaks","r_peaks"],
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

        for k in ["pleth_1","pleth_2","a_x","a_y","a_z","g_x","g_y","g_z","ecg","peaks"]:
            if m[k] is None:
                continue
            rec[k] = pd.to_numeric(df[m[k]], errors="coerce").to_numpy(dtype=np.float32)

        sid = p.stem.split("_")[0]
        out.setdefault(sid, []).append(rec)
    return out

# ----------------------------
# Model: STFT magnitude mask net
# ----------------------------
class MaskNet(nn.Module):
    """Predicts a magnitude mask M in (0,1) with shape (B,1,F,Tt)."""
    def __init__(self, in_ch: int, base: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(in_ch, base, 5, padding=2), nn.ReLU(True),
            nn.Conv1d(base, base, 5, padding=2), nn.ReLU(True),
            nn.MaxPool1d(2),
            nn.Conv1d(base, base*2, 5, padding=2), nn.ReLU(True),
            nn.MaxPool1d(2),
            nn.Conv1d(base*2, base*4, 3, padding=1), nn.ReLU(True),
        )
        self.head = nn.Sequential(
            nn.Conv1d(base*4, base*2, 1), nn.ReLU(True),
            nn.Conv1d(base*2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x_time: torch.Tensor, mag_shape: Tuple[int,int,int]) -> torch.Tensor:
        z = self.enc(x_time)     # (B,*,T')
        m1 = self.head(z)        # (B,1,T')
        B,F,Tt = mag_shape
        m1 = F.interpolate(m1, size=Tt, mode="linear", align_corners=False)  # (B,1,Tt)
        M = m1.unsqueeze(2).repeat(1,1,F,1)  # (B,1,F,Tt)
        return M

# ----------------------------
# STFT helpers + regularizers
# ----------------------------
def stft_mag_phase(x: torch.Tensor, n_fft: int, hop: int, win: int) -> Tuple[torch.Tensor, torch.Tensor]:
    window = torch.hann_window(win, device=x.device)
    X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True)
    return torch.abs(X), torch.angle(X)

def istft_from_mag_phase(mag: torch.Tensor, phase: torch.Tensor, n_fft: int, hop: int, win: int, length: int) -> torch.Tensor:
    window = torch.hann_window(win, device=mag.device)
    X = torch.polar(mag, phase)
    return torch.istft(X, n_fft=n_fft, hop_length=hop, win_length=win, window=window, length=length)

def freq_smoothness_reg(M: torch.Tensor) -> torch.Tensor:
    """Frequency-direction TV regularizer on mask M (B,1,F,Tt)."""
    return torch.abs(M[:,:,1:,:] - M[:,:,:-1,:]).mean()

def shape_loss_to_sit(y: torch.Tensor, sit_template: torch.Tensor) -> torch.Tensor:
    yt = (y - y.mean(dim=1, keepdim=True)) / (y.std(dim=1, keepdim=True) + 1e-8)
    st = (sit_template - sit_template.mean()) / (sit_template.std() + 1e-8)
    st = st[None,:].expand_as(yt)
    return F.l1_loss(yt, st)

# ----------------------------
# Peak metrics (uses funcs.aboypp_peak_hr if available)
# ----------------------------
def ecg_peak_indices_from_column(peaks_col: np.ndarray) -> np.ndarray:
    peaks_col = np.asarray(peaks_col).astype(np.float32)
    return np.where(peaks_col > 0.5)[0].astype(int)

def ppg_peak_indices_aboy(ppg: np.ndarray, fs: float) -> np.ndarray:
    if funcs is None or not hasattr(funcs, "aboypp_peak_hr"):
        from scipy.signal import find_peaks
        dist = int(fs * 60 / 180)
        pk, _ = find_peaks(ppg, distance=max(1, dist))
        return pk.astype(int)
    out = funcs.aboypp_peak_hr(ppg, fs=int(fs))
    return np.asarray(out.get("peaks_all", []), dtype=int)

def match_peaks_f1(ppg_peaks: np.ndarray, ecg_peaks: np.ndarray, tol_s: float, fs: float) -> Tuple[float,float,float,float]:
    tol = int(round(tol_s * fs))
    ppg_peaks = np.asarray(ppg_peaks, dtype=int)
    ecg_peaks = np.asarray(ecg_peaks, dtype=int)
    if len(ppg_peaks)==0 or len(ecg_peaks)==0:
        return 0.0, 0.0, 0.0, float("nan")

    used = np.zeros(len(ecg_peaks), dtype=bool)
    matches = []
    for p in ppg_peaks:
        j = int(np.argmin(np.abs(ecg_peaks - p)))
        if used[j]:
            continue
        if abs(ecg_peaks[j] - p) <= tol:
            used[j] = True
            matches.append(ecg_peaks[j] - p)
    tp = len(matches)
    fp = max(0, len(ppg_peaks) - tp)
    fn = max(0, len(ecg_peaks) - tp)
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2*prec*rec/(prec+rec+1e-12)
    mae_ms = float(np.mean(np.abs(np.asarray(matches, dtype=np.float32))) / fs * 1000.0) if tp>0 else float("nan")
    return float(prec), float(rec), float(f1), mae_ms

# ----------------------------
# Dataset
# ----------------------------
@dataclass
class WinCfg:
    fs: float
    win_sec: float
    hop_sec: float
    stft_n_fft: int
    stft_hop: int
    stft_win: int

class DenoiseDataset(Dataset):
    """Per-window dataset for a given activity (walk/run)."""
    def __init__(self, recs: List[dict], cfg: WinCfg, activity: str):
        self.cfg=cfg
        self.activity=activity
        self.items=[]
        self.feature_names = None

        for r in recs:
            if r.get("_activity") != activity:
                continue
            if not all(k in r for k in ["pleth_1","pleth_2","a_x","a_y","a_z","g_x","g_y","g_z","peaks"]):
                continue
            p1 = bandpass_ppg(r["pleth_1"], cfg.fs)
            p2 = bandpass_ppg(r["pleth_2"], cfg.fs)

            ax = r["a_x"] * 9.81; ay = r["a_y"] * 9.81; az = r["a_z"] * 9.81
            gx = np.deg2rad(r["g_x"]); gy = np.deg2rad(r["g_y"]); gz = np.deg2rad(r["g_z"])
            n = min(len(p2), len(ax), len(gx))
            p1=p1[:n]; p2=p2[:n]
            ax=ax[:n]; ay=ay[:n]; az=az[:n]
            gx=gx[:n]; gy=gy[:n]; gz=gz[:n]

            gax = lowpass(ax, cfg.fs); gay = lowpass(ay, cfg.fs); gaz = lowpass(az, cfg.fs)
            acc_dyn = np.column_stack([ax-gax, ay-gay, az-gaz]).astype(np.float32)
            gyro = np.column_stack([gx,gy,gz]).astype(np.float32)

            for s,e in window_slices(n, cfg.fs, cfg.win_sec, cfg.hop_sec):
                ppg_ir = p2[s:e]
                feat, names = extract_window_features(ppg_ir, acc_dyn[s:e], gyro[s:e], cfg.fs)
                self.feature_names = names
                self.items.append((r, s, e, p1[s:e], p2[s:e], acc_dyn[s:e], gyro[s:e], feat))

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        r, s, e, p1, p2, acc_dyn, gyro, feat = self.items[i]
        x_time = np.stack([
            zscore_np(p1),
            zscore_np(p2),
            zscore_np(acc_dyn[:,0]),
            zscore_np(acc_dyn[:,1]),
            zscore_np(acc_dyn[:,2]),
            zscore_np(gyro[:,0]),
            zscore_np(gyro[:,1]),
            zscore_np(gyro[:,2]),
        ], axis=0).astype(np.float32)  # (8,T)
        feat_ch = np.repeat(feat.astype(np.float32)[:,None], x_time.shape[1], axis=1)  # (F,T)
        x = np.concatenate([x_time, feat_ch], axis=0).astype(np.float32)  # (8+F,T)

        ecg_peaks = ecg_peak_indices_from_column(r["peaks"][s:e])
        return torch.from_numpy(x), torch.from_numpy(p2.astype(np.float32)), torch.from_numpy(ecg_peaks.astype(np.int64)), r["_name"]

# ----------------------------
# Phase median from SIT + subject-specific a (grid search)
# ----------------------------
def estimate_subject_phase_median(sit_recs: List[dict], fs: float) -> Dict[str, float]:
    phase = {}
    for r in sit_recs:
        sid = r["_name"].split("_")[0]
        if r.get("_activity") != "sit" or "peaks" not in r or "pleth_2" not in r:
            continue
        ppg = bandpass_ppg(r["pleth_2"], fs)
        ecg_pk = ecg_peak_indices_from_column(r["peaks"])
        ppg_pk = ppg_peak_indices_aboy(ppg, fs)
        if len(ecg_pk) < 5 or len(ppg_pk) < 5:
            continue
        offsets = []
        for p in ppg_pk:
            j = int(np.argmin(np.abs(ecg_pk - p)))
            offsets.append(int(ecg_pk[j] - p))
        if len(offsets) >= 10:
            phase.setdefault(sid, []).extend(offsets)
    for sid, offs in list(phase.items()):
        phase[sid] = float(np.median(np.asarray(offs)))
    return phase

def grid_search_a_for_subject(recs: List[dict],
                             sid: str,
                             activity: str,
                             fs: float,
                             base_phase_samples: float,
                             a_min: float = 0.5,
                             a_max: float = 1.5,
                             a_steps: int = 21,
                             tol_s: float = 0.08) -> float:
    best = (float("inf"), 1.0)
    grid = np.linspace(a_min, a_max, a_steps)
    for a in grid:
        maes = []
        for r in recs:
            if r.get("_activity") != activity:
                continue
            if r["_name"].split("_")[0] != sid:
                continue
            if "pleth_2" not in r or "peaks" not in r:
                continue
            ppg = bandpass_ppg(r["pleth_2"], fs)
            ecg_pk = ecg_peak_indices_from_column(r["peaks"])
            shift = int(round(a * base_phase_samples))
            ecg_pk_s = ecg_pk + shift
            ppg_pk = ppg_peak_indices_aboy(ppg, fs)
            _, _, _, mae_ms = match_peaks_f1(ppg_pk, ecg_pk_s, tol_s=tol_s, fs=fs)
            if np.isfinite(mae_ms):
                maes.append(mae_ms)
        if len(maes) >= 1:
            m = float(np.mean(maes))
            if m < best[0]:
                best = (m, float(a))
    return best[1]

# ----------------------------
# Evaluation
# ----------------------------
def eval_denoiser(model: nn.Module,
                 ds,
                 cfg: WinCfg,
                 a_table: dict,
                 phase_med: dict,
                 activity: str,
                 device: str,
                 tol_s: float = 0.08) -> dict:
    model.eval()
    ld = DataLoader(ds, batch_size=32, shuffle=False, drop_last=False)

    l1s = []
    snr_imps = []
    maes = []
    f1s = []
    precs = []
    recs = []

    with torch.no_grad():
        for x, p2, ecg_peaks_win, name in ld:
            x = x.to(device)
            p2 = p2.to(device)
            mag, ph = stft_mag_phase(p2, cfg.stft_n_fft, cfg.stft_hop, cfg.stft_win)
            M = model(x, mag.shape)
            mag_clean = (M.squeeze(1) * mag)
            y = istft_from_mag_phase(mag_clean, ph, cfg.stft_n_fft, cfg.stft_hop, cfg.stft_win, length=p2.shape[1])

            # proxy L1
            l1s.append(torch.mean(torch.abs(y - p2)).item())

            # proxy SNR improvement
            target = p2
            err_b = torch.mean((p2 - target)**2).item()
            err_a = torch.mean((y - target)**2).item()
            snr_imps.append(10.0 * math.log10((err_b + 1e-9) / (err_a + 1e-9)))

            y_np = y.detach().cpu().numpy()
            for bi in range(y_np.shape[0]):
                rec_name = name[bi]
                sid = rec_name.split("_")[0]
                y_sig = bandpass_ppg(y_np[bi], cfg.fs)
                ppg_pk = ppg_peak_indices_aboy(y_sig, cfg.fs)

                ecg_pk = ecg_peaks_win[bi].detach().cpu().numpy()
                ecg_pk = ecg_pk[ecg_pk >= 0]

                base = float(phase_med.get(sid, 0.0))
                a = float(a_table.get(sid, {}).get(activity, 1.0))
                shift = int(round(a * base))
                ecg_pk_s = ecg_pk + shift

                prec, rec, f1, mae_ms = match_peaks_f1(ppg_pk, ecg_pk_s, tol_s=tol_s, fs=cfg.fs)
                precs.append(prec); recs.append(rec); f1s.append(f1)
                if np.isfinite(mae_ms):
                    maes.append(mae_ms)

    return {
        "l1_proxy": float(np.mean(l1s)) if l1s else float("nan"),
        "snr_improvement_db": float(np.mean(snr_imps)) if snr_imps else float("nan"),
        "peak_timing_mae_ms": float(np.mean(maes)) if maes else float("nan"),
        "peak_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "peak_precision": float(np.mean(precs)) if precs else float("nan"),
        "peak_recall": float(np.mean(recs)) if recs else float("nan"),
        "n_windows": int(len(ds)),
    }

# ----------------------------
# Training per activity with subject-level no-leak CV + holdout
# ----------------------------
def train_one_activity(recs_by_subj: Dict[str, List[dict]],
                       cfg: WinCfg,
                       activity: str,
                       outdir: Path,
                       train_size: float,
                       n_splits: int,
                       epochs: int,
                       lr: float,
                       lam_shape: float,
                       lam_smooth: float,
                       seed: int = 42):
    outdir.mkdir(parents=True, exist_ok=True)

    subjects = np.array(sorted(recs_by_subj.keys()))
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    idx = np.arange(len(subjects))
    tr_i, ho_i = next(gss.split(idx, groups=subjects))
    tr_subjects = subjects[tr_i].tolist()
    ho_subjects = subjects[ho_i].tolist()

    tr_recs = [r for sid in tr_subjects for r in recs_by_subj[sid]]
    ho_recs = [r for sid in ho_subjects for r in recs_by_subj[sid]]
    sit_recs_train = [r for sid in tr_subjects for r in recs_by_subj[sid] if r.get("_activity") == "sit"]
    sit_recs_hold = [r for sid in ho_subjects for r in recs_by_subj[sid] if r.get("_activity") == "sit"]

    if len(sit_recs_train) == 0:
        raise RuntimeError("No SIT records in training subjects (needed for phase template/median).")

    # SIT template for shape proxy
    tmp = bandpass_ppg(sit_recs_train[0]["pleth_2"], cfg.fs)
    w = int(cfg.win_sec * cfg.fs)
    sit_template = torch.from_numpy(zscore_np(tmp[:w])).float()

    # Phase medians
    phase_med_tr = estimate_subject_phase_median(sit_recs_train, cfg.fs)
    phase_med_ho = estimate_subject_phase_median(sit_recs_hold, cfg.fs)

    # a_table (subject-specific per activity) via grid search
    a_table = {sid: {} for sid in subjects.tolist()}
    for sid in tr_subjects:
        base = phase_med_tr.get(sid, 0.0)
        a_table[sid][activity] = grid_search_a_for_subject(tr_recs, sid=sid, activity=activity, fs=cfg.fs, base_phase_samples=base)
    for sid in ho_subjects:
        base = phase_med_ho.get(sid, 0.0)
        a_table[sid][activity] = grid_search_a_for_subject(ho_recs, sid=sid, activity=activity, fs=cfg.fs, base_phase_samples=base)

    ds_tr_full = DenoiseDataset(tr_recs, cfg, activity)
    ds_ho = DenoiseDataset(ho_recs, cfg, activity)
    if len(ds_tr_full) == 0:
        raise RuntimeError(f"No training windows for activity={activity}.")
    in_ch = ds_tr_full[0][0].shape[0]
    feat_names = ds_tr_full.feature_names

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Groups per window for GroupKFold
    groups = []
    for item in ds_tr_full.items:
        r = item[0]
        groups.append(r["_name"].split("_")[0])
    groups = np.asarray(groups)

    idx_all = np.arange(len(ds_tr_full))
    gkf = GroupKFold(n_splits=min(n_splits, len(set(groups))))

    fold_rows = []
    for fold, (tri, vai) in enumerate(gkf.split(idx_all, groups=groups), start=1):
        tr_ds = torch.utils.data.Subset(ds_tr_full, tri)
        va_ds = torch.utils.data.Subset(ds_tr_full, vai)

        model = MaskNet(in_ch=in_ch).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        tr_ld = DataLoader(tr_ds, batch_size=16, shuffle=True, drop_last=False)
        va_ld = DataLoader(va_ds, batch_size=32, shuffle=False, drop_last=False)

        best_val = float("inf")
        best_state = None

        ep_bar = tqdm(range(1, epochs+1), desc=f"Denoiser[{activity}] fold{fold} epochs", leave=False)
        for ep in ep_bar:
            model.train()
            tr_loss = 0.0
            for x, p2, _, _name in tr_ld:
                x = x.to(dev)
                p2 = p2.to(dev)
                mag, ph = stft_mag_phase(p2, cfg.stft_n_fft, cfg.stft_hop, cfg.stft_win)
                M = model(x, mag.shape)
                mag_clean = (M.squeeze(1) * mag)
                y = istft_from_mag_phase(mag_clean, ph, cfg.stft_n_fft, cfg.stft_hop, cfg.stft_win, length=p2.shape[1])

                l_shape = shape_loss_to_sit(y, sit_template.to(dev))
                l_smooth = freq_smoothness_reg(M)
                loss = l_shape + lam_smooth * l_smooth

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                tr_loss += loss.item() * x.size(0)
            tr_loss /= max(1, len(tr_ds))

            model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for x, p2, _, _name in va_ld:
                    x = x.to(dev)
                    p2 = p2.to(dev)
                    mag, ph = stft_mag_phase(p2, cfg.stft_n_fft, cfg.stft_hop, cfg.stft_win)
                    M = model(x, mag.shape)
                    mag_clean = (M.squeeze(1) * mag)
                    y = istft_from_mag_phase(mag_clean, ph, cfg.stft_n_fft, cfg.stft_hop, cfg.stft_win, length=p2.shape[1])

                    l_shape = shape_loss_to_sit(y, sit_template.to(dev))
                    l_smooth = freq_smoothness_reg(M)
                    loss = l_shape + lam_smooth * l_smooth
                    va_loss += loss.item() * x.size(0)
            va_loss /= max(1, len(va_ds))

            ep_bar.set_postfix(train=f"{tr_loss:.4f}", val=f"{va_loss:.4f}")

            if va_loss < best_val:
                best_val = va_loss
                best_state = {k:v.detach().cpu() for k,v in model.state_dict().items()}

        model.load_state_dict(best_state)
        fold_eval = eval_denoiser(model, va_ds, cfg, a_table, phase_med_tr, activity, dev)
        fold_eval.update({"fold": fold, "best_val_loss": float(best_val)})
        fold_rows.append(fold_eval)

    # Final fit on all training windows (simple)
    model = MaskNet(in_ch=in_ch).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr_ld = DataLoader(ds_tr_full, batch_size=16, shuffle=True, drop_last=False)
    best_state = None
    best_proxy = float("inf")

    ep_bar = tqdm(range(1, epochs+1), desc=f"Denoiser[{activity}] epochs (final)", leave=True)
    for ep in ep_bar:
        model.train()
        tr_loss = 0.0
        for x, p2, _, _name in tr_ld:
            x = x.to(dev)
            p2 = p2.to(dev)
            mag, ph = stft_mag_phase(p2, cfg.stft_n_fft, cfg.stft_hop, cfg.stft_win)
            M = model(x, mag.shape)
            mag_clean = (M.squeeze(1) * mag)
            y = istft_from_mag_phase(mag_clean, ph, cfg.stft_n_fft, cfg.stft_hop, cfg.stft_win, length=p2.shape[1])
            l_shape = shape_loss_to_sit(y, sit_template.to(dev))
            l_smooth = freq_smoothness_reg(M)
            loss = l_shape + lam_smooth * l_smooth
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= max(1, len(ds_tr_full))
        ep_bar.set_postfix(train=f"{tr_loss:.4f}")
        if tr_loss < best_proxy:
            best_proxy = tr_loss
            best_state = {k:v.detach().cpu() for k,v in model.state_dict().items()}

    model.load_state_dict(best_state)
    hold_eval = eval_denoiser(model, ds_ho, cfg, a_table, phase_med_ho, activity, dev)
    hold_eval.update({"fold": "holdout", "best_val_loss": float(best_proxy)})

    # Save artifacts
    torch.save(best_state, outdir / f"{activity}_model.pt")
    (outdir / "a_table.json").write_text(json.dumps(a_table, ensure_ascii=False, indent=2), encoding="utf-8")

    dfm = pd.DataFrame(fold_rows + [hold_eval])
    dfm.to_csv(outdir / f"{activity}_denoiser_metrics.csv", index=False)

    summary = {
        "version": "denoiser_v8_masknet",
        "activity": activity,
        "in_channels": int(in_ch),
        "feature_names": feat_names,
        "stft": {"n_fft": cfg.stft_n_fft, "hop_length": cfg.stft_hop, "win_length": cfg.stft_win, "window": "hann"},
        "win": {"fs": cfg.fs, "win_sec": cfg.win_sec, "hop_sec": cfg.hop_sec},
        "lam_shape": lam_shape,
        "lam_smooth": lam_smooth,
        "cv": fold_rows,
        "holdout": hold_eval,
        "train_subjects": tr_subjects,
        "holdout_subjects": ho_subjects,
    }
    (outdir / f"{activity}_denoiser_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="results_denoiser_v8")
    ap.add_argument("--fs", type=float, default=500.0)
    ap.add_argument("--win", type=float, default=6.0)
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--train_size", type=float, default=0.8)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam_shape", type=float, default=0.05)
    ap.add_argument("--lam_smooth", type=float, default=0.05)
    ap.add_argument("--stft_n_fft", type=int, default=256)
    ap.add_argument("--stft_hop", type=int, default=64)
    ap.add_argument("--stft_win", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cfg = WinCfg(fs=args.fs, win_sec=args.win, hop_sec=args.hop,
                 stft_n_fft=args.stft_n_fft, stft_hop=args.stft_hop, stft_win=args.stft_win)

    sub2recs = load_physionet_csv(Path(args.data_root), fs=args.fs)

    for activity in ["walk", "run"]:
        train_one_activity(
            recs_by_subj=sub2recs,
            cfg=cfg,
            activity=activity,
            outdir=outdir / activity,
            train_size=args.train_size,
            n_splits=args.n_splits,
            epochs=args.epochs,
            lr=args.lr,
            lam_shape=args.lam_shape,
            lam_smooth=args.lam_smooth,
            seed=args.seed
        )

    dt = time.time() - t0
    print(f"[Done] Denoiser v8 finished in {dt/60:.1f} min. Outputs: {outdir}")

if __name__ == "__main__":
    if funcs is None:
        print("[WARN] funcs.py not importable. Peak metrics will use fallback find_peaks.")
    main()
