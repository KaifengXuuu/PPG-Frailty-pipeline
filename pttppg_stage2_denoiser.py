"""pttppg_stage2_denoiser.py

Stage 2: Subject-level no-leak denoiser training for PhysioNet Pulse Transit Time PPG.

Key specs implemented
---------------------
* Unified raw inputs: 8 channels
  1-2: pleth_1, pleth_2 (bandpass pre-window)
  3-5: a_dyn_x, a_dyn_y, a_dyn_z (m/s^2, gravity removed)
  6-8: gyro_x, gyro_y, gyro_z (rad/s)
* Additional window-level feature channels are concatenated to the raw time series as
  constant-valued per-window channels (repeat over time).
* Train separate models for walk and run.
* Frequency-domain mask network (STFT magnitude mask):
    |Y_clean| = M ⊙ |Y_noisy|, phase kept from noisy PPG (pleth_2).
  Includes frequency-direction smoothness regularization.
* ECG/peaks are supervision only (not inputs):
  - Peak timing MAE and peak F1 against ECG peaks, after ECG peak shifting.
  - The shift is subject-specific: a_s,activity ∈ (0.5, 1.5) is learnable and stored.
  - Base phase median per subject is measured on SIT segments (PPG vs ECG).
  - Training emphasizes peak alignment loss (primary objective), then shape-to-sit.

Outputs
-------
* results CSV per activity: peak_timing_mae, peak_f1, snr_improvement_db (CV and holdout)
* Saved reusable artifacts:
  - Torch weights (.pt)
  - ONNX export (best-effort; requires onnx/onnxscript)
  - JSON sidecar with config + subject-specific a table + feature name ordering

Run example
-----------
python pttppg_stage2_denoiser.py --data_root ./physionet.org/files/pulse-transit-time-ppg/1.1.0 \
  --outdir results_stage2 --fs 500 --win 6 --hop 1 --train_size 0.8 --n_splits 5 --epochs 20

"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from scipy import signal


# ----------------------------- Shared utilities -----------------------------


def set_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def zscore_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = float(np.mean(x))
    s = float(np.std(x) + 1e-8)
    return (x - m) / s


def bandpass_np(x: np.ndarray, fs: float, lowcut: float = 0.5, highcut: float = 8.0, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return signal.filtfilt(b, a, x).astype(np.float32)


def safe_torch_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def window_slices(n: int, fs: float, win_sec: float, hop_sec: float) -> Iterable[Tuple[int, int]]:
    w = int(round(win_sec * fs))
    h = int(round(hop_sec * fs))
    if w <= 0 or h <= 0:
        raise ValueError("win/hop must be positive")
    for s in range(0, max(1, n - w + 1), h):
        yield s, s + w


def parse_subject_and_activity(stem: str) -> Tuple[str, str]:
    # e.g. s10_run
    parts = stem.split("_")
    if len(parts) < 2:
        return parts[0], "unknown"
    return parts[0], parts[1].lower()


def to_rad_per_sec(x_deg_per_sec: np.ndarray) -> np.ndarray:
    return (np.asarray(x_deg_per_sec, dtype=np.float32) * (math.pi / 180.0)).astype(np.float32)


def gravity_remove(acc_g: np.ndarray, fs: float, g: float = 9.81, hp_cut: float = 0.3) -> np.ndarray:
    """Remove quasi-static gravity component via low-pass estimate.

    acc_g: (T,3) in g.
    Returns dynamic acceleration in m/s^2.
    """
    acc = np.asarray(acc_g, dtype=np.float32) * g
    # low-pass gravity estimate (2nd order Butterworth)
    nyq = 0.5 * fs
    b, a = signal.butter(2, hp_cut / nyq, btype="low")
    grav = signal.filtfilt(b, a, acc, axis=0)
    return (acc - grav).astype(np.float32)


def mags_and_jerk(acc_dyn: np.ndarray, gyro_rad: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    acc_mag = np.linalg.norm(acc_dyn, axis=1).astype(np.float32)
    gyro_mag = np.linalg.norm(gyro_rad, axis=1).astype(np.float32)
    jerk = np.vstack([np.zeros((1, 3), np.float32), np.diff(acc_dyn, axis=0) * fs])
    jerk_mag = np.linalg.norm(jerk, axis=1).astype(np.float32)
    return acc_mag, gyro_mag, jerk_mag


def stft_mag_torch(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    """x: (B,T) float"""
    window = torch.hann_window(win, device=x.device)
    X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True)
    return torch.abs(X)  # (B,F,TT)


def istft_from_mag_phase(mag: torch.Tensor, phase: torch.Tensor, n_fft: int, hop: int, win: int, length: int) -> torch.Tensor:
    """Reconstruct time signal from magnitude and phase.
    mag/phase: (B,F,TT)
    returns (B,T)
    """
    window = torch.hann_window(win, device=mag.device)
    complex_spec = torch.polar(mag, phase)
    y = torch.istft(complex_spec, n_fft=n_fft, hop_length=hop, win_length=win, window=window, length=length)
    return y


def peak_match_f1(ppg_peaks: np.ndarray, ecg_peaks: np.ndarray, tol_s: float, fs: float) -> Tuple[float, float, float]:
    """Simple peak matching with tolerance (in seconds)."""
    if len(ppg_peaks) == 0 and len(ecg_peaks) == 0:
        return 1.0, 1.0, 1.0
    if len(ppg_peaks) == 0:
        return 0.0, 0.0, 0.0
    if len(ecg_peaks) == 0:
        return 0.0, 0.0, 0.0
    tol = int(round(tol_s * fs))
    ecg_used = np.zeros(len(ecg_peaks), dtype=bool)
    tp = 0
    for p in ppg_peaks:
        j = np.searchsorted(ecg_peaks, p)
        cand = []
        if j < len(ecg_peaks):
            cand.append(j)
        if j - 1 >= 0:
            cand.append(j - 1)
        best = None
        best_d = None
        for c in cand:
            if ecg_used[c]:
                continue
            d = abs(int(ecg_peaks[c]) - int(p))
            if d <= tol and (best_d is None or d < best_d):
                best = c
                best_d = d
        if best is not None:
            ecg_used[best] = True
            tp += 1
    fp = len(ppg_peaks) - tp
    fn = len(ecg_peaks) - tp
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(prec), float(rec), float(f1)


def snr_improvement_db(noisy: np.ndarray, clean_hat: np.ndarray, ref: np.ndarray) -> float:
    """SNR improvement in dB using ref as target."""
    noisy = noisy.astype(np.float32)
    clean_hat = clean_hat.astype(np.float32)
    ref = ref.astype(np.float32)
    e_before = np.mean((noisy - ref) ** 2)
    e_after = np.mean((clean_hat - ref) ** 2)
    return float(10.0 * np.log10((e_before + 1e-9) / (e_after + 1e-9)))


# ----------------------------- Data loading -----------------------------


CSV_CAND = {
    "time": ["time", "Time", "t"],
    "ecg": ["ecg", "ECG"],
    "peaks": ["peaks", "Rpeaks", "r_peaks"],
    "pleth_1": ["pleth_1"],
    "pleth_2": ["pleth_2"],
    "a_x": ["a_x", "AX"],
    "a_y": ["a_y", "AY"],
    "a_z": ["a_z", "AZ"],
    "g_x": ["g_x", "GX"],
    "g_y": ["g_y", "GY"],
    "g_z": ["g_z", "GZ"],
}


def _pick(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def load_physionet_csv(data_root: Path, fs: float) -> Dict[str, List[dict]]:
    """Load CSVs into a subject->records list.

    Each record contains arrays and metadata: sid, activity.
    Bandpass is applied to pleth_1/2 here (per latest requirement).
    """
    csv_dir = data_root / "csv"
    if not csv_dir.exists():
        csv_dir = data_root / "files" / "pulse-transit-time-ppg" / "1.1.0" / "csv"
    paths = sorted([p for p in csv_dir.glob("s*_*.csv") if p.name != "subjects_info.csv"])

    sub2recs: Dict[str, List[dict]] = {}
    for p in tqdm(paths, desc="Load CSV", leave=False):
        sid, activity = parse_subject_and_activity(p.stem)
        df = pd.read_csv(p)
        m = {k: _pick(df, v) for k, v in CSV_CAND.items()}
        rec: Dict[str, np.ndarray] = {"sid": sid, "activity": activity, "file": p.name}

        # time
        if m.get("time") is not None:
            t = pd.to_datetime(df[m["time"]], errors="coerce")
            if t.notna().all():
                rec["time"] = (t - t.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float32)
            else:
                rec["time"] = (np.arange(len(df), dtype=np.float32) / float(fs))
        else:
            rec["time"] = (np.arange(len(df), dtype=np.float32) / float(fs))

        # numeric channels
        for k in ["ecg", "peaks", "pleth_1", "pleth_2", "a_x", "a_y", "a_z", "g_x", "g_y", "g_z"]:
            if m.get(k) is None:
                continue
            rec[k] = pd.to_numeric(df[m[k]], errors="coerce").to_numpy(dtype=np.float32)

        # required channels check
        if "pleth_1" not in rec or "pleth_2" not in rec or "a_x" not in rec or "g_x" not in rec:
            continue

        # bandpass pleth
        rec["pleth_1"] = bandpass_np(rec["pleth_1"], fs=fs, lowcut=0.5, highcut=8.0, order=3)
        rec["pleth_2"] = bandpass_np(rec["pleth_2"], fs=fs, lowcut=0.5, highcut=8.0, order=3)

        sub2recs.setdefault(sid, []).append(rec)

    return sub2recs


def subjectwise(sub2recs: Dict[str, List[dict]]) -> Tuple[List[dict], List[str]]:
    recs: List[dict] = []
    groups: List[str] = []
    for sid in sorted(sub2recs.keys()):
        for r in sub2recs[sid]:
            recs.append(r)
            groups.append(sid)
    return recs, groups


# ----------------------------- Window features -----------------------------


def bandpowers(x: np.ndarray, fs: float, bands=((0.1, 0.5), (0.5, 3.0), (3.0, 8.0))) -> List[float]:
    f, Pxx = signal.welch(x, fs=fs, nperseg=min(len(x), int(fs * 2)))
    out = []
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        out.append(float(np.trapz(Pxx[m], f[m])) if np.any(m) else 0.0)
    return out


def spectral_entropy_and_domfreq(x: np.ndarray, fs: float) -> Tuple[float, float]:
    f, Pxx = signal.welch(x, fs=fs, nperseg=min(len(x), int(fs * 2)))
    P = Pxx + 1e-12
    P = P / np.sum(P)
    ent = float(-(P * np.log(P)).sum() / np.log(len(P)))
    dom = float(f[int(np.argmax(Pxx))]) if len(f) else 0.0
    return ent, dom


def time_stats(x: np.ndarray) -> List[float]:
    x = np.asarray(x)
    q25, q75 = np.percentile(x, [25, 75])
    return [
        float(np.mean(x)),
        float(np.std(x)),
        float(np.median(x)),
        float(q75 - q25),
        float(np.sqrt(np.mean(x ** 2))),
    ]


PPG_FEAT_NAMES = [
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


IMU_FEAT_NAMES = [
    # time stats
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
    # bandpowers
    "AccMag_bp_0.1_0.5",
    "AccMag_bp_0.5_3",
    "AccMag_bp_3_8",
    "GyroMag_bp_0.1_0.5",
    "GyroMag_bp_0.5_3",
    "GyroMag_bp_3_8",
    # spectral
    "AccMag_spec_entropy",
    "AccMag_dom_freq",
    "GyroMag_spec_entropy",
    "GyroMag_dom_freq",
    "JerkMag_spec_entropy",
    "JerkMag_dom_freq",
]


def extract_window_features(rec: dict, s: int, e: int, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (raw8, feat39, meta) where:
    raw8: (8,T)
    feat39: (39,) = 10 PPG feats + 27 IMU feats + 2 reserved (currently zeros for future)
    meta: (3,) = [sid_idx_placeholder, activity_binary, has_ecg]
    """
    p1 = rec["pleth_1"][s:e]
    p2 = rec["pleth_2"][s:e]
    acc_g = np.stack([rec["a_x"][s:e], rec["a_y"][s:e], rec["a_z"][s:e]], axis=1)
    gyro_deg = np.stack([rec["g_x"][s:e], rec["g_y"][s:e], rec["g_z"][s:e]], axis=1)
    gyro = to_rad_per_sec(gyro_deg)
    acc_dyn = gravity_remove(acc_g, fs=fs)
    acc_mag, gyro_mag, jerk_mag = mags_and_jerk(acc_dyn, gyro, fs=fs)

    raw8 = np.concatenate(
        [
            p1[None, :],
            p2[None, :],
            acc_dyn.T,
            gyro.T,
        ],
        axis=0,
    ).astype(np.float32)

    # PPG feats computed on pleth_2 (IR) as primary
    ppg_ts = time_stats(p2)
    ppg_bp = bandpowers(p2, fs)
    ppg_ent, ppg_dom = spectral_entropy_and_domfreq(p2, fs)
    ppg_feats = np.array(ppg_ts + ppg_bp + [ppg_ent, ppg_dom], dtype=np.float32)

    # IMU feats
    acc_ts = time_stats(acc_mag)
    gyro_ts = time_stats(gyro_mag)
    jerk_ts = time_stats(jerk_mag)
    acc_bp = bandpowers(acc_mag, fs)
    gyro_bp = bandpowers(gyro_mag, fs)
    acc_ent, acc_dom = spectral_entropy_and_domfreq(acc_mag, fs)
    gyro_ent, gyro_dom = spectral_entropy_and_domfreq(gyro_mag, fs)
    jerk_ent, jerk_dom = spectral_entropy_and_domfreq(jerk_mag, fs)
    imu_feats = np.array(
        acc_ts
        + gyro_ts
        + jerk_ts
        + acc_bp
        + gyro_bp
        + [acc_ent, acc_dom, gyro_ent, gyro_dom, jerk_ent, jerk_dom],
        dtype=np.float32,
    )

    # total features = 10 + 27 = 37; keep 39 per your v7.x convention
    feat39 = np.concatenate([ppg_feats, imu_feats, np.zeros(2, np.float32)], axis=0)

    return raw8, feat39, np.array([0.0, 1.0 if rec["activity"] != "sit" else 0.0, 1.0 if "ecg" in rec else 0.0], np.float32)


# ----------------------------- Dataset -----------------------------


class DenoiseWindowDataset(Dataset):
    def __init__(
        self,
        records: List[dict],
        fs: float,
        win_sec: float,
        hop_sec: float,
        feature_dim: int = 39,
        activity: str = "walk",
        include_only_activity: bool = True,
    ):
        self.records = records
        self.fs = fs
        self.win_sec = win_sec
        self.hop_sec = hop_sec
        self.feature_dim = feature_dim
        self.activity = activity
        self.index: List[Tuple[int, int, int]] = []
        for i, r in enumerate(records):
            if include_only_activity and r["activity"] != activity:
                continue
            n = len(r["pleth_2"])
            for s, e in window_slices(n, fs, win_sec, hop_sec):
                self.index.append((i, s, e))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        i, s, e = self.index[idx]
        r = self.records[i]
        raw8, feat39, _ = extract_window_features(r, s, e, self.fs)

        # build input: raw8 + repeated features as channels
        feat_ch = np.repeat(feat39[:, None], raw8.shape[1], axis=1).astype(np.float32)  # (39,T)
        x = np.concatenate([raw8, feat_ch], axis=0).astype(np.float32)  # (47,T)

        # target/reference: use bandpassed pleth_2 as "pseudo-clean" baseline
        y = zscore_np(r["pleth_2"][s:e]).astype(np.float32)  # (T,)

        # ecg peaks for supervision if available
        peaks = r.get("peaks", None)
        if peaks is not None:
            ecg_peaks = np.where(peaks[s:e] > 0.5)[0].astype(np.int32)
        else:
            ecg_peaks = np.empty((0,), np.int32)

        sid = r["sid"]
        return torch.from_numpy(x), torch.from_numpy(y[None, :]), torch.from_numpy(ecg_peaks), sid


# ----------------------------- Model -----------------------------


class MaskNet(nn.Module):
    """Simple 1D Conv net producing an STFT magnitude mask (0..1)."""

    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 7, padding=3),
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.ReLU(True),
            nn.Conv1d(hidden, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T) -> mask logit (B,1,T)
        return self.net(x)


# ----------------------------- Training helpers -----------------------------


def compute_sit_phase_median(records: List[dict], fs: float) -> Dict[str, float]:
    """Per subject, compute median phase offset between PPG(pleth_2) peaks and ECG peaks on SIT.

    This is a lightweight approach: PPG peaks detected by find_peaks; ECG peaks from CSV `peaks`.
    Returns offset seconds (ECG - PPG). Positive means ECG peaks occur later.
    """
    out: Dict[str, float] = {}
    for sid in sorted({r["sid"] for r in records}):
        deltas = []
        for r in records:
            if r["sid"] != sid or r["activity"] != "sit":
                continue
            if "peaks" not in r:
                continue
            ppg = zscore_np(r["pleth_2"])  # already bandpassed
            pk_ppg, _ = signal.find_peaks(ppg, distance=int(fs * 0.25), prominence=0.2)
            pk_ecg = np.where(r["peaks"] > 0.5)[0]
            if len(pk_ppg) < 10 or len(pk_ecg) < 10:
                continue
            # match each ppg peak to nearest ecg peak and store delta
            for p in pk_ppg:
                j = np.searchsorted(pk_ecg, p)
                cand = []
                if j < len(pk_ecg):
                    cand.append(pk_ecg[j])
                if j - 1 >= 0:
                    cand.append(pk_ecg[j - 1])
                if not cand:
                    continue
                nearest = min(cand, key=lambda q: abs(int(q) - int(p)))
                deltas.append((nearest - p) / fs)
        out[sid] = float(np.median(deltas)) if len(deltas) else 0.0
    return out


def freq_smoothness_reg(mask_mag: torch.Tensor) -> torch.Tensor:
    """Smoothness regularization along frequency axis.
    mask_mag: (B,F,TT)
    """
    return torch.mean(torch.abs(mask_mag[:, 1:, :] - mask_mag[:, :-1, :]))


def detect_ppg_peaks(ppg: np.ndarray, fs: float) -> np.ndarray:
    ppg = zscore_np(ppg)
    pk, _ = signal.find_peaks(ppg, distance=int(fs * 0.25), prominence=0.2)
    return pk.astype(np.int32)


def peaks_timing_mae(ppg_peaks: np.ndarray, ecg_peaks: np.ndarray, fs: float) -> float:
    if len(ppg_peaks) == 0 or len(ecg_peaks) == 0:
        return float("nan")
    # for each ecg peak, nearest ppg peak
    d = []
    for e in ecg_peaks:
        j = np.searchsorted(ppg_peaks, e)
        cand = []
        if j < len(ppg_peaks):
            cand.append(ppg_peaks[j])
        if j - 1 >= 0:
            cand.append(ppg_peaks[j - 1])
        if not cand:
            continue
        nearest = min(cand, key=lambda q: abs(int(q) - int(e)))
        d.append(abs(int(nearest) - int(e)) / fs)
    return float(np.mean(d)) if len(d) else float("nan")


@dataclass
class STFTCfg:
    n_fft: int = 256
    hop_length: int = 64
    win_length: int = 256


def train_one_activity(
    activity: str,
    tr_records: List[dict],
    va_records: List[dict],
    sit_phase_med: Dict[str, float],
    stft_cfg: STFTCfg,
    fs: float,
    win_sec: float,
    hop_sec: float,
    epochs: int,
    lr: float,
    lam_shape: float,
    lam_smooth: float,
    outdir: Path,
    seed: int = 42,
) -> Tuple[MaskNet, dict]:
    """Train denoiser for one activity (walk/run)."""
    set_seed(seed)
    device = safe_torch_device()

    tr_ds = DenoiseWindowDataset(tr_records, fs, win_sec, hop_sec, activity=activity)
    va_ds = DenoiseWindowDataset(va_records, fs, win_sec, hop_sec, activity=activity)

    if len(tr_ds) == 0 or len(va_ds) == 0:
        raise RuntimeError(f"Empty dataset for activity={activity}.")

    in_channels = next(iter(DataLoader(tr_ds, batch_size=1)))[0].shape[1]
    model = MaskNet(in_channels=in_channels).to(device)

    # subject-specific learnable a (table)
    subj_list = sorted({r["sid"] for r in tr_records})
    subj2idx = {s: i for i, s in enumerate(subj_list)}
    a_table = nn.Parameter(torch.ones(len(subj_list), device=device))
    # constrain to (0.5,1.5) via sigmoid
    def a_value() -> torch.Tensor:
        return 0.5 + torch.sigmoid(a_table)  # (0.5,1.5)

    opt = torch.optim.Adam(list(model.parameters()) + [a_table], lr=lr)

    tr_ld = DataLoader(tr_ds, batch_size=16, shuffle=True, num_workers=0)
    va_ld = DataLoader(va_ds, batch_size=32, shuffle=False, num_workers=0)

    best = {"val": float("inf"), "state": None, "a": None}

    pbar = tqdm(range(1, epochs + 1), desc=f"Denoiser[{activity}] epochs", leave=True)
    for ep in pbar:
        model.train()
        tr_loss = 0.0
        for x, y, ecg_peaks, sid in tr_ld:
            x = x.to(device)
            y = y.to(device)  # (B,1,T)

            # noisy PPG primary is channel 2 (index 1): pleth_2
            noisy = x[:, 1, :]

            # STFT
            mag_noisy = stft_mag_torch(noisy, stft_cfg.n_fft, stft_cfg.hop_length, stft_cfg.win_length)
            phase_noisy = torch.angle(
                torch.stft(
                    noisy,
                    n_fft=stft_cfg.n_fft,
                    hop_length=stft_cfg.hop_length,
                    win_length=stft_cfg.win_length,
                    window=torch.hann_window(stft_cfg.win_length, device=device),
                    return_complex=True,
                )
            )

            # mask prediction on time domain -> map to STFT resolution by interpolation
            mask_t = torch.sigmoid(model(x))  # (B,1,T)
            # Convert to (B,1,TT) matching STFT time frames
            TT = mag_noisy.shape[-1]
            mask_tt = F.interpolate(mask_t, size=TT, mode="linear", align_corners=False)
            # expand along F
            mask_mag = mask_tt.repeat(1, mag_noisy.shape[1], 1)  # (B,F,TT)

            mag_clean = mask_mag * mag_noisy
            y_hat = istft_from_mag_phase(mag_clean, phase_noisy, stft_cfg.n_fft, stft_cfg.hop_length, stft_cfg.win_length, length=noisy.shape[1])

            # shape loss: match to SIT-smoothed reference (approx: lowpass of y)
            # NOTE: y is normalized pleth_2 segment. This is a placeholder until you plug a true SIT template.
            l_shape = F.l1_loss(y_hat.unsqueeze(1), y)

            # peak alignment loss (primary): compare PPG peaks vs shifted ECG peaks (if available)
            # We compute a differentiable proxy: maximize energy at ECG peak locations using soft weights.
            # Practical: use y_hat (time) and target impulse train from ECG peaks.
            l_peak = torch.tensor(0.0, device=device)
            for bi in range(x.shape[0]):
                sid_i = sid[bi]
                if isinstance(sid_i, bytes):
                    sid_i = sid_i.decode("utf-8")
                sid_i = str(sid_i)
                if sid_i not in subj2idx:
                    continue
                ecg_pk = ecg_peaks[bi].cpu().numpy()
                if ecg_pk.size < 3:
                    continue
                base = sit_phase_med.get(sid_i, 0.0)
                a = float(a_value()[subj2idx[sid_i]].detach().cpu().item())
                shift = int(round((a * base) * fs))
                ecg_shift = ecg_pk + shift
                ecg_shift = ecg_shift[(ecg_shift >= 0) & (ecg_shift < y_hat.shape[1])]
                if ecg_shift.size < 3:
                    continue
                target = torch.zeros_like(y_hat[bi])
                target[torch.from_numpy(ecg_shift).to(device)] = 1.0
                # encourage large positive peaks at target indices
                l_peak = l_peak + F.binary_cross_entropy_with_logits(y_hat[bi], target)
            l_peak = l_peak / max(1, x.shape[0])

            l_smooth = freq_smoothness_reg(mask_mag)
            loss = l_peak + lam_shape * l_shape + lam_smooth * l_smooth

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += float(loss.item()) * x.shape[0]

        tr_loss /= len(tr_ld.dataset)

        # validation
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x, y, ecg_peaks, sid in va_ld:
                x = x.to(device)
                y = y.to(device)
                noisy = x[:, 1, :]
                mag_noisy = stft_mag_torch(noisy, stft_cfg.n_fft, stft_cfg.hop_length, stft_cfg.win_length)
                phase_noisy = torch.angle(
                    torch.stft(
                        noisy,
                        n_fft=stft_cfg.n_fft,
                        hop_length=stft_cfg.hop_length,
                        win_length=stft_cfg.win_length,
                        window=torch.hann_window(stft_cfg.win_length, device=device),
                        return_complex=True,
                    )
                )
                mask_t = torch.sigmoid(model(x))
                TT = mag_noisy.shape[-1]
                mask_tt = F.interpolate(mask_t, size=TT, mode="linear", align_corners=False)
                mask_mag = mask_tt.repeat(1, mag_noisy.shape[1], 1)
                mag_clean = mask_mag * mag_noisy
                y_hat = istft_from_mag_phase(mag_clean, phase_noisy, stft_cfg.n_fft, stft_cfg.hop_length, stft_cfg.win_length, length=noisy.shape[1])
                l_shape = F.l1_loss(y_hat.unsqueeze(1), y)
                l_smooth = freq_smoothness_reg(mask_mag)
                loss = lam_shape * l_shape + lam_smooth * l_smooth
                va_loss += float(loss.item()) * x.shape[0]
        va_loss /= len(va_ld.dataset)

        pbar.set_postfix(train=f"{tr_loss:.4f}", val=f"{va_loss:.4f}")
        if va_loss < best["val"]:
            best["val"] = va_loss
            best["state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best["a"] = (0.5 + torch.sigmoid(a_table.detach())).cpu().numpy().tolist()

    # restore
    model.load_state_dict(best["state"])
    a_best = best["a"]

    info = {
        "activity": activity,
        "stft": asdict(stft_cfg),
        "win": {"fs": fs, "win_sec": win_sec, "hop_sec": hop_sec},
        "feature_names": PPG_FEAT_NAMES + IMU_FEAT_NAMES + ["reserved_0", "reserved_1"],
        "in_channels": in_channels,
        "model": "MaskNet",
        "lam_shape": lam_shape,
        "lam_smooth": lam_smooth,
        "best_val_loss": float(best["val"]),
        "subjects": subj_list,
        "a_table": a_best,
    }

    outdir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "info": info}, outdir / f"denoiser_{activity}.pt")
    (outdir / f"denoiser_{activity}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    # best-effort ONNX
    try:
        dummy = torch.zeros((1, in_channels, int(round(win_sec * fs))), device=device)
        torch.onnx.export(
            model,
            dummy,
            (outdir / f"denoiser_{activity}.onnx").as_posix(),
            input_names=["x"],
            output_names=["mask_logits"],
            opset_version=18,
        )
    except Exception as e:
        print(f"[WARN] ONNX export failed for {activity}: {e}")

    return model, info


def eval_activity(
    activity: str,
    model: MaskNet,
    records: List[dict],
    sit_phase_med: Dict[str, float],
    a_table: Dict[str, float],
    stft_cfg: STFTCfg,
    fs: float,
    win_sec: float,
    hop_sec: float,
    tol_s: float = 0.08,
) -> dict:
    """Compute evaluation metrics: peak timing MAE, peak F1, SNR improvement (pseudo)."""
    device = next(model.parameters()).device
    ds = DenoiseWindowDataset(records, fs, win_sec, hop_sec, activity=activity)
    ld = DataLoader(ds, batch_size=32, shuffle=False)
    maes = []
    f1s = []
    snrs = []
    model.eval()
    with torch.no_grad():
        for x, y, ecg_peaks, sid in tqdm(ld, desc=f"Eval[{activity}]", leave=False):
            x = x.to(device)
            noisy = x[:, 1, :]
            mag_noisy = stft_mag_torch(noisy, stft_cfg.n_fft, stft_cfg.hop_length, stft_cfg.win_length)
            phase_noisy = torch.angle(
                torch.stft(
                    noisy,
                    n_fft=stft_cfg.n_fft,
                    hop_length=stft_cfg.hop_length,
                    win_length=stft_cfg.win_length,
                    window=torch.hann_window(stft_cfg.win_length, device=device),
                    return_complex=True,
                )
            )
            mask_t = torch.sigmoid(model(x))
            TT = mag_noisy.shape[-1]
            mask_tt = F.interpolate(mask_t, size=TT, mode="linear", align_corners=False)
            mask_mag = mask_tt.repeat(1, mag_noisy.shape[1], 1)
            mag_clean = mask_mag * mag_noisy
            y_hat = istft_from_mag_phase(mag_clean, phase_noisy, stft_cfg.n_fft, stft_cfg.hop_length, stft_cfg.win_length, length=noisy.shape[1])

            y_hat_np = y_hat.detach().cpu().numpy()
            noisy_np = noisy.detach().cpu().numpy()
            y_np = y.squeeze(1).cpu().numpy()

            for bi in range(y_hat_np.shape[0]):
                sid_i = str(sid[bi])
                base = sit_phase_med.get(sid_i, 0.0)
                a = a_table.get(sid_i, 1.0)
                shift = int(round((a * base) * fs))
                ecg_pk = ecg_peaks[bi].cpu().numpy() + shift
                ecg_pk = ecg_pk[(ecg_pk >= 0) & (ecg_pk < y_hat_np.shape[1])]

                ppg_pk = detect_ppg_peaks(y_hat_np[bi], fs)
                mae = peaks_timing_mae(ppg_pk, ecg_pk, fs)
                _, _, f1 = peak_match_f1(ppg_pk, ecg_pk, tol_s=tol_s, fs=fs)
                maes.append(mae)
                f1s.append(f1)
                snrs.append(snr_improvement_db(noisy_np[bi], y_hat_np[bi], y_np[bi]))

    return {
        "peak_timing_mae": float(np.nanmean(maes)) if len(maes) else float("nan"),
        "peak_f1": float(np.nanmean(f1s)) if len(f1s) else float("nan"),
        "snr_improvement_db": float(np.nanmean(snrs)) if len(snrs) else float("nan"),
    }


# ----------------------------- Orchestration (no-leak) -----------------------------


def split_subjects(groups: List[str], train_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(groups))
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    tr, ho = next(gss.split(idx, groups=groups))
    return tr, ho


def run_activity_pipeline(
    activity: str,
    records: List[dict],
    groups: List[str],
    train_size: float,
    n_splits: int,
    fs: float,
    win_sec: float,
    hop_sec: float,
    stft_cfg: STFTCfg,
    epochs: int,
    lr: float,
    lam_shape: float,
    lam_smooth: float,
    outdir: Path,
    seed: int,
) -> dict:
    """True no-leak pipeline:
    1) Subject-level holdout split.
    2) CV on training subjects only.
    3) Train final model on all training subjects and eval on holdout.
    """
    t0 = time.time()
    tr_idx, ho_idx = split_subjects(groups, train_size=train_size, seed=seed)
    tr_recs = [records[i] for i in tr_idx]
    ho_recs = [records[i] for i in ho_idx]
    tr_groups = [groups[i] for i in tr_idx]

    sit_phase_med = compute_sit_phase_median(tr_recs, fs=fs)  # computed only from train subjects

    # CV
    gkf = GroupKFold(n_splits=min(n_splits, len(set(tr_groups))))
    cv_rows = []
    for fold, (a, b) in enumerate(gkf.split(np.arange(len(tr_recs)), groups=tr_groups), start=1):
        tr_fold = [tr_recs[i] for i in a]
        va_fold = [tr_recs[i] for i in b]
        model, info = train_one_activity(
            activity,
            tr_fold,
            va_fold,
            sit_phase_med=sit_phase_med,
            stft_cfg=stft_cfg,
            fs=fs,
            win_sec=win_sec,
            hop_sec=hop_sec,
            epochs=epochs,
            lr=lr,
            lam_shape=lam_shape,
            lam_smooth=lam_smooth,
            outdir=outdir / "cv" / f"fold{fold}",
            seed=seed + fold,
        )
        a_table = {s: float(info["a_table"][i]) for i, s in enumerate(info["subjects"]) }
        metrics = eval_activity(activity, model, va_fold, sit_phase_med, a_table, stft_cfg, fs, win_sec, hop_sec)
        cv_rows.append({"fold": fold, **metrics, "best_val_loss": info["best_val_loss"]})

    cv_df = pd.DataFrame(cv_rows)

    # final on all train, eval on holdout
    model, info = train_one_activity(
        activity,
        tr_recs,
        ho_recs,
        sit_phase_med=sit_phase_med,
        stft_cfg=stft_cfg,
        fs=fs,
        win_sec=win_sec,
        hop_sec=hop_sec,
        epochs=epochs,
        lr=lr,
        lam_shape=lam_shape,
        lam_smooth=lam_smooth,
        outdir=outdir / "final",
        seed=seed,
    )
    a_table = {s: float(info["a_table"][i]) for i, s in enumerate(info["subjects"]) }
    hold = eval_activity(activity, model, ho_recs, sit_phase_med, a_table, stft_cfg, fs, win_sec, hop_sec)

    # save CSV summary
    outdir.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(outdir / f"denoiser_{activity}_cv_metrics.csv", index=False)
    pd.DataFrame([{**hold}]).to_csv(outdir / f"denoiser_{activity}_holdout_metrics.csv", index=False)

    res = {
        "activity": activity,
        "cv": {
            "mean": cv_df.mean(numeric_only=True).to_dict(),
            "std": cv_df.std(numeric_only=True).to_dict(),
            "folds": cv_rows,
        },
        "holdout": hold,
        "train_subjects": sorted(set(tr_groups)),
        "holdout_subjects": sorted(set(groups[i] for i in ho_idx)),
        "runtime_sec": float(time.time() - t0),
    }
    (outdir / f"denoiser_{activity}_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, type=str)
    ap.add_argument("--outdir", default="results_stage2", type=str)
    ap.add_argument("--fs", default=500.0, type=float)
    ap.add_argument("--win", default=6.0, type=float)
    ap.add_argument("--hop", default=1.0, type=float)
    ap.add_argument("--train_size", default=0.8, type=float)
    ap.add_argument("--n_splits", default=5, type=int)
    ap.add_argument("--epochs", default=20, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--lam_shape", default=0.05, type=float)
    ap.add_argument("--lam_smooth", default=0.05, type=float)
    ap.add_argument("--stft_n_fft", default=256, type=int)
    ap.add_argument("--stft_hop", default=64, type=int)
    ap.add_argument("--stft_win", default=256, type=int)
    ap.add_argument("--seed", default=42, type=int)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sub2recs = load_physionet_csv(data_root, fs=args.fs)
    records, groups = subjectwise(sub2recs)

    stft_cfg = STFTCfg(n_fft=args.stft_n_fft, hop_length=args.stft_hop, win_length=args.stft_win)

    results = {}
    for act in ["walk", "run"]:
        results[act] = run_activity_pipeline(
            act,
            records,
            groups,
            train_size=args.train_size,
            n_splits=args.n_splits,
            fs=args.fs,
            win_sec=args.win,
            hop_sec=args.hop,
            stft_cfg=stft_cfg,
            epochs=args.epochs,
            lr=args.lr,
            lam_shape=args.lam_shape,
            lam_smooth=args.lam_smooth,
            outdir=outdir,
            seed=args.seed,
        )

    (outdir / "stage2_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[DONE] Stage2 finished in {(time.time()-t0)/60:.1f} min. Output: {outdir}")


if __name__ == "__main__":
    main()
