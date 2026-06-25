# pttppg_pipeline_v7_2_noleak_viz.py
# ------------------------------------------------------------
# v7.2 (NO-LEAK + VIZ) — subject-split first, then train/eval
#
# Additions in this version:
#  - For AE detector: export intuitive plots for BOTH CV and holdout:
#       * confusion_matrix.png
#       * roc.png
#       * pr.png
#       * score_hist.png
#
# Guarantees:
#  - Holdout subjects NEVER used for training, early stopping, threshold fitting, or hyperparam selection.
#  - CV performed ONLY within train subjects with subject-wise splits (GroupKFold).
#  - Holdout model uses inner_train/inner_val (from train partition) for early stopping/threshold, and evaluates on holdout.
# ------------------------------------------------------------

import os, json, argparse, warnings, time
warnings.filterwarnings("ignore")
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn import metrics

from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===================== Models =====================
class CNNBiLSTMAE(nn.Module):
    def __init__(self, in_ch: int = 1, hidden: int = 256, bottleneck: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=5, padding=2), nn.ReLU(True),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(True),
            nn.MaxPool1d(4),
            nn.Dropout(0.3),
        )
        self.bi_lstm = nn.LSTM(64, hidden // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.bottleneck = nn.Linear(hidden, bottleneck)
        self.decoder_lstm = nn.LSTM(bottleneck, hidden // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.to_channels = nn.Conv1d(hidden, in_ch, kernel_size=1)

    def forward(self, x):
        B, C, T = x.shape
        z = self.encoder(x).permute(0, 2, 1)
        z, _ = self.bi_lstm(z)
        z = self.bottleneck(z)
        z, _ = self.decoder_lstm(z)
        z = z.permute(0, 2, 1)
        out = F.interpolate(z, size=T, mode="linear", align_corners=False)
        return self.to_channels(out)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=p), nn.ReLU(True),
            nn.Conv1d(out_ch, out_ch, k, padding=p), nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class UNet1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base);     self.pool1 = nn.MaxPool1d(2)
        self.enc2 = ConvBlock(base, base*2);    self.pool2 = nn.MaxPool1d(2)
        self.enc3 = ConvBlock(base*2, base*4);  self.pool3 = nn.MaxPool1d(2)
        self.enc4 = ConvBlock(base*4, base*8)
        self.up3 = nn.ConvTranspose1d(base*8, base*4, 2, 2); self.dec3 = ConvBlock(base*8, base*4)
        self.up2 = nn.ConvTranspose1d(base*4, base*2, 2, 2); self.dec2 = ConvBlock(base*4, base*2)
        self.up1 = nn.ConvTranspose1d(base*2, base, 2, 2);   self.dec1 = ConvBlock(base*2, base)
        self.out = nn.Conv1d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        d3 = self.up3(e4); d3 = torch.cat([d3, e3], 1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], 1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], 1); d1 = self.dec1(d1)
        return self.out(d1)


class MultiResSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=(64, 128, 256), hops=(16, 32, 64), wins=(64, 128, 256)):
        super().__init__()
        self.fft = fft_sizes; self.h = hops; self.w = wins

    def stft_mag(self, x, n, h, w):
        return torch.abs(torch.stft(x, n_fft=n, hop_length=h, win_length=w, return_complex=True))

    def forward(self, x, y):
        x = x.squeeze(1); y = y.squeeze(1)
        loss = 0.0
        for n, h, w in zip(self.fft, self.h, self.w):
            loss = loss + F.l1_loss(self.stft_mag(x, n, h, w), self.stft_mag(y, n, h, w))
        return loss / len(self.fft)


class HRBandEnergyReg(nn.Module):
    def __init__(self, fs: int, band=(0.6, 3.5), n_fft=256, hop=64, win_length=256, w_in=0.05, w_out=0.05):
        super().__init__()
        f = torch.linspace(0, fs/2, n_fft//2 + 1)
        self.register_buffer("mask_in",  ((f >= band[0]) & (f <= band[1])).float().view(1, -1, 1))
        self.register_buffer("mask_out", ((f < band[0]) | (f > band[1])).float().view(1, -1, 1))
        self.n = n_fft; self.h = hop; self.w = win_length; self.wi = w_in; self.wo = w_out

    def forward(self, y):
        y = y.squeeze(1)
        Y = torch.stft(y, n_fft=self.n, hop_length=self.h, win_length=self.w, return_complex=True)
        mag = torch.abs(Y)
        mask_in  = self.mask_in.to(mag.device)
        mask_out = self.mask_out.to(mag.device)
        return self.wo*(mag*mask_out).mean() - self.wi*(mag*mask_in).mean()


# ===================== Utils =====================
try:
    import pywt
except Exception:
    pywt = None


def zscore(x: np.ndarray) -> np.ndarray:
    m, s = float(np.mean(x)), float(np.std(x) + 1e-8)
    return (x - m) / s


def dwt_compress(x: np.ndarray, wavelet: str = "db4", level: int = 2) -> np.ndarray:
    if pywt is None:
        return x
    c = pywt.wavedec(x, wavelet, level=level)
    a = c[0]
    xp = np.linspace(0, 1, len(a))
    xq = np.linspace(0, 1, len(x))
    return np.interp(xq, xp, a)


def butter_bandpass(sig: np.ndarray, fs: float, lo: float = 0.5, hi: float = 8.0, order: int = 3) -> np.ndarray:
    try:
        from scipy.signal import butter, filtfilt
    except Exception:
        return sig
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype="band")
    return filtfilt(b, a, sig)


def split_windows(N: int, fs: float, win: float, hop: float):
    w = int(win * fs); h = int(hop * fs)
    for s in range(0, max(1, N - w + 1), h):
        yield s, s + w


def acc_rms(acc_win: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(acc_win**2, axis=1))))


def infer_activity_from_stem(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) < 2:
        return "unknown"
    act = parts[1].lower().strip()
    return act if act in {"sit", "walk", "run"} else "unknown"


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===================== Plotting helpers =====================
def summarize_rows(rows: List[dict], keys: List[str]) -> dict:
    out = {"n": int(len(rows))}
    for k in keys:
        vals = [r.get(k, float("nan")) for r in rows]
        out[f"{k}_mean"] = float(np.nanmean(vals))
        out[f"{k}_std"]  = float(np.nanstd(vals))
    return out


def save_csv(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_cv_holdout(out_png: Path, title: str, cv_summary: dict, holdout_metrics: dict, keys: List[str]):
    means = [cv_summary.get(f"{k}_mean", np.nan) for k in keys]
    stds  = [cv_summary.get(f"{k}_std", np.nan) for k in keys]
    hold  = [holdout_metrics.get(k, np.nan) for k in keys]

    x = np.arange(len(keys))
    plt.figure(figsize=(11, 4))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.scatter(x, hold, marker="D")
    plt.xticks(x, keys, rotation=30, ha="right")
    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_confusion_matrix_png(out_png: Path, y_true: np.ndarray, y_pred: np.ndarray, title: str):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # rows true, cols pred
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["NoMotion(0)", "Motion(1)"])
    plt.yticks([0, 1], ["NoMotion(0)", "Motion(1)"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=170)
    plt.close()


def plot_roc_curve_png(out_png: Path, y_true: np.ndarray, scores: np.ndarray, title: str):
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = metrics.roc_curve(y_true, scores)
    auc = metrics.roc_auc_score(y_true, scores)
    plt.figure(figsize=(6, 4.5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"{title} | ROC AUC={auc:.3f}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=170)
    plt.close()


def plot_pr_curve_png(out_png: Path, y_true: np.ndarray, scores: np.ndarray, title: str):
    if len(np.unique(y_true)) < 2:
        return
    prec, rec, _ = metrics.precision_recall_curve(y_true, scores)
    ap = metrics.average_precision_score(y_true, scores)
    plt.figure(figsize=(6, 4.5))
    plt.plot(rec, prec)
    plt.title(f"{title} | AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=170)
    plt.close()


def plot_score_hist_png(out_png: Path, y_true: np.ndarray, scores: np.ndarray, thr: float, title: str):
    plt.figure(figsize=(8, 4.5))
    m0 = (y_true == 0)
    m1 = (y_true == 1)
    if m0.any():
        plt.hist(scores[m0], bins=60, alpha=0.6, label="NoMotion(0)")
    if m1.any():
        plt.hist(scores[m1], bins=60, alpha=0.6, label="Motion(1)")
    plt.axvline(thr, linestyle="--")
    plt.title(title)
    plt.xlabel("Reconstruction loss (score)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=170)
    plt.close()


# ===================== Data Loading (CSV) =====================
CAND = {
    "time": ["time", "Time", "t"],
    "ecg": ["ecg", "ECG"],
    "peaks": ["peaks", "Rpeaks", "r_peaks"],
    "pleth_4": ["pleth_4"],
    "pleth_5": ["pleth_5"],
    "pleth_6": ["pleth_6"],
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


def load_physionet_csv(root: Path, fs: float) -> Dict[str, List[dict]]:
    csv_dir = root / "csv"
    if not csv_dir.exists():
        csv_dir = root / "files" / "pulse-transit-time-ppg" / "1.1.0" / "csv"

    paths = sorted([p for p in csv_dir.glob("s*_*.csv") if p.name != "subjects_info.csv"])
    sub2recs: Dict[str, List[dict]] = {}

    for p in tqdm(paths, desc="Loading CSV files", leave=False):
        df = pd.read_csv(p)
        m = {k: _pick(df, v) for k, v in CAND.items()}
        rec: dict = {}

        for k in CAND.keys():
            if m.get(k) is None:
                continue
            col = m[k]

            if k == "time":
                t = pd.to_datetime(df[col], errors="coerce")
                if t.notna().all():
                    rec["time"] = (t - t.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float32)
                else:
                    rec["time"] = (np.arange(len(df), dtype=np.float32) / float(fs))
                continue

            rec[k] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)

        if "time" not in rec:
            rec["time"] = (np.arange(len(df), dtype=np.float32) / float(fs))

        sid = p.stem.split("_")[0]
        rec["subject"] = sid
        rec["activity"] = infer_activity_from_stem(p.stem)
        rec["filename"] = p.name

        sub2recs.setdefault(sid, []).append(rec)

    return sub2recs


# ===================== Datasets =====================
class DatasetAE(Dataset):
    """
    AE dataset with strict activity filter and optional clean-only window selection.
    - label is returned for evaluation (motion vs no-motion by IMU threshold).
    """

    def __init__(
        self,
        records: List[dict],
        fs: float,
        win: float,
        hop: float,
        motion_thresh: float,
        activity_filter: Optional[str] = None,
        clean_only: bool = False,
    ):
        self.records = records
        self.fs = fs
        self.win = win
        self.hop = hop
        self.motion_thresh = motion_thresh
        self.activity_filter = activity_filter
        self.clean_only = clean_only

        self.index: List[Tuple[int, int, int]] = []
        for i, r in enumerate(records):
            if self.activity_filter is not None and r.get("activity") != self.activity_filter:
                continue
            N = len(r["pleth_5"]) if "pleth_5" in r else len(r["pleth_4"])
            for s, e in split_windows(N, fs, win, hop):
                ppg_ref = (r["pleth_5"] if "pleth_5" in r else r["pleth_4"])[s:e].astype(np.float32)
                ax = r.get("a_x", np.zeros_like(ppg_ref))[s:e]
                ay = r.get("a_y", np.zeros_like(ppg_ref))[s:e]
                az = r.get("a_z", np.zeros_like(ppg_ref))[s:e]
                acc = np.column_stack([ax, ay, az]).astype(np.float32)
                label = int(acc_rms(acc) > self.motion_thresh)
                if self.clean_only and label == 1:
                    continue
                self.index.append((i, s, e))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i, s, e = self.index[idx]
        r = self.records[i]
        ppg = (r["pleth_5"] if "pleth_5" in r else r["pleth_4"])[s:e].astype(np.float32)

        ax = r.get("a_x", np.zeros_like(ppg))[s:e]
        ay = r.get("a_y", np.zeros_like(ppg))[s:e]
        az = r.get("a_z", np.zeros_like(ppg))[s:e]
        acc = np.column_stack([ax, ay, az]).astype(np.float32)
        label = int(acc_rms(acc) > self.motion_thresh)

        x = zscore(dwt_compress(ppg))[None, :].astype(np.float32)
        y = zscore(ppg)[None, :].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(label, dtype=torch.long)


def peaks_to_hr_bpm(peaks: np.ndarray, fs: float, min_peaks: int = 2) -> float:
    idx = np.flatnonzero(peaks > 0.5)
    if len(idx) < min_peaks:
        return float("nan")
    rr = np.diff(idx) / float(fs)
    rr = rr[(rr > 0.25) & (rr < 2.5)]  # 24-240 bpm
    if rr.size == 0:
        return float("nan")
    return float(60.0 / float(np.mean(rr)))


class DatasetDenoise(Dataset):
    """
    Inputs (NO ECG/peaks as input):
      - p5 noisy (fallback p4) : 1
      - acc (ax,ay,az)         : 3
      - gyro (gx,gy,gz)        : 3
      - p4 aux                 : 1
      -> total 8 channels

    Target proxy:
      - bandpassed p5

    Supervision-only:
      - hr_ecg from peaks (if exists)
    """

    def __init__(
        self,
        records: List[dict],
        fs: float,
        win: float,
        hop: float,
        motion_thresh: float,
        activity_filter: str,
    ):
        self.records = records
        self.fs = fs
        self.win = win
        self.hop = hop
        self.motion_thresh = motion_thresh
        self.activity_filter = activity_filter

        self.index: List[Tuple[int, int, int]] = []
        for i, r in enumerate(records):
            if r.get("activity") != self.activity_filter:
                continue
            N = len(r["pleth_5"]) if "pleth_5" in r else len(r["pleth_4"])
            for s, e in split_windows(N, fs, win, hop):
                self.index.append((i, s, e))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i, s, e = self.index[idx]
        r = self.records[i]

        p5 = (r["pleth_5"] if "pleth_5" in r else r["pleth_4"])[s:e].astype(np.float32)
        p4 = r.get("pleth_4", np.zeros_like(p5))[s:e].astype(np.float32)

        ax = r.get("a_x", np.zeros_like(p5))[s:e]
        ay = r.get("a_y", np.zeros_like(p5))[s:e]
        az = r.get("a_z", np.zeros_like(p5))[s:e]
        gx = r.get("g_x", np.zeros_like(p5))[s:e]
        gy = r.get("g_y", np.zeros_like(p5))[s:e]
        gz = r.get("g_z", np.zeros_like(p5))[s:e]

        x_list = [
            zscore(p5)[None, :],
            zscore(np.array([ax, ay, az], dtype=np.float32)),
            zscore(np.array([gx, gy, gz], dtype=np.float32)),
            zscore(p4)[None, :],
        ]
        x = np.vstack(x_list).astype(np.float32)

        y_proxy = zscore(butter_bandpass(p5, self.fs, 0.5, 8.0))[None, :].astype(np.float32)

        is_clean = int(acc_rms(np.column_stack([ax, ay, az]).astype(np.float32)) <= self.motion_thresh)

        if "peaks" in r:
            hr_ecg = peaks_to_hr_bpm(r["peaks"][s:e].astype(np.float32), fs=self.fs)
        else:
            hr_ecg = float("nan")

        return (
            torch.from_numpy(x),
            torch.from_numpy(y_proxy),
            torch.tensor(is_clean, dtype=torch.float32),
            torch.tensor(hr_ecg, dtype=torch.float32),
        )


# ===================== ECG HR supervision (differentiable) =====================
class SoftHRFromFFT(nn.Module):
    def __init__(self, fs: float, fmin: float = 0.6, fmax: float = 3.5, tau: float = 0.1):
        super().__init__()
        self.fs = float(fs); self.fmin = float(fmin); self.fmax = float(fmax); self.tau = float(tau)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() == 3:
            y = y.squeeze(1)
        B, T = y.shape
        Y = torch.fft.rfft(y, dim=-1)
        mag = torch.abs(Y)

        freqs = torch.fft.rfftfreq(T, d=1.0 / self.fs).to(mag.device)
        mask = (freqs >= self.fmin) & (freqs <= self.fmax)
        freqs_b = freqs[mask]
        mag_b = mag[:, mask]

        logits = mag_b / self.tau
        logits = logits - logits.max(dim=1, keepdim=True).values
        w = torch.softmax(logits, dim=1)
        f_hat = (w * freqs_b[None, :]).sum(dim=1)
        return 60.0 * f_hat


# ===================== Splits (NO-LEAK) =====================
def make_subject_splits(
    all_subjects: List[str],
    holdout_ratio: float,
    inner_val_ratio: float,
    seed: int,
    n_splits: int
) -> dict:
    subs = np.array(sorted(list(set(all_subjects))))
    idx = np.arange(len(subs))

    gss = GroupShuffleSplit(n_splits=1, train_size=1.0 - holdout_ratio, random_state=seed)
    tr_idx, ho_idx = next(gss.split(idx, groups=subs))
    train_subs = subs[tr_idx].tolist()
    hold_subs  = subs[ho_idx].tolist()

    subs_tr = np.array(sorted(train_subs))
    idx_tr  = np.arange(len(subs_tr))

    # inner split for early stopping + threshold fitting
    gss2 = GroupShuffleSplit(n_splits=1, train_size=1.0 - inner_val_ratio, random_state=seed + 11)
    inner_tr_idx, inner_va_idx = next(gss2.split(idx_tr, groups=subs_tr))
    inner_train_subs = subs_tr[inner_tr_idx].tolist()
    inner_val_subs   = subs_tr[inner_va_idx].tolist()

    gkf = GroupKFold(n_splits=min(n_splits, len(subs_tr)))
    cv_folds = [(subs_tr[a].tolist(), subs_tr[b].tolist()) for a, b in gkf.split(idx_tr, groups=subs_tr)]

    return {
        "train_subjects": train_subs,
        "holdout_subjects": hold_subs,
        "inner_train_subjects": inner_train_subs,
        "inner_val_subjects": inner_val_subs,
        "cv_folds": cv_folds,
    }


def records_from_subjects(sub2recs: Dict[str, List[dict]], subjects: List[str]) -> List[dict]:
    return [r for sid in subjects for r in sub2recs.get(sid, [])]


# ===================== Train / Eval (AE) =====================
def train_detector(
    train_ds: DatasetAE,
    val_ds: DatasetAE,
    epochs: int,
    lr: float,
    batch_train: int = 64,
    batch_val: int = 128,
    progress_desc: str = "AE epochs"
) -> Tuple[nn.Module, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNNBiLSTMAE(in_ch=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    TL = DataLoader(train_ds, batch_size=batch_train, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    VL = DataLoader(val_ds, batch_size=batch_val, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    best = {"val": 1e9, "state": None}

    for ep in tqdm(range(1, epochs + 1), desc=progress_desc, leave=False):
        model.train()
        tr = 0.0
        for x, y, _ in TL:
            x, y = x.to(device), y.to(device)
            yhat = model(x)
            loss = F.mse_loss(yhat, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += float(loss.detach().cpu().item()) * x.size(0)
        tr /= max(1, len(TL.dataset))

        model.eval()
        va = 0.0
        with torch.no_grad():
            for x, y, _ in VL:
                x, y = x.to(device), y.to(device)
                yhat = model(x)
                va += float(F.mse_loss(yhat, y).detach().cpu().item()) * x.size(0)
        va /= max(1, len(VL.dataset))

        if va < best["val"]:
            best["val"] = float(va)
            best["state"] = {k: v.cpu() for k, v in model.state_dict().items()}

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, float(best["val"])


def recon_losses(model: nn.Module, ds: DatasetAE) -> np.ndarray:
    device = next(model.parameters()).device
    L = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    losses = []
    with torch.no_grad():
        for x, y, _ in L:
            x, y = x.to(device), y.to(device)
            yhat = model(x)
            l = F.mse_loss(yhat, y, reduction="none").mean(dim=[1, 2]).detach().cpu().numpy()
            losses.append(l)
    return np.concatenate(losses) if losses else np.array([], dtype=np.float32)


def fit_detector_threshold_from_inner_val(
    model: nn.Module,
    inner_val_clean_sit_ds: DatasetAE,
    quantile: float = 0.95
) -> float:
    losses = recon_losses(model, inner_val_clean_sit_ds)
    if losses.size == 0:
        return float("nan")
    return float(np.quantile(losses, quantile))


def eval_detector_with_arrays(model: nn.Module, ds: DatasetAE, thr: float) -> dict:
    """
    Returns metrics + arrays for visualization:
      - y_true (0/1): from IMU threshold label
      - y_pred (0/1): score > thr
      - scores: recon losses
    """
    device = next(model.parameters()).device
    L = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    losses_list = []
    ytrue_list = []
    with torch.no_grad():
        for x, y, lab in L:
            x, y = x.to(device), y.to(device)
            yhat = model(x)
            loss = F.mse_loss(yhat, y, reduction="none").mean(dim=[1, 2]).detach().cpu().numpy()
            losses_list.append(loss)
            ytrue_list.append(lab.numpy())

    if not losses_list:
        return {
            "threshold": float(thr),
            "scores": np.array([], dtype=np.float32),
            "y_true": np.array([], dtype=np.int64),
            "y_pred": np.array([], dtype=np.int64),
            "n_windows": 0,
            "pr_auc": float("nan"),
            "roc_auc": float("nan"),
            "f1": float("nan"),
            "bal_acc": float("nan"),
        }

    scores = np.concatenate(losses_list)
    y_true = np.concatenate(ytrue_list).astype(int)
    y_pred = (scores > thr).astype(int)

    out = {"threshold": float(thr), "n_windows": int(scores.size)}

    if len(np.unique(y_true)) > 1:
        out["pr_auc"] = float(metrics.average_precision_score(y_true, scores))
        out["roc_auc"] = float(metrics.roc_auc_score(y_true, scores))
    else:
        out["pr_auc"] = float("nan")
        out["roc_auc"] = float("nan")

    if len(np.unique(y_pred)) > 1 and len(np.unique(y_true)) > 1:
        out["f1"] = float(metrics.f1_score(y_true, y_pred))
        out["bal_acc"] = float(metrics.balanced_accuracy_score(y_true, y_pred))
    else:
        out["f1"] = float("nan")
        out["bal_acc"] = float("nan")

    out["scores"] = scores
    out["y_true"] = y_true
    out["y_pred"] = y_pred
    return out


# ===================== Train / Eval (Denoiser) =====================
def train_denoiser(
    train_ds: DatasetDenoise,
    val_ds: DatasetDenoise,
    fs: float,
    epochs: int,
    lr: float,
    lam_proxy: float,
    lam_spec: float,
    lam_hrband: float,
    lam_cons: float,
    lam_ecg_hr: float,
    ecg_tau: float,
    batch_train: int = 16,
    batch_val: int = 32,
    progress_desc: str = "Denoiser epochs"
) -> Tuple[nn.Module, dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_ch = next(iter(DataLoader(train_ds, batch_size=1, shuffle=False)))[0].shape[1]

    model = UNet1D(in_ch=in_ch, out_ch=1, base=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    spec = MultiResSTFTLoss().to(device)
    hrband = HRBandEnergyReg(fs=int(fs), w_in=0.05, w_out=0.05).to(device)
    soft_hr = SoftHRFromFFT(fs=fs, tau=ecg_tau).to(device)

    TL = DataLoader(train_ds, batch_size=batch_train, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    VL = DataLoader(val_ds, batch_size=batch_val, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    best = {"val": 1e9, "state": None, "val_ecg_hr": float("nan")}

    for ep in tqdm(range(1, epochs + 1), desc=progress_desc, leave=False):
        model.train()
        tr = 0.0
        for x, y_proxy, is_clean, hr_ecg in TL:
            x = x.to(device)
            y_proxy = y_proxy.to(device)
            is_clean = is_clean.to(device)
            hr_ecg = hr_ecg.to(device)

            yhat = model(x)

            xa = x.clone(); xb = x.clone()
            xa[:, 0:1, :] = xa[:, 0:1, :] + torch.randn_like(xa[:, 0:1, :]) * 0.01
            xb[:, 0:1, :] = xb[:, 0:1, :] + torch.randn_like(xb[:, 0:1, :]) * 0.01
            ya = model(xa); yb = model(xb)

            if (is_clean > 0.5).any():
                l_proxy_term = F.l1_loss(yhat[is_clean > 0.5], y_proxy[is_clean > 0.5])
            else:
                l_proxy_term = torch.tensor(0.0, device=device)

            l_spec_term = spec(yhat, y_proxy)
            l_hrb_term = hrband(yhat)
            l_cons_term = F.l1_loss(ya, yb)

            l_ecg_term = torch.tensor(0.0, device=device)
            if lam_ecg_hr > 0:
                mask = torch.isfinite(hr_ecg)
                if mask.any():
                    hr_hat = soft_hr(yhat[mask])
                    l_ecg_term = F.l1_loss(hr_hat, hr_ecg[mask])

            loss = (
                lam_proxy * l_proxy_term
                + lam_spec * l_spec_term
                + lam_hrband * l_hrb_term
                + lam_cons * l_cons_term
                + lam_ecg_hr * l_ecg_term
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += float(loss.detach().cpu().item()) * x.size(0)

        tr /= max(1, len(TL.dataset))

        model.eval()
        va = 0.0
        va_hr_sum = 0.0
        va_hr_n = 0

        with torch.no_grad():
            for x, y_proxy, is_clean, hr_ecg in VL:
                x = x.to(device)
                y_proxy = y_proxy.to(device)
                hr_ecg = hr_ecg.to(device)

                yhat = model(x)
                l = lam_proxy * F.l1_loss(yhat, y_proxy) + lam_spec * spec(yhat, y_proxy) + lam_hrband * hrband(yhat)

                if lam_ecg_hr > 0:
                    mask = torch.isfinite(hr_ecg)
                    if mask.any():
                        hr_hat = soft_hr(yhat[mask])
                        l_ecg = F.l1_loss(hr_hat, hr_ecg[mask])
                        l = l + lam_ecg_hr * l_ecg
                        va_hr_sum += float(l_ecg.detach().cpu().item()) * int(mask.sum().item())
                        va_hr_n += int(mask.sum().item())

                va += float(l.detach().cpu().item()) * x.size(0)

        va /= max(1, len(VL.dataset))
        va_hr_mae = (va_hr_sum / va_hr_n) if va_hr_n > 0 else float("nan")

        if va < best["val"]:
            best["val"] = float(va)
            best["val_ecg_hr"] = float(va_hr_mae)
            best["state"] = {k: v.cpu() for k, v in model.state_dict().items()}

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, best


def eval_denoiser(model: nn.Module, ds: DatasetDenoise, fs: float, ecg_tau: float) -> dict:
    device = next(model.parameters()).device
    L = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    l1s: List[float] = []
    snr_imp: List[float] = []
    hr_losses: List[float] = []
    ecg_hr_err: List[float] = []

    hrband = HRBandEnergyReg(fs=int(fs)).to(device)
    soft_hr = SoftHRFromFFT(fs=fs, tau=ecg_tau).to(device)

    with torch.no_grad():
        for x, y_proxy, is_clean, hr_ecg in L:
            x = x.to(device)
            y_proxy = y_proxy.to(device)
            hr_ecg = hr_ecg.to(device)

            yhat = model(x)

            l1s.append(float(F.l1_loss(yhat, y_proxy).cpu().item()))

            noisy = x[:, 0:1, :]
            err_b = (noisy - y_proxy).pow(2).mean().cpu().item()
            err_a = (yhat - y_proxy).pow(2).mean().cpu().item()
            snr_imp.append(float(10*np.log10((err_b + 1e-9) / (err_a + 1e-9))))

            hr_losses.append(float(hrband(yhat).cpu().item()))

            mask = torch.isfinite(hr_ecg)
            if mask.any():
                hr_hat = soft_hr(yhat[mask])
                ecg_hr_err.append(float(torch.abs(hr_hat - hr_ecg[mask]).mean().cpu().item()))

    return {
        "l1": float(np.mean(l1s)) if l1s else float("nan"),
        "snr_improvement_db": float(np.mean(snr_imp)) if snr_imp else float("nan"),
        "hr_band_loss": float(np.mean(hr_losses)) if hr_losses else float("nan"),
        "ecg_hr_mae_bpm": float(np.mean(ecg_hr_err)) if ecg_hr_err else float("nan"),
        "n_windows": int(len(ds)),
    }


# ===================== Orchestration (NO-LEAK + VIZ) =====================
def run_ae_cv_and_holdout_noleak(
    sub2recs: Dict[str, List[dict]],
    splits: dict,
    fs: float,
    win: float,
    hop: float,
    motion_thresh: float,
    epochs_ae: int,
    lr: float,
    outdir: Path,
    eval_domains: Tuple[str, ...] = ("sit", "walk", "run"),
    thr_quantile: float = 0.95,
) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)

    holdout_subjects = splits["holdout_subjects"]
    inner_train_subjects = splits["inner_train_subjects"]
    inner_val_subjects = splits["inner_val_subjects"]
    cv_folds = splits["cv_folds"]

    inner_train_recs = records_from_subjects(sub2recs, inner_train_subjects)
    inner_val_recs = records_from_subjects(sub2recs, inner_val_subjects)
    holdout_recs = records_from_subjects(sub2recs, holdout_subjects)

    res = {"splits": splits, "holdout": {}, "cv": {}, "threshold_fit": {}}

    # Holdout AE train: sit-clean only
    tr_ds = DatasetAE(inner_train_recs, fs, win, hop, motion_thresh, activity_filter="sit", clean_only=True)
    va_ds = DatasetAE(inner_val_recs, fs, win, hop, motion_thresh, activity_filter="sit", clean_only=True)

    if len(tr_ds) == 0 or len(va_ds) == 0:
        for dom in eval_domains:
            res["holdout"][dom] = {"domain": dom, "error": "empty_inner_train_or_val_sit_clean"}
            res["cv"][dom] = {"folds": [], "summary": {"error": "skipped"}}
        (outdir / "ae_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        return res

    ae_hold, best_val = train_detector(
        tr_ds, va_ds,
        epochs=epochs_ae, lr=lr,
        progress_desc="AE holdout-train epochs (inner train/val)"
    )

    thr = fit_detector_threshold_from_inner_val(ae_hold, va_ds, quantile=thr_quantile)
    res["threshold_fit"] = {"source": "inner_val_sit_clean", "quantile": float(thr_quantile), "threshold": float(thr)}

    # ---- Holdout eval + plots per domain ----
    for dom in eval_domains:
        ho_ds = DatasetAE(holdout_recs, fs, win, hop, motion_thresh, activity_filter=dom, clean_only=False)
        if len(ho_ds) == 0:
            res["holdout"][dom] = {"domain": dom, "error": "empty_holdout_domain"}
            continue

        m = eval_detector_with_arrays(ae_hold, ho_ds, thr=thr)

        # plots
        dom_dir = outdir / "holdout_plots" / dom
        dom_dir.mkdir(parents=True, exist_ok=True)
        y_true = m["y_true"]; y_pred = m["y_pred"]; scores = m["scores"]
        if scores.size > 0:
            plot_confusion_matrix_png(dom_dir / "confusion_matrix.png", y_true, y_pred,
                                      title=f"AE Holdout Confusion | domain={dom}")
            plot_roc_curve_png(dom_dir / "roc.png", y_true, scores,
                               title=f"AE Holdout ROC | domain={dom}")
            plot_pr_curve_png(dom_dir / "pr.png", y_true, scores,
                              title=f"AE Holdout PR | domain={dom}")
            plot_score_hist_png(dom_dir / "score_hist.png", y_true, scores, thr=thr,
                                title=f"AE Holdout Score Dist | domain={dom} | thr={thr:.4g}")

        # strip arrays before json
        m_stripped = {k: v for k, v in m.items() if k not in ("scores", "y_true", "y_pred")}
        m_stripped.update({"domain": dom, "best_val_loss_inner": float(best_val)})
        res["holdout"][dom] = m_stripped

    # ---- CV: per domain, per fold ----
    for dom in eval_domains:
        fold_rows = []
        for fold_id, (fold_train_subs, fold_val_subs) in enumerate(
            tqdm(cv_folds, desc=f"AE CV folds (eval={dom})", leave=False), start=1
        ):
            fold_train_recs = records_from_subjects(sub2recs, fold_train_subs)
            fold_val_recs = records_from_subjects(sub2recs, fold_val_subs)

            fold_tr_ds = DatasetAE(fold_train_recs, fs, win, hop, motion_thresh, activity_filter="sit", clean_only=True)
            fold_va_sit_clean = DatasetAE(fold_val_recs, fs, win, hop, motion_thresh, activity_filter="sit", clean_only=True)
            fold_va_dom = DatasetAE(fold_val_recs, fs, win, hop, motion_thresh, activity_filter=dom, clean_only=False)

            if len(fold_tr_ds) == 0 or len(fold_va_sit_clean) == 0 or len(fold_va_dom) == 0:
                fold_rows.append({"fold": fold_id, "domain": dom, "error": "empty_dataset"})
                continue

            ae, fold_best = train_detector(
                fold_tr_ds, fold_va_sit_clean,
                epochs=epochs_ae, lr=lr,
                progress_desc=f"AE CV fold{fold_id} epochs"
            )

            fold_thr = fit_detector_threshold_from_inner_val(ae, fold_va_sit_clean, quantile=thr_quantile)
            m = eval_detector_with_arrays(ae, fold_va_dom, thr=fold_thr)

            # fold plots
            fold_plot_dir = outdir / "cv_plots" / dom / f"fold{fold_id:02d}"
            fold_plot_dir.mkdir(parents=True, exist_ok=True)
            y_true = m["y_true"]; y_pred = m["y_pred"]; scores = m["scores"]
            if scores.size > 0:
                plot_confusion_matrix_png(fold_plot_dir / "confusion_matrix.png", y_true, y_pred,
                                          title=f"AE CV Confusion | dom={dom} fold={fold_id}")
                plot_roc_curve_png(fold_plot_dir / "roc.png", y_true, scores,
                                   title=f"AE CV ROC | dom={dom} fold={fold_id}")
                plot_pr_curve_png(fold_plot_dir / "pr.png", y_true, scores,
                                  title=f"AE CV PR | dom={dom} fold={fold_id}")
                plot_score_hist_png(fold_plot_dir / "score_hist.png", y_true, scores, thr=fold_thr,
                                    title=f"AE CV Score Dist | dom={dom} fold={fold_id} | thr={fold_thr:.4g}")

            # strip arrays before storing
            m2 = {k: v for k, v in m.items() if k not in ("scores", "y_true", "y_pred")}
            m2.update({"fold": fold_id, "domain": dom, "best_val_loss": float(fold_best), "threshold": float(fold_thr)})
            fold_rows.append(m2)

        cv_summary = summarize_rows(fold_rows, ["pr_auc", "roc_auc", "f1", "bal_acc"])
        res["cv"][dom] = {"folds": fold_rows, "summary": cv_summary}
        save_csv(outdir / f"ae_cv_{dom}.csv", fold_rows)

        plot_cv_holdout(
            out_png=outdir / f"ae_{dom}_cv_vs_holdout.png",
            title=f"AE Detector (Train sit-clean) | Eval={dom} | CV vs Holdout (NO-LEAK)",
            cv_summary=cv_summary,
            holdout_metrics=res["holdout"].get(dom, {}),
            keys=["pr_auc", "roc_auc", "f1", "bal_acc"],
        )

    (outdir / "ae_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def run_denoise_cv_and_holdout_noleak(
    sub2recs: Dict[str, List[dict]],
    splits: dict,
    fs: float,
    win: float,
    hop: float,
    motion_thresh: float,
    activity: str,
    setup: int,
    epochs_dn: int,
    lr: float,
    outdir: Path,
    lam_proxy: float,
    lam_spec: float,
    lam_hrband: float,
    lam_cons: float,
    lam_ecg_hr: float,
    ecg_tau: float,
) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)

    inner_train_subjects = splits["inner_train_subjects"]
    inner_val_subjects = splits["inner_val_subjects"]
    holdout_subjects = splits["holdout_subjects"]
    cv_folds = splits["cv_folds"]

    inner_train_recs = records_from_subjects(sub2recs, inner_train_subjects)
    inner_val_recs = records_from_subjects(sub2recs, inner_val_subjects)
    holdout_recs = records_from_subjects(sub2recs, holdout_subjects)

    tr_ds = DatasetDenoise(inner_train_recs, fs, win, hop, motion_thresh, activity_filter=activity)
    va_ds = DatasetDenoise(inner_val_recs, fs, win, hop, motion_thresh, activity_filter=activity)
    ho_ds = DatasetDenoise(holdout_recs, fs, win, hop, motion_thresh, activity_filter=activity)

    res = {"splits": splits, "setup": setup, "activity": activity, "holdout": {}, "cv": {}}

    if len(tr_ds) == 0 or len(va_ds) == 0:
        res["holdout"] = {"error": "empty_inner_train_or_val_dataset"}
    else:
        dn_hold, best = train_denoiser(
            tr_ds, va_ds, fs,
            epochs=epochs_dn, lr=lr,
            lam_proxy=lam_proxy, lam_spec=lam_spec, lam_hrband=lam_hrband, lam_cons=lam_cons,
            lam_ecg_hr=lam_ecg_hr, ecg_tau=ecg_tau,
            progress_desc=f"Denoiser holdout-train epochs ({activity}, setup{setup})"
        )
        if len(ho_ds) == 0:
            res["holdout"] = {"error": "empty_holdout_dataset"}
        else:
            m = eval_denoiser(dn_hold, ho_ds, fs, ecg_tau=ecg_tau)
            m.update({"best_val_loss_inner": float(best["val"]), "best_val_ecg_hr": float(best.get("val_ecg_hr", float("nan")))})
            res["holdout"] = m

    fold_rows = []
    for fold_id, (fold_train_subs, fold_val_subs) in enumerate(
        tqdm(cv_folds, desc=f"Denoise CV folds ({activity}, setup{setup})", leave=False), start=1
    ):
        fold_train_recs = records_from_subjects(sub2recs, fold_train_subs)
        fold_val_recs = records_from_subjects(sub2recs, fold_val_subs)

        fold_tr_ds = DatasetDenoise(fold_train_recs, fs, win, hop, motion_thresh, activity_filter=activity)
        fold_va_ds = DatasetDenoise(fold_val_recs, fs, win, hop, motion_thresh, activity_filter=activity)

        if len(fold_tr_ds) == 0 or len(fold_va_ds) == 0:
            fold_rows.append({"fold": fold_id, "activity": activity, "setup": setup, "error": "empty_dataset"})
            continue

        dn, best = train_denoiser(
            fold_tr_ds, fold_va_ds, fs,
            epochs=epochs_dn, lr=lr,
            lam_proxy=lam_proxy, lam_spec=lam_spec, lam_hrband=lam_hrband, lam_cons=lam_cons,
            lam_ecg_hr=lam_ecg_hr, ecg_tau=ecg_tau,
            progress_desc=f"Denoiser CV fold{fold_id} epochs ({activity}, setup{setup})"
        )

        m = eval_denoiser(dn, fold_va_ds, fs, ecg_tau=ecg_tau)
        m.update({"fold": fold_id, "activity": activity, "setup": setup, "best_val_loss": float(best["val"]), "best_val_ecg_hr": float(best.get("val_ecg_hr", float("nan")))})
        fold_rows.append(m)

    cv_summary = summarize_rows(fold_rows, ["l1", "snr_improvement_db", "hr_band_loss", "ecg_hr_mae_bpm"])
    res["cv"] = {"folds": fold_rows, "summary": cv_summary}

    save_csv(outdir / f"denoise_cv_setup{setup}_{activity}.csv", fold_rows)

    plot_cv_holdout(
        out_png=outdir / f"denoise_setup{setup}_{activity}_cv_vs_holdout.png",
        title=f"Denoiser | activity={activity} | setup{setup} | CV vs Holdout (NO-LEAK)",
        cv_summary=cv_summary,
        holdout_metrics=res.get("holdout", {}),
        keys=["l1", "snr_improvement_db", "hr_band_loss", "ecg_hr_mae_bpm"],
    )

    (outdir / f"denoise_results_setup{setup}_{activity}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def run_experiment_v72_noleak(
    data_root: str,
    fs: float,
    win: float,
    hop: float,
    motion_thresh: float,
    n_splits: int,
    holdout_ratio: float,
    inner_val_ratio: float,
    seed: int,
    outdir: str,
    eval_domains: Tuple[str, ...],
    denoise_domains: Tuple[str, ...],
    epochs_ae: int,
    epochs_denoise: int,
    lr: float,
    thr_quantile: float,
    lam_spec: float,
    lam_hrband: float,
    lam_cons: float,
    lam_proxy_setup1: float,
    lam_proxy_setup2: float,
    lam_ecg_hr_setup2: float,
    ecg_tau: float,
) -> dict:
    set_seed(seed)
    t0 = time.time()

    outroot = Path(outdir)
    outroot.mkdir(parents=True, exist_ok=True)

    root = Path(data_root)
    sub2recs = load_physionet_csv(root, fs=float(fs))
    all_subjects = sorted(list(sub2recs.keys()))

    splits = make_subject_splits(
        all_subjects=all_subjects,
        holdout_ratio=holdout_ratio,
        inner_val_ratio=inner_val_ratio,
        seed=seed,
        n_splits=n_splits
    )
    (outroot / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")

    stage_bar = tqdm(total=1 + 2*len(denoise_domains), desc="Overall stages", position=0)

    ae_dir = outroot / "ae"
    ae_res = run_ae_cv_and_holdout_noleak(
        sub2recs=sub2recs,
        splits=splits,
        fs=fs, win=win, hop=hop,
        motion_thresh=motion_thresh,
        epochs_ae=epochs_ae,
        lr=lr,
        outdir=ae_dir,
        eval_domains=eval_domains,
        thr_quantile=thr_quantile,
    )
    stage_bar.update(1)

    den_dir = outroot / "denoise"
    den_all = {"setup1": {}, "setup2": {}}

    for act in denoise_domains:
        den_all["setup1"][act] = run_denoise_cv_and_holdout_noleak(
            sub2recs=sub2recs,
            splits=splits,
            fs=fs, win=win, hop=hop,
            motion_thresh=motion_thresh,
            activity=act, setup=1,
            epochs_dn=epochs_denoise,
            lr=lr,
            outdir=den_dir / "setup1",
            lam_proxy=lam_proxy_setup1,
            lam_spec=lam_spec,
            lam_hrband=lam_hrband,
            lam_cons=lam_cons,
            lam_ecg_hr=0.0,
            ecg_tau=ecg_tau,
        )
        stage_bar.update(1)

    for act in denoise_domains:
        den_all["setup2"][act] = run_denoise_cv_and_holdout_noleak(
            sub2recs=sub2recs,
            splits=splits,
            fs=fs, win=win, hop=hop,
            motion_thresh=motion_thresh,
            activity=act, setup=2,
            epochs_dn=epochs_denoise,
            lr=lr,
            outdir=den_dir / "setup2",
            lam_proxy=lam_proxy_setup2,
            lam_spec=lam_spec,
            lam_hrband=lam_hrband,
            lam_cons=lam_cons,
            lam_ecg_hr=lam_ecg_hr_setup2,
            ecg_tau=ecg_tau,
        )
        stage_bar.update(1)

    stage_bar.close()

    rows = []
    for setup_name in ["setup1", "setup2"]:
        for act in denoise_domains:
            cvsum = den_all[setup_name][act]["cv"]["summary"]
            ho = den_all[setup_name][act].get("holdout", {})
            rows.append({
                "setup": setup_name,
                "activity": act,
                "cv_l1_mean": cvsum.get("l1_mean"),
                "cv_l1_std": cvsum.get("l1_std"),
                "cv_snr_mean": cvsum.get("snr_improvement_db_mean"),
                "cv_snr_std": cvsum.get("snr_improvement_db_std"),
                "cv_ecg_hr_mae_mean": cvsum.get("ecg_hr_mae_bpm_mean"),
                "cv_ecg_hr_mae_std": cvsum.get("ecg_hr_mae_bpm_std"),
                "holdout_l1": ho.get("l1"),
                "holdout_snr": ho.get("snr_improvement_db"),
                "holdout_ecg_hr_mae": ho.get("ecg_hr_mae_bpm"),
            })
    pd.DataFrame(rows).to_csv(outroot / "summary_compare.csv", index=False)

    out = {"splits": splits, "ae": ae_res, "denoise": den_all}
    (outroot / "compare_all.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[v7.2 NO-LEAK + VIZ] Done in {(time.time()-t0)/60:.1f} min. Results: {outroot}")
    return out


# ===================== CLI =====================
def parse_cli():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--fs", type=float, default=500.0)
    ap.add_argument("--win", type=float, default=6.0)
    ap.add_argument("--hop", type=float, default=1.0)

    ap.add_argument("--epochs_ae", type=int, default=20)
    ap.add_argument("--epochs_denoise", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--motion_thresh", type=float, default=0.8)

    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--holdout_ratio", type=float, default=0.2)
    ap.add_argument("--inner_val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--thr_quantile", type=float, default=0.95)

    ap.add_argument("--lam_spec", type=float, default=0.5)
    ap.add_argument("--lam_hrband", type=float, default=0.1)
    ap.add_argument("--lam_cons", type=float, default=0.2)

    ap.add_argument("--lam_proxy_setup1", type=float, default=1.0)
    ap.add_argument("--lam_proxy_setup2", type=float, default=0.3)
    ap.add_argument("--lam_ecg_hr_setup2", type=float, default=1.0)
    ap.add_argument("--ecg_tau", type=float, default=0.1)

    ap.add_argument("--outdir", type=str, default="results_v72_noleak_viz")

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_cli()

    run_experiment_v72_noleak(
        data_root=args.data_root,
        fs=args.fs,
        win=args.win,
        hop=args.hop,
        motion_thresh=args.motion_thresh,
        n_splits=args.n_splits,
        holdout_ratio=args.holdout_ratio,
        inner_val_ratio=args.inner_val_ratio,
        seed=args.seed,
        outdir=args.outdir,
        eval_domains=("sit", "walk", "run"),
        denoise_domains=("walk", "run"),
        epochs_ae=args.epochs_ae,
        epochs_denoise=args.epochs_denoise,
        lr=args.lr,
        thr_quantile=args.thr_quantile,
        lam_spec=args.lam_spec,
        lam_hrband=args.lam_hrband,
        lam_cons=args.lam_cons,
        lam_proxy_setup1=args.lam_proxy_setup1,
        lam_proxy_setup2=args.lam_proxy_setup2,
        lam_ecg_hr_setup2=args.lam_ecg_hr_setup2,
        ecg_tau=args.ecg_tau,
    )
