from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-codex"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal as sp_signal
from sklearn import metrics
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

try:
    from ppg_peak_hr_gating_train import detect_ecg_rpeaks
except Exception as exc:  # pragma: no cover - fail loudly in normal runs.
    raise RuntimeError("asa_classifier.py requires detect_ecg_rpeaks from ppg_peak_hr_gating_train.py") from exc


EPS = 1e-6
ASA_VALUES = (1, 2, 3)
ASA_TO_LABEL = {1: 0, 2: 1, 3: 2}
LABEL_TO_ASA = {v: k for k, v in ASA_TO_LABEL.items()}
CLASS_NAMES = ["ASA1", "ASA2", "ASA3"]
DEVICE_TRACKS = ["SNUADC/PLETH", "SNUADC/ECG_II"]
SHORT_TRACKS = ["PLETH", "ECG_II"]

SPEC_NFFT = 256
SPEC_HOP = 32
SPEC_FMAX = 8.0

RR_FS = 4.0
HRV_DIM = 10
PPG_FEATURE_DIM = 16
INPUT_MODE_ALIASES = {
    "ppg": "ppg",
    "ppg_only": "ppg",
    "ppg+spec": "ppg_spec",
    "ppg_spec": "ppg_spec",
    "spec": "ppg_spec",
    "ppg+rr": "ppg_rr",
    "ppg_rr": "ppg_rr",
    "rr": "ppg_rr",
    "full": "full",
}
INPUT_MODES = ("ppg", "ppg_spec", "ppg_rr", "full")


@dataclass
class AsaConfig:
    target_fs: float = 64.0
    maxlen_sec: float = 1800.0
    win_sec: float = 30.0
    hop_sec: float = 15.0
    base_channels: int = 32
    fusion_dim: int = 128
    batch_size: int = 8
    epochs: int = 12
    patience: int = 8
    lr: float = 1e-3
    cv_folds: int = 5
    seed: int = 42
    num_workers: int = 0
    augment: bool = True
    input_mode: str = "ppg"
    pooling: str = "mean_std_topk"
    top_k_windows: int = 10
    balanced_sampler: bool = True
    loss: str = "focal"
    focal_gamma: float = 2.0
    ordinal_weight: float = 0.35
    logit_adjust_tau: float = 0.5
    tune_thresholds: bool = True


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def parse_input_modes(raw: str) -> List[str]:
    modes: List[str] = []
    for item in str(raw).split(","):
        key = item.strip().lower()
        if not key:
            continue
        mode = INPUT_MODE_ALIASES.get(key)
        if mode is None:
            raise ValueError(f"Unknown input mode '{item}'. Expected one of: {', '.join(INPUT_MODES)}")
        if mode not in modes:
            modes.append(mode)
    return modes or ["full"]


def mode_dir_name(mode: str) -> str:
    return str(mode).replace("+", "_").replace("/", "_")


def timestamped_run_dir(results_root: Path, run_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = "" if run_name in ("", "auto") else f"_{run_name}"
    base = ensure_dir(results_root) / f"{stamp}{suffix}"
    if not base.exists():
        return ensure_dir(base)
    idx = 2
    while True:
        cand = ensure_dir(results_root) / f"{stamp}{suffix}_{idx:02d}"
        if not cand.exists():
            return ensure_dir(cand)
        idx += 1


def interpolate_nan_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    valid = np.isfinite(x)
    if not np.any(valid):
        return np.zeros_like(x, dtype=np.float32)
    if np.all(valid):
        return x.astype(np.float32, copy=True)
    idx = np.arange(x.size)
    return np.interp(idx, idx[valid], x[valid]).astype(np.float32)


def window_starts(n: int, win: int, hop: int) -> List[int]:
    if n < win or win <= 0 or hop <= 0:
        return []
    return list(range(0, n - win + 1, hop))


def match_events(pred_idx: np.ndarray, true_idx: np.ndarray, tolerance_samples: int) -> Tuple[int, int, int, List[float]]:
    pred = list(map(int, np.asarray(pred_idx, dtype=np.int64)))
    true = list(map(int, np.asarray(true_idx, dtype=np.int64)))
    used: set[int] = set()
    tp = 0
    timing_errors: List[float] = []
    for t in true:
        best_j = None
        best_abs = int(tolerance_samples) + 1
        for j, p in enumerate(pred):
            if j in used:
                continue
            err = abs(p - t)
            if err <= tolerance_samples and err < best_abs:
                best_j = j
                best_abs = err
        if best_j is not None:
            used.add(best_j)
            tp += 1
            timing_errors.append(float(pred[best_j] - t))
    fp = len(pred) - tp
    fn = len(true) - tp
    return int(tp), int(fp), int(fn), timing_errors


def run_ptt_ecg_preflight(
    data_root: Path,
    outdir: Path,
    tolerance_sec: float,
    min_f1: float,
) -> Dict[str, Any]:
    csv_dir = data_root / "pulse-transit-time-ppg" / "1.1.0" / "csv"
    tp = fp = fn = 0
    timing_errors: List[float] = []
    per_record: List[Dict[str, Any]] = []
    fs = 500.0
    tol_samples = max(1, int(round(float(tolerance_sec) * fs)))
    for path in sorted(csv_dir.glob("s*_*.csv")):
        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            continue
        if not {"ecg", "peaks"}.issubset(header.columns):
            continue
        df = pd.read_csv(path, usecols=["ecg", "peaks"])
        ecg = pd.to_numeric(df["ecg"], errors="coerce").to_numpy(dtype=np.float32)
        true_idx = np.where(pd.to_numeric(df["peaks"], errors="coerce").fillna(0).to_numpy(dtype=np.float32) > 0.5)[0]
        pred_idx = detect_ecg_rpeaks(ecg, fs=fs)
        r_tp, r_fp, r_fn, r_err = match_events(pred_idx, true_idx, tolerance_samples=tol_samples)
        tp += r_tp
        fp += r_fp
        fn += r_fn
        timing_errors.extend([e / fs for e in r_err])
        prec = r_tp / max(r_tp + r_fp, EPS)
        rec = r_tp / max(r_tp + r_fn, EPS)
        f1 = 2.0 * prec * rec / max(prec + rec, EPS)
        per_record.append(
            {
                "record": path.name,
                "true_peaks": int(len(true_idx)),
                "pred_peaks": int(len(pred_idx)),
                "tp": int(r_tp),
                "fp": int(r_fp),
                "fn": int(r_fn),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
            }
        )
    precision = tp / max(tp + fp, EPS)
    recall = tp / max(tp + fn, EPS)
    f1 = 2.0 * precision * recall / max(precision + recall, EPS)
    err = np.asarray(timing_errors, dtype=np.float32)
    result = {
        "dataset": "pulse_transit_time_ppg",
        "records": int(len(per_record)),
        "tolerance_sec": float(tolerance_sec),
        "min_f1": float(min_f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "timing_mae_sec": float(np.mean(np.abs(err))) if err.size else None,
        "timing_bias_sec": float(np.mean(err)) if err.size else None,
        "timing_std_sec": float(np.std(err)) if err.size else None,
        "passed": bool(f1 >= float(min_f1)),
        "worst_records": sorted(per_record, key=lambda row: row["f1"])[:20],
    }
    save_json(outdir / "ecg_detector_preflight.json", result)
    if not result["passed"]:
        raise RuntimeError(
            f"ECG detector preflight failed: F1={f1:.4f} < {float(min_f1):.4f} "
            f"at +/-{1000.0 * float(tolerance_sec):.1f} ms."
        )
    return result


def load_vitaldb_case_table(max_cases_per_class: int, seed: int) -> pd.DataFrame:
    import vitaldb  # type: ignore

    clinical = vitaldb.load_clinical_data(list(range(1, 6389))).copy()
    caseids = sorted(map(int, vitaldb.find_cases(SHORT_TRACKS)))
    df = clinical[clinical["caseid"].isin(caseids)].copy()
    df = df[df["asa"].isin(list(ASA_VALUES))].copy()
    df["asa"] = df["asa"].astype(int)
    df["label"] = df["asa"].map(ASA_TO_LABEL).astype(int)
    df["subject_group"] = df["subjectid"].fillna(df["caseid"]).astype(str)
    df = df.sort_values(["asa", "subject_group", "caseid"]).reset_index(drop=True)
    if max_cases_per_class > 0:
        rng = np.random.RandomState(int(seed))
        sampled = []
        for asa, block in df.groupby("asa", sort=True):
            idx = np.arange(len(block))
            rng.shuffle(idx)
            sampled.append(block.iloc[idx[: int(max_cases_per_class)]])
        df = pd.concat(sampled, ignore_index=True).sort_values(["asa", "subject_group", "caseid"]).reset_index(drop=True)
    return df


def asa_distribution(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    total_cases = max(1, int(df["caseid"].nunique()))
    total_subjects = max(1, int(df["subject_group"].nunique()))
    for asa, block in df.groupby("asa", dropna=False, sort=True):
        cases = int(block["caseid"].nunique())
        subjects = int(block["subject_group"].nunique())
        rows.append(
            {
                "asa": int(asa) if pd.notna(asa) else None,
                "case_count": cases,
                "subjectid_count": subjects,
                "case_percent": float(round(cases / total_cases * 100.0, 2)),
                "subject_percent": float(round(subjects / total_subjects * 100.0, 2)),
            }
        )
    return rows


def stratified_group_holdout_split(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=int(seed))
    x = np.zeros((len(df), 1), dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)
    groups = df["subject_group"].to_numpy()
    train_idx, test_idx = next(splitter.split(x, y, groups))
    return df.iloc[train_idx].copy().reset_index(drop=True), df.iloc[test_idx].copy().reset_index(drop=True)


def load_or_cache_signal(caseid: int, cfg: AsaConfig, cache_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    import vitaldb  # type: ignore

    ensure_dir(cache_dir)
    cache_path = cache_dir / f"case_{int(caseid)}_fs{int(round(cfg.target_fs))}_max{int(round(cfg.maxlen_sec))}.npz"
    if cache_path.exists():
        try:
            cached = np.load(cache_path)
            return {"ppg": cached["ppg"].astype(np.float32), "ecg": cached["ecg"].astype(np.float32)}
        except Exception:
            pass

    interval = 1.0 / float(cfg.target_fs)
    vals: Optional[np.ndarray] = None
    try:
        vf = vitaldb.VitalFile(int(caseid), DEVICE_TRACKS, maxlen=float(cfg.maxlen_sec), interval=interval)
        vals = np.asarray(vf.to_numpy(DEVICE_TRACKS, interval), dtype=np.float32)
    except Exception:
        try:
            vals = np.asarray(vitaldb.load_case(int(caseid), SHORT_TRACKS, interval), dtype=np.float32)
        except Exception:
            vals = None
    if vals is None or vals.ndim != 2 or vals.shape[1] < 2:
        return None
    n_max = int(round(float(cfg.maxlen_sec) * float(cfg.target_fs)))
    if n_max > 0:
        vals = vals[:n_max]
    if vals.shape[0] < int(round(cfg.win_sec * cfg.target_fs)):
        return None
    ppg_raw = np.asarray(vals[:, 0], dtype=np.float32)
    ecg_raw = np.asarray(vals[:, 1], dtype=np.float32)
    ppg_finite_ratio = float(np.mean(np.isfinite(ppg_raw))) if ppg_raw.size else 0.0
    ecg_finite_ratio = float(np.mean(np.isfinite(ecg_raw))) if ecg_raw.size else 0.0
    if ppg_finite_ratio < 0.50 or ecg_finite_ratio < 0.50:
        return None
    ppg = interpolate_nan_1d(ppg_raw)
    ecg = interpolate_nan_1d(ecg_raw)
    if float(np.nanstd(ppg)) < EPS or float(np.nanstd(ecg)) < EPS:
        return None
    np.savez(cache_path, ppg=ppg.astype(np.float32), ecg=ecg.astype(np.float32))
    return {"ppg": ppg.astype(np.float32), "ecg": ecg.astype(np.float32)}


def load_records(df: pd.DataFrame, cfg: AsaConfig, cache_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in tqdm(df.to_dict("records"), desc="load/cache VitalDB cases", dynamic_ncols=True):
        sig = load_or_cache_signal(int(row["caseid"]), cfg, cache_dir)
        if sig is None:
            continue
        peaks = detect_ecg_rpeaks(sig["ecg"], fs=float(cfg.target_fs))
        records.append(
            {
                "caseid": int(row["caseid"]),
                "subject_group": str(row["subject_group"]),
                "subjectid": int(row["subjectid"]) if pd.notna(row.get("subjectid", np.nan)) else None,
                "asa": int(row["asa"]),
                "label": int(row["label"]),
                "ppg": sig["ppg"],
                "ecg": sig["ecg"],
                "ecg_peaks": peaks.astype(np.int32),
            }
        )
    return records


def record_table(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "caseid": rec["caseid"],
                "subject_group": rec["subject_group"],
                "subjectid": rec["subjectid"],
                "asa": rec["asa"],
                "label": rec["label"],
                "n_samples": int(len(rec["ppg"])),
                "n_ecg_peaks": int(len(rec["ecg_peaks"])),
            }
            for rec in records
        ]
    )


def split_records_by_df(records: Sequence[Dict[str, Any]], df: pd.DataFrame) -> List[Dict[str, Any]]:
    wanted = set(map(int, df["caseid"].tolist()))
    return [rec for rec in records if int(rec["caseid"]) in wanted]


def compute_log_spectrogram(x: np.ndarray, fs: float) -> np.ndarray:
    nperseg = SPEC_NFFT if x.size >= SPEC_NFFT else x.size
    noverlap = max(0, nperseg - SPEC_HOP)
    f, _, Sxx = sp_signal.spectrogram(
        x.astype(np.float32),
        fs=float(fs),
        nperseg=int(nperseg),
        noverlap=int(noverlap),
        window="hann",
        mode="magnitude",
    )
    keep = f <= SPEC_FMAX + 1e-9
    Sxx = Sxx[keep]
    return np.log(Sxx.astype(np.float32) + 1e-6)


def preprocess_ppg_window(x: np.ndarray, fs: float) -> np.ndarray:
    x = interpolate_nan_1d(np.asarray(x, dtype=np.float32))
    if x.size < 4 or float(np.std(x)) < EPS:
        return x.astype(np.float32, copy=True)
    y = sp_signal.detrend(x, type="linear").astype(np.float32)
    nyq = 0.5 * float(fs)
    lo = 0.4 / nyq
    hi = min(8.0 / nyq, 0.99)
    if 0.0 < lo < hi < 1.0 and y.size > int(round(fs * 3.0)):
        try:
            sos = sp_signal.butter(3, [lo, hi], btype="bandpass", output="sos")
            y = sp_signal.sosfiltfilt(sos, y).astype(np.float32)
        except Exception:
            y = y.astype(np.float32, copy=True)
    return y.astype(np.float32, copy=False)


def compute_ppg_features(x: np.ndarray, fs: float) -> np.ndarray:
    x = interpolate_nan_1d(np.asarray(x, dtype=np.float32))
    if x.size == 0:
        return np.zeros(PPG_FEATURE_DIM, dtype=np.float32)
    std = float(np.std(x))
    mean = float(np.mean(x))
    if std < EPS:
        return np.array([mean, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    centered = x - mean
    q25, q75 = np.percentile(x, [25, 75])
    rms = float(np.sqrt(np.mean(centered * centered)))
    slope = np.diff(x)
    zcr = float(np.mean(np.signbit(centered[1:]) != np.signbit(centered[:-1]))) if x.size > 1 else 0.0
    skew = float(np.mean((centered / max(std, EPS)) ** 3))
    kurt = float(np.mean((centered / max(std, EPS)) ** 4))

    duration_sec = max(float(x.size) / float(fs), EPS)
    peaks, props = sp_signal.find_peaks(
        x,
        distance=max(1, int(round(0.35 * float(fs)))),
        prominence=max(0.05 * std, EPS),
    )
    peak_rate = float(len(peaks) / duration_sec * 60.0)
    ibi_mean = ibi_std = ibi_cv = 0.0
    if len(peaks) >= 2:
        ibi = np.diff(peaks).astype(np.float32) / float(fs)
        ibi_mean = float(np.mean(ibi))
        ibi_std = float(np.std(ibi))
        ibi_cv = float(ibi_std / max(ibi_mean, EPS))

    amps: List[float] = []
    rise_times: List[float] = []
    for j, pk in enumerate(peaks):
        left = int(peaks[j - 1]) if j > 0 else max(0, int(pk) - int(round(1.5 * float(fs))))
        segment = x[left : int(pk) + 1]
        if segment.size == 0:
            continue
        trough_rel = int(np.argmin(segment))
        trough_idx = left + trough_rel
        amps.append(float(x[int(pk)] - x[trough_idx]))
        rise_times.append(float(max(0, int(pk) - trough_idx) / float(fs)))
    amp_mean = float(np.mean(amps)) if amps else 0.0
    amp_std = float(np.std(amps)) if amps else 0.0
    rise_median = float(np.median(rise_times)) if rise_times else 0.0
    expected_peak_rate = np.clip(peak_rate / 75.0, 0.0, 1.5)
    sqi = float(np.clip(expected_peak_rate, 0.0, 1.0) / (1.0 + ibi_cv))

    return np.array(
        [
            mean,
            std,
            float(q75 - q25),
            skew,
            kurt,
            rms,
            float(np.std(slope)) if slope.size else 0.0,
            zcr,
            peak_rate,
            ibi_mean,
            ibi_std,
            ibi_cv,
            amp_mean,
            amp_std,
            rise_median,
            sqi,
        ],
        dtype=np.float32,
    )


def expected_spec_shape(cfg: AsaConfig) -> Tuple[int, int]:
    win = int(round(cfg.win_sec * cfg.target_fs))
    dummy = np.zeros(win, dtype=np.float32)
    spec = compute_log_spectrogram(dummy, fs=cfg.target_fs)
    return int(spec.shape[0]), int(spec.shape[1])


def expected_rr_grid_len(cfg: AsaConfig) -> int:
    return int(round(cfg.win_sec * RR_FS))


def resample_rr_series(peak_indices: np.ndarray, win_start: int, win_len_samples: int, fs_signal: float, target_n: int) -> np.ndarray:
    out = np.zeros(int(target_n), dtype=np.float32)
    if peak_indices.size < 2:
        return out
    rr = np.diff(peak_indices) / float(fs_signal)
    rr_t = (peak_indices[1:] - int(win_start)) / float(fs_signal)
    valid = (rr_t >= 0) & (rr_t <= win_len_samples / float(fs_signal))
    rr = rr[valid]
    rr_t = rr_t[valid]
    if rr.size == 0:
        return out
    grid = np.linspace(0, win_len_samples / float(fs_signal), int(target_n), endpoint=False, dtype=np.float32)
    return np.interp(grid, rr_t, rr).astype(np.float32)


def compute_hrv_features(peak_indices: np.ndarray, fs_signal: float) -> np.ndarray:
    if peak_indices.size < 2:
        return np.zeros(HRV_DIM, dtype=np.float32)
    rr = np.diff(peak_indices) / float(fs_signal)
    if rr.size < 2:
        m = float(np.mean(rr))
        return np.array([m, 0.0, 0.0, 0.0, m, m, m, 0.0, 0.0, 0.0], dtype=np.float32)
    diffs = np.diff(rr)
    q25, q75 = np.percentile(rr, [25, 75])
    lf_power = 0.0
    hf_power = 0.0
    rr_t = peak_indices[1:].astype(np.float32) / float(fs_signal)
    if rr_t.size >= 4 and float(rr_t[-1] - rr_t[0]) > 8.0:
        try:
            grid = np.arange(float(rr_t[0]), float(rr_t[-1]), 1.0 / RR_FS, dtype=np.float32)
            if grid.size >= 8:
                rr_grid = np.interp(grid, rr_t, rr).astype(np.float32)
                rr_grid = rr_grid - float(np.mean(rr_grid))
                f, pxx = sp_signal.welch(rr_grid, fs=RR_FS, nperseg=min(256, rr_grid.size))
                lf = (f >= 0.04) & (f < 0.15)
                hf = (f >= 0.15) & (f < 0.40)
                lf_power = float(np.trapz(pxx[lf], f[lf])) if np.any(lf) else 0.0
                hf_power = float(np.trapz(pxx[hf], f[hf])) if np.any(hf) else 0.0
        except Exception:
            lf_power = 0.0
            hf_power = 0.0
    return np.array(
        [
            float(np.mean(rr)),
            float(np.std(rr)),
            float(np.sqrt(np.mean(diffs ** 2))),
            float(np.mean(np.abs(diffs) > 0.05)),
            float(np.median(rr)),
            float(np.min(rr)),
            float(np.max(rr)),
            float(q75 - q25),
            lf_power,
            hf_power,
        ],
        dtype=np.float32,
    )


class AsaCaseDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        cfg: AsaConfig,
        normalizers: Dict[str, Dict[str, Any]],
        train: bool = False,
    ):
        self.records = list(records)
        self.cfg = cfg
        self.normalizers = normalizers
        self.train = bool(train)
        self.win = int(round(cfg.win_sec * cfg.target_fs))
        self.hop = int(round(cfg.hop_sec * cfg.target_fs))
        self.rr_n = expected_rr_grid_len(cfg)
        self.spec_shape = expected_spec_shape(cfg)
        self.use_spec = cfg.input_mode in {"ppg_spec", "full"}
        self.use_rr = cfg.input_mode in {"ppg_rr", "full"}
        self.case_starts: List[List[int]] = [
            window_starts(len(rec["ppg"]), self.win, self.hop) for rec in self.records
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        starts = self.case_starts[idx]
        if not starts:
            starts = [0]
        ppg_full = np.asarray(rec["ppg"], dtype=np.float32)
        peaks = np.asarray(rec["ecg_peaks"], dtype=np.int64)
        norm = self.normalizers
        ppg_list = []
        spec_list = []
        rr_list = []
        hrv_list = []
        feat_list = []
        for st in starts:
            en = st + self.win
            seg = ppg_full[st:en]
            if seg.size < self.win:
                pad = np.zeros(self.win - seg.size, dtype=np.float32)
                seg = np.concatenate([seg, pad])
            seg = preprocess_ppg_window(seg, fs=self.cfg.target_fs)
            if self.train and self.cfg.augment:
                scale = float(np.random.uniform(0.85, 1.15))
                noise = np.random.normal(0.0, 0.01 * (float(np.std(seg)) + EPS), size=seg.shape).astype(np.float32)
                seg = seg * scale + noise
            seg_norm = (seg - float(norm["ppg_raw"]["mean"])) / max(float(norm["ppg_raw"]["std"]), EPS)
            ppg_list.append(seg_norm.astype(np.float32))

            ppg_feat = compute_ppg_features(seg, fs=self.cfg.target_fs)
            feat_mean = np.asarray(norm["ppg_features"]["mean"], dtype=np.float32)
            feat_std = np.maximum(np.asarray(norm["ppg_features"]["std"], dtype=np.float32), EPS)
            feat_list.append(((ppg_feat - feat_mean) / feat_std).astype(np.float32))

            if self.use_spec:
                spec = compute_log_spectrogram(seg, fs=self.cfg.target_fs)
                if spec.shape != self.spec_shape:
                    fixed = np.full(self.spec_shape, fill_value=float(norm["ppg_spec"]["mean"]), dtype=np.float32)
                    rmin = min(spec.shape[0], self.spec_shape[0])
                    cmin = min(spec.shape[1], self.spec_shape[1])
                    fixed[:rmin, :cmin] = spec[:rmin, :cmin]
                    spec = fixed
                spec_norm = (spec - float(norm["ppg_spec"]["mean"])) / max(float(norm["ppg_spec"]["std"]), EPS)
                spec_list.append(spec_norm.astype(np.float32))
            else:
                spec_list.append(np.zeros(self.spec_shape, dtype=np.float32))

            if self.use_rr:
                local = peaks[(peaks >= st) & (peaks < en)]
                rr_seq = resample_rr_series(local, st, self.win, self.cfg.target_fs, self.rr_n)
                rr_norm = (rr_seq - float(norm["rr_seq"]["mean"])) / max(float(norm["rr_seq"]["std"]), EPS)
                rr_list.append(rr_norm.astype(np.float32))

                hrv = compute_hrv_features(local, self.cfg.target_fs)
                hrv_mean = np.asarray(norm["hrv"]["mean"], dtype=np.float32)
                hrv_std = np.maximum(np.asarray(norm["hrv"]["std"], dtype=np.float32), EPS)
                hrv_norm = ((hrv - hrv_mean) / hrv_std).astype(np.float32)
                hrv_list.append(hrv_norm)
            else:
                rr_list.append(np.zeros(self.rr_n, dtype=np.float32))
                hrv_list.append(np.zeros(HRV_DIM, dtype=np.float32))
        return {
            "ppg_raw": np.stack(ppg_list, axis=0),
            "ppg_spec": np.stack(spec_list, axis=0),
            "rr_seq": np.stack(rr_list, axis=0),
            "hrv": np.stack(hrv_list, axis=0),
            "ppg_features": np.stack(feat_list, axis=0),
            "label": int(rec["label"]),
            "caseid": int(rec["caseid"]),
            "subject_group": str(rec["subject_group"]),
            "n_windows": int(len(starts)),
        }


def collate_cases(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    max_w = max(int(b["n_windows"]) for b in batch)
    B = len(batch)
    t_raw = batch[0]["ppg_raw"].shape[-1]
    f_bins, t_spec = batch[0]["ppg_spec"].shape[-2], batch[0]["ppg_spec"].shape[-1]
    t_rr = batch[0]["rr_seq"].shape[-1]
    d_hrv = batch[0]["hrv"].shape[-1]
    d_feat = batch[0]["ppg_features"].shape[-1]
    ppg_raw = np.zeros((B, max_w, t_raw), dtype=np.float32)
    ppg_spec = np.zeros((B, max_w, f_bins, t_spec), dtype=np.float32)
    rr_seq = np.zeros((B, max_w, t_rr), dtype=np.float32)
    hrv = np.zeros((B, max_w, d_hrv), dtype=np.float32)
    ppg_features = np.zeros((B, max_w, d_feat), dtype=np.float32)
    mask = np.zeros((B, max_w), dtype=bool)
    labels = np.zeros(B, dtype=np.int64)
    caseids: List[int] = []
    subjects: List[str] = []
    for i, b in enumerate(batch):
        w = int(b["n_windows"])
        ppg_raw[i, :w] = b["ppg_raw"]
        ppg_spec[i, :w] = b["ppg_spec"]
        rr_seq[i, :w] = b["rr_seq"]
        hrv[i, :w] = b["hrv"]
        ppg_features[i, :w] = b["ppg_features"]
        mask[i, :w] = True
        labels[i] = int(b["label"])
        caseids.append(int(b["caseid"]))
        subjects.append(str(b["subject_group"]))
    return {
        "ppg_raw": torch.from_numpy(ppg_raw),
        "ppg_spec": torch.from_numpy(ppg_spec),
        "rr_seq": torch.from_numpy(rr_seq),
        "hrv": torch.from_numpy(hrv),
        "ppg_features": torch.from_numpy(ppg_features),
        "mask": torch.from_numpy(mask),
        "label": torch.from_numpy(labels),
        "caseid": caseids,
        "subject_group": subjects,
    }


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k1: int = 7, k2: int = 5):
        super().__init__()
        p1 = k1 // 2
        p2 = k2 // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k1, padding=p1),
            nn.GroupNorm(1, out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=k2, padding=p2),
            nn.GroupNorm(1, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PpgRawBranch(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            ConvBlock1D(1, base),
            nn.AvgPool1d(2),
            ConvBlock1D(base, 2 * base),
            nn.AvgPool1d(2),
            ConvBlock1D(2 * base, 4 * base),
            nn.AvgPool1d(2),
            ConvBlock1D(4 * base, 4 * base),
        )
        self.out_dim = 8 * base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(x)
        avg = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
        mx = F.adaptive_max_pool1d(h, 1).squeeze(-1)
        return torch.cat([avg, mx], dim=1)


class PpgSpecBranch(nn.Module):
    def __init__(self, base: int = 16):
        super().__init__()
        self.enc = nn.Sequential(
            ConvBlock2D(1, base),
            nn.AvgPool2d(2),
            ConvBlock2D(base, 2 * base),
            nn.AvgPool2d(2),
            ConvBlock2D(2 * base, 4 * base),
        )
        self.out_dim = 8 * base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(x)
        avg = F.adaptive_avg_pool2d(h, 1).flatten(1)
        mx = F.adaptive_max_pool2d(h, 1).flatten(1)
        return torch.cat([avg, mx], dim=1)


class RrSeqBranch(nn.Module):
    def __init__(self, base: int = 16):
        super().__init__()
        self.enc = nn.Sequential(
            ConvBlock1D(1, base, k1=5, k2=3),
            nn.AvgPool1d(2),
            ConvBlock1D(base, 2 * base, k1=5, k2=3),
            nn.AvgPool1d(2),
            ConvBlock1D(2 * base, 4 * base, k1=3, k2=3),
        )
        self.out_dim = 8 * base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(x)
        avg = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
        mx = F.adaptive_max_pool1d(h, 1).squeeze(-1)
        return torch.cat([avg, mx], dim=1)


class HrvBranch(nn.Module):
    def __init__(self, in_dim: int = HRV_DIM, hidden: int = 32, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
            nn.GELU(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PpgFeatureBranch(nn.Module):
    def __init__(self, in_dim: int = PPG_FEATURE_DIM, hidden: int = 48, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, out_dim),
            nn.GELU(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiBranchAsaModel(nn.Module):
    def __init__(
        self,
        base_channels: int = 32,
        fusion_dim: int = 128,
        num_classes: int = 3,
        input_mode: str = "ppg",
        pooling: str = "mean_std_topk",
        top_k_windows: int = 10,
    ):
        super().__init__()
        self.input_mode = INPUT_MODE_ALIASES.get(str(input_mode).lower(), str(input_mode).lower())
        if self.input_mode not in INPUT_MODES:
            raise ValueError(f"Unknown input_mode={input_mode}")
        self.pooling = str(pooling).lower()
        self.top_k_windows = max(1, int(top_k_windows))
        self.use_spec = self.input_mode in {"ppg_spec", "full"}
        self.use_rr = self.input_mode in {"ppg_rr", "full"}
        self.ppg_raw = PpgRawBranch(base=base_channels)
        self.ppg_features = PpgFeatureBranch()
        in_dim = self.ppg_raw.out_dim + self.ppg_features.out_dim
        if self.use_spec:
            self.ppg_spec = PpgSpecBranch(base=max(8, base_channels // 2))
            in_dim += self.ppg_spec.out_dim
        else:
            self.ppg_spec = None
        if self.use_rr:
            self.rr_seq = RrSeqBranch(base=max(8, base_channels // 2))
            self.hrv = HrvBranch()
            in_dim += self.rr_seq.out_dim + self.hrv.out_dim
        else:
            self.rr_seq = None
            self.hrv = None
        self.fuse = nn.Sequential(
            nn.Linear(in_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        self.window_head = nn.Linear(fusion_dim, num_classes)
        self.head = nn.Linear(3 * fusion_dim, num_classes)

    def forward(
        self,
        ppg_raw: torch.Tensor,
        ppg_spec: torch.Tensor,
        rr_seq: torch.Tensor,
        hrv: torch.Tensor,
        ppg_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, W, T = ppg_raw.shape
        F_bins, T_spec = ppg_spec.shape[-2], ppg_spec.shape[-1]
        T_rr = rr_seq.shape[-1]
        D_hrv = hrv.shape[-1]
        D_feat = ppg_features.shape[-1]
        flat_raw = ppg_raw.reshape(B * W, 1, T)
        flat_spec = ppg_spec.reshape(B * W, 1, F_bins, T_spec)
        flat_rr = rr_seq.reshape(B * W, 1, T_rr)
        flat_hrv = hrv.reshape(B * W, D_hrv)
        flat_feat = ppg_features.reshape(B * W, D_feat)
        f_raw = self.ppg_raw(flat_raw)
        pieces = [f_raw, self.ppg_features(flat_feat)]
        if self.use_spec and self.ppg_spec is not None:
            pieces.append(self.ppg_spec(flat_spec))
        if self.use_rr and self.rr_seq is not None and self.hrv is not None:
            pieces.extend([self.rr_seq(flat_rr), self.hrv(flat_hrv)])
        feat = torch.cat(pieces, dim=-1)
        feat = self.fuse(feat).view(B, W, -1)
        mask_f = mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        mean_feat = (feat * mask_f).sum(dim=1) / denom
        centered = (feat - mean_feat.unsqueeze(1)) * mask_f
        std_feat = torch.sqrt((centered * centered).sum(dim=1) / denom + EPS)

        window_logits = self.window_head(feat)
        window_prob = F.softmax(window_logits, dim=-1)
        ordinal_weights = torch.arange(window_prob.shape[-1], device=window_prob.device, dtype=window_prob.dtype)
        window_scores = torch.sum(window_prob * ordinal_weights.view(1, 1, -1), dim=-1)
        window_scores = window_scores.masked_fill(~mask, -1e9)
        k = min(self.top_k_windows, W)
        top_idx = torch.topk(window_scores, k=k, dim=1).indices
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, feat.shape[-1])
        top_feat = torch.gather(feat, dim=1, index=gather_idx)
        top_mask = torch.gather(mask, dim=1, index=top_idx).float().unsqueeze(-1)
        top_mean_feat = (top_feat * top_mask).sum(dim=1) / top_mask.sum(dim=1).clamp(min=1.0)

        if self.pooling == "mean_std":
            top_mean_feat = torch.zeros_like(mean_feat)
        case_feat = torch.cat([mean_feat, std_feat, top_mean_feat], dim=-1)
        return self.head(case_feat), window_scores


def fit_normalizers(records: Sequence[Dict[str, Any]], cfg: AsaConfig, seed: int) -> Dict[str, Dict[str, Any]]:
    rng = np.random.RandomState(int(seed))
    win = int(round(cfg.win_sec * cfg.target_fs))
    hop = int(round(cfg.hop_sec * cfg.target_fs))
    rr_n = expected_rr_grid_len(cfg)
    use_spec = cfg.input_mode in {"ppg_spec", "full"}
    use_rr = cfg.input_mode in {"ppg_rr", "full"}

    raw_n = 0
    raw_sum = 0.0
    raw_sumsq = 0.0
    spec_n = 0
    spec_sum = 0.0
    spec_sumsq = 0.0
    rr_n_total = 0
    rr_sum = 0.0
    rr_sumsq = 0.0
    hrv_vals: List[np.ndarray] = []
    ppg_feature_vals: List[np.ndarray] = []

    sample_per_case = 4
    for rec in records:
        ppg = np.asarray(rec["ppg"], dtype=np.float32)
        peaks = np.asarray(rec["ecg_peaks"], dtype=np.int64)
        starts = window_starts(len(ppg), win, hop)
        if not starts:
            continue
        if len(starts) > sample_per_case:
            sel = rng.choice(starts, size=sample_per_case, replace=False)
        else:
            sel = list(starts)
        for st in sel:
            en = st + win
            seg = ppg[st:en]
            if seg.size < win:
                continue
            seg = preprocess_ppg_window(seg, fs=cfg.target_fs)
            raw_sum += float(np.sum(seg))
            raw_sumsq += float(np.sum(seg * seg))
            raw_n += int(seg.size)
            ppg_feature_vals.append(compute_ppg_features(seg, fs=cfg.target_fs))
            if use_spec:
                spec = compute_log_spectrogram(seg, fs=cfg.target_fs)
                spec_sum += float(np.sum(spec))
                spec_sumsq += float(np.sum(spec * spec))
                spec_n += int(spec.size)
            if use_rr:
                local = peaks[(peaks >= st) & (peaks < en)]
                rr_seq = resample_rr_series(local, st, win, cfg.target_fs, rr_n)
                rr_sum += float(np.sum(rr_seq))
                rr_sumsq += float(np.sum(rr_seq * rr_seq))
                rr_n_total += int(rr_seq.size)
                hrv_vals.append(compute_hrv_features(local, cfg.target_fs))

    def _stats(s: float, ss: float, n: int) -> Dict[str, float]:
        if n <= 0:
            return {"mean": 0.0, "std": 1.0}
        m = s / float(n)
        v = max(ss / float(n) - m * m, EPS)
        return {"mean": float(m), "std": float(math.sqrt(v))}

    if hrv_vals:
        hrv_arr = np.stack(hrv_vals, axis=0).astype(np.float32)
        hrv_mean = hrv_arr.mean(axis=0).astype(np.float32)
        hrv_std = np.maximum(hrv_arr.std(axis=0), EPS).astype(np.float32)
    else:
        hrv_mean = np.zeros(HRV_DIM, dtype=np.float32)
        hrv_std = np.ones(HRV_DIM, dtype=np.float32)
    if ppg_feature_vals:
        ppg_feature_arr = np.stack(ppg_feature_vals, axis=0).astype(np.float32)
        ppg_feature_mean = ppg_feature_arr.mean(axis=0).astype(np.float32)
        ppg_feature_std = np.maximum(ppg_feature_arr.std(axis=0), EPS).astype(np.float32)
    else:
        ppg_feature_mean = np.zeros(PPG_FEATURE_DIM, dtype=np.float32)
        ppg_feature_std = np.ones(PPG_FEATURE_DIM, dtype=np.float32)

    return {
        "ppg_raw": _stats(raw_sum, raw_sumsq, raw_n),
        "ppg_spec": _stats(spec_sum, spec_sumsq, spec_n),
        "rr_seq": _stats(rr_sum, rr_sumsq, rr_n_total),
        "hrv": {"mean": hrv_mean.tolist(), "std": hrv_std.tolist()},
        "ppg_features": {"mean": ppg_feature_mean.tolist(), "std": ppg_feature_std.tolist()},
    }


def compute_class_weights(records: Sequence[Dict[str, Any]]) -> torch.Tensor:
    y = np.asarray([int(rec["label"]) for rec in records], dtype=np.int64)
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(3), y=y)
    return torch.as_tensor(weights.astype(np.float32))


def compute_class_priors(records: Sequence[Dict[str, Any]]) -> torch.Tensor:
    y = np.asarray([int(rec["label"]) for rec in records], dtype=np.int64)
    counts = np.bincount(y, minlength=3).astype(np.float32)
    priors = (counts + 1.0) / float(np.sum(counts) + 3.0)
    return torch.as_tensor(priors.astype(np.float32))


def make_balanced_sampler(records: Sequence[Dict[str, Any]]) -> WeightedRandomSampler:
    y = np.asarray([int(rec["label"]) for rec in records], dtype=np.int64)
    counts = np.bincount(y, minlength=3).astype(np.float32)
    inv = 1.0 / np.maximum(counts, 1.0)
    weights = inv[y]
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


class AsaOrdinalLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor,
        class_priors: torch.Tensor,
        loss: str = "focal",
        focal_gamma: float = 2.0,
        ordinal_weight: float = 0.35,
        logit_adjust_tau: float = 0.5,
    ):
        super().__init__()
        self.register_buffer("class_weights", class_weights.float())
        self.register_buffer("log_priors", torch.log(torch.clamp(class_priors.float(), min=EPS)))
        self.loss = str(loss).lower()
        self.focal_gamma = float(focal_gamma)
        self.ordinal_weight = float(ordinal_weight)
        self.logit_adjust_tau = float(logit_adjust_tau)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        adjusted = logits + self.logit_adjust_tau * self.log_priors.unsqueeze(0)
        ce = F.cross_entropy(adjusted, target, weight=self.class_weights, reduction="none")
        if self.loss == "focal":
            pt = torch.exp(-ce).clamp(min=EPS, max=1.0)
            base = ((1.0 - pt) ** self.focal_gamma) * ce
        else:
            base = ce
        prob = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, num_classes=3).float()
        emd = torch.sum((torch.cumsum(prob, dim=1) - torch.cumsum(target_onehot, dim=1)) ** 2, dim=1)
        return torch.mean(base + self.ordinal_weight * emd)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for batch in loader:
        ppg_raw = batch["ppg_raw"].to(device)
        ppg_spec = batch["ppg_spec"].to(device)
        rr_seq = batch["rr_seq"].to(device)
        hrv = batch["hrv"].to(device)
        ppg_features = batch["ppg_features"].to(device)
        mask = batch["mask"].to(device)
        y = batch["label"].to(device)
        logits, _ = model(ppg_raw, ppg_spec, rr_seq, hrv, ppg_features, mask)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        bs = int(y.shape[0])
        total_loss += float(loss.detach().cpu().item()) * bs
        total += bs
        correct += int((torch.argmax(logits, dim=1) == y).sum().detach().cpu().item())
    return {"loss": total_loss / max(1, total), "accuracy": correct / max(1, total)}


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, pred: Optional[np.ndarray] = None) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float32)
    if pred is None:
        pred = np.argmax(prob, axis=1) if prob.size else np.zeros(0, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    out: Dict[str, Any] = {
        "n": int(y_true.size),
        "accuracy": float(metrics.accuracy_score(y_true, pred)) if y_true.size else None,
        "balanced_accuracy": float(metrics.balanced_accuracy_score(y_true, pred)) if y_true.size else None,
        "macro_f1": float(metrics.f1_score(y_true, pred, labels=[0, 1, 2], average="macro", zero_division=0)) if y_true.size else None,
        "weighted_f1": float(metrics.f1_score(y_true, pred, labels=[0, 1, 2], average="weighted", zero_division=0)) if y_true.size else None,
        "mae_asa_grade": float(np.mean(np.abs(pred - y_true))) if y_true.size else None,
        "within_1_accuracy": float(np.mean(np.abs(pred - y_true) <= 1)) if y_true.size else None,
        "quadratic_weighted_kappa": float(metrics.cohen_kappa_score(y_true, pred, labels=[0, 1, 2], weights="quadratic")) if y_true.size else None,
        "confusion_matrix": metrics.confusion_matrix(y_true, pred, labels=[0, 1, 2]).astype(int).tolist() if y_true.size else [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        "classification_report": metrics.classification_report(
            y_true,
            pred,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            zero_division=0,
            output_dict=True,
        )
        if y_true.size
        else {},
    }
    if y_true.size and len(np.unique(y_true)) == 3:
        y_bin = np.eye(3, dtype=np.uint8)[y_true]
        out["roc_auc_ovr_macro"] = float(metrics.roc_auc_score(y_bin, prob, average="macro", multi_class="ovr"))
        out["pr_auc_macro"] = float(np.mean([metrics.average_precision_score(y_bin[:, i], prob[:, i]) for i in range(3)]))
    else:
        out["roc_auc_ovr_macro"] = None
        out["pr_auc_macro"] = None
    return out


def predict_from_ordinal_thresholds(prob: np.ndarray, thresholds: Tuple[float, float]) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float32)
    if prob.size == 0:
        return np.zeros(0, dtype=np.int64)
    score = prob @ np.arange(3, dtype=np.float32)
    t1, t2 = thresholds
    pred = np.zeros(score.shape[0], dtype=np.int64)
    pred[score >= float(t1)] = 1
    pred[score >= float(t2)] = 2
    return pred


def tune_ordinal_thresholds(y_true: np.ndarray, prob: np.ndarray, metric_name: str = "macro_f1") -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float32)
    if y_true.size == 0 or prob.size == 0:
        return {"thresholds": [0.5, 1.5], "score": None, "metric": metric_name}
    best_score = -1.0
    best_thresholds = (0.5, 1.5)
    grid = np.linspace(0.20, 1.80, 33)
    for t1 in grid:
        for t2 in grid:
            if t2 <= t1:
                continue
            pred = predict_from_ordinal_thresholds(prob, (float(t1), float(t2)))
            if metric_name == "balanced_accuracy":
                score = float(metrics.balanced_accuracy_score(y_true, pred))
            else:
                score = float(metrics.f1_score(y_true, pred, labels=[0, 1, 2], average="macro", zero_division=0))
            if score > best_score:
                best_score = score
                best_thresholds = (float(t1), float(t2))
    return {"thresholds": list(best_thresholds), "score": float(best_score), "metric": metric_name}


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total = 0
    probs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    caseids: List[int] = []
    with torch.no_grad():
        for batch in loader:
            ppg_raw = batch["ppg_raw"].to(device)
            ppg_spec = batch["ppg_spec"].to(device)
            rr_seq = batch["rr_seq"].to(device)
            hrv = batch["hrv"].to(device)
            ppg_features = batch["ppg_features"].to(device)
            mask = batch["mask"].to(device)
            y = batch["label"].to(device)
            logits, _ = model(ppg_raw, ppg_spec, rr_seq, hrv, ppg_features, mask)
            loss = criterion(logits, y)
            bs = int(y.shape[0])
            total_loss += float(loss.detach().cpu().item()) * bs
            total += bs
            probs.append(F.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32))
            labels.append(y.detach().cpu().numpy().astype(np.int64))
            caseids.extend([int(c) for c in batch["caseid"]])
    if probs:
        prob_arr = np.concatenate(probs, axis=0)
        label_arr = np.concatenate(labels, axis=0)
    else:
        prob_arr = np.zeros((0, 3), dtype=np.float32)
        label_arr = np.zeros(0, dtype=np.int64)
    return {
        "loss": float(total_loss / max(1, total)),
        "case": classification_metrics(label_arr, prob_arr),
        "caseids": list(caseids),
        "case_true": label_arr.tolist(),
        "case_prob": prob_arr.tolist(),
    }


def save_confusion_plot(cm: Sequence[Sequence[int]], outpath: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=np.asarray(cm), display_labels=CLASS_NAMES)
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def save_roc_pr_plots(y_true: np.ndarray, prob: np.ndarray, outdir: Path, prefix: str) -> None:
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float32)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return
    y_bin = np.eye(3, dtype=np.uint8)[y_true]
    fig, ax = plt.subplots(figsize=(6, 5))
    plotted = False
    for i, name in enumerate(CLASS_NAMES):
        if len(np.unique(y_bin[:, i])) < 2:
            continue
        fpr, tpr, _ = metrics.roc_curve(y_bin[:, i], prob[:, i])
        auc = metrics.auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} AUC={auc:.3f}")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{prefix} ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_roc.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    plotted = False
    for i, name in enumerate(CLASS_NAMES):
        if len(np.unique(y_bin[:, i])) < 2:
            continue
        prec, rec, _ = metrics.precision_recall_curve(y_bin[:, i], prob[:, i])
        ap = metrics.average_precision_score(y_bin[:, i], prob[:, i])
        ax.plot(rec, prec, label=f"{name} AP={ap:.3f}")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{prefix} PR")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_pr.png", dpi=160)
    plt.close(fig)


def save_training_curves(history: Sequence[Dict[str, Any]], outdir: Path, prefix: str) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    for key, ylabel in (
        ("loss", "Loss"),
        ("case_macro_f1", "Case Macro F1"),
        ("case_balanced_accuracy", "Case Balanced Accuracy"),
    ):
        fig, ax = plt.subplots(figsize=(6, 4))
        plotted = False
        for split in ("train", "val"):
            vals = [row.get(f"{split}_{key}") for row in history]
            if any(v is not None for v in vals):
                ax.plot(epochs, vals, label=split)
                plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{prefix} {ylabel}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"{prefix}_{key}.png", dpi=160)
        plt.close(fig)


def train_fold(
    train_records: Sequence[Dict[str, Any]],
    val_records: Sequence[Dict[str, Any]],
    cfg: AsaConfig,
    outdir: Path,
    fold_name: str,
    device: torch.device,
) -> Dict[str, Any]:
    normalizers = fit_normalizers(train_records, cfg, seed=cfg.seed)
    train_ds = AsaCaseDataset(train_records, cfg, normalizers, train=True)
    val_ds = AsaCaseDataset(val_records, cfg, normalizers, train=False)
    sampler = make_balanced_sampler(train_records) if cfg.balanced_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_cases,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_cases,
    )
    model = MultiBranchAsaModel(
        base_channels=cfg.base_channels,
        fusion_dim=cfg.fusion_dim,
        input_mode=cfg.input_mode,
        pooling=cfg.pooling,
        top_k_windows=cfg.top_k_windows,
    ).to(device)
    class_weights = compute_class_weights(train_records).to(device)
    class_priors = compute_class_priors(train_records).to(device)
    criterion = AsaOrdinalLoss(
        class_weights=class_weights,
        class_priors=class_priors,
        loss=cfg.loss,
        focal_gamma=cfg.focal_gamma,
        ordinal_weight=cfg.ordinal_weight,
        logit_adjust_tau=cfg.logit_adjust_tau,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    best = {"score": -1.0, "epoch": 0, "state_dict": None}
    history: List[Dict[str, Any]] = []
    bad = 0
    pbar = tqdm(range(1, cfg.epochs + 1), desc=fold_name, dynamic_ncols=True)
    for epoch in pbar:
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va = evaluate_model(model, val_loader, criterion, device)
        row = {
            "epoch": int(epoch),
            "train_loss": float(tr["loss"]),
            "train_accuracy": float(tr["accuracy"]),
            "val_loss": float(va["loss"]),
            "val_case_macro_f1": va["case"]["macro_f1"],
            "val_case_balanced_accuracy": va["case"]["balanced_accuracy"],
        }
        history.append(row)
        score = float(va["case"]["macro_f1"] or 0.0)
        pbar.set_postfix(val_macro_f1=f"{score:.3f}", val_loss=f"{va['loss']:.4f}")
        if score > best["score"]:
            best = {
                "score": score,
                "epoch": int(epoch),
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])
    val_eval = evaluate_model(model, val_loader, criterion, device)
    threshold_info = tune_ordinal_thresholds(np.asarray(val_eval["case_true"]), np.asarray(val_eval["case_prob"])) if cfg.tune_thresholds else {"thresholds": [0.5, 1.5], "score": None, "metric": "disabled"}
    val_threshold_pred = predict_from_ordinal_thresholds(np.asarray(val_eval["case_prob"]), tuple(threshold_info["thresholds"]))
    val_eval["case_thresholded"] = classification_metrics(np.asarray(val_eval["case_true"]), np.asarray(val_eval["case_prob"]), pred=val_threshold_pred)
    save_training_curves(history, outdir, prefix=fold_name)
    save_confusion_plot(
        val_eval["case"]["confusion_matrix"],
        outdir / f"{fold_name}_case_confusion.png",
        f"{fold_name} val case confusion",
    )
    save_roc_pr_plots(np.asarray(val_eval["case_true"]), np.asarray(val_eval["case_prob"]), outdir, prefix=f"{fold_name}_case")
    ckpt = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "normalizers": normalizers,
        "cfg": asdict(cfg),
        "class_names": CLASS_NAMES,
        "best_epoch": int(best["epoch"]),
        "best_val_macro_f1": float(best["score"]),
        "ordinal_thresholds": threshold_info["thresholds"],
    }
    torch.save(ckpt, outdir / f"{fold_name}_model.pt")
    fit_info = {
        "best_epoch": int(best["epoch"]),
        "best_val_macro_f1": float(best["score"]),
        "history": history,
        "normalizers": normalizers,
        "class_weights": class_weights.detach().cpu().numpy().tolist(),
        "class_priors": class_priors.detach().cpu().numpy().tolist(),
        "ordinal_thresholds": threshold_info,
        "train_cases": int(len(train_records)),
        "val_cases": int(len(val_records)),
        "val_eval": val_eval,
    }
    save_json(outdir / f"{fold_name}_fit_info.json", fit_info)
    return fit_info


def ensemble_predict_test(
    fold_ckpts: Sequence[Dict[str, Any]],
    test_records: Sequence[Dict[str, Any]],
    cfg: AsaConfig,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    aggregated: Dict[int, np.ndarray] = {}
    truths: Dict[int, int] = {}
    counts: Dict[int, int] = {}
    for ckpt in fold_ckpts:
        model = MultiBranchAsaModel(
            base_channels=cfg.base_channels,
            fusion_dim=cfg.fusion_dim,
            input_mode=cfg.input_mode,
            pooling=cfg.pooling,
            top_k_windows=cfg.top_k_windows,
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        ds = AsaCaseDataset(test_records, cfg, ckpt["normalizers"], train=False)
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_cases)
        with torch.no_grad():
            for batch in loader:
                ppg_raw = batch["ppg_raw"].to(device)
                ppg_spec = batch["ppg_spec"].to(device)
                rr_seq = batch["rr_seq"].to(device)
                hrv = batch["hrv"].to(device)
                ppg_features = batch["ppg_features"].to(device)
                mask = batch["mask"].to(device)
                logits, _ = model(ppg_raw, ppg_spec, rr_seq, hrv, ppg_features, mask)
                prob = F.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
                for i, cid in enumerate(batch["caseid"]):
                    cid = int(cid)
                    if cid not in aggregated:
                        aggregated[cid] = np.zeros(3, dtype=np.float32)
                        truths[cid] = int(batch["label"][i].item())
                        counts[cid] = 0
                    aggregated[cid] += prob[i]
                    counts[cid] += 1
    cids = np.asarray(sorted(aggregated.keys()), dtype=np.int64)
    probs = np.stack([aggregated[int(c)] / max(counts[int(c)], 1) for c in cids], axis=0).astype(np.float32)
    truth = np.asarray([truths[int(c)] for c in cids], dtype=np.int64)
    return cids, truth, probs


def write_scorecard(path: Path, payload: Dict[str, Any]) -> None:
    def fmt(v: Any) -> str:
        if v is None:
            return "NA"
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return "NA"
            return f"{v:.4f}"
        return str(v)

    def add_dataset(lines: List[str], dataset: Dict[str, Any]) -> None:
        lines.extend(
            [
                "## Dataset",
                "- source: VitalDB cases with ASA, PLETH, and ECG_II",
                f"- ASA classes kept: {list(ASA_VALUES)}",
                "- ASA classes removed before training: 4, 6, NaN",
                f"- total_cases_after_signal_loading: {dataset['total_cases_after_signal_loading']}",
                f"- train_cases: {dataset['train_cases']}",
                f"- test_cases: {dataset['test_cases']}",
                f"- train_subjects: {dataset['train_subjects']}",
                f"- test_subjects: {dataset['test_subjects']}",
                "",
                "### ASA Distribution",
                "| split | ASA | cases | subjectids | case_percent | subject_percent |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for split_name, dist in dataset["distributions"].items():
            for row in dist:
                lines.append(
                    "| "
                    f"{split_name} | "
                    f"{row['asa']} | "
                    f"{row['case_count']} | "
                    f"{row['subjectid_count']} | "
                    f"{fmt(row['case_percent'])} | "
                    f"{fmt(row['subject_percent'])} |"
                )

    def add_preflight(lines: List[str], preflight: Optional[Dict[str, Any]]) -> None:
        lines.extend(["", "## ECG Peak Detector Preflight"])
        if preflight:
            lines.extend(
                [
                    f"- status: {'passed' if preflight.get('passed') else 'failed'}",
                    f"- tolerance_sec: {fmt(preflight.get('tolerance_sec'))}",
                    f"- min_f1: {fmt(preflight.get('min_f1'))}",
                    f"- precision: {fmt(preflight.get('precision'))}",
                    f"- recall: {fmt(preflight.get('recall'))}",
                    f"- f1: {fmt(preflight.get('f1'))}",
                    f"- timing_mae_sec: {fmt(preflight.get('timing_mae_sec'))}",
                ]
            )
        else:
            lines.append("- status: disabled")

    def add_test_metrics(lines: List[str], title: str, test: Dict[str, Any]) -> None:
        lines.extend(
            [
                "",
                title,
                f"- n: {fmt(test.get('n'))}",
                f"- accuracy: {fmt(test.get('accuracy'))}",
                f"- balanced_accuracy: {fmt(test.get('balanced_accuracy'))}",
                f"- macro_f1: {fmt(test.get('macro_f1'))}",
                f"- weighted_f1: {fmt(test.get('weighted_f1'))}",
                f"- mae_asa_grade: {fmt(test.get('mae_asa_grade'))}",
                f"- within_1_accuracy: {fmt(test.get('within_1_accuracy'))}",
                f"- quadratic_weighted_kappa: {fmt(test.get('quadratic_weighted_kappa'))}",
                f"- roc_auc_ovr_macro: {fmt(test.get('roc_auc_ovr_macro'))}",
                f"- pr_auc_macro: {fmt(test.get('pr_auc_macro'))}",
                f"- confusion_matrix: {test.get('confusion_matrix')}",
                "",
                "| class | precision | recall | f1 | support |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        report = test.get("classification_report", {})
        for name in CLASS_NAMES:
            row = report.get(name, {})
            lines.append(
                f"| {name} | {fmt(row.get('precision'))} | {fmt(row.get('recall'))} | "
                f"{fmt(row.get('f1-score'))} | {fmt(row.get('support'))} |"
            )

    def add_mode(lines: List[str], mode_payload: Dict[str, Any]) -> None:
        mode = mode_payload.get("input_mode", mode_payload.get("config", {}).get("cfg", {}).get("input_mode", "full"))
        cfg_row = mode_payload.get("config", {}).get("cfg", {})
        lines.extend(
            [
                "",
                f"## Model: {mode}",
                f"- signal span: first {fmt(cfg_row.get('maxlen_sec'))} sec",
                f"- window/hop: {fmt(cfg_row.get('win_sec'))} sec / {fmt(cfg_row.get('hop_sec'))} sec",
                f"- pooling: {cfg_row.get('pooling')} top_k_windows={cfg_row.get('top_k_windows')}",
                f"- loss: {cfg_row.get('loss')} + ordinal_weight={fmt(cfg_row.get('ordinal_weight'))}",
                f"- balanced_sampler: {cfg_row.get('balanced_sampler')}",
                f"- OOF ordinal thresholds: {mode_payload.get('oof_thresholds', {}).get('thresholds', 'NA')}",
                "",
                "### Cross-validation Folds",
                "| fold | train_cases | val_cases | best_epoch | val macro F1 | val bal acc | val QWK |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for fold in mode_payload.get("folds", []):
            val = fold["val_eval"]["case"]
            lines.append(
                "| "
                f"{fold['fold']} | "
                f"{fold['train_cases']} | "
                f"{fold['val_cases']} | "
                f"{fold['best_epoch']} | "
                f"{fmt(val.get('macro_f1'))} | "
                f"{fmt(val.get('balanced_accuracy'))} | "
                f"{fmt(val.get('quadratic_weighted_kappa'))} |"
            )
        cv = mode_payload.get("cv_summary", {})
        lines.extend(
            [
                "",
                f"- CV macro F1 mean/std: {fmt(cv.get('macro_f1_mean'))} / {fmt(cv.get('macro_f1_std'))}",
                f"- CV balanced accuracy mean/std: {fmt(cv.get('balanced_accuracy_mean'))} / {fmt(cv.get('balanced_accuracy_std'))}",
            ]
        )
        if "oof_thresholded" in mode_payload:
            add_test_metrics(lines, "### OOF Thresholded Validation", mode_payload["oof_thresholded"])
        add_test_metrics(lines, "### Test Final Ordinal Thresholded", mode_payload["test_case"])
        if "test_case_argmax" in mode_payload:
            add_test_metrics(lines, "### Test Argmax Diagnostic", mode_payload["test_case_argmax"])

    lines = ["# ASA Classifier Scorecard", ""]
    add_dataset(lines, payload["dataset"])
    add_preflight(lines, payload.get("ecg_preflight"))
    if "models" in payload:
        lines.extend(
            [
                "",
                "## Model Comparison",
                "| input_mode | OOF macro F1 | test macro F1 | test bal acc | test QWK | test MAE | ASA3 recall | ASA3 F1 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for mode, model_payload in payload["models"].items():
            oof = model_payload.get("oof_thresholded", {})
            test = model_payload.get("test_case", {})
            asa3 = test.get("classification_report", {}).get("ASA3", {})
            lines.append(
                f"| {mode} | {fmt(oof.get('macro_f1'))} | {fmt(test.get('macro_f1'))} | "
                f"{fmt(test.get('balanced_accuracy'))} | {fmt(test.get('quadratic_weighted_kappa'))} | "
                f"{fmt(test.get('mae_asa_grade'))} | {fmt(asa3.get('recall'))} | {fmt(asa3.get('f1-score'))} |"
            )
        for model_payload in payload["models"].values():
            add_mode(lines, model_payload)
    else:
        add_mode(lines, payload)
    lines.extend(
        [
            "",
            "## Notes",
            "- 80/20 subject-level holdout via StratifiedGroupKFold (first fold = test).",
            "- 5-fold StratifiedGroupKFold CV on the 80% training set; subjects do not leak across folds.",
            "- Default input now uses the first 30 minutes, split into 30s windows with 15s hop unless overridden.",
            "- RR is derived from ECG R-peak detection; HRV features are computed from RR intervals.",
            "- PPG branch includes detrended/bandpassed raw PPG, log spectrogram, SQI/shape/pulse-interval features depending on input_mode.",
            "- Final predictions use OOF-tuned ordinal thresholds; argmax is reported only as a diagnostic.",
            "- Case aggregation uses mean + std pooling over all windows plus top-k mean pooling over highest-risk windows.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train VitalDB ASA 1/2/3 classifier with configurable PPG/RR/HRV branches.")
    ap.add_argument("--data_root", default="physionet.org/files", type=str)
    ap.add_argument("--results_root", default="test_asa_classifier", type=str)
    ap.add_argument("--run_name", default="auto", type=str)
    ap.add_argument("--target_fs", default=64.0, type=float)
    ap.add_argument("--maxlen_sec", default=1800.0, type=float)
    ap.add_argument("--win_sec", default=30.0, type=float)
    ap.add_argument("--hop_sec", default=15.0, type=float)
    ap.add_argument("--base_channels", default=32, type=int)
    ap.add_argument("--fusion_dim", default=128, type=int)
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--epochs", default=12, type=int)
    ap.add_argument("--patience", default=8, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--cv_folds", default=5, type=int)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--num_workers", default=0, type=int)
    ap.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--input_modes", default="ppg", type=str)
    ap.add_argument("--pooling", default="mean_std_topk", choices=["mean_std_topk", "mean_std"], type=str)
    ap.add_argument("--top_k_windows", default=10, type=int)
    ap.add_argument("--balanced_sampler", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--loss", default="focal", choices=["ce", "focal"], type=str)
    ap.add_argument("--focal_gamma", default=2.0, type=float)
    ap.add_argument("--ordinal_weight", default=0.35, type=float)
    ap.add_argument("--logit_adjust_tau", default=0.5, type=float)
    ap.add_argument("--tune_thresholds", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max_cases_per_class", default=0, type=int)
    ap.add_argument("--cache_dir", default="", type=str)
    ap.add_argument("--device", default="cpu", type=str)
    ap.add_argument("--ecg_preflight", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ecg_preflight_tolerance_sec", default=0.010, type=float)
    ap.add_argument("--ecg_preflight_min_f1", default=0.95, type=float)
    ap.add_argument("--inspect_only", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cfg = AsaConfig(
        target_fs=float(args.target_fs),
        maxlen_sec=float(args.maxlen_sec),
        win_sec=float(args.win_sec),
        hop_sec=float(args.hop_sec),
        base_channels=int(args.base_channels),
        fusion_dim=int(args.fusion_dim),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        patience=int(args.patience),
        lr=float(args.lr),
        cv_folds=int(args.cv_folds),
        seed=int(args.seed),
        num_workers=int(args.num_workers),
        augment=bool(args.augment),
        input_mode="ppg",
        pooling=str(args.pooling),
        top_k_windows=int(args.top_k_windows),
        balanced_sampler=bool(args.balanced_sampler),
        loss=str(args.loss),
        focal_gamma=float(args.focal_gamma),
        ordinal_weight=float(args.ordinal_weight),
        logit_adjust_tau=float(args.logit_adjust_tau),
        tune_thresholds=bool(args.tune_thresholds),
    )
    if cfg.cv_folds != 5:
        print(f"[warning] requested cv_folds={cfg.cv_folds}; project specification asks for 5-fold CV.")
    data_root = Path(args.data_root)
    run_dir = timestamped_run_dir(Path(args.results_root), args.run_name)
    cache_dir = Path(args.cache_dir) if args.cache_dir else ensure_dir(Path(args.results_root) / "_vitaldb_signal_cache")
    save_json(run_dir / "config.json", {"args": vars(args), "cfg": asdict(cfg)})

    preflight_result: Optional[Dict[str, Any]] = None
    if args.ecg_preflight:
        preflight_result = run_ptt_ecg_preflight(
            data_root=data_root,
            outdir=run_dir,
            tolerance_sec=float(args.ecg_preflight_tolerance_sec),
            min_f1=float(args.ecg_preflight_min_f1),
        )

    df = load_vitaldb_case_table(max_cases_per_class=int(args.max_cases_per_class), seed=int(args.seed))
    train_df, test_df = stratified_group_holdout_split(df, seed=int(args.seed))
    dataset_summary = {
        "eligible_before_signal_loading_cases": int(len(df)),
        "eligible_before_signal_loading_subjects": int(df["subject_group"].nunique()),
        "removed_asa_classes": ["4", "6", "NaN"],
        "all_distribution_before_signal_loading": asa_distribution(df),
        "train_distribution_before_signal_loading": asa_distribution(train_df),
        "test_distribution_before_signal_loading": asa_distribution(test_df),
        "multi_asa_subject_count": int(df.groupby("subject_group")["asa"].nunique().gt(1).sum()),
    }
    save_json(run_dir / "dataset_summary_before_signal_loading.json", dataset_summary)
    train_df.to_csv(run_dir / "train_cases_before_signal_loading.csv", index=False)
    test_df.to_csv(run_dir / "test_cases_before_signal_loading.csv", index=False)
    if args.inspect_only:
        print(f"Saved inspection to: {run_dir}")
        return

    train_records = load_records(train_df, cfg, cache_dir=cache_dir)
    test_records = load_records(test_df, cfg, cache_dir=cache_dir)
    all_records = train_records + test_records
    loaded_train_df = record_table(train_records)
    loaded_test_df = record_table(test_records)
    loaded_all_df = record_table(all_records)
    loaded_train_df.to_csv(run_dir / "train_cases_loaded.csv", index=False)
    loaded_test_df.to_csv(run_dir / "test_cases_loaded.csv", index=False)

    loaded_summary = {
        "total_cases_after_signal_loading": int(len(all_records)),
        "train_cases": int(len(train_records)),
        "test_cases": int(len(test_records)),
        "train_subjects": int(loaded_train_df["subject_group"].nunique()) if len(loaded_train_df) else 0,
        "test_subjects": int(loaded_test_df["subject_group"].nunique()) if len(loaded_test_df) else 0,
        "distributions": {
            "all": asa_distribution(loaded_all_df) if len(loaded_all_df) else [],
            "train": asa_distribution(loaded_train_df) if len(loaded_train_df) else [],
            "test": asa_distribution(loaded_test_df) if len(loaded_test_df) else [],
        },
    }
    save_json(run_dir / "dataset_summary_loaded.json", loaded_summary)
    if len(train_records) < cfg.cv_folds * 3 or len(test_records) < 3:
        raise RuntimeError("Not enough loaded records for 5-fold ASA training/test evaluation.")

    device = torch.device(args.device)
    input_modes = parse_input_modes(args.input_modes)
    table = record_table(train_records).sort_values("caseid").reset_index(drop=True)
    x = np.zeros((len(table), 1), dtype=np.float32)
    y = table["label"].to_numpy(dtype=np.int64)
    groups = table["subject_group"].to_numpy()

    models_payload: Dict[str, Any] = {}
    for mode in input_modes:
        mode_cfg = replace(cfg, input_mode=mode)
        mode_dir = ensure_dir(run_dir / mode_dir_name(mode))
        print(f"\n=== Training ASA input_mode={mode} ===")
        splitter = StratifiedGroupKFold(n_splits=mode_cfg.cv_folds, shuffle=True, random_state=mode_cfg.seed)
        fold_summaries: List[Dict[str, Any]] = []
        fold_ckpts: List[Dict[str, Any]] = []
        fold_dir_root = ensure_dir(mode_dir / "folds")
        for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(x, y, groups), start=1):
            fold_dir = ensure_dir(fold_dir_root / f"fold_{fold_idx}")
            tr_df = table.iloc[tr_idx].copy()
            va_df = table.iloc[va_idx].copy()
            tr_records = split_records_by_df(train_records, tr_df)
            va_records = split_records_by_df(train_records, va_df)
            fit_info = train_fold(
                train_records=tr_records,
                val_records=va_records,
                cfg=mode_cfg,
                outdir=fold_dir,
                fold_name=f"fold_{fold_idx}",
                device=device,
            )
            ckpt = torch.load(fold_dir / f"fold_{fold_idx}_model.pt", map_location="cpu", weights_only=False)
            fold_ckpts.append(ckpt)
            fold_summaries.append(
                {
                    "fold": int(fold_idx),
                    "train_cases": int(len(tr_records)),
                    "val_cases": int(len(va_records)),
                    "train_subjects": int(tr_df["subject_group"].nunique()),
                    "val_subjects": int(va_df["subject_group"].nunique()),
                    "best_epoch": int(fit_info["best_epoch"]),
                    "best_val_macro_f1": float(fit_info["best_val_macro_f1"]),
                    "ordinal_thresholds": fit_info.get("ordinal_thresholds"),
                    "val_eval": fit_info["val_eval"],
                }
            )

        cv_macro = [float(f["val_eval"]["case"]["macro_f1"] or 0.0) for f in fold_summaries]
        cv_balacc = [float(f["val_eval"]["case"]["balanced_accuracy"] or 0.0) for f in fold_summaries]
        cv_qwk = [float(f["val_eval"]["case"]["quadratic_weighted_kappa"] or 0.0) for f in fold_summaries]
        cv_summary = {
            "macro_f1_mean": float(np.mean(cv_macro)) if cv_macro else None,
            "macro_f1_std": float(np.std(cv_macro)) if cv_macro else None,
            "balanced_accuracy_mean": float(np.mean(cv_balacc)) if cv_balacc else None,
            "balanced_accuracy_std": float(np.std(cv_balacc)) if cv_balacc else None,
            "quadratic_weighted_kappa_mean": float(np.mean(cv_qwk)) if cv_qwk else None,
            "quadratic_weighted_kappa_std": float(np.std(cv_qwk)) if cv_qwk else None,
        }
        oof_true = np.concatenate([np.asarray(f["val_eval"]["case_true"], dtype=np.int64) for f in fold_summaries], axis=0)
        oof_prob = np.concatenate([np.asarray(f["val_eval"]["case_prob"], dtype=np.float32) for f in fold_summaries], axis=0)
        oof_threshold_info = tune_ordinal_thresholds(oof_true, oof_prob) if mode_cfg.tune_thresholds else {"thresholds": [0.5, 1.5], "score": None, "metric": "disabled"}
        oof_thresholds = list(oof_threshold_info["thresholds"])
        oof_threshold_pred = predict_from_ordinal_thresholds(oof_prob, tuple(oof_thresholds))
        oof_threshold_metrics = classification_metrics(oof_true, oof_prob, pred=oof_threshold_pred)

        cids, truth, probs = ensemble_predict_test(fold_ckpts, test_records, mode_cfg, device)
        test_argmax_metrics = classification_metrics(truth, probs)
        test_threshold_pred = predict_from_ordinal_thresholds(probs, tuple(oof_thresholds))
        test_threshold_metrics = classification_metrics(truth, probs, pred=test_threshold_pred)
        save_confusion_plot(test_argmax_metrics["confusion_matrix"], mode_dir / "test_case_argmax_confusion.png", f"{mode} argmax test confusion")
        save_confusion_plot(test_threshold_metrics["confusion_matrix"], mode_dir / "test_case_confusion.png", f"{mode} thresholded test confusion")
        save_roc_pr_plots(truth, probs, mode_dir, prefix="test_case")
        pred_rows = []
        for cid, yt, pr, pred_thr in zip(cids.tolist(), truth.tolist(), probs.tolist(), test_threshold_pred.tolist()):
            argmax_pred = int(np.argmax(pr))
            pred_rows.append(
                {
                    "caseid": int(cid),
                    "true_label": int(yt),
                    "true_asa": int(LABEL_TO_ASA[int(yt)]),
                    "pred_label": int(pred_thr),
                    "pred_asa": int(LABEL_TO_ASA[int(pred_thr)]),
                    "argmax_pred_label": argmax_pred,
                    "argmax_pred_asa": int(LABEL_TO_ASA[argmax_pred]),
                    "prob_ASA1": float(pr[0]),
                    "prob_ASA2": float(pr[1]),
                    "prob_ASA3": float(pr[2]),
                    "ordinal_score": float(np.asarray(pr, dtype=np.float32) @ np.arange(3, dtype=np.float32)),
                }
            )
        pd.DataFrame(pred_rows).to_csv(mode_dir / "test_case_predictions.csv", index=False)

        mode_payload = {
            "input_mode": mode,
            "config": {"args": vars(args), "cfg": asdict(mode_cfg)},
            "dataset": loaded_summary,
            "ecg_preflight": preflight_result,
            "folds": fold_summaries,
            "cv_summary": cv_summary,
            "oof_thresholds": oof_threshold_info,
            "oof_thresholded": oof_threshold_metrics,
            "test_case": test_threshold_metrics,
            "test_case_argmax": test_argmax_metrics,
            "sklearn_reference": "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html",
        }
        save_json(mode_dir / "summary.json", mode_payload)
        write_scorecard(mode_dir / "scorecard.md", mode_payload)
        models_payload[mode] = mode_payload
        print(
            f"{mode} thresholded test: macro_f1={test_threshold_metrics.get('macro_f1'):.4f} "
            f"balanced_acc={test_threshold_metrics.get('balanced_accuracy'):.4f} "
            f"qwk={test_threshold_metrics.get('quadratic_weighted_kappa'):.4f}"
        )

    payload = {
        "config": {"args": vars(args), "cfg": asdict(cfg), "input_modes": input_modes},
        "dataset": loaded_summary,
        "ecg_preflight": preflight_result,
        "models": models_payload,
        "sklearn_reference": "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html",
    }
    save_json(run_dir / "summary.json", payload)
    write_scorecard(run_dir / "scorecard.md", payload)
    print(f"\nSaved ASA classifier results to: {run_dir}")


if __name__ == "__main__":
    main()
