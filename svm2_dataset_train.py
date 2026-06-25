import webbrowser
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import kurtosis, skew 

import dash
from dash import dcc, html, Input, Output, State, dash_table,no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import (StratifiedKFold, GroupKFold, train_test_split,
                                     GroupShuffleSplit, cross_validate)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,           
    balanced_accuracy_score,    
    precision_recall_fscore_support
)

import joblib

# ========================
# Global config
# ========================
FS_DEFAULT = 400  # Sampling frequency in Hz
MIN_BPM = 40  # Minimum expected heart rate for artifact rejection
MAX_BPM = 180  # Maximum expected heart rate for artifact rejection
DATASET_DIR = Path("datasets")     # Feature CSV snapshots
MODEL_DIR = Path("models")         # Saved models (pkl)
PORT = 8051
G = 9.81
for p in (DATASET_DIR, MODEL_DIR):
    p.mkdir(parents=True, exist_ok=True)
    

envi = 1

windows_address_1 = ["/mnt/d/Tubcloud/Shared/PPG/Test Data",
                   "/mnt/d/Tubcloud/Shared/PPG/Test Data/25July25"]

ubuntu_address_0 = ["/home/trinker/only_view/Test Data", 
                  "/home/trinker/only_view/Test Data/25July25"]


if envi:
    DEFAULT_FOLDER_MAIN = windows_address_1[0]
    DEFAULT_FOLDER = windows_address_1[1]
else:
    DEFAULT_FOLDER_MAIN = ubuntu_address_0[0]
    DEFAULT_FOLDER = ubuntu_address_0[1]
    

folder_options = [
    {"label": Path(p).as_posix(), "value": p}
    for p in [DEFAULT_FOLDER_MAIN, DEFAULT_FOLDER]
    if os.path.exists(p)
]


WORK = Path('.')
DIR_TRAIN_LABELED = WORK/"train_labeled"   # file/path/t0/t1/label (timeline labels)
DIR_TRAIN_RAW     = WORK/"train_raw"       # file-level raw segments (JSON vectors)
DIR_TRAIN_WIN     = WORK/"train_window"    # windowed features
DIR_TRAIN_VAL     = WORK/"train_val"
MODEL_DIR         = WORK/"models"
for p in (DIR_TRAIN_LABELED, DIR_TRAIN_RAW, DIR_TRAIN_WIN, MODEL_DIR, DIR_TRAIN_VAL):
    p.mkdir(parents=True, exist_ok=True)
    
# --------------------------------------
# Try to import your project core funcs
# --------------------------------------
try:
    from funcs import preprocess_ppg_min, imu_preprocess_with_kf
    print("import success")
    HAVE_CORE = True
except Exception:
    HAVE_CORE = False
    print("import fail")
    def preprocess_ppg_min(ppg, fs=FS_DEFAULT, hp_cut=0.2, mains=None):
        """Fallback: high-pass 0.2 Hz + optional mains notch (50/60 Hz)."""
        ppg = np.asarray(ppg, float).ravel()
        b, a = signal.butter(2, hp_cut/(0.5*fs), 'high')
        y = signal.filtfilt(b, a, ppg)
        if mains in (50, 60):
            b, a = signal.iirnotch(w0=float(mains), Q=30.0, fs=fs)
            y = signal.filtfilt(b, a, y)
        return y

    def imu_preprocess_with_kf(df: pd.DataFrame, fs=FS_DEFAULT, acc_fc=20, gyro_fc=40, static_secs=2.0):
        """Fallback: LP accel/gyro → magnitudes + jerk. (No EKF here.)"""
        G = 9.81
        acc = df[['AX','AY','AZ']].to_numpy(float) * G
        gyr = np.deg2rad(df[['GX','GY','GZ']].to_numpy(float))
        def lp(x, fc):
            b, a = signal.butter(4, fc/(0.5*fs), 'low')
            return signal.filtfilt(b, a, x, axis=0)
        acc_f = lp(acc, acc_fc)
        gyr_f = lp(gyr, gyro_fc)
        a_dyn = acc_f
        acc_mag  = np.linalg.norm(a_dyn, axis=1)
        gyro_mag = np.linalg.norm(gyr_f, axis=1)
        jerk     = np.diff(a_dyn, axis=0, prepend=a_dyn[:1]) * fs
        jerk_mag = np.linalg.norm(jerk, axis=1)
        return dict(acc_f=acc_f, gyr_f=gyr_f, a_dyn=a_dyn,
                    AccMag=acc_mag, GyroMag=gyro_mag, JerkMag=jerk_mag)
        
# ========================
# Feature engineering
# ========================



def welch_bandpower(x, fs, fmin, fmax, nperseg=None, noverlap=0.5):
    """Band power via Welch (Hann). Returns a scalar."""
    x = np.asarray(x, float).ravel()
    if nperseg is None:
        nperseg = int(min(len(x), 2*fs))
    nlap = int(noverlap * nperseg)
    f, P = signal.welch(x, fs=fs, window="hann", nperseg=nperseg, noverlap=nlap)
    m = (f >= fmin) & (f <= fmax)
    return float(np.trapz(P[m], f[m])) if np.any(m) else 0.0

def spectral_entropy(x, fs, fmax=10.0):
    """Shannon entropy of normalized Welch PSD up to fmax."""
    x = np.asarray(x, float).ravel()
    nperseg = int(min(len(x), 2*fs)) if len(x) else 256
    f, P = signal.welch(x, fs=fs, window="hann", nperseg=max(64, nperseg))
    m = f <= fmax
    p = P[m] + 1e-18
    p /= np.sum(p)
    return float(-(p * np.log(p)).sum())


def spectral_main_peak(x, fs, fmin=0.3, fmax=8.0):
    """Dominant frequency (Hz) within [fmin,fmax] from Welch PSD."""
    x = np.asarray(x, float).ravel()
    if x.size < 8:
        return 0.0
    nperseg = int(min(len(x), 2*fs))
    f, P = signal.welch(x, fs=fs, window="hann", nperseg=max(64, nperseg))
    band = (f >= fmin) & (f <= fmax)
    return float(f[band][np.argmax(P[band])]) if np.any(band) else 0.0


def compute_time_stats(x, prefix):
    """Basic time-domain stats; names carry a prefix (signal role)."""
    x = np.asarray(x, float).ravel()
    names = [f"{prefix}_{k}" for k in ("mean","std","rms","iqr","kurt","skew")]
    vals = [
            float(np.mean(x)),
            float(np.std(x)),
            float(np.sqrt(np.mean(x**2))),
            float(np.percentile(x, 75) - np.percentile(x, 25)),
            float(kurtosis(x, fisher=False)), # Pearson definition (normal=3)
            float(skew(x))
            ]
    return names, vals

def extract_features_window_noattitude(ppg_seg: np.ndarray, imu_mag: Dict[str,np.ndarray], fs: float) -> Tuple[List[str], np.ndarray]:
    """
    Features per window. Includes PPG + IMU:
    • Time stats: PPG, AccMag, GyroMag, JerkMag
    • Bandpowers: 0.1–0.5 / 0.5–3 / 3–8 Hz for PPG, AccMag, GyroMag
    • Spectral shape: spectral entropy + dominant peak for PPG, AccMag, GyroMag, JerkMag
    """
    # Minimal PPG preprocessing to remove drift & mains
    ppgm = preprocess_ppg_min(ppg_seg, fs=fs, hp_cut=0.2, mains=50)
    acc_mag, gyro_mag, jerk_mag = imu_mag['AccMag'], imu_mag['GyroMag'], imu_mag['JerkMag']


    feat_names, feat_vals = [], []
    # Time-domain stats for each channel
    for sig, pfx in [
    (ppgm, "ppg"), (acc_mag, "accmag"), (gyro_mag, "gyromag"), (jerk_mag, "jerkmag")
    ]:
        n, v = compute_time_stats(sig, pfx)
        feat_names += n; feat_vals += v


    # Bandpowers for selected channels
    bands = [(0.1,0.5),(0.5,3.0),(3.0,8.0)]
    for lo, hi in bands:
        feat_names += [f"ppg_bp_{lo}-{hi}", f"acc_bp_{lo}-{hi}", f"gyro_bp_{lo}-{hi}"]
        feat_vals += [
        welch_bandpower(ppgm, fs, lo, hi),
        welch_bandpower(acc_mag, fs, lo, hi),
        welch_bandpower(gyro_mag, fs, lo, hi)
        ]


    # Spectral-shape features for PPG + IMU (9): entropy & main peak freq
    for sig, pfx in [
    (ppgm, "ppg"), (acc_mag, "accmag"), (gyro_mag, "gyromag"), (jerk_mag, "jerkmag")
    ]:
        feat_names += [f"{pfx}_spec_entropy", f"{pfx}_main_freq"]
        feat_vals += [spectral_entropy(sig, fs, fmax=10.0), spectral_main_peak(sig, fs, fmin=0.1, fmax=8.0)]


    return feat_names, np.array(feat_vals, float)


# Sliding window helper
def make_windows(N: int, fs: float, win_sec: float, hop_sec: float):
    W, H = int(win_sec*fs), int(hop_sec*fs)
    for s in range(0, max(1, N-W+1), H):
        yield s, s+W

# =========================
# Feature extractor (extended)
# =========================
import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew

def _welch_psd(x, fs, fmax=8.0, nperseg=None, noverlap=None, detrend="constant"):
    """Welch PSD bounded to [0, fmax]. No global state."""
    x = np.asarray(x, float).ravel()
    if len(x) < 8:  # guard
        return np.array([0.0]), np.array([0.0])
    if nperseg is None:
        nperseg = max(64, min(1024, int(len(x)//2)))
    if noverlap is None:
        noverlap = nperseg//2
    f, Pxx = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap,
                          detrend=detrend, scaling="density")
    m = (f >= 0) & (f <= float(fmax))
    return f[m], Pxx[m]

def _band_power(f, Pxx, band, relative_to=(0.0, 8.0)):
    """Integrate power in band; optionally return relative power to reference band."""
    if len(f) == 0 or len(Pxx) == 0:
        return np.nan
    f = np.asarray(f); Pxx = np.asarray(Pxx)
    df = np.diff(f); df = np.r_[df, df[-1]]
    lo, hi = band
    m = (f >= lo) & (f <= hi)
    p = float(np.sum(Pxx[m]*df[m]))
    if relative_to is None:
        return p
    rlo, rhi = relative_to
    mr = (f >= rlo) & (f <= rhi)
    pref = float(np.sum(Pxx[mr]*df[mr])) + 1e-12
    return p / pref

def _spec_entropy(Pxx):
    """Shannon entropy normalized to [0,1]."""
    if len(Pxx) == 0:
        return np.nan
    P = np.maximum(np.asarray(Pxx, float), 1e-20)
    p = P / P.sum()
    H = -(p*np.log(p)).sum()
    return float(H / np.log(len(p)))

def _main_freq_in(f, Pxx, frange):
    """Argmax frequency restricted in frange."""
    if len(f) == 0: return np.nan
    lo, hi = frange
    m = (f >= lo) & (f <= hi)
    if not np.any(m): return np.nan
    idx = np.argmax(Pxx[m])
    return float(f[m][idx])

def _time_feats(x, prefix):
    """Window-only time stats; no dataset-level info."""
    x = np.asarray(x, float).ravel()
    if len(x) == 0:
        return {f"{prefix}_mean":np.nan, f"{prefix}_std":np.nan, f"{prefix}_rms":np.nan,
                f"{prefix}_iqr":np.nan, f"{prefix}_skew":np.nan, f"{prefix}_kurtosis":np.nan,
                f"{prefix}_ptp":np.nan}
    q25, q75 = np.percentile(x, [25, 75])
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std":  float(np.std(x, ddof=1) if len(x)>1 else 0.0),
        f"{prefix}_rms":  float(np.sqrt(np.mean(x**2))),
        f"{prefix}_iqr":  float(q75 - q25),
        f"{prefix}_skew": float(skew(x, bias=False)) if len(x) > 2 else 0.0,
        f"{prefix}_kurtosis": float(kurtosis(x, fisher=False, bias=False)) if len(x) > 3 else 3.0,
        f"{prefix}_ptp":  float(np.ptp(x)),
    }

def extract_features_window(
    fs: float,
    # PPG required (already minimally preprocessed for features, e.g., high-pass)
    ppg_win: np.ndarray,
    # IMU references (precomputed per-sample, same window slice)
    a_dyn_xyz: tuple[np.ndarray,np.ndarray,np.ndarray] | None = None,  # dynamic accel X/Y/Z
    accmag_win: np.ndarray | None = None,
    gyromag_win: np.ndarray | None = None,
    jerkmag_win: np.ndarray | None = None,
    # Attitude series over the window (roll/pitch, radians)
    roll_win: np.ndarray | None = None,
    pitch_win: np.ndarray | None = None,
    # Welch settings
    psd_fmax: float = 8.0,
    bp_bands: list[tuple[float,float]] = [(0.1,0.5),(0.5,3.0),(3.0,8.0)],
    nperseg: int | None = None,
    noverlap: int | None = None,
):
    """
    Extended feature set including:
      - PPG: time stats + PSD (0–8 Hz) band powers, spectral entropy, main_freq
      - Magnitudes: AccMag/GyroMag/JerkMag (if provided): time + PSD features
      - Axis-level dynamic acceleration: a_dyn_x/y/z (time + PSD features)
      - Attitude: roll/pitch (time stats + their derivative stats)
    All features are computed WINDOW-LOCAL and depend only on inputs; no globals.
    Returns (names:list[str], values:np.ndarray[float]).
    """
    names, vals = [], []

    # ---- PPG features ----
    tf = _time_feats(ppg_win, "ppg")
    names += tf.keys(); vals += tf.values()
    f, Pxx = _welch_psd(ppg_win, fs=fs, fmax=psd_fmax, nperseg=nperseg, noverlap=noverlap)
    for lo,hi in bp_bands:
        names.append(f"ppg_bp_{lo:.1f}_{hi:.1f}")
        vals.append(_band_power(f, Pxx, (lo,hi), relative_to=(0.0, psd_fmax)))
    names.append("ppg_spec_entropy"); vals.append(_spec_entropy(Pxx))
    names.append("ppg_main_freq");    vals.append(_main_freq_in(f, Pxx, (0.3, 5.0)))  # HR-ish band

    # ---- Magnitude features (if provided) ----
    def _mag_block(x, prefix):
        tfm = _time_feats(x, prefix); 
        nms, vls = list(tfm.keys()), list(tfm.values())
        fi, Pi = _welch_psd(x, fs=fs, fmax=psd_fmax, nperseg=nperseg, noverlap=noverlap)
        for lo,hi in bp_bands:
            nms.append(f"{prefix}_bp_{lo:.1f}_{hi:.1f}")
            vls.append(_band_power(fi, Pi, (lo,hi), relative_to=(0.0, psd_fmax)))
        nms.append(f"{prefix}_spec_entropy"); vls.append(_spec_entropy(Pi))
        nms.append(f"{prefix}_main_freq");    vls.append(_main_freq_in(fi, Pi, (0.0, psd_fmax)))
        return nms, vls

    if accmag_win is not None: 
        n,v = _mag_block(accmag_win, "accmag"); names+=n; vals+=v
    if gyromag_win is not None: 
        n,v = _mag_block(gyromag_win, "gyromag"); names+=n; vals+=v
    if jerkmag_win is not None: 
        n,v = _mag_block(jerkmag_win, "jerkmag"); names+=n; vals+=v

    # ---- Axis-level dynamic accel features (directional) ----
    if a_dyn_xyz is not None:
        ax, ay, az = a_dyn_xyz
        for arr, tag in [(ax,"adynx"), (ay,"adyny"), (az,"adynz")]:
            tfa = _time_feats(arr, tag); names += tfa.keys(); vals += tfa.values()
            fi, Pi = _welch_psd(arr, fs=fs, fmax=psd_fmax, nperseg=nperseg, noverlap=noverlap)
            for lo,hi in bp_bands:
                names.append(f"{tag}_bp_{lo:.1f}_{hi:.1f}")
                vals.append(_band_power(fi, Pi, (lo,hi), relative_to=(0.0, psd_fmax)))
            names.append(f"{tag}_spec_entropy"); vals.append(_spec_entropy(Pi))
            names.append(f"{tag}_main_freq");    vals.append(_main_freq_in(fi, Pi, (0.0, psd_fmax)))

    # ---- Attitude features (roll/pitch; and their rates) ----
    def _att_block(theta, prefix):
        tfatt = _time_feats(theta, prefix)
        nms, vls = list(tfatt.keys()), list(tfatt.values())
        # angular rate inside window (finite diff) – captures posture dynamics
        d = np.diff(theta, prepend=theta[:1]) * fs
        tfd = _time_feats(d, prefix + "_dot")
        nms += tfd.keys(); vls += tfd.values()
        return nms, vls

    if roll_win is not None:
        n,v = _att_block(np.asarray(roll_win, float), "att_roll"); names+=n; vals+=v
    if pitch_win is not None:
        n,v = _att_block(np.asarray(pitch_win, float), "att_pitch"); names+=n; vals+=v

    return list(names), np.asarray(vals, float)

#-------------------------Dash Func-------------------
def get_folder_options():
    """遍历 DEFAULT_FOLDER_MAIN 下的子文件夹，生成 Dropdown 选项；确保包含 DEFAULT_FOLDER。"""
    paths = []
    if os.path.isdir(DEFAULT_FOLDER_MAIN):
        for name in sorted(os.listdir(DEFAULT_FOLDER_MAIN)):
            p = os.path.join(DEFAULT_FOLDER_MAIN, name)
            if os.path.isdir(p):
                paths.append(p)
    # 确保 DEFAULT_FOLDER 在选项里（即使不在 DEFAULT_FOLDER_MAIN 下，也加入）
    if DEFAULT_FOLDER and os.path.exists(DEFAULT_FOLDER) and DEFAULT_FOLDER not in paths:
        paths.insert(0, DEFAULT_FOLDER)
    # label 显示目录名，value 为完整路径
    return [{'label': os.path.basename(p) or p, 'value': p} for p in paths]



# ========================
# Labels & colors
# ========================
LABEL_MAP = {
0: "Rest",
1: "Sit/Stand",
2: "Walk",
3: "Transition",
4: "StrongMotion",
}
LABEL_OPTIONS = [{"label": f"{k} - {v}", "value": k} for k,v in LABEL_MAP.items()]
LABEL_COLORS = {
0: "#2ecc71", # Rest – green
1: "#3498db", # Sit/Stand – blue
2: "#e67e22", # Walk – orange
3: "#f1c40f", # Transition – yellow
4: "#e74c3c", # StrongMotion – red
}

def discrete_colorscale_from_map(map_k2hex: Dict[int,str]):
    ks = sorted(map_k2hex.keys())
    if not ks:
        return "Viridis"
    vmin, vmax = ks[0], ks[-1]
    scale = []
    for k in ks:
        v = 0.0 if vmax==vmin else (k - vmin) / (vmax - vmin)
        scale.append([v, map_k2hex[k]])
        scale.append([min(v+1e-6,1.0), map_k2hex[k]])
    return scale


# ========================
# Dash app layout
# ========================
external_stylesheets: List[str] = []
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
webbrowser.open(f"http://localhost:{PORT}")
app.title = "SVM Dataset Builder + Trainer (PPG+IMU)"
folder_options = get_folder_options()

left_panel_style = {"border":"1px solid #eee","borderRadius":"8px","padding":"10px"}

right_panel_style = {"border":"1px solid #eee","borderRadius":"8px","padding":"10px"}

table_style_table = {"maxHeight":"260px","overflowY":"auto","overflowX":"auto","maxWidth":"100%","minWidth":"100%"}
table_style_cell  = {"minWidth":"110px","width":"140px","maxWidth":"240px","whiteSpace":"normal","textAlign":"left"}

app.layout = html.Div(style={"backgroundColor":"white","padding":"12px"}, children=[
    html.H2("PPG+IMU: label → train_labeled → train_raw → train_window → train"),
    html.Div(style={"display":"grid","gridTemplateColumns":"30% 1fr","gap":"12px"}, children=[

        # ------------------ Left control panel ------------------
        html.Div([
            html.H4("1) File & Columns"),
            dcc.Dropdown(id="input-folder", options=folder_options, value=DEFAULT_FOLDER,
                         clearable=False, placeholder='Select data folder', style={'width':'100%'}),
            dcc.Dropdown(id="ddl-files", options=[], placeholder="Select a CSV file...", style={"marginTop":6}),
            html.Div([
                dcc.Input(id="input-ppg-col", type="text", value="IR", placeholder="PPG column (IR/RED/custom)", style={"width":"48%"}),
                dcc.Input(id="input-fs", type="number", value=FS_DEFAULT, step=1, placeholder="Fs (Hz)", style={"width":"48%","float":"right"})
            ], style={"marginTop":6}),
            html.Div(id="div-head-preview", style={"marginTop":8}),

            html.H4("2) LabelStore (raw timeline) → save to train_labeled"),
            dcc.RangeSlider(id="rs-seg", min=0, max=10, step=0.5, value=[0,10], tooltip={"always_visible":True}),
            html.Div(id="txt-seg", style={"marginTop":4}),
            # —— 新增：Subject ID（不影响现有控件） ——
            html.Div([
                html.Label("Subject ID"),
                dcc.Input(
                    id="input-subject",               # ✅ 新增ID
                    type="text",
                    placeholder="e.g., S001",
                    debounce=True,
                    style={"width": "140px"}
                ),
            ], style={"marginTop": 4}),
            dcc.Dropdown(id="ddl-label", options=LABEL_OPTIONS, value=0, style={"marginTop":6}),
            html.Button("Add Labeled Segment", id="btn-add-label", n_clicks=0,
                        style={"width":"100%","marginTop":6,"backgroundColor":"#2ecc71","color":"white"}),
            html.Div(id="div-labelstore-preview", style={"marginTop":6}),
            html.Button("Save LabelStore → train_labeled CSV", id="btn-save-labeled", n_clicks=0, style={"width":"100%","marginTop":6}),
            html.Div(id="txt-save-labeled", style={"marginTop":6}),

            html.H4("3) Build train_raw from train_labeled (extract raw segments as JSON)"),
            html.Button("Scan train_labeled", id="btn-scan-labeled", n_clicks=0),
            dcc.Dropdown(id="ddl-train-labeled", options=[], placeholder="Select a train_labeled CSV", style={"marginTop":6}),
            html.Button("Materialize Raw Segments → train_raw", id="btn-build-raw", n_clicks=0, style={"width":"100%","marginTop":6}),
            html.Div(id="txt-build-raw", style={"marginTop":6}),

            html.H4("4) Merge train_raw datasets"),
            html.Button("Scan train_raw", id="btn-scan-raw", n_clicks=0),
            dcc.Dropdown(id="ddl-raw-a", options=[], placeholder="Select train_raw A", style={"marginTop":6}),
            dcc.Dropdown(id="ddl-raw-b", options=[], placeholder="Select train_raw B", style={"marginTop":6}),
            html.Button("Merge A + B → new train_raw", id="btn-merge-raw", n_clicks=0, style={"width":"100%","marginTop":6}),
            html.Div(id="txt-merge-raw", style={"marginTop":6}),

            html.H4("5) train_raw → train_window (preproc + window + features)"),
            html.Button("Scan train_raw", id="btn-scan-raw2", n_clicks=0),
            dcc.Dropdown(id="ddl-train-raw", options=[], placeholder="Select a train_raw CSV", style={"marginTop":6}),
            html.Div([
                    html.Label("Save target (train_window / train_val)"),
                    dcc.RadioItems(
                        id="ddl-save-target",                     # new ID
                        options=[{"label": "train_window", "value": "train_window"},
                                {"label": "train_val (external holdout)", "value": "train_val"}],
                        value="train_window",
                        labelStyle={"display": "block"}
                    ),
                    #html.Div(id="txt-trainwin-save-status", style={"marginTop": "4px", "fontSize": 12, "color": "#555"})  
                ], style={"marginTop": "8px"}),
            html.Div([
                    html.Div([
                        html.Div("win_sec — window length in seconds (feature window size)", style={"marginBottom":4}),
                        dcc.Input(id="input-win", type="number", value=3.0, step=0.5,
                                placeholder="win_sec", style={"width":"100%"})
                    ], style={"marginTop":6}),
                    html.Div([
                        html.Div("hop_sec — hop/step in seconds (stride between windows)", style={"marginBottom":4}),
                        dcc.Input(id="input-hop", type="number", value=1.0, step=0.5,
                                placeholder="hop_sec", style={"width":"100%"})
                    ], style={"marginTop":6}),
                    html.Div([
                        html.Div("min_overlap — reserved (unused here), keep default", style={"marginBottom":4}),
                        dcc.Input(id="input-minoverlap", type="number", value=0.7, step=0.05,
                                placeholder="min_overlap (unused)", style={"width":"100%"})
                    ], style={"marginTop":6}),
                ]),
            html.Button("Build Windowed Features → train_window", id="btn-build-win", n_clicks=0, style={"width":"100%","marginTop":6}),
            html.Div(id="txt-build-win", style={"marginTop":6}),

            html.H4("6) Train from train_window"),
            html.Button("Scan train_window", id="btn-scan-win", n_clicks=0),
            dcc.Dropdown(id="ddl-train-win", options=[], placeholder="Select a train_window CSV", style={"marginTop":6}),
            html.Div("Holdout from /train_val (external only)", style={"marginTop":6}),
            html.Button("Refresh /train_val list", id="btn-refresh-trainval", n_clicks=0),   # newid
            dcc.Dropdown(id="ddl-holdout-files", options=[], value=[], multi=True,           # newid
                        placeholder="Select CSV(s) in train_val for external holdout",
                        style={"marginTop":6}),
            html.Div("Group by file (CV/holdout) — prevent leakage across same file", style={"marginTop":6}),
            dcc.Checklist(id="chk-group-file",
                        options=[{"label":"Group by file (CV/holdout)", "value":"group"}],
                        value=["group"], inline=True),

            html.Div("Feature selection — exclude Gyro & Jerk features (default ON)", style={"marginTop":6}),
            dcc.Checklist(id="chk-excl-gj",
                        options=[{"label":"Exclude Gyro & Jerk features (default ON)", "value":"excl"}],
                        value=["excl"], inline=False, style={"marginTop":4}),
            html.Div("Feature selection — Exclude Axis & Attitude features (default OFF)", style={"marginTop":6}),
            dcc.Checklist(id="chk-excl-axis",
                            options=[{"label": "Exclude Axis & Attitude features (X/Y/Z + roll/pitch)", "value": "excl_axis"}],
                            value=[]), 
            html.Div("SVM kernel — RBF for non-linear boundaries; Linear for fast/linear", style={"marginTop":6}),
            dcc.RadioItems(id="ri-kernel",
               options=[{"label":"RBF","value":"rbf"},{"label":"Linear","value":"linear"}],
               value="rbf", inline=True),

            html.Div([
                html.Div("C — regularization strength (higher C = lower regularization)", style={"marginBottom":4}),
                dcc.Input(id="input-C", type="number", value=10.0, step=0.5,
                        placeholder="C", style={"width":"100%"})
            ], style={"marginTop":6}),
            html.Div([
                html.Div("gamma — RBF kernel width (scale/auto/float)", style={"marginBottom":4}),
                dcc.Input(id="input-gamma", type="text", value="scale",
                        placeholder="gamma", style={"width":"100%"})
            ], style={"marginTop":6}),

            html.Div("PCA (dimensionality reduction) — enable to reduce feature dimension", style={"marginTop":6}),
            dcc.Checklist(id="chk-pca",
                        options=[{"label":"Use PCA (var)", "value":"use"}],
                        value=["use"], inline=True),

            html.Div("PCA retained variance — keep this proportion of variance (0.80–0.99)", style={"marginTop":6}),
            dcc.Slider(id="sl-pca-var", min=0.80, max=0.99, step=0.01,
                    value=0.95, marks=None, tooltip={"always_visible":True}),

            html.Div([
                html.Div("CV folds — number of cross-validation folds (grouped by file)", style={"marginBottom":4}),
                dcc.Input(id="input-cv", type="number", value=5, step=1,
                        placeholder="CV folds", style={"width":"100%"})
            ], style={"marginTop":6}),
            html.Div([
                html.Div("Holdout ratio — validation split fraction (e.g., 0.2)", style={"marginBottom":4}),
                dcc.Input(id="input-test", type="number", value=0.2, step=0.05,
                        placeholder="Holdout ratio", style={"width":"100%"})
            ], style={"marginTop":6}),

            html.Button("Train Now", id="btn-train", n_clicks=0,
                        style={"width":"100%","marginTop":6,"backgroundColor":"#34495e","color":"white"}),
            html.Div(id="txt-train-status", style={"marginTop":6,"color":"#2c3e50"}),
        ], style=left_panel_style),

        # ------------------ Right visualization panel ------------------
        html.Div([
            html.H4("Preview: Raw PPG & IMU (selected file / segment)"),
            dcc.Graph(id="fig-raw", style={"height": "520px"}, config={"responsive": False}),

            html.H4("Welch Spectra (PPG, AccMag, GyroMag)"),
            dcc.Graph(id="fig-psd",  style={"height": "320px"}, config={"responsive": False}),

            html.H4("Feature Table (train_window head)"),
            dash_table.DataTable(id="table-feats", page_size=10,
                                 style_table=table_style_table, style_cell=table_style_cell),
            html.H4("Inference Preview on Selected Segment (predicted labels)"),
                dcc.Graph(id="fig-infer-preview", style={"height": "160px"}, config={"responsive": False}),

            html.H4("Training Quick Results"),
            html.Div(id="div-train-metrics"),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"8px"}, children=[
                dcc.Graph(id="fig-cm", style={"height": "360px"}, config={"responsive": False}),
                dcc.Graph(id="fig-f1", style={"height": "360px"}, config={"responsive": False}),
            ]),
        ], style=right_panel_style),
    ]),

    # Stores
    dcc.Store(id="store-folder", data=DEFAULT_FOLDER),
    dcc.Store(id="store-file-path"),
    dcc.Store(id="store-file-meta"),
    dcc.Store(id="store-labels", data=[]),
])


# ========================
# Callbacks: folder → files
# ========================
@app.callback(
    Output("ddl-files", "options"),
    Output("ddl-files", "value"),
    Output("store-folder", "data"),
    Input("input-folder", "value"),
    prevent_initial_call=True
)
def update_file_list(folder):
    if not folder or not os.path.exists(folder):
        return [], None, folder
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.csv')])
    opts = [{"label": Path(f).name, "value": os.path.join(folder, f)} for f in files]
    default_val = os.path.join(folder, files[0]) if files else None
    return opts, default_val, folder

# ========================
# Load & preview head
# ========================
@app.callback(
    Output("store-file-path", "data"),
    Output("div-head-preview", "children"),
    Output("store-file-meta", "data"),
    Output("rs-seg", "max"),
    Output("rs-seg", "value"),
    Input("ddl-files", "value"),
    State("input-fs", "value"),
)
def load_file(file_path, fs):
    if not file_path:
        return None, html.Div(""), None, 10, [0,10]
    try:
        df = pd.read_csv(file_path)
        head = df.head(3)
        table = dash_table.DataTable(
            data=head.to_dict("records"),
            columns=[{"name": c, "id": c} for c in head.columns],
            page_size=3,
            style_table={"overflowX":"auto","maxWidth":"100%"},
            style_cell=table_style_cell
        )
        N = len(df); fs = float(fs or FS_DEFAULT)
        dur = float(N/fs)
        meta = {"N": N, "duration": dur}
        new_max = max(5.0, round(dur, 2))
        return file_path, table, meta, new_max, [0.0, new_max]
    except Exception as e:
        return None, html.Div(f"Failed to load: {e}"), None, 10, [0,10]

@app.callback(Output("txt-seg", "children"), Input("rs-seg", "value"))
def seg_text(value):
    if not value:
        return "No segment selected."
    return f"Segment: {value[0]:.2f} s → {value[1]:.2f} s"

# ========================
# LabelStore add & save to train_labeled
# ========================
@app.callback(
    Output("store-labels", "data"),
    Output("div-labelstore-preview", "children"),
    Input("btn-add-label", "n_clicks"),
    State("ddl-files", "value"),
    State("rs-seg", "value"),
    State("ddl-label", "value"),
    State("store-labels", "data"),
    State("input-subject", "value"),
    prevent_initial_call=True
)
def add_labeled_interval(nc, file_path, seg, label_id, store_labels, subject_id):
    store_labels = store_labels or []
    if not file_path or not seg:
        return store_labels, html.Div("Select file and segment first.")
    t0, t1 = float(seg[0]), float(seg[1])
    if t1 <= t0:
        return store_labels, html.Div("Invalid segment: end ≤ start.")
    entry = dict(
        file=os.path.basename(file_path), 
        path=file_path, 
        t0=t0, 
        t1=t1, 
        label=int(label_id),
        subject=(subject_id or ""))
    store_labels.append(entry)
    df_prev = pd.DataFrame(store_labels[-12:])
    table = dash_table.DataTable(data=df_prev.to_dict("records"),
                                 columns=[{"name":c, "id":c} for c in df_prev.columns],
                                 page_size=12, style_table=table_style_table, style_cell=table_style_cell)
    return store_labels, html.Div([html.Div(f"LabelStore size: {len(store_labels)}"), table])

@app.callback(
    Output("txt-save-labeled", "children"),
    Input("btn-save-labeled", "n_clicks"),
    State("store-labels", "data"),
    prevent_initial_call=True
)
def save_labelstore(nc, labels_data):
    if not labels_data:
        return "LabelStore is empty."
    df = pd.DataFrame(labels_data)
    n_files = df['file'].nunique()
    counts = df['label'].value_counts().to_dict()
    label_part = "_".join([f"L{k}-{counts.get(k,0)}" for k in sorted(LABEL_MAP.keys())])
    tag = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = DIR_TRAIN_LABELED / f"labeled_F{n_files}_{label_part}_{tag}.csv"
    df.to_csv(out, index=False)
    return f"Saved train_labeled: {out.name} | rows={len(df)} | files={n_files}"

# ========================
# train_labeled → train_raw (extract raw segments as JSON rows)
# ========================
@app.callback(
    Output("ddl-train-labeled", "options"),
    Input("btn-scan-labeled", "n_clicks"),
    prevent_initial_call=True
)
def scan_labeled(_):
    files = sorted(DIR_TRAIN_LABELED.glob("*.csv"))
    return [{"label": f.name, "value": str(f)} for f in files]

# --- REPLACE the old build_train_raw() callback with this long-form version ---
@app.callback(
    Output("txt-build-raw", "children"),
    Input("btn-build-raw", "n_clicks"),
    State("ddl-train-labeled", "value"),
    State("input-ppg-col", "value"),
    State("input-fs", "value"),
    prevent_initial_call=True
)
def build_train_raw_long(nc, labeled_csv, ppg_col, fs):
    if not labeled_csv:
        return "Select a train_labeled CSV first."
    fs = float(fs or FS_DEFAULT)
    df_lab = pd.read_csv(labeled_csv)

    out_rows = []   # 我们先收集，再一次性 concat，提高效率
    files = df_lab['path'].unique().tolist()
    seg_counter = 0

    for fpath in files:
        sub = df_lab[df_lab['path'] == fpath].reset_index(drop=True)
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            return f"Failed to read {Path(fpath).name}: {e}"

        # 选择PPG列
        ppg_col_use = ppg_col if ppg_col in df.columns else ("IR" if "IR" in df.columns else df.columns[0])

        for _, r in sub.iterrows():
            s = int(max(0, r.t0 * fs))
            e = int(min(len(df), r.t1 * fs))
            if e - s < int(0.5 * fs):
                continue

            # 切片
            ppg = df[ppg_col_use].to_numpy(float)[s:e]
            has_acc = set(['AX','AY','AZ']).issubset(df.columns)
            has_gyr = set(['GX','GY','GZ']).issubset(df.columns)

            if has_acc:
                AX = df['AX'].to_numpy(float)[s:e]
                AY = df['AY'].to_numpy(float)[s:e]
                AZ = df['AZ'].to_numpy(float)[s:e]
            else:
                AX = AY = AZ = None

            if has_gyr:
                GX = df['GX'].to_numpy(float)[s:e]
                GY = df['GY'].to_numpy(float)[s:e]
                GZ = df['GZ'].to_numpy(float)[s:e]
            else:
                GX = GY = GZ = None

            # 逐样本长表
            N = len(ppg)
            t = np.arange(N, dtype=float) / fs
            base = {
                'file': os.path.basename(fpath),
                'seg_id': seg_counter,
                'label': int(r.label),
                'fs': fs,
                'ppg_col': ppg_col_use,
            }
            df_seg = pd.DataFrame({
                **base,
                't': t,
                'PPG': ppg
            })
            if AX is not None:
                df_seg['AX'] = AX; df_seg['AY'] = AY; df_seg['AZ'] = AZ
            if GX is not None:
                df_seg['GX'] = GX; df_seg['GY'] = GY; df_seg['GZ'] = GZ

            out_rows.append(df_seg)
            seg_counter += 1

    if not out_rows:
        return "No segments produced. Check labels/time ranges."

    out_df = pd.concat(out_rows, ignore_index=True)

    # 文件名包含文件数、各类样本数、时间戳（这里按“行数统计”不太直观，可按 seg_id 再做片段级计数）
    n_files = out_df['file'].nunique()
    counts = out_df.groupby('label')['t'].count().to_dict()  # 样本点数量统计
    label_part = "_".join([f"L{k}-{counts.get(k,0)}" for k in sorted(LABEL_MAP.keys())])
    tag = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = DIR_TRAIN_RAW / f"trainrawLONG_F{n_files}_{label_part}_{tag}.csv"

    # ⚠️ CSV 可能很大，推荐改 parquet：out.with_suffix('.parquet')
    out_df.to_csv(out, index=False)
    return f"Saved long-form train_raw: {out.name} | rows={len(out_df)} | files={n_files} | segments={out_df['seg_id'].nunique()}"


# ========================
# Merge train_raw A+B → new train_raw
# ========================
@app.callback(Output("ddl-raw-a", "options"), Output("ddl-raw-b", "options"), Input("btn-scan-raw", "n_clicks"), prevent_initial_call=True)
def scan_raw(_):
    files = sorted(DIR_TRAIN_RAW.glob("*.csv"))
    opts = [{"label": f.name, "value": str(f)} for f in files]
    return opts, opts

@app.callback(
    Output("txt-merge-raw", "children"),
    Input("btn-merge-raw", "n_clicks"),
    State("ddl-raw-a", "value"),
    State("ddl-raw-b", "value"),
    prevent_initial_call=True
)
def merge_raw(nc, A, B):
    if not A or not B:
        return "Select two train_raw files."

    try:
        dfA, dfB = pd.read_csv(A), pd.read_csv(B)

        # ---------- Detect schema ----------
        is_long_A = {'file','label','seg_id','t','PPG'}.issubset(dfA.columns)
        is_long_B = {'file','label','seg_id','t','PPG'}.issubset(dfB.columns)
        is_long = is_long_A and is_long_B

        if is_long:
            # ===== Long-form merge =====
            # dtype normalization
            for col in ['file','seg_id','label','t','PPG']:
                if col in dfA.columns:
                    if col in ('seg_id','label'):
                        dfA[col] = dfA[col].astype(int)
                    elif col == 't' or col == 'PPG':
                        dfA[col] = dfA[col].astype(float)
                if col in dfB.columns:
                    if col in ('seg_id','label'):
                        dfB[col] = dfB[col].astype(int)
                    elif col == 't' or col == 'PPG':
                        dfB[col] = dfB[col].astype(float)

            # 对齐列
            cols = sorted(set(dfA.columns) | set(dfB.columns))
            dfA = dfA.reindex(columns=cols)
            dfB = dfB.reindex(columns=cols)

            # ---- 关键：避免 seg_id 冲突（对 B 集）----
            # 规则：对于每个 file，B 的 seg_id += (A 中该 file 的 max(seg_id) + 1)
            if 'seg_id' in cols and 'file' in cols:
                maxA = dfA.groupby('file')['seg_id'].max() if not dfA.empty else pd.Series(dtype=float)
                # 给不存在于 A 的 file 也留 0 偏移
                for f in dfB['file'].unique():
                    off = (int(maxA.get(f, -1)) + 1) if not np.isnan(maxA.get(f, np.nan)) else 0
                    m = (dfB['file'] == f)
                    dfB.loc[m, 'seg_id'] = dfB.loc[m, 'seg_id'].astype(int) + off

            # 合并 + 去重
            df = pd.concat([dfA, dfB], ignore_index=True)
            # 依据长表主键去重（file, seg_id, t）——避免重复拼接
            key_cols = [c for c in ['file','seg_id','t'] if c in df.columns]
            if key_cols:
                df = df.drop_duplicates(subset=key_cols)

            # 统计 & 命名
            n_files = df['file'].nunique()
            counts = df['label'].value_counts().to_dict() if 'label' in df.columns else {}
            label_part = "_".join([f"L{k}-{counts.get(k,0)}" for k in sorted(counts.keys())]) if counts else "LNA"
            tag = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            out = DIR_TRAIN_RAW / f"trainrawLONG_MERGED_F{n_files}_{label_part}_{tag}.csv"
            df.to_csv(out, index=False)

            n_rows = len(df)
            n_segs = df['seg_id'].nunique() if 'seg_id' in df.columns else 'NA'
            return f"Merged (long-form) → {out.name} | rows={n_rows} | files={n_files} | segs={n_segs}"

        else:
            # ===== JSON-raw merge（你现有 schema）=====
            cols = sorted(set(dfA.columns) | set(dfB.columns))
            df = pd.concat([dfA.reindex(columns=cols), dfB.reindex(columns=cols)], ignore_index=True)
            # 对 JSON-raw 通常无需去重；若担心重复，可按 (file, label, ppg_json) 去重：
            if {'file','label','ppg_json'}.issubset(df.columns):
                df = df.drop_duplicates(subset=['file','label','ppg_json'])

            n_files = df['file'].nunique() if 'file' in df.columns else 0
            counts = df['label'].value_counts().to_dict() if 'label' in df.columns else {}
            label_part = "_".join([f"L{k}-{counts.get(k,0)}" for k in sorted(counts.keys())]) if counts else "LNA"
            tag = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            out = DIR_TRAIN_RAW / f"trainrawMERGED_F{n_files}_{label_part}_{tag}.csv"
            df.to_csv(out, index=False)
            return f"Merged (json-raw) → {out.name} | rows={len(df)} | files={n_files}"

    except Exception as e:
        return f"Merge failed: {e}"

# ========================
# train_raw → train_window (per file preprocess + window + features)
# ========================
@app.callback(Output("ddl-train-raw", "options"), Input("btn-scan-raw2", "n_clicks"), prevent_initial_call=True)
def scan_raw2(_):
    files = sorted(DIR_TRAIN_RAW.glob("*.csv"))
    return [{"label": f.name, "value": str(f)} for f in files]

# --- REPLACE the old build_train_window() with this version for long-form train_raw ---
@app.callback(
    Output("txt-build-win", "children"),
    Output("table-feats", "columns", allow_duplicate=True),
    Output("table-feats", "data", allow_duplicate=True),
    Input("btn-build-win", "n_clicks"),
    State("ddl-train-raw", "value"),
    State("input-win", "value"),
    State("input-hop", "value"),
    State("ddl-save-target", "value"), 
    prevent_initial_call=True
)
def build_train_window_from_raw_long(nc, raw_csv, win_sec, hop_sec, save_target, noattitude=None):
    if not raw_csv:
        return "Select a train_raw CSV first.", no_update, no_update

    # 读取长表
    dfR = pd.read_csv(raw_csv)
    need_cols = {'file','seg_id','label','fs','t','PPG'}
    if not need_cols.issubset(dfR.columns):
        return "train_raw schema mismatch: require columns file, seg_id, label, fs, t, PPG", no_update, no_update

    all_rows, feat_names = [], None

    # 逐 (file, seg_id) 重建片段 → 预处理IMU → 滑窗 → 特征
    for (file_i, seg_id), grp in dfR.groupby(['file','seg_id']):
        grp = grp.sort_values('t')
        fs = float(grp['fs'].iloc[0])
        label = int(grp['label'].iloc[0])

        ppg = grp['PPG'].to_numpy(float)
        N = len(ppg)

        # IMU 列可能不存在
        AX = grp['AX'].to_numpy(float) #if 'AX' in grp.columns else np.zeros(N)
        AY = grp['AY'].to_numpy(float)#if 'AY' in grp.columns else np.zeros(N)
        AZ = grp['AZ'].to_numpy(float) #if 'AZ' in grp.columns else np.zeros(N)
        GX = grp['GX'].to_numpy(float) #if 'GX' in grp.columns else np.zeros(N)
        GY = grp['GY'].to_numpy(float) #if 'GY' in grp.columns else np.zeros(N)
        GZ = grp['GZ'].to_numpy(float) #if 'GZ' in grp.columns else np.zeros(N)

        df_seg = pd.DataFrame({'AX':AX,'AY':AY,'AZ':AZ,'GX':GX,'GY':GY,'GZ':GZ})
        
        imu_out = imu_preprocess_with_kf(df_seg, fs=fs)
        acc_g = imu_out['acc_raw']
        gyro_d = imu_out['gyro_raw']
        acc = imu_out['acc']
        gyro = imu_out['gyro']
        acc_lp = imu_out['acc_f'] 
        gyro_lp = imu_out['gyr_f'] 
        jerk = imu_out['jerk'] 
        roll=imu_out['roll'] 
        pitch=imu_out['pitch'] 
        bg = imu_out['gyro_bias']
        g_body=imu_out['g_body']
        g_dir = imu_out['g_dir'] 
        a_dyn= imu_out['a_dyn']
        acc_mag = imu_out['AccMag'] 
        gyro_mag = imu_out['GyroMag'] 
        jerk_mag = imu_out['JerkMag']

        for s, e in make_windows(N, fs, float(win_sec), float(hop_sec)):
            if e - s < int(0.5 * fs):
                continue
            ppg_seg = ppg[s:e]
            if noattitude:
                
                imu_seg = {
                    'AccMag': acc_mag[s:e],
                    'GyroMag': gyro_mag[s:e],
                    'JerkMag': jerk_mag[s:e],
                }
                feat_names, feats = extract_features_window_noattitude(ppg_seg, imu_seg, fs)
            else:    
                ppg_win = preprocess_ppg_min(ppg_seg, fs=fs, hp_cut=0.2, mains=50)
                ax_win = a_dyn[s:e, 0]; ay_win = a_dyn[s:e, 1]; az_win = a_dyn[s:e, 2]
                accmag_win = acc_mag[s:e]
                gyromag_win = gyro_mag[s:e] #if 'GyroMag' in locals() else None
                jerkmag_win = jerk_mag[s:e] #if 'JerkMag' in locals() else None
                roll_win = roll[s:e]
                pitch_win = pitch[s:e]
                feat_names, feats = extract_features_window(
                    fs=fs,
                    ppg_win=ppg_win,
                    a_dyn_xyz=(ax_win, ay_win, az_win),
                    accmag_win=accmag_win,
                    gyromag_win=gyromag_win,
                    jerkmag_win=jerkmag_win,
                    roll_win=roll_win, pitch_win=pitch_win,
                    psd_fmax=8.0
                )
            # 仍然保留 window 的绝对时间，方便对齐（使用片段内起止时间）
            t0 = grp['t'].iloc[s]; t1 = grp['t'].iloc[e-1]
            all_rows.append(feats.tolist() + [label, t0, t1, file_i])

    if not all_rows:
        return "No windows generated — adjust win/hop.", no_update, no_update

    cols = feat_names + ["label","t_start","t_end","file"]
    ds_df = pd.DataFrame(all_rows, columns=cols)

    # 保存 train_window
    n_files = ds_df['file'].nunique()
    counts = ds_df['label'].value_counts().to_dict()
    label_part = "_".join([f"L{k}-{counts.get(k,0)}" for k in sorted(LABEL_MAP.keys())])
    tag = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    save_dir = DIR_TRAIN_WIN if save_target != "train_val" else DIR_TRAIN_VAL 
    fname = f"trainwin_F{n_files}_ws{float(win_sec):.2f}s_hs{float(hop_sec):.2f}s_{label_part}_{tag}.csv" 
    out = save_dir / fname  
    ds_df.to_csv(out, index=False)

    head = ds_df.head(12)
    cols_dash = [{"name": c, "id": c} for c in head.columns]
    return f"Saved to {save_dir.name}: {out.name} | rows={len(ds_df)} | files={n_files}", cols_dash, head.to_dict("records")


# ========================
# Preview: raw signals & PSD (unchanged)
# ========================
@app.callback(
    Output("fig-raw", "figure"),
    Output("fig-psd", "figure"),
    Input("ddl-files", "value"),
    Input("rs-seg", "value"),
    State("input-ppg-col", "value"),
    State("input-fs", "value"),
)
def preview_signals(file_path, seg, ppg_col, fs):
    fig_empty = go.Figure().update_layout(height=240, paper_bgcolor="white", plot_bgcolor="white")
    if not file_path or not seg:
        return fig_empty, fig_empty
    fs = float(fs or FS_DEFAULT)
    try:
        df = pd.read_csv(file_path)
        if ppg_col not in df.columns:
            ppg_col = "IR" if "IR" in df.columns else df.columns[0]
        t = np.arange(len(df)) / fs
        m = (t >= seg[0]) & (t <= seg[1])
        ppg = df[ppg_col].to_numpy(float)
        imu = imu_preprocess_with_kf(df, fs=fs)
        acc_mag, gyro_mag = imu['AccMag'], imu['GyroMag']

        fig_t = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                              subplot_titles=(f"PPG ({ppg_col})", "AccMag (m/s²)", "GyroMag (rad/s)"))
        fig_t.add_trace(go.Scatter(x=t[m], y=ppg[m], name="PPG", line=dict(color="purple")), row=1, col=1)
        fig_t.add_trace(go.Scatter(x=t[m], y=acc_mag[m], name="AccMag", line=dict(color="crimson")), row=2, col=1)
        fig_t.add_trace(go.Scatter(x=t[m], y=gyro_mag[m], name="GyroMag", line=dict(color="royalblue")), row=3, col=1)
        fig_t.update_layout(height=520, margin=dict(l=50,r=30,t=60,b=40), paper_bgcolor="white", plot_bgcolor="white")
        for r in (1,2,3):
            fig_t.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", row=r, col=1)
        fig_t.update_xaxes(title_text="Time (s)", row=3, col=1)

        def psd_curve(x, fs):
            x = x[m]
            if len(x) < int(2*fs):
                nperseg = max(128, int(len(x)//2))
            else:
                nperseg = int(2*fs)
            f, P = signal.welch(x, fs=fs, window="hann", nperseg=nperseg, noverlap=int(0.5*nperseg))
            return f, P
        f1,P1 = psd_curve(ppg, fs)
        f2,P2 = psd_curve(acc_mag, fs)
        f3,P3 = psd_curve(gyro_mag, fs)

        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=f1, y=P1, name="PPG", line=dict(color="purple")))
        fig_f.add_trace(go.Scatter(x=f2, y=P2, name="AccMag", line=dict(color="crimson")))
        fig_f.add_trace(go.Scatter(x=f3, y=P3, name="GyroMag", line=dict(color="royalblue")))
        fig_f.update_layout(height=320, margin=dict(l=50,r=30,t=50,b=40), paper_bgcolor="white", plot_bgcolor="white",
                            xaxis_title="Frequency (Hz)", yaxis_title="PSD", xaxis=dict(range=[0,8]))
        fig_f.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
        fig_f.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
        return fig_t, fig_f
    except Exception as e:
        fig = go.Figure().update_layout(title=f"Error: {e}", height=240, paper_bgcolor="white", plot_bgcolor="white")
        return fig, fig

# ========================
# Training from train_window
# ========================

def build_Xy_from_df(df: pd.DataFrame):
    meta_cols = {"label","t_start","t_end","file"}
    feat_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feat_cols].to_numpy(float)
    y = df["label"].to_numpy(int)
    groups = df["file"].to_numpy(object) if "file" in df.columns else None
    return X, y, groups, feat_cols

def select_feat_cols(feat_cols: List[str], exclude_gj: bool, exclude_axis: bool) -> List[str]:
    keep = feat_cols[:]
    print("GY:", exclude_gj)
    if exclude_gj:
        drop_prefixes = ("gyromag_", "gyro_bp_", "gyromag_spec_entropy", "gyromag_main_freq",
                        "jerkmag_", "jerk_bp_", "jerkmag_spec_entropy", "jerkmag_main_freq")
        keep = [c for c in feat_cols if not c.startswith(drop_prefixes)]
    if exclude_axis:
        drop_axis_prefix = ("adynx_", "adyny_", "adynz_", "att_roll_", "att_pitch_")
        keep = [c for c in keep if not c.startswith(drop_axis_prefix)]
    return keep

def train_pipeline_quick(df_win: pd.DataFrame, kernel: str, C_val: float, gamma_val: str|float,
                         use_pca: bool, pca_var: float, cv_folds: int, test_ratio: float,
                         group_by_file: bool, exclude_gj: bool, exclude_axis: bool, random_state: int = 42):
    """Leak-safe training:
       - Prefer grouped CV/holdout by file; fallback to seg_uid; else time-ordered split with a gap.
       - Feature selection switch: exclude Gyro/Jerk if requested.
    """
    # ------- features & labels -------
    X_all, y, groups_file, feat_cols_all = build_Xy_from_df(df_win)
    feat_cols_used = select_feat_cols(feat_cols_all, exclude_gj=exclude_gj, exclude_axis=exclude_axis)
    X = df_win[feat_cols_used].to_numpy(float)
    print("Selecteed Features for Training",len(feat_cols_used))
    y = df_win["label"].to_numpy(int)

    # ------- choose grouping key -------
    groups = None
    group_name = None
    if group_by_file and "file" in df_win.columns and df_win["file"].nunique() >= 2:
        groups = df_win["file"].astype(str).to_numpy()
        group_name = "file"
        print("data grouped")
    elif "seg_uid" in df_win.columns and df_win["seg_uid"].nunique() >= 2:
        groups = df_win["seg_uid"].astype(str).to_numpy()
        group_name = "seg_uid"

    # ------- pipeline -------
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=float(pca_var), svd_solver="full",
                                 whiten=False, random_state=random_state)))
    try:
        gval = float(gamma_val)
    except Exception:
        gval = gamma_val  # "scale"/"auto"
    clf = SVC(kernel=kernel,
              C=float(C_val),
              gamma=gval if kernel == "rbf" else "scale",
              class_weight="balanced",
              probability=True,
              random_state=random_state)
    steps.append(("clf", clf))
    pipe = Pipeline(steps)

    # ------- cross-validation (no leakage) -------
    scoring = ["accuracy","balanced_accuracy","precision_macro","recall_macro","f1_macro","f1_weighted"]
    if group_name is not None:
        print("cv grouped")
        cv = GroupKFold(n_splits=max(2, int(cv_folds)))
        scores = cross_validate(pipe, X, y, groups=groups, cv=cv,
                                scoring=scoring, n_jobs=-1, return_train_score=False)
    else:
        print("cv not grouped")
        # no grouping available → vanilla CV（注意这时只是报告用，holdout 再做防泄露切分）
        scores = cross_validate(pipe, X, y, cv=max(2, int(cv_folds)),
                                scoring=scoring, n_jobs=-1, return_train_score=False)

    # ------- holdout split (strict) -------
    if group_name is not None:
        print("holdout grouped")
        if test_ratio!=0:
            gss = GroupShuffleSplit(n_splits=1, train_size=1.0-float(test_ratio), random_state=random_state)
            idx_tr, idx_va = next(gss.split(X, y, groups))
            X_tr, X_va, y_tr, y_va = X[idx_tr], X[idx_va], y[idx_tr], y[idx_va]
        else:
            gss = GroupShuffleSplit(n_splits=1, train_size=1.0-float(0.1), random_state=random_state)
            idx_tr, idx_va = next(gss.split(X, y, groups))
            X_tr, X_va, y_tr, y_va = X, X[idx_va], y, y[idx_va]
            
    else:
        print("holdout not grouped")
        # time-ordered split + a small gap to reduce overlap leakage
        if "t_start" in df_win.columns and "t_end" in df_win.columns:
            t_mid = 0.5*(df_win["t_start"].to_numpy(float) + df_win["t_end"].to_numpy(float))
        else:
            # fallback: use row order
            t_mid = np.arange(len(df_win), dtype=float)
        order = np.argsort(t_mid)
        n = len(order)
        n_tr = int((1.0 - float(test_ratio)) * n)
        gap = max(1, int(0.01 * n))  # 1% gap at the split point
        idx_tr = order[:max(0, n_tr - gap)]
        idx_va = order[min(n, n_tr + gap):]
        # last resort: if something went wrong, do stratified random split
        if idx_tr.size == 0 or idx_va.size == 0:
            idx_tr, idx_va = train_test_split(
                np.arange(n),
                test_size=float(test_ratio),
                random_state=random_state,
                stratify=y if len(np.unique(y)) > 1 else None
            )
        X_tr, X_va, y_tr, y_va = X[idx_tr], X[idx_va], y[idx_tr], y[idx_va]

    # ------- fit & evaluate -------
    
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_va)

    acc_h  = accuracy_score(y_va, y_pred)
    balc_h = balanced_accuracy_score(y_va, y_pred)
    prec_h, rec_h, f1_h, _ = precision_recall_fscore_support(y_va, y_pred, average="macro", zero_division=0)
    f1w_h = precision_recall_fscore_support(y_va, y_pred, average="weighted", zero_division=0)[2]

    report = classification_report(y_va, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_va, y_pred, labels=sorted(np.unique(y)))

    # ------- save model (with reproducibility meta) -------
    tag = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"svm_motion_{kernel}_{tag}.pkl"
    joblib.dump(dict(
        pipeline=pipe,
        meta=dict(
            tag=tag, kernel=kernel, C=float(C_val), gamma=str(gamma_val),
            pca=bool(use_pca), pca_var=float(pca_var),
            cv_folds=int(cv_folds), test_ratio=float(test_ratio),
            group_by=(group_name or "none"),
            exclude_gj=bool(exclude_gj),
            feat_cols_used=feat_cols_used
        )
    ), model_path)

    cv_summary = {k.replace("test_",""): (float(np.mean(v)), float(np.std(v)))
                  for k, v in scores.items() if k.startswith("test_")}

    return pipe, dict(
        cv=cv_summary,
        holdout=dict(acc=acc_h, balc=balc_h, prec=prec_h, rec=rec_h, f1=f1_h, f1w=f1w_h, n=int(len(y_va))),
        report=report, cm=cm, model_path=str(model_path), classes=sorted(np.unique(y))
    )

# Confusion matrix rendering (row-% color + text)

def confusion_figure(cm: np.ndarray, classes: List[int]):
    cm = np.asarray(cm, int)
    rowsum = cm.sum(axis=1, keepdims=True)
    rowsum[rowsum==0] = 1
    pct = cm / rowsum
    cls_names = [f"{c}:{LABEL_MAP.get(c,'cls')}" for c in classes]
    text = [[f"{100*pct[i,j]:.1f}%(n={cm[i,j]})" for j in range(cm.shape[1])] for i in range(cm.shape[0])]
    fig = go.Figure(data=go.Heatmap(z=pct, x=cls_names, y=cls_names, colorscale="Blues",
                                    zmin=0.0, zmax=1.0, showscale=True, text=text, texttemplate="%{text}",
                                    hovertemplate="True=%{y}<br>Pred=%{x}<br>%{text}<extra></extra>"))
    fig.update_layout(height=360, margin=dict(l=60,r=40,t=50,b=50), paper_bgcolor="white", plot_bgcolor="white",
                      title="Confusion Matrix (row-normalized %)")
    fig.update_xaxes(title_text="Predicted")
    fig.update_yaxes(title_text="True")
    return fig

@app.callback(
    Output("ddl-train-win", "options"),
    Output("ddl-train-win", "value"),
    Input("btn-scan-win", "n_clicks"),
    prevent_initial_call=True
)
def scan_train_window(_):
    files = sorted(DIR_TRAIN_WIN.glob("*.csv"),
                   key=lambda p: p.stat().st_mtime,
                   reverse=True)
    opts = [{"label": f.name, "value": str(f)} for f in files]
    default = str(files[0]) if files else None
    return opts, default

@app.callback(
    Output("ddl-holdout-files", "options"),
    Input("btn-refresh-trainval", "n_clicks"),
    prevent_initial_call=True
)
def refresh_trainval_list(_):
    files = sorted(DIR_TRAIN_VAL.glob("*.csv"))
    return [{"label": f.name, "value": str(f)} for f in files]


# Train callback
@app.callback(
    Output("txt-train-status", "children"),
    Output("div-train-metrics", "children"),
    Output("fig-cm", "figure"),
    Output("fig-f1", "figure"),
    Input("btn-train", "n_clicks"),
    State("ddl-train-win", "value"),
    State("ri-kernel", "value"),
    State("input-C", "value"),
    State("input-gamma", "value"),
    State("chk-pca", "value"),
    State("sl-pca-var", "value"),
    State("input-cv", "value"),
    State("input-test", "value"),
    State("chk-group-file", "value"),
    State("chk-excl-gj","value"),
    State("chk-excl-axis","value"),
    State("ddl-holdout-files", "value"),
    prevent_initial_call=True
)
def train_now(nc, ds_file, kernel, C_val, gamma_val, chk_pca, pca_var, cv_folds, test_ratio, chk_group, chk_excl_gj, chk_excl_axis, holdout_paths):
    if not ds_file:
        return "Select a train_window CSV first.", "", go.Figure(), go.Figure()

    df = pd.read_csv(ds_file)
    print("Unique files:", df['file'].nunique())
    if 'seg_uid' in df.columns:
        print("Unique segments:", df['seg_uid'].nunique())

    use_pca = (chk_pca is not None and "use" in chk_pca)
    group_by_file = (chk_group is not None and "group" in chk_group)
    exclude_gj = (chk_excl_gj is not None and "excl" in chk_excl_gj)
    print("GJ by training",exclude_gj)
    exclude_axis = ("excl_axis" in (chk_excl_axis or []))
    pipe, info = train_pipeline_quick(df, kernel, float(C_val), gamma_val,
                                        use_pca, float(pca_var), int(cv_folds), float(test_ratio), group_by_file, exclude_gj, exclude_axis)
    if not holdout_paths:
        print( "❗Select holdout CSV(s) from /train_val.")
        return "❗Select holdout CSV(s) from /train_val.", "", go.Figure(), go.Figure()
    
    # 读入 holdout 并对齐训练用特征
    if isinstance(holdout_paths, str):
        holdout_paths = [holdout_paths]
    dfs_ho = []
    for pth in holdout_paths:
        try:
            print("Holdout read success")
            dfs_ho.append(pd.read_csv(pth))
        except Exception as e:
            print("Holdout read failed:", pth, e)
    if not dfs_ho:
        return "❗Empty holdout set.", "", go.Figure(), go.Figure()
    df_ho = pd.concat(dfs_ho, axis=0, ignore_index=True)

    feat_cols_used = info.get("feat_cols_used", None)
    if feat_cols_used is None:
        # 与 train_pipeline_quick 内部保持一致的选择
        X_all, y_all, _, feat_cols_all = build_Xy_from_df(df)
        feat_cols_used = select_feat_cols(feat_cols_all, exclude_gj=exclude_gj, exclude_axis=exclude_axis)

    # 对齐列顺序（缺失列补0）
    X_ho = df_ho.reindex(columns=feat_cols_used, fill_value=0.0).to_numpy(float)
    y_ho = df_ho["label"].astype(int).to_numpy()

    y_pred = pipe.predict(X_ho)
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                    precision_recall_fscore_support,
                                    classification_report, confusion_matrix)
    acc_h  = float(accuracy_score(y_ho, y_pred))
    balc_h = float(balanced_accuracy_score(y_ho, y_pred))
    prec_h, rec_h, f1_h, _ = precision_recall_fscore_support(y_ho, y_pred, average="macro", zero_division=0)
    f1w_h = precision_recall_fscore_support(y_ho, y_pred, average="weighted", zero_division=0)[2]
    report = classification_report(y_ho, y_pred, output_dict=True, zero_division=0)
    classes = sorted(np.unique(np.concatenate([y_ho, y_pred])))
    cm = confusion_matrix(y_ho, y_pred, labels=classes)

    # —— 组织文本与图：CV（来自 info） + Holdout（外部） ——
    cv = info['cv']
    def fmt(mu_sd): mu, sd = mu_sd; return f"{mu:.3f}±{sd:.3f}"
    txt = f"Model saved: {info['model_path']}"

    metrics_div = html.Div([
        html.Div([html.Strong("Cross-Validation (mean±std): "),
                    html.Span(f"Acc {fmt(cv['accuracy'])} | BalAcc {fmt(cv['balanced_accuracy'])} | "
                            f"Prec_macro {fmt(cv['precision_macro'])} | Rec_macro {fmt(cv['recall_macro'])} | "
                            f"F1_macro {fmt(cv['f1_macro'])} | F1_weighted {fmt(cv['f1_weighted'])}")]),
        html.Div([html.Strong("Holdout (external from /train_val): "),
                    html.Span(f"Acc {acc_h:.3f} | BalAcc {balc_h:.3f} | "
                            f"Prec_macro {prec_h:.3f} | Rec_macro {rec_h:.3f} | "
                            f"F1_macro {f1_h:.3f} | F1_weighted {f1w_h:.3f} | "
                            f"Samples {len(y_ho)}")])
    ])

    # 混淆矩阵
    fig_cm = confusion_figure(cm, classes)

    # per-class F1
    labels_txt = [f"{c}:{LABEL_MAP.get(c,'cls')}" for c in classes]
    f1_vals = [float(report.get(str(c), {}).get("f1-score", 0.0)) for c in classes]
    fig_f1 = go.Figure(data=go.Bar(x=labels_txt, y=f1_vals, name="F1"))
    fig_f1.update_layout(title="Per-class F1 — external holdout",
                            yaxis_title="F1", xaxis_title="Class",
                            height=320, paper_bgcolor="white", plot_bgcolor="white")

    return txt, metrics_div, fig_cm, fig_f1


    
@app.callback(
    Output("fig-infer-preview","figure"),
    Input("ddl-files","value"),
    Input("rs-seg","value"),
    State("input-ppg-col","value"),
    State("input-fs","value"),
    State("input-win","value"),
    State("input-hop","value"),
    prevent_initial_call=True
)
def infer_preview(file_path, seg, ppg_col, fs, win_sec, hop_sec):
    fig_empty = go.Figure().update_layout(title="No model / no segment", height=160,
                                          paper_bgcolor="white", plot_bgcolor="white")
    try:
        if not file_path or not seg:
            return fig_empty
        # 取最新模型
        models = sorted(MODEL_DIR.glob("svm_motion_*.pkl"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
        if not models:
            return go.Figure().update_layout(title="No trained model found", height=160,
                                             paper_bgcolor="white", plot_bgcolor="white")

        M = joblib.load(models[0])
        pipe = M["pipeline"]
        feat_used = M["meta"]["feat_cols_used"]
        print("Selected Features for Predicting",len(feat_used))
        # 读入选定文件并截取区间
        fs = float(fs or FS_DEFAULT)
        df = pd.read_csv(file_path)
        if ppg_col not in df.columns:
            ppg_col = "IR" if "IR" in df.columns else df.columns[0]
        t_all = np.arange(len(df))/fs
        mask = (t_all >= seg[0]) & (t_all <= seg[1])
        df_seg = df.loc[mask].reset_index(drop=True)
        ppg = df_seg[ppg_col].to_numpy(float)
        N = len(ppg)
        if N < int(max(1.0, float(win_sec))*fs):
            return go.Figure().update_layout(title="Window too short", height=160,
                                             paper_bgcolor="white", plot_bgcolor="white")

        # IMU 预处理（与训练一致）
        imu_out = imu_preprocess_with_kf(df_seg, fs=fs)
        acc_g = imu_out['acc_raw']
        gyro_d = imu_out['gyro_raw']
        acc = imu_out['acc']
        gyro = imu_out['gyro']
        acc_lp = imu_out['acc_f'] 
        gyro_lp = imu_out['gyr_f'] 
        jerk = imu_out['jerk'] 
        roll=imu_out['roll'] 
        pitch=imu_out['pitch'] 
        bg = imu_out['gyro_bias']
        g_body=imu_out['g_body']
        g_dir = imu_out['g_dir'] 
        a_dyn= imu_out['a_dyn']
        acc_mag = imu_out['AccMag'] 
        gyro_mag = imu_out['GyroMag'] 
        jerk_mag = imu_out['JerkMag']
        rows, names = [], None
        for s, e in make_windows(N, fs, float(win_sec), float(hop_sec)):
            ppg_win = preprocess_ppg_min(ppg, fs=fs, hp_cut=0.2, mains=50)
            ax_win = a_dyn[s:e, 0]; ay_win = a_dyn[s:e, 1]; az_win = a_dyn[s:e, 2]
            accmag_win = acc_mag[s:e]
            gyromag_win = gyro_mag[s:e] #if 'GyroMag' in locals() else None
            jerkmag_win = jerk_mag[s:e] #if 'JerkMag' in locals() else None
            roll_win = roll[s:e]
            pitch_win = pitch[s:e]
            names, feats = extract_features_window(
                fs=fs,
                ppg_win=ppg_win,
                a_dyn_xyz=(ax_win, ay_win, az_win),
                accmag_win=accmag_win,
                gyromag_win=gyromag_win,
                jerkmag_win=jerkmag_win,
                roll_win=roll_win, pitch_win=pitch_win,
                psd_fmax=8.0
            )
            rows.append(feats)
        if not rows:
            return go.Figure().update_layout(title="No windows generated", height=160,
                                             paper_bgcolor="white", plot_bgcolor="white")

        X_all = pd.DataFrame(rows, columns=names)
        # 对齐训练时的特征列
        X = X_all.reindex(columns=feat_used, fill_value=0.0).to_numpy(float)
        #print(X)
        print("Shape of Data Features DF for predicting Segment",np.shape(X))
        yhat = pipe.predict(X).astype(int)

        # 生成离散色热力图（无渐变条）
        ts = seg[0] + (np.arange(len(yhat))*float(hop_sec) + float(win_sec)/2.0)
        z = yhat[np.newaxis, :]
        cs = discrete_colorscale_from_map(LABEL_COLORS)
        fig = go.Figure(data=go.Heatmap(
            z=z, x=ts, y=["State"], colorscale=cs, showscale=False,
            zmin=min(LABEL_COLORS.keys()), zmax=max(LABEL_COLORS.keys())
        ))
        # 图例（离散标注）
        for k, color in LABEL_COLORS.items():
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                     marker=dict(color=color, size=10),
                                     name=f"{k}:{LABEL_MAP[k]}"))
        fig.update_layout(height=140, margin=dict(l=50, r=30, t=40, b=30),
                          title="Predicted Motion (discrete)",
                          paper_bgcolor="white", plot_bgcolor="white")
        return fig
    except Exception as e:
        return go.Figure().update_layout(title=f"Inference error: {e}", height=160,
                                         paper_bgcolor="white", plot_bgcolor="white")

# ========================
# Entrypoint
# ========================

def main():
    app.run(debug=True, port=8051)

if __name__ == "__main__":
    main()
