# PPG Signal Processing and Visualization with Dash (Interactive)

import os
import webbrowser
import numpy as np
import pandas as pd
import dash
from dash import html, dcc, Input, Output, State, ctx
import plotly.graph_objs as go
from scipy import signal
from dash.dash_table import DataTable
from dash.dash_table.Format import Format, Scheme
from plotly.subplots import make_subplots

 # Constants
FS = 400  # Sampling frequency in Hz
MIN_BPM = 40  # Minimum expected heart rate for artifact rejection
MAX_BPM = 180  # Maximum expected heart rate for artifact rejection
#DEFAULT_FOLDER_MAIN = "/mnt/d/Tubcloud/Shared/PPG/Test Data"
#DEFAULT_FOLDER = "/mnt/d/Tubcloud/Shared/PPG/Test Data/25July25"
#DEFAULT_FOLDER_MAIN = "/home/trinker/only_view/Test Data"
#DEFAULT_FOLDER = "/home/trinker/only_view/Test Data/25July25"
PORT: int = 8050                         # Dash port (auto‑open in browser)
G = 9.81


# --- Signal Processing Functions ---
def highpass_filter(sig, cutoff=0.5, fs=FS, order=2):
    """Apply high-pass Butterworth filter to remove baseline drift."""
    nyq = 0.5 * fs
    b, a = signal.butter(order, cutoff / nyq, btype='high')
    return signal.filtfilt(b, a, sig)

def bandpass_filter(sig, lowcut=0.5, highcut=5, fs=FS, order=5):
    """Apply band-pass Butterworth filter to isolate the heart rate frequency range."""
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return signal.filtfilt(b, a, sig)

def notch_filter(sig, notch_freq=50.0, fs=FS, Q=30):
    """Apply notch filter to remove power line interference (e.g., 50 or 60 Hz)."""
    b, a = signal.iirnotch(notch_freq, Q, fs)
    return signal.filtfilt(b, a, sig)

def wavelet_denoise(sig, level=4):
    """Approximate wavelet-like denoising using Savitzky-Golay filtering from scipy."""
    window_length = min(len(sig) // (2 ** level) * 2 + 1, len(sig))
    if window_length < 5:  # ensure minimum window size
        window_length = 5 if len(sig) >= 5 else len(sig) | 1
    return signal.savgol_filter(sig, window_length, polyorder=3)

def robust_std(x):
    x = np.asarray(x).ravel()
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return 1.4826 * mad

#-------------------------------HR & Peaks---------------------------
def estimate_hr(peaks, fs=FS):
    """Estimate heart rate from detected peak indices."""
    if len(peaks) < 2:
        return np.nan, []
    rr = np.diff(peaks) / fs
    return 60 / np.mean(rr), rr

def detect_common_peaks(ir, red, max_bpm=MAX_BPM):
    """Detect peaks that are present in both IR and RED signals (intersection)."""
    peaks_ir, _ = signal.find_peaks(ir, distance=int(FS*60/max_bpm), prominence=0.5)
    peaks_red, _ = signal.find_peaks(red, distance=int(FS*60/max_bpm), prominence=0.5)
    return peaks_ir #only use Ir
    #return np.intersect1d(peaks_ir, peaks_red)

# --- Artifact Rejection ---
def reject_artifacts(peaks, rr_intervals, lower_bpm=MIN_BPM, upper_bpm=MAX_BPM):
    """Filter out physiologically implausible peaks based on RR interval range."""
    lower_rr = 60 / upper_bpm
    upper_rr = 60 / lower_bpm
    valid_indices = np.where((rr_intervals >= lower_rr) & (rr_intervals <= upper_rr))[0]
    clean_peaks = [peaks[0]]
    for i in valid_indices + 1:
        clean_peaks.append(peaks[i])
    return np.array(clean_peaks)

def caculate_clean_peaks(peaks, fs=FS):
    """clean hr, clean rr, clean peaks"""
    hr, rr = estimate_hr(peaks, fs)
    if np.isnan(hr):
        print(f"⚠️ Could not estimate HR")
        return None

    clean_peaks = reject_artifacts(peaks, rr)
    if len(clean_peaks) < 2:
        print(f"⚠️ Not enough clean peaks after artifact rejection")
        return None
    hr_clean, clean_rr = estimate_hr(clean_peaks, fs)
    return hr_clean, clean_rr, clean_peaks

#------------hrv----------------
def calculate_hrv(rr):
    """Calculate heart rate variability as the standard deviation of RR intervals (SDNN)."""
    return np.std(rr) * 1000 if len(rr) >= 2 else np.nan

# ---------------------- Aboy++ peak detector (raw PPG) ----------------------
# —— Windowing utilities ——
def window_indices(N, fs, win_sec=1.0, hop_sec=0.5):
    w = int(win_sec * fs)
    h = int(hop_sec * fs)
    w = max(2, w); h = max(1, h)
    starts = np.arange(0, max(1, N-w+1), h, dtype=int)
    for s in starts:
        e = min(N, s + w)
        yield s, e
        
def _detect_maxima_adaptive(sig, fs, min_dist, amp_percentile=65, prom_scale=0.25):
    """DetectMaxima helper with percentile-based amplitude & prominence.
    1) pre-detect peaks loosely; 2) derive thresholds from percentile; 3) final detect.
    Returns indices and dict props.
    """
    pre_peaks, _ = signal.find_peaks(sig, distance=max(1, int(min_dist*0.8)))
    if len(pre_peaks) == 0:
        return np.array([], dtype=int), {}
    pk_vals = sig[pre_peaks]
    amp_thr = np.percentile(pk_vals, amp_percentile)
    prom_thr = max(1e-6, prom_scale * amp_thr)
    peaks, props = signal.find_peaks(sig, distance=int(min_dist), prominence=prom_thr)#, height=amp_thr*0.5)
    return peaks, props
# cacu minmum amp, one peak between 2 -peaks
def aboypp_peak_hr(ppg_raw, fs=FS, window_sec=10.0,
                    hp_cut=0.2, notch_hz=None,
                    init_HRi=0.0, amp_percentile=65,
                    low_cut=0.5, hi_cap=8.0,
                    min_bpm=MIN_BPM, max_bpm=MAX_BPM):
    """Aboy++ style windowed peak detector on (minimally-preprocessed) raw PPG.
    Steps per 10s window:
      • high-pass + optional notch; band-pass with dynamic upper cutoff set by previous HRi
      • DetectMaxima → preliminary peaks → PP-times t_pp; define Pd = t_pp above 30th percentile
      • HR index: if median(Pd)*0.5 < mean(Pd) < median(Pd)*1.5, HRi = std(Pd)/mean(Pd)*10 else keep previous
      • HRwin = fs / ((1+HRi)*3); Final peaks with distance ≥ 2*HRwin, prominence ≥ 25% avg systolic amp
      • Update HRi for next window
    Returns dict with: peaks_all, hr_series[(t_mid, hr)], HRi_series, hr_global, hrv_ms, rr
    """
    x = highpass_filter(ppg_raw, hp_cut, fs)
    if notch_hz:
        x = notch_filter(x, f0=notch_hz, fs=fs)
    N = len(x)
    W = int(window_sec * fs)
    peaks_all = []
    HRi = float(init_HRi)
    hr_series = []
    HRi_series = []

    for s, e in window_indices(N, fs, window_sec, window_sec):  # non-overlap
        seg = x[s:e]
        if len(seg) < max(64, int(0.5*fs)):
            continue
        # dynamic highcut from previous HRi (capped)
        high_cut = min(hi_cap, max(1.5, (1.0 + HRi) * 3.0))  # base=3Hz, scaled by (1+HRi)
        seg_f = bandpass_filter(seg, low_cut, high_cut, fs=fs, order=2)

        # preliminary detection for HR index
        dist0 = int(fs * 60 / max_bpm)
        pk0, _ = _detect_maxima_adaptive(seg_f, fs, min_dist=dist0, amp_percentile=amp_percentile, prom_scale=0.25)
        if len(pk0) >= 2:
            tpp = np.diff(pk0) / fs
            if len(tpp) > 0:
                q30 = np.percentile(tpp, 30)
                Pd = tpp[tpp >= q30]
                if len(Pd) >= 2:
                    m_med, m_mean = np.median(Pd), np.mean(Pd)
                    if (m_med*0.5) < m_mean < (m_med*1.5):
                        HRi = float(np.std(Pd) / (m_mean + 1e-12) * 10.0)
                # else: keep previous HRi

        # HR window & final constraint
        HRwin = fs / ((1.0 + HRi) * 3.0)
        min_dist_final = max(int(2 * HRwin), int(fs * 60 / max_bpm))

        # average systolic amplitude from top 30% peaks in seg_f
        if len(pk0) > 0:
            pk_vals = seg_f[pk0]
            top = pk_vals[np.argsort(pk_vals)][int(0.7*len(pk_vals)):] if len(pk_vals) >= 3 else pk_vals
            avg_sys = float(np.mean(top)) if len(top) else float(np.mean(pk_vals)) if len(pk_vals) else robust_std(seg_f)
        else:
            avg_sys = robust_std(seg_f)
        prom_final = max(1e-6, 0.25 * avg_sys)

        pk_final, _ = signal.find_peaks(seg_f, distance=min_dist_final, prominence=prom_final)
        # window HR for plot
        if len(pk_final) >= 2:
            hr_w, _ = estimate_hr(pk_final + s, fs)
            hr_series.append(( (s+e)/(2*fs), hr_w ))
        HRi_series.append(( (s+e)/(2*fs), HRi ))

        peaks_all.extend(list(pk_final + s))

    peaks_all = np.asarray(peaks_all, dtype=int)
    # global HR from all peaks with artifact rejection
    if len(peaks_all) >= 2:
        hr0, rr0 = estimate_hr(peaks_all, fs)
        peaks_clean = reject_artifacts(peaks_all, rr0, fs)
        if len(peaks_clean) >= 2:
            hr_g, rr = estimate_hr(peaks_clean, fs)
            hrv_ms = calculate_hrv(rr)
        else:
            hr_g, rr, hrv_ms = hr0, rr0, calculate_hrv(rr0)
    else:
        hr_g, rr, hrv_ms = np.nan, [], np.nan

    return dict(peaks_all=peaks_all,
                hr_series=np.asarray(hr_series) if len(hr_series) else np.empty((0,2)),
                HRi_series=np.asarray(HRi_series) if len(HRi_series) else np.empty((0,2)),
                hr_global=hr_g, hrv_ms=hrv_ms, rr=rr)
    
#  distance, remove dirty segments, add peaks at boundary of windows

def aboypp_peak_hr_windowed(
    ppg_raw, fs=400,
    window_sec=10.0, hop_sec=2.0,
    commit_tail_sec=4.0,           # ← 改成 4 s 提交带
    init_HRi=0.0, ema_alpha=0.3,
    dedup=True, dedup_coef=0.2,    # 合并后最小RR去重系数（0.5×RR_est）
    **aboy_kwargs
):
    """
    Overlapped Aboy++ wrapper:
      - 10 s window, 2 s hop (80% overlap) by default
      - per-window run of your aboypp_peak_hr(...)
      - commit tail 'commit_tail_sec' from each window (e.g., 4 s), then global de-dup
      - output per-window HR, HRi, HRwin (samples), and global peaks

    Returns
    -------
    dict: {
      t_mid: (W,)                 # window mid-time (s)
      hr_win: (W,)                # HR per window (BPM, NaN allowed)
      HRi_win: (W,)               # HRi per window (dimensionless)
      HRwin_samp: (W,)            # HRwin per window (in samples)
      minRR_sec: (W,)             # 2*HRwin / fs (seconds) - dynamic minimal RR line
      peaks_global: (P,)          # committed global peaks (deduped)
      per_win_range: list[(s,e)]  # sample range per window
      per_win_commit: list[(c0,c1)] # local commit range per window (samples)
    }
    """
    x = np.asarray(ppg_raw, float).ravel()
    N = len(x)
    W = int(window_sec * fs)
    H = int(hop_sec * fs)
    assert W > H > 0, "window must be larger than hop"
    tail = int(commit_tail_sec * fs)
    tail = max(1, min(tail, W))   # clamp

    t_mid, hr_win, HRi_win = [], [], []
    HRwin_samp, minRR_sec = [], []
    per_win_range, per_win_commit = [], []
    peaks_all = []

    HRi_prev = float(init_HRi)

    for s in range(0, max(1, N - W + 1), H):
        e = s + W
        seg = x[s:e]

        # 你的主算法 (要求 aboypp_peak_hr 返回 peaks_all / hr_global / HRi 或 rr)
        info = aboypp_peak_hr(seg, fs=fs, window_sec=window_sec,
                              init_HRi=HRi_prev, **aboy_kwargs)

        pk_loc = np.asarray(info.get("peaks_all", []), dtype=int)

        # HR per window
        hr_meas = float(info.get("hr_global", np.nan))
        if not np.isfinite(hr_meas):
            rr = info.get("rr", None)
            if rr is not None and len(rr):
                hr_meas = 60.0 / np.median(rr)

        # HRi per window（若主算法没返回HRi，这里可能为 NaN）
        HRi = float(info.get("HRi", np.nan))

        # HRwin（samples），以及 2×HRwin -> 秒
        if np.isfinite(HRi):
            hrwin = fs / ((1.0 + HRi) * 3.0)
            min_rr_s = (2.0 * hrwin) / fs
        else:
            hrwin = np.nan
            min_rr_s = np.nan

        # 传递 HRi → 下一窗（EMA）
        if np.isfinite(hr_meas):
            HRi_prev = (1 - ema_alpha) * HRi_prev + ema_alpha * hr_meas if HRi_prev > 0 else hr_meas

        # 记录 per-window 信息
        t_mid.append((s + e) / (2.0 * fs))
        hr_win.append(hr_meas)
        HRi_win.append(HRi)
        HRwin_samp.append(hrwin)
        minRR_sec.append(min_rr_s)
        per_win_range.append((s, e))

        # 提交 tail 4 s: [W - tail, W)
        c0, c1 = W - tail, W
        per_win_commit.append((c0, c1))
        if len(pk_loc):
            m = (pk_loc >= c0) & (pk_loc < c1)
            if np.any(m):
                peaks_all.extend((pk_loc[m] + s).tolist())

    # 合并输出数组
    t_mid   = np.asarray(t_mid, float)
    hr_win  = np.asarray(hr_win, float)
    HRi_win = np.asarray(HRi_win, float)
    HRwin_samp = np.asarray(HRwin_samp, float)
    minRR_sec  = np.asarray(minRR_sec, float)

    peaks_global = np.asarray(sorted(peaks_all), dtype=int)

    # 合并后去重（基于全局 HR 中位数）
    if dedup and len(peaks_global) > 1:
        valid_hr = hr_win[np.isfinite(hr_win)]
        hr_hat = float(np.median(valid_hr)) if len(valid_hr) else 75.0
        min_dist = int(dedup_coef * fs * 60.0 / np.clip(hr_hat, 40, 220))  # 0.5×RR_est
        keep = [peaks_global[0]]
        for p in peaks_global[1:]:
            if p - keep[-1] >= min_dist:
                keep.append(p)
        peaks_global = np.asarray(keep, int)

    return dict(
        t_mid=t_mid, hr_win=hr_win,
        HRi_win=HRi_win, HRwin_samp=HRwin_samp, minRR_sec=minRR_sec,
        peaks_global=peaks_global,
        per_win_range=per_win_range, per_win_commit=per_win_commit
    )



def build_aboypp_windowed_sync_figure(
    ppg_raw, fs, win_out,
    window_sec=10.0, hop_sec=2.0,
    height=880, show_commit_band=True, show_hr_points=True
):
    """
    Three-row synchronized figure (for Dash):
      Row 1: PPG waveform + shaded commit bands + committed peaks
      Row 2: HR per window (t_mid vs hr_win)
      Row 3: HRi(t) (left y-axis) and 2×HRwin minimal RR (seconds, right y-axis)

    Inputs
    ------
    ppg_raw : 1D array
    fs : float
    win_out : dict from aboypp_peak_hr_windowed(...)
      - 't_mid', 'hr_win', 'HRi_win', 'HRwin_samp', 'minRR_sec'
      - 'peaks_global', 'per_win_range', 'per_win_commit'
    """
    x = np.asarray(ppg_raw, float).ravel()
    t = np.arange(len(x)) / float(fs)

    t_mid   = np.asarray(win_out.get("t_mid", []), float)
    hr_win  = np.asarray(win_out.get("hr_win", []), float)
    HRi_win = np.asarray(win_out.get("HRi_win", []), float)
    minRR_s = np.asarray(win_out.get("minRR_sec", []), float)
    peaks_g = np.asarray(win_out.get("peaks_global", []), int)
    win_rng = win_out.get("per_win_range", [])
    win_cmt = win_out.get("per_win_commit", [])

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.06,
        specs=[[{}],[{}],[{"secondary_y": True}]]  # bottom subplot has secondary y
    )

    # --- Row 1: PPG + commit bands + peaks ---
    fig.add_trace(
        go.Scatter(x=t, y=x, name="PPG (raw/minimal)",
                   line=dict(color="purple", width=1.0)),
        row=1, col=1
    )
    if show_commit_band and len(win_rng) == len(win_cmt) and len(win_rng) > 0:
        for (s, e), (c0, c1) in zip(win_rng, win_cmt):
            t0 = (s + int(c0)) / fs
            t1 = (s + int(c1)) / fs
            fig.add_vrect(x0=t0, x1=t1, fillcolor="orange", opacity=0.15,
                          line_width=0, layer="below", row=1, col=1)
        # dummy legend entry
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode="markers",
                       marker=dict(color="orange", size=10),
                       name="Commit band (tail)"),
            row=1, col=1
        )
    if peaks_g.size:
        fig.add_trace(
            go.Scatter(x=t[peaks_g], y=x[peaks_g], mode="markers",
                       marker=dict(color="red", size=6),
                       name="Committed peaks"),
            row=1, col=1
        )

    # --- Row 2: HR per window ---
    mask_hr = np.isfinite(hr_win)
    if np.any(mask_hr):
        fig.add_trace(
            go.Scatter(
                x=t_mid[mask_hr], y=hr_win[mask_hr],
                mode="lines+markers" if show_hr_points else "lines",
                line=dict(width=2), marker=dict(size=5),
                name="HR per window (BPM)"
            ),
            row=2, col=1
        )

    # --- Row 3: HRi(t) and 2×HRwin minimal RR (seconds) ---
    mask_hri = np.isfinite(HRi_win)
    if np.any(mask_hri):
        fig.add_trace(
            go.Scatter(x=t_mid[mask_hri], y=HRi_win[mask_hri],
                       mode="lines+markers",
                       line=dict(color="#1f77b4", width=2),
                       marker=dict(size=5),
                       name="HRi (dimensionless)"),
            row=3, col=1, secondary_y=False
        )
    mask_rr = np.isfinite(minRR_s)
    if np.any(mask_rr):
        fig.add_trace(
            go.Scatter(x=t_mid[mask_rr], y=minRR_s[mask_rr],
                       mode="lines+markers",
                       line=dict(color="#2ca02c", width=2, dash="dash"),
                       marker=dict(size=5),
                       name="2×HRwin (minimal RR, s)"),
            row=3, col=1, secondary_y=True
        )

    # --- Axes & layout ---
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="HR (BPM)", row=2, col=1)
    fig.update_yaxes(title_text="HRi", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Min RR (s)", row=3, col=1, secondary_y=True)

    fig.update_xaxes(title_text="Time (s)", row=3, col=1)

    # Aesthetics
    fig.update_layout(
        title=f"Aboy++ — window={window_sec}s, hop={hop_sec}s (commit tail={int((win_cmt[0][1]-win_cmt[0][0])/fs) if win_cmt else 0}s)",
        height=height,
        margin=dict(l=50, r=30, t=50, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")

    return fig

#------------Spo2---------------
def compute_mean_R(ir, red, peaks, fs=FS):
    """Return mean R after PI + outlier filtering; NaN if none valid. For Spo2"""
    Rs = []
    for p0, p1 in zip(peaks[:-1], peaks[1:]):
        seg_ir, seg_red = ir[p0:p1], red[p0:p1]
        if seg_ir.size < (fs/5):
            continue
        ac_ir,  dc_ir  = np.ptp(seg_ir),  np.mean(seg_ir)
        ac_red, dc_red = np.ptp(seg_red), np.mean(seg_red)
        PI = ac_ir / dc_ir
        if PI < 0.02:               # perfusion too weak
            continue
        R = (ac_red/dc_red) / (ac_ir/dc_ir)
        Rs.append(R)
    if not Rs:
        print("None Ratio")
        return np.nan
    R_arr = np.asarray(Rs)
    print("Ratio_raw:", max(R_arr), min(R_arr), len(R_arr))
    med = np.median(R_arr)
    mad = np.median(np.abs(R_arr-med))
    good = R_arr[np.abs(R_arr-med) <= 3*mad] if mad else R_arr
    print("Ratio_filtered", max(good),min(good), len(good))
    print("R_mean:", R_arr.mean())
    print("R_clean_mean:", good.mean())
    return good.mean()

# old spo2 algo
def estimate_spo2_old(ir, red):
    ir_ac = np.ptp(ir)           
    ir_dc = np.mean(ir)          
    red_ac = np.ptp(red)
    red_dc = np.mean(red)
    ratio = (red_ac / red_dc) / (ir_ac / ir_dc)  
    print("ratio_old", ratio)
    return 110 - 25 * ratio   

def f_R_poly(R):
    """linear regression and parameters for SpO2"""
    c0, c1, c2 = 110.0, -25.0, 0 # need regression
    return c0 + c1*R + c2*(R**2) 

def estimate_spo2(ir, red, clean_peaks, fs=FS):
    """Estimate SpO₂ based on AC/DC ratio of red and infrared signals."""
    R_bar = compute_mean_R(ir, red, clean_peaks, fs)
    return np.nan if np.isnan(R_bar) else f_R_poly(R_bar)


# ---------------------- IMU FEATURES + MOVING-WINDOW CLASSIFIER -------------

MOTION_LABELS = ["Static", "StandUp", "SitDown", "Walking", "Resting", "Transition"]
LABEL_TO_ID = {name:i for i,name in enumerate(MOTION_LABELS)}
LABEL_COLORS = {
    "Static": "#141313",
    "Resting": "#4CAF50",
    "Walking": "#2196F3",
    "StandUp": "#FF9800",
    "SitDown": "#E91E63",
    "Transition": "#BDBDBD",
}
#--------basic----------
def imu_features(acc, gyro, fs):
    """
    input:acc, gyro (N,3)
    output:
      acc_mag   : √(ax²+ay²+az²)
      gyro_mag  : √(gx²+gy²+gz²)
      jerk_mag  : |d(acc)/dt|
    """
    acc_mag  = np.linalg.norm(acc,  axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    
    # jerk 
    jerk     = np.diff(acc, axis=0, prepend=acc[:1]) * fs
    jerk_mag = np.linalg.norm(jerk, axis=1)
    return acc_mag, gyro_mag, jerk_mag


def imu_bandpass_filter(sig, lowcut=0.1, highcut=520, fs=FS, order=5):
    """Apply band-pass Butterworth filter to isolate the heart rate frequency range."""
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return signal.filtfilt(b, a, sig, axis=0)

def butter_filt(sig, Wn, btype, order=4, axis=0):
    """Butterworth + filtfilt (zero-phase). Wn: normalized or [low, high]."""
    b, a = signal.butter(order, Wn, btype=btype)
    return signal.filtfilt(b, a, sig, axis=axis)

def lp(sig, fc, fs=FS, order=4, axis=0):
    return butter_filt(sig, fc/(0.5*fs), 'low', order=order, axis=axis)

def hp(sig, fc, fs=FS, order=2, axis=0):
    return butter_filt(sig, fc/(0.5*fs), 'high', order=order, axis=axis)

def notch(sig, f0=50.0, Q=30.0, fs=FS, axis=0):
    b, a = signal.iirnotch(f0, Q, fs)
    return signal.filtfilt(b, a, sig, axis=axis)

def preprocess_ppg_min(ppg, fs=FS, hp_cut=0.2, mains=None):
    """
    - Remove DC / drift (high-pass 0.1-0.3 Hz; default 0.2 Hz)
    - Optional mains notch at 50/60 Hz
    - Do NOT narrow band-pass yet (keep info for ANC / spectral methods)
    """
    y = hp(ppg, hp_cut, fs=fs, order=2, axis=0)
    if mains in (50, 60):
        y = notch(y, f0=float(mains), Q=30.0, fs=fs, axis=0)
    return y

def robust_mean(x, axis=0):
    """Median + MAD """
    x = np.asarray(x)
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis, keepdims=True) + 1e-12
    z = np.abs(x - med) / mad
    mask = (z < 3.5)  # Tukey-like rule
    
    if axis == 0:
        val = []
        for j in range(x.shape[1]):
            col = x[:, j]
            msk = mask[:, j]
            val.append(col[msk].mean() if msk.any() else col.mean())
        return np.array(val)
    else:
        raise NotImplementedError

def acc_to_rp_from_mean(acc_mean):
    """roll/pitch by acc"""
    ax, ay, az = acc_mean
    roll  = np.arctan2( ay,  np.sqrt(ax**2 + az**2) )
    pitch = np.arctan2(-ax,  np.sqrt(ay**2 + az**2) )
    return roll, pitch

def gravity_body_from_rp(roll, pitch):
    R_x = np.array([[1,0,0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll),  np.cos(roll)]])
    R_y = np.array([[ np.cos(pitch),0,np.sin(pitch)],
                    [0,1,0],
                    [-np.sin(pitch),0,np.cos(pitch)]])
    R = R_x @ R_y
    return R.T @ np.array([0,0,G])  # g in body frame


# —— Simple LPF gravity path (fast fallback) ——
def estimate_gravity_lpf(acc_mps2, fs=FS, fc=0.3):
    g_vec = lp(acc_mps2, fc, fs)
    a_dyn = acc_mps2 - g_vec
    g_dir = g_vec / (np.linalg.norm(g_vec, axis=1, keepdims=True) + 1e-9)
    return g_vec, a_dyn, g_dir

# —— Spectral helper ——
def _welch_peak(freqs, Pxx, f_lo=0.8, f_hi=3.0):
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        return np.nan, 0.0
    f_band, P_band = freqs[mask], Pxx[mask]
    i = int(np.argmax(P_band))
    fpk, ppk = float(f_band[i]), float(P_band[i])
    snr = ppk / (np.median(P_band) + 1e-12)
    return fpk, snr

#---------bias-----------
def estimate_bias_from_static(df, idx_start=int(5*FS), idx_end=int(100*FS), fs=FS,
                              acc_lp_fc=20, gyro_lp_fc=40,
                              index = True,
                              acc_in_g=True, gyro_in_dps=True):
    """
    df: DataFrame including AX,AY,AZ,GX,GY,GZ
    idx_start, idx_end: sampling sig [start, end)
    acc_in_g: acc unit in g, gyro_in_dps: rotete unit in °/s(dps)
    return:
      acc_bias (3,), gyro_bias (3,), roll0, pitch0, quality(dict)
    """
    if index:
        seg = slice(int(idx_start), int(idx_end))
        acc = df[['AX','AY','AZ']].to_numpy(float)[seg]
        gyro = df[['GX','GY','GZ']].to_numpy(float)[seg]
    else:
        acc = df[['AX','AY','AZ']].to_numpy(float)
        gyro = df[['GX','GY','GZ']].to_numpy(float)

    # transfer unit
    if acc_in_g:   acc = acc * G
    if gyro_in_dps: gyro = np.deg2rad(gyro)

    # lowpass
    acc_f  = lp(acc,  acc_lp_fc, fs, order=4, axis=0)
    gyro_f = lp(gyro, gyro_lp_fc, fs, order=4, axis=0)

    # robust mean
    acc_mean  = robust_mean(acc_f,  axis=0)
    gyro_mean = robust_mean(gyro_f, axis=0)

    # ini attitude by acc mean
    roll0, pitch0 = acc_to_rp_from_mean(acc_mean)

    #
    g_body0 = gravity_body_from_rp(roll0, pitch0)
    acc_bias  = acc_mean  - g_body0  # g + bias
    gyro_bias = gyro_mean  # gyro bias

    # quality check：acc_norm 与 g 的偏差，gyro 均方
    acc_residual = np.linalg.norm(acc_mean - acc_bias) - G
    gyro_rms = np.sqrt(np.mean(gyro_f**2, axis=0))
    quality = dict(
        acc_norm_error=float(acc_residual),
        gyro_rms=list(gyro_rms),
        window_len=int(idx_end-idx_start)
    )
    return acc_bias, gyro_bias, roll0, pitch0, quality


#---------EKF for attitude----------------
def ekf_attitude_rp(gyro, acc, fs=FS,
                    q_proc = np.array([5.0, 5.0,   # process noise for roll,pitch (bpm-ish scale -> just relative)
                                       0.05,0.05,0.05]),  # bias random-walk
                    r_meas_base = np.array([0.5, 0.5]),  # accel-derived roll/pitch noise (rad^2)
                    alpha_dyn_R = 3.0,
                    init=None):
    """
    Extended Kalman Filter over state:
        x = [roll, pitch, bgx, bgy, bgz]^T
    Process model (discrete Euler):
        roll_{k+1}  = roll_k  + dt * roll_dot(roll,pitch, gyro-bias)
        pitch_{k+1} = pitch_k + dt * pitch_dot(roll,pitch, gyro-bias)

    Measurement:
        z = [roll_acc, pitch_acc] from accelerometer

    r_meas adaptively upweighted when |acc| deviates from g (i.e., dynamic acceleration)
        scale = 1 + alpha_dyn_R * max(0, | |acc|-g | / g)

    Returns:
      roll, pitch(arrays)
    """
    dt = 1.0/fs
    N = len(acc)
    roll  = np.zeros(N)
    pitch = np.zeros(N)
    bg    = np.zeros((N,3))

    if init is not None:
        roll[0]  = float(init.get('roll0', 0.0))
        pitch[0] = float(init.get('pitch0', 0.0))
        bg[0]    = np.array(init.get('bg0', [0.0,0.0,0.0]), dtype=float)
    else:
        ax0, ay0, az0 = acc[0]
        roll[0]  = np.arctan2( ay0,  np.sqrt(ax0**2 + az0**2) )
        pitch[0] = np.arctan2(-ax0,  np.sqrt(ay0**2 + az0**2) )
        bg[0]    = np.zeros(3)

    P = np.diag([1.0, 1.0, 0.5,0.5,0.5])
    Q = np.diag(q_proc) * dt

    for k in range(1, N):
        r, p = roll[k-1], pitch[k-1]
        gx, gy, gz = gyro[k] - bg[k-1]

        roll_dot  = gx + gy*np.sin(r)*np.tan(p) + gz*np.cos(r)*np.tan(p)
        pitch_dot = gy*np.cos(r) - gz*np.sin(r)

        x_prev = np.array([r, p, *bg[k-1]])
        x_pred = x_prev + dt*np.array([roll_dot, pitch_dot, 0.0, 0.0, 0.0])

        s, c = np.sin(r), np.cos(r)
        tp = np.tan(p); sp2 = 1/np.cos(p)**2

        droll_droll  = gy*c*tp - gz*s*tp
        droll_dpitch = gy*s*sp2 + gz*c*sp2
        droll_dbgx   = -1.0
        droll_dbgy   = -s*tp
        droll_dbgz   = -c*tp

        dpitch_droll = -gy*s - gz*c
        dpitch_dpitch= 0.0
        dpitch_dbgx  = 0.0
        dpitch_dbgy  = -c
        dpitch_dbgz  =  s

        F = np.array([
            [1 + dt*droll_droll,   dt*droll_dpitch,   dt*droll_dbgx, dt*droll_dbgy, dt*droll_dbgz],
            [dt*dpitch_droll,      1 + dt*dpitch_dpitch, dt*dpitch_dbgx, dt*dpitch_dbgy, dt*dpitch_dbgz],
            [0,0,1,0,0],
            [0,0,0,1,0],
            [0,0,0,0,1]
        ])
        P = F @ P @ F.T + Q

        ax, ay, az = acc[k]
        roll_acc  = np.arctan2( ay,  np.sqrt(ax**2 + az**2) )
        pitch_acc = np.arctan2(-ax,  np.sqrt(ay**2 + az**2) )
        z = np.array([roll_acc, pitch_acc])

        acc_norm = np.linalg.norm(acc[k])
        dev = max(0.0, abs(acc_norm - G) / G)
        R = np.diag(r_meas_base * (1.0 + alpha_dyn_R * dev))

        H = np.array([[1,0,0,0,0],
                      [0,1,0,0,0]])
        y = z - H @ x_pred
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        x_upd = x_pred + (K @ y).ravel()
        P = (np.eye(5) - K @ H) @ P

        roll[k], pitch[k] = x_upd[0], x_upd[1]
        bg[k] = x_upd[2:5]

    return roll, pitch, bg



def gravity_from_rp(roll, pitch):
    """Gravity (m/s²) time series in body frame from arrays roll/pitch (yaw ignored)."""
    N = len(roll)
    g_body = np.zeros((N,3))
    for k,(r,p) in enumerate(zip(roll, pitch)):
        R_x = np.array([[1,0,0],
                        [0, np.cos(r), -np.sin(r)],
                        [0, np.sin(r),  np.cos(r)]])
        R_y = np.array([[ np.cos(p),0,np.sin(p)],
                        [0,1,0],
                        [-np.sin(p),0,np.cos(p)]])
        R = R_x @ R_y
        g_body[k] = R.T @ np.array([0,0,G])
    return g_body

global THRESHOLD 
THRESHOLD  =  dict(thr_acc_low=0.8, thr_gyro_low=np.deg2rad(30), thr_jerk_low=2.5, jerk_high=2.0, imp_th=0.6, snr_th=3.0)

# ---------------------- Motion classification (LPF or EKF) ------------------
def classify_window(a_dyn_win, gyro_win, gdir_win, fs,
                     thr_acc_low=0.8, thr_gyro_low=np.deg2rad(10), thr_jerk_low=0.5,
                     jerk_high=2.0, imp_th=0.6, snr_th=3.0):
    
    acc_mag = np.linalg.norm(a_dyn_win, axis=1)
    gyro_mag = np.linalg.norm(gyro_win, axis=1)
    acc_rms  = float(np.sqrt(np.mean(acc_mag**2)))
    gyro_rms = float(np.sqrt(np.mean(gyro_mag**2)))
    jerk     = np.diff(a_dyn_win, axis=0, prepend=a_dyn_win[:1]) * fs
    jerk_mag = np.linalg.norm(jerk, axis=1)
    jerk_rms = float(np.sqrt(np.mean(jerk_mag**2)))
    #print(jerk_rms)
    f, Pxx = signal.welch(acc_mag, fs=fs, window='hann', nperseg=min(len(acc_mag), int(2*fs)))
    fpk, snr = _welch_peak(f, Pxx, 0.8, 3.0)
    vproj = np.sum(np.sum(a_dyn_win * gdir_win, axis=1)) / fs
    if (gyro_rms < thr_gyro_low) and (acc_rms < thr_acc_low) and (jerk_rms < thr_jerk_low):
        return "Static"
    if (snr > snr_th) and (0.8 <= fpk <= 3.0) and (acc_rms > thr_acc_low):
        return "Walking"
    if (jerk_rms > jerk_high) and (abs(vproj) > imp_th):
        return "StandUp" if vproj > 0 else "SitDown"
    return "Transition"

def promote_resting(labels, times, min_rest_sec=5.0):
    out = labels[:]
    i = 0
    while i < len(labels):
        if labels[i] == "Static":
            j = i
            while j+1 < len(labels) and labels[j+1] == "Static":
                j += 1
            dur = times[j][1] - times[i][0]
            if dur >= min_rest_sec:
                for k in range(i, j+1):
                    out[k] = "Resting"
            i = j + 1
        else:
            i += 1
    return out

def samplewise_labels(N, fs, win_list, win_labels):
    C = len(MOTION_LABELS)
    votes = np.zeros((N, C), dtype=int)
    for (s,e), lab in zip(win_list, win_labels):
        votes[s:e, LABEL_TO_ID[lab]] += 1
    ids = votes.argmax(axis=1)
    return ids


def imu_preprocess_with_kf(df, cols=None, fs=FS, acc_fc=20, gyro_fc=40,  static_t0=5.0, static_t1=100.0):
    """
    Full IMU preprocessing:
      1) initial bias estimation over a static segment
      2) bias removal
      3) low-pass accel/gyro
      4) EKF (roll,pitch + gyro bias)
      5) remove gravity → dynamic acceleration a_dyn
      6) scalar metrics: AccMag, GyroMag, JerkMag
    Returns: dict with arrays and metrics.
    """
    if cols == None:
        cols = dict(AX='AX',AY='AY',AZ='AZ',GX='GX',GY='GY',GZ='GZ')
        
    # Load and convert units
    acc_g  = df[[cols['AX'], cols['AY'], cols['AZ']]].to_numpy(float)
    gyro_d = df[[cols['GX'], cols['GY'], cols['GZ']]].to_numpy(float)
    acc    = acc_g * G
    gyro   = np.deg2rad(gyro_d)
    
    # 1) estimate biases from a static segment [static_t0, static_t1) in seconds
    idx0 = int(static_t0 * fs)
    idx1 = int(static_t1 * fs)
    acc_b, gyr_b, r0, p0, qual = estimate_bias_from_static(df, idx0, idx1, fs=fs, index=True,
                                                            acc_in_g=True, gyro_in_dps=True)
    # 2) bias removal and LPF
    acc_bc = acc - acc_b
    gyro_bc= gyro - gyr_b
    acc_lp = lp(acc_bc, acc_fc, fs)
    gyro_lp= lp(gyro_bc, gyro_fc, fs)
    
    # 3) EKF attitude  (roll,pitch,gyro bias)
    init = dict(roll0=r0, pitch0=p0, bg0=gyr_b)
    roll, pitch, bg = ekf_attitude_rp(gyro_lp, acc_lp, fs=fs, init=None) # initial curently use first secs
    
    # 4) gravity removal
    g_body = gravity_from_rp(roll, pitch)
    a_dyn  = acc_lp - g_body
    g_dir  = g_body / (np.linalg.norm(g_body, axis=1, keepdims=True) + 1e-9)
    
    # 5) metrics
    acc_mag  = np.linalg.norm(a_dyn, axis=1)
    gyro_mag = np.linalg.norm((gyro_lp), axis=1)
    jerk     = np.diff(a_dyn, axis=0, prepend=a_dyn[:1]) * fs
    jerk_mag = np.linalg.norm(jerk, axis=1)
    
    output = dict(
        acc_raw = acc_g,
        gyro_raw = gyro_d,
        acc = acc,
        gyro = gyro,
        acc_f=acc_lp, 
        gyr_f=gyro_lp, 
        jerk = jerk, 
        roll=roll, 
        pitch=pitch, 
        gyro_bias=bg,
        g_body=g_body,
        g_dir = g_dir, 
        a_dyn=a_dyn,
        AccMag=acc_mag, 
        GyroMag=gyro_mag, 
        JerkMag=jerk_mag
    )

    return output

def classify_motion_from_df(df, fs=FS, cols=None, win_sec=1.0, hop_sec=0.5, thresholds=THRESHOLD,
                             use_ekf=True, static_t0=5.0, static_t1=100.0):
    """Motion classifier with two gravity-removal paths:
    - LPF path (default): fast, no static segment required.
    - EKF path (optional): uses static segment to estimate sensor biases, then EKF roll/pitch and gravity removal.
    """
    imu_out = imu_preprocess_with_kf(df, cols, fs, static_t0=static_t0, static_t1=static_t1)
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
    

    # Window-wise labels
    N = len(acc)
    win_list, win_labels, win_times = [], [], []
    thr = dict(thr_acc_low=0.4, thr_gyro_low=np.deg2rad(10), thr_jerk_low=0.5, jerk_high=2.0, imp_th=0.6, snr_th=3.0)
    if thresholds:
        thr.update(thresholds)
    for s,e in window_indices(N, fs, win_sec, hop_sec):
        lab = classify_window(a_dyn[s:e], (gyro_lp if use_ekf else lp(gyro, 40, fs))[s:e], g_dir[s:e], fs, **thr)
        win_list.append((s,e)); win_labels.append(lab); win_times.append((s/fs, e/fs))
    win_labels2 = promote_resting(win_labels, win_times, min_rest_sec=5.0)
    ids = samplewise_labels(N, fs, win_list, win_labels2)

    # Scalars for ANC
    acc_mag  = np.linalg.norm(a_dyn, axis=1)
    gyro_mag = np.linalg.norm((gyro_lp if use_ekf else lp(gyro, 40, fs)), axis=1)
    jerk     = np.diff(a_dyn, axis=0, prepend=a_dyn[:1]) * fs
    jerk_mag = np.linalg.norm(jerk, axis=1)
    t = df['Time'].values if 'Time' in df.columns else np.arange(N)/fs
    unique, counts = np.unique(ids, return_counts=True)
    frac = {MOTION_LABELS[i]: float(counts[j]/N) for j,i in enumerate(unique)}
    return dict(time=t, a_dyn=a_dyn, acc_mag=acc_mag, gyro_mag=gyro_mag, jerk_mag=jerk_mag,
                windows=win_list, labels=win_labels2, ids=ids, frac=frac)
# =========================================================
# 3) Adaptive Noise Cancellation (ANC) for PPG with IMU refs
#    ── Multi-input NLMS (reference = IMU features)
# =========================================================
def build_xref_from_motion(imu_out):
    """
    Construct the multi-reference matrix Xref for ANC.
    Columns: [a_dyn_x, a_dyn_y, a_dyn_z, AccMag, GyroMag, JerkMag]  → shape (N,6)
    imu_out: dict returned by imu_preprocess_with_kf(...),
             must contain keys: 'a_dyn','AccMag','GyroMag','JerkMag'
    """
    Xref = np.column_stack([
        imu_out['a_dyn'][:, 0],           # dynamic accel x (m/s^2)
        imu_out['a_dyn'][:, 1],           # dynamic accel y (m/s^2)
        imu_out['a_dyn'][:, 2],           # dynamic accel z (m/s^2)
        imu_out['acc_mag'],                # ||a_dyn|| (m/s^2)
        imu_out['gyro_mag'],               # ||omega|| (rad/s)
        imu_out['jerk_mag']                # ||d(a_dyn)/dt|| (m/s^3)
    ])
    names = ["a_dyn_x","a_dyn_y","a_dyn_z","AccMag","GyroMag","JerkMag"]
    return Xref, names

def standardize_cols(X, eps=1e-8):
    """Z-score columns to avoid scaling issues for NLMS."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + eps
    return (X - mu) / sd, mu, sd

def motion_mask_from_labels(ids, fs=FS, pad_sec=0.25):
    moving_ids = [LABEL_TO_ID[x] for x in ("Walking","StandUp","SitDown","Transition")]
    mask = np.isin(ids, moving_ids)
    pad = int(pad_sec*fs)
    if pad > 0 and mask.any():
        idx = np.where(mask)[0]
        for i in idx:
            mask[max(0,i-pad):min(len(mask), i+pad+1)] = True
    return mask

def nlms_multi(ppg, Xref, mu=0.02, eps=1e-6, win=None, return_weights=True):
    """
    Multi-input NLMS (Normalized LMS).
      Prediction:  y_hat[n] = w^T[n] x[n]
      Error:       e[n]     = ppg[n] - y_hat[n]
      Update:      w[n+1]   = w[n] + mu * e[n] * x[n] / (eps + ||x[n]||^2)   (only if win[n] is True)

    Inputs:
      ppg  : (N,)        — minimally preprocessed PPG (e.g., DC removed, mains notched)
      Xref : (N,M)       — references (IMU-derived), recommend M=6 as defined in build_xref_from_imu
      mu   : step size   — 0.01~0.05 (start with 0.02)
      win  : (N,) bool   — update gate (e.g., motion window). If None, always update.
    Returns:
      y_clean : (N,)     — residual (cleaned PPG)
      W       : (N,M)    — weight history (for diagnostics), returned iff return_weights=True
    """
    ppg = np.asarray(ppg, float).ravel()
    X   = np.asarray(Xref, float)
    N, M = X.shape
    y_clean = np.zeros(N, dtype=float)
    if return_weights:
        W = np.zeros((N, M), dtype=float)
    w = np.zeros(M, dtype=float)

    for n in range(N):
        x = X[n]
        y_hat = float(np.dot(w, x))
        e = ppg[n] - y_hat
        # gated adaptation
        if win is None or win[n]:
            denom = eps + np.dot(x, x)
            w += (mu * e * x) / denom
        y_clean[n] = e
        if return_weights:
            W[n] = w

    return (y_clean, W) if return_weights else (y_clean, None)

# ---------------------- HR from ANC (peak detection on ANC output) ----------

def hr_from_anc_pipeline(ppg_raw, imu_res, fs=FS,
                         hp_cut=0.2, notch_hz=None,
                         anc_mu=0.02, gate_pad=0.25,
                         bp_lo=0.5, bp_hi=5.0,
                         min_bpm=MIN_BPM, max_bpm=MAX_BPM):
    """End-to-end HR estimation on ANC-cleaned PPG (non-motion gating)."""
    y_min = highpass_filter(ppg_raw, hp_cut, fs)
    if notch_hz:
        y_min = notch_filter(y_min, f0=notch_hz, fs=fs)
    Xref, _ = build_xref_from_motion(imu_res)
    Xz, mu_x, sd_x = standardize_cols(Xref)
    gate = motion_mask_from_labels(imu_res['ids'], fs=fs, pad_sec=gate_pad)  # True=Motion
    y_clean, W = nlms_multi(y_min, Xz, mu=anc_mu, win=gate, return_weights=True)
    y_band = butter_filt(y_clean, [bp_lo/(0.5*fs), bp_hi/(0.5*fs)], 'band', order=2)
    dist = int(fs * 60 / max_bpm)
    good = ~gate
    prom_ref = robust_std(y_band[good]) if np.any(good) else robust_std(y_band)
    prom = max(1e-6, 0.5 * prom_ref)
    peaks_all, _ = signal.find_peaks(y_band, distance=dist, prominence=prom)
    peaks_nm = peaks_all[good[peaks_all]] if np.any(peaks_all) else np.array([], dtype=int)
    use_peaks = peaks_nm if len(peaks_nm) >= 2 else peaks_all
    hr0, rr0 = estimate_hr(use_peaks, fs)
    if np.isnan(hr0) or len(use_peaks) < 2:
        return dict(y_min=y_min, y_clean=y_clean, y_band=y_band, gate=gate,
                    peaks_all=peaks_all, peaks_nm=peaks_nm, peaks_clean=np.array([], int),
                    hr=np.nan, hrv=np.nan, rr=[])
    peaks_clean = reject_artifacts(use_peaks, rr0, fs)
    if len(peaks_clean) < 2:
        peaks_clean = use_peaks
    hr, rr = estimate_hr(peaks_clean, fs)
    hrv = calculate_hrv(rr)
    return dict(y_min=y_min, y_clean=y_clean, y_band=y_band, gate=gate,
                peaks_all=peaks_all, peaks_nm=peaks_nm, peaks_clean=peaks_clean,
                hr=hr, hrv=hrv, rr=rr)
# =========================================================
# 4) End-to-end: IMU preprocess + ANC on PPG (single window)
# =========================================================
    """def imu_preprocess_and_anc(ppg_raw, df_imu, fs=400.0,
                           # PPG minimal
                           hp_cut=0.2, mains=50,
                           # IMU preprocessing
                           acc_fc=20, gyro_fc=40, static_idx=None, static_secs=2.0,
                           # ANC
                           anc_mu=0.02,
                           thr_acc_g=1.3, thr_gyro_dps=150, thr_jerk=2.0, pad_sec=0.25,
                           return_weights=True):

    Full pipeline for ANC using IMU:
      1) PPG minimal preprocessing (high-pass + optional mains notch; no narrow band)
      2) IMU preprocessing with static bias removal + LP + EKF roll/pitch + gravity removal -> a_dyn & metrics
      3) Build multi-reference Xref from IMU features; Z-score columns
      4) Build motion window; NLMS only adapts when 'motion' is present
      5) NLMS multi-input to obtain cleaned PPG residual

    Returns dict:
      ppg_min : minimally preprocessed PPG
      imu     : dict from imu_preprocess_with_kf(...)
      anc     : dict with y_clean, W (weights), motion_win, Xref_z, zscore params, thresholds, mu
 
    # ---- 1) PPG minimal（你已有 preprocess_ppg_min） ----
    ppg_min = preprocess_ppg_min(ppg_raw, fs=fs, hp_cut=hp_cut, mains=mains)

    # ---- 2) IMU preprocessing（你已有 imu_preprocess_with_kf） ----
    imu_out = imu_preprocess_with_kf(df_imu, fs=fs,
                                     acc_fc=acc_fc, gyro_fc=gyro_fc,
                                     static_secs=static_secs, static_idx=static_idx)

    # ---- 3) Xref + Z-score ----
    Xref, feat_names = build_xref_from_imu(imu_out)
    Xref_z, mu_x, sd_x = standardize_cols(Xref)

    # ---- 4) Motion window ----
    motion_win = build_motion_window(
        imu_out['AccMag'], imu_out['GyroMag'], imu_out['JerkMag'],
        thr_acc_g=thr_acc_g, thr_gyro_dps=thr_gyro_dps, thr_jerk=thr_jerk,
        fs=fs, pad_sec=pad_sec
    )

    # ---- 5) NLMS multi-input ----
    y_clean, W = nlms_multi(ppg_min, Xref_z, mu=anc_mu, win=motion_win, return_weights=return_weights)

    return dict(
        ppg_min = ppg_min,
        imu     = imu_out,
        anc     = dict(
            y_clean   = y_clean,
            W         = W,
            Xref_z    = Xref_z,
            Xref_mu   = mu_x,
            Xref_sd   = sd_x,
            feat_names= feat_names,
            motion_win= motion_win,
            mu        = anc_mu,
            thr       = dict(acc_g=thr_acc_g, gyro_dps=thr_gyro_dps, jerk=thr_jerk),
            pad_sec   = pad_sec
        )
    )
    """

import numpy as np
from scipy import signal, interpolate
import plotly.graph_objects as go

# =========================================================
# 1) Baseline utilities
# =========================================================
def welch_psd(x, fs, fmax=None, nperseg=None, noverlap=0.5):
    """Compute Welch PSD with Hann window and optional fmax clip."""
    x = np.asarray(x, float).ravel()
    if nperseg is None:
        nperseg = min(len(x), int(8*fs))  # ~8s window by default
    nlap = int(noverlap * nperseg) if isinstance(noverlap, float) else int(noverlap)
    f, Pxx = signal.welch(x, fs=fs, window="hann", nperseg=nperseg, noverlap=nlap)
    if fmax is not None:
        m = f <= float(fmax)
        f, Pxx = f[m], Pxx[m]
    return f, Pxx + 1e-18

def estimate_hr_freq_welch(x, fs, fmin=0.6, fmax=3.5):
    """Rough HR frequency estimate (Hz) from PPG via Welch."""
    f, P = welch_psd(x, fs, fmax=fmax)
    band = (f >= fmin) & (f <= fmax)
    if not np.any(band):
        return np.nan
    j = np.argmax(P[band])
    f_hr = float(f[band][j])
    return f_hr

# =========================================================
# 2) EMD (sifting) + CEEMD-lite reference construction
# =========================================================
def _local_extrema(x):
    """Return indices of local maxima and minima of a 1D array."""
    # maxima: x[i-1] < x[i] >= x[i+1]; minima: x[i-1] > x[i] <= x[i+1]
    dx = np.diff(x)
    # zero-crossings of derivative with sign check
    max_idx = np.where((np.hstack([dx, 0]) < 0) & (np.hstack([0, dx]) > 0))[0]
    min_idx = np.where((np.hstack([dx, 0]) > 0) & (np.hstack([0, dx]) < 0))[0]
    # remove endpoints (can be unstable)
    max_idx = max_idx[(max_idx > 1) & (max_idx < len(x)-2)]
    min_idx = min_idx[(min_idx > 1) & (min_idx < len(x)-2)]
    return max_idx, min_idx

def _sift_once(x, t):
    """One sifting step: build upper/lower envelopes via cubic spline; return mean envelope."""
    max_idx, min_idx = _local_extrema(x)
    if len(max_idx) < 2 or len(min_idx) < 2:
        return None  # cannot build envelopes -> stop
    # Add endpoints by mirroring to mitigate edge effect
    def _pad(idx):
        return np.r_[0, idx, len(x)-1]
    xi_max, xi_min = _pad(max_idx), _pad(min_idx)

    # Envelope interpolation
    cs_max = interpolate.CubicSpline(t[xi_max], x[xi_max], bc_type="natural")
    cs_min = interpolate.CubicSpline(t[xi_min], x[xi_min], bc_type="natural")
    env_up = cs_max(t)
    env_lo = cs_min(t)
    m = 0.5 * (env_up + env_lo)
    return m

def emd_sift(x, fs, max_imfs=6, max_sift=10, sd_thresh=0.2):
    """
    Basic EMD sifting to extract IMFs from signal x.
    - max_imfs: maximum IMFs to extract
    - max_sift: max sifting iterations per IMF
    - sd_thresh: stop criterion (SD index)
    Returns: imfs (list of arrays), residual (array)
    """
    x = np.asarray(x, float).ravel()
    t = np.arange(len(x)) / fs
    imfs = []
    r = x.copy()

    for _ in range(max_imfs):
        h = r.copy()
        for _ in range(max_sift):
            m = _sift_once(h, t)
            if m is None:
                break
            h_prev = h
            h = h - m
            sd = np.sum((h_prev - h)**2) / (np.sum(h_prev**2) + 1e-18)
            if sd < sd_thresh:
                break
        # Check stoppage: not enough extrema to continue
        max_idx, min_idx = _local_extrema(h)
        if len(max_idx) + len(min_idx) < 2:
            break
        imfs.append(h)
        r = r - h
        # if residual is monotonic-ish, stop
        mx, mn = _local_extrema(r)
        if len(mx) < 1 or len(mn) < 1:
            break

    return imfs, r

def ceemd_reference(x, fs, pairs=6, noise_ratio=0.2, max_imfs=6, max_sift=10, sd_thresh=0.2,
                    protect_hr=True, protect_bw=0.25, protect_harmonics=2,
                    low_motion_hz=0.4, high_motion_hz=6.0):
    """
    CEEMD-lite: build motion-artifact reference u(n) by averaging IMFs over complementary noise pairs.
    Steps:
      1) For i in 1..pairs: add white noise 'n' (std=noise_ratio*std(x)) -> x+n and x-n
      2) EMD both, then average IMFs level-wise
      3) Classify IMFs: protect cardiac IMFs near HR (and its harmonics), sum the rest as 'u_ref'
    Returns:
      dict with: u_ref, imfs_avg (list), residual, motion_idx, cardiac_idx, f_hr
    """
    x = np.asarray(x, float).ravel()
    N = len(x)
    sigma = np.std(x) + 1e-12
    imfs_accum = None
    max_levels = 0

    rng = np.random.default_rng(2025)
    for _ in range(pairs):
        n = rng.standard_normal(N) * (noise_ratio * sigma)
        for sign in (+1.0, -1.0):
            imfs, res = emd_sift(x + sign * n, fs, max_imfs=max_imfs, max_sift=max_sift, sd_thresh=sd_thresh)
            L = len(imfs)
            if L == 0:
                continue
            if imfs_accum is None:
                max_levels = L
                imfs_accum = [imfs[k].astype(float) for k in range(L)]
            else:
                # extend to the current max levels
                if L > max_levels:
                    # pad previous with zeros
                    imfs_accum += [np.zeros_like(x) for _ in range(L - max_levels)]
                    max_levels = L
                # accumulate
                for k in range(L):
                    if k < len(imfs_accum):
                        imfs_accum[k] = imfs_accum[k] + imfs[k]
                    else:
                        imfs_accum.append(imfs[k].copy())

    if imfs_accum is None or max_levels == 0:
        # fallback: no IMFs -> reference is zeros
        return dict(u_ref=np.zeros_like(x), imfs_avg=[], residual=x.copy(),
                    motion_idx=[], cardiac_idx=[], f_hr=np.nan)

    # Average over (2*pairs) realizations
    denom = 2.0 * pairs
    imfs_avg = [imf / denom for imf in imfs_accum]
    residual = x - np.sum(imfs_avg, axis=0)

    # --- IMF selection: protect cardiac; others -> motion set M ---
    # rough HR (Hz) to define protect bands
    f_hr = estimate_hr_freq_welch(x, fs, fmin=0.6, fmax=3.5)
    motion_idx, cardiac_idx = [], []
    for k, ck in enumerate(imfs_avg):
        # dominant freq of IMF k
        f, P = welch_psd(ck, fs, fmax=8.0)
        fk = f[np.argmax(P)]
        is_cardiac = False
        if protect_hr and np.isfinite(f_hr):
            # protect f_hr ± protect_bw and its harmonics
            for h in range(1, protect_harmonics + 1):
                if abs(fk - h * f_hr) <= protect_bw:
                    is_cardiac = True
                    break
        # low drift and high wideband -> motion by default
        if not is_cardiac and (fk <= low_motion_hz or fk >= high_motion_hz):
            motion_idx.append(k)
        elif is_cardiac:
            cardiac_idx.append(k)
        else:
            # decide by energy overlap with a HR-bandpass version of x
            b, a = signal.butter(2, [0.6/(0.5*fs), 3.5/(0.5*fs)], btype='band')
            x_hr = signal.filtfilt(b, a, x)
            corr = np.corrcoef(ck, x_hr)[0,1]
            (cardiac_idx if abs(corr) >= 0.2 else motion_idx).append(k)

    # motion reference as sum of motion IMFs + residual below HR band
    u_ref = np.zeros_like(x)
    for k in motion_idx:
        u_ref += imfs_avg[k]
    # (Optional) If residual is very slow, treat as drift -> include
    f_res, P_res = welch_psd(residual, fs, fmax=2.0)
    if f_res[np.argmax(P_res)] < 0.4:
        u_ref += residual

    return dict(u_ref=u_ref, imfs_avg=imfs_avg, residual=residual,
                motion_idx=motion_idx, cardiac_idx=cardiac_idx, f_hr=f_hr)

# =========================================================
# 3) NLMS ANC (stable, leaky)
# =========================================================
def nlms_anc(x, u_ref, L=32, mu=0.1, eps=1e-6, leak=1e-4):
    """
    Normalized LMS ANC: estimate motion m^(n) = w^T u(n) and clean e = x - m^.
      x    : observed PPG (N,)
      u_ref: reference (N,) – from CEEMD motion estimate
      L    : FIR length (tapped delay line)
      mu   : NLMS step size (0<mu<2; typical 0.05~0.5)
      leak : small leakage to stabilize weights
    Returns: y (motion estimate), e (clean), W (weight history)
    """
    x = np.asarray(x, float).ravel()
    u = np.asarray(u_ref, float).ravel()
    N = len(x)
    y = np.zeros(N)
    e = np.zeros(N)
    W = np.zeros((N, L))
    w = np.zeros(L)

    # Pre-build delayed matrix (Toeplitz-like) efficiently
    buf = np.zeros(L)
    for n in range(N):
        # update tapped-delay line: u[n], u[n-1], ..., u[n-L+1]
        buf[1:] = buf[:-1]
        buf[0] = u[n]
        y[n] = float(np.dot(w, buf))
        e[n] = x[n] - y[n]
        denom = eps + np.dot(buf, buf)
        w = (1.0 - leak) * w + (mu * e[n] * buf) / denom
        W[n] = w
    return y, e, W

# =========================================================
# 4) End-to-end: CEEMD reference + LMS ANC
# =========================================================
def remove_ma_cemd_lms(ppg, fs=400,
                       ce_pairs=6, ce_noise_ratio=0.2, ce_max_imfs=6,
                       ce_max_sift=10, ce_sd=0.2,
                       protect_bw=0.25, protect_harm=2,
                       low_motion_hz=0.4, high_motion_hz=6.0,
                       lms_L=32, lms_mu=0.1, lms_leak=1e-4):
    """
    Full pipeline:
      - CEEMD-lite -> motion reference u_ref
      - NLMS ANC   -> y (motion estimate), e (clean PPG)
    Returns dict with: x, u_ref, y_ma, e_clean, debug fields
    """
    x = np.asarray(ppg, float).ravel()

    ce = ceemd_reference(
        x, fs,
        pairs=ce_pairs, noise_ratio=ce_noise_ratio,
        max_imfs=ce_max_imfs, max_sift=ce_max_sift, sd_thresh=ce_sd,
        protect_hr=True, protect_bw=protect_bw, protect_harmonics=protect_harm,
        low_motion_hz=low_motion_hz, high_motion_hz=high_motion_hz
    )
    u_ref = ce['u_ref']

    y_ma, e_clean, W = nlms_anc(x, u_ref, L=lms_L, mu=lms_mu, leak=lms_leak)

    return dict(
        x=x, u_ref=u_ref, y_ma=y_ma, e_clean=e_clean, W=W,
        ce=ce
    )

# =========================================================
# 5) Dash/Plotly: single shared axis figure (raw / MA / clean)
# =========================================================
def build_cemd_lms_figure(ppg, fs, out_dict, height=420, title="CE(M)D + LMS ANC (shared axis)"):
    """
    Build a Plotly figure for Dash that overlays:
        - Raw PPG x(n)
        - Motion estimate y(n) (ANC output)
        - Clean PPG e(n) = x - y
    All on a single shared y-axis & time x-axis.
    """
    x = out_dict["x"]
    y_ma = out_dict["y_ma"]
    e = out_dict["e_clean"]
    t = np.arange(len(x)) / float(fs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=x, name="PPG raw", line=dict(color="purple", width=1.2)))
    fig.add_trace(go.Scatter(x=t, y=y_ma, name="MA estimate (ANC)", line=dict(color="orange", width=1)))
    fig.add_trace(go.Scatter(x=t, y=e, name="PPG clean", line=dict(color="green", width=1.4)))

    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=50, r=30, t=50, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    fig.update_xaxes(title="Time (s)", showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(title="Amplitude", showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    return fig


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


