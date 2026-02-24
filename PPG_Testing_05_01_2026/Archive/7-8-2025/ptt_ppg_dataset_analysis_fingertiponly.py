import os
import numpy as np
import pandas as pd
import scipy.signal as sig
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt

def estimate_spo2_hrv_hr(red_mat, ir_mat, fs):
    """
    Estimate SpO2, HRV (SDNN), and HR from red and IR PPG signals.

    Args:
        red_mat (np.array): Red PPG signal array.
        ir_mat (np.array): IR PPG signal array.
        fs (int): Sampling frequency in Hz.

    Returns:
        spo2_list (list): Estimated SpO2 values per window.
        hrv_list (list): HRV (SDNN in ms) per window.
        hr_list (list): Heart rate (bpm) per window.
        quality_flags (list): Boolean list indicating quality of HR window.
        red_filt (np.array): Filtered red signal.
        ir_filt (np.array): Filtered IR signal.
    """
    # Bandpass filter design parameters
    hp_sos = sig.cheby2(N=4, rs=40, Wn=0.5, btype='highpass', fs=fs, output='sos')
    lp_sos = sig.cheby2(N=4, rs=40, Wn=10, btype='lowpass', fs=fs, output='sos')
    bp_sos = sig.cheby2(N=4, rs=40, Wn=[0.5, 10], btype='bandpass', fs=fs, output='sos')

    # Filter raw signals - remove noise outside HR range
    red_filt = sig.sosfiltfilt(hp_sos, red_mat)
    red_filt = sig.sosfiltfilt(lp_sos, red_filt)
    ir_filt = sig.sosfiltfilt(hp_sos, ir_mat)
    ir_filt = sig.sosfiltfilt(lp_sos, ir_filt)

    window_size = fs * 5  # 5-second windows
    num_windows = len(red_filt) // window_size

    spo2_list = []
    hrv_list = []
    hr_list = []
    quality_flags = []

    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size

        red_win = red_filt[start_idx:end_idx]
        ir_win = ir_filt[start_idx:end_idx]

        # AC component by bandpass filtering
        red_ac = sig.sosfiltfilt(bp_sos, red_win)
        ir_ac = sig.sosfiltfilt(bp_sos, ir_win)

        # DC component by lowpass filtering
        red_dc = sig.sosfiltfilt(lp_sos, red_win)
        ir_dc = sig.sosfiltfilt(lp_sos, ir_win)

        # Smooth AC with Savitzky-Golay filter
        red_ac_smooth = savgol_filter(red_ac, 51, 3)
        ir_ac_smooth = savgol_filter(ir_ac, 51, 3)

        # Compute peak-to-peak amplitude (AC) and mean (DC)
        ac_red = np.ptp(red_ac_smooth)
        ac_ir = np.ptp(ir_ac_smooth)
        dc_red = np.mean(red_dc)
        dc_ir = np.mean(ir_dc)

        # Calculate R ratio and estimate SpO2
        R = (ac_red / dc_red) / (ac_ir / dc_ir) if dc_red != 0 and dc_ir != 0 else 0
        spo2 = np.clip(104 - 17 * R, 85, 100)
        spo2_list.append(spo2)

        # Detect peaks in IR AC to estimate HR and HRV
        peaks, _ = find_peaks(ir_ac_smooth, distance=fs * 0.4)  # minimum 0.4s apart (~150 bpm max)

        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / fs  # seconds between beats
            hrv = np.std(rr_intervals) * 1000  # SDNN in milliseconds
            hr = 60 / np.mean(rr_intervals)    # bpm

            # Quality flag: HR within physiological plausible range
            quality = (40 <= hr <= 180)
        else:
            hrv = np.nan
            hr = np.nan
            quality = False

        hrv_list.append(hrv)
        hr_list.append(hr)
        quality_flags.append(quality)

    return spo2_list, hrv_list, hr_list, quality_flags, red_filt, ir_filt

# --- Config ---
use_single_file = False
single_file_path = "C:/Users/AHMS/Nextcloud/PPG/Test Data/7-8-2025/base.csv"
folder_path = "C:/Users/AHMS/Nextcloud/PPG/Test Data/7-8-2025"
fs = 400  # Sampling frequency in Hz

if use_single_file:
    files = [single_file_path]
else:
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]

plot_folder = os.path.join(folder_path, "plots")
os.makedirs(plot_folder, exist_ok=True)

summary_rows = []

for file_path in files:
    filename = os.path.basename(file_path)
    print(f"\n📂 Processing: {filename}")

    try:
        df = pd.read_csv(file_path)
        # Check fingertip columns
        if 'RedFinger' not in df.columns or 'IrFinger' not in df.columns:
            print(f"  ⚠️ Finger data missing in {filename}. Skipping...")
            continue

        red = df['RedFinger'].to_numpy()
        ir = df['IrFinger'].to_numpy()

        # Estimate metrics
        spo2_list, hrv_list, hr_list, quality_flags, red_filt, ir_filt = estimate_spo2_hrv_hr(red, ir, fs)

        # Filter HR for valid windows only
        hr_array = np.array(hr_list)
        quality_array = np.array(quality_flags)

        valid_hr = hr_array[quality_array]
        valid_spo2 = np.array(spo2_list)[quality_array]
        valid_hrv = np.array(hrv_list)[quality_array]

        # Compute mean and median for valid windows only
        if len(valid_hr) > 0:
            hr_mean = np.mean(valid_hr)
            hr_median = np.median(valid_hr)
            spo2_mean = np.mean(valid_spo2)
            hrv_mean = np.mean(valid_hrv)
        else:
            hr_mean = hr_median = spo2_mean = hrv_mean = np.nan

        # Time axis for raw signals
        t = np.arange(len(red_filt)) / fs
        # Time axis for windows (center of each 5-sec window)
        window_times = (np.arange(len(spo2_list)) + 0.5) * 5

        # --- Plotting ---
        plt.figure(figsize=(18, 14))

        # 1) Raw PPG signals filtered
        plt.subplot(4, 2, 1)
        plt.plot(t, red_filt, color='red', alpha=0.7, label='RedFinger')
        plt.plot(t, ir_filt, color='purple', alpha=0.7, label='IrFinger')
        plt.title(f"{filename} - Filtered Finger PPG Signals")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)

        # 2) SpO2 over time
        plt.subplot(4, 2, 2)
        plt.plot(window_times, spo2_list, 'o-', color='green')
        plt.title("Estimated SpO₂ Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("SpO₂ (%)")
        plt.ylim(80, 102)
        plt.grid(True)

        # 3) HRV over time
        plt.subplot(4, 2, 3)
        plt.plot(window_times, hrv_list, 'o-', color='blue')
        plt.title("Estimated HRV (SDNN) Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("HRV (ms)")
        plt.grid(True)

        # 4) HR over time (all windows)
        plt.subplot(4, 2, 4)
        plt.plot(window_times, hr_list, 'o-', color='orange', label='All Windows')
        plt.plot(window_times[quality_array], hr_array[quality_array], 'go', label='Valid Windows')
        plt.title("Estimated Heart Rate Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("HR (bpm)")
        plt.legend()
        plt.grid(True)

        # 5) HR histogram with mean and median lines
        plt.subplot(4, 2, 5)
        plt.hist(valid_hr, bins=20, color='orange', alpha=0.7)
        plt.axvline(hr_mean, color='red', linestyle='--', label=f"Mean HR: {hr_mean:.2f} bpm")
        plt.axvline(hr_median, color='blue', linestyle='--', label=f"Median HR: {hr_median:.2f} bpm")
        plt.title("HR Distribution (Valid Windows)")
        plt.xlabel("HR (bpm)")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)

        # 6) Quality flags over windows
        plt.subplot(4, 2, 6)
        plt.plot(window_times, quality_array.astype(int), 'ko-', alpha=0.7)
        plt.title("Quality Flags Per 5s Window (1=Valid, 0=Invalid)")
        plt.xlabel("Time (s)")
        plt.ylabel("Quality Flag")
        plt.ylim(-0.2, 1.2)
        plt.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(plot_folder, filename.replace(".csv", "_plot.png"))
        plt.savefig(plot_path)
        plt.close()
        print(f"  ✅ Plot saved to {plot_path}")

        # Append summary stats
        summary_rows.append({
            'Filename': filename,
            'Mean HR (bpm)': hr_mean,
            'Median HR (bpm)': hr_median,
            'Mean SpO2 (%)': spo2_mean,
            'Mean HRV (ms)': hrv_mean,
            'Valid Windows': len(valid_hr),
            'Total Windows': len(hr_list),
            'Percent Valid Windows': 100 * len(valid_hr) / len(hr_list) if len(hr_list) > 0 else np.nan
        })

    except Exception as e:
        print(f"  ❌ Error processing {filename}: {e}")

# Save summary CSV
summary_df = pd.DataFrame(summary_rows)
summary_csv_path = os.path.join(folder_path, "PPG_Processing_Summary.csv")
summary_df.to_csv(summary_csv_path, index=False)
print(f"\n📊 Summary CSV saved to {summary_csv_path}")
