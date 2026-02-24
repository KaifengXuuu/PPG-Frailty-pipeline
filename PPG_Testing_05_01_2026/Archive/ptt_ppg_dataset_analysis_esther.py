# %%
import os
import numpy as np
import pandas as pd
import scipy.signal as sig
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt

def estimate_spo2_hrv_hr(red_mat, ir_mat, fs):
    # Filters
    hp_sos = sig.cheby2(N=4, rs=40, Wn=0.5, btype='highpass', fs=fs, output='sos')
    lp_sos = sig.cheby2(N=4, rs=40, Wn=10, btype='lowpass', fs=fs, output='sos')
    bp_sos = sig.cheby2(N=4, rs=40, Wn=[0.5, 10], btype='bandpass', fs=fs, output='sos')

    # Use middle third
    start = len(red_mat) // 3
    end = 2 * len(red_mat) // 3
    red_mat = red_mat[start:end]
    ir_mat = ir_mat[start:end]

    # Filter
    red_filt = sig.sosfiltfilt(hp_sos, red_mat)
    red_filt = sig.sosfiltfilt(lp_sos, red_filt)
    ir_filt = sig.sosfiltfilt(hp_sos, ir_mat)
    ir_filt = sig.sosfiltfilt(lp_sos, ir_filt)

    window_size = fs * 5
    num_windows = len(red_filt) // window_size

    spo2_list = []
    hrv_list = []
    hr_list = []

    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size

        red_win = red_filt[start_idx:end_idx]
        ir_win = ir_filt[start_idx:end_idx]

        red_ac = sig.sosfiltfilt(bp_sos, red_win)
        ir_ac = sig.sosfiltfilt(bp_sos, ir_win)
        red_dc = sig.sosfiltfilt(lp_sos, red_win)
        ir_dc = sig.sosfiltfilt(lp_sos, ir_win)

        red_ac = savgol_filter(red_ac, 51, 3)
        ir_ac = savgol_filter(ir_ac, 51, 3)

        ac_red = np.ptp(red_ac)
        ac_ir = np.ptp(ir_ac)
        dc_red = np.mean(red_dc)
        dc_ir = np.mean(ir_dc)

        R = (ac_red / dc_red) / (ac_ir / dc_ir)
        spo2 = np.clip(104 - 17 * R, 85, 100)
        spo2_list.append(spo2)

        # HR and HRV from IR peaks
        peaks, _ = find_peaks(ir_ac, distance=fs * 0.4)
        if len(peaks) > 1:
            rr = np.diff(peaks) / fs
            hrv = np.std(rr) * 1000
            hr = 60 / np.mean(rr)
        else:
            hrv = np.nan
            hr = np.nan

        hrv_list.append(hrv)
        hr_list.append(hr)

    return (
        np.nanmean(spo2_list),
        spo2_list,
        np.nanmean(hrv_list),
        hrv_list,
        np.nanmean(hr_list),
        hr_list,
        red_mat,
        ir_mat
    )


# --- Batch processing setup ---
folder_path = "C:/Users/AHMS/Desktop/PPG/20250627TestData/Base"  # 🔁 Replace with your actual folder

fs = 500  # Hz
plot_folder = os.path.join(folder_path, "plots")
os.makedirs(plot_folder, exist_ok=True)

csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

for filename in csv_files:
    file_path = os.path.join(folder_path, filename)
    print(f"\nProcessing: {filename}")

    # Load file
    df = pd.read_csv(file_path)
    if 'pleth_1' not in df.columns or 'pleth_2' not in df.columns:
        print(f"⚠️ Skipped {filename} — missing required columns.")
        continue

    red_mat = df['pleth_1'].to_numpy()
    ir_mat = df['pleth_2'].to_numpy()

    # Estimate metrics
    spo2_avg, spo2_list, hrv_avg, hrv_list, hr_avg, hr_list, red_used, ir_used = estimate_spo2_hrv_hr(red_mat, ir_mat, fs)

    print(f"SpO₂: {spo2_avg:.1f}%, HRV: {hrv_avg:.1f} ms, HR: {hr_avg:.1f} bpm")

    # Time vectors
    t = np.arange(len(red_used)) / fs
    window_times = np.arange(len(spo2_list)) * 5 + (len(red_mat) // 3) / fs

    # Plot
    plt.figure(figsize=(16, 10))

    plt.subplot(4, 1, 1)
    plt.plot(t, red_used, label='Red', color='red', alpha=0.6)
    plt.plot(t, ir_used, label='IR', color='purple', alpha=0.6)
    plt.title(f'{filename} - Middle Third of Raw PPG Signals')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 2)
    plt.plot(window_times, spo2_list, 'o-', color='green')
    plt.title('Estimated SpO₂ Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('SpO₂ (%)')
    plt.ylim(80, 102)
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.plot(window_times, hrv_list, 'o-', color='blue')
    plt.title('Estimated HRV (SDNN) Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('HRV (ms)')
    plt.grid()

    plt.subplot(4, 1, 4)
    plt.plot(window_times, hr_list, 'o-', color='orange')
    plt.title('Estimated Heart Rate Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Heart Rate (bpm)')
    plt.grid()

    plt.tight_layout()
    
    # Save plot
    plot_filename = os.path.join(plot_folder, f"{os.path.splitext(filename)[0]}_plot.png")
    plt.savefig(plot_filename)
    plt.close()

print("\n✅ Done processing all files.")
