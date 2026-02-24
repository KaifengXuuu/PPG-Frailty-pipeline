import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# --- Settings ---
FS = 400  # Sampling frequency (Hz)
MIN_BPM = 40
MAX_BPM = 180
OUTPUT_DIR = "output_plots"

# --- Bandpass Filter ---
def bandpass_filter(signal_data, lowcut=0.5, highcut=5, fs=FS, order=2):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return signal.filtfilt(b, a, signal_data)

# --- Heart Rate Estimation ---
def estimate_hr(peaks, fs):
    if len(peaks) < 2:
        return np.nan, []
    rr_intervals = np.diff(peaks) / fs
    hr = 60 / np.mean(rr_intervals)
    return hr, rr_intervals

# --- HRV (SDNN) ---
def calculate_hrv(rr_intervals):
    return np.std(rr_intervals) * 1000 if len(rr_intervals) >= 2 else np.nan

# --- Artifact Rejection ---
def reject_artifacts(peaks, rr_intervals, fs, lower_bpm=MIN_BPM, upper_bpm=MAX_BPM):
    lower_rr = 60 / upper_bpm
    upper_rr = 60 / lower_bpm
    valid_indices = np.where((rr_intervals >= lower_rr) & (rr_intervals <= upper_rr))[0]
    clean_peaks = [peaks[0]]
    for i in valid_indices + 1:
        clean_peaks.append(peaks[i])
    return np.array(clean_peaks)

# --- SpO₂ Estimation ---
def estimate_spo2(ir, red):
    ir_ac = np.ptp(ir)
    ir_dc = np.mean(ir)
    red_ac = np.ptp(red)
    red_dc = np.mean(red)
    ratio = (red_ac / red_dc) / (ir_ac / ir_dc)
    return 110 - 25 * ratio  # Empirical

# --- Main Analysis ---
def analyze_ppg(csv_path, save_plots=True):
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    if not {'IR', 'RED'}.issubset(df.columns):
        print(f"❌ Required columns missing in {csv_path}")
        return None

    ir_raw = df['IR'].values
    red_raw = df['RED'].values
    time = df['Time'].values if 'Time' in df.columns else np.arange(len(ir_raw)) / FS

    ir_filtered = bandpass_filter(ir_raw)
    red_filtered = bandpass_filter(red_raw)

    min_distance = int(FS * 60 / MAX_BPM)
    peaks, _ = signal.find_peaks(ir_filtered, distance=min_distance, prominence=0.5)

    hr, rr_intervals = estimate_hr(peaks, FS)
    if np.isnan(hr):
        print(f"⚠️ Could not estimate HR for {csv_path}")
        return None

    clean_peaks = reject_artifacts(peaks, rr_intervals, FS)
    if len(clean_peaks) < 2:
        print(f"⚠️ Not enough clean peaks after artifact rejection in {csv_path}")
        return None

    clean_rr_intervals = np.diff(clean_peaks) / FS
    hr_clean, _ = estimate_hr(clean_peaks, FS)
    hrv = calculate_hrv(clean_rr_intervals)
    spo2 = estimate_spo2(ir_filtered, red_filtered)

    # Combine PPG
    combined_ppg = (ir_filtered + red_filtered) / 2

    print(f"\n📄 File: {os.path.basename(csv_path)}")
    print(f"📈 Estimated Heart Rate: {hr_clean:.3f} BPM")
    print(f"💓 Heart Rate Variability (SDNN): {hrv:.3f} ms")
    print(f"🩸 Estimated SpO₂: {spo2:.3f}%")

    # Plotting
    plt.figure(figsize=(14, 12))

    # 1. Combined PPG
    plt.subplot(4, 1, 1)
    plt.plot(time, combined_ppg, label='Combined PPG (IR + Red)', color='purple')
    plt.title('Combined PPG Signal (Full)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    # 2. IR signal with peaks
    plt.subplot(4, 1, 2)
    plt.plot(time, ir_raw, label='IR Raw', alpha=0.3, color='gray')
    plt.plot(time, ir_filtered, label='IR Filtered', linewidth=2, color='blue')
    plt.plot(time[peaks], ir_filtered[peaks], 'kx', label='Detected Peaks')
    plt.plot(time[clean_peaks], ir_filtered[clean_peaks], 'ro', label='Cleaned Peaks')
    plt.title('IR Signal with Detected and Cleaned Peaks')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    # 3. Red signal
    plt.subplot(4, 1, 3)
    plt.plot(time, red_raw, label='Red Raw', alpha=0.3, color='lightcoral')
    plt.plot(time, red_filtered, label='Red Filtered', linewidth=2, color='red')
    plt.title('Red Signal')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.legend()

    # 4. Zoomed-in Combined PPG
    plt.subplot(4, 1, 4)
    zoom_start, zoom_end = 125, 150
    zoom_mask = (time >= zoom_start) & (time <= zoom_end)
    plt.plot(time[zoom_mask], combined_ppg[zoom_mask], color='purple', label='Zoomed Combined PPG')
    plt.title(f'Zoomed Combined PPG Signal ({zoom_start}-{zoom_end} sec)')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # Save plot
    if save_plots:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        plot_path = os.path.join(OUTPUT_DIR, f"{base_name}_ppg_analysis.png")
        plt.savefig(plot_path)
        plt.close()
    else:
        plt.show()

    # Return rounded metrics
    return {
        "filename": os.path.basename(csv_path),
        "heart_rate_bpm": round(hr_clean, 3),
        "hrv_sdnn_ms": round(hrv, 3),
        "spo2_percent": round(spo2, 3),
    }

# --- Batch Process ---
def batch_process_folder(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in folder: {folder_path}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_data = []

    print(f"Processing {len(csv_files)} files in: {folder_path}\n")
    for f in csv_files:
        path = os.path.join(folder_path, f)
        metrics = analyze_ppg(path, save_plots=True)
        if metrics:
            summary_data.append(metrics)

    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        csv_path = os.path.join(OUTPUT_DIR, "summary.csv")
        df_summary.to_csv(csv_path, index=False, float_format="%.3f")
        print(f"\n✅ Summary CSV saved to: {csv_path}")
    else:
        print("\n⚠️ No valid data to save in summary.")

    print("\n✅ Batch processing complete.")

# --- Main ---
if __name__ == "__main__":
    folder_with_csv = "C:/Users/CML/Nextcloud/PPG/Test Data/25July25"
    batch_process_folder(folder_with_csv)


