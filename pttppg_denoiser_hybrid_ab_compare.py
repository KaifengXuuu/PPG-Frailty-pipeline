from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pttppg_denoiser_hybrid_core import (
    bandpass_filter,
    denoise_record,
    load_bundle,
    load_single_record,
    lowpass_filter,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path_a", required=True, type=str)
    ap.add_argument("--model_path_b", required=True, type=str)
    ap.add_argument("--label_a", default="A: raw + IMU", type=str)
    ap.add_argument("--label_b", default="B: raw + IMU + baseline", type=str)
    ap.add_argument("--csv_path", required=True, type=str)
    ap.add_argument("--start_sec", default=0.0, type=float)
    ap.add_argument("--duration_sec", default=20.0, type=float)
    ap.add_argument("--output_png", default="hybrid_denoiser_ab_compare.png", type=str)
    ap.add_argument("--device", default="cpu", type=str)
    ap.add_argument("--preview_lowpass_hz", default=10.0, type=float)
    ap.add_argument("--preview_band_low_hz", default=0.5, type=float)
    ap.add_argument("--preview_band_high_hz", default=8.0, type=float)
    ap.add_argument("--show", action="store_true")
    return ap.parse_args()


def preview_filter(sig: np.ndarray, fs: float, lowpass_hz: float, band_low_hz: float, band_high_hz: float) -> np.ndarray:
    x = np.asarray(sig, dtype=np.float32)
    x = lowpass_filter(x, fs=fs, cutoff_hz=lowpass_hz, order=2)
    x = bandpass_filter(x, fs=fs, lowcut=band_low_hz, highcut=band_high_hz, order=3)
    return x.astype(np.float32)


def main() -> None:
    args = parse_args()
    bundle_a = load_bundle(args.model_path_a, device=args.device)
    bundle_b = load_bundle(args.model_path_b, device=args.device)

    window_cfg_a = bundle_a["window_cfg"]
    window_cfg_b = bundle_b["window_cfg"]
    baseline_cfg_a = bundle_a["baseline_cfg"]
    baseline_cfg_b = bundle_b["baseline_cfg"]

    if window_cfg_a.fs != window_cfg_b.fs:
        raise RuntimeError("A/B bundles must use the same sampling rate")
    if baseline_cfg_a != baseline_cfg_b:
        raise RuntimeError("A/B bundles must use the same baseline config for fair comparison")

    rec = load_single_record(args.csv_path, fs=window_cfg_a.fs, baseline_cfg=baseline_cfg_a)
    out_a = denoise_record(copy.deepcopy(rec), bundle_a)
    out_b = denoise_record(copy.deepcopy(rec), bundle_b)

    fs = window_cfg_a.fs
    n = out_a["raw"].shape[1]
    start = int(max(0.0, args.start_sec) * fs)
    end = int(min(n, round((args.start_sec + args.duration_sec) * fs)))
    if end <= start:
        raise RuntimeError("Invalid preview range")

    raw_plot = np.stack(
        [
            preview_filter(out_a["raw"][0], fs, args.preview_lowpass_hz, args.preview_band_low_hz, args.preview_band_high_hz),
            preview_filter(out_a["raw"][1], fs, args.preview_lowpass_hz, args.preview_band_low_hz, args.preview_band_high_hz),
        ],
        axis=0,
    )
    baseline_plot = np.stack(
        [
            preview_filter(out_a["baseline"][0], fs, args.preview_lowpass_hz, args.preview_band_low_hz, args.preview_band_high_hz),
            preview_filter(out_a["baseline"][1], fs, args.preview_lowpass_hz, args.preview_band_low_hz, args.preview_band_high_hz),
        ],
        axis=0,
    )
    denoised_a_plot = np.stack(
        [
            preview_filter(out_a["denoised"][0], fs, args.preview_lowpass_hz, args.preview_band_low_hz, args.preview_band_high_hz),
            preview_filter(out_a["denoised"][1], fs, args.preview_lowpass_hz, args.preview_band_low_hz, args.preview_band_high_hz),
        ],
        axis=0,
    )
    denoised_b_plot = np.stack(
        [
            preview_filter(out_b["denoised"][0], fs, args.preview_lowpass_hz, args.preview_band_low_hz, args.preview_band_high_hz),
            preview_filter(out_b["denoised"][1], fs, args.preview_lowpass_hz, args.preview_band_low_hz, args.preview_band_high_hz),
        ],
        axis=0,
    )

    mode_a = bundle_a["model_cfg"].input_mode
    mode_b = bundle_b["model_cfg"].input_mode

    t = np.arange(start, end, dtype=np.float32) / float(fs)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    channel_labels = [("RED / pleth_1", 0), ("IR / pleth_2", 1)]

    for ax, (title, ch) in zip(axes, channel_labels):
        ax.plot(t, raw_plot[ch, start:end], label="Raw preview-filtered", linewidth=1.0, alpha=0.9)
        ax.plot(t, baseline_plot[ch, start:end], label="Linear baseline preview-filtered", linewidth=1.0, alpha=0.9)
        ax.plot(t, denoised_a_plot[ch, start:end], label=f"{args.label_a} ({mode_a})", linewidth=1.2, alpha=0.95)
        ax.plot(t, denoised_b_plot[ch, start:end], label=f"{args.label_b} ({mode_b})", linewidth=1.2, alpha=0.95)
        ax.set_title(title)
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        "Hybrid Denoiser A/B Compare: "
        f"{Path(args.csv_path).name} "
        f"(LP<{args.preview_lowpass_hz:g}Hz, BP {args.preview_band_low_hz:g}-{args.preview_band_high_hz:g}Hz)"
    )
    fig.tight_layout()
    fig.savefig(args.output_png, dpi=150)
    print(f"Saved A/B compare plot to: {args.output_png}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
