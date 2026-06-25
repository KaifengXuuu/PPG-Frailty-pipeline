from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pttppg_denoiser_hybrid_core import (
    bandpass_filter,
    load_bundle,
    load_single_record,
    denoise_record,
    lowpass_filter,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "denoiser_preview_output"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, type=str)
    ap.add_argument("--csv_path", required=True, type=str)
    ap.add_argument("--start_sec", default=0.0, type=float)
    ap.add_argument("--duration_sec", default=20.0, type=float)
    ap.add_argument("--output_png", default=None, type=str)
    ap.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), type=str)
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


def resolve_output_path(csv_path: Path, output_dir: Path, output_png: str | None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_png:
        out = Path(output_png)
        if out.is_absolute():
            out.parent.mkdir(parents=True, exist_ok=True)
            return out
        if out.suffix.lower() == ".png":
            return output_dir / out.name
        out.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        return out / f"{csv_path.stem}_{timestamp}.png"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return output_dir / f"{csv_path.stem}_{timestamp}.png"


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.model_path, device=args.device)
    window_cfg = bundle["window_cfg"]
    baseline_cfg = bundle["baseline_cfg"]
    csv_path = Path(args.csv_path)
    output_path = resolve_output_path(csv_path, Path(args.output_dir), args.output_png)

    rec = load_single_record(csv_path, fs=window_cfg.fs, baseline_cfg=baseline_cfg)
    out = denoise_record(rec, bundle)

    fs = window_cfg.fs
    n = out["raw"].shape[1]
    start = int(max(0.0, args.start_sec) * fs)
    end = int(min(n, round((args.start_sec + args.duration_sec) * fs)))
    if end <= start:
        raise RuntimeError("Invalid preview range")

    raw_plot = np.stack(
        [
            preview_filter(out["raw"][0], fs, args.preview_lowpass_hz, args.preview_band_low_hz, args.preview_band_high_hz),
            preview_filter(out["raw"][1], fs, args.preview_lowpass_hz, args.preview_band_low_hz, args.preview_band_high_hz),
        ],
        axis=0,
    )
    baseline_plot = np.stack(
        [
            preview_filter(out["baseline"][0], fs, args.preview_lowpass_hz, args.preview_band_low_hz, args.preview_band_high_hz),
            preview_filter(out["baseline"][1], fs, args.preview_lowpass_hz, args.preview_band_low_hz, args.preview_band_high_hz),
        ],
        axis=0,
    )
    denoised_plot = np.stack(
        [
            preview_filter(out["denoised"][0], fs, args.preview_lowpass_hz, args.preview_band_low_hz, args.preview_band_high_hz),
            preview_filter(out["denoised"][1], fs, args.preview_lowpass_hz, args.preview_band_low_hz, args.preview_band_high_hz),
        ],
        axis=0,
    )

    t = np.arange(start, end, dtype=np.float32) / float(fs)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    channel_labels = [("RED / pleth_1", 0), ("IR / pleth_2", 1)]

    for ax, (title, ch) in zip(axes, channel_labels):
        ax.plot(t, raw_plot[ch, start:end], label="Raw preview-filtered", linewidth=1.0, alpha=0.9)
        ax.plot(t, baseline_plot[ch, start:end], label="Linear baseline preview-filtered", linewidth=1.0, alpha=0.9)
        ax.plot(t, denoised_plot[ch, start:end], label="Hybrid denoised preview-filtered", linewidth=1.2, alpha=0.95)
        ax.set_title(title)
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        "Hybrid Denoiser Preview: "
        f"{csv_path.name} "
        f"(LP<{args.preview_lowpass_hz:g}Hz, BP {args.preview_band_low_hz:g}-{args.preview_band_high_hz:g}Hz)"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved preview plot to: {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
