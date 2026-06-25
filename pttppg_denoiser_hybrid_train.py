from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from pttppg_denoiser_hybrid_core import (
    BaselineConfig,
    INPUT_MODE_RAW_IMU,
    INPUT_MODE_RAW_IMU_BASELINE,
    LossConfig,
    ModelConfig,
    PriorConfig,
    WindowConfig,
    export_bundle_to_onnx,
    load_physionet_csv,
    save_bundle,
    train_hybrid_model,
)


def parse_lags(text: str) -> Tuple[int, ...]:
    vals = [v.strip() for v in text.split(",") if v.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("lags list cannot be empty")
    return tuple(int(v) for v in vals)


def split_subjects(subjects: Sequence[str], keep_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    subjects = sorted(subjects)
    if len(subjects) < 2:
        return list(subjects), []
    rng = np.random.RandomState(seed)
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    n_keep = int(round(len(shuffled) * keep_ratio))
    n_keep = max(1, min(len(shuffled) - 1, n_keep))
    return sorted(shuffled[:n_keep]), sorted(shuffled[n_keep:])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, type=str)
    ap.add_argument("--outdir", default="results_hybrid_denoiser", type=str)
    ap.add_argument("--fs", default=500.0, type=float)
    ap.add_argument("--win", default=6.0, type=float)
    ap.add_argument("--hop", default=1.0, type=float)
    ap.add_argument("--train_size", default=0.8, type=float)
    ap.add_argument("--val_ratio", default=0.15, type=float)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--epochs", default=12, type=int)
    ap.add_argument("--batch_size", default=16, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--patience", default=4, type=int)
    ap.add_argument("--ridge_alpha", default=8.0, type=float)
    ap.add_argument("--lags_ms", default="-200,-120,-80,-40,0,40,80,120,200", type=parse_lags)
    ap.add_argument("--base_channels", default=32, type=int)
    ap.add_argument(
        "--input_mode",
        default=INPUT_MODE_RAW_IMU_BASELINE,
        choices=(INPUT_MODE_RAW_IMU, INPUT_MODE_RAW_IMU_BASELINE),
        type=str,
    )
    ap.add_argument("--target_lowpass_hz", default=6.0, type=float)
    ap.add_argument("--target_band_low_hz", default=0.5, type=float)
    ap.add_argument("--target_band_high_hz", default=8.0, type=float)
    ap.add_argument("--template_len", default=256, type=int)
    ap.add_argument("--library_bins", default=5, type=int)
    ap.add_argument("--min_ibi_sec", default=0.35, type=float)
    ap.add_argument("--max_ibi_sec", default=1.60, type=float)
    ap.add_argument("--lam_artifact", default=1.0, type=float)
    ap.add_argument("--lam_clean", default=0.35, type=float)
    ap.add_argument("--lam_sit", default=0.75, type=float)
    ap.add_argument("--lam_teacher", default=None, type=float, help=argparse.SUPPRESS)
    ap.add_argument("--lam_peak", default=0.2, type=float)
    ap.add_argument("--lam_decorr", default=0.12, type=float)
    ap.add_argument("--lam_slope", default=0.18, type=float)
    ap.add_argument("--lam_anchor", default=0.12, type=float)
    ap.add_argument("--device", default="cpu", type=str)
    ap.add_argument("--max_subjects", default=None, type=int)
    ap.add_argument("--max_windows_per_record", default=None, type=int)
    ap.add_argument("--export_onnx", action="store_true")
    ap.add_argument("--onnx_opset", default=17, type=int)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    window_cfg = WindowConfig(fs=args.fs, win_sec=args.win, hop_sec=args.hop)
    baseline_cfg = BaselineConfig(ridge_alpha=args.ridge_alpha, lags_ms=args.lags_ms)
    model_cfg = ModelConfig(
        input_mode=args.input_mode,
        in_channels=11 if args.input_mode == INPUT_MODE_RAW_IMU else 15,
        base_channels=args.base_channels,
    )
    prior_cfg = PriorConfig(
        target_lowpass_hz=args.target_lowpass_hz,
        target_band_low=args.target_band_low_hz,
        target_band_high=args.target_band_high_hz,
        template_len=args.template_len,
        library_bins=args.library_bins,
        min_ibi_sec=args.min_ibi_sec,
        max_ibi_sec=args.max_ibi_sec,
    )
    lam_artifact = args.lam_artifact if args.lam_teacher is None else args.lam_teacher
    loss_cfg = LossConfig(
        artifact=lam_artifact,
        clean=args.lam_clean,
        sit=args.lam_sit,
        peak=args.lam_peak,
        decorr=args.lam_decorr,
        slope=args.lam_slope,
        anchor=args.lam_anchor,
    )

    sub2recs = load_physionet_csv(args.data_root, fs=args.fs, baseline_cfg=baseline_cfg)
    subjects = sorted(sub2recs.keys())
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]
        sub2recs = {sid: sub2recs[sid] for sid in subjects}

    if len(subjects) < 3:
        raise RuntimeError("Need at least 3 subjects for train/val/holdout split")

    trainval_subjects, holdout_subjects = split_subjects(subjects, keep_ratio=args.train_size, seed=args.seed)
    train_subjects, val_subjects = split_subjects(trainval_subjects, keep_ratio=1.0 - args.val_ratio, seed=args.seed + 1)
    if not val_subjects:
        train_subjects, val_subjects = train_subjects[:-1], train_subjects[-1:]

    train_records = [rec for sid in train_subjects for rec in sub2recs[sid]]
    val_records = [rec for sid in val_subjects for rec in sub2recs[sid]]

    model, beat_library, train_info = train_hybrid_model(
        train_records=train_records,
        val_records=val_records,
        window_cfg=window_cfg,
        baseline_cfg=baseline_cfg,
        model_cfg=model_cfg,
        prior_cfg=prior_cfg,
        loss_cfg=loss_cfg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        max_windows_per_record=args.max_windows_per_record,
    )

    splits = {
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "holdout_subjects": holdout_subjects,
        "seed": args.seed,
        "train_size": args.train_size,
        "val_ratio": args.val_ratio,
    }
    model_path = save_bundle(
        outdir=args.outdir,
        model=model,
        beat_library=beat_library,
        window_cfg=window_cfg,
        baseline_cfg=baseline_cfg,
        model_cfg=model_cfg,
        prior_cfg=prior_cfg,
        loss_cfg=loss_cfg,
        train_info=train_info,
        splits=splits,
    )

    elapsed = time.time() - t0
    print(f"Saved bundle to: {model_path}")
    if args.export_onnx:
        onnx_path = export_bundle_to_onnx(
            model_path=model_path,
            device=args.device,
            opset_version=args.onnx_opset,
        )
        print(f"Exported ONNX to: {onnx_path}")
    print(f"Best val total: {train_info['best_val_total']:.4f} at epoch {train_info['best_epoch']}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
