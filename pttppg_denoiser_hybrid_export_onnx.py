from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pttppg_denoiser_hybrid_core import export_bundle_to_onnx, load_bundle


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, type=str, help="Path to hybrid_denoiser.pt")
    ap.add_argument("--onnx_path", default="", type=str, help="Optional output .onnx path")
    ap.add_argument("--device", default="cpu", type=str)
    ap.add_argument("--opset", default=17, type=int)
    ap.add_argument("--validate", action="store_true")
    return ap.parse_args()


def validate_export(model_path: Path, onnx_path: Path, device: str) -> float:
    import onnxruntime as ort
    import torch

    bundle = load_bundle(model_path, device=device)
    model = bundle["model"]
    model_cfg = bundle["model_cfg"]
    window_cfg = bundle["window_cfg"]
    win = int(round(window_cfg.fs * window_cfg.win_sec))
    rng = np.random.RandomState(123)
    sample = rng.randn(2, model_cfg.in_channels, win).astype(np.float32)

    with torch.no_grad():
        torch_out = model(torch.from_numpy(sample).to(bundle["device"])).detach().cpu().numpy().astype(np.float32)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {sess.get_inputs()[0].name: sample})[0]
    diff = float(np.max(np.abs(torch_out - ort_out)))
    return diff


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    onnx_path = Path(args.onnx_path) if args.onnx_path else None
    exported = export_bundle_to_onnx(
        model_path=model_path,
        onnx_path=onnx_path,
        device=args.device,
        opset_version=args.opset,
    )
    print(f"Exported ONNX to: {exported}")

    if args.validate:
        diff = validate_export(model_path=model_path, onnx_path=exported, device=args.device)
        print(f"Validation max_abs_diff: {diff:.8f}")


if __name__ == "__main__":
    main()
