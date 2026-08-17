#!/usr/bin/env python3
"""Tiny non-scientific sklearn/torch ONNX-to-ORT compatibility probe."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import tempfile

import numpy as np

from ppg_frailty.contracts import to_strict_json_value
from ppg_frailty.onnx_winner import (
    _ort_readback,
    _parity_metrics,
    _sklearn_export,
    _torch_export,
)
from ppg_frailty.training import (
    ONNX_WINNER_ABSOLUTE_TOLERANCE,
    ONNX_WINNER_OPSET_VERSION,
    ONNX_WINNER_RELATIVE_TOLERANCE,
)
from ppg_frailty.provenance import sha256_file, stable_payload_sha256


def _versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in (
            "numpy", "onnx", "onnxruntime", "scikit-learn",
            "skl2onnx", "torch",
        )
    }


def _sklearn_probe(root: Path) -> dict[str, object]:
    from sklearn.linear_model import LogisticRegression

    x = np.asarray(
        (
            (-2.0, -1.0), (-1.5, -0.5), (-1.0, -1.5),
            (0.0, 2.0), (0.5, 1.5), (1.0, 2.5),
            (2.0, -1.0), (2.5, -0.5), (1.5, -1.5),
        ),
        dtype=np.float32,
    )
    model = LogisticRegression(
        random_state=42,
        solver="lbfgs",
        max_iter=500,
    )
    # Fixed pre-fitted converter fixture: no optimiser/training is executed.
    model.classes_ = np.asarray((0, 1, 2), dtype=np.int64)
    model.coef_ = np.asarray(
        ((0.5, -0.25), (-0.4, 0.75), (0.1, 0.2)),
        dtype=np.float64,
    )
    model.intercept_ = np.asarray((0.1, -0.2, 0.3), dtype=np.float64)
    model.n_features_in_ = 2
    model.n_iter_ = np.asarray((0,), dtype=np.int32)
    model_path = root / "tiny_logistic.onnx"
    backend, converter_version = _sklearn_export(
        model,
        {"x": x},
        ("x",),
        model_path,
        opset_version=ONNX_WINNER_OPSET_VERSION,
        class_order=(0, 1, 2),
    )
    reference = model.predict_proba(x).astype(np.float64)
    candidate, output_name = _ort_readback(
        model_path, {"x": x}, class_count=3
    )
    metrics = _parity_metrics(
        reference,
        candidate,
        absolute_tolerance=ONNX_WINNER_ABSOLUTE_TOLERANCE,
        relative_tolerance=ONNX_WINNER_RELATIVE_TOLERANCE,
    )
    if metrics["parity_passed"] is not True:
        raise RuntimeError("tiny sklearn ONNX parity failed")
    return {
        "status": "passed",
        "converter_backend": backend,
        "converter_version": converter_version,
        "output_name": output_name,
        "onnx_sha256": sha256_file(model_path),
        "onnx_bytes": model_path.stat().st_size,
        "parity": metrics,
    }


def _torch_probe(root: Path) -> dict[str, object]:
    import torch

    class TinyLogits(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 3)
            with torch.no_grad():
                self.linear.weight.copy_(
                    torch.tensor(
                        ((0.5, -0.25), (-0.4, 0.75), (0.1, 0.2)),
                        dtype=torch.float32,
                    )
                )
                self.linear.bias.copy_(
                    torch.tensor((0.1, -0.2, 0.3), dtype=torch.float32)
                )

        def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del mask
            return self.linear(x)

    x = np.asarray(((-1.0, 0.5), (0.25, 1.5), (2.0, -0.5)), dtype=np.float32)
    model = TinyLogits().eval()
    model_path = root / "tiny_torch.onnx"
    backend, converter_version = _torch_export(
        model,
        {"x": x},
        ("x",),
        model_path,
        opset_version=ONNX_WINNER_OPSET_VERSION,
    )
    with torch.no_grad():
        reference = torch.softmax(
            model(torch.as_tensor(x)), dim=-1
        ).cpu().numpy().astype(np.float64)
    candidate, output_name = _ort_readback(
        model_path, {"x": x}, class_count=3
    )
    metrics = _parity_metrics(
        reference,
        candidate,
        absolute_tolerance=ONNX_WINNER_ABSOLUTE_TOLERANCE,
        relative_tolerance=ONNX_WINNER_RELATIVE_TOLERANCE,
    )
    if metrics["parity_passed"] is not True:
        raise RuntimeError("tiny torch ONNX parity failed")
    return {
        "status": "passed",
        "converter_backend": backend,
        "converter_version": converter_version,
        "output_name": output_name,
        "onnx_sha256": sha256_file(model_path),
        "onnx_bytes": model_path.stat().st_size,
        "parity": metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="ppg-v2-onnx-tiny-") as raw:
        root = Path(raw)
        payload = {
            "schema_version": "ppg_frailty.onnx_profile_tiny_smoke.v2",
            "pipeline_generation": "final_pipeline_v2",
            "status": "passed",
            "scope": "synthetic_converter_and_ort_readback_only",
            "versions": _versions(),
            "sklearn_logistic": _sklearn_probe(root),
            "torch_module": _torch_probe(root),
            "training_executed": False,
            "scientific_model_fitted": False,
            "ablation_executed": False,
            "cross_validation_executed": False,
        }
    payload["payload_sha256"] = stable_payload_sha256(payload)
    normalized = to_strict_json_value(payload)
    if arguments.output is not None:
        target = arguments.output.resolve()
        if target.exists():
            raise FileExistsError(f"probe output overwrite forbidden: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                normalized,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(normalized, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
