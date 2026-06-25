from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pttppg_denoiser_onnx_runtime import (
    BaselineConfig,
    bandpass_filter,
    build_record_from_dataframe,
    lowpass_filter,
    denoise_record_onnx,
    load_onnx_bundle,
    ort_available,
    ort_status_message,
)

DEFAULT_PREVIEW_LOWPASS_HZ = 10.0
DEFAULT_PREVIEW_BAND_LOW_HZ = 0.5
DEFAULT_PREVIEW_BAND_HIGH_HZ = 8.0
DEFAULT_MAX_POINTS = 6000


def list_model_bundles_in_dir(dir_value: str) -> List[Dict[str, str]]:
    if not dir_value:
        return []
    path = Path(dir_value)
    if not path.exists() or not path.is_dir():
        return []
    files = sorted(p.name for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".onnx")
    return [{"label": name, "value": name} for name in files]


def resolve_model_bundle_path(bundle_dir: Optional[str], bundle_file: Optional[str]) -> str:
    if bundle_dir and bundle_file:
        path = Path(bundle_dir) / bundle_file
        if path.exists():
            return str(path)
    return ""


def empty_denoiser_figure(message: str) -> go.Figure:
    fig = go.Figure(layout=go.Layout(title=message, height=420, margin=dict(l=40, r=20, t=40, b=40)))
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14),
    )
    return fig


def preview_filter(sig: np.ndarray, fs: float) -> np.ndarray:
    x = np.asarray(sig, dtype=np.float32)
    x = lowpass_filter(x, fs=fs, cutoff_hz=DEFAULT_PREVIEW_LOWPASS_HZ, order=2)
    x = bandpass_filter(
        x,
        fs=fs,
        lowcut=DEFAULT_PREVIEW_BAND_LOW_HZ,
        highcut=DEFAULT_PREVIEW_BAND_HIGH_HZ,
        order=3,
    )
    return x.astype(np.float32)


@functools.lru_cache(maxsize=8)
def load_denoiser_bundle_cached(model_path: str, device: str = "cpu") -> Dict[str, Any]:
    del device
    return load_onnx_bundle(model_path)


def detector_motion_mask(det_v8: Optional[Dict[str, Any]], expected_len: int, use_detector_gate: bool) -> Optional[np.ndarray]:
    if not use_detector_gate or det_v8 is None or det_v8.get("pred_sample") is None:
        return None
    mask = np.asarray(det_v8["pred_sample"]).astype(bool).reshape(-1)
    if mask.size == expected_len:
        return mask
    if mask.size > expected_len:
        return mask[:expected_len]
    pad = np.zeros(expected_len - mask.size, dtype=bool)
    return np.concatenate([mask, pad], axis=0)


def _prepare_dataframe_for_denoiser(
    df: pd.DataFrame,
    ir_col: Optional[str],
    red_col: Optional[str],
) -> pd.DataFrame:
    out = df.copy()

    if ir_col and ir_col in out.columns:
        out["IR"] = pd.to_numeric(out[ir_col], errors="coerce")
    if red_col and red_col in out.columns:
        out["RED"] = pd.to_numeric(out[red_col], errors="coerce")

    if "IR" not in out.columns and "RED" in out.columns:
        out["IR"] = pd.to_numeric(out["RED"], errors="coerce")
    if "RED" not in out.columns and "IR" in out.columns:
        out["RED"] = pd.to_numeric(out["IR"], errors="coerce")
    return out


def _downsample_for_plot(time: np.ndarray, *signals: np.ndarray, max_points: int = DEFAULT_MAX_POINTS) -> tuple[np.ndarray, ...]:
    n = len(time)
    if n <= max_points:
        return (time, *signals)
    step = max(1, int(np.ceil(float(n) / float(max_points))))
    sl = slice(None, None, step)
    down = [np.asarray(time)[sl]]
    for sig in signals:
        down.append(np.asarray(sig)[sl])
    return tuple(down)


def _mask_to_segments(mask: Optional[np.ndarray], time: np.ndarray) -> List[tuple[float, float]]:
    if mask is None:
        return []
    m = np.asarray(mask).astype(bool).reshape(-1)
    if m.size == 0 or m.size != len(time):
        return []
    segs: List[tuple[float, float]] = []
    start: Optional[int] = None
    for idx, value in enumerate(m):
        if value and start is None:
            start = idx
        elif (not value) and start is not None:
            segs.append((float(time[start]), float(time[idx - 1])))
            start = None
    if start is not None:
        segs.append((float(time[start]), float(time[-1])))
    return segs


def run_denoiser_from_dataframe(
    df: pd.DataFrame,
    name: str,
    bundle_path: str,
    *,
    ir_col: Optional[str] = None,
    red_col: Optional[str] = None,
    motion_mask: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    if not ort_available():
        raise RuntimeError(ort_status_message())
    bundle = load_denoiser_bundle_cached(bundle_path, device=device)
    baseline_cfg: BaselineConfig = bundle["baseline_cfg"]
    fs = float(bundle["window_cfg"].fs)
    df_use = _prepare_dataframe_for_denoiser(df, ir_col=ir_col, red_col=red_col)
    rec = build_record_from_dataframe(df_use, name=name, fs=fs, baseline_cfg=baseline_cfg)
    out = denoise_record_onnx(rec, bundle, motion_mask=motion_mask)
    return {
        "bundle_path": bundle_path,
        "model_cfg": bundle["model_cfg"],
        "window_cfg": bundle["window_cfg"],
        "baseline_cfg": baseline_cfg,
        "time": np.asarray(rec["time"], dtype=np.float32),
        "raw": np.asarray(out["raw"], dtype=np.float32),
        "baseline": np.asarray(out["baseline"], dtype=np.float32),
        "denoised": np.asarray(out["denoised"], dtype=np.float32),
        "motion_mask": None if motion_mask is None else np.asarray(motion_mask).astype(bool),
    }


def build_denoiser_compare_figure(
    result_a: Dict[str, Any],
    *,
    label_a: str,
    result_b: Optional[Dict[str, Any]] = None,
    label_b: Optional[str] = None,
    use_detector_gate: bool = False,
) -> go.Figure:
    time = np.asarray(result_a["time"], dtype=np.float32)
    if result_b is not None and len(result_b["time"]) != len(time):
        raise ValueError("A/B denoiser outputs must share the same time axis length")

    raw_red = preview_filter(result_a["raw"][0], fs=float(result_a["window_cfg"].fs))
    raw_ir = preview_filter(result_a["raw"][1], fs=float(result_a["window_cfg"].fs))
    base_red = preview_filter(result_a["baseline"][0], fs=float(result_a["window_cfg"].fs))
    base_ir = preview_filter(result_a["baseline"][1], fs=float(result_a["window_cfg"].fs))
    den_a_red = preview_filter(result_a["denoised"][0], fs=float(result_a["window_cfg"].fs))
    den_a_ir = preview_filter(result_a["denoised"][1], fs=float(result_a["window_cfg"].fs))

    den_b_red = den_b_ir = None
    if result_b is not None:
        den_b_red = preview_filter(result_b["denoised"][0], fs=float(result_b["window_cfg"].fs))
        den_b_ir = preview_filter(result_b["denoised"][1], fs=float(result_b["window_cfg"].fs))

    ds = _downsample_for_plot(
        time,
        raw_red,
        base_red,
        den_a_red,
        raw_ir,
        base_ir,
        den_a_ir,
        den_b_red if den_b_red is not None else np.zeros_like(time),
        den_b_ir if den_b_ir is not None else np.zeros_like(time),
    )
    t_plot, raw_red, base_red, den_a_red, raw_ir, base_ir, den_a_ir, den_b_red_ds, den_b_ir_ds = ds

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("RED / pleth_1", "IR / pleth_2"),
    )

    fig.add_trace(go.Scatter(x=t_plot, y=raw_red, name="Raw preview-filtered", line=dict(color="#1f77b4", width=1.0)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_plot, y=base_red, name="Linear baseline", line=dict(color="#888888", width=1.0, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_plot, y=den_a_red, name=label_a, line=dict(color="#2ca02c", width=1.2)), row=1, col=1)

    fig.add_trace(go.Scatter(x=t_plot, y=raw_ir, name="Raw preview-filtered", line=dict(color="#1f77b4", width=1.0), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_plot, y=base_ir, name="Linear baseline", line=dict(color="#888888", width=1.0, dash="dot"), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_plot, y=den_a_ir, name=label_a, line=dict(color="#2ca02c", width=1.2), showlegend=False), row=2, col=1)

    if result_b is not None and label_b:
        fig.add_trace(go.Scatter(x=t_plot, y=den_b_red_ds, name=label_b, line=dict(color="#d62728", width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_plot, y=den_b_ir_ds, name=label_b, line=dict(color="#d62728", width=1.2), showlegend=False), row=2, col=1)

    seg_motion = _mask_to_segments(result_a.get("motion_mask"), time)
    if seg_motion:
        for row in (1, 2):
            for x0, x1 in seg_motion:
                fig.add_vrect(
                    x0=x0,
                    x1=x1,
                    fillcolor="rgba(220,0,0,0.10)",
                    line_width=0,
                    layer="below",
                    row=row,
                    col=1,
                )

    gate_note = "detector motion-only gate" if use_detector_gate and seg_motion else "full-signal denoise"
    fig.update_layout(
        title=(
            "Hybrid denoiser compare "
            f"({gate_note}, LP<{DEFAULT_PREVIEW_LOWPASS_HZ:g}Hz, "
            f"BP {DEFAULT_PREVIEW_BAND_LOW_HZ:g}-{DEFAULT_PREVIEW_BAND_HIGH_HZ:g}Hz)"
        ),
        height=760,
        margin=dict(l=40, r=20, t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)
    return fig


def compare_denoisers_from_dataframe(
    df: pd.DataFrame,
    name: str,
    *,
    bundle_path_a: str,
    bundle_path_b: Optional[str] = None,
    ir_col: Optional[str] = None,
    red_col: Optional[str] = None,
    det_v8: Optional[Dict[str, Any]] = None,
    use_detector_gate: bool = False,
    device: str = "cpu",
    label_a: Optional[str] = None,
    label_b: Optional[str] = None,
) -> go.Figure:
    if not ort_available():
        return empty_denoiser_figure(ort_status_message())
    if not bundle_path_a:
        return empty_denoiser_figure("Denoiser A bundle not selected")

    bundle_a = load_denoiser_bundle_cached(bundle_path_a, device=device)
    expected_len = len(df)
    motion_mask = detector_motion_mask(det_v8, expected_len=expected_len, use_detector_gate=use_detector_gate)
    result_a = run_denoiser_from_dataframe(
        df,
        name=name,
        bundle_path=bundle_path_a,
        ir_col=ir_col,
        red_col=red_col,
        motion_mask=motion_mask,
        device=device,
    )

    result_b = None
    if bundle_path_b:
        bundle_b = load_denoiser_bundle_cached(bundle_path_b, device=device)
        if float(bundle_a["window_cfg"].fs) != float(bundle_b["window_cfg"].fs):
            raise ValueError("Denoiser A/B bundles must use the same sampling rate")
        result_b = run_denoiser_from_dataframe(
            df,
            name=name,
            bundle_path=bundle_path_b,
            ir_col=ir_col,
            red_col=red_col,
            motion_mask=motion_mask,
            device=device,
        )

    label_a = label_a or f"Denoiser A ({result_a['model_cfg'].input_mode})"
    if result_b is not None:
        label_b = label_b or f"Denoiser B ({result_b['model_cfg'].input_mode})"

    return build_denoiser_compare_figure(
        result_a,
        label_a=label_a,
        result_b=result_b,
        label_b=label_b,
        use_detector_gate=use_detector_gate,
    )
