"""Display-only spectra of full-rate signals; never used as pipeline inputs."""
from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy import signal


def amplitude_spectra(signals: Mapping[str, np.ndarray], *, fs_hz: float = 400.0) -> dict:
    """Display ``abs(rfft(x)) / N`` without detrending or doubling positive bins.

    This matches the notebook's FFT amplitude convention (a bin-centred sine
    of amplitude A has height A/2), not Welch PSD. Complete signals use the
    full record. Gapped signals use the longest contiguous finite run, with
    its original bounds recorded; gaps are never removed and spliced across,
    padded, or averaged. Equal-length runs select the first. No model input
    is modified, and fewer than two contiguous samples yield no spectrum.
    """
    spectra, metadata = {}, {}
    for name, values in signals.items():
        values = np.asarray(values, dtype=np.float64).ravel()
        finite = np.isfinite(values)
        edges = np.diff(np.r_[False, finite, False].astype(np.int8))
        spans = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
        start, stop = max(spans, key=lambda span: span[1] - span[0], default=(0, 0))
        length = int(stop - start)
        available = length >= 2
        if available:
            spectra[name] = {
                "x": np.fft.rfftfreq(length, d=1.0 / fs_hz).tolist(),
                "y": (np.abs(np.fft.rfft(values[start:stop])) / length).tolist(),
            }
        metadata[name] = {
            "method": "rfft", "scope": "full_record" if finite.all() else "longest_finite_run",
            "fs_hz": fs_hz, "detrend": False, "window": "rectangular",
            "scaling": "abs(rfft)/N", "one_sided_doubling": False,
            "finite_samples": int(finite.sum()), "finite_runs": len(spans),
            "used_samples": length if available else 0,
            "segment_start_sample": int(start), "segment_stop_sample": int(stop),
            "frequency_resolution_hz": fs_hz / length if available else None,
            "status": "available" if available else "insufficient_contiguous_samples",
        }
    return {"fft_traces": spectra, "fft_metadata": metadata}


def power_spectra(signals: Mapping[str, np.ndarray], *, fs_hz: float = 400.0) -> dict:
    """Average Welch windows without joining gaps or removing DC/slow drift.

    Each finite run is analysed independently. Short runs use shorter windows
    on the same FFT grid; metadata records their actual frequency resolution.
    Weighting by the number of Welch windows avoids giving a tiny run the same
    weight as a long recording. This is a plot product, not a model feature.
    """
    spectra, metadata = {}, {}
    for name, values in signals.items():
        values = np.asarray(values, dtype=np.float64).ravel()
        finite = np.isfinite(values)
        edges = np.diff(np.r_[False, finite, False].astype(np.int8))
        spans = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
        nfft = min(len(values), 8192)
        total, windows, used, lengths = None, 0, 0, []
        for start, stop in spans:
            if stop - start < 2:
                continue
            length = min(nfft, stop - start)
            overlap = length // 2
            frequencies, density = signal.welch(
                values[start:stop], fs=fs_hz, window="hann", nperseg=length,
                noverlap=overlap, nfft=nfft, detrend=False, scaling="density")
            count = 1 + (stop - start - length) // (length - overlap)
            total = density * count if total is None else total + density * count
            windows += count
            used += length + (count - 1) * (length - overlap)
            lengths.append(int(length))
        if windows:
            spectra[name] = {"x": frequencies.tolist(), "y": (total / windows).tolist()}
        metadata[name] = {
            "method": "welch", "scope": "full_record", "fs_hz": fs_hz,
            "detrend": False, "scaling": "density", "window": "hann", "nfft": nfft,
            "window_samples_min": min(lengths, default=0),
            "window_samples_max": max(lengths, default=0),
            "finite_samples": int(finite.sum()), "used_samples": int(used),
            "welch_segments": int(windows), "finite_runs": len(spans),
            "status": "available" if windows else "insufficient_contiguous_samples",
        }
    return {"frequency_traces": spectra, "spectrum_metadata": metadata}
