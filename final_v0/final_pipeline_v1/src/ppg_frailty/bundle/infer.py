"""原始 recording 到 bundle 概率 / Raw recording to bundle probabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..training.bundle import LoadedBundle, load_bundle, predict_bundle_raw


def infer_raw_record(
    bundle: LoadedBundle | str | Path,
    record: Any,
) -> dict[str, np.ndarray]:
    """运行已冻结 adapter→model→file mean / Run the bundled raw adapter and model.

    中文：signal/view/window/representation 逻辑必须随 bundle 序列化为 adapter；
    这里不再从 metadata 临时重建第二条 raw-only pipeline。缺 adapter 会关闭失败。
    English: Signal, window and representation logic must be frozen in the bundled
    adapter. This facade does not reconstruct a second raw-only pipeline from metadata.
    """

    loaded = load_bundle(bundle) if isinstance(bundle, (str, Path)) else bundle
    probabilities = np.asarray(predict_bundle_raw(loaded, record), dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise RuntimeError("raw bundle adapter must yield at least one [window,class] probability row")
    file_probability = probabilities.mean(axis=0)
    total = float(file_probability.sum())
    if not np.isfinite(file_probability).all() or not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("raw bundle inference produced an invalid file probability")
    file_probability /= total
    return {"window_probabilities": probabilities, "file_probability": file_probability}


__all__ = ["infer_raw_record"]
