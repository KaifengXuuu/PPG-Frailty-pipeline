"""Environment-scoped model-input operational measurements for V2 reports."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Mapping

import numpy as np


CPU_BATCH1_WARMUP_RUNS = 10
CPU_BATCH1_MEASURED_RUNS = 100


@dataclass(frozen=True)
class OperationalMetrics:
    """One reproducible conda-ml CPU batch-1 model-input measurement."""

    parameter_count: int
    parameter_count_definition: str
    model_latency_p50_ms: float
    model_latency_p95_ms: float
    preprocessing_latency_p50_ms: float | None
    preprocessing_latency_p95_ms: float | None
    warmup_runs: int
    measured_runs: int
    batch_size: int
    device: str
    preprocessing_included_in_model_latency: bool
    python_executable: str
    python_version: str
    conda_environment: str
    torch_intraop_threads: int | None
    torch_interop_threads: int | None
    native_threadpools: tuple[Mapping[str, Any], ...]
    bundle_bytes: int | None = None
    measurement_scope: str = "conda_ml_cpu_batch1_model_input_to_probability"

    @property
    def inference_cost(self) -> dict[str, float]:
        return {
            "cpu_batch1_model_only_p50_ms": self.model_latency_p50_ms,
            "cpu_batch1_model_only_p95_ms": self.model_latency_p95_ms,
        }


def _conda_ml_identity() -> str:
    name = os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name
    if name != "ml":
        raise RuntimeError("operational measurements require the project conda ml environment")
    return name


def _count_learned_arrays(value: Any, visited: set[int]) -> int:
    identity = id(value)
    if identity in visited:
        return 0
    visited.add(identity)
    if isinstance(value, np.ndarray):
        return int(value.size)
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return int(value.numel())
    except ImportError:  # pragma: no cover
        pass
    if isinstance(value, Mapping):
        return sum(_count_learned_arrays(item, visited) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_count_learned_arrays(item, visited) for item in value)
    if hasattr(value, "__getstate__") and type(value).__module__.startswith("sklearn.tree._tree"):
        return _count_learned_arrays(value.__getstate__(), visited)
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        container_names = {
            "pipeline",
            "steps",
            "named_steps",
            "scaler",
            "transformer",
            "classifier",
        }
        return sum(
            _count_learned_arrays(item, visited)
            for name, item in attributes.items()
            if name.endswith("_") or name in container_names
        )
    return 0


def model_parameter_count(model: Any) -> tuple[int, str]:
    """Return trainable tensor parameters or fitted numeric-state elements."""

    try:
        import torch

        if isinstance(model, torch.nn.Module):
            count = sum(
                int(parameter.numel())
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            if count <= 0:
                raise ValueError("torch model has no trainable parameters")
            return count, "torch_trainable_parameter_elements"
    except ImportError:  # pragma: no cover
        pass
    count = _count_learned_arrays(model, set())
    if count <= 0:
        raise ValueError("estimator has no discoverable fitted numeric state")
    return count, "fitted_numeric_state_elements_including_preprocessing"


def measure_bundle_bytes(path: str | Path) -> int:
    """Measure immutable bundle payload bytes after bundle creation."""

    source = Path(path)
    if source.is_symlink():
        raise ValueError("bundle-size measurement rejects symlink roots")
    if source.is_file():
        return int(source.stat().st_size)
    if not source.is_dir():
        raise FileNotFoundError("bundle path does not exist")
    files = tuple(item for item in source.rglob("*") if item.is_file())
    if not files or any(item.is_symlink() for item in files):
        raise ValueError("bundle directory must contain regular non-symlink files")
    return int(sum(item.stat().st_size for item in files))


def _batch1(value: Any) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _batch1(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _batch1(item)
        return
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.shape[0] != 1:
            raise ValueError("every model input array must have batch size one")
        return
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.ndim == 0 or value.shape[0] != 1 or value.device.type != "cpu":
                raise ValueError("every tensor input must be CPU batch size one")
            return
    except ImportError:  # pragma: no cover
        pass


def _invoke_probability(model: Any, model_input: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(model, torch.nn.Module):
            if any(parameter.device.type != "cpu" for parameter in model.parameters()):
                raise ValueError("operational latency requires a CPU model")
            model.eval()
            with torch.inference_mode():
                if isinstance(model_input, Mapping):
                    output = model(**model_input)
                elif isinstance(model_input, (tuple, list)):
                    output = model(*model_input)
                else:
                    output = model(model_input)
                probability = (
                    output
                    if torch.all(output >= 0)
                    and torch.allclose(
                        output.sum(dim=-1),
                        torch.ones_like(output.sum(dim=-1)),
                        rtol=0.0,
                        atol=1e-5,
                    )
                    else torch.softmax(output, dim=-1)
                )
            return probability.detach().cpu().numpy().astype(np.float64)
    except ImportError:  # pragma: no cover
        pass
    if not hasattr(model, "predict_proba"):
        raise TypeError("non-torch operational model must expose predict_proba")
    if isinstance(model_input, Mapping):
        try:
            probability = model.predict_proba(**model_input)
        except TypeError:
            probability = model.predict_proba(model_input["x"])
    else:
        probability = model.predict_proba(model_input)
    return np.asarray(probability, dtype=np.float64)


def _measure(callable_: Callable[[], Any]) -> tuple[float, float]:
    for _ in range(CPU_BATCH1_WARMUP_RUNS):
        callable_()
    elapsed = np.empty(CPU_BATCH1_MEASURED_RUNS, dtype=np.float64)
    for index in range(CPU_BATCH1_MEASURED_RUNS):
        start = time.perf_counter_ns()
        callable_()
        elapsed[index] = (time.perf_counter_ns() - start) / 1_000_000.0
    return float(np.quantile(elapsed, 0.50)), float(np.quantile(elapsed, 0.95))


def measure_cpu_batch1_operational_metrics(
    model: Any,
    model_input: Any,
    *,
    preprocessing: Callable[[Any], Any] | None = None,
    preprocessing_input: Any = None,
    bundle_path: str | Path | None = None,
) -> OperationalMetrics:
    """Measure fixed conda-ml CPU batch-1 probability latency.

    Model-only timing begins at the already-materialised model input. Optional
    preprocessing is timed separately and is never included in model latency.
    """

    environment = _conda_ml_identity()
    _batch1(model_input)
    parameter_count, definition = model_parameter_count(model)
    try:
        from threadpoolctl import threadpool_info, threadpool_limits

        native_context = threadpool_limits(limits=1)
    except ImportError:  # pragma: no cover
        threadpool_info = lambda: []  # type: ignore[assignment]
        native_context = nullcontext()
    try:
        import torch

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        intraop = torch.get_num_threads()
        interop = torch.get_num_interop_threads()
    except ImportError:  # pragma: no cover
        torch = None
        previous_threads = None
        intraop = None
        interop = None
    try:
        with native_context:
            probability = _invoke_probability(model, model_input)
            if (
                probability.ndim != 2
                or probability.shape[0] != 1
                or probability.shape[1] < 2
                or not np.isfinite(probability).all()
                or np.any(probability < 0.0)
                or not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)
            ):
                raise ValueError("model-input boundary must return one valid probability row")
            model_p50, model_p95 = _measure(
                lambda: _invoke_probability(model, model_input)
            )
            native_info = tuple(
                {
                    key: item.get(key)
                    for key in (
                        "user_api",
                        "internal_api",
                        "prefix",
                        "version",
                        "num_threads",
                    )
                }
                for item in threadpool_info()
            )
            if preprocessing is None:
                preprocess_p50 = preprocess_p95 = None
            else:
                if preprocessing_input is None:
                    raise ValueError("preprocessing_input is required when timing preprocessing")
                preprocess_p50, preprocess_p95 = _measure(
                    lambda: preprocessing(preprocessing_input)
                )
    finally:
        if torch is not None and previous_threads is not None:
            torch.set_num_threads(previous_threads)
    return OperationalMetrics(
        parameter_count=parameter_count,
        parameter_count_definition=definition,
        model_latency_p50_ms=model_p50,
        model_latency_p95_ms=model_p95,
        preprocessing_latency_p50_ms=preprocess_p50,
        preprocessing_latency_p95_ms=preprocess_p95,
        warmup_runs=CPU_BATCH1_WARMUP_RUNS,
        measured_runs=CPU_BATCH1_MEASURED_RUNS,
        batch_size=1,
        device="cpu",
        preprocessing_included_in_model_latency=False,
        python_executable=sys.executable,
        python_version=platform.python_version(),
        conda_environment=environment,
        torch_intraop_threads=intraop,
        torch_interop_threads=interop,
        native_threadpools=native_info,
        bundle_bytes=(
            None if bundle_path is None else measure_bundle_bytes(bundle_path)
        ),
    )
