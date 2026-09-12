"""Observe the numerical runtime once at the V5 execution boundary."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
from typing import Any


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

def _nvidia_driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    versions = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return next(iter(versions)) if len(versions) == 1 else None

def observe_environment(
    *,
    include_accelerator: bool = True,
    accelerator_index: int = 0,
    package_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Record installed versions and backend state; never approve or reject a runtime."""

    packages = (
        {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions() if dist.metadata["Name"]}
        if package_names is None
        else {name: _version(name) for name in package_names}
    )
    observed: dict[str, Any] = {
        "python": platform.python_version(),
        "packages": dict(sorted(packages.items())),
        "determinism": {"cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
    }
    try:
        import torch
    except ImportError:
        if include_accelerator:
            observed["accelerator"] = {"cuda_available": False}
        return observed
    observed["determinism"].update(
        {
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        }
    )
    if not include_accelerator:
        return observed
    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if available else 0
    selected = available and 0 <= accelerator_index < count
    accelerator: dict[str, Any] = {
        "cuda_available": available,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "device_count": count,
        "selected_device_index": accelerator_index,
        "selected_device_available": selected,
        "driver_version": _nvidia_driver_version(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    # Reading properties initializes CUDA; observation must not preempt the trainer's setup.
    if selected and accelerator["cuda_initialized"]:
        accelerator.update(
            {
                "gpu_name": torch.cuda.get_device_name(accelerator_index),
                "compute_capability": list(torch.cuda.get_device_capability(accelerator_index)),
            }
        )
    observed["accelerator"] = accelerator
    return observed
