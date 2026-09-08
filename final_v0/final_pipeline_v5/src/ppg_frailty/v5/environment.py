"""Observe the numerical runtime once at the V5 execution boundary."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping

import yaml


DEFAULT_LOCK = Path(__file__).resolve().parents[3] / "requirements/environment-finalcase-lock.yaml"

@dataclass(frozen=True)
class EnvironmentCheck:
    status: str
    lock_id: str
    observed: Mapping[str, Any]
    mismatches: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "ppg_frailty.environment_check.v1",
            "status": self.status,
            "lock_id": self.lock_id,
            "observed": dict(self.observed),
            "mismatches": [dict(row) for row in self.mismatches],
        }

def load_environment_lock(path: str | Path = DEFAULT_LOCK) -> Mapping[str, Any]:
    source = Path(path).resolve()
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"environment lock root must be a mapping: {source}")
    if value.get("schema_version") != "ppg_frailty.environment_lock.v1":
        raise ValueError(f"unsupported environment lock schema: {source}")
    return value

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
    """Return the fields used by the checked-in numerical lock."""

    if package_names is None:
        package_names = tuple(load_environment_lock()["runtime"]["packages"])
    observed: dict[str, Any] = {
        "python": platform.python_version(),
        "packages": {name: _version(name) for name in package_names},
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
        "device_count": count,
        "selected_device_index": accelerator_index,
        "selected_device_available": selected,
        "driver_version": _nvidia_driver_version(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if selected:
        accelerator.update(
            {
                "gpu_name": torch.cuda.get_device_name(accelerator_index),
                "compute_capability": list(torch.cuda.get_device_capability(accelerator_index)),
            }
        )
    observed["accelerator"] = accelerator
    return observed

def check_environment(
    *,
    lock_path: str | Path = DEFAULT_LOCK,
    device: str = "cuda",
    require_determinism_env: bool = False,
) -> EnvironmentCheck:
    """Compare the active runtime with one numerical environment lock."""

    lock = load_environment_lock(lock_path)
    base, separator, suffix = str(device).strip().lower().partition(":")
    if base == "cuda" and separator and not suffix.isdigit():
        raise ValueError("CUDA device must be cuda or cuda:<non-negative-index>")
    wants_cuda = base == "cuda"
    index = int(suffix) if wants_cuda and separator else 0
    required_cuda = bool(lock.get("accelerator", {}).get("required_for_numeric_equivalence"))
    observed = observe_environment(
        include_accelerator=wants_cuda or required_cuda,
        accelerator_index=index,
        package_names=tuple(lock["runtime"]["packages"]),
    )
    mismatches: list[Mapping[str, Any]] = []

    def compare(field: str, expected: Any, actual: Any) -> None:
        if actual != expected:
            mismatches.append({"field": field, "expected": expected, "observed": actual})

    if required_cuda:
        compare("execution.device", "cuda", base)
    compare("runtime.python", str(lock["runtime"]["python"]), observed["python"])
    for name, expected in lock["runtime"]["packages"].items():
        compare(f"runtime.packages.{name}", str(expected), observed["packages"].get(name))
    if wants_cuda or required_cuda:
        actual = observed.get("accelerator", {})
        compare("accelerator.selected_device_available", True, actual.get("selected_device_available"))
        for field in ("gpu_name", "compute_capability", "driver_version", "torch_cuda", "cudnn"):
            compare(f"accelerator.{field}", lock["accelerator"][field], actual.get(field))
    if require_determinism_env:
        for field, expected in lock["determinism"].items():
            compare(f"determinism.{field}", expected, observed["determinism"].get(field))
    return EnvironmentCheck(
        "passed" if not mismatches else "failed",
        str(lock["lock_id"]),
        observed,
        tuple(mismatches),
    )

def require_environment(**kwargs: Any) -> EnvironmentCheck:
    result = check_environment(**kwargs)
    if result.mismatches:
        detail = "; ".join(
            f"{row['field']}: expected={row['expected']!r}, observed={row['observed']!r}" for row in result.mismatches
        )
        raise RuntimeError(f"finalcase environment lock mismatch: {detail}")
    return result

def prepare_deterministic_runtime(lock_path: str | Path | None = DEFAULT_LOCK) -> None:
    """Apply the same backend switches as the V2 trainer without touching RNG state."""

    expected = load_environment_lock(lock_path or DEFAULT_LOCK)["determinism"]
    try:
        import torch
    except ImportError:
        return
    torch.use_deterministic_algorithms(bool(expected["deterministic_algorithms"]))
    torch.backends.cudnn.deterministic = bool(expected["cudnn_deterministic"])
    torch.backends.cudnn.benchmark = bool(expected["cudnn_benchmark"])

def evaluate_environment(
    policy: str,
    *,
    device: str,
    lock_path: str | Path = DEFAULT_LOCK,
) -> EnvironmentCheck:
    """Single entry check used by CLI, sweep, and Dashboard training."""

    kwargs = {"lock_path": lock_path, "device": device, "require_determinism_env": True}
    if policy == "exact":
        prepare_deterministic_runtime(lock_path)
        return require_environment(**kwargs)
    if policy == "record":
        return check_environment(**kwargs)
    raise ValueError("environment policy must be exact or record")


__all__ = [
    "DEFAULT_LOCK",
    "EnvironmentCheck",
    "check_environment",
    "evaluate_environment",
    "load_environment_lock",
    "observe_environment",
    "prepare_deterministic_runtime",
    "require_environment",
]
