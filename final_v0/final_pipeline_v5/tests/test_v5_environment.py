from __future__ import annotations

from pathlib import Path
import tomllib

import torch

from ppg_frailty.v5 import environment
from ppg_frailty.v5.environment import check_environment, load_environment_lock


ROOT = Path(__file__).resolve().parents[1]


def test_finalcase_environment_lock_declares_numeric_contract() -> None:
    lock = load_environment_lock(ROOT / "requirements/environment-finalcase-lock.yaml")
    assert lock["numeric_equivalence"] == {
        "atol": 1.0e-6,
        "rtol": 0.0,
        "require_equal_row_identity": True,
        "require_equal_split_assignment": True,
    }
    assert lock["runtime"]["packages"]["torch"] == "2.9.1+cu126"
    assert lock["runtime"]["packages"]["nvidia-cublas-cu12"] == "12.6.4.1"
    assert lock["runtime"]["packages"]["nvidia-cudnn-cu12"] == "9.10.2.21"
    assert lock["runtime"]["packages"]["triton"] == "3.5.1"
    for name, version in {
        "python-dateutil": "2.9.0.post0",
        "pytz": "2025.2",
        "tzdata": "2025.3",
        "six": "1.17.0",
        "threadpoolctl": "3.6.0",
        "typing_extensions": "4.15.0",
        "packaging": "25.0",
        "setuptools": "80.9.0",
    }.items():
        assert str(lock["runtime"]["packages"][name]) == version
    assert lock["accelerator"]["required_for_numeric_equivalence"] is True
    assert lock["accelerator"]["torch_cuda"] == "12.6"
    assert lock["accelerator"]["cudnn"] == 91002
    assert lock["accelerator"]["driver_version"] == "560.81"


def test_environment_check_is_structured_without_requiring_a_gpu() -> None:
    result = check_environment(
        lock_path=ROOT / "requirements/environment-finalcase-lock.yaml",
        device="cpu",
    )
    assert result.status in {"passed", "failed"}
    payload = result.to_dict()
    assert payload["lock_id"] == "finalcase_v2_numeric_20260824"
    assert isinstance(payload["mismatches"], list)


def test_environment_observes_the_requested_cuda_index(monkeypatch) -> None:
    requested: list[int] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda index: requested.append(index) or "requested GPU",
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda index: requested.append(index) or (8, 6),
    )
    monkeypatch.setattr(environment, "_nvidia_driver_version", lambda: "driver")

    observed = environment.observe_environment(accelerator_index=2)

    assert requested == [2, 2]
    assert observed["accelerator"]["selected_device_index"] == 2
    assert observed["accelerator"]["selected_device_available"] is True


def test_request_lock_dependency_is_a_core_runtime_pin() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "filelock==3.20.0" in project["project"]["dependencies"]
