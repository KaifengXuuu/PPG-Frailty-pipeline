from __future__ import annotations

from types import SimpleNamespace

import torch

from ppg_frailty.v5 import environment
from ppg_frailty.training.trainer import configure_torch_determinism


def test_motion_sets_configured_backend_before_loading_weights(monkeypatch) -> None:
    from ppg_frailty import experiment
    from ppg_frailty.training import trainer

    events = []
    monkeypatch.setattr(trainer, "configure_torch_determinism", lambda enabled: events.append(("backend", enabled)))
    monkeypatch.setattr(experiment, "_runtime_imports", lambda: {
        "resolve_reused_motion_detector_config": lambda payload: payload,
        "load_reused_motion_detector": lambda *_args, **_kwargs: events.append(("load", None)),
    })
    sections = {
        "artifact": {"motion_detector_enabled": True, "motion_detector": {"evidence_path": "model.json"}},
        "training": {},
    }
    config = SimpleNamespace(section=sections.__getitem__)
    paths = SimpleNamespace(input_path=lambda value: value)
    # Default and explicit choices must precede all motion prediction on every invocation.
    for setting in ({}, {"deterministic_algorithms": False}, {"deterministic_algorithms": True}):
        sections["training"] = setting
        experiment._load_reused_motion_detector_for_config(config, paths)
        assert events[-2:] == [("backend", setting.get("deterministic_algorithms", True)), ("load", None)]
    sections["artifact"]["motion_detector_enabled"] = False
    events.clear()
    assert experiment._load_reused_motion_detector_for_config(config, paths) is None
    assert not events


def test_environment_observes_the_requested_cuda_index(monkeypatch) -> None:
    requested: list[int] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
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


def test_environment_observation_does_not_initialize_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(environment, "_nvidia_driver_version", lambda: None)

    def unexpected_initialization(*_args):
        raise AssertionError("observation must not initialize CUDA before deterministic setup")

    monkeypatch.setattr(torch.cuda, "get_device_properties", unexpected_initialization)
    observed = environment.observe_environment(package_names=())
    assert observed["accelerator"]["cuda_initialized"] is False
    assert "gpu_name" not in observed["accelerator"]


def test_environment_records_unlisted_versions_without_mutating_runtime(monkeypatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(environment, "_version", lambda name: "future-version" if name == "torch" else None)
    flags = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    observed = environment.observe_environment(
        include_accelerator=False, package_names=("torch", "missing-optional-package"),
    )
    assert observed["packages"] == {"torch": "future-version", "missing-optional-package": None}
    assert "accelerator" not in observed
    assert "CUBLAS_WORKSPACE_CONFIG" not in environment.os.environ
    assert flags == (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )


def test_environment_records_cpu_without_requiring_a_gpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(environment, "_nvidia_driver_version", lambda: None)
    observed = environment.observe_environment(package_names=())
    assert observed["accelerator"]["cuda_available"] is False
    assert observed["accelerator"]["selected_device_available"] is False


def test_shared_backend_configuration_preserves_flags_and_rng() -> None:
    original = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    rng = torch.get_rng_state().clone()
    try:
        for enabled in (True, False):
            configure_torch_determinism(enabled)
            assert torch.are_deterministic_algorithms_enabled() is enabled
            assert torch.backends.cudnn.deterministic is enabled
            assert torch.backends.cudnn.benchmark is not enabled
            assert torch.equal(rng, torch.get_rng_state())
    finally:
        torch.use_deterministic_algorithms(original[0])
        torch.backends.cudnn.deterministic = original[1]
        torch.backends.cudnn.benchmark = original[2]
