from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from ppg_frailty.v5 import checkpoints
from ppg_frailty.training import bundle as training_bundle


def test_checkpoint_side_effect_restores_python_and_numpy_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    random.seed(731)
    np.random.seed(731)
    torch.manual_seed(731)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.random.set_rng_state(torch_state)

    def fake_save_bundle(_model: object, directory: Path, **_kwargs: object) -> None:
        random.random()
        np.random.random()
        torch.rand(())
        directory.mkdir()
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "state_file": "state.pt",
                    "file_hashes": {"state.pt": "0" * 64},
                    "golden_parity_atol": 1e-6,
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(checkpoints, "save_bundle", fake_save_bundle)
    monkeypatch.setattr(checkpoints, "_bundle_metadata", lambda _payload: {})
    payload = checkpoints.FoldCheckpointPayload(
        model=object(),
        model_config={},
        input_spec={},
        golden_inputs={},
        pipeline_config={},
        cell_summary={},
    )

    checkpoints.save_fold_checkpoint(tmp_path / "bundle", payload)

    observed = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert observed == expected


def test_bundle_verification_ignores_request_env_and_does_not_touch_rng(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "PPG_FRAILTY_V5_TRAINING_REQUEST_BINDING",
        "not part of the model bundle contract",
    )
    random.seed(913)
    np.random.seed(913)
    torch.manual_seed(913)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.random.set_rng_state(torch_state)

    with pytest.raises(FileNotFoundError):
        training_bundle.verify_bundle(tmp_path / "missing", load_model=False)

    assert (random.random(), float(np.random.random()), float(torch.rand(()))) == expected
