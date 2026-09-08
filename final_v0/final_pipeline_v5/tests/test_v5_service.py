from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ppg_frailty.v5.service import (
    RefitOptions,
    _adopt_refit_bundle,
    _run_refits,
    preflight_refit_request,
)


HASH = "a" * 64
CASES = (
    {"case_id": "case_a", "config_id": "config_a", "config_sha256": HASH},
    {"case_id": "case_b", "config_id": "config_b", "config_sha256": "b" * 64},
)


def test_disabled_refit_is_a_noop_module(tmp_path: Path) -> None:
    assert preflight_refit_request(
        RefitOptions(),
        pipeline_root=tmp_path,
        cases=CASES,
        repeats=tuple(range(5)),
        folds=tuple(range(5)),
        resume_directory=None,
    ) is None


def test_refit_is_a_plain_optional_stage_for_new_or_resumed_runs(tmp_path: Path) -> None:
    result = preflight_refit_request(
        RefitOptions(enabled=True),
        pipeline_root=tmp_path,
        cases=CASES,
        repeats=(0, 1),
        folds=(0, 1, 2),
        resume_directory=None,
    )

    assert result == {
        "status": "enabled_after_outer_fold_training",
        "default_refit": False,
        "purpose": "configured_full_cohort_refit",
        "case_count": 2,
        "cases": list(CASES),
        "outer_cells": 6,
        "refit_scope": "complete eligible participant cohort for each resolved case",
    }


def test_refit_runs_every_case_and_publishes_relative_bundles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = tmp_path / "run"
    configs = study / "configs"
    configs.mkdir(parents=True)
    cases = []
    for case_id, digest in (("case_a", HASH), ("case_b", "b" * 64)):
        path = configs / f"{case_id}.yaml"
        path.write_text(f"config_id: {case_id}\n", encoding="utf-8")
        cases.append(
            {
                "case_id": case_id,
                "resolved_config_path": f"configs/{case_id}.yaml",
            }
        )
    (study / "study_manifest.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")

    import ppg_frailty.experiment as experiment

    digests = iter((HASH, "b" * 64))
    monkeypatch.setattr(
        experiment,
        "final_refit_preflight",
        lambda *_args, **_kwargs: {
            "config_hash": next(digests),
            "participant_count": 29,
            "participant_ids": [f"P{i:02d}" for i in range(29)],
            "manifest_hash": "m" * 64,
            "fold_registry_hash": "f" * 64,
        },
    )

    def execute(_config: Path, *, bundle_directory: Path, purpose: str) -> Path:
        assert purpose == "configured_full_cohort_refit"
        bundle_directory.mkdir(parents=True)
        (bundle_directory / "manifest.json").write_text("{}\n", encoding="utf-8")
        return bundle_directory

    monkeypatch.setattr(experiment, "execute_final_refit", execute)
    result = _run_refits(study, RefitOptions(enabled=True))

    assert [row["case_id"] for row in result] == ["case_a", "case_b"]
    assert all(row["participant_count"] == 29 for row in result)
    assert result[0]["bundle_manifest"] == "models/case_a/all29_refit/manifest.json"
    assert result[0]["bundle_manifest_sha256"] == hashlib.sha256(b"{}\n").hexdigest()


def test_existing_refit_is_reloaded_and_checked_against_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "all29_refit"
    bundle.mkdir()

    import ppg_frailty.training.bundle as bundle_module

    class Loaded:
        directory = bundle
        manifest = {"metadata": {"config_hash": HASH}}

    checked: list[object] = []
    monkeypatch.setattr(bundle_module, "load_bundle", lambda _path: Loaded())
    monkeypatch.setattr(bundle_module, "assert_golden_parity", checked.append)

    assert _adopt_refit_bundle(bundle, config_hash=HASH) == bundle.resolve()
    assert len(checked) == 1 and isinstance(checked[0], Loaded)
    with pytest.raises(ValueError, match="config differs"):
        _adopt_refit_bundle(bundle, config_hash="0" * 64)
