"""Every shipped motion reuse mode works without its original training tree."""
from __future__ import annotations

import builtins
import json
import shutil
from pathlib import Path

import pytest

from ppg_frailty.provenance import sha256_file
from ppg_frailty.quality import motion_bundle_adapter as adapter


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / (
    'artifacts/studies/static_line_b_staged_v2/'
    '20260820_225546_staged-static-05-pre-motion-ptt-v1/motion_internal'
)


@pytest.fixture
def relocated_bundle(tmp_path, monkeypatch):
    """Copy only shipped assets, and fail on any read from the old V2 tree."""
    target = tmp_path / 'motion_internal'
    shutil.copytree(BUNDLE, target)
    (tmp_path / 'splits').mkdir()
    shutil.copy2(ROOT / 'splits/sgkf5_seed42_v2.csv', tmp_path / 'splits')
    monkeypatch.setattr(adapter, 'pipeline_resource', lambda path: tmp_path / path)
    path_open, builtin_open = Path.open, builtins.open

    def check(path):
        if isinstance(path, (str, Path)) and 'final_pipeline_v2' in Path(path).parts:
            pytest.fail(f'Relocated motion bundle accessed original V2 file: {path}')

    def guarded_path_open(path, *args, **kwargs):
        check(path)
        return path_open(path, *args, **kwargs)

    def guarded_builtin_open(path, *args, **kwargs):
        check(path)
        return builtin_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded_path_open)
    monkeypatch.setattr(builtins, 'open', guarded_builtin_open)
    evidence_path = target / 'motion_internal_evidence.json'
    return evidence_path, json.loads(evidence_path.read_text())


@pytest.mark.parametrize('fold_index', range(5))
def test_matching_fold_loads_local_weight_and_packaged_split(relocated_bundle, fold_index):
    evidence_path, evidence = relocated_bundle
    cell = evidence['cell_evidence'][fold_index]
    train = tuple(cell['threshold']['participant_ids'])
    all_ids = set(evidence['final_model']['training_participant_ids'])
    oof = tuple(sorted(all_ids - set(train)))
    config = adapter.ReusedMotionDetectorConfig(
        enabled=True, evidence_path=evidence_path,
        expected_evidence_sha256=sha256_file(evidence_path), device='cpu',
        reuse_scope='matching_outer_fold_or_all29_final',
        expected_split_registry_sha256=evidence['split_registry_csv_sha256'],
    )
    loaded = adapter.load_reused_motion_detector(
        config, outer_train_participant_ids=train, outer_oof_participant_ids=oof,
    )
    model_path = Path(loaded.provenance['model_artifact_path'])
    assert model_path == evidence_path.parent / f'repeat_0/fold_{fold_index}/formal_motion_model.pt'
    assert sha256_file(model_path) == cell['model_artifact_sha256']
    assert loaded.threshold == cell['threshold']['threshold']
    assert loaded.provenance['valid_outer_oof_claim'] is True
    assert Path(loaded.provenance['validated_split_registry_path']) == (
        evidence_path.parent.parent / 'splits/sgkf5_seed42_v2.csv'
    )


@pytest.mark.parametrize('scope', [
    'all29_smoke_or_final_only', 'all29_frozen_in_sample_auxiliary',
    'matching_outer_fold_or_all29_final',
])
def test_final_model_loads_adjacent_weight_in_every_scope(relocated_bundle, scope):
    evidence_path, evidence = relocated_bundle
    loaded = adapter.load_reused_motion_detector(adapter.ReusedMotionDetectorConfig(
        enabled=True, evidence_path=evidence_path,
        expected_evidence_sha256=sha256_file(evidence_path), device='cpu', reuse_scope=scope,
    ))
    model_path = evidence_path.parent / 'final_all_internal/formal_motion_model.pt'
    assert Path(loaded.provenance['model_artifact_path']) == model_path
    assert sha256_file(model_path) == evidence['final_model']['artifact_sha256']
    assert loaded.threshold == evidence['final_threshold']['threshold']


def test_missing_local_fold_never_falls_back_to_original_or_all29(relocated_bundle):
    evidence_path, evidence = relocated_bundle
    cell = evidence['cell_evidence'][0]
    directory = Path('repeat_0/fold_0')
    (evidence_path.parent / directory / 'formal_motion_model.pt').unlink()
    with pytest.raises(FileNotFoundError, match='repeat_0/fold_0'):
        adapter._resolve_model_path(evidence_path, cell['model_artifact_path'], directory=directory)
