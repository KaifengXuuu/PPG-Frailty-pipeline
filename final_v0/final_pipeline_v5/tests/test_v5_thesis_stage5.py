"""Independent thesis tests reuse the existing kernels without extra training."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ppg_frailty.dashboard.tool_controls import plan_parameter_specs
from ppg_frailty.quality import stage5_pre as stage5
from ppg_frailty.v5 import specialized

ROOT = Path(__file__).resolve().parents[1]
PLANS = ROOT / 'configs/studies/thesis'
JOINT_PLAN = ROOT / 'configs/studies/static_line_b_staged_v2/stage5_pre.yaml'
MOTION_STAGES = ['internal_motion_oof', 'ptt_motion_external', 'ptt_motion_training_ablation',
                 'frailty29_reverse_evaluation', 'motion_model_comparison_package']


def _json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')


@pytest.fixture
def kernel_calls(monkeypatch):
    """Exercise real orchestration, replacing only expensive numerical entries."""
    calls = []

    def replace(name, stage, filename):
        def execute(*args, output_dir, **kwargs):
            calls.append((stage, kwargs))
            target = Path(output_dir) / filename
            _json(target, {'status': 'passed'})
            digest = stage5.sha256_file(target)
            return target if stage == MOTION_STAGES[-1] else SimpleNamespace(evidence_sha256=digest, report_sha256=digest)
        monkeypatch.setattr(stage5, name, execute)

    for name, stage, filename in zip(
        ['run_formal_internal_motion_reference', 'run_formal_ptt_motion_reference',
         'run_formal_ptt_motion_training_ablation', 'run_formal_internal_reverse_evaluation',
         '_write_motion_model_comparison_package'], MOTION_STAGES,
        ['motion_internal_evidence.json', 'motion_ptt_external_report.json',
         'motion_ptt_training_evidence.json', 'motion_internal_reverse_evaluation_report.json',
         'motion_model_comparison_manifest.json'], strict=True,
    ):
        replace(name, stage, filename)

    def denoiser(*args, **kwargs):
        calls.append(('ptt_denoiser_benchmark', kwargs))
        return {'status': 'passed', 'rows': [], 'summary_rows': [],
                'scoring_peak_detector': kwargs['scoring_peak_detector']}
    monkeypatch.setattr(stage5, 'run_ptt_denoiser_benchmark', denoiser)
    return calls


@pytest.mark.parametrize(('filename', 'prefix', 'sections'), [
    ('denoiser', '20260820_182324', ['ptt_dataset', 'denoiser_benchmark', 'report', 'result_backup']),
    ('motion_detector', '20260820_225546', ['ptt_dataset', 'motion_detector', 'motion_model_comparison', 'report', 'result_backup']),
])
def test_thesis_parameters_match_frozen_v2_run(filename, prefix, sections):
    plan = stage5.load_motion_peak_plan(PLANS / f'{filename}.yaml')
    history = ROOT.parent / 'final_pipeline_v2/artifacts/studies/static_line_b_staged_v2'
    source = next(history.glob(f'{prefix}*/resolved_plan.yaml'), None)
    if source is None:
        pytest.skip('Historical V2 artifacts are optional in a standalone V5 checkout')
    frozen = yaml.safe_load(source.read_text())
    for section in sections:
        assert plan.payload[section] == frozen[section]


@pytest.mark.parametrize(('filename', 'motion', 'denoiser'), [('motion_detector', True, False), ('denoiser', False, True)])
def test_plan_loaders_and_dash_controls_accept_independent_tests(filename, motion, denoiser):
    path = PLANS / f'{filename}.yaml'
    plan = stage5.load_motion_peak_plan(path)
    assert plan.payload['execution'] == {'motion': motion, 'denoiser': denoiser}
    assert specialized.validate_specialized_plan(path)['status'] == 'valid'
    controls = {spec['path']: spec for spec in plan_parameter_specs(plan.payload)}
    for name, value in [('motion', motion), ('denoiser', denoiser)]:
        assert controls[f'execution.{name}']['kind'] == 'boolean'
        assert controls[f'execution.{name}']['value'] is value
    if denoiser:
        assert 'motion_detector' not in plan.payload
        assert plan.payload['denoiser_benchmark']['scoring_peak_detector'] == 'aboy_project_v1'
        assert 'scoring_peak_detector_parameters' not in plan.payload['denoiser_benchmark']
    else:
        assert 'denoiser_benchmark' not in plan.payload


@pytest.mark.parametrize(('plan', 'include_denoiser', 'motion', 'denoiser'), [
    (PLANS / 'denoiser.yaml', True, False, True),
    (PLANS / 'motion_detector.yaml', True, True, False),
    (JOINT_PLAN, True, True, True),
    (JOINT_PLAN, False, True, False),
])
def test_runner_executes_only_selected_existing_stages(tmp_path, kernel_calls, plan, include_denoiser, motion, denoiser):
    progress = []
    output = stage5.run_motion_peak_study(plan, pipeline_root=ROOT, output_root=tmp_path,
                                         include_denoiser=include_denoiser, progress_sink=progress.append)
    expected = (MOTION_STAGES if motion else []) + (['ptt_denoiser_benchmark'] if denoiser else [])
    assert [name for name, kwargs in kernel_calls] == expected
    manifest = json.loads((output / 'study_manifest.json').read_text())
    assert manifest['status'] == 'passed'
    assert manifest['motion_enabled'] is motion
    assert manifest['denoiser_enabled'] is denoiser
    assert (manifest['training_device'] is not None) is motion
    assert progress[-1].current == progress[-1].total == len(expected)
    if denoiser:
        benchmark = stage5.load_motion_peak_plan(plan).payload['denoiser_benchmark']
        passed = kernel_calls[-1][1]
        assert passed['reducer_ids'] == benchmark['reducers']
        for field in ('segment_s', 'validation', 'activities', 'scoring_peak_detector'):
            assert passed[field] == benchmark[field]
        assert passed['scoring_peak_detector_parameters'] == benchmark.get('scoring_peak_detector_parameters')
    else:
        assert manifest['stages']['ptt_denoiser_benchmark']['status'] == ('skipped_by_plan' if include_denoiser else 'skipped_by_cli')


def test_completed_denoiser_run_resumes_without_motion_or_recomputation(tmp_path, kernel_calls):
    path = PLANS / 'denoiser.yaml'
    output = stage5.run_motion_peak_study(path, pipeline_root=ROOT, output_root=tmp_path)
    kernel_calls.clear()
    assert stage5.run_motion_peak_study(path, pipeline_root=ROOT, output_root=tmp_path, resume=output) == output
    assert kernel_calls == []


@pytest.mark.parametrize('motion_enabled', [False, True, None])
def test_specialized_model_export_only_runs_for_motion_training(tmp_path, monkeypatch, motion_enabled):
    manifest = {} if motion_enabled is None else {'motion_enabled': motion_enabled}
    _json(tmp_path / 'study_manifest.json', manifest)
    monkeypatch.setattr(specialized, 'export_specialized_data_excel', lambda *args, **kwargs: {'workbook': 'tables/pipeline_data.xlsx'})
    calls = []
    monkeypatch.setattr(specialized, 'export_motion_model_config',
                        lambda root: calls.append(root) or {'output_directory': 'model_config/test'})
    result = specialized._publish_specialized_artifact_contract(tmp_path, stage5.STAGE5_SCHEMA)
    assert result['model_trained'] is (motion_enabled is not False)
    assert calls == ([] if motion_enabled is False else [tmp_path])
    if motion_enabled is False:
        assert 'model_config_export' not in result


def test_denoiser_output_uses_existing_report_reader_without_motion_files(tmp_path, monkeypatch, kernel_calls):
    from ppg_frailty.reporting import specialized as reporting

    output = stage5.run_motion_peak_study(PLANS / 'denoiser.yaml', pipeline_root=ROOT, output_root=tmp_path / 'pipeline_output')
    summary = {'algorithm_or_reducer': 'identity', 'activity_group': 'dynamic', 'channel': 'RED',
               'participant_macro_ibi_ppi_rmse_ms': 123.0}
    _json(output / 'denoiser/denoiser_benchmark.json', {'rows': [], 'summary_rows': [summary]})
    monkeypatch.setattr(reporting, '_render', lambda *args, **kwargs: False)
    monkeypatch.setattr(reporting, '_plot_summary', lambda *args, **kwargs: None)
    monkeypatch.setattr(reporting, '_finish_report', lambda output, **kwargs: {table.name: table.rows for table in kwargs['tables']})
    tables = reporting.generate_motion_peak_report(output, output_dir=tmp_path / 'report_output')
    assert list(tables['denoiser_summary']) == [summary]
    assert list(tables['motion_detector_metrics']) == []
    assert not any(row['component_role'] == 'motion_detector' for row in tables['test_components'])
    assert len(tables['denoiser_algorithms']) == 7
