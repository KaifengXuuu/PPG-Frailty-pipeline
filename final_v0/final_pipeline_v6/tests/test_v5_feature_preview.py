"""Feature previews expose production intermediates without a second algorithm."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ppg_frailty import experiment
from ppg_frailty.config import load_config, validate_config_payload
from ppg_frailty.contracts import PulseResult, SignalRoute, to_strict_json_value
from ppg_frailty.dashboard.feature_preview import build_feature_preview, compute_window_rate_preview
from ppg_frailty.dashboard.workflow_service import WorkflowService
from ppg_frailty.quality.routing_timeline import (
    RoutingEvidence, build_routing_timeline, build_routing_windows, resolve_routing_evidence,
)
from ppg_frailty.signal.preprocess import build_signal_views

ROOT = Path(__file__).resolve().parents[1]
WINDOW_RATE_FEATURES = (
    'local_interval.ppi_mean_s', 'local_interval.ppi_median_s',
    'local_interval.ppi_population_sd_s', 'local_interval.hr_mean_bpm',
    'local_interval.hr_median_bpm', 'local_interval.hr_population_sd_bpm',
)


@pytest.fixture
def setup():
    config = load_config(ROOT / 'configs/presets/finalcase.yaml').to_dict()
    config['signal']['imu'] = {'gravity_method': 'low_pass_0p3hz'}
    config = validate_config_payload(config)
    time = np.arange(8000) / 400
    loaded = {'record_id': 'test', 'participant_id': 'person', 'fs_hz': 400,
              'ppg': np.column_stack((100 + np.sin(2 * np.pi * 1.2 * time),
                                      120 + 2 * np.sin(2 * np.pi * 1.2 * time + 0.1))),
              'acc': np.column_stack((.01 * np.sin(time), .01 * np.cos(time), 1 + .01 * np.sin(.3 * time))),
              'gyro': np.column_stack((np.sin(time), np.cos(time), np.sin(.3 * time))),
              'acc_unit': 'g', 'gyro_unit': 'deg/s'}
    state = experiment._RuntimeRecord(SimpleNamespace(record_id='test', participant_id='person', role='B'),
        views=build_signal_views(loaded, config), retained=True, route=SignalRoute.DIRECT, quality_tier='excellent')
    evidence = [resolve_routing_evidence(RoutingEvidence(window, 'off', False), role='B')
                for window in build_routing_windows('test', len(time))]
    state.routing_timeline = build_routing_timeline(record_id='test', participant_id='person', role='B',
        n_samples=len(time), evidence=evidence, config_sha256='fixture')
    return state, WorkflowService._report(config), config['features']


@pytest.mark.parametrize('kind', ['vector', 'matrix'])
def test_capture_does_not_change_values_or_number_of_kernel_calls(setup, monkeypatch, kind):
    state, report, features = setup
    api = dict(experiment._runtime_imports())
    calls = Counter()
    for name in ('compute_prv', 'detect_pulses_per_wavelength', 'extract_engineering_features',
                 'extract_morphology', 'extract_dual_optical', 'extract_window_features'):
        original = api[name]

        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        api[name] = counted
    monkeypatch.setattr(experiment, '_runtime_imports', lambda: api)
    plain, observed = deepcopy(state), deepcopy(state)
    function = experiment._extract_vector if kind == 'vector' else experiment._extract_matrix_features
    args = (report, features) if kind == 'vector' else (report,)
    function(plain, *args)
    baseline_calls = calls.copy()
    calls.clear()
    capture = {}
    function(observed, *args, capture=capture)
    assert calls == baseline_calls
    assert observed.retained == plain.retained is True, observed.reason
    assert observed.reason == plain.reason
    assert to_strict_json_value(observed.diagnostic_components) == to_strict_json_value(plain.diagnostic_components)
    for field in ('values', 'start_samples', 'valid_row_mask'):
        np.testing.assert_array_equal(getattr(observed.engineering.sequence, field), getattr(plain.engineering.sequence, field))
    np.testing.assert_array_equal(observed.engineering.value_validity, plain.engineering.value_validity)
    if kind == 'vector':
        np.testing.assert_array_equal(observed.vector.values, plain.vector.values)
        np.testing.assert_array_equal(observed.vector.validity, plain.vector.validity)
        assert capture[kind]['result'] is observed.vector
        assert {'prv', 'morphology', 'optical', 'pulse_used', 'registry'} <= capture[kind].keys()
    else:
        assert capture[kind]['engineering'] is observed.engineering
        assert len(capture[kind]['base_engineering'].sequence.channel_schema) == 115
        assert 'pulse_used' not in capture[kind] and 'prv' not in capture[kind]
        assert capture[kind]['windows'][0]['sources'][0]['interval_indices'].size > 0
    before = calls.copy()
    preview = build_feature_preview(observed, capture)
    assert calls == before  # Display conversion invokes no kernel.
    assert preview['metadata']['feature_count'] == (len(observed.vector.feature_names) if kind == 'vector' else 146)
    table_prefix = 'File · ' if kind == 'vector' else 'Window matrix 146 · '
    assert any(name.startswith(table_prefix) for name in preview['tables'])
    assert set(preview['signals']) == {'direct_RED', 'direct_IR'}
    assert all(preview['trace_groups'][name] == 'PPG / amplitude' for name in preview['signals'])
    rate_names = [name for name in preview['point_traces'] if '_PPI' in name or '_HR' in name]
    assert rate_names and any('_HR' in name for name in rate_names)
    assert all(preview['trace_figures'][name] == 'Beatwise PPI / HR' for name in rate_names)
    assert any(name.startswith('window_') for name in preview['point_traces']) == (kind == 'matrix')
    for name, style in preview['trace_styles'].items():
        trend = name.startswith('window_') or name in rate_names and not name.endswith('_invalid')
        assert style['mode'] == ('lines+markers' if trend else 'markers')
        if trend:
            assert style['connectgaps'] is False
        if name in rate_names:
            assert style['visible'] == ('legendonly' if name.endswith('_invalid') else True)


def test_file_groups_and_each_window_value_keep_real_validity(setup):
    state, report, features = setup
    capture = {}
    features = {**features, 'enabled_groups': ['ppi_basic_rate']}
    experiment._extract_vector(state, report, features, capture=capture)
    assert state.retained, state.reason
    preview = build_feature_preview(state, capture)
    groups = {name for name in preview['tables'] if name.startswith('File · ')}
    assert len(groups) == 7
    assert preview['metadata']['feature_count'] == len(state.vector.values) < 282
    rows = [row for name, rows in preview['tables'].items() if name.startswith('File · ') for row in rows]
    assert len(rows) == 282
    assert sum(row['enabled'] for row in rows) == len(state.vector.values)
    engineering_rows = [row for name, rows in preview['tables'].items() if name.startswith('Engineering 115 · ') for row in rows]
    assert len(engineering_rows) == state.engineering.sequence.values.size
    names = list(state.engineering.sequence.channel_schema)
    for row in engineering_rows:
        i, j = row['window_index'], names.index(row['feature'])
        np.testing.assert_equal(row['value'], state.engineering.sequence.values[i, j])
        assert row['valid'] == bool(state.engineering.value_validity[i, j])
    optical = capture['vector']['optical']
    assert len(preview['tables']['Dual optical · pairing']) == len(optical.pairing.rows)
    for row, audit in zip(preview['tables']['Dual optical · beats'], optical.beat_audit):
        assert row['optical_reason_codes'] == audit.reason_codes


def _mixed_state(state, report):
    api = experiment._runtime_imports()
    pulses = experiment._direct_pulses_for_state(state, api, report.peak_detector)
    state.processed_pulses_per_wavelength = {name: replace(pulse,
        source_route=SignalRoute.ARTIFACT_RATE_ONLY, detection_run_id='processed-' + pulse.detection_run_id,
        interval_run_ids=np.full(pulse.ppi_s.shape, 'processed-' + pulse.detection_run_id))
        for name, pulse in pulses.items()}
    state.processed_views = replace(state.views, x_ar=state.views.x_filter * .25,
        route=SignalRoute.ARTIFACT_RATE_ONLY,
        metadata={**state.views.metadata, 'rate_only': True, 'q_morph_state': 'not_applicable',
                  'artifact_output_valid_mask': np.ones(len(state.views.x_filter), dtype=bool)})
    state.routing_timeline = replace(state.routing_timeline, cells=tuple(
        replace(cell, final_tier='acceptable', source_route='processed', source_view='x_ar_400')
        if cell.start_sample_400 >= 4000 else cell for cell in state.routing_timeline.cells))
    state.quality_tier = 'mixed'
    return state


def _assert_source_rates(preview, pulse, source):
    midpoint = (pulse.peak_timestamps_s[pulse.interval_start_peak_indices]
                + pulse.peak_timestamps_s[pulse.interval_stop_peak_indices]) / 2
    for valid in (True, False):
        selected = np.isfinite(pulse.ppi_s) & (pulse.valid_interval_mask == valid)
        suffix = 'valid' if valid else 'invalid'
        ppi_name = f'{source}_{pulse.wavelength}_{source}_PPI_{suffix}'
        hr_name = ppi_name.replace('_PPI_', '_HR_')
        expected_ppi = pulse.ppi_s[selected]
        expected_hr = np.full(expected_ppi.shape, np.nan)
        usable = expected_ppi > 0
        np.divide(60., expected_ppi, out=expected_hr, where=usable)
        ppi_x, ppi_y = preview['point_traces'][ppi_name]
        hr_x, hr_y = preview['point_traces'][hr_name]
        observed = np.isfinite(ppi_y)
        np.testing.assert_array_equal(ppi_x[observed], midpoint[selected])
        np.testing.assert_array_equal(ppi_y[observed], expected_ppi)
        np.testing.assert_array_equal(hr_x, ppi_x)
        np.testing.assert_array_equal(hr_y[observed], expected_hr)
        assert np.isnan(hr_y[~observed]).all()
        assert preview['trace_groups'][ppi_name] == 'PPI / s'
        assert preview['trace_groups'][hr_name] == 'HR / bpm'
        assert preview['trace_figures'][ppi_name] == preview['trace_figures'][hr_name] == 'Beatwise PPI / HR'
        for name in (ppi_name, hr_name):
            assert preview['trace_styles'][name]['mode'] == ('lines+markers' if valid else 'markers')
            if valid:
                assert preview['trace_styles'][name]['connectgaps'] is False
            else:
                assert observed.all()


def test_used_composite_peaks_use_corresponding_waveform_and_keep_boundary_invalid(setup):
    state, report, features = setup
    state = _mixed_state(state, report)
    capture = {}
    experiment._extract_vector(state, report, features, capture=capture)
    assert state.retained, state.reason
    preview = build_feature_preview(state, capture)
    used = capture['vector']['pulse_used']
    table = preview['tables'][f'PPI · used_{used.wavelength}']
    boundaries = [row for row in table if row['source_route'] == 'routing_boundary']
    assert boundaries and all(not row['valid'] and np.isnan(row['ppi_s']) for row in boundaries)
    assert all(np.isnan(row['hr_bpm']) for row in boundaries)
    assert 'Dual optical · pairing' not in preview['tables']  # Mixed routing never ran optical.
    for source, waveform in [('direct', state.views.x_filter), ('processed', state.processed_views.x_ar)]:
        for pulse in capture['vector'][f'pulses_{source}'].values():
            _assert_source_rates(preview, pulse, source)
        x, y = preview['point_traces'][f'used_{used.wavelength}_{source}_peaks_accepted']
        assert len(x)
        np.testing.assert_array_equal(y, waveform[np.rint(x * 400).astype(int), 0 if used.wavelength == 'RED' else 1])
        ppi_name = f'used_{used.wavelength}_{source}_PPI_valid'
        hr_name = ppi_name.replace('_PPI_', '_HR_')
        np.testing.assert_array_equal(preview['point_traces'][hr_name][0], preview['point_traces'][ppi_name][0])
        np.testing.assert_array_equal(preview['point_traces'][hr_name][1], 60. / preview['point_traces'][ppi_name][1])


def test_matrix_preview_exposes_only_actual_selected_intervals_and_differences(setup):
    state, report, _ = setup
    state = _mixed_state(state, report)
    capture = {}
    experiment._extract_matrix_features(state, report, capture=capture)
    assert state.retained, state.reason
    preview = build_feature_preview(state, capture)
    assert not any(name.startswith('File · ') or name.startswith('PRV · ') for name in preview['tables'])
    assert preview['metadata']['feature_count'] == 146
    rows = preview['tables']['Matrix · used PPI']
    assert rows and all(row['valid'] for row in rows)
    expected = sum(len(source['interval_indices']) for window in capture['matrix']['windows']
                   for source in window.get('sources', ()))
    assert len(rows) == expected
    assert all(row['hr_bpm'] == 60. / row['ppi_s'] for row in rows)
    for source in ('direct', 'processed'):
        for pulse in capture['matrix'][f'pulses_{source}'].values():
            _assert_source_rates(preview, pulse, source)
        selected = [item for window in capture['matrix']['windows'] for item in window.get('sources', ())
                    if item['pulse'].source_route == (SignalRoute.DIRECT if source == 'direct' else SignalRoute.ARTIFACT_RATE_ONLY)]
        pulse = selected[0]['pulse']
        indices = np.unique(np.concatenate([item['interval_indices'] for item in selected]))
        assert indices.size
        midpoint = (pulse.peak_timestamps_s[pulse.interval_start_peak_indices[indices]]
                    + pulse.peak_timestamps_s[pulse.interval_stop_peak_indices[indices]]) / 2
        ppi_name = f'matrix_used_{source}_{pulse.wavelength}_PPI'
        hr_name = ppi_name.replace('_PPI', '_HR')
        ppi_x, ppi_y = preview['point_traces'][ppi_name]
        hr_x, hr_y = preview['point_traces'][hr_name]
        observed = np.isfinite(ppi_y)
        np.testing.assert_array_equal(ppi_x[observed], midpoint)
        np.testing.assert_array_equal(ppi_y[observed], pulse.ppi_s[indices])
        np.testing.assert_array_equal(hr_x, ppi_x)
        np.testing.assert_array_equal(hr_y[observed], 60. / pulse.ppi_s[indices])
        assert np.isnan(hr_y[~observed]).all()
        assert preview['trace_figures'][hr_name] == 'Beatwise PPI / HR'
        assert preview['trace_styles'][hr_name]['mode'] == 'lines+markers'
        assert preview['trace_styles'][hr_name]['connectgaps'] is False
    for row in preview['tables']['Matrix · used successive pairs']:
        assert row['right_interval_index'] == row['left_interval_index'] + 1
        assert row['delta_ppi_s'] == row['right_ppi_s'] - row['left_ppi_s']


@pytest.mark.parametrize('kind', ['vector', 'matrix', 'both'])
def test_all_peak_layers_share_four_ppg_waveforms_without_aliases(setup, kind):
    state, report, features = setup
    state = _mixed_state(state, report)
    capture = {}
    if kind in {'vector', 'both'}:
        experiment._extract_vector(state, report, features, capture=capture)
    if kind in {'matrix', 'both'}:
        experiment._extract_matrix_features(state, report, capture=capture)
    assert state.retained, state.reason
    preview = build_feature_preview(state, capture)
    expected = {'direct_RED': state.views.x_filter[:, 0], 'direct_IR': state.views.x_filter[:, 1],
                'processed_RED': state.processed_views.x_ar[:, 0], 'processed_IR': state.processed_views.x_ar[:, 1]}
    assert preview['signals'].keys() == expected.keys()
    for name, waveform in expected.items():
        np.testing.assert_array_equal(preview['signals'][name], waveform)
    peaks = [name for name in preview['point_traces'] if '_peaks' in name]
    assert peaks
    assert any(name.startswith('used_') for name in peaks) == (kind != 'matrix')
    assert any(name.startswith('matrix_used_') for name in peaks) == (kind != 'vector')
    assert all(preview['trace_groups'][name] == 'PPG / amplitude' for name in (*expected, *peaks))
    assert all(preview['trace_styles'][name]['mode'] == 'markers' for name in peaks)


@pytest.mark.parametrize('boundary', [
    'contiguous', 'rejected', 'missing', 'routing', 'source', 'run', 'endpoints', 'adjacency', 'selection',
])
def test_beat_trends_preserve_points_and_break_at_real_boundaries(setup, boundary):
    state, _, _ = setup
    timestamps = .25 + np.r_[0., np.cumsum([.75, .8, .85, .9, 1., .85, .95, 1.05, .9])]
    pulse = PulseResult(
        peaks=np.rint(timestamps * 400).astype(int), peak_timestamps_s=timestamps,
        accepted_peak_mask=np.ones(10, dtype=bool),
        interval_start_peak_indices=np.arange(8), interval_stop_peak_indices=np.arange(1, 9),
        ppi_s=np.diff(timestamps)[:8], valid_interval_mask=np.ones(8, dtype=bool),
        adjacency_mask=np.ones(8, dtype=bool), wavelength='RED', detector_version='fixture',
        confidence=np.ones(10), source_route=SignalRoute.DIRECT, detection_run_id='run-a',
        interval_run_ids=np.full(8, 'run-a'), interval_source_routes=np.full(8, 'direct', dtype=object),
    )
    expected_breaks = []
    if boundary in {'rejected', 'routing'}:
        pulse.valid_interval_mask[3] = False
    if boundary in {'missing', 'routing'}:
        pulse.ppi_s[3] = np.nan
    if boundary == 'routing':
        pulse.interval_source_routes[3] = 'routing_boundary'
    if boundary == 'source':
        pulse.interval_source_routes[3] = SignalRoute.ARTIFACT_RATE_ONLY.value
    if boundary == 'run':
        pulse.interval_run_ids[3:] = 'run-b'
        expected_breaks = [3]
    if boundary == 'endpoints':
        pulse.interval_start_peak_indices[3:] += 1
        pulse.interval_stop_peak_indices[3:] += 1
        pulse.ppi_s[:] = timestamps[pulse.interval_stop_peak_indices] - timestamps[pulse.interval_start_peak_indices]
        expected_breaks = [3]
    if boundary == 'adjacency':
        pulse.adjacency_mask[3] = False
        expected_breaks = [3, 4]
    selected = np.flatnonzero(pulse.valid_interval_mask & np.isfinite(pulse.ppi_s))
    if boundary == 'selection':
        selected = selected[selected != 3]
    capture = {'vector': {'pulses_direct': {'RED': pulse}}, 'matrix': {'windows': [{
        'start_sample': 0, 'stop_sample': 4000,
        'sources': [{'pulse': pulse, 'interval_indices': selected, 'delta_pairs': []}],
    }]}}
    preview = build_feature_preview(state, capture)
    midpoints = (timestamps[pulse.interval_start_peak_indices] + timestamps[pulse.interval_stop_peak_indices]) / 2
    direct_indices = np.flatnonzero(pulse.valid_interval_mask & np.isfinite(pulse.ppi_s)
                                   & (pulse.interval_source_routes == 'direct'))
    direct_breaks = [4] if boundary in {'rejected', 'missing', 'routing', 'source'} else expected_breaks
    matrix_breaks = ([3, 4] if boundary == 'source' else [4]
                     if boundary in {'rejected', 'missing', 'routing', 'selection'} else expected_breaks)
    for prefix, suffix, indices, breaks in (
        ('direct_RED_direct', '_valid', direct_indices, direct_breaks),
        ('matrix_used_direct_RED', '', selected, matrix_breaks),
    ):
        for quantity in ('PPI', 'HR'):
            name = f'{prefix}_{quantity}{suffix}'
            x, y = preview['point_traces'][name]
            observed = np.isfinite(y)
            values = pulse.ppi_s[indices] if quantity == 'PPI' else 60. / pulse.ppi_s[indices]
            np.testing.assert_array_equal(x[observed], midpoints[indices])
            np.testing.assert_array_equal(y[observed], values)
            np.testing.assert_array_equal(x[~observed], midpoints[breaks])
            assert len(x) == len(indices) + len(breaks)
            assert np.isfinite(x).all()  # NaN x would be lost by service time slicing.
            for separator in np.flatnonzero(~observed):
                assert observed[separator + 1] and x[separator] == x[separator + 1]
            assert preview['trace_styles'][name]['mode'] == 'lines+markers'
            assert preview['trace_styles'][name]['connectgaps'] is False
            clipped = WorkflowService(ROOT)._preview('features', preview, midpoints[2], midpoints[6] - midpoints[2])
            keep = (x >= midpoints[2]) & (x < midpoints[6])
            assert clipped['traces'][name] == to_strict_json_value({'x': x[keep], 'y': y[keep]})
            assert clipped['traces'][name]['y'].count(None) == len(breaks)


def test_unusable_ppi_has_no_hr_and_rejected_intervals_stay_separate(setup):
    state, report, _ = setup
    pulses = experiment._direct_pulses_for_state(state, experiment._runtime_imports(), report.peak_detector)
    pulse = pulses['RED']
    ppi, valid = pulse.ppi_s.copy(), pulse.valid_interval_mask.copy()
    ppi[:6] = [.5, .75, 0., -1., np.nan, np.inf]
    valid[:6] = [True, False, True, True, True, True]
    pulse = replace(pulse, ppi_s=ppi, valid_interval_mask=valid)
    preview = build_feature_preview(state, {'vector': {'pulses_direct': {'RED': pulse}}})
    _assert_source_rates(preview, pulse, 'direct')
    rows = preview['tables']['PPI · direct_RED']
    np.testing.assert_array_equal([row['hr_bpm'] for row in rows[:6]], [120., 80., np.nan, np.nan, np.nan, np.nan])
    invalid_x, invalid_hr = preview['point_traces']['direct_RED_direct_HR_invalid']
    assert rows[1]['time_s'] in invalid_x
    assert invalid_hr[invalid_x == rows[1]['time_s']] == 80.
    assert rows[1]['time_s'] not in preview['point_traces']['direct_RED_direct_HR_valid'][0]
    assert preview['trace_styles']['direct_RED_direct_HR_invalid']['marker']['symbol'] == 'x'
    assert preview['trace_styles']['direct_RED_direct_HR_invalid']['visible'] == 'legendonly'
    assert preview['trace_styles']['direct_RED_direct_PPI_invalid']['visible'] == 'legendonly'
    assert not any(name.startswith('window_') for name in preview['point_traces'])


def test_window_rates_copy_captured_matrix_values_and_preserve_validity(setup, monkeypatch):
    state, report, _ = setup
    capture = {}
    experiment._extract_matrix_features(state, report, capture=capture)
    assert state.retained, state.reason
    engineering = capture['matrix']['engineering']
    sequence = engineering.sequence
    original_values = sequence.values.copy()
    columns = [sequence.channel_schema.index(feature) for feature in WINDOW_RATE_FEATURES]
    windows = capture['matrix']['windows']
    midpoint = np.asarray([(window['start_sample'] + window['stop_sample']) / 800 for window in windows])
    assert any(window['start_sample'] > 0 for window in windows)
    capture['window_rate'] = {'status': 'unavailable', 'reason': 'must_use_actual_matrix'}
    _forbid_preview_extraction(monkeypatch, rendering=True)
    for masked in (False, True):
        if masked:
            value_validity = engineering.value_validity.copy()
            value_validity[0, columns[0]] = False
            row_validity = sequence.valid_row_mask.copy()
            row_validity[-1] = False
            engineering = replace(engineering, value_validity=value_validity,
                                  sequence=replace(sequence, valid_row_mask=row_validity))
            capture['matrix']['engineering'] = engineering
        preview = build_feature_preview(state, capture)
        assert preview['metadata']['window_rate']['source'] == 'actual_feature_matrix'
        assert preview['metadata']['window_rate']['status'] == 'available'
        assert 'Window rates · local_interval' not in preview['tables']
        assert {name for name in preview['point_traces'] if name.startswith('window_')} == {
            f'window_{feature}' for feature in WINDOW_RATE_FEATURES}
        for feature, column in zip(WINDOW_RATE_FEATURES, columns):
            name = f'window_{feature}'
            valid = engineering.value_validity[:, column] & engineering.sequence.valid_row_mask
            expected = np.where(valid, sequence.values[:, column], np.nan)
            np.testing.assert_array_equal(preview['point_traces'][name][0], midpoint)
            np.testing.assert_array_equal(preview['point_traces'][name][1], expected)
            assert preview['trace_figures'][name] == 'Window PPI / HR'
            assert preview['trace_groups'][name] == ('PPI / s' if '.ppi_' in feature else 'HR / bpm')
            assert preview['trace_styles'][name]['mode'] == 'lines+markers'
            assert preview['trace_styles'][name]['connectgaps'] is False
            assert preview['trace_styles'][name]['visible'] is True
        # Display masks never mutate the captured values, including finite values
        # whose column/row validity makes them unavailable for the plotted series.
        np.testing.assert_array_equal(engineering.sequence.values, original_values)
    for incomplete in ({}, {'matrix': {key: value for key, value in capture['matrix'].items() if key != 'engineering'}}):
        preview = build_feature_preview(state, incomplete)
        assert not any(name.startswith('window_') for name in preview['point_traces'])
        assert 'Window PPI / HR' not in preview['trace_figures'].values()


def _forbid_preview_extraction(monkeypatch, *, rendering=False):
    def forbidden(*args, **kwargs):
        pytest.fail('Preview reran a production extractor or an already-captured rate calculation')

    monkeypatch.setattr(experiment, '_direct_pulses_for_state', forbidden)
    for path in ('ppg_frailty.peaks.detect_pulses_per_wavelength',
                 'ppg_frailty.signal.prv.compute_prv',
                 'ppg_frailty.signal.morphology.extract_morphology',
                 'ppg_frailty.features.window_matrix.extract_window_features',
                 'ppg_frailty.features.window_matrix.extract_engineering_features'):
        monkeypatch.setattr(path, forbidden)
    if rendering:
        monkeypatch.setattr('ppg_frailty.dashboard.feature_preview.compute_window_rate_preview', forbidden)
        monkeypatch.setattr('ppg_frailty.features.window_matrix._rate_features', forbidden)


@pytest.mark.parametrize('kind', ['vector', 'matrix'])
@pytest.mark.parametrize('routing', ['direct', 'mixed'])
def test_independent_window_rates_equal_matrix_raw_values_and_masks(setup, monkeypatch, kind, routing):
    state, report, features = setup
    if routing == 'mixed':
        state = _mixed_state(state, report)
        # Distinct processed validity makes using the wrong route observable.
        for wavelength, pulse in state.processed_pulses_per_wavelength.items():
            valid = pulse.valid_interval_mask.copy()
            valid[np.flatnonzero(valid)[::3]] = False
            state.processed_pulses_per_wavelength[wavelength] = replace(
                pulse, valid_interval_mask=valid)
        cells = state.routing_timeline.cells
        state.routing_timeline = replace(state.routing_timeline, cells=(*cells[:-1], replace(
            cells[-1], final_tier='excluded', source_route='excluded', source_view='none')))
    expected_state, expected_capture = deepcopy(state), {}
    experiment._extract_matrix_features(expected_state, report, capture=expected_capture)
    assert expected_state.retained, expected_state.reason
    if kind == 'vector':
        capture = {}
        experiment._extract_vector(state, report, features, capture=capture)
        assert state.retained, state.reason
    else:
        state, capture = expected_state, expected_capture
    before = to_strict_json_value(deepcopy(state.__dict__))
    _forbid_preview_extraction(monkeypatch)
    product = compute_window_rate_preview(state, report, capture)
    assert product['status'] == 'available'
    assert product['scope'] == 'matrix_routing'
    assert product['source'] == 'matrix_window_rate_kernels_preview_only'
    expected = expected_capture['matrix']['engineering']
    assert product['windows'] == [
        {key: value for key, value in window.items() if key != 'sources'}
        for window in expected_capture['matrix']['windows']]
    for feature in WINDOW_RATE_FEATURES:
        column = product['feature_names'].index(feature)
        expected_column = expected.sequence.channel_schema.index(feature)
        np.testing.assert_array_equal(product['values'][:, column], expected.sequence.values[:, expected_column])
        np.testing.assert_array_equal(product['validity'][:, column], expected.value_validity[:, expected_column])
    if routing == 'mixed':
        assert {window['tier'] for window in product['windows']} == {'excellent', 'acceptable', 'excluded'}
        assert np.any(product['validity']) and np.any(~product['validity'])
    assert to_strict_json_value(state.__dict__) == before


def test_unrouted_window_rates_keep_crossing_and_tail_intervals_without_state_changes(setup, monkeypatch):
    state, report, _ = setup
    state.routing_timeline = None
    state.quality_tier = None
    timestamps = np.asarray([.2, 1., 1.9, 2.7, 3.8, 4.6, 5.5, 6.2, 7.4, 8.6, 9.5,
                             10.3, 11.4, 12.1, 13., 13.8, 14.6, 15.3, 16.5, 17.2, 18.1, 19., 19.8])
    count = len(timestamps) - 1
    pulse = PulseResult(
        peaks=np.rint(timestamps * 400).astype(int), peak_timestamps_s=timestamps,
        accepted_peak_mask=np.ones(count + 1, dtype=bool), confidence=np.ones(count + 1),
        interval_start_peak_indices=np.arange(count), interval_stop_peak_indices=np.arange(1, count + 1),
        ppi_s=np.diff(timestamps), valid_interval_mask=np.ones(count, dtype=bool),
        adjacency_mask=np.ones(count, dtype=bool), wavelength='RED', source_route=SignalRoute.DIRECT,
        detector_version='fixture', detection_run_id='fixture', interval_run_ids=np.full(count, 'fixture'),
        detector_id='fixture', selected_polarity=1, block_hri_provenance_hash='0' * 64,
        peak_ordinals=np.arange(count + 1), interval_rejection_reasons=('',) * count,
        detector_score=1., detector_coverage=1.)
    pulse.valid_interval_mask[5] = False
    capture = {'vector': {'pulses_direct': {'RED': pulse, 'IR': replace(pulse, wavelength='IR')}}}
    before = to_strict_json_value(deepcopy(state.__dict__))
    _forbid_preview_extraction(monkeypatch)
    product = compute_window_rate_preview(state, report, capture)
    assert product['status'] == 'available'
    assert product['scope'] == 'full_record_direct_quality_not_assessed'
    assert all(window['tier'] == 'not_assessed' and window['eligible'] for window in product['windows'])
    midpoints = (timestamps[:-1] + timestamps[1:]) / 2
    crossing = ((timestamps[:-1] < 8) & (timestamps[1:] > 8)
                | (timestamps[:-1] < 16) & (timestamps[1:] > 16))
    assert np.count_nonzero(crossing) == 2 and midpoints[-1] > 19
    included = np.zeros(count, dtype=bool)
    for row, window in enumerate(product['windows']):
        selected = ((midpoints >= window['start_sample'] / 400)
                    & (midpoints < window['stop_sample'] / 400) & pulse.valid_interval_mask)
        included |= selected
        ppi = pulse.ppi_s[selected]
        hr = 60. / ppi
        expected = [np.mean(ppi), np.median(ppi), np.std(ppi, ddof=0),
                    np.mean(hr), np.median(hr), np.std(hr, ddof=0)]
        columns = [product['feature_names'].index(feature) for feature in WINDOW_RATE_FEATURES]
        np.testing.assert_allclose(product['values'][row, columns], expected, rtol=0, atol=1e-12)
        assert product['validity'][row, columns].all()
    assert included[crossing].all() and included[-1] and not included[5]
    assert to_strict_json_value(state.__dict__) == before
    assert state.routing_timeline is None and state.direct_pulses_per_wavelength is None


@pytest.mark.parametrize('cap', [{'max_windows': 3}, {'max_window_fraction': .4}])
def test_window_rate_preview_honors_engineering_length_hop_alignment_and_cap(setup, monkeypatch, cap):
    state, report, _ = setup
    experiment._direct_pulses_for_state(state, experiment._runtime_imports(), report.peak_detector)
    report.window_profiles['engineering'].update({
        'window_seconds': 6., 'hop_seconds': 2.5, 'end_alignment': 'include_right_aligned_if_distinct',
        'max_windows': None, 'max_window_fraction': None, 'cap_policy': 'uniform_progress', **cap})
    _forbid_preview_extraction(monkeypatch)
    product = compute_window_rate_preview(state, report, {})
    assert [window['start_sample'] for window in product['windows']] == [0, 3000, 5600]
    assert [window['stop_sample'] for window in product['windows']] == [2400, 5400, 8000]
    preview = build_feature_preview(state, {'window_rate': product})
    for feature in WINDOW_RATE_FEATURES:
        np.testing.assert_array_equal(preview['point_traces'][f'window_{feature}'][0], [3., 10.5, 17.])


def test_extra_window_rates_render_without_recomputation_or_changing_file_features(setup, monkeypatch):
    state, report, features = setup
    capture = {}
    experiment._extract_vector(state, report, features, capture=capture)
    assert state.retained, state.reason
    baseline = build_feature_preview(state, capture)
    product = compute_window_rate_preview(state, report, capture)
    first_column = product['feature_names'].index(WINDOW_RATE_FEATURES[0])
    product['validity'][0, first_column] = False
    raw_values = product['values'].copy()
    before = to_strict_json_value(deepcopy(state.__dict__))
    _forbid_preview_extraction(monkeypatch, rendering=True)
    preview = build_feature_preview(state, {**capture, 'window_rate': product})
    assert preview['metadata']['window_rate'] == {
        'status': 'available', 'source': product['source'], 'scope': product['scope'],
        'window_count': len(product['windows']), 'valid_window_count': len(product['windows'])}
    assert preview['metadata']['feature_count'] == baseline['metadata']['feature_count'] == len(state.vector.values)
    assert preview['metadata']['displayed_file_feature_count'] == baseline['metadata']['displayed_file_feature_count'] == 282
    assert len(state.engineering.sequence.channel_schema) == 115
    for name, rows in baseline['tables'].items():
        assert to_strict_json_value(preview['tables'][name]) == to_strict_json_value(rows)
    rows = preview['tables']['Window rates · local_interval']
    assert len(rows) == 6 * len(product['windows'])
    assert {name for name in preview['point_traces'] if name.startswith('window_')} == {
        f'window_{feature}' for feature in WINDOW_RATE_FEATURES}
    midpoint = [(window['start_sample'] + window['stop_sample']) / 800 for window in product['windows']]
    for feature in WINDOW_RATE_FEATURES:
        column = product['feature_names'].index(feature)
        name = f'window_{feature}'
        expected = np.where(product['validity'][:, column], raw_values[:, column], np.nan)
        np.testing.assert_array_equal(preview['point_traces'][name][0], midpoint)
        np.testing.assert_array_equal(preview['point_traces'][name][1], expected)
        assert preview['trace_styles'][name]['mode'] == 'lines+markers'
        assert preview['trace_styles'][name]['connectgaps'] is False
        assert preview['trace_figures'][name] == 'Window PPI / HR'
        feature_rows = [row for row in rows if row['feature'] == feature]
        np.testing.assert_array_equal([row['value'] for row in feature_rows], expected)
        np.testing.assert_array_equal([row['valid'] for row in feature_rows], product['validity'][:, column])
    np.testing.assert_array_equal(product['values'], raw_values)
    assert to_strict_json_value(state.__dict__) == before


def test_missing_pulses_leave_window_rates_unavailable_without_inventing_plots(setup, monkeypatch):
    state, report, _ = setup
    _forbid_preview_extraction(monkeypatch)
    product = compute_window_rate_preview(state, report, {})
    assert product == {'status': 'unavailable', 'reason': 'peak_detection_unavailable'}
    preview = build_feature_preview(state, {'window_rate': product})
    assert preview['metadata']['window_rate'] == product
    assert preview['metadata']['feature_count'] == 0
    assert not preview['point_traces'] and not preview['tables']
    assert 'Window PPI / HR' not in preview['trace_figures'].values()
    assert state.direct_pulses_per_wavelength is None


def test_all_invalid_window_rates_explain_empty_trends(setup, monkeypatch):
    state, report, _ = setup
    pulses = experiment._direct_pulses_for_state(state, experiment._runtime_imports(), report.peak_detector)
    capture = {'vector': {'pulses_direct': {name: replace(
        pulse, valid_interval_mask=np.zeros(pulse.ppi_s.shape, dtype=bool)) for name, pulse in pulses.items()}}}
    product = compute_window_rate_preview(state, report, capture)
    assert product['status'] == 'available' and product['windows']
    assert not product['validity'].any() and np.isnan(product['values']).all()
    _forbid_preview_extraction(monkeypatch, rendering=True)
    preview = build_feature_preview(state, {**capture, 'window_rate': product})
    assert preview['metadata']['window_rate']['valid_window_count'] == 0
    assert preview['metadata']['window_rate']['reason'] == 'no_windows_with_sufficient_route_eligible_ppi'
    for feature in WINDOW_RATE_FEATURES:
        assert np.isnan(preview['point_traces'][f'window_{feature}'][1]).all()
    assert all(not row['valid'] and np.isnan(row['value'])
               for row in preview['tables']['Window rates · local_interval'])


def test_empty_window_plan_reports_its_reason_without_inventing_values(setup):
    state, report, _ = setup
    experiment._direct_pulses_for_state(state, experiment._runtime_imports(), report.peak_detector)
    report.window_profiles['engineering'].update(window_seconds=40., short_record_action='pad_right', min_valid_fraction=1.)
    product = compute_window_rate_preview(state, report, {})
    assert product['windows'] == []
    preview = build_feature_preview(state, {'window_rate': product})
    rate = preview['metadata']['window_rate']
    assert rate['window_count'] == rate['valid_window_count'] == 0
    assert rate['reason'] == 'no_windows_planned'
    for name in preview['point_traces']:
        if name.startswith('window_'):
            assert not preview['point_traces'][name][0].size
            assert not preview['point_traces'][name][1].size


def test_service_slices_beats_but_keeps_complete_window_trends_and_breaks():
    names = ('direct_RED_direct_PPI_valid', 'direct_RED_direct_HR_valid', 'window_local_interval.hr_mean_bpm')
    x = np.asarray([1., 5., 7.5, 7.5, 10.])
    value = {
        'point_traces': {name: (x, np.asarray([.8, .9, np.nan, .95, 1.])) for name in names},
        'trace_groups': dict(zip(names, ('PPI / s', 'HR / bpm', 'HR / bpm'))),
        'trace_figures': dict(zip(names, ('Beatwise PPI / HR', 'Beatwise PPI / HR', 'Window PPI / HR'))),
        'trace_styles': {name: {'mode': 'lines+markers', 'connectgaps': False} for name in names},
    }
    preview = WorkflowService(ROOT)._preview('features', value, 5., 5.)
    for name in names:
        expected = ({'x': x.tolist(), 'y': [.8, .9, None, .95, 1.]}
                    if name.startswith('window_') else {'x': [5., 7.5, 7.5], 'y': [.9, None, .95]})
        assert preview['traces'][name] == expected
    for field in ('trace_groups', 'trace_figures', 'trace_styles'):
        assert preview[field] == value[field]


def test_dropped_input_does_not_fabricate_products(setup):
    state, report, features = setup
    state.retained, state.reason = False, 'excluded_fixture'
    capture = {}
    experiment._extract_vector(state, report, features, capture=capture)
    experiment._extract_matrix_features(state, report, capture=capture)
    preview = build_feature_preview(state, capture)
    assert not preview['tables'] and not preview['point_traces']
    assert not preview['trace_figures']
    assert preview['metadata']['reason'] == 'excluded_fixture'


def test_rejected_record_keeps_real_upstream_peaks_but_discards_stale_capture(setup):
    state, report, features = setup
    state = _mixed_state(state, report)
    capture = {}
    experiment._extract_vector(state, report, features, capture=capture)
    assert capture['vector'].get('pulse_used') is not None
    state.retained = False
    experiment._extract_vector(state, report, features, capture=capture)
    preview = build_feature_preview(state, capture)
    assert capture['vector'] == {}
    assert any(name.startswith('Peaks · processed_') for name in preview['tables'])
    assert not any(name.startswith(('Peaks · used_', 'File · ', 'PRV · ')) for name in preview['tables'])
    assert preview['metadata']['feature_count'] == 0
