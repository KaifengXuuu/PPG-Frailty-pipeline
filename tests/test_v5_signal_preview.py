"""Plots preserve acquisition-grid spectra and do not alter production arrays."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import signal

from ppg_frailty.dashboard.workflow_controls import apply_control_values, default_configuration
from ppg_frailty.dashboard.workflow_service import WorkflowService
from ppg_frailty.module_registry import list_modules


ROOT = Path(__file__).resolve().parents[1]
CALIBRATED = {'calibrated_roll_pitch_ekf', 'profile_a_lowpass_0p3hz',
              'sensor_filter_only_no_gravity_removal'}


def source(duration=12.0):
    time = np.arange(round(duration * 400)) / 400.0
    ppg = np.column_stack((100 + 5 * np.sin(2*np.pi*.05*time) + np.sin(2*np.pi*1.2*time),
                           120 + 8 * np.sin(2*np.pi*.05*time) + 2*np.sin(2*np.pi*1.2*time+.1)))
    acc = np.column_stack((.01*np.sin(time), .015*np.sin(.3*time), 1+.01*np.sin(.2*time)))
    gyro = np.column_stack((np.sin(time), np.cos(time), np.sin(time/2)))
    row = SimpleNamespace(record_id='signal-fixture', participant_id='person', role='B', class_id=1,
                          class_name='Robust', fs=400.0, n_samples=len(time), duration_s=duration,
                          source_path='signal-fixture.csv', qc_status='pass')
    loaded = dict(record_id=row.record_id, participant_id=row.participant_id, fs_hz=400.0,
                  ppg=ppg, acc=acc, gyro=gyro, acc_unit='g', gyro_unit='deg/s')
    return {'row': row, 'rows': [row], 'loaded': loaded, 'signals': {}, 'metadata': {}}


def service_for(data):
    service = WorkflowService(ROOT)
    service._input = lambda *args: data
    return service


@pytest.mark.parametrize('fs_hz', [100.0, 400.0])
def test_fft_matches_notebook_amplitude_convention_and_preserves_dc(fs_hz):
    from ppg_frailty.dashboard.signal_preview import amplitude_spectra

    time = np.arange(int(fs_hz * 4)) / fs_hz
    values = 7 + 6 * np.sin(2 * np.pi * 10 * time)
    original = values.copy()
    values.setflags(write=False)
    result = amplitude_spectra({'input': values}, fs_hz=fs_hz)
    spectrum = result['fft_traces']['input']
    frequency, amplitude = np.asarray(spectrum['x']), np.asarray(spectrum['y'])
    assert frequency[0] == 0 and frequency[-1] == fs_hz / 2
    assert amplitude[0] == pytest.approx(7)
    peak = 1 + np.argmax(amplitude[1:])
    assert frequency[peak] == 10
    assert amplitude[peak] == pytest.approx(3)  # abs(rfft)/N, no positive-bin doubling.
    metadata = result['fft_metadata']['input']
    assert metadata['method'] == 'rfft' and metadata['scope'] == 'full_record'
    assert metadata['detrend'] is False and metadata['one_sided_doubling'] is False
    assert metadata['scaling'] == 'abs(rfft)/N'
    assert metadata['frequency_resolution_hz'] == .25
    assert metadata['used_samples'] == len(values)
    np.testing.assert_array_equal(values, original)


def test_fft_uses_longest_finite_run_without_splicing_gaps_or_averaging():
    from ppg_frailty.dashboard.signal_preview import amplitude_spectra

    short = np.full(100, 99.0)
    long = 3 + 2 * np.sin(2 * np.pi * 20 * np.arange(800) / 400)
    values = np.concatenate((short, [np.nan, np.inf], long, [np.nan], np.full(800, -9)))
    original = values.copy()
    result = amplitude_spectra({'gapped': values})
    spectrum, metadata = result['fft_traces']['gapped'], result['fft_metadata']['gapped']
    np.testing.assert_array_equal(spectrum['x'], np.fft.rfftfreq(len(long), 1/400))
    np.testing.assert_array_equal(spectrum['y'], np.abs(np.fft.rfft(long)) / len(long))
    assert spectrum['y'][0] == pytest.approx(3)
    assert metadata['scope'] == 'longest_finite_run'
    assert metadata['segment_start_sample'] == 102
    assert metadata['segment_stop_sample'] == 902
    assert metadata['used_samples'] == 800 and metadata['finite_samples'] == 1700
    assert metadata['finite_runs'] == 3
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize('values', [[], [np.nan, np.inf], [4], [1, np.nan, 2, np.nan]])
def test_fft_without_two_contiguous_finite_samples_is_explicitly_unavailable(values):
    from ppg_frailty.dashboard.signal_preview import amplitude_spectra

    result = amplitude_spectra({'input': np.asarray(values)})
    assert result['fft_traces'] == {}
    metadata = result['fft_metadata']['input']
    assert metadata['status'] == 'insufficient_contiguous_samples'
    assert metadata['used_samples'] == 0 and metadata['frequency_resolution_hz'] is None


def test_fft_zero_and_constant_inputs_have_real_zero_and_dc_bins():
    from ppg_frailty.dashboard.signal_preview import amplitude_spectra

    result = amplitude_spectra({'zero': np.zeros(16), 'constant': np.full(16, -4.0)})
    assert result['fft_traces']['zero']['y'] == [0.0] * 9
    assert result['fft_traces']['constant']['y'] == [4.0] + [0.0] * 8
    assert all(row['status'] == 'available' for row in result['fft_metadata'].values())


def test_spectrum_preserves_dc_and_uses_original_400hz_samples():
    from ppg_frailty.dashboard.signal_preview import power_spectra

    time = np.arange(24000) / 400.0
    values = 7 + np.sin(2*np.pi*80*time)
    original = values.copy()
    values.setflags(write=False)
    result = power_spectra({'input': values})
    frequency = np.asarray(result['frequency_traces']['input']['x'])
    power = np.asarray(result['frequency_traces']['input']['y'])
    metadata = result['spectrum_metadata']['input']
    assert frequency[0] == 0 and frequency[-1] == 200
    assert power[0] > 100  # Welch's default constant detrending would erase this DC term.
    high = frequency > 10
    assert abs(frequency[high][np.argmax(power[high])] - 80) < .1
    assert metadata['detrend'] is False and metadata['nfft'] == 8192
    assert metadata['finite_samples'] == len(values)
    np.testing.assert_array_equal(values, original)


def test_spectrum_keeps_gap_spans_separate_and_weights_real_welch_windows():
    from ppg_frailty.dashboard.signal_preview import power_spectra

    first = 4 + np.sin(np.arange(16401) * .1)
    second = -6 + np.cos(np.arange(4096) * .2)
    values = np.concatenate((first, np.full(13, np.nan), second, [np.nan, 99, np.nan]))
    result = power_spectra({'gapped': values})
    rows = result['frequency_traces']['gapped']
    expected = []
    weights = []
    for span in (first, second):
        size = min(len(span), 8192)
        frequency, power = signal.welch(span, fs=400, window='hann', nperseg=size,
                                        noverlap=size//2, nfft=8192, detrend=False)
        weights.append(1 + (len(span)-size)//(size-size//2))
        expected.append(power)
    np.testing.assert_array_equal(rows['x'], frequency)
    np.testing.assert_allclose(rows['y'], np.average(expected, axis=0, weights=weights), rtol=1e-13, atol=0)
    metadata = result['spectrum_metadata']['gapped']
    assert metadata['finite_samples'] == len(first)+len(second)+1
    assert metadata['used_samples'] == 16384+len(second)  # Last 17 samples do not fill another Welch window.
    assert metadata['welch_segments'] == sum(weights)
    assert metadata['window_samples_min'] == len(second)
    assert metadata['window_samples_max'] == 8192


def test_missing_or_singleton_samples_do_not_create_a_synthetic_spectrum():
    from ppg_frailty.dashboard.signal_preview import power_spectra

    result = power_spectra({'empty': np.array([]), 'missing': np.full(12, np.nan),
                           'singletons': np.array([1, np.nan, 2, np.nan])})
    assert result['frequency_traces'] == {}
    for metadata in result['spectrum_metadata'].values():
        assert metadata['used_samples'] == 0 and metadata['welch_segments'] == 0


def test_ppg_has_only_actual_inputs_and_outputs_and_filter_removes_baseline():
    from ppg_frailty.signal.preprocess import preprocess_ppg_pair

    data = source(120)
    data['loaded']['ppg'][:, 0] += .2*np.sin(2*np.pi*80*np.arange(48000)/400)
    config = default_configuration(ROOT)
    before = deepcopy(config)
    service = service_for(data)
    result = service.analyse('ppg', config, 'signal-fixture', start_s=5, duration_s=4)
    preview = result['previews']['ppg']
    stage = service._sessions['default']['signal-fixture']['ppg']['value']
    assert set(preview['frequency_traces']) == {
        'raw_RED', 'raw_IR', 'native_RED', 'native_IR',
        'filtered_RED', 'filtered_IR'}
    assert set(preview['fft_traces']) == set(preview['frequency_traces'])
    for color in ('RED', 'IR'):
        raw = preview['frequency_traces']['raw_' + color]
        filtered = preview['frequency_traces']['filtered_' + color]
        assert filtered['y'][0] < 1e-4*raw['y'][0]
        band = (np.asarray(raw['x']) > 0) & (np.asarray(raw['x']) < .15)
        assert np.sum(np.asarray(filtered['y'])[band]) < .01*np.sum(np.asarray(raw['y'])[band])
        assert preview['frequency_groups']['raw_' + color] == preview['frequency_groups']['filtered_' + color]
        assert preview['trace_groups']['raw_' + color] == preview['trace_groups']['native_' + color]
        assert preview['trace_groups']['raw_' + color] == preview['trace_groups']['filtered_' + color]
        assert preview['trace_styles']['native_' + color]['visible'] == 'legendonly'
        assert preview['trace_figures']['raw_' + color] == 'Raw PPG'
        assert preview['trace_figures']['native_' + color] == 'Raw PPG'
        assert preview['trace_figures']['filtered_' + color] == 'Filtered PPG'
        fft = preview['fft_traces']['raw_' + color]
        np.testing.assert_array_equal(fft['y'], np.abs(np.fft.rfft(stage['signals']['raw_' + color])) / 48000)
        assert preview['fft_groups']['raw_' + color].endswith('FFT amplitude / raw counts')
    assert set(preview['trace_groups'].values()) == {'PPG / raw counts'}
    visible = [name for name in preview['traces'] if preview['trace_styles'].get(name, {}).get('visible', True) is True]
    assert set(visible) == {'raw_RED', 'raw_IR', 'filtered_RED', 'filtered_IR'}
    expected = preprocess_ppg_pair(data['loaded']['ppg'])
    for actual, reference in zip(stage['ppg_result'][:2], expected[:2]):
        np.testing.assert_array_equal(actual, reference)
    assert config == before
    again = service.analyse('ppg', config, 'signal-fixture', start_s=40, duration_s=30)
    assert again['computed_stages'] == []
    assert again['previews']['ppg']['frequency_traces'] == preview['frequency_traces']
    assert again['previews']['ppg']['fft_traces'] == preview['fft_traces']
    assert preview['traces']['raw_RED']['x'][0] == 5
    assert again['previews']['ppg']['traces']['raw_RED']['x'][0] == 40
    assert len(again['previews']['ppg']['traces']['raw_RED']['x']) <= 2000
    # A 30-second display is thinned below the 80 Hz Nyquist requirement; its
    # PSD must nevertheless still locate the original 400 Hz-grid component.
    spectrum = again['previews']['ppg']['frequency_traces']['raw_RED']
    frequency, power = np.asarray(spectrum['x']), np.asarray(spectrum['y'])
    high = frequency > 20
    assert abs(frequency[high][np.argmax(power[high])] - 80) < .1


def test_ppg_raw_gaps_are_visible_without_replacing_repaired_algorithm_input():
    data = source()
    data['loaded']['ppg'][200:204, 0] = np.nan
    original = data['loaded']['ppg'].copy()
    service = service_for(data)
    preview = service.analyse('ppg', default_configuration(ROOT), 'signal-fixture')['previews']['ppg']
    metadata = preview['metadata']['spectrum']
    assert metadata['raw_RED']['finite_samples'] == len(original)-4
    assert metadata['native_RED']['finite_samples'] == len(original)
    fft_metadata = preview['metadata']['fft']
    assert fft_metadata['raw_RED']['scope'] == 'longest_finite_run'
    assert fft_metadata['raw_RED']['segment_start_sample'] == 204
    assert fft_metadata['raw_RED']['used_samples'] == len(original)-204
    assert fft_metadata['native_RED']['scope'] == 'full_record'
    stage = service._sessions['default']['signal-fixture']['ppg']['value']
    assert np.isnan(stage['signals']['raw_RED'][200:204]).all()
    assert np.isfinite(stage['ppg_result'][0]).all()
    assert np.isfinite(stage['ppg_result'][1]).all()
    np.testing.assert_array_equal(data['loaded']['ppg'], original)


@pytest.mark.parametrize('method', [item['module_id'] for item in list_modules('imu_gravity')])
def test_every_imu_branch_shows_si_inputs_and_unchanged_actual_outputs(method):
    from ppg_frailty.signal.imu import convert_acceleration, convert_gyro
    from ppg_frailty.signal.motion_imu import _convert_profile_acceleration, fit_motion_imu_calibration
    from ppg_frailty.signal.preprocess import build_signal_views, roll_pitch_ekf_config_from_resolved

    data = source()
    original_arrays = {key: data['loaded'][key].copy() for key in ('ppg', 'acc', 'gyro')}
    changes = {'signal.imu.gravity_method': method}
    if method in CALIBRATED:
        changes.update({'signal.imu.gravity_mps2': 9.7, 'signal.imu.calibration_start_s': .5,
                        'signal.imu.calibration_stop_s': 3.0})
    config = apply_control_values(default_configuration(ROOT), changes, pipeline_root=ROOT)
    service = service_for(data)
    result = service.analyse('imu', config, 'signal-fixture')
    stage = service._sessions['default']['signal-fixture']['imu']['value']
    preview = result['previews']['imu']
    expected_source = dict(data['loaded'])
    if method in CALIBRATED:
        parameters = roll_pitch_ekf_config_from_resolved(config['signal']['imu'])
        expected_source['imu_calibration'] = fit_motion_imu_calibration(
            expected_source['acc'], expected_source['gyro'], participant_id='person', file_id='signal-fixture',
            source_role='B', fs_hz=400, acceleration_unit='g', gyroscope_unit='deg/s', config=parameters)
        raw_acc = _convert_profile_acceleration(expected_source['acc'], 'g', gravity_mps2=9.7)
    else:
        raw_acc = convert_acceleration(expected_source['acc'], 'g')
    raw_gyro = convert_gyro(expected_source['gyro'], 'deg/s')
    expected = build_signal_views(expected_source, config)
    for field in ('x_native', 'x_filter', 'x_analysis_rate', 'source_valid_mask', 'repair_mask'):
        np.testing.assert_array_equal(getattr(stage['views'], field), getattr(expected, field))
    for name, value in expected.imu_processed.items():
        np.testing.assert_array_equal(stage['views'].imu_processed[name], value)
        array = np.asarray(value)
        names = [name] if array.ndim == 1 else [name+'_'+axis for axis in 'xyz'] if array.shape[1] == 3 else []
        assert set(names) <= preview['traces'].keys()
    for index, axis in enumerate('xyz'):
        np.testing.assert_array_equal(stage['signals']['raw_acc_mps2_'+axis], raw_acc[:, index])
        np.testing.assert_array_equal(stage['signals']['raw_gyro_rads_'+axis], raw_gyro[:, index])
        assert {prefix+axis for prefix in ('raw_acc_mps2_', 'raw_gyro_rads_', 'acc_mps2_',
                                           'gyro_rads_', 'dynamic_acc_mps2_')} <= preview['frequency_traces'].keys()
        assert 'FFT amplitude / m/s²' in preview['fft_groups']['raw_acc_mps2_' + axis]
        assert 'FFT amplitude / rad/s' in preview['fft_groups']['raw_gyro_rads_' + axis]
        fft = preview['fft_traces']['raw_acc_mps2_' + axis]
        np.testing.assert_array_equal(fft['y'], np.abs(np.fft.rfft(raw_acc[:, index])) / len(raw_acc))
    assert set(preview['fft_traces']) == set(preview['frequency_traces'])
    assert not {'imu_valid_mask', 'gravity_valid_mask', 'gravity_confidence', 'repair_mask'} & preview['frequency_traces'].keys()
    if method == 'sensor_filter_only_no_gravity_removal':
        np.testing.assert_array_equal(expected.imu_processed['gravity_mps2'], 0)
        np.testing.assert_array_equal(expected.imu_processed['dynamic_acc_mps2'], expected.imu_processed['acc_mps2'])
    for name, values in original_arrays.items():
        np.testing.assert_array_equal(data['loaded'][name], values)
