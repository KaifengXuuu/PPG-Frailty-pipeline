"""Preview display preserves stage data, physical units and feature-table rows."""
from copy import deepcopy

from ppg_frailty.dashboard.app import _render_preview


def graphs(preview):
    def walk(components):
        for component in components if isinstance(components, (list, tuple)) else [components]:
            if component.__class__.__name__ == 'Graph':
                yield component.figure
            elif hasattr(component, 'children'):
                yield from walk(component.children)
    return list(walk(_render_preview(preview)))


def test_explicit_time_groups_preserve_ppg_views_and_marker_styles():
    names = ('native_RED', 'filled_RED', 'filtered_RED',
             'direct_RED', 'direct_points', 'processed_RED', 'processed_points',
             'actual_RED', 'actual_points')
    groups = ('Raw PPG', 'Gap-repaired PPG', 'Actual bandpass',
              'direct', 'direct', 'processed', 'processed', 'actual-used', 'actual-used')
    preview = {
        'traces': {name: {'x': [0., .2], 'y': [1., 2.]} for name in names},
        'trace_groups': dict(zip(names, groups)),
        'trace_styles': {name: {'mode': 'markers', 'marker': {'symbol': symbol, 'size': 8}}
                         for name, symbol in zip(names[4::2], ('circle', 'diamond', 'x'))},
    }
    preview['trace_styles']['filled_RED'] = {'line': {'dash': 'dot'}}
    original = deepcopy(preview)
    figure, = graphs(preview)
    plotted = {trace.name: trace for trace in figure.data}
    assert [annotation.text for annotation in figure.layout.annotations] == list(dict.fromkeys(groups))
    assert len({plotted[name].yaxis for name in names[:3]}) == 3
    for signal, peaks in zip(names[3::2], names[4::2]):
        assert plotted[signal].yaxis == plotted[peaks].yaxis
        assert plotted[peaks].mode == 'markers'
        assert plotted[peaks].marker.size == 8
    assert plotted['processed_points'].marker.symbol == 'diamond'
    assert plotted['filled_RED'].line.dash == 'dot'
    assert figure.layout.xaxis6.title.text == 'Time / s'
    assert preview == original


def test_window_and_beatwise_rate_figures_keep_separate_units_and_actual_points():
    preview = {
        'traces': {
            'direct_RED': {'x': [0., 1.], 'y': [100., 101.]},
            'window_ppi_mean': {'x': [5., 7.], 'y': [.8, None]},
            'window_hr_mean': {'x': [5., 7.], 'y': [76., None]},
            'window_hr_sd': {'x': [5., 7.], 'y': [4., None]},
            'beat_PPI_valid': {'x': [.4, 1.3], 'y': [.8, 1.]},
            'beat_HR_valid': {'x': [.4, 1.3], 'y': [75., 60.]},
            'beat_HR_invalid': {'x': [2.5], 'y': [250.]},
        },
        'trace_groups': {
            'direct_RED': 'PPG', 'window_ppi_mean': 'PPI / s',
            'window_hr_mean': 'HR / bpm', 'window_hr_sd': 'HR / bpm',
            'beat_PPI_valid': 'PPI / s', 'beat_HR_valid': 'HR / bpm', 'beat_HR_invalid': 'HR / bpm'},
        'trace_figures': {
            **{name: 'Window PPI / HR' for name in ('window_ppi_mean', 'window_hr_mean', 'window_hr_sd')},
            **{name: 'Beatwise PPI / HR' for name in ('beat_PPI_valid', 'beat_HR_valid', 'beat_HR_invalid')}},
        'trace_styles': {
            name: {'mode': 'lines+markers', 'connectgaps': False, 'visible': True}
            for name in ('window_ppi_mean', 'window_hr_mean', 'beat_PPI_valid', 'beat_HR_valid')},
    }
    preview['trace_styles']['window_hr_sd'] = {'mode': 'lines+markers', 'connectgaps': False, 'visible': True}
    preview['trace_styles']['beat_HR_invalid'] = {'mode': 'markers', 'marker': {'symbol': 'x'}}
    original = deepcopy(preview)
    waveform, window, beat = graphs(preview)
    assert len(waveform.data) == 1
    for figure, title in ((window, 'Window PPI / HR'), (beat, 'Beatwise PPI / HR')):
        assert figure.layout.uirevision == title
        assert [annotation.text for annotation in figure.layout.annotations] == ['PPI / s', 'HR / bpm']
        assert figure.layout.xaxis2.title.text == 'Time / s'
        for trace in figure.data:
            assert list(trace.x) == preview['traces'][trace.name]['x']
            assert list(trace.y) == preview['traces'][trace.name]['y']
            assert trace.mode == ('markers' if trace.name.endswith('_invalid') else 'lines+markers')
            if not trace.name.endswith('_invalid'):
                assert trace.connectgaps is False
                assert trace.visible is True
            assert trace.yaxis == ('y' if preview['trace_groups'][trace.name] == 'PPI / s' else 'y2')
        figure.to_json(validate=True)
    assert window.data[2].visible is True
    assert beat.data[2].marker.symbol == 'x'
    assert preview == original


def test_ppg_raw_and_filtered_have_separate_figures_without_rescaling():
    names = ('raw_RED', 'raw_IR', 'filtered_RED', 'filtered_IR', 'native_RED', 'native_IR')
    preview = {
        'traces': {name: {'x': [0., 1.], 'y': [value, value + .1]}
                   for name, value in zip(names, (100., 120., .1, .2, 100., 120.))},
        'trace_groups': {name: 'PPG / raw counts' for name in names},
        'trace_figures': {name: 'Filtered PPG' if name.startswith('filtered_') else 'Raw PPG'
                          for name in names},
        'trace_styles': {name: {'visible': 'legendonly'} for name in names[-2:]},
    }
    raw, filtered = graphs(preview)
    assert raw.layout.uirevision == 'Raw PPG'
    assert filtered.layout.uirevision == 'Filtered PPG'
    assert {trace.name for trace in filtered.data} == {'filtered_RED', 'filtered_IR'}
    assert sum(trace.visible != 'legendonly' for trace in raw.data) == 2
    for figure in (raw, filtered):
        assert [item.text for item in figure.layout.annotations] == ['PPG / raw counts']
        assert {trace.yaxis for trace in figure.data} == {'y'}
        for trace in figure.data:
            assert list(trace.y) == preview['traces'][trace.name]['y']


def test_frequency_axis_is_hz_and_units_are_separate_without_rewriting_zero_psd():
    preview = {
        'traces': {'native_AX': {'x': [0., 1.], 'y': [0., 1.]}},
        'trace_groups': {'native_AX': 'Acceleration / m/s²'},
        'frequency_traces': {
            name: {'x': [0., 1., 2.], 'y': [0., .01, 2.]}
            for name in ('native_AX', 'filtered_AX', 'native_GX', 'filtered_GX')},
        'frequency_groups': {'native_AX': 'Acceleration PSD / (m/s²)²/Hz',
                             'filtered_AX': 'Acceleration PSD / (m/s²)²/Hz',
                             'native_GX': 'Angular velocity PSD / (rad/s)²/Hz',
                             'filtered_GX': 'Angular velocity PSD / (rad/s)²/Hz'},
    }
    original = deepcopy(preview)
    time, spectrum = graphs(preview)
    plotted = {trace.name: trace for trace in spectrum.data}
    assert time.layout.xaxis.title.text == 'Time / s'
    assert spectrum.layout.xaxis2.title.text == 'Frequency / Hz'
    assert spectrum.layout.xaxis.type == spectrum.layout.xaxis2.type == 'linear'
    assert spectrum.layout.yaxis.type == spectrum.layout.yaxis2.type == 'log'
    assert spectrum.layout.yaxis.title.text == spectrum.layout.yaxis2.title.text == 'PSD'
    assert plotted['native_AX'].yaxis == plotted['filtered_AX'].yaxis
    assert plotted['native_GX'].yaxis == plotted['filtered_GX'].yaxis
    assert plotted['native_AX'].yaxis != plotted['native_GX'].yaxis
    for name, trace in plotted.items():
        assert list(trace.x) == preview['frequency_traces'][name]['x']
        assert list(trace.y) == preview['frequency_traces'][name]['y']
    assert preview == original


def test_zero_only_spectrum_is_visible_on_linear_axis():
    spectrum, = graphs({'frequency_traces': {'zero': {'x': [0, 1], 'y': [0, 0]}}})
    assert spectrum.layout.yaxis.type == 'linear'
    assert spectrum.layout.xaxis.title.text == 'Frequency / Hz'
    assert list(spectrum.data[0].x) == [0, 1]
    assert list(spectrum.data[0].y) == [0, 0]


def test_nonfinite_psd_serialized_as_null_is_not_replaced_or_compared_with_zero():
    # Strict JSON turns unavailable/overflowed PSD values into None upstream.
    spectrum, = graphs({'frequency_traces': {
        'missing': {'x': [0, 1], 'y': [None, None]},
        'partial': {'x': [0, 1], 'y': [None, .1]}},
        'frequency_groups': {'missing': 'Missing PSD', 'partial': 'Partial PSD'}})
    assert spectrum.layout.yaxis.type == 'linear'
    assert spectrum.layout.yaxis2.type == 'log'
    assert list(spectrum.data[0].y) == [None, None]
    assert list(spectrum.data[1].y) == [None, .1]
    spectrum.to_json(validate=True)


def test_all_feature_tables_and_dimensions_remain_reachable_without_row_truncation():
    tables = {title: [{'dimension': index, 'feature': f'{title}_{index}', 'value': index / 7}
                      for index in range(count)]
              for title, count in (('pulse_morphology', 31), ('PRV', 17), ('feature_matrix', 121))}
    original = deepcopy(tables)
    rendered = _render_preview({'tables': tables, 'metadata': {'stage': 'features'}})
    headings = [component.children for component in rendered if component.__class__.__name__ == 'H4']
    displayed = [component for component in rendered if component.__class__.__name__ == 'DataTable']
    assert headings == ['Window PPI / HR', *[title.replace('_', ' ') for title in tables]]
    assert len(displayed) == len(tables)
    for table, rows in zip(displayed, tables.values()):
        assert table.data == rows
        assert table.page_size == 12
        assert table.sort_action == table.filter_action == 'native'
        assert {column['id'] for column in table.columns} == {'dimension', 'feature', 'value'}
    assert all(component.__class__.__name__ != 'Details' for component in rendered)
    assert tables == original


def test_feature_categories_are_closed_once_with_every_table_row_preserved():
    categories = {
        'engineering features': ['Engineering 115 · RED', 'Window matrix 146 · imu', 'Matrix · windows'],
        'file level features': ['File · engineering_summary', 'File · dual_optical', 'Dual optical · beats'],
        'time series features': ['File · ppi_basic_rate', 'File · hrv_time_domain', 'File · hrv_nonlinear',
                                 'Peaks · used_RED', 'PPI · processed_RED', 'Matrix · used PPI'],
        'freq domain features': ['File · hrv_spectral'],
        'morphology features': ['File · morphology', 'Morphology · beats'],
    }
    tables = {title: [{'feature': title, 'value': index} for index in range(30)]
              for titles in categories.values() for title in titles}
    preview = {'tables': tables, 'metadata': {'stage': 'features'}}
    original = deepcopy(preview)
    panels = [item for item in _render_preview(preview) if item.__class__.__name__ == 'Details']
    assert [panel.children[0].children for panel in panels] == list(categories)
    for panel, titles in zip(panels, categories.values()):
        assert panel.open is False
        assert not any(child.__class__.__name__ == 'Details' for child in panel.children)
        rendered = [child for child in panel.children if child.__class__.__name__ == 'DataTable']
        assert [table.data for table in rendered] == [tables[title] for title in titles]
    assert preview == original
    # Folding applies only to the feature stage, not model/report/other previews.
    assert not any(item.__class__.__name__ == 'Details' for item in _render_preview({'tables': tables}))


def test_role_widget_shows_families_and_the_exact_loaded_members():
    from ppg_frailty.dashboard.app import _parameter_control
    widget = _parameter_control({'path': 'roles', 'kind': 'multi', 'value': ['B', 'R'],
                                'choices': ['B', 'R', 'S', 'W'], 'resolved_roles': ['B', 'R1']})
    checklist = widget.children[1]
    assert [item['value'] for item in checklist.options] == ['B', 'R', 'S', 'W']
    assert checklist.value == ['B', 'R']
    assert widget.children[-1].children == 'Resolved roles: B, R1'


def test_legends_stay_above_plot_area_with_space_for_many_peak_labels():
    names = [f'processed_RED_peaks_{index}' for index in range(18)]
    preview = {'traces': {name: {'x': [0, 1], 'y': [2, 3]} for name in names},
               'trace_groups': {name: 'PPG' for name in names}}
    figure, = graphs(preview)
    assert figure.layout.legend.y > 1
    assert figure.layout.legend.yanchor == 'bottom'
    assert figure.layout.legend.orientation == 'h'
    assert figure.layout.margin.t >= 24 * len(names) / 2
    assert figure.layout.height - figure.layout.margin.t - figure.layout.margin.b >= 300
    figure.to_json(validate=True)


def test_fft_is_expanded_linear_amplitude_and_psd_is_closed_without_data_changes():
    values = {'raw_RED': {'x': [0., 1., 200.], 'y': [1e5, 100., 0.]}}
    preview = {'fft_traces': deepcopy(values), 'frequency_traces': deepcopy(values)}
    original = deepcopy(preview)
    components = _render_preview(preview)
    fft, = [component.figure for component in components if component.__class__.__name__ == 'Graph']
    panel, = [component for component in components if component.__class__.__name__ == 'Details']
    assert panel.open is False
    assert panel.children[0].children == 'Frequency domain / PSD'
    assert fft.layout.uirevision == 'Frequency domain / FFT amplitude'
    assert fft.layout.yaxis.type == 'linear'
    assert fft.layout.yaxis.title.text == 'FFT amplitude'
    assert tuple(fft.layout.xaxis.range) == (.01, 8.)
    assert tuple(fft.layout.yaxis.range) == (0., 105.)  # Out-of-view DC does not squash the visible band.
    assert list(fft.data[0].x) == values['raw_RED']['x']
    assert list(fft.data[0].y) == values['raw_RED']['y']
    assert graphs(preview)[1].layout.yaxis.type == 'log'
    assert preview == original


def test_missing_window_rates_keep_visible_plot_and_explain_unavailable_status():
    for reason in ('peak_detection_unavailable', 'no_windows_planned',
                   'no_windows_with_sufficient_route_eligible_ppi'):
        preview = {'metadata': {'stage': 'features', 'window_rate': {
            'status': 'unavailable', 'reason': reason, 'valid_window_count': 0}}}
        components = _render_preview(preview)
        figure, = graphs(preview)
        assert figure.layout.uirevision == 'Window PPI / HR'
        assert [item.text for item in figure.layout.annotations] == ['PPI / s', 'HR / bpm']
        message, = [item.children for item in components if getattr(item, 'role', None) == 'status']
        assert reason in message
        figure.to_json(validate=True)
