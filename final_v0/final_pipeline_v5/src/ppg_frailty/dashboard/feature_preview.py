"""Display captured feature products without rerunning their extraction.

Tables retain unavailable values and the extractor's validity separately. Plot
coordinates use captured peak/interval positions. Beatwise HR is only the unit
conversion 60/PPI; window statistics come from the production matrix or its
shared window/rate kernels, independently of the selected model representation.
Valid trends connect consecutive intervals, never bridging rejected intervals
or routing boundaries. Unavailable windows remain gaps in window trends.
"""
from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from ..contracts import SignalRoute, to_strict_json_value


def _route(value: Any) -> str:
    return str(getattr(value, 'value', value))


def _source(value: Any) -> str:
    route = _route(value)
    if route in {SignalRoute.DIRECT.value, SignalRoute.IDENTITY.value}:
        return 'direct'
    return 'processed' if route == SignalRoute.ARTIFACT_RATE_ONLY.value else route


def _diagnostics(value: Mapping[str, Any], prefix: str = '') -> list[dict[str, Any]]:
    rows = []
    for key, item in value.items():
        name = f'{prefix}.{key}' if prefix else str(key)
        if isinstance(item, Mapping):
            rows.extend(_diagnostics(item, name))
        else:
            rows.append({'field': name, 'value': to_strict_json_value(item)})
    return rows


def _interval_rows(pulse: Any) -> list[dict[str, Any]]:
    routes = pulse.interval_source_routes
    reasons = pulse.interval_rejection_reasons
    rows = []
    for index, value in enumerate(pulse.ppi_s):
        left, right = int(pulse.interval_start_peak_indices[index]), int(pulse.interval_stop_peak_indices[index])
        start, stop = float(pulse.peak_timestamps_s[left]), float(pulse.peak_timestamps_s[right])
        rows.append({'interval_index': index, 'start_peak_index': left, 'stop_peak_index': right,
                     'start_s': start, 'stop_s': stop, 'time_s': (start + stop) / 2,
                     'ppi_s': float(value), 'hr_bpm': 60.0 / float(value) if np.isfinite(value) and value > 0 else np.nan,
                     'valid': bool(pulse.valid_interval_mask[index]),
                     'adjacent': bool(pulse.adjacency_mask[index]), 'wavelength': pulse.wavelength,
                     'source_route': _route(pulse.source_route) if routes is None else str(routes[index]),
                     'run_id': str(pulse.interval_run_ids[index]),
                     'rejection_reason': reasons[index] if index < len(reasons) else None})
    return rows


def _sequence_tables(tables: dict, name: str, extraction: Any) -> None:
    if extraction is None:
        return
    sequence = extraction.sequence
    tiers = getattr(extraction, 'row_tiers', ())
    # One dimension per row keeps 115/146 values and masks individually auditable;
    # channel/local-feature groups avoid a single hundreds-column UI table.
    for column, feature in enumerate(sequence.channel_schema):
        table = tables.setdefault(f'{name} · {feature.split(".")[0]}', [])
        for row, start in enumerate(sequence.start_samples):
            table.append({'window_index': row, 'start_sample': int(start), 'start_s': float(start) / 400,
                          'feature': feature, 'value': float(sequence.values[row, column]),
                          'valid': bool(extraction.value_validity[row, column]),
                          'row_valid': bool(sequence.valid_row_mask[row]),
                          'tier': tiers[row] if tiers else None})


def compute_window_rate_preview(state: Any, report: Any, capture: Mapping[str, Any]) -> dict[str, Any]:
    """Run only the matrix's window/rate kernels using already-detected pulses.

    This exploration product never changes the record, selected representation,
    or quality routing. With routing off, a local full-record interval scope
    avoids inventing assessed quality or additional routing-cell boundaries.
    """
    from ..data.windows import WindowPlan
    from ..features.window_matrix import RATE_WINDOW_NAMES, _rate_features
    from ..peaks import select_reference_wavelength
    from ..quality.routing_timeline import matrix_row_route
    from ..signal.views import CANONICAL_FS_HZ

    product = capture.get('matrix') or capture.get('vector', {})
    direct = product.get('pulses_direct') or state.direct_pulses_per_wavelength
    if not direct:
        return {'status': 'unavailable', 'reason': 'peak_detection_unavailable'}
    direct_pulse = direct[select_reference_wavelength(direct)]
    processed = product.get('pulses_processed') or state.processed_pulses_per_wavelength
    pulses = (direct_pulse,) + ((processed[select_reference_wavelength(processed)],) if processed else ())
    n_samples = len(state.views.x_filter)
    timeline = state.routing_timeline
    routed = timeline is not None
    if not routed:
        # Only the existing interval kernel reads this local scope. It is not a
        # quality result and must not be assigned to state.routing_timeline.
        timeline = SimpleNamespace(n_samples=n_samples, cells=(SimpleNamespace(
            cell_id='preview_full_record', start_sample_400=0, stop_sample_400=n_samples,
            source_route='direct', final_tier='excellent'),))
    plan = WindowPlan(source_record_id=state.row.record_id, **report.window_profiles['engineering'])
    planned = plan.plan(n_samples, CANONICAL_FS_HZ)
    values = np.full((len(planned), len(RATE_WINDOW_NAMES)), np.nan)
    validity = np.zeros(values.shape, dtype=bool)
    windows = []
    for index, window in enumerate(planned):
        start, stop = int(window.start_sample), int(window.end_sample)
        tier, eligible = matrix_row_route(timeline, start, stop) if routed else ('excellent', True)
        windows.append({'window_index': index, 'start_sample': start, 'stop_sample': stop,
                        'tier': tier if routed else 'not_assessed', 'eligible': eligible})
        if eligible:
            values[index], validity[index] = _rate_features(
                pulses[:1] if tier == 'excellent' else pulses, timeline, start, stop)
    return {'status': 'available', 'scope': 'matrix_routing' if routed else 'full_record_direct_quality_not_assessed',
            'source': 'matrix_window_rate_kernels_preview_only', 'windows': windows,
            'values': values, 'validity': validity, 'feature_names': RATE_WINDOW_NAMES}


def build_feature_preview(state: Any, capture: Mapping[str, Any]) -> dict[str, Any]:
    """Convert ``capture`` from the production extractors to workflow preview data.

    ``capture['vector']`` and ``capture['matrix']`` are optional independent
    namespaces. Missing products stay absent: in particular matrix extraction
    does not invent recording-level PRV, optical aggregates or a composite pulse.
    Arrays remain server-side; WorkflowService owns display slicing/serialization.
    """
    tables: dict[str, list] = {}
    signals: dict[str, Any] = {}
    points: dict[str, tuple[Any, Any]] = {}
    groups: dict[str, str] = {}
    styles: dict[str, dict] = {}
    figures: dict[str, str] = {}
    vector = capture.get('vector', {})
    matrix = capture.get('matrix', {})
    rates = capture.get('window_rate', {})
    interval_cache: dict[int, list[dict[str, Any]]] = {}

    def interval_rows(pulse: Any) -> list[dict[str, Any]]:
        if id(pulse) not in interval_cache:
            interval_cache[id(pulse)] = _interval_rows(pulse)
        return interval_cache[id(pulse)]

    metadata: dict[str, Any] = {'retained': state.retained, 'reason': state.reason,
        'feature_count': 0, 'fitted_transforms_applied': False, 'captured_products': {},
        'missing_values': 'Uncomputed products remain absent; NaN and validity are not imputed.',
        'beat_validity_scope': 'Local extractor validity; final route eligibility is applied by the production aggregates.',
        'window_validity_scope': '115-feature extraction has no per-row tier field (shown as null); the 146-feature result carries final row tiers. Matrix base-115 values are before matrix routing masks.',
        'matrix_pulse_semantics': 'Window-selected source intervals, not a fabricated global composite pulse.'}
    waveforms = {'direct': state.views.x_filter} if state.views is not None else {}
    processed = getattr(state, 'processed_views', None)
    if processed is not None and processed.x_ar is not None:
        waveforms['processed'] = processed.x_ar
    elif state.views is not None and state.views.x_ar is not None:
        waveforms['processed'] = state.views.x_ar

    def add_wave(source: str, wavelength: str) -> Any:
        waveform = waveforms.get(source)
        if waveform is None:
            return None
        name = f'{source}_{wavelength}'
        signals[name] = waveform[:, 0 if wavelength == 'RED' else 1]
        groups[name] = 'PPG / amplitude'
        return signals[name]

    for source in waveforms:
        for wavelength in ('RED', 'IR'):
            add_wave(source, wavelength)

    def add_intervals(rows: list[dict], prefix: str, suffix: str = '', symbol: str = 'circle') -> None:
        """Use the same adjacent-peak midpoint for both PPI and its HR conversion."""
        times = np.asarray([row['time_s'] for row in rows])
        connect = symbol == 'circle'
        breaks = [i for i in range(1, len(rows)) if connect and not (
            rows[i]['interval_index'] == rows[i-1]['interval_index'] + 1
            and rows[i]['start_peak_index'] == rows[i-1]['stop_peak_index']
            and rows[i]['run_id'] == rows[i-1]['run_id']
            and rows[i]['source_route'] == rows[i-1]['source_route']
            and rows[i]['adjacent'] and rows[i-1]['adjacent'])]
        # A finite x for each NaN separator keeps the break when the preview's
        # time range is sliced; no original point is moved, removed or imputed.
        for quantity, field, unit in (('PPI', 'ppi_s', 's'), ('HR', 'hr_bpm', 'bpm')):
            name = f'{prefix}_{quantity}{suffix}'
            points[name] = (np.insert(times, breaks, times[breaks]),
                            np.insert(np.asarray([row[field] for row in rows]), breaks, np.nan))
            groups[name] = f'{quantity} / {unit}'
            figures[name] = 'Beatwise PPI / HR'
            styles[name] = {'mode': 'lines+markers' if connect else 'markers', 'connectgaps': False,
                            'line': {'shape': 'linear'}, 'marker': {'symbol': symbol, 'size': 6},
                            'visible': 'legendonly' if suffix == '_invalid' else True}

    def add_pulse(pulse: Any, label: str) -> None:
        wavelength = str(pulse.wavelength)
        prefix = f'{label}_{wavelength}'
        intervals = interval_rows(pulse)
        tables[f'PPI · {prefix}'] = intervals
        origins: list[set[str]] = [set() for _ in pulse.peaks]
        for interval in intervals:
            if interval['source_route'] != 'routing_boundary':
                for index in (interval['start_peak_index'], interval['stop_peak_index']):
                    origins[index].add(_source(interval['source_route']))
        if pulse.interval_source_routes is None:
            origins = [{_source(pulse.source_route)} for _ in pulse.peaks]
        peak_rows = []
        for index, sample in enumerate(pulse.peaks):
            peak_rows.append({'peak_index': index,
                'peak_ordinal': None if pulse.peak_ordinals is None else int(pulse.peak_ordinals[index]),
                'sample': int(sample), 'time_s': float(pulse.peak_timestamps_s[index]),
                'accepted': bool(pulse.accepted_peak_mask[index]), 'confidence': float(pulse.confidence[index]),
                'wavelength': wavelength, 'source': ','.join(sorted(origins[index])) or 'unavailable',
                'detector_id': pulse.detector_id, 'run_id': pulse.detection_run_id})
        tables[f'Peaks · {prefix}'] = peak_rows
        for source in sorted(set().union(*origins)):
            wave = add_wave(source, wavelength)
            if wave is None:
                continue
            for accepted, symbol in ((True, 'circle'), (False, 'x')):
                selection = np.asarray([source in origin and bool(pulse.accepted_peak_mask[index]) == accepted
                                        for index, origin in enumerate(origins)], dtype=bool)
                samples = np.asarray(pulse.peaks)[selection]
                name = f'{prefix}_{source}_peaks_{"accepted" if accepted else "rejected"}'
                points[name] = (np.asarray(pulse.peak_timestamps_s)[selection], wave[samples])
                groups[name] = 'PPG / amplitude'
                styles[name] = {'mode': 'markers', 'marker': {'symbol': symbol, 'size': 7}}
        for source in sorted({row['source_route'] for row in intervals} - {'routing_boundary'}):
            for valid, symbol in ((True, 'circle'), (False, 'x')):
                rows = [row for row in intervals if row['source_route'] == source and row['valid'] == valid
                        and np.isfinite(row['ppi_s'])]
                add_intervals(rows, f'{prefix}_{_source(source)}', '_valid' if valid else '_invalid', symbol)

    # A vector and matrix may reference the same direct detector runs; do not
    # duplicate them merely because two capture namespaces were supplied.
    for source in ('direct', 'processed'):
        pulses = (matrix.get(f'pulses_{source}') or vector.get(f'pulses_{source}')
                  or getattr(state, f'{source}_pulses_per_wavelength', None) or {})
        for pulse in pulses.values():
            add_pulse(pulse, source)
    if vector.get('pulse_used') is not None:
        add_pulse(vector['pulse_used'], 'used')

    registry = vector.get('complete_registry')
    result = vector.get('result')
    if registry is not None:
        values, validity = vector.get('values', {}), vector.get('validity', {})
        final = {} if result is None else dict(zip(result.feature_names, zip(result.values, result.validity)))
        enabled = set(vector['registry'].names)
        for definition in registry.definitions:
            name = definition.name
            tables.setdefault(f'File · {definition.group}', []).append({
                'feature': name, 'value': values.get(name), 'valid': bool(validity.get(name, False)),
                'computed': name in values, 'enabled': name in enabled,
                'vector_value': None if name not in final else float(final[name][0]),
                'vector_valid': None if name not in final else bool(final[name][1]),
                'units': definition.units, 'source': definition.source_signal_view,
                'aggregation': definition.aggregation_rule, 'validity_rule': definition.validity_rule})
        metadata.update(feature_count=0 if result is None else len(result.feature_names), registry_sha256=registry.sha256)
        metadata.update(displayed_file_feature_count=len(registry.names), computed_file_feature_count=len(values))
    prv = vector.get('prv')
    if prv is not None:
        registered = set() if registry is None else set(registry.names)
        tables['PRV · non-predictor diagnostics' if registry is not None else 'PRV · calculated before failure'] = [
            {'feature': name, 'value': value, 'valid': bool(prv.validity[name])}
            for name, value in prv.values.items() if f'prv.{name}' not in registered]
        metadata['prv'] = {'reasons': prv.reasons, 'time_domain_eligible': prv.time_domain_eligible,
            'frequency_domain_eligible': prv.frequency_domain_eligible, 'sample_entropy_eligible': prv.sample_entropy_eligible,
            'source_route': _route(prv.source_route), 'detection_run_id': prv.detection_run_id,
            'interval_timestamps_s': prv.interval_timestamps_s, 'configuration': prv.configuration}

    _sequence_tables(tables, 'Engineering 115', matrix.get('base_engineering') if matrix else vector.get('engineering'))
    _sequence_tables(tables, 'Window matrix 146', matrix.get('engineering'))
    morphology = matrix.get('morphology') or vector.get('morphology')
    morph_pulse = matrix.get('morphology_pulse') or vector.get('morphology_pulse')
    if morphology is not None:
        tables['Morphology · beats'] = [dict(
            peak_index=index, sample=int(sample), time_s=float(morph_pulse.peak_timestamps_s[index]),
            wavelength=morph_pulse.wavelength, accepted=bool(morph_pulse.accepted_peak_mask[index]),
            **{name: float(values[index]) for name, values in morphology.beat_values.items()},
            **{name + '_valid': bool(values[index]) for name, values in morphology.beat_validity.items()})
            for index, sample in enumerate(morph_pulse.peaks)]
        metadata['morphology_reasons'] = morphology.reasons
    optical = vector.get('optical')
    if optical is not None:
        tables['Dual optical · pairing'] = [asdict(row) for row in optical.pairing.rows]
        tables['Dual optical · beats'] = []
        for index, audit in enumerate(optical.beat_audit):
            row = asdict(audit)
            row['optical_reason_codes'] = row.pop('reason_codes')
            row.update(row.pop('pairing'))
            row.update({name: float(values[index]) for name, values in optical.beat_values.items()})
            row.update({name + '_valid': bool(values[index]) for name, values in optical.beat_validity.items()})
            tables['Dual optical · beats'].append(row)
        tables['Dual optical · diagnostics'] = _diagnostics(optical.diagnostics)
        metadata['optical_reasons'] = optical.reasons

    if matrix:
        used, differences, windows = [], [], []
        selected_by_pulse: dict[int, tuple[Any, set[int]]] = {}
        for window in matrix.get('windows', ()):
            header = {key: value for key, value in window.items() if key != 'sources'}
            windows.append(header)
            for selected in window.get('sources', ()):
                pulse = selected['pulse']
                rows = interval_rows(pulse)
                used.extend({**header, **rows[int(index)]} for index in selected['interval_indices'])
                selected_by_pulse.setdefault(id(pulse), (pulse, set()))[1].update(map(int, selected['interval_indices']))
                differences.extend({**header, 'wavelength': pulse.wavelength, 'source_route': _route(pulse.source_route),
                    'left_interval_index': left, 'right_interval_index': right,
                    'left_ppi_s': float(pulse.ppi_s[left]), 'right_ppi_s': float(pulse.ppi_s[right]),
                    'delta_ppi_s': delta, 'run_id': pulse.detection_run_id}
                    for left, right, delta in selected['delta_pairs'])
        tables['Matrix · windows'] = windows
        tables['Matrix · used PPI'] = used
        tables['Matrix · used successive pairs'] = differences
        for pulse, indices in selected_by_pulse.values():
            selected_rows = [interval_rows(pulse)[index] for index in sorted(indices)]
            source, wavelength = _source(pulse.source_route), str(pulse.wavelength)
            add_intervals(selected_rows, f'matrix_used_{source}_{wavelength}')
            wave = add_wave(source, wavelength)
            if wave is not None:
                peak_indices = sorted({row[field] for row in selected_rows for field in ('start_peak_index', 'stop_peak_index')})
                name = f'matrix_used_{source}_{wavelength}_peaks'
                points[name] = (pulse.peak_timestamps_s[peak_indices], wave[pulse.peaks[peak_indices]])
                groups[name] = 'PPG / amplitude'
                styles[name] = {'mode': 'markers', 'marker': {'symbol': 'circle', 'size': 7}}
        extraction = matrix.get('engineering')
        sequence = getattr(extraction, 'sequence', None)
        if sequence is not None:
            rates = {'status': 'available', 'source': 'actual_feature_matrix', 'scope': 'matrix_routing',
                     'windows': windows, 'values': sequence.values, 'feature_names': sequence.channel_schema,
                     'validity': extraction.value_validity & sequence.valid_row_mask[:, None]}
        metadata.update(matrix_schema=getattr(sequence, 'schema_version', None),
                        feature_count=0 if sequence is None else len(sequence.channel_schema),
                        window_count=0 if sequence is None else len(sequence.start_samples))
    if rates:
        metadata['window_rate'] = {key: rates[key] for key in ('status', 'source', 'scope', 'reason') if key in rates}
        if rates['status'] == 'available':
            windows = rates['windows']
            metadata['window_rate']['window_count'] = len(windows)
            valid_windows = np.zeros(len(windows), dtype=bool)
            times = np.asarray([(window['start_sample'] + window['stop_sample']) / 800.0 for window in windows])
            for quantity, unit in (('ppi', 's'), ('hr', 'bpm')):
                for statistic in ('mean', 'median', 'population_sd'):
                    feature = f'local_interval.{quantity}_{statistic}_{unit}'
                    column = rates['feature_names'].index(feature)
                    valid_windows |= rates['validity'][:, column]
                    values = np.where(rates['validity'][:, column], rates['values'][:, column], np.nan)
                    name = f'window_{feature}'
                    points[name] = (times, values)
                    groups[name] = f'{quantity.upper()} / {unit}'
                    figures[name] = 'Window PPI / HR'
                    styles[name] = {'mode': 'lines+markers', 'connectgaps': False,
                                    'line': {'shape': 'linear'}, 'marker': {'size': 7}, 'visible': True}
                    if rates['source'] != 'actual_feature_matrix':
                        tables.setdefault('Window rates · local_interval', []).extend(
                            {'window_index': window['window_index'], 'start_s': window['start_sample'] / 400.0,
                             'stop_s': window['stop_sample'] / 400.0, 'feature': feature,
                             'value': float(value), 'valid': bool(valid)}
                            for window, value, valid in zip(windows, values, rates['validity'][:, column]))
            metadata['window_rate']['valid_window_count'] = int(valid_windows.sum())
            if not valid_windows.any():
                metadata['window_rate']['reason'] = ('no_windows_planned' if not windows else
                    'no_windows_with_sufficient_route_eligible_ppi')
    for namespace, product in (('vector', vector), ('matrix', matrix)):
        if product:
            metadata['captured_products'][namespace] = sorted(product)
            tables[f'{namespace} · diagnostics'] = _diagnostics(product.get('diagnostics', {}))
    return {'tables': tables, 'signals': signals, 'point_traces': points, 'trace_groups': groups,
            'trace_styles': styles, 'trace_figures': figures, 'metadata': metadata}
