"""Read-only V5 stage previews backed by the canonical numerical modules."""
from __future__ import annotations
import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import numpy as np


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class RecordChoice:
    record_id: str
    participant_id: str
    role: str
    class_name: str
    duration_s: float


@dataclass(frozen=True)
class PreviewResult:
    record_id: str
    participant_id: str
    role: str
    class_name: str
    fs_hz: float
    start_s: float
    duration_s: float
    traces: Mapping[str, np.ndarray]
    spectra: Mapping[str, tuple[np.ndarray, np.ndarray]]
    statistics: tuple[Mapping[str, Any], ...]
    stage_rows: tuple[Mapping[str, Any], ...]
    stage_metadata: Mapping[str, Any]

    @property
    def time_s(self) -> np.ndarray:
        count = int(round(self.duration_s * self.fs_hz))
        return self.start_s + np.arange(count, dtype=np.float64) / self.fs_hz


class PipelinePreviewService:
    """Load one manifest record and expose canonical stage outputs.

    Full-record preprocessing is performed before selecting the displayed
    segment, so zero-phase filtering and IMU state are identical to the
    pipeline path rather than a visually convenient slice-only approximation.
    """
    DEFAULT_TRACES = ('raw_red', 'raw_ir', 'filtered_red', 'filtered_ir', 'dynamic_acceleration_magnitude',
                      'angular_velocity_magnitude', 'jerk_magnitude')
    CANONICAL_CONFIG_PATHS = ('configs/presets/baseline.yaml', 'configs/presets/finalcase.yaml', 'configs/presets/feature_vector.yaml',
                              'configs/presets/feature_matrix.yaml', 'configs/presets/fusion.yaml')

    def __init__(self, pipeline_root: str | Path | None = None) -> None:
        inferred = Path(__file__).resolve().parents[3]
        self.pipeline_root = Path(pipeline_root or inferred).resolve()
        self.repository_root = self.pipeline_root.parents[1]

    def config_paths(self) -> tuple[Path, ...]:
        return tuple((path for name in self.CANONICAL_CONFIG_PATHS if (path := (self.pipeline_root / name)).is_file()))

    def study_directories(self) -> tuple[Path, ...]:
        root = self.pipeline_root / 'pipeline_output'
        if not root.is_dir():
            return ()
        studies = {manifest.parent.resolve() for manifest in root.rglob('study_manifest.json') if manifest.is_file()}
        return tuple(sorted(studies, key=lambda path: (path.stat().st_mtime, path.as_posix()), reverse=True))

    def study_plan_paths(self) -> tuple[str, ...]:
        """Return reusable study plans relative to the pipeline root."""
        root = self.pipeline_root / 'configs' / 'studies'
        return tuple((path.relative_to(self.pipeline_root).as_posix() for path in sorted(root.glob('*.yaml')) if path.is_file()))

    def records(self, manifest_path: str | Path | None = None) -> tuple[RecordChoice, ...]:
        rows = self._manifest_rows(manifest_path)
        return tuple((RecordChoice(record_id=str(row.record_id),
                                   participant_id=str(row.participant_id),
                                   role=str(row.role),
                                   class_name=str(row.class_name),
                                   duration_s=float(row.duration_s)) for row in rows))

    def preview(self,
                *,
                config_path: str | Path | None = None,
                config_payload: Mapping[str, Any] | None = None,
                record_id: str,
                start_s: float = 0.0,
                duration_s: float = 20.0,
                trace_names: Iterable[str] | None = None,
                stage_names: Iterable[str] | None = None) -> PreviewResult:
        from ppg_frailty.config import PipelineConfig, canonical_json_bytes, load_config, validate_config_payload
        import hashlib
        from ppg_frailty.data.manifest import load_internal_manifest
        from ppg_frailty.pipeline import PipelinePaths, _load_record
        from ppg_frailty.signal import fit_motion_imu_calibration, roll_pitch_ekf_config_from_resolved
        from ppg_frailty.signal.preprocess import build_signal_views
        if config_payload is not None:
            if config_path is not None:
                raise ValueError('preview accepts config_path or config_payload, not both')
            resolved = validate_config_payload(config_payload)
            config = PipelineConfig(payload=resolved,
                                    source_path='dashboard_resolved_configuration',
                                    sha256=hashlib.sha256(canonical_json_bytes(resolved)).hexdigest())
        else:
            if config_path is None:
                raise ValueError('preview requires config_path or config_payload')
            config_file = self._pipeline_file(config_path)
            config = load_config(config_file)
        manifest_file = self._pipeline_file(str(config.section('manifest')['path']))
        rows = load_internal_manifest(manifest_file)
        matches = [row for row in rows if str(row.record_id) == str(record_id)]
        if len(matches) != 1:
            raise ValueError(f'record_id must identify exactly one manifest row: {record_id}')
        row = matches[0]
        paths = PipelinePaths.discover()
        loaded = _load_record(row, paths, max_samples=None)
        values = np.column_stack(
            (np.asarray(loaded['ppg'], dtype=np.float64), np.asarray(loaded['acc'],
                                                                     dtype=np.float64), np.asarray(loaded['gyro'], dtype=np.float64)))
        signal_config = config.section('signal')
        imu_config = signal_config['imu']
        payload = {**loaded, 'participant_id': str(row.participant_id)}
        calibration_row = None
        calibration = None
        if str(imu_config['gravity_method']) in {
                'calibrated_roll_pitch_ekf', 'profile_a_lowpass_0p3hz', 'sensor_filter_only_no_gravity_removal'
        }:
            candidates = sorted((item
                                 for item in rows if str(item.participant_id) == str(row.participant_id) and str(item.role) == 'B' and
                                 (str(item.qc_status) in {'pass', 'pass_with_warnings'})),
                                key=lambda item: (-float(item.duration_s), str(item.record_id)))
            if not candidates:
                raise ValueError(f'canonical preview requires a same-participant role-B calibration record: {row.participant_id}')
            calibration_row = candidates[0]
            calibration_loaded = _load_record(calibration_row, paths, max_samples=None)
            calibration = fit_motion_imu_calibration(np.asarray(calibration_loaded['acc'], dtype=np.float64),
                                                     np.asarray(calibration_loaded['gyro'], dtype=np.float64),
                                                     participant_id=str(row.participant_id),
                                                     file_id=str(calibration_row.record_id),
                                                     source_role='B',
                                                     fs_hz=float(calibration_row.fs),
                                                     acceleration_unit=str(calibration_loaded['acc_unit']),
                                                     gyroscope_unit=str(calibration_loaded['gyro_unit']),
                                                     config=roll_pitch_ekf_config_from_resolved(imu_config))
            payload['imu_calibration'] = calibration
        views = build_signal_views(payload, config.to_dict())
        all_traces = self._trace_map(values, views)
        requested = self.DEFAULT_TRACES if trace_names is None else tuple(trace_names)
        unknown = sorted(set(requested) - set(all_traces))
        if unknown:
            raise ValueError(f'unknown preview traces: {unknown}')
        fs_hz = float(row.fs)
        if not np.isfinite(start_s) or not np.isfinite(duration_s):
            raise ValueError('preview start/duration must be finite')
        start_index = max(0, int(round(float(start_s) * fs_hz)))
        duration_samples = max(1, int(round(float(duration_s) * fs_hz)))
        end_index = min(values.shape[0], start_index + duration_samples)
        if start_index >= end_index:
            raise ValueError('preview segment lies outside the recording')
        traces = {name: np.asarray(all_traces[name][start_index:end_index], dtype=np.float64) for name in requested}
        spectra = {name: self._spectrum(trace, fs_hz) for name, trace in traces.items()}
        statistics = tuple((self._trace_statistics(name, trace) for name, trace in traces.items()))
        metadata = {
            'config_id': config.config_id,
            'config_hash': config.sha256,
            'source_path': str(row.source_path),
            'source_hash': str(row.source_hash),
            'qc_status': str(row.qc_status),
            'signal_route': views.route.value,
            'gravity_method': views.metadata.get('gravity_method'),
            'imu_status': views.metadata.get('imu_status'),
            'imu_calibration_record_id': None if calibration_row is None else str(calibration_row.record_id),
            'imu_calibration_sha256': None if calibration is None else calibration.artifact_sha256,
            'ppg_qc_metrics': views.metadata.get('ppg_qc_metrics', {}),
            'pipeline_preview_only': True,
            'model_training_executed': False,
            'signal_trace_scope': 'selected_segment',
            'module_stage_scope': 'full_record_after_full_record_preprocessing'
        }
        all_stage_rows = self._module_stage_rows(row=row, views=views, resolved_config=config.to_dict())
        selected_stages = {
            str(value)
            for value in (('source', 'preprocess', 'quality_route', 'pulse_ppi', 'morphology', 'engineering', 'representation_model',
                           'aggregation') if stage_names is None else tuple(stage_names))
        }
        stage_rows = tuple((item for item in all_stage_rows if str(item['stage']) in selected_stages))
        return PreviewResult(record_id=str(row.record_id),
                             participant_id=str(row.participant_id),
                             role=str(row.role),
                             class_name=str(row.class_name),
                             fs_hz=fs_hz,
                             start_s=start_index / fs_hz,
                             duration_s=(end_index - start_index) / fs_hz,
                             traces=traces,
                             spectra=spectra,
                             statistics=statistics,
                             stage_rows=stage_rows,
                             stage_metadata=metadata)

    def study_table(self, study_dir: str | Path, relative_path: str) -> tuple[list[dict[str, Any]], list[str]]:
        import pandas as pd
        root = self._study_dir(study_dir)
        target = (root / relative_path).resolve()
        target.relative_to(root)
        if target.suffix.lower() == '.csv':
            frame = pd.read_csv(target)
        elif target.suffix.lower() == '.parquet':
            frame = pd.read_parquet(target)
        elif target.suffix.lower() == '.json':
            frame = pd.read_json(target)
        else:
            raise ValueError('study table must be CSV, Parquet, or JSON')
        return (frame.head(500).to_dict(orient='records'), [str(value) for value in frame.columns])

    def study_table_paths(self, study_dir: str | Path) -> tuple[str, ...]:
        root = self._study_dir(study_dir)
        paths: list[str] = []
        for suffix in ('*.csv', '*.parquet', '*.json'):
            for path in root.rglob(suffix):
                if path.is_file():
                    paths.append(path.relative_to(root).as_posix())
        return tuple(sorted(set(paths)))

    def study_figure_paths(self, study_dir: str | Path) -> tuple[str, ...]:
        root = self._study_dir(study_dir)
        paths = [
            path.relative_to(root).as_posix() for path in root.rglob('*')
            if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.svg'}
        ]
        return tuple(sorted(paths))

    def study_figure_data_uri(self, study_dir: str | Path, relative_path: str) -> str:
        root = self._study_dir(study_dir)
        target = (root / relative_path).resolve()
        target.relative_to(root)
        if not target.is_file() or target.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.svg'}:
            raise ValueError('study figure must be PNG, JPEG, or SVG')
        mime = mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
        return f'data:{mime};base64,' + base64.b64encode(target.read_bytes()).decode('ascii')

    def _manifest_rows(self, manifest_path: str | Path | None) -> list[Any]:
        from ppg_frailty.data.manifest import load_internal_manifest
        path = self._pipeline_file(manifest_path or 'manifests/internal_records_v2.csv')
        return load_internal_manifest(path)

    def _pipeline_file(self, value: str | Path) -> Path:
        candidate = Path(value)
        target = candidate.resolve() if candidate.is_absolute() else (self.pipeline_root / candidate).resolve()
        target.relative_to(self.pipeline_root)
        if not target.is_file():
            raise FileNotFoundError(target)
        return target

    def _study_dir(self, value: str | Path) -> Path:
        candidate = Path(value)
        target = candidate.resolve() if candidate.is_absolute() else (self.pipeline_root / candidate).resolve()
        allowed = tuple(((self.pipeline_root / name).resolve() for name in ('pipeline_output', 'report_output')))
        if not any((_is_relative_to(target, root) for root in allowed)):
            raise ValueError('study/report directory must remain in pipeline_output or report_output')
        if not target.is_dir():
            raise FileNotFoundError(target)
        return target

    @staticmethod
    def _trace_map(values: np.ndarray, views: Any) -> dict[str, np.ndarray]:
        processed = views.imu_processed

        def first(*names: str) -> np.ndarray:
            for name in names:
                if name in processed:
                    value = np.asarray(processed[name], dtype=np.float64)
                    if value.ndim == 1:
                        return value
            raise KeyError(f'canonical IMU output missing any of: {names}')

        return {
            'raw_red':
            values[:, 0],
            'raw_ir':
            values[:, 1],
            'filtered_red':
            np.asarray(views.x_filter[:, 0], dtype=np.float64),
            'filtered_ir':
            np.asarray(views.x_filter[:, 1], dtype=np.float64),
            'dynamic_acceleration_magnitude':
            first('dynamic_magnitude', 'dynamic_acc_magnitude', 'acc_dynamic_magnitude', 'a_dynamic_magnitude'),
            'angular_velocity_magnitude':
            first('gyro_magnitude', 'angular_velocity_magnitude', 'omega'),
            'jerk_magnitude':
            first('jerk_magnitude', 'jerk')
        }

    @staticmethod
    def _module_stage_rows(*, row: Any, views: Any, resolved_config: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        """Build readable, non-training previews from the real public modules."""
        from ppg_frailty.data.windows import WindowPlan
        from ppg_frailty.features.engineering import extract_engineering_features
        from ppg_frailty.module_registry import resolve_peak_detector_config, resolve_window_config
        from ppg_frailty.peaks import select_reference_wavelength
        from ppg_frailty.signal.prv import PrvConfig
        from ppg_frailty.signal import compute_prv, detect_pulses_per_wavelength, extract_dual_optical, extract_morphology
        output: list[dict[str, Any]] = []

        def add(stage: str, metric: str, value: Any, status: str = 'available') -> None:
            if isinstance(value, np.generic):
                value = value.item()
            output.append({'stage': stage, 'metric': metric, 'value': value, 'status': status})

        sample_count = int(views.x_filter.shape[0])
        duration_s = sample_count / float(row.fs)
        add('source', 'record_id', str(row.record_id))
        add('source', 'participant_id', str(row.participant_id))
        add('source', 'role', str(row.role))
        add('source', 'class_name', str(row.class_name))
        add('source', 'sample_count', sample_count)
        add('source', 'duration_s', duration_s)
        add('preprocess', 'signal_route', views.route.value)
        add('preprocess', 'gravity_method', views.metadata.get('gravity_method'))
        add('preprocess', 'imu_valid_fraction', views.metadata.get('imu_valid_fraction'))
        for name, value in sorted(dict(views.metadata.get('ppg_qc_metrics', {})).items()):
            add('preprocess', f'ppg_qc.{name}', value)
        quality = resolved_config.get('quality', {})
        quality_mode = str(quality.get('mode', 'off')) if isinstance(quality, Mapping) else 'invalid'
        add('quality_route', 'quality_mode', quality_mode)
        add('quality_route', 'segment_state',
            'not_applied_thresholds_deferred' if quality_mode == 'off' else 'see_formal_quality_diagnostics',
            'N/A' if quality_mode == 'off' else 'available')
        add('quality_route', 'classification_action', 'keep_unchanged')
        pulse = None
        try:
            detector = resolve_peak_detector_config(resolved_config.get('signal', {}))
            add('pulse_ppi', 'detector_id', detector['detector_id'])
            add('pulse_ppi', 'min_observation_sec', detector['min_observation_sec'])
            add('pulse_ppi', 'min_peaks', detector['min_peaks'])
            pulses_per_wavelength = detect_pulses_per_wavelength(views,
                                                                 detector_id=detector['detector_id'],
                                                                 min_observation_sec=detector['min_observation_sec'],
                                                                 min_peaks=detector['min_peaks'],
                                                                 detector_parameters=detector.get('parameters'))
            pulse = pulses_per_wavelength[select_reference_wavelength(pulses_per_wavelength)]
            add('pulse_ppi', 'wavelength', pulse.wavelength)
            add('pulse_ppi', 'detected_peak_count', int(pulse.peaks.size))
            add('pulse_ppi', 'accepted_peak_count', int(np.count_nonzero(pulse.accepted_peak_mask)))
            add('pulse_ppi', 'valid_interval_count', int(np.count_nonzero(pulse.valid_interval_mask)))
            add('pulse_ppi', 'detection_run_id', pulse.detection_run_id)
            for wavelength in ('RED', 'IR'):
                channel_pulse = pulses_per_wavelength[wavelength]
                prefix = f'{wavelength.lower()}.'
                add('pulse_ppi', prefix + 'detected_peak_count', int(channel_pulse.peaks.size))
                add('pulse_ppi', prefix + 'accepted_peak_count', int(np.count_nonzero(channel_pulse.accepted_peak_mask)))
                add('pulse_ppi', prefix + 'selected_polarity', int(channel_pulse.selected_polarity))
                add('pulse_ppi', prefix + 'detector_score', float(channel_pulse.detector_score))
                add('pulse_ppi', prefix + 'detector_coverage', float(channel_pulse.detector_coverage))
            prv = compute_prv(pulse,
                              observation_duration_s=duration_s,
                              role=str(row.role),
                              route=views.route,
                              q_rate_qualified=quality_mode != 'route',
                              config=PrvConfig.from_mapping(resolved_config.get('features', {})))
            for name, value in sorted(prv.values.items()):
                add('pulse_ppi', f'prv.{name}', value if np.isfinite(value) else None,
                    'available' if prv.validity.get(name, False) else 'unavailable')
            add('pulse_ppi', 'prv_reasons', list(prv.reasons))
            if views.route.value in {'direct_x_filter', 'identity_direct'}:
                optical = extract_dual_optical(views.x_native, views.x_filter, pulses_per_wavelength, route=views.route)
                add('dual_optical', 'schema_version', optical.schema_version)
                add('dual_optical', 'reference_wavelength', optical.pairing.reference_wavelength)
                add('dual_optical', 'paired_cycle_count', len(optical.pairing.paired_rows))
                add('dual_optical', 'paired_valid_optical_count', int(sum((row.optical_valid for row in optical.beat_audit))))
                for name, value in sorted(optical.aggregate_values.items()):
                    valid = bool(optical.aggregate_validity[name])
                    add('dual_optical', name, value if valid else None, 'available' if valid else 'unavailable')
        except Exception as exc:
            add('pulse_ppi', 'module_error', f'{type(exc).__name__}: {exc}', 'failed')
        if pulse is not None:
            try:
                morphology = extract_morphology(views.x_filter, pulse, route=views.route, fs_hz=float(row.fs))
                for name, value in sorted(morphology.aggregate_values.items()):
                    add('morphology', name, value if np.isfinite(value) else None,
                        'available' if morphology.aggregate_validity.get(name, False) else 'unavailable')
                add('morphology', 'reasons', list(morphology.reasons))
            except Exception as exc:
                add('morphology', 'module_error', f'{type(exc).__name__}: {exc}', 'failed')
        else:
            add('morphology', 'status', 'pulse_detection_unavailable', 'N/A')
        try:
            windows = resolve_window_config(resolved_config['windows'])['engineering']
            plan = WindowPlan(source_record_id=str(row.record_id), **windows)
            engineering = extract_engineering_features(views, plan=plan)
            feature_values = np.asarray(engineering.sequence.values, dtype=np.float64)
            validity = np.asarray(engineering.value_validity, dtype=bool)
            add('engineering', 'window_count', int(feature_values.shape[0]))
            add('engineering', 'feature_count', int(feature_values.shape[1]))
            add('engineering', 'available_feature_fraction',
                float(np.mean(validity)) if validity.size else None, 'available' if validity.size else 'N/A')
            for index, name in enumerate(engineering.sequence.channel_schema):
                usable = validity[:, index] & np.isfinite(feature_values[:, index]) if validity.size else np.zeros(0, dtype=bool)
                add('engineering', f'median.{name}',
                    float(np.median(feature_values[usable, index])) if np.any(usable) else None,
                    'available' if np.any(usable) else 'unavailable')
        except Exception as exc:
            add('engineering', 'module_error', f'{type(exc).__name__}: {exc}', 'failed')
        representation_mode = str(resolved_config.get('representation_mode', ''))
        add('representation_model', 'representation_mode', representation_mode)
        add('representation_model', 'preview_status', 'pre_fit_inputs_only_model_fit_not_executed', 'N/A')
        add('aggregation', 'hierarchy', resolved_config.get('aggregation', {}).get('hierarchy'))
        add('aggregation', 'preview_status', 'requires_outer_oof_predictions_not_executed', 'N/A')
        return tuple(output)

    @staticmethod
    def _spectrum(values: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
        from scipy import signal
        array = np.asarray(values, dtype=np.float64)
        if array.size < 8:
            return (np.empty(0), np.empty(0))
        frequency, power = signal.welch(array,
                                        fs=fs_hz,
                                        nperseg=min(array.size, int(fs_hz * 8)),
                                        detrend='constant',
                                        scaling='density')
        keep = frequency <= min(40.0, fs_hz / 2.0)
        return (frequency[keep], power[keep])

    @staticmethod
    def _trace_statistics(name: str, values: np.ndarray) -> dict[str, Any]:
        array = np.asarray(values, dtype=np.float64)
        return {
            'trace': name,
            'samples': int(array.size),
            'mean': float(np.mean(array)),
            'std': float(np.std(array)),
            'minimum': float(np.min(array)),
            'maximum': float(np.max(array)),
            'rms': float(np.sqrt(np.mean(np.square(array))))
        }
