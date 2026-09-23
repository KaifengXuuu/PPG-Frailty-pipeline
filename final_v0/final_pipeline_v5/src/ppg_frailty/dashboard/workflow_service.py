"""Notebook-style, read-only execution of the existing numerical workflow.

Arrays stay on the server. Each browser session has a small recording LRU;
stage identities include their upstream identity and the controls they consume.
Changing a plot range never reruns an algorithm. No stage fits a classifier,
SQI calibrator, or fold transform; participant B calibration is preprocessing.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import copy, deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np


STAGES = ("input", "ppg", "imu", "motion", "quality", "denoiser", "features", "representation", "model", "aggregation")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def _clone_state(state: Any) -> Any:
    """Copy mutable bookkeeping without duplicating full-record signal arrays."""
    result = copy(state)
    for name in state._DICT_FIELDS:
        setattr(result, name, deepcopy(getattr(state, name)))
    return result


def _prediction_fields(config: Mapping[str, Any], probabilities: Any) -> dict[str, Any]:
    """Name the highest existing probability without altering model outputs."""
    if not len(probabilities):
        return {}
    index = int(np.argmax(probabilities))
    manifest = config["manifest"]
    return {"predicted_class_id": manifest["class_id_order"][index],
            "predicted_class_name": manifest["class_name_order"][index]}


class WorkflowService:
    """Execute a requested stage after automatically filling missing upstreams."""

    def __init__(self, pipeline_root: str | Path | None = None, *, max_sessions: int = 4, max_records: int = 12):
        self.pipeline_root = Path(pipeline_root or Path(__file__).resolve().parents[3]).resolve()
        self.repository_root = self.pipeline_root.parents[1]
        self.max_sessions, self.max_records = max_sessions, max_records
        self._sessions: OrderedDict[str, OrderedDict[str, dict[str, Any]]] = OrderedDict()
        self._lock = RLock()

    def clear(self, session_id: str) -> None:
        """Release one browser's temporary arrays; never touch persistent cache."""
        with self._lock:
            self._sessions.pop(str(session_id), None)

    def cached_previews(self, session_id: str, record_id: str, start_s: float = 0.0,
                        duration_s: float = 20.0) -> dict[str, Any]:
        """Read surviving outputs after a failed stage, without doing any work."""
        with self._lock:
            entries = self._sessions.get(str(session_id), {}).get(str(record_id), {})
            return {stage: self._preview(stage, entries[stage]["value"], start_s, duration_s)
                    for stage in STAGES if stage in entries}

    def analyse(self, stage: str, config_payload: Mapping[str, Any], record_id: str,
                session_id: str = "default", selections: Mapping[str, Any] | None = None,
                start_s: float = 0.0, duration_s: float = 20.0) -> dict[str, Any]:
        if stage not in STAGES:
            raise ValueError(f"unknown workflow stage: {stage}")
        config, selected = deepcopy(dict(config_payload)), dict(selections or {})
        with self._lock:
            entries, computed, reused = self._ensure(stage, config, str(record_id), str(session_id), selected)
            previews = {
                name: self._preview(name, entries[name]["value"], start_s, duration_s)
                for name in STAGES[:STAGES.index(stage) + 1]
            }
            return {"requested_stage": stage, "record_id": str(record_id), "computed_stages": computed,
                    "reused_stages": reused, "previews": previews}

    def _ensure(self, stage: str, config: dict[str, Any], record_id: str, session_id: str,
                selected: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
        records = self._sessions.setdefault(session_id, OrderedDict())
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)
        entries = records.setdefault(record_id, {})
        records.move_to_end(record_id)
        while len(records) > self.max_records:
            records.popitem(last=False)
        computed, reused = [], []
        upstream = ""
        context: dict[str, Any] = {}
        for name in STAGES[:STAGES.index(stage) + 1]:
            controls = self._stage_controls(name, config, selected, record_id)
            if name in {"imu", "model"}:
                rows = context["input"]["rows"]
                participant = context["input"]["row"].participant_id
                wanted = set(map(str, selected.get("record_ids") or ()))
                participants = {participant, *(row.participant_id for row in rows if str(row.record_id) in wanted)}
                sources = [self._file_identity(Path(row.source_path) if Path(row.source_path).is_absolute()
                           else self.repository_root / row.source_path) for row in rows
                           if (name == "imu" and row.role == "B" and row.participant_id == participant)
                           or (name == "model" and (str(row.record_id) in wanted
                               or (row.role == "B" and row.participant_id in participants)))]
                controls = (controls, sources)
            key = _digest((upstream, controls))
            cached = entries.get(name)
            if name == "input" and cached is not None:
                path = cached["value"]["loaded"].get("source_path")
                if cached.get("source_identity") != self._file_identity(path):
                    cached = None
            if cached is None or cached["key"] != key:
                # Drop stale descendants immediately, including previously shown model results.
                for stale in STAGES[STAGES.index(name):]:
                    entries.pop(stale, None)
                result = getattr(self, "_" + name)(config, selected, context, record_id, session_id)
                entries[name] = {"key": key, "value": result}
                if name == "input":
                    entries[name]["source_identity"] = self._file_identity(result["loaded"].get("source_path"))
                computed.append(name)
            else:
                reused.append(name)
            context[name] = entries[name]["value"]
            upstream = key
        return entries, computed, reused

    def _path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.pipeline_root / path).resolve()

    def _file_identity(self, value: Any) -> Any:
        if not value:
            return None
        path = self._path(str(value))
        if not path.exists():
            return str(path)
        stat = path.stat()
        return (str(path), stat.st_size, stat.st_mtime_ns)

    def _stage_controls(self, stage: str, config: Mapping[str, Any], selected: Mapping[str, Any], record_id: str) -> Any:
        signal, quality, artifact = config["signal"], config["quality"], config["artifact"]
        if stage == "input":
            files = selected.get("files", [])
            return (record_id, config["manifest"], selected.get("participant_id"), files,
                    [(row.get("path"), self._file_identity(row.get("path"))) for row in files],
                    self._file_identity(selected.get("source_path")),
                    self._file_identity(config["manifest"].get("path")))
        if stage == "ppg":
            return (signal["ppg_filter"], signal["gap_repair"], quality.get("flatline_duration_s", 1.0))
        if stage == "imu":
            return (signal["imu"], signal["accelerometer_input_unit"], signal["gyroscope_input_unit"],
                    selected.get("calibration_record_id"), self._file_identity(selected.get("calibration_path")))
        if stage == "motion":
            return (artifact.get("motion_detector_enabled"), artifact.get("motion_detector"), config["routing"],
                    self._file_identity(selected.get("motion_bundle") or artifact.get("motion_detector", {}).get("evidence_path")))
        if stage == "quality":
            return ({key: val for key, val in quality.items() if key != "window_selection"},
                    signal.get("peak_detector"), config["routing"], self._file_identity(selected.get("sqi_artifact")))
        if stage == "denoiser":
            return (artifact, config["representation_mode"])
        if stage == "features":
            return (config["features"], signal.get("peak_detector"), config["windows"].get("engineering"),
                    config["representation_mode"])
        if stage == "representation":
            return (config["representation_mode"], config["windows"], signal["normalization"],
                    signal["dl_resampling"], quality.get("window_selection"),
                    {key: config["aggregation"].get(key) for key in ("quality_weighting", "quality_weight_source")})
        if stage == "model":
            value = selected.get("model_bundle") or selected.get("model_export")
            root = self._path(value) if value else None
            weight_files = [] if root is None or not root.is_dir() else [self._file_identity(path)
                for path in sorted(root.rglob("*")) if path.is_file()]
            return (config["model"], config["training"], config["roles"], selected, weight_files)
        return config["aggregation"]

    def _input(self, config: dict[str, Any], selected: dict[str, Any], context: dict[str, Any],
               record_id: str, session_id: str) -> dict[str, Any]:
        from ..data.manifest import load_internal_manifest
        from ..pipeline import PipelinePaths, _load_record

        files = list(selected.get("files") or [])
        if selected.get("source_path") and not files:
            files = [{"path": selected["source_path"], "file_id": record_id,
                      "role": selected.get("role", "B"), "label": selected.get("label")}]
        if files:
            rows = [self._csv_row(item, selected.get("participant_id", "participant"),
                                 class_names=tuple(config["manifest"]["class_name_order"])) for item in files]
            labels = {row.class_id for row in rows if row.class_id >= 0}
            if len(labels) > 1:
                raise ValueError("All labelled files for one participant must share one label.")
            if labels:
                label = next(iter(labels))
                rows = [replace(row, class_id=label, class_name=config["manifest"]["class_name_order"][label])
                        if row.class_id < 0 else row for row in rows]
        else:
            rows = load_internal_manifest(self._path(config["manifest"]["path"]))
        matches = [row for row in rows if str(row.record_id) == record_id]
        if len(matches) != 1:
            raise ValueError(f"select one recording to preview: {record_id}")
        row = matches[0]
        source = Path(row.source_path)
        # Explicit CSVs may live outside the repository; reuse the identical loader/QC.
        root = source.parent if source.is_absolute() else self.repository_root
        loaded = _load_record(row, PipelinePaths(self.pipeline_root, root), max_samples=None)
        loaded["participant_id"] = str(row.participant_id)
        return {"row": row, "rows": rows, "loaded": loaded,
                "signals": {"RED": loaded["ppg"][:, 0], "IR": loaded["ppg"][:, 1],
                            **{name: loaded["acc"][:, index] for index, name in enumerate(("AX", "AY", "AZ"))},
                            **{name: loaded["gyro"][:, index] for index, name in enumerate(("GX", "GY", "GZ"))}},
                "metadata": {"record_id": record_id, "participant_id": row.participant_id, "role": row.role,
                             "samples": row.n_samples, "duration_s": row.duration_s, "source": str(row.source_path)}}

    def _csv_row(self, item: Mapping[str, Any], participant_id: str,
                 *, class_names: tuple[str, ...] = ()) -> Any:
        from ..contracts import ManifestRow
        from ..v5.inference_service import _CHANNELS, _UNITS, _csv_identity, _label_id
        path = self._path(item["path"])
        content_hash, count = _csv_identity(path)
        label = _label_id(item.get("label"), class_names)
        class_id = -1 if label is None else label
        return ManifestRow(record_id=str(item.get("file_id") or path.stem), participant_id=str(participant_id),
                           class_id=class_id, class_name="unlabelled" if class_id < 0 else class_names[class_id],
                           class_name_provenance_alias="dashboard_input", class_source="user_supplied",
                           label_record_id="", role=str(item.get("role", "B")), source_path=str(path),
                           source_hash=content_hash, source_version="dashboard_input", fs=400.0,
                           n_samples=count, duration_s=count / 400.0, channel_schema=_CHANNELS,
                           channel_units=dict(_UNITS), synchrony_status="row_aligned_eight_channel_fixed_grid_no_timestamp",
                           reference_available=False, qc_status="pass", qc_reasons=(), manifest_version="dashboard_input")

    def _ppg(self, config: dict[str, Any], selected: dict[str, Any], context: dict[str, Any],
             record_id: str, session_id: str) -> dict[str, Any]:
        from ..signal.preprocess import preprocess_ppg_pair
        from .signal_preview import amplitude_spectra, power_spectra
        signal, source = config["signal"], context["input"]["loaded"]
        filtering = signal["ppg_filter"]
        result = preprocess_ppg_pair(source["ppg"], fs_hz=source["fs_hz"], timestamps_s=source.get("timestamps_s"),
                                     max_gap_samples=signal["gap_repair"]["max_gap_samples"],
                                     flatline_sec=config["quality"].get("flatline_duration_s", 1.0),
                                     filter_low_hz=filtering["low_hz"], filter_high_hz=filtering["high_hz"],
                                     filter_order=filtering["order"])
        native, filtered, qc = result
        signals, groups, frequency_groups = {}, {}, {}
        for index, channel in enumerate(("RED", "IR")):
            for prefix, values in (("raw", source["ppg"]), ("native", native), ("filtered", filtered)):
                name = f"{prefix}_{channel}"
                signals[name] = values[:, index]
                groups[name] = "PPG / raw counts"
                frequency_groups[name] = f"{channel} · PSD / raw counts²/Hz"
        spectrum = power_spectra(signals, fs_hz=source["fs_hz"])
        fft = amplitude_spectra(signals, fs_hz=source["fs_hz"])
        return {"ppg_result": result, "signals": signals, "trace_groups": groups,
                "trace_styles": {f"native_{channel}": {"visible": "legendonly"} for channel in ("RED", "IR")},
                "trace_figures": {name: "Filtered PPG" if name.startswith("filtered_") else "Raw PPG" for name in signals},
                "frequency_traces": spectrum["frequency_traces"], "frequency_groups": frequency_groups,
                "fft_traces": fft["fft_traces"],
                "fft_groups": {name: f"{name.rsplit('_', 1)[-1]} · FFT amplitude / raw counts" for name in signals},
                "metadata": {"filter": filtering, "repaired_samples": int(np.count_nonzero(qc.repair_mask)),
                             "source_valid_fraction": float(np.mean(qc.source_valid_mask)), "qc": qc.metrics,
                             "signal_sources": {"raw": "original input, gaps preserved", "native": "gap-repaired input",
                                                "filtered": "actual pipeline output"},
                             "spectrum": spectrum["spectrum_metadata"], "fft": fft["fft_metadata"]}}

    def _imu(self, config: dict[str, Any], selected: dict[str, Any], context: dict[str, Any],
             record_id: str, session_id: str) -> dict[str, Any]:
        from ..pipeline import PipelinePaths, _load_record
        from ..signal.imu import convert_acceleration, convert_gyro
        from ..signal.motion_imu import fit_motion_imu_calibration, _convert_profile_acceleration
        from ..signal.preprocess import build_signal_views, roll_pitch_ekf_config_from_resolved
        from .signal_preview import amplitude_spectra, power_spectra
        source, row = dict(context["input"]["loaded"]), context["input"]["row"]
        imu = config["signal"]["imu"]
        calibration_id = None
        if imu["gravity_method"] in {"calibrated_roll_pitch_ekf", "profile_a_lowpass_0p3hz",
                                     "sensor_filter_only_no_gravity_removal"}:
            if selected.get("calibration_path"):
                calibration_row = self._csv_row({"path": selected["calibration_path"], "role": "B"}, row.participant_id)
            else:
                candidates = [item for item in context["input"]["rows"]
                              if item.participant_id == row.participant_id and item.role == "B"
                              and item.qc_status in {"pass", "pass_with_warnings"}]
                if selected.get("calibration_record_id"):
                    candidates = [item for item in candidates if item.record_id == selected["calibration_record_id"]]
                if not candidates:
                    raise ValueError("Select a B recording from the same participant for IMU calibration.")
                calibration_row = sorted(candidates, key=lambda item: (-float(item.duration_s), str(item.record_id)))[0]
            calibration_id = str(calibration_row.record_id)
            path = Path(calibration_row.source_path)
            paths = PipelinePaths(self.pipeline_root, path.parent if path.is_absolute() else self.repository_root)
            loaded = source if calibration_id == record_id else _load_record(calibration_row, paths, max_samples=None)
            source["imu_calibration"] = fit_motion_imu_calibration(
                loaded["acc"], loaded["gyro"], participant_id=str(row.participant_id), file_id=calibration_id,
                source_role="B", fs_hz=float(calibration_row.fs), acceleration_unit=loaded["acc_unit"],
                gyroscope_unit=loaded["gyro_unit"], config=roll_pitch_ekf_config_from_resolved(imu))
        views = build_signal_views(source, config, ppg_result=context["ppg"]["ppg_result"])
        raw_acc = (_convert_profile_acceleration(source["acc"], source["acc_unit"],
                    gravity_mps2=roll_pitch_ekf_config_from_resolved(imu).gravity_mps2)
                   if calibration_id is not None else convert_acceleration(source["acc"], source["acc_unit"]))
        raw_gyro = convert_gyro(source["gyro"], source["gyro_unit"])
        traces = {f"raw_{name}_{axis}": values[:, index]
                  for name, values in (("acc_mps2", raw_acc), ("gyro_rads", raw_gyro))
                  for index, axis in enumerate("xyz")}
        for name, values in views.imu_processed.items():
            values = np.asarray(values)
            if values.ndim == 1:
                traces[name] = values
            elif values.ndim == 2 and values.shape[1] == 3:
                traces.update({f"{name}_{axis}": values[:, index] for index, axis in enumerate("xyz")})
        groups, frequency_groups = {}, {}
        for name in traces:
            axis = name.rsplit("_", 1)[-1]
            if name.startswith(("raw_acc_mps2_", "acc_mps2_", "dynamic_acc_mps2_", "gravity_mps2_")):
                group = f"Acceleration {axis.upper()} / m/s²"
            elif name.startswith(("raw_gyro_rads_", "gyro_rads_")):
                group = f"Angular velocity {axis.upper()} / rad/s"
            elif name.startswith("jerk_"):
                group = "Acceleration change / m/s³"
            elif name in {"roll_rad", "pitch_rad"}:
                group = "Orientation / rad"
            elif name in {"acc_magnitude", "dynamic_magnitude"}:
                group = "Acceleration magnitude / m/s²"
            elif name == "gyro_magnitude":
                group = "Angular velocity magnitude / rad/s"
            else:
                group = "Validity / state"
            groups[name] = group
            if name.startswith(("raw_acc_mps2_", "acc_mps2_", "dynamic_acc_mps2_", "raw_gyro_rads_", "gyro_rads_")):
                frequency_groups[name] = group.replace(" / ", " · PSD / (") + ")²/Hz"
        frequency_signals = {name: traces[name] for name in frequency_groups}
        spectrum = power_spectra(frequency_signals)
        fft = amplitude_spectra(frequency_signals)
        return {"views": views, "signals": traces, "trace_groups": groups,
                "frequency_traces": spectrum["frequency_traces"], "frequency_groups": frequency_groups,
                "fft_traces": fft["fft_traces"],
                "fft_groups": {name: groups[name].replace(" / ", " · FFT amplitude / ") for name in frequency_groups},
                "metadata": {"gravity_method": views.metadata["gravity_method"], "calibration_record_id": calibration_id,
                             "imu_status": views.metadata["imu_status"], "imu_valid_fraction": views.metadata["imu_valid_fraction"],
                             "original_units": {"acceleration": source["acc_unit"], "gyroscope": source["gyro_unit"]},
                             "raw_preview_units": {"acceleration": "m/s²", "gyroscope": "rad/s"},
                             "processed_source": "actual imu_processed arrays, including the selected gravity policy",
                             "spectrum": spectrum["spectrum_metadata"], "fft": fft["fft_metadata"]}}

    def _motion(self, config: dict[str, Any], selected: dict[str, Any], context: dict[str, Any],
                record_id: str, session_id: str) -> dict[str, Any]:
        from ..quality.motion_bundle_adapter import (infer_reused_motion_windows, load_reused_motion_detector,
                                                     motion_recording_from_signal_views, resolve_reused_motion_detector_config)
        if not config["artifact"].get("motion_detector_enabled", False):
            return {"series": None, "detector": None, "metadata": {"enabled": False, "state": "off"}}
        payload = dict(config["artifact"]["motion_detector"])
        evidence = selected.get("motion_bundle") or payload.get("evidence_path")
        if not evidence:
            raise ValueError("Select existing motion weights/evidence; Analyse never trains a detector.")
        evidence = self._path(evidence)
        payload.update(enabled=True, evidence_path=evidence,
                       expected_evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())
        detector = load_reused_motion_detector(resolve_reused_motion_detector_config(payload))
        row, views = context["input"]["row"], context["imu"]["views"]
        # The weights stay fixed; changed IMU controls are intentional exploratory inputs.
        runtime_hash = views.metadata.get("imu_diagnostics", {}).get("ekf_config_sha256")
        adapter_detector = replace(detector, ekf_config_sha256=runtime_hash) if runtime_hash else detector
        recording = motion_recording_from_signal_views(views, detector=adapter_detector, record_id=record_id,
                                                        participant_id=str(row.participant_id), role=str(row.role))
        series = infer_reused_motion_windows(detector, recording)
        rows = [asdict(item) for item in series.decisions]
        valid = [item for item in series.decisions if item.probability is not None]
        return {"series": series, "detector": detector, "tables": {"windows": rows},
                "point_traces": {"motion_probability": ([item.centre_sample_400 / 400.0 for item in valid],
                                                        [item.probability for item in valid])},
                "metadata": {"enabled": True, "threshold": series.threshold, "window_count": len(rows),
                             "file_median_probability": series.file_median_probability_diagnostic,
                             "preprocessing_changed_from_training": runtime_hash != detector.ekf_config_sha256}}

    @staticmethod
    def _peak_config(config: Mapping[str, Any]) -> dict[str, Any]:
        from ..experiment import _peak_detection_runtime_kwargs
        from ..module_registry import resolve_peak_detector_config
        return _peak_detection_runtime_kwargs(resolve_peak_detector_config(config["signal"]))

    def _quality(self, config: dict[str, Any], selected: dict[str, Any], context: dict[str, Any],
                 record_id: str, session_id: str) -> dict[str, Any]:
        from ..experiment import _slice_global_pulse_for_routing, _slice_signal_views_for_routing
        from ..peaks import select_reference_wavelength
        from ..peaks.resolver import detect_pulses_per_wavelength
        from ..quality.routing_timeline import RoutingEvidence, build_routing_windows
        from ..signal.sqi import SqiCalibrator, SqiConfig, SqiDiagnosticConfig, evaluate_quality, evaluate_quality_diagnostics
        views, mode = context["imu"]["views"], config["quality"].get("mode", "off")
        sqi_mapping = {key: val for key, val in config["quality"].items() if key != "window_selection"}
        sqi = SqiConfig.from_quality_mapping(sqi_mapping)
        if mode != "route":
            sqi = replace(sqi, calibrator="fixed_formula_thresholds_v1")
        calibrator = None
        if mode == "route" and sqi.calibrator == "outer_train_empirical_quantiles_v1":
            path = selected.get("sqi_artifact")
            if not path:
                raise ValueError("Select an existing fitted SQI calibrator; Analyse never fits the current participant.")
            payload = json.loads(self._path(path).read_text(encoding="utf-8"))
            payload = payload.get("calibrator", payload.get("sqi_calibrator", payload))
            calibrator = SqiCalibrator(bounds={key: tuple(value) for key, value in payload["bounds"].items()},
                                       fitted_on_participant_ids=tuple(payload["fitted_on_participant_ids"]),
                                       method=payload.get("method", "outer_train_empirical_quantiles_v1"))
        pulses = detect_pulses_per_wavelength(views, **self._peak_config(config)) if mode != "off" else None
        pulse = pulses[select_reference_wavelength(pulses)] if pulses is not None else None
        motion = context["motion"]["series"]
        motion_rows = {} if motion is None else {item.routing_window_id: item for item in motion.decisions}
        routing = config["routing"]
        windows = build_routing_windows(record_id, views.x_filter.shape[0], fs_hz=routing["fs_hz"],
                                         window_s=routing["window_s"], hop_s=routing["hop_s"])
        evidence, diagnostic_rows = [], []
        for window in windows:
            quality = None
            if mode != "off":
                local_views = _slice_signal_views_for_routing(views, window.start_sample_400, window.stop_sample_400)
                local_pulse = _slice_global_pulse_for_routing(pulse, window.start_sample_400, window.stop_sample_400)
                quality = evaluate_quality(local_views, config=sqi, calibrator=calibrator, pulse=local_pulse,
                                           **self._peak_config(config))
                if mode == "diagnostics_only":
                    result = evaluate_quality_diagnostics(local_views, config=SqiDiagnosticConfig.from_resolved(config),
                                                          pulse=local_pulse, **self._peak_config(config))
                    diagnostic_rows.append({"window_id": window.routing_window_id, "diagnostics": result})
            motion_row = motion_rows.get(window.routing_window_id)
            evidence.append(RoutingEvidence(window=window, sqi_mode=mode, sqi_assessed=mode != "off",
                direct_q_rate_score=None if quality is None else quality.q_rate.score,
                direct_q_rate_state=None if quality is None else quality.q_rate.state.value,
                direct_q_morph_score=None if quality is None else quality.q_morph.score,
                direct_q_morph_state=None if quality is None else quality.q_morph.state.value,
                motion_detector_enabled=motion is not None,
                motion_probability=None if motion_row is None else motion_row.probability,
                motion_threshold=None if motion_row is None else motion_row.threshold,
                motion_state="off" if motion is None else "unavailable" if motion_row is None else motion_row.motion_state))
        traces = {name: ([item.window.centre_s for item in evidence], [getattr(item, attr) for item in evidence])
                  for name, attr in (("Q_rate", "direct_q_rate_score"), ("Q_morph", "direct_q_morph_score"))
                  if any(getattr(item, attr) is not None for item in evidence)}
        return {"evidence": evidence, "sqi": sqi, "calibrator": calibrator, "pulses": pulses, "point_traces": traces,
                "tables": {"windows": [asdict(item) for item in evidence], "diagnostics": diagnostic_rows},
                "metadata": {"mode": mode, "window_count": len(windows), "fitted_on_current_input": False}}

    def _denoiser(self, config: dict[str, Any], selected: dict[str, Any], context: dict[str, Any],
                  record_id: str, session_id: str) -> dict[str, Any]:
        from ..artifacts import run_artifact_route
        from ..contracts import SignalRoute
        from ..experiment import (_RuntimeRecord, _retain_without_quality_routing,
                                  _slice_global_pulse_for_routing, _slice_signal_views_for_routing)
        from ..config import PipelineConfig
        from ..peaks import select_reference_wavelength
        from ..peaks.resolver import detect_pulses_per_wavelength
        from ..quality.routing_timeline import build_routing_timeline, resolve_routing_evidence
        from ..signal.sqi import evaluate_quality
        views, row, quality = context["imu"]["views"], context["input"]["row"], context["quality"]
        artifact, mode = config["artifact"], config["representation_mode"]
        if config["quality"]["mode"] == "off" and not artifact["motion_detector_enabled"] and not artifact["denoiser_enabled"]:
            # This is the production fast path, not an all-pass routing timeline:
            # introducing 8-second support here would change short/tail windows.
            state = _RuntimeRecord(row, views=views)
            _retain_without_quality_routing([state], PipelineConfig(config, "dashboard", _digest(config)), diagnostics_only=False)
            return {"state": state, "signals": {"direct_RED": views.x_filter[:, 0], "direct_IR": views.x_filter[:, 1]},
                    "tables": {"routing": [state.route_artifact]},
                    "metadata": {"enabled": False, "reducer": "identity", "status": "off", "invoked": False,
                                 "retained": True, "native_and_direct_views_preserved": True}}
        allow_recovery = mode == "feature_vector" and artifact["degraded_policy"] == "denoise_then_extract_rate_features"
        def resolve(item: Any) -> Any:
            return resolve_routing_evidence(item, role=str(row.role),
                allow_rate_feature_recovery_without_direct_sqi=allow_recovery)
        evidence = [resolve(replace(item, denoiser_enabled=artifact["denoiser_enabled"])) for item in quality["evidence"]]
        state = _RuntimeRecord(row, views=views, direct_pulses_per_wavelength=quality["pulses"],
                               artifact_name="identity", artifact_version="identity_v1")
        requested = any(item.denoiser_requested for item in evidence)
        signals = {"direct_RED": views.x_filter[:, 0], "direct_IR": views.x_filter[:, 1]}
        status = "not_requested"
        if requested:
            outcome = run_artifact_route(views, artifact["reducer"], parameters=artifact["parameters"])
            state.artifact_name, state.artifact_version = outcome.result.reducer_id, outcome.result.reducer_version
            status = outcome.result.status
            if outcome.views is not None and outcome.route is SignalRoute.ARTIFACT_RATE_ONLY:
                state.processed_views = outcome.views
                state.processed_pulses_per_wavelength = detect_pulses_per_wavelength(outcome.views, **self._peak_config(config))
                pulse = state.processed_pulses_per_wavelength[select_reference_wavelength(state.processed_pulses_per_wavelength)]
                signals.update(reduced_RED=outcome.views.x_ar[:, 0], reduced_IR=outcome.views.x_ar[:, 1])
                updated = []
                for item in evidence:
                    if item.denoiser_requested:
                        window = item.window
                        local = _slice_signal_views_for_routing(outcome.views, window.start_sample_400, window.stop_sample_400)
                        local_pulse = _slice_global_pulse_for_routing(pulse, window.start_sample_400, window.stop_sample_400)
                        post = evaluate_quality(local, config=quality["sqi"], calibrator=quality["calibrator"],
                                                pulse=local_pulse, **self._peak_config(config))
                        item = resolve(replace(item, denoiser_status="success", post_q_rate_score=post.q_rate.score,
                                               post_q_rate_state=post.q_rate.state.value))
                    updated.append(item)
                evidence = updated
            else:
                evidence = [resolve(replace(item, denoiser_status=status)) if item.denoiser_requested else item for item in evidence]
        state.routing_timeline = build_routing_timeline(record_id=record_id, participant_id=str(row.participant_id),
            role=str(row.role), n_samples=views.x_filter.shape[0], evidence=evidence, config_sha256=_digest(config))
        excellent = [cell for cell in state.routing_timeline.cells if cell.final_tier == "excellent" and cell.source_route == "direct"]
        acceptable = [cell for cell in state.routing_timeline.cells if cell.final_tier == "acceptable"]
        state.retained = bool(excellent if mode in {"raw", "fusion"} else excellent + acceptable)
        state.shape_features_eligible = bool(excellent)
        state.quality_tier = "mixed" if excellent and acceptable else "excellent" if excellent else "acceptable" if acceptable else "excluded"
        state.route = SignalRoute.DIRECT if excellent or any(cell.source_route == "direct" for cell in acceptable) else SignalRoute.ARTIFACT_RATE_ONLY if acceptable else SignalRoute.DROPPED
        state.route_status = "retained_window_routing_timeline" if state.retained else "dropped_no_representation_eligible_routing_cell"
        state.reason = None if state.retained else state.route_status
        return {"state": state, "signals": signals, "tables": {"routing": [asdict(item) for item in state.routing_timeline.cells]},
                "metadata": {"enabled": artifact["denoiser_enabled"], "reducer": artifact["reducer"], "status": status,
                             "invoked": requested, "retained": state.retained, "quality_tier": state.quality_tier,
                             "native_and_direct_views_preserved": True}}

    @staticmethod
    def _report(config: Mapping[str, Any]) -> Any:
        from ..module_registry import resolve_peak_detector_config, resolve_window_config
        return SimpleNamespace(window_profiles=resolve_window_config(config["windows"]),
                               peak_detector=resolve_peak_detector_config(config["signal"]))

    def _features(self, config: dict[str, Any], selected: dict[str, Any], context: dict[str, Any],
                  record_id: str, session_id: str) -> dict[str, Any]:
        from ..experiment import _extract_vector, _extract_matrix_features
        from .feature_preview import build_feature_preview, compute_window_rate_preview
        state = _clone_state(context["denoiser"]["state"])
        capture = {}
        report = self._report(config)
        if config["representation_mode"] == "feature_matrix":
            _extract_matrix_features(state, report, capture=capture)
        else:
            _extract_vector(state, report, config["features"], capture=capture)
        if capture.get("matrix", {}).get("engineering") is None:
            try:
                preview_state = context["denoiser"]["state"]
                if config["representation_mode"] == "feature_matrix" and preview_state.retained and preview_state.routing_timeline is None:
                    # Matrix extraction stops before peak detection without a
                    # routing timeline. Detect once for the independent preview;
                    # keep the production matrix state/reason untouched.
                    from ..experiment import _direct_pulses_for_state, _runtime_imports
                    preview_state = _clone_state(preview_state)
                    _direct_pulses_for_state(preview_state, _runtime_imports(), report.peak_detector)
                capture["window_rate"] = compute_window_rate_preview(preview_state, report, capture)
            except (ValueError, RuntimeError) as exc:
                # An unavailable exploration plot must not change model eligibility.
                capture["window_rate"] = {"status": "unavailable", "reason": str(exc)}
        preview = build_feature_preview(state, capture)
        preview["metadata"].update(
            representation_mode=config["representation_mode"],
            feature_use="exploration_only_not_raw_model_input" if config["representation_mode"] == "raw" else "pipeline_features",
            retained=state.retained, reason=state.reason, fitted_transforms_applied=False,
            missing_values="remain NaN until a saved training transform is applied")
        return {"state": state, **preview}

    def _representation(self, config: dict[str, Any], selected: dict[str, Any], context: dict[str, Any],
                        record_id: str, session_id: str) -> dict[str, Any]:
        from ..config import PipelineConfig
        from ..experiment import _apply_window_quality_selection, _extract_raw
        from ..signal.resample import prepare_configured_dl_input
        # Feature exploration must not exclude a usable raw recording when a feature is unavailable.
        source = context["denoiser"] if config["representation_mode"] == "raw" else context["features"]
        state = _clone_state(source["state"])
        mode, tables, traces = config["representation_mode"], {}, {}
        metadata = {"mode": mode, "retained": state.retained, "canonical_fs_hz": 400.0}
        if mode in {"raw", "fusion"}:
            _extract_raw(state, self._report(config), config["signal"])
            if state.raw_windows is not None:
                selection = _apply_window_quality_selection([state], PipelineConfig(config, "dashboard", _digest(config)),
                                                            train_ids=(), oof_ids=(str(state.row.participant_id),))
                raw = state.raw_windows
                dl, mask, provenance = prepare_configured_dl_input(raw.values, raw.valid_mask,
                    target_fs_hz=float(config["signal"]["dl_resampling"]["target_fs_hz"]))
                metadata.update(shape=list(dl.shape), canonical_window_shape=list(raw.values.shape),
                                normalization=config["signal"]["normalization"], resampling=provenance, selection=selection)
                tables["windows"] = [{"window": index, "start_s": int(start) / 400.0,
                    "valid_samples": int(np.sum(raw.valid_mask[index]))} for index, start in enumerate(raw.start_samples)]
                fs = float(config["signal"]["dl_resampling"]["target_fs_hz"])
                traces = {f"first_window_{channel}": (np.arange(dl.shape[2]) / fs, dl[0, index])
                          for index, channel in enumerate(("RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ"))}
        if mode in {"feature_vector", "fusion"} and state.vector is not None:
            metadata["feature_vector_shape"] = list(state.vector.values.shape)
            # Keep the exact selected model vector visible here; the feature
            # stage additionally exposes intermediate and disabled-group values.
            from ..features.registry import registry_for_feature_names
            registry = registry_for_feature_names(state.vector.feature_names)
            for definition, value, valid in zip(registry.definitions, state.vector.values, state.vector.validity):
                tables.setdefault(f"file_{definition.group}", []).append({
                    "feature": definition.canonical_name, "value": float(value), "valid": bool(valid),
                    "unit": definition.units})
        if mode == "feature_matrix" and state.engineering is not None:
            sequence = state.engineering.sequence
            metadata.update(shape=list(sequence.values.T.shape), transform="saved fold transform applied only at model Analyse")
            tables["matrix_windows"] = [{"window": index, "start_s": int(start) / 400.0,
                "valid_features": int(np.sum(state.engineering.value_validity[index]))}
                for index, start in enumerate(sequence.start_samples)]
        metadata.update(retained=state.retained, reason=state.reason)
        return {"state": state, "point_traces": traces, "tables": tables, "metadata": metadata}

    def _model(self, config: dict[str, Any], selected: dict[str, Any], context: dict[str, Any],
               record_id: str, session_id: str) -> dict[str, Any]:
        from ..training.bundle import load_bundle
        from ..v5.inference_service import predict_workflow_record, resolve_workflow_bundle
        export = selected.get("model_export") or selected.get("model_bundle")
        if not export:
            raise ValueError("Select pretrained model weights; Analyse never invokes training.")
        bundle_path = (self._path(selected["model_bundle"]) if selected.get("model_bundle") else
                       resolve_workflow_bundle(export, case_id=selected.get("model_case"), pipeline_root=self.pipeline_root))
        bundle = load_bundle(bundle_path)
        record_ids = selected.get("record_ids") or [item.get("file_id") or Path(item["path"]).stem for item in selected.get("files", [])] or [record_id]
        states, probabilities, identities = [], [], []
        for current_id in record_ids:
            if str(current_id) == record_id:
                state = context["representation"]["state"]
            else:
                entries, _, _ = self._ensure("representation", config, str(current_id), session_id, selected)
                state = entries["representation"]["value"]["state"]
            if str(state.row.role) not in config["roles"] or str(state.row.role)[:1] not in config["training"]["classifier_role_families"]:
                continue
            states.append(state)
            if not state.retained:
                continue
            result = predict_workflow_record(bundle, config_payload=config, participant_id=str(state.row.participant_id),
                record_id=str(state.row.record_id), role=str(state.row.role), raw_windows=state.raw_windows,
                feature_vector=state.vector, matrix_features=state.engineering, route=state.route.value,
                label=None if int(state.row.class_id) < 0 else int(state.row.class_id))
            probabilities.append(np.asarray(result["probabilities"]))
            identities.extend(result["identities"])
        values = np.concatenate(probabilities) if probabilities else np.empty((0, int(config["model"].get("n_classes", 3))))
        rows = [{"record_id": item.file_id, "participant_id": item.participant_id, "window_id": item.window_id,
                 **{f"p_{index}": float(value) for index, value in enumerate(probability)},
                 **_prediction_fields(config, probability)}
                for item, probability in zip(identities, values)]
        rejected = [{"record_id": state.row.record_id, "reason": state.reason} for state in states if not state.retained]
        return {"states": states, "probabilities": values, "identities": identities,
                "tables": {"predictions": rows, "excluded_recordings": rejected},
                "metadata": {"bundle": str(bundle_path), "prediction_count": len(rows), "recording_count": len(states),
                             "model_training_executed": False, "configuration_source": "current_controls",
                             "class_names": list(config["manifest"]["class_name_order"]),
                             "class_ids": list(config["manifest"]["class_id_order"])}}

    def _aggregation(self, config: dict[str, Any], selected: dict[str, Any], context: dict[str, Any],
                     record_id: str, session_id: str) -> dict[str, Any]:
        from ..experiment import _make_oof
        source, aggregation = context["model"], config["aggregation"]
        identity = _digest(config)
        common = {"config_hash": identity, "manifest_hash": _digest([
            (state.row.record_id, getattr(state.row, "source_hash", "interactive_input")) for state in source["states"]]),
            "fold_hash": "interactive_inference_not_a_fold", "preprocessing_hash": _digest(config["signal"]),
            "feature_hash": _digest((config["features"], config["representation_mode"])),
            "model_hash": hashlib.sha256((Path(source["metadata"]["bundle"]) / "manifest.json").read_bytes()).hexdigest()}
        common.update(repeat=0, fold=0, split_seed=0, training_seed=int(config["training"].get("seed", 42)),
                      representation_mode=config["representation_mode"], class_order=())
        rows = _make_oof(source["states"], tuple(sorted({str(state.row.participant_id) for state in source["states"]})),
                         source["identities"], source["probabilities"], common,
                         balance_line=aggregation["balance_line"], quality_weighting=aggregation.get("quality_weighting", False),
                         quality_weight_source=aggregation.get("quality_weight_source", "none"))
        return {"tables": {name: [{**asdict(item), **(_prediction_fields(config, item.probabilities) if item.retained else {})}
                                 for item in values]
                           for name, values in zip(("windows", "recordings", "roles", "participants"), rows)},
                "metadata": {"balance_line": aggregation["balance_line"], "scope": "interactive_predictions_not_outer_oof",
                             "class_names": list(config["manifest"]["class_name_order"]),
                             "class_ids": list(config["manifest"]["class_id_order"])}}

    def _preview(self, stage: str, value: Mapping[str, Any], start_s: float, duration_s: float) -> dict[str, Any]:
        from ..contracts import to_strict_json_value
        if not np.isfinite([start_s, duration_s]).all() or start_s < 0 or duration_s <= 0:
            raise ValueError("Preview start must be nonnegative and duration positive.")
        traces, metrics = {}, []
        for name, signal in value.get("signals", {}).items():
            signal = np.asarray(signal)
            first, last = int(round(start_s * 400)), min(len(signal), int(round((start_s + duration_s) * 400)))
            stride = max(1, int(np.ceil((last - first) / 2000)))
            index = np.arange(first, max(first, last), stride)
            traces[name] = {"x": (index / 400.0).tolist(), "y": signal[index].tolist()}
            window = signal[first:last]
            if window.size:
                metrics.append({"signal": name, "samples": len(window), "mean": float(np.mean(window)),
                                "std": float(np.std(window)), "minimum": float(np.min(window)), "maximum": float(np.max(window))})
        for name, (x, y) in value.get("point_traces", {}).items():
            x, y = np.asarray(x), np.asarray(y)
            # Window trends summarize the whole recording, not the short peak
            # inspection range: a window centre may lie outside that range.
            keep = (np.ones(x.shape, dtype=bool) if value.get("trace_figures", {}).get(name) == "Window PPI / HR"
                    else (x >= start_s) & (x < start_s + duration_s))
            markers = "markers" in value.get("trace_styles", {}).get(name, {}).get("mode", "") or name.endswith("_peaks")
            stride = 1 if markers else max(1, int(np.ceil(np.count_nonzero(keep) / 2000)))
            traces[name] = {"x": x[keep][::stride].tolist(), "y": y[keep][::stride].tolist()}
        tables = {name: list(rows) for name, rows in value.get("tables", {}).items()}
        if metrics:
            tables["metrics"] = metrics
        metadata = {**value.get("metadata", {}), "stage": stage, "algorithm_scope": "full_record",
                    "display_start_s": start_s, "display_duration_s": duration_s, "model_training_executed": False}
        if "window_rate" in metadata:
            metadata["window_rate"] = {**metadata["window_rate"], "display_scope": "full_record"}
        return to_strict_json_value({"traces": traces, "tables": tables, "metadata": metadata,
            **{key: value[key] for key in ("trace_groups", "trace_styles", "trace_figures", "frequency_traces", "frequency_groups",
                                         "fft_traces", "fft_groups") if key in value}})
