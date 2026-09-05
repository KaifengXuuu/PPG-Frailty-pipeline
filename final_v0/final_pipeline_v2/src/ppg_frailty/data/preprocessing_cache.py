"""Domain adapters for the immutable recording preprocessing cache.

Only supervision-free, recording-local prefixes are represented here.  The
operational record identity and calibration-source role remain auditable, but
frailty/motion targets and all label-derived state have no serialization path.
Fold-fitted scalers, SQI calibrators, thresholds, routing masks, samplers and
class weights are likewise excluded.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import scipy

from ..contracts import SignalRoute, to_strict_json_value
from ..provenance import stable_payload_sha256
from ..representations.motion import MotionWindowTensors
from ..representations.raw import RawWindows
from ..signal.motion_imu import (
    MOTION_IMU_CALIBRATION_SCHEMA,
    MotionImuCalibration,
    RollPitchEkfConfig,
)
from ..signal.views import CanonicalSignalViews
from .cache import CacheMissError
from .recording_cache import (
    NamedSourceDependency,
    OrderedModuleSpec,
    RecordingCacheAccessError,
    RecordingCacheBuild,
    RecordingCacheCorruptionError,
    RecordingCacheError,
    RecordingCacheIdentity,
    RecordingCacheSourceError,
    RecordingPreprocessingCache,
)


SUPPORTED_NAMESPACES = frozenset(
    {
        "imu_calibration",
        "canonical_signal_views",
        "motion_windows",
        "raw_windows",
    }
)
_TUPLE_EKF_FIELDS = frozenset(
    {
        "process_covariance_diagonal_per_second",
        "observation_covariance_diagonal_rad2",
        "initial_covariance_diagonal",
    }
)


@lru_cache(maxsize=8192)
def _verified_source_digest(
    path_text: str,
    device: int,
    inode: int,
    size: int,
    mtime_ns: int,
    ctime_ns: int,
) -> str:
    """Hash a stable file identity once per process."""

    del device, inode, size, mtime_ns, ctime_ns
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=512)
def _implementation_file_sha256(
    path_text: str,
    device: int,
    inode: int,
    size: int,
    mtime_ns: int,
    ctime_ns: int,
) -> str:
    del device, inode, size, mtime_ns, ctime_ns
    return hashlib.sha256(Path(path_text).read_bytes()).hexdigest()


def _implementation_dependency_sha256(
    dependencies: Mapping[str, str | Path],
) -> str:
    """Hash one explicit, named implementation dependency manifest.

    Dependency names and membership are part of the digest, while absolute
    checkout paths are deliberately excluded so identical code is portable.
    """

    if not dependencies:
        raise ValueError("implementation dependency manifest cannot be empty")
    try:
        manifest = []
        for name, path in sorted(dependencies.items()):
            resolved = Path(path).resolve()
            stat = resolved.stat()
            manifest.append(
                {
                    "dependency": str(name),
                    "filename": resolved.name,
                    "sha256": _implementation_file_sha256(
                        str(resolved),
                        int(stat.st_dev),
                        int(stat.st_ino),
                        int(stat.st_size),
                        int(stat.st_mtime_ns),
                        int(stat.st_ctime_ns),
                    ),
                }
            )
        return stable_payload_sha256(manifest)
    except RecordingCacheError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise RecordingCacheAccessError(
            "cannot hash preprocessing implementation dependency manifest"
        ) from exc


def _module(
    module_id: str,
    version: str,
    implementation_sha256: str,
    parameters: Mapping[str, Any],
) -> OrderedModuleSpec:
    return OrderedModuleSpec(
        module_id=module_id,
        module_version=version,
        implementation_sha256=implementation_sha256,
        enabled=True,
        parameters=to_strict_json_value(dict(parameters)),
    )


def _strict_clone(value: Any) -> Any:
    return json.loads(
        json.dumps(
            to_strict_json_value(value),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )


@dataclass
class PreprocessingCacheSession:
    """One-cell cache session with separate operational provenance."""

    mode: str
    root: Path
    namespaces: frozenset[str]
    repository_root: Path
    pipeline_root: Path
    store: RecordingPreprocessingCache | None
    events: list[dict[str, Any]] = field(default_factory=list)
    identities: dict[str, dict[str, Any]] = field(default_factory=dict)
    calibration_refs: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        paths: Any,
    ) -> "PreprocessingCacheSession":
        payload = dict(value or {})
        mode = str(payload.get("mode", "off"))
        if mode not in {"off", "read_only", "read_write"}:
            raise ValueError("unknown preprocessing cache mode")
        if payload.get("verify_source_sha256", True) is not True:
            raise ValueError("preprocessing cache cannot disable source verification")
        namespaces = frozenset(
            str(item)
            for item in payload.get("namespaces", sorted(SUPPORTED_NAMESPACES))
        )
        if not namespaces or not namespaces <= SUPPORTED_NAMESPACES:
            raise ValueError("unsupported preprocessing cache namespace")

        if mode == "off" and not hasattr(paths, "pipeline_root"):
            # Disabled means no filesystem access.  Lightweight legacy-bridge
            # fixtures intentionally provide no repository paths, and an off
            # cache must not turn that unrelated absence into a runtime error.
            inert_root = Path.cwd().resolve()
            return cls(
                mode=mode,
                root=inert_root / "artifacts/studies/cache",
                namespaces=namespaces,
                repository_root=Path(
                    getattr(paths, "repository_root", inert_root)
                ).resolve(),
                pipeline_root=inert_root,
                store=None,
            )

        pipeline_root = Path(paths.pipeline_root).resolve()
        canonical_lexical_root = pipeline_root / "artifacts/studies/cache"
        requested = Path(payload.get("root", "artifacts/studies/cache"))
        lexical_root = Path(
            os.path.abspath(
                requested if requested.is_absolute() else pipeline_root / requested
            )
        )
        lexical_root.relative_to(canonical_lexical_root)
        current = lexical_root
        while True:
            if current.exists() and current.is_symlink():
                raise ValueError("preprocessing cache root may not traverse symlinks")
            if current == pipeline_root:
                break
            current = current.parent
        resolved = lexical_root.resolve(strict=False)
        store = None if mode == "off" else RecordingPreprocessingCache(resolved)
        return cls(
            mode=mode,
            root=resolved,
            namespaces=namespaces,
            repository_root=Path(paths.repository_root).resolve(),
            pipeline_root=pipeline_root,
            store=store,
        )

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def _source(self, row: Any, *, name: str) -> NamedSourceDependency:
        try:
            source = (self.repository_root / str(row.source_path)).resolve()
            source.relative_to(self.repository_root)
            if not source.is_file() or source.is_symlink():
                raise FileNotFoundError(source)
            stat = source.stat()
            observed = _verified_source_digest(
                str(source),
                int(stat.st_dev),
                int(stat.st_ino),
                int(stat.st_size),
                int(stat.st_mtime_ns),
                int(stat.st_ctime_ns),
            )
            expected = str(row.source_hash)
        except RecordingCacheError:
            raise
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise RecordingCacheSourceError(
                f"cannot verify recording source before cache access: "
                f"{getattr(row, 'record_id', '<unknown>')}"
            ) from exc
        if observed != expected:
            raise RecordingCacheSourceError(
                f"source hash drift before cache access: {row.record_id}"
            )
        return NamedSourceDependency(
            name=name,
            sha256=expected,
            properties={
                "record_id": str(row.record_id),
                "source_version": str(row.source_version),
                "fs_hz": float(row.fs),
                "n_samples": int(row.n_samples),
                "duration_s": float(row.duration_s),
                "channel_schema": list(row.channel_schema),
                "channel_units": dict(row.channel_units),
                "synchrony_status": str(row.synchrony_status),
                "reference_available": bool(row.reference_available),
                "qc_status": str(row.qc_status),
                "qc_reasons": list(row.qc_reasons),
                "manifest_version": str(row.manifest_version),
            },
        )

    def _operate(
        self,
        identity: RecordingCacheIdentity,
        builder: Callable[[], RecordingCacheBuild],
    ) -> tuple[Mapping[str, np.ndarray], Mapping[str, Any], str]:
        started = time.perf_counter()
        namespace_active = identity.namespace in self.namespaces
        if not self.enabled or not namespace_active:
            built = builder()
            disposition = "cache_off" if not self.enabled else "namespace_bypassed"
            arrays, attributes = built.arrays, built.attributes
            path = None
        elif self.mode == "read_only":
            assert self.store is not None
            try:
                entry = self.store.load(identity)
                arrays, attributes = entry.arrays, entry.attributes
                disposition, path = "hit", entry.path
            except CacheMissError:
                built = builder()
                arrays, attributes = built.arrays, built.attributes
                disposition, path = "read_only_miss_computed", None
        else:
            assert self.store is not None
            result = self.store.get_or_compute(identity, builder)
            arrays, attributes = result.entry.arrays, result.entry.attributes
            disposition, path = result.disposition, result.entry.path
        elapsed = time.perf_counter() - started
        key = identity.key
        self.identities.setdefault(key, identity.to_payload())
        self.events.append(
            {
                "namespace": identity.namespace,
                "layer": identity.layer,
                "recording_id": identity.recording_id,
                "cache_key": key,
                "disposition": disposition,
                "elapsed_seconds": elapsed,
                "logical_array_bytes": int(
                    sum(np.asarray(item).nbytes for item in arrays.values())
                ),
                "entry_path": (
                    None
                    if path is None
                    else Path(path).relative_to(self.root).as_posix()
                ),
                "module_chain": [
                    module.module_id for module in identity.module_chain
                ],
                "affects_predictions": False,
            }
        )
        return arrays, attributes, key

    def calibration(
        self,
        row: Any,
        imu_config: Mapping[str, Any],
        builder: Callable[[], MotionImuCalibration],
    ) -> MotionImuCalibration:
        from .. import contracts as contracts_module
        from .. import experiment as experiment_module
        from .. import pipeline as pipeline_module
        from .. import provenance as provenance_module
        from ..signal import imu as imu_module
        from ..signal import motion_imu as motion_module
        from ..signal import preprocess as preprocess_module
        from ..signal import views as views_module
        from . import qc as qc_module
        from . import schema as schema_module

        source = self._source(row, name="same_participant_B_source")
        implementation = _implementation_dependency_sha256(
            {
                "cache_domain_adapter": __file__,
                "contracts": contracts_module.__file__,
                "data_schema": schema_module.__file__,
                "experiment_callsite": experiment_module.__file__,
                "recording_loader": pipeline_module.__file__,
                "physical_qc": qc_module.__file__,
                "provenance_hashing": provenance_module.__file__,
                "signal_imu_units": imu_module.__file__,
                "signal_motion_imu": motion_module.__file__,
                "signal_preprocess_config": preprocess_module.__file__,
                "signal_views": views_module.__file__,
            }
        )
        identity = RecordingCacheIdentity(
            namespace="imu_calibration",
            layer="same_participant_B_sensor_calibration",
            recording_id=str(row.record_id),
            source_dependencies=(source,),
            module_chain=(
                _module(
                    "fit_motion_imu_calibration",
                    MOTION_IMU_CALIBRATION_SCHEMA,
                    implementation,
                    {
                        "participant_id": str(row.participant_id),
                        "source_role": str(row.role),
                        "fs_hz": float(row.fs),
                        "acceleration_unit": "g",
                        "gyroscope_unit": "deg/s",
                        "config": dict(imu_config),
                    },
                ),
            ),
            producer_sha256=implementation,
            output_schema={
                "type": "MotionImuCalibration",
                "schema_version": MOTION_IMU_CALIBRATION_SCHEMA,
                "array_names": ["acceleration_bias_mps2", "gyroscope_bias_rads"],
            },
            extra={"numpy_version": np.__version__, "scipy_version": scipy.__version__},
        )

        def pack() -> RecordingCacheBuild:
            value = builder()
            value.validate()
            return RecordingCacheBuild(
                arrays={
                    "acceleration_bias_mps2": value.acceleration_bias_mps2,
                    "gyroscope_bias_rads": value.gyroscope_bias_rads,
                },
                attributes={
                    "participant_id": value.participant_id,
                    "file_id": value.file_id,
                    "source_role": value.source_role,
                    "initial_roll_rad": value.initial_roll_rad,
                    "initial_pitch_rad": value.initial_pitch_rad,
                    "calibration_start_sample": value.calibration_start_sample,
                    "calibration_stop_sample": value.calibration_stop_sample,
                    "calibration_quality": value.calibration_quality,
                    "config": asdict(value.config),
                    "artifact_sha256": value.artifact_sha256,
                    "schema_version": value.schema_version,
                },
            )

        arrays, attributes, key = self._operate(identity, pack)
        try:
            config_payload = dict(attributes["config"])
            for name in _TUPLE_EKF_FIELDS:
                config_payload[name] = tuple(config_payload[name])
            value = MotionImuCalibration(
                participant_id=str(attributes["participant_id"]),
                file_id=str(attributes["file_id"]),
                source_role=str(attributes["source_role"]),
                acceleration_bias_mps2=np.asarray(
                    arrays["acceleration_bias_mps2"]
                ),
                gyroscope_bias_rads=np.asarray(arrays["gyroscope_bias_rads"]),
                initial_roll_rad=float(attributes["initial_roll_rad"]),
                initial_pitch_rad=float(attributes["initial_pitch_rad"]),
                calibration_start_sample=int(
                    attributes["calibration_start_sample"]
                ),
                calibration_stop_sample=int(attributes["calibration_stop_sample"]),
                calibration_quality=dict(attributes["calibration_quality"]),
                config=RollPitchEkfConfig(**config_payload),
                artifact_sha256=str(attributes["artifact_sha256"]),
                schema_version=str(attributes["schema_version"]),
            )
            value.validate()
        except RecordingCacheError:
            raise
        except Exception as exc:
            raise RecordingCacheCorruptionError(
                f"cannot unpack cached IMU calibration: {key}"
            ) from exc
        self.calibration_refs[value.artifact_sha256] = {
            "cache_key": key,
            "source_sha256": source.sha256,
            "source_record_id": str(row.record_id),
        }
        return value

    def canonical_views(
        self,
        row: Any,
        *,
        maximum_samples: int | None,
        signal_config: Mapping[str, Any],
        quality_preprocess_config: Mapping[str, Any],
        calibration: MotionImuCalibration | None,
        builder: Callable[[], tuple[CanonicalSignalViews, Mapping[str, Any], Mapping[str, Any]]],
    ) -> tuple[CanonicalSignalViews, dict[str, Any], dict[str, Any], str]:
        from .. import contracts as contracts_module
        from .. import experiment as experiment_module
        from .. import module_registry as module_registry_module
        from .. import normalization as normalization_module
        from .. import pipeline as pipeline_module
        from ..signal import imu as imu_module
        from ..signal import motion_imu as motion_module
        from ..signal import preprocess as preprocess_module
        from ..signal import resample as resample_module
        from ..signal import views as views_module
        from . import qc as qc_module
        from . import schema as schema_module

        dependencies = [self._source(row, name="target_recording_source")]
        calibration_key = None
        if calibration is not None:
            reference = self.calibration_refs.get(calibration.artifact_sha256)
            calibration_key = None if reference is None else reference["cache_key"]
            dependencies.append(
                NamedSourceDependency(
                    name="imu_calibration_artifact",
                    sha256=(
                        calibration.artifact_sha256
                        if calibration_key is None
                        else calibration_key
                    ),
                    properties={
                        "artifact_sha256": calibration.artifact_sha256,
                        "participant_id": calibration.participant_id,
                        "file_id": calibration.file_id,
                        "upstream_cache_key": calibration_key,
                    },
                )
            )
        loader_dependencies = {
            "cache_domain_adapter": __file__,
            "data_schema": schema_module.__file__,
            "experiment_callsite": experiment_module.__file__,
            "physical_qc": qc_module.__file__,
            "recording_loader": pipeline_module.__file__,
        }
        preprocess_dependencies = {
            "cache_domain_adapter": __file__,
            "contracts": contracts_module.__file__,
            "module_registry": module_registry_module.__file__,
            "normalization": normalization_module.__file__,
            "signal_imu": imu_module.__file__,
            "signal_motion_imu": motion_module.__file__,
            "signal_preprocess": preprocess_module.__file__,
            "signal_resample": resample_module.__file__,
            "signal_views": views_module.__file__,
        }
        loader_sha = _implementation_dependency_sha256(loader_dependencies)
        preprocess_sha = _implementation_dependency_sha256(
            preprocess_dependencies
        )
        identity = RecordingCacheIdentity(
            namespace="canonical_signal_views",
            layer="physical_filter_imu_and_signal_views",
            recording_id=str(row.record_id),
            source_dependencies=tuple(dependencies),
            module_chain=(
                _module(
                    "physical_recording_qc_and_source_loader",
                    "physical_recording_qc_v2",
                    loader_sha,
                    {"maximum_samples": maximum_samples},
                ),
                _module(
                    "ppg_filter_and_gap_repair",
                    "canonical_400hz_amplitude_preserving_v2",
                    preprocess_sha,
                    {
                        "signal": dict(signal_config),
                        "quality_preprocess": dict(quality_preprocess_config),
                    },
                ),
                _module(
                    str(signal_config["imu"]["gravity_method"]),
                    "processed_imu_physical_v2",
                    preprocess_sha,
                    {
                        "imu": dict(signal_config["imu"]),
                        "calibration_artifact_sha256": (
                            None if calibration is None else calibration.artifact_sha256
                        ),
                    },
                ),
                _module(
                    "canonical_signal_views",
                    "canonical_signal_views_v1",
                    preprocess_sha,
                    {"route": "direct", "fs_hz": 400.0},
                ),
            ),
            producer_sha256=_implementation_dependency_sha256(
                {**loader_dependencies, **preprocess_dependencies}
            ),
            output_schema={
                "type": "CanonicalSignalViews",
                "route": "direct",
                "fs_hz": 400.0,
                "ppg_channels": ["RED", "IR"],
                "imu_view": "processed_imu_physical",
            },
            extra={
                "maximum_samples": maximum_samples,
                "numpy_version": np.__version__,
                "scipy_version": scipy.__version__,
            },
        )

        def pack() -> RecordingCacheBuild:
            views, qc, profile = builder()
            views.validate()
            if views.route is not SignalRoute.DIRECT or views.x_ar is not None:
                raise ValueError("only pristine direct canonical views are cacheable")
            arrays: dict[str, np.ndarray] = {
                "x_native": views.x_native,
                "x_filter": views.x_filter,
                "x_analysis_rate": views.x_analysis_rate,
                "source_valid_mask": views.source_valid_mask,
                "repair_mask": views.repair_mask,
            }
            for name, values in sorted(views.imu_processed.items()):
                arrays[f"imu__{name}"] = np.asarray(values)
            return RecordingCacheBuild(
                arrays=arrays,
                attributes={
                    "metadata": _strict_clone(views.metadata),
                    "imu_keys": sorted(views.imu_processed),
                    "route": views.route.value,
                    "physical_qc_evidence": _strict_clone(qc),
                    "physical_qc_profile": _strict_clone(profile),
                },
            )

        arrays, attributes, key = self._operate(identity, pack)
        try:
            imu = {
                str(name): np.asarray(arrays[f"imu__{name}"])
                for name in attributes["imu_keys"]
            }
            views = CanonicalSignalViews(
                x_native=np.asarray(arrays["x_native"]),
                x_filter=np.asarray(arrays["x_filter"]),
                x_analysis_rate=np.asarray(arrays["x_analysis_rate"]),
                imu_processed=imu,
                metadata=dict(attributes["metadata"]),
                source_valid_mask=np.asarray(arrays["source_valid_mask"]),
                repair_mask=np.asarray(arrays["repair_mask"]),
                x_ar=None,
                route=SignalRoute(str(attributes["route"])),
            )
            views.validate()
            qc = dict(attributes["physical_qc_evidence"])
            profile = dict(attributes["physical_qc_profile"])
        except RecordingCacheError:
            raise
        except Exception as exc:
            raise RecordingCacheCorruptionError(
                f"cannot unpack cached canonical signal views: {key}"
            ) from exc
        return (
            views,
            qc,
            profile,
            key,
        )

    def raw_windows(
        self,
        row: Any,
        *,
        upstream_views_key: str,
        plan: Any,
        normalization: Mapping[str, Any] | None,
        builder: Callable[[], RawWindows],
    ) -> tuple[RawWindows, str]:
        from .. import contracts as contracts_module
        from ..data import windows as windows_module
        from .. import normalization as normalization_module
        from ..representations import raw as raw_module
        from ..signal import views as views_module

        raw_sha = _implementation_dependency_sha256(
            {
                "cache_domain_adapter": __file__,
                "contracts": contracts_module.__file__,
                "normalization": normalization_module.__file__,
                "raw_representation": raw_module.__file__,
                "signal_views": views_module.__file__,
                "window_contract": windows_module.__file__,
            }
        )
        identity = RecordingCacheIdentity(
            namespace="raw_windows",
            layer="pristine_pre_routing_raw_dl_windows",
            recording_id=str(row.record_id),
            source_dependencies=(
                self._source(row, name="target_recording_source"),
                NamedSourceDependency(
                    name="canonical_signal_views",
                    sha256=upstream_views_key,
                    properties={"upstream_cache_key": upstream_views_key},
                ),
            ),
            module_chain=(
                _module(
                    "raw_window_plan",
                    "window_plan_v2",
                    raw_sha,
                    asdict(plan),
                ),
                _module(
                    "x_dl_all8_window_norm",
                    "raw_all8_window_normalization_v2",
                    raw_sha,
                    {"normalization": dict(normalization or {})},
                ),
            ),
            producer_sha256=raw_sha,
            output_schema={
                "type": "RawWindows",
                "layout": "N_8_T",
                "routing_mask_applied": False,
                "labels_present": False,
            },
            extra={"numpy_version": np.__version__},
        )

        def pack() -> RecordingCacheBuild:
            value = builder()
            if (
                value.window_quality_scores is not None
                or value.window_aggregation_mask is not None
            ):
                raise ValueError("only pristine pre-routing raw windows are cacheable")
            _validate_raw_windows(value)
            return RecordingCacheBuild(
                arrays={
                    "values": value.values,
                    "valid_mask": value.valid_mask,
                    "start_samples": value.start_samples,
                },
                attributes={
                    "candidate_count": int(value.candidate_count),
                    "dropped_invalid_count": int(value.dropped_invalid_count),
                    "provenance": _strict_clone(value.provenance),
                },
            )

        arrays, attributes, key = self._operate(identity, pack)
        try:
            value = RawWindows(
                values=np.asarray(arrays["values"]),
                valid_mask=np.asarray(arrays["valid_mask"]),
                start_samples=np.asarray(arrays["start_samples"]),
                candidate_count=int(attributes["candidate_count"]),
                dropped_invalid_count=int(attributes["dropped_invalid_count"]),
                provenance=dict(attributes["provenance"]),
            )
            _validate_raw_windows(value)
        except RecordingCacheError:
            raise
        except Exception as exc:
            raise RecordingCacheCorruptionError(
                f"cannot unpack cached raw windows: {key}"
            ) from exc
        return value, key

    def motion_windows(
        self,
        row: Any,
        *,
        upstream_views_key: str,
        recording: Any,
        profile_id: str,
        builder: Callable[[], MotionWindowTensors],
    ) -> tuple[MotionWindowTensors, str]:
        """Cache pristine 8 s/2 s motion tensors before any fold scaler/model.

        The stored tensor contains window-normalized RED/IR plus physical-unit
        IMU axes.  It deliberately excludes the fold-fitted IMU transform,
        motion probabilities, threshold states, SQI evidence and route masks.
        """

        from .. import contracts as contracts_module
        from ..quality import motion_adapters as motion_adapters_module
        from ..quality import motion_bundle_adapter as bundle_adapter_module
        from ..representations import motion as motion_module
        from ..signal import motion_imu as motion_imu_module
        from ..signal import views as views_module

        schema = motion_module.motion_network_schema_payload(profile_id)
        implementation = _implementation_dependency_sha256(
            {
                "cache_domain_adapter": __file__,
                "contracts": contracts_module.__file__,
                "motion_bundle_adapter": bundle_adapter_module.__file__,
                "motion_input_adapter": motion_adapters_module.__file__,
                "motion_representation": motion_module.__file__,
                "signal_motion_imu": motion_imu_module.__file__,
                "signal_views": views_module.__file__,
            }
        )
        identity = RecordingCacheIdentity(
            namespace="motion_windows",
            layer="pristine_pre_scaler_motion_8s_hop2s_windows",
            recording_id=str(row.record_id),
            source_dependencies=(
                self._source(row, name="target_recording_source"),
                NamedSourceDependency(
                    name="canonical_signal_views",
                    sha256=upstream_views_key,
                    properties={"upstream_cache_key": upstream_views_key},
                ),
            ),
            module_chain=(
                _module(
                    "motion_recording_from_signal_views",
                    "stage5_reused_motion_recording_adapter_v1",
                    implementation,
                    {
                        "fs_hz": float(recording.fs_hz),
                        "identity_metadata_cached": False,
                    },
                ),
                _module(
                    "build_motion_window_tensors",
                    str(schema["schema_version"]),
                    implementation,
                    schema,
                ),
            ),
            producer_sha256=implementation,
            output_schema={
                "type": "MotionWindowTensors",
                "profile_id": profile_id,
                "schema_sha256": stable_payload_sha256(schema),
                "layout": "window_channel_sample",
                "fold_imu_scaler_applied": False,
                "motion_probabilities_present": False,
                "route_masks_present": False,
                "labels_present": False,
            },
            extra={"numpy_version": np.__version__},
        )

        def pack() -> RecordingCacheBuild:
            value = builder()
            value.validate()
            if value.profile_id != profile_id:
                raise ValueError("motion-window profile differs from cache identity")
            return RecordingCacheBuild(
                arrays={
                    "values": value.values,
                    "start_samples": value.start_samples,
                },
                attributes={
                    "profile_id": value.profile_id,
                    "channel_schema": list(value.channel_schema),
                    "schema_sha256": value.schema_sha256,
                },
            )

        arrays, attributes, key = self._operate(identity, pack)
        try:
            value = MotionWindowTensors(
                values=np.asarray(arrays["values"]),
                start_samples=np.asarray(arrays["start_samples"]),
                # Identity/group/activity fields are deliberately restored from
                # the current manifest instead of being persisted in this
                # label-free signal cache.
                record_id=str(recording.record_id),
                participant_id=str(recording.participant_id),
                role_or_activity=str(recording.role_or_activity),
                dataset_id=str(recording.dataset_id),
                profile_id=str(attributes["profile_id"]),
                channel_schema=tuple(attributes["channel_schema"]),
                schema_sha256=str(attributes["schema_sha256"]),
            )
            value.validate()
            if value.profile_id != profile_id:
                raise ValueError("cached motion-window identity drift")
        except RecordingCacheError:
            raise
        except Exception as exc:
            raise RecordingCacheCorruptionError(
                f"cannot unpack cached motion windows: {key}"
            ) from exc
        return value, key

    def audit_payload(self) -> dict[str, Any]:
        counts: dict[str, dict[str, int]] = {}
        for event in self.events:
            namespace = str(event["namespace"])
            disposition = str(event["disposition"])
            counts.setdefault(namespace, {})[disposition] = (
                counts.setdefault(namespace, {}).get(disposition, 0) + 1
            )
        return {
            "schema_version": "ppg_frailty.preprocessing_cache_audit.v1",
            "mode": self.mode,
            "root": self.root.relative_to(self.pipeline_root).as_posix(),
            "namespaces": sorted(self.namespaces),
            "source_verification": "sha256_once_per_process_per_stable_stat_identity",
            "affects_predictions": False,
            # ``labels_cached`` is retained for reporter compatibility; the two
            # explicit fields remove ambiguity around the auditable B-role used
            # for participant-local IMU calibration.
            "labels_cached": False,
            "supervision_target_labels_cached": False,
            "supervision_target_fields_cached": False,
            "calibration_source_role_cached": bool(self.calibration_refs),
            "operational_identity_metadata_only": True,
            "fold_local_artifacts_cached": False,
            "fold_imu_scaler_cached": False,
            "sqi_calibrator_cached": False,
            "motion_probabilities_cached": False,
            "thresholds_cached": False,
            "route_masks_cached": False,
            "counts": counts,
            "logical_array_bytes": int(
                sum(int(event["logical_array_bytes"]) for event in self.events)
            ),
            "elapsed_seconds": float(
                sum(float(event["elapsed_seconds"]) for event in self.events)
            ),
            "events": list(self.events),
            "identities": dict(sorted(self.identities.items())),
        }


def _validate_raw_windows(value: RawWindows) -> None:
    values = np.asarray(value.values)
    valid = np.asarray(value.valid_mask)
    starts = np.asarray(value.start_samples)
    if values.ndim != 3 or values.shape[1] != 8 or values.dtype != np.float32:
        raise ValueError("cached raw values must have float32 N-by-8-by-T layout")
    if valid.shape != (values.shape[0], values.shape[2]) or valid.dtype != bool:
        raise ValueError("cached raw validity mask is misaligned")
    if starts.shape != (values.shape[0],) or not np.issubdtype(starts.dtype, np.integer):
        raise ValueError("cached raw start samples are misaligned")
    if not np.isfinite(values).all() or np.any(starts < 0):
        raise ValueError("cached raw windows contain invalid values")
    if starts.size > 1 and np.any(np.diff(starts) <= 0):
        raise ValueError("cached raw start samples are not strictly increasing")
    if value.candidate_count < values.shape[0] or value.dropped_invalid_count < 0:
        raise ValueError("cached raw window counts are invalid")


__all__ = ["PreprocessingCacheSession", "SUPPORTED_NAMESPACES"]
