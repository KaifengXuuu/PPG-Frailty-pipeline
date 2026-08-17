"""Source-bound ONNX winner export and ONNX Runtime readback gate.

The gate operates only at the frozen model-input-to-probability boundary.  It
never trains, selects, or labels an ablation as a winner.  A successful
certificate is published atomically with the ONNX model and both probability
matrices; unsupported converters produce a distinct non-certificate report.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import to_strict_json_value
from .provenance import sha256_file, stable_payload_sha256
from .training.bundle import (
    FINAL_BUNDLE_PARITY_ATOL,
    ONNX_WINNER_ABSOLUTE_TOLERANCE,
    ONNX_WINNER_OPSET_VERSION,
    ONNX_WINNER_RELATIVE_TOLERANCE,
)


PRODUCER_ENTRY_ID = "onnx_winner_export_and_ort_readback_source_bound_v2"
PRODUCER_SOURCE_RELATIVE_PATH = "src/ppg_frailty/onnx_winner.py"
CERTIFICATE_SCHEMA = "ppg_frailty.onnx_winner_gate_certificate.v2"
ARTIFACT_INDEX_SCHEMA = "ppg_frailty.onnx_winner_artifact_index.v2"
BOUNDARY = "model_input_to_probabilities"
_HEX = frozenset("0123456789abcdef")
_ESTIMATOR_CONVERTER_MODEL_IDS = frozenset(
    {"logistic_regression", "rbf_svm", "extra_trees"}
)
_CERTIFICATE_FIELDS = frozenset(
    {
        "schema_version", "pipeline_generation", "status",
        "producer_entry_id", "producer_source_relative_path",
        "producer_source_sha256", "selection_record_file_sha256",
        "manual_selection_sha256", "config_id", "config_hash",
        "model_machine_id", "final_refit_execution_hash",
        "bundle_manifest_sha256", "bundle_manifest_bytes",
        "final_refit_attestation_file_sha256",
        "final_refit_attestation_payload_sha256",
        "bundle_run_hash", "bundle_golden_case_sha256", "input_spec_hash",
        "source_snapshot_sha256", "code_commit", "git_status_sha256",
        "dependency_gate_sha256", "dependency_lock_file_sha256",
        "dependency_profile_ids", "runtime_versions", "class_order",
        "boundary", "opset_version", "input_names", "input_shapes",
        "input_dtypes", "output_name", "converter_backend",
        "converter_version", "onnx_model_sha256", "onnx_model_bytes",
        "export_executed", "onnxruntime_readback_executed",
        "onnxruntime_version", "case_count", "python_probabilities_sha256",
        "onnx_probabilities_sha256", "python_probabilities_file_sha256",
        "onnx_probabilities_file_sha256", "absolute_tolerance",
        "relative_tolerance", "maximum_absolute_error",
        "maximum_relative_error", "mean_absolute_error",
        "argmax_class_order_match", "parity_passed",
        "trusted_context_baseline_sha256", "trusted_context_checkpoints",
    }
)


class OnnxWinnerUnsupported(RuntimeError):
    """A model family or concrete converter path has no faithful ONNX route."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(f"{reason_code}:{detail}")
        self.reason_code = str(reason_code)
        self.detail = str(detail)


@dataclass(frozen=True)
class _TrustedContext:
    paths: Any
    bundle_directory: Path
    selection: dict[str, Any]
    config: Any
    manifest: dict[str, Any]
    dependency_gate: dict[str, Any]
    source_state: dict[str, Any]
    source_snapshot_sha256: str
    code_commit: str
    bundle_manifest_sha256: str
    final_refit_attestation: dict[str, Any]
    final_refit_attestation_file_sha256: str


def _trusted_context_identity(context: _TrustedContext) -> dict[str, Any]:
    """Freeze every source/dependency/selection/bundle authority input."""

    payload = {
        "source_state": context.source_state,
        "source_snapshot_sha256": context.source_snapshot_sha256,
        "code_commit": context.code_commit,
        "dependency_gate": context.dependency_gate,
        "selection_record_file_sha256":
            context.selection["selection_record_file_sha256"],
        "manual_selection_sha256":
            context.selection["manual_selection_sha256"],
        "config_hash": context.config.sha256,
        "bundle_manifest_sha256": context.bundle_manifest_sha256,
        "final_refit_attestation_file_sha256":
            context.final_refit_attestation_file_sha256,
    }
    return {
        **payload,
        "trusted_context_sha256": stable_payload_sha256(payload),
    }


def _trusted_context_checkpoint(
    phase: str,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "phase": str(phase),
        "trusted_context_sha256": identity["trusted_context_sha256"],
    }
    return {**payload, "checkpoint_sha256": stable_payload_sha256(payload)}


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in _HEX for character in text)


def _strict_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"ONNX artifact overwrite forbidden: {path}")
    path.write_text(
        json.dumps(
            to_strict_json_value(dict(payload)),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant forbidden: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=lambda pairs: _unique_object(pairs),
    )
    if not isinstance(value, dict):
        raise ValueError(f"strict JSON root must be an object: {path}")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key forbidden: {key}")
        output[key] = value
    return output


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256(b"ppg_frailty_onnx_array_v2\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _runtime_versions() -> dict[str, str]:
    def version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "not_installed"

    return {
        "onnx": version("onnx"),
        "onnxruntime": version("onnxruntime"),
        "skl2onnx": version("skl2onnx"),
        "scikit_learn": version("scikit-learn"),
        "torch": version("torch"),
        "numpy": str(np.__version__),
    }


def _producer_source_sha256(pipeline_root: Path) -> str:
    source = (pipeline_root / PRODUCER_SOURCE_RELATIVE_PATH).resolve()
    source.relative_to(pipeline_root.resolve())
    return sha256_file(source)


def _write_artifact_index(directory: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.name == "artifact_index.json":
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                f"ONNX producer staging contains non-regular entry: {path.name}"
            )
        artifacts.append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    payload = {
        "schema_version": ARTIFACT_INDEX_SCHEMA,
        "pipeline_generation": "final_pipeline_v2",
        "artifacts": artifacts,
        "index_payload_sha256": stable_payload_sha256(artifacts),
    }
    _strict_json(directory / "artifact_index.json", payload)
    return payload


def _verify_artifact_index(directory: Path) -> dict[str, Any]:
    index_path = directory / "artifact_index.json"
    payload = _load_json(index_path)
    if (
        payload.get("schema_version") != ARTIFACT_INDEX_SCHEMA
        or payload.get("pipeline_generation") != "final_pipeline_v2"
    ):
        raise ValueError("ONNX producer artifact index contract drift")
    rows = payload.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise ValueError("ONNX producer artifact index is empty")
    expected_names: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("ONNX producer artifact index row invalid")
        name = str(row.get("path", ""))
        if Path(name).name != name or name in expected_names:
            raise ValueError("ONNX producer artifact index path invalid")
        target = directory / name
        if (
            not target.is_file()
            or target.is_symlink()
            or int(row.get("bytes", -1)) != target.stat().st_size
            or str(row.get("sha256", "")) != sha256_file(target)
        ):
            raise ValueError(f"ONNX producer artifact hash drift: {name}")
        expected_names.add(name)
    entries = tuple(directory.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in entries):
        raise ValueError("ONNX producer contains directory or symlink entry")
    actual_names = {path.name for path in entries}
    if actual_names != expected_names | {"artifact_index.json"}:
        raise ValueError("ONNX producer artifact file set drift")
    if payload.get("index_payload_sha256") != stable_payload_sha256(rows):
        raise ValueError("ONNX producer artifact index payload hash drift")
    return payload


def _verify_bundle_manifest_files(bundle: Path, manifest: Mapping[str, Any]) -> None:
    file_hashes = manifest.get("file_hashes")
    if not isinstance(file_hashes, Mapping) or not file_hashes:
        raise ValueError("winner bundle file hash manifest missing")
    for raw_name, expected_sha in file_hashes.items():
        name = str(raw_name)
        if Path(name).name != name or not _is_sha256(expected_sha):
            raise ValueError("winner bundle file identity invalid")
        artifact = bundle / name
        if (
            not artifact.is_file()
            or artifact.is_symlink()
            or sha256_file(artifact) != str(expected_sha)
        ):
            raise ValueError(f"winner bundle artifact hash mismatch: {name}")


def _trusted_context(
    *,
    bundle_directory: str | Path,
    selection_record: str | Path,
    expected_selection_file_sha256: str,
    expected_final_bundle_manifest_sha256: str,
    final_refit_attestation_path: str | Path,
    expected_final_refit_attestation_sha256: str,
    config_path: str | Path,
) -> _TrustedContext:
    """Verify source, exact live dependencies, selection, and bundle before load."""

    from .config import dependency_gate_report, load_config
    from .experiment import (
        _code_commit,
        _require_scientific_source_gate,
        _source_snapshot_sha256,
        verify_manual_selection_record,
    )
    from .pipeline import PipelinePaths

    paths = PipelinePaths.discover()
    root = paths.pipeline_root.resolve()
    bundle = Path(bundle_directory).resolve()
    selection_path = Path(selection_record).resolve()
    attestation_path = Path(final_refit_attestation_path).resolve()
    config_file = Path(config_path).resolve()
    for candidate in (bundle, selection_path, attestation_path, config_file):
        candidate.relative_to(root)
    try:
        config_file.relative_to(root / "configs")
    except ValueError as exc:
        raise ValueError(
            "ONNX formal config must resolve inside tracked configs"
        ) from exc
    selection = verify_manual_selection_record(
        selection_path,
        expected_file_sha256=expected_selection_file_sha256,
    )
    config = load_config(config_file)
    if (
        config.config_id != selection["config_id"]
        or config.sha256 != selection["config_hash"]
    ):
        raise ValueError("ONNX config identity differs from manual selection")
    source_state = _require_scientific_source_gate(paths)
    source_snapshot = _source_snapshot_sha256(paths)
    code_commit = _code_commit(paths)
    dependency_gate = dependency_gate_report(
        config,
        operation="onnx_winner_gate",
        profiles_path=root / "requirements/profiles.json",
        lock_path=root / "locks/profiles.lock.json",
        require_exact_lock=True,
    )
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"winner bundle manifest missing: {manifest_path}")
    manifest = _load_json(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != str(expected_final_bundle_manifest_sha256):
        raise ValueError("final_bundle_manifest_sha256_mismatch")
    _verify_bundle_manifest_files(bundle, manifest)
    from .training.bundle import (
        FINAL_REFIT_ATTESTATION_SCHEMA,
        TRUSTED_FINAL_REFIT_BUNDLE_KIND,
        _validate_trusted_final_refit_manifest,
    )

    if manifest.get("bundle_kind") != TRUSTED_FINAL_REFIT_BUNDLE_KIND:
        raise ValueError("winner requires trusted_final_refit_v2 bundle kind")
    strict_final_identity = _validate_trusted_final_refit_manifest(manifest)
    expected_attestation_path = bundle / "final_refit_attestation.json"
    if attestation_path != expected_attestation_path:
        raise ValueError("final_refit_attestation_path_not_canonical")
    observed_attestation_sha = sha256_file(attestation_path)
    if observed_attestation_sha != str(expected_final_refit_attestation_sha256):
        raise ValueError("final_refit_attestation_sha256_mismatch")
    attestation = _load_json(attestation_path)
    expected_attestation_fields = {
        "schema_version", "pipeline_generation", "status", "bundle_kind",
        "bundle_manifest_relative_path", "bundle_manifest_sha256",
        "bundle_manifest_bytes", "selection_record_file_sha256",
        "manual_selection_sha256", "config_hash", "model_id",
        "execution_hash", "dataset_hash", "bundle_materialization_hash",
        "source_snapshot_sha256", "scientific_source_gate_sha256",
        "participant_count", "participant_ids", "training_seeds",
        "attestation_payload_sha256",
    }
    payload_without_hash = {
        key: value
        for key, value in attestation.items()
        if key != "attestation_payload_sha256"
    }
    if (
        set(attestation) != expected_attestation_fields
        or attestation.get("schema_version") != FINAL_REFIT_ATTESTATION_SCHEMA
        or attestation.get("pipeline_generation") != "final_pipeline_v2"
        or attestation.get("status")
        != "verified_trusted_final_refit_publication"
        or attestation.get("bundle_kind") != TRUSTED_FINAL_REFIT_BUNDLE_KIND
        or attestation.get("bundle_manifest_relative_path")
        != "manifest.json"
        or attestation.get("bundle_manifest_sha256") != manifest_sha
        or attestation.get("bundle_manifest_bytes") != manifest_path.stat().st_size
        or attestation.get("selection_record_file_sha256")
        != selection["selection_record_file_sha256"]
        or attestation.get("manual_selection_sha256")
        != selection["manual_selection_sha256"]
        or attestation.get("config_hash") != selection["config_hash"]
        or attestation.get("model_id") != strict_final_identity["model_id"]
        or attestation.get("execution_hash")
        != strict_final_identity["execution_hash"]
        or attestation.get("dataset_hash")
        != strict_final_identity["dataset_hash"]
        or attestation.get("bundle_materialization_hash")
        != strict_final_identity["bundle_materialization_hash"]
        or attestation.get("source_snapshot_sha256")
        != strict_final_identity["source_snapshot_hash"]
        or attestation.get("scientific_source_gate_sha256")
        != strict_final_identity["scientific_source_gate_sha256"]
        or attestation.get("participant_count") != 29
        or attestation.get("participant_ids")
        != strict_final_identity["participant_ids"]
        or attestation.get("training_seeds")
        != strict_final_identity["training_seeds"]
        or attestation.get("attestation_payload_sha256")
        != stable_payload_sha256(payload_without_hash)
    ):
        raise ValueError("final_refit_attestation_identity_drift")
    metadata = manifest.get("metadata")
    final_identity = (
        metadata.get("final_refit_identity")
        if isinstance(metadata, Mapping) else None
    )
    if (
        manifest.get("pipeline_generation") != "final_pipeline_v2"
        or manifest.get("config_hash") != selection["config_hash"]
        or manifest.get("machine_model_id") != selection["model_machine_id"]
        or manifest.get("source_snapshot_hash") != source_snapshot
        or not isinstance(final_identity, Mapping)
        or final_identity.get("manual_selection_hash")
        != selection["manual_selection_sha256"]
        or final_identity.get("config_hash") != selection["config_hash"]
        or final_identity.get("execution_hash") != manifest.get("run_hash")
        or manifest.get("golden_parity_atol") != FINAL_BUNDLE_PARITY_ATOL
    ):
        raise ValueError("winner bundle selection/source/refit identity mismatch")
    return _TrustedContext(
        paths=paths,
        bundle_directory=bundle,
        selection=selection,
        config=config,
        manifest=manifest,
        dependency_gate=dependency_gate,
        source_state=source_state,
        source_snapshot_sha256=source_snapshot,
        code_commit=code_commit,
        bundle_manifest_sha256=manifest_sha,
        final_refit_attestation=attestation,
        final_refit_attestation_file_sha256=observed_attestation_sha,
    )


def _golden_model_inputs(loaded: Any) -> tuple[dict[str, np.ndarray], np.ndarray]:
    from .training.bundle import predict_bundle

    golden_path = loaded.directory / "golden.npz"
    with np.load(golden_path, allow_pickle=False) as archive:
        raw_inputs = {
            key.removeprefix("input__"): np.asarray(archive[key])
            for key in archive.files
            if key.startswith("input__")
        }
        expected = np.asarray(archive["expected_probabilities"], dtype=np.float64)
    transforms = loaded.transforms
    if (
        transforms is None
        or getattr(transforms, "boundary", None)
        != "already_preprocessed_and_fitted_transforms_applied_model_input"
        or not hasattr(transforms, "transform_inputs")
    ):
        raise OnnxWinnerUnsupported(
            "non_identity_model_input_transform",
            "winner ONNX boundary requires the frozen final identity transform archive",
        )
    model_inputs = {
        str(name): np.asarray(value)
        for name, value in transforms.transform_inputs(raw_inputs).items()
    }
    probability = np.asarray(predict_bundle(loaded, raw_inputs), dtype=np.float64)
    if (
        expected.shape != probability.shape
        or not np.allclose(
            expected,
            probability,
            atol=FINAL_BUNDLE_PARITY_ATOL,
            rtol=0.0,
        )
    ):
        raise RuntimeError("winner bundle golden probability drift before ONNX export")
    return model_inputs, probability


def _ordered_input_names(inputs: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    names = set(inputs)
    if names == {"x"}:
        return ("x",)
    if names == {"x", "mask"}:
        return ("x", "mask")
    fusion = {"window_bag", "window_mask", "file_features", "sample_mask"}
    if names == fusion:
        return ("window_bag", "window_mask", "file_features", "sample_mask")
    raise OnnxWinnerUnsupported(
        "unsupported_model_input_mapping",
        f"model-input keys are {sorted(names)}",
    )


def _torch_export(
    model: Any,
    inputs: Mapping[str, np.ndarray],
    input_names: tuple[str, ...],
    output_path: Path,
    *,
    opset_version: int,
) -> tuple[str, str]:
    import torch

    class ProbabilityBoundary(torch.nn.Module):
        def __init__(self, inner: Any, names: tuple[str, ...]) -> None:
            super().__init__()
            self.inner = inner
            self.names = names

        def forward(self, *values: Any) -> Any:
            mapped = dict(zip(self.names, values))
            if "window_bag" in mapped:
                logits = self.inner(
                    mapped["window_bag"].float(),
                    mapped["window_mask"].bool(),
                    mapped["file_features"].float(),
                    mapped["sample_mask"].bool(),
                )
                return torch.softmax(logits, dim=-1)
            mask = mapped.get("mask")
            typed_mask = None if mask is None else mask.bool()
            if hasattr(self.inner, "predict_probabilities"):
                return self.inner.predict_probabilities(
                    mapped["x"].float(), typed_mask
                )
            logits = self.inner(mapped["x"].float(), typed_mask)
            return torch.softmax(logits, dim=-1)

    wrapper = ProbabilityBoundary(model.eval(), input_names).eval()
    arguments = tuple(
        torch.as_tensor(inputs[name], device="cpu")
        for name in input_names
    )
    dynamic_axes = {name: {0: "batch"} for name in input_names}
    dynamic_axes["probabilities"] = {0: "batch"}
    try:
        torch.onnx.export(
            wrapper,
            arguments,
            output_path,
            input_names=list(input_names),
            output_names=["probabilities"],
            dynamic_axes=dynamic_axes,
            opset_version=int(opset_version),
            export_params=True,
            do_constant_folding=True,
            dynamo=False,
        )
    except Exception as exc:  # converter support varies by reviewed architecture.
        raise OnnxWinnerUnsupported(
            "torch_onnx_export_unsupported",
            f"{type(exc).__name__}:{exc}",
        ) from exc
    return "torch.onnx.export_legacy", importlib.metadata.version("torch")


def _sklearn_export(
    model: Any,
    inputs: Mapping[str, np.ndarray],
    input_names: tuple[str, ...],
    output_path: Path,
    *,
    opset_version: int,
    class_order: tuple[int, ...],
) -> tuple[str, str]:
    if input_names != ("x",):
        raise OnnxWinnerUnsupported(
            "sklearn_mask_or_multi_input_not_supported",
            f"estimator input names are {input_names}",
        )
    observed_classes = tuple(int(value) for value in getattr(model, "classes_", ()))
    if observed_classes != class_order:
        raise OnnxWinnerUnsupported(
            "sklearn_class_order_unavailable_or_drifted",
            f"observed classes are {observed_classes}",
        )
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import (
            DoubleTensorType,
            FloatTensorType,
        )

        x = np.asarray(inputs["x"])
        if x.ndim != 2 or x.dtype not in {np.dtype("float32"), np.dtype("float64")}:
            raise OnnxWinnerUnsupported(
                "sklearn_input_dtype_or_rank_unsupported",
                f"x has shape={x.shape}, dtype={x.dtype}",
            )
        tensor_type = (
            FloatTensorType([None, int(x.shape[1])])
            if x.dtype == np.dtype("float32")
            else DoubleTensorType([None, int(x.shape[1])])
        )
        converted = convert_sklearn(
            model,
            name="ppg_frailty_winner_probability_graph_v2",
            initial_types=[("x", tensor_type)],
            target_opset=int(opset_version),
            options={id(model): {"zipmap": False}},
        )
        output_path.write_bytes(converted.SerializeToString())
    except OnnxWinnerUnsupported:
        raise
    except Exception as exc:
        raise OnnxWinnerUnsupported(
            "sklearn_onnx_converter_unsupported",
            f"{type(exc).__name__}:{exc}",
        ) from exc
    return "skl2onnx.convert_sklearn", importlib.metadata.version("skl2onnx")


def _ort_type_cast(value: np.ndarray, type_name: str) -> np.ndarray:
    mapping = {
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(bool)": np.bool_,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
    }
    if type_name not in mapping:
        raise OnnxWinnerUnsupported(
            "onnxruntime_input_type_unsupported", type_name
        )
    return np.asarray(value, dtype=mapping[type_name])


def _ort_readback(
    model_path: Path,
    inputs: Mapping[str, np.ndarray],
    *,
    class_count: int,
) -> tuple[np.ndarray, str]:
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    declared = {item.name: item for item in session.get_inputs()}
    if set(declared) != set(inputs):
        raise RuntimeError(
            "ONNX Runtime input names differ from exported model-input mapping"
        )
    feed = {
        name: _ort_type_cast(np.asarray(inputs[name]), declared[name].type)
        for name in declared
    }
    outputs = session.run(None, feed)
    output_meta = session.get_outputs()
    candidates = [
        (np.asarray(value), meta.name)
        for value, meta in zip(outputs, output_meta)
        if isinstance(value, np.ndarray)
        and value.ndim == 2
        and value.shape[1] == int(class_count)
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "ONNX Runtime must expose exactly one [sample,class] probability output"
        )
    return np.asarray(candidates[0][0], dtype=np.float64), candidates[0][1]


def _parity_metrics(
    python_probabilities: np.ndarray,
    onnx_probabilities: np.ndarray,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    reference = np.asarray(python_probabilities, dtype=np.float64)
    candidate = np.asarray(onnx_probabilities, dtype=np.float64)
    if (
        reference.ndim != 2
        or candidate.shape != reference.shape
        or reference.shape[1] < 2
    ):
        raise ValueError("ONNX/Python probability shapes must match [sample,class]")
    for name, value in (("python", reference), ("onnx", candidate)):
        if (
            not np.isfinite(value).all()
            or np.any(value < 0.0)
            or not np.allclose(value.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
        ):
            raise ValueError(f"{name} output is not a finite probability matrix")
    absolute = np.abs(reference - candidate)
    denominator = np.maximum(np.abs(reference), 1e-12)
    relative = absolute / denominator
    maximum_absolute = float(np.max(absolute))
    maximum_relative = float(np.max(relative))
    mean_absolute = float(np.mean(absolute))
    argmax_match = bool(
        np.array_equal(np.argmax(reference, axis=1), np.argmax(candidate, axis=1))
    )
    passed = bool(
        maximum_absolute <= float(absolute_tolerance)
        and maximum_relative <= float(relative_tolerance)
        and argmax_match
    )
    return {
        "absolute_tolerance": float(absolute_tolerance),
        "relative_tolerance": float(relative_tolerance),
        "maximum_absolute_error": maximum_absolute,
        "maximum_relative_error": maximum_relative,
        "mean_absolute_error": mean_absolute,
        "argmax_class_order_match": argmax_match,
        "parity_passed": passed,
    }


def validate_certificate_parity_metrics(certificate: Mapping[str, Any]) -> None:
    """Reject a caller-authored pass flag unless both frozen bounds really pass."""

    import math

    names = (
        "absolute_tolerance", "relative_tolerance", "maximum_absolute_error",
        "maximum_relative_error", "mean_absolute_error",
    )
    try:
        numeric = {name: float(certificate[name]) for name in names}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("ONNX winner certificate parity fields invalid") from exc
    if (
        any(not math.isfinite(value) or value < 0.0 for value in numeric.values())
        or numeric["absolute_tolerance"] != ONNX_WINNER_ABSOLUTE_TOLERANCE
        or numeric["relative_tolerance"] != ONNX_WINNER_RELATIVE_TOLERANCE
        or numeric["mean_absolute_error"] > numeric["maximum_absolute_error"]
        or numeric["maximum_absolute_error"] > numeric["absolute_tolerance"]
        or numeric["maximum_relative_error"] > numeric["relative_tolerance"]
        or certificate.get("argmax_class_order_match") is not True
        or certificate.get("parity_passed") is not True
    ):
        raise ValueError("onnx_winner_certificate_parity_metrics_invalid")


def validate_certificate_export_policy(certificate: Mapping[str, Any]) -> None:
    """Require the independently frozen exporter policy before trusting parity."""

    if certificate.get("opset_version") != ONNX_WINNER_OPSET_VERSION:
        raise ValueError("onnx_winner_certificate_opset_policy_invalid")
    validate_certificate_parity_metrics(certificate)


def _unsupported_payload(
    context: _TrustedContext,
    error: OnnxWinnerUnsupported,
) -> dict[str, Any]:
    return {
        "schema_version": "ppg_frailty.onnx_winner_unsupported.v2",
        "pipeline_generation": "final_pipeline_v2",
        "status": "unsupported_no_certificate",
        "producer_entry_id": PRODUCER_ENTRY_ID,
        "producer_source_sha256": _producer_source_sha256(
            context.paths.pipeline_root
        ),
        "config_id": context.selection["config_id"],
        "config_hash": context.selection["config_hash"],
        "model_machine_id": context.selection["model_machine_id"],
        "manual_selection_sha256":
            context.selection["manual_selection_sha256"],
        "bundle_manifest_sha256": sha256_file(
            context.bundle_directory / "manifest.json"
        ),
        "reason_code": error.reason_code,
        "detail": error.detail,
        "certificate_emitted": False,
        "winner_release_ready": False,
        "training_executed": False,
    }


def produce_onnx_winner_certificate(
    *,
    bundle_directory: str | Path,
    selection_record: str | Path,
    expected_selection_file_sha256: str,
    expected_final_bundle_manifest_sha256: str,
    final_refit_attestation_path: str | Path,
    expected_final_refit_attestation_sha256: str,
    config_path: str | Path,
    output_directory: str | Path,
    confirm_onnx_execution: bool,
) -> dict[str, Any]:
    """Export and read back a manually selected final bundle; never train."""

    if not confirm_onnx_execution:
        raise PermissionError(
            "winner ONNX export requires --confirm-onnx-execution"
        )
    context_arguments = {
        "bundle_directory": bundle_directory,
        "selection_record": selection_record,
        "expected_selection_file_sha256": expected_selection_file_sha256,
        "expected_final_bundle_manifest_sha256":
            expected_final_bundle_manifest_sha256,
        "final_refit_attestation_path": final_refit_attestation_path,
        "expected_final_refit_attestation_sha256":
            expected_final_refit_attestation_sha256,
        "config_path": config_path,
    }
    context = _trusted_context(**context_arguments)
    baseline_identity = _trusted_context_identity(context)
    context_checkpoints = [
        _trusted_context_checkpoint("entry_preflight", baseline_identity)
    ]
    root = context.paths.pipeline_root.resolve()
    target = Path(output_directory).resolve()
    try:
        output_relative = target.relative_to(root)
    except ValueError as exc:
        raise ValueError("ONNX winner output must remain inside V2 artifacts") from exc
    if not output_relative.parts or output_relative.parts[0] != "artifacts":
        raise ValueError("ONNX winner output must remain inside V2 artifacts")
    if target.exists():
        raise FileExistsError(f"ONNX winner output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        from .training.bundle import load_bundle

        loaded = load_bundle(context.bundle_directory)
        model_inputs, python_probabilities = _golden_model_inputs(loaded)
        input_names = _ordered_input_names(model_inputs)
        class_order = tuple(
            int(value) for value in context.manifest["metadata"]["class_order"]
        )
        if class_order != tuple(range(len(class_order))):
            raise OnnxWinnerUnsupported(
                "noncanonical_class_order",
                f"class order is {class_order}",
            )
        model_path = staging / "winner.onnx"
        kind = str(context.manifest.get("kind", ""))
        machine_id = str(context.manifest.get("machine_model_id", ""))
        if kind == "torch":
            backend, converter_version = _torch_export(
                loaded.model,
                model_inputs,
                input_names,
                model_path,
                opset_version=ONNX_WINNER_OPSET_VERSION,
            )
        elif kind == "estimator":
            if machine_id not in _ESTIMATOR_CONVERTER_MODEL_IDS:
                raise OnnxWinnerUnsupported(
                    "estimator_family_has_no_reviewed_converter",
                    machine_id,
                )
            backend, converter_version = _sklearn_export(
                loaded.model,
                model_inputs,
                input_names,
                model_path,
                opset_version=ONNX_WINNER_OPSET_VERSION,
                class_order=class_order,
            )
        else:
            raise OnnxWinnerUnsupported(
                "bundle_model_kind_unsupported", kind
            )
        import onnx

        onnx.checker.check_model(onnx.load(str(model_path)))
        onnx_probabilities, output_name = _ort_readback(
            model_path,
            model_inputs,
            class_count=len(class_order),
        )
        parity = _parity_metrics(
            python_probabilities,
            onnx_probabilities,
            absolute_tolerance=ONNX_WINNER_ABSOLUTE_TOLERANCE,
            relative_tolerance=ONNX_WINNER_RELATIVE_TOLERANCE,
        )
        if parity["parity_passed"] is not True:
            raise RuntimeError(
                "winner ONNX export failed probability/class-order parity"
            )
        reverified = _trusted_context(**context_arguments)
        reverified_identity = _trusted_context_identity(reverified)
        if reverified_identity != baseline_identity:
            raise RuntimeError(
                "onnx_trusted_context_changed_after_export_or_readback"
            )
        context = reverified
        context_checkpoints.append(
            _trusted_context_checkpoint(
                "post_export_ort_prepublish",
                reverified_identity,
            )
        )
        python_path = staging / "python_probabilities.npy"
        onnx_path = staging / "onnx_probabilities.npy"
        np.save(python_path, python_probabilities, allow_pickle=False)
        np.save(onnx_path, onnx_probabilities, allow_pickle=False)
        manifest_path = context.bundle_directory / "manifest.json"
        final_identity = context.manifest["metadata"]["final_refit_identity"]
        dependency_lock_path = root / "locks/profiles.lock.json"
        certificate = {
            "schema_version": CERTIFICATE_SCHEMA,
            "pipeline_generation": "final_pipeline_v2",
            "status": "passed",
            "producer_entry_id": PRODUCER_ENTRY_ID,
            "producer_source_relative_path": PRODUCER_SOURCE_RELATIVE_PATH,
            "producer_source_sha256": _producer_source_sha256(root),
            "selection_record_file_sha256":
                context.selection["selection_record_file_sha256"],
            "manual_selection_sha256":
                context.selection["manual_selection_sha256"],
            "config_id": context.selection["config_id"],
            "config_hash": context.selection["config_hash"],
            "model_machine_id": machine_id,
            "final_refit_execution_hash": final_identity["execution_hash"],
            "bundle_manifest_sha256": sha256_file(manifest_path),
            "bundle_manifest_bytes": manifest_path.stat().st_size,
            "final_refit_attestation_file_sha256":
                context.final_refit_attestation_file_sha256,
            "final_refit_attestation_payload_sha256":
                context.final_refit_attestation["attestation_payload_sha256"],
            "bundle_run_hash": context.manifest["run_hash"],
            "bundle_golden_case_sha256": context.manifest["golden_case_hash"],
            "input_spec_hash": context.manifest["input_spec_hash"],
            "source_snapshot_sha256": context.source_snapshot_sha256,
            "code_commit": context.code_commit,
            "git_status_sha256": context.source_state["status_sha256"],
            "dependency_gate_sha256":
                stable_payload_sha256(context.dependency_gate),
            "dependency_lock_file_sha256": sha256_file(dependency_lock_path),
            "dependency_profile_ids":
                list(context.dependency_gate["required_profile_ids"]),
            "trusted_context_baseline_sha256":
                baseline_identity["trusted_context_sha256"],
            "trusted_context_checkpoints": context_checkpoints,
            "runtime_versions": _runtime_versions(),
            "class_order": list(class_order),
            "boundary": BOUNDARY,
            "opset_version": ONNX_WINNER_OPSET_VERSION,
            "input_names": list(input_names),
            "input_shapes": {
                name: list(np.asarray(model_inputs[name]).shape)
                for name in input_names
            },
            "input_dtypes": {
                name: str(np.asarray(model_inputs[name]).dtype)
                for name in input_names
            },
            "output_name": output_name,
            "converter_backend": backend,
            "converter_version": converter_version,
            "onnx_model_sha256": sha256_file(model_path),
            "onnx_model_bytes": model_path.stat().st_size,
            "export_executed": True,
            "onnxruntime_readback_executed": True,
            "onnxruntime_version": importlib.metadata.version("onnxruntime"),
            "case_count": int(python_probabilities.shape[0]),
            "python_probabilities_sha256":
                _array_sha256(python_probabilities),
            "onnx_probabilities_sha256":
                _array_sha256(onnx_probabilities),
            "python_probabilities_file_sha256": sha256_file(python_path),
            "onnx_probabilities_file_sha256": sha256_file(onnx_path),
            **parity,
        }
        if set(certificate) != _CERTIFICATE_FIELDS:
            raise RuntimeError("internal ONNX certificate field schema drift")
        validate_certificate_export_policy(certificate)
        certificate_path = staging / "certificate.json"
        _strict_json(certificate_path, certificate)
        _write_artifact_index(staging)
        _verify_artifact_index(staging)
        if {path.name for path in staging.iterdir()} != {
            "winner.onnx", "certificate.json", "python_probabilities.npy",
            "onnx_probabilities.npy", "artifact_index.json",
        }:
            raise RuntimeError("ONNX success staging artifact set drift")
        if target.exists():
            raise FileExistsError(f"ONNX winner target appeared: {target}")
        if (
            _trusted_context_identity(
                _trusted_context(**context_arguments)
            ) != baseline_identity
        ):
            raise RuntimeError(
                "onnx_trusted_context_changed_before_atomic_publish"
            )
        os.rename(staging, target)
        return {
            "schema_version": "ppg_frailty.onnx_winner_production.v2",
            "pipeline_generation": "final_pipeline_v2",
            "status": "certificate_produced_pending_release_preflight",
            "output_directory": str(target),
            "certificate_file_sha256": sha256_file(
                target / "certificate.json"
            ),
            "artifact_index_sha256": sha256_file(
                target / "artifact_index.json"
            ),
            "winner_release_ready": False,
            "training_executed": False,
        }
    except OnnxWinnerUnsupported as error:
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        payload = _unsupported_payload(context, error)
        _strict_json(staging / "unsupported.json", payload)
        _write_artifact_index(staging)
        _verify_artifact_index(staging)
        if {path.name for path in staging.iterdir()} != {
            "unsupported.json", "artifact_index.json",
        }:
            raise RuntimeError("ONNX unsupported staging artifact set drift")
        if target.exists():
            raise FileExistsError(f"ONNX winner target appeared: {target}")
        if (
            _trusted_context_identity(
                _trusted_context(**context_arguments)
            ) != baseline_identity
        ):
            raise RuntimeError(
                "onnx_trusted_context_changed_before_unsupported_publish"
            )
        os.rename(staging, target)
        return {
            **payload,
            "output_directory": str(target),
            "artifact_index_sha256": sha256_file(
                target / "artifact_index.json"
            ),
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def winner_release_preflight(
    *,
    bundle_directory: str | Path,
    onnx_model_path: str | Path,
    certificate_path: str | Path,
    expected_certificate_file_sha256: str,
    selection_record: str | Path,
    expected_selection_file_sha256: str,
    expected_final_bundle_manifest_sha256: str,
    final_refit_attestation_path: str | Path,
    expected_final_refit_attestation_sha256: str,
    config_path: str | Path,
) -> dict[str, Any]:
    """Re-run the trusted readback and verify every certificate binding."""

    from .pipeline import PipelinePaths

    producer_root = Path(certificate_path).resolve().parent
    model_path = Path(onnx_model_path).resolve()
    certificate_file = Path(certificate_path).resolve()
    root = PipelinePaths.discover().pipeline_root.resolve()
    for path in (producer_root, model_path, certificate_file):
        path.relative_to(root)
    producer_relative = producer_root.relative_to(root)
    if (
        not producer_relative.parts
        or producer_relative.parts[0] != "artifacts"
    ):
        raise ValueError("winner ONNX producer must reside below artifacts")
    if (
        model_path != producer_root / "winner.onnx"
        or certificate_file != producer_root / "certificate.json"
    ):
        raise ValueError("winner ONNX model/certificate must use canonical producer names")
    _verify_artifact_index(producer_root)
    producer_entries = tuple(producer_root.iterdir())
    if (
        any(not path.is_file() or path.is_symlink() for path in producer_entries)
        or {path.name for path in producer_entries} != {
        "winner.onnx", "certificate.json", "python_probabilities.npy",
        "onnx_probabilities.npy", "artifact_index.json",
        }
    ):
        raise ValueError("winner ONNX producer success artifact set drift")
    context = _trusted_context(
        bundle_directory=bundle_directory,
        selection_record=selection_record,
        expected_selection_file_sha256=expected_selection_file_sha256,
        expected_final_bundle_manifest_sha256=
            expected_final_bundle_manifest_sha256,
        final_refit_attestation_path=final_refit_attestation_path,
        expected_final_refit_attestation_sha256=
            expected_final_refit_attestation_sha256,
        config_path=config_path,
    )
    observed_certificate_sha = sha256_file(certificate_file)
    if observed_certificate_sha != str(expected_certificate_file_sha256):
        raise ValueError("onnx_winner_certificate_file_sha256_mismatch")
    certificate = _load_json(certificate_file)
    if set(certificate) != _CERTIFICATE_FIELDS:
        raise ValueError("onnx_winner_certificate_field_schema_drift")
    manifest_path = context.bundle_directory / "manifest.json"
    final_identity = context.manifest["metadata"]["final_refit_identity"]
    lock_path = root / "locks/profiles.lock.json"
    current_context_identity = _trusted_context_identity(context)
    expected_context_checkpoints = [
        _trusted_context_checkpoint(
            "entry_preflight",
            current_context_identity,
        ),
        _trusted_context_checkpoint(
            "post_export_ort_prepublish",
            current_context_identity,
        ),
    ]
    expected = {
        "schema_version": CERTIFICATE_SCHEMA,
        "pipeline_generation": "final_pipeline_v2",
        "status": "passed",
        "producer_entry_id": PRODUCER_ENTRY_ID,
        "producer_source_relative_path": PRODUCER_SOURCE_RELATIVE_PATH,
        "producer_source_sha256": _producer_source_sha256(root),
        "selection_record_file_sha256":
            context.selection["selection_record_file_sha256"],
        "manual_selection_sha256":
            context.selection["manual_selection_sha256"],
        "config_id": context.selection["config_id"],
        "config_hash": context.selection["config_hash"],
        "model_machine_id": context.selection["model_machine_id"],
        "final_refit_execution_hash": final_identity["execution_hash"],
        "bundle_manifest_sha256": sha256_file(manifest_path),
        "bundle_manifest_bytes": manifest_path.stat().st_size,
        "final_refit_attestation_file_sha256":
            context.final_refit_attestation_file_sha256,
        "final_refit_attestation_payload_sha256":
            context.final_refit_attestation["attestation_payload_sha256"],
        "bundle_run_hash": context.manifest["run_hash"],
        "bundle_golden_case_sha256": context.manifest["golden_case_hash"],
        "input_spec_hash": context.manifest["input_spec_hash"],
        "source_snapshot_sha256": context.source_snapshot_sha256,
        "code_commit": context.code_commit,
        "git_status_sha256": context.source_state["status_sha256"],
        "dependency_gate_sha256":
            stable_payload_sha256(context.dependency_gate),
        "dependency_lock_file_sha256": sha256_file(lock_path),
        "dependency_profile_ids":
            list(context.dependency_gate["required_profile_ids"]),
        "trusted_context_baseline_sha256":
            current_context_identity["trusted_context_sha256"],
        "trusted_context_checkpoints": expected_context_checkpoints,
        "runtime_versions": _runtime_versions(),
        "class_order": list(context.manifest["metadata"]["class_order"]),
        "boundary": BOUNDARY,
        "opset_version": ONNX_WINNER_OPSET_VERSION,
        "absolute_tolerance": ONNX_WINNER_ABSOLUTE_TOLERANCE,
        "relative_tolerance": ONNX_WINNER_RELATIVE_TOLERANCE,
        "onnx_model_sha256": sha256_file(model_path),
        "onnx_model_bytes": model_path.stat().st_size,
        "export_executed": True,
        "onnxruntime_readback_executed": True,
        "onnxruntime_version": importlib.metadata.version("onnxruntime"),
    }
    for name, value in expected.items():
        if certificate.get(name) != value:
            raise ValueError(f"onnx_winner_certificate_identity_drift:{name}")
    if (
        int(certificate.get("case_count", 0)) <= 0
        or not str(certificate.get("converter_backend", "")).strip()
        or not str(certificate.get("converter_version", "")).strip()
        or not str(certificate.get("output_name", "")).strip()
    ):
        raise ValueError("onnx_winner_certificate_execution_contract_invalid")
    if context.manifest.get("kind") == "torch":
        expected_converter = (
            "torch.onnx.export_legacy",
            importlib.metadata.version("torch"),
        )
    elif context.manifest.get("kind") == "estimator":
        expected_converter = (
            "skl2onnx.convert_sklearn",
            importlib.metadata.version("skl2onnx"),
        )
    else:
        raise ValueError("onnx_winner_bundle_converter_kind_unsupported")
    if (
        certificate.get("converter_backend"),
        certificate.get("converter_version"),
    ) != expected_converter:
        raise ValueError("onnx_winner_converter_identity_drift")
    validate_certificate_export_policy(certificate)

    import onnx
    from .training.bundle import load_bundle

    onnx.checker.check_model(onnx.load(str(model_path)))
    loaded = load_bundle(context.bundle_directory)
    model_inputs, python_probabilities = _golden_model_inputs(loaded)
    input_names = _ordered_input_names(model_inputs)
    if (
        certificate["input_names"] != list(input_names)
        or certificate["input_shapes"] != {
            name: list(np.asarray(model_inputs[name]).shape)
            for name in input_names
        }
        or certificate["input_dtypes"] != {
            name: str(np.asarray(model_inputs[name]).dtype)
            for name in input_names
        }
    ):
        raise ValueError("onnx_winner_certificate_model_input_schema_drift")
    onnx_probabilities, output_name = _ort_readback(
        model_path,
        model_inputs,
        class_count=len(certificate["class_order"]),
    )
    archived_python = np.load(
        producer_root / "python_probabilities.npy", allow_pickle=False
    )
    archived_onnx = np.load(
        producer_root / "onnx_probabilities.npy", allow_pickle=False
    )
    if (
        certificate["case_count"] != int(python_probabilities.shape[0])
        or
        not np.array_equal(archived_python, python_probabilities)
        or not np.array_equal(archived_onnx, onnx_probabilities)
        or certificate["output_name"] != output_name
    ):
        raise ValueError("onnx_winner_archived_probability_readback_drift")
    probability_bindings = {
        "python_probabilities_sha256": _array_sha256(python_probabilities),
        "onnx_probabilities_sha256": _array_sha256(onnx_probabilities),
        "python_probabilities_file_sha256": sha256_file(
            producer_root / "python_probabilities.npy"
        ),
        "onnx_probabilities_file_sha256": sha256_file(
            producer_root / "onnx_probabilities.npy"
        ),
    }
    for name, value in probability_bindings.items():
        if certificate.get(name) != value:
            raise ValueError(f"onnx_winner_probability_binding_drift:{name}")
    metrics = _parity_metrics(
        python_probabilities,
        onnx_probabilities,
        absolute_tolerance=float(certificate["absolute_tolerance"]),
        relative_tolerance=float(certificate["relative_tolerance"]),
    )
    if any(certificate.get(name) != value for name, value in metrics.items()):
        raise ValueError("onnx_winner_certificate_recomputed_parity_drift")
    _verify_artifact_index(producer_root)
    if {path.name for path in producer_root.iterdir()} != {
        "winner.onnx", "certificate.json", "python_probabilities.npy",
        "onnx_probabilities.npy", "artifact_index.json",
    }:
        raise ValueError("winner ONNX producer changed during release preflight")
    final_context = _trusted_context(
        bundle_directory=bundle_directory,
        selection_record=selection_record,
        expected_selection_file_sha256=expected_selection_file_sha256,
        expected_final_bundle_manifest_sha256=
            expected_final_bundle_manifest_sha256,
        final_refit_attestation_path=final_refit_attestation_path,
        expected_final_refit_attestation_sha256=
            expected_final_refit_attestation_sha256,
        config_path=config_path,
    )
    if _trusted_context_identity(final_context) != current_context_identity:
        raise RuntimeError(
            "onnx_trusted_context_changed_during_release_preflight"
        )
    return {
        "schema_version": "ppg_frailty.winner_release_preflight.v2",
        "pipeline_generation": "final_pipeline_v2",
        "status": "ready_for_explicit_winner_release",
        "config_id": context.selection["config_id"],
        "config_hash": context.selection["config_hash"],
        "manual_selection_sha256":
            context.selection["manual_selection_sha256"],
        "final_refit_execution_hash": final_identity["execution_hash"],
        "bundle_manifest_sha256": sha256_file(manifest_path),
        "onnx_model_sha256": sha256_file(model_path),
        "onnx_certificate_file_sha256": observed_certificate_sha,
        "onnx_export_and_runtime_readback_verified": True,
        "release_executed": False,
    }


__all__ = [
    "ARTIFACT_INDEX_SCHEMA",
    "BOUNDARY",
    "CERTIFICATE_SCHEMA",
    "OnnxWinnerUnsupported",
    "PRODUCER_ENTRY_ID",
    "produce_onnx_winner_certificate",
    "validate_certificate_export_policy",
    "validate_certificate_parity_metrics",
    "winner_release_preflight",
]
