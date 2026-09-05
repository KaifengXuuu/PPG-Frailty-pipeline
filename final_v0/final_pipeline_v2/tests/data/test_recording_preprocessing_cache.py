"""Recording-level preprocessing cache safety and concurrency tests."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

import numpy as np


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_ROOT / "src"))

from ppg_frailty.data.recording_cache import (  # noqa: E402
    ImmutableCacheConflictError,
    NamedSourceDependency,
    OrderedModuleSpec,
    RecordingCacheBuild,
    RecordingCacheAccessError,
    RecordingCacheCorruptionError,
    RecordingCacheIdentity,
    RecordingPreprocessingCache,
)
def _identity(
    *,
    modules: tuple[OrderedModuleSpec, ...] | None = None,
    dependencies: tuple[NamedSourceDependency, ...] | None = None,
) -> RecordingCacheIdentity:
    return RecordingCacheIdentity(
        namespace="stage5",
        layer="canonical_views",
        recording_id="participant_01/B",
        source_dependencies=dependencies
        or (
            NamedSourceDependency(
                name="target_recording",
                sha256="1" * 64,
                properties={"sampling_hz": 400.0, "units": ["a.u.", "m/s2"]},
            ),
            NamedSourceDependency(
                name="static_calibration_recording",
                sha256="2" * 64,
                properties={"role": "B"},
            ),
        ),
        module_chain=modules
        or (
            OrderedModuleSpec(
                module_id="ppg_filter",
                module_version="v2",
                implementation_sha256="3" * 64,
                enabled=True,
                parameters={"highpass_hz": 0.2},
            ),
            OrderedModuleSpec(
                module_id="calibrated_ekf",
                module_version="v1",
                implementation_sha256="4" * 64,
                enabled=True,
                parameters={"alpha_r": 3.0},
            ),
        ),
        producer_sha256="5" * 64,
        output_schema={
            "channels": ["RED", "IR", "A_dyn_x", "A_dyn_y", "A_dyn_z"],
            "dtype": "float32",
        },
        extra={"offline_zero_phase": True},
    )


def _rewrite_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _concurrent_get_or_compute(
    root: str,
    identity: RecordingCacheIdentity,
    start: multiprocessing.synchronize.Event,
    counter: multiprocessing.sharedctypes.Synchronized,
    queue: multiprocessing.queues.Queue,
) -> None:
    cache = RecordingPreprocessingCache(root)
    start.wait(timeout=10.0)

    def builder() -> RecordingCacheBuild:
        with counter.get_lock():
            counter.value += 1
        time.sleep(0.1)
        return RecordingCacheBuild(
            arrays={"view": np.arange(12, dtype=np.float32).reshape(3, 4)},
            attributes={"sampling_hz": 400.0},
        )

    try:
        result = cache.get_or_compute(identity, builder)
        queue.put(("ok", result.disposition))
    except BaseException as exc:  # pragma: no cover - only reported to parent
        queue.put(("error", repr(exc)))


class RecordingCacheIdentityTests(unittest.TestCase):
    def test_ordered_module_chain_changes_key(self) -> None:
        first, second = _identity().module_chain
        forward = _identity(modules=(first, second))
        reversed_chain = _identity(modules=(second, first))
        self.assertNotEqual(forward.key, reversed_chain.key)
        self.assertEqual(
            [item["position"] for item in forward.to_payload()["module_chain"]],
            [0, 1],
        )

    def test_named_dependency_argument_order_is_canonical(self) -> None:
        first, second = _identity().source_dependencies
        self.assertEqual(
            _identity(dependencies=(first, second)).key,
            _identity(dependencies=(second, first)).key,
        )
        changed_calibration = NamedSourceDependency(
            name=second.name,
            sha256="9" * 64,
            properties=second.properties,
        )
        self.assertNotEqual(
            _identity(dependencies=(first, second)).key,
            _identity(dependencies=(first, changed_calibration)).key,
        )

    def test_duplicate_dependency_names_and_nonfinite_parameters_are_rejected(self) -> None:
        source = NamedSourceDependency(name="target", sha256="1" * 64)
        with self.assertRaisesRegex(ValueError, "unique"):
            _identity(dependencies=(source, source)).to_payload()
        invalid_module = OrderedModuleSpec(
            module_id="filter",
            module_version="v1",
            implementation_sha256="2" * 64,
            enabled=True,
            parameters={"cutoff": float("nan")},
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _identity(modules=(invalid_module,)).to_payload()


class RecordingPreprocessingCacheTests(unittest.TestCase):
    def test_runtime_symlink_escape_is_cache_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "cache"
            outside = parent / "outside"
            root.mkdir()
            outside.mkdir()
            cache = RecordingPreprocessingCache(root)
            (root / "v1").symlink_to(outside, target_is_directory=True)

            with self.assertRaises(RecordingCacheAccessError):
                cache.load(_identity())

    def test_round_trip_is_mmap_and_manifest_verifies_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = RecordingPreprocessingCache(directory)
            values = np.arange(24, dtype=np.float32).reshape(6, 4)
            starts = np.asarray([0, 800, 1600], dtype=np.int64)
            scalar = np.asarray(400.0, dtype=np.float64)
            result = cache.put_arrays(
                _identity(),
                {
                    "signal_view": values,
                    "window_starts": starts,
                    "sampling_hz": scalar,
                },
                attributes={"sampling_hz": 400.0, "channel_order": ["RED", "IR"]},
            )
            self.assertEqual(result.disposition, "written")
            self.assertIsInstance(result.entry.arrays["signal_view"], np.memmap)
            self.assertFalse(result.entry.arrays["signal_view"].flags.writeable)
            np.testing.assert_array_equal(result.entry.arrays["signal_view"], values)

            metadata = json.loads(
                (result.entry.path / "metadata.json").read_text(encoding="utf-8")
            )
            signal = metadata["arrays"]["signal_view"]
            self.assertEqual(signal["dtype"], values.dtype.str)
            self.assertEqual(signal["shape"], [6, 4])
            self.assertEqual(signal["logical_nbytes"], values.nbytes)
            self.assertEqual(len(signal["file_sha256"]), 64)
            self.assertEqual(len(signal["content_sha256"]), 64)
            self.assertTrue((result.entry.path / "COMMITTED.json").is_file())
            self.assertEqual(result.entry.arrays["sampling_hz"].shape, ())

            observed = cache.load(_identity())
            np.testing.assert_array_equal(observed.arrays["window_starts"], starts)

    def test_identical_put_is_idempotent_but_conflicting_put_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = RecordingPreprocessingCache(directory)
            identity = _identity()
            expected = np.arange(6, dtype=np.float32)
            first = cache.put_arrays(identity, {"view": expected}, attributes={"a": 1})
            metadata = first.entry.path / "metadata.json"
            before = metadata.stat().st_mtime_ns
            second = cache.put_arrays(identity, {"view": expected}, attributes={"a": 1})
            self.assertEqual(second.disposition, "existing")
            self.assertEqual(metadata.stat().st_mtime_ns, before)

            with self.assertRaises(ImmutableCacheConflictError):
                cache.put_arrays(
                    identity,
                    {"view": expected + 1.0},
                    attributes={"a": 1},
                )
            with self.assertRaises(ImmutableCacheConflictError):
                cache.put_arrays(identity, {"view": expected}, attributes={"a": 2})

    def test_object_arrays_are_rejected_before_numpy_can_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = RecordingPreprocessingCache(directory)
            with self.assertRaisesRegex(TypeError, "object dtype"):
                cache.put_arrays(
                    _identity(),
                    {"unsafe": np.asarray([{"secret": 1}], dtype=object)},
                )

    def test_missing_commit_and_unexpected_files_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = RecordingPreprocessingCache(directory)
            result = cache.put_arrays(
                _identity(),
                {"view": np.arange(4, dtype=np.float64)},
            )
            commit = result.entry.path / "COMMITTED.json"
            commit.unlink()
            with self.assertRaises(RecordingCacheCorruptionError):
                cache.load(_identity())

        with tempfile.TemporaryDirectory() as directory:
            cache = RecordingPreprocessingCache(directory)
            result = cache.put_arrays(
                _identity(),
                {"view": np.arange(4, dtype=np.float64)},
            )
            (result.entry.path / "unexpected.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(RecordingCacheCorruptionError):
                cache.load(_identity())

    def test_array_and_metadata_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = RecordingPreprocessingCache(directory)
            result = cache.put_arrays(
                _identity(),
                {"view": np.arange(8, dtype=np.float32)},
            )
            array_path = result.entry.path / "arrays" / "view.npy"
            payload = bytearray(array_path.read_bytes())
            payload[-1] ^= 1
            array_path.write_bytes(payload)
            with self.assertRaisesRegex(RecordingCacheCorruptionError, "file hash"):
                cache.load(_identity())

        with tempfile.TemporaryDirectory() as directory:
            cache = RecordingPreprocessingCache(directory)
            result = cache.put_arrays(
                _identity(),
                {"view": np.arange(8, dtype=np.float32)},
            )
            metadata_path = result.entry.path / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["unexpected"] = True
            _rewrite_json(metadata_path, metadata)
            commit_path = result.entry.path / "COMMITTED.json"
            commit = json.loads(commit_path.read_text(encoding="utf-8"))
            commit["metadata_sha256"] = hashlib.sha256(
                metadata_path.read_bytes()
            ).hexdigest()
            _rewrite_json(commit_path, commit)
            with self.assertRaisesRegex(RecordingCacheCorruptionError, "schema mismatch"):
                cache.load(_identity())

    def test_declared_shape_tampering_fails_even_with_updated_metadata_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = RecordingPreprocessingCache(directory)
            result = cache.put_arrays(
                _identity(),
                {"view": np.arange(8, dtype=np.float32)},
            )
            metadata_path = result.entry.path / "metadata.json"
            commit_path = result.entry.path / "COMMITTED.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["arrays"]["view"]["shape"] = [2, 4]
            _rewrite_json(metadata_path, metadata)
            commit = json.loads(commit_path.read_text(encoding="utf-8"))
            commit["metadata_sha256"] = hashlib.sha256(
                metadata_path.read_bytes()
            ).hexdigest()
            _rewrite_json(commit_path, commit)
            with self.assertRaisesRegex(RecordingCacheCorruptionError, "shape mismatch"):
                cache.load(_identity())

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux fcntl")
    def test_fcntl_lock_allows_only_one_concurrent_builder(self) -> None:
        context = multiprocessing.get_context("fork")
        with tempfile.TemporaryDirectory() as directory:
            start = context.Event()
            counter = context.Value("i", 0)
            queue = context.Queue()
            processes = [
                context.Process(
                    target=_concurrent_get_or_compute,
                    args=(directory, _identity(), start, counter, queue),
                )
                for _ in range(2)
            ]
            for process in processes:
                process.start()
            start.set()
            messages = [queue.get(timeout=10.0) for _ in processes]
            for process in processes:
                process.join(timeout=10.0)
                self.assertEqual(process.exitcode, 0)

            self.assertEqual(counter.value, 1)
            self.assertEqual(sorted(messages), [("ok", "hit"), ("ok", "written")])
            staging = Path(directory) / "staging"
            self.assertEqual(list(staging.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
