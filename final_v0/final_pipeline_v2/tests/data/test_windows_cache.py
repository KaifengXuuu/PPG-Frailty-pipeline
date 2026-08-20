"""统一切窗与内容寻址缓存测试 / Window and cache contract tests."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_ROOT / "src"))

from ppg_frailty.data import (  # noqa: E402
    CacheIdentity,
    ContentAddressedCache,
    WindowPlan,
    extract_window,
)
from ppg_frailty.data.cache import StaleCacheError  # noqa: E402
from ppg_frailty.data.windows import ShortRecordError  # noqa: E402


class WindowPlanTests(unittest.TestCase):
    """工程与 DL 共用同一可审计索引 / Engineering and DL share indices."""

    def test_start_aligned_tail_has_padding_mask(self) -> None:
        plan = WindowPlan(
            source_record_id="record-a",
            window_seconds=2.0,
            hop_seconds=1.0,
            end_alignment="start",
            short_record_action="reject",
            include_padded_tail=True,
            max_windows=None,
            cap_policy="not_applicable",
        )
        windows = plan.plan(n_samples=105, fs=10.0)
        self.assertEqual(windows[-1].start_sample, 90)
        self.assertEqual(windows[-1].valid_length, 15)
        self.assertEqual(sum(windows[-1].padding_mask), 5)
        source = np.arange(105, dtype=np.float64)
        extracted = extract_window(source, windows[-1], pad_value=-1.0)
        self.assertEqual(extracted.shape, (20,))
        np.testing.assert_array_equal(extracted[-5:], np.full(5, -1.0))

    def test_short_record_policy_is_explicit(self) -> None:
        rejecting = WindowPlan(
            source_record_id="short",
            window_seconds=2.0,
            hop_seconds=1.0,
            end_alignment="start",
            short_record_action="reject",
            include_padded_tail=False,
            max_windows=None,
            cap_policy="not_applicable",
        )
        with self.assertRaises(ShortRecordError):
            rejecting.plan(n_samples=8, fs=10.0)
        padding = WindowPlan(
            source_record_id="short",
            window_seconds=2.0,
            hop_seconds=1.0,
            end_alignment="start",
            short_record_action="pad_right",
            include_padded_tail=False,
            max_windows=None,
            cap_policy="not_applicable",
        )
        result = padding.plan(n_samples=8, fs=10.0)
        self.assertEqual(result[0].valid_length, 8)
        self.assertEqual(sum(result[0].padding_mask), 12)

    def test_uniform_cap_preserves_progress_endpoints(self) -> None:
        capped = WindowPlan(
            source_record_id="long",
            window_seconds=2.0,
            hop_seconds=1.0,
            end_alignment="start",
            short_record_action="reject",
            include_padded_tail=False,
            max_windows=3,
            cap_policy="uniform_progress",
        )
        starts = [
            window.start_sample
            for window in capped.plan(n_samples=100, fs=10.0)
        ]
        self.assertEqual(starts, [0, 40, 80])

    def test_right_aligned_window_is_appended_to_the_start_anchored_grid(self) -> None:
        plan = WindowPlan(
            source_record_id="legacy-grid",
            window_seconds=3.0,
            hop_seconds=3.0,
            end_alignment="include_right_aligned_if_distinct",
            short_record_action="reject",
            include_padded_tail=False,
            max_windows=None,
            cap_policy="not_applicable",
        )
        starts = [
            window.start_sample
            for window in plan.plan(n_samples=100, fs=10.0)
        ]
        self.assertEqual(starts, [0, 30, 60, 70])

        exact_endpoint = [
            window.start_sample
            for window in plan.plan(n_samples=90, fs=10.0)
        ]
        self.assertEqual(exact_endpoint, [0, 30, 60])

    def test_fractional_cap_reuses_uniform_progress_planner(self) -> None:
        fractional = WindowPlan(
            source_record_id="long",
            window_seconds=2.0,
            hop_seconds=1.0,
            end_alignment="start",
            short_record_action="reject",
            include_padded_tail=False,
            max_windows=None,
            cap_policy="uniform_progress",
            max_window_fraction=0.5,
        )
        # Nine complete candidates -> ceil(9 * .5) == 5, preserving the
        # migrated classifier's positive fractional-cap semantics and using the same
        # endpoint-preserving selector as the absolute cap.
        starts = [
            window.start_sample
            for window in fractional.plan(n_samples=100, fs=10.0)
        ]
        self.assertEqual(starts, [0, 20, 40, 60, 80])
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            WindowPlan(
                source_record_id="ambiguous",
                window_seconds=2.0,
                hop_seconds=1.0,
                end_alignment="start",
                short_record_action="reject",
                include_padded_tail=False,
                max_windows=3,
                cap_policy="uniform_progress",
                max_window_fraction=0.5,
            ).plan(n_samples=100, fs=10.0)


class ContentAddressedCacheTests(unittest.TestCase):
    """缓存身份绑定来源/config/schema/producer/fold / Audit cache identity."""

    def setUp(self) -> None:
        self.identity = CacheIdentity(
            namespace="unit_test",
            source_sha256=("0" * 64, "1" * 64),
            config_sha256="2" * 64,
            schema_sha256=("3" * 64,),
            producer_sha256="4" * 64,
            fold_file_sha256="5" * 64,
            extra={"window_plan": "explicit-v1"},
        )

    def test_raw_payload_sha_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = ContentAddressedCache(directory)
            payload_path = cache.put_bytes(self.identity, b"abc")
            metadata_path = payload_path.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["payload_sha256"],
                hashlib.sha256(b"abc").hexdigest(),
            )
            self.assertEqual(cache.get_bytes(self.identity), b"abc")
            payload_path.write_bytes(b"abd")
            with self.assertRaises(StaleCacheError):
                cache.get_bytes(self.identity)

    def test_npz_disables_pickle_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = ContentAddressedCache(directory)
            expected = np.arange(6, dtype=np.float64).reshape(3, 2)
            cache.put_npz(self.identity, {"values": expected})
            observed = cache.get_npz(self.identity)["values"]
            np.testing.assert_array_equal(observed, expected)


if __name__ == "__main__":
    unittest.main()
