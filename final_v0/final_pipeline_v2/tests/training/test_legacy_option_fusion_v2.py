"""Focused tests for fused legacy algorithm options in ordinary V2."""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy import signal
import yaml

from ppg_frailty.config import canonical_json_bytes, validate_config_payload
from ppg_frailty.experiment import (
    _RuntimeRecord,
    _apply_window_quality_selection,
    _make_oof,
    _materialize_representation_dataset,
)
from ppg_frailty.models.pisd_port import (
    _balanced_discovery_indices,
    _class_window_balanced_discovery_indices,
)
from ppg_frailty.quality.window_selection import (
    WindowSelectionConfig,
    legacy_per_file_top_fraction_mask,
    legacy_window_sqi_scores,
    mark_raw_windows_for_aggregation,
    select_raw_windows,
)
from ppg_frailty.representations.raw import RawWindows
from ppg_frailty.training import (
    RawWindowDataset,
    SampleIdentity,
    TrainingConfig,
    UnifiedTrainer,
    configured_class_weight_vector,
    outer_train_class_counts,
)
from ppg_frailty.contracts import SignalRoute


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "configs/reference_static_role_aware_v2.yaml"
FEATURE_REFERENCE = ROOT / "configs/reference_static_feature_vector_v2.yaml"


def _dataset() -> RawWindowDataset:
    rows = (
        ("p0", 0),
        ("p0", 0),
        ("p0", 0),
        ("p0", 0),
        ("p1", 1),
        ("p2", 1),
        ("p3", 2),
    )
    identities = tuple(
        SampleIdentity(
            participant_id=participant,
            file_id=f"{participant}_B",
            role="B",
            label=label,
            signal_route="direct",
            window_id=f"w{index}",
        )
        for index, (participant, label) in enumerate(rows)
    )
    return RawWindowDataset(
        np.ones((len(rows), 2, 32), dtype=np.float32),
        identities,
    )


class ClassCountBasisTests(unittest.TestCase):
    def test_inverse_and_effective_number_consume_selected_basis(self) -> None:
        dataset = _dataset()
        np.testing.assert_array_equal(
            outer_train_class_counts(
                dataset,
                3,
                class_count_basis="participant",
            ),
            (1.0, 2.0, 1.0),
        )
        np.testing.assert_array_equal(
            outer_train_class_counts(dataset, 3, class_count_basis="row"),
            (4.0, 2.0, 1.0),
        )
        participant = configured_class_weight_vector(
            dataset,
            class_weighting="effective_number",
            class_count_basis="participant",
            class_weight_beta=0.9,
            n_classes=3,
        )
        row = configured_class_weight_vector(
            dataset,
            class_weighting="effective_number",
            class_count_basis="row",
            class_weight_beta=0.9,
            n_classes=3,
        )
        self.assertFalse(np.allclose(participant, row))

        balanced = UnifiedTrainer(
            TrainingConfig(
                loss="balanced_softmax",
                class_weighting="none",
                class_count_basis="row",
            )
        )._criterion(dataset)
        np.testing.assert_array_equal(
            balanced.class_counts.detach().cpu().numpy(),
            (4.0, 2.0, 1.0),
        )

    def test_legacy_weight_names_materialize_to_one_strategy_plus_basis(self) -> None:
        participant = TrainingConfig(
            class_weighting="outer_train_inverse_frequency"
        )
        self.assertEqual(participant.class_weighting, "inverse_frequency")
        self.assertEqual(participant.class_count_basis, "participant")
        row = TrainingConfig(
            class_weighting="outer_train_window_inverse_frequency"
        )
        self.assertEqual(row.class_weighting, "inverse_frequency")
        self.assertEqual(row.class_count_basis, "row")

        payload = yaml.safe_load(REFERENCE.read_text(encoding="utf-8"))
        payload["training"]["class_weighting"] = "effective_number"
        payload["training"]["class_weight_beta"] = 0.9
        payload["training"]["class_count_basis"] = "row"
        resolved = validate_config_payload(payload)
        self.assertEqual(resolved["training"]["class_count_basis"], "row")


class WindowSelectionTests(unittest.TestCase):
    @staticmethod
    def _legacy_reference_scores(values: np.ndarray) -> np.ndarray:
        standardized = []
        for raw_window in np.asarray(values, dtype=np.float32):
            segment = raw_window.T
            median = np.median(segment, axis=0, keepdims=True)
            q75 = np.percentile(segment, 75, axis=0, keepdims=True)
            q25 = np.percentile(segment, 25, axis=0, keepdims=True)
            robust = (q75 - q25) / 1.349
            standard = np.std(segment, axis=0, keepdims=True)
            scale = np.where(robust > 1e-6, robust, standard)
            standardized.append(
                np.clip((segment - median) / (scale + 1e-6), -8.0, 8.0).T
            )
        scores = []
        for window in standardized:
            red, ir = window[0].astype(float), window[1].astype(float)
            ppg = ir if np.std(ir) >= np.std(red) else red
            acc = np.linalg.norm(window[2:5].astype(float), axis=0)
            gyro = np.linalg.norm(window[5:8].astype(float), axis=0)
            ppg_std = float(np.std(ppg))
            frequencies, psd = signal.welch(
                ppg, fs=400.0, nperseg=min(512, ppg.size)
            )
            total = float(np.trapezoid(psd, frequencies)) + 1e-12
            band = (frequencies >= 0.5) & (frequencies <= 3.0)
            spectral_ratio = float(
                np.trapezoid(psd[band], frequencies[band]) / total
            )
            peaks, _ = signal.find_peaks(
                (ppg - np.median(ppg)) / (ppg_std + 1e-8),
                distance=112,
                prominence=0.3,
            )
            if peaks.size >= 3:
                intervals = np.diff(peaks).astype(float)
                stability = 1.0 / (
                    1.0 + float(np.std(intervals) / (np.mean(intervals) + 1e-8))
                )
                density = min(1.0, float(peaks.size) / max(2.0, ppg.size / 400.0 * 3.0))
            else:
                stability = density = 0.0
            motion = float(
                np.sqrt(np.mean(np.square(acc)))
                + 0.25 * np.sqrt(np.mean(np.square(gyro)))
            )
            scores.append(
                0.40 * spectral_ratio
                + 0.35 * stability
                + 0.15 * density
                + 0.10 / (1.0 + max(0.0, motion - 1.0))
            )
        result = np.asarray(scores, dtype=np.float32)
        if result.size and np.max(result) > np.min(result):
            lower = float(np.percentile(result, 5))
            upper = float(np.percentile(result, 95))
            result = np.clip(
                (result - lower) / (upper - lower + 1e-8), 0.0, 1.0
            ).astype(np.float32)
        return result

    def test_legacy_selector_keeps_ceil_fraction_and_one_minimum(self) -> None:
        scores = np.asarray((0.1, 0.8, 0.2, 0.7, 0.5), dtype=np.float64)
        keep = legacy_per_file_top_fraction_mask(scores, keep_fraction=0.50)
        self.assertEqual(int(keep.sum()), math.ceil(scores.size * 0.50))
        self.assertEqual(set(np.flatnonzero(keep)), {1, 3, 4})
        self.assertEqual(
            int(
                legacy_per_file_top_fraction_mask(
                    np.asarray((0.2,)), keep_fraction=0.01
                ).sum()
            ),
            1,
        )

    def test_real_scores_and_raw_window_contract_are_filtered_together(self) -> None:
        time = np.arange(400, dtype=np.float64) / 400.0
        clean = np.sin(2.0 * np.pi * 1.2 * time)
        values = np.zeros((4, 8, 400), dtype=np.float32)
        for index in range(4):
            values[index, 0] = clean + 0.01 * index
            values[index, 1] = clean
            values[index, 2:5] = float(index)
        masks = np.ones((4, 400), dtype=bool)
        scores = legacy_window_sqi_scores(values, masks)
        self.assertEqual(scores.shape, (4,))
        self.assertTrue(np.isfinite(scores).all())
        raw = RawWindows(
            values=values,
            valid_mask=masks,
            start_samples=np.arange(4, dtype=np.int64) * 200,
            candidate_count=4,
            dropped_invalid_count=0,
        )
        selected, provenance = select_raw_windows(
            raw,
            WindowSelectionConfig(
                policy="legacy_per_file_top_fraction",
                keep_fraction=0.5,
            ),
        )
        self.assertEqual(selected.values.shape[0], 2)
        self.assertEqual(selected.valid_mask.shape[0], 2)
        self.assertEqual(selected.start_samples.shape[0], 2)
        retained_indices = np.searchsorted(
            raw.start_samples,
            selected.start_samples,
        )
        np.testing.assert_allclose(
            selected.window_quality_scores,
            scores[retained_indices],
        )
        self.assertFalse(provenance["uses_labels"])
        self.assertFalse(provenance["cross_file_statistics"])
        np.testing.assert_allclose(
            scores,
            self._legacy_reference_scores(values),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_application_scope_matches_historical_train_only_or_all_partitions(self) -> None:
        values = np.arange(4 * 8 * 64, dtype=np.float32).reshape(4, 8, 64)
        raw = RawWindows(
            values=values,
            valid_mask=np.ones((4, 64), dtype=bool),
            start_samples=np.arange(4, dtype=np.int64) * 32,
            candidate_count=4,
            dropped_invalid_count=0,
        )

        class Config:
            def __init__(self, scope: str) -> None:
                self.scope = scope

            def section(self, name: str) -> dict[str, object]:
                if name == "quality":
                    return {
                        "window_selection": {
                            "policy": "legacy_per_file_top_fraction",
                            "keep_fraction": 0.5,
                            "application_scope": self.scope,
                        }
                    }
                if name == "aggregation":
                    return {
                        "quality_weighting": True,
                        "quality_weight_source": "legacy_window_sqi",
                    }
                raise KeyError(name)

        def states() -> list[_RuntimeRecord]:
            return [
                _RuntimeRecord(
                    row=SimpleNamespace(participant_id="train", record_id="train_f"),
                    raw_windows=copy.deepcopy(raw),
                ),
                _RuntimeRecord(
                    row=SimpleNamespace(participant_id="oof", record_id="oof_f"),
                    raw_windows=copy.deepcopy(raw),
                ),
            ]

        historical = states()
        evidence = _apply_window_quality_selection(
            historical,
            Config("outer_train_only"),
            train_ids=("train",),
            oof_ids=("oof",),
        )
        self.assertEqual(historical[0].raw_windows.values.shape[0], 2)
        self.assertEqual(historical[1].raw_windows.values.shape[0], 4)
        self.assertEqual(historical[0].raw_windows.window_quality_scores.shape, (2,))
        self.assertEqual(historical[1].raw_windows.window_quality_scores.shape, (4,))
        self.assertEqual(
            evidence["partition_counts"]["outer_oof"]["selection_applied_files"],
            0,
        )
        self.assertEqual(
            evidence["partition_counts"]["outer_oof"]["scored_files"],
            1,
        )
        self.assertIsNotNone(evidence["score_vector_bundle_sha256"])

        parallel = states()
        _apply_window_quality_selection(
            parallel,
            Config("all_partitions"),
            train_ids=("train",),
            oof_ids=("oof",),
        )
        self.assertEqual(parallel[0].raw_windows.values.shape[0], 2)
        self.assertEqual(parallel[1].raw_windows.values.shape[0], 2)

        historical_aggregation = states()
        aggregation_evidence = _apply_window_quality_selection(
            historical_aggregation,
            Config("legacy_train_and_aggregation"),
            train_ids=("train",),
            oof_ids=("oof",),
        )
        self.assertEqual(historical_aggregation[0].raw_windows.values.shape[0], 2)
        self.assertEqual(historical_aggregation[1].raw_windows.values.shape[0], 4)
        self.assertEqual(
            int(
                np.count_nonzero(
                    historical_aggregation[1].raw_windows.window_aggregation_mask
                )
            ),
            2,
        )
        self.assertEqual(
            aggregation_evidence["partition_counts"]["outer_oof"][
                "aggregation_selection_applied_files"
            ],
            1,
        )

    def test_legacy_heldout_selection_and_weighting_are_orthogonal(self) -> None:
        time = np.arange(128, dtype=np.float64) / 400.0
        values = np.zeros((4, 8, 128), dtype=np.float32)
        for index in range(4):
            values[index, 0] = np.sin(2.0 * np.pi * (1.0 + index) * time)
            values[index, 1] = np.cos(2.0 * np.pi * (1.0 + index) * time)
            values[index, 2:5] = float(index) * 0.25
        raw = RawWindows(
            values=values,
            valid_mask=np.ones((4, 128), dtype=bool),
            start_samples=np.arange(4, dtype=np.int64) * 64,
            candidate_count=4,
            dropped_invalid_count=0,
        )
        marked, _ = mark_raw_windows_for_aggregation(
            raw,
            WindowSelectionConfig(
                policy="legacy_per_file_top_fraction",
                keep_fraction=0.5,
                application_scope="legacy_train_and_aggregation",
            ),
        )
        state = _RuntimeRecord(
            row=SimpleNamespace(
                participant_id="P1",
                record_id="F1",
                role="B",
                class_id=0,
            ),
            route=SignalRoute.DIRECT,
            retained=True,
            route_status="retained_direct",
            raw_windows=marked,
        )
        dataset = _materialize_representation_dataset(
            [state],
            ("P1",),
            "raw",
            quality_weight_source="legacy_window_sqi",
        )
        probabilities = np.asarray(
            (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.5, 0.5, 0.0),
            ),
            dtype=np.float64,
        )
        common = {
            "repeat": 0,
            "fold": 0,
            "split_seed": 42,
            "training_seed": 42,
            "config_hash": "config",
            "manifest_hash": "manifest",
            "fold_hash": "fold",
            "preprocessing_hash": "preprocess",
            "feature_hash": "feature",
            "model_hash": "model",
            "representation_mode": "raw",
            "class_order": (0, 1, 2),
        }
        window_rows, weighted_files, _, weighted_subjects = _make_oof(
            [state],
            ("P1",),
            dataset.identities,
            probabilities,
            common,
            balance_line="line_a_equal_files",
            quality_weighting=True,
            quality_weight_source="legacy_window_sqi",
        )
        keep = np.asarray(marked.window_aggregation_mask, dtype=bool)
        scores = np.asarray(marked.window_quality_scores, dtype=np.float64)
        expected_weighted = np.average(
            probabilities[keep],
            axis=0,
            weights=scores[keep],
        )
        expected_weighted /= expected_weighted.sum()
        self.assertEqual(len(window_rows), 4)
        self.assertTrue(all(row.retained for row in window_rows))
        np.testing.assert_allclose(
            weighted_files[0].probabilities,
            expected_weighted,
        )
        np.testing.assert_allclose(
            weighted_subjects[0].probabilities,
            expected_weighted,
        )

        _, ordinary_files, _, _ = _make_oof(
            [state],
            ("P1",),
            dataset.identities,
            probabilities,
            common,
            balance_line="line_a_equal_files",
            quality_weighting=False,
            quality_weight_source="none",
        )
        expected_ordinary = probabilities[keep].mean(axis=0)
        expected_ordinary /= expected_ordinary.sum()
        np.testing.assert_allclose(
            ordinary_files[0].probabilities,
            expected_ordinary,
        )

    def test_legacy_window_score_reaches_identity_oof_and_weighted_file_mean(self) -> None:
        raw = RawWindows(
            values=np.ones((2, 8, 64), dtype=np.float32),
            valid_mask=np.ones((2, 64), dtype=bool),
            start_samples=np.asarray((0, 32), dtype=np.int64),
            candidate_count=2,
            dropped_invalid_count=0,
            window_quality_scores=np.asarray((0.9, 0.1), dtype=np.float32),
        )
        state = _RuntimeRecord(
            row=SimpleNamespace(
                participant_id="P1",
                record_id="F1",
                role="B",
                class_id=0,
            ),
            route=SignalRoute.DIRECT,
            retained=True,
            route_status="retained_direct",
            raw_windows=raw,
        )
        dataset = _materialize_representation_dataset(
            [state],
            ("P1",),
            "raw",
            quality_weight_source="legacy_window_sqi",
        )
        np.testing.assert_allclose(
            [identity.quality_score for identity in dataset.identities],
            (0.9, 0.1),
            atol=1e-7,
        )
        common = {
            "repeat": 0,
            "fold": 0,
            "split_seed": 42,
            "training_seed": 42,
            "config_hash": "config",
            "manifest_hash": "manifest",
            "fold_hash": "fold",
            "preprocessing_hash": "preprocess",
            "feature_hash": "feature",
            "model_hash": "model",
            "representation_mode": "raw",
            "class_order": (0, 1, 2),
        }
        window_rows, file_rows, _, subject_rows = _make_oof(
            [state],
            ("P1",),
            dataset.identities,
            np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
            common,
            balance_line="line_a_equal_files",
            quality_weighting=True,
            quality_weight_source="legacy_window_sqi",
        )
        np.testing.assert_allclose(
            [row.quality_score for row in window_rows],
            (0.9, 0.1),
            atol=1e-7,
        )
        np.testing.assert_allclose(file_rows[0].probabilities, (0.9, 0.1, 0.0))
        np.testing.assert_allclose(subject_rows[0].probabilities, (0.9, 0.1, 0.0))

    def test_active_policy_is_raw_fusion_only_and_hash_materialized(self) -> None:
        raw_payload = yaml.safe_load(REFERENCE.read_text(encoding="utf-8"))
        raw_payload["quality"]["window_selection"] = {
            "policy": "legacy_per_file_top_fraction",
            "keep_fraction": 0.7,
        }
        resolved = validate_config_payload(raw_payload)
        self.assertEqual(
            resolved["quality"]["window_selection"]["score_algorithm"],
            "legacy_cardiac_motion_window_sqi_v1",
        )
        self.assertEqual(
            resolved["quality"]["window_selection"]["application_scope"],
            "outer_train_only",
        )
        all_partitions = copy.deepcopy(raw_payload)
        all_partitions["quality"]["window_selection"]["application_scope"] = (
            "all_partitions"
        )
        resolved_all = validate_config_payload(all_partitions)
        self.assertNotEqual(
            hashlib.sha256(canonical_json_bytes(resolved)).hexdigest(),
            hashlib.sha256(canonical_json_bytes(resolved_all)).hexdigest(),
        )
        aggregation_only = copy.deepcopy(raw_payload)
        aggregation_only["quality"]["window_selection"]["application_scope"] = (
            "legacy_train_and_aggregation"
        )
        resolved_aggregation_only = validate_config_payload(aggregation_only)
        self.assertEqual(
            resolved_aggregation_only["quality"]["window_selection"][
                "application_scope"
            ],
            "legacy_train_and_aggregation",
        )
        self.assertNotEqual(
            hashlib.sha256(canonical_json_bytes(resolved)).hexdigest(),
            hashlib.sha256(
                canonical_json_bytes(resolved_aggregation_only)
            ).hexdigest(),
        )
        feature_payload = yaml.safe_load(
            FEATURE_REFERENCE.read_text(encoding="utf-8")
        )
        feature_payload["quality"]["window_selection"] = copy.deepcopy(
            raw_payload["quality"]["window_selection"]
        )
        with self.assertRaisesRegex(ValueError, "raw or fusion"):
            validate_config_payload(feature_payload)


class DiscoveryBalanceTests(unittest.TestCase):
    def test_legacy_class_window_and_participant_file_strategies_differ(self) -> None:
        labels = np.asarray([0] * 8 + [1] * 8, dtype=np.int64)
        participants = tuple(
            ["p0_many"] * 7 + ["p0_one"] + ["p1_many"] * 7 + ["p1_one"]
        )
        files = tuple(f"f{index % 3}" for index in range(labels.size))
        legacy = _class_window_balanced_discovery_indices(
            labels,
            maximum=8,
            seed=42,
        )
        hierarchical = _balanced_discovery_indices(
            labels,
            participants,
            files,
            maximum=8,
            seed=42,
        )
        self.assertEqual(legacy.size, 8)
        self.assertEqual(hierarchical.size, 8)
        self.assertEqual(np.bincount(labels[legacy]).tolist(), [4, 4])
        self.assertFalse(np.array_equal(legacy, hierarchical))


if __name__ == "__main__":
    unittest.main()
