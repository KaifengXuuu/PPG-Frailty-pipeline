"""Safe model-input adapter and deployment aggregation tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ppg_frailty.bundle import (
    ParticipantFileInput,
    build_model_input_adapter,
    infer_participant,
)
from ppg_frailty.models import ModelInputSpec
from ppg_frailty.training import LoadedBundle, input_spec_sha256, predict_bundle_raw


class _ProbabilityEstimator:
    model_id = "logistic_regression"

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        score = np.clip(values[:, 0], 0.0, 1.0)
        probability = np.column_stack((score, 1.0 - score, np.ones_like(score)))
        return probability / probability.sum(axis=1, keepdims=True)


class BundleParticipantInferenceTests(unittest.TestCase):
    def _bundle(
        self,
        line: str,
        roles: tuple[str, ...] = ("B", "R"),
    ) -> LoadedBundle:
        directory = Path(tempfile.gettempdir())
        spec = ModelInputSpec(
            "feature_vector",
            n_classes=3,
            feature_names=("feature_a", "feature_b"),
        )
        input_hash = input_spec_sha256(spec)
        adapter = build_model_input_adapter(
            "feature_vector",
            input_schema_hash=input_hash,
            allowed_role_families=roles,
        )
        return LoadedBundle(
            model=_ProbabilityEstimator(),
            transforms=None,
            manifest={
                "input_spec": {
                    "representation_mode": "feature_vector",
                    "n_channels": 0,
                    "n_classes": 3,
                    "n_file_features": 0,
                    "feature_names": ("feature_a", "feature_b"),
                    "channel_schema": (),
                },
                "input_spec_hash": input_hash,
                "pipeline_adapter_contract": {
                    "status": "bundled",
                    "representation_mode": "feature_vector",
                    "input_schema_hash": input_hash,
                    "allowed_role_families": list(roles),
                    "boundary": adapter.boundary,
                },
                "metadata": {
                    "aggregation_rule": line,
                    "representation_mode": "feature_vector",
                },
            },
            directory=directory,
            pipeline_adapter=adapter,
        )

    def test_line_a_equal_files_and_line_b_equal_available_role_families(self) -> None:
        files = (
            ParticipantFileInput("B1", "B", {"x": np.asarray([0.9, 0.0])}),
            ParticipantFileInput("R1", "R1", {"x": np.asarray([0.1, 0.0])}),
            ParticipantFileInput("R2", "R2", {"x": np.asarray([0.3, 0.0])}),
        )
        line_a = infer_participant(self._bundle("line_a_equal_files"), files)
        line_b = infer_participant(self._bundle("line_b_equal_role_families"), files)
        expected_files = np.asarray(
            [
                _ProbabilityEstimator().predict_proba(item.record["x"][None, :])[0]
                for item in files
            ]
        )
        np.testing.assert_allclose(
            line_a["participant_probability"],
            expected_files.mean(axis=0),
        )
        expected_line_b = np.vstack(
            (expected_files[0], expected_files[1:].mean(axis=0))
        ).mean(axis=0)
        np.testing.assert_allclose(
            line_b["participant_probability"],
            expected_line_b,
        )
        self.assertEqual(set(line_b["role_family_probabilities"]), {"B", "R"})

    def test_unresolved_raw_device_record_fails_closed(self) -> None:
        adapter = build_model_input_adapter(
            "raw",
            input_schema_hash="b" * 64,
        )
        with self.assertRaisesRegex(TypeError, "preprocessing is unresolved"):
            adapter.transform_record(object())

    def test_s_and_w_roles_are_rejected_outside_current_training_scope(self) -> None:
        for role in ("S", "W2"):
            with self.subTest(role=role), self.assertRaisesRegex(
                ValueError, "outside this bundle training scope B,R"
            ):
                infer_participant(
                    self._bundle("line_a_equal_files"),
                    [ParticipantFileInput("outside", role, {"x": np.asarray([0.5, 0.0])})],
                )

    def test_auxiliary_roles_are_allowed_when_bundle_was_trained_for_them(self) -> None:
        loaded = self._bundle(
            "line_b_equal_role_families",
            roles=("B", "S", "W"),
        )
        result = infer_participant(
            loaded,
            [
                ParticipantFileInput("B1", "B", {"x": np.asarray([0.9, 0.0])}),
                ParticipantFileInput("S1", "S1", {"x": np.asarray([0.2, 0.0])}),
                ParticipantFileInput("W1", "W2", {"x": np.asarray([0.4, 0.0])}),
            ],
        )
        self.assertEqual(
            set(result["role_family_probabilities"]),
            {"B", "S", "W"},
        )

    def test_swapped_schema_adapter_is_rejected_before_transform(self) -> None:
        loaded = self._bundle("line_a_equal_files")
        swapped = LoadedBundle(
            model=loaded.model,
            transforms=loaded.transforms,
            manifest=loaded.manifest,
            directory=loaded.directory,
            pipeline_adapter=build_model_input_adapter(
                "feature_vector", input_schema_hash="b" * 64
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "not bound to its input schema"):
            predict_bundle_raw(swapped, {"x": np.asarray([0.5, 0.0])})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
