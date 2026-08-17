"""Train-only FeatureVector transform and finite fusion tensor tests."""

from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from ppg_frailty.contracts import SignalRoute
from ppg_frailty.features import (
    build_feature_vector,
    default_registry,
    fit_fold_feature_vector_transform,
    transform_feature_vector,
    transform_feature_vector_batch,
)


_FIXTURE_PREDICTOR = "prv.ppi_mean_s"


def vector(value: float | None):
    values = {} if value is None else {_FIXTURE_PREDICTOR: value}
    validity = {} if value is None else {_FIXTURE_PREDICTOR: True}
    return build_feature_vector(
        values,
        feature_validity=validity,
        provenance={"route": SignalRoute.DIRECT.value},
    )


class VectorTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = (vector(1.0), vector(3.0))
        self.artifact = fit_fold_feature_vector_transform(
            self.train,
            ["train_a", "train_b"],
            fitted_on_participant_ids=["train_a", "train_b"],
            outer_train_participant_ids=["train_a", "train_b"],
            outer_oof_participant_ids=["heldout"],
        )

    def test_context_preserves_nan_validity_and_roster_provenance(self) -> None:
        transformed = transform_feature_vector(vector(None), self.artifact)
        self.assertTrue(np.isnan(transformed.values).all())
        self.assertFalse(transformed.validity.any())
        self.assertTrue(transformed.provenance["fold_standardized"])
        self.assertEqual(
            transformed.provenance["feature_vector_transform_sha256"],
            self.artifact.artifact_sha256,
        )
        self.assertEqual(
            transformed.provenance["feature_vector_transform_fitted_on_participant_ids"],
            ["train_a", "train_b"],
        )

    def test_matrix_context_and_fusion_tensor_share_one_artifact(self) -> None:
        heldout = vector(5.0)
        batch = transform_feature_vector_batch([heldout, vector(None)], self.artifact)
        registry = default_registry()
        index = registry.names.index(_FIXTURE_PREDICTOR)
        width = len(registry.names)
        self.assertEqual(batch.fusion_tensor.shape, (2, 2 * width))
        self.assertTrue(np.isfinite(batch.fusion_tensor).all())
        self.assertAlmostEqual(batch.contexts[0].values[index], 3.0)
        self.assertEqual(batch.fusion_tensor[0, index], 3.0)
        self.assertEqual(batch.fusion_tensor[0, width + index], 1.0)
        self.assertEqual(batch.fusion_tensor[1, index], 0.0)
        self.assertEqual(batch.fusion_tensor[1, width + index], 0.0)
        self.assertEqual(
            batch.provenance["feature_vector_transform_sha256"],
            batch.contexts[0].provenance["feature_vector_transform_sha256"],
        )

    def test_valid_count_is_part_of_artifact_identity(self) -> None:
        """valid count 被篡改必须使哈希校验失败 / Counts are hash-bound."""

        tampered = replace(
            self.artifact,
            valid_count=self.artifact.valid_count + 1,
        )
        with self.assertRaisesRegex(ValueError, "artifact identity drift"):
            tampered.validate()

    def test_heldout_participant_cannot_fit_transform(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-training participant"):
            fit_fold_feature_vector_transform(
                [vector(1.0), vector(5.0)],
                ["train_a", "heldout"],
                fitted_on_participant_ids=["train_a", "heldout"],
                outer_train_participant_ids=["train_a"],
                outer_oof_participant_ids=["heldout"],
            )


if __name__ == "__main__":
    unittest.main()
