"""Standard-library safe negatives for the registered V2-019 DL-only cases."""

from __future__ import annotations

import unittest

import numpy as np

from ppg_frailty.models.time_scale import (
    build_fixed_kernel_resampling_cases,
    create_fixed_kernel_resampling_model,
    prepare_fixed_kernel_dl_input,
)


class FixedKernelSafeContracts(unittest.TestCase):
    def test_exact_12_case_single_factor_registry(self) -> None:
        cases = build_fixed_kernel_resampling_cases()
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({case.case_id for case in cases}), 12)
        self.assertTrue(all(case.scientific_status == "registered_not_run" for case in cases))

    def test_interaction_and_non_target_model_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "factor interactions"):
            create_fixed_kernel_resampling_model(
                "CompactCNN1D",
                n_channels=2,
                n_classes=3,
                dl_fs_hz=160.0,
                raw_window_seconds=10.0,
                dilation=1,
            )
        with self.assertRaisesRegex(ValueError, "CompactCNN1D/InceptionTimeFull"):
            create_fixed_kernel_resampling_model(
                "ShapeFormerChannelSpecificOSD",
                n_channels=2,
                n_classes=3,
                dl_fs_hz=400.0,
            )

    def test_100hz_view_is_antialiased_and_does_not_mutate_400hz_source(self) -> None:
        rng = np.random.default_rng(42)
        source = rng.normal(size=(1, 2, 2000)).astype(np.float32)
        original = source.copy()
        mask = np.zeros((1, 2000), dtype=bool)
        mask[:, :1600] = True
        output, output_mask, provenance = prepare_fixed_kernel_dl_input(
            source,
            mask,
            "compactcnn1d__fs_100",
        )
        self.assertEqual(output.shape, (1, 2, 500))
        self.assertEqual(output_mask.shape, (1, 500))
        self.assertEqual(int(output_mask.sum()), 400)
        self.assertEqual(provenance["resample_up"], 1)
        self.assertEqual(provenance["resample_down"], 4)
        self.assertIs(provenance["engineering_features_resampled"], False)
        np.testing.assert_array_equal(source, original)


if __name__ == "__main__":
    unittest.main()
