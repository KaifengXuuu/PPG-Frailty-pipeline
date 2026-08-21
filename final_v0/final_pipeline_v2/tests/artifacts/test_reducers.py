"""Identity/NLMS/SSA/STFT/BSS contract tests / 所有 reducer 合同测试。"""

from __future__ import annotations

import unittest

import numpy as np

from ppg_frailty.artifact import (
    FastIcaBssReducer,
    IdentityReducer,
    NlmsConfig,
    NlmsReducer,
    NmfBssReducer,
    PcaBssReducer,
    SpectralMaskConfig,
    SpectralMaskReducer,
    SsaConfig,
    SsaReducer,
    get_reducer,
)
from ppg_frailty.artifact.bss import (
    FastIcaBssConfig,
    NmfBssConfig,
    PcaBssConfig,
)
from ppg_frailty.artifacts.base import (
    IMU_REFERENCE_AXES6_PROFILE_ID,
    IMU_REFERENCE_DERIVED9_AUGMENTATION_PROFILE_ID,
)


def contaminated_fixture(seconds: float = 4.0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """生成 true cardiac + shared motion mixture / Build a deterministic mixture."""

    samples = int(seconds * 400)
    time = np.arange(samples) / 400.0
    cardiac = np.sin(2 * np.pi * 1.2 * time)
    motion = np.sin(2 * np.pi * 2.3 * time) + 0.3 * np.sin(2 * np.pi * 0.7 * time)
    ppg = np.column_stack((cardiac + 0.8 * motion, 0.7 * cardiac - 0.5 * motion))
    dynamic = np.column_stack((motion, 0.5 * motion, -0.25 * motion))
    gyro = np.column_stack((0.2 * motion, -0.1 * motion, 0.05 * motion))
    imu = {
        "dynamic_acc_mps2": dynamic,
        "gyro_rads": gyro,
        "dynamic_magnitude": np.linalg.norm(dynamic, axis=1),
        "gyro_magnitude": np.linalg.norm(gyro, axis=1),
        "jerk_magnitude": np.linalg.norm(
            np.diff(dynamic, axis=0, prepend=dynamic[:1]) * 400.0,
            axis=1,
        ),
    }
    return ppg, imu


class ReducerTest(unittest.TestCase):
    """验证 deterministic result 和单通道失败 / Verify deterministic fail-closed results."""

    def test_identity_is_exact(self) -> None:
        ppg, imu = contaminated_fixture()
        result = IdentityReducer().reduce(ppg, imu)
        self.assertEqual(result.status, "success")
        self.assertTrue(result.is_identity)
        self.assertTrue(np.array_equal(result.x_ar, ppg))

    def test_nlms_delay_taps_and_missing_imu_failure(self) -> None:
        ppg, imu = contaminated_fixture()
        reducer = NlmsReducer(NlmsConfig(taps_per_delay=4, delay_taps=(0, 3, 7)))
        result = reducer.reduce(ppg, imu)
        failed = reducer.reduce(ppg, None)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.x_ar.shape, ppg.shape)
        self.assertEqual(result.diagnostics["tap_offsets_samples"], tuple(range(0, 11)))
        self.assertEqual(
            result.diagnostics["reference_names"],
            ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
        )
        self.assertEqual(
            result.diagnostics["imu_reference_profile"],
            IMU_REFERENCE_AXES6_PROFILE_ID,
        )
        self.assertEqual(failed.status, "failed")
        self.assertIsNone(failed.x_ar)

    def test_denoiser_derived_channels_require_named_augmentation(self) -> None:
        ppg, imu = contaminated_fixture()
        result = NlmsReducer(
            NlmsConfig(
                taps_per_delay=2,
                delay_taps=(0, 2),
                imu_reference_profile=(
                    IMU_REFERENCE_DERIVED9_AUGMENTATION_PROFILE_ID
                ),
            )
        ).reduce(ppg, imu)
        self.assertEqual(result.status, "success", result.reasons)
        self.assertEqual(
            result.diagnostics["reference_names"][-3:],
            ("dynamic_magnitude", "gyro_magnitude", "jerk_magnitude"),
        )
        self.assertEqual(len(result.diagnostics["reference_names"]), 9)

    def test_ssa_and_spectral_preserve_alignment(self) -> None:
        ppg, imu = contaminated_fixture()
        ssa = SsaReducer(SsaConfig(embedding_samples=64, max_components=6)).reduce(ppg, imu)
        spectral = SpectralMaskReducer(
            SpectralMaskConfig(
                stft_window_s=1.0,
                stft_hop_s=0.5,
                imu_mask_quantile=0.75,
                mask_strength=0.8,
                preserve_band_hz=(0.5, 3.0),
            )
        ).reduce(ppg, imu)
        for result in (ssa, spectral):
            self.assertEqual(result.status, "success", result.reasons)
            self.assertEqual(result.x_ar.shape, ppg.shape)
            self.assertTrue(result.alignment["same_time_grid"])
        self.assertEqual(spectral.reducer_id, "spectral_mask")
        self.assertEqual(spectral.reducer_version, "spectral_mask_v1")
        self.assertEqual(spectral.diagnostics["hop_samples_effective"], 200)

    def test_formal_spectral_yaml_maps_strictly(self) -> None:
        """逐键接受 formal YAML，未知键拒绝 / Accept exact YAML; reject unknowns."""

        parameters = {
            "stft_window_s": 4.0,
            "stft_hop_s": 1.0,
            "imu_mask_quantile": 0.75,
            "mask_strength": 0.8,
            "preserve_band_hz": [0.5, 3.0],
        }
        reducer = get_reducer("spectral_mask", parameters)
        self.assertIsInstance(reducer, SpectralMaskReducer)
        self.assertEqual(reducer.config.preserve_band_hz, (0.5, 3.0))
        with self.assertRaises(ValueError):
            get_reducer("spectral_mask", {**parameters, "unregistered_knob": 1})

    def test_all_bss_require_two_channels(self) -> None:
        ppg, imu = contaminated_fixture()
        reducers = (
            PcaBssReducer(PcaBssConfig()),
            FastIcaBssReducer(FastIcaBssConfig(max_iter=2000)),
            NmfBssReducer(NmfBssConfig(nperseg=128, max_iter=2000)),
        )
        for reducer in reducers:
            with self.subTest(reducer=reducer.reducer_id):
                result = reducer.reduce(ppg, imu)
                self.assertEqual(result.status, "success", result.reasons)
                self.assertEqual(result.x_ar.shape, ppg.shape)
                if reducer.reducer_id in {"pca_bss", "fastica_bss"}:
                    self.assertEqual(
                        result.diagnostics["motion_reference_names"],
                        ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
                    )
                single = reducer.reduce(ppg[:, :1], imu)
                self.assertEqual(single.status, "failed")
                self.assertIsNone(single.x_ar)

    def test_pca_and_fastica_share_the_registered_bss_implementation(self) -> None:
        """Facade/factory must reuse one implementation, not copy study algorithms."""

        ppg, imu = contaminated_fixture()
        for reducer_id, reducer_type, expected_version in (
            ("pca_bss", PcaBssReducer, "pca_component_select_v2"),
            ("fastica_bss", FastIcaBssReducer, "fastica_component_select_v2"),
        ):
            with self.subTest(reducer=reducer_id):
                first_reducer = get_reducer(reducer_id)
                second_reducer = get_reducer(reducer_id)
                self.assertIsInstance(first_reducer, reducer_type)
                self.assertEqual(first_reducer.reducer_version, expected_version)
                self.assertEqual(
                    first_reducer.__class__.__module__,
                    "ppg_frailty.artifacts.bss",
                )
                first = first_reducer.reduce(ppg, imu)
                second = second_reducer.reduce(ppg, imu)
                self.assertEqual(first.status, "success", first.reasons)
                self.assertEqual(second.status, "success", second.reasons)
                np.testing.assert_allclose(first.x_ar, second.x_ar, rtol=0.0, atol=0.0)

    def test_pca_and_fastica_fail_closed_and_propagate_imu_validity(self) -> None:
        ppg, imu = contaminated_fixture()
        valid = np.ones(ppg.shape[0], dtype=bool)
        valid[300:360] = False
        imu["imu_valid_mask"] = valid
        for reducer_id in ("pca_bss", "fastica_bss"):
            with self.subTest(reducer=reducer_id):
                reducer = get_reducer(reducer_id)
                result = reducer.reduce(ppg, imu)
                self.assertEqual(result.status, "success", result.reasons)
                propagated = np.asarray(
                    result.diagnostics["output_valid_mask"], dtype=bool
                )
                np.testing.assert_array_equal(propagated, valid)
                self.assertAlmostEqual(
                    result.diagnostics["output_valid_fraction"],
                    float(np.mean(valid)),
                )

                missing_imu = reducer.reduce(ppg, None)
                self.assertEqual(missing_imu.status, "failed")
                self.assertIsNone(missing_imu.x_ar)
                self.assertIn("IMU references are required", missing_imu.reasons[0])

                rank_deficient = reducer.reduce(
                    np.column_stack((ppg[:, 0], ppg[:, 0])), imu
                )
                self.assertEqual(rank_deficient.status, "failed")
                self.assertIsNone(rank_deficient.x_ar)
                self.assertIn("rank-deficient", rank_deficient.reasons[0])

    def test_bss_router_rejects_parameters_unused_by_selected_algorithm(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown reducer parameters"):
            get_reducer("pca_bss", {"max_iter": 10})
        with self.assertRaisesRegex(ValueError, "unknown reducer parameters"):
            get_reducer("fastica_bss", {"nmf_rank": 3})
        with self.assertRaisesRegex(ValueError, "unknown reducer parameters"):
            get_reducer("nmf_bss", {"imu_reference_profile": "axes6_si_v1"})

    def test_learned_route_is_registered_unsupported(self) -> None:
        ppg, imu = contaminated_fixture()
        result = get_reducer("learned_denoiser").reduce(ppg, imu)
        self.assertEqual(result.status, "unsupported")
        self.assertIsNone(result.x_ar)

    def test_ssa_fails_when_no_component_meets_registered_minimum(self) -> None:
        """SSA 阈值不达标必须失败 / SSA must not argmax-fallback below threshold."""

        ppg, imu = contaminated_fixture()
        reducer = SsaReducer(
            SsaConfig(
                embedding_samples=64,
                max_components=6,
                minimum_cardiac_concentration=1.0,
            )
        )
        result = reducer.reduce(ppg, imu)
        self.assertEqual(result.status, "failed")
        self.assertIsNone(result.x_ar)
        self.assertIn("minimum_cardiac_concentration", result.reasons[0])

    def test_imu_invalid_rows_propagate_to_nlms_and_spectral_masks(self) -> None:
        """invalid IMU 不得伪装有效 artifact / Invalid IMU must remain masked."""

        ppg, imu = contaminated_fixture(seconds=6.0)
        valid = np.ones(ppg.shape[0], dtype=bool)
        valid[300:360] = False
        imu["imu_valid_mask"] = valid
        nlms = NlmsReducer(
            NlmsConfig(taps_per_delay=2, delay_taps=(0, 3))
        ).reduce(ppg, imu)
        spectral = SpectralMaskReducer(
            SpectralMaskConfig(
                stft_window_s=1.0, stft_hop_s=0.5,
                imu_mask_quantile=0.75, mask_strength=0.8,
                preserve_band_hz=(0.5, 3.0),
            )
        ).reduce(ppg, imu)
        for result in (nlms, spectral):
            self.assertEqual(result.status, "success", result.reasons)
            propagated = np.asarray(result.diagnostics["output_valid_mask"], dtype=bool)
            self.assertEqual(propagated.shape, (ppg.shape[0],))
            self.assertFalse(propagated[300:360].any())
            self.assertLess(result.diagnostics["output_valid_fraction"], 1.0)
        self.assertIn("suppression_fraction_by_channel", spectral.diagnostics)
        self.assertAlmostEqual(
            spectral.confidence,
            float(np.mean(spectral.diagnostics["retained_signal_agreement_by_channel"])),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
