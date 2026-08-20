"""Engineering, ten-field registry, vector, and matrix tests / 特征合同测试。"""

from __future__ import annotations

from dataclasses import fields, replace
import unittest

import numpy as np

from ppg_frailty.contracts import (
    ArtifactReductionResult,
    EngineeringFeatureSequence,
    OrderedFeatureMatrixV1,
    SignalRoute,
)
from ppg_frailty.data.windows import WindowPlan
from ppg_frailty.features import (
    ENGINEERING_SCHEMA_VERSION,
    EngineeringExtraction,
    build_feature_vector,
    build_ordered_matrix,
    default_registry,
    engineering_feature_names,
    engineering_welch_parameters,
    extract_engineering_features,
    fit_fold_feature_transform,
    fit_fold_feature_vector_transform,
    summarize_engineering,
    transform_engineering,
    transform_feature_vector,
)
from ppg_frailty.signal import build_signal_views
from ppg_frailty.representations import validate_feature_matrix


def resolved_config() -> dict[str, object]:
    """最小显式 resolved signal profile / Minimal explicit resolved profile."""

    return {
        "signal": {
            "internal_fs_hz": 400.0,
            "peak_detector": {
                "detector_id": "aboy_project_v1",
                "failure_action": "fail_closed_no_fallback",
            },
            "channel_order": ["RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ"],
            "ppg_native_unit": "raw_counts",
            "accelerometer_input_unit": "m/s2",
            "gyroscope_input_unit": "rad/s",
            "ppg_filter": {
                "family": "butterworth_sos",
                "order": 3,
                "low_hz": 0.2,
                "high_hz": 8.0,
                "phase": "zero_phase",
                "short_signal_policy": "reject",
                "notch_enabled": False,
            },
            "analysis_view": {
                "direct_source": "x_filter_0p2_to_8hz", "non_identity_source": "aligned_x_ar",
                "non_identity_semantics": "rate_only", "additional_filter": "none",
            },
            "gap_repair": {
                "method": "linear_inside_only", "max_gap_samples": 100,
                "edge_extrapolation": False, "all_missing_channel_action": "reject_record",
            },
            "imu": {
                "gravity_method": "quaternion_error_state_ekf",
                "initialization": "online_no_precalibration",
                "comparison_method": "lowpass_0p3hz",
                "sensor_lowpass_acc_hz": 20.0,
                "sensor_lowpass_gyro_hz": 40.0,
                "gravity_lowpass_hz": 0.3,
                "output_units": {"acceleration": "m/s^2", "gyroscope": "rad/s", "jerk": "m/s^3"},
                "required_axes": 6,
                "failure_action": "fail_closed",
            },
            "dl_resampling": {
                "enabled": False, "target_fs_hz": 400.0, "method": "polyphase_anti_alias",
                "preserve_feature_grid_hz": 400.0,
            },
            "normalization": {
                "raw_ppg": "per_window_median_iqr_over_1p349_sd_finite",
                "raw_imu": (
                    "outer_training_participant_only_median_iqr_over_1p349_"
                    "population_sd_then_one_axes6"
                ),
                "iqr_fallback": "standard_deviation_then_finite_one",
                "clip_after_scale": [-8.0, 8.0],
            },
        },
        "quality": {"long_gap_max_samples": 100, "flatline_duration_s": 1.0},
    }


def views_fixture(seconds: float = 20.0):
    """构建三窗同步记录 / Build a synchronized three-window record."""

    samples = int(seconds * 400)
    time = np.arange(samples) / 400.0
    pulse = np.sin(2 * np.pi * 1.2 * time)
    motion = 0.1 * np.sin(2 * np.pi * 0.8 * time)
    return build_signal_views(
        {
            "record_id": "feature_fixture",
            "fs_hz": 400.0,
            "ppg": np.column_stack(
                (1000 + 20 * pulse, 1200 + 15 * np.sin(2 * np.pi * 1.2 * time + 0.1))
            ),
            "acc": np.column_stack((motion, np.zeros(samples), np.full(samples, 9.80665))),
            "gyro": np.zeros((samples, 3)),
            "acc_unit": "m/s2",
            "gyro_unit": "rad/s",
        },
        resolved_config(),
    )


def engineering_plan() -> WindowPlan:
    """唯一显式 10 s/5 s WindowPlan / Sole explicit engineering plan."""

    return WindowPlan(
        source_record_id="feature_fixture",
        window_seconds=10.0,
        hop_seconds=5.0,
        end_alignment="start",
        short_record_action="reject",
        include_padded_tail=False,
        max_windows=None,
        cap_policy="not_applicable",
    )


def direct_extraction() -> EngineeringExtraction:
    """构建 direct 工程序列 / Build the direct engineering sequence."""

    return extract_engineering_features(views_fixture(), plan=engineering_plan())


class EngineeringRegistryTest(unittest.TestCase):
    """验证工程序列、注册表和 validity / Verify engineering and validity contracts."""

    def test_direct_engineering_schema_and_missingness_are_truthful(self) -> None:
        extraction = direct_extraction()
        names = engineering_feature_names()
        self.assertEqual(extraction.sequence.values.shape, (3, len(names)))
        # 中文：合成 fixture 的若干 IMU 轴恒定；skew/entropy 不可定义，不能强标 valid。
        # English: Constant synthetic IMU axes make skew/entropy undefined.
        self.assertTrue(
            np.array_equal(np.isfinite(extraction.sequence.values), extraction.value_validity)
        )
        for name in (
            "ppg_red.mean",
            "ppg_red.population_sd",
            "ppg_red.total_power",
            "ppg_red.dominant_frequency_hz",
            "ppg_red.spectral_centroid_hz",
            "ppg_red.bandpower_0.5_3_hz",
            "acc_magnitude.mean",
            "acc_magnitude.total_power",
            "angular_rate_magnitude.mean",
            "jerk_magnitude.mean",
            "acc_dynamic_x.population_sd",
        ):
            self.assertTrue(extraction.value_validity[:, names.index(name)].all())
        self.assertEqual(extraction.sequence.schema_version, ENGINEERING_SCHEMA_VERSION)
        self.assertFalse(
            extraction.value_validity[:, names.index("gyro_y.skew_bias_corrected")].any()
        )
        self.assertEqual(extraction.sequence.start_samples.tolist(), [0, 2000, 4000])

    def test_extractor_accepts_configurable_complete_window_plan(self) -> None:
        overlapping_plan = replace(engineering_plan(), hop_seconds=2.5)
        overlapping = extract_engineering_features(
            views_fixture(), plan=overlapping_plan
        )
        self.assertEqual(
            overlapping.sequence.start_samples.tolist(),
            [0, 1000, 2000, 3000, 4000],
        )
        capped_plan = replace(engineering_plan(), max_windows=2, cap_policy="uniform_progress")
        capped = extract_engineering_features(views_fixture(), plan=capped_plan)
        self.assertEqual(capped.sequence.start_samples.tolist(), [0, 4000])

        padded_plan = replace(
            engineering_plan(),
            short_record_action="pad_right",
            min_valid_fraction=0.5,
        )
        with self.assertRaisesRegex(ValueError, "complete unpadded"):
            extract_engineering_features(views_fixture(), plan=padded_plan)

    def test_exact_115_column_order_and_axis_time_only_contract(self) -> None:
        names = engineering_feature_names()
        self.assertEqual(len(names), 115)
        self.assertEqual(len(set(names)), 115)
        self.assertEqual(names[0], "ppg_red.mean")
        self.assertEqual(names[14], "ppg_ir.mean")
        self.assertEqual(names[28], "acc_magnitude.mean")
        self.assertEqual(names[43], "angular_rate_magnitude.mean")
        self.assertEqual(names[58], "jerk_magnitude.mean")
        self.assertEqual(names[73], "acc_dynamic_x.mean")
        self.assertEqual(names[-1], "gyro_z.pearson_kurtosis")
        for channel in (
            "acc_magnitude",
            "angular_rate_magnitude",
            "jerk_magnitude",
        ):
            for statistic in (
                "total_power",
                "normalized_spectral_entropy",
                "dominant_frequency_hz",
                "spectral_centroid_hz",
                "bandpower_0.5_3_hz",
                "bandpower_8_20_hz",
            ):
                self.assertIn(f"{channel}.{statistic}", names)
        forbidden_axis_tokens = (
            ".total_power",
            ".normalized_spectral_entropy",
            ".dominant_frequency_hz",
            ".spectral_centroid_hz",
            ".bandpower_",
        )
        for channel in (
            "acc_dynamic_x",
            "acc_dynamic_y",
            "acc_dynamic_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
        ):
            self.assertFalse(
                any(
                    name.startswith(f"{channel}.")
                    and any(token in name for token in forbidden_axis_tokens)
                    for name in names
                )
            )

    def test_welch_contract_and_known_ppg_sinusoid(self) -> None:
        self.assertEqual(engineering_welch_parameters(4000, 400.0), (1600, 800))
        extraction = direct_extraction()
        names = engineering_feature_names()
        dominant = extraction.sequence.values[
            :, names.index("ppg_red.dominant_frequency_hz")
        ]
        entropy = extraction.sequence.values[
            :, names.index("ppg_red.normalized_spectral_entropy")
        ]
        cardiac_power = extraction.sequence.values[
            :, names.index("ppg_red.bandpower_0.5_3_hz")
        ]
        self.assertTrue(np.allclose(dominant, 1.25, atol=0.26))
        self.assertTrue(np.all((entropy >= 0.0) & (entropy <= 1.0)))
        self.assertTrue(np.all(cardiac_power > 0.0))

    def test_magnitudes_use_canonical_processed_outputs(self) -> None:
        views = views_fixture()
        extraction = extract_engineering_features(views, plan=engineering_plan())
        names = engineering_feature_names()
        first = slice(0, 4000)
        self.assertAlmostEqual(
            extraction.sequence.values[0, names.index("acc_magnitude.mean")],
            float(np.nanmean(views.imu_processed["dynamic_magnitude"][first])),
        )
        self.assertAlmostEqual(
            extraction.sequence.values[
                0, names.index("angular_rate_magnitude.mean")
            ],
            float(np.nanmean(views.imu_processed["gyro_magnitude"][first])),
        )
        replaced_imu = {
            **views.imu_processed,
            "jerk_magnitude": np.full(views.x_filter.shape[0], 3.25),
        }
        replaced_views = replace(views, imu_processed=replaced_imu)
        replaced_extraction = extract_engineering_features(
            replaced_views, plan=engineering_plan()
        )
        self.assertAlmostEqual(
            replaced_extraction.sequence.values[
                0, names.index("jerk_magnitude.mean")
            ],
            3.25,
        )

    def test_nonidentity_ppg_engineering_is_unavailable(self) -> None:
        views = views_fixture()
        result = ArtifactReductionResult(
            x_ar=0.8 * views.x_filter,
            reducer_id="test_nonidentity",
            reducer_version="test_v1",
            is_identity=False,
            status="success",
            confidence=1.0,
            diagnostics={},
            parameters={},
            channel_available=(True, True),
            alignment={"same_time_grid": True, "fs_hz": 400.0},
        )
        routed = views.with_artifact_result(result)
        extraction = extract_engineering_features(routed, plan=engineering_plan())
        ppg_slots = engineering_feature_names().index("acc_magnitude.mean")
        self.assertTrue(np.isnan(extraction.sequence.values[:, :ppg_slots]).all())
        self.assertFalse(extraction.value_validity[:, :ppg_slots].any())
        self.assertTrue(
            np.array_equal(np.isfinite(extraction.sequence.values), extraction.value_validity)
        )
        self.assertTrue(
            extraction.value_validity[
                :, engineering_feature_names().index("acc_dynamic_x.population_sd")
            ].all()
        )

    def test_default_file_aggregation_is_mean_population_sd(self) -> None:
        names = engineering_feature_names()
        matrix = np.zeros((2, len(names)), dtype=np.float64)
        matrix[:, 0] = [1.0, 3.0]
        matrix[:, 1] = [3.0, 7.0]
        sequence = EngineeringFeatureSequence(
            values=matrix,
            start_samples=np.array([0, 2000]),
            valid_row_mask=np.array([True, True]),
            channel_schema=names,
            schema_version=ENGINEERING_SCHEMA_VERSION,
        )
        extraction = EngineeringExtraction(
            sequence,
            np.ones(matrix.shape, dtype=bool),
            SignalRoute.DIRECT,
            (),
        )
        values, validity = summarize_engineering(extraction)
        self.assertEqual(values[f"engineering.{names[0]}.mean"], 2.0)
        self.assertEqual(values[f"engineering.{names[0]}.population_sd"], 1.0)
        self.assertEqual(values[f"engineering.{names[1]}.mean"], 5.0)
        self.assertEqual(values[f"engineering.{names[1]}.population_sd"], 2.0)
        self.assertTrue(all(validity.values()))
        self.assertFalse(any(name.endswith(".median") for name in values))
        self.assertEqual(len(values), 230)
        self.assertEqual(len(validity), 230)

    def test_registry_has_all_ten_required_fields_and_stable_hash(self) -> None:
        registry = default_registry()
        required = {
            "canonical_name", "formula_algorithm", "units", "source_signal_view",
            "endpoint_role_eligibility", "level", "aggregation_rule", "validity_rule",
            "missing_value_policy", "provenance_version",
        }
        self.assertTrue(required.issubset({item.name for item in fields(registry.definitions[0])}))
        for definition in registry.definitions:
            for field_name in required:
                self.assertTrue(str(getattr(definition, field_name)).strip())
        self.assertEqual(registry.sha256, default_registry().sha256)
        self.assertEqual(len(registry.names), 282)
        self.assertEqual(registry.schema_version, "feature_vector_282_v3")
        self.assertNotEqual(registry.schema_version, "feature_vector_v1")
        self.assertIn("optical.red_ir_ac_ratio_median", registry.names)
        self.assertIn("optical.red_ir_dc_ratio_median", registry.names)
        self.assertNotIn("optical.red_ir_cardiac_coherence", registry.names)
        self.assertFalse(any(name.startswith("sqi.") for name in registry.names))
        self.assertNotIn("prv.coverage", registry.names)

    def test_matrix_places_value_validity_channels_in_model_tensor(self) -> None:
        extraction = direct_extraction()
        transform = fit_fold_feature_transform(
            [extraction],
            fitted_on_participant_ids=["train"],
            outer_train_participant_ids=["train"],
            outer_oof_participant_ids=["heldout"],
        )
        transformed = transform_engineering(extraction, transform)
        registry = default_registry()
        raw_context = build_feature_vector(
            {"prv.ppi_mean_s": 0.9},
            feature_validity={"prv.ppi_mean_s": True},
            provenance={"route": SignalRoute.DIRECT.value},
        )
        vector_transform = fit_fold_feature_vector_transform(
            [raw_context],
            ["train"],
            fitted_on_participant_ids=["train"],
            outer_train_participant_ids=["train"],
            outer_oof_participant_ids=["heldout"],
        )
        context = transform_feature_vector(raw_context, vector_transform)
        matrix = build_ordered_matrix(
            transformed,
            context=context,
            provenance={"route": SignalRoute.DIRECT.value},
        )
        self.assertIs(validate_feature_matrix(matrix), matrix)
        engineering_count = len(engineering_feature_names())
        registry_count = len(registry.names)
        self.assertEqual(matrix.values.shape, (2 * (engineering_count + registry_count), 32))
        self.assertEqual(int(np.sum(matrix.row_mask)), 3)
        self.assertEqual(matrix.context_schema[:registry_count], registry.names)
        self.assertEqual(
            matrix.context_schema[registry_count:],
            tuple(f"{name}.validity" for name in registry.names),
        )
        context_index = registry.names.index("prv.ppi_mean_s")
        context_value_row = 2 * engineering_count + context_index
        context_validity_row = (
            2 * engineering_count + registry_count + context_index
        )
        self.assertTrue(np.all(matrix.values[context_value_row, :3] == 0.0))
        self.assertTrue(np.all(matrix.values[context_validity_row, :3] == 1.0))
        self.assertTrue(np.all(matrix.values[:, 3:] == 0.0))
        gyro_index = engineering_feature_names().index("gyro_y.skew_bias_corrected")
        self.assertTrue(np.all(matrix.values[engineering_count + gyro_index, :3] == 0.0))
        self.assertEqual(
            matrix.provenance["validity_encoding"], "paired_explicit_0_1_channels_v1"
        )
        self.assertEqual(
            matrix.schema_version,
            f"ordered_feature_matrix_d794_by_32_registry-{registry.sha256[:12]}_v3",
        )
        compact = build_ordered_matrix(
            transformed,
            context=context,
            provenance={"route": SignalRoute.DIRECT.value},
            k=2,
        )
        self.assertIs(validate_feature_matrix(compact), compact)
        self.assertEqual(compact.values.shape[1], 2)
        self.assertEqual(compact.row_mask.shape, (2,))
        self.assertEqual(compact.provenance["matrix_k"], 2)
        self.assertEqual(
            compact.schema_version,
            f"ordered_feature_matrix_d794_by_2_registry-{registry.sha256[:12]}_v3",
        )
        padded = build_ordered_matrix(
            transformed,
            context=context,
            provenance={"route": SignalRoute.DIRECT.value},
            k=7,
        )
        self.assertIs(validate_feature_matrix(padded), padded)
        self.assertEqual(padded.values.shape[1], 7)
        self.assertEqual(int(np.sum(padded.row_mask)), 3)
        self.assertTrue(np.all(padded.values[:, 3:] == 0.0))
        for invalid_k in (0, -1, True, 4097, 2.5):
            with self.subTest(invalid_matrix_k=invalid_k):
                with self.assertRaises(ValueError):
                    build_ordered_matrix(
                        transformed,
                        context=context,
                        provenance={"route": SignalRoute.DIRECT.value},
                        k=invalid_k,
                    )
        stale_values = np.zeros((1, 94), dtype=np.float64)
        stale = EngineeringExtraction(
            EngineeringFeatureSequence(
                values=stale_values,
                start_samples=np.array([0]),
                valid_row_mask=np.array([True]),
                channel_schema=tuple(f"old_{index}" for index in range(94)),
                schema_version=(
                    "engineering_10s_hop5s_workflow_v1+fold_robust_v1"
                ),
            ),
            np.ones_like(stale_values, dtype=bool),
            SignalRoute.DIRECT,
            (),
        )
        with self.assertRaisesRegex(ValueError, "stale or inconsistent"):
            build_ordered_matrix(
                stale,
                context=context,
                provenance={"route": SignalRoute.DIRECT.value},
            )
        old_matrix = OrderedFeatureMatrixV1(
            values=np.zeros((1, 32), dtype=np.float64),
            row_mask=np.ones(32, dtype=bool),
            channel_schema=("old.feature",),
            context_schema=("old.feature",),
            schema_version="ordered_feature_matrix_v1",
            provenance={},
        )
        with self.assertRaisesRegex(ValueError, "stale or inconsistent"):
            validate_feature_matrix(old_matrix)

    def test_artifact_route_invalidates_morphology_slot_and_unknowns_fail(self) -> None:
        registry = default_registry()
        target = "morphology.amplitude_median"
        vector = build_feature_vector(
            {target: 5.0},
            feature_validity={target: True},
            provenance={"route": SignalRoute.ARTIFACT_RATE_ONLY.value},
        )
        index = registry.names.index(target)
        self.assertFalse(vector.validity[index])
        self.assertTrue(np.isnan(vector.values[index]))
        with self.assertRaises(ValueError):
            build_feature_vector(
                {"participant_id": 99.0},
                feature_validity={"participant_id": True},
                provenance={"route": SignalRoute.DIRECT.value},
            )


if __name__ == "__main__":
    unittest.main()
