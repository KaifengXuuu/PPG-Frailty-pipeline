"""Focused V2 fixed-kernel-samples construction tests / V2 固定核样本数构造测试。"""

from __future__ import annotations

import pytest
import torch
import numpy as np

from ppg_frailty.models.time_scale import (
    ABLATION_ID,
    COMPACT_KERNEL_SAMPLES,
    INCEPTION_KERNEL_SAMPLES,
    build_fixed_kernel_resampling_cases,
    create_fixed_kernel_resampling_model,
    materialize_fixed_kernel_case_config,
    prepare_fixed_kernel_dl_input,
)


def test_registry_is_reference_plus_single_factor_only() -> None:
    cases = build_fixed_kernel_resampling_cases()
    assert len(cases) == 12
    assert len({case.case_id for case in cases}) == 12
    assert {case.model_name for case in cases} == {"CompactCNN1D", "InceptionTimeFull"}
    expected = {
        (400.0, 5.0, 1),
        (400.0, 10.0, 1),
        (100.0, 5.0, 1),
        (160.0, 5.0, 1),
        (200.0, 5.0, 1),
        (400.0, 5.0, 2),
    }
    for model_name in {"CompactCNN1D", "InceptionTimeFull"}:
        selected = [case for case in cases if case.model_name == model_name]
        assert {(case.dl_fs_hz, case.raw_window_seconds, case.dilation) for case in selected} == expected
        assert all(case.comparison_id == ABLATION_ID for case in selected)
        assert all(case.scientific_status == "registered_not_run" for case in selected)


def test_factor_interaction_and_non_target_models_fail_closed() -> None:
    with pytest.raises(ValueError, match="factor interactions"):
        create_fixed_kernel_resampling_model(
            "CompactCNN1D",
            n_channels=2,
            n_classes=3,
            dl_fs_hz=160.0,
            raw_window_seconds=10.0,
            dilation=1,
        )
    with pytest.raises(ValueError, match="CompactCNN1D/InceptionTimeFull"):
        create_fixed_kernel_resampling_model(
            "ShapeFormerChannelSpecificOSD",
            n_channels=2,
            n_classes=3,
            dl_fs_hz=400.0,
        )


def test_compact_and_inception_custom_controls_reach_forward_smoke() -> None:
    compact = create_fixed_kernel_resampling_model(
        "CompactCNN1D",
        n_channels=2,
        n_classes=3,
        dl_fs_hz=160.0,
    )
    inception = create_fixed_kernel_resampling_model(
        "InceptionTimeFull",
        n_channels=2,
        n_classes=3,
        dl_fs_hz=400.0,
        dilation=2,
    )
    compact.eval()
    inception.eval()
    with torch.no_grad():
        assert tuple(compact(torch.randn(1, 2, 64)).shape) == (1, 3)
        assert tuple(inception(torch.randn(1, 2, 64)).shape) == (1, 3)
    assert compact.kernel_sizes == COMPACT_KERNEL_SAMPLES
    assert inception.kernel_sizes == INCEPTION_KERNEL_SAMPLES
    assert inception.dilation == 2
    assert compact.fixed_kernel_resampling_provenance["automatic_execution"] is False


def test_all_12_cases_materialize_and_downsample_only_the_dl_view() -> None:
    cases = build_fixed_kernel_resampling_cases()
    identities = {case.case_id: materialize_fixed_kernel_case_config(case) for case in cases}
    assert len(identities) == 12
    assert all(payload["automatic_execution"] is False for payload in identities.values())
    assert all(payload["canonical_feature_and_peak_fs_hz"] == 400.0 for payload in identities.values())
    assert identities["compactcnn1d__context_10s"]["sequence_length_samples"] == 4000
    assert identities["inceptiontimefull__dilation_2"]["dilation"] == 2

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
    assert output.shape == (1, 2, 500)
    assert output_mask.shape == (1, 500)
    assert int(output_mask.sum()) == 400
    assert provenance["resample_up"] == 1
    assert provenance["resample_down"] == 4
    assert provenance["anti_aliasing"] == "scipy_signal_resample_poly_kaiser_beta5"
    assert provenance["engineering_features_resampled"] is False
    assert np.array_equal(source, original)
